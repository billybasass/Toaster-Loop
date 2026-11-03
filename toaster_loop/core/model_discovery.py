"""Model discovery over a base model folder (Hugging Face hub cache style).

Users point ``models_dir`` (config / env ``TOASTER_LOOP_MODELS_DIR``) at a folder
like ``~/.cache/huggingface/hub`` and the dashboard offers a dropdown instead
of a hand-typed path. Two kinds of artifacts are discovered:

* **GGUF files** — serve-only; found under the HF hub layout
  ``models--<org>--<name>/snapshots/<rev>/*.gguf`` (symlinks resolved) and as
  plain ``*.gguf`` files anywhere in the tree.
* **HF checkpoints** — training base models; a snapshot directory counts when
  it contains ``config.json``. GGUFs cannot be trained on (see worker), so the
  Train wizard lists these instead.

When ``models_dir`` is unset we fall back to ``$HF_HOME/hub`` or
``~/.cache/huggingface/hub`` when either exists, so a machine that already
downloads models with huggingface-cli needs zero configuration.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Safety caps: hub caches can hold huge trees; keep responses bounded.
MAX_RESULTS = 300
_SCAN_SKIP_DIRS = {"refs", "blobs", ".no_exist"}

#: Split (multi-part) GGUF naming: ``Foo-00001-of-00003.gguf`` (HF/llama.cpp
#: convention) and the older ``Foo.split-01.gguf``. Serving a shard other than
#: the first — or offering every shard as its own model — is the classic
#: "llama-server exited early with code 1" trap, so discovery folds shards
#: into ONE entry (pointing at shard 1) and every spawn path resolves any
#: shard back to shard 1 via ``resolve_gguf_shards``.
_SHARD_SUFFIX_RE = re.compile(
    r"[-.](\d{1,5}-of-\d{1,5}|split-\d{1,3})(?=\.[Gg][Gg][Uu][Ff]$)")

#: Companion GGUFs that are never standalone models: vision projectors
#: (``mmproj*.gguf``) and MTP drafters (``mtp*.gguf``). Loading one directly
#: fails with "tensor 'token_embd.weight' not found" — they must only ever be
#: passed via --mmproj / --model-draft next to a real main model.
_COMPANION_NAME_RE = re.compile(
    r"(^|[-._ ])(mmproj|mtp)([-._ ]|$)", re.IGNORECASE)


def is_companion_gguf(name: str) -> bool:
    """True for mmproj projector / mtp drafter GGUFs (not serveable alone)."""
    return _COMPANION_NAME_RE.search(Path(name).stem) is not None


def shard_group(name: str) -> Optional[Tuple[str, int]]:
    """('model-Q4_K_M', 1) for 'model-Q4_K_M-00001-of-00003.gguf'; None when
    the filename is not a split-GGUF shard part. Only unambiguous conventions
    count (``N-of-M`` / ``split-NN``) — plain numeric tails like
    ``model-4k`` stay untouched."""
    m = _SHARD_SUFFIX_RE.search(name)
    if not m:
        return None
    idx_raw = m.group(1)
    if "-of-" in idx_raw:
        idx = int(idx_raw.split("-of-")[0])
    else:  # split-NN
        idx = int(idx_raw.split("-")[1])
    return name[: m.start()], idx


def snapshot_link_for_blob(path: Path) -> Optional[Path]:
    """HF hub caches store weights as hash-named ``blobs/<hash>`` files that
    ``snapshots/<rev>/<visible name>.gguf`` symlinks point at. Shard detection,
    mmproj/MTP sibling lookup and llama-server all need the *visible* snapshot
    filename (blob hashes carry none), so map a blob path back to one of its
    snapshot symlinks. Non-blob paths (and blobs with no link) return None."""
    parts = path.parts
    try:
        i = len(parts) - 1 - list(reversed(parts)).index("blobs")
    except ValueError:
        return None
    if i == 0 or not parts[i - 1].startswith("models--"):
        return None
    snaps = Path(*parts[:i]) / "snapshots"
    target = path.resolve()
    matches: List[Path] = []
    try:
        for snap in snaps.iterdir():
            if not snap.is_dir():
                continue
            for link in snap.rglob("*.gguf"):
                try:
                    if link.resolve() == target:
                        matches.append(link)
                except OSError:
                    continue
    except OSError:
        return None
    if not matches:
        return None
    # Prefer the first shard of a split group (what llama-server wants).
    def _rank(p: Path) -> Tuple[int, int, str]:
        grp = shard_group(p.name)
        return (0 if grp else 1, grp[1] if grp else 0, str(p))
    return sorted(matches, key=_rank)[0]


def resolve_gguf_path(path: str) -> str:
    """Canonical serve path: a hub blob's snapshot symlink when applicable,
    otherwise the input unchanged. Spawn paths and companion (mmproj/MTP)
    lookups should run every user-supplied path through this first."""
    p = Path(path).expanduser()
    link = snapshot_link_for_blob(p)
    return str(link) if link is not None else str(p)


def resolve_gguf_shards(path: str) -> str:
    """If ``path`` is a shard of a split GGUF, return the FIRST shard's path
    (what llama-server -m wants). Hub blob-hash paths are first mapped back to
    their snapshot symlink (blob names hide the shard pattern). Unsharded
    paths and unresolvable cases are returned unchanged — callers keep their
    own existence checks."""
    p = Path(resolve_gguf_path(path)).expanduser()
    grp = shard_group(p.name)
    if grp is None or grp[1] <= 1:
        return str(p)
    base, _idx = grp
    firsts: List[Tuple[int, Path]] = []
    try:
        for f in p.parent.iterdir():
            sg = shard_group(f.name)
            if sg and sg[0] == base and sg[1] == 1:
                firsts.append((0, f))
    except OSError:
        return str(p)
    if not firsts:
        return str(p)
    return str(sorted(firsts)[0][1])


def gguf_total_size(path: str) -> Optional[int]:
    """Summed bytes of every shard of a split GGUF (single file: its size).
    None when nothing under the group can be stat'ed."""
    p = Path(resolve_gguf_path(path)).expanduser()
    grp = shard_group(p.name)
    if grp is None:
        try:
            return p.stat().st_size
        except OSError:
            return None
    base, _idx = grp
    total = 0
    found = False
    try:
        for f in p.parent.iterdir():
            sg = shard_group(f.name)
            if sg and sg[0] == base:
                try:
                    total += f.stat().st_size
                    found = True
                except OSError:
                    pass
    except OSError:
        return None
    return total if found else None


def resolve_models_dir(models_dir: Optional[str] = None) -> Optional[Path]:
    """Resolve the base model folder: explicit arg > TOASTER_LOOP_MODELS_DIR env >
    HF cache defaults ($HF_HOME/hub, then ~/.cache/huggingface/hub).

    An *explicit* folder (arg or env) that doesn't exist returns None — we
    never silently scan a different directory than the one configured. The
    HF-cache fallbacks only apply when nothing explicit is set.
    """
    explicit: List[Path] = []
    if models_dir:
        explicit.append(Path(models_dir).expanduser())
    env = os.environ.get("TOASTER_LOOP_MODELS_DIR", "").strip()
    if env:
        explicit.append(Path(env).expanduser())
    if explicit:
        for c in explicit:
            if c.is_dir():
                return c
        return None
    hf_home = os.environ.get("HF_HOME", "").strip()
    fallbacks: List[Path] = []
    if hf_home:
        fallbacks.append(Path(hf_home).expanduser() / "hub")
    fallbacks.append(Path.home() / ".cache" / "huggingface" / "hub")
    for c in fallbacks:
        if c.is_dir():
            return c
    return None


def _repo_id(entry: Path) -> str:
    """models--Qwen--Qwen3-8B => Qwen/Qwen3-8B"""
    return entry.name[len("models--"):].replace("--", "/")


def _hub_entries(base: Path) -> List[Path]:
    return sorted(
        (e for e in base.iterdir()
         if e.is_dir() and e.name.startswith("models--")),
        key=lambda p: p.name,
    )


def _size(path: Path) -> Optional[int]:
    try:
        return path.stat().st_size
    except OSError:
        return None


def _quant_of(name: str) -> Optional[str]:
    """Best-effort quant tag from a GGUF filename (q4_k_m, Q5_K_S, ...)."""
    import re

    m = re.search(r"[-._]([Qq]\d(_[A-Za-z0-9]+)*)[-._]", name)
    return m.group(1) if m else None


def discover_gguf_models(base: Path, limit: int = MAX_RESULTS) -> List[Dict[str, Any]]:
    """Serve-ready GGUF files under ``base`` (HF hub layout + flat/loose files).

    Split (multi-part) GGUFs are folded into ONE entry per model: the path is
    the first shard (what llama-server wants), size_bytes sums all shards and
    ``shard_count`` records the part count, so the disk pickers never offer
    shard 2-of-3 as a standalone model."""
    out: List[Dict[str, Any]] = []
    seen: set = set()

    def add(path: Path, name: str) -> None:
        try:
            rp = path.resolve()
        except OSError:
            return
        if not rp.is_file():
            return
        # mmproj projectors / mtp drafters are companions, not models: serving
        # one directly fails ("tensor 'token_embd.weight' not found"). They
        # stay out of the picker and are auto-attached at spawn time.
        if is_companion_gguf(path.name):
            return
        # Shard detection works on the *visible* filename (HF hub snapshots
        # symlink to hash-named blobs — resolving first would erase the name).
        grp = shard_group(path.name)
        shards = 1
        if grp is not None:
            sibs: List[Tuple[int, Path]] = []
            try:
                for f in path.parent.iterdir():
                    sg = shard_group(f.name)
                    if sg and sg[0] == grp[0]:
                        sibs.append((sg[1], f))
            except OSError:
                sibs = []
            sibs.sort()
            shards = len(sibs) or 1
            # Represent the group by its lowest present shard; skip any other
            # part. (If shard 1 is missing but 2 exists, still list the model.)
            if sibs and sibs[0][1] != path:
                return
            if sibs:
                path = sibs[0][1]
                try:
                    rp = path.resolve()
                except OSError:
                    return
        key = str(rp)
        if key in seen:
            return
        seen.add(key)
        out.append({
            "name": name,
            # Visible (unresolved) path: HF snapshots symlink to hash-named
            # blobs whose names hide shard/companion structure — keep the
            # snapshot filename so spawn + sibling lookups work.
            "path": str(path),
            "kind": "gguf",
            "size_bytes": (gguf_total_size(str(path)) if grp else None)
                          or _size(rp),
            "quant": _quant_of(path.name),
            "shard_count": shards,
        })

    for entry in _hub_entries(base):
        repo = _repo_id(entry)
        snaps = entry / "snapshots"
        if not snaps.is_dir():
            continue
        for snap in sorted(snaps.iterdir()):
            if not snap.is_dir():
                continue
            rev = snap.name[:8]
            for g in sorted(snap.rglob("*.gguf")):
                # Shard label drops the -000NN-of- suffix so the group's
                # first shard represents all parts under one name.
                stem_only = shard_group(g.name)
                display_stem = stem_only[0] if stem_only else g.stem
                label = f"{repo} · {display_stem}"
                if len(list(snaps.iterdir())) > 1:
                    label += f" ({rev})"
                add(g, label)

    # Loose GGUFs anywhere outside the hub layout (skipping blobs/refs dirs).
    for g in sorted(base.rglob("*.gguf")):
        if any(part in _SCAN_SKIP_DIRS or part.startswith("models--")
               for part in g.parts):
            continue
        grp = shard_group(g.name)
        add(g, grp[0] if grp else g.stem)

    return out[:limit]


def discover_base_models(base: Path, limit: int = MAX_RESULTS) -> List[Dict[str, Any]]:
    """Training-capable HF checkpoints under ``base`` (snapshot dirs with a
    config.json). GGUF repos without an HF snapshot are excluded."""
    out: List[Dict[str, Any]] = []
    multi_cache = len(list(_hub_entries(base))) > 20  # perf guard: skip rglob on big hubs

    for entry in _hub_entries(base):
        repo = _repo_id(entry)
        snaps = entry / "snapshots"
        if not snaps.is_dir():
            continue
        revs = [s for s in sorted(snaps.iterdir()) if s.is_dir()]
        many_revs = len(revs) > 1
        for snap in revs:
            cfg = snap / "config.json"
            if not cfg.is_file():
                continue
            try:
                rp = snap.resolve()
            except OSError:
                continue
            name = f"{repo} ({snap.name[:8]})" if many_revs else repo
            out.append({
                "name": name,
                "path": str(rp),
                "kind": "hf",
                "size_bytes": None,
            })
            if len(out) >= limit:
                return out

    # Flat directories containing config.json (only cheap top-level scans;
    # skip the recursive walk on very large caches).
    if not multi_cache:
        for cfg in sorted(base.rglob("config.json")):
            d = cfg.parent
            if any(part.startswith("models--") or part in _SCAN_SKIP_DIRS
                   for part in d.parts):
                continue
            try:
                rp = d.resolve()
            except OSError:
                continue
            if any(o["path"] == str(rp) for o in out):
                continue
            out.append({"name": d.name, "path": str(rp), "kind": "hf",
                        "size_bytes": None})
            if len(out) >= limit:
                break

    return out


def discover_models(models_dir: Optional[str] = None) -> Dict[str, Any]:
    """Discover everything under the resolved base folder.

    Returns {models_dir, exists, gguf: [...], base_models: [...]}. Never
    raises for a missing/unreadable dir — the UI just shows an empty picker.
    """
    base = resolve_models_dir(models_dir)
    if base is None:
        return {"models_dir": models_dir or None, "exists": False,
                "gguf": [], "base_models": []}
    try:
        gguf = discover_gguf_models(base)
        bases = discover_base_models(base)
    except OSError:
        gguf, bases = [], []
    return {"models_dir": str(base), "exists": True,
            "gguf": gguf, "base_models": bases}

"""Resolve a GGUF's canonical HF training base — pointer, never dequantize.

A GGUF is a serving artifact; trainers need HF safetensors (see D-7 and the
worker's export step). When a user has only a GGUF, the wrong move is a lossy
GGUF→HF conversion (third-party, per-arch, silently corrupt-but-loadable on
any tensor-name mistake — exactly what D-14 exists to prevent). The right move
is to treat the GGUF as a *pointer*: its KV metadata records ``general.
basename`` (llama.cpp's name for the source model), and GGUF repos on the Hub
declare their upstream via the card's ``base_model`` field. Either way, the HF
repo is *resolved*, then downloaded through the normal hub path — training
always runs on real HF weights.

Resolution order (cheapest first, network only when needed):

1. **Local HF checkpoint** whose repo/dir name matches the GGUF basename →
   ready to train, zero bytes moved.
2. **Hub card provenance**: if we know which hub repo the GGUF came from
   (``mmproj.source_repo_for`` decodes the models--org--name layout and the
   download manifests), read its ``base_model`` card field.
3. **Hub search** by basename, keeping only non-GGUF repos (training-capable),
   ranked by downloads.

Nothing here guesses silently: an unresolved GGUF returns candidates + a clear
reason, and the caller (UI) offers manual repo-id entry as the last resort.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .gguf_meta import gguf_arch, gguf_basename


def normalize_model_name(s: str) -> str:
    """Aggressive name-fold for matching: lowercase, alphanumerics only.

    'Qwen3-8B' == 'qwen3_8b' == 'Qwen 3.8 B' — quant tags are stripped first
    so ``Qwen3-8B-Q4_K_M`` folds onto ``qwen38b`` too."""
    s = re.sub(r"(?i)(?:^|[-._ ])(?:u?d-)?(?:iq\d[a-z0-9_]*|q\d+(?:_[a-z0-9]+)*)"
               r"(?=$|[-._ ])", " ", str(s))
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _scan_local_hf(basename: str, roots: List[Path]) -> List[Dict[str, Any]]:
    """HF snapshot dirs under ``roots`` whose repo/dir name matches the
    basename. Hub layout (models--org--name) gives the real repo id; flat dirs
    contribute their own name."""
    want = normalize_model_name(basename)
    if not want:
        return []
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for base in roots:
        if not base.is_dir():
            continue
        # Hub layout first: cheap top-level scan of models--* dirs.
        try:
            entries = sorted(base.iterdir())
        except OSError:
            continue
        for entry in entries:
            if not entry.is_dir() or not entry.name.startswith("models--"):
                continue
            repo = entry.name[len("models--"):].replace("--", "/")
            name_part = repo.split("/")[-1]
            if normalize_model_name(name_part) != want:
                continue
            snaps = entry / "snapshots"
            if not snaps.is_dir():
                continue
            for snap in sorted(snaps.iterdir(), reverse=True):
                if (snap / "config.json").is_file() and str(snap) not in seen:
                    seen.add(str(snap))
                    out.append({"path": str(snap), "repo_id": repo,
                                "source": "local"})
        # Flat checkpoint dirs one level down (a common manual layout).
        for entry in entries:
            if not entry.is_dir() or entry.name.startswith("models--"):
                continue
            if normalize_model_name(entry.name) != want:
                continue
            if (entry / "config.json").is_file() and str(entry) not in seen:
                seen.add(str(entry))
                out.append({"path": str(entry), "repo_id": entry.name,
                            "source": "local"})
    return out


def _candidate(repo_id: str, *, downloads: int = 0, source: str,
               gguf: bool = False) -> Dict[str, Any]:
    return {"repo_id": repo_id, "downloads": int(downloads or 0),
            "source": source, "gguf": bool(gguf)}


def _card_base_models(repo_id: str) -> List[Dict[str, Any]]:
    """base_model entries declared on a hub repo's card ([{repo_id,...}])."""
    from .modelhub import HubError, model_detail

    try:
        d = model_detail(repo_id, kind="models")
    except HubError:
        return []
    out: List[Dict[str, Any]] = []
    for bm in d.get("base_models") or []:
        rid = str(bm).strip()
        if "/" in rid and not rid.upper().endswith("-GGUF"):
            out.append(_candidate(rid, source="card"))
    return out


def _search_candidates(basename: str, arch: Optional[str],
                       limit: int = 8) -> List[Dict[str, Any]]:
    """Hub search for non-GGUF repos matching the basename. GGUF repos are
    dropped (they can't train); when the GGUF's arch is known, GGUF-only
    architectures can't be verified server-side so names + downloads rank it."""
    from .modelhub import HubError, search_models

    try:
        rows = search_models(basename, limit=max(limit * 3, 24),
                             gguf_only=False, kind="models")
    except HubError:
        return []
    want = normalize_model_name(basename)
    out: List[Dict[str, Any]] = []
    for r in rows:
        rid = str(r.get("repo_id") or "")
        if not rid or r.get("gguf"):  # GGUF repos are not training bases
            continue
        name_part = rid.split("/")[-1]
        norm = normalize_model_name(name_part)
        # Accept exact folds and common suffixed variants (Instruct/GGUF-free):
        # qwen38b matches qwen38b, qwen38binstruct; never qwen314b.
        if not (norm == want or norm.startswith(want) or want in norm):
            continue
        out.append(_candidate(rid, downloads=r.get("downloads", 0),
                              source="search"))
    out.sort(key=lambda c: -c["downloads"])
    return out[:limit]


def resolve_gguf_base(gguf_path: str, *, home: Optional[Path] = None,
                      models_dir: Optional[str] = None) -> Dict[str, Any]:
    """Best-effort resolution of the HF base a GGUF was made from.

    Returns::

        {gguf, basename, arch, source_repo,
         local:   [{path, repo_id, source:"local"}],       # trainable now
         remote:  [{repo_id, downloads, source}...],        # downloadable
         reason:  str | None}                               # when nothing found

    Never raises for a plain GGUF with no provenance — ``reason`` explains and
    the UI falls back to manual repo-id entry. Network failures degrade to an
    empty ``remote`` list, not an error."""
    from .model_discovery import resolve_gguf_path
    from .modelhub import validate_repo_id, HubError

    path = Path(resolve_gguf_path(gguf_path)).expanduser()
    result: Dict[str, Any] = {"gguf": str(path), "basename": None,
                              "arch": None, "source_repo": None,
                              "local": [], "remote": [], "reason": None}

    if not path.is_file():
        result["reason"] = f"GGUF not found: {path}"
        return result

    basename = gguf_basename(str(path))
    arch = gguf_arch(str(path))
    result["basename"] = basename
    result["arch"] = arch

    # Provenance of the GGUF itself (hub layout dir name or download manifests).
    from .mmproj import source_repo_for

    source_repo = source_repo_for(str(path))
    if source_repo:
        try:
            result["source_repo"] = validate_repo_id(source_repo)
        except HubError:
            pass

    # Where HF checkpoints live: explicit dir + the home/models fallback the
    # dashboard catalog already merges (see api/models.h_model_catalog).
    from .model_discovery import resolve_models_dir

    roots: List[Path] = []
    md = resolve_models_dir(models_dir)
    if md is not None:
        roots.append(md)
    if home is not None:
        extra = Path(home) / "models"
        if extra.is_dir() and extra not in roots:
            roots.append(extra)

    # 1. Already on disk?
    search_names = [n for n in (basename,) if n]
    # Without a basename, the source repo's own name is the best local key.
    if not search_names and result["source_repo"]:
        search_names = [result["source_repo"].split("/")[-1]]
    for nm in search_names:
        result["local"].extend(_scan_local_hf(nm, roots))
    if result["local"]:
        return result

    # 2. The GGUF's own hub card declares its upstream base_model(s).
    if result["source_repo"]:
        result["remote"].extend(_card_base_models(result["source_repo"]))

    # 3. Hub search by basename (non-GGUF repos only).
    if basename:
        have = {c["repo_id"] for c in result["remote"]}
        result["remote"].extend(
            c for c in _search_candidates(basename, arch)
            if c["repo_id"] not in have)

    if not result["remote"]:
        if not basename:
            result["reason"] = (
                "this GGUF carries no general.basename metadata and no hub "
                "provenance — pick the HF base repo manually")
        else:
            result["reason"] = (
                f"no training-capable (non-GGUF) repo found for "
                f"{basename!r} — pick the HF base repo manually")
    return result

"""Vision projector (mmproj) detection + download, reference style.

The user never types an mmproj path. toaster_loop finds the
multimodal projector that belongs to a GGUF and offers a load toggle:

* **Detected** — an ``mmproj*.gguf`` sitting next to the model GGUF (studio's
  Model Hub downloads ride the mmproj companion along with every quant
  variant; see core/modelhub.py ``list_variants``). The UI shows a
  "load vision projector" toggle for it.
* **Downloadable** — no local file, but the model came from a hub repo we can
  identify (the ``toaster_loop_download.json`` manifest written by modelhub, or an
  ``origin.json`` written by huggingface_hub) and that repo publishes an
  mmproj companion. The UI offers to download it; it lands *in the same
  folder as the GGUF* (snapshot dir), exactly like studio.
* **None** — plain text model: no toggle, nothing to load.

``resolve_mmproj`` collapses this into what a spawn needs: an explicit path
(advanced/manual override) or the auto-detected sibling when vision is
enabled and no explicit disable was set.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

# mmproj companion filenames: mmproj.gguf, mmproj-model.gguf,
# mmproj-F16.gguf, Qwen3-VL-mmproj-BF16.gguf ... (any *.gguf with an
# "mmproj" token in the stem).
_MMPROJ_NAME_RE = re.compile(r"(^|[-._ ])mmproj([-._ ]|$)", re.IGNORECASE)

#: MTP drafter companion filenames: mtp.gguf, mtp-Qwen3.8-Flash-Next-Q8_0.gguf,
#: Foo-mtp.gguf ... (any *.gguf with an "mtp" token in the stem). Hub GGUF repos
#: repos ship them in an ``MTP/`` folder next to the quant directories.
_MTP_NAME_RE = re.compile(r"(^|[-._ ])mtp([-._ ]|$)", re.IGNORECASE)


def is_mmproj_name(name: str) -> bool:
    """True when a filename looks like a multimodal projector GGUF."""
    p = Path(name)
    return p.suffix.lower() == ".gguf" and bool(_MMPROJ_NAME_RE.search(p.stem))


def is_mtp_name(name: str) -> bool:
    """True when a filename looks like an MTP drafter GGUF."""
    p = Path(name)
    return p.suffix.lower() == ".gguf" and bool(_MTP_NAME_RE.search(p.stem))


def _search_roots(gguf: Path) -> List[Path]:
    """Folders to scan for companions, nearest first: the model's own folder,
    ``MTP``/mmproj-named subfolders of it, the enclosing HF snapshot root and
    its ``MTP``-named subdirectories (hub repos nest quants like
    ``snapshots/<rev>/UD-IQ4_XS/model.gguf`` with ``snapshots/<rev>/MTP/``)."""
    roots: List[Path] = []

    def _add(p: Path) -> None:
        if p.is_dir() and p not in roots:
            roots.append(p)

    _add(gguf.parent)
    try:
        for d in sorted(gguf.parent.iterdir()):
            if d.is_dir() and re.search(r"mtp|mmproj", d.name, re.IGNORECASE):
                _add(d)
    except OSError:
        pass
    # Walk up to the snapshot root (…/snapshots/<rev>) when inside one.
    up = gguf.parent
    while up.parts and up.name != "snapshots" and len(up.parts) > 1:
        up = up.parent
    if up.parts and up.name == "snapshots":
        snap_root = None
        try:
            revs = sorted(p for p in up.iterdir() if p.is_dir())
        except OSError:
            revs = []
        # The snapshot containing the model is <rev> = gguf's ancestor.
        for rev in revs:
            try:
                if gguf.is_relative_to(rev):
                    snap_root = rev
                    break
            except (OSError, ValueError):
                continue
        if snap_root is not None and snap_root != gguf.parent:
            _add(snap_root)
            try:
                for d in sorted(snap_root.iterdir()):
                    if d.is_dir() and re.search(r"mtp|mmproj", d.name,
                                                re.IGNORECASE):
                        _add(d)
            except OSError:
                pass
    return [r for r in roots if r.is_dir()]


def _visible(p: Path) -> str:
    """Absolute path that keeps the *visible* .gguf filename: resolving an HF
    snapshot symlink lands on the hash-named blob (fine to read, but sibling
    detection elsewhere keys off names). Non-symlinks resolve normally."""
    try:
        rp = p.resolve()
    except OSError:
        return str(p)
    if "blobs" in rp.parts and p.suffix.lower() == ".gguf":
        try:
            return str(p.absolute())
        except OSError:
            return str(p)
    return str(rp)


def _find_companion(gguf_path: str,
                    name_test) -> Optional[str]:
    try:
        gguf = Path(gguf_path).expanduser()
    except (OSError, ValueError):
        return None
    # Hub blob-hash paths carry no companion info — map back to the visible
    # snapshot symlink so sibling folders can be searched by name.
    from .model_discovery import resolve_gguf_path

    gguf = Path(resolve_gguf_path(str(gguf)))
    for folder in _search_roots(gguf):
        try:
            cands = sorted(p for p in folder.glob("*.gguf")
                           if p.is_file() and name_test(p.name))
        except OSError:
            continue
        if cands:
            return _visible(cands[0])
    return None


def find_local(gguf_path: str) -> Optional[str]:
    """mmproj GGUF sitting next to ``gguf_path`` (same folder, or the enclosing
    snapshot root / mmproj subfolder for HF-hub layouts), if any. Deterministic
    order (nearest folder first, then sorted name)."""
    return _find_companion(gguf_path, is_mmproj_name)


def find_local_mtp(gguf_path: str) -> Optional[str]:
    """MTP drafter GGUF belonging to ``gguf_path`` (same folder, an adjacent
    ``MTP/`` directory, or the snapshot root). When several exist, prefer a
    non-"shared" one (quant-specific), then by name — deterministic."""
    from .model_discovery import resolve_gguf_path

    gguf = Path(resolve_gguf_path(str(gguf_path)))
    for folder in _search_roots(gguf):
        try:
            cands = sorted(p for p in folder.glob("*.gguf")
                           if p.is_file() and is_mtp_name(p.name))
        except OSError:
            continue
        if cands:
            preferred = [p for p in cands
                         if "shared" not in p.name.lower()]
            return _visible((preferred or cands)[0])
    return None


def _repo_from_manifest(snapshot: Path) -> Optional[str]:
    """Repo id recorded by toaster_loop's modelhub manifest, if present."""
    f = snapshot / "toaster_loop_download.json"
    if f.is_file():
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            rid = str(data.get("repo_id") or "").strip()
            return rid or None
        except (OSError, json.JSONDecodeError):
            return None
    return None


def _repo_from_origin(snapshot: Path) -> Optional[str]:
    """Repo id recorded by huggingface_hub's origin.json (downloaded models)."""
    f = snapshot / "origin.json"
    if f.is_file():
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            rid = str(data.get("repo_id") or "").strip()
            return rid or None
        except (OSError, json.JSONDecodeError):
            return None
    return None


def source_repo_for(gguf_path: str) -> Optional[str]:
    """Best-effort hub repo id a local GGUF came from.

    Walks up from the file looking for the HF-hub layout
    ``models--<org>--<name>/snapshots/<rev>/`` (repo id decoded from the dir
    name), then per-snapshot manifests (toaster_loop's own, then huggingface_hub's).
    Loose files with no provenance return None — we never guess a repo."""
    try:
        p = Path(gguf_path).expanduser().resolve()
    except OSError:
        return None
    parts = p.parts
    for i, part in enumerate(parts):
        if part.startswith("models--") and i + 1 < len(parts) \
                and parts[i + 1] == "snapshots":
            return part[len("models--"):].replace("--", "/")
    # No hub layout: check manifests near the file (snapshot dir = parents).
    for snap in [p.parent, *p.parents[:3]]:
        rid = _repo_from_manifest(snap) or _repo_from_origin(snap)
        if rid:
            return rid
    return None


def remote_mmproj_candidates(repo_id: str) -> List[Dict[str, Any]]:
    """mmproj files a hub repo publishes ([{rfilename,size}]). Raises HubError
    on network/auth problems (caller surfaces the message)."""
    from .modelhub import repo_files, validate_repo_id

    validate_repo_id(repo_id)
    _, sibs = repo_files(repo_id)
    return [s for s in sibs if is_mmproj_name(s["rfilename"])]


def download_for(gguf_path: str, repo_id: Optional[str] = None,
                 rfilename: Optional[str] = None) -> Dict[str, Any]:
    """Download the mmproj companion into the GGUF's own folder (catalog-
    studio behavior). Returns {path, repo_id, rfilename, size_bytes}.

    ``repo_id`` overrides provenance detection; ``rfilename`` picks a specific
    candidate when the repo ships several (default: first sorted)."""
    gguf = Path(gguf_path).expanduser().resolve()
    if not gguf.is_file():
        raise FileNotFoundError(f"GGUF not found: {gguf}")
    repo_id = repo_id or source_repo_for(str(gguf))
    if not repo_id:
        raise ValueError("cannot determine which hub repo this model came "
                         "from — no mmproj to download")
    from .modelhub import repo_files

    sha, _ = repo_files(repo_id)
    cands = remote_mmproj_candidates(repo_id)
    if not cands:
        raise ValueError(f"{repo_id} publishes no mmproj projector")
    pick = None
    if rfilename:
        pick = next((c for c in cands if c["rfilename"] == rfilename), None)
        if pick is None:
            raise ValueError(f"{repo_id} has no mmproj named {rfilename!r}")
    else:
        pick = cands[0]
    dest = (gguf.parent / Path(pick["rfilename"]).name).resolve()
    # Containment: the repo-relative name must not escape the GGUF's folder.
    if dest.parent != gguf.parent:
        raise ValueError(f"unsafe mmproj filename: {pick['rfilename']!r}")
    _fetch(repo_id, sha, pick["rfilename"], dest)
    return {"path": str(dest), "repo_id": repo_id,
            "rfilename": pick["rfilename"],
            "size_bytes": dest.stat().st_size}


def _fetch(repo_id: str, sha: str, rfilename: str, dest: Path) -> None:
    """Stream one hub file to ``dest`` (resumable via a .part sibling), reusing
    modelhub's endpoint/token conventions."""
    import httpx

    from .modelhub import REQUEST_TIMEOUT, HubError, _headers, endpoint

    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = _headers()
    resume_from = part.stat().st_size if part.exists() else 0
    if resume_from:
        headers["Range"] = f"bytes={resume_from}-"
    url = (f"{endpoint()}/{repo_id}/resolve/main/{rfilename}"
           .replace("resolve/main", f"resolve/{sha}", 1))
    mode = "ab" if resume_from else "wb"
    try:
        with httpx.stream("GET", url, headers=headers, timeout=REQUEST_TIMEOUT,
                          follow_redirects=True) as r:
            if r.status_code == 416 and part.exists():
                return  # already complete
            if r.status_code in (401, 403):
                raise HubError(f"{rfilename}: authorization required — "
                               "set HF_TOKEN")
            if r.status_code not in (200, 206):
                raise HubError(f"{rfilename}: download failed "
                               f"(HTTP {r.status_code})")
            if resume_from and r.status_code == 200:
                mode, resume_from = "wb", 0
            with open(part, mode) as fh:
                for chunk in r.iter_raw():
                    fh.write(chunk)
    except httpx.HTTPError as e:
        raise HubError(f"{rfilename}: connection dropped — retry resumes it") from e
    part.replace(dest)


def resolve_mmproj(gguf_path: Optional[str],
                   run_settings: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """mmproj path to pass to llama-server for this spawn, or None.

    Precedence (run_settings keys; see serving/run_args.py):
      * ``vision: false``               — never load a projector.
      * explicit ``mmproj_path``        — manual override (advanced); when the
        file is gone, treated as absent rather than crashing the spawn.
      * auto (default / ``vision: true``) — sibling mmproj next to the GGUF."""
    rs = run_settings or {}
    if not gguf_path:
        return None
    vision_off = rs.get("vision") is False
    if vision_off:
        return None
    explicit = str(rs.get("mmproj_path") or "").strip()
    if explicit:
        p = Path(explicit).expanduser()
        return str(p) if p.is_file() else None
    if rs.get("vision") is True or "vision" not in rs:
        return find_local(gguf_path)
    return None

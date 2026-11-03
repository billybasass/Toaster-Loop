"""Artifact provenance attestation (inspiration pass 3, item #23).

Dataset manifests carry *trace* provenance, but imported datasets and base
models recorded only a name/path — lineage could say "trained on X" without
proving which bytes X was. Pattern taken from AGPL studio's
``core/training/provenance.py`` (concept only): a versioned
``resource_provenance`` block per resource with an attested/incomplete status,
a weight-file inventory, and the model-load mode actually used.

Design points:

  * **Hash what we can, attest honestly when we can't.** A local file or
    directory gets sha256 inventories; a hub repo id we never downloaded is
    ``incomplete`` with the reason recorded — a lie of omission would poison
    lineage audits later.
  * **Bounded hashing.** Weight files are huge; hashing streams in 4 MiB
    chunks, and an upper bound (``MAX_HASH_BYTES``) refuses to stall training
    on pathological inventories (oversized entries record size only).
  * **Cheap now, painful later.** The block rides inside existing JSON
    documents (dataset manifest) or a new nullable TEXT column (registry), so
    old readers never see changed types and old DBs upgrade in place.

The worker communicates its load mode through the job log
(``resource_provenance`` event, same channel as ``train_done`` losses); the
pipeline parses it with ``parse_worker_provenance`` and persists it — no new
IPC.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

#: Bump when the block SHAPE changes; readers may rely on it.
PROVENANCE_VERSION = 1

#: Files above this get size-only entries (streaming 50 GB weights during a
#: train submit is not "cheap"; sizes still catch most accidental swaps).
MAX_HASH_BYTES = 8 * 1024 * 1024 * 1024  # 8 GiB

_CHUNK = 4 * 1024 * 1024

PathLike = Union[str, Path]


def hash_file(path: PathLike) -> Optional[str]:
    """sha256 of a file's bytes (streamed), or None when too large/unreadable."""
    p = Path(str(path))
    try:
        size = p.stat().st_size
    except OSError:
        return None
    if size > MAX_HASH_BYTES:
        return None
    h = hashlib.sha256()
    try:
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(_CHUNK), b""):
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()


def file_entry(relpath: str, path: PathLike) -> Dict[str, Any]:
    """One inventory entry: {relpath, size, sha256?} — sha256 omitted when the
    file was too large or unreadable (size still attests something)."""
    p = Path(str(path))
    entry: Dict[str, Any] = {"relpath": str(relpath)}
    try:
        entry["size"] = p.stat().st_size
    except OSError:
        entry["size"] = 0
    digest = hash_file(p)
    if digest:
        entry["sha256"] = digest
    return entry


def attest_files(root: PathLike, *,
                 patterns: Optional[Iterable[str]] = None,
                 max_entries: int = 200) -> List[Dict[str, Any]]:
    """Inventory files under ``root`` (recursive, sorted, capped).

    ``patterns`` are glob patterns relative to root (default: everything).
    Returns [] for a missing root — callers decide whether that is incomplete
    or acceptable. The cap keeps manifests bounded; sorted order makes the
    inventory deterministic across runs.
    """
    root_p = Path(str(root))
    if not root_p.is_dir():
        return []
    files: List[Path] = []
    pats = list(patterns) if patterns else ["**/*"]
    seen = set()
    for pat in pats:
        for f in root_p.glob(pat):
            if f.is_file() and str(f) not in seen:
                seen.add(str(f))
                files.append(f)
    files.sort(key=lambda f: str(f.relative_to(root_p)))
    return [file_entry(str(f.relative_to(root_p)), f)
            for f in files[:max_entries]]


def attest_path(path: PathLike, *, patterns: Optional[Iterable[str]] = None,
                load_mode: Optional[str] = None,
                source: Optional[str] = None) -> Dict[str, Any]:
    """Build a versioned resource_provenance block for one resource.

    Works for a single file (imported dataset JSONL) or a directory (base
    model dir). ``status`` is ``attested`` when at least one entry carries a
    hash and the path existed, else ``incomplete`` with a machine-readable
    ``reason`` — downstream audits can filter on status without guessing.
    """
    p = Path(str(path))
    block: Dict[str, Any] = {"version": PROVENANCE_VERSION}
    if source:
        block["source"] = str(source)
    if load_mode:
        block["load_mode"] = load_mode
    if p.is_file():
        files = [file_entry(p.name, p)]
    elif p.is_dir():
        files = attest_files(p, patterns=patterns)
    else:
        files = []
    block["files"] = files
    if files and any("sha256" in e for e in files):
        block["status"] = "attested"
    else:
        block["status"] = "incomplete"
        block["reason"] = ("path missing or unreadable" if not p.exists()
                           else "no file could be hashed (too large?)")
    return block


def attest_hub_repo(repo_id: str, *, filename: Optional[str] = None,
                    etag: Optional[str] = None) -> Dict[str, Any]:
    """Provenance for a resource identified only by hub repo id.

    We never hash what we did not download — this records the identity plus
    whatever transport metadata (etag) was observed, honestly marked
    incomplete. modelhub fills the hashes when it materializes the file.
    """
    block: Dict[str, Any] = {
        "version": PROVENANCE_VERSION,
        "source": f"hf:{repo_id}",
        "status": "incomplete",
        "reason": "hub repo not materialized locally at attestation time",
        "files": [],
    }
    if filename:
        block["files"].append({"relpath": filename,
                               **({"etag": etag} if etag else {})})
    return block


# Model-load modes the worker can truthfully report (studio's vocabulary).
LOAD_MODE_RUNTIME_4BIT = "runtime-4bit"        # bnb quantization_config applied
LOAD_MODE_UNQUANTIZED = "unquantized"          # native bf16/fp16 load
LOAD_MODE_PREQUANTIZED = "prequantized"        # GGUF/AWQ/GPTQ bytes on disk


def worker_provenance_event(load_mode: str, base_model: str,
                            *, extra: Optional[Dict[str, Any]] = None
                            ) -> Dict[str, Any]:
    """Shape of the job-log event the worker emits after loading the model."""
    ev: Dict[str, Any] = {
        "type": "resource_provenance",
        "resource": "base_model",
        "source": str(base_model),
        "load_mode": load_mode,
    }
    if extra:
        ev.update(extra)
    return ev


def parse_worker_provenance(log_path: Optional[str]) -> Optional[Dict[str, Any]]:
    """Extract the last ``resource_provenance`` event from a worker job log.

    Same JSONL channel and tolerance as pipeline.parse_worker_losses: older
    workers emit nothing -> None (registry column stays NULL; absence is
    "unattested", not an error).
    """
    if not log_path:
        return None
    p = Path(str(log_path))
    if not p.is_file():
        return None
    found: Optional[Dict[str, Any]] = None
    try:
        for line in p.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line or "resource_provenance" not in line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(ev, dict) and ev.get("type") == "resource_provenance":
                found = ev
    except OSError:
        return None
    return found

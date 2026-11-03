"""Disk janitor — recoverable archival of aged terminal models (backlog #16).

Curator invariants (built in):

  * **Never delete — only archive.** Archival moves artifacts under
    ``<home>/archive/<model_id>/`` and is exactly restorable via
    :meth:`ModelRegistry.restore_model`.
  * **Pinned bypasses automation.** Pinned models are never touched.
  * **State file, not a scheduler.** Maintenance runs at an existing
    entrypoint (``toaster_loop train auto-check`` — already the cron hook) and
    persists ``{last_run_at, run_count, last_summary}`` in
    ``<home>/janitor_state.json`` instead of owning a timer.

Safety invariants:
  * Only statuses in ``ARCHIVABLE_STATUSES`` (superseded/failed/rolled_back)
    are eligible — never candidate/validated/promoted, never active.
  * Only models older than ``days`` are eligible. The age reference is the
    semantically right timestamp per row: ``promoted_at`` when present (the
    model was live until something replaced/rolled it — that is when it
    stopped being useful), else ``created_at`` (a failed validation means it
    never went live; its birth date is the only honest reference).
  * Traces, datasets and adapters of active/validated/candidate models are
    never touched: archival only ever moves paths recorded on the eligible
    row itself.

Config (YAML ``janitor:`` section, see config.example.yaml):
  enabled: false (default — conservative; opt in via config or run manually
    with ``toaster_loop models janitor``)
  retention_days: 30
  disk_retention_days: 90   — aged *operational* files (worker logs, server
    logs, chat uploads, batch checkpoints) are deleted after this many days.
    Unlike model artifacts these have no restore path, so the default is
    conservative and ``--dry-run`` lists exactly what would be removed.
    0 disables disk cleanup entirely.

Disk-cleanup safety: only files inside an explicit allow-list of operational
directories (``train_runs/*.log``, ``logs/*``, ``uploads/*``, ``batches/*.json``,
``cache/spillover/*``) older than the cutoff are ever removed — never model,
dataset, buffer or archive contents.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .registry import ARCHIVABLE_STATUSES, ModelRegistry, ModelRow, RegistryError

logger = logging.getLogger("toaster_loop.janitor")

STATE_FILENAME = "janitor_state.json"
DEFAULT_RETENTION_DAYS = 30


#: Operational file patterns cleaned by the disk pass (relative to home).
#: Deliberately excludes models/, datasets/, buffer/, archive/ — those are
#: user data with restore paths or provenance value and age out via the
#: model-archive rules above, never via blanket deletion.
DISK_CLEAN_TARGETS: Tuple[Tuple[str, str], ...] = (
    ("train_runs", "*.log"),   # finished worker logs
    ("logs", "*"),             # dashboard/server logs
    ("uploads", "*"),          # chat image attachments
    ("batches", "*.json"),     # batch checkpoints (traces are the truth)
    ("cache/spillover", "*"),  # spilled tool outputs behind stubs
)
DEFAULT_DISK_RETENTION_DAYS = 90


@dataclass
class JanitorConfig:
    """Resolved ``janitor:`` YAML section."""

    enabled: bool = False  # conservative default; documented in config.example.yaml
    retention_days: int = DEFAULT_RETENTION_DAYS
    disk_retention_days: int = DEFAULT_DISK_RETENTION_DAYS  # 0 disables


def janitor_config_from_yaml(section: Optional[Dict[str, Any]]) -> JanitorConfig:
    """Build a JanitorConfig from the raw ``janitor:`` YAML dict (tolerant)."""
    data = section or {}
    jc = JanitorConfig()
    if "enabled" in data:
        jc.enabled = bool(data["enabled"])
    if data.get("retention_days") is not None:
        try:
            jc.retention_days = int(data["retention_days"])
        except (TypeError, ValueError):
            logger.warning("janitor.retention_days not an int; keeping default %d",
                           DEFAULT_RETENTION_DAYS)
    if data.get("disk_retention_days") is not None:
        try:
            jc.disk_retention_days = int(data["disk_retention_days"])
        except (TypeError, ValueError):
            logger.warning("janitor.disk_retention_days not an int; keeping "
                           "default %d", DEFAULT_DISK_RETENTION_DAYS)
    return jc


# ------------------------------------------------------------------ state file

def state_path(home: Path) -> Path:
    return Path(home) / STATE_FILENAME


def load_state(home: Path) -> Dict[str, Any]:
    """Read janitor state; corrupt/missing reads fall back to defaults."""
    p = state_path(home)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {
                "last_run_at": data.get("last_run_at"),
                "run_count": int(data.get("run_count", 0)),
                "last_summary": data.get("last_summary"),
            }
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        pass
    return {"last_run_at": None, "run_count": 0, "last_summary": None}


def save_state(home: Path, state: Dict[str, Any]) -> None:
    """Atomic write (tmp + os.replace), same pattern as serve_state.json."""
    p = state_path(home)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp, p)


# ------------------------------------------------------------------- janitor

def _parse_ts(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def age_reference(m: ModelRow) -> Optional[datetime]:
    """The semantically right timestamp for 'how old is this terminal model'.

    promoted_at when the model ever went live (its usefulness ended at/after
    promotion, not at registration); created_at otherwise (failed candidates
    never served — birth date is the only reference). None when neither
    parses -> the row is treated as *not old* and left alone.
    """
    return _parse_ts(m.promoted_at) or _parse_ts(m.created_at)


def _path_bytes(p: Optional[str]) -> int:
    if not p:
        return 0
    path = Path(p)
    try:
        if path.is_file():
            return path.stat().st_size
        if path.is_dir():
            return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    except OSError:
        pass
    return 0


def _candidate_bytes(m: ModelRow) -> int:
    """Artifact footprint; adapter-dir contents counted once even when the
    GGUF is nested inside it (mirrors archive_model's move dedup)."""
    adapter, gguf = m.adapter_dir, m.gguf_path
    if adapter and gguf:
        try:
            if Path(gguf).is_relative_to(adapter):
                return _path_bytes(adapter)
        except (OSError, ValueError):
            pass
    return _path_bytes(adapter) + _path_bytes(gguf)


def find_candidates(
    registry: ModelRegistry, days: int, now: Optional[datetime] = None
) -> List[Tuple[ModelRow, int]]:
    """Terminal, unpinned, unused, aged models eligible for archival (oldest first).

    Pass 4 #29 (idea adapted from PIASystem (MIT) app/core/adapter_manager.py):
    age alone cannot distinguish the quietly-excellent served adapter from the
    dead one, so a model whose ``last_used_at`` falls *inside* the aging window
    still has a usage signal and is excluded. ``last_used_at`` None/unparseable
    means "no usage signal" -> the age rules decide (fallback policy).
    """
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)
    out: List[Tuple[ModelRow, int]] = []
    for m in registry.list_models():
        if m.status not in ARCHIVABLE_STATUSES:
            continue  # never candidate/validated/promoted/archived/active
        if m.pinned:
            continue  # invariant: pinned bypasses automation
        ref = age_reference(m)
        if ref is None or ref > cutoff:
            continue  # unparseable timestamps are conservatively "not old"
        last_used = _parse_ts(m.last_used_at)
        if last_used is not None and last_used > cutoff:
            continue  # used more recently than the window -> leave it alone
        out.append((m, _candidate_bytes(m)))
    out.sort(key=lambda t: age_reference(t[0]) or datetime.min.replace(tzinfo=timezone.utc))
    return out


def find_disk_files(
    home: Path, days: int, now: Optional[datetime] = None
) -> List[Tuple[Path, int]]:
    """Aged operational files (mtime older than cutoff) inside the allow-listed
    targets. Returns [(path, size_bytes)] sorted oldest-first. Never follows
    symlinks and never leaves ``home``."""
    now = now or datetime.now(timezone.utc)
    cutoff_ts = (now - timedelta(days=days)).timestamp()
    aged: List[Tuple[float, Path, int]] = []
    home = Path(home)
    for rel_dir, pattern in DISK_CLEAN_TARGETS:
        d = home / rel_dir
        if not d.is_dir():
            continue
        try:
            entries = [p for p in d.glob(pattern) if p.is_file() and not p.is_symlink()]
        except OSError:
            continue
        for f in entries:
            try:
                st = f.stat()
            except OSError:
                continue
            if st.st_mtime <= cutoff_ts:
                aged.append((st.st_mtime, f, st.st_size))
    aged.sort(key=lambda t: t[0])
    out = [(f, size) for _, f, size in aged]
    return out


def run_disk_cleanup(
    home: Path, days: int, dry_run: bool = False
) -> Dict[str, Any]:
    """Delete aged operational files (logs/uploads/spillover/checkpoints).

    Returns {removed: [relpaths], freed_bytes, dry_run, errors}. ``days`` <= 0
    disables the pass. Deletion is final for these artifacts (no restore path
    exists), which is why only allow-listed directories are ever touched."""
    summary: Dict[str, Any] = {"removed": [], "freed_bytes": 0,
                               "dry_run": bool(dry_run), "errors": []}
    if days <= 0:
        return summary
    for f, size in find_disk_files(Path(home), days):
        rel = str(f.relative_to(home))
        if dry_run:
            summary["removed"].append(rel)
            summary["freed_bytes"] += size
            continue
        try:
            f.unlink()
            summary["removed"].append(rel)
            summary["freed_bytes"] += size
        except OSError as e:
            logger.warning("janitor: could not remove %s: %s", f, e)
            summary["errors"].append(f"{rel}: {e}")
    return summary


def run_janitor(
    home: Path,
    days: int = DEFAULT_RETENTION_DAYS,
    dry_run: bool = False,
    registry: Optional[ModelRegistry] = None,
    actor: str = "cli",
    disk_days: int = DEFAULT_DISK_RETENTION_DAYS,
) -> Dict[str, Any]:
    """Archive aged terminal models; return the run summary dict.

    Summary: {archived: [model_id…], freed_bytes: int, dry_run: bool,
    errors: [str…], disk: {...}}. The state file (last_run_at/run_count/
    last_summary) is updated on real runs only — a dry run must not advance
    scheduler-ish state or touch any file besides reading. Never raises for
    per-model failures: they land in ``errors`` and the run continues.

    ``actor`` stamps the ledger entries of automated archives (pass4 #31):
    CLI callers keep "cli", scheduled runs pass "scheduler".

    ``disk_days`` additionally prunes aged operational files (see
    :func:`run_disk_cleanup`); 0 disables that pass.
    """
    home = Path(home)
    own_registry = registry is None
    reg = registry or ModelRegistry(home / "buffer" / "toaster_loop.db")
    summary: Dict[str, Any] = {"archived": [], "freed_bytes": 0,
                               "dry_run": bool(dry_run), "errors": []}
    try:
        for m, size in find_candidates(reg, days):
            if dry_run:
                summary["archived"].append(m.model_id)
                summary["freed_bytes"] += size
                continue
            try:
                reg.archive_model(m.model_id, home=home, actor=actor)
                summary["archived"].append(m.model_id)
                summary["freed_bytes"] += size
            except (RegistryError, OSError) as e:
                logger.warning("janitor: could not archive %s: %s", m.model_id, e)
                summary["errors"].append(f"{m.model_id}: {e}")

        disk = run_disk_cleanup(home, disk_days, dry_run=dry_run)
        summary["disk"] = disk
        summary["freed_bytes"] += disk["freed_bytes"]
        summary["errors"].extend(disk["errors"])

        if not dry_run:
            state = load_state(home)
            state["last_run_at"] = datetime.now(timezone.utc).isoformat(
                timespec="seconds")
            state["run_count"] += 1
            state["last_summary"] = {"archived": summary["archived"],
                                     "freed_bytes": summary["freed_bytes"]}
            save_state(home, state)
    finally:
        if own_registry:
            reg.close()
    return summary

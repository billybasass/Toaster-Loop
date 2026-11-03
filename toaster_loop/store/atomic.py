"""Atomic JSON/text writes that survive Windows file locking (WP1).

Why this exists: the classic tmp + os.replace pattern used everywhere in
toaster_loop breaks on Windows when the tmp name is FIXED. Two writers hitting the
same record (e.g. the delegation-wake run persisting a turn while the client
saves its transcript) collide on ``<file>.json.tmp`` — the second writer's
write_text/os.replace dies with PermissionError [WinError 32] because the
first still holds the handle, and Windows Defender / Search indexer holding
a scan handle on the freshly written tmp does the same. Linux is immune
(rename over an open file is fine), which is exactly why threads "vanished"
only on Windows: every save that hit the collision raised, and the caller's
broad-except swallowed it.

Fix, two parts (both required):
  * UNIQUE tmp name per write — ``<name>.<pid>.<uuid8>.tmp`` (the pattern
    proven in dashboard/api/config.py, plus uuid entropy because the
    dashboard is threaded: one pid writes concurrently).
  * SHORT RETRY around os.replace for transient sharing violations only
    (PermissionError == WinError 32/5; other OSErrors are real failures and
    propagate immediately so a crash-mid-write stays loud — see the existing
    test_thread_write_is_atomic posture).

os.chmod(0o600) is kept for POSIX hardening but is a near-noop on Windows
(HFSD maps it to the read-only attribute only); nothing here relies on it.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

#: os.replace attempts before giving up on a transient sharing violation.
REPLACE_ATTEMPTS = 5

#: Base backoff between replace attempts (linear: 50/100/150/200 ms). Short
#: by design — an AV handle typically releases within tens of ms; anything
#: still locked after the last attempt is a real problem and surfaces.
REPLACE_BACKOFF_S = 0.05


def _is_transient(e: OSError) -> bool:
    """Sharing violation / lock, as opposed to a genuine failure.

    PermissionError covers WinError 32 (file in use — AV/indexer or a peer
    writer) and WinError 5 (access denied); check winerror directly too in
    case a non-standard OSError carries it."""
    if isinstance(e, PermissionError):
        return True
    return getattr(e, "winerror", None) in (32, 33, 5)


def replace_with_retry(src: Path, dst: Path, *,
                       attempts: int = REPLACE_ATTEMPTS,
                       backoff_s: float = REPLACE_BACKOFF_S,
                       sleep: Callable[[float], None] = time.sleep) -> None:
    """os.replace with a bounded retry on transient sharing violations.

    Raises the last PermissionError if every attempt is refused (callers add
    context: path, thread id). Non-transient OSErrors propagate untouched —
    a disk-full or permission-on-directory failure must not be masked by
    retries."""
    last: Optional[OSError] = None
    for i in range(max(1, attempts)):
        try:
            os.replace(src, dst)
            return
        except OSError as e:
            if not _is_transient(e):
                raise
            last = e
            if i + 1 < attempts:
                sleep(backoff_s * (i + 1))
    assert last is not None  # loop ran >=1 attempt, all transient
    raise last


def tmp_path_for(p: Path) -> Path:
    """Unique sibling tmp name for an atomic write.

    pid+uuid because BOTH cross-process writers (CLI + dashboard) and
    in-process threads can target the same record; a fixed ".json.tmp" is
    exactly the collision WP1 is fixing."""
    return p.with_name(f"{p.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")


def atomic_write_text(p: Path, text: str, *,
                      chmod_mode: Optional[int] = 0o600,
                      sleep: Callable[[float], None] = time.sleep) -> None:
    """Durable write: unique tmp in the same dir -> optional chmod -> retried
    os.replace. Cleans up its tmp on failure so no stale *.tmp litters the
    directory (the doctor flags leftovers as the smoking gun)."""
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = tmp_path_for(p)
    try:
        tmp.write_text(text, encoding="utf-8")
        if chmod_mode is not None:
            try:
                # POSIX owner-only hardening; near-noop on Windows (kept but
                # never relied on — the retry loop is what fixes WinError 32).
                os.chmod(tmp, chmod_mode)
            except OSError:
                pass
        replace_with_retry(tmp, p, sleep=sleep)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:  # best effort; doctor catches what survives
                pass


def atomic_write_json(p: Path, obj: Any, *, indent: int = 2,
                      chmod_mode: Optional[int] = 0o600,
                      sleep: Callable[[float], None] = time.sleep) -> None:
    """atomic_write_text for JSON records (ensure_ascii=False + indent, the
    shape every chat/thread-style file in toaster_loop uses)."""
    atomic_write_text(p, json.dumps(obj, ensure_ascii=False, indent=indent),
                      chmod_mode=chmod_mode, sleep=sleep)

"""Cross-process advisory locks — POSIX flock, Windows msvcrt, pidfile fallback.

Why this exists: the easy-loop lock (training/easy.py) and the scheduler's
device-slot locks used to degrade to *no-op* on NT (`fcntl` is absent there),
so a dashboard submit and a Task-Scheduler cron could double-train on the same
GPU from two processes without any signal. This module gives every call site
one interface with three backends, best available first:

  flock     POSIX (Linux/macOS). Identical mechanics to the previous inline
            code — behavior there is unchanged, byte for byte.
  msvcrt    Windows: ``msvcrt.locking`` byte-range lock (LK_NBLCK) on a real
            open handle. Kernel-dropped on process death, no staleness logic.
  pidfile   Last-resort fallback for exotic platforms with neither: an atomic
            O_CREAT|O_EXCL lockfile containing the holder pid; a stale lock
            (pid not alive, per serving.runtime.pid_alive) is reclaimed.

There is deliberately NO silent no-op backend anymore: if even opening the
lock path fails, acquire returns None ("busy/broken") rather than pretending
the lock was taken. ``locking_mode()`` reports what this platform actually
provides so ``doctor`` can warn honestly on NT when only pidfile semantics
are available (advisory across cooperating processes; a process that ignores
the protocol is not blocked).

Interface (all call sites keep the same handle-or-None contract):

    handle = acquire(lock_path)   # truthy handle == ACQUIRED, None == busy
    release(handle)               # idempotent, never raises
"""

from __future__ import annotations

import errno
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional, TextIO

logger = logging.getLogger("toaster_loop.proclock")

#: Backend names reported by locking_mode().
MODE_FLOCK = "flock"
MODE_MSVCRT = "msvcrt"
MODE_PIDFILE = "pidfile"

_fcntl: Any = None
try:  # POSIX advisory locking
    import fcntl as _fcntl  # noqa: F811

    _HAVE_FCNTL = True
except ImportError:  # pragma: no cover - platform dependent
    _HAVE_FCNTL = False

_msvcrt: Any = None
if os.name == "nt":  # pragma: no cover - exercised via injected primitives
    try:
        import msvcrt as _msvcrt
    except ImportError:
        _msvcrt = None


class LockHandle:
    """Opaque acquire() result. `mode` records which backend holds it so
    release() and callers can reason about what "held" actually means."""

    __slots__ = ("path", "mode", "_fh", "_pid")

    def __init__(self, path: Path, mode: str, fh: Optional[TextIO] = None,
                 pid: Optional[int] = None) -> None:
        self.path = path
        self.mode = mode
        self._fh = fh
        self._pid = pid if pid is not None else os.getpid()

    def __bool__(self) -> bool:  # uniform truthy-handle contract
        return True


def locking_mode() -> str:
    """Which backend acquire() will use on THIS platform."""
    if _HAVE_FCNTL:
        return MODE_FLOCK
    if _msvcrt is not None:
        return MODE_MSVCRT
    return MODE_PIDFILE


# ------------------------------------------------------------- flock backend

def _acquire_flock(path: Path) -> Optional[LockHandle]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a+", encoding="utf-8")
    try:
        _fcntl.flock(fh.fileno(), _fcntl.LOCK_EX | _fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    return LockHandle(path, MODE_FLOCK, fh=fh)


def _release_flock(h: LockHandle) -> None:
    if h._fh is not None:
        try:
            _fcntl.flock(h._fh.fileno(), _fcntl.LOCK_UN)
        except Exception:  # noqa: BLE001 - release must never raise
            pass
        try:
            h._fh.close()
        except OSError:
            pass


# ------------------------------------------------------------ msvcrt backend

def _acquire_msvcrt(path: Path) -> Optional[LockHandle]:
    """Windows byte-range lock. LK_NBLCK on bytes [0,1) of the lockfile; the
    kernel releases it when the owning process dies (any exit path), matching
    flock's staleness semantics — no pid heuristics needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a+", encoding="utf-8")
    try:
        fh.seek(0)
        _msvcrt.locking(fh.fileno(), _msvcrt.LK_NBLCK, 1)
    except OSError:
        fh.close()
        return None
    return LockHandle(path, MODE_MSVCRT, fh=fh)


def _release_msvcrt(h: LockHandle) -> None:
    if h._fh is not None:
        try:
            h._fh.seek(0)
            _msvcrt.locking(h._fh.fileno(), _msvcrt.LK_UNLCK, 1)
        except Exception:  # noqa: BLE001 - release must never raise
            pass
        try:
            h._fh.close()
        except OSError:
            pass


# ------------------------------------------------------------ pidfile backend

#: A lockfile older than this whose pid we cannot verify is treated as stale
#: (crash between create and write, or a foreign-format leftover).
PIDFILE_MAX_AGE_S = 3600.0


def _pid_alive(pid: int) -> Optional[bool]:
    """Liveness probe — serving.runtime.pid_alive when importable (it knows
    the Windows OpenProcess trick); plain os.kill fallback otherwise."""
    try:
        from ..serving.runtime import pid_alive

        return pid_alive(pid)
    except Exception:  # noqa: BLE001 - runtime imports httpx-adjacent parents
        pass
    if os.name == "nt":
        # os.kill(pid, 0) on Windows TERMINATES the target (TerminateProcess
        # for any non-console signal value) — never probe liveness that way.
        return None
    try:
        os.kill(pid, 0)
    except OSError as e:
        return False if e.errno == errno.ESRCH else None
    return True


def _read_pidfile(path: Path) -> Optional[int]:
    try:
        txt = path.read_text(encoding="utf-8", errors="replace").strip()
        # First integer token; tolerate trailing newline/partial writes.
        tok = txt.split()[0] if txt else ""
        return int(tok) if tok.lstrip("-").isdigit() else None
    except (OSError, ValueError):
        return None


def _acquire_pidfile(path: Path) -> Optional[LockHandle]:
    """Atomic-create lockfile holding our pid. Contention is resolved honestly:
    holder alive -> busy; holder dead (or file unverifiable and old) -> steal
    once and retry. NOT a security boundary — cooperating processes only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in (0, 1):
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            holder = _read_pidfile(path)
            alive = _pid_alive(holder) if holder is not None else None
            stale = False
            if holder == os.getpid():
                stale = True  # our own leftover after a crash in-process
            elif alive is False:
                stale = True
            elif alive is None and holder is None:
                try:
                    stale = (time.time() - path.stat().st_mtime) > PIDFILE_MAX_AGE_S
                except OSError:
                    stale = True
            if not stale or attempt:
                return None
            try:
                path.unlink()
            except OSError:
                return None
            continue
        except OSError:
            return None
        try:
            os.write(fd, str(os.getpid()).encode("ascii"))
        finally:
            os.close(fd)
        return LockHandle(path, MODE_PIDFILE, pid=os.getpid())
    return None


def _release_pidfile(h: LockHandle) -> None:
    # Only remove OUR lockfile (a stale-steal may have handed it to someone
    # else meanwhile; the pid inside is the ownership token).
    if _read_pidfile(h.path) == h._pid:
        try:
            h.path.unlink()
        except OSError:
            pass


_BACKENDS = {
    MODE_FLOCK: (_acquire_flock, _release_flock),
    MODE_MSVCRT: (_acquire_msvcrt, _release_msvcrt),
    MODE_PIDFILE: (_acquire_pidfile, _release_pidfile),
}

#: Warn once per process when the platform only offers pidfile semantics.
_warned_weak = False


def acquire(lock_path: Any) -> Optional[LockHandle]:
    """Non-blocking exclusive cross-process lock on ``lock_path``.

    Returns a truthy handle while held (release() to give it back), None when
    another process holds it — same contract every previous inline flock used.
    Never raises for lock reasons; only programming errors (bad path types)
    propagate."""
    global _warned_weak
    path = Path(lock_path)
    mode = locking_mode()
    if mode == MODE_PIDFILE and not _warned_weak:
        _warned_weak = True
        logger.warning(
            "cross-process locks on this platform fall back to a pid+liveness "
            "lockfile (%s): advisory only — processes that ignore the protocol "
            "are NOT blocked (no flock/msvcrt available)", path)
    acquire_fn, _ = _BACKENDS[mode]
    return acquire_fn(path)


def release(handle: Optional[Any]) -> None:
    """Give back a handle from acquire(). Idempotent; never raises."""
    if handle is None:
        return
    if isinstance(handle, LockHandle):
        _, release_fn = _BACKENDS[handle.mode]
        release_fn(handle)
        return
    # Legacy/foreign handles (plain file objects from older call sites):
    # closing releases flock implicitly.
    try:
        handle.close()  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001 - release must never raise
        pass


def holder_pid(lock_path: Any) -> Optional[int]:
    """Best-effort read of the pid recorded inside a lockfile (informational:
    used to NAME the contending process in error messages)."""
    return _read_pidfile(Path(lock_path))


def write_holder(handle: Optional[LockHandle], pid: Optional[int] = None) -> None:
    """Record ``pid`` inside a held lock (seek+truncate rewrite), for call
    sites that name the holder in contention errors. No-op on the pidfile
    backend (the pid is already the file's content)."""
    if handle is None or handle._fh is None:
        return
    try:
        handle._fh.seek(0)
        handle._fh.truncate()
        handle._fh.write(str(pid if pid is not None else os.getpid()))
        handle._fh.flush()
    except Exception:  # noqa: BLE001 - informational only, never fatal
        pass


def check_function(system: Optional[str] = None) -> tuple:
    """Doctor probe: (ok, lines). ok=False only when locking is entirely
    unavailable; pidfile-only platforms get a warning-level line but stay ok
    (the doctor caller decides how loud to render it)."""
    mode = locking_mode()
    nt = (system or sys.platform) == "win32"
    if mode == MODE_FLOCK:
        return True, [f"cross-process locks: flock ({mode})"]
    if mode == MODE_MSVCRT:
        return True, [f"cross-process locks: msvcrt byte-range locking ({mode})"]
    lines = [f"cross-process locks: pidfile fallback only ({mode})",
             "  no fcntl/msvcrt on this platform — loop/device-slot locks are "
             "advisory (pid + liveness steal); a process that ignores the "
             "lockfile can still double-train"]
    # On NT without msvcrt something is deeply nonstandard; flag it.
    return (not nt), lines

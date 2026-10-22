"""Engine watchdog (WP4): frozen llama-server detection + opt-in recovery.

Inspired by Strata's watchdog concept — re-implemented inline, no imports
from it (D-12). The problem: a llama-server child can wedge mid-request
(deadlock / driver stall) while its process stays alive; the pid check in
``runtime.pid_alive`` says "healthy" forever and every dependent run hangs.

Detection rule — an instance counts as FROZEN when ALL THREE hold over a
configurable window (default 90 s) WHILE at least one request is in flight:

  (a) no response progress — a request has been continuously in flight for
      the whole window and none completed (in-flight tracking below);
  (b) process CPU-time delta ~0 — /proc/<pid>/stat utime+stime grew by less
      than ``watchdog_cpu_threshold_ms`` (default 10 ms) over the sample;
  (c) disk-IO delta ~0 — /proc/<pid>/io read_bytes+write_bytes grew by at
      most ``watchdog_io_threshold_bytes`` (default 0); when those keys are
      absent (older kernels) we fall back to rchar/wchar from the same file;
      when the whole file is unreadable there is NO io signal and the
      watchdog never kills on partial evidence.

Why CPU+IO deltas and not just "no response": a healthy-but-slow server
computing a long prompt burns CPU steadily (b fails) — it must NEVER be
killed. A wedged server sits at zero CPU *and* zero IO while a request is
parked on it. Silence with no request in flight is idleness, not a freeze:
condition (a) gates the whole rule so an idle server is never touched.

Recovery: verify pid-ownership via the shared WP4 helper
``runtime.terminate_owned_pid`` (the exact sequence deploy.stop_owned_server /
instances.stop_recorded use — a foreign/recycled pid is NEVER killed; we log
and clear the stale state file instead), SIGTERM with SIGKILL escalation
after a grace period, then restart with exponential backoff. Restarts are
capped (max N within a window) and every cap hit logs loudly AND emits
``restart_capped`` — no infinite respawn loop. The restart itself goes
through ``supervisor.start_instance_server``, so the existing crash-loop
budget (#28) also guards against a server that dies immediately on boot.

Events (dashboard bus kind ``serve.watchdog``, plus logger):
  freeze_suspect     first stalled sample of a window (per instance)
  freeze_detected    the full window elapsed under all three conditions
  instance_killed    pid-verified kill completed
  stale_state_cleared pid alive but NOT ours — never killed, state cleared
  instance_restarted backoff restart succeeded {attempt n, backoff_s}
  restart_failed     restart attempt raised (surfaced honestly)
  restart_capped     max restarts within the window reached — watching stops

OPT-IN: with no config changes nothing here runs — ``serve.watchdog_enabled``
defaults to false and neither the dashboard boot nor ``sable serve`` start
the loop unless it is set. Health probes never happen at all (we read
/proc, not HTTP), so there are zero new network paths.

In-flight tracking: requests to a served endpoint flow through two places —
the agent runner's chat calls (core/agent_runner.py) and the dashboard's
chat proxy (dashboard/app.py h_proxy_chat). Each wraps its upstream call in
``track_request(base_url)``; the counter is a lock-guarded dict, a no-op-cheap
side-effect-free tally. Streaming progress needs no extra hooks: a streaming
generation burns CPU/IO, so condition (b)/(c) already capture "progressing".

Testability: every time source and every /proc read is injectable
(``clock``/``sleep``/``read_proc``), and ``step()`` is one pure sample-verdict
cycle the loop simply calls — tests drive it directly with fabricated samples
and a fake clock instead of sleeping 90 s.

Platforms: Linux samples CPU+IO from /proc. macOS gets CPU-only evidence via
``ps -o cputime`` — the verdict then judges stalls on CPU alone (io_source
"cpu_only"). Windows has no cheap stdlib per-pid CPU-time probe, so sampling
is unavailable there: we log once at start and never start the loop (never
kill on partial evidence).
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("sable.serving.watchdog")

#: serve.* keys + defaults (documented in config.example.yaml / README).
WATCHDOG_DEFAULTS: Dict[str, Any] = {
    "watchdog_enabled": False,
    "watchdog_window_seconds": 90.0,
    "watchdog_sample_interval_seconds": 5.0,
    "watchdog_cpu_threshold_ms": 10.0,
    "watchdog_io_threshold_bytes": 0,
    "watchdog_kill_grace_seconds": 8.0,
    "watchdog_restart_backoff_base_seconds": 2.0,
    "watchdog_max_restarts": 3,
    "watchdog_restart_window_seconds": 600.0,
}


@dataclass
class WatchdogConfig:
    """Typed view of the serve.* watchdog keys (defaults = OFF)."""

    enabled: bool = False
    window_seconds: float = 90.0
    sample_interval_seconds: float = 5.0
    cpu_threshold_ms: float = 10.0
    io_threshold_bytes: int = 0
    kill_grace_seconds: float = 8.0
    restart_backoff_base_seconds: float = 2.0
    max_restarts: int = 3
    restart_window_seconds: float = 600.0


def watchdog_config(cfg: Any) -> WatchdogConfig:
    """WatchdogConfig from a loaded SableConfig's raw ``serve:`` dict.

    Lenient like the rest of serve.* (calibration): unknown keys are ignored,
    mistyped values fall back to their default with a warning — a typo must
    never crash serving."""
    sec = getattr(cfg, "serve", None) or {}
    if not isinstance(sec, dict):
        return WatchdogConfig()
    wc = WatchdogConfig()
    raw_en = sec.get("watchdog_enabled")
    if raw_en is not None:
        wc.enabled = str(raw_en).strip().lower() in ("1", "true", "yes", "on") \
            if not isinstance(raw_en, bool) else bool(raw_en)
    for name, attr, cast in (
        ("watchdog_window_seconds", "window_seconds", float),
        ("watchdog_sample_interval_seconds", "sample_interval_seconds", float),
        ("watchdog_cpu_threshold_ms", "cpu_threshold_ms", float),
        ("watchdog_io_threshold_bytes", "io_threshold_bytes", int),
        ("watchdog_kill_grace_seconds", "kill_grace_seconds", float),
        ("watchdog_restart_backoff_base_seconds",
         "restart_backoff_base_seconds", float),
        ("watchdog_max_restarts", "max_restarts", int),
        ("watchdog_restart_window_seconds", "restart_window_seconds", float),
    ):
        raw = sec.get(name)
        if raw is None:
            continue
        try:
            setattr(wc, attr, cast(raw))
        except (TypeError, ValueError):
            logger.warning("watchdog config: %s=%r invalid; keeping default %r",
                           name, raw, getattr(wc, attr))
    if wc.window_seconds < wc.sample_interval_seconds:
        logger.warning("watchdog config: window %ss < sample interval %ss; "
                       "raising window to the sample interval",
                       wc.window_seconds, wc.sample_interval_seconds)
        wc.window_seconds = wc.sample_interval_seconds
    return wc


# ------------------------------------------------------------------ /proc IO

#: SC_CLK_TCK fallback (Linux convention) when sysconf is unavailable.
_HZ_FALLBACK = 100


def _clk_tck() -> int:
    try:
        hz = os.sysconf("SC_CLK_TCK")
        if hz and hz > 0:
            return int(hz)
    except (ValueError, OSError, AttributeError):
        pass
    return _HZ_FALLBACK


@dataclass(frozen=True)
class ProcSample:
    """One /proc reading of a served process."""

    t: float                 # clock value at read time (injected clock)
    cpu_s: float             # utime+stime in seconds
    io_bytes: int            # read+write bytes (see io_source)
    io_source: str           # "read_write" | "rchar_wchar" | "none"


def _have_ps_cpu() -> bool:
    """macOS/FreeBSD: ``ps -o cputime`` gives per-pid CPU time (vendor-neutral
    POSIX tool, present on every darwin install)."""
    return sys.platform == "darwin" and shutil.which("ps") is not None


def proc_sampling_available() -> bool:
    """True when per-pid CPU sampling is possible at all.

    Linux: /proc/<pid>/stat (plus /proc/<pid>/io for the IO evidence).
    macOS: ``ps -o cputime`` gives CPU time only — the verdict then judges on
    CPU alone (io_source="cpu_only"; see freeze_verdict). Windows stays
    unsupported: no cheap stdlib per-pid CPU-time probe exists (tasklist has
    no CPU column), and the watchdog refuses to kill without evidence."""
    if sys.platform.startswith("linux"):
        try:
            Path(f"/proc/{os.getpid()}/stat").read_text(errors="ignore")
            return True
        except OSError:
            return False
    return _have_ps_cpu()


def _ps_cpu_seconds(pid: int) -> Optional[float]:
    """CPU seconds for one pid via ``ps -o cputime=`` (darwin path).

    FORMAT is cumulative CPU time as [HH:]MM:SS[.cs] — sum of user+sys,
    exactly the /proc utime+stime analogue. Returns None when ps fails or the
    pid is gone (same contract as read_proc_sample)."""
    exe = shutil.which("ps")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "-o", "cputime=", "-p", str(pid)],
                             capture_output=True, text=True, timeout=5)
        if out.returncode != 0:
            return None
        txt = (out.stdout or "").strip()
    except Exception:
        return None
    if not txt:
        return None
    try:
        parts = [float(p) for p in txt.split(":")]
    except ValueError:
        return None
    secs = 0.0
    for p in parts:
        secs = secs * 60.0 + p
    # Single-element parse means bare seconds (never seen from ps, but safe).
    return secs if len(parts) > 1 else secs


def read_proc_sample(pid: int, *, clock: Callable[[], float] = time.monotonic,
                     hz: Optional[int] = None) -> Optional[ProcSample]:
    """Read (cpu-time, io-bytes) for one pid; None when unreadable.

    Linux — CPU: /proc/<pid>/stat fields 14/15 (utime/stime, clock ticks). The
    comm field can contain spaces and parens, so we split after the LAST ')'.
    IO: /proc/<pid>/io read_bytes+write_bytes; when those keys are absent we
    fall back to rchar+wchar from the same file (documented in the module
    docstring); an unreadable io file yields io_source="none", which the
    verdict treats as missing evidence (never kill on it).

    macOS — CPU via ``ps -o cputime``; IO is structurally unavailable, so
    io_source="cpu_only" and the verdict judges on CPU alone. Windows returns
    None (unsupported; watchdog stays off)."""
    if sys.platform == "darwin":
        cpu_s = _ps_cpu_seconds(pid)
        if cpu_s is None:
            return None
        return ProcSample(t=clock(), cpu_s=cpu_s, io_bytes=0,
                          io_source="cpu_only")
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(errors="ignore")
        rest = stat[stat.rindex(")") + 2:].split()
        # rest[0] is field 3 (state); utime is field 14 -> rest[11].
        utime = int(rest[11])
        stime = int(rest[12])
    except (OSError, ValueError):
        return None
    hz = hz or _clk_tck()
    io_bytes, io_source = 0, "none"
    try:
        raw = Path(f"/proc/{pid}/io").read_text(errors="ignore")
        vals = {}
        for line in raw.splitlines():
            k, _, v = line.partition(":")
            try:
                vals[k.strip()] = int(v.strip())
            except ValueError:
                continue
        if "read_bytes" in vals and "write_bytes" in vals:
            io_bytes = vals["read_bytes"] + vals["write_bytes"]
            io_source = "read_write"
        elif "rchar" in vals and "wchar" in vals:
            io_bytes = vals["rchar"] + vals["wchar"]
            io_source = "rchar_wchar"
    except OSError:
        pass  # no io evidence; verdict machinery refuses to kill on it
    return ProcSample(t=clock(), cpu_s=(utime + stime) / float(hz),
                      io_bytes=io_bytes, io_source=io_source)


# ------------------------------------------------------------- pure verdict

#: Verdicts from freeze_verdict.
V_IDLE = "idle"          # no request in flight — never a freeze
V_BUSY = "busy"          # cpu or io activity — healthy (but maybe slow)
V_STALLED = "stalled"    # in-flight + zero cpu + zero io — one suspect sample
V_UNKNOWN = "unknown"    # missing evidence (unreadable proc/io)


def freeze_verdict(prev: Optional[ProcSample], cur: ProcSample,
                   cfg: WatchdogConfig, inflight_age: float) -> str:
    """One pure sample-pair verdict — the heart of the detection rule.

    ``inflight_age`` is how long a request has been continuously in flight
    against this instance (0 when nothing is in flight). Conditions (b)/(c)
    are evaluated over the prev->cur interval; the *window* logic (how long
    stalls must persist) lives in EngineWatchdog so this stays side-effect
    free and directly testable."""
    if inflight_age <= 0:
        return V_IDLE
    if prev is None:
        return V_UNKNOWN
    if cur.io_source == "cpu_only" and prev.io_source == "cpu_only":
        # macOS path: IO evidence is structurally unavailable (no /proc), so
        # the AND-condition degrades to CPU-only by design — a zero CPU delta
        # with a request in flight IS the stall signal there. io_bytes stays
        # 0, so the io_threshold check below can never fire on its own.
        cpu_delta_ms = (cur.cpu_s - prev.cpu_s) * 1000.0
        return V_BUSY if cpu_delta_ms > cfg.cpu_threshold_ms else V_STALLED
    if cur.io_source == "none" or prev.io_source == "none":
        # No io evidence either way — refuse to infer a freeze from CPU alone.
        return V_UNKNOWN
    cpu_delta_ms = (cur.cpu_s - prev.cpu_s) * 1000.0
    io_delta = cur.io_bytes - prev.io_bytes
    if cpu_delta_ms > cfg.cpu_threshold_ms:
        return V_BUSY
    if io_delta > cfg.io_threshold_bytes:
        return V_BUSY
    return V_STALLED


# ------------------------------------------------------- in-flight tracking

_inflight_lock = threading.Lock()
#: base_url -> (count, monotonic-time the count left zero)
_inflight: Dict[str, Tuple[int, float]] = {}


def _norm(url: str) -> str:
    return str(url or "").rstrip("/")


def note_request_start(base_url: str, *,
                       clock: Callable[[], float] = time.monotonic) -> None:
    key = _norm(base_url)
    if not key:
        return
    now = clock()
    with _inflight_lock:
        count, since = _inflight.get(key, (0, 0.0))
        _inflight[key] = (count + 1, now if count == 0 else since)
        # WP5: every request start also refreshes the last-activity clock the
        # idle-unload loop reads (one registry, one lock — no extra hooks).
        _last_activity[key] = now


def note_request_end(base_url: str) -> None:
    key = _norm(base_url)
    if not key:
        return
    with _inflight_lock:
        count, since = _inflight.get(key, (0, 0.0))
        if count <= 1:
            _inflight.pop(key, None)
        else:
            _inflight[key] = (count - 1, since)


def inflight_age(base_url: str, *,
                 clock: Callable[[], float] = time.monotonic) -> float:
    """Seconds a request has been continuously in flight (0 when idle)."""
    key = _norm(base_url)
    with _inflight_lock:
        entry = _inflight.get(key)
        if not entry or entry[0] <= 0:
            return 0.0
        return max(0.0, clock() - entry[1])


def inflight_count(base_url: str) -> int:
    with _inflight_lock:
        return (_inflight.get(_norm(base_url)) or (0, 0.0))[0]


def reset_inflight() -> None:
    """Test helper: drop all in-flight tallies."""
    with _inflight_lock:
        _inflight.clear()


# ------------------------------------------------- last-activity tracking
# WP5 idle-unload reuses this same lock-guarded registry for "when did this
# endpoint last see a request". note_request_start stamps activity too, so no
# extra hooks are needed on the request paths; note_activity lets the idle
# loop (and tests) seed/refresh the clock explicitly. idle_seconds returns
# None when the endpoint was never seen — the caller seeds it rather than
# assuming an infinite idle window.

_last_activity: Dict[str, float] = {}


def note_activity(base_url: str, *,
                  clock: Callable[[], float] = time.monotonic) -> None:
    key = _norm(base_url)
    if not key:
        return
    with _inflight_lock:
        _last_activity[key] = clock()


def idle_seconds(base_url: str, *,
                 clock: Callable[[], float] = time.monotonic) -> Optional[float]:
    """Seconds since the last tracked request to this endpoint; None when no
    activity was ever recorded (caller seeds the window from now)."""
    key = _norm(base_url)
    with _inflight_lock:
        seen = _last_activity.get(key)
        if seen is None:
            return None
        return max(0.0, clock() - seen)


def reset_activity() -> None:
    """Test helper: drop all last-activity stamps."""
    with _inflight_lock:
        _last_activity.clear()


@contextlib.contextmanager
def track_request(base_url: str):
    """Count one upstream request as in flight for the duration of the block.

    Side-effect-free when the watchdog is off (a dict tally under a lock);
    never raises into the request path."""
    try:
        note_request_start(base_url)
        yield
    finally:
        note_request_end(base_url)


# ---------------------------------------------------------------- watchdog

#: instance state file record for one watched engine.
@dataclass
class _WatchTarget:
    instance_id: str
    pid: int
    base_url: str
    prev: Optional[ProcSample] = None
    suspect_since: Optional[float] = None


EmitFn = Callable[[str, Dict[str, Any]], None]
RestartFn = Callable[[str], bool]


class EngineWatchdog:
    """Periodic freeze detector over the managed serving instances.

    All time sources and process readers are injectable so tests can drive
    ``step()`` with a fake clock and fabricated /proc samples; the real loop
    is just ``step() + sleep(interval)``. ``emit(kind, payload)`` receives
    the watchdog events (dashboard bus glue); default logs only."""

    def __init__(self, cfg: WatchdogConfig, *, home: Any = None,
                 emit: Optional[EmitFn] = None,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep,
                 read_proc: Optional[Callable[[int], Optional[ProcSample]]] \
                 = None,
                 restart_fn: Optional[RestartFn] = None,
                 list_targets: Optional[Callable[[], List[_WatchTarget]]] \
                 = None) -> None:
        self.cfg = cfg
        self.home = home
        self.clock = clock
        self.sleep = sleep
        self.emit = emit or (lambda kind, payload: None)
        self.read_proc = read_proc or (
            lambda pid: read_proc_sample(pid, clock=clock))
        self._restart_fn = restart_fn
        self._list_targets = list_targets
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        #: instance_id -> target (fresh each discovery pass, state preserved)
        self._targets: Dict[str, _WatchTarget] = {}
        #: instance_id -> [monotonic restart stamps]
        self._restarts: Dict[str, List[float]] = {}
        #: instance_id -> monotonic time until which restarts are capped
        self._capped_until: Dict[str, float] = {}
        self._unavailable_logged = False

    # ------------------------------------------------------------- events
    def _event(self, name: str, **payload: Any) -> None:
        try:
            self.emit("serve.watchdog", {"event": name, **payload})
        except Exception:  # a dead bus must never kill the watchdog
            logger.debug("watchdog emit failed", exc_info=True)

    # ---------------------------------------------------------- discovery
    def discover(self) -> List[_WatchTarget]:
        """Spawn-mode instances with a live recorded pid (state files).

        Attach-mode states have no pid — nothing to watch. The reserved
        "default" folds in the legacy serve_state.json, matching the router
        pool's view."""
        if self.home is None:
            return []
        from pathlib import Path as _P

        from . import instances as srv_inst
        from .deploy import read_serve_state

        home = _P(self.home)
        out: List[_WatchTarget] = []
        try:
            defs = srv_inst.load_instances(home)
        except Exception as e:  # noqa: BLE001 — discovery never raises
            logger.debug("watchdog: instance enumeration failed: %s", e)
            defs = []
        for inst in defs:
            iid = str(inst.get("id") or "")
            if not iid:
                continue
            st = srv_inst.read_state(home, iid)
            if (st or {}).get("mode") == "attach":
                continue
            pid = (st or {}).get("pid")
            url = str((st or {}).get("base_url") or "")
            if not pid or not url:
                continue
            out.append(_WatchTarget(instance_id=iid, pid=int(pid),
                                    base_url=url))
        # The legacy view may carry a default server the defs above did not
        # (the config-derived default has no state file of its own).
        if srv_inst.DEFAULT_ID not in {t.instance_id for t in out}:
            st = read_serve_state(home)
            if st and st.get("pid") and st.get("mode", "spawn") != "attach" \
                    and st.get("base_url"):
                out.append(_WatchTarget(
                    instance_id=srv_inst.DEFAULT_ID, pid=int(st["pid"]),
                    base_url=str(st["base_url"])))
        return out

    # --------------------------------------------------------------- step
    def step(self) -> None:
        """One sample+verdict cycle over every watched instance.

        Pure enough to test directly: reads (injected) proc samples, applies
        freeze_verdict, drives the suspect->detected state machine and the
        kill/restart/cap flow. The background loop only adds sleep()."""
        now = self.clock()
        targets = (self._list_targets() if self._list_targets
                   is not None else self.discover())
        for t in targets:
            prev_state = self._targets.get(t.instance_id)
            if prev_state is not None:
                # preserve detection state across discovery passes
                t.prev = prev_state.prev
                t.suspect_since = prev_state.suspect_since
            cur = None
            try:
                cur = self.read_proc(t.pid)
            except Exception:  # noqa: BLE001 — a dead reader is no evidence
                cur = None
            if cur is None:
                if not self._unavailable_logged:
                    logger.warning(
                        "watchdog: /proc sampling unavailable for pid %s "
                        "(instance %s); detection disabled on this platform",
                        t.pid, t.instance_id)
                    self._unavailable_logged = True
                self._targets.pop(t.instance_id, None)
                continue
            age = inflight_age(t.base_url, clock=self.clock)
            verdict = freeze_verdict(t.prev, cur, self.cfg, age)
            handled = False
            if verdict == V_STALLED:
                if t.suspect_since is None:
                    t.suspect_since = now
                    self._event("freeze_suspect", instance_id=t.instance_id,
                                pid=t.pid, base_url=t.base_url,
                                window_seconds=self.cfg.window_seconds)
                    logger.warning(
                        "watchdog: %s (pid %s) stalled — in-flight request "
                        "with zero CPU/IO delta; watching for %.0fs",
                        t.instance_id, t.pid, self.cfg.window_seconds)
                elif now - t.suspect_since >= self.cfg.window_seconds:
                    self._on_freeze(t, now)
                    # _on_freeze owns the target's fate (kill/restart/unwatch);
                    # do NOT re-arm it below with the stale pid.
                    handled = True
            else:
                if verdict != V_UNKNOWN:
                    if t.suspect_since is not None:
                        logger.info("watchdog: %s recovered before the "
                                    "window elapsed (%s)",
                                    t.instance_id, verdict)
                    t.suspect_since = None
                # V_UNKNOWN keeps the suspect clock but never advances
                # detection — missing evidence must not kill.
            if handled:
                continue
            t.prev = cur
            self._targets[t.instance_id] = t
        # drop vanished instances
        live = {t.instance_id for t in targets}
        for gone in [k for k in self._targets if k not in live]:
            self._targets.pop(gone, None)

    # ------------------------------------------------------ kill + restart
    def _on_freeze(self, t: _WatchTarget, now: float) -> None:
        """freeze_detected -> ownership-verified kill -> backoff restart."""
        iid = t.instance_id
        if now < self._capped_until.get(iid, 0.0):
            # Already gave up on this instance within the window: no further
            # kills/restarts (the cap event was announced once, loudly).
            logger.debug("watchdog: %s capped; ignoring further freeze", iid)
            self._targets.pop(iid, None)
            return
        stalled_for = round(now - (t.suspect_since or now), 1)
        # Clear this target's detection state up front — whatever happens
        # below, the next window starts from scratch (the same object may be
        # handed back by discovery with stale fields otherwise).
        t.suspect_since = None
        t.prev = None
        self._event("freeze_detected", instance_id=t.instance_id, pid=t.pid,
                    base_url=t.base_url, stalled_for_seconds=stalled_for)
        logger.error(
            "watchdog: FREEZE detected on %s (pid %s) — request in flight, "
            "zero CPU and zero IO for %.0fs", t.instance_id, t.pid,
            self.cfg.window_seconds)
        killed = self._kill_verified(t)
        if not killed:
            # stale/foreign pid or kill failure: never restart onto a pid we
            # do not own; drop the target (a fresh discovery re-arms it).
            self._targets.pop(t.instance_id, None)
            return
        self._restart_with_backoff(t, now)

    def _kill_verified(self, t: _WatchTarget) -> bool:
        """Kill exactly one instance under the ownership rule. True = killed.

        In-process children (this process spawned them — the shared registry
        is itself the strongest ownership proof) go through
        ``supervisor.stop_instance``. Everything else goes through the SAME
        WP4-extracted helper the stop paths use:
        ``runtime.terminate_owned_pid`` verifies via ``pid_alive`` and refuses
        foreign/recycled pids (verdict "foreign" -> log + clear stale state,
        never kill)."""
        from . import instances as srv_inst
        from . import runtime
        from .supervisor import owned_instance_ids, stop_instance

        if t.instance_id in owned_instance_ids():
            stopped = stop_instance(t.instance_id,
                                    grace=self.cfg.kill_grace_seconds)
            if stopped:
                self._event("instance_killed", instance_id=t.instance_id,
                            pid=t.pid, base_url=t.base_url,
                            verified="in-process-registry")
                logger.error("watchdog: killed frozen engine %s (pid %s)",
                             t.instance_id, t.pid)
            return bool(stopped)

        verdict = runtime.terminate_owned_pid(
            t.pid, grace=self.cfg.kill_grace_seconds)
        if verdict == "foreign":
            logger.error(
                "watchdog: pid %s for instance %s exists but is NOT owned by "
                "sable (stale state / recycled pid) — never killing; "
                "clearing stale state", t.pid, t.instance_id)
            self._event("stale_state_cleared", instance_id=t.instance_id,
                        pid=t.pid, reason="pid not owned by sable")
            try:
                if self.home is not None:
                    srv_inst.clear_state(self.home, t.instance_id)
            except OSError:
                pass
            return False
        if verdict == "gone":
            logger.warning("watchdog: instance %s pid %s already exited",
                           t.instance_id, t.pid)
            self._event("stale_state_cleared", instance_id=t.instance_id,
                        pid=t.pid, reason="process already exited")
            return False
        self._event("instance_killed", instance_id=t.instance_id, pid=t.pid,
                    base_url=t.base_url, verified="pid_alive-ownership")
        logger.error("watchdog: killed frozen engine %s (pid %s)",
                     t.instance_id, t.pid)
        return True

    def _restart_with_backoff(self, t: _WatchTarget, now: float) -> None:
        """Exponential-backoff restart, capped within a window."""
        iid = t.instance_id
        stamps = [s for s in self._restarts.get(iid, [])
                  if now - s <= self.cfg.restart_window_seconds]
        attempt = len(stamps) + 1
        if attempt > self.cfg.max_restarts:
            logger.error(
                "watchdog: RESTART CAP reached for %s — %d restarts within "
                "%.0fs; giving up (no further automatic restarts; start it "
                "explicitly to clear)", iid, len(stamps),
                self.cfg.restart_window_seconds)
            self._event("restart_capped", instance_id=iid, pid=t.pid,
                        base_url=t.base_url, restarts=len(stamps),
                        window_seconds=self.cfg.restart_window_seconds)
            self._capped_until[iid] = now + self.cfg.restart_window_seconds
            self._targets.pop(iid, None)
            return
        backoff = self.cfg.restart_backoff_base_seconds * (2 ** (attempt - 1))
        self.sleep(backoff)
        ok = False
        err: Optional[str] = None
        try:
            fn = self._restart_fn
            if fn is None and self.home is not None:
                fn = restart_instance(self.cfg, self.home)
            if fn is None:
                raise RuntimeError("no restart function configured")
            ok = bool(fn(iid))
        except Exception as e:  # noqa: BLE001 — surfaced as an event
            err = f"{type(e).__name__}: {e}"[:300]
        stamps.append(now)
        self._restarts[iid] = stamps
        if ok:
            self._event("instance_restarted", instance_id=iid,
                        base_url=t.base_url, attempt=attempt,
                        backoff_seconds=backoff)
            logger.error("watchdog: restarted %s (attempt %d after %.1fs "
                         "backoff)", iid, attempt, backoff)
            # Re-arm watching with a clean slate; the new pid is discovered
            # on the next pass from the refreshed state file.
            self._targets.pop(iid, None)
        else:
            self._event("restart_failed", instance_id=iid, base_url=t.base_url,
                        attempt=attempt, backoff_seconds=backoff,
                        error=err or "restart returned false")
            logger.error("watchdog: restart of %s failed (attempt %d): %s",
                         iid, attempt, err or "returned false")
            self._targets.pop(iid, None)

    # ------------------------------------------------------------ lifecycle
    def run(self) -> None:
        """Blocking loop: step + interruptible sleep. Used via a daemon
        thread by start_watchdog(); tests call step() directly."""
        while not self._stop.is_set():
            try:
                self.step()
            except Exception:  # noqa: BLE001 — the loop must survive anything
                logger.exception("watchdog: step failed; continuing")
            self._stop.wait(self.cfg.sample_interval_seconds)

    def start(self) -> bool:
        """Start the background loop (no-op when disabled/unavailable)."""
        if not self.cfg.enabled:
            return False
        if not proc_sampling_available():
            if not self._unavailable_logged:
                logger.warning(
                    "watchdog: enabled but /proc sampling is unavailable on "
                    "this platform — engine freeze detection is off and "
                    "nothing will ever be killed")
                self._unavailable_logged = True
            return False
        if self._thread is not None and self._thread.is_alive():
            return True
        self._stop.clear()
        self._thread = threading.Thread(target=self.run, daemon=True,
                                        name="engine-watchdog")
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()


# ------------------------------------------------------- default restart fn

def restart_instance(cfg: WatchdogConfig, home: Any) -> RestartFn:
    """Default recovery: respawn the instance via the canonical machinery.

    Goes through ``supervisor.start_instance_server`` so the crash-loop
    budget (#28) also refuses to respawn a server that dies immediately on
    boot — a second layer under the watchdog's own restart cap. Rewrites the
    per-instance state file (and the legacy serve_state.json for "default")
    in the same shape the dashboard writes, so every reader (router pool,
    metrics, panels) sees the fresh server."""

    def _restart(instance_id: str) -> bool:
        from pathlib import Path as _P

        from . import instances as srv_inst
        from .supervisor import start_instance_server

        home_p = _P(home)
        inst = srv_inst.get_instance(home_p, instance_id)
        if inst is None:
            logger.error("watchdog: cannot restart %s — no instance record",
                         instance_id)
            return False
        old_state = srv_inst.read_state(home_p, instance_id) or {}
        run_inst = dict(inst)
        run_inst["port"] = int(old_state.get("port") or 0)
        handle = start_instance_server(
            _watchdog_parent_cfg(), run_inst,
            log_path=home_p / "logs" / f"server-{instance_id}.log",
        )
        new_state = {
            "mode": "spawn",
            "pid": handle.process.pid if handle.process else None,
            "host": "127.0.0.1",
            "port": int(old_state.get("port") or 0),
            "base_url": handle.base_url,
            "model_name": handle.model_name,
            "gguf_path": str(inst.get("gguf") or ""),
            "instance_id": instance_id,
            "role": inst.get("role"),
            "restarted_by": "watchdog",
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        srv_inst.write_state(home_p, instance_id, new_state)
        if instance_id == srv_inst.DEFAULT_ID:
            from .deploy import write_serve_state

            write_serve_state(home_p, new_state)
        return True

    return _restart


#: The watchdog restarts with the process-wide loaded config; the dashboard
#: glue sets this (tests inject their own restart_fn and never touch it).
_PARENT_CFG: Any = None


def _watchdog_parent_cfg() -> Any:
    if _PARENT_CFG is not None:
        return _PARENT_CFG
    from ..core.config import load_config

    return load_config()


# --------------------------------------------------------------- boot glue

def start_watchdog(cfg: Any, home: Any, *, emit: Optional[EmitFn] = None,
                   restart_fn: Optional[RestartFn] = None
                   ) -> Optional[EngineWatchdog]:
    """Boot hook: create + start the watchdog IF serve.watchdog_enabled.

    Returns the engine (for stopping) or None when disabled/unavailable —
    with no config changes this returns None immediately and behavior is
    byte-identical to pre-WP4."""
    wc = watchdog_config(cfg)
    if not wc.enabled:
        return None
    global _PARENT_CFG
    _PARENT_CFG = cfg
    wd = EngineWatchdog(wc, home=home, emit=emit, restart_fn=restart_fn)
    if not wd.start():
        return None
    logger.info(
        "engine watchdog started (window=%.0fs interval=%.0fs cpu<=%.0fms "
        "io<=%dB max_restarts=%d/%.0fs) — frozen engines will be killed and "
        "restarted automatically",
        wc.window_seconds, wc.sample_interval_seconds, wc.cpu_threshold_ms,
        wc.io_threshold_bytes, wc.max_restarts, wc.restart_window_seconds)
    return wd

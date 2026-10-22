"""GPU coexistence (WP5): idle-unload / lazy-load for served instances.

Inspired by Strata's idle-unload / lazy-load coexistence pattern — re-implemented
inline, no imports from it (D-12). The problem: a llama-server holds its model
resident in VRAM while nobody is chatting with it, and the training scheduler's
resource gate then refuses perfectly good jobs ("GPU free memory X < Y MiB").
Two serving-side mechanisms plus one handshake close the loop:

* **Idle unload** (``idle_unload_seconds`` > 0): a supervisor-side timer — the
  sibling loop to the WP4 engine watchdog, same injectable-clock design —
  gracefully stops (SIGTERM through the existing owned-stop paths;
  ``terminate_owned_pid`` only escalates to SIGKILL after the grace window)
  any running instance with NO tracked request for the whole idle window, and
  records it in an ``unloaded`` state so the dashboard can show it. A tick
  never unloads while the WP4 in-flight counter for that endpoint is > 0.

* **Lazy load** (``lazy_load: true``): when a request resolves to an instance
  that sits in the unloaded state, the resolve path transparently re-launches
  it (blocking — cold-start latency is accepted by design) and then forwards.
  With lazy_load off, requests to an unloaded instance fail with a clear
  message telling the user to load it or enable lazy_load.

* **Train+serve handshake** (``min_free_vram_mib`` > 0 AND idle unload enabled):
  when a job FAILS the scheduler's free-VRAM gate but WOULD PASS assuming the
  served instances were unloaded, the scheduler asks this module to unload
  them first (same graceful path), polls ``gpu_free_mib`` until the VRAM is
  actually free (bounded timeout), and only then admits the job. Under
  ``coexist_policy: refuse`` (the default) nothing is unloaded — an event
  explains that unloading WOULD have admitted the job.

Race policy (documented choice): unload + load are serialized per instance id
under a lock. A request arriving mid-unload blocks on that lock; when it wins,
lazy_load reloads transparently and forwards, or (lazy off) it gets a clear
retryable error naming the instance. The scheduler's VRAM wait happens AFTER
the unload lock is released, so a lazy reload racing a training launch can
lose the VRAM race — llama-server then fails its healthcheck with an honest
ServeError instead of corrupting anything (run lazy_load=false alongside
coexist_policy=unload if you want strict train-first behavior).

Events (dashboard bus kind ``serve.coexist``):
  instance_unloaded       {id, reason: idle | manual | coexist}
  instance_loaded         {id, reason: lazy | manual | coexist-attach}
  coexist_unload_triggered {job_id, free_mib, required_mib, projected_free_mib, targets}
  coexist_refused         {job_id, free_mib, required_mib, projected_free_mib, would_pass_if_unloaded: true}

OPT-IN: every key defaults off (``idle_unload_seconds: 0``, ``lazy_load:
false``, ``min_free_vram_mib: 0``, ``coexist_policy: refuse``). With no config
changes ``build()`` returns None and every hook in agent_runner / dashboard /
scheduler is a no-op — behavior is byte-identical to pre-WP5. Loopback-only;
no new network paths (unloading touches only owned pids, loading reuses the
canonical spawn machinery).

Testability: clock/sleep/free-vram probe are injectable and ``step()`` is one
pure decide-and-act cycle, mirroring ``EngineWatchdog``.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("sable.serving.coexist")

#: serve.* keys + defaults (documented in config.example.yaml / README).
COEXIST_DEFAULTS: Dict[str, Any] = {
    "idle_unload_seconds": 0.0,      # 0 = disabled
    "lazy_load": False,
    "min_free_vram_mib": 0,          # 0 = unset (no coexistence handshake)
    "coexist_policy": "refuse",      # refuse | unload
}


def _as_bool(raw: Any, default: bool) -> bool:
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


@dataclass
class CoexistConfig:
    """Typed view of the serve.* coexistence keys (defaults = OFF)."""

    idle_unload_seconds: float = 0.0
    lazy_load: bool = False
    min_free_vram_mib: int = 0
    coexist_policy: str = "refuse"       # refuse | unload
    #: loop cadence (not a busy-poll: one pass per interval, Event.wait based)
    sample_interval_seconds: float = 5.0
    #: SIGTERM grace before the owned-stop path escalates to SIGKILL
    unload_grace_seconds: float = 8.0
    #: handshake: how long to wait for VRAM to actually free after unload
    vram_wait_timeout_seconds: float = 60.0
    vram_poll_interval_seconds: float = 2.0

    @property
    def idle_enabled(self) -> bool:
        return self.idle_unload_seconds > 0

    @property
    def handshake_enabled(self) -> bool:
        # The rescue path only arms when the user declared BOTH a training VRAM
        # floor and an idle-unload window (WP5 spec item 5).
        return self.min_free_vram_mib > 0 and self.idle_enabled

    @property
    def any_enabled(self) -> bool:
        return (self.idle_enabled or self.lazy_load
                or self.handshake_enabled)


def coexist_config(cfg: Any) -> CoexistConfig:
    """CoexistConfig from a loaded SableConfig's raw ``serve:`` dict.

    Lenient like watchdog_config/calibration: unknown keys ignored, mistyped
    values fall back to their default with a warning."""
    sec = getattr(cfg, "serve", None) or {}
    if not isinstance(sec, dict):
        return CoexistConfig()
    cc = CoexistConfig()
    raw = sec.get("idle_unload_seconds")
    if raw is not None:
        try:
            cc.idle_unload_seconds = float(raw)
        except (TypeError, ValueError):
            logger.warning("coexist config: idle_unload_seconds=%r invalid; "
                           "keeping default %r", raw,
                           cc.idle_unload_seconds)
    if sec.get("lazy_load") is not None:
        cc.lazy_load = _as_bool(sec.get("lazy_load"), False)
    raw = sec.get("min_free_vram_mib")
    if raw is not None:
        try:
            cc.min_free_vram_mib = int(float(raw))
        except (TypeError, ValueError):
            logger.warning("coexist config: min_free_vram_mib=%r invalid; "
                           "keeping default %r", raw, cc.min_free_vram_mib)
    pol = sec.get("coexist_policy")
    if pol is not None:
        pol_s = str(pol).strip().lower()
        if pol_s in ("refuse", "unload"):
            cc.coexist_policy = pol_s
        else:
            logger.warning("coexist config: coexist_policy=%r unknown; "
                           "keeping default %r (safe)", pol, cc.coexist_policy)
    for name, attr, cast in (
        ("coexist_sample_interval_seconds", "sample_interval_seconds", float),
        ("coexist_unload_grace_seconds", "unload_grace_seconds", float),
        ("coexist_vram_wait_timeout_seconds", "vram_wait_timeout_seconds",
         float),
        ("coexist_vram_poll_interval_seconds", "vram_poll_interval_seconds",
         float),
    ):
        raw = sec.get(name)
        if raw is None:
            continue
        try:
            setattr(cc, attr, cast(raw))
        except (TypeError, ValueError):
            logger.warning("coexist config: %s=%r invalid; keeping default %r",
                           name, raw, getattr(cc, attr))
    return cc


# ------------------------------------------------------------- per-instance
# serialization. Unload and (lazy) load of one instance id are mutually
# exclusive; requests racing an unload block here and then either reload
# (lazy_load) or surface the retryable error (documented race policy).

_locks_guard = threading.Lock()
_instance_locks: Dict[Tuple[str, str], threading.Lock] = {}


def _lock_for(home: Any, instance_id: str) -> threading.Lock:
    key = (str(home), str(instance_id))
    with _locks_guard:
        lock = _instance_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _instance_locks[key] = lock
        return lock


def _norm(url: str) -> str:
    return str(url or "").rstrip("/")


class CoexistError(RuntimeError):
    pass


EmitFn = Callable[[str, Dict[str, Any]], None]


class Coexistence:
    """Idle-unload / lazy-load engine + the train<->serve handshake.

    All time/VRAM sources are injectable; tests drive ``step()`` directly with
    a fake clock and a scripted free-VRAM probe. ``emit(kind, payload)``
    receives events for the dashboard bus (default: log only)."""

    def __init__(self, cfg: CoexistConfig, *, parent_cfg: Any = None,
                 home: Any = None, emit: Optional[EmitFn] = None,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep,
                 free_vram: Optional[Callable[[], Optional[int]]] = None,
                 ) -> None:
        self.cfg = cfg
        #: the full SableConfig (needed by start_instance_server); when
        #: omitted callers/tests rely on load_config() at spawn time.
        self.parent_cfg = parent_cfg
        self.home = Path(home) if home is not None else None
        self.clock = clock
        self.sleep = sleep
        self.emit = emit or (lambda kind, payload: None)
        # Late-bound through the module so tests can monkeypatch either this
        # attribute or scheduler.gpu_free_mib itself.
        self._free_vram = free_vram or _gpu_free_mib

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        #: base_url -> monotonic time of the most recent tracked activity
        #  (first sighting seeds the idle window from loop-start, documented)
        self._last_seen: Dict[str, float] = {}
        #: instance ids currently mid-unload (status surfaces "unloading")
        self._unloading: set = set()

    # --------------------------------------------------------------- events
    def _event(self, name: str, **payload: Any) -> None:
        try:
            self.emit("serve.coexist", {"event": name, **payload})
        except Exception:  # a dead bus must never kill serving
            logger.debug("coexist emit failed", exc_info=True)

    # ------------------------------------------------------ state helpers
    def _require_home(self) -> Path:
        if self.home is None:
            raise CoexistError("coexistence has no home")
        return self.home

    def _read_state(self, iid: str) -> Optional[Dict[str, Any]]:
        from . import instances as srv_inst
        from .deploy import read_serve_state

        home = self._require_home()
        st = srv_inst.read_state(home, iid)
        if not st and iid == srv_inst.DEFAULT_ID:
            st = read_serve_state(home)
        return st

    def _write_state(self, iid: str, state: Dict[str, Any]) -> None:
        from . import instances as srv_inst
        from .deploy import write_serve_state

        home = self._require_home()
        srv_inst.write_state(home, iid, state)
        if iid == srv_inst.DEFAULT_ID:
            write_serve_state(home, state)

    def _clear_state(self, iid: str) -> None:
        from . import instances as srv_inst
        from .deploy import clear_serve_state

        home = self._require_home()
        srv_inst.clear_state(home, iid)
        if iid == srv_inst.DEFAULT_ID:
            clear_serve_state(home)

    def load_state_of(self, iid: str) -> str:
        """loaded | unloaded | unloading | stopped (dashboard-facing label)."""
        if iid in self._unloading:
            return "unloading"
        st = self._read_state(iid)
        if not st:
            return "stopped"
        from . import instances as srv_inst

        if srv_inst.state_alive(st):
            return "loaded"
        return "unloaded" if st.get("unloaded") else "stopped"

    def running_targets(self) -> List[Dict[str, Any]]:
        """Running spawn-mode instances with pid + base_url (unloadable set).

        Attach-mode states are never touched — they own nothing we could free.
        The reserved default folds in the legacy serve_state.json, matching
        the watchdog's discovery."""
        from . import instances as srv_inst

        out: List[Dict[str, Any]] = []
        if self.home is None:
            return out
        try:
            defs = srv_inst.load_instances(self._require_home())
        except Exception as e:  # noqa: BLE001 — discovery never raises
            logger.debug("coexist: instance enumeration failed: %s", e)
            defs = []
        for inst in defs:
            iid = str(inst.get("id") or "")
            if not iid:
                continue
            st = self._read_state(iid)
            if not st or st.get("mode") == "attach" or st.get("unloaded"):
                continue
            pid, url = st.get("pid"), str(st.get("base_url") or "")
            if not pid or not url:
                continue
            from .runtime import pid_alive

            if pid_alive(int(pid)) is False:
                continue  # dead pid => not "running"; stale state untouched
            out.append({"id": iid, "pid": int(pid), "base_url": url,
                        "gguf_path": st.get("gguf_path"),
                        "context_length": (inst or {}).get("context_length")})
        return out

    # ------------------------------------------------------------- unload
    def unload_instance(self, iid: str, *, reason: str = "manual",
                        check_inflight: bool = True) -> bool:
        """Gracefully stop one instance and record it as ``unloaded``.

        Ownership rule preserved: in-process children go through
        ``supervisor.stop_instance``; cross-process pids through
        ``instances.stop_recorded`` (the shared verify-then-kill helper — a
        foreign/recycled pid is never killed). SIGTERM first; SIGKILL only via
        the grace escalation inside those paths. True when a stop happened."""
        from . import instances as srv_inst
        from .watchdog import inflight_count

        if self.home is None:
            return False
        lock = _lock_for(self.home, iid)
        with lock:
            st = self._read_state(iid)
            if not st or st.get("mode") == "attach" or st.get("unloaded"):
                return False
            url = str(st.get("base_url") or "")
            if check_inflight and url and inflight_count(url) > 0:
                logger.info("coexist: %s has a request in flight — not "
                            "unloading", iid)
                return False
            self._unloading.add(iid)
            try:
                from .supervisor import stop_instance
                stopped = stop_instance(iid, grace=self.cfg.unload_grace_seconds)
                if not stopped:
                    stopped = srv_inst.stop_recorded(
                        self.home, iid, grace=self.cfg.unload_grace_seconds)
                # stop_recorded clears the state file on success; make sure no
                # stale pid/state survives either way before writing the tomb.
                self._clear_state(iid)
                marker = dict(st)
                marker.pop("pid", None)
                marker["unloaded"] = True
                marker["unload_reason"] = reason
                marker["unloaded_at"] = time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                self._write_state(iid, marker)
            finally:
                self._unloading.discard(iid)
            logger.info("coexist: unloaded instance %s (reason=%s)", iid,
                        reason)
            self._event("instance_unloaded", id=iid, reason=reason,
                        base_url=url or None)
            return bool(stopped)

    def unload_all(self, *, reason: str = "coexist") -> List[str]:
        """Unload every currently-running instance (handshake entry). Skips
        anything with a request in flight; returns the ids actually unloaded."""
        unloaded: List[str] = []
        for t in self.running_targets():
            if self.unload_instance(t["id"], reason=reason):
                unloaded.append(t["id"])
        return unloaded

    # -------------------------------------------------------------- load
    def load_instance(self, iid: str, *, reason: str = "manual",
                      ) -> Dict[str, Any]:
        """(Re)launch one instance; returns the fresh state record.

        Reuses the canonical spawn machinery (``supervisor.start_instance_
        server`` — crash-loop budget included), mirroring watchdog.restart_-
        instance so every reader (router pool, panels) sees the same shape.
        Raises CoexistError when the instance cannot be loaded."""
        from . import instances as srv_inst
        from .supervisor import ServeError, start_instance_server

        if self.home is None:
            raise CoexistError("coexistence has no home")
        lock = _lock_for(self.home, iid)
        with lock:
            st = self._read_state(iid)
            if st and not st.get("unloaded") and srv_inst.state_alive(st):
                return st  # already loaded (double-load race lost politely)
            inst = srv_inst.get_instance(self.home, iid, cfg=self.parent_cfg)
            if inst is None:
                raise CoexistError(f"no instance record for {iid!r}")

            # Attach instances: re-attach = healthcheck + state write; nothing
            # is ever spawned or killed.
            if inst.get("attach_url"):
                from .supervisor import healthcheck
                base = str(inst["attach_url"])
                if not healthcheck(base, timeout=5.0):
                    raise CoexistError(
                        f"attach target {base} is not healthy — cannot load "
                        f"instance {iid}")
                state = {"mode": "attach", "base_url": base,
                         "attach_url": base,
                         "model_name": inst.get("name") or "attached-model",
                         "instance_id": iid,
                         "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                     time.gmtime())}
                self._write_state(iid, state)
                self._event("instance_loaded", id=iid, reason=reason,
                            base_url=base)
                return state

            gguf = inst.get("gguf")
            if not gguf:
                raise CoexistError(
                    f"instance {iid} has neither gguf nor attach_url")
            old_state = st or {}
            run_inst = dict(inst)
            port = int(old_state.get("port") or 0) or None
            if not port:
                try:
                    port = srv_inst.pick_free_port(self.home, self.parent_cfg,
                                                   exclude_id=iid)
                except Exception:  # parent cfg may be absent in unit tests
                    from .supervisor import _free_port
                    port = _free_port("127.0.0.1")
            run_inst["port"] = port
            parent = self.parent_cfg
            if parent is None:
                from ..core.config import load_config
                parent = load_config()
            try:
                handle = start_instance_server(
                    parent, run_inst,
                    log_path=self.home / "logs" / f"server-{iid}.log")
            except ServeError as e:
                raise CoexistError(f"load of {iid} failed: {e}") from e
            state = {
                "mode": "spawn",
                "pid": handle.process.pid if handle.process else None,
                "host": "127.0.0.1", "port": port,
                "base_url": handle.base_url, "model_name": handle.model_name,
                "gguf_path": str(gguf), "instance_id": iid,
                "role": inst.get("role"),
                "loaded_by": f"coexist:{reason}",
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                            time.gmtime()),
            }
            self._write_state(iid, state)
            logger.info("coexist: loaded instance %s (reason=%s) at %s", iid,
                        reason, handle.base_url)
            self._event("instance_loaded", id=iid, reason=reason,
                        base_url=handle.base_url)
            return state

    # ------------------------------------------------- request-path hooks
    def ensure_loaded(self, iid: str) -> Optional[Dict[str, Any]]:
        """Lazy-load hook for a resolve path that holds the instance id.

        Returns the fresh state when lazy_load is on and the instance sits in
        the unloaded state; None when nothing applies (caller keeps its normal
        not-running error). Raises CoexistError only when a reload was
        attempted and failed."""
        if not self.cfg.lazy_load:
            return None
        st = self._read_state(iid)
        if not st or not st.get("unloaded"):
            return None  # never-unloaded stopped instance: legacy behavior
        return self.load_instance(iid, reason="lazy")

    def unloaded_guidance(self, iid: str) -> Optional[str]:
        """Clear message when the instance is idle/coexist-unloaded and lazy
        load is off; None otherwise."""
        st = self._read_state(iid)
        if st and st.get("unloaded") and not self.cfg.lazy_load:
            return (f"server instance {iid} was unloaded to free VRAM "
                    f"(reason={st.get('unload_reason', 'idle')}); load it "
                    "(POST /api/servers/"
                    f"{iid}/load or `sable instances load {iid}`) or set "
                    "serve.lazy_load: true to reload automatically")
        return None

    def reload_for_url(self, base_url: str) -> bool:
        """Connection-failure hook (agent runner): reload the unloaded
        instance whose recorded endpoint matches ``base_url``. True when a
        reload happened (caller should retry its request exactly once)."""
        if not self.cfg.lazy_load or self.home is None:
            return False
        target = _norm(base_url)
        try:
            from . import instances as srv_inst
            defs = srv_inst.load_instances(self.home)
        except Exception:  # noqa: BLE001
            return False
        for inst in defs:
            iid = str(inst.get("id") or "")
            if not iid:
                continue
            st = self._read_state(iid)
            if st and st.get("unloaded") and _norm(
                    st.get("base_url") or "") == target:
                try:
                    self.load_instance(iid, reason="lazy")
                    return True
                except CoexistError as e:
                    logger.error("coexist: lazy reload of %s failed: %s",
                                 iid, e)
                    return False
        return False

    # ---------------------------------------------------------- idle loop
    def step(self) -> None:
        """One idle-audit pass (pure enough to test directly): seed/refresh
        activity from the WP4 in-flight tallies and unload instances whose
        idle window elapsed. The background loop only adds sleep()."""
        if not self.cfg.idle_enabled or self.home is None:
            return
        from .watchdog import idle_seconds, inflight_count, note_activity

        for t in self.running_targets():
            url = t["base_url"]
            if inflight_count(url) > 0:
                # In flight => never a candidate. Deliberately NO stamp
                # refresh here: request STARTS already stamp activity, so the
                # idle window measures from the last start — a server busy
                # with one long request becomes idle once that request ends
                # (refreshing per tick would make it immortal).
                continue
            age = idle_seconds(url, clock=self.clock)
            if age is None:
                # First sighting of this endpoint: seed the window from now —
                # a freshly started server gets its full idle window before
                # the loop would ever consider it stale.
                note_activity(url, clock=self.clock)
                continue
            if age >= self.cfg.idle_unload_seconds:
                self.unload_instance(t["id"], reason="idle")

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                self.step()
            except Exception:  # noqa: BLE001 — the loop must survive anything
                logger.exception("coexist: step failed; continuing")
            self._stop.wait(self.cfg.sample_interval_seconds)

    def start_idle_loop(self) -> bool:
        """Start the background idle loop (no-op unless idle unload enabled)."""
        if not self.cfg.idle_enabled:
            return False
        if self._thread is not None and self._thread.is_alive():
            return True
        self._stop.clear()
        self._thread = threading.Thread(target=self.run, daemon=True,
                                        name="serve-idle-unload")
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------- handshake (WP5 #5)
    def _projected_free_if_unloaded(self, free_mib: int
                                    ) -> Tuple[int, List[Dict[str, Any]]]:
        """(projected free MiB, unloadable targets with estimates). Estimates
        come from the same Phase-6 heuristic the over-commit advisory uses; an
        instance whose footprint cannot be estimated is EXCLUDED from the
        projection (conservative — never claim freed VRAM we cannot name)."""
        from .instances import estimate_model_vram_mib

        projected = int(free_mib)
        targets: List[Dict[str, Any]] = []
        for t in self.running_targets():
            est = None
            if t.get("gguf_path"):
                with contextlib.suppress(Exception):
                    est = estimate_model_vram_mib(
                        str(t["gguf_path"]), t.get("context_length"))
            if est is None:
                continue
            projected += int(est)
            targets.append({**t, "est_mib": int(est)})
        return projected, targets

    def rescue_gate(self, job_id: str, required_mib: int) -> bool:
        """Scheduler handshake: called when the resource gate just FAILED.

        Returns True when an unload was performed and the caller should
        re-evaluate the gate; False (refuse exactly as today) in every other
        case — disabled handshake, no telemetry, gate failure not explained by
        VRAM, or a policy of ``refuse`` (which emits ``coexist_refused`` with
        the headroom numbers so the user can see what would have happened)."""
        if not self.cfg.handshake_enabled:
            return False
        free = self._free_vram()
        if free is None:
            return False  # no telemetry => nothing provable, keep old refusal
        required_mib = int(required_mib)
        if free >= required_mib:
            # The gate failed for some other reason (AC power…): unloading
            # served models would not help — refuse as today.
            return False
        projected, targets = self._projected_free_if_unloaded(free)
        if projected < required_mib:
            logger.info("coexist: job %s needs %d MiB; even unloading %s "
                        "(-> %d MiB projected) stays short — refusing",
                        job_id, required_mib,
                        [t["id"] for t in targets] or "nothing", projected)
            return False
        if self.cfg.coexist_policy != "unload":
            self._event("coexist_refused", job_id=job_id, free_mib=int(free),
                        required_mib=required_mib,
                        projected_free_mib=int(projected),
                        instances=[t["id"] for t in targets],
                        would_pass_if_unloaded=True)
            logger.info(
                "coexist: job %s WOULD be admitted if served instances "
                "(%s) were unloaded (%d -> %d MiB >= %d), but "
                "serve.coexist_policy=refuse — refusing (set it to 'unload' "
                "to opt in)", job_id, [t["id"] for t in targets], free,
                projected, required_mib)
            return False
        self._event("coexist_unload_triggered", job_id=job_id,
                    free_mib=int(free), required_mib=required_mib,
                    projected_free_mib=int(projected),
                    targets=[t["id"] for t in targets])
        logger.info("coexist: unloading %s for job %s (%d MiB free, need %d)",
                    [t["id"] for t in targets], job_id, free, required_mib)
        self.unload_all(reason="coexist")
        return self.wait_vram_freed(required_mib)

    def wait_vram_freed(self, required_mib: int) -> bool:
        """Poll the VRAM probe until >= required or the timeout elapses.

        The GPU releases a process's memory asynchronously after SIGTERM;
        without this wait the very next gate check could still see the old
        number and refuse. Returns whether the bar was met (the caller
        re-evaluates the real gate either way — an honest second refusal is
        better than a fake admission)."""
        deadline = self.clock() + self.cfg.vram_wait_timeout_seconds
        while True:
            free = self._free_vram()
            if free is not None and free >= int(required_mib):
                return True
            if self.clock() >= deadline:
                logger.warning(
                    "coexist: VRAM did not reach %d MiB within %.0fs after "
                    "unload (last reading: %s)", required_mib,
                    self.cfg.vram_wait_timeout_seconds, free)
                return False
            self.sleep(self.cfg.vram_poll_interval_seconds)


def _gpu_free_mib() -> Optional[int]:
    """Live free-VRAM probe via the scheduler's vendor tooling (the exact
    source the training gate uses — one truth per number)."""
    from ..training.scheduler import gpu_free_mib

    return gpu_free_mib()


# --------------------------------------------------------------- boot glue

def build(cfg: Any, home: Any, *, emit: Optional[EmitFn] = None
          ) -> Optional[Coexistence]:
    """Construct the coexistence engine IF any WP5 key is set.

    With no config changes every default is off and this returns None
    immediately — every call site (dashboard boot, scheduler wiring, agent
    runner hook) then behaves exactly as before WP5."""
    cc = coexist_config(cfg)
    if not cc.any_enabled:
        return None
    coex = Coexistence(cc, parent_cfg=cfg, home=home, emit=emit)
    logger.info(
        "coexistence armed (idle_unload=%.0fs lazy_load=%s "
        "min_free_vram_mib=%d policy=%s)", cc.idle_unload_seconds,
        cc.lazy_load, cc.min_free_vram_mib, cc.coexist_policy)
    return coex


#: Process-wide active engine (set by the dashboard boot / CLI entry points;
#: the agent-runner's connection-failure hook has no cfg/home of its own).
_ACTIVE: Optional[Coexistence] = None


def set_active(coex: Optional[Coexistence]) -> None:
    global _ACTIVE
    _ACTIVE = coex


def active() -> Optional[Coexistence]:
    return _ACTIVE

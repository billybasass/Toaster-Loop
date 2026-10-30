"""Agent discovery + delegation (feature-wave Phase 2; D-17 async inbox).

Lets one agent profile hand a subtask to another profile on the SAME project.
Response shapes (success / timeout-with-diagnostics / error payloads) follow
modeled on the delegation tool contract in spirit.

Execution model — D-17 (amends D-11): ``delegate_task`` DISPATCHES by default.
It registers a durable record (``core/delegation_store.py`` — SQLite, the
table doubles as the parent thread's inbox), submits the child to a bounded
process-wide dispatcher pool, and returns ``{status: "dispatched",
delegation_id}`` immediately. The legacy synchronous wait survives behind an
explicit ``wait=true`` (config ``delegation.async_default: false`` restores it
globally): the child runs on the shared dispatcher pool while the parent
polls up to a bounded timeout, then returns the same timeout-with-diagnostics
payload as before.

Durable registry: every lifecycle transition writes through to the store
(queued -> running via the atomic claim; terminal set writes result + error +
``inbox_state='pending'`` for async delegations with a parent thread — Phase 2
consumes those). The in-process ``_delegations`` OrderedDict stays as the hot
cache (heartbeats + ``check_delegation`` fast path); on cache miss reads fall
back to the store, so ``check_delegation`` survives restarts (audit §4 item
3). A liveness watchdog enforces ``delegation.liveness_timeout_seconds`` on
queue dwell + heartbeat staleness (wedge-detect -> cooperative cancel ->
``wedged``), and startup reconcile marks crash-orphaned rows ``interrupted``.

Fault posture mirrors the rest of core: tools never raise into the loop;
errors come back as JSON payloads with an ``error`` field (which
core/failure_types.classify_tool_result correctly scores as a failure).

Execution reuse: the child runs through ``core.agent_runner.create_runner`` —
the target profile's own system prompt / tool subset / workspace override —
so guardrails, compaction, spill and trace recording behave exactly like a
normal run. The child gets its own trace whose header metadata links
``delegation_id``, ``parent_run_id`` and the project (the parent's trace id
does not exist yet while the parent is mid-run — traces are recorded
post-run — so the linkage key is the parent run id; see D-11).

Lineage: when the tool executes with thread context (dashboard chat passes
the current thread id), an entry {agent, delegation_id, status, created} is
appended to the thread JSON's ``delegations`` list at <home>/chat/*.json —
the shape Phase 1's explorer-tree lineage chips already tolerate. Without
thread context (CLI) lineage is skipped and everything else still works.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from . import delegation_store
from .failure_types import FAILURE_CANCELLED

MAX_RESULT_CHARS = 8_000        # child answer kept in the payload
# Audit §6 floor: now that timeout/cancel is cooperative (§2), a mistyped tiny
# timeout abandons AND cancel-signals a perfectly healthy child, so anything
# below 15s is clamped up (with a "warning" field in the tool payload — see
# _resolve_timeout). Tests needing sub-second waits monkeypatch this constant.
MIN_TIMEOUT_S = 15.0
MAX_TIMEOUT_S = 3_600.0
DELEGATION_BLOCKED_TOOLS_NOTE = "nested delegation is not allowed (depth limit 1)"

#: Heartbeat store-write throttle (§3.1): at most one SQLite write per this
#: many seconds per delegation — the loop-boundary tick can fire far more
#: often than the watchdog needs.
TOUCH_THROTTLE_S = 2.0

_THREAD_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _age_seconds(ts: Optional[str]) -> Optional[float]:
    """Seconds elapsed since an ISO-8601 timestamp (None when unparseable —
    tolerant reads, same ethos as the store)."""
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - dt).total_seconds()


# ---------------------------------------------------------------------------
# In-process registries (one dashboard/CLI process owns its own runs)
# ---------------------------------------------------------------------------

_registry_lock = threading.Lock()
#: run_id -> {profile, project, started} for every in-flight runner run
#: (populated by SableAgentClient.run when a delegation context exists).
_active_runs: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
#: delegation_id -> record dict, bounded FIFO (check_delegation hot cache;
#: the durable copy lives in delegation_store).
_delegations: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_DELEGATIONS_CAP = 500
#: Audit §6: lightweight terminal summaries of records pruned by the FIFO cap
#: (chosen over a bare "was pruned" error message because existing structures
#: make it trivial: one deque + the fields check_delegation already returns).
#: Lets check_delegation answer with the real terminal state plus a
#: "(pruned)" marker instead of a confusing "unknown delegation id". Bounded,
#: in-process only — the durable history stays the Traces store.
_PRUNED_RING_CAP = 50
_pruned_summaries: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
#: parent wait-loop poll slice (s): cheap heartbeat granularity while the
#: child is still registered active in ``_active_runs``.
_HEARTBEAT_POLL_S = 0.25
#: delegation_id -> execution closure submitted to the dispatcher but not yet
#: claimed (see _pump). The DB `queued` status is the real queue; this map
#: only carries the per-record context (profile/task/ctx) the worker needs.
_pending_exec: "Dict[str, Callable[[], None]]" = {}

# ---------------------------------------------------------------------------
# Wake hooks + watcher state (D-17 Phase 2, spec §4.1)
# ---------------------------------------------------------------------------

#: thread_id -> callable(thread_id) -> {"started": bool, ...}. The dashboard's
#: chat API registers its ``start_inbox_wake_run`` lazily per active thread
#: (on the first run spawned for that thread). With no hook registered (CLI,
#: or dashboard not running) inbox entries simply stay pending until the next
#: wake opportunity or a real user message drains them.
_wake_hooks: "Dict[str, Callable[[str], Dict[str, Any]]]" = {}
_wake_hooks_lock = threading.Lock()
#: Ping-pong guard (spec §10 risk #2): consecutive watcher-delivered wakes per
#: thread with no intervening *user* run. A wake run that itself delegates
#: every turn could otherwise ping-pong runs indefinitely; the streak cap
#: bounds the chain (depth-1 + queue_max bound each individual hop). Reset by
#: any user-initiated run on the thread and by explicit delivery success after
#: a quiet period.
_wake_streaks: "Dict[str, int]" = {}
WAKE_STREAK_MAX = 8
#: Set by nudge_delivery() (a chat run just ended) so the watcher loop wakes
#: from its retry sleep immediately instead of idling out wake_retry_seconds.
_wake_nudge: Optional[threading.Event] = None


def register_wake_hook(thread_id: str, hook: Callable[[str], Dict[str, Any]]
                       ) -> None:
    """Register the per-thread wake deliverer (dashboard chat API)."""
    if not thread_id:
        return
    with _wake_hooks_lock:
        _wake_hooks[thread_id] = hook


def unregister_wake_hook(thread_id: str) -> None:
    with _wake_hooks_lock:
        _wake_hooks.pop(thread_id, None)
        _wake_streaks.pop(thread_id, None)


def get_wake_hook(thread_id: str) -> Optional[Callable[[str], Dict[str, Any]]]:
    with _wake_hooks_lock:
        return _wake_hooks.get(thread_id)


def reset_wake_streak(thread_id: str) -> None:
    """A real (user-initiated) run on the thread clears its wake streak — the
    ping-pong guard only counts watcher-driven wakes back-to-back."""
    with _wake_hooks_lock:
        _wake_streaks.pop(thread_id, None)


def nudge_delivery() -> None:
    """Ask the watcher loop to run a delivery pass now (called when a chat
    run ends and frees its thread's single-flight slot, spec §4.3). Cheap and
    safe from any thread; a no-op before/after the dispatcher exists."""
    ev = _wake_nudge
    if ev is not None:
        ev.set()


def register_run(profile: Optional[str], project: Optional[str] = None,
                 run_id: Optional[str] = None) -> str:
    """Mark a runner execution in-flight; returns the registry key."""
    key = run_id or uuid.uuid4().hex[:12]
    with _registry_lock:
        _active_runs[key] = {"profile": profile, "project": project,
                             "started": _utcnow()}
        # Heartbeat: a child runner registers under its delegation id
        # (child_ctx["run_id"] == delegation_id), so registration is the
        # first activity signal for that delegation record.
        rec = _delegations.get(key)
        if rec is not None:
            rec["last_activity_ts"] = _utcnow()
    return key


def unregister_run(key: str) -> None:
    with _registry_lock:
        _active_runs.pop(key, None)
        # Heartbeat: the child runner leaving the active set is its final
        # liveness signal (the terminal record write follows immediately).
        rec = _delegations.get(key)
        if rec is not None and rec.get("status") in ("running", "timed_out"):
            rec["last_activity_ts"] = _utcnow()


def active_runs() -> List[Dict[str, Any]]:
    """Snapshot of in-flight runs (list of {profile, project, started})."""
    with _registry_lock:
        return [dict(r) for r in _active_runs.values()]


def running_count_for(profile: str) -> int:
    with _registry_lock:
        return sum(1 for r in _active_runs.values() if r.get("profile") == profile)


def _store_delegation(record: Dict[str, Any]) -> None:
    with _registry_lock:
        _delegations[record["delegation_id"]] = record
        while len(_delegations) > _DELEGATIONS_CAP:
            old_id, old = next(iter(_delegations.items()))
            _delegations.pop(old_id)
            # Audit §6: keep a cheap terminal summary of the evicted record so
            # check_delegation can stay honest after eviction (see ring note).
            if old.get("_settled"):
                _pruned_summaries[old_id] = {
                    k: old.get(k) for k in
                    ("agent", "status", "result", "child_trace_id", "error",
                     "created", "last_activity_ts", "timeout_warning")}
                while len(_pruned_summaries) > _PRUNED_RING_CAP:
                    _pruned_summaries.pop(next(iter(_pruned_summaries)))


def get_delegation(delegation_id: str) -> Optional[Dict[str, Any]]:
    with _registry_lock:
        rec = _delegations.get(delegation_id)
        return dict(rec) if rec else None


def get_pruned_summary(delegation_id: str) -> Optional[Dict[str, Any]]:
    """Terminal summary of a record evicted by the FIFO cap (audit §6), or
    None when this process never saw the id (or it is still in the registry)."""
    with _registry_lock:
        s = _pruned_summaries.get(delegation_id)
        return dict(s) if s else None


def reset_state() -> None:
    """Clear registries + dispatcher (tests)."""
    try:
        shutdown_dispatcher(grace_s=0.1)
    except Exception:  # noqa: BLE001 — teardown must never raise
        pass
    with _registry_lock:
        _active_runs.clear()
        _delegations.clear()
        _pruned_summaries.clear()
        _pending_exec.clear()
    with _wake_hooks_lock:
        _wake_hooks.clear()
        _wake_streaks.clear()


# ---------------------------------------------------------------------------
# Durable write-through helpers (cache record -> store row)
# ---------------------------------------------------------------------------

def _row_from_record(record: Dict[str, Any], cfg: Any) -> Dict[str, Any]:
    """Map a hot-cache record onto the store's column vocabulary."""
    return {
        "delegation_id": record["delegation_id"],
        "status": record.get("status", "queued"),
        "agent": record.get("agent"),
        "task_preview": (record.get("task") or "")[:300],
        "project": record.get("project"),
        "thread": record.get("thread"),
        "parent_run_id": record.get("parent_run_id"),
        "depth": int(record.get("depth", 1)),
        "created_ts": record.get("created") or _utcnow(),
        "started_ts": record.get("started_ts"),
        "settled_ts": record.get("settled_ts"),
        "last_activity_ts": record.get("last_activity_ts"),
        "liveness_timeout": float(record.get(
            "liveness_timeout",
            getattr(cfg, "delegation_liveness_timeout_seconds", 600.0))),
        "result": record.get("result"),
        "error": record.get("error"),
        "child_trace_id": record.get("child_trace_id"),
        "plan_task": record.get("plan_task"),
        "wait_mode": 1 if record.get("wait_mode") else 0,
        "inbox_state": record.get("inbox_state", "none"),
    }


def _store_write(record: Dict[str, Any], **fields: Any) -> None:
    """Best-effort durable write-through for a cache record (never raises —
    the in-process state machine must not break on store hiccups; the traces
    store remains the ultimate history)."""
    cfg = record.get("_cfg_ref")
    try:
        delegation_store.update_delegation(record["delegation_id"], cfg=cfg,
                                           **fields)
    except Exception:  # noqa: BLE001
        pass


def touch(delegation_id: str) -> None:
    """Liveness heartbeat tick (§3.1): stamp ``last_activity_ts`` in the hot
    cache and (throttled to >= 1 per TOUCH_THROTTLE_S) in the store. Called
    by the child runner's ``_heartbeat`` hook at loop boundaries and after
    each tool call; safe to call for unknown ids (no-op)."""
    did = str(delegation_id or "")
    if not did:
        return
    now_m = time.monotonic()
    with _registry_lock:
        rec = _delegations.get(did)
        if rec is None:
            return
        last = rec.get("_last_touch_mono")
        if last is not None and now_m - last < TOUCH_THROTTLE_S:
            rec["last_activity_ts"] = _utcnow()  # cache tick is free
            return
        rec["_last_touch_mono"] = now_m
        rec["last_activity_ts"] = _utcnow()
    _store_write(rec, last_activity_ts=_utcnow())


# ---------------------------------------------------------------------------
# Profiles / projects lookup (config + <home>/projects/*.json)
# ---------------------------------------------------------------------------

def _profiles(cfg: Any) -> List[Dict[str, Any]]:
    profs = getattr(cfg, "profiles", None) or []
    return [p for p in profs if isinstance(p, dict)]


def find_profile(cfg: Any, name: str) -> Optional[Dict[str, Any]]:
    for p in _profiles(cfg):
        if p.get("name") == name:
            return p
    return None


def profile_description(prof: Dict[str, Any]) -> str:
    desc = str(prof.get("description") or "").strip()
    if desc:
        return desc.splitlines()[0][:160]
    prompt = str(prof.get("system_prompt") or "").strip()
    return prompt.splitlines()[0][:160] if prompt else "(no description)"


def load_project(cfg: Any, project_id: str) -> Optional[Dict[str, Any]]:
    """Read <home>/projects/<id>.json (same file layout as the dashboard)."""
    if not project_id or not _THREAD_ID_RE.match(project_id):
        return None
    from ..store.paths import resolve_sable_home

    p = resolve_sable_home(cfg) / "projects" / f"{project_id}.json"
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def project_agent_names(cfg: Any, project_id: Optional[str]) -> List[str]:
    """Agents attached to the active project; all profiles when no project."""
    if project_id:
        proj = load_project(cfg, project_id) or {}
        agents = proj.get("agents")
        return [str(a) for a in agents] if isinstance(agents, list) else []
    return [str(p.get("name")) for p in _profiles(cfg) if p.get("name")]


# ---------------------------------------------------------------------------
# Thread lineage (<home>/chat/<thread>.json "delegations" list)
# ---------------------------------------------------------------------------

def _thread_path(cfg: Any, thread_id: str) -> Optional[Path]:
    if not thread_id or not _THREAD_ID_RE.match(thread_id):
        return None
    from ..store.paths import resolve_sable_home

    return resolve_sable_home(cfg) / "chat" / f"{thread_id}.json"


def thread_record_delegation(cfg: Any, thread_id: Optional[str],
                             record: Dict[str, Any]) -> None:
    """Append/update the delegation entry on the thread JSON. Best-effort:
    lineage failures never break a delegation (or the agent loop)."""
    p = _thread_path(cfg, thread_id or "")
    if p is None:
        return  # no/invalid thread context (e.g. CLI) -> lineage skipped
    try:
        thread: Dict[str, Any] = {}
        if p.is_file():
            try:
                loaded = json.loads(p.read_text())
                if isinstance(loaded, dict):
                    thread = loaded
            except (OSError, json.JSONDecodeError):
                thread = {}
        entry = {
            "agent": record.get("agent"),
            "delegation_id": record.get("delegation_id"),
            "status": record.get("status"),
            "created": record.get("created"),
        }
        if record.get("child_trace_id"):
            entry["trace_id"] = record["child_trace_id"]
        delegs = thread.get("delegations")
        if not isinstance(delegs, list):
            delegs = []
        for i, d in enumerate(delegs):
            if isinstance(d, dict) and d.get("delegation_id") == entry["delegation_id"]:
                delegs[i] = entry
                break
        else:
            delegs.append(entry)
        thread.setdefault("id", p.stem)
        thread.setdefault("created", record.get("created"))
        thread["delegations"] = delegs
        thread["updated"] = _utcnow()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(thread, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        os.replace(tmp, p)
    except Exception:  # lineage is observability, never fatal
        pass


# ---------------------------------------------------------------------------
# Payload helpers (reference-shaped responses; JSON strings for the tool result)
# ---------------------------------------------------------------------------

def _payload(**fields: Any) -> str:
    return json.dumps(fields, ensure_ascii=False, default=str)


def _error_payload(message: str, **extra: Any) -> str:
    return _payload(status="failed", error=message, **extra)


# ---------------------------------------------------------------------------
# Dispatcher — bounded supervised pool + liveness watchdog (D-17 §3)
# ---------------------------------------------------------------------------

_disp_lock = threading.Lock()
_executor: Optional[ThreadPoolExecutor] = None
#: cfg handed to the most recent init/ensure — used for store path resolution
#: and watchdog knobs when a caller has no ctx (watchdog, shutdown).
_dispatcher_cfg: Any = None
_accepting = True
_watchdog_stop: Optional[threading.Event] = None
_watchdog_thread: Optional[threading.Thread] = None


def _default_cfg() -> Any:
    """Best-effort cfg for store path resolution outside a tool call."""
    if _dispatcher_cfg is not None:
        return _dispatcher_cfg
    try:  # last resort: env/home-resolved config (cheap, no file requirement)
        from .config import load_config

        return load_config()
    except Exception:  # noqa: BLE001 — path resolution falls back to env too
        return None


#: The watcher loop's sleep cadence is normally
#: ``delegation.wake_retry_seconds``; tests monkeypatch this module attribute
#: to keep delivery deterministic without sleeping on the real interval.
_override_wake_retry_s: Optional[float] = None


def init_dispatcher(cfg: Any) -> None:
    """Create the bounded worker pool + liveness watchdog (idempotent).
    Called at dashboard/CLI startup; also lazily on first dispatch. The same
    loop also drives the D-17 Phase 2 inbox wake watcher (spec §4.1: "watcher
    loop (same watchdog thread is fine)"), sleeping
    ``delegation.wake_retry_seconds`` between passes and waking early via
    ``nudge_delivery()`` when a chat run ends."""
    global _executor, _dispatcher_cfg, _accepting, _watchdog_stop, \
        _watchdog_thread, _wake_nudge
    with _disp_lock:
        _dispatcher_cfg = cfg
        _accepting = True
        if _wake_nudge is None:
            _wake_nudge = threading.Event()
        if _executor is None:
            n = int(getattr(cfg, "delegation_max_concurrent", 2) or 2)
            _executor = ThreadPoolExecutor(
                max_workers=max(1, n), thread_name_prefix="deleg-worker")
        if _watchdog_thread is None or not _watchdog_thread.is_alive():
            _watchdog_stop = threading.Event()
            interval = float(getattr(cfg, "delegation_liveness_check_interval_seconds", 5.0) or 5.0)
            wake_retry = float(getattr(cfg, "delegation_wake_retry_seconds",
                                       2.0) or 2.0)
            stop = _watchdog_stop
            nudge = _wake_nudge

            def _wd_loop() -> None:
                # One loop drives both duties (spec §4.1): the wake pass runs
                # at the (fast, wake_retry_seconds) cadence; the liveness scan
                # keeps its own slower interval. The shared nudge Event lets an
                # ended chat run (nudge_delivery) — or shutdown — trigger an
                # immediate iteration instead of idling out the retry window.
                last_liveness = time.monotonic()
                while not stop.is_set():
                    if nudge.wait(timeout=max(0.2,
                                              _override_wake_retry_s
                                              or wake_retry)):
                        nudge.clear()
                    if stop.is_set():
                        break
                    try:
                        run_wake_pass(_dispatcher_cfg)
                    except Exception:  # noqa: BLE001 — watcher never dies
                        pass
                    now = time.monotonic()
                    if now - last_liveness >= max(1.0, interval):
                        last_liveness = now
                        try:
                            _liveness_scan_once(_dispatcher_cfg)
                        except Exception:  # noqa: BLE001
                            pass

            _watchdog_thread = threading.Thread(
                target=_wd_loop, daemon=True, name="deleg-watchdog")
            _watchdog_thread.start()


def _ensure_dispatcher(cfg: Any) -> None:
    # NOTE: no lock here — init_dispatcher takes _disp_lock itself (the lock
    # is non-reentrant; a double-acquire from here would self-deadlock).
    global _dispatcher_cfg
    if _executor is None or not _accepting:
        init_dispatcher(cfg)
    elif cfg is not None:
        with _disp_lock:
            _dispatcher_cfg = cfg


def shutdown_dispatcher(grace_s: float = 5.0) -> None:
    """Stop accepting, cooperatively cancel every live child (they settle
    ``cancelled``), wait up to ``grace_s``, then mark stragglers
    ``interrupted`` in the store. Called from server shutdown right after
    ``shutdown_chat_runs()`` — runners before backends."""
    global _executor, _accepting, _watchdog_stop, _watchdog_thread
    with _disp_lock:
        _accepting = False
        ex = _executor
        stop_ev, wd = _watchdog_stop, _watchdog_thread
        _executor = None
        _watchdog_stop = None
        _watchdog_thread = None
    if stop_ev is not None:
        stop_ev.set()
    nudge = _wake_nudge
    if nudge is not None:
        nudge.set()  # wake the loop so it observes the stop event promptly
    # Cooperative cancel of everything still live in this process.
    with _registry_lock:
        live = [(did, r) for did, r in _delegations.items()
                if not r.get("_settled")]
        for _, r in live:
            ev = r.get("cancel_event")
            if isinstance(ev, threading.Event):
                ev.set()
    deadline = time.monotonic() + max(0.0, float(grace_s))
    while time.monotonic() < deadline:
        with _registry_lock:
            unsettled = [did for did, r in _delegations.items()
                         if not r.get("_settled")]
        if not unsettled:
            break
        time.sleep(0.05)
    # Stragglers: mark interrupted (cache + durable store).
    with _registry_lock:
        stragglers = [(did, r) for did, r in _delegations.items()
                      if not r.get("_settled")]
    cfg = _default_cfg()
    for did, rec in stragglers:
        with _registry_lock:
            rec["status"] = "interrupted"
            rec["error"] = ("dispatcher shut down while the child was in "
                            "flight")
            rec["_settled"] = True
        try:
            delegation_store.update_delegation(
                did, cfg=cfg, status="interrupted", settled_ts=_utcnow(),
                error="dispatcher shut down while the child was in flight",
                inbox_state=("pending" if rec.get("thread")
                             and not rec.get("wait_mode") else "none"))
        except Exception:  # noqa: BLE001
            pass
    if ex is not None:
        ex.shutdown(wait=False)
    if wd is not None and wd.is_alive():
        wd.join(timeout=2.0)


def _queued_count(cfg: Any) -> int:
    return len(delegation_store.list_delegations(
        statuses=("queued",), limit=10_000, cfg=cfg))


def _pump() -> None:
    """Dispatcher worker body: claim the oldest queued row (atomic FIFO —
    the DB is the queue) and run its registered execution closure. A pump
    that finds nothing to claim was either cancelled before start or had its
    row claimed by a sibling — either way, no work, no error."""
    cfg = _default_cfg()
    try:
        row = delegation_store.claim_next_queued(cfg=cfg)
    except Exception:  # noqa: BLE001 — store hiccup: retry on next dispatch
        return
    if row is None:
        return
    did = row["delegation_id"]
    with _registry_lock:
        run_fn = _pending_exec.pop(did, None)
        rec = _delegations.get(did)
        if rec is not None and not rec.get("_settled"):
            rec["status"] = "running"
            rec["started_ts"] = row.get("started_ts") or _utcnow()
            rec["last_activity_ts"] = _utcnow()
    if run_fn is None:
        # Row was queued but its execution context vanished — either it was
        # cancelled before start (cache already settled; mirror that terminal
        # state) or the process is shutting down (mark interrupted). Either
        # way an honest terminal state beats a forever-running phantom.
        with _registry_lock:
            rec2 = _delegations.get(did)
            settled2 = bool(rec2 and rec2.get("_settled"))
            final_status = rec2.get("status") if rec2 else None
            inbox2 = ("pending" if rec2 and rec2.get("thread")
                      and not rec2.get("wait_mode") else "none")
        try:
            if settled2 and final_status:
                delegation_store.update_delegation(
                    did, cfg=cfg, status=final_status, settled_ts=_utcnow(),
                    inbox_state=inbox2, expect_status="running")
            else:
                delegation_store.update_delegation(
                    did, cfg=cfg, status="interrupted", settled_ts=_utcnow(),
                    error="dispatcher lost the execution context",
                    expect_status="running")
        except Exception:  # noqa: BLE001
            pass
        return
    try:
        run_fn()
    except Exception:  # noqa: BLE001 — worker faults settle inside run_fn
        pass


# ---------------------------------------------------------------------------
# Liveness watchdog (D-17 §3): queue starvation + heartbeat staleness
# ---------------------------------------------------------------------------

def _settle_wedged(row: Dict[str, Any], error: str, cfg: Any) -> bool:
    """CAS a non-terminal row to ``wedged`` (+ inbox pending when it has a
    thread and was dispatched async). Returns whether we won the race."""
    did = row["delegation_id"]
    inbox = ("pending" if row.get("thread") and not row.get("wait_mode")
             else "none")
    won = False
    try:
        won = delegation_store.update_delegation(
            did, cfg=cfg, status="wedged", settled_ts=_utcnow(), error=error,
            last_activity_ts=_utcnow(), inbox_state=inbox,
            expect_status=row.get("status"))
    except Exception:  # noqa: BLE001
        return False
    if not won:
        return False
    with _registry_lock:
        rec = _delegations.get(did)
        if rec is not None and not rec.get("_settled"):
            rec["status"] = "wedged"
            rec["error"] = error
            rec["last_activity_ts"] = _utcnow()
            rec["_settled"] = True
    return True


def _liveness_scan_once(cfg: Any = None) -> int:
    """One watchdog pass over the non-terminal rows (spec §3):

    - ``queued`` older than the liveness window -> ``wedged`` (dispatcher
      starvation is itself a wedge signal);
    - ``running`` with a stale heartbeat -> set the cooperative cancel event
      and flip to ``cancelling`` (in-process records only; cross-process
      orphans belong to startup reconcile);
    - ``cancelling`` that never observed the cancel within one more full
      window -> ``wedged``.

    Returns the number of rows touched. Runs from the watchdog thread and is
    directly callable by tests (no sleeping required)."""
    cfg = cfg if cfg is not None else _default_cfg()
    liveness = float(getattr(cfg, "delegation_liveness_timeout_seconds", 600.0))
    touched = 0
    try:
        rows = delegation_store.list_delegations(
            statuses=delegation_store.NON_TERMINAL_STATUSES, limit=10_000,
            cfg=cfg)
    except Exception:  # noqa: BLE001 — scan is best-effort per tick
        return 0
    for row in rows:
        status = row.get("status")
        if status == "queued":
            age = _age_seconds(row.get("created_ts"))
            if age is not None and age > liveness:
                if _settle_wedged(
                        row, "child never started within the liveness window",
                        cfg):
                    touched += 1
        elif status == "running":
            age = _age_seconds(row.get("last_activity_ts")
                               or row.get("started_ts"))
            if age is None or age <= liveness:
                continue
            did = row["delegation_id"]
            with _registry_lock:
                rec = _delegations.get(did)
            if rec is None or rec.get("_settled"):
                # Cross-process orphan or already settling: startup reconcile
                # owns the first, the worker settles the second.
                continue
            ev = rec.get("cancel_event")
            if isinstance(ev, threading.Event):
                ev.set()
            err = f"liveness timeout after {liveness:g}s of silence"
            try:
                moved = delegation_store.update_delegation(
                    did, cfg=cfg, status="cancelling", error=err,
                    expect_status="running")
            except Exception:  # noqa: BLE001
                continue
            if moved:
                with _registry_lock:
                    if not rec.get("_settled"):
                        rec["status"] = "cancelling"
                        rec["error"] = err
                touched += 1
        elif status == "cancelling":
            # One full extra window after the cancel was requested (the row
            # was already stale by `liveness` when we marked it, so total
            # staleness > 2 * liveness means the child never observed it).
            age = _age_seconds(row.get("last_activity_ts"))
            if age is not None and age > 2 * liveness:
                if _settle_wedged(
                        row,
                        f"never observed cancel after liveness timeout "
                        f"({liveness:g}s)", cfg):
                    touched += 1
    return touched


# ---------------------------------------------------------------------------
# Inbox wake watcher (D-17 Phase 2, spec §4.1/§4.4)
# ---------------------------------------------------------------------------

_WAKE_CONTINUE_NOTE = (
    "Continue the user's task using these results. If more work was waiting "
    "on them, proceed now; otherwise report back to the user.")


def format_inbox_message(rows: List[Dict[str, Any]]) -> str:
    """Synthetic user-role wake turn (spec §4.4): one bracket-tagged block
    listing every delivered entry oldest-first. User-role (not system) so
    every chat template handles it; the same header is used when a real user
    message drains the inbox concurrently, so traces stay interpretable."""
    n = len(rows)
    lines = [f"[Delegation results — {n} "
             f"{'entry' if n == 1 else 'entries'} settled since your last "
             "turn]", ""]
    for r in rows:
        did = r.get("delegation_id")
        agent = r.get("agent")
        status = r.get("status")
        bits = [f"- delegation_id={did}", f"agent={agent}",
                f"status={status}"]
        took_bits = []
        try:
            a_start = _age_seconds(r.get("started_ts"))
            a_settle = _age_seconds(r.get("settled_ts"))
            if a_start is not None and a_settle is not None \
                    and a_start >= a_settle:
                took_bits.append(f"took {a_start - a_settle:.0f}s")
        except Exception:  # noqa: BLE001 — cosmetic field only
            pass
        if r.get("child_trace_id"):
            took_bits.append(f"trace {r['child_trace_id']}")
        if took_bits:
            bits[-1] += f" ({', '.join(took_bits)})"
        lines.append(" ".join(bits))
        body = None
        if status == "completed" and r.get("result"):
            body = str(r["result"])[:MAX_RESULT_CHARS]
        elif r.get("error"):
            body = f"error: {r['error']}"
        if body:
            lines.append(f"  {body}")
    lines.append("")
    lines.append(_WAKE_CONTINUE_NOTE)
    return "\n".join(lines)


def try_deliver(thread_id: str, cfg: Any = None) -> Dict[str, Any]:
    """Attempt one inbox delivery for a thread via its registered wake hook.

    Returns the hook's result dict (``{"started": bool, ...}``). No hook (CLI,
    dashboard not running for this thread yet) -> entries stay pending. The
    ping-pong guard (§10 risk #2): at most ``WAKE_STREAK_MAX`` consecutive
    watcher-driven wakes per thread with no intervening user run — combined
    with the depth-1 cap and queue_max, a pathological parent that delegates
    every turn cannot chain runs unbounded. A real user message on the thread
    (or its drain) resets the streak."""
    hook = get_wake_hook(thread_id)
    if hook is None:
        return {"started": False, "no_hook": True}
    with _wake_hooks_lock:
        streak = _wake_streaks.get(thread_id, 0)
        if streak >= WAKE_STREAK_MAX:
            return {"started": False, "suppressed": "wake_streak",
                    "streak": streak}
    try:
        res = hook(thread_id)
    except Exception as e:  # noqa: BLE001 — a bad hook never kills the watcher
        return {"started": False, "error": f"{type(e).__name__}: {e}"}
    if isinstance(res, dict) and res.get("started"):
        with _wake_hooks_lock:
            _wake_streaks[thread_id] = streak + 1
        return res
    return res if isinstance(res, dict) else {"started": False}


def run_wake_pass(cfg: Any = None) -> int:
    """One watcher pass: for every thread with pending inbox entries and a
    registered wake hook, attempt delivery. Busy threads (single-flight) stay
    pending and are retried on the next tick (``delegation.wake_retry_seconds``
    default 2.0), or drained by the next real user message. Returns the number
    of wake runs started. Directly callable by tests — no sleeping required."""
    cfg = cfg if cfg is not None else _default_cfg()
    try:
        threads = delegation_store.pending_inbox_threads(cfg=cfg)
    except Exception:  # noqa: BLE001 — pass is best-effort per tick
        return 0
    started = 0
    for tid in threads:
        if try_deliver(tid, cfg=cfg).get("started"):
            started += 1
    return started


# ---------------------------------------------------------------------------
# delegate_task execution
# ---------------------------------------------------------------------------

def _resolve_timeout(cfg: Any, timeout_seconds: Any) -> "tuple[float, Optional[str]]":
    """Clamp the requested wait into [MIN_TIMEOUT_S, MAX_TIMEOUT_S].

    Returns (timeout, warning). Audit §6: with cooperative cancel landed (§2),
    a mistyped tiny timeout no longer just returns early — it abandons AND
    cancel-signals a healthy child — so anything under MIN_TIMEOUT_S is clamped
    up and the caller gets an explicit warning for the tool payload instead of
    silently honoring the footgun. Kept in-tool (not config load) because the
    floor is about *this call's* wait, per-call overrides included; mirrors the
    existing clamp-and-note pattern used elsewhere in core."""
    raw = timeout_seconds
    if raw is None:
        raw = getattr(cfg, "delegation_timeout_seconds", 300)
    warning: Optional[str] = None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        val = 300.0
    clamped = max(MIN_TIMEOUT_S, min(val, MAX_TIMEOUT_S))
    if clamped != val:
        warning = (f"timeout {val:g}s clamped to minimum {MIN_TIMEOUT_S:g}s"
                   if val < MIN_TIMEOUT_S else
                   f"timeout {val:g}s clamped to maximum {MAX_TIMEOUT_S:g}s")
    return clamped, warning


def _resolve_peer_profile(cfg: Any, agent: str) -> Optional[Dict[str, Any]]:
    """Profile dict for a <agent>@<host> peer agent, or None. Lazy, guarded
    import: dashboard.peers is pure stdlib+httpx state over peers.json under
    SABLE_HOME (sanctioned core->dashboard-state seam; standalone installs
    without the dashboard extra keep working)."""
    try:
        from ..dashboard import peers as peers_mod
    except ImportError:
        return None
    from ..store.paths import resolve_sable_home

    try:
        return peers_mod.resolve_remote_agent(resolve_sable_home(cfg), agent)
    except Exception:
        return None


def _run_child(cfg: Any, prof: Dict[str, Any], task: str, ctx: Dict[str, Any],
               delegation_id: str, out: Dict[str, Any],
               plan_task: Optional[str] = None,
               cancel_event: Optional[threading.Event] = None) -> None:
    """Execute the child run in-place (called on the dispatcher worker).
    Reuses create_runner so guardrails/compaction/recording are identical to a
    normal run. Results land in `out` (the caller's result box); the worker
    then settles the registry record from it, so a timed-out parent that
    stopped waiting still sees the child's terminal state land in the record.

    ``plan_task`` (WP12): task id whose prompt text was already resolved by
    execute_delegation; it stamps the child trace (additive plan_task field)
    so the traces panel links child runs to graph nodes.

    ``cancel_event`` (audit §2 bug B): the delegation's cooperative cancel
    flag, forwarded verbatim into ``create_runner(cancel_event=...)`` — the
    SAME mechanism h_chat_cancel uses for dashboard runs. The runner polls it
    at loop boundaries and returns with failure_reason="cancelled".

    Remote peer profiles (D-16, Option A): the child LOOP stays local; only
    inference crosses the wire. base_url/api_key/model are overridden to the
    peer's /remote/v1 gateway (mirrors chat.py's remote branch), and a
    multi-instance peer is pinned via the X-Sable-Instance header."""
    from .agent_runner import create_runner
    from ..store.recorder import make_recorder

    base_url = ctx.get("base_url") or ""
    model = prof.get("model_name") or ctx.get("model") or ""
    api_key = ctx.get("api_key") if "api_key" in ctx \
        else getattr(cfg, "attach_api_key", None)
    extra_headers: Optional[Dict[str, str]] = None
    if prof.get("remote"):
        # Remote profiles ignore the parent-inherited endpoint entirely.
        base_url = str(prof.get("peer_url") or "").rstrip("/") + "/remote/v1"
        api_key = prof.get("peer_token") or None
        model = str(prof.get("model_name") or "model")
        inst = str(prof.get("remote_instance") or "").strip()
        if inst:
            extra_headers = {"X-Sable-Instance": inst}
    project_id = ctx.get("project")
    proj = load_project(cfg, project_id) if project_id else None

    child_ctx: Dict[str, Any] = {
        "cfg": cfg, "base_url": base_url, "model": model,
        "profile": prof.get("name"), "project": project_id,
        "thread": None,                 # child never writes thread lineage itself
        "run_id": delegation_id,
        "depth": int(ctx.get("depth", 0)) + 1,
    }
    metadata = {
        "delegation_id": delegation_id,
        # Parent trace does not exist while the parent runs (recorded
        # post-run) — link by run id instead; see DECISIONS D-11.
        "parent_run_id": ctx.get("run_id"),
        "parent_trace_id": ctx.get("trace_id"),
        "delegated_from": ctx.get("profile") or "default",
    }
    # Easy training (same rule as dashboard chat): when the loop is enabled
    # and covers THIS child profile, stamp its name into traces.agent_profile
    # so curation / readiness scope to the child agent's slice. Otherwise the
    # recorder keeps the backend name (legacy behavior).
    from ..training.easy import EasyTrainingConfig

    ec = EasyTrainingConfig.from_yaml(
        getattr(cfg, "easy_training", None) or {})
    prof_name = str(prof.get("name") or "")
    rec_profile = (prof_name
                   if ec.enabled and prof_name
                   and prof_name in ec.all_profiles else None)
    recorder = make_recorder(cfg, project=project_id, metadata=metadata,
                             profile=rec_profile, plan_task=plan_task)
    tools = prof.get("tools")
    runner = create_runner(
        cfg, base_url, model, api_key=api_key,
        recorder=recorder,
        system_prompt=str(prof.get("system_prompt") or "") or None,
        enabled_tool_names=list(tools) if isinstance(tools, list) else None,
        workspace_root_override=prof.get("workspace_root"),
        project_workspace_dir=(proj or {}).get("workspace_dir"),
        delegation_context=child_ctx,
        # Projects: the child sees the same '## Project' section as its
        # parent (description + full agent roster with roles), so a
        # delegated agent knows the project and every teammate's job.
        project_context=proj,
        # Per-agent capabilities: the CHILD profile governs its own reach-out
        # (merged over global agent.capabilities inside create_runner).
        capabilities=prof.get("capabilities"),
        # Phase 5: child inherits the parent's project for index routing.
        project_id=project_id,
        # Pass 4 #38: the child profile selects its playbooks (no-op unless
        # playbooks.enabled).
        profile_name=prof.get("name"),
        # Audit §2 bug B: cooperative cancel — same runner mechanism as the
        # dashboard Stop button. None keeps exactly the old behavior.
        cancel_event=cancel_event,
        # Remote peer agents: X-Sable-Instance for multi-instance gateways.
        extra_headers=extra_headers)
    # D-17 §3.1: intra-loop liveness heartbeat — the runner ticks
    # delegation.touch() at every loop boundary and after each tool call so a
    # long (but healthy) tool call never looks wedged to the watchdog.
    # Default None on every other caller = zero behavior change.
    try:
        runner._heartbeat = lambda: touch(delegation_id)
    except Exception:  # noqa: BLE001 — exotic stub runners may forbid attrs
        pass
    t0 = time.monotonic()
    res = runner.run(task)
    # Training flow-back (D-16): a remote-inference child's transcript is
    # pushed to the model's owner via /remote/v1/traces/import, exactly like
    # chat's flow-back. Best-effort: a dead peer never breaks the child. The
    # LOCAL trace attribution stays the full `tim@host` name (recorded above).
    if prof.get("remote"):
        try:
            from ..dashboard import peers as peers_mod
            peers_mod.push_remote_trace(prof, task, res, t0)
        except Exception:  # noqa: BLE001 — flow-back is best-effort
            pass
    out["child_trace_id"] = res.trace_id
    out["completed"] = bool(res.completed)
    out["failure_reason"] = res.failure_reason
    out["result"] = (res.final_response or "")[:MAX_RESULT_CHARS]


def execute_delegation(agent: str, task: str, context: str = "",
                       timeout_seconds: Any = None,
                       plan_task: Optional[str] = None,
                       _ctx: Optional[Dict[str, Any]] = None,
                       wait: Optional[bool] = None) -> str:
    """Core of the delegate_task tool. Returns a JSON payload string.

    ``plan_task`` (WP12, optional): id of a task in the active project's
    persistent plan; when set the child prompt is built from that task with
    build_task_prompt (title/description/details/test-strategy + completed-
    dependency context) and the child trace is stamped with plan_task. The
    free-text ``task`` still rides along as extra instruction.

    ``wait`` (D-17): ``None`` -> config ``delegation.async_default`` (true:
    dispatch immediately, result arrives via inbox/check_delegation);
    ``True`` -> the legacy D-11 bounded synchronous wait; ``False`` -> force
    async. Deterministic internal callers (plan_loop-style code that needs
    results inline) must pass ``wait=True`` explicitly."""
    ctx = dict(_ctx or {})
    cfg = ctx.get("cfg")
    if cfg is None:
        return _error_payload("delegation unavailable: no configuration context "
                              "(runner built without create_runner)")
    if not getattr(cfg, "delegation_enabled", True):
        return _error_payload("delegation disabled (config delegation.enabled=false)")
    if int(ctx.get("depth", 0)) >= 1:
        return _error_payload(DELEGATION_BLOCKED_TOOLS_NOTE)
    agent = str(agent or "").strip()
    task = str(task or "").strip()
    if not agent:
        return _error_payload("agent is required (target profile name)")
    if not task:
        return _error_payload("task is required (non-empty subtask description)")

    # Target must be an agent of the active project (or any profile when no
    # project is set) AND exist in the profile list. A local miss with an "@"
    # in the name falls back to the peers registry (D-16: `tim@host` remote
    # agents run their loop HERE with inference on the peer).
    allowed = project_agent_names(cfg, ctx.get("project"))
    prof = find_profile(cfg, agent)
    if prof is None and "@" in agent:
        prof = _resolve_peer_profile(cfg, agent)
    if prof is None:
        known = ", ".join(allowed) or "(none configured)"
        if "@" in agent:
            return _error_payload(
                f"unknown agent {agent!r} — peer offline, never probed, or no "
                "such agent there (connect/refresh it from the Serve page); "
                f"available: {known}", agent=agent, code="peer_unreachable")
        return _error_payload(f"unknown agent {agent!r}; available: {known}",
                              agent=agent)
    if ctx.get("project") and agent not in allowed:
        return _error_payload(
            f"agent {agent!r} is not attached to the active project; "
            f"available: {', '.join(allowed) or '(none)'}", agent=agent)
    parent_profile = ctx.get("profile")
    if parent_profile and agent == parent_profile:
        return _error_payload(f"cannot delegate to self (profile {agent!r} is "
                              "already running this task)", agent=agent)

    # Healthcheck gate BEFORE the registry record exists: an unreachable peer
    # must not leave a phantom `running` record or lineage entry. Message is
    # byte-identical to chat.py's 503 text — one honest sentence both paths.
    if prof.get("remote"):
        from urllib.parse import urlparse

        from ..serving.supervisor import healthcheck as _hc

        purl = str(prof.get("peer_url") or "")
        pbase = purl.rstrip("/") + "/remote/v1"
        if not purl or not _hc(pbase, timeout=5.0,
                               api_key=prof.get("peer_token") or None):
            host = urlparse(purl).netloc or purl
            return _error_payload(
                f"remote sable at {host} is not serving right now (its "
                "dashboard must be up with expose_lan on, and a model "
                "running)", agent=agent, code="peer_unreachable")

    # ---- D-17: resolve wait mode + dispatcher saturation gate -------------
    if wait is None:
        async_mode = bool(getattr(cfg, "delegation_async_default", True))
    else:
        async_mode = not bool(wait)
    queue_max = int(getattr(cfg, "delegation_queue_max", 16) or 16)
    try:
        queued_now = _queued_count(cfg)
    except Exception:  # noqa: BLE001 — count failure never blocks a dispatch
        queued_now = 0
    if queued_now >= queue_max:
        return _error_payload(
            f"dispatcher saturated — {queued_now} delegations queued; "
            "retry shortly", code="dispatcher_saturated")

    timeout, timeout_warning = _resolve_timeout(cfg, timeout_seconds)
    record: Dict[str, Any] = {
        "delegation_id": uuid.uuid4().hex[:12],
        "agent": agent,
        "task": task[:300],
        # Async dispatch starts 'queued' (the DB is the queue — the bounded
        # pool claims FIFO); the sync path also flows queued -> running via
        # the same claim so max_concurrent holds everywhere.
        "status": "queued",
        "created": _utcnow(),
        "project": ctx.get("project"),
        "thread": ctx.get("thread"),
        "parent_run_id": ctx.get("run_id"),
        "depth": int(ctx.get("depth", 0)) + 1,
        "timeout_seconds": timeout,
        # Audit §6: set when the requested timeout was clamped (footgun guard);
        # surfaced verbatim in every tool payload for this delegation.
        "timeout_warning": timeout_warning,
        "started_mono": time.monotonic(),
        "result": None,
        "child_trace_id": None,
        "error": None,
        # Heartbeat (audit §2 bug A / D-17 §3): bumped on registration/
        # deregistration of the child runner, each wait-loop slice, and the
        # runner's intra-loop _heartbeat tick (delegation.touch). The liveness
        # watchdog keys wedge-detection off its staleness.
        "last_activity_ts": _utcnow(),
        # Cooperative cancel (audit §2 bug B): set on parent timeout, via
        # cancel_delegation(), the watchdog, or when the parent chat run is
        # cancelled. Forwarded into create_runner(cancel_event=...) for the
        # child.
        "cancel_event": threading.Event(),
        # Internal: True once a terminal state (completed/failed/cancelled/
        # wedged/interrupted) has been settled into the record; guards against
        # double-settling and lets a late child overwrite only running/timed_out/
        # cancelling.
        "_settled": False,
        # D-17: durable bookkeeping.
        "wait_mode": 0 if async_mode else 1,
        "liveness_timeout": float(getattr(
            cfg, "delegation_liveness_timeout_seconds", 600.0)),
        "inbox_state": "none",
        # Non-serialized refs for durable write-through (underscore keys are
        # never copied into payloads or the pruned ring).
        "_cfg_ref": cfg,
    }
    _store_delegation(record)

    def _notify() -> None:
        """Lineage + bus observer for the record's current state. Safe to call
        from either the parent or the child worker thread."""
        record["updated"] = _utcnow()
        thread_record_delegation(cfg, record.get("thread"), record)
        cb = ctx.get("on_update")
        if callable(cb):
            try:
                cb({k: record.get(k) for k in
                    ("delegation_id", "agent", "status", "project", "thread",
                     "child_trace_id", "parent_run_id")})
            except Exception:  # observers never break the delegation
                pass

    def _finish(payload_fields: Dict[str, Any]) -> str:
        # Audit §6: a clamped timeout must be visible in the tool result, not
        # silently honored — attach the warning to every payload path.
        if record.get("timeout_warning") and "timeout_warning" not in payload_fields:
            payload_fields["timeout_warning"] = record["timeout_warning"]
        _notify()
        return _payload(**payload_fields)

    # Lineage entry appears as soon as the delegation exists (status queued).
    thread_record_delegation(cfg, record.get("thread"), record)

    full_task = task if not context else f"{task}\n\nContext:\n{context}"
    # WP12: a plan-task id upgrades the child prompt to the full task graph
    # context (bounded); the free-text instruction stays appended. A bad/
    # unknown id degrades to the plain prompt with a note — delegation must
    # never fail just because the graph moved under us.
    if plan_task:
        try:
            from .plans import build_task_prompt, load_plan

            pstore = load_plan(ctx.get("project"))
            full_task = (build_task_prompt(pstore, str(plan_task))
                         + f"\n\nExtra instruction:\n{full_task}")
        except Exception as e:  # noqa: BLE001 - degrade, never break delegation
            full_task = (f"{full_task}\n\n(Note: plan task {plan_task!r} "
                         f"could not be expanded: {e})")
    box: Dict[str, Any] = {}

    def _settle_from_box() -> None:
        """Write the child's terminal state into the registry record AND the
        durable store (audit §2 bug A + D-17). Runs on the worker thread when
        the child finishes — even if the parent already stopped waiting — so a
        ``timed_out`` record is reconciled to completed/failed/cancelled.
        Guarded by ``_settled`` so exactly one terminal transition wins;
        ``running``, ``timed_out`` and ``cancelling`` are the only states it
        may replace (prefer-real-result). Terminal settle for an ASYNC
        delegation with a parent thread also flips ``inbox_state='pending'``
        so Phase 2 can deliver it as a wake turn."""
        cancel_flag = record["cancel_event"].is_set()
        fr = box.get("failure_reason")
        with _registry_lock:
            was_abandoned = record.get("status") in ("timed_out", "cancelling")
            if record.get("_settled"):
                return
            if box.get("exception"):
                record["status"] = "failed"
                record["error"] = str(box["exception"])
            elif fr == FAILURE_CANCELLED or (cancel_flag
                                             and not box.get("completed")):
                # Bug B: the child observed cancellation (runner returned
                # failure_reason="cancelled", or exited incomplete while a
                # cancel was pending) -> cooperative terminal state. A child
                # that genuinely COMPLETED despite a racy cancel keeps its
                # real result (prefer-real-result rule).
                record["status"] = "cancelled"
                record["error"] = ("delegation cancelled; child stopped "
                                   "cooperatively")
            elif box.get("completed"):
                record["status"] = "completed"
            else:
                record["status"] = "failed"
                record["error"] = (f"child run did not complete "
                                   f"({fr or 'unknown reason'})")
            for k in ("child_trace_id", "completed", "failure_reason", "result"):
                record[k] = box.get(k)
            record["last_activity_ts"] = _utcnow()
            record["settled_ts"] = _utcnow()
            record["_settled"] = True
            # Inbox write (D-17): every async terminal status with a parent
            # thread lands pending — an honest "your child died" beats
            # silence. Sync (wait=true) results were already delivered in the
            # tool payload, so they never double-post to the inbox.
            if record.get("thread") and not record.get("wait_mode"):
                record["inbox_state"] = "pending"
        _store_write(
            record,
            status=record["status"], result=record.get("result"),
            error=record.get("error"),
            child_trace_id=record.get("child_trace_id"),
            settled_ts=record.get("settled_ts"),
            last_activity_ts=record.get("last_activity_ts"),
            inbox_state=record.get("inbox_state", "none"))
        if was_abandoned:
            # The parent is gone (already returned timed_out): the child's
            # late terminal state still needs lineage + bus propagation.
            _notify()

    def _run_and_settle() -> None:
        """The dispatcher closure for THIS delegation: execute the child and
        settle (called by _pump after the atomic queued->running claim)."""
        try:
            _run_child(cfg, prof, full_task, ctx, record["delegation_id"],
                       box, plan_task=str(plan_task or "") or None,
                       cancel_event=record["cancel_event"])
        except Exception as e:  # child blew up; surfaced below as failed
            box["exception"] = f"{type(e).__name__}: {e}"
        _settle_from_box()

    # Order matters: the exec closure is registered BEFORE the durable row
    # becomes visible to pumps, so a sibling pump claiming this FIFO row can
    # never find its context missing (the row only exists once both cache and
    # exec map hold it).
    with _registry_lock:
        _pending_exec[record["delegation_id"]] = _run_and_settle
    try:  # durable insert (idempotent DDL inside); cache stays authoritative
        delegation_store.insert_delegation(
            _row_from_record(record, cfg), cfg=cfg)
    except Exception:  # noqa: BLE001 — store down => cache-only legacy shape
        pass
    _ensure_dispatcher(cfg)
    try:
        with _disp_lock:
            if _executor is None:
                init_dispatcher(cfg)
            ex = _executor
        if ex is not None:
            ex.submit(_pump)
    except Exception as e:  # noqa: BLE001 — pool dead (shutdown race)
        with _registry_lock:
            _pending_exec.pop(record["delegation_id"], None)
            record["status"] = "failed"
            record["error"] = f"dispatcher unavailable: {e}"
            record["_settled"] = True
        return _finish({"status": "failed",
                        "delegation_id": record["delegation_id"],
                        "agent": agent, "error": record["error"]})

    # ---- async path (D-17 default): return dispatched immediately ---------
    if async_mode:
        fields: Dict[str, Any] = {
            "status": "dispatched",
            "delegation_id": record["delegation_id"],
            "agent": agent,
            "queued_ahead": max(0, queued_now),
        }
        if record.get("thread"):
            fields["note"] = ("child runs in the background; its result "
                              "arrives as a follow-up turn — do NOT busy-"
                              "poll; check_delegation(id) any time")
        else:
            fields["note"] = "no chat thread context — poll check_delegation"
        return _finish(fields)

    # ---- legacy synchronous path (wait=true, D-11 semantics) --------------
    # Parent-level cancel (audit §2 bug B): chat passes its run's cancel Event
    # into the delegation context; a set parent event flips this delegation's
    # cooperative cancel too. CLI/stubs pass none and behave exactly as before.
    parent_cancel = ctx.get("cancel_event")
    timed_out = False
    cancelled_by_parent = False
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            break
        # Heartbeat wait slice: cheap poll granularity so last_activity_ts
        # keeps ticking while the child runner is registered in _active_runs.
        time.sleep(min(_HEARTBEAT_POLL_S, remaining))
        with _registry_lock:
            settled_now = bool(record.get("_settled"))
            if record["delegation_id"] in _active_runs:
                record["last_activity_ts"] = _utcnow()
        if settled_now:
            break
        # Cancel sources: the parent run's Event (chat Stop) or an external
        # cancel_delegation(id) call — both flip this delegation's event. If
        # the child settled concurrently, prefer its real terminal state.
        if ((parent_cancel is not None and parent_cancel.is_set())
                or record["cancel_event"].is_set()):
            with _registry_lock:
                settled_now = bool(record.get("_settled"))
            if settled_now:
                break
            record["cancel_event"].set()
            cancelled_by_parent = True
            break
    if cancelled_by_parent:
        # Child stops at its next loop boundary and settles the record to
        # "cancelled"; mark the transitional state so late completion still
        # reconciles through _settle_from_box. Never wait on the child here.
        with _registry_lock:
            if not record.get("_settled"):
                record["status"] = "cancelling"
        _store_write(record, status="cancelling")
        return _finish({
            "status": "cancelled",
            "delegation_id": record["delegation_id"],
            "agent": agent,
            "error": ("parent run cancelled; child notified to stop "
                      "(record settles to cancelled)"),
        })
    if timed_out:
        # Race boundary: the child may have settled right as we timed out —
        # prefer its real terminal state over the timeout payload.
        with _registry_lock:
            settled = bool(record.get("_settled"))
        elapsed = round(time.monotonic() - record["started_mono"], 2)
        if not settled:
            with _registry_lock:
                if not record.get("_settled"):
                    record["status"] = "timed_out"
                    record["error"] = (
                        f"child agent {agent!r} timed out after {timeout}s "
                        "without returning")
            # Bug B: cooperative stop instead of pure abandonment — the child
            # polls this event at runner loop boundaries. If it observes it,
            # the record reconciles timed_out -> cancelled; if the child
            # finishes anyway (older runners, in-flight syscall) its terminal
            # state still lands via _settle_from_box. A row still 'queued'
            # (pool saturation) gets its pending exec dropped so no pump ever
            # starts it late.
            record["cancel_event"].set()
            with _registry_lock:
                _pending_exec.pop(record["delegation_id"], None)
            _store_write(record, status="timed_out", error=record["error"])
            return _finish({
                "status": "timed_out",
                "delegation_id": record["delegation_id"],
                "agent": agent,
                "error": record["error"],
                "timeout_seconds": timeout,
                "elapsed_s": elapsed,
                "partial_output": None,
                "diagnostics": {
                    "child_agent": agent,
                    "project": record.get("project"),
                    "task_preview": task[:200],
                    "note": ("child notified to stop (cooperative cancel); "
                             "its trace (if any) is still recorded; re-check "
                             "with check_delegation for the settled state"),
                },
            })

    # Normal path: the worker has settled the record — build the payload from
    # its terminal state (identical shapes to before, plus cancelled).
    status = record.get("status")
    if status == "completed":
        return _finish({
            "status": "completed",
            "delegation_id": record["delegation_id"],
            "agent": agent,
            "result": record.get("result"),
            "child_trace_id": record.get("child_trace_id"),
            "duration_s": round(time.monotonic() - record["started_mono"], 2),
        })
    if status == "cancelled":
        return _finish({
            "status": "cancelled",
            "delegation_id": record["delegation_id"],
            "agent": agent,
            "error": record.get("error"),
            "child_trace_id": record.get("child_trace_id"),
        })
    if box.get("exception"):
        return _finish({
            "status": "failed",
            "delegation_id": record["delegation_id"],
            "agent": agent,
            "error": box["exception"],
            "child_trace_id": box.get("child_trace_id"),
        })
    return _finish({
        "status": "failed",
        "delegation_id": record["delegation_id"],
        "agent": agent,
        "error": record.get("error"),
        "failure_reason": box.get("failure_reason"),
        "partial_output": record.get("result"),
        "child_trace_id": record.get("child_trace_id"),
    })


def cancel_delegation(delegation_id: str) -> bool:
    """Cooperatively cancel a delegation (audit §2 bug B). Sets the record's
    cancel Event — the child runner polls it at loop boundaries and stops
    cleanly, settling the record to ``cancelled``. Works on ``queued``,
    ``running`` and already-``timed_out`` records: a queued row never started
    anything, so it settles straight to ``cancelled`` (state machine §3.2);
    running/timed_out flip to ``cancelling`` until the child observes it.
    Returns False for unknown or already-settled ids."""
    did = str(delegation_id or "").strip()
    with _registry_lock:
        rec = _delegations.get(did)
        if rec is None or rec.get("_settled"):
            return False
        ev: Optional[threading.Event] = rec.get("cancel_event")
        if ev is None:
            return False
        ev.set()
        was_queued = rec.get("status") == "queued"
        if was_queued:
            # Never started: settle cancelled right here (drop the exec too).
            _pending_exec.pop(did, None)
            rec["status"] = "cancelled"
            rec["error"] = "delegation cancelled before the child started"
            rec["settled_ts"] = _utcnow()
            rec["_settled"] = True
            if rec.get("thread") and not rec.get("wait_mode"):
                rec["inbox_state"] = "pending"
        elif rec.get("status") in ("running", "timed_out"):
            rec["status"] = "cancelling"
    cfg = rec.get("_cfg_ref") if rec else None
    try:
        if was_queued:
            delegation_store.update_delegation(
                did, cfg=cfg, status="cancelled", settled_ts=_utcnow(),
                error="delegation cancelled before the child started",
                inbox_state=("pending" if (rec or {}).get("thread")
                             and not (rec or {}).get("wait_mode") else "none"),
                expect_status="queued")
        else:
            delegation_store.update_delegation(
                did, cfg=cfg, status="cancelling", expect_status="running")
    except Exception:  # noqa: BLE001 — durable best-effort
        pass
    return True


# ---------------------------------------------------------------------------
# check_delegation (cache fast path -> pruned ring -> durable store)
# ---------------------------------------------------------------------------

def _payload_from_store_row(row: Dict[str, Any], cfg: Any) -> str:
    """check_delegation answer from a durable row (post-restart / cache-miss
    fallback — audit §4 item 3)."""
    status = row.get("status")
    fields: Dict[str, Any] = {
        "status": status,
        "delegation_id": row.get("delegation_id"),
        "agent": row.get("agent"),
        "created": row.get("created_ts"),
        "last_activity_ts": row.get("last_activity_ts"),
    }
    if status == "queued":
        try:
            fields["queued_ahead"] = delegation_store.count_queued_before(
                str(row.get("created_ts") or ""), cfg=cfg)
        except Exception:  # noqa: BLE001
            pass
    if status in ("completed", "failed", "timed_out", "cancelled", "wedged",
                  "interrupted"):
        fields.update({
            "result": row.get("result"),
            "child_trace_id": row.get("child_trace_id"),
            "error": row.get("error"),
        })
    return _payload(**fields)


def check_delegation_status(delegation_id: str) -> str:
    # The child worker settles the record itself (bug A fix), so reads are
    # always current; a still-running child that finishes after a timeout
    # flips timed_out -> completed/failed/cancelled without any polling here.
    did = str(delegation_id or "").strip()
    rec = get_delegation(did)
    if rec is None:
        # Audit §6: distinguish "this process never saw that id" from "we saw
        # it but the bounded registry pruned it". The last _PRUNED_RING_CAP
        # settled evictions return their real terminal summary with a
        # "(pruned)" marker; anything older/unknown falls back to the durable
        # store (D-17: survives restarts) before the legacy error.
        pruned = get_pruned_summary(did)
        if pruned is not None:
            fields = {
                "status": pruned.get("status"),
                "delegation_id": did,
                "agent": pruned.get("agent"),
                "created": pruned.get("created"),
                "last_activity_ts": pruned.get("last_activity_ts"),
                "result": pruned.get("result"),
                "child_trace_id": pruned.get("child_trace_id"),
                "error": pruned.get("error"),
                "pruned": True,  # (pruned) marker per audit §6
                "note": ("record was pruned from the in-memory registry "
                         "(cap/age); this is its terminal summary and its "
                         "trace remains in the Traces store"),
            }
            if pruned.get("timeout_warning"):
                fields["timeout_warning"] = pruned["timeout_warning"]
            return _payload(**fields)
        try:
            row = delegation_store.get_delegation(did, cfg=_default_cfg())
        except Exception:  # noqa: BLE001 — store unavailable => legacy error
            row = None
        if row is not None:
            return _payload_from_store_row(row, _default_cfg())
        return _error_payload(
            f"unknown delegation id {delegation_id!r} "
            "(ids are tracked in-process for the current session; settled "
            "records pruned by the registry cap answer with a (pruned) "
            "terminal summary)")
    fields = {
        "status": rec.get("status"),
        "delegation_id": rec.get("delegation_id"),
        "agent": rec.get("agent"),
        "created": rec.get("created"),
        # Heartbeat (groundwork for liveness timeout): last time the registry
        # saw activity for this run.
        "last_activity_ts": rec.get("last_activity_ts"),
    }
    if rec.get("status") == "queued":
        try:
            fields["queued_ahead"] = delegation_store.count_queued_before(
                str(rec.get("created") or ""), cfg=rec.get("_cfg_ref"))
        except Exception:  # noqa: BLE001
            pass
    if rec.get("status") in ("completed", "failed", "timed_out", "cancelled",
                             "wedged", "interrupted"):
        fields.update({
            "result": rec.get("result"),
            "child_trace_id": rec.get("child_trace_id"),
            "error": rec.get("error"),
        })
    return _payload(**fields)


def _peer_serving_states(cfg: Any) -> Dict[str, str]:
    """host -> "serving" | "idle" | "unknown" for remembered peers (audit §6).

    Reads ONLY the cached probe snapshots already persisted in peers.json —
    zero network calls, so list_agents stays as cheap as it was. A peer with
    no snapshot yet (or a failed last probe) reports "unknown"; serving means
    at least one model instance was running on the peer at last probe."""
    from urllib.parse import urlparse

    try:
        from ..dashboard import peers as peers_mod
    except ImportError:
        return {}
    from ..store.paths import resolve_sable_home

    out: Dict[str, str] = {}
    try:
        recs = peers_mod.load_peers(resolve_sable_home(cfg))
    except Exception:  # peers state is best-effort for a listing
        return {}
    for p in recs:
        snap = p.get("snapshot") or {}
        if not snap.get("ok"):
            state = "unknown"
        else:
            servers = snap.get("servers") or []
            busy = bool((snap.get("server") or {}).get("running")) or any(
                isinstance(s, dict) and s.get("running") for s in servers)
            state = "serving" if busy else "idle"
        u = urlparse(str(p.get("url") or ""))
        for key in {u.hostname, u.netloc}:
            if key:
                out[key] = state
    return out


def list_agents_text(_ctx: Optional[Dict[str, Any]] = None) -> str:
    """Core of the list_agents tool: JSON list of candidate agents."""
    ctx = dict(_ctx or {})
    cfg = ctx.get("cfg")
    if cfg is None:
        return _error_payload("delegation unavailable: no configuration context")
    names = project_agent_names(cfg, ctx.get("project"))
    out: List[Dict[str, Any]] = []
    # Cached peer serving states, built at most once per render (audit §6):
    # snapshot-only lookup, never a blocking probe.
    peer_states: Optional[Dict[str, str]] = None
    for name in names:
        # Remote peer agents (`tim@host`) aren't local profiles; resolve them
        # through the peers registry so their persona's first line shows up
        # instead of "(no description)". Helper already swallows failures.
        prof = (find_profile(cfg, name)
                or (_resolve_peer_profile(cfg, name) if "@" in name else None)
                or {"name": name})
        running = running_count_for(name)
        entry: Dict[str, Any] = {
            "agent": name,
            "description": profile_description(prof),
            "running": running > 0,
            "running_runs": running,
            "is_self": bool(ctx.get("profile")) and name == ctx.get("profile"),
        }
        if "@" in name:
            if peer_states is None:
                peer_states = _peer_serving_states(cfg)
            entry["peer_serving"] = peer_states.get(
                name.rpartition("@")[2], "unknown")
        out.append(entry)
    return _payload(
        status="ok",
        project=ctx.get("project"),
        scope=("project agents" if ctx.get("project") else "all configured profiles"),
        agents=out)

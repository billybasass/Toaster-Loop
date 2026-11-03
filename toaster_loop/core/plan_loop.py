"""WP12 guarded autonomous plan loop (`toaster_loop plan loop`).

Runs unblocked plan tasks back-to-back with real agent runs, then closes each
task ONLY on explicit completion: the task must have been flipped to ``done``
via the ``plan_progress`` tool during its own run. Anything else (the agent
stopped without closing, hit the verification nudge, or crashed) marks the
task ``blocked`` with a machine-readable reason — the loop never decides on
the agent's behalf that work is finished.

Toaster Loop-style constraints (WP12 spec §6):

* Dashboard-style **daemon thread** with a JSON progress file
  (``<home>/plans/<project>.loop.json``) — NOT the GPU scheduler: the loop
  drives an already-serving endpoint, it does not own one. The CLI attaches
  to the same module functions; the dashboard runs it on its own thread.
* **Single-flight machine-wide**: one loop at a time (any project). A second
  start raises :class:`LoopConflictError` — never queues silently.
* **No permission bypass ever**: each task run uses the task's ``profile``
  (or the project default profile) through ``create_runner``, so the
  profile's tool subset and capability grants apply unchanged; capabilities
  stay off by default.
* **Stops**: max_tasks, no unblocked tasks, a blocked/failed task once
  ``loop.max_consecutive_failures`` is reached, cancel (Event), or dashboard
  shutdown (daemon thread dies best-effort: the progress file and any
  in_progress task stay visible, no zombie state).
* SSE kind ``plan.loop`` {project, task_id, status, iteration} per event via
  an injected ``publish`` callable (dashboard bus; CLI prints instead).

Task runs have NO timeout: an agent may take as long as it needs to finish a
task. The loop waits for each run to settle; only cancel (Event) or dashboard
shutdown (daemon thread dies best-effort) interrupt an in-flight run.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .plans import (build_task_prompt, load_plan, next_task, plans_config,
                    save_plan, set_status)

logger = logging.getLogger("toaster_loop.core.plan_loop")

LOOP_FILE_SUFFIX = ".loop.json"

#: In-memory single-flight: at most ONE loop machine-wide (any project). The
#: dashboard and CLI are separate processes; each enforces its own in-process
#: flight plus the on-disk running marker below, which covers cross-process.
_active_lock = threading.Lock()
_current: Optional["PlanLoop"] = None


class LoopConflictError(RuntimeError):
    """A plan loop is already running (single-flight)."""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Progress file (<home>/plans/<project>.loop.json)
# ---------------------------------------------------------------------------

def loop_path(project_id: Optional[str], home: Path) -> Path:
    from .plans import plans_dir

    pid = str(project_id or "default").strip() or "default"
    return plans_dir(home) / (pid + LOOP_FILE_SUFFIX)


def read_loop_progress(project_id: Optional[str], home: Path
                       ) -> Optional[Dict[str, Any]]:
    """Tolerant read of the loop progress file; None when absent/corrupt."""
    p = loop_path(project_id, home)
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def write_loop_progress(project_id: Optional[str], home: Path,
                        progress: Dict[str, Any]) -> None:
    """Atomic tmp + os.replace (same discipline as plan/project stores)."""
    p = loop_path(project_id, home)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=".loop-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)
        os.replace(tmp, p)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

#: run_fn(prompt, task, profile_name) -> dict with at least:
#:   {"completed": bool, "failure_reason": str|None}  (AgentResult-shaped ok)
RunFn = Callable[[str, Dict[str, Any], Optional[str]], Dict[str, Any]]


class PlanLoop:
    """One guarded loop run over a project's plan. Use start_plan_loop()."""

    def __init__(self, cfg: Any, project_id: Optional[str], *,
                 max_tasks: Optional[int] = None,
                 run_fn: Optional[RunFn] = None,
                 publish: Optional[Callable[[str, Dict[str, Any]], None]] = None,
                 resume: bool = False, force: bool = False):
        global _current
        with _active_lock:
            if _current is not None and not getattr(_current, "finished",
                                                    True):
                raise LoopConflictError(
                    f"a plan loop is already running for project "
                    f"{_current.project_id or 'default'} "
                    f"(iteration {_current.progress.get('iteration')})")
            # Cross-process marker: a previous run of THIS project still says
            # running => refuse too (two CLIs against one TOASTER_LOOP_HOME). A hard
            # kill leaves this marker stale forever, so `force` overrides it
            # after an in-process check confirms nothing local is running.
            prior = read_loop_progress(project_id, _home(cfg))
            if not force and prior and prior.get("status") == "running" \
                    and str(prior.get("project") or "") == \
                    str(project_id or ""):
                raise LoopConflictError(
                    "progress file says a loop is still running for this "
                    "project (stale? pass force/--force, or delete the "
                    ".loop.json to override)")
            _current = self  # reserved now; cleared on exit OR init failure
        try:
            self._init_body(cfg, project_id, max_tasks=max_tasks,
                            run_fn=run_fn, publish=publish, resume=resume)
        except Exception:
            # Never leave a half-built instance holding the single-flight
            # slot (a config error here would wedge every later start).
            with _active_lock:
                if _current is self:
                    _current = None
            raise

    def _init_body(self, cfg: Any, project_id: Optional[str], *,
                   max_tasks: Optional[int] = None,
                   run_fn: Optional[RunFn] = None,
                   publish: Optional[Callable[[str, Dict[str, Any]], None]] = None,
                   resume: bool = False) -> None:
        self.cfg = cfg
        self.project_id = str(project_id or "").strip() or None
        pcfg = plans_config(cfg)
        lcfg = pcfg.get("loop") or {}
        self.max_tasks = int(max_tasks if max_tasks is not None
                             else lcfg.get("max_tasks", 10))
        self.require_verification = bool(lcfg.get("require_verification", True))
        self.max_consecutive_failures = max(
            1, int(lcfg.get("max_consecutive_failures", 1)))
        self.publish = publish or (lambda kind, payload: None)
        self.cancel_event = threading.Event()
        self.finished = False
        # The default runner shares the loop's cancel event so cancel()
        # reaches in-flight agent runs at their loop boundaries. There is no
        # per-task timeout: a run goes as long as it needs; cancel is the only
        # interrupt. Signature-tolerant for injected/legacy run_fn factories
        # that predate the kwarg.
        if run_fn is not None:
            self._run_fn = run_fn
        else:
            try:
                self._run_fn = _default_run_fn(
                    cfg, self.project_id, cancel_event=self.cancel_event)
            except TypeError:
                self._run_fn = _default_run_fn(cfg, self.project_id)
        self.thread: Optional[threading.Thread] = None
        self.resume = resume

        prior = read_loop_progress(self.project_id, _home(cfg)) if resume \
            else None
        self.progress: Dict[str, Any] = {
            "project": self.project_id or "",
            "status": "running",
            "started": _utcnow(),
            "iteration": int((prior or {}).get("iteration") or 0),
            "max_tasks": self.max_tasks,
            "tasks": list((prior or {}).get("tasks") or []) if resume else [],
            "stop_reason": None,
        }
        _write(self)

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> "PlanLoop":
        """Launch on a daemon thread (dashboard-style long action)."""
        self.thread = threading.Thread(target=self._guarded,
                                       name=f"plan-loop-"
                                       f"{self.project_id or 'default'}",
                                       daemon=True)
        self.thread.start()
        return self

    def cancel(self) -> bool:
        """Request cancellation; True if this loop was still running."""
        if self.finished:
            return False
        self.cancel_event.set()
        return True

    def wait(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        if self.thread is not None:
            self.thread.join(timeout=timeout)
        return self.progress

    # -- main loop ----------------------------------------------------------

    def _guarded(self) -> None:
        global _current
        try:
            self._run_all()
        except Exception as e:  # loop must die with a recorded reason, never silently
            logger.exception("plan loop crashed")
            self.progress["stop_reason"] = f"error: {e}"
            self.progress["status"] = "error"
        finally:
            self.finished = True
            if self.progress.get("status") == "running":
                # Cancellation or unexpected exit path: keep it honest.
                self.progress["status"] = "cancelled" \
                    if self.cancel_event.is_set() else "stopped"
            self.progress["ended"] = _utcnow()
            try:
                _write(self)
            except OSError:
                pass
            # Release a llama-server this loop spawned (no-op in attach
            # mode / injected run_fns without a close hook).
            closer = getattr(self._run_fn, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:  # noqa: BLE001
                    logger.debug("plan loop endpoint close failed",
                                 exc_info=True)
            with _active_lock:
                if _current is self:
                    _current = None

    def _run_all(self) -> None:
        consecutive_failures = 0
        done_count = 0
        while not self.cancel_event.is_set():
            if done_count >= self.max_tasks:
                self.progress["stop_reason"] = "max_tasks"
                return
            store = load_plan(self.project_id)
            task = next_task(store)
            if task is None:
                self.progress["stop_reason"] = "no_unblocked_tasks"
                return
            task_id = task["id"]
            self.progress["iteration"] += 1
            self.progress["current_task"] = task_id
            self._emit(task_id, "started")

            prompt = build_task_prompt(store, task_id)
            result = self._run_task(prompt, task)

            # Re-read: the agent's plan_progress calls mutated the store.
            store = load_plan(self.project_id)
            cur = next((t for t in store["tasks"] if t["id"] == task_id), None)
            status_now = (cur or {}).get("status", "?")

            outcome, reason = self._classify(result, status_now)
            if outcome != "done":
                # Mark blocked with the reason unless the agent moved it on.
                if cur is not None and status_now not in ("done", "cancelled"):
                    try:
                        from .plans import lock as _plan_lock
                        with _plan_lock():
                            # Re-read under the lock: the store may have been
                            # edited since our post-run load above.
                            store = load_plan(self.project_id)
                            cur2 = next((t for t in store["tasks"]
                                         if t["id"] == task_id), None)
                            if cur2 is not None and cur2["status"] not in \
                                    ("done", "cancelled"):
                                set_status(store, task_id, "blocked",
                                           note=f"loop: {reason}")
                                save_plan(store)
                    except Exception as e:  # noqa: BLE001 - report, keep going
                        logger.warning("loop could not mark %s blocked: %s",
                                       task_id, e)
            entry = {"id": task_id, "title": (cur or task).get("title", ""),
                     "iteration": self.progress["iteration"],
                     "outcome": outcome, "reason": reason,
                     "at": _utcnow()}
            self.progress["tasks"].append(entry)
            _write(self)

            if outcome == "done":
                done_count += 1
                consecutive_failures = 0
                self._emit(task_id, "done")
            else:
                consecutive_failures += 1
                self._emit(task_id, "blocked")
                if consecutive_failures >= self.max_consecutive_failures:
                    self.progress["stop_reason"] = (
                        f"consecutive_failures ({reason})")
                    return
        self.progress["stop_reason"] = "cancelled"

    def _classify(self, result: Dict[str, Any], status_now: str
                  ) -> tuple:
        """Close ONLY on explicit done; everything else is a blocked reason."""
        if status_now == "done":
            return "done", ""
        if status_now == "cancelled":
            return "skipped", "task cancelled during run"
        fr = result.get("failure_reason")
        if self.require_verification and fr == "verification_nudge":
            return "blocked", ("require_verification: run ended on the "
                               "verification nudge without closing the task")
        if not result.get("ok", False):
            detail = fr or result.get("error") or "run failed"
            return "blocked", f"run failure: {detail}"
        if status_now == "in_progress":
            return "blocked", ("agent never closed the task via plan_progress "
                               "(left in_progress)")
        return "blocked", f"task not closed by the run (status={status_now})"

    def _run_task(self, prompt: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Run one task to completion — no deadline. The agent takes as long
        as it needs; the loop thread simply waits. Exceptions surface as a
        blocked outcome instead of killing the loop."""
        box: Dict[str, Any] = {}
        prof = task.get("profile") or None
        try:
            box.update(self._run_fn(prompt, task, prof) or {})
        except Exception as e:  # noqa: BLE001 - surfaced as blocked
            box["ok"] = False
            box["error"] = f"{type(e).__name__}: {e}"
        box.setdefault("ok", False)
        return box

    def _emit(self, task_id: str, status: str) -> None:
        try:
            self.publish("plan.loop", {
                "project": self.project_id or "",
                "task_id": task_id, "status": status,
                "iteration": self.progress["iteration"],
            })
        except Exception:  # noqa: BLE001 - SSE is best-effort
            pass


def _home(cfg: Any) -> Path:
    from ..store.paths import resolve_toaster_loop_home

    return resolve_toaster_loop_home(cfg)


def _write(loop: PlanLoop) -> None:
    write_loop_progress(loop.project_id, _home(loop.cfg), loop.progress)


# ---------------------------------------------------------------------------
# Default run_fn: real agent runs through create_runner (profile grants kept)
# ---------------------------------------------------------------------------

def _default_run_fn(cfg: Any, project_id: Optional[str], *,
                    cancel_event: Optional[threading.Event] = None) -> RunFn:
    """Resolve the serving endpoint ONCE per loop (lazily on first task), then
    run each task through create_runner with the task's profile (tool subset +
    capabilities are whatever that profile grants — no bypass).

    The endpoint is deliberately NOT torn down between tasks: in spawn mode a
    stop/start per iteration would reload the whole model for every task.
    The callable carries a ``close()`` attribute the loop invokes when it ends
    (see PlanLoop._guarded), which stops a supervisor THIS loop spawned —
    attach-mode servers are never touched.

    Cancellation is forwarded into create_runner so runner.run stops at its
    own boundaries when the user cancels. There is no per-task deadline: each
    run goes as long as it needs."""

    state: Dict[str, Any] = {}          # {"supervisor": ...} once resolved

    def _endpoint() -> tuple:
        if "base_url" not in state:
            from ..cli import _resolve_endpoint

            base_url, model, supervisor = _resolve_endpoint(
                cfg, spawn_if_needed=True)
            state["base_url"] = base_url
            state["model"] = model
            state["supervisor"] = supervisor
        return state["base_url"], state["model"]

    def run(prompt: str, task: Dict[str, Any],
            profile_name: Optional[str]) -> Dict[str, Any]:
        from ..store.recorder import make_recorder
        from .agent_runner import create_runner
        from .delegation import load_project

        base_url, model = _endpoint()
        prof = None
        if profile_name:
            profs = cfg.profiles if isinstance(cfg.profiles, list) else []
            prof = next((p for p in profs
                         if isinstance(p, dict)
                         and p.get("name") == profile_name), None)
        tools = (prof or {}).get("tools")
        if isinstance(tools, list) and "plan_progress" not in tools:
            # The loop only closes a task when the agent flips it to done via
            # plan_progress; without that tool every iteration deterministically
            # ends blocked("never closed"). Warn loudly instead of looping dry.
            logger.warning(
                "plan loop: profile %r does not grant plan_progress — tasks "
                "can never close and will all end blocked", profile_name)
        proj = load_project(cfg, project_id) if project_id else None
        recorder = make_recorder(cfg, project=project_id,
                                 plan_task=task.get("id"))
        runner = create_runner(
            cfg, base_url, str((prof or {}).get("model_name") or model),
            api_key=getattr(cfg, "attach_api_key", None) or None,
            recorder=recorder,
            system_prompt=str((prof or {}).get("system_prompt") or "")
            or None,
            enabled_tool_names=[str(t) for t in tools]
            if isinstance(tools, list) else None,
            workspace_root_override=(prof or {}).get("workspace_root"),
            project_workspace_dir=(proj or {}).get("workspace_dir"),
            project_context=proj,
            capabilities=(prof or {}).get("capabilities"),
            cancel_event=cancel_event,
            project_id=project_id,
            profile_name=(prof or {}).get("name"))
        res = runner.run(prompt)
        return {"ok": bool(getattr(res, "completed", False)),
                "failure_reason": getattr(res, "failure_reason", None),
                "trace_id": getattr(res, "trace_id", None)}

    def close() -> None:
        sup = state.get("supervisor")
        if sup is not None:
            try:
                sup.stop()
            except Exception:  # noqa: BLE001
                pass
    run.close = close  # type: ignore[attr-defined]
    return run


# ---------------------------------------------------------------------------
# Public helpers (CLI + dashboard)
# ---------------------------------------------------------------------------

def start_plan_loop(cfg: Any, project_id: Optional[str], *,
                    max_tasks: Optional[int] = None,
                    run_fn: Optional[RunFn] = None,
                    publish: Optional[Callable[[str, Dict[str, Any]], None]] = None,
                    resume: bool = False, force: bool = False) -> PlanLoop:
    """Create + start a loop on its daemon thread. Raises LoopConflictError
    when another loop (in-process or marked running on disk) holds the slot;
    ``force`` overrides only the stale on-disk marker, never a live loop."""
    return PlanLoop(cfg, project_id, max_tasks=max_tasks, run_fn=run_fn,
                    publish=publish, resume=resume, force=force).start()


def cancel_plan_loop(project_id: Optional[str]) -> bool:
    """Cancel the in-process loop if it serves this project."""
    with _active_lock:
        cur = _current
    if cur is None or cur.finished:
        return False
    if str(cur.project_id or "") != str(project_id or ""):
        return False
    return cur.cancel()


def current_loop() -> Optional[PlanLoop]:
    with _active_lock:
        cur = _current
    return None if (cur is None or cur.finished) else cur

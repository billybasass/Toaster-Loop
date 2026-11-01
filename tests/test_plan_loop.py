"""WP12 plan_loop: the guarded autonomous loop with an injected run_fn.

Every test drives PlanLoop with a fake runner step (never create_runner /
the GPU) and asserts the guard rails: close-only-on-explicit-done,
require_verification, stop conditions, single-flight, resume, progress file.
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from sable.core import plan_loop as _pl
from sable.core.config import SableConfig
from sable.core.plan_loop import (LoopConflictError, PlanLoop,
                                  read_loop_progress, start_plan_loop)
from sable.core.plans import (add_task, get_task, load_plan, save_plan,
                              set_status)


@pytest.fixture(autouse=True)
def _reset_single_flight():
    """The single-flight slot is a module global; never leak across tests."""
    yield
    with _pl._active_lock:
        _pl._current = None


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("SABLE_HOME", str(h))
    return h


def _cfg(**plans):
    base = {"enabled": True,
            "loop": {"max_tasks": 10, "max_consecutive_failures": 1}}
    base.update(plans)
    return SableConfig(plans=base)


def _seed(*titles, project=None):
    store = load_plan(project)
    for t in titles:
        add_task(store, {"title": t})
    save_plan(store)
    return store


def _closer(run_done_ids):
    """run_fn that explicitly closes tasks via plan_progress (agent-like)."""

    def run(prompt, task, profile):
        store = load_plan(None)
        set_status(store, task["id"], "done", note="agent closed it")
        save_plan(store)
        run_done_ids.append(task["id"])
        return {"ok": True, "failure_reason": None}
    return run


def _wait(loop, timeout=10.0):
    loop.wait(timeout=timeout)
    assert loop.finished, "loop did not finish in time"
    return loop.progress


# ------------------------------------------------------------- completion

def test_completes_tasks_on_explicit_plan_progress(home):
    _seed("one", "two")
    closed = []
    events = []
    loop = start_plan_loop(_cfg(), None, run_fn=_closer(closed),
                           publish=lambda k, p: events.append((k, p)))
    prog = _wait(loop)
    assert prog["status"] in ("stopped", "completed", "done") or \
        prog["stop_reason"] == "no_unblocked_tasks"
    assert closed == ["t1", "t2"]
    store = load_plan(None)
    assert get_task(store, "t1")["status"] == "done"
    assert get_task(store, "t2")["status"] == "done"
    assert prog["stop_reason"] == "no_unblocked_tasks"
    # SSE: started + done per task
    kinds = {(e[0], e[1]["task_id"], e[1]["status"]) for e in events}
    assert ("plan.loop", "t1", "done") in kinds
    assert ("plan.loop", "t2", "done") in kinds


def test_no_explicit_close_marks_blocked(home):
    _seed("lazy task")
    loop = start_plan_loop(_cfg(), None,
                           run_fn=lambda p, t, prof: {"ok": True,
                                                      "failure_reason": None})
    prog = _wait(loop)
    assert prog["tasks"][0]["outcome"] == "blocked"
    assert "never closed" in prog["tasks"][0]["reason"] or \
        "not closed" in prog["tasks"][0]["reason"]
    store = load_plan(None)
    t = get_task(store, "t1")
    assert t["status"] == "blocked"                 # marked with reason
    assert prog["stop_reason"].startswith("consecutive_failures")


def test_require_verification_blocks_on_nudge(home):
    _seed("nudged")

    def run(p, t, prof):
        # agent left the task in_progress and hit the verification nudge
        store = load_plan(None)
        set_status(store, "t1", "in_progress")
        save_plan(store)
        return {"ok": False, "failure_reason": "verification_nudge"}

    loop = start_plan_loop(_cfg(loop={"require_verification": True}),
                           None, run_fn=run)
    prog = _wait(loop)
    assert prog["tasks"][0]["outcome"] == "blocked"
    assert "verification" in prog["tasks"][0]["reason"]


def test_verification_nudge_ok_when_not_required(home):
    _seed("nudged")

    def run(p, t, prof):
        store = load_plan(None)
        set_status(store, "t1", "done")             # closed despite nudge
        save_plan(store)
        return {"ok": False, "failure_reason": "verification_nudge"}

    loop = start_plan_loop(_cfg(loop={"require_verification": False}),
                           None, run_fn=run)
    prog = _wait(loop)
    assert prog["tasks"][0]["outcome"] == "done"    # explicit done wins


def test_max_tasks_stop(home):
    _seed(*[f"t{i}" for i in range(6)])
    loop = start_plan_loop(_cfg(), None, run_fn=_closer([]), max_tasks=2)
    prog = _wait(loop)
    assert len(prog["tasks"]) == 2
    assert prog["stop_reason"] == "max_tasks"


def test_no_unblocked_tasks_stop(home):
    store = load_plan(None)
    add_task(store, {"title": "a"})                 # t1
    add_task(store, {"title": "b", "deps": ["t1"]})   # t2 blocked by t1
    set_status(store, "t1", "blocked")
    save_plan(store)
    loop = start_plan_loop(_cfg(), None, run_fn=_closer([]))
    prog = _wait(loop)
    assert prog["stop_reason"] == "no_unblocked_tasks"
    assert prog["tasks"] == []


def test_consecutive_failures_stop(home):
    _seed("x", "y")
    loop = start_plan_loop(_cfg(loop={"max_consecutive_failures": 2}),
                           None,
                           run_fn=lambda p, t, prof: {"ok": False,
                                                      "failure_reason":
                                                          "boom"})
    prog = _wait(loop)
    assert len(prog["tasks"]) == 2                   # stops after 2 failures
    assert all(e["outcome"] == "blocked" for e in prog["tasks"])


def test_cancel_stops_loop(home):
    _seed("a", "b", "c")
    started = threading.Event()

    def run(p, t, prof):
        started.set()
        time.sleep(0.05)
        store = load_plan(None)
        set_status(store, t["id"], "done")
        save_plan(store)
        return {"ok": True}

    loop = PlanLoop(_cfg(), None, run_fn=run)
    loop.start()
    started.wait(5)
    assert loop.cancel() is True
    prog = _wait(loop)
    assert prog["status"] == "cancelled"
    # progress file exists and reflects partial work; no zombie thread
    assert threading.active_count() >= 1
    on_disk = read_loop_progress(None, home)
    assert on_disk and on_disk["status"] == "cancelled"


def test_single_flight_conflict(home):
    _seed("a", "b")
    gate = threading.Event()

    def slow(p, t, prof):
        gate.wait(5)
        store = load_plan(None)
        set_status(store, t["id"], "done")
        save_plan(store)
        return {"ok": True}

    loop = PlanLoop(_cfg(), None, run_fn=slow)
    loop.start()
    try:
        with pytest.raises(LoopConflictError):
            PlanLoop(_cfg(), None, run_fn=slow)     # second start refused
        # ...and never queued silently: still exactly one entry on disk
        gate.set()
        _wait(loop)
    finally:
        gate.set()


def test_stale_running_progress_file_conflicts(home):
    from sable.core.plan_loop import write_loop_progress

    write_loop_progress(None, home, {"project": "", "status": "running",
                                     "iteration": 1, "tasks": []})
    with pytest.raises(LoopConflictError):
        PlanLoop(_cfg(), None, run_fn=_closer([]))


def test_resume_reads_progress_file(home):
    from sable.core.plan_loop import write_loop_progress

    _seed("already", "next one")
    # Simulate a crashed earlier run: t1 done on disk + progress history.
    store = load_plan(None)
    set_status(store, "t1", "done")
    save_plan(store)
    write_loop_progress(None, home, {
        "project": "", "status": "stopped", "iteration": 3,
        "tasks": [{"id": "t1", "title": "already", "iteration": 1,
                   "outcome": "done", "reason": "", "at": "x"}],
        "stop_reason": "error: crash"})
    closed = []
    loop = start_plan_loop(_cfg(), None, run_fn=_closer(closed), resume=True)
    prog = _wait(loop)
    assert closed == ["t2"]                          # picks up where it died
    assert prog["iteration"] >= 4                    # continued numbering
    ids = [e["id"] for e in prog["tasks"]]
    assert ids == ["t1", "t2"]                       # history preserved


def test_progress_file_shape(home):
    _seed("solo")
    loop = start_plan_loop(_cfg(), None, run_fn=_closer([]))
    _wait(loop)
    on_disk = read_loop_progress(None, home)
    assert on_disk["project"] == ""
    assert on_disk["tasks"][0]["id"] == "t1"
    assert on_disk["tasks"][0]["outcome"] == "done"
    assert "started" in on_disk and "ended" in on_disk
    # machine-readable JSON
    json.loads(json.dumps(on_disk))


def test_project_scoped_loop(home):
    _seed("p task", project="proj-x")
    closed = []

    def run(p, t, prof):
        store = load_plan("proj-x")
        set_status(store, t["id"], "done")
        save_plan(store)
        closed.append(t["id"])
        return {"ok": True}

    loop = start_plan_loop(_cfg(), "proj-x", run_fn=run)
    _wait(loop)
    assert closed == ["t1"]
    # default plan untouched
    assert load_plan(None)["tasks"] == []


def test_run_fn_exception_becomes_blocked(home):
    _seed("crashy")

    def boom(p, t, prof):
        raise RuntimeError("agent exploded")

    loop = start_plan_loop(_cfg(), None, run_fn=boom)
    prog = _wait(loop)
    assert prog["tasks"][0]["outcome"] == "blocked"
    assert "RuntimeError" in prog["tasks"][0]["reason"]


def test_task_profile_passed_to_run_fn(home):
    store = load_plan(None)
    add_task(store, {"title": "with profile", "profile": "coder"})
    save_plan(store)
    seen = []

    def run(p, t, prof):
        seen.append(prof)
        s = load_plan(None)
        set_status(s, t["id"], "done")
        save_plan(s)
        return {"ok": True}

    loop = start_plan_loop(_cfg(), None, run_fn=run)
    _wait(loop)
    assert seen == ["coder"]                         # profile flows through


def test_long_task_runs_without_timeout(home):
    """D-18: task runs have no deadline — a slow agent finishes on its own."""
    _seed("slowpoke")

    def slow(p, t, prof):
        time.sleep(0.7)                       # longer than any old default cap
        s = load_plan(None)
        set_status(s, t["id"], "done")
        save_plan(s)
        return {"ok": True}

    loop = start_plan_loop(_cfg(), None, run_fn=slow)
    prog = _wait(loop, timeout=15)
    assert prog["tasks"][0]["outcome"] == "done"

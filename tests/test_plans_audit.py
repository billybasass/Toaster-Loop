"""WP12 plans audit (2026-11 pass): store-lock concurrency, scoped graph
validation, loop server lifecycle + cancel forwarding, int-coercion 400s,
Start-task guards, and the ".json" project-id collision.

One regression test per finding; each fails against the pre-fix code in the
way described in its docstring.
"""

from __future__ import annotations

import json
import threading

import pytest

from sable.core.config import SableConfig
from sable.dashboard.app import DashboardApp


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("SABLE_HOME", str(h))
    return h


def _app(home):
    return DashboardApp(SableConfig(plans={"enabled": True}), home=home)


def _project(home, pid="proj-a"):
    d = home / "projects"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{pid}.json").write_text(json.dumps(
        {"id": pid, "name": pid, "workspace_dir": str(home),
         "agents": [], "description": "", "agent_roles": {},
         "created": "x", "updated": "x"}))
    return pid


# ------------------------------------------------- finding 1: lost updates

def test_concurrent_plan_progress_saves_do_not_lose_tasks(home):
    """plans.lock() must serialize load->modify->save: N threads each adding
    a task via the locked writer pattern all survive in the file. Before the
    fix _LOCK was declared but never taken, so late writers clobbered earlier
    saves (lost updates)."""
    from sable.core import plans as P

    P.save_plan({"project": "proj-a", "tasks": []})
    errors = []

    def add(i):
        try:
            for _ in range(30):
                with P.lock():
                    store = P.load_plan("proj-a")
                    try:
                        P.add_task(store, {"title": f"t-{i}"})
                    except P.PlanError:
                        return  # already added by this thread's retry
                    P.save_plan(store)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    ts = [threading.Thread(target=add, args=(i,)) for i in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors
    store = P.load_plan("proj-a")
    titles = {t["title"] for t in store["tasks"]}
    assert len(titles) == 8, f"lost updates: only {len(titles)} of 8 survived"


# ------------------------------------------- finding 2: scoped validation

def test_add_succeeds_despite_preexisting_dangling_dep(home):
    """A dangling dep elsewhere in the store (manual edit / keep-on-load
    policy) must not brick every future add. Pre-fix: _op_add validated the
    GLOBAL report and rejected all adds until the old edge was hand-fixed."""
    from sable.core import plans as P

    _project(home)
    # Seed a plan containing one task with a dangling dep, on disk.
    store = {"project": "proj-a", "tasks": []}
    P.add_task(store, {"id": "t1", "title": "old", "deps": ["gone"]})
    P.save_plan(store)

    app = _app(home)
    r = app.handle("POST", "/api/plans/proj-a",
                   body={"op": "add", "title": "brand new"})
    assert r.status == 200, f"add blocked by unrelated dangling dep: {r.body}"
    # ...and the pre-existing problem is still REPORTED (not silently fixed).
    g = app.handle("GET", "/api/plans/proj-a")
    assert g.body["validate"]["ok"] is False
    assert any(d["dep"] == "gone" for d in
               g.body["validate"]["dangling_deps"])


def test_add_still_refused_when_it_creates_a_cycle(home):
    """Scoping must not weaken cycle protection: an add whose deps close a
    cycle is still refused."""
    from sable.core import plans as P

    _project(home)
    store = {"project": "proj-a", "tasks": []}
    P.add_task(store, {"id": "t1", "title": "a", "deps": ["t2"]})
    P.save_plan(store)
    app = _app(home)
    # t2 depends on t1 -> cycle t1 -> t2 -> t1.
    r = app.handle("POST", "/api/plans/proj-a",
                   body={"op": "add", "id": "t2", "title": "b",
                         "deps": ["t1"]})
    assert r.status == 400 and "cycle" in r.body["error"]["message"].lower()


def test_plan_add_tool_scoped_to_new_problems(home):
    """Same scoped rule for the agent's plan_add tool."""
    from sable.core import plans as P
    from sable.core.tools import plan_add

    store = {"project": None, "tasks": []}
    P.add_task(store, {"id": "t1", "title": "old", "deps": ["gone"]})
    P.save_plan(store)
    out = plan_add(title="fresh", _plan_ctx={"project": None})
    assert out.startswith("added task"), out


# --------------------------------- finding 3: loop endpoint lifecycle

def test_loop_resolves_endpoint_once_and_closes_at_end(home, monkeypatch):
    """The default run_fn resolves the serving endpoint ONCE for the whole
    loop and exposes close(); PlanLoop must call it when the loop ends.
    Pre-fix: _resolve_endpoint+stop ran per task -> spawn mode reloaded the
    model every iteration (inside the timeout)."""
    from sable.core import plan_loop as pl

    calls = {"resolve": 0, "close": 0}

    class FakeSup:
        def stop(self):
            calls["close"] += 1

    def fake_resolve(cfg, spawn_if_needed=True):
        calls["resolve"] += 1
        return "http://x/v1", "m", FakeSup()

    monkeypatch.setattr("sable.cli._resolve_endpoint", fake_resolve)
    made = []

    class R:
        completed, failure_reason, trace_id = True, None, "tr"

    def fake_runner(cfg, base_url, model, **kw):
        made.append(kw)

        class _R:
            def run(self, prompt):
                return R()
        return _R()

    monkeypatch.setattr("sable.core.agent_runner.create_runner", fake_runner)
    monkeypatch.setattr("sable.store.recorder.make_recorder",
                        lambda *a, **k: None)
    monkeypatch.setattr("sable.core.delegation.load_project",
                        lambda *a, **k: None)

    cancel = threading.Event()
    fn = pl._default_run_fn(SableConfig(), "p", cancel_event=cancel)
    task = {"id": "t1"}
    fn("a", task, None)
    fn("b", task, None)
    # (fn carries a close() hook; assigned attribute invisible to Pylance)
    assert calls["resolve"] == 1, "endpoint must resolve once per loop"
    assert made and all(k.get("cancel_event") is not None for k in made), \
        "loop cancel_event must reach create_runner (finding 4)"
    getattr(fn, "close")()
    assert calls["close"] == 1


def test_loop_guarded_close_called_on_completion(home, monkeypatch):
    """PlanLoop._guarded invokes the run_fn's close hook when it ends."""
    from sable.core import plan_loop as pl

    closed = []

    def run_fn(prompt, task, prof):
        return {"ok": True}

    run_fn.close = lambda: closed.append(True)  # type: ignore[attr-defined]
    loop = pl.PlanLoop(SableConfig(), None, max_tasks=0, run_fn=run_fn)
    loop.start()
    loop.wait(timeout=5)
    assert closed == [True]


# ------------------------------------------------- finding 5: int coercion

def test_int_body_fields_reject_junk_with_400(home):
    _project(home)
    app = _app(home)
    r = app.handle("POST", "/api/plans/proj-a",
                   body={"op": "loop-start", "max_tasks": "lots"})
    assert r.status == 400 and "integer" in r.body["error"]["message"]
    r = app.handle("POST", "/api/plans/proj-a",
                   body={"op": "import", "text": "do a thing",
                         "num_tasks": "abc"})
    assert r.status == 400 and "integer" in r.body["error"]["message"]


# ------------------------------------------------ finding 7: .json ids

def test_project_id_ending_json_rejected(home):
    """'foo.json' would collide on disk with 'foo' (both -> foo.json)."""
    from sable.core import plans as P

    with pytest.raises(ValueError):
        P.plan_path("evil.json")
    app = _app(home)
    r = app.handle("GET", "/api/plans/evil.json")
    assert r.status == 400


# ---------------------------------------- finding 6: Start task guards

def test_start_task_rejects_terminal_and_running(home):
    from sable.core import plans as P

    _project(home)
    store = {"project": "proj-a", "tasks": []}
    P.add_task(store, {"id": "t1", "title": "done one"})
    P.set_status(store, "t1", "done")
    P.save_plan(store)
    app = _app(home)
    r = app.handle("POST", "/api/plans/proj-a",
                   body={"op": "start", "task_id": "t1"})
    assert r.status == 409 and r.body["error"]["code"] == "task_closed"

    P.set_status(P.load_plan("proj-a"), "t1", "pending")
    st = P.load_plan("proj-a")
    P.set_status(st, "t1", "in_progress")
    P.save_plan(st)
    r = app.handle("POST", "/api/plans/proj-a",
                   body={"op": "start", "task_id": "t1"})
    assert r.status == 409 and r.body["error"]["code"] == "task_running"


def test_start_task_cleans_orphan_thread_on_run_failure(home, monkeypatch):
    """If chat admission fails, the seeded thread must not linger."""
    from sable.core import plans as P

    _project(home)
    store = {"project": "proj-a", "tasks": []}
    P.add_task(store, {"id": "t1", "title": "go"})
    P.save_plan(store)
    app = _app(home)

    def boom(q, b):
        from sable.dashboard.api.base import HttpError

        raise HttpError(409, "single-flight: busy")
    monkeypatch.setattr(app, "h_chat_stream", boom)

    r = app.handle("POST", "/api/plans/proj-a",
                   body={"op": "start", "task_id": "t1"})
    assert r.status == 502
    threads = app.handle("GET", "/api/chat/threads").body["threads"]
    assert not any(t.get("task") == "t1" for t in threads), \
        "orphan seeded thread survived a failed start"


# ------------------------------------------- stale marker + force flag

def test_stale_loop_marker_requires_force(home, monkeypatch):
    """A .loop.json left 'running' by a hard kill refuses restarts unless
    force is passed; force never overrides a LIVE in-process loop."""
    from sable.core import plan_loop as pl

    def run_fn(prompt, task, prof):
        return {"ok": True}

    # Seed a stale marker for project "stale-p".
    pl.write_loop_progress("stale-p", home, {"project": "stale-p",
                                             "status": "running"})
    with pytest.raises(pl.LoopConflictError):
        pl.PlanLoop(SableConfig(), "stale-p", run_fn=run_fn)
    loop = pl.PlanLoop(SableConfig(), "stale-p", max_tasks=0,
                       run_fn=run_fn, force=True)
    loop.start()
    loop.wait(timeout=5)
    # While a live loop holds the slot, even force must refuse.
    if not getattr(pl, "_current", None):
        live = pl.PlanLoop(SableConfig(), "other-p", run_fn=run_fn)
        live.start()
        try:
            with pytest.raises(pl.LoopConflictError):
                pl.PlanLoop(SableConfig(), "stale-p", run_fn=run_fn,
                            force=True)
        finally:
            live.cancel()


def test_loop_blocked_mark_uses_lock_and_rereads(home):
    """The loop's blocked-marking re-reads under the lock (a concurrent edit
    between the post-run load and the mark is not clobbered)."""
    from sable.core import plan_loop as pl, plans as P

    store = {"project": None, "tasks": []}
    P.add_task(store, {"id": "t1", "title": "x"})
    P.save_plan(store)

    loop = object.__new__(pl.PlanLoop)   # exercise the block in isolation
    loop.project_id = None
    loop.cfg = SableConfig()
    loop.progress = {"iteration": 1, "tasks": []}
    loop.publish = lambda *a: None
    loop.cancel_event = threading.Event()
    loop.require_verification = True
    loop.max_consecutive_failures = 99
    loop.max_tasks = 10
    loop.finished = False

    def run_fn(prompt, task, prof):
        # Simulate a concurrent dashboard edit landing DURING classification:
        # the blocked-mark must not overwrite it with a stale store copy.
        with P.lock():
            s = P.load_plan(None)
            P.update_task(s, "t1", {"details": "concurrent edit"})
            P.save_plan(s)
        return {"ok": False, "failure_reason": "boom"}
    loop._run_fn = run_fn

    loop._run_all()
    final = P.load_plan(None)["tasks"][0]
    assert final["status"] == "blocked"          # loop still marks blocked...
    assert "concurrent edit" in final["details"]  # ...without losing the edit

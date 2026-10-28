"""WP12 plans: dashboard API (DashboardApp.handle direct) + SSE events."""

from __future__ import annotations

import json

import pytest

from sable.core.agent_runner import AgentResult
from sable.core.config import SableConfig
from sable.dashboard.app import DashboardApp


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("SABLE_HOME", str(h))
    return h


def _app(home, plans_enabled=True):
    cfg = SableConfig(plans={"enabled": True} if plans_enabled else {})
    return DashboardApp(cfg, home=home)


def _project(home, pid="proj-a"):
    d = home / "projects"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{pid}.json").write_text(json.dumps(
        {"id": pid, "name": pid, "workspace_dir": str(home),
         "agents": [], "description": "", "agent_roles": {},
         "created": "x", "updated": "x"}))
    return pid


# ---------------------------------------------------------------- gating

def test_routes_404_when_disabled(home):
    app = _app(home, plans_enabled=False)
    r = app.handle("GET", "/api/plans/default")
    assert r.status == 404 and r.body["error"]["code"] == "plans_disabled"
    r = app.handle("POST", "/api/plans/default", body={"op": "add",
                                                       "title": "x"})
    assert r.status == 404


def test_unknown_project_404(home):
    app = _app(home)
    r = app.handle("GET", "/api/plans/nope")
    assert r.status == 404 and r.body["error"]["code"] == "unknown_project"


def test_invalid_project_id_400(home):
    app = _app(home)
    r = app.handle("GET", "/api/plans/..evil..")
    assert r.status in (400, 404)   # pattern reject or route miss, never 500


# ------------------------------------------------------------------- CRUD

def test_add_get_update_set_status(home):
    _project(home)
    app = _app(home)
    sub = app.bus.subscribe()

    r = app.handle("POST", "/api/plans/proj-a",
                   body={"op": "add", "title": "first", "priority": "high"})
    assert r.status == 200 and r.body["task"]["id"] == "t1"
    ev = sub.poll(timeout=2)
    assert ev and ev["kind"] == "plan.updated"
    assert ev["payload"]["project"] == "proj-a"
    assert ev["payload"]["task_ids"] == ["t1"]

    r = app.handle("GET", "/api/plans/proj-a")
    assert r.status == 200
    body = r.body
    assert [t["id"] for t in body["tasks"]] == ["t1"]
    assert body["counts"]["total"] == 1
    assert body["validate"]["ok"] is True

    r = app.handle("POST", "/api/plans/proj-a",
                   body={"op": "update", "task_id": "t1",
                         "title": "renamed", "labels": ["x"]})
    assert r.status == 200 and r.body["task"]["title"] == "renamed"

    r = app.handle("POST", "/api/plans/proj-a",
                   body={"op": "set-status", "task_id": "t1",
                         "status": "done", "note": "shipped"})
    assert r.status == 200 and r.body["task"]["status"] == "done"

    r = app.handle("GET", "/api/plans/proj-a")
    t1 = r.body["tasks"][0]
    assert t1["status"] == "done" and "] shipped" in t1["details"]
    assert r.body["counts"]["done"] == 1


def test_depends_and_validate_ops(home):
    _project(home)
    app = _app(home)
    app.handle("POST", "/api/plans/proj-a", body={"op": "add", "title": "a"})
    app.handle("POST", "/api/plans/proj-a", body={"op": "add", "title": "b"})
    r = app.handle("POST", "/api/plans/proj-a",
                   body={"op": "depends", "task_id": "t2", "dep_id": "t1"})
    assert r.status == 200 and r.body["task"]["deps"] == ["t1"]
    # cycle refused with 400, machine-readable
    r = app.handle("POST", "/api/plans/proj-a",
                   body={"op": "depends", "task_id": "t1", "dep_id": "t2"})
    assert r.status == 400
    # remove-dependency path
    r = app.handle("POST", "/api/plans/proj-a",
                   body={"op": "depends", "task_id": "t2", "dep_id": "t1",
                         "remove": True})
    assert r.status == 200 and r.body["task"]["deps"] == []
    r = app.handle("POST", "/api/plans/proj-a", body={"op": "validate"})
    assert r.status == 200 and r.body["validate"]["ok"] is True


def test_unknown_task_404(home):
    _project(home)
    app = _app(home)
    r = app.handle("POST", "/api/plans/proj-a",
                   body={"op": "set-status", "task_id": "t9",
                         "status": "done"})
    assert r.status == 404


def test_unknown_op_400(home):
    _project(home)
    app = _app(home)
    r = app.handle("POST", "/api/plans/proj-a", body={"op": "frobnicate"})
    assert r.status == 400


# ----------------------------------------------------------------- import

def test_import_text_op(home):
    _project(home)
    app = _app(home)
    sub = app.bus.subscribe()
    r = app.handle("POST", "/api/plans/proj-a", body={
        "op": "import", "text": "Set up CI [priority=high]\n"
                                "Write tests [deps=t1]\n"})
    assert r.status == 200 and r.body["ok"]
    ids = [t["id"] for t in r.body["added"]]
    assert ids == ["t1", "t2"]
    ev = sub.poll(timeout=2)
    assert ev and ev["kind"] == "plan.updated"


def test_import_binary_400(home):
    _project(home)
    app = _app(home)
    r = app.handle("POST", "/api/plans/proj-a", body={
        "op": "import", "text": {"nested": "not a string"}})
    assert r.status == 400


# ------------------------------------------------------------------ start

def test_start_creates_thread_run_in_progress(home, monkeypatch):
    from fake_server import FakeOpenAIServer

    pid = _project(home)
    app = _app(home)

    class StubRunner:
        name = "stub"

        def __init__(self, cfg, base_url, model, api_key=None, recorder=None,
                     **kwargs):
            self.recorder = recorder
            self.base_url = base_url
            self.model = model

        def run(self, task, history=None):
            messages = [{"role": "user", "content": task},
                        {"role": "assistant", "content": "done"}]
            rec = self.recorder.record(
                task=task, messages=messages, backend="stub",
                model_alias=self.model, base_url=self.base_url,
                completed=True)
            return AgentResult(final_response="done", messages=messages,
                               backend="stub", tokens_in=1, tokens_out=1,
                               trace_id=rec.trace_id if rec else None)

    monkeypatch.setattr("sable.core.agent_runner.create_runner", StubRunner)
    app.handle("POST", "/api/plans/default", body={"op": "add",
                                                   "title": "ship it",
                                                   "description": "d"})
    with FakeOpenAIServer() as fs:
        (home / "serve_state.json").write_text(json.dumps({
            "mode": "attach", "base_url": fs.base_url,
            "model_name": "fake-model",
            "attach_url": fs.base_url.rsplit("/v1", 1)[0]}))
        r = app.handle("POST", "/api/plans/default",
                       body={"op": "start", "task_id": "t1"})
    assert r.status == 200, r.body
    thread_id = r.body["thread"]
    assert r.body["task"]["status"] == "in_progress"

    # thread exists, carries the task key + seeded prompt
    tr = app.handle("GET", f"/api/chat/threads/{thread_id}")
    assert tr.status == 200
    thread = tr.body["thread"]
    assert thread["task"] == "t1"
    first_user = next(m for m in thread["messages"]
                      if m.get("role") == "user")
    assert "# Task t1: ship it" in first_user["content"]

    # trace stamped with plan_task both ways
    from sable.store.db import TraceStore
    store = TraceStore(home / "buffer" / "sable.db")
    try:
        rows = store.list_traces()
        assert rows and all(rw.plan_task == "t1" for rw in rows)
    finally:
        store.close()
    # task.runs carries the thread linkage
    gr = app.handle("GET", "/api/plans/default")
    t1 = [t for t in gr.body["tasks"] if t["id"] == "t1"][0]
    assert any(run.get("thread") == thread_id for run in t1["runs"])


# -------------------------------------------------------------- loop ops

def test_loop_start_and_cancel_via_api(home, monkeypatch):
    import threading as _th

    from sable.core import plan_loop as pl

    _project(home)
    app = _app(home)
    app.handle("POST", "/api/plans/proj-a", body={"op": "add",
                                                  "title": "loop me"})
    gate = _th.Event()
    started = _th.Event()

    def slow_run(prompt, task, profile):
        started.set()
        gate.wait(5)
        from sable.core import plans as P
        store = P.load_plan("proj-a")
        P.set_status(store, task["id"], "done")
        P.save_plan(store)
        return {"ok": True}

    monkeypatch.setattr(pl, "_default_run_fn", lambda cfg, p: slow_run)
    sub = app.bus.subscribe()
    r = app.handle("POST", "/api/plans/proj-a",
                   body={"op": "loop-start", "max_tasks": 1})
    assert r.status == 200 and r.body["ok"]

    # single-flight: second start conflicts (409, never queued)
    started.wait(5)
    r2 = app.handle("POST", "/api/plans/proj-a", body={"op": "loop-start"})
    assert r2.status == 409 and r2.body["error"]["code"] == "loop_running"

    # loop progress visible through GET while running
    g = app.handle("GET", "/api/plans/proj-a")
    assert g.body["loop"]["running"] is True

    # SSE plan.loop frames arrive
    kinds = []
    _th_id = None
    for _ in range(10):
        ev = sub.poll(timeout=2)
        if ev is None:
            break
        kinds.append((ev["kind"], ev["payload"].get("status")))
        if ("plan.loop", "started") in kinds:
            break
    assert ("plan.loop", "started") in kinds

    r3 = app.handle("POST", "/api/plans/proj-a", body={"op": "loop-cancel"})
    assert r3.status == 200 and r3.body["cancelled"] is True
    gate.set()
    loop = pl.current_loop()
    if loop is not None:
        loop.wait(timeout=5)
    with pl._active_lock:
        assert pl._current is None or pl._current.finished


def test_loop_cancel_when_nothing_running(home):
    _project(home)
    app = _app(home)
    r = app.handle("POST", "/api/plans/proj-a", body={"op": "loop-cancel"})
    assert r.status == 200 and r.body["cancelled"] is False

"""Chat run lifecycle tests — cooperative cancel + run registry (the orphan bug).

Covers: the runner honors a cancel_event at its loop boundaries (pre-loop,
between iterations, between batched tool calls) and finishes with
failure_reason="cancelled" plus a well-formed message list; the dashboard
exposes GET /api/chat/runs (registry + pruning of done runs); a second
POST /api/chat/stream for a thread with an in-flight run gets a clean 409
run_in_flight carrying the active run_id (single-flight, no duplicate agent);
and cancel is excluded from both infra and model failure buckets by the
failure classifier (recorded, but no DPO material, no auto-retry).
"""

from __future__ import annotations

import inspect
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fake_server import FakeOpenAIServer  # noqa: E402
from sable.core import agent_runner as ar  # noqa: E402
from sable.core.agent_runner import SableAgentClient  # noqa: E402
from sable.core.failure_types import (  # noqa: E402
    FAILURE_CANCELLED,
    is_infra_reason,
    is_model_reason,
)
from sable.dashboard.app import DashboardApp  # noqa: E402
from sable.core.config import SableConfig  # noqa: E402

# Shared stub for DashboardApp._resolve_chat_target. The old per-test
# single-positional-arg lambdas broke the moment the method grew a kwarg
# (routing_hint did exactly this). Instead of freezing today's shape, bind
# against the REAL signature captured once at import: any call the real
# method would accept passes (future kwargs included), while unknown/typo'd
# kwarg names or too many positional args fail loudly as AssertionError.
# Stubs are installed on app INSTANCES, so the helper receives the call's
# arguments without `self` — hence the dummy self in bind().
_REAL_RESOLVE_SIG = inspect.signature(DashboardApp._resolve_chat_target)


def fake_resolve_chat_target(*args, **kwargs):
    """(base, key, model) of a dead endpoint; never dialed in these tests."""
    try:
        _REAL_RESOLVE_SIG.bind(None, *args, **kwargs)
    except TypeError as e:
        raise AssertionError(
            f"fake_resolve_chat_target called with {args!r} {kwargs!r}, "
            f"invalid for the real signature {_REAL_RESOLVE_SIG}: {e}"
        ) from None
    return ("http://127.0.0.1:1", None, "m")


@pytest.fixture(autouse=True)
def _no_sleeps(monkeypatch):
    monkeypatch.setattr(ar.SableAgentClient, "_sleep",
                        staticmethod(lambda s: None))


# ------------------------------------------------------------------ helpers

class TwoToolCallHandler(BaseHTTPRequestHandler):
    """Every POST returns ONE assistant turn with TWO tool calls (ids a, b).

    The loop executes them in order; a slow first impl lets a test set the
    cancel event mid-batch (boundary 2). Counts requests so a test can prove
    the pre-loop boundary short-circuits before any model call.
    """

    requests = 0

    def log_message(self, format, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        type(self).requests += 1
        msg = {"role": "assistant", "content": None, "tool_calls": [
            {"id": "a", "type": "function",
             "function": {"name": "list_dir", "arguments": "{}"}},
            {"id": "b", "type": "function",
             "function": {"name": "read_file",
                          "arguments": json.dumps({"path": "x.txt"})}},
        ]}
        body = json.dumps({
            "id": "x", "object": "chat.completion", "model": "m",
            "choices": [{"index": 0, "message": msg,
                         "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                      "total_tokens": 2}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _client(srv, cancel_event=None):
    client = SableAgentClient(base_url=srv.base_url, model="m",
                                max_iterations=5,
                                guardrails_enabled=False,
                                verification_nudge=False,
                                block_sensitive_files=False,
                                cancel_event=cancel_event)
    # Deterministic, workspace-free tool impls (containment never bound:
    # direct construction with no workspace_roots, per house convention).
    client.tool_impls["list_dir"] = lambda **kw: "dir listing ok"
    client.tool_impls["read_file"] = lambda **kw: "file contents ok"
    return client


def _assert_well_formed(messages):
    """Every assistant tool_call has exactly one tool result for its id."""
    calls = []
    results = []
    for m in messages:
        if m.get("role") == "assistant":
            calls += [tc.get("id") for tc in (m.get("tool_calls") or [])]
        elif m.get("role") == "tool":
            results.append(m.get("tool_call_id"))
    assert calls == results, f"unbalanced tool messages: {calls} vs {results}"


# ------------------------------------------------------------ runner cancel

def test_cancel_preloop_never_calls_model():
    TwoToolCallHandler.requests = 0
    ev = threading.Event()
    ev.set()  # stop pressed before the run even starts
    with FakeOpenAIServer(handler_cls=TwoToolCallHandler) as srv:
        res = _client(srv, cancel_event=ev).run("do something")
    assert res.completed is False
    assert res.failure_reason == FAILURE_CANCELLED
    assert TwoToolCallHandler.requests == 0, "cancel must not touch the model"


def test_cancel_between_iterations_stops_before_second_call():
    TwoToolCallHandler.requests = 0
    ev = threading.Event()
    with FakeOpenAIServer(handler_cls=TwoToolCallHandler) as srv:
        client = _client(srv, cancel_event=ev)
        # Deterministic boundary: stop pressed the instant the first model
        # call returns — after its batch executes, before the second call.
        orig = client.retry_chat
        def cancel_after_first(messages, *a, **k):
            if not ev.is_set():
                ev.set()
            return orig(messages, *a, **k)
        client.retry_chat = cancel_after_first
        res = client.run("do something")
    assert res.completed is False
    assert res.failure_reason == FAILURE_CANCELLED
    # Exactly one model call: the first turn's batch completed, the second
    # iteration was refused at the pre-model-call boundary.
    assert TwoToolCallHandler.requests == 1
    _assert_well_formed(res.messages)


def test_cancel_midbatch_skips_remaining_tool_calls():
    TwoToolCallHandler.requests = 0
    ev = threading.Event()
    with FakeOpenAIServer(handler_cls=TwoToolCallHandler) as srv:
        client = _client(srv, cancel_event=ev)
        # Slow first tool: the cancel lands WHILE it runs, so the loop
        # stops at the between-calls boundary and must skip call "b".
        client.tool_impls["list_dir"] = lambda **kw: (time.sleep(0.3),
                                                      "dir listing ok")[1]

        def cancel_midtool():
            time.sleep(0.1)
            ev.set()

        t = threading.Thread(target=cancel_midtool, daemon=True)
        t.start()
        res = client.run("do something")
        t.join()
    assert res.completed is False
    assert res.failure_reason == FAILURE_CANCELLED
    _assert_well_formed(res.messages)
    skipped = [m for m in res.messages if m.get("role") == "tool"
               and "skipped" in str(m.get("content"))]
    assert len(skipped) == 1
    assert skipped[0]["tool_call_id"] == "b"


def test_no_cancel_event_keeps_old_behavior():
    """cancel_event=None (CLI, stubs, delegation children): untouched."""
    TwoToolCallHandler.requests = 0
    with FakeOpenAIServer(handler_cls=TwoToolCallHandler) as srv:
        client = _client(srv)  # no event
        # No cancel ever set: run must burn through max_iterations like before.
        res = client.run("do something")
    assert res.completed is False
    assert res.failure_reason == "max_iterations"
    assert TwoToolCallHandler.requests == 5
    _assert_well_formed(res.messages)


# ---------------------------------------------------- dashboard endpoints

@pytest.fixture()
def app(tmp_path, monkeypatch):
    home = tmp_path / "uf-lifecycle"
    monkeypatch.setenv("SABLE_HOME", str(home))
    a = DashboardApp(SableConfig(), home=home)
    return a


def _seed_run(app, run_id, thread, done=False, status="running"):
    ev = threading.Event()
    ev.set() if done else None
    app._chat_runs[run_id] = {"cancel": ev, "done": done, "thread": thread,
                              "model": "m", "profile": None,
                              "status": status}


def test_runs_registry_lists_inflight_and_prunes_done(app):
    _seed_run(app, "aaaa", "t1")            # in flight
    _seed_run(app, "bbbb", "t2", done=True)  # finished
    r = app.handle("GET", "/api/chat/runs")
    assert r.status == 200
    runs = r.body["runs"]
    assert [x["run_id"] for x in runs] == ["aaaa"]
    assert runs[0]["thread"] == "t1" and runs[0]["status"] == "running"
    assert "bbbb" not in app._chat_runs, "done runs must be pruned"


def test_stream_second_post_same_thread_gets_409_with_run_id(app, monkeypatch):
    # Bypass model resolution: the single-flight check happens after the
    # target is resolved, so stub it with a dead (never-dialed) URL.
    monkeypatch.setattr(app, "_resolve_chat_target", fake_resolve_chat_target)
    _seed_run(app, "active1", "t1")
    resp = app.handle("POST", "/api/chat/stream",
                      body={"prompt": "again", "thread": "t1"})
    body = json.loads(resp.json())
    assert resp.status == 409
    err = body["error"]
    assert err["code"] == "run_in_flight"
    assert err["run_id"] == "active1", "UI needs the id to adopt/cancel"


def test_stream_threadless_post_not_subject_to_single_flight(app, monkeypatch):
    """Runs without a thread id (fresh chats) must not be blocked by a
    thread's in-flight run — and must not themselves block anything."""
    monkeypatch.setattr(app, "_resolve_chat_target", fake_resolve_chat_target)
    _seed_run(app, "active1", "t1")
    # A different thread is free to run.
    resp = app.handle("POST", "/api/chat/stream",
                      body={"prompt": "hi", "thread": "other"})
    assert resp.status == 200  # started (its worker will fail harmlessly)


def test_cancelled_is_neither_infra_nor_model():
    assert not is_infra_reason(FAILURE_CANCELLED)
    assert not is_model_reason(FAILURE_CANCELLED)
    assert not is_infra_reason("backend_error:" + FAILURE_CANCELLED)


# ------------------------------------------------- shutdown drain (step 3)

def test_shutdown_drains_cooperative_runs(app):
    """A run whose worker watches its cancel event finishes inside the grace
    window: shutdown sets every cancel flag and waits for done."""
    _seed_run(app, "slow1", "t1")

    def _worker():
        run = app._chat_runs["slow1"]
        while not run["cancel"].wait(timeout=0.05):
            pass
        run["done"] = True

    threading.Thread(target=_worker, daemon=True).start()
    n = app.shutdown_chat_runs(grace_s=3.0)
    assert n == 1
    assert app._chat_runs["slow1"]["done"] is True


def test_shutdown_marks_stubborn_runs_interrupted(app):
    """A run that ignores cancel must not hang shutdown: after the grace
    window it is marked interrupted (documenting the lost trace)."""
    _seed_run(app, "stub1", "t2")
    n = app.shutdown_chat_runs(grace_s=0.3)
    assert n == 1
    assert app._chat_runs["stub1"]["status"] == "interrupted"
    assert app._chat_runs["stub1"]["cancel"].is_set()


def test_shutdown_with_no_runs_is_noop(app):
    assert app.shutdown_chat_runs(grace_s=0.1) == 0


def test_unstarted_stream_leaves_no_phantom_registry_entry(app, monkeypatch):
    """The registry entry is created on the stream's first iteration. A POST
    whose SSE never starts (admission rejection path) must not leave a run
    that single-flight would 409 against forever."""
    monkeypatch.setattr(app, "_resolve_chat_target", fake_resolve_chat_target)
    resp = app.handle("POST", "/api/chat/stream",
                      body={"prompt": "hi", "thread": "t9"})
    assert resp.status == 200 and resp.stream is not None
    # Never iterate the stream (simulates server-side rejection before write).
    resp.stream.close()
    runs = app.handle("GET", "/api/chat/runs").body["runs"]
    assert runs == [], "unstarted streams must not occupy the registry"

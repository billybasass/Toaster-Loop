"""WP2: delegated agent tasks appear as sub-threads in the threads list.

Covers the acceptance shapes:
  * dispatch WITH a parent chat thread creates a real child thread record
    (<home>/chat/dlg-<delegation_id>.json) carrying parent_thread /
    delegation_id / agent (and it nests under the parent in the listing);
  * opening the sub-thread mid-run re-attaches to the live stream through the
    existing chat-run machinery (registry + frame buffer + run.done close),
    and the finished transcript is persisted to the child file;
  * threadless dispatches keep today's behavior (no child record at all);
  * deleting a parent deletes its children.
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from sable.core import delegation, delegation_store
from sable.core.config import SableConfig


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("SABLE_HOME", str(h))
    return h


@pytest.fixture()
def _clean(home):
    delegation.reset_state()
    delegation.register_child_run_bridge(None)
    yield
    delegation.reset_state()
    delegation.register_child_run_bridge(None)


def _cfg(**kw) -> SableConfig:
    cfg = SableConfig(**kw)
    cfg.profiles = [{"name": "coder", "system_prompt": "code",
                     "tools": ["read_file"]}]
    return cfg


def _ctx(cfg, thread=None):
    return {"cfg": cfg, "base_url": "http://127.0.0.1:1/v1", "model": "m",
            "profile": "parent", "project": None, "thread": thread,
            "run_id": "run-parent", "depth": 0}


def _wait_until(fn, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if fn():
            return True
        time.sleep(0.03)
    return False


@pytest.fixture()
def dash(home):
    """DashboardApp (registers the child-run bridge in __init__)."""
    from sable.dashboard.app import DashboardApp

    return DashboardApp(SableConfig(), home=home)


def _sse(text: str):
    out, kind = [], None
    for line in text.split("\n"):
        if line.startswith("event:"):
            kind = line[6:].strip()
        elif line.startswith("data:") and kind:
            try:
                out.append((kind, json.loads(line[5:].strip())))
            except json.JSONDecodeError:
                pass
            kind = None
    return out


# --------------------------------------------------------------------------
# Child thread record creation
# --------------------------------------------------------------------------

def test_dispatch_with_thread_creates_child_record(home, _clean, monkeypatch):
    """Acceptance #1: dispatch WITH thread context materializes a child
    record with parent_thread/delegation_id/agent set, and the listing nests
    it under the parent row."""
    cfg = _cfg()
    monkeypatch.setattr(delegation, "_run_child",
                        lambda *a, **k: a[5].update(
                            {"completed": True, "result": "child done",
                             "child_trace_id": "tr-child1"}))
    (home / "chat").mkdir(exist_ok=True)
    payload = json.loads(delegation.execute_delegation(
        "coder", "summarize the deploy log", _ctx=_ctx(cfg, thread="t-p1")))
    assert payload["status"] == "dispatched"
    did = payload["delegation_id"]

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        row = delegation_store.get_delegation(did)
        if row and row["status"] in ("completed", "failed", "cancelled"):
            break
        time.sleep(0.03)

    p = home / "chat" / f"dlg-{did}.json"
    assert p.is_file(), "child thread record must exist next to the parent"
    t = json.loads(p.read_text())
    assert t["id"] == f"dlg-{did}"
    assert t["parent_thread"] == "t-p1"
    assert t["delegation_id"] == did
    assert t["agent"] == "coder"
    assert t["delegation_status"] == "completed"
    # The child's own final reply is persisted by the bridge when present;
    # without a dashboard the record still exists with the seeded task turn.
    assert t["messages"][0]["role"] == "user"
    assert "summarize the deploy log" in t["messages"][0]["content"]


def test_threadless_dispatch_creates_no_child(home, _clean, monkeypatch):
    """CLI/threadless delegations keep today's behavior: no child record."""
    cfg = _cfg()
    monkeypatch.setattr(delegation, "_run_child",
                        lambda *a, **k: a[5].update(
                            {"completed": True, "result": "x"}))
    payload = json.loads(delegation.execute_delegation(
        "coder", "no thread here", _ctx=_ctx(cfg, thread=None)))
    did = payload["delegation_id"]
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        row = delegation_store.get_delegation(did)
        if row and row["status"] in ("completed", "failed", "cancelled"):
            break
        time.sleep(0.03)
    assert not list((home / "chat").glob("dlg-*.json")) \
        if (home / "chat").is_dir() else True


# --------------------------------------------------------------------------
# Threads listing nests children (additive shape, legacy files unaffected)

def test_threads_list_nests_children(home, _clean, dash, monkeypatch):
    cfg = _cfg()
    monkeypatch.setattr(delegation, "_run_child",
                        lambda *a, **k: a[5].update(
                            {"completed": True, "result": "ok"}))
    dash.handle("PUT", "/api/chat/threads/t-nest",
                body={"messages": [{"role": "user", "content": "hi"}]})
    payload = json.loads(delegation.execute_delegation(
        "coder", "nested child task", _ctx=_ctx(cfg, thread="t-nest")))
    did = payload["delegation_id"]
    assert _wait_until(
        lambda: (home / "chat" / f"dlg-{did}.json").is_file())

    rows = {r["id"]: r for r in
            dash.handle("GET", "/api/chat/threads").body["threads"]}
    child = rows[f"dlg-{did}"]
    assert child["parent_thread"] == "t-nest"
    assert child["delegation_id"] == did
    assert child["agent"] == "coder"
    # Parent + legacy threads read the additive keys as None (no migration).
    assert rows["t-nest"]["parent_thread"] is None

    # A client save against the child must not strip the lineage keys.
    r = dash.handle("PUT", f"/api/chat/threads/dlg-{did}",
                    body={"messages": [{"role": "user", "content": "x"}]})
    assert r.status == 200
    t = json.loads((home / "chat" / f"dlg-{did}.json").read_text())
    assert t["parent_thread"] == "t-nest"
    assert t["delegation_id"] == did


# --------------------------------------------------------------------------
# Mid-run open re-attaches to the live stream

def test_midrun_open_reattaches_and_persists(home, _clean, dash, monkeypatch):
    """Acceptance #3: with a bridge (dashboard) attached, a child run streams
    through the chat-run machinery — opening mid-run replays buffered frames
    and follows live, then the final transcript lands in the child file."""
    cfg = _cfg()
    release = threading.Event()

    def fake_child(cfg_, prof, task, ctx, delegation_id, out, plan_task=None,
                   cancel_event=None):
        sink = ctx.get("sink")
        assert callable(sink), "bridge must attach a sink for threaded runs"
        sink("message.delta", {"text": "child says "})
        sink("message.delta", {"text": "hi"})
        release.wait(timeout=10)
        out.update({"completed": True, "result": "child says hi",
                    "child_trace_id": "tr-live"})

    monkeypatch.setattr(delegation, "_run_child", fake_child)
    payload = json.loads(delegation.execute_delegation(
        "coder", "stream for me", _ctx=_ctx(cfg, thread="t-live")))
    did = payload["delegation_id"]
    run_id = f"dlg-{did}"

    assert _wait_until(lambda: run_id in dash._chat_runs), \
        "bridge must register the child run when it starts"
    assert not dash._chat_runs[run_id].get("done")

    # Open mid-run: re-attach replays every buffered frame...
    resp = dash.handle("GET", f"/api/chat/runs/{run_id}/stream")
    assert resp.status == 200
    it = iter(resp.stream)
    frames = [next(it), next(it), next(it)]  # run.started + 2 deltas
    parsed = _sse("".join(frames))
    kinds = [k for k, _ in parsed]
    assert "run.started" in kinds
    text = "".join(d.get("text", "") for k, d in parsed
                   if k == "message.delta")
    assert text.startswith("child says ")

    # ...then follow live until the run closes.
    release.set()
    tail = []
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            f = next(it)
        except StopIteration:
            break
        tail.append(f)
        if "run.done" in f or "run.error" in f:
            break
    assert any("run.done" in f for f in tail), \
        "attached reader must see run.done live"

    assert _wait_until(
        lambda: dash._chat_runs.get(run_id, {}).get("done"))
    # Final transcript persisted to the child thread file (WP1 atomic path).
    t = json.loads((home / "chat" / f"dlg-{did}.json").read_text())
    roles = [m["role"] for m in t["messages"]]
    assert roles == ["user", "assistant"]
    assert t["messages"][1]["content"] == "child says hi"
    assert t["delegation_status"] == "completed"


# --------------------------------------------------------------------------
# Lifecycle: parent delete cascades to children

def test_parent_delete_cascades_children(home, _clean, dash, monkeypatch):
    cfg = _cfg()
    monkeypatch.setattr(delegation, "_run_child",
                        lambda *a, **k: a[5].update(
                            {"completed": True, "result": "ok"}))
    dash.handle("PUT", "/api/chat/threads/t-casc",
                body={"messages": []})
    payload = json.loads(delegation.execute_delegation(
        "coder", "cascade me", _ctx=_ctx(cfg, thread="t-casc")))
    did = payload["delegation_id"]
    child = home / "chat" / f"dlg-{did}.json"
    assert _wait_until(lambda: child.is_file())

    r = dash.handle("DELETE", "/api/chat/threads/t-casc")
    assert r.status == 200
    assert not (home / "chat" / "t-casc.json").exists()
    assert not child.exists(), "deleting the parent deletes its children"

    # ?keep_children=1 opts out (second parent + child, first pair gone).
    dash.handle("PUT", "/api/chat/threads/t-keep", body={"messages": []})
    payload2 = json.loads(delegation.execute_delegation(
        "coder", "keep me", _ctx=_ctx(cfg, thread="t-keep")))
    did2 = payload2["delegation_id"]
    child2 = home / "chat" / f"dlg-{did2}.json"
    assert _wait_until(lambda: child2.is_file())
    r = dash.handle("DELETE", "/api/chat/threads/t-keep",
                    query={"keep_children": "1"})
    assert r.status == 200
    assert not (home / "chat" / "t-keep.json").exists()
    assert child2.exists(), "keep_children must leave the child file"


def test_child_status_badges_follow_transitions(home, _clean, dash,
                                                monkeypatch):
    """The child record's delegation_status tracks queued -> running -> done
    so the UI badge is data-driven (bus events repaint; file is durable)."""
    cfg = _cfg()
    gate = threading.Event()

    def fake_child(cfg_, prof, task, ctx, delegation_id, out, plan_task=None,
                   cancel_event=None):
        p = home / "chat" / f"dlg-{delegation_id}.json"
        assert json.loads(p.read_text())["delegation_status"] == "running"
        gate.wait(timeout=10)
        out.update({"completed": True, "result": "ok"})

    monkeypatch.setattr(delegation, "_run_child", fake_child)
    payload = json.loads(delegation.execute_delegation(
        "coder", "badge me", _ctx=_ctx(cfg, thread="t-badge2")))
    did = payload["delegation_id"]
    child = home / "chat" / f"dlg-{did}.json"
    assert _wait_until(lambda: child.is_file())
    gate.set()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        row = delegation_store.get_delegation(did)
        if row and row["status"] == "completed":
            break
        time.sleep(0.03)
    assert _wait_until(lambda: json.loads(
        child.read_text())["delegation_status"] == "completed")

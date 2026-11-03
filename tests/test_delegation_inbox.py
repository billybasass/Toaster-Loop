"""D-17 Phase 1 + Phase 2 + Phase 3: durable store, dispatcher, liveness,
inbox wake, and polish (terminal-row janitor, busy-poll guardrail,
pending-inbox surfacing, wake-run trace metadata).

Covers the plans/delegation-inbox.md §9 test plan rows for Phase 1 (store
round-trip / FIFO claim atomicity / take_pending flips once / reconcile-on-
startup per status; dispatcher pool bound + saturation payload; D-18 removed
the liveness watchdog (children have no timeout — old rows are never wedged);
the runner `_heartbeat` hook; `wait=True` vs async dispatch
payloads; config defaults/overrides; check_delegation surviving a registry
reset) and Phase 2 (wake hook registry; watcher delivers when thread idle and
defers when busy; drain-on-user-message prepends the block + marks delivered;
the wake run persists its transcript, carries inbox_wake metadata, and emits
the `delegation.wake` bus event; concurrent take_pending delivers exactly once;
reconcile leaves undelivered entries for the next wake; the ping-pong streak
guard). Watcher cadence is faked by calling `run_wake_pass`/`try_deliver`
directly — no sleeping on `wake_retry_seconds`.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import pytest

from sable.core import delegation, delegation_store
from sable.core.config import SableConfig, load_config

from fake_server import FakeOpenAIServer  # noqa: E402


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("SABLE_HOME", str(h))
    return h


@pytest.fixture()
def _clean(home):
    delegation.reset_state()
    yield
    delegation.reset_state()


def _cfg(**kw) -> SableConfig:
    cfg = SableConfig(**kw)
    cfg.profiles = [{"name": "coder", "system_prompt": "code",
                     "tools": ["read_file"]}]
    return cfg


def _ctx(cfg, thread=None):
    return {"cfg": cfg, "base_url": "http://127.0.0.1:1/v1", "model": "m",
            "profile": "parent", "project": None, "thread": thread,
            "run_id": "run-parent", "depth": 0}


def _iso(offset_s: float = 0.0) -> str:
    dt = datetime.now(timezone.utc) + timedelta(seconds=offset_s)
    return dt.isoformat(timespec="seconds")


def _rec(did: str, status: str = "queued", **over) -> Dict[str, Any]:
    row = {"delegation_id": did, "status": status, "agent": "coder",
           "task_preview": "t", "created_ts": _iso(),
           "liveness_timeout": 600.0}
    row.update(over)
    return row


def _fake_child(out_fields, block_on=None, ignore_cancel=False):
    """_run_child stand-in: settles immediately with out_fields, or blocks on
    an Event until it is set (optionally honoring cancel_event)."""
    def fake(cfg_, prof, task, ctx, delegation_id, out, plan_task=None,
             cancel_event=None):
        if block_on is not None:
            if ignore_cancel:
                block_on.wait(timeout=10)
            else:
                deadline = time.monotonic() + 10
                while (time.monotonic() < deadline and not block_on.is_set()
                       and not (cancel_event and cancel_event.is_set())):
                    time.sleep(0.02)
        out.update(out_fields)
    return fake


def _wait_until(fn, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if fn():
            return True
        time.sleep(0.03)
    return False


# ---------------------------------------------------------------------------
# Store: round-trip, whitelist, list
# ---------------------------------------------------------------------------

def test_store_roundtrip_and_update_whitelist(home):
    delegation_store.insert_delegation(_rec("d1"))
    row = delegation_store.get_delegation("d1")
    assert row is not None and row["status"] == "queued"
    assert row["agent"] == "coder" and row["inbox_state"] == "none"

    assert delegation_store.update_delegation("d1", status="running",
                                              started_ts=_iso()) is True
    assert delegation_store.get_delegation("d1")["status"] == "running"

    with pytest.raises(ValueError, match="unknown delegation column"):
        delegation_store.update_delegation("d1", status_typo="x")

    # unknown id -> None; update on missing row -> False
    assert delegation_store.get_delegation("nope") is None
    assert delegation_store.update_delegation("nope", status="running") is False


def test_store_tolerant_read_of_odd_values(home):
    delegation_store.insert_delegation(_rec("d-odd"))
    delegation_store.update_delegation(
        "d-odd", status="completed", result="\x00\xff weird \u00e9 bytes",
        settled_ts=_iso())
    row = delegation_store.get_delegation("d-odd")
    assert row["status"] == "completed" and "weird" in row["result"]


def test_store_list_filters(home):
    delegation_store.insert_delegation(_rec("a1", status="queued"))
    delegation_store.insert_delegation(_rec("a2", status="completed",
                                            thread="th-1", project="p1"))
    assert {r["delegation_id"] for r in
            delegation_store.list_delegations(statuses=("queued",))} == {"a1"}
    assert {r["delegation_id"] for r in
            delegation_store.list_delegations(thread="th-1")} == {"a2"}
    assert {r["delegation_id"] for r in
            delegation_store.list_delegations(project="p1")} == {"a2"}


# ---------------------------------------------------------------------------
# Store: FIFO claim + atomicity under threads
# ---------------------------------------------------------------------------

def test_claim_next_queued_fifo(home):
    # created_ts order decides (same-second ties are fine — FIFO within the
    # index order); insert oldest first with explicit timestamps.
    delegation_store.insert_delegation(_rec("old", created_ts=_iso(-30)))
    delegation_store.insert_delegation(_rec("mid", created_ts=_iso(-20)))
    delegation_store.insert_delegation(_rec("new", created_ts=_iso(-10)))

    first = delegation_store.claim_next_queued()
    assert first["delegation_id"] == "old" and first["status"] == "running"
    assert first["started_ts"]
    second = delegation_store.claim_next_queued()
    assert second["delegation_id"] == "mid"
    third = delegation_store.claim_next_queued()
    assert third["delegation_id"] == "new"
    assert delegation_store.claim_next_queued() is None


def test_claim_atomic_under_threads(home):
    n = 12
    for i in range(n):
        delegation_store.insert_delegation(
            _rec(f"c{i:02d}", created_ts=_iso(-100 + i)))
    claimed: list = []
    lock = threading.Lock()

    def grab():
        while True:
            row = delegation_store.claim_next_queued()
            if row is None:
                return
            with lock:
                claimed.append(row["delegation_id"])

    ts = [threading.Thread(target=grab, daemon=True) for _ in range(6)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=15)
    assert sorted(claimed) == sorted(f"c{i:02d}" for i in range(n)), \
        "every row claimed exactly once across racing threads"


# ---------------------------------------------------------------------------
# Store: inbox mechanics
# ---------------------------------------------------------------------------

def test_pending_inbox_take_flips_exactly_once(home):
    delegation_store.insert_delegation(
        _rec("i1", status="completed", thread="th-A", inbox_state="pending",
             settled_ts=_iso(-5)))
    delegation_store.insert_delegation(
        _rec("i2", status="failed", thread="th-A", inbox_state="pending",
             error="boom", settled_ts=_iso(-10)))
    delegation_store.insert_delegation(
        _rec("i3", status="completed", thread="th-B", inbox_state="pending"))
    delegation_store.insert_delegation(
        _rec("i4", status="completed", thread=None, inbox_state="none"))

    assert set(delegation_store.pending_inbox_threads()) == {"th-A", "th-B"}
    batch = delegation_store.take_pending_inbox("th-A")
    # ordered by settled_ts: i1 (older? -5 vs -10) -> i2 first
    assert [r["delegation_id"] for r in batch] == ["i2", "i1"]
    # second take delivers nothing (flipped exactly once)
    assert delegation_store.take_pending_inbox("th-A") == []
    assert delegation_store.get_delegation("i1")["inbox_state"] == "delivered"
    delegation_store.mark_delivered_run(["i1", "i2"], "wake-run-9")
    assert delegation_store.get_delegation("i1")["delivered_run_id"] \
        == "wake-run-9"


# ---------------------------------------------------------------------------
# Store: reconcile_on_startup (crash simulation)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("status", ["queued", "running", "cancelling"])
def test_reconcile_marks_non_terminal_interrupted(home, status):
    delegation_store.insert_delegation(
        _rec(f"r-{status}", status=status, thread="th-1"))
    delegation_store.insert_delegation(  # terminal rows untouched
        _rec("r-done", status="completed"))
    n = delegation_store.reconcile_on_startup()
    assert n == 1
    for s in (status,):
        row = delegation_store.get_delegation(f"r-{s}")
        assert row["status"] == "interrupted"
        assert "restarted while the child was in flight" in row["error"]
        assert row["inbox_state"] == "pending"  # has a thread
    assert delegation_store.get_delegation("r-done")["status"] == "completed"
    # idempotent: second run sweeps nothing
    assert delegation_store.reconcile_on_startup() == 0


def test_reconcile_threadless_row_stays_inbox_none(home):
    delegation_store.insert_delegation(_rec("solo", status="running"))
    assert delegation_store.reconcile_on_startup() == 1
    row = delegation_store.get_delegation("solo")
    assert row["status"] == "interrupted"
    assert row["inbox_state"] == "none"


# ---------------------------------------------------------------------------
# Dispatcher: bounded pool + saturation
# ---------------------------------------------------------------------------

def test_dispatcher_pool_bound_serializes(home, monkeypatch, _clean):
    """max_concurrent=1: two dispatches never run their children at once."""
    cfg = _cfg(delegation_max_concurrent=1)
    state = {"active": 0, "max_active": 0}
    slock = threading.Lock()

    def counting_child(cfg_, prof, task, ctx, delegation_id, out,
                       plan_task=None, cancel_event=None):
        with slock:
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
        time.sleep(0.25)
        with slock:
            state["active"] -= 1
        out.update({"completed": True, "result": "ok",
                    "child_trace_id": None, "failure_reason": None})

    monkeypatch.setattr(delegation, "_run_child", counting_child)
    payloads = [json.loads(delegation.execute_delegation(
        "coder", f"task {i}", _ctx=_ctx(cfg))) for i in (1, 2)]
    assert all(p["status"] == "dispatched" for p in payloads), payloads
    ids = [p["delegation_id"] for p in payloads]

    assert _wait_until(lambda: all(
        json.loads(delegation.check_delegation_status(d))["status"]
        == "completed" for d in ids), timeout=10)
    assert state["max_active"] == 1, \
        f"pool bound violated: saw {state['max_active']} concurrent children"


def test_dispatcher_saturation_error_payload(home, _clean):
    cfg = _cfg(delegation_queue_max=1)
    # one queued row already in the store fills the queue
    delegation_store.insert_delegation(_rec("blocker", created_ts=_iso(-5)))
    out = json.loads(delegation.execute_delegation(
        "coder", "one too many", _ctx=_ctx(cfg)))
    assert out["status"] == "failed"
    assert out["code"] == "dispatcher_saturated"
    assert "queued; retry shortly" in out["error"]


# ---------------------------------------------------------------------------
# Liveness watchdog is GONE (D-18): children have no timeout and are never
# wedge-detected. These former watchdog tests now assert the opposite: an old,
# silent row stays untouched forever.
# ---------------------------------------------------------------------------

def test_aged_queued_row_never_wedged(home):
    cfg = _cfg()
    delegation_store.insert_delegation(
        _rec("starved", created_ts=_iso(-700), thread="th-1"))
    row = delegation_store.get_delegation("starved")
    assert row["status"] == "queued"          # no watchdog -> stays queued


def test_aged_running_row_never_cancelled(home):
    cfg = _cfg()
    delegation_store.insert_delegation(
        _rec("alive", status="running", started_ts=_iso(-700),
             last_activity_ts=_iso(-400)))
    assert delegation_store.get_delegation("alive")["status"] == "running"


def test_touch_throttles_store_writes(home, monkeypatch):
    """§3.1: cache ticks are free; store writes throttled to >=2s apart."""
    cfg = _cfg()
    monkeypatch.setattr(delegation, "_run_child",
                        _fake_child({}, block_on=threading.Event()))
    # stand up a record in the cache without dispatching a child thread
    rec = {"delegation_id": "tick1", "status": "running", "_settled": False,
           "_cfg_ref": cfg}
    delegation._store_delegation(rec)
    delegation_store.insert_delegation(_rec("tick1", status="running"))

    writes = []
    real_update = delegation_store.update_delegation

    def spy(did, cfg=None, **f):
        writes.append(f)
        return real_update(did, cfg=cfg, **f)
    monkeypatch.setattr(delegation_store, "update_delegation", spy)

    delegation.touch("tick1")   # first tick: store write
    before = len(writes)
    delegation.touch("tick1")   # throttled: cache only
    assert len(writes) == before
    assert delegation.get_delegation("tick1")["last_activity_ts"]
    monkeypatch.setattr(delegation, "TOUCH_THROTTLE_S", 0.0)
    delegation.touch("tick1")   # throttle off -> write again
    assert len(writes) == before + 1


# ---------------------------------------------------------------------------
# Runner heartbeat hook (§3.1)
# ---------------------------------------------------------------------------

def test_runner_heartbeat_hook_per_iteration_and_tool(home):
    """The loop ticks _heartbeat at each iteration boundary and after each
    tool call; default None changes nothing."""
    from http.server import BaseHTTPRequestHandler

    from sable.core.agent_runner import SableAgentClient
    from tests.fake_server import FakeOpenAIServer

    calls = {"n": 0}
    n_req = {"i": 0}

    class ToolOnceThenAnswer(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(
                int(self.headers.get("Content-Length", 0))))
            msgs = body.get("messages") or []
            if not any(m.get("role") == "tool" for m in msgs):
                reply = {"role": "assistant", "content": None,
                         "tool_calls": [{
                             "id": "t1", "type": "function",
                             "function": {"name": "noop",
                                          "arguments": "{}"}}]}
            else:
                reply = {"role": "assistant", "content": "done"}
            out = json.dumps({"choices": [{"index": 0, "message": reply,
                                           "finish_reason": "stop"}],
                              "usage": {}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
            n_req["i"] += 1

    with FakeOpenAIServer(handler_cls=ToolOnceThenAnswer) as srv:
        client = SableAgentClient(base_url=srv.base_url, model="m",
                                  tool_impls={"noop": lambda **k: "ok"})
        # default: hook absent/None -> plain run works untouched
        assert client.run("go").completed

        ticks = []
        client2 = SableAgentClient(base_url=srv.base_url, model="m",
                                   tool_impls={"noop": lambda **k: "ok"})
        client2._heartbeat = lambda: ticks.append(1)
        assert client2.run("go").completed
    # >= 1 tick per iteration (2) + 1 after the tool call
    assert len(ticks) >= 3, ticks


# ---------------------------------------------------------------------------
# wait=True vs async dispatch payloads + inbox population
# ---------------------------------------------------------------------------

def test_async_dispatch_payload_and_inbox_pending(home, monkeypatch, _clean):
    cfg = _cfg()
    monkeypatch.setattr(delegation, "_run_child", _fake_child(
        {"completed": True, "result": "child answer",
         "child_trace_id": "tr-kid", "failure_reason": None}))
    payload = json.loads(delegation.execute_delegation(
        "coder", "background thing", _ctx=_ctx(cfg, thread="th-7")))
    assert payload["status"] == "dispatched"
    did = payload["delegation_id"]
    assert "check_delegation" in payload["note"]

    assert _wait_until(lambda: json.loads(
        delegation.check_delegation_status(did))["status"] == "completed")
    poll = json.loads(delegation.check_delegation_status(did))
    assert poll["result"] == "child answer"
    # Phase-2 hook: settled async delegation with a thread -> inbox pending
    row = delegation_store.get_delegation(did)
    assert row["inbox_state"] == "pending" and row["thread"] == "th-7"
    assert row in delegation_store.take_pending_inbox("th-7")


def test_async_threadless_note_and_no_inbox(home, monkeypatch, _clean):
    cfg = _cfg()
    monkeypatch.setattr(delegation, "_run_child", _fake_child(
        {"completed": True, "result": "x", "child_trace_id": None,
         "failure_reason": None}))
    payload = json.loads(delegation.execute_delegation(
        "coder", "no thread", _ctx=_ctx(cfg)))  # thread=None
    assert payload["status"] == "dispatched"
    assert payload["note"] == "no chat thread context — poll check_delegation"
    did = payload["delegation_id"]
    assert _wait_until(lambda: json.loads(
        delegation.check_delegation_status(did))["status"] == "completed")
    assert delegation_store.get_delegation(did)["inbox_state"] == "none"


def test_async_default_false_restores_sync(home, monkeypatch, _clean):
    cfg = _cfg(delegation_async_default=False)
    monkeypatch.setattr(delegation, "_run_child", _fake_child(
        {"completed": True, "result": "sync answer",
         "child_trace_id": None, "failure_reason": None}))
    payload = json.loads(delegation.execute_delegation(
        "coder", "legacy", _ctx=_ctx(cfg)))
    assert payload["status"] == "completed" and payload["result"] == "sync answer"
    # explicit wait=False still forces async even under the sync default
    p2 = json.loads(delegation.execute_delegation(
        "coder", "forced async", wait=False, _ctx=_ctx(cfg)))
    assert p2["status"] == "dispatched"


def test_sync_wait_never_writes_inbox(home, monkeypatch, _clean):
    cfg = _cfg()
    monkeypatch.setattr(delegation, "_run_child", _fake_child(
        {"completed": True, "result": "inline", "child_trace_id": None,
         "failure_reason": None}))
    payload = json.loads(delegation.execute_delegation(
        "coder", "sync with thread", wait=True, _ctx=_ctx(cfg,
                                                           thread="th-9")))
    assert payload["status"] == "completed"
    row = delegation_store.get_delegation(payload["delegation_id"])
    assert row["inbox_state"] == "none"  # result was delivered inline
    assert row["wait_mode"] == 1


def test_cancel_queued_delegation_settles_cancelled(home, monkeypatch, _clean):
    """State machine §3.2: queued -> cancelled before start."""
    cfg = _cfg()
    stop = threading.Event()
    # occupy the single worker with a blocking child so the next dispatch
    # stays 'queued'
    monkeypatch.setattr(delegation, "_run_child",
                        _fake_child({}, block_on=stop))
    p1 = json.loads(delegation.execute_delegation(
        "coder", "occupier", _ctx=_ctx(cfg)))
    assert _wait_until(lambda: delegation.get_delegation(
        p1["delegation_id"])["status"] == "running")
    p2 = json.loads(delegation.execute_delegation(
        "coder", "queued one", _ctx=_ctx(cfg)))
    assert p2["status"] == "dispatched"
    did2 = p2["delegation_id"]
    assert delegation.get_delegation(did2)["status"] == "queued"

    assert delegation.cancel_delegation(did2) is True
    poll = json.loads(delegation.check_delegation_status(did2))
    assert poll["status"] == "cancelled"
    assert delegation_store.get_delegation(did2)["status"] == "cancelled"
    stop.set()


# ---------------------------------------------------------------------------
# Durable store survives registry reset (audit §4 item 3)
# ---------------------------------------------------------------------------

def test_check_delegation_survives_registry_reset(home, monkeypatch, _clean):
    cfg = _cfg()
    monkeypatch.setattr(delegation, "_run_child", _fake_child(
        {"completed": True, "result": "persisted",
         "child_trace_id": "tr-p", "failure_reason": None}))
    payload = json.loads(delegation.execute_delegation(
        "coder", "outlive the cache", _ctx=_ctx(cfg)))
    did = payload["delegation_id"]
    assert _wait_until(lambda: delegation.get_delegation(did) and
                       delegation.get_delegation(did).get("_settled"))

    # wipe the in-process registries (simulated restart; keep the dispatcher)
    with delegation._registry_lock:
        delegation._delegations.clear()
        delegation._pruned_summaries.clear()
    assert delegation.get_delegation(did) is None

    poll = json.loads(delegation.check_delegation_status(did))
    assert poll["status"] == "completed", poll
    assert poll["result"] == "persisted"
    assert poll["child_trace_id"] == "tr-p"


def test_check_delegation_queued_reports_ahead(home, monkeypatch, _clean):
    cfg = _cfg()
    stop = threading.Event()
    monkeypatch.setattr(delegation, "_run_child",
                        _fake_child({}, block_on=stop))
    p1 = json.loads(delegation.execute_delegation(
        "coder", "first", _ctx=_ctx(cfg)))
    assert _wait_until(lambda: delegation.get_delegation(
        p1["delegation_id"])["status"] == "running")
    p2 = json.loads(delegation.execute_delegation(
        "coder", "second", _ctx=_ctx(cfg)))
    poll = json.loads(delegation.check_delegation_status(p2["delegation_id"]))
    assert poll["status"] == "queued"
    assert poll["queued_ahead"] == 0
    stop.set()


# ---------------------------------------------------------------------------
# Config: defaults, YAML section, env overrides
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = SableConfig()
    assert cfg.delegation_max_concurrent == 2
    assert cfg.delegation_queue_max == 16
    assert cfg.delegation_wake_retry_seconds == 2.0
    assert cfg.delegation_async_default is True
    # D-18: timeout/liveness knobs are gone — children have no deadlines.
    assert not hasattr(cfg, "delegation_timeout_seconds")
    assert not hasattr(cfg, "delegation_liveness_timeout_seconds")


def test_config_yaml_section(tmp_path, monkeypatch):
    monkeypatch.delenv("SABLE_DELEGATION_MAX", raising=False)
    f = tmp_path / "c.yaml"
    f.write_text(
        "delegation:\n"
        "  enabled: true\n"
        "  max_concurrent: 5\n"
        "  queue_max: 9\n"
        "  wake_retry_seconds: 3\n"
        "  async_default: false\n")
    cfg = load_config(str(f))
    assert cfg.delegation_max_concurrent == 5
    assert cfg.delegation_queue_max == 9
    assert cfg.delegation_wake_retry_seconds == 3.0
    assert cfg.delegation_async_default is False


def test_config_env_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("SABLE_CONFIG", str(tmp_path / "none.yaml"))
    monkeypatch.setenv("SABLE_DELEGATION_MAX", "7")
    cfg = load_config()
    assert cfg.delegation_max_concurrent == 7


# ===========================================================================
# Phase 2 — inbox wake (spec §4, §9): hooks, watcher pass, message format,
# single-flight deferral, ping-pong guard, exactly-once delivery. The cadence
# is faked by calling run_wake_pass/try_deliver directly (no sleeping on
# wake_retry_seconds, per spec risk #5).
# ===========================================================================

def _pending_row(did: str, thread: str = "th-w", **over) -> Dict[str, Any]:
    row = _rec(did, status="completed", thread=thread, inbox_state="pending",
               settled_ts=_iso(-1), result=f"result of {did}",
               child_trace_id=f"tr-{did}")
    row.update(over)
    return row


class _FakeWakeHook:
    """Records delivery attempts; `busy` simulates the single-flight check
    refusing (thread has an in-flight run -> entry stays pending)."""

    def __init__(self, busy: bool = False):
        self.calls: list = []
        self.busy = busy

    def __call__(self, thread_id: str) -> Dict[str, Any]:
        self.calls.append(thread_id)
        if self.busy:
            return {"started": False, "busy": True}
        rows = delegation_store.take_pending_inbox(thread_id)
        if not rows:
            return {"started": False, "empty": True}
        delegation_store.mark_delivered_run(
            [r["delegation_id"] for r in rows], f"wake-{len(self.calls)}")
        return {"started": True, "run_id": f"wake-{len(self.calls)}",
                "delegation_ids": [r["delegation_id"] for r in rows]}


def test_wake_hook_registry_roundtrip(home, _clean):
    hook = _FakeWakeHook()
    delegation.register_wake_hook("th-x", hook)
    assert delegation.get_wake_hook("th-x") is hook
    delegation.unregister_wake_hook("th-x")
    assert delegation.get_wake_hook("th-x") is None


def test_wake_pass_delivers_when_thread_idle(home, _clean):
    delegation_store.insert_delegation(_pending_row("w1"))
    delegation_store.insert_delegation(_pending_row("w2", settled_ts=_iso(-2)))
    hook = _FakeWakeHook()
    delegation.register_wake_hook("th-w", hook)
    assert delegation.run_wake_pass() == 1
    # ONE batched wake per thread per pass (both entries in one delivery)
    assert hook.calls == ["th-w"]
    row1 = delegation_store.get_delegation("w1")
    assert row1["inbox_state"] == "delivered"
    assert row1["delivered_run_id"]
    # nothing left pending -> next pass starts no wake
    assert delegation.run_wake_pass() == 0


def test_wake_defers_while_thread_busy_then_fires(home, _clean):
    delegation_store.insert_delegation(_pending_row("w3"))
    hook = _FakeWakeHook(busy=True)
    delegation.register_wake_hook("th-w", hook)
    assert delegation.run_wake_pass() == 0
    row = delegation_store.get_delegation("w3")
    assert row["inbox_state"] == "pending", "busy thread must not consume"
    # the run ends -> watcher retries and delivers promptly
    hook.busy = False
    assert delegation.run_wake_pass() == 1
    assert delegation_store.get_delegation("w3")["inbox_state"] == "delivered"


def test_wake_without_hook_stays_pending(home, _clean):
    delegation_store.insert_delegation(_pending_row("w4", thread="th-nohook"))
    assert delegation.run_wake_pass() == 0
    assert delegation_store.get_delegation("w4")["inbox_state"] == "pending"


def test_format_inbox_message_shape(home):
    rows = [
        _rec("a1b2c3", status="completed", thread="th-w", agent="reviewer",
             result="all good", child_trace_id="trk9",
             started_ts=_iso(-85), settled_ts=_iso(-1)),
        _rec("d4e5f6", status="wedged", thread="th-w", agent="tester",
             error="liveness timeout after 600s of silence"),
    ]
    msg = delegation.format_inbox_message(rows)
    assert msg.startswith("[Delegation results")
    assert "delegation_id=a1b2c3" in msg and "agent=reviewer" in msg
    assert "all good" in msg and "trk9" in msg and "took 84s" in msg
    assert "error: liveness timeout after 600s of silence" in msg
    assert "Continue the user's task" in msg


def test_take_pending_exactly_once_across_concurrent_wakes(home):
    for i in range(10):
        delegation_store.insert_delegation(
            _pending_row(f"x{i}", settled_ts=_iso(-20 + i)))
    delivered: list = []
    dlock = threading.Lock()

    def racer():
        rows = delegation_store.take_pending_inbox("th-w")
        if rows:
            with dlock:
                delivered.extend(r["delegation_id"] for r in rows)

    ts = [threading.Thread(target=racer, daemon=True) for _ in range(6)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=15)
    assert sorted(delivered) == sorted(f"x{i}" for i in range(10)), \
        "each entry delivered by exactly one racing wake consumer"


def test_wake_streak_guard_bounds_ping_pong(home, _clean, monkeypatch):
    """§10 risk #2: consecutive watcher wakes without a user run stop at
    WAKE_STREAK_MAX; a real user run (reset_wake_streak) re-arms delivery."""
    monkeypatch.setattr(delegation, "WAKE_STREAK_MAX", 3)
    hook = _FakeWakeHook()
    delegation.register_wake_hook("th-w", hook)
    for i in range(6):
        delegation_store.insert_delegation(_pending_row(f"s{i}"))
        delegation.try_deliver("th-w")
    assert len(hook.calls) == 3, "wake chain must stop at the streak cap"
    res = delegation.try_deliver("th-w")
    assert res.get("started") is False and res.get("suppressed") == "wake_streak"
    # user message on the thread clears the streak -> delivery resumes
    delegation.reset_wake_streak("th-w")
    delegation_store.insert_delegation(_pending_row("s9"))
    assert delegation.try_deliver("th-w").get("started") is True


def test_reconcile_entries_delivered_on_next_wake(home, _clean):
    """A crash-interrupted child leaves an inbox entry; the next wake (or user
    message) delivers it — reconcile never drops undelivered entries."""
    delegation_store.insert_delegation(
        _rec("crashed", status="running", thread="th-w"))
    assert delegation_store.reconcile_on_startup() == 1
    assert delegation_store.get_delegation("crashed")["inbox_state"] == "pending"
    hook = _FakeWakeHook()
    delegation.register_wake_hook("th-w", hook)
    assert delegation.run_wake_pass() == 1
    row = delegation_store.get_delegation("crashed")
    assert row["inbox_state"] == "delivered"
    msg = delegation.format_inbox_message([row])
    assert "interrupted" in msg and "restarted while the child was" in msg


# ---------------------------------------------------------------------------
# Phase 2 dashboard: start_inbox_wake_run / drain-on-user-message / wake event
# ---------------------------------------------------------------------------

class _StubRunner:
    """create_runner stub for chat-run tests: records the prompt + delegation
    context it was built with and answers immediately."""

    made: list = []

    def __init__(self, cfg, base_url, model, **kw):
        self.on_event = None
        self.kw = kw

    def run(self, prompt, history=None, previous_summary=None):
        from sable.core.agent_runner import AgentResult

        _StubRunner.made.append({"prompt": prompt,
                                 "ctx": self.kw.get("delegation_context")})
        return AgentResult(final_response="ack",
                           messages=[{"role": "assistant", "content": "ack"}],
                           trace_id="tr-stub")


@pytest.fixture()
def dash(home, monkeypatch):
    """DashboardApp wired to a fake serving endpoint + stub runner."""
    from sable.dashboard.app import DashboardApp

    _StubRunner.made = []

    def _create(cfg, base_url, model, **kw):
        return _StubRunner(cfg, base_url, model, **kw)

    monkeypatch.setattr("sable.core.agent_runner.create_runner", _create)
    with FakeOpenAIServer() as fs:
        (home / "serve_state.json").write_text(json.dumps({
            "mode": "attach", "base_url": fs.base_url,
            "model_name": "fake-model"}))
        yield DashboardApp(SableConfig(), home=home)


def _wait_run_done(app, run_id, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if app._chat_runs.get(run_id, {}).get("done"):
            return True
        time.sleep(0.02)
    return False


def _drain_stream(resp, max_frames=200):
    frames = []
    for i, f in enumerate(resp.stream or []):
        frames.append(f)
        if "event: run.done" in f or "event: run.error" in f \
                or i >= max_frames:
            break
    return "".join(frames)


def test_wake_run_starts_with_synthetic_turn_and_persists(home, _clean, dash):
    """Settle an async child -> wake run starts whose user message contains
    the result; transcript persisted; delegation.wake bus event emitted."""
    delegation_store.insert_delegation(
        _pending_row("wake1", thread="t-wake", result="child says hi"))
    # register the dashboard hook lazily, like a first chat run would
    delegation.register_wake_hook(
        "t-wake", lambda t: dash.start_inbox_wake_run(t))
    sub = dash.bus.subscribe()

    assert delegation.run_wake_pass(dash.cfg) == 1
    wake = _StubRunner.made[-1]
    assert "[Delegation results" in wake["prompt"]
    assert "child says hi" in wake["prompt"]
    assert wake["ctx"]["thread"] == "t-wake"
    assert wake["ctx"]["inbox_wake"] is True

    run_id = next(iter(dash._chat_runs))
    assert _wait_run_done(dash, run_id)
    t = json.loads((home / "chat" / "t-wake.json").read_text())
    roles = [m["role"] for m in t["messages"]]
    assert roles == ["user", "assistant"]
    assert "child says hi" in t["messages"][0]["content"]

    ev = None
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and ev is None:
        e = sub.poll(timeout=0.2)
        if e and e["kind"] == "delegation.wake":
            ev = e
    assert ev is not None, "delegation.wake bus event must be published"
    assert ev["payload"]["thread"] == "t-wake"
    assert ev["payload"]["run_id"] == run_id
    assert ev["payload"]["delegation_ids"] == ["wake1"]
    assert delegation_store.get_delegation("wake1")["delivered_run_id"] \
        == run_id


def test_single_flight_suppresses_wake_then_fires(home, _clean, dash):
    """Wake obeys the same single-flight rule: while a run is in flight the
    entries stay pending; after it ends the wake fires."""
    delegation_store.insert_delegation(
        _pending_row("wake2", thread="t-busy"))
    # seed an in-flight run for the thread (exactly the registry shape)
    dash._chat_runs["inflight-1"] = {"cancel": threading.Event(),
                                     "done": False, "thread": "t-busy"}
    res = dash.start_inbox_wake_run("t-busy")
    assert res["started"] is False and res.get("busy")
    assert delegation_store.get_delegation("wake2")["inbox_state"] == "pending"
    assert not _StubRunner.made, "no run may start while busy"
    # the in-flight run finishes -> wake now starts
    dash._chat_runs["inflight-1"]["done"] = True
    res2 = dash.start_inbox_wake_run("t-busy")
    assert res2["started"] is True
    assert delegation_store.get_delegation("wake2")["inbox_state"] \
        == "delivered"


def test_user_message_drains_pending_inbox_no_wake(home, _clean, dash):
    """D-17 "user wins": a real message on a thread with pending entries
    prepends the block to its own prompt and marks them delivered — no wake."""
    delegation_store.insert_delegation(
        _pending_row("wake3", thread="t-drain", result="queued answer"))
    resp = dash.handle("POST", "/api/chat/stream",
                       body={"prompt": "hello there", "thread": "t-drain"})
    assert resp.status == 200
    _drain_stream(resp)
    run = _StubRunner.made[-1]
    assert run["prompt"].startswith("[Delegation results")
    assert "queued answer" in run["prompt"]
    assert run["prompt"].endswith("hello there")
    assert run["ctx"]["inbox_wake"] is False
    row = delegation_store.get_delegation("wake3")
    assert row["inbox_state"] == "delivered"
    assert row["delivered_run_id"]  # stamped against the user's run id


def test_chat_wake_endpoint_forces_delivery(home, _clean, dash):
    """POST /api/chat/wake is the deterministic trigger (spec §4.5)."""
    delegation_store.insert_delegation(
        _pending_row("wake4", thread="t-ep"))
    r = dash.handle("POST", "/api/chat/wake", body={"thread": "t-ep"})
    assert r.status == 200 and r.body["started"] is True
    assert _StubRunner.made[-1]["prompt"].count("delegation_id=wake4") == 1
    assert delegation_store.get_delegation("wake4")["inbox_state"] \
        == "delivered"
    bad = dash.handle("POST", "/api/chat/wake", body={})
    assert bad.status == 400


def test_run_end_nudges_watcher_delivery(home, _clean, monkeypatch):
    """§4.3: a chat worker's finally calls nudge_delivery so the shared
    watcher loop delivers queued entries promptly instead of at the next tick."""
    nudged = []
    monkeypatch.setattr(delegation, "nudge_delivery",
                        lambda: nudged.append(1))
    delegation.register_wake_hook("t-n", lambda t: {"started": False})
    # exercise the real code path through a user run on a thread
    from sable.dashboard.app import DashboardApp

    def _create(cfg, base_url, model, **kw):
        return _StubRunner(cfg, base_url, model, **kw)

    monkeypatch.setattr("sable.core.agent_runner.create_runner", _create)
    with FakeOpenAIServer() as fs:
        (home / "serve_state.json").write_text(json.dumps({
            "mode": "attach", "base_url": fs.base_url,
            "model_name": "fake-model"}))
        app = DashboardApp(SableConfig(), home=home)
        resp = app.handle("POST", "/api/chat/stream",
                          body={"prompt": "go", "thread": "t-n"})
        _drain_stream(resp)
    assert nudged, "run completion must nudge the wake watcher"


# ===========================================================================
# Phase 3 — polish (spec §8): store janitor + busy-poll guardrail +
# pending-inbox surfacing + wake-run trace badge.
# ===========================================================================

_AGE_S = 40 * 86400  # "aged" reference: 40 days ago (> default 30d window)


def _terminal_row(did: str, status: str = "completed", **over) -> Dict[str, Any]:
    row = _rec(did, status=status, settled_ts=_iso(-_AGE_S),
               created_ts=_iso(-_AGE_S))
    row.update(over)
    return row


# ------------------------------------------------------------- janitor rules

def test_prune_age_window_terminal_only(home):
    """Age rule: terminal + not-pending rows settled before the cutoff go;
    fresh terminals, non-terminal rows, timed_out (still reconcilable) and
    undelivered inbox entries are all preserved."""
    delegation_store.insert_delegation(_terminal_row("old-done"))
    delegation_store.insert_delegation(_terminal_row("old-fail",
                                                     status="failed"))
    delegation_store.insert_delegation(_terminal_row(
        "old-pending", thread="th-1", inbox_state="pending"))
    delegation_store.insert_delegation(_terminal_row("fresh-done",
                                                     settled_ts=_iso(),
                                                     created_ts=_iso()))
    delegation_store.insert_delegation(_rec("still-running",
                                            status="running"))
    delegation_store.insert_delegation(_terminal_row("old-timeout",
                                                     status="timed_out"))
    out = delegation_store.prune_terminal_rows(days=30)
    assert out["deleted_aged"] == 2 and out["deleted"] == 2
    assert delegation_store.get_delegation("old-done") is None
    assert delegation_store.get_delegation("old-fail") is None
    # protected: undelivered inbox, fresh row, non-terminal, timed_out
    assert delegation_store.get_delegation("old-pending")["status"] \
        == "completed"
    assert delegation_store.get_delegation("fresh-done") is not None
    assert delegation_store.get_delegation("still-running") is not None
    assert delegation_store.get_delegation("old-timeout") is not None


def test_prune_row_cap_evicts_oldest_prunable(home):
    """Cap rule: only the newest `max_rows` prunable terminal rows survive;
    undelivered inbox rows neither count against nor fall to the cap."""
    for i in range(5):
        delegation_store.insert_delegation(_terminal_row(
            f"c{i}", settled_ts=_iso(-_AGE_S - 100 + i * 10),
            created_ts=_iso(-_AGE_S - 100 + i * 10),
            inbox_state="delivered"))
    delegation_store.insert_delegation(_terminal_row(
        "c-pending", thread="th-9", inbox_state="pending",
        settled_ts=_iso(-_AGE_S - 200), created_ts=_iso(-_AGE_S - 200)))
    out = delegation_store.prune_terminal_rows(days=365, max_rows=2)
    assert out["deleted_capped"] == 3, out
    kept = {r["delegation_id"] for r in
            delegation_store.list_delegations(limit=100)}
    # newest two delivered rows survive; the pending row is never touched
    assert kept == {"c3", "c4", "c-pending"}, kept


def test_prune_keeps_rows_without_settled_ts_from_age(home):
    """Unparseable/missing settled_ts is conservatively NOT old (age rule);
    the cap can still evict it, oldest-first by created_ts."""
    delegation_store.insert_delegation(_rec("no-settle", status="completed"))
    out = delegation_store.prune_terminal_rows(days=0)  # cutoff = now
    assert out["deleted_aged"] == 0
    assert delegation_store.get_delegation("no-settle") is not None


def test_janitor_pass_wired_and_best_effort(home, monkeypatch):
    """delegation.janitor_pass delegates to the store and never raises."""
    delegation_store.insert_delegation(_terminal_row("jp"))  # aged by helper
    out = delegation.janitor_pass(_cfg())
    assert out["deleted"] == 1
    assert delegation_store.get_delegation("jp") is None

    def boom(*a, **k):
        raise RuntimeError("store down")
    monkeypatch.setattr(delegation.delegation_store, "prune_terminal_rows",
                        boom)
    assert delegation.janitor_pass(_cfg()) == {"deleted_aged": 0,
                                               "deleted_capped": 0,
                                               "deleted": 0}


# ----------------------------------------------------- busy-poll guardrail

def test_busy_poll_guard_appends_hint_after_rapid_checks(home, monkeypatch,
                                                         _clean):
    """D-17 §10 risk #1: after POLL_GUARD_MAX_CHECKS rapid checks of the same
    delegation by one parent run (with an inbox), the payload carries a gentle
    'results arrive automatically' hint — earlier checks stay clean."""
    monkeypatch.setattr(delegation, "POLL_GUARD_MAX_CHECKS", 2)
    cfg = _cfg()
    ctx = _ctx(cfg, thread="th-poll")
    monkeypatch.setattr(delegation, "_run_child", _fake_child(
        {"completed": True, "result": "r", "child_trace_id": None,
         "failure_reason": None}))
    payload = json.loads(delegation.execute_delegation(
        "coder", "poll me", _ctx=ctx))
    did = payload["delegation_id"]
    # wait on the cache directly — polling through check_delegation here would
    # consume the guard budget being measured.
    assert _wait_until(lambda: bool((delegation.get_delegation(did) or {})
                                    .get("_settled")))

    notes = [json.loads(delegation.check_delegation_status(did, _ctx=ctx))
             .get("poll_guard") for _ in range(3)]
    assert notes[0] is None and notes[1] is None
    assert notes[2] and "busy-poll" in notes[2]
    assert "arrive automatically" in notes[2]

    # A different parent run has its own budget (no inherited hint).
    other = dict(_ctx(cfg, thread="th-other"), run_id="run-other")
    assert json.loads(delegation.check_delegation_status(
        did, _ctx=other)).get("poll_guard") is None


def test_busy_poll_guard_silent_without_thread(home, monkeypatch, _clean):
    """Threadless (CLI) polling has no inbox — it stays honest and unhinted."""
    monkeypatch.setattr(delegation, "POLL_GUARD_MAX_CHECKS", 1)
    cfg = _cfg()
    ctx = _ctx(cfg)  # thread=None
    monkeypatch.setattr(delegation, "_run_child", _fake_child(
        {"completed": True, "result": "r", "child_trace_id": None,
         "failure_reason": None}))
    did = json.loads(delegation.execute_delegation(
        "coder", "cli poll", _ctx=ctx))["delegation_id"]
    assert _wait_until(lambda: delegation.get_delegation(did) and
                       delegation.get_delegation(did).get("_settled"))
    for _ in range(4):
        out = json.loads(delegation.check_delegation_status(did, _ctx=ctx))
        assert out.get("poll_guard") is None


def test_check_delegation_tool_threads_ctx(home, monkeypatch, _clean):
    """core/tools.py check_delegation passes the runner ctx through so the
    guardrail actually engages on model calls."""
    from sable.core import tools

    monkeypatch.setattr(delegation, "POLL_GUARD_MAX_CHECKS", 1)
    seen = {}
    real = delegation.check_delegation_status

    def spy(did, _ctx=None):
        seen["ctx"] = _ctx
        return real(did, _ctx=_ctx)
    monkeypatch.setattr(delegation, "check_delegation_status", spy)
    ctx = {"run_id": "r-7"}
    out = json.loads(tools.check_delegation(id="d-x", _ctx=ctx))
    assert seen["ctx"] == ctx
    assert out["status"] == "failed"  # unknown id -> honest error payload


# ------------------------------------- pending-inbox surfacing + wake badge

def test_threads_list_surfaces_pending_inbox_counts(home, _clean, dash):
    """GET /api/chat/threads carries per-thread undelivered inbox counts for
    the UI badge (0 when nothing waits; store errors degrade to no badges)."""
    dash.handle("PUT", "/api/chat/threads/t-badge", body={"messages": []})
    dash.handle("PUT", "/api/chat/threads/t-quiet", body={"messages": []})
    delegation_store.insert_delegation(
        _pending_row("b1", thread="t-badge"))
    delegation_store.insert_delegation(
        _pending_row("b2", thread="t-badge", settled_ts=_iso(-2)))
    rows = dash.handle("GET", "/api/chat/threads").body["threads"]
    by_id = {t["id"]: t for t in rows}
    assert by_id["t-badge"]["pending_inbox"] == 2
    assert by_id["t-quiet"]["pending_inbox"] == 0


def test_wake_run_trace_carries_inbox_wake_metadata(home, _clean, dash,
                                                    monkeypatch):
    """Phase 3: the wake run's recorder gets metadata {inbox_wake: True} so
    the traces panel can badge it; user runs get none."""
    from sable.store import recorder as recorder_mod

    made = []
    real = recorder_mod.make_recorder

    def spy(cfg, *a, metadata=None, **kw):
        made.append(metadata)
        return real(cfg, *a, metadata=metadata, **kw)
    monkeypatch.setattr(recorder_mod, "make_recorder", spy)

    delegation_store.insert_delegation(_pending_row("meta1", thread="t-meta"))
    delegation.register_wake_hook("t-meta",
                                  lambda t: dash.start_inbox_wake_run(t))
    assert delegation.run_wake_pass(dash.cfg) == 1
    assert made and made[-1] == {"inbox_wake": True}

    resp = dash.handle("POST", "/api/chat/stream",
                       body={"prompt": "plain", "thread": "t-plain"})
    _drain_stream(resp)
    assert made[-1] is None


# ------------------------------------------------- audit fixes (2026-11 pass)

def test_wake_run_registers_eagerly_under_single_flight(home, _clean, dash,
                                                         monkeypatch):
    """Audit finding #2: the wake path registers its run inside the
    single-flight lock (register_now), so a concurrent user POST for the same
    thread cannot slip between spawn and first-read and double-start.

    Observed INSIDE the old gap: _spawn_chat_run is wrapped and the registry
    inspected the instant it returns — before the caller admits the stream.
    Before the fix the entry only appeared after that admission."""
    delegation_store.insert_delegation(
        _pending_row("race1", thread="t-race"))

    real = dash._spawn_chat_run
    seen = {}

    def spy(**kw):
        resp = real(**kw)
        seen["registered"] = kw["run_id"] in dash._chat_runs
        return resp
    monkeypatch.setattr(dash, "_spawn_chat_run", spy)

    res = dash.start_inbox_wake_run("t-race")
    assert res["started"] is True
    assert seen.get("registered"), \
        "wake run must be in the registry before its stream is admitted"


def test_wake_admission_failure_requeues_entries(home, _clean, dash,
                                                 monkeypatch):
    """Audit finding #3: if admitting the wake stream blows up before the
    worker ever ran, the taken inbox entries go back to pending instead of
    dying with a phantom delivered_run_id."""
    delegation_store.insert_delegation(
        _pending_row("requeue1", thread="t-re"))

    real = dash._spawn_chat_run

    def boom(**kw):
        resp = real(**kw)
        # Sabotage the stream so first-read raises like an admission failure.
        class _Bad:
            def __iter__(self):
                raise RuntimeError("admission exploded")
        resp.stream = _Bad()
        return resp
    monkeypatch.setattr(dash, "_spawn_chat_run", boom)

    res = dash.start_inbox_wake_run("t-re")
    assert res["started"] is False
    assert "admission failed" in (res.get("error") or "")
    assert delegation_store.get_delegation("requeue1")["inbox_state"] \
        == "pending", "entries must return to pending on failed admission"
    # The phantom registry entry was evicted, so the thread is usable again.
    assert "run_id" not in res or res["run_id"] not in dash._chat_runs


def test_user_drain_marks_persisted_turn_inbox(home, _clean, dash):
    """Audit finding #6/#7: a drained "user wins" turn persists with an
    explicit inbox flag (server-stamped, no content sniffing), and the
    persisted prompt carries the drained block."""
    delegation_store.insert_delegation(
        _pending_row("drain1", thread="t-dmark", result="queued answer"))
    resp = dash.handle("POST", "/api/chat/stream",
                       body={"prompt": "hello there", "thread": "t-dmark"})
    text = _drain_stream(resp)
    assert "event: run.started" in text
    # run.started echoes the authoritative (drained) prompt + inbox flag.
    started = next(d for k, d in _sse(text) if k == "run.started")
    assert started["inbox"] is True
    assert "[Delegation results" in started["prompt"]

    t = json.loads((home / "chat" / "t-dmark.json").read_text())
    user = t["messages"][0]
    assert user["role"] == "user" and user.get("inbox") is True
    assert "queued answer" in user["content"]


def test_plain_run_not_marked_inbox(home, _clean, dash):
    resp = dash.handle("POST", "/api/chat/stream",
                       body={"prompt": "plain turn", "thread": "t-plain2"})
    text = _drain_stream(resp)
    started = next(d for k, d in _sse(text) if k == "run.started")
    assert started["inbox"] is False
    t = json.loads((home / "chat" / "t-plain2.json").read_text())
    assert not t["messages"][0].get("inbox")


def test_thread_delete_cancels_run_and_blocks_resurrection(home, _clean,
                                                           monkeypatch):
    """Audit finding: deleting a thread mid-run must cancel the run AND stop
    its worker from re-creating the deleted file when it finishes."""
    import threading as _th

    from sable.core.agent_runner import AgentResult

    class _SlowRunner:
        def __init__(self, cfg, base_url, model, **kw):
            self.on_event = None
            self.kw = kw
            self.started = _th.Event()

        def run(self, prompt, history=None, previous_summary=None):
            self.started.set()
            for _ in range(200):     # up to 10s, polls the cancel event
                ev = self.kw.get("cancel_event")
                if ev is not None and ev.is_set():
                    break
                time.sleep(0.05)
            return AgentResult(final_response="late answer",
                               messages=[{"role": "assistant",
                                          "content": "late answer"}],
                               trace_id="tr-late")

    from sable.dashboard.app import DashboardApp

    slow = {}

    def _create(cfg, base_url, model, **kw):
        r = _SlowRunner(cfg, base_url, model, **kw)
        slow["r"] = r
        return r

    with FakeOpenAIServer() as fs:
        (home / "serve_state.json").write_text(json.dumps({
            "mode": "attach", "base_url": fs.base_url,
            "model_name": "fake-model"}))
        import sable.core.agent_runner as ar
        old = ar.create_runner
        ar.create_runner = _create
        try:
            app = DashboardApp(SableConfig(), home=home)
            # The real client creates the thread file before starting a run.
            app.handle("PUT", "/api/chat/threads/t-del",
                       body={"messages": [{"role": "user",
                                           "content": "long job"}]})
            resp = app.handle("POST", "/api/chat/stream",
                              body={"prompt": "long job", "thread": "t-del"})
            it = iter(resp.stream)
            next(it)                       # admitted; worker starts
            assert slow["r"].started.wait(5)
            r = app.handle("DELETE", "/api/chat/threads/t-del")
            assert r.status == 200
            assert not (home / "chat" / "t-del.json").exists()
            # The cancel event reached the runner...
            run_id = next(iter(app._chat_runs))
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if app._chat_runs[run_id].get("done"):
                    break
                time.sleep(0.02)
            assert app._chat_runs[run_id].get("done"), "run must finish fast"
            # ...and its persistence step refused to recreate the file.
            time.sleep(0.2)
            assert not (home / "chat" / "t-del.json").exists(), \
                "deleted thread must not resurrect from a finishing run"
        finally:
            ar.create_runner = old


def _sse(text):
    """Parse an SSE transcript into (kind, data) pairs."""
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


def test_follow_chat_run_announces_dropped_frames():
    """Audit finding #11: a reader attaching after ring-buffer rollover gets
    exactly one stream.note frame naming the dropped count."""
    from sable.dashboard.api.chat import _follow_chat_run

    run = {"frames": ["event: message.delta\ndata: {}\n\n",
                      "event: run.done\ndata: {}\n\n"],
           "dropped": 7, "done": True, "cond": threading.Condition()}
    text = "".join(_follow_chat_run(run))
    notes = [d for k, d in _sse(text) if k == "stream.note"]
    assert len(notes) == 1 and notes[0]["dropped"] == 7
    # Frames that survived the rollover still replay.
    kinds = [k for k, _ in _sse(text)]
    assert "run.done" in kinds


def test_threads_list_pagination(home, _clean, dash):
    """Audit gap: h_threads_list supports ?limit/&offset without changing the
    default return-everything behavior."""
    for i in range(5):
        dash.handle("PUT", f"/api/chat/threads/t-pg{i}",
                    body={"messages": []})
    full = dash.handle("GET", "/api/chat/threads").body["threads"]
    assert len(full) == 5
    page = dash.handle("GET", "/api/chat/threads",
                       query={"limit": "2", "offset": "1"}).body
    assert [t["id"] for t in page["threads"]] == ["t-pg1", "t-pg2"]


def test_thread_put_filters_non_dict_delegations(home, _clean, dash):
    """Audit gap: a client PUT with junk delegation entries keeps only dicts,
    so owner chips can never render '?' forever."""
    r = dash.handle("PUT", "/api/chat/threads/t-dels", body={
        "messages": [], "delegations": [
            {"agent": "coder", "status": "done"}, "junk", 42]})
    assert r.status == 200
    dels = r.body["thread"]["delegations"]
    assert dels == [{"agent": "coder", "status": "done"}]

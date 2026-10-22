"""Inspiration pass 4 #28 — circuit breaker + serve crash-loop budget.

Covers: the CircuitBreaker state machine (threshold, open blocks, half-open
probe exactly-once, success closes, failure re-opens) against an injectable
clock; runner integration (a backend failing 5 consecutive attempts opens the
breaker and the next run fails fast with failure_reason=backend_circuit_open
without touching the network); judge transport shares the breaker; and the
supervisor crash-loop budget (3 failed starts in-window -> refused; explicit
user start clears).
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import cast

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fake_server import FakeOpenAIServer  # noqa: E402
from sable.core import circuit  # noqa: E402
from sable.core.agent_runner import SableAgentClient  # noqa: E402
from sable.core.circuit import CircuitBreaker, get_breaker  # noqa: E402
from sable.core.failure_types import (  # noqa: E402
    FAILURE_BACKEND_CIRCUIT_OPEN,
    is_infra_reason,
)


class FakeClock:
    def __init__(self, t: float = 100.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


# ------------------------------------------------------------- state machine

def test_opens_after_threshold_consecutive_failures():
    b = CircuitBreaker(failure_threshold=5, cooldown_seconds=60.0,
                       clock=FakeClock())
    for i in range(4):
        b.record_failure()
        assert b.state == "closed", i
    assert b.is_allowed() is True
    b.record_failure()  # 5th consecutive
    assert b.state == "open"
    assert b.is_allowed() is False
    assert b.failures == 5


def test_success_resets_streak():
    b = CircuitBreaker(failure_threshold=3, cooldown_seconds=60.0,
                       clock=FakeClock())
    b.record_failure()
    b.record_failure()
    b.record_success()
    assert b.failures == 0 and b.state == "closed"
    b.record_failure()
    b.record_failure()
    assert b.state == "closed"  # streak restarted, not cumulative


def test_open_blocks_until_cooldown_then_half_open_probe_once():
    clock = FakeClock()
    b = CircuitBreaker(failure_threshold=2, cooldown_seconds=10.0, clock=clock)
    b.record_failure()
    b.record_failure()
    assert b.state == "open"
    assert b.is_allowed() is False
    assert b.seconds_until_probe() > 0

    clock.advance(9.0)
    assert b.is_allowed() is False  # still cooling down
    clock.advance(2.0)
    assert b.state == "half_open"
    assert b.is_allowed() is True   # probe granted
    assert b.is_allowed() is False  # exactly one probe in flight


def test_probe_success_closes_probe_failure_reopens():
    clock = FakeClock()
    b = CircuitBreaker(failure_threshold=2, cooldown_seconds=10.0, clock=clock)
    b.record_failure()
    b.record_failure()
    clock.advance(11.0)
    assert b.is_allowed() is True   # probe slot
    b.record_success()
    assert b.state == "closed"
    assert b.is_allowed() is True

    # now the failure path: probe -> fail -> OPEN with fresh cooldown
    b.record_failure()
    b.record_failure()
    clock.advance(11.0)
    assert b.is_allowed() is True   # half-open probe
    b.record_failure()              # probe failed
    assert b.state == "open"
    # Fresh cooldown started at the probe failure: full window remains.
    assert 9.9 < b.seconds_until_probe() <= 10.0


def test_stuck_half_open_probe_is_released():
    clock = FakeClock()
    b = CircuitBreaker(failure_threshold=1, cooldown_seconds=5.0, clock=clock)
    b.record_failure()
    assert b.state == "open"
    clock.advance(6.0)
    assert b.is_allowed() is True   # probe taken, caller "dies"
    assert b.is_allowed() is False  # in flight within cooldown window
    clock.advance(6.0)              # probe considered stale
    assert b.is_allowed() is True


def test_snapshot_fields():
    clock = FakeClock()
    b = CircuitBreaker(failure_threshold=1, cooldown_seconds=5.0, clock=clock)
    snap = b.snapshot()
    assert snap["state"] == "closed" and snap["opened_at"] is None
    b.record_failure()
    snap = b.snapshot()
    assert snap["state"] == "open"
    assert snap["failures"] == 1
    assert snap["opened_at"] is not None  # wall-clock epoch for display
    assert cast(float, snap["seconds_until_probe"]) > 0


def test_registry_shared_and_thread_safe():
    b1 = get_breaker("http://shared/v1")
    b2 = get_breaker("http://shared/v1")
    assert b1 is b2
    assert "http://shared/v1" in circuit.all_breakers()

    errors = []

    def hammer():
        try:
            for _ in range(200):
                b1.is_allowed()
                b1.record_failure()
                b1.record_success()
        except Exception as e:  # pragma: no cover
            errors.append(e)

    ts = [threading.Thread(target=hammer) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=10)
    assert not errors


def test_circuit_open_reason_is_infra_classified():
    assert is_infra_reason(FAILURE_BACKEND_CIRCUIT_OPEN)
    assert is_infra_reason(f"backend_error:{FAILURE_BACKEND_CIRCUIT_OPEN}")


# ------------------------------------------------------------ runner wiring

class DeadHandler(BaseHTTPRequestHandler):
    """Every chat POST answers 500 — a wedged backend."""

    hits = 0

    def log_message(self, format, *args):  # noqa: A002 - silence access log
        pass

    def do_POST(self):
        type(self).hits += 1
        body = json.dumps({"error": "wedged"}).encode()
        self.send_response(500)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture(autouse=True)
def _no_backoff_sleeps(monkeypatch):
    from sable.core import agent_runner as ar
    monkeypatch.setattr(ar.SableAgentClient, "_sleep",
                        staticmethod(lambda s: None))


def test_wedged_backend_opens_breaker_and_next_run_fails_fast(tmp_path):
    DeadHandler.hits = 0
    with FakeOpenAIServer(handler_cls=DeadHandler) as srv:
        client = SableAgentClient(base_url=srv.base_url, model="m",
                                    timeout=2.0)
        # Run 1: 3 attempts, all fail (streak=3 < threshold 5).
        r1 = client.run("task one")
        assert r1.failure_reason == "backend_error:connection"
        assert DeadHandler.hits == 3

        # Run 2: attempts 4+5 push the streak over the threshold and OPEN the
        # circuit mid-run; the next attempt is refused without the network.
        r2 = client.run("task two")
        assert r2.failure_reason == FAILURE_BACKEND_CIRCUIT_OPEN
        assert DeadHandler.hits == 5

        # Run 3: fully fail-fast — zero additional network hits.
        r3 = client.run("task three")
        assert r3.failure_reason == FAILURE_BACKEND_CIRCUIT_OPEN
        assert DeadHandler.hits == 5

        snap = circuit.all_breakers()[srv.base_url]
        assert snap["state"] == "open"


def test_runner_success_closes_streak(tmp_path):
    """One blip + success must not march toward OPEN (consecutive semantics)."""
    with FakeOpenAIServer() as srv:  # healthy fake
        client = SableAgentClient(base_url=srv.base_url, model="fake-model")
        result = client.run("read the fake server source")
    assert result.completed is True
    assert circuit.all_breakers()[srv.base_url]["state"] == "closed"


def test_judge_shares_the_runner_breaker():
    from sable.curate.judge import make_chat_client

    url = "http://127.0.0.1:9/v1"  # nothing listens here
    breaker = get_breaker(url)
    for _ in range(5):
        breaker.record_failure()
    assert breaker.state == "open"

    client_fn = make_chat_client(url, timeout=1.0, transport_attempts=3)
    with pytest.raises(RuntimeError, match="circuit open"):
        client_fn([{"role": "user", "content": "grade me"}], "m")


# --------------------------------------------------- supervisor crash budget

def test_crash_loop_budget_refuses_after_three_in_window():
    from sable.serving import supervisor as sup

    key = "inst-a"
    assert sup.crash_loop_status(key) is None
    sup.record_start_failure(key)
    sup.record_start_failure(key)
    assert not sup.crash_loop_refused(key)
    sup.record_start_failure(key)
    assert sup.crash_loop_refused(key)
    st = sup.crash_loop_status(key)
    assert st is not None
    assert st["failures"] == 3 and st["refused"] is True


def test_crash_loop_window_expiry(monkeypatch):
    from sable.serving import supervisor as sup

    fake = {"t": 1000.0}
    monkeypatch.setattr(sup.time, "monotonic", lambda: fake["t"])
    key = "inst-b"
    for _ in range(3):
        sup.record_start_failure(key)
    assert sup.crash_loop_refused(key)
    fake["t"] += sup.CRASH_LOOP_WINDOW_S + 1
    assert not sup.crash_loop_refused(key)


def test_clear_crash_loop_allows_retry():
    from sable.serving import supervisor as sup

    key = "inst-c"
    for _ in range(3):
        sup.record_start_failure(key)
    assert sup.crash_loop_refused(key)
    assert sup.clear_crash_loop(key) is True
    assert not sup.crash_loop_refused(key)
    assert sup.crash_loop_status(key) is None
    assert sup.clear_crash_loop(key) is False  # idempotent


def test_start_instance_server_refuses_when_over_budget(tmp_path):
    from sable.serving import supervisor as sup

    key = "wedged"
    for _ in range(sup.CRASH_LOOP_MAX_FAILURES):
        sup.record_start_failure(key)
    with pytest.raises(sup.ServeError, match="crash loop"):
        sup.start_instance_server(None, {"id": key,
                                         "gguf": str(tmp_path / "x.gguf")})


def test_dashboard_row_surfaces_crash_loop(tmp_path, monkeypatch):
    from sable.core.config import SableConfig
    from sable.dashboard.app import DashboardApp
    from sable.serving import supervisor as sup

    home = tmp_path / "uf-cl"
    monkeypatch.setenv("SABLE_HOME", str(home))
    app = DashboardApp(SableConfig(llama_auto_setup=False), home=home)
    created = app.handle("POST", "/api/servers",
                         body={"name": "budgeted", "gguf": "/nope.gguf"})
    iid = created.body["server"]["id"]

    for _ in range(sup.CRASH_LOOP_MAX_FAILURES):
        sup.record_start_failure(iid)
    rows = app.handle("GET", "/api/servers").body["servers"]
    row = next(r for r in rows if r["id"] == iid)
    assert row["crash_loop"] is not None
    assert row["crash_loop"]["refused"] is True

    # User-initiated start clears the refusal before attempting (the spawn
    # then fails on the missing GGUF, but NOT with the crash-loop message).
    resp = app.handle("POST", f"/api/servers/{iid}/start", body={})
    assert resp.status == 500
    assert not sup.crash_loop_refused(iid)

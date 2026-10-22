"""WP2c — dashboard metric honesty: raw + honest variant pairs.

Every derived rate that can mislead must travel with its raw denominator /
evidence-strength variant. Unit tests for the pure helpers in
dashboard/metrics.py plus an endpoint test asserting BOTH sides of each pair
are present in /api/metrics, and the /api/status honesty field.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sable.core.config import SableConfig  # noqa: E402
from sable.dashboard.app import DashboardApp  # noqa: E402
from sable.dashboard.metrics import (  # noqa: E402
    liveness_strength,
    promotion_denominator,
    score_coverage,
)
from sable.registry import ModelRegistry  # noqa: E402
from sable.store.db import TraceRow, TraceStore  # noqa: E402


# ---------------------------------------------------------------- pure helpers

def test_score_coverage():
    assert score_coverage(100, 10) == 0.1
    assert score_coverage(3, 3) == 1.0
    assert score_coverage(0, 0) is None
    assert score_coverage(10, 0) == 0.0


def test_promotion_denominator():
    assert promotion_denominator(2, 1) == 3
    assert promotion_denominator(0, 0) == 0
    assert promotion_denominator(5, 2) == 7


def test_liveness_strength():
    assert liveness_strength("spawn") == "pid_probe"
    assert liveness_strength(None) == "pid_probe"
    assert liveness_strength("attach") == "state_file"


# ---------------------------------------------------------------- endpoint

@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "uf-honesty"
    monkeypatch.setenv("SABLE_HOME", str(h))
    return h


@pytest.fixture()
def app(home):
    return DashboardApp(SableConfig(), home=home)


def _seed(home):
    """4 traces, only 1 scored; 1 promote op (rate would read 100%)."""
    store = TraceStore(Path(home) / "buffer" / "sable.db")
    for i, score in enumerate((0.5, None, None, None)):
        tid = f"h{i}"
        p = Path(store.db_path).parent / f"{tid}.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"kind": "trace_header"}) + "\n")
        store.insert_trace(TraceRow(
            trace_id=tid, session_id=None,
            timestamp="2026-09-20T10:00:00+00:00",
            agent_profile="p", model_alias="m", model_version=None,
            endpoint_label=None, task_summary="t", status="success",
            outcome_score=score, num_tool_calls=0, tokens_in=1,
            tokens_out=1, duration_s=1.0, jsonl_path=str(p)))
    store.close()
    reg = ModelRegistry(Path(home) / "buffer" / "sable.db")
    m = reg.register_model("base-a")
    reg.set_validation(m.model_id, {"ok": True}, passed=True)
    reg.promote(m.model_id)  # single decision -> pass rate would say 100%
    reg.close()


def _sample(text: str, prefix: str):
    for line in text.splitlines():
        if line.startswith(prefix + " "):
            return line.rsplit(" ", 1)[1]
    return None


def test_metrics_endpoint_exposes_both_variants(app, home):
    _seed(home)
    resp = app.handle("GET", "/api/metrics", query={"refresh": "1"})
    assert resp.status == 200
    text = resp.raw.decode("utf-8")

    # pair 1: score mean (rate) + its raw denominator + coverage ratio
    assert _sample(text, "sable_outcome_score_mean") is not None
    assert _sample(text, "sable_traces_scored_total") == "1"
    cov = _sample(text, "sable_score_coverage_ratio")
    assert cov is not None and abs(float(cov) - 0.25) < 1e-9

    # pair 2: promotion pass RATE + its raw decision count (1/1 decisions)
    rate = _sample(text, "sable_promotion_pass_rate")
    assert rate is not None and abs(float(rate) - 1.0) < 1e-9
    assert _sample(text, "sable_promotion_decisions_total") == "1"

    # pair 3: instance liveness + evidence strength split
    assert "sable_instances_up{id=" in text
    assert 'sable_instance_up_strength{id="default",strength=' in text


def test_metrics_empty_home_still_valid(app):
    """Honest absence: no traces -> no coverage gauge, but exposition stays
    valid and the counters exist at zero."""
    resp = app.handle("GET", "/api/metrics", query={"refresh": "1"})
    text = resp.raw.decode("utf-8")
    assert "sable_score_coverage_ratio " not in text
    assert _sample(text, "sable_traces_scored_total") == "0"
    # promotion rate AND its denominator are honest-absent with zero decisions
    assert "sable_promotion_pass_rate " not in text
    assert "sable_promotion_decisions_total " not in text


def test_status_endpoint_carries_scored_denominator(app, home):
    _seed(home)
    resp = app.handle("GET", "/api/status")
    assert resp.status == 200
    buf = resp.body["buffer"]
    # raw counts beside the avg (the JS overview renders "1/4 scored")
    assert buf["total_traces"] == 4
    assert buf["scored_traces"] == 1
    assert buf["unscored_traces"] == 3

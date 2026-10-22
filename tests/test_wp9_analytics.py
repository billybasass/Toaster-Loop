"""WP9 tests — curation funnel analytics, validation telemetry, /metrics route.

9a: every Curater.curate() run appends one funnel snapshot under
    <home>/analytics/curation_funnels.jsonl with honest denominators; two runs
    accumulate history; prefix-overlap dups count as dedup_rejects.
9b: the analytics store persists validation-gate summaries; the API merges
    them chronologically with promote/rollback events from the registry ledger.
9c: GET /metrics on the dashboard's existing loopback server returns the
    Prometheus exposition (200 + text/plain; version=0.0.4), valid format
    (re-parsed by a tiny stdlib parser here), label escaping edge cases, vllm:
    mirrors ONLY where semantics match, no rate without its denominator, and
    all pre-existing families intact. Token auth parity: no backdoor.

All hermetic: SABLE_HOME at a tmp dir (conftest autouse), no network beyond
the loopback test server, no GPU.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Dict, Optional

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sable.core.config import SableConfig  # noqa: E402
from sable.curate import Curater, CurationConfig  # noqa: E402
from sable.dashboard.app import DashboardApp  # noqa: E402
from sable.registry import ModelRegistry  # noqa: E402
from sable.store import analytics  # noqa: E402
from sable.store.db import TraceRow, TraceStore  # noqa: E402


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "uf-wp9"
    monkeypatch.setenv("SABLE_HOME", str(h))
    return h


@pytest.fixture()
def app(home):
    return DashboardApp(SableConfig(), home=home)


# ------------------------------------------------------------------- seeding

def seed_trace(store: TraceStore, home: Path, *, trace_id: str, messages,
               status: str = "success", score: Optional[float] = 0.8,
               timestamp: str = "2026-09-01T10:00:00+00:00") -> None:
    """Minimal transcript + index row (same shape as tests/test_curate.py)."""
    jsonl_path = home / "buffer" / "2026" / "09" / f"{trace_id}.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "x", encoding="utf-8") as f:
        f.write(json.dumps({
            "kind": "trace_header", "schema_version": 1, "trace_id": trace_id,
            "session_id": None, "timestamp": timestamp,
            "agent_profile": "fallback", "model_alias": "test-model",
            "model_version": None, "endpoint_label": None,
            "task": (messages[0].get("content") if messages else "") or "",
        }) + "\n")
        for m in messages:
            f.write(json.dumps({"kind": "message", "message": m}) + "\n")
        f.write(json.dumps({
            "kind": "trace_footer", "schema_version": 1, "status": status,
            "completed": status == "success", "outcome_score": score,
            "num_tool_calls": 0, "tokens_in": None, "tokens_out": None,
            "duration_s": None,
        }) + "\n")
    store.insert_trace(TraceRow(
        trace_id=trace_id, session_id=None, timestamp=timestamp,
        agent_profile="fallback", model_alias="test-model",
        model_version=None, endpoint_label=None, task_summary="seed",
        status=status, outcome_score=score, num_tool_calls=0, tokens_in=None,
        tokens_out=None, duration_s=None, jsonl_path=str(jsonl_path),
    ))


OK_MSGS = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"},
]
DUP_MSGS = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "four"},
    {"role": "user", "content": "sure?"},
    {"role": "assistant", "content": "yes, four"},
]
OTHER_MSGS = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Capital of France?"},
    {"role": "assistant", "content": "Paris"},
]


# ================================================================== 9a funnel

def test_funnel_snapshot_persisted_with_honest_denominators(home):
    store = TraceStore(home / "buffer" / "sable.db")
    try:
        seed_trace(store, home, trace_id="t1", messages=OK_MSGS, score=0.9)
        seed_trace(store, home, trace_id="t2", messages=OTHER_MSGS, score=None)
        res = Curater(store, home).curate(CurationConfig(min_score=0.5),
                                          name="wp9")
    finally:
        store.close()

    recs = analytics.read_curation_funnels(home)
    assert len(recs) == 1
    r = recs[0]
    assert r["dataset_id"] == res.dataset_id
    assert r["traces_scanned"] == 2
    assert r["loaded"] == 2
    assert r["curated"] == res.claimed
    # honest denominators: histogram carries scored/unscored beside buckets
    hist = r["score_histogram"]
    assert hist["scored"] == 1 and hist["unscored"] == 1
    assert hist["total"] == 2
    assert sum(hist["bins"]) == hist["scored"]
    # 0.9 -> bin index 9
    assert hist["bins"][9] == 1
    assert r["ts"] and r["created_at"]


def test_funnel_history_accumulates_over_runs(home):
    store = TraceStore(home / "buffer" / "sable.db")
    try:
        seed_trace(store, home, trace_id="a1", messages=OK_MSGS)
        Curater(store, home).curate(CurationConfig(), name="one")
        seed_trace(store, home, trace_id="b1", messages=OTHER_MSGS,
                   timestamp="2026-09-01T11:00:00+00:00")
        Curater(store, home).curate(CurationConfig(), name="two")
    finally:
        store.close()
    recs = analytics.read_curation_funnels(home)
    assert len(recs) == 2  # append-only history
    ids = [r["dataset_id"] for r in recs]
    assert len(set(ids)) == 2
    # dataset filter returns only that dataset's runs
    one = analytics.read_curation_funnels(home, dataset=ids[0])
    assert [r["dataset_id"] for r in one] == [ids[0]]


def test_funnel_dedup_rejects_counted_on_dup_fixture(home):
    store = TraceStore(home / "buffer" / "sable.db")
    try:
        # same first-user turn + profile -> one conversation lineage; the
        # longer (DUP) trace wins, the shorter one is a dedup reject.
        seed_trace(store, home, trace_id="d1", messages=OK_MSGS)
        seed_trace(store, home, trace_id="d2", messages=DUP_MSGS,
                   timestamp="2026-09-01T10:30:00+00:00")
        res = Curater(store, home).curate(CurationConfig(), name="dupes")
    finally:
        store.close()
    r = analytics.read_curation_funnels(home)[-1]
    assert r["traces_scanned"] == 2
    assert r["dedup_rejects"] == 1
    assert r["curated"] == res.claimed == 1


def test_funnel_api_returns_history_and_latest_diff(app, home):
    store = TraceStore(home / "buffer" / "sable.db")
    try:
        seed_trace(store, home, trace_id="x1", messages=OK_MSGS)
        Curater(store, home).curate(CurationConfig(), name="r1")
        seed_trace(store, home, trace_id="x2", messages=OTHER_MSGS,
                   timestamp="2026-09-02T10:00:00+00:00")
        Curater(store, home).curate(CurationConfig(), name="r2")
    finally:
        store.close()

    resp = app.handle("GET", "/api/analytics/curation")
    assert resp.status == 200
    body = resp.body
    assert len(body["history"]) == 2
    # latest_diff reuses dataset_diff over the two newest datasets
    diff = body["latest_diff"]
    assert diff is not None and "error" not in diff
    assert set(diff) >= {"trace_diff", "counts", "identical"}

    resp1 = app.handle("GET", "/api/analytics/curation",
                       query={"dataset": body["history"][0]["dataset_id"]})
    assert resp1.status == 200
    assert len(resp1.body["history"]) == 1
    # single dataset -> no pair to diff, honest null (not an error)
    assert resp1.body["latest_diff"] is None


# ============================================================ 9b validation

def _fake_gate_report(passed: int, total: int, ok: bool) -> Dict:
    results = [{"prompt": f"p{i}", "pass": i < passed, "checks": [
        {"kind": "substring", "pass": i < passed}]} for i in range(total)]
    return {"ok": ok, "pass_rate": passed / total if total else 0.0,
            "min_pass_rate": 1.0, "total": total, "passed": passed,
            "results": results, "started_at": "2026-10-01T10:00:00+00:00",
            "finished_at": "2026-10-01T10:01:00+00:00"}


def test_validation_summary_counts_and_skips():
    rep = _fake_gate_report(1, 3, False)
    rep["results"][2] = {"prompt": "p2", "pass": False,
                         "error": "ConnectionError: boom"}
    rec = analytics.summarize_validation_report(rep, "v7", iteration=2)
    assert rec["model_id"] == "v7" and rec["iteration"] == 2
    assert rec["passed"] == 1 and rec["failed"] == 1 and rec["skipped"] == 1
    assert rec["gate_verdict"] == "fail"
    assert rec["ran_tasks"] == 2  # transport-skips excluded from the ran set


def test_validation_api_merges_promotion_events_chronologically(app, home):
    # two fake gate runs, explicitly timestamped for deterministic ordering
    analytics.record_validation_run(home, {
        "ts": "2026-10-01T10:00:00+00:00", **analytics.summarize_validation_report(
            _fake_gate_report(1, 2, False), "v1")})
    analytics.record_validation_run(home, {
        "ts": "2026-10-03T10:00:00+00:00", **analytics.summarize_validation_report(
            _fake_gate_report(2, 2, True), "v1")})

    # registry: v1 validated -> promoted (ledger records both ops)
    reg = ModelRegistry(home / "buffer" / "sable.db")
    try:
        reg.register_model("base-a", dataset_id=None)
        reg.set_validation("v1", {"ok": True}, passed=True, actor="test")
        reg.promote("v1", actor="test", intent="wp9 test promote")
    finally:
        reg.close()

    resp = app.handle("GET", "/api/analytics/validation", query={"model": "v1"})
    assert resp.status == 200
    body = resp.body
    assert [r["pass_rate"] for r in body["history"]] == [0.5, 1.0]
    kinds = [e["op"] for e in body["events"]]
    assert "promote" in kinds
    tl = body["timeline"]
    ts = [e["ts"] for e in tl]
    assert ts == sorted(ts)  # chronological merge
    assert {e["kind"] for e in tl} == {"validation", "promotion"}


# ================================================================ 9c metrics

def _metrics_text(app) -> str:
    return app.handle("GET", "/api/metrics").raw.decode("utf-8")


def test_metrics_exposition_parses_back_with_help_type(app, home):
    # seed one trace so counters are non-trivial
    store = TraceStore(home / "buffer" / "sable.db")
    try:
        seed_trace(store, home, trace_id="m1", messages=OK_MSGS)
    finally:
        store.close()
    text = _metrics_text(app)
    assert text.endswith("\n")
    # HELP immediately followed by TYPE for every family that has samples
    lines = text.splitlines()
    helped = set()
    typed = set()
    for i, line in enumerate(lines):
        if line.startswith("# HELP "):
            name = line.split()[2]
            assert lines[i + 1].startswith(f"# TYPE {name} "), \
                f"HELP {name} not followed by TYPE"
            helped.add(name)
        elif line.startswith("# TYPE "):
            typed.add(line.split()[2])
    assert helped == typed
    # every sample line's family (minus _sum/_count/quantile suffixes) is known
    for line in lines:
        if line.startswith("#") or not line:
            continue
        head = line.split("{")[0].split(" ")[0]
        base = head
        for suffix in ("_sum", "_count"):
            if head.endswith(suffix):
                base = head[:-len(suffix)]
                break
        assert base in helped or head in helped, f"orphan sample {head}"


def test_metrics_label_escaping_quotes_backslash(app, home):
    """Instance id with quote/backslash/newline-free edge chars must be escaped
    per spec: \\\\ , \\" inside label values; the raw line stays parseable."""
    inst_dir = home / "serving" / "instances"
    inst_dir.mkdir(parents=True)
    (inst_dir / "weird.json").write_text(json.dumps(
        {"id": 'a"b\\c', "name": "weird", "role": "custom"}), encoding="utf-8")
    text = _metrics_text(app)
    # the escaped form appears; the raw quote inside a value never does
    assert '{id="a\\"b\\\\c"}' in text
    # every label value on that line is well-formed: value chars are either
    # non-quote/non-backslash or valid escapes
    for line in text.splitlines():
        if 'a\\"b' in line:
            assert re.fullmatch(
                r'sable_instances_up\{id="(?:[^"\\]|\\.)*"\} [0-9]+', line), line
            break
    else:
        pytest.fail("escaped instance sample not found")


def test_metrics_vllm_mirrors_only_where_semantics_match(app, home):
    # one scored trace so the WP2c coverage ratio family is emitted at all,
    # and one promotion decision so the pass-rate denominator pair exists
    store = TraceStore(home / "buffer" / "sable.db")
    try:
        seed_trace(store, home, trace_id="mm1", messages=OK_MSGS, score=0.5)
    finally:
        store.close()
    reg = ModelRegistry(home / "buffer" / "sable.db")
    try:
        reg.register_model("base-a")
        reg.set_validation("v1", {"ok": True}, passed=True, actor="test")
        reg.promote("v1", actor="test")
    finally:
        reg.close()
    text = _metrics_text(app)
    fams = {l.split()[2] for l in text.splitlines() if l.startswith("# TYPE")}
    # semantics match -> mirrors present
    assert "vllm:prompt_tokens_total" in fams
    assert "vllm:generation_tokens_total" in fams
    # semantics differ / not measurable -> deliberately absent
    assert "vllm:num_requests_total" not in fams
    assert "vllm:time_to_first_token_seconds" not in fams
    assert "vllm:num_requests_waiting" not in fams
    # canonical sable families intact (WP2 honest variants included)
    for fam in ("sable_traces_total", "sable_traces_scored_total",
                "sable_score_coverage_ratio", "sable_promotion_decisions_total",
                "sable_instance_up_strength", "sable_requests_total",
                "sable_jobs_queued"):
        assert fam in fams, f"missing family {fam}"


def test_metrics_token_auth_parity_on_metrics_route(home):
    """GET /metrics enforces the dashboard token exactly like /api/metrics —
    no unauthenticated backdoor on the scrape alias."""
    from sable.dashboard.server import BIND_HOST, make_server

    httpd = make_server(SableConfig(), port=0, home=home, token="scr4pe")
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    base = f"http://{BIND_HOST}:{httpd.server_address[1]}"
    try:
        import httpx
        assert httpx.get(base + "/metrics", timeout=5).status_code == 401
        r = httpx.get(base + "/metrics", timeout=5,
                      headers={"X-Sable-Token": "scr4pe"})
        assert r.status_code == 200
        assert r.headers["content-type"] == "text/plain; version=0.0.4"
        assert "# HELP" in r.text
        # GET query token allowed (scrapers cannot set headers)
        assert httpx.get(base + "/metrics?token=scr4pe",
                         timeout=5).status_code == 200
        # cross-site Origin still blocked on the alias
        assert httpx.get(base + "/metrics", timeout=5,
                         headers={"Origin": "http://evil.example"},
                         ).status_code == 403
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_metrics_route_200_content_type_unauth_disabled(home):
    """HTTP-level happy path: token disabled -> /metrics is scrapeable."""
    from sable.dashboard.server import BIND_HOST, make_server

    httpd = make_server(SableConfig(), port=0, home=home, token="")
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    base = f"http://{BIND_HOST}:{httpd.server_address[1]}"
    try:
        import httpx
        r = httpx.get(base + "/metrics", timeout=5)
        assert r.status_code == 200
        assert r.headers["content-type"] == "text/plain; version=0.0.4"
        # /api/metrics still works identically (pre-existing route unchanged)
        r2 = httpx.get(base + "/api/metrics", timeout=5)
        assert r2.status_code == 200
        assert r2.text == r.text  # same TTL-cached body
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_metrics_no_rate_without_denominator(app):
    """WP9 honesty rule over the rendered text: every *_ratio / *_rate gauge
    must be accompanied by its denominator counter family."""
    text = _metrics_text(app)
    fams = {l.split()[2] for l in text.splitlines() if l.startswith("# TYPE")}
    pairs = {
        "sable_score_coverage_ratio": "sable_traces_scored_total",
        "sable_promotion_pass_rate": "sable_promotion_decisions_total",
    }
    for rate, denom in pairs.items():
        if rate in text:  # only emitted when non-empty
            assert f"# TYPE {denom} counter" in text, \
                f"rate {rate} rendered without denominator {denom}"
    assert fams  # sanity: something was parsed

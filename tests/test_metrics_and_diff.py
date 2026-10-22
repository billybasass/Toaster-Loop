"""Feature-bundle tests: /api/metrics (Prometheus exposition) + dataset diff.

Item A: GET /api/metrics returns a hand-rolled Prometheus text exposition —
spec-valid lines, HELP/TYPE for every family, values matching seeded fixtures,
correct content type, TTL-cache behavior, and token-auth parity over the real
server (no backdoor).

Item B: GET /api/datasets/diff?a=&b= compares two dataset versions from
manifest trace-id sets + per-file sha256 hashes; missing datasets -> 404,
malformed manifests / missing params -> 400; identical datasets -> all
unchanged with hash-equal true. Same semantics via `sable datasets diff`.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sable.core.config import SableConfig  # noqa: E402
from sable.curate.dataset_diff import (diff_datasets)
from sable.dashboard.app import DashboardApp  # noqa: E402
from sable.registry import ModelRegistry  # noqa: E402
from sable.store.db import TraceRow, TraceStore  # noqa: E402


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "uf-md"
    monkeypatch.setenv("SABLE_HOME", str(h))
    return h


@pytest.fixture()
def app(home):
    return DashboardApp(SableConfig(), home=home)


# ============================================================ text-format gate

# Prometheus text exposition spec patterns (metric/label names + sample
# syntax). Family names here are sable_* so [a-zA-Z_:] covers them.
_RE_NAME = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*$")
_RE_HELP = re.compile(r"^# HELP [a-zA-Z_:][a-zA-Z0-9_:]* .+$")
_RE_TYPE = re.compile(
    r"^# TYPE [a-zA-Z_:][a-zA-Z0-9_:]* "
    r"(counter|gauge|summary|histogram|untyped)$")
_RE_LABELS = (r"\{(?:[a-zA-Z_][a-zA-Z0-9_]*=\"(?:[^\"\\]|\\.)*\")"
              r"(?:,[a-zA-Z_][a-zA-Z0-9_]*=\"(?:[^\"\\]|\\.)*\")*\}")
_RE_SAMPLE = re.compile(
    r"^[a-zA-Z_:][a-zA-Z0-9_:]*(?:" + _RE_LABELS + r")?"
    r" -?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?$")


def assert_valid_exposition(text: str) -> Dict[str, str]:
    """Validate the whole body against the text-format grammar.

    Returns {family: type}. Raises AssertionError with the offending line.
    """
    assert text.endswith("\n"), "exposition must end with a newline"
    types: Dict[str, str] = {}
    pending: Optional[str] = None  # family awaiting its TYPE line
    for i, line in enumerate(text.splitlines()):
        if not line.strip():
            continue
        if line.startswith("# HELP "):
            assert _RE_HELP.match(line), f"line {i}: bad HELP: {line!r}"
            pending = line.split()[2]
            assert pending not in types, f"duplicate HELP for {pending}"
            continue
        if line.startswith("# TYPE "):
            m = _RE_TYPE.match(line)
            assert m, f"line {i}: bad TYPE: {line!r}"
            fam = line.split()[2]
            assert pending == fam, \
                f"line {i}: TYPE for {fam} without an adjacent HELP"
            types[fam] = m.group(1)
            pending = None
            continue
        assert line.startswith("#") is False or line.startswith("# EOF"), \
            f"line {i}: unknown comment form: {line!r}"
        m = _RE_SAMPLE.match(line)
        assert m, f"line {i}: bad sample line: {line!r}"
        family = line.split("{")[0].split(" ")[0]
        # summary/histogram sub-samples (_sum/_count/quantile/bucket) belong
        # to the parent family declared by TYPE.
        root = family
        for suffix in ("_sum", "_count", "_bucket"):
            if root.endswith(suffix) and root[: -len(suffix)] in types:
                root = root[: -len(suffix)]
        assert root in types, f"line {i}: sample {family} has no TYPE"
    return types


def _seed_rich_home(home):
    """Traces + jobs (+gate reason) + promotion/rollback + rft/mined ledgers."""
    store = TraceStore(Path(home) / "buffer" / "sable.db")
    for n, status in ((5, "success"), (2, "failure"), (1, "partial")):
        for i in range(n):
            tid = f"{status[0]}{i}"
            p = Path(store.db_path).parent / f"{tid}.jsonl"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"kind": "trace_header"}) + "\n")
            store.insert_trace(TraceRow(
                trace_id=tid, session_id=None,
                timestamp="2026-09-20T10:00:00+00:00",
                agent_profile="p", model_alias="m", model_version=None,
                endpoint_label=None, task_summary="t", status=status,
                outcome_score=0.5, num_tool_calls=0, tokens_in=7,
                tokens_out=10, duration_s=float(i + 1), jsonl_path=str(p)))
    # rft ledger: one kept, two sampled-but-unkept
    keys = [("k1", "seed", None), ("k2", "seed", None), ("k3", "seed", None)]
    store.claim_rft_prompts(keys)
    store.mark_rft_prompts_emitted(["k1", "k2", "k3"], "ds-x", kept_keys=["k1"])
    # mined goldens: one mined, one dropped
    store.claim_mined_goldens(["g1", "g2"])
    store.record_mined_golden("g1", prompt_key="p1", status="mined",
                              enabled=True)
    store.record_mined_golden("g2", prompt_key="p2", status="dropped",
                              reason="no stable lines")
    # gate events (v9): one durable block episode — seeds the cumulative
    # counter; the job param below is the current-state gauge side.
    store.record_gate_event("seed-blocked-1", "disk gate")
    store.close()

    reg = ModelRegistry(Path(home) / "buffer" / "sable.db")
    j1 = reg.enqueue_job("sft")
    reg.claim_next_queued(pid=4242)          # -> running, started_at set
    reg.enqueue_job("curate")                # stays queued
    reg.set_job_param(j1.job_id, "_gate_reason", "disk gate")
    m1 = reg.register_model("base-a")
    reg.set_validation(m1.model_id, {"ok": True}, passed=True)
    reg.promote(m1.model_id)                 # ledger op promote
    m2 = reg.register_model("base-a")
    reg.set_validation(m2.model_id, {"ok": True}, passed=True)
    reg.promote(m2.model_id)
    reg.rollback(base_model="base-a")        # ledger op rollback
    reg.close()


def _metrics(app, refresh=True):
    """GET /api/metrics through the router (query goes in as a dict —
    DashboardApp.handle strips nothing; server.py parses the query string)."""
    q = {"refresh": "1"} if refresh else {}
    resp = app.handle("GET", "/api/metrics", query=q)
    assert resp.status == 200, resp.body
    return resp


def _sample_value(text: str, prefix: str, labels: str = "") -> Optional[str]:
    """Value of one exact sample line (prefix + optional label set)."""
    needle = prefix + (labels or "")
    for line in text.splitlines():
        if line.startswith(needle + " ") or line.startswith(needle + "{"):
            if labels and not line.startswith(prefix + labels):
                continue
            return line.rsplit(" ", 1)[1]
    return None


# ---------------------------------------------------------------- metrics: A

def test_metrics_exposition_valid_and_complete(app, home):
    _seed_rich_home(home)
    resp = _metrics(app)
    text = resp.raw.decode("utf-8")
    types = assert_valid_exposition(text)
    expected = {
        "sable_traces_total": "counter",
        "sable_outcome_score_mean": "gauge",
        "sable_tokens_in_total": "counter",
        "sable_tokens_out_total": "counter",
        "sable_tokens_per_second": "summary",
        "sable_request_latency_seconds": "summary",
        "sable_jobs_total": "counter",
        "sable_job_queue_wait_seconds": "summary",
        "sable_gate_blocks_total": "counter",
        "sable_jobs_gate_blocked": "gauge",
        "sable_promotions_total": "counter",
        "sable_promotion_pass_rate": "gauge",
        "sable_instances_up": "gauge",
        "sable_rft_samples_kept_total": "counter",
        "sable_rft_samples_dropped_total": "counter",
        "sable_mined_goldens_total": "counter",
    }
    assert expected == {k: types[k] for k in expected}, sorted(types)
    # every family has HELP and TYPE (grammar check above already enforced
    # adjacency; this enforces presence per family).
    helps = {l.split()[2] for l in text.splitlines() if l.startswith("# HELP")}
    tys = {l.split()[2] for l in text.splitlines() if l.startswith("# TYPE")}
    assert helps == tys == set(types)


def test_metrics_values_match_seed(app, home):
    _seed_rich_home(home)
    text = _metrics(app).raw.decode("utf-8")
    assert _sample_value(text, "sable_traces_total",
                         '{status="success"}') == "5"
    assert _sample_value(text, "sable_traces_total",
                         '{status="failure"}') == "2"
    assert _sample_value(text, "sable_traces_total",
                         '{status="partial"}') == "1"
    assert _sample_value(text, "sable_traces_total",
                         '{status="unknown"}') == "0"
    # 8 traces x tokens_in=7 / tokens_out=10
    assert _sample_value(text, "sable_tokens_in_total") == "56"
    assert _sample_value(text, "sable_tokens_out_total") == "80"
    assert _sample_value(text, "sable_outcome_score_mean") == "0.5"
    # durations: success 1..5, failure 1..2, partial 1 -> count 8.
    assert _sample_value(text, "sable_request_latency_seconds_count") == "8"
    # throughput 10/d sorted [2, 2.5, 3.3333, 5, 5, 10, 10, 10] -> p50 rank 4 = 5
    assert _sample_value(
        text, "sable_tokens_per_second{quantile=\"0.5\"}") == "5"
    # jobs: one running sft + one queued curate (state counts)
    assert _sample_value(text, "sable_jobs_total",
                         '{kind="sft",state="running"}') == "1"
    assert _sample_value(text, "sable_jobs_total",
                         '{kind="curate",state="queued"}') == "1"
    # gate blocks: counter = durable block episodes (one seeded event);
    # gauge = jobs currently carrying a non-null _gate_reason (the same one)
    assert _sample_value(text, "sable_gate_blocks_total") == "1"
    assert _sample_value(text, "sable_jobs_gate_blocked") == "1"
    # promotions: 2 promote ops, 1 rollback op in the ledger -> pass-rate 2/3
    assert _sample_value(text, "sable_promotions_total",
                         '{result="pass"}') == "2"
    assert _sample_value(text, "sable_promotions_total",
                         '{result="fail"}') == "1"
    rate_s = _sample_value(text, "sable_promotion_pass_rate")
    assert rate_s is not None
    rate = float(rate_s)
    assert abs(rate - 2 / 3) < 1e-6
    # ledgers
    assert _sample_value(text, "sable_rft_samples_kept_total") == "1"
    assert _sample_value(text, "sable_rft_samples_dropped_total") == "2"
    assert _sample_value(text, "sable_mined_goldens_total",
                         '{status="mined"}') == "1"
    assert _sample_value(text, "sable_mined_goldens_total",
                         '{status="dropped"}') == "1"
    # default instance exists (config-derived) but is not running
    assert _sample_value(text, "sable_instances_up",
                         '{id="default"}') == "0"


def test_metrics_empty_home_is_honest(app):
    resp = _metrics(app)
    text = resp.raw.decode("utf-8")
    types = assert_valid_exposition(text)
    assert "sable_traces_total" in types
    # zero-valued status counters still present (valid empty family)
    assert _sample_value(text, "sable_traces_total",
                         '{status="success"}') == "0"
    # score mean honest-absent when nothing is scored
    assert "sable_outcome_score_mean " not in text


def test_metrics_content_type(app):
    resp = _metrics(app)
    assert resp.content_type == "text/plain; version=0.0.4"


def test_metrics_ttl_cache(app, home):
    _seed_rich_home(home)
    body1 = _metrics(app).raw.decode()
    # New data lands in the DB *after* the first scrape...
    store = TraceStore(Path(home) / "buffer" / "sable.db")
    p = Path(store.db_path).parent / "late.jsonl"
    p.write_text(json.dumps({"kind": "trace_header"}) + "\n")
    store.insert_trace(TraceRow(
        trace_id="late", session_id=None,
        timestamp="2026-09-21T10:00:00+00:00", agent_profile="p",
        model_alias="m", model_version=None, endpoint_label=None,
        task_summary="t", status="success", outcome_score=0.5,
        num_tool_calls=0, tokens_in=1, tokens_out=1, duration_s=1.0,
        jsonl_path=str(p)))
    store.close()
    # ...still served from the TTL cache...
    body2 = _metrics(app, refresh=False).raw.decode()
    assert body2 == body1
    # ...until an explicit refresh bypasses it.
    body3 = _metrics(app).raw.decode()
    assert _sample_value(body3, "sable_traces_total",
                         '{status="success"}') == "6"
    assert _sample_value(body2, "sable_traces_total",
                         '{status="success"}') == "5"


def test_metrics_percentile_matches_insights(app, home):
    """Quantiles reuse insights.percentile (ceil nearest-rank)."""
    from sable.store.insights import percentile

    store = TraceStore(Path(home) / "buffer" / "sable.db")
    durs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    for i, d in enumerate(durs):
        tid = f"p{i}"
        p = Path(store.db_path).parent / f"{tid}.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"kind": "trace_header"}) + "\n")
        store.insert_trace(TraceRow(
            trace_id=tid, session_id=None,
            timestamp="2026-09-20T10:00:00+00:00", agent_profile="p",
            model_alias="m", model_version=None, endpoint_label=None,
            task_summary="t", status="success", outcome_score=None,
            num_tool_calls=0, tokens_in=None, tokens_out=None,
            duration_s=d, jsonl_path=str(p)))
    store.close()
    text = _metrics(app).raw.decode()
    srt = sorted(durs)
    assert _sample_value(
        text, "sable_request_latency_seconds{quantile=\"0.5\"}") \
        == _fmt(percentile(srt, 50))
    assert _sample_value(
        text, "sable_request_latency_seconds{quantile=\"0.9\"}") \
        == _fmt(percentile(srt, 90))
    assert _sample_value(
        text, "sable_request_latency_seconds{quantile=\"0.95\"}") \
        == _fmt(percentile(srt, 95))


def _fmt(v) -> str:
    from sable.dashboard.metrics import _fmt_num
    return _fmt_num(v)


# ------------------------------------------------------------------ auth: A

def _start_server_with_token(home, token=None):
    from sable.dashboard.server import BIND_HOST, make_server

    httpd = make_server(SableConfig(), port=0, home=home, token=token)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, f"http://{BIND_HOST}:{httpd.server_address[1]}"


def test_metrics_token_auth_parity(home):
    """No backdoor: with a token configured, /api/metrics demands it exactly
    like /api/status (header or GET query param)."""
    httpd, base = _start_server_with_token(home, token="scr4pe")
    try:
        import httpx

        assert httpx.get(base + "/api/metrics", timeout=5).status_code == 401
        r = httpx.get(base + "/api/metrics", timeout=5,
                      headers={"X-Sable-Token": "scr4pe"})
        assert r.status_code == 200
        assert r.headers["content-type"] == "text/plain; version=0.0.4"
        # GET query token allowed for scrapers (same rule as SSE/EventSource)
        assert httpx.get(base + "/api/metrics?token=scr4pe",
                         timeout=5).status_code == 200
    finally:
        httpd.shutdown()
        httpd.server_close()


# ================================================================ dataset diff

def _make_dataset(root: Path, ds_id: str, *, sft_ids=None, dpo_ids=None,
                  created_at="2026-09-01T00:00:00+00:00", source=None,
                  sft_lines=None, extra_lines=None):
    """Write a dataset dir with manifest (+optional raw-file overrides)."""
    d = root / ds_id
    d.mkdir(parents=True, exist_ok=True)
    files: Dict[str, Any] = {
        "sft.jsonl": {"count": len(sft_ids or []),
                      "trace_ids": list(sft_ids or []), "format": "sharegpt"},
        "dpo.jsonl": {"count": len(dpo_ids or []),
                      "trace_ids": list(dpo_ids or []),
                      "format": "trl-preference"},
    }
    manifest = {"dataset_id": ds_id, "created_at": created_at,
                "files": files}
    if source:
        manifest["source"] = source
    (d / "manifest.json").write_text(json.dumps(manifest))
    if sft_lines is not None:
        (d / "sft.jsonl").write_text("".join(sft_lines))
    elif sft_ids:
        (d / "sft.jsonl").write_text(
            "".join(json.dumps({"conversations": [], "trace_id": t}) + "\n"
                    for t in sft_ids))
    if dpo_ids:
        (d / "dpo.jsonl").write_text(
            "".join(json.dumps({"metadata": {"chosen_trace_id": t}}) + "\n"
                    for t in dpo_ids))
    if extra_lines:
        (d / "extra.jsonl").write_text("".join(extra_lines))
    return d


def _pair(home):
    """Two datasets: overlap {t2,t3}, unique a:{t1}, unique b:{t4,t5}; one
    modified file (sft content differs), identical dpo file."""
    root = Path(home) / "datasets"
    _make_dataset(root, "ds-a", sft_ids=["t1", "t2", "t3"], dpo_ids=["d1"],
                  created_at="2026-09-01T00:00:00+00:00")
    _make_dataset(root, "ds-b", sft_ids=["t2", "t3", "t4", "t5"],
                  dpo_ids=["d1"], created_at="2026-09-08T00:00:00+00:00")
    # make ds-a's sft.jsonl content differ from what its ids would produce
    (root / "ds-a" / "sft.jsonl").write_text(
        json.dumps({"conversations": [], "trace_id": "t1"}) + "\n"
        + json.dumps({"conversations": [{"from": "human", "value": "old"}],
                      "trace_id": "t2"}) + "\n"
        + json.dumps({"conversations": [], "trace_id": "t3"}) + "\n")
    return root


def test_dataset_diff_sets_and_hashes(app, home):
    _pair(home)
    resp = app.handle("GET", "/api/datasets/diff",
                      query={"a": "ds-a", "b": "ds-b"})
    assert resp.status == 200
    body = resp.body
    td = body["trace_diff"]["sft.jsonl"]
    assert td["added_count"] == 2 and sorted(td["added_sample"]) == ["t4", "t5"]
    assert td["removed_count"] == 1 and td["removed_sample"] == ["t1"]
    assert td["unchanged_count"] == 2 and td["unchanged_sample"] == ["t2", "t3"]
    # hash verdicts: sft differs (modified file), dpo equal
    assert body["files"]["sft.jsonl"]["hash_equal"] is False
    assert body["files"]["dpo.jsonl"]["hash_equal"] is True
    # count deltas: sft 3 -> 4, dpo 1 -> 1
    assert body["counts"]["sft_delta"] == 1
    assert body["counts"]["dpo_delta"] == 0
    # row-level content diff sees the modified t2 row + added t5-side rows
    rows = body["rows"]["sft.jsonl"]
    assert rows["content_equal"] is False
    assert rows["rows_added"] >= 1 and rows["rows_removed"] >= 1
    # metadata: both curate-side (no source) -> equal; created delta one week
    assert body["source_equal"] is True
    assert body["created_at_delta_s"] == 7 * 86400.0
    assert body["identical"] is False


def test_dataset_diff_source_field(app, home):
    root = Path(home) / "datasets"
    _make_dataset(root, "cur", sft_ids=["t1"])
    _make_dataset(root, "rft", sft_ids=["t1"], source="rft")
    body = app.handle("GET", "/api/datasets/diff",
                      query={"a": "cur", "b": "rft"}).body
    assert body["source_equal"] is False
    assert body["b"]["source"] == "rft" and body["a"]["source"] is None


def test_dataset_diff_identical(app, home):
    root = Path(home) / "datasets"
    _make_dataset(root, "x1", sft_ids=["t1", "t2"], dpo_ids=["d1"])
    _make_dataset(root, "x2", sft_ids=["t1", "t2"], dpo_ids=["d1"])
    body = app.handle("GET", "/api/datasets/diff",
                      query={"a": "x1", "b": "x2"}).body
    assert body["identical"] is True
    td = body["trace_diff"]["sft.jsonl"]
    assert td["added_count"] == 0 and td["removed_count"] == 0
    assert td["unchanged_count"] == 2
    assert all(f["hash_equal"] is True for f in body["files"].values())
    # same-id compare is trivially identical
    same = app.handle("GET", "/api/datasets/diff",
                      query={"a": "x1", "b": "x1"}).body
    assert same["identical"] is True


def test_dataset_diff_missing_ids_404(app, home):
    root = _pair(home)
    r = app.handle("GET", "/api/datasets/diff",
                   query={"a": "ds-a", "b": "nope"})
    assert r.status == 404
    assert "no dataset" in r.body["error"]["message"]
    # a directory without manifest.json is not a dataset -> 404 as well
    (root / "hollow").mkdir()
    r2 = app.handle("GET", "/api/datasets/diff",
                    query={"a": "hollow", "b": "ds-a"})
    assert r2.status == 404


def test_dataset_diff_malformed_manifest_400(app, home):
    root = _pair(home)
    (root / "bad" ).mkdir()
    (root / "bad" / "manifest.json").write_text("{not json")
    r = app.handle("GET", "/api/datasets/diff",
                   query={"a": "bad", "b": "ds-a"})
    assert r.status == 400
    assert r.body["error"]["code"] == "bad_request"


def test_dataset_diff_missing_params_400(app, home):
    _pair(home)
    assert app.handle("GET", "/api/datasets/diff",
                      query={"a": "ds-a"}).status == 400
    assert app.handle("GET", "/api/datasets/diff", query={}).status == 400


def test_dataset_diff_library_and_cli(home, monkeypatch, capsys):
    root = _pair(home)
    d = diff_datasets(root, "ds-a", "ds-b")
    assert d["trace_diff"]["sft.jsonl"]["added_count"] == 2
    # deterministic: two runs byte-equal payloads
    assert json.dumps(d, sort_keys=True) == json.dumps(
        diff_datasets(root, "ds-a", "ds-b"), sort_keys=True)

    from sable.cli import cmd_datasets_diff
    import argparse
    monkeypatch.setenv("SABLE_HOME", str(home))
    ns = argparse.Namespace(verbose=False, config=None, a="ds-a", b="ds-b")
    assert cmd_datasets_diff(ns) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["counts"]["sft_delta"] == 1
    # missing dataset -> exit 1 with an error line
    ns2 = argparse.Namespace(verbose=False, config=None, a="ds-a", b="gone")
    assert cmd_datasets_diff(ns2) == 1


def test_datasets_cli_parser_has_diff():
    from sable.cli import build_parser

    ns = build_parser().parse_args(["datasets", "diff", "a1", "b2"])
    assert ns.func.__name__ == "cmd_datasets_diff"
    assert (ns.a, ns.b) == ("a1", "b2")

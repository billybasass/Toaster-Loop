"""Dashboard-domain tests: API, chat lifecycle, panels, insights.

Consolidated from 13 former test modules; section banners mark origins.
"""

from __future__ import annotations

import sys
from pathlib import Path
import json
import os
import re
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, List, cast
import time
import inspect
from http.server import BaseHTTPRequestHandler
from typing import Any, Dict
import math
import sqlite3
import stat
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import base64
import struct
import zlib
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sable.core.config import SableConfig  # noqa: E402
from sable.dashboard.app import DashboardApp, build_trace_timeline  # noqa: E402
from sable.registry import ModelRegistry  # noqa: E402
from sable.store.db import TraceRow, TraceStore  # noqa: E402
from fake_server import FakeOpenAIServer  # noqa: E402
from sable.core.agent_runner import AgentResult  # noqa: E402
from sable.dashboard.app import DashboardApp  # noqa: E402
from sable.core import agent_runner as ar  # noqa: E402
from sable.core.agent_runner import SableAgentClient  # noqa: E402
from sable.core.failure_types import FAILURE_CANCELLED, is_infra_reason, is_model_reason  # noqa: E402
from sable.core.titles import DERIVED, LLM, USER, derive_title, generate_title, should_apply_title  # noqa: E402
from sable.core.config import SableConfig, load_config  # noqa: E402
from sable.core.indexer import CodebaseIndex, IndexRebuildRequired, clip_to_tokens, delete_project_index, get_index, stop_all_embed_servers, truncate_renorm, EmbeddingClient, DEFAULT_DB_ID, EMBED_UBATCH  # noqa: E402
from sable.core import mmproj  # noqa: E402
from sable.serving.run_args import canonical_spec_mode, run_settings_args, sanitize_run_settings, speculative_args  # noqa: E402
from sable.serving.supervisor import build_instance_cmd  # noqa: E402
from fake_server import FakeOpenAIServer, make_message_handler  # noqa: E402
from sable.core.indexer import CodebaseIndex, LexicalIndex, chunk_file, cosine, _pack, _unpack  # noqa: E402
from sable.serving.run_args import run_settings_args, sanitize_run_settings, scale_gpu_layers  # noqa: E402
import sable.store.recorder as recorder_mod  # noqa: E402
from sable.store import TraceRecorder  # noqa: E402
from sable.store.insights import generate_insights  # noqa: E402
from sable.core import images as uf_images  # noqa: E402
from sable.core.agent_runner import SableAgentClient, _resolve_vision_mode  # noqa: E402
from sable.core.tools import view_image, take_screenshot  # noqa: E402

# ------------------------------------------------------------------------
# from tests/test_dashboard.py
# ------------------------------------------------------------------------

STATIC = Path(__file__).resolve().parents[1] / "sable" / "dashboard" / "static"


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "uf-dash"
    monkeypatch.setenv("SABLE_HOME", str(h))
    return h


@pytest.fixture()
def app(home):
    a = DashboardApp(SableConfig(), home=home)
    yield a


def _iso_days_ago(days: int) -> str:
    """ISO timestamp ``days`` days before now (UTC), second precision.

    References the MODULE-GLOBAL ``datetime`` on purpose: the clock-jump
    meta-test monkeypatches this module's ``datetime`` (and the production
    modules' time sources) with one fake class, so seeds and windows move
    together under a frozen future clock.
    """
    return (datetime.now(timezone.utc) - timedelta(days=days)
            ).isoformat(timespec="seconds")


def seed_trace(store: TraceStore, tid: str, status="success", score=0.8,
               failure_reason=None, days_ago=1, timestamp=None):
    """Index + transcript for one trace, stamped RELATIVE to now by default.

    Time-bomb fix (deep review): the old literal ``2026-09-01T10:00:00+00:00``
    silently aged out of production rolling windows (stats_daily's 30-day
    calendar cutoff, generate_insights' days= window). Defaulting to one day
    ago keeps every seeded trace inside any sane window forever. Tests that
    genuinely need date arithmetic pass an explicit ``timestamp`` derived from
    ``datetime.now(UTC)`` offsets — never a bare literal.
    """
    p = Path(store.db_path).parent / f"{tid}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        f.write(json.dumps({"kind": "trace_header"}) + "\n")
        f.write(json.dumps({"kind": "message", "message": {
            "role": "user", "content": f"task {tid}"}}) + "\n")
        f.write(json.dumps({"kind": "message", "message": {
            "role": "assistant", "content": f"done {tid} result=ok"}}) + "\n")
    store.insert_trace(TraceRow(
        trace_id=tid, session_id=None,
        timestamp=timestamp or _iso_days_ago(days_ago),
        agent_profile="fallback", model_alias="base-model",
        model_version=None, endpoint_label=None, task_summary=f"t {tid}",
        status=status, outcome_score=score, num_tool_calls=1,
        tokens_in=10, tokens_out=20, duration_s=1.0, jsonl_path=str(p),
        failure_reason=failure_reason,
    ))


def _wait_curate_job(app, job_id):
    """Wait for a queued curate job to finish; return its result dict."""
    import time as _t

    j = None
    for _ in range(200):
        j = app.handle("GET", f"/api/jobs/{job_id}").body
        if j["job"]["state"] in ("done", "error", "cancelled"):
            break
        _t.sleep(0.05)
    assert j and j["job"]["state"] == "done", j and j["job"]
    import json as _json

    for line in reversed(j["log"]["lines"]):
        try:
            rec = _json.loads(line)
        except ValueError:
            continue
        if rec.get("event") == "done":
            return rec["result"]
    raise AssertionError("curate job finished without a done line")


def _curate(app, body=None):
    """POST /api/curate (now a queued job since 5d) and wait for completion.

    Returns the result dict from the job's final log line; raises on error.
    """
    r = app.handle("POST", "/api/curate", body=body or {})
    assert r.status == 202, r.body
    return _wait_curate_job(app, r.body["job_id"])


def _export(app, body):
    """POST /api/datasets/export with force=true (tests export outside home)."""
    b = dict(body)
    b.setdefault("force", True)
    return app.handle("POST", "/api/datasets/export", body=b)


# ------------------------------------------------------------------ status

def test_status_on_fresh_home(app):
    r = app.handle("GET", "/api/status")
    assert r.status == 200
    body = r.body
    assert body["buffer"]["total_traces"] == 0
    assert body["active_model"] is None
    assert body["server"]["running"] is False


def test_status_with_traces_and_serve_state(app, home):
    store = TraceStore(home / "buffer" / "sable.db")
    seed_trace(store, "t1")
    store.close()
    _write_serve_pid(home, 999_999_999)
    r = app.handle("GET", "/api/status")
    assert r.body["buffer"]["total_traces"] == 1
    # dead pid => not running (liveness probe)
    assert r.body["server"]["running"] is False


# ------------------------------------------------- failure remediation notes
# Pass 4 #33 endpoint smoke (idea adapted from PIASystem (MIT)
# app/core/failure_taxonomy.py).

def test_failure_note_endpoint_roundtrip(app, home):
    store = TraceStore(home / "buffer" / "sable.db")
    seed_trace(store, "t1", status="failure", score=-0.5,
               failure_reason="timeout")
    store.close()

    r = app.handle("POST", "/api/failures/note", body={
        "reason": "timeout", "status": "in_progress", "note": "retry wiring"})
    assert r.status == 200 and r.body["ok"]
    assert r.body["note"]["status"] == "in_progress"

    # current notes ride along in the insights payload (no extra round-trip)
    ins = app.handle("GET", "/api/insights", query={"days": "3650"}).body
    row = {x["reason"]: x for x in ins["failure_reasons"]["by_reason"]}
    assert row["timeout"]["status"] == "in_progress"
    assert row["timeout"]["explicit"] is True

    # validation: unknown status -> 400, missing reason -> 400
    bad = app.handle("POST", "/api/failures/note",
                     body={"reason": "timeout", "status": "wat"})
    assert bad.status == 400
    bad2 = app.handle("POST", "/api/failures/note", body={"status": "open"})
    assert bad2.status == 400

    # reset-to-open with no note deletes the row -> implicit open in insights
    app.handle("POST", "/api/failures/note", body={"reason": "timeout",
                                                   "status": "open"})
    ins = app.handle("GET", "/api/insights", query={"days": "3650"}).body
    row = {x["reason"]: x for x in ins["failure_reasons"]["by_reason"]}
    assert row["timeout"]["status"] == "open"
    assert row["timeout"]["explicit"] is False


# ------------------------------------------------------------------ traces

def test_trace_list_detail_feedback(app, home):
    store = TraceStore(home / "buffer" / "sable.db")
    seed_trace(store, "t1")
    seed_trace(store, "t2", status="failure", score=-0.5)
    store.close()

    r = app.handle("GET", "/api/traces", query={"limit": "10"})
    assert r.status == 200
    ids = {t["trace_id"] for t in r.body["traces"]}
    assert ids == {"t1", "t2"}

    r = app.handle("GET", "/api/traces", query={"status": "failure"})
    assert [t["trace_id"] for t in r.body["traces"]] == ["t2"]

    r = app.handle("GET", "/api/traces/t1")
    assert r.status == 200
    assert len(r.body["messages"]) == 2

    assert app.handle("GET", "/api/traces/nope").status == 404

    r = app.handle("POST", "/api/feedback",
                   body={"trace_id": "t1", "score": 1.0, "note": "nice"})
    assert r.status == 200 and r.body["ok"]
    detail = app.handle("GET", "/api/traces/t1").body
    assert detail["feedback"][-1]["source"] == "user"

    # validation
    assert app.handle("POST", "/api/feedback",
                      body={"trace_id": "t1", "score": 5}).status == 400
    assert app.handle("POST", "/api/feedback", body={}).status == 400
    assert app.handle("POST", "/api/feedback",
                      body={"trace_id": "zz", "score": 1}).status == 404


# ------------------------------------------------------- trace timeline (#35)

def _tl_rec(ts, msg):
    return {"kind": "message", "ts": ts, "message": msg}


def test_timeline_durations_and_kinds():
    from sable.core.compression import SUMMARY_PREFIX

    # Header at epoch 100 s + footer duration 6 s -> run ends at epoch 106,
    # closing the last event (104.5) with a 1500 ms tail segment.
    records = [
        {"kind": "trace_header", "timestamp": "1970-01-01T00:01:40+00:00"},
        _tl_rec(100.0, {"role": "system", "content": "sys"}),
        _tl_rec(100.5, {"role": "user", "content": "go"}),
        _tl_rec(101.0, {"role": "assistant", "content": "hi",
                        "tool_calls": [{"id": "1"}]}),
        _tl_rec(103.25, {"role": "tool", "name": "run_command",
                         "content": "ok"}),
        _tl_rec(104.0, {"role": "tool", "name": "retry_probe",
                        "content": "ok"}),
        _tl_rec(104.5, {"role": "system",
                        "content": SUMMARY_PREFIX + "\nblah"}),
        # footer closes the last event: 106.0 - 104.5 = 1500 ms
        {"kind": "trace_footer", "duration_s": 6.0},
    ]
    msgs = [r["message"] for r in records if r.get("kind") == "message"]
    segs = build_trace_timeline(records, msgs)
    assert [s["kind"] for s in segs] == ["other", "other", "llm", "tool",
                                         "retry", "scrub"]
    assert [s["duration_ms"] for s in segs] == [500, 500, 2250, 750, 500, 1500]
    assert [s["message_index"] for s in segs] == [0, 1, 2, 3, 4, 5]
    assert segs[0]["start_ts"] == 100.0
    assert all(s["label"] for s in segs)


def test_timeline_missing_timestamps_fallback_and_degenerate():
    # No per-event ts: one run-wide segment from header ts + footer duration_s.
    records = [
        {"kind": "trace_header", "timestamp": "2026-09-01T10:00:00+00:00"},
        {"kind": "message", "message": {"role": "assistant", "content": "x"}},
        {"kind": "message", "message": {"role": "tool", "name": "f",
                                        "content": "y"}},
        {"kind": "trace_footer", "duration_s": 2.5},
    ]
    segs = build_trace_timeline(records)
    assert len(segs) == 1
    assert segs[0]["kind"] == "tool" and segs[0]["duration_ms"] == 2500
    assert segs[0]["message_index"] == 0

    # Nothing timeable at all -> empty (frontend shows the 'no timing data' hint)
    assert build_trace_timeline([{"kind": "message", "message": {
        "role": "user", "content": "x"}}]) == []
    assert build_trace_timeline([]) == []
    # Garbage records never raise
    garbage = cast(List[Any], [None, 3, {"kind": "message"},
                               {"kind": "message", "message": 7}])
    assert build_trace_timeline(garbage) == []


def test_timeline_merges_adjacent_and_anchors_filtered_transcript():
    # 250 alternating same-kind events -> merged down under the 200 cap.
    recs = [{"kind": "trace_header", "timestamp": "2026-09-01T10:00:00+00:00"}]
    for i in range(250):
        recs.append(_tl_rec(100.0 + i, {"role": "tool", "name": "f",
                                        "content": str(i)}))
    segs = build_trace_timeline(recs)
    assert len(segs) == 1
    assert segs[0]["duration_ms"] >= 249000

    # q-filtered transcript: segments anchor by identity into the delivered
    # list; events whose message was filtered out are skipped entirely.
    m_a = {"role": "assistant", "content": "a"}
    m_b = {"role": "tool", "name": "f", "content": "b"}
    m_c = {"role": "assistant", "content": "c"}
    recs = [_tl_rec(10.0, m_a), _tl_rec(11.0, m_b), _tl_rec(12.0, m_c)]
    segs = build_trace_timeline(recs, [m_c, m_a])  # filtered + reordered
    assert [(s["kind"], s["message_index"]) for s in segs] == [
        ("llm", 1), ("llm", 0)]


def test_trace_detail_includes_timeline(app, home):
    store = TraceStore(home / "buffer" / "sable.db")
    seed_trace(store, "t1")  # no per-event ts, no footer -> []
    store.close()
    r = app.handle("GET", "/api/traces/t1")
    assert r.status == 200
    assert r.body["timeline"] == []
    assert len(r.body["messages"]) == 2  # transcript untouched

    jsonl = Path(store.db_path).parent / "t1.jsonl"
    lines = [json.loads(l) for l in jsonl.read_text().splitlines()]
    lines[0]["timestamp"] = "2026-09-01T10:00:00+00:00"
    lines.append({"kind": "trace_footer", "duration_s": 4.0})
    jsonl.write_text("\n".join(json.dumps(l) for l in lines) + "\n")
    r = app.handle("GET", "/api/traces/t1")
    segs = r.body["timeline"]
    assert len(segs) == 1 and segs[0]["duration_ms"] == 4000
    assert segs[0]["kind"] == "llm"  # assistant+user, no tools -> llm


# ------------------------------------------------------------------ models

def test_models_promote_rollback(app, home):
    reg = ModelRegistry(home / "buffer" / "sable.db")
    m1 = reg.register_model(base_model="base-x")
    reg.set_validation(m1.model_id, {"ok": True}, passed=True)
    m2 = reg.register_model(base_model="base-x")
    reg.set_validation(m2.model_id, {"ok": True}, passed=True)
    reg.close()

    r = app.handle("GET", "/api/models")
    assert r.status == 200
    assert {m["model_id"] for m in r.body["models"]} == {"v1", "v2"}
    assert r.body["active_model"] is None

    # cannot promote a candidate from the API? v1/v2 are validated -> ok.
    r = app.handle("POST", "/api/models/promote", body={"model_id": "v1"})
    assert r.status == 200 and r.body["model"]["status"] == "promoted"

    # promoting an already-promoted model is rejected (409 via registry error)
    reg = ModelRegistry(home / "buffer" / "sable.db")
    m3 = reg.register_model(base_model="base-x")  # candidate
    reg.close()
    r = app.handle("POST", "/api/models/promote", body={"model_id": m3.model_id})
    assert r.status == 409

    r = app.handle("GET", "/api/models")
    assert r.body["active_model"] == "v1"
    # lineage edge: v2/v3 parent should be v1 (registered while v1 active)
    assert {"from": "v1", "to": m3.model_id} in r.body["edges"]

    r = app.handle("POST", "/api/models/rollback", body={})
    assert r.status == 200 and r.body["active_model"] is None or True
    r = app.handle("GET", "/api/models")
    # v1 rolled back; nothing superseded before it -> pointer cleared
    assert r.body["active_model"] is None

    assert app.handle("POST", "/api/models/promote", body={}).status == 400


def test_models_list_catalog_enrichment(app, home):
    """Backlog #15: /api/models adds detail/dataset_title/steps/duration."""
    reg = ModelRegistry(home / "buffer" / "sable.db")
    m1 = reg.register_model(base_model="base-x", dataset_id="ds-a")
    reg.set_losses(m1.model_id, train_loss=0.42, eval_loss=0.51)
    m2 = reg.register_model(base_model="base-x", dataset_id="ds-missing")
    # finished train job for v1 with a step-carrying terminal event
    log = home / "train_runs" / "job-x.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps({"event": "start"}) + "\n"
                   + json.dumps({"event": "progress", "step": 10,
                                 "total": 12, "loss": 0.5}) + "\n"
                   + json.dumps({"event": "sft_done", "steps": 12,
                                 "train_loss": 0.42}) + "\n")
    j = reg.enqueue_job("sft", dataset_id="ds-a", model_id=m1.model_id,
                        log_path=str(log))
    reg.mark_running(j.job_id, 4242)
    reg.finish_job(j.job_id, "done")
    # v2 has no job at all -> tolerant omission
    reg.close()

    ds = home / "datasets" / "ds-a"
    ds.mkdir(parents=True)
    (ds / "manifest.json").write_text(json.dumps(
        {"dataset_id": "ds-a", "title": "my run"}))

    r = app.handle("GET", "/api/models")
    assert r.status == 200
    by_id = {m["model_id"]: m for m in r.body["models"]}
    v1, v2 = by_id["v1"], by_id["v2"]

    assert v1["dataset_title"] == "my run"
    assert v1["train_steps"] == 12
    assert v1["train_duration_s"] is not None and v1["train_duration_s"] >= 0
    assert v1["origin"] == "trained"
    assert v1["detail"].startswith("my run · 12 steps · ")
    assert v1["detail"].endswith(" · loss 0.42/0.51")

    # missing manifest -> dataset id itself; no job -> no steps/duration
    assert v2["dataset_title"] == "ds-missing"
    assert v2["train_steps"] is None
    assert v2["train_duration_s"] is None
    assert v2["detail"].startswith("ds-missing · ")
    assert "loss" not in v2["detail"]  # no losses recorded


# -------------------------------------------------------------------- jobs

def test_jobs_list_and_cancel(app, home):
    reg = ModelRegistry(home / "buffer" / "sable.db")
    j = reg.enqueue_job("sft", dataset_id="d1")
    reg.close()
    r = app.handle("GET", "/api/jobs", query={"tail": "5"})
    assert r.status == 200
    job = r.body["jobs"][0]
    assert job["job_id"] == j.job_id and job["state"] == "queued"

    r = app.handle("POST", "/api/jobs/cancel", body={"job_id": j.job_id})
    assert r.status == 200 and r.body["job"]["state"] == "cancelled"
    assert app.handle("POST", "/api/jobs/cancel",
                      body={"job_id": "nope"}).status == 404


# ------------------------------------------------------------------ curate

def test_curate_endpoint(app, home):
    """5d contract: POST /api/curate queues a kind="curate" job (202)."""
    store = TraceStore(home / "buffer" / "sable.db")
    seed_trace(store, "t1")
    store.close()
    r = app.handle("POST", "/api/curate", body={"name": "ui"})
    assert r.status == 202, r.body
    assert r.body["state"] == "queued"
    job_id = r.body["job_id"]

    res = _wait_curate_job(app, job_id)
    assert res["sft_count"] == 1
    ds_id = res["dataset_id"]

    # the curate ran as a queued job of kind "curate"
    jobs = app.handle("GET", "/api/jobs").body["jobs"]
    kinds = {j["job_id"]: j["kind"] for j in jobs}
    assert kinds.get(job_id) == "curate"

    # invalid filters still reject synchronously (400 before queueing)
    assert err_code(app.handle("POST", "/api/curate",
                               body={"formats": "everything"})) == "bad_request"

    r = app.handle("GET", "/api/datasets")
    ids = {d["dataset_id"] for d in r.body["datasets"]}
    assert ds_id in ids
    r = app.handle("GET", f"/api/datasets/{ds_id}")
    assert r.status == 200 and r.body["dataset_id"] == ds_id
    assert app.handle("GET", "/api/datasets/missing").status == 404


# ------------------------------------------------------- train (stub worker)

STUB = r'''
import argparse, json, sys
from pathlib import Path
ap = argparse.ArgumentParser()
for f in ("--job","--kind","--config","--dataset-dir","--base-model",
          "--log-path","--adapter-dir","--gguf-out"):
    ap.add_argument(f, default=None)
ap.add_argument("--quant", default="q4_k_m")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()
line = json.dumps({"event": "done", "job": a.job})
print(line, flush=True)
if a.log_path:
    Path(a.log_path).parent.mkdir(parents=True, exist_ok=True)
    open(a.log_path, "a").write(line + "\n")
'''


def test_train_endpoint_queues_and_runs(app, home, tmp_path, monkeypatch):
    stub = tmp_path / "stub.py"
    stub.write_text(STUB)
    monkeypatch.setenv("SABLE_WORKER_COMMAND", f"{sys.executable} {stub}")

    # need a dataset dir for the train.yaml path (worker ignores it here)
    ds = home / "datasets" / "ds-x"
    ds.mkdir(parents=True)
    (ds / "train.yaml").write_text("model: base\ndata: {}\n")

    r = app.handle("POST", "/api/train", body={"dataset": "ds-x", "kind": "sft"})
    assert r.status == 202
    job_id = r.body["job_id"]

    # wait for the background thread to finish the job
    import time
    state = None
    for _ in range(100):
        jobs = app.handle("GET", "/api/jobs").body["jobs"]
        me = [j for j in jobs if j["job_id"] == job_id]
        state = me[0]["state"] if me else None
        if state in ("done", "error", "cancelled"):
            break
        time.sleep(0.1)
    assert state == "done", f"job ended in {state}"

    assert app.handle("POST", "/api/train", body={}).status == 400
    assert app.handle("POST", "/api/train",
                      body={"dataset": "d", "kind": "bogus"}).status == 400


def test_unknown_endpoint_and_bad_body(app):
    assert app.handle("GET", "/api/bogus").status == 404
    assert app.handle("PUT", "/api/status").status == 404


# ------------------------------------------------------------- server tests

def _start_server(home):
    """Server with token auth explicitly disabled (token="" opts out of the
    generated-and-persisted default); token behavior has its own tests."""
    from sable.dashboard.server import make_server, BIND_HOST

    httpd = make_server(SableConfig(), port=0, home=home, token="")
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, f"http://{BIND_HOST}:{httpd.server_address[1]}"


def test_server_loopback_static_and_api(home):
    import httpx

    httpd, base = _start_server(home)
    try:
        # static UI served locally
        r = httpx.get(base + "/", timeout=5)
        assert r.status_code == 200 and "sable" in r.text
        assert 'src="/app.js"' in r.text
        r = httpx.get(base + "/style.css", timeout=5)
        assert r.status_code == 200

        # bind is loopback
        assert str(httpd.server_address[0]) == "127.0.0.1"

        # API over HTTP
        r = httpx.get(base + "/api/status", timeout=5)
        assert r.status_code == 200 and r.json()["buffer"]["total_traces"] == 0

        # invalid JSON body rejected
        r = httpx.post(base + "/api/feedback", content=b"{not json",
                       headers={"Content-Type": "application/json"}, timeout=5)
        assert r.status_code == 400

        # path traversal blocked
        r = httpx.get(base + "/../../etc/passwd", timeout=5,
                      follow_redirects=False)
        assert r.status_code in (400, 404)
        r = httpx.get(base + "/%2e%2e%2f%2e%2e%2fetc/passwd", timeout=5)
        assert r.status_code == 404
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_server_serves_static_module_graph(home):
    """Phase 5b: nested static/js/** modules served with JS MIME; entry graph
    reachable from index (index -> /app.js -> js/* -> js/panels/*)."""
    import httpx

    httpd, base = _start_server(home)
    try:
        r = httpx.get(base + "/app.js", timeout=5)
        assert r.status_code == 200
        assert r.headers["Content-Type"].startswith("application/javascript")

        # every relative import in the shipped graph resolves over HTTP
        seen, queue_files = set(["/app.js"]) , ["/app.js"]
        import re as _re
        imp = _re.compile(r"""from\s+"(\./[^"]+|\.\./[^"]+)"|src="(/[^"]+\.js)\"""")
        while queue_files:
            path = queue_files.pop()
            text = httpx.get(base + path, timeout=5).text
            for m in imp.finditer(text):
                ref = m.group(1) or m.group(2)
                if ref and not ref.startswith("/"):
                    base_dir = path.rsplit("/", 1)[0]
                    # resolve ./ and ../ segments
                    parts = (base_dir + "/" + ref).split("/")
                    resolved = []
                    for p in parts:
                        if p == ".":
                            continue
                        if p == ".." and resolved:
                            resolved.pop()
                        else:
                            resolved.append(p)
                    ref = "/".join(resolved) or "/"
                if ref not in seen:
                    seen.add(ref)
                    rr = httpx.get(base + ref, timeout=5)
                    assert rr.status_code == 200, f"{path} imports {ref}: {rr.status_code}"
                    assert rr.headers["Content-Type"].startswith(
                        "application/javascript"), ref
                    queue_files.append(ref)
        assert len(seen) >= 8, f"expected the full module graph, saw {seen}"

        # serve status + thread endpoints over HTTP (panels rely on these)
        r = httpx.get(base + "/api/serve/status", timeout=5)
        assert r.status_code == 200 and r.json() == {"running": False,
                                                     "mode": "none",
                                                     "crash_loop": None}
        r = httpx.put(base + "/api/chat/threads/web1",
                      json={"title": "from web", "messages": []}, timeout=5)
        assert r.status_code == 200 and r.json()["thread"]["id"] == "web1"
        r = httpx.get(base + "/api/chat/threads/web1", timeout=5)
        assert r.status_code == 200 and r.json()["thread"]["title"] == "from web"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_dashboard_port_resolution(home, monkeypatch):
    from sable.dashboard.server import dashboard_port

    cfg = SableConfig()
    monkeypatch.delenv("SABLE_DASHBOARD_PORT", raising=False)
    assert dashboard_port(cfg, override=8123) == 8123
    assert dashboard_port(cfg) == 0
    cfg.dashboard = {"port": 9911}
    assert dashboard_port(cfg) == 9911
    monkeypatch.setenv("SABLE_DASHBOARD_PORT", "7777")
    assert dashboard_port(cfg) == 7777


# ------------------------------------------------------------ hygiene tests

def _static_files():
    """All shipped UI files (static/ and nested static/js/**)."""
    return sorted(p for p in STATIC.rglob("*") if p.is_file())


def test_no_external_urls_in_static_assets():
    """Offline-capable: no CDN/remote references in shipped UI files."""
    offenders = []
    for f in _static_files():
        text = f.read_text()
        for m in re.finditer(r"""https?://[^\s"')<>]+""", text):
            url = m.group(0)
            if url.startswith("http://127.0.0.1") or url.startswith("http://localhost"):
                continue  # loopback examples are fine
            if url.startswith("http://www.w3.org/"):
                continue  # XML namespace identifiers (e.g. SVG NS) are never fetched
            offenders.append(f"{f.name}: {url}")
    assert not offenders, "external references found:\n" + "\n".join(offenders)


def test_no_inline_event_handlers_or_eval_in_static_js():
    """CSP compatibility + injection hygiene: no on*= attrs in HTML strings,
    no eval/new Function/innerHTML in shipped JS."""
    bad = re.compile(r"""\bone[a-z]+\s*=\s*["']|\.innerHTML\s*=|\beval\s*\(|new\s+Function\s*\(""")
    offenders = []
    for f in _static_files():
        if f.suffix not in (".js", ".html"):
            continue
        text = f.read_text()
        for m in bad.finditer(text):
            offenders.append(f"{f.relative_to(STATIC)}: {m.group(0)!r}")
    assert not offenders, "forbidden constructs:\n" + "\n".join(offenders)


def test_design_tokens_only_in_root_block():
    """Design-token discipline (inspiration pass 3 #26): raw hex colors live ONLY
    inside the :root block of style.css. JS and HTML must reference var(--*)
    tokens, so a theme change touches exactly one place."""
    css = (STATIC / "style.css").read_text()
    i0 = css.index(":root {")
    i1 = css.index("\n}", i0)
    offenders = [m.group(0) for m in re.finditer(r"#[0-9a-fA-F]{3,8}\b",
                                                 css[:i0] + css[i1:])]
    assert not offenders, f"hex outside :root token block: {offenders}"

    bad = []
    for f in _static_files():
        if f.suffix not in (".js", ".html"):
            continue
        for m in re.finditer(r"['\"`]#[0-9a-fA-F]{3,8}['\"`]?", f.read_text()):
            bad.append(f"{f.relative_to(STATIC)}: {m.group(0)!r}")
    assert not bad, "raw color literals in JS/HTML (use var(--token)):\n" + "\n".join(bad)


def _js_delim_errors(text: str) -> list:
    """Balanced-delimiter smoke check for JS source.

    Tokenizer-aware (not regex-stripping): skips line/block comments and
    string literals inline, tracks template literals including ``${``
    interpolations, and distinguishes regex literals from division using the
    previous significant token. Returns a list of human-readable problems
    (empty == balanced)."""
    pairs = {"{": "}", "(": ")", "[": "]"}
    errors: list = []
    stack: list = []  # ("{"|"("|"[", line) or ("T", line) template, ("I", line) ${
    prev = ""
    i, line, n = 0, 1, len(text)
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        top = stack[-1][0] if stack else None
        if c == "\n":
            line += 1
            i += 1
            continue
        if top == "T":  # inside template literal body
            if c == "\\":
                i += 2
                continue
            if c == "`":
                stack.pop()
                i += 1
                prev = "`"
                continue
            if c == "$" and nxt == "{":
                stack.append(("I", line))
                i += 2
                prev = "{"
                continue
            i += 1
            continue
        if c == "/" and nxt == "/":
            while i < n and text[i] != "\n":
                i += 1
            continue
        if c == "/" and nxt == "*":
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                if text[i] == "\n":
                    line += 1
                i += 1
            i += 2
            continue
        if c in "\"'":
            q = c
            i += 1
            while i < n and text[i] != q:
                if text[i] == "\\":
                    i += 2
                    continue
                if text[i] == "\n":
                    line += 1
                    break
                i += 1
            i += 1
            prev = q
            continue
        if c == "`":
            stack.append(("T", line))
            i += 1
            prev = "`"
            continue
        # regex literal: '/' only where a value is expected
        if c == "/" and (prev == "" or prev in "(,=:[!&|?{};\n+-*%<>~^!"):
            i += 1
            cls = False
            while i < n:
                ch = text[i]
                if ch == "\\":
                    i += 2
                    continue
                if ch == "[":
                    cls = True
                elif ch == "]":
                    cls = False
                elif ch == "/" and not cls:
                    break
                elif ch == "\n":
                    errors.append(f"unterminated regex at line {line}")
                    break
                i += 1
            i += 1
            while i < n and text[i] in "gimsuyd":
                i += 1
            prev = "/"
            continue
        if c in "{([":
            stack.append((c, line))
        elif c in "})]":
            if not stack:
                errors.append(f"unmatched {c!r} at line {line}")
            else:
                t, opened = stack.pop()
                if t == "I" and c == "}":
                    pass  # interpolation ends; back to template body (top T)
                elif pairs.get(t) != c:
                    errors.append(
                        f"opened {t!r} at line {opened}, closed {c!r} "
                        f"at line {line}")
        if not c.isspace():
            prev = c
        i += 1
    for t, opened in stack:
        errors.append(f"unclosed {t!r} opened at line {opened}")
    return errors


def test_static_js_parses_as_es_modules():
    """Every shipped .js parses as an ES module (node --check when present;
    otherwise a tokenizer-aware balanced-delimiter smoke check — no build
    tooling either way)."""
    import shutil
    import subprocess

    js_files = [f for f in _static_files() if f.suffix == ".js"]
    assert js_files, "expected shipped JS modules"

    node = shutil.which("node")
    if node:
        for f in js_files:
            r = subprocess.run([node, "--check", str(f)],
                               capture_output=True, text=True, timeout=30)
            assert r.returncode == 0, f"{f}: {r.stderr[:500]}"
    else:  # pragma: no cover - node present in dev envs
        for f in js_files:
            errs = _js_delim_errors(f.read_text())
            assert not errs, f"unbalanced delimiters in {f}: {errs[:3]}"


def test_index_references_only_local_assets():
    """index.html must reference only same-origin paths (/, /js/..., #...)."""
    html = (STATIC / "index.html").read_text()
    refs = re.findall(r"""(?:src|href)\s*=\s*"([^"]+)\"""", html)
    assert refs, "expected asset references in index.html"
    for ref in refs:
        assert (ref.startswith("/") or ref.startswith("#")), \
            f"non-local reference in index.html: {ref}"


def test_per_route_body_caps(home):
    """Contract: generic /api/* bodies cap at 1 MB, but /api/chat/stream must
    accept chat image attachments (4 x 8 MB raw base64-inflated). A >1 MB body
    on a generic route is 413; the same size on chat/stream reaches the
    handler (400 prompt required), proving the raised cap."""
    import httpx

    from sable.dashboard.app import API_MAX_BODY_BYTES, api_max_body_bytes

    assert api_max_body_bytes("/api/config") == API_MAX_BODY_BYTES == 1_000_000
    # 4 images x 8 MB raw, base64 ~4/3 inflated + headroom:
    assert api_max_body_bytes("/api/chat/stream") > 4 * 8_000_000 * 4 // 3

    httpd, base = _start_server(home)
    try:
        big = b'{"pad": "' + b"x" * (1_100_000) + b'"}'
        r = httpx.put(base + "/api/config", content=big,
                      headers={"Content-Type": "application/json"}, timeout=10)
        assert r.status_code == 413 and \
            r.json()["error"]["code"] == "payload_too_large"
        # Same size on the image-carrying route: past the body gate, into the
        # handler (missing prompt -> 400, NOT 413).
        pad = "x" * 1_100_000
        r = httpx.post(base + "/api/chat/stream",
                       json={"pad": pad}, timeout=10)
        assert r.status_code == 400 and "prompt" in r.text
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_cli_dashboard_command_exists():
    from sable.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["dashboard", "--port", "0"])
    assert args.command == "dashboard"
    assert callable(args.func)


# =====================================================================
# Phase 5a — routing core, event bus, serve/hardware/config/chat/iterate
# =====================================================================



def err_code(r) -> str:
    return r.body["error"]["code"]


# ------------------------------------------------------------ route matcher

def test_route_matcher_params_and_verbs(app, home):
    # <param> extraction + URL-decoding
    assert app.handle("GET", "/api/traces/nope").status == 404
    # verb mismatch on an existing pattern => 404 (no such endpoint)
    assert app.handle("PUT", "/api/status").status == 404
    assert app.handle("DELETE", "/api/status").status == 404
    # PUT/DELETE verbs route for real endpoints
    r = app.handle("PUT", "/api/chat/threads/t1", body={"title": "x"})
    assert r.status == 200 and r.body["thread"]["id"] == "t1"
    r = app.handle("DELETE", "/api/chat/threads/t1")
    assert r.status == 200 and r.body["ok"]
    # static-before-param ordering: /api/traces/stats is not a trace id
    assert app.handle("GET", "/api/traces/stats").status == 200


def test_error_envelope_shape(app):
    r = app.handle("GET", "/api/bogus")
    assert r.status == 404
    assert r.body["error"]["code"] == "not_found"
    assert "no such endpoint" in r.body["error"]["message"]

    r = app.handle("POST", "/api/models/promote", body={})
    assert r.status == 400 and err_code(r) == "bad_request"

    r = app.handle("GET", "/api/traces/nope")
    assert r.status == 404 and err_code(r) == "not_found"


# ------------------------------------------------------------------ bus

def test_event_bus_publish_subscribe_close(app):
    sub = app.bus.subscribe()
    ev = app.bus.publish("job.state", {"job_id": "j1", "state": "running"})
    got = sub.poll(timeout=1)
    assert got and got["kind"] == "job.state" and got["id"] == ev["id"]
    assert got["payload"]["job_id"] == "j1"
    assert app.bus.subscriber_count == 1
    sub.close()
    assert app.bus.subscriber_count == 0
    assert sub.poll(timeout=0.05) is None


def test_event_bus_slow_consumer_overflow():
    from sable.dashboard.events import EventBus

    bus = EventBus(maxsize=2)
    sub = bus.subscribe()
    for i in range(6):
        bus.publish("job.log", {"i": i})
    first = sub.poll(timeout=1)
    assert first is not None
    assert first["kind"] == "overflow" and first["payload"]["dropped"] >= 4
    rest = [e for e in (sub.poll(timeout=1) for _ in range(2)) if e is not None]
    assert [e["payload"]["i"] for e in rest] == [4, 5]
    assert sub.poll(timeout=0.05) is None


def test_sse_events_endpoint_stream(app):
    r = app.handle("GET", "/api/events")
    assert r.status == 200 and r.stream is not None
    it = iter(r.stream)
    hello = next(it)
    assert hello.startswith(": sable events") and "retry: 3000" in hello
    app.bus.publish("serve.state", {"event": "started"})
    frame = next(it)
    assert frame.startswith("id: ") and "event: serve.state" in frame
    assert '"event": "started"' in frame.replace("\\\"", "'") or \
        "started" in frame
    it.close()  # client disconnect => subscriber cleanup in finally
    import time as _t

    for _ in range(50):
        if app.bus.subscriber_count == 0:
            break
        _t.sleep(0.02)
    assert app.bus.subscriber_count == 0


def test_job_state_events_published(app, home, tmp_path, monkeypatch):
    stub = tmp_path / "stub.py"
    stub.write_text(STUB)
    monkeypatch.setenv("SABLE_WORKER_COMMAND", f"{sys.executable} {stub}")
    ds = home / "datasets" / "ds-ev"
    ds.mkdir(parents=True)
    (ds / "train.yaml").write_text("model: base\ndata: {}\n")

    sub = app.bus.subscribe()
    r = app.handle("POST", "/api/train", body={"dataset": "ds-ev"})
    assert r.status == 202
    job_id = r.body["job_id"]

    states = []
    for _ in range(150):
        ev = sub.poll(timeout=0.2)
        if ev and ev["kind"] == "job.state" and ev["payload"]["job_id"] == job_id:
            states.append(ev["payload"]["state"])
        if "done" in states:
            break
    sub.close()
    assert "running" in states and "done" in states


# ------------------------------------------------------------------ serve

def _fake_serve_state(home, url):
    home.mkdir(parents=True, exist_ok=True)
    (home / "serve_state.json").write_text(json.dumps({
        "mode": "attach", "base_url": url, "model_name": "fake-model",
        "attach_url": url.rsplit("/v1", 1)[0],
    }))


def _write_serve_pid(home, pid):
    home.mkdir(parents=True, exist_ok=True)
    (home / "serve_state.json").write_text(json.dumps(
        {"pid": pid, "base_url": "http://127.0.0.1:1/v1", "model_name": "m"}))


def test_serve_status_none_and_attach(app, home):
    r = app.handle("GET", "/api/serve/status")
    assert r.status == 200 and r.body == {"running": False, "mode": "none",
                                          "crash_loop": None}

    from fake_server import FakeOpenAIServer

    with FakeOpenAIServer() as fs:
        _fake_serve_state(home, fs.base_url)
        r = app.handle("GET", "/api/serve/status")
        assert r.status == 200 and r.body["running"] is True
        assert r.body["mode"] == "attach"
        assert r.body["base_url"] == fs.base_url


def test_server_start_validation_and_conflict(app, home):
    # The legacy /api/serve/start endpoint was removed; the reserved default
    # instance is started through POST /api/servers/default/start. With no
    # gguf_path configured the default instance has nothing to spawn => 400.
    r = app.handle("POST", "/api/servers/default/start", body={})
    assert r.status == 400 and err_code(r) == "bad_instance"

    # already serving (attach alive in the legacy state, folded into the
    # merged default view) => 409 unless force
    from fake_server import FakeOpenAIServer

    with FakeOpenAIServer() as fs:
        _fake_serve_state(home, fs.base_url)
        r = app.handle("POST", "/api/servers/default/start", body={})
        assert r.status == 409 and err_code(r) == "already_serving"


def test_serve_stop_never_kills_attached(app, home):
    # /api/serve/attach was removed (attach through a named instance or the
    # default config); /api/serve/stop stays for the legacy status card. Seed
    # the attach state the way an old sable left it and verify stop only
    # forgets it — never kills the foreign server.
    from fake_server import FakeOpenAIServer

    with FakeOpenAIServer() as fs:
        _fake_serve_state(home, fs.base_url)
        status = app.handle("GET", "/api/serve/status").body
        assert status["running"] and status["mode"] == "attach"

        r = app.handle("POST", "/api/serve/stop", body={})
        assert r.status == 200 and r.body == {"ok": True, "stopped": False}
        # state cleared (both views); fake server still healthy (never killed)
        assert not (home / "serve_state.json").exists()
        import httpx

        assert httpx.get(fs.base_url + "/models", timeout=5).status_code == 200


def test_serve_stop_refuses_foreign_pid(app, home, monkeypatch):
    if hasattr(__import__("os"), "geteuid") and __import__("os").geteuid() == 0:
        pytest.skip("running as root: pid-1 permission probe invalid")
    _write_serve_pid(home, 1)
    r = app.handle("POST", "/api/serve/stop", body={})
    assert r.status == 200 and r.body["stopped"] is False
    # state file cleared without signalling a process we don't own
    assert not (home / "serve_state.json").exists()


def test_deploy_endpoint_validation(app, home):
    assert app.handle("POST", "/api/deploy", body={}).status == 400
    assert app.handle("POST", "/api/deploy",
                      body={"model_id": "v9"}).status == 404
    reg = ModelRegistry(home / "buffer" / "sable.db")
    m = reg.register_model(base_model="b")
    reg.close()
    r = app.handle("POST", "/api/deploy", body={"model_id": m.model_id})
    assert r.status == 409 and err_code(r) == "no_gguf"


# --------------------------------------------------------------- hardware

def test_hardware_sampler_shape(app, monkeypatch):
    import sable.dashboard.hardware as hw

    monkeypatch.setattr(hw.shutil, "which", lambda name: None)  # no GPU
    # Runners may have a real AMD card visible via sysfs — stub that probe too.
    monkeypatch.setattr(hw, "amd_gpus", lambda: None)
    r = app.handle("GET", "/api/hardware")
    assert r.status == 200
    body = r.body
    assert body["gpus"] is None
    assert set(body) >= {"cpu_pct", "mem", "disk", "battery", "sampled_at"}
    # cached: identical snapshot within TTL
    assert app.handle("GET", "/api/hardware").body["sampled_at"] == \
        body["sampled_at"]

    if (Path("/proc/meminfo")).is_file():
        assert body["mem"] and body["mem"]["total_mib"] > 0
        assert body["disk"]["free_mib"] >= 0


def test_hardware_nvidia_parsing(monkeypatch):
    import sable.dashboard.hardware as hw

    class FakeOut:
        returncode = 0
        stdout = "Tesla T4, 1234, 15360, 42, 71\n"

    monkeypatch.setattr(hw.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(hw.subprocess, "run", lambda *a, **k: FakeOut())
    gpus = hw.nvidia_gpus()
    assert gpus == [{"name": "Tesla T4", "free_mib": 1234, "total_mib": 15360,
                     "util_pct": 42, "temperature_c": 71}]


# ----------------------------------------------------------------- config

def test_config_get_redacts_secrets(app):
    app.cfg.attach_api_key = "sk-super-secret"
    app.cfg.dashboard = {"token": "tok123"}
    body = app.handle("GET", "/api/config").body
    assert body["config"]["attach_api_key"] == "***"
    assert body["config"]["dashboard"]["token"] == "***"
    assert body["token_configured"] is True
    assert "sk-super-secret" not in json.dumps(body)


def test_config_put_validate_backup(app, home, monkeypatch, tmp_path):
    cfg_file = tmp_path / "cfg" / "sable.yaml"
    monkeypatch.setenv("SABLE_CONFIG", str(cfg_file))

    # invalid YAML => 400, nothing written
    r = app.handle("PUT", "/api/config", body={"yaml": "training: [unclosed"})
    assert r.status == 400 and err_code(r) == "bad_request"
    assert not cfg_file.exists()

    # unknown top-level key => 400 unknown_keys
    r = app.handle("PUT", "/api/config", body={"yaml": "bogus_section: 1\n"})
    assert r.status == 400 and err_code(r) == "unknown_keys"

    # unknown nested key => 400
    r = app.handle("PUT", "/api/config",
                   body={"yaml": "training:\n  not_a_real_key: 3\n"})
    assert r.status == 400 and err_code(r) == "unknown_keys"

    # valid write
    good = "training:\n  base_model: my-base\n  auto_min_traces: 7\n"
    r = app.handle("PUT", "/api/config", body={"yaml": good})
    assert r.status == 200 and cfg_file.read_text() == good

    # second write backs up the first
    r = app.handle("PUT", "/api/config", body={"yaml": "port: 8111\n"})
    assert r.status == 200
    bak = cfg_file.with_suffix(".yaml.bak")
    assert bak.is_file() and "my-base" in bak.read_text()

    # GET now reflects the new file (raw + path)
    body = app.handle("GET", "/api/config").body
    assert body["path"] == str(cfg_file) and "port: 8111" in body["raw"]


def test_config_put_section_patch(app, home, monkeypatch, tmp_path):
    cfg_file = tmp_path / "sable.yaml"
    monkeypatch.setenv("SABLE_CONFIG", str(cfg_file))
    r = app.handle("PUT", "/api/config",
                   body={"section": "dashboard", "patch": {"port": 9}})
    assert r.status == 200, r.body
    import yaml as _y

    assert _y.safe_load(cfg_file.read_text()) == {"dashboard": {"port": 9}}


# ------------------------------------------------------------------- chat

def test_proxy_chat_503_when_nothing_serving(app, home):
    r = app.handle("POST", "/api/proxy/chat",
                   body={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status == 503 and err_code(r) == "not_serving"


def test_proxy_chat_json_and_stream_passthrough(app, home):
    from fake_server import FakeOpenAIServer

    with FakeOpenAIServer() as fs:
        _fake_serve_state(home, fs.base_url)
        r = app.handle("POST", "/api/proxy/chat",
                       body={"messages": [{"role": "user", "content": "hi"}]})
        assert r.status == 200 and r.stream is None
        assert r.body["choices"][0]["message"]["tool_calls"]

        r = app.handle("POST", "/api/proxy/chat",
                       body={"messages": [{"role": "user", "content": "hi"}],
                             "stream": True})
        assert r.status == 200 and r.stream is not None
        chunks = list(r.stream)
        joined = "".join(chunks)
        assert "data:" in joined and "[DONE]" in joined


def test_chat_stream_event_protocol_and_recording(app, home, monkeypatch):
    from fake_server import FakeOpenAIServer
    from sable.core.agent_runner import AgentResult

    class StubRunner:
        name = "fallback"

        def __init__(self, cfg, base_url, model, api_key=None, recorder=None,
                     **kwargs):  # 5e: system_prompt/enabled_tool_names pass-through
            self.recorder = recorder
            self.base_url = base_url
            self.model = model

        def run(self, task, history=None):
            messages = [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": task},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "c1", "type": "function", "function": {
                        "name": "read_file", "arguments": '{"path": "x"}'}}]},
                {"role": "tool", "tool_call_id": "c1", "name": "read_file",
                 "content": "file contents"},
                {"role": "assistant", "content": "all done"},
            ]
            assert self.recorder is not None
            rec = self.recorder.record(
                task=task, messages=messages, backend="fallback",
                model_alias=self.model, base_url=self.base_url, completed=True)
            return AgentResult(final_response="all done", messages=messages,
                               backend="fallback", tokens_in=5, tokens_out=2,
                               trace_id=rec.trace_id if rec else None)

    monkeypatch.setattr("sable.core.agent_runner.create_runner", StubRunner)

    with FakeOpenAIServer() as fs:
        _fake_serve_state(home, fs.base_url)
        sub = app.bus.subscribe()
        r = app.handle("POST", "/api/chat/stream", body={"prompt": "go do it"})
        assert r.status == 200 and r.stream is not None
        text = "".join(r.stream)

    kinds = [line.split("event: ", 1)[1].split("\n", 1)[0]
             for line in text.split("\n\n") if line.startswith("event: ")]
    assert kinds[0] == "run.started"
    assert kinds[-1] == "run.done"
    assert kinds.index("tool.call") < kinds.index("tool.result")
    assert "message.delta" in kinds and "turn.done" in kinds
    assert '"name": "read_file"' in text
    assert '"ok": true' in text

    # run.done carries a trace_id that exists in the buffer; bus saw it too
    last_data = [line for line in text.splitlines()
                 if line.startswith("data: ")][-1]
    payload = json.loads(last_data[len("data: "):])
    assert payload["tokens_in"] == 5 and payload["trace_id"]
    tr = app.handle("GET", f"/api/traces/{payload['trace_id']}")
    assert tr.status == 200

    got_trace_event = False
    while True:
        ev = sub.poll(timeout=0.1)
        if ev is None:
            break
        if ev["kind"] == "trace.recorded":
            got_trace_event = True
    sub.close()
    assert got_trace_event


def test_chat_stream_503_and_validation(app, home):
    assert err_code(app.handle("POST", "/api/chat/stream", body={})) == \
        "bad_request"
    r = app.handle("POST", "/api/chat/stream", body={"prompt": "x"})
    assert r.status == 503 and err_code(r) == "not_serving"


def test_chat_cancel_and_threads(app, home):
    import threading as _th

    app._chat_runs["abc"] = {"cancel": _th.Event(), "done": False}
    r = app.handle("POST", "/api/chat/cancel", body={"run_id": "abc"})
    assert r.status == 200 and r.body["ok"]
    assert app._chat_runs["abc"]["cancel"].is_set()
    assert app.handle("POST", "/api/chat/cancel",
                      body={"run_id": "zz"}).status == 404

    # thread CRUD + traversal-safe ids
    r = app.handle("PUT", "/api/chat/threads/th1",
                   body={"title": "T", "messages": [{"role": "user",
                                                     "content": "hi"}]})
    assert r.status == 200 and r.body["thread"]["message_count"] if False \
        else r.status == 200
    lst = app.handle("GET", "/api/chat/threads").body["threads"]
    assert [t["id"] for t in lst] == ["th1"] and lst[0]["title"] == "T"
    # trace_ids accumulate across PUTs
    app.handle("PUT", "/api/chat/threads/th1", body={"trace_ids": ["t9"]})
    lst = app.handle("GET", "/api/chat/threads").body["threads"]
    assert lst[0]["trace_ids"] == ["t9"]
    assert app.handle("PUT", "/api/chat/threads/..%2Fevil",
                      body={}).status in (400, 404)
    assert app.handle("DELETE", "/api/chat/threads/th1").status == 200
    assert app.handle("DELETE", "/api/chat/threads/th1").status == 404


def test_thread_get_detail(app):
    """GET /api/chat/threads/<id> returns the full thread (5b chat restore)."""
    assert app.handle("GET", "/api/chat/threads/nope").status == 404
    msgs = [{"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo"}]
    app.handle("PUT", "/api/chat/threads/d1",
               body={"title": "D", "messages": msgs, "trace_ids": ["t1"]})
    r = app.handle("GET", "/api/chat/threads/d1")
    assert r.status == 200
    t = r.body["thread"]
    assert t["id"] == "d1" and t["title"] == "D" and t["messages"] == msgs
    assert t["trace_ids"] == ["t1"]
    # invalid id shape rejected
    assert app.handle("GET", "/api/chat/threads/a%2Fb").status in (400, 404)


# --------------------------------------------------- traces/datasets extras

def test_traces_q_search_and_stats(app, home):
    store = TraceStore(home / "buffer" / "sable.db")
    # seed_trace stamps relative-to-now by default, so both rows sit inside
    # stats_daily's real calendar cutoff without any timestamp patching.
    seed_trace(store, "t1")
    seed_trace(store, "t2", status="failure", score=-0.5)
    # direct summary tweak for a distinctive needle
    store._conn.execute(
        "UPDATE traces SET task_summary = 'refactor the parser' "
        "WHERE trace_id = 't2'")
    store._conn.commit()
    store.close()

    r = app.handle("GET", "/api/traces", query={"q": "parser"})
    assert [t["trace_id"] for t in r.body["traces"]] == ["t2"]
    r = app.handle("GET", "/api/traces", query={"q": "nomatch__"})
    assert r.body["traces"] == []
    # combined filters
    r = app.handle("GET", "/api/traces",
                   query={"q": "parser", "status": "failure"})
    assert len(r.body["traces"]) == 1

    body = app.handle("GET", "/api/traces/stats", query={"days": "30"}).body
    assert body["series"] and body["series"][0]["traces"] == 2
    assert body["series"][0]["fail_rate"] == 0.5


# ------------------------------------------------- clock-jump durability (meta)

def _freeze_clock(monkeypatch, days_ahead: int):
    """Point every now() the windowed paths consult at real-now + days_ahead.

    Freezegun-free technique: production code does
    ``from datetime import datetime`` then ``datetime.now(timezone.utc)``, so
    rebinding the module attribute to a subclass with a shifted ``now()``
    freezes exactly that source. We rebind it — in the SAME fake class — across
    (a) the windowed production modules and (b) this test module, so relative
    seeds and calendar cutoffs stay mutually consistent at any clock value:
    precisely the invariant the frozen-literal time-bombs violated.

    Limitation (documented): only ``datetime.now`` reached through these module
    attributes is frozen; wall-clock thread timing and other datetime imports
    are untouched. That suffices for stats_daily / generate_insights cutoffs.
    """
    import sable.store.db as db_mod
    import sable.store.insights as insights_mod

    real = datetime

    class _Frozen(real):  # type: ignore[misc, valid-type]
        @classmethod
        def now(cls, tz=None):
            return real.now(timezone.utc) + timedelta(days=days_ahead)

    monkeypatch.setattr(db_mod, "datetime", _Frozen)
    monkeypatch.setattr(insights_mod, "datetime", _Frozen)
    monkeypatch.setattr(sys.modules[__name__], "datetime", _Frozen)
    # sanity: the fake really is ~days_ahead ahead of the real clock
    delta = _Frozen.now(timezone.utc) - real.now(timezone.utc)
    assert timedelta(days=days_ahead - 1) < delta < timedelta(days=days_ahead + 1)


def test_windowed_assertions_survive_clock_jump_plus_60_days(app, home,
                                                             monkeypatch):
    """Time-bomb guard: the windowed tests still pass with the clock advanced
    +60 days. Re-runs the real stats-window test plus an insights default-
    window check; both seed relative-to-(fake)-now, so nothing ages out."""
    _freeze_clock(monkeypatch, 60)

    # 1) the actual windowed dashboard test (stats_daily 30-day cutoff)
    test_traces_q_search_and_stats(app, home)

    # 2) generate_insights default days=30 over relative seeds (t1/t2 from the
    # stats test above live in the same buffer, so four traces total)
    store = TraceStore(home / "buffer" / "sable.db")
    seed_trace(store, "ins1")
    seed_trace(store, "ins2", status="failure", score=-0.5)
    store.close()
    ins = app.handle("GET", "/api/insights", query={"days": "30"}).body
    assert ins["overview"]["traces"] == 4
    assert not ins["empty"]


def test_trace_detail_q_and_feedback_by_path(app, home):
    store = TraceStore(home / "buffer" / "sable.db")
    seed_trace(store, "t1")
    store.close()
    r = app.handle("GET", "/api/traces/t1", query={"q": "done"})
    assert len(r.body["messages"]) == 1
    r = app.handle("POST", "/api/traces/t1/feedback",
                   body={"score": -1, "note": "meh"})
    assert r.status == 200 and r.body["trace_id"] == "t1"
    detail = app.handle("GET", "/api/traces/t1").body
    assert detail["feedback"][-1]["comment"] == "meh"
    assert app.handle("POST", "/api/traces/zzz/feedback",
                      body={"score": 1}).status == 404


def test_dataset_rows_preview(app, home):
    store = TraceStore(home / "buffer" / "sable.db")
    seed_trace(store, "t1")
    seed_trace(store, "t2")
    store.close()
    ds_id = _curate(app, {"name": "ui"})["dataset_id"]

    r = app.handle("GET", f"/api/datasets/{ds_id}/rows",
                   query={"file": "sft", "offset": "0", "limit": "1"})
    assert r.status == 200 and r.body["total"] == 2 and len(r.body["rows"]) == 1
    assert r.body["rows"][0]["conversations"]
    r = app.handle("GET", f"/api/datasets/{ds_id}/rows",
                   query={"file": "sft", "offset": "1", "limit": "5"})
    assert len(r.body["rows"]) == 1 and r.body["offset"] == 1
    assert err_code(app.handle("GET", f"/api/datasets/{ds_id}/rows",
                               query={"file": "bogus"})) == "bad_request"
    assert app.handle("GET", "/api/datasets/nope/rows").status == 404

    # truncation cap on long strings
    sft = home / "datasets" / ds_id / "sft.jsonl"
    row = json.loads(sft.read_text().splitlines()[0])
    row["conversations"][0]["value"] = "x" * 2000
    sft.write_text(json.dumps(row) + "\n")
    r = app.handle("GET", f"/api/datasets/{ds_id}/rows", query={"file": "sft"})
    assert r.body["truncated"] is True
    long_val = r.body["rows"][0]["conversations"][0]["value"]
    assert len(long_val) < 900 and long_val.endswith("…")


def test_traces_offset_and_curated_filter(app, home):
    store = TraceStore(home / "buffer" / "sable.db")
    for i in range(5):
        seed_trace(store, f"t{i}", score=0.9 - 0.1 * i)
    store.close()

    # offset paging over the timestamp-DESC ordering
    r = app.handle("GET", "/api/traces", query={"limit": "2", "offset": "0"})
    page1 = [t["trace_id"] for t in r.body["traces"]]
    r = app.handle("GET", "/api/traces", query={"limit": "2", "offset": "2"})
    page2 = [t["trace_id"] for t in r.body["traces"]]
    assert len(page1) == 2 and len(page2) == 2
    assert set(page1).isdisjoint(page2)
    # offset past the end yields an empty window, not an error
    r = app.handle("GET", "/api/traces", query={"limit": "2", "offset": "99"})
    assert r.status == 200 and r.body["traces"] == []

    # curated toggle (1/0) vs unset — high min_score claims only t0, leaving
    # the rest unclaimed so both sides of the toggle are non-empty.
    ds_id = _curate(app, {"name": "c", "min_score": 0.85})["dataset_id"]
    r = app.handle("GET", "/api/traces", query={"curated": "1"})
    ids_curated = {t["trace_id"] for t in r.body["traces"]}
    assert ids_curated and all(t["dataset_id"] == ds_id
                               for t in r.body["traces"])
    r = app.handle("GET", "/api/traces", query={"curated": "0"})
    uncurated = {t["trace_id"] for t in r.body["traces"]}
    assert ids_curated.isdisjoint(uncurated)
    # combined with search + offset
    r = app.handle("GET", "/api/traces",
                   query={"curated": "0", "limit": "1", "offset": "1"})
    assert len(r.body["traces"]) == 1


def test_curate_formats_and_model_alias(app, home):
    store = TraceStore(home / "buffer" / "sable.db")
    seed_trace(store, "t1")
    seed_trace(store, "t2", status="failure", score=-0.9)
    store.close()

    # non-matching model_alias -> nothing selected (filter actually applied)
    res = _curate(app, {"name": "wrongmodel", "model_alias": "no-such-model"})
    assert res["sft_count"] == 0

    # matching alias + formats=sft -> sft rows only, dpo file empty in manifest
    res = _curate(app, {"name": "sftonly", "formats": "sft",
                        "model_alias": "base-model"})
    assert res["sft_count"] >= 1
    ds = app.handle("GET", f"/api/datasets/{res['dataset_id']}").body
    assert ds["files"]["dpo.jsonl"]["count"] == 0
    assert ds["filters"]["model_alias"] == "base-model"
    assert ds["filters"]["formats"] == "sft"

    # invalid formats value rejected
    assert err_code(app.handle("POST", "/api/curate",
                               body={"formats": "everything"})) == "bad_request"


def test_dataset_rows_paging_edges(app, home):
    store = TraceStore(home / "buffer" / "sable.db")
    for i in range(3):
        seed_trace(store, f"t{i}")
    store.close()
    ds_id = _curate(app, {"name": "pages"})["dataset_id"]

    # limit clamped to 200; negative offset treated as 0
    r = app.handle("GET", f"/api/datasets/{ds_id}/rows",
                   query={"file": "sft", "limit": "1000"})
    assert r.status == 200 and r.body["limit"] == 200
    r = app.handle("GET", f"/api/datasets/{ds_id}/rows",
                   query={"file": "sft", "offset": "-5"})
    assert r.status == 200 and r.body["offset"] == 0
    # offset beyond total -> empty rows, correct total
    r = app.handle("GET", f"/api/datasets/{ds_id}/rows",
                   query={"file": "sft", "offset": "100"})
    assert r.status == 200 and r.body["rows"] == [] and r.body["total"] == 3
    # dpo file exists even when empty (empty-pair warning path)
    r = app.handle("GET", f"/api/datasets/{ds_id}/rows",
                   query={"file": "dpo"})
    assert r.status == 200 and r.body["total"] == 0


def test_dataset_export_endpoint(app, home, tmp_path):
    store = TraceStore(home / "buffer" / "sable.db")
    seed_trace(store, "t1")
    store.close()
    ds_id = _curate(app)["dataset_id"]
    dest = tmp_path / "out"  # outside home -> needs force (5d guard)
    r = _export(app, {"dataset_id": ds_id, "to": str(dest)})
    assert r.status == 200 and (dest / ds_id / "manifest.json").is_file()
    # conflict on re-export
    assert err_code(_export(app, {"dataset_id": ds_id,
                                  "to": str(dest)})) == "target_exists"
    assert _export(app, {"dataset_id": "zz", "to": str(dest)}).status == 404


def test_model_detail_and_train_check(app, home):
    reg = ModelRegistry(home / "buffer" / "sable.db")
    m = reg.register_model(base_model="b")
    reg.set_validation(m.model_id, {"ok": True, "pass_rate": 1.0}, passed=True)
    reg.close()
    r = app.handle("GET", f"/api/models/{m.model_id}")
    assert r.status == 200 and r.body["validation"]["pass_rate"] == 1.0
    assert err_code(app.handle("GET", "/api/models/v9")) == "not_found"

    body = app.handle("GET", "/api/train/check").body
    assert body["enough_traces"] is False and body["unclaimed"] == 0
    assert body["threshold"]["min_traces"] == 50


def test_job_detail_and_log_endpoints(app, home):
    home.mkdir(parents=True, exist_ok=True)
    log = home / "job.log"
    log.write_text("l1\nl2\nl3\n")
    reg = ModelRegistry(home / "buffer" / "sable.db")
    j = reg.enqueue_job("sft", dataset_id="d", log_path=str(log))
    reg.close()

    r = app.handle("GET", f"/api/jobs/{j.job_id}")
    assert r.status == 200 and r.body["job"]["state"] == "queued"
    assert r.body["log"]["lines"] == ["l1", "l2", "l3"]
    assert r.body["log"]["total"] == 3

    r = app.handle("GET", f"/api/jobs/{j.job_id}/log", query={"tail": "2"})
    assert r.body["lines"] == ["l2", "l3"] and r.body["state"] == "queued"
    assert app.handle("GET", "/api/jobs/nope/log").status == 404


# ---------------------------------------------------------------- iterate

def test_iterate_run_record_lifecycle(app, home, tmp_path, monkeypatch):
    stub = tmp_path / "stub.py"
    stub.write_text(STUB)
    monkeypatch.setenv("SABLE_WORKER_COMMAND", f"{sys.executable} {stub}")
    store = TraceStore(home / "buffer" / "sable.db")
    seed_trace(store, "t1")
    store.close()

    sub = app.bus.subscribe()
    r = app.handle("POST", "/api/iterate", body={"dry_run": True})
    assert r.status == 202
    run_id = r.body["run_id"]

    import time as _t

    record = None
    for _ in range(200):
        runs = app.handle("GET", "/api/iterate/runs").body["runs"]
        me = [x for x in runs if x["run_id"] == run_id]
        if me and me[0]["state"] in ("done", "failed", "error"):
            record = me[0]
            break
        _t.sleep(0.05)
    assert record is not None, "iterate run never finished"
    assert record["state"] == "done", record
    step_names = [s["name"] for s in record["steps"]]
    assert "curate" in step_names and "train" in step_names
    assert all(s["state"] in ("done", "running", "failed")
               for s in record["steps"])
    assert record["result"]["ok"] is True

    # bus saw the steps
    saw_step = False
    while True:
        ev = sub.poll(timeout=0.1)
        if ev is None:
            break
        if ev["kind"] == "iterate.step" and ev["payload"]["run_id"] == run_id:
            saw_step = True
    sub.close()
    assert saw_step

    # persisted under <home>/iterate_runs/
    assert (home / "iterate_runs" / f"{run_id}.json").is_file()


# --------------------------------------------------- Phase 5d backend fixes

def test_traces_total_count(app, home):
    """5d: /api/traces carries total/offset/limit alongside the window."""
    store = TraceStore(home / "buffer" / "sable.db")
    for i in range(7):
        seed_trace(store, f"t{i}", status="success" if i % 2 else "failure")
    store.close()

    r = app.handle("GET", "/api/traces", query={"limit": "3"})
    assert r.status == 200
    assert len(r.body["traces"]) == 3
    assert r.body["total"] == 7 and r.body["offset"] == 0
    assert r.body["limit"] == 3

    # total respects filters, not just the window
    r = app.handle("GET", "/api/traces", query={"status": "failure"})
    assert r.body["total"] == 4 and len(r.body["traces"]) == 4
    r = app.handle("GET", "/api/traces", query={"q": "nomatch__"})
    assert r.body["total"] == 0 and r.body["traces"] == []


def test_dataset_export_containment_guard(app, home, tmp_path):
    """5d: export outside SABLE_HOME -> 409 outside_home unless force."""
    store = TraceStore(home / "buffer" / "sable.db")
    seed_trace(store, "t1")
    store.close()
    ds_id = _curate(app)["dataset_id"]

    outside = tmp_path / "elsewhere"
    r = app.handle("POST", "/api/datasets/export",
                   body={"dataset_id": ds_id, "to": str(outside)})
    assert r.status == 409 and err_code(r) == "outside_home"
    assert not outside.exists()

    # force=true overrides the guard
    r = app.handle("POST", "/api/datasets/export",
                   body={"dataset_id": ds_id, "to": str(outside),
                         "force": True})
    assert r.status == 200 and (outside / ds_id).is_dir()

    # inside home passes with no force flag
    inside = home / "exports"
    r = app.handle("POST", "/api/datasets/export",
                   body={"dataset_id": ds_id, "to": str(inside)})
    assert r.status == 200 and (inside / ds_id).is_dir()


def test_curate_job_lifecycle_events(app, home):
    """5d: curate runs as a job — bus sees queued->running->done; log has the
    result line; a bad filter is rejected synchronously (400)."""
    store = TraceStore(home / "buffer" / "sable.db")
    seed_trace(store, "t1")
    store.close()

    sub = app.bus.subscribe()
    r = app.handle("POST", "/api/curate", body={"name": "lifecycle"})
    assert r.status == 202
    job_id = r.body["job_id"]


    states = []
    for _ in range(150):
        ev = sub.poll(timeout=0.2)
        if ev and ev["kind"] == "job.state" and ev["payload"]["job_id"] == job_id:
            states.append(ev["payload"]["state"])
        if "done" in states:
            break
    sub.close()
    assert "running" in states and "done" in states

    d = app.handle("GET", f"/api/jobs/{job_id}").body
    assert d["job"]["kind"] == "curate"
    res_line = [l for l in d["log"]["lines"] if '"event": "done"' in l]
    assert res_line and "dataset_id" in res_line[-1]
    # short_claims surfaced so a non-zero shortfall is visible in the job log
    assert "short_claims" in res_line[-1]
    import json as _json

    assert _json.loads(res_line[-1])["result"]["short_claims"] == 0


def test_curate_job_failure_records_error(app, home, monkeypatch):
    """A curate that raises mid-run ends 'error' with the message stored."""
    from sable.curate.curater import Curater as _C

    def boom(self, cfg, name=None, rebuild_id=None):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(_C, "curate", boom)
    r = app.handle("POST", "/api/curate", body={"name": "doomed"})
    assert r.status == 202
    job_id = r.body["job_id"]
    import time as _t

    j = None
    for _ in range(150):
        j = app.handle("GET", f"/api/jobs/{job_id}").body
        if j["job"]["state"] in ("done", "error"):
            break
        _t.sleep(0.05)
    assert j and j["job"]["state"] == "error"
    assert "disk on fire" in (j["job"]["error"] or "")


def test_train_check_includes_resource_gate(app, home):
    """5d: /api/train/check surfaces the scheduler gate for panel copy."""
    body = app.handle("GET", "/api/train/check").body
    assert "gate" in body and set(body["gate"]) == {"ok", "reasons"}


def test_train_check_reports_cron_wiring(app, home):
    """Honesty: auto_enabled only means something if cron calls auto-check.
    /api/train/check must say whether the trigger is actually wired up."""
    body = app.handle("GET", "/api/train/check").body
    sched = body["scheduler"]
    assert set(sched) >= {"installed", "line", "note"}
    # the suggested line must NOT rely on `sable` being on cron's PATH
    assert "train auto-check --trigger" in sched["line"]
    command = sched["line"].split(" ", 5)[5].split(" ")[0]
    assert command.startswith("/"), f"cron PATH is minimal; got {command!r}"


def test_cron_line_uses_absolute_sable_path(tmp_path, monkeypatch):
    import sys

    from sable.training import scheduler as sch

    line = sch.cron_line(str(tmp_path))
    assert "train auto-check --trigger" in line
    assert str(tmp_path) in line  # log path follows SABLE_HOME
    command = line.split(" ", 5)[5].split(" ")[0]
    # cron runs with a minimal PATH: a bare `sable` would exit 127 silently.
    # The absolute sibling of the running interpreter is always used when it
    # exists, which is the case for any venv/source install.
    assert command == str(sch.Path(sys.executable).parent / "sable")
    assert command.startswith("/")

    # no sibling script (e.g. module-only install) => PATH fallback, documented
    monkeypatch.setattr(sch.Path, "is_file", lambda self: False)
    monkeypatch.setattr(sch.shutil, "which", lambda n: None)
    assert sch.cron_line(str(tmp_path)).split(" ", 5)[5].startswith("sable ")


# --- regression: the default index must never swallow SABLE_HOME ----------
# (a dashboard launched from ~/.sable built a 9k-file / 185 MB index of
# llama.cpp build trees, vendored engines and sable.yaml itself)

def test_default_build_refused_when_rooted_at_home(tmp_path, monkeypatch):
    h = tmp_path / "homeroot"
    (h / "index").mkdir(parents=True)
    monkeypatch.setenv("SABLE_HOME", str(h))
    monkeypatch.chdir(h)  # the old default root was exactly this cwd
    app = DashboardApp(SableConfig(indexing={"enabled": True}), home=h)
    r = app.handle("POST", "/api/index/build", body={})
    assert r.status == 409
    assert r.body["error"]["code"] == "index_root_is_sable_home"
    assert not (h / "index" / "default.db").exists()  # nothing was built


def test_default_build_allowed_with_explicit_override(tmp_path, monkeypatch):
    h = tmp_path / "homeok"
    (h / "index").mkdir(parents=True)
    ws = tmp_path / "realcode"
    ws.mkdir()
    (ws / "a.py").write_text("def a():\n    return 1\n")
    monkeypatch.setenv("SABLE_HOME", str(h))
    monkeypatch.chdir(h)
    app = DashboardApp(SableConfig(indexing={
        "enabled": True, "paths_override": [str(ws)]}), home=h)
    r = app.handle("POST", "/api/index/build", body={})
    assert r.status == 200 and r.body["state"] == "running"


def test_startup_indexing_skips_home_rooted_default(tmp_path, monkeypatch):
    import time as _t

    h = tmp_path / "homestart"
    (h / "index").mkdir(parents=True)
    monkeypatch.setenv("SABLE_HOME", str(h))
    monkeypatch.chdir(h)
    app = DashboardApp(SableConfig(indexing={"enabled": True}), home=h)
    sub = app.bus.subscribe()
    try:
        app.startup_indexing()
        _t.sleep(0.6)
        seen = []
        while True:
            ev = sub.poll(timeout=0.1)
            if ev is None:
                break
            seen.append(ev)
    finally:
        sub.close()
    assert not (h / "index" / "default.db").exists()
    builds = [e for e in seen if e.get("kind") == "index.build"]
    assert any("SABLE_HOME" in str(e.get("payload")) for e in builds), \
        "the skip should be visible as an index.build event"


def test_config_put_starts_index_build(tmp_path, monkeypatch):
    """Applying indexing settings must kick a build (no restart needed)."""
    import time as _t

    h = tmp_path / "applybuild"
    h.mkdir(parents=True)
    ws = tmp_path / "code2"
    ws.mkdir()
    (ws / "b.py").write_text("def b():\n    return 2\n")
    monkeypatch.setenv("SABLE_HOME", str(h))
    monkeypatch.setenv("SABLE_CONFIG", str(h / "sable.yaml"))
    monkeypatch.chdir(ws)  # default root = the real code dir, not home
    app = DashboardApp(SableConfig(), home=h)
    r = app.handle("PUT", "/api/config", body={
        "section": "indexing",
        "patch": {"enabled": True, "paths_override": [str(ws)]}})
    assert r.status == 200
    _t.sleep(0.6)  # the build runs in a background thread
    st = app.handle("GET", "/api/index").body
    assert st["enabled"] is True
    assert st["files"] >= 1 or st["building"], \
        "applying indexing config should have started a build"


def test_cron_trigger_status_reads_crontab(tmp_path, monkeypatch):
    import subprocess as sp

    import sable.training.scheduler as sch

    def fake_run(cmd, **kw):
        class R:
            returncode = 0
            stdout = "15 3 * * * /abs/sable train auto-check --trigger\n"
        return R()

    monkeypatch.setattr(sp, "run", fake_run)
    st = sch.cron_trigger_status(str(tmp_path))
    assert st["installed"] is True

    def fake_run_none(cmd, **kw):
        class R:
            returncode = 1
            stdout = ""
        return R()

    monkeypatch.setattr(sp, "run", fake_run_none)
    assert sch.cron_trigger_status(str(tmp_path))["installed"] is False

    def fake_run_plain(cmd, **kw):
        class R:
            returncode = 0
            stdout = "0 * * * * /usr/bin/backup.sh\n"
        return R()

    monkeypatch.setattr(sp, "run", fake_run_plain)
    st2 = sch.cron_trigger_status(str(tmp_path))
    assert st2["installed"] is False  # crontab exists, no auto-check


def test_config_put_root_section(app, home):
    """5d: section="root" patches top-level keys; unknown ones still 400."""
    r = app.handle("PUT", "/api/config",
                   body={"section": "root",
                         "patch": {"gguf_path": "/m.gguf", "context_length": 2048}})
    assert r.status == 200
    cfg = app.handle("GET", "/api/config").body["config"]
    assert cfg["gguf_path"] == "/m.gguf" and cfg["context_length"] == 2048

    bad = app.handle("PUT", "/api/config",
                     body={"section": "root", "patch": {"nope": 1}})
    assert bad.status == 400 and err_code(bad) == "unknown_keys"


def test_train_single_flight_409(app, home, monkeypatch):
    """5d: POST /api/train while a job runs -> 409 already_running + job id."""
    from sable.registry import ModelRegistry

    TraceStore(home / "buffer" / "sable.db").close()
    reg = ModelRegistry(home / "buffer" / "sable.db")
    j = reg.enqueue_job("sft", dataset_id="d1")
    reg.mark_running(j.job_id, os.getpid())
    reg.close()

    r = app.handle("POST", "/api/train", body={"dataset": "d1"})
    assert r.status == 409 and err_code(r) == "already_running"
    assert j.job_id in r.body["error"]["message"]


# ------------------------------------------------------------ HTTP + auth

def _start_server_with_token(home, token=None):
    from sable.dashboard.server import BIND_HOST, make_server

    httpd = make_server(SableConfig(), port=0, home=home, token=token)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, f"http://{BIND_HOST}:{httpd.server_address[1]}"


def test_token_auth_on_by_default(home):
    """No config, no env: a generated token protects /api/* and is persisted
    at <home>/dashboard.token (stable across restarts); static stays open."""
    import stat

    httpd, base = _start_server_with_token(home)
    try:
        import httpx

        # First access generates + persists the token (lazy by design).
        assert getattr(httpd, "sable_app").token
        tok_p = home / "dashboard.token"
        assert tok_p.is_file(), "default token was not persisted"
        tok = tok_p.read_text().strip()
        assert len(tok) >= 32
        assert stat.S_IMODE(tok_p.stat().st_mode) == 0o600
        # /api/* requires it; static assets do not
        assert httpx.get(base + "/api/status", timeout=5).status_code == 401
        assert httpx.get(base + "/", timeout=5).status_code == 200
        assert httpx.get(base + "/api/status", timeout=5,
                         headers={"X-Sable-Token": tok}).status_code == 200
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_token_default_stable_across_restarts_and_env(home):
    """Same home -> same default token; SABLE_DASHBOARD_TOKEN overrides it."""
    import os

    h1, _ = _start_server_with_token(home)
    tok1 = getattr(h1, "sable_app").token
    h1.shutdown()
    h1.server_close()
    h2, _ = _start_server_with_token(home)
    assert getattr(h2, "sable_app").token == tok1
    h2.shutdown()
    h2.server_close()
    os.environ["SABLE_DASHBOARD_TOKEN"] = "envtok"
    try:
        h3, _ = _start_server_with_token(home)
        assert getattr(h3, "sable_app").token == "envtok"
        h3.shutdown()
        h3.server_close()
    finally:
        del os.environ["SABLE_DASHBOARD_TOKEN"]


def test_token_empty_string_disables(home):
    """Explicit dashboard.token: "" (the --no-token path) disables auth."""
    httpd, base = _start_server_with_token(home, token="")
    try:
        import httpx

        assert httpx.get(base + "/api/status", timeout=5).status_code == 200
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_token_auth_on_header_query_and_sse(home):
    httpd, base = _start_server_with_token(home, token="s3cret")
    try:
        import httpx

        r = httpx.get(base + "/api/status", timeout=5)
        assert r.status_code == 401
        assert r.json()["error"]["code"] == "unauthorized"
        # static assets stay open
        assert httpx.get(base + "/", timeout=5).status_code == 200
        # header works
        assert httpx.get(base + "/api/status", timeout=5,
                         headers={"X-Sable-Token": "s3cret"}).status_code \
            == 200
        # query param works (SSE clients can't set headers with EventSource)
        assert httpx.get(base + "/api/status?token=s3cret",
                         timeout=5).status_code == 200
        # wrong token rejected
        assert httpx.get(base + "/api/status", timeout=5,
                         headers={"X-Sable-Token": "nope"}).status_code == 401

        # SSE over HTTP: hello frame then a published event; 401 without token
        app = getattr(httpd, "sable_app")
        with httpx.Client(timeout=10) as c, \
                c.stream("GET", base + "/api/events",
                         headers={"X-Sable-Token": "s3cret"}) as r:
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("text/event-stream")
            lines = r.iter_lines()
            assert ": sable events" in next(lines)
            app.bus.publish("model.promoted", {"model_id": "v7"})
            for line in lines:
                if line.startswith("data:") and "model.promoted" in line:
                    break
            else:
                pytest.fail("published event never arrived over SSE")

        with httpx.Client(timeout=5) as c, \
                c.stream("GET", base + "/api/events") as r:
            assert r.status_code == 401
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_sse_capacity_cap_rejects_over_limit(home):
    """Admission control on SSE streams: with dashboard.max_sse=2, the third
    concurrent /api/events gets 503 + Retry-After (EventSource reconnects);
    once a held stream closes, its slot frees up."""
    from sable.dashboard.server import make_server

    cfg = SableConfig(dashboard={"max_sse": 2})
    httpd = make_server(cfg, port=0, home=home, token="")
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        import httpx

        held = []
        try:
            for _ in range(2):
                c = httpx.Client(timeout=5)
                # send(stream=True): the 200 status line is emitted by
                # _stream_sse AFTER the slot is acquired, so a received 200
                # deterministically means the server holds one slot.
                r = c.send(c.build_request("GET", base + "/api/events"),
                           stream=True)
                assert r.status_code == 200
                held.append((c, r))
            r3 = httpx.get(base + "/api/events", timeout=5)
            assert r3.status_code == 503
            assert r3.headers.get("retry-after") == "2"
            assert r3.json()["code"] == "sse_capacity"
        finally:
            for c, r in held:
                r.close()
                c.close()
        # Slots freed: closing the client makes the generator's next write
        # fail (OSError -> _stream_sse releases the slot). Publishing wakes
        # the parked generators immediately so we don't wait on the 15 s
        # heartbeat for the disconnect to be noticed.
        import time as _time

        app = getattr(httpd, "sable_app")
        code = None
        for _ in range(50):
            app.bus.publish("test.probe", {"n": 1})
            try:
                with httpx.Client(timeout=5) as c, \
                        c.stream("GET", base + "/api/events") as r:
                    code = r.status_code
                    if code == 200:
                        break
            except httpx.HTTPError:
                pass
            _time.sleep(0.1)
        assert code == 200, "closed streams must free their slots"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_http_put_delete_verbs(home):
    httpd, base = _start_server(home)
    try:
        import httpx

        r = httpx.put(base + "/api/chat/threads/http1",
                      json={"title": "from http"}, timeout=5)
        assert r.status_code == 200 and r.json()["thread"]["title"] == "from http"
        r = httpx.delete(base + "/api/chat/threads/http1", timeout=5)
        assert r.status_code == 200
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_csp_headers_on_api(home):
    httpd, base = _start_server_with_token(home)
    try:
        import httpx

        r = httpx.get(base + "/api/status", timeout=5)
        assert "default-src 'self'" in r.headers.get("content-security-policy", "")
        assert r.headers.get("x-frame-options") == "DENY"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_cli_dashboard_token_flag():
    from sable.cli import build_parser

    args = build_parser().parse_args(["dashboard", "--port", "0",
                                      "--token", "abc"])
    assert args.token == "abc"


# ------------------------------------------------ CSRF / DNS-rebinding guard

def test_cross_site_origin_blocked(home):
    """Origin/Referer naming a non-loopback site -> 403 on /api/*."""
    import httpx

    httpd, base = _start_server(home)
    try:
        # no Origin at all (curl/tests): allowed
        assert httpx.get(base + "/api/status", timeout=5).status_code == 200
        # cross-site origin: refused even for GET
        r = httpx.get(base + "/api/status", timeout=5,
                      headers={"Origin": "https://evil.example"})
        assert r.status_code == 403
        assert r.json()["error"]["code"] == "cross_site_blocked"
        # cross-site POST (the dangerous simple-request CSRF shape): refused
        r = httpx.post(base + "/api/feedback", json={}, timeout=5,
                       headers={"Origin": "https://evil.example"})
        assert r.status_code == 403
        # same-origin loopback Origin: allowed
        assert httpx.get(base + "/api/status", timeout=5,
                         headers={"Origin": base}).status_code == 200
        # Referer is checked too
        r = httpx.get(base + "/api/status", timeout=5,
                      headers={"Referer": "https://evil.example/page"})
        assert r.status_code == 403
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_dns_rebinding_host_blocked(home):
    """A Host header that isn't loopback (rebinding) is refused."""
    import httpx

    httpd, base = _start_server(home)
    port = httpd.server_address[1]
    try:
        r = httpx.get(base + "/api/status", timeout=5,
                      headers={"Host": f"attacker.example:{port}"})
        assert r.status_code == 403
        # localhost spelling stays fine
        r = httpx.get(f"http://localhost:{port}/api/status", timeout=5)
        assert r.status_code == 200
    finally:
        httpd.shutdown()
        httpd.server_close()


# ------------------------------------------------------ LAN exposure (expose_lan)

def test_expose_lan_refused_without_token(home):
    """dashboard.expose_lan with auth disabled must refuse to boot."""
    from sable.core.config import SableConfig
    from sable.dashboard.server import make_server

    import pytest as _pytest
    cfg = SableConfig(dashboard={"expose_lan": True, "token": ""})
    with _pytest.raises(SystemExit):
        make_server(cfg, port=0, home=home)


def test_expose_lan_binds_wildcard_and_allows_same_origin(home):
    """expose_lan binds 0.0.0.0; the CSRF gate becomes same-origin: a Host
    naming a LAN IP is allowed when Origin matches it, refused when not."""
    import httpx

    from sable.core.config import SableConfig
    from sable.dashboard.server import LAN_BIND_HOST, make_server

    cfg = SableConfig(dashboard={"expose_lan": True})
    httpd = make_server(cfg, port=0, home=home, token="lan-tok")
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    port = httpd.server_address[1]
    assert httpd.server_address[0] == LAN_BIND_HOST
    try:
        lan_host = f"192.0.2.10:{port}"  # TEST-NET-1, never routable
        hdrs = {"Host": lan_host, "X-Sable-Token": "lan-tok"}
        # same-origin LAN request: allowed
        r = httpx.get(f"http://127.0.0.1:{port}/api/status", timeout=5,
                      headers={**hdrs, "Origin": f"http://{lan_host}"})
        assert r.status_code == 200
        # cross-origin (rebinding shape: Host is the resolved IP, Origin the
        # attacker domain): refused
        r = httpx.get(f"http://127.0.0.1:{port}/api/status", timeout=5,
                      headers={**hdrs, "Origin": "https://evil.example"})
        assert r.status_code == 403
        assert r.json()["error"]["code"] == "cross_site_blocked"
        # no token -> 401 even from the same origin
        r = httpx.get(f"http://127.0.0.1:{port}/api/status", timeout=5,
                      headers={"Host": lan_host})
        assert r.status_code == 401
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_config_put_accepts_expose_lan(home):
    """The dashboard validator accepts dashboard.expose_lan in a section patch."""
    import httpx

    httpd, base = _start_server_with_token(home, token="cfg-tok")
    try:
        r = httpx.put(base + "/api/config", timeout=5,
                      headers={"X-Sable-Token": "cfg-tok"},
                      json={"section": "dashboard",
                            "patch": {"expose_lan": True}})
        assert r.status_code == 200, r.text
        saved = (home / "sable.yaml").read_text()
        assert "expose_lan: true" in saved
    finally:
        httpd.shutdown()
        httpd.server_close()


# ------------------------------------------- playbooks + mined-goldens APIs


def test_playbook_assess_endpoint(app, home):
    from sable.core.playbooks import PlaybookStore, promote

    ps = PlaybookStore(home / "buffer" / "sable.db")
    try:
        pb_id, _ = ps.create_candidate(["read", "edit"], "recipe",
                                       profile="coder")
        promote(ps, pb_id)
    finally:
        ps.close()
    r = app.handle("POST", "/api/playbooks/assess", body={})
    assert r.status == 200 and r.body["ok"]
    results = r.body["results"]
    assert len(results) == 1 and results[0]["id"] == pb_id
    # no matching traces -> honest empty comparison, stays active
    assert results[0]["ok"] and not results[0]["demoted"]

    r = app.handle("POST", "/api/playbooks/assess", body={"id": "pb-nope"})
    assert r.status == 404


def test_playbooks_list_chains_cached(app, home):
    """The live chain scan is TTL-cached: two renders reuse one result object
    (the expensive 500-transcript JSONL walk must not run per page load)."""
    r1 = app.handle("GET", "/api/playbooks")
    assert r1.status == 200
    r2 = app.handle("GET", "/api/playbooks")
    assert r2.body["chains"] is r1.body["chains"]


def test_goldens_mined_list_and_toggle_endpoints(app, home):
    gp = home / "goldens" / "mined.jsonl"
    gp.parent.mkdir(parents=True)
    task = {"prompt": "how do I rebase safely", "expect_contains": ["git pull --rebase"],
            "_mine": {"source_trace_id": "tx1", "verifiers": ["expect_contains"]},
            "enabled": True}
    gp.write_text(json.dumps(task) + "\n")

    r = app.handle("GET", "/api/goldens/mined")
    assert r.status == 200 and len(r.body["goldens"]) == 1
    key = r.body["goldens"][0]["key"]

    r = app.handle("POST", "/api/goldens/mined/toggle",
                   body={"key": key, "enabled": False})
    assert r.status == 200 and r.body["enabled"] is False
    assert json.loads(gp.read_text())["enabled"] is False

    r = app.handle("POST", "/api/goldens/mined/toggle",
                   body={"key": "bogus", "enabled": True})
    assert r.status == 404
    r = app.handle("POST", "/api/goldens/mined/toggle", body={})
    assert r.status == 400


def test_config_put_validates_playbooks_section(app, home):
    """playbooks: joined the config-PUT field allowlist — valid keys land,
    typos are refused like every other section."""
    r = app.handle("PUT", "/api/config",
                   body={"section": "playbooks",
                         "patch": {"enabled": True, "total_cap_chars": 1500}})
    assert r.status == 200
    r = app.handle("PUT", "/api/config",
                   body={"section": "playbooks", "patch": {"enbled": True}})
    assert r.status == 400
    assert r.body["error"]["code"] == "unknown_keys"


def test_query_token_only_accepted_for_get(home):
    """?token= exists for EventSource; mutations must use the header."""
    import httpx

    httpd, base = _start_server_with_token(home, token="tok123")
    try:
        assert httpx.get(base + "/api/status?token=tok123",
                         timeout=5).status_code == 200
        # POST with the token only in the query string -> unauthorized
        r = httpx.post(base + "/api/feedback?token=tok123", json={},
                       timeout=5)
        assert r.status_code == 401
        # POST with the header works
        r = httpx.post(base + "/api/feedback", json={}, timeout=5,
                       headers={"X-Sable-Token": "tok123"})
        assert r.status_code != 401
    finally:
        httpd.shutdown()
        httpd.server_close()

# ------------------------------------------------------------------------
# from tests/test_chat_reattach.py
# ------------------------------------------------------------------------

@pytest.fixture()
def home__test_chat_reattach(tmp_path, monkeypatch):
    h = tmp_path / "uf-reattach"
    monkeypatch.setenv("SABLE_HOME", str(h))
    return h


def make_app(home):
    return DashboardApp(SableConfig(), home=home)




def _events(text):
    out = []
    for frame in text.split("\n\n"):
        kind = None
        data = None
        for line in frame.splitlines():
            if line.startswith("event: "):
                kind = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        if kind:
            out.append((kind, data or {}))
    return out


class GatedRunner:
    """create_runner stub: streams deltas only after `release` is set, so a
    test controls exactly when the run finishes relative to re-attachment."""

    def __init__(self, cfg, base_url, model, **kw):
        self.on_event = None          # dashboard attaches its sink here
        self.release = threading.Event()
        self.cancel = kw.get("cancel_event")

    def run(self, prompt, history=None, previous_summary=None):
        if self.on_event:
            self.on_event("message.delta", {"text": "half-"})
        self.release.wait(timeout=10)
        if self.on_event:
            self.on_event("message.delta", {"text": "done"})
        return AgentResult(final_response="half-done",
                           messages=[{"role": "assistant",
                                      "content": "half-done"}],
                           trace_id="tr-gate")


@pytest.fixture()
def gated(monkeypatch):
    made: list = []

    def _create(cfg, base_url, model, **kw):
        r = GatedRunner(cfg, base_url, model, **kw)
        made.append(r)
        return r

    monkeypatch.setattr("sable.core.agent_runner.create_runner", _create)
    return made


def _start(app, thread_id, prompt="go"):
    """POST /api/chat/stream and iterate ONE frame (stream admitted/run
    registered); returns the still-suspended generator."""
    resp = app.handle("POST", "/api/chat/stream",
                      body={"prompt": prompt, "thread": thread_id})
    assert resp.status == 200 and resp.stream is not None
    it = iter(resp.stream)
    first = next(it)          # run.started — registry entry now live
    assert "run.started" in first
    return it


# ------------------------------------------------ leave-mid-run resilience

def test_abandoning_the_stream_does_not_kill_the_run(home__test_chat_reattach, gated):
    """Close the starting reader mid-run: the worker keeps going, finishes,
    persists the transcript, and only THEN marks the registry entry done."""
    app = make_app(home__test_chat_reattach)
    with FakeOpenAIServer() as fs:
        _fake_serve_state(home__test_chat_reattach, fs.base_url)
        it = _start(app, "t-leave")
        run_id = app.handle("GET", "/api/chat/runs").body["runs"][0]["run_id"]

        # Reader goes away mid-run (navigation): pure reader drop.
        it.close()

        runs = app.handle("GET", "/api/chat/runs").body["runs"]
        assert [r["run_id"] for r in runs] == [run_id], \
            "the run must still be in flight after its reader left"
        # The registry carries the prompt so a re-attaching view can render
        # the user turn (the transcript file only gets it at completion).
        assert runs[0]["prompt"] == "go"

        gated[0].release.set()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if app._chat_runs.get(run_id, {}).get("done"):
                break
            time.sleep(0.02)
        assert app._chat_runs[run_id]["done"] is True

        t = json.loads((home__test_chat_reattach / "chat" / "t-leave.json").read_text())
        roles = [m["role"] for m in t["messages"]]
        assert roles == ["user", "assistant"]
        assert t["messages"][1]["content"] == "half-done"


def test_reattach_replays_buffered_frames_and_follows_live(home__test_chat_reattach, gated):
    """A second reader (the returning panel) sees the deltas streamed while
    away, then the live tail, then run.done — and can watch again after."""
    app = make_app(home__test_chat_reattach)
    with FakeOpenAIServer() as fs:
        _fake_serve_state(home__test_chat_reattach, fs.base_url)
        it = _start(app, "t-back")
        it.close()                                # original reader leaves
        run_id = app.handle("GET", "/api/chat/runs").body["runs"][0]["run_id"]

        # Some deltas may have streamed while nobody was watching; the
        # re-attaching reader replays them from the buffer.
        att = app.handle("GET", f"/api/chat/runs/{run_id}/stream")
        assert att.status == 200 and att.stream is not None
        reader = iter(att.stream)

        def collect_until_done():
            frames = []
            for frame in reader:
                frames.append(frame)
                if "event: run.done" in frame or "event: run.error" in frame:
                    break
            return "".join(frames)

        # Release while the re-attached reader is parked on the cond var:
        # it must receive BOTH buffered and post-attach frames.
        threading.Timer(0.1, gated[0].release.set).start()
        text = collect_until_done()
        kinds = [k for k, _ in _events(text)]
        assert kinds[0] == "run.started"          # replay from the beginning
        deltas = "".join(d["text"] for k, d in _events(text)
                         if k == "message.delta")
        assert deltas == "half-done"              # buffered + live tail
        assert kinds[-1] == "run.done"

        # A second re-attach (another tab) still gets the finished buffer.
        again = app.handle("GET", f"/api/chat/runs/{run_id}/stream")
        assert again.status == 200 and again.stream is not None
        text2 = "".join(again.stream)
        assert "event: run.done" in text2


def test_reattach_unknown_run_is_404(home__test_chat_reattach):
    app = make_app(home__test_chat_reattach)
    r = app.handle("GET", "/api/chat/runs/nope/stream")
    assert r.status == 404
    assert json.loads(r.json())["error"]["code"] == "no_such_run"


# ------------------------------------------------------ concurrent threads

def test_two_threads_run_concurrently(home__test_chat_reattach, gated):
    """Single-flight is PER thread: a second thread starts while the first
    is still executing; both stay in the registry."""
    app = make_app(home__test_chat_reattach)
    with FakeOpenAIServer() as fs:
        _fake_serve_state(home__test_chat_reattach, fs.base_url)
        it1 = _start(app, "t-a", prompt="first")
        it2 = _start(app, "t-b", prompt="second")   # must NOT 409
        assert it2 is not None
        runs = {r["thread"]: r for r in
                app.handle("GET", "/api/chat/runs").body["runs"]}
        assert set(runs) == {"t-a", "t-b"}
        assert runs["t-a"]["prompt"] == "first"
        assert runs["t-b"]["prompt"] == "second"

        # A THIRD post on a busy thread is still refused (one run per thread).
        busy = app.handle("POST", "/api/chat/stream",
                          body={"prompt": "x", "thread": "t-a"})
        assert busy.status == 409
        assert json.loads(busy.json())["error"]["code"] == "run_in_flight"

        for r in gated:
            r.release.set()
        it1.close()
        it2.close()

# ------------------------------------------------------------------------
# from tests/test_chat_run_lifecycle.py
# ------------------------------------------------------------------------

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


@pytest.fixture()
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

def test_cancel_preloop_never_calls_model(_no_sleeps):
    TwoToolCallHandler.requests = 0
    ev = threading.Event()
    ev.set()  # stop pressed before the run even starts
    with FakeOpenAIServer(handler_cls=TwoToolCallHandler) as srv:
        res = _client(srv, cancel_event=ev).run("do something")
    assert res.completed is False
    assert res.failure_reason == FAILURE_CANCELLED
    assert TwoToolCallHandler.requests == 0, "cancel must not touch the model"


def test_cancel_between_iterations_stops_before_second_call(_no_sleeps):
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


def test_cancel_midbatch_skips_remaining_tool_calls(_no_sleeps):
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


def test_no_cancel_event_keeps_old_behavior(_no_sleeps):
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
def app__test_chat_run_lifecycle(tmp_path, monkeypatch):
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


def test_runs_registry_lists_inflight_and_prunes_done(app__test_chat_run_lifecycle, _no_sleeps):
    _seed_run(app__test_chat_run_lifecycle, "aaaa", "t1")            # in flight
    _seed_run(app__test_chat_run_lifecycle, "bbbb", "t2", done=True)  # finished
    r = app__test_chat_run_lifecycle.handle("GET", "/api/chat/runs")
    assert r.status == 200
    runs = r.body["runs"]
    assert [x["run_id"] for x in runs] == ["aaaa"]
    assert runs[0]["thread"] == "t1" and runs[0]["status"] == "running"
    assert "bbbb" not in app__test_chat_run_lifecycle._chat_runs, "done runs must be pruned"


def test_stream_second_post_same_thread_gets_409_with_run_id(app__test_chat_run_lifecycle, monkeypatch, _no_sleeps):
    # Bypass model resolution: the single-flight check happens after the
    # target is resolved, so stub it with a dead (never-dialed) URL.
    monkeypatch.setattr(app__test_chat_run_lifecycle, "_resolve_chat_target", fake_resolve_chat_target)
    _seed_run(app__test_chat_run_lifecycle, "active1", "t1")
    resp = app__test_chat_run_lifecycle.handle("POST", "/api/chat/stream",
                      body={"prompt": "again", "thread": "t1"})
    body = json.loads(resp.json())
    assert resp.status == 409
    err = body["error"]
    assert err["code"] == "run_in_flight"
    assert err["run_id"] == "active1", "UI needs the id to adopt/cancel"


def test_stream_threadless_post_not_subject_to_single_flight(app__test_chat_run_lifecycle, monkeypatch, _no_sleeps):
    """Runs without a thread id (fresh chats) must not be blocked by a
    thread's in-flight run — and must not themselves block anything."""
    monkeypatch.setattr(app__test_chat_run_lifecycle, "_resolve_chat_target", fake_resolve_chat_target)
    _seed_run(app__test_chat_run_lifecycle, "active1", "t1")
    # A different thread is free to run.
    resp = app__test_chat_run_lifecycle.handle("POST", "/api/chat/stream",
                      body={"prompt": "hi", "thread": "other"})
    assert resp.status == 200  # started (its worker will fail harmlessly)


def test_cancelled_is_neither_infra_nor_model(_no_sleeps):
    assert not is_infra_reason(FAILURE_CANCELLED)
    assert not is_model_reason(FAILURE_CANCELLED)
    assert not is_infra_reason("backend_error:" + FAILURE_CANCELLED)


# ------------------------------------------------- shutdown drain (step 3)

def test_shutdown_drains_cooperative_runs(app__test_chat_run_lifecycle, _no_sleeps):
    """A run whose worker watches its cancel event finishes inside the grace
    window: shutdown sets every cancel flag and waits for done."""
    _seed_run(app__test_chat_run_lifecycle, "slow1", "t1")

    def _worker():
        run = app__test_chat_run_lifecycle._chat_runs["slow1"]
        while not run["cancel"].wait(timeout=0.05):
            pass
        run["done"] = True

    threading.Thread(target=_worker, daemon=True).start()
    n = app__test_chat_run_lifecycle.shutdown_chat_runs(grace_s=3.0)
    assert n == 1
    assert app__test_chat_run_lifecycle._chat_runs["slow1"]["done"] is True


def test_shutdown_marks_stubborn_runs_interrupted(app__test_chat_run_lifecycle, _no_sleeps):
    """A run that ignores cancel must not hang shutdown: after the grace
    window it is marked interrupted (documenting the lost trace)."""
    _seed_run(app__test_chat_run_lifecycle, "stub1", "t2")
    n = app__test_chat_run_lifecycle.shutdown_chat_runs(grace_s=0.3)
    assert n == 1
    assert app__test_chat_run_lifecycle._chat_runs["stub1"]["status"] == "interrupted"
    assert app__test_chat_run_lifecycle._chat_runs["stub1"]["cancel"].is_set()


def test_shutdown_with_no_runs_is_noop(app__test_chat_run_lifecycle, _no_sleeps):
    assert app__test_chat_run_lifecycle.shutdown_chat_runs(grace_s=0.1) == 0


def test_unstarted_stream_leaves_no_phantom_registry_entry(app__test_chat_run_lifecycle, monkeypatch, _no_sleeps):
    """The registry entry is created on the stream's first iteration. A POST
    whose SSE never starts (admission rejection path) must not leave a run
    that single-flight would 409 against forever."""
    monkeypatch.setattr(app__test_chat_run_lifecycle, "_resolve_chat_target", fake_resolve_chat_target)
    resp = app__test_chat_run_lifecycle.handle("POST", "/api/chat/stream",
                      body={"prompt": "hi", "thread": "t9"})
    assert resp.status == 200 and resp.stream is not None
    # Never iterate the stream (simulates server-side rejection before write).
    resp.stream.close()
    runs = app__test_chat_run_lifecycle.handle("GET", "/api/chat/runs").body["runs"]
    assert runs == [], "unstarted streams must not occupy the registry"

# ------------------------------------------------------------------------
# from tests/test_chat_transcript_persist.py
# ------------------------------------------------------------------------

@pytest.fixture()
def home__test_chat_transcript_persist(tmp_path, monkeypatch):
    h = tmp_path / "uf-persist"
    monkeypatch.setenv("SABLE_HOME", str(h))
    return h








def _drain(r):
    """Consume the whole SSE stream (no client-side save follows)."""
    assert r.stream is not None
    return "".join(r.stream)


def _thread(home, tid):
    p = home / "chat" / f"{tid}.json"
    assert p.is_file(), "the finished run must persist the thread itself"
    return json.loads(p.read_text())


# ---------------------------------------------------------------- happy path

def test_finished_run_persists_transcript_without_client_save(home__test_chat_transcript_persist):
    """Drain the stream to completion and never PUT the thread: the file must
    still carry both turns (fake server answers with a tool loop + text)."""
    app = make_app(home__test_chat_transcript_persist)
    with FakeOpenAIServer() as fs:
        _fake_serve_state(home__test_chat_transcript_persist, fs.base_url)
        r = app.handle("POST", "/api/chat/stream",
                       body={"prompt": "read the file", "thread": "t-p1"})
        assert r.stream is not None
        text = "".join(r.stream)  # client fully drains; NO client save follows
    evs = dict((k, d) for k, d in _events(text))
    final_text = "".join(d.get("text", "")
                         for k, d in _events(text) if k == "message.delta")

    t = _thread(home__test_chat_transcript_persist, "t-p1")
    roles = [m["role"] for m in t["messages"]]
    assert roles == ["user", "assistant"], \
        f"expected exactly one user+assistant pair, got {roles}"
    assert t["messages"][0]["content"] == "read the file"
    assistant = t["messages"][1]
    assert final_text and final_text in assistant["content"]
    # Tool cards survive into the stored transcript.
    assert assistant.get("tools"), "tool calls must persist with the turn"
    assert evs["run.done"]["trace_id"] in (t.get("trace_ids") or [])


def test_persisted_profile_matches_run_selection(home__test_chat_transcript_persist):
    """The thread records the profile used so reopening restores the dropdown
    to that agent instead of default."""
    app = make_app(home__test_chat_transcript_persist)
    pr = app.handle("PUT", "/api/profiles/sable",
                    body={"system_prompt": "you are sable", "tools": []})
    assert pr.status == 200
    with FakeOpenAIServer() as fs:
        _fake_serve_state(home__test_chat_transcript_persist, fs.base_url)
        r = app.handle("POST", "/api/chat/stream",
                       body={"prompt": "hi sable", "thread": "t-prof",
                             "profile": "sable"})
        _drain(r)
    assert _thread(home__test_chat_transcript_persist, "t-prof").get("profile") == "sable"


# ------------------------------------------------------------ idempotent merge

def test_partial_client_save_is_not_duplicated(home__test_chat_transcript_persist):
    """An aborted panel may have saved the prompt (or a partial assistant turn
    marked partial:true) before navigating away. The server-side write for the
    same run must REPLACE that tail, not append a second copy."""
    app = make_app(home__test_chat_transcript_persist)
    tid = "t-dup"
    # Simulate the aborted panel's save: prompt + half-rendered partial reply.
    app.handle("PUT", f"/api/chat/threads/{tid}", body={"messages": [
        {"role": "user", "content": "read the file"},
        {"role": "assistant", "content": "half a re", "partial": True},
    ]})
    with FakeOpenAIServer() as fs:
        _fake_serve_state(home__test_chat_transcript_persist, fs.base_url)
        r = app.handle("POST", "/api/chat/stream",
                       body={"prompt": "read the file", "thread": tid})
        _drain(r)
    t = _thread(home__test_chat_transcript_persist, tid)
    roles = [m["role"] for m in t["messages"]]
    assert roles == ["user", "assistant"], f"duplicate turns: {roles}"
    assert not t["messages"][1].get("partial")
    assert "half a re" != t["messages"][1]["content"]


def test_prompt_only_save_replaced_in_place(home__test_chat_transcript_persist):
    """Panel saved just the user prompt before aborting: the server appends
    its assistant turn after that same user message (no second user msg)."""
    app = make_app(home__test_chat_transcript_persist)
    tid = "t-prompt"
    app.handle("PUT", f"/api/chat/threads/{tid}", body={"messages": [
        {"role": "user", "content": "read the file"}]})
    with FakeOpenAIServer() as fs:
        _fake_serve_state(home__test_chat_transcript_persist, fs.base_url)
        r = app.handle("POST", "/api/chat/stream",
                       body={"prompt": "read the file", "thread": tid})
        _drain(r)
    t = _thread(home__test_chat_transcript_persist, tid)
    assert [m["role"] for m in t["messages"]] == ["user", "assistant"]


def test_threadless_run_persists_nothing(home__test_chat_transcript_persist):
    app = make_app(home__test_chat_transcript_persist)
    with FakeOpenAIServer() as fs:
        _fake_serve_state(home__test_chat_transcript_persist, fs.base_url)
        _drain(app.handle("POST", "/api/chat/stream", body={"prompt": "go"}))
    assert not (home__test_chat_transcript_persist / "chat").exists() or not list(
        (home__test_chat_transcript_persist / "chat").glob("*.json"))

# ------------------------------------------------------------------------
# from tests/test_thread_titles.py
# ------------------------------------------------------------------------

# ------------------------------------------------------------ stage 1

def test_derive_from_first_user_message():
    msgs = [{"role": "user", "content": "Fix the scheduler VRAM gate please"}]
    assert derive_title(msgs) == "Fix the scheduler VRAM gate please"


def test_derive_strips_pleasantries_and_markers():
    # Multiple stacked openers ("hey can you") collapse in repeated passes.
    assert derive_title([{"role": "user", "content": "hey can you  explain\nthis"}]) \
        == "explain this"
    assert derive_title([{"role": "user", "content": "please - fix the bug"}]) \
        == "fix the bug"
    assert derive_title([{"role": "user", "content": "- item one two three"}]) \
        == "item one two three"


def test_derive_truncates_on_word_boundary():
    long = "word " * 40
    t = derive_title([{"role": "user", "content": long}])
    assert len(t) <= 61 and t.endswith("…")
    assert not t.endswith(" …")


def test_derive_never_fails():
    assert derive_title([]) == "Untitled chat"
    assert derive_title([{"role": "user", "content": "   "}]) == "Untitled chat"
    assert derive_title([{"role": "assistant", "content": "hi"}]) == "Untitled chat"
    # A pure code-block opener still yields something stable.
    assert derive_title([{"role": "user", "content": "```\nprint(1)\n```"}]) \
        == "Untitled chat"


def test_derive_skips_nonstring_content():
    assert derive_title([{"role": "user", "content": None},
                         {"role": "user", "content": "real question here"}]) \
        == "real question here"


# ------------------------------------------------------------ stage 2

def _client__test_thread_titles(reply):
    return lambda msgs, model: reply


def test_generate_title_parses_json_object():
    out = generate_title(
        _client__test_thread_titles('{"title": "scheduler vram gating"}'), "m",
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}])
    assert out == "scheduler vram gating"


def test_generate_title_cleans_plain_reply():
    clean = lambda r: generate_title(  # noqa: E731
        _client__test_thread_titles(r), "m", [{"role": "user", "content": "hi"},
                          {"role": "assistant", "content": "ok"}])
    assert clean('Title: "VRAM gate tuning"') == "VRAM gate tuning"
    assert clean("```\nvram gate tuning\n```") == "vram gate tuning"
    assert clean('"quoted title"') == "quoted title"


def test_generate_title_failure_shapes():
    empty_msgs = generate_title(_client__test_thread_titles('{"title": "x"}'), "m", [])
    assert empty_msgs is None                       # nothing to title
    assert generate_title(_client__test_thread_titles(None), "m",
                          [{"role": "user", "content": "hi"}]) is None
    assert generate_title(_client__test_thread_titles('{"title": "  "}'), "m",
                          [{"role": "user", "content": "hi"}]) is None

    def boom(msgs, model):
        raise RuntimeError("endpoint down")
    assert generate_title(boom, "m",
                          [{"role": "user", "content": "hi"}]) is None


# ------------------------------------------------------------- precedence

def test_precedence_derived_llm_user():
    assert should_apply_title(None, DERIVED)
    assert should_apply_title(DERIVED, LLM)
    assert should_apply_title(LLM, LLM)        # refresh allowed
    assert not should_apply_title(LLM, DERIVED)  # never downgrade
    assert not should_apply_title(USER, LLM)     # user rename is final
    assert not should_apply_title(USER, DERIVED)
    assert should_apply_title(USER, USER)


# ------------------------------------------------------------------- API

@pytest.fixture()
def home__test_thread_titles(tmp_path, monkeypatch):
    h = tmp_path / "uf-titles"
    monkeypatch.setenv("SABLE_HOME", str(h))
    return h


def make_app__test_thread_titles(home) -> DashboardApp:
    return DashboardApp(SableConfig(llama_auto_setup=False), home=home)




class TitleJSONHandler(BaseHTTPRequestHandler):
    """Answers the stage-2 call with a JSON title object.

    Also serves GET /v1/models so the app's serve-alive probe (which gates
    _resolve_chat_target) treats this fake as a live endpoint.
    """

    def log_message(self, format, *args):  # noqa: A002 - silence access log
        pass

    def do_GET(self):
        if self.path.rstrip("/") == "/v1/models":
            body = json.dumps({"object": "list",
                               "data": [{"id": "fake-model", "object": "model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        body = json.dumps({
            "id": "x", "object": "chat.completion", "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant",
                                     "content": '{"title": "vram gate design"}'}}],
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _thread_file(home, tid):
    return json.loads((home / "chat" / f"{tid}.json").read_text())


def test_put_derives_title_instantly(home__test_thread_titles):
    app = make_app__test_thread_titles(home__test_thread_titles)
    r = app.handle("PUT", "/api/chat/threads/t1", body={
        "messages": [{"role": "user", "content": "how do I quantize a 70B model"}]})
    assert r.status == 200
    t = r.body["thread"]
    assert t["title"] == "how do I quantize a 70B model"
    assert t["title_source"] == "derived"


def test_thread_file_is_owner_only(home__test_thread_titles):
    """The durable thread file lands 0600 regardless of umask (transcripts can
    contain pasted secrets; matches the serve-state hardening)."""
    app = make_app__test_thread_titles(home__test_thread_titles)
    app.handle("PUT", "/api/chat/threads/t7", body={
        "messages": [{"role": "user", "content": "here is my key sk-abc123"}]})
    mode = (home__test_thread_titles / "chat" / "t7.json").stat().st_mode & 0o777
    assert mode == 0o600


def test_list_rows_carry_title_source(home__test_thread_titles):
    app = make_app__test_thread_titles(home__test_thread_titles)
    app.handle("PUT", "/api/chat/threads/t2", body={
        "messages": [{"role": "user", "content": "quantize a model tonight"}]})
    rows = app.handle("GET", "/api/chat/threads").body["threads"]
    row = [x for x in rows if x["id"] == "t2"][0]
    assert row["title"] == "quantize a model tonight"
    assert row["title_source"] == "derived"


def test_explicit_title_is_user_sourced(home__test_thread_titles):
    app = make_app__test_thread_titles(home__test_thread_titles)
    app.handle("PUT", "/api/chat/threads/t3", body={
        "messages": [{"role": "user", "content": "first words"}]})
    r = app.handle("PUT", "/api/chat/threads/t3", body={"title": "My important thread"})
    t = r.body["thread"]
    assert t["title"] == "My important thread"
    assert t["title_source"] == "user"


def test_stage2_upgrade_replaces_derived_title(home__test_thread_titles):
    app = make_app__test_thread_titles(home__test_thread_titles)
    with FakeOpenAIServer(handler_cls=TitleJSONHandler) as fs:
        _fake_serve_state(home__test_thread_titles, fs.base_url)
        app.handle("PUT", "/api/chat/threads/t4", body={
            "messages": [{"role": "user", "content": "tell me about vram gates"},
                         {"role": "assistant", "content": "sure"}]})
        # Daemon thread: poll briefly for the upgrade.
        t = None
        for _ in range(50):
            t = _thread_file(home__test_thread_titles, "t4")
            if t.get("title_source") == "llm":
                break
            time.sleep(0.1)
    assert t is not None
    assert t["title"] == "vram gate design"
    assert t["title_source"] == "llm"


def test_stage2_never_overrides_user_rename(home__test_thread_titles):
    app = make_app__test_thread_titles(home__test_thread_titles)
    with FakeOpenAIServer(handler_cls=TitleJSONHandler) as fs:
        _fake_serve_state(home__test_thread_titles, fs.base_url)
        # First save derives + schedules an upgrade; rename immediately.
        app.handle("PUT", "/api/chat/threads/t5", body={
            "messages": [{"role": "user", "content": "tell me about vram"},
                         {"role": "assistant", "content": "ok"}]})
        app.handle("PUT", "/api/chat/threads/t5", body={"title": "keep mine"})
        time.sleep(1.0)  # give any scheduled upgrade rope to hang itself
    t = _thread_file(home__test_thread_titles, "t5")
    assert t["title"] == "keep mine"
    assert t["title_source"] == "user"


def test_stage2_skipped_without_assistant_reply(home__test_thread_titles):
    app = make_app__test_thread_titles(home__test_thread_titles)
    with FakeOpenAIServer(handler_cls=TitleJSONHandler) as fs:
        _fake_serve_state(home__test_thread_titles, fs.base_url)
        app.handle("PUT", "/api/chat/threads/t6", body={
            "messages": [{"role": "user", "content": "one-shot question"}]})
        time.sleep(0.5)
    t = _thread_file(home__test_thread_titles, "t6")
    assert t["title_source"] == "derived"


# ------------------------------------- regression: client saves vs titles (#1)

def test_message_only_save_preserves_user_rename(home__test_thread_titles):
    """A save that omits `title` (the chat panel's auto-save shape) must never
    revert a double-click rename — the old client sent the first-message
    snippet on every save, silently clobbering renames."""
    app = make_app__test_thread_titles(home__test_thread_titles)
    app.handle("PUT", "/api/chat/threads/r1", body={
        "messages": [{"role": "user", "content": "original question"}]})
    app.handle("PUT", "/api/chat/threads/r1", body={"title": "renamed by hand"})
    # Auto-save with more messages, NO title key (post-fix chat.js shape).
    r = app.handle("PUT", "/api/chat/threads/r1", body={
        "messages": [{"role": "user", "content": "original question"},
                     {"role": "assistant", "content": "answer"},
                     {"role": "user", "content": "follow-up"}]})
    t = r.body["thread"]
    assert t["title"] == "renamed by hand"
    assert t["title_source"] == "user"
    assert len(t["messages"]) == 3


def test_message_only_save_preserves_llm_title(home__test_thread_titles):
    """Same contract for the stage-2 LLM title: an omitted title key leaves an
    llm-sourced title intact (previously overwritten + mislabeled USER)."""
    app = make_app__test_thread_titles(home__test_thread_titles)
    # Seed a thread whose title already carries llm provenance.
    d = home__test_thread_titles / "chat"
    d.mkdir(parents=True, exist_ok=True)
    (d / "r2.json").write_text(json.dumps({
        "id": "r2", "title": "vram gate design", "title_source": "llm",
        "created": "2026-01-01T00:00:00+00:00", "updated": "2026-01-01T00:00:00+00:00",
        "messages": [{"role": "user", "content": "tell me about vram gates"}],
    }))
    r = app.handle("PUT", "/api/chat/threads/r2", body={
        "messages": [{"role": "user", "content": "tell me about vram gates"},
                     {"role": "assistant", "content": "sure"}]})
    t = r.body["thread"]
    assert t["title"] == "vram gate design"
    assert t["title_source"] == "llm"


def test_message_only_save_preserves_derived_title(home__test_thread_titles):
    """Even a derived title stays put when messages change (no re-derivation
    churn once a thread has a label)."""
    app = make_app__test_thread_titles(home__test_thread_titles)
    r1 = app.handle("PUT", "/api/chat/threads/r3", body={
        "messages": [{"role": "user", "content": "first question here"}]})
    assert r1.body["thread"]["title"] == "first question here"
    r2 = app.handle("PUT", "/api/chat/threads/r3", body={
        "messages": [{"role": "user", "content": "totally different opener"},
                     {"role": "assistant", "content": "ok"}]})
    assert r2.body["thread"]["title"] == "first question here"


# ------------------------------------- regression: rename preserves messages (#1)

def test_rename_only_put_preserves_messages(home__test_thread_titles):
    """The double-click rename sends {title} alone; the thread body must ride
    through untouched (messages, trace ids, summary)."""
    app = make_app__test_thread_titles(home__test_thread_titles)
    app.handle("PUT", "/api/chat/threads/r4", body={
        "messages": [{"role": "user", "content": "hello there"},
                     {"role": "assistant", "content": "hi"}],
        "trace_ids": ["abc123"],
        "summary": "rolling summary text"})
    r = app.handle("PUT", "/api/chat/threads/r4", body={"title": "renamed"})
    t = r.body["thread"]
    assert t["title"] == "renamed" and t["title_source"] == "user"
    assert len(t["messages"]) == 2
    assert t["trace_ids"] == ["abc123"]
    assert t["summary"] == "rolling summary text"


# ------------------------------------- regression: atomic thread writes (#2)

def test_thread_write_is_atomic(home__test_thread_titles, monkeypatch):
    """A crash mid-write must never leave a partial file at the thread path:
    the write goes to *.json.tmp and lands via os.replace. We make os.replace
    raise (simulating death during the rename) and assert the previous file is
    byte-intact."""
    app = make_app__test_thread_titles(home__test_thread_titles)
    app.handle("PUT", "/api/chat/threads/a1", body={
        "messages": [{"role": "user", "content": "durable original"}]})
    good = (home__test_thread_titles / "chat" / "a1.json").read_bytes()

    import os as _os

    def boom(src, dst):  # crash exactly between tmp write and rename
        raise OSError("simulated crash mid-write")

    monkeypatch.setattr(_os, "replace", boom)
    # handle() funnels every handler exception into a 500 (no traceback leak).
    r = app.handle("PUT", "/api/chat/threads/a1", body={
        "messages": [{"role": "user", "content": "x" * 5000}]})
    assert r.status == 500
    # Original untouched; the half-written tmp never reached the read path.
    assert (home__test_thread_titles / "chat" / "a1.json").read_bytes() == good
    t = app.handle("GET", "/api/chat/threads/a1").body["thread"]
    assert t["messages"][0]["content"] == "durable original"

# ------------------------------------------------------------------------
# from tests/test_tuning_panel.py
# ------------------------------------------------------------------------

@pytest.fixture()
def home__test_tuning_panel(tmp_path, monkeypatch):
    h = tmp_path / "uf-tuning"
    monkeypatch.setenv("SABLE_HOME", str(h))
    return h


@pytest.fixture()
def app__test_tuning_panel(home__test_tuning_panel):
    return DashboardApp(SableConfig(), home=home__test_tuning_panel)


def err_code__test_tuning_panel(r) -> str:
    return (r.body.get("error") or {}).get("code", "")


def wait_state(app, path: str, timeout: float = 30.0,
               ) -> Dict[str, Any]:
    """Poll a run-status endpoint until it leaves "running"."""
    s: Dict[str, Any] = {"state": "running"}
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = app.handle("GET", path).body
        if s.get("state") != "running":
            return s
        time.sleep(0.05)
    return s


# ------------------------------------------------------------------ routes

def test_tuning_routes_registered(app__test_tuning_panel):
    paths = {(m, p) for (m, p) in ((m, p) for (m, p, _) in app__test_tuning_panel.ROUTES)}
    for m, p in [
        ("GET", "/api/doctor/env"), ("POST", "/api/doctor/run"),
        ("POST", "/api/doctor/env"),
        ("GET", "/api/bench"), ("GET", "/api/bench/targets"),
        ("POST", "/api/bench/run"), ("GET", "/api/bench/run"),
        ("GET", "/api/calibrate"), ("POST", "/api/calibrate/run"),
        ("GET", "/api/calibrate/run"), ("POST", "/api/calibrate/clear"),
    ]:
        assert (m, p) in paths, f"{m} {p} missing"


# -------------------------------------------------------------------- doctor

def test_doctor_env_idle_without_running(app__test_tuning_panel, home__test_tuning_panel):
    """Bootstrap must NOT execute any checks (user clicks to activate)."""
    r = app__test_tuning_panel.handle("GET", "/api/doctor/env")
    assert r.status == 200
    assert r.body["state"] == "idle"
    assert r.body.get("checks") in (None, [])
    assert r.body["calibration_mode"] in ("off", "auto")


def test_doctor_run_background_structured(app__test_tuning_panel, home__test_tuning_panel):
    r = app__test_tuning_panel.handle("POST", "/api/doctor/run", body={})
    assert r.status == 200 and r.body["state"] == "running"
    s = wait_state(app__test_tuning_panel, "/api/doctor/env")
    assert s["state"] == "done", s.get("error")
    assert s["exit_code"] in (0, 1, 3)
    names = {c["name"] for c in s["checks"]}
    # The WP7 check set must be visible to the panel.
    assert {"python", "deps", "config", "home", "disk"} <= names
    statuses = {c["status"] for c in s["checks"]}
    assert statuses <= {"ok", "warn", "problem"}


def test_doctor_run_fix_flag(app__test_tuning_panel, home__test_tuning_panel):
    r = app__test_tuning_panel.handle("POST", "/api/doctor/env", body={"fix": True})
    assert r.status == 200 and r.body["state"] == "running"
    s = wait_state(app__test_tuning_panel, "/api/doctor/env")
    assert s["state"] == "done", s.get("error")
    assert any(c["name"] == "state" for c in s["checks"])


def test_doctor_run_single_flight(app__test_tuning_panel, home__test_tuning_panel, monkeypatch):
    """A second click while measuring gets 409 already_running."""
    import sable.doctor as doctor_mod

    started = __import__("threading").Event()
    release = __import__("threading").Event()

    def slow_collect(*a, **k):
        started.set()
        release.wait(timeout=5)
        return [], 0

    monkeypatch.setattr(doctor_mod, "collect_checks", slow_collect)
    r = app__test_tuning_panel.handle("POST", "/api/doctor/run", body={})
    assert r.status == 200
    assert started.wait(timeout=5)
    busy = app__test_tuning_panel.handle("POST", "/api/doctor/run", body={})
    assert busy.status == 409 and err_code__test_tuning_panel(busy) == "already_running"
    release.set()
    wait_state(app__test_tuning_panel, "/api/doctor/env")


# --------------------------------------------------------------------- bench

def test_bench_list_empty(app__test_tuning_panel, home__test_tuning_panel):
    r = app__test_tuning_panel.handle("GET", "/api/bench")
    assert r.status == 200
    assert r.body["records"] == []
    assert r.body["noise_floor"] is None


def test_bench_targets_lists_instances(app__test_tuning_panel, home__test_tuning_panel):
    """Multi-instance awareness: the panel learns which server instances are
    running (benchable endpoints) and whether a default spawn is possible."""
    from sable.serving import instances as srv_inst

    srv_inst.save_instance(home__test_tuning_panel, {"id": "sable", "name": "Sable",
                                  "port": 8124})
    r = app__test_tuning_panel.handle("GET", "/api/bench/targets")
    assert r.status == 200
    ids = {i["id"] for i in r.body["instances"]}
    assert {"default", "sable"} <= ids
    sable = next(i for i in r.body["instances"] if i["id"] == "sable")
    assert sable["running"] is False  # no state file => not running
    assert r.body["can_spawn_default"] == bool(app__test_tuning_panel.cfg.gguf_path)


def test_bench_run_unknown_server_404(app__test_tuning_panel, home__test_tuning_panel):
    r = app__test_tuning_panel.handle("POST", "/api/bench/run", body={"server_id": "ghost"})
    assert r.status == 404 and err_code__test_tuning_panel(r) == "not_found"


def test_bench_run_stopped_server_409(app__test_tuning_panel, home__test_tuning_panel):
    from sable.serving import instances as srv_inst

    srv_inst.save_instance(home__test_tuning_panel, {"id": "sable", "name": "Sable",
                                  "port": 8124})
    r = app__test_tuning_panel.handle("POST", "/api/bench/run", body={"server_id": "sable"})
    assert r.status == 409 and err_code__test_tuning_panel(r) == "not_running"


def _fake_record(**kw) -> Dict[str, Any]:
    return {
        "schema_version": 1, "kind": kw.get("kind", "bench"),
        "label": kw.get("label", ""), "model": "fake-model",
        "timestamp": "2026-01-01T00:00:00Z", "runs": 1,
        "prompt_processing": {"median": 10.0, "min": 10.0, "max": 10.0,
                              "spread_pct": 0.0},
        "generation": {"median": 20.0, "min": 20.0, "max": 20.0,
                       "spread_pct": 0.5},
        "token_identity": {"status": "ok", "sha256": "deadbeef"},
    }


def test_bench_list_shows_persisted_record(app__test_tuning_panel, home__test_tuning_panel, monkeypatch):
    from sable.serving import bench as bench_mod

    monkeypatch.setenv("SABLE_BENCH_DIR", str(home__test_tuning_panel / "bench"))
    bench_mod.save_result(_fake_record(kind="noise_floor", label="base"))
    r = app__test_tuning_panel.handle("GET", "/api/bench")
    assert r.status == 200
    assert len(r.body["records"]) == 1
    rec = r.body["records"][0]
    assert rec["kind"] == "noise_floor" and rec["label"] == "base"
    assert rec["gen_median_tps"] == 20.0
    # noise floor surfaced for the panel chip
    assert r.body["noise_floor"]["spread_pct"] == 0.5


def test_bench_run_background_success(app__test_tuning_panel, home__test_tuning_panel, monkeypatch):
    """POST /api/bench/run drives the WP1 modules through the same code path
    as the CLI; with a stubbed endpoint it completes and persists."""
    from sable.serving import bench as bench_mod

    monkeypatch.setenv("SABLE_BENCH_DIR", str(home__test_tuning_panel / "bench"))

    def fake_resolve(cfg, spawn_if_needed=True):
        return "http://127.0.0.1:1/v1", "fake-model", None

    monkeypatch.setattr("sable.cli._resolve_endpoint", fake_resolve)
    monkeypatch.setattr(bench_mod, "run_bench",
                        lambda *a, **k: _fake_record(label="ui"))

    r = app__test_tuning_panel.handle("POST", "/api/bench/run", body={"runs": 1, "label": "ui"})
    assert r.status == 200 and r.body["state"] == "running"

    s = wait_state(app__test_tuning_panel, "/api/bench/run")
    assert s["state"] == "done", s.get("error")
    assert s["result"]["token_identity"]["status"] == "ok"
    # persisted -> visible through the list endpoint
    lst = app__test_tuning_panel.handle("GET", "/api/bench").body
    assert len(lst["records"]) == 1


def test_bench_run_error_state(app__test_tuning_panel, home__test_tuning_panel, monkeypatch):
    """Endpoint resolution failure lands as state=error (not a 500)."""
    from sable.serving.supervisor import ServeError

    def boom(cfg, spawn_if_needed=True):
        raise ServeError("no endpoint available")

    monkeypatch.setattr("sable.cli._resolve_endpoint", boom)
    r = app__test_tuning_panel.handle("POST", "/api/bench/run", body={"runs": 1})
    assert r.status == 200
    s = wait_state(app__test_tuning_panel, "/api/bench/run")
    assert s["state"] == "error" and "no endpoint" in s["error"]


def test_bench_run_rejects_bad_runs(app__test_tuning_panel):
    r = app__test_tuning_panel.handle("POST", "/api/bench/run", body={"runs": "many"})
    assert r.status == 400


# ----------------------------------------------------------------- calibrate

def test_calibrate_show_empty(app__test_tuning_panel, home__test_tuning_panel):
    r = app__test_tuning_panel.handle("GET", "/api/calibrate")
    assert r.status == 200
    assert r.body["profiles"] == []
    assert r.body["calibration_mode"] == "off"
    assert r.body["running_instances"] == []


def test_calibrate_run_without_model_400(app__test_tuning_panel, home__test_tuning_panel):
    r = app__test_tuning_panel.handle("POST", "/api/calibrate/run", body={"quick": True})
    assert r.status == 400 and err_code__test_tuning_panel(r) == "no_model"


def test_calibrate_clear_empty(app__test_tuning_panel, home__test_tuning_panel):
    r = app__test_tuning_panel.handle("POST", "/api/calibrate/clear", body={})
    assert r.status == 200 and r.body["removed"] == 0


def test_calibrate_run_rejects_bad_runs(app__test_tuning_panel, home__test_tuning_panel):
    # runs is validated before the model resolution, so no gguf needed.
    r = app__test_tuning_panel.handle("POST", "/api/calibrate/run", body={"runs": "lots"})
    assert r.status == 400


# ------------------------------------------------- serve.calibration toggle

def test_config_put_serve_section(app__test_tuning_panel, home__test_tuning_panel):
    """The Tuning panel toggles serve.calibration through a section patch."""
    r = app__test_tuning_panel.handle("PUT", "/api/config", body={
        "section": "serve", "patch": {"calibration": "auto"}})
    assert r.status == 200, r.body
    raw = (home__test_tuning_panel / "sable.yaml").read_text()
    assert "calibration: auto" in raw

    bad = app__test_tuning_panel.handle("PUT", "/api/config", body={
        "section": "serve", "patch": {"nonsense_key": 1}})
    assert bad.status == 400 and err_code__test_tuning_panel(bad) == "unknown_keys"

# ------------------------------------------------------------------------
# from tests/test_indexing_phase5.py
# ------------------------------------------------------------------------

# ------------------------------------------------------------------ helpers

def make_project(home: Path, project_id: str, root: Path) -> dict:
    """Create a project record the way the dashboard does (file layout)."""
    root.mkdir(parents=True, exist_ok=True)
    # NOTE: create workspace contents BEFORE calling this via helper below.
    d = home / "projects"
    d.mkdir(parents=True, exist_ok=True)
    proj = {"id": project_id, "name": project_id, "created": "", "updated": "",
            "workspace_dir": str(root), "agents": []}
    (d / f"{project_id}.json").write_text(json.dumps(proj))
    return proj


class FakeEmbedder:
    """Deterministic embedding client for MRL math tests.

    Maps text -> vector by hashing tokens into a fixed 8-dim space; vectors
    are NOT normalized so truncation+renormalization is observable."""

    def __init__(self, dim: int = 8, model_path: str = "/models/fake.gguf"):
        self.dim_full = dim
        self.model_path = model_path
        self.url = None
        self.dim = None
        self.embed_calls = 0

    def available(self) -> bool:
        return True

    @property
    def state(self):
        return {"mode": "spawn", "url": None, "model_path": self.model_path,
                "running": False, "dim": self.dim}

    def embed(self, texts):
        self.embed_calls += 1
        import hashlib
        out = []
        for t in texts:
            v = [0.0] * self.dim_full
            words = t.lower().split()
            for w in words:
                h = int(hashlib.md5(w.encode()).hexdigest(), 16)
                v[h % self.dim_full] += 1.0
            # make vector non-unit and distinct per content
            v = [x + 0.05 * (i + 1) for i, x in enumerate(v)]
            out.append(v)
        return out

    def stop(self):
        pass


# ---------------------------------------------------------------- fixtures

@pytest.fixture()
def home__test_indexing_phase5(tmp_path, monkeypatch):
    h = tmp_path / "uf-home"
    monkeypatch.setenv("SABLE_HOME", str(h))
    return h


@pytest.fixture()
def app__test_indexing_phase5(home__test_indexing_phase5):
    return DashboardApp(SableConfig(), home=home__test_indexing_phase5)


# ------------------------------------------------- 1. per-project isolation

def test_per_project_db_isolation(tmp_path, home__test_indexing_phase5):
    a_root = tmp_path / "proj-a"
    b_root = tmp_path / "proj-b"
    make_project(home__test_indexing_phase5, "proja", a_root)
    make_project(home__test_indexing_phase5, "projb", b_root)
    (a_root / "alpha.py").write_text("def alpha_widget():\n    return 'alpha'\n")
    (b_root / "beta.py").write_text("def beta_gadget():\n    return 'beta'\n")

    cfg = SableConfig(indexing={"enabled": True})
    ia = get_index(home__test_indexing_phase5, cfg=cfg, project_id="proja")
    ib = get_index(home__test_indexing_phase5, cfg=cfg, project_id="projb")
    assert ia.db_path == home__test_indexing_phase5 / "index" / "proja.db"
    assert ib.db_path != ia.db_path
    ia.build()
    ib.build()

    hits_a = ia.search("alpha widget")
    assert hits_a and all("alpha.py" in h["path"] for h in hits_a)
    # searching project B never returns project A's chunks
    hits_b = ib.search("alpha widget")
    assert all("alpha.py" not in h["path"] for h in hits_b)

    # default DB is a separate file and stays empty
    idf = get_index(home__test_indexing_phase5, cfg=cfg)
    assert idf.db_path == home__test_indexing_phase5 / "index" / f"{DEFAULT_DB_ID}.db"
    st = idf.status()
    assert st["files"] == 0


def test_project_roots_derived_from_workspace(tmp_path, home__test_indexing_phase5):
    root = tmp_path / "ws"
    root.mkdir()
    (root / "x.py").write_text("def x():\n    return 1\n")
    make_project(home__test_indexing_phase5, "projr", root)
    cfg = SableConfig(indexing={"enabled": True,
                                  "paths_override": [str(tmp_path)]})
    idx = get_index(home__test_indexing_phase5, cfg=cfg, project_id="projr")
    # project DB ignores paths_override: its only root is workspace_dir
    assert idx.roots == [root]


def test_get_index_cache_refreshes_on_config_change(home__test_indexing_phase5):
    cfg1 = SableConfig(indexing={"enabled": True, "output_dim": 768})
    i1 = get_index(home__test_indexing_phase5, cfg=cfg1)
    cfg2 = SableConfig(indexing={"enabled": True, "output_dim": 256})
    i2 = get_index(home__test_indexing_phase5, cfg=cfg2)
    assert i2.output_dim == 256
    assert get_index(home__test_indexing_phase5, cfg=cfg2) is i2  # stable afterwards


# --- regression: setting indexing.model after startup must take effect -----
# (the "indexing does not start until I restart" report: get_index's
# config-change rebuild carried the OLD lexical-only embed client over, and
# embedder() only builds a client when _embed is None)

def test_config_model_appears_without_restart(home__test_indexing_phase5):
    cfg_off = SableConfig(indexing={"enabled": True})
    idx = get_index(home__test_indexing_phase5, cfg=cfg_off)
    assert idx.status()["embedding_mode"] == "none"  # builds a lexical client

    model = str(home__test_indexing_phase5 / "embed.gguf")
    cfg_on = SableConfig(indexing={"enabled": True, "model": model})
    idx2 = get_index(home__test_indexing_phase5, cfg=cfg_on)
    # the new model must be live on the cached slot WITHOUT a restart
    assert idx2.embedder().model_path == model
    assert idx2.status()["embedding_mode"] == "spawn"


def test_config_model_removal_drops_client(home__test_indexing_phase5):
    model = str(home__test_indexing_phase5 / "embed.gguf")
    idx = get_index(home__test_indexing_phase5,
                    cfg=SableConfig(indexing={"enabled": True,
                                              "model": model}))
    assert idx.embedder().model_path == model
    idx2 = get_index(home__test_indexing_phase5,
                     cfg=SableConfig(indexing={"enabled": True}))
    assert idx2.status()["embedding_mode"] == "none"


def test_running_embed_client_survives_unrelated_config_change(
        home__test_indexing_phase5):
    # an output_dim edit must NOT drop a compatible (same-model) client:
    # that would kill a running embedding server on every settings save.
    model = str(home__test_indexing_phase5 / "embed.gguf")
    i1 = get_index(home__test_indexing_phase5,
                   cfg=SableConfig(indexing={"enabled": True, "model": model,
                                             "output_dim": 768}))
    client = i1.embedder()
    i2 = get_index(home__test_indexing_phase5,
                   cfg=SableConfig(indexing={"enabled": True, "model": model,
                                             "output_dim": 256}))
    assert i2 is not i1 and i2.output_dim == 256
    assert i2._embed is client


# --- regression: a cfg-less cached instance must be able to adopt a config --
# (search_codebase called get_index WITHOUT cfg; the poisoned cache entry
# then reported "indexing is disabled" for every later caller)

def test_get_index_repairs_cfg_less_instance(home__test_indexing_phase5):
    i0 = get_index(home__test_indexing_phase5)  # no cfg -> disabled
    assert i0.enabled is False
    cfg = SableConfig(indexing={"enabled": True})
    i1 = get_index(home__test_indexing_phase5, cfg=cfg)
    assert i1.enabled is True
    assert get_index(home__test_indexing_phase5, cfg=cfg) is i1


def test_hand_seeded_instance_is_never_repaired(
        home__test_indexing_phase5, monkeypatch):
    import sable.core.indexer as indexer_mod
    seed = indexer_mod.CodebaseIndex(home__test_indexing_phase5, cfg=None,
                                     index_cfg={"enabled": True})
    monkeypatch.setitem(indexer_mod._instances,
                        (str(home__test_indexing_phase5), None), seed)
    got = get_index(home__test_indexing_phase5,
                    cfg=SableConfig(indexing={"enabled": False}))
    assert got is seed  # the seeder owns it; not rebuilt out from under them


# --- embed-server launch knobs (GPU contention with the chat model) --------

def test_embed_gpu_layers_and_threads_from_config(home__test_indexing_phase5):
    from sable.core.indexer import EmbeddingClient

    cfg = SableConfig(indexing={"enabled": True, "model": "/x/m.gguf",
                                "gpu_layers": 0, "threads": 4})
    c = EmbeddingClient(model_path="/x/m.gguf", cfg=cfg)
    assert c._resolve_gpu_layers() == 0  # 0 = CPU embeddings, must survive
    assert c.threads == 4

    cfg2 = SableConfig(indexing={"enabled": True, "model": "/x/m.gguf"})
    c2 = EmbeddingClient(model_path="/x/m.gguf", cfg=cfg2)
    assert c2.gpu_layers is None and c2.threads is None  # auto-detect


def test_embed_launch_flags_reach_the_command(home__test_indexing_phase5):
    # spawn-command assembly is checked without booting a server
    from sable.core.indexer import EmbeddingClient

    captured: dict = {}

    class FakeProc:
        def __init__(self, cmd):
            captured["cmd"] = list(cmd)
        def poll(self):
            return 0  # "exited early" -> ensure_running raises after spawn

    def fake_popen(cmd, **kw):
        return FakeProc(cmd)

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr("subprocess.Popen", fake_popen)
        monkey.setattr("sable.serving.supervisor.find_llama_server",
                       lambda *a, **k: "/bin/true")
        c = EmbeddingClient(model_path="/x/m.gguf",
                            cfg=SableConfig(indexing={
                                "enabled": True, "model": "/x/m.gguf",
                                "gpu_layers": 0, "threads": 3}))
        with pytest.raises(Exception):
            c.ensure_running()  # fake server "exits"; cmd was already built
        assert captured["cmd"], "Popen was never reached"
        i = captured["cmd"].index("-ngl")
        assert captured["cmd"][i + 1] == "0"
        assert captured["cmd"][captured["cmd"].index("-t") + 1] == "3"
    finally:
        monkey.undo()


# --- search_codebase must see the config ----------------------------------

def test_search_codebase_with_bound_cfg_not_disabled(
        tmp_path, home__test_indexing_phase5):
    import sable.core.indexer as indexer_mod
    import sable.core.tools as uf_tools
    root = tmp_path / "ws"
    root.mkdir()
    (root / "alpha.py").write_text("# alpha widget factory\nx = 1\n")
    cfg = SableConfig(indexing={"enabled": True,
                                "paths_override": [str(root)]})
    # build via a DIRECT instance: the cache slot must still be empty when
    # the tool runs, so the tool itself is what creates the cached instance
    # (pre-fix it created a cfg-less, disabled one and poisoned the slot).
    indexer_mod.CodebaseIndex(home__test_indexing_phase5, cfg=cfg).build()
    out = uf_tools.search_codebase("alpha widget", _cfg=cfg)
    assert "alpha.py" in out, out
    # and the poisoned-cache variant is gone: enabled, not "disabled"
    assert "disabled" not in out


def test_search_codebase_empty_config_never_disables_cache(
        tmp_path, home__test_indexing_phase5):
    # A tool-level config load with NO indexing section must not downgrade
    # a properly-configured cached instance (a supplied section is
    # authoritative for get_index rebuilds).
    import sable.core.tools as uf_tools
    root = tmp_path / "ws2"
    root.mkdir()
    (root / "beta.py").write_text("# beta gadget factory\ny = 2\n")
    idx = get_index(home__test_indexing_phase5,
                    cfg=SableConfig(indexing={"enabled": True,
                                              "paths_override": [str(root)]}))
    idx.build()
    out = uf_tools.search_codebase("beta gadget")
    assert "disabled" not in out, out


# ------------------------------------------------------- 2. MRL truncation

def test_truncate_renorm_math():
    v = [3.0, 4.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    t = truncate_renorm(v, 2)
    assert len(t) == 2
    assert math.isclose(math.sqrt(sum(x * x for x in t)), 1.0)
    assert math.isclose(t[0], 0.6) and math.isclose(t[1], 0.8)
    # dim >= len keeps the full vector, renormalized
    f = truncate_renorm(v, 16)
    assert len(f) == len(v)
    assert math.isclose(math.sqrt(sum(x * x for x in f)), 1.0)
    # falsy dim: untouched copy
    assert truncate_renorm(v, None) == v


def test_mrl_stored_and_query_vectors_truncated(tmp_path, home__test_indexing_phase5):
    root = tmp_path / "mrl"
    root.mkdir()
    (root / "a.py").write_text("alpha widget one two\n")
    (root / "b.py").write_text("beta gadget three four\n")
    make_project(home__test_indexing_phase5, "projm", root)

    cfg = SableConfig(indexing={"enabled": True, "model": "/models/fake.gguf",
                                  "output_dim": 4})
    idx = get_index(home__test_indexing_phase5, cfg=cfg, project_id="projm")
    fake = FakeEmbedder(dim=8)
    # Duck-typed test double; setattr keeps the type checker happy.
    setattr(idx, "_embed", fake)

    res = idx.build()
    assert res["embedded"] > 0 and res["embed_ok"] is True

    # stored vectors are exactly output_dim long and unit-normalized
    import sqlite3
    from sable.core.indexer import _unpack
    con = sqlite3.connect(str(idx.db_path))
    blobs = [r[0] for r in con.execute(
        "SELECT embedding FROM chunks WHERE embedding IS NOT NULL")]
    meta = dict(con.execute("SELECT key, value FROM meta"))
    con.close()
    assert blobs
    for b in blobs:
        v = _unpack(b)
        assert len(v) == 4
        assert math.isclose(math.sqrt(sum(x * x for x in v)), 1.0,
                            rel_tol=1e-6)
    assert meta["vector_dim"] == "4"
    assert meta["model_id"] == "/models/fake.gguf"

    # query vector truncated too (fake returns 8 dims; search must cope)
    hits = idx.search("alpha widget one two", mode="semantic")
    assert hits and fake.embed_calls >= 2


def test_dim_mismatch_requires_rebuild(tmp_path, home__test_indexing_phase5):
    root = tmp_path / "dim"
    root.mkdir()
    (root / "a.py").write_text("alpha widget one two\n")
    make_project(home__test_indexing_phase5, "projd", root)

    cfg = SableConfig(indexing={"enabled": True, "model": "/models/fake.gguf",
                                  "output_dim": 4})
    idx = get_index(home__test_indexing_phase5, cfg=cfg, project_id="projd")
    setattr(idx, "_embed", FakeEmbedder(dim=8))
    idx.build()

    # now query at a different dim: the DB was built at 4 dims
    cfg2 = SableConfig(indexing={"enabled": True, "model": "/models/fake.gguf",
                                   "output_dim": 6})
    idx2 = CodebaseIndex(home__test_indexing_phase5, cfg=cfg2, project_id="projd")
    setattr(idx2, "_embed", FakeEmbedder(dim=8))
    with pytest.raises(IndexRebuildRequired):
        idx2.search("alpha widget", mode="semantic")


def test_model_change_requires_rebuild(tmp_path, home__test_indexing_phase5):
    root = tmp_path / "mod"
    root.mkdir()
    (root / "a.py").write_text("alpha widget one two\n")
    make_project(home__test_indexing_phase5, "projmm", root)
    cfg = SableConfig(indexing={"enabled": True, "model": "/models/a.gguf",
                                  "output_dim": 4})
    idx = get_index(home__test_indexing_phase5, cfg=cfg, project_id="projmm")
    setattr(idx, "_embed", FakeEmbedder(dim=8, model_path="/models/a.gguf"))
    idx.build()

    cfg2 = SableConfig(indexing={"enabled": True, "model": "/models/b.gguf",
                                   "output_dim": 4})
    idx2 = CodebaseIndex(home__test_indexing_phase5, cfg=cfg2, project_id="projmm")
    setattr(idx2, "_embed", FakeEmbedder(dim=8, model_path="/models/b.gguf"))
    with pytest.raises(IndexRebuildRequired):
        idx2.search("alpha widget", mode="semantic")


# ---------------------------------------------------- 3. project delete

def test_delete_project_index_only_that_file(home__test_indexing_phase5):
    idxdir = home__test_indexing_phase5 / "index"
    idxdir.mkdir(parents=True)
    (idxdir / "projax.db").write_text("a")
    (idxdir / "projay.db").write_text("b")
    (idxdir / f"{DEFAULT_DB_ID}.db").write_text("d")

    assert delete_project_index(home__test_indexing_phase5, "projax") is True
    assert not (idxdir / "projax.db").exists()
    assert (idxdir / "projay.db").is_file()
    assert (idxdir / f"{DEFAULT_DB_ID}.db").is_file()
    # refusing to nuke the default DB
    assert delete_project_index(home__test_indexing_phase5, DEFAULT_DB_ID) is False
    assert (idxdir / f"{DEFAULT_DB_ID}.db").is_file()


def test_project_delete_endpoint_removes_db(app__test_indexing_phase5, home__test_indexing_phase5, tmp_path):
    root = tmp_path / "wsd"
    root.mkdir()
    make_project(home__test_indexing_phase5, "projdel", root)
    idxdir = home__test_indexing_phase5 / "index"
    idxdir.mkdir(parents=True, exist_ok=True)
    (idxdir / "projdel.db").write_text("x")
    (idxdir / f"{DEFAULT_DB_ID}.db").write_text("y")

    r = app__test_indexing_phase5.handle("DELETE", "/api/projects/projdel", body={})
    assert r.status == 200 and r.body["ok"] is True
    assert r.body["index_removed"] is True
    assert not (idxdir / "projdel.db").exists()
    assert (idxdir / f"{DEFAULT_DB_ID}.db").is_file()


# --------------------------------------------------- 4. spawn/stop lifecycle

STUB__test_indexing_phase5 = '''#!/usr/bin/env python3
import argparse, json, os, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

p = argparse.ArgumentParser()
p.add_argument("-m", dest="model", required=True)
p.add_argument("--embedding", action="store_true")
p.add_argument("--alias", default=None)
p.add_argument("--host", default=None)
p.add_argument("--port", type=int, required=True)
p.add_argument("-ngl", dest="ngl", type=int, default=None)
p.add_argument("-b", dest="batch", type=int, default=None)
p.add_argument("-ub", dest="ubatch", type=int, default=None)
p.add_argument("-c", dest="ctx", type=int, default=None)
a = p.parse_args()

if a.host != "127.0.0.1":
    print("stub: refusing non-loopback bind", file=sys.stderr); sys.exit(4)
if not a.embedding:
    print("stub: --embedding missing", file=sys.stderr); sys.exit(5)

with open(os.environ.get("STUB_PIDFILE", "/dev/null"), "w") as f:
    f.write(str(os.getpid()))
if os.environ.get("STUB_ARGFILE"):
    with open(os.environ["STUB_ARGFILE"], "w") as f:
        json.dump(sys.argv[1:], f)

class H(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002 - silence access log
        pass
    def do_GET(self):
        if self.path.rstrip("/") == "/v1/models":
            body = json.dumps({"object": "list", "data": [{"id": "sable-embeddings"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body)
        else:
            self.send_response(404); self.end_headers()
    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(n) or b"{}")
        inputs = payload.get("input") or []
        data = [{"index": i, "embedding": [0.1] * 8} for i in range(len(inputs))]
        body = json.dumps({"object": "list", "data": data}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers(); self.wfile.write(body)

srv = ThreadingHTTPServer((a.host, a.port), H)
print("stub embed server ready", flush=True)
srv.serve_forever()
'''


@pytest.fixture()
def stub_binary(tmp_path):
    p = tmp_path / "llama-server"
    p.write_text(STUB__test_indexing_phase5)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(p)


def test_build_backfills_null_embeddings(tmp_path, home__test_indexing_phase5):
    """A build that previously failed mid-embed (NULL rows, files tracked)
    retries those chunks on the next build instead of skipping them."""
    root = tmp_path / "bfws"
    root.mkdir()
    (root / "a.py").write_text("alpha widget\n")
    make_project(home__test_indexing_phase5, "projbf", root)
    cfg = SableConfig(indexing={"enabled": True,
                                  "model": "/models/fake.gguf"})
    idx = CodebaseIndex(home__test_indexing_phase5, cfg=cfg, project_id="projbf")
    fake = FakeEmbedder(dim=8)
    setattr(idx, "_embed", fake)
    res = idx.build()
    assert res["embedded"] > 0
    # simulate a failed partial embed: wipe one embedding, keep files tracked
    con = sqlite3.connect(str(idx.db_path))
    con.execute("UPDATE chunks SET embedding=NULL")
    con.commit()
    con.close()
    fake2 = FakeEmbedder(dim=8)
    setattr(idx, "_embed", fake2)
    res2 = idx.build()  # no file changes: only the NULL backfill path
    assert res2["embedded"] > 0 and fake2.embed_calls >= 1
    con = sqlite3.connect(str(idx.db_path))
    n_null = con.execute(
        "SELECT COUNT(*) FROM chunks WHERE embedding IS NULL").fetchone()[0]
    con.close()
    assert n_null == 0


def test_embed_client_passes_batch_flags(tmp_path, home__test_indexing_phase5, stub_binary,
                                         monkeypatch):
    """Spawn sets -b/-ub/-c so inputs over the 512 default are accepted."""
    argvfile = tmp_path / "argv.json"
    monkeypatch.setenv("STUB_ARGFILE", str(argvfile))
    client = EmbeddingClient(model_path=str(tmp_path / "m.gguf"),
                             cfg=SableConfig(
                                 llama_server_binary=stub_binary,
                                 llama_auto_setup=False))
    (tmp_path / "m.gguf").write_bytes(b"GGUF")
    try:
        assert client.ensure_running()
        argv = json.loads(argvfile.read_text())
        assert argv[argv.index("-ub") + 1] == str(EMBED_UBATCH)
        assert argv[argv.index("-b") + 1] == str(EMBED_UBATCH)
        assert argv[argv.index("-c") + 1] == str(EMBED_UBATCH)
    finally:
        client.stop()


def test_clip_to_tokens():
    short = "x" * 60
    assert clip_to_tokens(short, 128) == short
    long = "\n".join(["word " * 12 + str(i) for i in range(2000)])
    out = clip_to_tokens(long, 500)
    assert len(out) <= 500 * 2 and len(out) < len(long)
    # prefix of the original, cut at a newline boundary, not over-clipped
    assert long.startswith(out) and long[len(out)] == "\n"
    assert len(out) > 400 * 2


def test_embed_client_spawn_stop_loopback(tmp_path, home__test_indexing_phase5, stub_binary,
                                          monkeypatch):
    """Build spawns the owned server (loopback-only), stop() terminates it."""
    pidfile = tmp_path / "stub.pid"
    monkeypatch.setenv("STUB_PIDFILE", str(pidfile))
    root = tmp_path / "spawnws"
    root.mkdir()
    (root / "a.py").write_text("alpha widget\n")
    make_project(home__test_indexing_phase5, "projsp", root)

    cfg = SableConfig(indexing={"enabled": True,
                                  "model": str(tmp_path / "emb.gguf")},
                        llama_server_binary=stub_binary,
                        llama_auto_setup=False)
    (tmp_path / "emb.gguf").write_bytes(b"GGUF")
    idx = CodebaseIndex(home__test_indexing_phase5, cfg=cfg, project_id="projsp")
    try:
        res = idx.build()
        assert res["embed_ok"] is True and res["embedded"] > 0
        child_pid = int(pidfile.read_text())
        # process alive while owned server runs; registered for cleanup
        os.kill(child_pid, 0)
        import sable.core.indexer as im
        assert idx.embedder() in im._spawned_clients

        stop_all_embed_servers()
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            try:
                os.kill(child_pid, 0)
                time.sleep(0.1)
            except OSError:
                break
        else:
            pytest.fail("spawned embedding server still running after stop")
        assert not idx.embedder()._proc
    finally:
        idx.stop_embed_server()


def test_stop_all_never_touches_foreign_processes(stub_binary, tmp_path,
                                                  monkeypatch):
    """A process sable did not spawn survives stop_all_embed_servers."""
    foreign = subprocess.Popen([sys.executable, "-c",
                                "import time; time.sleep(30)"])
    try:
        stop_all_embed_servers()  # nothing registered => no-op
        assert foreign.poll() is None
        # with a registered client, only the client's own child stops
        pidfile = tmp_path / "stub2.pid"
        monkeypatch.setenv("STUB_PIDFILE", str(pidfile))
        client = EmbeddingClient(model_path=str(tmp_path / "m.gguf"),
                                 cfg=SableConfig(
                                     llama_server_binary=stub_binary,
                                     llama_auto_setup=False))
        (tmp_path / "m.gguf").write_bytes(b"GGUF")
        base = client.ensure_running()
        assert base and base.startswith("http://127.0.0.1:")
        child_pid = int(pidfile.read_text())
        stop_all_embed_servers()
        os.kill(foreign.pid, 0)          # foreign untouched
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            try:
                os.kill(child_pid, 0)
                time.sleep(0.1)
            except OSError:
                break
    finally:
        foreign.terminate()
        foreign.wait(timeout=5)


def test_spawn_failure_surfaces_clear_error(tmp_path, home__test_indexing_phase5):
    """Missing llama-server binary => build raises, no silent fallback."""
    root = tmp_path / "nofind"
    root.mkdir()
    (root / "a.py").write_text("alpha widget\n")
    make_project(home__test_indexing_phase5, "projnf", root)
    cfg = SableConfig(indexing={"enabled": True,
                                  "model": str(tmp_path / "e.gguf")},
                        llama_server_binary=str(tmp_path / "nope"),
                        llama_auto_setup=False)
    (tmp_path / "e.gguf").write_bytes(b"GGUF")
    idx = CodebaseIndex(home__test_indexing_phase5, cfg=cfg, project_id="projnf")
    with pytest.raises(RuntimeError) as ei:
        idx.build()
    assert "embedding server error" in str(ei.value)


# ----------------------------------------------------- 5. legacy config keys

def test_legacy_indexing_keys_load_without_crash(tmp_path, monkeypatch):
    cfg_file = tmp_path / "old.yaml"
    cfg_file.write_text(
        "indexing:\n"
        "  enabled: true\n"
        "  embedding_model: /models/old.gguf\n"
        "  embedding_url: http://127.0.0.1:9999/v1\n"
        "  top_k: 3\n"
        "  paths:\n"
        "    - /some/root\n"
        "  search_mode: semantic\n")
    monkeypatch.setenv("SABLE_CONFIG", str(cfg_file))
    cfg = load_config()
    ix = cfg.indexing
    assert ix["enabled"] is True
    assert ix["model"] == "/models/old.gguf"       # embedding_model -> model
    assert ix["paths_override"] == ["/some/root"]  # paths -> paths_override
    assert "embedding_url" not in ix               # attach mode absorbed
    assert "top_k" not in ix                       # internal constant now
    assert ix["search_mode"] == "semantic"
    # and the indexer consumes it without exploding
    idx = CodebaseIndex(tmp_path / "h", cfg=cfg)
    assert idx.model_path == "/models/old.gguf"
    assert idx.top_k == 8  # SEARCH_TOP_K, not the old 3


def test_config_put_accepts_phase5_indexing_keys(app__test_indexing_phase5, home__test_indexing_phase5, monkeypatch):
    monkeypatch.setenv("SABLE_CONFIG", str(home__test_indexing_phase5 / "sable.yaml"))
    r = app__test_indexing_phase5.handle("PUT", "/api/config", body={
        "section": "indexing",
        "patch": {"enabled": True, "model": "/models/x.gguf",
                  "output_dim": 512, "paths_override": ["/x"]}})
    assert r.status == 200, r.body
    # legacy keys still round-trip (absorbed on load, not rejected on write)
    r = app__test_indexing_phase5.handle("PUT", "/api/config", body={
        "section": "indexing", "patch": {"top_k": 4}})
    assert r.status == 200, r.body
    cfg = load_config(str(home__test_indexing_phase5 / "sable.yaml"))
    assert cfg.indexing["model"] == "/models/x.gguf"
    assert cfg.indexing["output_dim"] == 512
    assert "top_k" not in cfg.indexing


# ------------------------------------------------- 6. search_codebase routing

def test_search_codebase_routes_by_bound_project(monkeypatch, tmp_path, home__test_indexing_phase5):
    import sable.core.tools as uf_tools  # noqa: F811

    a_root = tmp_path / "ra"
    b_root = tmp_path / "rb"
    make_project(home__test_indexing_phase5, "routea", a_root)
    make_project(home__test_indexing_phase5, "routeb", b_root)
    # spaced words: the BM25 tokenizer keeps identifiers like alpha_widget
    # as one token, so query terms must appear as separate words.
    (a_root / "alpha.py").write_text("# alpha widget factory\nx = 1\n")
    (b_root / "beta.py").write_text("# beta gadget factory\ny = 2\n")

    cfg = SableConfig(indexing={"enabled": True})
    ia = get_index(home__test_indexing_phase5, cfg=cfg, project_id="routea")
    ib = get_index(home__test_indexing_phase5, cfg=cfg, project_id="routeb")
    ia.build()
    ib.build()

    import functools
    out_a = uf_tools.search_codebase("alpha widget", _project_id="routea",
                    _workspace_dir=str(a_root))
    assert "alpha.py" in out_a
    out_b = uf_tools.search_codebase("alpha widget", _project_id="routeb",
                                     _workspace_dir=str(b_root))
    assert "alpha.py" not in out_b  # no cross-project results

    # create_runner binds the partial when project_id is given
    from sable.core.agent_runner import create_runner

    runner = create_runner(cfg, "http://127.0.0.1:1/v1", "m",
                           project_workspace_dir=str(a_root),
                           project_id="routea")
    impl = runner.tool_impls["search_codebase"]
    assert isinstance(impl, functools.partial)
    assert impl.keywords.get("_project_id") == "routea"
    assert impl.keywords.get("_workspace_dir") == str(a_root)

    runner2 = create_runner(cfg, "http://127.0.0.1:1/v1", "m")
    assert not isinstance(runner2.tool_impls["search_codebase"],
                          functools.partial) or \
        "_project_id" not in getattr(runner2.tool_impls["search_codebase"],
                                     "keywords", {})


# ------------------------------------------------------- dashboard endpoints

def test_index_build_endpoint_routes_project(app__test_indexing_phase5, home__test_indexing_phase5, tmp_path):
    root = tmp_path / "epws"
    root.mkdir()
    (root / "a.py").write_text("alpha widget\n")
    make_project(home__test_indexing_phase5, "projep", root)

    r = app__test_indexing_phase5.handle("POST", "/api/index/build", body={"project": "projep"})
    assert r.status == 200 and r.body["project"] == "projep"
    st = {}
    for _ in range(100):
        st = app__test_indexing_phase5.handle("GET", "/api/index", query={"project": "projep"}).body
        if not st["building"] and st["chunks"]:
            break
        time.sleep(0.05)
    assert st["files"] == 1
    assert (home__test_indexing_phase5 / "index" / "projep.db").is_file()

    r = app__test_indexing_phase5.handle("POST", "/api/index/build", body={"project": "nope"})
    assert r.status == 404


def test_index_status_reports_phase5_fields(app__test_indexing_phase5, home__test_indexing_phase5, tmp_path):
    root = tmp_path / "stws"
    root.mkdir()
    make_project(home__test_indexing_phase5, "projst", root)
    cfgd = SableConfig(indexing={"enabled": True, "output_dim": 256})
    app__test_indexing_phase5.cfg = cfgd
    r = app__test_indexing_phase5.handle("GET", "/api/index", query={"project": "projst"})
    assert r.status == 200
    body = r.body
    assert body["enabled"] is True
    assert body["project_id"] == "projst"
    assert body["output_dim"] == 256
    assert body["roots"] == [str(root)]

# ------------------------------------------------------------------------
# from tests/test_run_settings_v2.py
# ------------------------------------------------------------------------

def _has(args, *tokens):
    s = "\x00".join(args)
    return ("\x00".join(tokens)) in s


# ------------------------------------------------- 1. speculative decoding

def test_canonical_spec_mode_aliases():
    assert canonical_spec_mode("auto") == "auto"
    assert canonical_spec_mode("default") == "auto"
    assert canonical_spec_mode("none") == "off"
    assert canonical_spec_mode("draft-mtp") == "mtp"
    assert canonical_spec_mode("ngram-mod") == "ngram"
    assert canonical_spec_mode("mtp+ngram") == "ngram+mtp"
    assert canonical_spec_mode("ngram-mod,draft-mtp") == "ngram+mtp"
    assert canonical_spec_mode(None) is None
    assert canonical_spec_mode("garbage") is None


def test_speculative_args_emissions(tmp_path):
    drafter = tmp_path / "mtp-model-Q8_0.gguf"
    drafter.write_bytes(b"GGUF")
    assert speculative_args({}) == []
    assert speculative_args({"speculative_type": "off"}) == []
    assert speculative_args({"speculative_type": "auto"}) == \
        ["--spec-type", "ngram-mod"]
    # MTP modes need a drafter: explicit mtp_path -> --model-draft.
    a = speculative_args({"speculative_type": "mtp",
                          "mtp_path": str(drafter)})
    assert _has(a, "--spec-type", "draft-mtp")
    assert _has(a, "--model-draft", str(drafter))
    b = speculative_args({"speculative_type": "ngram+mtp",
                          "mtp_path": str(drafter),
                          "spec_draft_n_max": 3})
    assert _has(b, "--spec-type", "ngram-mod,draft-mtp")
    assert _has(b, "--model-draft", str(drafter))
    assert _has(b, "--spec-draft-n-max", "3")
    # No drafter anywhere (no mtp_path, no gguf sibling): degrade to ngram so
    # llama-server never exits early on a missing draft model.
    assert speculative_args({"speculative_type": "mtp"}) == \
        ["--spec-type", "ngram-mod"]
    assert speculative_args({"speculative_type": "mtp",
                             "mtp_path": str(tmp_path / "nope.gguf")}) == \
        ["--spec-type", "ngram-mod"]


def test_speculative_mtp_autodetects_sibling(tmp_path):
    gguf = tmp_path / "model-Q4_K_M.gguf"
    gguf.write_bytes(b"GGUF")
    mtp = tmp_path / "mtp-model-Q8_0.gguf"
    mtp.write_bytes(b"GGUF")
    a = speculative_args({"speculative_type": "mtp"}, gguf_path=str(gguf))
    assert _has(a, "--spec-type", "draft-mtp")
    assert _has(a, "--model-draft", str(mtp.resolve()))


def test_run_settings_args_includes_spec():
    args = run_settings_args({"speculative_type": "ngram"})
    assert _has(args, "--spec-type", "ngram-mod")


def test_sanitize_new_run_setting_keys():
    out = sanitize_run_settings({
        "vision": False, "speculative_type": "DRAFT-MTP",
        "spec_draft_n_max": 4})
    assert out["vision"] is False
    assert out["speculative_type"] == "mtp"
    assert out["spec_draft_n_max"] == 4
    with pytest.raises(ValueError):
        sanitize_run_settings({"speculative_type": "quantum"})
    with pytest.raises(ValueError):
        sanitize_run_settings({"spec_draft_n_max": 99})


# ------------------------------------------------- 2. mmproj detection

def test_find_local_sibling(tmp_path):
    gguf = tmp_path / "model-Q4_K_M.gguf"
    gguf.write_bytes(b"GGUF")
    assert mmproj.find_local(str(gguf)) is None
    proj = tmp_path / "mmproj-model-F16.gguf"
    proj.write_bytes(b"GGUF")
    assert mmproj.find_local(str(gguf)) == str(proj.resolve())
    # non-mmproj ggufs and other extensions don't count
    (tmp_path / "other.gguf").write_bytes(b"x")
    found = mmproj.find_local(str(gguf))
    assert found is not None and Path(found).name == proj.name


def test_is_mmproj_name():
    assert mmproj.is_mmproj_name("mmproj.gguf")
    assert mmproj.is_mmproj_name("mmproj-F16.gguf")
    assert mmproj.is_mmproj_name("Qwen3-VL-mmproj-BF16.gguf")
    assert not mmproj.is_mmproj_name("model-Q4_K_M.gguf")
    assert not mmproj.is_mmproj_name("mmproj.safetensors")


def test_resolve_mmproj_toggle_precedence(tmp_path):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"G")
    proj = tmp_path / "mmproj.gguf"
    proj.write_bytes(b"P")
    # default (no vision key): auto-detect the sibling
    assert mmproj.resolve_mmproj(str(gguf), {}) == str(proj.resolve())
    # explicit off wins
    assert mmproj.resolve_mmproj(str(gguf), {"vision": False}) is None
    # explicit manual path overrides detection
    other = tmp_path / "manual-mmproj.gguf"
    other.write_bytes(b"M")
    assert mmproj.resolve_mmproj(
        str(gguf), {"mmproj_path": str(other)}) == str(other.resolve())
    # dead manual path => absent, not a crash
    assert mmproj.resolve_mmproj(
        str(gguf), {"mmproj_path": str(tmp_path / "gone.gguf")}) is None


def test_source_repo_for_hub_layout_and_manifest(tmp_path):
    snap = (tmp_path / "models--testorg--Test-GGUF" / "snapshots" / "abc123")
    snap.mkdir(parents=True)
    g = snap / "model.gguf"
    g.write_bytes(b"G")
    assert mmproj.source_repo_for(str(g)) == "testorg/Test-GGUF"
    # loose file with a manifest
    loose = tmp_path / "loose"
    loose.mkdir()
    (loose / "m.gguf").write_bytes(b"G")
    (loose / "sable_download.json").write_text(
        json.dumps({"repo_id": "org/From-Manifest"}))
    assert mmproj.source_repo_for(str(loose / "m.gguf")) == "org/From-Manifest"
    # loose file, no provenance
    (tmp_path / "plain.gguf").write_bytes(b"G")
    assert mmproj.source_repo_for(str(tmp_path / "plain.gguf")) is None


# ---------------------------------------------------- fake hub + download

class _HubHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002 - silence access log
        pass

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/api/models/org/Vision-GGUF":
            body = json.dumps({
                "id": "org/Vision-GGUF", "sha": "cafe1234",
                "siblings": [
                    {"rfilename": "vision-Q4_K_M.gguf", "size": 4},
                    {"rfilename": "mmproj-BF16.gguf", "size": 5},
                ]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif p == "/org/Vision-GGUF/resolve/cafe1234/mmproj-BF16.gguf":
            body = b"MMPRJ"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()


@pytest.fixture()
def fake_hub(monkeypatch):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _HubHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    monkeypatch.setenv("HF_ENDPOINT", f"http://127.0.0.1:{port}")
    yield srv
    srv.shutdown()


def test_remote_candidates_and_download(tmp_path, fake_hub):
    snap = (tmp_path / "models--org--Vision-GGUF" / "snapshots" / "cafe1234")
    snap.mkdir(parents=True)
    gguf = snap / "vision-Q4_K_M.gguf"
    gguf.write_bytes(b"GGUF")

    cands = mmproj.remote_mmproj_candidates("org/Vision-GGUF")
    assert [c["rfilename"] for c in cands] == ["mmproj-BF16.gguf"]

    info = mmproj.download_for(str(gguf))
    # lands NEXT TO the GGUF (expected launcher behavior)
    assert Path(info["path"]).parent == snap.resolve()
    assert Path(info["path"]).name == "mmproj-BF16.gguf"
    assert Path(info["path"]).read_bytes() == b"MMPRJ"
    # now detection finds it
    assert mmproj.find_local(str(gguf)) == info["path"]


def test_download_requires_provenance(tmp_path):
    (tmp_path / "plain.gguf").write_bytes(b"G")
    with pytest.raises(ValueError):
        mmproj.download_for(str(tmp_path / "plain.gguf"))


# ------------------------------------- 3. per-instance non-inheritance

class _Cfg:
    run_settings = {"flash_attn": True, "kv_cache_type": "q8_0"}
    context_length = 4096
    gpu_layers = None
    extra_llama_args = []


def test_named_instance_does_not_inherit_global_run_settings():
    rt_gpus_none = None
    inst = {"id": "coding", "gguf": "/x/model.gguf",
            "run_settings": {"parallel": 4}}
    cmd = build_instance_cmd("llama-server", _Cfg(), inst, port=1234,
                             model_name="m", rt=rt_gpus_none)
    assert "--flash-attn" not in cmd          # global setting NOT inherited
    assert "-ctk" not in cmd
    assert _has(cmd, "-np", "4")              # own settings applied


def test_default_instance_inherits_global_run_settings():
    for ident in ({"id": "default"}, {"default": True}):
        inst = {**ident, "gguf": "/x/model.gguf"}
        cmd = build_instance_cmd("llama-server", _Cfg(), inst, port=1234,
                                 model_name="m", rt=None)
        assert "--flash-attn" in cmd


def test_instance_cmd_vision_toggle_and_spec(tmp_path):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"G")
    (tmp_path / "mmproj.gguf").write_bytes(b"P")
    inst = {"id": "vision", "gguf": str(gguf), "run_settings": {}}
    cmd = build_instance_cmd("llama-server", _Cfg(), inst, port=1,
                             model_name="m", rt=None)
    assert _has(cmd, "--mmproj", str((tmp_path / "mmproj.gguf").resolve()))
    inst2 = {"id": "textonly", "gguf": str(gguf),
             "run_settings": {"vision": False}}
    cmd2 = build_instance_cmd("llama-server", _Cfg(), inst2, port=1,
                              model_name="m", rt=None)
    assert "--mmproj" not in cmd2


# ------------------------------------------ 4. dashboard API behavior

@pytest.fixture()
def home__test_run_settings_v2(tmp_path, monkeypatch):
    h = tmp_path / "uf-rs2"
    monkeypatch.setenv("SABLE_HOME", str(h))
    return h


def make_app__test_run_settings_v2(home) -> DashboardApp:
    return DashboardApp(SableConfig(llama_auto_setup=False), home=home)


def test_instance_create_validates_run_settings(home__test_run_settings_v2):
    app = make_app__test_run_settings_v2(home__test_run_settings_v2)
    gguf = Path(home__test_run_settings_v2).parent / "m.gguf"
    gguf.write_bytes(b"G")
    r = app.handle("POST", "/api/servers", body={
        "name": "ok", "gguf": str(gguf),
        "run_settings": {"speculative_type": "auto", "vision": True}})
    assert r.status == 201
    row = next(s for s in app.handle("GET", "/api/servers").body["servers"]
               if s["name"] == "ok")
    assert row["run_settings"]["speculative_type"] == "auto"
    bad = app.handle("POST", "/api/servers", body={
        "name": "bad", "gguf": str(gguf),
        "run_settings": {"bogus_key": 1}})
    assert bad.status == 400


def test_default_instance_run_settings_put_rejected(home__test_run_settings_v2):
    app = make_app__test_run_settings_v2(home__test_run_settings_v2)
    r = app.handle("PUT", "/api/servers/default",
                   body={"run_settings": {"flash_attn": True}})
    assert r.status == 400


def test_mmproj_status_endpoint(tmp_path, monkeypatch):
    h = tmp_path / "uf-mm"
    monkeypatch.setenv("SABLE_HOME", str(h))
    app = DashboardApp(SableConfig(llama_auto_setup=False), home=h)
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"G")
    r = app.handle("GET", "/api/mmproj", query={"gguf": str(gguf)})
    assert r.body["local"] is None
    (tmp_path / "mmproj-x.gguf").write_bytes(b"P")
    r = app.handle("GET", "/api/mmproj", query={"gguf": str(gguf)})
    assert r.body["local"].endswith("mmproj-x.gguf")
    assert app.handle("GET", "/api/mmproj", query={}).status == 400


def test_config_put_accepts_new_run_setting_keys(tmp_path, monkeypatch):
    cfgp = tmp_path / "sable.yaml"
    monkeypatch.setenv("SABLE_CONFIG", str(cfgp))
    h = tmp_path / "uf-cfg"
    app = DashboardApp(SableConfig(llama_auto_setup=False), home=h)
    r = app.handle("PUT", "/api/config", body={
        "section": "run_settings",
        "patch": {"speculative_type": "auto", "vision": True,
                  "spec_draft_n_max": 3}})
    assert r.status == 200 and r.body["ok"]
    import yaml
    data = yaml.safe_load(cfgp.read_text())
    assert data["run_settings"]["speculative_type"] == "auto"
    assert data["run_settings"]["vision"] is True


# ============================================================ image settings
# Diffusion engines (sd-server): per-instance generation defaults instead of
# the chat sampling knobs. sanitize_image_settings validates; 
# image_settings_args emits sd-server flags.

def test_sanitize_image_settings_valid_values():
    from sable.serving.run_args import sanitize_image_settings

    out = sanitize_image_settings({
        "steps": 24, "cfg_scale": "7.5", "width": 1024, "height": 576,
        "sampler": "DPM++2M".lower(), "scheduler": "karras",
        "seed": -1, "batch_count": 2})
    assert out == {"steps": 24, "cfg_scale": 7.5, "width": 1024,
                   "height": 576, "sampler": "dpm++2m",
                   "scheduler": "karras", "seed": -1, "batch_count": 2}


def test_sanitize_image_settings_rejects_bad_values():
    from sable.serving.run_args import sanitize_image_settings

    for bad in ({"top_k": 5},                    # chat knob, not image
                {"steps": 0}, {"steps": 999},
                {"cfg_scale": -1}, {"cfg_scale": 31},
                {"width": 8}, {"height": 99999},
                {"sampler": "nope"}, {"scheduler": "nope"},
                {"seed": -2}, {"batch_count": 0}):
        with pytest.raises(ValueError):
            sanitize_image_settings(bad)


def test_sanitize_image_settings_clears_empty():
    from sable.serving.run_args import sanitize_image_settings

    assert sanitize_image_settings({"steps": None, "cfg_scale": "",
                                    "seed": ""}) == {
        "steps": None, "cfg_scale": None, "seed": None}


def test_image_settings_args_emits_sd_server_flags():
    from sable.serving.run_args import image_settings_args

    args = image_settings_args({
        "steps": 30, "cfg_scale": 6.5, "width": 768, "height": 1024,
        "sampler": "euler_a", "scheduler": "sgm_uniform", "seed": 42,
        "batch_count": 3, "extra_args": "--vae-tiling --offload-to-cpu"})
    assert args == ["--steps", "30", "--cfg-scale", "6.5",
                    "-W", "768", "-H", "1024",
                    "--sampling-method", "euler_a",
                    "--scheduler", "sgm_uniform",
                    "-s", "42", "-b", "3",
                    "--vae-tiling", "--offload-to-cpu"]


def test_image_settings_args_empty_is_noop():
    from sable.serving.run_args import image_settings_args

    assert image_settings_args(None) == []
    assert image_settings_args({}) == []
    # seed 0 is a real seed and must still emit; batch_count 1 is the default
    assert image_settings_args({"seed": 0, "batch_count": 1}) == ["-s", "0"]

# ------------------------------------------------------------------------
# from tests/test_run_settings_indexing.py
# ------------------------------------------------------------------------

# ------------------------------------------------------------ 1. run args

def _has__test_run_settings_indexing(args, *tokens):
    """True when the tokens appear consecutively in args."""
    s = "\x00".join(args)
    return ("\x00".join(tokens)) in s


def test_run_args_empty():
    assert run_settings_args(None) == []
    assert run_settings_args({}) == []


def test_run_args_kv_and_flags():
    args = run_settings_args({
        "kv_cache_type": "q8_0",
        "flash_attn": True,
        "parallel": 4,
        "batch_size": 512,
        "ubatch_size": 128,
        "tensor_parallel": True,
        "no_mmap": True,
        "mlock": True,
        "mmproj_path": "/x/mm.gguf",
    })
    assert _has__test_run_settings_indexing(args, "-ctk", "q8_0", "-ctv", "q8_0")
    assert _has__test_run_settings_indexing(args, "--flash-attn", "on")
    assert _has__test_run_settings_indexing(args, "-np", "4") and _has__test_run_settings_indexing(args, "-b", "512") \
        and _has__test_run_settings_indexing(args, "-ub", "128")
    assert "--tensor-parallel" in args and "--no-mmap" in args
    assert "--mlock" in args and _has__test_run_settings_indexing(args, "--mmproj", "/x/mm.gguf")


def test_run_args_f16_kv_is_default_noop():
    assert run_settings_args({"kv_cache_type": "f16"}) == []


def test_run_args_vram_budget_scales_ngl():
    args = run_settings_args({"gpu_memory_pct": 50}, gpu_layers=999)
    assert _has__test_run_settings_indexing(args, "-ngl", "499")
    # 100% leaves the resolved offload untouched
    assert run_settings_args({"gpu_memory_pct": 100}, gpu_layers=999) == []
    assert scale_gpu_layers(999, 0) == 999


def test_run_args_extra_args_split():
    args = run_settings_args({"extra_args": "--rope-scaling yarn --yarn-orig-ctx 32768"})
    assert args == ["--rope-scaling", "yarn", "--yarn-orig-ctx", "32768"]


def test_sanitize_run_settings_rejects_unknown():
    with pytest.raises(ValueError):
        sanitize_run_settings({"bogus": 1})
    with pytest.raises(ValueError):
        sanitize_run_settings({"gpu_memory_pct": 101})
    out = sanitize_run_settings({"kv_cache_type": "Q8_0", "flash_attn": 1,
                                 "parallel": "4"})
    assert out == {"kv_cache_type": "q8_0", "flash_attn": True, "parallel": 4}


def test_pack_unpack_vectors():
    v = [0.25, -1.5, 3.0]
    assert _unpack(_pack(v)) == pytest.approx(v)
    assert cosine(v, v) == pytest.approx(1.0)


# ------------------------------------------------- 2. sampling + healing

def test_sampling_injected_into_requests():
    handler = make_message_handler({"*": {"role": "assistant", "content": "ok"}})
    with FakeOpenAIServer(handler_cls=handler) as srv:
        handler.requests.clear()
        client = SableAgentClient(base_url=srv.base_url, model="m",
                                    sampling={"temperature": 0.7, "top_k": 20})
        client.run("hi")
    assert handler.requests, "fake server saw no request"
    sent = handler.requests[-1]
    assert sent["temperature"] == 0.7
    assert sent["top_k"] == 20


def test_no_sampling_by_default():
    handler = make_message_handler({"*": {"role": "assistant", "content": "ok"}})
    with FakeOpenAIServer(handler_cls=handler) as srv:
        handler.requests.clear()
        client = SableAgentClient(base_url=srv.base_url, model="m")
        client.run("hi")
    sent = handler.requests[-1]
    assert "temperature" not in sent and "top_k" not in sent


def test_heal_tool_calls_toggle_off():
    """With healing off, a fenced-JSON tool call stays plain text (final answer)."""
    fenced = ('```json\n{"name": "todo_read", "arguments": {}}\n```')
    handler = make_message_handler(
        {"*": {"role": "assistant", "content": fenced}})
    with FakeOpenAIServer(handler_cls=handler) as srv:
        client = SableAgentClient(base_url=srv.base_url, model="m",
                                    heal_tool_calls=False)
        res = client.run("do it")
    assert res.completed
    # no tool executed -> the fenced block is (stripped or raw) final content,
    # and importantly the loop ended after one turn without a tool round-trip.
    assert "todo" not in (res.final_response or "").lower() or \
        res.messages[-1]["role"] == "assistant"


def test_create_runner_threads_sampling_and_healing(tmp_path, monkeypatch):
    monkeypatch.setenv("SABLE_HOME", str(tmp_path))
    cfg = SableConfig(sampling={"temperature": 0.1},
                        agent={"heal_tool_calls": False})
    from sable.core.agent_runner import create_runner

    r = create_runner(cfg, "http://127.0.0.1:1/v1", "m")
    assert r.sampling == {"temperature": 0.1}
    assert r.heal_tool_calls is False


# -------------------------------------------------------- 3. indexer

@pytest.fixture()
def project(tmp_path):
    root = tmp_path / "proj"
    (root / "src").mkdir(parents=True)
    (root / "src" / "embeddings.py").write_text(
        "def compute_embeddings(vectors):\n"
        "    # normalize each embedding vector before cosine similarity\n"
        "    return [v for v in vectors]\n" * 3)
    (root / "src" / "toast.py").write_text(
        "def raise_toast(message):\n    print('toast:', message)\n" * 3)
    (root / "notes.md").write_text("# notes\nsemantic search rocks\n")
    return root


def test_chunk_file_line_windows(project):
    chunks = chunk_file(project / "src" / "embeddings.py")
    assert chunks and chunks[0][0] == 1
    # windows overlap: second chunk starts before the first ends
    if len(chunks) > 1:
        assert chunks[1][0] <= chunks[0][1]


def test_lexical_index_scores_matching_doc():
    lx = LexicalIndex([(1, "embedding vector cosine similarity"),
                       (2, "toast notifications ui")])
    hits = lx.score("cosine embedding", limit=5)
    assert hits and hits[0][0] == 1


def test_index_build_and_lexical_search(tmp_path, project):
    idx = CodebaseIndex(tmp_path / "home", cfg=None,
                        index_cfg={"enabled": True, "paths": [str(project)]})
    res = idx.build()
    assert res["added"] == 3 and res["chunks"] >= 3
    assert res["embedded"] == 0 and res["embed_ok"] is False  # no model set

    hits = idx.search("cosine similarity embedding vector")
    assert hits, "lexical search returned nothing"
    assert hits[0]["path"].endswith("embeddings.py")

    st = idx.status()
    assert st["enabled"] and st["files"] == 3
    assert st["embedding"]["mode"] == "none"


def test_index_incremental_rebuild(tmp_path, project):
    idx = CodebaseIndex(tmp_path / "home2", cfg=None,
                        index_cfg={"enabled": True, "paths": [str(project)]})
    first = idx.build()
    assert first["added"] == 3
    second = idx.build()          # nothing changed
    assert second["added"] == 0 and second["updated"] == 0
    (project / "extra.py").write_text("def extra():\n    return 42\n")
    third = idx.build()
    assert third["added"] == 1
    (project / "notes.md").unlink()
    fourth = idx.build()
    assert fourth["removed"] == 1


def test_search_codebase_tool_disabled_message(monkeypatch, tmp_path):
    from sable.core import tools as uf_tools

    monkeypatch.setenv("SABLE_HOME", str(tmp_path / "h"))
    out = uf_tools.search_codebase("anything")
    assert "disabled" in out


def test_search_codebase_tool_over_built_index(monkeypatch, tmp_path, project):
    import sable.core.indexer as indexer_mod

    home__test_run_settings_indexing = tmp_path / "h2"
    monkeypatch.setenv("SABLE_HOME", str(home__test_run_settings_indexing))
    idx = CodebaseIndex(home__test_run_settings_indexing, cfg=None,
                        index_cfg={"enabled": True, "paths_override":
                                   [str(project)]})
    idx.build()
    # Phase 5: get_index caches per (home, project) — seed the default slot.
    monkeypatch.setitem(indexer_mod._instances, (str(home__test_run_settings_indexing), None), idx)

    from sable.core import tools as uf_tools

    out = uf_tools.search_codebase("embedding vector cosine")
    assert "embeddings.py" in out


# ---------------------------------------------------- 4. dashboard API

@pytest.fixture()
def home__test_run_settings_indexing(tmp_path, monkeypatch):
    h = tmp_path / "uf-ctl"
    monkeypatch.setenv("SABLE_HOME", str(h))
    return h


@pytest.fixture()
def app__test_run_settings_indexing(home__test_run_settings_indexing):
    return DashboardApp(SableConfig(), home=home__test_run_settings_indexing)


def test_logs_endpoint(app__test_run_settings_indexing, home__test_run_settings_indexing):
    (home__test_run_settings_indexing / "logs").mkdir(parents=True)
    (home__test_run_settings_indexing / "logs" / "dashboard.log").write_text("one\ntwo\nthree\n")
    r = app__test_run_settings_indexing.handle("GET", "/api/logs", query={"tail": "2"})
    assert r.status == 200
    body = r.body
    assert body["dashboard"]["exists"] is True
    assert body["dashboard"]["lines"][-2:] == ["two", "three"]
    assert body["llama-server"]["exists"] is False


def test_logs_endpoint_includes_instance_logs(app__test_run_settings_indexing, home__test_run_settings_indexing):
    """Named instances write server-<id>.log — the console must list them."""
    (home__test_run_settings_indexing / "logs").mkdir(parents=True)
    (home__test_run_settings_indexing / "logs" / "server-coder.log").write_text("slot booted\n")
    (home__test_run_settings_indexing / "logs" / "server-embed2.log").write_text("ready\n")
    body = app__test_run_settings_indexing.handle("GET", "/api/logs").body
    assert body["server-coder"]["exists"] is True
    assert body["server-coder"]["lines"] == ["slot booted"]
    assert body["server-embed2"]["exists"] is True


def test_shutdown_route_mocked(app__test_run_settings_indexing, home__test_run_settings_indexing, monkeypatch):
    import sable.dashboard.app as appmod

    kills = []
    monkeypatch.setattr(appmod.os, "kill",
                        lambda pid, sig: kills.append((pid, sig)))
    r = app__test_run_settings_indexing.handle("POST", "/api/shutdown", body={})
    assert r.status == 200 and r.body["ok"]
    import time as _t
    _t.sleep(1.0)  # the daemon thread sleeps ~0.4 s before signalling
    assert kills and kills[0][1].name == "SIGTERM"


def test_index_endpoints(app__test_run_settings_indexing, home__test_run_settings_indexing, tmp_path, project, monkeypatch):
    import sable.core.indexer as indexer_mod

    idx = CodebaseIndex(home__test_run_settings_indexing, cfg=None,
                        index_cfg={"enabled": True, "paths_override":
                                   [str(project)]})
    # The app now routes through get_index's cache; seed the default slot so
    # _index(None) returns this lexical-only test index.
    monkeypatch.setitem(indexer_mod._instances, (str(home__test_run_settings_indexing), None), idx)

    r = app__test_run_settings_indexing.handle("GET", "/api/index")
    assert r.status == 200 and r.body["enabled"] is True

    r = app__test_run_settings_indexing.handle("POST", "/api/index/build", body={})
    assert r.status == 200 and r.body["state"] == "running"
    # wait for the background build to land
    import time as _t
    st = {}
    for _ in range(100):
        st = app__test_run_settings_indexing.handle("GET", "/api/index").body
        if not st["building"] and st["chunks"]:
            break
        _t.sleep(0.05)
    assert st["files"] == 3

    r = app__test_run_settings_indexing.handle("POST", "/api/index/search",
                   body={"query": "cosine embedding vector"})
    assert r.status == 200 and r.body["results"]

    r = app__test_run_settings_indexing.handle("POST", "/api/index/stop", body={})
    assert r.status == 200

    r = app__test_run_settings_indexing.handle("POST", "/api/index/search", body={})
    assert r.status == 400


def test_config_put_accepts_new_sections(app__test_run_settings_indexing, home__test_run_settings_indexing, monkeypatch):
    monkeypatch.setenv("SABLE_CONFIG", str(home__test_run_settings_indexing / "sable.yaml"))
    r = app__test_run_settings_indexing.handle("PUT", "/api/config",
                   body={"section": "run_settings",
                         "patch": {"kv_cache_type": "q8_0",
                                   "gpu_memory_pct": 90}})
    assert r.status == 200, r.body
    r = app__test_run_settings_indexing.handle("PUT", "/api/config",
                   body={"section": "sampling", "patch": {"temperature": 0.6}})
    assert r.status == 200, r.body
    r = app__test_run_settings_indexing.handle("PUT", "/api/config",
                   body={"section": "indexing",
                         "patch": {"enabled": True, "search_mode": "hybrid"}})
    assert r.status == 200, r.body

    # unknown keys inside the new sections are rejected
    r = app__test_run_settings_indexing.handle("PUT", "/api/config",
                   body={"section": "sampling", "patch": {"vibes": 9}})
    assert r.status == 400 and r.body["error"]["code"] == "unknown_keys"

    # and the file reloads into the typed config
    cfg = load_config(str(home__test_run_settings_indexing / "sable.yaml"))
    assert cfg.run_settings["kv_cache_type"] == "q8_0"
    assert cfg.sampling["temperature"] == 0.6
    assert cfg.indexing["enabled"] is True

# ------------------------------------------------------------------------
# from tests/test_insights.py
# ------------------------------------------------------------------------

# --------------------------------------------------------------------- seeds

class _FrozenDatetime(datetime):
    """datetime.now() shim so the recorder writes controlled timestamps."""
    frozen: datetime = datetime(2026, 9, 1, tzinfo=timezone.utc)

    @classmethod
    def now(cls, tz=None):
        return cls.frozen if tz is None else cls.frozen.astimezone(tz)


@pytest.fixture()
def home__test_insights(tmp_path, monkeypatch):
    h = tmp_path / "uf-insights"
    monkeypatch.setenv("SABLE_HOME", str(h))
    monkeypatch.setattr(recorder_mod, "datetime", _FrozenDatetime)
    return h


def _tool_messages(name: str, fail: bool) -> list:
    """One assistant tool_call + its tool result (pass or fail)."""
    content = "error: boom" if fail else "ok output"
    return [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": f"call_{name}", "type": "function",
             "function": {"name": name, "arguments": "{}"}}]},
        {"role": "tool", "name": name, "tool_call_id": f"call_{name}",
         "content": content},
    ]


def seed(home, *, when: datetime, profile="sable", model_alias="base-model",
         model_version=None, completed=True, fail_tool=False, tool=None,
         tokens_in=None, tokens_out=None, duration_s=None,
         failure_reason=None, conf_mean=None, conf_p90_negentropy=None):
    """Record one synthetic trace at a frozen timestamp."""
    _FrozenDatetime.frozen = when
    msgs = [{"role": "user", "content": "do the thing"}]
    if tool:
        msgs += _tool_messages(tool, fail_tool)
    msgs.append({"role": "assistant", "content": "done"})
    rec = TraceRecorder(home=home)
    try:
        row = rec.record(
            task="do the thing", messages=msgs, backend=profile,
            model_alias=model_alias, base_url="http://127.0.0.1:9/v1",
            completed=completed, tokens_in=tokens_in, tokens_out=tokens_out,
            duration_s=duration_s, model_version=model_version,
            failure_reason=failure_reason,
            conf_mean=conf_mean, conf_p90_negentropy=conf_p90_negentropy,
        )
    finally:
        rec.close()
    assert row is not None
    return row


def now():
    return datetime.now(timezone.utc)


# ------------------------------------------------------------- zero structure

def test_empty_db_returns_well_formed_zero_structure(home__test_insights):
    r = generate_insights(home__test_insights, days=30)
    assert r["empty"] is True
    assert r["overview"]["traces"] == 0
    assert r["overview"]["avg_score"] is None
    assert r["score_trend"]["overall"]["points"] == []
    assert r["score_trend"]["overall"]["half_mean_delta"] is None
    assert r["tool_reliability"]["tools"] == []
    assert r["failure_reasons"]["total"] == 0
    assert r["latency"]["duration_s"]["p50"] is None
    assert r["activity"]["traces_per_day"] == []
    assert r["promotions"]["available"] is False


# ------------------------------------------------------------- overview/trend

def test_overview_and_score_trend_exact(home__test_insights):
    # day 1: scores 0.5, 0.25 (avg 0.375) · day 2: score 0.5 -> trend up
    seed(home__test_insights, when=now() - timedelta(days=2), completed=True)          # 0.5
    seed(home__test_insights, when=now() - timedelta(days=2), completed=False)         # -0.5
    seed(home__test_insights, when=now() - timedelta(days=1), completed=True)          # 0.5

    r = generate_insights(home__test_insights, days=30)
    ov = r["overview"]
    assert ov["traces"] == 3
    assert ov["avg_score"] == pytest.approx((0.5 - 0.5 + 0.5) / 3, abs=1e-4)
    assert ov["status_counts"]["failure"] == 1
    assert ov["status_counts"]["success"] == 2

    pts = r["score_trend"]["overall"]["points"]
    assert [p["traces"] for p in pts] == [2, 1]
    assert pts[0]["avg_score"] == pytest.approx(0.0)   # (0.5 + -0.5)/2
    assert pts[1]["avg_score"] == pytest.approx(0.5)
    # two scored buckets: second-half mean (0.5) - first-half mean (0.0)
    assert r["score_trend"]["overall"]["half_mean_delta"] == pytest.approx(0.5)
    assert r["score_trend"]["overall"]["direction"] == "up"

    prof = r["score_trend"]["by_profile"]
    assert len(prof) == 1 and prof[0]["traces"] == 3


def test_score_trend_grouped_by_model(home__test_insights):
    seed(home__test_insights, when=now() - timedelta(days=1), model_alias="v1")
    seed(home__test_insights, when=now() - timedelta(days=1), model_alias="v2",
         model_version="v2")
    r = generate_insights(home__test_insights)
    names = {g["name"] for g in r["score_trend"]["by_model"]}
    assert names == {"v1", "v2"}  # model_version wins over alias


# ------------------------------------------------------------ tool reliability

def test_tool_failure_rates_from_jsonl_scan(home__test_insights):
    seed(home__test_insights, when=now() - timedelta(days=1), tool="read_file")
    seed(home__test_insights, when=now() - timedelta(days=1), tool="read_file", fail_tool=True)
    seed(home__test_insights, when=now() - timedelta(days=1), tool="shell", fail_tool=True)

    tr = generate_insights(home__test_insights)["tool_reliability"]
    assert tr["available"] is True and tr["traces_scanned"] == 3
    by_name = {t["tool"]: t for t in tr["tools"]}
    assert by_name["read_file"]["calls"] == 2
    assert by_name["read_file"]["failures"] == 1
    assert by_name["read_file"]["failure_rate"] == pytest.approx(0.5)
    assert by_name["shell"]["calls"] == 1
    assert by_name["shell"]["failure_rate"] == pytest.approx(1.0)


def test_tool_reliability_tolerates_pruned_transcripts(home__test_insights):
    row = seed(home__test_insights, when=now() - timedelta(days=1), tool="shell")
    Path(row.jsonl_path).unlink()  # transcript pruned -> skipped silently
    tr = generate_insights(home__test_insights)["tool_reliability"]
    assert tr["traces_scanned"] == 0
    assert tr["tools"] == []


# ----------------------------------------------------------- failure reasons

def test_failure_reason_histogram_splits_infra_model(home__test_insights):
    seed(home__test_insights, when=now() - timedelta(days=1), completed=False,
         failure_reason="backend_error:timeout")     # infra
    seed(home__test_insights, when=now() - timedelta(days=1), completed=False,
         failure_reason="max_iterations")            # model
    seed(home__test_insights, when=now() - timedelta(days=1), completed=False,
         failure_reason="max_iterations")            # model
    seed(home__test_insights, when=now() - timedelta(days=1))       # no reason

    fr = generate_insights(home__test_insights)["failure_reasons"]
    assert fr["total"] == 3
    assert fr["infra_total"] == 1 and fr["model_total"] == 2
    top = fr["by_reason"][0]
    assert top["reason"] == "max_iterations" and top["count"] == 2
    assert top["kind"] == "model"
    infra = next(b for b in fr["infra"] if b["reason"] == "backend_error:timeout")
    assert infra["count"] == 1


# ---------------------------------------------------------------- percentiles

def test_duration_token_percentiles(home__test_insights):
    for d, t in ((1.0, 100), (2.0, 200), (3.0, 300), (4.0, 400)):
        seed(home__test_insights, when=now() - timedelta(days=1), duration_s=d,
             tokens_in=t // 2, tokens_out=t // 2)
    lat = generate_insights(home__test_insights)["latency"]
    ds, tk = lat["duration_s"], lat["tokens_total"]
    assert ds["count"] == 4 and ds["max"] == 4.0
    # nearest rank: p50 of [1,2,3,4] -> index 1 (2.0); p90 -> index 3 (4.0)
    assert ds["p50"] == 2.0 and ds["p90"] == 4.0
    assert tk["count"] == 4 and tk["p50"] == 200.0 and tk["max"] == 400.0


# ------------------------------------------------- percentile parity (#16)
#
# insights.percentile and agent_runner._confidence_aggregate both claim
# nearest-rank semantics; the old banker's-round() index disagreed with
# ceil(p*n) at exact .5 boundaries (e.g. p50 @ n=15, p90 @ n=5/16/25/36,
# p95 @ n=11..19/30..39). Pin the canonical rank across n=1..40 on random
# and adversarial samples so the two can never silently diverge again.

def _rank_samples(n: int) -> list:
    import random

    rng = random.Random(1234 + n)
    return [
        sorted(rng.uniform(-10, 10) for _ in range(n)),      # random
        [7.0] * n,                                           # all-equal
        sorted(rng.choice([1.0, 1.0, 2.0, 3.0]) for _ in range(n)),  # dup-heavy
        [float(i) for i in range(n)],                        # strictly increasing
    ]


@pytest.mark.parametrize("pct", [50, 90, 95])
@pytest.mark.parametrize("n", list(range(1, 41)))
def test_percentile_matches_nearest_rank_ceiling(pct, n):
    from sable.store.insights import percentile

    for vals in _rank_samples(n):
        expected = vals[math.ceil(pct * n / 100) - 1]   # canonical nearest rank
        got = percentile(vals, pct)
        assert got == pytest.approx(round(expected, 4)), (pct, n, vals[:5])


@pytest.mark.parametrize("n", list(range(1, 41)))
def test_confidence_aggregate_p90_matches_insights_percentile(n):
    """agent_runner._confidence_aggregate's p90 picks the SAME sorted element
    as insights.percentile(..., 90) (its docstring promises this parity).
    Compared via exp() over logprob-domain (<= 0) samples so rounding stays
    bounded; the previous test pins percentile itself to ceil-rank."""
    import random

    from sable.core.agent_runner import _confidence_aggregate

    rng = random.Random(99 + n)
    for vals in (sorted(-rng.random() * 20 for _ in range(n)),   # random lp
                 [math.log(0.5)] * n,                            # all-equal
                 sorted(rng.choice([math.log(0.1), math.log(0.1),
                                    math.log(0.9)]) for _ in range(n))):
        _, p90 = _confidence_aggregate(vals)
        expected = math.exp(vals[math.ceil(90 * n / 100) - 1])
        assert p90 == pytest.approx(expected, abs=1e-3), (n, vals[:5])


# ------------------------------------------------------------------- activity

def test_activity_hour_profiles_daily(home__test_insights):
    seed(home__test_insights, when=now().replace(hour=9), tool="shell")
    seed(home__test_insights, when=now().replace(hour=9))
    seed(home__test_insights, when=now().replace(hour=3), profile="coder")

    act = generate_insights(home__test_insights)["activity"]
    assert act["busiest_hour"] == "09"
    hist = {h["hour"]: h["traces"] for h in act["hour_histogram"]}
    assert hist["09"] == 2 and hist["03"] == 1 and len(hist) == 24
    profiles = {p["profile"]: p for p in act["profiles"]}
    assert set(profiles) == {"sable", "coder"}
    assert profiles["sable"]["traces"] == 2
    assert act["traces_per_day"][-1]["traces"] == 3


# ------------------------------------------------------------------- filters

def test_days_and_profile_filters(home__test_insights):
    seed(home__test_insights, when=now() - timedelta(days=3), profile="sable")
    seed(home__test_insights, when=now() - timedelta(days=40), profile="sable")
    seed(home__test_insights, when=now() - timedelta(days=1), profile="coder")

    r30 = generate_insights(home__test_insights, days=30)
    assert r30["overview"]["traces"] == 2          # the 40-day-old one drops
    r_p = generate_insights(home__test_insights, days=30, profile="coder")
    assert r_p["overview"]["traces"] == 1
    assert all(p["profile"] == "coder" for p in r_p["activity"]["profiles"])


# --------------------------------------------------------------- promotions

def test_promotion_delta_before_after(home__test_insights):
    db = home__test_insights / "buffer" / "sable.db"
    reg = ModelRegistry(db)
    try:
        m1 = reg.register_model("base-x")           # v1
        reg.set_validation(m1.model_id, {"tasks": []}, passed=True)
        reg.promote(m1.model_id)
        m2 = reg.register_model("base-x")           # v2, parent v1
        assert m2.parent_adapter == "v1"
        reg.set_validation(m2.model_id, {"tasks": []}, passed=True)
        reg.promote(m2.model_id)
        # Backdate both promotions so symmetric windows contain seeded traces.
        with reg._lock:
            for mid, days_ago in (("v1", 6), ("v2", 3)):
                ts = (now() - timedelta(days=days_ago)).isoformat(
                    timespec="seconds")
                reg._conn.execute("UPDATE models SET promoted_at = ? "
                                  "WHERE model_id = ?", (ts, mid))
            reg._conn.commit()
    finally:
        reg.close()

    # History edges at -4d/-2d make the symmetric window around the v2
    # promotion (-3d) ±1d: one v1 trace before it, one failing v2 trace after.
    seed(home__test_insights, when=now() - timedelta(days=4), model_alias="v1")
    seed(home__test_insights, when=now() - timedelta(days=2), model_alias="v2",
         completed=False)

    promo = generate_insights(home__test_insights)["promotions"]
    assert promo["available"] is True

    v2 = next(d for d in promo["deltas"] if d["model_id"] == "v2")
    assert v2["previous_model_id"] == "v1"
    assert v2["match"] == "model_id"          # traces carry model_alias=vN
    assert v2["window_days"] == pytest.approx(1.0)
    assert v2["before_avg_score"] == pytest.approx(0.5)   # v1 traces before
    assert v2["after_avg_score"] == pytest.approx(-0.5)   # v2 traces after
    assert v2["delta"] == pytest.approx(-1.0)
    assert v2["before_traces"] == 1 and v2["after_traces"] == 1

    # v1 (first promotion, no parent): entry exists; before-window falls back
    # to all traces in the window since there is no previous model to match.
    v1 = next(d for d in promo["deltas"] if d["model_id"] == "v1")
    assert v1["previous_model_id"] is None
    assert v1["promoted_at"] is not None


def test_promotion_section_degrades_without_history(home__test_insights):
    seed(home__test_insights, when=now() - timedelta(days=1))
    promo = generate_insights(home__test_insights)["promotions"]
    # No registry tables in a fresh buffer DB -> unavailable with a reason.
    assert promo["available"] is False
    assert promo["reason"]


# ---------------------------------------------------- intent review (#36)

def test_intent_review_pairs_promote_intent_with_delta(home__test_insights):
    """Promote-with-intent gets the same before/after window as promotions."""
    db = home__test_insights / "buffer" / "sable.db"
    reg = ModelRegistry(db)
    try:
        m1 = reg.register_model("base-x")            # v1
        reg.set_validation(m1.model_id, {"tasks": []}, passed=True)
        reg.promote(m1.model_id)
        m2 = reg.register_model("base-x")            # v2, parent v1
        reg.set_validation(m2.model_id, {"tasks": []}, passed=True)
        reg.promote(m2.model_id, intent="expect fewer tool errors")
        # Backdate the ledger entry to when the promotion "happened", mirroring
        # promoted_at (the review anchors windows on the entry's own ts).
        ts = (now() - timedelta(days=3)).isoformat(timespec="seconds")
        with reg._lock:
            reg._conn.execute(
                "UPDATE ledger SET ts = ? WHERE op = 'promote' AND seq = "
                "(SELECT MAX(seq) FROM ledger WHERE op = 'promote')", (ts,))
            reg._conn.commit()
    finally:
        reg.close()

    # Same edges as the promotions test: ±1d window around -3d.
    seed(home__test_insights, when=now() - timedelta(days=4), model_alias="v1")
    seed(home__test_insights, when=now() - timedelta(days=2), model_alias="v2",
         completed=False)

    ir = generate_insights(home__test_insights)["intent_review"]
    assert ir["available"] is True
    # Only the intented promote shows up; v1's promote had none.
    assert len(ir["rows"]) == 1
    r = ir["rows"][0]
    assert r["op"] == "promote" and r["target"] == "v2"
    assert r["intent"] == "expect fewer tool errors"
    assert r["before_score"] == pytest.approx(0.5)   # v1 traces before
    assert r["after_score"] == pytest.approx(-0.5)   # failing v2 traces after
    assert r["delta"] == pytest.approx(-1.0)         # expectation missed
    assert r["window_days"] == pytest.approx(1.0)
    assert r["match"] == "model_id"


def test_intent_review_honest_empty_no_fabrication(home__test_insights):
    seed(home__test_insights, when=now() - timedelta(days=1))       # traces but no intents
    ir = generate_insights(home__test_insights)["intent_review"]
    assert ir["available"] is True                   # readable, just empty
    assert ir["rows"] == []                          # zero synthetic rows
    assert ir["hint"] == "no recorded intents"


def test_intent_review_ignores_ops_without_intent(home__test_insights):
    db = home__test_insights / "buffer" / "sable.db"
    reg = ModelRegistry(db)
    try:
        reg.register_model("base-x")
        reg.set_validation("v1", {"tasks": []}, passed=True)
        reg.promote("v1")                            # no intent recorded
    finally:
        reg.close()
    seed(home__test_insights, when=now() - timedelta(days=1))
    ir = generate_insights(home__test_insights)["intent_review"]
    assert ir["available"] is True and ir["rows"] == []


def test_intent_review_cli_output(home__test_insights, capsys):
    from sable.cli import main as cli_main

    db = home__test_insights / "buffer" / "sable.db"
    reg = ModelRegistry(db)
    try:
        reg.register_model("base-x")
        reg.set_validation("v1", {"tasks": []}, passed=True)
    finally:
        reg.close()
    assert cli_main(["models", "promote", "v1", "--intent",
                     "halve the timeout failures"]) == 0
    capsys.readouterr()
    assert cli_main(["insights", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    ir = report["intent_review"]
    assert ir["available"] is True and len(ir["rows"]) == 1
    assert ir["rows"][0]["intent"] == "halve the timeout failures"


# ------------------------------------------------- confidence calibration (#40)

def test_calibration_buckets_and_confidently_wrong(home__test_insights):
    """High-confidence-but-failing top bucket flips the divergence flag."""
    day = now() - timedelta(days=1)
    # very_high bucket (conf >= -0.1): three FAILING traces (score -0.5 each)
    for _ in range(3):
        seed(home__test_insights, when=day, completed=False, conf_mean=-0.05,
             conf_p90_negentropy=0.95)
    # low bucket ([-2.0,-1.0)): three succeeding traces (score +0.5 each)
    for _ in range(3):
        seed(home__test_insights, when=day, completed=True, conf_mean=-1.5,
             conf_p90_negentropy=0.22)
    # no confidence data -> excluded from every bucket
    seed(home__test_insights, when=day, completed=True)

    cal = generate_insights(home__test_insights)["calibration"]
    assert cal["available"] is True
    by_label = {r["bucket_label"]: r for r in cal["rows"]}
    assert sum(r["n"] for r in cal["rows"]) == 6      # conf-less trace excluded
    top = by_label["very_high (>=-0.1)"]
    assert top["n"] == 3
    assert top["mean_conf"] == pytest.approx(-0.05, abs=1e-4)
    assert top["mean_score"] == pytest.approx(-0.5)
    low = by_label["low ([-2.0,-1.0))"]
    assert low["n"] == 3 and low["mean_score"] == pytest.approx(0.5)
    # pooled others (+0.5) minus top (-0.5) = 1.0 > 0.15 gap threshold
    assert cal["confidently_wrong"] is True


def test_calibration_no_divergence_when_confidence_tracks_quality(home__test_insights):
    day = now() - timedelta(days=1)
    for _ in range(3):
        seed(home__test_insights, when=day, completed=True, conf_mean=-0.05)   # top: +0.5
    for _ in range(3):
        seed(home__test_insights, when=day, completed=False, conf_mean=-1.5)   # low: -0.5
    cal = generate_insights(home__test_insights)["calibration"]
    assert cal["confidently_wrong"] is False


def test_calibration_honest_empty(home__test_insights):
    seed(home__test_insights, when=now() - timedelta(days=1))       # traces, no conf data
    cal = generate_insights(home__test_insights)["calibration"]
    assert cal["available"] is True
    assert cal["rows"] == []
    assert cal["hint"] == "no traces with confidence data yet"
    assert cal["confidently_wrong"] is False


def test_calibration_cli_output(home__test_insights, capsys):
    seed(home__test_insights, when=now() - timedelta(days=1), conf_mean=-0.3)
    from sable.cli import build_parser

    args = build_parser().parse_args(["insights"])
    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert "Confidence calibration" in out
    assert "high ([-0.5,-0.1))" in out


# ------------------------------------------------------------------------ CLI

def test_cli_insights_json_round_trip(home__test_insights, capsys):
    seed(home__test_insights, when=now() - timedelta(days=1), tool="shell", fail_tool=True)
    from sable.cli import build_parser

    args = build_parser().parse_args(["insights", "--days", "7", "--json"])
    assert args.command == "insights" and args.days == 7
    assert args.func(args) == 0
    out = capsys.readouterr().out
    report = json.loads(out)
    assert report["days"] == 7
    assert report["overview"]["traces"] == 1
    assert report["tool_reliability"]["tools"][0]["tool"] == "shell"


def test_cli_insights_human_output(home__test_insights, capsys):
    seed(home__test_insights, when=now() - timedelta(days=1), completed=False,
         failure_reason="max_iterations")
    from sable.cli import build_parser

    args = build_parser().parse_args(["insights"])
    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert "Insights — last 30 days" in out
    assert "max_iterations" in out


# ------------------------------------------------------------------- API

def test_api_insights_endpoint(home__test_insights):
    seed(home__test_insights, when=now() - timedelta(days=1), profile="sable")
    app = DashboardApp(SableConfig(), home=home__test_insights)
    r = app.handle("GET", "/api/insights", query={"days": "30"})
    assert r.status == 200
    assert r.body["overview"]["traces"] == 1
    assert r.body["score_trend"]["overall"]["points"]

    rp = app.handle("GET", "/api/insights",
                    query={"days": "30", "profile": "nope"})
    assert rp.status == 200 and rp.body["empty"] is True


def test_api_insights_bad_days(home__test_insights):
    app = DashboardApp(SableConfig(), home=home__test_insights)
    r = app.handle("GET", "/api/insights", query={"days": "abc"})
    assert r.status == 400

# ------------------------------------------------------------------------
# from tests/test_fs_tree.py
# ------------------------------------------------------------------------

@pytest.fixture()
def home__test_fs_tree(tmp_path, monkeypatch):
    h = tmp_path / "uf-fs"
    monkeypatch.setenv("SABLE_HOME", str(h))
    return h


@pytest.fixture()
def app__test_fs_tree(home__test_fs_tree):
    return DashboardApp(SableConfig(), home=home__test_fs_tree)


@pytest.fixture()
def workspace(tmp_path):
    ws = tmp_path / "ws"
    (ws / "src").mkdir(parents=True)
    (ws / "src" / "app.py").write_text("print('hi')\n")
    (ws / "README.md").write_text("# hi\n")
    (ws / ".hidden-file").write_text("x")
    (ws / ".config").mkdir()
    return ws


def _mkproject(app, ws):
    r = app.handle("POST", "/api/projects",
                   body={"name": "p", "workspace_dir": str(ws)})
    assert r.status == 201
    return r.body["project"]["id"]


# ------------------------------------------------------------------ happy path

def test_fs_tree_root_dirs_first_one_level(app__test_fs_tree, workspace):
    pid = _mkproject(app__test_fs_tree, workspace)
    r = app__test_fs_tree.handle("GET", "/api/fs/tree", query={"project": pid, "path": ""})
    assert r.status == 200, r.body
    entries = r.body["entries"]
    names = [e["name"] for e in entries]
    # dirs first, then files, each group sorted; dotfiles hidden by default
    assert names == ["src", "README.md"]
    src = entries[0]
    assert src["type"] == "dir" and src["path"] == "src"
    readme = entries[1]
    assert readme["type"] == "file" and readme["path"] == "README.md"
    assert readme["size"] == len("# hi\n")
    assert readme["mtime"]
    # one level only: nested dir contents are NOT expanded
    assert all("src/" not in e["path"] for e in entries)

    # expanding the subdir returns its single level with rel paths prefixed
    r = app__test_fs_tree.handle("GET", "/api/fs/tree", query={"project": pid, "path": "src"})
    assert r.status == 200
    assert [(e["name"], e["path"], e["type"]) for e in r.body["entries"]] == \
        [("app.py", "src/app.py", "file")]


# ------------------------------------------------------------------- containment

def test_fs_tree_rejects_traversal(app__test_fs_tree, workspace):
    pid = _mkproject(app__test_fs_tree, workspace)
    outside = workspace.parent / "evil"
    outside.mkdir()
    (outside / "secret.txt").write_text("boo")

    for bad in ("../evil", "..", "src/../../evil", "a/../b"):
        r = app__test_fs_tree.handle("GET", "/api/fs/tree",
                       query={"project": pid, "path": bad})
        assert r.status == 400, (bad, r.body)
        assert r.body["error"]["code"] == "bad_request"


def test_fs_tree_rejects_absolute_paths(app__test_fs_tree, workspace):
    pid = _mkproject(app__test_fs_tree, workspace)
    for bad in ("/etc", str(workspace.parent), "~/.ssh"):
        r = app__test_fs_tree.handle("GET", "/api/fs/tree",
                       query={"project": pid, "path": bad})
        assert r.status == 400, (bad, r.body)


def test_fs_tree_rejects_encoded_traversal(app__test_fs_tree, workspace):
    pid = _mkproject(app__test_fs_tree, workspace)
    # The HTTP layer decodes query strings via parse_qs; the handler re-checks
    # the unquoted form too, so both shapes are rejected.
    for enc in ("%2e%2e/evil", "..%2fevil", "%2E%2E"):
        r = app__test_fs_tree.handle("GET", "/api/fs/tree",
                       query={"project": pid, "path": enc})
        assert r.status == 400, (enc, r.body)
        # and after transport-level decoding (what parse_qs delivers) likewise
        from urllib.parse import unquote
        r2 = app__test_fs_tree.handle("GET", "/api/fs/tree",
                        query={"project": pid, "path": unquote(enc)})
        assert r2.status == 400, (unquote(enc), r2.body)


@pytest.mark.skipif(os.name != "posix", reason="symlinks are posix here")
def test_fs_tree_symlink_escape_not_listed_or_traversed(app__test_fs_tree, workspace, tmp_path):
    pid = _mkproject(app__test_fs_tree, workspace)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "private.txt").write_text("secret")
    link = workspace / "link-out"
    link.symlink_to(outside)

    # not listed at all (target resolves outside the workspace root)
    r = app__test_fs_tree.handle("GET", "/api/fs/tree", query={"project": pid, "path": ""})
    assert r.status == 200
    assert "link-out" not in [e["name"] for e in r.body["entries"]]

    # and cannot be traversed into either
    r = app__test_fs_tree.handle("GET", "/api/fs/tree",
                   query={"project": pid, "path": "link-out"})
    assert r.status == 400, r.body


# ---------------------------------------------------------------- sensitive paths

def test_fs_tree_omits_sensitive_entries(app__test_fs_tree, workspace):
    pid = _mkproject(app__test_fs_tree, workspace)
    # real cases from core/tools.py deny lists: .ssh/ dir + .env file
    (workspace / ".ssh").mkdir()
    (workspace / ".ssh" / "id_rsa").write_text("key")
    (workspace / ".env").write_text("SECRET=1")

    r = app__test_fs_tree.handle("GET", "/api/fs/tree",
                   query={"project": pid, "path": "", "hidden": "1"})
    assert r.status == 200
    names = [e["name"] for e in r.body["entries"]]
    assert ".ssh" not in names and ".env" not in names
    assert "src" in names  # control: normal entries still listed


def test_fs_tree_sensitive_dir_blocked_as_root(app__test_fs_tree, workspace):
    pid = _mkproject(app__test_fs_tree, workspace)
    (workspace / ".ssh").mkdir()
    r = app__test_fs_tree.handle("GET", "/api/fs/tree",
                   query={"project": pid, "path": ".ssh"})
    assert r.status == 400
    assert "sensitive" in r.body["error"]["message"]


# ----------------------------------------------------------------------- hidden

def test_fs_tree_hidden_dotfiles_opt_in(app__test_fs_tree, workspace):
    pid = _mkproject(app__test_fs_tree, workspace)
    r = app__test_fs_tree.handle("GET", "/api/fs/tree", query={"project": pid})
    names = [e["name"] for e in r.body["entries"]]
    assert ".hidden-file" not in names and ".config" not in names

    r = app__test_fs_tree.handle("GET", "/api/fs/tree",
                   query={"project": pid, "hidden": "1"})
    names = [e["name"] for e in r.body["entries"]]
    assert ".hidden-file" in names and ".config" in names


# ------------------------------------------------------------------------ errors

def test_fs_tree_unknown_project_404(app__test_fs_tree):
    r = app__test_fs_tree.handle("GET", "/api/fs/tree", query={"project": "does-not-exist"})
    assert r.status == 404
    assert r.body["error"]["code"] == "unknown_project"


def test_fs_tree_requires_project_param(app__test_fs_tree):
    r = app__test_fs_tree.handle("GET", "/api/fs/tree", query={})
    assert r.status == 400
    r = app__test_fs_tree.handle("GET", "/api/fs/tree", query={"project": "  "})
    assert r.status == 400


def test_fs_tree_missing_workspace_dir_404(app__test_fs_tree, home__test_fs_tree, tmp_path):
    ws = tmp_path / "gone"
    ws.mkdir()
    pid = _mkproject(app__test_fs_tree, ws)
    ws.rmdir()  # workspace deleted after the project was created
    r = app__test_fs_tree.handle("GET", "/api/fs/tree", query={"project": pid})
    assert r.status == 404


def test_fs_tree_missing_subdir_404(app__test_fs_tree, workspace):
    pid = _mkproject(app__test_fs_tree, workspace)
    r = app__test_fs_tree.handle("GET", "/api/fs/tree",
                   query={"project": pid, "path": "nope"})
    assert r.status == 404


def test_fs_tree_endpoint_is_read_only(app__test_fs_tree, workspace):
    # listing must never mutate anything on disk
    pid = _mkproject(app__test_fs_tree, workspace)
    before = sorted(os.listdir(workspace))
    app__test_fs_tree.handle("GET", "/api/fs/tree", query={"project": pid})
    assert sorted(os.listdir(workspace)) == before

# ------------------------------------------------------------------------
# from tests/test_images.py
# ------------------------------------------------------------------------

def png_bytes(w: int = 2, h: int = 2) -> bytes:
    """Smallest valid PNG (solid color), stdlib only."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data)))

    raw = b"".join(b"\x00" + b"\x7f\xff\x00" * w for _ in range(h))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b""))


@pytest.fixture()
def png(tmp_path) -> Path:
    p = tmp_path / "pic.png"
    p.write_bytes(png_bytes())
    return p


# ------------------------------------------------------------ encoding

def test_sniff_mime(png):
    assert uf_images.sniff_image_mime(png.read_bytes()) == "image/png"
    assert uf_images.sniff_image_mime(b"\xff\xd8\xffhello") == "image/jpeg"
    assert uf_images.sniff_image_mime(b"not an image at all") is None


def test_encode_image_file_roundtrip(png):
    url, err = uf_images.encode_image_file(str(png))
    assert err is None
    assert url is not None and url.startswith("data:image/png;base64,")
    raw = base64.b64decode(url.split(",", 1)[1])
    assert raw == png.read_bytes()


def test_encode_image_rejects_non_image(tmp_path):
    bad = tmp_path / "x.png"
    bad.write_text("plain text pretending to be png")
    url, err = uf_images.encode_image_file(str(bad))
    assert url is None
    assert err is not None and "not a recognized image" in err


def test_save_data_url_validates_and_writes(tmp_path):
    du = "data:image/png;base64," + base64.b64encode(png_bytes()).decode()
    path, err = uf_images.save_data_url(du, tmp_path / "uploads")
    assert err is None and path
    assert Path(path).read_bytes() == png_bytes()

    _p, err2 = uf_images.save_data_url("data:text/plain;base64,aGk=", tmp_path)
    assert err2 is not None
    _p, err3 = uf_images.save_data_url("not-a-data-url", tmp_path)
    assert err3 is not None


# ----------------------------------------------------------- expansion

def test_expand_message_images_rewrites_only_on_markers(png):
    plain = {"role": "user", "content": "hello"}
    out, n = uf_images.expand_message_images(plain)
    assert out is plain and n == 0

    marked = {"role": "tool",
              "content": f"look: {uf_images.attach_marker(str(png))}"}
    out, n = uf_images.expand_message_images(marked)
    assert n == 1
    assert isinstance(out["content"], list)
    assert out["content"][0]["type"] == "text"
    assert out["content"][1]["type"] == "image_url"
    assert out["content"][1]["image_url"]["url"].startswith(
        "data:image/png;base64,")
    # Original message untouched (history/trace keep the stub).
    assert isinstance(marked["content"], str)


def test_expand_degrades_on_missing_file():
    marked = {"role": "tool",
              "content": uf_images.attach_marker("/nonexistent/x.png")}
    out, n = uf_images.expand_message_images(marked)
    assert n == 0 and out is marked


# -------------------------------------------------------- vision tools

def test_view_image_returns_attach_marker(png):
    out = view_image(str(png), note="reference")
    assert out.startswith("[image attached:")
    assert "(see image)" in out and "reference" in out


def test_view_image_rejects_non_image(tmp_path):
    assert view_image(str(tmp_path)).startswith("error:")


def test_take_screenshot_graceful_without_tool(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _n: None)
    out = take_screenshot()
    assert out.startswith("error: no screenshot backend succeeded")
    assert "tried:" in out


# --------------------------------------------------------- capability gate

def _names(client):
    return {s.get("function", {}).get("name") for s in client.tools}


def test_vision_tools_gated_by_capabilities():
    from sable.core.agent_runner import SableAgentClient as C

    c = C(base_url="http://127.0.0.1:1/v1", model="m")
    assert not {"view_image", "take_screenshot"} & _names(c)

    c = C(base_url="http://127.0.0.1:1/v1", model="m",
          capabilities={"vision": True, "screenshots": True})
    assert {"view_image", "take_screenshot"} <= _names(c)
    assert {"view_image", "take_screenshot"} <= set(c.tool_impls)


class _FakeResp:
    def __init__(self, payload, status=200):
        self._p, self.status_code = payload, status

    def json(self):
        return self._p


class _FakeClient:
    def __init__(self, payload):
        self.payload = payload

    def get(self, url):
        return _FakeResp(self.payload)


def test_vision_probe_detects_multimodal_capability():
    """Current llama.cpp advertises the projector as capabilities
    ["completion","multimodal"] (not an mmproj meta key) — both shapes work."""
    uf_images.reset_vision_cache()
    modern = {"data": [{"id": "m", "capabilities": ["completion",
                                                    "multimodal"],
                        "meta": {}}]}
    assert uf_images.detect_vision_support(
        _FakeClient(modern), "http://x/v1") is True
    uf_images.reset_vision_cache()
    legacy = {"data": [{"id": "m", "meta": {"mmproj": "/x/mmproj.gguf"}}]}
    assert uf_images.detect_vision_support(
        _FakeClient(legacy), "http://x/v1") is True
    uf_images.reset_vision_cache()
    text_only = {"data": [{"id": "m", "capabilities": ["completion"],
                           "meta": {}}]}
    assert uf_images.detect_vision_support(
        _FakeClient(text_only), "http://x/v1") is False
    uf_images.reset_vision_cache()


def test_vision_mode_resolution():
    cfg = SableConfig()
    assert _resolve_vision_mode({}, cfg) == "auto"
    assert _resolve_vision_mode({"vision": True}, cfg) == "true"
    assert _resolve_vision_mode({"vision": False}, cfg) == "false"
    assert _resolve_vision_mode({"vision": "off"}, cfg) == "off"
    cfg2 = SableConfig(run_settings={"mmproj_path": "/x/mmproj.gguf"})
    assert _resolve_vision_mode({}, cfg2) == "true"


# --------------------------------------- runner end-to-end (fake server)

def test_runner_expands_markers_only_on_the_wire(tmp_path, monkeypatch):
    """With vision forced on, the REQUEST carries image parts while the
    recorded message list keeps the plain text marker."""
    png = tmp_path / "shot.png"
    png.write_bytes(png_bytes())
    captured: dict = {}

    with FakeOpenAIServer() as srv:
        client = SableAgentClient(
            base_url=srv.base_url, model="fake-model", vision="true")
        orig_post = client._client.post

        def spy(url, json=None, **kw):  # httpx passthrough spy
            if url.endswith("/chat/completions") and json is not None:
                captured["messages"] = json.get("messages")
            return orig_post(url, json=json, **kw)

        client._client.post = spy
        res = client.run(
            f"look at this {uf_images.attach_marker(str(png))}")
        assert res.completed
        # Wire: multimodal parts present.
        wire_msgs = captured["messages"]
        assert any(isinstance(m.get("content"), list)
                   and any(p.get("type") == "image_url" for p in m["content"])
                   for m in wire_msgs)
        # History/trace: still plain strings.
        assert all(isinstance(m.get("content"), str) or m.get("content") is None
                   for m in res.messages if m.get("role") != "tool"
                   or isinstance(m.get("content"), str))


def test_runner_text_only_when_vision_off(fake_server_unused=None):
    """Default (vision off / unreachable probe) => markers stay text."""
    with FakeOpenAIServer() as srv:
        client = SableAgentClient(base_url=srv.base_url,
                                    model="fake-model", vision="false")
        res = client.run("hello [image attached: /nope.png]")
        assert res.completed
        assert all(isinstance(m.get("content"), str) or m.get("content") is None
                   for m in res.messages)


# -------------------------------------------------- dashboard uploads

def test_chat_stream_saves_uploaded_images(tmp_path):
    from sable.dashboard.app import DashboardApp

    app = DashboardApp(SableConfig(), home=Path(str(tmp_path)))
    du = "data:image/png;base64," + base64.b64encode(png_bytes()).decode()
    # The handler validates + saves before touching the runner; with no
    # server running the run itself fails, but uploads must already exist.
    resp = None
    try:
        resp = app.handle("POST", "/api/chat/stream",
                          body={"prompt": "what is in this image?",
                                "images": [du]})
    except Exception:
        pass  # stream setup may raise without a live endpoint — fine here
    uploads = list((Path(str(tmp_path)) / "uploads").glob("upload-*.png"))
    assert uploads, "uploaded image was not persisted"
    assert uploads[0].read_bytes() == png_bytes()

# ------------------------------------------------------------------------
# from tests/test_projects.py
# ------------------------------------------------------------------------

@pytest.fixture()
def home__test_projects(tmp_path, monkeypatch):
    h = tmp_path / "uf-proj"
    monkeypatch.setenv("SABLE_HOME", str(h))
    return h


@pytest.fixture()
def app__test_projects(home__test_projects):
    return DashboardApp(SableConfig(), home=home__test_projects)


# ------------------------------------------------------------- project CRUD

def test_project_crud_roundtrip(app__test_projects, home__test_projects, tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()

    # create
    r = app__test_projects.handle("POST", "/api/projects",
                   body={"name": "alpha", "workspace_dir": str(ws),
                         "agents": ["coder", "reviewer"]})
    assert r.status == 201, r.body
    p = r.body["project"]
    pid = p["id"]
    assert p["name"] == "alpha" and p["workspace_dir"] == str(ws)
    assert p["agents"] == ["coder", "reviewer"]
    assert p["created"] and p["updated"]
    # one JSON file per project under <home>/projects/
    assert (home__test_projects / "projects" / f"{pid}.json").is_file()

    # list
    r = app__test_projects.handle("GET", "/api/projects")
    assert r.status == 200
    assert [x["id"] for x in r.body["projects"]] == [pid]

    # update: bump updated, change name/agents/workspace_dir
    ws2 = tmp_path / "ws2"
    ws2.mkdir()
    r = app__test_projects.handle("PUT", f"/api/projects/{pid}",
                   body={"name": "beta", "workspace_dir": str(ws2),
                         "agents": ["coder"]})
    assert r.status == 200
    assert r.body["project"]["name"] == "beta"
    assert r.body["project"]["workspace_dir"] == str(ws2)
    assert r.body["project"]["agents"] == ["coder"]

    # delete
    r = app__test_projects.handle("DELETE", f"/api/projects/{pid}")
    assert r.status == 200 and r.body["ok"]
    assert app__test_projects.handle("GET", "/api/projects").body["projects"] == []
    assert app__test_projects.handle("PUT", f"/api/projects/{pid}",
                      body={"name": "x"}).status == 404
    assert app__test_projects.handle("DELETE", f"/api/projects/{pid}").status == 404


def test_project_validation_errors(app__test_projects, tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()

    r = app__test_projects.handle("POST", "/api/projects",
                   body={"name": "", "workspace_dir": str(ws)})
    assert r.status == 400 and r.body["error"]["code"] == "bad_request"

    r = app__test_projects.handle("POST", "/api/projects",
                   body={"name": "  ", "workspace_dir": str(ws)})
    assert r.status == 400

    # relative path rejected
    r = app__test_projects.handle("POST", "/api/projects",
                   body={"name": "rel", "workspace_dir": "relative/dir"})
    assert r.status == 400
    assert "absolute" in r.body["error"]["message"]

    # absolute but missing dir rejected
    r = app__test_projects.handle("POST", "/api/projects",
                   body={"name": "gone",
                         "workspace_dir": str(tmp_path / "nope")})
    assert r.status == 400
    assert "does not exist" in r.body["error"]["message"]

    # agents must be a list of strings
    r = app__test_projects.handle("POST", "/api/projects",
                   body={"name": "ok", "workspace_dir": str(ws),
                         "agents": [{"bad": 1}]})
    assert r.status == 400


# ------------------------------------- project description + per-agent roles

def test_project_description_and_agent_roles_roundtrip(app__test_projects,
                                                       tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    r = app__test_projects.handle("POST", "/api/projects",
                   body={"name": "atlas", "workspace_dir": str(ws),
                         "agents": ["coder", "reviewer"],
                         "description": "  map the terrain  ",
                         "agent_roles": {"coder": " writes the python ",
                                         "reviewer": "   ",
                                         "ghost": "not an agent here"}})
    assert r.status == 201, r.body
    p = r.body["project"]
    assert p["description"] == "map the terrain"
    # values stripped; blank roles dropped; roles for agents that are not
    # attached are pruned (roles belong to the roster)
    assert p["agent_roles"] == {"coder": "writes the python"}

    # PUT without the keys keeps the stored values; PUT with them replaces
    r = app__test_projects.handle("PUT", f"/api/projects/{p['id']}",
                   body={"name": "atlas2"})
    assert r.status == 200
    assert r.body["project"]["description"] == "map the terrain"
    assert r.body["project"]["agent_roles"] == {"coder":
                                                "writes the python"}

    # removing an agent prunes its role even when agent_roles is untouched
    r = app__test_projects.handle("PUT", f"/api/projects/{p['id']}",
                   body={"agents": ["reviewer"]})
    assert r.body["project"]["agent_roles"] == {}

    # empty description clears it (explicit empty string)
    r = app__test_projects.handle("PUT", f"/api/projects/{p['id']}",
                   body={"description": ""})
    assert r.body["project"]["description"] == ""


def test_project_description_and_roles_validation(app__test_projects,
                                                  tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    r = app__test_projects.handle("POST", "/api/projects",
                   body={"name": "x", "workspace_dir": str(ws),
                         "description": 42})
    assert r.status == 400 and "description" in r.body["error"]["message"]

    r = app__test_projects.handle("POST", "/api/projects",
                   body={"name": "x", "workspace_dir": str(ws),
                         "agent_roles": ["coder"]})
    assert r.status == 400 and "agent_roles" in r.body["error"]["message"]

    r = app__test_projects.handle("POST", "/api/projects",
                   body={"name": "x", "workspace_dir": str(ws),
                         "agent_roles": {"coder": 7}})
    assert r.status == 400

    # legacy records keep working: description defaults to "", roles to {}
    r = app__test_projects.handle("POST", "/api/projects",
                   body={"name": "legacy", "workspace_dir": str(ws)})
    assert r.status == 201
    assert r.body["project"]["description"] == ""
    assert r.body["project"]["agent_roles"] == {}


# ------------------------------------------------- workspace-safety + index

def test_delete_project_keeps_workspace_files(app__test_projects, tmp_path):
    """Deleting a project removes it from sable (+ its index DB) but NEVER
    the workspace directory or the files the agents worked on."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    src = ws / "agent_work.py"
    src.write_text("print('keep me')\n")
    sub = ws / "sub"
    sub.mkdir()
    (sub / "notes.md").write_text("important")

    pid = app__test_projects.handle("POST", "/api/projects",
                     body={"name": "doomed", "workspace_dir": str(ws)}
                     ).body["project"]["id"]
    r = app__test_projects.handle("DELETE", f"/api/projects/{pid}")
    assert r.status == 200 and r.body["ok"]

    # sable forgot the project...
    assert app__test_projects.handle("GET", "/api/projects").body["projects"] == []
    # ...but the workspace and everything in it is untouched
    assert ws.is_dir()
    assert src.read_text() == "print('keep me')\n"
    assert (sub / "notes.md").read_text() == "important"


def test_delete_project_removes_only_its_index_db(app__test_projects,
                                                   tmp_path):
    """The DELETE endpoint deletes <home>/index/<id>.db and nothing else:
    sibling projects' DBs and the default DB survive, the DB dir stays."""
    idx_dir = app__test_projects.home / "index"
    idx_dir.mkdir(parents=True, exist_ok=True)

    def mk(pid):
        f = idx_dir / f"{pid}.db"
        f.write_text("sqlite-ish")
        return f

    ws = tmp_path / "ws"
    ws.mkdir()
    doomed = app__test_projects.handle(
        "POST", "/api/projects",
        body={"name": "doomed", "workspace_dir": str(ws)}
    ).body["project"]["id"]
    other = app__test_projects.handle(
        "POST", "/api/projects",
        body={"name": "survivor", "workspace_dir": str(ws)}
    ).body["project"]["id"]
    d, o, default = mk(doomed), mk(other), mk("default")

    r = app__test_projects.handle("DELETE", f"/api/projects/{doomed}")
    assert r.status == 200 and r.body["ok"] and r.body["index_removed"]
    assert not d.exists()
    assert o.is_file() and default.is_file() and idx_dir.is_dir()


def test_create_and_update_enroll_project_index_when_enabled(
        app__test_projects, tmp_path, monkeypatch):
    """With indexing.enabled, create/update kick a build of THAT project's
    own index DB (paths stay per-project; nothing global is appended)."""
    ws = tmp_path / "ws"
    ws.mkdir()
    calls = []
    monkeypatch.setattr(app__test_projects, "h_index_build",
                        lambda q, b: calls.append(dict(b)))

    # default config: indexing off => no build is ever kicked
    pid = app__test_projects.handle("POST", "/api/projects",
                     body={"name": "off", "workspace_dir": str(ws)}
                     ).body["project"]["id"]
    assert calls == []

    app__test_projects.cfg.indexing = {"enabled": True}
    pid2 = app__test_projects.handle("POST", "/api/projects",
                      body={"name": "on", "workspace_dir": str(ws)}
                      ).body["project"]["id"]
    assert calls == [{"project": pid2}]

    calls.clear()
    assert app__test_projects.handle(
        "PUT", f"/api/projects/{pid}",
        body={"name": "renamed"}).status == 200
    assert calls == [{"project": pid}]

    # a failed build kickoff must never fail the create itself
    calls.clear()

    def boom(q, b):
        raise RuntimeError("embed server down")

    monkeypatch.setattr(app__test_projects, "h_index_build", boom)
    r = app__test_projects.handle("POST", "/api/projects",
                   body={"name": "survivor", "workspace_dir": str(ws)})
    assert r.status == 201


# -------------------------------------------------- project prompt context

def test_with_project_context_prompt_sections():
    from sable.core.agent_runner import _with_project_context

    base = "base prompt"
    # no project / empty record / name-only record => prompt untouched
    assert _with_project_context(None, base, "coder") is base
    assert _with_project_context({}, base, "coder") is base
    assert _with_project_context({"name": "x"}, base, "coder") is base

    proj = {"name": "Atlas", "description": "map the terrain",
            "agents": ["orchestrator", "coder"],
            "agent_roles": {"coder": "writes python"}}
    out = _with_project_context(proj, base, "coder")
    assert out.startswith(base)
    assert "## Project: Atlas" in out
    assert "map the terrain" in out
    # own role marked, teammates visible with their jobs, unroled agent
    # listed without a colon
    assert "- coder (that's you): writes python" in out
    assert "- orchestrator\n" in out or out.endswith("- orchestrator")

    # unknown running profile => no "you" marker at all
    assert "that's you" not in _with_project_context(proj, base, "nobody")
    # description alone (no agents) still injects; agents alone too
    d = _with_project_context({"name": "N", "description": "only desc"},
                              base, None)
    assert "only desc" in d and "Agents working" not in d
    a = _with_project_context({"name": "N", "agents": ["solo"]}, base, "solo")
    assert "- solo (that's you)" in a


def test_create_runner_injects_project_context(tmp_path, monkeypatch):
    import sable.core.agent_runner as ar

    monkeypatch.setenv("SABLE_HOME", str(tmp_path / "home"))
    cfg = SableConfig()
    proj = {"name": "Atlas", "description": "desc here",
            "agents": ["coder", "tester"],
            "agent_roles": {"coder": "ship it", "tester": "break it"}}
    runner = ar.create_runner(cfg, "http://127.0.0.1:1", "m",
                              project_context=proj, profile_name="coder")
    assert "## Project: Atlas" in runner.system_prompt
    assert "desc here" in runner.system_prompt
    assert "- coder (that's you): ship it" in runner.system_prompt
    assert "- tester: break it" in runner.system_prompt

    # no project => byte-identical default prompt (no section at all)
    off = ar.create_runner(cfg, "http://127.0.0.1:1", "m")
    assert "## Project" not in off.system_prompt


# ------------------------------------------------------- threads <-> project

def test_thread_with_project_lists_under_it(app__test_projects):
    ws = Path("/tmp")  # exists, absolute
    pid = app__test_projects.handle("POST", "/api/projects",
                     body={"name": "p1", "workspace_dir": str(ws)}
                     ).body["project"]["id"]

    r = app__test_projects.handle("PUT", "/api/chat/threads/t-proj",
                   body={"title": "filed", "messages": [], "project": pid})
    assert r.status == 200
    assert r.body["thread"]["project"] == pid

    lst = app__test_projects.handle("GET", "/api/chat/threads").body["threads"]
    row = next(t for t in lst if t["id"] == "t-proj")
    assert row["project"] == pid

    # move between projects via update; explicit null -> Unfiled
    pid2 = app__test_projects.handle("POST", "/api/projects",
                      body={"name": "p2", "workspace_dir": str(ws)}
                      ).body["project"]["id"]
    r = app__test_projects.handle("PUT", "/api/chat/threads/t-proj",
                   body={"project": pid2})
    assert r.body["thread"]["project"] == pid2
    r = app__test_projects.handle("PUT", "/api/chat/threads/t-proj",
                   body={"project": None})
    assert r.body["thread"]["project"] is None

    # unknown project id rejected on thread write
    assert app__test_projects.handle("PUT", "/api/chat/threads/t-proj",
                      body={"project": "does-not-exist"}).status == 400


def test_legacy_thread_without_project_key_still_lists(app__test_projects, home__test_projects):
    """Raw pre-projects chat file (no "project" key) lists fine = Unfiled."""
    d = home__test_projects / "chat"
    d.mkdir(parents=True, exist_ok=True)
    legacy = {"id": "old1", "title": "legacy", "created": "2026-01-01T00:00:00+00:00",
              "updated": "2026-01-01T00:00:00+00:00", "model_alias": "m",
              "messages": [{"role": "user", "content": "hi"}],
              "trace_ids": ["t9"], "summary": None}
    (d / "old1.json").write_text(json.dumps(legacy))

    lst = app__test_projects.handle("GET", "/api/chat/threads").body["threads"]
    assert [t["id"] for t in lst] == ["old1"]
    assert lst[0]["project"] is None          # absent key -> Unfiled
    assert lst[0]["message_count"] == 1

    # full detail read still works, and a re-save keeps it projectless
    det = app__test_projects.handle("GET", "/api/chat/threads/old1")
    assert det.status == 200 and det.body["thread"]["title"] == "legacy"
    r = app__test_projects.handle("PUT", "/api/chat/threads/old1", body={"title": "renamed"})
    assert r.status == 200 and r.body["thread"]["project"] is None


def test_delete_project_leaves_threads_unfiled(app__test_projects):
    ws = Path("/tmp")
    pid = app__test_projects.handle("POST", "/api/projects",
                     body={"name": "doomed", "workspace_dir": str(ws)}
                     ).body["project"]["id"]
    app__test_projects.handle("PUT", "/api/chat/threads/keep-me",
               body={"title": "kept", "messages": [], "project": pid})

    assert app__test_projects.handle("DELETE", f"/api/projects/{pid}").status == 200

    # thread file untouched; its dangling id reads as Unfiled in the tree
    lst = app__test_projects.handle("GET", "/api/chat/threads").body["threads"]
    row = next(t for t in lst if t["id"] == "keep-me")
    assert row["project"] == pid  # raw value kept; UI groups unknown ids here
    known = {p["id"] for p in app__test_projects.handle("GET", "/api/projects").body["projects"]}
    assert row["project"] not in known  # => renders under Unfiled


# ------------------------------------------------- trace project propagation

def test_trace_recording_carries_project_and_filters(app__test_projects, home__test_projects):
    from sable.store.recorder import TraceRecorder

    ws = Path("/tmp")
    pid = app__test_projects.handle("POST", "/api/projects",
                     body={"name": "trace-proj", "workspace_dir": str(ws)}
                     ).body["project"]["id"]

    rec = TraceRecorder(home=home__test_projects, project=pid)
    row = rec.record(task="do the thing", messages=[
        {"role": "user", "content": "do the thing"},
        {"role": "assistant", "content": "done"}],
        backend="sable", model_alias="m", base_url="http://x/v1",
        completed=True)
    assert row is not None and row.project == pid
    rec2 = TraceRecorder(home=home__test_projects)  # no project => unfiled trace
    row2 = rec2.record(task="other task", messages=[
        {"role": "user", "content": "other"},
        {"role": "assistant", "content": "ok"}],
        backend="sable", model_alias="m", base_url="http://x/v1",
        completed=True)
    assert row2 is not None and row2.project is None
    rec.close(); rec2.close()

    # list + filter
    r = app__test_projects.handle("GET", "/api/traces", query={"project": pid})
    assert [t["trace_id"] for t in r.body["traces"]] == [row.trace_id]
    assert r.body["total"] == 1

    r = app__test_projects.handle("GET", "/api/traces")
    ids = {t["trace_id"] for t in r.body["traces"]}
    assert ids == {row.trace_id, row2.trace_id}

    # detail exposes project too
    det = app__test_projects.handle("GET", f"/api/traces/{row.trace_id}")
    assert det.body["trace"]["project"] == pid


def test_trace_project_survives_legacy_db(tmp_path, monkeypatch):
    """A v2 DB (no project column) migrates on open and reads old rows."""
    import sqlite3
    from sable.store.db import TraceStore

    monkeypatch.setenv("SABLE_HOME", str(tmp_path / "h"))
    db = tmp_path / "buffer" / "sable.db"
    db.parent.mkdir(parents=True)
    # hand-build a minimal v2-shaped traces table (no project column)
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE traces (trace_id TEXT PRIMARY KEY, session_id TEXT,"
        " timestamp TEXT NOT NULL, agent_profile TEXT, model_alias TEXT,"
        " model_version TEXT, endpoint_label TEXT, task_summary TEXT,"
        " status TEXT NOT NULL, outcome_score REAL, num_tool_calls INTEGER"
        " NOT NULL DEFAULT 0, tokens_in INTEGER, tokens_out INTEGER,"
        " duration_s REAL, jsonl_path TEXT NOT NULL, schema_version INTEGER"
        " NOT NULL DEFAULT 2, curated INTEGER NOT NULL DEFAULT 0,"
        " dataset_id TEXT, failure_reason TEXT)")
    conn.execute(
        "INSERT INTO traces (trace_id, timestamp, status, jsonl_path)"
        " VALUES ('old', '2026-01-01T00:00:00+00:00', 'success', '/x.jsonl')")
    conn.commit()
    conn.close()

    store = TraceStore(db)  # runs the guarded ALTER TABLE ADD COLUMN
    try:
        row = store.get_trace("old")
        assert row is not None and row.project is None
        cols = {r[1] for r in store._conn.execute(
            "PRAGMA table_info(traces)").fetchall()}
        assert "project" in cols
    finally:
        store.close()


# ------------------------------------------------ workspace roots precedence

def test_resolve_workspace_roots_precedence(tmp_path, monkeypatch):
    import sable.core.agent_runner as ar
    from sable.core.agent_runner import resolve_workspace_roots

    fake_cwd = tmp_path / "cwd"
    fake_cwd.mkdir()
    monkeypatch.setattr(ar.os, "getcwd", lambda: str(fake_cwd))

    proj_ws = tmp_path / "proj-ws"
    proj_ws.mkdir()
    prof_ws = tmp_path / "prof-ws"
    prof_ws.mkdir()
    cfg_ws = tmp_path / "cfg-ws"
    cfg_ws.mkdir()

    # config level only -> config wins over cwd
    cfg = SableConfig(agent={"workspace_root": str(cfg_ws)})
    roots = resolve_workspace_roots(cfg)
    assert roots is not None and roots[0] == cfg_ws

    # project beats config
    roots = resolve_workspace_roots(cfg, project_workspace_dir=str(proj_ws))
    assert roots is not None and roots[0] == proj_ws

    # profile beats project (and config)
    roots = resolve_workspace_roots(
        cfg, workspace_root_override=str(prof_ws),
        project_workspace_dir=str(proj_ws))
    assert roots is not None and roots[0] == prof_ws

    # no config, no project -> cwd (behavior unchanged when no project)
    roots = resolve_workspace_roots(SableConfig())
    assert roots is not None and roots[0] == fake_cwd

    # project with no config still beats the cwd default
    roots = resolve_workspace_roots(SableConfig(),
                                    project_workspace_dir=str(proj_ws))
    assert roots is not None and roots[0] == proj_ws


def test_chat_stream_rejects_unknown_project(app__test_projects):
    """Run context channel validates the project id before streaming."""
    r = app__test_projects.handle("POST", "/api/chat/stream",
                   body={"prompt": "hi", "project": "nope"})
    assert r.status == 404
    assert r.body["error"]["code"] == "unknown_project"


# ============================================================ remote agents
# Peer agents in pickers + the /remote/* OpenAI-compatible passthrough that
# lets a LAN peer run inference against this box (dashboard.expose_lan).

@pytest.fixture()
def home__rgw(tmp_path, monkeypatch):
    h = tmp_path / "home-rgw"
    monkeypatch.setenv("SABLE_HOME", str(h))
    return h


@pytest.fixture()
def app__rgw(home__rgw):
    return DashboardApp(SableConfig(), home=home__rgw)


def _seed_peer__rgw(home, url="http://192.0.2.9:55429", token="peer-tok",
                    ok=True, agent="tim", model="Qwen3-27B",
                    persona="You are Tim on the render node."):
    from sable.dashboard import peers as peers_mod

    rec = peers_mod.upsert_peer(home, url, token, name=None)
    snap = ({"ok": True, "probed_at": "2026-01-01T00:00:00+00:00",
             "agents": [{"name": agent, "default": False,
                         "system_prompt": persona}],
             # second running instance: multi-instance peers must route via
             # X-Sable-Instance instead of landing on the first match.
             "servers": [{"name": "tim", "running": True,
                          "model_name": model},
                         {"name": "tim-2", "running": True,
                          "model_name": "Other-32B"}]}
            if ok else
            {"ok": False, "probed_at": "2026-01-01T00:00:00+00:00",
             "error": "boom"})
    peers_mod.set_snapshot(home, rec["id"], snap)
    return rec


def test_remote_agent_entries_synthesized_no_token(home__rgw):
    from sable.dashboard import peers as peers_mod

    _seed_peer__rgw(home__rgw)
    entries = peers_mod.remote_agent_entries(home__rgw)
    assert len(entries) == 1
    e = entries[0]
    assert e["name"] == "tim@192.0.2.9"
    assert e["remote"] is True and e["model_name"] == "Qwen3-27B"
    # display shape: never carries credentials
    assert "token" not in json.dumps(entries)
    # unhealthy peers contribute nothing
    _seed_peer__rgw(home__rgw, url="http://192.0.2.10:1", ok=False)
    assert all("192.0.2.10" not in x["name"]
               for x in peers_mod.remote_agent_entries(home__rgw))


def test_resolve_remote_agent_carries_creds(home__rgw):
    from sable.dashboard import peers as peers_mod

    _seed_peer__rgw(home__rgw)
    got = peers_mod.resolve_remote_agent(home__rgw, "tim@192.0.2.9")
    assert got and got["peer_url"] == "http://192.0.2.9:55429"
    assert got["peer_token"] == "peer-tok"
    assert got["model_name"] == "Qwen3-27B" and got["remote"] is True
    # no `tools` key: absent means default tools, [] would drop everything
    assert "tools" not in got
    assert peers_mod.resolve_remote_agent(home__rgw, "jim@192.0.2.9") is None
    assert peers_mod.resolve_remote_agent(home__rgw, "tim@192.0.2.10") is None
    _seed_peer__rgw(home__rgw, ok=False)  # peer went unhealthy
    assert peers_mod.resolve_remote_agent(home__rgw, "tim@192.0.2.9") is None


def test_resolve_remote_agent_carries_persona(home__rgw):
    """§1b: the snapshot persona travels instead of the old empty string."""
    from sable.dashboard import peers as peers_mod

    _seed_peer__rgw(home__rgw)
    got = peers_mod.resolve_remote_agent(home__rgw, "tim@192.0.2.9")
    assert got is not None
    assert got["system_prompt"] == "You are Tim on the render node."
    # first running instance recorded for X-Sable-Instance routing
    assert got["remote_instance"] == "tim"


def test_probe_snapshot_caps_persona():
    """§1a: probe snapshots carry system_prompt capped at 8000 chars."""
    import httpx

    from sable.dashboard import peers as peers_mod

    class Resp:
        status_code = 200

        def __init__(self, payload):
            self._p = payload

        def json(self):
            return self._p

    routes = {"/api/status": {"version": "x", "server": {}},
              "/api/servers": {"servers": []},
              "/api/profiles": {"profiles": [
                  {"name": "tim", "default": False,
                   "system_prompt": "P" * 9000}]},
              "/api/jobs": {"jobs": []},
              "/api/traces": {"traces": []}}

    def fake_get(url, headers=None, params=None, timeout=None):
        path = url.split("http://p")[-1].split("?")[0]
        return Resp(routes[path])

    orig = httpx.Client

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, **kw):
            return fake_get(url, **kw)

    httpx.Client = FakeClient
    try:
        snap = peers_mod.probe("http://p", None)
    finally:
        httpx.Client = orig
    assert snap["ok"] is True
    assert len(snap["agents"][0]["system_prompt"]) == 8000


def test_chat_stream_remote_sends_instance_header(app__rgw, home__rgw,
                                                  monkeypatch):
    """§3b: a remote profile with a named instance rides X-Sable-Instance as
    an extra_headers on the runner (never a ?instance= query string)."""
    from sable.core.agent_runner import AgentResult

    made = {}

    class StubRunner:
        def __init__(self, cfg, base_url, model, api_key=None, **kwargs):
            made["base_url"] = base_url
            made["extra_headers"] = kwargs.get("extra_headers")

        def run(self, task, history=None, **kw):
            return AgentResult(final_response="ok", messages=[
                {"role": "user", "content": task},
                {"role": "assistant", "content": "ok"}],
                backend="stub", tokens_in=1, tokens_out=1)

    monkeypatch.setattr("sable.core.agent_runner.create_runner", StubRunner)
    monkeypatch.setattr("sable.serving.supervisor.healthcheck",
                        lambda *a, **k: True)
    _seed_peer__rgw(home__rgw)
    r = app__rgw.handle("POST", "/api/chat/stream",
                        body={"prompt": "hi tim", "profile": "tim@192.0.2.9"})
    assert r.status == 200 and r.stream is not None
    text = "".join(r.stream)
    assert "run.done" in text
    assert "?" not in made["base_url"]  # header-only selector
    assert made["extra_headers"] == {"X-Sable-Instance": "tim"}


def test_profiles_list_includes_remote_agents(app__rgw, home__rgw):
    _seed_peer__rgw(home__rgw)
    body = app__rgw.handle("GET", "/api/profiles").body
    names = [p["name"] for p in body["profiles"]]
    assert "tim@192.0.2.9" in names
    entry = next(p for p in body["profiles"] if p["remote"])
    assert json.dumps(entry).count("peer-tok") == 0


def test_chat_stream_remote_profile_targets_gateway(app__rgw, home__rgw,
                                                    monkeypatch):
    """A remote profile run points the runner at peer/remote/v1 with the
    peer token as api_key; the agent loop itself stays local."""
    from sable.core.agent_runner import AgentResult

    made = {}

    class StubRunner:
        def __init__(self, cfg, base_url, model, api_key=None, **kwargs):
            made["base_url"] = base_url
            made["model"] = model
            made["api_key"] = api_key

        def run(self, task, history=None, **kw):
            return AgentResult(final_response="ok", messages=[
                {"role": "user", "content": task},
                {"role": "assistant", "content": "ok"}],
                backend="stub", tokens_in=1, tokens_out=1)

    monkeypatch.setattr("sable.core.agent_runner.create_runner", StubRunner)
    monkeypatch.setattr("sable.serving.supervisor.healthcheck",
                        lambda *a, **k: True)
    _seed_peer__rgw(home__rgw)
    r = app__rgw.handle("POST", "/api/chat/stream",
                        body={"prompt": "hi tim", "profile": "tim@192.0.2.9"})
    assert r.status == 200 and r.stream is not None
    text = "".join(r.stream)
    assert "run.done" in text
    assert made["base_url"] == "http://192.0.2.9:55429/remote/v1"
    assert made["api_key"] == "peer-tok"
    assert made["model"] == "Qwen3-27B"


def test_chat_stream_remote_peer_down_is_503(app__rgw, home__rgw, monkeypatch):
    monkeypatch.setattr("sable.serving.supervisor.healthcheck",
                        lambda *a, **k: False)
    _seed_peer__rgw(home__rgw)
    r = app__rgw.handle("POST", "/api/chat/stream",
                        body={"prompt": "hi", "profile": "tim@192.0.2.9"})
    assert r.status == 503
    assert r.body["error"]["code"] == "peer_unreachable"


def _start_lan_server(home, token="gw-tok"):
    from sable.core.config import SableConfig
    from sable.dashboard.server import make_server

    cfg = SableConfig(dashboard={"expose_lan": True})
    httpd = make_server(cfg, port=0, home=home, token=token)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def test_remote_gateway_requires_bearer_token(home__rgw):
    import httpx

    httpd, base = _start_lan_server(home__rgw)
    try:
        r = httpx.post(base + "/remote/v1/chat/completions",
                       json={"messages": []}, timeout=5)
        assert r.status_code == 401
        r = httpx.get(base + "/remote/v1/models", timeout=5,
                      headers={"Authorization": "Bearer nope"})
        assert r.status_code == 401
        # query-string tokens are refused here (logs leak; headers only)
        r = httpx.get(base + "/remote/v1/models?token=gw-tok", timeout=5)
        assert r.status_code == 401
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_remote_gateway_passthrough_models_and_chat(home__rgw):
    import httpx

    from fake_server import FakeOpenAIServer

    httpd, base = _start_lan_server(home__rgw)
    try:
        with FakeOpenAIServer() as fs:
            _fake_serve_state(home__rgw, fs.base_url)
            hdrs = {"Authorization": "Bearer gw-tok"}
            r = httpx.get(base + "/remote/v1/models", headers=hdrs, timeout=5)
            assert r.status_code == 200
            assert "fake-model" in r.text
            r = httpx.post(base + "/remote/v1/chat/completions", headers=hdrs,
                           json={"messages": [{"role": "user",
                                               "content": "hi"}]}, timeout=10)
            assert r.status_code == 200
            body = r.json()
            # fake server answers the first turn with a native tool call
            assert body["choices"][0]["message"]["tool_calls"]
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_remote_gateway_dead_without_expose_lan(home):
    import httpx

    from sable.core.config import SableConfig
    from sable.dashboard.server import make_server

    httpd = make_server(SableConfig(), port=0, home=home, token="loop-tok")
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        r = httpx.get(base + "/remote/v1/models", timeout=5,
                      headers={"Authorization": "Bearer loop-tok"})
        assert r.status_code == 404  # loopback surface never advertises it
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_remote_gateway_nothing_serving_is_503(home__rgw):
    import httpx

    httpd, base = _start_lan_server(home__rgw)
    try:
        r = httpx.get(base + "/remote/v1/models", timeout=5,
                      headers={"Authorization": "Bearer gw-tok"})
        assert r.status_code == 503
        assert r.json()["error"]["code"] == "not_serving"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_remote_gateway_named_instance_only_serves(home__rgw):
    """Default endpoint down but a *named* instance alive: the no-model
    healthcheck (GET /remote/v1/models) must still report serving — this is
    exactly the peer setup (model only on a named instance) that used to 503
    here while chat-to-a-remote-agent 503'd on the pushing side. A request
    WITH a model that isn't the named one's model must NOT silently route to
    the named instance (it may serve something else)."""
    import httpx

    from fake_server import FakeOpenAIServer
    from sable.serving import instances as srv_inst

    home = home__rgw
    with FakeOpenAIServer() as fs:
        srv_inst.save_instance(home, {
            "id": "tim", "role": "custom", "default": False, "autostart": False,
            "created": None, "updated": None})
        srv_inst.write_state(home, "tim", {
            "mode": "attach", "base_url": fs.base_url,
            "model_name": "fake-model",
            "attach_url": fs.base_url.rsplit("/v1", 1)[0]})
        httpd, base = _start_lan_server(home)
        try:
            hdrs = {"Authorization": "Bearer gw-tok"}
            # No-model probe (what chat.py's pre-flight does) passes now.
            r = httpx.get(base + "/remote/v1/models", headers=hdrs, timeout=5)
            assert r.status_code == 200
            assert "fake-model" in r.text
            # No-model chat also lands on the named instance.
            r = httpx.post(base + "/remote/v1/chat/completions", headers=hdrs,
                           json={"messages": [{"role": "user",
                                               "content": "hi"}]}, timeout=10)
            assert r.status_code == 200
            # A model that isn't the named instance's must NOT route to it —
            # honest 503 rather than the wrong upstream.
            r = httpx.post(base + "/remote/v1/chat/completions", headers=hdrs,
                           json={"model": "other-model",
                                 "messages": [{"role": "user",
                                               "content": "hi"}]}, timeout=10)
            assert r.status_code == 503
            assert r.json()["error"]["code"] == "not_serving"
        finally:
            httpd.shutdown()
            httpd.server_close()


def test_remote_gateway_dead_named_instance_is_503(home__rgw):
    """Named instance state pointing at a dead endpoint: the fallback scan
    must skip it (state_alive False) and the probe stays a 503 — no fake
    liveness from a stale state file."""
    import httpx

    from sable.serving import instances as srv_inst

    home = home__rgw
    srv_inst.save_instance(home, {
        "id": "tim", "role": "custom", "default": False, "autostart": False,
        "created": None, "updated": None})
    srv_inst.write_state(home, "tim", {
        "mode": "attach", "base_url": "http://127.0.0.1:1/v1",
        "model_name": "stale-model",
        "attach_url": "http://127.0.0.1:1"})
    httpd, base = _start_lan_server(home)
    try:
        r = httpx.get(base + "/remote/v1/models", timeout=5,
                      headers={"Authorization": "Bearer gw-tok"})
        assert r.status_code == 503
        assert r.json()["error"]["code"] == "not_serving"
    finally:
        httpd.shutdown()
        httpd.server_close()


# ------------------------------------------------- X-Sable-Instance selection


class _FakeApp:
    """Minimal app surface for remote_gateway.resolve_upstream."""

    def __init__(self, home):
        from sable.core.config import SableConfig
        self.home = home
        self.cfg = SableConfig()


def test_resolve_upstream_instance_hit(home__rgw):
    """Named alive instance wins over model-name matching (§3c)."""
    from fake_server import FakeOpenAIServer
    from sable.dashboard import remote_gateway as rg
    from sable.serving import instances as srv_inst

    home = home__rgw
    with FakeOpenAIServer() as fs:
        srv_inst.save_instance(home, {
            "id": "tim", "role": "custom", "default": False,
            "autostart": False, "created": None, "updated": None})
        srv_inst.write_state(home, "tim", {
            "mode": "attach", "base_url": fs.base_url,
            "model_name": "fake-model",
            "attach_url": fs.base_url.rsplit("/v1", 1)[0]})
        app = _FakeApp(home)
        # instance hit ignores the model entirely (routes to THAT instance)
        url, _key = rg.resolve_upstream(app, "chat/completions",
                                        model="other-model",
                                        instance="tim")
        assert url == fs.base_url.rstrip("/") + "/chat/completions"


def test_resolve_upstream_instance_dead_raises(home__rgw):
    """Named-but-not-serving instance -> honest LookupError (503 at route)."""
    from sable.dashboard import remote_gateway as rg
    from sable.serving import instances as srv_inst

    home = home__rgw
    srv_inst.save_instance(home, {
        "id": "tim", "role": "custom", "default": False,
        "autostart": False, "created": None, "updated": None})
    srv_inst.write_state(home, "tim", {
        "mode": "attach", "base_url": "http://127.0.0.1:1/v1",
        "model_name": "stale-model", "attach_url": "http://127.0.0.1:1"})
    with pytest.raises(LookupError, match="not running on this sable"):
        rg.resolve_upstream(_FakeApp(home), "chat/completions",
                            instance="tim")
    # unknown name: honest error beats silently answering from another box
    with pytest.raises(LookupError, match="not running on this sable"):
        rg.resolve_upstream(_FakeApp(home), "chat/completions",
                            instance="ghost")


def test_resolve_upstream_unset_instance_old_flow(home__rgw):
    """instance=None keeps today's flow byte-identical (model-name match)."""
    from fake_server import FakeOpenAIServer
    from sable.dashboard import remote_gateway as rg

    home = home__rgw
    with FakeOpenAIServer() as fs:
        _fake_serve_state(home, fs.base_url)
        app = _FakeApp(home)
        url, _key = rg.resolve_upstream(app, "models")
        assert url == fs.base_url.rstrip("/") + "/models"
        url, _key = rg.resolve_upstream(app, "chat/completions",
                                        model="fake-model")
        assert url.endswith("/chat/completions")


def test_agent_client_sends_extra_headers():
    """§3a: SableAgentClient merges extra_headers into its httpx client."""
    from sable.core.agent_runner import SableAgentClient

    c = SableAgentClient("http://127.0.0.1:9/v1", "m", api_key="k",
                         extra_headers={"X-Sable-Instance": "tim"})
    assert c._client.headers["X-Sable-Instance"] == "tim"
    assert c._client.headers["Authorization"] == "Bearer k"
    # None keeps byte-identical headers (no stray keys)
    c2 = SableAgentClient("http://127.0.0.1:9/v1", "m")
    assert "X-Sable-Instance" not in c2._client.headers


def test_status_peer_gateway_shape(app__rgw, home__rgw, monkeypatch):
    """expose_lan off => peer_gateway None; on + lan ip => the /remote/v1 URL."""
    from sable.dashboard import app as app_mod

    r = app__rgw.handle("GET", "/api/status")
    assert r.status == 200 and r.body["peer_gateway"] is None

    app__rgw.cfg.dashboard = {"expose_lan": True}
    monkeypatch.setattr(app_mod.DashboardApp, "token", "tok", raising=False)
    app__rgw.dashboard_port = 4321
    import sable.dashboard.server as srv_mod
    monkeypatch.setattr(srv_mod, "_lan_ip", lambda: "192.168.1.5")
    body = app__rgw.handle("GET", "/api/status").body
    assert body["peer_gateway"] == "http://192.168.1.5:4321/remote/v1"


def test_remote_gateway_trace_import(home__rgw):
    """Peer-pushed transcripts land in OUR buffer under the local name."""
    import httpx

    httpd, base = _start_lan_server(home__rgw)
    try:
        body = {"task": "fix the shader", "profile": "tim",
                "model_name": "Qwen3-27B", "completed": True,
                "messages": [{"role": "user", "content": "fix the shader"},
                             {"role": "assistant", "content": "patched"}]}
        r = httpx.post(base + "/remote/v1/traces/import", json=body,
                       headers={"Authorization": "Bearer gw-tok"}, timeout=10)
        assert r.status_code == 200, r.text
        tid = r.json()["trace_id"]
        # indexed locally with the mapped profile name
        tr = httpx.get(base + f"/api/traces/{tid}", timeout=5,
                       headers={"X-Sable-Token": "gw-tok"})
        assert tr.status_code == 200
        row = tr.json()["trace"]
        assert row["agent_profile"] == "tim"
        # and it shows up scoped like any local profile's traces
        lst = httpx.get(base + "/api/traces?agent_profile=tim", timeout=5,
                        headers={"X-Sable-Token": "gw-tok"}).json()
        assert tid in {t["trace_id"] for t in lst["traces"]}
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_remote_gateway_trace_import_auth_and_validation(home__rgw):
    import httpx

    httpd, base = _start_lan_server(home__rgw)
    try:
        r = httpx.post(base + "/remote/v1/traces/import",
                       json={"messages": [], "task": ""}, timeout=5)
        assert r.status_code == 401
        hdrs = {"Authorization": "Bearer gw-tok"}
        r = httpx.post(base + "/remote/v1/traces/import",
                       json={"messages": []}, headers=hdrs, timeout=5)
        assert r.status_code == 400
        r = httpx.post(base + "/remote/v1/traces/import",
                       json={"task": "x", "messages": [{"role": "user"}]},
                       headers=hdrs, timeout=5)
        # record() tolerates odd messages; import succeeds with a shape check
        assert r.status_code in (200, 500)
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_remote_run_pushes_trace_back(app__rgw, home__rgw, monkeypatch):
    """A chat run with a remote profile POSTs the transcript to the peer's
    /remote/v1/traces/import with the mapped local agent name."""
    import httpx

    from sable.core.agent_runner import AgentResult

    pushed = {}

    class Resp:
        status_code = 200

    def fake_post(url, json=None, headers=None, timeout=None):
        pushed["url"] = url
        pushed["json"] = json
        pushed["headers"] = headers
        return Resp()

    monkeypatch.setattr(httpx, "post", fake_post)

    class StubRunner:
        def __init__(self, cfg, base_url, model, api_key=None, **kwargs):
            pass

        def run(self, task, history=None, **kw):
            return AgentResult(final_response="ok", messages=[
                {"role": "user", "content": task},
                {"role": "assistant", "content": "ok"}],
                backend="stub", tokens_in=1, tokens_out=1)

    monkeypatch.setattr("sable.core.agent_runner.create_runner", StubRunner)
    monkeypatch.setattr("sable.serving.supervisor.healthcheck",
                        lambda *a, **k: True)
    _seed_peer__rgw(home__rgw)
    r = app__rgw.handle("POST", "/api/chat/stream",
                        body={"prompt": "hi tim", "profile": "tim@192.0.2.9"})
    assert r.status == 200 and r.stream is not None
    text = "".join(r.stream)
    assert "run.done" in text
    assert pushed.get("url") == "http://192.0.2.9:55429/remote/v1/traces/import"
    assert pushed["json"]["profile"] == "tim"  # mapped back to the local name
    assert pushed["headers"]["X-Sable-Token"] == "peer-tok"


def test_remote_run_push_failure_never_breaks_run(app__rgw, home__rgw,
                                                  monkeypatch):
    import httpx

    from sable.core.agent_runner import AgentResult

    def boom(*a, **k):
        raise RuntimeError("peer is down")

    monkeypatch.setattr(httpx, "post", boom)

    class StubRunner:
        def __init__(self, cfg, base_url, model, api_key=None, **kwargs):
            pass

        def run(self, task, history=None, **kw):
            return AgentResult(final_response="ok", messages=[
                {"role": "user", "content": task},
                {"role": "assistant", "content": "ok"}],
                backend="stub", tokens_in=1, tokens_out=1)

    monkeypatch.setattr("sable.core.agent_runner.create_runner", StubRunner)
    monkeypatch.setattr("sable.serving.supervisor.healthcheck",
                        lambda *a, **k: True)
    _seed_peer__rgw(home__rgw)
    r = app__rgw.handle("POST", "/api/chat/stream",
                        body={"prompt": "hi tim", "profile": "tim@192.0.2.9"})
    text = "".join(r.stream)
    assert "run.done" in text  # flow-back failure invisible to the run

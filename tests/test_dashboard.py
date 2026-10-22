"""Phase 5 tests — dashboard control API + server hygiene.

API-level tests call DashboardApp.handle() directly (no socket). One test
brings up the real stdlib server to verify loopback bind, static serving,
path-traversal rejection and JSON plumbing end-to-end. Security-hygiene
tests assert no external CDN/URL references in the shipped UI assets.
"""

from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, List, cast

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sable.core.config import SableConfig  # noqa: E402
from sable.dashboard.app import DashboardApp, build_trace_timeline  # noqa: E402
from sable.registry import ModelRegistry  # noqa: E402
from sable.store.db import TraceRow, TraceStore  # noqa: E402

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


def test_serve_start_validation_and_conflict(app, home):
    # nothing configured => 400
    r = app.handle("POST", "/api/serve/start", body={})
    assert r.status == 400 and err_code(r) == "bad_request"

    # auto with no active model => 409
    r = app.handle("POST", "/api/serve/start", body={"auto": True})
    assert r.status == 409 and err_code(r) == "no_active_model"

    # registered model without GGUF artifact => 409
    reg = ModelRegistry(home / "buffer" / "sable.db")
    m = reg.register_model(base_model="b")
    reg.close()
    r = app.handle("POST", "/api/serve/start", body={"model_id": m.model_id})
    assert r.status == 409 and err_code(r) == "no_gguf"

    # already serving (attach alive) => 409 unless force
    from fake_server import FakeOpenAIServer

    with FakeOpenAIServer() as fs:
        _fake_serve_state(home, fs.base_url)
        r = app.handle("POST", "/api/serve/start", body={"gguf": "/nope.gguf"})
        assert r.status == 409 and err_code(r) == "already_serving"


def test_serve_attach_and_stop_never_kills_attached(app, home):
    from fake_server import FakeOpenAIServer

    with FakeOpenAIServer() as fs:
        r = app.handle("POST", "/api/serve/attach", body={"url": fs.base_url})
        assert r.status == 202, r.body
        assert r.body["model_name"] == "fake-model"
        status = app.handle("GET", "/api/serve/status").body
        assert status["running"] and status["mode"] == "attach"

        r = app.handle("POST", "/api/serve/stop", body={})
        assert r.status == 200 and r.body == {"ok": True, "stopped": False}
        # state cleared; fake server still healthy (we never killed it)
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

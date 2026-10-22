"""Phase 5e tests — agent profiles, config unset semantics, auto-check trigger.

Coverage:
  * /api/profiles CRUD + default handling + validation (DashboardApp.handle)
  * /api/chat/stream honoring a profile: system prompt reaches the fake OpenAI
    server as the request's system message and the tool subset is enforced on
    the fallback client (tools list in the wire payload).
  * PUT /api/config section-patch unset semantics (null deletes a key)
  * `sable train auto-check --trigger`: no-op when thresholds unmet; runs
    the iterate loop (stub worker, dry-run) when met; gate-blocked => no run.
  * static hygiene: stub panel removed; nav covers the final panel set.
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sable.core.config import SableConfig  # noqa: E402
from sable.dashboard.app import DashboardApp  # noqa: E402
from sable.store.db import TraceRow, TraceStore  # noqa: E402

STATIC = Path(__file__).resolve().parents[1] / "sable" / "dashboard" / "static"


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "uf-5e"
    monkeypatch.setenv("SABLE_HOME", str(h))
    return h


@pytest.fixture()
def app(home):
    return DashboardApp(SableConfig(), home=home)


def seed_trace(store: TraceStore, tid: str, score=0.8, curated=False):
    from datetime import datetime, timedelta, timezone

    p = Path(store.db_path).parent / f"{tid}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        f.write(json.dumps({"kind": "trace_header"}) + "\n")
        f.write(json.dumps({"kind": "message", "message": {
            "role": "user", "content": f"task {tid}"}}) + "\n")
        f.write(json.dumps({"kind": "message", "message": {
            "role": "assistant", "content": f"done {tid}"}}) + "\n")
    # relative-to-now stamp (time-bomb sweep): never ages out of windowed
    # assertions (stats_daily / insights calendar cutoffs).
    ts = (datetime.now(timezone.utc) - timedelta(days=1)
          ).isoformat(timespec="seconds")
    store.insert_trace(TraceRow(
        trace_id=tid, session_id=None, timestamp=ts,
        agent_profile="fallback", model_alias="base-model",
        model_version=None, endpoint_label=None, task_summary=f"t {tid}",
        status="success", outcome_score=score, num_tool_calls=0,
        tokens_in=10, tokens_out=20, duration_s=1.0, jsonl_path=str(p),
        curated=bool(curated),
    ))


# ---------------------------------------------------------------- profiles CRUD

def test_profiles_crud_and_default(app, home):
    r = app.handle("GET", "/api/profiles")
    assert r.status == 200
    assert r.body["profiles"] == [] and r.body["default_profile"] is None
    assert "read_file" in r.body["known_tools"]

    # create
    r = app.handle("PUT", "/api/profiles/coder", body={
        "system_prompt": "You are a careful coder.",
        "tools": ["read_file"], "model_name": None})
    assert r.status == 200 and r.body["profile"]["name"] == "coder"

    # persisted to the config file (profiles: section)
    raw = (home / "sable.yaml").read_text()
    assert "coder" in raw and "careful coder" in raw

    # upsert same name updates in place
    app.handle("PUT", "/api/profiles/coder", body={
        "system_prompt": "v2 prompt", "tools": ["read_file", "list_dir"]})
    lst = app.handle("GET", "/api/profiles").body["profiles"]
    assert len(lst) == 1 and lst[0]["system_prompt"] == "v2 prompt"
    assert lst[0]["tools"] == ["list_dir", "read_file"]

    # default: unknown name rejected, known accepted, cleared with null
    assert app.handle("PUT", "/api/profiles/default",
                      body={"default": "ghost"}).status == 404
    r = app.handle("PUT", "/api/profiles/default", body={"default": "coder"})
    assert r.status == 200
    assert app.handle("GET", "/api/profiles").body["default_profile"] == "coder"

    # deleting the default clears it
    assert app.handle("DELETE", "/api/profiles/coder").status == 200
    d = app.handle("GET", "/api/profiles").body
    assert d["profiles"] == [] and d["default_profile"] is None
    assert app.handle("DELETE", "/api/profiles/coder").status == 404


def test_profile_validation(app):
    assert app.handle("PUT", "/api/profiles/bad name!", body={}).status == 400
    assert app.handle("PUT", "/api/profiles/ok",
                      body={"tools": "notalist"}).status == 400
    assert app.handle("PUT", "/api/profiles/ok",
                      body={"system_prompt": 5}).status == 400


# ------------------------------------------------------- chat stream + profile

class _RecordingHandler(BaseHTTPRequestHandler):
    """Answers plain-text completions and records every request payload."""
    payloads: list = []

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        body = json.dumps({"object": "list",
                           "data": [{"id": "fake-model", "object": "model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if self.path.rstrip("/") == "/tokenize":
            # Improvement #4: exact-token probes hit the server ROOT, not
            # /v1/chat/completions. Answer 404 (counter degrades to chars/4)
            # and keep them out of `payloads`: assertions below read chat
            # completion bodies.
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        type(self).payloads.append(payload)
        body = json.dumps({
            "id": "cmpl", "object": "chat.completion", "model": "fake-model",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "ok done"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_chat_stream_honors_profile(app, home, monkeypatch):
    """Profile system prompt + tool subset reach the wire (fallback client)."""
    from fake_server import FakeOpenAIServer

    _RecordingHandler.payloads = []

    app.handle("PUT", "/api/profiles/coder", body={
        "system_prompt": "PROFILE-PROMPT-MARKER",
        "tools": ["list_dir"],
    })
    with FakeOpenAIServer(handler_cls=_RecordingHandler) as fs:
        home.mkdir(parents=True, exist_ok=True)
        (home / "serve_state.json").write_text(json.dumps(
            {"mode": "attach", "base_url": fs.base_url,
             "model_name": "fake-model"}))
        r = app.handle("POST", "/api/chat/stream",
                       body={"prompt": "say hi", "profile": "coder"})
        assert r.status == 200 and r.stream is not None
        text = "".join(r.stream)

    assert "run.done" in text
    # unknown profile => clean 404 envelope, not a stream error
    bad = app.handle("POST", "/api/chat/stream",
                     body={"prompt": "x", "profile": "ghost"})
    assert bad.status == 404 and bad.body["error"]["code"] == "unknown_profile"

    payloads = _RecordingHandler.payloads
    assert payloads, "fake server never called"
    first = payloads[0]
    sys_msgs = [m for m in first["messages"] if m.get("role") == "system"]
    assert sys_msgs and sys_msgs[0]["content"] == "PROFILE-PROMPT-MARKER"
    tool_names = {t["function"]["name"] for t in first.get("tools", [])}
    assert tool_names == {"list_dir"}  # read_file subsetted away


# ------------------------------------------------------ config unset semantics

def test_config_patch_null_unsets_key(app, home):
    home.mkdir(parents=True, exist_ok=True)
    (home / "sable.yaml").write_text(
        "attach_url: http://127.0.0.1:9/v1\nport: 1234\n")
    r = app.handle("PUT", "/api/config", body={
        "section": "root", "patch": {"attach_url": None}})
    assert r.status == 200
    import yaml

    data = yaml.safe_load((home / "sable.yaml").read_text())
    assert "attach_url" not in data and data["port"] == 1234


# --------------------------------------------------- auto-check --trigger CLI

STUB_WORKER = r'''
import argparse, json, sys
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("--job", required=True); ap.add_argument("--kind", required=True)
ap.add_argument("--config", default=""); ap.add_argument("--dataset-dir", default=None)
ap.add_argument("--base-model", default=None); ap.add_argument("--log-path", default=None)
ap.add_argument("--adapter-dir", default=None); ap.add_argument("--gguf-out", default=None)
ap.add_argument("--quant", default="q4_k_m"); ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()
line = json.dumps({"event": "done", "job": a.job})
print(line, flush=True)
if a.log_path:
    Path(a.log_path).parent.mkdir(parents=True, exist_ok=True)
    open(a.log_path, "a").write(line + "\n")
'''


def _run_auto_check(monkeypatch, home, trigger=False, dry_run=False):
    import sable.cli as cli

    monkeypatch.setenv("SABLE_HOME", str(home))
    monkeypatch.setenv("SABLE_CONFIG", str(home / "sable.yaml"))
    argv = ["train", "auto-check"]
    if trigger:
        argv.append("--trigger")
    if dry_run:
        argv.append("--dry-run")
    return cli.main(argv)


def test_auto_check_trigger_only_when_gate_ok(monkeypatch, home, tmp_path, capsys):
    # thresholds low enough that a couple of good traces suffice
    home.mkdir(parents=True, exist_ok=True)
    (home / "sable.yaml").write_text(
        "training:\n  auto_min_traces: 2\n  auto_min_score: 0.5\n")
    stub = tmp_path / "stub_worker.py"
    stub.write_text(STUB_WORKER)
    monkeypatch.setenv("SABLE_WORKER_COMMAND", f"{sys.executable} {stub}")

    # 1) not enough traces -> no trigger, exit 0 with --trigger, no jobs
    rc = _run_auto_check(monkeypatch, home, trigger=True, dry_run=True)
    assert rc == 0
    out = capsys.readouterr().out
    assert "no trigger" in out and "iterate" not in out.lower().split("no trigger")[1]

    # 2) enough good traces -> --trigger runs iterate (dry-run, stub worker)
    store = TraceStore(home / "buffer" / "sable.db")
    seed_trace(store, "t1", score=0.9)
    seed_trace(store, "t2", score=0.8)
    seed_trace(store, "t3", score=0.7)
    store.close()
    rc = _run_auto_check(monkeypatch, home, trigger=True, dry_run=True)
    out = capsys.readouterr().out
    assert rc == 0 and "starting iterate loop" in out
    assert '"stage": "train"' in out or "train" in out  # pipeline report printed
    # an iterate job actually ran through the scheduler/registry
    from sable.registry import ModelRegistry

    reg = ModelRegistry(home / "buffer" / "sable.db")
    try:
        kinds = [j.kind for j in reg.list_jobs()]
        states = [j.state for j in reg.list_jobs()]
    finally:
        reg.close()
    assert "sft" in kinds and "done" in states

    # 3) resource gate blocked -> trigger reports and exits without running
    (home / "sable.yaml").write_text(
        "training:\n  auto_min_traces: 2\n  auto_min_score: 0.5\n"
        "  min_free_vram_mib: 999999999\n")
    monkeypatch.setattr("sable.training.scheduler.gpu_free_mib",
                        lambda: 1, raising=True)
    store = TraceStore(home / "buffer" / "sable.db")
    seed_trace(store, "t4", score=0.9)
    seed_trace(store, "t5", score=0.9)
    # unclaim the earlier ones so curation has material again
    for tid in ("t1", "t2", "t3"):
        store._conn.execute("UPDATE traces SET curated=0 WHERE trace_id=?", (tid,))
    store._conn.commit()
    store.close()
    rc = _run_auto_check(monkeypatch, home, trigger=True, dry_run=True)
    out = capsys.readouterr().out
    assert rc == 0 and "gate blocked" in out
    assert "starting iterate loop" not in out


# ------------------------------------------------------------- static hygiene

def test_stub_panel_removed_and_nav_complete():
    assert not (STATIC / "js" / "panels" / "stub.js").exists(), \
        "stub panel must be gone after 5e"
    app_js = (STATIC / "app.js").read_text()
    assert "stub.js" not in app_js
    for panel in ("overview", "chat", "serve", "models", "jobs", "traces",
                  "datasets", "train", "iterate", "schedule", "profiles",
                  "settings", "hardware"):
        assert f'["{panel}",' in app_js, f"{panel} missing from nav"
    # every registered panel file exists and is imported
    import re

    for m in re.finditer(r'import \w+ from "\./js/panels/(\w+)\.js"', app_js):
        assert (STATIC / "js" / "panels" / f"{m.group(1)}.js").is_file()


def test_profiles_routes_registered(app):
    paths = {(m, p) for (m, p, _) in app.ROUTES}
    assert ("GET", "/api/profiles") in paths
    assert ("PUT", "/api/profiles/<name>") in paths
    assert ("DELETE", "/api/profiles/<name>") in paths
    assert ("PUT", "/api/profiles/default") in paths

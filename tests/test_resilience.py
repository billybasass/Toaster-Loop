"""Backlog #2/#7 integration tests — runner retry + failure_reason plumbing.

Covers: transient connection blip recovers via retry_chat (no failure trace);
final backend failure records failure_reason in footer + index; max-iteration
runs record "max_iterations"; the curater excludes infra-failure traces from
DPO negatives; golden-suite transport retry survives a single HTTP blip.
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fake_server import FakeOpenAIServer, _Handler, make_scripted_handler  # noqa: E402
from sable.core.agent_runner import SableAgentClient  # noqa: E402
from sable.store.db import TraceRow, TraceStore  # noqa: E402
from sable.store.recorder import TraceRecorder  # noqa: E402
from sable.training.validate import run_golden_suite  # noqa: E402


@pytest.fixture()
def sable_home(tmp_path, monkeypatch):
    home = tmp_path / "uf-home"
    monkeypatch.setenv("SABLE_HOME", str(home))
    return home


@pytest.fixture(autouse=True)
def _no_sleeps(monkeypatch):
    """Retry backoff must not slow the suite down."""
    from sable.core import agent_runner as ar
    monkeypatch.setattr(ar.SableAgentClient, "_sleep", staticmethod(lambda s: None))
    import sable.training.validate as v
    monkeypatch.setattr(v, "_sleep", lambda s: None)


# --------------------------------------------------------------- flaky server

class FlakyOnceHandler(_Handler):
    """First chat POST dies with a connection reset; then behaves normally."""

    fail_once = True  # class attr flipped by the handler itself

    def do_POST(self):
        if type(self).fail_once:
            type(self).fail_once = False
            self.close_connection = True
            try:
                self.wfile.close()
            except Exception:
                pass
            return
        super().do_POST()


class AlwaysToolCallHandler(BaseHTTPRequestHandler):
    """Never converges: forces the max-iterations exit."""

    def log_message(self, format, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        msg = {"role": "assistant", "content": None, "tool_calls": [{
            "id": "c", "type": "function",
            "function": {"name": "list_dir", "arguments": "{}"}}]}
        body = json.dumps({
            "id": "x", "object": "chat.completion",
            "model": payload.get("model", "m"),
            "choices": [{"index": 0, "message": msg,
                         "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                      "total_tokens": 2}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class BlipThenOkHandler(BaseHTTPRequestHandler):
    """First POST answers HTTP 503 (restart blip); afterwards replies '4'."""

    blipped = False

    def log_message(self, format, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if not type(self).blipped:
            type(self).blipped = True
            self._send({"error": "loading model"}, 503)
            return
        self._send({
            "id": "x", "object": "chat.completion",
            "model": payload.get("model", "m"),
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": "the answer is 4"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                      "total_tokens": 2}})

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ------------------------------------------------------------------- runner

def test_transient_connection_blip_recovered_no_failure_trace(sable_home):
    FlakyOnceHandler.fail_once = True
    rec = TraceRecorder(home=sable_home)
    with FakeOpenAIServer(handler_cls=FlakyOnceHandler) as srv:
        client = SableAgentClient(base_url=srv.base_url, model="fake-model",
                                    recorder=rec)
        result = client.run("read the fake server source")
    assert result.completed is True
    assert "Done." in result.final_response
    assert result.failure_reason is None
    assert result.trace_id is not None

    footer = _footer_for(sable_home, result.trace_id)
    assert footer["failure_reason"] is None
    assert footer["status"] == "success"


def test_final_backend_failure_records_failure_reason(sable_home):
    rec = TraceRecorder(home=sable_home)
    client = SableAgentClient(base_url="http://127.0.0.1:9/v1", model="m",
                                timeout=2.0, recorder=rec)
    result = client.run("hello")
    assert result.completed is False
    assert result.failure_reason == "backend_error:connection"
    assert result.trace_id is not None

    footer = _footer_for(sable_home, result.trace_id)
    assert footer["failure_reason"] == "backend_error:connection"
    assert footer["completed"] is False  # completed semantics unchanged

    store = TraceStore(sable_home / "buffer" / "sable.db")
    try:
        row = store.get_trace(result.trace_id)
        assert row is not None
        assert row.failure_reason == "backend_error:connection"
    finally:
        store.close()


def test_max_iterations_failure_reason():
    with FakeOpenAIServer(handler_cls=AlwaysToolCallHandler) as srv:
        client = SableAgentClient(base_url=srv.base_url, model="m",
                                    max_iterations=2)
        result = client.run("loop forever")
    assert result.completed is False
    assert result.failure_reason == "max_iterations"


def _footer_for(home: Path, trace_id: str) -> dict:
    matches = list((home / "buffer").rglob(f"{trace_id}.jsonl"))
    assert len(matches) == 1
    lines = [json.loads(l) for l in matches[0].read_text().splitlines()]
    return lines[-1]


# ------------------------------------------------------------------ curater

def _seed(store, home, *, trace_id, messages, status="success", score=0.5,
          timestamp="2026-09-01T10:00:00+00:00", failure_reason=None):
    jsonl_path = home / "buffer" / "2026" / "09" / f"{trace_id}.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "x", encoding="utf-8") as f:
        f.write(json.dumps({"kind": "trace_header", "schema_version": 2,
                            "trace_id": trace_id, "task": "same task",
                            "failure_reason": failure_reason}) + "\n")
        for m in messages:
            f.write(json.dumps({"kind": "message", "message": m}) + "\n")
        f.write(json.dumps({"kind": "trace_footer", "schema_version": 2,
                            "status": status, "outcome_score": score,
                            "failure_reason": failure_reason}) + "\n")
    store.insert_trace(TraceRow(
        trace_id=trace_id, session_id=None, timestamp=timestamp,
        agent_profile="sable", model_alias="m", model_version=None,
        endpoint_label=None, task_summary="seed", status=status,
        outcome_score=score, num_tool_calls=0, tokens_in=None,
        tokens_out=None, duration_s=None, jsonl_path=str(jsonl_path),
        failure_reason=failure_reason,
    ))


def test_curater_excludes_infra_failures_from_dpo_negatives(sable_home):
    store = TraceStore(sable_home / "buffer" / "sable.db")
    good = [{"role": "user", "content": "same task"},
            {"role": "assistant", "content": "great answer"}]
    infra_bad = [{"role": "user", "content": "same task"},
                 {"role": "assistant", "content": "server hiccuped"}]
    model_bad = [{"role": "user", "content": "same task"},
                 {"role": "assistant", "content": "terrible answer"}]
    _seed(store, sable_home, trace_id="good", messages=good, score=1.0)
    _seed(store, sable_home, trace_id="infra", messages=infra_bad,
          status="failure", score=-1.0, timestamp="2026-09-01T09:00:00+00:00",
          failure_reason="backend_error:connection")
    _seed(store, sable_home, trace_id="modelbad", messages=model_bad,
          status="failure", score=-1.0, timestamp="2026-09-01T08:00:00+00:00",
          failure_reason="max_iterations")

    from sable.curate.curater import (CurationConfig, conversation_group_key,
                                        load_trace, mine_dpo_pairs)
    rows = store.query_for_curation(include_curated=False,
                                    statuses=["success", "partial", "unknown",
                                              "failure"])
    groups = {}
    for r in rows:
        t = load_trace(store, r)
        assert t is not None
        groups.setdefault(
            conversation_group_key(t.messages, t.row.agent_profile), []
        ).append(t)
    pairs = mine_dpo_pairs(groups, CurationConfig(include_failures=True))
    assert len(pairs) == 1
    # infra failure excluded despite ranking lowest; model failure used instead
    assert pairs[0].rejected_trace_id == "modelbad"
    assert pairs[0].chosen_trace_id == "good"
    store.close()


# ----------------------------------------------------------------- validate

def test_golden_suite_survives_single_http_blip():
    BlipThenOkHandler.blipped = False
    tasks = [{"prompt": "what is 2+2?", "expect_contains": ["4"]}]
    with FakeOpenAIServer(handler_cls=BlipThenOkHandler) as srv:
        res = run_golden_suite(srv.base_url, "m", tasks)
    assert res["pass_rate"] == 1.0


def test_golden_suite_reports_persistent_failure():
    script = {"*": "nonsense"}
    with FakeOpenAIServer(handler_cls=make_scripted_handler(script)) as srv:
        res = run_golden_suite(srv.base_url, "m",
                               [{"prompt": "q", "expect_contains": ["4"]}])
    assert res["pass_rate"] == 0.0

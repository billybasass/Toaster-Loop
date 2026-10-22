"""Sampling + thinking settings and live token streaming.

Covers:
  1. serving/run_args.sanitize_sampling — validation/coercion of sampling and
     thinking keys (studio-style enable_thinking / reasoning_effort /
     preserve_thinking).
  2. serving/run_args.request_sampling_payload — thinking controls reach
     llama.cpp via chat_format_template_controls AND stay top-level.
  3. dashboard instances API — per-instance `sampling:` stored on create,
     validated, editable via PUT (Serve page flow).
  4. profiles API — per-profile sampling override validation + storage.
  5. agent runner streaming — SableAgentClient with an on_event sink
     forwards content/reasoning deltas live from an SSE backend.
  6. /api/chat/stream — real runner against the fake server streams live
     events (message.delta / tool.call / tool.result) without synthesizing
     duplicates; run.started carries context_window.

All hermetic: local fake OpenAI servers only.
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fake_server import FakeOpenAIServer  # noqa: E402
from sable.core.agent_runner import SableAgentClient  # noqa: E402
from sable.core.config import SableConfig  # noqa: E402
from sable.dashboard.app import DashboardApp  # noqa: E402
from sable.serving.run_args import (  # noqa: E402
    request_sampling_payload, sanitize_sampling,
)


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "uf-sts"
    monkeypatch.setenv("SABLE_HOME", str(h))
    return h


def make_app(home) -> DashboardApp:
    return DashboardApp(SableConfig(llama_auto_setup=False), home=home)


# ----------------------------------------------------- 1. sanitize_sampling

def test_sanitize_sampling_valid_and_coercion():
    out = sanitize_sampling({
        "temperature": "0.7", "top_p": 0.95, "top_k": "40",
        "min_p": 0.05, "repeat_penalty": 1.1, "presence_penalty": -0.5,
        "max_tokens": "512", "penalty_free_repeat": 64.0,
        "enable_thinking": True, "reasoning_effort": "HIGH",
        "preserve_thinking": False,
    })
    assert out["temperature"] == 0.7
    assert out["top_k"] == 40 and isinstance(out["top_k"], int)
    assert out["max_tokens"] == 512
    assert out["penalty_free_repeat"] == 64 and isinstance(
        out["penalty_free_repeat"], int)
    assert out["enable_thinking"] is True
    assert out["reasoning_effort"] == "high"
    assert out["preserve_thinking"] is False


def test_sanitize_sampling_rejects_bad_values():
    with pytest.raises(ValueError):
        sanitize_sampling({"temperature": 5})       # > 2.0
    with pytest.raises(ValueError):
        sanitize_sampling({"top_k": 0})             # must be >= 1
    with pytest.raises(ValueError):
        sanitize_sampling({"reasoning_effort": "ultra"})
    with pytest.raises(ValueError):
        sanitize_sampling({"nonsense": 1})
    with pytest.raises(ValueError):
        sanitize_sampling({"temperature": "hot"})


def test_sanitize_sampling_empty_clears():
    out = sanitize_sampling({"temperature": "", "reasoning_effort": None})
    assert out == {"temperature": None, "reasoning_effort": None}


# ----------------------------------------------- 2. request payload shaping

def test_request_sampling_payload_thinking_controls():
    frag = request_sampling_payload({
        "temperature": 0.4, "enable_thinking": True,
        "reasoning_effort": "low", "preserve_thinking": None,
    })
    assert frag["temperature"] == 0.4
    # top-level (vLLM-style) AND the llama.cpp template-controls channel
    assert frag["enable_thinking"] is True
    assert frag["reasoning_effort"] == "low"
    assert frag["chat_format_template_controls"] == {
        "enable_thinking": True, "reasoning_effort": "low"}
    # None values dropped entirely
    assert "preserve_thinking" not in frag


def test_request_sampling_payload_no_controls_no_map():
    frag = request_sampling_payload({"temperature": 1.0})
    assert frag == {"temperature": 1.0}
    assert request_sampling_payload({}) == {}
    assert request_sampling_payload(None) == {}


# ------------------------------------------------- 3. instance sampling API

def _create(app, body):
    return app.handle("POST", "/api/servers", body=body)


def test_instance_create_stores_sampling(home):
    app = make_app(home)
    r = _create(app, {"name": "coder", "gguf": "/models/x.gguf",
                      "sampling": {"temperature": 0.3,
                                   "enable_thinking": True,
                                   "reasoning_effort": "high"}})
    assert r.status == 201
    srv = r.body["server"]
    assert srv["sampling"]["temperature"] == 0.3
    assert srv["sampling"]["enable_thinking"] is True
    assert srv["sampling"]["reasoning_effort"] == "high"


def test_instance_create_rejects_bad_sampling(home):
    app = make_app(home)
    r = _create(app, {"name": "coder", "gguf": "/models/x.gguf",
                      "sampling": {"temperature": 9}})
    assert r.status == 400
    assert "sampling" in r.body["error"]["message"]


def test_instance_put_updates_sampling(home):
    app = make_app(home)
    sid = _create(app, {"name": "c", "gguf": "/models/x.gguf"}).body["server"]["id"]
    r = app.handle("PUT", f"/api/servers/{sid}",
                   body={"sampling": {"reasoning_effort": "medium"}})
    assert r.status == 200
    assert r.body["server"]["sampling"] == {"reasoning_effort": "medium"}


# ----------------------------------------------------- 4. profile sampling

def test_profile_stores_sampling_override(home):
    app = make_app(home)
    r = app.handle("PUT", "/api/profiles/coder", body={
        "system_prompt": "hi", "tools": [],
        "sampling": {"temperature": 0.2, "enable_thinking": False}})
    assert r.status == 200
    prof = r.body["profile"]
    assert prof["sampling"]["temperature"] == 0.2
    assert prof["sampling"]["enable_thinking"] is False


def test_profile_rejects_bad_sampling(home):
    r = make_app(home).handle("PUT", "/api/profiles/coder",
                              body={"sampling": {"reasoning_effort": "zzz"}})
    assert r.status == 400


# --------------------------------------------------- 5. runner streaming

class ReasoningSSEHandler(BaseHTTPRequestHandler):
    """Streams one reply as reasoning deltas then content deltas."""

    def log_message(self, format, *args):  # noqa: A002 - silence access log
        pass

    def do_POST(self):
        chunks = [
            {"delta": {"role": "assistant"}},
            {"delta": {"reasoning_content": "thinking... "}},
            {"delta": {"reasoning_content": "hard."}},
            {"delta": {"content": "Hello "}},
            {"delta": {"content": "world"}},
            {"delta": {}, "finish_reason": "stop"},
        ]
        body = "".join(
            "data: " + json.dumps({"id": "x", "object": "chat.completion.chunk",
                                   "model": "m",
                                   "choices": [dict(index=0, **c)]}) + "\n\n"
            for c in chunks) + "data: [DONE]\n\n"
        raw = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(raw)


def test_agent_client_streams_content_and_reasoning():
    events = []
    with FakeOpenAIServer(handler_cls=ReasoningSSEHandler) as fs:
        client = SableAgentClient(fs.base_url, "m", tools=[], tool_impls={})
        client.on_event = lambda kind, data: events.append((kind, data))
        data = client._chat([{"role": "user", "content": "hi"}])
    kinds = [k for k, _ in events]
    assert kinds.count("reasoning.delta") == 2
    assert kinds.count("message.delta") == 2
    text = "".join(d["text"] for k, d in events if k == "message.delta")
    reasoning = "".join(d["text"] for k, d in events if k == "reasoning.delta")
    assert text == "Hello world"
    assert reasoning == "thinking... hard."
    # The returned message carries only visible content (studio semantics).
    msg = data["choices"][0]["message"]
    assert msg["content"] == "Hello world"


class PlainJSONHandler(BaseHTTPRequestHandler):
    """Ignores stream=true and answers with a plain JSON completion."""

    def log_message(self, format, *args):  # noqa: A002 - silence access log
        pass

    def do_POST(self):
        body = json.dumps({
            "id": "x", "object": "chat.completion", "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "plain"}}],
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_agent_client_falls_back_when_server_ignores_stream():
    events = []
    with FakeOpenAIServer(handler_cls=PlainJSONHandler) as fs:
        client = SableAgentClient(fs.base_url, "m", tools=[], tool_impls={})
        client.on_event = lambda kind, data: events.append(kind)
        data = client._chat([{"role": "user", "content": "hi"}])
    assert data["choices"][0]["message"]["content"] == "plain"
    assert events == []  # no deltas invented


# ---------------------------------------------------- 6. chat/stream live

def _fake_serve_state(home, url):
    home.mkdir(parents=True, exist_ok=True)
    (home / "serve_state.json").write_text(json.dumps({
        "mode": "attach", "base_url": url, "model_name": "fake-model",
        "attach_url": url.rsplit("/v1", 1)[0],
    }))


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


def test_chat_stream_live_events_no_duplicates(home):
    app = make_app(home)
    with FakeOpenAIServer() as fs:
        _fake_serve_state(home, fs.base_url)
        r = app.handle("POST", "/api/chat/stream", body={"prompt": "go"})
        assert r.stream is not None
        text = "".join(r.stream)
    evs = _events(text)
    kinds = [k for k, _ in evs]
    assert kinds[0] == "run.started"
    assert kinds[-1] == "run.done"
    # run.started carries the context window for the UI tracker
    assert evs[0][1]["context_window"] > 0
    # The fake server streams; deltas arrive live and tool events fire once.
    assert "message.delta" in kinds
    n_tool_call = kinds.count("tool.call")
    n_tool_result = kinds.count("tool.result")
    assert n_tool_call == n_tool_result >= 1
    # No synthesized duplicates: each tool call id appears exactly once per phase
    ids = [d["id"] for k, d in evs if k == "tool.call"]
    assert len(ids) == len(set(ids))
    # Final assistant content streamed live (exactly the fake's answer once)
    final = "".join(d.get("text", "") for k, d in evs if k == "message.delta")
    assert final.count("Done. The file was read successfully.") == 1


def test_chat_stream_sampling_merge_precedence(home):
    """profile sampling wins over instance sampling wins over global."""
    app = make_app(home)
    # global
    app.handle("PUT", "/api/config", body={"section": "sampling",
                                           "patch": {"temperature": 0.9,
                                                     "top_k": 50}})
    sid = _create(app, {
        "name": "bound", "gguf": "/models/x.gguf",
        "sampling": {"temperature": 0.5, "enable_thinking": True},
    }).body["server"]["id"]
    app.handle("PUT", "/api/profiles/pro", body={
        "system_prompt": "x", "tools": [], "server_instance": sid,
        "sampling": {"temperature": 0.1}})

    seen = {}

    class CaptureHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A002 - silence access log
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            seen.update(payload)
            body = json.dumps({
                "id": "x", "object": "chat.completion", "model": "m",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant",
                                         "content": "ok"}}],
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), CaptureHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}/v1"
        # Start the named instance against the capture endpoint by faking its
        # state file (spawn would need a real llama-server binary).
        sdir = home / "serving" / "state"
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / f"{sid}.json").write_text(json.dumps({
            "mode": "spawn", "base_url": base, "model_name": "m",
            "pid": 1, "port": httpd.server_address[1],
        }))
        r = app.handle("POST", "/api/chat/stream",
                       body={"prompt": "hi", "profile": "pro"})
        assert r.stream is not None
        "".join(r.stream)
    finally:
        httpd.shutdown()
        httpd.server_close()

    assert seen.get("temperature") == 0.1      # profile wins
    assert seen.get("top_k") == 50             # global survives the merge
    assert seen.get("enable_thinking") is True  # instance thinking applies
    assert seen.get("chat_format_template_controls", {}).get(
        "enable_thinking") is True

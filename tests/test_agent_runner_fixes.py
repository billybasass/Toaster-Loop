"""Regression tests for the six agent_runner deep-review fixes.

Covers (per fix number from the review):
  1. ``_summarize`` routes its sampling map through request_sampling_payload,
     so thinking controls reach the server in the shaped form (top-level AND
     chat_format_template_controls) instead of raw-spread keys strict servers
     reject with a 400.
  2. The stream -> non-stream fallback never leaves duplicated deltas behind:
     when partial message.delta events already went out, exactly one
     ``message.reset`` replacement event carries the authoritative text.
  3. Streaming tool-call assembly is robust to servers that omit the per-delta
     ``index`` field: parallel calls distinguished by id assemble separately,
     and interleaved argument fragments of one call stay together (proper
     indices keep working verbatim).
  4. A caller-supplied custom ``tools`` list is INTERSECTED with the profile's
     ``enabled_tool_names`` (with a WARNING naming what was dropped) instead
     of silently bypassing the filter; without an enabled list it passes
     through untouched.
  6. Tool-argument INFO logging goes through the trace redactor: secret shapes
     from test_redact never appear in captured INFO logs during a tool run.

All hermetic: local fake OpenAI servers only (fake_server facilities).
"""

from __future__ import annotations

import json
import logging
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fake_server import FakeOpenAIServer  # noqa: E402
from sable.core.agent_runner import SableAgentClient  # noqa: E402


def _sse(chunks):
    """Render scripted choice-dicts as an SSE body (one chunk per frame)."""
    out = "".join(
        "data: " + json.dumps({"id": "x", "object": "chat.completion.chunk",
                               "model": "m",
                               "choices": [dict(index=0, **c)]}) + "\n\n"
        for c in chunks) + "data: [DONE]\n\n"
    return out.encode()


class _Silent(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002 - silence access log
        pass


# ------------------------------------------------- Fix 1: _summarize routing

class SummarizeCaptureHandler(_Silent):
    """Answers the summarizer call; records the wire payload on the class."""

    seen: dict = {}

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        type(self).seen = json.loads(self.rfile.read(length) or b"{}")
        body = json.dumps({
            "id": "s", "object": "chat.completion", "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant",
                                     "content": "a concise summary"}}],
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_summarize_routes_sampling_through_request_payload():
    SummarizeCaptureHandler.seen = {}
    with FakeOpenAIServer(handler_cls=SummarizeCaptureHandler) as fs:
        client = SableAgentClient(
            fs.base_url, "m", tools=[], tool_impls={},
            sampling={"temperature": 0.3, "enable_thinking": False,
                      "reasoning_effort": "low"})
        out = client._summarize("condense this please")
    assert out == "a concise summary"
    seen = SummarizeCaptureHandler.seen
    # Sampling numbers ride top-level; thinking controls land in BOTH shapes —
    # exactly what request_sampling_payload produces for the main path.
    assert seen["temperature"] == 0.3
    assert seen["enable_thinking"] is False
    assert seen["reasoning_effort"] == "low"
    assert seen["chat_format_template_controls"] == {
        "enable_thinking": False, "reasoning_effort": "low"}


def test_summarize_no_sampling_keys_stays_minimal():
    SummarizeCaptureHandler.seen = {}
    with FakeOpenAIServer(handler_cls=SummarizeCaptureHandler) as fs:
        client = SableAgentClient(fs.base_url, "m", tools=[], tool_impls={})
        assert client._summarize("hi") is not None
    seen = SummarizeCaptureHandler.seen
    assert "chat_format_template_controls" not in seen
    assert set(seen) == {"model", "messages"}


# ------------------------------- Fix 2: stream fallback emits message.reset

class PartialStreamThenBreakHandler(_Silent):
    """Streaming requests push two content deltas then a chunk that breaks
    delta processing (mid-flight failure); non-streaming requests answer with
    the full plain JSON completion."""

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if payload.get("stream"):
            raw = (
                _sse([{"delta": {"content": "Hel"}},
                      {"delta": {"content": "lo "}}])[:-len(b"data: [DONE]\n\n")]
                # choices as a string: iterating yields str, .get raises —
                # simulates a server corrupting the stream mid-flight.
                + ("data: " + json.dumps({"id": "x", "choices": "bad"})
                   + "\n\n").encode())
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(raw)
            return
        body = json.dumps({
            "id": "x", "object": "chat.completion", "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant",
                                     "content": "Hello world"}}],
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_stream_fallback_replaces_deltas_with_reset():
    events = []
    with FakeOpenAIServer(handler_cls=PartialStreamThenBreakHandler) as fs:
        client = SableAgentClient(fs.base_url, "m", tools=[], tool_impls={})
        client.on_event = lambda kind, data: events.append((kind, data))
        data = client._chat([{"role": "user", "content": "hi"}])
    kinds = [k for k, _ in events]
    # Partial deltas went out before the break...
    assert kinds[:2] == ["message.delta", "message.delta"]
    # ...and exactly one replacement event follows; no further deltas emitted.
    assert kinds.count("message.reset") == 1
    assert kinds[-1] == "message.reset"
    reset_text = next(d for k, d in events if k == "message.reset")["text"]
    assert reset_text == "Hello world"
    # The returned completion is the authoritative one-shot answer.
    assert data["choices"][0]["message"]["content"] == "Hello world"
    assert client._no_stream is True


def test_stream_fallback_without_prior_deltas_no_reset():
    """Empty-stream fallback (nothing streamed) must NOT invent a reset."""

    class EmptyStreamHandler(_Silent):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            if payload.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(b"data: [DONE]\n\n")
                return
            body = json.dumps({
                "id": "x", "object": "chat.completion", "model": "m",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant",
                                         "content": "plain"}}],
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    events = []
    with FakeOpenAIServer(handler_cls=EmptyStreamHandler) as fs:
        client = SableAgentClient(fs.base_url, "m", tools=[], tool_impls={})
        client.on_event = lambda kind, data: events.append(kind)
        data = client._chat([{"role": "user", "content": "hi"}])
    assert data["choices"][0]["message"]["content"] == "plain"
    assert events == []  # nothing streamed => nothing to replace


# --------------------------- Fix 3: index-less tool-call delta assembly

def _stream_completion(chunks):
    """Replay `chunks` (list of choice-dicts) as one SSE completion and
    return the dict client._chat assembles from it."""

    class SSEHandler(_Silent):
        def do_POST(self):
            raw = _sse(chunks)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(raw)

    with FakeOpenAIServer(handler_cls=SSEHandler) as srv:
        client = SableAgentClient(srv.base_url, "m", tools=[], tool_impls={})
        client.on_event = lambda kind, data: None  # enable streaming path
        return client._chat([{"role": "user", "content": "go"}])


def _tool_calls_of(data):
    return (data["choices"][0]["message"].get("tool_calls") or [])


def test_tool_call_assembly_with_proper_indices():
    chunks = [
        {"delta": {"tool_calls": [
            {"index": 0, "id": "call_a", "type": "function",
             "function": {"name": "read_file", "arguments": "{\"pa"}}]}},
        {"delta": {"tool_calls": [
            {"index": 1, "id": "call_b", "type": "function",
             "function": {"name": "list_dir", "arguments": "{\"p"}}]}},
        {"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": "th\": \"x\"}"}}]}},
        {"delta": {"tool_calls": [
            {"index": 1, "function": {"arguments": "ath\": \"y\"}"}}]}},
    ]
    tcs = _tool_calls_of(_stream_completion(chunks))
    assert [t["id"] for t in tcs] == ["call_a", "call_b"]
    assert tcs[0]["function"]["name"] == "read_file"
    assert json.loads(tcs[0]["function"]["arguments"]) == {"path": "x"}
    assert tcs[1]["function"]["name"] == "list_dir"
    assert json.loads(tcs[1]["function"]["arguments"]) == {"path": "y"}


def test_tool_call_assembly_all_none_indices_parallel_calls():
    """Servers that omit `index` entirely: a fresh id opens a new slot."""
    chunks = [
        {"delta": {"tool_calls": [
            {"id": "call_a", "type": "function",
             "function": {"name": "read_file",
                          "arguments": "{\"path\": \"a\"}"}}]}},
        {"delta": {"tool_calls": [
            {"id": "call_b", "type": "function",
             "function": {"name": "list_dir",
                          "arguments": "{\"path\": \"b\"}"}}]}},
    ]
    tcs = _tool_calls_of(_stream_completion(chunks))
    assert [t["id"] for t in tcs] == ["call_a", "call_b"]
    assert tcs[0]["function"]["name"] == "read_file"
    assert json.loads(tcs[0]["function"]["arguments"]) == {"path": "a"}
    assert tcs[1]["function"]["name"] == "list_dir"
    assert json.loads(tcs[1]["function"]["arguments"]) == {"path": "b"}


def test_tool_call_assembly_interleaved_fragments_single_call():
    """Argument fragments WITHOUT id/index keep accumulating into the open
    call; only a fresh id advances the cursor."""
    chunks = [
        {"delta": {"tool_calls": [
            {"id": "call_a", "type": "function",
             "function": {"name": "search_files", "arguments": "{\"qu"}}]}},
        {"delta": {"tool_calls": [
            {"function": {"arguments": "ery\": \"he"}}]}},
        {"delta": {"tool_calls": [
            {"function": {"arguments": "llo\"}"}}]}},
    ]
    tcs = _tool_calls_of(_stream_completion(chunks))
    assert len(tcs) == 1
    assert tcs[0]["id"] == "call_a"
    assert json.loads(tcs[0]["function"]["arguments"]) == {"query": "hello"}


# ------------------------------------------------ Fix 4: tools ∩ enabled list

_CUSTOM_TOOLS = [
    {"type": "function", "function": {"name": "read_file",
                                      "parameters": {}}},
    {"type": "function", "function": {"name": "write_file",
                                      "parameters": {}}},
    {"type": "function", "function": {"name": "list_dir",
                                      "parameters": {}}},
]


def _names(schemas):
    return {s.get("function", {}).get("name") for s in schemas}


def test_custom_tools_intersected_with_enabled_tool_names(caplog):
    with caplog.at_level(logging.WARNING, logger="sable.agent"):
        client = SableAgentClient(
            "http://127.0.0.1:1/v1", "m",
            tools=[dict(t) for t in _CUSTOM_TOOLS],
            tool_impls={"read_file": lambda **kw: "",
                        "write_file": lambda **kw: "",
                        "list_dir": lambda **kw: ""},
            enabled_tool_names=["read_file", "list_dir"])
    assert _names(client.tools) == {"read_file", "list_dir"}
    assert set(client.tool_impls) == {"read_file", "list_dir"}
    # The dropped tool is reported loudly, not silently.
    msgs = [r.getMessage() for r in caplog.records
            if "enabled_tool_names" in r.getMessage()]
    assert msgs and "write_file" in msgs[0]


def test_custom_tools_passthrough_without_enabled_list():
    client = SableAgentClient(
        "http://127.0.0.1:1/v1", "m",
        tools=[dict(t) for t in _CUSTOM_TOOLS],
        tool_impls={"read_file": lambda **kw: "",
                    "write_file": lambda **kw: "",
                    "list_dir": lambda **kw: ""})
    assert _names(client.tools) == {"read_file", "write_file", "list_dir"}


# --------------------------------- Fix 6: tool args redacted in INFO logging

class ToolCallOnceHandler(_Silent):
    """First request answers with a native read_file tool call whose args
    embed an OpenAI-shaped secret; after a tool result arrives, finishes."""

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        messages = payload.get("messages", [])
        saw_tool_result = any(m.get("role") == "tool" for m in messages)
        if not saw_tool_result:
            msg = {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "read_file",
                             "arguments": json.dumps({
                                 "path": "/etc/hosts",
                                 "token": "sk-abcdef0123456789abcd"})}}]}
            finish = "tool_calls"
        else:
            msg = {"role": "assistant", "content": "done"}
            finish = "stop"
        body = json.dumps({
            "id": "x", "object": "chat.completion", "model": "m",
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_tool_args_redacted_in_info_logs(caplog, tmp_path):
    secret = "sk-abcdef0123456789abcd"
    with caplog.at_level(logging.INFO, logger="sable.agent"):
        with FakeOpenAIServer(handler_cls=ToolCallOnceHandler) as fs:
            client = SableAgentClient(
                fs.base_url, "m",
                tools=[{"type": "function", "function": {
                    "name": "read_file", "parameters": {}}}],
                tool_impls={"read_file": lambda path="", **kw: f"file:{path}"},
                max_iterations=3)
            res = client.run("read the hosts file")
    assert res.final_response == "done"
    blob = "\n".join(r.getMessage() for r in caplog.records)
    assert secret not in blob
    # The tool call itself was still logged (debuggability preserved)...
    assert "tool read_file" in blob
    # ...with the credential masked by the trace redactor.
    assert "[REDACTED]" in blob


# Fix 5 is a pure-whitespace change verified with `git diff -w`; no test.

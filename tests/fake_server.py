"""A fake OpenAI-compatible server for smoke tests (stdlib only).

Serves:
  GET  /v1/models           -> one model "fake-model"
  POST /v1/chat/completions -> scripted responses. If the last user/tool
      turn indicates tool results arrived, returns a final answer; otherwise
      (first call) emits a native tool_call for read_file to exercise the loop.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # silence
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") == "/v1/models":
            self._send({"object": "list", "data": [{"id": "fake-model", "object": "model"}]})
        else:
            self._send({"error": "not found"}, 404)

    def _completion_payload(self, payload):
        messages = payload.get("messages", [])
        saw_tool_result = any(m.get("role") == "tool" for m in messages)
        if not saw_tool_result:
            msg = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": json.dumps({"path": __file__}),
                        },
                    }
                ],
            }
            finish = "tool_calls"
        else:
            msg = {"role": "assistant", "content": "Done. The file was read successfully."}
            finish = "stop"
        return {
            "id": "cmpl-fake",
            "object": "chat.completion",
            "model": payload.get("model", "fake-model"),
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    def _send_sse(self, payload):
        """Server-Sent Events streaming response (stream=true)."""
        data = self._completion_payload(payload)
        choice = data["choices"][0]
        msg = choice["message"]

        def chunk(delta, finish_reason=None):
            return "data: " + json.dumps(
                {
                    "id": data["id"],
                    "object": "chat.completion.chunk",
                    "model": data["model"],
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
                }
            ) + "\n\n"

        parts = [chunk({"role": "assistant"})]
        if msg.get("tool_calls"):
            tc = msg["tool_calls"][0]
            parts.append(
                chunk(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["function"]["name"],
                                    "arguments": tc["function"]["arguments"],
                                },
                            }
                        ]
                    }
                )
            )
        if msg.get("content"):
            parts.append(chunk({"content": msg["content"]}))
        parts.append(chunk({}, finish_reason=choice["finish_reason"]))
        parts.append("data: [DONE]\n\n")

        body = "".join(parts).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if not self.path.startswith("/v1/chat/completions"):
            self._send({"error": "not found"}, 404)
            return
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if payload.get("stream"):
            self._send_sse(payload)
        else:
            self._send(self._completion_payload(payload))


def make_scripted_handler(script):
    """Handler class answering chat completions from a scripted map.

    ``script``: dict mapping substring-of-last-user-message -> reply string,
    plus optional key "*" (default reply) and callable values. Used by the
    Phase 4 validation-gate tests to deterministically pass/fail golden tasks.
    """

    class ScriptedHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # silence
            pass

        def _send(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/") == "/v1/models":
                self._send({"object": "list",
                            "data": [{"id": "scripted-model", "object": "model"}]})
            else:
                self._send({"error": "not found"}, 404)

        def do_POST(self):
            if not self.path.startswith("/v1/chat/completions"):
                self._send({"error": "not found"}, 404)
                return
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            messages = payload.get("messages", [])
            last_user = ""
            for m in reversed(messages):
                if m.get("role") == "user" and isinstance(m.get("content"), str):
                    last_user = m["content"]
                    break
            reply = script.get("*", "ok")
            for key, val in script.items():
                if key != "*" and key in last_user:
                    reply = val
                    break
            if callable(reply):
                reply = reply(last_user)
            self._send({
                "id": "cmpl-scripted",
                "object": "chat.completion",
                "model": payload.get("model", "scripted-model"),
                "choices": [{"index": 0,
                             "message": {"role": "assistant", "content": reply},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                          "total_tokens": 2},
            })

    return ScriptedHandler


def make_message_handler(script):
    """Handler class answering chat completions with full scripted messages.

    Like ``make_scripted_handler`` but values are complete assistant message
    dicts (``{"content": ..., "tool_calls": ...}``) instead of plain strings,
    so tests can script native tool-call replies (backlog #5). Values may also
    be callables ``payload -> message dict``. Lookup: substring match on the
    last user message, "*" default. Every request payload is appended to the
    class attribute ``requests`` so tests can assert what was sent (e.g. that
    expect_tool_call tasks ship tools=TOOL_SCHEMAS).
    """

    class MessageHandler(BaseHTTPRequestHandler):
        requests: list = []  # reset per handler class via make_message_handler

        def log_message(self, format, *args):  # silence
            pass

        def _send(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if not self.path.startswith("/v1/chat/completions"):
                self._send({"error": "not found"}, 404)
                return
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            type(self).requests.append(payload)
            messages = payload.get("messages", [])
            last_user = ""
            for m in reversed(messages):
                if m.get("role") == "user" and isinstance(m.get("content"), str):
                    last_user = m["content"]
                    break
            reply: Any = script.get("*", {"role": "assistant", "content": "ok"})
            for key, val in script.items():
                if key != "*" and key in last_user:
                    reply = val
                    break
            if callable(reply):
                reply = reply(payload)
            msg = dict(reply)
            finish = "tool_calls" if msg.get("tool_calls") else "stop"
            self._send({
                "id": "cmpl-msg",
                "object": "chat.completion",
                "model": payload.get("model", "msg-model"),
                "choices": [{"index": 0, "message": msg,
                             "finish_reason": finish}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                          "total_tokens": 2},
            })

    return MessageHandler


def fake_tokenize_count(text: str) -> int:
    """Deterministic fake tokenizer (improvement #4 tests): ceil(len/2).

    Deliberately distinct from the chars/4 heuristic so tests can prove the
    exact /tokenize path is actually driving gate decisions.
    """
    if not text:
        return 0
    return -(-len(text) // 2)


def make_tokenize_handler(script=None, *, initial_tokenize_status=200):
    """Handler class: scripted chat completions + a POST /tokenize endpoint.

    Chat behavior mirrors ``make_scripted_handler`` (substring match on the
    last user message, "*" default). /tokenize answers with the fake
    ceil(len/2) tokenizer and records every request's content in the class
    attribute ``tokenize_requests`` so tests can assert cache hits avoided a
    second HTTP call. ``tokenize_status`` (class attr, mutable per test)
    forces non-200 responses to exercise the fallback/circuit-breaker paths.
    """
    script_map = dict(script or {})

    class TokenizingHandler(BaseHTTPRequestHandler):
        tokenize_requests: list = []  # contents POSTed to /tokenize
        tokenize_status = int(initial_tokenize_status)

        def log_message(self, format, *args):  # silence
            pass

        def _send(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/") == "/v1/models":
                self._send({"object": "list",
                            "data": [{"id": "tok-model", "object": "model"}]})
            else:
                self._send({"error": "not found"}, 404)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            if self.path.rstrip("/") == "/tokenize":
                content = payload.get("content")
                # Record EVERY arriving request (even ones we then fail with a
                # forced status): the circuit-breaker tests assert that an
                # open breaker lets nothing reach the server at all.
                if isinstance(content, str):
                    type(self).tokenize_requests.append(content)
                if type(self).tokenize_status != 200:
                    self._send({"error": "tokenize unavailable"},
                               type(self).tokenize_status)
                    return
                if not isinstance(content, str):
                    self._send({"error": "content must be a string"}, 400)
                    return
                n = fake_tokenize_count(content)
                self._send({"tokens": list(range(n)), "count": n})
                return
            if not self.path.startswith("/v1/chat/completions"):
                self._send({"error": "not found"}, 404)
                return
            messages = payload.get("messages", [])
            last_user = ""
            for m in reversed(messages):
                if m.get("role") == "user" and isinstance(m.get("content"), str):
                    last_user = m["content"]
                    break
            reply = script_map.get("*", "ok")
            for key, val in script_map.items():
                if key != "*" and key in last_user:
                    reply = val
                    break
            if callable(reply):
                reply = reply(last_user)
            self._send({
                "id": "cmpl-tok",
                "object": "chat.completion",
                "model": payload.get("model", "tok-model"),
                "choices": [{"index": 0,
                             "message": {"role": "assistant", "content": reply},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                          "total_tokens": 2},
            })

    return TokenizingHandler


class FakeOpenAIServer:
    def __init__(self, handler_cls=None):
        self._httpd = ThreadingHTTPServer(
            ("127.0.0.1", 0), handler_cls or _Handler
        )
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}/v1"

    def start(self) -> "FakeOpenAIServer":
        self._thread.start()
        return self

    def stop(self):
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

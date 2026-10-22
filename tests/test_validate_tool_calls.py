"""Backlog item #5 tests — expect_tool_call checks in the golden suite.

Proves the tool-call grammar tripwire: native tool_calls responses and
text-emitted calls (recovered via core/tool_calls.parse_text_tool_calls) pass,
while prose-only answers FAIL an expect_tool_call check with kind="tool_call"
in the report — the silent-regression hole a quantization/merge bug leaves
open when substring checks still hit. Also covers arg_contains matching,
mixed substring+tool_call tasks, malformed argument JSON (fails cleanly), and
that transport retry (backlog #2) still applies to tool-call requests.
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fake_server import FakeOpenAIServer, make_message_handler  # noqa: E402
from sable.core.tools import TOOL_SCHEMAS  # noqa: E402
from sable.training.validate import run_golden_suite  # noqa: E402


@pytest.fixture(autouse=True)
def _no_sleeps(monkeypatch):
    """Retry backoff must not slow the suite down."""
    import sable.training.validate as v
    monkeypatch.setattr(v, "_sleep", lambda s: None)


def native_call_msg(name="read_file", arguments=None, content=None):
    if arguments is None:
        arguments = json.dumps({"path": "notes/todo.md"})
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": name, "arguments": arguments},
        }],
    }


def run(script, tasks):
    handler_cls = make_message_handler(script)
    with FakeOpenAIServer(handler_cls=handler_cls) as srv:
        res = run_golden_suite(srv.base_url, "m", tasks)
    return res, handler_cls.requests


TOOL_TASK = {
    "prompt": "read notes/todo.md please",
    "expect_tool_call": {"name": "read_file", "arg_contains": {"path": "notes"}},
}


# ------------------------------------------------------------- happy paths

def test_native_tool_calls_passes_expect_tool_call():
    res, reqs = run({"read notes": native_call_msg()}, [TOOL_TASK])
    assert res["pass_rate"] == 1.0
    entry = res["results"][0]
    assert entry["pass"] is True
    kinds = [c["kind"] for c in entry["checks"]]
    assert kinds == ["tool_call"]


def test_tool_task_request_ships_production_schemas():
    _, reqs = run({"read notes": native_call_msg()}, [TOOL_TASK])
    assert len(reqs) == 1
    sent = reqs[0].get("tools")
    assert sent == TOOL_SCHEMAS  # exact same schemas the agent uses


def test_substring_only_task_sends_no_tools_field():
    script = {"2+2": {"role": "assistant", "content": "4"}}
    res, reqs = run(script, [{"prompt": "what is 2+2?", "expect_contains": ["4"]}])
    assert res["pass_rate"] == 1.0
    assert "tools" not in reqs[0]  # byte-identical to pre-#5 traffic


def test_text_emitted_tool_call_recovered_and_passes():
    # A text-grammar model: fenced JSON block, no native field. The grammar
    # regression must still NOT be flagged — parse_text_tool_calls recovers it.
    msg = {
        "role": "assistant",
        "content": 'Sure.\n```json\n{"name": "read_file", '
                   '"arguments": {"path": "notes/todo.md"}}\n```',
    }
    res, _ = run({"read notes": msg}, [TOOL_TASK])
    assert res["pass_rate"] == 1.0
    check = res["results"][0]["checks"][0]
    assert check["kind"] == "tool_call" and check["pass"] is True


def test_name_only_check_accepts_any_args():
    msg = native_call_msg(name="list_dir", arguments=json.dumps({"path": "."}))
    res, _ = run({"read notes": msg},
                 [{"prompt": "read notes", "expect_tool_call": {"name": "list_dir"}}])
    assert res["pass_rate"] == 1.0


def test_no_name_accepts_any_tool_call():
    res, _ = run({"read notes": native_call_msg(name="search_files")},
                 [{"prompt": "read notes", "expect_tool_call": {}}])
    assert res["pass_rate"] == 1.0


# ------------------------------------------------------------- failure paths

def test_prose_only_fails_with_tool_call_kind():
    msg = {"role": "assistant", "content": "The file says hello."}
    res, _ = run({"read notes": msg}, [TOOL_TASK])
    assert res["pass_rate"] == 0.0
    entry = res["results"][0]
    assert entry["pass"] is False
    check = next(c for c in entry["checks"] if not c["pass"])
    assert check["kind"] == "tool_call"
    assert "prose only" in check["reason"]
    assert "read_file" in entry["error"] and "tool_call" in entry["error"]


def test_wrong_tool_name_fails():
    res, _ = run({"read notes": native_call_msg(name="write_file")}, [TOOL_TASK])
    assert res["passed"] == 0
    check = res["results"][0]["checks"][0]
    assert check["kind"] == "tool_call" and not check["pass"]
    assert "write_file" in check["reason"]


def test_arg_contains_match_is_case_insensitive_substring():
    msg = native_call_msg(arguments=json.dumps({"path": "/tmp/NOTES/todo.md"}))
    res, _ = run({"read notes": msg}, [TOOL_TASK])
    assert res["pass_rate"] == 1.0


def test_arg_contains_mismatch_fails():
    msg = native_call_msg(arguments=json.dumps({"path": "src/main.py"}))
    res, _ = run({"read notes": msg}, [TOOL_TASK])
    assert res["passed"] == 0
    check = res["results"][0]["checks"][0]
    assert check["kind"] == "tool_call" and not check["pass"]
    assert "path" in check["reason"]


def test_arg_contains_missing_key_fails():
    msg = native_call_msg(arguments=json.dumps({"offset": 1}))
    res, _ = run({"read notes": msg}, [TOOL_TASK])
    assert res["passed"] == 0
    assert "missing" in res["results"][0]["checks"][0]["reason"]


def test_malformed_args_json_fails_cleanly():
    # Unparseable arguments string — must fail the check, never raise.
    msg = native_call_msg(arguments="{not json at all")
    res, _ = run({"read notes": msg}, [TOOL_TASK])
    assert res["passed"] == 0
    check = res["results"][0]["checks"][0]
    assert check["kind"] == "tool_call" and not check["pass"]
    assert "malformed" in check["reason"]


# ---------------------------------------------------------------- mixed

def test_mixed_substring_and_tool_call_checks():
    task = {
        "prompt": "read notes/todo.md and summarize",
        "expect_contains": ["todo"],
        "expect_tool_call": {"name": "read_file"},
    }
    # Reply satisfies both: native call plus prose containing "todo".
    msg = native_call_msg(content="Working on todo.md now")
    res, _ = run({"read notes": msg}, [task])
    assert res["pass_rate"] == 1.0
    kinds = sorted(c["kind"] for c in res["results"][0]["checks"])
    assert kinds == ["substring", "tool_call"]

    # Substring hits but no tool call -> task fails on the tool_call kind.
    prose = {"role": "assistant", "content": "todo.md says nothing"}
    res, _ = run({"read notes": prose}, [task])
    assert res["passed"] == 0
    entry = res["results"][0]
    ok_kinds = {c["kind"] for c in entry["checks"] if c["pass"]}
    bad_kinds = {c["kind"] for c in entry["checks"] if not c["pass"]}
    assert ok_kinds == {"substring"} and bad_kinds == {"tool_call"}


def test_substring_failure_reported_with_substring_kind():
    task = {
        "prompt": "read notes/todo.md",
        "expect_contains": ["banana"],
        "expect_tool_call": {"name": "read_file"},
    }
    res, _ = run({"read notes": native_call_msg()}, [task])
    assert res["passed"] == 0
    entry = res["results"][0]
    bad = [c for c in entry["checks"] if not c["pass"]]
    assert len(bad) == 1 and bad[0]["kind"] == "substring"


# ------------------------------------------------------------- transport

class BlipThenToolCallHandler(BaseHTTPRequestHandler):
    """First POST dies with a connection reset; second returns a tool call."""

    blipped = False

    def log_message(self, format, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if not type(self).blipped:
            type(self).blipped = True
            self.close_connection = True
            try:
                self.wfile.close()
            except Exception:
                pass
            return
        body = json.dumps({
            "id": "x", "object": "chat.completion",
            "model": payload.get("model", "m"),
            "choices": [{"index": 0, "message": native_call_msg(),
                         "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                      "total_tokens": 2},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_transport_retry_applies_to_tool_call_tasks():
    BlipThenToolCallHandler.blipped = False
    with FakeOpenAIServer(handler_cls=BlipThenToolCallHandler) as srv:
        res = run_golden_suite(srv.base_url, "m", [TOOL_TASK])
    assert res["pass_rate"] == 1.0


# --------------------------------------------------- backward compatibility

def test_legacy_report_fields_preserved():
    script = {"hi": {"role": "assistant", "content": "hello world"}}
    res, _ = run(script, [{"prompt": "hi", "expect_contains": ["hello"],
                           "expect_not_contains": ["nope"]}])
    entry = res["results"][0]
    assert entry["missing"] == [] and entry["forbidden_hit"] == []
    assert "reply_preview" in entry and entry["pass"] is True

"""Verification-stop nudge (inspiration pass 3, item #19).

Unit coverage for the passive ledger (core/verification.py) plus runner-loop
integration through the scripted message handler: edit-then-finish gets one
nudge; verifying clears it; a second unverified finish records
failure_reason="verification_nudge"; prose edits never trigger anything.
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fake_server import FakeOpenAIServer  # noqa: E402
from sable.core.agent_runner import SableAgentClient  # noqa: E402
from sable.core.verification import (  # noqa: E402
    VerificationLedger,
    is_code_path,
)


# ------------------------------------------------------------ path policy

@pytest.mark.parametrize("path,want", [
    ("src/app.py", True),
    ("pkg/mod.js", True),
    ("x.html", True),
    ("Makefile", True),
    ("dockerfile", True),
    ("README.md", False),
    ("docs/guide.mdx", False),
    ("data/rows.jsonl", False),
    ("config.yaml", False),
    ("LICENSE", False),
    ("CHANGELOG", False),
    ("notes.txt", False),
    ("", False),
])
def test_is_code_path(path, want):
    assert is_code_path(path) is want


# ---------------------------------------------------------------- ledger

def test_ledger_edit_marks_dirty_and_nudge_lists_paths():
    led = VerificationLedger()
    led.observe("write_file", {"path": "a.py"}, "wrote a.py\n✓ syntax ok")
    assert led.pending_paths() == []          # fresh passing check
    led.observe("write_file", {"path": "b.py"}, "wrote b.py\n✗ SYNTAX ERROR x")
    assert led.pending_paths() == ["b.py"]
    nudge = led.build_nudge()
    assert nudge is not None and "b.py" in nudge
    assert led.build_nudge() is None          # exactly one nudge


def test_ledger_unverdict_edit_is_unverified():
    led = VerificationLedger()
    led.observe("write_file", {"path": "notes.md"}, "wrote notes.md")
    assert led.pending_paths() == []          # prose exempt
    led.observe("patch", {"path": "c.ts"}, "patched c.ts")  # no ✓/✗ verdict
    assert led.pending_paths() == ["c.ts"]


def test_ledger_failed_edit_leaves_no_trace():
    led = VerificationLedger()
    led.observe("write_file", {"path": "d.py"}, "error: workspace containment")
    assert led.pending_paths() == []


def test_ledger_run_command_clears_evidence():
    led = VerificationLedger()
    led.observe("patch", {"path": "e.py"}, "patched e.py\n✗ SYNTAX ERROR x")
    assert led.pending_paths() == ["e.py"]
    led.observe("run_command", {"command": "pytest"}, "all tests passed")
    assert led.pending_paths() == []
    # A FAILED command is not evidence.
    led.observe("patch", {"path": "f.py"}, "patched f.py")
    led.observe("run_command", {"command": "pytest"}, "error: boom")
    assert led.pending_paths() == ["f.py"]


# ------------------------------------------------------------ runner loop

def _tool_call_msg(name, args):
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_1", "type": "function",
         "function": {"name": name, "arguments": json.dumps(args)}}]}


class ReplySequenceHandler(BaseHTTPRequestHandler):
    """Replies per request index from a class-level script (then repeats last).

    Also records the messages of every request so tests can inspect whether
    the nudge was injected and where."""

    script: list = []
    requests: list = []

    def log_message(self, format, *args):  # noqa: A002 - silence access log
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        idx = len(type(self).requests)
        type(self).requests.append(payload.get("messages") or [])
        msg = type(self).script[min(idx, len(type(self).script) - 1)]
        finish = "tool_calls" if msg.get("tool_calls") else "stop"
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


def _client(script, tmp_path, monkeypatch, **kw):
    """Client with a stubbed write_file impl (no real FS needed)."""

    def fake_write(path, content, **_):
        # Mirrors the real post-write checker contract: a plain write carries
        # no verdict (dirty until something verifies it); writing the literal
        # content "VERIFIED" simulates a run whose check passed (✓ mark).
        if content == "VERIFIED":
            return f"wrote {path}\n✓ syntax ok"
        return f"wrote {path}"

    ReplySequenceHandler.script = script
    ReplySequenceHandler.requests = []
    fs = FakeOpenAIServer(handler_cls=ReplySequenceHandler)
    fs.start()
    client = SableAgentClient(
        fs.base_url, "m",
        tools=[{"type": "function", "function": {
            "name": "write_file", "description": "w",
            "parameters": {"type": "object",
                           "properties": {"path": {"type": "string"},
                                          "content": {"type": "string"}}}}}],
        tool_impls={"write_file": fake_write},
        max_iterations=6, **kw)
    return fs, client


def test_edit_then_done_gets_one_nudge_then_verified_finish(tmp_path):
    script = [
        _tool_call_msg("write_file", {"path": "a.py", "content": "x"}),
        {"role": "assistant", "content": "all done!"},          # unverified
        _tool_call_msg("write_file", {"path": "a.py",
                                      "content": "VERIFIED"}),  # fresh ✓
        {"role": "assistant", "content": "verified via re-write"},
    ]
    fs, client = _client(script, tmp_path, None)
    try:
        res = client.run("edit a.py")
    finally:
        fs.stop()
    assert res.completed and res.failure_reason is None
    # The nudge went in as a synthetic user message after the first "done".
    nudges = [m for m in res.messages
              if m.get("role") == "user" and "verification" in
              str(m.get("content", "")).lower()]
    assert len(nudges) == 1
    assert "a.py" in nudges[0]["content"]
    assert res.final_response == "verified via re-write"


def test_second_unverified_finish_records_failure_reason(tmp_path):
    script = [
        _tool_call_msg("write_file", {"path": "b.py", "content": "x"}),
        {"role": "assistant", "content": "done!"},   # unverified -> nudge
        {"role": "assistant", "content": "really done"},  # still unverified
    ]
    fs, client = _client(script, tmp_path, None)
    try:
        res = client.run("edit b.py")
    finally:
        fs.stop()
    assert res.completed is True
    assert res.failure_reason == "verification_nudge"
    # Exactly one nudge in history — no nagging on the third turn.
    nudges = [m for m in res.messages if m.get("role") == "user"
              and "[System:" in str(m.get("content", ""))]
    assert len(nudges) == 1


def test_prose_edit_never_nudges(tmp_path):
    script = [
        _tool_call_msg("write_file", {"path": "README.md", "content": "x"}),
        {"role": "assistant", "content": "docs updated"},
    ]
    fs, client = _client(script, tmp_path, None)
    try:
        res = client.run("edit the readme")
    finally:
        fs.stop()
    assert res.failure_reason is None
    assert all("[System:" not in str(m.get("content", ""))
               for m in res.messages if m.get("role") == "user"
               and m["content"] != "edit the readme")


def test_nudge_disabled_by_flag(tmp_path):
    script = [
        _tool_call_msg("write_file", {"path": "c.py", "content": "x"}),
        {"role": "assistant", "content": "done!"},
    ]
    fs, client = _client(script, tmp_path, None, verification_nudge=False)
    try:
        res = client.run("edit c.py")
    finally:
        fs.stop()
    assert res.failure_reason is None
    assert res.final_response == "done!"

"""Backlog #3 tests — tool-loop guardrails + repetition guard.

Covers: signature stability (arg key order irrelevant), the threshold ladder
(warn -> block -> halt in sequence), idempotent no-progress detection,
mutating tools never no-progress blocked, config-off disables everything,
and runner integration: a same-tool failure spiral halts before
max_iterations with failure_reason guardrail_halt:*; warn/block surfaced in
tool results; repetition loops truncated in the final answer and summary.
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fake_server import FakeOpenAIServer  # noqa: E402
from sable.core.agent_runner import SableAgentClient, create_runner  # noqa: E402
from sable.core.config import SableConfig  # noqa: E402
from sable.core.failure_types import is_infra_reason  # noqa: E402
from sable.core.guardrails import (  # noqa: E402
    IDEMPOTENT_TOOL_NAMES,
    MUTATING_TOOL_NAMES,
    REPETITION_MARKER,
    ToolCallSignature,
    ToolLoopGuardrails,
    detect_repetition,
    find_loop,
    truncate_repetition,
)


@pytest.fixture()
def sable_home(tmp_path, monkeypatch):
    home = tmp_path / "uf-home"
    monkeypatch.setenv("SABLE_HOME", str(home))
    return home


# ------------------------------------------------------------- signatures

def test_signature_ignores_arg_key_order():
    a = ToolCallSignature.from_call("patch", {"path": "x", "old_string": "a"})
    b = ToolCallSignature.from_call("patch", {"old_string": "a", "path": "x"})
    assert a == b


def test_signature_accepts_json_string_args():
    a = ToolCallSignature.from_call("patch", '{"b": 2, "a": 1}')
    b = ToolCallSignature.from_call("patch", {"a": 1, "b": 2})
    assert a == b
    # malformed / non-mapping args collapse to {} deterministically
    assert (ToolCallSignature.from_call("patch", "{oops")
            == ToolCallSignature.from_call("patch", None))


def test_signature_differs_per_tool_and_args():
    base = ToolCallSignature.from_call("patch", {"a": 1})
    assert base != ToolCallSignature.from_call("write_file", {"a": 1})
    assert base != ToolCallSignature.from_call("patch", {"a": 2})


# --------------------------------------------------------- threshold ladder

def test_exact_failure_ladder_warn_then_block_then_halt():
    g = ToolLoopGuardrails()
    args = {"path": "f.py", "old_string": "x"}

    actions = [g.after_call("patch", args, "error: nope").action for _ in range(5)]
    # call 1 allow; calls 2..5 warn (exact_failure_warn_after=2)
    assert actions == ["allow", "warn", "warn", "warn", "warn"]

    # after 5 identical failures the 6th identical call is blocked before exec
    d = g.before_call("patch", args)
    assert d.action == "block" and d.code == "repeated_exact_failure_block"

    # if the model keeps re-failing the same tool (counted anyway), the
    # same-tool counter halts at 8
    decisions = [g.after_call("patch", {"path": f"f{i}.py"}, "error: nope")
                 for i in range(3)]
    assert decisions[-1].action == "halt"
    assert decisions[-1].code == "same_tool_failure_halt"


def test_same_tool_varied_args_warn_then_halt():
    g = ToolLoopGuardrails()
    acts = [g.after_call("patch", {"path": f"f{i}.py"}, "error: boom").action
            for i in range(8)]
    # same_tool_failure_warn_after=3 -> warn from the 3rd failure on
    assert acts[0] == "allow" and acts[1] == "allow"
    assert acts[2] == "warn"
    assert acts[-1] == "halt"


def test_success_clears_failure_streaks():
    g = ToolLoopGuardrails()
    args = {"path": "f.py"}
    g.after_call("patch", args, "error: one")
    g.after_call("patch", args, "error: two")  # would warn on next failure
    d = g.after_call("patch", args, "ok: written")
    assert d.action == "allow"
    d2 = g.after_call("patch", args, "error: three")
    assert d2.action == "allow"  # counters were reset by the success


# ------------------------------------------------------------ no progress

def test_idempotent_no_progress_warn_then_block():
    g = ToolLoopGuardrails()
    args = {"path": "a.txt"}
    acts = [g.after_call("read_file", args, "same content").action
            for _ in range(5)]
    assert acts == ["allow", "warn", "warn", "warn", "warn"]
    d = g.before_call("read_file", args)
    assert d.action == "block" and d.code == "idempotent_no_progress_block"


def test_idempotent_changed_result_resets_no_progress():
    g = ToolLoopGuardrails()
    args = {"path": "a.txt"}
    g.after_call("read_file", args, "v1")
    g.after_call("read_file", args, "v2")  # changed -> streak resets
    d = g.after_call("read_file", args, "v3")
    assert d.action == "allow"


def test_mutating_tools_never_no_progress_blocked():
    g = ToolLoopGuardrails()
    args = {"path": "a.txt", "content": "x"}
    for _ in range(6):  # identical successful writes: never flagged
        d = g.after_call("write_file", args, "ok: wrote a.txt")
        assert d.action == "allow"
    assert g.before_call("write_file", args).action == "allow"


def test_unknown_tools_treated_as_mutating():
    g = ToolLoopGuardrails()
    for _ in range(6):
        assert g.after_call("mystery_tool", {"a": 1}, "same").action == "allow"
    assert g.before_call("mystery_tool", {"a": 1}).action == "allow"


def test_tool_sets_match_spec():
    assert IDEMPOTENT_TOOL_NAMES == {
        "read_file", "list_dir", "search_files", "todo_read", "search_traces"}
    assert MUTATING_TOOL_NAMES == {"write_file", "patch", "todo_write"}


# ------------------------------------------------------------------- off

def test_disabled_controller_always_allows():
    g = ToolLoopGuardrails(enabled=False)
    for _ in range(10):
        assert g.after_call("patch", {"a": 1}, "error: boom").action == "allow"
        assert g.before_call("patch", {"a": 1}).action == "allow"


# ------------------------------------------------------- repetition guard

def _loop_text(prefix: str = "", repeats: int = 12) -> str:
    unit = "the quick brown fox jumps over the lazy dog again and again.\n"  # 60 chars
    return prefix + unit * repeats


def test_detect_repetition_flags_loop():
    text = _loop_text()
    assert len(text) >= 400
    assert detect_repetition(text) is True
    # loop starts at index 0 (no prefix in this fixture)
    assert find_loop(text) == 0


def test_detect_repetition_ignores_normal_text():
    normal = "sable records traces to ~/.sable/buffer. " * 3
    assert detect_repetition(normal) is False
    assert detect_repetition("short loop loop loop") is False
    assert detect_repetition("") is False


def test_truncate_keeps_prefix_and_marks():
    text = _loop_text(prefix="Here is the answer you asked for.\n")
    out = truncate_repetition(text)
    assert out.startswith("Here is the answer you asked for.")
    assert out.endswith(REPETITION_MARKER)
    assert len(out) < len(text)


def test_truncate_clean_text_unchanged():
    text = "a perfectly normal response.\n" * 3
    assert truncate_repetition(text) == text


# ------------------------------------------------------ runner integration

class AlwaysFailingPatchHandler(BaseHTTPRequestHandler):
    """Always requests the same patch call; args vary per request (counter)
    so the exact-failure block never pre-empts the same-tool halt."""

    n = 0

    def log_message(self, format, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        type(self).n += 1
        msg = {"role": "assistant", "content": None, "tool_calls": [{
            "id": f"c{type(self).n}", "type": "function",
            "function": {"name": "patch", "arguments": json.dumps(
                {"path": f"missing{type(self).n}.py",
                 "old_string": "x", "new_string": "y"})}}]}
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


class IdenticalFailingPatchHandler(BaseHTTPRequestHandler):
    """Re-issues the byte-identical failing patch call forever."""

    def log_message(self, format, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        msg = {"role": "assistant", "content": None, "tool_calls": [{
            "id": "c", "type": "function",
            "function": {"name": "patch", "arguments": json.dumps(
                {"path": "missing.py", "old_string": "x",
                 "new_string": "y"})}}]}
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


def failing_patch_impl(**kwargs) -> str:
    return "error: patch failed: file not found"


@pytest.fixture(autouse=True)
def _reset_handler_counters():
    AlwaysFailingPatchHandler.n = 0
    yield


def test_run_halts_before_max_iterations_with_guardrail_reason(sable_home):
    from sable.store.recorder import TraceRecorder
    rec = TraceRecorder(home=sable_home)
    with FakeOpenAIServer(handler_cls=AlwaysFailingPatchHandler) as srv:
        client = SableAgentClient(base_url=srv.base_url, model="m",
                                    tool_impls={"patch": failing_patch_impl},
                                    max_iterations=12, recorder=rec)
        result = client.run("fix the file")
    assert result.completed is False
    assert result.failure_reason == "guardrail_halt:same_tool_failure_halt"
    assert is_infra_reason(result.failure_reason) is False  # model-ish failure
    assert AlwaysFailingPatchHandler.n < 12  # halted before max_iterations
    assert result.trace_id is not None

    footer = _footer_for(sable_home, result.trace_id)
    assert footer["failure_reason"] == "guardrail_halt:same_tool_failure_halt"
    assert footer["completed"] is False


def test_run_warn_and_block_surfaced_in_tool_results():
    with FakeOpenAIServer(handler_cls=IdenticalFailingPatchHandler) as srv:
        client = SableAgentClient(base_url=srv.base_url, model="m",
                                    tool_impls={"patch": failing_patch_impl},
                                    max_iterations=8)
        result = client.run("fix the file")
    tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
    # warn nudge appended to a tool result
    assert any("[Tool loop warning: repeated_exact_failure_warning" in m["content"]
               for m in tool_msgs)
    # block -> synthetic error tool result, execution skipped, loop continued
    assert any(m["content"].startswith("error: blocked by guardrail "
                                       "(repeated_exact_failure_block)")
               for m in tool_msgs)
    # identical-call spiral ends at max_iterations (blocks never halt)
    assert result.failure_reason == "max_iterations"


def test_final_answer_repetition_truncated():
    loop = _loop_text(prefix="Answer:\n")

    class LoopAnswerHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            body = json.dumps({
                "id": "x", "object": "chat.completion",
                "model": payload.get("model", "m"),
                "choices": [{"index": 0,
                             "message": {"role": "assistant", "content": loop},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                          "total_tokens": 2}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    with FakeOpenAIServer(handler_cls=LoopAnswerHandler) as srv:
        client = SableAgentClient(base_url=srv.base_url, model="m")
        result = client.run("tell me a story")
    assert result.completed is True
    assert result.final_response.endswith(REPETITION_MARKER)
    assert result.final_response.startswith("Answer:")
    assert len(result.final_response) < len(loop)


def test_summarize_output_repetition_truncated():
    loop = _loop_text(prefix="Summary:\n")

    class LoopSummaryHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_POST(self):
            body = json.dumps({
                "id": "x", "object": "chat.completion", "model": "m",
                "choices": [{"index": 0,
                             "message": {"role": "assistant", "content": loop},
                             "finish_reason": "stop"}],
                "usage": {}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    with FakeOpenAIServer(handler_cls=LoopSummaryHandler) as srv:
        client = SableAgentClient(base_url=srv.base_url, model="m")
        out = client._summarize("summarize please")
    assert out is not None
    assert out.endswith(REPETITION_MARKER)
    assert len(out) < len(loop)


def test_guardrails_off_runs_to_max_iterations(sable_home):
    with FakeOpenAIServer(handler_cls=AlwaysFailingPatchHandler) as srv:
        client = SableAgentClient(base_url=srv.base_url, model="m",
                                    tool_impls={"patch": failing_patch_impl},
                                    max_iterations=4, guardrails_enabled=False)
        result = client.run("fix the file")
    assert result.failure_reason == "max_iterations"
    assert AlwaysFailingPatchHandler.n == 4


def test_create_runner_config_knob():
    cfg_on = SableConfig()
    assert create_runner(cfg_on, base_url="http://127.0.0.1:9/v1",
                         model="m").guardrails_enabled is True
    cfg_off = SableConfig(agent={"guardrails": False})
    assert create_runner(cfg_off, base_url="http://127.0.0.1:9/v1",
                         model="m").guardrails_enabled is False


# ------------------------------------------------------------------ utils

def _footer_for(home: Path, trace_id: str) -> dict:
    matches = list((home / "buffer").rglob(f"{trace_id}.jsonl"))
    assert len(matches) == 1
    lines = [json.loads(l) for l in matches[0].read_text().splitlines()]
    return lines[-1]

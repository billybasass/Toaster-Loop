"""Backlog #7 unit tests — core/failure_types.py shared classifier."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sable.core.failure_types import (  # noqa: E402
    classify_tool_result,
    is_cancelled_reason,
    is_infra_reason,
    is_model_reason,
)


# ------------------------------------------------------ classify_tool_result

def test_golden_pass_shapes():
    ok = [
        "a.txt\nb.txt",
        "wrote 12 chars to /tmp/x",
        json.dumps({"ok": True}),
        json.dumps({"returncode": 0, "stdout": "hi"}),
        json.dumps({"exit_code": 0}),
    ]
    for content in ok:
        assert classify_tool_result("t", content) == (False, "none"), content


def test_golden_failure_shapes():
    assert classify_tool_result("read_file", "error: not a file: x") == (
        True, "error_tag")
    assert classify_tool_result("read_file", "Error reading x: boom") == (
        True, "error_tag")
    assert classify_tool_result(
        "patch", "Traceback (most recent call last):\n  ...") == (
        True, "traceback")
    assert classify_tool_result("patch", "ValueError exception raised here") == (
        True, "exception")
    assert classify_tool_result("t", json.dumps({"error": "bad args"})) == (
        True, "tool_error_json")
    assert classify_tool_result("t", json.dumps({"returncode": 1})) == (
        True, "tool_error_json")
    assert classify_tool_result("t", json.dumps({"exit_code": 2})) == (
        True, "tool_error_json")
    assert classify_tool_result("t", "") == (True, "empty")
    assert classify_tool_result("t", "   \n ") == (True, "empty")


def test_json_non_dict_and_invalid_json_tolerated():
    assert classify_tool_result("t", "[1, 2, 3]") == (False, "none")
    assert classify_tool_result("t", "{not valid json error}") == (False, "none")
    # traceback beats generic error tag priority
    assert classify_tool_result(
        "t", "error while running: Traceback (most recent call last):") == (
        True, "traceback")


# ------------------------------------- HIGH #5 false-positive regression corpus

def test_false_positive_corpus_classifies_as_success():
    """Bare 'traceback'/'exception' mentions are NOT failures (HIGH #5).

    Every entry below is ordinary coding-agent tool output that the old
    substring checks mis-classified as a failed tool result.
    """
    fp = [
        # read_file on Python source with an except clause + parenthesized raise
        "def parse(text):\n"
        "    try:\n"
        "        return json.loads(text)\n"
        "    except Exception as e:\n"
        '        raise ValueError("bad input") from e\n',
        # grep/ripgrep listing files that mention the words
        "docs/errors.md:14:The Traceback module renders call stacks\n"
        'core/failure_types.py:80:# bare substring "traceback" false positives\n'
        "src/handler.py:9:    except Exception:\n"
        "        pass  # swallow",
        # JSON payload with exception-flavoured keys
        json.dumps({"exception_count": 3, "traceback": None, "status": "ok"}),
        # prose mentioning exceptions and tracebacks (lower-case words)
        "The handler catches exceptions from the backend and logs a "
        "traceback when one occurs.",
    ]
    for content in fp:
        assert classify_tool_result("read_file", content) == (False, "none"), \
            repr(content)


def test_true_positive_shapes_still_fail():
    """Real traceback bodies and exception-class lines still classify as failure."""
    tb = ("Traceback (most recent call last):\n"
          '  File "x.py", line 3, in <module>\n'
          "    1 / 0\n"
          "ZeroDivisionError: division by zero")
    assert classify_tool_result("bash", tb) == (True, "traceback")
    # trailing frames still carry the canonical header -> traceback wins
    assert classify_tool_result(
        "patch",
        "process exited nonzero\nTraceback (most recent call last):\n  ...") == (
        True, "traceback")
    # exception-class + colon shapes (bare raise output, dotted module classes)
    assert classify_tool_result("bash", "ValueError: bad input") == (
        True, "exception")
    assert classify_tool_result("t", "json.JSONDecodeError: Expecting value") == (
        True, "exception")
    # ERROR:/error: tool-failure tags preserved
    assert classify_tool_result("t", "ERROR: command not found") == (
        True, "error_tag")
    assert classify_tool_result("t", "error: no such file") == (
        True, "error_tag")


def test_is_cancelled_reason():
    assert is_cancelled_reason("cancelled")
    assert is_cancelled_reason("backend_error:cancelled")  # composed form
    for r in ("", "connection", "timeout", "max_iterations",
              "backend_error:timeout", "guardrail_halt"):
        assert not is_cancelled_reason(r)


# ------------------------------------------------------------- reason gates

def test_is_infra_reason():
    for r in ("connection", "timeout", "model_not_found"):
        assert is_infra_reason(r)
        assert is_infra_reason(f"backend_error:{r}")  # composed runner form
    for r in ("", "max_iterations", "context_overflow", "guardrail_halt",
              "format_error", "payload_too_large"):
        assert not is_infra_reason(r)


def test_is_model_reason():
    for r in ("context_overflow", "payload_too_large", "format_error",
              "max_iterations", "guardrail_halt"):
        assert is_model_reason(r)
        assert is_model_reason(f"backend_error:{r}")
    for r in ("", "connection", "timeout", "model_not_found"):
        assert not is_model_reason(r)

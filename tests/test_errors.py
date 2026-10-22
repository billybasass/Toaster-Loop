"""Backlog #2 unit tests — core/errors.py classification (golden strings)."""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sable.core.agent_runner import AgentBackendError  # noqa: E402
from sable.core.errors import (  # noqa: E402
    FailoverReason,
    classify_exception,
    classify_http_status,
)


# ------------------------------------------------------------- http statuses

def test_classify_http_status_golden_strings():
    ce = classify_http_status(413, "request body too large")
    assert (ce.reason, ce.retryable, ce.should_compress) == (
        FailoverReason.PAYLOAD_TOO_LARGE, True, True)

    ce = classify_http_status(
        400, "the prompt is longer than the context length of 4096 tokens")
    assert (ce.reason, ce.should_compress) == (FailoverReason.CONTEXT_OVERFLOW, True)

    ce = classify_http_status(404, '{"error": "model not found"}')
    assert (ce.reason, ce.retryable) == (FailoverReason.MODEL_NOT_FOUND, True)

    ce = classify_http_status(400, "malformed tool call json in request")
    assert (ce.reason, ce.strip_tools, ce.retryable) == (
        FailoverReason.FORMAT_ERROR, True, True)

    ce = classify_http_status(400, "something else entirely")
    assert (ce.reason, ce.retryable) == (FailoverReason.FORMAT_ERROR, False)

    ce = classify_http_status(500, "boom")
    assert (ce.reason, ce.retryable) == (FailoverReason.CONNECTION, True)

    ce = classify_http_status(418, "teapot")
    assert (ce.reason, ce.retryable) == (FailoverReason.UNKNOWN, False)


# ---------------------------------------------------------------- exceptions

def test_classify_exception_transport_types():
    ce = classify_exception(httpx.ConnectError("connection refused"))
    assert (ce.reason, ce.retryable) == (FailoverReason.CONNECTION, True)

    ce = classify_exception(httpx.ReadTimeout("timed out"))
    assert (ce.reason, ce.retryable) == (FailoverReason.TIMEOUT, True)

    # wrapped transport error via __cause__ (runner raises from e)
    inner = httpx.RemoteProtocolError("peer closed connection")
    wrapped = AgentBackendError("Cannot reach inference server")
    wrapped.__cause__ = inner
    assert classify_exception(wrapped).reason == FailoverReason.CONNECTION


def test_classify_exception_message_rules():
    ce = classify_exception(AgentBackendError(
        "Chat completion failed: HTTP 500 — llama server OOM-retry"))
    assert (ce.reason, ce.retryable) == (FailoverReason.CONNECTION, True)

    ce = classify_exception(RuntimeError(
        "requested tokens 9000 exceeds the model context length of 8192"))
    assert (ce.reason, ce.should_compress) == (FailoverReason.CONTEXT_OVERFLOW, True)

    ce = classify_exception(RuntimeError("The requested model does not exist"))
    assert ce.reason == FailoverReason.MODEL_NOT_FOUND

    ce = classify_exception(RuntimeError(
        'server rejected tool_calls: failed to parse json arguments'))
    assert (ce.reason, ce.strip_tools) == (FailoverReason.FORMAT_ERROR, True)


def test_classify_exception_unknown_is_retryable_once():
    ce = classify_exception(RuntimeError("something we have never seen"))
    assert (ce.reason, ce.retryable) == (FailoverReason.UNKNOWN, True)

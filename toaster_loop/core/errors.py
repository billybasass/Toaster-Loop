"""Minimal error taxonomy for the loopback runner (backlog item #2).

Built in
(attribution: classify -> ClassifiedError(reason, retryable, hints) with
priority-ordered rules so the retry loop acts on hints instead of
re-inspecting the error). Only the loopback-relevant subset of
``FailoverReason`` survives — toaster_loop talks to one llama.cpp server it owns,
so no provider/auth/billing machinery exists here.

Two entry points:

    classify_exception(exc)          # exception type + message sniffing
    classify_http_status(status, body)  # non-200 from the local server
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import httpx


class FailoverReason:
    """Loopback-relevant failure reasons (string constants)."""

    CONNECTION = "connection"              # transport failure — backoff + retry
    TIMEOUT = "timeout"                    # read/connect timeout — retry
    CONTEXT_OVERFLOW = "context_overflow"  # prompt too big — compress, retry once
    PAYLOAD_TOO_LARGE = "payload_too_large"  # HTTP 413 — compress, retry once
    FORMAT_ERROR = "format_error"          # HTTP 400 — maybe strip tools, retry
    MODEL_NOT_FOUND = "model_not_found"    # 404 / hot-swap race — re-list models
    UNKNOWN = "unknown"                    # unclassifiable


@dataclass
class ClassifiedError:
    """Classification verdict + recovery hints for the retry loop."""

    reason: str
    retryable: bool = False
    should_compress: bool = False   # force the compression path before retry
    strip_tools: bool = False       # drop the tools argument before retry
    message: str = ""


# --- message/body sniffing patterns (priority-ordered below) ----------------

_CONTEXT_PATTERNS = (
    "context length", "context_length", "too long", "exceeds the",
    "n_positions", "maximum context", "requested tokens",
)
_PAYLOAD_PATTERNS = ("payload too large", "request entity too large")
_MALFORMED_TOOL_PATTERNS = (
    "malformed", "not valid json", "invalid json", "json parse",
    "failed to parse", "tool call",
)
_MODEL_NOT_FOUND_PATTERNS = (
    "model not found", "model_not_found", "does not exist",
    "invalid model", "no such model",
)
_TIMEOUT_PATTERNS = ("timed out", "timeout")
_CONNECTION_PATTERNS = (
    "cannot reach", "connection", "refused", "reset by peer",
    "unreachable", "broken pipe", "eof occurred",
)

_HTTP_STATUS_RE = re.compile(r"HTTP (\d{3})")


def classify_http_status(status: int, body: str = "") -> ClassifiedError:
    """Classify a non-200 response from the local OpenAI-compatible server."""
    b = (body or "").lower()
    if status == 413 or any(p in b for p in _PAYLOAD_PATTERNS):
        return ClassifiedError(FailoverReason.PAYLOAD_TOO_LARGE, retryable=True,
                               should_compress=True, message=f"HTTP {status}")
    if any(p in b for p in _CONTEXT_PATTERNS):
        return ClassifiedError(FailoverReason.CONTEXT_OVERFLOW, retryable=True,
                               should_compress=True, message=f"HTTP {status}")
    if status == 404:
        return ClassifiedError(FailoverReason.MODEL_NOT_FOUND, retryable=True,
                               message=f"HTTP {status}")
    if status == 408 or any(p in b for p in _TIMEOUT_PATTERNS):
        return ClassifiedError(FailoverReason.TIMEOUT, retryable=True,
                               message=f"HTTP {status}")
    if status == 400:
        if any(p in b for p in _MALFORMED_TOOL_PATTERNS):
            # Malformed tool-call json etc: one tools-stripped retry may help.
            return ClassifiedError(FailoverReason.FORMAT_ERROR, retryable=True,
                                   strip_tools=True, message=f"HTTP {status}")
        return ClassifiedError(FailoverReason.FORMAT_ERROR, retryable=False,
                               message=f"HTTP {status}")
    if status == 429 or 500 <= status:
        # llama-server restarts/OOM-retries surface as 5xx/429 — transient.
        return ClassifiedError(FailoverReason.CONNECTION, retryable=True,
                               message=f"HTTP {status}")
    return ClassifiedError(FailoverReason.UNKNOWN, retryable=False,
                           message=f"HTTP {status}")


def classify_exception(exc: BaseException) -> ClassifiedError:
    """Priority-ordered classification of an exception raised at a call site."""
    # 1. transport types (timeout first: it is a TransportError subclass)
    if isinstance(exc, httpx.TimeoutException):
        return ClassifiedError(FailoverReason.TIMEOUT, retryable=True,
                               message=str(exc))
    if isinstance(exc, (httpx.TransportError, ConnectionError, OSError)):
        return ClassifiedError(FailoverReason.CONNECTION, retryable=True,
                               message=str(exc))
    cause = exc.__cause__
    if cause is not None and cause is not exc:
        if isinstance(cause, httpx.TimeoutException):
            return ClassifiedError(FailoverReason.TIMEOUT, retryable=True,
                                   message=str(exc))
        if isinstance(cause, (httpx.TransportError, ConnectionError, OSError)):
            return ClassifiedError(FailoverReason.CONNECTION, retryable=True,
                                   message=str(exc))

    # 2. embedded HTTP status (runner wraps server errors as "HTTP <code> — …")
    text = str(exc)
    m = _HTTP_STATUS_RE.search(text)
    if m:
        return classify_http_status(int(m.group(1)), text)

    # 3. message substrings, most specific first
    low = text.lower()
    if any(p in low for p in _CONTEXT_PATTERNS):
        return ClassifiedError(FailoverReason.CONTEXT_OVERFLOW, retryable=True,
                               should_compress=True, message=text)
    if any(p in low for p in _PAYLOAD_PATTERNS):
        return ClassifiedError(FailoverReason.PAYLOAD_TOO_LARGE, retryable=True,
                               should_compress=True, message=text)
    if any(p in low for p in _MODEL_NOT_FOUND_PATTERNS):
        return ClassifiedError(FailoverReason.MODEL_NOT_FOUND, retryable=True,
                               message=text)
    if any(p in low for p in _MALFORMED_TOOL_PATTERNS) and "json" in low:
        return ClassifiedError(FailoverReason.FORMAT_ERROR, retryable=True,
                               strip_tools=True, message=text)
    if any(p in low for p in _TIMEOUT_PATTERNS):
        return ClassifiedError(FailoverReason.TIMEOUT, retryable=True,
                               message=text)
    if any(p in low for p in _CONNECTION_PATTERNS):
        return ClassifiedError(FailoverReason.CONNECTION, retryable=True,
                               message=text)
    # 4. catch-all: one capped backoff retry can rescue transient blips.
    return ClassifiedError(FailoverReason.UNKNOWN, retryable=True, message=text)

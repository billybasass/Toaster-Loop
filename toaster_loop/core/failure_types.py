"""Single source of truth for tool-result failure classification (backlog #7).

Detection shape built in
``agent/tool_result_classification.py`` (attribution: one shared classifier
so the recorder heuristic, the curater verifier and future guardrails can
never disagree about what "this tool result failed" means).

toaster_loop previously duplicated — and slightly diverged — this logic in
``store/recorder.py::heuristic_score`` and
``curate/curater.py::verifier_tool_errors``. Both now call
``classify_tool_result``.
"""

from __future__ import annotations

import json
import re
from typing import Tuple

# Failure kinds ("" == not failed). Keep in sync with tests/test_core.py.
KIND_NONE = "none"
KIND_ERROR_TAG = "error_tag"          # content starts with "error"/"Error"
KIND_TRACEBACK = "traceback"          # python traceback text
KIND_EXCEPTION = "exception"          # exception-flavoured text
KIND_TOOL_ERROR_JSON = "tool_error_json"  # JSON body with error / nonzero exit
KIND_EMPTY = "empty"                  # empty/whitespace result

# Runner-level failure_reason constants (trace footer; backlog #2/#7). The
# runner records "backend_error:<reason>" by formatting with the classified
# FailoverReason from core/errors.py.
FAILURE_MAX_ITERATIONS = "max_iterations"
FAILURE_GUARDRAIL_HALT = "guardrail_halt"  # reserved: guardrails land in a later batch
FAILURE_BACKEND_ERROR_PREFIX = "backend_error:"
# Inspiration pass 3 #19: the run finished, but only after a verification nudge
# was already spent and code stayed unverified ("claimed done, never checked").
FAILURE_VERIFICATION_NUDGE = "verification_nudge"
# Inspiration pass 4 #28: the per-backend circuit breaker is OPEN, so the call
# was refused without touching the network. Infra-classified (see below), so
# curation down-weights these traces exactly like connection faults.
FAILURE_BACKEND_CIRCUIT_OPEN = "backend_circuit_open"
# A human pressed Stop. Not an infra fault (the backend was healthy) and not a
# model fault (nothing about the model's behavior is at issue) — so it lands in
# the "other" bucket: recorded for the buffer, but excluded from DPO-negative
# material by is_model_reason and NOT auto-retried by batch resume.
FAILURE_CANCELLED = "cancelled"


# ------------------------------------------------------------- tool-result shapes
# HIGH #5: shape-based detection replaces the old bare-substring checks. The old
# `"traceback" in low` / `"exception" in low` turned the classifier into a
# false-positive machine for a coding agent: read_file on Python source with
# `except Exception as e:` or grep output listing the word "Traceback" counted
# as a failed tool result — penalizing heuristic_score, feeding
# drop_tool_error_only filtering and polluting DPO-negative selection.

# Canonical Python traceback header (case-insensitive); tolerant of leading
# output before it ("error while running: Traceback (most recent call last):").
_TRACEBACK_RE = re.compile(r"\btraceback \(most recent call last\b", re.IGNORECASE)

# Exception-class shape: a word-boundary-anchored, optionally dotted identifier
# ending in one of the canonical suffixes, followed either by ":" — the shape
# real raises print ("ValueError: bad input", "json.JSONDecodeError: ...") — or
# by the word "exception" ("ValueError exception raised here"). The bare word
# "exception" (prose, `except Exception as e:`, an "exception_count" JSON key)
# matches neither alternative. Case-sensitive on purpose: Python exception
# classes are CamelCase; lowercase prose/error-tag text stays out of this kind.
_EXCEPTION_CLASS_RE = re.compile(
    r"(?<!except )(?<!raise )"  # source-code "except ValueError:" / "raise Foo(x):"
    r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*"
    r"(?:Error|Exception|Failure|Interrupt|Exit)(?![A-Za-z0-9_])"
    r"(?::|\s+exception\b)"
)


def classify_tool_result(name: str, content: str) -> Tuple[bool, str]:
    """Return (failed, kind) for one tool-result payload.

    Priority-ordered so the most specific kinds win: JSON error bodies and
    tracebacks beat the plain "error" tag; empty is last (a blank result from
    any tool is suspicious but harmless to score). ``name`` is accepted for
    future per-tool policy but currently unused.
    """
    if content is None:
        return True, KIND_EMPTY
    text = content if isinstance(content, str) else json.dumps(content, default=str)
    stripped = text.strip()
    if not stripped:
        return True, KIND_EMPTY

    low = stripped.lower()

    # Structured JSON error bodies first (superset of the old curater check).
    if stripped[0] in "{[":
        try:
            data = json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            data = None
        if isinstance(data, dict):
            if data.get("error"):
                return True, KIND_TOOL_ERROR_JSON
            for key in ("returncode", "exit_code"):
                v = data.get(key)
                if isinstance(v, int) and not isinstance(v, bool) and v != 0:
                    return True, KIND_TOOL_ERROR_JSON

    # HIGH #5: shape-based detection. The old bare-substring checks turned the
    # classifier into a false-positive machine for a coding agent: read_file on
    # Python source mentioning "except Exception:" or grep output listing the
    # word "Traceback" counted as a failed tool result (penalizing
    # heuristic_score, feeding drop_tool_error_only, polluting DPO negatives).
    if _TRACEBACK_RE.search(stripped):
        return True, KIND_TRACEBACK
    if _EXCEPTION_CLASS_RE.search(stripped):
        return True, KIND_EXCEPTION
    if low.startswith("error"):
        return True, KIND_ERROR_TAG
    return False, KIND_NONE


# ------------------------------------------------------------------ reasons

_INFRA_REASONS = frozenset({"connection", "timeout", "model_not_found",
                            FAILURE_BACKEND_CIRCUIT_OPEN})
_MODEL_REASONS = frozenset({
    "context_overflow", "payload_too_large", "format_error",
    "max_iterations", "guardrail_halt",
})


def is_infra_reason(reason: str) -> bool:
    """True when a failure_reason denotes an infra fault, not a model fault.

    Accepts bare reasons ("connection") and the runner's composed form
    ("backend_error:connection"). Infra failures must never be used as DPO
    negative material and should auto-retry on batch resume (backlog #8).
    """
    if not reason:
        return False
    r = reason.lower()
    if r.startswith(FAILURE_BACKEND_ERROR_PREFIX):
        r = r[len(FAILURE_BACKEND_ERROR_PREFIX):]
    return r in _INFRA_REASONS


def is_model_reason(reason: str) -> bool:
    """True when a failure_reason denotes model behaviour worth training on."""
    if not reason:
        return False
    r = reason.lower()
    if r.startswith(FAILURE_BACKEND_ERROR_PREFIX):
        r = r[len(FAILURE_BACKEND_ERROR_PREFIX):]
    return r in _MODEL_REASONS


def is_cancelled_reason(reason: str) -> bool:
    """True when a failure_reason means a human pressed Stop (HIGH #4).

    Cancelled runs are score-NEUTRAL: neither the recorder (NULL
    outcome_score at write time) nor any mean-score gate / curation query
    may count them as -0.5 failures. Accepts the bare reason and the
    composed "backend_error:cancelled" form, mirroring the other gates.
    """
    if not reason:
        return False
    r = reason.lower()
    if r.startswith(FAILURE_BACKEND_ERROR_PREFIX):
        r = r[len(FAILURE_BACKEND_ERROR_PREFIX):]
    return r == FAILURE_CANCELLED

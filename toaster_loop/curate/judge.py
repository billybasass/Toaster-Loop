"""LLM-as-judge trace scoring (backlog item #12, opt-in).

Asks the *served model itself* to grade borderline traces against a small
rubric and stores the verdict as a feedback row ``source="judge"`` next to the
existing ``"heuristic"`` / ``"user"`` rows. Effective-score precedence where
the curater picks scores becomes: **user > judge > heuristic** (see
curater.load_trace).

BOUNDED CONTEXT (
MIT — bounded review context so a pathological trace can't blow up the
prompt): at most ``max_msgs`` recent user/assistant/tool messages, each capped
at ``per_msg_cap`` characters; tool results are trimmed harder (~200 chars in
the judge view — the judge needs to see that a tool ran and roughly what came
back, not 40 KB of file dump). The verdict re-enters the system as *data*
(a feedback row), never as instructions.

FORCED JSON: the rubric demands exactly ``{"score": 0.0-1.0, "reason": "..."}``.
Parsing is defensive — strict JSON, then a fenced ``` block, then the first
``{...}`` substring; anything unparseable returns None and counts as skipped,
never crashes curation.

BIAS CAVEAT (loud, per spec): the judged model grading its OWN traces is
biased toward its own style and self-assessment. Judge scores are a tiebreaker
for borderline selection near ``min_score`` — NOT ground truth. Never disable
human feedback in favor of judge scores.

Cost: every judge call is a real GPU inference. That is why this is purely
flag-driven (``toaster-loop curate --judge N``); no config key turns it on, and the
default path makes ZERO LLM calls.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger("toaster_loop.curate.judge")

#: Rubric lives in ONE constant so prompt/eval tests share the same source.
JUDGE_RUBRIC = (
    "You are grading one AI agent trace for training-data quality. Weigh:\n"
    "1. task completion: did the agent actually accomplish the user's request?\n"
    "2. tool-use correctness: were tools chosen and called sensibly, and were "
    "their results used truthfully?\n"
    "3. no hallucinated results: every claim grounded in observed output, "
    "nothing invented.\n"
    "Respond with ONLY a single JSON object and nothing else — no prose, no "
    'code fences:\n{"score": <number between 0.0 and 1.0>, "reason": '
    '"<one short sentence>"}'
)

# bounded message count + per-message cap.
DEFAULT_MAX_MSGS = 10
DEFAULT_PER_MSG_CAP = 800
#: tool results get a harder trim in the judge view (context, not forensics).
TOOL_VIEW_CAP = 200

_TRUNC = " […]"


def _clip(text: str, cap: int) -> str:
    text = text.strip()
    if len(text) <= cap:
        return text
    return text[:cap].rstrip() + _TRUNC


def _message_view(m: Dict[str, Any]) -> Optional[str]:
    """One transcript line for the judge view; None to skip the message."""
    role = m.get("role")
    if role == "tool":
        c = m.get("content")
        if not isinstance(c, str):
            c = json.dumps(c, ensure_ascii=False, default=str)
        return f"[tool:{m.get('name', 'tool')}] {_clip(c, TOOL_VIEW_CAP)}"
    if role in ("user", "assistant"):
        c = m.get("content")
        if not isinstance(c, str):
            c = json.dumps(c, ensure_ascii=False, default=str) if c else ""
        calls = m.get("tool_calls") or []
        parts = []
        if c.strip():
            parts.append(_clip(c, DEFAULT_PER_MSG_CAP))
        for tc in calls[:4]:  # bounded even when the cap math is bypassed
            fn = (tc or {}).get("function") or {}
            args = str(fn.get("arguments") or "")
            parts.append(f"<tool_call {fn.get('name', '?')} "
                         f"{_clip(args, 120)}>")
        if not parts:
            return None  # empty stub turn adds nothing for the judge
        return f"[{role}] " + "\n".join(parts)
    return None  # system prompt excluded from the excerpt (rubric has context)


def build_judge_messages(
    messages: Sequence[Dict[str, Any]],
    max_msgs: int = DEFAULT_MAX_MSGS,
    per_msg_cap: int = DEFAULT_PER_MSG_CAP,
) -> List[Dict[str, str]]:
    """Bounded review prompt: rubric system turn + recent-message excerpt.

    Keeps the LAST ``max_msgs`` judgeable messages (user/assistant/tool), each
    clipped to ``per_msg_cap`` chars (tool results to TOOL_VIEW_CAP). Global
    cap is therefore max_msgs * per_msg_cap — deterministic and small.
    """
    views: List[str] = []
    for m in reversed(list(messages or [])):
        if not isinstance(m, dict):
            continue
        line = _message_view(m)
        if line is None:
            continue
        if len(line) > per_msg_cap + len(f"[{m.get('role', 'tool')}] ") + 40:
            line = line[:per_msg_cap] + _TRUNC
        views.append(line)
        if len(views) >= max_msgs:
            break
    views.reverse()
    excerpt = "\n".join(views) if views else "(no readable messages)"
    return [
        {"role": "system", "content": JUDGE_RUBRIC},
        {"role": "user",
         "content": ("Grade this agent trace (oldest first):\n\n"
                     + excerpt + "\n\nRemember: JSON object only.")},
    ]


_JSON_BLOCK = re.compile(r"\{.*\}", re.S)
_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


def parse_judge_reply(text: Optional[str]) -> Optional[Dict[str, Any]]:
    """Defensively extract ``{"score": float, "reason": str}`` from a reply.

    Tries strict JSON -> fenced block -> first {...} substring. Returns None
    (skip) on anything unparseable or with an invalid score; never raises.
    """
    if not text:
        return None
    candidates: List[str] = [text.strip()]
    m = _FENCE.search(text)
    if m:
        candidates.append(m.group(1))
    m = _JSON_BLOCK.search(text)
    if m:
        candidates.append(m.group(0))
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(obj, dict):
            continue
        score = obj.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            continue
        reason = obj.get("reason")
        return {
            "score": max(0.0, min(1.0, float(score))),
            "reason": str(reason)[:500] if reason is not None else "",
        }
    return None


def judge_trace(
    trace: Any,
    client_fn: Callable[[List[Dict[str, str]], str], Optional[str]],
    model: str,
    max_msgs: int = DEFAULT_MAX_MSGS,
    per_msg_cap: int = DEFAULT_PER_MSG_CAP,
) -> Optional[Dict[str, Any]]:
    """Score one trace via the served model. Returns verdict dict or None.

    ``trace`` is a LoadedTrace or a raw OpenAI-format message list.
    ``client_fn(messages, model) -> reply_text`` performs the transport (see
    make_chat_client); any transport failure or unparseable reply yields None,
    which callers count as *skipped* — judging must never break curation.
    """
    messages = getattr(trace, "messages", trace)
    try:
        reply = client_fn(
            build_judge_messages(messages, max_msgs=max_msgs,
                                 per_msg_cap=per_msg_cap),
            model,
        )
    except Exception as e:  # transport blip => skip this trace, keep curating
        logger.warning("judge call failed (skipped): %s", e)
        return None
    if reply is None:
        return None
    verdict = parse_judge_reply(reply)
    if verdict is None:
        logger.warning("judge reply unparseable (skipped): %.120r", reply)
    return verdict


def select_borderline(
    traces: Sequence[Any],
    min_score: float,
    band: float,
    n: int,
) -> List[Any]:
    """Top-N borderline traces: |effective_score − min_score| < band.

    Closest-to-threshold first (the actual tiebreaker candidates); ties break
    on trace_id for determinism. NULL-scored traces are never borderline —
    missing scores must not be treated as zero (Phase 2 note #1).
    """
    cands = []
    for t in traces:
        eff = getattr(t, "effective_score", None)
        if eff is None:
            continue
        dist = abs(float(eff) - float(min_score))
        if dist < band:
            tid = getattr(getattr(t, "row", None), "trace_id", str(id(t)))
            cands.append((dist, tid, t))
    cands.sort(key=lambda x: (x[0], x[1]))
    return [t for _, _, t in cands[: max(0, int(n))]]


def make_chat_client(
    base_url: str,
    api_key: Optional[str] = None,
    timeout: float = 120.0,
    transport_attempts: int = 3,
    backoff_s: tuple = (1.0, 2.0),
    max_tokens: int = 300,
    extra_payload: Optional[Dict[str, Any]] = None,
) -> Callable[[List[Dict[str, str]], str], Optional[str]]:
    """HTTP client_fn posting one chat completion to the served model.

    Mirrors training/validate.py::_post_task's classified transport retry
    (transient connection/timeout/5xx retried; deterministic errors fail fast).
    Raises on final failure — judge_trace turns that into a skip.

    ``max_tokens`` sizes the completion (judge verdicts stay 300; pass-4 #39
    prompt-mutation calls need more room). ``extra_payload`` merges arbitrary
    request keys (e.g. thinking-off controls for mutation generation, where a
    reasoning block would pollute "output only the new prompt") — same
    enable_thinking / chat_format_template_controls dual shape as
    serving/run_args.request_sampling_payload.
    """
    import httpx
    import time

    from ..core.circuit import get_breaker
    from ..core.errors import classify_exception, classify_http_status

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    # Pass 4 #28: same per-backend breaker the agent runner uses (keyed by the
    # identical base_url string), so judge calls fail fast while a wedged
    # llama.cpp instance has its circuit open instead of adding load.
    breaker = get_breaker(base_url)

    def client(messages: List[Dict[str, str]], model: str) -> str:
        body = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,  # grading should be deterministic
        }
        if extra_payload:
            body.update(extra_payload)
        last_err = "no attempt made"
        for attempt in range(max(1, transport_attempts)):
            if not breaker.is_allowed():
                raise RuntimeError(
                    f"judge request refused: backend circuit open for {base_url}")
            if attempt:
                time.sleep(backoff_s[min(attempt - 1, len(backoff_s) - 1)])
            try:
                r = httpx.post(base_url.rstrip("/") + "/chat/completions",
                               json=body, timeout=timeout, headers=headers)
            except Exception as e:
                breaker.record_failure()
                ce = classify_exception(e)
                last_err = f"{type(e).__name__}: {e}"
                if ce.retryable and attempt + 1 < transport_attempts:
                    logger.warning("judge transport blip (%s), retrying",
                                   ce.reason)
                    continue
                raise
            if r.status_code == 200:
                breaker.record_success()
                data = r.json()
                msg = ((data.get("choices") or [{}])[0].get("message")) or {}
                return msg.get("content") or ""
            breaker.record_failure()
            ce = classify_http_status(r.status_code, r.text[:500])
            last_err = f"HTTP {r.status_code}"
            if ce.retryable and attempt + 1 < transport_attempts:
                logger.warning("judge HTTP %d (%s), retrying",
                               r.status_code, ce.reason)
                continue
            raise RuntimeError(f"judge request failed: {last_err}")
        raise RuntimeError(f"judge request failed: {last_err}")

    return client

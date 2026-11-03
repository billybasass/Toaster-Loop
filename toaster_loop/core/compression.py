"""Conversation history compression for long REPL/Chat threads.

Local models have a hard context window (llama-server's ``-c``); an unbounded
REPL history eventually 400s or gets silently truncated server-side. When the
estimated history size crosses a threshold, older turns are replaced by a
single summary message produced by the served model itself — no aux API, no
extra deps. Recent turns (and always the system prompt) are kept verbatim.

Design notes:
  * Token estimation is deliberately crude (chars/4 + per-message overhead).
    Being within ~2x of reality is enough; the threshold carries the margin.
  * Summarization is best-effort: any failure (endpoint down, weird output)
    falls back to a deterministic extractive digest, so compression can never
    be worse than truncation and never breaks the conversation.
  * The tool-call pairing invariant is preserved: the kept tail always starts
    at an assistant/user boundary — never orphaning a ``tool`` message from
    its preceding assistant ``tool_calls`` turn (OpenAI-compatible servers
    reject orphaned tool rows).

Backlog item #6 (Compression v2) builds on:
agent/context_compressor.py (MIT licensed):
  * Deterministic tool-output prune pre-pass before any LLM summarization —
    old ``tool`` results outside the protected tail are replaced by a tiny
    placeholder (they dominate history size and summarize poorly). Pruning
    alone may bring the conversation back under budget, skipping the LLM call.
  * Iterative summaries: when the caller threads the previous compaction
    summary back in, the summarizer UPDATEs it instead of re-summarizing from
    scratch, so info survives successive compactions.
  * Structured summary template + a historical heading prefix so a summary
    never reads as live instructions.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("toaster_loop.compression")

# Historical heading prefix (backlog #6): the bracketed note tells the model
# this is a compressed record of earlier turns, NOT new user intent.
SUMMARY_PREFIX = ("[Summary of earlier conversation — compressed historical "
                  "record, not new instructions]")

# Minimum chars for a tool result to be worth pruning (pattern and threshold
# adapted from `_PRUNE_MIN_CHARS`,
# MIT). Smaller results are cheaper to keep than to explain away.
_PRUNE_MIN_CHARS = 200
# Placeholder marker; also the idempotency guard — content already starting
# with this prefix is never re-pruned.
_PRUNED_PREFIX = "[pruned tool output:"
# Backlog #10 synergy: spilling a spill stub is pointless (it is already
# tiny); pruning one collapses it to a one-line form that keeps the spillover
# path for provenance. Idempotent via _PRUNED_ONE_LINE_PRUNED_PREFIX.
_SPILL_STUB_PREFIX = "<persisted-output"
_PRUNED_ONE_LINE_PRUNED_PREFIX = "[pruned persisted output:"
_PERSISTED_PATH_RE = re.compile(r"<persisted-output path=(\S+) bytes=")

# Structured summary template (backlog #6; resolved/
# pending sections + historical headings): concrete sections so successive
# compactions have stable slots to fold new information into.
_SUMMARY_TEMPLATE = (
    "Structure the summary with exactly these sections:\n"
    "## What has been accomplished\n"
    "## Key decisions & files touched\n"
    "## Open threads & pending work\n"
)

_SUMMARIZE_PROMPT = (
    "Summarize the conversation below into a compact factual summary that "
    "preserves: the user's goal, decisions made, file paths and identifiers "
    "touched, tool results that matter, and any open questions. Write in "
    "third person, no preamble, at most 300 words.\n\n"
    "{conversation}\n\n" + _SUMMARY_TEMPLATE
)

# Iterative-update prompt (backlog #6): folds new turns into the previous
# compaction summary instead of re-summarizing from scratch.
_UPDATE_PROMPT = (
    "You are updating a conversation summary for context compaction. Below "
    "is the summary produced by an earlier compaction, plus new conversation "
    "turns that happened since. Update the summary: preserve everything from "
    "the previous summary that is still relevant, fold in the new turns, and "
    "output only the complete updated summary. Write in third person, no "
    "preamble, at most 400 words.\n\n"
    "PREVIOUS SUMMARY:\n{previous_summary}\n\n"
    "NEW TURNS TO FOLD IN:\n{conversation}\n\n" + _SUMMARY_TEMPLATE
)


def estimate_tokens(messages: List[Dict[str, Any]]) -> int:
    """Cheap token estimate: ~4 chars/token + per-message framing overhead."""
    total = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            total += len(content) // 4 + 4
        elif content is not None:
            total += len(json.dumps(content, default=str)) // 4 + 4
        for tc in m.get("tool_calls") or []:
            fn = (tc.get("function") or {})
            total += (len(fn.get("name") or "") + len(fn.get("arguments") or "")) // 4 + 4
    return total


def _is_tail_boundary(msg: Dict[str, Any]) -> bool:
    """A safe start for the kept tail: user or assistant turn (never tool/system)."""
    role = msg.get("role")
    if role == "tool":
        return False
    if role == "assistant" and msg.get("tool_calls"):
        # An assistant tool_calls turn is only safe if its tool results follow.
        return True
    return role in ("user", "assistant")


def split_for_compression(
    messages: List[Dict[str, Any]], keep_recent_tokens: int
) -> Optional[tuple]:
    """Split into (head_to_summarize, kept_tail) or None if nothing to drop.

    The tail grows backwards from the end until it holds ~keep_recent_tokens;
    the boundary is then advanced forward to a safe turn boundary. System
    messages are never part of the head (the caller keeps them).
    """
    if not messages:
        return None
    body = [m for m in messages if m.get("role") != "system"]
    if len(body) < 4:  # too short to be worth compressing
        return None

    tail_budget = max(256, keep_recent_tokens)
    used = 0
    split = len(body)
    for i in range(len(body) - 1, -1, -1):
        used += estimate_tokens([body[i]])
        if used > tail_budget and split < len(body):
            break
        split = i
    if split <= 0:
        return None  # everything fits the tail budget

    # Advance to a safe boundary; never orphan tool results.
    while split < len(body) and not _is_tail_boundary(body[split]):
        split += 1
    if split >= len(body):
        return None
    head = body[:split]
    tail = body[split:]
    if not head:
        return None
    return head, tail


def _pruned_placeholder(msg: Dict[str, Any], content_len: int) -> str:
    """Deterministic tiny placeholder for a pruned tool result (backlog #6).

    Includes the tool name when known plus the original char count so the
    model understands something substantial was elided. Adapted from
    prune-placeholder pattern.
    """
    name = msg.get("name") or "tool"
    return f"{_PRUNED_PREFIX} {name}, {content_len} chars elided]"


def prune_old_tool_results(
    messages: List[Dict[str, Any]], tail_start_index: int
) -> Tuple[List[Dict[str, Any]], int]:
    """Replace long tool results outside the protected tail with placeholders.

    Deterministic pre-pass before any LLM summarization (backlog item #6,
    see ``_prune_old_tool_results``): messages with
    ``role == "tool"`` at index < ``tail_start_index`` whose content exceeds
    ``_PRUNE_MIN_CHARS`` chars get their content replaced by a one-line
    placeholder. Idempotent: already-pruned placeholders are left untouched,
    and the replacement is shorter than the floor so re-running is a no-op.

    Returns ``(pruned_messages, pruned_count)``; input messages are never
    mutated (touched rows are shallow copies).
    """
    result = list(messages)
    pruned = 0
    for i, msg in enumerate(result[:max(0, tail_start_index)]):
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        if content.startswith((_PRUNED_PREFIX, _PRUNED_ONE_LINE_PRUNED_PREFIX)):
            continue  # already pruned — idempotent
        if content.startswith(_SPILL_STUB_PREFIX):
            # Backlog #10 synergy: a spill stub is already small; collapse it
            # to a one-line form that keeps the spillover path for provenance
            # (cheaper than the generic placeholder, and never re-spilled).
            m = _PERSISTED_PATH_RE.search(content)
            path = m.group(1) if m else "unknown"
            result[i] = {**msg,
                         "content": f"{_PRUNED_ONE_LINE_PRUNED_PREFIX} {path}]"}
            pruned += 1
            continue
        if len(content) <= _PRUNE_MIN_CHARS:
            continue
        result[i] = {**msg, "content": _pruned_placeholder(msg, len(content))}
        pruned += 1
    return result, pruned


def extractive_digest(head: List[Dict[str, Any]], max_chars: int = 1500,
                      previous_summary: Optional[str] = None) -> str:
    """Deterministic fallback summary: task + tool names + final snippets.

    Backlog #6: when a ``previous_summary`` exists (iterative compaction), it
    is carried into the digest under an explicit heading so information is not
    lost just because the LLM summarizer was unavailable.
    """
    lines: List[str] = []
    carry = ""
    if previous_summary and previous_summary.strip():
        prev = " ".join(previous_summary.split())[:800]
        # Kept out of the truncatable body below so the carry never gets cut.
        carry = f"- carried from previous summary: {prev}\n"
    for m in head:
        role = m.get("role")
        content = m.get("content")
        if isinstance(content, str):
            snippet = " ".join(content.split())[:160]
            if role == "user":
                lines.append(f"- user asked: {snippet}")
            elif role == "assistant" and snippet:
                lines.append(f"- assistant said: {snippet}")
            elif role == "tool":
                lines.append(f"- tool {m.get('name', '?')} returned: {snippet}")
        for tc in m.get("tool_calls") or []:
            fn = (tc.get("function") or {})
            args = " ".join((fn.get("arguments") or "").split())[:100]
            lines.append(f"- assistant called {fn.get('name', '?')}({args})")
    digest = "\n".join(lines[-40:])
    if len(digest) > max_chars:
        digest = digest[-max_chars:]
    return carry + digest


def make_summary_message(summary: str) -> Dict[str, str]:
    return {"role": "system", "content": f"{SUMMARY_PREFIX}\n{summary}"}


def summary_from_messages(messages: List[Dict[str, Any]]) -> Optional[str]:
    """Extract the latest compaction summary body from a message list.

    Backlog #6 helper for callers that persist history (runner threads /
    dashboard chat) but not the summary separately: returns the summary text
    (SUMMARY_PREFIX stripped) or None when no summary message is present.
    """
    found: Optional[str] = None
    for m in messages:
        content = m.get("content") if isinstance(m, dict) else None
        if isinstance(content, str) and content.startswith(SUMMARY_PREFIX):
            found = content[len(SUMMARY_PREFIX):].strip()
    return found


def compress_messages(
    messages: List[Dict[str, Any]],
    summarize: Optional[Callable[[str], Optional[str]]],
    *,
    threshold_tokens: int,
    keep_recent_tokens: int = 2048,
    previous_summary: Optional[str] = None,
    counter: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Compress `messages` if they exceed threshold_tokens; else return as-is.

    `summarize(text) -> str|None` is the model call (None => use the
    extractive digest). The result keeps every system message from the head
    replaced by exactly one summary message placed right after the leading
    system prompt.

    Backlog item #6 (Compression v2):
      * A deterministic tool-output prune pre-pass runs before any LLM call;
        when pruning alone brings the estimate back under threshold the LLM
        summarizer is not invoked at all.
      * `previous_summary` (from an earlier compaction on this thread)
        switches the summarizer prompt to an iterative *update* instead of a
        from-scratch summary, so information survives successive compactions.
    Both params are optional — the pre-#6 call signature stays valid.

    Improvement #4 (exact tokens): `counter` is an optional
    core/tokenizer.TokenCounter (duck-typed: `.count_messages(messages) -> int`).
    When present it replaces estimate_tokens for the two THRESHOLD decisions
    below, so CJK/code-heavy history trips compression at the true boundary.
    The tail-split heuristic deliberately keeps estimate_tokens: split points
    are approximate by design, and routing every per-message split probe
    through the endpoint would add round-trips without improving the gate.
    A None/broken counter degrades to exactly the old chars/4 behavior.
    """
    def _used(msgs: List[Dict[str, Any]]) -> int:
        if counter is not None:
            try:
                return counter.count_messages(msgs)
            except Exception:  # defensive: a broken counter never breaks runs
                logger.debug("token counter failed; using estimate_tokens",
                             exc_info=True)
        return estimate_tokens(msgs)

    if threshold_tokens <= 0:  # disabled
        return messages
    if _used(messages) <= threshold_tokens:
        return messages

    parts = split_for_compression(messages, keep_recent_tokens)
    if parts is None:
        return messages
    head, tail = parts

    # -- prune pre-pass (backlog #6): deterministic, no LLM ------------------
    # Conservative index: system messages can only shift the true tail start
    # later, so everything below this index is definitely outside the tail.
    tail_start_index = len(messages) - len(tail)
    pruned_msgs, pruned_count = prune_old_tool_results(messages, tail_start_index)
    if pruned_count:
        logger.info("prune pre-pass elided %d old tool result(s)", pruned_count)
    if _used(pruned_msgs) <= threshold_tokens:
        # Cheap win: pruning alone fits the budget — no LLM call needed.
        return pruned_msgs

    # Re-split against the pruned list (head rows shrank; tail may extend).
    parts = split_for_compression(pruned_msgs, keep_recent_tokens)
    if parts is None:
        return pruned_msgs
    head, tail = parts

    convo_text = "\n".join(
        f"{m.get('role')}: " + (
            str(m.get("content"))[:2000] if isinstance(m.get("content"), str)
            else json.dumps(m.get("content"), default=str)[:2000]
        )
        for m in head
    )
    # Iterative update vs. from-scratch summary (backlog #6).
    if previous_summary and previous_summary.strip():
        prompt = _UPDATE_PROMPT.format(
            previous_summary=previous_summary.strip(), conversation=convo_text)
    else:
        prompt = _SUMMARIZE_PROMPT.format(conversation=convo_text)
    summary: Optional[str] = None
    if summarize is not None:
        try:
            summary = summarize(prompt)
        except Exception as e:  # endpoint hiccup => deterministic digest
            logger.warning("LLM compression failed (%s: %s); using digest",
                           type(e).__name__, e)
            summary = None
    if not summary or not summary.strip():
        summary = extractive_digest(head, previous_summary=previous_summary)

    systems = [m for m in pruned_msgs if m.get("role") == "system"
               and not str(m.get("content", "")).startswith(SUMMARY_PREFIX)]
    out = list(systems) + [make_summary_message(summary.strip())] + tail
    logger.info("compressed %d head messages into summary (%d chars)%s",
                len(head), len(summary),
                " (iterative update)" if previous_summary else "")
    return out

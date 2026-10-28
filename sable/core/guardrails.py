"""Tool-loop guardrails + repetition guard (backlog item #3).

Detection/decision shape built in
``agent/tool_guardrails.py`` and ``agent/repetition_guard.py`` — the compact
core only: signature-hashed call identity, the per-run threshold ladder, the
idempotent-vs-mutating tool split, and the *side-effect-free controller*
pattern (the controller never mutates messages; the runner decides what a
decision becomes). Threshold defaults are conservative.

Why this exists: small quantized GGUFs spiral — re-issuing the identical
failing ``patch`` call until ``max_iterations`` burns out, and truncated
outputs degenerate into verbatim echo loops. The controller catches both;
failure determination itself is delegated to the single source of truth in
``core/failure_types.classify_tool_result`` so guardrails, recorder and
curater can never disagree about what "failed" means.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from .failure_types import classify_tool_result

# ---------------------------------------------------------------------------
# Tool identity sets for the sable toolset (core/tools.py). Unknown tools
# are treated as MUTATING (conservative: they can never earn a "no progress"
# block, which is only safe for read-only tools).
# ---------------------------------------------------------------------------

IDEMPOTENT_TOOL_NAMES = frozenset(
    {"read_file", "list_dir", "search_files", "todo_read", "search_traces",
     # WP12 plan reads (pure store reads; safe for no-progress detection).
     "plan_next", "plan_view"}
)

MUTATING_TOOL_NAMES = frozenset({"write_file", "patch", "todo_write",
                                 # WP12 plan writes (graph mutations).
                                 "plan_progress", "plan_add"})


def _sha256(value: str) -> str:
    # surrogatepass: tool args can carry unpaired surrogates from model
    # output; the hash only needs deterministic bytes.
    return hashlib.sha256(value.encode("utf-8", "surrogatepass")).hexdigest()


def canonical_tool_args(args: Any) -> str:
    """Compact sorted-key JSON for tool args (arg key order becomes irrelevant)."""
    return json.dumps(
        args if isinstance(args, Mapping) else {},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _coerce_args(args: Any) -> Mapping[str, Any]:
    """Parse tool args that may arrive as a JSON string (raw function.arguments)."""
    if isinstance(args, Mapping):
        return args
    if isinstance(args, str) and args.strip():
        try:
            parsed = json.loads(args)
        except (json.JSONDecodeError, ValueError):
            return {}
        if isinstance(parsed, Mapping):
            return parsed
    return {}


@dataclass(frozen=True)
class ToolCallSignature:
    """Stable identity for (tool_name, canonical args). Key order irrelevant."""

    tool_name: str
    args_hash: str

    @classmethod
    def from_call(cls, tool_name: str, args: Any) -> "ToolCallSignature":
        return cls(tool_name=tool_name,
                   args_hash=_sha256(canonical_tool_args(_coerce_args(args))))


@dataclass(frozen=True)
class Decision:
    """Controller verdict: allow | warn | block | halt (+ reason metadata)."""

    action: str = "allow"
    code: str = "allow"
    message: str = ""
    tool_name: str = ""
    count: int = 0
    signature: Optional[ToolCallSignature] = None

    @property
    def allows_execution(self) -> bool:
        return self.action in ("allow", "warn")


class ToolLoopGuardrails:
    """Per-run controller for repeated failing / non-progressing tool calls.

    Side-effect free: ``before_call``/``after_call`` only observe and return
    :class:`Decision`s. The runner turns warn into a nudge appended to the
    tool result, block into a synthetic error tool result (execution skipped,
    loop continues), and halt into a controlled stop with
    ``failure_reason="guardrail_halt:<code>"``.

    Threshold ladder: identical failing call repeated
    ``exact_failure_warn_after`` times warns and ``exact_failure_block_after``
    times blocks; any failure of the same tool warns at
    ``same_tool_failure_warn_after`` and halts at ``same_tool_failure_halt_after``;
    an idempotent tool returning an identical result repeatedly warns at
    ``no_progress_warn_after`` and blocks at ``no_progress_block_after``.

    ``enabled=False`` (config ``agent.guardrails: false``) makes every call
    allow unconditionally.
    """

    def __init__(
        self,
        enabled: bool = True,
        exact_failure_warn_after: int = 2,
        exact_failure_block_after: int = 5,
        same_tool_failure_warn_after: int = 3,
        same_tool_failure_halt_after: int = 8,
        no_progress_warn_after: int = 2,
        no_progress_block_after: int = 5,
    ) -> None:
        self.enabled = bool(enabled)
        self.exact_failure_warn_after = exact_failure_warn_after
        self.exact_failure_block_after = exact_failure_block_after
        self.same_tool_failure_warn_after = same_tool_failure_warn_after
        self.same_tool_failure_halt_after = same_tool_failure_halt_after
        self.no_progress_warn_after = no_progress_warn_after
        self.no_progress_block_after = no_progress_block_after
        # per-run state (one instance per run() call)
        self._exact_failure_counts: dict[ToolCallSignature, int] = {}
        self._same_tool_failure_counts: dict[str, int] = {}
        # signature -> (result_hash, consecutive identical-result count)
        self._no_progress: dict[ToolCallSignature, tuple[str, int]] = {}

    # -- decision points -----------------------------------------------------

    def before_call(self, tool_name: str, args: Any) -> Decision:
        """Gate a pending call: block repeated exact failures / no-progress reads."""
        if not self.enabled:
            return Decision(tool_name=tool_name)
        sig = ToolCallSignature.from_call(tool_name, args)

        exact_count = self._exact_failure_counts.get(sig, 0)
        if exact_count >= self.exact_failure_block_after:
            return Decision(
                action="block", code="repeated_exact_failure_block",
                message=(
                    f"Blocked {tool_name}: the same tool call failed {exact_count} "
                    "times with identical arguments. Stop retrying it unchanged; "
                    "change strategy or explain the blocker."
                ),
                tool_name=tool_name, count=exact_count, signature=sig,
            )

        if self._is_idempotent(tool_name):
            record = self._no_progress.get(sig)
            if record is not None and record[1] >= self.no_progress_block_after:
                return Decision(
                    action="block", code="idempotent_no_progress_block",
                    message=(
                        f"Blocked {tool_name}: this read-only call returned the same "
                        f"result {record[1]} times. Stop repeating it unchanged; use "
                        "the result already provided or try a different query."
                    ),
                    tool_name=tool_name, count=record[1], signature=sig,
                )

        return Decision(tool_name=tool_name, signature=sig)

    def after_call(self, tool_name: str, args: Any, result_content: str) -> Decision:
        """Observe one completed call's result and update counters."""
        if not self.enabled:
            return Decision(tool_name=tool_name)
        sig = ToolCallSignature.from_call(tool_name, args)
        failed, _kind = classify_tool_result(tool_name, result_content or "")

        if failed:
            exact_count = self._exact_failure_counts.get(sig, 0) + 1
            self._exact_failure_counts[sig] = exact_count
            self._no_progress.pop(sig, None)
            same_count = self._same_tool_failure_counts.get(tool_name, 0) + 1
            self._same_tool_failure_counts[tool_name] = same_count

            if same_count >= self.same_tool_failure_halt_after:
                return Decision(
                    action="halt", code="same_tool_failure_halt",
                    message=(
                        f"Stopped {tool_name}: it failed {same_count} times this run. "
                        "Stop retrying the same failing tool path and choose a "
                        "different approach."
                    ),
                    tool_name=tool_name, count=same_count, signature=sig,
                )
            if exact_count >= self.exact_failure_warn_after:
                return Decision(
                    action="warn", code="repeated_exact_failure_warning",
                    message=(
                        f"{tool_name} has failed {exact_count} times with identical "
                        "arguments. This looks like a loop; inspect the error and "
                        "change strategy instead of retrying it unchanged."
                    ),
                    tool_name=tool_name, count=exact_count, signature=sig,
                )
            if same_count >= self.same_tool_failure_warn_after:
                return Decision(
                    action="warn", code="same_tool_failure_warning",
                    message=(
                        f"{tool_name} has failed {same_count} times this run. This "
                        "looks like a loop. Diagnose the latest error, then try "
                        "different arguments or a different tool instead of "
                        "retrying the same failing path."
                    ),
                    tool_name=tool_name, count=same_count, signature=sig,
                )
            return Decision(tool_name=tool_name, count=exact_count, signature=sig)

        # Success clears failure streaks (established semantics).
        self._exact_failure_counts.pop(sig, None)
        self._same_tool_failure_counts.pop(tool_name, None)

        if not self._is_idempotent(tool_name):
            # Mutating tools never accumulate no-progress: repeating a
            # "successful" write/patch is not provably pointless.
            self._no_progress.pop(sig, None)
            return Decision(tool_name=tool_name, signature=sig)

        result_hash = _sha256(result_content or "")
        previous = self._no_progress.get(sig)
        repeat_count = 1
        if previous is not None and previous[0] == result_hash:
            repeat_count = previous[1] + 1
        self._no_progress[sig] = (result_hash, repeat_count)

        if repeat_count >= self.no_progress_warn_after:
            return Decision(
                action="warn", code="idempotent_no_progress_warning",
                message=(
                    f"{tool_name} returned the same result {repeat_count} times. "
                    "Use the result already provided or change the query instead "
                    "of repeating it unchanged."
                ),
                tool_name=tool_name, count=repeat_count, signature=sig,
            )
        return Decision(tool_name=tool_name, count=repeat_count, signature=sig)

    # -- helpers ---------------------------------------------------------------

    def _is_idempotent(self, tool_name: str) -> bool:
        # Unknown tools are conservatively mutating => never no-progress blocked.
        return tool_name in IDEMPOTENT_TOOL_NAMES


def synthetic_blocked_result(decision: Decision) -> str:
    """role=tool content for a blocked call (starts with "error:" so the
    recorder heuristic counts it, and classify_tool_result agrees)."""
    return f"error: blocked by guardrail ({decision.code}): {decision.message}"


def append_guardrail_guidance(result_content: str, decision: Decision) -> str:
    """Append warn/halt guidance to the current tool result content."""
    if decision.action not in ("warn", "halt") or not decision.message:
        return result_content
    label = "Tool loop hard stop" if decision.action == "halt" else "Tool loop warning"
    return (result_content or "") + (
        f"\n\n[{label}: {decision.code}; count={decision.count}; {decision.message}]"
    )


# ---------------------------------------------------------------------------
# Repetition guard (streak detector)
# ---------------------------------------------------------------------------

#: A fragment must be at least this long before the check runs at all.
MIN_FRAGMENT_LENGTH = 400
#: Exact-repeat window length; a verbatim repeat of this many chars is far
#: beyond ordinary phrasing reuse.
REPEAT_WINDOW = 60
#: Minimum repeats for a loop signal, even for short fragments.
MIN_REPEAT_COUNT = 5
#: Repeated windows must cover at least this fraction of the fragment.
DOMINANCE_RATIO = 0.5

#: Marker left where a repetition loop was cut.
REPETITION_MARKER = "…[repetition truncated]"


def find_loop(text: str) -> Optional[int]:
    """Index where a repetition loop starts, or None when ``text`` is clean.

    A fragment is loop-dominated when one ``REPEAT_WINDOW``-char substring
    appears often enough that its occurrences cover at least DOMINANCE_RATIO
    of the fragment (and >= MIN_REPEAT_COUNT times). Returns the first
    occurrence index of the dominant window so callers can truncate *at* the
    loop start instead of dropping the whole response. Fail-open on short or
    non-string input — never flags text the guard cannot confidently judge.
    """
    if not isinstance(text, str):
        return None
    n = len(text)
    if n < MIN_FRAGMENT_LENGTH:
        return None

    # Fast path: one normalized line duplicated often enough (the most common
    # echo shape — a repeated sentence/paragraph on its own line).
    counts_by_line: dict[str, int] = {}
    for line in text.splitlines():
        norm = line.strip()
        if norm:
            counts_by_line[norm] = counts_by_line.get(norm, 0) + 1
    for line, c in counts_by_line.items():
        if c >= MIN_REPEAT_COUNT and c * len(line) >= n * DOMINANCE_RATIO:
            return text.find(line)

    # General path: fixed windows sliding one char at a time; catches loops
    # that do not align to line boundaries.
    needed = max(MIN_REPEAT_COUNT, math.ceil(n * DOMINANCE_RATIO / REPEAT_WINDOW))
    counts: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    for i in range(n - REPEAT_WINDOW + 1):
        key = text[i:i + REPEAT_WINDOW]
        if key not in first_seen:
            first_seen[key] = i
        c = counts.get(key, 0) + 1
        if c >= needed:
            return first_seen[key]
        counts[key] = c
    return None


def detect_repetition(text: str) -> bool:
    """True when ``text`` is dominated by verbatim repeated fragments."""
    return find_loop(text) is not None


def truncate_repetition(text: str) -> str:
    """Cut a repetition loop out of ``text``, keeping everything before it.

    Returns the text unchanged when no loop is detected; otherwise keeps the
    prefix up to the loop start and appends ``REPETITION_MARKER`` so the trace
    shows truncation happened (lossless-ish safety, never drops content that
    preceded the loop).
    """
    if not isinstance(text, str) or not text:
        return text
    idx = find_loop(text)
    if idx is None:
        return text
    kept = text[:idx].rstrip()
    if not kept:  # loop starts at the very beginning: keep one window for context
        kept = text[:REPEAT_WINDOW]
    return f"{kept}\n{REPETITION_MARKER}"

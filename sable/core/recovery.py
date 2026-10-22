"""Reasoning-loop detection + truncation (WP8).

A distinct failure class from the verbatim echo loops caught by
``core.guardrails.truncate_repetition``: the model CYCLES on a thought --
re-emitting (near-)identical paragraphs, or spiraling through
"Wait, I need to..." hedging cycles -- without ever converging to an answer
or a tool call. Verbatim-window repetition detection does not fire on these
because each lap is rephrased slightly or separated by interleaved
``think``-tag noise.

Design (deliberately conservative -- a false positive truncates a GOOD
final answer, which is worse than a missed loop):

  * the one think-scrubber (core/think_scrub.strip_think_blocks) runs
    FIRST, so loops hidden inside/around leaked reasoning blocks are seen
    as the same text the user would get;
  * three cheap heuristics, each requiring BOTH a repetition count AND a
  dominance share of the whole text (a healthy 20-item list repeating a
  bullet stem, or code repeating a pattern, stays far below the shares);
  * paragraph granularity (blank-line separated): single-newline structures
  -- lists, code blocks, tables -- are ONE unit and are never compared
  lap-by-lap;
  * near-duplicate comparison uses difflib with a quick_ratio prefilter
  and greedy representative clustering (one ratio call per existing
  cluster per unit, hard budget): linear-ish in text length, no
  quadratic blowup, no recursion.

The consumer policy lives in agent_runner: loop detected AND no tool call
parsed -> truncate at the first lap (keep everything before it) and reuse
the verification-nudge mechanism (synthetic user turn, at most once per
run) to retry -- no second retry path is invented here.

Inspiration: kimi-k3's drift-corpus discipline around reasoning-loop
recovery, re-implemented inline per D-12 (no imports).
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .think_scrub import strip_think_blocks

__all__ = ["LoopVerdict", "detect_reasoning_loop", "truncate_at_loop",
           "LOOP_MARKER"]

#: Marker left where a reasoning loop was cut (mirrors the repetition
#: marker's shape so traces show which guard fired).
LOOP_MARKER = "\u2026[reasoning loop truncated]"

#: Detection is off below this size -- short answers are never loops.
MIN_TEXT_CHARS = 320

#: Paragraph-level exact-repeat heuristics.
MIN_PARAGRAPH_CHARS = 60      # shorter units never count as "a lap"
MIN_LAPS = 3                  # same paragraph >= this many times
LAP_DOMINANCE = 0.35          # laps must cover >= this share of the text

#: Near-duplicate lap heuristics (rephrased cycling).
NEAR_MIN_CHARS = 120          # only long prose blocks may match fuzzily
NEAR_RATIO = 0.84
NEAR_MIN_LAPS = 3
MAX_NEAR_COMPARISONS = 400    # hard budget for the difflib ratio calls

#: Hedging-cycle heuristics ("Wait, actually... no, hold on...").
HEDGE_RE = re.compile(
    r"^(?:wait\b|but\s+wait\b|hmm+\b|hold\s+on\b|actually\b|no[,.\s]|"
    r"no\s+wait\b|i\s+(?:should|need|have\s+to)\b|let\s+me\s+(?:think|"
    r"reconsider|re-?think)\b|on\s+(?:second|further)\s+thought\b|"
    r"reconsidering\b)",
    re.I,
)
MIN_HEDGES = 4
HEDGE_DOMINANCE = 0.25

#: Cap on how many paragraph comparisons the near-dupe pass performs.
_MAX_PARAGRAPHS_SCANNED = 4000

#: List-item guard (false-positive): bullet/number prefixes mark STRUCTURED
#: content -- inventories, changelogs, code-adjacent lists routinely share a
#: long verbatim stem and differ only in an index ("item 0/item 1", "- a/- b"
#: stems at ratio() > 0.95). A reasoning loop cycles on THOUGHTS (prose
#: paragraphs); a list is never a lap, so bullet/numbered units are excluded
#: from both repeat passes entirely.
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*\u2022\u00b7+]>?|\d+[.)])\s")


@dataclass(frozen=True)
class LoopVerdict:
    """Structured loop verdict. ``loop=False`` verdicts carry no payload."""

    loop: bool = False
    repeated_unit: str = ""     # RAW text of the first lap (for truncation)
    repeats: int = 0
    start: Optional[int] = None  # offset of first lap, None if unknown
    reason: str = ""            # paragraph_repeats | near_duplicates |
                                # hedging_cycles | ""


_NO = LoopVerdict()

_P_BREAK = re.compile(r"\n[ \t]*\n")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower().rstrip(".!:;")


def _paragraphs(text: str) -> List[Tuple[int, str]]:
    """(offset, raw) for blank-line-separated blocks."""
    out: List[Tuple[int, str]] = []
    pos = 0
    for m in _P_BREAK.finditer(text):
        chunk = text[pos:m.start()]
        if chunk.strip():
            out.append((pos + (len(chunk) - len(chunk.lstrip("\n"))),
                        chunk.strip("\n")))
        pos = m.end()
    if text[pos:].strip():
        out.append((pos + (len(text) - pos - len(text[pos:].lstrip("\n"))),
                    text[pos:].strip("\n")))
    return out


def _near(a: str, b: str) -> bool:
    # autojunk=False: the default heuristic drops "popular" elements for
    # sequences >= 200 items -- on prose paragraphs that guts ratio()
    # (0.98 quick vs 0.47 real on a typo-level difference). One matcher
    # per ratio call: quick_ratio() consumes the matcher's internal
    # matching state, so reusing the object for ratio() is silently wrong.
    if difflib.SequenceMatcher(None, a, b,
                               autojunk=False).quick_ratio() < NEAR_RATIO:
        return False
    return difflib.SequenceMatcher(None, a, b,
                                   autojunk=False).ratio() >= NEAR_RATIO


def detect_reasoning_loop(text: str) -> LoopVerdict:
    """Conservative reasoning-loop detection over assistant text.

    Returns a :class:`LoopVerdict`; ``loop=True`` only when one of the
    three heuristics passes BOTH its repeat-count and dominance threshold.
    Never raises; anything odd or short fails open to ``loop=False``.
    """
    try:
        if not isinstance(text, str) or len(text) < MIN_TEXT_CHARS:
            return _NO
        scrubbed = strip_think_blocks(text) if "<" in text else text
        n = len(scrubbed)
        if n < MIN_TEXT_CHARS:
            return _NO

        paras = _paragraphs(scrubbed)[:_MAX_PARAGRAPHS_SCANNED]

        # -- 1. exact normalized paragraph repeats -------------------------
        counts: dict = {}
        first_idx: dict = {}
        for idx, (off, chunk) in enumerate(paras):
            if len(chunk) < MIN_PARAGRAPH_CHARS:
                continue
            if _LIST_ITEM_RE.match(chunk):
                continue  # structured list item: never a lap
            key = _norm(chunk)
            if not key:
                continue
            counts[key] = counts.get(key, 0) + 1
            first_idx.setdefault(key, (idx, off, chunk))
            if (counts[key] >= MIN_LAPS
                    and counts[key] * len(chunk) >= LAP_DOMINANCE * n):
                return LoopVerdict(True, chunk, counts[key], off,
                                   "paragraph_repeats")

        # -- 2. near-duplicate laps (rephrased cycling) --------------------
        # Only long prose units. Greedy clustering: each unit joins the
        # cluster whose REPRESENTATIVE it is near-duplicate of (the rep is
        # the first lap of that cycle), comparing at most one ratio per
        # cluster -> O(clusters) per unit, hard-capped overall. A cluster
        # needs NEAR_MIN_LAPS members AND dominance to flag.
        big = [(off, chunk) for off, chunk in paras
               if len(chunk) >= NEAR_MIN_CHARS
               and not _LIST_ITEM_RE.match(chunk)]
        # clusters: [norm_rep, raw_rep, first_off, count]
        clusters: List[List] = []
        budget = MAX_NEAR_COMPARISONS
        for off, chunk in big:
            head = _norm(chunk)
            joined = False
            for cl in clusters:
                if budget <= 0:
                    break
                budget -= 1
                if _near(head, cl[0]):
                    cl[3] += 1  # first_off (cl[2]) stays at lap 1
                    joined = True
                    if (cl[3] >= NEAR_MIN_LAPS
                            and cl[3] * len(cl[1]) >= LAP_DOMINANCE * n):
                        return LoopVerdict(True, cl[1], cl[3], cl[2],
                                           "near_duplicates")
                    break
            if not joined and budget > 0:
                clusters.append([head, chunk, off, 1])
            if budget <= 0:
                break

        # -- 3. hedging cycles ---------------------------------------------
        hedges: List[Tuple[int, str]] = []
        pos = 0
        for line in scrubbed.splitlines(keepends=True):
            if HEDGE_RE.match(line.strip()):
                hedges.append((pos, line.strip()))
            pos += len(line)
        hedge_chars = sum(len(h) for _, h in hedges)
        if (len(hedges) >= MIN_HEDGES
                and hedge_chars >= HEDGE_DOMINANCE * n):
            return LoopVerdict(True, hedges[0][1], len(hedges),
                               hedges[0][0], "hedging_cycles")

        return _NO
    except Exception:  # noqa: BLE001 - detection must never break a turn
        return _NO


def truncate_at_loop(text: str,
                     verdict: Optional[LoopVerdict] = None) -> str:
    """Cut *text* at the FIRST occurrence of the loop's repeated unit.

    Keeps everything before the first lap and appends ``LOOP_MARKER``; the
    first lap itself is kept when the loop starts at the very beginning
    (context is better than nothing). If the unit cannot be located (e.g.
    the caller passes different text than was analysed), the text is
    returned unchanged -- fail open.
    """
    if verdict is None:
        verdict = detect_reasoning_loop(text)
    if not verdict.loop or not isinstance(text, str) or not text:
        return text
    idx = text.find(verdict.repeated_unit) if verdict.repeated_unit else -1
    if idx < 0:
        idx = verdict.start if verdict.start is not None else -1
        if idx < 0 or idx > len(text):
            return text
    kept = text[:idx].rstrip()
    if not kept:
        # loop at the very start: keep one lap for context
        kept = text[:idx + len(verdict.repeated_unit)].rstrip()
    return f"{kept}\n{LOOP_MARKER}"

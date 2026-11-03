"""Scrubber for leaked reasoning/thinking blocks in assistant text.

Chat-template-mismatched GGUFs sometimes emit their chain-of-thought *inline*
in plain ``content`` — either as closed ``<think>...</think>`` pairs or as an
unterminated block that runs to the end of the reply. Left alone, those leak
into the live answer, the trace JSONL and (worst case) curated SFT rows as
"gpt" content.

This module is the single binding point:

  * ``THINK_TAG_NAMES`` — the one central tag-name list; every surface that
    hides reasoning uses it, so a new tag added here is covered everywhere.
  * ``StreamingThinkScrubber`` — stateful per-delta scrubber for the SSE path.
    It holds partial tags at delta boundaries until resolved (a naive per-
    delta regex erases an opening ``<think>`` that arrives alone and then
    streams the reasoning through), ``flush()`` releases held-back prose that
    turned out not to be a tag, ``reset()`` runs at the top of every turn.
  * ``strip_think_blocks`` — whole-string convenience for finalizing a
    complete message (non-streaming backends, compaction summaries).

Semantics (deliberate):

  * Closed ``<tag>X</tag>`` pairs are always removed, even inline mid-sentence
    — an inline pair is almost certainly leaked reasoning, not prose.
  * An unterminated open tag only starts a suppressed block at a *block
    boundary* (stream start, or preceding text on the current line is
    whitespace-only), so documentation/prose that merely *mentions*
    ``<thinking>`` mid-line is never suppressed.
  * Orphan close tags (no matching open) are always dropped — pure noise.
  * At ``flush()`` inside an unterminated block the held tail is discarded:
    leaking partial reasoning is worse than a truncated answer.

Built-in think-tag scrubber.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

__all__ = [
    "THINK_TAG_NAMES",
    "THINK_OPEN_TAGS",
    "THINK_CLOSE_TAGS",
    "StreamingThinkScrubber",
    "strip_think_blocks",
]

# The one list of model reasoning tag names. Consumers match case-insensitively,
# so the literal tags are lowercase (the uppercase name below lowercases to a
# distinct tag used by at least one family of templates).
THINK_TAG_NAMES: Tuple[str, ...] = (
    "think",
    "thinking",
    "reasoning",
    "thought",
    "REASONING_SCRATCHPAD",
)
THINK_OPEN_TAGS: Tuple[str, ...] = tuple(f"<{n.lower()}>" for n in THINK_TAG_NAMES)
THINK_CLOSE_TAGS: Tuple[str, ...] = tuple(f"</{n.lower()}>" for n in THINK_TAG_NAMES)


def _first_tag(buf_lower: str, tags: Tuple[str, ...]) -> Tuple[int, int]:
    """(earliest_index, tag_length) over *tags* in a lowercased buffer, or (-1, 0)."""
    hits = [(idx, len(tag)) for tag in tags if (idx := buf_lower.find(tag)) != -1]
    return min(hits) if hits else (-1, 0)


class StreamingThinkScrubber:
    """Stateful per-turn scrubber for streamed assistant text.

    State: ``_in_block`` (inside an open block; deltas discarded), ``_buf``
    (held-back partial-tag tail at the delta boundary), ``_prior_newline``
    (True iff the last emission ended with ``\\n`` or nothing has been emitted
    yet — decides whether an open tag at buffer position 0 is at a block
    boundary).
    """

    _ALL_TAGS: Tuple[str, ...] = THINK_OPEN_TAGS + THINK_CLOSE_TAGS
    _MAX_TAG_LEN: int = max(len(t) for t in _ALL_TAGS)
    # Orphan close tags plus trailing horizontal whitespace/newlines.
    _ORPHAN_CLOSE_RE = re.compile(
        "(?:" + "|".join(re.escape(t) for t in THINK_CLOSE_TAGS) + r")[ \t\n\r]*",
        re.IGNORECASE,
    )

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset all state. Call at the top of every new turn."""
        self._in_block: bool = False
        self._buf: str = ""
        self._prior_newline: bool = True

    # -- public ------------------------------------------------------------

    def feed(self, text: str) -> str:
        """Feed one delta; return its scrubbed visible portion.

        Returns "" when the delta is entirely reasoning or is being held back
        pending resolution of a partial tag at the boundary.
        """
        if not text:
            return ""
        buf = self._buf + text
        self._buf = ""
        out: List[str] = []

        while buf:
            if self._in_block:
                idx, tlen = _first_tag(buf.lower(), THINK_CLOSE_TAGS)
                if idx < 0:
                    # Still inside: drop everything except a possible partial
                    # close-tag prefix at the tail.
                    self._hold_partial(buf, THINK_CLOSE_TAGS)
                    break
                self._in_block = False
                buf = buf[idx + tlen:]
                continue

            # Priority 1: earliest closed pair (always noise). Priority 2:
            # unterminated open tag at a block boundary (gated so prose that
            # merely mentions '<thinking>' survives). Earliest position wins.
            pair = self._earliest_closed_pair(buf)
            open_idx, open_len = self._open_at_boundary(buf, out)
            if pair is not None and (open_idx < 0 or pair[0] <= open_idx):
                self._emit(out, buf[:pair[0]])
                buf = buf[pair[1]:]
                continue
            if open_idx >= 0:
                self._emit(out, buf[:open_idx])
                self._in_block = True
                buf = buf[open_idx + open_len:]
                continue

            # Priority 3: a full close tag while not in a block is orphan
            # noise (the state machine has no pending open it could match).
            # Dropping it here — instead of letting hold_partial keep it alive
            # as a "partial prefix" — stops a split orphan from re-merging
            # with later prose un-stripped.
            close_idx, close_len = _first_tag(buf.lower(), THINK_CLOSE_TAGS)
            if close_idx >= 0:
                self._emit(out, buf[:close_idx])
                buf = buf[close_idx + close_len:]
                continue

            # Priority 4: a complete open tag NOT at a block boundary is held
            # (not emitted) until the pair resolves: the opening tag and its
            # close routinely arrive in different deltas, and emitting it here
            # would either print the tag or — worse — stream the reasoning
            # that follows once the close shows up. If no close ever arrives,
            # flush() releases the held text verbatim (it was prose after all).
            any_open_idx, _ = _first_tag(buf.lower(), THINK_OPEN_TAGS)
            if any_open_idx >= 0:
                self._emit(out, buf[:any_open_idx])
                self._buf = buf[any_open_idx:]
                break

            # Nothing resolvable: hold a partial-tag tail so a tag split
            # across deltas is not missed; emit the rest.
            self._emit(out, self._hold_partial(buf, self._ALL_TAGS))
            break

        return "".join(out)

    def flush(self) -> str:
        """End-of-stream flush.

        Inside an unterminated block the held-back tail is discarded (leaking
        partial reasoning is worse than a truncated answer); otherwise it is
        emitted verbatim. Always clears the boundary flag — an intra-turn retry
        flushes and streams again without ``reset()``, and a stale False flag
        would make the new stream's opening tag look mid-line.
        """
        tail = "" if self._in_block else self._buf
        self._buf = ""
        self._in_block = False
        self._prior_newline = True
        return self._strip_orphan_close_tags(tail) if tail else ""

    # -- internals -----------------------------------------------------------

    def _emit(self, out: List[str], text: str) -> None:
        """Append visible prose (orphan close tags stripped), track newline flag."""
        text = self._strip_orphan_close_tags(text)
        if text:
            out.append(text)
            self._prior_newline = text.endswith("\n")

    def _hold_partial(self, buf: str, tags: Tuple[str, ...]) -> str:
        """Move a trailing partial-tag prefix of *buf* into ``_buf``; return the rest."""
        held = self._max_partial_suffix(buf, tags)
        self._buf = buf[-held:] if held else ""
        return buf[:-held] if held else buf

    @staticmethod
    def _earliest_closed_pair(buf: str) -> Optional[Tuple[int, int]]:
        """(start, end) of the earliest ``<tag>...</tag>`` pair (case-insensitive)."""
        buf_lower = buf.lower()
        pairs = []
        for open_tag, close_tag in zip(THINK_OPEN_TAGS, THINK_CLOSE_TAGS,
                                       strict=True):
            open_idx = buf_lower.find(open_tag)
            if open_idx == -1:
                continue
            close_idx = buf_lower.find(close_tag, open_idx + len(open_tag))
            if close_idx != -1:
                pairs.append((open_idx, close_idx + len(close_tag)))
        return min(pairs) if pairs else None

    def _open_at_boundary(self, buf: str, emitted: List[str]) -> Tuple[int, int]:
        """Earliest open tag that sits at a block boundary -> (idx, len), else (-1, 0)."""
        buf_lower = buf.lower()
        hits: List[Tuple[int, int]] = []
        for tag in THINK_OPEN_TAGS:
            idx = buf_lower.find(tag)
            while idx != -1 and not self._is_block_boundary(buf, idx, emitted):
                idx = buf_lower.find(tag, idx + 1)
            if idx != -1:
                hits.append((idx, len(tag)))
        return min(hits) if hits else (-1, 0)

    def _is_block_boundary(self, buf: str, idx: int, emitted: List[str]) -> bool:
        """True iff an open tag at *idx* begins a block: position 0 after a
        newline-terminated (or no) prior emission, or any position whose
        preceding text on the current line is whitespace-only."""
        prior_newline = (emitted[-1].endswith("\n")
                         if emitted else self._prior_newline)
        if idx == 0:
            return prior_newline
        preceding = buf[:idx]
        last_nl = preceding.rfind("\n")
        # No newline in this buffer yet: the previous emission must also have
        # ended with one for this to still be a fresh line.
        return (prior_newline if last_nl == -1 else True) \
            and preceding[last_nl + 1:].strip() == ""

    @classmethod
    def _max_partial_suffix(cls, buf: str, tags: Tuple[str, ...]) -> int:
        """Longest buf-suffix that is a strict prefix of any tag.

        (Full matches are real tags and are handled by the main loop.)"""
        buf_lower = buf.lower()
        for i in range(min(len(buf_lower), cls._MAX_TAG_LEN - 1), 0, -1):
            suffix = buf_lower[-i:]
            if any(len(tag) > i and tag.startswith(suffix) for tag in tags):
                return i
        return 0

    @classmethod
    def _strip_orphan_close_tags(cls, text: str) -> str:
        """Remove close tags with no matching open (always noise), plus their
        trailing whitespace."""
        return cls._ORPHAN_CLOSE_RE.sub("", text) if "</" in text else text


def strip_think_blocks(text: str) -> str:
    """Whole-string scrub for a complete message.

    Runs the text through a fresh scrubber (feed + flush), giving identical
    semantics to the streaming path: closed pairs removed, unterminated block
    at a block boundary suppressed, orphan close tags dropped. Idempotent —
    already-scrubbed text passes through unchanged.
    """
    if not text or ("<" not in text):
        return text
    s = StreamingThinkScrubber()
    out = s.feed(text)
    tail = s.flush()
    return out + tail

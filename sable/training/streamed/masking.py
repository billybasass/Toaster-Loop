"""Per-turn loss masking for ShareGPT conversations (WP10 step 2).

STRICTER than airllm's messages loader (D-12: ``airllm``'s ``lora_data.py``
was READ as reference only, never imported): that loader masks everything
*up to the last assistant turn*, so with a multi-turn conversation every
non-final user/tool/system span between assistant turns is TRAINED ON. This
module masks EVERY non-assistant token — leading and intermediate user
turns, system turns, tool/function turns, and consecutive-role boundaries —
with ``-100``, and keeps loss on ALL assistant turns.

Mechanism (spec: "tokenized per-turn then concatenated"): each turn's text
is the character delta between growing chat-template renderings; tokens are
produced per turn and concatenated with label masks attached per turn, so a
cross-turn merge can never leak loss into a masked span. When the tokenizer
does not compose over prefixes (some templates inject state) or is not a
fast tokenizer (no exact offsets), we fall back to whole-text tokenization
with char-region masking computed from the same prefix renderings — still
all-non-assistant -100, never the airllm last-turn-only semantics. A row
whose template cannot be aligned at all returns ``None`` so the caller can
skip it instead of training unmasked (stricter than falling back to "train
on everything").

First-turn-start boundaries: the generation-prompt header of an assistant
turn (e.g. ``<|start|>assistant``) is rendered as part of the *previous*
turn's delta via ``apply_chat_template(msgs[:k], add_generation_prompt=True)``
and therefore carries label -100; only the assistant content + terminator
are trainable. No cross-turn spurious loss.

Torch-free and tokenizer-agnostic (duck-typed ``tokenizer``), so unit tests
drive it with a fake template tokenizer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

IGNORE_INDEX = -100

#: ShareGPT 'from' roles -> OpenAI roles (same mapping worker.py uses).
ROLE_MAP = {
    "human": "user",
    "gpt": "assistant",
    "system": "system",
    "tool": "tool",
    "function_call": "assistant",
}


def sharegpt_to_messages(conversations: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Convert one ShareGPT row to messages; unknown roles are dropped."""
    messages: List[Dict[str, str]] = []
    for turn in conversations or []:
        role = ROLE_MAP.get(str(turn.get("from", "")).lower())
        if role is None:
            continue
        messages.append({"role": role, "content": turn.get("value") or ""})
    return messages


def _render(tokenizer: Any, messages: List[Dict[str, str]], *,
            gen_prompt: bool) -> Optional[str]:
    if not messages:
        # transformers >= 5 refuses to template an empty conversation; the
        # empty prefix is "" by definition (templates with a prologue before
        # the first message fold it into turn 0's delta — the boundary check
        # in mask_conversation catches any resulting mismatch).
        return ""
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=gen_prompt)
    except Exception:
        return None


def assistant_char_regions(tokenizer: Any,
                           messages: List[Dict[str, str]]) -> Optional[List[Tuple[int, int]]]:
    """Char regions of every assistant turn inside the full rendering.

    The k-th message spans ``[len(render(msgs[:k], gen_prompt=True)),
    len(render(msgs[:k+1]))))``; only assistant-role messages are kept.
    Returns None when the template does not compose over prefixes (the
    caller must then skip the row, NOT train unmasked).
    """
    if not any(m["role"] == "assistant" for m in messages):
        return None
    full = _render(tokenizer, messages, gen_prompt=False)
    if full is None:
        return None
    regions: List[Tuple[int, int]] = []
    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        prefix = _render(tokenizer, messages[:i], gen_prompt=True)
        upto = _render(tokenizer, messages[: i + 1], gen_prompt=False)
        if prefix is None or upto is None:
            return None
        if not full.startswith(prefix) or not full.startswith(upto):
            return None
        start, end = len(prefix), len(upto)
        if end > start:
            regions.append((start, end))
    return regions or None


def _tokenize_text(tokenizer: Any, text: str) -> List[int]:
    enc = tokenizer(text, add_special_tokens=False)
    # transformers returns a BatchEncoding (not always a dict subclass);
    # prefer the mapping access and fall back to treating it as ids.
    try:
        ids = enc["input_ids"]
    except (TypeError, KeyError, IndexError):
        ids = list(enc)
    return [int(t) for t in ids]


def mask_conversation(tokenizer: Any, messages: List[Dict[str, str]],
                      max_length: int) -> Optional[Dict[str, List[int]]]:
    """One message list -> {input_ids, attention_mask, labels} or None.

    Masking semantics (stricter than airllm's messages loader): every token
    NOT inside an assistant turn gets label -100 — that includes system,
    user and tool turns anywhere in the conversation, not just the prefix
    before the final assistant turn. All assistant turns contribute loss.

    Strategy: per-turn tokenize + concatenate (exact per-turn boundaries);
    verify the concatenation equals a whole-text tokenization at each
    boundary. When verification fails (cross-turn BPE merges) fall back to
    whole-text tokenization with char-region masking over offsets — same
    -100 semantics. Returns None when neither path is exact (slow
    tokenizer / non-composing template / fully-masked row), so the caller
    skips the row instead of guessing.
    """
    if not messages:
        return None
    full_text = _render(tokenizer, messages, gen_prompt=False)
    if full_text is None or not full_text.strip():
        return None

    per_turn = mask_per_turn(tokenizer, messages)
    if per_turn is not None:
        ids, labels = per_turn
        # Boundary check: per-turn concatenation must reproduce the whole
        # text's tokenization, else a merge crossed a turn boundary.
        whole = _tokenize_text(tokenizer, full_text)
        if whole[: len(ids)] == ids and len(ids) <= len(whole):
            return _pack(ids[:max_length], labels[:max_length])

    return mask_by_char_regions(tokenizer, messages, full_text, max_length)


def mask_per_turn(tokenizer: Any,
                  messages: List[Dict[str, str]]) -> Optional[Tuple[List[int], List[int]]]:
    """Tokenize turn-by-turn; assistant turns keep labels, all others -100.

    Each turn's text is the delta between ``render(msgs[:k], gen_prompt=
    True)`` and ``render(msgs[:k+1])`` — so an assistant turn's header
    (generation prompt) belongs to the previous span and is masked with it.
    Returns None when any rendering step fails or a delta is not a suffix
    of the growing rendering (template injects state).
    """
    ids: List[int] = []
    labels: List[int] = []
    prev = _render(tokenizer, [], gen_prompt=True)
    if prev is None:
        return None
    for k, msg in enumerate(messages):
        upto = _render(tokenizer, messages[: k + 1], gen_prompt=False)
        nxt = _render(tokenizer, messages[: k + 1], gen_prompt=True)
        if upto is None or nxt is None:
            return None
        if not upto.startswith(prev):
            return None
        delta = upto[len(prev):]
        trainable = msg["role"] == "assistant"
        tok = _tokenize_text(tokenizer, delta)
        ids.extend(tok)
        labels.extend(tok if trainable else [IGNORE_INDEX] * len(tok))
        prev = nxt
    if not any(l != IGNORE_INDEX for l in labels):
        return None
    return ids, labels


def mask_by_char_regions(tokenizer: Any, messages: List[Dict[str, str]],
                         full_text: str, max_length: int
                         ) -> Optional[Dict[str, List[int]]]:
    """Whole-text fallback: fast-tokenizer offsets vs assistant char regions."""
    if not getattr(tokenizer, "is_fast", False):
        return None
    regions = assistant_char_regions(tokenizer, messages)
    if regions is None:
        return None
    try:
        enc = tokenizer(full_text, truncation=True, max_length=max_length,
                        add_special_tokens=False, return_offsets_mapping=True)
    except Exception:
        return None
    ids = [int(t) for t in enc["input_ids"]]
    offsets = enc.get("offset_mapping") or []
    if len(offsets) != len(ids):
        return None
    labels = [
        tid if any(a < end and b > start for start, end in regions)
        else IGNORE_INDEX
        for tid, (a, b) in zip(ids, offsets)
    ]
    if all(l == IGNORE_INDEX for l in labels):
        return None
    return _pack(ids, labels)


def _pack(ids: List[int], labels: List[int]) -> Dict[str, List[int]]:
    return {
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "labels": labels,
    }

"""Two-stage chat-thread titles (inspiration pass 3, item #18).

Dashboard chat threads used to fall back to the file stem — with the buffer
growing via batch + REPL, untitled threads became unfindable. Pattern adapted
built in:

  * **Stage 1 — instant, deterministic** (``derive_title``): a title computed
    from the opening user message. It cannot fail and needs no model, so every
    thread gets a usable label the moment it is written.

  * **Stage 2 — one small-model call** (``generate_title``): upgrades the
    derived label to a concise summary. Runs OFF the critical path (a daemon
    thread fires after the turn persists), thinking disabled, JSON-constrained;
    any failure just leaves the stage-1 title in place.

Storage enforces a provenance precedence ``derived < llm < user``: a rename by
the human is authoritative and is never overwritten by an automatic upgrade,
and an LLM title is only ever replaced by a better LLM title or a user edit —
never silently clobbered by a re-derivation.

The LLM call is injected (``client_fn(messages, model) -> Optional[str]``), the
same bounded-call shape ``curate/judge.py`` uses, so this module stays free of
HTTP and trivially testable.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional

# Provenance ranks: a stored title is only replaced when the incoming source
# ranks >= the existing one (user edits win outright).
TITLE_SOURCE_RANK: Dict[str, int] = {"derived": 0, "llm": 1, "user": 2}
DERIVED = "derived"
LLM = "llm"
USER = "user"

MAX_TITLE_CHARS = 60

# Leading scaffolding we strip from an opening message before titling: list
# markers, quoting, common "please/hey/can you" openers. Keeps the derived
# label about the task, not the pleasantries.
_LEADING_NOISE = re.compile(
    r"^[\s>*#\-•\d.)\]]+"
    r"|^(?:hey|hi|hello|so|ok|okay|please|pls|can you|could you|would you|"
    r"i (?:want|need) you to|let'?s)\b[\s,:-]+",
    re.IGNORECASE,
)


def derive_title(messages: List[Dict[str, Any]],
                 max_chars: int = MAX_TITLE_CHARS) -> str:
    """Stage 1: instant deterministic title from the first user message.

    Collapses whitespace, strips list markers / pleasantries / code fences, and
    truncates on a word boundary. Falls back to a stable placeholder when there
    is nothing usable (empty thread, image-only turn). Never raises.
    """
    text = ""
    for m in messages or []:
        if isinstance(m, dict) and m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str) and c.strip():
                text = c
                break
    if not text.strip():
        return "Untitled chat"
    # Drop fenced code blocks (a pasted snippet is noise in a title).
    text = re.sub(r"```.*?(?:```|$)", " ", text, flags=re.S)
    # Strip leading scaffolding repeatedly ("hey can you - please …").
    for _ in range(5):
        stripped = _LEADING_NOISE.sub("", text, count=1)
        if stripped == text:
            break
        text = stripped
    text = re.sub(r"\s+", " ", text).strip(" \t:,-—–>#")
    if not text:
        return "Untitled chat"
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    # Truncate on a word boundary when one is near the end.
    sp = cut.rfind(" ")
    if sp >= max_chars - 15:
        cut = cut[:sp]
    return cut.rstrip(" \t,;:-—–") + "…"


def _clean_llm_title(raw: Optional[str]) -> str:
    """Normalize a model-produced title: strip quotes/fences/prefixes/roles."""
    if not raw:
        return ""
    t = raw.strip()
    # JSON-constrained reply: {"title": "..."} (also tolerate a bare string).
    m = re.search(r"\{.*\}", t, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and isinstance(obj.get("title"), str):
                t = obj["title"]
        except (json.JSONDecodeError, TypeError):
            pass
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.M).strip()
    t = re.sub(r'^(?:title)\s*[:\-]\s*', "", t, flags=re.IGNORECASE)
    t = t.strip().strip("\"'`“”‘’ ")
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > MAX_TITLE_CHARS:
        t = t[:MAX_TITLE_CHARS].rsplit(" ", 1)[0].rstrip(" ,;:-—–") + "…"
    return t


def generate_title(
    client_fn: Callable[[List[Dict[str, str]], str], Optional[str]],
    model: str,
    messages: List[Dict[str, Any]],
) -> Optional[str]:
    """Stage 2: one bounded small-model call for a concise title.

    ``client_fn(messages, model)`` performs the actual completion and returns
    the raw text (or None). Thinking is disabled via the sampling-free minimal
    prompt; the reply is JSON-constrained. Returns a cleaned title or None on
    any failure/unusable output — the caller keeps the stage-1 label.
    """
    # Bounded excerpt: first user turn + first assistant reply, each clipped.
    excerpt: List[str] = []
    for m in messages or []:
        if not isinstance(m, dict) or m.get("role") not in ("user", "assistant"):
            continue
        c = m.get("content")
        if isinstance(c, str) and c.strip():
            excerpt.append(f"{m['role']}: {c.strip()[:400]}")
        if len(excerpt) >= 3:
            break
    if not excerpt:
        return None
    prompt = [
        {"role": "system",
         "content": 'You write short titles for chat conversations. Reply with '
                    'ONLY a JSON object: {"title": "<3-6 words, no quotes or '
                    'trailing punctuation>"}. No other text.'},
        {"role": "user",
         "content": "Title this conversation:\n" + "\n".join(excerpt)},
    ]
    try:
        raw = client_fn(prompt, model)
    except Exception:  # a title upgrade must never surface an error
        return None
    title = _clean_llm_title(raw)
    return title or None


def should_apply_title(existing_source: Optional[str],
                       new_source: str) -> bool:
    """Provenance gate: may ``new_source`` overwrite ``existing_source``?

    Precedence derived < llm < user. A missing existing title accepts any
    source; a user rename accepts only another user edit.
    """
    if not existing_source:
        return True
    return TITLE_SOURCE_RANK.get(new_source, 0) >= \
        TITLE_SOURCE_RANK.get(existing_source, 0)

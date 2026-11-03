"""Regex-based secret redaction for durable artifacts (backlog item #1).

Traces recorded by the agent become SFT/DPO training data -> LoRA -> GGUF ->
served weights, so any credential that reaches the JSONL buffer today ends up
in the model forever. This module masks vendor API keys, tokens, private-key
blocks and key=value secret assignments *before* the durable write. The live
in-memory conversation is deliberately left raw (see store/recorder.py: only
the buffer copy passes through here).

Pattern families designed for this codebase
``agent/redact.py`` (attribution: vendor prefix list, short-token masking
rule, sensitive JSON body-key names) — slimmed to the ~15 families that
matter for this machine and rewritten for toaster_loop's stdlib-only constraint.

Masking rule: tokens <= 12 chars are fully masked; longer ones
keep first 6 + last 4 with ``…[REDACTED]…`` between, which preserves
debuggability without leaking the exact token length. PEM blocks are replaced
entirely with ``[REDACTED_PRIVATE_KEY]``.

Leak-path coverage note: this module closes the *record-time* path; read-time
blocking of credential files lives in core/tools.py (``block_sensitive_files``,
defense-in-depth). Together they cover the two ways a secret
enters a trace. An import-time env snapshot is intentionally NOT taken —
redaction here is unconditional at record time and cannot be toggled from
inside the conversation.

Performance: all regexes are compiled at import; ``redact_text`` is a single
pass over an ordered pattern list. Masked output contains ``[REDACTED`` which
the masker recognizes, making ``redact_text`` idempotent by construction.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

REDACTED_MARK = "[REDACTED]"
PEM_PLACEHOLDER = "[REDACTED_PRIVATE_KEY]"

# ---------------------------------------------------------------- prefixes
# Known API-key/token prefixes: prefix + contiguous token chars.
# (_PREFIX_PATTERNS)
_PREFIX_PATTERNS = [
    r"sk-[A-Za-z0-9_-]{10,}",          # OpenAI incl. sk-proj-/sk-ant-
    r"ghp_[A-Za-z0-9]{10,}",           # GitHub PAT (classic)
    r"gho_[A-Za-z0-9]{10,}",           # GitHub OAuth access token
    r"github_pat_[A-Za-z0-9_]{10,}",   # GitHub PAT (fine-grained)
    r"AKIA[0-9A-Z]{16}",               # AWS access key id
    r"ASIA[0-9A-Z]{16}",               # AWS temporary access key id
    r"hf_[A-Za-z0-9_-]{10,}",          # HuggingFace token
    r"xox[baprs]-[A-Za-z0-9-]{10,}",   # Slack bot/app/user tokens
    r"xapp-\d+-[A-Za-z0-9-]{10,}",     # Slack app-level token
    r"AIza[A-Za-z0-9_-]{20,}",         # Google API key
    r"sk_live_[A-Za-z0-9]{10,}",       # Stripe secret key (live)
    r"pk_live_[A-Za-z0-9]{10,}",       # Stripe publishable key (live)
    r"glpat-[A-Za-z0-9_\-]{10,}",      # GitLab personal access token
    r"npm_[A-Za-z0-9]{10,}",           # npm access token
]

_TELEGRAM_RE = re.compile(r"\b\d{8,}:AA[A-Za-z0-9_-]{30,}")  # bot token
_DISCORD_RE = re.compile(
    r"https://(?:can\.|dg\.)?(?:discord|discordapp)\.com/api/webhooks/"
    r"\d{15,}/[A-Za-z0-9_-]{20,}"
)
_JWT_RE = re.compile(
    r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}(?:\.[A-Za-z0-9_=-]*)?"
)
_PEM_RE = re.compile(
    r"-----BEGIN[A-Z ]*PRIVATE KEY-----[\s\S]*?-----END[A-Z ]*PRIVATE KEY-----"
)

# JSON-style fields: values under sensitive key names are masked regardless of
# value shape (opaque tokens match no prefix regex). Best-effort regex over the
# raw text — never parses the document, so malformed JSON is a non-event.
# (_SENSITIVE_BODY_KEYS)
_JSON_KEY_NAMES = (
    r"password|passwd|secret|api_?key|token|authorization|access_token"
    r"|refresh_token|client_secret|private_key|jwt|auth_token"
)
_JSON_FIELD_RE = re.compile(
    rf'"({_JSON_KEY_NAMES})"\s*:\s*"((?:[^"\\]|\\.)*)"',
    re.IGNORECASE,
)

# Generic KEY=value / KEY: value heuristic (env files, dumps, prose). The key
# must be a whole word optionally prefixed by `UPPER_`-style namespaces
# (OPENAI_API_KEY=…); the optional auth-scheme word keeps
# "Authorization: Bearer <cred>" from masking only the scheme. Values that are
# programmatic env lookups (os.getenv(...) etc.) are code, not secrets.
_KV_NAME = (
    r"(?:[A-Za-z0-9_]*_)?(?:password|passwd|secret|token|api_?key|authorization)"
)
_KV_RE = re.compile(
    rf"(?<![\w-])({_KV_NAME})(?![\w])(\s*[=:]\s*)"
    rf"((?:Bearer|Basic|Token|Digest)\s+)?(['\"`]?)([^\s'\"`,;)}}\]]+)\4",
    re.IGNORECASE,
)
_ENV_LOOKUP_RE = re.compile(
    r"^(?:os\.(?:getenv|environ)|process\.env|\$\{?ENV)"
)

# Ordered single pass. PEM first so its body never interacts with the rest.
_PATTERNS: List[tuple] = [
    (re.compile(p), "full") for p in _PREFIX_PATTERNS
] + [
    (_TELEGRAM_RE, "full"),
    (_DISCORD_RE, "full"),
    (_JWT_RE, "full"),
    (_JSON_FIELD_RE, "json"),
    (_KV_RE, "kv"),
]


def _mask_token(tok: str) -> str:
    """Mask one secret: short tokens fully, long ones keep first 6 + last 4.

    Already-masked values (anything containing ``[REDACTED``) pass through
    unchanged — this is what makes redact_text idempotent. The check uses the
    truncated marker so a re-match on ``…[REDACTED`` (regex value classes stop
    at ``]``) is still recognized as already masked.
    """
    if not tok or "[REDACTED" in tok:
        return tok
    if len(tok) <= 12:
        return REDACTED_MARK
    return f"{tok[:6]}…{REDACTED_MARK}…{tok[-4:]}"


def _sub_full(m: "re.Match[str]") -> str:
    return _mask_token(m.group(0))


def _sub_json(m: "re.Match[str]") -> str:
    return f'"{m.group(1)}": "{_mask_token(m.group(2))}"'


def _sub_kv(m: "re.Match[str]") -> str:
    value = m.group(5)
    if _ENV_LOOKUP_RE.match(value):
        return m.group(0)  # code snippet (os.getenv(...) etc.), not a secret
    quote = m.group(4)
    return f"{m.group(1)}{m.group(2)}{m.group(3) or ''}{quote}{_mask_token(value)}{quote}"


_SUBS = {"full": _sub_full, "json": _sub_json, "kv": _sub_kv}


def redact_text(text: str) -> str:
    """Redact known secret shapes from a string. Idempotent; never raises."""
    if not text:
        return text
    out = _PEM_RE.sub(PEM_PLACEHOLDER, text)
    for rx, kind in _PATTERNS:
        out = rx.sub(_SUBS[kind], out)
    return out


def redact_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a redacted deep-ish copy of an OpenAI-style message list.

    Masks string ``content`` fields (and text parts of multimodal content
    lists) plus tool-call ``function.arguments`` strings. Never mutates the
    input — the live conversation stays raw; only durable copies are scrubbed.
    """
    out: List[Dict[str, Any]] = []
    for m in messages or []:
        if not isinstance(m, dict):
            out.append(m)
            continue
        nm = dict(m)
        content = nm.get("content")
        if isinstance(content, str):
            nm["content"] = redact_text(content)
        elif isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    p = dict(part)
                    p["text"] = redact_text(p["text"])
                    parts.append(p)
                else:
                    parts.append(part)
            nm["content"] = parts
        tool_calls = nm.get("tool_calls")
        if isinstance(tool_calls, list):
            new_tcs = []
            for tc in tool_calls:
                if isinstance(tc, dict):
                    ntc = dict(tc)
                    fn = ntc.get("function")
                    if isinstance(fn, dict):
                        nfn = dict(fn)
                        args = nfn.get("arguments")
                        if isinstance(args, str):
                            nfn["arguments"] = redact_text(args)
                        ntc["function"] = nfn
                    new_tcs.append(ntc)
                else:
                    new_tcs.append(tc)
            nm["tool_calls"] = new_tcs
        out.append(nm)
    return out

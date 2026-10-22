"""Exact token counting via llama.cpp's ``POST /tokenize`` (improvement #4).

Token budgets across sable were estimated as ``len(text) // 4`` (chars/4 —
see core/compression.estimate_tokens). That is systematically wrong for CJK,
code and other tokenizer-edge-case content, which skews compression triggers.
llama.cpp's server exposes a real tokenizer endpoint:

    POST {root}/tokenize  {"content": "<text>", "add_special": false}
                      ->  {"tokens": [...], "count": N}

``/tokenize`` is served at the HTTP **root** (not under ``/v1``), so this
module strips a trailing ``/vN`` from the OpenAI-style base_url it is given.

Design:
  * :class:`TokenCounter` — one per endpoint. ``count(text)`` returns the
    exact token count; on ANY failure (connection refused, timeout, non-200,
    malformed body) it degrades to ``len(text) // 4`` for that call and never
    raises into callers.
  * LRU cache (bounded, default 1024 entries, keyed by exact text). Repeated
    message content across compression passes hits the cache instead of the
    network. LIMITATION: exact-match only — no longest-cached-prefix reuse
    (prefix matching over an LRU is O(keys) string compares per lookup; not
    cheap enough to justify here).
  * Circuit breaker: reuses core/circuit.CircuitBreaker (own instance, NOT
    the shared per-backend registry — a broken /tokenize must not poison the
    chat breaker). After ``failure_threshold`` consecutive failures probing
    stops until ``cooldown_seconds`` pass (half-open single probe recovers).
    One WARNING is logged when the circuit opens, not per failed call.
  * Short timeout (default 2 s): counting sits at gate-decision points only
    (whole-message granularity); it must never stall a run behind a wedged
    tokenizer probe. Callers must NOT count inside streaming delta loops.

HTTP client: httpx, matching core/agent_runner.py conventions (no new deps).
"""

from __future__ import annotations

import json
import logging
import re
import threading
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional

import httpx

from .circuit import CircuitBreaker

logger = logging.getLogger("sable.tokenizer")

#: Per-message chat-template framing overhead (role markers, delimiters) that
#: content-only tokenization cannot see. llama.cpp's /tokenize accepts a
#: ``messages`` + ``use_server_template`` form on recent builds, but it is
#: version-dependent; we keep the guaranteed content-based form and add this
#: documented constant per message (and per tool_call), matching the +4 that
#: estimate_tokens always used so the degraded path stays byte-identical.
MESSAGE_FRAMING_TOKENS = 4

_V_SUFFIX_RE = re.compile(r"/v\d+$")


class TokenCounter:
    """Exact token counts for one served endpoint, with graceful fallback.

    ``count(text) -> int`` and ``count_messages(messages) -> int`` never
    raise: anything other than a clean 200 answer falls back to chars/4 for
    that call. Thread-safe (the dashboard and a run may share one counter).
    """

    def __init__(self, base_url: str, *, timeout: float = 2.0,
                 cache_size: int = 1024, failure_threshold: int = 3,
                 cooldown_seconds: float = 60.0,
                 framing_overhead: int = MESSAGE_FRAMING_TOKENS,
                 client: Optional[httpx.Client] = None,
                 clock: Optional[Callable[[], float]] = None) -> None:
        # /tokenize lives at the server root; OpenAI base_urls end in /v1.
        self.base_url = base_url.rstrip("/")
        self.tokenize_url = _V_SUFFIX_RE.sub("", self.base_url) + "/tokenize"
        self.timeout = float(timeout)
        self.framing_overhead = int(framing_overhead)
        self._cache_size = max(1, int(cache_size))
        self._cache: "OrderedDict[str, int]" = OrderedDict()
        self._lock = threading.Lock()
        # Own breaker instance (see module docstring): consecutive-failure
        # counting + cooldown + half-open probe, same semantics as #28.
        self._breaker = CircuitBreaker(failure_threshold=failure_threshold,
                                       cooldown_seconds=cooldown_seconds,
                                       clock=clock)
        self._client = client  # lazily created unless injected
        self._owns_client = client is None
        self._client_lock = threading.Lock()
        self._warn_open = False

    # ------------------------------------------------------------------ api

    def count(self, text: str) -> int:
        """Exact token count of *text*; chars/4 fallback; never raises."""
        if not text:
            return 0
        with self._lock:
            hit = self._cache.get(text)
            if hit is not None:
                self._cache.move_to_end(text)
                return hit
        if not self._breaker.is_allowed():
            # Circuit open: don't even probe — degrade silently (the WARNING
            # already went out when the circuit opened).
            return len(text) // 4
        try:
            n = self._post_tokenize(text)
        except Exception as exc:  # noqa: BLE001 — degrade, never raise
            self._breaker.record_failure()
            if (not self._warn_open
                    and self._breaker.state == "open"):
                self._warn_open = True
                logger.warning(
                    "tokenize endpoint %s failing (%s: %s); falling back to "
                    "chars/4 estimates until it recovers",
                    self.tokenize_url, type(exc).__name__, exc)
            return len(text) // 4
        self._breaker.record_success()
        self._warn_open = False
        with self._lock:
            self._cache[text] = n
            self._cache.move_to_end(text)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return n

    def count_messages(self, messages: List[Dict[str, Any]]) -> int:
        """Token count of an OpenAI-style message list (whole-message gate use).

        Counts each message's content (JSON-serializing non-string parts) and
        every tool_call name+arguments, plus ``framing_overhead`` per counted
        part to approximate chat-template wrapping. The shape mirrors
        core.compression.estimate_tokens one-for-one (a message with no
        content and no tool_calls contributes 0 there, so also here): when the
        endpoint is degraded this equals estimate_tokens EXACTLY — same
        chars/4 math, same +4 overhead — and only a live tokenizer changes
        gate outcomes.
        """
        total = 0
        for m in messages:
            if not isinstance(m, dict):
                continue
            content = m.get("content")
            if isinstance(content, str):
                total += self.count(content) + self.framing_overhead
            elif content is not None:
                total += (self.count(json.dumps(content, default=str))
                          + self.framing_overhead)
            for tc in m.get("tool_calls") or []:
                fn = (tc.get("function") or {})
                blob = f"{fn.get('name') or ''}{fn.get('arguments') or ''}"
                total += self.count(blob) + self.framing_overhead
        return total

    @property
    def exact_available(self) -> bool:
        """False while the circuit is open (calls are falling back)."""
        return self._breaker.state != "open"

    def close(self) -> None:
        """Close the HTTP client — only when this counter created it.

        An injected client (the runner's own httpx.Client, shared so the
        tokenize probe reuses its auth headers and pool) stays open; its
        owner closes it.
        """
        with self._client_lock:
            if self._client is not None and self._owns_client:
                try:
                    self._client.close()
                except Exception:
                    pass
                self._client = None

    # ------------------------------------------------------------- internals

    def _http(self) -> httpx.Client:
        with self._client_lock:
            if self._client is None:
                self._client = httpx.Client(timeout=self.timeout)
            return self._client

    def _post_tokenize(self, text: str) -> int:
        # Per-request timeout override: a shared client may carry the runner's
        # long chat timeout; tokenizer probes stay short (self.timeout).
        r = self._http().post(self.tokenize_url, json={
            "content": text, "add_special": False}, timeout=self.timeout)
        if r.status_code != 200:
            raise RuntimeError(
                f"HTTP {r.status_code} from {self.tokenize_url}: "
                f"{r.text[:200]}")
        data = r.json()
        tokens = data.get("tokens")
        if isinstance(tokens, list):
            return len(tokens)
        count = data.get("count")
        if isinstance(count, int) and not isinstance(count, bool):
            return count
        raise RuntimeError(f"malformed /tokenize response: {str(data)[:200]}")

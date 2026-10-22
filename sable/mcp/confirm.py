"""Plan-then-confirm token store for mutating MCP tools (WP6).

Every mutating tool runs two-phase: the first call returns a human-readable
PLAN plus an opaque confirm_token; the second call — presenting that token —
executes. Properties enforced here:

  * **Opaque + random**: tokens come from ``secrets`` (URL-safe, 256 bits of
    entropy); they carry no structure or capability beyond the store entry.
  * **Server-side, in-memory only**: nothing is persisted to disk; a server
    restart invalidates every outstanding plan (documented behavior).
  * **TTL**: entries expire after ``ttl_s`` seconds (default 300) so an
    abandoned plan cannot linger indefinitely.
  * **Single-use**: consuming a token deletes it atomically; a replay of the
    same token is refused.
  * **Bound to (tool, args-hash)**: a token issued for one tool/argument
    combination cannot execute a different call — the hash covers the tool
    name and the canonical JSON of the arguments.

STDLIB-ONLY (D-12).
"""

from __future__ import annotations

import hashlib
import json
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

#: Default lifetime of an outstanding confirmation plan, in seconds.
DEFAULT_TTL_S = 300.0


def args_fingerprint(tool: str, args: Dict[str, Any]) -> str:
    """Deterministic SHA-256 over (tool, args) — the token's binding key.

    ``sort_keys`` + compact separators make the digest independent of dict
    ordering and whitespace, so an equivalent call always matches.
    """
    blob = json.dumps({"tool": tool, "args": args}, sort_keys=True,
                      separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass
class _Entry:
    fingerprint: str
    expires_at: float


class ConfirmStore:
    """In-memory single-use confirmation tokens bound to (tool, args-hash)."""

    def __init__(self, ttl_s: float = DEFAULT_TTL_S,
                 clock: Optional[Callable[[], float]] = None) -> None:
        self.ttl_s = float(ttl_s)
        # Injectable clock (monotonic by default) so tests can expire tokens
        # without sleeping.
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._entries: Dict[str, _Entry] = {}

    def issue(self, tool: str, args: Dict[str, Any]) -> str:
        """Create and remember a token bound to this exact (tool, args) call."""
        token = secrets.token_urlsafe(32)
        entry = _Entry(fingerprint=args_fingerprint(tool, args),
                       expires_at=self._clock() + self.ttl_s)
        with self._lock:
            self._prune_locked()
            self._entries[token] = entry
        return token

    def consume(self, token: str, tool: str,
                args: Dict[str, Any]) -> Optional[str]:
        """Validate + consume *token*; returns None on success, else a reason.

        Successful consumption DELETES the entry (single use). Any failure
        leaves the entry untouched except expiry, so a wrong-token guess
        cannot destroy a legitimately pending plan.
        """
        if not isinstance(token, str) or not token:
            return "confirm_token is required to execute this tool"
        fp = args_fingerprint(tool, args)
        now = self._clock()
        with self._lock:
            entry = self._entries.get(token)
            if entry is None:
                return ("unknown or already-used confirm_token — plans are "
                        "single-use and expire; call the tool again to get a "
                        "fresh plan")
            if now >= entry.expires_at:
                del self._entries[token]
                return ("confirm_token expired — the plan is only valid for "
                        f"{self.ttl_s:.0f}s; call the tool again for a fresh "
                        "plan")
            if entry.fingerprint != fp:
                # Do NOT consume: the token was issued for a different call.
                return ("confirm_token does not match this call — it is bound "
                        "to the exact tool and arguments of the plan it was "
                        "issued for")
            del self._entries[token]
            return None

    def pending_count(self) -> int:
        with self._lock:
            self._prune_locked()
            return len(self._entries)

    def _prune_locked(self) -> None:
        now = self._clock()
        for t in [t for t, e in self._entries.items()
                  if now >= e.expires_at]:
            del self._entries[t]

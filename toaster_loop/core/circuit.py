# Adapted from PIASystem (MIT) app/core/error_recovery.py
"""Per-backend circuit breaker (inspiration pass 4 #28).

A wedged llama.cpp backend must not be hammered by every agent run and every
judge call: after ``failure_threshold`` *consecutive* transport-level failures
the breaker OPENs and calls are refused for ``cooldown_seconds``. After the
cooldown it HALF_OPENs and exactly one probe call is allowed through; probe
success closes the circuit, probe failure re-opens it with a fresh cooldown.

Self-contained (stdlib only) and thread-safe — the dashboard and the runner
can both touch the same breaker. No config loading lives here: callers pass
thresholds to ``get_breaker`` (defaults apply on first creation of a key).
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Dict, Optional

CLOSED = "closed"
OPEN = "open"
HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    """Call refused because the backend's circuit is open (fail fast)."""


class CircuitBreaker:
    """CLOSED -> OPEN after N consecutive failures; HALF_OPEN single probe."""

    def __init__(self, failure_threshold: int = 5,
                 cooldown_seconds: float = 60.0,
                 clock: Optional[Callable[[], float]] = None) -> None:
        self.failure_threshold = max(1, int(failure_threshold))
        self.cooldown_seconds = float(cooldown_seconds)
        # Injectable monotonic clock (tests); production uses time.monotonic.
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._state = CLOSED
        self._failures = 0
        self._opened_at = 0.0          # monotonic, cooldown math
        self._opened_wall: Optional[float] = None  # wall epoch, display only
        self._probe_inflight = False
        self._probe_started = 0.0

    # ------------------------------------------------------------------ state

    @property
    def state(self) -> str:
        with self._lock:
            self._maybe_half_open()
            return self._state

    @property
    def failures(self) -> int:
        with self._lock:
            return self._failures

    @property
    def opened_at(self) -> Optional[float]:
        """Wall-clock epoch of the last OPEN transition (display only)."""
        with self._lock:
            return self._opened_wall if self._state == OPEN else None

    def seconds_until_probe(self) -> float:
        with self._lock:
            if self._state != OPEN:
                return 0.0
            left = self.cooldown_seconds - (self._clock() - self._opened_at)
            return max(0.0, left)

    # ----------------------------------------------------------------- guards

    def is_allowed(self) -> bool:
        """True when a call may proceed. In HALF_OPEN grants exactly one."""
        with self._lock:
            self._maybe_half_open()
            if self._state == OPEN:
                return False
            if self._state == HALF_OPEN:
                # A stuck probe (caller died mid-call) is released after one
                # more cooldown so the breaker can never wedge permanently.
                if self._probe_inflight and \
                        (self._clock() - self._probe_started) < self.cooldown_seconds:
                    return False
                self._probe_inflight = True
                self._probe_started = self._clock()
            return True

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._state = CLOSED
            self._probe_inflight = False

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._state == HALF_OPEN:
                self._trip()  # probe failed: fresh cooldown
            elif self._state == CLOSED and \
                    self._failures >= self.failure_threshold:
                self._trip()

    # ---------------------------------------------------------------- helpers

    def _maybe_half_open(self) -> None:
        if self._state == OPEN and \
                (self._clock() - self._opened_at) >= self.cooldown_seconds:
            self._state = HALF_OPEN

    def _trip(self) -> None:
        self._state = OPEN
        self._opened_at = self._clock()
        self._opened_wall = time.time()
        self._probe_inflight = False

    # --------------------------------------------------------------- display

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            self._maybe_half_open()
            left = 0.0
            if self._state == OPEN:
                left = max(0.0, self.cooldown_seconds
                           - (self._clock() - self._opened_at))
            return {
                "state": self._state,
                "failures": self._failures,
                "opened_at": self._opened_wall if self._state == OPEN else None,
                "seconds_until_probe": round(left, 3),
            }


# ---------------------------------------------------------------- registry

_registry: Dict[str, CircuitBreaker] = {}
_registry_lock = threading.Lock()


def get_breaker(key: str, *, failure_threshold: int = 5,
                cooldown_seconds: float = 60.0,
                clock: Optional[Callable[[], float]] = None) -> CircuitBreaker:
    """Breaker for `key` (backend base_url), created on first use.

    Threshold args only take effect at creation; later callers reuse the
    existing breaker so runner and judge share one view of a backend."""
    with _registry_lock:
        br = _registry.get(key)
        if br is None:
            br = CircuitBreaker(failure_threshold=failure_threshold,
                                cooldown_seconds=cooldown_seconds, clock=clock)
            _registry[key] = br
        return br


def all_breakers() -> Dict[str, Dict[str, object]]:
    """Snapshot of every registered breaker, for status surfaces."""
    with _registry_lock:
        return {k: b.snapshot() for k, b in _registry.items()}

"""Thread-safe in-process event bus for the dashboard (Phase 5a).

One ``EventBus`` instance lives on each ``DashboardApp``. Producers (job
scheduler transitions, serve start/stop/deploy, registry promote/rollback,
trace recorder hook, iterate steps) call ``publish(kind, payload)``; SSE
clients consume through per-connection ``Subscriber`` queues.

Design notes
------------
* Per-subscriber bounded ``queue.Queue``: a slow consumer never blocks a
  producer. On overflow the oldest pending event is dropped and the subscriber
  emits a single synthetic ``overflow`` event (payload ``{"dropped": N}``)
  before resuming, so clients know to refetch state.
* Publishing to a bus with no subscribers is free (no allocation beyond the
  event dict — kept trivial on purpose).
* Event kinds: ``job.state``, ``job.log``,
  ``serve.state``, ``model.promoted``, ``model.rolled_back``,
  ``trace.recorded``, ``iterate.step``, ``delegation.update`` (Phase 2:
  payload {delegation_id, agent, status, project, thread, child_trace_id,
  parent_run_id}), ``delegation.wake`` (D-17 Phase 2: an inbox wake run was
  started — payload {thread, run_id, delegation_ids}; the chat panel refreshes
  the thread to render the synthetic turn).
"""

from __future__ import annotations

import itertools
import json
import queue
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Subscriber:
    """One consumer's view of the bus (per SSE connection)."""

    def __init__(self, bus: "EventBus", maxsize: int = 256):
        self._bus = bus
        self._q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=maxsize)
        self._dropped = 0
        self._pending: Optional[Dict[str, Any]] = None  # held during overflow
        self.closed = False

    def _offer(self, event: Dict[str, Any]) -> None:
        """Producer-side enqueue. Drops the oldest event when full."""
        if self.closed:
            return
        try:
            self._q.put_nowait(event)
        except queue.Full:
            try:
                self._q.get_nowait()  # drop oldest
            except queue.Empty:  # pragma: no cover - race only
                pass
            self._dropped += 1
            try:
                self._q.put_nowait(event)
            except queue.Full:  # pragma: no cover - defensive
                pass

    def poll(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Next event, or None on timeout. Emits an ``overflow`` marker first."""
        if self._pending is not None:
            out, self._pending = self._pending, None
            return out
        if self.closed and self._q.empty():
            return None
        try:
            ev = self._q.get(timeout=timeout)
        except queue.Empty:
            return None
        if self._dropped:
            n, self._dropped = self._dropped, 0
            # Hold the real event for the next poll; surface the marker ahead
            # of it (re-queueing would move it to the tail and reorder).
            self._pending = ev
            return {"id": -1, "kind": "overflow", "ts": _utcnow(),
                    "payload": {"dropped": n}}
        return ev

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self._bus.unsubscribe(self)


class EventBus:
    """Fan-out publish/subscribe with bounded per-client queues."""

    def __init__(self, maxsize: int = 256):
        self._lock = threading.Lock()
        self._subs: list[Subscriber] = []
        self._seq = itertools.count(1)
        self._maxsize = maxsize

    def subscribe(self) -> Subscriber:
        sub = Subscriber(self, maxsize=self._maxsize)
        with self._lock:
            self._subs.append(sub)
        return sub

    def unsubscribe(self, sub: Subscriber) -> None:
        with self._lock:
            try:
                self._subs.remove(sub)
            except ValueError:
                pass

    def publish(self, kind: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Broadcast one event; returns the event dict (useful in tests)."""
        event = {
            "id": next(self._seq),
            "kind": kind,
            "ts": _utcnow(),
            "payload": payload or {},
        }
        with self._lock:
            subs = list(self._subs)
        for s in subs:
            s._offer(event)
        return event

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subs)


def format_sse(event: Dict[str, Any]) -> str:
    """Render a bus event as an SSE frame (``event:``/``data:`` lines)."""
    data = json.dumps(event, ensure_ascii=False, default=str)
    return f"id: {event.get('id')}\nevent: {event.get('kind')}\ndata: {data}\n\n"


def sse_frame(kind: str, payload: Dict[str, Any]) -> str:
    """Ad-hoc SSE frame (used by the chat response stream, not the bus)."""
    return f"event: {kind}\ndata: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"

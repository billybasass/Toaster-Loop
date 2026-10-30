"""Trace listing/detail, feedback, failure remediation notes and
the read-time swimlane timeline (#35).
"""

# Domain slice of the former monolithic dashboard/app.py: a pure move,
# no behavior change. Handlers resolve through DashboardApp attribute
# lookup (see ROUTES in app.py).

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


from .base import ApiContext, HttpError, _utcnow


def _delegation_header(jsonl_path: str) -> Optional[Dict[str, Any]]:
    """Read-time extraction of the trace header's ``delegation`` linkage block
(audit §3 chip): first JSONL line only, so listing stays cheap. None for
ordinary runs and pruned/corrupt files — readers use tolerant .get()."""
    try:
        with open(jsonl_path, encoding="utf-8") as f:
            rec = json.loads(f.readline() or "{}")
    except (OSError, ValueError):
        return None
    if not isinstance(rec, dict):
        return None
    d = rec.get("delegation")
    return d if isinstance(d, dict) else None


class TracesRoutes(ApiContext):
    """Trace listing/detail, feedback, failure remediation notes and
the read-time swimlane timeline (#35).
    """

    def h_traces_list(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        limit = int(q.get("limit", "50") or 50)
        offset = max(0, int(q.get("offset", "0") or 0))
        curated_raw = q.get("curated")
        curated: Optional[bool] = None
        if curated_raw not in (None, ""):
            curated = curated_raw.lower() in ("1", "true", "yes")
        store = self._store()
        try:
            rows = store.list_traces(
                limit=min(limit, 500),
                status=q.get("status") or None,
                since=q.get("since") or None,
                q=q.get("q") or None,
                agent_profile=q.get("profile") or None,
                model_alias=q.get("model_alias") or None,
                curated=curated,
                offset=offset,
                project=q.get("project") or None,
            )
            # 5d: total across the whole filtered set (not just this window)
            # so pagers can render "page X of Y".
            total = store.count_traces(
                status=q.get("status") or None,
                since=q.get("since") or None,
                q=q.get("q") or None,
                agent_profile=q.get("profile") or None,
                model_alias=q.get("model_alias") or None,
                curated=curated,
                project=q.get("project") or None,
            )
        finally:
            store.close()
        traces = []
        for r in rows:
            d = r.to_dict()
            # Audit §3: surface the header's delegation linkage so the list can
            # render a chip without per-row detail fetches. Key omitted entirely
            # for ordinary runs (tolerant .get() client-side).
            dg = _delegation_header(r.jsonl_path)
            if dg:
                d["delegation"] = dg
            traces.append(d)
        return {"traces": traces, "total": total,
                "offset": offset, "limit": min(limit, 500)}

    def h_traces_stats(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        days = int(q.get("days", "30") or 30)
        store = self._store()
        try:
            series = store.stats_daily(days=days)
        finally:
            store.close()
        return {"series": series}

    def h_insights(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        """Backlog #11: insights report over the buffer DB (read-only)."""
        from ...store.insights import generate_insights

        try:
            days = int(q.get("days", "30") or 30)
        except ValueError:
            raise HttpError(400, "days must be an integer") from None
        return generate_insights(self.home / "buffer" / "sable.db",
                                 days=days, profile=q.get("profile") or None)

    def h_trace_detail(self, q: Dict[str, str], b: Dict[str, Any],
                       trace_id: str) -> Dict[str, Any]:
        store = self._store()
        try:
            row = store.get_trace(trace_id)
            feedback = store.get_feedback(trace_id) if row else []
        finally:
            store.close()
        if row is None:
            raise HttpError(404, f"no trace {trace_id}")
        messages: List[Dict[str, Any]] = []
        records: List[Dict[str, Any]] = []
        try:
            with open(row.jsonl_path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    records.append(rec)
                    if rec.get("kind") == "message":
                        messages.append(rec["message"])
        except (OSError, json.JSONDecodeError):
            pass  # transcript file may be pruned; index still valid
        out: Dict[str, Any] = {"trace": row.to_dict(), "feedback": feedback,
                               "messages": messages}
        qq = q.get("q")
        if qq:
            needle = qq.lower()
            out["messages"] = [
                m for m in messages
                if needle in json.dumps(m, ensure_ascii=False, default=str).lower()
            ]
        # Pass 4 #35: derived swimlane segments (read-time only; stored trace
        # untouched). Anchored into the *delivered* (possibly q-filtered) list.
        out["timeline"] = build_trace_timeline(records, out["messages"])
        return out

    def h_feedback(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        return self.add_feedback(b)

    def h_trace_feedback(self, q: Dict[str, str], b: Dict[str, Any],
                         trace_id: str) -> Dict[str, Any]:
        body = dict(b)
        body.pop("trace_id", None)
        body.pop("last", None)
        body["trace_id"] = trace_id
        return self.add_feedback(body)

    def add_feedback(self, b: Dict[str, Any]) -> Dict[str, Any]:
        score = b.get("score")
        note = b.get("note")
        if score is not None:
            try:
                score = float(score)
            except (TypeError, ValueError):
                raise HttpError(400, "score must be a number") from None
            if not -1.0 <= score <= 1.0:
                raise HttpError(400, "score must be within [-1, 1]")
        if score is None and not note:
            raise HttpError(400, "provide score and/or note")

        store = self._store()
        try:
            row = store.latest_trace() if b.get("last") else store.get_trace(
                str(b.get("trace_id", ""))
            )
            if row is None:
                raise HttpError(404, "no such trace")
            store.add_feedback(
                row.trace_id, score=score, comment=note, source="user",
                timestamp=_utcnow(),
            )
        finally:
            store.close()
        self.bus.publish("trace.feedbacked", {"trace_id": row.trace_id})
        return {"ok": True, "trace_id": row.trace_id}

    def h_failure_note_set(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        reason = str(b.get("reason") or "").strip()
        if not reason:
            raise HttpError(400, "reason required")
        status = str(b.get("status") or "").strip().lower()
        from ...store.db import FAILURE_NOTE_STATUSES
        if status not in FAILURE_NOTE_STATUSES:
            raise HttpError(
                400, f"status must be one of {', '.join(FAILURE_NOTE_STATUSES)}")
        note = str(b.get("note") or "")
        profile = (str(b.get("profile")).strip() or None) if b.get("profile") else None
        store = self._store()
        try:
            store.set_failure_note(reason, status, note=note, profile=profile)
            current = store.get_failure_note(reason, profile=profile)
        finally:
            store.close()
        return {"ok": True, "reason": reason, "profile": profile,
                "status": status, "note": current}


# ------------------------------------------- module-level helpers

TIMELINE_KINDS = ("llm", "tool", "retry", "scrub", "other")

TIMELINE_MAX_SEGMENTS = 200

def _timeline_ts(value: Any) -> Optional[float]:
    """Epoch seconds from an ISO string / epoch number; None when unusable."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        return v if v >= 0 else None
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except (TypeError, ValueError):
        return None

def _timeline_kind(m: Dict[str, Any]) -> Tuple[str, str]:
    """(kind, label) for one trace message; kind always in TIMELINE_KINDS."""
    role = str(m.get("role") or "")
    if role == "tool":
        name = str(m.get("name") or "tool")
        return ("retry", name) if name.startswith("retry_") else ("tool", name)
    if role == "assistant":
        return ("llm", "assistant")
    if role == "system":
        # Compression inserts exactly one rolling-summary system message whose
        # content starts with compression.SUMMARY_PREFIX (core/compression.py).
        try:
            from ...core.compression import SUMMARY_PREFIX
            if str(m.get("content") or "").startswith(SUMMARY_PREFIX):
                return ("scrub", "compression summary")
        except Exception:  # pragma: no cover - defensive; kind stays 'other'
            pass
        return ("other", "system")
    if role == "user":
        return ("other", "user")
    return ("other", role or "unknown")

def build_trace_timeline(records: List[Dict[str, Any]],
                         messages: Optional[List[Dict[str, Any]]] = None,
                         ) -> List[Dict[str, Any]]:
    """Derive swimlane segments from raw trace JSONL records (read-time only).

    Stored traces are never mutated; durations are computed here from
    per-event timestamps. Timestamp lookup order per message record: the
    record's own ``ts``/``timestamp``, then the embedded message's fields —
    today's recorder writes none of these, so a plain trace yields only the
    single run-wide segment anchored at the header timestamp and closed by
    footer ``duration_s`` (the "everything was one llm/tool phase" fallback is
    deliberately honest: zero-duration event streams render nothing client-side).

    Each returned entry: {start_ts, duration_ms, kind, label, message_index}
    where kind is in TIMELINE_KINDS and message_index indexes ``messages``
    (the delivered transcript; defaults to the messages found in ``records``).
    Degenerate inputs yield [] — never raises.
    """
    header: Optional[Dict[str, Any]] = None
    footer: Optional[Dict[str, Any]] = None
    msg_records: List[Tuple[Optional[float], Dict[str, Any]]] = []
    for rec in records or []:
        if not isinstance(rec, dict):
            continue
        kind = rec.get("kind")
        if kind == "trace_header":
            header = rec
        elif kind == "trace_footer":
            footer = rec
        elif kind == "message" and isinstance(rec.get("message"), dict):
            ts = _timeline_ts(rec.get("ts"))
            if ts is None:
                ts = _timeline_ts(rec.get("timestamp"))
            if ts is None:
                ts = _timeline_ts(rec["message"].get("ts"))
            if ts is None:
                ts = _timeline_ts(rec["message"].get("timestamp"))
            msg_records.append((ts, rec["message"]))

    # message_index must anchor into the *delivered* transcript (the detail
    # view may have q-filtered it): match by object identity first, then fall
    # back to positional order over the unfiltered records.
    id_index: Dict[int, int] = {}
    if messages is not None:
        for i, m in enumerate(messages):
            id_index.setdefault(id(m), i)

    def _anchor(i: int, m: Dict[str, Any]) -> Optional[int]:
        if messages is None:
            return i
        # Identity match only: when the transcript was q-filtered, a message
        # that did not survive filtering has no valid index -> skip its segment.
        return id_index.get(id(m))

    raw_start = _timeline_ts((header or {}).get("timestamp"))
    dur_s = (footer or {}).get("duration_s")
    end_ts: Optional[float] = None
    if isinstance(dur_s, (int, float)) and not isinstance(dur_s, bool):
        if raw_start is not None:
            end_ts = raw_start + max(0.0, float(dur_s))
        else:
            end_ts = max(0.0, float(dur_s))

    timed = [(ts, i, m) for i, (ts, m) in enumerate(msg_records) if ts is not None]
    timed.sort(key=lambda t: t[0])

    segs: List[Dict[str, Any]] = []
    if timed:
        # Per-event durations: each message owns the gap until the next event.
        for n, (ts, i, m) in enumerate(timed):
            nxt = timed[n + 1][0] if n + 1 < len(timed) else end_ts
            duration_ms = int(round(((nxt - ts) * 1000.0) if (nxt is not None
                                                               and nxt >= ts) else 0))
            kind, label = _timeline_kind(m)
            mi = _anchor(i, m)
            if mi is None:
                continue
            segs.append({"start_ts": round(ts, 3), "duration_ms": duration_ms,
                         "kind": kind, "label": label, "message_index": mi})
    elif raw_start is not None or end_ts is not None:
        # No per-event timestamps (current recorder): one run-wide segment
        # spanning header timestamp -> header + footer.duration_s, colored by
        # the dominant activity so slow runs are still visible as one bar.
        kinds = [(_timeline_kind(m)[0], m) for _, m in msg_records]
        if any(k == "tool" for k, _ in kinds):
            kind, label = "tool", "run (tools)"
        elif any(k == "llm" for k, _ in kinds):
            kind, label = "llm", "run"
        else:
            kind, label = "other", "run"
        start = raw_start if raw_start is not None else 0.0
        duration_ms = int(round(((end_ts - start) * 1000.0)
                                if (end_ts is not None and end_ts >= start)
                                else 0))
        segs.append({"start_ts": round(start, 3), "duration_ms": duration_ms,
                     "kind": kind, "label": label, "message_index": 0})

    # Cap: merge adjacent same-kind segments (keep first anchor) until under
    # the limit; if merging cannot help (all kinds alternate) keep the first
    # TIMELINE_MAX_SEGMENTS so rendering stays bounded.
    def _merge_once(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for s in items:
            if out and out[-1]["kind"] == s["kind"]:
                out[-1] = dict(out[-1], duration_ms=out[-1]["duration_ms"]
                               + s["duration_ms"], merged=True)
            else:
                out.append(dict(s))
        return out

    while len(segs) > TIMELINE_MAX_SEGMENTS:
        merged = _merge_once(segs)
        if len(merged) >= len(segs):
            segs = segs[:TIMELINE_MAX_SEGMENTS]
            break
        segs = merged
    return segs

"""Insights engine over the experience buffer (backlog item #11).

Read-only analytics over the existing SQLite trace index (+ a bounded lazy
scan of JSONL transcripts for per-tool stats). Answers the two questions the
loop currently lacks: "what should I curate next?" (failure-reason histogram,
per-tool failure rates, score trend) and "what did training break?"
(before/after-promotion deltas from the model registry).

Built in
(``InsightsEngine.generate(days)``).
Cost estimation is deliberately skipped: toaster_loop is loopback-only.

Sections (all tolerant of an empty DB / missing columns — zeros/None, never a
crash):

- ``score_trend``       mean outcome_score bucketed per day, overall and
                        grouped by profile / model (COALESCE(model_version,
                        model_alias)). Trend metric is the labelled
                        first-half vs second-half mean delta
                        (``half_mean_delta``) — simple and honest for the
                        sparse series a personal loop produces.
- ``tool_reliability``  per-tool call/failure counts. The index stores only
                        ``num_tool_calls`` per trace, so per-tool stats are
                        derived lazily from the JSONL message rows with a
                        bounded scan (newest traces first, cap below). Traces
                        whose transcript was pruned are skipped silently.
- ``failure_reasons``   GROUP BY over non-null ``traces.failure_reason``,
                        split infra vs model via core/failure_types.
- ``latency``           p50/p90/max of duration_s and total tokens (nearest
                        rank; columns are nullable so counts accompany them).
- ``activity``          traces/day series, hour-of-day histogram + busiest
                        hour, per-profile table.
- ``promotions``        before/after-promotion score deltas: for every model
                        row with a ``promoted_at`` timestamp (registry v4),
                        compare mean outcome_score in a symmetric window
                        around the promotion. Traces are matched to a model by
                        ``model_alias``/``model_version`` == registry
                        model_id (what ``serve --auto`` records); when no
                        trace matches, the comparison falls back to all
                        traces in the window and is marked
                        ``match="any_model"``. Histories recorded before the
                        ``promoted_at`` column exist report
                        ``available: false`` with a reason (degraded).

No new dependencies: sqlite3 + stdlib only. Pure reads — never mutates state.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.failure_types import classify_tool_result, is_infra_reason, \
    is_model_reason
from .db import TraceRow, TraceStore
from .db import _row_to_trace as _db_row_to_trace

logger = logging.getLogger("toaster_loop.store")

# Insights engine shape — generate(days)
# returning one dict of breakdowns; skipped there: cost estimation.

# Per-trace tool stats are not in the index (only num_tool_calls), so they are
# derived from JSONL message rows with a bounded scan of the newest traces in
# the window. Older/pruned transcripts beyond this cap are not scanned —
# `traces_scanned` reports coverage.
MAX_TOOL_SCAN_TRACES = 500

# Symmetric before/after-promotion windows are capped at this many days on
# each side of the promotion timestamp (also never exceeds available data).
PROMOTION_WINDOW_CAP_DAYS = 30

# Trend direction threshold on half_mean_delta (outcome scores live in
# [-1, 1], so 0.02 is a visible-but-small shift).
TREND_FLAT_EPS = 0.02

# Confidence calibration (pass 4 #40): five fixed conf_mean buckets over the
# mean per-token logprob (negative; closer to 0 = more confident). Boundaries
# are round nats and cover the practical range of local-GGUF averages: below
# -2.0 is very hesitant, above -0.1 is near-deterministic. Traces whose
# conf_mean falls outside [CONF_BUCKET_EDGES[0], ...] land in the outermost
# open bucket.
CONF_BUCKET_EDGES = (-math.inf, -2.0, -1.0, -0.5, -0.1, math.inf)
CONF_BUCKET_LABELS = (
    "very_low (<-2.0)", "low ([-2.0,-1.0))", "mid ([-1.0,-0.5))",
    "high ([-0.5,-0.1))", "very_high (>=-0.1)",
)

# `confidently_wrong` divergence: the highest-confidence bucket scores at
# least this much below the mean score of all other buckets, and both sides
# need CONFIDENTLY_WRONG_MIN_N traces to count (tiny buckets are noise).
CONFIDENTLY_WRONG_GAP = 0.15
CONFIDENTLY_WRONG_MIN_N = 3


# ------------------------------------------------------------------- helpers

def _db_path(home_or_db: Any) -> Path:
    """Accept a home dir, an explicit db path, or a TraceStore."""
    if isinstance(home_or_db, TraceStore):
        return Path(home_or_db.db_path)
    p = Path(home_or_db).expanduser()
    if p.suffix in (".db", ".sqlite", ".sqlite3"):
        return p
    return p / "buffer" / "toaster_loop.db"


def _mean(vals: List[float]) -> Optional[float]:
    return round(sum(vals) / len(vals), 4) if vals else None


def percentile(sorted_vals: List[float], pct: float) -> Optional[float]:
    """Nearest-rank percentile over an already-sorted list (canonical helper).

    Rank is ``ceil(pct/100 * n)`` (classic nearest-rank definition). The old
    ``round()`` used banker's rounding, which disagreed with the documented
    semantics at exact .5 boundaries (e.g. p50 of 15 items rounded down to
    rank 7 instead of 8); integer-scaled ceil can't hit that class of error.

    Public: dashboard/metrics.py imports this so exposition quantiles match
    the insights report exactly. core/agent_runner mirrors the one-line rank
    (core must not import store at module level — lazy-import-only direction)
    and is pinned to it by the parity tests in test_dashboard.py.
    """
    if not sorted_vals:
        return None
    n = len(sorted_vals)
    # math.ceil over the exact integer product: pct*n is an exact int and a
    # correctly-rounded float division can never cross an integer boundary
    # here (non-integer quotients sit >= 0.01 away from one).
    idx = max(0, min(n - 1, math.ceil(pct * n / 100) - 1))
    v = sorted_vals[idx]
    return round(float(v), 4)


def _parse_ts(ts: str) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _direction(delta: Optional[float]) -> str:
    if delta is None or abs(delta) < TREND_FLAT_EPS:
        return "flat"
    return "up" if delta > 0 else "down"


def _half_mean_delta(points: List[Dict[str, Any]]) -> Optional[float]:
    """Mean(second half of scored buckets) - mean(first half), by bucket date.

    Buckets are chronological; unscored buckets (avg_score None) are dropped.
    Returns None when fewer than two scored buckets exist.
    """
    scores = [p["avg_score"] for p in points if p.get("avg_score") is not None]
    if len(scores) < 2:
        return None
    mid = len(scores) // 2
    first, second = scores[:mid] or scores[:1], scores[mid:]
    a, b = _mean(first), _mean(second)
    if a is None or b is None:
        return None
    return round(b - a, 4)


def _daily_points(rows: List[TraceRow]) -> List[Dict[str, Any]]:
    """Per-day {date, traces, avg_score} buckets (chronological)."""
    by_day: Dict[str, List[Optional[float]]] = defaultdict(list)
    for r in rows:
        by_day[r.timestamp[:10]].append(r.outcome_score)
    out = []
    for day in sorted(by_day):
        scored = [s for s in by_day[day] if s is not None]
        out.append({
            "date": day,
            "traces": len(by_day[day]),
            "avg_score": _mean(scored),
        })
    return out


# ------------------------------------------------------------------- sections

def _overview(rows: List[TraceRow]) -> Dict[str, Any]:
    scored = [r.outcome_score for r in rows if r.outcome_score is not None]
    status_counts: Dict[str, int] = dict(
        Counter(r.status for r in rows).items()
    )
    failures = sum(1 for r in rows if r.status == "failure")
    return {
        "traces": len(rows),
        "scored_traces": len(scored),
        "avg_score": _mean(list(scored)),
        "status_counts": {
            s: status_counts.get(s, 0)
            for s in ("success", "failure", "partial", "unknown")
        },
        "failure_rate": round(failures / len(rows), 4) if rows else None,
    }


def _score_trend(rows: List[TraceRow]) -> Dict[str, Any]:
    points = _daily_points(rows)
    delta = _half_mean_delta(points)

    def grouped(key_fn) -> List[Dict[str, Any]]:
        groups: Dict[str, List[TraceRow]] = defaultdict(list)
        for r in rows:
            k = key_fn(r)
            if k:
                groups[k].append(r)
        out = []
        for name in sorted(groups):
            gpts = _daily_points(groups[name])
            d = _half_mean_delta(gpts)
            scored = [r.outcome_score for r in groups[name]
                      if r.outcome_score is not None]
            out.append({
                "name": name,
                "traces": len(groups[name]),
                "avg_score": _mean(scored),
                "half_mean_delta": d,
                "direction": _direction(d),
            })
        return out

    return {
        "bucket": "day",
        "trend_metric": "half_mean_delta (mean of per-day avg score, second "
                        "half of window minus first half)",
        "overall": {
            "points": points,
            "half_mean_delta": delta,
            "direction": _direction(delta),
        },
        "by_profile": grouped(lambda r: r.agent_profile),
        "by_model": grouped(lambda r: r.model_version or r.model_alias),
    }


def _tool_stats_from_jsonl(path: str) -> List[tuple]:
    """[(tool_name, failed_bool)] for one trace transcript; [] on any problem.

    Pairs assistant tool_calls with their tool results by call id (falling
    back to the tool message's own ``name``); classification uses the shared
    core/failure_types classifier so numbers match heuristic_score.
    """
    try:
        names_by_id: Dict[str, str] = {}
        pairs: List[tuple] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("kind") != "message":
                    continue
                m = rec.get("message") or {}
                role = m.get("role")
                if role == "assistant":
                    for tc in m.get("tool_calls") or []:
                        fn = (tc.get("function") or {}).get("name") or "unknown"
                        if tc.get("id"):
                            names_by_id[tc["id"]] = fn
                elif role == "tool":
                    name = m.get("name") or names_by_id.get(
                        m.get("tool_call_id") or "", "unknown")
                    content = m.get("content")
                    if not isinstance(content, str):
                        content = json.dumps(content, default=str)
                    failed, _kind = classify_tool_result(name, content)
                    pairs.append((name, failed))
        return pairs
    except (OSError, ValueError, json.JSONDecodeError):
        return []  # pruned/corrupt transcript: skip this trace silently


def _tool_reliability(rows: List[TraceRow]) -> Dict[str, Any]:
    """Per-tool failure rates via bounded lazy JSONL scan (newest first)."""
    scan = rows[:MAX_TOOL_SCAN_TRACES]  # rows arrive newest-first
    calls: Counter = Counter()
    fails: Counter = Counter()
    scanned = 0
    for r in scan:
        pairs = _tool_stats_from_jsonl(r.jsonl_path)
        if pairs:
            scanned += 1
        for name, failed in pairs:
            calls[name] += 1
            if failed:
                fails[name] += 1
    tools = [
        {
            "tool": name,
            "calls": n,
            "failures": fails.get(name, 0),
            "failure_rate": round(fails.get(name, 0) / n, 4) if n else None,
        }
        for name, n in calls.most_common()
    ]
    return {
        "available": bool(calls),
        "source": "jsonl_message_scan",
        "traces_scanned": scanned,
        "traces_in_window": len(rows),
        "scan_cap": MAX_TOOL_SCAN_TRACES,
        "note": "per-tool counts are not stored on the index row; derived "
                "lazily from transcripts (bounded scan, newest first)",
        "tools": tools,
    }


def _failure_reasons(rows: List[TraceRow],
                     notes: Optional[List[Dict[str, Any]]] = None,
                     profile: Optional[str] = None) -> Dict[str, Any]:
    counter = Counter(r.failure_reason for r in rows if r.failure_reason)
    # Pass 4 #33: annotate each reason with its current remediation status +
    # note (failure_notes table; profile-scoped row wins over the global one).
    # A reason without any row renders as implicit OPEN — same status string,
    # just explicit=False — so readers never special-case "missing".
    def _remediation(reason: str) -> Dict[str, Any]:
        note = None
        for n in notes or []:  # profile-scoped first, then global fallback
            if n.get("reason") != reason:
                continue
            if n.get("profile") == profile and profile:
                note = n
                break
            if n.get("profile") is None and note is None:
                note = n
        return {
            "status": (note or {}).get("status", "open"),
            "note": (note or {}).get("note") or "",
            "ts": (note or {}).get("ts"),
            "explicit": note is not None,
        }

    buckets: Dict[str, List[Dict[str, Any]]] = {"infra": [], "model": [],
                                                "other": []}
    for reason, n in counter.most_common():
        kind = ("infra" if is_infra_reason(reason)
                else "model" if is_model_reason(reason) else "other")
        buckets[kind].append({"reason": reason, "count": n,
                              **_remediation(reason)})
    return {
        "total": sum(counter.values()),
        "infra_total": sum(b["count"] for b in buckets["infra"]),
        "model_total": sum(b["count"] for b in buckets["model"]),
        "by_reason": [
            {"reason": reason, "count": n,
             "kind": ("infra" if is_infra_reason(reason)
                      else "model" if is_model_reason(reason) else "other"),
             **_remediation(reason)}
            for reason, n in counter.most_common()
        ],
        **buckets,
    }


def _percentiles(rows: List[TraceRow]) -> Dict[str, Any]:
    durs = sorted(r.duration_s for r in rows if r.duration_s is not None)
    toks = sorted(
        float((r.tokens_in or 0) + (r.tokens_out or 0))
        for r in rows
        if r.tokens_in is not None or r.tokens_out is not None
    )

    def stats(vals: List[float]) -> Dict[str, Any]:
        return {
            "count": len(vals),
            "p50": percentile(vals, 50),
            "p90": percentile(vals, 90),
            "max": round(float(vals[-1]), 4) if vals else None,
        }

    return {"duration_s": stats(durs), "tokens_total": stats(toks)}


def _activity(rows: List[TraceRow]) -> Dict[str, Any]:
    per_day = [{"date": p["date"], "traces": p["traces"]}
               for p in _daily_points(rows)]
    hours: Counter = Counter(
        r.timestamp[11:13] for r in rows
        if len(r.timestamp) >= 13 and r.timestamp[10:11] == "T"
    )
    busiest = hours.most_common(1)[0][0] if hours else None

    profiles_out = []
    by_profile: Dict[str, List[TraceRow]] = defaultdict(list)
    for r in rows:
        by_profile[r.agent_profile or "(none)"].append(r)
    for name in sorted(by_profile, key=lambda k: (-len(by_profile[k]), k)):
        grp = by_profile[name]
        scored = [r.outcome_score for r in grp if r.outcome_score is not None]
        fails = sum(1 for r in grp if r.status == "failure")
        profiles_out.append({
            "profile": name,
            "traces": len(grp),
            "avg_score": _mean(scored),
            "failure_rate": round(fails / len(grp), 4) if grp else None,
        })

    return {
        "traces_per_day": per_day,
        "hour_histogram": [{"hour": f"{h:02d}", "traces": hours.get(f"{h:02d}", 0)}
                           for h in range(24)],
        "busiest_hour": busiest,
        "profiles": profiles_out,
    }


# ------------------------------------------------- shared promotion windows

def _score_window(all_rows: List[TraceRow], model_id: Optional[str],
                  start: datetime, end: datetime, *,
                  inclusive_end: bool = False):
    """(mean, n, matched_by_alias) over [start, end) (or [start, end]).

    Before-windows use [start, end) and after-windows use [start, end] so a
    trace sitting exactly on the newest history edge (which defines `end`
    itself) still counts. When model_id matches nothing in the window the
    comparison falls back to all traces there (matched_by_alias False).
    """
    def keep(r: TraceRow) -> bool:
        return _in(r.timestamp, start, end, inclusive_end=inclusive_end)

    if model_id is not None:
        sel = [r for r in all_rows
               if keep(r)
               and (r.model_alias == model_id or r.model_version == model_id)]
        if sel:
            return (_mean([r.outcome_score for r in sel
                           if r.outcome_score is not None]),
                    len(sel), True)
    sel = [r for r in all_rows if keep(r)]
    return (_mean([r.outcome_score for r in sel
                   if r.outcome_score is not None]),
            len(sel), False)


def _promotion_window(all_rows: List[TraceRow], oldest: datetime,
                      newest: datetime, pt: datetime,
                      model_id: Optional[str],
                      prev_model_id: Optional[str]) -> Dict[str, Any]:
    """Symmetric before/after window scores around promotion time ``pt``.

    Shared by the ``promotions`` section and the pass-4 #36 ``intent_review``
    so both compute deltas with identical logic. Empty-ish (all None) when no
    symmetric window fits inside the available history.
    """
    out: Dict[str, Any] = {
        "window_days": None,
        "before_avg_score": None, "after_avg_score": None, "delta": None,
        "before_traces": 0, "after_traces": 0, "match": None,
    }
    half = min((pt - oldest).total_seconds(),
               (newest - pt).total_seconds(),
               PROMOTION_WINDOW_CAP_DAYS * 86400.0)
    if half <= 0:
        return out
    lo, hi = pt - timedelta(seconds=half), pt + timedelta(seconds=half)
    after_m, after_n, after_alias = _score_window(
        all_rows, model_id, pt, hi, inclusive_end=True)
    before_m, before_n, before_alias = _score_window(
        all_rows, prev_model_id, lo, pt)
    out.update({
        "window_days": round(half / 86400.0, 2),
        "before_avg_score": before_m,
        "after_avg_score": after_m,
        "delta": (round(after_m - before_m, 4)
                  if before_m is not None and after_m is not None else None),
        "before_traces": before_n,
        "after_traces": after_n,
        "match": ("model_id" if (after_alias or before_alias)
                  else "any_model"),
    })
    return out


def _read_traces(conn: sqlite3.Connection,
                 profile: Optional[str]) -> List[TraceRow]:
    """All trace rows (full history; only the profile filter applies).

    Tolerates a DB without the traces table yet (registry-only home): [] —
    honest-empty, never fabricated.
    """
    tsel, targs = "SELECT * FROM traces", []
    if profile:
        tsel += " WHERE agent_profile = ?"
        targs = [profile]
    try:
        return [_db_row_to_trace(r) for r in conn.execute(tsel, targs)]
    except sqlite3.OperationalError:
        return []


def _promotions(db_path: Path, profile: Optional[str]) -> Dict[str, Any]:
    """Before/after-promotion score deltas (registry models + traces).

    Requires the ``models.promoted_at`` column (registry v4). Older histories
    report available=false; individual rows lacking promoted_at are skipped.
    Trace-to-model matching is model_alias/model_version == model_id (what
    `serve --auto` records); with no matching traces the row falls back to all
    traces in the window and marks match="any_model".
    """
    out: Dict[str, Any] = {"available": False, "reason": None, "deltas": []}
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error:
        out["reason"] = "registry database not readable"
        return out
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(models)")}
        if not cols:
            out["reason"] = "no models registered yet"
            return out
        promoted_at_col = "promoted_at" if "promoted_at" in cols else None
        if promoted_at_col is None:
            # Pre-v4 registry: promotion times were never recorded.
            out["reason"] = ("registry predates the promoted_at column — "
                             "promotion deltas unavailable for old history")
            return out

        promos = conn.execute(
            "SELECT model_id, base_model, parent_adapter, promoted_at "
            "FROM models WHERE promoted_at IS NOT NULL "
            "ORDER BY promoted_at"
        ).fetchall()
        if not promos:
            out["reason"] = "no promotions recorded yet"
            return out

        # Promotion windows use the full history (not the --days window),
        # only honouring the profile filter.
        all_rows = _read_traces(conn, profile)
        all_ts = sorted(filter(None, (_parse_ts(r.timestamp) for r in all_rows)))
        if not all_ts:
            out["reason"] = "no traces to compare"
            return out
        oldest, newest = all_ts[0], all_ts[-1]

        for p in promos:
            entry: Dict[str, Any] = {
                "model_id": p["model_id"],
                "base_model": p["base_model"],
                "previous_model_id": p["parent_adapter"],
                "promoted_at": p["promoted_at"],
            }
            pt = _parse_ts(p["promoted_at"])
            if pt is not None:
                entry.update(_promotion_window(
                    all_rows, oldest, newest, pt,
                    p["model_id"], p["parent_adapter"]))
            else:
                entry.update({"window_days": None, "before_avg_score": None,
                              "after_avg_score": None, "delta": None,
                              "before_traces": 0, "after_traces": 0,
                              "match": None})
            out["deltas"].append(entry)
        out["available"] = True
        return out
    except sqlite3.Error as e:  # never crash insights on registry oddities
        logger.debug("promotion deltas unavailable: %s", e)
        out["reason"] = f"registry read failed: {e}"
        return out
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _archival_watch(db_path: Path, days: int) -> Dict[str, Any]:
    """Archival-eligible models that are still used (pass 4 #29).

    The janitor excludes models whose ``last_used_at`` falls inside the aging
    window; this surfaces the same tension read-only: terminal, unpinned rows
    (the only archivable statuses) with a recent activation stamp. Cheap
    single SELECT over the registry tables; never raises.
    """
    out: Dict[str, Any] = {"available": False, "reason": None,
                           "still_used": []}
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error:
        out["reason"] = "registry database not readable"
        return out
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(models)")}
        if not cols or "last_used_at" not in cols:
            out["reason"] = ("registry predates the last_used_at column — "
                             "activation watch unavailable")
            return out
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(
            timespec="seconds")
        rows = conn.execute(
            "SELECT model_id, status, base_model, last_used_at FROM models "
            "WHERE status IN ('superseded','failed','rolled_back') "
            "AND pinned = 0 AND last_used_at IS NOT NULL "
            "AND last_used_at >= ? ORDER BY last_used_at DESC",
            (cutoff,),
        ).fetchall()
        out["still_used"] = [dict(r) for r in rows]
        out["available"] = True
        return out
    except sqlite3.Error as e:
        logger.debug("archival watch unavailable: %s", e)
        out["reason"] = f"registry read failed: {e}"
        return out
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _in(ts: str, start: datetime, end: datetime, *,
        inclusive_end: bool = False) -> bool:
    t = _parse_ts(ts)
    if t is None:
        return False
    return (start <= t <= end) if inclusive_end else (start <= t < end)


# Idea adapted from PIASystem (MIT) app/core/evolution_trace.py — improvement
# events with typed causality ("we did X expecting Y"). Their version
# fabricated demo entries when empty; this port keeps read paths honest-empty:
# rows only ever come from real ledger entries with a recorded intent.
def _intent_review(db_path: Path, profile: Optional[str]) -> Dict[str, Any]:
    """Recorded intent vs observed score delta, side by side (pass 4 #36).

    For every ledger entry with a non-null ``intent`` whose op is promote or
    rollback, pair the "why" with the same symmetric before/after score window
    the ``promotions`` section computes — anchored at the ledger entry's own
    timestamp, so historical ops keep their original reference point even if
    ``models.promoted_at`` was later overwritten. Honesty constraints: no
    intents -> empty rows with a hint; nothing is ever synthesised, and old
    entries without intent stay invisible here (they were never recorded).
    """
    out: Dict[str, Any] = {"available": False, "reason": None, "rows": [],
                           "hint": None}
    if not Path(db_path).exists():
        # Fresh home: no DB has ever existed, so no intent was ever recorded.
        out["available"] = True
        out["hint"] = "no recorded intents"
        return out
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error:
        out["reason"] = "registry database not readable"
        return out
    try:
        if not {r[1] for r in conn.execute("PRAGMA table_info(ledger)")}:
            # Honest-empty: no ledger yet means no recorded intents ever.
            out["available"] = True
            out["hint"] = "no recorded intents"
            return out
        intents = conn.execute(
            "SELECT seq, ts, actor, op, target_id, intent, before_json, "
            "after_json FROM ledger "
            "WHERE intent IS NOT NULL AND op IN ('promote','rollback') "
            "ORDER BY seq DESC"
        ).fetchall()
        if not intents:
            out["available"] = True
            out["hint"] = "no recorded intents"
            return out

        all_rows = _read_traces(conn, profile)
        all_ts = sorted(filter(None, (_parse_ts(r.timestamp)
                                      for r in all_rows)))
        oldest = all_ts[0] if all_ts else None
        newest = all_ts[-1] if all_ts else None

        for e in intents:
            row: Dict[str, Any] = {
                "seq": e["seq"], "ts": e["ts"], "op": e["op"],
                "target": e["target_id"], "intent": e["intent"],
                "before_score": None, "after_score": None, "delta": None,
                "window_days": None, "before_traces": 0, "after_traces": 0,
                "match": None,
            }
            pt = _parse_ts(e["ts"])
            if pt is not None and oldest is not None and newest is not None:
                try:
                    before = json.loads(e["before_json"]) \
                        if e["before_json"] else {}
                    after = json.loads(e["after_json"]) \
                        if e["after_json"] else {}
                except (ValueError, TypeError):
                    before, after = {}, {}
                # The model whose scores should improve after the op: for a
                # promote that's the target; for a rollback the restored one.
                improving = (e["target_id"] if e["op"] == "promote"
                             else after.get("restored"))
                prev_id = before.get("previous_active") if e["op"] == "promote" \
                    else e["target_id"]
                win = _promotion_window(all_rows, oldest, newest, pt,
                                        improving, prev_id)
                row.update({
                    "before_score": win["before_avg_score"],
                    "after_score": win["after_avg_score"],
                    "delta": win["delta"],
                    "window_days": win["window_days"],
                    "before_traces": win["before_traces"],
                    "after_traces": win["after_traces"],
                    "match": win["match"],
                })
            out["rows"].append(row)
        out["available"] = True
        return out
    except sqlite3.Error as e:  # never crash insights on ledger oddities
        logger.debug("intent review unavailable: %s", e)
        out["reason"] = f"ledger read failed: {e}"
        return out
    finally:
        try:
            conn.close()
        except Exception:
            pass


# Idea adapted from PIASystem (MIT) app/core/calibration_engine.py — bin
# predicted confidence against actual correctness per domain (Brier/ECE);
# ported here as a cheap bucketed reliability view over the trace index.
def _calibration(rows: List[TraceRow]) -> Dict[str, Any]:
    """Bucket finished traces by conf_mean, mean quality score per bucket.

    Quality score is the same definition ``_promotions`` / intent_review use:
    ``traces.outcome_score`` (the heuristic baseline kept in sync with the
    latest explicit *user* feedback by TraceStore.add_feedback; judge rows are
    excluded there by design, so both views stay consistent).

    Honest-empty: no traces carry logprob data -> available:true, empty rows +
    hint. Only aggregate floats are ever read — per-token arrays were never
    stored.

    ``confidently_wrong``: True when the highest-confidence bucket's mean
    score trails every other bucket's mean by more than CONFIDENTLY_WRONG_GAP
    (each side needing at least CONFIDENTLY_WRONG_MIN_N traces) — the "this
    adapter is confidently wrong" detector from the pass-4 idea. Record +
    report only; nothing here changes retry or promotion behavior.
    """
    out: Dict[str, Any] = {"available": True, "rows": [], "hint": None,
                           "confidently_wrong": False,
                           "bucket_edges": [None if math.isinf(e) else e
                                            for e in CONF_BUCKET_EDGES],
                           "note": "quality score = traces.outcome_score "
                                   "(user feedback synced; judge excluded), "
                                   "same as promotions/intent_review"}
    scored = [(float(r.conf_mean), r) for r in rows
              if r.conf_mean is not None]
    if not scored:
        out["hint"] = "no traces with confidence data yet"
        return out

    buckets: List[List[TraceRow]] = [[] for _ in CONF_BUCKET_LABELS]
    bucket_confs: List[List[float]] = [[] for _ in CONF_BUCKET_LABELS]
    for cm, r in scored:
        for i in range(len(CONF_BUCKET_LABELS)):
            if CONF_BUCKET_EDGES[i] <= cm < CONF_BUCKET_EDGES[i + 1]:
                buckets[i].append(r)
                bucket_confs[i].append(cm)
                break

    means: List[Optional[float]] = []
    for label, grp, confs in zip(CONF_BUCKET_LABELS, buckets, bucket_confs,
                                 strict=True):
        scores = [r.outcome_score for r in grp if r.outcome_score is not None]
        m = _mean(scores)
        means.append(m)
        out["rows"].append({
            "bucket_label": label,
            "n": len(grp),
            "mean_conf": _mean(confs),
            "mean_score": m,
        })

    # Divergence flag: top bucket vs the pooled mean of all other buckets.
    top_i = len(CONF_BUCKET_LABELS) - 1
    others = [r for i, grp in enumerate(buckets) if i != top_i for r in grp]
    top_scores = [r.outcome_score for r in buckets[top_i]
                  if r.outcome_score is not None]
    other_scores = [r.outcome_score for r in others
                    if r.outcome_score is not None]
    if (len(top_scores) >= CONFIDENTLY_WRONG_MIN_N
            and len(other_scores) >= CONFIDENTLY_WRONG_MIN_N):
        top_m = sum(top_scores) / len(top_scores)
        other_m = sum(other_scores) / len(other_scores)
        out["confidently_wrong"] = bool(
            other_m - top_m > CONFIDENTLY_WRONG_GAP)
    return out


def intent_review(home_or_db: Any,
                  profile: Optional[str] = None) -> Dict[str, Any]:
    """Public entry to the ``intent_review`` section (CLI history --intents).

    Same honest-empty contract as the report section: rows only from real
    ledger entries with recorded intents; never any synthetic data.
    """
    return _intent_review(_db_path(home_or_db), profile or None)


# --------------------------------------------------------------------- main

def generate_insights(home_or_db: Any, days: int = 30,
                      profile: Optional[str] = None) -> Dict[str, Any]:
    """Compute the insights report over the buffer DB (backlog item #11).

    Args:
        home_or_db: TOASTER_LOOP_HOME dir, explicit toaster_loop.db path, or TraceStore.
        days: look-back window for trace sections (promotion windows use the
              full history, capped at PROMOTION_WINDOW_CAP_DAYS per side).
        profile: optional agent_profile filter applied to every section.

    Returns a JSON-serialisable dict; never raises on empty/odd data.
    """
    days = max(1, int(days))
    db_path = _db_path(home_or_db)
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(
        timespec="seconds")

    rows: List[TraceRow] = []
    failure_notes: List[Dict[str, Any]] = []
    if db_path.exists():
        store = TraceStore(db_path)
        try:
            rows = store.list_traces(since=cutoff, agent_profile=profile or None)
            # Pass 4 #33: remediation lifecycle annotations for the failures
            # section (failure_notes table; empty on a DB without any notes).
            failure_notes = store.list_failure_notes()
        finally:
            store.close()
    rows.sort(key=lambda r: r.timestamp, reverse=True)  # newest first

    report: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "days": days,
        "profile_filter": profile or None,
        "empty": not rows,
        "overview": _overview(rows),
        "score_trend": _score_trend(rows),
        "tool_reliability": _tool_reliability(rows),
        "failure_reasons": _failure_reasons(rows, failure_notes,
                                            profile or None),
        "latency": _percentiles(rows),
        "activity": _activity(rows),
        "promotions": _promotions(db_path, profile or None),
        # Pass 4 #29: archival-eligible terminal models with a recent
        # activation (the janitor will leave these alone).
        "archival_watch": _archival_watch(db_path, days),
        # Pass 4 #36: recorded intent vs observed delta (honest-empty).
        "intent_review": _intent_review(db_path, profile or None),
        # Pass 4 #40: confidence calibration from logprobs (record+report only).
        "calibration": _calibration(rows),
    }
    return report

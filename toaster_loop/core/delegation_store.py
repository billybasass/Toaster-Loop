"""Durable delegation registry + inbox state (D-17, audit §4 Phase 1).

SQLite store for delegation records and the parent-thread inbox queue. The
delegation lifecycle is a mutable state machine (queued -> running -> terminal,
heartbeat stamps, claim transitions, crash-recovery sweeps), which fits SQLite
far better than an append-only JSONL log — and toaster_loop already runs three SQLite
surfaces (``store/db.py`` traces index, registry DB, per-project index DBs)
with WAL + busy_timeout, so the conventions and fixtures exist. A separate
file from the trace buffer keeps training data decoupled from a runtime work
queue, and lets `toaster-loop doctor` treat this as disposable state (deleting
~/.toaster-loop is still a full reset per D-10).

Conventions mirror ``store/db.py`` verbatim: module ``SCHEMA_VERSION``,
idempotent ``CREATE TABLE IF NOT EXISTS`` DDL, WAL pragma, single-writer
assumption with busy timeout, tolerant reads. One connection per call (no
global handle, matching the registry style); the DB path resolves per call
via ``resolve_toaster_loop_home`` so tests that monkeypatch ``TOASTER_LOOP_HOME`` are
naturally isolated.

The table IS the inbox: an "inbox message" is a settled delegation with
``inbox_state='pending'`` for a given thread. Batching multiple results into
one wake turn falls out of ``take_pending_inbox(thread)`` (Phase 2 consumes;
Phase 1 only populates).
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

SCHEMA_VERSION = 1

#: Every status the CHECK constraint accepts. Terminal statuses (no outgoing
#: transitions except reconcile's crash sweep): completed, failed, cancelled,
#: wedged, interrupted — plus timed_out which only occurs on the legacy
#: wait=true path and reconciles to a real terminal state when the abandoned
#: child settles (D-11 semantics retained).
STATUS_VALUES = ("queued", "running", "completed", "failed", "timed_out",
                 "cancelling", "cancelled", "wedged", "interrupted")

TERMINAL_STATUSES = frozenset(
    {"completed", "failed", "cancelled", "wedged", "interrupted"})

#: Statuses a crash can interrupt (never requeued — side effects may already
#: have happened mid-child; reconcile marks them `interrupted`).
NON_TERMINAL_STATUSES = ("queued", "running", "cancelling")

_SCHEMA = f"""
PRAGMA schema_version = {SCHEMA_VERSION};

CREATE TABLE IF NOT EXISTS delegations (
    delegation_id   TEXT PRIMARY KEY,
    status          TEXT NOT NULL CHECK (status IN
      ('queued','running','completed','failed','timed_out',
       'cancelling','cancelled','wedged','interrupted')),
    agent           TEXT NOT NULL,
    task_preview    TEXT NOT NULL,            -- 300 chars, as today
    project         TEXT,
    thread          TEXT,                     -- parent chat thread or NULL
    parent_run_id   TEXT,
    depth           INTEGER NOT NULL DEFAULT 1,
    created_ts      TEXT NOT NULL,
    started_ts      TEXT,
    settled_ts      TEXT,
    last_activity_ts TEXT,
    liveness_timeout REAL NOT NULL,           -- resolved at dispatch time
    result          TEXT,                     -- capped MAX_RESULT_CHARS
    error           TEXT,
    child_trace_id  TEXT,
    plan_task       TEXT,
    wait_mode       INTEGER NOT NULL DEFAULT 0, -- 1 = legacy sync caller
    inbox_state     TEXT NOT NULL DEFAULT 'none' CHECK (inbox_state IN
      ('none','pending','delivered')),
    delivered_run_id TEXT                     -- wake run that consumed it
);

CREATE INDEX IF NOT EXISTS idx_deleg_thread ON delegations(thread, inbox_state);
CREATE INDEX IF NOT EXISTS idx_deleg_status ON delegations(status);
"""

#: Whitelisted writable columns for `update_delegation` (everything except the
#: primary key). Anything else raises — a typo'd field must never silently
#: no-op a status transition.
_UPDATABLE = frozenset({
    "status", "agent", "task_preview", "project", "thread", "parent_run_id",
    "depth", "created_ts", "started_ts", "settled_ts", "last_activity_ts",
    "liveness_timeout", "result", "error", "child_trace_id", "plan_task",
    "wait_mode", "inbox_state", "delivered_run_id",
})

_write_lock = threading.Lock()


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def db_path(cfg: Any = None) -> Path:
    """Path of the delegation DB (not created here)."""
    from ..store.paths import resolve_toaster_loop_home

    return resolve_toaster_loop_home(cfg) / "delegations" / "delegations.db"


def open_db(cfg: Any = None) -> sqlite3.Connection:
    """Open (creating/mkdir'ing as needed) the delegation DB with WAL +
    idempotent DDL. ``autocommit`` isolation so callers control transactions
    explicitly (claim/take use BEGIN IMMEDIATE)."""
    p = db_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p), timeout=10.0)
    conn.isolation_level = None
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.executescript(_SCHEMA)
    return conn


def _row_to_dict(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    """Tolerant read: unknown/NULL columns pass through untouched."""
    return dict(row) if row is not None else None


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def insert_delegation(record: Dict[str, Any], cfg: Any = None) -> None:
    """Insert one delegation row. `record` carries the store column names;
    missing optional columns default (status required)."""
    cols = ["delegation_id", "status"] + [c for c in _UPDATABLE if c in record]
    vals = [record.get(c) for c in cols]
    sql = (f"INSERT OR REPLACE INTO delegations ({', '.join(cols)}) "
           f"VALUES ({', '.join('?' for _ in cols)})")
    conn = open_db(cfg)
    try:
        with _write_lock:
            conn.execute(sql, vals)
    finally:
        conn.close()


def update_delegation(delegation_id: str, cfg: Any = None, *,
                      expect_status: Optional[str] = None,
                      **fields: Any) -> bool:
    """The single write path for status/heartbeat/settle transitions.
    Whitelisted columns only. When ``expect_status`` is given the update is
    guarded (``AND status = ?``) and returns whether a row was actually
    transitioned — the atomic CAS used by claim/cancel flows."""
    unknown = set(fields) - _UPDATABLE
    if unknown:
        raise ValueError(f"unknown delegation column(s): {sorted(unknown)}")
    if not fields:
        return False
    sets = ", ".join(f"{k} = ?" for k in fields)
    vals: List[Any] = list(fields.values())
    sql = f"UPDATE delegations SET {sets} WHERE delegation_id = ?"
    vals.append(delegation_id)
    if expect_status is not None:
        sql += " AND status = ?"
        vals.append(expect_status)
    conn = open_db(cfg)
    try:
        with _write_lock:
            cur = conn.execute(sql, vals)
            return cur.rowcount > 0
    finally:
        conn.close()


def get_delegation(delegation_id: str, cfg: Any = None
                   ) -> Optional[Dict[str, Any]]:
    conn = open_db(cfg)
    try:
        row = conn.execute(
            "SELECT * FROM delegations WHERE delegation_id = ?",
            (delegation_id,)).fetchone()
        return _row_to_dict(row)
    finally:
        conn.close()


def list_delegations(statuses: Optional[Sequence[str]] = None,
                     thread: Optional[str] = None,
                     project: Optional[str] = None,
                     limit: int = 100, cfg: Any = None
                     ) -> List[Dict[str, Any]]:
    where: List[str] = []
    vals: List[Any] = []
    if statuses:
        where.append(f"status IN ({', '.join('?' for _ in statuses)})")
        vals.extend(statuses)
    if thread is not None:
        where.append("thread = ?")
        vals.append(thread)
    if project is not None:
        where.append("project = ?")
        vals.append(project)
    sql = "SELECT * FROM delegations"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY created_ts LIMIT ?"
    vals.append(int(limit))
    conn = open_db(cfg)
    try:
        return [_row_to_dict(r) for r in conn.execute(sql, vals)]  # type: ignore[misc]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Queue mechanics (dispatcher's FIFO claim — no separate queue object)
# ---------------------------------------------------------------------------

def claim_next_queued(cfg: Any = None) -> Optional[Dict[str, Any]]:
    """Atomically transition the oldest ``queued`` row to ``running`` and
    return it, or None when the queue is empty. SQLite<3.35-safe: SELECT then
    guarded UPDATE with a rowcount check, retried when another claimer won the
    race (SELECT-then-UPDATE double-claim resolves on the status guard)."""
    now = _utcnow()
    conn = open_db(cfg)
    try:
        for _ in range(16):  # bounded retry; contention is tiny at this scale
            with _write_lock:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    row = conn.execute(
                        "SELECT delegation_id FROM delegations "
                        "WHERE status='queued' ORDER BY created_ts LIMIT 1"
                    ).fetchone()
                    if row is None:
                        conn.execute("COMMIT")
                        return None
                    did = row[0]
                    cur = conn.execute(
                        "UPDATE delegations SET status='running', "
                        "started_ts=?, last_activity_ts=? "
                        "WHERE delegation_id=? AND status='queued'",
                        (now, now, did))
                    if cur.rowcount != 1:
                        conn.execute("ROLLBACK")
                        continue  # lost the race; try the next row
                    out = conn.execute(
                        "SELECT * FROM delegations WHERE delegation_id = ?",
                        (did,)).fetchone()
                    conn.execute("COMMIT")
                    return _row_to_dict(out)
                except Exception:
                    try:
                        conn.execute("ROLLBACK")
                    except Exception:
                        pass
                    raise
    finally:
        conn.close()


def count_queued_before(created_ts: str, cfg: Any = None) -> int:
    """How many queued delegations were enqueued strictly before `created_ts`
    (check_delegation's ``queued_ahead`` field)."""
    conn = open_db(cfg)
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM delegations "
            "WHERE status='queued' AND created_ts < ?",
            (created_ts,)).fetchone()
        return int(row[0]) if row else 0
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Inbox (the table doubles as the per-thread inbox — D-17)
# ---------------------------------------------------------------------------

def pending_inbox_threads(cfg: Any = None) -> List[str]:
    """Distinct threads with undelivered settled results."""
    conn = open_db(cfg)
    try:
        rows = conn.execute(
            "SELECT DISTINCT thread FROM delegations "
            "WHERE inbox_state='pending' AND thread IS NOT NULL").fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()


def pending_inbox_counts(cfg: Any = None) -> Dict[str, int]:
    """thread -> number of undelivered entries (Phase 3: the thread-list
    badge and the dashboard-restart "N results waiting" affordance; one
    indexed GROUP BY over idx_deleg_thread)."""
    conn = open_db(cfg)
    try:
        rows = conn.execute(
            "SELECT thread, COUNT(*) FROM delegations "
            "WHERE inbox_state='pending' AND thread IS NOT NULL "
            "GROUP BY thread").fetchall()
        return {r[0]: int(r[1]) for r in rows}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Janitor (D-17 Phase 3): prune aged / over-cap terminal rows
# ---------------------------------------------------------------------------

def _cutoff_iso(days: int) -> str:
    return (datetime.now(timezone.utc)
            - timedelta(days=max(0, int(days)))).isoformat(timespec="seconds")


def prune_terminal_rows(cfg: Any = None, *, days: int = 30,
                        max_rows: int = 1000) -> Dict[str, int]:
    """Delete old / surplus TERMINAL delegation rows (Phase 3 janitor).

    Mirrors the registry janitor's conservative posture (age window + hard
    cap, terminal-only): ``delegations.db`` is a runtime work queue, not the
    history of record — traces stay in the buffer store forever.

    Hard rules:
      * only ``TERMINAL_STATUSES`` rows are ever deleted — never
        queued/running/cancelling, and never ``timed_out`` (it can still
        reconcile to a real terminal state when an abandoned child settles);
      * rows with ``inbox_state='pending'`` are NEVER deleted — an
        undelivered result outranks any retention window (the parent must
        still learn its child finished/died);
      * rows without a parseable ``settled_ts`` are conservatively kept from
        the age rule (the row cap can still evict them, oldest-first).

    Age rule: terminal + not-pending + settled before ``days`` ago.
    Cap rule: keep only the newest ``max_rows`` prunable terminal rows
    (oldest beyond the cap go). Returns {deleted_aged, deleted_capped,
    deleted}. Never raises on a missing DB — opening creates it."""
    q = ", ".join("?" for _ in TERMINAL_STATUSES)
    statuses = tuple(sorted(TERMINAL_STATUSES))
    cutoff = _cutoff_iso(days)
    keep = max(1, int(max_rows))
    conn = open_db(cfg)
    aged = capped = 0
    try:
        with _write_lock:
            conn.execute("BEGIN IMMEDIATE")
            try:
                cur = conn.execute(
                    f"DELETE FROM delegations WHERE status IN ({q}) "
                    "AND inbox_state != 'pending' "
                    "AND settled_ts IS NOT NULL AND settled_ts < ?",
                    statuses + (cutoff,))
                aged = max(0, cur.rowcount)
                # LIMIT -1 OFFSET n == "everything after the newest n".
                cur = conn.execute(
                    "DELETE FROM delegations WHERE delegation_id IN ("
                    "  SELECT delegation_id FROM delegations"
                    f"  WHERE status IN ({q}) AND inbox_state != 'pending'"
                    "  ORDER BY COALESCE(settled_ts, created_ts) DESC"
                    "  LIMIT -1 OFFSET ?)",
                    statuses + (keep,))
                capped = max(0, cur.rowcount)
                conn.execute("COMMIT")
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise
    finally:
        conn.close()
    return {"deleted_aged": aged, "deleted_capped": capped,
            "deleted": aged + capped}


def take_pending_inbox(thread: str, cfg: Any = None) -> List[Dict[str, Any]]:
    """Atomically flip a thread's pending entries to ``delivered`` and return
    them oldest-first. Exactly one caller wins each entry (BEGIN IMMEDIATE +
    status guard), so two wake consumers never double-deliver. Setting
    ``delivered_run_id`` happens afterwards via ``update_delegation`` — the
    wake run is started by the same caller, so the two-step is fine."""
    conn = open_db(cfg)
    try:
        with _write_lock:
            conn.execute("BEGIN IMMEDIATE")
            try:
                rows = conn.execute(
                    "SELECT * FROM delegations WHERE thread=? AND "
                    "inbox_state='pending' ORDER BY settled_ts",
                    (thread,)).fetchall()
                ids = [r["delegation_id"] for r in rows]
                if ids:
                    q = ", ".join("?" for _ in ids)
                    conn.execute(
                        f"UPDATE delegations SET inbox_state='delivered' "
                        f"WHERE delegation_id IN ({q}) "
                        f"AND inbox_state='pending'", ids)
                conn.execute("COMMIT")
                return [_row_to_dict(r) for r in rows]  # type: ignore[misc]
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise
    finally:
        conn.close()


def mark_delivered_run(delegation_ids: Iterable[str], run_id: str,
                       cfg: Any = None) -> None:
    """Stamp the wake run that consumed each delivered entry (provenance)."""
    ids = list(delegation_ids)
    if not ids:
        return
    conn = open_db(cfg)
    try:
        with _write_lock:
            for did in ids:
                conn.execute(
                    "UPDATE delegations SET delivered_run_id=? "
                    "WHERE delegation_id=?", (run_id, did))
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Crash recovery
# ---------------------------------------------------------------------------

def reconcile_on_startup(cfg: Any = None) -> int:
    """Mark every non-terminal row ``interrupted`` after a process restart.

    Side effects may already have happened mid-child, so rows are NEVER
    requeued — an honest ``interrupted`` beats silent resurrection (D-17).
    Rows with a parent thread also become inbox-pending so the parent learns
    its child died instead of waiting forever. Returns the number of swept
    rows; terminal rows are untouched."""
    now = _utcnow()
    q = ", ".join("?" for _ in NON_TERMINAL_STATUSES)
    conn = open_db(cfg)
    try:
        with _write_lock:
            conn.execute("BEGIN IMMEDIATE")
            try:
                rows = conn.execute(
                    f"SELECT delegation_id, thread FROM delegations "
                    f"WHERE status IN ({q})",
                    NON_TERMINAL_STATUSES).fetchall()
                for did, thread in rows:
                    conn.execute(
                        "UPDATE delegations SET status='interrupted', "
                        "settled_ts=?, error=?, "
                        "inbox_state=CASE WHEN thread IS NOT NULL "
                        "THEN 'pending' ELSE inbox_state END "
                        "WHERE delegation_id=?",
                        (now,
                         "toaster_loop process restarted while the child was in "
                         "flight",
                         did))
                conn.execute("COMMIT")
                return len(rows)
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise
    finally:
        conn.close()

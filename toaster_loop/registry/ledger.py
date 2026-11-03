# Adapted from PIASystem (MIT) app/core/intelligence_ledger.py
"""Append-only, hash-chained operation ledger over registry mutations.

Every meaningful state change (promote / rollback / archive / restore / pin /
set_losses / job transitions) appends one entry here so the sequence that got
the registry to its current state stays reconstructable. Entries chain via
SHA-256:

    entry      = {"seq", "ts", "actor", "op", "target_id",
                  "before_json", "after_json", "intent"}  # all fields but `hash`
    canonical  = json.dumps(entry, sort_keys=True, separators=(",", ":"))
    hash       = sha256((prev_hash + canonical).encode("utf-8")).hexdigest()

The first entry (seq 1) chains from the literal string ``GENESIS_HASH``.
verify_ledger() re-walks the chain from seq 1 and reports the first mismatch.

Transactionality: record() NEVER commits — it inserts inside the caller's
transaction so a failed ledger insert rolls back the mutation it records.
That is a deliberate choice over swallowing ledger errors: history that
silently goes missing is worse than a refused mutation.

Documented exception: playbook lifecycle flips (core/playbooks.py
PlaybookStore.set_status) log a warning and continue when record() fails.
Playbook promote/retire is an operator convenience whose local row state is
the source of truth for prompt injection; refusing the flip on a ledger hiccup
would wedge the feature with no user-visible benefit. Registry promotions —
the mutations this ledger exists to protect — keep the strict all-or-nothing
contract.

Append-only guardrail: production code paths must never UPDATE or DELETE
ledger rows; only tests tamper deliberately (to prove verification catches it).

Pass-4 #36 hook: the nullable ``intent`` column stores the "why" of an op
(one-liner + expected effect) and participates in the hash like any other
entry field, so intent edits would break the chain too.

Stdlib only (hashlib, json, sqlite3).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

GENESIS_HASH = "genesis"  # prev_hash of the seq-1 entry (documented convention)

# Pass-4 #36: intents are one-liners; cap keeps ledger entries small. Longer
# text is truncated with a trailing ellipsis *before* hashing, so the chain
# always covers exactly what was stored.
INTENT_MAX_CHARS = 200


def normalize_intent(intent: Optional[str]) -> Optional[str]:
    """Canonical form for a recorded intent: collapsed whitespace, capped.

    Empty/whitespace-only collapses to None (an absent intent, not ''), so
    read paths can trust ``intent IS NOT NULL`` to mean "someone wrote a why".
    """
    if intent is None:
        return None
    s = " ".join(str(intent).split())
    if not s:
        return None
    if len(s) > INTENT_MAX_CHARS:
        return s[:INTENT_MAX_CHARS - 1].rstrip() + "\u2026"
    return s

LEDGER_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS ledger (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT    NOT NULL,
    actor       TEXT    NOT NULL,
    op          TEXT    NOT NULL,
    target_id   TEXT,
    before_json TEXT,
    after_json  TEXT,
    hash        TEXT    NOT NULL,
    prev_hash   TEXT    NOT NULL,
    intent      TEXT                       -- pass-4 #36 hook: why the op ran
);
"""


def _dump(value: Any) -> Optional[str]:
    """Canonical JSON for a before/after payload; None stays NULL."""
    if value is None:
        return None
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def _canonical(entry: Dict[str, Any]) -> str:
    return json.dumps(entry, sort_keys=True, separators=(",", ":"))


def ensure_ledger(conn: sqlite3.Connection) -> None:
    """Create the ledger table if missing (idempotent; safe in a transaction)."""
    conn.execute(LEDGER_TABLE_DDL)


def record(conn: sqlite3.Connection, actor: str, op: str,
           target_id: Optional[str], before: Any = None, after: Any = None,
           intent: Optional[str] = None) -> str:
    """Append one chained entry; return its hash. Does NOT commit.

    Runs inside the caller's transaction so the entry and the mutation it
    describes land atomically. When no transaction is open yet, BEGIN
    IMMEDIATE serializes the prev-hash read + insert against other registry
    handles on the same DB (no chain forks; see PRAGMA busy_timeout in
    ModelRegistry.__init__).
    """
    ensure_ledger(conn)
    if not conn.in_transaction:
        conn.execute("BEGIN IMMEDIATE")
    intent = normalize_intent(intent)
    prev = conn.execute(
        "SELECT hash FROM ledger ORDER BY seq DESC LIMIT 1").fetchone()
    prev_hash = prev[0] if prev else GENESIS_HASH
    seq = conn.execute(
        "SELECT COALESCE(MAX(seq), 0) + 1 FROM ledger").fetchone()[0]
    entry = {
        "seq": seq,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "actor": actor,
        "op": op,
        "target_id": target_id,
        "before_json": _dump(before),
        "after_json": _dump(after),
        "intent": intent,
    }
    h = hashlib.sha256(
        (prev_hash + _canonical(entry)).encode("utf-8")).hexdigest()
    conn.execute(
        "INSERT INTO ledger (seq, ts, actor, op, target_id, before_json,"
        " after_json, hash, prev_hash, intent) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (seq, entry["ts"], actor, op, target_id, entry["before_json"],
         entry["after_json"], h, prev_hash, intent),
    )
    return h


def verify_ledger(conn: sqlite3.Connection):
    """Re-walk the chain from seq 1. Returns (ok, first_bad_seq, total).

    An entry is bad when its stored prev_hash breaks the link or its stored
    hash mismatches sha256(prev_hash + canonical(entry)). A DB without a
    ledger table verifies as an empty, intact chain.
    """
    try:
        rows = conn.execute(
            "SELECT seq, ts, actor, op, target_id, before_json, after_json,"
            " hash, prev_hash, intent FROM ledger ORDER BY seq").fetchall()
    except sqlite3.OperationalError:
        return True, None, 0
    prev = GENESIS_HASH
    for r in rows:
        entry = {"seq": r[0], "ts": r[1], "actor": r[2], "op": r[3],
                 "target_id": r[4], "before_json": r[5], "after_json": r[6],
                 "intent": r[9]}
        if r[8] != prev:  # prev_hash link broken
            return False, r[0], len(rows)
        h = hashlib.sha256(
            (prev + _canonical(entry)).encode("utf-8")).hexdigest()
        if h != r[7]:
            return False, r[0], len(rows)
        prev = r[7]
    return True, None, len(rows)


def recent_ops(conn: sqlite3.Connection, limit: int = 20) -> List[Dict[str, Any]]:
    """Newest-first entries for display (CLI history / dashboard overview)."""
    try:
        rows = conn.execute(
            "SELECT seq, ts, actor, op, target_id, before_json, after_json,"
            " intent FROM ledger ORDER BY seq DESC LIMIT ?",
            (max(1, int(limit)),),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    out = []
    for r in rows:
        out.append({
            "seq": r[0], "ts": r[1], "actor": r[2], "op": r[3],
            "target_id": r[4],
            "before": json.loads(r[5]) if r[5] else None,
            "after": json.loads(r[6]) if r[6] else None,
            "intent": r[7],
        })
    return out

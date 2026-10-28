"""SQLite index for the experience buffer.

Schema v1 (SCHEMA_VERSION below). Two tables:

traces   — one row per agent task execution. Metadata/index only; the full
           conversation lives in the JSONL file pointed to by ``jsonl_path``.
feedback — explicit scores/notes (user/judge/heuristic) attached to a trace.

Design notes for Phase 3 (curation):
- ``outcome_score`` is the heuristic auto-score at record time (nullable);
  aggregated user feedback lands in ``feedback`` and takes precedence there.
- ``curated`` + ``dataset_id`` let the curater claim traces idempotently.
- Single-writer assumption (one sable process at a time) — matches
  ARCHITECTURE.md §5; WAL is enabled anyway for cheap concurrent readers
  (`sable traces list` while an agent run is in flight).
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("sable.store")

# v2 (backlog #2/#7): traces.failure_reason (nullable) — runner-level failure
# mode ("max_iterations" | "backend_error:<reason>" | "guardrail_halt").
# v3 (projects): traces.project (nullable) — project id the run belonged to;
# NULL for runs outside a project and for every pre-projects row.
# v4 (pass 4 #40): traces.conf_mean / traces.conf_p90_negentropy (nullable
# REALs) — cheap per-trace confidence aggregates from llama.cpp logprobs
# (formulas at core/agent_runner._confidence_aggregate). NULL whenever the
# `sampling.logprobs` knob was off or the backend returned no logprobs.
# v5 (pass 4 #33): failure_notes table — remediation lifecycle per
# (reason, profile-scope) on top of the failure_reason taxonomy. New table,
# so the idempotent CREATE TABLE IF NOT EXISTS in _SCHEMA is the whole
# migration (no ALTER needed; old DBs simply gain the table on next open).
# v6 (pass 4 #38): playbooks table — distilled recurring tool-chain recipes
# injected into profile system prompts. New table again, so the idempotent
# CREATE TABLE IF NOT EXISTS in _SCHEMA is the whole migration; CRUD lives
# in core/playbooks.py (PlaybookStore), which re-runs the same DDL
# defensively so it also works against DBs opened before this bump.
# v7 (RFT): rft_prompts table — provenance ledger for rejection-sampling
# (STaR) prompt selection: one row per sampled prompt key, so re-running
# `sable rft` never resamples the same prompt forever. New table again, so
# the idempotent CREATE TABLE IF NOT EXISTS in _SCHEMA is the whole migration.
# v8 (mine_goldens): mined_goldens table — claim/dedup/provenance ledger for
# auto-mined golden tasks (curate/mine_goldens.py). Same new-table-again
# pattern: the idempotent CREATE TABLE IF NOT EXISTS is the whole migration.
# v9 (metrics): gate_events table — durable append-only log of resource-gate
# block episodes (training/scheduler.py records one row per none->blocked
# transition or reason change), so sable_gate_blocks_total is a true
# cumulative counter instead of a gauge over jobs._gate_reason (which the
# scheduler clears on unblock). New table again: idempotent CREATE TABLE IF
# NOT EXISTS is the whole migration.
# v10 (WP12 plans): traces.plan_task (nullable) — plan task id the run was
# started from ("Start task" / plan loop / delegate_task plan_task=); NULL for
# every other run and every pre-WP12 row (tolerant read, no data migration).
SCHEMA_VERSION = 10

STATUS_VALUES = ("success", "failure", "partial", "unknown")

# Idea adapted from PIASystem (MIT) app/core/failure_taxonomy.py —
# RemediationStatus{OPEN, ACKNOWLEDGED, IN_PROGRESS, RESOLVED} +
# update_remediation_status(). Their numpy heatmap layer is deliberately
# skipped (store/insights.py already counts failures per window), and
# first/last-seen are NOT stored either: both are trivially derivable from
# traces at read time (insights computes the windows already), so we prefer
# deriving over storing counters.
FAILURE_NOTE_STATUSES = ("open", "acknowledged", "in_progress", "resolved")

_SCHEMA = f"""
PRAGMA schema_version = {SCHEMA_VERSION};

CREATE TABLE IF NOT EXISTS traces (
    -- ISO-8601 UTC timestamp; agent_profile is the backend name ('sable').
    trace_id        TEXT PRIMARY KEY,
    session_id      TEXT,
    timestamp       TEXT NOT NULL,
    agent_profile   TEXT,
    model_alias     TEXT,
    model_version   TEXT,
    endpoint_label  TEXT,
    task_summary    TEXT,
    status          TEXT NOT NULL CHECK (status IN ('success','failure','partial','unknown')),
    outcome_score   REAL,
    num_tool_calls  INTEGER NOT NULL DEFAULT 0,
    tokens_in       INTEGER,
    tokens_out      INTEGER,
    duration_s      REAL,
    jsonl_path      TEXT NOT NULL,
    schema_version  INTEGER NOT NULL DEFAULT {SCHEMA_VERSION},
    curated         INTEGER NOT NULL DEFAULT 0,
    dataset_id      TEXT,
    -- v2: runner-level failure mode; NULL on success/old rows (backlog #2/#7)
    failure_reason  TEXT,
    -- v3: project id (projects/<id> under SABLE_HOME); NULL = unfiled
    project         TEXT,
    -- v4 (pass 4 #40): confidence aggregates; NULL without logprob data
    conf_mean               REAL,
    conf_p90_negentropy     REAL,
    -- v10 (WP12 plans): plan task id the run was started from; NULL = none
    plan_task       TEXT
);

CREATE INDEX IF NOT EXISTS idx_traces_timestamp ON traces(timestamp);
CREATE INDEX IF NOT EXISTS idx_traces_status ON traces(status);
CREATE INDEX IF NOT EXISTS idx_traces_curated ON traces(curated);

CREATE TABLE IF NOT EXISTS feedback (
    -- source: user | judge | heuristic; score normalized to [-1, 1].
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id    TEXT NOT NULL REFERENCES traces(trace_id) ON DELETE CASCADE,
    source      TEXT NOT NULL DEFAULT 'user',
    score       REAL,
    comment     TEXT,
    timestamp   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_feedback_trace ON feedback(trace_id);

CREATE TABLE IF NOT EXISTS failure_notes (
    -- Remediation state per (reason, profile-scope): one current-status row
    -- per scope. profile NULL = global scope for that reason. ts is the ISO
    -- time of the last status write — curation compares it against trace
    -- timestamps to decide "the bug was fixed after this trace was recorded".
    reason   TEXT NOT NULL,
    profile  TEXT,
    status   TEXT NOT NULL CHECK (status IN ('open','acknowledged','in_progress','resolved')),
    note     TEXT NOT NULL DEFAULT '',
    ts       TEXT NOT NULL
);
-- The logical PRIMARY KEY is (reason, profile), but SQLite never dedups
-- NULLs inside a PK, and profile NULL means "global scope" — so uniqueness
-- lives in this expression index over the coalesced scope instead.
CREATE UNIQUE INDEX IF NOT EXISTS idx_failure_notes_scope
    ON failure_notes(reason, IFNULL(profile, ''));

CREATE TABLE IF NOT EXISTS playbooks (
    -- Pass 4 #38: distilled tool-chain recipes (core/playbooks.py). Status
    -- lifecycle candidate -> active | retired; retired rows are KEPT
    -- (archive-don't-delete ethos, backlog #16). promoted_ts marks the era
    -- boundary the correlational demote check compares outcomes around.
    -- score_history_json: capped list (~20) of gate/assessment entries
    -- with keys ts, n, mean_with, mean_without (nullable), judge_mean
    -- (nullable).
    id                TEXT PRIMARY KEY,
    profile           TEXT,
    chain_json        TEXT NOT NULL,
    text              TEXT NOT NULL,
    status            TEXT NOT NULL CHECK (status IN ('candidate','active','retired')),
    created_ts        TEXT NOT NULL,
    promoted_ts       TEXT,
    score_history_json TEXT NOT NULL DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS idx_playbooks_status ON playbooks(status);

CREATE TABLE IF NOT EXISTS rft_prompts (
    -- RFT (rejection-sampling fine-tuning) provenance ledger: one row per
    -- sampled prompt, keyed by the normalized-prompt hash. Inserted with
    -- INSERT OR IGNORE *before* sampling (the atomic dedup claim), so two
    -- concurrent rft runs never both sample a prompt. dataset_id lands after
    -- emission (NULL = sampled but produced no kept row yet); kept marks
    -- whether the run verified at least one candidate for this prompt.
    prompt_key   TEXT PRIMARY KEY,
    source       TEXT NOT NULL CHECK (source IN ('trace','golden','seed')),
    source_id    TEXT,               -- trace_id when source='trace'
    sampled_at   TEXT NOT NULL,
    dataset_id   TEXT,
    kept         INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_rft_prompts_sampled ON rft_prompts(sampled_at);

CREATE TABLE IF NOT EXISTS mined_goldens (
    -- v8 (curate/mine_goldens.py): claim/dedup/provenance ledger for goldens
    -- auto-mined from 👍 traces. One row per source trace, inserted with
    -- INSERT OR IGNORE *before* distillation (the atomic claim-before-mine,
    -- same BEGIN IMMEDIATE pattern as rft_prompts), so a re-run never mines
    -- the same trace twice and concurrent runs each own a trace at most once.
    -- status: claimed (in flight) | mined (golden persisted) | dropped
    -- (candidate rejected — reason recorded; rows stay so it is not retried
    -- forever). prompt_key is the normalized-prompt hash used to dedup
    -- against previously mined goldens; task_json keeps the exact task dict
    -- that was written to the golden suite (provenance + audit).
    trace_id     TEXT PRIMARY KEY,
    prompt_key   TEXT NOT NULL,
    claimed_at   TEXT NOT NULL,
    status       TEXT NOT NULL CHECK (status IN ('claimed','mined','dropped')),
    reason       TEXT,
    enabled      INTEGER NOT NULL DEFAULT 0,
    task_json    TEXT
);

CREATE INDEX IF NOT EXISTS idx_mined_goldens_prompt ON mined_goldens(prompt_key);

CREATE TABLE IF NOT EXISTS gate_events (
    -- v9 (dashboard/metrics.py): durable append-only log of resource-gate
    -- block EPISODES recorded by training/scheduler.py — one row per
    -- none->blocked transition (the scheduler clears params._gate_reason on
    -- unblock, so the jobs table alone is only a current-state gauge).
    -- sable_gate_blocks_total counts these rows: monotonic across restarts.
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id    TEXT NOT NULL,
    reason    TEXT NOT NULL,
    ts        TEXT NOT NULL
);
"""

#: The playbooks DDL as a standalone constant so core/playbooks.py can
#: re-run it defensively against DB files opened by older builds (same
#: idempotent-CREATE migration family as everything above).
PLAYBOOKS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS playbooks (
    id                TEXT PRIMARY KEY,
    profile           TEXT,
    chain_json        TEXT NOT NULL,
    text              TEXT NOT NULL,
    status            TEXT NOT NULL CHECK (status IN ('candidate','active','retired')),
    created_ts        TEXT NOT NULL,
    promoted_ts       TEXT,
    score_history_json TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_playbooks_status ON playbooks(status);
"""


@dataclass
class TraceRow:
    trace_id: str
    session_id: Optional[str]
    timestamp: str
    agent_profile: Optional[str]
    model_alias: Optional[str]
    model_version: Optional[str]
    endpoint_label: Optional[str]
    task_summary: Optional[str]
    status: str
    outcome_score: Optional[float]
    num_tool_calls: int
    tokens_in: Optional[int]
    tokens_out: Optional[int]
    duration_s: Optional[float]
    jsonl_path: str
    schema_version: int = SCHEMA_VERSION
    curated: bool = False
    dataset_id: Optional[str] = None
    # v2 (backlog #2/#7): "max_iterations" | "backend_error:<reason>" |
    # "guardrail_halt" | None. Readers must tolerate its absence in old DBs.
    failure_reason: Optional[str] = None
    # v3 (projects): project id or None (= Unfiled). Absent in old DBs.
    project: Optional[str] = None
    # v4 (pass 4 #40): per-trace confidence aggregates from logprobs — mean of
    # per-token logprobs, and exp(logprob) of the p90 token. Both None when no
    # logprob data (knob off / unsupported backend). Absent in old DBs.
    conf_mean: Optional[float] = None
    conf_p90_negentropy: Optional[float] = None
    # v10 (WP12 plans): plan task id the run was started from, or None.
    # Absent in old DBs (tolerant read via _row_to_trace defaults).
    plan_task: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["curated"] = bool(d["curated"])
        return d


class TraceStore:
    """Thin SQLite wrapper over the buffer index. Thread-safe for one owner."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            # WAL makes readers cheap, but a second writer (CLI `sable
            # feedback` while the dashboard records, or a curate thread) used
            # to hit "database is locked" instantly. Wait like the registry
            # does instead of erroring.
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.execute("PRAGMA foreign_keys=ON")
            ver = self._conn.execute("PRAGMA schema_version").fetchone()[0]
            if ver < SCHEMA_VERSION:
                # Legacy hook kept for visibility; PRAGMA schema_version is
                # not a reliable migration key (DDL bumps it automatically),
                # so migrations are shape-detected and idempotent below.
                pass
            self._migrate()
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def _migrate(self) -> None:
        """Idempotent in-place migrations, keyed on actual table shape.

        PRAGMA schema_version is written by _SCHEMA and bumped automatically
        by any DDL statement, so it cannot identify the previous release —
        every migration here must detect its target column/table directly.
        Old rows read tolerantly (see _row_to_trace).
        """
        tables = {r[0] for r in self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        if "traces" in tables:
            cols = {r[1] for r in self._conn.execute(
                "PRAGMA table_info(traces)").fetchall()}
            if "failure_reason" not in cols:
                # v1 -> v2 (backlog #2/#7): nullable column; old rows read as
                # NULL and _row_to_trace tolerates either shape.
                self._conn.execute(
                    "ALTER TABLE traces ADD COLUMN failure_reason TEXT")
            if "project" not in cols:
                # v2 -> v3 (projects): nullable column, same tolerant pattern —
                # old rows read as NULL (= Unfiled), no data migration needed.
                self._conn.execute(
                    "ALTER TABLE traces ADD COLUMN project TEXT")
            if "conf_mean" not in cols:
                # v3 -> v4 (pass 4 #40): nullable REAL confidence aggregates;
                # old rows read as NULL (no logprob data ever recorded).
                self._conn.execute(
                    "ALTER TABLE traces ADD COLUMN conf_mean REAL")
            if "conf_p90_negentropy" not in cols:
                self._conn.execute(
                    "ALTER TABLE traces ADD COLUMN conf_p90_negentropy REAL")
            if "plan_task" not in cols:
                # v9 -> v10 (WP12 plans): nullable column, same tolerant
                # pattern — old rows read as NULL (no plan linkage), no data
                # migration needed.
                self._conn.execute(
                    "ALTER TABLE traces ADD COLUMN plan_task TEXT")
            # HIGH #4 (cancelled-run score neutrality): pre-fix recorders
            # stamped cancelled runs with the -0.5 non-completion penalty,
            # dragging down should_auto_train's mean and every score aggregate.
            # The recorder now NULLs cancelled scores at write time; this
            # idempotent UPDATE retro-applies that to existing rows (both the
            # bare and composed runner reason forms). NULL-scored rows are
            # already excluded everywhere by the Phase 2 "never treat missing
            # as zero" rule, so no read-side changes were needed. The old -0.5
            # heuristic feedback rows stay untouched: they never feed any mean
            # (the curater resolves effective scores from traces.outcome_score;
            # only user/judge feedback rows can override it).
            self._conn.execute(
                "UPDATE traces SET outcome_score = NULL "
                "WHERE lower(failure_reason) IN ('cancelled', "
                "'backend_error:cancelled') AND outcome_score IS NOT NULL")
        # NB: PRAGMA schema_version is refreshed by _SCHEMA on every open.

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass

    # ------------------------------------------------------------------ write

    def insert_trace(self, row: TraceRow) -> bool:
        """Insert a trace; return True when the row was written.

        A trace_id collision is SKIPPED (ON CONFLICT DO NOTHING), never
        replaced: INSERT OR REPLACE used to clobber load-bearing curation
        state (curated/dataset_id — see claim_traces_owned) plus any synced
        outcome_score on collision. No caller needs replace-to-update; if one
        ever does, give it an explicit separate method. Returns False when a
        duplicate was skipped.
        """
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO traces (
                       trace_id, session_id, timestamp, agent_profile, model_alias,
                       model_version, endpoint_label, task_summary, status,
                       outcome_score, num_tool_calls, tokens_in, tokens_out,
                       duration_s, jsonl_path, schema_version, curated, dataset_id,
                       failure_reason, project, conf_mean, conf_p90_negentropy,
                       plan_task
                   ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(trace_id) DO NOTHING""",
                 (
                     row.trace_id, row.session_id, row.timestamp, row.agent_profile,
                     row.model_alias, row.model_version, row.endpoint_label,
                     row.task_summary, row.status, row.outcome_score,
                     row.num_tool_calls, row.tokens_in, row.tokens_out,
                     row.duration_s, row.jsonl_path, row.schema_version,
                     int(row.curated), row.dataset_id, row.failure_reason,
                     row.project, row.conf_mean, row.conf_p90_negentropy,
                     row.plan_task,
                 ),
            )
            self._conn.commit()
            inserted = (cur.rowcount or 0) > 0
            if not inserted:
                logger.info("insert_trace: duplicate trace_id %s skipped "
                            "(existing row kept)", row.trace_id)
            return inserted

    def add_feedback(self, trace_id: str, score: Optional[float], comment: Optional[str],
                     source: str = "user", timestamp: str = "") -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO feedback (trace_id, source, score, comment, timestamp) "
                "VALUES (?,?,?,?,?)",
                (trace_id, source, score, comment, timestamp),
            )
            # Keep traces.outcome_score in sync with the latest explicit feedback
            # so curation queries have a single column to sort on. Heuristic
            # rows are excluded here — they're already set at record time.
            # Judge rows (backlog #12) are NOT synced either: outcome_score
            # keeps the heuristic value and the curater applies the
            # user > judge > heuristic precedence from the feedback table, so
            # a judge pass never clobbers the record-time baseline.
            if score is not None and source not in ("heuristic", "judge"):
                self._conn.execute(
                    "UPDATE traces SET outcome_score = ? WHERE trace_id = ?",
                    (score, trace_id),
                )
            self._conn.commit()
            return int(cur.lastrowid or 0)

    # ------------------------------------------------------- curation claiming

    def claim_traces(self, trace_ids: List[str], dataset_id: str) -> int:
        """Idempotently mark traces as curated into ``dataset_id``.

        Only rows that are currently unclaimed flip to claimed, so a second
        call (or a concurrent curate) never steals a trace already owned by
        another dataset version. Returns the number of rows actually claimed.
        Thin count-wrapper over :meth:`claim_traces_owned`.
        """
        return len(self.claim_traces_owned(trace_ids, dataset_id))

    def claim_traces_owned(self, trace_ids: List[str],
                           dataset_id: str) -> List[str]:
        """Atomically claim unclaimed traces; return the IDs this call owns.

        Claim-before-write primitive for the curater (finding #8): every flip
        is a guarded ``UPDATE ... WHERE curated = 0`` with rowcount accounting
        inside ONE BEGIN IMMEDIATE transaction — the same serialization
        pattern as registry.claim_next_queued — so concurrent curates can
        never both own the same trace, and a mid-list failure rolls the whole
        claim back rather than leaving half a dataset's claims committed.
        Returned list follows input order; IDs already owned by another
        dataset are simply absent (the caller reports the shortfall and must
        drop those traces from its outputs).
        """
        claimed: List[str] = []
        if not trace_ids:
            return claimed
        with self._lock:
            try:
                if not self._conn.in_transaction:
                    self._conn.execute("BEGIN IMMEDIATE")
                for tid in trace_ids:
                    cur = self._conn.execute(
                        "UPDATE traces SET curated = 1, dataset_id = ? "
                        "WHERE trace_id = ? AND curated = 0",
                        (dataset_id, tid),
                    )
                    if (cur.rowcount or 0) > 0:
                        claimed.append(tid)
                self._conn.commit()
            except Exception:
                try:
                    self._conn.rollback()
                except sqlite3.Error:  # pragma: no cover - defensive
                    pass
                raise
        return claimed

    def release_claims(self, trace_ids: List[str], dataset_id: str) -> int:
        """Release claims held by ``dataset_id`` for exactly these traces.

        Scoped on ``dataset_id = ?`` so a caller can never hand back a trace
        another dataset owns — the curater uses this to return selected-but-
        ultimately-unused traces (e.g. the chosen side of a DPO pair dropped
        by a concurrent claim) to the buffer. Returns rows released.
        """
        n = 0
        if not trace_ids:
            return n
        with self._lock:
            for tid in trace_ids:
                cur = self._conn.execute(
                    "UPDATE traces SET curated = 0, dataset_id = NULL "
                    "WHERE trace_id = ? AND dataset_id = ?",
                    (tid, dataset_id),
                )
                n += cur.rowcount or 0
            self._conn.commit()
        return n

    def reset_claims(self, dataset_id: str) -> int:
        """Release every claim held by ``dataset_id`` (--rebuild support)."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE traces SET curated = 0, dataset_id = NULL "
                "WHERE dataset_id = ?",
                (dataset_id,),
            )
            self._conn.commit()
            return cur.rowcount or 0

    def claimed_dataset_ids(self) -> List[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT dataset_id FROM traces "
                "WHERE curated = 1 AND dataset_id IS NOT NULL ORDER BY dataset_id"
            ).fetchall()
        return [r[0] for r in rows]

    # ----------------------------------------------------------------- prune

    def prune_traces(self, before_iso: str, keep_curated: bool = True,
                     dry_run: bool = False) -> Dict[str, Any]:
        """Delete traces (index rows + transcript JSONL) older than ``before_iso``.

        Buffer retention primitive (production-readiness): the append-only
        buffer otherwise grows without bound on long-lived installs. Safety
        rules, in order:

        - Curated traces are KEPT unless ``keep_curated=False`` — a trace that
          feeds an existing dataset is training data, not operational noise.
        - Feedback rows disappear with their trace via ON DELETE CASCADE
          (foreign_keys=ON is set at open).
        - Transcript files are unlinked best-effort; failures land in
          ``errors`` and never abort the run (the index row still goes, so a
          retry can finish the job).

        Timestamps are ISO-8601 UTC strings, so plain string comparison is
        correct (same rule as ``list_traces(since=...)``). Returns
        ``{deleted, kept_curated, freed_bytes, errors, dry_run}``.
        """
        cond = "timestamp < ?"
        args: List[Any] = [before_iso]
        if keep_curated:
            cond += " AND curated = 0"
        with self._lock:
            rows = self._conn.execute(
                f"SELECT trace_id, jsonl_path FROM traces WHERE {cond}",
                args,
            ).fetchall()
            kept_curated = 0
            if keep_curated:
                kept_curated = self._conn.execute(
                    "SELECT COUNT(*) FROM traces WHERE timestamp < ? "
                    "AND curated = 1", (before_iso,)
                ).fetchone()[0]
        summary: Dict[str, Any] = {
            "deleted": 0, "kept_curated": int(kept_curated),
            "freed_bytes": 0, "errors": [], "dry_run": bool(dry_run),
        }
        for r in rows:
            tid, jp = r[0], r[1]
            size = 0
            try:
                size = Path(jp).stat().st_size
            except OSError:
                pass  # missing transcript: index-only delete still correct
            if dry_run:
                summary["deleted"] += 1
                summary["freed_bytes"] += size
                continue
            try:
                with self._lock:
                    self._conn.execute(
                        "DELETE FROM traces WHERE trace_id = ?", (tid,))
                    self._conn.commit()
            except sqlite3.Error as e:
                logger.warning("prune_traces: could not delete %s: %s", tid, e)
                summary["errors"].append(f"{tid}: {e}")
                continue
            summary["deleted"] += 1
            summary["freed_bytes"] += size
            try:
                Path(jp).unlink(missing_ok=True)
            except OSError as e:
                logger.warning("prune_traces: transcript unlink failed %s: %s",
                               jp, e)
                summary["errors"].append(f"{jp}: {e}")
        if not dry_run and summary["deleted"]:
            # Reclaim space in the index; best-effort (fails harmlessly when
            # another connection holds a read transaction open).
            try:
                with self._lock:
                    self._conn.execute("VACUUM")
            except sqlite3.Error:  # pragma: no cover - defensive
                pass
        return summary

    # ------------------------------------------------------------------- read

    def get_trace(self, trace_id: str) -> Optional[TraceRow]:
        with self._lock:
            r = self._conn.execute(
                "SELECT * FROM traces WHERE trace_id = ?", (trace_id,)
            ).fetchone()
        return _row_to_trace(r) if r else None

    def latest_trace(self) -> Optional[TraceRow]:
        with self._lock:
            r = self._conn.execute(
                "SELECT * FROM traces ORDER BY timestamp DESC, rowid DESC LIMIT 1"
            ).fetchone()
        return _row_to_trace(r) if r else None

    def list_traces(self, limit: Optional[int] = None, status: Optional[str] = None,
                    since: Optional[str] = None, q: Optional[str] = None,
                    agent_profile: Optional[str] = None,
                    model_alias: Optional[str] = None,
                    curated: Optional[bool] = None,
                    offset: int = 0,
                    project: Optional[str] = None) -> List[TraceRow]:
        conds: List[str] = []
        args: List[Any] = []
        if status:
            conds.append("status = ?")
            args.append(status)
        if curated is not None:
            conds.append("curated = ?")
            args.append(1 if curated else 0)
        if since:
            conds.append("timestamp >= ?")
            args.append(since)
        if agent_profile:
            conds.append("agent_profile = ?")
            args.append(agent_profile)
        if model_alias:
            conds.append("model_alias = ?")
            args.append(model_alias)
        if project:
            conds.append("project = ?")
            args.append(project)
        if q:
            # Literal text search over the task summary (Phase 5a; ranked/FTS
            # ranked search is deferred). LIKE wildcards in q are escaped.
            like = (q.replace("\\", "\\\\").replace("%", "\\%")
                     .replace("_", "\\_"))
            conds.append(r"task_summary LIKE ? ESCAPE '\'")
            args.append(f"%{like}%")
        sql = "SELECT * FROM traces"
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        sql += " ORDER BY timestamp DESC, rowid DESC"
        if limit is not None:
            sql += " LIMIT ?"
            args.append(int(limit))
        if offset:
            # OFFSET requires LIMIT in SQLite; default a full-scan-size cap.
            if limit is None:
                sql += " LIMIT -1"
            sql += " OFFSET ?"
            args.append(int(offset))
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return [_row_to_trace(r) for r in rows]

    def count_traces(self, status: Optional[str] = None,
                     since: Optional[str] = None, q: Optional[str] = None,
                     agent_profile: Optional[str] = None,
                     model_alias: Optional[str] = None,
                     curated: Optional[bool] = None,
                     project: Optional[str] = None) -> int:
        """Total rows matching the list_traces filters (ignores paging).

        Kept in lockstep with list_traces' WHERE clause so pagers can show
        "page X of Y" against the same window (Phase 5d).
        """
        conds: List[str] = []
        args: List[Any] = []
        if status:
            conds.append("status = ?")
            args.append(status)
        if curated is not None:
            conds.append("curated = ?")
            args.append(1 if curated else 0)
        if since:
            conds.append("timestamp >= ?")
            args.append(since)
        if agent_profile:
            conds.append("agent_profile = ?")
            args.append(agent_profile)
        if model_alias:
            conds.append("model_alias = ?")
            args.append(model_alias)
        if project:
            conds.append("project = ?")
            args.append(project)
        if q:
            like = (q.replace("\\", "\\\\").replace("%", "\\%")
                     .replace("_", "\\_"))
            conds.append(r"task_summary LIKE ? ESCAPE '\'")
            args.append(f"%{like}%")
        sql = "SELECT COUNT(*) FROM traces"
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        with self._lock:
            return int(self._conn.execute(sql, args).fetchone()[0])

    def stats_daily(self, days: int = 30) -> List[Dict[str, Any]]:
        """Per-day trace counts / mean score / failure rate (last ``days`` days).

        Day granularity is the first 10 chars of the ISO timestamp; the window
        is a real calendar cutoff (ISO strings compare chronologically), so a
        buffer whose only traces are months old returns an empty series rather
        than showing them under a "last 30 days" heading. Days with no traces
        are omitted; ordering is chronological.
        """
        cutoff_days = max(1, int(days))
        cutoff = (datetime.now(timezone.utc) - timedelta(days=cutoff_days)
                  ).isoformat(timespec="seconds")
        with self._lock:
            rows = self._conn.execute(
                "SELECT substr(timestamp, 1, 10) AS day,"
                " COUNT(*) AS n,"
                " AVG(outcome_score) AS avg_score,"
                " SUM(CASE WHEN status = 'failure' THEN 1 ELSE 0 END) AS fails"
                " FROM traces WHERE timestamp >= ?"
                " GROUP BY day ORDER BY day DESC LIMIT ?",
                (cutoff, cutoff_days),
            ).fetchall()
        series: List[Dict[str, Any]] = []
        for r in reversed(rows):
            n = int(r["n"] or 0)
            series.append({
                "date": r["day"],
                "traces": n,
                "avg_score": r["avg_score"],
                "fail_rate": (int(r["fails"] or 0) / n) if n else None,
            })
        return series

    def query_for_curation(
        self,
        *,
        include_curated: bool = False,
        statuses: Optional[List[str]] = None,
        since: Optional[str] = None,
        model_alias: Optional[str] = None,
        agent_profile: Optional[str] = None,
        project: Optional[str] = None,
    ) -> List[TraceRow]:
        """Candidate traces for the curater (oldest first, deterministic).

        By default already-claimed traces (``curated = 1``) are excluded so
        re-running curation never re-exports a trace owned by another dataset.
        ``outcome_score`` filtering stays in the curater (NULL-score handling:
        Phase 2 note — missing tokens/scores must not be treated as zero).
        """
        q = "SELECT * FROM traces"
        conds: List[str] = []
        args: List[Any] = []
        if not include_curated:
            conds.append("curated = 0")
        if statuses:
            conds.append("status IN (" + ",".join("?" for _ in statuses) + ")")
            args.extend(statuses)
        if since:
            conds.append("timestamp >= ?")
            args.append(since)
        if model_alias:
            conds.append("model_alias = ?")
            args.append(model_alias)
        if agent_profile:
            conds.append("agent_profile = ?")
            args.append(agent_profile)
        if project:
            conds.append("project = ?")
            args.append(project)
        if conds:
            q += " WHERE " + " AND ".join(conds)
        q += " ORDER BY timestamp ASC, rowid ASC"
        with self._lock:
            rows = self._conn.execute(q, args).fetchall()
        return [_row_to_trace(r) for r in rows]

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._conn.execute("SELECT COUNT(*) FROM traces").fetchone()[0]
            by_status = dict(
                self._conn.execute(
                    "SELECT status, COUNT(*) FROM traces GROUP BY status"
                ).fetchall()
            )
            scored = self._conn.execute(
                "SELECT AVG(outcome_score), COUNT(*) FROM traces "
                "WHERE outcome_score IS NOT NULL"
            ).fetchone()
            tok = self._conn.execute(
                "SELECT COALESCE(SUM(tokens_in),0), COALESCE(SUM(tokens_out),0), "
                "COALESCE(SUM(num_tool_calls),0) FROM traces"
            ).fetchone()
            curated = self._conn.execute(
                "SELECT COUNT(*) FROM traces WHERE curated = 1"
            ).fetchone()[0]
            fb = self._conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        return {
            "total_traces": total,
            "by_status": by_status,
            "scored_traces": scored[1],
            "avg_outcome_score": scored[0],
            "curated_traces": curated,
            "feedback_rows": fb,
            "tokens_in_total": tok[0],
            "tokens_out_total": tok[1],
            "tool_calls_total": tok[2],
        }

    def get_feedback(self, trace_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, trace_id, source, score, comment, timestamp "
                "FROM feedback WHERE trace_id = ? ORDER BY id",
                (trace_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------- failure remediation notes
    # Idea adapted from PIASystem (MIT) app/core/failure_taxonomy.py —
    # update_remediation_status() over a fixed status set. Semantics here:
    # * one current-status row per (reason, profile-scope); profile NULL is
    #   the global scope for that reason.
    # * No row == implicit OPEN (reasons nobody has triaged yet). Explicit
    #   `set_failure_note(reason, "open", note="")` deletes the row and puts
    #   the reason back to untracked — reset-to-open with no note is
    #   indistinguishable from never-tracked, by design.
    # * first/last-seen are derived at read time from traces (insights), not
    #   stored.

    def get_failure_note(self, reason: str,
                         profile: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Current note for (reason, profile); falls back to the global row."""
        with self._lock:
            if profile:
                r = self._conn.execute(
                    "SELECT reason, profile, status, note, ts FROM failure_notes "
                    "WHERE reason = ? AND profile = ?",
                    (reason, profile),
                ).fetchone()
                if r:
                    return dict(r)
            r = self._conn.execute(
                "SELECT reason, profile, status, note, ts FROM failure_notes "
                "WHERE reason = ? AND profile IS NULL",
                (reason,),
            ).fetchone()
        return dict(r) if r else None

    def set_failure_note(self, reason: str, status: str, note: str = "",
                         profile: Optional[str] = None) -> None:
        """Upsert the remediation status for one (reason, profile-scope).

        Raises ValueError on an unknown status. ``status == "open"`` with an
        empty note deletes the row instead (back to implicit-open/untracked);
        open-with-note is a legitimate tracked state ("seen, not handling").
        """
        if status not in FAILURE_NOTE_STATUSES:
            raise ValueError(
                f"unknown remediation status {status!r} "
                f"(expected one of {', '.join(FAILURE_NOTE_STATUSES)})")
        if not reason:
            raise ValueError("reason required")
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        profile = profile or None
        with self._lock:
            if status == "open" and not note.strip():
                if profile:
                    self._conn.execute(
                        "DELETE FROM failure_notes WHERE reason = ? AND profile = ?",
                        (reason, profile),
                    )
                else:
                    self._conn.execute(
                        "DELETE FROM failure_notes "
                        "WHERE reason = ? AND profile IS NULL",
                        (reason,),
                    )
            else:
                # Delete-then-insert keeps the expression-index uniqueness
                # simple (no ON CONFLICT target dance for the IFNULL scope).
                if profile:
                    self._conn.execute(
                        "DELETE FROM failure_notes WHERE reason = ? AND profile = ?",
                        (reason, profile),
                    )
                else:
                    self._conn.execute(
                        "DELETE FROM failure_notes "
                        "WHERE reason = ? AND profile IS NULL",
                        (reason,),
                    )
                self._conn.execute(
                    "INSERT INTO failure_notes (reason, profile, status, note, ts) "
                    "VALUES (?,?,?,?,?)",
                    (reason, profile, status, note, ts),
                )
            self._conn.commit()

    def list_failure_notes(self) -> List[Dict[str, Any]]:
        """All current remediation rows (implicit-open reasons absent)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT reason, profile, status, note, ts FROM failure_notes "
                "ORDER BY reason, profile"
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------- RFT prompt ledger
    # Rejection-sampling (STaR) dedup provenance. claim_rft_prompts mirrors
    # claim_traces_owned's pattern: guarded INSERT OR IGNORE with rowcount
    # accounting inside ONE BEGIN IMMEDIATE transaction, so concurrent rft
    # runs each newly-claim a prompt at most once. Rows are never deleted —
    # the ledger is what stops re-runs from resampling forever.

    def claim_rft_prompts(self, items: List[tuple]) -> List[str]:
        """Atomically record freshly-sampled prompts; return newly claimed keys.

        ``items``: list of ``(prompt_key, source, source_id)`` tuples
        (source_id may be None). A key already present in the ledger is NOT
        returned — the caller must skip that prompt this run. Insert order
        follows input order; duplicates within one call claim once.
        """
        claimed: List[str] = []
        if not items:
            return claimed
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._lock:
            try:
                if not self._conn.in_transaction:
                    self._conn.execute("BEGIN IMMEDIATE")
                for key, source, source_id in items:
                    cur = self._conn.execute(
                        "INSERT OR IGNORE INTO rft_prompts"
                        " (prompt_key, source, source_id, sampled_at)"
                        " VALUES (?,?,?,?)",
                        (key, source, source_id, ts),
                    )
                    if (cur.rowcount or 0) > 0:
                        claimed.append(key)
                self._conn.commit()
            except Exception:
                try:
                    self._conn.rollback()
                except sqlite3.Error:  # pragma: no cover - defensive
                    pass
                raise
        return claimed

    def mark_rft_prompts_emitted(self, keys: List[str], dataset_id: str,
                                 kept_keys: Optional[List[str]] = None) -> int:
        """Stamp the consuming rft dataset on ledger rows; mark kept ones.

        Returns rows updated. ``kept_keys`` (subset of ``keys``) flips
        ``kept = 1`` — provenance for which prompts actually shipped a row.
        """
        if not keys:
            return 0
        kept_set = set(kept_keys or [])
        n = 0
        with self._lock:
            for key in keys:
                cur = self._conn.execute(
                    "UPDATE rft_prompts SET dataset_id = ?, kept = ? "
                    "WHERE prompt_key = ?",
                    (dataset_id, 1 if key in kept_set else 0, key),
                )
                n += cur.rowcount or 0
            self._conn.commit()
        return n

    def rft_prompt_seen(self, prompt_key: str) -> bool:
        with self._lock:
            return self._conn.execute(
                "SELECT 1 FROM rft_prompts WHERE prompt_key = ?", (prompt_key,)
            ).fetchone() is not None

    def list_rft_prompt_keys(self) -> List[str]:
        """Every ledger key, oldest sample first (selection's dedup filter)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT prompt_key FROM rft_prompts ORDER BY sampled_at, "
                "prompt_key"
            ).fetchall()
        return [r[0] for r in rows]

    # ------------------------------------------------- mined-golden ledger (v8)
    # Auto-mining goldens from 👍 traces (curate/mine_goldens.py). Mirrors the
    # rft_prompts pattern: claim BEFORE distilling with guarded INSERT OR
    # IGNORE inside ONE BEGIN IMMEDIATE transaction, so concurrent mining runs
    # each own a trace at most once and re-runs never re-mine. Rows are never
    # deleted — a dropped candidate staying in the ledger is what stops the
    # next run from retrying the same failure forever.

    def claim_mined_goldens(self, trace_ids: List[str], *,
                            stale_claim_minutes: int = 30) -> List[str]:
        """Atomically claim traces for golden-mining; return newly claimed ids.

        A trace already in the ledger (mined, dropped, or freshly in flight)
        is NOT returned — the caller must skip it. Insert order follows input
        order.

        Crash recovery: a run that dies between claiming and finalizing would
        otherwise strand its traces in status 'claimed' forever (the design
        keeps mined/dropped rows so failures are not retried, but an
        *unfinalized* claim records no decision at all). Before claiming,
        'claimed' rows older than ``stale_claim_minutes`` are released — that
        status only ever exists mid-run, and mining a handful of JSONL traces
        takes seconds, so an aged claim is by definition orphaned. Finalized
        (mined/dropped) rows are NEVER touched, so the no-retry guarantee for
        real outcomes stands.
        """
        claimed: List[str] = []
        if not trace_ids:
            return claimed
        now = datetime.now(timezone.utc)
        ts = now.isoformat(timespec="seconds")
        stale_cutoff = (now - timedelta(
            minutes=max(1, int(stale_claim_minutes)))
        ).isoformat(timespec="seconds")
        with self._lock:
            try:
                if not self._conn.in_transaction:
                    self._conn.execute("BEGIN IMMEDIATE")
                self._conn.execute(
                    "DELETE FROM mined_goldens"
                    " WHERE status = 'claimed' AND claimed_at < ?",
                    (stale_cutoff,),
                )
                for tid in trace_ids:
                    cur = self._conn.execute(
                        "INSERT OR IGNORE INTO mined_goldens"
                        " (trace_id, prompt_key, claimed_at, status)"
                        " VALUES (?,?,?,?)",
                        (tid, "", ts, "claimed"),
                    )
                    if (cur.rowcount or 0) > 0:
                        claimed.append(tid)
                self._conn.commit()
            except Exception:
                try:
                    self._conn.rollback()
                except sqlite3.Error:  # pragma: no cover - defensive
                    pass
                raise
        return claimed

    def record_mined_golden(self, trace_id: str, *, prompt_key: str,
                            status: str, reason: Optional[str] = None,
                            enabled: bool = False,
                            task_json: Optional[str] = None) -> int:
        """Finalize one claimed ledger row (status mined|dropped).

        Also stamps the final prompt_key (the claim inserts a placeholder),
        ``enabled`` (does the golden participate in the suite?) and the exact
        task dict persisted to the golden store. Returns rows updated.
        """
        if status not in ("mined", "dropped"):
            raise ValueError(f"unknown mined-golden status {status!r}")
        with self._lock:
            cur = self._conn.execute(
                "UPDATE mined_goldens SET prompt_key = ?, status = ?, reason = ?,"
                " enabled = ?, task_json = ? WHERE trace_id = ?",
                (prompt_key, status, reason, 1 if enabled else 0,
                 task_json, trace_id),
            )
            self._conn.commit()
            return cur.rowcount or 0

    def mined_golden_prompt_keys(self) -> List[str]:
        """Prompt keys already mined (dedup filter; '' placeholder excluded)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT prompt_key FROM mined_goldens "
                "WHERE prompt_key <> '' ORDER BY prompt_key"
            ).fetchall()
        return [r[0] for r in rows]

    # --------------------------------------------------- gate events (v9)
    # Durable resource-gate block episodes (training/scheduler.py writes,
    # dashboard/metrics.py counts). Episode dedup lives at the write site
    # (the scheduler knows the previous _gate_reason); this ledger is a plain
    # append-only log — rows are never updated or deleted.

    def record_gate_event(self, job_id: str, reason: str) -> None:
        """Append one gate-block event (one block episode)."""
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._lock:
            self._conn.execute(
                "INSERT INTO gate_events (job_id, reason, ts) VALUES (?,?,?)",
                (job_id, reason, ts),
            )
            self._conn.commit()

    def count_gate_events(self) -> int:
        """Total recorded gate-block events (cumulative counter value)."""
        with self._lock:
            return int(self._conn.execute(
                "SELECT COUNT(*) FROM gate_events").fetchone()[0])

    def query_positive_feedback_traces(self, *, min_score: float = 1.0,
                                       limit: int = 50) -> List[TraceRow]:
        """Traces with user 👍 feedback (score >= ``min_score``), unmined ones
        first is NOT assumed — the caller filters against its own claim set;
        this query only joins feedback. Oldest first (deterministic)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT t.* FROM traces t "
                "JOIN feedback f ON f.trace_id = t.trace_id "
                "WHERE f.source = 'user' AND f.score >= ? "
                "ORDER BY t.timestamp ASC, t.rowid ASC LIMIT ?",
                (float(min_score), int(limit)),
            ).fetchall()
        return [_row_to_trace(r) for r in rows]


def _row_to_trace(r: sqlite3.Row) -> TraceRow:
    d = dict(r)
    d["curated"] = bool(d.get("curated"))
    # Tolerant read for v1 DBs opened before the migration ran (or rows
    # written by older builds): absent column => None, never KeyError.
    d.setdefault("failure_reason", None)
    d.setdefault("project", None)
    d.setdefault("conf_mean", None)
    d.setdefault("conf_p90_negentropy", None)
    d.setdefault("plan_task", None)
    return TraceRow(**d)

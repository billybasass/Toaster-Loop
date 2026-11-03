"""Model registry: fine-tuned model lineage, aliases, promotion and rollback.

State lives in the same SQLite file as the experience buffer
(``<home>/buffer/toaster_loop.db``) — ARCHITECTURE.md §4 puts jobs + registry in the
single-writer daemon DB. Tables ``models``, ``jobs`` and ``active_pointers``
are created idempotently so opening the registry on a fresh home works too.

Model status state machine (Phase 4):

    candidate ──validate pass──▶ validated ──promote──▶ promoted
        │                             │                      │
        └──validate fail──▶ failed    └──validation re-run──┘   │
                                                               │
        promoted(new) supersedes old: old promoted ──▶ superseded
            rollback: current promoted ──▶ rolled_back, previous
            (most recent superseded) returns to promoted + active pointer.
    
    Backlog item #16 adds the recoverable-archive terminal branch (
    curator invariant: never delete — archive is recoverable):
    
        superseded / failed / rolled_back ──archive──▶ archived
        archived ──restore──▶ previous terminal status (exact file move-back)
    
    ``archived`` is reachable ONLY via the explicit ``archive_model`` call
    (CLI or disk-janitor) and only from the three terminal statuses; a
    ``pinned`` model bypasses automation entirely.
    
    ``parent_adapter`` carries the lineage edge: when a candidate is registered
    while an active model already exists for the same base_model, the active
    model_id is recorded as parent (vN → vN-1 chain).

The ``jobs`` table backs the Phase 4 scheduler (training/scheduler.py): one
row per worker subprocess with FIFO ordering on ``queued_at``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
import threading
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .ledger import LEDGER_TABLE_DDL, recent_ops, record, verify_ledger

logger = logging.getLogger("toaster_loop.registry")

REGISTRY_SCHEMA_VERSION = 11  # v11 (pass4 #43): jobs gains depends_on_json

#: Cross-profile transfer-probe history per model row stays small: newest-
#: first list, oldest evicted past this cap (pass 4 #37).
TRANSFERS_CAP = 20

MODEL_STATUSES = ("candidate", "validated", "promoted", "rolled_back", "failed",
                  "superseded", "archived")
# Statuses the archive transition may start from (backlog #16). Active and
# pre-promotion statuses are never archivable — supersede first.
ARCHIVABLE_STATUSES = ("superseded", "failed", "rolled_back")
JOB_KINDS = ("sft", "dpo", "grpo", "export", "validate", "curate",
             "shadow_eval", "evolve", "rft", "mine_goldens", "curate_source")
JOB_STATES = ("queued", "running", "done", "error", "cancelled")
# Job kinds executed in-process (no worker subprocess): curate by the dashboard
# thread, shadow_eval by the scheduler itself (pass 4 #32 — pure stdlib+httpx
# work needing this process's registry/store handles), evolve likewise (pass 4
# #39 — prompt search over goldens, same shape), rft likewise (rejection
# sampling: K chat completions + judge/golden verification against the served
# endpoint — HTTP I/O, no torch, needs this process's store/registry handles),
# mine_goldens likewise (distilling 👍 traces into golden tasks: stdlib JSONL
# + SQLite only; the optional judge call is httpx I/O like rft's).
# The stale-worker reaper must never touch these.
INPROCESS_JOB_KINDS = ("curate", "shadow_eval", "evolve", "rft",
                       "mine_goldens", "curate_source")

_MODELS_TABLE_DDL = """
CREATE TABLE models (
    model_id        TEXT PRIMARY KEY,
    base_model      TEXT NOT NULL,
    parent_adapter  TEXT,
    dataset_id      TEXT,
    manifest_hash   TEXT,
    adapter_dir     TEXT,
    gguf_path       TEXT,
    quant           TEXT,
    status          TEXT NOT NULL CHECK (status IN
                      ('candidate','validated','promoted','rolled_back','failed','superseded','archived')),
    created_at      TEXT NOT NULL,
    validation_json TEXT,
    train_loss      REAL,
    eval_loss       REAL,
    promoted_at     TEXT,
    pinned          INTEGER NOT NULL DEFAULT 0,
    pre_archive_paths TEXT,
    provenance_json TEXT,               -- pass3 #23
    last_used_at    TEXT,               -- pass4 #29
    shadow_json   TEXT,                 -- pass4 #32
    transfers_json TEXT                 -- pass4 #37: ALTER order — keep last
);
"""

_JOBS_TABLE_DDL = """
CREATE TABLE jobs (
    job_id       TEXT PRIMARY KEY,
    kind         TEXT NOT NULL CHECK (kind IN ('sft','dpo','grpo','export','validate','curate','shadow_eval','evolve','rft','mine_goldens','curate_source')),
    dataset_id   TEXT,
    model_id     TEXT,
    state        TEXT NOT NULL CHECK (state IN ('queued','running','done','error','cancelled')),
    log_path     TEXT,
    pid          INTEGER,
    queued_at    TEXT NOT NULL,
    started_at   TEXT,
    finished_at  TEXT,
    error        TEXT,
    params_json  TEXT,
    depends_on_json TEXT                 -- pass4 #43: job ids this job waits
                                         -- for; ALTER order — keep last
);
"""

_SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
    model_id        TEXT PRIMARY KEY,
    base_model      TEXT NOT NULL,
    parent_adapter  TEXT,               -- lineage edge: previous promoted model_id
    dataset_id      TEXT,
    manifest_hash   TEXT,               -- pinned per Phase 3 note
    adapter_dir     TEXT,
    gguf_path       TEXT,
    quant           TEXT,
    status          TEXT NOT NULL CHECK (status IN
                      ('candidate','validated','promoted','rolled_back','failed','superseded','archived')),
    created_at      TEXT NOT NULL,
    validation_json TEXT,
    train_loss      REAL,               -- backlog #4: final aggregate train loss
    eval_loss       REAL,               -- backlog #4: final eval loss (nullable)
    promoted_at     TEXT,               -- backlog #11: when this row was last promoted
    pinned          INTEGER NOT NULL DEFAULT 0,  -- backlog #16: bypass automation
    pre_archive_paths TEXT,             -- backlog #16: JSON {previous_status, paths{}}
    provenance_json TEXT,               -- pass3 #23: JSON resource_provenance block
                                        -- (base-model load mode + weight hashes)
    last_used_at    TEXT,               -- pass4 #29: ISO timestamp of the last
                                        -- activation (serve attach / batch run)
    shadow_json   TEXT,                 -- pass4 #32: latest shadow-baseline
                                        -- comparison JSON (judge-scored replay
                                        -- vs the active model)
    transfers_json TEXT                 -- pass4 #37: cross-profile transfer
                                        -- probe history (JSON list, newest
                                        -- first, capped); ALTER order — must
                                        -- stay the last column
);

CREATE TABLE IF NOT EXISTS jobs (
    job_id       TEXT PRIMARY KEY,
    kind         TEXT NOT NULL CHECK (kind IN ('sft','dpo','grpo','export','validate','curate','shadow_eval','evolve','rft','mine_goldens','curate_source')),
    dataset_id   TEXT,
    model_id     TEXT,
    state        TEXT NOT NULL CHECK (state IN ('queued','running','done','error','cancelled')),
    log_path     TEXT,
    pid          INTEGER,
    queued_at    TEXT NOT NULL,
    started_at   TEXT,
    finished_at  TEXT,
    error        TEXT,
    params_json  TEXT,
    depends_on_json TEXT                 -- pass4 #43: job ids this job waits
                                         -- for; ALTER order — keep last
);

CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state);
CREATE INDEX IF NOT EXISTS idx_jobs_queued ON jobs(queued_at);

CREATE TABLE IF NOT EXISTS active_pointers (
    base_model  TEXT PRIMARY KEY,
    model_id    TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
""" + LEDGER_TABLE_DDL  # pass4 #31: operation ledger (append-only)


# Ledger transactionality note (pass4 #31): record() never commits, so the
# entry rides the mutator's own transaction. A failed hash insert therefore
# rolls back the mutation it records — deliberate and stronger than a
# best-effort ledger: history that silently goes missing is worse than a
# refused operation. BEGIN IMMEDIATE inside record() serializes the prev-hash
# read + insert across registry handles on the same DB file (no chain forks);
# busy_timeout keeps a second writer waiting instead of erroring out.


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class ModelRow:
    model_id: str
    base_model: str
    parent_adapter: Optional[str]
    dataset_id: Optional[str]
    manifest_hash: Optional[str]
    adapter_dir: Optional[str]
    gguf_path: Optional[str]
    quant: Optional[str]
    status: str
    created_at: str
    validation: Optional[Dict[str, Any]] = None
    # Backlog item #4: training losses recorded when the train job completes.
    # eval_loss stays None when no eval split ran (tiny dataset / DPO).
    train_loss: Optional[float] = None
    eval_loss: Optional[float] = None
    # Backlog item #11: timestamp of the most recent promotion (insights
    # before/after deltas). Rows never promoted read None.
    promoted_at: Optional[str] = None
    # Backlog item #16: pinned models bypass the disk janitor and refuse
    # automated archival. pre_archive records {previous_status, paths{adapter_dir,
    # gguf_path}, archive_dir} while status == 'archived'; None otherwise.
    pinned: bool = False
    pre_archive: Optional[Dict[str, Any]] = None
    # Inspiration pass 3 #23: versioned resource_provenance block describing the
    # base model actually loaded (load mode + weight inventory). None for rows
    # predating attestation and for runs whose worker emitted no event.
    provenance: Optional[Dict[str, Any]] = None
    # Inspiration pass 4 #29: ISO timestamp of the model's last activation
    # (serve attach / batch run — NOT per token). None when never used; the
    # janitor treats that as "no usage signal" and falls back to age rules.
    last_used_at: Optional[str] = None
    # Inspiration pass 4 #32: latest shadow-baseline comparison (judge-scored
    # replay of a held-out trace set against the then-active model). Shape:
    # {ts, baseline_id, n, candidate_mean, baseline_mean, improvement,
    #  is_regression, per_task: [...], judge_model}. None until a
    # `shadow_eval` job has run for this row.
    shadow: Optional[Dict[str, Any]] = None
    # Inspiration pass 4 #37: cross-profile transfer-probe history — each entry
    # {ts, profile, baseline_id, n, candidate_mean, baseline_mean, improvement,
    # is_regression} from running this candidate against ANOTHER profile's
    # replay set (baseline = that profile's active model, or None for a
    # candidate-only probe). Newest first, capped at TRANSFERS_CAP. Tolerant
    # read: missing/corrupt hydrates to [], never an error.
    transfers: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Pass 4 #41: derived capability tier (registry/tiers.py), computed at
        # read time from the row's own evidence — there is deliberately NO
        # tier column: derivation-on-read keeps it honest and migration-free.
        # T3 additionally needs served-trace stats (a TraceStore query), so
        # this row-only view caps at T2; surfaces with a store recompute the
        # full tier (dashboard app._attach_tiers, `models list`).
        from .tiers import derive_tier, tier_name

        d["tier"] = derive_tier(self)
        d["tier_name"] = tier_name(d["tier"])
        return d


@dataclass
class JobRow:
    job_id: str
    kind: str
    dataset_id: Optional[str]
    model_id: Optional[str]
    state: str
    log_path: Optional[str] = None
    pid: Optional[int] = None
    queued_at: str = ""
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    # Inspiration pass 4 #43: job dependency edges. A queued job whose deps
    # are not ALL done stays unclaimable; a dep that ends error/cancelled
    # cascade-fails this job ("dependency <id> <state>"). Idea adapted from
    # PIASystem (MIT) app/core/task_graph.py (DAG with dependency resolution).
    depends_on: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RegistryError(RuntimeError):
    pass


class ModelRegistry:
    """models/jobs/active-pointer store. Thread-safe for one owner process."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._migrate_model_losses()
            self._migrate_promoted_at()
            self._migrate_archive_columns()
            self._migrate_provenance_column()
            self._migrate_last_used_at()
            self._migrate_shadow_column()
            self._migrate_transfers_column()
            self._migrate_jobs_depends_column()
            self._conn.executescript(_SCHEMA)
            self._migrate_job_kinds()
            self._migrate_job_kinds_shadow()
            self._migrate_job_kinds_evolve()
            self._migrate_job_kinds_rft()
            self._migrate_job_kinds_mine_goldens()
            self._migrate_job_kinds_curate_source()
            self._migrate_models_status_check()
            self._conn.commit()

    def _migrate_jobs_depends_column(self) -> None:
        """Add jobs.depends_on_json to pre-pass4 tables (inspiration pass 4 #43).

        Same shape-detection style as _migrate_transfers_column: idempotent,
        reads stay tolerant (_row_to_job defaults absent/corrupt to []). Runs
        BEFORE the job-kind CHECK rebuilds so their positional
        ``INSERT INTO jobs SELECT *`` copies see matching column order
        (depends_on_json last in both DDLs)."""
        cols = {r[1] for r in self._conn.execute(
            "PRAGMA table_info(jobs)").fetchall()}
        if not cols:  # fresh DB — _SCHEMA creates the column directly
            return
        if "depends_on_json" not in cols:
            self._conn.execute(
                "ALTER TABLE jobs ADD COLUMN depends_on_json TEXT")

    def _migrate_transfers_column(self) -> None:
        """Add models.transfers_json to pre-pass4 tables (inspiration pass 4 #37).

        Same shape-detection style as _migrate_shadow_column: idempotent, and
        reads stay tolerant (_row_to_model defaults an absent/corrupt column
        to []). Runs BEFORE _migrate_models_status_check so that rebuild's
        SELECT * sees matching column order (transfers_json last in both DDLs).
        """
        cols = {r[1] for r in self._conn.execute(
            "PRAGMA table_info(models)").fetchall()}
        if not cols:  # fresh DB — _SCHEMA creates the column directly
            return
        if "transfers_json" not in cols:
            self._conn.execute(
                "ALTER TABLE models ADD COLUMN transfers_json TEXT")

    def _migrate_shadow_column(self) -> None:
        """Add models.shadow_json to pre-pass4 tables (inspiration pass 4 #32).

        Same shape-detection style as _migrate_last_used_at: idempotent, and
        reads stay tolerant (_row_to_model defaults an absent column to None).
        Runs BEFORE _migrate_models_status_check so that rebuild's SELECT *
        sees matching column order (shadow_json last in both DDLs).
        """
        cols = {r[1] for r in self._conn.execute(
            "PRAGMA table_info(models)").fetchall()}
        if not cols:  # fresh DB — _SCHEMA creates the column directly
            return
        if "shadow_json" not in cols:
            self._conn.execute("ALTER TABLE models ADD COLUMN shadow_json TEXT")

    def _migrate_job_kinds_curate_source(self) -> None:
        """Rebuild the jobs table when its kind CHECK predates 'curate_source'
        (WP3d folder/website curation jobs).

        Same family as _migrate_job_kinds_mine_goldens (SQLite bakes CHECK
        constraints into the stored DDL). Idempotent: runs only when the
        stored DDL lacks 'curate_source'.
        """
        row = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='jobs'"
        ).fetchone()
        if row is None or "'curate_source'" in (row[0] or ""):
            return
        self._conn.executescript(
            "ALTER TABLE jobs RENAME TO jobs_old_cs;"
            + _JOBS_TABLE_DDL
            + "INSERT INTO jobs SELECT * FROM jobs_old_cs;"
            "DROP TABLE jobs_old_cs;"
            "CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state);"
            "CREATE INDEX IF NOT EXISTS idx_jobs_queued ON jobs(queued_at);"
        )

    def _migrate_job_kinds_mine_goldens(self) -> None:
        """Rebuild the jobs table when its kind CHECK predates 'mine_goldens'.

        Same family as _migrate_job_kinds_rft (SQLite bakes CHECK constraints
        into the stored DDL). Idempotent: runs only when the stored DDL lacks
        'mine_goldens' — a DB that still needed an older rebuild gets the new
        DDL from that earlier migration and is skipped here.
        """
        row = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='jobs'"
        ).fetchone()
        if row is None or "'mine_goldens'" in (row[0] or ""):
            return
        self._conn.executescript(
            "ALTER TABLE jobs RENAME TO jobs_old_mg;"
            + _JOBS_TABLE_DDL
            + "INSERT INTO jobs SELECT * FROM jobs_old_mg;"
            "DROP TABLE jobs_old_mg;"
            "CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state);"
            "CREATE INDEX IF NOT EXISTS idx_jobs_queued ON jobs(queued_at);"
        )

    def _migrate_job_kinds_rft(self) -> None:
        """Rebuild the jobs table when its kind CHECK predates 'rft'.

        Same family as _migrate_job_kinds/_shadow/_evolve (SQLite bakes CHECK
        constraints into the stored DDL). Idempotent: runs only when the stored
        DDL lacks 'rft' — a DB that still needed an older rebuild gets the new
        DDL from that earlier migration and is skipped.
        """
        row = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='jobs'"
        ).fetchone()
        if row is None or "'rft'" in (row[0] or ""):
            return
        self._conn.executescript(
            "ALTER TABLE jobs RENAME TO jobs_old_rft;"
            + _JOBS_TABLE_DDL
            + "INSERT INTO jobs SELECT * FROM jobs_old_rft;"
            "DROP TABLE jobs_old_rft;"
            "CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state);"
            "CREATE INDEX IF NOT EXISTS idx_jobs_queued ON jobs(queued_at);"
        )

    def _migrate_job_kinds_evolve(self) -> None:
        """Rebuild the jobs table when its kind CHECK predates 'evolve'.

        Same family as _migrate_job_kinds/_migrate_job_kinds_shadow (SQLite
        bakes CHECK constraints into the stored DDL). Idempotent: runs only
        when the stored DDL lacks 'evolve' — a DB that still needed an older
        rebuild gets the new DDL from that earlier migration and is skipped.
        """
        row = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='jobs'"
        ).fetchone()
        if row is None or "'evolve'" in (row[0] or ""):
            return
        self._conn.executescript(
            "ALTER TABLE jobs RENAME TO jobs_old_39;"
            + _JOBS_TABLE_DDL
            + "INSERT INTO jobs SELECT * FROM jobs_old_39;"
            "DROP TABLE jobs_old_39;"
            "CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state);"
            "CREATE INDEX IF NOT EXISTS idx_jobs_queued ON jobs(queued_at);"
        )

    def _migrate_job_kinds_shadow(self) -> None:
        """Rebuild the jobs table when its kind CHECK predates 'shadow_eval'.

        Same family as _migrate_job_kinds (SQLite bakes CHECK constraints into
        the stored DDL). Idempotent: runs only when the stored DDL lacks
        'shadow_eval' — a DB that still needed the grpo migration above gets
        the new DDL from that rebuild and is skipped here.
        """
        row = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='jobs'"
        ).fetchone()
        if row is None or "'shadow_eval'" in (row[0] or ""):
            return
        self._conn.executescript(
            "ALTER TABLE jobs RENAME TO jobs_old_32;"
            + _JOBS_TABLE_DDL
            + "INSERT INTO jobs SELECT * FROM jobs_old_32;"
            "DROP TABLE jobs_old_32;"
            "CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state);"
            "CREATE INDEX IF NOT EXISTS idx_jobs_queued ON jobs(queued_at);"
        )

    def _migrate_archive_columns(self) -> None:
        """Add models.pinned / models.pre_archive_paths to pre-v5 tables (#16).

        Same shape-detection style as _migrate_promoted_at: idempotent, and
        reads stay tolerant (_row_to_model defaults absent columns to None).
        """
        cols = {r[1] for r in self._conn.execute(
            "PRAGMA table_info(models)").fetchall()}
        if not cols:  # fresh DB — _SCHEMA creates the columns directly
            return
        if "pinned" not in cols:
            self._conn.execute(
                "ALTER TABLE models ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0")
        if "pre_archive_paths" not in cols:
            self._conn.execute(
                "ALTER TABLE models ADD COLUMN pre_archive_paths TEXT")

    def _migrate_provenance_column(self) -> None:
        """Add models.provenance_json to pre-pass3 tables (inspiration #23).

        Same shape-detection style as the other migrations: idempotent, and
        reads stay tolerant (_row_to_model defaults an absent column to None).
        Runs BEFORE _migrate_models_status_check so that rebuild's SELECT *
        sees matching column order (provenance_json then last_used_at in both
        DDLs).
        """
        cols = {r[1] for r in self._conn.execute(
            "PRAGMA table_info(models)").fetchall()}
        if not cols:  # fresh DB — _SCHEMA creates the column directly
            return
        if "provenance_json" not in cols:
            self._conn.execute(
                "ALTER TABLE models ADD COLUMN provenance_json TEXT")

    def _migrate_last_used_at(self) -> None:
        """Add models.last_used_at to pre-pass4 tables (inspiration pass 4 #29).

        Same shape-detection style as _migrate_provenance_column: idempotent,
        and reads stay tolerant (_row_to_model defaults an absent column to
        None). Runs BEFORE _migrate_models_status_check so that rebuild's
        SELECT * sees matching column order (last_used_at last in both DDLs).
        """
        cols = {r[1] for r in self._conn.execute(
            "PRAGMA table_info(models)").fetchall()}
        if not cols:  # fresh DB — _SCHEMA creates the column directly
            return
        if "last_used_at" not in cols:
            self._conn.execute("ALTER TABLE models ADD COLUMN last_used_at TEXT")

    def _migrate_models_status_check(self) -> None:
        """Rebuild the models table when its status CHECK predates 'archived' (#16).

        SQLite CHECK constraints are baked into the stored DDL, so a DB created
        before backlog #16 would reject status='archived'. Mirrors
        _migrate_job_kinds: idempotent — runs only when the stored DDL lacks
        'archived'. Column order matches (post-ALTER) so INSERT ... SELECT *
        is safe; the models table carries no secondary indexes or FKs.
        """
        row = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='models'"
        ).fetchone()
        if row is None or "'archived'" in (row[0] or ""):
            return
        self._conn.executescript(
            "ALTER TABLE models RENAME TO models_old_16;"
            + _MODELS_TABLE_DDL
            + "INSERT INTO models SELECT * FROM models_old_16;"
            "DROP TABLE models_old_16;"
        )

    def _migrate_promoted_at(self) -> None:
        """Add the models.promoted_at column to pre-v4 models tables (#11).

        Same shape-detection style as _migrate_model_losses: idempotent, and
        reads stay tolerant (_row_to_model defaults absent columns to None).
        """
        cols = {r[1] for r in self._conn.execute(
            "PRAGMA table_info(models)").fetchall()}
        if not cols:  # fresh DB — _SCHEMA creates the column directly
            return
        if "promoted_at" not in cols:
            self._conn.execute("ALTER TABLE models ADD COLUMN promoted_at TEXT")

    def _migrate_model_losses(self) -> None:
        """Add train_loss/eval_loss columns to pre-v3 models tables (#4).

        Follows the store/db.py pattern: keyed on actual table shape, not the
        version pragma, so it is idempotent and safe on any existing DB. Reads
        stay tolerant (_row_to_model uses .get-style access).
        """
        cols = {r[1] for r in self._conn.execute(
            "PRAGMA table_info(models)").fetchall()}
        if not cols:  # fresh DB — _SCHEMA creates the columns directly
            return
        for col in ("train_loss", "eval_loss"):
            if col not in cols:
                self._conn.execute(
                    f"ALTER TABLE models ADD COLUMN {col} REAL")

    def _migrate_job_kinds(self) -> None:
        """Rebuild the jobs table when its kind CHECK predates current kinds.

        SQLite CHECK constraints are baked into the table definition, so an
        existing DB created before the `curate`/`grpo` job kinds needs a
        rebuild. Idempotent: runs only when the stored DDL lacks 'grpo'
        (the newest kind; anything older is missing it too).
        """
        row = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='jobs'"
        ).fetchone()
        if row is None or "'grpo'" in (row[0] or ""):
            return
        self._conn.executescript(
            "ALTER TABLE jobs RENAME TO jobs_old_5d;"
            + _JOBS_TABLE_DDL
            + "INSERT INTO jobs SELECT * FROM jobs_old_5d;"
            "DROP TABLE jobs_old_5d;"
            # the rename moved the old indexes onto jobs_old_5d; drop removed
            # them, so recreate for the fresh table.
            "CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state);"
            "CREATE INDEX IF NOT EXISTS idx_jobs_queued ON jobs(queued_at);"
        )

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass

    # ------------------------------------------------------------- models

    def register_model(
        self,
        base_model: str,
        dataset_id: Optional[str] = None,
        manifest_hash: Optional[str] = None,
        quant: Optional[str] = None,
        parent_adapter: Optional[str] = None,
        actor: str = "cli",
        intent: Optional[str] = None,
    ) -> ModelRow:
        """Register a new candidate model with auto vN id.

        ``parent_adapter`` defaults to the currently active model for this
        base_model (lineage chain), or None when nothing is promoted yet.
        ``intent`` (pass-4 #36) is the optional "why" recorded on the ledger
        entry ("we trained on D expecting Y").
        """
        with self._lock:
            if parent_adapter is None:
                r = self._conn.execute(
                    "SELECT model_id FROM active_pointers WHERE base_model = ?",
                    (base_model,),
                ).fetchone()
                parent_adapter = r[0] if r else None
            n = self._conn.execute("SELECT COUNT(*) FROM models").fetchone()[0]
            model_id = f"v{n + 1}"
            while self._conn.execute(
                "SELECT 1 FROM models WHERE model_id = ?", (model_id,)
            ).fetchone():
                n += 1
                model_id = f"v{n + 1}"
            row = ModelRow(
                model_id=model_id,
                base_model=base_model,
                parent_adapter=parent_adapter,
                dataset_id=dataset_id,
                manifest_hash=manifest_hash,
                adapter_dir=None,
                gguf_path=None,
                quant=quant,
                status="candidate",
                created_at=_utcnow(),
                validation=None,
            )
            self._conn.execute(
                "INSERT INTO models (model_id, base_model, parent_adapter, dataset_id,"
                " manifest_hash, adapter_dir, gguf_path, quant, status, created_at,"
                " validation_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    row.model_id, row.base_model, row.parent_adapter, row.dataset_id,
                    row.manifest_hash, row.adapter_dir, row.gguf_path, row.quant,
                    row.status, row.created_at, None,
                ),
            )
            record(self._conn, actor, "register", model_id, before=None,
                   after={"status": "candidate", "base_model": base_model,
                          "parent_adapter": parent_adapter},
                   intent=intent)
            self._conn.commit()
        return row

    def get_model(self, model_id: str) -> Optional[ModelRow]:
        with self._lock:
            r = self._conn.execute(
                "SELECT * FROM models WHERE model_id = ?", (model_id,)
            ).fetchone()
        return _row_to_model(r) if r else None

    def list_models(self, base_model: Optional[str] = None) -> List[ModelRow]:
        q = "SELECT * FROM models"
        args: List[Any] = []
        if base_model:
            q += " WHERE base_model = ?"
            args.append(base_model)
        # numeric-ish vN ordering (v2 < v10), creation time as tiebreak
        q += " ORDER BY LENGTH(model_id), model_id, created_at"
        with self._lock:
            rows = self._conn.execute(q, args).fetchall()
        return [_row_to_model(r) for r in rows]

    def set_artifacts(
        self,
        model_id: str,
        adapter_dir: Optional[str] = None,
        gguf_path: Optional[str] = None,
        quant: Optional[str] = None,
    ) -> None:
        sets: List[str] = []
        args: List[Any] = []
        if adapter_dir is not None:
            sets.append("adapter_dir = ?")
            args.append(str(adapter_dir))
        if gguf_path is not None:
            sets.append("gguf_path = ?")
            args.append(str(gguf_path))
        if quant is not None:
            sets.append("quant = ?")
            args.append(quant)
        if not sets:
            return
        args.append(model_id)
        with self._lock:
            self._conn.execute(
                f"UPDATE models SET {', '.join(sets)} WHERE model_id = ?", args
            )
            self._conn.commit()

    def set_provenance(self, model_id: str,
                       provenance: Optional[Dict[str, Any]]) -> None:
        """Store the resource_provenance block on a model row (pass 3 #23).

        The worker reports the base-model load mode it actually used through
        the job log; the pipeline parses it and calls this after training. A
        None/empty block is a no-op — absence stays "unattested", never an
        overwrite of prior attestation.
        """
        if not provenance:
            return
        with self._lock:
            self._conn.execute(
                "UPDATE models SET provenance_json = ? WHERE model_id = ?",
                (json.dumps(provenance, ensure_ascii=False), model_id),
            )
            self._conn.commit()

    def record_activation(self, model_id: str) -> bool:
        """Stamp ``last_used_at = now`` on a registry model (pass 4 #29).

        Idea adapted from PIASystem (MIT) app/core/adapter_manager.py —
        age-based policy should see actual *use*, not just creation/promotion
        dates. Cheap single UPDATE, safe to call on serve-attach paths: it
        never raises; an unknown model_id logs and returns False.

        Returns True when a row was updated, False otherwise.
        """
        try:
            with self._lock:
                cur = self._conn.execute(
                    "UPDATE models SET last_used_at = ? WHERE model_id = ?",
                    (_utcnow(), model_id),
                )
                self._conn.commit()
                if cur.rowcount == 0:
                    logger.debug("record_activation: unknown model %s", model_id)
                    return False
            return True
        except sqlite3.Error as e:
            logger.warning("record_activation(%s) failed: %s", model_id, e)
            return False

    def set_losses(
        self,
        model_id: str,
        train_loss: Optional[float] = None,
        eval_loss: Optional[float] = None,
        actor: str = "cli",
        intent: Optional[str] = None,
    ) -> None:
        """Record final training losses on the model row (backlog item #4).

        Only non-None values are written; eval_loss stays NULL when no eval
        split ran (tiny datasets, DPO). Ledgered (pass4 #31) with a minimal
        before/after loss diff.
        """
        sets: List[str] = []
        args: List[Any] = []
        if train_loss is not None:
            sets.append("train_loss = ?")
            args.append(float(train_loss))
        if eval_loss is not None:
            sets.append("eval_loss = ?")
            args.append(float(eval_loss))
        if not sets:
            return
        args.append(model_id)
        with self._lock:
            cur = self._conn.execute(
                "SELECT train_loss, eval_loss FROM models WHERE model_id = ?",
                (model_id,),
            ).fetchone()
            self._conn.execute(
                f"UPDATE models SET {', '.join(sets)} WHERE model_id = ?", args
            )
            record(self._conn, actor, "set_losses", model_id,
                   before={"train_loss": cur[0], "eval_loss": cur[1]} if cur else None,
                   after={"train_loss": float(train_loss) if train_loss is not None
                          else (cur[0] if cur else None),
                           "eval_loss": float(eval_loss) if eval_loss is not None
                          else (cur[1] if cur else None)},
                   intent=intent)
            self._conn.commit()

    def set_validation(
        self, model_id: str, results: Dict[str, Any], passed: bool,
        actor: str = "cli",
    ) -> None:
        """Store validation results; candidate/validated -> validated|failed.

        A promoted model is never downgraded here (promotion re-validation is a
        Phase 5 nicety); only candidate/validated models flip status.
        """
        with self._lock:
            r = self._conn.execute(
                "SELECT status FROM models WHERE model_id = ?", (model_id,)
            ).fetchone()
            if r is None:
                raise RegistryError(f"no model {model_id}")
            status = r[0]
            new_status = "validated" if passed else "failed"
            if status in ("promoted", "superseded"):
                new_status = status
            self._conn.execute(
                "UPDATE models SET status = ?, validation_json = ? WHERE model_id = ?",
                (new_status, json.dumps(results), model_id),
            )
            record(self._conn, actor, "validate", model_id,
                   before={"status": status}, after={"status": new_status})
            self._conn.commit()

    def set_shadow(
        self, model_id: str, shadow: Optional[Dict[str, Any]],
        actor: str = "cli", intent: Optional[str] = None,
    ) -> None:
        """Store the latest shadow-baseline comparison on the model row (#32).

        Idea adapted from PIASystem (MIT) app/core/evaluation_loop.py — the
        ShadowBaselineComparison lands next to eval_loss so promotion and the
        UI can consult "did the candidate actually regress on real tasks".
        Ledgered with a minimal before/after verdict diff (pass4 #31). A None
        payload clears the column honestly (never silently keeps stale data).
        """
        with self._lock:
            cur = self._conn.execute(
                "SELECT shadow_json FROM models WHERE model_id = ?",
                (model_id,),
            ).fetchone()
            if cur is None:
                raise RegistryError(f"no model {model_id}")
            before = json.loads(cur[0]) if cur[0] else None
            self._conn.execute(
                "UPDATE models SET shadow_json = ? WHERE model_id = ?",
                (json.dumps(shadow, ensure_ascii=False) if shadow is not None
                 else None, model_id),
            )

            def _verdict(v):
                if not isinstance(v, dict):
                    return None
                return {"improvement": v.get("improvement"),
                        "is_regression": v.get("is_regression"),
                        "baseline_id": v.get("baseline_id")}

            record(self._conn, actor, "set_shadow", model_id,
                   before=_verdict(before), after=_verdict(shadow),
                   intent=intent)
            self._conn.commit()

    def record_transfer(
        self, model_id: str, entry: Dict[str, Any],
        actor: str = "cli", intent: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Append one cross-profile transfer-probe result (pass 4 #37).

        Idea adapted from PIASystem (MIT) app/core/transfer_bridge.py — a
        mediated transfer records the shadow-simulation score per destination
        domain on the artifact row. Entries are {ts, profile, baseline_id, n,
        candidate_mean, baseline_mean, improvement, is_regression}; stored
        newest-first in ``models.transfers_json``, capped at TRANSFERS_CAP
        (oldest evicted) so the row stays small. Ledgered (pass4 #31) with a
        minimal before/after diff of the list length + newest entry.

        Returns the new full history. Does NOT touch shadow_json — that column
        is the promotion verdict for the model's OWN profile; transfers are a
        separate axis.
        """
        if not isinstance(entry, dict) or not entry.get("profile"):
            raise RegistryError("transfer entry needs at least a 'profile'")
        with self._lock:
            cur = self._conn.execute(
                "SELECT transfers_json FROM models WHERE model_id = ?",
                (model_id,),
            ).fetchone()
            if cur is None:
                raise RegistryError(f"no model {model_id}")
            history = _loads_transfers(cur[0])
            history = [entry] + history
            del history[TRANSFERS_CAP:]  # newest-first cap, evict oldest
            self._conn.execute(
                "UPDATE models SET transfers_json = ? WHERE model_id = ?",
                (json.dumps(history, ensure_ascii=False), model_id),
            )
            record(self._conn, actor, "record_transfer", model_id,
                   before={"count": len(history) - 1},
                   after={"count": len(history),
                          "profile": entry.get("profile"),
                          "improvement": entry.get("improvement")},
                   intent=intent)
            self._conn.commit()
        return history

    def promote(self, model_id: str, actor: str = "cli",
                intent: Optional[str] = None) -> ModelRow:
        """Promote a validated model; previous promoted for the base -> superseded.

        ``intent`` (pass-4 #36): optional one-liner on the ledger entry —
        "we promoted X expecting Y" — later paired with the observed score
        delta by insights ``intent_review``.
        """
        with self._lock:
            r = self._conn.execute(
                "SELECT * FROM models WHERE model_id = ?", (model_id,)
            ).fetchone()
            if r is None:
                raise RegistryError(f"no model {model_id}")
            m = _row_to_model(r)
            if m.status not in ("validated", "rolled_back"):
                raise RegistryError(
                    f"cannot promote {model_id} with status {m.status!r} "
                    "(must be validated first)"
                )
            prev = self._conn.execute(
                "SELECT model_id FROM active_pointers WHERE base_model = ?",
                (m.base_model,),
            ).fetchone()
            if prev and prev[0] != model_id:
                self._conn.execute(
                    "UPDATE models SET status = 'superseded' WHERE model_id = ? "
                    "AND status = 'promoted'",
                    (prev[0],),
                )
            self._conn.execute(
                "UPDATE models SET status = 'promoted', promoted_at = ? "
                "WHERE model_id = ?",
                (_utcnow(), model_id),
            )
            self._conn.execute(
                "INSERT INTO active_pointers (base_model, model_id, updated_at) "
                "VALUES (?,?,?) ON CONFLICT(base_model) DO UPDATE SET "
                "model_id = excluded.model_id, updated_at = excluded.updated_at",
                (m.base_model, model_id, _utcnow()),
            )
            record(self._conn, actor, "promote", model_id,
                   before={"status": m.status,
                           "previous_active": prev[0] if prev else None},
                   after={"status": "promoted"}, intent=intent)
            self._conn.commit()
        out = self.get_model(model_id)
        assert out is not None
        return out

    def rollback(self, base_model: Optional[str] = None,
                 actor: str = "cli",
                 intent: Optional[str] = None) -> Optional[ModelRow]:
        """Roll active pointer back to the previously promoted model.

        Marks the current promoted model ``rolled_back`` and restores the most
        recent ``superseded`` model for that base (status -> promoted, pointer
        updated). Returns the newly active model, or None when there is nothing
        to roll back to (pointer cleared in that case).
        """
        with self._lock:
            if base_model:
                p = self._conn.execute(
                    "SELECT base_model, model_id FROM active_pointers WHERE base_model = ?",
                    (base_model,),
                ).fetchone()
            else:
                p = self._conn.execute(
                    "SELECT base_model, model_id FROM active_pointers "
                    "ORDER BY updated_at DESC LIMIT 1"
                ).fetchone()
            if p is None:
                return None
            base, current = p[0], p[1]

            # Candidate for restore: most recent superseded (or rolled_back)
            # model of the same base, excluding the current one.
            prev = self._conn.execute(
                "SELECT model_id FROM models WHERE base_model = ? AND model_id != ? "
                "AND status IN ('superseded','rolled_back') ORDER BY LENGTH(model_id) DESC, "
                "model_id DESC, created_at DESC LIMIT 1",
                (base, current),
            ).fetchone()

            self._conn.execute(
                "UPDATE models SET status = 'rolled_back' WHERE model_id = ?",
                (current,),
            )
            if prev is None:
                self._conn.execute(
                    "DELETE FROM active_pointers WHERE base_model = ?", (base,)
                )
                record(self._conn, actor, "rollback", current,
                       before={"status": "promoted"},
                       after={"status": "rolled_back", "restored": None},
                       intent=intent)
                self._conn.commit()
                return None
            self._conn.execute(
                "UPDATE models SET status = 'promoted', promoted_at = ? "
                "WHERE model_id = ?",
                (_utcnow(), prev[0]),
            )
            self._conn.execute(
                "UPDATE active_pointers SET model_id = ?, updated_at = ? "
                "WHERE base_model = ?",
                (prev[0], _utcnow(), base),
            )
            record(self._conn, actor, "rollback", current,
                   before={"status": "promoted"},
                   after={"status": "rolled_back", "restored": prev[0]},
                   intent=intent)
            self._conn.commit()
        return self.get_model(prev[0])

    def active_model(self, base_model: Optional[str] = None) -> Optional[ModelRow]:
        """The model the ``active`` alias currently resolves to."""
        with self._lock:
            if base_model:
                p = self._conn.execute(
                    "SELECT model_id FROM active_pointers WHERE base_model = ?",
                    (base_model,),
                ).fetchone()
            else:
                p = self._conn.execute(
                    "SELECT model_id FROM active_pointers ORDER BY updated_at DESC LIMIT 1"
                ).fetchone()
            if p is None:
                return None
            r = self._conn.execute(
                "SELECT * FROM models WHERE model_id = ?", (p[0],)
            ).fetchone()
        return _row_to_model(r) if r else None

    # ------------------------------------------- archive lifecycle (#16)

    @property
    def home(self) -> Path:
        """State root inferred from the standard <home>/buffer/toaster_loop.db layout."""
        return self.db_path.resolve().parent.parent

    def _archive_root(self, home: Optional[Path] = None) -> Path:
        return (Path(home) if home is not None else self.home) / "archive"

    def set_pinned(self, model_id: str, pinned: bool,
                   actor: str = "cli") -> ModelRow:
        """Pin/unpin a model (backlog item #16).

        Pinned models bypass the disk janitor and refuse archival — mirrors
        the curator invariant "pinned skills bypass all auto-
        transitions". Pinning is allowed in any status (a pin on a terminal
        model protects it from the janitor; on an active one it is inert).
        """
        with self._lock:
            r = self._conn.execute(
                "SELECT 1 FROM models WHERE model_id = ?", (model_id,)
            ).fetchone()
            if r is None:
                raise RegistryError(f"no model {model_id}")
            cur = self._conn.execute(
                "SELECT pinned FROM models WHERE model_id = ?", (model_id,)
            ).fetchone()
            self._conn.execute(
                "UPDATE models SET pinned = ? WHERE model_id = ?",
                (1 if pinned else 0, model_id),
            )
            record(self._conn, actor, "pin" if pinned else "unpin", model_id,
                   before={"pinned": bool(cur[0])} if cur else None,
                   after={"pinned": bool(pinned)})
            self._conn.commit()
        out = self.get_model(model_id)
        assert out is not None
        return out

    def archive_model(self, model_id: str,
                      home: Optional[Path] = None,
                      actor: str = "cli") -> ModelRow:
        """Move a terminal model's artifacts under ``<home>/archive/<id>/`` (#16).

        Recoverable by design (invariant: never delete — archive is
        recoverable): the GGUF file and adapter dir referenced on the row are
        moved (not copied, not removed) into a per-model directory under the
        archive root; name collisions inside the archive get ``-2``/``-3``…
        suffixes. The original paths plus the previous status are recorded in
        ``pre_archive_paths`` (JSON) so :meth:`restore_model` is exact, and
        the artifact columns are cleared so nothing can serve them by accident.

        Refusals: unknown id; already archived; status outside
        ``ARCHIVABLE_STATUSES`` (superseded/failed/rolled_back — active and
        pre-promotion models must be superseded/failed first); pinned models.
        """
        with self._lock:
            r = self._conn.execute(
                "SELECT * FROM models WHERE model_id = ?", (model_id,)
            ).fetchone()
            if r is None:
                raise RegistryError(f"no model {model_id}")
            m = _row_to_model(r)
            if m.status == "archived":
                raise RegistryError(f"model {model_id} is already archived")
            if m.pinned:
                raise RegistryError(
                    f"refusing to archive pinned model {model_id} "
                    "(`toaster_loop models unpin " + model_id + "` first)"
                )
            if m.status not in ARCHIVABLE_STATUSES:
                hint = ""
                if m.status == "promoted":
                    hint = (" — it is the current/previous promoted model; "
                            "supersede or roll back first")
                elif m.status in ("candidate", "validated"):
                    hint = (" — only terminal models (superseded/failed/"
                            "rolled_back) can be archived; let validation "
                            "fail or supersede it first")
                raise RegistryError(
                    f"cannot archive {model_id} with status {m.status!r}"
                    + hint
                )

            # Independent artifact paths: skip a GGUF nested inside the
            # adapter dir (moving the dir already carries it along).
            raw_paths = [p for p in (m.adapter_dir, m.gguf_path) if p]
            paths = [
                p for p in raw_paths
                if not any(p != q and Path(p).is_relative_to(q)
                           for q in raw_paths)
            ]

            dest_root = _unique_path(self._archive_root(home) / model_id)
            moved: Dict[str, str] = {}  # original path -> archived path
            try:
                if paths:
                    dest_root.mkdir(parents=True, exist_ok=True)
                for p in paths:
                    src = Path(p)
                    if not src.exists():
                        raise RegistryError(
                            f"cannot archive {model_id}: artifact missing: {src}"
                        )
                    dest = _unique_path(dest_root / src.name)
                    shutil.move(str(src), str(dest))
                    moved[str(src)] = str(dest)
            except Exception:
                # Best-effort rollback of partial moves; never lose files.
                for orig, archived in moved.items():
                    try:
                        os.makedirs(os.path.dirname(orig), exist_ok=True)
                        shutil.move(archived, orig)
                    except OSError:
                        pass
                raise

            record = {
                "previous_status": m.status,
                "paths": moved,
                # Row-column snapshot so restore repopulates exactly.
                "adapter_dir": m.adapter_dir,
                "gguf_path": m.gguf_path,
                "archive_dir": str(dest_root) if paths else None,
                "archived_at": _utcnow(),
            }
            self._conn.execute(
                "UPDATE models SET status = 'archived', adapter_dir = NULL, "
                "gguf_path = NULL, pre_archive_paths = ? WHERE model_id = ?",
                (json.dumps(record), model_id),
            )
            # NB: local `record` dict shadows the ledger helper here; the
            # ledger call below uses the module-level function explicitly.
            from . import ledger as _ledger
            _ledger.record(self._conn, actor, "archive", model_id,
                           before={"status": m.status},
                           after={"status": "archived"})
            self._conn.commit()
        out = self.get_model(model_id)
        assert out is not None
        return out

    def restore_model(self, model_id: str,
                      actor: str = "cli") -> ModelRow:
        """Move an archived model's artifacts back to their recorded paths (#16).

        Exact inverse of :meth:`archive_model`: every path recorded in
        ``pre_archive_paths`` must still exist inside the archive and its
        original destination must be free (otherwise refuse with a clear
        error — no silent overwrites). Status returns to the pre-archive
        status; the artifact columns are repopulated.
        """
        with self._lock:
            r = self._conn.execute(
                "SELECT * FROM models WHERE model_id = ?", (model_id,)
            ).fetchone()
            if r is None:
                raise RegistryError(f"no model {model_id}")
            m = _row_to_model(r)
            if m.status != "archived":
                raise RegistryError(
                    f"cannot restore {model_id} with status {m.status!r} "
                    "(only archived models are restorable)"
                )
            rec = m.pre_archive or {}
            paths: Dict[str, str] = rec.get("paths") or {}

            # Validate everything before touching the filesystem.
            for orig, dest in paths.items():
                if not Path(dest).exists():
                    raise RegistryError(
                        f"archived artifact missing for {model_id}: {dest} "
                        "(archive tampered with?)"
                    )
                if Path(orig).exists():
                    raise RegistryError(
                        f"cannot restore {model_id}: target path already "
                        f"exists: {orig}"
                    )

            moved_back: List[str] = []
            try:
                for orig, dest in paths.items():
                    os.makedirs(os.path.dirname(orig), exist_ok=True)
                    shutil.move(dest, orig)
                    moved_back.append(orig)
            except Exception:
                for orig in moved_back:  # partial failure -> put back what moved
                    try:
                        shutil.move(orig, paths[orig])
                    except OSError:
                        pass
                raise

            # Tidy: drop the per-model archive dir once it is empty.
            adir = rec.get("archive_dir")
            if adir:
                try:
                    os.rmdir(adir)
                except OSError:
                    pass  # non-empty (other content) — leave it alone

            # Repopulate artifact columns from the pre-archive row snapshot.
            self._conn.execute(
                "UPDATE models SET status = ?, adapter_dir = ?, gguf_path = ?, "
                "pre_archive_paths = NULL WHERE model_id = ?",
                (rec.get("previous_status", "superseded"),
                 rec.get("adapter_dir"), rec.get("gguf_path"), model_id),
            )
            from . import ledger as _ledger  # local `rec`/shadow safety
            _ledger.record(self._conn, actor, "restore", model_id,
                           before={"status": "archived"},
                           after={"status": rec.get("previous_status",
                                                    "superseded")})
            self._conn.commit()
        out = self.get_model(model_id)
        assert out is not None
        return out

    # --------------------------------------------------------------- jobs

    def enqueue_job(
        self,
        kind: str,
        dataset_id: Optional[str] = None,
        model_id: Optional[str] = None,
        log_path: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        actor: Optional[str] = None,
        depends_on: Optional[List[str]] = None,
    ) -> JobRow:
        """Append a queued job. ``actor`` (pass4 #31/#32): when set, the
        enqueue itself is ledgered ("someone asked for this work"); existing
        callers leave it None and keep the pre-#32 behavior (job *transitions*
        were always ledgered; enqueues are only from here on, opt-in).

        ``depends_on`` (pass4 #43): job ids this job waits for. Validated at
        enqueue time — unknown ids and self-dependencies are refused outright
        (a typo should fail the submit, not wedge the queue forever); cycles
        cannot be created because deps must already exist."""
        if kind not in JOB_KINDS:
            raise RegistryError(f"unknown job kind {kind!r}")
        deps = [str(d) for d in (depends_on or [])]
        with self._lock:
            for d in deps:
                if not d:
                    raise RegistryError("empty dependency id")
                if self._conn.execute(
                    "SELECT 1 FROM jobs WHERE job_id = ?", (d,)
                ).fetchone() is None:
                    raise RegistryError(
                        f"depends_on references unknown job {d!r}")
            n = self._conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
            job_id = f"job-{n + 1:04d}"
            while self._conn.execute(
                "SELECT 1 FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone():
                n += 1
                job_id = f"job-{n + 1:04d}"
            row = JobRow(
                job_id=job_id,
                kind=kind,
                dataset_id=dataset_id,
                model_id=model_id,
                state="queued",
                log_path=log_path,
                queued_at=_utcnow(),
                params=params or {},
                depends_on=deps,
            )
            self._conn.execute(
                "INSERT INTO jobs (job_id, kind, dataset_id, model_id, state,"
                " log_path, queued_at, params_json, depends_on_json)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    row.job_id, row.kind, row.dataset_id, row.model_id, row.state,
                    row.log_path, row.queued_at, json.dumps(row.params),
                    json.dumps(row.depends_on) if row.depends_on else None,
                ),
            )
            if actor is not None:
                record(self._conn, actor, "job_queued", row.job_id,
                       before=None,
                       after={"kind": kind, "model_id": model_id,
                              "state": "queued"})
            self._conn.commit()
        return row

    def set_job_log_path(self, job_id: str, log_path: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET log_path = ? WHERE job_id = ?",
                (str(log_path), job_id),
            )
            self._conn.commit()

    def set_job_param(self, job_id: str, key: str, value: Any) -> None:
        """Merge one key into the job's params_json (scheduler device slots)."""
        with self._lock:
            r = self._conn.execute(
                "SELECT params_json FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if r is None:
                return
            try:
                params = json.loads(r[0]) if r[0] else {}
            except (ValueError, TypeError):
                params = {}
            if not isinstance(params, dict):
                params = {}
            params[key] = value
            self._conn.execute(
                "UPDATE jobs SET params_json = ? WHERE job_id = ?",
                (json.dumps(params), job_id),
            )
            self._conn.commit()

    def get_job(self, job_id: str) -> Optional[JobRow]:
        with self._lock:
            r = self._conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        return _row_to_job(r) if r else None

    def list_jobs(self, state: Optional[str] = None) -> List[JobRow]:
        q = "SELECT * FROM jobs"
        args: List[Any] = []
        if state:
            q += " WHERE state = ?"
            args.append(state)
        q += " ORDER BY queued_at, job_id"
        with self._lock:
            rows = self._conn.execute(q, args).fetchall()
        return [_row_to_job(r) for r in rows]

    def next_queued(self) -> Optional[JobRow]:
        """Next claimable queued job (pass4 #43: deps must all be done).

        FIFO over ``queued_at`` among jobs whose every dependency has reached
        ``done``. A job blocked on unfinished work is skipped, not the queue:
        a later independent job still runs while an earlier dependent one
        waits (the PIASystem task_graph "parallel-ready" idea). Jobs whose deps
        went terminal-bad are cascade-failed by ``finish_job``/``cancel_job``,
        so anything still waiting here has live deps."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM jobs WHERE state = 'queued' ORDER BY queued_at, job_id"
            ).fetchall()
            for r in rows:
                deps = _loads_depends_on(r["depends_on_json"]) \
                    if "depends_on_json" in r.keys() else []
                if not deps:
                    return _row_to_job(r)
                placeholders = ",".join("?" * len(deps))
                done = self._conn.execute(
                    f"SELECT COUNT(*) FROM jobs WHERE state = 'done' "
                    f"AND job_id IN ({placeholders})",
                    deps,
                ).fetchone()[0]
                if done == len(deps):
                    return _row_to_job(r)
            return None

    def blocked_queued(self) -> List[JobRow]:
        """Queued jobs waiting on unfinished dependencies (jobs panel #43)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM jobs WHERE state = 'queued' ORDER BY queued_at, job_id"
            ).fetchall()
            out: List[JobRow] = []
            for r in rows:
                deps = _loads_depends_on(r["depends_on_json"]) \
                    if "depends_on_json" in r.keys() else []
                if not deps:
                    continue
                placeholders = ",".join("?" * len(deps))
                done = self._conn.execute(
                    f"SELECT COUNT(*) FROM jobs WHERE state = 'done' "
                    f"AND job_id IN ({placeholders})",
                    deps,
                ).fetchone()[0]
                if done != len(deps):
                    out.append(_row_to_job(r))
            return out

    def _cascade_fail_deps(self, failed_job_id: str, state: str,
                           actor: str) -> None:
        """Fail queued jobs whose deps went terminal-bad (pass4 #43).

        Transitive: a chain A->B->C where A errors cascades to B and then C in
        one pass. Lock-free helper — call with ``self._lock`` held."""
        frontier = [failed_job_id]
        seen = set(frontier)
        while frontier:
            current = frontier.pop()
            rows = self._conn.execute(
                "SELECT job_id, depends_on_json FROM jobs WHERE state = 'queued'"
            ).fetchall()
            frontier = []
            for r in rows:
                deps = _loads_depends_on(r["depends_on_json"]) \
                    if "depends_on_json" in r.keys() else []
                if current not in deps or r["job_id"] in seen:
                    continue
                seen.add(r["job_id"])
                self._conn.execute(
                    "UPDATE jobs SET state = 'error', error = ?, finished_at = ? "
                    "WHERE job_id = ? AND state = 'queued'",
                    (f"dependency {current} {state}", _utcnow(), r["job_id"]),
                )
                record(self._conn, actor, "job_error", r["job_id"],
                       before={"state": "queued"},
                       after={"state": "error",
                              "error": f"dependency {current} {state}"})
                frontier.append(r["job_id"])

    def running_jobs(self) -> List[JobRow]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM jobs WHERE state = 'running' ORDER BY started_at, job_id"
            ).fetchall()
        return [_row_to_job(r) for r in rows]

    def claim_next_queued(self, pid: int, skip_kinds=(),
                          actor: str = "dispatcher") -> Optional[JobRow]:
        """Atomically claim the next FIFO queued job (queued -> running).

        The queue drain's ownership primitive: one guarded UPDATE with a
        ``state = 'queued'`` predicate inside a BEGIN IMMEDIATE transaction
        (same serialization pattern as ledger.record), so two dispatcher
        threads — or a second dashboard process on this DB — can never claim
        the same row. ``pid`` is recorded as a placeholder for the claiming
        process; when a worker subprocess spawns, ``mark_running`` overwrites
        it with the real worker pid, and ``requeue_job`` releases the claim
        only while this same pid still holds it.

        ``skip_kinds`` excludes job kinds some other executor owns (the
        dispatcher never runs ``curate`` — the dashboard thread does).
        Dependency filtering matches ``next_queued``: a job with unfinished
        deps is skipped, not the queue. Returns the claimed row, or None when
        nothing is claimable."""
        skip = [str(k) for k in (skip_kinds or ())]
        with self._lock:
            if not self._conn.in_transaction:
                self._conn.execute("BEGIN IMMEDIATE")
            try:
                q = "SELECT * FROM jobs WHERE state = 'queued'"
                args: List[Any] = []
                if skip:
                    q += " AND kind NOT IN (" + ",".join("?" for _ in skip) + ")"
                    args.extend(skip)
                q += " ORDER BY queued_at, job_id"
                rows = self._conn.execute(q, args).fetchall()
                for r in rows:
                    deps = _loads_depends_on(r["depends_on_json"]) \
                        if "depends_on_json" in r.keys() else []
                    if deps:
                        placeholders = ",".join("?" * len(deps))
                        done = self._conn.execute(
                            f"SELECT COUNT(*) FROM jobs WHERE state = 'done' "
                            f"AND job_id IN ({placeholders})",
                            deps,
                        ).fetchone()[0]
                        if done != len(deps):
                            continue
                    cur = self._conn.execute(
                        "UPDATE jobs SET state = 'running', pid = ?, started_at = ? "
                        "WHERE job_id = ? AND state = 'queued'",
                        (int(pid), _utcnow(), r["job_id"]),
                    )
                    if cur.rowcount == 0:
                        continue  # lost the flip — never overwrite another state
                    record(self._conn, actor, "job_running", r["job_id"],
                           before={"state": "queued"}, after={"state": "running"})
                    fresh = self._conn.execute(
                        "SELECT * FROM jobs WHERE job_id = ?", (r["job_id"],)
                    ).fetchone()
                    self._conn.commit()
                    return _row_to_job(fresh) if fresh else None
                self._conn.commit()
                return None
            except Exception:
                try:
                    self._conn.rollback()
                except sqlite3.Error:  # pragma: no cover - defensive
                    pass
                raise

    def requeue_job(self, job_id: str, pid: int,
                    actor: str = "dispatcher") -> bool:
        """Release a dispatcher claim: running(placeholder pid) back to queued.

        Guarded on ``state = 'running' AND pid = ?`` so it can only undo this
        process's own un-spawned claim — never a genuinely running job (whose
        pid is the worker's) and never a terminal state. Params (including a
        ``_gate_reason`` from a blocked attempt) are preserved."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE jobs SET state = 'queued', pid = NULL, started_at = NULL "
                "WHERE job_id = ? AND state = 'running' AND pid = ?",
                (job_id, int(pid)),
            )
            ok = (cur.rowcount or 0) == 1
            if ok:
                record(self._conn, actor, "job_requeued", job_id,
                       before={"state": "running"}, after={"state": "queued"})
            self._conn.commit()
            return ok

    def mark_running(self, job_id: str, pid: int, actor: str = "scheduler") -> bool:
        """Flip a QUEUED job to running. Returns False when the job is gone or
        no longer queued (e.g. cancelled while queued) so in-process runners
        can abort instead of resurrecting a terminal state.
        """
        with self._lock:
            cur = self._conn.execute(
                "UPDATE jobs SET state = 'running', pid = ?, started_at = ? "
                "WHERE job_id = ? AND state = 'queued'",
                (pid, _utcnow(), job_id),
            )
            if cur.rowcount == 0:
                # Not queued anymore — never overwrite a terminal state.
                self._conn.commit()
                return False
            record(self._conn, actor, "job_running", job_id,
                   before={"state": "queued"}, after={"state": "running"})
            self._conn.commit()
            return True

    def mark_worker_started(self, job_id: str, pid: int,
                            actor: str = "scheduler") -> bool:
        """Record the real worker pid on a dispatcher claim (see
        ``claim_next_queued``). Guarded on ``state = 'running' AND pid = <our
        process>`` so it only upgrades this process's own placeholder claim to
        the spawned worker's pid — never another process's running job, and
        never a terminal state (a cancel landing between claim and spawn wins
        with rowcount 0, letting the caller kill the just-spawned worker)."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE jobs SET pid = ?, started_at = ? "
                "WHERE job_id = ? AND state = 'running' AND pid = ?",
                (pid, _utcnow(), job_id, os.getpid()),
            )
            ok = (cur.rowcount or 0) == 1
            if ok:
                record(self._conn, actor, "job_running", job_id,
                       before={"state": "running"}, after={"state": "running"})
            self._conn.commit()
            return ok

    def finish_job(
        self, job_id: str, state: str, error: Optional[str] = None,
        actor: str = "scheduler",
    ) -> None:
        """Worker-side completion hook (also used by the scheduler's reaper)."""
        if state not in ("done", "error", "cancelled"):
            raise RegistryError(f"bad terminal state {state!r}")
        with self._lock:
            # Idempotent: never resurrect a cancelled job.
            cur = self._conn.execute(
                "SELECT state FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if cur and cur[0] == "cancelled":
                return
            self._conn.execute(
                "UPDATE jobs SET state = ?, error = ?, finished_at = ? WHERE job_id = ?",
                (state, error, _utcnow(), job_id),
            )
            record(self._conn, actor, f"job_{state}", job_id,
                   before={"state": cur[0] if cur else None},
                   after={"state": state, "error": error})
            if state in ("error", "cancelled"):
                # pass4 #43: dependents of failed work never run — fail them
                # now instead of leaving them queued forever.
                self._cascade_fail_deps(job_id, state, actor)
            self._conn.commit()

    def cancel_job(self, job_id: str, actor: str = "cli") -> Optional[JobRow]:
        """Mark a queued/running job cancelled. Caller handles signalling."""
        with self._lock:
            r = self._conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if r is None:
                return None
            if r["state"] in ("queued", "running"):
                self._conn.execute(
                    "UPDATE jobs SET state = 'cancelled', finished_at = ? "
                    "WHERE job_id = ?",
                    (_utcnow(), job_id),
                )
                record(self._conn, actor, "job_cancelled", job_id,
                       before={"state": r["state"]},
                       after={"state": "cancelled"})
                # pass4 #43: cancelling a dep fails its queued dependents too.
                self._cascade_fail_deps(job_id, "cancelled", actor)
                self._conn.commit()
        return self.get_job(job_id)

    # ------------------------------------------------------- ledger (#31)

    def history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Recent ledger entries, newest first (CLI history / overview)."""
        with self._lock:
            return recent_ops(self._conn, limit=limit)

    def verify_ledger_chain(self):
        """Re-walk the hash chain. Returns (ok, first_bad_seq, total)."""
        with self._lock:
            return verify_ledger(self._conn)


def _unique_path(base: Path) -> Path:
    """First free path at/next to ``base``: base, base-2, base-3… (#16).

    Used for archive-destination collisions (two models with the same GGUF
    filename, or archiving the same id twice after a restore).
    """
    if not base.exists():
        return base
    stem, suffix = base.stem, base.suffix
    for i in range(2, 10_000):
        cand = base.with_name(f"{stem}-{i}{suffix}")
        if not cand.exists():
            return cand
    raise RegistryError(f"cannot find a free archive name near {base}")


def days_since_last_use(model: ModelRow,
                        now: Optional[datetime] = None) -> Optional[float]:
    """Days since the model's last activation; None when never used.

    Pass 4 #29 (adapted from PIASystem (MIT) app/core/adapter_manager.py).
    Unparseable timestamps read as None too — callers treat None as "no usage
    signal" and fall back to the age rules (see registry.janitor).
    """
    if not model.last_used_at:
        return None
    try:
        ts = datetime.fromisoformat(model.last_used_at)
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    return max(0.0, (now - ts).total_seconds() / 86400.0)


def _row_to_model(r: sqlite3.Row) -> ModelRow:
    d = dict(r)
    vj = d.pop("validation_json", None)
    validation = json.loads(vj) if vj else None
    # Tolerant reads (backlog #4): rows read from a DB whose models table
    # predates the loss columns simply lack the keys -> default None.
    train_loss = d.pop("train_loss", None)
    eval_loss = d.pop("eval_loss", None)
    # Backlog #11: same tolerance for promoted_at on pre-v4 tables.
    promoted_at = d.pop("promoted_at", None)
    # Backlog #16: pinned + pre_archive_paths tolerant reads on pre-v5 tables.
    pap = d.pop("pre_archive_paths", None)
    pre_archive = json.loads(pap) if pap else None
    pinned = bool(d.pop("pinned", 0))
    # Pass3 #23: provenance_json tolerant read (pre-attestation rows lack it).
    pj = d.pop("provenance_json", None)
    provenance = json.loads(pj) if pj else None
    # Pass4 #29: last_used_at tolerant read on pre-activation tables.
    last_used_at = d.pop("last_used_at", None)
    # Pass4 #32: shadow_json tolerant read (pre-shadow rows lack it).
    sj = d.pop("shadow_json", None)
    shadow = json.loads(sj) if sj else None
    # Pass4 #37: transfers_json tolerant read — missing/corrupt hydrates to
    # [] so a bad blob can never crash row hydration.
    transfers = _loads_transfers(d.pop("transfers_json", None))
    return ModelRow(validation=validation, train_loss=train_loss,
                    eval_loss=eval_loss, promoted_at=promoted_at,
                    pinned=pinned, pre_archive=pre_archive,
                    provenance=provenance, last_used_at=last_used_at,
                    shadow=shadow, transfers=transfers, **d)


def _loads_transfers(raw: Optional[str]) -> List[Dict[str, Any]]:
    """Parse models.transfers_json defensively: anything odd reads as []."""
    if not raw:
        return []
    try:
        v = json.loads(raw)
    except (ValueError, TypeError):
        logger.warning("transfers_json unreadable — treating as empty history")
        return []
    if not isinstance(v, list):
        return []
    return [e for e in v if isinstance(e, dict)]


def _loads_depends_on(raw: Optional[str]) -> List[str]:
    """Tolerant read for jobs.depends_on_json (pass4 #43): missing/corrupt
    hydrates to [] — a job with unreadable deps runs unblocked rather than
    wedging the queue."""
    if not raw:
        return []
    try:
        v = json.loads(raw)
    except (ValueError, TypeError):
        return []
    return [str(x) for x in v] if isinstance(v, list) else []


def _row_to_job(r: sqlite3.Row) -> JobRow:
    d = dict(r)
    pj = d.pop("params_json", None)
    params = json.loads(pj) if pj else {}
    # Pass4 #43: depends_on_json tolerant read (pre-#43 rows lack the column
    # entirely when opened through SELECT-derived dicts — default to []).
    deps = _loads_depends_on(d.pop("depends_on_json", None))
    return JobRow(params=params, depends_on=deps, **d)

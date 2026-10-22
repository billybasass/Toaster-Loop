"""Phase 4 tests — model registry: CRUD, lineage, active pointer, transitions."""

from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sable.registry import (  # noqa: E402
    ModelRegistry,
    RegistryError,
    days_since_last_use,
)


@pytest.fixture()
def registry(tmp_path):
    reg = ModelRegistry(tmp_path / "reg.db")
    yield reg
    reg.close()


def test_register_auto_ids_and_candidate_status(registry):
    m1 = registry.register_model("base-a", dataset_id="d1", manifest_hash="h1",
                                 quant="q4_k_m")
    m2 = registry.register_model("base-a", dataset_id="d2", manifest_hash="h2")
    assert m1.model_id == "v1" and m2.model_id == "v2"
    assert m1.status == "candidate"
    assert m2.parent_adapter is None  # nothing promoted yet -> no lineage edge


def test_artifacts_and_validation_storage(registry):
    m = registry.register_model("base-a")
    registry.set_artifacts(m.model_id, adapter_dir="/tmp/a", gguf_path="/tmp/a.gguf",
                           quant="q5_k_m")
    got = registry.get_model(m.model_id)
    assert got.adapter_dir == "/tmp/a"
    assert got.gguf_path == "/tmp/a.gguf"
    assert got.quant == "q5_k_m"

    registry.set_validation(m.model_id, {"pass_rate": 1.0}, passed=True)
    got = registry.get_model(m.model_id)
    assert got.status == "validated"
    assert got.validation == {"pass_rate": 1.0}


def test_validate_fail_marks_failed(registry):
    m = registry.register_model("base-a")
    registry.set_validation(m.model_id, {"pass_rate": 0.2}, passed=False)
    assert registry.get_model(m.model_id).status == "failed"


def test_promote_requires_validated(registry):
    m = registry.register_model("base-a")
    with pytest.raises(RegistryError):
        registry.promote(m.model_id)  # still candidate


def test_promote_sets_active_pointer_and_supersedes(registry):
    m1 = registry.register_model("base-a")
    registry.set_validation(m1.model_id, {}, passed=True)
    registry.promote(m1.model_id)
    assert registry.get_model("v1").status == "promoted"
    assert registry.active_model().model_id == "v1"

    # second model gains lineage to the active one
    m2 = registry.register_model("base-a")
    assert m2.parent_adapter == "v1"
    registry.set_validation(m2.model_id, {}, passed=True)
    registry.promote(m2.model_id)

    assert registry.get_model("v1").status == "superseded"
    assert registry.get_model("v2").status == "promoted"
    assert registry.active_model().model_id == "v2"


def test_rollback_restores_previous(registry):
    m1 = registry.register_model("base-a")
    registry.set_validation(m1.model_id, {}, passed=True)
    registry.promote("v1")
    m2 = registry.register_model("base-a")
    registry.set_validation(m2.model_id, {}, passed=True)
    registry.promote("v2")

    active = registry.rollback()
    assert active is not None and active.model_id == "v1"
    assert active.status == "promoted"
    assert registry.get_model("v2").status == "rolled_back"
    assert registry.active_model().model_id == "v1"


def test_rollback_with_no_history_clears_pointer(registry):
    registry.register_model("base-a")
    registry.set_validation("v1", {}, passed=True)
    registry.promote("v1")
    active = registry.rollback()
    assert active is None
    assert registry.active_model() is None


def test_rollback_can_repromote_rolled_back(registry):
    registry.register_model("base-a")
    registry.set_validation("v1", {}, passed=True)
    registry.promote("v1")
    registry.rollback()
    # rolled_back models are promotable again (re-deploy prior champion)
    m = registry.promote("v1")
    assert m.status == "promoted"
    assert registry.active_model().model_id == "v1"


def test_per_base_active_pointers(registry):
    registry.register_model("base-a")
    registry.register_model("base-b")
    registry.set_validation("v1", {}, passed=True)
    registry.set_validation("v2", {}, passed=True)
    registry.promote("v1")
    registry.promote("v2")
    assert registry.active_model("base-a").model_id == "v1"
    assert registry.active_model("base-b").model_id == "v2"
    rb = registry.rollback(base_model="base-b")
    assert rb is None  # no previous for base-b
    assert registry.active_model("base-a").model_id == "v1"


def test_list_models_orders_numerically(registry):
    for _ in range(10):
        registry.register_model("base-a")
    ids = [m.model_id for m in registry.list_models()]
    assert ids == [f"v{i}" for i in range(1, 11)]
    # numeric ordering: v9 before v10
    assert ids.index("v9") < ids.index("v10")


# ------------------------------------------------------------------ jobs table

def test_job_enqueue_and_fifo(registry):
    j1 = registry.enqueue_job("sft", dataset_id="d")
    j2 = registry.enqueue_job("export", model_id="v1")
    assert j1.state == "queued" and j2.state == "queued"
    nxt = registry.next_queued()
    # same-second timestamps: insertion order via job_id tiebreak
    assert nxt.job_id == j1.job_id


def test_job_lifecycle_and_cancel(registry):
    j = registry.enqueue_job("sft")
    registry.mark_running(j.job_id, pid=999999)
    assert registry.get_job(j.job_id).state == "running"
    registry.finish_job(j.job_id, "done")
    assert registry.get_job(j.job_id).state == "done"

    j2 = registry.enqueue_job("sft")
    assert registry.cancel_job(j2.job_id).state == "cancelled"
    # finish after cancel must not resurrect
    registry.finish_job(j2.job_id, "done")
    assert registry.get_job(j2.job_id).state == "cancelled"


def test_job_bad_kind_rejected(registry):
    with pytest.raises(RegistryError):
        registry.enqueue_job("banana")


# ------------------------------------------- pass 4 #29: activation tracking

def test_record_activation_updates_row(registry):
    m = registry.register_model("base-a")
    assert m.last_used_at is None
    assert registry.record_activation(m.model_id) is True
    got = registry.get_model(m.model_id)
    assert got.last_used_at is not None
    # re-activation refreshes the stamp (backdate, then bump: second-granular
    # timestamps make direct string comparison flaky, so compare ages instead)
    registry._conn.execute(
        "UPDATE models SET last_used_at = ? WHERE model_id = ?",
        ((datetime.now(timezone.utc) - timedelta(days=5))
         .isoformat(timespec="seconds"), m.model_id),
    )
    registry._conn.commit()
    aged = days_since_last_use(registry.get_model(m.model_id))
    assert aged is not None and 4.9 < aged < 5.2
    assert registry.record_activation(m.model_id) is True
    refreshed = registry.get_model(m.model_id)
    d0 = days_since_last_use(refreshed)
    assert refreshed is not None and d0 is not None and d0 < 1.0


def test_record_activation_unknown_id_safe(registry):
    # Must not raise: called on serve-attach paths where failures cascade.
    assert registry.record_activation("v99") is False


def test_days_since_last_use_semantics(registry):
    m = registry.register_model("base-a")
    # never used -> None ("no usage signal" for the janitor fallback)
    assert days_since_last_use(registry.get_model(m.model_id)) is None

    registry._conn.execute(
        "UPDATE models SET last_used_at = ? WHERE model_id = ?",
        ((datetime.now(timezone.utc) - timedelta(days=3, hours=12))
         .isoformat(timespec="seconds"), m.model_id),
    )
    registry._conn.commit()
    d = days_since_last_use(registry.get_model(m.model_id))
    assert d is not None and 3.4 < d < 3.6

    # unparseable timestamps read as None, never raise
    registry._conn.execute(
        "UPDATE models SET last_used_at = 'not-a-timestamp' WHERE model_id = ?",
        (m.model_id,),
    )
    registry._conn.commit()
    assert days_since_last_use(registry.get_model(m.model_id)) is None


def test_migration_adds_last_used_at_column(tmp_path):
    db = tmp_path / "pre29.db"
    conn = sqlite3.connect(db)
    # Hand-built pre-#29 table: every column up to provenance_json, no last_used_at.
    conn.execute("""
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
            provenance_json TEXT
        )""")
    conn.execute(
        "INSERT INTO models (model_id, base_model, status, created_at) "
        "VALUES ('v1','base-a','superseded','2026-01-01T00:00:00+00:00')")
    conn.commit()
    conn.close()

    reg = ModelRegistry(db)
    try:
        cols = {r[1] for r in reg._conn.execute("PRAGMA table_info(models)")}
        assert "last_used_at" in cols
        m = reg.get_model("v1")
        assert m is not None and m.last_used_at is None  # tolerant read
        assert reg.record_activation("v1") is True
        after = reg.get_model("v1")
        assert after is not None and after.last_used_at is not None
    finally:
        reg.close()

    # idempotent: reopening an already-migrated DB must not fail
    reg2 = ModelRegistry(db)
    try:
        again = reg2.get_model("v1")
        assert again is not None and again.last_used_at is not None
    finally:
        reg2.close()

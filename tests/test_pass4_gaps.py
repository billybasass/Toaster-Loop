"""Pass-4 gap-closure tests: #43 job dependency edges and #34 RankScaler.

#43 (idea adapted from PIASystem app/core/task_graph.py): jobs gain a
``depends_on`` edge list — unclaimable until every dep is done, cascade-failed
when a dep ends error/cancelled, validated at enqueue.

#34 (idea adapted from PIASystem app/core/training_engine.py RankScaler): pure
loss-stream watcher + zero-padded adapter seeding; torch-free tests drive the
scaler directly and parse rank_history out of worker logs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sable.registry import ModelRegistry  # noqa: E402
from sable.registry.registry import RegistryError  # noqa: E402
from sable.training.pipeline import parse_worker_rank_history  # noqa: E402
from sable.training.worker import (  # noqa: E402
    RankScaler,
    rank_scaling_from_config,
)


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "ufgap"
    monkeypatch.setenv("SABLE_HOME", str(h))
    return h


@pytest.fixture()
def registry(home):
    reg = ModelRegistry(home / "buffer" / "sable.db")
    yield reg
    reg.close()


# ================================================================ #43 registry

def test_enqueue_rejects_unknown_dep(registry):
    with pytest.raises(RegistryError, match="unknown job"):
        registry.enqueue_job("sft", depends_on=["job-9999"])


def test_dep_blocks_next_queued_until_done(registry):
    a = registry.enqueue_job("sft")
    b = registry.enqueue_job("export", depends_on=[a.job_id])
    # A later independent job is claimable while the dependent one waits.
    c = registry.enqueue_job("dpo")
    assert registry.next_queued().job_id == a.job_id
    registry.finish_job(a.job_id, "error", error="boom")
    # b cascade-failed with the dep reason; c still runs.
    rb = registry.get_job(b.job_id)
    assert rb.state == "error"
    assert "dependency" in (rb.error or "") and a.job_id in (rb.error or "")
    assert registry.next_queued().job_id == c.job_id


def test_dep_unblocks_on_success(registry):
    a = registry.enqueue_job("sft")
    b = registry.enqueue_job("export", depends_on=[a.job_id])
    assert registry.next_queued().job_id == a.job_id
    registry.finish_job(a.job_id, "done")
    assert registry.next_queued().job_id == b.job_id
    assert registry.blocked_queued() == []


def test_blocked_queued_lists_waiting_jobs(registry):
    a = registry.enqueue_job("sft")
    b = registry.enqueue_job("export", depends_on=[a.job_id])
    blocked = registry.blocked_queued()
    assert [j.job_id for j in blocked] == [b.job_id]
    assert blocked[0].depends_on == [a.job_id]


def test_transitive_cascade(registry):
    a = registry.enqueue_job("sft")
    b = registry.enqueue_job("export", depends_on=[a.job_id])
    c = registry.enqueue_job("validate", depends_on=[b.job_id])
    registry.finish_job(a.job_id, "error", error="boom")
    for jid in (b.job_id, c.job_id):
        row = registry.get_job(jid)
        assert row.state == "error"


def test_cancel_cascades(registry):
    a = registry.enqueue_job("sft")
    b = registry.enqueue_job("export", depends_on=[a.job_id])
    registry.cancel_job(a.job_id)
    rb = registry.get_job(b.job_id)
    assert rb.state == "error"
    assert "cancelled" in (rb.error or "")


def test_depends_on_roundtrips_through_row(registry):
    a = registry.enqueue_job("sft")
    b = registry.enqueue_job("export", depends_on=[a.job_id])
    got = registry.get_job(b.job_id)
    assert got.depends_on == [a.job_id]
    d = got.to_dict()
    assert d["depends_on"] == [a.job_id]


def test_legacy_db_migration_adds_column(home):
    """Pre-#43 jobs table (no depends_on_json) upgrades in place."""
    import sqlite3

    db = home / "buffer" / "sable.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE jobs (
            job_id TEXT PRIMARY KEY,
            kind TEXT NOT NULL CHECK (kind IN ('sft','dpo','grpo','export',
              'validate','curate','shadow_eval','evolve')),
            dataset_id TEXT, model_id TEXT,
            state TEXT NOT NULL CHECK (state IN ('queued','running','done',
              'error','cancelled')),
            log_path TEXT, pid INTEGER, queued_at TEXT NOT NULL,
            started_at TEXT, finished_at TEXT, error TEXT, params_json TEXT
        );
        INSERT INTO jobs (job_id, kind, state, queued_at)
        VALUES ('job-0001', 'sft', 'done', '2026-01-01T00:00:00+00:00');
        """
    )
    conn.commit()
    conn.close()

    reg = ModelRegistry(db)
    try:
        row = reg.get_job("job-0001")
        assert row is not None and row.depends_on == []
        b = reg.enqueue_job("export", depends_on=["job-0001"])
        nxt = reg.next_queued()
        assert nxt is not None and nxt.job_id == b.job_id  # dep already done
    finally:
        reg.close()


# ============================================================ #43 scheduler

STUB_WORKER = r'''
import argparse, json, os, sys
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("--job", required=True)
ap.add_argument("--kind", required=True)
ap.add_argument("--config", default="")
ap.add_argument("--dataset-dir", default=None)
ap.add_argument("--base-model", default=None)
ap.add_argument("--log-path", default=None)
ap.add_argument("--adapter-dir", default=None)
ap.add_argument("--gguf-out", default=None)
ap.add_argument("--quant", default="q4_k_m")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()
def emit(ev, **kw):
    line = json.dumps({"event": ev, "job": a.job, **kw})
    print(line, flush=True)
    if a.log_path:
        Path(a.log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(a.log_path, "a") as f:
            f.write(line + "\n")
emit("start"); emit("done")
'''


@pytest.fixture()
def stub(tmp_path):
    p = tmp_path / "stub_worker.py"
    p.write_text(STUB_WORKER)
    return str(p)


def test_run_job_refuses_blocked_job(registry, stub):
    from sable.training.scheduler import Scheduler, SchedulerError, TrainingConfig

    sched = Scheduler(registry=registry, cfg=TrainingConfig(),
                      worker_command=[sys.executable, stub])
    a = sched.submit("sft")
    b = sched.submit("export", depends_on=[a.job_id])
    with pytest.raises(SchedulerError, match="waits on dependencies"):
        sched.run_job(registry.get_job(b.job_id))
    # Still queued (not corrupted), and runs once the dep finishes.
    assert registry.get_job(b.job_id).state == "queued"
    registry.finish_job(a.job_id, "done")
    done = sched.run_job(registry.get_job(b.job_id))
    assert done.state == "done"


def test_run_pending_respects_edges(registry, stub):
    from sable.training.scheduler import Scheduler, TrainingConfig

    sched = Scheduler(registry=registry, cfg=TrainingConfig(),
                      worker_command=[sys.executable, stub])
    a = sched.submit("sft")
    b = sched.submit("export", depends_on=[a.job_id])
    finished = sched.run_pending()
    ids = [j.job_id for j in finished]
    assert ids == [a.job_id, b.job_id]  # topological despite submit order
    assert registry.get_job(b.job_id).state == "done"


# ================================================================= #34 scaler

def test_rank_scaling_disabled_by_default():
    sc = rank_scaling_from_config({})
    assert not sc.enabled
    assert sc.observe(1, 1.0) is False
    assert sc.plan_growth() is None


def test_rank_scaling_parses_mapping_knob():
    sc = rank_scaling_from_config({"training": {"rank_scaling": {
        "max_rank": 128, "min_improvement_steps": 2,
        "min_rel_improvement": 0.05}}})
    assert sc.enabled and sc.max_rank == 128
    assert sc.min_improvement_steps == 2
    assert abs(sc.min_rel_improvement - 0.05) < 1e-9


def test_scaler_triggers_after_consecutive_improvements():
    sc = RankScaler(enabled=True, min_improvement_steps=3,
                    min_rel_improvement=0.01)
    sc.start_stage(64)
    assert sc.observe(1, 1.00) is False   # seeds best
    assert sc.observe(2, 0.98) is False   # -2% (streak 1)
    assert sc.observe(3, 0.96) is False   # streak 2
    assert sc.observe(4, 0.94) is True    # streak 3 -> trigger
    assert sc.plan_growth() == 128


def test_scaler_streak_resets_on_non_improvement():
    sc = RankScaler(enabled=True, min_improvement_steps=3)
    sc.start_stage(64)
    sc.observe(1, 1.0)
    sc.observe(2, 0.98)
    assert sc.observe(3, 0.99) is False   # streak reset
    assert sc.streak == 0
    assert sc.observe(4, 0.95) is False   # one improvement again
    assert sc.triggered is False


def test_scaler_caps_at_max_rank():
    sc = RankScaler(enabled=True, max_rank=64, min_improvement_steps=1)
    sc.start_stage(64)
    assert sc.plan_growth() is None       # already at cap
    assert sc.observe(1, 1.0) is False
    assert sc.observe(2, 0.5) is False    # trigger impossible below cap


def test_scaler_records_growth_history():
    sc = RankScaler(enabled=True, min_improvement_steps=1)
    sc.start_stage(64)
    sc.observe(1, 1.0)
    assert sc.observe(2, 0.5) is True
    entry = sc.record_growth(128, step=2)
    assert entry == {"from_r": 64, "to_r": 128, "step": 2, "best_loss": 0.5}
    assert sc.rank_history == [entry]


def test_parse_worker_rank_history(tmp_path):
    log = tmp_path / "job.log"
    lines = [
        {"event": "progress", "step": 1, "loss": 1.0},
        {"event": "sft_done", "train_loss": 0.5, "eval_loss": None,
         "rank_history": [{"from_r": 64, "to_r": 128, "step": 9,
                           "best_loss": 0.5}]},
    ]
    log.write_text("\n".join(json.dumps(r) for r in lines))
    assert parse_worker_rank_history(str(log)) == lines[1]["rank_history"]

    # No history field => [] (scaling off is the norm).
    log.write_text(json.dumps({"event": "sft_done", "train_loss": 0.5}))
    assert parse_worker_rank_history(str(log)) == []
    assert parse_worker_rank_history(None) == []
    assert parse_worker_rank_history("/nonexistent/path") == []

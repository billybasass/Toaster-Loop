"""Pass 4 #32 — shadow-baseline evaluation before promotion.

Covers: replay-set selection (positive-feedback recency, determinism, infra
exclusion, honest-empty), comparison math + regression thresholds incl. the
config knob, end-to-end job execution with two fake servers + stub judge
(verdict stored on the registry row), scheduler claim path with the VRAM
gate, and the promotion warn/block wiring in the pipeline.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fake_server import FakeOpenAIServer, make_scripted_handler  # noqa: E402
from sable.evaluation.shadow import (  # noqa: E402
    NO_REPLAY_SET,
    ShadowError,
    build_replay_set,
    compare_scores,
    run_shadow_eval,
    shadow_promotion_check,
)
from sable.registry import ModelRegistry  # noqa: E402
from sable.store.db import TraceRow, TraceStore  # noqa: E402
from sable.training.scheduler import Scheduler, SchedulerError, TrainingConfig  # noqa: E402


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "uf-shadow"
    monkeypatch.setenv("SABLE_HOME", str(h))
    (h / "buffer").mkdir(parents=True)
    return h


@pytest.fixture()
def store(home):
    s = TraceStore(home / "buffer" / "sable.db")
    yield s
    s.close()


# ------------------------------------------------------------------ seeding

from typing import Optional  # noqa: E402


def seed_trace(store, home, tid, *, ts, profile="coder", status="success",
               score: Optional[float] = 0.9, failure_reason=None, user_fb=None,
               task=None, reply="the old answer"):
    p = home / "buffer" / f"{tid}.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        f.write(json.dumps({"kind": "trace_header"}) + "\n")
        f.write(json.dumps({"kind": "message", "message": {
            "role": "system", "content": "sp"}}) + "\n")
        f.write(json.dumps({"kind": "message", "message": {
            "role": "user", "content": task or f"task {tid}"}}) + "\n")
        f.write(json.dumps({"kind": "message", "message": {
            "role": "assistant", "content": reply}}) + "\n")
    store.insert_trace(TraceRow(
        trace_id=tid, session_id=None, timestamp=ts, agent_profile=profile,
        model_alias="m", model_version=None, endpoint_label=None,
        task_summary="s", status=status, outcome_score=score,
        num_tool_calls=0, tokens_in=None, tokens_out=None, duration_s=None,
        jsonl_path=str(p), failure_reason=failure_reason,
    ))
    if user_fb is not None:
        store.add_feedback(tid, user_fb, None, source="user", timestamp=ts)


# ------------------------------------------------------- replay-set selection

def test_replay_set_picks_positive_recent_deterministically(store, home):
    seed_trace(store, home, "t1", ts="2026-09-01T00:00:00+00:00", score=0.9)
    seed_trace(store, home, "t2", ts="2026-09-02T00:00:00+00:00", score=0.1)
    seed_trace(store, home, "t3", ts="2026-09-03T00:00:00+00:00", score=None,
               user_fb=0.8)  # positive via USER feedback despite NULL heuristic
    seed_trace(store, home, "t4", ts="2026-09-04T00:00:00+00:00",
               profile="other", score=0.9)

    rs = build_replay_set(store, "coder", n=8)
    assert [r["trace_id"] for r in rs] == ["t3", "t1"]  # ts desc; t2 low-score
    assert rs[0]["task"] == "task t3"
    assert rs[0]["reference_response"] == "the old answer"

    # deterministic across calls + n cap
    assert build_replay_set(store, "coder", n=8) == rs
    assert [r["trace_id"] for r in build_replay_set(store, "coder", n=1)] == ["t3"]


def test_replay_set_excludes_infra_and_failures(store, home):
    seed_trace(store, home, "ok", ts="2026-09-05T00:00:00+00:00")
    seed_trace(store, home, "infra", ts="2026-09-06T00:00:00+00:00",
               status="partial", failure_reason="backend_error:connection")
    seed_trace(store, home, "fail", ts="2026-09-07T00:00:00+00:00",
               status="failure")
    rs = build_replay_set(store, "coder", n=8)
    assert [r["trace_id"] for r in rs] == ["ok"]


def test_replay_set_honest_empty(store, home):
    seed_trace(store, home, "t1", ts="2026-09-01T00:00:00+00:00", score=0.05)
    assert build_replay_set(store, "coder", n=8) == []
    assert build_replay_set(store, "nobody", n=8) == []


# ------------------------------------------------------------ compare math

def test_compare_scores_math_and_threshold():
    per = [{"candidate_score": 0.9, "baseline_score": 0.5},
           {"candidate_score": 0.7, "baseline_score": 0.6}]
    c = compare_scores(per)
    assert c["candidate_mean"] == pytest.approx(0.8)
    assert c["baseline_mean"] == pytest.approx(0.55)
    assert c["improvement"] == pytest.approx(0.25)
    assert c["is_regression"] is False
    assert c["regression_threshold"] == pytest.approx(-0.05)

    reg = compare_scores([{"candidate_score": 0.4, "baseline_score": 0.5}])
    assert reg["improvement"] == pytest.approx(-0.1)
    assert reg["is_regression"] is True  # -0.1 < -0.05 (default 5%)

    # knob override: 20% tolerance makes the same delta a non-regression
    loose = compare_scores([{"candidate_score": 0.4, "baseline_score": 0.5}],
                           max_regression_pct=20.0)
    assert loose["is_regression"] is False

    # None scores are skipped, means stay honest (never coerced to zero)
    mixed = compare_scores([{"candidate_score": None, "baseline_score": 0.5},
                            {"candidate_score": 0.3, "baseline_score": None}])
    assert mixed["skipped"] == 2
    assert mixed["improvement"] is None
    assert mixed["is_regression"] is False

    empty = compare_scores([])
    assert empty["n"] == 0 and empty["candidate_mean"] is None


# --------------------------------------------------------------- e2e job

def _stub_judge_client_for(score_map):
    """judge_client_for seam: score by substring of the judged transcript."""
    def client_for(base_url, model_name):
        def client(messages, model):
            text = json.dumps(messages)
            for needle, sc in score_map.items():
                if needle in text:
                    return json.dumps({"score": sc, "reason": "stub"})
            return json.dumps({"score": 0.5, "reason": "stub-mid"})
        return client
    return client_for


def test_run_shadow_eval_end_to_end_stores_comparison(home):
    store = TraceStore(home / "buffer" / "sable.db")
    reg = ModelRegistry(home / "buffer" / "sable.db")
    seed_trace(store, home, "t1", ts="2026-09-01T00:00:00+00:00")
    seed_trace(store, home, "t2", ts="2026-09-02T00:00:00+00:00")

    base = reg.register_model("Base/Model")            # v1 -> active
    reg.set_validation(base.model_id, {"ok": True}, passed=True)
    reg.promote(base.model_id)
    cand = reg.register_model("Base/Model")            # v2 candidate

    # two fake servers responding differently per side (spec requirement);
    # the judge stub grades "candidate reply" high and "baseline reply" low.
    cand_srv = FakeOpenAIServer(handler_cls=make_scripted_handler(
        {"*": "candidate reply: excellent grounded answer"})).start()
    base_srv = FakeOpenAIServer(handler_cls=make_scripted_handler(
        {"*": "baseline reply: weak unhelpful mush"})).start()

    @contextmanager
    def endpoint_for(model):
        srv = cand_srv if model.model_id == cand.model_id else base_srv
        yield srv.base_url, model.model_id

    try:
        result = run_shadow_eval(
            reg, store, candidate_id=cand.model_id, profile="coder", n=8,
            home=home, endpoint_for=endpoint_for,
            judge_client_for=_stub_judge_client_for(
                {"excellent grounded": 0.9, "weak unhelpful": 0.2}),
        )
    finally:
        cand_srv.stop()
        base_srv.stop()

    assert result["baseline_id"] == base.model_id
    assert result["n"] == 2
    assert result["candidate_mean"] == pytest.approx(0.9)
    assert result["baseline_mean"] == pytest.approx(0.2)
    assert result["improvement"] == pytest.approx(0.7)
    assert result["is_regression"] is False
    # judge calls bounded: 2 per task => exactly n candidate + n baseline rows
    assert len(result["per_task"]) == 2

    row = reg.get_model(cand.model_id)
    assert row is not None
    stored = row.shadow
    assert stored is not None and stored["baseline_id"] == base.model_id
    assert set(stored) >= {"ts", "baseline_id", "n", "candidate_mean",
                           "baseline_mean", "improvement", "is_regression",
                           "per_task", "judge_model"}

    # verdict storing is ledgered (pass4 #31 actor plumbing)
    ops = [e["op"] for e in reg.history(limit=10)]
    assert "set_shadow" in ops
    store.close()
    reg.close()


def test_run_shadow_eval_regression_flag_and_honest_skips(home):
    store = TraceStore(home / "buffer" / "sable.db")
    reg = ModelRegistry(home / "buffer" / "sable.db")
    seed_trace(store, home, "t1", ts="2026-09-01T00:00:00+00:00")
    base = reg.register_model("Base/Model")
    reg.set_validation(base.model_id, {"ok": True}, passed=True)
    reg.promote(base.model_id)
    cand = reg.register_model("Base/Model")

    @contextmanager
    def endpoint_for(model):
        yield "http://127.0.0.1:1/v1", model.model_id

    gen_calls = {"n": 0}

    def generate(base_url, model_name, task):
        gen_calls["n"] += 1
        return "CANDIDATE" if "v2" in model_name else "BASELINE"

    result = run_shadow_eval(
        reg, store, candidate_id=cand.model_id, profile="coder",
        home=home, endpoint_for=endpoint_for, generate_fn=generate,
        judge_client_for=_stub_judge_client_for({"CANDIDATE": 0.2,
                                                 "BASELINE": 0.9}),
    )
    assert result["is_regression"] is True
    # bounded: exactly one generate + one judge per side per task = 2xN
    assert gen_calls["n"] == 2

    # no replay set -> honest skip, nothing stored
    reg2_home = home.parent / "empty-home"
    reg2_home.mkdir()
    (reg2_home / "buffer").mkdir()
    store2 = TraceStore(reg2_home / "buffer" / "sable.db")
    reg2 = ModelRegistry(reg2_home / "buffer" / "sable.db")
    m = reg2.register_model("Base/Model")
    skip = run_shadow_eval(reg2, store2, candidate_id=m.model_id,
                           profile="coder", home=reg2_home)
    assert skip == {"skipped": True, "reason": NO_REPLAY_SET}
    row_m = reg2.get_model(m.model_id)
    assert row_m is not None and row_m.shadow is None

    # no baseline (candidate itself active) -> honest skip too
    reg2.set_validation(m.model_id, {"ok": True}, passed=True)
    reg2.promote(m.model_id)
    skip2 = run_shadow_eval(reg2, store2, candidate_id=m.model_id,
                            profile="coder", home=reg2_home,
                            endpoint_for=endpoint_for)
    assert skip2["skipped"] is True
    store.close(); reg.close(); store2.close(); reg2.close()


def test_run_shadow_eval_all_skipped_raises(home):
    """Zero scored tasks must be an error, never a fake 'no regression'."""
    store = TraceStore(home / "buffer" / "sable.db")
    reg = ModelRegistry(home / "buffer" / "sable.db")
    seed_trace(store, home, "t1", ts="2026-09-01T00:00:00+00:00")
    base = reg.register_model("Base/Model")
    reg.set_validation(base.model_id, {"ok": True}, passed=True)
    reg.promote(base.model_id)
    cand = reg.register_model("Base/Model")

    @contextmanager
    def endpoint_for(model):  # never reached: generation fails first
        yield "http://127.0.0.1:1/v1", model.model_id

    with pytest.raises(ShadowError, match="zero scored"):
        run_shadow_eval(reg, store, candidate_id=cand.model_id,
                        profile="coder", home=home, endpoint_for=endpoint_for,
                        generate_fn=lambda *a: None)  # every reply fails
    store.close()
    reg.close()


# --------------------------------------------------------- scheduler path

def test_scheduler_claims_shadow_eval_respects_vram_gate(home, monkeypatch):
    reg = ModelRegistry(home / "buffer" / "sable.db")
    ran = []

    def runner(job):
        ran.append(job.job_id)
        return {"ok": True}

    sched = Scheduler(registry=reg, cfg=TrainingConfig(min_free_vram_mib=1),
                      inprocess_runners={"shadow_eval": runner})
    job = sched.submit("shadow_eval", model_id="v9", enqueue_actor="cli")
    # enqueue itself is ledgered when an actor is passed (#32)
    hist = reg.history(limit=5)
    assert hist[0]["op"] == "job_queued"

    # VRAM gate closed -> job stays queued with the reason recorded
    monkeypatch.setattr("sable.training.scheduler.gpu_free_mib", lambda: 1)
    sched.cfg.min_free_vram_mib = 100_000
    with pytest.raises(SchedulerError, match="resource gate"):
        sched.run_job(job)
    jrow = reg.get_job(job.job_id)
    assert jrow is not None and jrow.state == "queued"
    assert ran == []

    # gate open -> in-process runner executes, job completes
    monkeypatch.setattr("sable.training.scheduler.gpu_free_mib", lambda: 10**7)
    done = sched.run_pending()
    assert [j.state for j in done] == ["done"]
    assert ran == [job.job_id]
    reg.close()


def test_scheduler_shadow_without_runner_fails_honestly(home, monkeypatch):
    monkeypatch.setattr("sable.training.scheduler.gpu_free_mib", lambda: None)
    reg = ModelRegistry(home / "buffer" / "sable.db")
    sched = Scheduler(registry=reg, cfg=TrainingConfig(),
                      inprocess_runners={})  # no shadow_eval runner
    # also disable the lazy default by pointing the module import at nothing:
    job = sched.submit("shadow_eval", model_id="v9")
    import sable.evaluation.shadow as sh
    monkeypatch.setattr(sh, "make_shadow_job_runner",
                        lambda *a, **k: (_ for _ in ()).throw(
                            ImportError("no store")))
    done = sched.run_pending()
    assert done[0].state == "error"
    reg.close()


# ------------------------------------------------------------ promotion gate

def test_shadow_promotion_check_semantics():
    assert shadow_promotion_check(None) == {"warned": False,
                                            "blocked": False, "message": None}
    assert shadow_promotion_check({"skipped": True})["warned"] is False
    good = {"is_regression": False, "baseline_id": "v1", "n": 4,
            "improvement": 0.1, "regression_threshold": -0.05}
    assert shadow_promotion_check(good)["message"] is None

    bad = {"is_regression": True, "baseline_id": "v1", "n": 4,
           "improvement": -0.2, "regression_threshold": -0.05}
    warn = shadow_promotion_check(bad)
    assert warn["warned"] is True and warn["blocked"] is False
    assert "v1" in warn["message"]

    strict = shadow_promotion_check(bad, blocks=True)
    assert strict["blocked"] is True


def test_iterate_promote_warns_then_blocks_with_strict_flag(home, tmp_path,
                                                            monkeypatch):
    """Regression verdict: default warn still promotes; strict flag blocks."""
    from sable.curate import CurationConfig
    from sable.training.pipeline import run_iterate
    from sable.training.validate import ValidationConfig

    stub = tmp_path / "stub.py"
    stub.write_text(
        "import argparse, json, sys\n"
        "from pathlib import Path\n"
        "ap = argparse.ArgumentParser()\n"
        "for f in (('--job', True), ('--kind', True)):\n"
        "    ap.add_argument(f[0], required=f[1])\n"
        "for f in ('--config','--dataset-dir','--base-model','--log-path',"
        "'--adapter-dir','--gguf-out'):\n"
        "    ap.add_argument(f, default=None)\n"
        "ap.add_argument('--quant', default='q4_k_m')\n"
        "ap.add_argument('--dry-run', action='store_true')\n"
        "a = ap.parse_args()\n"
        "if a.kind in ('sft','dpo'):\n"
        "    out = Path(a.config).parent\n"
        "    (out/'adapter').mkdir(parents=True, exist_ok=True)\n"
        "    (out/'adapter'/'adapter_config.json').write_text('{}')\n"
        "elif a.kind == 'export':\n"
        "    gg = Path(a.gguf_out); gg.parent.mkdir(parents=True, exist_ok=True)\n"
        "    gg.write_bytes(b'GGUF-stub')\n")
    store = TraceStore(home / "buffer" / "sable.db")
    reg = ModelRegistry(home / "buffer" / "sable.db")
    seed_trace(store, home, "t1", ts="2026-09-01T00:00:00+00:00",
               profile="fallback")

    def starter(gguf_path):
        srv = FakeOpenAIServer(handler_cls=make_scripted_handler(
            {"*": "ok-answer"}))
        srv.start()
        return srv

    verdict = {"is_regression": True, "baseline_id": "v0", "n": 3,
               "improvement": -0.2, "regression_threshold": -0.05}
    monkeypatch.setattr(
        "sable.evaluation.shadow.shadow_promotion_check",
        lambda shadow, *, blocks=False: {
            "warned": not blocks, "blocked": blocks,
            "message": "shadow regressed (test)"})

    def go(strict):
        # The stored-shadow plumbing is covered by the e2e test above; here we
        # drive the promote-stage gate directly: shadow_promotion_check is
        # imported inside the stage, so patching the source module works.
        sched = Scheduler(registry=reg, cfg=TrainingConfig(base_model="B/M"),
                          worker_command=[sys.executable, str(stub)])
        return run_iterate(
            home=home, registry=reg, scheduler=sched, store=store,
            curation_cfg=CurationConfig(min_score=0.5),
            training_cfg=TrainingConfig(base_model="B/M"),
            validation_cfg=ValidationConfig(
                min_pass_rate=1.0,
                golden_tasks=[{"prompt": "q", "expect_contains": ["ok-answer"]}],
                shadow_blocks_promotion=strict),
            name="sh", server_starter=starter)

    r = go(strict=False)
    assert r.ok and r.promoted, r.to_dict()
    assert (r.validation or {}).get("shadow_gate", {}).get("warned") is True

    # strict: next candidate blocked at promote
    seed_trace(store, home, "t2", ts="2026-09-02T00:00:00+00:00",
               profile="fallback")
    r2 = go(strict=True)
    assert not r2.ok and r2.stage == "promote"
    assert "shadow regressed" in (r2.error or "")
    store.close()
    reg.close()


# ---------------------------------------------------------------- migration

def test_shadow_migration_on_old_db_shapes(tmp_path):
    db = tmp_path / "old.db"
    reg = ModelRegistry(db)
    reg.register_model("B/M")
    reg.close()

    # Simulate a pre-#32 DB: drop the shadow column and rebuild jobs with the
    # old kind CHECK (as _migrate_job_kinds-era tables looked).
    conn = sqlite3.connect(str(db))
    conn.execute("ALTER TABLE models DROP COLUMN shadow_json")
    conn.execute("ALTER TABLE jobs RENAME TO jobs_old_x")
    conn.execute(
        "CREATE TABLE jobs (job_id TEXT PRIMARY KEY,"
        " kind TEXT NOT NULL CHECK (kind IN"
        " ('sft','dpo','grpo','export','validate','curate')),"
        " dataset_id TEXT, model_id TEXT, state TEXT NOT NULL CHECK (state IN"
        " ('queued','running','done','error','cancelled')), log_path TEXT,"
        " pid INTEGER, queued_at TEXT NOT NULL, started_at TEXT, finished_at TEXT,"
        " error TEXT, params_json TEXT)")
    # Explicit column list (not SELECT *): the live table may carry newer
    # columns (#43 depends_on_json) than this simulated pre-#32 shape.
    conn.execute(
        "INSERT INTO jobs (job_id, kind, dataset_id, model_id, state,"
        " log_path, pid, queued_at, started_at, finished_at, error,"
        " params_json)"
        " SELECT job_id, kind, dataset_id, model_id, state, log_path, pid,"
        " queued_at, started_at, finished_at, error, params_json"
        " FROM jobs_old_x")
    conn.execute("DROP TABLE jobs_old_x")
    conn.commit()
    conn.close()

    reg2 = ModelRegistry(db)  # runs _migrate_shadow_column + kind rebuild
    try:
        m = reg2.get_model("v1")
        assert m is not None and m.shadow is None  # tolerant read
        job = reg2.enqueue_job("shadow_eval", model_id="v1")  # CHECK accepts
        assert job.state == "queued"
        reg2.set_shadow("v1", {"is_regression": False, "n": 1})
        row = reg2.get_model("v1")
        assert row is not None and row.shadow is not None
        assert row.shadow["n"] == 1
    finally:
        reg2.close()

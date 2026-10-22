"""Backlog item #4 — eval split, progress/eval events, registry losses, gate.

The worker can't run real training in CI (no GPU/torch), so everything here
targets the pure functions extracted from training/worker.py (split, cadence,
event formatting, protocol callback driven with fake trainer state) plus the
registry migration and the pipeline promotion gate (stub worker + fake server,
same pattern as tests/test_iterate.py).
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sable.registry.registry import ModelRegistry  # noqa: E402
from sable.training.pipeline import (  # noqa: E402
    eval_regression_check,
    parse_worker_losses,
)
from sable.training.worker import (  # noqa: E402
    EVAL_MIN_ROWS,
    EVAL_MIN_TOTAL_ROWS,
    eval_event,
    find_named_eval_files,
    find_resumable_checkpoint,
    make_protocol_callback,
    maybe_eval_split,
    plan_cadence,
    progress_event,
    resolve_resume_path,
)


def _mk_ckpt(root: Path, step: int, *, weights=True, state=True,
             global_step=None, max_steps=None):
    """Create a checkpoint-<step> dir with the requested contents."""
    ck = root / f"checkpoint-{step}"
    ck.mkdir(parents=True)
    if weights:
        (ck / "adapter_model.safetensors").write_text("w")
    if state:
        gs = step if global_step is None else global_step
        ms = (step * 2) if max_steps is None else max_steps
        (ck / "trainer_state.json").write_text(
            json.dumps({"global_step": gs, "max_steps": ms}))
    return ck


# ------------------------------------------------------------- eval split

def test_no_split_for_tiny_datasets():
    rows = list(range(19))  # below the 20-row floor
    train, ev = maybe_eval_split(rows, 0.1, 42)
    assert ev is None
    assert sorted(train) == rows


def test_no_split_when_eval_side_too_small():
    rows = list(range(25))
    train, ev = maybe_eval_split(rows, 0.05, 42)  # 1 eval row < min 2
    assert ev is None
    train2, ev2 = maybe_eval_split(rows, 0.1, 42)  # 2 eval rows -> splits
    assert ev2 is not None and len(ev2) == 2 and len(train2) == 23


def test_no_split_when_fraction_zero_or_negative():
    rows = list(range(100))
    train, ev = maybe_eval_split(rows, 0.0, 42)
    assert ev is None and len(train) == 100


def test_split_is_deterministic_and_disjoint():
    rows = [{"i": i} for i in range(50)]
    t1, e1 = maybe_eval_split(rows, 0.2, seed=42)
    t2, e2 = maybe_eval_split(rows, 0.2, seed=42)
    assert e1 is not None and e2 is not None
    assert [r["i"] for r in t1] == [r["i"] for r in t2]
    assert [r["i"] for r in e1] == [r["i"] for r in e2]
    ids_t = {r["i"] for r in t1}
    ids_e = {r["i"] for r in e1}
    assert not (ids_t & ids_e)              # disjoint
    assert ids_t | ids_e == set(range(50))  # complete
    assert len(e1) == 10                    # 20% of 50


def test_split_seed_changes_the_split():
    rows = list(range(50))
    _, e1 = maybe_eval_split(rows, 0.2, seed=42)
    _, e2 = maybe_eval_split(rows, 0.2, seed=7)
    assert e1 is not None and e2 is not None
    assert sorted(e1) != sorted(e2)


def test_split_preserves_row_order_in_eval():
    rows = list(range(40))
    _, ev = maybe_eval_split(rows, 0.25, seed=42)
    assert ev is not None
    assert ev == sorted(ev)


def test_split_floor_policy_constants_are_declared():
    # Inspiration pass 3 #22: the floors are declared policy, and the default
    # arguments of maybe_eval_split ARE those constants (one source of truth).
    import inspect
    sig = inspect.signature(maybe_eval_split)
    assert sig.parameters["min_total"].default == EVAL_MIN_TOTAL_ROWS
    assert sig.parameters["min_eval"].default == EVAL_MIN_ROWS
    assert EVAL_MIN_TOTAL_ROWS > EVAL_MIN_ROWS > 0


# ------------------------------------------------- named eval splits (#22)

def test_named_eval_autodetect_next_to_train_files(tmp_path):
    train = tmp_path / "sft.jsonl"
    train.write_text("{}\n")
    ev = tmp_path / "eval.jsonl"
    ev.write_text("{}\n")
    assert find_named_eval_files({}, [str(train)]) == [str(ev)]


def test_named_eval_explicit_data_sec_wins(tmp_path):
    train = tmp_path / "sft.jsonl"
    train.write_text("{}\n")
    (tmp_path / "eval.jsonl").write_text("{}\n")
    holdout = tmp_path / "holdout.jsonl"
    holdout.write_text("{}\n")
    out = find_named_eval_files({"eval_dataset": str(holdout)}, [str(train)])
    assert out == [str(holdout)]  # explicit list beats auto-detection
    out2 = find_named_eval_files({"eval_dataset": [str(holdout), "missing.jsonl"]},
                                 [str(train)])
    assert out2 == [str(holdout)]  # missing entries dropped


def test_named_eval_never_becomes_a_train_file(tmp_path):
    train = tmp_path / "sft.jsonl"
    (tmp_path / "eval.jsonl").write_text("{}\n")
    # A dataset that (wrongly) lists eval.jsonl as a TRAINING file must not
    # get it back as its own holdout.
    assert find_named_eval_files({}, [str(train),
                                      str(tmp_path / "eval.jsonl")]) == []


def test_named_eval_empty_when_nothing_exists(tmp_path):
    train = tmp_path / "sft.jsonl"
    train.write_text("{}\n")
    assert find_named_eval_files({}, [str(train)]) == []


# ------------------------------------------------- checkpoint resume (#21)

def test_resume_requires_weights_and_state(tmp_path):
    _mk_ckpt(tmp_path, 10, weights=False)          # state only -> skip
    _mk_ckpt(tmp_path, 20, state=False)            # weights only -> skip
    assert find_resumable_checkpoint(tmp_path) is None
    good = _mk_ckpt(tmp_path, 30)
    assert find_resumable_checkpoint(tmp_path) == str(good)


def test_resume_picks_highest_step_not_lexical(tmp_path):
    _mk_ckpt(tmp_path, 9)
    _mk_ckpt(tmp_path, 100)   # lexically "checkpoint-100" < "checkpoint-9"
    _mk_ckpt(tmp_path, 50)
    assert find_resumable_checkpoint(tmp_path) == str(tmp_path / "checkpoint-100")


def test_resume_skips_finished_runs(tmp_path):
    # output_dir is shared across runs of a dataset: a checkpoint whose
    # trainer state shows the run completed must never auto-resume.
    done = _mk_ckpt(tmp_path, 40, global_step=60, max_steps=60)
    assert find_resumable_checkpoint(tmp_path) is None
    interrupted = _mk_ckpt(tmp_path, 20, global_step=20, max_steps=60)
    assert find_resumable_checkpoint(tmp_path) == str(interrupted)
    # Zero-step checkpoint: nothing trained yet -> start clean instead.
    shutil.rmtree(interrupted)
    _mk_ckpt(tmp_path, 5, global_step=0, max_steps=60)
    assert find_resumable_checkpoint(tmp_path) is None


def test_resume_unreadable_state_counts_unfinished(tmp_path):
    ck = tmp_path / "checkpoint-7"
    ck.mkdir()
    (ck / "adapter_model.safetensors").write_text("w")
    (ck / "trainer_state.json").write_text("{trunc")  # crashed mid-save
    assert find_resumable_checkpoint(tmp_path) == str(ck)


def test_resume_ignores_junk_dirs(tmp_path):
    (tmp_path / "checkpoint-abc").mkdir()
    (tmp_path / "not-a-checkpoint").mkdir()
    (tmp_path / "outputs").mkdir()          # no recursion into subdirs
    assert find_resumable_checkpoint(tmp_path) is None
    assert find_resumable_checkpoint(tmp_path / "missing") is None


def test_resolve_resume_knob_semantics(tmp_path):
    ck = _mk_ckpt(tmp_path, 12)
    # auto (the default): discover
    assert resolve_resume_path({}, tmp_path) == str(ck)
    # explicit false: never
    assert resolve_resume_path({"resume": False}, tmp_path) is None
    assert resolve_resume_path({"resume": "off"}, tmp_path) is None
    # explicit valid path: used verbatim
    assert resolve_resume_path({"resume": str(ck)}, tmp_path) == str(ck)
    # explicit bogus path: refuse loudly, never silently restart
    with pytest.raises(ValueError):
        resolve_resume_path({"resume": str(tmp_path / "nope")}, tmp_path)


def test_resolve_resume_auto_with_nothing_to_resume(tmp_path):
    assert resolve_resume_path({}, tmp_path) is None


# ------------------------------------------------------------- cadence plan

def test_cadence_steps_when_run_longer_than_save_interval():
    c = plan_cadence(1000, batch_size=2, grad_accum=4, epochs=3, save_steps=0)
    assert c["save_strategy"] == "steps" and c["save_steps"] == 50
    assert c["eval_strategy"] == "steps" and c["eval_steps"] == 50


def test_cadence_honors_explicit_save_steps():
    c = plan_cadence(1000, save_steps=25)
    assert c["save_steps"] == 25 and c["eval_steps"] == 25


def test_cadence_epoch_when_run_shorter_than_save_interval():
    c = plan_cadence(40, batch_size=2, grad_accum=4, epochs=1, save_steps=50)
    assert c["save_strategy"] == "epoch" and c["eval_strategy"] == "epoch"
    assert "save_steps" not in c


def test_cadence_max_steps_respected():
    c = plan_cadence(10_000, max_steps=10, save_steps=50)
    assert c["save_strategy"] == "epoch"  # tiny run: 10 steps <= 50
    c2 = plan_cadence(10_000, max_steps=200, save_steps=50)
    assert c2["save_strategy"] == "steps"


def test_cadence_empty_dataset_no_crash():
    c = plan_cadence(0)
    assert c["save_strategy"] == "no" and c["eval_strategy"] == "no"


# --------------------------------------------------------- event formatting

def test_progress_event_shape():
    ev = progress_event(12, 120, 1.234)
    assert ev == {"event": "progress", "step": 12, "total": 120, "loss": 1.234}
    # backward-compatible: None fields omitted entirely
    assert progress_event(3) == {"event": "progress", "step": 3}


def test_eval_event_shape():
    assert eval_event(20, 0.5) == {"event": "eval", "step": 20, "eval_loss": 0.5}
    assert eval_event(20, None) == {"event": "eval", "step": 20}


class FakeState:
    def __init__(self, step=10, max_steps=100, log_history=None):
        self.global_step = step
        self.max_steps = max_steps
        self.log_history = log_history or []


class FakeArgs:
    max_steps = 100


def test_protocol_callback_emits_json_lines(tmp_path, capsys):
    log = tmp_path / "job.log"
    cb = make_protocol_callback(str(log))

    cb.on_log(FakeArgs(), FakeState(step=10), logs={"loss": 1.5})
    cb.on_evaluate(FakeArgs(), FakeState(step=20), metrics={"eval_loss": 0.75})

    lines = [json.loads(l) for l in log.read_text().splitlines()]
    prog = lines[0]
    assert prog["event"] == "progress" and prog["step"] == 10
    assert prog["total"] == 100 and prog["loss"] == pytest.approx(1.5)
    ev = lines[1]
    assert ev["event"] == "eval" and ev["step"] == 20
    assert ev["eval_loss"] == pytest.approx(0.75)
    # stdout mirror (the scheduler tails stdout too)
    out = capsys.readouterr().out.splitlines()
    assert json.loads(out[0])["event"] == "progress"
    assert cb.last_eval_loss == pytest.approx(0.75)


def test_protocol_callback_loss_from_log_history_fallback(tmp_path):
    log = tmp_path / "job.log"
    cb = make_protocol_callback(str(log))
    state = FakeState(step=30, log_history=[{"loss": 2.0}, {"loss": 1.75}])
    cb.on_log(FakeArgs(), state, logs={"lr": 1e-4})  # no loss in this log dict
    rec = json.loads(log.read_text().splitlines()[0])
    assert rec["loss"] == pytest.approx(1.75)


# ---------------------------------------------------- registry losses + gate

def test_registry_migration_adds_loss_columns(tmp_path):
    db = tmp_path / "sable.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    # Simulate a pre-#4 models table (no train_loss/eval_loss columns).
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE models ("
        " model_id TEXT PRIMARY KEY, base_model TEXT NOT NULL,"
        " parent_adapter TEXT, dataset_id TEXT, manifest_hash TEXT,"
        " adapter_dir TEXT, gguf_path TEXT, quant TEXT, status TEXT NOT NULL,"
        " created_at TEXT NOT NULL, validation_json TEXT)")
    con.execute(
        "INSERT INTO models (model_id, base_model, status, created_at)"
        " VALUES ('v1', 'B/M', 'promoted', '2026-01-01T00:00:00+00:00')")
    con.commit()
    con.close()

    reg = ModelRegistry(db)
    try:
        # Tolerant read of the legacy row: losses default to None.
        m = reg.get_model("v1")
        assert m is not None and m.train_loss is None and m.eval_loss is None
        reg.set_losses("v1", train_loss=1.25, eval_loss=0.9)
        m = reg.get_model("v1")
        assert m is not None
        assert m.train_loss == pytest.approx(1.25)
        assert m.eval_loss == pytest.approx(0.9)
    finally:
        reg.close()

    # Reopen: values persisted; migration is idempotent.
    reg2 = ModelRegistry(db)
    try:
        m = reg2.get_model("v1")
        assert m is not None
        assert m.eval_loss == pytest.approx(0.9)
    finally:
        reg2.close()


def test_set_losses_only_writes_provided_fields(tmp_path):
    reg = ModelRegistry(tmp_path / "r.db")
    try:
        reg.register_model("B/M")
        reg.set_losses("v1", train_loss=2.0)
        m = reg.get_model("v1")
        assert m is not None
        assert m.train_loss == pytest.approx(2.0) and m.eval_loss is None
        reg.set_losses("v1")  # no-op, must not clear anything
        m_after = reg.get_model("v1")
        assert m_after is not None
        assert m_after.train_loss == pytest.approx(2.0)
    finally:
        reg.close()


def test_parse_worker_losses(tmp_path):
    log = tmp_path / "job.log"
    log.write_text(
        json.dumps({"event": "start"}) + "\n"
        + json.dumps({"event": "progress", "step": 1, "loss": 2.0}) + "\n"
        + json.dumps({"event": "sft_done", "train_loss": 0.42,
                      "eval_loss": 0.5}) + "\n")
    assert parse_worker_losses(str(log)) == {"train_loss": 0.42,
                                             "eval_loss": 0.5}
    # older/stub worker logs without the fields -> both None
    log.write_text(json.dumps({"event": "done"}) + "\n")
    assert parse_worker_losses(str(log)) == {"train_loss": None,
                                             "eval_loss": None}
    assert parse_worker_losses(None) == {"train_loss": None, "eval_loss": None}
    assert parse_worker_losses(str(tmp_path / "missing.log"))["train_loss"] is None


# ---------------------------------------------------------- promotion gate

def test_gate_no_op_without_both_losses():
    for cand, parent in [(None, 1.0), (1.0, None), (None, None), (1.0, 0.0)]:
        res = eval_regression_check(cand, parent, gate="block")
        assert not res["blocked"] and not res["warned"]
        assert res["regression_pct"] is None


def test_gate_warn_default():
    # 2.0 vs parent 1.0 = +100% regression, limit 25% -> warn only by default
    res = eval_regression_check(2.0, 1.0)
    assert res["warned"] and not res["blocked"]
    assert res["regression_pct"] == pytest.approx(100.0)
    assert "eval_loss regressed" in res["message"]


def test_gate_block():
    res = eval_regression_check(1.3, 1.0, gate="block", max_pct=25.0)
    assert res["blocked"] and not res["warned"]
    # improvement / within tolerance never fires
    assert not eval_regression_check(0.9, 1.0, gate="block")["blocked"]
    assert not eval_regression_check(1.2, 1.0, gate="block", max_pct=25.0)["blocked"]


# ------------------------------------------------ pipeline wiring (stubbed)

STUB = r'''
import argparse, json, os, sys
from pathlib import Path
ap = argparse.ArgumentParser()
for f in (("--job", True), ("--kind", True), ("--config", False),
          ("--dataset-dir", False), ("--base-model", False),
          ("--log-path", False), ("--adapter-dir", False),
          ("--gguf-out", False)):
    ap.add_argument(f[0], required=f[1], default=None)
ap.add_argument("--quant", default="q4_k_m")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

def emit(ev, **kw):
    line = json.dumps({"event": ev, "job": a.job, **kw})
    print(line, flush=True)
    if a.log_path:
        Path(a.log_path).parent.mkdir(parents=True, exist_ok=True)
        open(a.log_path, "a").write(line + "\n")

emit("start")
if a.kind in ("sft", "dpo"):
    out = Path(a.config).parent if a.config else Path(".")
    (out / "adapter").mkdir(parents=True, exist_ok=True)
    (out / "adapter" / "adapter_config.json").write_text("{}")
    # backlog #4: terminal event carries the losses
    emit("sft_done", adapter_dir=str(out / "adapter"),
         train_loss=float(os.environ.get("STUB_TRAIN_LOSS", "1.0")),
         eval_loss=float(os.environ.get("STUB_EVAL_LOSS", "-1")) or None)
elif a.kind == "export":
    gg = Path(a.gguf_out)
    gg.parent.mkdir(parents=True, exist_ok=True)
    gg.write_bytes(b"GGUF-stub")
emit("done")
'''


@pytest.fixture()
def env(tmp_path, monkeypatch):
    home = tmp_path / "uf-eval"
    monkeypatch.setenv("SABLE_HOME", str(home))
    stub = tmp_path / "stub.py"
    stub.write_text(STUB)
    yield home, stub


def seed_trace(store, home, i, score=0.9):
    ts = f"2026-09-01T1{i:09d}+00:00"[:25]
    p = home / "buffer" / "2026" / "09" / f"ev-{i}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    msgs = [
        {"role": "user", "content": f"task {i}"},
        {"role": "assistant", "content": f"done task {i} result=ok"},
    ]
    with open(p, "w") as f:
        f.write(json.dumps({"kind": "trace_header"}) + "\n")
        for m in msgs:
            f.write(json.dumps({"kind": "message", "message": m}) + "\n")
        f.write(json.dumps({"kind": "trace_footer", "status": "success"}) + "\n")
    from sable.store.db import TraceRow
    store.insert_trace(TraceRow(
        trace_id=f"ev-{i}", session_id=None, timestamp=ts,
        agent_profile="fallback", model_alias="base-model",
        model_version=None, endpoint_label=None, task_summary="t",
        status="success", outcome_score=score, num_tool_calls=0,
        tokens_in=None, tokens_out=None, duration_s=None, jsonl_path=str(p),
    ))


def _starter(reply="ok-answer"):
    from fake_server import FakeOpenAIServer, make_scripted_handler

    def starter(gguf_path):
        srv = FakeOpenAIServer(handler_cls=make_scripted_handler({"*": reply}))
        srv.start()
        return srv
    return starter


def _run(home, stub, reg, store, monkeypatch, *, eval_loss, gate):
    from sable.curate import CurationConfig
    from sable.training.pipeline import run_iterate
    from sable.training.scheduler import Scheduler, TrainingConfig
    from sable.training.validate import ValidationConfig

    monkeypatch.setenv("STUB_EVAL_LOSS", str(eval_loss))
    sched = Scheduler(registry=reg, cfg=TrainingConfig(base_model="B/M"),
                      worker_command=[sys.executable, str(stub)])
    return run_iterate(
        home=home, registry=reg, scheduler=sched, store=store,
        curation_cfg=CurationConfig(min_score=0.5),
        training_cfg=TrainingConfig(base_model="B/M", eval_gate=gate,
                                    max_eval_regression_pct=25.0),
        validation_cfg=ValidationConfig(
            min_pass_rate=1.0,
            golden_tasks=[{"prompt": "q", "expect_contains": ["ok-answer"]}]),
        name="ev", server_starter=_starter(),
    )


def test_pipeline_stores_losses_and_gate_blocks(tmp_path, env, monkeypatch):
    from sable.store.db import TraceStore

    home, stub = env
    store = TraceStore(home / "buffer" / "sable.db")
    reg = ModelRegistry(home / "buffer" / "sable.db")
    for i in range(2):
        seed_trace(store, home, i)

    r1 = _run(home, stub, reg, store, monkeypatch, eval_loss=1.0, gate="warn")
    assert r1.ok and r1.promoted, r1.to_dict()
    m1 = reg.get_model("v1")
    assert m1 is not None
    assert m1.train_loss == pytest.approx(1.0)
    assert m1.eval_loss == pytest.approx(1.0)

    # second run: parent eval 1.0 -> candidate 2.0 (+100%)
    for i in range(2, 4):
        seed_trace(store, home, i)
    r2 = _run(home, stub, reg, store, monkeypatch, eval_loss=2.0, gate="warn")
    assert r2.ok and r2.promoted  # warn mode: promotion proceeds
    m2 = reg.get_model("v2")
    assert m2 is not None
    assert m2.eval_loss == pytest.approx(2.0)
    assert m2.validation is not None
    assert m2.validation["eval_gate"]["blocked"] is False

    store.close()
    reg.close()


def test_pipeline_gate_block_refuses_promotion(env, monkeypatch):
    from sable.store.db import TraceStore

    home, stub = env
    store = TraceStore(home / "buffer" / "sable.db")
    reg = ModelRegistry(home / "buffer" / "sable.db")
    for i in range(2):
        seed_trace(store, home, i)
    _run(home, stub, reg, store, monkeypatch, eval_loss=1.0, gate="warn")

    for i in range(2, 4):
        seed_trace(store, home, i)
    r2 = _run(home, stub, reg, store, monkeypatch, eval_loss=5.0, gate="block")
    assert not r2.ok and r2.stage == "promote"
    assert "eval gate blocked" in (r2.error or "")
    m2 = reg.get_model("v2")
    assert m2 is not None and m2.status != "promoted"
    active = reg.active_model()
    assert active is not None and active.model_id == "v1"  # parent stays live

    store.close()
    reg.close()


def test_pipeline_gate_skipped_without_eval_data(env, monkeypatch):
    """Stub worker without eval_loss -> gate silently skipped (compat)."""
    from sable.store.db import TraceStore

    home, stub = env
    monkeypatch.setenv("STUB_EVAL_LOSS", "0")  # -> None via `or None`
    store = TraceStore(home / "buffer" / "sable.db")
    reg = ModelRegistry(home / "buffer" / "sable.db")
    for i in range(2):
        seed_trace(store, home, i)
    r = _run(home, stub, reg, store, monkeypatch, eval_loss=0, gate="block")
    assert r.ok and r.promoted
    m = reg.get_model("v1")
    assert m is not None
    assert m.eval_loss is None

    store.close()
    reg.close()

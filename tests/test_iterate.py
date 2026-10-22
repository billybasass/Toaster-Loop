"""Phase 4 e2e — `iterate` loop closure with stubbed worker + fake server.

dataset -> candidate model -> trained adapter (stub) -> GGUF (stub converter)
-> validated (scripted server) -> promoted; second iterate yields v2 with
lineage parent=v1; rollback returns the active pointer to v1. No GPU, no
torch, no llama.cpp: the "worker" is a python stub honoring the worker CLI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fake_server import FakeOpenAIServer, make_scripted_handler  # noqa: E402
from sable.curate import CurationConfig  # noqa: E402
from sable.registry import ModelRegistry  # noqa: E402
from sable.store.db import TraceRow, TraceStore  # noqa: E402
from sable.training.pipeline import run_iterate  # noqa: E402
from sable.training.scheduler import Scheduler, TrainingConfig  # noqa: E402
from sable.training.validate import ValidationConfig  # noqa: E402

# Stub worker: same contract as tests/test_scheduler.py plus export GGUF copy.
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

# fail knobs via env file (subprocess env is inherited; tests write markers)
marker = os.environ.get("STUB_FAIL_FILE")
if marker and Path(marker).is_file():
    emit("error", error="stub failure")
    sys.exit(2)

emit("start")
if a.kind in ("sft", "dpo"):
    out = Path(a.config).parent if a.config else Path(".")
    (out / "adapter").mkdir(parents=True, exist_ok=True)
    (out / "adapter" / "adapter_config.json").write_text(json.dumps(
        {"base_model": a.base_model or "stub", "r": 8}))
elif a.kind == "export":
    gg = Path(a.gguf_out)
    gg.parent.mkdir(parents=True, exist_ok=True)
    # "convert" by copying the fixture GGUF the test points at
    fixture = os.environ.get("STUB_FIXTURE_GGUF")
    if fixture and Path(fixture).is_file():
        gg.write_bytes(Path(fixture).read_bytes())
    else:
        gg.write_bytes(b"GGUF-stub")
emit("done")
'''


@pytest.fixture()
def env(tmp_path, monkeypatch):
    home = tmp_path / "uf-iterate"
    monkeypatch.setenv("SABLE_HOME", str(home))
    stub = tmp_path / "stub.py"
    stub.write_text(STUB)
    fixture_gguf = tmp_path / "tiny.gguf"
    fixture_gguf.write_bytes(b"GGUF" + b"\x00" * 32)
    monkeypatch.setenv("STUB_FIXTURE_GGUF", str(fixture_gguf))
    yield home, stub, fixture_gguf


def seed_trace(store, home, i, messages, score=0.9):
    ts = f"2026-09-01T1{i:09d}+00:00"[:25]
    p = home / "buffer" / "2026" / "09" / f"iter-{i}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        f.write(json.dumps({"kind": "trace_header"}) + "\n")
        for m in messages:
            f.write(json.dumps({"kind": "message", "message": m}) + "\n")
        f.write(json.dumps({"kind": "trace_footer", "status": "success"}) + "\n")
    store.insert_trace(TraceRow(
        trace_id=f"iter-{i}", session_id=None, timestamp=ts,
        agent_profile="fallback", model_alias="base-model",
        model_version=None, endpoint_label=None, task_summary="t",
        status="success", outcome_score=score, num_tool_calls=0,
        tokens_in=None, tokens_out=None, duration_s=None, jsonl_path=str(p),
    ))


def good_trace(n):
    return [
        {"role": "system", "content": "helpful"},
        {"role": "user", "content": f"task number {n}"},
        {"role": "assistant", "content": f"done with task {n} result=ok"},
    ]


GOLDEN = [{"prompt": "anything", "expect_contains": ["ok-answer"]}]


def starter_factory(reply="ok-answer"):
    def starter(gguf_path):
        srv = FakeOpenAIServer(handler_cls=make_scripted_handler(
            {"*": reply}))
        srv.start()
        return srv
    return starter


_starter_replies = starter_factory


def run_loop(home, stub, registry, store, kind="sft", dataset_id=None,
             min_pass=1.0, fail_file=None):
    import os

    if fail_file:
        os.environ["STUB_FAIL_FILE"] = str(fail_file)
    try:
        sched = Scheduler(registry=registry, cfg=TrainingConfig(
            base_model="Base/HF-Model", gguf_quant="q4_k_m"),
            worker_command=[sys.executable, str(stub)])
        return run_iterate(
            home=home, registry=registry, scheduler=sched, store=store,
            curation_cfg=CurationConfig(min_score=0.5),
            training_cfg=TrainingConfig(base_model="Base/HF-Model"),
            validation_cfg=ValidationConfig(min_pass_rate=min_pass,
                                            golden_tasks=GOLDEN),
            name="run", kind=kind, dataset_id=dataset_id,
            server_starter=starter_factory(),
        )
    finally:
        if fail_file:
            os.environ.pop("STUB_FAIL_FILE", None)


def test_iterate_full_loop_v1_then_v2_then_rollback(env):
    home, stub, fixture = env
    store = TraceStore(home / "buffer" / "sable.db")
    reg = ModelRegistry(home / "buffer" / "sable.db")
    for i in range(3):
        seed_trace(store, home, i, good_trace(i))

    r1 = run_loop(home, stub, reg, store)
    assert r1.ok and r1.promoted, r1.to_dict()
    assert r1.model_id == "v1"
    assert r1.stages == ["curate", "register", "train", "export",
                         "validate", "promote"]
    # artifacts exist
    assert r1.gguf_path is not None
    gguf = Path(r1.gguf_path)
    assert gguf.is_file() and gguf.read_bytes().startswith(b"GGUF")
    assert r1.adapter_dir is not None
    assert Path(r1.adapter_dir, "adapter_config.json").is_file()
    active = reg.active_model()
    assert active is not None and active.model_id == "v1"
    assert r1.validation is not None
    assert r1.validation["pass_rate"] == 1.0

    # second iterate -> v2 with lineage to v1 (needs fresh traces: claims are
    # consumed; seed two more)
    for i in range(3, 5):
        seed_trace(store, home, i, good_trace(i))
    r2 = run_loop(home, stub, reg, store)
    assert r2.ok and r2.promoted
    assert r2.model_id == "v2"
    m2 = reg.get_model("v2")
    assert m2 is not None and m2.parent_adapter == "v1"
    active = reg.active_model()
    assert active is not None and active.model_id == "v2"
    m1 = reg.get_model("v1")
    assert m1 is not None and m1.status == "superseded"

    # rollback -> active pointer back to v1
    rb = reg.rollback()
    assert rb is not None and rb.model_id == "v1"
    active = reg.active_model()
    assert active is not None and active.model_id == "v1"
    store.close()
    reg.close()


def test_iterate_validation_failure_blocks_promotion(env, tmp_path):
    home, stub, fixture = env
    store = TraceStore(home / "buffer" / "sable.db")
    reg = ModelRegistry(home / "buffer" / "sable.db")
    for i in range(2):
        seed_trace(store, home, i, good_trace(i))

    # golden task that the scripted server can never satisfy
    sched = Scheduler(registry=reg, cfg=TrainingConfig(base_model="B/M"),
                      worker_command=[sys.executable, str(stub)])
    res = run_iterate(
        home=home, registry=reg, scheduler=sched, store=store,
        curation_cfg=CurationConfig(min_score=0.5),
        training_cfg=TrainingConfig(base_model="B/M"),
        validation_cfg=ValidationConfig(
            min_pass_rate=1.0,
            golden_tasks=[{"prompt": "q", "expect_contains": ["never-served"]}],
        ),
        name="bad", server_starter=_starter_replies("ok"),
    )
    assert not res.ok and res.stage == "validate"
    assert res.model_id is not None
    failed = reg.get_model(res.model_id)
    assert failed is not None and failed.status == "failed"
    assert reg.active_model() is None  # nothing promoted
    store.close()
    reg.close()


def test_iterate_train_failure_keeps_candidate_unpromoted(env):
    home, stub, fixture = env
    store = TraceStore(home / "buffer" / "sable.db")
    reg = ModelRegistry(home / "buffer" / "sable.db")
    for i in range(2):
        seed_trace(store, home, i, good_trace(i))
    fail = tmp_marker = home / "FAIL"
    fail.write_text("")
    r = run_loop(home, stub, reg, store, fail_file=fail)
    assert not r.ok and r.stage == "train"
    assert r.model_id is not None
    m = reg.get_model(r.model_id)
    assert m is not None and m.status == "failed"  # validation error recorded by pipeline
    assert reg.active_model() is None
    store.close()
    reg.close()


def test_iterate_export_failure_keeps_adapter(env, monkeypatch):
    home, stub, fixture = env
    store = TraceStore(home / "buffer" / "sable.db")
    reg = ModelRegistry(home / "buffer" / "sable.db")
    for i in range(2):
        seed_trace(store, home, i, good_trace(i))

    # worker that trains fine but fails export
    bad_stub = home.parent / "bad_export.py"
    bad_stub.write_text(STUB.replace('elif a.kind == "export":',
                                     'elif a.kind == "export":\n    sys.exit(7)'))
    sched = Scheduler(registry=reg, cfg=TrainingConfig(base_model="B/M"),
                      worker_command=[sys.executable, str(bad_stub)])
    res = run_iterate(
        home=home, registry=reg, scheduler=sched, store=store,
        curation_cfg=CurationConfig(min_score=0.5),
        training_cfg=TrainingConfig(base_model="B/M"),
        validation_cfg=ValidationConfig(min_pass_rate=1.0,
                                        golden_tasks=GOLDEN),
        name="x", server_starter=starter_factory(),
    )
    assert not res.ok and res.stage == "export"
    # adapter artifact kept; model still candidate (data fine, GGUF missing)
    assert res.adapter_dir is not None
    assert Path(res.adapter_dir).is_dir()
    assert res.model_id is not None
    cand = reg.get_model(res.model_id)
    assert cand is not None and cand.status == "candidate"
    store.close()
    reg.close()


def test_iterate_dry_run_stops_before_training_artifacts(env):
    home, stub, fixture = env
    store = TraceStore(home / "buffer" / "sable.db")
    reg = ModelRegistry(home / "buffer" / "sable.db")
    for i in range(2):
        seed_trace(store, home, i, good_trace(i))
    sched = Scheduler(registry=reg, cfg=TrainingConfig(base_model="B/M"),
                      worker_command=[sys.executable, str(stub)])
    res = run_iterate(
        home=home, registry=reg, scheduler=sched, store=store,
        curation_cfg=CurationConfig(min_score=0.5),
        training_cfg=TrainingConfig(base_model="B/M"),
        validation_cfg=ValidationConfig(min_pass_rate=1.0,
                                        golden_tasks=GOLDEN),
        name="dry", dry_run=True, server_starter=starter_factory(),
    )
    assert res.ok and not res.promoted
    assert res.stages == ["curate", "register", "train"]
    assert res.gguf_path is None
    store.close()
    reg.close()

"""Production-readiness deep-dive tests (continual training + hardening).

Covers the fixes for the training-subsystem deep-dive findings:

  * run fingerprinting + fingerprint-scoped auto-resume (stale-checkpoint
    poisoning),
  * base_adapter resolution (continual training entry points),
  * worker CLI contract (--config absent => honest JSON error event),
  * scheduler --base-adapter argv passthrough,
  * finalize_standalone_train lineage (parent_adapter + base inheritance),
  * export-method attestation (parse_worker_export_method),
  * deploy gate for gguf-lora artifacts,
  * doctor check_model_artifacts.

All torch-free: GPU paths are guarded by source-level assertions where a
real model would be needed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from toaster_loop.registry import ModelRegistry
from toaster_loop.training.pipeline import (
    parse_worker_export_method,
)
from toaster_loop.training.worker import (
    RUN_MARKER_FILENAME,
    find_resumable_checkpoint,
    resolve_base_adapter,
    resolve_resume_path,
    run_fingerprint,
    run_marker_matches,
    write_run_marker,
)


# ------------------------------------------------------------- fingerprint

def _cfg(**over):
    cfg = {
        "model": "base/model",
        "data": {"local_dataset": ["/d/sft.jsonl"]},
        "training": {"learning_rate": 2e-4, "num_epochs": 3},
        "lora": {"lora_r": 64},
    }
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k] = {**cfg[k], **v}
        else:
            cfg[k] = v
    return cfg


def test_run_fingerprint_stable_and_sensitive():
    assert run_fingerprint(_cfg()) == run_fingerprint(_cfg())
    # Identity-defining changes flip the fingerprint...
    assert run_fingerprint(_cfg()) != run_fingerprint(_cfg(model="other/model"))
    assert run_fingerprint(
        _cfg()) != run_fingerprint(_cfg(training={"base_adapter": "/x/a"}))
    assert run_fingerprint(
        _cfg()) != run_fingerprint(_cfg(lora={"lora_r": 32}))
    # ...but cadence knobs (deliberately excluded) do not.
    assert run_fingerprint(
        _cfg()) == run_fingerprint(_cfg(training={"save_steps": 10}))


def test_run_marker_missing_matches_corrupt_does_not(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    # No marker (legacy run) => match, keep old resume behavior.
    assert run_marker_matches(out, "deadbeef") is True
    write_run_marker(out, "deadbeef")
    assert run_marker_matches(out, "deadbeef") is True
    assert run_marker_matches(out, "cafe0000") is False
    (out / RUN_MARKER_FILENAME).write_text("not json{{")
    assert run_marker_matches(out, "deadbeef") is False


def _fake_checkpoint(root: Path, step: int = 5) -> Path:
    ck = root / f"checkpoint-{step}"
    ck.mkdir(parents=True)
    (ck / "adapter_model.safetensors").write_bytes(b"x")
    # Unfinished: global_step < max_steps => resumable.
    (ck / "trainer_state.json").write_text(
        json.dumps({"global_step": step, "max_steps": 100}))
    return ck


def test_auto_resume_scoped_by_fingerprint(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    _fake_checkpoint(out)
    cfg = _cfg()
    fp = run_fingerprint(cfg)
    # Matching marker (or none) => auto-discovery finds the checkpoint.
    assert resolve_resume_path({"resume": "auto"}, out, None,
                               fingerprint=fp) is not None
    write_run_marker(out, fp)
    assert resolve_resume_path({"resume": "auto"}, out, None,
                               fingerprint=fp) is not None
    # Foreign marker => refuse auto-resume (the poisoning fix).
    write_run_marker(out, "ffffffff")
    assert resolve_resume_path({"resume": "auto"}, out, None,
                               fingerprint=fp) is None
    # Explicit paths still bypass the scoping (train resume stays possible).
    ck = find_resumable_checkpoint(out)
    assert ck is not None
    assert resolve_resume_path({"resume": ck}, out, None,
                               fingerprint=fp) == ck


# --------------------------------------------------------- base adapter

def test_resolve_base_adapter_validation(tmp_path):
    assert resolve_base_adapter({}) is None
    bad = tmp_path / "not-an-adapter"
    bad.mkdir()
    with pytest.raises(ValueError, match="not a training artifact"):
        resolve_base_adapter({"training": {"base_adapter": str(bad)}})
    with pytest.raises(ValueError, match="not a directory"):
        resolve_base_adapter({"training": {"base_adapter": str(tmp_path / "x")}})
    good = tmp_path / "adapter"
    good.mkdir()
    (good / "adapter_config.json").write_text("{}")
    assert resolve_base_adapter(
        {"training": {"base_adapter": str(good)}}) == str(good)
    # CLI override wins over the yaml knob.
    other = tmp_path / "other"
    other.mkdir()
    (other / "training_type.json").write_text('{"training_type": "full"}')
    assert resolve_base_adapter(
        {"training": {"base_adapter": str(good)}},
        cli_override=str(other)) == str(other)


def test_worker_cli_missing_config_emits_error_event(tmp_path):
    """Finding #7: --config absent must produce a JSON error event + rc 1,
    not an argparse exit-2 with an empty log."""
    from toaster_loop.training.worker import main as worker_main

    log = tmp_path / "job.log"
    rc = worker_main(["--job", "job-x", "--kind", "sft",
                      "--log-path", str(log)])
    assert rc == 1
    events = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
    kinds = [e.get("event") for e in events]
    assert "start" in kinds and "error" in kinds
    err = [e for e in events if e.get("event") == "error"][0]
    assert "train.yaml" in err["error"]


def test_scheduler_argv_passes_base_adapter(tmp_path):
    from toaster_loop.training.scheduler import Scheduler, TrainingConfig

    reg = ModelRegistry(tmp_path / "r.db")
    try:
        sched = Scheduler(registry=reg, cfg=TrainingConfig())
        job = reg.enqueue_job(
            "sft", params={"train_yaml": str(tmp_path / "t.yaml"),
                           "base_adapter": str(tmp_path / "adapter")})
        argv = sched.worker_argv(job)
        i = argv.index("--base-adapter")
        assert argv[i + 1] == str(tmp_path / "adapter")
        # Absent param => flag omitted entirely (never a dangling option).
        job2 = reg.enqueue_job("sft", params={"train_yaml": "t.yaml"})
        assert "--base-adapter" not in sched.worker_argv(job2)
    finally:
        reg.close()


# ------------------------------------------- export method + finalize

def test_parse_worker_export_method(tmp_path):
    log = tmp_path / "e.log"
    log.write_text(
        json.dumps({"event": "start"}) + "\n"
        + json.dumps({"event": "export_done", "gguf_path": "/a/x.gguf",
                      "method": "llama.cpp"}) + "\n")
    assert parse_worker_export_method(str(log)) == "llama.cpp"
    log.write_text(json.dumps({"event": "export_done",
                               "method": "gguf-lora"}) + "\n")
    assert parse_worker_export_method(str(log)) == "gguf-lora"
    assert parse_worker_export_method(str(tmp_path / "missing.log")) is None


def test_finalize_records_lineage_and_inherits_base(tmp_path):
    """Continual training bookkeeping: parent_adapter edge + the child row
    inherits the parent's resolved base instead of unspecified-base."""
    from toaster_loop.training.pipeline import finalize_standalone_train
    from toaster_loop.training.scheduler import Scheduler, TrainingConfig

    home = tmp_path / "home"
    (home / "datasets" / "ds2").mkdir(parents=True)
    reg = ModelRegistry(home / "buffer" / "toaster_loop.db")
    try:
        parent = reg.register_model(base_model="good/base", actor="test")
        adapter = tmp_path / "adapter-v2"
        adapter.mkdir()
        log = tmp_path / "j.log"
        log.write_text(json.dumps({"event": "sft_done",
                                   "adapter_dir": str(adapter),
                                   "train_loss": 0.5}) + "\n")
        sched = Scheduler(registry=reg, cfg=TrainingConfig())
        res = finalize_standalone_train(
            home=home, registry=reg, scheduler=sched, dataset_id="ds2",
            log_path=str(log), base_model="", actor="test",
            submit_export=False, parent_adapter=parent.model_id)
        assert res["ok"] and res["model_id"]
        row = reg.get_model(res["model_id"])
        assert row is not None
        # Base inherited from the parent row (no more unspecified-base).
        assert row.base_model == "good/base"
        # Real training edge vN -> vN+1.
        assert row.parent_adapter == parent.model_id
    finally:
        reg.close()


def test_finalize_attests_needs_base_resolution(tmp_path):
    from toaster_loop.training.pipeline import finalize_standalone_train
    from toaster_loop.training.scheduler import Scheduler, TrainingConfig

    home = tmp_path / "home"
    (home / "datasets" / "ds3").mkdir(parents=True)
    reg = ModelRegistry(home / "buffer" / "toaster_loop.db")
    try:
        adapter = tmp_path / "adapter-v1"
        adapter.mkdir()
        log = tmp_path / "j.log"
        log.write_text(json.dumps({"event": "sft_done",
                                   "adapter_dir": str(adapter)}) + "\n")
        sched = Scheduler(registry=reg, cfg=TrainingConfig())
        res = finalize_standalone_train(
            home=home, registry=reg, scheduler=sched, dataset_id="ds3",
            log_path=str(log), base_model="", actor="test",
            submit_export=False)
        row = reg.get_model(res["model_id"])
        assert row is not None
        assert row.base_model == "unspecified-base"
        prov = row.provenance or {}
        assert prov.get("needs_base_resolution") is True
    finally:
        reg.close()


# ------------------------------------------------------------- deploy gate

def test_deploy_blocks_gguf_lora_artifact(tmp_path):
    from toaster_loop.core.config import ToasterLoopConfig
    from toaster_loop.serving.deploy import deploy_model
    from toaster_loop.serving.supervisor import ServeError

    home = tmp_path / "home"
    reg = ModelRegistry(home / "buffer" / "toaster_loop.db")
    try:
        m = reg.register_model(base_model="b", actor="test")
        gguf = tmp_path / "m.gguf"
        gguf.write_bytes(b"GGUF")
        reg.set_artifacts(m.model_id, gguf_path=str(gguf))
        reg.set_provenance(m.model_id, {"version": 1,
                                        "export_method": "gguf-lora"})
        with pytest.raises(ServeError, match="gguf-lora"):
            deploy_model(home, reg, m.model_id, ToasterLoopConfig())
    finally:
        reg.close()


# ---------------------------------------------------------------- doctor

def test_doctor_model_artifacts(tmp_path):
    from toaster_loop.doctor import check_model_artifacts

    hub = str(Path.home() / ".cache" / "huggingface" / "hub")

    def row(mid, adapter="", gguf="", base="b", prov=None):
        return SimpleNamespace(model_id=mid, status="candidate",
                               adapter_dir=adapter or None,
                               gguf_path=gguf or None, base_model=base,
                               provenance=prov)

    good_adapter = tmp_path / "a"
    good_adapter.mkdir()
    good_gguf = tmp_path / "m.gguf"
    good_gguf.write_bytes(b"GGUF")

    ok = check_model_artifacts(tmp_path, list_models_fn=lambda: [
        row("v1", adapter=str(good_adapter), gguf=str(good_gguf))])
    assert ok.status == "ok"

    stranded = check_model_artifacts(tmp_path, list_models_fn=lambda: [
        row("v2", adapter=hub + "/models--toaster-loop--ds/snapshots/x")])
    assert stranded.status == "problem"
    assert any("hub cache" in l for l in stranded.lines)

    missing = check_model_artifacts(tmp_path, list_models_fn=lambda: [
        row("v3", adapter=str(tmp_path / "gone"))])
    assert missing.status == "problem"
    assert any("missing on disk" in l for l in missing.lines)

    warn = check_model_artifacts(tmp_path, list_models_fn=lambda: [
        row("v4", base="unspecified-base"),
        row("v5", prov={"export_method": "gguf-lora"})])
    assert warn.status == "warn"
    assert any("base_model unresolved" in l for l in warn.lines)


# ------------------------------------------------- source-level hardening

def test_dpo_grpo_use_hardened_load_and_seeds():
    """Findings #4: DPO/GRPO must load via load_model_hardened, set the seed,
    honor resume, and support base_adapter continuation (no bare torch_dtype
    =auto loads remain in those functions)."""
    from toaster_loop.training import worker as wmod

    src = Path(wmod.__file__).read_text()
    dpo = src[src.index("def run_dpo"):src.index("def reward_format")]
    grpo = src[src.index("def run_grpo"):src.index("def find_llama_cpp_root")]
    for fn_src, name in ((dpo, "run_dpo"), (grpo, "run_grpo")):
        assert "load_model_hardened(" in fn_src, name
        assert "_tf.set_seed(seed)" in fn_src, name
        assert "resolve_resume_path(" in fn_src, name
        assert "base_adapter" in fn_src, name
        assert "torch_dtype=\"auto\"" not in fn_src, (
            f"{name} still has a bare auto-dtype load")
        assert "nan_abort_step" in fn_src, name


def test_sft_continual_training_wiring_present():
    from toaster_loop.training import worker as wmod

    src = Path(wmod.__file__).read_text()
    sft = src[src.index("def run_sft"):src.index("def run_dpo")]
    assert "resolve_base_adapter(" in sft
    assert "PeftModel.from_pretrained(\n                    model, base_adapter," \
        " is_trainable=True)" in sft or (
        "PeftModel.from_pretrained(model, base_adapter, is_trainable=True)"
        in " ".join(sft.split()))
    assert "save_total_limit" in sft  # checkpoint rotation
    assert "write_run_marker(out_dir, run_fingerprint(cfg))" in sft


# ================================================================ pass 3: ML-stack window

def test_mlstack_parse_version():
    from toaster_loop.core.mlstack import parse_version

    assert parse_version("2.13.0+rocm7.2") == (2, 13, 0)
    assert parse_version("5.0rc1") == (5, 0, 0)
    assert parse_version("3") == (3, 0, 0)
    assert parse_version("") is None
    assert parse_version("nightly-dev") is None


def test_mlstack_window_problems():
    from toaster_loop.core.mlstack import window_problems

    # All inside the declared window -> no problems.
    good = {"torch": "2.6.0+cpu", "transformers": "4.51.3", "peft": "0.15.2",
            "trl": "0.19.1", "datasets": "3.6.0", "accelerate": "1.7.0"}
    assert window_problems(good) == []
    # Absent packages are not problems (presence is a different check).
    assert window_problems({"torch": "2.6.0"}) == []
    # Out-of-window trl (the DPO/GRPO API-drift package) -> actionable line.
    probs = window_problems({**good, "trl": "2.1.0"})
    assert len(probs) == 1 and "trl 2.1.0" in probs[0] and ">=0.12,<2" in probs[0]
    # Below the floor too.
    assert any("peft" in p for p in window_problems({**good, "peft": "0.9.0"}))
    # Unparseable versions are reported, not guessed.
    assert any("unparseable" in p for p in window_problems({"torch": "?"}))


def test_mlstack_window_matches_pyproject():
    """The SUPPORTED table and pyproject [training] bounds must agree — a
    bump of one without the other is exactly the drift this guards."""
    import re

    from toaster_loop.core.mlstack import SUPPORTED

    src = Path(__file__).resolve().parents[1].joinpath("pyproject.toml").read_text()
    block = src[src.index("[project.optional-dependencies]"):src.index("[project.scripts]")]
    for pkg, (lo, hi) in SUPPORTED.items():
        m = re.search(rf'"{pkg}>=(\d+(?:\.\d+)?),<(\d+)"', block)
        assert m, f"{pkg} bound missing from pyproject [training]"
        assert (m.group(1), m.group(2)) == (lo, hi), (
            f"{pkg}: pyproject says >={m.group(1)},<{m.group(2)} but "
            f"mlstack.SUPPORTED says >={lo},<{hi}")


def test_mlstack_worker_gate_and_override(monkeypatch, tmp_path):
    from toaster_loop.core import mlstack

    bad = {"trl": "2.5.0"}
    monkeypatch.setattr(mlstack, "installed_versions", lambda **k: dict(bad))
    monkeypatch.delenv(mlstack.ALLOW_OVERRIDE_ENV, raising=False)
    msg = mlstack.worker_gate()
    assert msg and "trl 2.5.0" in msg and mlstack.ALLOW_OVERRIDE_ENV in msg
    # Escape hatch downgrades the stop to None (worker logs a warning instead).
    monkeypatch.setenv(mlstack.ALLOW_OVERRIDE_ENV, "1")
    assert mlstack.worker_gate() is None


def test_worker_main_refuses_out_of_window_stack(monkeypatch, tmp_path):
    """The gate runs inside worker.main BEFORE any training kind: the job log
    carries an honest JSON error event (same contract as every failure)."""
    from toaster_loop.core import mlstack
    from toaster_loop.training.worker import main as worker_main

    monkeypatch.setattr(mlstack, "worker_gate", lambda: "ML-stack window violated — test")
    log = tmp_path / "job.log"
    rc = worker_main(["--job", "j", "--kind", "sft", "--config", "x.yaml",
                      "--log-path", str(log)])
    assert rc == 1
    events = [json.loads(l) for l in log.read_text().splitlines() if l.startswith("{")]
    errs = [e for e in events if e.get("event") == "error"]
    assert errs and "window violated" in errs[0]["error"]


# ================================================================ pass 3: DPO schema normalization

def test_dpo_row_for_trl_normalizes_mixed_schema():
    """The curater emits role/content prompt + STRING completions; TRL's
    is_conversational inspects one arbitrary key then concatenates
    prompt+completion -> TypeError mid-tokenization on trl>=0.19 (caught by
    the real-torch CI smoke). Conversational rows get wrapped completions."""
    from toaster_loop.training.worker import dpo_row_for_trl

    mixed = {"prompt": [{"role": "user", "content": "hi"}],
             "chosen": "yo", "rejected": "no"}
    out = dpo_row_for_trl(mixed)
    assert out["prompt"] == mixed["prompt"]
    assert out["chosen"] == [{"role": "assistant", "content": "yo"}]
    assert out["rejected"] == [{"role": "assistant", "content": "no"}]
    # Fully conversational rows pass through untouched.
    conv = {"prompt": [{"role": "user", "content": "hi"}],
            "chosen": [{"role": "assistant", "content": "yo"}],
            "rejected": [{"role": "assistant", "content": "no"}]}
    assert dpo_row_for_trl(conv) == conv
    # Fully plain rows pass through untouched.
    plain = {"prompt": "the sky is", "chosen": " blue", "rejected": " green"}
    assert dpo_row_for_trl(plain) == plain


def test_dpo_grpo_configs_pin_eval_strategy_off():
    """transformers>=5 defaults eval_strategy=epoch and hard-fails a Trainer
    without eval_dataset; the DPO/GRPO paths run no eval loop, so both must
    pin eval_strategy="no" at construction (real-torch smoke regression)."""
    from toaster_loop.training import worker as wmod

    src = Path(wmod.__file__).read_text()
    dpo = src[src.index("def run_dpo"):src.index("def reward_format")]
    grpo = src[src.index("def run_grpo"):src.index("def find_llama_cpp_root")]
    assert 'eval_strategy="no"' in dpo
    assert 'eval_strategy="no"' in grpo


# ================================================================ pass 3: cross-process locks

def _live_foreign_pid() -> int:
    """A pid that is definitely alive and not us (the test runner's parent)."""
    import os

    return os.getppid()


def test_proclock_flock_backend_exclusion(tmp_path):
    from toaster_loop.core import proclock

    assert proclock.locking_mode() == proclock.MODE_FLOCK  # POSIX test host
    p = tmp_path / "x.lock"
    h1 = proclock.acquire(p)
    assert h1 is not None and h1.mode == proclock.MODE_FLOCK
    assert proclock.acquire(p) is None  # second holder turned away
    proclock.release(h1)
    h2 = proclock.acquire(p)
    assert h2 is not None
    proclock.release(h2)
    proclock.release(None)  # idempotent


def test_proclock_pidfile_backend_on_linux(tmp_path, monkeypatch):
    """NT-style fallback exercised ON LINUX: force the pidfile backend by
    hiding flock/msvcrt (the platform flag the task asks us to fake)."""
    from toaster_loop.core import proclock

    monkeypatch.setattr(proclock, "_HAVE_FCNTL", False)
    monkeypatch.setattr(proclock, "_msvcrt", None)
    assert proclock.locking_mode() == proclock.MODE_PIDFILE

    p = tmp_path / "loop.lock"
    h = proclock.acquire(p)
    assert h is not None and h.mode == proclock.MODE_PIDFILE
    assert int(p.read_text().strip()) == os.getpid()
    # Contention: a live foreign holder keeps us out.
    p.unlink()
    p.write_text(str(_live_foreign_pid()))
    assert proclock.acquire(p) is None
    # Stale lock (dead pid) is reclaimed, and the new holder's pid wins.
    p.unlink()
    p.write_text("999999")  # almost-certainly-dead pid
    h2 = proclock.acquire(p)
    assert h2 is not None
    assert int(p.read_text().strip()) == os.getpid()
    proclock.release(h2)
    assert not p.exists()  # release removes OUR lockfile


def test_proclock_msvcrt_backend_on_linux(tmp_path, monkeypatch):
    """msvcrt byte-range locking exercised with the primitive injected: a fake
    msvcrt whose LK_NBLCK raises OSError when the byte is already locked."""
    from toaster_loop.core import proclock

    # Real msvcrt locks the FILE byte (not the fd), so model one lock slot:
    # held -> any further LK_NBLCK raises; only LK_UNLCK frees it.
    state = {"held": False}

    class FakeMsvcrt:
        LK_NBLCK, LK_UNLCK = 1, 2

        @staticmethod
        def locking(fd, mode, nbytes):
            if mode == FakeMsvcrt.LK_NBLCK:
                if state["held"]:
                    raise OSError("byte range already locked")
                state["held"] = True
            else:
                state["held"] = False

    monkeypatch.setattr(proclock, "_HAVE_FCNTL", False)
    monkeypatch.setattr(proclock, "_msvcrt", FakeMsvcrt())
    assert proclock.locking_mode() == proclock.MODE_MSVCRT

    p = tmp_path / "slot.lock"
    h1 = proclock.acquire(p)
    assert h1 is not None and h1.mode == proclock.MODE_MSVCRT
    assert proclock.acquire(p) is None  # contention via the injected primitive
    proclock.release(h1)
    h2 = proclock.acquire(p)
    assert h2 is not None
    proclock.release(h2)


def test_scheduler_slot_lock_uses_proclock(tmp_path, monkeypatch):
    """The device-slot lock now routes through core.proclock: on the NT-style
    pidfile fallback (faked here), a live foreign holder blocks dispatch."""
    from toaster_loop.core import proclock
    from toaster_loop.training import scheduler as sched

    monkeypatch.setattr(proclock, "_HAVE_FCNTL", False)
    monkeypatch.setattr(proclock, "_msvcrt", None)
    db = tmp_path / "buffer" / "toaster_loop.db"
    lock = sched._slot_lock_path(db, 0)
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(str(_live_foreign_pid()))
    assert sched._try_slot_lock(db, 0) is None  # was a silent no-op sentinel!
    lock.write_text("999999")
    h = sched._try_slot_lock(db, 0)
    assert h is not None
    sched._release_slot_lock(h)


def test_doctor_proc_locks(tmp_path, monkeypatch):
    from toaster_loop.core import proclock
    from toaster_loop.doctor import check_proc_locks

    # POSIX host: flock available -> ok.
    res = check_proc_locks()
    assert res.status == "ok" and "flock" in res.lines[0]
    # Pidfile-only on a POSIX box: honest warn, not a failure.
    monkeypatch.setattr(proclock, "_HAVE_FCNTL", False)
    monkeypatch.setattr(proclock, "_msvcrt", None)
    assert check_proc_locks().status == "warn"
    # NT without msvcrt (the old silent no-op): problem — must be loud.
    bad = check_proc_locks(system="win32")
    assert bad.status == "problem"
    assert any("advisory" in l for l in bad.lines)


# ================================================================ pass 3: gguf-lora serving

def _write_kv_gguf(path, *, arch="llama", gtype=None):
    """Minimal GGUF v3 KV-only file (reader stops after KV; test_gguf_base
    pattern)."""
    import struct as _struct

    def b(s):
        e = s.encode()
        return _struct.pack("<Q", len(e)) + e

    kvs = b("general.architecture") + _struct.pack("<I", 8) + b(arch)
    n = 1
    if gtype is not None:
        kvs += b("general.type") + _struct.pack("<I", 8) + b(gtype)
        n += 1
    path.write_bytes(b"GGUF"[:0] + b"GGUF" + _struct.pack("<I", 3)
                     + _struct.pack("<Q", 0) + _struct.pack("<Q", n) + kvs)
    return path


def test_lora_args_helper():
    from toaster_loop.serving.run_args import lora_args

    assert lora_args(None) == []
    assert lora_args("") == []
    assert lora_args("/x/lora.gguf") == ["--lora", "/x/lora.gguf"]


def test_deploy_blocks_gguf_lora_without_base(tmp_path):
    """An unpaired gguf-lora row still raises — but now the message names the
    pairing knobs instead of telling the user it's impossible."""
    from toaster_loop.core.config import ToasterLoopConfig
    from toaster_loop.serving.deploy import deploy_model
    from toaster_loop.serving.supervisor import ServeError

    home = tmp_path / "home"
    reg = ModelRegistry(home / "buffer" / "toaster_loop.db")
    try:
        m = reg.register_model(base_model="b", actor="test")
        adapter = _write_kv_gguf(tmp_path / "m.gguf", gtype="adapter")
        reg.set_artifacts(m.model_id, gguf_path=str(adapter))
        reg.set_provenance(m.model_id, {"version": 1,
                                        "export_method": "gguf-lora"})
        with pytest.raises(ServeError, match="base"):
            deploy_model(home, reg, m.model_id, ToasterLoopConfig())
    finally:
        reg.close()


def test_deploy_blocks_arch_mismatched_pair(tmp_path, monkeypatch):
    from toaster_loop.core.config import ToasterLoopConfig
    from toaster_loop.serving.deploy import deploy_model
    from toaster_loop.serving.supervisor import ServeError

    monkeypatch.setenv("TOASTER_LOOP_LORA_BASE_GGUF", "")
    home = tmp_path / "home"
    reg = ModelRegistry(home / "buffer" / "toaster_loop.db")
    try:
        m = reg.register_model(base_model="b", actor="test")
        adapter = _write_kv_gguf(tmp_path / "m.gguf", arch="qwen2",
                                 gtype="adapter")
        base = _write_kv_gguf(tmp_path / "base.gguf", arch="llama")
        reg.set_artifacts(m.model_id, gguf_path=str(adapter))
        reg.set_provenance(m.model_id, {"version": 1,
                                        "export_method": "gguf-lora",
                                        "base_gguf": str(base)})
        with pytest.raises(ServeError, match="architecture"):
            deploy_model(home, reg, m.model_id, ToasterLoopConfig())
    finally:
        reg.close()


def test_deploy_serves_paired_gguf_lora(tmp_path, monkeypatch):
    """Properly paired row (adapter parses + provenance base resolves + arch
    match): deploy spawns llama-server with -m BASE --lora ADAPTER (base via
    run_args.lora_args), and the result records the pairing honestly."""
    from toaster_loop.core.config import ToasterLoopConfig
    from toaster_loop.serving import deploy as dep

    home = tmp_path / "home"
    reg = ModelRegistry(home / "buffer" / "toaster_loop.db")
    try:
        m = reg.register_model(base_model="b", actor="test")
        adapter = _write_kv_gguf(tmp_path / "m.gguf", gtype="adapter")
        base = _write_kv_gguf(tmp_path / "base.gguf")
        reg.set_artifacts(m.model_id, gguf_path=str(adapter))
        reg.set_provenance(m.model_id, {"version": 1,
                                        "export_method": "gguf-lora",
                                        "base_gguf": str(base)})

        seen = {}

        def fake_start(cfg, gguf_path, model_name=None, model_id=None,
                       home=None, lora_gguf=None):
            seen["gguf"] = gguf_path
            seen["lora"] = lora_gguf
            return {"base_url": "http://127.0.0.1:9/v1", "model_name": model_name,
                    "gguf_path": gguf_path}

        monkeypatch.setattr(dep, "start_server_detached", fake_start)
        res = dep.deploy_model(home, reg, m.model_id, ToasterLoopConfig())
        assert seen["gguf"] == str(base)          # -m is the BASE model
        assert seen["lora"] == str(adapter)       # --lora carries the adapter
        assert res["lora_adapter"] == str(adapter)
    finally:
        reg.close()


def test_resolve_lora_pair_passthrough_for_normal_rows(tmp_path):
    """Rows that are not gguf-lora exports resolve to None (zero behavior
    change for every existing deploy path)."""
    from toaster_loop.core.config import ToasterLoopConfig
    from toaster_loop.serving.deploy import resolve_lora_pair

    home = tmp_path / "home"
    reg = ModelRegistry(home / "buffer" / "toaster_loop.db")
    try:
        m = reg.register_model(base_model="b", actor="test")
        g = tmp_path / "plain.gguf"
        g.write_bytes(b"GGUF")
        reg.set_artifacts(m.model_id, gguf_path=str(g))
        assert resolve_lora_pair(home, reg, m.model_id,
                                 ToasterLoopConfig()) is None
    finally:
        reg.close()


def test_spawn_cmd_includes_lora_before_extra_args():
    """Source-level guard (spawn path needs a real binary otherwise): the
    --lora pair rides between run_settings and extra_llama_args so explicit
    user extras keep last-wins precedence."""
    from toaster_loop.serving import deploy as dep_mod

    src = Path(dep_mod.__file__).read_text()
    seg = src[src.index("def _start_server_detached_once"):
              src.index("def resolve_lora_pair") if "def resolve_lora_pair" in src[src.index("def _start_server_detached_once"):] else src.index("def deploy_model")]
    assert "+ lora_args(lora_gguf) \\\\" in seg.replace("\r\n", "\n") or "lora_args(lora_gguf)" in seg
    i_lora = seg.index("lora_args(lora_gguf)")
    i_extra = seg.index("list(cfg.extra_llama_args)")
    assert i_lora < i_extra

"""Phase 4 tests — training scheduler: FIFO order, single-flight, cancel,
error capture from a failing worker stub, and the auto-trigger check.

Workers are python stub scripts (injectable worker_command), so nothing here
needs GPU/torch.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sable.registry import ModelRegistry  # noqa: E402
from sable.store.db import TraceRow, TraceStore  # noqa: E402
from sable.training.scheduler import (  # noqa: E402
    Scheduler,
    SchedulerError,
    TrainingConfig,
    resource_gate_ok,
    should_auto_train,
)

STUB_WORKER = r'''
import argparse, json, os, sys, tempfile, time
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

# behaviour knobs via env (per-invocation in tests)
mode = os.environ.get("STUB_MODE", "ok")
if mode == "sleep":
    emit("start")
    time.sleep(float(os.environ.get("STUB_SLEEP", "30")))
    emit("done")
elif mode == "fail":
    emit("error", error="stub failure")
    sys.exit(3)
else:
    emit("start")
    # produce artifacts so pipeline can find them. NEVER fall back to the CWD:
    # jobs submitted without a train.yaml must not leak an adapter/ dir into
    # whatever directory pytest happens to run from (use a temp dir instead).
    if a.kind in ("sft", "dpo", "grpo"):
        out = Path(a.config).parent if a.config else Path(tempfile.mkdtemp())
        (out / "adapter").mkdir(parents=True, exist_ok=True)
        (out / "adapter" / "adapter_config.json").write_text("{}")
    elif a.kind == "export":
        if a.gguf_out:
            gg = Path(a.gguf_out)
            gg.parent.mkdir(parents=True, exist_ok=True)
            gg.write_bytes(b"GGUF-stub")
    emit("done")
'''


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "uf4"
    monkeypatch.setenv("SABLE_HOME", str(h))
    return h


@pytest.fixture()
def registry(home):
    reg = ModelRegistry(home / "buffer" / "sable.db")
    yield reg
    reg.close()


@pytest.fixture()
def stub(tmp_path):
    p = tmp_path / "stub_worker.py"
    p.write_text(STUB_WORKER)
    return str(p)


def make_sched(registry, stub, env=None):
    sched = Scheduler(
        registry=registry,
        cfg=TrainingConfig(),
        worker_command=[sys.executable, stub],
    )
    return sched


def submit(sched, registry, kind="sft"):
    return sched.submit(kind, dataset_id="ds1", log_dir=Path("/tmp") / "uf4logs")


# ------------------------------------------------------------------- ordering

def test_fifo_order(registry, stub):
    sched = make_sched(registry, stub)
    j1 = sched.submit("sft", log_dir=None)
    j2 = sched.submit("export", log_dir=None)
    assert registry.next_queued().job_id == j1.job_id
    results = sched.run_pending()
    assert [r.job_id for r in results] == [j1.job_id, j2.job_id]
    assert all(r.state == "done" for r in results)


def test_single_flight(registry, stub, monkeypatch):
    """A running job blocks a second from starting (second waits/raises)."""
    sched = make_sched(registry, stub)
    monkeypatch.setenv("STUB_MODE", "sleep")
    monkeypatch.setenv("STUB_SLEEP", "5")
    j1 = sched.submit("sft")
    import threading

    t = threading.Thread(target=lambda: sched.run_job(j1), daemon=True)
    t.start()
    # wait until job1 is marked running
    for _ in range(100):
        if registry.get_job(j1.job_id).state == "running":
            break
        time.sleep(0.05)

    j2 = sched.submit("sft")
    with pytest.raises(SchedulerError):
        sched.run_job(j2)  # refuses to co-schedule on one GPU
    assert registry.get_job(j2.job_id).state == "queued"

    monkeypatch.setenv("STUB_MODE", "ok")  # affects future spawns only
    sched.cancel(j1.job_id)
    t.join(timeout=10)


def test_parallel_slots_allow_concurrent_jobs(registry, stub, monkeypatch):
    """max_parallel_jobs=2: two jobs co-schedule on device slots 0 and 1;
    a third is refused. Device env comes from the slot mapping."""
    sched = Scheduler(
        registry=registry,
        cfg=TrainingConfig(max_parallel_jobs=2, gpu_ids=[5, 6]),
        worker_command=[sys.executable, stub],
    )
    monkeypatch.setenv("STUB_MODE", "sleep")
    monkeypatch.setenv("STUB_SLEEP", "60")
    import threading

    j1 = sched.submit("sft")
    j2 = sched.submit("sft")
    t1 = threading.Thread(target=lambda: sched.run_job(j1), daemon=True)
    t2 = threading.Thread(target=lambda: sched.run_job(j2), daemon=True)
    t1.start(); t2.start()
    states: set = set()
    for _ in range(100):
        states = {registry.get_job(j.job_id).state for j in (j1, j2)}
        if states == {"running"}:
            break
        time.sleep(0.05)
    assert states == {"running"}, "both slots should run concurrently"
    # device slots recorded from the gpu_ids mapping
    devs = sorted(registry.get_job(j.job_id).params["_device"] for j in (j1, j2))
    assert devs == [0, 1]

    j3 = sched.submit("sft")
    with pytest.raises(SchedulerError):
        sched.run_job(j3)  # all slots busy

    sched.cancel(j1.job_id); sched.cancel(j2.job_id)
    t1.join(timeout=10); t2.join(timeout=10)


def test_device_env_pinned_per_slot(registry, stub, monkeypatch, tmp_path):
    """The worker subprocess sees CUDA/HIP_VISIBLE_DEVICES for its slot."""
    probe = tmp_path / "env_probe.py"
    # run_job redirects the worker's stdout to DEVNULL when a log path is
    # set, so the probe records what it saw into SABLE_PROBE_OUT instead.
    # Output file is named after the device the probe SAW, from a dir set
    # once — no per-thread env var to race on between submit and spawn.
    probe.write_text(
        "import json, os, time\n"
        "d = os.environ.get('SABLE_PROBE_DIR')\n"
        "if d:\n"
        "    cuda = os.environ.get('CUDA_VISIBLE_DEVICES', 'none')\n"
        "    with open(os.path.join(d, cuda + '.json'), 'w') as f:\n"
        "        json.dump({'cuda': cuda,"
        " 'hip': os.environ.get('HIP_VISIBLE_DEVICES')}, f)\n"
        "time.sleep(1.5)  # hold the slot so the second claim overlaps\n"
    )
    sched = Scheduler(
        registry=registry,
        cfg=TrainingConfig(max_parallel_jobs=2, gpu_ids=[3, 7]),
        worker_command=[sys.executable, str(probe)],
    )
    j1 = sched.submit("sft", log_dir=None)
    j2 = sched.submit("sft", log_dir=None)

    outs_dir = tmp_path / "probe"
    outs_dir.mkdir()
    monkeypatch.setenv("SABLE_PROBE_DIR", str(outs_dir))
    # Both must run CONCURRENTLY: sequentially, the second job would reuse
    # the freed slot 0. The probe writes <device>.json for the device it saw.
    import threading

    def run(j):
        try:
            sched.run_job(j)
        except Exception:
            pass

    t1 = threading.Thread(target=run, args=(j1,))
    t2 = threading.Thread(target=run, args=(j2,))
    t1.start(); t2.start(); t1.join(timeout=30); t2.join(timeout=30)

    seen = sorted(p.stem for p in outs_dir.glob("*.json"))
    assert seen == ["3", "7"], f"expected one worker per device, saw {seen}"
    for p in outs_dir.glob("*.json"):
        data = json.loads(p.read_text())
        assert data["cuda"] == data["hip"]


def test_cancel_running_sigterms_worker(registry, stub, monkeypatch):
    sched = make_sched(registry, stub)
    monkeypatch.setenv("STUB_MODE", "sleep")
    monkeypatch.setenv("STUB_SLEEP", "60")
    j = sched.submit("sft")
    import threading

    t = threading.Thread(target=lambda: sched.run_job(j), daemon=True)
    t.start()
    job = None
    for _ in range(100):
        job = registry.get_job(j.job_id)
        if job.state == "running" and job.pid:
            break
        time.sleep(0.05)
    assert job is not None and job.pid
    pid = job.pid
    sched.cancel(j.job_id)
    t.join(timeout=15)
    assert registry.get_job(j.job_id).state == "cancelled"
    # worker process gone (SIGTERM'd)
    time.sleep(0.3)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_cancel_queued(registry, stub):
    sched = make_sched(registry, stub)
    j = sched.submit("sft")
    cancelled = sched.cancel(j.job_id)
    assert cancelled is not None and cancelled.state == "cancelled"
    # run_pending must skip cancelled jobs
    assert sched.run_pending() == []


def test_error_capture_from_failing_worker(registry, stub, monkeypatch):
    sched = make_sched(registry, stub)
    monkeypatch.setenv("STUB_MODE", "fail")
    j = sched.submit("sft", log_dir=Path(str(registry.db_path.parent)) / "runs")
    done = sched.run_job(j)
    assert done.state == "error"
    assert "rc=3" in (done.error or "")
    # worker's own error line captured from the log tail
    assert "stub failure" in (done.error or "")


def test_reap_stale_dead_pid(registry, stub):
    sched = make_sched(registry, stub)
    j = sched.submit("sft")
    registry.mark_running(j.job_id, pid=999_999)  # dead pid
    reaped = sched.reap_stale()
    assert j.job_id in reaped
    assert registry.get_job(j.job_id).state == "error"


def test_worker_log_written(registry, stub):
    sched = make_sched(registry, stub)
    j = sched.submit("sft", log_dir=Path(str(registry.db_path.parent)) / "runs2")
    sched.run_job(j)
    assert j.log_path is not None
    logp = Path(j.log_path)
    assert logp.is_file()
    events = [json.loads(l)["event"] for l in logp.read_text().splitlines()]
    assert "start" in events and "done" in events


# ---------------------------------------------------------------- auto check

def seed(store, home, n, score, prefix="t"):
    for i in range(n):
        tid = f"{prefix}{i}"
        p = home / "buffer" / f"{tid}.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("")
        store.insert_trace(TraceRow(
            trace_id=tid, session_id=None, timestamp="2026-09-01T00:00:00+00:00",
            agent_profile="fallback", model_alias="m", model_version=None,
            endpoint_label=None, task_summary="s", status="success",
            outcome_score=score, num_tool_calls=0, tokens_in=None,
            tokens_out=None, duration_s=None, jsonl_path=str(p),
        ))


def test_should_auto_train(home):
    store = TraceStore(home / "buffer" / "sable.db")
    d = should_auto_train(store, min_traces=3, min_score=0.5)
    assert d["trigger"] is False and d["unclaimed_traces"] == 0

    seed(store, home, 2, 0.9)
    d = should_auto_train(store, min_traces=3, min_score=0.5)
    assert d["trigger"] is False  # not enough traces

    seed(store, home, 1, 0.9, prefix="u")
    d = should_auto_train(store, min_traces=3, min_score=0.5)
    assert d["trigger"] is True and d["mean_score"] == pytest.approx(0.9)

    # low mean -> no trigger
    store2_path = home / "buffer" / "other.db"
    store2 = TraceStore(store2_path)
    seed(store2, home, 5, 0.1)
    d = should_auto_train(store2, min_traces=3, min_score=0.5)
    assert d["trigger"] is False
    store.close(); store2.close()


def seed_cancelled(store, home, n, score, prefix="c"):
    """HIGH #4 shape: pre-fix recorder stamped cancelled runs with -0.5."""
    for i in range(n):
        tid = f"{prefix}{i}"
        p = home / "buffer" / f"{tid}.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("")
        store.insert_trace(TraceRow(
            trace_id=tid, session_id=None, timestamp="2026-09-01T00:00:00+00:00",
            agent_profile="fallback", model_alias="m", model_version=None,
            endpoint_label=None, task_summary="s", status="failure",
            outcome_score=score, num_tool_calls=0, tokens_in=None,
            tokens_out=None, duration_s=None, jsonl_path=str(p),
            failure_reason="cancelled",
        ))


def test_should_auto_train_ignores_cancelled_runs(home):
    """Cancelled runs must not move the mean-score gate (HIGH #4).

    Post-fix recorders NULL cancelled scores at write time; the db.py
    migration retro-NULLs legacy stamped rows on open. Either way, by the
    time should_auto_train reads the buffer the cancelled trace contributes
    no score — the mean stays at the non-cancelled value.
    """
    store = TraceStore(home / "buffer" / "sable.db")
    seed(store, home, 3, 0.9)
    d = should_auto_train(store, min_traces=3, min_score=0.5)
    assert d["trigger"] is True and d["mean_score"] == pytest.approx(0.9)

    # legacy stamped -0.5 cancelled row: the TraceStore-open migration NULLs it
    seed_cancelled(store, home, 1, -0.5)
    store.close()
    store = TraceStore(home / "buffer" / "sable.db")  # reopen -> migration
    d = should_auto_train(store, min_traces=3, min_score=0.5)
    assert d["trigger"] is True
    assert d["mean_score"] == pytest.approx(0.9), \
        "cancelled run must not drag the auto-train mean"
    # it still counts as an unclaimed trace row (count, not score)
    assert d["unclaimed_traces"] == 4 and d["scored_traces"] == 3
    store.close()


def test_resource_gate_skips_without_nvidia(monkeypatch):
    cfg = TrainingConfig(min_free_vram_mib=999_999, require_ac_power=True)
    monkeypatch.setattr("sable.training.scheduler.gpu_free_mib", lambda: None)
    monkeypatch.setattr("sable.training.scheduler.on_ac_power", lambda: None)
    assert resource_gate_ok(cfg)["ok"] is True  # unknown => skip gracefully

    monkeypatch.setattr("sable.training.scheduler.gpu_free_mib", lambda: 100)
    assert resource_gate_ok(cfg)["ok"] is False


# ---------------------------------------------------------- AMD/ROCm VRAM gate

_ROCM_JSON = json.dumps({
    "card0": {
        "GPU": "0",
        "VRAM Total Memory (B)": 103079215104,   # ~98 GiB unified pool
        "VRAM Total Used Memory (B)": 2147483648,  # 2 GiB used
    }
})

_ROCM_TEXT = (
    "GPU[0]\t\t: VRAM Total Memory (B): 51539607552\n"
    "GPU[0]\t\t: VRAM Total Used Memory (B): 51539607552\n"
)


class _Proc:
    def __init__(self, rc, out):
        self.returncode, self.stdout = rc, out


def test_rocm_free_mib_json(monkeypatch):
    import sable.training.scheduler as sched

    monkeypatch.setattr(sched.shutil, "which", lambda n: "/usr/bin/rocm-smi" if n == "rocm-smi" else None)
    monkeypatch.setattr(sched.subprocess, "run", lambda *a, **k: _Proc(0, _ROCM_JSON))
    # (103079215104 - 2147483648) bytes = 96296 MiB free
    assert sched._rocm_free_mib() == (103079215104 - 2147483648) // (1024 * 1024)


def test_rocm_free_mib_text_fallback(monkeypatch):
    import sable.training.scheduler as sched

    monkeypatch.setattr(sched.shutil, "which", lambda n: "/usr/bin/rocm-smi" if n == "rocm-smi" else None)
    calls = []

    def fake_run(*a, **k):
        calls.append(a[0])
        # --json unsupported -> rc!=0 first, then text table
        return _Proc(1, "") if "--json" in a[0] else _Proc(0, _ROCM_TEXT)

    monkeypatch.setattr(sched.subprocess, "run", fake_run)
    assert sched._rocm_free_mib() == 0  # fully used pool -> 0 MiB free


def test_gpu_free_mib_falls_through_vendors(monkeypatch):
    import sable.training.scheduler as sched

    # nvidia probe fails (e.g. busy/unsupported) -> rocm result still wins.
    monkeypatch.setattr(sched, "_nvidia_free_mib", lambda: None)
    monkeypatch.setattr(sched, "_rocm_free_mib", lambda: 12345)
    assert sched.gpu_free_mib() == 12345
    monkeypatch.setattr(sched, "_rocm_free_mib", lambda: None)
    assert sched.gpu_free_mib() is None


# ------------------------------------------- deep-dive fixes: cancel + slots

def test_cancel_inprocess_kind_never_signals_own_pid(registry, stub, monkeypatch):
    """In-process jobs record OUR pid; cancelling must not SIGTERM ourselves."""

    sched = make_sched(registry, stub)
    j = sched.submit("curate")
    registry.mark_running(j.job_id, os.getpid())

    import sable.training.scheduler as sched_mod

    sent = []
    monkeypatch.setattr(sched_mod.os, "kill",
                        lambda pid, sig: sent.append((pid, sig)))
    out = sched.cancel(j.job_id)
    assert out is not None and out.state == "cancelled"
    assert sent == [], f"must not signal any pid, sent {sent}"


def test_cancel_self_pid_guard_even_for_gpu_kinds(registry, stub, monkeypatch):
    """Defense in depth: a job whose pid equals ours is never signalled."""
    sched = make_sched(registry, stub)
    j = sched.submit("sft")
    registry.mark_running(j.job_id, os.getpid())

    import sable.training.scheduler as sched_mod

    sent = []
    monkeypatch.setattr(sched_mod.os, "kill",
                        lambda pid, sig: sent.append((pid, sig)))
    out = sched.cancel(j.job_id)
    assert out is not None and out.state == "cancelled"
    assert sent == []


def test_run_inprocess_aborts_when_cancelled_while_queued(registry, stub):
    """A cancel landing before the runner starts must keep 'cancelled'."""
    ran = []
    sched = Scheduler(registry=registry, cfg=TrainingConfig(),
                      worker_command=[sys.executable, stub],
                      inprocess_runners={"curate": lambda job: ran.append(job.job_id)})
    j = sched.submit("curate")
    sched.cancel(j.job_id)
    # Direct dispatch of the cancelled job (bypassing run_pending's queue read):
    # mark_running must refuse and the runner must never execute.
    sched._run_inprocess(registry.get_job(j.job_id), slot=0, wait_gate=False)
    assert ran == []
    assert registry.get_job(j.job_id).state == "cancelled"


def test_slot_lock_excludes_other_process(registry, stub):
    """A flock held elsewhere (simulating another sable process's worker on
    device 0) makes the claim fall through / refuse instead of double-booking."""
    import fcntl

    sched = make_sched(registry, stub)
    lock_path = Path(str(registry.db_path) + ".slot0.lock")
    holder = open(lock_path, "a+")
    fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        assert sched._claim_device_slot("job-x") is None  # only slot 0 exists
    finally:
        holder.close()
    # after release the same claim succeeds
    claim = sched._claim_device_slot("job-x")
    assert claim is not None and claim[0] == 0
    sched._release_device_slot("job-x", claim[1])


def test_training_config_warns_on_unknown_keys(caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="sable.training"):
        TrainingConfig.from_yaml({"training": {"min_free_vram_mb": 8192,
                                               "eval_seed": 7}})
    assert any("min_free_vram_mb" in r.message for r in caplog.records)


# ------------------------------------------------------ worker timeout (gap #4)

def test_worker_timeout_default_is_finite():
    """The default must NOT be 0: a hung torch worker would hold its device
    slot flock forever. 4 h is the documented sane cap; 0 stays as an opt-in
    escape hatch."""
    assert TrainingConfig().worker_timeout_s == 14400.0
    assert TrainingConfig.from_yaml(
        {"training": {"worker_timeout_s": 0}}).worker_timeout_s == 0


def test_worker_timeout_kills_hung_worker_and_releases_slot(registry, stub,
                                                            monkeypatch):
    """A worker sleeping past worker_timeout_s gets SIGTERM'd (then SIGKILL
    after the grace), the job ends 'error' with a timeout reason, and the
    device slot is claimable again immediately."""
    monkeypatch.setenv("STUB_MODE", "sleep")
    monkeypatch.setenv("STUB_SLEEP", "60")
    sched = Scheduler(registry=registry,
                      cfg=TrainingConfig(worker_timeout_s=0.5),
                      worker_command=[sys.executable, stub])
    j = sched.submit("sft", log_dir=None)
    done = sched.run_job(j)          # returns after terminate+reap (~<10 s)
    assert done.state == "error"
    assert "timeout" in (done.error or "")
    # worker really died: the recorded pid no longer exists
    worker_pid = registry.get_job(j.job_id).pid
    if worker_pid:
        with pytest.raises(ProcessLookupError):
            os.kill(worker_pid, 0)
    # and the device slot flock was released by run_job's finally-block
    claim = sched._claim_device_slot("next-job")
    assert claim is not None
    sched._release_device_slot("next-job", claim[1])


def test_worker_timeout_zero_disables_wait_timeout(registry, stub, monkeypatch):
    """Explicit 0 keeps the escape hatch: proc.wait() is called with
    timeout=None (no cap). Stubbed Popen — nothing is spawned."""
    calls = {}

    class FakeProc:
        pid = 424242

        def wait(self, timeout=None):
            calls["timeout"] = timeout
            return 0

    monkeypatch.setattr("sable.training.scheduler.subprocess.Popen",
                        lambda *a, **k: FakeProc())
    sched = Scheduler(registry=registry,
                      cfg=TrainingConfig(worker_timeout_s=0.0),
                      worker_command=[sys.executable, stub])
    j = sched.submit("sft", log_dir=None)
    done = sched.run_job(j)
    assert calls["timeout"] is None
    assert done.state == "done"


def test_worker_timeout_nonzero_passed_to_wait(registry, stub, monkeypatch):
    """Non-zero config reaches proc.wait(timeout=...) verbatim."""
    calls = {}

    class FakeProc:
        pid = 424243

        def wait(self, timeout=None):
            calls["timeout"] = timeout
            return 0

    monkeypatch.setattr("sable.training.scheduler.subprocess.Popen",
                        lambda *a, **k: FakeProc())
    sched = Scheduler(registry=registry,
                      cfg=TrainingConfig(worker_timeout_s=123.0),
                      worker_command=[sys.executable, stub])
    j = sched.submit("sft", log_dir=None)
    done = sched.run_job(j)
    assert calls["timeout"] == 123.0
    assert done.state == "done"

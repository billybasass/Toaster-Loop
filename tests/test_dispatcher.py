"""Training-queue dispatcher + stale-job reaper tests.

Covers the three production gaps one dispatcher loop closes:
(a) queued jobs draining without a submit thread (and two rapid submits both
executing), (b) phantom `running` rows reaped so submits stop 409-ing,
(c) concurrent claimers never double-claiming a row, (d) gate-blocked jobs
re-attempted once their gate passes, (e) one dispatcher per DB enforced by a
pid-lockfile with dead-holder reclaim. Plus the startup workspace-root logging.

Workers are python stub scripts (injectable worker_command), so nothing here
needs GPU/torch — same seams as test_scheduler.py.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sable.core.config import SableConfig  # noqa: E402
from sable.dashboard.app import DashboardApp  # noqa: E402
from sable.registry import ModelRegistry  # noqa: E402
from sable.training.scheduler import (  # noqa: E402
    DISPATCH_TICK_S,
    DispatcherAlreadyRunning,
    JobDispatcher,
    Scheduler,
    TrainingConfig,
    _dispatcher_lock_path,
)

STUB_WORKER = r'''
import argparse, json, os, sys, time
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

mode = os.environ.get("STUB_MODE", "ok")
if mode == "sleep":
    emit("start")
    time.sleep(float(os.environ.get("STUB_SLEEP", "30")))
    emit("done")
else:
    emit("start")
    emit("done")
'''


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "ufdisp"
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


@pytest.fixture(autouse=True)
def _reset_slot_bookkeeping():
    """Per-test isolation for the module-global device-slot state.

    _INFLIGHT_CLAIMS is process-global (same family as conftest's
    _reset_runtime_counters): a daemon execution thread from an earlier test
    can still hold slot 0 briefly; without this reset the next test's claim
    would refuse against a phantom in-flight owner."""
    import sable.training.scheduler as sched_mod

    with sched_mod._SLOT_LOCK:
        sched_mod._INFLIGHT_CLAIMS.clear()
    yield
    with sched_mod._SLOT_LOCK:
        sched_mod._INFLIGHT_CLAIMS.clear()


def make_sched(registry, stub):
    return Scheduler(registry=registry, cfg=TrainingConfig(),
                     worker_command=[sys.executable, stub])


def wait_for(fn, timeout=15.0, interval=0.05):
    """Poll fn() until truthy; returns its value (last on timeout)."""
    deadline = time.time() + timeout
    val = None
    while time.time() < deadline:
        val = fn()
        if val:
            return val
        time.sleep(interval)
    return val


# ------------------------------------------------- (a) queue drain, no stranding

def test_drain_queue_runs_every_queued_job(registry, stub):
    """Jobs enqueued without any submit thread still execute when a holder
    drains — the exact 'nothing ever drains the queue' defect."""
    sched = make_sched(registry, stub)
    j1 = registry.enqueue_job("sft")
    j2 = registry.enqueue_job("export")
    finished = sched.drain_queue()
    assert {j.job_id for j in finished} == {j1.job_id, j2.job_id}
    states = [registry.get_job(j.job_id).state for j in (j1, j2)]
    assert states == ["done", "done"]


def test_dispatcher_loop_drains_without_submit_thread(home, registry, stub):
    """A JobDispatcher tick picks up a queued job nobody is waiting on."""
    def factory():
        return make_sched(ModelRegistry(home / "buffer" / "sable.db"), stub)

    j = registry.enqueue_job("sft")
    d = JobDispatcher(db_path=registry.db_path, scheduler_factory=factory,
                      tick_s=0.2)
    d.start()
    try:
        ok = wait_for(lambda: registry.get_job(j.job_id).state == "done")
        assert ok, f"job stranded in {registry.get_job(j.job_id).state}"
    finally:
        d.stop()


def test_two_rapid_submits_both_execute(home, monkeypatch, stub):
    """(a) POST /api/train twice fast with two slots: both jobs execute,
    neither strands in `queued`."""
    monkeypatch.setenv("SABLE_WORKER_COMMAND", f"{sys.executable} {stub}")
    app = DashboardApp(SableConfig(training={"max_parallel_jobs": 2}),
                       home=home)
    for ds in ("ds-a", "ds-b"):
        d = home / "datasets" / ds
        d.mkdir(parents=True)
        (d / "train.yaml").write_text("model: base\ndata: {}\n")

    r1 = app.handle("POST", "/api/train", body={"dataset": "ds-a"})
    r2 = app.handle("POST", "/api/train", body={"dataset": "ds-b"})
    assert r1.status == 202 and r2.status == 202
    ids = [r1.body["job_id"], r2.body["job_id"]]

    reg = ModelRegistry(home / "buffer" / "sable.db")

    def _state(job_id):
        cur = reg.get_job(job_id)
        assert cur is not None
        return cur.state

    try:
        ok = wait_for(lambda: all(
            _state(i) in ("done", "error", "cancelled")
            for i in ids))
        assert ok, [_state(i) for i in ids]
        # both actually ran (not cancelled by a race) and nothing is stranded
        assert {_state(i) for i in ids} == {"done"}
        assert reg.list_jobs("queued") == []
    finally:
        reg.close()


# ------------------------------------------- (b) phantom running rows get reaped

def test_phantom_running_row_reaped_at_startup(home, registry):
    """A dead worker's `running` row must not survive boot holding submits."""
    j = registry.enqueue_job("sft")
    registry.mark_running(j.job_id, pid=999_999)  # dead pid -> phantom

    app = DashboardApp(SableConfig(), home=home)
    try:
        d = app.startup_dispatcher()
        assert d is not None
        row = wait_for(lambda: registry.get_job(j.job_id).state == "error")
        assert row, registry.get_job(j.job_id).state
        assert "reaped" in (registry.get_job(j.job_id).error or "")
    finally:
        app.stop_dispatcher()


def test_phantom_row_no_longer_returns_409(home, registry, monkeypatch, stub):
    """(b) With the phantom reaped by the submit-path pre-check, POST /api/train
    is accepted instead of 409 already_running forever."""
    monkeypatch.setenv("SABLE_WORKER_COMMAND", f"{sys.executable} {stub}")
    j = registry.enqueue_job("sft")
    registry.mark_running(j.job_id, pid=999_999)

    app = DashboardApp(SableConfig(), home=home)
    ds = home / "datasets" / "ds-p"
    ds.mkdir(parents=True)
    (ds / "train.yaml").write_text("model: base\ndata: {}\n")

    r = app.handle("POST", "/api/train", body={"dataset": "ds-p"})
    assert r.status == 202, r.body
    ok = wait_for(lambda: registry.get_job(j.job_id).state == "error")
    assert ok  # phantom reaped; new job accepted


def test_live_running_row_still_returns_409(home, registry):
    """The reap must not weaken the real single-flight guard: a LIVE running
    row (our own pid) still refuses with 409 already_running."""
    app = DashboardApp(SableConfig(), home=home)
    j = registry.enqueue_job("sft")
    registry.mark_running(j.job_id, os.getpid())  # alive -> not reaped

    r = app.handle("POST", "/api/train", body={"dataset": "d1"})
    assert r.status == 409 and r.body["error"]["code"] == "already_running"
    assert j.job_id in r.body["error"]["message"]


# --------------------------------------- (c) concurrent claimers, one owner

def test_concurrent_claimers_never_double_claim(home):
    """(c) N threads racing claim_next_queued over M rows: every row claimed at
    most once (guarded UPDATE + IMMEDIATE transaction)."""
    reg_a = ModelRegistry(home / "buffer" / "sable.db")
    ids = [reg_a.enqueue_job("sft").job_id for _ in range(12)]

    claimed: list = []
    claim_lock = threading.Lock()

    def racer():
        reg = ModelRegistry(home / "buffer" / "sable.db")
        try:
            while True:
                job = reg.claim_next_queued(os.getpid())
                if job is None:
                    return
                with claim_lock:
                    claimed.append(job.job_id)
        finally:
            reg.close()

    threads = [threading.Thread(target=racer) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert sorted(claimed) == sorted(ids), "every row claimed exactly once"
    # all rows landed running under the claim
    reg_b = ModelRegistry(home / "buffer" / "sable.db")
    try:
        for i in ids:
            cur = reg_b.get_job(i)
            assert cur is not None and cur.state == "running"
        assert reg_b.list_jobs("queued") == []
    finally:
        reg_a.close()
        reg_b.close()


def test_requeue_only_releases_own_unspawned_claim(registry):
    """requeue_job is pid-guarded: it releases this process's placeholder claim
    but never another holder's running row or a terminal state."""
    mine = registry.enqueue_job("sft")
    claimed = registry.claim_next_queued(os.getpid())
    assert claimed is not None and claimed.job_id == mine.job_id
    assert registry.requeue_job(mine.job_id, os.getpid()) is True
    assert registry.get_job(mine.job_id).state == "queued"

    # a genuinely running row (different pid) cannot be requeued by us
    theirs = registry.enqueue_job("sft")
    registry.mark_running(theirs.job_id, pid=999_998)
    assert registry.requeue_job(theirs.job_id, os.getpid()) is False
    assert registry.get_job(theirs.job_id).state == "running"


def test_claim_skips_dependents_and_kinds(registry):
    """claim_next_queued honors #43 dependency edges and skip_kinds — same
    queue-skipping semantics as next_queued."""
    a = registry.enqueue_job("sft")
    b = registry.enqueue_job("export", depends_on=[a.job_id])
    c = registry.enqueue_job("curate")

    first = registry.claim_next_queued(os.getpid(), skip_kinds=("curate",))
    assert first is not None and first.job_id == a.job_id  # b blocked, c skipped
    registry.finish_job(a.job_id, "done")

    nxt = registry.claim_next_queued(os.getpid(), skip_kinds=("curate",))
    assert nxt is not None and nxt.job_id == b.job_id
    assert registry.claim_next_queued(os.getpid(),
                                      skip_kinds=("curate",)) is None


# ------------------------------------------- (d) gate-blocked jobs get retried

def test_gate_blocked_job_retried_when_gate_passes(registry, stub, monkeypatch):
    """(d) A job blocked by the VRAM gate stays queued with its _gate_reason;
    once the gate passes, a later drain runs it. The gate itself is never
    weakened — while it fails, the claim is simply released back to queued."""
    import sable.training.scheduler as sched_mod

    monkeypatch.setattr(sched_mod, "gpu_free_mib", lambda: 100)
    sched = Scheduler(registry=registry,
                      cfg=TrainingConfig(min_free_vram_mib=99_999),
                      worker_command=[sys.executable, stub])
    j = sched.submit("sft")

    assert sched.drain_queue() == []  # gate refuses -> claim released
    row = registry.get_job(j.job_id)
    assert row.state == "queued", "blocked job must not strand claimed/running"
    assert "free memory" in (row.params.get("_gate_reason") or "")

    # gate still closed: another drain attempt keeps it queued, reason intact
    assert sched.drain_queue() == []
    assert registry.get_job(j.job_id).state == "queued"

    monkeypatch.setattr(sched_mod, "gpu_free_mib", lambda: 10 ** 7)
    finished = sched.drain_queue()  # gate passes -> retried and completed
    assert [x.job_id for x in finished] == [j.job_id]
    row = registry.get_job(j.job_id)
    assert row.state == "done"
    assert row.params.get("_gate_reason") is None  # cleared on successful start


def test_gate_block_events_dedup_per_episode(registry, stub, monkeypatch):
    """Gate-block events are cumulative but episode-deduped: repeated blocked
    drain attempts (the dispatcher retries a queued job every tick) with the
    SAME reason record ONE event; unblocking and re-blocking records a second.
    This is what makes sable_gate_blocks_total a true counter while the
    jobs-side _gate_reason stays a cleared-on-unblock gauge."""
    import sqlite3

    import sable.training.scheduler as sched_mod

    def gate_events():
        conn = sqlite3.connect(str(registry.db_path))
        try:
            return conn.execute(
                "SELECT job_id, reason FROM gate_events ORDER BY id").fetchall()
        finally:
            conn.close()

    monkeypatch.setattr(sched_mod, "gpu_free_mib", lambda: 100)
    sched = Scheduler(registry=registry,
                      cfg=TrainingConfig(min_free_vram_mib=99_999),
                      worker_command=[sys.executable, stub])
    j = sched.submit("sft")

    # several blocked drain attempts (dispatcher-tick loop shape): the same
    # gate reason every time -> exactly one episode event.
    for _ in range(4):
        assert sched.drain_queue() == []
    assert registry.get_job(j.job_id).state == "queued"
    assert len(gate_events()) == 1

    # unblock (job runs; reason cleared) then re-block on a fresh episode:
    monkeypatch.setattr(sched_mod, "gpu_free_mib", lambda: 10 ** 7)
    sched.drain_queue()
    assert registry.get_job(j.job_id).state == "done"

    j2 = sched.submit("sft")
    monkeypatch.setattr(sched_mod, "gpu_free_mib", lambda: 100)
    for _ in range(3):
        assert sched.drain_queue() == []
    ev = gate_events()
    assert len(ev) == 2, "second block episode must add exactly one event"
    assert ev[1][0] == j2.job_id


def test_run_claimed_releases_gate_refusal(registry, stub, monkeypatch):
    """run_claimed turns a SchedulerError (gate/slot busy) into a requeued job
    owned by nobody, so the next tick can retry it."""
    import sable.training.scheduler as sched_mod

    monkeypatch.setattr(sched_mod, "gpu_free_mib", lambda: 100)
    sched = Scheduler(registry=registry,
                      cfg=TrainingConfig(min_free_vram_mib=99_999),
                      worker_command=[sys.executable, stub])
    j = registry.enqueue_job("sft")
    claimed = registry.claim_next_queued(os.getpid())
    assert claimed is not None and claimed.job_id == j.job_id

    with pytest.raises(sched_mod.SchedulerError):
        sched.run_claimed(claimed)
    assert registry.get_job(j.job_id).state == "queued"


# ------------------------------- (e) one dispatcher per DB: pid-lockfile

def test_second_dispatcher_refuses_while_lock_held(home, registry, stub):
    """(e) Contention fails loudly naming the holder pid and lock path; after
    the holder stops, a new dispatcher starts."""
    def factory():
        return make_sched(ModelRegistry(home / "buffer" / "sable.db"), stub)

    d1 = JobDispatcher(db_path=registry.db_path, scheduler_factory=factory,
                       tick_s=0.2)
    d1.start()
    try:
        lock_path = _dispatcher_lock_path(registry.db_path)
        assert lock_path.is_file()
        assert lock_path.read_text().strip() == str(os.getpid())

        d2 = JobDispatcher(db_path=registry.db_path,
                           scheduler_factory=factory, tick_s=0.2)
        with pytest.raises(DispatcherAlreadyRunning) as ei:
            d2.start()
        msg = str(ei.value)
        assert str(lock_path) in msg and str(os.getpid()) in msg
    finally:
        d1.stop()

    # lock released -> a fresh dispatcher starts (stale pid reclaimed)
    d3 = JobDispatcher(db_path=registry.db_path, scheduler_factory=factory,
                       tick_s=0.2)
    d3.start()
    try:
        assert d3.is_running()
    finally:
        d3.stop()


def test_stale_lock_from_dead_pid_is_reclaimed(home, registry, stub):
    """A lockfile left behind by a dead process (no flock held) never wedges
    the queue: start succeeds and rewrites the recorded pid."""
    def factory():
        return make_sched(ModelRegistry(home / "buffer" / "sable.db"), stub)

    lock_path = _dispatcher_lock_path(registry.db_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("999999")  # dead holder's stale pid, no flock

    d = JobDispatcher(db_path=registry.db_path, scheduler_factory=factory,
                      tick_s=0.2)
    d.start()
    try:
        assert d.is_running()
        assert lock_path.read_text().strip() == str(os.getpid())
    finally:
        d.stop()


def test_app_ensure_dispatcher_contention(home):
    """Two DashboardApps over one home: the second ensure_dispatcher raises,
    and succeeds again once the first app stops its dispatcher."""
    app1 = DashboardApp(SableConfig(), home=home)
    app2 = DashboardApp(SableConfig(), home=home)
    try:
        assert app1.ensure_dispatcher() is not None
        with pytest.raises(DispatcherAlreadyRunning):
            app2.ensure_dispatcher()
    finally:
        app1.stop_dispatcher()
    try:
        assert app2.ensure_dispatcher() is not None
    finally:
        app2.stop_dispatcher()


# ------------------------------------------------ startup workspace-root log

def test_workspace_root_fallback_logged(caplog, home):
    """agent.workspace_root unset -> the cwd fallback (often $HOME) is logged
    loudly at WARNING on the startup path."""
    app = DashboardApp(SableConfig(), home=home)
    try:
        with caplog.at_level(logging.WARNING, logger="sable.dashboard"):
            app.startup_dispatcher()
        assert any(
            "workspace root" in r.getMessage()
            and str(Path(os.getcwd())) in r.getMessage()
            and r.levelno == logging.WARNING
            for r in caplog.records)
    finally:
        app.stop_dispatcher()


def test_workspace_root_explicit_logged_at_info(caplog, home):
    """An explicitly configured root is logged at INFO (resolved value), no
    fallback warning."""
    ws = home / "ws"
    ws.mkdir(parents=True)
    app = DashboardApp(SableConfig(agent={"workspace_root": str(ws)}),
                       home=home)
    with caplog.at_level(logging.INFO, logger="sable.dashboard"):
        app._log_workspace_root()
    msgs = [r for r in caplog.records if "workspace root" in r.getMessage()]
    assert any(str(ws) in r.getMessage() and r.levelno == logging.INFO
               for r in msgs)
    assert not any(r.levelno == logging.WARNING for r in msgs)


def test_workspace_root_null_warns_containment_disabled(caplog, home):
    """Explicit null (UNSET sentinel) -> containment-disabled warning."""
    from sable.core.config import UNSET

    app = DashboardApp(SableConfig(agent={"workspace_root": UNSET}),
                       home=home)
    with caplog.at_level(logging.WARNING, logger="sable.dashboard"):
        app._log_workspace_root()
    assert any("DISABLED" in r.getMessage() for r in caplog.records
               if "workspace root" in r.getMessage())


# --------------------------------------------------------------- constants

def test_dispatch_tick_constant_sane():
    """The documented tick interval exists and is a sane cadence next to
    TrainingConfig.poll_interval_s (1 s)."""
    assert 1.0 <= DISPATCH_TICK_S <= 60.0

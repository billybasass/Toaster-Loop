"""WP5 GPU-coexistence tests — hermetic: injected clocks/sleepers, fake VRAM
probes, monkeypatched spawn/stop paths, real state files under tmp_path.
No GPU, no real llama-server (VRAM-free timing is FAKED here — there is no
real-GPU repro in CI)."""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sable.core.config import SableConfig  # noqa: E402
from sable.dashboard.app import DashboardApp, HttpError  # noqa: E402
from sable.registry import ModelRegistry  # noqa: E402
from sable.serving import coexist as cx  # noqa: E402
from sable.serving import instances as srv_inst  # noqa: E402
from sable.serving import watchdog as wd  # noqa: E402
from sable.training.scheduler import (  # noqa: E402
    Scheduler,
    SchedulerError,
    TrainingConfig,
)

URL = "http://127.0.0.1:8123/v1"


class FakeClock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "uf5"
    monkeypatch.setenv("SABLE_HOME", str(h))
    wd.reset_inflight()
    wd.reset_activity()
    yield h
    wd.reset_inflight()
    wd.reset_activity()


def make_instance(home, iid="alpha", gguf="/nonexistent/alpha.gguf"):
    srv_inst.save_instance(Path(home), {"id": iid, "name": iid.title(),
                                        "gguf": gguf})


def write_live_state(home, iid="alpha", url=URL, port=8123):
    # pid = our own pid => state_alive() (os.kill probe) says running, and
    # nothing here ever kills it: stop paths are always monkeypatched.
    srv_inst.write_state(Path(home), iid, {
        "mode": "spawn", "pid": os.getpid(), "host": "127.0.0.1",
        "port": port, "base_url": url, "model_name": "alpha-model",
        "gguf_path": "/nonexistent/alpha.gguf", "instance_id": iid,
    })


def write_unloaded_state(home, iid="alpha", url=URL, port=8123):
    srv_inst.write_state(Path(home), iid, {
        "mode": "spawn", "unloaded": True, "unload_reason": "idle",
        "port": port, "base_url": url, "model_name": "alpha-model",
        "gguf_path": "/nonexistent/alpha.gguf", "instance_id": iid,
    })


class FakeHandle:
    def __init__(self, base_url=URL):
        # Only .pid is ever consulted by the load path (state record).
        class _P:
            pid = os.getpid()

        self.process = _P()
        self.base_url = base_url
        self.model_name = "alpha-model"


def patch_spawn(monkeypatch, log):
    def fake_start(cfg, inst, **kw):
        log.append(("spawn", str(inst.get("id"))))
        return FakeHandle()
    monkeypatch.setattr("sable.serving.supervisor.start_instance_server",
                        fake_start)


def patch_stop(monkeypatch, log):
    def fake_stop(iid, grace=8.0):
        log.append(("stop", iid, grace))
        return True
    monkeypatch.setattr("sable.serving.supervisor.stop_instance", fake_stop)


# ----------------------------------------------------------------- config

def test_defaults_all_off():
    cc = cx.coexist_config(SableConfig())
    assert cc.idle_unload_seconds == 0.0
    assert cc.lazy_load is False
    assert cc.min_free_vram_mib == 0
    assert cc.coexist_policy == "refuse"
    assert not cc.idle_enabled and not cc.handshake_enabled
    assert not cc.any_enabled


def test_build_returns_none_when_disabled():
    assert cx.build(SableConfig(), "/tmp/whatever") is None


def test_build_armed_with_lazy_load():
    cfg = SableConfig(serve={"lazy_load": "yes"})
    coex = cx.build(cfg, "/tmp/whatever")
    assert coex is not None and coex.cfg.lazy_load is True


def test_lenient_parsing_and_bad_values():
    cfg = SableConfig(serve={
        "idle_unload_seconds": "300",       # string float ok
        "lazy_load": "on",                  # truthy string
        "min_free_vram_mib": "8192.0",      # float-ish int
        "coexist_policy": "UNLOAD",         # case-insensitive
    })
    cc = cx.coexist_config(cfg)
    assert cc.idle_unload_seconds == 300.0
    assert cc.lazy_load is True
    assert cc.min_free_vram_mib == 8192
    assert cc.coexist_policy == "unload"

    bad = SableConfig(serve={"idle_unload_seconds": "soon",
                               "min_free_vram_mib": "lots",
                               "coexist_policy": "yolo"})
    cc2 = cx.coexist_config(bad)  # warns, keeps defaults — never raises
    assert cc2.idle_unload_seconds == 0.0
    assert cc2.min_free_vram_mib == 0
    assert cc2.coexist_policy == "refuse"


def test_handshake_requires_both_keys():
    only_floor = cx.coexist_config(
        SableConfig(serve={"min_free_vram_mib": 4096}))
    assert not only_floor.handshake_enabled  # idle_unload off => no rescue
    both = cx.coexist_config(SableConfig(
        serve={"min_free_vram_mib": 4096, "idle_unload_seconds": 120}))
    assert both.handshake_enabled


# ------------------------------------------------- watchdog activity clock

def test_note_activity_and_idle_seconds():
    clock = FakeClock(t=50.0)
    assert wd.idle_seconds("http://a/v1", clock=clock) is None  # never seen
    wd.note_activity("http://a/v1/", clock=clock)               # normalizes /
    clock.advance(12.0)
    assert wd.idle_seconds("http://a/v1", clock=clock) == pytest.approx(12.0)
    # a request start refreshes the stamp too (single registry, WP5 reuse)
    wd.note_request_start("http://a/v1", clock=clock)
    assert wd.idle_seconds("http://a/v1", clock=clock) == pytest.approx(0.0)


# ------------------------------------------------------------- idle unload

def build_engine(home, *, idle=60.0, lazy=False, min_free=0, policy="refuse",
                 clock=None):
    cc = cx.CoexistConfig(idle_unload_seconds=idle, lazy_load=lazy,
                          min_free_vram_mib=min_free, coexist_policy=policy)
    events = []
    return cx.Coexistence(cc, home=home, clock=clock or FakeClock(),
                          sleep=lambda s: None,
                          emit=lambda kind, p: events.append((kind, p))), events


def test_idle_unload_fires_once_per_idle_period(home, monkeypatch):
    make_instance(home)
    write_live_state(home)
    monkeypatch.setattr("sable.serving.runtime.pid_alive", lambda pid: True)
    stops = []
    patch_stop(monkeypatch, stops)

    clock = FakeClock()
    coex, events = build_engine(home, idle=60.0, clock=clock)
    wd.note_activity(URL, clock=clock)          # seed the window from now

    clock.advance(30.0)
    coex.step()                                   # half the window: nothing
    assert stops == []

    clock.advance(31.0)                           # 61 s idle > 60 s window
    coex.step()
    assert stops == [("stop", "alpha", coex.cfg.unload_grace_seconds)]

    # state file is now an `unloaded` tomb and the event fired exactly once
    st = srv_inst.read_state(Path(home), "alpha")
    assert st["unloaded"] is True and st["unload_reason"] == "idle"
    assert "pid" not in st
    kinds = [p["event"] for k, p in events if p.get("event")]
    assert kinds == ["instance_unloaded"]

    # further steps: nothing running anymore => no second stop ever
    clock.advance(3600.0)
    coex.step()
    coex.step()
    assert len(stops) == 1


def test_idle_never_unloads_while_inflight(home, monkeypatch):
    make_instance(home)
    write_live_state(home)
    monkeypatch.setattr("sable.serving.runtime.pid_alive", lambda pid: True)
    stops = []
    patch_stop(monkeypatch, stops)

    clock = FakeClock()
    coex, _ = build_engine(home, idle=60.0, clock=clock)
    wd.note_activity(URL, clock=clock)
    wd.note_request_start(URL, clock=clock)       # request in flight
    clock.advance(10_000.0)
    coex.step()
    assert stops == []                            # never mid-request

    wd.note_request_end(URL)
    coex.step()                                   # idle again -> now allowed
    assert len(stops) == 1


def test_disabled_idle_loop_never_starts(home):
    coex, _ = build_engine(home, idle=0.0)
    assert coex.start_idle_loop() is False


# --------------------------------------------------------- lazy load paths

def test_lazy_load_via_resolve_named_instance(home, monkeypatch):
    make_instance(home)
    write_unloaded_state(home)
    spawns = []
    patch_spawn(monkeypatch, spawns)
    cfg = SableConfig(serve={"lazy_load": True})
    app = DashboardApp(cfg, home=home)

    base, _key, model = app._resolve_chat_target("alpha")
    assert base == URL and model == "alpha-model"
    assert ("spawn", "alpha") in spawns
    # the tomb was replaced by a live record
    assert srv_inst.read_state(Path(home), "alpha").get("pid") == os.getpid()


def test_lazy_off_unloaded_gets_clear_error(home):
    make_instance(home)
    write_unloaded_state(home)
    app = DashboardApp(SableConfig(), home=home)
    with pytest.raises(HttpError) as ei:
        app._resolve_chat_target("alpha")
    assert ei.value.status == 503
    assert ei.value.code == "instance_unloaded"
    msg = str(ei.value).lower()
    assert "unloaded" in msg and "lazy_load" in msg


def test_lazy_never_loaded_keeps_legacy_503(home):
    make_instance(home)  # definition only, no state file at all
    app = DashboardApp(SableConfig(), home=home)
    with pytest.raises(HttpError) as ei:
        app._resolve_chat_target("alpha")
    assert ei.value.code == "not_serving"   # pre-WP5 message untouched


def test_default_instance_tomb_lazy_and_guidance(home, monkeypatch):
    from sable.serving.deploy import write_serve_state

    tomb = {"mode": "spawn", "unloaded": True, "unload_reason": "coexist",
            "port": 8123, "base_url": URL, "model_name": "default-model"}
    write_serve_state(Path(home), tomb)
    app = DashboardApp(SableConfig(), home=home)
    with pytest.raises(HttpError) as ei:
        app._resolve_chat_target()
    assert ei.value.code == "instance_unloaded"

    # lazy on: the default instance reloads through the canonical path
    # (the config-derived default needs a gguf_path to be spawnable).
    spawns = []
    patch_spawn(monkeypatch, spawns)
    cfg2 = SableConfig(serve={"lazy_load": True},
                         gguf_path="/nonexistent/default.gguf")
    app2 = DashboardApp(cfg2, home=home)
    base, _k, model = app2._resolve_chat_target()
    # the fake handle reports its own model name; what matters is that the
    # tomb was transparently reloaded at the recorded URL.
    assert base == URL and model
    assert ("spawn", "default") in spawns


def test_agent_runner_reloads_once_on_dead_endpoint(home, monkeypatch):
    from sable.core.agent_runner import AgentBackendError, SableAgentClient

    client = SableAgentClient.__new__(SableAgentClient)
    client.base_url = URL
    calls: list = []

    def inner(messages, tools=None):
        calls.append(1)
        if len(calls) == 1:
            raise AgentBackendError("Cannot reach inference server")
        return {"ok": True}
    client._chat_inner = inner

    # no active engine => the original error propagates untouched (pre-WP5)
    monkeypatch.setattr("sable.serving.coexist.active", lambda: None)
    with pytest.raises(AgentBackendError):
        client._chat([{"role": "user", "content": "hi"}])
    assert len(calls) == 1

    # active engine WITH the URL in unloaded state => reload once + retry once
    reloads = []
    calls.clear()   # next inner call raises (len==1), retry succeeds (len==2)

    class FakeCoex:
        def reload_for_url(self, url):
            reloads.append(url)
            return True
    monkeypatch.setattr("sable.serving.coexist.active",
                        lambda: FakeCoex())
    assert client._chat([{"role": "user", "content": "hi"}]) == {"ok": True}
    assert reloads == [URL] and len(calls) == 2   # exactly one retry

    # engine present but the URL is NOT an unloaded instance => no retry
    calls.clear()
    reloads.clear()

    class NoReloadCoex:
        def reload_for_url(self, url):
            return False
    monkeypatch.setattr("sable.serving.coexist.active",
                        lambda: NoReloadCoex())
    with pytest.raises(AgentBackendError):
        client._chat([{"role": "user", "content": "hi"}])
    assert reloads == [] and len(calls) == 1


def test_agent_runner_hook_is_inert_without_engine(home, monkeypatch):
    from sable.core.agent_runner import AgentBackendError, SableAgentClient

    monkeypatch.setattr("sable.serving.coexist.active", lambda: None)
    client = SableAgentClient.__new__(SableAgentClient)
    client.base_url = URL

    def inner(messages, tools=None):
        raise AgentBackendError("down")
    client._chat_inner = inner
    with pytest.raises(AgentBackendError):
        client._chat([])   # raises exactly as pre-WP5


# ------------------------------------------- explicit endpoints + CLI round

def test_dashboard_unload_load_status_roundtrip(home, monkeypatch):
    make_instance(home)
    write_live_state(home)
    monkeypatch.setattr("sable.serving.runtime.pid_alive", lambda pid: True)
    patch_stop(monkeypatch, [])
    spawns = []
    patch_spawn(monkeypatch, spawns)
    app = DashboardApp(SableConfig(), home=home)

    row = app.handle("GET", "/api/servers/alpha/status")
    assert row.body["load_state"] == "loaded"

    res = app.handle("POST", "/api/servers/alpha/unload", body={})
    assert res.body["unloaded"] is True
    assert res.body["state"] == "unloaded"
    row = app.handle("GET", "/api/servers/alpha/status")
    assert row.body["load_state"] == "unloaded"
    assert row.body["running"] is False

    res = app.handle("POST", "/api/servers/alpha/load", body={})
    assert res.status == 202 and res.body["state"] == "loaded"
    assert ("spawn", "alpha") in spawns
    row = app.handle("GET", "/api/servers/alpha/status")
    assert row.body["load_state"] == "loaded"

    # loading an already-loaded instance is idempotent (200, no new spawn)
    res = app.handle("POST", "/api/servers/alpha/load", body={})
    assert res.status == 200
    assert spawns.count(("spawn", "alpha")) == 1


def test_dashboard_unload_inflight_is_retryable_409(home, monkeypatch):
    make_instance(home)
    write_live_state(home)
    stops = []
    patch_stop(monkeypatch, stops)
    app = DashboardApp(SableConfig(), home=home)
    wd.note_request_start(URL)
    # handle() converts HttpError into the JSON error envelope (status 409)
    res = app.handle("POST", "/api/servers/alpha/unload", body={})
    assert res.status == 409
    assert res.body["error"]["code"] == "request_in_flight"
    assert stops == []


def test_cli_instances_actions_round_trip(home, monkeypatch, capsys):
    from sable.cli import cmd_instances_load, cmd_instances_status, \
        cmd_instances_unload

    make_instance(home)
    write_live_state(home)
    monkeypatch.setattr("sable.serving.runtime.pid_alive", lambda pid: True)
    patch_stop(monkeypatch, [])
    spawns = []
    patch_spawn(monkeypatch, spawns)
    monkeypatch.setattr("sable.cli.load_config",
                        lambda *a, **k: SableConfig())

    assert cmd_instances_status(_Args("status")) == 0
    out = capsys.readouterr().out
    assert "alpha" in out and "loaded" in out

    assert cmd_instances_unload(_Args("unload", "alpha")) == 0
    assert "unloaded" in capsys.readouterr().out
    assert cmd_instances_status(_Args("status")) == 0
    assert "unloaded" in capsys.readouterr().out

    assert cmd_instances_load(_Args("load", "alpha")) == 0
    assert ("spawn", "alpha") in spawns
    assert cmd_instances_unload(_Args("unload", "nope")) == 1   # unknown id


class _Args:
    def __init__(self, command, instance_id=None):
        self.command = command
        self.instances_command = command
        self.instance_id = instance_id
        self.config = None
        self.verbose = False


# --------------------------------------------------- coexistence handshake

def scripted_free(seq):
    """Fake gpu_free_mib: return seq values in order, repeat the last."""
    state = {"i": 0}

    def probe():
        i = min(state["i"], len(seq) - 1)
        state["i"] += 1
        return seq[i]
    return probe


def rescue_engine(home, monkeypatch, *, policy, free_seq, est=9000,
                  targets=None):
    cc = cx.CoexistConfig(idle_unload_seconds=60.0, min_free_vram_mib=10_000,
                          coexist_policy=policy,
                          vram_poll_interval_seconds=0.0)
    events: list = []
    coex = cx.Coexistence(cc, home=home, clock=FakeClock(),
                          sleep=lambda s: None,
                          free_vram=scripted_free(free_seq),
                          emit=lambda k, p: events.append(p))
    if targets is not None:
        monkeypatch.setattr(coex, "running_targets", lambda: list(targets))
    monkeypatch.setattr("sable.serving.instances.estimate_model_vram_mib",
                        lambda gguf, ctx=None: est)
    return coex, events


def test_rescue_refuse_emits_event_with_headroom(home, monkeypatch):
    targets = [{"id": "alpha", "pid": 1, "base_url": URL,
                "gguf_path": "/m.gguf", "context_length": None}]
    # est 20_000: free 500 < required 10_000 BUT projected 20_500 >= it =>
    # the "would pass if unloaded" branch under policy=refuse.
    coex, events = rescue_engine(home, monkeypatch, policy="refuse",
                                 free_seq=[500], targets=targets, est=20_000)
    unloaded = []
    monkeypatch.setattr(coex, "unload_all",
                        lambda **kw: unloaded.append(1) or [])

    assert coex.rescue_gate("job1", 10_000) is False
    assert unloaded == []                       # refuse => nothing stopped
    refused = [e for e in events if e.get("event") == "coexist_refused"]
    assert len(refused) == 1
    ev = refused[0]
    assert ev["free_mib"] == 500 and ev["required_mib"] == 10_000
    assert ev["projected_free_mib"] == 20_500   # 500 + one unloaded instance
    assert ev["would_pass_if_unloaded"] is True


def test_rescue_unload_triggers_waits_and_admits(home, monkeypatch):
    targets = [{"id": "alpha", "pid": 1, "base_url": URL,
                "gguf_path": "/m.gguf", "context_length": None}]
    # probe: still busy right after unload, freed on the next poll; the
    # projection (500 + 20_000 est) clears the 10_000 bar.
    coex, events = rescue_engine(home, monkeypatch, policy="unload",
                                 free_seq=[500, 500, 12_000],
                                 targets=targets, est=20_000)
    calls = []

    def fake_unload_all(**kw):
        calls.append(kw.get("reason"))
        return ["alpha"]
    monkeypatch.setattr(coex, "unload_all", fake_unload_all)

    assert coex.rescue_gate("job1", 10_000) is True
    assert calls == ["coexist"]
    trig = [e for e in events if e.get("event") == "coexist_unload_triggered"]
    assert len(trig) == 1 and trig[0]["targets"] == ["alpha"]
    assert trig[0]["free_mib"] == 500 and trig[0]["required_mib"] == 10_000


def test_rescue_still_insufficient_refuses_without_unload(home, monkeypatch):
    targets = [{"id": "alpha", "pid": 1, "base_url": URL,
                "gguf_path": "/m.gguf", "context_length": None}]
    coex, events = rescue_engine(home, monkeypatch, policy="unload",
                                 free_seq=[500], targets=targets, est=1000)
    calls = []
    monkeypatch.setattr(coex, "unload_all",
                        lambda **kw: calls.append(1) or [])
    # 500 + 1000 projected < 10_000 required => refuse even under unload
    assert coex.rescue_gate("job1", 10_000) is False
    assert calls == []
    assert events == []


def test_rescue_disabled_never_touches_anything(home, monkeypatch):
    coex, _ = rescue_engine(home, monkeypatch, policy="unload",
                            free_seq=[500])
    coex.cfg = cx.CoexistConfig(coexist_policy="unload")  # handshake off
    calls = []
    monkeypatch.setattr(coex, "unload_all", lambda **kw: calls.append(1) or [])
    assert coex.rescue_gate("job1", 10_000) is False
    assert calls == []


# --------------------------------- scheduler gate matrix (end-to-end-ish)

@pytest.fixture()
def registry(home):
    reg = ModelRegistry(Path(home) / "buffer" / "sable.db")
    yield reg
    reg.close()


class RecordingRescue:
    def __init__(self, admit_after):
        self.calls: list = []
        self.admit_after = admit_after   # return value of rescue_gate

    def rescue_gate(self, job_id, required_mib):
        self.calls.append((job_id, required_mib))
        return self.admit_after


def _sched(registry, monkeypatch, *, coexist, free_seq):
    sched = Scheduler(registry=registry,
                      cfg=TrainingConfig(min_free_vram_mib=10_000),
                      coexist=coexist)
    monkeypatch.setattr("sable.training.scheduler.gpu_free_mib",
                        scripted_free(free_seq))
    spawned: list = []
    sched._spawn_worker = lambda job, slot: spawned.append(job.job_id)
    return sched, spawned


def test_gate_matrix_refuse_exactly_as_today(registry, monkeypatch):
    # min_free set but NO coexist wired (default) -> plain refusal, unchanged
    sched, spawned = _sched(registry, monkeypatch, coexist=None,
                            free_seq=[500])
    job = sched.submit("sft", log_dir=None)
    with pytest.raises(SchedulerError) as ei:
        sched.run_job(job)
    assert "resource gate blocked start" in str(ei.value)
    assert spawned == []


def test_gate_matrix_refuse_policy_blocks_but_records(registry, monkeypatch):
    rescue = RecordingRescue(admit_after=False)   # policy=refuse behaviour
    sched, spawned = _sched(registry, monkeypatch, coexist=rescue,
                            free_seq=[500])
    job = sched.submit("sft", log_dir=None)
    with pytest.raises(SchedulerError):
        sched.run_job(job)
    assert rescue.calls == [(job.job_id, 10_000)]  # consulted exactly once
    assert spawned == []


def test_gate_matrix_unload_policy_admits_after_rescue(registry, monkeypatch):
    rescue = RecordingRescue(admit_after=True)
    # first probe fails the gate; after the (faked) unload the re-check passes
    sched, spawned = _sched(registry, monkeypatch, coexist=rescue,
                            free_seq=[500, 12_000])
    job = sched.submit("sft", log_dir=None)
    out = sched.run_job(job)
    assert out.state == "running" or spawned == [job.job_id]
    assert spawned == [job.job_id]
    assert rescue.calls == [(job.job_id, 10_000)]


def test_gate_passes_untouched_when_vram_sufficient(registry, monkeypatch):
    rescue = RecordingRescue(admit_after=True)
    sched, spawned = _sched(registry, monkeypatch, coexist=rescue,
                            free_seq=[50_000])
    job = sched.submit("sft", log_dir=None)
    sched.run_job(job)
    assert spawned == [job.job_id]
    assert rescue.calls == []          # gate passed => rescue never consulted


# ------------------------------------------------- race-safety (documented)

def test_load_serializes_behind_unload_no_double_launch(home, monkeypatch):
    """Documented race policy: unload/load take a per-instance lock. A load
    that arrives mid-unload blocks and then reloads exactly once — no
    concurrent double-launch."""
    make_instance(home)
    write_unloaded_state(home)
    spawns = []
    spawn_gate = threading.Event()

    def slow_spawn(cfg, inst, **kw):
        spawns.append(str(inst.get("id")))
        spawn_gate.wait(timeout=5)
        return FakeHandle()
    monkeypatch.setattr("sable.serving.supervisor.start_instance_server",
                        slow_spawn)

    coex, _ = build_engine(home, idle=60.0, lazy=True)
    results: list = []

    def loader():
        results.append(coex.load_instance("alpha", reason="lazy"))

    t1 = threading.Thread(target=loader)
    t1.start()
    # first loader is inside load_instance (spawn blocked); fire a second
    t2 = threading.Thread(target=loader)
    t2.start()
    time.sleep(0.3)
    assert len(spawns) == 1            # the second caller is still queued
    spawn_gate.set()
    t1.join(timeout=5)
    t2.join(timeout=5)
    assert not t1.is_alive() and not t2.is_alive()
    # exactly ONE launch: the queued caller found the live record and no-op'd
    assert spawns == ["alpha"]
    assert len(results) == 2


def test_request_racing_unload_gets_retryable_or_reloads(home, monkeypatch):
    """lazy off => the racing request surfaces the clear unloaded error;
    lazy on => it transparently reloads. Both are the documented choices."""
    make_instance(home)
    write_unloaded_state(home)
    patch_spawn(monkeypatch, [])

    app_off = DashboardApp(SableConfig(), home=home)
    with pytest.raises(HttpError) as ei:
        app_off._resolve_chat_target("alpha")
    assert ei.value.code == "instance_unloaded"

    app_on = DashboardApp(SableConfig(serve={"lazy_load": True}), home=home)
    base, _k, _m = app_on._resolve_chat_target("alpha")
    assert base == URL


def test_default_off_proof_engine_inert(home):
    """With no config changes: engine constructed but every automatic path is
    off — idle loop refuses to start, rescue declines, guidance is None."""
    coex = cx.build(SableConfig(), home)
    assert coex is None                       # not even constructed

    lazy_off = cx.Coexistence(cx.CoexistConfig(), home=home)
    assert lazy_off.start_idle_loop() is False
    assert lazy_off.rescue_gate("j", 10) is False
    assert lazy_off.ensure_loaded("alpha") is None
    assert lazy_off.unloaded_guidance("alpha") is None
    assert lazy_off.reload_for_url(URL) is False

"""WP4 engine watchdog tests — hermetic: injected clocks/sleepers, fabricated
/proc samples, monkeypatched ownership verdicts. No GPU, no real llama-server,
no sleeping 90 s (the fake clock drives the whole window)."""

from __future__ import annotations

import pytest

from sable.core.config import SableConfig
from sable.serving import watchdog as wd


# ------------------------------------------------------------------ helpers

class FakeClock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def sample(cpu_s: float, io_bytes: int, *, source: str = "read_write",
           t: float = 0.0) -> wd.ProcSample:
    return wd.ProcSample(t=t, cpu_s=cpu_s, io_bytes=io_bytes,
                         io_source=source)


def make_target(iid: str = "wdtest", pid: int = 424242,
                url: str = "http://127.0.0.1:9/v1") -> wd._WatchTarget:
    return wd._WatchTarget(instance_id=iid, pid=pid, base_url=url)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """Watchdog harness: fake clock/sleep, scripted proc samples, event log,
    restart recorder. Ownership verdict controlled per test via the returned
    namespace's `.kill_verdict` (read at call time)."""

    class Harness:
        def __init__(self):
            self.clock = FakeClock()
            self.sleeps: list = []
            self.events: list = []
            self.restarts: list = []
            self.kill_calls: list = []
            self.kill_verdict = "killed"
            self.samples: list = []  # ProcSample | None, consumed per step

        def read_proc(self, pid):
            if not self.samples:
                raise AssertionError("read_proc called with no scripted sample")
            return self.samples.pop(0)

        def emit(self, kind, payload):
            assert kind == "serve.watchdog"
            self.events.append(payload)

        def restart_fn(self, iid):
            self.restarts.append(iid)
            return True

        def terminate_owned_pid(self, pid, grace=8.0):
            self.kill_calls.append((pid, grace))
            return self.kill_verdict

    h = Harness()
    monkeypatch.setattr("sable.serving.runtime.terminate_owned_pid",
                        h.terminate_owned_pid)
    wd.reset_inflight()
    yield h
    wd.reset_inflight()


def build(envh, *, window=90.0, interval=5.0, max_restarts=3, targets=None):
    cfg = wd.WatchdogConfig(
        enabled=True, window_seconds=window,
        sample_interval_seconds=interval,
        kill_grace_seconds=7.0, max_restarts=max_restarts)
    return wd.EngineWatchdog(
        cfg, home=None, emit=envh.emit, clock=envh.clock,
        sleep=envh.sleeps.append, read_proc=envh.read_proc,
        restart_fn=envh.restart_fn,
        list_targets=(lambda: targets if targets is not None else []))


def event_names(h) -> list:
    return [e["event"] for e in h.events]


# ------------------------------------------------------- pure verdict rules

def test_verdict_silence_with_inflight_is_stalled():
    cfg = wd.WatchdogConfig()
    prev = sample(10.0, 100)
    cur = sample(10.0 + 0.005, 100)  # 5 ms cpu (< 10 ms), io unchanged
    assert wd.freeze_verdict(prev, cur, cfg, inflight_age=95) == wd.V_STALLED


def test_verdict_healthy_but_slow_busy_never_stalls():
    """CPU delta above threshold (model genuinely computing) => BUSY."""
    cfg = wd.WatchdogConfig()
    prev = sample(10.0, 100)
    cur = sample(14.0, 100)  # 4 s cpu in one sample interval
    assert wd.freeze_verdict(prev, cur, cfg, inflight_age=95) == wd.V_BUSY


def test_verdict_io_activity_alone_counts_as_progress():
    cfg = wd.WatchdogConfig()
    prev = sample(10.0, 100)
    cur = sample(10.0, 100 + cfg.io_threshold_bytes + 1)
    assert wd.freeze_verdict(prev, cur, cfg, inflight_age=95) == wd.V_BUSY


def test_verdict_idle_never_triggers():
    """No request in flight: silence is idleness, not a freeze."""
    cfg = wd.WatchdogConfig()
    prev = sample(10.0, 100)
    cur = sample(10.0, 100)
    assert wd.freeze_verdict(prev, cur, cfg, inflight_age=0) == wd.V_IDLE


def test_verdict_missing_evidence_never_stalls():
    cfg = wd.WatchdogConfig()
    prev_none = sample(10.0, 0, source="none")
    cur = sample(10.0, 0, source="read_write")
    assert wd.freeze_verdict(prev_none, cur, cfg, 95) == wd.V_UNKNOWN
    assert wd.freeze_verdict(None, cur, cfg, 95) == wd.V_UNKNOWN


# ------------------------------------------------------- in-flight tracking

def test_track_request_counts_and_age():
    clock = FakeClock(t=50.0)
    assert wd.inflight_count("http://x/v1") == 0
    assert wd.inflight_age("http://x/v1", clock=clock) == 0.0
    # age tracking with the injected clock
    wd.note_request_start("http://x/v1", clock=clock)
    clock.advance(12.0)
    assert wd.inflight_count("http://x/v1") == 1
    assert wd.inflight_age("http://x/v1", clock=clock) == pytest.approx(12.0)
    wd.note_request_end("http://x/v1")
    assert wd.inflight_count("http://x/v1") == 0
    # context manager counts (and normalizes trailing slash), releasing even
    # when the body raises
    with wd.track_request("http://x/v1/"):
        assert wd.inflight_count("http://x/v1") == 1
    with pytest.raises(RuntimeError):
        with wd.track_request("http://x/v1"):
            raise RuntimeError("boom")
    assert wd.inflight_count("http://x/v1") == 0


# ------------------------------------------------- full state machine tests

def test_freeze_detected_kill_verified_then_restart_backoff(env):
    """Silence + zero CPU delta + zero IO delta + request in flight for the
    full window -> freeze_suspect -> freeze_detected -> kill ONLY after the
    ownership helper verifies -> restart scheduled with backoff."""
    clock = env.clock
    url = "http://127.0.0.1:9/v1"
    wd.note_request_start(url, clock=clock)  # request in flight from t=1000

    t = make_target(url=url)
    w = build(env, window=90.0, interval=5.0, targets=[t])

    # step 1: first sample (no previous) -> unknown, arms prev
    env.samples = [sample(10.0, 500)]
    w.step()
    assert event_names(env) == []

    # steps over the window: stalled samples (cpu delta 2 ms < 10 ms, io flat).
    # suspect arms at t=1005; 17 more stalled samples reach t=1090 = 85 s —
    # still short of the 90 s window.
    for i in range(18):
        clock.advance(5.0)
        env.samples = [sample(10.0 + 0.002 * (i + 1), 500)]
        w.step()

    assert event_names(env) == ["freeze_suspect"]  # exactly one suspect event
    assert env.kill_calls == []                    # window NOT yet elapsed

    clock.advance(5.0)  # t = 1095 -> suspect_since 1005, 90 s elapsed
    env.samples = [sample(10.0 + 0.04, 500)]
    w.step()

    assert event_names(env) == ["freeze_suspect", "freeze_detected",
                                "instance_killed", "instance_restarted"]
    # kill went through the SHARED ownership-verified helper with our pid
    assert env.kill_calls == [(424242, 7.0)]
    assert env.restarts == ["wdtest"]
    assert env.sleeps == [2.0]  # first backoff = base * 2^0

    det = env.events[1]
    assert det["instance_id"] == "wdtest" and det["pid"] == 424242
    killed = env.events[2]
    assert killed["verified"].startswith("pid_alive")
    restarted = env.events[3]
    assert restarted["attempt"] == 1 and restarted["backoff_seconds"] == 2.0


def test_busy_compute_never_killed(env):
    """Same silence on the wire, but CPU delta > threshold every sample:
    healthy-but-slow must NEVER be killed, even with a request in flight."""
    clock = env.clock
    url = "http://127.0.0.1:9/v1"
    wd.note_request_start(url, clock=clock)
    t = make_target(url=url)
    w = build(env, window=30.0, interval=5.0, targets=[t])

    env.samples = [sample(1.0, 10)]
    w.step()
    for i in range(12):  # way past the 30 s window
        clock.advance(5.0)
        env.samples = [sample(1.0 + 0.5 * (i + 1), 10)]  # CPU burning
        w.step()
    assert env.kill_calls == []
    assert env.restarts == []
    assert env.events == []


def test_no_request_in_flight_never_triggers(env):
    """Zero CPU, zero IO — but nothing in flight: idle, not frozen."""
    clock = env.clock
    t = make_target()
    w = build(env, window=10.0, interval=5.0, targets=[t])

    env.samples = [sample(10.0, 77)]
    w.step()
    for _ in range(6):
        clock.advance(5.0)
        env.samples = [sample(10.0, 77)]
        w.step()
    assert env.kill_calls == []
    assert env.events == []


def test_request_completing_mid_window_clears_suspect(env):
    """Stall starts, then the server wakes up and burns CPU before the window
    elapses -> suspect cleared, nothing killed."""
    clock = env.clock
    url = "http://127.0.0.1:9/v1"
    wd.note_request_start(url, clock=clock)
    t = make_target(url=url)
    w = build(env, window=30.0, interval=5.0, targets=[t])

    env.samples = [sample(5.0, 10)]
    w.step()
    clock.advance(5.0)
    env.samples = [sample(5.0, 10)]  # stalled -> suspect
    w.step()
    assert event_names(env) == ["freeze_suspect"]

    clock.advance(5.0)
    env.samples = [sample(9.0, 10)]  # CPU back on: recovered
    w.step()
    for i in range(8):  # long past the window — CPU keeps burning every sample
        clock.advance(5.0)
        env.samples = [sample(9.0 + 0.4 * (i + 2), 10)]
        w.step()
    assert env.kill_calls == []
    assert "freeze_detected" not in event_names(env)


def test_foreign_pid_never_killed_stale_state_cleared(env, tmp_path):
    """Ownership check fails (foreign/recycled pid): NEVER kill — log and
    clear the stale state file instead; no restart onto a pid we don't own."""
    from sable.serving import instances as srv_inst

    clock = env.clock
    url = "http://127.0.0.1:9/v1"
    wd.note_request_start(url, clock=clock)
    t = make_target(iid="wdstale", pid=654321, url=url)

    # stale state file present (would be cleared on the foreign verdict)
    srv_inst.write_state(tmp_path, "wdstale", {"mode": "spawn", "pid": 654321,
                                               "base_url": url})
    env.kill_verdict = "foreign"
    w = build(env, window=10.0, interval=5.0, targets=[t])
    # give the watchdog a home so state clearing has somewhere to look
    w.home = tmp_path

    env.samples = [sample(3.0, 42)]
    w.step()
    for i in range(3):
        clock.advance(5.0)
        env.samples = [sample(3.0, 42)]
        w.step()

    assert event_names(env) == ["freeze_suspect", "freeze_detected",
                                "stale_state_cleared"]
    # the helper was consulted (that IS the ownership verification) and its
    # foreign verdict meant no kill signal was ever sent by us
    assert env.kill_calls == [(654321, 7.0)]
    assert env.restarts == []                       # never restart onto it
    assert srv_inst.read_state(tmp_path, "wdstale") is None  # state cleared


def test_backoff_sequence_and_restart_cap(env):
    """Backoff doubles per attempt; at max_restarts within the window a loud
    restart_capped event fires and further freezes do nothing (no loop)."""
    clock = env.clock
    url = "http://127.0.0.1:9/v1"
    wd.note_request_start(url, clock=clock)

    def fresh_target():
        return [make_target(iid="wdcap", pid=11111, url=url)]

    w = build(env, window=10.0, interval=5.0, max_restarts=3,
              targets=fresh_target())

    def one_freeze_cycle():
        env.samples = [sample(2.0, 8)]
        w.step()
        stalled_first = False
        for _ in range(4):
            clock.advance(5.0)
            env.samples = [sample(2.0, 8)]
            w.step()
            stalled_first = True
        assert stalled_first

    for attempt in range(1, 4):
        one_freeze_cycle()
        assert len(env.restarts) == attempt
    assert env.sleeps == [2.0, 4.0, 8.0]  # base * 2^(attempt-1)

    # 4th freeze inside the window: capped — loud event, no kill-loop
    one_freeze_cycle()
    names = event_names(env)
    assert names.count("restart_capped") == 1
    assert len(env.restarts) == 3          # no 4th restart
    assert env.sleeps == [2.0, 4.0, 8.0]   # no extra backoff sleep
    cap = next(e for e in env.events if e["event"] == "restart_capped")
    assert cap["instance_id"] == "wdcap" and cap["restarts"] == 3

    # and it stays capped: another full cycle changes nothing
    one_freeze_cycle()
    assert len(env.restarts) == 3


def test_restart_failure_surfaced_not_swallowed(env):
    clock = env.clock
    url = "http://127.0.0.1:9/v1"
    wd.note_request_start(url, clock=clock)
    t = make_target(iid="wdfail", pid=22222, url=url)

    def boom(iid):
        raise RuntimeError("spawn exploded")

    w = build(env, window=10.0, interval=5.0, targets=[t])
    w._restart_fn = boom

    env.samples = [sample(1.0, 1)]
    w.step()
    for _ in range(3):
        clock.advance(5.0)
        env.samples = [sample(1.0, 1)]
        w.step()
    names = event_names(env)
    assert names == ["freeze_suspect", "freeze_detected",
                     "instance_killed", "restart_failed"]
    fail = env.events[-1]
    assert "spawn exploded" in fail["error"] and fail["attempt"] == 1


# -------------------------------------------------------------- config gate

def test_config_defaults_off_and_start_watchdog_noop():
    """With NO config changes the watchdog does nothing at all: defaults are
    off and start_watchdog returns None without spawning a thread."""
    cfg = SableConfig()
    wc = wd.watchdog_config(cfg)
    assert wc.enabled is False
    assert wc.window_seconds == 90.0
    assert wd.start_watchdog(cfg, home=None) is None

    # explicit false also off
    cfg2 = SableConfig(serve={"watchdog_enabled": False})
    assert wd.watchdog_config(cfg2).enabled is False
    assert wd.start_watchdog(cfg2, home=None) is None


def test_engine_start_refuses_when_disabled():
    w = wd.EngineWatchdog(wd.WatchdogConfig(enabled=False))
    assert w.start() is False


def test_opt_in_enables_but_proc_rules_apply(monkeypatch):
    cfg = SableConfig(serve={"watchdog_enabled": "true",
                               "watchdog_window_seconds": 12})
    wc = wd.watchdog_config(cfg)
    assert wc.enabled is True and wc.window_seconds == 12.0

    # unavailable /proc -> start() logs once and declines (never kills)
    monkeypatch.setattr(wd, "proc_sampling_available", lambda: False)
    w = wd.EngineWatchdog(wc)
    assert w.start() is False


# ------------------------------------------------------------- /proc reader

class _Inline:
    """File-object stand-in whose read_text returns fixed content."""

    def __init__(self, text: str):
        self._t = text

    def read_text(self, errors="ignore"):
        return self._t


def _fake_proc(files: dict):
    """Path replacement for the watchdog module: known /proc keys resolve to
    inline content, anything else raises (unreachable proc)."""

    def fake_path(p):
        key = str(p)
        if key in files:
            return _Inline(files[key])
        raise OSError(f"no such file (test): {key}")

    return fake_path


def test_read_proc_sample_parses_stat_and_io(monkeypatch):
    files = {
        "/proc/7/stat": "7 (llama-server) S 1 7 0 0 -1 4194304 100 0 0 0 "
                        "1234 567 0 0 20 0 40 0 99999 0 0",
        "/proc/7/io": ("rchar: 100\nwchar: 50\nread_bytes: 300\n"
                       "write_bytes: 200\ncancelled_write_bytes: 0\n"),
    }
    monkeypatch.setattr(wd, "Path", _fake_proc(files))

    s = wd.read_proc_sample(7, clock=lambda: 5.0, hz=100)
    assert s is not None
    assert s.cpu_s == pytest.approx((1234 + 567) / 100.0)
    assert s.io_bytes == 500 and s.io_source == "read_write"


def test_read_proc_sample_io_fallback_to_rchar(monkeypatch):
    files = {
        "/proc/8/stat": "8 (srv) S 1 8 0 0 -1 0 0 0 0 0 10 20 0 0 20 0 1 0 0 0 0",
        "/proc/8/io": "rchar: 40\nwchar: 5\ncancelled_write_bytes: 0\n",
    }
    monkeypatch.setattr(wd, "Path", _fake_proc(files))
    s = wd.read_proc_sample(8, clock=lambda: 1.0, hz=100)
    assert s is not None
    assert s.cpu_s == pytest.approx(0.3)
    assert s.io_bytes == 45 and s.io_source == "rchar_wchar"


def test_read_proc_sample_unreadable_io_is_none_source(monkeypatch):
    files = {
        "/proc/9/stat": "9 (srv) S 1 9 0 0 -1 0 0 0 0 0 5 5 0 0 20 0 1 0 0 0 0",
    }
    monkeypatch.setattr(wd, "Path", _fake_proc(files))
    s = wd.read_proc_sample(9, clock=lambda: 0.0, hz=100)
    assert s is not None
    assert s.io_source == "none"
    # and the verdict refuses to kill on missing io evidence
    cfg = wd.WatchdogConfig()
    assert wd.freeze_verdict(s, s, cfg, inflight_age=999) == wd.V_UNKNOWN


def test_read_proc_sample_stat_comm_with_parens(monkeypatch):
    """comm may contain spaces AND parens — parsing must split on the LAST ')'."""
    stat = "42 (weird (na)me srv) S 1 42 0 0 -1 0 0 0 0 0 77 33 0 0 20 0 1 0 0 0 0"
    monkeypatch.setattr(wd, "Path", _fake_proc({"/proc/42/stat": stat}))
    s = wd.read_proc_sample(42, clock=lambda: 0.0, hz=100)
    assert s is not None
    assert s.cpu_s == pytest.approx((77 + 33) / 100.0)


def test_read_proc_sample_gone_pid():
    # pid that cannot exist -> None (never raises)
    assert wd.read_proc_sample(2_147_483_647, clock=lambda: 0.0) is None

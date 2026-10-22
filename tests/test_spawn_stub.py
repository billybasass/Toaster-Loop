"""Spawn-mode supervisor tests using a stub llama-server (no GGUF needed).

The stub parses the same flags sable passes (`-m`, `--alias`, `--host`,
`--port`, `-c`) and serves /v1/models, letting us verify: command wiring,
health-check loop, dynamic port, clean SIGTERM shutdown, and clear errors on
early child exit. Plus the review #9/#11 regressions: no parent-side log-fd
leak per spawn, legacy `ServingSupervisor.start` children visible to
`stop_all_servers()`, and the crash-loop budget on the legacy path.
"""

from __future__ import annotations

import os
import stat
import time

import pytest

from sable.core.config import SableConfig
from sable.serving.supervisor import (
    ServeError, ServingSupervisor, _start_instance_server,
    clear_crash_loop, crash_loop_status, owned_instance_ids,
    record_start_failure, stop_all_servers,
)

STUB = '''#!/usr/bin/env python3
import argparse, json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

p = argparse.ArgumentParser()
p.add_argument("-m", dest="model", required=True)
p.add_argument("--alias", default=None)
p.add_argument("--host", default="127.0.0.1")
p.add_argument("--port", type=int, required=True)
p.add_argument("-c", dest="ctx", type=int, default=None)
p.add_argument("-ngl", dest="ngl", type=int, default=None)  # GPU offload (runtime.py)
a = p.parse_args()

if a.ctx is not None and a.ctx < 65536:
    print("stub: context too small", file=sys.stderr); sys.exit(3)

class H(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002 - silence access log
        pass
    def do_GET(self):
        if self.path.rstrip("/") == "/v1/models":
            body = json.dumps({"object": "list", "data": [
                {"id": a.alias or a.model, "object": "model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404); self.end_headers()

srv = ThreadingHTTPServer((a.host, a.port), H)
print("stub llama-server ready", flush=True)
srv.serve_forever()
'''


@pytest.fixture()
def stub_binary(tmp_path):
    p = tmp_path / "llama-server"
    p.write_text(STUB)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(p)


@pytest.fixture()
def fake_gguf(tmp_path):
    g = tmp_path / "test-model-q4_k_m.gguf"
    g.write_bytes(b"GGUF")
    return str(g)


def test_spawn_health_stop(tmp_path, stub_binary, fake_gguf):
    cfg = SableConfig(
        gguf_path=fake_gguf,
        llama_server_binary=stub_binary,
        port=0,
        context_length=65536,
    )
    sup = ServingSupervisor(cfg)
    handle = sup.start(ready_timeout=20)
    try:
        assert handle.owned is True
        assert handle.process is not None and handle.process.poll() is None
        assert handle.base_url.startswith("http://127.0.0.1:")
        assert handle.model_name == "test-model-q4_k_m"  # derived from gguf stem

        import httpx

        r = httpx.get(handle.base_url + "/models", timeout=5)
        assert r.status_code == 200
        assert r.json()["data"][0]["id"] == "test-model-q4_k_m"
    finally:
        sup.stop()
    assert handle.process.poll() is not None  # child terminated on stop


def test_spawn_early_exit_clear_error(tmp_path, stub_binary):
    """Child dying during startup => ServeError with exit code, no hang."""
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"GGUF")
    cfg = SableConfig(
        gguf_path=str(gguf),
        llama_server_binary=stub_binary,
        context_length=4096,  # stub exits with code 3
    )
    t0 = time.monotonic()
    with pytest.raises(ServeError) as ei:
        ServingSupervisor(cfg).start(ready_timeout=20)
    assert time.monotonic() - t0 < 15  # failed fast, didn't wait full timeout
    assert "exited early" in str(ei.value)


# ------------------------------------------------------- fd hygiene (review #9)

def _open_fd_count() -> int:
    return len(os.listdir("/proc/self/fd"))


def test_spawn_does_not_leak_log_fds(tmp_path, stub_binary):
    """Every spawn opens the log file for the child; the PARENT copy must be
    closed right after Popen (#9). Spawns that fail while tracebacks are still
    retained (the dashboard keeps recent errors) must not pin one fd each."""
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"GGUF")
    logp = tmp_path / "logs" / "srv.log"
    cfg = SableConfig(
        llama_server_binary=stub_binary,
        context_length=4096,  # stub exits immediately => failure path per spawn
        llama_auto_setup=False,
    )
    inst = {"id": "fd-probe", "gguf": str(gguf), "port": 0}

    base = _open_fd_count()
    retained: list = []
    for _ in range(12):
        try:
            _start_instance_server(cfg, inst, log_path=logp, ready_timeout=5)
        except ServeError as e:
            retained.append(e)  # keep frames alive: an unclosed fd would stay open
    assert _open_fd_count() <= base + 1, (
        f"log fds leaked across spawns: {base} -> {_open_fd_count()}")


def test_healthy_spawn_stop_does_not_leak_fds(tmp_path, stub_binary, fake_gguf):
    """Success path too: start/stop cycles keep the parent fd count flat."""
    cfg = SableConfig(
        gguf_path=fake_gguf, llama_server_binary=stub_binary,
        port=0, context_length=65536, llama_auto_setup=False,
    )
    logp = tmp_path / "logs" / "srv.log"
    inst = {"id": "fd-probe-ok", "gguf": fake_gguf, "port": 0}

    base = _open_fd_count()
    for _ in range(4):
        handle = _start_instance_server(cfg, inst, log_path=logp, ready_timeout=20)
        assert handle.process is not None and handle.process.poll() is None
        handle.stop(grace=5)
    assert _open_fd_count() <= base + 1, (
        f"fds leaked across healthy spawn/stop cycles: {base} -> {_open_fd_count()}")


# ------------------------------------- consolidation (review #11): one registry

def test_legacy_start_is_visible_to_stop_all_servers(tmp_path, stub_binary,
                                                     fake_gguf):
    """A legacy ServingSupervisor.start() child lands in the shared ownership
    registry: owned_instance_ids() sees it and stop_all_servers() terminates
    it (previously these two stacks were disjoint and the child orphaned)."""
    cfg = SableConfig(
        gguf_path=fake_gguf, llama_server_binary=stub_binary,
        port=0, context_length=65536, llama_auto_setup=False,
    )
    sup = ServingSupervisor(cfg)
    handle = sup.start(ready_timeout=20)
    try:
        assert handle.process is not None
        assert "default" in owned_instance_ids()

        stop_all_servers()

        deadline = time.monotonic() + 8
        while time.monotonic() < deadline and handle.process.poll() is None:
            time.sleep(0.1)
        assert handle.process.poll() is not None, (
            "legacy-started child survived stop_all_servers()")
        assert "default" not in owned_instance_ids()
    finally:
        sup.stop()  # idempotent after the registry already stopped it


def test_supervisor_stop_clears_shared_registry(tmp_path, stub_binary,
                                                fake_gguf):
    """The reverse direction: supervisor.stop() must also pop the shared
    registry entry so stop_all_servers has nothing stale left."""
    cfg = SableConfig(
        gguf_path=fake_gguf, llama_server_binary=stub_binary,
        port=0, context_length=65536, llama_auto_setup=False,
    )
    sup = ServingSupervisor(cfg)
    handle = sup.start(ready_timeout=20)
    assert "default" in owned_instance_ids()
    sup.stop()
    assert handle.process is not None
    assert handle.process.poll() is not None
    assert "default" not in owned_instance_ids()


# ----------------------------------- crash-loop budget on the legacy path (#11)

def test_legacy_start_respects_crash_loop_budget(tmp_path, stub_binary):
    """A flapping GGUF (child exits during startup) records one failure per
    cycle; after CRASH_LOOP_MAX_FAILURES consecutive fast crashes further
    starts are refused and the state surfaces the crash loop."""
    from sable.serving import supervisor as sup_mod

    gguf = tmp_path / "flap.gguf"
    gguf.write_bytes(b"GGUF")
    cfg = SableConfig(
        gguf_path=str(gguf), llama_server_binary=stub_binary,
        context_length=4096,  # stub exits with code 3 => crash every cycle
        llama_auto_setup=False,
    )
    for _ in range(sup_mod.CRASH_LOOP_MAX_FAILURES):
        with pytest.raises(ServeError):
            ServingSupervisor(cfg).start(ready_timeout=10)

    st = crash_loop_status("default")
    assert st is not None and st["failures"] >= sup_mod.CRASH_LOOP_MAX_FAILURES
    assert st["refused"] is True

    # Budget exhausted: the next attempt refuses WITHOUT spawning a child.
    with pytest.raises(ServeError, match="crash loop"):
        ServingSupervisor(cfg).start(ready_timeout=10)

    # An explicit user start clears the refusal and retries normally (fails on
    # the flapping stub again, but with the real spawn error, not the refusal).
    clear_crash_loop("default")
    with pytest.raises(ServeError) as ei:
        ServingSupervisor(cfg).start(ready_timeout=10)
    assert "crash loop" not in str(ei.value)
    assert "exited early" in str(ei.value)


# ------------------------- CLI deploy detached path: budget + state parity (#4)

def test_cli_deploy_records_crash_loop_failure(tmp_path, stub_binary):
    """start_server_detached (the `sable models deploy` path) records each
    failed spawn/healthcheck cycle under the "default" key so the dashboard's
    crash_loop_status is honest — but an explicit CLI invocation never refuses:
    it clears the budget up front (user-intent signal, like `sable serve`)
    and re-arms only on failure."""
    from sable.serving import supervisor as sup_mod
    from sable.serving.deploy import start_server_detached

    clear_crash_loop("default")
    gguf = tmp_path / "flap-cli.gguf"
    gguf.write_bytes(b"GGUF")
    cfg = SableConfig(
        gguf_path=str(gguf), llama_server_binary=stub_binary, port=0,
        context_length=4096,  # stub exits with code 3 => crash every cycle
        llama_auto_setup=False,
    )
    home = tmp_path / "uf-cli"
    with pytest.raises(ServeError):
        start_server_detached(cfg, str(gguf), home=home)
    st = crash_loop_status("default")
    assert st is not None and st["failures"] == 1

    # A prior crash loop never blocks the next explicit CLI deploy: the call
    # proceeds to a real spawn attempt (cleared up front, re-recorded on fail).
    for _ in range(sup_mod.CRASH_LOOP_MAX_FAILURES):
        with pytest.raises(ServeError) as ei:
            start_server_detached(cfg, str(gguf), home=home)
        assert "crash loop" not in str(ei.value)  # never a refusal — real error
        assert "exited early" in str(ei.value)
    # Each explicit attempt clears then re-records exactly one failure.
    st = crash_loop_status("default")
    assert st is not None and st["failures"] == 1
    clear_crash_loop("default")


def test_cli_deploy_healthy_clears_budget_and_writes_parity_state(
        tmp_path, stub_binary):
    """A successful CLI deploy (a) clears any recorded crash-loop history —
    same as every explicit start path — and (b) writes serve_state.json with
    the SAME field shape dashboard _write_instance_state produces for the
    default spawn record, so router pool building and metrics
    sable_instances_up see the CLI-detached server correctly."""
    from sable.serving.deploy import read_serve_state, start_server_detached

    clear_crash_loop("default")
    gguf = tmp_path / "good-model-q4_k_m.gguf"
    gguf.write_bytes(b"GGUF")
    cfg = SableConfig(
        gguf_path=str(gguf), llama_server_binary=stub_binary, port=0,
        context_length=65536, llama_auto_setup=False,
    )
    home = tmp_path / "uf-cli-ok"
    record_start_failure("default")  # pre-existing crash history
    state = start_server_detached(cfg, str(gguf), model_id="v9", home=home)
    try:
        assert crash_loop_status("default") is None  # healthy start resets it

        written = read_serve_state(home)
        assert written is not None and written["pid"] == state["pid"]
        # Shape parity with dashboard _write_instance_state's spawn record.
        dashboard_spawn_keys = {
            "mode", "pid", "host", "port", "base_url", "model_name",
            "model_id", "gguf_path", "instance_id", "started_at"}
        assert set(written) == dashboard_spawn_keys
        assert written["mode"] == "spawn"
        assert written["instance_id"] == "default"

        # Readers tolerate it: instances.state_alive pid-probes the spawn
        # state exactly like a dashboard-written one.
        from sable.serving import instances as srv_inst
        assert srv_inst.state_alive(written) is True
    finally:
        # Detach-by-design means nothing auto-stops it; clean up here.
        import signal
        try:
            os.kill(state["pid"], signal.SIGTERM)
        except ProcessLookupError:
            pass
        clear_crash_loop("default")

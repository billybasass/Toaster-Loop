"""Feature-wave Phase 6 tests: multi-model serving (N named instances).

Covers:
  1. Coexistence — two spawned stub instances on different ports, independent
     health; stopping one leaves the other healthy.
  2. CRUD + validation — port conflict rejected, duplicate name rejected,
     non-loopback attach URL rejected, attach instances never spawn/kill.
  3. Backward compat — legacy single-instance config/endpoints map to the
     reserved "default" instance (rows mirror serve_state.json both ways).
  4. Profile binding — a profile's server_instance routes /api/chat/stream to
     that instance's endpoint; unset keeps the default target. Unknown => 404,
     stopped => 503.
  5. VRAM advisory — over-commit yields a warning field (spawn still starts);
     force bypasses the estimate entirely.
  6. Shutdown — stop_all_servers stops owned children only; a foreign process
     survives.

All hermetic: stub llama-server binaries (like test_spawn_stub.py /
test_indexing_phase5.py) + FakeOpenAIServer for attach targets; conftest
redirects SABLE_HOME.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sable.core.config import SableConfig  # noqa: E402
from sable.dashboard.app import DashboardApp  # noqa: E402
from sable.serving import instances as srv_inst  # noqa: E402
from sable.serving.supervisor import (  # noqa: E402
    ServeError, require_loopback, stop_all_servers,
)

from fake_server import FakeOpenAIServer  # noqa: E402


# ------------------------------------------------------------------ fixtures

STUB = '''#!/usr/bin/env python3
"""Stub llama-server: parses sable's flags, records argv, serves /v1/models."""
import argparse, json, os, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

p = argparse.ArgumentParser()
p.add_argument("-m", dest="model", required=True)
p.add_argument("--alias", default=None)
p.add_argument("--host", default=None)
p.add_argument("--port", type=int, required=True)
p.add_argument("-c", dest="ctx", type=int, default=None)
p.add_argument("-ngl", dest="ngl", type=int, default=None)
p.add_argument("--embedding", action="store_true")
a, _extra = p.parse_known_args()

if a.host != "127.0.0.1":
    print("stub: refusing non-loopback bind", file=sys.stderr); sys.exit(4)

with open(os.environ.get("STUB_PIDFILE", os.devnull), "w") as f:
    f.write(str(os.getpid()))
argvf = os.environ.get("STUB_ARGVFILE")
if argvf:
    with open(argvf, "w") as f:
        json.dump({"embedding": bool(a.embedding), "alias": a.alias,
                   "ctx": a.ctx}, f)

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
            self.end_headers(); self.wfile.write(body)
        else:
            self.send_response(404); self.end_headers()

srv = ThreadingHTTPServer((a.host, a.port), H)
print("stub llama-server ready", flush=True)
srv.serve_forever()
'''


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "uf-p6"
    monkeypatch.setenv("SABLE_HOME", str(h))
    return h


@pytest.fixture()
def stub_binary(tmp_path, monkeypatch):
    p = tmp_path / "llama-server"
    p.write_text(STUB)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("STUB_PIDFILE", str(tmp_path / "stub.pid"))
    return str(p)


@pytest.fixture()
def ggufs(tmp_path):
    a = tmp_path / "model-a-q4_k_m.gguf"
    a.write_bytes(b"GGUF")
    b = tmp_path / "model-b-q4_k_m.gguf"
    b.write_bytes(b"GGUF")
    return str(a), str(b)


def make_app(home, **cfg_kwargs) -> DashboardApp:
    return DashboardApp(SableConfig(llama_auto_setup=False, **cfg_kwargs),
                        home=home)


def create_server(app, name, gguf=None, attach_url=None, **extra):
    body = {"name": name}
    if gguf:
        body["gguf"] = gguf
    if attach_url:
        body["attach_url"] = attach_url
    body.update(extra)
    return app.handle("POST", "/api/servers", body=body)


def row_of(app, server_id):
    rows = app.handle("GET", "/api/servers").body["servers"]
    return next((r for r in rows if r["id"] == server_id), None)


def must_row(app, server_id):
    """row_of() that fails loudly when the row is missing."""
    row = row_of(app, server_id)
    assert row is not None, f"server row {server_id!r} missing"
    return row


@pytest.fixture()
def cleanup_servers():
    """Kill any instance children a test left behind (own registry only)."""
    yield
    stop_all_servers()


# ------------------------------------------------------- 1. coexistence

def test_two_instances_coexist_and_stop_independently(home, stub_binary,
                                                      ggufs, cleanup_servers):
    app = make_app(home, llama_server_binary=stub_binary)
    a_gguf, b_gguf = ggufs
    ra = create_server(app, "alpha", gguf=a_gguf)
    rb = create_server(app, "beta", gguf=b_gguf)
    assert ra.status == 201 and rb.status == 201
    ida, idb = ra.body["server"]["id"], rb.body["server"]["id"]

    sa = app.handle("POST", f"/api/servers/{ida}/start", body={})
    sb = app.handle("POST", f"/api/servers/{idb}/start", body={})
    assert sa.status == 202 and sb.status == 202, (sa.body, sb.body)
    url_a, url_b = sa.body["base_url"], sb.body["base_url"]
    assert url_a != url_b

    # both healthy, independently
    for url in (url_a, url_b):
        r = httpx.get(url + "/models", timeout=5)
        assert r.status_code == 200
    assert must_row(app, ida)["running"] is True
    assert must_row(app, idb)["running"] is True

    # stop one; the other stays healthy
    st = app.handle("POST", f"/api/servers/{ida}/stop", body={})
    assert st.status == 200 and st.body["stopped"] is True
    assert must_row(app, ida)["running"] is False
    assert must_row(app, idb)["running"] is True
    r = httpx.get(url_b + "/models", timeout=5)
    assert r.status_code == 200

    app.handle("POST", f"/api/servers/{idb}/stop", body={})
    assert must_row(app, idb)["running"] is False


def test_embedding_role_instance_spawns_with_flag(home, stub_binary, tmp_path,
                                                  monkeypatch,
                                                  cleanup_servers):
    """An embedding-role instance passes --embedding (role tag honored)."""
    argvf = tmp_path / "argv.json"
    monkeypatch.setenv("STUB_ARGVFILE", str(argvf))
    app = make_app(home, llama_server_binary=stub_binary)
    g = tmp_path / "embed.gguf"
    g.write_bytes(b"GGUF")
    r = create_server(app, "embed1", gguf=str(g), role="embedding")
    sid = r.body["server"]["id"]
    start = app.handle("POST", f"/api/servers/{sid}/start", body={})
    assert start.status == 202
    meta = json.loads(argvf.read_text())
    assert meta["embedding"] is True
    app.handle("POST", f"/api/servers/{sid}/stop", body={})


# ------------------------------------------------------ 2. CRUD + validation

def test_crud_and_validation(home, ggufs):
    app = make_app(home)
    a_gguf, _ = ggufs

    r = create_server(app, "solo", gguf=a_gguf, port=45_712)
    assert r.status == 201
    sid = r.body["server"]["id"]

    # duplicate name (case-insensitive) rejected
    dup = create_server(app, "SOLO", gguf=a_gguf)
    assert dup.status == 409 and dup.body["error"]["code"] == "duplicate_name"

    # port conflict with the existing instance rejected
    conflict = create_server(app, "other", gguf=a_gguf, port=45_712)
    assert conflict.status == 409 \
        and conflict.body["error"]["code"] == "port_conflict"

    # neither gguf nor attach => 400
    bad = create_server(app, "bad")
    assert bad.status == 400

    # both => 400 (ambiguous target)
    both = app.handle("POST", "/api/servers", body={
        "name": "both", "gguf": a_gguf,
        "attach_url": "http://127.0.0.1:9/v1"})
    assert both.status == 400

    # update: rename + change port away from conflict is fine
    upd = app.handle("PUT", f"/api/servers/{sid}",
                     body={"name": "solo2", "gguf": a_gguf, "port": 0})
    assert upd.status == 200 and upd.body["server"]["name"] == "solo2"

    # delete removes the definition file
    d = app.handle("DELETE", f"/api/servers/{sid}", body={})
    assert d.status == 200
    assert row_of(app, sid) is None


def test_attach_url_must_be_loopback(home):
    app = make_app(home)
    r = create_server(app, "remote", attach_url="http://10.0.0.5:8080/v1")
    assert r.status == 400 and "loopback" in r.body["error"]["message"]
    with pytest.raises(ServeError):
        require_loopback("http://example.com/v1")
    assert require_loopback("http://localhost:1234/v1")


def test_attach_instance_never_spawns_or_kills(home):
    app = make_app(home)
    with FakeOpenAIServer() as fs:
        r = create_server(app, "attached", attach_url=fs.base_url)
        assert r.status == 201
        sid = r.body["server"]["id"]

        start = app.handle("POST", f"/api/servers/{sid}/start", body={})
        assert start.status == 202
        st = srv_inst.read_state(home, sid)
        assert st is not None
        assert st["mode"] == "attach" and not st.get("pid")  # spawned nothing

        row = must_row(app, sid)
        assert row["running"] is True and row["mode"] == "attach"

        stop = app.handle("POST", f"/api/servers/{sid}/stop", body={})
        assert stop.status == 200 and stop.body["stopped"] is False
        # the foreign endpoint survives — detach only forgets it
        assert httpx.get(fs.base_url + "/models", timeout=5).status_code == 200


# ------------------------------------------------------- 3. backward compat

def test_default_instance_mirrors_legacy_config(home, ggufs):
    a_gguf, _ = ggufs
    app = make_app(home, gguf_path=a_gguf)
    rows = app.handle("GET", "/api/servers").body["servers"]
    defaults = [r for r in rows if r["id"] == "default"]
    assert len(defaults) == 1 and defaults[0]["default"] is True
    assert defaults[0]["gguf"] == a_gguf

    # default cannot be deleted
    d = app.handle("DELETE", "/api/servers/default", body={})
    assert d.status == 400


def test_legacy_attach_shows_in_default_row(home):
    from sable.serving.deploy import write_serve_state

    app = make_app(home)
    with FakeOpenAIServer() as fs:
        write_serve_state(home, {"mode": "attach", "base_url": fs.base_url,
                                 "model_name": "fake-model",
                                 "attach_url": fs.base_url.rsplit("/v1", 1)[0]})
        row = must_row(app, "default")
        assert row["running"] is True and row["mode"] == "attach"

        # the new stop endpoint stops/forgets the legacy default too
        r = app.handle("POST", "/api/servers/default/stop", body={})
        assert r.status == 200 and r.body["stopped"] is False
        assert srv_inst.read_state(home, "default") is None


def test_new_start_writes_legacy_serve_state(home, stub_binary, ggufs,
                                             cleanup_servers):
    """Starting the default instance via /api/servers keeps serve_state.json
    in sync so old panels/endpoints keep working."""
    from sable.serving.deploy import read_serve_state

    app = make_app(home, llama_server_binary=stub_binary)
    # give the reserved default a gguf through config (legacy key)
    app.cfg.gguf_path = ggufs[0]
    r = app.handle("POST", "/api/servers/default/start", body={})
    assert r.status == 202, r.body
    st = read_serve_state(home)
    assert st and st["base_url"] == r.body["base_url"]
    legacy_status = app.handle("GET", "/api/serve/status").body
    assert legacy_status["running"] is True

    app.handle("POST", "/api/servers/default/stop", body={})
    assert app.handle("GET", "/api/serve/status").body["running"] is False


# ------------------------------------------------------- 4. profile binding

def _stub_runner_capture(monkeypatch, calls):
    """create_runner stub recording (base_url, model) per construction."""
    from sable.core.agent_runner import AgentResult

    class StubRunner:
        def __init__(self, cfg, base_url, model, **kw):
            calls.append({"base_url": base_url, "model": model, **{
                k: kw.get(k) for k in ("sampling_override",)}})
            self.kw = kw

        def run(self, prompt, history=None, previous_summary=None):
            return AgentResult(final_response="ok",
                               messages=[{"role": "assistant",
                                          "content": "ok"}])

    monkeypatch.setattr("sable.core.agent_runner.create_runner", StubRunner)


def test_profile_server_instance_binding(home, monkeypatch):
    app = make_app(home)
    calls: list = []
    _stub_runner_capture(monkeypatch, calls)

    with FakeOpenAIServer() as default_fs, FakeOpenAIServer() as b_fs:
        # legacy default serve target
        from sable.serving.deploy import write_serve_state
        write_serve_state(home, {"mode": "attach",
                                 "base_url": default_fs.base_url,
                                 "model_name": "default-model"})
        # named instance B (attach) running
        r = create_server(app, "b", attach_url=b_fs.base_url,
                          sampling={"temperature": 0.1})
        sid = r.body["server"]["id"]
        assert app.handle("POST", f"/api/servers/{sid}/start",
                          body={}).status == 202

        # profile bound to B
        pr = app.handle("PUT", "/api/profiles/coder", body={
            "system_prompt": "code", "tools": [],
            "server_instance": sid})
        assert pr.status == 200
        assert pr.body["profile"]["server_instance"] == sid

        rr = app.handle("POST", "/api/chat/stream",
                        body={"prompt": "hi", "profile": "coder"})
        assert rr.status == 200 and rr.stream is not None
        "".join(rr.stream)
        assert calls[-1]["base_url"].rstrip("/") == b_fs.base_url.rstrip("/")
        # per-instance sampling override reached create_runner
        assert calls[-1]["sampling_override"] == {"temperature": 0.1}

        # unset profile => default target
        pr2 = app.handle("PUT", "/api/profiles/plain", body={
            "system_prompt": "", "tools": []})
        assert "server_instance" not in pr2.body["profile"]
        rr = app.handle("POST", "/api/chat/stream",
                        body={"prompt": "hi", "profile": "plain"})
        assert rr.stream is not None
        "".join(rr.stream)
        assert calls[-1]["base_url"].rstrip("/") == \
            default_fs.base_url.rstrip("/")
        assert calls[-1]["sampling_override"] is None


def test_profile_binding_errors(home):
    app = make_app(home)
    # unknown instance => 404 (resolved before the stream starts)
    pr = app.handle("PUT", "/api/profiles/ghost", body={
        "system_prompt": "", "tools": [], "server_instance": "nope"})
    assert pr.status == 200
    r = app.handle("POST", "/api/chat/stream",
                   body={"prompt": "hi", "profile": "ghost"})
    assert r.status == 404 and r.body["error"]["code"] == "unknown_server"

    # known but stopped => 503 naming the instance
    g = home / "x.gguf"
    g.write_bytes(b"GGUF")
    cr = create_server(app, "stopped-one", gguf=str(g))
    sid = cr.body["server"]["id"]
    app.handle("PUT", "/api/profiles/usesStopped", body={
        "system_prompt": "", "tools": [], "server_instance": sid})
    r = app.handle("POST", "/api/chat/stream",
                   body={"prompt": "hi", "profile": "usesStopped"})
    assert r.status == 503 and r.body["error"]["code"] == "not_serving"


# --------------------------------------------------------- 5. VRAM advisory

def test_vram_warning_and_force_bypass(home, stub_binary, ggufs, tmp_path,
                                       monkeypatch, cleanup_servers):
    app = make_app(home, llama_server_binary=stub_binary)
    big = tmp_path / "big.gguf"
    with open(big, "wb") as f:  # sparse 4 GiB => estimate way over "free"
        f.truncate(4 * 1024 ** 3)

    monkeypatch.setattr(srv_inst, "free_vram_mib", lambda: 512)

    r = create_server(app, "biggy", gguf=str(big))
    sid = r.body["server"]["id"]
    start = app.handle("POST", f"/api/servers/{sid}/start", body={})
    assert start.status == 202
    assert start.body["warning"] and "VRAM" in start.body["warning"]
    # warn, don't block: the server actually started
    assert must_row(app, sid)["running"] is True
    app.handle("POST", f"/api/servers/{sid}/stop", body={})

    # force bypasses the estimate entirely (no warning field)
    start2 = app.handle("POST", f"/api/servers/{sid}/start",
                        body={"force": True})
    assert start2.status == 202 and start2.body["warning"] is None
    app.handle("POST", f"/api/servers/{sid}/stop", body={})


def test_vram_advisory_skipped_without_telemetry(home, ggufs, monkeypatch):
    monkeypatch.setattr(srv_inst, "free_vram_mib", lambda: None)
    est = srv_inst.vram_advisory({"gguf": ggufs[0]})
    assert est is None


# ----------------------------------------------------------- 6. shutdown

def test_stop_all_servers_only_touches_owned(home, stub_binary, ggufs,
                                             monkeypatch):
    foreign = subprocess.Popen([sys.executable, "-c",
                                "import time; time.sleep(30)"])
    app = make_app(home, llama_server_binary=stub_binary)
    try:
        r = create_server(app, "owned", gguf=ggufs[0])
        sid = r.body["server"]["id"]
        start = app.handle("POST", f"/api/servers/{sid}/start", body={})
        assert start.status == 202
        child_pid = int((home.parent / "stub.pid").read_text())
        os.kill(child_pid, 0)  # alive

        stop_all_servers()

        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            try:
                os.kill(child_pid, 0)
                time.sleep(0.1)
            except OSError:
                break
        else:
            pytest.fail("owned instance child still running after shutdown")
        assert foreign.poll() is None  # never kill what we didn't spawn
    finally:
        foreign.kill()
        stop_all_servers()


def test_ports_in_use_sees_other_instances(home, ggufs):
    app = make_app(home)
    create_server(app, "p1", gguf=ggufs[0], port=45_999)
    used = srv_inst.ports_in_use(home, cfg=app.cfg)
    assert 45_999 in used
    assert srv_inst.check_port_conflict(home, app.cfg, 45_999) is True
    assert srv_inst.check_port_conflict(home, app.cfg, 46_000) is False
    # excluding its own id frees the port for updates
    sid = must_row(app, "p1")["id"]
    assert srv_inst.check_port_conflict(home, app.cfg, 45_999,
                                        exclude_id=sid) is False


# ------------------------- legacy /api/serve/start via instance machinery

def test_serve_start_registers_owned_handle_and_state(home, stub_binary,
                                                      ggufs, cleanup_servers):
    """h_serve_start now spawns through start_instance_server: the child lands
    in the _owned_handles registry (shutdown coverage), serve_state.json keeps
    its legacy shape for old readers, and /api/serve/stop reaches the child."""
    from sable.serving.supervisor import owned_instance_ids

    app = make_app(home, llama_server_binary=stub_binary)
    r = app.handle("POST", "/api/serve/start", body={"gguf": ggufs[0]})
    assert r.status == 202, r.body
    assert r.body["ok"] is True and r.body["base_url"].startswith("http://")

    # registry visibility: the reserved default id is an owned instance now
    assert "default" in owned_instance_ids()

    # legacy state file consumers (shadow eval, status panels): compatible shape
    st = json.loads((home / "serve_state.json").read_text())
    assert st["mode"] == "spawn" and st["pid"] and st["base_url"]
    assert st["instance_id"] == "default"
    assert int((home.parent / "stub.pid").read_text()) == st["pid"]
    # per-instance state mirrors it (Phase-6 view)
    mirror = srv_inst.read_state(home, "default")
    assert mirror and mirror["pid"] == st["pid"]

    # healthy endpoint + status surface agree
    assert httpx.get(r.body["base_url"] + "/models", timeout=5).status_code == 200
    status = app.handle("GET", "/api/serve/status").body
    assert status["running"] is True and status["pid"] == st["pid"]

    # stop path reaches the in-process handle: child dies, registry empties
    pid = st["pid"]
    sr = app.handle("POST", "/api/serve/stop", body={})
    assert sr.status == 200 and sr.body["stopped"] is True
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
            time.sleep(0.1)
        except OSError:
            break
    else:
        pytest.fail("default-server child survived /api/serve/stop")
    assert "default" not in owned_instance_ids()
    assert not (home / "serve_state.json").exists()


def test_serve_start_crash_loop_refusal_surfaces(home, stub_binary, ggufs,
                                                 cleanup_servers):
    """Over-budget starts are refused THROUGH the endpoint (500), while an
    explicit user start clears the budget and succeeds (#28 semantics)."""
    from sable.serving import supervisor as sup

    app = make_app(home, llama_server_binary=stub_binary)
    for _ in range(sup.CRASH_LOOP_MAX_FAILURES):
        sup.record_start_failure("default")

    # autostart=True keeps the refusal in force (mirrors h_server_start):
    # start_instance_server refuses before ever spawning.
    r = app.handle("POST", "/api/serve/start",
                   body={"gguf": ggufs[0], "autostart": True})
    assert r.status == 500 and "crash loop" in r.body["error"]["message"]

    # explicit user start clears the crash-loop state and spawns normally
    r2 = app.handle("POST", "/api/serve/start", body={"gguf": ggufs[1]})
    assert r2.status == 202, r2.body
    assert sup.crash_loop_status("default") is None  # healthy start reset it
    app.handle("POST", "/api/serve/stop", body={})


def test_serve_start_log_location(home, stub_binary, ggufs, cleanup_servers):
    """Spawn log stays at <home>/logs/server-default.log (user-visible path)."""
    app = make_app(home, llama_server_binary=stub_binary)
    r = app.handle("POST", "/api/serve/start", body={"gguf": ggufs[0]})
    assert r.status == 202
    logp = home / "logs" / "server-default.log"
    assert logp.is_file() and b"ready" in logp.read_bytes()
    app.handle("POST", "/api/serve/stop", body={})


def test_state_files_with_api_key_are_owner_only(home):
    """serve_state.json and serving/state/<id>.json can embed attach_api_key;
    both writers must land the file 0600 (default umask would leave them
    world-readable)."""
    from sable.serving.deploy import write_serve_state

    write_serve_state(home, {"mode": "attach", "base_url": "http://127.0.0.1:1",
                             "attach_api_key": "sk-secret"})
    mode = (home / "serve_state.json").stat().st_mode & 0o777
    assert mode == 0o600, f"serve_state.json mode {mode:o}, want 600"

    srv_inst.write_state(home, "inst-keyed",
                         {"pid": os.getpid(), "attach_api_key": "sk-secret"})
    p = srv_inst.instance_state_path(home, "inst-keyed")
    mode = p.stat().st_mode & 0o777
    assert mode == 0o600, f"instance state mode {mode:o}, want 600"

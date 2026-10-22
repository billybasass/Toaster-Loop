"""Inference-engine registry tests (serving/engines.py + dashboard API).

Hermetic: tmp SABLE_HOME (fixture), no GPU, no network, no spawns. The spawn
path is exercised only through argv/env builders and error paths (unknown /
non-openai / missing-binary engines) — nothing is launched.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sable.core.config import SableConfig  # noqa: E402
from sable.dashboard.app import DashboardApp  # noqa: E402
from sable.serving import engines as eng_mod  # noqa: E402
from sable.serving import instances as srv_inst  # noqa: E402
from sable.serving.supervisor import ServeError  # noqa: E402


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "uf-engines"
    monkeypatch.setenv("SABLE_HOME", str(h))
    return h


@pytest.fixture()
def app(home):
    return DashboardApp(SableConfig(), home=home)


def err_code(r) -> str:
    return (r.body.get("error") or {}).get("code", "")


# ------------------------------------------------------------- registry core

def test_builtin_engine_always_listed(home):
    engines = eng_mod.list_engines(home)
    assert engines[0]["id"] == eng_mod.BUILTIN_ID
    assert engines[0]["builtin"] is True
    # builtin is never persisted
    assert not (home / "serving" / "engines").exists()


def test_validate_engine_body_defaults_and_normalization():
    e = eng_mod.validate_engine_body({
        "name": "vLLM", "binary": "vllm",
        "args_template": "--model {model} --port {port}",
    })
    assert e["kind"] == "openai"
    assert e["binary"] == "vllm"
    # string templates are shlex-split
    assert e["args_template"] == ["--model", "{model}", "--port", "{port}"]

    # openai kind with no template => loopback-safe default template
    e2 = eng_mod.validate_engine_body({"name": "X", "binary": "x-server"})
    assert "{model}" in " ".join(e2["args_template"])
    assert "127.0.0.1" in " ".join(e2["args_template"])

    # kind other => no default template (inventory only)
    e3 = eng_mod.validate_engine_body({"name": "SD", "binary": "sd-cli",
                                       "kind": "other"})
    assert e3["args_template"] == []


def test_validate_engine_body_rejects_bad_input():
    with pytest.raises(ServeError):
        eng_mod.validate_engine_body({"binary": "x"})           # no name
    with pytest.raises(ServeError):
        eng_mod.validate_engine_body({"name": "n"})             # no binary
    with pytest.raises(ServeError):
        eng_mod.validate_engine_body({"name": "n", "binary": "b",
                                      "kind": "magic"})
    with pytest.raises(ServeError):
        eng_mod.validate_engine_body({"name": "n", "binary": "b",
                                      "env": "not-a-dict"})


def test_builtin_id_reserved(home):
    with pytest.raises(ServeError):
        eng_mod.validate_engine_body({"id": eng_mod.BUILTIN_ID,
                                      "name": "n", "binary": "b"})
    with pytest.raises(ServeError):
        eng_mod.delete_engine_file(home, eng_mod.BUILTIN_ID)


def test_save_get_delete_roundtrip(home):
    e = eng_mod.validate_engine_body({"name": "vLLM", "binary": "vllm"})
    e["id"] = eng_mod.new_engine_id("vLLM", ["llama-cpp"])
    assert e["id"] == "vLLM"
    eng_mod.save_engine(home, e)
    again = eng_mod.get_engine(home, "vLLM")
    assert again and again["binary"] == "vllm"
    ids = [x["id"] for x in eng_mod.list_engines(home)]
    assert ids == ["llama-cpp", "vLLM"]
    assert eng_mod.delete_engine_file(home, "vLLM") is True
    assert eng_mod.get_engine(home, "vLLM") is None


def test_new_engine_id_avoids_builtin_and_collisions():
    taken = ["llama-cpp", "vllm"]
    assert eng_mod.new_engine_id("llama-cpp", taken) != "llama-cpp"
    assert eng_mod.new_engine_id("vllm", taken) == "vllm-2"


def test_probe_missing_binary(home, monkeypatch):
    e = eng_mod.validate_engine_body({"name": "X", "binary": "definitely-not-here"})
    monkeypatch.setattr(eng_mod.shutil, "which", lambda _n: None)
    assert eng_mod.probe(e) == {"found": False, "path": None}


def test_probe_finds_binary(home, monkeypatch):
    e = eng_mod.validate_engine_body({"name": "X", "binary": "x-server"})
    monkeypatch.setattr(eng_mod.shutil, "which",
                        lambda _n: "/usr/local/bin/x-server")
    p = eng_mod.probe(e)
    assert p["found"] and p["path"] == "/usr/local/bin/x-server"


# ------------------------------------------------------------- command build

def test_build_engine_cmd_substitutions():
    e = {"kind": "openai", "args_template":
         ["serve", "--model", "{model}", "--host", "{host}",
          "--port", "{port}", "--literal", "{unknown}"]}
    cmd = eng_mod.build_engine_cmd(e, "/bin/x", model="/m/model.gguf",
                                   host="127.0.0.1", port=8123)
    assert cmd == ["/bin/x", "serve", "--model", "/m/model.gguf",
                   "--host", "127.0.0.1", "--port", "8123",
                   "--literal", "{unknown}"]


def test_build_engine_cmd_default_template():
    cmd = eng_mod.build_engine_cmd({"kind": "openai"}, "/bin/x",
                                   model="m.gguf", host="127.0.0.1", port=9)
    assert cmd[0] == "/bin/x"
    assert "{model}" not in " ".join(cmd)
    assert "m.gguf" in cmd and "9" in cmd


def test_engine_env_stringifies():
    assert eng_mod.engine_env({"env": {"A": 1, "B": None}}) == {"A": "1"}
    assert eng_mod.engine_env({}) == {}


# ------------------------------------------------------- instance binding

def test_instance_body_engine_defaults_to_none():
    inst = srv_inst.validate_instance_body(
        {"name": "s", "gguf": "/tmp/x.gguf"})
    assert inst["engine"] is None


def test_instance_body_llama_cpp_alias_is_default_path():
    inst = srv_inst.validate_instance_body(
        {"name": "s", "gguf": "/tmp/x.gguf", "engine": "llama-cpp"})
    assert inst["engine"] is None


def test_instance_body_keeps_registered_engine():
    inst = srv_inst.validate_instance_body(
        {"name": "s", "gguf": "/tmp/x.gguf", "engine": "vllm"})
    assert inst["engine"] == "vllm"


def test_default_instance_reads_serve_default_engine():
    cfg = SableConfig()
    assert srv_inst.default_instance(cfg)["engine"] is None
    cfg.serve = {"default_engine": "vllm"}
    assert srv_inst.default_instance(cfg)["engine"] == "vllm"
    cfg.serve = {"default_engine": "  "}   # blank => built-in path
    assert srv_inst.default_instance(cfg)["engine"] is None


def test_engines_in_use(home):
    e = eng_mod.validate_engine_body({"name": "vLLM", "binary": "vllm"})
    e["id"] = "vllm"
    eng_mod.save_engine(home, e)
    srv_inst.save_instance(home, {"id": "coder", "name": "coder",
                                  "gguf": "/tmp/x.gguf", "engine": "vllm"})
    assert eng_mod.engines_in_use(home, "vllm") == ["coder"]
    assert eng_mod.engines_in_use(home, "other") == []


# ---------------------------------------------------------- dashboard API

def test_engines_routes_registered(app):
    paths = {(m, p) for (m, p, _) in app.ROUTES}
    for m, p in [("GET", "/api/engines"), ("POST", "/api/engines"),
                 ("PUT", "/api/engines/<engine_id>"),
                 ("DELETE", "/api/engines/<engine_id>")]:
        assert (m, p) in paths, f"{m} {p} missing"


def test_engines_list_builtin_first(app, home):
    r = app.handle("GET", "/api/engines")
    assert r.status == 200
    ids = [e["id"] for e in r.body["engines"]]
    assert ids[0] == "llama-cpp"
    assert r.body["default_engine"] is None


def test_engines_create_update_delete(app, home):
    r = app.handle("POST", "/api/engines", body={
        "name": "vLLM", "binary": "vllm",
        "args_template": "--model {model} --port {port}"})
    assert r.status == 201, r.body
    eid = r.body["engine"]["id"]
    assert eid == "vLLM"

    upd = app.handle("PUT", f"/api/engines/{eid}", body={"notes": "local"})
    assert upd.status == 200 and upd.body["engine"]["notes"] == "local"

    lst = app.handle("GET", "/api/engines")
    row = next(e for e in lst.body["engines"] if e["id"] == eid)
    assert row["installed"] is False  # binary absent in the test env

    dele = app.handle("DELETE", f"/api/engines/{eid}")
    assert dele.status == 200
    ids = [e["id"] for e in app.handle("GET", "/api/engines").body["engines"]]
    assert eid not in ids


def test_engines_builtin_is_immutable(app, home):
    bad = app.handle("PUT", "/api/engines/llama-cpp", body={"name": "x"})
    assert bad.status == 400 and err_code(bad) == "reserved_engine"
    bad2 = app.handle("DELETE", "/api/engines/llama-cpp")
    assert bad2.status == 400 and err_code(bad2) == "reserved_engine"


def test_engines_delete_blocked_while_in_use(app, home):
    r = app.handle("POST", "/api/engines", body={"name": "vLLM",
                                                 "binary": "vllm"})
    eid = r.body["engine"]["id"]
    srv_inst.save_instance(home, {"id": "coder", "name": "coder",
                                  "gguf": "/tmp/x.gguf", "engine": eid})
    dele = app.handle("DELETE", f"/api/engines/{eid}")
    assert dele.status == 409 and err_code(dele) == "in_use"


def test_instance_create_rejects_unknown_engine(app, home):
    r = app.handle("POST", "/api/servers", body={
        "name": "coder", "gguf": "/tmp/x.gguf", "engine": "ghost"})
    assert r.status == 400 and err_code(r) == "unknown_engine"


def test_instance_create_accepts_registered_engine(app, home):
    e = app.handle("POST", "/api/engines", body={"name": "vLLM",
                                                 "binary": "vllm"})
    eid = e.body["engine"]["id"]
    r = app.handle("POST", "/api/servers", body={
        "name": "coder", "gguf": "/tmp/x.gguf", "engine": eid})
    assert r.status == 201, r.body
    assert r.body["server"]["engine"] == eid


def test_config_put_serve_default_engine(app, home):
    """serve.default_engine is a recognized serve key (Settings page writes
    it through the section patch)."""
    r = app.handle("PUT", "/api/config", body={
        "section": "serve", "patch": {"default_engine": "vllm"}})
    assert r.status == 200, r.body
    raw = (home / "sable.yaml").read_text()
    assert "default_engine: vllm" in raw


# ------------------------------------------------------- spawn error paths

def test_engine_spawn_unknown_engine_error(home, monkeypatch):
    from sable.serving.supervisor import _engine_spawn_cmd

    cfg = SableConfig()
    monkeypatch.setenv("SABLE_HOME", str(home))
    with pytest.raises(ServeError, match="unknown inference engine"):
        _engine_spawn_cmd(cfg, {}, "ghost", gguf="m.gguf", port=1)


def test_engine_spawn_non_openai_refuses(home, monkeypatch):
    from sable.serving.supervisor import _engine_spawn_cmd

    e = eng_mod.validate_engine_body({"name": "SD", "binary": "sd-cli",
                                      "kind": "other"})
    e["id"] = "sd"
    eng_mod.save_engine(home, e)
    monkeypatch.setenv("SABLE_HOME", str(home))
    with pytest.raises(ServeError, match="only"):
        _engine_spawn_cmd(SableConfig(), {}, "sd", gguf="m.gguf", port=1)


def test_engine_spawn_missing_binary_error(home, monkeypatch):
    from sable.serving.supervisor import _engine_spawn_cmd

    e = eng_mod.validate_engine_body({"name": "vLLM", "binary": "vllm"})
    e["id"] = "vllm"
    eng_mod.save_engine(home, e)
    monkeypatch.setenv("SABLE_HOME", str(home))
    monkeypatch.setattr(eng_mod.shutil, "which", lambda _n: None)
    with pytest.raises(ServeError, match="binary not found"):
        _engine_spawn_cmd(SableConfig(), {}, "vllm", gguf="m.gguf",
                          port=1)

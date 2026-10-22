"""Per-agent capabilities: internet (web_fetch) and run programs (run_command).

Covers the three enforcement layers:
  1. core/tools.resolve_capabilities — default-off, global then profile merge
  2. core/agent_runner — capability tools never exposed unless enabled; both
     schema AND impl gated; profile tool-subset interaction
  3. dashboard profiles API — capabilities validation + persistence, and the
     chat handler threading them into create_runner (delegation included)

Plus direct unit tests of web_fetch SSRF posture and run_command behavior.
All hermetic; conftest redirects SABLE_HOME to a tmp dir.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sable.core.agent_runner import SableAgentClient, create_runner
from sable.core.config import SableConfig
from sable.core.tools import (
    CAPABILITY_TOOLS,
    resolve_capabilities,
    run_command,
    web_fetch,
)


# ------------------------------------------------------------- resolution

def test_capabilities_default_off():
    assert resolve_capabilities() == {
        "internet": False, "run_commands": False,
        "vision": False, "screenshots": False}


def test_capabilities_global_then_profile_merge():
    caps = resolve_capabilities({"capabilities": {"internet": True}},
                                {"internet": False, "run_commands": True})
    assert caps["internet"] is False and caps["run_commands"] is True


def test_capabilities_unknown_keys_ignored():
    caps = resolve_capabilities({"capabilities": {"teleport": True}},
                                {"launch_missiles": True, "internet": 1})
    assert caps["internet"] is True and caps["run_commands"] is False


# ------------------------------------------------------------- runner gate

def _tool_names(client: SableAgentClient):
    return {s.get("function", {}).get("name") for s in client.tools}


def test_default_runner_exposes_no_capability_tools():
    client = SableAgentClient(base_url="http://127.0.0.1:1/v1", model="m")
    names = _tool_names(client)
    assert "web_fetch" not in names and "run_command" not in names
    assert "read_file" in names  # sanity: the normal set is intact
    for tool in CAPABILITY_TOOLS:
        assert tool not in client.tool_impls


def test_capability_enabled_exposes_schema_and_impl():
    client = SableAgentClient(
        base_url="http://127.0.0.1:1/v1", model="m",
        capabilities={"internet": True, "run_commands": False})
    names = _tool_names(client)
    assert "web_fetch" in names
    assert "run_command" not in names
    assert "web_fetch" in client.tool_impls
    assert "run_command" not in client.tool_impls


def test_capability_tool_honors_profile_tool_subset():
    # Profile lists web_fetch but the capability is OFF => still not exposed.
    client = SableAgentClient(
        base_url="http://127.0.0.1:1/v1", model="m",
        enabled_tool_names=["read_file", "web_fetch"])
    assert "web_fetch" not in _tool_names(client)
    assert "web_fetch" not in client.tool_impls

    # Capability ON grants the tool even when the subset does not list it.
    client = SableAgentClient(
        base_url="http://127.0.0.1:1/v1", model="m",
        enabled_tool_names=["read_file"],
        capabilities={"internet": True})
    assert "web_fetch" in _tool_names(client)
    assert "web_fetch" in client.tool_impls
    # ...and unrelated tools the profile omitted stay gone.
    assert "write_file" not in _tool_names(client)


def test_create_runner_merges_global_and_profile_capabilities():
    cfg = SableConfig(agent={"capabilities": {"internet": True}})
    runner = create_runner(cfg, base_url="http://127.0.0.1:1/v1", model="m")
    assert "web_fetch" in _tool_names(runner)

    # Profile override turns the global default back off.
    runner = create_runner(cfg, base_url="http://127.0.0.1:1/v1", model="m",
                           capabilities={"internet": False})
    assert "web_fetch" not in _tool_names(runner)
    assert "web_fetch" not in runner.tool_impls


def test_create_runner_web_fetch_config_binds():
    cfg = SableConfig(tools={"web_fetch": {"allow_private_networks": True}})
    runner = create_runner(cfg, base_url="http://127.0.0.1:1/v1", model="m",
                           capabilities={"internet": True})
    impl = runner.tool_impls["web_fetch"]
    # Bound via functools.partial — the model cannot forge the knob.
    assert getattr(impl, "keywords", {}).get("allow_private_networks") is True


# ------------------------------------------------------------ web_fetch

def test_web_fetch_refuses_non_http_schemes():
    out = web_fetch("file:///etc/passwd")
    assert out.startswith("error: blocked:")
    assert "scheme" in out


def test_web_fetch_blocks_loopback_and_private_by_default():
    for url in ("http://127.0.0.1/x", "http://localhost/x",
                "http://169.254.169.254/latest/meta-data/",
                "http://10.0.0.5/x", "http://[::1]/x"):
        out = web_fetch(url)
        assert out.startswith("error: blocked:"), url


def test_web_fetch_allow_private_knob(tmp_path):
    # With the knob on, the address-space rule is relaxed (the fetch itself
    # then fails at connect time against a dead port — NOT as "blocked").
    out = web_fetch("http://127.0.0.1:9/", allow_private_networks=True)
    assert not out.startswith("error: blocked:")
    assert out.startswith("error: web_fetch failed")


# ------------------------------------------------------------ run_command

def test_run_command_captures_streams_and_exit_code():
    out = run_command("echo hello; echo oops >&2; exit 3")
    assert "exit code: 3" in out
    assert "hello" in out and "oops" in out


def test_run_command_timeout_kills_child():
    out = run_command("sleep 30", timeout_seconds=1)
    assert "timed out" in out


def test_run_command_rejects_empty():
    assert run_command("   ").startswith("error:")


# --------------------------------------------------- dashboard profiles API

def _app(home):
    from sable.dashboard.app import DashboardApp

    return DashboardApp(SableConfig(), home=home)


def test_profiles_api_validates_and_persists_capabilities(tmp_path):
    app = _app(str(tmp_path))
    r = app.handle("PUT", "/api/profiles/researcher", body={
        "system_prompt": "browse for me",
        "tools": ["read_file"],
        "capabilities": {"internet": True, "run_commands": False},
    })
    assert r.status < 400, r.body
    assert r.body["profile"]["capabilities"] == {"internet": True,
                                                 "run_commands": False}

    lst = app.handle("GET", "/api/profiles")
    profs = {p["name"]: p for p in lst.body["profiles"]}
    assert profs["researcher"]["capabilities"]["internet"] is True
    assert lst.body["capability_toggles"] == {
        "internet": "web_fetch", "run_commands": "run_command",
        "vision": "view_image", "screenshots": "take_screenshot"}
    assert lst.body["global_capabilities"] == {"internet": False,
                                               "run_commands": False,
                                               "vision": False,
                                               "screenshots": False}


def test_profiles_api_rejects_unknown_capability(tmp_path):
    app = _app(str(tmp_path))
    r = app.handle("PUT", "/api/profiles/evil", body={
        "capabilities": {"internet": True, "teleport": True}})
    assert r.status == 400
    assert "unknown capability" in r.body["error"]["message"]


def test_profiles_api_rejects_non_dict_capabilities(tmp_path):
    app = _app(str(tmp_path))
    r = app.handle("PUT", "/api/profiles/evil", body={"capabilities": ["a"]})
    assert r.status == 400


def test_profile_capabilities_survive_config_roundtrip(tmp_path):
    import yaml

    from sable.core.config import load_config

    app = _app(str(tmp_path))
    app.handle("PUT", "/api/profiles/webkit", body={
        "system_prompt": "s", "tools": [],
        "capabilities": {"internet": True}})
    cfg_file = Path(str(tmp_path)) / "sable.yaml"
    assert cfg_file.is_file()
    data = yaml.safe_load(cfg_file.read_text())
    prof = next(p for p in data["profiles"] if p["name"] == "webkit")
    assert prof["capabilities"] == {"internet": True}

    cfg = load_config(str(cfg_file))
    runner = create_runner(cfg, base_url="http://127.0.0.1:1/v1", model="m",
                           capabilities=prof["capabilities"])
    assert "web_fetch" in _tool_names(runner)
    assert "run_command" not in _tool_names(runner)


# --------------------------------------------------------------- delegation

def test_delegation_child_inherits_profile_capabilities(tmp_path, monkeypatch):
    """_run_child passes the CHILD profile's capabilities into create_runner."""
    from sable.core import delegation as deleg
    from sable.core.agent_runner import create_runner as real_create

    seen: dict = {}

    def spy(cfg, base_url, model, **kw):
        seen["capabilities"] = kw.get("capabilities")
        raise RuntimeError("stop-after-spy")

    monkeypatch.setattr("sable.core.agent_runner.create_runner", spy)
    cfg = SableConfig(profiles=[{
        "name": "surf", "tools": [], "capabilities": {"internet": True}}])
    with pytest.raises(RuntimeError, match="stop-after-spy"):
        deleg._run_child(cfg, cfg.profiles[0], "task", {"base_url": "u"},
                         "dlg1", {})
    assert seen["capabilities"] == {"internet": True}
    # sanity: real_create untouched for other tests
    assert real_create is not None

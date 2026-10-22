"""One-shot feature presets (core/presets.py + POST /api/features/enable_all).

Coverage: the patch list is well-formed and passes the dashboard's own
section/key validation; the conditional switches (LSP, calibration) follow
their predicates; preview writes nothing; apply flips exactly the intended
switches through the normal config path and leaves the deliberate exclusions
(easy_training, janitor) untouched; a mid-apply failure rolls the whole thing
back (same all-or-nothing contract as the easy-training enable).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sable.core.config import SableConfig, load_config  # noqa: E402
from sable.core.presets import (  # noqa: E402
    ENABLE_ALL_SWITCHES,
    enable_all_diff,
    enable_all_patches,
)
from sable.dashboard.app import DashboardApp  # noqa: E402


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "sable-presets"
    monkeypatch.setenv("SABLE_HOME", str(h))
    return h


@pytest.fixture()
def app(home):
    a = DashboardApp(SableConfig(), home=home)
    yield a


# --------------------------------------------------------------- patch list

def test_patches_cover_known_sections_and_bool_switches():
    for section, patch in ENABLE_ALL_SWITCHES.items():
        assert all(isinstance(v, bool) for v in patch.values())
    # The headline features must be in the list.
    assert ENABLE_ALL_SWITCHES["skills"]["enabled"] is True
    assert ENABLE_ALL_SWITCHES["playbooks"]["enabled"] is True
    assert ENABLE_ALL_SWITCHES["routing"]["enabled"] is True


def test_easy_training_and_janitor_never_in_preset():
    # Cardinal: the loop and auto-archive are decisions, not switches.
    assert "easy_training" not in ENABLE_ALL_SWITCHES
    assert "janitor" not in ENABLE_ALL_SWITCHES


def test_conditional_switches_off_without_env(home, monkeypatch):
    # No language server on PATH, no calibration profile => neither appears.
    monkeypatch.setattr("shutil.which", lambda *a, **k: None)
    sections = [p["section"] for p in enable_all_patches(SableConfig(), home=str(home))]
    tools = [p for p in enable_all_patches(SableConfig(), home=str(home))
             if p["section"] == "tools"]
    assert not tools  # lsp patch absent
    # serve only ever carries watchdog (a plain switch), never calibration here
    for p in enable_all_patches(SableConfig(), home=str(home)):
        if p["section"] == "serve":
            assert set(p["patch"]) == {"watchdog_enabled"}
    assert "serve" in sections  # watchdog itself is unconditional


def test_lsp_included_when_server_on_path(home, monkeypatch):
    monkeypatch.setattr("shutil.which",
                        lambda name, **k: "/usr/bin/" + name
                        if name == "pyright-langserver" else None)
    tools = [p for p in enable_all_patches(SableConfig(), home=str(home))
             if p["section"] == "tools"]
    assert tools and tools[0]["patch"]["lsp"]["enabled"] is True


def test_diff_reports_only_differing_keys():
    cfg = SableConfig()
    cfg.skills = {"enabled": True}  # already on => absent from diff
    d = {(x["section"], x["key"]): x for x in enable_all_diff(cfg)}
    assert ("skills", "enabled") not in d
    assert d[("routing", "enabled")]["to"] is True
    assert d[("validation", "shadow_enabled")]["to"] is True


# ---------------------------------------------------------------- endpoints

def test_preview_writes_nothing(app, home):
    cfg_file = home / "sable.yaml"
    assert not cfg_file.exists()
    r = app.handle("POST", "/api/features/enable_all", body={"preview": True})
    assert r.status == 200
    assert isinstance(r.body["diff"], list) and r.body["diff"]
    assert not cfg_file.exists()
    # live config untouched
    assert not (app.cfg.skills or {}).get("enabled")


def test_apply_flips_switches_and_preserves_unrelated_keys(app, home):
    app.handle("PUT", "/api/config", body={
        "section": "skills", "patch": {"index_total_cap_chars": 999}})
    app.handle("PUT", "/api/config", body={
        "section": "easy_training", "patch": {"enabled": False}})
    r = app.handle("POST", "/api/features/enable_all", body={})
    assert r.status == 200, r.body
    cfg = load_config(str(home / "sable.yaml"))
    assert (cfg.skills or {}).get("enabled") is True
    assert (cfg.skills or {}).get("index_total_cap_chars") == 999  # merge kept it
    assert (cfg.playbooks or {}).get("enabled") is True
    assert (cfg.routing or {}).get("enabled") is True
    assert (cfg.serve or {}).get("watchdog_enabled") is True
    assert (cfg.validation or {}).get("shadow_enabled") is True
    assert (cfg.rft or {}).get("enabled") is True
    assert (cfg.evolution or {}).get("enabled") is True
    assert (cfg.goldens_mine or {}).get("enabled") is True
    # deliberate exclusions untouched
    assert not (cfg.easy_training or {}).get("enabled")
    assert not (cfg.janitor or {}).get("enabled")


def test_apply_idempotent_empty_diff_second_time(app, home):
    assert app.handle("POST", "/api/features/enable_all", body={}).status == 200
    r = app.handle("POST", "/api/features/enable_all",
                   body={"preview": True})
    assert r.status == 200 and r.body["diff"] == []


def test_rollback_on_failed_patch(app, home, monkeypatch):
    # Poison the SECOND section-patch so the first write lands and then must
    # revert. handle() converts PUT failures to ApiResponse (never raises),
    # which is exactly the contract h_features_enable_all rolls back on.
    called = {"n": 0}
    orig = DashboardApp.handle

    def flaky(self, method, path, query=None, body=None):
        if method == "PUT" and path == "/api/config":
            called["n"] += 1
            if called["n"] == 2:
                from sable.dashboard.app import ApiResponse
                return ApiResponse(500, {"error": {
                    "code": "test", "message": "boom"}})
        return orig(self, method, path, query, body)

    monkeypatch.setattr(DashboardApp, "handle", flaky)
    # handle() converts the handler's HttpError into an error ApiResponse.
    r = orig(app, "POST", "/api/features/enable_all", body={})
    assert r.status >= 400
    assert r.body["error"]["message"] == "boom"
    assert called["n"] == 2  # first landed, second failed, then rollback
    # After rollback neither the live config nor the file has any switch on.
    assert not (app.cfg.skills or {}).get("enabled")
    assert not (app.cfg.playbooks or {}).get("enabled")
    cfg_file = home / "sable.yaml"
    if cfg_file.is_file():
        cfg = load_config(str(cfg_file))
        assert not (cfg.skills or {}).get("enabled")
        assert not (cfg.playbooks or {}).get("enabled")

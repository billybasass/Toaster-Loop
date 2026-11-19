"""Self-update system tests (core/updater.py + dashboard routes + CLI).

Hermetic: no network (the GitHub Releases call and every git subprocess are
monkeypatched), no real installs mutated.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from toaster_loop.core import updater
from toaster_loop.core.config import ToasterLoopConfig


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "uf-upd"
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(h))
    return h


# ------------------------------------------------------------------ helpers

def _cache(home: Path, **over) -> dict:
    rec = {"checked_at": datetime.now(timezone.utc).isoformat(),
           "current_version": "0.1.0", "repo": "o/r", "install": "git",
           "available": False, "remote_version": None,
           "url": "https://github.com/o/r", "error": None}
    rec.update(over)
    home.mkdir(parents=True, exist_ok=True)
    (home / updater.CHECK_FILENAME).write_text(json.dumps(rec))
    return rec


# ------------------------------------------------------------------ versions

def test_parse_version():
    assert updater.parse_version("v0.2.1") == (0, 2, 1)
    assert updater.parse_version("0.10.3-rc1") == (0, 10, 3)
    assert updater.parse_version("garbage") is None
    assert updater.parse_version("") is None


def test_is_newer():
    assert updater._is_newer("v0.2.0", "0.1.9")
    assert not updater._is_newer("v0.1.0", "0.1.0")
    assert not updater._is_newer("0.0.9", "0.1.0")
    # Unparseable either side -> never claim an update.
    assert not updater._is_newer("nightly", "0.1.0")
    assert not updater._is_newer("v1.0.0", "?")


def test_repo_from_url():
    assert updater._repo_from_url(
        "https://github.com/billybasass/Toaster-Loop.git") == \
        "billybasass/Toaster-Loop"
    assert updater._repo_from_url(
        "git@github.com:acme/toaster-loop.git") == "acme/toaster-loop"
    assert updater._repo_from_url("nonsense") is None


def test_update_config_defaults_and_tolerance():
    uc = updater.update_config(ToasterLoopConfig())
    assert uc["enabled"] is True and uc["repo"] is None
    assert uc["interval_hours"] == updater.DEFAULT_INTERVAL_HOURS

    cfg = ToasterLoopConfig(update={"enabled": False, "repo": "me/fork",
                                    "interval_hours": "6"})
    uc = updater.update_config(cfg)
    assert uc["enabled"] is False and uc["repo"] == "me/fork"
    assert uc["interval_hours"] == 6.0

    # Garbage interval falls back to the default, never raises.
    cfg = ToasterLoopConfig(update={"interval_hours": "soon"})
    assert updater.update_config(cfg)["interval_hours"] == \
        updater.DEFAULT_INTERVAL_HOURS


def test_resolve_repo_config_override():
    cfg = ToasterLoopConfig(update={"repo": "https://github.com/me/fork.git"})
    assert updater.resolve_repo(cfg, root=None) == "me/fork"


def test_resolve_repo_default_without_git(monkeypatch):
    monkeypatch.setattr(updater, "git_install_root", lambda: None)
    cfg = ToasterLoopConfig()
    assert updater.resolve_repo(cfg) == updater.DEFAULT_REPO


# --------------------------------------------------------------- check flow

def test_check_disabled_never_runs(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(updater, "_run_check",
                        lambda cfg: calls.append(1) or {})
    cfg = ToasterLoopConfig(update={"enabled": False})
    st = updater.check_for_update(cfg, home=tmp_path, force=True)
    assert calls == [] and st["available"] is False
    assert st.get("enabled") is False


def test_check_uses_fresh_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(updater, "_run_check",
                        lambda cfg: pytest.fail("must not re-check"))
    _cache(tmp_path, available=True, remote_version="9.9.9")
    st = updater.check_for_update(ToasterLoopConfig(), home=tmp_path)
    assert st["available"] is True and st.get("cached") is True
    assert st["remote_version"] == "9.9.9"


def test_check_stale_cache_refetches(monkeypatch, tmp_path):
    old = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
    _cache(tmp_path, checked_at=old)
    monkeypatch.setattr(
        updater, "_run_check",
        lambda cfg: {"available": True, "current_version": "0.1.0",
                     "remote_version": "0.2.0", "checked_at": old})
    st = updater.check_for_update(ToasterLoopConfig(), home=tmp_path)
    assert st["available"] is True and not st.get("cached")


def test_check_degrades_on_error_with_cache(monkeypatch, tmp_path):
    # Stale cache + forced re-check that fails -> last-known state with the
    # failure recorded in `error`.
    old = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
    _cache(tmp_path, available=False, checked_at=old)

    def boom(cfg):
        raise RuntimeError("no network")

    monkeypatch.setattr(updater, "_run_check", boom)
    st = updater.check_for_update(ToasterLoopConfig(), home=tmp_path)
    assert st["available"] is False and "check failed" in st["error"]
    assert st.get("cached") is True


def test_check_degrades_on_error_without_cache(monkeypatch, tmp_path):
    def boom(cfg):
        raise RuntimeError("no network")

    monkeypatch.setattr(updater, "_run_check", boom)
    st = updater.check_for_update(ToasterLoopConfig(), home=tmp_path)
    assert st["available"] is False and "check failed" in st["error"]


def test_run_check_release_newer(monkeypatch, tmp_path):
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(tmp_path))
    monkeypatch.setattr(updater, "git_install_root", lambda: None)
    monkeypatch.setattr(updater, "_remote_release",
                        lambda repo: {"tag": "v9.9.9", "url": "u", "notes": ""})
    st = updater._run_check(ToasterLoopConfig())
    assert st["available"] is True and st["remote_version"] == "9.9.9"
    # Result persisted for the next throttled read.
    assert (tmp_path / updater.CHECK_FILENAME).is_file()


def test_run_check_git_ahead_without_release(monkeypatch, tmp_path):
    """The common dev-push case: no release cut, but origin moved."""
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(tmp_path))
    root = tmp_path / "repo"
    monkeypatch.setattr(updater, "git_install_root", lambda: root)
    monkeypatch.setattr(updater, "_remote_release",
                        lambda repo: {"error": "no releases"})
    monkeypatch.setattr(
        updater, "git_state",
        lambda r: {"branch": "main", "head": "a" * 40, "dirty": False})
    monkeypatch.setattr(updater, "_git",
                        lambda r, *a: ("b" * 40 + "\trefs/heads/main"
                                       if a[0] == "ls-remote" else ""))
    st = updater._run_check(ToasterLoopConfig())
    assert st["available"] is True          # git channel caught it
    assert st["git"]["remote_head"] == "b" * 12


def test_run_check_git_up_to_date(monkeypatch, tmp_path):
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(tmp_path))
    root = tmp_path / "repo"
    monkeypatch.setattr(updater, "git_install_root", lambda: root)
    monkeypatch.setattr(updater, "_remote_release",
                        lambda repo: {"error": "no releases"})
    monkeypatch.setattr(
        updater, "git_state",
        lambda r: {"branch": "main", "head": "a" * 40, "dirty": False})
    monkeypatch.setattr(updater, "_git",
                        lambda r, *a: ("a" * 40 + "\trefs/heads/main"
                                       if a[0] == "ls-remote" else ""))
    st = updater._run_check(ToasterLoopConfig())
    assert st["available"] is False


# ---------------------------------------------------------------- apply

def test_apply_refuses_non_git(monkeypatch):
    monkeypatch.setattr(updater, "git_install_root", lambda: None)
    with pytest.raises(updater.UpdateError, match="not a git-checkout"):
        updater.apply_update(ToasterLoopConfig())


def test_apply_refuses_dirty_worktree(monkeypatch, tmp_path):
    monkeypatch.setattr(updater, "git_install_root", lambda: tmp_path)
    monkeypatch.setattr(updater, "git_state",
                        lambda r: {"branch": "main", "head": "a" * 40,
                                   "dirty": True})
    with pytest.raises(updater.UpdateError, match="uncommitted"):
        updater.apply_update(ToasterLoopConfig())


def test_apply_refuses_detached_head(monkeypatch, tmp_path):
    monkeypatch.setattr(updater, "git_install_root", lambda: tmp_path)
    monkeypatch.setattr(updater, "git_state",
                        lambda r: {"branch": "HEAD", "head": "a" * 40,
                                   "dirty": False})
    with pytest.raises(updater.UpdateError, match="detached"):
        updater.apply_update(ToasterLoopConfig())


def test_apply_noop_when_already_current(monkeypatch, tmp_path):
    monkeypatch.setattr(updater, "git_install_root", lambda: tmp_path)
    monkeypatch.setattr(updater, "git_state",
                        lambda r: {"branch": "main", "head": "a" * 40,
                                   "dirty": False})
    monkeypatch.setattr(updater, "_git_strict",
                        lambda r, *a, **k: "a" * 40)
    res = updater.apply_update(ToasterLoopConfig())
    assert res["changed"] is False


def test_apply_pull_and_reinstall(monkeypatch, tmp_path):
    monkeypatch.setattr(updater, "git_install_root", lambda: tmp_path)
    monkeypatch.setattr(updater, "git_state",
                        lambda r: {"branch": "main", "head": "a" * 40,
                                   "dirty": False})
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "toaster-loop"\nversion = "0.2.0"\n')
    calls = []

    def fake_strict(root, *a, **k):
        calls.append(a[0])
        return "b" * 40 if a[0] == "rev-parse" else ""

    monkeypatch.setattr(updater, "_git_strict", fake_strict)
    pip = {}

    def fake_run(cmd, **k):
        pip["cmd"] = cmd

        class R:
            returncode = 0
            stdout = stderr = ""
        return R()

    monkeypatch.setattr(updater.subprocess, "run", fake_run)
    res = updater.apply_update(ToasterLoopConfig())
    assert calls == ["fetch", "rev-parse", "merge"]
    assert res["changed"] is True and res["to"] == "0.2.0"
    assert res["restart_required"] is True
    # Reinstall keeps the editable mode (setup.sh workflow).
    assert "-e" in pip["cmd"] and str(tmp_path) in pip["cmd"]


def test_apply_pip_failure_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(updater, "git_install_root", lambda: tmp_path)
    monkeypatch.setattr(updater, "git_state",
                        lambda r: {"branch": "main", "head": "a" * 40,
                                   "dirty": False})
    monkeypatch.setattr(updater, "_git_strict",
                        lambda r, *a, **k: ("b" * 40 if a[0] == "rev-parse"
                                            else ""))

    def fake_run(cmd, **k):
        class R:
            returncode = 1
            stdout = ""
            stderr = "pip exploded"
        return R()

    monkeypatch.setattr(updater.subprocess, "run", fake_run)
    with pytest.raises(updater.UpdateError, match="pip reinstall failed"):
        updater.apply_update(ToasterLoopConfig())


def test_format_status_shapes():
    s = updater.format_status({"available": True, "current_version": "0.1.0",
                               "remote_version": "0.2.0", "install": "git",
                               "repo": "o/r", "url": "u"})
    assert "AVAILABLE" in s and "0.2.0" in s
    s = updater.format_status({"available": False, "current_version": "0.1.0",
                               "remote_version": "0.1.0", "install": "package",
                               "repo": "o/r"})
    assert "up to date" in s


# ------------------------------------------------------- config round-trip

def test_update_section_loads(tmp_path, monkeypatch):
    monkeypatch.setenv("TOASTER_LOOP_CONFIG", str(tmp_path / "c.yaml"))
    (tmp_path / "c.yaml").write_text(
        "update:\n  enabled: false\n  repo: me/fork\n  interval_hours: 6\n")
    from toaster_loop.core.config import load_config

    cfg = load_config()
    assert cfg.update == {"enabled": False, "repo": "me/fork",
                          "interval_hours": 6}


# --------------------------------------------------------- dashboard routes

@pytest.fixture()
def app(home):
    from toaster_loop.dashboard.app import DashboardApp

    a = DashboardApp(ToasterLoopConfig(), home=home)
    yield a


def test_update_routes_registered(app):
    paths = {(m, p) for (m, p, _) in app.ROUTES}
    assert ("GET", "/api/update") in paths
    assert ("POST", "/api/update/check") in paths
    assert ("POST", "/api/update/apply") in paths


def test_update_get_reads_cache(app, home, monkeypatch):
    def boom(cfg):  # must not be called: fresh cache answers
        raise AssertionError("network on GET /api/update")

    monkeypatch.setattr(updater, "_run_check", boom)
    _cache(home, available=True, remote_version="9.9.9")
    r = app.handle("GET", "/api/update")
    assert r.status == 200
    assert r.body["available"] is True and r.body["remote_version"] == "9.9.9"


def test_update_check_forces(monkeypatch, app):
    monkeypatch.setattr(updater, "_run_check",
                        lambda cfg: {"available": False,
                                     "current_version": "0.1.0"})
    r = app.handle("POST", "/api/update/check", body={})
    assert r.status == 200 and r.body["available"] is False


def test_update_apply_single_flight(app, monkeypatch):
    import threading

    gate = threading.Event()
    monkeypatch.setattr(
        updater, "apply_update",
        lambda cfg, log_fn=None: gate.wait(5) or {"changed": True})
    r = app.handle("POST", "/api/update/apply", body={})
    assert r.status == 200 and r.body["state"] == "running"
    r2 = app.handle("POST", "/api/update/apply", body={})
    assert r2.status == 409 and r2.body["error"]["code"] == "already_running"
    gate.set()

    # Status endpoint reflects the run to completion.
    for _ in range(100):
        st = app.handle("GET", "/api/update/apply")
        if st.body["state"] != "running":
            break
        import time as _t
        _t.sleep(0.02)
    assert st.body["state"] == "done"


def test_update_apply_error_state(app, monkeypatch):
    def boom(cfg, log_fn=None):
        raise updater.UpdateError("dirty worktree")

    monkeypatch.setattr(updater, "apply_update", boom)
    r = app.handle("POST", "/api/update/apply", body={})
    assert r.status == 200
    for _ in range(100):
        st = app.handle("GET", "/api/update/apply")
        if st.body["state"] != "running":
            break
        import time as _t
        _t.sleep(0.02)
    assert st.body["state"] == "error"
    assert "dirty worktree" in (st.body["error"] or "")


def test_config_put_accepts_update_section(app, home, monkeypatch):
    cfg_file = home / "toaster_loop.yaml"
    monkeypatch.setenv("TOASTER_LOOP_CONFIG", str(cfg_file))
    r = app.handle("PUT", "/api/config", body={
        "section": "update", "patch": {"enabled": False}})
    assert r.status == 200
    import yaml

    data = yaml.safe_load(cfg_file.read_text()) or {}
    assert data["update"]["enabled"] is False


# ------------------------------------------------------------------ CLI

def test_cli_upgrade_check_exit_codes(monkeypatch, tmp_path, capsys):
    from toaster_loop.cli import main

    monkeypatch.setenv("TOASTER_LOOP_HOME", str(tmp_path))
    monkeypatch.setattr(updater, "_run_check",
                        lambda cfg: {"available": True,
                                     "current_version": "0.1.0",
                                     "remote_version": "0.2.0",
                                     "checked_at": None, "install": "git",
                                     "repo": "o/r", "url": "u",
                                     "error": None})
    assert main(["upgrade", "--check"]) == 10   # EXIT_UPDATE_AVAILABLE
    out = capsys.readouterr().out
    assert "AVAILABLE" in out

    monkeypatch.setattr(updater, "_run_check",
                        lambda cfg: {"available": False,
                                     "current_version": "0.1.0",
                                     "remote_version": "0.1.0",
                                     "checked_at": None, "install": "git",
                                     "repo": "o/r", "url": "u",
                                     "error": None})
    assert main(["upgrade", "--check"]) == 0


def test_cli_upgrade_parser_exists():
    from toaster_loop.cli import build_parser

    args = build_parser().parse_args(["upgrade", "--check"])
    assert args.command == "upgrade" and args.check is True

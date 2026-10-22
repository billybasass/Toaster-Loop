"""Settings-persistence regression tests (PHASE 0).

Bug: the dashboard wrote YAML to <SABLE_HOME>/sable.yaml (or
$SABLE_CONFIG) but load_config() only read a file when config_path or
SABLE_CONFIG was set — so profiles/run_settings/sampling/indexing saved
through the UI silently vanished on the next startup.

These tests pin the fixed precedence:
  explicit config_path arg > SABLE_CONFIG env > <SABLE_HOME>/sable.yaml
  (if it exists) > built-in defaults.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sable.core.config import (  # noqa: E402
    DEFAULT_CONTEXT_LENGTH,
    load_config,
    resolve_config_path,
)
from sable.core.config import SableConfig  # noqa: E402
from sable.dashboard.app import DashboardApp  # noqa: E402


@pytest.fixture()
def home(tmp_path, monkeypatch):
    """tmp SABLE_HOME with no config file and no SABLE_CONFIG env."""
    h = tmp_path / "uf-home"
    monkeypatch.setenv("SABLE_HOME", str(h))
    monkeypatch.delenv("SABLE_CONFIG", raising=False)
    return h


def _write_yaml(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


# ------------------------------------------------------- regression (the bug)

def test_dashboard_saved_profile_persists_across_restart(home, tmp_path):
    """PUT /api/profiles writes <home>/sable.yaml; a fresh load_config()
    with only SABLE_HOME set must see the profile (no config_path arg,
    no SABLE_CONFIG env)."""
    app = DashboardApp(SableConfig(), home=home)
    r = app.handle(
        "PUT", "/api/profiles/coder",
        body={"system_prompt": "you ship code", "tools": ["read_file"]},
    )
    assert r.status == 200, r.body
    r = app.handle("PUT", "/api/profiles/default", body={"default": "coder"})
    assert r.status == 200, r.body

    assert (home / "sable.yaml").is_file()  # written where the fix reads

    cfg = load_config()  # fresh process equivalent: no args, env-only home
    names = {p["name"] for p in cfg.profiles}
    assert "coder" in names
    coder = next(p for p in cfg.profiles if p["name"] == "coder")
    assert coder["system_prompt"] == "you ship code"
    assert coder["tools"] == ["read_file"]
    assert cfg.default_profile == "coder"


def test_dashboard_saved_settings_sections_persist(home):
    """run_settings / sampling / indexing saved via PUT /api/config survive a
    fresh load_config() (same regression, other settings the user reported).

    Phase 5: `top_k` is absorbed by the indexer normalization (internal
    constant now) — persistence of the section itself is what's asserted."""
    app = DashboardApp(SableConfig(), home=home)
    for section, patch in (
        ("run_settings", {"flash_attn": True, "parallel": 4}),
        ("sampling", {"temperature": 0.42, "top_k": 40}),
        ("indexing", {"enabled": True, "output_dim": 512}),
    ):
        r = app.handle("PUT", "/api/config", body={"section": section,
                                                   "patch": patch})
        assert r.status == 200, r.body

    cfg = load_config()
    assert cfg.run_settings.get("flash_attn") is True
    assert cfg.run_settings.get("parallel") == 4
    assert cfg.sampling.get("temperature") == pytest.approx(0.42)
    assert cfg.indexing.get("enabled") is True
    assert cfg.indexing.get("output_dim") == 512


# ---------------------------------------------------------------- precedence

def test_precedence_explicit_arg_wins(home, tmp_path, monkeypatch):
    arg_file = _write_yaml(tmp_path / "arg.yaml", {"model_name": "from-arg"})
    env_file = _write_yaml(tmp_path / "env.yaml", {"model_name": "from-env"})
    _write_yaml(home / "sable.yaml", {"model_name": "from-home"})
    monkeypatch.setenv("SABLE_CONFIG", str(env_file))

    assert load_config(str(arg_file)).model_name == "from-arg"


def test_precedence_env_wins_over_home_fallback(home, tmp_path, monkeypatch):
    env_file = _write_yaml(tmp_path / "env.yaml", {"model_name": "from-env"})
    _write_yaml(home / "sable.yaml", {"model_name": "from-home"})
    monkeypatch.setenv("SABLE_CONFIG", str(env_file))

    assert load_config().model_name == "from-env"


def test_precedence_home_fallback_used_without_env(home):
    _write_yaml(home / "sable.yaml", {"model_name": "from-home"})

    assert load_config().model_name == "from-home"
    assert resolve_config_path() == home / "sable.yaml"


def test_resolve_config_path_precedence_order(home, tmp_path, monkeypatch):
    arg_file = _write_yaml(tmp_path / "arg.yaml", {})
    env_file = _write_yaml(tmp_path / "env.yaml", {})
    _write_yaml(home / "sable.yaml", {})
    monkeypatch.setenv("SABLE_CONFIG", str(env_file))

    assert resolve_config_path(str(arg_file)) == arg_file
    assert resolve_config_path() == env_file
    monkeypatch.delenv("SABLE_CONFIG")
    assert resolve_config_path() == home / "sable.yaml"


# ------------------------------------------------------------------ fallbacks

def test_missing_home_file_falls_back_to_defaults(home):
    """No file anywhere => built-in defaults, silently (no error)."""
    cfg = load_config()
    assert cfg.context_length == DEFAULT_CONTEXT_LENGTH
    assert cfg.profiles == []
    assert cfg.default_profile is None
    assert cfg.run_settings == {}
    assert cfg.sampling == {}
    assert cfg.indexing == {}
    assert resolve_config_path() is None


def test_env_config_path_missing_is_not_fatal(home, tmp_path, monkeypatch):
    """SABLE_CONFIG pointing at a missing file keeps prior behavior:
    defaults, no exception (only an explicit config_path arg hard-errors)."""
    monkeypatch.setenv("SABLE_CONFIG", str(tmp_path / "nope.yaml"))
    cfg = load_config()
    assert cfg.context_length == DEFAULT_CONTEXT_LENGTH


def test_explicit_missing_config_path_still_raises(home):
    with pytest.raises(FileNotFoundError):
        load_config(str(home / "does-not-exist.yaml"))


# ------------------------------------------------- unknown top-level keys (#12)

def test_load_config_warns_on_unknown_top_level_key(home, caplog):
    """A typo'd knob used to vanish silently; load_config now warns like
    TrainingConfig.from_yaml does, naming every dropped key."""
    import logging

    _write_yaml(home / "sable.yaml",
                {"contex_length": 4096, "trainging": {"eval_seed": 1}})
    with caplog.at_level(logging.WARNING, logger="sable.config"):
        cfg = load_config()
    msgs = [r.getMessage() for r in caplog.records
            if r.levelno >= logging.WARNING]
    joined = "\n".join(msgs)
    assert "contex_length" in joined
    assert "trainging" in joined
    # the typo'd knobs are still dropped (defaults apply)
    assert cfg.context_length == DEFAULT_CONTEXT_LENGTH


def test_load_config_silent_on_known_keys(home, caplog):
    import logging

    _write_yaml(home / "sable.yaml", {
        "context_length": 8192, "model_name": "m", "recording": {"enabled": True},
        "delegation": {"timeout_seconds": 60}, "training": {"eval_seed": 3},
        "agent": {"guardrails": True}, "tools": {"lint_on_write": False},
    })
    with caplog.at_level(logging.WARNING, logger="sable.config"):
        cfg = load_config()
    assert not [r for r in caplog.records
                if r.levelno >= logging.WARNING and "unknown key" in r.message]
    assert cfg.context_length == 8192


def test_env_var_overrides_beat_home_fallback_file(home):
    """SABLE_* env field overrides must still win over the fallback file."""
    _write_yaml(home / "sable.yaml", {"context_length": 1234})
    import os

    old = os.environ.get("SABLE_CONTEXT")
    os.environ["SABLE_CONTEXT"] = "9999"
    try:
        assert load_config().context_length == 9999
    finally:
        if old is None:
            del os.environ["SABLE_CONTEXT"]
        else:
            os.environ["SABLE_CONTEXT"] = old

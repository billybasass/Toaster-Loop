"""WP12 plans: config registration + strict-unknown validation."""

from __future__ import annotations

import pytest
import yaml

from sable.core.config import SableConfig, load_config


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "cfg-home"
    monkeypatch.setenv("SABLE_HOME", str(h))
    return h


def _write(home, data):
    p = home / "config.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(data))
    return p


def test_default_config_has_empty_plans_section(home):
    cfg = load_config(_write(home, {"home": str(home)}))
    assert cfg.plans == {}          # OFF by default: zero behavior change


def test_full_plans_section_loads(home):
    cfg = load_config(_write(home, {
        "home": str(home),
        "plans": {"enabled": True, "inject_summary": False,
                  "max_inject_chars": 500,
                  "loop": {"require_verification": False, "max_tasks": 3,
                           "max_consecutive_failures": 2}},
    }))
    assert cfg.plans["enabled"] is True
    assert cfg.plans["loop"]["max_tasks"] == 3


def test_plans_keys_registered_in_strict_mode(home, monkeypatch):
    """Every documented plans key loads IDENTICALLY under strict unknowns."""
    monkeypatch.setenv("SABLE_CONFIG_STRICT", "1")
    data = {"home": str(home),
            "plans": {"enabled": True, "inject_summary": True,
                      "max_inject_chars": 800,
                      "loop": {"require_verification": True,
                               "max_tasks": 10,
                               "max_consecutive_failures": 1}}}
    cfg = load_config(_write(home, data))   # must NOT raise
    assert cfg.plans == data["plans"]


def test_unknown_plans_subkey_flagged_in_strict_mode(home, monkeypatch):
    from sable.core.config import ConfigError

    monkeypatch.setenv("SABLE_CONFIG_STRICT", "1")
    with pytest.raises(ConfigError) as e:
        load_config(_write(home, {"home": str(home),
                                  "plans": {"enabled": True,
                                            "bogus_knob": 1}}))
    assert "bogus_knob" in str(e.value)


def test_mistyped_plans_scalar_reported(home, monkeypatch):
    from sable.core.config import ConfigError

    monkeypatch.setenv("SABLE_CONFIG_STRICT", "1")
    with pytest.raises(ConfigError):
        load_config(_write(home, {"home": str(home),
                                  "plans": {"enabled": "yes-please"}}))


def test_sable_config_dataclass_default():
    assert SableConfig().plans == {}

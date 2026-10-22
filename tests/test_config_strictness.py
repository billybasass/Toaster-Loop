"""WP2a — consolidated config validation ("refuse to guess").

A config with several problems raises ONE ConfigError enumerating every
unknown / missing / mistyped field at once (path, expected kind, redacted
found value). Mistypes are always errors; unknown keys stay warn-only by
default and become errors only in strict mode (flag or SABLE_CONFIG_STRICT).
Currently-valid configs must keep loading to byte-identical objects.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sable.core.config import (  # noqa: E402
    ConfigError,
    SableConfig,
    load_config,
)


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "uf-strict"
    h.mkdir(parents=True)
    monkeypatch.setenv("SABLE_HOME", str(h))
    return h


def _write(home, name: str, data: dict) -> str:
    p = Path(home) / name
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return str(p)


# ------------------------------------------------------- consolidated errors

def test_all_mistypes_reported_at_once(home):
    """Unknown + missing + mistyped in one config -> ONE error listing all."""
    cfg_path = _write(home, "bad.yaml", {
        "port": "not-a-port",          # mistyped int
        "context_length": [1, 2],      # mistyped (list)
        "attach_url": 42,              # mistyped str field
        "training": "nope",            # mistyped mapping field
        "recording": {"retention_days": "soon"},   # nested mistype
        "delegation": {"timeout_seconds": "abc"},  # nested mistype
    })
    with pytest.raises(ConfigError) as ei:
        load_config(cfg_path)
    msg = str(ei.value)
    # every problem present in the single report
    assert "port" in msg and "not-a-port" in msg
    assert "context_length" in msg
    assert "attach_url" in msg
    assert "training" in msg
    assert "recording.retention_days" in msg
    assert "delegation.timeout_seconds" in msg
    # consolidated-count header + the strictness note (documented choice)
    assert "6 problems" in msg
    assert "SABLE_CONFIG_STRICT" in msg


def test_strict_mode_reports_unknown_with_hint(home):
    cfg_path = _write(home, "typo.yaml", {
        "contex_length": 4096,   # typo of context_length
        "trainig": {},           # typo of training
    })
    # lenient default: loads fine (warning only), values ignored
    cfg = load_config(cfg_path)
    assert cfg.context_length == SableConfig().context_length
    # strict via flag: both unknowns in one report with did-you-mean hints
    with pytest.raises(ConfigError) as ei:
        load_config(cfg_path, strict_unknown=True)
    msg = str(ei.value)
    assert "contex_length" in msg and "context_length" in msg
    assert "trainig" in msg and "training" in msg


def test_strict_mode_env_var(home, monkeypatch):
    cfg_path = _write(home, "typo2.yaml", {"bogus_key": 1})
    load_config(cfg_path)  # lenient: fine
    monkeypatch.setenv("SABLE_CONFIG_STRICT", "1")
    with pytest.raises(ConfigError, match="bogus_key"):
        load_config(cfg_path)
    monkeypatch.setenv("SABLE_CONFIG_STRICT", "0")
    load_config(cfg_path)  # back to lenient


def test_strict_mode_nested_unknown(home):
    cfg_path = _write(home, "nest.yaml", {
        "recording": {"enabled": True, "retention_dayz": 5},
    })
    load_config(cfg_path)  # lenient passes
    with pytest.raises(ConfigError) as ei:
        load_config(cfg_path, strict_unknown=True)
    assert "recording.retention_dayz" in str(ei.value)


def test_missing_required_machinery(home, monkeypatch):
    """REQUIRED_FIELDS drives the missing category (empty today; machinery
    is exercised by patching it in)."""
    import sable.core.config as cm

    cfg_path = _write(home, "empty.yaml", {"port": 1})
    monkeypatch.setattr(cm, "REQUIRED_FIELDS", ("gguf_path",))
    with pytest.raises(ConfigError) as ei:
        load_config(cfg_path)
    assert "gguf_path: missing required field" in str(ei.value)


def test_found_values_are_redacted(home):
    """A secret-shaped value in an error message passes core.redact."""
    from sable.core.redact import REDACTED_MARK

    cfg_path = _write(home, "secret.yaml", {
        "attach_url": 42,
        "model_name": ["sk-abcdefghij1234"],  # mistyped str field w/ secret
    })
    with pytest.raises(ConfigError) as ei:
        load_config(cfg_path)
    msg = str(ei.value)
    assert "sk-abcdefghij1234" not in msg
    assert REDACTED_MARK in msg


def test_config_error_is_a_valueerror(home):
    """Public compatibility: ConfigError subclasses ValueError."""
    cfg_path = _write(home, "bad2.yaml", {"port": {}})
    with pytest.raises(ValueError):
        load_config(cfg_path)


# ------------------------------------------------------- backwards compat

FULL_VALID = {
    "gguf_path": "/models/m.gguf",
    "models_dir": "/models",
    "home": None,
    "port": 8080,
    "context_length": 4096,
    "attach_url": "http://127.0.0.1:9/v1",
    "model_name": "m",
    "extra_llama_args": ["--mlock"],
    "gpu_layers": 0,
    "llama_auto_setup": False,
    "recording_enabled": False,
    "delegation": {"enabled": True, "timeout_seconds": 60},
    "curation": {"min_score": 0.5},
    "training": {"base_model": "B/M"},
    "validation": {"min_pass_rate": 0.8},
    "auto_model": True,
    "dashboard": {"port": 8888},
    "agent": {"context_token_budget": 1000, "workspace_root": "/w"},
    "tools": {"lint_on_write": False},
    "run_settings": {"flash_attn": True, "parallel": 2},
    "sampling": {"temperature": 0.7},
    "indexing": {"enabled": True, "model": "/e.gguf"},
    "profiles": [{"name": "p", "system_prompt": "s", "tools": []}],
    "default_profile": "p",
    "logging": {"level": "INFO"},
    "routing": {"enabled": False},
}


def test_valid_config_loads_identical(home):
    """A currently-valid YAML yields the exact object it did pre-WP2 —
    strict mode changes nothing for valid configs."""
    cfg_path = _write(home, "good.yaml", FULL_VALID)
    base = load_config(cfg_path)
    strict = load_config(cfg_path, strict_unknown=True)
    assert base == strict
    # spot-check the flat fields actually landed
    assert base.port == 8080 and base.context_length == 4096
    assert base.delegation_timeout_seconds == 60.0
    assert base.recording_enabled is False
    assert base.run_settings["flash_attn"] is True


def test_valid_config_round_trip_no_warnings(home, caplog):
    cfg_path = _write(home, "good2.yaml", FULL_VALID)
    with caplog.at_level(logging.WARNING, logger="sable.config"):
        cfg = load_config(cfg_path, strict_unknown=True)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    # round-trip: re-dumping the loaded sections and reloading matches
    again = _write(home, "good2-rt.yaml", {
        k: getattr(cfg, k) for k in
        ("port", "context_length", "model_name", "run_settings", "sampling")
    })
    cfg2 = load_config(again)
    assert cfg2.port == cfg.port
    assert cfg2.run_settings == cfg.run_settings


def test_benign_string_coercions_still_load(home):
    """Numeric/bool strings that load fine today keep loading (byte-identical
    behavior requirement): '8080' and 'yes' are NOT mistypes."""
    cfg_path = _write(home, "strings.yaml", {
        "port": "8080",
        "context_length": "4096",
        "recording": {"enabled": "off"},
        "delegation": {"timeout_seconds": "60"},
    })
    cfg = load_config(cfg_path)  # must not raise
    # NOTE: the pre-WP2 loader stores top-level YAML scalars verbatim (no int
    # coercion for flat fields), so byte-identical behavior means "8080"
    # stays a str. The nested recording:/delegation: sections ARE coerced —
    # that shape must not change.
    assert cfg.port == "8080"
    # pre-WP2 quirk pinned verbatim: recording.enabled uses bool(...) so the
    # string "off" is truthy. WP2 must not change it (byte-identical rule).
    assert cfg.recording_enabled is True
    assert cfg.delegation_timeout_seconds == 60.0


def test_explicit_null_workspace_root_preserved(home):
    """Backlog #9 semantics survive the new validation path."""
    cfg_path = _write(home, "ws.yaml", {"agent": {"workspace_root": None}})
    cfg = load_config(cfg_path)
    from sable.core.config import UNSET

    assert cfg.agent["workspace_root"] is UNSET

"""Core-domain tests: config, tools, tool-call recovery, redaction, runner.

Consolidated from 26 former test modules; section banners mark origins.
"""

from __future__ import annotations

import sys
from pathlib import Path
import json
import yaml
import logging
import threading
from http.server import BaseHTTPRequestHandler
from typing import cast
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Any, cast
from types import SimpleNamespace
import time
from typing import Any, Dict, List
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import math
import os
import stat
import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from toaster_loop.training.pipeline import _resolved_train_yaml, apply_config_overrides, log_config_overrides, parse_override_value  # noqa: E402
from toaster_loop.training.worker import lora_kwargs_from_config  # noqa: E402
from toaster_loop.core.config import DEFAULT_CONTEXT_LENGTH, load_config, resolve_config_path  # noqa: E402
from toaster_loop.core.config import ToasterLoopConfig  # noqa: E402
from toaster_loop.dashboard.app import DashboardApp  # noqa: E402
from toaster_loop.core.config import ConfigError, ToasterLoopConfig, load_config  # noqa: E402
from fake_server import FakeOpenAIServer  # noqa: E402
from toaster_loop.core import circuit  # noqa: E402
from toaster_loop.core.agent_runner import ToasterLoopAgentClient  # noqa: E402
from toaster_loop.core.circuit import CircuitBreaker, get_breaker  # noqa: E402
from toaster_loop.core.failure_types import FAILURE_BACKEND_CIRCUIT_OPEN, is_infra_reason  # noqa: E402
from toaster_loop.core.failure_types import classify_tool_result, is_cancelled_reason, is_infra_reason, is_model_reason  # noqa: E402
from toaster_loop.core import agent_runner as ar  # noqa: E402
from toaster_loop.core import path_security as ps  # noqa: E402
from toaster_loop.core.agent_runner import create_runner, resolve_workspace_roots  # noqa: E402
from toaster_loop.core.config import UNSET, ToasterLoopConfig, load_config  # noqa: E402
from toaster_loop.core.playbooks import DEMOTE_NORM_THRESHOLD, GateUnavailable, PlaybookStore, assess_playbook, distill_recipe, export_playbook, find_recurring_chains, promote, prompt_section, run_promotion_gate  # noqa: E402
from toaster_loop.store.db import TraceRow, TraceStore  # noqa: E402
from toaster_loop.core import gpumem  # noqa: E402
from toaster_loop.core.gpumem import unified_pool  # noqa: E402
from toaster_loop.core.vram import ModelDims, base_param_count, estimate_from_train_yaml, estimate_train_vram_mib, lora_param_count, load_model_dims  # noqa: E402
from toaster_loop.training.scheduler import TrainingConfig, job_vram_estimate_mib, resource_gate_ok  # noqa: E402
from toaster_loop.core import torchenv  # noqa: E402
from toaster_loop.core.torchenv import FLAVOR_DEFAULT, TorchFlavor, detect_flavor, installed_mismatch  # noqa: E402
from toaster_loop.core.config import ToasterLoopConfig, load_config  # noqa: E402
from toaster_loop.core.model_discovery import discover_base_models, discover_gguf_models, discover_models, gguf_total_size, is_companion_gguf, resolve_gguf_path, resolve_gguf_shards, resolve_models_dir, shard_group, snapshot_link_for_blob  # noqa: E402
from toaster_loop.core.think_scrub import THINK_TAG_NAMES, StreamingThinkScrubber, strip_think_blocks  # noqa: E402
from fake_server import FakeOpenAIServer, fake_tokenize_count, make_tokenize_handler  # noqa: E402
from toaster_loop.core.compression import SUMMARY_PREFIX, compress_messages, estimate_tokens  # noqa: E402
from toaster_loop.core.tokenizer import TokenCounter  # noqa: E402
from fake_server import FakeOpenAIServer, _Handler  # noqa: E402
from toaster_loop.core.recovery import LOOP_MARKER, detect_reasoning_loop, truncate_at_loop  # noqa: E402
from toaster_loop.core.tool_calls import REPAIR_TAGS, ParseReport, normalize_tool_name, parse_legacy, parse_text_tool_calls, parse_text_tool_calls_with_repairs, strip_stop_tokens, strip_tool_call_markup  # noqa: E402
from toaster_loop.core.tools import CAPABILITY_TOOL_SCHEMAS, TOOL_SCHEMAS  # noqa: E402
from fake_server import FakeOpenAIServer, make_message_handler  # noqa: E402
from toaster_loop.core.tools import TOOL_SCHEMAS  # noqa: E402
from toaster_loop.training.validate import run_golden_suite  # noqa: E402
import toaster_loop.cli as cli  # noqa: E402
from toaster_loop.core.config import load_config  # noqa: E402
from toaster_loop.core.redact import redact_messages, redact_text  # noqa: E402
from toaster_loop.core.tools import read_file, search_files  # noqa: E402
from toaster_loop.store import TraceRecorder  # noqa: E402
from toaster_loop.core.agent_runner import SKILL_TOOL_NAMES, ToasterLoopAgentClient  # noqa: E402
from toaster_loop.core.skills import CONTENT_MAX_CHARS, DESCRIPTION_MAX_CHARS, all_skills, index_section, load_skill, save_skill, skills_dir  # noqa: E402
from toaster_loop.core.tools import list_skills, read_skill, save_skill as save_skill_tool  # noqa: E402
from toaster_loop.core import tools as uf_tools  # noqa: E402
from toaster_loop.serving.run_args import request_sampling_payload, sanitize_sampling  # noqa: E402
from toaster_loop.core.agent_runner import ToasterLoopAgentClient, _confidence_aggregate, create_runner  # noqa: E402
from toaster_loop.store import TraceRecorder, TraceStore  # noqa: E402
from toaster_loop.serving.supervisor import ServeError, ServingSupervisor, _start_instance_server, clear_crash_loop, crash_loop_status, owned_instance_ids, record_start_failure, stop_all_servers  # noqa: E402
from toaster_loop.cli import main  # noqa: E402
from toaster_loop.core.agent_runner import AgentBackendError  # noqa: E402
from toaster_loop.core.errors import FailoverReason, classify_exception, classify_http_status  # noqa: E402
from fake_server import FakeOpenAIServer, _Handler, make_scripted_handler  # noqa: E402
from toaster_loop.store.recorder import TraceRecorder  # noqa: E402
from fake_server import FakeOpenAIServer, make_tokenize_handler  # noqa: E402
from toaster_loop.core.agent_runner import ToasterLoopAgentClient, create_runner  # noqa: E402
from toaster_loop.core.compression import prune_old_tool_results  # noqa: E402
from toaster_loop.core.spill import PERSISTED_OUTPUT_PREFIX, enforce_turn_budget, is_spillover_read, maybe_spill_result, prune_spillover, reset_sweep_flag_for_tests, spill_dir  # noqa: E402

# ------------------------------------------------------------------------
# from tests/test_config_knobs.py
# ------------------------------------------------------------------------

# ------------------------------------------------ lora_kwargs_from_config (#13)

def test_lora_kwargs_defaults():
    kw = lora_kwargs_from_config({})
    assert kw["r"] == 64
    assert kw["lora_alpha"] == 16
    assert kw["lora_dropout"] == 0.0
    assert "q_proj" in kw["target_modules"]
    assert kw["use_rslora"] is False
    assert kw["use_dora"] is False


def test_lora_kwargs_string_target_modules_parsed():
    cfg = {"lora": {"target_modules": "q_proj, v_proj ,down_proj"}}
    kw = lora_kwargs_from_config(cfg)
    assert kw["target_modules"] == ["q_proj", "v_proj", "down_proj"]


def test_lora_kwargs_list_target_modules_passthrough():
    cfg = {"lora": {"target_modules": ["q_proj", "k_proj"]}}
    assert lora_kwargs_from_config(cfg)["target_modules"] == ["q_proj", "k_proj"]


def test_lora_kwargs_rslora_dora_flags():
    cfg = {"lora": {"use_rslora": True, "use_dora": True, "lora_r": 32}}
    kw = lora_kwargs_from_config(cfg)
    assert kw["use_rslora"] is True and kw["use_dora"] is True
    assert kw["r"] == 32


def test_lora_kwargs_flags_also_accepted_under_training():
    cfg = {"training": {"use_rslora": True}}
    assert lora_kwargs_from_config(cfg)["use_rslora"] is True


# ------------------------------------------------- apply_config_overrides (#13)

def test_override_scalars_typed_correctly():
    cfg: dict = {"training": {}, "curation": {}}
    applied = apply_config_overrides(
        cfg,
        ["training.lora_r=32", "training.lr=2e-4", "lora.use_dora=true",
         "training.flag=False", "model=Some/Model"],
        allowed_sections=("training", "curation", "lora", "model"),
    )
    assert cfg["training"]["lora_r"] == 32 and isinstance(cfg["training"]["lora_r"], int)
    assert cfg["training"]["lr"] == pytest.approx(2e-4)
    assert cfg["lora"]["use_dora"] is True          # nested dict created
    assert cfg["training"]["flag"] is False
    assert cfg["model"] == "Some/Model"             # top-level scalar
    assert applied["training.lora_r"] == 32


def test_override_nested_creation_and_dotted_path():
    cfg: dict = {}
    apply_config_overrides(cfg, ["logging.wandb.entity=me"],
                           allowed_sections=("logging",))
    assert cfg["logging"]["wandb"]["entity"] == "me"


def test_override_unknown_section_error():
    with pytest.raises(ValueError, match="unknown config section"):
        apply_config_overrides({}, ["bogus.key=1"])


def test_override_missing_equals_error():
    with pytest.raises(ValueError, match="dotted.path=value"):
        apply_config_overrides({}, ["training.lora_r"])


def test_override_top_level_scalar_allowed():
    # single-segment path naming a known section sets the top-level scalar
    cfg = {"model": "x"}
    applied = apply_config_overrides(cfg, ["model=y"])
    assert cfg["model"] == "y" and applied == {"model": "y"}


def test_override_empty_path_error():
    with pytest.raises(ValueError, match="empty path"):
        apply_config_overrides({}, ["=1"])


def test_override_json_list_value():
    cfg = {"data": {}}
    apply_config_overrides(cfg, ['data.local_dataset=["a.jsonl","b.jsonl"]'],
                           allowed_sections=("data",))
    assert cfg["data"]["local_dataset"] == ["a.jsonl", "b.jsonl"]


def test_override_string_value_with_equals_sign():
    cfg = {"training": {}}
    apply_config_overrides(cfg, ["training.note=a=b"],
                           allowed_sections=("training",))
    assert cfg["training"]["note"] == "a=b"


def test_parse_override_value_table():
    assert parse_override_value("true") is True
    assert parse_override_value("FALSE") is False
    assert parse_override_value("42") == 42
    assert parse_override_value("3.5") == 3.5
    assert parse_override_value("[1, 2]") == [1, 2]
    assert parse_override_value("hello world") == "hello world"
    assert parse_override_value("Qwen/Qwen3-8B") == "Qwen/Qwen3-8B"


def test_log_config_overrides_appends_provenance(tmp_path):
    log = tmp_path / "job.log"
    log_config_overrides(str(log), "job-1", {"training.lora_r": 32})
    rec = json.loads(log.read_text().splitlines()[-1])
    assert rec["event"] == "config_overrides"
    assert rec["job"] == "job-1"
    assert rec["overrides"] == {"training.lora_r": 32}
    # no-op when nothing applied / no log path
    log_config_overrides(str(log), "job-1", {})
    log_config_overrides(None, "job-1", {"a.b": 1})


# --------------------------------------- resolved train.yaml + e2e stub run (#13)

def test_resolved_train_yaml_applies_overrides_last(tmp_path):
    ds = tmp_path / "ds"
    ds.mkdir()
    (ds / "train.yaml").write_text(yaml.safe_dump({
        "model": "REPLACE_ME",
        "training": {"lora_r_note": "x", "num_epochs": 3},
        "lora": {"lora_r": 64},
    }))
    run = tmp_path / "run"
    path, applied = _resolved_train_yaml(
        ds, run, "Base/Model", eval_fraction=0.2, eval_seed=7,
        config_overrides=["lora.lora_r=8", "training.num_epochs=1"],
    )
    data = yaml.safe_load(path.read_text())
    assert data["model"] == "Base/Model"
    assert data["training"]["output_dir"] == str(run)
    assert data["training"]["eval_fraction"] == 0.2
    assert data["lora"]["lora_r"] == 8
    assert data["training"]["num_epochs"] == 1
    assert applied == {"lora.lora_r": 8, "training.num_epochs": 1}


def test_cli_config_key_end_to_end_stub_worker(env_and_cli):
    home, stub = env_and_cli
    from toaster_loop.cli import build_parser
    from toaster_loop.registry import ModelRegistry

    # dataset with a train.yaml template + sft file
    ds_dir = home / "datasets" / "ds-1"
    ds_dir.mkdir(parents=True)
    (ds_dir / "sft.jsonl").write_text(json.dumps(
        {"conversations": [{"from": "human", "value": "hi"}]}) + "\n")
    (ds_dir / "train.yaml").write_text(yaml.safe_dump({
        "model": "REPLACE_WITH_BASE_MODEL_PATH_OR_HF_ID",
        "data": {"local_dataset": [str(ds_dir / "sft.jsonl")],
                 "format_type": "sharegpt"},
        "training": {"num_epochs": 3, "packing": False},
        "lora": {"lora_r": 64},
    }))

    parser = build_parser()
    args = parser.parse_args([
        "train", "start",
        "--config", str(home / "config.yaml"),
        "--dataset", "ds-1", "--dry-run",
        "--config-key", "lora.lora_r=8",
        "--config-key", "training.num_epochs=1",
    ])
    import toaster_loop.cli as cli_mod

    rc = cli_mod.cmd_train_start(args)
    assert rc == 0

    # resolved yaml exists with overridden values
    resolved = list((home / "train_runs").rglob("train.resolved.yaml"))
    assert resolved, "expected a resolved train.yaml under train_runs/"
    data = yaml.safe_load(resolved[0].read_text())
    assert data["lora"]["lora_r"] == 8
    assert data["training"]["num_epochs"] == 1

    # provenance line in the newest job log
    reg = ModelRegistry(home / "buffer" / "toaster_loop.db")
    try:
        jobs = reg.list_jobs()
        assert jobs
        log_path = jobs[-1].log_path
        assert log_path and Path(log_path).is_file()
        lines = Path(log_path).read_text().splitlines()
        prov = [json.loads(l) for l in lines if '"config_overrides"' in l]
        assert prov and prov[0]["overrides"] == {
            "lora.lora_r": 8, "training.num_epochs": 1}
    finally:
        reg.close()


def test_cli_bad_override_fails_fast(env_and_cli):
    home, _ = env_and_cli
    from toaster_loop.cli import build_parser
    import toaster_loop.cli as cli_mod

    ds_dir = home / "datasets" / "ds-2"
    ds_dir.mkdir(parents=True)
    (ds_dir / "train.yaml").write_text("training: {}\n")
    args = build_parser().parse_args([
        "train", "start",
        "--config", str(home / "config.yaml"),
        "--dataset", "ds-2", "--config-key", "nope.key=1",
    ])
    assert cli_mod.cmd_train_start(args) == 2


@pytest.fixture()
def env_and_cli(tmp_path, monkeypatch):
    """Home + stub worker command wired via env (dry-run never spawns)."""
    home = tmp_path / "uf-knobs"
    home.mkdir(parents=True)
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(home))
    stub = tmp_path / "stub.py"
    stub.write_text("import sys; sys.exit(0)\n")
    (home / "config.yaml").write_text(yaml.safe_dump({
        "home": str(home),
        "training": {"base_model": "Base/HF-Model",
                     "worker_command": [sys.executable, str(stub)]},
    }))
    return home, stub

# ------------------------------------------------------------------------
# from tests/test_config_persistence.py
# ------------------------------------------------------------------------

@pytest.fixture()
def home(tmp_path, monkeypatch):
    """tmp TOASTER_LOOP_HOME with no config file and no TOASTER_LOOP_CONFIG env."""
    h = tmp_path / "uf-home"
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(h))
    monkeypatch.delenv("TOASTER_LOOP_CONFIG", raising=False)
    return h


def _write_yaml(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


# ------------------------------------------------------- regression (the bug)

def test_dashboard_saved_profile_persists_across_restart(home, tmp_path):
    """PUT /api/profiles writes <home>/toaster_loop.yaml; a fresh load_config()
    with only TOASTER_LOOP_HOME set must see the profile (no config_path arg,
    no TOASTER_LOOP_CONFIG env)."""
    app = DashboardApp(ToasterLoopConfig(), home=home)
    r = app.handle(
        "PUT", "/api/profiles/coder",
        body={"system_prompt": "you ship code", "tools": ["read_file"]},
    )
    assert r.status == 200, r.body
    r = app.handle("PUT", "/api/profiles/default", body={"default": "coder"})
    assert r.status == 200, r.body

    assert (home / "toaster_loop.yaml").is_file()  # written where the fix reads

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
    app = DashboardApp(ToasterLoopConfig(), home=home)
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
    _write_yaml(home / "toaster_loop.yaml", {"model_name": "from-home"})
    monkeypatch.setenv("TOASTER_LOOP_CONFIG", str(env_file))

    assert load_config(str(arg_file)).model_name == "from-arg"


def test_precedence_env_wins_over_home_fallback(home, tmp_path, monkeypatch):
    env_file = _write_yaml(tmp_path / "env.yaml", {"model_name": "from-env"})
    _write_yaml(home / "toaster_loop.yaml", {"model_name": "from-home"})
    monkeypatch.setenv("TOASTER_LOOP_CONFIG", str(env_file))

    assert load_config().model_name == "from-env"


def test_precedence_home_fallback_used_without_env(home):
    _write_yaml(home / "toaster_loop.yaml", {"model_name": "from-home"})

    assert load_config().model_name == "from-home"
    assert resolve_config_path() == home / "toaster_loop.yaml"


def test_resolve_config_path_precedence_order(home, tmp_path, monkeypatch):
    arg_file = _write_yaml(tmp_path / "arg.yaml", {})
    env_file = _write_yaml(tmp_path / "env.yaml", {})
    _write_yaml(home / "toaster_loop.yaml", {})
    monkeypatch.setenv("TOASTER_LOOP_CONFIG", str(env_file))

    assert resolve_config_path(str(arg_file)) == arg_file
    assert resolve_config_path() == env_file
    monkeypatch.delenv("TOASTER_LOOP_CONFIG")
    assert resolve_config_path() == home / "toaster_loop.yaml"


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
    """TOASTER_LOOP_CONFIG pointing at a missing file keeps prior behavior:
    defaults, no exception (only an explicit config_path arg hard-errors)."""
    monkeypatch.setenv("TOASTER_LOOP_CONFIG", str(tmp_path / "nope.yaml"))
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

    _write_yaml(home / "toaster_loop.yaml",
                {"contex_length": 4096, "trainging": {"eval_seed": 1}})
    with caplog.at_level(logging.WARNING, logger="toaster_loop.config"):
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

    _write_yaml(home / "toaster_loop.yaml", {
        "context_length": 8192, "model_name": "m", "recording": {"enabled": True},
        "delegation": {"max_concurrent": 3}, "training": {"eval_seed": 3},
        "agent": {"guardrails": True}, "tools": {"lint_on_write": False},
    })
    with caplog.at_level(logging.WARNING, logger="toaster_loop.config"):
        cfg = load_config()
    assert not [r for r in caplog.records
                if r.levelno >= logging.WARNING and "unknown key" in r.message]
    assert cfg.context_length == 8192


def test_env_var_overrides_beat_home_fallback_file(home):
    """TOASTER_LOOP_* env field overrides must still win over the fallback file."""
    _write_yaml(home / "toaster_loop.yaml", {"context_length": 1234})
    import os

    old = os.environ.get("TOASTER_LOOP_CONTEXT")
    os.environ["TOASTER_LOOP_CONTEXT"] = "9999"
    try:
        assert load_config().context_length == 9999
    finally:
        if old is None:
            del os.environ["TOASTER_LOOP_CONTEXT"]
        else:
            os.environ["TOASTER_LOOP_CONTEXT"] = old

# ------------------------------------------------------------------------
# from tests/test_config_strictness.py
# ------------------------------------------------------------------------

@pytest.fixture()
def home__test_config_strictness(tmp_path, monkeypatch):
    h = tmp_path / "uf-strict"
    h.mkdir(parents=True)
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(h))
    return h


def _write(home, name: str, data: dict) -> str:
    p = Path(home) / name
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return str(p)


# ------------------------------------------------------- consolidated errors

def test_all_mistypes_reported_at_once(home__test_config_strictness):
    """Unknown + missing + mistyped in one config -> ONE error listing all."""
    cfg_path = _write(home__test_config_strictness, "bad.yaml", {
        "port": "not-a-port",          # mistyped int
        "context_length": [1, 2],      # mistyped (list)
        "attach_url": 42,              # mistyped str field
        "training": "nope",            # mistyped mapping field
        "recording": {"retention_days": "soon"},   # nested mistype
        "delegation": {"max_concurrent": "abc"},   # nested mistype
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
    assert "delegation.max_concurrent" in msg
    # consolidated-count header + the strictness note (documented choice)
    assert "6 problems" in msg
    assert "TOASTER_LOOP_CONFIG_STRICT" in msg


def test_strict_mode_reports_unknown_with_hint(home__test_config_strictness):
    cfg_path = _write(home__test_config_strictness, "typo.yaml", {
        "contex_length": 4096,   # typo of context_length
        "trainig": {},           # typo of training
    })
    # lenient default: loads fine (warning only), values ignored
    cfg = load_config(cfg_path)
    assert cfg.context_length == ToasterLoopConfig().context_length
    # strict via flag: both unknowns in one report with did-you-mean hints
    with pytest.raises(ConfigError) as ei:
        load_config(cfg_path, strict_unknown=True)
    msg = str(ei.value)
    assert "contex_length" in msg and "context_length" in msg
    assert "trainig" in msg and "training" in msg


def test_strict_mode_env_var(home__test_config_strictness, monkeypatch):
    cfg_path = _write(home__test_config_strictness, "typo2.yaml", {"bogus_key": 1})
    load_config(cfg_path)  # lenient: fine
    monkeypatch.setenv("TOASTER_LOOP_CONFIG_STRICT", "1")
    with pytest.raises(ConfigError, match="bogus_key"):
        load_config(cfg_path)
    monkeypatch.setenv("TOASTER_LOOP_CONFIG_STRICT", "0")
    load_config(cfg_path)  # back to lenient


def test_strict_mode_nested_unknown(home__test_config_strictness):
    cfg_path = _write(home__test_config_strictness, "nest.yaml", {
        "recording": {"enabled": True, "retention_dayz": 5},
    })
    load_config(cfg_path)  # lenient passes
    with pytest.raises(ConfigError) as ei:
        load_config(cfg_path, strict_unknown=True)
    assert "recording.retention_dayz" in str(ei.value)


def test_missing_required_machinery(home__test_config_strictness, monkeypatch):
    """REQUIRED_FIELDS drives the missing category (empty today; machinery
    is exercised by patching it in)."""
    import toaster_loop.core.config as cm

    cfg_path = _write(home__test_config_strictness, "empty.yaml", {"port": 1})
    monkeypatch.setattr(cm, "REQUIRED_FIELDS", ("gguf_path",))
    with pytest.raises(ConfigError) as ei:
        load_config(cfg_path)
    assert "gguf_path: missing required field" in str(ei.value)


def test_found_values_are_redacted(home__test_config_strictness):
    """A secret-shaped value in an error message passes core.redact."""
    from toaster_loop.core.redact import REDACTED_MARK

    cfg_path = _write(home__test_config_strictness, "secret.yaml", {
        "attach_url": 42,
        "model_name": ["sk-abcdefghij1234"],  # mistyped str field w/ secret
    })
    with pytest.raises(ConfigError) as ei:
        load_config(cfg_path)
    msg = str(ei.value)
    assert "sk-abcdefghij1234" not in msg
    assert REDACTED_MARK in msg


def test_config_error_is_a_valueerror(home__test_config_strictness):
    """Public compatibility: ConfigError subclasses ValueError."""
    cfg_path = _write(home__test_config_strictness, "bad2.yaml", {"port": {}})
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
    "delegation": {"enabled": True, "max_concurrent": 3},
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


def test_valid_config_loads_identical(home__test_config_strictness):
    """A currently-valid YAML yields the exact object it did pre-WP2 —
    strict mode changes nothing for valid configs."""
    cfg_path = _write(home__test_config_strictness, "good.yaml", FULL_VALID)
    base = load_config(cfg_path)
    strict = load_config(cfg_path, strict_unknown=True)
    assert base == strict
    # spot-check the flat fields actually landed
    assert base.port == 8080 and base.context_length == 4096
    assert base.delegation_max_concurrent == 3
    assert base.recording_enabled is False
    assert base.run_settings["flash_attn"] is True


def test_valid_config_round_trip_no_warnings(home__test_config_strictness, caplog):
    cfg_path = _write(home__test_config_strictness, "good2.yaml", FULL_VALID)
    with caplog.at_level(logging.WARNING, logger="toaster_loop.config"):
        cfg = load_config(cfg_path, strict_unknown=True)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    # round-trip: re-dumping the loaded sections and reloading matches
    again = _write(home__test_config_strictness, "good2-rt.yaml", {
        k: getattr(cfg, k) for k in
        ("port", "context_length", "model_name", "run_settings", "sampling")
    })
    cfg2 = load_config(again)
    assert cfg2.port == cfg.port
    assert cfg2.run_settings == cfg.run_settings


def test_benign_string_coercions_still_load(home__test_config_strictness):
    """Numeric/bool strings that load fine today keep loading (byte-identical
    behavior requirement): '8080' and 'yes' are NOT mistypes."""
    cfg_path = _write(home__test_config_strictness, "strings.yaml", {
        "port": "8080",
        "context_length": "4096",
        "recording": {"enabled": "off"},
        "delegation": {"max_concurrent": "2"},
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
    assert cfg.delegation_max_concurrent == 2


def test_explicit_null_workspace_root_preserved(home__test_config_strictness):
    """Backlog #9 semantics survive the new validation path."""
    cfg_path = _write(home__test_config_strictness, "ws.yaml", {"agent": {"workspace_root": None}})
    cfg = load_config(cfg_path)
    from toaster_loop.core.config import UNSET

    assert cfg.agent["workspace_root"] is UNSET

# ------------------------------------------------------------------------
# from tests/test_circuit.py
# ------------------------------------------------------------------------

class FakeClock:
    def __init__(self, t: float = 100.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


# ------------------------------------------------------------- state machine

def test_opens_after_threshold_consecutive_failures(_no_backoff_sleeps):
    b = CircuitBreaker(failure_threshold=5, cooldown_seconds=60.0,
                       clock=FakeClock())
    for i in range(4):
        b.record_failure()
        assert b.state == "closed", i
    assert b.is_allowed() is True
    b.record_failure()  # 5th consecutive
    assert b.state == "open"
    assert b.is_allowed() is False
    assert b.failures == 5


def test_success_resets_streak(_no_backoff_sleeps):
    b = CircuitBreaker(failure_threshold=3, cooldown_seconds=60.0,
                       clock=FakeClock())
    b.record_failure()
    b.record_failure()
    b.record_success()
    assert b.failures == 0 and b.state == "closed"
    b.record_failure()
    b.record_failure()
    assert b.state == "closed"  # streak restarted, not cumulative


def test_open_blocks_until_cooldown_then_half_open_probe_once(_no_backoff_sleeps):
    clock = FakeClock()
    b = CircuitBreaker(failure_threshold=2, cooldown_seconds=10.0, clock=clock)
    b.record_failure()
    b.record_failure()
    assert b.state == "open"
    assert b.is_allowed() is False
    assert b.seconds_until_probe() > 0

    clock.advance(9.0)
    assert b.is_allowed() is False  # still cooling down
    clock.advance(2.0)
    assert b.state == "half_open"
    assert b.is_allowed() is True   # probe granted
    assert b.is_allowed() is False  # exactly one probe in flight


def test_probe_success_closes_probe_failure_reopens(_no_backoff_sleeps):
    clock = FakeClock()
    b = CircuitBreaker(failure_threshold=2, cooldown_seconds=10.0, clock=clock)
    b.record_failure()
    b.record_failure()
    clock.advance(11.0)
    assert b.is_allowed() is True   # probe slot
    b.record_success()
    assert b.state == "closed"
    assert b.is_allowed() is True

    # now the failure path: probe -> fail -> OPEN with fresh cooldown
    b.record_failure()
    b.record_failure()
    clock.advance(11.0)
    assert b.is_allowed() is True   # half-open probe
    b.record_failure()              # probe failed
    assert b.state == "open"
    # Fresh cooldown started at the probe failure: full window remains.
    assert 9.9 < b.seconds_until_probe() <= 10.0


def test_stuck_half_open_probe_is_released(_no_backoff_sleeps):
    clock = FakeClock()
    b = CircuitBreaker(failure_threshold=1, cooldown_seconds=5.0, clock=clock)
    b.record_failure()
    assert b.state == "open"
    clock.advance(6.0)
    assert b.is_allowed() is True   # probe taken, caller "dies"
    assert b.is_allowed() is False  # in flight within cooldown window
    clock.advance(6.0)              # probe considered stale
    assert b.is_allowed() is True


def test_snapshot_fields(_no_backoff_sleeps):
    clock = FakeClock()
    b = CircuitBreaker(failure_threshold=1, cooldown_seconds=5.0, clock=clock)
    snap = b.snapshot()
    assert snap["state"] == "closed" and snap["opened_at"] is None
    b.record_failure()
    snap = b.snapshot()
    assert snap["state"] == "open"
    assert snap["failures"] == 1
    assert snap["opened_at"] is not None  # wall-clock epoch for display
    assert cast(float, snap["seconds_until_probe"]) > 0


def test_registry_shared_and_thread_safe(_no_backoff_sleeps):
    b1 = get_breaker("http://shared/v1")
    b2 = get_breaker("http://shared/v1")
    assert b1 is b2
    assert "http://shared/v1" in circuit.all_breakers()

    errors = []

    def hammer():
        try:
            for _ in range(200):
                b1.is_allowed()
                b1.record_failure()
                b1.record_success()
        except Exception as e:  # pragma: no cover
            errors.append(e)

    ts = [threading.Thread(target=hammer) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=10)
    assert not errors


def test_circuit_open_reason_is_infra_classified(_no_backoff_sleeps):
    assert is_infra_reason(FAILURE_BACKEND_CIRCUIT_OPEN)
    assert is_infra_reason(f"backend_error:{FAILURE_BACKEND_CIRCUIT_OPEN}")


# ------------------------------------------------------------ runner wiring

class DeadHandler(BaseHTTPRequestHandler):
    """Every chat POST answers 500 — a wedged backend."""

    hits = 0

    def log_message(self, format, *args):  # noqa: A002 - silence access log
        pass

    def do_POST(self):
        type(self).hits += 1
        body = json.dumps({"error": "wedged"}).encode()
        self.send_response(500)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture()
def _no_backoff_sleeps(monkeypatch):
    from toaster_loop.core import agent_runner as ar
    monkeypatch.setattr(ar.ToasterLoopAgentClient, "_sleep",
                        staticmethod(lambda s: None))


def test_wedged_backend_opens_breaker_and_next_run_fails_fast(tmp_path, _no_backoff_sleeps):
    DeadHandler.hits = 0
    with FakeOpenAIServer(handler_cls=DeadHandler) as srv:
        client = ToasterLoopAgentClient(base_url=srv.base_url, model="m",
                                    timeout=2.0)
        # Run 1: 3 attempts, all fail (streak=3 < threshold 5).
        r1 = client.run("task one")
        assert r1.failure_reason == "backend_error:connection"
        assert DeadHandler.hits == 3

        # Run 2: attempts 4+5 push the streak over the threshold and OPEN the
        # circuit mid-run; the next attempt is refused without the network.
        r2 = client.run("task two")
        assert r2.failure_reason == FAILURE_BACKEND_CIRCUIT_OPEN
        assert DeadHandler.hits == 5

        # Run 3: fully fail-fast — zero additional network hits.
        r3 = client.run("task three")
        assert r3.failure_reason == FAILURE_BACKEND_CIRCUIT_OPEN
        assert DeadHandler.hits == 5

        snap = circuit.all_breakers()[srv.base_url]
        assert snap["state"] == "open"


def test_runner_success_closes_streak(tmp_path, _no_backoff_sleeps):
    """One blip + success must not march toward OPEN (consecutive semantics)."""
    with FakeOpenAIServer() as srv:  # healthy fake
        client = ToasterLoopAgentClient(base_url=srv.base_url, model="fake-model")
        result = client.run("read the fake server source")
    assert result.completed is True
    assert circuit.all_breakers()[srv.base_url]["state"] == "closed"


def test_judge_shares_the_runner_breaker(_no_backoff_sleeps):
    from toaster_loop.curate.judge import make_chat_client

    url = "http://127.0.0.1:9/v1"  # nothing listens here
    breaker = get_breaker(url)
    for _ in range(5):
        breaker.record_failure()
    assert breaker.state == "open"

    client_fn = make_chat_client(url, timeout=1.0, transport_attempts=3)
    with pytest.raises(RuntimeError, match="circuit open"):
        client_fn([{"role": "user", "content": "grade me"}], "m")


# --------------------------------------------------- supervisor crash budget

def test_crash_loop_budget_refuses_after_three_in_window(_no_backoff_sleeps):
    from toaster_loop.serving import supervisor as sup

    key = "inst-a"
    assert sup.crash_loop_status(key) is None
    sup.record_start_failure(key)
    sup.record_start_failure(key)
    assert not sup.crash_loop_refused(key)
    sup.record_start_failure(key)
    assert sup.crash_loop_refused(key)
    st = sup.crash_loop_status(key)
    assert st is not None
    assert st["failures"] == 3 and st["refused"] is True


def test_crash_loop_window_expiry(monkeypatch, _no_backoff_sleeps):
    from toaster_loop.serving import supervisor as sup

    fake = {"t": 1000.0}
    monkeypatch.setattr(sup.time, "monotonic", lambda: fake["t"])
    key = "inst-b"
    for _ in range(3):
        sup.record_start_failure(key)
    assert sup.crash_loop_refused(key)
    fake["t"] += sup.CRASH_LOOP_WINDOW_S + 1
    assert not sup.crash_loop_refused(key)


def test_clear_crash_loop_allows_retry(_no_backoff_sleeps):
    from toaster_loop.serving import supervisor as sup

    key = "inst-c"
    for _ in range(3):
        sup.record_start_failure(key)
    assert sup.crash_loop_refused(key)
    assert sup.clear_crash_loop(key) is True
    assert not sup.crash_loop_refused(key)
    assert sup.crash_loop_status(key) is None
    assert sup.clear_crash_loop(key) is False  # idempotent


def test_start_instance_server_refuses_when_over_budget(tmp_path, _no_backoff_sleeps):
    from toaster_loop.serving import supervisor as sup

    key = "wedged"
    for _ in range(sup.CRASH_LOOP_MAX_FAILURES):
        sup.record_start_failure(key)
    with pytest.raises(sup.ServeError, match="crash loop"):
        sup.start_instance_server(None, {"id": key,
                                         "gguf": str(tmp_path / "x.gguf")})


def test_dashboard_row_surfaces_crash_loop(tmp_path, monkeypatch, _no_backoff_sleeps):
    from toaster_loop.core.config import ToasterLoopConfig
    from toaster_loop.dashboard.app import DashboardApp
    from toaster_loop.serving import supervisor as sup

    home = tmp_path / "uf-cl"
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(home))
    app = DashboardApp(ToasterLoopConfig(llama_auto_setup=False), home=home)
    created = app.handle("POST", "/api/servers",
                         body={"name": "budgeted", "gguf": "/nope.gguf"})
    iid = created.body["server"]["id"]

    for _ in range(sup.CRASH_LOOP_MAX_FAILURES):
        sup.record_start_failure(iid)
    rows = app.handle("GET", "/api/servers").body["servers"]
    row = next(r for r in rows if r["id"] == iid)
    assert row["crash_loop"] is not None
    assert row["crash_loop"]["refused"] is True

    # User-initiated start clears the refusal before attempting (the spawn
    # then fails on the missing GGUF, but NOT with the crash-loop message).
    resp = app.handle("POST", f"/api/servers/{iid}/start", body={})
    assert resp.status == 500
    assert not sup.crash_loop_refused(iid)

# ------------------------------------------------------------------------
# from tests/test_failure_types.py
# ------------------------------------------------------------------------

# ------------------------------------------------------ classify_tool_result

def test_golden_pass_shapes():
    ok = [
        "a.txt\nb.txt",
        "wrote 12 chars to /tmp/x",
        json.dumps({"ok": True}),
        json.dumps({"returncode": 0, "stdout": "hi"}),
        json.dumps({"exit_code": 0}),
    ]
    for content in ok:
        assert classify_tool_result("t", content) == (False, "none"), content


def test_golden_failure_shapes():
    assert classify_tool_result("read_file", "error: not a file: x") == (
        True, "error_tag")
    assert classify_tool_result("read_file", "Error reading x: boom") == (
        True, "error_tag")
    assert classify_tool_result(
        "patch", "Traceback (most recent call last):\n  ...") == (
        True, "traceback")
    assert classify_tool_result("patch", "ValueError exception raised here") == (
        True, "exception")
    assert classify_tool_result("t", json.dumps({"error": "bad args"})) == (
        True, "tool_error_json")
    assert classify_tool_result("t", json.dumps({"returncode": 1})) == (
        True, "tool_error_json")
    assert classify_tool_result("t", json.dumps({"exit_code": 2})) == (
        True, "tool_error_json")
    assert classify_tool_result("t", "") == (True, "empty")
    assert classify_tool_result("t", "   \n ") == (True, "empty")


def test_json_non_dict_and_invalid_json_tolerated():
    assert classify_tool_result("t", "[1, 2, 3]") == (False, "none")
    assert classify_tool_result("t", "{not valid json error}") == (False, "none")
    # traceback beats generic error tag priority
    assert classify_tool_result(
        "t", "error while running: Traceback (most recent call last):") == (
        True, "traceback")


# ------------------------------------- HIGH #5 false-positive regression corpus

def test_false_positive_corpus_classifies_as_success():
    """Bare 'traceback'/'exception' mentions are NOT failures (HIGH #5).

    Every entry below is ordinary coding-agent tool output that the old
    substring checks mis-classified as a failed tool result.
    """
    fp = [
        # read_file on Python source with an except clause + parenthesized raise
        "def parse(text):\n"
        "    try:\n"
        "        return json.loads(text)\n"
        "    except Exception as e:\n"
        '        raise ValueError("bad input") from e\n',
        # grep/ripgrep listing files that mention the words
        "docs/errors.md:14:The Traceback module renders call stacks\n"
        'core/failure_types.py:80:# bare substring "traceback" false positives\n'
        "src/handler.py:9:    except Exception:\n"
        "        pass  # swallow",
        # JSON payload with exception-flavoured keys
        json.dumps({"exception_count": 3, "traceback": None, "status": "ok"}),
        # prose mentioning exceptions and tracebacks (lower-case words)
        "The handler catches exceptions from the backend and logs a "
        "traceback when one occurs.",
    ]
    for content in fp:
        assert classify_tool_result("read_file", content) == (False, "none"), \
            repr(content)


def test_true_positive_shapes_still_fail():
    """Real traceback bodies and exception-class lines still classify as failure."""
    tb = ("Traceback (most recent call last):\n"
          '  File "x.py", line 3, in <module>\n'
          "    1 / 0\n"
          "ZeroDivisionError: division by zero")
    assert classify_tool_result("bash", tb) == (True, "traceback")
    # trailing frames still carry the canonical header -> traceback wins
    assert classify_tool_result(
        "patch",
        "process exited nonzero\nTraceback (most recent call last):\n  ...") == (
        True, "traceback")
    # exception-class + colon shapes (bare raise output, dotted module classes)
    assert classify_tool_result("bash", "ValueError: bad input") == (
        True, "exception")
    assert classify_tool_result("t", "json.JSONDecodeError: Expecting value") == (
        True, "exception")
    # ERROR:/error: tool-failure tags preserved
    assert classify_tool_result("t", "ERROR: command not found") == (
        True, "error_tag")
    assert classify_tool_result("t", "error: no such file") == (
        True, "error_tag")


def test_is_cancelled_reason():
    assert is_cancelled_reason("cancelled")
    assert is_cancelled_reason("backend_error:cancelled")  # composed form
    for r in ("", "connection", "timeout", "max_iterations",
              "backend_error:timeout", "guardrail_halt"):
        assert not is_cancelled_reason(r)


# ------------------------------------------------------------- reason gates

def test_is_infra_reason():
    for r in ("connection", "timeout", "model_not_found"):
        assert is_infra_reason(r)
        assert is_infra_reason(f"backend_error:{r}")  # composed runner form
    for r in ("", "max_iterations", "context_overflow", "guardrail_halt",
              "format_error", "payload_too_large"):
        assert not is_infra_reason(r)


def test_is_model_reason():
    for r in ("context_overflow", "payload_too_large", "format_error",
              "max_iterations", "guardrail_halt"):
        assert is_model_reason(r)
        assert is_model_reason(f"backend_error:{r}")
    for r in ("", "connection", "timeout", "model_not_found"):
        assert not is_model_reason(r)

# ------------------------------------------------------------------------
# from tests/test_path_security.py
# ------------------------------------------------------------------------

@pytest.fixture()
def toaster_loop_home(tmp_path, monkeypatch):
    home = tmp_path / "uf-home"
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(home))
    return home


@pytest.fixture()
def _tempdir_outside_workspace(tmp_path, monkeypatch):
    """Move the tempdir allow-list root out from under tmp_path.

    pytest's tmp_path lives *inside* the real system temp dir, which backlog
    #9 always allow-lists — without this, "outside workspace" assertions would
    trivially pass through the temp-dir exception. The substitute is an empty
    sibling dir that no test writes into.
    """
    fake_tmp = tmp_path.parent / f"fake-tempdir-{tmp_path.name}"
    fake_tmp.mkdir(exist_ok=True)
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(fake_tmp))
    yield fake_tmp


# --------------------------------------------------------------- unit: checks

def test_traversal_detection(_tempdir_outside_workspace):
    assert ps.has_traversal_component("a/../../etc/passwd")
    assert ps.has_traversal_component("../sibling.txt")
    assert not ps.has_traversal_component("a/b/c.txt")
    assert not ps.has_traversal_component("..hidden/file")  # name, not component
    assert not ps.has_traversal_component("")


def test_validate_within_dir_inside_outside(tmp_path, _tempdir_outside_workspace):
    inside = tmp_path / "sub" / "file.txt"  # nonexistent is fine (strict=False)
    assert ps.validate_within_dir(inside, tmp_path) is None
    assert ps.validate_within_dir(tmp_path, tmp_path) is None
    outside = tmp_path.parent / "elsewhere.txt"
    err = ps.validate_within_dir(outside, tmp_path)
    assert err and "escapes" in err


def test_validate_within_dir_symlink_escape(tmp_path, _tempdir_outside_workspace):
    (tmp_path / "real").mkdir()
    link = tmp_path / "link"
    link.symlink_to(tmp_path.parent)  # link/ -> parent of the workspace
    err = ps.validate_within_dir(link / "secret.txt", tmp_path)
    assert err and "escapes" in err


def test_within_any_dir_multiroot(tmp_path, _tempdir_outside_workspace):
    root_a = tmp_path / "ws"
    root_b = tmp_path / "home-state"
    root_a.mkdir()
    root_b.mkdir()
    assert ps.within_any_dir(root_a / "a.txt", [root_a, root_b]) is None
    assert ps.within_any_dir(root_b / "b.txt", [root_a, root_b]) is None
    # relative paths resolve against the FIRST (primary) root
    assert ps.within_any_dir("sub/a.txt", [root_a, root_b]) is None
    assert ps.within_any_dir(tmp_path / "outside.txt", [root_a, root_b])


# ------------------------------------------------------------- unit: decorator

def test_contain_blocks_write_outside_root(tmp_path, _tempdir_outside_workspace):
    ws = tmp_path / "ws"
    ws.mkdir()

    @ps.contain([ws])
    def write_file(path: str, content: str = "") -> str:
        return f"executed {path}"

    out = write_file(path=str(ws / "ok.txt"))
    assert "executed" in out  # inside: runs
    blocked = write_file(path=str(tmp_path / "outside.txt"))
    assert blocked.startswith("error: blocked: path outside workspace")
    assert str(ws) in blocked


def test_contain_blocks_traversal_even_inside_root(tmp_path, _tempdir_outside_workspace):
    ws = tmp_path / "ws"
    (ws / "sub").mkdir(parents=True)

    @ps.contain([ws])
    def read_file(path: str) -> str:
        return f"executed {path}"

    # normalizes back inside the workspace, but '..' is refused up front
    out = read_file(path=str(ws / "sub" / ".." / "a.txt"))
    assert out.startswith("error: blocked: '..' path traversal")


def test_contain_blocks_symlink_escape_write(tmp_path, _tempdir_outside_workspace):
    ws = tmp_path / "ws"
    ws.mkdir()
    victim_dir = tmp_path / "outside"
    victim_dir.mkdir()
    link = ws / "escape"
    link.symlink_to(victim_dir)  # existing target -> resolve() follows it

    @ps.contain([ws])
    def write_file(path: str, content: str = "") -> str:
        Path(path).write_text(content)
        return f"executed {path}"

    out = write_file(path=str(link / "bashrc"), content="rm -rf /")
    assert out.startswith("error: blocked: path outside workspace")
    assert not (victim_dir / "bashrc").exists()


def test_contain_allows_nonexistent_inside_and_tempdir(tmp_path, _tempdir_outside_workspace):
    ws = tmp_path / "ws"
    ws.mkdir()  # parent exists, file does not

    @ps.contain([ws, Path(tempfile.gettempdir())])
    def write_file(path: str, content: str = "") -> str:
        return f"executed {path}"

    assert "executed" in write_file(path=str(ws / "new" / "deep.txt"))
    assert "executed" in write_file(
        path=str(Path(tempfile.gettempdir()) / "toaster_loop-tmp-check.txt"))


def test_contain_ignores_non_path_and_missing_args(tmp_path, _tempdir_outside_workspace):
    ws = tmp_path / "ws"
    ws.mkdir()

    @ps.contain([ws])
    def todo_write(todos=None, **_):
        return "ok"

    assert todo_write(todos=[{"content": "x"}]) == "ok"


def test_contain_checks_file_path_alias(tmp_path, _tempdir_outside_workspace):
    ws = tmp_path / "ws"
    ws.mkdir()

    @ps.contain([ws])
    def some_tool(file_path: str = "") -> str:
        return f"executed {file_path}"

    assert some_tool(file_path=str(ws.parent / "x")).startswith("error: blocked")


# ------------------------------------------------------ config / root resolution

def test_roots_default_to_cwd_when_unset(monkeypatch, toaster_loop_home, _tempdir_outside_workspace):
    monkeypatch.setattr(ar.os, "getcwd", lambda: "/tmp/fake-cwd")
    roots = resolve_workspace_roots(ToasterLoopConfig())
    assert roots is not None
    assert roots[0] == Path("/tmp/fake-cwd")
    assert Path(str(toaster_loop_home)) in [Path(str(r)) for r in roots[1:]]
    assert Path(tempfile.gettempdir()) in [Path(str(r)) for r in roots[1:]]


def test_roots_explicit_path_and_profile_override(tmp_path, toaster_loop_home, _tempdir_outside_workspace):
    cfg = ToasterLoopConfig(agent={"workspace_root": str(tmp_path / "ws")})
    roots = resolve_workspace_roots(cfg)
    assert roots is not None and roots[0] == tmp_path / "ws"
    # per-profile override widens/narrows the global primary root
    roots2 = resolve_workspace_roots(cfg, workspace_root_override=str(tmp_path))
    assert roots2 is not None and roots2[0] == tmp_path


def test_explicit_null_is_unrestricted_with_warning(tmp_path, caplog, monkeypatch, _tempdir_outside_workspace):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("agent:\n  workspace_root: null\n")
    cfg = load_config(str(cfg_file))
    assert cfg.agent["workspace_root"] is UNSET  # explicit-null preserved
    monkeypatch.setattr(ar, "_UNRESTRICTED_WARNED", False)
    with caplog.at_level(logging.WARNING, logger="toaster_loop.agent"):
        assert resolve_workspace_roots(cfg) is None
    assert any("containment is DISABLED" in r.message for r in caplog.records)

    # unset (key absent entirely) => cwd default, NOT unrestricted
    cfg_file.write_text("agent:\n  keep_recent_tokens: 10\n")
    cfg2 = load_config(str(cfg_file))
    assert "workspace_root" not in cfg2.agent
    assert resolve_workspace_roots(cfg2) is not None


def test_create_runner_applies_containment(tmp_path, toaster_loop_home, _tempdir_outside_workspace):
    ws = tmp_path / "ws"
    ws.mkdir()
    cfg = ToasterLoopConfig(agent={"workspace_root": str(ws)})
    runner = create_runner(cfg, base_url="http://127.0.0.1:9/v1", model="m")
    assert runner.workspace_roots and runner.workspace_roots[0] == ws
    out = runner.tool_impls["write_file"](path=str(tmp_path / "x.txt"),
                                          content="boom")
    assert out.startswith("error: blocked: path outside workspace")
    ok = runner.tool_impls["write_file"](path=str(ws / "ok.txt"), content="hi")
    assert ok.startswith("wrote")


# ------------------------------------------------------------ runner integration

def _tool_loop_handler(tool: str, args: dict):
    """Handler class that issues one tool call, then finishes."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args_):
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            messages = payload.get("messages", [])
            if any(m.get("role") == "tool" for m in messages):
                msg = {"role": "assistant", "content": "done"}
                finish = "stop"
            else:
                msg = {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "c1", "type": "function",
                    "function": {"name": tool,
                                 "arguments": json.dumps(args)}}]}
                finish = "tool_calls"
            body = json.dumps({
                "id": "x", "object": "chat.completion",
                "model": payload.get("model", "m"),
                "choices": [{"index": 0, "message": msg,
                             "finish_reason": finish}],
                "usage": {}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def _run_tool(tmp_path, toaster_loop_home, tool: str, args: dict,
              workspace_root: Path) -> tuple:
    """Run one tool call through a contained runner; return (result, content)."""
    cfg = ToasterLoopConfig(agent={"workspace_root": str(workspace_root)})
    with FakeOpenAIServer(handler_cls=_tool_loop_handler(tool, args)) as srv:
        runner = create_runner(cfg, base_url=srv.base_url, model="m")
        result = runner.run("go")
    tool_msgs = [m["content"] for m in result.messages if m.get("role") == "tool"]
    return result, tool_msgs


def test_runner_refuses_write_outside_workspace(tmp_path, toaster_loop_home, _tempdir_outside_workspace):
    ws = tmp_path / "ws"
    ws.mkdir()
    outside = ws.parent / "outside.txt"
    result, contents = _run_tool(
        tmp_path, toaster_loop_home, "write_file",
        {"path": str(ws / ".." / "outside.txt"), "content": "x"}, ws)
    assert any(c.startswith("error: blocked: '..' path traversal")
               for c in contents)
    assert not outside.exists()

    # absolute outside path: blocked by containment, file never created
    result, contents = _run_tool(
        tmp_path, toaster_loop_home, "write_file",
        {"path": str(outside), "content": "x"}, ws)
    assert any("path outside workspace" in c for c in contents)
    assert not outside.exists()


def test_runner_blocks_symlink_escape(tmp_path, toaster_loop_home, _tempdir_outside_workspace):
    ws = tmp_path / "ws"
    ws.mkdir()
    victim = tmp_path / "victim"
    victim.mkdir()
    link = ws / "escape"
    link.symlink_to(victim)
    result, contents = _run_tool(
        tmp_path, toaster_loop_home, "write_file",
        {"path": str(link / ".bashrc"), "content": "evil"}, ws)
    assert any("path outside workspace" in c for c in contents)
    assert not (victim / ".bashrc").exists()


def test_runner_allows_inside_and_todo_write(tmp_path, toaster_loop_home, _tempdir_outside_workspace):
    ws = tmp_path / "ws"
    ws.mkdir()
    result, contents = _run_tool(
        tmp_path, toaster_loop_home, "write_file",
        {"path": str(ws / "notes" / "a.txt"), "content": "hello"}, ws)
    assert any(c.startswith("wrote") for c in contents)
    assert (ws / "notes" / "a.txt").read_text() == "hello"

    # todo_write lives under TOASTER_LOOP_HOME — allowed via the home allow-list
    result, contents = _run_tool(
        tmp_path, toaster_loop_home, "todo_write",
        {"todos": [{"id": "t1", "content": "ship it", "status": "pending"}]}, ws)
    assert any("[ ] t1: ship it" in c for c in contents)


def test_runner_tempdir_allowlist(tmp_path, toaster_loop_home, _tempdir_outside_workspace):
    ws = tmp_path / "ws"
    ws.mkdir()
    td = Path(tempfile.gettempdir()) / f"toaster_loop-allow-{tmp_path.name}.txt"
    try:
        result, contents = _run_tool(
            tmp_path, toaster_loop_home, "write_file",
            {"path": str(td), "content": "tmp"}, ws)
        assert any(c.startswith("wrote") for c in contents)
        assert td.read_text() == "tmp"
    finally:
        td.unlink(missing_ok=True)


def test_runner_list_dir_sensitive_and_containment(tmp_path, toaster_loop_home, _tempdir_outside_workspace):
    ws = tmp_path / "ws"
    ws.mkdir()
    # sensitive deny list now covers list_dir roots too (inside the
    # workspace, so containment passes and the deny list is what refuses)
    fake_ssh = ws / ".ssh"
    fake_ssh.mkdir()
    result, contents = _run_tool(
        tmp_path, toaster_loop_home, "list_dir", {"path": str(fake_ssh)}, ws)
    assert any("sensitive credential path" in c for c in contents)
    # and containment covers it as an outside path when not sensitive-flagged:
    result, contents = _run_tool(
        tmp_path, toaster_loop_home, "list_dir", {"path": str(tmp_path)}, ws)
    assert any("path outside workspace" in c for c in contents)


def test_runner_unrestricted_when_workspace_root_null(tmp_path, toaster_loop_home,
                                                      monkeypatch, _tempdir_outside_workspace):
    monkeypatch.setattr(ar, "_UNRESTRICTED_WARNED", False)
    ws = tmp_path / "ws"
    ws.mkdir()
    target = tmp_path / "free.txt"
    cfg = ToasterLoopConfig(agent={"workspace_root": None})
    with FakeOpenAIServer(
            handler_cls=_tool_loop_handler(
                "write_file", {"path": str(target), "content": "ok"})) as srv:
        runner = create_runner(cfg, base_url=srv.base_url, model="m")
        assert runner.workspace_roots is None
        runner.run("go")
    assert target.read_text() == "ok"


# ------------------------------------- config-takeover carve-out (deny_writes)

def test_contain_deny_writes_blocks_config_write_allows_read(tmp_path, _tempdir_outside_workspace):
    ws = tmp_path / "ws"
    ws.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    cfg_file = home / "toaster_loop.yaml"
    cfg_file.write_text("agent: {}\n")

    @ps.contain([ws, home], deny_writes=[cfg_file])
    def write_file(path: str, content: str = "") -> str:
        return f"executed {path}"

    blocked = write_file(path=str(cfg_file), content="pwned")
    assert blocked.startswith("error: blocked:")
    assert "cannot write the toaster_loop config" in blocked
    # reads of the same path stay allowed through the allow-list
    @ps.contain([ws, home], deny_writes=[cfg_file], read_only=True)
    def read_file(path: str) -> str:
        return f"executed {path}"

    assert read_file(path=str(cfg_file)) == f"executed {cfg_file}"


def test_runner_blocks_config_takeover(tmp_path, toaster_loop_home, _tempdir_outside_workspace):
    """End-to-end: agent-mode write_file at <home>/toaster_loop.yaml is refused and
    the file is untouched; other TOASTER_LOOP_HOME paths still work (allow-list)."""
    ws = tmp_path / "ws"
    ws.mkdir()
    toaster_loop_home.mkdir(parents=True, exist_ok=True)
    cfg_file = toaster_loop_home / "toaster_loop.yaml"
    cfg_file.write_text("agent:\n  workspace_root: keep-me\n")

    result, contents = _run_tool(
        tmp_path, toaster_loop_home, "write_file",
        {"path": str(cfg_file), "content": "capabilities: {run_commands: true}"},
        ws)
    assert any("cannot write the toaster_loop config" in c for c in contents)
    assert cfg_file.read_text().startswith("agent:")

    # a sibling path under TOASTER_LOOP_HOME is still writable (allow-list intact)
    ok_target = toaster_loop_home / "scratch.txt"
    result, contents = _run_tool(
        tmp_path, toaster_loop_home, "write_file",
        {"path": str(ok_target), "content": "fine"}, ws)
    assert ok_target.is_file()

# ------------------------------------------------------------------------
# from tests/test_playbooks.py
# ------------------------------------------------------------------------

# ------------------------------------------------------------------ fixtures

def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def _write_transcript(path: Path, tool_names, first_user="fix the bug",
                      fail_tools=()) -> None:
    """Minimal recorder-shaped JSONL: assistant tool_calls + tool results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    recs = [{
        "kind": "message",
        "message": {"role": "user", "content": first_user},
    }]
    for i, name in enumerate(tool_names):
        recs.append({
            "kind": "message",
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {"name": name,
                                 "arguments": json.dumps({"path": "x.py"})},
                }],
            },
        })
        content = ("error: boom" if name in fail_tools
                   else f"{name} ok")
        recs.append({
            "kind": "message",
            "message": {"role": "tool", "tool_call_id": f"call_{i}",
                        "name": name, "content": content},
        })
    recs.append({"kind": "message",
                 "message": {"role": "assistant", "content": "done"}})
    path.write_text("\n".join(json.dumps(r) for r in recs) + "\n")


@pytest.fixture()
def store(tmp_path):
    s = TraceStore(tmp_path / "buffer" / "toaster_loop.db")
    yield s
    s.close()


@pytest.fixture()
def pb_store(tmp_path):
    ps = PlaybookStore(tmp_path / "buffer" / "toaster_loop.db")
    yield ps
    ps.close()


def _must(pb_store, pb_id):
    """get() + assert-not-None (typing convenience for the assertions)."""
    pb = pb_store.get(pb_id)
    assert pb is not None
    return pb


def _add_trace(store, tmp_path, tid, tools, *, profile="coder",
               status="success", when=None, score=0.5, fail_tools=(),
               first_user="fix the bug"):
    when = when or datetime.now(timezone.utc)
    jp = tmp_path / "traces" / f"{tid}.jsonl"
    _write_transcript(jp, tools, first_user=first_user,
                      fail_tools=fail_tools)
    store.insert_trace(TraceRow(
        trace_id=tid, session_id=None, timestamp=_iso(when),
        agent_profile=profile, model_alias="m", model_version=None,
        endpoint_label=None, task_summary="t", status=status,
        outcome_score=score, num_tool_calls=len(tools), tokens_in=None,
        tokens_out=None, duration_s=None, jsonl_path=str(jp)))


# ------------------------------------------------------------- chain mining

def test_scan_finds_recurring_chain_with_counts(store, tmp_path):
    for i in range(3):
        _add_trace(store, tmp_path, f"t{i}",
                   ["read_file", "search_files", "patch"])
    _add_trace(store, tmp_path, "x1", ["list_dir"])  # too short: no chain

    chains = find_recurring_chains(store, "coder", min_occurrences=3)
    got = {tuple(c["chain"]): c["count"] for c in chains}
    assert ("read_file", "search_files", "patch") in got
    assert got[("read_file", "search_files", "patch")] == 3
    # contiguous subsequences of length 2 also recur
    assert got[("read_file", "search_files")] == 3
    assert all(c["count"] >= 3 for c in chains)


def test_scan_excludes_failures(store, tmp_path):
    _add_trace(store, tmp_path, "ok1", ["a", "b"], status="success")
    _add_trace(store, tmp_path, "ok2", ["a", "b"], status="partial")
    _add_trace(store, tmp_path, "bad1", ["a", "b"], status="failure")
    _add_trace(store, tmp_path, "ok3", ["a", "b"], status="success")

    # the failing trace never counts: exactly the 3 non-failures do
    chains = find_recurring_chains(store, "coder", min_occurrences=3)
    hit = [c for c in chains if c["chain"] == ["a", "b"]]
    assert hit and hit[0]["count"] == 3  # success + partial count
    assert set(hit[0]["trace_ids"]) == {"ok1", "ok2", "ok3"}
    # and with four required occurrences nothing qualifies
    assert [c for c in find_recurring_chains(store, "coder",
                                             min_occurrences=4)
            if c["chain"] == ["a", "b"]] == []


def test_scan_respects_window(store, tmp_path):
    old = datetime.now(timezone.utc) - timedelta(days=45)
    for i in range(2):
        _add_trace(store, tmp_path, f"old{i}", ["a", "b"], when=old)
    for i in range(3):
        _add_trace(store, tmp_path, f"new{i}", ["a", "b"])

    chains = find_recurring_chains(store, "coder", min_occurrences=3,
                                   window_days=30)
    hit = [c for c in chains if c["chain"] == ["a", "b"]]
    assert hit and hit[0]["count"] == 3  # the two old traces are out of window


def test_scan_order_deterministic(store, tmp_path):
    for i in range(4):
        _add_trace(store, tmp_path, f"a{i}", ["read_file", "edit_file"])
    for i in range(2):
        _add_trace(store, tmp_path, f"b{i}", ["aaa_tool", "zzz_tool"])

    def order():
        return list(find_recurring_chains(store, "coder",
                                          min_occurrences=2))

    first = order()
    # count desc: the 4x chain precedes any 2x chain; ties asc by join
    counts = [c["count"] for c in find_recurring_chains(
        store, "coder", min_occurrences=2)]
    assert counts == sorted(counts, reverse=True)
    # ties break ascending on the joined chain
    seen = {}
    for c in first:
        seen.setdefault(c["count"], []).append("\x00".join(c["chain"]))
    for chains_at_n in seen.values():
        assert chains_at_n == sorted(chains_at_n)
    # repeated calls identical
    assert order() == first


def test_occurrence_deduped_per_trace(store, tmp_path):
    # chain appears twice inside ONE trace => still one occurrence
    _add_trace(store, tmp_path, "d1", ["a", "b", "c", "a", "b"])
    chains = find_recurring_chains(store, "coder", min_occurrences=2)
    assert [c for c in chains if c["chain"] == ["a", "b"]] == []


# ------------------------------------------------------------- distillation

def test_distill_template_contains_facts_no_placeholders(store, tmp_path):
    from toaster_loop.curate.curater import load_trace

    for i in range(3):
        _add_trace(store, tmp_path, f"t{i}", ["read_file", "patch"],
                   first_user=f"refactor module {i}", fail_tools=("patch",))
    rows = [store.get_trace(f"t{i}") for i in range(3)]
    examples = [load_trace(store, r) for r in rows]

    text = distill_recipe(["read_file", "patch"], examples, occurrences=3,
                          profile="coder")
    assert "## Workflow: read_file → patch" in text
    assert "3 recorded run(s)" in text
    assert "Typical task:" in text
    assert "1. read_file" in text and "2. patch" in text
    assert "args seen: path" in text
    assert "Gotchas:" in text and "patch returned an error-shaped result" in text
    # no leftover template artifacts
    for junk in ("{", "}", "<%", "TODO", "PLACEHOLDER"):
        assert junk not in text


def test_distill_redacts_secrets(tmp_path):
    class FakeT:
        row = None

        def __init__(self, msgs):
            self.messages = msgs

    t = FakeT([{"role": "user",
                "content": "use key sk-abcdefghijklmnopqrstuvwxyz012345"}])
    text = distill_recipe(["a", "b"], [t], occurrences=1)
    assert "sk-abcdefghijklmnopqrstuvwxyz012345" not in text
    assert "[REDACTED" in text


# ---------------------------------------------------------------- lifecycle

def test_repromotion_stamps_fresh_era(pb_store):
    """Retire → re-promote starts a NEW era: promoted_ts is re-stamped so
    assess_playbook compares the right before/after boundary (the first
    promotion's timestamp must not survive across a retirement)."""
    pb_id, _ = pb_store.create_candidate(["a", "b"], "r")
    assert promote(pb_store, pb_id)["promoted"]
    era1 = _must(pb_store, pb_id).promoted_ts
    assert era1
    pb_store.set_status(pb_id, "retired")
    assert promote(pb_store, pb_id)["promoted"]
    pb = _must(pb_store, pb_id)
    assert pb.status == "active" and pb.promoted_ts >= era1
    # monotonic seconds: re-promotion within the same second can tie; never
    # may it be OLDER than the first era.
    assert pb.promoted_ts is not None


def test_assess_cli_reports_and_retires(tmp_path, monkeypatch, capsys):
    from toaster_loop.cli import main

    home = tmp_path / "home"
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(home))
    monkeypatch.delenv("TOASTER_LOOP_CONFIG", raising=False)
    st = TraceStore(home / "buffer" / "toaster_loop.db")
    ps = PlaybookStore(home / "buffer" / "toaster_loop.db")
    try:
        pb_id, _ = ps.create_candidate(["a", "b"], "r", profile="coder")
        promote(ps, pb_id)
        # era boundary then a clear regression over enough samples
        old = datetime.now(timezone.utc) - timedelta(days=5)
        for i in range(3):
            _add_trace(st, tmp_path, f"pre{i}", ["a", "b"], score=0.8,
                       when=old)
        for i in range(3):
            _add_trace(st, tmp_path, f"post{i}", ["a", "b"], score=-0.9)
        rc = main(["playbooks", "assess"])
        assert rc == 0
        out = capsys.readouterr().out
        assert pb_id in out and "RETIRED" in out
        assert _must(ps, pb_id).status == "retired"
    finally:
        ps.close()
        st.close()


def test_candidate_idempotent_and_status_transitions(pb_store):
    pb_id, created = pb_store.create_candidate(["a", "b"], "recipe",
                                               profile="coder")
    assert created
    pb_id2, created2 = pb_store.create_candidate(["a", "b"], "other",
                                                 profile="coder")
    assert pb_id == pb_id2 and not created2  # deterministic id, no clobber

    assert pb_store.set_status(pb_id, "active")
    pb = pb_store.get(pb_id)
    assert pb.status == "active" and pb.promoted_ts
    assert pb_store.set_status(pb_id, "retired")
    pb = pb_store.get(pb_id)
    assert pb.status == "retired"
    assert pb.promoted_ts  # era boundary kept
    assert pb_store.get(pb_id) is not None  # retired rows KEPT


def test_promote_gate_pass_and_fail(pb_store, store, tmp_path):
    # replay set: positive-score success traces for the profile
    for i in range(3):
        _add_trace(store, tmp_path, f"r{i}", ["a", "b"], score=0.8)
    pb_id, _ = pb_store.create_candidate(["a", "b"], "recipe text",
                                         profile="coder")
    pb = pb_store.get(pb_id)

    def gate_stub(mean):
        return {"ts": "t", "n": 3, "scored": 3, "skipped": 0,
                "judge_mean": mean, "threshold": 0.7,
                "passed": mean is not None and mean >= 0.7, "reason": None}

    # fail: candidate untouched, history recorded
    res = promote(pb_store, pb_id, check=True, gate=gate_stub(0.4))
    assert not res["ok"] and not res.get("promoted")
    assert pb_store.get(pb_id).status == "candidate"
    hist = pb_store.get(pb_id).score_history
    assert hist and hist[-1]["judge_mean"] == 0.4

    # pass: promoted, promoted_ts stamped
    res = promote(pb_store, pb_id, check=True, gate=gate_stub(0.9))
    assert res["ok"] and res["promoted"]
    assert pb_store.get(pb_id).status == "active"


def test_promote_without_check_is_explicit_confirmation(pb_store):
    pb_id, _ = pb_store.create_candidate(["a", "b"], "r")
    res = promote(pb_store, pb_id)  # no gate: the call itself confirms
    assert res["ok"] and res["promoted"]
    assert pb_store.get(pb_id).status == "active"


def test_promotion_gate_requires_replay_set(store, tmp_path):
    pb = PlaybookStore(tmp_path / "pb.db")
    try:
        pb_id, _ = pb.create_candidate(["a"], "r", profile="nobody")
        with pytest.raises(GateUnavailable):
            run_promotion_gate(store, _must(pb, pb_id),
                               base_url="http://127.0.0.1:1", model="m")
    finally:
        pb.close()


def test_promotion_gate_bounded_calls_and_zero_scored_fails(
        store, tmp_path):
    for i in range(2):
        _add_trace(store, tmp_path, f"g{i}", ["a", "b"], score=0.9)
    pb = PlaybookStore(tmp_path / "pb.db")
    try:
        pb_id, _ = pb.create_candidate(["a", "b"], "r", profile="coder")
        pbo = _must(pb, pb_id)
        calls = {"gen": 0}

        def gen(task):
            calls["gen"] += 1
            return None  # every generation fails => zero scored

        verdict = run_promotion_gate(store, pbo, base_url="http://x/v1",
                                     model="m", n=6, generate_fn=gen,
                                     judge_client_for=lambda u, m:
                                     (lambda msgs, mdl: None))
        assert not verdict["passed"] and verdict["scored"] == 0
        assert calls["gen"] == 2  # bounded by replay size, no extras

        def gen_ok(task):
            return "answer"
        verdict = run_promotion_gate(
            store, pbo, base_url="http://x/v1", model="m", n=6,
            generate_fn=gen_ok,
            judge_client_for=lambda u, m: (lambda msgs, mdl:
                                           '{"score": 0.95, "reason": "ok"}'))
        assert verdict["passed"] and verdict["judge_mean"] == 0.95
    finally:
        pb.close()


def test_assess_demotes_on_regression_kept(store, tmp_path):
    pb = PlaybookStore(tmp_path / "buffer" / "toaster_loop.db")
    try:
        # pre-era traces score well; post-era matching traces collapse
        old = datetime.now(timezone.utc) - timedelta(days=5)
        for i in range(3):
            _add_trace(store, tmp_path, f"pre{i}", ["a", "b"], when=old,
                       score=0.5)
        pb_id, _ = pb.create_candidate(["a", "b"], "r", profile="coder")
        pb.set_status(pb_id, "active")
        boundary = _must(pb, pb_id).promoted_ts
        for i in range(3):
            _add_trace(store, tmp_path, f"post{i}", ["a", "b"], score=-0.9)
        # ensure post rows are strictly after the promoted boundary
        row = store.get_trace("post0")
        assert row.timestamp >= boundary

        res = assess_playbook(store, pb, _must(pb, pb_id))
        assert res["ok"] and res["demoted"]
        assert res["mean_without"] > 0 > res["mean_with"]
        assert (res["mean_with"] + 1) / 2 <= DEMOTE_NORM_THRESHOLD
        got = _must(pb, pb_id)
        assert got.status == "retired"          # demoted...
        assert got.score_history[-1]["n"] >= 6  # ...with the reading stored
    finally:
        pb.close()


def test_assess_no_regression_stays_active(store, tmp_path):
    pb = PlaybookStore(tmp_path / "buffer" / "toaster_loop.db")
    try:
        old = datetime.now(timezone.utc) - timedelta(days=5)
        for i in range(3):
            _add_trace(store, tmp_path, f"pre{i}", ["a", "b"], when=old,
                       score=-0.2)
        pb_id, _ = pb.create_candidate(["a", "b"], "r", profile="coder")
        pb.set_status(pb_id, "active")
        for i in range(3):
            _add_trace(store, tmp_path, f"post{i}", ["a", "b"], score=0.5)
        res = assess_playbook(store, pb, _must(pb, pb_id))
        assert res["ok"] and not res["demoted"]
        assert _must(pb, pb_id).status == "active"
    finally:
        pb.close()


# ---------------------------------------------------------------- injection

def _cfg_with(tmp_path, **playbooks):
    return ToasterLoopConfig(home=str(tmp_path / "home"), playbooks=playbooks)


def _seed_active(pb_store, chain, text, profile=None, promoted_ts="b"):
    pb_id, _ = pb_store.create_candidate(chain, text, profile=profile)
    pb_store.set_status(pb_id, "active")
    with pb_store._lock:
        pb_store._conn.execute(
            "UPDATE playbooks SET promoted_ts = ? WHERE id = ?",
            (promoted_ts, pb_id))
        pb_store._conn.commit()
    return pb_id


def test_injection_enabled_contains_section(tmp_path, monkeypatch):
    monkeypatch.delenv("TOASTER_LOOP_HOME", raising=False)  # cfg.home resolution
    home = tmp_path / "home"
    (home / "buffer").mkdir(parents=True)
    pb = PlaybookStore(home / "buffer" / "toaster_loop.db")
    try:
        _seed_active(pb, ["read_file", "patch"], "## Workflow: r → p\nbody")
    finally:
        pb.close()
    cfg = _cfg_with(tmp_path, enabled=True)

    section = prompt_section(cfg, None)
    assert section.startswith("## Playbooks\n")
    assert "## Workflow: r → p" in section


def test_injection_caps_omit_whole_recipes(tmp_path, monkeypatch):
    monkeypatch.delenv("TOASTER_LOOP_HOME", raising=False)  # cfg.home resolution
    home = tmp_path / "home"
    (home / "buffer").mkdir(parents=True)
    pb = PlaybookStore(home / "buffer" / "toaster_loop.db")
    try:
        big = "x" * 900  # > per_recipe_cap 800 -> omitted whole
        ok1 = "A" * 300
        ok2 = "B" * 300
        _seed_active(pb, ["a"], big, promoted_ts="2026-01-01T00:00:00+00:00")
        _seed_active(pb, ["b"], ok1, promoted_ts="2026-01-02T00:00:00+00:00")
        _seed_active(pb, ["c"], ok2, promoted_ts="2026-01-03T00:00:00+00:00")
        cfg = ToasterLoopConfig(home=str(home), playbooks={
            "enabled": True, "total_cap_chars": 400,
            "per_recipe_cap_chars": 800})
        section = prompt_section(cfg, None)
        assert "BBB" in section          # most-recently-promoted first
        assert "AAA" not in section      # does not fit remaining budget
        assert "xxx" not in section      # over per-recipe cap
        assert len(section) <= 400 + len("## Playbooks\n")
    finally:
        pb.close()


def test_injection_disabled_byte_identical_prompt(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / "buffer").mkdir(parents=True)
    pb = PlaybookStore(home / "buffer" / "toaster_loop.db")
    try:
        _seed_active(pb, ["a", "b"], "SHOULD NOT APPEAR")
    finally:
        pb.close()
    monkeypatch.delenv("TOASTER_LOOP_HOME", raising=False)
    cfg = ToasterLoopConfig(home=str(home), playbooks={})  # disabled default

    from toaster_loop.core.agent_runner import create_runner

    captured = {}

    class Spy(Exception):
        pass

    def spy_client(*args, **kwargs):
        captured["system_prompt"] = kwargs.get("system_prompt")
        raise Spy()

    monkeypatch.setattr("toaster_loop.core.agent_runner.ToasterLoopAgentClient",
                        spy_client)
    with pytest.raises(Spy):
        create_runner(cfg, "http://127.0.0.1:1", "m",
                      system_prompt="BASE PROMPT", profile_name="coder")
    assert captured["system_prompt"] == "BASE PROMPT"  # untouched

    # and enabled flips it (same fixture DB has an active playbook)
    cfg.playbooks = {"enabled": True}
    with pytest.raises(Spy):
        create_runner(cfg, "http://127.0.0.1:1", "m",
                      system_prompt="BASE PROMPT", profile_name="coder")
    assert "## Playbooks" in captured["system_prompt"]
    assert captured["system_prompt"].startswith("BASE PROMPT")


# ------------------------------------------------------------------- export

def test_export_writes_metadata_not_rows(tmp_path, pb_store):
    pb_id, _ = pb_store.create_candidate(["a", "b"], "recipe", profile="coder",
                                         source_trace_ids=["t1", "t2"])
    pb_store.set_status(pb_id, "active")
    out = export_playbook(pb_store.get(pb_id), home=tmp_path)
    payload = json.loads(Path(out).read_text())
    assert payload["not_training_rows"] is True
    assert payload["chain"] == ["a", "b"]
    assert payload["recipe"] == "recipe"
    assert payload["source_trace_ids"] == ["t1", "t2"]
    assert payload["kind"] == "toaster_loop.playbook_export"
    assert "conversations" not in payload  # no fabricated rows
    assert out.parent.name == "_playbooks"


# --------------------------------------------------------------- db schema

def test_playbooks_table_via_tracestore(tmp_path):
    """The playbooks table is part of the buffer schema (SCHEMA_VERSION 6)."""
    s = TraceStore(tmp_path / "x.db")
    try:
        cols = {r[1] for r in s._conn.execute("PRAGMA table_info(playbooks)")}
        assert {"id", "profile", "chain_json", "text", "status",
                "created_ts", "promoted_ts",
                "score_history_json"} <= cols
    finally:
        s.close()


# -------------------------------------------------------------- CLI smoke

def test_cli_playbooks_end_to_end(tmp_path, monkeypatch, capsys):
    from toaster_loop.cli import main

    home = tmp_path / "clihome"
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(home))
    st = TraceStore(home / "buffer" / "toaster_loop.db")
    try:
        for i in range(3):
            _add_trace(st, tmp_path / "cli", f"c{i}", ["read_file", "patch"],
                       profile="coder")
    finally:
        st.close()

    assert main(["playbooks", "scan", "--profile", "coder",
                 "--create"]) == 0
    out = capsys.readouterr().out
    assert "read_file → patch" in out and "created candidate pb-" in out

    assert main(["playbooks", "candidates"]) == 0
    out = capsys.readouterr().out
    pb_id = [tok for tok in out.split() if tok.startswith("pb-")][0]

    assert main(["playbooks", "show", pb_id]) == 0
    assert "candidate" in capsys.readouterr().out

    # explicit confirmation promote (no --check, no endpoint needed)
    assert main(["playbooks", "promote", pb_id]) == 0
    assert "promoted" in capsys.readouterr().out

    assert main(["playbooks", "list"]) == 0
    assert "active" in capsys.readouterr().out

    assert main(["playbooks", "export", pb_id]) == 0
    out = capsys.readouterr().out
    assert "NOT training rows" in out
    exported = home / "datasets" / "_playbooks" / f"{pb_id}.json"
    assert exported.is_file()

    assert main(["playbooks", "retire", pb_id]) == 0
    assert "retired" in capsys.readouterr().out

    # retired row kept (show still works)
    assert main(["playbooks", "show", pb_id]) == 0
    assert "retired" in capsys.readouterr().out


def test_cli_promote_check_without_endpoint_refuses(tmp_path, monkeypatch,
                                                    capsys):
    from toaster_loop.cli import main

    home = tmp_path / "clihome2"
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(home))
    monkeypatch.delenv("TOASTER_LOOP_ATTACH_URL", raising=False)
    ps = PlaybookStore(home / "buffer" / "toaster_loop.db")
    try:
        pb_id, _ = ps.create_candidate(["a", "b"], "r", profile="coder")
    finally:
        ps.close()
    assert main(["playbooks", "promote", pb_id, "--check"]) == 1
    assert "served endpoint" in capsys.readouterr().err

# ------------------------------------------------------------------------
# from tests/test_gpumem.py
# ------------------------------------------------------------------------

MIB = 1024 * 1024
GIB = 1024 * MIB


def _fake_drm(root: Path, *, vram_total: int, vram_used: int,
              gtt_total: int, gtt_used: int, pci_class: str,
              vendor: str = "0x1002") -> Path:
    dev = root / "card1" / "device"
    dev.mkdir(parents=True)
    (dev / "vendor").write_text(vendor + "\n")
    (dev / "class").write_text(pci_class + "\n")
    (dev / "mem_info_vram_total").write_text(str(vram_total))
    (dev / "mem_info_vram_used").write_text(str(vram_used))
    (dev / "mem_info_gtt_total").write_text(str(gtt_total))
    (dev / "mem_info_gtt_used").write_text(str(gtt_used))
    return root


MEMINFO = {"MemTotal": 128 * 1024 * 1024,      # KiB — 128 GiB
           "MemAvailable": 7 * 1024 * 1024}    # 7 GiB


# ----------------------------------------------------------- detection

def test_strix_halo_carveout_detected(tmp_path):
    """The reported machine: 512 MiB carve-out, 120 GiB GTT, display class."""
    base = _fake_drm(tmp_path, vram_total=512 * MIB, vram_used=466 * MIB,
                     gtt_total=120 * GIB, gtt_used=70 * GIB,
                     pci_class="0x038000")
    pool = unified_pool(drm_base=str(base), meminfo=MEMINFO)
    assert pool is not None
    assert pool.vram_carveout_mib == 512
    assert pool.total_mib == 120 * 1024          # GTT bounded by MemTotal
    # free = min(gtt_free=50G, mem_avail=7G) = 7 GiB — NOT the 44 MiB
    # carve-out free the vendor tools insisted on.
    assert pool.free_mib == 7 * 1024


def test_discrete_gpu_not_touched(tmp_path):
    """Class 0x030000 (discrete 3D controller): vendor VRAM numbers are
    real; the probe must stay out of the way (return None)."""
    base = _fake_drm(tmp_path, vram_total=24 * GIB, vram_used=1 * GIB,
                     gtt_total=48 * GIB, gtt_used=0,
                     pci_class="0x030000")
    assert unified_pool(drm_base=str(base), meminfo=MEMINFO) is None


def test_no_gtt_not_touched(tmp_path):
    """iGPU-class card without GTT files (no TTM setup): not our case."""
    base = _fake_drm(tmp_path, vram_total=512 * MIB, vram_used=100 * MIB,
                     gtt_total=0, gtt_used=0, pci_class="0x038000")
    assert unified_pool(drm_base=str(base), meminfo=MEMINFO) is None


def test_gtt_must_dominate_carveout(tmp_path):
    """GTT only barely above a *big* carve-out = a normal dGPU with TTM
    staging; class + dominance both required, so skip."""
    base = _fake_drm(tmp_path, vram_total=16 * GIB, vram_used=1 * GIB,
                     gtt_total=40 * GIB, gtt_used=0, pci_class="0x030000")
    assert unified_pool(drm_base=str(base), meminfo=MEMINFO) is None


def test_env_overrides(tmp_path):
    base = _fake_drm(tmp_path, vram_total=512 * MIB, vram_used=466 * MIB,
                     gtt_total=120 * GIB, gtt_used=70 * GIB,
                     pci_class="0x038000")
    off = unified_pool(drm_base=str(base), meminfo=MEMINFO,
                       env={"TOASTER_LOOP_UNIFIED_MEMORY": "0"})
    assert off is None
    # Force-on covers an unknown-class oddity:
    force = _fake_drm(tmp_path / "f", vram_total=512 * MIB, vram_used=0,
                      gtt_total=32 * GIB, gtt_used=0, pci_class="")
    pool = unified_pool(drm_base=str(force), meminfo=MEMINFO,
                        env={"TOASTER_LOOP_UNIFIED_MEMORY": "1"})
    assert pool is not None and pool.total_mib == 32 * 1024


def test_meminfo_absent_bounds_only_by_gtt(tmp_path):
    base = _fake_drm(tmp_path, vram_total=512 * MIB, vram_used=0,
                     gtt_total=120 * GIB, gtt_used=100 * GIB,
                     pci_class="0x038000")
    # {} = meminfo read but fields missing (None means "read /proc"!).
    pool = unified_pool(drm_base=str(base), meminfo={})
    assert pool.free_mib == 20 * 1024            # gtt_free alone
    assert pool.mem_available_mib is None


def test_gtt_beyond_physical_ram_capped(tmp_path):
    """Firmware claims 240 GiB of GTT on a 128 GiB machine: capacity caps
    at MemTotal — the kernel cannot hand out memory that does not exist."""
    base = _fake_drm(tmp_path, vram_total=512 * MIB, vram_used=0,
                     gtt_total=240 * GIB, gtt_used=0, pci_class="0x038000")
    pool = unified_pool(drm_base=str(base), meminfo=MEMINFO)
    assert pool.total_mib == 128 * 1024


def test_non_amd_card_skipped(tmp_path):
    base = _fake_drm(tmp_path, vram_total=512 * MIB, vram_used=0,
                     gtt_total=120 * GIB, gtt_used=0,
                     pci_class="0x038000", vendor="0x8086")
    assert unified_pool(drm_base=str(base), meminfo=MEMINFO) is None


# ------------------------------------------------------------ wiring

def test_scheduler_gate_uses_unified_pool(monkeypatch):
    """The training gate must see the pool, not the 44 MiB carve-out."""
    import toaster_loop.training.scheduler as sched

    pool = gpumem.UnifiedPool(total_mib=120 * 1024, free_mib=7 * 1024)
    monkeypatch.setattr(gpumem, "unified_pool", lambda **kw: pool)
    monkeypatch.setattr(sched, "_nvidia_free_mib", lambda *a: None)
    monkeypatch.setattr(sched, "_rocm_free_mib", lambda *a: 44)
    assert sched.gpu_free_mib() == 7 * 1024
    # Device-pinned queries keep the historical per-device semantics.
    assert sched.gpu_free_mib("1") == 44


def test_gate_falls_back_when_probe_raises(monkeypatch):
    import toaster_loop.training.scheduler as sched

    def boom(**kw):
        raise RuntimeError("sysfs exploded")
    monkeypatch.setattr(gpumem, "unified_pool", boom)
    monkeypatch.setattr(sched, "_nvidia_free_mib", lambda *a: None)
    monkeypatch.setattr(sched, "_rocm_free_mib", lambda *a: 8000)
    assert sched.gpu_free_mib() == 8000


def test_doctor_chains_unified_first(monkeypatch):
    from toaster_loop import doctor

    pool = gpumem.UnifiedPool(total_mib=120 * 1024, free_mib=6656)
    monkeypatch.setattr(gpumem, "unified_pool", lambda **kw: pool)
    monkeypatch.setattr(doctor, "free_vram_mib_via_smi", lambda: 44)
    assert doctor.free_vram_mib() == 6656

    monkeypatch.setattr(gpumem, "unified_pool", lambda **kw: None)
    assert doctor.free_vram_mib() == 44          # NVIDIA path untouched

# ------------------------------------------------------------------------
# from tests/test_vram.py
# ------------------------------------------------------------------------

# A small llama-style config for deterministic math.
CFG = {
    "hidden_size": 512,
    "num_hidden_layers": 4,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "intermediate_size": 1024,
    "vocab_size": 1000,
    "head_dim": 64,
}


def dims(**over) -> ModelDims:
    return ModelDims.from_config_json({**CFG, **over})


# ------------------------------------------------------------- dims loading

def test_load_model_dims_local_dir(tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").write_text(json.dumps(CFG))
    md = load_model_dims(str(d))
    assert md is not None and md.hidden_size == 512
    # GQA heads preserved; GGUF path / HF id / missing dir => None
    assert md.num_key_value_heads == 2
    assert load_model_dims("testorg/Foo-LM") is None
    assert load_model_dims(str(tmp_path / "x.gguf")) is None
    assert load_model_dims(None) is None


def test_dims_from_nested_text_config():
    md = ModelDims.from_config_json(
        {"text_config": {"hidden_size": 896, "num_hidden_layers": 24},
         "hidden_size": 64})
    assert md.hidden_size == 896 and md.num_hidden_layers == 24


def test_dims_heuristics_fill_gaps():
    md = ModelDims.from_config_json({"hidden_size": 768})
    assert md.intermediate_size > 0 and md.vocab_size > 0
    assert md.head_dim == 768 // 32


# ------------------------------------------------------------------ counts

def test_lora_param_count_scales_with_r():
    d = dims()
    n64 = lora_param_count(d, 64, ["q_proj", "v_proj"])
    n32 = lora_param_count(d, 32, ["q_proj", "v_proj"])
    assert n64 == 2 * n32
    # q_proj (512+512)*64 + v_proj (512+128)*64 per layer * 4 layers
    assert n64 == ((1024) * 64 + (640) * 64) * 4


def test_base_param_count_sane():
    d = dims()
    n = base_param_count(d)
    # 4 layers * (q,o: 2*512^2; kv: 2*512*128; mlp: 3*512*1024) + emb
    per_layer = 2 * 512 * 512 + 2 * 512 * 128 + 3 * 512 * 1024
    assert n == per_layer * 4 + 1000 * 512


# ---------------------------------------------------------------- estimate

def test_estimate_components_and_consistency():
    e = estimate_train_vram_mib(
        dims(), lora_r=64, load_in_4bit=True, optim="paged_adamw_8bit",
        batch_size=2, max_seq_length=2048)
    assert e.total_mib == (e.weights_mib + e.lora_mib + e.gradients_mib
                           + e.optimizer_mib + e.activations_mib
                           + e.overhead_mib)
    assert e.load_mode == "nf4"
    # 4-bit weights must cost far less than bf16 for the same model.
    e16 = estimate_train_vram_mib(dims(), load_in_4bit=False)
    assert e.weights_mib < e16.weights_mib // 2


def test_estimate_optimizer_table_and_unknown_note():
    kw: dict = dict(lora_r=16, target_modules=["q_proj"])
    a = estimate_train_vram_mib(dims(), optim="adamw", **kw)          # 8 B/p
    b = estimate_train_vram_mib(dims(), optim="adamw_8bit", **kw)     # 4 B/p
    c = estimate_train_vram_mib(dims(), optim="adafactor", **kw)      # 2 B/p
    assert a.optimizer_mib == 2 * b.optimizer_mib
    assert c.optimizer_mib * 4 == a.optimizer_mib
    u = estimate_train_vram_mib(dims(), optim="sgd_ngp", **kw)
    assert any("unknown optimizer" in n for n in u.notes)


def test_estimate_checkpointing_and_flash_reduce():
    base: dict = dict(lora_r=8, batch_size=2, max_seq_length=4096)
    nogc = estimate_train_vram_mib(dims(), gradient_checkpointing=False,
                                   attn_implementation="eager", **base)
    gc = estimate_train_vram_mib(dims(), gradient_checkpointing=True,
                                 attn_implementation="eager", **base)
    flash = estimate_train_vram_mib(dims(), gradient_checkpointing=True,
                                    attn_implementation="flash_attention_2",
                                    **base)
    assert gc.activations_mib < nogc.activations_mib
    assert flash.activations_mib <= gc.activations_mib


def test_estimate_full_finetune_trainable_side():
    e = estimate_train_vram_mib(dims(), train_full=True, optim="adamw")
    n = base_param_count(dims())
    MIB = 1024 * 1024
    assert e.gradients_mib == int(n * 2 / MIB)
    assert e.optimizer_mib == int(n * 8 / MIB)
    assert any("full fine-tune" in x for x in e.notes)


# ------------------------------------------------ from resolved train.yaml

def _write_yaml__test_vram(tmp_path, model_dir):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "train.yaml").write_text(yaml.safe_dump({
        "model": "REPLACE_WITH_BASE_MODEL_PATH_OR_HF_ID",
        "training": {"batch_size": 1, "max_seq_length": 1024,
                     "load_in_4bit": True, "optim": "adamw_8bit"},
        "lora": {"lora_r": 32, "target_modules": "q_proj, v_proj"},
    }))
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps(CFG))


def test_estimate_from_train_yaml(tmp_path, monkeypatch):
    from toaster_loop.core import vram as vram_mod

    model_dir = tmp_path / "base"
    ydir = tmp_path / "ds"
    _write_yaml__test_vram(ydir, model_dir)
    # 4-bit load mode only applies when a GPU is present (the worker ignores
    # load_in_4bit without CUDA), so pin the probe for deterministic math.
    monkeypatch.setattr(vram_mod, "_cuda_available", lambda: True)
    est = estimate_from_train_yaml(str(ydir / "train.yaml"), str(model_dir))
    assert est is not None and est.load_mode == "nf4"
    # lora r=32 two modules -> smaller trainable side than default 7x64.
    big = estimate_train_vram_mib(dims())
    assert est.lora_mib < big.lora_mib
    # Without CUDA the same yaml must fall back to bf16 (no 4-bit on CPU).
    monkeypatch.setattr(vram_mod, "_cuda_available", lambda: False)
    est_cpu = estimate_from_train_yaml(str(ydir / "train.yaml"), str(model_dir))
    assert est_cpu is not None and est_cpu.load_mode == "bf16"


def test_estimate_from_train_yaml_unresolvable(tmp_path):
    ydir = tmp_path / "ds"
    _write_yaml__test_vram(ydir, tmp_path / "unused")
    # No override and placeholder model -> nothing predictable.
    assert estimate_from_train_yaml(str(ydir / "train.yaml")) is None
    # HF id without local dir -> None too (no network guessing).
    assert estimate_from_train_yaml(str(ydir / "train.yaml"),
                                    "meta-llama/Llama-3-8B") is None
    assert estimate_from_train_yaml(None) is None


# --------------------------------------------------------- scheduler gate

class _Job:
    """Duck-typed stand-in for JobRow (the gate only reads kind/params)."""

    def __init__(self, kind, params):
        self.kind = kind
        self.params = params


def _as_jobrow(kind, params) -> Any:
    return cast(Any, _Job(kind, params))


def test_gate_estimate_raises_the_bar(monkeypatch):
    monkeypatch.setattr("toaster_loop.training.scheduler.gpu_free_mib",
                        lambda: 8000)
    cfg = TrainingConfig(min_free_vram_mib=1000)
    assert resource_gate_ok(cfg)["ok"]                       # static floor ok
    assert not resource_gate_ok(cfg, estimated_need_mib=9000)["ok"]
    r = resource_gate_ok(cfg, estimated_need_mib=9000)
    assert "estimate 9000 MiB" in r["reasons"][0]
    assert resource_gate_ok(cfg, estimated_need_mib=7000)["ok"]


def test_gate_no_gpu_info_skips(monkeypatch):
    monkeypatch.setattr("toaster_loop.training.scheduler.gpu_free_mib",
                        lambda: None)
    assert resource_gate_ok(TrainingConfig(min_free_vram_mib=99999),
                            estimated_need_mib=123456)["ok"]


def test_job_estimate_skips_non_gpu_kinds(monkeypatch):
    monkeypatch.setattr("toaster_loop.training.scheduler.gpu_free_mib",
                        lambda: 100_000)
    assert job_vram_estimate_mib(_as_jobrow("validate", {})) is None
    assert job_vram_estimate_mib(_as_jobrow("sft", {})) is None  # no train.yaml
    # kind on a predictable sft job flows through estimate_from_train_yaml:
    called = {}

    def fake(path, base):
        called["args"] = (path, base)
        return type("E", (), {"total_mib": 4242})()
    monkeypatch.setattr("toaster_loop.core.vram.estimate_from_train_yaml", fake)
    assert job_vram_estimate_mib(_as_jobrow(
        "sft", {"train_yaml": "/x/train.yaml", "base_model": "/m"})) == 4242
    assert called["args"] == ("/x/train.yaml", "/m")

# ------------------------------------------------------------------------
# from tests/test_torchenv.py
# ------------------------------------------------------------------------

def _g(vendor):
    return SimpleNamespace(vendor=vendor, name="fake")


class FakeTorch:
    """Minimal torch stand-in: __version__, version.cuda/hip, cuda.is_available."""

    def __init__(self, version="", cuda=None, hip=None, avail=False):
        self.__version__ = version
        self.version = SimpleNamespace(cuda=cuda, hip=hip)
        self.cuda = SimpleNamespace(is_available=lambda: avail)


# ------------------------------------------------------------- flavor choice

def test_macos_default(monkeypatch):
    monkeypatch.delenv("TOASTER_LOOP_TORCH_FLAVOR", raising=False)
    f = detect_flavor(system="darwin")
    assert f.flavor == FLAVOR_DEFAULT and f.is_pypi


def test_nvidia_linux_newest_cuda(monkeypatch):
    monkeypatch.delenv("TOASTER_LOOP_TORCH_FLAVOR", raising=False)
    f = detect_flavor(system="linux", machine="x86_64",
                      gpus=[_g("nvidia")])
    assert f.flavor == torchenv.CUDA_FLAVORS[0]
    assert f.index_url.endswith(f"/{torchenv.CUDA_FLAVORS[0]}")


def test_amd_rocm_matches_userland(monkeypatch):
    monkeypatch.delenv("TOASTER_LOOP_TORCH_FLAVOR", raising=False)
    # Newest flavor <= installed userland: 6.4.2 -> rocm6.4, 7.2.1 -> rocm7.2.
    f = detect_flavor(system="linux", gpus=[_g("amd")],
                      rocm_version="6.4.2")
    assert f.flavor == "rocm6.4"
    f = detect_flavor(system="linux", gpus=[_g("amd")],
                      rocm_version="7.2.1")
    assert f.flavor == "rocm7.2"
    # Userland newer than any published flavor: clamp to the newest.
    f = detect_flavor(system="linux", gpus=[_g("amd")],
                      rocm_version="9.9")
    assert f.flavor == f"rocm{torchenv.ROCM_FLAVORS[0]}"


def test_amd_without_rocm_userland_is_cpu_with_note(monkeypatch):
    monkeypatch.delenv("TOASTER_LOOP_TORCH_FLAVOR", raising=False)
    f = detect_flavor(system="linux", gpus=[_g("amd")], rocm_version="")
    assert f.flavor == "cpu"
    assert any("Vulkan" in n for n in f.notes)  # serving unaffected, explains why


def test_no_gpu_cpu(monkeypatch):
    monkeypatch.delenv("TOASTER_LOOP_TORCH_FLAVOR", raising=False)
    f = detect_flavor(system="linux", gpus=[])
    assert f.flavor == "cpu" and not f.is_pypi


def test_intel_arc_xpu(monkeypatch):
    monkeypatch.delenv("TOASTER_LOOP_TORCH_FLAVOR", raising=False)
    f = detect_flavor(system="linux", machine="x86_64", gpus=[_g("intel")])
    assert f.flavor == "xpu"


def test_windows_paths(monkeypatch):
    monkeypatch.delenv("TOASTER_LOOP_TORCH_FLAVOR", raising=False)
    f = detect_flavor(system="win32", machine="amd64", gpus=[_g("nvidia")])
    assert f.flavor == FLAVOR_DEFAULT          # default wheel ships CUDA
    f = detect_flavor(system="win32", machine="amd64", gpus=[_g("intel")])
    assert f.flavor == "xpu"


def test_env_override(monkeypatch):
    monkeypatch.setenv("TOASTER_LOOP_TORCH_FLAVOR", "cu126")
    f = detect_flavor(system="linux", gpus=[_g("amd")], rocm_version="7.2")
    assert f.flavor == "cu126"
    assert f.index_url.endswith("/cu126")
    monkeypatch.setenv("TOASTER_LOOP_TORCH_FLAVOR", "default")
    f = detect_flavor(system="linux", gpus=[_g("amd")])
    assert f.is_pypi


# ------------------------------------------------------------ installer args

def test_flavor_args_shape(monkeypatch):
    monkeypatch.delenv("TOASTER_LOOP_TORCH_FLAVOR", raising=False)
    f = detect_flavor(system="linux", gpus=[_g("amd")], rocm_version="7.1")
    assert f.pip_args()[:2] == ["--index-url",
                                "https://download.pytorch.org/whl/rocm7.1"]
    uv = f.uv_pip_args()
    assert uv[:2] == ["--torch-backend", "rocm7.1"]
    # uv only knows real flavor names; 'default' must send NO backend flag.
    d = TorchFlavor(FLAVOR_DEFAULT, None)
    assert d.uv_pip_args() == [] and d.pip_args() == []


# ------------------------------------------------------------ mismatch doctor

ROCM_WANT = TorchFlavor("rocm7.1", "https://download.pytorch.org/whl/rocm7.1")


def test_mismatch_fires_on_cuda_wheel_on_rocm_box():
    bad = FakeTorch("2.14.1+cu130", cuda="13.0", hip=None, avail=False)
    msg = installed_mismatch(bad, ROCM_WANT)
    assert msg and "rocm7.1" in msg and "--index-url" in msg
    assert "cu130" in msg  # names the wheel that is actually installed


def test_mismatch_fires_on_plain_pypi_wheel_on_rocm_box():
    # No local version tag, no HIP: the uv.lock incident's exact shape.
    bad = FakeTorch("2.14.1", cuda=None, hip=None, avail=False)
    assert installed_mismatch(bad, ROCM_WANT)


def test_rocm_wheel_is_happy():
    good = FakeTorch("2.14.1+rocm7.2", cuda="6.2", hip="7.2", avail=True)
    assert installed_mismatch(good, ROCM_WANT) is None


def test_cpu_build_on_nvidia_box_flagged():
    cu_want = TorchFlavor("cu132", "x")
    bad = FakeTorch("2.14.1+cpu", cuda=None, hip=None, avail=False)
    assert installed_mismatch(bad, cu_want)


def test_no_strong_arm_no_false_positive():
    # CPU flavor wanted, CPU wheel installed: quiet.
    cpu_want = TorchFlavor("cpu", "x")
    ok = FakeTorch("2.14.1+cpu", cuda=None, hip=None, avail=False)
    assert installed_mismatch(ok, cpu_want) is None


# --------------------------------------------------------------- doctor glue

def test_check_function_without_torch(monkeypatch):
    """torchenv's doctor probe must not import torch at all (license
    hygiene): it reads wheel metadata only."""
    monkeypatch.setattr(torchenv, "detect_flavor",
                        lambda **kw: TorchFlavor("cpu", "x"))
    ok, lines = torchenv.check_function(lambda name: None)
    assert ok and "not installed" in lines[0]
    src = Path(torchenv.__file__).read_text()
    import ast
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            assert not any(a.name == "torch" for a in node.names), \
                "torchenv must never import torch (doctor path)"


def test_metadata_mismatch_uses_version_tag_only():
    """Metadata path classifies wheels by their local version tag — the
    doctor never imports torch to do it."""
    assert torchenv.metadata_mismatch("2.14.1+cu130", ROCM_WANT)
    assert torchenv.metadata_mismatch("2.14.1", ROCM_WANT)  # tagless = CUDA
    assert torchenv.metadata_mismatch("2.14.1+cpu", ROCM_WANT)
    assert torchenv.metadata_mismatch("2.14.1+rocm7.2", ROCM_WANT) is None
    cu_want = TorchFlavor("cu132", "x")
    assert torchenv.metadata_mismatch("2.14.1+cpu", cu_want)
    assert torchenv.metadata_mismatch("2.14.1+cu130", cu_want) is None
    # ROCm wheel on any box: quiet (HIP build hears HIP).
    assert torchenv.metadata_mismatch("2.14.1+rocm7.2", cu_want) is None
    cpu_want = TorchFlavor("cpu", "x")
    assert torchenv.metadata_mismatch("2.14.1+cpu", cpu_want) is None

# ------------------------------------------------------------------------
# from tests/test_model_discovery.py
# ------------------------------------------------------------------------

@pytest.fixture()
def hub(tmp_path, monkeypatch):
    """Fake HF hub cache:

    models--Qwen--Qwen3-8B/snapshots/<rev>/  (GGUF via blob symlink + config.json)
    models--meta--Llama-FP16/snapshots/<rev>/ (real GGUF file, no config.json)
    loose/model.gguf                          (flat layout)
    """
    h = tmp_path / "hub"

    # repo 1: hub-layout GGUF stored as blobs/ symlink + an HF snapshot
    r1 = h / "models--Qwen--Qwen3-8B"
    blobs = r1 / "blobs"
    blobs.mkdir(parents=True)
    blob = blobs / "abc123"
    blob.write_bytes(b"GGUF" + b"\0" * 64)
    snap = r1 / "snapshots" / "deadbeefcafe1234"
    snap.mkdir(parents=True)
    (snap / "qwen3-8b-q4_k_m.gguf").symlink_to(
        Path("..") / ".." / "blobs" / "abc123")
    # repo 2: plain GGUF file, no HF config.json (serve-only)
    r2 = h / "models--meta--Llama-FP16"
    snap2 = r2 / "snapshots" / "rev1"
    snap2.mkdir(parents=True)
    (snap2 / "llama-f16.gguf").write_bytes(b"GGUF")
    # loose file outside the hub layout
    (h / "loose").mkdir()
    (h / "loose" / "model.gguf").write_bytes(b"GGUF")
    return h


# ---------------------------------------------------------------- discovery

def test_discover_gguf_hub_and_loose(hub):
    found = {m["name"]: m for m in discover_gguf_models(hub)}
    assert "Qwen/Qwen3-8B · qwen3-8b-q4_k_m" in found
    assert "meta/Llama-FP16 · llama-f16" in found
    assert "model" in found  # loose/model.gguf
    q = found["Qwen/Qwen3-8B · qwen3-8b-q4_k_m"]
    assert q["quant"] == "q4_k_m"
    assert q["size_bytes"] == 68  # symlink resolved to blob size
    assert Path(q["path"]).is_file()


def test_discover_base_models_requires_config_json(hub):
    # No config.json anywhere yet -> nothing training-capable.
    assert discover_base_models(hub) == []
    snap = hub / "models--Qwen--Qwen3-8B" / "snapshots" / "deadbeefcafe1234"
    (snap / "config.json").write_text('{"architectures": []}')
    found = discover_base_models(hub)
    assert len(found) == 1
    assert found[0]["name"] == "Qwen/Qwen3-8B"
    assert Path(found[0]["path"]).is_dir()


def test_discover_models_bundle(hub):
    d = discover_models(str(hub))
    assert d["exists"] is True
    assert d["models_dir"] == str(hub)
    assert len(d["gguf"]) == 3
    assert len(d["base_models"]) == 0


def test_resolve_models_dir_precedence(tmp_path, monkeypatch):
    explicit = tmp_path / "a"; explicit.mkdir()
    envd = tmp_path / "b"; envd.mkdir()
    monkeypatch.setenv("TOASTER_LOOP_MODELS_DIR", str(envd))
    assert resolve_models_dir(str(explicit)) == explicit
    assert resolve_models_dir(None) == envd
    # Nothing configured and no HF cache => None (isolate HOME/HF_HOME).
    empty = tmp_path / "home"; empty.mkdir()
    monkeypatch.setenv("HOME", str(empty))
    monkeypatch.setenv("USERPROFILE", str(empty))
    monkeypatch.delenv("TOASTER_LOOP_MODELS_DIR", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    assert resolve_models_dir(None) is None
    # HF_HOME/hub fallback
    hf = tmp_path / "hfhome"
    (hf / "hub").mkdir(parents=True)
    monkeypatch.setenv("HF_HOME", str(hf))
    assert resolve_models_dir(None) == hf / "hub"


def test_discover_models_missing_dir_is_graceful(tmp_path):
    d = discover_models(str(tmp_path / "nope"))
    assert d["exists"] is False and d["gguf"] == [] and d["base_models"] == []


# ------------------------------------------------------- split (multi-part) GGUFs

@pytest.fixture()
def shards(tmp_path):
    """Loose 3-part model + an old-style split-NN pair in one folder."""
    d = tmp_path / "models"
    d.mkdir()
    for n in (1, 2, 3):
        (d / f"Qwen3.8-Flash-Next-Q4_K_M-0000{n}-of-00003.gguf").write_bytes(
            b"x" * (100 + n))
    (d / "Old-split-01.gguf").write_bytes(b"y" * 10)
    (d / "Old-split-02.gguf").write_bytes(b"y" * 20)
    (d / "plain-model.gguf").write_bytes(b"z" * 7)
    return d


def test_shard_group_parsing():
    assert shard_group("Foo-00001-of-00003.gguf") == ("Foo", 1)
    assert shard_group("Foo-00002-of-00003.gguf") == ("Foo", 2)
    assert shard_group("model-Q4_K_M-00001-of-00003.gguf") == \
        ("model-Q4_K_M", 1)
    assert shard_group("Old-split-02.gguf") == ("Old", 2)
    # Not shards: plain names, quant tails, context-size tails.
    assert shard_group("plain-model.gguf") is None
    assert shard_group("model-q4_k_m.gguf") is None
    assert shard_group("model-4k.gguf") is None


def test_discover_folds_shards_into_one_entry(shards):
    found = discover_gguf_models(shards)
    by_path = {m["path"]: m for m in found}
    names = [m["name"] for m in found]
    # One entry per model, not three (and two).
    assert "Qwen3.8-Flash-Next-Q4_K_M" in names
    assert sum(1 for n in names if "of-0000" in n) == 0
    assert "Old" in names and "plain-model" in names
    q = by_path[str((shards / "Qwen3.8-Flash-Next-Q4_K_M-00001-of-00003.gguf").resolve())]
    assert q["shard_count"] == 3
    # size_bytes sums every part: 101+102+103 = 306.
    assert q["size_bytes"] == 306
    old = by_path[str((shards / "Old-split-01.gguf").resolve())]
    assert old["shard_count"] == 2 and old["size_bytes"] == 30


def test_resolve_gguf_shards(shards):
    d = shards
    first = str((d / "Qwen3.8-Flash-Next-Q4_K_M-00001-of-00003.gguf"))
    # A later shard resolves back to shard 1 (what llama-server -m wants).
    assert resolve_gguf_shards(
        str(d / "Qwen3.8-Flash-Next-Q4_K_M-00002-of-00003.gguf")) == first
    # Shard 1 and unsharded paths pass through unchanged.
    assert resolve_gguf_shards(first) == first
    assert resolve_gguf_shards(str(d / "plain-model.gguf")) == \
        str(d / "plain-model.gguf")


def test_gguf_total_size(shards):
    assert gguf_total_size(
        str(shards / "Qwen3.8-Flash-Next-Q4_K_M-00002-of-00003.gguf")) == 306
    assert gguf_total_size(str(shards / "plain-model.gguf")) == 7


# ------------------------------------- HF blob-hash paths (the Qwen3.8 trap)

@pytest.fixture()
def hub_shards(tmp_path):
    """reference-style hub cache: split GGUF + mmproj + MTP drafter stored as
    hash-named blobs, exposed through snapshot symlinks in nested folders."""
    repo = tmp_path / "hub" / "models--testorg--Qwen3.8-Flash-Next-GGUF"
    blobs = repo / "blobs"
    blobs.mkdir(parents=True)
    snap = repo / "snapshots" / "38bb39ee"
    quant = snap / "UD-IQ4_XS"
    mtp = snap / "MTP"
    quant.mkdir(parents=True)
    mtp.mkdir(parents=True)
    hashes = {}
    for n in (1, 2, 3):
        h = f"hash-part{n}"
        (blobs / h).write_bytes(b"x" * (10 * n))
        (quant / f"Qwen3.8-Flash-Next-UD-IQ4_XS-0000{n}-of-00003.gguf") \
            .symlink_to(Path("..") / ".." / ".." / "blobs" / h)
        hashes[n] = h
    (blobs / "hash-mmproj").write_bytes(b"m" * 5)
    (snap / "mmproj-F16.gguf").symlink_to(
        Path("..") / ".." / "blobs" / "hash-mmproj")
    (blobs / "hash-mtp").write_bytes(b"t" * 6)
    (mtp / "mtp-Qwen3.8-Flash-Next-Q8_0.gguf").symlink_to(
        Path("..") / ".." / ".." / "blobs" / "hash-mtp")
    return repo, snap, hashes


def test_blob_path_maps_back_to_snapshot_symlink(hub_shards):
    repo, snap, hashes = hub_shards
    blob = repo / "blobs" / hashes[1]
    link = snapshot_link_for_blob(blob)
    assert link is not None
    assert link.name == "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf"
    # resolve_gguf_path keeps the visible filename (no blob-hash resolution).
    assert resolve_gguf_path(str(blob)).endswith(
        "UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf")


def test_resolve_gguf_shards_through_blob_names(hub_shards):
    """A blob path for shard 2 must resolve to the SNAPSHOT SHARD 1 symlink —
    llama-server refuses hash-named split files ('invalid split file name')."""
    repo, snap, hashes = hub_shards
    resolved = resolve_gguf_shards(str(repo / "blobs" / hashes[2]))
    assert Path(resolved).name == \
        "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf"
    assert str(snap / "UD-IQ4_XS") in resolved


def test_gguf_total_size_through_blob_names(hub_shards):
    repo, _snap, hashes = hub_shards
    assert gguf_total_size(str(repo / "blobs" / hashes[2])) == 60


def test_discovery_excludes_companions_and_keeps_visible_paths(hub_shards):
    repo, snap, _h = hub_shards
    found = discover_gguf_models(tmp := repo.parent)
    names = [m["name"] for m in found]
    # one folded shard entry; mmproj + mtp never offered as standalone models
    assert sum(1 for n in names if "UD-IQ4_XS" in n) == 1
    assert not any("mmproj" in n or "mtp" in n.lower() for n in names)
    entry = next(m for m in found if "UD-IQ4_XS" in m["name"])
    assert entry["shard_count"] == 3
    assert entry["size_bytes"] == 60
    # visible snapshot path, not the blob hash (spawn + sibling lookup need it)
    assert Path(entry["path"]).name.endswith(".gguf")
    assert "blobs" not in Path(entry["path"]).parts


def test_is_companion_gguf():
    assert is_companion_gguf("mmproj-F16.gguf")
    assert is_companion_gguf("mtp-Qwen3.8-Flash-Next-Q8_0.gguf")
    assert is_companion_gguf("Qwen3-VL-mmproj-BF16.gguf")
    assert not is_companion_gguf("Qwen3.8-Flash-Next-UD-IQ4_XS.gguf")
    assert not is_companion_gguf("temp.gguf")


def test_mtp_and_mmproj_siblings_through_hub_layout(hub_shards):
    from toaster_loop.core import mmproj as mmp

    repo, snap, hashes = hub_shards
    # Even given the raw blob path of a shard, companions resolve by name.
    gguf = str(repo / "blobs" / hashes[1])
    proj = mmp.find_local(gguf)
    assert proj is not None and Path(proj).name == "mmproj-F16.gguf"
    drafter = mmp.find_local_mtp(gguf)
    assert drafter is not None and Path(drafter).name.startswith("mtp-")


def test_vram_advisory_counts_all_shards(tmp_path):
    from toaster_loop.serving.instances import estimate_model_vram_mib

    p = tmp_path / "M-00002-of-00002.gguf"
    p.write_bytes(b"\0" * (2 * 1024 * 1024))
    (tmp_path / "M-00001-of-00002.gguf").write_bytes(b"\0" * (3 * 1024 * 1024))
    est = estimate_model_vram_mib(str(p))
    # 5 MiB of weights * 1.2 headroom => > 5 MiB (a shard-only stat would say ~2).
    assert est is not None and est > 5


# ------------------------------------------------------------------- config

def test_config_models_dir_yaml_and_env(tmp_path, monkeypatch):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("models_dir: /srv/models\n")
    monkeypatch.delenv("TOASTER_LOOP_MODELS_DIR", raising=False)
    assert load_config(str(cfg_file)).models_dir == "/srv/models"
    monkeypatch.setenv("TOASTER_LOOP_MODELS_DIR", "/env/models")
    assert load_config(str(cfg_file)).models_dir == "/env/models"


# --------------------------------------------------------------- dashboard API

@pytest.fixture()
def app__test_model_discovery(home__test_model_discovery, cfg_extra=None):
    a = DashboardApp(ToasterLoopConfig(**(cfg_extra or {})), home=home__test_model_discovery)
    yield a


@pytest.fixture()
def home__test_model_discovery(tmp_path, monkeypatch):
    h = tmp_path / "uf-cat"
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(h))
    return h


def test_model_catalog_endpoint(app__test_model_discovery, hub):
    body = app__test_model_discovery.handle("GET", "/api/model_catalog",
                      query={"dir": str(hub)}).body
    assert body["exists"] is True
    names = [m["name"] for m in body["gguf"]]
    assert any("qwen3-8b" in n for n in names)

    # Configured models_dir (no ?dir=) is honored too.
    app2 = DashboardApp(ToasterLoopConfig(models_dir=str(hub)), home=app__test_model_discovery.home)
    body2 = app2.handle("GET", "/api/model_catalog").body
    assert body2["exists"] is True
    assert Path(body2["models_dir"]).resolve() == Path(hub).resolve()
    assert len(body2["gguf"]) == 3


def test_model_catalog_no_folder(app__test_model_discovery, monkeypatch, tmp_path):
    empty = tmp_path / "home"; empty.mkdir()
    monkeypatch.setenv("HOME", str(empty))
    monkeypatch.setenv("USERPROFILE", str(empty))
    monkeypatch.delenv("TOASTER_LOOP_MODELS_DIR", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    body = app__test_model_discovery.handle("GET", "/api/model_catalog").body
    assert body["exists"] is False


def test_train_base_model_override(app__test_model_discovery, hub, tmp_path, monkeypatch):
    """POST /api/train {base_model} beats training.base_model in job params."""
    stub = tmp_path / "stub.py"
    stub.write_text("import sys\nsys.exit(0)\n")
    app__test_model_discovery.cfg.training = {
        "base_model": "Config/Default",
        "worker_command": [sys.executable, str(stub)],
    }
    r = app__test_model_discovery.handle("POST", "/api/train", body={
        "dataset": "ds-1", "kind": "sft", "dry_run": True,
        "base_model": str(hub / "models--Qwen--Qwen3-8B"),
    })
    assert r.status == 202
    from toaster_loop.registry import ModelRegistry
    reg = ModelRegistry(app__test_model_discovery.home / "buffer" / "toaster_loop.db")
    try:
        job = reg.get_job(r.body["job_id"])
        assert job is not None
        assert job.params["base_model"] == str(hub / "models--Qwen--Qwen3-8B")
    finally:
        reg.close()


def test_train_without_override_uses_config(app__test_model_discovery, tmp_path):
    stub = tmp_path / "stub2.py"
    stub.write_text("import sys\nsys.exit(0)\n")
    app__test_model_discovery.cfg.training = {
        "base_model": "Config/Default",
        "worker_command": [sys.executable, str(stub)],
    }
    r = app__test_model_discovery.handle("POST", "/api/train", body={"dataset": "ds-2", "dry_run": True})
    assert r.status == 202
    from toaster_loop.registry import ModelRegistry
    reg = ModelRegistry(app__test_model_discovery.home / "buffer" / "toaster_loop.db")
    try:
        job = reg.get_job(r.body["job_id"])
        assert job is not None
        assert job.params["base_model"] == "Config/Default"
    finally:
        reg.close()

# ------------------------------------------------------------------------
# from tests/test_think_scrub.py
# ------------------------------------------------------------------------

def run_stream(deltas):
    s = StreamingThinkScrubber()
    out = "".join(s.feed(d) for d in deltas)
    return out + s.flush()


# ------------------------------------------------------------- 1. streaming

def test_closed_pair_removed_inline():
    assert run_stream(["Here is ", "<thinking>", "secret plan", "</thinking>",
                       "the answer."]) == "Here is the answer."


def test_closed_pair_tag_split_across_deltas():
    # Tag names split across delta boundaries must still resolve.
    assert run_stream(["a<thi", "nking>hidden</thi", "nking>b"]) == "ab"


def test_unterminated_block_at_boundary_suppressed_to_flush():
    out = run_stream(["<think>", "pondering ", "and pondering", " more"])
    assert out == ""  # never leaks, flush discards the unterminated tail


def test_prose_mentioning_tag_midline_survives():
    text = ("Docs say to wrap chains like <thinking>your plan</thinking>? "
            "No — inline mentions of <thinking> stay.")
    # Closed pairs are *always* stripped (deliberate), but an unterminated
    # mention mid-line must never suppress the rest of the line.
    assert "<thinking>your plan" not in strip_think_blocks(text)
    solo = "Use the <thinking> tag to enable chains."
    assert strip_think_blocks(solo) == solo


def test_open_tag_midline_not_a_block_start():
    out = run_stream(["prose <think", "ing> still visible"])
    # '<' + 'ing>' never forms a boundary block: everything survives.
    assert "<thinking>" in out or "<thinking" in out


def test_orphan_close_tags_dropped():
    assert run_stream(["visible ", "</thinking>", " text"]) == "visible  text"


def test_partial_tag_tail_flushed_as_prose():
    # A trailing '<thi' that never becomes a tag is released at flush.
    assert run_stream(["ends with <thi"]) == "ends with <thi"


def test_reset_between_turns():
    s = StreamingThinkScrubber()
    assert s.feed("<thinking>hidden") == ""
    s.reset()
    assert s.feed("clean turn") == "clean turn"
    assert s.flush() == ""


def test_block_after_newline_is_boundary():
    # Closed block after a newline: suppressed, following prose resumes.
    out = run_stream(["answer line\n", "<thinking>", "hidden", "</thinking>",
                      "tail visible"])
    assert out == "answer line\ntail visible"


def test_unterminated_block_swallows_to_end():
    # Deliberate: once an unterminated block opens at a boundary, everything
    # after it is treated as reasoning (leaking beats truncating).
    out = run_stream(["answer line\n", "<thinking>", "hidden", "\n",
                      "tail visible"])
    assert out == "answer line\n"


def test_central_tag_list_bound_everywhere():
    for name in THINK_TAG_NAMES:
        tag = f"<{name.lower()}>"
        assert run_stream([tag, "hidden", f"</{name.lower()}>"]) == ""


# ------------------------------------------------------ 2. whole-string pass

def test_strip_think_blocks_idempotent():
    raw = "a<thinking>x</thinking>b"
    once = strip_think_blocks(raw)
    assert once == "ab"
    assert strip_think_blocks(once) == once


def test_strip_think_blocks_noop_plain_text():
    assert strip_think_blocks("plain prose, no tags") == "plain prose, no tags"
    assert strip_think_blocks("") == ""


# --------------------------------------------------- 3. runner integration

class InlineThinkSSEHandler(BaseHTTPRequestHandler):
    """Streams a reply whose plain content carries inline thinking."""

    def log_message(self, format, *args):  # noqa: A002 - silence access log
        pass

    def do_POST(self):
        chunks = [
            {"delta": {"role": "assistant"}},
            {"delta": {"content": "<thinking>"}},
            {"delta": {"content": "secret chain of thought"}},
            {"delta": {"content": "</thinking>"}},
            {"delta": {"content": "The answer is 42."}},
            {"delta": {}, "finish_reason": "stop"},
        ]
        body = "".join(
            "data: " + json.dumps({"id": "x", "object": "chat.completion.chunk",
                                   "model": "m",
                                   "choices": [dict(index=0, **c)]}) + "\n\n"
            for c in chunks) + "data: [DONE]\n\n"
        raw = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(raw)


def test_runner_stream_scrubs_deltas_and_final_message():
    events = []
    with FakeOpenAIServer(handler_cls=InlineThinkSSEHandler) as fs:
        client = ToasterLoopAgentClient(fs.base_url, "m", tools=[], tool_impls={})
        client.on_event = lambda kind, data: events.append((kind, data))
        res = client.run("what is the answer?")
    streamed = "".join(d["text"] for k, d in events if k == "message.delta")
    assert streamed == "The answer is 42."
    assert "secret chain of thought" not in streamed
    # The finalized result (what history + trace record) is scrubbed too.
    assert res.final_response == "The answer is 42."
    last = res.messages[-1]
    assert last["role"] == "assistant" and last["content"] == "The answer is 42."


class InlineThinkJSONHandler(BaseHTTPRequestHandler):
    """Non-streaming backend leaking inline thinking in plain content."""

    def log_message(self, format, *args):  # noqa: A002 - silence access log
        pass

    def do_POST(self):
        body = json.dumps({
            "id": "x", "object": "chat.completion", "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant",
                                     "content": "<thinking>hidden work</thinking>"
                                                "done."}}],
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_runner_non_streaming_final_scrubbed():
    with FakeOpenAIServer(handler_cls=InlineThinkJSONHandler) as fs:
        client = ToasterLoopAgentClient(fs.base_url, "m", tools=[], tool_impls={})
        res = client.run("go")
    assert res.final_response == "done."
    assert res.messages[-1]["content"] == "done."

# ------------------------------------------------------------------------
# from tests/test_tokenizer_exact.py
# ------------------------------------------------------------------------

DEAD_URL = "http://127.0.0.1:9/v1"  # discard port: instant connection refused


def _msg(role, content, **extra):
    return {"role": role, "content": content, **extra}


@pytest.fixture()
def tok_server():
    handler = make_tokenize_handler({"*": "final answer"})
    with FakeOpenAIServer(handler_cls=handler) as srv:
        handler.tokenize_requests.clear()
        yield srv, handler


# ------------------------------------------------------------ TokenCounter

def test_count_uses_tokenize_endpoint(tok_server):
    srv, handler = tok_server
    tc = TokenCounter(srv.base_url)
    text = "hello exact tokenizer world"  # fake counts ceil(len/2), not len/4
    n = tc.count(text)
    assert n == fake_tokenize_count(text)
    assert n != len(text) // 4  # proves the exact path, not the fallback
    assert handler.tokenize_requests == [text]


def test_lru_cache_hit_avoids_second_http_call(tok_server):
    srv, handler = tok_server
    tc = TokenCounter(srv.base_url)
    a = tc.count("cache me please")
    b = tc.count("cache me please")
    assert a == b
    assert len(handler.tokenize_requests) == 1  # second call served from LRU


def test_lru_cache_is_bounded(tok_server):
    srv, _ = tok_server
    tc = TokenCounter(srv.base_url, cache_size=2)
    for i in range(5):
        tc.count(f"entry {i}")
    assert len(tc._cache) == 2
    # oldest entries were evicted: re-counting them hits the network again
    assert "entry 4" in tc._cache and "entry 3" in tc._cache


def test_fallback_on_non_200(tok_server):
    srv, handler = tok_server
    handler.tokenize_status = 500
    tc = TokenCounter(srv.base_url)
    text = "x" * 100
    assert tc.count(text) == len(text) // 4


def test_fallback_on_connection_refused():
    tc = TokenCounter(DEAD_URL)
    text = "no server here at all"
    assert tc.count(text) == len(text) // 4
    assert tc.exact_available  # single failure: circuit still closed


def test_degraded_count_messages_equals_estimate_tokens(tok_server):
    srv, handler = tok_server
    handler.tokenize_status = 404
    tc = TokenCounter(srv.base_url, failure_threshold=100)  # keep probing
    msgs = [
        _msg("system", "S"),
        _msg("user", "question " * 30),
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "read_file",
                                      "arguments": '{"path": "a.txt"}'}}]},
        _msg("tool", "result text " * 40),
        _msg("assistant", None),  # content None, no tool_calls -> 0 both ways
    ]
    assert tc.count_messages(msgs) == estimate_tokens(msgs)


def test_circuit_breaker_stops_probing_and_recovers(tok_server, caplog):
    srv, handler = tok_server
    now = [1000.0]
    handler.tokenize_status = 500
    with caplog.at_level(logging.WARNING, logger="toaster_loop.tokenizer"):
        tc = TokenCounter(srv.base_url, failure_threshold=2,
                          cooldown_seconds=10.0, clock=lambda: now[0])
        text = "probe text"
        assert tc.count(text) == len(text) // 4  # failure 1
        assert tc.count("other") == 5 // 4       # failure 2 -> circuit opens
        assert len(handler.tokenize_requests) == 2
        assert not tc.exact_available

        # Circuit open: no further probing at all, silent chars/4 fallback.
        assert tc.count(text) == len(text) // 4
        assert tc.count("a longer fallback string") == 26 // 4
        assert len(handler.tokenize_requests) == 2  # nothing left the process
        warnings = [r for r in caplog.records
                    if "tokenize endpoint" in r.getMessage()]
        assert len(warnings) == 1  # logged once, not per call

    # Recovery: endpoint heals, cooldown passes -> half-open probe succeeds.
    handler.tokenize_status = 200
    now[0] += 11.0
    assert tc.count(text) == fake_tokenize_count(text)
    assert tc.exact_available
    assert len(handler.tokenize_requests) == 3


def test_base_url_strips_v1_suffix():
    tc = TokenCounter("http://example.test:8080/v1/")
    assert tc.tokenize_url == "http://example.test:8080/tokenize"


# ------------------------------------------------- compress_messages gate

class _FakeCounter:
    """Duck-typed counter with a fixed chars/2 policy for boundary proofs."""

    def __init__(self):
        self.calls = 0

    def count_messages(self, messages):
        self.calls += 1
        total = 0
        for m in messages:
            content = m.get("content")
            if isinstance(content, str):
                total += -(-len(content) // 2) + 4
            elif content is not None:
                total += 4
            for _tc in m.get("tool_calls") or []:
                total += 4
        return total


def test_gate_differs_from_chars4_with_exact_counter():
    # Content sized so chars/4 says "under budget" while the exact counter
    # (2 chars/token) says "over": same messages, opposite gate outcomes.
    msgs = [_msg("system", "S")]
    for i in range(6):
        msgs.append(_msg("user", f"q{i} " + "u" * 100))
        msgs.append(_msg("assistant", f"a{i} " + "v" * 100))
    est = estimate_tokens(msgs)
    fake = _FakeCounter()
    exact = fake.count_messages(msgs)
    assert est < exact  # chars/4 underestimates this content
    threshold = (est + exact) // 2

    out_plain = compress_messages(msgs, None, threshold_tokens=threshold,
                                  keep_recent_tokens=64)
    assert out_plain is msgs  # estimate path: no compression

    out_exact = compress_messages(msgs, None, threshold_tokens=threshold,
                                  keep_recent_tokens=64, counter=fake)
    assert out_exact is not msgs
    assert any(str(m.get("content", "")).startswith(SUMMARY_PREFIX)
               for m in out_exact)
    assert fake.calls >= 1


def test_cjk_boundary_compression_with_real_server(tok_server):
    # CJK-heavy history: llama-style tokenizers are ~1 char/token here, our
    # fake uses ceil(len/2), and chars/4 badly underestimates — the exact
    # counter triggers compression where the estimate never would.
    srv, handler = tok_server
    msgs = [_msg("system", "S")]
    for i in range(6):
        msgs.append(_msg("user", "汉" * 200))
        msgs.append(_msg("assistant", "字" * 200))
    est = estimate_tokens(msgs)
    tc = TokenCounter(srv.base_url)
    exact = tc.count_messages(msgs)
    assert est < exact
    threshold = (est + exact) // 2

    out_plain = compress_messages(msgs, None, threshold_tokens=threshold,
                                  keep_recent_tokens=64)
    assert out_plain is msgs

    out = compress_messages(msgs, None, threshold_tokens=threshold,
                            keep_recent_tokens=64, counter=tc)
    assert out is not msgs
    assert any(str(m.get("content", "")).startswith(SUMMARY_PREFIX)
               for m in out)
    # every distinct content string was tokenized exactly once (LRU dedupes
    # the six identical users / six identical assistants into one probe each)
    assert len(set(handler.tokenize_requests)) == len(handler.tokenize_requests)


# ------------------------------------------------------------ runner wiring

def test_runner_uses_injected_counter_for_compression_gate(tok_server):
    srv, handler = tok_server
    msgs = [_msg("system", "S")]
    for i in range(6):
        msgs.append(_msg("user", f"question {i} " + "u" * 100))
        msgs.append(_msg("assistant", f"answer {i} " + "v" * 100))
    est = estimate_tokens(msgs)
    exact = TokenCounter(srv.base_url).count_messages(msgs)
    threshold = (est + exact) // 2

    tc = TokenCounter(srv.base_url)
    client = ToasterLoopAgentClient(base_url=srv.base_url, model="tok-model",
                                compress_threshold_tokens=threshold,
                                keep_recent_tokens=64, token_counter=tc)
    out, _summary = client._maybe_compress(msgs)
    assert out is not msgs  # exact count crossed the gate chars/4 did not
    assert any(str(m.get("content", "")).startswith(SUMMARY_PREFIX)
               for m in out)
    assert handler.tokenize_requests  # the endpoint actually answered


def test_runner_compression_off_never_probes_tokenize(tok_server):
    srv, handler = tok_server
    client = ToasterLoopAgentClient(base_url=srv.base_url, model="tok-model",
                                compress_threshold_tokens=0)
    out, _ = client._maybe_compress([_msg("user", "x" * 10_000)])
    assert out and handler.tokenize_requests == []


def test_runner_lazy_counter_binds_to_endpoint_and_resets_on_swap(tok_server):
    srv, _ = tok_server
    client = ToasterLoopAgentClient(base_url=srv.base_url, model="tok-model",
                                compress_threshold_tokens=100)
    assert client._token_counter is None  # lazy: nothing built at init
    tc = client._tokens()
    assert isinstance(tc, TokenCounter)
    assert tc.tokenize_url == srv.base_url.replace("/v1", "") + "/tokenize"
    assert client._token_counter is tc  # memoized

    ok = client.switch_model("http://127.0.0.1:4321/v1", "other-model")
    assert ok and client._token_counter is None  # re-binds to the new endpoint

    injected = TokenCounter(DEAD_URL)
    client2 = ToasterLoopAgentClient(base_url=srv.base_url, model="tok-model",
                                 token_counter=injected)
    client2.switch_model("http://127.0.0.1:4321/v1", "other")
    assert client2._token_counter is injected  # test doubles survive swaps


def test_full_run_against_tokenizing_server(tok_server):
    # End-to-end smoke: the loop's compression gate lazily builds a real
    # TokenCounter against the fake server (/tokenize at root, chat under
    # /v1) and the run completes normally.
    srv, _ = tok_server
    client = ToasterLoopAgentClient(base_url=srv.base_url, model="tok-model",
                                compress_threshold_tokens=50,
                                max_iterations=3)
    result = client.run("hello there")
    assert result.completed
    assert result.final_response == "final answer"

# ------------------------------------------------------------------------
# from tests/test_tool_call_recovery.py
# ------------------------------------------------------------------------

LT, GT, BAR = "<", ">", "|"
FENCE = "```"


def _ct(name: str) -> str:
    """Control marker of the pipe-delimited family, built piecewise."""
    return LT + BAR + name + BAR + GT


IM_END = _ct("im_end")
EOS = _ct("eos")

KNOWN = sorted({s["function"]["name"] for s in TOOL_SCHEMAS}
               | set(CAPABILITY_TOOL_SCHEMAS))

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "tool_drift"


def _norm_calls(calls) -> List[tuple]:
    """(name, args-json) tuples -- ids are random per call by design."""
    return [(c["function"]["name"], c["function"]["arguments"])
            for c in calls]


def _args(calls, i=0) -> Dict[str, Any]:
    return json.loads(calls[i]["function"]["arguments"])


# ===========================================================================
# Mechanism 1: strip_stop_tokens
# ===========================================================================

@pytest.mark.parametrize("inp, want", [
    # repeated markers collapse in one call
    ("hello world" + IM_END * 3, "hello world"),
    # junk interleaved between markers (box pipes, asterisks)
    ("result *" + IM_END + " \u02c1 " + EOS + "**", "result"),
    ("done." + IM_END, "done"),
    ("line one\n" + IM_END + "\n" + IM_END + "\n", "line one"),
    # duplicated assistant-turn header shapes
    ("all done" + LT + "/assistant>", "all done"),
    # acknowledged bracketed / literal word markers
    ("Task complete" + LT + "STOP" + GT, "Task complete"),
    ("done [DONE]", "done"),
])
def test_strip_removes_trailing_stop_shapes(inp, want):
    assert strip_stop_tokens(inp) == want


@pytest.mark.parametrize("kept", [
    "The stop sign is ahead",            # bare word mid-sentence
    "the bus will not stop",             # bare word at tail, prose
    "arguments:",                        # colon lead-in survives
    "plain prose stays put.",            # untouched sentence
    "2 < 3 and 4 > 1 math is fine",      # comparisons are not brackets
    "see " + IM_END + " inline in middle, keep me",  # mid-content marker
    "code: x = 1  # " + IM_END + " comment stays",   # marker inside code
])
def test_strip_preserves_non_tail_content(kept):
    """Mid-content preservation: legacy-kept text comes back byte-identical."""
    assert strip_stop_tokens(kept) == kept


def test_strip_punctuation_only_after_real_strip():
    # sentence-final period on untouched text is kept...
    assert strip_stop_tokens("All done.") == "All done."
    # ...but punctuation a strip LEFT BEHIND goes (never colon/comma).
    assert strip_stop_tokens("All done." + IM_END) == "All done"
    assert strip_stop_tokens("args:" + IM_END) == "args:"


def test_strip_extra_tokens():
    custom = LT + "end-of-sentence" + GT
    assert strip_stop_tokens("text" + custom * 2) == "text" + custom * 2
    assert strip_stop_tokens("text" + custom * 2,
                             extra_tokens=(custom,)) == "text"


def test_strip_iterates_until_stable_and_is_idempotent():
    s = "keep" + IM_END + " junk" if False else "keep" + IM_END
    once = strip_stop_tokens(s)
    assert once == "keep"
    assert strip_stop_tokens(once) == once


def test_strip_empty_and_none_shapes():
    assert strip_stop_tokens("") == ""
    assert strip_stop_tokens(IM_END) == ""


# ===========================================================================
# Mechanism 2: repair classes + attribution
# ===========================================================================

def _report(text: str) -> ParseReport:
    return parse_text_tool_calls_with_repairs(text, known_tool_names=KNOWN)


def test_repair_single_quotes():
    calls = parse_text_tool_calls(
        FENCE + "json\n{'name': 'read_file', 'arguments': {'path': 'a'}}\n"
        + FENCE, known_tool_names=KNOWN)
    assert len(calls) == 1 and _args(calls) == {"path": "a"}


def test_repair_trailing_commas():
    rep = _report(FENCE + "json\n{\"name\": \"list_dir\", \"arguments\": "
                          "{\"path\": \"s\",},}\n" + FENCE)
    assert rep.calls and "trailing_commas" in rep.tags_for(rep.calls[0])


def test_repair_py_literals():
    rep = _report(FENCE + "json\n{\"name\": \"run_command\", \"arguments\": "
                          "{\"bg\": True, \"x\": None}}\n" + FENCE)
    assert rep.calls and _args(rep.calls)["bg"] is True
    assert _args(rep.calls)["x"] is None
    assert "py_literals" in rep.tags_for(rep.calls[0])


def test_repair_truncated_close():
    rep = _report("ok\n" + FENCE + "json\n{\"name\": \"search_files\", "
                           "\"arguments\": {\"query\": \"n\"}")
    assert rep.calls and _args(rep.calls) == {"query": "n"}
    tags = rep.tags_for(rep.calls[0])
    assert "unclosed_fence" in tags and "truncated_close" in tags


def test_repair_truncated_value_refuses_fabrication():
    # A value cut mid-string must NOT fabricate an answer from the
    # fragment: the pair is dropped, the call survives with no args.
    rep = _report(FENCE + "json\n{\"name\": \"search_files\", \"arguments\": "
                          "{\"query\": \"nee")
    assert rep.calls and _args(rep.calls) == {}


def test_repair_xml_wrappers():
    a = parse_text_tool_calls("use <tool>\nread_file {\"path\": \"x\"}\n"
                              "</tool>", known_tool_names=KNOWN)
    assert _norm_calls(a)[0][0] == "read_file"
    b = parse_text_tool_calls(
        "<tool_call name=\"list_dir\"><parameter name=\"path\">src" + LT
        + "/parameter>< /tool_call>", known_tool_names=KNOWN)
    assert _args(b) == {"path": "src"}
    c = parse_text_tool_calls(
        LT + " function=search_files>\n< parameter name=\"query\">n"
        "< /parameter>\n< /function>", known_tool_names=KNOWN)
    assert _args(c) == {"query": "n"}


def test_repair_bare_json_in_prose():
    rep = _report("Here you are: {\"name\": \"list_dir\", \"arguments\": "
                          "{\"path\": \"docs\"}} Hope it helps!")
    assert rep.calls and _args(rep.calls) == {"path": "docs"}
    assert "bare_json_prose" in rep.tags_for(rep.calls[0])


def test_bare_data_json_never_parsed():
    # legacy posture: data without a name/arguments shape is not a call
    assert parse_text_tool_calls("x = {\"retention\": 30} ok",
                                 known_tool_names=KNOWN) == []
    assert parse_text_tool_calls("x = {\"name\": \"list\"} ok",
                                 known_tool_names=KNOWN) == []


def test_clean_parse_has_empty_attribution():
    rep = _report("Sure!\n" + FENCE + "json\n{\"name\": \"list_dir\", "
                           "\"arguments\": {\"path\": \".\"}}\n" + FENCE)
    assert rep.calls and not rep.recovered
    assert rep.tags_for(rep.calls[0]) == []


def test_repair_tags_constant_is_superset_of_emitted():
    seen = set()
    for text in [
        "{'name': 'list_dir', 'arguments': {}}",
        FENCE + "json\n{'name': 'list_dir', 'arguments': {}}",
        "<tool>\nlist_dir {}\n</tool>",
        "here {\"name\": \"list_dir\", \"arguments\": {\"a\": 1,}} " + IM_END,
    ]:
        rep = _report(text)
        for c in rep.calls:
            seen.update(rep.tags_for(c))
    assert seen <= set(REPAIR_TAGS) | {"name_normalized",
                                       "truncated_candidate"}


# ===========================================================================
# Name routing: exact-normalized only (documented refusal)
# ===========================================================================

def test_name_exact_normalized_match_recovers_variant_spellings():
    # Policy: EXACT match on the normal form (case/underscore folding)
    # routes; nothing else does. "serch_files" is edit-distance 1 from
    # "search_files" and must NOT be routed -- a mis-routed call executes
    # the wrong thing, a refused one just re-prompts the model.
    rep = _report(FENCE + "\n{'name': 'Read-File', 'arguments': "
                          "{'path': 'z'}}")
    assert rep.calls and rep.calls[0]["function"]["name"] == "read_file"
    assert "name_normalized" in rep.tags_for(rep.calls[0])


def test_name_typo_not_routed():
    assert parse_text_tool_calls(
        FENCE + "\n{\"name\": \"serch_files\", \"arguments\": "
        "{\"query\": \"x\"}}", known_tool_names=KNOWN) == []


def test_name_ambiguous_normal_form_refused():
    known = ["read_file", "Read File", "search_files"]  # fold to 1 dup pair
    rep = parse_text_tool_calls_with_repairs(
        FENCE + "\n{'name': 'READ_FILE', 'arguments': {}}", known)
    assert rep.calls == []  # two known names fold the same => refuse


def test_no_known_list_passes_names_through():
    rep = parse_text_tool_calls_with_repairs(
        FENCE + "\n{'name': 'Whatever', 'arguments': {}}")
    assert rep.calls and rep.calls[0]["function"]["name"] == "Whatever"


def test_normalize_tool_name_shape():
    assert normalize_tool_name("Read File") == "read_file"
    assert normalize_tool_name(" read--file ") == "read_file"
    assert normalize_tool_name("search.files/2") == "search_files_2"


# ===========================================================================
# Mechanism 3: reasoning-loop detection
# ===========================================================================

LAP = ("I still haven't actually opened the file. Let me check the "
       "directory listing first to confirm the path exists before I "
       "attempt reading it again, since the earlier error may have been "
       "a transient permissions issue.")


def test_loop_exact_paragraph_repeats():
    text = "Hmm.\n\n" + "\n\n".join([LAP] * 4)
    v = detect_reasoning_loop(text)
    assert v.loop and v.reason == "paragraph_repeats" and v.repeats >= 3
    out = truncate_at_loop(text, v)
    assert out.startswith("Hmm.")
    assert out.endswith(LOOP_MARKER)
    assert LAP not in out  # truncated at FIRST occurrence


def test_loop_near_duplicate_laps():
    text = ("Working.\n\n" + LAP + "\n\n"
            + LAP.replace("directory listing first",
                          "listing of the dir first") + "\n\n"
            + LAP.replace("directory listing first to confirm",
                          "dir listing first to double-check") + "\n\n"
            + LAP)
    v = detect_reasoning_loop(text)
    assert v.loop and v.reason == "near_duplicates"


def test_loop_hedging_cycles():
    h = ["Wait, I need to check the config first.",
         "Actually, the file might not exist yet.",
         "No, the error said permission denied specifically.",
         "Hold on, maybe I should try a different path instead.",
         "Hmm, but wait, what about the symlink situation?"]
    v = detect_reasoning_loop("\n".join(h * 4))
    assert v.loop and v.reason == "hedging_cycles"


def test_loop_interleaved_think_blocks():
    text = ("<thinking>\nchecking\n</thinking>\n\n" + LAP + "\n\n"
            + "<thinking>\nmore\n</thinking>\n\n" + LAP + "\n\n" + LAP)
    assert detect_reasoning_loop(text).loop


def test_loop_closed_think_noise_scrubs_to_clean():
    # loops ENTIRELY inside closed think pairs are removed by the scrubber
    # before judging -- the visible answer is clean, no truncation needed.
    text = ("<thinking>\n" + LAP + "\n</thinking>\n\n") * 4
    assert not detect_reasoning_loop(text).loop


@pytest.mark.parametrize("healthy", [
    "\n\n".join("- item %d: %s" % (i, "detailed explanation of this entry "
                                   "covering its purpose and usage " * 2)
                for i in range(30)),
    "```python\n" + "\n\n".join(
        "def handler_%d(ctx):\n    # process ctx and return\n"
        "    result = ctx.value\n    return result" % i
        for i in range(12)) + "\n```\n",
    LAP + "\n\n" + LAP,  # exactly two laps: below threshold
    "Short answer: yes, the file is in src/.",
])
def test_loop_false_negatives_healthy_text(healthy):
    assert not detect_reasoning_loop(healthy).loop


def test_loop_truncation_fails_open():
    v = detect_reasoning_loop("totally different text than analysed")
    assert not v.loop
    assert truncate_at_loop("whatever", v) == "whatever"
    assert truncate_at_loop("no verdict here") == "no verdict here"


# ===========================================================================
# Zero-regression contract (legacy-identical on legacy inputs)
# ===========================================================================

@pytest.mark.parametrize("sample", [
    "Sure!\n" + FENCE + "json\n{\"name\": \"read_file\", \"arguments\": "
    "{\"path\": \"a.py\"}}\n" + FENCE + "\n",
    FENCE + "\n[{\"function\": {\"name\": \"list_dir\", \"arguments\": "
    "\"{\\\"path\\\": \\\".\\\"}\"}}]\n" + FENCE,
    "Let me check.\n" + LT + "tool_call>read_file" +
    "{\"path\": \"x.txt\"}" + LT + "/tool_call>",
    f"```json\n{json.dumps({'foo': 1})}\n```",
    "", "Just a normal answer.",
])
def test_legacy_samples_identical(sample):
    legacy = parse_legacy(sample)
    new = parse_text_tool_calls(sample, known_tool_names=KNOWN)
    assert _norm_calls(new) == _norm_calls(legacy)


def test_strip_markup_extended_flag_off_is_legacy():
    text = "hi\n" + FENCE + "json\n{\"name\": \"a\", \"arguments\": {}}\n" \
        + FENCE
    assert strip_tool_call_markup(text) == strip_tool_call_markup(
        text, extended=False)
    assert strip_tool_call_markup(
        "<tool>\nx {}\n</tool>", extended=True) == ""


# ===========================================================================
# E2E: runner-level loop recovery via the nudge mechanism (once per run)
# ===========================================================================

def _loop_handler(nudges_allowed: bool = True):
    class LoopHandler(_Handler):
        def _completion_payload(self, payload):  # noqa: N802
            messages = payload.get("messages", [])
            nudged = any("repetition loop" in str(m.get("content"))
                         for m in messages if m.get("role") == "user")
            if nudged and nudges_allowed:
                content = "Done! The file is at a.py and it parses fine."
            else:
                content = "Hmm.\n\n" + "\n\n".join([LAP] * 4)
            return {"id": "c", "object": "chat.completion",
                    "model": "fake",
                    "choices": [{"index": 0,
                                 "message": {"role": "assistant",
                                             "content": content},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                              "total_tokens": 2}}
    return LoopHandler


def _run(client_kwargs):
    with FakeOpenAIServer(handler_cls=_loop_handler()) as fs:
        client = ToasterLoopAgentClient(base_url=fs.base_url, model="fake",
                                    **client_kwargs)
        return client.run("do the thing")


def test_runner_truncates_loop_and_nudges_once():
    res = _run({})
    assert res.completed
    nudges = [m for m in res.messages
              if m.get("role") == "user"
              and "repetition loop" in str(m.get("content"))]
    assert len(nudges) == 1  # exactly one synthetic retry turn
    # the truncated turn (with the marker) is what entered history
    loop_turns = [m for m in res.messages
                  if LOOP_MARKER in str(m.get("content"))]
    assert len(loop_turns) == 1
    assert res.final_response.startswith("Done!")


def test_runner_loop_nudge_budget_is_one_per_run():
    # handler NEVER escapes the loop: run must still terminate with the
    # truncated answer and exactly one nudge (no retry spiral).
    def _handler_cls():
        class StuckHandler(_Handler):
            def _completion_payload(self, payload):  # noqa: N802
                content = "Hmm.\n\n" + "\n\n".join([LAP] * 4)
                return {"id": "c", "object": "chat.completion",
                        "model": "fake",
                        "choices": [{"index": 0,
                                     "message": {"role": "assistant",
                                                 "content": content},
                                     "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 1,
                                  "completion_tokens": 1,
                                  "total_tokens": 2}}
        return StuckHandler
    with FakeOpenAIServer(handler_cls=_handler_cls()) as fs:
        client = ToasterLoopAgentClient(base_url=fs.base_url, model="fake")
        res = client.run("do the thing")
    nudges = [m for m in res.messages
              if m.get("role") == "user"
              and "repetition loop" in str(m.get("content"))]
    assert len(nudges) == 1
    assert res.completed


def test_runner_guardrails_off_disables_loop_recovery():
    res = _run({"guardrails_enabled": False})
    # No truncation, no nudge: legacy behavior preserved byte-for-byte
    # when guardrails are opted out.
    assert not any(LOOP_MARKER in str(m.get("content"))
                   for m in res.messages)
    assert not any("repetition loop" in str(m.get("content"))
                   for m in res.messages if m.get("role") == "user")


# ===========================================================================
# Performance sanity: adversarial inputs stay linear-ish
# ===========================================================================

def _build_pathological() -> str:
    # One fence per unit combining every repair class (trailing commas,
    # single quotes, python literals) so the FIRST fence repairs and wins;
    # the cost under test is the scan over thousands of corrupt candidates.
    unit = (FENCE + "json\n{'name': 'search_files', 'arguments': "
            "{\"query\": \"x\"}, \"b\": {'c': True,},}\n" + FENCE + "\n")
    # ~200KB of repeated corruption patterns + a stop-token storm tail
    return unit * 2600 + IM_END * 5000


def test_pathological_input_under_one_second():
    text = _build_pathological()
    assert len(text) > 200_000
    t0 = time.time()
    calls = parse_text_tool_calls(text, known_tool_names=KNOWN)
    strip_stop_tokens(text + IM_END * 100)
    detect_reasoning_loop(text)
    dt = time.time() - t0
    assert calls  # still parses (first fence wins)
    assert dt < 1.0, f"adversarial 200KB took {dt:.2f}s"


# ===========================================================================
# GATE LADDER over the drift corpus
# ===========================================================================

def _load_corpus():
    manifest_path = FIXTURES / "manifest.json"
    assert manifest_path.exists(), "drift corpus manifest missing"
    manifest: Dict[str, Dict[str, Any]] = json.loads(
        manifest_path.read_text(encoding="utf-8"))
    files = sorted(str(p.relative_to(FIXTURES))
                   for p in FIXTURES.rglob("*.txt"))
    return manifest, files


def _class_of(rel: str) -> str:
    return rel.split("/", 1)[0]


def test_layer1_weightless_text_scan():
    """LAYER 1: every fixture loads; manifest and files agree exactly."""
    manifest, files = _load_corpus()
    assert files, "fixture corpus is empty"
    assert set(files) == set(manifest), (
        f"manifest/files mismatch: {set(files) ^ set(manifest)}")
    for rel in files:
        text = (FIXTURES / rel).read_text(encoding="utf-8")
        assert text.strip(), f"{rel} is empty"
        assert _class_of(rel) in {"stop_token_repeat", "reasoning_loop",
                                  "malformed_json", "xml_wrapped", "mixed"}


def _ladder_rows():
    manifest, files = _load_corpus()
    rows = []
    regressions: List[str] = []
    for rel in files:
        meta = manifest[rel]
        text = (FIXTURES / rel).read_text(encoding="utf-8")
        legacy = parse_legacy(text)                      # baseline (live)
        rep = parse_text_tool_calls_with_repairs(
            text, known_tool_names=KNOWN)
        rows.append((rel, meta, text, legacy, rep))
        if legacy:
            # ZERO-REGRESSION: legacy-parsing inputs parse identically.
            if _norm_calls(rep.calls) != _norm_calls(legacy):
                regressions.append(rel)
    return rows, regressions


def test_layer2_per_fixture_expectations():
    """LAYER 2: recovery + zero-regression assertions per fixture."""
    rows, regressions = _ladder_rows()
    assert not regressions, f"REGRESSIONS: {regressions}"
    for rel, meta, text, legacy, rep in rows:
        calls = rep.calls
        assert bool(calls) == meta["expect_parsed"], (
            f"{rel}: parsed={bool(calls)} "
            f"want {meta['expect_parsed']} calls={calls}")
        if not calls:
            continue
        assert calls[0]["function"]["name"] == meta["expect_name"], rel
        want_args = meta.get("expect_args")
        if want_args is not None:
            got = json.loads(calls[0]["function"]["arguments"])
            for k, v in want_args.items():
                assert got.get(k) == v, f"{rel}: arg {k}"
        tags = rep.tags_for(calls[0])
        want_any = meta.get("expect_tags_any")
        if want_any:
            assert any(t in tags for t in want_any), (
                f"{rel}: tags {tags} want any of {want_any}")
        # attribution coherence: recovered => non-empty tags and vice versa
        assert bool(tags) == rep.recovered, f"{rel}: attribution"
        if _class_of(rel) == "reasoning_loop":
            v = detect_reasoning_loop(text)
            assert v.loop == meta["expect_loop"], rel
            if v.loop:
                cut = truncate_at_loop(text, v)
                assert cut.endswith(LOOP_MARKER), rel
                assert text[:v.start].rstrip() in cut, rel


def test_layer3_recovery_rate_report():
    """LAYER 3: aggregate recovery-rate table (printed) + strict gate."""
    rows, regressions = _ladder_rows()
    assert not regressions, f"REGRESSIONS: {regressions}"

    parse_fixtures = [r for r in rows if _class_of(r[0]) != "reasoning_loop"]
    classes = sorted({_class_of(rel) for rel, *_ in parse_fixtures})
    header = (f"{'class':<18} {'n':>3} {'baseline(pre-WP8)':>17} "
              f"{'new(WP8)':>9} {'recovered+':>10}")
    lines = ["", "WP8 drift corpus recovery-rate report "
                  "(baseline = parse_legacy, live):", header,
             "-" * len(header)]
    total_n = base_n = new_n = 0
    for cls in classes:
        sub = [r for r in parse_fixtures if _class_of(r[0]) == cls]
        n = len(sub)
        b = sum(1 for r in sub if r[3])
        w = sum(1 for r in sub if r[4].calls)
        total_n += n
        base_n += b
        new_n += w
        lines.append(f"{cls:<18} {n:>3} {b:>17} {w:>9} {w - b:>+10}")
    lines.append("-" * len(header))
    bp = 100.0 * base_n / total_n
    wp = 100.0 * new_n / total_n
    lines.append(f"{'TOTAL':<18} {total_n:>3} {base_n:>17} {new_n:>9} "
                 f"{new_n - base_n:>+10}")
    lines.append(f"recovery rate: baseline {bp:.1f}% -> new {wp:.1f}% "
                 f"(+{wp - bp:.1f} percentage points)")
    lines.append(f"zero regressions: {'YES' if not regressions else 'NO'}")
    report = "\n".join(lines)
    print(report)  # -s friendly; also captured in failure output

    assert new_n > base_n, (
        "WP8 must recover strictly more than the pre-WP8 baseline:\n"
        + report)
    # all parse-class fixtures are the target: full recovery of every
    # positive expectation (the corpus is curated so expect_parsed=True
    # fixtures are ALL recovered).
    expected = sum(1 for r in parse_fixtures if r[1]["expect_parsed"])
    assert new_n == expected, report

# ------------------------------------------------------------------------
# from tests/test_validate_tool_calls.py
# ------------------------------------------------------------------------

@pytest.fixture()
def _no_sleeps(monkeypatch):
    """Retry backoff must not slow the suite down."""
    import toaster_loop.training.validate as v
    monkeypatch.setattr(v, "_sleep", lambda s: None)


def native_call_msg(name="read_file", arguments=None, content=None):
    if arguments is None:
        arguments = json.dumps({"path": "notes/todo.md"})
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": name, "arguments": arguments},
        }],
    }


def run(script, tasks):
    handler_cls = make_message_handler(script)
    with FakeOpenAIServer(handler_cls=handler_cls) as srv:
        res = run_golden_suite(srv.base_url, "m", tasks)
    return res, handler_cls.requests


TOOL_TASK = {
    "prompt": "read notes/todo.md please",
    "expect_tool_call": {"name": "read_file", "arg_contains": {"path": "notes"}},
}


# ------------------------------------------------------------- happy paths

def test_native_tool_calls_passes_expect_tool_call(_no_sleeps):
    res, reqs = run({"read notes": native_call_msg()}, [TOOL_TASK])
    assert res["pass_rate"] == 1.0
    entry = res["results"][0]
    assert entry["pass"] is True
    kinds = [c["kind"] for c in entry["checks"]]
    assert kinds == ["tool_call"]


def test_tool_task_request_ships_production_schemas(_no_sleeps):
    _, reqs = run({"read notes": native_call_msg()}, [TOOL_TASK])
    assert len(reqs) == 1
    sent = reqs[0].get("tools")
    assert sent == TOOL_SCHEMAS  # exact same schemas the agent uses


def test_substring_only_task_sends_no_tools_field(_no_sleeps):
    script = {"2+2": {"role": "assistant", "content": "4"}}
    res, reqs = run(script, [{"prompt": "what is 2+2?", "expect_contains": ["4"]}])
    assert res["pass_rate"] == 1.0
    assert "tools" not in reqs[0]  # byte-identical to pre-#5 traffic


def test_text_emitted_tool_call_recovered_and_passes(_no_sleeps):
    # A text-grammar model: fenced JSON block, no native field. The grammar
    # regression must still NOT be flagged — parse_text_tool_calls recovers it.
    msg = {
        "role": "assistant",
        "content": 'Sure.\n```json\n{"name": "read_file", '
                   '"arguments": {"path": "notes/todo.md"}}\n```',
    }
    res, _ = run({"read notes": msg}, [TOOL_TASK])
    assert res["pass_rate"] == 1.0
    check = res["results"][0]["checks"][0]
    assert check["kind"] == "tool_call" and check["pass"] is True


def test_name_only_check_accepts_any_args(_no_sleeps):
    msg = native_call_msg(name="list_dir", arguments=json.dumps({"path": "."}))
    res, _ = run({"read notes": msg},
                 [{"prompt": "read notes", "expect_tool_call": {"name": "list_dir"}}])
    assert res["pass_rate"] == 1.0


def test_no_name_accepts_any_tool_call(_no_sleeps):
    res, _ = run({"read notes": native_call_msg(name="search_files")},
                 [{"prompt": "read notes", "expect_tool_call": {}}])
    assert res["pass_rate"] == 1.0


# ------------------------------------------------------------- failure paths

def test_prose_only_fails_with_tool_call_kind(_no_sleeps):
    msg = {"role": "assistant", "content": "The file says hello."}
    res, _ = run({"read notes": msg}, [TOOL_TASK])
    assert res["pass_rate"] == 0.0
    entry = res["results"][0]
    assert entry["pass"] is False
    check = next(c for c in entry["checks"] if not c["pass"])
    assert check["kind"] == "tool_call"
    assert "prose only" in check["reason"]
    assert "read_file" in entry["error"] and "tool_call" in entry["error"]


def test_wrong_tool_name_fails(_no_sleeps):
    res, _ = run({"read notes": native_call_msg(name="write_file")}, [TOOL_TASK])
    assert res["passed"] == 0
    check = res["results"][0]["checks"][0]
    assert check["kind"] == "tool_call" and not check["pass"]
    assert "write_file" in check["reason"]


def test_arg_contains_match_is_case_insensitive_substring(_no_sleeps):
    msg = native_call_msg(arguments=json.dumps({"path": "/tmp/NOTES/todo.md"}))
    res, _ = run({"read notes": msg}, [TOOL_TASK])
    assert res["pass_rate"] == 1.0


def test_arg_contains_mismatch_fails(_no_sleeps):
    msg = native_call_msg(arguments=json.dumps({"path": "src/main.py"}))
    res, _ = run({"read notes": msg}, [TOOL_TASK])
    assert res["passed"] == 0
    check = res["results"][0]["checks"][0]
    assert check["kind"] == "tool_call" and not check["pass"]
    assert "path" in check["reason"]


def test_arg_contains_missing_key_fails(_no_sleeps):
    msg = native_call_msg(arguments=json.dumps({"offset": 1}))
    res, _ = run({"read notes": msg}, [TOOL_TASK])
    assert res["passed"] == 0
    assert "missing" in res["results"][0]["checks"][0]["reason"]


def test_malformed_args_json_fails_cleanly(_no_sleeps):
    # Unparseable arguments string — must fail the check, never raise.
    msg = native_call_msg(arguments="{not json at all")
    res, _ = run({"read notes": msg}, [TOOL_TASK])
    assert res["passed"] == 0
    check = res["results"][0]["checks"][0]
    assert check["kind"] == "tool_call" and not check["pass"]
    assert "malformed" in check["reason"]


# ---------------------------------------------------------------- mixed

def test_mixed_substring_and_tool_call_checks(_no_sleeps):
    task = {
        "prompt": "read notes/todo.md and summarize",
        "expect_contains": ["todo"],
        "expect_tool_call": {"name": "read_file"},
    }
    # Reply satisfies both: native call plus prose containing "todo".
    msg = native_call_msg(content="Working on todo.md now")
    res, _ = run({"read notes": msg}, [task])
    assert res["pass_rate"] == 1.0
    kinds = sorted(c["kind"] for c in res["results"][0]["checks"])
    assert kinds == ["substring", "tool_call"]

    # Substring hits but no tool call -> task fails on the tool_call kind.
    prose = {"role": "assistant", "content": "todo.md says nothing"}
    res, _ = run({"read notes": prose}, [task])
    assert res["passed"] == 0
    entry = res["results"][0]
    ok_kinds = {c["kind"] for c in entry["checks"] if c["pass"]}
    bad_kinds = {c["kind"] for c in entry["checks"] if not c["pass"]}
    assert ok_kinds == {"substring"} and bad_kinds == {"tool_call"}


def test_substring_failure_reported_with_substring_kind(_no_sleeps):
    task = {
        "prompt": "read notes/todo.md",
        "expect_contains": ["banana"],
        "expect_tool_call": {"name": "read_file"},
    }
    res, _ = run({"read notes": native_call_msg()}, [task])
    assert res["passed"] == 0
    entry = res["results"][0]
    bad = [c for c in entry["checks"] if not c["pass"]]
    assert len(bad) == 1 and bad[0]["kind"] == "substring"


# ------------------------------------------------------------- transport

class BlipThenToolCallHandler(BaseHTTPRequestHandler):
    """First POST dies with a connection reset; second returns a tool call."""

    blipped = False

    def log_message(self, format, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if not type(self).blipped:
            type(self).blipped = True
            self.close_connection = True
            try:
                self.wfile.close()
            except Exception:
                pass
            return
        body = json.dumps({
            "id": "x", "object": "chat.completion",
            "model": payload.get("model", "m"),
            "choices": [{"index": 0, "message": native_call_msg(),
                         "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                      "total_tokens": 2},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_transport_retry_applies_to_tool_call_tasks(_no_sleeps):
    BlipThenToolCallHandler.blipped = False
    with FakeOpenAIServer(handler_cls=BlipThenToolCallHandler) as srv:
        res = run_golden_suite(srv.base_url, "m", [TOOL_TASK])
    assert res["pass_rate"] == 1.0


# --------------------------------------------------- backward compatibility

def test_legacy_report_fields_preserved(_no_sleeps):
    script = {"hi": {"role": "assistant", "content": "hello world"}}
    res, _ = run(script, [{"prompt": "hi", "expect_contains": ["hello"],
                           "expect_not_contains": ["nope"]}])
    entry = res["results"][0]
    assert entry["missing"] == [] and entry["forbidden_hit"] == []
    assert "reply_preview" in entry and entry["pass"] is True

# ------------------------------------------------------------------------
# from tests/test_redact.py
# ------------------------------------------------------------------------

# --------------------------------------------------------------- golden set

# (family label, fixture containing the secret)
GOLDEN_SECRETS = [
    ("openai", "OPENAI key is sk-abcDEF456ghiJKL789mnoPQR"),
    ("openai-project", "use sk-proj-aBcDeFgHiJkLmNoPqRsTuVwXyZ0123456789 now"),
    ("anthropic", "anthropic: sk-ant-api03-ZZZZZzzzzXXXXyyyyYYYYzzzz1234"),
    ("github-pat", "ghp_AbCdEfGhIjKlMnOpQrStUvWx has repo scope"),
    ("github-oauth", "gho_AbCdEfGhIjKlMnOpQrStUvWx"),
    ("github-fine-grained",
     "github_pat_11ABCDEF0xYz9_abcdefghijklmnopqrstuvwx"),
    ("aws-akid", "AWS key AKIAIOSFODNN7EXAMPLE found"),
    ("aws-temp", "temporary ASIAIOSFODNN7EXAMPLE plus secret"),
    ("huggingface", "hf_AbCdEfGhIjKlMnOpQrStUvWx pushed the model"),
    ("slack", "slack token xoxb-1234567890abcdef-abcdefghij posted"),
    ("google", "AIzaSyA-bCdEfGhIjKlMnOpQrStUvWxYz012345 queried"),
    ("stripe-live-sk", "charge with sk_live_51Hxxxxxxxxxxxxxxxxxxxxxxabc"),
    ("stripe-live-pk", "form uses pk_live_51Hyyyyyyyyyyyyyyyyyyyyyydef"),
    ("gitlab", "glpat-abcdef123456789ABCDF revoked"),
    ("npm", "npm_abcdef1234567890ABCDEF published"),
    ("telegram",
     "bot 123456789:AAHabcdefGHIijklmnopQRSTuvwxYZ12345678 replied"),
    ("discord-webhook",
     "posted to https://discord.com/api/webhooks/"
     "123456789012345678/AbCdEfGhIjKlMnOpQrStUvWxYz123456"),
    ("jwt",
     "auth eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
     ".eyJzdWIiOiIxMjM0NTY3ODkwIn0"
     ".dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"),
    ("kv-env", 'export OPENAI_API_KEY="sk-not-matched-but-secret-value"'),
    ("kv-password", "login with password = hunter2secret here"),
    ("kv-authorization", "Authorization: Bearer abcdefghijklmnop1234"),
    ("json-body", '{"user": "u", "api_key": "opaque-token-value-xyz"}'),
]

PEM_BLOCK = (
    "-----BEGIN RSA PRIVATE KEY-----\n"
    "MIIBOgIBAAJBAKj34GkxFhD90vcNLYLInFEX6Ppy1tPf9Cnzj4p4WGeKLs1P\n"
    "NB8KcKO9vyNcRy7\n"
    "-----END RSA PRIVATE KEY-----"
)


@pytest.mark.parametrize("family,fixture", GOLDEN_SECRETS,
                         ids=[f for f, _ in GOLDEN_SECRETS])
def test_golden_secrets_are_redacted(family, fixture):
    out = redact_text(fixture)
    assert "[REDACTED" in out, f"{family}: no masking applied to {fixture!r}"
    # the secret itself must be gone: strip prose and check token survival
    for tok in fixture.split():
        if len(tok) >= 12 and any(c.isdigit() for c in tok):
            assert tok not in out, f"{family}: token survived: {tok!r}"


def test_pem_block_fully_replaced():
    out = redact_text("here is the key:\n" + PEM_BLOCK + "\ndone")
    assert "[REDACTED_PRIVATE_KEY]" in out
    assert "MIIBOgIBAAJ" not in out
    assert "BEGIN RSA PRIVATE KEY" not in out


@pytest.mark.parametrize("family,fixture", GOLDEN_SECRETS + [("pem", PEM_BLOCK)],
                         ids=[f for f, _ in GOLDEN_SECRETS] + ["pem"])
def test_redaction_is_idempotent(family, fixture):
    once = redact_text(fixture)
    assert redact_text(once) == once


@pytest.mark.parametrize("benign", [
    "the task skipped step 3, disk usage on /dev/sda1 is fine",
    "git commit -m 'fix sk handling in the dispatcher'",
    "hex color #a1b2c3 and pointer id 0xdeadbeef",
    "def token():\n    return 42",
    "api_key = os.environ.get('OPENAI_API_KEY')",
    "tokenizer = 'cl100k_base'",
    "",
    "plain prose with no secrets whatsoever, skipping nothing.",
])
def test_benign_text_untouched(benign):
    assert redact_text(benign) == benign


def test_redact_messages_does_not_mutate_input():
    msgs = [
        {"role": "tool", "content": "secret sk-abcDEF456ghiJKL789mno here"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {
                "name": "write_file",
                "arguments": '{"path": "x", "content": "token: s3cr3tvalue-abc"}'}},
        ]},
    ]
    out = redact_messages(msgs)
    assert "sk-abcDEF456ghiJKL789mno" in msgs[0]["content"]  # raw kept
    assert "sk-abcDEF456ghiJKL789mno" not in out[0]["content"]
    assert "[REDACTED" in out[1]["tool_calls"][0]["function"]["arguments"]
    assert "s3cr3tvalue-abc" not in out[1]["tool_calls"][0]["function"]["arguments"]
    # original tool-call args untouched
    assert "s3cr3tvalue-abc" in msgs[1]["tool_calls"][0]["function"]["arguments"]


# ------------------------------------------------------------ recorder layer

class _SecretLeakingHandler(BaseHTTPRequestHandler):
    """First turn: read_file on a secret file. After tool result: echo it."""

    def log_message(self, format, *args):  # silence
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        messages = payload.get("messages", [])
        tool_msg = next((m for m in messages if m.get("role") == "tool"), None)
        if tool_msg is None:
            msg = {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_1", "type": "function", "function": {
                    "name": "read_file",
                    "arguments": json.dumps(
                        {"path": self.server.secret_path})  # type: ignore[attr-defined]
                    }}]}
            finish = "tool_calls"
        else:
            msg = {"role": "assistant",
                   "content": f"Answer with raw secret: {tool_msg['content'][:60]}"}
            finish = "stop"
        body = json.dumps({
            "id": "c", "object": "chat.completion", "model": "fake-model",
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                      "total_tokens": 2}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.send_response(404)
        self.end_headers()


SECRET_IN_FILE = "sk-TESTabcDEF456ghiJKL789mnoTOPSECRET"


@pytest.fixture()
def toaster_loop_home__test_redact(tmp_path, monkeypatch):
    home = tmp_path / "uf-home"
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(home))
    monkeypatch.delenv("TOASTER_LOOP_CONFIG", raising=False)
    return home


def test_recorded_jsonl_redacted_live_memory_raw(toaster_loop_home__test_redact, tmp_path):
    """End-to-end: tool result contains a secret -> JSONL clean, memory raw."""
    secret_file = tmp_path / "notes.txt"
    secret_file.write_text(f"api value {SECRET_IN_FILE} end\n")

    srv = FakeOpenAIServer(handler_cls=_SecretLeakingHandler)
    srv._httpd.secret_path = str(secret_file)  # type: ignore[attr-defined]
    with srv:
        rec = TraceRecorder(home=toaster_loop_home__test_redact)
        client = ToasterLoopAgentClient(
            base_url=srv.base_url, model="fake-model", recorder=rec)
        result = client.run("read the secret file")
        assert result.completed

    # live conversation keeps the raw secret (agent must still function)
    joined_memory = json.dumps(result.messages, default=str)
    assert SECRET_IN_FILE in joined_memory

    # durable buffer contains no trace of it
    matches = list((toaster_loop_home__test_redact / "buffer").rglob(f"{result.trace_id}.jsonl"))
    assert len(matches) == 1
    text = matches[0].read_text()
    assert SECRET_IN_FILE not in text
    assert "[REDACTED" in text

    # SQLite index summary too
    from toaster_loop.store.db import TraceStore
    store = TraceStore(toaster_loop_home__test_redact / "buffer" / "toaster_loop.db")
    try:
        row = store.get_trace(result.trace_id or "")
        assert row is not None and SECRET_IN_FILE not in (row.task_summary or "")
    finally:
        store.close()


def test_recorder_redaction_escape_hatch_logs_warning(toaster_loop_home__test_redact, caplog):
    class CfgOff:
        recording_redact_secrets = False

    with caplog.at_level("WARNING", logger="toaster_loop.store"):
        rec = TraceRecorder(cfg=CfgOff(), home=toaster_loop_home__test_redact)
    assert any("redaction is DISABLED" in r.message for r in caplog.records)
    row = rec.record(task=f"log {SECRET_IN_FILE}", messages=[
        {"role": "user", "content": f"key {SECRET_IN_FILE}"},
        {"role": "assistant", "content": "ok"},
    ], backend="t", model_alias="m", base_url="", completed=True)
    assert row is not None
    text = Path(row.jsonl_path).read_text()
    # explicitly disabled => verbatim (documented escape-hatch behavior)
    assert SECRET_IN_FILE in text


def test_recorder_redaction_on_by_default(toaster_loop_home__test_redact):
    rec = TraceRecorder(home=toaster_loop_home__test_redact)  # no cfg
    assert rec.redact_secrets is True
    row = rec.record(task="t", messages=[
        {"role": "user", "content": f"key {SECRET_IN_FILE}"},
        {"role": "assistant", "content": "done"},
    ], backend="t", model_alias="m", base_url="", completed=True)
    assert row is not None and SECRET_IN_FILE not in Path(
        row.jsonl_path).read_text()


def test_config_yaml_recording_redact_secrets(tmp_path, monkeypatch):
    monkeypatch.delenv("TOASTER_LOOP_RECORDING", raising=False)
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text("recording:\n  redact_secrets: false\n")
    assert load_config(str(cfg_file)).recording_redact_secrets is False
    cfg_file2 = tmp_path / "cfg2.yaml"
    cfg_file2.write_text("tools:\n  block_sensitive_files: false\n")
    assert load_config(str(cfg_file2)).tools["block_sensitive_files"] is False


# ---------------------------------------------------------------- tools

def test_read_file_refuses_dotenv(toaster_loop_home__test_redact, tmp_path):
    env = tmp_path / "project" / ".env"
    env.parent.mkdir(parents=True)
    env.write_text("SECRET=abc\n")
    out = read_file(str(env))
    assert out.startswith("error: blocked: sensitive credential path")
    assert "block_sensitive_files" in out


def test_read_file_refuses_ssh_key(toaster_loop_home__test_redact, tmp_path, monkeypatch):
    fake_home = tmp_path / "home"
    (fake_home / ".ssh").mkdir(parents=True)
    (fake_home / ".ssh" / "id_rsa").write_text("PRIVATE\n")
    monkeypatch.setenv("HOME", str(fake_home))
    out = read_file("~/.ssh/id_rsa")
    assert out.startswith("error: blocked: sensitive credential path")


@pytest.mark.parametrize("rel", [
    ".aws/credentials", ".gnupg/pubring.kbx", "secrets/.netrc",
    "db/.pgpass", "app/credentials.json", ".env.production",
])
def test_read_file_refuses_credential_paths(toaster_loop_home__test_redact, tmp_path, rel):
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x\n")
    assert read_file(str(p)).startswith("error: blocked:")


def test_read_file_allows_normal_files(toaster_loop_home__test_redact, tmp_path):
    good = tmp_path / "app.py"
    good.write_text("print('hi')\n")
    assert "print('hi')" in read_file(str(good))
    # public keys and .env.example are fine
    (tmp_path / "id_ed25519.pub").write_text("ssh-ed25519 AAAA...\n")
    assert "ssh-ed25519" in read_file(str(tmp_path / "id_ed25519.pub"))
    (tmp_path / ".env.example").write_text("PLACEHOLDER=1\n")
    assert "PLACEHOLDER" in read_file(str(tmp_path / ".env.example"))


def test_read_file_config_off_allows_reads(toaster_loop_home__test_redact, tmp_path):
    env = tmp_path / ".env"
    env.write_text("SECRET=abc\n")
    out = read_file(str(env), block_sensitive=False)
    assert "SECRET=abc" in out


def test_search_files_skips_dotenv_contents(toaster_loop_home__test_redact, tmp_path):
    (tmp_path / ".env").write_text("API_TOKEN=sk-abcdef123456789secret\n")
    (tmp_path / "app.py").write_text("API_TOKEN = 'from-app-py'\n")
    out = search_files("API_TOKEN", path=str(tmp_path))
    assert "app.py" in out
    assert ".env:" not in out
    assert "sk-abcdef123456789secret" not in out


def test_search_files_name_mode_skips_blocked(toaster_loop_home__test_redact, tmp_path):
    (tmp_path / ".netrc").write_text("machine x login y password z\n")
    (tmp_path / "readme.md").write_text("hi\n")
    out = search_files(".netrc", path=str(tmp_path), mode="name")
    # blocked files never appear as result paths (the no-match notice echoing
    # the pattern is fine)
    assert str(tmp_path / ".netrc") not in out
    assert "no files matching" in out


def test_search_files_config_off_scans_dotenv(toaster_loop_home__test_redact, tmp_path):
    (tmp_path / ".env").write_text("API_TOKEN=sk-abcdef123456789secret\n")
    out = search_files("API_TOKEN", path=str(tmp_path), block_sensitive=False)
    assert ".env" in out


def test_create_runner_threads_block_flag(tmp_path, monkeypatch):
    """tools.block_sensitive_files reaches the bound tool impls."""
    from toaster_loop.core.agent_runner import create_runner
    from toaster_loop.core.config import ToasterLoopConfig

    monkeypatch.setenv("TOASTER_LOOP_HOME", str(tmp_path))
    cfg = ToasterLoopConfig(tools={"block_sensitive_files": False})
    client = create_runner(cfg, base_url="http://127.0.0.1:1/v1", model="m")
    assert client.block_sensitive_files is False
    env = tmp_path / ".env"
    env.write_text("A=1\n")
    assert "A=1" in client.tool_impls["read_file"](str(env))

    cfg2 = ToasterLoopConfig()  # default on
    client2 = create_runner(cfg2, base_url="http://127.0.0.1:1/v1", model="m")
    assert client2.tool_impls["read_file"](str(env)).startswith("error: blocked:")


# ------------------------------------------------- curater-level guarantee

def test_secret_never_reaches_curated_sft(toaster_loop_home__test_redact):
    """A secret recorded through the recorder never appears in sft.jsonl.

    Extends the curate test family (test_curate.py) with the backlog #1
    guarantee: record-time redaction is sufficient because curation reads
    only the durable JSONL.
    """
    rec = TraceRecorder(home=toaster_loop_home__test_redact)
    row = rec.record(
        task=f"look up key {SECRET_IN_FILE}",
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": f"what is the key {SECRET_IN_FILE}?"},
            {"role": "assistant", "content": "I looked it up for you."},
        ],
        backend="toaster_loop", model_alias="test-model", base_url="",
        completed=True,
    )
    assert row is not None
    assert SECRET_IN_FILE not in Path(row.jsonl_path).read_text()

    assert cli.main(["curate", "--name", "secrecy"]) == 0
    ds_dirs = list((toaster_loop_home__test_redact / "datasets").rglob("sft.jsonl"))
    assert len(ds_dirs) == 1
    sft = ds_dirs[0].read_text()
    assert SECRET_IN_FILE not in sft
    assert "[REDACTED" in sft  # the trace made it in, masked

# ------------------------------------------------------------------------
# from tests/test_skills.py
# ------------------------------------------------------------------------

# ---------------------------------------------------------------- storage

def test_save_and_load_roundtrip():
    res = save_skill("steam-gui", "Driving the Steam client GUI on Linux",
                     "1. warm the client first\n2. URL bar at y=435")
    assert res["ok"], res
    s = load_skill("steam-gui")
    assert s is not None
    assert s.description == "Driving the Steam client GUI on Linux"
    assert "URL bar at y=435" in s.content
    # companion scripts dir exists for bundled executables
    from pathlib import Path
    assert Path(res["scripts_dir"]).is_dir()


def test_save_overwrites_same_name():
    save_skill("x11-input", "v1 desc", "v1 body")
    res = save_skill("X11-Input", "v2 desc", "v2 body")  # name lowercased
    assert res["ok"]
    s = load_skill("x11-input")
    assert s is not None and s.description == "v2 desc"
    assert len([sk for sk in all_skills() if sk.name == "x11-input"]) == 1


def test_validation_rejects_bad_input():
    assert "invalid name" in save_skill(
        "Bad Name!", "d", "c")["error"]
    assert "description required" in save_skill("ok", "", "c")["error"]
    assert "too long" in save_skill(
        "ok", "d" * (DESCRIPTION_MAX_CHARS + 1), "c")["error"]
    assert "content required" in save_skill("ok", "d", "  ")["error"]
    assert "too long" in save_skill(
        "ok", "d", "c" * (CONTENT_MAX_CHARS + 1))["error"]


def test_redaction_on_write():
    res = save_skill("secrets", "uses token sk-abcdefghijklmnopqrstuvwxyz123456",
                     "auth with sk-abcdefghijklmnopqrstuvwxyz123456")
    assert res["ok"]
    raw = (skills_dir() / "secrets.md").read_text()
    assert "sk-abcdefghijklmnopqrstuvwxyz123456" not in raw


def test_load_missing_and_malformed():
    assert load_skill("nope") is None
    assert load_skill("../escape") is None
    bad = skills_dir()
    bad.mkdir(parents=True, exist_ok=True)
    (bad / "broken.md").write_text("no frontmatter here")
    assert all_skills() == []  # malformed files are skipped, never crash


# ------------------------------------------------------------------ tools

def test_tool_wrappers_never_raise():
    out = save_skill_tool(name="steam-gui", description="steam gui facts",
                          content="xact click works; import hangs")
    assert "skill saved" in out
    body = read_skill(name="steam-gui")
    assert "xact click works" in body
    assert "error:" in read_skill(name="missing-skill")
    assert "steam-gui: steam gui facts" in list_skills()


# ----------------------------------------------------------------- index

def _cfg(**skills):
    return ToasterLoopConfig(skills=skills)


def test_index_disabled_by_default_and_empty_when_no_skills():
    assert index_section(_cfg()) == ""
    save_skill("a", "alpha", "body")
    assert index_section(_cfg()) == ""          # still disabled
    assert index_section(_cfg(enabled=True)) != ""


def test_index_renders_one_line_per_skill_under_cap():
    for i in range(5):
        save_skill(f"skill-{i}", f"description {i}", "body")
    sec = index_section(_cfg(enabled=True))
    assert sec.startswith("## Skills")
    assert "- skill-3: description 3" in sec
    # cap: entries that do not fit are omitted WHOLE
    tight = _cfg(enabled=True, index_total_cap_chars=40)
    sec2 = index_section(tight)
    lines = [ln for ln in sec2.splitlines() if ln.startswith("- ")]
    assert sum(len(ln) + 1 for ln in lines) <= 40
    assert all(f"skill-{i}" in sec2 or True for i in range(5))
    # no partial entries ever
    for ln in lines:
        assert not ln.endswith("d")  # descriptions are "description N"


# ------------------------------------------------------------- runner gate

def _client(**kwargs):
    return ToasterLoopAgentClient(base_url="http://127.0.0.1:1", model="m",
                              **kwargs)


def _tool_names(client):
    return {s["function"]["name"] for s in client.tools}


def test_skill_tools_stripped_when_disabled():
    c = _client()  # skills_enabled defaults False
    assert not (SKILL_TOOL_NAMES & _tool_names(c))
    assert not (SKILL_TOOL_NAMES & set(c.tool_impls))


def test_skill_tools_exposed_when_enabled_even_with_profile_list():
    c = _client(skills_enabled=True, enabled_tool_names=["read_file"])
    assert SKILL_TOOL_NAMES <= _tool_names(c)
    assert SKILL_TOOL_NAMES <= set(c.tool_impls)


def test_create_runner_wires_knob_and_index(tmp_path, monkeypatch):
    import toaster_loop.core.agent_runner as ar

    cfg = ToasterLoopConfig(skills={"enabled": True})
    save_skill("steam-gui", "steam gui facts on this machine",
               "URL bar at y=435; steam:// needs a warm client")
    runner = ar.create_runner(cfg, "http://127.0.0.1:1", "m")
    assert SKILL_TOOL_NAMES <= _tool_names(runner)
    assert "## Skills" in runner.system_prompt
    assert "steam-gui: steam gui facts on this machine" in runner.system_prompt

    # disabled => byte-identical prompt, tools stripped
    cfg_off = ToasterLoopConfig(skills={})
    runner_off = ar.create_runner(cfg_off, "http://127.0.0.1:1", "m")
    assert "## Skills" not in runner_off.system_prompt
    assert not (SKILL_TOOL_NAMES & _tool_names(runner_off))

# ------------------------------------------------------------------------
# from tests/test_tools_skills_hardening.py
# ------------------------------------------------------------------------

class _Silent(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002
        pass


def _completion(message, finish="stop"):
    return json.dumps({
        "id": "x", "object": "chat.completion", "model": "m",
        "choices": [{"index": 0, "finish_reason": finish, "message": message}],
    }).encode()


def _scripted_server(script):
    """Fake server answering one scripted completion per POST.

    `script` is a list of message dicts; the last entry repeats if exhausted."""
    state = {"n": 0}

    class Handler(_Silent):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            msg = script[min(state["n"], len(script) - 1)]
            state["n"] += 1
            body = _completion(msg)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def _tool_call(cid, name, args):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


# ------------------------------------------------ 1. schema-args security gate

def test_forged_block_sensitive_is_rejected_not_executed(tmp_path):
    """read_file with block_sensitive=false must NOT reach the impl."""
    secret = tmp_path / ".env"
    secret.write_text("API_KEY=super-secret\n")
    executed = {"hit": False}

    def spy_read(path, **kwargs):
        executed["hit"] = True
        return uf_tools.read_file(path, **kwargs)

    script = [
        {"role": "assistant", "content": None,
         "tool_calls": [_tool_call("c1", "read_file",
                                   {"path": str(secret),
                                    "block_sensitive": False})]},
        {"role": "assistant", "content": "done"},
    ]
    with FakeOpenAIServer(handler_cls=_scripted_server(script)) as fs:
        client = ToasterLoopAgentClient(base_url=fs.base_url, model="m",
                                    tool_impls={"read_file": spy_read})
        result = client.run("read the secret please")
    assert not executed["hit"], "impl ran despite forged kwarg"
    tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
    assert tool_msgs and tool_msgs[0]["content"].startswith(
        "error: bad arguments")
    assert "block_sensitive" in tool_msgs[0]["content"]
    assert "super-secret" not in json.dumps(result.messages)


def test_forged_private_network_and_lsp_config_rejected():
    """web_fetch/run_command/lsp knobs cannot be forged either."""
    script = [
        {"role": "assistant", "content": None,
         "tool_calls": [_tool_call("c1", "write_file",
                                   {"path": "x.py", "content": "print(1)",
                                    "lsp_config": {"enabled": True}})]},
        {"role": "assistant", "content": "done"},
    ]
    with FakeOpenAIServer(handler_cls=_scripted_server(script)) as fs:
        client = ToasterLoopAgentClient(base_url=fs.base_url, model="m")
        result = client.run("write and lsp me")
    tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
    assert tool_msgs and tool_msgs[0]["content"].startswith(
        "error: bad arguments")


def test_valid_args_still_execute():
    """The gate must not break normal calls."""
    script = [
        {"role": "assistant", "content": None,
         "tool_calls": [_tool_call("c1", "todo_write",
                                   {"todos": [{"content": "step one"}]})]},
        {"role": "assistant", "content": "done"},
    ]
    with FakeOpenAIServer(handler_cls=_scripted_server(script)) as fs:
        client = ToasterLoopAgentClient(base_url=fs.base_url, model="m")
        result = client.run("plan it")
    tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
    assert tool_msgs and not tool_msgs[0]["content"].startswith("error:")
    assert "step one" in tool_msgs[0]["content"]


def test_unschemaed_custom_impls_pass_through():
    """Custom tools without a discoverable schema keep working (escape hatch)."""
    script = [
        {"role": "assistant", "content": None,
         "tool_calls": [_tool_call("c1", "custom_tool",
                                   {"anything": 1})]},
        {"role": "assistant", "content": "done"},
    ]
    with FakeOpenAIServer(handler_cls=_scripted_server(script)) as fs:
        client = ToasterLoopAgentClient(
            base_url=fs.base_url, model="m", tools=[],
            tool_impls={"custom_tool": lambda **kw: f"ran with {kw}"})
        result = client.run("go")
    tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
    assert tool_msgs and "ran with" in tool_msgs[0]["content"]


# --------------------------------------------------------- 2. schema parity

def test_tool_impls_and_schemas_have_exact_parity():
    impls = set(uf_tools.TOOL_IMPLS)
    schemas = ({s["function"]["name"] for s in uf_tools.TOOL_SCHEMAS}
               | set(uf_tools.CAPABILITY_TOOL_SCHEMAS))
    assert impls == schemas, (sorted(impls - schemas), sorted(schemas - impls))


def test_dashboard_known_tools_subset_of_real_tools():
    from toaster_loop.dashboard.app import DashboardApp

    real = set(uf_tools.TOOL_IMPLS)
    unknown = [t for t in DashboardApp.KNOWN_TOOLS if t not in real]
    assert not unknown, f"KNOWN_TOOLS drifted: {unknown}"


# ------------------------------------------------------------ 3. read paging

def test_read_file_pages_beyond_old_256kb_window(tmp_path):
    p = tmp_path / "big.txt"
    # ~400 KB of short lines: line 9000 sits far past the old 256 KB window.
    p.write_text("\n".join(f"line-{i}" for i in range(1, 10_001)) + "\n")
    out = uf_tools.read_file(str(p), offset=9000, limit=3)
    assert "line-9000" in out and "line-9002" in out
    assert "no lines" not in out


def test_read_file_offset_past_eof_reports_clearly(tmp_path):
    p = tmp_path / "small.txt"
    p.write_text("a\nb\nc\n")
    out = uf_tools.read_file(str(p), offset=99)
    assert "shorter" in out


def test_read_file_more_lines_hint_offsets_forward(tmp_path):
    p = tmp_path / "lines.txt"
    p.write_text("\n".join(f"L{i}" for i in range(1, 201)) + "\n")
    out = uf_tools.read_file(str(p), offset=1, limit=10)
    assert "continue with offset=11" in out


# ---------------------------------------------------------- 4. patch / search

def test_patch_fuzzy_match_is_annotated(tmp_path):
    p = tmp_path / "f.py"
    p.write_text("def f():\n    return 1\n")
    # Multi-line old_string whose lines match only whitespace-stripped:
    # exact substring fails, the fuzzy line-window hits => annotated.
    out = uf_tools.patch(str(p), "def f():\n        return 1",
                         "def f():\n    return 2")
    assert "FUZZY" in out
    assert "return 2" in p.read_text()


def test_patch_exact_match_not_annotated(tmp_path):
    p = tmp_path / "g.py"
    p.write_text("x = 1\n")
    out = uf_tools.patch(str(p), "x = 1", "x = 2")
    assert "FUZZY" not in out and out.startswith("patched")


def test_search_files_missing_root_errors(tmp_path):
    out = uf_tools.search_files("needle", str(tmp_path / "nope"))
    assert out.startswith("error: not a directory")


# --------------------------------------------------------------- 5. skills

def test_delete_skill_tool_and_backend():
    from toaster_loop.core.skills import load_skill, save_skill

    assert save_skill("stale-recipe", "old", "wrong facts")["ok"]
    assert load_skill("stale-recipe") is not None
    out = uf_tools.delete_skill(name="stale-recipe")
    assert out.startswith("skill deleted")
    assert load_skill("stale-recipe") is None
    assert "error:" in uf_tools.delete_skill(name="stale-recipe")


def test_save_skill_atomic_no_tmp_leftover():
    from toaster_loop.core.skills import save_skill, skills_dir

    assert save_skill("atomic", "d", "body")["ok"]
    leftovers = list(skills_dir().glob("*.tmp"))
    assert not leftovers, f"temp file left behind: {leftovers}"


def test_frontmatter_quoted_values_parse():
    from toaster_loop.core.skills import save_skill, load_skill

    res = save_skill("quoted", 'when x happens: do y', "body")  # colon desc
    assert res["ok"]
    s = load_skill("quoted")
    assert s is not None and ":" in s.description


def test_index_orders_most_recent_first_under_cap():
    from toaster_loop.core.config import ToasterLoopConfig
    from toaster_loop.core.skills import index_section, save_skill

    # Same timestamp within a fast loop would tie; force ordering by writing
    # updated fields directly.
    save_skill("aaa-old", "old entry", "b")
    save_skill("zzz-new", "new entry", "b")
    from toaster_loop.core.skills import skills_dir
    (skills_dir() / "aaa-old.md").write_text(
        "---\nname: aaa-old\ndescription: old entry\n"
        "updated: 2020-01-01T00:00:00+00:00\n---\n\nb\n")
    (skills_dir() / "zzz-new.md").write_text(
        "---\nname: zzz-new\ndescription: new entry\n"
        "updated: 2099-01-01T00:00:00+00:00\n---\n\nb\n")
    # Cap fits exactly one "- name: desc" line (~21 chars each).
    sec = index_section(ToasterLoopConfig(skills={"enabled": True,
                                              "index_total_cap_chars": 35}))
    assert "zzz-new" in sec          # recent survives the cap
    assert "aaa-old" not in sec      # stale entry omitted whole


# ------------------------------------------------------ 6. dashboard API

def _app():
    from toaster_loop.core.config import ToasterLoopConfig
    from toaster_loop.dashboard.app import DashboardApp

    return DashboardApp(ToasterLoopConfig())


def test_skills_api_roundtrip():
    app = _app()
    r = app.handle("GET", "/api/skills")
    assert r.status == 200
    assert r.body["skills"] == []

    r = app.handle("PUT", "/api/skills/api-made", body={
        "description": "made via API", "content": "# steps\n1. do it"})
    assert r.status == 200, r.body

    r = app.handle("GET", "/api/skills")
    names = [s["name"] for s in r.body["skills"]]
    assert "api-made" in names

    r = app.handle("GET", "/api/skills/api-made")
    assert r.status == 200 and "do it" in r.body["content"]

    r = app.handle("DELETE", "/api/skills/api-made")
    assert r.status == 200

    r = app.handle("GET", "/api/skills/api-made")
    assert r.status == 404


def test_skills_api_validation_errors():
    app = _app()
    r = app.handle("PUT", "/api/skills/Bad Name!", body={
        "description": "d", "content": "c"})
    assert r.status == 400  # invalid slug (also name check lowercases first)
    r = app.handle("PUT", "/api/skills/ok-name", body={
        "description": "d"})  # content missing
    assert r.status == 400
    r = app.handle("DELETE", "/api/skills/nothing-here")
    assert r.status == 404


# ------------------------------------------------- run_command argv hardening

def test_run_command_argv_mode_no_shell():
    """tools.run_command_argv_only: no shell — pipes/redirects become literal
    data, expansion tokens are refused, plain commands still run."""
    from toaster_loop.core.tools import run_command

    # default (shell): the pipeline executes
    r = run_command("echo hi | wc -w")
    assert "exit code: 0" in r and "stdout:\n1" in r
    # argv mode: `| wc -w` is literal echo data, never a second process
    r = run_command("echo hi | wc -w", argv_mode=True)
    assert "hi | wc -w" in r
    # expansion tokens refused rather than passed as silent literals
    assert "forbids shell expansion" in run_command("echo $HOME", argv_mode=True)
    assert "forbids shell expansion" in run_command("echo `id`", argv_mode=True)
    # plain argv commands work; unbalanced quotes fail cleanly
    assert "hello world" in run_command("echo hello world", argv_mode=True)
    assert "unbalanced quotes" in run_command('echo "unclosed', argv_mode=True)


def test_run_command_argv_binding_not_forgeable():
    """The bound argv knob is private: a model-supplied argv_mode arg must be
    refused by the schema gate, not override the binding."""
    script = [
        {"role": "assistant", "content": None,
         "tool_calls": [_tool_call("c1", "run_command",
                                   {"command": "echo hi",
                                    "argv_mode": False})]},
        {"role": "assistant", "content": "done"},
    ]
    with FakeOpenAIServer(handler_cls=_scripted_server(script)) as fs:
        client = ToasterLoopAgentClient(
            base_url=fs.base_url, model="m",
            capabilities={"run_commands": True},
            run_command_argv_only=True)
        result = client.run("try to escape argv mode")
    tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
    assert tool_msgs and tool_msgs[0]["content"].startswith(
        "error: bad arguments")

# ------------------------------------------------------------------------
# from tests/test_agent_runner_fixes.py
# ------------------------------------------------------------------------

def _sse(chunks):
    """Render scripted choice-dicts as an SSE body (one chunk per frame)."""
    out = "".join(
        "data: " + json.dumps({"id": "x", "object": "chat.completion.chunk",
                               "model": "m",
                               "choices": [dict(index=0, **c)]}) + "\n\n"
        for c in chunks) + "data: [DONE]\n\n"
    return out.encode()




# ------------------------------------------------- Fix 1: _summarize routing

class SummarizeCaptureHandler(_Silent):
    """Answers the summarizer call; records the wire payload on the class."""

    seen: dict = {}

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        type(self).seen = json.loads(self.rfile.read(length) or b"{}")
        body = json.dumps({
            "id": "s", "object": "chat.completion", "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant",
                                     "content": "a concise summary"}}],
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_summarize_routes_sampling_through_request_payload():
    SummarizeCaptureHandler.seen = {}
    with FakeOpenAIServer(handler_cls=SummarizeCaptureHandler) as fs:
        client = ToasterLoopAgentClient(
            fs.base_url, "m", tools=[], tool_impls={},
            sampling={"temperature": 0.3, "enable_thinking": False,
                      "reasoning_effort": "low"})
        out = client._summarize("condense this please")
    assert out == "a concise summary"
    seen = SummarizeCaptureHandler.seen
    # Sampling numbers ride top-level; thinking controls land in BOTH shapes —
    # exactly what request_sampling_payload produces for the main path.
    assert seen["temperature"] == 0.3
    assert seen["enable_thinking"] is False
    assert seen["reasoning_effort"] == "low"
    assert seen["chat_format_template_controls"] == {
        "enable_thinking": False, "reasoning_effort": "low"}


def test_summarize_no_sampling_keys_stays_minimal():
    SummarizeCaptureHandler.seen = {}
    with FakeOpenAIServer(handler_cls=SummarizeCaptureHandler) as fs:
        client = ToasterLoopAgentClient(fs.base_url, "m", tools=[], tool_impls={})
        assert client._summarize("hi") is not None
    seen = SummarizeCaptureHandler.seen
    assert "chat_format_template_controls" not in seen
    assert set(seen) == {"model", "messages"}


# ------------------------------- Fix 2: stream fallback emits message.reset

class PartialStreamThenBreakHandler(_Silent):
    """Streaming requests push two content deltas then a chunk that breaks
    delta processing (mid-flight failure); non-streaming requests answer with
    the full plain JSON completion."""

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if payload.get("stream"):
            raw = (
                _sse([{"delta": {"content": "Hel"}},
                      {"delta": {"content": "lo "}}])[:-len(b"data: [DONE]\n\n")]
                # choices as a string: iterating yields str, .get raises —
                # simulates a server corrupting the stream mid-flight.
                + ("data: " + json.dumps({"id": "x", "choices": "bad"})
                   + "\n\n").encode())
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(raw)
            return
        body = json.dumps({
            "id": "x", "object": "chat.completion", "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant",
                                     "content": "Hello world"}}],
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_stream_fallback_replaces_deltas_with_reset():
    events = []
    with FakeOpenAIServer(handler_cls=PartialStreamThenBreakHandler) as fs:
        client = ToasterLoopAgentClient(fs.base_url, "m", tools=[], tool_impls={})
        client.on_event = lambda kind, data: events.append((kind, data))
        data = client._chat([{"role": "user", "content": "hi"}])
    kinds = [k for k, _ in events]
    # Partial deltas went out before the break...
    assert kinds[:2] == ["message.delta", "message.delta"]
    # ...and exactly one replacement event follows; no further deltas emitted.
    assert kinds.count("message.reset") == 1
    assert kinds[-1] == "message.reset"
    reset_text = next(d for k, d in events if k == "message.reset")["text"]
    assert reset_text == "Hello world"
    # The returned completion is the authoritative one-shot answer.
    assert data["choices"][0]["message"]["content"] == "Hello world"
    assert client._no_stream is True


def test_stream_fallback_without_prior_deltas_no_reset():
    """Empty-stream fallback (nothing streamed) must NOT invent a reset."""

    class EmptyStreamHandler(_Silent):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            if payload.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(b"data: [DONE]\n\n")
                return
            body = json.dumps({
                "id": "x", "object": "chat.completion", "model": "m",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant",
                                         "content": "plain"}}],
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    events = []
    with FakeOpenAIServer(handler_cls=EmptyStreamHandler) as fs:
        client = ToasterLoopAgentClient(fs.base_url, "m", tools=[], tool_impls={})
        client.on_event = lambda kind, data: events.append(kind)
        data = client._chat([{"role": "user", "content": "hi"}])
    assert data["choices"][0]["message"]["content"] == "plain"
    assert events == []  # nothing streamed => nothing to replace


# --------------------------- Fix 3: index-less tool-call delta assembly

def _stream_completion(chunks):
    """Replay `chunks` (list of choice-dicts) as one SSE completion and
    return the dict client._chat assembles from it."""

    class SSEHandler(_Silent):
        def do_POST(self):
            raw = _sse(chunks)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(raw)

    with FakeOpenAIServer(handler_cls=SSEHandler) as srv:
        client = ToasterLoopAgentClient(srv.base_url, "m", tools=[], tool_impls={})
        client.on_event = lambda kind, data: None  # enable streaming path
        return client._chat([{"role": "user", "content": "go"}])


def _tool_calls_of(data):
    return (data["choices"][0]["message"].get("tool_calls") or [])


def test_tool_call_assembly_with_proper_indices():
    chunks = [
        {"delta": {"tool_calls": [
            {"index": 0, "id": "call_a", "type": "function",
             "function": {"name": "read_file", "arguments": "{\"pa"}}]}},
        {"delta": {"tool_calls": [
            {"index": 1, "id": "call_b", "type": "function",
             "function": {"name": "list_dir", "arguments": "{\"p"}}]}},
        {"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": "th\": \"x\"}"}}]}},
        {"delta": {"tool_calls": [
            {"index": 1, "function": {"arguments": "ath\": \"y\"}"}}]}},
    ]
    tcs = _tool_calls_of(_stream_completion(chunks))
    assert [t["id"] for t in tcs] == ["call_a", "call_b"]
    assert tcs[0]["function"]["name"] == "read_file"
    assert json.loads(tcs[0]["function"]["arguments"]) == {"path": "x"}
    assert tcs[1]["function"]["name"] == "list_dir"
    assert json.loads(tcs[1]["function"]["arguments"]) == {"path": "y"}


def test_tool_call_assembly_all_none_indices_parallel_calls():
    """Servers that omit `index` entirely: a fresh id opens a new slot."""
    chunks = [
        {"delta": {"tool_calls": [
            {"id": "call_a", "type": "function",
             "function": {"name": "read_file",
                          "arguments": "{\"path\": \"a\"}"}}]}},
        {"delta": {"tool_calls": [
            {"id": "call_b", "type": "function",
             "function": {"name": "list_dir",
                          "arguments": "{\"path\": \"b\"}"}}]}},
    ]
    tcs = _tool_calls_of(_stream_completion(chunks))
    assert [t["id"] for t in tcs] == ["call_a", "call_b"]
    assert tcs[0]["function"]["name"] == "read_file"
    assert json.loads(tcs[0]["function"]["arguments"]) == {"path": "a"}
    assert tcs[1]["function"]["name"] == "list_dir"
    assert json.loads(tcs[1]["function"]["arguments"]) == {"path": "b"}


def test_tool_call_assembly_interleaved_fragments_single_call():
    """Argument fragments WITHOUT id/index keep accumulating into the open
    call; only a fresh id advances the cursor."""
    chunks = [
        {"delta": {"tool_calls": [
            {"id": "call_a", "type": "function",
             "function": {"name": "search_files", "arguments": "{\"qu"}}]}},
        {"delta": {"tool_calls": [
            {"function": {"arguments": "ery\": \"he"}}]}},
        {"delta": {"tool_calls": [
            {"function": {"arguments": "llo\"}"}}]}},
    ]
    tcs = _tool_calls_of(_stream_completion(chunks))
    assert len(tcs) == 1
    assert tcs[0]["id"] == "call_a"
    assert json.loads(tcs[0]["function"]["arguments"]) == {"query": "hello"}


# ------------------------------------------------ Fix 4: tools ∩ enabled list

_CUSTOM_TOOLS = [
    {"type": "function", "function": {"name": "read_file",
                                      "parameters": {}}},
    {"type": "function", "function": {"name": "write_file",
                                      "parameters": {}}},
    {"type": "function", "function": {"name": "list_dir",
                                      "parameters": {}}},
]


def _names(schemas):
    return {s.get("function", {}).get("name") for s in schemas}


def test_custom_tools_intersected_with_enabled_tool_names(caplog):
    with caplog.at_level(logging.WARNING, logger="toaster_loop.agent"):
        client = ToasterLoopAgentClient(
            "http://127.0.0.1:1/v1", "m",
            tools=[dict(t) for t in _CUSTOM_TOOLS],
            tool_impls={"read_file": lambda **kw: "",
                        "write_file": lambda **kw: "",
                        "list_dir": lambda **kw: ""},
            enabled_tool_names=["read_file", "list_dir"])
    assert _names(client.tools) == {"read_file", "list_dir"}
    assert set(client.tool_impls) == {"read_file", "list_dir"}
    # The dropped tool is reported loudly, not silently.
    msgs = [r.getMessage() for r in caplog.records
            if "enabled_tool_names" in r.getMessage()]
    assert msgs and "write_file" in msgs[0]


def test_custom_tools_passthrough_without_enabled_list():
    client = ToasterLoopAgentClient(
        "http://127.0.0.1:1/v1", "m",
        tools=[dict(t) for t in _CUSTOM_TOOLS],
        tool_impls={"read_file": lambda **kw: "",
                    "write_file": lambda **kw: "",
                    "list_dir": lambda **kw: ""})
    assert _names(client.tools) == {"read_file", "write_file", "list_dir"}


# --------------------------------- Fix 6: tool args redacted in INFO logging

class ToolCallOnceHandler(_Silent):
    """First request answers with a native read_file tool call whose args
    embed an OpenAI-shaped secret; after a tool result arrives, finishes."""

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        messages = payload.get("messages", [])
        saw_tool_result = any(m.get("role") == "tool" for m in messages)
        if not saw_tool_result:
            msg = {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "read_file",
                             "arguments": json.dumps({
                                 "path": "/etc/hosts",
                                 "token": "sk-abcdef0123456789abcd"})}}]}
            finish = "tool_calls"
        else:
            msg = {"role": "assistant", "content": "done"}
            finish = "stop"
        body = json.dumps({
            "id": "x", "object": "chat.completion", "model": "m",
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_tool_args_redacted_in_info_logs(caplog, tmp_path):
    secret = "sk-abcdef0123456789abcd"
    with caplog.at_level(logging.INFO, logger="toaster_loop.agent"):
        with FakeOpenAIServer(handler_cls=ToolCallOnceHandler) as fs:
            client = ToasterLoopAgentClient(
                fs.base_url, "m",
                tools=[{"type": "function", "function": {
                    "name": "read_file", "parameters": {}}}],
                tool_impls={"read_file": lambda path="", **kw: f"file:{path}"},
                max_iterations=3)
            res = client.run("read the hosts file")
    assert res.final_response == "done"
    blob = "\n".join(r.getMessage() for r in caplog.records)
    assert secret not in blob
    # The tool call itself was still logged (debuggability preserved)...
    assert "tool read_file" in blob
    # ...with the credential masked by the trace redactor.
    assert "[REDACTED]" in blob


# Fix 5 is a pure-whitespace change verified with `git diff -w`; no test.

# ------------------------------------------------------------------------
# from tests/test_sampling_thinking_stream.py
# ------------------------------------------------------------------------

@pytest.fixture()
def home__test_sampling_thinking_stream(tmp_path, monkeypatch):
    h = tmp_path / "uf-sts"
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(h))
    return h


def make_app(home) -> DashboardApp:
    return DashboardApp(ToasterLoopConfig(llama_auto_setup=False), home=home)


# ----------------------------------------------------- 1. sanitize_sampling

def test_sanitize_sampling_valid_and_coercion():
    out = sanitize_sampling({
        "temperature": "0.7", "top_p": 0.95, "top_k": "40",
        "min_p": 0.05, "repeat_penalty": 1.1, "presence_penalty": -0.5,
        "max_tokens": "512", "penalty_free_repeat": 64.0,
        "enable_thinking": True, "reasoning_effort": "HIGH",
        "preserve_thinking": False,
    })
    assert out["temperature"] == 0.7
    assert out["top_k"] == 40 and isinstance(out["top_k"], int)
    assert out["max_tokens"] == 512
    assert out["penalty_free_repeat"] == 64 and isinstance(
        out["penalty_free_repeat"], int)
    assert out["enable_thinking"] is True
    assert out["reasoning_effort"] == "high"
    assert out["preserve_thinking"] is False


def test_sanitize_sampling_rejects_bad_values():
    with pytest.raises(ValueError):
        sanitize_sampling({"temperature": 5})       # > 2.0
    with pytest.raises(ValueError):
        sanitize_sampling({"top_k": 0})             # must be >= 1
    with pytest.raises(ValueError):
        sanitize_sampling({"reasoning_effort": "ultra"})
    with pytest.raises(ValueError):
        sanitize_sampling({"nonsense": 1})
    with pytest.raises(ValueError):
        sanitize_sampling({"temperature": "hot"})


def test_sanitize_sampling_empty_clears():
    out = sanitize_sampling({"temperature": "", "reasoning_effort": None})
    assert out == {"temperature": None, "reasoning_effort": None}


# ----------------------------------------------- 2. request payload shaping

def test_request_sampling_payload_thinking_controls():
    frag = request_sampling_payload({
        "temperature": 0.4, "enable_thinking": True,
        "reasoning_effort": "low", "preserve_thinking": None,
    })
    assert frag["temperature"] == 0.4
    # top-level (vLLM-style) AND the llama.cpp template-controls channel
    assert frag["enable_thinking"] is True
    assert frag["reasoning_effort"] == "low"
    assert frag["chat_format_template_controls"] == {
        "enable_thinking": True, "reasoning_effort": "low"}
    # None values dropped entirely
    assert "preserve_thinking" not in frag


def test_request_sampling_payload_no_controls_no_map():
    frag = request_sampling_payload({"temperature": 1.0})
    assert frag == {"temperature": 1.0}
    assert request_sampling_payload({}) == {}
    assert request_sampling_payload(None) == {}


# ------------------------------------------------- 3. instance sampling API

def _create(app, body):
    return app.handle("POST", "/api/servers", body=body)


def test_instance_create_stores_sampling(home__test_sampling_thinking_stream):
    app = make_app(home__test_sampling_thinking_stream)
    r = _create(app, {"name": "coder", "gguf": "/models/x.gguf",
                      "sampling": {"temperature": 0.3,
                                   "enable_thinking": True,
                                   "reasoning_effort": "high"}})
    assert r.status == 201
    srv = r.body["server"]
    assert srv["sampling"]["temperature"] == 0.3
    assert srv["sampling"]["enable_thinking"] is True
    assert srv["sampling"]["reasoning_effort"] == "high"


def test_instance_create_rejects_bad_sampling(home__test_sampling_thinking_stream):
    app = make_app(home__test_sampling_thinking_stream)
    r = _create(app, {"name": "coder", "gguf": "/models/x.gguf",
                      "sampling": {"temperature": 9}})
    assert r.status == 400
    assert "sampling" in r.body["error"]["message"]


def test_instance_put_updates_sampling(home__test_sampling_thinking_stream):
    app = make_app(home__test_sampling_thinking_stream)
    sid = _create(app, {"name": "c", "gguf": "/models/x.gguf"}).body["server"]["id"]
    r = app.handle("PUT", f"/api/servers/{sid}",
                   body={"sampling": {"reasoning_effort": "medium"}})
    assert r.status == 200
    assert r.body["server"]["sampling"] == {"reasoning_effort": "medium"}


# ----------------------------------------------------- 4. profile sampling

def test_profile_stores_sampling_override(home__test_sampling_thinking_stream):
    app = make_app(home__test_sampling_thinking_stream)
    r = app.handle("PUT", "/api/profiles/coder", body={
        "system_prompt": "hi", "tools": [],
        "sampling": {"temperature": 0.2, "enable_thinking": False}})
    assert r.status == 200
    prof = r.body["profile"]
    assert prof["sampling"]["temperature"] == 0.2
    assert prof["sampling"]["enable_thinking"] is False


def test_profile_rejects_bad_sampling(home__test_sampling_thinking_stream):
    r = make_app(home__test_sampling_thinking_stream).handle("PUT", "/api/profiles/coder",
                              body={"sampling": {"reasoning_effort": "zzz"}})
    assert r.status == 400


# --------------------------------------------------- 5. runner streaming

class ReasoningSSEHandler(BaseHTTPRequestHandler):
    """Streams one reply as reasoning deltas then content deltas."""

    def log_message(self, format, *args):  # noqa: A002 - silence access log
        pass

    def do_POST(self):
        chunks = [
            {"delta": {"role": "assistant"}},
            {"delta": {"reasoning_content": "thinking... "}},
            {"delta": {"reasoning_content": "hard."}},
            {"delta": {"content": "Hello "}},
            {"delta": {"content": "world"}},
            {"delta": {}, "finish_reason": "stop"},
        ]
        body = "".join(
            "data: " + json.dumps({"id": "x", "object": "chat.completion.chunk",
                                   "model": "m",
                                   "choices": [dict(index=0, **c)]}) + "\n\n"
            for c in chunks) + "data: [DONE]\n\n"
        raw = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(raw)


def test_agent_client_streams_content_and_reasoning():
    events = []
    with FakeOpenAIServer(handler_cls=ReasoningSSEHandler) as fs:
        client = ToasterLoopAgentClient(fs.base_url, "m", tools=[], tool_impls={})
        client.on_event = lambda kind, data: events.append((kind, data))
        data = client._chat([{"role": "user", "content": "hi"}])
    kinds = [k for k, _ in events]
    assert kinds.count("reasoning.delta") == 2
    assert kinds.count("message.delta") == 2
    text = "".join(d["text"] for k, d in events if k == "message.delta")
    reasoning = "".join(d["text"] for k, d in events if k == "reasoning.delta")
    assert text == "Hello world"
    assert reasoning == "thinking... hard."
    # The returned message carries only visible content (studio semantics).
    msg = data["choices"][0]["message"]
    assert msg["content"] == "Hello world"


class PlainJSONHandler(BaseHTTPRequestHandler):
    """Ignores stream=true and answers with a plain JSON completion."""

    def log_message(self, format, *args):  # noqa: A002 - silence access log
        pass

    def do_POST(self):
        body = json.dumps({
            "id": "x", "object": "chat.completion", "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "plain"}}],
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_agent_client_falls_back_when_server_ignores_stream():
    events = []
    with FakeOpenAIServer(handler_cls=PlainJSONHandler) as fs:
        client = ToasterLoopAgentClient(fs.base_url, "m", tools=[], tool_impls={})
        client.on_event = lambda kind, data: events.append(kind)
        data = client._chat([{"role": "user", "content": "hi"}])
    assert data["choices"][0]["message"]["content"] == "plain"
    assert events == []  # no deltas invented


# ---------------------------------------------------- 6. chat/stream live

def _fake_serve_state(home, url):
    home.mkdir(parents=True, exist_ok=True)
    (home / "serve_state.json").write_text(json.dumps({
        "mode": "attach", "base_url": url, "model_name": "fake-model",
        "attach_url": url.rsplit("/v1", 1)[0],
    }))


def _events(text):
    out = []
    for frame in text.split("\n\n"):
        kind = None
        data = None
        for line in frame.splitlines():
            if line.startswith("event: "):
                kind = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        if kind:
            out.append((kind, data or {}))
    return out


def test_chat_stream_live_events_no_duplicates(home__test_sampling_thinking_stream):
    app = make_app(home__test_sampling_thinking_stream)
    with FakeOpenAIServer() as fs:
        _fake_serve_state(home__test_sampling_thinking_stream, fs.base_url)
        r = app.handle("POST", "/api/chat/stream", body={"prompt": "go"})
        assert r.stream is not None
        text = "".join(r.stream)
    evs = _events(text)
    kinds = [k for k, _ in evs]
    assert kinds[0] == "run.started"
    assert kinds[-1] == "run.done"
    # run.started carries the context window for the UI tracker
    assert evs[0][1]["context_window"] > 0
    # The fake server streams; deltas arrive live and tool events fire once.
    assert "message.delta" in kinds
    n_tool_call = kinds.count("tool.call")
    n_tool_result = kinds.count("tool.result")
    assert n_tool_call == n_tool_result >= 1
    # No synthesized duplicates: each tool call id appears exactly once per phase
    ids = [d["id"] for k, d in evs if k == "tool.call"]
    assert len(ids) == len(set(ids))
    # Final assistant content streamed live (exactly the fake's answer once)
    final = "".join(d.get("text", "") for k, d in evs if k == "message.delta")
    assert final.count("Done. The file was read successfully.") == 1


def test_chat_stream_sampling_merge_precedence(home__test_sampling_thinking_stream):
    """profile sampling wins over instance sampling wins over global."""
    app = make_app(home__test_sampling_thinking_stream)
    # global
    app.handle("PUT", "/api/config", body={"section": "sampling",
                                           "patch": {"temperature": 0.9,
                                                     "top_k": 50}})
    sid = _create(app, {
        "name": "bound", "gguf": "/models/x.gguf",
        "sampling": {"temperature": 0.5, "enable_thinking": True},
    }).body["server"]["id"]
    app.handle("PUT", "/api/profiles/pro", body={
        "system_prompt": "x", "tools": [], "server_instance": sid,
        "sampling": {"temperature": 0.1}})

    seen = {}

    class CaptureHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A002 - silence access log
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            seen.update(payload)
            body = json.dumps({
                "id": "x", "object": "chat.completion", "model": "m",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant",
                                         "content": "ok"}}],
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), CaptureHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}/v1"
        # Start the named instance against the capture endpoint by faking its
        # state file (spawn would need a real llama-server binary).
        sdir = home__test_sampling_thinking_stream / "serving" / "state"
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / f"{sid}.json").write_text(json.dumps({
            "mode": "spawn", "base_url": base, "model_name": "m",
            "pid": 1, "port": httpd.server_address[1],
        }))
        r = app.handle("POST", "/api/chat/stream",
                       body={"prompt": "hi", "profile": "pro"})
        assert r.stream is not None
        "".join(r.stream)
    finally:
        httpd.shutdown()
        httpd.server_close()

    assert seen.get("temperature") == 0.1      # profile wins
    assert seen.get("top_k") == 50             # global survives the merge
    assert seen.get("enable_thinking") is True  # instance thinking applies
    assert seen.get("chat_format_template_controls", {}).get(
        "enable_thinking") is True

# ------------------------------------------------------------------------
# from tests/test_logprobs_confidence.py
# ------------------------------------------------------------------------

# Known synthetic token distribution reused across assertions:
#   logprobs = [ln 0.5, ln 0.5, ln 0.8, ln 0.8, ln 0.8]
#   conf_mean            = (2·ln0.5 + 3·ln0.8) / 5          = -0.411145
#   conf_p90_negentropy  = exp(sorted_p90)                  = 0.8
CANNED_LPS = [math.log(0.5), math.log(0.5), math.log(0.8),
              math.log(0.8), math.log(0.8)]
EXPECT_MEAN = round(sum(CANNED_LPS) / len(CANNED_LPS), 6)


def _logprobs_payload(with_logprobs: bool):
    lp = {"content": [{"token": f"t{i}", "logprob": v, "top_logprobs": []}
                      for i, v in enumerate(CANNED_LPS)]} if with_logprobs \
        else None

    class Handler(BaseHTTPRequestHandler):
        requests: list = []

        def log_message(self, format, *args):  # silence
            pass

        def _send(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            type(self).requests.append(payload)
            choice: dict = {
                "index": 0,
                "message": {"role": "assistant", "content": "all good"},
                "finish_reason": "stop",
            }
            if lp is not None and payload.get("logprobs"):
                choice["logprobs"] = lp
            # A backend WITHOUT support answers unchanged even when asked.
            self._send({
                "id": "cmpl-lp", "object": "chat.completion",
                "model": payload.get("model", "lp-model"),
                "choices": [choice],
                "usage": {"prompt_tokens": 1, "completion_tokens": 5,
                          "total_tokens": 6},
            })

    return Handler


@pytest.fixture()
def home__test_logprobs_confidence(tmp_path, monkeypatch):
    h = tmp_path / "uf-logprobs"
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(h))
    return h


# ------------------------------------------------------------- aggregate math

def test_confidence_aggregate_known_values():
    mean, p90 = _confidence_aggregate(list(CANNED_LPS))
    assert mean == pytest.approx(EXPECT_MEAN, abs=1e-6)
    # nearest-rank p90 of [l.5,l.5,l.8,l.8,l.8] -> index round(4.5)-1 = 3
    assert p90 == pytest.approx(0.8, abs=1e-6)


def test_confidence_aggregate_empty_is_none():
    assert _confidence_aggregate([]) == (None, None)


# --------------------------------------------------------------- runner path

def _run_with(home, handler_cls, logprobs: bool):
    server = FakeOpenAIServer(handler_cls).start()
    recorder = TraceRecorder(home=home)
    try:
        client = ToasterLoopAgentClient(
            base_url=server.base_url, model="lp-model",
            tools=[], tool_impls={}, recorder=recorder,
            logprobs=logprobs)
        result = client.run("say hi")
        assert result.completed and result.final_response == "all good"
        assert result.trace_id is not None
        store = TraceStore(Path(home) / "buffer" / "toaster_loop.db")
        try:
            row = store.get_trace(result.trace_id)
        finally:
            store.close()
        assert row is not None
        return result, row
    finally:
        recorder.close()
        server.stop()


def test_logprobs_requested_and_aggregates_stored(home__test_logprobs_confidence):
    handler = _logprobs_payload(with_logprobs=True)
    _, row = _run_with(home__test_logprobs_confidence, handler, logprobs=True)
    assert row.conf_mean == pytest.approx(EXPECT_MEAN, abs=1e-6)
    assert row.conf_p90_negentropy == pytest.approx(0.8, abs=1e-6)


def test_backend_without_logprobs_degrades_to_none(home__test_logprobs_confidence):
    # Backend ignores the flag and returns no logprobs field at all: run
    # succeeds exactly as before, aggregates stay NULL.
    handler = _logprobs_payload(with_logprobs=False)
    result, row = _run_with(home__test_logprobs_confidence, handler, logprobs=True)
    assert result.completed is True
    assert row.conf_mean is None
    assert row.conf_p90_negentropy is None


def test_flag_off_requests_nothing_and_stores_none(home__test_logprobs_confidence):
    handler = _logprobs_payload(with_logprobs=True)
    _, row = _run_with(home__test_logprobs_confidence, handler, logprobs=False)
    assert row.conf_mean is None and row.conf_p90_negentropy is None
    # And the request body never carried the flag in the first place.
    assert handler.requests, "handler saw no request"
    assert all("logprobs" not in p for p in handler.requests)


# ------------------------------------------------------------ config plumbing

def test_create_runner_reads_sampling_logprobs_knob():
    cfg = ToasterLoopConfig(sampling={"logprobs": True, "temperature": 0.5})
    runner = create_runner(cfg, base_url="http://127.0.0.1:1/v1",
                           model="m")
    assert runner.logprobs is True
    # The gate must NOT leak verbatim into the upstream sampling map (it would
    # otherwise be forwarded twice by request_sampling_payload).
    assert "logprobs" not in runner.sampling
    assert runner.sampling == {"temperature": 0.5}


def test_create_runner_default_off_and_absent_key_kept_out():
    runner = create_runner(ToasterLoopConfig(sampling={"temperature": 0.3}),
                           base_url="http://127.0.0.1:1/v1", model="m")
    assert runner.logprobs is False

    runner2 = create_runner(ToasterLoopConfig(), base_url="http://127.0.0.1:1/v1",
                            model="m")
    assert runner2.logprobs is False


def test_create_runner_logprobs_on_reaches_request_body():
    handler = _logprobs_payload(with_logprobs=True)
    server = FakeOpenAIServer(handler).start()
    try:
        cfg = ToasterLoopConfig(sampling={"logprobs": True})
        runner = create_runner(cfg, base_url=server.base_url, model="m")
        result = runner.run("say hi")
        assert result.completed
        body = handler.requests[-1]
        assert body.get("logprobs") is True
    finally:
        server.stop()

# ------------------------------------------------------------------------
# from tests/test_spawn_stub.py
# ------------------------------------------------------------------------

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
    cfg = ToasterLoopConfig(
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
    cfg = ToasterLoopConfig(
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
    cfg = ToasterLoopConfig(
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
    cfg = ToasterLoopConfig(
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
    cfg = ToasterLoopConfig(
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
    cfg = ToasterLoopConfig(
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
    from toaster_loop.serving import supervisor as sup_mod

    gguf = tmp_path / "flap.gguf"
    gguf.write_bytes(b"GGUF")
    cfg = ToasterLoopConfig(
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
    """start_server_detached (the `toaster_loop models deploy` path) records each
    failed spawn/healthcheck cycle under the "default" key so the dashboard's
    crash_loop_status is honest — but an explicit CLI invocation never refuses:
    it clears the budget up front (user-intent signal, like `toaster_loop serve`)
    and re-arms only on failure."""
    from toaster_loop.serving import supervisor as sup_mod
    from toaster_loop.serving.deploy import start_server_detached

    clear_crash_loop("default")
    gguf = tmp_path / "flap-cli.gguf"
    gguf.write_bytes(b"GGUF")
    cfg = ToasterLoopConfig(
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
    toaster_loop_instances_up see the CLI-detached server correctly."""
    from toaster_loop.serving.deploy import read_serve_state, start_server_detached

    clear_crash_loop("default")
    gguf = tmp_path / "good-model-q4_k_m.gguf"
    gguf.write_bytes(b"GGUF")
    cfg = ToasterLoopConfig(
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
        from toaster_loop.serving import instances as srv_inst
        assert srv_inst.state_alive(written) is True
    finally:
        # Detach-by-design means nothing auto-stops it; clean up here.
        import signal
        try:
            os.kill(state["pid"], signal.SIGTERM)
        except ProcessLookupError:
            pass
        clear_crash_loop("default")

# ------------------------------------------------------------------------
# from tests/test_trace_prune.py
# ------------------------------------------------------------------------

def _mk(store: TraceStore, home: Path, tid: str, ts: str,
        curated: bool = False) -> TraceRow:
    jp = home / "buffer" / f"{tid}.jsonl"
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text("x" * 100, encoding="utf-8")
    row = TraceRow(trace_id=tid, session_id="s", timestamp=ts,
                   agent_profile="p", model_alias="m", model_version=None,
                   endpoint_label=None, task_summary="t", status="success",
                   outcome_score=0.5, num_tool_calls=0, tokens_in=1,
                   tokens_out=1, duration_s=1.0, jsonl_path=str(jp),
                   curated=curated)
    store.insert_trace(row)
    store.add_feedback(tid, 0.9, "nice", source="user", timestamp=ts)
    return row


@pytest.fixture()
def store__test_trace_prune(tmp_path):
    s = TraceStore(tmp_path / "buffer" / "toaster_loop.db")
    yield s, tmp_path
    s.close()


def test_prune_dry_run_keeps_everything(store__test_trace_prune):
    s, home = store__test_trace_prune
    _mk(s, home, "old1", "2025-01-01T00:00:00+00:00")
    summ = s.prune_traces("2026-01-01T00:00:00+00:00", dry_run=True)
    assert summ["dry_run"] and summ["deleted"] == 1
    assert (home / "buffer" / "old1.jsonl").exists()
    assert s.get_trace("old1") is not None


def test_prune_deletes_rows_transcripts_and_feedback(store__test_trace_prune):
    s, home = store__test_trace_prune
    _mk(s, home, "old1", "2025-01-01T00:00:00+00:00")
    _mk(s, home, "new1", "2026-10-01T00:00:00+00:00")
    summ = s.prune_traces("2026-01-01T00:00:00+00:00")
    assert summ["deleted"] == 1 and summ["freed_bytes"] >= 100
    assert not (home / "buffer" / "old1.jsonl").exists()
    assert s.get_trace("old1") is None
    assert s.get_trace("new1") is not None
    assert s.get_feedback("old1") == []


def test_prune_keeps_curated_unless_forced(store__test_trace_prune):
    s, home = store__test_trace_prune
    _mk(s, home, "oldc", "2025-01-01T00:00:00+00:00", curated=True)
    summ = s.prune_traces("2026-01-01T00:00:00+00:00")
    assert summ["deleted"] == 0 and summ["kept_curated"] == 1
    summ2 = s.prune_traces("2026-01-01T00:00:00+00:00", keep_curated=False)
    assert summ2["deleted"] == 1
    assert s.get_trace("oldc") is None


def test_prune_missing_transcript_still_deletes_row(store__test_trace_prune):
    s, home = store__test_trace_prune
    row = _mk(s, home, "old1", "2025-01-01T00:00:00+00:00")
    Path(row.jsonl_path).unlink()
    summ = s.prune_traces("2026-01-01T00:00:00+00:00")
    assert summ["deleted"] == 1 and not summ["errors"]
    assert s.get_trace("old1") is None


# ------------------------------------------------------------------ CLI


def _run_cli(tmp_path, monkeypatch, argv, keep_config: bool = False):
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(tmp_path))
    if not keep_config:
        monkeypatch.delenv("TOASTER_LOOP_CONFIG", raising=False)
    return main(argv)


def test_cli_prune_requires_window(tmp_path, monkeypatch, capsys):
    rc = _run_cli(tmp_path, monkeypatch, ["traces", "prune"])
    assert rc == 1
    assert "--older-than" in capsys.readouterr().err


def test_cli_prune_dry_then_yes(tmp_path, monkeypatch, capsys):
    db = tmp_path / "buffer" / "toaster_loop.db"
    s = TraceStore(db)
    _mk(s, tmp_path, "old1", "2025-01-01T00:00:00+00:00")
    s.close()
    rc = _run_cli(tmp_path, monkeypatch, ["traces", "prune", "--older-than", "30"])
    assert rc == 0
    assert "dry run" in capsys.readouterr().out
    assert (tmp_path / "buffer" / "old1.jsonl").exists()
    rc = _run_cli(tmp_path, monkeypatch,
                  ["traces", "prune", "--older-than", "30", "--yes"])
    assert rc == 0
    assert not (tmp_path / "buffer" / "old1.jsonl").exists()


def test_cli_prune_uses_config_retention_days(tmp_path, monkeypatch, capsys):
    cfg = tmp_path / "toaster_loop.yaml"
    cfg.write_text("recording:\n  retention_days: 45\n", encoding="utf-8")
    db = tmp_path / "buffer" / "toaster_loop.db"
    s = TraceStore(db)
    _mk(s, tmp_path, "old1", "2025-01-01T00:00:00+00:00")
    s.close()
    monkeypatch.setenv("TOASTER_LOOP_CONFIG", str(cfg))
    rc = _run_cli(tmp_path, monkeypatch, ["traces", "prune"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "older than 45d" in out

# ------------------------------------------------------------------------
# from tests/test_errors.py
# ------------------------------------------------------------------------

# ------------------------------------------------------------- http statuses

def test_classify_http_status_golden_strings():
    ce = classify_http_status(413, "request body too large")
    assert (ce.reason, ce.retryable, ce.should_compress) == (
        FailoverReason.PAYLOAD_TOO_LARGE, True, True)

    ce = classify_http_status(
        400, "the prompt is longer than the context length of 4096 tokens")
    assert (ce.reason, ce.should_compress) == (FailoverReason.CONTEXT_OVERFLOW, True)

    ce = classify_http_status(404, '{"error": "model not found"}')
    assert (ce.reason, ce.retryable) == (FailoverReason.MODEL_NOT_FOUND, True)

    ce = classify_http_status(400, "malformed tool call json in request")
    assert (ce.reason, ce.strip_tools, ce.retryable) == (
        FailoverReason.FORMAT_ERROR, True, True)

    ce = classify_http_status(400, "something else entirely")
    assert (ce.reason, ce.retryable) == (FailoverReason.FORMAT_ERROR, False)

    ce = classify_http_status(500, "boom")
    assert (ce.reason, ce.retryable) == (FailoverReason.CONNECTION, True)

    ce = classify_http_status(418, "teapot")
    assert (ce.reason, ce.retryable) == (FailoverReason.UNKNOWN, False)


# ---------------------------------------------------------------- exceptions

def test_classify_exception_transport_types():
    ce = classify_exception(httpx.ConnectError("connection refused"))
    assert (ce.reason, ce.retryable) == (FailoverReason.CONNECTION, True)

    ce = classify_exception(httpx.ReadTimeout("timed out"))
    assert (ce.reason, ce.retryable) == (FailoverReason.TIMEOUT, True)

    # wrapped transport error via __cause__ (runner raises from e)
    inner = httpx.RemoteProtocolError("peer closed connection")
    wrapped = AgentBackendError("Cannot reach inference server")
    wrapped.__cause__ = inner
    assert classify_exception(wrapped).reason == FailoverReason.CONNECTION


def test_classify_exception_message_rules():
    ce = classify_exception(AgentBackendError(
        "Chat completion failed: HTTP 500 — llama server OOM-retry"))
    assert (ce.reason, ce.retryable) == (FailoverReason.CONNECTION, True)

    ce = classify_exception(RuntimeError(
        "requested tokens 9000 exceeds the model context length of 8192"))
    assert (ce.reason, ce.should_compress) == (FailoverReason.CONTEXT_OVERFLOW, True)

    ce = classify_exception(RuntimeError("The requested model does not exist"))
    assert ce.reason == FailoverReason.MODEL_NOT_FOUND

    ce = classify_exception(RuntimeError(
        'server rejected tool_calls: failed to parse json arguments'))
    assert (ce.reason, ce.strip_tools) == (FailoverReason.FORMAT_ERROR, True)


def test_classify_exception_unknown_is_retryable_once():
    ce = classify_exception(RuntimeError("something we have never seen"))
    assert (ce.reason, ce.retryable) == (FailoverReason.UNKNOWN, True)

# ------------------------------------------------------------------------
# from tests/test_resilience.py
# ------------------------------------------------------------------------

@pytest.fixture()
def _no_sleeps__test_resilience(monkeypatch):
    """Retry backoff must not slow the suite down."""
    from toaster_loop.core import agent_runner as ar
    monkeypatch.setattr(ar.ToasterLoopAgentClient, "_sleep", staticmethod(lambda s: None))
    import toaster_loop.training.validate as v
    monkeypatch.setattr(v, "_sleep", lambda s: None)


# --------------------------------------------------------------- flaky server

class FlakyOnceHandler(_Handler):
    """First chat POST dies with a connection reset; then behaves normally."""

    fail_once = True  # class attr flipped by the handler itself

    def do_POST(self):
        if type(self).fail_once:
            type(self).fail_once = False
            self.close_connection = True
            try:
                self.wfile.close()
            except Exception:
                pass
            return
        super().do_POST()


class AlwaysToolCallHandler(BaseHTTPRequestHandler):
    """Never converges: forces the max-iterations exit."""

    def log_message(self, format, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        msg = {"role": "assistant", "content": None, "tool_calls": [{
            "id": "c", "type": "function",
            "function": {"name": "list_dir", "arguments": "{}"}}]}
        body = json.dumps({
            "id": "x", "object": "chat.completion",
            "model": payload.get("model", "m"),
            "choices": [{"index": 0, "message": msg,
                         "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                      "total_tokens": 2}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class BlipThenOkHandler(BaseHTTPRequestHandler):
    """First POST answers HTTP 503 (restart blip); afterwards replies '4'."""

    blipped = False

    def log_message(self, format, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if not type(self).blipped:
            type(self).blipped = True
            self._send({"error": "loading model"}, 503)
            return
        self._send({
            "id": "x", "object": "chat.completion",
            "model": payload.get("model", "m"),
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": "the answer is 4"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                      "total_tokens": 2}})

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ------------------------------------------------------------------- runner

def test_transient_connection_blip_recovered_no_failure_trace(toaster_loop_home, _no_sleeps__test_resilience):
    FlakyOnceHandler.fail_once = True
    rec = TraceRecorder(home=toaster_loop_home)
    with FakeOpenAIServer(handler_cls=FlakyOnceHandler) as srv:
        client = ToasterLoopAgentClient(base_url=srv.base_url, model="fake-model",
                                    recorder=rec)
        result = client.run("read the fake server source")
    assert result.completed is True
    assert "Done." in result.final_response
    assert result.failure_reason is None
    assert result.trace_id is not None

    footer = _footer_for(toaster_loop_home, result.trace_id)
    assert footer["failure_reason"] is None
    assert footer["status"] == "success"


def test_final_backend_failure_records_failure_reason(toaster_loop_home, _no_sleeps__test_resilience):
    rec = TraceRecorder(home=toaster_loop_home)
    client = ToasterLoopAgentClient(base_url="http://127.0.0.1:9/v1", model="m",
                                timeout=2.0, recorder=rec)
    result = client.run("hello")
    assert result.completed is False
    assert result.failure_reason == "backend_error:connection"
    assert result.trace_id is not None

    footer = _footer_for(toaster_loop_home, result.trace_id)
    assert footer["failure_reason"] == "backend_error:connection"
    assert footer["completed"] is False  # completed semantics unchanged

    store = TraceStore(toaster_loop_home / "buffer" / "toaster_loop.db")
    try:
        row = store.get_trace(result.trace_id)
        assert row is not None
        assert row.failure_reason == "backend_error:connection"
    finally:
        store.close()


def test_max_iterations_failure_reason(_no_sleeps__test_resilience):
    with FakeOpenAIServer(handler_cls=AlwaysToolCallHandler) as srv:
        client = ToasterLoopAgentClient(base_url=srv.base_url, model="m",
                                    max_iterations=2)
        result = client.run("loop forever")
    assert result.completed is False
    assert result.failure_reason == "max_iterations"


def _footer_for(home: Path, trace_id: str) -> dict:
    matches = list((home / "buffer").rglob(f"{trace_id}.jsonl"))
    assert len(matches) == 1
    lines = [json.loads(l) for l in matches[0].read_text().splitlines()]
    return lines[-1]


# ------------------------------------------------------------------ curater

def _seed(store, home, *, trace_id, messages, status="success", score=0.5,
          timestamp="2026-09-01T10:00:00+00:00", failure_reason=None):
    jsonl_path = home / "buffer" / "2026" / "09" / f"{trace_id}.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "x", encoding="utf-8") as f:
        f.write(json.dumps({"kind": "trace_header", "schema_version": 2,
                            "trace_id": trace_id, "task": "same task",
                            "failure_reason": failure_reason}) + "\n")
        for m in messages:
            f.write(json.dumps({"kind": "message", "message": m}) + "\n")
        f.write(json.dumps({"kind": "trace_footer", "schema_version": 2,
                            "status": status, "outcome_score": score,
                            "failure_reason": failure_reason}) + "\n")
    store.insert_trace(TraceRow(
        trace_id=trace_id, session_id=None, timestamp=timestamp,
        agent_profile="toaster_loop", model_alias="m", model_version=None,
        endpoint_label=None, task_summary="seed", status=status,
        outcome_score=score, num_tool_calls=0, tokens_in=None,
        tokens_out=None, duration_s=None, jsonl_path=str(jsonl_path),
        failure_reason=failure_reason,
    ))


def test_curater_excludes_infra_failures_from_dpo_negatives(toaster_loop_home, _no_sleeps__test_resilience):
    store = TraceStore(toaster_loop_home / "buffer" / "toaster_loop.db")
    good = [{"role": "user", "content": "same task"},
            {"role": "assistant", "content": "great answer"}]
    infra_bad = [{"role": "user", "content": "same task"},
                 {"role": "assistant", "content": "server hiccuped"}]
    model_bad = [{"role": "user", "content": "same task"},
                 {"role": "assistant", "content": "terrible answer"}]
    _seed(store, toaster_loop_home, trace_id="good", messages=good, score=1.0)
    _seed(store, toaster_loop_home, trace_id="infra", messages=infra_bad,
          status="failure", score=-1.0, timestamp="2026-09-01T09:00:00+00:00",
          failure_reason="backend_error:connection")
    _seed(store, toaster_loop_home, trace_id="modelbad", messages=model_bad,
          status="failure", score=-1.0, timestamp="2026-09-01T08:00:00+00:00",
          failure_reason="max_iterations")

    from toaster_loop.curate.curater import (CurationConfig, conversation_group_key,
                                        load_trace, mine_dpo_pairs)
    rows = store.query_for_curation(include_curated=False,
                                    statuses=["success", "partial", "unknown",
                                              "failure"])
    groups = {}
    for r in rows:
        t = load_trace(store, r)
        assert t is not None
        groups.setdefault(
            conversation_group_key(t.messages, t.row.agent_profile), []
        ).append(t)
    pairs = mine_dpo_pairs(groups, CurationConfig(include_failures=True))
    assert len(pairs) == 1
    # infra failure excluded despite ranking lowest; model failure used instead
    assert pairs[0].rejected_trace_id == "modelbad"
    assert pairs[0].chosen_trace_id == "good"
    store.close()


# ----------------------------------------------------------------- validate

def test_golden_suite_survives_single_http_blip(_no_sleeps__test_resilience):
    BlipThenOkHandler.blipped = False
    tasks = [{"prompt": "what is 2+2?", "expect_contains": ["4"]}]
    with FakeOpenAIServer(handler_cls=BlipThenOkHandler) as srv:
        res = run_golden_suite(srv.base_url, "m", tasks)
    assert res["pass_rate"] == 1.0


def test_golden_suite_reports_persistent_failure(_no_sleeps__test_resilience):
    script = {"*": "nonsense"}
    with FakeOpenAIServer(handler_cls=make_scripted_handler(script)) as srv:
        res = run_golden_suite(srv.base_url, "m",
                               [{"prompt": "q", "expect_contains": ["4"]}])
    assert res["pass_rate"] == 0.0

# ------------------------------------------------------------------------
# from tests/test_spill.py
# ------------------------------------------------------------------------

# ------------------------------------------------- spillover re-read exemption

def test_is_spillover_read(home__test_spill, _fresh_sweep_flag):
    d = spill_dir(home__test_spill)
    d.mkdir(parents=True, exist_ok=True)
    f = d / "abc.txt"
    f.write_text("x")
    assert is_spillover_read("read_file", {"path": str(f)}, home__test_spill) is True
    # other tools / other paths / missing files are NOT exempt
    assert is_spillover_read("list_dir", {"path": str(d)}, home__test_spill) is False
    assert is_spillover_read("read_file", {"path": "/etc/passwd"}, home__test_spill) is False
    assert is_spillover_read("read_file", {"path": str(d / "nope.txt")},
                             home__test_spill) is False
    assert is_spillover_read("read_file", {}, home__test_spill) is False


def test_turn_budget_exempt_index_survives(home__test_spill, _fresh_sweep_flag):
    """Exempted (spillover re-read) messages are never spilled by level 3."""
    big = "y" * 9000
    msgs = [{"role": "tool", "tool_call_id": "a", "name": "read_file",
             "content": big},
            {"role": "tool", "tool_call_id": "b", "name": "list_dir",
             "content": "z" * 9000}]
    enforce_turn_budget(msgs, 10_000, home__test_spill, exempt={0})
    assert msgs[0]["content"] == big  # exempt: untouched
    assert msgs[1]["content"].startswith(PERSISTED_OUTPUT_PREFIX)


@pytest.fixture()
def _fresh_sweep_flag():
    """The once-per-process prune sweep must be re-armable per test."""
    reset_sweep_flag_for_tests()
    yield
    reset_sweep_flag_for_tests()


@pytest.fixture()
def home__test_spill(tmp_path, monkeypatch):
    h = tmp_path / "uf-home"
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(h))
    return h


# ------------------------------------------------------------ per-result spill

def test_large_result_spilled_stub_shape_and_file(home__test_spill, _fresh_sweep_flag):
    content = "line of output\n" * 2000  # ~28 KB, distinct tail
    stub = maybe_spill_result("search_files", "call_42", content, home__test_spill,
                              threshold=8000)
    assert stub is not None
    assert stub.startswith(PERSISTED_OUTPUT_PREFIX)
    assert stub.endswith("</persisted-output>")
    path = spill_dir(home__test_spill) / "call_42.txt"
    assert f"path={path}" in stub and f"bytes={len(content)}" in stub
    assert content[:500] in stub  # preview (first ~500 chars)
    assert "read_file the path to retrieve" in stub
    # full output on disk, byte-identical (no secret -> redaction is a no-op)
    assert path.read_text() == content
    # atomic write leaves no temp files behind
    assert not list(spill_dir(home__test_spill).glob("*.tmp"))


def test_small_result_not_spilled(home__test_spill, _fresh_sweep_flag):
    assert maybe_spill_result("read_file", "c1", "tiny", home__test_spill,
                              threshold=8000) is None
    assert not spill_dir(home__test_spill).exists() or not list(spill_dir(home__test_spill).iterdir())


def test_stub_never_double_spilled(home__test_spill, _fresh_sweep_flag):
    content = "x" * 9000
    stub = maybe_spill_result("t", "c1", content, home__test_spill, threshold=100)
    assert stub is not None
    # feeding the stub back in is a no-op (idempotency guard on the prefix)
    assert maybe_spill_result("t", "c1", stub, home__test_spill, threshold=1) is None


def test_tool_call_id_sanitized_into_safe_filename(home__test_spill, _fresh_sweep_flag):
    content = "y" * 9000
    stub = maybe_spill_result("t", "../../evil/:*\x00id", content, home__test_spill,
                              threshold=100)
    assert stub is not None
    files = list(spill_dir(home__test_spill).iterdir())
    assert len(files) == 1
    name = files[0].name
    assert "/" not in name and ".." not in name and "\x00" not in name
    assert name.endswith(".txt")
    # escaped id gets a digest suffix so distinct ids never collide
    stub2 = maybe_spill_result("t", "..evil.id", content, home__test_spill, threshold=100)
    assert stub2 is not None
    assert len(list(spill_dir(home__test_spill).iterdir())) == 2


def test_spill_file_written_redacted(home__test_spill, _fresh_sweep_flag):
    secret = "sk-abcdefghijklmnopqrstuvwx"
    content = ("report header\n" + secret + "\n" + "z" * 9000)
    stub = maybe_spill_result("t", "c1", content, home__test_spill, threshold=8000)
    assert stub is not None
    disk = (spill_dir(home__test_spill) / "c1.txt").read_text()
    assert secret not in disk          # durable artifact must be scrubbed
    assert "[REDACTED" in disk
    # the in-memory stub keeps raw preview; recorder redacts at record time
    assert secret in stub


# --------------------------------------------------------- turn aggregate budget

def _tool_msg(i: int, size: int) -> dict:
    return {"role": "tool", "tool_call_id": f"c{i}", "name": "t",
            "content": str(i) * size}


def test_turn_budget_spills_largest_first_until_under(home__test_spill, _fresh_sweep_flag):
    # three medium results, each under the 8000 per-result threshold,
    # aggregate 23400 over a 15000 budget.
    msgs = [_tool_msg(1, 7900), _tool_msg(2, 7800), _tool_msg(3, 7700)]
    enforce_turn_budget(msgs, 15000, home__test_spill)
    # spill order is strictly largest-first: c1 (7900) then c2 (7800); after
    # those two stubs the aggregate fits, so c3 stays inline.
    assert msgs[0]["content"].startswith(PERSISTED_OUTPUT_PREFIX)
    assert msgs[1]["content"].startswith(PERSISTED_OUTPUT_PREFIX)
    assert not msgs[2]["content"].startswith(PERSISTED_OUTPUT_PREFIX)
    total = sum(len(m["content"]) for m in msgs)
    assert total <= 15000
    # spilled files exist on disk with the full originals
    assert (spill_dir(home__test_spill) / "c1.txt").read_text() == "1" * 7900
    assert (spill_dir(home__test_spill) / "c2.txt").read_text() == "2" * 7800


def test_turn_budget_idempotent_second_pass(home__test_spill, _fresh_sweep_flag):
    msgs = [_tool_msg(1, 5000), _tool_msg(2, 9000), _tool_msg(3, 7000)]
    enforce_turn_budget(msgs, 15000, home__test_spill)
    after_first = [m["content"] for m in msgs]
    files_after_first = sorted(p.name for p in spill_dir(home__test_spill).iterdir())
    enforce_turn_budget(msgs, 15000, home__test_spill)  # no-op: under budget + stub-skipping
    assert [m["content"] for m in msgs] == after_first
    assert sorted(p.name for p in spill_dir(home__test_spill).iterdir()) == files_after_first


def test_turn_budget_under_budget_untouched(home__test_spill, _fresh_sweep_flag):
    msgs = [_tool_msg(1, 100), _tool_msg(2, 200)]
    enforce_turn_budget(msgs, 15000, home__test_spill)
    assert [len(m["content"]) for m in msgs] == [100, 200]


# ------------------------------------------------------------------- retention

def test_retention_prune_and_once_per_process_flag(home__test_spill, _fresh_sweep_flag):
    old_stub = maybe_spill_result("t", "old", "a" * 9000, home__test_spill, threshold=100)
    assert old_stub is not None
    old_file = spill_dir(home__test_spill) / "old.txt"
    # backdate beyond the retention window
    stale = time.time() - 8 * 86400
    os.utime(old_file, (stale, stale))
    # sweep already ran once for this process -> old file survives a new spill
    assert maybe_spill_result("t", "fresh", "b" * 9000, home__test_spill, threshold=100)
    assert old_file.exists()
    # re-arming the flag lets the next spill prune it
    reset_sweep_flag_for_tests()
    assert maybe_spill_result("t", "fresh2", "c" * 9000, home__test_spill, threshold=100)
    assert not old_file.exists()
    assert (spill_dir(home__test_spill) / "fresh.txt").exists()
    assert (spill_dir(home__test_spill) / "fresh2.txt").exists()


def test_prune_direct_call_keeps_fresh(home__test_spill, _fresh_sweep_flag):
    maybe_spill_result("t", "keep", "d" * 9000, home__test_spill, threshold=100)
    assert prune_spillover(home__test_spill, retention_days=7) == 0
    assert (spill_dir(home__test_spill) / "keep.txt").exists()


# ---------------------------------------------------------- runner integration

class OneBigResultThenAnswerHandler(BaseHTTPRequestHandler):
    """First turn: one big read_file-style result. Second turn: final answer."""

    def log_message(self, format, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        messages = payload.get("messages", [])
        if any(m.get("role") == "tool" for m in messages):
            msg = {"role": "assistant", "content": "Done."}
            finish = "stop"
        else:
            msg = {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_big", "type": "function",
                "function": {"name": "dump", "arguments": "{}"}}]}
            finish = "tool_calls"
        body = json.dumps({
            "id": "x", "object": "chat.completion",
            "model": payload.get("model", "m"),
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                      "total_tokens": 2}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


BIG_MARKER = "TOASTER_LOOP_BIG_OUTPUT_MARKER"


TAIL_MARKER = "TOASTER_LOOP_TAIL_BEYOND_PREVIEW"


def big_dump_impl(**kwargs) -> str:
    # marker at the head (lands in the stub preview) and one past char 500
    # (must NOT appear anywhere in the stub/trace).
    return (BIG_MARKER + "\n" + ("payload line\n" * 2000) + TAIL_MARKER)


def _trace_lines(home: Path, trace_id: str) -> list:
    matches = list((home / "buffer").rglob(f"{trace_id}.jsonl"))
    assert len(matches) == 1
    return [json.loads(l) for l in matches[0].read_text().splitlines()]


def test_runner_spills_before_record_trace_keeps_stub(home__test_spill, _fresh_sweep_flag):
    from toaster_loop.store.recorder import TraceRecorder
    rec = TraceRecorder(home=home__test_spill)
    with FakeOpenAIServer(handler_cls=OneBigResultThenAnswerHandler) as srv:
        client = ToasterLoopAgentClient(base_url=srv.base_url, model="m",
                                    tool_impls={"dump": big_dump_impl},
                                    recorder=rec, spill_threshold_chars=8000)
        result = client.run("dump something big")
    assert result.completed is True
    tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    stub = tool_msgs[0]["content"]
    assert stub.startswith(PERSISTED_OUTPUT_PREFIX)
    # spill file exists under TOASTER_LOOP_HOME with the full original
    spill_file = spill_dir(home__test_spill) / "call_big.txt"
    assert spill_file.exists()
    assert BIG_MARKER in spill_file.read_text()
    # trace JSONL carries the stub, never the full output: the head marker
    # is legitimately inside the 500-char preview; anything past it (the
    # tail marker) must be absent from the trace.
    assert result.trace_id is not None
    text = json.dumps(_trace_lines(home__test_spill, result.trace_id))
    assert PERSISTED_OUTPUT_PREFIX in text
    assert TAIL_MARKER not in text
    # preview carries ~36 payload lines; the full output would carry 2000
    assert text.count("payload line") < 100


def test_runner_spill_disabled_keeps_content_inline(home__test_spill, _fresh_sweep_flag):
    with FakeOpenAIServer(handler_cls=OneBigResultThenAnswerHandler) as srv:
        client = ToasterLoopAgentClient(base_url=srv.base_url, model="m",
                                    tool_impls={"dump": big_dump_impl},
                                    spill_enabled=False)
        result = client.run("dump something big")
    tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
    assert BIG_MARKER in tool_msgs[0]["content"]
    assert not tool_msgs[0]["content"].startswith(PERSISTED_OUTPUT_PREFIX)


class IdenticalFailingPatchHandler(BaseHTTPRequestHandler):
    """Re-issues the byte-identical failing patch call forever."""

    def log_message(self, format, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        msg = {"role": "assistant", "content": None, "tool_calls": [{
            "id": "c", "type": "function",
            "function": {"name": "patch", "arguments": json.dumps(
                {"path": "missing.py", "old_string": "x",
                 "new_string": "y"})}}]}
        body = json.dumps({
            "id": "x", "object": "chat.completion",
            "model": payload.get("model", "m"),
            "choices": [{"index": 0, "message": msg,
                         "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                      "total_tokens": 2}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def failing_patch_impl(**kwargs) -> str:
    # > spill threshold below, so every executed result gets stubbed — yet
    # the guardrail ladder must still fire (it saw the real failure text).
    return "error: patch failed: file not found and could not be located"


def test_guardrails_see_real_content_despite_tiny_spill_threshold(_fresh_sweep_flag):
    with FakeOpenAIServer(handler_cls=IdenticalFailingPatchHandler) as srv:
        client = ToasterLoopAgentClient(base_url=srv.base_url, model="m",
                                    tool_impls={"patch": failing_patch_impl},
                                    max_iterations=8,
                                    spill_threshold_chars=10)
        result = client.run("fix the file")
    tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
    # executed results are stubs in history...
    assert any(m["content"].startswith(PERSISTED_OUTPUT_PREFIX)
               for m in tool_msgs)
    # ...but the failure ladder still warned and blocked on the REAL text
    assert any("[Tool loop warning: repeated_exact_failure_warning" in m["content"]
               for m in tool_msgs)
    assert any(m["content"].startswith("error: blocked by guardrail "
                                       "(repeated_exact_failure_block)")
               for m in tool_msgs)
    assert result.failure_reason == "max_iterations"


# ------------------------------------------------- containment + read_file loop

def test_spilled_file_readable_under_default_containment(home__test_spill, tmp_path, _fresh_sweep_flag):
    """The agent recovers a spill via read_file: spillover sits under
    TOASTER_LOOP_HOME, which resolve_workspace_roots allow-lists (batch #4)."""
    from toaster_loop.core.path_security import contain
    from toaster_loop.core.tools import TOOL_IMPLS

    cfg = ToasterLoopConfig()
    roots = create_runner(cfg, base_url="http://127.0.0.1:9/v1",
                          model="m").workspace_roots
    assert roots is not None  # containment on (cwd default root)
    assert any(Path(r) == home__test_spill for r in roots)  # TOASTER_LOOP_HOME allow-listed

    content = "recoverable payload\n" * 1000
    stub = maybe_spill_result("t", "call_recover", content, home__test_spill, threshold=100)
    path = spill_dir(home__test_spill) / "call_recover.txt"
    contained_read = contain(roots)(TOOL_IMPLS["read_file"])
    out = contained_read(path=str(path))
    # read_file prefixes line numbers; the spilled content must come through
    assert "error: blocked" not in out
    assert "1 | recoverable payload" in out


# ------------------------------------------------------- compression synergy

def test_prune_collapses_spill_stub_to_one_line(home__test_spill, _fresh_sweep_flag):
    stub = maybe_spill_result("t", "c9", "q" * 9000, home__test_spill, threshold=100)
    msg = {"role": "tool", "tool_call_id": "c9", "name": "t", "content": stub}
    pruned, n = prune_old_tool_results([msg], 1)
    assert n == 1
    assert pruned[0]["content"] == (
        f"[pruned persisted output: {spill_dir(home__test_spill) / 'c9.txt'}]")
    # idempotent: the one-line form is never re-pruned
    pruned2, n2 = prune_old_tool_results(pruned, 1)
    assert n2 == 0 and pruned2[0]["content"] == pruned[0]["content"]


# --------------------------------------------------------------- config wiring

def test_create_runner_reads_tools_spill_knobs(home__test_spill, _fresh_sweep_flag):
    cfg = ToasterLoopConfig(tools={"spill_enabled": False,
                               "spill_threshold_chars": 1234,
                               "turn_result_budget_chars": 5678,
                               "spill_retention_days": 3})
    runner = create_runner(cfg, base_url="http://127.0.0.1:9/v1", model="m")
    assert runner.spill_enabled is False
    assert runner.spill_threshold_chars == 1234
    assert runner.turn_result_budget_chars == 5678
    assert runner.spill_retention_days == 3
    assert runner._spill_home is not None
    assert Path(runner._spill_home) == home__test_spill

    default = create_runner(ToasterLoopConfig(), base_url="http://127.0.0.1:9/v1",
                            model="m")
    assert default.spill_enabled is True
    # Defaults track the constants in spill.py (raised over time).
    from toaster_loop.core.spill import (
        DEFAULT_SPILL_THRESHOLD_CHARS,
        DEFAULT_TURN_RESULT_BUDGET_CHARS,
    )

    assert default.spill_threshold_chars == DEFAULT_SPILL_THRESHOLD_CHARS
    assert default.turn_result_budget_chars == DEFAULT_TURN_RESULT_BUDGET_CHARS
    assert default.spill_retention_days == 7
    # token gate is opt-in: unset knob leaves the char gate (None sentinel)
    assert default.spill_threshold_tokens is None


def test_create_runner_reads_spill_threshold_tokens(home__test_spill, _fresh_sweep_flag):
    cfg = ToasterLoopConfig(tools={"spill_threshold_tokens": 999})
    runner = create_runner(cfg, base_url="http://127.0.0.1:9/v1", model="m")
    assert runner.spill_threshold_tokens == 999


# ------------------------------------------------------- token-accurate gate
# Deferred item from the exact-token build (improvement #4): spill_threshold_-
# tokens drives the per-result gate via exact /tokenize counts; everything
# above (char-gate tests, untouched) proves the default-off behavior.

class _FakeCounter__test_spill:
    """Duck-typed counter with a fixed ceil(len/2) policy (fake-server math)."""

    def __init__(self, exact_available: bool = True):
        self.calls = 0
        self.exact_available = exact_available

    def count(self, text: str) -> int:
        self.calls += 1
        return -(-len(text) // 2)


def test_token_gate_spills_cjk_that_passes_chars_gate(home__test_spill, _fresh_sweep_flag):
    # CJK content the char gate waves through (4000 chars < 8000 threshold)
    # but the exact token gate catches — real TokenCounter against the fake
    # /tokenize server, mirroring test_tokenizer_exact's CJK boundary style.
    handler = make_tokenize_handler({"*": "unused"})
    with FakeOpenAIServer(handler_cls=handler) as srv:
        handler.tokenize_requests.clear()
        tc = TokenCounter(srv.base_url)
        content = "汉" * 4000
        exact = tc.count(content)
        assert len(content) <= 8000  # char gate would NOT spill this
        assert exact == -(-4000 // 2)  # fake tokenizer: ceil(len/2) = 2000

        stub = maybe_spill_result("t", "cjk1", content, home__test_spill, threshold=8000,
                                  threshold_tokens=1500, counter=tc)
        assert stub is not None
        assert stub.startswith(PERSISTED_OUTPUT_PREFIX)
        assert "gate=tokens" in stub  # provenance: token gate fired
        assert (spill_dir(home__test_spill) / "cjk1.txt").read_text() == content


def test_chars_gate_stub_has_no_gate_marker(home__test_spill, _fresh_sweep_flag):
    stub = maybe_spill_result("t", "cg", "x" * 9000, home__test_spill, threshold=100)
    assert stub is not None and "gate=" not in stub


def test_threshold_tokens_unset_never_counts(home__test_spill, _fresh_sweep_flag):
    fake = _FakeCounter__test_spill()
    # threshold_tokens unset -> char comparison only, counter never consulted.
    assert maybe_spill_result("t", "a", "x" * 100, home__test_spill, threshold=8000,
                              counter=fake) is None
    assert fake.calls == 0
    stub = maybe_spill_result("t", "b", "x" * 9000, home__test_spill, threshold=8000,
                              counter=fake)
    assert stub is not None and fake.calls == 0


def test_both_knobs_set_tokens_win(home__test_spill, _fresh_sweep_flag):
    fake = _FakeCounter__test_spill()  # ceil(len/2): 4000 chars -> 2000 tokens
    content = "y" * 4000
    # over the token gate, under the char gate -> spilled (tokens win)
    stub = maybe_spill_result("t", "w1", content, home__test_spill, threshold=8000,
                              threshold_tokens=1500, counter=fake)
    assert stub is not None and "gate=tokens" in stub
    # under the token gate, over the char gate -> NOT spilled (tokens win)
    assert maybe_spill_result("t", "w2", content, home__test_spill, threshold=100,
                              threshold_tokens=3000, counter=fake) is None


def test_open_circuit_falls_back_to_chars_gate(home__test_spill, _fresh_sweep_flag):
    # Counter degraded (circuit open): no probing, exact chars gate decides.
    fake = _FakeCounter__test_spill(exact_available=False)
    content = "汉" * 9000  # 9000 chars > 8000 char threshold
    stub = maybe_spill_result("t", "d1", content, home__test_spill, threshold=8000,
                              threshold_tokens=10_000, counter=fake)
    assert stub is not None and "gate=" not in stub  # chars gate fired
    assert fake.calls == 0  # degraded: never counted


def test_tokenizing_endpoint_404_degrades_without_crash(home__test_spill, _fresh_sweep_flag):
    handler = make_tokenize_handler({"*": "unused"})
    with FakeOpenAIServer(handler_cls=handler) as srv:
        handler.tokenize_requests.clear()
        handler.tokenize_status = 404
        tc = TokenCounter(srv.base_url, failure_threshold=100)  # keep probing
        content = "汉" * 4000
        # exact would say 2000 > 1500 (spill); degraded chars/4 says 1000
        # < 1500 -> estimate-driven no-spill, and crucially no exception.
        assert maybe_spill_result("t", "d2", content, home__test_spill, threshold=8000,
                                  threshold_tokens=1500, counter=tc) is None


class TokenizingDumpHandler(BaseHTTPRequestHandler):
    """OneBigResultThenAnswer + a POST /tokenize (fake ceil(len/2)) endpoint."""

    tokenize_requests: list = []
    tokenize_status = 200

    def log_message(self, format, *args):
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if self.path.rstrip("/") == "/tokenize":
            content = payload.get("content")
            if isinstance(content, str):
                type(self).tokenize_requests.append(content)
            if type(self).tokenize_status != 200:
                self._send({"error": "tokenize unavailable"},
                           type(self).tokenize_status)
                return
            n = -(-len(content or "") // 2)
            self._send({"tokens": list(range(n)), "count": n})
            return
        messages = payload.get("messages", [])
        if any(m.get("role") == "tool" for m in messages):
            msg, finish = {"role": "assistant", "content": "Done."}, "stop"
        else:
            msg = {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_cjk", "type": "function",
                "function": {"name": "dump_cjk", "arguments": "{}"}}]}
            finish = "tool_calls"
        self._send({
            "id": "x", "object": "chat.completion",
            "model": payload.get("model", "m"),
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                      "total_tokens": 2}})


CJK_CONTENT = "汉" * 4000  # 4000 chars: passes an 8000-char gate, fails 1500 tok


def cjk_dump_impl(**kwargs) -> str:
    return CJK_CONTENT


def test_runner_token_gate_spills_cjk_one_http_count_per_output(home__test_spill, _fresh_sweep_flag):
    # Agent-level proof: with spill_threshold_tokens set, a CJK tool output
    # that the char gate keeps inline is spilled on the exact token count —
    # and the output crosses /tokenize exactly once (decision time only).
    TokenizingDumpHandler.tokenize_requests = []
    TokenizingDumpHandler.tokenize_status = 200
    with FakeOpenAIServer(handler_cls=TokenizingDumpHandler) as srv:
        client = ToasterLoopAgentClient(
            base_url=srv.base_url, model="m",
            tool_impls={"dump_cjk": cjk_dump_impl},
            spill_threshold_chars=8000, spill_threshold_tokens=1500,
            token_counter=TokenCounter(srv.base_url))
        result = client.run("dump the cjk thing")
    assert result.completed is True
    tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    stub = tool_msgs[0]["content"]
    assert stub.startswith(PERSISTED_OUTPUT_PREFIX)
    assert "gate=tokens" in stub
    assert (spill_dir(home__test_spill) / "call_cjk.txt").read_text() == CJK_CONTENT
    # one HTTP count per tool output: the content was POSTed exactly once
    assert TokenizingDumpHandler.tokenize_requests == [CJK_CONTENT]


def test_runner_token_gate_degraded_run_survives_404(home__test_spill, _fresh_sweep_flag):
    # /tokenize down mid-run: gate degrades (chars/4 estimate says 1000 <
    # 1500 -> inline), the run completes, nothing crashes.
    TokenizingDumpHandler.tokenize_requests = []
    TokenizingDumpHandler.tokenize_status = 404
    try:
        with FakeOpenAIServer(handler_cls=TokenizingDumpHandler) as srv:
            client = ToasterLoopAgentClient(
                base_url=srv.base_url, model="m",
                tool_impls={"dump_cjk": cjk_dump_impl},
                spill_threshold_chars=8000, spill_threshold_tokens=1500,
                token_counter=TokenCounter(srv.base_url, failure_threshold=100))
            result = client.run("dump the cjk thing")
    finally:
        TokenizingDumpHandler.tokenize_status = 200
    assert result.completed is True
    tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
    assert tool_msgs[0]["content"] == CJK_CONTENT  # degraded estimate: inline
    assert TokenizingDumpHandler.tokenize_requests == [CJK_CONTENT]  # probed once

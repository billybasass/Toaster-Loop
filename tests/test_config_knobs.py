"""Backlog items #13 (training knobs + --config-key overrides) tests.

Pure-function coverage for the torch-free pieces: lora_kwargs_from_config,
apply_config_overrides/parse_override_value, plus an end-to-end stub-worker
run asserting the overridden value lands in the resolved train.yaml and a
provenance line lands in the job log.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sable.training.pipeline import (  # noqa: E402
    _resolved_train_yaml,
    apply_config_overrides,
    log_config_overrides,
    parse_override_value,
)
from sable.training.worker import lora_kwargs_from_config  # noqa: E402


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
    from sable.cli import build_parser
    from sable.registry import ModelRegistry

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
    import sable.cli as cli_mod

    rc = cli_mod.cmd_train_start(args)
    assert rc == 0

    # resolved yaml exists with overridden values
    resolved = list((home / "train_runs").rglob("train.resolved.yaml"))
    assert resolved, "expected a resolved train.yaml under train_runs/"
    data = yaml.safe_load(resolved[0].read_text())
    assert data["lora"]["lora_r"] == 8
    assert data["training"]["num_epochs"] == 1

    # provenance line in the newest job log
    reg = ModelRegistry(home / "buffer" / "sable.db")
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
    from sable.cli import build_parser
    import sable.cli as cli_mod

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
    monkeypatch.setenv("SABLE_HOME", str(home))
    stub = tmp_path / "stub.py"
    stub.write_text("import sys; sys.exit(0)\n")
    (home / "config.yaml").write_text(yaml.safe_dump({
        "home": str(home),
        "training": {"base_model": "Base/HF-Model",
                     "worker_command": [sys.executable, str(stub)]},
    }))
    return home, stub

"""Training VRAM estimation + scheduler gate integration (#20).

Covers core/vram.py math (pure, torch-free), the local config.json loader,
the resolved-train.yaml helper, and the scheduler resource gate consuming a
predicted footprint (queue-with-reason instead of dispatch-and-pray).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, cast

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sable.core.vram import (  # noqa: E402
    ModelDims,
    base_param_count,
    estimate_from_train_yaml,
    estimate_train_vram_mib,
    lora_param_count,
    load_model_dims,
)
from sable.training.scheduler import (  # noqa: E402
    TrainingConfig,
    job_vram_estimate_mib,
    resource_gate_ok,
)

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

def _write_yaml(tmp_path, model_dir):
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
    from sable.core import vram as vram_mod

    model_dir = tmp_path / "base"
    ydir = tmp_path / "ds"
    _write_yaml(ydir, model_dir)
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
    _write_yaml(ydir, tmp_path / "unused")
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
    monkeypatch.setattr("sable.training.scheduler.gpu_free_mib",
                        lambda: 8000)
    cfg = TrainingConfig(min_free_vram_mib=1000)
    assert resource_gate_ok(cfg)["ok"]                       # static floor ok
    assert not resource_gate_ok(cfg, estimated_need_mib=9000)["ok"]
    r = resource_gate_ok(cfg, estimated_need_mib=9000)
    assert "estimate 9000 MiB" in r["reasons"][0]
    assert resource_gate_ok(cfg, estimated_need_mib=7000)["ok"]


def test_gate_no_gpu_info_skips(monkeypatch):
    monkeypatch.setattr("sable.training.scheduler.gpu_free_mib",
                        lambda: None)
    assert resource_gate_ok(TrainingConfig(min_free_vram_mib=99999),
                            estimated_need_mib=123456)["ok"]


def test_job_estimate_skips_non_gpu_kinds(monkeypatch):
    monkeypatch.setattr("sable.training.scheduler.gpu_free_mib",
                        lambda: 100_000)
    assert job_vram_estimate_mib(_as_jobrow("validate", {})) is None
    assert job_vram_estimate_mib(_as_jobrow("sft", {})) is None  # no train.yaml
    # kind on a predictable sft job flows through estimate_from_train_yaml:
    called = {}

    def fake(path, base):
        called["args"] = (path, base)
        return type("E", (), {"total_mib": 4242})()
    monkeypatch.setattr("sable.core.vram.estimate_from_train_yaml", fake)
    assert job_vram_estimate_mib(_as_jobrow(
        "sft", {"train_yaml": "/x/train.yaml", "base_model": "/m"})) == 4242
    assert called["args"] == ("/x/train.yaml", "/m")

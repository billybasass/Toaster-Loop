"""training_type: full (full fine-tune path) + 4-bit auto-tiering.

The tier policy is the "works on all consumer hardware" answer: small
models bf16 (better gradients, fits anyway), >=3B QLoRA, full-FT never
4-bit, explicit config always wins. The full-FT path reuses the adapter
dir contract with a training_type.json marker so export skips the merge.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sable.core.vram import (  # noqa: E402
    ModelDims,
    estimate_from_train_yaml,
    resolve_load_in_4bit,
)
from sable.training.worker import (  # noqa: E402
    adapter_is_full_finetune,
    muon_optimizer_factory,
)

BIG = ModelDims(hidden_size=8192, num_hidden_layers=40,
                num_attention_heads=32, num_key_value_heads=8,
                intermediate_size=28672, vocab_size=128256)   # ~40B-ish
SMALL = ModelDims(hidden_size=1024, num_hidden_layers=12,
                  num_attention_heads=16, num_key_value_heads=8,
                  intermediate_size=3072, vocab_size=151936)  # ~0.3B-ish


# ------------------------------------------------------------- tier policy

def test_tier_policy_table():
    """The policy table in one place: (explicit, dims, full, gpu) -> result."""
    cases = [
        # (explicit, dims, train_full, gpu) -> (expect_4bit, why-substring)
        (True,  SMALL, False, True,  True,  "explicit"),
        (False, BIG,   False, True,  False, "explicit"),
        (None,  BIG,   True,  True,  False, "full fine-tune"),
        (None,  BIG,   False, False, False, "no GPU"),
        (None,  BIG,   False, True,  True,  "4-bit"),
        (None,  SMALL, False, True,  False, "bf16"),
        (None,  None,  False, True,  True,  "unknown"),
    ]
    for explicit, dims, full, gpu, want, why in cases:
        load, reason = resolve_load_in_4bit(explicit=explicit, dims=dims,
                                            train_full=full, gpu_available=gpu)
        assert load == want, (explicit, dims, full, gpu, reason)
        assert why in reason, (explicit, dims, full, gpu, reason)


def test_explicit_always_wins():
    assert resolve_load_in_4bit(explicit=True, dims=SMALL,
                                train_full=False, gpu_available=True)[0]
    assert not resolve_load_in_4bit(explicit=False, dims=BIG,
                                    train_full=False, gpu_available=True)[0]


def test_full_ft_never_4bit():
    load, why = resolve_load_in_4bit(explicit=None, dims=BIG,
                                     train_full=True, gpu_available=True)
    assert not load and "full fine-tune" in why


def test_no_gpu_never_4bit():
    load, _ = resolve_load_in_4bit(explicit=None, dims=BIG,
                                   train_full=False, gpu_available=False)
    assert not load


def test_size_tier():
    big_load, big_why = resolve_load_in_4bit(
        explicit=None, dims=BIG, train_full=False, gpu_available=True)
    small_load, small_why = resolve_load_in_4bit(
        explicit=None, dims=SMALL, train_full=False, gpu_available=True)
    assert big_load and "4-bit" in big_why
    assert not small_load and "bf16" in small_why


def test_unknown_dims_conservative_default():
    load, why = resolve_load_in_4bit(explicit=None, dims=None,
                                     train_full=False, gpu_available=True)
    assert load and "unknown" in why


def test_yaml_estimate_flows_training_type(tmp_path):
    """training_type: full must reach the estimator (bf16 weights, all
    params trainable) through estimate_from_train_yaml."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({
        "hidden_size": 1024, "num_hidden_layers": 12,
        "num_attention_heads": 16, "num_key_value_heads": 8,
        "intermediate_size": 3072, "vocab_size": 4096}))
    train_yaml = tmp_path / "train.yaml"
    train_yaml.write_text(yaml.safe_dump({
        "model": str(model_dir),
        "training": {"training_type": "full", "max_seq_length": 2048},
    }))
    est_full = estimate_from_train_yaml(str(train_yaml))
    assert est_full is not None and est_full.load_mode == "bf16"
    assert any("full fine-tune" in n for n in est_full.notes)
    # Same config as lora -> smaller total (adapters only train).
    train_yaml.write_text(yaml.safe_dump({
        "model": str(model_dir),
        "training": {"training_type": "lora", "load_in_4bit": False,
                     "max_seq_length": 2048},
    }))
    est_lora = estimate_from_train_yaml(str(train_yaml))
    assert est_lora.total_mib < est_full.total_mib
    assert est_full.optimizer_mib > 4 * est_lora.optimizer_mib


# -------------------------------------------------------- full-FT artifacts

def test_full_ft_marker_roundtrip(tmp_path):
    assert not adapter_is_full_finetune(tmp_path)
    (tmp_path / "training_type.json").write_text(
        json.dumps({"training_type": "full", "model": "x/y"}))
    assert adapter_is_full_finetune(tmp_path)
    (tmp_path / "training_type.json").write_text("not json {{{")
    assert not adapter_is_full_finetune(tmp_path)


def test_dry_run_reports_training_type(tmp_path):
    """Dry-run must surface training_type (and reject bogus values)."""
    from sable.training.worker import run_sft

    cfg = {"model": "X/Y",
           "data": {"local_dataset": [], "format_type": "sharegpt"},
           "training": {"training_type": "full"}}
    # Empty dataset raises before the type check? No: type check is at
    # parse time, files validated first — craft a real file.
    ds = tmp_path / "sft.jsonl"
    ds.write_text(json.dumps({"conversations": []}) + "\n")
    cfg["data"]["local_dataset"] = [str(ds)]
    out = run_sft(cfg, None, dry_run=True)
    assert out["dry_run"] and out["training_type"] == "full"

    cfg["training"]["training_type"] = "bogus"
    try:
        run_sft(cfg, None, dry_run=True)
        raise AssertionError("bogus training_type must raise")
    except ValueError as e:
        assert "training_type" in str(e)


def test_dpo_grpo_reject_full(tmp_path):
    from sable.training.worker import run_dpo, run_grpo

    dpo_file = tmp_path / "dpo.jsonl"
    dpo_file.write_text(json.dumps(
        {"prompt": [{"role": "user", "content": "a"}],
         "chosen": "x", "rejected": "y"}) + "\n")
    cfg = {"model": "X/Y",
           "training": {"training_type": "full"},
           "_dataset_dir": str(tmp_path)}
    try:
        run_dpo(cfg, None, dry_run=True)
        raise AssertionError("DPO must refuse training_type=full")
    except ValueError as e:
        assert "SFT only" in str(e)

    # GRPO needs prompt rows in the dataset dir; reuse dpo.jsonl shape by
    # writing prompts.jsonl if the loader wants it — guard fires before
    # heavy lifting either way as long as rows exist. Use a prompts file.
    (tmp_path / "prompts.jsonl").write_text(json.dumps(
        {"prompt": "hi"}) + "\n")
    for attempt in range(2):
        try:
            run_grpo(cfg, None, dry_run=True)
            raise AssertionError("GRPO must refuse training_type=full")
        except ValueError as e:
            assert "SFT only" in str(e)
            break
        except FileNotFoundError as e:
            if attempt:
                raise AssertionError(
                    f"GRPO guard never fired (row loader kept failing: {e})")
            # Row loader runs first: give it the file shape it wants.
            (tmp_path / "grpo_prompts.jsonl").write_text(json.dumps(
                {"prompt": "hi"}) + "\n")


# ----------------------------------------------------------------- Muon

def test_muon_factory_splits_params():
    import torch

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Linear(8, 4)       # 2D weight + 1D bias
            self.norm = torch.nn.Parameter(torch.ones(4))

    m = M()
    factory = muon_optimizer_factory(lr=2e-4, weight_decay=0.01)
    # HF calls is_optimizer_factory(cls) -> not an Optimizer subclass.
    from transformers.trainer_optimizer import is_optimizer_factory
    assert is_optimizer_factory(factory)

    opt = factory(m, betas=(0.9, 0.999), eps=1e-8)
    n_muon = len(opt.optimizers[0].param_groups[0]["params"])
    n_adam = len(opt.optimizers[1].param_groups[0]["params"])
    assert n_muon == 1                      # only the Linear weight is 2D
    assert n_adam == 2                      # bias + norm go to AdamW
    # LR policy: Muon at its own default, AdamW at the run's lr.
    assert opt.optimizers[0].param_groups[0]["lr"] == 1e-3
    assert opt.optimizers[1].param_groups[0]["lr"] == 2e-4

    # step() runs across both children; state_dict round-trips.
    for p in m.parameters():
        p.grad = torch.ones_like(p)
    opt.step()
    sd = opt.state_dict()
    opt.load_state_dict(sd)


def test_muon_factory_rejects_no_2d():
    import torch

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.v = torch.nn.Parameter(torch.ones(3))

    factory = muon_optimizer_factory(lr=1e-3, weight_decay=0.0)
    try:
        factory(M())
        raise AssertionError("expected RuntimeError for 2D-less model")
    except RuntimeError as e:
        assert "no 2D trainable" in str(e)

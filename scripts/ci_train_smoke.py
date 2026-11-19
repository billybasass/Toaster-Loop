#!/usr/bin/env python3
"""CPU tiny-model training smoke (production-readiness passes 1-3).

The train -> adapter path is the actual product, yet CI never executed it:
the test suite deliberately keeps torch lazily imported (license-hygiene
test), and the 9-OS matrix syncs `--extra dev` only. This script runs the
REAL worker entry point (`toaster_loop.training.worker.main`) end-to-end on
CPU against a ~1 MB random-init model, so any transformers/trl/peft API
break in the locked dependency set fails CI in minutes instead of breaking
a user's `iterate` loop.

Deliberately NOT mocked: config loading (load_train_config), ShareGPT
parsing, chat-template rendering, response-only loss masking, peft wrapping,
SFTTrainer/DPOTrainer construction + a real train step, and adapter save all
run for real. Only the base model is tiny.

Coverage per kind (pass 3):
  sft   full end-to-end train + adapter-save assertions.
  dpo   full end-to-end train (the DPO schema/constructor churn is exactly
        where a TRL bump breaks silently, so this trains for real too).
  grpo  honest dry-run only: GRPOTrainer construction needs vLLM/gen-config
        machinery that is generation-heavy and flaky on CPU runners — we run
        the worker's --dry-run path (config + dataset + row-schema validation)
        rather than fake a training step.

Run from the repo root with the [training] extra installed:
    uv sync --frozen --extra dev --extra training
    uv run python scripts/ci_train_smoke.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Callable, List, Tuple

# Random-init ~1 MB Llama-architecture model; its q/k/v/o/gate/up/down_proj
# names match the worker's default lora target_modules exactly, so the real
# peft wrapping path is exercised. No chat template -> exercises the
# role-prefixed fallback renderer too (both are supported paths).
TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _train_yaml(root: Path) -> Path:
    dataset = root / "sft.jsonl"
    rows = [
        {"conversations": [{"from": "human", "value": f"say {w}"},
                           {"from": "gpt", "value": w}]}
        for w in ("alpha", "bravo", "charlie", "delta")
    ]
    dataset.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    train_yaml = root / "train.yaml"
    train_yaml.write_text(
        f"""
model: {TINY_MODEL}
data:
  local_dataset: [{dataset}]
  format_type: sharegpt
training:
  output_dir: {root / 'out'}
  num_epochs: 1
  learning_rate: 0.0002
  max_seq_length: 64
  batch_size: 2
  gradient_checkpointing: false
lora:
  lora_r: 4
  lora_alpha: 8
""".lstrip()
    )
    return train_yaml


def _run_worker(kind: str, argv_extra: List[str], log_path: Path) -> Tuple[int, str]:
    from toaster_loop.training.worker import main as worker_main

    rc = worker_main([
        "--job", f"ci-smoke-{kind}", "--kind", kind, *argv_extra,
        "--log-path", str(log_path),
    ])
    return rc, log_path.read_text(errors="ignore")[-2000:]


def smoke_sft(root: Path) -> int:
    train_yaml = _train_yaml(root)
    log = root / "sft.log"
    rc, tail = _run_worker("sft", ["--config", str(train_yaml)], log)
    if rc != 0:
        print(f"[sft] worker exited {rc}; job log tail:\n{tail}", file=sys.stderr)
        return 1
    adapter = root / "out" / "adapter"
    missing = [f for f in ["adapter_config.json"] if not (adapter / f).is_file()]
    saved = sorted(p.name for p in adapter.iterdir()) if adapter.is_dir() else []
    if missing:
        print(f"[sft] adapter dir incomplete: {missing}; found {saved}",
              file=sys.stderr)
        return 1
    print(f"[sft] OK — adapter artifacts: {saved}")
    return 0


def smoke_dpo(root: Path) -> int:
    # TRL preference schema: {prompt: [...], chosen: str, rejected: str}.
    dataset_dir = root / "dpo"
    dataset_dir.mkdir(exist_ok=True)
    rows = [
        {"prompt": [{"role": "user", "content": f"say {w}"}],
         "chosen": w, "rejected": f"not {w}"}
        for w in ("alpha", "bravo", "charlie")
    ]
    (dataset_dir / "dpo.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")

    train_yaml = root / "train_dpo.yaml"
    train_yaml.write_text(
        f"""
model: {TINY_MODEL}
training:
  output_dir: {root / 'out_dpo'}
  num_epochs: 1
  learning_rate: 0.0002
  max_seq_length: 64
  batch_size: 2
  gradient_checkpointing: false
lora:
  lora_r: 4
  lora_alpha: 8
""".lstrip()
    )
    log = root / "dpo.log"
    rc, tail = _run_worker("dpo", ["--config", str(train_yaml),
                                   "--dataset-dir", str(dataset_dir)], log)
    if rc != 0:
        print(f"[dpo] worker exited {rc}; job log tail:\n{tail}", file=sys.stderr)
        return 1
    adapter = root / "out_dpo" / "adapter"
    missing = [f for f in ["adapter_config.json"] if not (adapter / f).is_file()]
    saved = sorted(p.name for p in adapter.iterdir()) if adapter.is_dir() else []
    if missing:
        print(f"[dpo] adapter dir incomplete: {missing}; found {saved}",
              file=sys.stderr)
        return 1
    # The train step must actually have run (not just constructed a trainer).
    if '"event": "train_dpo"' not in tail and "train_dpo" not in tail:
        print("[dpo] log shows no train_dpo phase event", file=sys.stderr)
        return 1
    print(f"[dpo] OK — adapter artifacts: {saved}")
    return 0


def smoke_grpo_dry(root: Path) -> int:
    # Honest CPU scope: GRPO generation is heavy/flaky on CI runners, so we
    # exercise the worker's dry-run path (dataset derivation + row schema +
    # config resolution) instead of pretending a train step ran.
    dataset_dir = root / "grpo"
    dataset_dir.mkdir(exist_ok=True)
    rows = [
        {"prompt": [{"role": "user", "content": f"say {w}"}], "solution": w}
        for w in ("alpha", "bravo")
    ]
    (dataset_dir / "grpo.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")

    train_yaml = root / "train_grpo.yaml"
    train_yaml.write_text(
        f"""
model: {TINY_MODEL}
training:
  output_dir: {root / 'out_grpo'}
  num_epochs: 1
lora:
  lora_r: 4
  lora_alpha: 8
""".lstrip()
    )
    log = root / "grpo.log"
    rc, tail = _run_worker("grpo", ["--config", str(train_yaml),
                                    "--dataset-dir", str(dataset_dir),
                                    "--dry-run"], log)
    if rc != 0:
        print(f"[grpo dry] worker exited {rc}; job log tail:\n{tail}",
              file=sys.stderr)
        return 1
    if "dry_run_ok" not in tail:
        print("[grpo dry] no dry_run_ok event in log", file=sys.stderr)
        return 1
    print("[grpo] OK (dry-run scope — construction/generation skipped on CPU)")
    return 0


def main() -> int:
    import torch  # noqa: F401  (fail fast with a clear message if missing)

    from toaster_loop.core.mlstack import check_function as mlstack_check

    ok, lines = mlstack_check()
    print("ml-stack window:", "OK" if ok else "VIOLATED")
    for ln in lines:
        print(" ", ln)
    if not ok:
        return 1

    checks: List[Tuple[str, Callable[[Path], int]]] = [
        ("sft", smoke_sft),
        ("dpo", smoke_dpo),
        ("grpo", smoke_grpo_dry),
    ]
    with tempfile.TemporaryDirectory(prefix="tl-ci-smoke-") as td:
        root = Path(td)
        failed = [name for name, fn in checks if fn(root) != 0]
    if failed:
        print(f"train smoke FAILED: {', '.join(failed)}", file=sys.stderr)
        return 1
    print("train smoke OK — sft + dpo trained for real, grpo dry-run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

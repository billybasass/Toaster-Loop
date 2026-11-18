#!/usr/bin/env python3
"""CPU tiny-model training smoke (production-readiness pass, finding #3).

The train -> adapter path is the actual product, yet CI never executed it:
the test suite deliberately keeps torch lazily imported (license-hygiene
test), and the 9-OS matrix syncs `--extra dev` only. This script runs the
REAL worker entry point (`toaster_loop.training.worker.main`) end-to-end on
CPU against a ~1 MB random-init model, so any transformers/trl/peft API
break in the locked dependency set fails CI in minutes instead of breaking
a user's `iterate` loop.

Deliberately NOT mocked: config loading (load_train_config), ShareGPT
parsing, chat-template rendering, response-only loss masking, peft wrapping,
SFTTrainer, and adapter save all run for real. Only the base model is tiny.

Run from the repo root with the [training] extra installed:
    uv sync --frozen --extra dev --extra training
    uv run python scripts/ci_train_smoke.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# Random-init ~1 MB Llama-architecture model; its q/k/v/o/gate/up/down_proj
# names match the worker's default lora target_modules exactly, so the real
# peft wrapping path is exercised. No chat template -> exercises the
# role-prefixed fallback renderer too (both are supported paths).
TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def main() -> int:
    import torch  # noqa: F401  (fail fast with a clear message if missing)

    with tempfile.TemporaryDirectory(prefix="tl-ci-smoke-") as td:
        root = Path(td)
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

        from toaster_loop.training.worker import main as worker_main

        rc = worker_main([
            "--job", "ci-smoke", "--kind", "sft",
            "--config", str(train_yaml),
            "--log-path", str(root / "job.log"),
        ])
        log_tail = (root / "job.log").read_text(errors="ignore")[-2000:]
        if rc != 0:
            print(f"worker exited {rc}; job log tail:\n{log_tail}",
                  file=sys.stderr)
            return 1
        adapter = root / "out" / "adapter"
        expected = ["adapter_config.json"]
        missing = [f for f in expected if not (adapter / f).is_file()]
        saved = sorted(p.name for p in adapter.iterdir()) if adapter.is_dir() else []
        if missing:
            print(f"adapter dir incomplete: {missing}; found {saved}",
                  file=sys.stderr)
            return 1
        print(f"train smoke OK — adapter artifacts: {saved}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

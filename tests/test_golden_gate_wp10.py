"""WP10 acceptance: golden-task gate through the normal promotion path, CPU.

Closes the last WP10 acceptance gap ("streamed backend trains a checkpoint
that passes the golden task validation gate through the normal registry
promotion path") on a CPU-only box, via the real code paths end-to-end:

    streamed worker -> run_export (gguf-lora) -> ServingSupervisor spawns the
    REAL llama-server for a locally-built tiny f32 llama base GGUF with the
    adapter (--lora through cfg.extra_llama_args) -> training.validate.
    validate_model golden gate (positive + negative control) -> ModelRegistry
    register/set_validation/promote.

The golden is a MINIMAL DETERMINISTIC tripwire tied to the measured adapter
effect on greedy output (see docs/DECISIONS.md D-15 and the ROADMAP WP10
entry); it exercises the gate, it does not claim quality.

Skips cleanly when heavy deps or an EXISTING llama-server binary are absent —
the test never triggers a llama.cpp clone/build (hermetic rule). Measured
runtime on this box: ~6 s per run; verified deterministic across repeats.
Reproduction script with full evidence JSON: experiments/wp10_golden_gate/
run_gate.py (`python -m experiments.wp10_golden_gate.run_gate`).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _deps():
    for mod in ("torch", "transformers", "tokenizers", "safetensors"):
        pytest.importorskip(mod)
    from experiments.wp10_golden_gate.run_gate import find_binary

    binary = find_binary()
    if not binary:
        pytest.skip("no existing llama-server binary (never auto-build)")
    return binary


def test_golden_gate_promotion_cpu(tmp_path):
    """Full CPU proof; asserts the evidence JSON end to end."""
    _deps()
    from experiments.wp10_golden_gate.run_gate import run

    ev = run(work=tmp_path / "work", keep=True)

    assert ev["train"]["backend"] == "streamed"
    assert ev["export"]["gguf_path"].endswith(".gguf")
    assert ev["probe"]["changed"] is True          # adapter measurably active
    assert ev["gate_with_adapter"]["ok"] is True   # REAL gate passed
    assert ev["gate_with_adapter"]["pass_rate"] == 1.0
    assert ev["gate_without_adapter"]["ok"] is False  # golden not vacuous
    assert ev["promotion"]["active"] is True       # normal registry promotion

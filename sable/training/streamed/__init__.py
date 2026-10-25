"""Streamed-LoRA training spine — WP10 (wired into the worker, step 2+3).

Selected via ``training.backend: streamed`` (default "peft" keeps the
established path byte-identical); see :mod:`sable.training.streamed.runner`
for the worker seam and docs/SPIKE-WP10.md for the mechanism proofs.

Four mechanisms for LoRA fine-tuning of HF checkpoints that do not fit in
VRAM, re-implemented INLINE per D-12. Inspiration was READ from the local
airllm reference tree (Apache-2.0): ``airllm/airllm/air_llm/airllm/
airllm_lora.py`` (_StreamedModule, prefetch/evict layer streaming),
``.../lora_linear.py`` (PEFT-free LoRALinear) and ``.../chunked_ce.py``
(chunked linear cross-entropy). Nothing here imports airllm (or any other
reference project) — see tests/test_hygiene.py.

Modules:
  streaming   (a) meta-device shell + per-layer weight streaming with a
              bounded read-ahead window and optional threaded prefetch.
  lora_linear (b) PEFT-free LoRALinear keeping the frozen base ``weight``
              Parameter in place (checkpoint-shaped state_dict).
  recompute   (c) CPU-held autograd graph: no_grad forward + per-layer
              backward recompute; dropout MUST be 0 (enforced).
  chunked_ce  (d) chunked linear cross-entropy — never materialises the
              full [N, V] logits.

Import boundary: this package is import-safe WITHOUT torch installed. Every
torch / safetensors touch happens lazily inside functions (the training-extra
seam approved for WP10 — the same policy worker.py lives under; see
tests/test_hygiene.py). Classes that must subclass torch types are
built by lazy factories and exposed via PEP 562 module ``__getattr__``.

Modules added in steps 2–3:
  masking     per-turn -100 loss masking (stricter than airllm's loader).
  runner      the worker seam: split -> meta shell -> inject_lora ->
              StreamedRecomputeTrainer.step -> chunked CE -> artifacts.
  gguf_lora   frozen lora-state-dict -> GGUF-LoRA writer (stdlib-only) +
              the pinned one-reference-arch tensor-name adapter.
"""

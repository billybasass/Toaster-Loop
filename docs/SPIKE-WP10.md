# SPIKE — WP10 step 1: streamed-LoRA backend (findings note)

> **Status update (steps 2–3 done):** the package IS now wired into
> `worker.py` behind `training.backend: streamed`; see docs/ROADMAP.md
> "WP10 steps 2–3" for what was proven here vs hardware-unverified.
> The text below is the step-1 record.

Timeboxed mechanism spike in [`sable/training/streamed/`](../sable/training/streamed/)
with tests in [`tests/test_streamed_lora_wp10.py`](../tests/test_streamed_lora_wp10.py).
NOT wired into `worker.py` (Step 2, gated separately). All four mechanisms
re-implemented inline; airllm (Apache-2.0) read as reference only — cited in
module docstrings, never imported (D-12; the import allowlist in
`test_license_hygiene.py` still passes and now also bans `airllm`-family names
by construction).

## Machine reality (honesty per rule 5)

This spike ran on a machine that DOES have torch 2.14.1 (CPU-only, no CUDA)
and safetensors 0.8.0 — contrary to the work-package assumption of "likely no
torch". Therefore every numeric proof below was **executed here on real
torch tensors (CPU)**; only CUDA-specific paths (empty_cache on evict, device
copies) are code-reviewed-only. The package is additionally proven
import-safe with torch blocked via a meta_path blocker (test asserts the AST
guarantee: no module-level heavy imports).

## Proof matrix

| mechanism | status here | evidence |
|---|---|---|
| (a) layer streaming + bounded window (sync + threaded prefetch, real safetensors shards written in-test, meta-shell fill via `torch.utils.swap_tensors`) | **proven-executed-here** (CPU; CUDA eviction path code-reviewed-only) | 8 streamer tests + `test_layer_streaming_from_real_safetensors` |
| (b) PEFT-free LoRALinear (frozen base, checkpoint-shaped state_dict, y = base + s·B(Ax)) | **proven-executed-here** | `test_lora_linear_numerics_and_frozen_base`, `test_lora_state_dict_layout` |
| (c) CPU-held graph + backward recompute (grads == dense autograd reference; loss decreases over 15 streamed steps; dropout p>0 rejected at construction) | **proven-executed-here** (fp64, deterministic seed) | `test_recompute_matches_dense_reference`, `test_streamed_training_loss_decreases`, `test_dropout_guard_rejects_p_gt_zero` |
| (d) chunked linear CE vs `F.cross_entropy` on [2·8,1024]×[1024,64], fp64, ignore_index path, d_hidden gradient equality; pure-python online-LSE reference matches stdlib softmax | **proven-executed-here** (CPU) | `test_chunked_ce_matches_cross_entropy`, 3 pure-numerics tests |

Full suite at time of writing: **1744 passed**, `ruff check sable tests`
clean, `import sable.training.streamed` succeeds with torch blocked.

## Key design decisions

* **Prefetch depth default 1** (window = 2 layers). Proven invariant
  `resident ≤ 1 + prefetch_depth` in sync and threaded modes; over-fetch is
  dropped, not queued (deterministic RAM). One worker thread only — a single
  serialized reader keeps peak RAM equal to the window; deeper parallelism
  buys little when compute dominates at layer granularity. Failed read-aheads
  defer their error to `acquire()` so a bad shard never explodes mid-loop.
* **Eviction policy**: LRU-of-one by construction — `release(i)` right after
  layer i's backward recompute; forward also releases immediately (backward
  re-acquires/reloads). Streamer is a cache: acquire-after-release reloads.
  Abandoned in-flight loads are discarded exactly once when they land.
* **CPU optimizer-state layout**: grads accumulate into CPU parameter
  `.grad`s; `StreamedRecomputeTrainer.collect_grads_cpu()` is the explicit
  hand-off snapshot keyed by `id(param)`; optimizer stays a plain torch SGD
  over CPU tensors between steps. Nothing device-resident survives a step.
* **LoRA init**: kaiming A / zeros B ⇒ adapter starts as exact no-op AND
  `dL/dA == 0` on the first step (only B moves; A from step 2). Tests assert
  this precisely — it is standard LoRA behavior, not a bug.
* **Meta-shell fill**: assigning `.data` onto a meta Parameter raises
  ("incompatible tensor type"); the correct primitive is
  `torch.utils.swap_tensors(param, Parameter(cpu_tensor))`, which preserves
  Parameter identity (state_dict keys keep matching shard keys — required by
  Step 3's GGUF-LoRA mapper).
* **Chunked CE keeps lm_head frozen** (`weight` stashed, not an apply()
  input): d_weight is intentionally not produced. Matches the streamed-LoRA
  contract (head untrained); documented in code and test.
* **Dropout p=0 enforced twice**: LoRALinear ctor raises on `dropout>0`;
  `StreamedRecomputeTrainer` scans modules (`assert_no_stochastic_dropout`,
  duck-typed so it runs torch-free). Recompute replays the forward — any
  stochastic op silently corrupts gradients. Loud, per spec.

## Risks for Step 2 (worker.py integration)

1. **Seam shape (recommended)**: a new job kind or `training.streamed: true`
   opt-in flag handled INSIDE worker.py's existing sft path — worker builds
   the streamed stack (its lazy-import conventions already fit) and calls
   `inject_lora` + `StreamedRecomputeTrainer.step` in its own loop instead of
   SFTTrainer. Keep `chunked_linear_cross_entropy(hidden, lm_head.weight,
   labels)` as the loss head. The spike's current heavy-import exemption for
   `training/streamed/*` (torch+safetensors, function-level only) can then be
   folded into whatever file-scope policy Step 2 chooses; keep it narrow.
2. **Architecture support**: the spike streams generic Linear stacks. Real
   transformers decoder layers (attention, RoPE, RMSNorm, masks, MoE routing)
   must run in-tree per version — airllm hard-codes per-arch forward glue and
   still breaks across transformers releases. Expect a pinned-transformers
   matrix or an arch adapter interface; this is the biggest Step-2 cost/risk.
3. **Disk cost of layer splitting**: re-sharding by layer costs ~1× model
   size while both trees exist; `delete_original` reclaim only after
   verification passes. Adapter save is tiny (lora_state_dict), but merged
   export for GGUF still needs a full merge pass (existing export path).
4. **bf16 reality**: proofs ran fp32/fp64 on CPU. bf16 recompute equality and
   CUDA empty_cache cadence (per-layer eviction churn can thrash the caching
   allocator — may need `empty_cache` every N evictions) are untested here;
   Step 2 needs a GPU box in the loop before any speed claim (`bench --noise-floor` rules apply).
5. **Throughput honesty**: no perf numbers were taken (CPU-only box, toy
   sizes). Nothing here claims speed; the spike proves correctness-of-mechanism.

## GO / NO-GO for Steps 2–3

**GO, conditional.** All four mechanisms are proven executed-here on real
tensors; the seam is clean and nothing existing changed behaviorally (opt-in
rule held: new package + two documented test-allowlist exemptions only).
Conditions: (1) Step 2 starts with the arch-adapter design for ONE pinned
transformers version + ONE reference architecture before generalizing;
(2) any speed/quality claim deferred to a measured GPU run per cardinal
rule 2; (3) Step 3's GGUF-LoRA mapper proceeds only against the frozen
`lora_state_dict` name→tensor contract asserted by `test_lora_state_dict_layout`.

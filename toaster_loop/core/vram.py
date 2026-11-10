"""Training-time VRAM footprint estimation (inspiration pass 3, item #20).

The scheduler's resource gate used a static ``min_free_vram_mib`` floor and the
worker discovered OOM by dying. This module predicts what a resolved
``train.yaml`` will actually need so the gate can queue-with-reason instead of
dispatch-and-pray.

Concept: the *formula shape* (weights + LoRA params + optimizer bytes/param +
gradient fraction + activation term scaled by attention implementation) is a
well-known decomposition (Eymard et al. "Reducing Activation Recomputation"
and general VRAM-estimation practice); every constant below is our own and
documented, deliberately conservative (over-estimating delays a job;
under-estimating kills one).

Model (LoRA fine-tune — the only mode toaster_loop trains):

    total = weights + lora_weights + gradients + optimizer
            + activations + context_overhead

  * ``weights``      base model params x bytes/param by load mode
                     (bf16=2, nf4-with-double-quant ~= 0.56, q8 ~= 1.06).
  * ``lora_weights`` r*(in+out) per targeted linear module, from the base
                     config dims + lora.r/target_modules.
  * ``gradients``    trainable params x grad bytes (bf16 = 2).
  * ``optimizer``    trainable params x table lookup (fp32 m+v = 8 B;
                     8-bit optimizers = 4 B; adafactor ~= 2 B).
  * ``activations``  per-layer saved-activation formula, heavily reduced under
                     gradient checkpointing; the quadratic score-matrix term
                     is a transient peak (one layer's worth under GC, all
                     layers under eager) and is floored — not paid in full —
                     for any fused attention impl (flash *or* sdpa).
  * ``context``      CUDA/HIP context + fragmentation slack (fixed MiB plus a
                     fraction of weights).

Pure stdlib; reads only a local model directory's ``config.json``. HF ids
without a cached dir return ``None`` dims -> callers skip the gate rather than
guess blind.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

MIB = 1024 * 1024

# Bytes per parameter for the frozen base weights, by load mode. nf4 with
# double quantization lands ~0.5-0.6; we take the conservative end.
WEIGHT_BYTES_PER_PARAM: Dict[str, float] = {
    "float32": 4.0,
    "bf16": 2.0,
    "fp16": 2.0,
    "nf4": 0.56,
    "int8": 1.06,
}

# Optimizer state bytes per *trainable* parameter (LoRA adapters only).
# Keyed by the HF `optim` name; unknown names fall back to DEFAULT.
OPT_BYTES_PER_PARAM: Dict[str, float] = {
    "adamw": 8.0, "adamw_torch": 8.0, "paged_adamw_32bit": 8.0,
    "adamw_apex_fused": 8.0, "adamw_bnb_8bit": 4.0, "adamw_8bit": 4.0,
    "paged_adamw_8bit": 4.0, "adafactor": 2.0, "sgd": 4.0,
    "sgd_nesterov": 4.0, "adagrad": 8.0,
}
DEFAULT_OPT_BYTES = 8.0

# CUDA/HIP context + allocator fragmentation: fixed floor plus a fraction of
# the weight footprint (fragmentation grows with allocation size).
CONTEXT_OVERHEAD_MIB = 600.0
FRAGMENTATION_FRACTION = 0.10

# Activation constants (elements per layer, Eymard-style decomposition):
#   elems/layer = b*s*h * ACT_LINEAR_TERMS   (+ separate transient score term)
# Under gradient checkpointing only block inputs (+ norm copies) are saved:
ACT_LINEAR_TERMS = 34.0       # no-GC: all per-block activations kept
ACT_GC_TERMS = 2.0            # GC: block input + norm copy, per layer
ACT_ATTN_QUADRATIC = 5.0
ACT_FLASH_FACTOR = 0.10       # fused impls (flash AND sdpa) floor the score term
ACT_MIN_BYTES_PER_ELEM = 2    # bf16 activations

# Deep-dive finding #13 (long-sequence underprediction): two terms the
# Eymard decomposition above omits, both dominant at long context —
#   * CE_LOGITS_BYTES: the loss step materializes the FULL logits
#     [b, s, vocab] before cross_entropy. HF can upcast to fp32, but the
#     measured peak on this stack (bf16 logits; chunked CE in the streamed
#     backend never materializes [N, V] at all) prices at bf16 — 2 B/elem.
#     The old fp32 4.0 alone overcharged a 250k-vocab model at 16k x 2 by
#     ~15 GiB (jobs 38/39 estimated ~103 GiB vs ~35 GiB observed). Gradient
#     checkpointing does NOT shrink this term: the head runs once.
#   * ACT_LORA_RESERVE_TERMS: per targeted module, the adapter's [b, s, r]
#     A-output is saved for backward (the training-reserve heuristic from the
#     deep-dive survey: seq x batch x rank). Linear in r: high-rank long-
#     context runs used to predict the same activations as r=8 ones. Under GC
#     only one block's worth is live.
CE_LOGITS_BYTES = 2.0         # bf16 logits during CE (measured peak)
ACT_LORA_RESERVE_TERMS = 1.0  # adapter A-output elements per (b*s*r, module)

# Realistic sequence occupancy: every activation term above is priced at
# FULL max_seq_length, but with packing disabled real batches are padded only
# to the longest sample in the batch — typically well under the cap (the
# 16k-context Qwen runs measured ~35 GiB against an 82 GiB activations
# component). Scale all per-token activation bytes by this factor so the
# estimate tracks observed peaks instead of a worst-case full-occupancy pad.
ACT_SEQ_FILL_FACTOR = 0.5


@dataclass
class ModelDims:
    """The config.json dimensions the estimator needs."""

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int = 0          # 0 => MHA (= num_attention_heads)
    intermediate_size: int = 0            # 0 => heuristic 8/3 * hidden
    vocab_size: int = 0                   # 0 => heuristic hidden*8
    head_dim: int = 0                     # 0 => hidden // heads
    tie_word_embeddings: bool = True

    @staticmethod
    def from_config_json(data: Dict[str, Any]) -> "ModelDims":
        cfg = dict(data)
        # Multimodal wrappers often nest the text dims under text_config.
        if isinstance(cfg.get("text_config"), dict):
            merged = dict(cfg["text_config"])
            for k, v in cfg.items():
                merged.setdefault(k, v)
            cfg = merged
        heads = int(cfg.get("num_attention_heads") or 32)
        hidden = int(cfg.get("hidden_size") or 4096)
        return ModelDims(
            hidden_size=hidden,
            num_hidden_layers=int(cfg.get("num_hidden_layers") or 32),
            num_attention_heads=heads,
            num_key_value_heads=int(cfg.get("num_key_value_heads") or heads),
            intermediate_size=int(cfg.get("intermediate_size") or 0)
            or int(round(hidden * 8 / 3)),
            vocab_size=int(cfg.get("vocab_size") or hidden * 8),
            head_dim=int(cfg.get("head_dim") or max(1, hidden // heads)),
            tie_word_embeddings=bool(cfg.get("tie_word_embeddings", True)),
        )


def load_model_dims(base_model: Optional[str]) -> Optional[ModelDims]:
    """Read dims from a *local* HF-format model dir's config.json.

    Returns None for HF ids, GGUF paths, or missing dirs — the caller skips
    the estimate (offline-first: no network resolution here).
    """
    if not base_model:
        return None
    p = Path(str(base_model)).expanduser()
    cfg_json = p / "config.json"
    if not (p.is_dir() and cfg_json.is_file()):
        return None
    try:
        return ModelDims.from_config_json(json.loads(cfg_json.read_text()))
    except (OSError, ValueError):
        return None


# ------------------------------------------------------------------ shapes

def _linear_shapes(d: ModelDims) -> Dict[str, Tuple[int, int]]:
    """(in_features, out_features) of one instance of each standard module."""
    h, kv = d.hidden_size, d.num_key_value_heads * d.head_dim
    inter = d.intermediate_size
    shapes = {
        "q_proj": (h, h), "k_proj": (h, kv), "v_proj": (h, kv), "o_proj": (h, h),
        "gate_proj": (h, inter), "up_proj": (h, inter), "down_proj": (inter, h),
        # common alternates on other architectures
        "Wqkv": (h, h + 2 * kv), "out_proj": (h, h),
        "w1": (h, inter), "w2": (inter, h), "w3": (h, inter),
        "up_proj_NN": (h, inter),
    }
    if not d.tie_word_embeddings:
        shapes["lm_head"] = (h, d.vocab_size)
    shapes["embed_tokens"] = (d.vocab_size, h)
    return shapes


def base_param_count(d: ModelDims) -> int:
    """Rough total parameter count from config dims (no bias terms)."""
    kv = d.num_key_value_heads * d.head_dim
    per_layer = (d.hidden_size * d.hidden_size      # q_proj
                 + 2 * d.hidden_size * kv           # k+v_proj
                 + d.hidden_size * d.hidden_size    # o_proj
                 + 3 * d.hidden_size * d.intermediate_size)
    total = per_layer * d.num_hidden_layers
    total += d.vocab_size * d.hidden_size           # input embeddings
    if not d.tie_word_embeddings:
        total += d.hidden_size * d.vocab_size       # lm_head
    return total


def lora_param_count(d: ModelDims, r: int, target_modules: List[str]) -> int:
    """Trainable adapter params: sum over layers of r*(in+out) per target."""
    shapes = _linear_shapes(d)
    per_layer = 0
    for name in target_modules:
        s = shapes.get(name)
        if s is None:
            # Unknown module name: assume a square hidden x hidden linear.
            s = (d.hidden_size, d.hidden_size)
        per_layer += r * (s[0] + s[1])
    return max(0, per_layer) * d.num_hidden_layers


# ------------------------------------------------------------- estimation

@dataclass
class VramEstimate:
    """Predicted training footprint, all MiB, with the component breakdown."""

    total_mib: int
    weights_mib: int
    lora_mib: int
    gradients_mib: int
    optimizer_mib: int
    activations_mib: int
    overhead_mib: int
    load_mode: str
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total_mib": self.total_mib,
            "weights_mib": self.weights_mib,
            "lora_mib": self.lora_mib,
            "gradients_mib": self.gradients_mib,
            "optimizer_mib": self.optimizer_mib,
            "activations_mib": self.activations_mib,
            "overhead_mib": self.overhead_mib,
            "load_mode": self.load_mode,
            "notes": self.notes,
        }


def estimate_train_vram_mib(
    dims: ModelDims,
    *,
    lora_r: int = 64,
    target_modules: Optional[List[str]] = None,
    load_in_4bit: bool = True,
    optim: str = "",
    batch_size: int = 2,
    max_seq_length: int = 16384,
    gradient_checkpointing: bool = True,
    attn_implementation: str = "sdpa",
    train_full: bool = False,
) -> VramEstimate:
    """Predict the peak VRAM footprint of one training run (conservative)."""
    notes: List[str] = []
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]
    optim_key = str(optim or "").strip().lower()

    # --- load mode -------------------------------------------------------
    if train_full:
        load_mode = "bf16"  # full fine-tunes are not run 4-bit here
    elif load_in_4bit:
        load_mode = "nf4"
    else:
        load_mode = "bf16"
    w_bytes = WEIGHT_BYTES_PER_PARAM.get(load_mode, 2.0)

    n_base = base_param_count(dims)
    weights_mib = int(n_base * w_bytes / MIB)

    # --- trainable side ----------------------------------------------------
    if train_full:
        trainable = n_base
        notes.append("full fine-tune assumed (train_full=true)")
    else:
        trainable = lora_param_count(dims, int(lora_r), list(target_modules))
    lora_mib = int(trainable * 2 / MIB) if not train_full else 0
    gradients_mib = int(trainable * 2 / MIB)
    opt_bytes = OPT_BYTES_PER_PARAM.get(optim_key, DEFAULT_OPT_BYTES)
    if optim_key and optim_key not in OPT_BYTES_PER_PARAM:
        notes.append(f"unknown optimizer {optim_key!r}: assumed "
                     f"{DEFAULT_OPT_BYTES:.0f} B/param state")
    optimizer_mib = int(trainable * opt_bytes / MIB)

    # --- activations --------------------------------------------------------
    # Saved-per-layer term (inputs to every block), reduced by GC. Kept
    # separate from the score-matrix term because they scale differently:
    # saved activations are linear in s and exist for ALL layers, while the
    # b*heads*s*s attention scores are a TRANSIENT peak — autograd saves them
    # per layer only without checkpointing; under GC (or fused kernels) at
    # most one layer's scores exist at a time. Multiplying the quadratic term
    # into the all-layers sum inflated a 27B/16k-context estimate to ~1 TiB
    # and starved the scheduler gate forever.
    b = max(1, int(batch_size))
    s = max(1, int(max_seq_length))
    h = dims.hidden_size
    if gradient_checkpointing:
        act_elems = b * s * h * ACT_GC_TERMS * dims.num_hidden_layers
        score_layers = 1  # recomputed per layer: one score matrix at a time
    else:
        act_elems = b * s * h * ACT_LINEAR_TERMS * dims.num_hidden_layers
        score_layers = dims.num_hidden_layers
    quadratic = ACT_ATTN_QUADRATIC * dims.num_attention_heads * s / h
    # Only "eager" materializes the full score matrix; flash AND sdpa use
    # fused kernels that never allocate it (floor kept as slack).
    if str(attn_implementation).lower() != "eager":
        quadratic *= ACT_FLASH_FACTOR
    act_elems += b * s * h * quadratic * score_layers

    # LoRA adapter activation reserve (finding #13): [b, s, r] per targeted
    # module. GC bounds it to one block's modules; without GC every layer's
    # adapter outputs are saved for backward.
    if not train_full:
        n_targets = max(1, len(target_modules))
        reserve_layers = 1 if gradient_checkpointing else dims.num_hidden_layers
        act_elems += b * s * int(lora_r) * n_targets * reserve_layers \
            * ACT_LORA_RESERVE_TERMS

    # Cross-entropy logits peak (finding #13): [b, s, vocab]. This is the
    # term that made long-context runs OOM despite a "safe" estimate —
    # gradient checkpointing cannot shrink it (the head computes once).
    ce_elems = b * s * dims.vocab_size
    # Realistic occupancy: price all per-token activation bytes at typical
    # batch fill, not full max_seq_length padding (see ACT_SEQ_FILL_FACTOR).
    activations_mib = int((act_elems * ACT_MIN_BYTES_PER_ELEM
                           + ce_elems * CE_LOGITS_BYTES)
                          * ACT_SEQ_FILL_FACTOR / MIB)

    # --- overhead -------------------------------------------------------------
    overhead_mib = int(CONTEXT_OVERHEAD_MIB
                       + weights_mib * FRAGMENTATION_FRACTION)

    total = (weights_mib + lora_mib + gradients_mib + optimizer_mib
             + activations_mib + overhead_mib)
    return VramEstimate(
        total_mib=total, weights_mib=weights_mib, lora_mib=lora_mib,
        gradients_mib=gradients_mib, optimizer_mib=optimizer_mib,
        activations_mib=activations_mib, overhead_mib=overhead_mib,
        load_mode=load_mode, notes=notes,
    )


def estimate_from_train_yaml(
    train_yaml_path: Optional[str],
    base_model_override: Optional[str] = None,
    *,
    python: Optional[str] = None,
) -> Optional[VramEstimate]:
    """End-to-end helper: resolved train.yaml (+ base model) -> estimate.

    Returns None when the base model's dims cannot be determined locally —
    callers treat that as "skip the VRAM prediction", never as zero.

    ``python`` is the interpreter training will actually run under
    (training.python). On ROCm boxes it selects which environment the
    bitsandbytes capability probe runs in — get this wrong (a torch-less
    daemon) and the estimate conservatively prices bf16-only, which is
    precisely what queue-blocked working QLoRA jobs before.
    """
    import yaml  # stdlib-adjacent: already a hard dependency of toaster_loop

    dims_src: Optional[str] = None
    tr: Dict[str, Any] = {}
    lora: Dict[str, Any] = {}
    if train_yaml_path and Path(str(train_yaml_path)).is_file():
        try:
            data = yaml.safe_load(Path(str(train_yaml_path)).read_text()) or {}
        except (OSError, ValueError):
            data = {}
        if isinstance(data, dict):
            tr = data.get("training") or {}
            lora = data.get("lora") or {}
            dims_src = str(data.get("model") or "")
    model = str(base_model_override or "").strip() or (dims_src or "").strip()
    if not model or model.startswith("REPLACE_"):
        return None
    dims = load_model_dims(model)
    if dims is None:
        return None

    target_modules = lora.get("target_modules")
    if isinstance(target_modules, str):
        target_modules = [t.strip() for t in target_modules.split(",") if t.strip()]

    load_in_4bit_explicit = tr.get("load_in_4bit")
    if load_in_4bit_explicit is not None and not isinstance(
            load_in_4bit_explicit, bool):
        load_in_4bit_explicit = bool(load_in_4bit_explicit)
    train_full = str(tr.get("training_type", "lora")).lower().strip() == "full"
    has_cuda = _cuda_available()
    load_in_4bit, tier_reason = resolve_load_in_4bit(
        explicit=load_in_4bit_explicit, dims=dims, train_full=train_full,
        gpu_available=has_cuda)
    # Pre-quantized checkpoints (compressed-tensors FP8/NVFP4 etc.) are NOT
    # loaded as nf4: the worker skips runtime 4-bit and transformers
    # dequantizes them to bf16 on non-NVIDIA hardware. Charging the gate
    # 0.56 B/param let a 27B FP8 run through at an ~70 GiB estimate while
    # really needing ~50 GiB of weights alone — then it OOM-died. Be honest.
    if load_in_4bit and _is_prequantized(model):
        load_in_4bit = False
        tier_reason += "; pre-quantized ckpt loads dequantized bf16"
    # ROCm: runtime nf4 is only safe where bitsandbytes actually works on
    # this GPU+wheel combination — decided by the child-process probe in
    # core/bnbprobe.py, not by a blanket "any AMD is bf16-only" guess. The
    # gate must price the SAME load the worker will perform (the old
    # blanket guard priced 27B jobs at bf16 and queue-blocked them forever
    # on machines where QLoRA fits).
    if load_in_4bit and _bnb_unavailable_rocm(python):
        load_in_4bit = False
        tier_reason += "; ROCm bitsandbytes nf4 probe unavailable/failed -> bf16 base"
    est = estimate_train_vram_mib(
        dims,
        lora_r=int(lora.get("lora_r", 64)),
        target_modules=target_modules,
        load_in_4bit=load_in_4bit and has_cuda,
        optim=str(tr.get("optim", "") or ""),
        batch_size=int(tr.get("batch_size", 2)),
        max_seq_length=int(tr.get("max_seq_length") or 16384),
        gradient_checkpointing=bool(tr.get("gradient_checkpointing", True)),
        attn_implementation=str(tr.get("attn_implementation", "sdpa") or "sdpa"),
        train_full=train_full,
    )
    est.notes.append(f"load tier: {tier_reason}")
    return est


def _cuda_available() -> bool:
    """Best-effort GPU presence without importing torch (scheduler side)."""
    import shutil

    return bool(shutil.which("nvidia-smi") or shutil.which("rocm-smi"))


def _is_prequantized(model_path: str) -> bool:
    """True when a local model dir's config.json carries quantization_config."""
    try:
        cfg_json = Path(str(model_path)).expanduser() / "config.json"
        return cfg_json.is_file() and "quantization_config" in json.loads(
            cfg_json.read_text())
    except (OSError, ValueError):
        return False


def _bnb_unavailable_rocm(python: Optional[str] = None) -> bool:
    """Mirror the worker's ROCm bitsandbytes *probe* on the gate side.

    Historically this returned True for any AMD box (pre-7.2 ROCm torch
    wheels SIGSEGV'd under bitsandbytes), which priced every large-model
    job as bf16 and queue-blocked QLoRA runs that now work. The real fix is
    to ask the hardware: core/bnbprobe.py exercises nf4 quantize+forward+
    backward in a throwaway child and caches positive verdicts, so the gate
    prices what the worker will actually load.

    ``python`` must be the interpreter training runs under (training.python);
    the probe needs torch, which the daemon itself deliberately lacks. Without
    it the probe fails environmentally and we stay conservative (bf16 pricing)
    — the old behavior, never a false admission. An AMD box is identified by
    rocm-smi present and nvidia-smi absent (a true hybrid sets
    TOASTER_LOOP_BNB_ROCM to decide explicitly)."""
    import shutil

    if not (shutil.which("rocm-smi") and not shutil.which("nvidia-smi")):
        return False
    try:
        from .bnbprobe import gate_says_bnb_off

        return gate_says_bnb_off(python)
    except Exception:  # noqa: BLE001 - probe machinery absent => stay safe
        return True


# --------------------------------------------------------- 4-bit tier policy

#: Auto-tier boundaries (params). < 3B: bf16 base (loads fast, fits, better
#: gradients than a 4-bit base); >= 3B with a GPU: nf4 base (QLoRA) —
#: bf16 would exceed consumer/unified pools. Chosen from the Strix Halo
#: measurements: 0.6B bf16 full-FT peaked at 5 GiB; 4-bit base loaded at
#: 0.96 GiB and trained correctly on the ROCm wheel.
TIER_4BIT_MIN_PARAMS = 3_000_000_000


def resolve_load_in_4bit(
    *,
    explicit: Optional[bool],
    dims: Optional["ModelDims"],
    train_full: bool,
    gpu_available: bool,
) -> Tuple[bool, str]:
    """Decide base-weight quantization for a run. Returns (load_in_4bit, why).

    Rules (the "works on close-to-all models, on consumer hardware" policy):
      * an explicit train.yaml value always wins (back-compat: templates
        that pin true/false keep their exact behavior);
      * full fine-tunes never train on a 4-bit base (dequantized anyway,
        half-precision is the sane floor);
      * no GPU -> False (the worker ignores 4-bit without CUDA anyway);
      * model >= 3B params -> 4-bit QLoRA; smaller -> bf16.
    Unknown dims (no local config.json) keep the historical default (True
    with a GPU): conservative on memory, which is the failure we can least
    afford on a shared pool.
    """
    if explicit is not None:
        return bool(explicit), f"explicit config ({explicit})"
    if train_full:
        return False, "full fine-tune: bf16 base"
    if not gpu_available:
        return False, "no GPU: native precision"
    if dims is None:
        return True, "auto-tier: model size unknown, defaulting to 4-bit"
    n = base_param_count(dims)
    if n >= TIER_4BIT_MIN_PARAMS:
        return True, f"auto-tier: {n / 1e9:.1f}B params >= 3B -> 4-bit QLoRA"
    return False, f"auto-tier: {n / 1e9:.1f}B params < 3B -> bf16 base"

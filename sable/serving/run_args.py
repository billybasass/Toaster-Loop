"""Map the `run_settings:` config section to llama-server CLI arguments.

Inspired by a run-settings panel (context length, KV cache
dtype, parallel slots, batch sizes, tensor parallelism, vision, VRAM budget,
mmap/mlock, extra args). sable keeps a small, explicit subset that maps
1:1 onto llama-server flags — no magic, everything visible in the command
line we spawn.

Precedence inside a launch command (supervisor/deploy build it left to right):
  binary + model/alias/host/port/-c  ->  hardware args (-ngl)  ->
  run_settings args  ->  cfg.extra_llama_args
so later (more explicit) sources win over earlier ones for duplicate flags
(llama.cpp takes the last occurrence).

`gpu_memory_pct` is a soft VRAM budget: when gpu_layers is unset (auto), the
full-offload layer count is scaled by the budget percentage, approximating
"fit the model into X% of VRAM" without needing model-size math.

Speculative decoding mirrors the standard speculative-decoding mode list (auto / ngram / mtp /
chained / off) but keeps only the llama.cpp emissions we can do 1:1 — no
per-model capability probing, so a mode whose support the binary lacks makes
llama-server exit early with its own error instead of silently falling back.

Vision (mmproj): `vision` is a load toggle for the projector GGUF that sits
next to the model GGUF (auto-detected when `gguf_path` is passed — see
core/mmproj.py); `mmproj_path` stays as an advanced manual override.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional

KV_CACHE_TYPES = ("f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl")

#: Speculative decoding modes (reference-style). Emissions:
#:   auto        -> --spec-type ngram-mod  (zero-VRAM self-speculation; the
#:                  mode studio's Auto resolves to for most models)
#:   ngram       -> --spec-type ngram-mod
#:   mtp         -> --spec-type draft-mtp          (needs an MTP head/drafter)
#:   ngram+mtp   -> --spec-type ngram-mod,draft-mtp (chained, one flag)
#:   off / unset -> nothing
SPEC_MODES = ("auto", "ngram", "mtp", "ngram+mtp", "off")

#: Legacy/alias spellings accepted on input (studio's own compatibility map).
_LEGACY_SPEC_MODES = {
    "default": "auto",
    "none": "off",
    "disable": "off",
    "disabled": "off",
    "ngram-mod": "ngram",
    "ngram-simple": "ngram",
    "draft-mtp": "mtp",
    "mtp+ngram": "ngram+mtp",
}


def canonical_spec_mode(value: Any) -> Optional[str]:
    """Normalize a speculative_type input onto SPEC_MODES; None when unset.

    Unknown strings collapse to None (no speculation) so a stale value can
    never crash the spawn; llama.cpp comma-chains map like studio's."""
    if value is None:
        return None
    s = str(value).strip().lower()
    if not s:
        return None
    if s in SPEC_MODES:
        return s
    if s in _LEGACY_SPEC_MODES:
        return _LEGACY_SPEC_MODES[s]
    pieces = {p.strip() for p in s.split(",") if p.strip()}
    has_mtp = bool(pieces & {"mtp", "draft-mtp"})
    has_ngram = bool(pieces & {"ngram", "ngram-mod", "ngram-simple"})
    if has_mtp and has_ngram:
        return "ngram+mtp"
    if has_mtp:
        return "mtp"
    if has_ngram:
        return "ngram"
    return None


def _int(v: Any) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def scale_gpu_layers(full_layers: int, gpu_memory_pct: Optional[int]) -> int:
    """Scale an offload layer count by a VRAM budget percentage (1..100).

    100 (or unset) => unchanged. Lower budgets keep proportionally fewer
    layers on the GPU; floors at 0 (CPU fallback).
    """
    if not gpu_memory_pct or gpu_memory_pct >= 100:
        return full_layers
    pct = max(1, min(100, int(gpu_memory_pct)))
    return max(0, int(full_layers * pct / 100))


def speculative_args(run_settings: Dict[str, Any],
                     gguf_path: Optional[str] = None) -> List[str]:
    """llama-server flags for the speculative-decoding knobs.

    `speculative_type` picks the mode (see SPEC_MODES); unset/off emits
    nothing. `spec_draft_n_max` (1..16) tunes draft length via
    --spec-draft-n-max and rides along with any drafter mode.

    MTP modes need a drafter GGUF: explicit `mtp_path` wins, else an
    ``mtp*.gguf`` companion sitting next to the model (hub repos ship one in an
    ``MTP/`` folder) is auto-detected and passed via --model-draft. When no
    drafter can be found the mode degrades to ngram-mod instead of letting
    llama-server exit early with a draft-model error."""
    args: List[str] = []
    mode = canonical_spec_mode(run_settings.get("speculative_type"))
    if not mode or mode == "off":
        return args
    drafter: Optional[str] = None
    if "mtp" in mode:
        explicit = str(run_settings.get("mtp_path") or "").strip()
        drafter = explicit if explicit and Path(explicit).is_file() else None
        if drafter is None and gguf_path:
            from ..core.mmproj import find_local_mtp

            drafter = find_local_mtp(gguf_path)
    if "mtp" in mode and drafter is None:
        mode = "ngram"  # no drafter available: fall back to self-speculation
    spec_type = {
        "auto": "ngram-mod",
        "ngram": "ngram-mod",
        "mtp": "draft-mtp",
        "ngram+mtp": "ngram-mod,draft-mtp",
    }[mode]
    args += ["--spec-type", spec_type]
    if drafter:
        args += ["--model-draft", drafter]
    n_max = _int(run_settings.get("spec_draft_n_max"))
    if n_max and 1 <= n_max <= 16:
        args += ["--spec-draft-n-max", str(n_max)]
    return args


def run_settings_args(run_settings: Optional[Dict[str, Any]],
                      gpu_layers: Optional[int] = None,
                      gguf_path: Optional[str] = None) -> List[str]:
    """llama-server args for the `run_settings:` section.

    `gpu_layers` (resolved -ngl value) is used only to interpret
    `gpu_memory_pct`; pass None when offload is handled elsewhere.
    `gguf_path` enables vision auto-detection: with `vision` not turned off
    and no explicit `mmproj_path`, an mmproj projector sitting next to the
    GGUF is loaded automatically (see core/mmproj.py).
    Unknown keys are ignored silently (config validation rejects them).
    """
    rs = run_settings or {}
    args: List[str] = []

    kv = str(rs.get("kv_cache_type") or "").strip().lower()
    if kv and kv != "f16" and kv in KV_CACHE_TYPES:
        args += ["-ctk", kv, "-ctv", kv]

    if rs.get("flash_attn"):
        # llama.cpp >= b5xxx treats --flash-attn as value-taking (on|off|auto);
        # a bare flag swallows the next argument (e.g. --mmproj) and exits 1.
        args += ["--flash-attn", "on"]

    ngl_scaled = None
    pct = _int(rs.get("gpu_memory_pct"))
    if gpu_layers is not None and pct:
        scaled = scale_gpu_layers(gpu_layers, pct)
        if scaled != gpu_layers:
            ngl_scaled = scaled
    if ngl_scaled is not None:
        args += ["-ngl", str(ngl_scaled)]

    par = _int(rs.get("parallel"))
    if par and par > 1:
        args += ["-np", str(par)]

    bs = _int(rs.get("batch_size"))
    if bs and bs > 0:
        args += ["-b", str(bs)]
    ub = _int(rs.get("ubatch_size"))
    if ub and ub > 0:
        args += ["-ub", str(ub)]

    if rs.get("tensor_parallel"):
        args.append("--tensor-parallel")
    if rs.get("no_mmap"):
        args.append("--no-mmap")
    if rs.get("mlock"):
        args.append("--mlock")

    from ..core.mmproj import resolve_mmproj

    if gguf_path is not None:
        # Model context available: toggle + sibling auto-detection (resolve).
        mmproj = resolve_mmproj(gguf_path, rs)
    else:
        # No model context (legacy callers): manual path only.
        mmproj = str(rs.get("mmproj_path") or "").strip() or None
    if mmproj:
        args += ["--mmproj", mmproj]

    args += speculative_args(rs, gguf_path=gguf_path)

    extra = str(rs.get("extra_args") or "").strip()
    if extra:
        try:
            args += shlex.split(extra)
        except ValueError:
            pass  # unbalanced quotes: skip rather than crash the spawn

    return args


#: Sampling knobs (reference style): the OpenAI-request parameters the
#: built-in runner and playground merge into every /chat/completions call.
#: Stored per server instance (and in the global `sampling:` section); they
#: are REQUEST-time parameters, not llama-server CLI flags.
SAMPLING_NUMERIC = {
    "temperature": (0.0, 2.0),
    "top_p": (0.0, 1.0),
    "min_p": (0.0, 1.0),
    "repeat_penalty": (0.5, 2.0),
    "presence_penalty": (-2.0, 2.0),
    "frequency_penalty": (-2.0, 2.0),
    "penalty_free_repeat": (1, 1000),
}
SAMPLING_INT = {"top_k", "max_tokens"}
#: Thinking / reasoning controls (studio: enable_thinking boolean,
#: reasoning_effort low|medium|high, preserve_thinking keeps historical
#:  blocks from past assistant turns in the prompt).
#: llama.cpp applies them via chat_format_template_controls; vLLM-style
#: servers accept top-level. We send both shapes; unknown keys are ignored.
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high")


def sanitize_sampling(patch: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce/validate a sampling (+thinking) patch from the API.

    Returns only explicitly-set keys (empty values clear to None so callers
    can drop them). Raises ValueError on unknown keys or out-of-range values.
    """
    out: Dict[str, Any] = {}
    allowed = (set(SAMPLING_NUMERIC) | set(SAMPLING_INT)
               | {"enable_thinking", "reasoning_effort", "preserve_thinking"})
    unknown = [k for k in patch if k not in allowed]
    if unknown:
        raise ValueError("unknown sampling keys: "
                         + ", ".join(map(str, sorted(unknown))))
    for key, (lo, hi) in SAMPLING_NUMERIC.items():
        if key in patch:
            v = patch[key]
            if v is None or v == "":
                out[key] = None
                continue
            try:
                f = float(v)
            except (TypeError, ValueError):
                raise ValueError(f"{key} must be a number") from None
            if not (lo <= f <= hi):
                raise ValueError(f"{key} must be between {lo} and {hi}")
            out[key] = int(f) if key == "penalty_free_repeat" else f
    for key in SAMPLING_INT:
        if key in patch:
            v = _int(patch[key])
            if patch[key] in (None, ""):
                out[key] = None
            elif v is None or v < 1:
                raise ValueError(f"{key} must be a positive integer")
            else:
                out[key] = v
    if "enable_thinking" in patch:
        out["enable_thinking"] = (bool(patch["enable_thinking"])
                                  if patch["enable_thinking"] is not None
                                  else None)
    if "preserve_thinking" in patch:
        out["preserve_thinking"] = (bool(patch["preserve_thinking"])
                                    if patch["preserve_thinking"] is not None
                                    else None)
    if "reasoning_effort" in patch:
        raw = patch["reasoning_effort"]
        if raw is None or not str(raw).strip():
            out["reasoning_effort"] = None
        else:
            v = str(raw).strip().lower()
            if v not in REASONING_EFFORTS:
                raise ValueError(
                    f"reasoning_effort must be one of {REASONING_EFFORTS}")
            out["reasoning_effort"] = v
    return out


def request_sampling_payload(sampling: Optional[Dict[str, Any]]
                             ) -> Dict[str, Any]:
    """OpenAI-request fragment for a stored sampling map.

    Sampling numbers ride top-level; the thinking controls are also packed
    into `chat_format_template_controls` (llama.cpp's template-control
    channel). Empty maps produce {}; None values are dropped."""
    s = sampling or {}
    frag: Dict[str, Any] = {k: v for k, v in s.items() if v is not None}
    controls = {k: frag[k] for k in
                ("enable_thinking", "reasoning_effort", "preserve_thinking")
                if k in frag}
    # Thinking keys stay top-level too (vLLM / studio-style servers read them
    # there); llama.cpp reads the controls map and ignores unknown params.
    if controls:
        frag["chat_format_template_controls"] = controls
    return frag


def sanitize_run_settings(patch: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce/validate a run_settings patch from the API. Raises ValueError."""
    out: Dict[str, Any] = {}
    allowed = {
        "kv_cache_type", "flash_attn", "parallel", "batch_size",
        "ubatch_size", "tensor_parallel", "no_mmap", "mlock",
        "mmproj_path", "vision", "speculative_type", "spec_draft_n_max",
        "mtp_path", "gpu_memory_pct", "extra_args",
    }
    unknown = [k for k in patch if k not in allowed]
    if unknown:
        raise ValueError("unknown run_settings keys: " + ", ".join(map(str, sorted(unknown))))
    if "kv_cache_type" in patch:
        v = str(patch["kv_cache_type"] or "").strip().lower()
        if v and v not in KV_CACHE_TYPES:
            raise ValueError(f"kv_cache_type must be one of {KV_CACHE_TYPES}")
        out["kv_cache_type"] = v or None
    for flag in ("flash_attn", "tensor_parallel", "no_mmap", "mlock"):
        if flag in patch:
            out[flag] = bool(patch[flag])
    for key in ("parallel", "batch_size", "ubatch_size"):
        if key in patch:
            v = _int(patch[key])
            if v is not None and v > 0:
                out[key] = v
            else:
                out[key] = None
    if "gpu_memory_pct" in patch:
        v = _int(patch["gpu_memory_pct"])
        if v is not None and not (1 <= v <= 100):
            raise ValueError("gpu_memory_pct must be 1..100")
        out["gpu_memory_pct"] = v
    if "vision" in patch:
        out["vision"] = bool(patch["vision"]) \
            if patch["vision"] is not None else None
    if "speculative_type" in patch:
        raw = patch["speculative_type"]
        if raw is None or not str(raw).strip():
            out["speculative_type"] = None
        else:
            mode = canonical_spec_mode(raw)
            if mode is None:
                raise ValueError(
                    f"speculative_type must be one of {SPEC_MODES}")
            out["speculative_type"] = mode
    if "spec_draft_n_max" in patch:
        v = _int(patch["spec_draft_n_max"])
        if v is not None and not (1 <= v <= 16):
            raise ValueError("spec_draft_n_max must be 1..16")
        out["spec_draft_n_max"] = v
    for key in ("mmproj_path", "mtp_path", "extra_args"):
        if key in patch:
            s = str(patch[key] or "").strip()
            out[key] = s or None
    return out


# ---------------------------------------------------------------- diffusion
# Image-generation engines (stable-diffusion.cpp's sd-server) take generation
# defaults on the command line — steps/CFG/size/sampler, not top_k. Stored
# per instance as ``image_settings`` (the chat run_settings/sampling maps are
# meaningless for a diffusion server). Flag names verified against
# ``sd-server --help``: --steps / --cfg-scale / -W / -H / --sampling-method /
# --scheduler / -s.

#: Sampler names accepted by sd-server (--sampling-method).
SD_SAMPLERS = (
   "euler", "euler_a", "heun", "dpm2", "dpm++2s_a", "dpm++2m", "dpm++2mv2",
   "ipndm", "ipndm_v", "lcm", "ddim_trailing", "tcd",
)

#: Sigma schedulers accepted by sd-server (--scheduler).
SD_SCHEDULERS = (
   "discrete", "karras", "exponential", "ays", "sgm_uniform", "simple",
   "smoothstep", "lcm",
)


def sanitize_image_settings(patch: Dict[str, Any]) -> Dict[str, Any]:
   """Coerce/validate an image_settings patch (diffusion engines). Raises
   ValueError on unknown keys or out-of-range values; empty clears to None."""
   out: Dict[str, Any] = {}
   allowed = {"steps", "cfg_scale", "width", "height", "sampler",
              "scheduler", "seed", "batch_count", "extra_args"}
   unknown = [k for k in patch if k not in allowed]
   if unknown:
       raise ValueError("unknown image_settings keys: "
                        + ", ".join(map(str, sorted(unknown))))
   if "steps" in patch:
       v = _int(patch["steps"])
       if patch["steps"] in (None, ""):
           out["steps"] = None
       elif v is None or not (1 <= v <= 150):
           raise ValueError("steps must be 1..150")
       else:
           out["steps"] = v
   if "cfg_scale" in patch:
       if patch["cfg_scale"] in (None, ""):
           out["cfg_scale"] = None
       else:
           try:
               f = float(patch["cfg_scale"])
           except (TypeError, ValueError):
               raise ValueError("cfg_scale must be a number") from None
           if not (0.0 <= f <= 30.0):
               raise ValueError("cfg_scale must be 0..30")
           out["cfg_scale"] = f
   for key in ("width", "height"):
       if key in patch:
           v = _int(patch[key])
           if patch[key] in (None, ""):
               out[key] = None
           elif v is None or not (64 <= v <= 8192):
               raise ValueError(f"{key} must be 64..8192")
           else:
               out[key] = v
   for key, options in (("sampler", SD_SAMPLERS),
                        ("scheduler", SD_SCHEDULERS)):
       if key in patch:
           raw = patch[key]
           if raw is None or not str(raw).strip():
               out[key] = None
           else:
               v = str(raw).strip().lower()
               if v not in options:
                   raise ValueError(f"{key} must be one of {options}")
               out[key] = v
   if "seed" in patch:
       v = _int(patch["seed"])
       if patch["seed"] in (None, ""):
           out["seed"] = None
       elif v is None or not (-1 <= v <= 2 ** 31 - 1):
           raise ValueError("seed must be >= -1 (-1 = random)")
       else:
           out["seed"] = v
   if "batch_count" in patch:
       v = _int(patch["batch_count"])
       if patch["batch_count"] in (None, ""):
           out["batch_count"] = None
       elif v is None or not (1 <= v <= 16):
           raise ValueError("batch_count must be 1..16")
       else:
           out["batch_count"] = v
   if "extra_args" in patch:
       s = str(patch["extra_args"] or "").strip()
       out["extra_args"] = s or None
   return out


def image_settings_args(image_settings: Optional[Dict[str, Any]]
                       ) -> List[str]:
   """sd-server CLI args for an image_settings map (generation defaults the
   server applies to every request unless overridden). Unknown keys ignored
   (sanitize_image_settings rejects them upstream)."""
   im = image_settings or {}
   args: List[str] = []
   steps = _int(im.get("steps"))
   if steps and steps > 0:
       args += ["--steps", str(steps)]
   cfg = im.get("cfg_scale")
   if cfg is not None and cfg != "":
       try:
           args += ["--cfg-scale", str(float(cfg))]
       except (TypeError, ValueError):
           pass
   w = _int(im.get("width"))
   if w and w > 0:
       args += ["-W", str(w)]
   h = _int(im.get("height"))
   if h and h > 0:
       args += ["-H", str(h)]
   sampler = str(im.get("sampler") or "").strip()
   if sampler:
       args += ["--sampling-method", sampler]
   scheduler = str(im.get("scheduler") or "").strip()
   if scheduler:
       args += ["--scheduler", scheduler]
   seed = _int(im.get("seed"))
   if seed is not None and seed >= -1:
       args += ["-s", str(seed)]
   batch = _int(im.get("batch_count"))
   if batch and batch > 1:
       args += ["-b", str(batch)]
   extra = str(im.get("extra_args") or "").strip()
   if extra:
       try:
           args += shlex.split(extra)
       except ValueError:
           pass  # unbalanced quotes: skip rather than crash the spawn
   return args

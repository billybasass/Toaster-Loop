"""(b) PEFT-free LoRALinear keeping the frozen base ``weight`` in place.

Inspiration (D-12: read as reference, re-implemented inline, never imported):
``airllm/airllm/air_llm/airllm/lora_linear.py``. WHY PEFT is not used here:
peft wraps modules behind its own LoraLayer abstraction and expects the base
weights to be materialized (it will try to move the whole model). In a
streamed trainer the base ``weight`` Parameter is swapped from disk per layer;
the adapter must live *inside* the same module so shard keys keep matching.

The one property that matters for Step 3 (GGUF-LoRA artifact pipeline):
``state_dict()`` of the wrapped module contains ``<module>.weight`` (the
original checkpoint tensor, frozen) plus ``<module>.lora_A`` / ``lora_B`` —
i.e. a module-name -> tensor mapping that a GGUF mapper can consume without
knowing peft's naming scheme.

torch is a training extra: imported LAZILY. The LoRALinear class subclasses
``torch.nn.Module``, so it is built by :func:`build_lora_module` and exposed
through PEP 562 ``__getattr__`` — ``from ... import LoRALinear`` works, while
importing this module without torch installed still succeeds (attribute
access then raises a clear ImportError).
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Optional, Tuple

#: Linear leaf names that are safe to adapt in mainstream decoder blocks
#: (attention + MLP projections). Mirrors the shape of airllm's target list;
#: arch-specific extras get appended by Step 2 per architecture.
DEFAULT_LORA_TARGETS: Tuple[str, ...] = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


def _torch():
    import torch  # lazy: training extra

    return torch


_LORA_CLASS: Any = None
_PACKED_HELPERS: Optional[dict] = None


def build_lora_module() -> Any:
    """Build (once) and return the ``LoRALinear`` nn.Module subclass."""
    global _LORA_CLASS
    if _LORA_CLASS is not None:
        return _LORA_CLASS
    torch = _torch()
    import torch.nn as nn
    import torch.nn.functional as F

    class LoRALinear(nn.Module):
        """``y = x W^T + b + scaling * (x A^T) B^T`` with frozen base ``W``.

        The original ``weight`` Parameter OBJECT is kept (not copied), so the
        module's state_dict layout still matches the checkpoint exactly — the
        streamed loader swaps ``self.weight.data`` in and out per layer, and
        a later GGUF-LoRA mapper resolves ``<module>.weight`` keys as usual.

        Only ``lora_A`` [r, in] and ``lora_B`` [out, r] require grad. Init is
        the standard LoRA pair (kaiming-uniform A, zeros B) so the adapter
        starts as an exact no-op regardless of base dtype/device.
        """

        def __init__(self, linear: Any, *, r: int = 16, alpha: int = 32,
                     dropout: float = 0.0, device: Any = None,
                     dtype: Any = None) -> None:
            super().__init__()
            if r < 1:
                raise ValueError(f"LoRA rank must be >= 1, got {r}")
            if dropout:
                # Streamed backward RECOMPUTES the forward (recompute.py); a
                # stochastic dropout mask would not replay and gradients would
                # silently diverge. Loudly forbidden, not silently disabled.
                raise ValueError(
                    "LoRA dropout must be 0 in streamed mode: backward "
                    "recompute requires a deterministic forward pass")
            self.in_features = int(linear.in_features)
            self.out_features = int(linear.out_features)
            self.r = int(r)
            self.alpha = float(alpha)
            self.scaling = self.alpha / self.r

            # Keep the SAME Parameter objects so shard keys <module>.weight /
            # <module>.bias keep matching after injection.
            self.weight = linear.weight
            self.bias = getattr(linear, "bias", None)
            self.weight.requires_grad_(False)
            if self.bias is not None:
                self.bias.requires_grad_(False)

            # Adapter storage: never meta. A meta base gives no device/dtype
            # to inherit, so fall back to cpu/fp32 for the trainable part.
            if device is None:
                dev = self.weight.device
                device = torch.device("cpu") if dev.type == "meta" else dev
            if dtype is None:
                dt = self.weight.dtype
                dtype = torch.float32 if dt in (torch.float32,) and \
                    self.weight.device.type == "meta" else dt

            self.lora_A = nn.Parameter(
                torch.empty(self.r, self.in_features, device=device, dtype=dtype))
            self.lora_B = nn.Parameter(
                torch.empty(self.out_features, self.r, device=device, dtype=dtype))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

        def forward(self, x: Any) -> Any:
            base = F.linear(x, self.weight, self.bias)
            a = self.lora_A.to(dtype=x.dtype)
            b = self.lora_B.to(dtype=x.dtype)
            return base + ((x @ a.t()) @ b.t()) * self.scaling

    _LORA_CLASS = LoRALinear
    return LoRALinear


def __getattr__(name: str) -> Any:
    """PEP 562: expose ``LoRALinear`` lazily (it needs torch at class-creation
    time, not at module-import time)."""
    if name == "LoRALinear":
        return build_lora_module()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ------------------------------------------------------------------ injection
def inject_lora(module: Any, target_modules: Iterable[str] = DEFAULT_LORA_TARGETS,
                *, r: int = 16, alpha: int = 32, dropout: float = 0.0,
                device: Any = None, dtype: Any = None) -> int:
    """Replace matching ``nn.Linear`` children with LoRALinear (in place).

    Returns the number of modules wrapped. Recurses through all children;
    already-wrapped modules are skipped (idempotent). Requires torch — call
    only from the streamed training path, never at daemon import time.
    """
    LoRALinear = build_lora_module()
    import torch.nn as nn

    targets = set(target_modules)
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            continue
        if isinstance(child, nn.Linear) and name in targets:
            setattr(module, name, LoRALinear(
                child, r=r, alpha=alpha, dropout=dropout,
                device=device, dtype=dtype))
            n += 1
        else:
            n += inject_lora(child, targets, r=r, alpha=alpha, dropout=dropout,
                             device=device, dtype=dtype)
    return n


# ------------------------------------------------------------- adapter state
def lora_parameters(module: Any) -> list:
    """Trainable parameters only (names containing ``lora_``)."""
    return [p for nm, p in module.named_parameters() if "lora_" in nm]


def lora_state_dict(module: Any) -> dict:
    """Adapter-only checkpoint: {module-name -> cpu tensor} for lora_* keys.

    Flat module-name -> tensor mapping (the format Step 3's GGUF-LoRA mapper
    will consume). Base weights are NOT included — they live in the shard
    tree already; duplicating them here would double disk cost.
    """
    return {nm: p.detach().cpu()
            for nm, p in module.named_parameters() if "lora_" in nm}


def load_lora_state_dict(module: Any, state: dict, *, strict: bool = True) -> list:
    """Copy adapter tensors into ``module``; returns the missing-key list.

    Copies via ``param.data.copy_`` so Parameter identity (and therefore any
    optimizer state keyed on the parameter) survives a reload.
    """
    missing: list = []
    owned = dict(module.named_parameters())
    for nm, value in state.items():
        param = owned.get(nm)
        if param is None:
            missing.append(nm)
            continue
        with _torch().no_grad():
            param.data.copy_(value.to(device=param.device, dtype=param.dtype))
    if strict and missing:
        raise KeyError(f"LoRA keys not found in module: {missing[:8]}")
    return missing

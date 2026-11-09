"""bitsandbytes nf4 capability probe for AMD/ROCm boxes — stdlib parent, torch child.

Why this exists: the worker used to hard-block runtime nf4 on ANY ROCm build
("the PyPI bitsandbytes SIGSEGVs on Strix Halo" — true while the installed
torch wheel was a pre-7.2 ROCm build whose bundled HSA/rocprofiler stack died
at the first kernel launch, pytorch/pytorch#173125). That blanket guard is now
the bug: with a working wheel (rocm7.2+) bitsandbytes nf4 trains fine on the
same gfx1151 hardware, yet the guard forced bf16 loads — pricing a 27B model
at ~136 GiB against a 120 GiB unified pool and queueing it forever while
QLoRA's ~101 GiB fits comfortably.

The honest replacement is a *capability probe*: exercise the real nf4 path
(quantize + forward + backward-through-frozen-weight) in a THROWAWAY CHILD
process — a genuine SIGSEGV must never take the caller down — and let both
sides price the same reality: the worker's load decision AND core.vram's
scheduler-gate estimate.

Caching policy (deliberately asymmetric):
  * POSITIVE verdicts are cached durably under TOASTER_LOOP_HOME/cache/, keyed
    to the ROCm userland release (+ torch version when the reader can see it).
    A working nf4 stack keeps working until one of those changes.
  * NEGATIVE verdicts are NOT written to disk — a failure may be transient
    (GPU held by another process, daemon interpreter without torch installed),
    and a poisoned "bf16-only" cache would queue-block large-model jobs
    forever: exactly the bug this module exists to fix. Negatives are memoized
    in-process for _NEG_MEMO_TTL_S so a retrying dispatcher does not hammer
    the GPU, and recomputed after that.

License hygiene (D-12): nothing here imports torch/bitsandbytes in THIS
process; the child does, via ``sys.executable`` or an explicit interpreter.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

#: Env force knob (pre-existing name): "1/true/yes/on" forces bitsandbytes on,
#: "0/false/no/off" forces it off; anything else (unset) = probe decides.
ENV_FORCE = "TOASTER_LOOP_BNB_ROCM"

_CACHE_NAME = "bnb_rocm_probe.json"
_PROBE_TIMEOUT_S = 300.0
_NEG_MEMO_TTL_S = 60.0

#: In-process memo of recent negative verdicts: {python_executable: (ok, why, ts)}.
_neg_memo: Dict[str, Tuple[bool, str, float]] = {}

#: Child program. Models the QLoRA shape exactly: quantize a Linear4bit (the
#: historical SIGSEGV site), run a bf16 forward, and backprop THROUGH the
#: frozen quantized weight — in a real LoRA run only adapter params get grads,
#: so x requires grad and lin.weight does not. ``stage`` in the emitted JSON
#: tells the parent WHERE it failed, which is what decides cacheability.
_CHILD_SRC = r"""
import json, sys
out = {"ok": False, "stage": "import"}
try:
    import torch
    import bitsandbytes as bnb
except Exception as e:  # noqa: BLE001 - environmental (no stack here)
    out["detail"] = f"{type(e).__name__}: {e}"
    print(json.dumps(out)); sys.exit(2)
out["torch"] = torch.__version__
out["hip"] = getattr(torch.version, "hip", None)
out["stage"] = "device"
try:
    if not torch.cuda.is_available():
        out["detail"] = "no CUDA/HIP device visible to the probe child"
        print(json.dumps(out)); sys.exit(3)
    out["device"] = torch.cuda.get_device_name(0)
    out["stage"] = "probe"
    lin = bnb.nn.Linear4bit(64, 32, bias=False,
                            compute_dtype=torch.bfloat16, quant_type="nf4")
    lin.to("cuda")
    lin.weight.requires_grad_(False)
    x = torch.randn(2, 8, 64, device="cuda", dtype=torch.bfloat16,
                    requires_grad=True)
    y = lin(x)
    y.sum().backward()
    grad_ok = x.grad is not None and bool(torch.isfinite(x.grad).all())
    torch.cuda.synchronize()
    out["ok"] = bool(grad_ok)
    out["detail"] = ("Linear4bit nf4 quantize + forward + backward-through-"
                     "frozen-weight ok" if grad_ok
                     else "backward produced no finite input gradient")
except Exception as e:  # noqa: BLE001 - the child exists to absorb this (incl. SIGSEGV)
    out["detail"] = f"{type(e).__name__}: {e}"
print(json.dumps(out))
sys.exit(0 if out["ok"] else 1)
"""


def _truthy(raw: str) -> bool:
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _falsy(raw: str) -> bool:
    return raw.strip().lower() in ("0", "false", "no", "off")


def force_override() -> Optional[bool]:
    """TOASTER_LOOP_BNB_ROCM as an explicit True/False, or None to probe."""
    raw = os.environ.get(ENV_FORCE, "")
    if _truthy(raw):
        return True
    if _falsy(raw):
        return False
    return None


def cache_path() -> Path:
    """<TOASTER_LOOP_HOME>/cache/bnb_rocm_probe.json (dir not created here)."""
    from ..store.paths import resolve_toaster_loop_home

    return resolve_toaster_loop_home() / "cache" / _CACHE_NAME


def _installed_torch_version() -> str:
    """torch wheel version via metadata only (never imports torch)."""
    try:
        from importlib.metadata import version as _pkg_version

        return _pkg_version("torch")
    except Exception:  # noqa: BLE001 - torch absent => no cache match anyway
        return ""


def _userland_key() -> str:
    """ROCm userland release (wheel capability is keyed to it; a GPU stack
    upgrade must invalidate a cached verdict). Empty string when unknown."""
    try:
        from .torchenv import rocm_userland_version

        return rocm_userland_version() or ""
    except Exception:  # noqa: BLE001 - probe-side nicety, never fatal
        return ""


def load_cached() -> Optional[Dict[str, Any]]:
    """The cached POSITIVE verdict if present and still valid; None otherwise.

    Staleness rules (the gate side usually has NO torch installed — the
    daemon deliberately keeps the ML stack in the training env):
      * ROCm userland must match what was probed: a GPU-stack upgrade
        invalidates the verdict unconditionally;
      * if THIS process can see a torch, its version must match the probed
        one (a wheel swap changes capability);
      * otherwise fall back to "the probed interpreter still exists" — as
        close to the same environment as we can verify without importing.
    """
    try:
        raw = json.loads(cache_path().read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict) or not raw.get("ok"):
        return None
    if str(raw.get("userland", "")) != _userland_key():
        return None
    local_torch = _installed_torch_version()
    if local_torch:
        if str(raw.get("torch_version", "")) != local_torch:
            return None
    else:
        py = str(raw.get("python", ""))
        if not py or not Path(py).exists():
            return None
    return raw


def _write_cache(rec: Dict[str, Any]) -> None:
    """Persist a POSITIVE verdict atomically (negatives never land here)."""
    try:
        p = cache_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec))
        os.replace(tmp, p)  # atomic: a torn cache can never mis-gate
    except OSError:
        pass  # cache is an optimization; the verdict still returns


def probe(python: Optional[str] = None) -> Dict[str, Any]:
    """Run the nf4 capability check in a child process. Never raises.

    ``python`` selects the interpreter (defaults to this one — from inside
    the worker that IS the training env). Positive results are cached;
    negatives are returned uncached so a transient GPU condition cannot
    pin the whole system to bf16 pricing.
    """
    exe = str(python or sys.executable)
    rec: Dict[str, Any] = {
        "ok": False,
        "stage": "",
        "detail": "",
        "python": exe,
        "torch_version": _installed_torch_version(),
        "userland": _userland_key(),
    }
    try:
        proc = subprocess.run([exe, "-c", _CHILD_SRC], capture_output=True,
                              text=True, timeout=_PROBE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        rec["detail"] = (f"probe child timed out after {int(_PROBE_TIMEOUT_S)}s "
                         "(GPU busy or hung?)")
        return rec
    except OSError as e:
        rec["detail"] = f"cannot spawn probe child: {e}"
        return rec
    tail = [l for l in (proc.stderr or "").strip().splitlines() if l.strip()][-2:]
    try:
        info = json.loads((proc.stdout or "").strip().splitlines()[-1])
    except (ValueError, IndexError):
        info = {}
    rec["stage"] = str(info.get("stage", ""))
    rec["ok"] = bool(proc.returncode == 0 and info.get("ok"))
    rec["detail"] = str(info.get("detail") or " ".join(tail)
                        or f"probe child exited rc={proc.returncode}")
    for k in ("torch", "hip", "device"):
        if info.get(k):
            rec[k] = info[k]
    if rec["ok"]:
        _write_cache(rec)
    return rec


def rocm_bnb_supported(python: Optional[str] = None, *,
                       use_cache: bool = True) -> Tuple[bool, str]:
    """Can runtime nf4 be trusted on this AMD box? Returns (ok, why).

    Order: explicit TOASTER_LOOP_BNB_ROCM override > cached positive verdict
    > recent in-process negative memo > live probe. Callers must only consult
    this on a ROCm/HIP torch build; non-ROCm paths keep their own
    bitsandbytes handling.
    """
    forced = force_override()
    if forced is not None:
        return forced, f"forced by {ENV_FORCE}"
    if use_cache:
        cached = load_cached()
        if cached is not None:
            return True, f"cached probe: {cached.get('detail', '')}"
        memo = _neg_memo.get(str(python or sys.executable))
        if memo and (time.monotonic() - memo[2]) < _NEG_MEMO_TTL_S:
            return False, f"{memo[1]} (re-probes in " \
                          f"{int(_NEG_MEMO_TTL_S - (time.monotonic() - memo[2]))}s)"
    rec = probe(python)
    if not rec["ok"]:
        _neg_memo[str(python or sys.executable)] = (False, str(rec["detail"]),
                                                    time.monotonic())
    return bool(rec["ok"]), str(rec["detail"])


def gate_says_bnb_off(python: Optional[str] = None) -> bool:
    """core.vram's view: should the scheduler price this AMD box as bf16-only?

    Probing is cheap (a few seconds, a throwaway child) and the alternative —
    assuming bf16 — is what queue-blocked working QLoRA runs. A daemon without
    torch in its interpreter gets an environmental failure back and stays
    conservative (bf16 pricing), which is exactly the old behavior; pointing
    ``training.python`` at the GPU-enabled interpreter fixes it."""
    forced = force_override()
    if forced is not None:
        return not forced  # forced ON => "bf16-only" is False
    ok, _why = rocm_bnb_supported(python)
    return not ok

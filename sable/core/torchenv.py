"""torch build-flavor resolution — "training works on any hardware".

Why this exists: ``pip install torch`` on Linux resolves from plain PyPI,
which ships the *CUDA* wheel. On an AMD (ROCm), Apple (MPS) or CPU-only box
that wheel imports fine but ``torch.cuda.is_available()`` is silently False —
training then crawls on CPU (or dies in bitsandbytes) with no error. This
saw real damage: a Strix Halo (gfx1151) box trained on CPU for months because
uv.lock pinned ``torch`` from PyPI (a ``+cu130`` build).

The rule: the torch build must match the accelerator, and the right builds
live on per-flavor indexes (``https://download.pytorch.org/whl/<flavor>``).
This module is stdlib-only (no torch import): it classifies the machine via
``serving.runtime`` detection + ROCm userland probes and returns the flavor
name, the index URL, and ready-made pip/uv argument lists. ``setup.sh`` /
``setup.ps1`` install torch from the right index BEFORE ``[training]``
extras resolve; ``doctor`` uses :func:`installed_mismatch` to flag an
already-installed wheel that can never see the GPU it sits next to.

Override: ``SABLE_TORCH_FLAVOR`` (env) or the ``flavor`` argument accepts
``default`` (plain PyPI: macOS/MPS and Windows-CUDA ship MPS/CUDA in the
default wheel), ``cpu``, ``cu126|cu130|cu132``, ``rocm6.4|rocm7.2|...``,
``xpu``.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

#: Where torch publishes its per-accelerator wheels.
PYTORCH_WHL_BASE = "https://download.pytorch.org/whl"

#: Sentinel flavor meaning "install from plain PyPI" (no special index).
FLAVOR_DEFAULT = "default"

#: CUDA wheel flavors published on download.pytorch.org (newest first).
CUDA_FLAVORS = ("cu132", "cu130", "cu126")

#: ROCm wheel flavors published on download.pytorch.org (newest first).
#: A machine with ROCm userland version X should get the newest flavor <= X.
ROCM_FLAVORS = ("7.2", "7.1", "7.0", "6.4", "6.3", "6.2", "6.1", "6.0", "5.7")

#: Linux CPU wheels come from the dedicated index (small, no CUDA bloat).
LINUX_CPU_FLAVOR = "cpu"


@dataclass
class TorchFlavor:
    """The torch build this machine should install, and how to install it."""

    flavor: str                      # default | cpu | cuNNN | rocmX.Y | xpu
    index_url: Optional[str]         # None => plain PyPI
    notes: List[str] = field(default_factory=list)

    @property
    def is_pypi(self) -> bool:
        return self.index_url is None

    def pip_args(self) -> List[str]:
        """Extra args for ``pip install torch`` (index pinning only)."""
        if self.index_url:
            return ["--index-url", self.index_url]
        return []

    def uv_pip_args(self) -> List[str]:
        """Extra args for ``uv pip install torch``.

        uv understands ``--torch-backend`` natively (it also rewrites the
        local version tag), which is more robust than a hand-rolled index;
        plain index args still work for the flavors it doesn't know."""
        if self.flavor == FLAVOR_DEFAULT:
            return []
        return ["--torch-backend", self.flavor, "--extra-index-url",
                PYTORCH_WHL_BASE]

    def __str__(self) -> str:  # log-friendly
        return f"{self.flavor} ({self.index_url or 'PyPI'})"


def _env_override() -> str:
    return os.environ.get("SABLE_TORCH_FLAVOR", "").strip().lower()


def rocm_userland_version() -> Optional[str]:
    """Installed ROCm version like '7.2.1', or None when no ROCm compiler
    runtime is present. We require hipcc (or hipcc's own version string) —
    not merely rocminfo — because a CUDA-capable box may still ship
    rocminfo, and torch's ROCm wheels need the HIP runtime to load.

    Upstream ROCm installs land in /opt/rocm with a .info/version file;
    Ubuntu/Debian packages ROCm into the system tree (/usr) with no such
    file, so hipcc output and finally dpkg are consulted in order."""
    vfile = Path("/opt/rocm/.info/version")
    try:
        if vfile.is_file():
            v = vfile.read_text().strip()
            if re.match(r"^\d+\.\d+", v):
                return v
    except OSError:
        pass
    import shutil
    import subprocess
    if shutil.which("hipcc"):
        try:
            out = (subprocess.run(["hipcc", "--version"], capture_output=True,
                                   text=True, timeout=8).stdout or "")
            # Upstream prints "ROCM version: 6.2.4"; Ubuntu prints
            # "HIP version: 7.1.52801-9999" (HIP versioning — the major.minor
            # still tracks the ROCm release).
            m = (re.search(r"ROCM version:\s*(\d+\.\d+[\.\d]*)", out)
                 or re.search(r"HIP version:\s*(\d+\.\d+)", out))
            if m:
                return m.group(1)
        except Exception:
            pass
        # hipcc exists but its output parsed oddly: fall through to dpkg,
        # then give up honestly (None) rather than guess the oldest flavor.
        out = _run(["dpkg-query", "-W", "-f", "${Version}", "rocm-core"])
        m = re.search(r"(\d+\.\d+)", out or "")
        if m:
            return m.group(1)
    return None


def _run(cmd: List[str], timeout: float = 8.0) -> Optional[str]:
    import shutil
    import subprocess
    exe = shutil.which(cmd[0])
    if not exe:
        return None
    try:
        return subprocess.run([exe, *cmd[1:]], capture_output=True, text=True,
                              timeout=timeout).stdout
    except Exception:
        return None


def _rocm_flavor(version: str) -> str:
    """Newest published ROCm wheel flavor <= the installed userland."""
    def key(v: str) -> Tuple[int, ...]:
        return tuple(int(p) for p in re.findall(r"\d+", v)[:3]) or (0,)
    try:
        want = key(version)
    except ValueError:
        want = key("0.0")
    for f in ROCM_FLAVORS:  # newest first
        if key(f) <= want:
            return f"rocm{f}"
    return f"rocm{ROCM_FLAVORS[-1]}"


def _detect_gpus() -> List[Any]:
    """GPUs via serving.runtime.detect_runtime(). That module is stdlib-only,
    but its parent package __init__ imports httpx — which a half-built setup
    venv may not have yet. Fall back to loading runtime.py by path, and to
    an empty list when even that fails (=> CPU flavor, honest and safe)."""
    try:
        from ..serving import runtime  # type: ignore[import-not-found]
        return list(runtime.detect_runtime().gpus)
    except Exception:
        pass
    try:
        import importlib.util
        import sys
        rt = Path(__file__).resolve().parents[1] / "serving" / "runtime.py"
        spec = importlib.util.spec_from_file_location(
            "sable_runtime_probe", rt)
        mod = importlib.util.module_from_spec(spec)
        # @dataclass resolves annotations via sys.modules at class-build
        # time: the module must be registered BEFORE exec_module.
        sys.modules["sable_runtime_probe"] = mod
        try:
            spec.loader.exec_module(mod)  # stdlib-only: no sable imports
            return list(mod.detect_runtime().gpus)
        finally:
            sys.modules.pop("sable_runtime_probe", None)
    except Exception:
        return []


def detect_flavor(system: Optional[str] = None,
                  machine: Optional[str] = None,
                  gpus: Optional[List[Any]] = None,
                  rocm_version: Optional[str] = None,
                  ) -> TorchFlavor:
    """Choose the torch build for this machine. Pure-ish: every probe is an
    argument so tests can fake the hardware. ``gpus`` entries need a
    ``.vendor`` attribute (serving.runtime.GpuInfo fits). ``rocm_version``:
    None probes this machine; "" asserts "no ROCm userland" (no probe)."""
    ov = _env_override()
    notes: List[str] = []
    if ov:
        notes.append(f"SABLE_TORCH_FLAVOR={ov} (override)")
        if ov == FLAVOR_DEFAULT:
            return TorchFlavor(FLAVOR_DEFAULT, None, notes)
        idx = (f"{PYTORCH_WHL_BASE}/{ov}" if ov != LINUX_CPU_FLAVOR
               else f"{PYTORCH_WHL_BASE}/{LINUX_CPU_FLAVOR}")
        return TorchFlavor(ov, idx, notes)

    import platform as _pl
    import sys
    system = (system or sys.platform or _pl.system().lower()).lower()
    if system.startswith("darwin"):
        # Apple: PyPI default wheels ARE the MPS builds. No index needed.
        return TorchFlavor(FLAVOR_DEFAULT, None,
                           ["Apple Silicon: default wheel ships MPS"])
    machine = (machine or _pl.machine() or "").lower()
    if gpus is None:
        gpus = _detect_gpus()
        system = system or _pl.system().lower()
    vendors = {getattr(g, "vendor", "") for g in (gpus or [])}

    if system == "win32":
        # PyPI default wheels are CUDA builds on Windows; Intel Arc has xpu.
        if "nvidia" in vendors:
            return TorchFlavor(FLAVOR_DEFAULT, None,
                               ["Windows+NVIDIA: default wheel ships CUDA"])
        if "intel" in vendors:
            return TorchFlavor("xpu", f"{PYTORCH_WHL_BASE}/xpu",
                               ["Windows+Intel Arc: xpu build"])
        return TorchFlavor(FLAVOR_DEFAULT, None,
                           ["Windows, no discrete GPU: default wheel "
                            "(CPU); pass SABLE_TORCH_FLAVOR to force one"])

    # Linux (and other unixes):
    nvidia = "nvidia" in vendors
    if nvidia:
        return TorchFlavor(CUDA_FLAVORS[0],
                           f"{PYTORCH_WHL_BASE}/{CUDA_FLAVORS[0]}",
                           ["NVIDIA GPU: CUDA build"])
    if "amd" in vendors:
        rocm_version = (rocm_userland_version() or ""
                        if rocm_version is None else rocm_version)
        if rocm_version:
            f = _rocm_flavor(rocm_version)
            notes.append(f"AMD GPU + ROCm userland {rocm_version}: {f} build")
            return TorchFlavor(f, f"{PYTORCH_WHL_BASE}/{f}", notes)
        notes.append(
            "AMD GPU but no ROCm userland (/opt/rocm, hipcc): torch has no "
            "Vulkan backend, so training falls back to CPU. Install the ROCm "
            "stack, or SABLE_TORCH_FLAVOR to override. (Serving is "
            "unaffected: llama.cpp uses Vulkan.)")
        return TorchFlavor(LINUX_CPU_FLAVOR,
                           f"{PYTORCH_WHL_BASE}/{LINUX_CPU_FLAVOR}", notes)
    if "intel" in vendors and machine in ("x86_64", "amd64"):
        return TorchFlavor("xpu", f"{PYTORCH_WHL_BASE}/xpu",
                           ["Intel GPU: xpu build"])
    notes.append("No GPU detected: CPU build")
    return TorchFlavor(LINUX_CPU_FLAVOR,
                       f"{PYTORCH_WHL_BASE}/{LINUX_CPU_FLAVOR}", notes)


# ------------------------------------------------------- installed-wheel check

def installed_mismatch(torch_mod: Any, want: TorchFlavor
                       ) -> Optional[str]:
    """Describe why the INSTALLED torch can never use this machine's GPU, or
    None when it looks fine. This is the check whose absence cost a Strix
    Halo machine months of silent CPU training: a ``+cu130`` wheel imports
    and reports versions happily while ``cuda.is_available()`` is False."""
    version = str(getattr(torch_mod, "__version__", ""))
    hip = getattr(getattr(torch_mod, "version", None), "hip", None)
    cuda = getattr(getattr(torch_mod, "version", None), "cuda", None)
    try:
        cuda_ok = bool(torch_mod.cuda.is_available())
    except Exception:
        cuda_ok = False
    return _mismatch_for(version=version, hip=hip, cuda=cuda,
                         cuda_ok=cuda_ok, want=want)


def metadata_mismatch(version: str, want: TorchFlavor) -> Optional[str]:
    """installed_mismatch from the wheel's *version string alone* — no torch
    import (license hygiene: heavy imports stay worker-only). The local
    version tag (+cu130, +rocm7.2, +cpu) classifies the wheel; a tagless
    "2.14.1" from plain PyPI IS the CUDA wheel on Linux. Runtime
    availability probing is the worker's job, not the doctor's."""
    m = re.search(r"\+(cpu|cu\d+|rocm[\d.]+|xpu)", version)
    if m:
        actual = m.group(1)
    elif want.flavor.startswith("rocm") or want.flavor.startswith("cu"):
        actual = "pypi-default(=CUDA on Linux)"
    else:
        actual = "unknown"
    return _mismatch_for(version=version, hip=actual.startswith("rocm") or None,
                         cuda=None if actual.startswith("rocm")
                         or actual == "cpu" else "yes",
                         cuda_ok=False, want=want)


def _mismatch_for(version: str, hip: Optional[str], cuda: Optional[str],
                  cuda_ok: bool, want: TorchFlavor) -> Optional[str]:
    m = re.search(r"\+(cpu|cu\d+|rocm[\d.]+|xpu)", version)
    actual = m.group(1) if m else ("pypi-cuda" if cuda else "unknown")
    # Any +rocm wheel (or torch.version.hip present) is a ROCm build: it
    # hears HIP on this machine, including newer-flavor-on-older-userland
    # (rocm7.2 wheel + 7.1 userland works and is common).
    if hip:
        return None
    if want.flavor.startswith("rocm"):
        hint = ("reinstall with: pip install torch --index-url "
                f"{PYTORCH_WHL_BASE}/{want.flavor}")
        return (f"torch {version or '?'} is a {actual} build but this "
                f"machine wants {want.flavor} (AMD GPU + ROCm). "
                f"cuda.is_available()={cuda_ok} — training would silently "
                f"run on CPU. {hint}")
    if want.flavor.startswith("cu") and cuda is None and not cuda_ok:
        return (f"torch {version or '?'} ({actual}) has no CUDA support but "
                f"an NVIDIA GPU is present. reinstall with: pip install torch "
                f"--index-url {PYTORCH_WHL_BASE}/{want.flavor}")
    if want.flavor == "cpu" and cuda_ok:
        return None  # CPU build that sees a GPU is contradictory, but harmless
    return None


def check_function(find_spec: Callable[[str], Any]
                   ) -> Tuple[bool, List[str]]:
    """Doctor probe: (ok, lines). Reads only the installed wheel's version
    *metadata* (importlib.metadata) — torch is never imported here (heavy
    imports are worker-only; see tests/test_hygiene.py)."""
    lines: List[str] = []
    if find_spec("torch") is None:
        return True, ["torch: not installed (training extras absent)"]
    try:
        from importlib.metadata import version as _pkg_version
        tv = _pkg_version("torch")
    except Exception:
        tv = ""
    want = detect_flavor()
    problem = metadata_mismatch(tv, want)
    lines.append(f"torch {tv or '?'}: wants {want.flavor}")
    if want.notes:
        lines.extend(f"  {n}" for n in want.notes)
    if problem:
        return False, lines + [f"  {problem}"]
    return True, lines

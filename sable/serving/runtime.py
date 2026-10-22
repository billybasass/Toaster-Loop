"""Runtime detection + llama.cpp self-setup — "runs on any GPU".

sable never assumes a particular vendor. This module:

  1. detects the available accelerator (NVIDIA CUDA / AMD ROCm / Apple Metal /
     Vulkan / CPU-only) using only cheap stdlib probes and vendor CLIs
     (nvidia-smi, rocminfo, vulkaninfo) — no torch import, so the daemon stays
     light (see tests/test_license_hygiene.py);
  2. turns that into llama-server launch args (``-ngl`` offload) so a GPU build
     gets full offload and a CPU-only box simply runs on the CPU;
  3. can *set up llama.cpp itself*: clone + build with the backend matching the
     detected hardware into ``<SABLE_HOME>/llama.cpp`` when no suitable
     binary exists anywhere else.

Auto-setup is opt-out (env ``SABLE_NO_AUTO_SETUP=1`` or config
``llama_auto_setup: false``) and only ever runs when no llama-server was found
— it never rebuilds over an existing install, and never touches
foreign per-installer directories.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger("sable.runtime")

#: llama.cpp convention for "offload as many layers as fit".
FULL_OFFLOAD_LAYERS = 999

#: Default upstream; overridable per install via config `llama_repo` /
#: `sable setup --repo` (e.g. a vendor fork like
#: https://github.com/Lychee-Technology/llama-cpp-for-strix-halo).
LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp"

#: Git ref (branch/tag) built by default when nothing else is configured.
LLAMA_CPP_REF = "master"

# Backend names (stable, reported by `sable doctor` / dashboard).
BACKEND_CUDA = "cuda"
BACKEND_ROCM = "rocm"
BACKEND_METAL = "metal"
BACKEND_VULKAN = "vulkan"
BACKEND_CPU = "cpu"


@dataclass
class GpuInfo:
    vendor: str          # nvidia | amd | apple | other
    name: str
    free_mib: Optional[int] = None
    total_mib: Optional[int] = None


@dataclass
class RuntimeInfo:
    system: str                      # linux | darwin | windows | ...
    machine: str                     # x86_64 | arm64 | ...
    gpus: List[GpuInfo] = field(default_factory=list)
    backend: str = BACKEND_CPU       # chosen llama.cpp backend
    backend_notes: List[str] = field(default_factory=list)


def _run(cmd: List[str], timeout: float = 8.0) -> Optional[str]:
    """Best-effort subprocess text output; None on any failure."""
    exe = shutil.which(cmd[0])
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, *cmd[1:]], capture_output=True, text=True, timeout=timeout
        )
        if out.returncode != 0:
            return None
        return (out.stdout or "") + (out.stderr or "")
    except Exception:
        return None


# --------------------------------------------------------------------- GPUs

def _vulkan_vendor(name: str) -> str:
    """Coarse vendor classification from a Vulkan deviceName."""
    n = (name or "").lower()
    if "nvidia" in n or "geforce" in n or "rtx" in n or "gtx" in n:
        return "nvidia"
    if ("amd" in n or "radeon" in n or "rdna" in n or "vega" in n
            or "graphics api" in n):
        return "amd"
    if "intel" in n or "arc" in n or re.search(r"\buhd?\b", n) or "hd graphics" in n:
        return "intel"
    if "apple" in n or "m1" in n or "m2" in n or "m3" in n or "m4" in n:
        return "apple"
    return "other"


def _vulkan_gpus() -> List[GpuInfo]:
    """Enumerate GPUs via ``vulkaninfo --summary`` (portable across OSes).

    Best-effort: parses per-device ``deviceName`` and sums the DEVICE_LOCAL
    memory heaps for a total. Software rasterizers (llvmpipe/swrast) are
    skipped — they are not accelerators. Never raises; empty list when
    vulkaninfo is absent or unparseable."""
    out = _run(["vulkaninfo", "--summary"], timeout=15.0)
    if not out:
        return []
    def _size_token(s: str) -> Optional[int]:
        m = re.search(r"(0x[0-9a-fA-F]{6,}|\d{7,})", s)
        if not m:
            return None
        tok = m.group(1)
        return int(tok, 16) if tok.startswith("0x") else int(tok)

    gpus: List[GpuInfo] = []
    cur_name: Optional[str] = None
    cur_local_mib = 0
    is_cpu_dev = False
    heap_local = False  # current memoryHeaps block carries DEVICE_LOCAL_BIT

    def _flush() -> None:
        nonlocal cur_name, cur_local_mib, is_cpu_dev, heap_local
        if (cur_name and not is_cpu_dev
                and "llvmpipe" not in cur_name.lower()
                and "swrast" not in cur_name.lower()
                and "software" not in cur_name.lower()):
            gpus.append(GpuInfo(
                vendor=_vulkan_vendor(cur_name), name=cur_name,
                total_mib=cur_local_mib or None))
        cur_name, cur_local_mib = None, 0
        is_cpu_dev, heap_local = False, False

    for line in out.splitlines():
        s = line.strip()
        if "deviceName" in s and "=" in s:
            _flush()
            cur_name = s.split("=", 1)[1].strip().strip('"')
        elif cur_name is None:
            continue
        elif "deviceType" in s and "PHYSICAL_DEVICE_TYPE_CPU" in s:
            is_cpu_dev = True
        elif "heapFlags" in s or "DEVICE_LOCAL" in s:
            # Start of a heap block (real vulkaninfo puts heapFlags before
            # size); older dumps put both on one line with the size.
            heap_local = "DEVICE_LOCAL" in s
            if heap_local and "size" in s:
                v = _size_token(s)
                if v:
                    cur_local_mib += v // (1024 * 1024)
        elif "size" in s and "=" in s and heap_local:
            v = _size_token(s)
            if v:
                cur_local_mib += v // (1024 * 1024)
    _flush()
    return gpus


def _amdgpu_vram_fill(gpus: List[GpuInfo]) -> None:
    """Populate free/total VRAM on AMD GpuInfo entries where possible.

    Two portable probes, both best-effort:
      * sysfs ``mem_info_vram_total/used`` (Linux + amdgpu driver — works
        without ROCm userland, including unified-memory APUs);
      * ``rocm-smi --showmeminfo vram --json`` (any OS with ROCm userland).
    Entries that neither probe covers keep free_mib=None (unknown, not zero)."""
    todo = [g for g in gpus if g.vendor == "amd" and g.total_mib is None]
    if not todo:
        return
    # 1. sysfs by card index order (same enumeration order as _amdgpu_sysfs).
    base = Path("/sys/class/drm")
    if base.is_dir():
        cards = sorted(base.glob("card[0-9]"))
        for i, g in enumerate(todo):
            if i >= len(cards) or g.total_mib is not None:
                continue
            dev = cards[i] / "device"
            try:
                tot = int((dev / "mem_info_vram_total").read_text().strip())
                used = int((dev / "mem_info_vram_used").read_text().strip())
                g.total_mib = tot // (1024 * 1024)
                g.free_mib = max(0, (tot - used)) // (1024 * 1024)
            except (OSError, ValueError):
                continue
    if all(g.total_mib is not None for g in todo):
        return
    # 2. rocm-smi JSON: {"card0": {"VRAM Total Memory (B)": ..., "... Used ...": ...}}
    out = _run(["rocm-smi", "--showmeminfo", "vram", "--json"], timeout=15.0)
    if not out:
        return
    try:
        data = json.loads(out)
    except ValueError:
        return
    per_card = []
    for key in sorted(data, key=lambda k: (len(k), k)):
        v = data[key]
        if not isinstance(v, dict):
            continue
        tot_b = v.get("VRAM Total Memory (B)")
        used_b = v.get("VRAM Total Used Memory (B)")
        if tot_b is None:
            continue
        per_card.append((int(tot_b), int(used_b or 0)))
    for i, g in enumerate(todo):
        if g.total_mib is None and i < len(per_card):
            tot_b, used_b = per_card[i]
            g.total_mib = tot_b // (1024 * 1024)
            g.free_mib = max(0, tot_b - used_b) // (1024 * 1024)


def _detect_gpus_impl() -> Tuple[List[GpuInfo], bool, bool, bool]:
    """Enumerate accelerators; returns
    ``(gpus, rocm_userland, nvidia_native, amd_native)``.

    The ROCm flag comes from the SAME rocminfo invocation as the AMD name
    enumeration — detect_runtime used to re-run it (up to 8 s) just to learn
    whether userland exists. ``nvidia_native``/``amd_native`` record which
    vendors a NATIVE probe (nvidia-smi / rocminfo/sysfs) saw: backend choice
    must not treat a Vulkan-enumerated name as CUDA/ROCm evidence, or an
    NVIDIA GPU behind Vulkan-only userland would get a CUDA build it cannot
    compile."""
    gpus: List[GpuInfo] = []
    nvidia_native = False

    # NVIDIA via nvidia-smi (same probe as dashboard/hardware.py).
    out = _run([
        "nvidia-smi",
        "--query-gpu=name,memory.free,memory.total",
        "--format=csv,noheader,nounits",
    ])
    if out:
        for line in out.splitlines():
            cols = [c.strip() for c in line.split(",")]
            if len(cols) >= 3 and cols[1].isdigit() and cols[2].isdigit():
                gpus.append(GpuInfo(
                    vendor="nvidia", name=cols[0],
                    free_mib=int(cols[1]), total_mib=int(cols[2]),
                ))
        nvidia_native = bool(gpus)

    # AMD via rocminfo (only when ROCm userland is installed). Keep only real
    # compute targets (gfxNNNN): rocminfo also prints CPU agents and
    # amdgcn/generic pseudo-devices which are not GPUs.
    rocm_userland = False
    out = _run(["rocminfo"])
    if out is not None:
        rocm_userland = True
        seen = set()
        for line in out.splitlines():
            line = line.strip()
            if not line.startswith("Name:"):
                continue
            name = line.split(":", 1)[1].strip()
            if (
                re.fullmatch(r"gfx[0-9a-f]+(-[a-z0-9]+)*", name)
                and "generic" not in name
                and name not in seen
            ):
                seen.add(name)
                gpus.append(GpuInfo(vendor="amd", name=name))

    # AMD via the kernel driver (sysfs) — works without ROCm userland, which
    # matters on APUs like Strix Halo where rocminfo is often not installed.
    if not any(g.vendor == "amd" for g in gpus):
        gpus += _amdgpu_sysfs()

    # AMD VRAM accounting (sysfs / rocm-smi) — names alone left the doctor
    # tiers and over-commit warnings dark on every AMD box.
    _amdgpu_vram_fill(gpus)
    amd_native = any(g.vendor == "amd" for g in gpus)

    # Apple SoC (integrated GPU; Metal always available on darwin arm64).
    if sys.platform == "darwin":
        has_apple_gpu = platform.machine() == "arm64" or "Apple" in str(
            _run(["sysctl", "-n", "machdep.cpu.brand_string"], timeout=3.0) or ""
        )
        if has_apple_gpu and not gpus:
            gpus.append(GpuInfo(vendor="apple", name=f"Apple Silicon ({platform.machine()})"))

    # Vulkan enumeration as the cross-vendor fallback: makes Intel/Arc (and
    # any GPU without a CUDA/ROCm probe tool) visible on all three OSes. Only
    # consulted when nothing was found, so nvidia/amd entries never duplicate.
    if not gpus and _has_vulkan():
        gpus = _vulkan_gpus()

    return gpus, rocm_userland, nvidia_native, amd_native


def detect_gpus() -> List[GpuInfo]:
    """Enumerate accelerators with cheap vendor probes (never raises)."""
    gpus, _, _, _ = _detect_gpus_impl()
    return gpus


def _amdgpu_sysfs() -> List[GpuInfo]:
    """AMD GPUs visible to the amdgpu driver via /sys/class/drm (never raises).

    Detection only: backend selection still requires ROCm userland (rocminfo)
    or falls through to Vulkan/CPU, so a bare amdgpu box is never handed a
    HIP build it cannot run."""
    out: List[GpuInfo] = []
    base = Path("/sys/class/drm")
    if not base.is_dir():
        return out
    for card in sorted(base.glob("card[0-9]")):
        dev = card / "device"
        try:
            if (dev / "vendor").read_text().strip().lower() != "0x1002":
                continue
        except OSError:
            continue
        name = "AMD GPU (amdgpu)"
        try:
            ue = (dev / "uevent").read_text()
            ids = dict(l.split("=", 1) for l in ue.splitlines() if "=" in l)
            if ids.get("PCI_ID"):
                name = f"AMD GPU [{ids['PCI_ID'].lower()}]"
        except OSError:
            pass
        out.append(GpuInfo(vendor="amd", name=name))
    return out


def _is_strix_halo() -> bool:
    """AMD Strix Halo (Ryzen AI Max 300 series) via /proc/cpuinfo — the board's
    unified memory matters even when rocminfo/gfx1151 naming is unavailable."""
    try:
        txt = Path("/proc/cpuinfo").read_text(errors="ignore").lower()
    except OSError:
        return False
    return "ryzen ai max" in txt or "strix halo" in txt


def _has_vulkan() -> bool:
    if shutil.which("vulkaninfo") or shutil.which("vulkaninfo.exe"):
        return True
    # LunarG SDK layout on Windows.
    if sys.platform == "win32" and os.environ.get("VULKAN_SDK"):
        return True
    return False


def detect_runtime() -> RuntimeInfo:
    """Detect platform + GPUs and choose the llama.cpp backend to use."""
    info = RuntimeInfo(system=platform.system().lower(), machine=platform.machine())
    # One rocminfo invocation shared with GPU enumeration (used to run twice,
    # at up to 8 s each, on every detection call).
    info.gpus, rocm_userland, nvidia_native, amd_native = _detect_gpus_impl()

    vendors = {g.vendor for g in info.gpus}
    # CUDA/ROCm decisions use the NATIVE-probe flags, not vendor names: a
    # Vulkan-enumerated "NVIDIA ..." entry is evidence of a GPU, not of CUDA
    # userland — choosing CUDA there would build against a missing toolkit.
    if nvidia_native:
        info.backend = BACKEND_CUDA
    elif amd_native and rocm_userland:
        # rocminfo answered => ROCm userland exists => HIP build works.
        info.backend = BACKEND_ROCM
    elif "apple" in vendors:
        info.backend = BACKEND_METAL
    elif _has_vulkan():
        info.backend = BACKEND_VULKAN
        if info.gpus:
            info.backend_notes.append(
                "GPU found without CUDA/ROCm userland; using the portable Vulkan backend."
            )
    elif "amd" in vendors:
        # amdgpu kernel driver but no ROCm userland and no vulkaninfo: a HIP
        # build couldn't run, so stay CPU-only but say what we saw.
        info.backend = BACKEND_CPU
        info.backend_notes.append(
            "AMD GPU visible via the amdgpu driver, but no ROCm userland "
            "(rocminfo) found — install ROCm (or Vulkan) for GPU offload."
        )
    else:
        info.backend = BACKEND_CPU
        if sys.platform == "darwin":
            # Intel Macs without Metal-capable GPUs still build CPU-only.
            info.backend_notes.append("No GPU runtime detected; CPU-only serving.")
        else:
            info.backend_notes.append(
                "No supported GPU runtime found; llama.cpp will run on the CPU."
            )
    return info


def gpu_layers(gpu_layers_cfg: Optional[int], info: Optional[RuntimeInfo] = None) -> int:
    """Resolve the -ngl value: explicit config wins, else auto by detection."""
    if gpu_layers_cfg is not None:
        return max(0, int(gpu_layers_cfg))
    info = info or detect_runtime()
    return FULL_OFFLOAD_LAYERS if info.backend != BACKEND_CPU else 0


def launch_args(gpu_layers_cfg: Optional[int],
                info: Optional[RuntimeInfo] = None) -> List[str]:
    """llama-server args derived from the detected hardware (currently -ngl)."""
    ngl = gpu_layers(gpu_layers_cfg, info)
    if ngl <= 0:
        return []  # CPU-only: llama.cpp default (0) is correct.
    return ["-ngl", str(ngl)]


# --------------------------------------------------- unified-memory systems

#: Explicit override: SABLE_UNIFIED_MEMORY=1 forces on, =0 forces off.
UNIFIED_MEMORY_ENV = "GGML_CUDA_ENABLE_UNIFIED_MEMORY"


def _is_jetson() -> bool:
    """NVIDIA Jetson/Tegra SoCs (unified CPU/GPU memory)."""
    if Path("/etc/nv_tegra_release").is_file():
        return True
    try:
        compat = Path("/proc/device-tree/compatible").read_bytes().decode(
            errors="ignore")
        if "nvidia" in compat and ("tegra" in compat or "jetson" in compat):
            return True
    except OSError:
        pass
    return False


def is_unified_memory(info: Optional[RuntimeInfo] = None) -> bool:
    """True on unified-memory systems where GGML_CUDA_ENABLE_UNIFIED_MEMORY=1
    helps — AMD Strix Halo (gfx1151 APU), NVIDIA Jetson/Tegra, and similar
    shared-memory SoCs. Apple Silicon is excluded deliberately: macOS manages
    its own unified memory and llama.cpp's Metal path ignores this variable.

    Override with SABLE_UNIFIED_MEMORY=0/1 (e.g. on an unusual board)."""
    override = os.environ.get("SABLE_UNIFIED_MEMORY", "").strip().lower()
    if override in ("1", "true", "yes", "on"):
        return True
    if override in ("0", "false", "no", "off"):
        return False
    if sys.platform == "darwin":
        return False  # Macs excluded by design (see docstring)
    info = info or detect_runtime()
    # Strix Halo: the gfx1151 APU (also matches its iGPU naming).
    for g in info.gpus:
        name = (g.name or "").lower()
        if "gfx1151" in name or "strix halo" in name or "ryzen ai max+ 395" in name:
            return True
    # sysfs-detected AMD GPU on a Strix Halo board (rocminfo absent, so the
    # gfx naming above never matched).
    if any(g.vendor == "amd" for g in info.gpus) and _is_strix_halo():
        return True
    return _is_jetson()


def unified_memory_env(info: Optional[RuntimeInfo] = None) -> dict:
    """Extra env for llama-server children on unified-memory hardware."""
    if is_unified_memory(info):
        return {UNIFIED_MEMORY_ENV: "1"}
    return {}


# ------------------------------------------------------- llama.cpp locations

def sable_home_dir() -> Path:
    """<SABLE_HOME> (default ~/.sable) — parent of managed llama.cpp installs."""
    home = os.environ.get("SABLE_HOME")
    return Path(home).expanduser() if home else Path.home() / ".sable"


def sable_llama_root(backend: Optional[str] = None) -> Path:
    """Where a sable-managed llama.cpp install lives.

    The default (``backend=None``) is the classic ``<SABLE_HOME>/llama.cpp`` —
    auto-setup's home and whatever backend it first built. Each *additional*
    backend gets its own sibling checkout ``<SABLE_HOME>/llama.cpp-<backend>``
    so installing Vulkan can never overwrite (or be overwritten by) the ROCm
    build, and both can coexist as separate engines."""
    root = sable_home_dir() / "llama.cpp"
    b = str(backend or "").strip().lower()
    return root if not b else root.parent / f"llama.cpp-{b}"


def search_dirs() -> List[Path]:
    """All directories probed for a llama-server build, in precedence order.

    sable is fully standalone: it uses its own managed build
    (<SABLE_HOME>/llama.cpp) or generic system installs (PATH, /opt). It does
    NOT probe foreign installer directories. Users who deliberately want that
    binary can point SABLE_LLAMA_CPP at it explicitly.
    """
    dirs: List[Path] = []
    env_root = os.environ.get("SABLE_LLAMA_CPP")
    if env_root:
        p = Path(env_root).expanduser()
        dirs += [p / "build" / "bin", p / "build" / "bin" / "Release",
                 p / "bin", p / "bin" / "Release", p]
    dirs += [
        sable_llama_root() / "build" / "bin",          # our auto-setup install
        sable_llama_root() / "build" / "bin" / "Release",  # MSVC multi-config
    ]
    # Dedicated per-backend installs (llama.cpp-vulkan etc.): their binaries
    # are discoverable too, after the primary install wins precedence.
    try:
        for rec in list_installs():
            if rec.get("primary"):
                continue  # already covered above
            r = Path(rec["root"])
            dirs += [r / "build" / "bin", r / "build" / "bin" / "Release"]
    except Exception:  # pragma: no cover - defensive listing
        pass
    dirs += [
        Path.home() / ".local" / "bin",
        Path("/usr/local/bin"),
        Path("/usr/bin"),
        Path("/opt/homebrew/bin"),        # Homebrew on Apple Silicon
        Path("/opt/llama.cpp/build/bin"),
    ]
    return dirs


# --------------------------------------------------------- llama.cpp setup

class SetupError(RuntimeError):
    pass


def _require_tool(name: str, hint: str) -> str:
    exe = shutil.which(name)
    if not exe:
        raise SetupError(f"`{name}` is required to build llama.cpp but was not found. {hint}")
    return exe


def _cmake_flags(backend: str) -> List[str]:
    """cmake flags for one backend.

    The non-selected GPU backends are explicitly turned OFF: cmake caches
    -DGGML_* in CMakeCache.txt, so re-configuring an existing build dir that
    was built ROCm with only -DGGML_VULKAN=ON would silently keep HIP enabled
    (the old ROCm build wins and the backend switch appears to do nothing).
    """
    if backend == BACKEND_METAL:
        return []  # Metal is the default macOS build; no flags to pass.
    flags = ["-DGGML_CUDA=OFF", "-DGGML_HIP=OFF", "-DGGML_VULKAN=OFF"]
    if backend == BACKEND_CUDA:
        flags[0] = "-DGGML_CUDA=ON"
    elif backend == BACKEND_ROCM:
        flags[1] = "-DGGML_HIP=ON"
    elif backend == BACKEND_VULKAN:
        flags[2] = "-DGGML_VULKAN=ON"
    else:  # cpu (and anything unknown): native CPU build
        flags.append("-DGGML_NATIVE=ON")
    return flags


#: Provenance file written next to our install (NOT inside the checkout, so a
#: `git clean`/re-clone can never fake it). Records repo/ref/backend so update
#: and doctor know what this build came from.
STATE_FILENAME = "llama_cpp_install.json"


def _state_filename(backend: Optional[str] = None) -> str:
    """llama_cpp_install.json for the primary install, llama_cpp_install-<be>.json
    for a dedicated per-backend slot (each backend keeps its own provenance)."""
    b = str(backend or "").strip().lower()
    return STATE_FILENAME if not b \
        else f"{STATE_FILENAME[:-5]}-{b}.json"


def state_path(root: Optional[Path] = None,
               backend: Optional[str] = None) -> Path:
    home = Path(root).parent if root else sable_home_dir()
    return home / _state_filename(backend)


def read_install_state(root: Optional[Path] = None,
                       backend: Optional[str] = None):
    p = state_path(root, backend)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_install_state(state: dict, root: Optional[Path] = None,
                         backend: Optional[str] = None) -> None:
    p = state_path(root, backend)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def install_root_for(backend: Optional[str]) -> Tuple[Path, Optional[str]]:
    """Route a requested backend to (install dir, state-file backend key).

    The primary slot (<SABLE_HOME>/llama.cpp) is used when it is free or
    already belongs to this backend; any *other* backend gets its own sibling
    checkout llama.cpp-<backend> so installs never overwrite each other."""
    b = str(backend or "").strip().lower() or None
    primary = sable_llama_root()
    if not b:
        return primary, None
    if read_install_state(backend=b) is not None:
        return sable_llama_root(b), b          # dedicated slot in use
    st = read_install_state()
    have = str((st or {}).get("backend") or "").strip().lower()
    if not st or not have or have == b:
        return primary, None                   # primary free or same backend
    return sable_llama_root(b), b


def list_installs() -> List[dict]:
    """Every sable-managed llama.cpp install (primary + per-backend slots)
    with provenance and the current binary path. Pure file probes."""
    out: List[dict] = []
    hd = sable_home_dir()
    stem = STATE_FILENAME[:-5]  # "llama_cpp_install"
    try:
        files = sorted(hd.glob(f"{stem}*.json"))
    except OSError:  # pragma: no cover - unreadable home
        return out
    for p in files:
        try:
            st = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(st, dict):
            continue
        be = p.stem[len(stem) + 1:] if len(p.stem) > len(stem) else None
        root = sable_llama_root(be)
        exe = llama_server_path(root)
        rec = dict(st)
        rec["backend"] = str(st.get("backend") or be or "")
        rec["primary"] = be is None
        rec["root"] = str(root)
        rec["binary"] = str(exe) if exe else None
        out.append(rec)
    return out


def git_head(root: Path) -> Optional[str]:
    """Short HEAD sha of a checkout (None when not a git repo / git missing)."""
    out = _run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], timeout=5.0)
    return out.strip() if out else None


def _vulkan_sdk_root_valid(p: Path) -> bool:
    """A usable Vulkan SDK root for llama.cpp: cmake package config for
    SPIRV-Headers AND the C++ headers ggml-vulkan includes."""
    try:
        return ((p / "share" / "cmake" / "SPIRV-Headers"
                 / "SPIRV-HeadersConfig.cmake").is_file()
                and (p / "include" / "spirv" / "unified1"
                     / "spirv.hpp").is_file())
    except OSError:  # pragma: no cover - unreadable path
        return False


def find_vulkan_sdk() -> Optional[Path]:
    """Locate a Vulkan SDK root whose SPIRV-Headers cmake config actually
    resolves. Guards against a stale VULKAN_SDK env (e.g. pointing at the SDK
    dir without its arch subdir): llama.cpp's ggml-vulkan CMakeLists trusts
    $ENV{VULKAN_SDK}, and a wrong value configures fine but then dies mid-
    build with 'spirv/unified1/spirv.hpp: No such file or directory'."""
    cands: List[Path] = []
    env = os.environ.get("VULKAN_SDK")
    if env:
        p = Path(env).expanduser()
        cands += [p, p.parent]  # tolerate the arch-dir mistake both ways
    try:
        home_vulkan = Path.home() / ".vulkan"
        if home_vulkan.is_dir():
            for ver in sorted(home_vulkan.iterdir(), reverse=True):
                cands.append(ver / "x86_64")
                cands.append(ver)
    except OSError:  # pragma: no cover - unreadable home
        pass
    cands += [Path("/usr"), Path("/usr/local")]
    for c in cands:
        if _vulkan_sdk_root_valid(c):
            return c
    return None


def _vulkan_build_env(log) -> Tuple[dict, List[str]]:
    """(extra env, cmake flags) that pin a *working* Vulkan SDK for the build.

    Returns ({}, []) when nothing usable exists — caller then raises with an
    actionable message. The explicit -DSPIRV-Headers_DIR also overrides any
    stale cached value from a previous misconfigured run."""
    root = find_vulkan_sdk()
    if root is None:
        return {}, []
    env = {"VULKAN_SDK": str(root)}
    flags = [f"-DSPIRV-Headers_DIR={root / 'share' / 'cmake' / 'SPIRV-Headers'}"]
    log(f"using Vulkan SDK at {root}")
    return env, flags


def setup_llama_cpp(
    backend: Optional[str] = None,
    target_dir: Optional[Path] = None,
    log_fn=None,
    repo: Optional[str] = None,
    ref: Optional[str] = None,
    update: bool = False,
) -> Path:
    """Clone + build llama.cpp for the detected (or given) backend.

    ``repo``/``ref`` choose the source (default upstream master; forks like a
    vendor's strix-halo build are first-class). With ``update=True`` an
    existing checkout is fetched and reset to ``ref`` before rebuilding — this
    is sable's llama.cpp update system. When repo changes between runs, the
    remote URL is switched in place.

    Each backend lives in its own checkout: the primary slot
    (<SABLE_HOME>/llama.cpp) hosts the first-installed backend, and any
    additional backend gets ``<SABLE_HOME>/llama.cpp-<backend>`` — so adding
    Vulkan next to an existing ROCm build never touches the working install.

    Returns the directory containing the built binaries. Long-running (many
    minutes); streams progress through ``log_fn(line)`` (default: logger.info).
    Raises SetupError with actionable messages when prerequisites are missing.
    """
    log = log_fn or (lambda line: logger.info("%s", line))
    info = detect_runtime()
    backend = backend or info.backend
    repo = (repo or "").strip() or LLAMA_CPP_REPO
    ref = (ref or "").strip() or LLAMA_CPP_REF

    git = _require_tool("git", "Install git (e.g. `apt install git` / xcode-select / winget).")
    cmake = _require_tool(
        "cmake", "Install cmake (>=3.18): apt/brew/winget, or pip install cmake."
    )

    if target_dir is not None:
        root = Path(target_dir).expanduser()
        state_be: Optional[str] = str(backend)
    else:
        root, state_be = install_root_for(backend)
        if root != sable_llama_root():
            log(f"backend {backend} installs to its own checkout at {root} "
                "(the existing llama.cpp install is left untouched)")
    build_bin = root / "build" / "bin"

    if not (root / "CMakeLists.txt").is_file():
        root.parent.mkdir(parents=True, exist_ok=True)
        log(f"cloning {repo} (ref={ref}) into {root} …")
        r = subprocess.run(
            [git, "clone", "--depth", "1", "--branch", ref, repo, str(root)],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            # --branch fails for some tags on shallow clones; retry full clone.
            log("shallow clone failed; retrying full clone …")
            r = subprocess.run(
                [git, "clone", repo, str(root)], capture_output=True, text=True
            )
            if r.returncode != 0:
                raise SetupError(
                    f"git clone failed (is the network reachable?): "
                    f"{(r.stderr or '').strip()[:400]}"
                )
    else:
        log(f"reusing existing checkout at {root}")

    # Point the checkout at the requested repo (switch remotes when changed).
    cur_remote = _run(["git", "-C", str(root), "remote", "get-url", "origin"], timeout=5.0)
    if cur_remote and cur_remote.strip().rstrip("/") != repo.rstrip("/"):
        log(f"switching origin remote to {repo} …")
        r = subprocess.run(
            [git, "-C", str(root), "remote", "set-url", "origin", repo],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise SetupError(f"could not switch remote to {repo}: {(r.stderr or '').strip()[:300]}")
        update = True  # a repo switch must re-fetch

    # Safety net: this checkout's recorded backend differs from the one being
    # built (possible only via an explicit target_dir). cmake caches -DGGML_*
    # in CMakeCache.txt, so drop the stale build tree or the old backend wins.
    prev = read_install_state(backend=state_be) if state_be is not None \
        else read_install_state()
    if prev is None:
        # Explicit target_dir with no matching slot state: find any managed
        # install record that lives at this exact directory.
        prev = next((rec for rec in list_installs()
                     if Path(str(rec.get("root") or "")) == root), None)
    if (prev and str(prev.get("backend") or "") not in ("", str(backend))
            and (root / "build" / "CMakeCache.txt").is_file()):
        log(f"backend changed ({prev.get('backend')} -> {backend}); "
            "clearing stale cmake build cache …")
        try:
            shutil.rmtree(root / "build")
        except OSError as e:  # pragma: no cover - permissions edge
            log(f"warning: could not remove {root / 'build'}: {e}")

    if update:
        log(f"fetching + resetting to {ref} …")
        for cmd in (
            [git, "-C", str(root), "fetch", "--tags", "--force", "origin",
             f"+{ref}:refs/remotes/origin/{ref}"],
            [git, "-C", str(root), "checkout", str(ref)],
            [git, "-C", str(root), "reset", "--hard", f"origin/{ref}"],
        ):
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                # checkout may fail on a shallow clone missing history; fetch deeper once.
                subprocess.run([git, "-C", str(root), "fetch", "--unshallow"],
                               capture_output=True, text=True)
                r2 = subprocess.run(cmd, capture_output=True, text=True)
                if r2.returncode != 0:
                    step = cmd[3] if len(cmd) > 3 else " ".join(cmd[3:])
                    raise SetupError(
                        f"update failed at git step '{step}': "
                        f"{(r.stderr or r2.stderr or '').strip()[:400]}"
                    )

    # Vulkan: pin a *verified* SDK root for configure AND build. llama.cpp's
    # ggml-vulkan CMake trusts $ENV{VULKAN_SDK}, and a stale/missing value
    # resolves SPIRV-Headers to a dead path — configure passes, then the build
    # dies at 'spirv/unified1/spirv.hpp: No such file or directory'. The
    # explicit -DSPIRV-Headers_DIR also overrides any stale cached lookup.
    child_env = dict(os.environ)
    vulkan_flags: List[str] = []
    if str(backend) == BACKEND_VULKAN:
        venv_extra, vulkan_flags = _vulkan_build_env(log)
        if not venv_extra:
            raise SetupError(
                "Vulkan backend needs a Vulkan SDK with SPIRV-Headers "
                "(spirv/unified1/spirv.hpp + its cmake config). Install one "
                "(e.g. apt install spirv-headers vulkan-sdk, or the LunarG "
                "SDK) and re-run; CPU fallback: `sable setup --backend cpu`.")
        child_env.update(venv_extra)

    log(f"configuring llama.cpp for backend={backend} …")
    cfg_cmd = [cmake, "-S", str(root), "-B", str(root / "build"), "-DCMAKE_BUILD_TYPE=Release"]
    cfg_cmd += _cmake_flags(backend) + vulkan_flags
    r = subprocess.run(cfg_cmd, capture_output=True, text=True, env=child_env)
    if r.returncode != 0:
        raise SetupError(
            f"cmake configure failed for backend={backend}. A working C/C++ compiler "
            f"is required (gcc/clang/MSVC). Tail of cmake output:\n{(r.stderr or r.stdout)[-1200:]}"
        )

    jobs = str(max(1, min(8, (os.cpu_count() or 4))))
    log(f"building llama-server + llama-quantize (-j{jobs}) — this can take a while …")
    build = subprocess.Popen(
        [cmake, "--build", str(root / "build"), "--config", "Release",
         "-j", jobs, "-t", "llama-server", "llama-quantize"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        env=child_env,
    )
    assert build.stdout is not None
    for line in build.stdout:
        log(line.rstrip())
    rc = build.wait()
    if rc != 0:
        extra = ""
        if str(backend) == BACKEND_VULKAN:
            extra = (" For Vulkan, 'spirv/unified1/spirv.hpp not found' means "
                     "the SPIRV-Headers package is missing from the SDK path "
                     "(install spirv-headers or a full LunarG Vulkan SDK).")
        raise SetupError(
            f"llama.cpp build failed (rc={rc}). Backend {backend} needs its SDK "
            "(CUDA toolkit / ROCm / Vulkan SDK); re-run with a different backend "
            f"via `sable setup --backend cpu` as a fallback.{extra}"
        )

    exe = llama_server_path(root)
    if exe is None:
        raise SetupError(f"build finished but no llama-server binary under {build_bin}")
    _write_install_state({
        "repo": repo,
        "ref": ref,
        "backend": backend,
        "commit": git_head(root),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, backend=state_be)
    log(f"llama.cpp ready at {exe} (repo={repo} ref={ref} backend={backend})")
    return build_bin


def llama_server_path(root: Path) -> Optional[Path]:
    # build/bin/Release + bin/Release: MSVC's cmake generator is multi-config
    # and drops binaries into a per-configuration subdir — without these the
    # Windows auto-setup builds fine but discovery reports "not found".
    for cand in (root / "build" / "bin", root / "build" / "bin" / "Release",
                 root / "bin", root / "bin" / "Release", root):
        p = cand / ("llama-server.exe" if os.name == "nt" else "llama-server")
        if p.is_file() and os.access(p, os.X_OK):
            return p
    return None


def remote_ref_commit(repo: str, ref: str, timeout: float = 15.0) -> Optional[str]:
    """Remote HEAD sha for repo@ref (None when unreachable). Network probe."""
    out = _run(["git", "ls-remote", repo, ref], timeout=timeout)
    if not out:
        return None
    line = out.strip().splitlines()[0]
    return line.split("\t")[0][:12] if "\t" in line else None


def check_llama_update(cfg=None, backend: Optional[str] = None) -> dict:
    """Compare one install's recorded commit with the remote ref HEAD.

    ``backend`` selects which managed install to check (None => the primary
    slot; unknown-to-primary backends resolve through install_root_for()).
    Returns {installed: bool, repo, ref, backend, local_commit, remote_commit,
    update_available, reason}. Never raises; network failure =>
    reason explains why.
    """
    root, state_be = install_root_for(backend)
    st = read_install_state(backend=state_be)
    if st is None or not (root / "CMakeLists.txt").is_file():
        return {
            "installed": False,
            "update_available": False,
            "reason": "no sable-managed llama.cpp install (run `sable setup`)",
        }
    repo = st.get("repo") or LLAMA_CPP_REPO
    ref = st.get("ref") or LLAMA_CPP_REF
    local = git_head(root) or st.get("commit")
    res = {
        "installed": True,
        "repo": repo,
        "ref": ref,
        "backend": st.get("backend"),
        "built_at": st.get("built_at"),
        "local_commit": local,
        "remote_commit": None,
        "update_available": False,
        "reason": None,
    }
    remote = remote_ref_commit(repo, ref)
    if remote is None:
        res["reason"] = "could not reach remote (offline?)"
        return res
    res["remote_commit"] = remote
    res["update_available"] = bool(local) and remote[: len(local)] != local
    return res


# ------------------------------------------------ cross-platform process bits

def pid_alive(pid: int) -> Optional[bool]:
    """True/False when determinable; None when not ours to know (permission).

    Windows-safe: os.kill(pid, 0) TERMINATES processes there, so it must never
    be used as a liveness probe on NT.
    """
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not h:
            return None  # gone or not ours to inspect
        try:
            code = ctypes.c_ulong()
            if kernel32.GetExitCodeProcess(h, ctypes.byref(code)):
                return code.value == STILL_ACTIVE
            return True
        finally:
            kernel32.CloseHandle(h)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return None  # alive but not ours


def terminate_pid(pid: int) -> bool:
    """Ask a pid to exit (POSIX SIGTERM; Windows taskkill tree). Best-effort."""
    if os.name == "nt":
        r = subprocess.run(
            ["taskkill", "/PID", str(pid), "/T"],
            capture_output=True, text=True,
        )
        return r.returncode == 0
    try:
        os.kill(pid, signal.SIGTERM)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def terminate_owned_pid(pid: int, grace: float = 8.0) -> str:
    """Verify-then-kill ONE pid under the sable ownership rule.

    WP4 extraction: the exact sequence deploy.stop_owned_server and
    instances.stop_recorded had each inlined — ``pid_alive`` decides:
    True => ours, SIGTERM with SIGKILL escalation after ``grace``;
    False => already gone; None => alive but NOT ours (stale state file /
    recycled pid) — never touched. The engine watchdog reuses this so every
    kill path in the codebase shares one ownership check.

    Returns "killed", "gone" or "foreign". Callers keep their own
    state-file cleanup (this function only ever touches the process)."""
    alive = pid_alive(pid)
    if alive is False:
        return "gone"
    if alive is None:
        return "foreign"
    logger.info("Stopping sable-owned llama-server pid=%s", pid)
    terminate_pid(pid)
    for _ in range(int(grace * 2)):
        time.sleep(0.5)
        if pid_alive(pid) is not True:
            break
    else:
        if os.name != "nt":
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:  # pragma: no cover - windows
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                           capture_output=True)
    return "killed"


def detached_popen_kwargs() -> dict:
    """Popen kwargs for a child that must outlive the CLI on any OS."""
    if os.name == "nt":
        DETACHED_PROCESS = 0x00000008
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        return {"creationflags": DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def uninstall_llama_cpp(log_fn=None, backend: Optional[str] = None) -> bool:
    """Remove one sable-managed llama.cpp install + its state file.

    ``backend`` selects the dedicated slot (llama.cpp-<backend>); None targets
    the primary <SABLE_HOME>/llama.cpp. Only ever touches SABLE_HOME-owned
    checkouts — never foreign or system installs (those are not ours).
    """
    import shutil as _sh

    log = log_fn or (lambda line: logger.info("%s", line))
    root = sable_llama_root(backend)
    state_be = str(backend or "").strip().lower() or None
    removed = False
    if root.is_dir():
        # Safety: refuse anything that doesn't look like our checkout target.
        if not (root / ".git").is_dir() and not (root / "CMakeLists.txt").is_file():
            log(f"refusing to remove {root}: not a llama.cpp checkout")
            return False
        log(f"removing {root} …")
        _sh.rmtree(root)
        removed = True
    try:
        state_path(backend=state_be).unlink()
        removed = True
    except FileNotFoundError:
        pass
    return removed


def auto_setup_enabled(cfg) -> bool:
    """Auto-setup unless env/config opted out."""
    if os.environ.get("SABLE_NO_AUTO_SETUP", "").strip().lower() in ("1", "true", "yes"):
        return False
    return bool(getattr(cfg, "llama_auto_setup", True))


def setup_in_progress() -> bool:
    """Guard against concurrent auto-setup (e.g. dashboard + CLI)."""
    return _setup_lock.locked()


_setup_lock = threading.Lock()

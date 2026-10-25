"""Hardware sampler for the dashboard — zero new dependencies (Phase 5a).

CPU/RAM come from ``/proc``; GPU from a single ``nvidia-smi`` subprocess call
when the binary exists (graceful ``None``s otherwise); disk free via
``shutil.disk_usage`` at SABLE_HOME. No psutil (D-1: pyyaml + httpx only).

Results are cached briefly (default 2 s TTL) so panel polling at ~2 s intervals
costs one sample, and the nvidia-smi subprocess never runs per-request at high
frequency. All functions are best-effort: any failure degrades that field to
None rather than raising.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_TTL_S = 2.0

_NVIDIA_QUERY = "name,memory.free,memory.total,utilization.gpu,temperature.gpu"


def _read_proc_stat() -> Optional[Dict[str, float]]:
    """Aggregate CPU jiffies snapshot (None when /proc is unavailable)."""
    try:
        with open("/proc/stat", encoding="ascii") as f:
            line = f.readline()
    except OSError:
        return None
    parts = line.split()
    if not parts or parts[0] != "cpu":
        return None
    try:
        vals = [float(x) for x in parts[1:]]
    except ValueError:  # pragma: no cover - malformed /proc
        return None
    idle = vals[3] + (vals[4] if len(vals) > 4 else 0.0)  # idle + iowait
    return {"total": sum(vals), "idle": idle}


def cpu_percent(prev: Optional[Dict[str, float]], cur: Dict[str, float]) -> Optional[float]:
    """Busy % between two snapshots (None on division guard)."""
    if prev is None:
        return None
    dt = cur["total"] - prev["total"]
    di = cur["idle"] - prev["idle"]
    if dt <= 0:
        return None
    return round(max(0.0, min(100.0, 100.0 * (dt - di) / dt)), 1)


def memory_info() -> Optional[Dict[str, Any]]:
    """{total_mib, available_mib, used_mib} from /proc/meminfo; None off-Linux."""
    try:
        with open("/proc/meminfo", encoding="ascii") as f:
            fields = {}
            for line in f:
                k, _, v = line.partition(":")
                fields[k.strip()] = v.strip()
    except OSError:
        return None
    try:
        total_kb = int(fields["MemTotal"].split()[0])
        avail_kb = int(fields.get("MemAvailable", "0").split()[0])
    except (KeyError, ValueError, IndexError):  # pragma: no cover
        return None
    used_kb = max(0, total_kb - avail_kb)
    return {
        "total_mib": round(total_kb / 1024),
        "available_mib": round(avail_kb / 1024),
        "used_mib": round(used_kb / 1024),
    }


def amd_gpus() -> Optional[List[Dict[str, Any]]]:
    """AMD GPUs via sysfs (amdgpu driver) — no vendor CLI required.

    Covers desktop RDNA cards and APUs alike (AMD Strix Halo / Ryzen AI Max+
    395 with unified CPU/GPU memory). ``mem_info_vram_*`` exists on amdgpu;
    on unified-memory APUs it can read 0, so the totals fall back to system
    RAM (the GPU really does share it) and usage comes from
    ``gpu_busy_percent``. Temperature best-effort via the card's hwmon."""
    base = Path("/sys/class/drm")
    if not base.is_dir():
        return None
    gpus: List[Dict[str, Any]] = []
    for card in sorted(base.glob("card[0-9]")):
        dev = card / "device"
        try:
            vendor = (dev / "vendor").read_text().strip().lower()
        except OSError:
            continue  # virtual/unbound card
        if vendor != "0x1002":  # PCI vendor id of AMD
            continue

        def _kb_to_mib(p: Path) -> Optional[int]:
            # mem_info_* files report BYTES (docs/ABI say KiB; in practice
            # every amdgpu kernel ships bytes). One //1024 here labeled KiB
            # as MiB — a 1024x inflation, e.g. a 512 MiB carve-out showed
            # as "524288 MiB".
            try:
                return round(int(p.read_text().strip()) // (1024 * 1024))
            except (OSError, ValueError):
                return None

        total_mib = _kb_to_mib(dev / "mem_info_vram_total")
        used_mib = _kb_to_mib(dev / "mem_info_vram_used")
        if used_mib is None:
            used_mib = 0
        name = "AMD GPU"
        try:
            ue = (dev / "uevent").read_text()
            ids = dict(l.split("=", 1) for l in ue.splitlines() if "=" in l)
            if ids.get("PCI_ID"):
                name = f"AMD GPU [{ids['PCI_ID'].lower()}]"
        except OSError:
            pass
        # APU with unified memory: vram_total reads 0 — report system RAM,
        # which is what llama.cpp actually allocates against.
        if not total_mib:
            mem = memory_info()
            total_mib = mem["total_mib"] if mem else None
            used_mib = mem["used_mib"] if mem else None
        # TTM/carve-out setups: vram_total reads a tiny BIOS carve-out
        # (512 MiB on a 128 GiB pool) while the GPU actually allocates
        # against the shared pool. core.gpumem decides (PCI class + GTT
        # dominance), so discrete cards keep their real VRAM numbers.
        unified = None
        try:
            from ..core import gpumem
            unified = gpumem.unified_pool()
        except Exception:  # noqa: BLE001 — display probe failure => carve-out
            pass
        if (unified is not None and unified.card
                and unified.card == card.name):
            total_mib = unified.total_mib
            used_mib = max(0, total_mib - unified.free_mib)
        util: Optional[int] = None
        try:
            util = int((dev / "gpu_busy_percent").read_text().strip())
        except (OSError, ValueError):
            pass
        temp: Optional[int] = None
        try:  # hwmon*/temp1_input (millicelsius) under the card device
            for hm in sorted(dev.glob("hwmon/hwmon*")):
                lbl = hm / "temp1_input"
                if lbl.is_file():
                    temp = round(int(lbl.read_text().strip()) / 1000)
                    break
        except (OSError, ValueError):
            pass
        gpus.append({
            "name": name,
            "free_mib": (max(0, total_mib - (used_mib or 0))
                         if total_mib is not None else None),
            "total_mib": total_mib,
            "util_pct": util,
            "temperature_c": temp,
        })
    return gpus or None


def nvidia_gpus(timeout: float = 6.0) -> Optional[List[Dict[str, Any]]]:
    """GPU list via one nvidia-smi call; None when the tool is absent/fails."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, f"--query-gpu={_NVIDIA_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=timeout,
        )
        if out.returncode != 0:
            return None
    except (OSError, subprocess.SubprocessError):
        return None

    def _int(s: str) -> Optional[int]:
        s = s.strip()
        try:
            return int(s)
        except ValueError:
            return None  # "[N/A]" on some drivers

    gpus: List[Dict[str, Any]] = []
    for line in out.stdout.splitlines():
        if not line.strip():
            continue
        cols = [c.strip() for c in line.split(",")]
        if len(cols) < 5:
            continue
        gpus.append({
            "name": cols[0],
            "free_mib": _int(cols[1]),
            "total_mib": _int(cols[2]),
            "util_pct": _int(cols[3]),
            "temperature_c": _int(cols[4]),
        })
    return gpus or None


def battery_status() -> Optional[Dict[str, Any]]:
    """{ac: bool} from /sys/class/power_supply; None when no battery info."""
    base = Path("/sys/class/power_supply")
    if not base.is_dir():
        return None
    try:
        entries = list(base.iterdir())
    except OSError:
        return None
    has_battery = has_mains = False
    online: Optional[bool] = None
    for e in entries:
        tfile = e / "type"
        if not tfile.is_file():
            continue
        try:
            t = tfile.read_text().strip().lower()
        except OSError:
            continue
        if t == "battery":
            has_battery = True
        elif t == "mains":
            has_mains = True
            try:
                online = (e / "online").read_text().strip() == "1"
            except OSError:
                online = None
    if not (has_battery or has_mains) or online is None:
        return None
    return {"ac": online}


def sample(home: Path, ttl_s: float = DEFAULT_TTL_S,
           _state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Snapshot hardware state; cached for ``ttl_s`` in ``_state``.

    ``_state`` lets callers (DashboardApp) own the cache without module-level
    globals; a dict with keys ``t`` (monotonic), ``snap`` and ``cpu_prev``.
    CPU% needs two /proc samples: the first call returns cpu_pct=None, every
    later call computes against the previous snapshot (which is always kept).
    """
    if _state is None:
        _state = {}
    now = time.monotonic()
    cached = _state.get("snap")
    if cached is not None and now - _state.get("t", 0.0) < ttl_s:
        return cached

    cur = _read_proc_stat()
    pct = cpu_percent(_state.get("cpu_prev"), cur) if cur else None
    if cur is not None:
        _state["cpu_prev"] = cur

    du = None
    try:
        h = Path(home)
        probe = h if h.exists() else h.parent if h.parent.exists() else Path("/")
        usage = shutil.disk_usage(probe)
        du = {"total_mib": round(usage.total / (1024 * 1024)),
              "free_mib": round(usage.free / (1024 * 1024))}
    except OSError:
        pass

    gpus = nvidia_gpus()
    if not gpus:
        # No NVIDIA userland (or hybrid box without the dGPU active): try the
        # kernel driver path so AMD/Intel GPUs show up too.
        gpus = amd_gpus()
    snap = {
        "cpu_pct": pct,
        "mem": memory_info(),
        "gpus": gpus,
        "disk": du,
        "battery": battery_status(),
        "sampled_at": round(now, 3),
    }
    _state["t"] = now
    _state["snap"] = snap
    return snap


class HardwareSampler:
    """Cached sampler bound to one home dir; thread-safe."""

    def __init__(self, home: Path, ttl_s: float = DEFAULT_TTL_S):
        self.home = Path(home)
        self.ttl_s = ttl_s
        self._state: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return sample(self.home, ttl_s=self.ttl_s, _state=self._state)

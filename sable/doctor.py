"""sable doctor — MEASURE the environment, never assume (WP7).

Pattern inspiration: kimi-k3-doctor's "a doctor script that measures rather
than assumes" idea — re-implemented inline, no imports from any reference
project (docs/DECISIONS.md D-12). Stdlib-only like the dashboard/MCP layers:
disk via ``shutil.disk_usage`` (the same statvfs semantics export/training
writes face), RAM via ``/proc/meminfo`` MemAvailable, VRAM via a local
``nvidia-smi`` call when present (absence is reported honestly, never
assumed). Zero network: nothing here downloads or probes non-loopback hosts.

Design for testability: every probe is an injectable callable / fixture text,
so ``tests/test_doctor_wp7.py`` runs the full logic hermetically with tmp
SABLE_HOME and monkeypatched disk/RAM probes — no subprocess needed. The
CLI wrapper (``cli.cmd_doctor``) is thin.

Checks (each returns ok | warn | problem):
  python      interpreter inside the supported range (pyproject: >=3.11,<3.14)
  deps        pyyaml + httpx present; [training] extras detected (info only)
  config      load_config() passes; a ConfigError is printed VERBATIM -> exit 3
  home        SABLE_HOME writable
  disk        free space on the SAME filesystems export/training write to
              (home, training output_dir when set, temp/staging dir), warned
              against the largest registered model + safety margin
  memory      MemAvailable -> recommended serve context / train batch presets
  vram        free VRAM via nvidia-smi when available -> -ngl/context tiers
  binary      llama-server discovery (reuses serving.supervisor.find path)
  ports       configured loopback ports bindable (live sable instances OK)
  state       stale instance state files with dead pids (report; --fix removes
              ONLY files whose pid is deterministically gone per the existing
              ownership rule — never foreign/recycled pids)

Next-command logic: findings map to copy-paste remediations, e.g. no
persisted noise floor -> ``sable bench --noise-floor``; serve.calibration
auto with no stored profile -> ``sable calibrate --quick``; llama-server
missing -> ``sable setup``.

Exit codes (core/exitcodes.py taxonomy): 0 all good (warnings allowed),
1 actionable problems found, 3 invalid configuration.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .core.exitcodes import EXIT_CONFIG_INVALID, EXIT_ERROR, EXIT_OK

# ------------------------------------------------------------------ presets
# Formulas are printed next to their results so every recommendation can be
# re-derived from the measured numbers — no hidden constants in prose.

#: Supported interpreter range (mirrors pyproject requires-python).
PY_MIN = (3, 11)
PY_MAX_EXC = (3, 14)

#: serve context-length tiers by MemAvailable MiB: (min_mib, context).
SERVE_CTX_TIERS: Tuple[Tuple[int, int], ...] = (
    (65536, 32768),
    (32768, 16384),
    (16384, 8192),
    (0, 4096),
)

#: -ngl / context-cap tiers by free-VRAM MiB: (min_mib, ngl, ctx_cap).
VRAM_TIERS: Tuple[Tuple[int, int, int], ...] = (
    (24576, 999, 32768),   # full offload, big context
    (12288, 999, 16384),   # full offload, medium context
    (4096, 24, 8192),      # partial offload (~24 layers), small context
    (0, 0, 4096),          # CPU inference
)

#: train batch-size formula: one full sample per 16 GiB of MemAvailable,
#: capped at 8 (gradient accumulation is the better lever beyond that).
TRAIN_RAM_PER_SAMPLE_MIB = 16384
TRAIN_BATCH_CAP = 8

#: Disk safety margin for the largest-model check: max(2 GiB, 10 % of size).
DISK_MARGIN_FLOOR_MIB = 2048
DISK_MARGIN_PCT = 0.10


@dataclass
class CheckResult:
    """One doctor check outcome. status: ok | warn | problem."""

    name: str
    status: str
    lines: List[str] = field(default_factory=list)
    next_commands: List[str] = field(default_factory=list)


def _fmt_mib(mib: float) -> str:
    if mib >= 1024:
        return f"{mib / 1024:.1f} GiB"
    return f"{int(mib)} MiB"


# ------------------------------------------------------------------- checks

def check_python(version: Optional[Tuple[int, int]] = None) -> CheckResult:
    v = version or sys.version_info[:2]
    if PY_MIN <= v < PY_MAX_EXC:
        return CheckResult("python", "ok", [f"{v[0]}.{v[1]} (supported "
                                           f"{PY_MIN[0]}.{PY_MIN[1]}..{PY_MAX_EXC[0]}.{PY_MAX_EXC[1]-1})"])
    return CheckResult(
        "python", "problem",
        [f"{v[0]}.{v[1]} is outside the supported range "
         f"{PY_MIN[0]}.{PY_MIN[1]}..{PY_MAX_EXC[0]}.{PY_MAX_EXC[1]-1}"],
        next_commands=["run: install Python 3.11-3.13 and recreate the venv, "
                       "then pip install -e ."])


def check_deps(find_spec: Callable[[str], Any] = importlib.util.find_spec
               ) -> CheckResult:
    core = {"pyyaml": "yaml", "httpx": "httpx"}
    missing = [name for name, mod in core.items() if find_spec(mod) is None]
    lines: List[str] = []
    for name, mod in core.items():
        lines.append(f"{name}: {'present' if find_spec(mod) else 'MISSING'}")
    extras = ["torch", "transformers", "peft", "trl"]
    have = [m for m in extras if find_spec(m) is not None]
    lines.append(
        f"[training] extras: {len(have)}/{len(extras)} present"
        + ("" if have else " (train/export jobs need `pip install -e '.[training]'`)"))
    if missing:
        return CheckResult("deps", "problem", lines,
                           next_commands=["run: pip install -e ."])
    return CheckResult("deps", "ok", lines)


def check_config(config_path: Optional[str], *, out: Callable[[str], None],
                 load_config_fn: Optional[Callable[..., Any]] = None
                 ) -> CheckResult:
    """load_config() passthrough: a ConfigError is printed VERBATIM (the
    consolidated report lists EVERY field — fix them all at once)."""
    if load_config_fn is None:
        from .core.config import load_config as load_config_fn  # type: ignore
    try:
        cfg = load_config_fn(config_path)
    except Exception as e:  # noqa: BLE001 - surface whatever the loader reports
        out(str(e))
        return CheckResult("config", "problem",
                           [f"invalid configuration ({type(e).__name__})"],
                           next_commands=["fix every field listed above, then re-run: sable doctor"])
    return CheckResult("config", "ok", ["parses"]) if cfg is not None \
        else CheckResult("config", "ok", [])


def check_home_writable(home: Path) -> CheckResult:
    try:
        home.mkdir(parents=True, exist_ok=True)
        probe = home / ".doctor-write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as e:
        return CheckResult("home", "problem", [f"{home}: not writable ({e})"],
                           next_commands=[f"fix permissions: chmod u+w {home}"])
    return CheckResult("home", "ok", [f"{home} writable"])


# --------------------------------------------------------------------- disk

def export_relevant_paths(cfg: Any, home: Path) -> List[Path]:
    """Filesystems the export/training flow actually writes to: the state
    root (registry artifacts, datasets), the configured training output_dir,
    and the temp dir used for staging (worker merges + atomic renames)."""
    paths: List[Path] = [home]
    tr = getattr(cfg, "training", None) or {}
    out_dir = tr.get("output_dir") if isinstance(tr, dict) else None
    if out_dir:
        paths.append(Path(str(out_dir)).expanduser())
    paths.append(Path(tempfile.gettempdir()))
    # De-duplicate by resolved filesystem location while preserving order.
    seen: Dict[int, Path] = {}
    uniq: List[Path] = []
    for p in paths:
        try:
            key = os.stat(str(p) if p.exists() else str(p.parent)).st_dev
        except OSError:
            continue
        if key not in seen:
            seen[key] = p
            uniq.append(p)
    return uniq


def largest_registered_model_bytes(home: Path) -> Tuple[int, Optional[str]]:
    """Biggest GGUF artifact in the model registry (bytes, model_id).

    Opens the registry read-only; a missing DB / table simply means nothing
    registered yet (0 bytes). Never mutates anything."""
    db = Path(home) / "buffer" / "sable.db"
    if not db.is_file():
        return 0, None
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            rows = conn.execute(
                "SELECT model_id, gguf_path FROM models "
                "WHERE gguf_path IS NOT NULL").fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return 0, None
    best, best_id = 0, None
    for model_id, gguf in rows:
        if not gguf:
            continue
        try:
            from .core.model_discovery import gguf_total_size

            size = gguf_total_size(str(gguf)) or 0
        except Exception:  # noqa: BLE001 - registry hygiene, best effort
            size = 0
        if size > best:
            best, best_id = int(size), str(model_id)
    return best, best_id


def check_disk(paths: Sequence[Path], required_bytes: int,
               largest_label: Optional[str] = None, *,
               disk_usage_fn: Callable[[str], Any] = shutil.disk_usage
               ) -> CheckResult:
    """Free space on every export-relevant filesystem vs the required amount
    (largest registered model + safety margin). Same statvfs semantics
    (shutil.disk_usage) that the write path faces."""
    margin_bytes = int(max(DISK_MARGIN_FLOOR_MIB * 1024 * 1024,
                           required_bytes * DISK_MARGIN_PCT))
    need = required_bytes + margin_bytes
    lines = [f"required for largest model ({largest_label or 'none registered'}) "
             f"+ {DISK_MARGIN_PCT:.0%} margin (min {_fmt_mib(DISK_MARGIN_FLOOR_MIB)}): "
             f"{_fmt_mib(need / (1024 * 1024))}"]
    problems: List[str] = []
    for p in paths:
        try:
            usage = disk_usage_fn(str(p))
        except OSError as e:
            lines.append(f"{p}: UNREADABLE ({e})")
            problems.append(str(p))
            continue
        free_mib = usage.free / (1024 * 1024)
        ok = usage.free >= need
        lines.append(f"{p}: free {_fmt_mib(free_mib)} "
                     f"{'OK' if ok else 'BELOW REQUIREMENT'}")
        if not ok:
            problems.append(str(p))
    if problems:
        return CheckResult(
            "disk", "problem", lines,
            next_commands=["free space or point training.export output_dir at a "
                           "bigger volume, then re-run: sable doctor",
                           "prune old artifacts: sable models janitor"])
    return CheckResult("disk", "ok", lines)


# ------------------------------------------------------------------- memory

def parse_meminfo_available_kib(text: str) -> Optional[int]:
    """MemAvailable (KiB) from /proc/meminfo text; None when absent."""
    for line in text.splitlines():
        k, _, v = line.partition(":")
        if k.strip() == "MemAvailable":
            try:
                return int(v.strip().split()[0])
            except (ValueError, IndexError):
                return None
    return None


def _mem_available_darwin_kib(run: Callable[..., Any] = subprocess.run
                              ) -> Optional[int]:
    """macOS available RAM (KiB) from vm_stat + sysctl hw.page_size.

    'available' approximates Linux MemAvailable: free + inactive +
    speculative pages (reclaimable without pressure). Vendor CLI only — no
    psutil (D-1 stdlib/vendor-CLI rule)."""
    try:
        page = None
        r = run(["sysctl", "-n", "hw.pagesize"], capture_output=True,
                text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip().isdigit():
            page = int(r.stdout.strip())
        if not page:
            return None
        r = run(["vm_stat"], capture_output=True, text=True, timeout=5)
        if r.returncode != 0:
            return None
    except (OSError, subprocess.SubprocessError):
        return None
    counts: Dict[str, int] = {}
    for line in (r.stdout or "").splitlines():
        k, _, v = line.partition(":")
        v = v.strip().rstrip(".")
        if v.isdigit():
            counts[k.strip()] = int(v)
    avail_pages = sum(counts.get(name, 0) for name in
                      ("Pages free", "Pages inactive", "Pages speculative"))
    if not avail_pages and not counts:
        return None
    return avail_pages * page // 1024


def _mem_available_windows_kib() -> Optional[int]:
    """Windows available physical memory (KiB) via GlobalMemoryStatusEx —
    pure ctypes, stdlib-only."""
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(  # type: ignore[attr-defined]
                ctypes.byref(stat)):
            return None
        return int(stat.ullAvailPhys // 1024)
    except Exception:  # noqa: BLE001 — probe failure => unknown, never assume
        return None


def read_meminfo(path: str = "/proc/meminfo") -> Optional[int]:
    """Available RAM in KiB on any OS; None when unmeasurable.

    Linux reads /proc/meminfo (explicit ``path`` stays test-injectable).
    macOS: vm_stat + sysctl pagesize. Windows: GlobalMemoryStatusEx. The
    Linux path is byte-identical to before."""
    try:
        with open(path, encoding="ascii") as f:
            return parse_meminfo_available_kib(f.read())
    except OSError:
        pass
    if sys.platform == "darwin":
        return _mem_available_darwin_kib()
    if sys.platform == "win32":
        return _mem_available_windows_kib()
    return None


def free_vram_mib_via_smi(run: Callable[..., Any] = subprocess.run
                          ) -> Optional[int]:
    """Min free VRAM across GPUs from a local nvidia-smi call; None when the
    binary is absent or unresponsive (reported honestly, never assumed)."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        r = run([exe, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=6)
        if r.returncode != 0:
            return None
        vals = [int(float(x)) for x in r.stdout.split() if x.strip()]
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return min(vals) if vals else None


def free_vram_mib() -> Optional[int]:
    """Min free VRAM across GPUs, vendor-chained: nvidia-smi first, then the
    scheduler's AMD probe (rocm-smi). Keeps the doctor honest on AMD boxes
    instead of going dark off-NVIDIA. None when no telemetry exists."""
    vram = free_vram_mib_via_smi()
    if vram is not None:
        return vram
    try:
        from .training.scheduler import _rocm_free_mib

        return _rocm_free_mib()
    except Exception:  # noqa: BLE001 — probe failure => unknown, skip tiers
        return None


def presets_from_measurements(available_ram_mib: Optional[int],
                              free_vram_mib: Optional[int]) -> Dict[str, Any]:
    """Recommended config presets computed from measured numbers.

    Formulas (also printed by the doctor so each number is re-derivable):
      serve context  = tier(MemAvailable)  [>=64GiB→32768, >=32→16384,
                       >=16→8192, else 4096], capped by the free-VRAM tier cap
                       when VRAM telemetry exists.
      serve -ngl     = VRAM tier: >=24GiB→999, >=12→999, >=4→24 (partial),
                       else 0 (CPU).
      train batch    = clamp(MemAvailable // 16 GiB, 1, 8)."""
    ctx = None
    if available_ram_mib is not None:
        for min_mib, c in SERVE_CTX_TIERS:
            if available_ram_mib >= min_mib:
                ctx = c
                break
    ngl, ctx_cap = None, None
    if free_vram_mib is not None:
        for min_mib, n, cap in VRAM_TIERS:
            if free_vram_mib >= min_mib:
                ngl, ctx_cap = n, cap
                break
    if ctx is not None and ctx_cap is not None:
        ctx = min(ctx, ctx_cap)
    batch = None
    if available_ram_mib is not None:
        batch = max(1, min(TRAIN_BATCH_CAP,
                           available_ram_mib // TRAIN_RAM_PER_SAMPLE_MIB))
    lines: List[str] = []
    if available_ram_mib is None:
        lines.append("memory: MemAvailable unavailable (/proc/meminfo not "
                     "readable) — RAM-based presets skipped")
    else:
        lines.append(
            f"serve context: {_fmt_mib(available_ram_mib)} MemAvailable -> "
            f"-c {ctx} (tier rule: >=64GiB→32768, >=32GiB→16384, "
            f">=16GiB→8192, else 4096)")
        lines.append(
            f"train batch size: {_fmt_mib(available_ram_mib)} // "
            f"{_fmt_mib(TRAIN_RAM_PER_SAMPLE_MIB)} per sample -> "
            f"batch_size {batch} (capped at {TRAIN_BATCH_CAP})")
    if free_vram_mib is None:
        lines.append("vram: unavailable (no nvidia-smi) — -ngl tier skipped; "
                     "auto-detect stays in charge")
    else:
        lines.append(
            f"serve offload: {_fmt_mib(free_vram_mib)} free VRAM -> -ngl {ngl}, "
            f"context cap {ctx_cap} (tier rule: >=24GiB→999/32768, "
            f">=12GiB→999/16384, >=4GiB→24/8192, else 0/4096)")
    return {"serve_context": ctx, "serve_ngl": ngl, "train_batch_size": batch,
            "available_ram_mib": available_ram_mib,
            "free_vram_mib": free_vram_mib, "lines": lines}


def check_memory(available_ram_mib: Optional[int],
                 free_vram_mib: Optional[int]) -> CheckResult:
    p = presets_from_measurements(available_ram_mib, free_vram_mib)
    lines = [] if available_ram_mib is None else \
        [f"MemAvailable: {_fmt_mib(available_ram_mib)}"]
    lines += list(p["lines"])
    status = "warn" if (available_ram_mib is None and free_vram_mib is None) \
        else "ok"
    return CheckResult("memory", status, lines)


# -------------------------------------------------------------- llama binary

def check_llama_binary(binary_probe: Callable[[], str]) -> CheckResult:
    """binary_probe() returns a path or raises (ServeError)."""
    try:
        binary = binary_probe()
    except Exception as e:  # noqa: BLE001 - ServeError or anything discovery does
        return CheckResult("llama-server", "problem", [f"NOT FOUND — {e}"],
                           next_commands=["run: sable setup"])
    return CheckResult("llama-server", "ok", [str(binary)])


# -------------------------------------------------------------------- ports

def port_bindable(port: int, host: str = "127.0.0.1") -> bool:
    """True when nothing currently holds the loopback port."""
    if not port:
        return True
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((host, int(port)))
        return True
    except OSError:
        return False
    finally:
        s.close()


def check_ports(ports: Sequence[int], live_owned_ports: Sequence[int] = (), *,
                bind_fn: Callable[[int], bool] = port_bindable) -> CheckResult:
    """Configured (static) ports must be bindable unless a live sable-
    managed instance currently owns them (then it is expected, not a conflict)."""
    lines: List[str] = []
    problems: List[int] = []
    owned = set(live_owned_ports)
    for p in sorted(set(int(x) for x in ports if x)):
        if bind_fn(p):
            lines.append(f"port {p}: free")
        elif p in owned:
            lines.append(f"port {p}: busy — held by a live sable instance (OK)")
        else:
            lines.append(f"port {p}: BUSY and not owned by a live sable "
                         f"instance — launch would fail")
            problems.append(p)
    if problems:
        return CheckResult(
            "ports", "problem", lines,
            next_commands=["free the port (stop the foreign process) or use a "
                           "dynamic port: sable serve --port 0"])
    return CheckResult("ports", "ok", lines or ["no static ports configured"])


# --------------------------------------------------------------- stale state

def find_stale_states(home: Path, *,
                      alive_fn: Callable[[int], Optional[bool]],
                      ) -> List[Tuple[Path, int]]:
    """Instance state files under <home>/serving/state/ whose recorded pid is
    deterministically dead. Attach-mode states (no pid) and pids that are
    alive-but-not-ours (pid_alive => None, the ownership rule) are NEVER
    listed — they must not be deleted."""
    from .serving.instances import state_dir

    out: List[Tuple[Path, int]] = []
    d = state_dir(Path(home))
    if not d.is_dir():
        return out
    for f in sorted(d.glob("*.json")):
        try:
            st = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(st, dict) or st.get("mode") == "attach":
            continue
        pid = st.get("pid")
        if not pid:
            continue
        try:
            pid_i = int(pid)
        except (TypeError, ValueError):
            continue
        if alive_fn(pid_i) is False:  # deterministically gone only
            out.append((f, pid_i))
    return out


def fix_stale_states(stale: Sequence[Tuple[Path, int]]) -> List[str]:
    """--fix path: remove the listed stale state files. Callers must pass
    ONLY entries from find_stale_states (already ownership-filtered)."""
    removed: List[str] = []
    for p, _pid in stale:
        try:
            p.unlink()
            removed.append(str(p))
        except OSError:
            pass
    return removed


def check_stale_state(home: Path, *, alive_fn: Callable[[int], Optional[bool]],
                      fix: bool = False) -> CheckResult:
    stale = find_stale_states(home, alive_fn=alive_fn)
    if not stale:
        return CheckResult("state", "ok", ["no stale instance state files"])
    lines = [f"{p} (pid {pid} dead)" for p, pid in stale]
    if fix:
        removed = fix_stale_states(stale)
        lines.append(f"--fix: removed {len(removed)} state file(s) "
                     "(only pids verified gone; foreign pids untouched)")
        return CheckResult("state", "ok" if len(removed) == len(stale) else "warn",
                           lines)
    cleanup = " ".join(str(p) for p, _ in stale)
    return CheckResult("state", "problem", lines,
                       next_commands=[f"review then remove: rm {cleanup}",
                                      "or let doctor do it: sable doctor --fix"])


# ------------------------------------------------------- measurement states

def check_noise_floor(home: Path, *,
                      load_fn: Optional[Callable[..., Optional[Dict[str, Any]]]] = None,
                      ) -> CheckResult:
    if load_fn is None:
        from .serving.bench import load_noise_floor as load_fn  # type: ignore
    rec = load_fn(home=home)
    if rec is not None:
        return CheckResult("noise-floor", "ok",
                           [f"persisted baseline present ({rec.get('label') or 'unlabeled'})"])
    return CheckResult(
        "noise-floor", "warn",
        ["no persisted noise-floor baseline — effect-vs-noise judgments "
         "fall back to the 3% default threshold"],
        next_commands=["run: sable bench --noise-floor"])


def check_calibration_profile(cfg: Any, home: Path, *,
                              list_fn: Optional[Callable[..., List[Dict[str, Any]]]] = None,
                              mode_fn: Optional[Callable[[Any], str]] = None,
                              ) -> CheckResult:
    if mode_fn is None or list_fn is None:
        from .serving.calibrate import calibration_mode as mode_fn  # type: ignore
        from .serving.calibrate import list_profiles as list_fn  # type: ignore
    mode = mode_fn(cfg)
    if mode != "auto":
        return CheckResult("calibration", "ok",
                           [f"serve.calibration: {mode} (profile not needed)"])
    profiles = list_fn(home=home)
    if profiles:
        return CheckResult("calibration", "ok",
                           [f"{len(profiles)} stored profile(s) present"])
    return CheckResult(
        "calibration", "warn",
        ["serve.calibration: auto but no stored calibration profile — serve "
         "runs with uncalibrated defaults"],
        next_commands=["run: sable calibrate --quick"])


# ------------------------------------------------------------------ runner

def collect_checks(*, config_path: Optional[str] = None, fix: bool = False,
                   out: Callable[[str], None] = print,
                   load_config_fn: Optional[Callable[..., Any]] = None,
                   disk_usage_fn: Callable[..., Any] = shutil.disk_usage,
                   meminfo_kib: Optional[int] = None,
                   vram_mib: Optional[int] = None,
                   binary_probe: Optional[Callable[[], str]] = None,
                   alive_fn: Optional[Callable[[int], Optional[bool]]] = None,
                   ) -> Tuple[List[CheckResult], int]:
    """Run every measurement check; return (results, exit_code).

    Exit code per core/exitcodes.py: 0 ok (warnings allowed), 1 actionable
    problems found, 3 invalid configuration. Pure-ish: every environment
    probe is injectable so tests run hermetically without subprocesses."""
    # Default probes wire to the real environment only when not injected.
    if meminfo_kib is None:
        meminfo_kib = read_meminfo()
    if vram_mib is None:
        # Vendor-chained (nvidia-smi -> rocm-smi): AMD boxes report numbers
        # instead of going dark off-NVIDIA.
        vram_mib = free_vram_mib()
    if alive_fn is None:
        from .serving.runtime import pid_alive as alive_fn  # type: ignore

    results: List[CheckResult] = []
    results.append(check_python())
    results.append(check_deps())

    cfg_res = check_config(config_path, out=out, load_config_fn=load_config_fn)
    if cfg_res.status == "problem":
        results.append(cfg_res)
        return results, EXIT_CONFIG_INVALID
    # Re-load for the config object (check_config already proved it parses;
    # caching would risk stale env interactions — a second parse is cheap).
    cfg = (load_config_fn or _default_load_config())(config_path)
    results.append(cfg_res)

    from .store.paths import resolve_sable_home

    home = Path(resolve_sable_home(cfg))
    results.append(check_home_writable(home))

    # Disk the way export reads it: same filesystems, largest model + margin.
    need_bytes, need_id = largest_registered_model_bytes(home)
    results.append(check_disk(export_relevant_paths(cfg, home), need_bytes,
                              need_id, disk_usage_fn=disk_usage_fn))

    ram_mib = int(meminfo_kib // 1024) if meminfo_kib is not None else None
    results.append(check_memory(ram_mib, vram_mib))

    if binary_probe is not None:
        probe = binary_probe
    else:
        from .cli import runtime_find_binary

        def _default_probe(_cfg=cfg):  # noqa: ANN202 - closure over cfg
            return runtime_find_binary(_cfg)
        probe = _default_probe

    results.append(check_llama_binary(probe))

    # Ports: configured static port + every managed instance's port; live
    # sable states excuse a busy port.
    from .serving.instances import load_instances, read_state, state_alive

    ports: List[int] = []
    live_owned: List[int] = []
    if getattr(cfg, "port", 0):
        ports.append(int(cfg.port))
    for inst in load_instances(home, cfg=cfg):
        iid = str(inst.get("id"))
        p = inst.get("port")
        if p:
            ports.append(int(p))
        st = read_state(home, iid)
        if st and st.get("port") and state_alive(st):
            live_owned.append(int(st["port"]))
    results.append(check_ports(ports, live_owned))

    results.append(check_stale_state(home, alive_fn=alive_fn, fix=fix))
    results.append(check_noise_floor(home))
    results.append(check_calibration_profile(cfg, home))

    if any(r.status == "problem" for r in results):
        return results, EXIT_ERROR
    return results, EXIT_OK


def run_doctor(*, config_path: Optional[str] = None, fix: bool = False,
               out: Callable[[str], None] = print,
               **injections: Any) -> int:
    """Print the full report; returns the exit code (see collect_checks)."""
    results, rc = collect_checks(config_path=config_path, fix=fix, out=out,
                                 **injections)
    _print_report(results, out)
    return rc


def main(argv: Optional[Sequence[str]] = None) -> int:
    """`python -m sable.doctor` entry (the CLI `sable doctor --env` path
    calls run_doctor directly)."""
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m sable.doctor",
        description="Measure the environment (disk/RAM/VRAM/config/state) "
                    "and print recommended presets + next commands.")
    p.add_argument("--config", help="Path to sable YAML config file")
    p.add_argument("--fix", action="store_true",
                   help="Remove stale instance state files whose pid is "
                        "verified gone (ownership rule respected; foreign "
                        "pids are never touched)")
    args = p.parse_args(argv)
    return run_doctor(config_path=args.config, fix=args.fix)


def _default_load_config():
    from .core.config import load_config

    return load_config


def _print_report(results: Sequence[CheckResult],
                  out: Callable[[str], None]) -> None:
    marks = {"ok": "OK  ", "warn": "WARN", "problem": "FAIL"}
    out("sable doctor — measured environment report")
    out("=" * 44)
    for r in results:
        out(f"[{marks.get(r.status, '?')}] {r.name}")
        for line in r.lines:
            out(f"       {line}")
    nexts: List[str] = []
    for r in results:
        nexts.extend(r.next_commands)
    if nexts:
        out("")
        out("next commands:")
        for c in nexts:
            out(f"  $ {c}")


if __name__ == "__main__":
    raise SystemExit(main())

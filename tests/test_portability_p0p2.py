"""Portability-audit regressions (P0-P2): GPU pinning, per-device VRAM gate,
Vulkan/AMD enumeration, macOS watchdog CPU-only verdicts, doctor RAM probes.

Everything here is hermetic: no GPU, no subprocesses beyond injected fakes,
tmp SABLE_HOME via conftest. The cardinal rule under test throughout:
no config changes => byte-identical behavior (defaults untouched).
"""

from __future__ import annotations

import pytest

from sable.serving import runtime
from sable.serving.instances import (
    ServeError,
    _normalize_gpu_id,
    instance_device_env,
)
from sable.serving.supervisor import _gpu_pin_args


# ----------------------------------------------------- instance gpu_id pinning

def test_gpu_id_absent_means_no_env_and_no_args():
    """No pin => spawn environment and command line are untouched (default)."""
    inst = {"id": "a", "gguf": "x.gguf"}
    assert instance_device_env(inst) == {}

    class _RT:
        backend = runtime.BACKEND_CUDA

    assert _gpu_pin_args(inst, {}, _RT()) == []


def test_gpu_id_int_sets_visible_devices():
    env = instance_device_env({"gpu_id": 1})
    assert env == {"CUDA_VISIBLE_DEVICES": "1", "HIP_VISIBLE_DEVICES": "1"}


def test_gpu_id_list_joins_comma():
    env = instance_device_env({"gpu_id": [0, 1]})
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert env["HIP_VISIBLE_DEVICES"] == "0,1"


def test_gpu_id_normalization_rules():
    assert _normalize_gpu_id(None) is None
    assert _normalize_gpu_id("") is None
    assert _normalize_gpu_id([]) is None
    assert _normalize_gpu_id(2) == 2
    assert _normalize_gpu_id("1") == 1          # numeric string from forms
    assert _normalize_gpu_id([0, 1]) == [0, 1]
    for bad in (-1, "x", True, [True], ["a"], 1.5):
        with pytest.raises(ServeError):
            _normalize_gpu_id(bad)


def test_gpu_pin_args_vulkan_uses_device_flags():
    class _RT:
        backend = runtime.BACKEND_VULKAN

    assert _gpu_pin_args({"gpu_id": [0, 1]}, {}, _RT()) == [
        "--device", "0", "--device", "1"]
    assert _gpu_pin_args({"gpu_id": 2}, {}, _RT()) == ["--device", "2"]


def test_gpu_pin_args_tensor_parallel_needs_layer_split():
    class _RT:
        backend = runtime.BACKEND_CUDA

    # Pinned multi-GPU + tensor_parallel => llama.cpp only splits when told.
    assert _gpu_pin_args({"gpu_id": [0, 1]}, {"tensor_parallel": True}, _RT()) == [
        "--split-mode", "layer"]
    # Single device or no TP => nothing (CUDA pinning rides the env instead).
    assert _gpu_pin_args({"gpu_id": 1}, {"tensor_parallel": True}, _RT()) == []
    assert _gpu_pin_args({"gpu_id": [0, 1]}, {}, _RT()) == []


# ------------------------------------------------- per-device VRAM gate (C2)

def test_resource_gate_device_none_keeps_zero_arg_probe(monkeypatch):
    """Default path: gpu_free_mib() is called with NO arguments — stubs that
    predate the device parameter keep working, and behavior is unchanged."""
    from sable.training.scheduler import TrainingConfig, resource_gate_ok

    calls = []

    def fake_free():
        calls.append(None)
        return 10_000

    monkeypatch.setattr("sable.training.scheduler.gpu_free_mib", fake_free)
    assert resource_gate_ok(TrainingConfig(min_free_vram_mib=5000))["ok"]
    assert calls == [None]


def test_resource_gate_passes_device_when_pinned(monkeypatch):
    from sable.training.scheduler import TrainingConfig, resource_gate_ok

    seen = []

    def fake_free(device=None):
        seen.append(device)
        return 10_000 if device == "1" else 100

    monkeypatch.setattr("sable.training.scheduler.gpu_free_mib", fake_free)
    # Device 1 has room => admits even though GPU 0 (global min) is full.
    ok = resource_gate_ok(TrainingConfig(min_free_vram_mib=5000), device="1")
    assert ok["ok"] and seen == ["1"]


def test_scheduler_gate_device_none_without_gpu_ids():
    """Unpinned config (the default) => the global-minimum probe, unchanged."""
    from sable.training.scheduler import Scheduler, TrainingConfig

    sched = Scheduler.__new__(Scheduler)
    sched.cfg = TrainingConfig()
    assert sched._gate_device(0) is None

    sched.cfg = TrainingConfig(gpu_ids=[1, 3])
    assert sched._gate_device(0) == "1"
    assert sched._gate_device(1) == "3"


def test_nvidia_free_mib_per_device(monkeypatch):
    import sable.training.scheduler as sched

    csv = "0, 24000\n1, 512\n"

    class R:
        returncode = 0
        stdout = csv

    monkeypatch.setattr(sched.shutil, "which", lambda n: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(sched.subprocess, "run", lambda *a, **k: R())
    assert sched._nvidia_free_mib("1") == 512
    assert sched._nvidia_free_mib("9") is None   # absent device => unknown


def test_rocm_free_mib_per_device(monkeypatch):
    import json as _json

    import sable.training.scheduler as sched

    payload = {"card0": {"VRAM Total Memory (B)": 16 * 2**30,
                         "VRAM Total Used Memory (B)": 8 * 2**30},
               "card1": {"VRAM Total Memory (B)": 16 * 2**30,
                         "VRAM Total Used Memory (B)": 15 * 2**30}}

    class R:
        returncode = 0
        stdout = _json.dumps(payload)

    monkeypatch.setattr(sched.shutil, "which", lambda n: "/usr/bin/rocm-smi")
    monkeypatch.setattr(sched.subprocess, "run", lambda *a, **k: R())
    assert sched._rocm_free_mib("1") == 2**10      # 1 GiB free on card1
    assert sched._rocm_free_mib() == 2**10         # global min
    assert sched._rocm_free_mib("7") is None       # absent => never guess


# ----------------------------------------------- Vulkan / AMD enumeration C1

def test_vulkan_gpus_parses_summary_and_skips_software(monkeypatch):
    summary = """
GPU0:
        deviceName         = NVIDIA GeForce RTX 4090
        deviceType         = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
                heapFlags    = (DEVICE_LOCAL_BIT)
                size         = 0x0000003de0000000
GPU1:
        deviceName         = llvmpipe (LLVM 15.0, 256 bits)
        deviceType         = PHYSICAL_DEVICE_TYPE_CPU
                heapFlags    = (DEVICE_LOCAL_BIT)
                size         = 0x0000003de0000000
"""
    monkeypatch.setattr(runtime, "_run", lambda cmd, timeout=8.0: summary)
    gpus = runtime._vulkan_gpus()
    assert len(gpus) == 1                     # llvmpipe skipped (CPU device)
    assert gpus[0].vendor == "nvidia"
    assert gpus[0].total_mib and gpus[0].total_mib > 20_000


def test_vulkan_vendor_classification():
    assert runtime._vulkan_vendor("Intel Arc A770") == "intel"
    assert runtime._vulkan_vendor("AMD Radeon RX 7900 XTX") == "amd"
    assert runtime._vulkan_vendor("NVIDIA GeForce RTX 3060") == "nvidia"
    assert runtime._vulkan_vendor("Apple M3") == "apple"
    assert runtime._vulkan_vendor("Some Unknown GPU") == "other"


def test_detect_gpus_falls_back_to_vulkan_when_nothing_else(monkeypatch):
    monkeypatch.setattr(runtime, "_run",
                        lambda cmd, timeout=8.0: None)   # no nvidia/rocm
    monkeypatch.setattr(runtime, "_amdgpu_sysfs", lambda: [])
    monkeypatch.setattr(runtime, "_has_vulkan", lambda: True)
    monkeypatch.setattr(runtime.sys, "platform", "linux")  # skip darwin probe
    monkeypatch.setattr(
        runtime, "_vulkan_gpus",
        lambda: [runtime.GpuInfo(vendor="intel", name="Intel Arc A750")])
    gpus = runtime.detect_gpus()
    assert [g.name for g in gpus] == ["Intel Arc A750"]


def test_amdgpu_vram_fill_from_sysfs(monkeypatch, tmp_path):
    """sysfs mem_info_vram_* populates free/total on AMD entries."""
    # The unified-pool override runs first; this test is about the sysfs
    # fill path, and the *real* /sys on this box may be a TTM/APU setup.
    from sable.core import gpumem as _gm
    monkeypatch.setattr(_gm, "unified_pool", lambda **kw: None)
    card = tmp_path / "card0" / "device"
    card.mkdir(parents=True)
    (card / "mem_info_vram_total").write_text(str(16 * 2**30))
    (card / "mem_info_vram_used").write_text(str(4 * 2**30))

    import pathlib as _pl

    real_path = _pl.Path

    class FakeBase:
        @staticmethod
        def is_dir():
            return True

        @staticmethod
        def glob(pat):
            return sorted((tmp_path).glob("card[0-9]"))

    monkeypatch.setattr(runtime, "Path", lambda p: (
        FakeBase() if str(p) == "/sys/class/drm" else real_path(p)))
    g = runtime.GpuInfo(vendor="amd", name="gfx1100")
    runtime._amdgpu_vram_fill([g])
    assert g.total_mib == 16 * 1024
    assert g.free_mib == 12 * 1024


# ------------------------------------------------- watchdog macOS CPU-only B1

def test_freeze_verdict_cpu_only_path():
    from sable.serving.watchdog import (
        V_BUSY,
        V_STALLED,
        ProcSample,
        WatchdogConfig,
        freeze_verdict,
    )

    cfg = WatchdogConfig(window_seconds=90)
    prev = ProcSample(t=0.0, cpu_s=10.0, io_bytes=0, io_source="cpu_only")
    busy = ProcSample(t=5.0, cpu_s=10.5, io_bytes=0, io_source="cpu_only")
    idle = ProcSample(t=5.0, cpu_s=10.0, io_bytes=0, io_source="cpu_only")
    assert freeze_verdict(prev, busy, cfg, inflight_age=30) == V_BUSY
    assert freeze_verdict(prev, idle, cfg, inflight_age=30) == V_STALLED
    # Nothing in flight => never a freeze regardless of CPU.
    assert freeze_verdict(prev, idle, cfg, inflight_age=0) == "idle"


def test_ps_cpu_seconds_parsing(monkeypatch):
    import sable.serving.watchdog as wd

    class R:
        returncode = 0

        def __init__(self, out):
            self.stdout = out

    monkeypatch.setattr(wd.shutil, "which", lambda n: "/bin/ps")
    monkeypatch.setattr(wd.subprocess, "run", lambda *a, **k: R(" 01:23:45\n"))
    assert wd._ps_cpu_seconds(1) == 1 * 3600 + 23 * 60 + 45
    monkeypatch.setattr(wd.subprocess, "run", lambda *a, **k: R(" 00:02.50\n"))
    assert wd._ps_cpu_seconds(1) == pytest.approx(2.50)
    # Empty output (pid gone) => None.
    monkeypatch.setattr(wd.subprocess, "run", lambda *a, **k: R(""))
    assert wd._ps_cpu_seconds(1) is None


# ------------------------------------------------------- doctor RAM fallbacks B2

def test_read_meminfo_linux_path_unchanged(tmp_path):
    from sable import doctor

    p = tmp_path / "meminfo"
    p.write_text("MemTotal: 1000000 kB\nMemAvailable: 500000 kB\n")
    assert doctor.read_meminfo(str(p)) == 500_000


def test_read_meminfo_darwin(monkeypatch):
    from sable import doctor

    monkeypatch.setattr(doctor.sys, "platform", "darwin")

    class R:
        returncode = 0

        def __init__(self, out):
            self.stdout = out

    def fake_run(cmd, **kw):
        if "hw.pagesize" in cmd[-1]:
            return R("4096\n")
        return R("""Mach Virtual Memory Statistics:
Pages free:                             65536.
Pages inactive:                        262144.
Pages speculative:                       8192.
""")

    # Pass the probe explicitly: the default arg binds subprocess.run at
    # import time, so monkeypatching the module attribute is too late.
    kib = doctor._mem_available_darwin_kib(run=fake_run)
    # (65536 + 262144 + 8192) pages * 4 KiB/page
    assert kib == (65536 + 262144 + 8192) * 4


def test_free_vram_mib_chains_to_rocm(monkeypatch):
    from sable import doctor

    # This box may itself BE a unified-pool machine; the chain under test
    # starts at the vendor probes.
    from sable.core import gpumem as _gm
    monkeypatch.setattr(_gm, "unified_pool", lambda **kw: None)
    monkeypatch.setattr(doctor, "free_vram_mib_via_smi", lambda: None)
    monkeypatch.setattr("sable.training.scheduler._rocm_free_mib",
                        lambda device=None: 8192)
    assert doctor.free_vram_mib() == 8192

    monkeypatch.setattr("sable.training.scheduler._rocm_free_mib",
                        lambda device=None: None)
    assert doctor.free_vram_mib() is None


# ------------------------------------------------- llama.cpp discovery paths B3

def test_llama_server_path_finds_msvc_release_dir(tmp_path):
    """MSVC's multi-config generator puts binaries under build/bin/Release."""
    rel = tmp_path / "build" / "bin" / "Release"
    rel.mkdir(parents=True)
    exe = rel / ("llama-server.exe" if __import__("os").name == "nt"
                 else "llama-server")
    exe.write_text("")
    exe.chmod(0o755)
    found = runtime.llama_server_path(tmp_path)
    assert found is not None and str(rel) in str(found)


def test_search_dirs_include_homebrew_and_release(tmp_path, monkeypatch):
    monkeypatch.setenv("SABLE_LLAMA_CPP", str(tmp_path / "llama.cpp"))
    dirs = runtime.search_dirs()
    assert tmp_path / "llama.cpp" / "build" / "bin" / "Release" in dirs
    assert any("homebrew" in str(d) for d in dirs)

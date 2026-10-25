"""Unified GPU-memory budget probe (core/gpumem.py) + its gate/doctor wiring.

The failure this family guards against: a 128 GiB unified-memory Strix Halo
(BIOS carve-out 512 MiB, amdgpu.ttm pool 120 GiB) where every vendor probe
lies — rocm-smi/sysfs "VRAM" says 44 MiB free (the carve-out), torch/HIP
says 128 GiB free (reclaimable GTT). The scheduler gate refused training
with "44 MiB free" while the machine had 7 GiB genuinely available.
"""

from __future__ import annotations

import json  # noqa: F401  (kept for payload-style tests if extended)
from pathlib import Path

from sable.core import gpumem
from sable.core.gpumem import unified_pool

MIB = 1024 * 1024
GIB = 1024 * MIB


def _fake_drm(root: Path, *, vram_total: int, vram_used: int,
              gtt_total: int, gtt_used: int, pci_class: str,
              vendor: str = "0x1002") -> Path:
    dev = root / "card1" / "device"
    dev.mkdir(parents=True)
    (dev / "vendor").write_text(vendor + "\n")
    (dev / "class").write_text(pci_class + "\n")
    (dev / "mem_info_vram_total").write_text(str(vram_total))
    (dev / "mem_info_vram_used").write_text(str(vram_used))
    (dev / "mem_info_gtt_total").write_text(str(gtt_total))
    (dev / "mem_info_gtt_used").write_text(str(gtt_used))
    return root


MEMINFO = {"MemTotal": 128 * 1024 * 1024,      # KiB — 128 GiB
           "MemAvailable": 7 * 1024 * 1024}    # 7 GiB


# ----------------------------------------------------------- detection

def test_strix_halo_carveout_detected(tmp_path):
    """The reported machine: 512 MiB carve-out, 120 GiB GTT, display class."""
    base = _fake_drm(tmp_path, vram_total=512 * MIB, vram_used=466 * MIB,
                     gtt_total=120 * GIB, gtt_used=70 * GIB,
                     pci_class="0x038000")
    pool = unified_pool(drm_base=str(base), meminfo=MEMINFO)
    assert pool is not None
    assert pool.vram_carveout_mib == 512
    assert pool.total_mib == 120 * 1024          # GTT bounded by MemTotal
    # free = min(gtt_free=50G, mem_avail=7G) = 7 GiB — NOT the 44 MiB
    # carve-out free the vendor tools insisted on.
    assert pool.free_mib == 7 * 1024


def test_discrete_gpu_not_touched(tmp_path):
    """Class 0x030000 (discrete 3D controller): vendor VRAM numbers are
    real; the probe must stay out of the way (return None)."""
    base = _fake_drm(tmp_path, vram_total=24 * GIB, vram_used=1 * GIB,
                     gtt_total=48 * GIB, gtt_used=0,
                     pci_class="0x030000")
    assert unified_pool(drm_base=str(base), meminfo=MEMINFO) is None


def test_no_gtt_not_touched(tmp_path):
    """iGPU-class card without GTT files (no TTM setup): not our case."""
    base = _fake_drm(tmp_path, vram_total=512 * MIB, vram_used=100 * MIB,
                     gtt_total=0, gtt_used=0, pci_class="0x038000")
    assert unified_pool(drm_base=str(base), meminfo=MEMINFO) is None


def test_gtt_must_dominate_carveout(tmp_path):
    """GTT only barely above a *big* carve-out = a normal dGPU with TTM
    staging; class + dominance both required, so skip."""
    base = _fake_drm(tmp_path, vram_total=16 * GIB, vram_used=1 * GIB,
                     gtt_total=40 * GIB, gtt_used=0, pci_class="0x030000")
    assert unified_pool(drm_base=str(base), meminfo=MEMINFO) is None


def test_env_overrides(tmp_path):
    base = _fake_drm(tmp_path, vram_total=512 * MIB, vram_used=466 * MIB,
                     gtt_total=120 * GIB, gtt_used=70 * GIB,
                     pci_class="0x038000")
    off = unified_pool(drm_base=str(base), meminfo=MEMINFO,
                       env={"SABLE_UNIFIED_MEMORY": "0"})
    assert off is None
    # Force-on covers an unknown-class oddity:
    force = _fake_drm(tmp_path / "f", vram_total=512 * MIB, vram_used=0,
                      gtt_total=32 * GIB, gtt_used=0, pci_class="")
    pool = unified_pool(drm_base=str(force), meminfo=MEMINFO,
                        env={"SABLE_UNIFIED_MEMORY": "1"})
    assert pool is not None and pool.total_mib == 32 * 1024


def test_meminfo_absent_bounds_only_by_gtt(tmp_path):
    base = _fake_drm(tmp_path, vram_total=512 * MIB, vram_used=0,
                     gtt_total=120 * GIB, gtt_used=100 * GIB,
                     pci_class="0x038000")
    # {} = meminfo read but fields missing (None means "read /proc"!).
    pool = unified_pool(drm_base=str(base), meminfo={})
    assert pool.free_mib == 20 * 1024            # gtt_free alone
    assert pool.mem_available_mib is None


def test_gtt_beyond_physical_ram_capped(tmp_path):
    """Firmware claims 240 GiB of GTT on a 128 GiB machine: capacity caps
    at MemTotal — the kernel cannot hand out memory that does not exist."""
    base = _fake_drm(tmp_path, vram_total=512 * MIB, vram_used=0,
                     gtt_total=240 * GIB, gtt_used=0, pci_class="0x038000")
    pool = unified_pool(drm_base=str(base), meminfo=MEMINFO)
    assert pool.total_mib == 128 * 1024


def test_non_amd_card_skipped(tmp_path):
    base = _fake_drm(tmp_path, vram_total=512 * MIB, vram_used=0,
                     gtt_total=120 * GIB, gtt_used=0,
                     pci_class="0x038000", vendor="0x8086")
    assert unified_pool(drm_base=str(base), meminfo=MEMINFO) is None


# ------------------------------------------------------------ wiring

def test_scheduler_gate_uses_unified_pool(monkeypatch):
    """The training gate must see the pool, not the 44 MiB carve-out."""
    import sable.training.scheduler as sched

    pool = gpumem.UnifiedPool(total_mib=120 * 1024, free_mib=7 * 1024)
    monkeypatch.setattr(gpumem, "unified_pool", lambda **kw: pool)
    monkeypatch.setattr(sched, "_nvidia_free_mib", lambda *a: None)
    monkeypatch.setattr(sched, "_rocm_free_mib", lambda *a: 44)
    assert sched.gpu_free_mib() == 7 * 1024
    # Device-pinned queries keep the historical per-device semantics.
    assert sched.gpu_free_mib("1") == 44


def test_gate_falls_back_when_probe_raises(monkeypatch):
    import sable.training.scheduler as sched

    def boom(**kw):
        raise RuntimeError("sysfs exploded")
    monkeypatch.setattr(gpumem, "unified_pool", boom)
    monkeypatch.setattr(sched, "_nvidia_free_mib", lambda *a: None)
    monkeypatch.setattr(sched, "_rocm_free_mib", lambda *a: 8000)
    assert sched.gpu_free_mib() == 8000


def test_doctor_chains_unified_first(monkeypatch):
    from sable import doctor

    pool = gpumem.UnifiedPool(total_mib=120 * 1024, free_mib=6656)
    monkeypatch.setattr(gpumem, "unified_pool", lambda **kw: pool)
    monkeypatch.setattr(doctor, "free_vram_mib_via_smi", lambda: 44)
    assert doctor.free_vram_mib() == 6656

    monkeypatch.setattr(gpumem, "unified_pool", lambda **kw: None)
    assert doctor.free_vram_mib() == 44          # NVIDIA path untouched

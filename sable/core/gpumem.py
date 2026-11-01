"""Unified-GPU-memory budget probe — the number vendor tools lie about.

Why this exists: on an APU (AMD Strix Halo / Ryzen AI Max+), a dGPU with its
BIOS carve-out set small, or anything routed through TTM with GTT
(``amdgpu.gttsize``/``config DRM_CARD``), the *vendor* memory reports are all
wrong for capacity planning in the same direction:

* ``rocm-smi --showmeminfo vram`` / sysfs ``mem_info_vram_*`` report only the
  tiny BIOS carve-out (this author's box: **512 MiB total, 44 MiB free**);
* ``rocm-smi --showmeminfo gtt`` reports the GTT *table* capacity (120 GiB),
  which is an upper limit, not what the OS can actually give the GPU;
* torch's ``mem_get_info`` over HIP reports the whole GTT pool as free
  (**128 GiB free** on a box with 7 GiB actually available) because
  unified pages are reclaimable "for free" until they aren't;
* the truthful number is the intersection of the two budgets the kernel
  enforces at allocation time: system ``MemAvailable`` (TTM pages ARE system
  RAM) and GTT-free (table entries). ``min`` of the two is honest.

Which device to trust is decided by the PCI class, not by vendor CLI output:
``0x038000`` is a *display/integrated* controller (an iGPU has no private
RAM — whatever the tools call "VRAM" is carved shared memory), ``0x030000``
is a 3D *discrete* controller whose sysfs/vendor VRAM numbers are real. A
discrete GPU without Resizable BAR looks identical to an iGPU in sysfs
(vram_total = the 256 MiB legacy BAR), which is why the class check, not the
sizes, is the discriminator here.

Detection order (first match wins), no vendor CLI, no root:
  1. ``SABLE_UNIFIED_MEMORY`` env override (same knob serving/runtime uses —
     keep the two in sync by construction);
  2. sysfs: an AMD card with class ``0x038000`` whose ``mem_info_gtt_total``
     dominates ``mem_info_vram_total``;
  3. not unified -> callers keep their existing vendor probes untouched.

When no sysfs exists at all (Windows iGPU, macOS) we do not guess.

Stdlib-only, injectable paths, no imports from serving/ (core must not reach
upward; the board-name heuristic is duplicated by design from
serving/runtime.py's private copy).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional

MIB = 1024 * 1024

#: PCI classes meaning "this GPU's memory IS system memory".
_CLASS_DISPLAY = "0x038000"     # display controller (iGPU: always shared)
_CLASS_VIDEO = "0x038001"       # video controller (VGA-compatible secondary)
_UNIFIED_CLASSES = (_CLASS_DISPLAY, _CLASS_VIDEO)

#: GTT below this relative to a claimed "VRAM" total would be nonsense;
#: guards against a firmware quirk reading gtt_total < vram_total.
_MIN_GTT_DOMINANCE = 4

#: AMD PCI vendor id (sysfs ``device/vendor``).
_AMD_VENDOR = "0x1002"


@dataclass
class UnifiedPool:
    """The GPU's real memory budget on a unified-memory system, in MiB."""

    total_mib: int              # pool capacity the kernel can hand out
    free_mib: int               # min(gtt_free, mem_available) — honest bound
    gtt_total_mib: Optional[int] = None
    gtt_used_mib: Optional[int] = None
    mem_available_mib: Optional[int] = None
    vram_carveout_mib: Optional[int] = None   # what rocm-smi insists is "all"
    card: str = ""
    source: str = "sysfs"

    def as_dict(self) -> Dict[str, object]:
        return {
            "total_mib": self.total_mib,
            "free_mib": self.free_mib,
            "gtt_total_mib": self.gtt_total_mib,
            "gtt_used_mib": self.gtt_used_mib,
            "mem_available_mib": self.mem_available_mib,
            "vram_carveout_mib": self.vram_carveout_mib,
            "card": self.card,
            "source": self.source,
        }


def _read_int(path: Path) -> Optional[int]:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def meminfo_kib() -> Optional[Dict[str, int]]:
    """MemTotal/MemAvailable (KiB) from /proc/meminfo; None off-Linux."""
    try:
        text = Path("/proc/meminfo").read_text()
    except OSError:
        return None
    out: Dict[str, int] = {}
    for line in text.splitlines():
        key, _, val = line.partition(":")
        if key.strip() in ("MemTotal", "MemAvailable"):
            parts = val.split()
            if parts:
                try:
                    out[key.strip()] = int(parts[0])
                except ValueError:
                    pass
    return out if "MemAvailable" in out else None


def _amd_cards(drm_base: Path) -> list:
    """AMD drm cards as (card_dir, pci_class_or_None); sorted, virtual skipped."""
    cards = []
    try:
        entries = sorted(drm_base.glob("card[0-9]"))
    except OSError:
        return cards
    for card in entries:
        dev = card / "device"
        try:
            vendor = (dev / "vendor").read_text().strip().lower()
        except OSError:
            continue
        if vendor != _AMD_VENDOR:
            continue
        pci_class = None
        try:  # first line of the config-space class file
            pci_class = (dev / "class").read_text().strip().lower() or None
        except OSError:
            pass
        cards.append((card, pci_class))
    return cards


def unified_pool(drm_base: Optional[str] = None,
                 meminfo: Optional[Dict[str, int]] = None,
                 env: Optional[Dict[str, str]] = None,
                 ) -> Optional[UnifiedPool]:
    """The GPU memory budget on unified-memory systems, None when not one.

    Returns a pool only for an AMD *integrated* (display-class) controller
    whose GTT dwarfs its VRAM carve-out — exactly the TTM/carve-out setups
    where the vendor reports are wrong. A discrete card (class 0x030000)
    returns None so its honest VRAM numbers stay in charge.
    """
    env_map: Mapping[str, str] = os.environ if env is None else env
    override = str(env_map.get("SABLE_UNIFIED_MEMORY", "")).strip().lower()
    if override in ("0", "false", "no", "off"):
        return None
    base = Path(drm_base) if drm_base else Path("/sys/class/drm")
    if not base.is_dir():
        return None
    if meminfo is None:
        meminfo = meminfo_kib()

    for card, pci_class in _amd_cards(base):
        dev = card / "device"
        vram_total = _read_int(dev / "mem_info_vram_total")
        gtt_total = _read_int(dev / "mem_info_gtt_total")
        gtt_used = _read_int(dev / "mem_info_gtt_used")
        if gtt_total is None or gtt_total <= 0:
            continue
        unified = (
            (pci_class in _UNIFIED_CLASSES and gtt_total >= max(
                1, (vram_total or 0)) * _MIN_GTT_DOMINANCE)
            or override in ("1", "true", "yes", "on")
        )
        if not unified:
            continue  # discrete card (or class unknown without override)

        mem_avail_b = (meminfo or {}).get("MemAvailable", 0) * 1024
        mem_total_b = (meminfo or {}).get("MemTotal", 0) * 1024
        # The kernel can never hand out more than the machine has; some
        # firmwares report GTT table sizes beyond physical RAM.
        capacity = min(gtt_total, mem_total_b) if mem_total_b else gtt_total
        gtt_free = max(0, gtt_total - (gtt_used or 0))
        # Two independent budgets gate an allocation; the smaller one wins.
        # A missing MemAvailable (meminfo unreadable) must NOT force free
        # to 0 — then gtt_free alone bounds it.
        bounds = [capacity, gtt_free] + ([mem_avail_b] if mem_avail_b else [])
        free_b = min(bounds)
        return UnifiedPool(
            total_mib=capacity // MIB,
            free_mib=free_b // MIB,
            gtt_total_mib=gtt_total // MIB,
            gtt_used_mib=(gtt_used or 0) // MIB,
            mem_available_mib=(mem_avail_b // MIB) if mem_avail_b else None,
            vram_carveout_mib=(vram_total // MIB) if vram_total else None,
            card=card.name,
        )
    return None

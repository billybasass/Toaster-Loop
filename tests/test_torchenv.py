"""core/torchenv.py: torch build-flavor resolution + the mismatch doctor.

The bug class this guards (real incident): `pip install torch` on Linux
resolves the CUDA wheel from plain PyPI; on an AMD+ROCm box it imports fine
but cuda.is_available() is False and training silently crawls on CPU. The
resolver must map hardware -> flavor/index, and the doctor must scream when
the installed wheel can never see the GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sable.core import torchenv  # noqa: E402
from sable.core.torchenv import (  # noqa: E402
    FLAVOR_DEFAULT,
    TorchFlavor,
    detect_flavor,
    installed_mismatch,
)


def _g(vendor):
    return SimpleNamespace(vendor=vendor, name="fake")


class FakeTorch:
    """Minimal torch stand-in: __version__, version.cuda/hip, cuda.is_available."""

    def __init__(self, version="", cuda=None, hip=None, avail=False):
        self.__version__ = version
        self.version = SimpleNamespace(cuda=cuda, hip=hip)
        self.cuda = SimpleNamespace(is_available=lambda: avail)


# ------------------------------------------------------------- flavor choice

def test_macos_default(monkeypatch):
    monkeypatch.delenv("SABLE_TORCH_FLAVOR", raising=False)
    f = detect_flavor(system="darwin")
    assert f.flavor == FLAVOR_DEFAULT and f.is_pypi


def test_nvidia_linux_newest_cuda(monkeypatch):
    monkeypatch.delenv("SABLE_TORCH_FLAVOR", raising=False)
    f = detect_flavor(system="linux", machine="x86_64",
                      gpus=[_g("nvidia")])
    assert f.flavor == torchenv.CUDA_FLAVORS[0]
    assert f.index_url.endswith(f"/{torchenv.CUDA_FLAVORS[0]}")


def test_amd_rocm_matches_userland(monkeypatch):
    monkeypatch.delenv("SABLE_TORCH_FLAVOR", raising=False)
    # Newest flavor <= installed userland: 6.4.2 -> rocm6.4, 7.2.1 -> rocm7.2.
    f = detect_flavor(system="linux", gpus=[_g("amd")],
                      rocm_version="6.4.2")
    assert f.flavor == "rocm6.4"
    f = detect_flavor(system="linux", gpus=[_g("amd")],
                      rocm_version="7.2.1")
    assert f.flavor == "rocm7.2"
    # Userland newer than any published flavor: clamp to the newest.
    f = detect_flavor(system="linux", gpus=[_g("amd")],
                      rocm_version="9.9")
    assert f.flavor == f"rocm{torchenv.ROCM_FLAVORS[0]}"


def test_amd_without_rocm_userland_is_cpu_with_note(monkeypatch):
    monkeypatch.delenv("SABLE_TORCH_FLAVOR", raising=False)
    f = detect_flavor(system="linux", gpus=[_g("amd")], rocm_version="")
    assert f.flavor == "cpu"
    assert any("Vulkan" in n for n in f.notes)  # serving unaffected, explains why


def test_no_gpu_cpu(monkeypatch):
    monkeypatch.delenv("SABLE_TORCH_FLAVOR", raising=False)
    f = detect_flavor(system="linux", gpus=[])
    assert f.flavor == "cpu" and not f.is_pypi


def test_intel_arc_xpu(monkeypatch):
    monkeypatch.delenv("SABLE_TORCH_FLAVOR", raising=False)
    f = detect_flavor(system="linux", machine="x86_64", gpus=[_g("intel")])
    assert f.flavor == "xpu"


def test_windows_paths(monkeypatch):
    monkeypatch.delenv("SABLE_TORCH_FLAVOR", raising=False)
    f = detect_flavor(system="win32", machine="amd64", gpus=[_g("nvidia")])
    assert f.flavor == FLAVOR_DEFAULT          # default wheel ships CUDA
    f = detect_flavor(system="win32", machine="amd64", gpus=[_g("intel")])
    assert f.flavor == "xpu"


def test_env_override(monkeypatch):
    monkeypatch.setenv("SABLE_TORCH_FLAVOR", "cu126")
    f = detect_flavor(system="linux", gpus=[_g("amd")], rocm_version="7.2")
    assert f.flavor == "cu126"
    assert f.index_url.endswith("/cu126")
    monkeypatch.setenv("SABLE_TORCH_FLAVOR", "default")
    f = detect_flavor(system="linux", gpus=[_g("amd")])
    assert f.is_pypi


# ------------------------------------------------------------ installer args

def test_flavor_args_shape(monkeypatch):
    monkeypatch.delenv("SABLE_TORCH_FLAVOR", raising=False)
    f = detect_flavor(system="linux", gpus=[_g("amd")], rocm_version="7.1")
    assert f.pip_args()[:2] == ["--index-url",
                                "https://download.pytorch.org/whl/rocm7.1"]
    uv = f.uv_pip_args()
    assert uv[:2] == ["--torch-backend", "rocm7.1"]
    # uv only knows real flavor names; 'default' must send NO backend flag.
    d = TorchFlavor(FLAVOR_DEFAULT, None)
    assert d.uv_pip_args() == [] and d.pip_args() == []


# ------------------------------------------------------------ mismatch doctor

ROCM_WANT = TorchFlavor("rocm7.1", "https://download.pytorch.org/whl/rocm7.1")


def test_mismatch_fires_on_cuda_wheel_on_rocm_box():
    bad = FakeTorch("2.14.1+cu130", cuda="13.0", hip=None, avail=False)
    msg = installed_mismatch(bad, ROCM_WANT)
    assert msg and "rocm7.1" in msg and "--index-url" in msg
    assert "cu130" in msg  # names the wheel that is actually installed


def test_mismatch_fires_on_plain_pypi_wheel_on_rocm_box():
    # No local version tag, no HIP: the uv.lock incident's exact shape.
    bad = FakeTorch("2.14.1", cuda=None, hip=None, avail=False)
    assert installed_mismatch(bad, ROCM_WANT)


def test_rocm_wheel_is_happy():
    good = FakeTorch("2.14.1+rocm7.2", cuda="6.2", hip="7.2", avail=True)
    assert installed_mismatch(good, ROCM_WANT) is None


def test_cpu_build_on_nvidia_box_flagged():
    cu_want = TorchFlavor("cu132", "x")
    bad = FakeTorch("2.14.1+cpu", cuda=None, hip=None, avail=False)
    assert installed_mismatch(bad, cu_want)


def test_no_strong_arm_no_false_positive():
    # CPU flavor wanted, CPU wheel installed: quiet.
    cpu_want = TorchFlavor("cpu", "x")
    ok = FakeTorch("2.14.1+cpu", cuda=None, hip=None, avail=False)
    assert installed_mismatch(ok, cpu_want) is None


# --------------------------------------------------------------- doctor glue

def test_check_function_without_torch(monkeypatch):
    """torchenv's doctor probe must not import torch at all (license
    hygiene): it reads wheel metadata only."""
    monkeypatch.setattr(torchenv, "detect_flavor",
                        lambda **kw: TorchFlavor("cpu", "x"))
    ok, lines = torchenv.check_function(lambda name: None)
    assert ok and "not installed" in lines[0]
    src = Path(torchenv.__file__).read_text()
    import ast
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            assert not any(a.name == "torch" for a in node.names), \
                "torchenv must never import torch (doctor path)"


def test_metadata_mismatch_uses_version_tag_only():
    """Metadata path classifies wheels by their local version tag — the
    doctor never imports torch to do it."""
    assert torchenv.metadata_mismatch("2.14.1+cu130", ROCM_WANT)
    assert torchenv.metadata_mismatch("2.14.1", ROCM_WANT)  # tagless = CUDA
    assert torchenv.metadata_mismatch("2.14.1+cpu", ROCM_WANT)
    assert torchenv.metadata_mismatch("2.14.1+rocm7.2", ROCM_WANT) is None
    cu_want = TorchFlavor("cu132", "x")
    assert torchenv.metadata_mismatch("2.14.1+cpu", cu_want)
    assert torchenv.metadata_mismatch("2.14.1+cu130", cu_want) is None
    # ROCm wheel on any box: quiet (HIP build hears HIP).
    assert torchenv.metadata_mismatch("2.14.1+rocm7.2", cu_want) is None
    cpu_want = TorchFlavor("cpu", "x")
    assert torchenv.metadata_mismatch("2.14.1+cpu", cpu_want) is None

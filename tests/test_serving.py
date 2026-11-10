"""Serving-domain tests: runtime, engines, calibration, watchdog, router.

Consolidated from 15 former test modules; section banners mark origins.
"""

from __future__ import annotations

import sys
from pathlib import Path
import os
import json
from typing import Any, cast
import time
import argparse
import threading
import logging
import stat
import subprocess
import httpx
from http.server import BaseHTTPRequestHandler
import sqlite3
from contextlib import contextmanager
from typing import Optional
from types import SimpleNamespace
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from toaster_loop.core.config import ToasterLoopConfig  # noqa: E402
from toaster_loop.serving import runtime  # noqa: E402
from toaster_loop.serving.supervisor import ServeError, find_llama_server  # noqa: E402
from fake_server import FakeOpenAIServer  # noqa: E402
from toaster_loop.core import tools as uf_tools  # noqa: E402
from toaster_loop.core.agent_runner import ToasterLoopAgentClient  # noqa: E402
from toaster_loop.core.compression import SUMMARY_PREFIX, compress_messages, estimate_tokens, split_for_compression  # noqa: E402
from toaster_loop.core.tool_calls import parse_text_tool_calls, strip_tool_call_markup  # noqa: E402
from toaster_loop.training.worker import assistant_char_regions, sharegpt_to_messages, tokenize_with_labels  # noqa: E402
from toaster_loop.dashboard.app import DashboardApp  # noqa: E402
from toaster_loop.serving import engines as eng_mod  # noqa: E402
from toaster_loop.serving import instances as srv_inst  # noqa: E402
from toaster_loop.serving.supervisor import ServeError  # noqa: E402
from toaster_loop.serving import engine_installs as ei  # noqa: E402
from toaster_loop.serving import bench  # noqa: E402
from toaster_loop.serving import calibrate as cal  # noqa: E402
from toaster_loop.dashboard.app import DashboardApp, HttpError  # noqa: E402
from toaster_loop.registry import ModelRegistry  # noqa: E402
from toaster_loop.serving import coexist as cx  # noqa: E402
from toaster_loop.serving import watchdog as wd  # noqa: E402
from toaster_loop.training.scheduler import Scheduler, SchedulerError, TrainingConfig  # noqa: E402
from toaster_loop.store.db import TraceRow, TraceStore  # noqa: E402
from toaster_loop.training.scheduler import Scheduler, SchedulerError, TrainingConfig, resource_gate_ok, should_auto_train  # noqa: E402
from toaster_loop.core.config import ToasterLoopConfig, load_config  # noqa: E402
from toaster_loop.registry.registry import ModelRegistry  # noqa: E402
from toaster_loop.serving import router as R  # noqa: E402
from toaster_loop.serving.supervisor import ServeError, require_loopback, stop_all_servers  # noqa: E402
from toaster_loop.dashboard import peers as peers_mod  # noqa: E402
from fake_server import FakeOpenAIServer, make_scripted_handler  # noqa: E402
from toaster_loop.evaluation.shadow import NO_REPLAY_SET, ShadowError, build_replay_set, compare_scores, run_shadow_eval, shadow_promotion_check  # noqa: E402
from toaster_loop.serving.instances import ServeError, _normalize_gpu_id, instance_device_env  # noqa: E402
from toaster_loop.serving.supervisor import _gpu_pin_args  # noqa: E402
from toaster_loop import doctor  # noqa: E402
from toaster_loop.core.config import ConfigError, ToasterLoopConfig  # noqa: E402
from toaster_loop.core.exitcodes import EXIT_CONFIG_INVALID, EXIT_ERROR, EXIT_OK  # noqa: E402

# ------------------------------------------------------------------------
# from tests/test_runtime.py
# ------------------------------------------------------------------------

def _info(backend: str, **kw) -> runtime.RuntimeInfo:
    return runtime.RuntimeInfo(system="linux", machine="x86_64", backend=backend, **kw)


# ------------------------------------------------------------------ -ngl

def test_explicit_gpu_layers_wins(monkeypatch):
    monkeypatch.setattr(runtime, "detect_runtime", lambda: _info("cpu"))
    assert runtime.gpu_layers(32) == 32
    assert runtime.gpu_layers(0) == 0  # explicit 0 forces CPU
    assert runtime.launch_args(32) == ["-ngl", "32"]
    assert runtime.launch_args(0) == []


def test_auto_offload_by_backend(monkeypatch):
    for backend, expected in (
        ("cuda", runtime.FULL_OFFLOAD_LAYERS),
        ("rocm", runtime.FULL_OFFLOAD_LAYERS),
        ("metal", runtime.FULL_OFFLOAD_LAYERS),
        ("vulkan", runtime.FULL_OFFLOAD_LAYERS),
        ("cpu", 0),
    ):
        monkeypatch.setattr(runtime, "detect_runtime", lambda b=backend: _info(b))
        assert runtime.gpu_layers(None) == expected


def test_launch_args_cpu_empty(monkeypatch):
    monkeypatch.setattr(runtime, "detect_runtime", lambda: _info("cpu"))
    assert runtime.launch_args(None) == []


# ---------------------------------------------------------- detect_runtime

def test_detect_runtime_backend_selection(monkeypatch):
    # (gpus, rocm_userland, nvidia_native, amd_native) — the tuple
    # _detect_gpus_impl returns; backend choice keys off the NATIVE-probe
    # flags, not vendor names (a Vulkan-enumerated name is not CUDA evidence).
    def impl(*items, rocm=False, nv=None, amd=None):
        gpus = [runtime.GpuInfo(vendor=v, name=n) for v, n in items]
        nvidia_native = any(g.vendor == "nvidia" for g in gpus) if nv is None else nv
        amd_native = any(g.vendor == "amd" for g in gpus) if amd is None else amd

        def _f():
            return gpus, rocm, nvidia_native, amd_native
        return _f

    monkeypatch.setattr(runtime, "_detect_gpus_impl",
                        impl(("nvidia", "RTX 4090")))
    assert runtime.detect_runtime().backend == "cuda"

    # NVIDIA outranks an AMD card reported alongside it.
    monkeypatch.setattr(runtime, "_detect_gpus_impl",
                        impl(("amd", "gfx"), ("nvidia", "T4"), rocm=True))
    assert runtime.detect_runtime().backend == "cuda"

    monkeypatch.setattr(runtime, "_detect_gpus_impl",
                        impl(("amd", "gfx1100"), rocm=True))
    assert runtime.detect_runtime().backend == "rocm"

    # AMD without ROCm userland and no Vulkan => CPU with an explanatory note.
    monkeypatch.setattr(runtime, "_has_vulkan", lambda: False)
    monkeypatch.setattr(runtime, "_detect_gpus_impl",
                        impl(("amd", "gfx1100"), rocm=False))
    info = runtime.detect_runtime()
    assert info.backend == "cpu"
    assert any("ROCm" in n for n in info.backend_notes)

    monkeypatch.setattr(runtime, "_detect_gpus_impl", impl(("apple", "M3")))
    assert runtime.detect_runtime().backend == "metal"

    # A Vulkan-enumerated NVIDIA (no nvidia-smi) must NOT pick CUDA.
    monkeypatch.setattr(runtime, "_has_vulkan", lambda: True)
    monkeypatch.setattr(
        runtime, "_detect_gpus_impl",
        impl(("nvidia", "NVIDIA GeForce RTX 4090 (Vulkan)"), nv=False))
    assert runtime.detect_runtime().backend == "vulkan"

    monkeypatch.setattr(runtime, "_detect_gpus_impl", impl())
    assert runtime.detect_runtime().backend == "vulkan"

    monkeypatch.setattr(runtime, "_has_vulkan", lambda: False)
    info = runtime.detect_runtime()
    assert info.backend == "cpu"
    assert info.backend_notes  # CPU-only must explain itself


def test_detect_gpus_never_raises(monkeypatch):
    # Every vendor probe fails => empty list, no exception.
    monkeypatch.setattr(runtime, "_run", lambda *a, **k: None)
    monkeypatch.setattr(runtime, "_amdgpu_sysfs", lambda: [])
    assert runtime.detect_gpus() == []


# --------------------------------------------------- unified-memory systems

def _um_info(gpu_names, backend="rocm"):
    info = _info(backend)
    info.gpus = [runtime.GpuInfo(vendor="amd", name=n) for n in gpu_names]
    return info


def test_unified_memory_strix_halo(monkeypatch):
    monkeypatch.delenv("TOASTER_LOOP_UNIFIED_MEMORY", raising=False)
    monkeypatch.setattr(runtime, "_is_jetson", lambda: False)
    monkeypatch.setattr(runtime, "_is_strix_halo", lambda: False)
    assert runtime.is_unified_memory(_um_info(["gfx1151"])) is True
    assert runtime.is_unified_memory(
        _um_info(["AMD RYZEN AI MAX+ 395 w/ Radeon 8060S"])) is True
    # A discrete AMD card is NOT unified memory.
    assert runtime.is_unified_memory(_um_info(["gfx1030"])) is False


def test_unified_memory_strix_halo_sysfs(monkeypatch):
    # rocminfo-less Strix Halo: sysfs GPU name carries no gfx1151, but the
    # cpuinfo probe identifies the board => unified memory on.
    monkeypatch.delenv("TOASTER_LOOP_UNIFIED_MEMORY", raising=False)
    monkeypatch.setattr(runtime, "_is_jetson", lambda: False)
    monkeypatch.setattr(runtime, "_is_strix_halo", lambda: True)
    assert runtime.is_unified_memory(
        _um_info(["AMD GPU [1002:1586]"])) is True


def test_unified_memory_jetson(monkeypatch):
    monkeypatch.delenv("TOASTER_LOOP_UNIFIED_MEMORY", raising=False)
    monkeypatch.setattr(runtime, "_is_jetson", lambda: True)
    assert runtime.is_unified_memory(_um_info([], backend="cuda")) is True


def test_unified_memory_mac_excluded(monkeypatch):
    monkeypatch.delenv("TOASTER_LOOP_UNIFIED_MEMORY", raising=False)
    monkeypatch.setattr(runtime, "_is_jetson", lambda: False)
    monkeypatch.setattr(runtime.sys, "platform", "darwin")
    assert runtime.is_unified_memory(_um_info([("apple M4")], backend="metal")) is False


def test_unified_memory_override(monkeypatch):
    monkeypatch.setenv("TOASTER_LOOP_UNIFIED_MEMORY", "0")
    assert runtime.is_unified_memory(_um_info(["gfx1151"])) is False
    monkeypatch.setenv("TOASTER_LOOP_UNIFIED_MEMORY", "1")
    assert runtime.is_unified_memory(_um_info(["gfx1030"])) is True


def test_unified_memory_env_value(monkeypatch):
    monkeypatch.delenv("TOASTER_LOOP_UNIFIED_MEMORY", raising=False)
    monkeypatch.setattr(runtime, "_is_jetson", lambda: False)
    monkeypatch.setattr(runtime, "_is_strix_halo", lambda: False)
    assert runtime.unified_memory_env(_um_info(["gfx1151"])) == {
        "GGML_CUDA_ENABLE_UNIFIED_MEMORY": "1"}
    assert runtime.unified_memory_env(_um_info(["gfx1030"])) == {}


def test_detect_gpus_parses_nvidia_csv(monkeypatch):
    def fake_run(cmd, timeout=8.0):
        if cmd[0] == "nvidia-smi":
            return "NVIDIA GeForce RTX 4090, 20480, 24564\nTesla T4, 1234, 15360\n"
        return None

    monkeypatch.setattr(runtime, "_run", fake_run)
    monkeypatch.setattr(runtime, "_amdgpu_sysfs", lambda: [])
    gpus = runtime.detect_gpus()
    assert [(g.vendor, g.free_mib, g.total_mib) for g in gpus] == [
        ("nvidia", 20480, 24564),
        ("nvidia", 1234, 15360),
    ]


# ------------------------------------------------------------ search dirs

def test_search_dirs_include_toaster_loop_and_env_root(tmp_path, monkeypatch):
    monkeypatch.setenv("TOASTER_LOOP_LLAMA_CPP", str(tmp_path / "llama.cpp"))
    dirs = runtime.search_dirs()
    assert tmp_path / "llama.cpp" / "build" / "bin" in dirs
    # toaster_loop is standalone: it must never depend on (or probe) the separate
    # unrelated per-installer directories.
    homes = [d for d in dirs if ".unrelated-installer" in str(d)]
    assert not homes, "toaster_loop must not search foreign installer dirs"


def test_discover_external_server_url_opt_in_only(monkeypatch):
    # No explicit env var => no probing at all (no default-port scan).
    from toaster_loop.core.config import discover_external_server_url

    monkeypatch.delenv("TOASTER_LOOP_EXTERNAL_SERVER_URL", raising=False)
    assert discover_external_server_url() is None


def test_toaster_loop_llama_root_uses_home_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(tmp_path))
    assert runtime.toaster_loop_llama_root() == tmp_path / "llama.cpp"


# ------------------------------------------------------------ auto_setup

def test_auto_setup_enabled_default(monkeypatch):
    monkeypatch.delenv("TOASTER_LOOP_NO_AUTO_SETUP", raising=False)
    assert runtime.auto_setup_enabled(ToasterLoopConfig()) is True


def test_auto_setup_env_optout(monkeypatch):
    monkeypatch.setenv("TOASTER_LOOP_NO_AUTO_SETUP", "1")
    assert runtime.auto_setup_enabled(ToasterLoopConfig()) is False


def test_auto_setup_config_optout(monkeypatch):
    monkeypatch.delenv("TOASTER_LOOP_NO_AUTO_SETUP", raising=False)
    assert runtime.auto_setup_enabled(ToasterLoopConfig(llama_auto_setup=False)) is False


# ---------------------------------------------------- find_llama_server

def test_find_binary_explicit_missing_raises(tmp_path):
    with pytest.raises(ServeError, match="not found at"):
        find_llama_server(str(tmp_path / "nope" / "llama-server"))


def test_find_binary_auto_setup_disabled_no_clone(monkeypatch, tmp_path):
    # No binary anywhere + auto_setup off => actionable error, never a clone.
    monkeypatch.setattr(runtime, "search_dirs", lambda: [])
    monkeypatch.setattr(runtime.shutil, "which", lambda name: None)

    def boom(*a, **k):
        raise AssertionError("must not clone when auto_setup is disabled")

    monkeypatch.setattr(runtime, "setup_llama_cpp", boom)
    with pytest.raises(ServeError, match="toaster_loop setup"):
        find_llama_server(auto_setup=False)


def test_find_binary_triggers_setup_and_retries(monkeypatch, tmp_path):
    # Nothing found initially; the fake setup drops a binary into a search dir.
    stub = tmp_path / "llama-server"
    calls = {"n": 0}

    def fake_setup(backend=None, target_dir=None, log_fn=None, **kw):
        stub.write_text("#!/bin/sh\nexit 0\n")
        stub.chmod(0o755)

    def fake_which(name):
        if name != "llama-server":
            return "/usr/bin/git"
        calls["n"] += 1
        return str(stub) if calls["n"] > 1 and stub.exists() else None

    monkeypatch.setattr(runtime, "search_dirs", lambda: [])
    monkeypatch.setattr(runtime.shutil, "which", fake_which)
    monkeypatch.setattr(runtime, "setup_llama_cpp", fake_setup)
    monkeypatch.setattr(runtime, "detect_runtime", lambda: _info("cpu"))

    assert find_llama_server(auto_setup=True) == str(stub)


def test_find_binary_setup_error_wrapped(monkeypatch):
    monkeypatch.setattr(runtime, "search_dirs", lambda: [])
    monkeypatch.setattr(runtime.shutil, "which", lambda name: None)

    def fail(backend=None, target_dir=None, log_fn=None, **kw):
        raise runtime.SetupError("cmake missing")

    monkeypatch.setattr(runtime, "setup_llama_cpp", fail)
    monkeypatch.setattr(runtime, "detect_runtime", lambda: _info("cpu"))
    with pytest.raises(ServeError, match="auto-setup failed: cmake missing"):
        find_llama_server(auto_setup=True)


# --------------------------------------------------------------- config

def test_gpu_layers_env_override(monkeypatch):
    from toaster_loop.core.config import load_config

    monkeypatch.setenv("TOASTER_LOOP_GPU_LAYERS", "17")
    assert load_config().gpu_layers == 17


def test_auto_setup_env_override(monkeypatch):
    from toaster_loop.core.config import load_config

    monkeypatch.setenv("TOASTER_LOOP_LLAMA_AUTO_SETUP", "false")
    assert load_config().llama_auto_setup is False


def test_llama_source_env_overrides(monkeypatch):
    from toaster_loop.core.config import load_config

    monkeypatch.setenv("TOASTER_LOOP_LLAMA_REPO", "https://example.com/fork/llama.cpp")
    monkeypatch.setenv("TOASTER_LOOP_LLAMA_REF", "gfx1151")
    monkeypatch.setenv("TOASTER_LOOP_LLAMA_BACKEND", "vulkan")
    cfg = load_config()
    assert cfg.llama_repo == "https://example.com/fork/llama.cpp"
    assert cfg.llama_ref == "gfx1151"
    assert cfg.llama_backend == "vulkan"


# ------------------------------------------------- install state + updates

def test_install_state_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(tmp_path))
    st = {"repo": "https://example.com/fork", "ref": "b1", "backend": "vulkan",
          "commit": "abc1234", "built_at": "2026-09-15T00:00:00Z"}
    runtime._write_install_state(st)
    assert runtime.read_install_state() == st
    # State lives OUTSIDE the checkout (a git clean can't fake it).
    assert runtime.state_path() == tmp_path / runtime.STATE_FILENAME


def test_check_llama_update_not_installed(tmp_path, monkeypatch):
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(tmp_path))
    res = runtime.check_llama_update()
    assert res["installed"] is False and res["update_available"] is False


def test_check_llama_update_behind(tmp_path, monkeypatch):
    # Fake checkout + state; remote probe faked to a different sha.
    root = tmp_path / "llama.cpp"
    (root / ".git").mkdir(parents=True)
    (root / "CMakeLists.txt").write_text("project")
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(tmp_path))
    runtime._write_install_state({"repo": "r", "ref": "master", "backend": "cpu",
                                  "commit": "aaa111"})
    monkeypatch.setattr(runtime, "git_head", lambda r: "aaa111")
    monkeypatch.setattr(runtime, "remote_ref_commit", lambda repo, ref: "bbb222")
    res = runtime.check_llama_update()
    assert res["update_available"] is True

    monkeypatch.setattr(runtime, "remote_ref_commit", lambda repo, ref: "aaa11199")
    assert runtime.check_llama_update()["update_available"] is False


def test_check_llama_update_offline(tmp_path, monkeypatch):
    root = tmp_path / "llama.cpp"
    (root / ".git").mkdir(parents=True)
    (root / "CMakeLists.txt").write_text("x")
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(tmp_path))
    runtime._write_install_state({"repo": "r", "ref": "m", "commit": "a"})
    monkeypatch.setattr(runtime, "git_head", lambda r: "a")
    monkeypatch.setattr(runtime, "remote_ref_commit", lambda repo, ref: None)
    res = runtime.check_llama_update()
    assert res["update_available"] is False and "offline" in res["reason"]


def test_uninstall_only_touches_managed_install(tmp_path, monkeypatch):
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(tmp_path))
    # Nothing installed => no-op.
    assert runtime.uninstall_llama_cpp() is False

    root = tmp_path / "llama.cpp"
    (root / ".git").mkdir(parents=True)
    runtime._write_install_state({"repo": "r"})
    assert runtime.uninstall_llama_cpp() is True
    assert not root.exists() and runtime.read_install_state() is None


def test_uninstall_refuses_non_checkout(tmp_path, monkeypatch):
    # Safety: a same-named directory that isn't a llama.cpp checkout survives.
    root = tmp_path / "llama.cpp"
    (root / "important").mkdir(parents=True)
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(tmp_path))
    assert runtime.uninstall_llama_cpp() is False
    assert (root / "important").is_dir()


def test_cmake_flags_per_backend():
    # The selected backend is ON and the other GPU backends are explicitly OFF:
    # cmake caches -DGGML_* in CMakeCache.txt, so switching an existing build
    # dir from ROCm to Vulkan without turning HIP off keeps building ROCm.
    def flags(b):
        return set(runtime._cmake_flags(b))
    assert "-DGGML_CUDA=ON" in flags("cuda")
    assert "-DGGML_HIP=OFF" in flags("cuda") and "-DGGML_VULKAN=OFF" in flags("cuda")
    assert "-DGGML_HIP=ON" in flags("rocm")
    assert "-DGGML_CUDA=OFF" in flags("rocm") and "-DGGML_VULKAN=OFF" in flags("rocm")
    assert "-DGGML_VULKAN=ON" in flags("vulkan")
    assert "-DGGML_CUDA=OFF" in flags("vulkan") and "-DGGML_HIP=OFF" in flags("vulkan")
    assert runtime._cmake_flags("metal") == []  # default macOS build, no flags
    assert "-DGGML_NATIVE=ON" in flags("cpu")
    for off in ("-DGGML_CUDA=OFF", "-DGGML_HIP=OFF", "-DGGML_VULKAN=OFF"):
        assert off in flags("cpu")


def test_setup_backend_switch_clears_stale_build_cache(tmp_path, monkeypatch):
    """Regression: switching rocm -> vulkan must drop the old CMakeCache or
    cmake silently keeps the cached HIP build (dropdown appears to do nothing)."""
    root = tmp_path / "llama.cpp"
    (root / ".git").mkdir(parents=True)
    (root / "CMakeLists.txt").write_text("project(x)\n")
    cache = root / "build" / "CMakeCache.txt"
    cache.parent.mkdir()
    cache.write_text("GGML_HIP:BOOL=ON\n")
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(tmp_path))
    runtime._write_install_state({"repo": "r", "ref": "master",
                                  "backend": "rocm", "commit": "abc"})

    cleared = []

    class _P:  # minimal fake subprocess.run result
        returncode = 0
        stdout = ""
        stderr = ""

    class _FakeBuild:  # minimal fake Popen for the cmake --build step
        stdout = iter([])

        def wait(self):
            return 0

    monkeypatch.setattr(runtime, "detect_runtime",
                        lambda: runtime.RuntimeInfo(system="linux",
                                                    machine="x86_64"))
    monkeypatch.setattr(runtime, "_require_tool",
                        lambda name, hint: f"/usr/bin/{name}")
    monkeypatch.setattr(runtime.subprocess, "run", lambda *a, **k: _P())
    monkeypatch.setattr(runtime.subprocess, "Popen",
                        lambda *a, **k: _FakeBuild())
    monkeypatch.setattr(runtime.shutil, "rmtree",
                        lambda p, *a, **k: cleared.append(str(p)))
    # No binary after the (faked) build => SetupError; we only care that the
    # stale cache was removed before cmake configure ran.
    monkeypatch.setattr(runtime, "llama_server_path", lambda root: None)

    with pytest.raises(runtime.SetupError):
        runtime.setup_llama_cpp(backend="vulkan", target_dir=root,
                                log_fn=lambda _s: None)
    assert cleared and cleared[0].endswith("llama.cpp/build"), \
        "stale build cache not cleared on backend switch"


# ------------------------------------------------- per-backend install slots

def test_toaster_loop_llama_root_per_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(tmp_path))
    assert runtime.toaster_loop_llama_root() == tmp_path / "llama.cpp"
    assert runtime.toaster_loop_llama_root("vulkan") == tmp_path / "llama.cpp-vulkan"
    assert runtime.toaster_loop_llama_root("ROCM") == tmp_path / "llama.cpp-rocm"


def test_install_root_for_routing(tmp_path, monkeypatch):
    """Primary slot hosts the first backend; a *different* backend routes to
    its own llama.cpp-<backend> checkout instead of overwriting it."""
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(tmp_path))
    # empty home => anything uses the primary slot
    root, be = runtime.install_root_for("vulkan")
    assert root == tmp_path / "llama.cpp" and be is None
    # primary holds rocm => vulkan gets its own slot, rocm keeps the primary
    runtime._write_install_state({"backend": "rocm"})
    root, be = runtime.install_root_for("vulkan")
    assert root == tmp_path / "llama.cpp-vulkan" and be == "vulkan"
    root, be = runtime.install_root_for("rocm")
    assert root == tmp_path / "llama.cpp" and be is None
    # once vulkan has its own state, it keeps resolving to its slot
    runtime._write_install_state({"backend": "vulkan"}, backend="vulkan")
    root, be = runtime.install_root_for("vulkan")
    assert root == tmp_path / "llama.cpp-vulkan" and be == "vulkan"


def test_per_backend_state_files_are_independent(tmp_path, monkeypatch):
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(tmp_path))
    runtime._write_install_state({"backend": "rocm", "commit": "aaa"})
    runtime._write_install_state({"backend": "vulkan", "commit": "bbb"},
                                 backend="vulkan")
    assert (tmp_path / "llama_cpp_install.json").is_file()
    assert (tmp_path / "llama_cpp_install-vulkan.json").is_file()
    st = runtime.read_install_state()
    stv = runtime.read_install_state(backend="vulkan")
    assert st is not None and st["backend"] == "rocm"
    assert stv is not None and stv["backend"] == "vulkan"


def test_list_installs_reports_every_slot(tmp_path, monkeypatch):
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(tmp_path))
    runtime._write_install_state({"backend": "rocm", "repo": "r"})
    runtime._write_install_state({"backend": "vulkan", "repo": "r"},
                                 backend="vulkan")
    # primary has a binary; the vulkan slot does not (yet)
    exe = tmp_path / "llama.cpp" / "build" / "bin" / "llama-server"
    exe.parent.mkdir(parents=True)
    exe.write_text("#!/bin/sh\n")
    os.chmod(exe, 0o755)

    installs = {i["backend"]: i for i in runtime.list_installs()}
    assert set(installs) == {"rocm", "vulkan"}
    assert installs["rocm"]["primary"] is True
    assert installs["rocm"]["binary"] == str(exe)
    assert installs["vulkan"]["primary"] is False
    assert installs["vulkan"]["binary"] is None


def test_search_dirs_includes_backend_slots(tmp_path, monkeypatch):
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(tmp_path))
    runtime._write_install_state({"backend": "rocm"}, backend=None)
    runtime._write_install_state({"backend": "vulkan"}, backend="vulkan")
    dirs = [str(p) for p in runtime.search_dirs()]
    assert str(tmp_path / "llama.cpp" / "build" / "bin") in dirs
    assert str(tmp_path / "llama.cpp-vulkan" / "build" / "bin") in dirs


def test_uninstall_backend_slot_keeps_primary(tmp_path, monkeypatch):
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(tmp_path))
    prim = tmp_path / "llama.cpp"
    (prim / ".git").mkdir(parents=True)
    slot = tmp_path / "llama.cpp-vulkan"
    (slot / ".git").mkdir(parents=True)
    runtime._write_install_state({"backend": "rocm"})
    runtime._write_install_state({"backend": "vulkan"}, backend="vulkan")

    assert runtime.uninstall_llama_cpp(backend="vulkan") is True
    assert prim.is_dir() and slot.is_dir() is False
    assert runtime.read_install_state() is not None       # primary intact
    assert runtime.read_install_state(backend="vulkan") is None


# ------------------------------------------------------ Vulkan SDK probing

def _fake_sdk(root: Path) -> Path:
    """Materialize a minimal-but-valid Vulkan SDK tree under root."""
    (root / "share" / "cmake" / "SPIRV-Headers").mkdir(parents=True)
    (root / "share" / "cmake" / "SPIRV-Headers"
     / "SPIRV-HeadersConfig.cmake").write_text("# cfg\n")
    (root / "include" / "spirv" / "unified1").mkdir(parents=True)
    (root / "include" / "spirv" / "unified1" / "spirv.hpp").write_text("//h\n")
    return root


def test_find_vulkan_sdk_prefers_env_when_valid(tmp_path, monkeypatch):
    good = _fake_sdk(tmp_path / "sdk" / "x86_64")
    monkeypatch.setenv("VULKAN_SDK", str(good))
    assert runtime.find_vulkan_sdk() == good


def test_find_vulkan_sdk_rejects_stale_env_and_falls_back(tmp_path, monkeypatch):
    """Regression: a VULKAN_SDK pointing at the wrong dir used to pass through
    untouched — llama.cpp then configured but died mid-build on
    spirv/unified1/spirv.hpp. Now we validate and fall back."""
    good = _fake_sdk(tmp_path / ".vulkan" / "1.4.363.0" / "x86_64")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("VULKAN_SDK", str(tmp_path / "does-not-exist"))
    assert runtime.find_vulkan_sdk() == good


def test_find_vulkan_sdk_none_when_nothing_usable(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("VULKAN_SDK", str(tmp_path / "nope"))
    # /usr and /usr/local may exist on the runner; only assert no crash + that
    # a bogus env does not win.
    got = runtime.find_vulkan_sdk()
    assert got is None or _vulkan_ok(got)


def _vulkan_ok(p: Path) -> bool:
    return (p / "share" / "cmake" / "SPIRV-Headers"
            / "SPIRV-HeadersConfig.cmake").is_file()


def test_vulkan_build_env_flags(tmp_path, monkeypatch):
    good = _fake_sdk(tmp_path / "sdk")
    monkeypatch.setenv("VULKAN_SDK", str(good))
    env, flags = runtime._vulkan_build_env(lambda _s: None)
    assert env["VULKAN_SDK"] == str(good)
    assert any(f.startswith("-DSPIRV-Headers_DIR=") for f in flags)


def test_setup_vulkan_without_sdk_raises_actionable(tmp_path, monkeypatch):
    """No usable SDK => clear error before any build starts (not a hang or a
    mid-compile spirv.hpp failure)."""
    root = tmp_path / "llama.cpp"
    (root / ".git").mkdir(parents=True)
    (root / "CMakeLists.txt").write_text("project(x)\n")
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(tmp_path))
    monkeypatch.setattr(runtime, "detect_runtime",
                        lambda: runtime.RuntimeInfo(system="linux",
                                                    machine="x86_64"))
    monkeypatch.setattr(runtime, "_require_tool",
                        lambda name, hint: f"/usr/bin/{name}")
    monkeypatch.setattr(runtime, "find_vulkan_sdk", lambda: None)

    started = []

    class _P:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, *a, **k):
        started.append(" ".join(str(c) for c in cmd))
        return _P()
    monkeypatch.setattr(runtime.subprocess, "run", fake_run)

    with pytest.raises(runtime.SetupError, match="SPIRV-Headers"):
        runtime.setup_llama_cpp(backend="vulkan", target_dir=root,
                                log_fn=lambda _s: None)
    assert not any("--build" in c for c in started)  # failed fast


# ------------------------------------------------------- dashboard API

def test_dashboard_runtime_endpoints(tmp_path, monkeypatch):
    from toaster_loop.core.config import ToasterLoopConfig
    from toaster_loop.dashboard.app import DashboardApp

    monkeypatch.setenv("TOASTER_LOOP_HOME", str(tmp_path))
    app = DashboardApp(ToasterLoopConfig(), home=tmp_path)

    r = app.handle("GET", "/api/runtime", query={"check": "0"})
    assert r.status == 200
    body = r.body
    assert body["backend"] in ("cuda", "rocm", "metal", "vulkan", "cpu")
    assert body["configured"]["repo"] == runtime.LLAMA_CPP_REPO
    assert "install" in body and body["backends"]

    assert app.handle("GET", "/api/runtime/setup").body == {"state": "idle"}

    # Validation happens before any thread spawn.
    r = app.handle("POST", "/api/runtime/setup", body={"backend": "quantum"})
    assert r.status == 400 and r.body["error"]["code"] == "bad_request"


def test_dashboard_runtime_setup_runs_and_serializes(tmp_path, monkeypatch):
    import threading
    import time as _t

    from toaster_loop.core.config import ToasterLoopConfig
    from toaster_loop.dashboard.app import DashboardApp

    monkeypatch.setenv("TOASTER_LOOP_HOME", str(tmp_path))
    app = DashboardApp(ToasterLoopConfig(), home=tmp_path)

    started = threading.Event()
    release = threading.Event()

    def fake_setup(backend=None, log_fn=None, repo=None, ref=None, update=False):
        started.set()
        assert log_fn is not None
        log_fn("fake build line")
        release.wait(5)

    monkeypatch.setattr(runtime, "setup_llama_cpp", fake_setup)

    r = app.handle("POST", "/api/runtime/setup", body={"backend": "vulkan"})
    assert r.status == 200 and r.body["state"] == "running"
    assert started.wait(5)
    # Second request while running => 409 single-flight.
    r2 = app.handle("POST", "/api/runtime/setup", body={})
    assert r2.status == 409 and r2.body["error"]["code"] == "already_running"

    release.set()
    st = {}
    for _ in range(50):
        st = app.handle("GET", "/api/runtime/setup").body
        if st.get("state") == "done":
            break
        _t.sleep(0.1)
    assert st.get("state") == "done"
    assert "fake build line" in "\n".join(st.get("log", []))

# ------------------------------------------------------------------------
# from tests/test_runtime_features.py
# ------------------------------------------------------------------------

LT = "<"  # avoid editor/template tooling mangling tag-like test fixtures


@pytest.fixture(scope="module")
def fake_server():
    with FakeOpenAIServer() as srv:
        yield srv


# ------------------------------------------------------ 1. text tool calls

def test_parse_fenced_json_call():
    content = (
        "Sure!\n```json\n"
        + json.dumps({"name": "read_file", "arguments": {"path": "a.py"}})
        + "\n```\n"
    )
    calls = parse_text_tool_calls(content)
    assert len(calls) == 1
    fn = calls[0]["function"]
    assert fn["name"] == "read_file"
    assert json.loads(fn["arguments"]) == {"path": "a.py"}
    assert calls[0]["id"].startswith("call_text_")
    # prose survives, block is stripped
    assert strip_tool_call_markup(content) == "Sure!"


def test_parse_openai_nested_and_list():
    content = ("```json\n"
               + json.dumps([{"function": {"name": "list_dir",
                                           "arguments": '{"path": "."}'}}])
               + "\n```")
    calls = parse_text_tool_calls(content)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "list_dir"
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "."}


def test_parse_xmlish_call():
    content = (f"Let me check.\n{LT}tool_call>read_file"
               + json.dumps({"path": "x.txt"}) + f"{LT}/tool_call>")
    calls = parse_text_tool_calls(content)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "read_file"


def test_parse_negatives():
    assert parse_text_tool_calls(None) == []
    assert parse_text_tool_calls("Just a normal answer.") == []
    # non-tool JSON is left alone
    assert parse_text_tool_calls(f"```json\n{json.dumps({'foo': 1})}\n```") == []
    # leaked chat-template markup => never rewritten
    im = LT + "|"
    assert parse_text_tool_calls(
        f"{im}im_start{im}assistant hi {im}im_end{im}") == []


def test_agent_loop_executes_text_tool_call():
    """End-to-end: a server that returns a fenced JSON call gets it executed."""
    from fake_server import _Handler

    class TextCallHandler(_Handler):
        def _completion_payload(self, payload):  # noqa: N802
            messages = payload.get("messages", [])
            saw_tool_result = any(m.get("role") == "tool" for m in messages)
            if not saw_tool_result:
                msg = {"role": "assistant", "content":
                       'Working.\n```json\n{"name": "list_dir", '
                       '"arguments": {"path": "."}}\n```'}
                finish = "stop"  # NOTE: no native tool_calls field at all
            else:
                msg = {"role": "assistant",
                       "content": "Saw the directory listing."}
                finish = "stop"
            return {
                "id": "cmpl-text", "object": "chat.completion",
                "model": payload.get("model", "fake-model"),
                "choices": [{"index": 0, "message": msg,
                             "finish_reason": finish}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                          "total_tokens": 2},
            }

    with FakeOpenAIServer(handler_cls=TextCallHandler) as fs:
        client = ToasterLoopAgentClient(base_url=fs.base_url, model="fake-model")
        result = client.run("list stuff")
    assert result.completed
    roles = [m.get("role") for m in result.messages]
    assert "tool" in roles  # the text call was actually executed
    assert "Saw the directory listing." in result.final_response


# ------------------------------------------------------ 2. compression

def _msg(role, content):
    return {"role": role, "content": content}


def test_estimate_tokens_scales_with_size():
    small = estimate_tokens([_msg("user", "hi")])
    big = estimate_tokens([_msg("user", "x" * 4000)])
    assert big > small * 10


def test_split_and_compress_replaces_head():
    msgs = [_msg("system", "SYS")]
    for i in range(30):
        msgs.append(_msg("user", f"question {i} " + "y" * 200))
        msgs.append(_msg("assistant", f"answer {i} " + "z" * 200))

    parts = split_for_compression(msgs[1:], keep_recent_tokens=300)
    assert parts is not None
    head, tail = parts
    assert head and tail
    assert tail[0]["role"] in ("user", "assistant")

    out = compress_messages(msgs, summarize=None, threshold_tokens=10,
                            keep_recent_tokens=300)
    # system prompt kept first, exactly one summary message, tail preserved
    assert out[0]["content"] == "SYS"
    summaries = [m for m in out if str(m["content"]).startswith(SUMMARY_PREFIX)]
    assert len(summaries) == 1
    assert out[-len(tail):] == tail
    # idempotent: recompressing under threshold is a no-op
    again = compress_messages(out, summarize=None, threshold_tokens=10 ** 9)
    assert again == out


def test_compress_disabled_and_short_noop():
    msgs = [_msg("system", "S"), _msg("user", "hi")]
    assert compress_messages(msgs, None, threshold_tokens=0) == msgs
    assert compress_messages(msgs, None, threshold_tokens=10_000) == msgs


def test_llm_summarizer_failure_falls_back_to_digest():
    msgs = [_msg("system", "S")]
    for i in range(20):
        msgs.append(_msg("user", f"q{i} " + "u" * 300))
        msgs.append(_msg("assistant", f"a{i} " + "v" * 300))

    def boom(prompt):
        raise RuntimeError("endpoint down")

    out = compress_messages(msgs, boom, threshold_tokens=10,
                            keep_recent_tokens=200)
    summary = next(m for m in out
                   if str(m["content"]).startswith(SUMMARY_PREFIX))
    assert "user asked" in summary["content"]  # extractive digest shape


def test_tool_call_pairing_preserved():
    """Tail must never start with an orphan tool message."""
    msgs: list = [_msg("system", "S")]
    for i in range(10):
        msgs.append(_msg("user", f"q{i}"))
        msgs.append({"role": "assistant", "content": None,
                     "tool_calls": [{"id": f"c{i}", "type": "function",
                                     "function": {"name": "list_dir",
                                                  "arguments": "{}"}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}",
                     "content": "x" * 300})
    out = compress_messages(msgs, None, threshold_tokens=10,
                            keep_recent_tokens=250)
    body = [m for m in out if not str(m["content"] or "").startswith(SUMMARY_PREFIX)
            and m.get("role") != "system"]
    assert body[0]["role"] != "tool"


# ---------------------------------------------------------- 3. todo tools

@pytest.fixture()
def uf_home(tmp_path, monkeypatch):
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(tmp_path / "uf"))
    return tmp_path / "uf"


def test_todo_roundtrip_and_merge(uf_home):
    assert uf_tools.todo_read() == "(todo list is empty)"
    out = uf_tools.todo_write(todos=[
        {"id": "a", "content": "first", "status": "pending"},
        {"id": "b", "content": "second"},
    ])
    assert "[ ] a: first" in out and "[?] b: second" not in out  # b defaults pending
    out = uf_tools.todo_write(todos=[{"id": "a", "content": "first",
                                      "status": "in_progress"}], merge=True)
    assert "[~] a: first" in out and "[ ] b: second" in out
    # replace semantics drop old items
    out = uf_tools.todo_write(todos=[{"id": "z", "content": "only",
                                      "status": "done"}])
    assert out.strip() == "[x] z: only"


def test_todo_validation(uf_home):
    # deliberately wrong type: the tool must reject non-list input gracefully
    assert uf_tools.todo_write(todos=cast(Any, "nope")).startswith("error")
    assert uf_tools.todo_write(todos=[{"content": ""}]).startswith("error")


def test_todo_salvages_markdown_strings(uf_home):
    """Weak models emit '- [ ] text' checklist lines instead of objects."""
    out = uf_tools.todo_write(todos=["- [ ] map source tree",
                                     "[x] done thing",
                                     "plain item"])
    assert "map source tree" in out and "done thing" in out
    assert "plain item" in out
    # status marker parsed from the checkbox -> line renders as "[x] …"
    assert any(l.startswith("[x] ") and "done thing" in l
               for l in out.splitlines())
    # oversized list refused, existing state untouched
    many = [{"content": f"t{i}"} for i in range(uf_tools.TODO_MAX_ITEMS + 1)]
    assert uf_tools.todo_write(todos=many).startswith("error")


# ------------------------------------------------------- 4. search_traces

def test_search_traces_over_buffer(uf_home):
    from toaster_loop.store import TraceRecorder

    rec = TraceRecorder(home=uf_home)
    row = rec.record(
        task="rename the ffmpeg output files",
        messages=[{"role": "user", "content": "rename the ffmpeg output files"},
                  {"role": "assistant", "content": "used mmv to glob-rename"}],
        backend="toaster_loop", model_alias="m", base_url="http://x/v1",
        completed=True,
    )
    rec.close()
    assert row is not None

    out = uf_tools.search_traces(pattern="ffmpeg")
    assert row.trace_id in out and "[success]" in out
    assert "no matching traces" in uf_tools.search_traces(pattern="zzz-nope-zzz")
    assert uf_tools.search_traces(status="bogus").startswith("error")
    # empty pattern lists recent
    assert row.trace_id in uf_tools.search_traces()


def test_search_traces_empty_buffer(uf_home):
    assert "no traces" in uf_tools.search_traces(pattern="anything")


def test_tool_schemas_expose_new_tools():
    names = {s["function"]["name"] for s in uf_tools.TOOL_SCHEMAS}
    assert {"todo_write", "todo_read", "search_traces"} <= names
    # Every impl is either a default schema or capability-gated; the
    # capability tools must NOT be in the default set (per-agent opt-in).
    assert set(names) | set(uf_tools.CAPABILITY_TOOLS) == set(uf_tools.TOOL_IMPLS)
    assert not names & set(uf_tools.CAPABILITY_TOOLS)
    assert set(uf_tools.CAPABILITY_TOOL_SCHEMAS) == set(uf_tools.CAPABILITY_TOOLS)


# ------------------------------------- 5. masking helpers (no torch needed)

class FakeTok:
    """Minimal chat-template tokenizer over a fixed unigram vocab."""

    pad_token_id = 0
    eos_token = " "
    is_fast = True

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False):
        out = "".join(f"[{m['role']}]{m['content']}" for m in messages)
        if add_generation_prompt:
            out += "[assistant]"
        return out

    def __call__(self, text, truncation=False, max_length=None,
                 return_offsets_mapping=False, add_special_tokens=True):
        # tokens = each char; offsets = exact char positions
        toks = list(text)
        enc: dict = {"input_ids": [ord(c) for c in toks],
                     "attention_mask": [1] * len(toks)}
        if return_offsets_mapping:
            enc["offset_mapping"] = [(i, i + 1) for i in range(len(toks))]
        return enc


def test_assistant_char_regions_and_labels():
    tok = FakeTok()
    convs = [
        {"from": "system", "value": "sys"},
        {"from": "human", "value": "hello"},
        {"from": "gpt", "value": "WORLD"},
        {"from": "human", "value": "again"},
        {"from": "gpt", "value": "END"},
    ]
    regions = assistant_char_regions(tok, convs)
    assert regions is not None and len(regions) == 2
    full = tok.apply_chat_template(sharegpt_to_messages(convs), tokenize=False)
    # The generation-prompt header stays in the prefix; the supervised region
    # is exactly the assistant content.
    assert {full[s:e] for s, e in regions} == {"WORLD", "END"}
    for start, _end in regions:
        assert full[:start].endswith("[assistant]")

    enc = tokenize_with_labels(tok, convs, max_length=10_000,
                               mask_instructions=True)
    assert enc is not None
    labels = enc["labels"]
    assert len(labels) == len(enc["input_ids"])
    # the response letters are supervised...
    text = full
    for i, (tok_id, label) in enumerate(zip(enc["input_ids"], labels)):
        ch = chr(tok_id)
        if ch in "WORLD" and text[i] == ch:
            assert label != -100
    # ...and instruction text is not
    for i, ch in enumerate(text):
        if ch == "h" and text[i - 1 : i + 5] == "[human":
            assert labels[i] == -100


def test_tokenize_with_labels_noncomposing_template():
    class WeirdTok(FakeTok):
        def apply_chat_template(self, messages, tokenize=False,
                                add_generation_prompt=False):
            base = super().apply_chat_template(messages, tokenize,
                                               add_generation_prompt)
            return f"{len(messages)}::{base}"  # prefix length varies

    convs = [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "yo"}]
    assert assistant_char_regions(WeirdTok(), convs) is None
    assert tokenize_with_labels(WeirdTok(), convs, 100, True) is None

# ------------------------------------------------------------------------
# from tests/test_engines.py
# ------------------------------------------------------------------------

@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "uf-engines"
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(h))
    return h


@pytest.fixture()
def app(home):
    return DashboardApp(ToasterLoopConfig(), home=home)


def err_code(r) -> str:
    return (r.body.get("error") or {}).get("code", "")


# ------------------------------------------------------------- registry core

def test_builtin_engine_always_listed(home):
    engines = eng_mod.list_engines(home)
    assert engines[0]["id"] == eng_mod.BUILTIN_ID
    assert engines[0]["builtin"] is True
    # builtin is never persisted
    assert not (home / "serving" / "engines").exists()


def test_validate_engine_body_defaults_and_normalization():
    e = eng_mod.validate_engine_body({
        "name": "vLLM", "binary": "vllm",
        "args_template": "--model {model} --port {port}",
    })
    assert e["kind"] == "openai"
    assert e["binary"] == "vllm"
    # string templates are shlex-split
    assert e["args_template"] == ["--model", "{model}", "--port", "{port}"]

    # openai kind with no template => loopback-safe default template
    e2 = eng_mod.validate_engine_body({"name": "X", "binary": "x-server"})
    assert "{model}" in " ".join(e2["args_template"])
    assert "127.0.0.1" in " ".join(e2["args_template"])

    # kind other => no default template (inventory only)
    e3 = eng_mod.validate_engine_body({"name": "SD", "binary": "sd-cli",
                                       "kind": "other"})
    assert e3["args_template"] == []


def test_validate_engine_body_rejects_bad_input():
    with pytest.raises(ServeError):
        eng_mod.validate_engine_body({"binary": "x"})           # no name
    with pytest.raises(ServeError):
        eng_mod.validate_engine_body({"name": "n"})             # no binary
    with pytest.raises(ServeError):
        eng_mod.validate_engine_body({"name": "n", "binary": "b",
                                      "kind": "magic"})
    with pytest.raises(ServeError):
        eng_mod.validate_engine_body({"name": "n", "binary": "b",
                                      "env": "not-a-dict"})


def test_builtin_id_reserved(home):
    with pytest.raises(ServeError):
        eng_mod.validate_engine_body({"id": eng_mod.BUILTIN_ID,
                                      "name": "n", "binary": "b"})
    with pytest.raises(ServeError):
        eng_mod.delete_engine_file(home, eng_mod.BUILTIN_ID)


def test_save_get_delete_roundtrip(home):
    e = eng_mod.validate_engine_body({"name": "vLLM", "binary": "vllm"})
    e["id"] = eng_mod.new_engine_id("vLLM", ["llama-cpp"])
    assert e["id"] == "vLLM"
    eng_mod.save_engine(home, e)
    again = eng_mod.get_engine(home, "vLLM")
    assert again and again["binary"] == "vllm"
    ids = [x["id"] for x in eng_mod.list_engines(home)]
    assert ids == ["llama-cpp", "vLLM"]
    assert eng_mod.delete_engine_file(home, "vLLM") is True
    assert eng_mod.get_engine(home, "vLLM") is None


def test_new_engine_id_avoids_builtin_and_collisions():
    taken = ["llama-cpp", "vllm"]
    assert eng_mod.new_engine_id("llama-cpp", taken) != "llama-cpp"
    assert eng_mod.new_engine_id("vllm", taken) == "vllm-2"


def test_probe_missing_binary(home, monkeypatch):
    e = eng_mod.validate_engine_body({"name": "X", "binary": "definitely-not-here"})
    monkeypatch.setattr(eng_mod.shutil, "which", lambda _n: None)
    assert eng_mod.probe(e) == {"found": False, "path": None}


def test_probe_finds_binary(home, monkeypatch):
    e = eng_mod.validate_engine_body({"name": "X", "binary": "x-server"})
    monkeypatch.setattr(eng_mod.shutil, "which",
                        lambda _n: "/usr/local/bin/x-server")
    p = eng_mod.probe(e)
    assert p["found"] and p["path"] == "/usr/local/bin/x-server"


# ------------------------------------------------------------- command build

def test_build_engine_cmd_substitutions():
    e = {"kind": "openai", "args_template":
         ["serve", "--model", "{model}", "--host", "{host}",
          "--port", "{port}", "--literal", "{unknown}"]}
    cmd = eng_mod.build_engine_cmd(e, "/bin/x", model="/m/model.gguf",
                                   host="127.0.0.1", port=8123)
    assert cmd == ["/bin/x", "serve", "--model", "/m/model.gguf",
                   "--host", "127.0.0.1", "--port", "8123",
                   "--literal", "{unknown}"]


def test_build_engine_cmd_default_template():
    cmd = eng_mod.build_engine_cmd({"kind": "openai"}, "/bin/x",
                                   model="m.gguf", host="127.0.0.1", port=9)
    assert cmd[0] == "/bin/x"
    assert "{model}" not in " ".join(cmd)
    assert "m.gguf" in cmd and "9" in cmd


def test_validate_diffusion_kind_gets_default_template():
    e = eng_mod.validate_engine_body({"name": "SD", "binary": "sd-server",
                                      "kind": "diffusion"})
    assert e["kind"] == "diffusion"
    # loopback-safe sd-server defaults (no /v1; --listen-ip/--listen-port)
    assert "--model" in e["args_template"]
    assert "127.0.0.1" in " ".join(e["args_template"])
    assert "--listen-port" in e["args_template"]


def test_build_engine_cmd_diffusion_default_template():
    cmd = eng_mod.build_engine_cmd({"kind": "diffusion"}, "/bin/sd-server",
                                   model="m.gguf", host="127.0.0.1", port=9)
    assert cmd[0] == "/bin/sd-server"
    assert "--model" in cmd and "m.gguf" in cmd
    assert "--listen-ip" in cmd and "127.0.0.1" in cmd
    assert "--listen-port" in cmd and "9" in cmd


def test_engine_env_stringifies():
    assert eng_mod.engine_env({"env": {"A": 1, "B": None}}) == {"A": "1"}
    assert eng_mod.engine_env({}) == {}


# ------------------------------------------------------- instance binding

def test_instance_body_engine_defaults_to_none():
    inst = srv_inst.validate_instance_body(
        {"name": "s", "gguf": "/tmp/x.gguf"})
    assert inst["engine"] is None


def test_instance_body_llama_cpp_alias_is_default_path():
    inst = srv_inst.validate_instance_body(
        {"name": "s", "gguf": "/tmp/x.gguf", "engine": "llama-cpp"})
    assert inst["engine"] is None


def test_instance_body_keeps_registered_engine():
    inst = srv_inst.validate_instance_body(
        {"name": "s", "gguf": "/tmp/x.gguf", "engine": "vllm"})
    assert inst["engine"] == "vllm"


def test_instance_body_image_settings_validated():
    inst = srv_inst.validate_instance_body({
        "name": "art", "gguf": "/tmp/sdxl.gguf", "engine": "sd-cpp",
        "image_settings": {"steps": 28, "cfg_scale": 4.5, "width": 1024,
                           "height": 1024, "sampler": "euler_a"}})
    assert inst["image_settings"]["steps"] == 28
    assert inst["image_settings"]["cfg_scale"] == 4.5
    assert inst["image_settings"]["sampler"] == "euler_a"
    with pytest.raises(ServeError):
        srv_inst.validate_instance_body({
            "name": "art", "gguf": "/tmp/x.gguf",
            "image_settings": {"steps": 999}})      # out of range
    with pytest.raises(ServeError):
        srv_inst.validate_instance_body({
            "name": "art", "gguf": "/tmp/x.gguf",
            "image_settings": {"top_k": 5}})          # not an image knob


def test_default_instance_reads_serve_default_engine():
    cfg = ToasterLoopConfig()
    assert srv_inst.default_instance(cfg)["engine"] is None
    cfg.serve = {"default_engine": "vllm"}
    assert srv_inst.default_instance(cfg)["engine"] == "vllm"
    cfg.serve = {"default_engine": "  "}   # blank => built-in path
    assert srv_inst.default_instance(cfg)["engine"] is None


def test_engines_in_use(home):
    e = eng_mod.validate_engine_body({"name": "vLLM", "binary": "vllm"})
    e["id"] = "vllm"
    eng_mod.save_engine(home, e)
    srv_inst.save_instance(home, {"id": "coder", "name": "coder",
                                  "gguf": "/tmp/x.gguf", "engine": "vllm"})
    assert eng_mod.engines_in_use(home, "vllm") == ["coder"]
    assert eng_mod.engines_in_use(home, "other") == []


# ---------------------------------------------------------- dashboard API

def test_engines_routes_registered(app):
    paths = {(m, p) for (m, p, _) in app.ROUTES}
    for m, p in [("GET", "/api/engines"), ("POST", "/api/engines"),
                 ("PUT", "/api/engines/<engine_id>"),
                 ("DELETE", "/api/engines/<engine_id>")]:
        assert (m, p) in paths, f"{m} {p} missing"


def test_engines_list_builtin_first(app, home):
    r = app.handle("GET", "/api/engines")
    assert r.status == 200
    ids = [e["id"] for e in r.body["engines"]]
    assert ids[0] == "llama-cpp"
    assert r.body["default_engine"] is None


def test_engines_create_update_delete(app, home):
    r = app.handle("POST", "/api/engines", body={
        "name": "vLLM", "binary": "vllm",
        "args_template": "--model {model} --port {port}"})
    assert r.status == 201, r.body
    eid = r.body["engine"]["id"]
    assert eid == "vLLM"

    upd = app.handle("PUT", f"/api/engines/{eid}", body={"notes": "local"})
    assert upd.status == 200 and upd.body["engine"]["notes"] == "local"

    lst = app.handle("GET", "/api/engines")
    row = next(e for e in lst.body["engines"] if e["id"] == eid)
    assert row["installed"] is False  # binary absent in the test env

    dele = app.handle("DELETE", f"/api/engines/{eid}")
    assert dele.status == 200
    ids = [e["id"] for e in app.handle("GET", "/api/engines").body["engines"]]
    assert eid not in ids


def test_engines_builtin_is_immutable(app, home):
    bad = app.handle("PUT", "/api/engines/llama-cpp", body={"name": "x"})
    assert bad.status == 400 and err_code(bad) == "reserved_engine"
    bad2 = app.handle("DELETE", "/api/engines/llama-cpp")
    assert bad2.status == 400 and err_code(bad2) == "reserved_engine"


def test_engines_delete_blocked_while_in_use(app, home):
    r = app.handle("POST", "/api/engines", body={"name": "vLLM",
                                                 "binary": "vllm"})
    eid = r.body["engine"]["id"]
    srv_inst.save_instance(home, {"id": "coder", "name": "coder",
                                  "gguf": "/tmp/x.gguf", "engine": eid})
    dele = app.handle("DELETE", f"/api/engines/{eid}")
    assert dele.status == 409 and err_code(dele) == "in_use"


def test_instance_create_rejects_unknown_engine(app, home):
    r = app.handle("POST", "/api/servers", body={
        "name": "coder", "gguf": "/tmp/x.gguf", "engine": "ghost"})
    assert r.status == 400 and err_code(r) == "unknown_engine"


def test_instance_create_accepts_registered_engine(app, home):
    e = app.handle("POST", "/api/engines", body={"name": "vLLM",
                                                 "binary": "vllm"})
    eid = e.body["engine"]["id"]
    r = app.handle("POST", "/api/servers", body={
        "name": "coder", "gguf": "/tmp/x.gguf", "engine": eid})
    assert r.status == 201, r.body
    assert r.body["server"]["engine"] == eid


def test_config_put_serve_default_engine(app, home):
    """serve.default_engine is a recognized serve key (Settings page writes
    it through the section patch)."""
    r = app.handle("PUT", "/api/config", body={
        "section": "serve", "patch": {"default_engine": "vllm"}})
    assert r.status == 200, r.body
    raw = (home / "toaster_loop.yaml").read_text()
    assert "default_engine: vllm" in raw


# ------------------------------------------------------- spawn error paths

def test_engine_spawn_unknown_engine_error(home, monkeypatch):
    from toaster_loop.serving.supervisor import _engine_spawn_cmd

    cfg = ToasterLoopConfig()
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(home))
    with pytest.raises(ServeError, match="unknown inference engine"):
        _engine_spawn_cmd(cfg, {}, "ghost", gguf="m.gguf", port=1)


def test_engine_spawn_non_openai_refuses(home, monkeypatch):
    from toaster_loop.serving.supervisor import _engine_spawn_cmd

    e = eng_mod.validate_engine_body({"name": "SD", "binary": "sd-cli",
                                      "kind": "other"})
    e["id"] = "sd"
    eng_mod.save_engine(home, e)
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(home))
    with pytest.raises(ServeError, match="only"):
        _engine_spawn_cmd(ToasterLoopConfig(), {}, "sd", gguf="m.gguf", port=1)


def _fake_sd_server(home, monkeypatch):
    exe = home / "sd-server"
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_text("#!/bin/sh\n")
    os.chmod(exe, 0o755)
    e = eng_mod.validate_engine_body({"name": "SD", "binary": str(exe),
                                      "kind": "diffusion"})
    e["id"] = "sd-cpp"
    eng_mod.save_engine(home, e)
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(home))
    return exe


def test_engine_spawn_diffusion_appends_image_settings(home, monkeypatch):
    """A diffusion spawn carries the instance's generation defaults as
    sd-server flags (steps/cfg/size/sampler), not llama.cpp run settings."""
    from toaster_loop.serving.supervisor import _engine_spawn_cmd

    exe = _fake_sd_server(home, monkeypatch)
    inst = {"image_settings": {"steps": 30, "cfg_scale": 6.5,
                               "width": 768, "height": 1024,
                               "sampler": "dpm++2m", "seed": -1}}
    cmd, env, cwd, kind = _engine_spawn_cmd(
        ToasterLoopConfig(), inst, "sd-cpp", gguf="/m/sdxl.gguf", port=1234)
    assert kind == "diffusion" and cwd is None
    joined = " ".join(cmd)
    assert cmd[0] == str(exe)
    assert "--steps 30" in joined and "--cfg-scale 6.5" in joined
    assert "-W 768" in joined and "-H 1024" in joined
    assert "--sampling-method dpm++2m" in joined and "-s -1" in joined


def test_engine_spawn_diffusion_without_image_settings(home, monkeypatch):
    from toaster_loop.serving.supervisor import _engine_spawn_cmd

    exe = _fake_sd_server(home, monkeypatch)
    cmd, env, cwd, kind = _engine_spawn_cmd(
        ToasterLoopConfig(), {}, "sd-cpp", gguf="/m/x.gguf", port=1)
    assert kind == "diffusion"
    assert "--steps" not in cmd  # nothing set => server defaults


def test_engine_spawn_openai_kind_reports_kind(home, monkeypatch):
    """An openai-kind engine reports its kind so the health wait targets
    /v1/models (not the diffusion root probe)."""
    from toaster_loop.serving.supervisor import _engine_spawn_cmd

    exe = home / "x-server"
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_text("#!/bin/sh\n")
    os.chmod(exe, 0o755)
    e = eng_mod.validate_engine_body({"name": "X", "binary": str(exe),
                                      "kind": "openai"})
    e["id"] = "x"
    eng_mod.save_engine(home, e)
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(home))
    _, _, _, kind = _engine_spawn_cmd(ToasterLoopConfig(), {}, "x",
                                      gguf="m.gguf", port=1)
    assert kind == "openai"


def test_engine_spawn_uses_engine_cwd(home, monkeypatch):
    """strata-style engines run from their checkout (cwd on the record)."""
    from toaster_loop.serving.supervisor import _engine_spawn_cmd

    exe = home / "python"
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_text("#!/bin/sh\n")
    os.chmod(exe, 0o755)
    e = eng_mod.validate_engine_body({
        "name": "Strata", "binary": str(exe), "kind": "openai",
        "args_template": ["-m", "serve.server", "--config", "{model}"],
        "cwd": str(home)})
    e["id"] = "strata"
    eng_mod.save_engine(home, e)
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(home))
    _, _, cwd, kind = _engine_spawn_cmd(ToasterLoopConfig(), {}, "strata",
                                        gguf="/c/run.json", port=1)
    assert cwd == str(home) and kind == "openai"


def test_engine_spawn_missing_binary_error(home, monkeypatch):
    from toaster_loop.serving.supervisor import _engine_spawn_cmd

    e = eng_mod.validate_engine_body({"name": "vLLM", "binary": "vllm"})
    e["id"] = "vllm"
    eng_mod.save_engine(home, e)
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(home))
    monkeypatch.setattr(eng_mod.shutil, "which", lambda _n: None)
    with pytest.raises(ServeError, match="binary not found"):
        _engine_spawn_cmd(ToasterLoopConfig(), {}, "vllm", gguf="m.gguf",
                          port=1)


def test_instance_api_roundtrips_image_settings(app, home):
    """A diffusion instance stores image_settings through the servers API."""
    eng = app.handle("POST", "/api/engines", body={
        "name": "SD", "binary": "sd-server", "kind": "diffusion"})
    eid = eng.body["engine"]["id"]
    r = app.handle("POST", "/api/servers", body={
        "name": "art", "gguf": "/tmp/sdxl.gguf", "engine": eid,
        "image_settings": {"steps": 28, "cfg_scale": 4.5}})
    assert r.status == 201, r.body
    assert r.body["server"]["image_settings"] == {"steps": 28,
                                                  "cfg_scale": 4.5}
    upd = app.handle("PUT", "/api/servers/art", body={
        "image_settings": {"steps": 12}})
    assert upd.status == 200 and \
        upd.body["server"]["image_settings"] == {"steps": 12}

# ------------------------------------------------------------------------
# from tests/test_engine_installs.py
# ------------------------------------------------------------------------

@pytest.fixture()
def home__test_engine_installs(tmp_path, monkeypatch):
    h = tmp_path / "uf-installs"
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(h))
    return h


@pytest.fixture()
def app__test_engine_installs(home__test_engine_installs):
    return DashboardApp(ToasterLoopConfig(), home=home__test_engine_installs)




def fake_host(system="linux", backend="cpu", gpus=()):
    """Host profile shaped like ei.detect_host()'s output. Passing it
    explicitly keeps every test hermetic — no vendor CLI probes."""
    return {"system": system, "machine": "x86_64", "backend": backend,
            "gpus": list(gpus)}


@pytest.fixture()
def cached_host(monkeypatch):
    """Pin the process-level host cache so install_recipe()'s detect_host()
    never shells out to nvidia-smi/rocminfo/vulkaninfo in tests. Individual
    tests override by passing host= or re-patching ei.detect_host."""
    monkeypatch.setattr(ei, "_HOST_CACHE", fake_host())


# ------------------------------------------------------------------ catalog

def test_catalog_contains_the_four_engines(cached_host):
    ids = ei.catalog_ids()
    assert ids == ["vllm", "airllm", "strata", "sd-cpp"]
    vllm = ei.get_recipe("vllm")
    assert vllm["kind"] == "openai"
    assert "{model}" in " ".join(vllm["args_template"])
    assert "127.0.0.1" not in " ".join(vllm["args_template"])  # host via {host}
    # strata's serve/server.py is an OpenAI-compatible server launched from
    # its own run-config JSON (model_kind json), not a GGUF path
    strata = ei.get_recipe("strata")
    assert strata["kind"] == "openai" and strata["model_kind"] == "json"
    assert strata["args_template"][:2] == ["-m", "serve.server"]
    # sd-cpp spawns as a diffusion image server (sd-server)
    sd = ei.get_recipe("sd-cpp")
    assert sd["kind"] == "diffusion" and sd["model_kind"] == "gguf"
    assert "--listen-port" in " ".join(sd["args_template"])
    # AirLLM is a library — inventory-only (never spawned by toaster_loop)
    assert ei.get_recipe("airllm")["kind"] == "other"
    with pytest.raises(KeyError):
        ei.get_recipe("nope")


def test_catalog_recipes_validate_as_engine_bodies(cached_host):
    """Every recipe must produce a body the engine registry accepts."""
    for r in ei.CATALOG:
        st = {"binary": "/opt/x/bin/thing", "installed_at": "2026-01-01T00:00:00Z"}
        body = ei.engine_body_for(r["id"], st)
        eng = eng_mod.validate_engine_body(body)
        assert eng["name"] == r["name"]
        assert eng["kind"] == r["kind"]


def test_paths_under_home(home__test_engine_installs, cached_host):
    assert str(ei.venv_dir(home__test_engine_installs, "vllm")).startswith(str(home__test_engine_installs))
    assert str(ei.src_dir(home__test_engine_installs, "sd-cpp")).startswith(str(home__test_engine_installs))
    assert str(ei.state_path(home__test_engine_installs, "airllm")).startswith(str(home__test_engine_installs))
    with pytest.raises(KeyError):
        ei.engine_dir(home__test_engine_installs, "escape")


# ------------------------------------------------------------ state file

def test_state_roundtrip_and_installed_probe(home__test_engine_installs, cached_host):
    assert ei.read_state(home__test_engine_installs, "vllm") is None
    assert ei.recipe_installed(home__test_engine_installs, "vllm") is False

    ei.write_state(home__test_engine_installs, "vllm", {"binary": str(home__test_engine_installs / "missing"), "x": 1})
    st = ei.read_state(home__test_engine_installs, "vllm")
    assert st is not None and st["x"] == 1
    # state recorded but the binary vanished => not installed
    assert ei.recipe_installed(home__test_engine_installs, "vllm") is False

    b = home__test_engine_installs / "real-binary"
    b.write_text("#!/bin/sh\n")
    ei.write_state(home__test_engine_installs, "vllm", {"binary": str(b)})
    assert ei.recipe_installed(home__test_engine_installs, "vllm") is True


def test_list_catalog_probes_without_subprocess(home__test_engine_installs, monkeypatch, cached_host):
    def _boom(*a, **k):  # listing must never shell out
        raise AssertionError("listing endpoint ran a subprocess")
    monkeypatch.setattr(ei.subprocess, "Popen", _boom)
    monkeypatch.setattr(ei.subprocess, "run", _boom)
    rows = ei.list_catalog(home__test_engine_installs)
    assert [r["id"] for r in rows] == ei.catalog_ids()
    assert all(r["installed"] is False for r in rows)


# ------------------------------------------------------- install (fake shell)

class FakeShell:
    """Records every command; fakes success and materializes outputs.

    ``fail_when(cmd)`` lets a test fail one specific step (e.g. pip) while
    everything else succeeds."""

    def __init__(self, home: Path, fail_when=None):
        self.home = home
        self.cmds: list[list[str]] = []
        self.fail_when = fail_when or (lambda _c: False)

    def stream(self, cmd, log, cwd=None):
        self.cmds.append([str(c) for c in cmd])
        joined = " ".join(str(c) for c in cmd)
        if self.fail_when(cmd):
            log(f"[fake] FAIL {joined}")
            return 1
        # venv creation actually creates the interpreter path expected later
        if "-m venv" in joined:
            dest = Path(cmd[-1])
            bindir = "Scripts" if os.name == "nt" else "bin"
            py = dest / bindir / ("python.exe" if os.name == "nt" else "python")
            py.parent.mkdir(parents=True, exist_ok=True)
            py.write_text("#!/bin/sh\n")
        # a clone materializes the checkout (with .git so update reuses it)
        if "clone" in cmd:
            dest = Path(cmd[-1])
            (dest / ".git").mkdir(parents=True, exist_ok=True)
        # a cmake build materializes the expected binary next to the build dir
        if "--build" in cmd:
            build_dir = Path(str(cmd[cmd.index("--build") + 1]))
            bindir = build_dir / "bin"
            bindir.mkdir(parents=True, exist_ok=True)
            exe = bindir / ("sd-cli.exe" if os.name == "nt" else "sd-cli")
            exe.write_text("#!/bin/sh\n")
            os.chmod(exe, 0o755)
        log(f"[fake] {joined}")
        return 0

    def stream_out(self, cmd):
        self.cmds.append([str(c) for c in cmd])
        if "rev-parse" in joined_of(cmd):
            return "deadbee"
        return ""

    def require(self, tool, hint):
        return f"/usr/bin/{tool}"


def joined_of(cmd) -> str:
    return " ".join(str(c) for c in cmd)


@pytest.fixture()
def fake_shell__test_engine_installs(home__test_engine_installs, monkeypatch):
    fs = FakeShell(home__test_engine_installs)
    monkeypatch.setattr(ei, "_stream", fs.stream)
    monkeypatch.setattr(ei, "_stream_out", fs.stream_out)
    monkeypatch.setattr(ei, "_require", fs.require)
    return fs


def test_install_pip_recipe_creates_venv_and_state(home__test_engine_installs, fake_shell__test_engine_installs, cached_host):
    st = ei.install_recipe("vllm", home=home__test_engine_installs, log_fn=lambda _s: None)
    assert st["method"] == "pip"
    assert Path(st["binary"]).is_file()  # faked venv python exists
    assert "venv" in st["binary"] and "engines" in st["binary"]
    assert ei.recipe_installed(home__test_engine_installs, "vllm") is True
    joined = "\n".join(" ".join(c) for c in fake_shell__test_engine_installs.cmds)
    assert "-m venv" in joined
    assert "pip" in joined and "install" in joined and "vllm" in joined
    # pip NEVER runs against the toaster_loop interpreter — only the venv python
    pip_cmds = [c for c in fake_shell__test_engine_installs.cmds if "pip" in c]
    assert pip_cmds
    venv_py = str(ei.venv_python(home__test_engine_installs, "vllm"))
    for c in pip_cmds:
        assert c[0] == venv_py, f"pip ran outside the private venv: {c}"
    assert sys.executable not in [c[0] for c in pip_cmds]


def test_install_pip_failure_raises_install_error(home__test_engine_installs, fake_shell__test_engine_installs, monkeypatch, cached_host):
    # only the pip step fails; venv creation succeeds
    fs = FakeShell(home__test_engine_installs, fail_when=lambda c: "pip" in c)
    monkeypatch.setattr(ei, "_stream", fs.stream)
    with pytest.raises(ei.InstallError, match="pip install"):
        ei.install_recipe("vllm", home=home__test_engine_installs, log_fn=lambda _s: None)
    assert ei.recipe_installed(home__test_engine_installs, "vllm") is False
    assert ei.read_state(home__test_engine_installs, "vllm") is None  # no state on failure


def test_install_git_pip_recipe_clones_editable(home__test_engine_installs, fake_shell__test_engine_installs, monkeypatch, cached_host):
    # A pip-installable checkout (setuptools setup.py) still installs editable.
    monkeypatch.setattr(ei, "_stream", _app_style_clone(
        fake_shell__test_engine_installs, "strata",
        "from setuptools import setup\nsetup(name='strata')\n"))
    st = ei.install_recipe("strata", home=home__test_engine_installs, log_fn=lambda _s: None)
    joined = "\n".join(" ".join(c) for c in fake_shell__test_engine_installs.cmds)
    assert "git clone" in joined and "niko1221/strata" in joined
    assert "pip" in joined and "-e" in joined
    assert st["commit"] == "deadbee"
    assert st["variant"] == "editable"
    assert Path(st["binary"]).is_file()


def test_install_git_pip_requirements_file_uses_dash_r(home__test_engine_installs, fake_shell__test_engine_installs,
                                                       monkeypatch, cached_host):
    """A requirements.txt inside the checkout must install via `-r <file>`.

    Regression: passing the path as a positional spec makes pip fail with
    "Invalid requirement: … Expected package name at the start of dependency
    specifier" (the exact strata install failure)."""
    req = ei.src_dir(home__test_engine_installs, "strata") / "requirements.txt"

    def stream_with_req(cmd, log, cwd=None):
        rc = fake_shell__test_engine_installs.stream(cmd, log, cwd=cwd)
        if "clone" in cmd:  # clone materialized the checkout — add its reqs
            req.parent.mkdir(parents=True, exist_ok=True)
            req.write_text("numpy\n")
        return rc
    monkeypatch.setattr(ei, "_stream", stream_with_req)

    st = ei.install_recipe("strata", home=home__test_engine_installs, log_fn=lambda _s: None)
    pip_cmds = [c for c in fake_shell__test_engine_installs.cmds if "pip" in c and "install" in c]
    req_cmds = [c for c in pip_cmds if str(req) in " ".join(c)]
    assert req_cmds, f"requirements file never installed: {pip_cmds}"
    for c in req_cmds:
        assert "-r" in c, f"requirements path passed positionally: {c}"
    assert st["method"] == "git_pip"


def _app_style_clone(fs, rid, setup_body):
    """After a fake clone materializes the checkout, drop in an application-
    style setup.py (argparse CLI, no setuptools — the strata repo shape)."""
    def stream(cmd, log, cwd=None):
        rc = fs.stream(cmd, log, cwd=cwd)
        if "clone" in cmd:
            sdir = Path(cmd[-1])
            (sdir / "setup.py").write_text(setup_body, encoding="utf-8")
        return rc
    return stream


APP_STYLE_SETUP_PY = """#!/usr/bin/env python3
import argparse
p = argparse.ArgumentParser()
p.add_argument("--family", choices=["qwen", "swift"])
args = p.parse_args()
print("run the server")
"""

SETUPTOOLS_STYLE_SETUP_PY = """from setuptools import setup
setup(name="thing", version="0.1")
"""


def test_is_pip_installable_detects_app_style_setup_py(home__test_engine_installs, cached_host):
    sdir = ei.src_dir(home__test_engine_installs, "strata")
    sdir.mkdir(parents=True)
    assert ei._is_pip_installable(sdir) is False          # nothing there
    (sdir / "setup.py").write_text(APP_STYLE_SETUP_PY)
    assert ei._is_pip_installable(sdir) is False          # argparse-only CLI
    (sdir / "pyproject.toml").write_text("[project]\nname='x'\n")
    assert ei._is_pip_installable(sdir) is True           # pyproject wins


def test_install_app_style_repo_skips_editable(home__test_engine_installs, fake_shell__test_engine_installs, monkeypatch, cached_host):
    """strata regression: `pip install -e .` on an application setup.py dies
    with `setup.py: error: unrecognized arguments: egg_info`. Install must
    succeed as checkout+deps instead."""
    monkeypatch.setattr(ei, "_stream", _app_style_clone(fake_shell__test_engine_installs, "strata",
                                                        APP_STYLE_SETUP_PY))
    st = ei.install_recipe("strata", home=home__test_engine_installs, log_fn=lambda _s: None)
    joined = "\n".join(" ".join(c) for c in fake_shell__test_engine_installs.cmds)
    assert "-e" not in joined  # editable install never attempted
    assert st["method"] == "git_pip" and st["variant"] == "checkout"
    assert Path(st["binary"]).is_file()
    assert ei.recipe_installed(home__test_engine_installs, "strata") is True


def test_install_editable_egg_info_failure_downgrades(home__test_engine_installs, fake_shell__test_engine_installs,
                                                      monkeypatch, cached_host):
    """Even when detection passes (pyproject present), an egg_info failure at
    pip runtime downgrades to checkout instead of erroring."""
    def egg_failing_stream(cmd, log, cwd=None):
        rc = fake_shell__test_engine_installs.stream(cmd, log, cwd=cwd)
        if "clone" in cmd:  # pip-installable metadata on disk
            (Path(cmd[-1]) / "pyproject.toml").write_text(
                "[build-system]\nrequires=['setuptools']\n")
        if "-e" in [str(c) for c in cmd]:  # exact arg, not a path substring
            log("setup.py: error: unrecognized arguments: egg_info")
            return 1
        return rc
    monkeypatch.setattr(ei, "_stream", egg_failing_stream)

    st = ei.install_recipe("strata", home=home__test_engine_installs, log_fn=lambda _s: None)
    assert st["variant"] == "checkout"
    assert ei.recipe_installed(home__test_engine_installs, "strata") is True


def test_install_editable_other_failure_still_raises(home__test_engine_installs, fake_shell__test_engine_installs,
                                                     monkeypatch, cached_host):
    """A normal pip failure (network, build error…) must NOT be swallowed by
    the egg_info downgrade."""
    def failing_stream(cmd, log, cwd=None):
        rc = fake_shell__test_engine_installs.stream(cmd, log, cwd=cwd)
        if "clone" in cmd:
            (Path(cmd[-1]) / "pyproject.toml").write_text("[project]\n")
        if "-e" in [str(c) for c in cmd]:  # exact arg, not a path substring
            log("error: connection timed out")
            return 1
        return rc
    monkeypatch.setattr(ei, "_stream", failing_stream)

    with pytest.raises(ei.InstallError, match="pip install"):
        ei.install_recipe("strata", home=home__test_engine_installs, log_fn=lambda _s: None)


def test_install_git_cmake_recipe_builds_binary(home__test_engine_installs, fake_shell__test_engine_installs, cached_host):
    st = ei.install_recipe("sd-cpp", home=home__test_engine_installs, log_fn=lambda _s: None)
    joined = "\n".join(" ".join(c) for c in fake_shell__test_engine_installs.cmds)
    assert "--recursive" in joined  # ggml submodule
    assert "cmake" in joined and "--build" in joined
    assert st["binary"].endswith("sd-cli" + (".exe" if os.name == "nt" else ""))
    assert ei.recipe_installed(home__test_engine_installs, "sd-cpp") is True


def test_install_git_cmake_missing_binary_errors(home__test_engine_installs, monkeypatch, cached_host):
    monkeypatch.setattr(ei, "_stream", lambda cmd, log, cwd=None: 0)
    monkeypatch.setattr(ei, "_require", lambda tool, hint: f"/usr/bin/{tool}")
    monkeypatch.setattr(ei, "_git_commit", lambda dest: None)
    with pytest.raises(ei.InstallError, match="sd-server"):
        ei.install_recipe("sd-cpp", home=home__test_engine_installs, log_fn=lambda _s: None)


def test_install_update_reuses_checkout(home__test_engine_installs, fake_shell__test_engine_installs, cached_host):
    ei.install_recipe("strata", home=home__test_engine_installs, log_fn=lambda _s: None)
    fake_shell__test_engine_installs.cmds.clear()
    st2 = ei.install_recipe("strata", home=home__test_engine_installs, update=True,
                            log_fn=lambda _s: None)
    joined = "\n".join(" ".join(c) for c in fake_shell__test_engine_installs.cmds)
    assert "git clone" not in joined          # reused, not re-cloned
    assert "fetch" in joined and "reset" in joined
    assert st2["method"] == "git_pip"


# ---------------------------------------------------------- dashboard API

def test_catalog_routes_registered(app__test_engine_installs, cached_host):
    paths = {(m, p) for (m, p, _) in app__test_engine_installs.ROUTES}
    for m, p in [("GET", "/api/engine-catalog"),
                 ("POST", "/api/engine-catalog/<recipe_id>/install"),
                 ("GET", "/api/engine-catalog/<recipe_id>/install")]:
        assert (m, p) in paths, f"{m} {p} missing"


def test_catalog_endpoint_lists_all_not_installed(app__test_engine_installs, home__test_engine_installs, cached_host):
    r = app__test_engine_installs.handle("GET", "/api/engine-catalog")
    assert r.status == 200
    ids = [c["id"] for c in r.body["catalog"]]
    assert ids == ["vllm", "airllm", "strata", "sd-cpp"]
    assert all(c["installed"] is False for c in r.body["catalog"])
    assert all(c["installing"] is False for c in r.body["catalog"])


def test_install_status_unknown_recipe_404(app__test_engine_installs, home__test_engine_installs, cached_host):
    r = app__test_engine_installs.handle("GET", "/api/engine-catalog/ghost/install")
    assert r.status == 404 and err_code(r) == "unknown_recipe"
    r2 = app__test_engine_installs.handle("POST", "/api/engine-catalog/ghost/install", body={})
    assert r2.status == 404 and err_code(r2) == "unknown_recipe"


def _wait_done(app, rid, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = app.handle("GET", f"/api/engine-catalog/{rid}/install").body
        if s.get("state") in ("done", "error", "idle"):
            return s
        time.sleep(0.02)
    raise AssertionError(f"install {rid} never finished")


def test_install_endpoint_runs_and_registers_engine(app__test_engine_installs, home__test_engine_installs, monkeypatch, cached_host):
    calls = []

    def fake_install(recipe_id, *, home, ref=None, update=False,
                     backend=None, force=False, log_fn=None):
        calls.append(recipe_id)
        b = home / "engines" / recipe_id / "venv" / "bin" / "vllm"
        b.parent.mkdir(parents=True, exist_ok=True)
        b.write_text("#!/bin/sh\n")
        os.chmod(b, 0o755)  # probe() requires the exec bit
        return {"recipe": recipe_id, "method": "pip", "binary": str(b),
                "installed_at": "2026-01-01T00:00:00Z"}

    monkeypatch.setattr(ei, "install_recipe", fake_install)
    r = app__test_engine_installs.handle("POST", "/api/engine-catalog/vllm/install", body={})
    assert r.status == 200 and r.body["state"] == "running"
    done = _wait_done(app__test_engine_installs, "vllm")
    assert done["state"] == "done", done.get("error")
    assert calls == ["vllm"]

    # engine got registered with the recipe id and install binary
    lst = app__test_engine_installs.handle("GET", "/api/engines").body["engines"]
    row = next(e for e in lst if e["id"] == "vllm")
    assert row["kind"] == "openai"
    assert row["installed"] is True
    assert row["binary"].endswith("vllm")
    # vLLM spawn argv: `vllm serve {model} --host {host} --port {port}`
    cmd = eng_mod.build_engine_cmd(row, row["binary"], model="m.gguf",
                                   host="127.0.0.1", port=8001)
    assert cmd[:2] == [row["binary"], "serve"] and "8001" in cmd


def test_install_endpoint_reports_error(app__test_engine_installs, home__test_engine_installs, monkeypatch, cached_host):
    def boom(recipe_id, *, home, ref=None, update=False, backend=None,
             force=False, log_fn=None):
        raise ei.InstallError("cmake missing")

    monkeypatch.setattr(ei, "install_recipe", boom)
    r = app__test_engine_installs.handle("POST", "/api/engine-catalog/sd-cpp/install", body={})
    assert r.status == 200
    done = _wait_done(app__test_engine_installs, "sd-cpp")
    assert done["state"] == "error" and "cmake missing" in done["error"]
    ids = [e["id"] for e in app__test_engine_installs.handle("GET", "/api/engines").body["engines"]]
    assert "sd-cpp" not in ids  # failed install registers nothing


def test_install_endpoint_conflict_while_running(app__test_engine_installs, home__test_engine_installs, cached_host):
    # Seed the per-recipe run-slot with a live (fake) install record.
    task = app__test_engine_installs._install_task("vllm")
    task.record = {"state": "running", "log": []}
    r = app__test_engine_installs.handle("POST", "/api/engine-catalog/vllm/install", body={})
    assert r.status == 409 and err_code(r) == "already_running"


def test_install_register_false_skips_registry(app__test_engine_installs, home__test_engine_installs, monkeypatch, cached_host):
    def fake_install(recipe_id, *, home, ref=None, update=False, backend=None,
                     force=False, log_fn=None):
        return {"recipe": recipe_id, "method": "pip",
                "binary": "/nowhere/airllm-python"}

    monkeypatch.setattr(ei, "install_recipe", fake_install)
    r = app__test_engine_installs.handle("POST", "/api/engine-catalog/airllm/install",
                   body={"register": False})
    assert r.status == 200
    done = _wait_done(app__test_engine_installs, "airllm")
    assert done["state"] == "done"
    ids = [e["id"] for e in app__test_engine_installs.handle("GET", "/api/engines").body["engines"]]
    assert "airllm" not in ids


# ------------------------------------------------- hardware / OS awareness

def test_vllm_blocked_on_windows_and_macos(cached_host):
    """docs.vllm.ai: Linux only (Windows via WSL). Hard block, not warning."""
    for system in ("windows", "darwin"):
        c = ei.recipe_compat("vllm", fake_host(system=system))
        assert c["ok"] is False and system in c["blocked"]
    ok = ei.recipe_compat("vllm", fake_host(system="linux"))
    assert ok["ok"] is True


def test_vllm_variant_follows_gpu_vendor(cached_host):
    nvidia = [{"vendor": "nvidia", "name": "RTX 4090"}]
    amd = [{"vendor": "amd", "name": "gfx1100 (Radeon RX 7900)"}]
    c_n = ei.recipe_compat("vllm", fake_host(backend="cuda", gpus=nvidia))
    assert c_n["variant"] == "cuda" and not c_n["warnings"]
    c_a = ei.recipe_compat("vllm", fake_host(backend="rocm", gpus=amd))
    assert c_a["variant"] == "rocm"
    c_none = ei.recipe_compat("vllm", fake_host())  # no GPUs
    assert c_none["ok"] is True and c_none["warnings"]  # advisory only


def test_vllm_rocm_install_uses_rocm_wheel_index(home__test_engine_installs, fake_shell__test_engine_installs, monkeypatch, cached_host):
    """AMD host => pip must get --extra-index-url wheels.vllm.ai/rocm/."""
    monkeypatch.setattr(ei, "detect_host", lambda: fake_host(
        backend="rocm", gpus=[{"vendor": "amd", "name": "gfx942"}]))
    ei.install_recipe("vllm", home=home__test_engine_installs, log_fn=lambda _s: None)
    pip_cmds = [c for c in fake_shell__test_engine_installs.cmds if "pip" in c and "install" in c]
    assert any("wheels.vllm.ai/rocm/" in " ".join(c) for c in pip_cmds)
    st = ei.read_state(home__test_engine_installs, "vllm")
    assert st is not None and st["variant"] == "rocm"


def test_vllm_cuda_install_uses_plain_pypi(home__test_engine_installs, fake_shell__test_engine_installs, monkeypatch, cached_host):
    monkeypatch.setattr(ei, "detect_host", lambda: fake_host(
        backend="cuda", gpus=[{"vendor": "nvidia", "name": "RTX 4090"}]))
    ei.install_recipe("vllm", home=home__test_engine_installs, log_fn=lambda _s: None)
    pip_cmds = [c for c in fake_shell__test_engine_installs.cmds if "pip" in c and "install" in c]
    assert pip_cmds and not any("--extra-index-url" in c for c in pip_cmds)


def test_install_blocked_on_incompatible_os(home__test_engine_installs, fake_shell__test_engine_installs, monkeypatch, cached_host):
    monkeypatch.setattr(ei, "detect_host", lambda: fake_host(system="windows"))
    with pytest.raises(ei.InstallError, match="does not support windows"):
        ei.install_recipe("vllm", home=home__test_engine_installs, log_fn=lambda _s: None)
    assert fake_shell__test_engine_installs.cmds == []          # nothing ran — no network touched
    # force overrides the gate (user knows best; upstream will fail later)
    st = ei.install_recipe("vllm", home=home__test_engine_installs, force=True,
                           log_fn=lambda _s: None)
    assert st["method"] == "pip"


def test_sd_cpp_backend_flags_per_host(home__test_engine_installs, fake_shell__test_engine_installs, cached_host):
    cases = {
        "cuda": (["-DSD_CUDA=ON"], None),
        "metal": (["-DSD_METAL=ON"], None),
        "vulkan": (["-DSD_VULKAN=ON"], None),
        "cpu": ([], None),
        "rocm": (["-DSD_HIPBLAS=ON", "-DGPU_TARGETS=gfx1100",
                  "-DAMDGPU_TARGETS=gfx1100"],
                 [{"vendor": "amd", "name": "gfx1100 (Radeon RX 7900)"}]),
    }
    for be, (expected, gpus) in cases.items():
        flags = ei._sd_cmake_flags(be, fake_host(backend=be, gpus=gpus or ()))
        assert flags == expected, f"backend {be}"


def test_sd_cpp_install_passes_detected_backend(home__test_engine_installs, monkeypatch, cached_host):
    cfg_cmds = []

    def rec(cmd, log, cwd=None):
        if "-S" in cmd:
            cfg_cmds.append([str(c) for c in cmd])
        return FakeShell.stream(fs, cmd, log, cwd=cwd)
    fs = FakeShell(home__test_engine_installs)
    monkeypatch.setattr(ei, "_stream", rec)
    monkeypatch.setattr(ei, "_stream_out", fs.stream_out)
    monkeypatch.setattr(ei, "_require", fs.require)
    monkeypatch.setattr(ei, "detect_host", lambda: fake_host(
        backend="vulkan", gpus=[{"vendor": "intel", "name": "Arc A770"}]))
    st = ei.install_recipe("sd-cpp", home=home__test_engine_installs, log_fn=lambda _s: None)
    assert any("-DSD_VULKAN=ON" in c for c in cfg_cmds)
    assert st["backend"] == "vulkan"


def test_sd_cpp_backend_override(home__test_engine_installs, monkeypatch, cached_host):
    cfg_cmds = []

    def rec(cmd, log, cwd=None):
        if "-S" in cmd:
            cfg_cmds.append([str(c) for c in cmd])
        return FakeShell.stream(fs, cmd, log, cwd=cwd)
    fs = FakeShell(home__test_engine_installs)
    monkeypatch.setattr(ei, "_stream", rec)
    monkeypatch.setattr(ei, "_stream_out", fs.stream_out)
    monkeypatch.setattr(ei, "_require", fs.require)
    ei.install_recipe("sd-cpp", home=home__test_engine_installs, backend="cuda",
                      log_fn=lambda _s: None)
    assert any("-DSD_CUDA=ON" in c for c in cfg_cmds)


def test_airllm_warns_without_nvidia(cached_host):
    c = ei.recipe_compat("airllm", fake_host())
    assert c["ok"] is True and c["warnings"]
    c2 = ei.recipe_compat("airllm", fake_host(
        backend="cuda", gpus=[{"vendor": "nvidia", "name": "T4"}]))
    assert not c2["warnings"]


def test_amd_gfx_target_parsing(cached_host):
    h = fake_host(gpus=[{"vendor": "amd", "name": "gfx942 (MI300X)"}])
    assert ei.amd_gfx_target(h) == "gfx942"
    assert ei.amd_gfx_target(fake_host()) is None


def test_catalog_endpoint_includes_compat_and_host(app__test_engine_installs, monkeypatch, cached_host):
    monkeypatch.setattr(ei, "detect_host", lambda: fake_host(system="windows"))
    r = app__test_engine_installs.handle("GET", "/api/engine-catalog")
    assert r.status == 200
    assert r.body["host"]["system"] == "windows"
    by_id = {c["id"]: c for c in r.body["catalog"]}
    assert by_id["vllm"]["compat"]["ok"] is False
    assert by_id["sd-cpp"]["compat"]["ok"] is True


def test_install_endpoint_rejects_incompatible_host(app__test_engine_installs, home__test_engine_installs, monkeypatch, cached_host):
    monkeypatch.setattr(ei, "detect_host", lambda: fake_host(system="windows"))
    r = app__test_engine_installs.handle("POST", "/api/engine-catalog/vllm/install", body={})
    assert r.status == 400 and err_code(r) == "incompatible_host"
    # force goes through (install itself is faked here)
    monkeypatch.setattr(ei, "install_recipe", lambda *a, **k: {
        "recipe": "vllm", "method": "pip", "binary": "/x"})
    r2 = app__test_engine_installs.handle("POST", "/api/engine-catalog/vllm/install",
                    body={"force": True})
    assert r2.status == 200


def test_reregister_preserves_user_edits(app__test_engine_installs, home__test_engine_installs, cached_host):
    """Re-running an install refreshes binary/args but keeps user notes."""
    eng = eng_mod.validate_engine_body({"name": "vLLM", "binary": "/old/vllm",
                                        "kind": "openai"})
    eng["id"] = "vllm"
    eng["notes"] = "my custom notes"
    eng_mod.save_engine(home__test_engine_installs, eng)

    app__test_engine_installs._register_installed_engine("vllm", {
        "binary": "/new/venv/bin/vllm", "installed_at": "2026-02-02T00:00:00Z"})
    again = eng_mod.get_engine(home__test_engine_installs, "vllm")
    assert again is not None
    assert again["binary"] == "/new/venv/bin/vllm"
    assert again["notes"] == "my custom notes"


# ------------------------------------------------- spawnable recipe bodies

def test_strata_engine_body_spawns_as_openai_server(home__test_engine_installs, cached_host):
    """Strata registers as an OpenAI-compatible engine whose {model} is its
    run-config.json, run from the checkout (cwd) via `python -m serve.server`."""
    st = {"binary": str(home__test_engine_installs / "engines/strata/venv/bin/python"),
          "prefix": str(home__test_engine_installs / "engines/strata/src"),
          "installed_at": "2026-01-01T00:00:00Z"}
    body = ei.engine_body_for("strata", st)
    eng = eng_mod.validate_engine_body(body)
    assert eng["kind"] == "openai"
    assert eng["model_kind"] == "json"
    assert eng["cwd"] == str(home__test_engine_installs / "engines/strata/src")
    cmd = eng_mod.build_engine_cmd(eng, eng["binary"], model="/c/run.json",
                                   host="127.0.0.1", port=8095)
    assert cmd[1:3] == ["-m", "serve.server"]
    assert "--config" in cmd and "/c/run.json" in cmd
    assert "--port" in cmd and "8095" in cmd


def test_sd_cpp_engine_body_is_diffusion_server(home__test_engine_installs, cached_host):
    st = {"binary": str(home__test_engine_installs / "engines/sd-cpp/src/build/bin/sd-server"),
          "installed_at": "2026-01-01T00:00:00Z"}
    body = ei.engine_body_for("sd-cpp", st)
    eng = eng_mod.validate_engine_body(body)
    assert eng["kind"] == "diffusion"
    assert eng.get("model_kind") == "gguf"
    cmd = eng_mod.build_engine_cmd(eng, "/x/sd-server", model="/m/sdxl.gguf",
                                   host="127.0.0.1", port=1234)
    assert "--model" in cmd and "/m/sdxl.gguf" in cmd
    assert "--listen-ip" in cmd and "127.0.0.1" in cmd
    assert "--listen-port" in cmd and "1234" in cmd


# ------------------------------------------------------- recipe sync

def _fake_install(home, rid, binary_name):
    b = ei.engine_dir(home, rid) / "bin" / (
        binary_name + (".exe" if os.name == "nt" else ""))
    b.parent.mkdir(parents=True, exist_ok=True)
    b.write_text("#!/bin/sh\n")
    os.chmod(b, 0o755)
    ei.write_state(home, rid, {"recipe": rid, "binary": str(b),
                               "installed_at": "2026-01-01T00:00:00Z"})
    return b


def test_sync_upgrades_legacy_sd_cpp_other_to_diffusion(home__test_engine_installs, cached_host):
    """Installs recorded while sd-cpp was inventory-only flip to diffusion
    once the built tree has sd-server next to the recorded binary."""
    cli = _fake_install(home__test_engine_installs, "sd-cpp", "sd-cli")
    srv = cli.parent / ("sd-server" + (".exe" if os.name == "nt" else ""))
    srv.write_text("#!/bin/sh\n")
    os.chmod(srv, 0o755)
    eng = eng_mod.validate_engine_body({"name": "stable-diffusion.cpp",
                                        "binary": str(cli), "kind": "other"})
    eng["id"] = "sd-cpp"
    eng_mod.save_engine(home__test_engine_installs, eng)

    refreshed = ei.sync_installed_recipes(home__test_engine_installs)
    assert refreshed == ["sd-cpp"]
    again = eng_mod.get_engine(home__test_engine_installs, "sd-cpp")
    assert again is not None
    assert again["kind"] == "diffusion"
    assert again["binary"] == str(srv)          # promoted to the server binary
    assert "--listen-port" in " ".join(again["args_template"])
    # second pass is a no-op
    assert ei.sync_installed_recipes(home__test_engine_installs) == []


def test_sync_keeps_sd_cpp_inventory_without_server_binary(home__test_engine_installs, cached_host):
    """No sd-server on disk => do not promote to spawnable (a CLI could never
    answer the healthcheck); record stays inventory-only."""
    cli = _fake_install(home__test_engine_installs, "sd-cpp", "sd-cli")
    eng = eng_mod.validate_engine_body({"name": "stable-diffusion.cpp",
                                        "binary": str(cli), "kind": "other"})
    eng["id"] = "sd-cpp"
    eng_mod.save_engine(home__test_engine_installs, eng)
    assert ei.sync_installed_recipes(home__test_engine_installs) == []
    again = eng_mod.get_engine(home__test_engine_installs, "sd-cpp")
    assert again is not None and again["kind"] == "other"


def test_sync_upgrades_strata_to_spawnable_openai(home__test_engine_installs, cached_host):
    st_bin = _fake_install(home__test_engine_installs, "strata", "python")
    eng = eng_mod.validate_engine_body({"name": "Strata",
                                        "binary": str(st_bin),
                                        "kind": "other"})
    eng["id"] = "strata"
    eng_mod.save_engine(home__test_engine_installs, eng)
    refreshed = ei.sync_installed_recipes(home__test_engine_installs)
    assert refreshed == ["strata"]
    again = eng_mod.get_engine(home__test_engine_installs, "strata")
    assert again is not None
    assert again["kind"] == "openai"
    assert again["model_kind"] == "json"
    assert again["args_template"][:2] == ["-m", "serve.server"]


def test_sync_repairs_stale_strata_venv_cwd(home__test_engine_installs, cached_host):
    """Installs recorded while strata was inventory-only carry the VENV dir as
    prefix; spawning `python -m serve.server` from there dies with
    ModuleNotFoundError. Sync must re-derive cwd from the checkout layout."""
    st_bin = _fake_install(home__test_engine_installs, "strata", "python")
    # The git checkout run_from_checkout needs on disk (serve/server.py lives
    # inside it; toaster_loop spawns with cwd=checkout).
    src = ei.src_dir(home__test_engine_installs, "strata")
    (src / ".git").mkdir(parents=True)
    (src / "serve").mkdir()
    (src / "serve" / "server.py").write_text("")
    # Legacy state: prefix is the VENV (written before strata was spawnable).
    ei.write_state(home__test_engine_installs, "strata", {"recipe": "strata",
                                    "binary": str(st_bin),
                                    "prefix": str(ei.venv_dir(home__test_engine_installs, "strata")),
                                    "installed_at": "2026-01-01T00:00:00Z"})
    eng = eng_mod.validate_engine_body(
        {"name": "Strata", "binary": str(st_bin), "kind": "openai",
         "args_template": ["-m", "serve.server"], "cwd":
             str(ei.venv_dir(home__test_engine_installs, "strata")), "model_kind": "json"})
    eng["id"] = "strata"
    eng_mod.save_engine(home__test_engine_installs, eng)

    refreshed = ei.sync_installed_recipes(home__test_engine_installs)
    assert refreshed == ["strata"]
    again = eng_mod.get_engine(home__test_engine_installs, "strata")
    assert again is not None
    assert again["cwd"] == str(src)  # repaired to the checkout
    # second pass is a no-op
    assert ei.sync_installed_recipes(home__test_engine_installs) == []


def test_sync_skips_engines_not_matching_install_state(home__test_engine_installs, cached_host):
    """A hand-registered engine that merely shares a recipe id but whose
    binary differs from the install state is left alone."""
    _fake_install(home__test_engine_installs, "sd-cpp", "sd-server")
    other = home__test_engine_installs / "somewhere-else" / ("sd-server" + (".exe" if os.name == "nt" else ""))
    other.parent.mkdir(parents=True, exist_ok=True)
    other.write_text("#!/bin/sh\n")
    os.chmod(other, 0o755)
    eng = eng_mod.validate_engine_body({"name": "My SD", "binary": str(other),
                                        "kind": "diffusion"})
    eng["id"] = "sd-cpp"
    eng_mod.save_engine(home__test_engine_installs, eng)
    assert ei.sync_installed_recipes(home__test_engine_installs) == []
    again = eng_mod.get_engine(home__test_engine_installs, "sd-cpp")
    assert again is not None and again["binary"] == str(other)


def test_engines_list_endpoint_syncs_installed_recipes(app__test_engine_installs, home__test_engine_installs, cached_host):
    """GET /api/engines refreshes recipe-managed records (so the Serve page
    sees the new kind without a reinstall)."""
    cli = _fake_install(home__test_engine_installs, "sd-cpp", "sd-cli")
    srv = cli.parent / ("sd-server" + (".exe" if os.name == "nt" else ""))
    srv.write_text("#!/bin/sh\n")
    os.chmod(srv, 0o755)
    eng = eng_mod.validate_engine_body({"name": "stable-diffusion.cpp",
                                        "binary": str(cli), "kind": "other"})
    eng["id"] = "sd-cpp"
    eng_mod.save_engine(home__test_engine_installs, eng)
    r = app__test_engine_installs.handle("GET", "/api/engines")
    assert r.status == 200
    row = next(e for e in r.body["engines"] if e["id"] == "sd-cpp")
    assert row["kind"] == "diffusion"

# ------------------------------------------------------------------------
# from tests/test_calibrate.py
# ------------------------------------------------------------------------

# ------------------------------------------------------------------ fixtures

@pytest.fixture()
def cal_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("TOASTER_LOOP_CALIBRATE_DIR", str(tmp_path / "cal"))
    monkeypatch.setenv("TOASTER_LOOP_BENCH_DIR", str(tmp_path / "bench"))
    monkeypatch.setenv("TOASTER_LOOP_MACHINE_ID", "test-machine")
    return tmp_path


@pytest.fixture()
def gguf(tmp_path):
    g = tmp_path / "model.gguf"
    g.write_bytes(b"GGUF" + b"\x00" * 500)
    return g


def fake_rec(gen_median, streams="tok-base", spread=1.0):
    """Minimal WP1-shaped bench record for the measure seam."""
    return {
        "kind": bench.KIND_BENCH,
        "config": {"prompts_sha256": "psha"},
        "token_identity": {"status": "ok", "streams_sha256": streams},
        "generation": {"median": gen_median, "spread_pct": spread},
        "prompt_processing": {"median": gen_median * 4, "spread_pct": spread},
    }


def perf_measure(perf_by_knob, base=100.0, drift_keys=(), error_keys=()):
    """measure_fn backed by a static perf table.

    ``perf_by_knob``: {knob: {value: gen_median}}; anything unlisted measures
    at ``base``. Keys in drift_keys answer with a DIFFERENT token stream;
    keys in error_keys raise (server failed to start / crashed)."""

    def measure(settings, runs, label):  # noqa: ARG001
        med = base
        streams = "tok-base"
        for knob, table in perf_by_knob.items():
            value = settings.get(knob)
            if value is not None and value in table:
                key = (knob, str(value))
                if key in error_keys:
                    raise RuntimeError("server died mid-bench")
                if key in drift_keys:
                    streams = "tok-drift"
                med *= table[value] / 100.0
        return fake_rec(med, streams=streams)

    return measure


def persist_noise_floor(spread_pct):
    bench.save_result({
        "kind": bench.KIND_NOISE_FLOOR,
        "timestamp": "2026-10-11T00:00:00Z",
        "label": "floor",
        "generation": {"median": 100.0, "spread_pct": spread_pct},
    })


def row_for(profile, knob, value):
    for row in profile["knobs"][knob]["candidates"]:
        if row["value"] == value:
            return row
    raise AssertionError(f"no row for {knob}={value}")


# ------------------------------------------------------------- sweep adoption

def test_sweep_adopts_only_gains_above_threshold(cal_dirs, gguf):
    persist_noise_floor(1.0)  # noise floor 1% -> threshold max(3%, 1%) = 3%
    measure = perf_measure({
        "kv_cache_type": {"bf16": 102.0,   # +2%  -> rejected (below 3%)
                          "q8_0": 110.0,   # +10% -> adopted
                          "q4_0": 95.0},   # slower -> rejected
    })
    prof = cal.run_sweep(object(), gguf_path=str(gguf), runs=2,
                         knobs=["kv_cache_type"], measure_fn=measure)

    assert row_for(prof, "kv_cache_type", "bf16")["verdict"] == "rejected"
    assert row_for(prof, "kv_cache_type", "q8_0")["verdict"] == "adopted"
    assert row_for(prof, "kv_cache_type", "q4_0")["verdict"] == "rejected"
    # default f16 equals the current value -> recorded as baseline, not measured
    assert row_for(prof, "kv_cache_type", "f16")["verdict"] == "baseline"
    assert prof["knobs"]["kv_cache_type"]["adopted"] == "q8_0"
    assert prof["adopted_settings"] == {"kv_cache_type": "q8_0"}
    # delta recorded against the running best (110 vs 100 = +10%)
    assert row_for(prof, "kv_cache_type", "q8_0")["delta_pct"] == \
        pytest.approx(10.0, abs=0.01)


def test_sweep_noise_floor_overrides_three_percent(cal_dirs, gguf):
    persist_noise_floor(12.0)  # threshold max(3%, 12%) = 12%
    measure = perf_measure({"kv_cache_type": {"q8_0": 110.0}})  # +10% < 12%
    prof = cal.run_sweep(object(), gguf_path=str(gguf), runs=2,
                         knobs=["kv_cache_type"], measure_fn=measure)
    row = row_for(prof, "kv_cache_type", "q8_0")
    assert row["verdict"] == "rejected"
    assert "threshold" in row.get("reason", "")
    assert prof["adopted_settings"] == {}


def test_sweep_token_drift_is_invalid_never_adopted(cal_dirs, gguf):
    persist_noise_floor(1.0)
    measure = perf_measure({"kv_cache_type": {"q8_0": 150.0}},
                           drift_keys={("kv_cache_type", "q8_0")})
    prof = cal.run_sweep(object(), gguf_path=str(gguf), runs=2,
                         knobs=["kv_cache_type"], measure_fn=measure)
    row = row_for(prof, "kv_cache_type", "q8_0")
    assert row["verdict"] == "invalid"  # +50% but tokens changed -> never a win
    assert prof["knobs"]["kv_cache_type"]["adopted"] is None
    assert prof["adopted_settings"] == {}


def test_sweep_estimates_and_persists_noise_floor_when_absent(cal_dirs, gguf):
    assert bench.load_noise_floor() is None
    measure = perf_measure({"kv_cache_type": {"q8_0": 110.0}})
    prof = cal.run_sweep(object(), gguf_path=str(gguf), runs=2,
                         knobs=["kv_cache_type"], measure_fn=measure)
    nf = bench.load_noise_floor()
    assert nf is not None, "a noise-floor record must be persisted on first use"
    assert prof["noise_floor"]["spread_pct"] == \
        nf["generation"]["spread_pct"]


def test_sweep_error_candidate_skipped_and_sweep_continues(cal_dirs, gguf):
    persist_noise_floor(1.0)
    measure = perf_measure(
        {"flash_attn": {True: 120.0}, "kv_cache_type": {"q8_0": 115.0}},
        error_keys={("flash_attn", "True")})
    prof = cal.run_sweep(object(), gguf_path=str(gguf), runs=2,
                         knobs=["flash_attn", "kv_cache_type"],
                         measure_fn=measure)
    assert row_for(prof, "flash_attn", True)["verdict"] == "error"
    # sweep continued past the failing candidate into the next knob
    assert row_for(prof, "kv_cache_type", "q8_0")["verdict"] == "adopted"
    assert prof["adopted_settings"] == {"kv_cache_type": "q8_0"}


def test_sweep_unmeasurable_baseline_raises(cal_dirs, gguf):
    persist_noise_floor(1.0)

    def dead(settings, runs, label):  # noqa: ARG001
        raise RuntimeError("no server reachable")

    with pytest.raises(cal.CalibrateError, match="nothing could be measured"):
        cal.run_sweep(object(), gguf_path=str(gguf), runs=2,
                      knobs=["flash_attn"], measure_fn=dead)


def test_sweep_spec_mtp_without_drafter_records_tested_as(cal_dirs, gguf):
    persist_noise_floor(1.0)
    measure = perf_measure({"speculative_type": {"ngram": 110.0,
                                                 "mtp": 130.0}})
    prof = cal.run_sweep(object(), gguf_path=str(gguf), runs=2,
                         knobs=["speculative_type"], measure_fn=measure)
    mtp_row = row_for(prof, "speculative_type", "mtp")
    # No mtp*.gguf next to the model -> run_args fell back to ngram-mod.
    assert mtp_row["tested_as"] == "ngram (mtp unavailable)"
    ngram_row = row_for(prof, "speculative_type", "ngram")
    assert ngram_row.get("tested_as") == "ngram"


def test_sweep_coordinate_wise_best_carries_across_knobs(cal_dirs, gguf):
    persist_noise_floor(1.0)
    # kv q8_0 wins (+10% -> best 110); flash True adds a multiplicative +4%,
    # i.e. +4% over the NEW best (110*1.04 = 114.4) -> adopted too.
    measure = perf_measure({
        "kv_cache_type": {"bf16": 102, "q8_0": 110, "q4_0": 95},
        "flash_attn": {True: 104.0},
    })
    prof = cal.run_sweep(object(), gguf_path=str(gguf), runs=2,
                         knobs=["kv_cache_type", "flash_attn"],
                         measure_fn=measure)
    assert prof["adopted_settings"] == {"kv_cache_type": "q8_0",
                                        "flash_attn": True}


def test_sweep_quick_preset_is_smaller_and_overridable():
    full = cal.resolve_grid()
    quick = cal.resolve_grid(quick=True)
    assert set(quick) < set(full)
    subset = cal.resolve_grid(["kv_cache_type", "flash_attn"])
    assert set(subset) == {"kv_cache_type", "flash_attn"}
    with pytest.raises(ValueError, match="unknown knob"):
        cal.resolve_grid(["bogus_knob"])


# ----------------------------------------------------------- profile storage

def test_profile_save_load_roundtrip_keyed_by_machine_and_model(cal_dirs, gguf):
    prof = {
        "profile_version": cal.PROFILE_VERSION,
        "machine_id": cal.machine_id(),
        "model_hash": cal.model_hash(gguf),
        "timestamp": "2026-10-11T00:00:00Z",
        "adopted_settings": {"kv_cache_type": "q8_0"},
        "knobs": {},
    }
    p = cal.save_profile(prof)
    assert p.is_file() and str(cal_dirs / "cal") in str(p)
    mhash = cal.model_hash(gguf)
    assert mhash is not None
    loaded = cal.load_profile(cal.machine_id(), mhash)
    assert loaded is not None
    assert loaded["adopted_settings"] == {"kv_cache_type": "q8_0"}
    # different machine or model -> no match
    assert cal.load_profile("other-machine", mhash) is None
    other = Path(cal_dirs / "other.gguf")
    other.write_bytes(b"different bytes entirely")
    mhash_other = cal.model_hash(other)
    assert mhash_other is not None
    assert cal.load_profile(cal.machine_id(), mhash_other) is None
    listing = cal.list_profiles()
    assert len(listing) == 1 and listing[0]["machine_id"] == "test-machine"
    assert cal.clear_profiles() == 1
    assert cal.list_profiles() == []


def test_model_hash_stable_and_content_sensitive(gguf, tmp_path):
    h1 = cal.model_hash(gguf)
    assert h1 and h1 == cal.model_hash(gguf)
    g2 = tmp_path / "b.gguf"
    g2.write_bytes(b"GGUX" + b"\x00" * 500)  # first byte differs
    assert cal.model_hash(g2) != h1
    g3 = tmp_path / "c.gguf"
    g3.write_bytes(b"GGUF" + b"\x00" * 499)  # size differs
    assert cal.model_hash(g3) != h1
    assert cal.model_hash(tmp_path / "missing.gguf") is None


def test_machine_id_env_and_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv("TOASTER_LOOP_MACHINE_ID", "boxed-42")
    assert cal.machine_id() == "boxed-42"
    monkeypatch.delenv("TOASTER_LOOP_MACHINE_ID")
    mid = cal.machine_id()  # /etc/machine-id or hostname fallback
    assert mid and isinstance(mid, str)


# ----------------------------------------------- serving integration (opt-in)

class _Cfg:
    def __init__(self, serve=None, run_settings=None):
        self.serve = serve or {}
        self.run_settings = run_settings or {}
        self.context_length = 4096
        self.gpu_layers = None
        self.extra_llama_args = []


def _store_profile(gguf, adopted):
    cal.save_profile({
        "profile_version": cal.PROFILE_VERSION,
        "machine_id": cal.machine_id(),
        "model_hash": cal.model_hash(gguf),
        "timestamp": "2026-10-11T00:00:00Z",
        "knobs": {k: {"adopted": v} for k, v in adopted.items()},
    })


def test_apply_calibration_precedence_matrix(cal_dirs, gguf):
    _store_profile(gguf, {"kv_cache_type": "q8_0", "flash_attn": True})

    # 1. calibration OFF (default) -> {} — today's behavior exactly.
    assert cal.apply_calibration(_Cfg(), str(gguf)) == {}
    # unknown mode value also stays off (never guesses).
    assert cal.apply_calibration(_Cfg(serve={"calibration": "yolo"}),
                                 str(gguf)) == {}
    # 2. opt-in + matching profile -> adopted values.
    got = cal.apply_calibration(_Cfg(serve={"calibration": "auto"}), str(gguf))
    assert got == {"kv_cache_type": "q8_0", "flash_attn": True}
    # 3. opt-in but no profile for this model -> {}.
    other = gguf.parent / "other.gguf"
    other.write_bytes(b"unrelated model bytes")
    assert cal.apply_calibration(_Cfg(serve={"calibration": "auto"}),
                                 str(other)) == {}
    # 4. opt-in but unreadable model -> {}.
    assert cal.apply_calibration(_Cfg(serve={"calibration": "auto"}),
                                 str(gguf.parent / "nope.gguf")) == {}


def test_build_instance_cmd_default_path_unchanged(cal_dirs, gguf):
    """No serve.calibration key: argv is exactly what pre-WP3 produced."""
    from toaster_loop.serving import supervisor
    from toaster_loop.serving.runtime import RuntimeInfo

    _store_profile(gguf, {"kv_cache_type": "q8_0"})
    rt = RuntimeInfo(system="linux", machine="x86_64", backend="cpu")
    inst = {"id": "default", "default": True, "gguf": str(gguf),
            "run_settings": {}}
    cmd = supervisor.build_instance_cmd("llama-server", _Cfg(), inst,
                                        port=1234, model_name="m", rt=rt)
    assert "-ctk" not in cmd, "calibrated values must NOT leak without opt-in"


def test_build_instance_cmd_opt_in_adopts_and_manual_wins(cal_dirs, gguf):
    from toaster_loop.serving import supervisor
    from toaster_loop.serving.runtime import RuntimeInfo

    _store_profile(gguf, {"kv_cache_type": "q8_0", "parallel": 4})
    rt = RuntimeInfo(system="linux", machine="x86_64", backend="cpu")

    # opt-in: calibrated values become defaults for the default instance.
    cfg = _Cfg(serve={"calibration": "auto"})
    inst = {"id": "default", "default": True, "gguf": str(gguf),
            "run_settings": {}}
    cmd = supervisor.build_instance_cmd("llama-server", cfg, inst,
                                        port=1234, model_name="m", rt=rt)
    assert "-ctk" in cmd and cmd[cmd.index("-ctk") + 1] == "q8_0"
    assert "-np" in cmd and cmd[cmd.index("-np") + 1] == "4"

    # manual explicit config ALWAYS overrides the calibrated value.
    cfg2 = _Cfg(serve={"calibration": "auto"},
                run_settings={"kv_cache_type": "q4_0"})
    cmd2 = supervisor.build_instance_cmd("llama-server", cfg2, inst,
                                         port=1234, model_name="m", rt=rt)
    assert cmd2[cmd2.index("-ctk") + 1] == "q4_0"


# ------------------------------------------------------------------ CLI wiring

def cli_main(argv):
    from toaster_loop import cli

    return cli.main(argv)


def test_cli_calibrate_help_and_flags():
    from toaster_loop import cli

    ns = cli.build_parser().parse_args(
        ["calibrate", "--quick", "--knobs", "flash_attn", "--runs", "2"])
    assert ns.command == "calibrate"
    assert ns.quick and ns.knobs == "flash_attn" and ns.runs == 2


def test_cli_list_show_clear_roundtrip(cal_dirs, gguf, capsys):
    assert cli_main(["calibrate", "--list"]) == 0
    assert "no calibration profiles" in capsys.readouterr().out

    _store_profile(gguf, {"kv_cache_type": "q8_0"})
    assert cli_main(["calibrate", "--list"]) == 0
    out = capsys.readouterr().out
    assert "test-machine" in out and "active" in out

    assert cli_main(["calibrate", "--show", "--model", str(gguf)]) == 0
    out = capsys.readouterr().out
    assert "machine_id:  test-machine" in out
    assert '"kv_cache_type": "q8_0"' in out

    assert cli_main(["calibrate", "--clear"]) == 0
    assert "removed 1" in capsys.readouterr().out
    assert cal.list_profiles() == []


def test_cli_show_without_profile_exits_1(cal_dirs, gguf, capsys):
    assert cli_main(["calibrate", "--show", "--model", str(gguf)]) == 1
    assert "no calibration profile" in capsys.readouterr().out


def test_cli_sweep_without_model_exits_2(cal_dirs, capsys):
    assert cli_main(["calibrate"]) == 2
    assert "no GGUF to calibrate" in capsys.readouterr().err


def test_cli_unknown_knob_exits_2(cal_dirs, gguf, capsys):
    assert cli_main(["calibrate", "--model", str(gguf),
                     "--knobs", "not_a_knob"]) == 2
    err = capsys.readouterr().err
    assert "unknown knob" in err


def test_cli_dead_endpoint_exits_1(cal_dirs, gguf, capsys):
    # Attach to a dead loopback port: nothing can be measured -> exit 1.
    rc = cli_main(["calibrate", "--model", str(gguf), "--runs", "1",
                   "--knobs", "flash_attn",
                   "--attach", "http://127.0.0.1:9/v1"])
    assert rc == 1
    assert "error" in capsys.readouterr().err


def test_cli_nonloopback_attach_exits_2(cal_dirs, gguf, capsys):
    rc = cli_main(["calibrate", "--model", str(gguf),
                   "--attach", "http://example.com/v1"])
    assert rc == 2
    assert "loopback" in capsys.readouterr().err


def test_cli_sweep_attach_end_to_end(cal_dirs, gguf, capsys, monkeypatch):
    """Full CLI sweep through the attach seam against a fake server.

    The candidate knobs are NOT applied to an attached foreign server (honest
    mode: profile records endpoint_mode=attach), so this exercises wiring +
    rendering, not real autotune — spawn mode is covered via run_sweep above."""
    from fake_server import FakeOpenAIServer
    # _BenchHandler is defined in the bench section of this module.

    _BenchHandler.requests = []
    with FakeOpenAIServer(handler_cls=_BenchHandler) as srv:
        rc = cli_main(["calibrate", "--model", str(gguf), "--runs", "1",
                       "--quick", "--attach", srv.base_url])
    out = capsys.readouterr().out
    assert rc == 0
    assert "toaster_loop calibrate" in out
    assert "adoption threshold" in out
    profs = cal.list_profiles()
    assert len(profs) == 1
    assert profs[0]["endpoint_mode"] == "attach"


def test_rendering_smoke(cal_dirs, gguf):
    persist_noise_floor(2.0)
    measure = perf_measure({"kv_cache_type": {"q8_0": 110.0}})
    prof = cal.run_sweep(object(), gguf_path=str(gguf), runs=2,
                         knobs=["kv_cache_type"], measure_fn=measure)
    table = cal.format_summary(prof)
    assert "knob" in table and "verdict" in table
    assert "adopted: kv_cache_type=q8_0" in table
    shown = cal.format_profile_show(prof)
    assert "machine_id:" in shown and "noise floor" in shown

# ------------------------------------------------------------------------
# from tests/test_coexist_wp5.py
# ------------------------------------------------------------------------

URL = "http://127.0.0.1:8123/v1"


class FakeClock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


@pytest.fixture()
def home__test_coexist_wp5(tmp_path, monkeypatch):
    h = tmp_path / "uf5"
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(h))
    wd.reset_inflight()
    wd.reset_activity()
    yield h
    wd.reset_inflight()
    wd.reset_activity()


def make_instance(home, iid="alpha", gguf="/nonexistent/alpha.gguf"):
    srv_inst.save_instance(Path(home), {"id": iid, "name": iid.title(),
                                        "gguf": gguf})


def write_live_state(home, iid="alpha", url=URL, port=8123):
    # pid = our own pid => state_alive() (os.kill probe) says running, and
    # nothing here ever kills it: stop paths are always monkeypatched.
    srv_inst.write_state(Path(home), iid, {
        "mode": "spawn", "pid": os.getpid(), "host": "127.0.0.1",
        "port": port, "base_url": url, "model_name": "alpha-model",
        "gguf_path": "/nonexistent/alpha.gguf", "instance_id": iid,
    })


def write_unloaded_state(home, iid="alpha", url=URL, port=8123):
    srv_inst.write_state(Path(home), iid, {
        "mode": "spawn", "unloaded": True, "unload_reason": "idle",
        "port": port, "base_url": url, "model_name": "alpha-model",
        "gguf_path": "/nonexistent/alpha.gguf", "instance_id": iid,
    })


class FakeHandle:
    def __init__(self, base_url=URL):
        # Only .pid is ever consulted by the load path (state record).
        class _P:
            pid = os.getpid()

        self.process = _P()
        self.base_url = base_url
        self.model_name = "alpha-model"


def patch_spawn(monkeypatch, log):
    def fake_start(cfg, inst, **kw):
        log.append(("spawn", str(inst.get("id"))))
        return FakeHandle()
    monkeypatch.setattr("toaster_loop.serving.supervisor.start_instance_server",
                        fake_start)


def patch_stop(monkeypatch, log):
    def fake_stop(iid, grace=8.0):
        log.append(("stop", iid, grace))
        return True
    monkeypatch.setattr("toaster_loop.serving.supervisor.stop_instance", fake_stop)


# ----------------------------------------------------------------- config

def test_defaults_all_off():
    cc = cx.coexist_config(ToasterLoopConfig())
    assert cc.idle_unload_seconds == 0.0
    assert cc.lazy_load is False
    assert cc.min_free_vram_mib == 0
    assert cc.coexist_policy == "refuse"
    assert not cc.idle_enabled and not cc.handshake_enabled
    assert not cc.any_enabled


def test_build_returns_none_when_disabled():
    assert cx.build(ToasterLoopConfig(), "/tmp/whatever") is None


def test_build_armed_with_lazy_load():
    cfg = ToasterLoopConfig(serve={"lazy_load": "yes"})
    coex = cx.build(cfg, "/tmp/whatever")
    assert coex is not None and coex.cfg.lazy_load is True


def test_lenient_parsing_and_bad_values():
    cfg = ToasterLoopConfig(serve={
        "idle_unload_seconds": "300",       # string float ok
        "lazy_load": "on",                  # truthy string
        "min_free_vram_mib": "8192.0",      # float-ish int
        "coexist_policy": "UNLOAD",         # case-insensitive
    })
    cc = cx.coexist_config(cfg)
    assert cc.idle_unload_seconds == 300.0
    assert cc.lazy_load is True
    assert cc.min_free_vram_mib == 8192
    assert cc.coexist_policy == "unload"

    bad = ToasterLoopConfig(serve={"idle_unload_seconds": "soon",
                               "min_free_vram_mib": "lots",
                               "coexist_policy": "yolo"})
    cc2 = cx.coexist_config(bad)  # warns, keeps defaults — never raises
    assert cc2.idle_unload_seconds == 0.0
    assert cc2.min_free_vram_mib == 0
    assert cc2.coexist_policy == "refuse"


def test_handshake_requires_both_keys():
    only_floor = cx.coexist_config(
        ToasterLoopConfig(serve={"min_free_vram_mib": 4096}))
    assert not only_floor.handshake_enabled  # idle_unload off => no rescue
    both = cx.coexist_config(ToasterLoopConfig(
        serve={"min_free_vram_mib": 4096, "idle_unload_seconds": 120}))
    assert both.handshake_enabled


# ------------------------------------------------- watchdog activity clock

def test_note_activity_and_idle_seconds():
    clock = FakeClock(t=50.0)
    assert wd.idle_seconds("http://a/v1", clock=clock) is None  # never seen
    wd.note_activity("http://a/v1/", clock=clock)               # normalizes /
    clock.advance(12.0)
    assert wd.idle_seconds("http://a/v1", clock=clock) == pytest.approx(12.0)
    # a request start refreshes the stamp too (single registry, WP5 reuse)
    wd.note_request_start("http://a/v1", clock=clock)
    assert wd.idle_seconds("http://a/v1", clock=clock) == pytest.approx(0.0)


# ------------------------------------------------------------- idle unload

def build_engine(home, *, idle=60.0, lazy=False, min_free=0, policy="refuse",
                 clock=None):
    cc = cx.CoexistConfig(idle_unload_seconds=idle, lazy_load=lazy,
                          min_free_vram_mib=min_free, coexist_policy=policy)
    events = []
    return cx.Coexistence(cc, home=home, clock=clock or FakeClock(),
                          sleep=lambda s: None,
                          emit=lambda kind, p: events.append((kind, p))), events


def test_idle_unload_fires_once_per_idle_period(home__test_coexist_wp5, monkeypatch):
    make_instance(home__test_coexist_wp5)
    write_live_state(home__test_coexist_wp5)
    monkeypatch.setattr("toaster_loop.serving.runtime.pid_alive", lambda pid: True)
    stops = []
    patch_stop(monkeypatch, stops)

    clock = FakeClock()
    coex, events = build_engine(home__test_coexist_wp5, idle=60.0, clock=clock)
    wd.note_activity(URL, clock=clock)          # seed the window from now

    clock.advance(30.0)
    coex.step()                                   # half the window: nothing
    assert stops == []

    clock.advance(31.0)                           # 61 s idle > 60 s window
    coex.step()
    assert stops == [("stop", "alpha", coex.cfg.unload_grace_seconds)]

    # state file is now an `unloaded` tomb and the event fired exactly once
    st = srv_inst.read_state(Path(home__test_coexist_wp5), "alpha")
    assert st is not None
    assert st["unloaded"] is True and st["unload_reason"] == "idle"
    assert "pid" not in st
    kinds = [p["event"] for k, p in events if p.get("event")]
    assert kinds == ["instance_unloaded"]

    # further steps: nothing running anymore => no second stop ever
    clock.advance(3600.0)
    coex.step()
    coex.step()
    assert len(stops) == 1


def test_idle_never_unloads_while_inflight(home__test_coexist_wp5, monkeypatch):
    make_instance(home__test_coexist_wp5)
    write_live_state(home__test_coexist_wp5)
    monkeypatch.setattr("toaster_loop.serving.runtime.pid_alive", lambda pid: True)
    stops = []
    patch_stop(monkeypatch, stops)

    clock = FakeClock()
    coex, _ = build_engine(home__test_coexist_wp5, idle=60.0, clock=clock)
    wd.note_activity(URL, clock=clock)
    wd.note_request_start(URL, clock=clock)       # request in flight
    clock.advance(10_000.0)
    coex.step()
    assert stops == []                            # never mid-request

    wd.note_request_end(URL)
    coex.step()                                   # idle again -> now allowed
    assert len(stops) == 1


def test_disabled_idle_loop_never_starts(home__test_coexist_wp5):
    coex, _ = build_engine(home__test_coexist_wp5, idle=0.0)
    assert coex.start_idle_loop() is False


# --------------------------------------------------------- lazy load paths

def test_lazy_load_via_resolve_named_instance(home__test_coexist_wp5, monkeypatch):
    make_instance(home__test_coexist_wp5)
    write_unloaded_state(home__test_coexist_wp5)
    spawns = []
    patch_spawn(monkeypatch, spawns)
    cfg = ToasterLoopConfig(serve={"lazy_load": True})
    app = DashboardApp(cfg, home=home__test_coexist_wp5)

    base, _key, model = app._resolve_chat_target("alpha")
    assert base == URL and model == "alpha-model"
    assert ("spawn", "alpha") in spawns
    # the tomb was replaced by a live record
    st = srv_inst.read_state(Path(home__test_coexist_wp5), "alpha")
    assert st is not None and st.get("pid") == os.getpid()


def test_lazy_off_unloaded_gets_clear_error(home__test_coexist_wp5):
    make_instance(home__test_coexist_wp5)
    write_unloaded_state(home__test_coexist_wp5)
    app = DashboardApp(ToasterLoopConfig(), home=home__test_coexist_wp5)
    with pytest.raises(HttpError) as ei:
        app._resolve_chat_target("alpha")
    assert ei.value.status == 503
    assert ei.value.code == "instance_unloaded"
    msg = str(ei.value).lower()
    assert "unloaded" in msg and "lazy_load" in msg


def test_lazy_never_loaded_keeps_legacy_503(home__test_coexist_wp5):
    make_instance(home__test_coexist_wp5)  # definition only, no state file at all
    app = DashboardApp(ToasterLoopConfig(), home=home__test_coexist_wp5)
    with pytest.raises(HttpError) as ei:
        app._resolve_chat_target("alpha")
    assert ei.value.code == "not_serving"   # pre-WP5 message untouched


def test_default_instance_tomb_lazy_and_guidance(home__test_coexist_wp5, monkeypatch):
    from toaster_loop.serving.deploy import write_serve_state

    tomb = {"mode": "spawn", "unloaded": True, "unload_reason": "coexist",
            "port": 8123, "base_url": URL, "model_name": "default-model"}
    write_serve_state(Path(home__test_coexist_wp5), tomb)
    app = DashboardApp(ToasterLoopConfig(), home=home__test_coexist_wp5)
    with pytest.raises(HttpError) as ei:
        app._resolve_chat_target()
    assert ei.value.code == "instance_unloaded"

    # lazy on: the default instance reloads through the canonical path
    # (the config-derived default needs a gguf_path to be spawnable).
    spawns = []
    patch_spawn(monkeypatch, spawns)
    cfg2 = ToasterLoopConfig(serve={"lazy_load": True},
                         gguf_path="/nonexistent/default.gguf")
    app2 = DashboardApp(cfg2, home=home__test_coexist_wp5)
    base, _k, model = app2._resolve_chat_target()
    # the fake handle reports its own model name; what matters is that the
    # tomb was transparently reloaded at the recorded URL.
    assert base == URL and model
    assert ("spawn", "default") in spawns


def test_agent_runner_reloads_once_on_dead_endpoint(home__test_coexist_wp5, monkeypatch):
    from toaster_loop.core.agent_runner import AgentBackendError, ToasterLoopAgentClient

    client = ToasterLoopAgentClient.__new__(ToasterLoopAgentClient)
    client.base_url = URL
    calls: list = []

    def inner(messages, tools=None):
        calls.append(1)
        if len(calls) == 1:
            raise AgentBackendError("Cannot reach inference server")
        return {"ok": True}
    client._chat_inner = inner

    # no active engine => the original error propagates untouched (pre-WP5)
    monkeypatch.setattr("toaster_loop.serving.coexist.active", lambda: None)
    with pytest.raises(AgentBackendError):
        client._chat([{"role": "user", "content": "hi"}])
    assert len(calls) == 1

    # active engine WITH the URL in unloaded state => reload once + retry once
    reloads = []
    calls.clear()   # next inner call raises (len==1), retry succeeds (len==2)

    class FakeCoex:
        def reload_for_url(self, url):
            reloads.append(url)
            return True
    monkeypatch.setattr("toaster_loop.serving.coexist.active",
                        lambda: FakeCoex())
    assert client._chat([{"role": "user", "content": "hi"}]) == {"ok": True}
    assert reloads == [URL] and len(calls) == 2   # exactly one retry

    # engine present but the URL is NOT an unloaded instance => no retry
    calls.clear()
    reloads.clear()

    class NoReloadCoex:
        def reload_for_url(self, url):
            return False
    monkeypatch.setattr("toaster_loop.serving.coexist.active",
                        lambda: NoReloadCoex())
    with pytest.raises(AgentBackendError):
        client._chat([{"role": "user", "content": "hi"}])
    assert reloads == [] and len(calls) == 1


def test_agent_runner_hook_is_inert_without_engine(home__test_coexist_wp5, monkeypatch):
    from toaster_loop.core.agent_runner import AgentBackendError, ToasterLoopAgentClient

    monkeypatch.setattr("toaster_loop.serving.coexist.active", lambda: None)
    client = ToasterLoopAgentClient.__new__(ToasterLoopAgentClient)
    client.base_url = URL

    def inner(messages, tools=None):
        raise AgentBackendError("down")
    client._chat_inner = inner
    with pytest.raises(AgentBackendError):
        client._chat([])   # raises exactly as pre-WP5


# ------------------------------------------- explicit endpoints + CLI round

def test_dashboard_unload_load_status_roundtrip(home__test_coexist_wp5, monkeypatch):
    make_instance(home__test_coexist_wp5)
    write_live_state(home__test_coexist_wp5)
    monkeypatch.setattr("toaster_loop.serving.runtime.pid_alive", lambda pid: True)
    patch_stop(monkeypatch, [])
    spawns = []
    patch_spawn(monkeypatch, spawns)
    app = DashboardApp(ToasterLoopConfig(), home=home__test_coexist_wp5)

    row = app.handle("GET", "/api/servers/alpha/status")
    assert row.body["load_state"] == "loaded"

    res = app.handle("POST", "/api/servers/alpha/unload", body={})
    assert res.body["unloaded"] is True
    assert res.body["state"] == "unloaded"
    row = app.handle("GET", "/api/servers/alpha/status")
    assert row.body["load_state"] == "unloaded"
    assert row.body["running"] is False

    res = app.handle("POST", "/api/servers/alpha/load", body={})
    assert res.status == 202 and res.body["state"] == "loaded"
    assert ("spawn", "alpha") in spawns
    row = app.handle("GET", "/api/servers/alpha/status")
    assert row.body["load_state"] == "loaded"

    # loading an already-loaded instance is idempotent (200, no new spawn)
    res = app.handle("POST", "/api/servers/alpha/load", body={})
    assert res.status == 200
    assert spawns.count(("spawn", "alpha")) == 1


def test_dashboard_unload_inflight_is_retryable_409(home__test_coexist_wp5, monkeypatch):
    make_instance(home__test_coexist_wp5)
    write_live_state(home__test_coexist_wp5)
    stops = []
    patch_stop(monkeypatch, stops)
    app = DashboardApp(ToasterLoopConfig(), home=home__test_coexist_wp5)
    wd.note_request_start(URL)
    # handle() converts HttpError into the JSON error envelope (status 409)
    res = app.handle("POST", "/api/servers/alpha/unload", body={})
    assert res.status == 409
    assert res.body["error"]["code"] == "request_in_flight"
    assert stops == []


def test_cli_instances_actions_round_trip(home__test_coexist_wp5, monkeypatch, capsys):
    from toaster_loop.cli import cmd_instances_load, cmd_instances_status, \
        cmd_instances_unload

    make_instance(home__test_coexist_wp5)
    write_live_state(home__test_coexist_wp5)
    monkeypatch.setattr("toaster_loop.serving.runtime.pid_alive", lambda pid: True)
    patch_stop(monkeypatch, [])
    spawns = []
    patch_spawn(monkeypatch, spawns)
    monkeypatch.setattr("toaster_loop.cli.load_config",
                        lambda *a, **k: ToasterLoopConfig())

    assert cmd_instances_status(_Args("status")) == 0
    out = capsys.readouterr().out
    assert "alpha" in out and "loaded" in out

    assert cmd_instances_unload(_Args("unload", "alpha")) == 0
    assert "unloaded" in capsys.readouterr().out
    assert cmd_instances_status(_Args("status")) == 0
    assert "unloaded" in capsys.readouterr().out

    assert cmd_instances_load(_Args("load", "alpha")) == 0
    assert ("spawn", "alpha") in spawns
    assert cmd_instances_unload(_Args("unload", "nope")) == 1   # unknown id


class _Args(argparse.Namespace):
    def __init__(self, command, instance_id=None):
        super().__init__()
        self.command = command
        self.instances_command = command
        self.instance_id = instance_id
        self.config = None
        self.verbose = False


# --------------------------------------------------- coexistence handshake

def scripted_free(seq):
    """Fake gpu_free_mib: return seq values in order, repeat the last."""
    state = {"i": 0}

    def probe():
        i = min(state["i"], len(seq) - 1)
        state["i"] += 1
        return seq[i]
    return probe


def rescue_engine(home, monkeypatch, *, policy, free_seq, est=9000,
                  targets=None):
    cc = cx.CoexistConfig(idle_unload_seconds=60.0, min_free_vram_mib=10_000,
                          coexist_policy=policy,
                          vram_poll_interval_seconds=0.0)
    events: list = []
    coex = cx.Coexistence(cc, home=home, clock=FakeClock(),
                          sleep=lambda s: None,
                          free_vram=scripted_free(free_seq),
                          emit=lambda k, p: events.append(p))
    if targets is not None:
        monkeypatch.setattr(coex, "running_targets", lambda: list(targets))
    monkeypatch.setattr("toaster_loop.serving.instances.estimate_model_vram_mib",
                        lambda gguf, ctx=None: est)
    return coex, events


def test_rescue_refuse_emits_event_with_headroom(home__test_coexist_wp5, monkeypatch):
    targets = [{"id": "alpha", "pid": 1, "base_url": URL,
                "gguf_path": "/m.gguf", "context_length": None}]
    # est 20_000: free 500 < required 10_000 BUT projected 20_500 >= it =>
    # the "would pass if unloaded" branch under policy=refuse.
    coex, events = rescue_engine(home__test_coexist_wp5, monkeypatch, policy="refuse",
                                 free_seq=[500], targets=targets, est=20_000)
    unloaded = []
    monkeypatch.setattr(coex, "unload_all",
                        lambda **kw: unloaded.append(1) or [])

    assert coex.rescue_gate("job1", 10_000) is False
    assert unloaded == []                       # refuse => nothing stopped
    refused = [e for e in events if e.get("event") == "coexist_refused"]
    assert len(refused) == 1
    ev = refused[0]
    assert ev["free_mib"] == 500 and ev["required_mib"] == 10_000
    assert ev["projected_free_mib"] == 20_500   # 500 + one unloaded instance
    assert ev["would_pass_if_unloaded"] is True


def test_rescue_unload_triggers_waits_and_admits(home__test_coexist_wp5, monkeypatch):
    targets = [{"id": "alpha", "pid": 1, "base_url": URL,
                "gguf_path": "/m.gguf", "context_length": None}]
    # probe: still busy right after unload, freed on the next poll; the
    # projection (500 + 20_000 est) clears the 10_000 bar.
    coex, events = rescue_engine(home__test_coexist_wp5, monkeypatch, policy="unload",
                                 free_seq=[500, 500, 12_000],
                                 targets=targets, est=20_000)
    calls = []

    def fake_unload_all(**kw):
        calls.append(kw.get("reason"))
        return ["alpha"]
    monkeypatch.setattr(coex, "unload_all", fake_unload_all)

    assert coex.rescue_gate("job1", 10_000) is True
    assert calls == ["coexist"]
    trig = [e for e in events if e.get("event") == "coexist_unload_triggered"]
    assert len(trig) == 1 and trig[0]["targets"] == ["alpha"]
    assert trig[0]["free_mib"] == 500 and trig[0]["required_mib"] == 10_000


def test_rescue_still_insufficient_refuses_without_unload(home__test_coexist_wp5, monkeypatch):
    targets = [{"id": "alpha", "pid": 1, "base_url": URL,
                "gguf_path": "/m.gguf", "context_length": None}]
    coex, events = rescue_engine(home__test_coexist_wp5, monkeypatch, policy="unload",
                                 free_seq=[500], targets=targets, est=1000)
    calls = []
    monkeypatch.setattr(coex, "unload_all",
                        lambda **kw: calls.append(1) or [])
    # 500 + 1000 projected < 10_000 required => refuse even under unload
    assert coex.rescue_gate("job1", 10_000) is False
    assert calls == []
    assert events == []


def test_rescue_disabled_never_touches_anything(home__test_coexist_wp5, monkeypatch):
    coex, _ = rescue_engine(home__test_coexist_wp5, monkeypatch, policy="unload",
                            free_seq=[500])
    coex.cfg = cx.CoexistConfig(coexist_policy="unload")  # handshake off
    calls = []
    monkeypatch.setattr(coex, "unload_all", lambda **kw: calls.append(1) or [])
    assert coex.rescue_gate("job1", 10_000) is False
    assert calls == []


# --------------------------------- scheduler gate matrix (end-to-end-ish)

@pytest.fixture()
def registry__test_coexist_wp5(home__test_coexist_wp5):
    reg = ModelRegistry(Path(home__test_coexist_wp5) / "buffer" / "toaster_loop.db")
    yield reg
    reg.close()


class RecordingRescue:
    def __init__(self, admit_after):
        self.calls: list = []
        self.admit_after = admit_after   # return value of rescue_gate

    def rescue_gate(self, job_id, required_mib):
        self.calls.append((job_id, required_mib))
        return self.admit_after


def _sched(registry, monkeypatch, *, coexist, free_seq):
    sched = Scheduler(registry=registry,
                      cfg=TrainingConfig(min_free_vram_mib=10_000),
                      coexist=coexist)
    monkeypatch.setattr("toaster_loop.training.scheduler.gpu_free_mib",
                        scripted_free(free_seq))
    spawned: list = []
    sched._spawn_worker = lambda job, slot: spawned.append(job.job_id)
    return sched, spawned


def test_gate_matrix_refuse_exactly_as_today(registry__test_coexist_wp5, monkeypatch):
    # min_free set but NO coexist wired (default) -> plain refusal, unchanged
    sched, spawned = _sched(registry__test_coexist_wp5, monkeypatch, coexist=None,
                            free_seq=[500])
    job = sched.submit("sft", log_dir=None)
    with pytest.raises(SchedulerError) as ei:
        sched.run_job(job)
    assert "resource gate blocked start" in str(ei.value)
    assert spawned == []


def test_gate_matrix_refuse_policy_blocks_but_records(registry__test_coexist_wp5, monkeypatch):
    rescue = RecordingRescue(admit_after=False)   # policy=refuse behaviour
    sched, spawned = _sched(registry__test_coexist_wp5, monkeypatch, coexist=rescue,
                            free_seq=[500])
    job = sched.submit("sft", log_dir=None)
    with pytest.raises(SchedulerError):
        sched.run_job(job)
    assert rescue.calls == [(job.job_id, 10_000)]  # consulted exactly once
    assert spawned == []


def test_gate_matrix_unload_policy_admits_after_rescue(registry__test_coexist_wp5, monkeypatch):
    rescue = RecordingRescue(admit_after=True)
    # first probe fails the gate; after the (faked) unload the re-check passes
    sched, spawned = _sched(registry__test_coexist_wp5, monkeypatch, coexist=rescue,
                            free_seq=[500, 12_000])
    job = sched.submit("sft", log_dir=None)
    out = sched.run_job(job)
    assert out.state == "running" or spawned == [job.job_id]
    assert spawned == [job.job_id]
    assert rescue.calls == [(job.job_id, 10_000)]


def test_gate_passes_untouched_when_vram_sufficient(registry__test_coexist_wp5, monkeypatch):
    rescue = RecordingRescue(admit_after=True)
    sched, spawned = _sched(registry__test_coexist_wp5, monkeypatch, coexist=rescue,
                            free_seq=[50_000])
    job = sched.submit("sft", log_dir=None)
    sched.run_job(job)
    assert spawned == [job.job_id]
    assert rescue.calls == []          # gate passed => rescue never consulted


# ------------------------------------------------- race-safety (documented)

def test_load_serializes_behind_unload_no_double_launch(home__test_coexist_wp5, monkeypatch):
    """Documented race policy: unload/load take a per-instance lock. A load
    that arrives mid-unload blocks and then reloads exactly once — no
    concurrent double-launch."""
    make_instance(home__test_coexist_wp5)
    write_unloaded_state(home__test_coexist_wp5)
    spawns = []
    spawn_gate = threading.Event()

    def slow_spawn(cfg, inst, **kw):
        spawns.append(str(inst.get("id")))
        spawn_gate.wait(timeout=5)
        return FakeHandle()
    monkeypatch.setattr("toaster_loop.serving.supervisor.start_instance_server",
                        slow_spawn)

    coex, _ = build_engine(home__test_coexist_wp5, idle=60.0, lazy=True)
    results: list = []

    def loader():
        results.append(coex.load_instance("alpha", reason="lazy"))

    t1 = threading.Thread(target=loader)
    t1.start()
    # first loader is inside load_instance (spawn blocked); fire a second
    t2 = threading.Thread(target=loader)
    t2.start()
    time.sleep(0.3)
    assert len(spawns) == 1            # the second caller is still queued
    spawn_gate.set()
    t1.join(timeout=5)
    t2.join(timeout=5)
    assert not t1.is_alive() and not t2.is_alive()
    # exactly ONE launch: the queued caller found the live record and no-op'd
    assert spawns == ["alpha"]
    assert len(results) == 2


def test_request_racing_unload_gets_retryable_or_reloads(home__test_coexist_wp5, monkeypatch):
    """lazy off => the racing request surfaces the clear unloaded error;
    lazy on => it transparently reloads. Both are the documented choices."""
    make_instance(home__test_coexist_wp5)
    write_unloaded_state(home__test_coexist_wp5)
    patch_spawn(monkeypatch, [])

    app_off = DashboardApp(ToasterLoopConfig(), home=home__test_coexist_wp5)
    with pytest.raises(HttpError) as ei:
        app_off._resolve_chat_target("alpha")
    assert ei.value.code == "instance_unloaded"

    app_on = DashboardApp(ToasterLoopConfig(serve={"lazy_load": True}), home=home__test_coexist_wp5)
    base, _k, _m = app_on._resolve_chat_target("alpha")
    assert base == URL


def test_default_off_proof_engine_inert(home__test_coexist_wp5):
    """With no config changes: engine constructed but every automatic path is
    off — idle loop refuses to start, rescue declines, guidance is None."""
    coex = cx.build(ToasterLoopConfig(), home__test_coexist_wp5)
    assert coex is None                       # not even constructed

    lazy_off = cx.Coexistence(cx.CoexistConfig(), home=home__test_coexist_wp5)
    assert lazy_off.start_idle_loop() is False
    assert lazy_off.rescue_gate("j", 10) is False
    assert lazy_off.ensure_loaded("alpha") is None
    assert lazy_off.unloaded_guidance("alpha") is None
    assert lazy_off.reload_for_url(URL) is False

# ------------------------------------------------------------------------
# from tests/test_watchdog_wp4.py
# ------------------------------------------------------------------------

# ------------------------------------------------------------------ helpers



def sample(cpu_s: float, io_bytes: int, *, source: str = "read_write",
           t: float = 0.0) -> wd.ProcSample:
    return wd.ProcSample(t=t, cpu_s=cpu_s, io_bytes=io_bytes,
                         io_source=source)


def make_target(iid: str = "wdtest", pid: int = 424242,
                url: str = "http://127.0.0.1:9/v1") -> wd._WatchTarget:
    return wd._WatchTarget(instance_id=iid, pid=pid, base_url=url)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """Watchdog harness: fake clock/sleep, scripted proc samples, event log,
    restart recorder. Ownership verdict controlled per test via the returned
    namespace's `.kill_verdict` (read at call time)."""

    class Harness:
        def __init__(self):
            self.clock = FakeClock()
            self.sleeps: list = []
            self.events: list = []
            self.restarts: list = []
            self.kill_calls: list = []
            self.kill_verdict = "killed"
            self.samples: list = []  # ProcSample | None, consumed per step

        def read_proc(self, pid):
            if not self.samples:
                raise AssertionError("read_proc called with no scripted sample")
            return self.samples.pop(0)

        def emit(self, kind, payload):
            assert kind == "serve.watchdog"
            self.events.append(payload)

        def restart_fn(self, iid):
            self.restarts.append(iid)
            return True

        def terminate_owned_pid(self, pid, grace=8.0):
            self.kill_calls.append((pid, grace))
            return self.kill_verdict

    h = Harness()
    monkeypatch.setattr("toaster_loop.serving.runtime.terminate_owned_pid",
                        h.terminate_owned_pid)
    wd.reset_inflight()
    yield h
    wd.reset_inflight()


def build(envh, *, window=90.0, interval=5.0, max_restarts=3, targets=None):
    cfg = wd.WatchdogConfig(
        enabled=True, window_seconds=window,
        sample_interval_seconds=interval,
        kill_grace_seconds=7.0, max_restarts=max_restarts)
    return wd.EngineWatchdog(
        cfg, home=None, emit=envh.emit, clock=envh.clock,
        sleep=envh.sleeps.append, read_proc=envh.read_proc,
        restart_fn=envh.restart_fn,
        list_targets=(lambda: targets if targets is not None else []))


def event_names(h) -> list:
    return [e["event"] for e in h.events]


# ------------------------------------------------------- pure verdict rules

def test_verdict_silence_with_inflight_is_stalled():
    cfg = wd.WatchdogConfig()
    prev = sample(10.0, 100)
    cur = sample(10.0 + 0.005, 100)  # 5 ms cpu (< 10 ms), io unchanged
    assert wd.freeze_verdict(prev, cur, cfg, inflight_age=95) == wd.V_STALLED


def test_verdict_healthy_but_slow_busy_never_stalls():
    """CPU delta above threshold (model genuinely computing) => BUSY."""
    cfg = wd.WatchdogConfig()
    prev = sample(10.0, 100)
    cur = sample(14.0, 100)  # 4 s cpu in one sample interval
    assert wd.freeze_verdict(prev, cur, cfg, inflight_age=95) == wd.V_BUSY


def test_verdict_io_activity_alone_counts_as_progress():
    cfg = wd.WatchdogConfig()
    prev = sample(10.0, 100)
    cur = sample(10.0, 100 + cfg.io_threshold_bytes + 1)
    assert wd.freeze_verdict(prev, cur, cfg, inflight_age=95) == wd.V_BUSY


def test_verdict_idle_never_triggers():
    """No request in flight: silence is idleness, not a freeze."""
    cfg = wd.WatchdogConfig()
    prev = sample(10.0, 100)
    cur = sample(10.0, 100)
    assert wd.freeze_verdict(prev, cur, cfg, inflight_age=0) == wd.V_IDLE


def test_verdict_missing_evidence_never_stalls():
    cfg = wd.WatchdogConfig()
    prev_none = sample(10.0, 0, source="none")
    cur = sample(10.0, 0, source="read_write")
    assert wd.freeze_verdict(prev_none, cur, cfg, 95) == wd.V_UNKNOWN
    assert wd.freeze_verdict(None, cur, cfg, 95) == wd.V_UNKNOWN


# ------------------------------------------------------- in-flight tracking

def test_track_request_counts_and_age():
    clock = FakeClock(t=50.0)
    assert wd.inflight_count("http://x/v1") == 0
    assert wd.inflight_age("http://x/v1", clock=clock) == 0.0
    # age tracking with the injected clock
    wd.note_request_start("http://x/v1", clock=clock)
    clock.advance(12.0)
    assert wd.inflight_count("http://x/v1") == 1
    assert wd.inflight_age("http://x/v1", clock=clock) == pytest.approx(12.0)
    wd.note_request_end("http://x/v1")
    assert wd.inflight_count("http://x/v1") == 0
    # context manager counts (and normalizes trailing slash), releasing even
    # when the body raises
    with wd.track_request("http://x/v1/"):
        assert wd.inflight_count("http://x/v1") == 1
    with pytest.raises(RuntimeError):
        with wd.track_request("http://x/v1"):
            raise RuntimeError("boom")
    assert wd.inflight_count("http://x/v1") == 0


# ------------------------------------------------- full state machine tests

def test_freeze_detected_kill_verified_then_restart_backoff(env):
    """Silence + zero CPU delta + zero IO delta + request in flight for the
    full window -> freeze_suspect -> freeze_detected -> kill ONLY after the
    ownership helper verifies -> restart scheduled with backoff."""
    clock = env.clock
    url = "http://127.0.0.1:9/v1"
    wd.note_request_start(url, clock=clock)  # request in flight from t=1000

    t = make_target(url=url)
    w = build(env, window=90.0, interval=5.0, targets=[t])

    # step 1: first sample (no previous) -> unknown, arms prev
    env.samples = [sample(10.0, 500)]
    w.step()
    assert event_names(env) == []

    # steps over the window: stalled samples (cpu delta 2 ms < 10 ms, io flat).
    # suspect arms at t=1005; 17 more stalled samples reach t=1090 = 85 s —
    # still short of the 90 s window.
    for i in range(18):
        clock.advance(5.0)
        env.samples = [sample(10.0 + 0.002 * (i + 1), 500)]
        w.step()

    assert event_names(env) == ["freeze_suspect"]  # exactly one suspect event
    assert env.kill_calls == []                    # window NOT yet elapsed

    clock.advance(5.0)  # t = 1095 -> suspect_since 1005, 90 s elapsed
    env.samples = [sample(10.0 + 0.04, 500)]
    w.step()

    assert event_names(env) == ["freeze_suspect", "freeze_detected",
                                "instance_killed", "instance_restarted"]
    # kill went through the SHARED ownership-verified helper with our pid
    assert env.kill_calls == [(424242, 7.0)]
    assert env.restarts == ["wdtest"]
    assert env.sleeps == [2.0]  # first backoff = base * 2^0

    det = env.events[1]
    assert det["instance_id"] == "wdtest" and det["pid"] == 424242
    killed = env.events[2]
    assert killed["verified"].startswith("pid_alive")
    restarted = env.events[3]
    assert restarted["attempt"] == 1 and restarted["backoff_seconds"] == 2.0


def test_busy_compute_never_killed(env):
    """Same silence on the wire, but CPU delta > threshold every sample:
    healthy-but-slow must NEVER be killed, even with a request in flight."""
    clock = env.clock
    url = "http://127.0.0.1:9/v1"
    wd.note_request_start(url, clock=clock)
    t = make_target(url=url)
    w = build(env, window=30.0, interval=5.0, targets=[t])

    env.samples = [sample(1.0, 10)]
    w.step()
    for i in range(12):  # way past the 30 s window
        clock.advance(5.0)
        env.samples = [sample(1.0 + 0.5 * (i + 1), 10)]  # CPU burning
        w.step()
    assert env.kill_calls == []
    assert env.restarts == []
    assert env.events == []


def test_no_request_in_flight_never_triggers(env):
    """Zero CPU, zero IO — but nothing in flight: idle, not frozen."""
    clock = env.clock
    t = make_target()
    w = build(env, window=10.0, interval=5.0, targets=[t])

    env.samples = [sample(10.0, 77)]
    w.step()
    for _ in range(6):
        clock.advance(5.0)
        env.samples = [sample(10.0, 77)]
        w.step()
    assert env.kill_calls == []
    assert env.events == []


def test_request_completing_mid_window_clears_suspect(env):
    """Stall starts, then the server wakes up and burns CPU before the window
    elapses -> suspect cleared, nothing killed."""
    clock = env.clock
    url = "http://127.0.0.1:9/v1"
    wd.note_request_start(url, clock=clock)
    t = make_target(url=url)
    w = build(env, window=30.0, interval=5.0, targets=[t])

    env.samples = [sample(5.0, 10)]
    w.step()
    clock.advance(5.0)
    env.samples = [sample(5.0, 10)]  # stalled -> suspect
    w.step()
    assert event_names(env) == ["freeze_suspect"]

    clock.advance(5.0)
    env.samples = [sample(9.0, 10)]  # CPU back on: recovered
    w.step()
    for i in range(8):  # long past the window — CPU keeps burning every sample
        clock.advance(5.0)
        env.samples = [sample(9.0 + 0.4 * (i + 2), 10)]
        w.step()
    assert env.kill_calls == []
    assert "freeze_detected" not in event_names(env)


def test_foreign_pid_never_killed_stale_state_cleared(env, tmp_path):
    """Ownership check fails (foreign/recycled pid): NEVER kill — log and
    clear the stale state file instead; no restart onto a pid we don't own."""
    from toaster_loop.serving import instances as srv_inst

    clock = env.clock
    url = "http://127.0.0.1:9/v1"
    wd.note_request_start(url, clock=clock)
    t = make_target(iid="wdstale", pid=654321, url=url)

    # stale state file present (would be cleared on the foreign verdict)
    srv_inst.write_state(tmp_path, "wdstale", {"mode": "spawn", "pid": 654321,
                                               "base_url": url})
    env.kill_verdict = "foreign"
    w = build(env, window=10.0, interval=5.0, targets=[t])
    # give the watchdog a home so state clearing has somewhere to look
    w.home = tmp_path

    env.samples = [sample(3.0, 42)]
    w.step()
    for i in range(3):
        clock.advance(5.0)
        env.samples = [sample(3.0, 42)]
        w.step()

    assert event_names(env) == ["freeze_suspect", "freeze_detected",
                                "stale_state_cleared"]
    # the helper was consulted (that IS the ownership verification) and its
    # foreign verdict meant no kill signal was ever sent by us
    assert env.kill_calls == [(654321, 7.0)]
    assert env.restarts == []                       # never restart onto it
    assert srv_inst.read_state(tmp_path, "wdstale") is None  # state cleared


def test_backoff_sequence_and_restart_cap(env):
    """Backoff doubles per attempt; at max_restarts within the window a loud
    restart_capped event fires and further freezes do nothing (no loop)."""
    clock = env.clock
    url = "http://127.0.0.1:9/v1"
    wd.note_request_start(url, clock=clock)

    def fresh_target():
        return [make_target(iid="wdcap", pid=11111, url=url)]

    w = build(env, window=10.0, interval=5.0, max_restarts=3,
              targets=fresh_target())

    def one_freeze_cycle():
        env.samples = [sample(2.0, 8)]
        w.step()
        stalled_first = False
        for _ in range(4):
            clock.advance(5.0)
            env.samples = [sample(2.0, 8)]
            w.step()
            stalled_first = True
        assert stalled_first

    for attempt in range(1, 4):
        one_freeze_cycle()
        assert len(env.restarts) == attempt
    assert env.sleeps == [2.0, 4.0, 8.0]  # base * 2^(attempt-1)

    # 4th freeze inside the window: capped — loud event, no kill-loop
    one_freeze_cycle()
    names = event_names(env)
    assert names.count("restart_capped") == 1
    assert len(env.restarts) == 3          # no 4th restart
    assert env.sleeps == [2.0, 4.0, 8.0]   # no extra backoff sleep
    cap = next(e for e in env.events if e["event"] == "restart_capped")
    assert cap["instance_id"] == "wdcap" and cap["restarts"] == 3

    # and it stays capped: another full cycle changes nothing
    one_freeze_cycle()
    assert len(env.restarts) == 3


def test_restart_failure_surfaced_not_swallowed(env):
    clock = env.clock
    url = "http://127.0.0.1:9/v1"
    wd.note_request_start(url, clock=clock)
    t = make_target(iid="wdfail", pid=22222, url=url)

    def boom(iid):
        raise RuntimeError("spawn exploded")

    w = build(env, window=10.0, interval=5.0, targets=[t])
    w._restart_fn = boom

    env.samples = [sample(1.0, 1)]
    w.step()
    for _ in range(3):
        clock.advance(5.0)
        env.samples = [sample(1.0, 1)]
        w.step()
    names = event_names(env)
    assert names == ["freeze_suspect", "freeze_detected",
                     "instance_killed", "restart_failed"]
    fail = env.events[-1]
    assert "spawn exploded" in fail["error"] and fail["attempt"] == 1


# -------------------------------------------------------------- config gate

def test_config_defaults_off_and_start_watchdog_noop():
    """With NO config changes the watchdog does nothing at all: defaults are
    off and start_watchdog returns None without spawning a thread."""
    cfg = ToasterLoopConfig()
    wc = wd.watchdog_config(cfg)
    assert wc.enabled is False
    assert wc.window_seconds == 90.0
    assert wd.start_watchdog(cfg, home=None) is None

    # explicit false also off
    cfg2 = ToasterLoopConfig(serve={"watchdog_enabled": False})
    assert wd.watchdog_config(cfg2).enabled is False
    assert wd.start_watchdog(cfg2, home=None) is None


def test_engine_start_refuses_when_disabled():
    w = wd.EngineWatchdog(wd.WatchdogConfig(enabled=False))
    assert w.start() is False


def test_opt_in_enables_but_proc_rules_apply(monkeypatch):
    cfg = ToasterLoopConfig(serve={"watchdog_enabled": "true",
                               "watchdog_window_seconds": 12})
    wc = wd.watchdog_config(cfg)
    assert wc.enabled is True and wc.window_seconds == 12.0

    # unavailable /proc -> start() logs once and declines (never kills)
    monkeypatch.setattr(wd, "proc_sampling_available", lambda: False)
    w = wd.EngineWatchdog(wc)
    assert w.start() is False


# ------------------------------------------------------------- /proc reader

class _Inline:
    """File-object stand-in whose read_text returns fixed content."""

    def __init__(self, text: str):
        self._t = text

    def read_text(self, errors="ignore"):
        return self._t


def _fake_proc(files: dict):
    """Path replacement for the watchdog module: known /proc keys resolve to
    inline content, anything else raises (unreachable proc)."""

    def fake_path(p):
        key = str(p)
        if key in files:
            return _Inline(files[key])
        raise OSError(f"no such file (test): {key}")

    return fake_path


def test_read_proc_sample_parses_stat_and_io(monkeypatch):
    files = {
        "/proc/7/stat": "7 (llama-server) S 1 7 0 0 -1 4194304 100 0 0 0 "
                        "1234 567 0 0 20 0 40 0 99999 0 0",
        "/proc/7/io": ("rchar: 100\nwchar: 50\nread_bytes: 300\n"
                       "write_bytes: 200\ncancelled_write_bytes: 0\n"),
    }
    monkeypatch.setattr(wd, "Path", _fake_proc(files))

    s = wd.read_proc_sample(7, clock=lambda: 5.0, hz=100)
    assert s is not None
    assert s.cpu_s == pytest.approx((1234 + 567) / 100.0)
    assert s.io_bytes == 500 and s.io_source == "read_write"


def test_read_proc_sample_io_fallback_to_rchar(monkeypatch):
    files = {
        "/proc/8/stat": "8 (srv) S 1 8 0 0 -1 0 0 0 0 0 10 20 0 0 20 0 1 0 0 0 0",
        "/proc/8/io": "rchar: 40\nwchar: 5\ncancelled_write_bytes: 0\n",
    }
    monkeypatch.setattr(wd, "Path", _fake_proc(files))
    s = wd.read_proc_sample(8, clock=lambda: 1.0, hz=100)
    assert s is not None
    assert s.cpu_s == pytest.approx(0.3)
    assert s.io_bytes == 45 and s.io_source == "rchar_wchar"


def test_read_proc_sample_unreadable_io_is_none_source(monkeypatch):
    files = {
        "/proc/9/stat": "9 (srv) S 1 9 0 0 -1 0 0 0 0 0 5 5 0 0 20 0 1 0 0 0 0",
    }
    monkeypatch.setattr(wd, "Path", _fake_proc(files))
    s = wd.read_proc_sample(9, clock=lambda: 0.0, hz=100)
    assert s is not None
    assert s.io_source == "none"
    # and the verdict refuses to kill on missing io evidence
    cfg = wd.WatchdogConfig()
    assert wd.freeze_verdict(s, s, cfg, inflight_age=999) == wd.V_UNKNOWN


def test_read_proc_sample_stat_comm_with_parens(monkeypatch):
    """comm may contain spaces AND parens — parsing must split on the LAST ')'."""
    stat = "42 (weird (na)me srv) S 1 42 0 0 -1 0 0 0 0 0 77 33 0 0 20 0 1 0 0 0 0"
    monkeypatch.setattr(wd, "Path", _fake_proc({"/proc/42/stat": stat}))
    s = wd.read_proc_sample(42, clock=lambda: 0.0, hz=100)
    assert s is not None
    assert s.cpu_s == pytest.approx((77 + 33) / 100.0)


def test_read_proc_sample_gone_pid():
    # pid that cannot exist -> None (never raises)
    assert wd.read_proc_sample(2_147_483_647, clock=lambda: 0.0) is None

# ------------------------------------------------------------------------
# from tests/test_scheduler.py
# ------------------------------------------------------------------------

STUB_WORKER = r'''
import argparse, json, os, sys, tempfile, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--job", required=True)
ap.add_argument("--kind", required=True)
ap.add_argument("--config", default="")
ap.add_argument("--dataset-dir", default=None)
ap.add_argument("--base-model", default=None)
ap.add_argument("--log-path", default=None)
ap.add_argument("--adapter-dir", default=None)
ap.add_argument("--gguf-out", default=None)
ap.add_argument("--quant", default="q4_k_m")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

def emit(ev, **kw):
    line = json.dumps({"event": ev, "job": a.job, **kw})
    print(line, flush=True)
    if a.log_path:
        Path(a.log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(a.log_path, "a") as f:
            f.write(line + "\n")

# behaviour knobs via env (per-invocation in tests)
mode = os.environ.get("STUB_MODE", "ok")
if mode == "sleep":
    emit("start")
    time.sleep(float(os.environ.get("STUB_SLEEP", "30")))
    emit("done")
elif mode == "fail":
    emit("error", error="stub failure")
    sys.exit(3)
else:
    emit("start")
    # produce artifacts so pipeline can find them. NEVER fall back to the CWD:
    # jobs submitted without a train.yaml must not leak an adapter/ dir into
    # whatever directory pytest happens to run from (use a temp dir instead).
    if a.kind in ("sft", "dpo", "grpo"):
        out = Path(a.config).parent if a.config else Path(tempfile.mkdtemp())
        (out / "adapter").mkdir(parents=True, exist_ok=True)
        (out / "adapter" / "adapter_config.json").write_text("{}")
    elif a.kind == "export":
        if a.gguf_out:
            gg = Path(a.gguf_out)
            gg.parent.mkdir(parents=True, exist_ok=True)
            gg.write_bytes(b"GGUF-stub")
    emit("done")
'''


@pytest.fixture()
def home__test_scheduler(tmp_path, monkeypatch):
    h = tmp_path / "uf4"
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(h))
    return h


@pytest.fixture()
def registry__test_scheduler(home__test_scheduler):
    reg = ModelRegistry(home__test_scheduler / "buffer" / "toaster_loop.db")
    yield reg
    reg.close()


@pytest.fixture()
def stub(tmp_path):
    p = tmp_path / "stub_worker.py"
    p.write_text(STUB_WORKER)
    return str(p)


def make_sched(registry, stub, env=None):
    sched = Scheduler(
        registry=registry,
        cfg=TrainingConfig(),
        worker_command=[sys.executable, stub],
    )
    return sched


def submit(sched, registry, kind="sft"):
    return sched.submit(kind, dataset_id="ds1", log_dir=Path("/tmp") / "uf4logs")


# ------------------------------------------------------------------- ordering

def test_fifo_order(registry__test_scheduler, stub):
    sched = make_sched(registry__test_scheduler, stub)
    j1 = sched.submit("sft", log_dir=None)
    j2 = sched.submit("export", log_dir=None)
    assert registry__test_scheduler.next_queued().job_id == j1.job_id
    results = sched.run_pending()
    assert [r.job_id for r in results] == [j1.job_id, j2.job_id]
    assert all(r.state == "done" for r in results)


def test_single_flight(registry__test_scheduler, stub, monkeypatch):
    """A running job blocks a second from starting (second waits/raises)."""
    sched = make_sched(registry__test_scheduler, stub)
    monkeypatch.setenv("STUB_MODE", "sleep")
    monkeypatch.setenv("STUB_SLEEP", "5")
    j1 = sched.submit("sft")
    import threading

    t = threading.Thread(target=lambda: sched.run_job(j1), daemon=True)
    t.start()
    # wait until job1 is marked running
    for _ in range(100):
        if registry__test_scheduler.get_job(j1.job_id).state == "running":
            break
        time.sleep(0.05)

    j2 = sched.submit("sft")
    with pytest.raises(SchedulerError):
        sched.run_job(j2)  # refuses to co-schedule on one GPU
    assert registry__test_scheduler.get_job(j2.job_id).state == "queued"

    monkeypatch.setenv("STUB_MODE", "ok")  # affects future spawns only
    sched.cancel(j1.job_id)
    t.join(timeout=10)


def test_parallel_slots_allow_concurrent_jobs(registry__test_scheduler, stub, monkeypatch):
    """max_parallel_jobs=2: two jobs co-schedule on device slots 0 and 1;
    a third is refused. Device env comes from the slot mapping."""
    sched = Scheduler(
        registry=registry__test_scheduler,
        cfg=TrainingConfig(max_parallel_jobs=2, gpu_ids=[5, 6]),
        worker_command=[sys.executable, stub],
    )
    monkeypatch.setenv("STUB_MODE", "sleep")
    monkeypatch.setenv("STUB_SLEEP", "60")
    import threading

    j1 = sched.submit("sft")
    j2 = sched.submit("sft")
    t1 = threading.Thread(target=lambda: sched.run_job(j1), daemon=True)
    t2 = threading.Thread(target=lambda: sched.run_job(j2), daemon=True)
    t1.start(); t2.start()
    states: set = set()
    for _ in range(100):
        states = {registry__test_scheduler.get_job(j.job_id).state for j in (j1, j2)}
        if states == {"running"}:
            break
        time.sleep(0.05)
    assert states == {"running"}, "both slots should run concurrently"
    # device slots recorded from the gpu_ids mapping
    devs = sorted(registry__test_scheduler.get_job(j.job_id).params["_device"] for j in (j1, j2))
    assert devs == [0, 1]

    j3 = sched.submit("sft")
    with pytest.raises(SchedulerError):
        sched.run_job(j3)  # all slots busy

    sched.cancel(j1.job_id); sched.cancel(j2.job_id)
    t1.join(timeout=10); t2.join(timeout=10)


def test_device_env_pinned_per_slot(registry__test_scheduler, stub, monkeypatch, tmp_path):
    """The worker subprocess sees CUDA/HIP_VISIBLE_DEVICES for its slot."""
    probe = tmp_path / "env_probe.py"
    # run_job redirects the worker's stdout to DEVNULL when a log path is
    # set, so the probe records what it saw into TOASTER_LOOP_PROBE_OUT instead.
    # Output file is named after the device the probe SAW, from a dir set
    # once — no per-thread env var to race on between submit and spawn.
    probe.write_text(
        "import json, os, time\n"
        "d = os.environ.get('TOASTER_LOOP_PROBE_DIR')\n"
        "if d:\n"
        "    cuda = os.environ.get('CUDA_VISIBLE_DEVICES', 'none')\n"
        "    with open(os.path.join(d, cuda + '.json'), 'w') as f:\n"
        "        json.dump({'cuda': cuda,"
        " 'hip': os.environ.get('HIP_VISIBLE_DEVICES')}, f)\n"
        "time.sleep(1.5)  # hold the slot so the second claim overlaps\n"
    )
    sched = Scheduler(
        registry=registry__test_scheduler,
        cfg=TrainingConfig(max_parallel_jobs=2, gpu_ids=[3, 7]),
        worker_command=[sys.executable, str(probe)],
    )
    j1 = sched.submit("sft", log_dir=None)
    j2 = sched.submit("sft", log_dir=None)

    outs_dir = tmp_path / "probe"
    outs_dir.mkdir()
    monkeypatch.setenv("TOASTER_LOOP_PROBE_DIR", str(outs_dir))
    # Both must run CONCURRENTLY: sequentially, the second job would reuse
    # the freed slot 0. The probe writes <device>.json for the device it saw.
    import threading

    def run(j):
        try:
            sched.run_job(j)
        except Exception:
            pass

    t1 = threading.Thread(target=run, args=(j1,))
    t2 = threading.Thread(target=run, args=(j2,))
    t1.start(); t2.start(); t1.join(timeout=30); t2.join(timeout=30)

    seen = sorted(p.stem for p in outs_dir.glob("*.json"))
    assert seen == ["3", "7"], f"expected one worker per device, saw {seen}"
    for p in outs_dir.glob("*.json"):
        data = json.loads(p.read_text())
        assert data["cuda"] == data["hip"]


def test_cancel_running_sigterms_worker(registry__test_scheduler, stub, monkeypatch):
    sched = make_sched(registry__test_scheduler, stub)
    monkeypatch.setenv("STUB_MODE", "sleep")
    monkeypatch.setenv("STUB_SLEEP", "60")
    j = sched.submit("sft")
    import threading

    t = threading.Thread(target=lambda: sched.run_job(j), daemon=True)
    t.start()
    job = None
    for _ in range(100):
        job = registry__test_scheduler.get_job(j.job_id)
        if job.state == "running" and job.pid:
            break
        time.sleep(0.05)
    assert job is not None and job.pid
    pid = job.pid
    sched.cancel(j.job_id)
    t.join(timeout=15)
    assert registry__test_scheduler.get_job(j.job_id).state == "cancelled"
    # worker process gone (SIGTERM'd)
    time.sleep(0.3)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_cancel_queued(registry__test_scheduler, stub):
    sched = make_sched(registry__test_scheduler, stub)
    j = sched.submit("sft")
    cancelled = sched.cancel(j.job_id)
    assert cancelled is not None and cancelled.state == "cancelled"
    # run_pending must skip cancelled jobs
    assert sched.run_pending() == []


def test_error_capture_from_failing_worker(registry__test_scheduler, stub, monkeypatch):
    sched = make_sched(registry__test_scheduler, stub)
    monkeypatch.setenv("STUB_MODE", "fail")
    j = sched.submit("sft", log_dir=Path(str(registry__test_scheduler.db_path.parent)) / "runs")
    done = sched.run_job(j)
    assert done.state == "error"
    assert "rc=3" in (done.error or "")
    # worker's own error line captured from the log tail
    assert "stub failure" in (done.error or "")


def test_reap_stale_dead_pid(registry__test_scheduler, stub):
    sched = make_sched(registry__test_scheduler, stub)
    j = sched.submit("sft")
    registry__test_scheduler.mark_running(j.job_id, pid=999_999)  # dead pid
    reaped = sched.reap_stale()
    assert j.job_id in reaped
    assert registry__test_scheduler.get_job(j.job_id).state == "error"


def test_worker_log_written(registry__test_scheduler, stub):
    sched = make_sched(registry__test_scheduler, stub)
    j = sched.submit("sft", log_dir=Path(str(registry__test_scheduler.db_path.parent)) / "runs2")
    sched.run_job(j)
    assert j.log_path is not None
    logp = Path(j.log_path)
    assert logp.is_file()
    events = [json.loads(l)["event"] for l in logp.read_text().splitlines()]
    assert "start" in events and "done" in events


# ---------------------------------------------------------------- auto check

def seed(store, home, n, score, prefix="t"):
    for i in range(n):
        tid = f"{prefix}{i}"
        p = home / "buffer" / f"{tid}.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("")
        store.insert_trace(TraceRow(
            trace_id=tid, session_id=None, timestamp="2026-09-01T00:00:00+00:00",
            agent_profile="fallback", model_alias="m", model_version=None,
            endpoint_label=None, task_summary="s", status="success",
            outcome_score=score, num_tool_calls=0, tokens_in=None,
            tokens_out=None, duration_s=None, jsonl_path=str(p),
        ))


def test_should_auto_train(home__test_scheduler):
    store = TraceStore(home__test_scheduler / "buffer" / "toaster_loop.db")
    d = should_auto_train(store, min_traces=3, min_score=0.5)
    assert d["trigger"] is False and d["unclaimed_traces"] == 0

    seed(store, home__test_scheduler, 2, 0.9)
    d = should_auto_train(store, min_traces=3, min_score=0.5)
    assert d["trigger"] is False  # not enough traces

    seed(store, home__test_scheduler, 1, 0.9, prefix="u")
    d = should_auto_train(store, min_traces=3, min_score=0.5)
    assert d["trigger"] is True and d["mean_score"] == pytest.approx(0.9)

    # low mean -> no trigger
    store2_path = home__test_scheduler / "buffer" / "other.db"
    store2 = TraceStore(store2_path)
    seed(store2, home__test_scheduler, 5, 0.1)
    d = should_auto_train(store2, min_traces=3, min_score=0.5)
    assert d["trigger"] is False
    store.close(); store2.close()


def seed_cancelled(store, home, n, score, prefix="c"):
    """HIGH #4 shape: pre-fix recorder stamped cancelled runs with -0.5."""
    for i in range(n):
        tid = f"{prefix}{i}"
        p = home / "buffer" / f"{tid}.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("")
        store.insert_trace(TraceRow(
            trace_id=tid, session_id=None, timestamp="2026-09-01T00:00:00+00:00",
            agent_profile="fallback", model_alias="m", model_version=None,
            endpoint_label=None, task_summary="s", status="failure",
            outcome_score=score, num_tool_calls=0, tokens_in=None,
            tokens_out=None, duration_s=None, jsonl_path=str(p),
            failure_reason="cancelled",
        ))


def test_should_auto_train_ignores_cancelled_runs(home__test_scheduler):
    """Cancelled runs must not move the mean-score gate (HIGH #4).

    Post-fix recorders NULL cancelled scores at write time; the db.py
    migration retro-NULLs legacy stamped rows on open. Either way, by the
    time should_auto_train reads the buffer the cancelled trace contributes
    no score — the mean stays at the non-cancelled value.
    """
    store = TraceStore(home__test_scheduler / "buffer" / "toaster_loop.db")
    seed(store, home__test_scheduler, 3, 0.9)
    d = should_auto_train(store, min_traces=3, min_score=0.5)
    assert d["trigger"] is True and d["mean_score"] == pytest.approx(0.9)

    # legacy stamped -0.5 cancelled row: the TraceStore-open migration NULLs it
    seed_cancelled(store, home__test_scheduler, 1, -0.5)
    store.close()
    store = TraceStore(home__test_scheduler / "buffer" / "toaster_loop.db")  # reopen -> migration
    d = should_auto_train(store, min_traces=3, min_score=0.5)
    assert d["trigger"] is True
    assert d["mean_score"] == pytest.approx(0.9), \
        "cancelled run must not drag the auto-train mean"
    # it still counts as an unclaimed trace row (count, not score)
    assert d["unclaimed_traces"] == 4 and d["scored_traces"] == 3
    store.close()


def test_resource_gate_skips_without_nvidia(monkeypatch):
    cfg = TrainingConfig(min_free_vram_mib=999_999, require_ac_power=True)
    monkeypatch.setattr("toaster_loop.training.scheduler.gpu_free_mib", lambda: None)
    monkeypatch.setattr("toaster_loop.training.scheduler.on_ac_power", lambda: None)
    assert resource_gate_ok(cfg)["ok"] is True  # unknown => skip gracefully

    monkeypatch.setattr("toaster_loop.training.scheduler.gpu_free_mib", lambda: 100)
    assert resource_gate_ok(cfg)["ok"] is False


# ---------------------------------------------------------- AMD/ROCm VRAM gate

_ROCM_JSON = json.dumps({
    "card0": {
        "GPU": "0",
        "VRAM Total Memory (B)": 103079215104,   # ~98 GiB unified pool
        "VRAM Total Used Memory (B)": 2147483648,  # 2 GiB used
    }
})

_ROCM_TEXT = (
    "GPU[0]\t\t: VRAM Total Memory (B): 51539607552\n"
    "GPU[0]\t\t: VRAM Total Used Memory (B): 51539607552\n"
)


class _Proc:
    def __init__(self, rc, out):
        self.returncode, self.stdout = rc, out


def test_rocm_free_mib_json(monkeypatch):
    import toaster_loop.training.scheduler as sched

    monkeypatch.setattr(sched.shutil, "which", lambda n: "/usr/bin/rocm-smi" if n == "rocm-smi" else None)
    monkeypatch.setattr(sched.subprocess, "run", lambda *a, **k: _Proc(0, _ROCM_JSON))
    # (103079215104 - 2147483648) bytes = 96296 MiB free
    assert sched._rocm_free_mib() == (103079215104 - 2147483648) // (1024 * 1024)


def test_rocm_free_mib_text_fallback(monkeypatch):
    import toaster_loop.training.scheduler as sched

    monkeypatch.setattr(sched.shutil, "which", lambda n: "/usr/bin/rocm-smi" if n == "rocm-smi" else None)
    calls = []

    def fake_run(*a, **k):
        calls.append(a[0])
        # --json unsupported -> rc!=0 first, then text table
        return _Proc(1, "") if "--json" in a[0] else _Proc(0, _ROCM_TEXT)

    monkeypatch.setattr(sched.subprocess, "run", fake_run)
    assert sched._rocm_free_mib() == 0  # fully used pool -> 0 MiB free


def test_gpu_free_mib_falls_through_vendors(monkeypatch):
    import toaster_loop.training.scheduler as sched

    # Vendor-chain semantics only: neutralize the unified-pool probe so
    # this test means the same thing on a TTM/APU box as on a dGPU one.
    from toaster_loop.core import gpumem as _gm
    monkeypatch.setattr(_gm, "unified_pool", lambda **kw: None)
    # nvidia probe fails (e.g. busy/unsupported) -> rocm result still wins.
    monkeypatch.setattr(sched, "_nvidia_free_mib", lambda: None)
    monkeypatch.setattr(sched, "_rocm_free_mib", lambda: 12345)
    assert sched.gpu_free_mib() == 12345
    monkeypatch.setattr(sched, "_rocm_free_mib", lambda: None)
    assert sched.gpu_free_mib() is None


# ------------------------------------------- deep-dive fixes: cancel + slots

def test_cancel_inprocess_kind_never_signals_own_pid(registry__test_scheduler, stub, monkeypatch):
    """In-process jobs record OUR pid; cancelling must not SIGTERM ourselves."""

    sched = make_sched(registry__test_scheduler, stub)
    j = sched.submit("curate")
    registry__test_scheduler.mark_running(j.job_id, os.getpid())

    import toaster_loop.training.scheduler as sched_mod

    sent = []
    monkeypatch.setattr(sched_mod.os, "kill",
                        lambda pid, sig: sent.append((pid, sig)))
    out = sched.cancel(j.job_id)
    assert out is not None and out.state == "cancelled"
    assert sent == [], f"must not signal any pid, sent {sent}"


def test_cancel_self_pid_guard_even_for_gpu_kinds(registry__test_scheduler, stub, monkeypatch):
    """Defense in depth: a job whose pid equals ours is never signalled."""
    sched = make_sched(registry__test_scheduler, stub)
    j = sched.submit("sft")
    registry__test_scheduler.mark_running(j.job_id, os.getpid())

    import toaster_loop.training.scheduler as sched_mod

    sent = []
    monkeypatch.setattr(sched_mod.os, "kill",
                        lambda pid, sig: sent.append((pid, sig)))
    out = sched.cancel(j.job_id)
    assert out is not None and out.state == "cancelled"
    assert sent == []


def test_run_inprocess_aborts_when_cancelled_while_queued(registry__test_scheduler, stub):
    """A cancel landing before the runner starts must keep 'cancelled'."""
    ran = []
    sched = Scheduler(registry=registry__test_scheduler, cfg=TrainingConfig(),
                      worker_command=[sys.executable, stub],
                      inprocess_runners={"curate": lambda job: ran.append(job.job_id)})
    j = sched.submit("curate")
    sched.cancel(j.job_id)
    # Direct dispatch of the cancelled job (bypassing run_pending's queue read):
    # mark_running must refuse and the runner must never execute.
    sched._run_inprocess(registry__test_scheduler.get_job(j.job_id), slot=0, wait_gate=False)
    assert ran == []
    assert registry__test_scheduler.get_job(j.job_id).state == "cancelled"


def test_slot_lock_excludes_other_process(registry__test_scheduler, stub):
    """A flock held elsewhere (simulating another toaster_loop process's worker on
    device 0) makes the claim fall through / refuse instead of double-booking."""
    import fcntl

    sched = make_sched(registry__test_scheduler, stub)
    lock_path = Path(str(registry__test_scheduler.db_path) + ".slot0.lock")
    holder = open(lock_path, "a+")
    fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        assert sched._claim_device_slot("job-x") is None  # only slot 0 exists
    finally:
        holder.close()
    # after release the same claim succeeds
    claim = sched._claim_device_slot("job-x")
    assert claim is not None and claim[0] == 0
    sched._release_device_slot("job-x", claim[1])


def test_training_config_warns_on_unknown_keys(caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="toaster_loop.training"):
        TrainingConfig.from_yaml({"training": {"min_free_vram_mb": 8192,
                                               "eval_seed": 7}})
    assert any("min_free_vram_mb" in r.message for r in caplog.records)


# ------------------------------------------------------ worker timeout (gap #4)

def test_worker_timeout_default_is_disabled():
    """The default is 0 = NO timeout: multi-hour LoRA runs legitimately
    exceed any fixed cap and jobs 38/39 were SIGTERM'd mid-training by the
    old 4 h default. A positive value in YAML still arms the timeout path."""
    assert TrainingConfig().worker_timeout_s == 0
    assert TrainingConfig.from_yaml(
        {"training": {"worker_timeout_s": 7200}}).worker_timeout_s == 7200


def test_worker_timeout_kills_hung_worker_and_releases_slot(registry__test_scheduler, stub,
                                                            monkeypatch):
    """A worker sleeping past worker_timeout_s gets SIGTERM'd (then SIGKILL
    after the grace), the job ends 'error' with a timeout reason, and the
    device slot is claimable again immediately."""
    monkeypatch.setenv("STUB_MODE", "sleep")
    monkeypatch.setenv("STUB_SLEEP", "60")
    sched = Scheduler(registry=registry__test_scheduler,
                      cfg=TrainingConfig(worker_timeout_s=0.5),
                      worker_command=[sys.executable, stub])
    j = sched.submit("sft", log_dir=None)
    done = sched.run_job(j)          # returns after terminate+reap (~<10 s)
    assert done.state == "error"
    assert "timeout" in (done.error or "")
    # worker really died: the recorded pid no longer exists
    worker_pid = registry__test_scheduler.get_job(j.job_id).pid
    if worker_pid:
        with pytest.raises(ProcessLookupError):
            os.kill(worker_pid, 0)
    # and the device slot flock was released by run_job's finally-block
    claim = sched._claim_device_slot("next-job")
    assert claim is not None
    sched._release_device_slot("next-job", claim[1])


def test_worker_timeout_zero_disables_wait_timeout(registry__test_scheduler, stub, monkeypatch):
    """Explicit 0 keeps the escape hatch: proc.wait() is called with
    timeout=None (no cap). Stubbed Popen — nothing is spawned."""
    calls = {}

    class FakeProc:
        pid = 424242

        def wait(self, timeout=None):
            calls["timeout"] = timeout
            return 0

    monkeypatch.setattr("toaster_loop.training.scheduler.subprocess.Popen",
                        lambda *a, **k: FakeProc())
    sched = Scheduler(registry=registry__test_scheduler,
                      cfg=TrainingConfig(worker_timeout_s=0.0),
                      worker_command=[sys.executable, stub])
    j = sched.submit("sft", log_dir=None)
    done = sched.run_job(j)
    assert calls["timeout"] is None
    assert done.state == "done"


def test_worker_timeout_nonzero_passed_to_wait(registry__test_scheduler, stub, monkeypatch):
    """Non-zero config reaches proc.wait(timeout=...) verbatim."""
    calls = {}

    class FakeProc:
        pid = 424243

        def wait(self, timeout=None):
            calls["timeout"] = timeout
            return 0

    monkeypatch.setattr("toaster_loop.training.scheduler.subprocess.Popen",
                        lambda *a, **k: FakeProc())
    sched = Scheduler(registry=registry__test_scheduler,
                      cfg=TrainingConfig(worker_timeout_s=123.0),
                      worker_command=[sys.executable, stub])
    j = sched.submit("sft", log_dir=None)
    done = sched.run_job(j)
    assert calls["timeout"] == 123.0
    assert done.state == "done"

# ------------------------------------------------------------------------
# from tests/test_serving_router.py
# ------------------------------------------------------------------------

# ------------------------------------------------------------------ helpers

def _mk_registry_model(home: Path, gguf: str, *, validated: bool = False,
                       shadow_ok: bool = False) -> str:
    db = home / "buffer" / "toaster_loop.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    reg = ModelRegistry(db)
    try:
        row = reg.register_model(base_model="base")
        reg.set_artifacts(row.model_id, gguf_path=gguf)
        if validated:
            reg.set_validation(row.model_id, {"ok": True, "pass_rate": 1.0},
                               passed=True)
        if shadow_ok:
            reg.set_shadow(row.model_id, {"is_regression": False})
        return row.model_id
    finally:
        reg.close()


def _mk_running_instance(home: Path, iid: str, *, gguf: str | None = None,
                         base_url: str, role: str = "custom",
                         alive: bool = True) -> None:
    inst = {"id": iid, "name": iid, "gguf": gguf, "attach_url": None,
            "port": 0, "role": role, "context_length": None,
            "run_settings": {}, "sampling": {}, "autostart": False}
    srv_inst.save_instance(home, inst)
    st = {"mode": "spawn", "pid": 999_999_999 if not alive else None,
          "base_url": base_url, "model_name": f"{iid}-model"}
    if alive:
        import os
        st["pid"] = os.getpid()  # ours => state_alive True without a server
    srv_inst.write_state(home, iid, st)


def _two_tier_pool(home: Path):
    """small.gguf -> T0 (no evidence), big.gguf -> T2 (validated+shadow)."""
    _mk_registry_model(home, "/models/small.gguf")
    _mk_registry_model(home, "/models/big.gguf", validated=True, shadow_ok=True)
    _mk_running_instance(home, "small", gguf="/models/small.gguf",
                         base_url="http://127.0.0.1:4001/v1")
    _mk_running_instance(home, "big", gguf="/models/big.gguf",
                         base_url="http://127.0.0.1:4002/v1")


class _StubResult:
    def __init__(self, completed=True, failure_reason=None, response="ok"):
        self.completed = completed
        self.failure_reason = failure_reason
        self.final_response = response


class _StubRunner:
    """Scripted runner: each queued behavior is either a result or an exc."""

    def __init__(self, behaviors):
        self._behaviors = list(behaviors)

    def run(self, task, history=None, previous_summary=None):
        b = self._behaviors.pop(0) if self._behaviors else _StubResult()
        if isinstance(b, Exception):
            raise b
        return b


# -------------------------------------------------------------------- config

def test_routing_config_defaults_off():
    rc = R.RoutingConfig()
    assert rc.enabled is False
    assert rc.escalate_on_failure is True
    assert rc.prefer == "lowest_tier"
    assert rc.refusal_markers == []


def test_routing_config_parses_section_and_ignores_unknown(caplog):
    rc = R.RoutingConfig.from_yaml({"routing": {
        "enabled": True, "prefer": "highest_tier", "bogus_key": 1}})
    assert rc.enabled is True and rc.prefer == "highest_tier"
    assert "bogus_key" in caplog.text  # warned, not crashed


def test_load_config_routing_section_no_warning(tmp_path, caplog):
    p = tmp_path / "toaster_loop.yaml"
    p.write_text(
        "routing:\n"
        "  enabled: true\n"
        "  escalate_on_failure: false\n"
        "  prefer: lowest_tier\n"
        "  high_context_tokens: 4096\n"
        "  tool_heavy_min_tools: 8\n"
        "  profile_hints: {coding: high}\n"
        "  refusal_markers: ['i cannot']\n",
        encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        cfg = load_config(str(p))
    assert not [r for r in caplog.records if "unknown key" in r.getMessage()]
    assert cfg.routing["enabled"] is True
    assert R.routing_config(cfg).escalate_on_failure is False
    assert R.routing_config(cfg).profile_hints == {"coding": "high"}


def test_bad_prefer_falls_back_with_warning(caplog):
    rc = R.RoutingConfig.from_yaml({"routing": {"prefer": "sideways"}})
    assert rc.prefer == "lowest_tier"
    assert "sideways" in caplog.text


# -------------------------------------------------------------- pool + tiers

def test_pool_joins_registry_tiers(tmp_path):
    _two_tier_pool(tmp_path)
    pool = R.build_pool(tmp_path)
    by_id = {c.instance_id: c for c in pool}
    assert set(by_id) == {"small", "big"}
    assert by_id["small"].tier == 0 and by_id["small"].tier_source == "registry"
    assert by_id["big"].tier == 2
    assert by_id["big"].base_url == "http://127.0.0.1:4002/v1"


def test_pool_skips_dead_and_embedding(tmp_path):
    _mk_running_instance(tmp_path, "dead", gguf=None,
                         base_url="http://127.0.0.1:1/v1", alive=False)
    _mk_running_instance(tmp_path, "embed", gguf=None,
                         base_url="http://127.0.0.1:2/v1", role="embedding")
    assert R.build_pool(tmp_path) == []


def test_untiered_fallback_when_no_registry_row(tmp_path):
    _mk_running_instance(tmp_path, "mystery", gguf="/models/none.gguf",
                         base_url="http://127.0.0.1:5/v1")
    pool = R.build_pool(tmp_path)
    assert len(pool) == 1
    assert pool[0].tier == R.UNTIERED_TIER
    assert pool[0].tier_source == "untiered"


# ------------------------------------------------------------ classification

def test_classify_rule_table():
    rc = R.RoutingConfig()
    # nothing fired -> tier-agnostic passthrough
    assert R.classify_request(rc, prompt="hi") is None
    # explicit hints are direct answers
    assert R.classify_request(rc, prompt="x" * 200_000,
                              routing_hint="low") == 0
    assert R.classify_request(rc, prompt="hi", routing_hint="high") == 3
    assert R.classify_request(rc, prompt="hi", routing_hint="auto") is None
    # long context -> >= 2
    long_prompt = "x" * (4 * rc.high_context_tokens + 8)
    assert R.classify_request(rc, prompt=long_prompt) == 2
    # tool-heavy profile -> >= 1
    tools = ["a", "b", "c", "d", "e", "f"]
    assert R.classify_request(rc, prompt="hi", enabled_tool_names=tools) == 1
    # both fire -> max wins
    assert R.classify_request(rc, prompt=long_prompt,
                              enabled_tool_names=tools) == 2
    # per-profile hint only applies when no explicit hint
    rc2 = R.RoutingConfig(profile_hints={"coding": "high"})
    assert R.classify_request(rc2, prompt="hi", profile_name="coding") == 3
    assert R.classify_request(rc2, prompt="hi", profile_name="other") is None
    assert R.classify_request(rc2, prompt="hi", profile_name="coding",
                              routing_hint="low") == 0


# ----------------------------------------------------------------- selection

def _cand(iid: str, tier: int) -> R.Candidate:
    return R.Candidate(instance_id=iid, tier=tier,
                       base_url=f"http://127.0.0.1/{iid}", model_name=iid)


def test_select_lowest_tier_meeting_requirement():
    pool = [_cand("t2", 2), _cand("t0", 0), _cand("t1", 1)]
    assert R.select_target(pool, None).target.instance_id == "t0"
    assert R.select_target(pool, 1).target.instance_id == "t1"
    assert R.select_target(pool, 2).target.instance_id == "t2"
    d = R.select_target(pool, 3)  # nothing meets it -> highest + honest note
    assert d.target.instance_id == "t2"
    assert d.note is not None and "highest available" in d.note
    hi = R.select_target(pool, None, prefer="highest_tier")
    assert hi.target.instance_id == "t2"


def test_select_empty_pool_error_names_tiers():
    with pytest.raises(R.RoutingError) as e:
        R.select_target([], 2)
    assert "tier >= 2" in str(e.value)
    with pytest.raises(R.RoutingError):
        R.select_target([], None)


def test_select_skips_circuit_open_instances():
    pool = [_cand("small", 0), _cand("big", 2)]
    from toaster_loop.core.circuit import get_breaker

    br = get_breaker(pool[0].base_url)
    for _ in range(br.failure_threshold):
        br.record_failure()
    assert br.state == "open"
    d = R.select_target(pool, None)  # small is skipped even though cheaper
    assert d.target.instance_id == "big"
    with pytest.raises(R.RoutingError, match="circuit-open"):
        R.select_target([pool[0]], None)


# ---------------------------------------------------------------- escalation

def _routed_pool():
    return [_cand("small", 0), _cand("mid", 1), _cand("big", 2)]


def test_run_routed_escalates_on_backend_error_result():
    pool = _routed_pool()
    router = R.AutoRouter(R.RoutingConfig(), pool)
    made = []

    def make(cand):
        made.append(cand.instance_id)
        if cand.instance_id == "small":
            return _StubRunner([_StubResult(
                completed=False, failure_reason="backend_error:connection")])
        return _StubRunner([_StubResult(response="big answered")])

    out = R.run_routed(router, make, "do a thing")
    assert made == ["small", "mid"]  # next-higher tier above the failed one
    assert out.attempts == [("small", 0), ("mid", 1)]
    assert out.result.final_response == "big answered"
    assert out.target.instance_id == "mid"
    # breaker recorded exactly one failure on the failed backend only
    from toaster_loop.core.circuit import get_breaker

    assert get_breaker("http://127.0.0.1/small").failures == 1
    assert get_breaker("http://127.0.0.1/mid").failures == 0


def test_run_routed_escalates_on_backend_error_exception():
    from toaster_loop.core.agent_runner import AgentBackendError

    pool = _routed_pool()
    router = R.AutoRouter(R.RoutingConfig(), pool)

    def make(cand):
        if cand.instance_id == "small":
            # context-overflow 400 surfaces as AgentBackendError in the runner
            return _StubRunner([AgentBackendError(
                "Chat completion failed: HTTP 400 — context_size_exceeded")])
        return _StubRunner([_StubResult()])

    out = R.run_routed(router, make, "long task")
    assert [a[0] for a in out.attempts] == ["small", "mid"]
    assert out.result.completed is True


def test_run_routed_no_escalation_on_model_behavior_failure():
    pool = _routed_pool()
    router = R.AutoRouter(R.RoutingConfig(), pool)

    def make(cand):
        return _StubRunner([_StubResult(
            completed=False, failure_reason="max_iterations")])

    out = R.run_routed(router, make, "task")
    assert [a[0] for a in out.attempts] == ["small"]  # bigger model repeats it
    assert out.result.failure_reason == "max_iterations"


def test_run_routed_escalation_disabled_keeps_first_failure():
    pool = _routed_pool()
    router = R.AutoRouter(R.RoutingConfig(escalate_on_failure=False), pool)

    def make(cand):
        return _StubRunner([_StubResult(
            completed=False, failure_reason="backend_error:connection")])

    out = R.run_routed(router, make, "task")
    assert [a[0] for a in out.attempts] == ["small"]
    assert "escalation exhausted" in (out.note or "")


def test_run_routed_refusal_markers_opt_in():
    pool = _routed_pool()
    # off by default: refusal-shaped answer is returned untouched
    router = R.AutoRouter(R.RoutingConfig(), pool)
    out = R.run_routed(router, lambda c: _StubRunner(
        [_StubResult(response="Sorry, I cannot do that.")]), "task")
    assert [a[0] for a in out.attempts] == ["small"]

    router2 = R.AutoRouter(R.RoutingConfig(refusal_markers=["i cannot"]), pool)
    out2 = R.run_routed(router2, lambda c: _StubRunner(
        [_StubResult(response="Sorry, I cannot do that.")
         if c.instance_id == "small" else _StubResult(response="done")]),
        "task")
    assert [a[0] for a in out2.attempts] == ["small", "mid"]
    # refusal is NOT an infra fault: no breaker failure recorded
    from toaster_loop.core.circuit import get_breaker

    assert get_breaker("http://127.0.0.1/small").failures == 0


def test_escalation_skips_circuit_open_higher_tier():
    pool = _routed_pool()
    from toaster_loop.core.circuit import get_breaker

    br = get_breaker("http://127.0.0.1/mid")
    for _ in range(br.failure_threshold):
        br.record_failure()
    router = R.AutoRouter(R.RoutingConfig(), pool)

    def make(cand):
        if cand.instance_id == "big":
            return _StubRunner([_StubResult()])
        return _StubRunner([_StubResult(
            completed=False, failure_reason="backend_error:connection")])

    out = R.run_routed(router, make, "task", history=[{"role": "user",
                                                       "content": "x" * 40000}])
    # requirement 2 -> big directly; small/mid never tried
    assert [a[0] for a in out.attempts] == ["big"]


def test_run_routed_single_instance_unchanged():
    """Single-instance pool: routed run hits exactly that instance once —
    behavior identical to an unrouted call."""
    pool = [_cand("only", 1)]
    router = R.AutoRouter(R.RoutingConfig(), pool)
    seen = []

    def make(cand):
        seen.append((cand.instance_id, cand.base_url))
        return _StubRunner([_StubResult(response="solo")])

    out = R.run_routed(router, make, "task")
    assert seen == [("only", "http://127.0.0.1/only")]
    assert out.attempts == [("only", 1)]
    assert out.result.final_response == "solo"


# ------------------------------------------------------- dashboard seam (opt-in)

def _dashboard_app(tmp_path, routing: dict):
    from toaster_loop.dashboard.app import DashboardApp

    return DashboardApp(ToasterLoopConfig(routing=routing), home=tmp_path)


def test_seam_default_off_never_touches_router(tmp_path, monkeypatch):
    # Legacy serve_state resolution must be used untouched when disabled.
    (tmp_path / "serve_state.json").write_text(
        '{"mode": "spawn", "pid": %d, "base_url": "http://127.0.0.1:7/v1",'
        ' "model_name": "legacy"}' % __import__("os").getpid(),
        encoding="utf-8")

    def _boom(*a, **k):
        raise AssertionError("router must not run when disabled")

    monkeypatch.setattr(R, "build_pool", _boom)
    app = _dashboard_app(tmp_path, {})  # routing absent => off
    base, key, model = app._resolve_chat_target(None)
    assert base == "http://127.0.0.1:7/v1" and model == "legacy"


def test_seam_routes_to_lowest_tier_when_enabled(tmp_path):
    _two_tier_pool(tmp_path)
    app = _dashboard_app(tmp_path, {"enabled": True})
    base, _key, model = app._resolve_chat_target(None)
    assert base == "http://127.0.0.1:4001/v1"  # small tier wins by default
    assert model == "small-model"

    base_hi, _, model_hi = app._resolve_chat_target(None, routing_hint="high")
    assert base_hi == "http://127.0.0.1:4002/v1"  # hint escalates selection
    assert model_hi == "big-model"


def test_seam_explicit_binding_bypasses_routing(tmp_path):
    _two_tier_pool(tmp_path)
    app = _dashboard_app(tmp_path, {"enabled": True})
    # Explicit instance choice wins even though routing prefers small.
    base, _key, model = app._resolve_chat_target("big")
    assert base == "http://127.0.0.1:4002/v1"


def test_seam_routing_error_falls_back_to_legacy_503(tmp_path):
    from toaster_loop.dashboard.app import HttpError

    app = _dashboard_app(tmp_path, {"enabled": True})
    with pytest.raises(HttpError) as e:  # honest legacy error, no crash
        app._resolve_chat_target(None)
    assert e.value.status == 503

# ------------------------------------------------------------------------
# from tests/test_serving_phase6.py
# ------------------------------------------------------------------------

# ------------------------------------------------------------------ fixtures

STUB = '''#!/usr/bin/env python3
"""Stub llama-server: parses toaster_loop's flags, records argv, serves /v1/models."""
import argparse, json, os, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

p = argparse.ArgumentParser()
p.add_argument("-m", dest="model", required=True)
p.add_argument("--alias", default=None)
p.add_argument("--host", default=None)
p.add_argument("--port", type=int, required=True)
p.add_argument("-c", dest="ctx", type=int, default=None)
p.add_argument("-ngl", dest="ngl", type=int, default=None)
p.add_argument("--embedding", action="store_true")
a, _extra = p.parse_known_args()

if a.host != "127.0.0.1":
    print("stub: refusing non-loopback bind", file=sys.stderr); sys.exit(4)

with open(os.environ.get("STUB_PIDFILE", os.devnull), "w") as f:
    f.write(str(os.getpid()))
argvf = os.environ.get("STUB_ARGVFILE")
if argvf:
    with open(argvf, "w") as f:
        json.dump({"embedding": bool(a.embedding), "alias": a.alias,
                   "ctx": a.ctx}, f)

class H(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002 - silence access log
        pass
    def do_GET(self):
        if self.path.rstrip("/") == "/v1/models":
            body = json.dumps({"object": "list", "data": [
                {"id": a.alias or a.model, "object": "model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body)
        else:
            self.send_response(404); self.end_headers()

srv = ThreadingHTTPServer((a.host, a.port), H)
print("stub llama-server ready", flush=True)
srv.serve_forever()
'''


@pytest.fixture()
def home__test_serving_phase6(tmp_path, monkeypatch):
    h = tmp_path / "uf-p6"
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(h))
    return h


@pytest.fixture()
def stub_binary(tmp_path, monkeypatch):
    p = tmp_path / "llama-server"
    p.write_text(STUB)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("STUB_PIDFILE", str(tmp_path / "stub.pid"))
    return str(p)


@pytest.fixture()
def ggufs(tmp_path):
    a = tmp_path / "model-a-q4_k_m.gguf"
    a.write_bytes(b"GGUF")
    b = tmp_path / "model-b-q4_k_m.gguf"
    b.write_bytes(b"GGUF")
    return str(a), str(b)


def make_app(home, **cfg_kwargs) -> DashboardApp:
    return DashboardApp(ToasterLoopConfig(llama_auto_setup=False, **cfg_kwargs),
                        home=home)


def create_server(app, name, gguf=None, attach_url=None, **extra):
    body = {"name": name}
    if gguf:
        body["gguf"] = gguf
    if attach_url:
        body["attach_url"] = attach_url
    body.update(extra)
    return app.handle("POST", "/api/servers", body=body)


def row_of(app, server_id):
    rows = app.handle("GET", "/api/servers").body["servers"]
    return next((r for r in rows if r["id"] == server_id), None)


def must_row(app, server_id):
    """row_of() that fails loudly when the row is missing."""
    row = row_of(app, server_id)
    assert row is not None, f"server row {server_id!r} missing"
    return row


@pytest.fixture()
def cleanup_servers():
    """Kill any instance children a test left behind (own registry only)."""
    yield
    stop_all_servers()


# ------------------------------------------------------- 1. coexistence

def test_two_instances_coexist_and_stop_independently(home__test_serving_phase6, stub_binary,
                                                      ggufs, cleanup_servers):
    app = make_app(home__test_serving_phase6, llama_server_binary=stub_binary)
    a_gguf, b_gguf = ggufs
    ra = create_server(app, "alpha", gguf=a_gguf)
    rb = create_server(app, "beta", gguf=b_gguf)
    assert ra.status == 201 and rb.status == 201
    ida, idb = ra.body["server"]["id"], rb.body["server"]["id"]

    sa = app.handle("POST", f"/api/servers/{ida}/start", body={})
    sb = app.handle("POST", f"/api/servers/{idb}/start", body={})
    assert sa.status == 202 and sb.status == 202, (sa.body, sb.body)
    url_a, url_b = sa.body["base_url"], sb.body["base_url"]
    assert url_a != url_b

    # both healthy, independently
    for url in (url_a, url_b):
        r = httpx.get(url + "/models", timeout=5)
        assert r.status_code == 200
    assert must_row(app, ida)["running"] is True
    assert must_row(app, idb)["running"] is True

    # stop one; the other stays healthy
    st = app.handle("POST", f"/api/servers/{ida}/stop", body={})
    assert st.status == 200 and st.body["stopped"] is True
    assert must_row(app, ida)["running"] is False
    assert must_row(app, idb)["running"] is True
    r = httpx.get(url_b + "/models", timeout=5)
    assert r.status_code == 200

    app.handle("POST", f"/api/servers/{idb}/stop", body={})
    assert must_row(app, idb)["running"] is False


def test_embedding_role_instance_spawns_with_flag(home__test_serving_phase6, stub_binary, tmp_path,
                                                  monkeypatch,
                                                  cleanup_servers):
    """An embedding-role instance passes --embedding (role tag honored)."""
    argvf = tmp_path / "argv.json"
    monkeypatch.setenv("STUB_ARGVFILE", str(argvf))
    app = make_app(home__test_serving_phase6, llama_server_binary=stub_binary)
    g = tmp_path / "embed.gguf"
    g.write_bytes(b"GGUF")
    r = create_server(app, "embed1", gguf=str(g), role="embedding")
    sid = r.body["server"]["id"]
    start = app.handle("POST", f"/api/servers/{sid}/start", body={})
    assert start.status == 202
    meta = json.loads(argvf.read_text())
    assert meta["embedding"] is True
    app.handle("POST", f"/api/servers/{sid}/stop", body={})


# ------------------------------------------------------ 2. CRUD + validation

def test_crud_and_validation(home__test_serving_phase6, ggufs):
    app = make_app(home__test_serving_phase6)
    a_gguf, _ = ggufs

    r = create_server(app, "solo", gguf=a_gguf, port=45_712)
    assert r.status == 201
    sid = r.body["server"]["id"]

    # duplicate name (case-insensitive) rejected
    dup = create_server(app, "SOLO", gguf=a_gguf)
    assert dup.status == 409 and dup.body["error"]["code"] == "duplicate_name"

    # port conflict with the existing instance rejected
    conflict = create_server(app, "other", gguf=a_gguf, port=45_712)
    assert conflict.status == 409 \
        and conflict.body["error"]["code"] == "port_conflict"

    # neither gguf nor attach => 400
    bad = create_server(app, "bad")
    assert bad.status == 400

    # both => 400 (ambiguous target)
    both = app.handle("POST", "/api/servers", body={
        "name": "both", "gguf": a_gguf,
        "attach_url": "http://127.0.0.1:9/v1"})
    assert both.status == 400

    # update: rename + change port away from conflict is fine
    upd = app.handle("PUT", f"/api/servers/{sid}",
                     body={"name": "solo2", "gguf": a_gguf, "port": 0})
    assert upd.status == 200 and upd.body["server"]["name"] == "solo2"

    # delete removes the definition file
    d = app.handle("DELETE", f"/api/servers/{sid}", body={})
    assert d.status == 200
    assert row_of(app, sid) is None


def test_attach_url_must_be_loopback(home__test_serving_phase6):
    app = make_app(home__test_serving_phase6)
    r = create_server(app, "remote", attach_url="http://10.0.0.5:8080/v1")
    assert r.status == 400 and "loopback" in r.body["error"]["message"]
    with pytest.raises(ServeError):
        require_loopback("http://example.com/v1")
    assert require_loopback("http://localhost:1234/v1")


def test_attach_instance_never_spawns_or_kills(home__test_serving_phase6):
    app = make_app(home__test_serving_phase6)
    with FakeOpenAIServer() as fs:
        r = create_server(app, "attached", attach_url=fs.base_url)
        assert r.status == 201
        sid = r.body["server"]["id"]

        start = app.handle("POST", f"/api/servers/{sid}/start", body={})
        assert start.status == 202
        st = srv_inst.read_state(home__test_serving_phase6, sid)
        assert st is not None
        assert st["mode"] == "attach" and not st.get("pid")  # spawned nothing

        row = must_row(app, sid)
        assert row["running"] is True and row["mode"] == "attach"

        stop = app.handle("POST", f"/api/servers/{sid}/stop", body={})
        assert stop.status == 200 and stop.body["stopped"] is False
        # the foreign endpoint survives — detach only forgets it
        assert httpx.get(fs.base_url + "/models", timeout=5).status_code == 200


# ------------------------------------------------------- 3. backward compat

def test_default_instance_mirrors_legacy_config(home__test_serving_phase6, ggufs):
    a_gguf, _ = ggufs
    app = make_app(home__test_serving_phase6, gguf_path=a_gguf)
    rows = app.handle("GET", "/api/servers").body["servers"]
    defaults = [r for r in rows if r["id"] == "default"]
    assert len(defaults) == 1 and defaults[0]["default"] is True
    assert defaults[0]["gguf"] == a_gguf

    # default cannot be deleted
    d = app.handle("DELETE", "/api/servers/default", body={})
    assert d.status == 400


def test_legacy_attach_shows_in_default_row(home__test_serving_phase6):
    from toaster_loop.serving.deploy import write_serve_state

    app = make_app(home__test_serving_phase6)
    with FakeOpenAIServer() as fs:
        write_serve_state(home__test_serving_phase6, {"mode": "attach", "base_url": fs.base_url,
                                 "model_name": "fake-model",
                                 "attach_url": fs.base_url.rsplit("/v1", 1)[0]})
        row = must_row(app, "default")
        assert row["running"] is True and row["mode"] == "attach"

        # the new stop endpoint stops/forgets the legacy default too
        r = app.handle("POST", "/api/servers/default/stop", body={})
        assert r.status == 200 and r.body["stopped"] is False
        assert srv_inst.read_state(home__test_serving_phase6, "default") is None


def test_new_start_writes_legacy_serve_state(home__test_serving_phase6, stub_binary, ggufs,
                                             cleanup_servers):
    """Starting the default instance via /api/servers keeps serve_state.json
    in sync so old panels/endpoints keep working."""
    from toaster_loop.serving.deploy import read_serve_state

    app = make_app(home__test_serving_phase6, llama_server_binary=stub_binary)
    # give the reserved default a gguf through config (legacy key)
    app.cfg.gguf_path = ggufs[0]
    r = app.handle("POST", "/api/servers/default/start", body={})
    assert r.status == 202, r.body
    st = read_serve_state(home__test_serving_phase6)
    assert st and st["base_url"] == r.body["base_url"]
    legacy_status = app.handle("GET", "/api/serve/status").body
    assert legacy_status["running"] is True

    app.handle("POST", "/api/servers/default/stop", body={})
    assert app.handle("GET", "/api/serve/status").body["running"] is False


# ------------------------------------------------------- 4. profile binding

def _stub_runner_capture(monkeypatch, calls):
    """create_runner stub recording (base_url, model) per construction."""
    from toaster_loop.core.agent_runner import AgentResult

    class StubRunner:
        def __init__(self, cfg, base_url, model, **kw):
            calls.append({"base_url": base_url, "model": model, **{
                k: kw.get(k) for k in ("sampling_override",)}})
            self.kw = kw

        def run(self, prompt, history=None, previous_summary=None):
            return AgentResult(final_response="ok",
                               messages=[{"role": "assistant",
                                          "content": "ok"}])

    monkeypatch.setattr("toaster_loop.core.agent_runner.create_runner", StubRunner)


def test_profile_server_instance_binding(home__test_serving_phase6, monkeypatch):
    app = make_app(home__test_serving_phase6)
    calls: list = []
    _stub_runner_capture(monkeypatch, calls)

    with FakeOpenAIServer() as default_fs, FakeOpenAIServer() as b_fs:
        # legacy default serve target
        from toaster_loop.serving.deploy import write_serve_state
        write_serve_state(home__test_serving_phase6, {"mode": "attach",
                                 "base_url": default_fs.base_url,
                                 "model_name": "default-model"})
        # named instance B (attach) running
        r = create_server(app, "b", attach_url=b_fs.base_url,
                          sampling={"temperature": 0.1})
        sid = r.body["server"]["id"]
        assert app.handle("POST", f"/api/servers/{sid}/start",
                          body={}).status == 202

        # profile bound to B
        pr = app.handle("PUT", "/api/profiles/coder", body={
            "system_prompt": "code", "tools": [],
            "server_instance": sid})
        assert pr.status == 200
        assert pr.body["profile"]["server_instance"] == sid

        rr = app.handle("POST", "/api/chat/stream",
                        body={"prompt": "hi", "profile": "coder"})
        assert rr.status == 200 and rr.stream is not None
        "".join(rr.stream)
        assert calls[-1]["base_url"].rstrip("/") == b_fs.base_url.rstrip("/")
        # per-instance sampling override reached create_runner
        assert calls[-1]["sampling_override"] == {"temperature": 0.1}

        # unset profile => default target
        pr2 = app.handle("PUT", "/api/profiles/plain", body={
            "system_prompt": "", "tools": []})
        assert "server_instance" not in pr2.body["profile"]
        rr = app.handle("POST", "/api/chat/stream",
                        body={"prompt": "hi", "profile": "plain"})
        assert rr.stream is not None
        "".join(rr.stream)
        assert calls[-1]["base_url"].rstrip("/") == \
            default_fs.base_url.rstrip("/")
        assert calls[-1]["sampling_override"] is None


def test_profile_binding_errors(home__test_serving_phase6):
    app = make_app(home__test_serving_phase6)
    # unknown instance => 404 (resolved before the stream starts)
    pr = app.handle("PUT", "/api/profiles/ghost", body={
        "system_prompt": "", "tools": [], "server_instance": "nope"})
    assert pr.status == 200
    r = app.handle("POST", "/api/chat/stream",
                   body={"prompt": "hi", "profile": "ghost"})
    assert r.status == 404 and r.body["error"]["code"] == "unknown_server"

    # known but stopped => 503 naming the instance
    g = home__test_serving_phase6 / "x.gguf"
    g.write_bytes(b"GGUF")
    cr = create_server(app, "stopped-one", gguf=str(g))
    sid = cr.body["server"]["id"]
    app.handle("PUT", "/api/profiles/usesStopped", body={
        "system_prompt": "", "tools": [], "server_instance": sid})
    r = app.handle("POST", "/api/chat/stream",
                   body={"prompt": "hi", "profile": "usesStopped"})
    assert r.status == 503 and r.body["error"]["code"] == "not_serving"


# --------------------------------------------------------- 5. VRAM advisory

def test_vram_warning_and_force_bypass(home__test_serving_phase6, stub_binary, ggufs, tmp_path,
                                       monkeypatch, cleanup_servers):
    app = make_app(home__test_serving_phase6, llama_server_binary=stub_binary)
    big = tmp_path / "big.gguf"
    with open(big, "wb") as f:  # sparse 4 GiB => estimate way over "free"
        f.truncate(4 * 1024 ** 3)

    monkeypatch.setattr(srv_inst, "free_vram_mib", lambda: 512)

    r = create_server(app, "biggy", gguf=str(big))
    sid = r.body["server"]["id"]
    start = app.handle("POST", f"/api/servers/{sid}/start", body={})
    assert start.status == 202
    assert start.body["warning"] and "VRAM" in start.body["warning"]
    # warn, don't block: the server actually started
    assert must_row(app, sid)["running"] is True
    app.handle("POST", f"/api/servers/{sid}/stop", body={})

    # force bypasses the estimate entirely (no warning field)
    start2 = app.handle("POST", f"/api/servers/{sid}/start",
                        body={"force": True})
    assert start2.status == 202 and start2.body["warning"] is None
    app.handle("POST", f"/api/servers/{sid}/stop", body={})


def test_vram_advisory_skipped_without_telemetry(home__test_serving_phase6, ggufs, monkeypatch):
    monkeypatch.setattr(srv_inst, "free_vram_mib", lambda: None)
    est = srv_inst.vram_advisory({"gguf": ggufs[0]})
    assert est is None


# ----------------------------------------------------------- 6. shutdown

def test_stop_all_servers_only_touches_owned(home__test_serving_phase6, stub_binary, ggufs,
                                             monkeypatch):
    foreign = subprocess.Popen([sys.executable, "-c",
                                "import time; time.sleep(30)"])
    app = make_app(home__test_serving_phase6, llama_server_binary=stub_binary)
    try:
        r = create_server(app, "owned", gguf=ggufs[0])
        sid = r.body["server"]["id"]
        start = app.handle("POST", f"/api/servers/{sid}/start", body={})
        assert start.status == 202
        child_pid = int((home__test_serving_phase6.parent / "stub.pid").read_text())
        os.kill(child_pid, 0)  # alive

        stop_all_servers()

        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            try:
                os.kill(child_pid, 0)
                time.sleep(0.1)
            except OSError:
                break
        else:
            pytest.fail("owned instance child still running after shutdown")
        assert foreign.poll() is None  # never kill what we didn't spawn
    finally:
        foreign.kill()
        stop_all_servers()


def test_ports_in_use_sees_other_instances(home__test_serving_phase6, ggufs):
    app = make_app(home__test_serving_phase6)
    create_server(app, "p1", gguf=ggufs[0], port=45_999)
    used = srv_inst.ports_in_use(home__test_serving_phase6, cfg=app.cfg)
    assert 45_999 in used
    assert srv_inst.check_port_conflict(home__test_serving_phase6, app.cfg, 45_999) is True
    assert srv_inst.check_port_conflict(home__test_serving_phase6, app.cfg, 46_000) is False
    # excluding its own id frees the port for updates
    sid = must_row(app, "p1")["id"]
    assert srv_inst.check_port_conflict(home__test_serving_phase6, app.cfg, 45_999,
                                        exclude_id=sid) is False


# ------------- reserved default instance via the canonical start endpoint
# (the legacy /api/serve/start endpoint was removed; these paths now run
# through POST /api/servers/default/start with the config-derived default)

def test_default_start_registers_owned_handle_and_state(home__test_serving_phase6, stub_binary,
                                                        ggufs, cleanup_servers):
    """Starting the reserved default spawns through start_instance_server: the
    child lands in the _owned_handles registry (shutdown coverage),
    serve_state.json keeps its legacy shape for old readers, and
    /api/serve/stop reaches the child."""
    from toaster_loop.serving.supervisor import owned_instance_ids

    app = make_app(home__test_serving_phase6, llama_server_binary=stub_binary,
                   gguf_path=ggufs[0])
    r = app.handle("POST", "/api/servers/default/start", body={})
    assert r.status == 202, r.body
    assert r.body["ok"] is True and r.body["base_url"].startswith("http://")

    # registry visibility: the reserved default id is an owned instance now
    assert "default" in owned_instance_ids()

    # legacy state file consumers (shadow eval, status panels): compatible shape
    st = json.loads((home__test_serving_phase6 / "serve_state.json").read_text())
    assert st["mode"] == "spawn" and st["pid"] and st["base_url"]
    assert st["instance_id"] == "default"
    assert int((home__test_serving_phase6.parent / "stub.pid").read_text()) == st["pid"]
    # per-instance state mirrors it (Phase-6 view)
    mirror = srv_inst.read_state(home__test_serving_phase6, "default")
    assert mirror and mirror["pid"] == st["pid"]

    # healthy endpoint + status surface agree
    assert httpx.get(r.body["base_url"] + "/models", timeout=5).status_code == 200
    status = app.handle("GET", "/api/serve/status").body
    assert status["running"] is True and status["pid"] == st["pid"]

    # stop path reaches the in-process handle: child dies, registry empties
    pid = st["pid"]
    sr = app.handle("POST", "/api/serve/stop", body={})
    assert sr.status == 200 and sr.body["stopped"] is True
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
            time.sleep(0.1)
        except OSError:
            break
    else:
        pytest.fail("default-server child survived /api/serve/stop")
    assert "default" not in owned_instance_ids()
    assert not (home__test_serving_phase6 / "serve_state.json").exists()


def test_default_start_crash_loop_refusal_surfaces(home__test_serving_phase6, stub_binary, ggufs,
                                                   cleanup_servers):
    """Over-budget starts are refused THROUGH the endpoint (500), while an
    explicit user start clears the budget and succeeds (#28 semantics)."""
    from toaster_loop.serving import supervisor as sup

    app = make_app(home__test_serving_phase6, llama_server_binary=stub_binary,
                   gguf_path=ggufs[0])
    for _ in range(sup.CRASH_LOOP_MAX_FAILURES):
        sup.record_start_failure("default")

    # autostart=True keeps the refusal in force: start_instance_server refuses
    # before ever spawning.
    r = app.handle("POST", "/api/servers/default/start", body={"autostart": True})
    assert r.status == 500 and "crash loop" in r.body["error"]["message"]

    # explicit user start clears the crash-loop state and spawns normally
    r2 = app.handle("POST", "/api/servers/default/start", body={})
    assert r2.status == 202, r2.body
    assert sup.crash_loop_status("default") is None  # healthy start reset it
    app.handle("POST", "/api/serve/stop", body={})


def test_default_start_log_location(home__test_serving_phase6, stub_binary, ggufs, cleanup_servers):
    """Spawn log stays at <home>/logs/server-default.log (user-visible path)."""
    app = make_app(home__test_serving_phase6, llama_server_binary=stub_binary,
                   gguf_path=ggufs[0])
    r = app.handle("POST", "/api/servers/default/start", body={})
    assert r.status == 202
    logp = home__test_serving_phase6 / "logs" / "server-default.log"
    assert logp.is_file() and b"ready" in logp.read_bytes()
    app.handle("POST", "/api/serve/stop", body={})


def test_merged_state_accessors_fold_legacy_default(home__test_serving_phase6):
    """The compatibility seam: read/write/clear_state_merged handle the
    reserved default's two state views (per-instance file + legacy
    serve_state.json) in one call; non-default ids are plain passthrough."""
    h = home__test_serving_phase6
    # No state anywhere -> merged read is None.
    assert srv_inst.read_state_merged(h, "default") is None

    # Legacy-only write (an old toaster_loop / CLI path): merged read folds it in.
    from toaster_loop.serving.deploy import write_serve_state
    write_serve_state(h, {"mode": "attach", "base_url": "http://127.0.0.1:9/v1"})
    st = srv_inst.read_state_merged(h, "default")
    assert st and st["base_url"] == "http://127.0.0.1:9/v1"

    # Per-instance file wins once present (fresher record is authoritative).
    srv_inst.write_state(h, "default", {"mode": "spawn", "pid": os.getpid(),
                                        "base_url": "http://127.0.0.1:8/v1"})
    st = srv_inst.read_state_merged(h, "default")
    assert st["mode"] == "spawn"

    # Merged write dual-writes: both views land and agree.
    srv_inst.write_state_merged(h, "default", {"mode": "spawn", "pid": 4242})
    assert srv_inst.read_state(h, "default")["pid"] == 4242
    from toaster_loop.serving.deploy import read_serve_state
    assert read_serve_state(h)["pid"] == 4242

    # Merged clear removes BOTH views.
    srv_inst.clear_state_merged(h, "default")
    assert srv_inst.read_state(h, "default") is None
    assert read_serve_state(h) is None

    # Non-default ids never touch the legacy file.
    srv_inst.write_state_merged(h, "other", {"pid": 1})
    assert read_serve_state(h) is None
    assert srv_inst.read_state_merged(h, "other") == {"pid": 1}


def test_migrate_legacy_state_one_time_copies_to_mirror(home__test_serving_phase6):
    """Boot migration: legacy serve_state.json -> serving/state/default.json.
    Idempotent; never creates a definition record; leaves the legacy file."""
    h = home__test_serving_phase6
    # Nothing to migrate -> False, no files created.
    assert srv_inst.migrate_legacy_state(h) is False

    from toaster_loop.serving.deploy import write_serve_state
    write_serve_state(h, {"mode": "spawn", "pid": 77, "base_url": "http://x/v1"})
    assert srv_inst.migrate_legacy_state(h) is True
    mirror = srv_inst.read_state(h, "default")
    assert mirror and mirror["pid"] == 77
    # Legacy file untouched (still dual-written by the seam; old readers live).
    from toaster_loop.serving.deploy import read_serve_state
    assert read_serve_state(h) is not None
    # Definition record deliberately NOT created (stays config-derived).
    assert not srv_inst.instance_path(h, "default").exists()
    # Idempotent: second run does nothing.
    assert srv_inst.migrate_legacy_state(h) is False


def test_state_files_with_api_key_are_owner_only(home__test_serving_phase6):
    """serve_state.json and serving/state/<id>.json can embed attach_api_key;
    both writers must land the file 0600 (default umask would leave them
    world-readable)."""
    from toaster_loop.serving.deploy import write_serve_state

    write_serve_state(home__test_serving_phase6, {"mode": "attach", "base_url": "http://127.0.0.1:1",
                             "attach_api_key": "sk-secret"})
    mode = (home__test_serving_phase6 / "serve_state.json").stat().st_mode & 0o777
    assert mode == 0o600, f"serve_state.json mode {mode:o}, want 600"

    srv_inst.write_state(home__test_serving_phase6, "inst-keyed",
                         {"pid": os.getpid(), "attach_api_key": "sk-secret"})
    p = srv_inst.instance_state_path(home__test_serving_phase6, "inst-keyed")
    mode = p.stat().st_mode & 0o777
    assert mode == 0o600, f"instance state mode {mode:o}, want 600"

# ------------------------------------------------------------------------
# from tests/test_bench.py
# ------------------------------------------------------------------------

# ------------------------------------------------------------- fake servers

class _BenchHandler(BaseHTTPRequestHandler):
    """Deterministic chat-completions server: replies are a pure function of
    the prompt, so repeated runs produce identical token streams. Optionally
    emits llama.cpp-style ``timings`` when TIMINGS is set on the class."""

    REPLY = "the answer"
    # e.g. {"prompt_per_second": 100.0, "predicted_per_second": 20.0}
    TIMINGS: dict | None = None
    requests: list = []

    def log_message(self, format, *args):  # silence
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") == "/v1/models":
            self._send({"object": "list",
                        "data": [{"id": "bench-model", "object": "model"}]})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        if not self.path.startswith("/v1/chat/completions"):
            self._send({"error": "not found"}, 404)
            return
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        type(self).requests.append(payload)
        last_user = ""
        for m in reversed(payload.get("messages", [])):
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                last_user = m["content"]
                break
        data = {
            "id": "cmpl-bench",
            "object": "chat.completion",
            "model": payload.get("model", "bench-model"),
            "choices": [{"index": 0,
                         "message": {"role": "assistant",
                                     "content": self.REPLY + "|" + last_user[:8]},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 7,
                      "total_tokens": 19},
        }
        timings = type(self).TIMINGS
        if timings is not None:
            data["timings"] = dict(timings)
        self._send(data)


class _DriftHandler(_BenchHandler):
    """Nondeterministic server: reply changes on every request, so repeated
    temperature-0 runs produce DIVERGING token streams."""

    counter = 0

    def do_POST(self):
        type(self).counter += 1
        self.REPLY = f"drift-{type(self).counter}"
        super().do_POST()


@pytest.fixture()
def bench_home(tmp_path, monkeypatch):
    d = tmp_path / "benchhome"
    monkeypatch.setenv("TOASTER_LOOP_BENCH_DIR", str(d))
    return d


# ---------------------------------------------------------------- statistics

def test_compute_stats_known_values():
    s = bench.compute_stats([10.0, 20.0, 30.0])
    assert s["median"] == 20.0
    assert s["min"] == 10.0
    assert s["max"] == 30.0
    # spread = (30-10)/20 * 100 = 100%
    assert s["spread_pct"] == pytest.approx(100.0)


def test_compute_stats_even_count_median():
    s = bench.compute_stats([2.0, 4.0, 6.0, 8.0])
    assert s["median"] == 5.0
    assert s["spread_pct"] == pytest.approx((8.0 - 2.0) / 5.0 * 100.0)


def test_compute_stats_zero_median_spread_undefined():
    s = bench.compute_stats([0.0, 0.0])
    assert s["median"] == 0.0
    assert s["spread_pct"] is None


def test_compute_stats_empty_raises():
    with pytest.raises(ValueError):
        bench.compute_stats([])


# ------------------------------------------------------------- measurement

def test_run_bench_repeat_runs_and_identity_ok():
    from fake_server import FakeOpenAIServer

    _BenchHandler.requests = []
    with FakeOpenAIServer(handler_cls=_BenchHandler) as srv:
        rec = bench.run_bench(srv.base_url, "bench-model",
                              prompts=["alpha", "beta"], runs=3)
    assert rec["schema_version"] == bench.SCHEMA_VERSION
    assert rec["kind"] == bench.KIND_BENCH
    assert len(rec["per_run"]) == 3
    assert all(r["completion_tokens"] == 14 for r in rec["per_run"])
    gen = [r["gen_toks_per_s"] for r in rec["per_run"]]
    assert all(g and g > 0 for g in gen)
    # stats block consistent with per-run values
    assert rec["generation"]["min"] == min(gen)
    assert rec["generation"]["max"] == max(gen)
    # deterministic server => identical streams across runs
    assert rec["token_identity"]["status"] == "ok"
    assert rec["token_identity"]["divergent_runs"] == []
    # temperature 0 is what we actually sent
    assert all(r["temperature"] == 0 for r in _BenchHandler.requests)


def test_run_bench_uses_server_timings_when_present():
    from fake_server import FakeOpenAIServer

    _BenchHandler.TIMINGS = {"prompt_per_second": 100.0,
                             "predicted_per_second": 20.0}
    try:
        with FakeOpenAIServer(handler_cls=_BenchHandler) as srv:
            rec = bench.run_bench(srv.base_url, "bench-model",
                                  prompts=["one"], runs=2)
    finally:
        _BenchHandler.TIMINGS = None
    assert rec["timings_source"] == "server"
    assert rec["generation"]["median"] == pytest.approx(20.0)
    assert rec["prompt_processing"]["median"] == pytest.approx(100.0)


def test_run_bench_token_identity_drift_detected():
    from fake_server import FakeOpenAIServer

    _DriftHandler.counter = 0
    with FakeOpenAIServer(handler_cls=_DriftHandler) as srv:
        rec = bench.run_bench(srv.base_url, "bench-model",
                              prompts=["alpha"], runs=3)
    ti = rec["token_identity"]
    assert ti["status"] == "drift"
    # runs 2 and 3 differ from run 1's stream
    assert ti["divergent_runs"] == [2, 3]


def test_run_bench_connection_failure_clear_message():
    with pytest.raises(bench.BenchError, match="llama-server running"):
        bench.run_bench("http://127.0.0.1:9/v1", "m", prompts=["x"], runs=1)


# ------------------------------------------------------------- persistence

def test_bench_dir_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("TOASTER_LOOP_BENCH_DIR", str(tmp_path / "custom"))
    assert bench.bench_dir() == tmp_path / "custom"


def test_bench_dir_under_home():
    from toaster_loop.store.paths import resolve_toaster_loop_home

    assert bench.bench_dir() == resolve_toaster_loop_home() / "bench"


def test_save_load_roundtrip(bench_home):
    rec = {"schema_version": bench.SCHEMA_VERSION, "kind": bench.KIND_BENCH,
           "timestamp": "2026-10-11T12:00:00Z", "label": "exp-a",
           "generation": {"median": 42.0}}
    p = bench.save_result(rec)
    assert p.is_file()
    assert str(bench_home) in str(p)
    loaded = bench.load_results()
    assert len(loaded) == 1
    assert loaded[0]["label"] == "exp-a"
    assert loaded[0]["generation"]["median"] == 42.0


def test_save_result_filename_keyed_timestamp_label(bench_home):
    rec = {"kind": bench.KIND_NOISE_FLOOR,
           "timestamp": "2026-10-11T12:34:56Z", "label": "floor one"}
    p = bench.save_result(rec)
    assert p.name == "noise_floor-20261011T123456Z-floor-one.json"


def test_load_results_filters_and_skips_corrupt(bench_home):
    bench_home.mkdir(parents=True)
    (bench_home / "a.json").write_text(json.dumps(
        {"kind": bench.KIND_BENCH, "label": "x",
         "timestamp": "2026-10-10T00:00:00Z"}))
    (bench_home / "b.json").write_text(json.dumps(
        {"kind": bench.KIND_NOISE_FLOOR, "label": "floor",
         "timestamp": "2026-10-11T00:00:00Z"}))
    (bench_home / "junk.json").write_text("{not json")
    assert len(bench.load_results()) == 2
    assert [r["label"] for r in bench.load_results(
        kind=bench.KIND_NOISE_FLOOR)] == ["floor"]
    assert [r["label"] for r in bench.load_results(label="x")] == ["x"]


def test_noise_floor_roundtrip_and_newest_wins(bench_home):
    old = {"kind": bench.KIND_NOISE_FLOOR, "label": "",
           "timestamp": "2026-10-01T00:00:00Z",
           "generation": {"median": 10.0, "spread_pct": 5.0}}
    new = dict(old, timestamp="2026-10-11T00:00:00Z",
               generation={"median": 12.0, "spread_pct": 3.0})
    bench.save_result(old)
    bench.save_result(new)
    nf = bench.load_noise_floor()
    assert nf is not None
    assert nf["timestamp"] == "2026-10-11T00:00:00Z"
    assert nf["generation"]["spread_pct"] == 3.0


# ------------------------------------------------------- identity rule / cmp

def _mk_record(sha, gen_median, spread=4.0):
    return {
        "kind": bench.KIND_BENCH,
        "config": {"prompts_sha256": sha},
        "token_identity": {"status": "ok", "streams_sha256": sha + "-tok"},
        "generation": {"median": gen_median, "spread_pct": spread},
        "prompt_processing": {"median": gen_median * 4, "spread_pct": spread},
    }


def test_compare_valid_speedup_and_noise_floor_flag():
    a = _mk_record("sha1", 100.0, spread=4.0)
    b = _mk_record("sha1", 110.0, spread=4.0)  # +10% > 4% floor
    c = _mk_record("sha1", 102.0, spread=4.0)  # +2% < 4% floor
    cmp_ok = bench.compare_records(a, b)
    assert cmp_ok["valid"] is True
    assert cmp_ok["verdict"] == "VALID"
    assert cmp_ok["generation"]["speedup"] == pytest.approx(0.10)
    assert cmp_ok["generation"]["within_noise_floor"] is False

    cmp_noise = bench.compare_records(a, c)
    assert cmp_noise["valid"] is True
    assert cmp_noise["generation"]["within_noise_floor"] is True


def test_compare_invalid_when_tokens_differ():
    a = _mk_record("sha1", 100.0)
    b = _mk_record("sha1", 150.0)
    b["token_identity"]["streams_sha256"] = "different-tokens"
    cmp_res = bench.compare_records(a, b)
    assert cmp_res["valid"] is False
    assert cmp_res["verdict"] == "INVALID"
    assert any("INVALID" in r for r in cmp_res["reasons"])


def test_compare_invalid_on_different_prompt_sets():
    a = _mk_record("sha1", 100.0)
    b = _mk_record("sha2", 150.0)
    b["token_identity"]["streams_sha256"] = "sha2-tok"
    cmp_res = bench.compare_records(a, b)
    assert cmp_res["valid"] is False
    assert any("prompts" in r for r in cmp_res["reasons"])


def test_compare_invalid_on_within_run_drift():
    a = _mk_record("sha1", 100.0)
    b = _mk_record("sha1", 150.0)
    b["token_identity"]["status"] = "drift"
    cmp_res = bench.compare_records(a, b)
    assert cmp_res["valid"] is False


# ------------------------------------------------------------------ prompts

def test_load_prompts_file(tmp_path):
    f = tmp_path / "p.txt"
    f.write_text("# comment\n\none prompt\ntwo prompt\n")
    assert bench.load_prompts_file(f) == ["one prompt", "two prompt"]
    empty = tmp_path / "e.txt"
    empty.write_text("# only comments\n")
    with pytest.raises(bench.BenchError):
        bench.load_prompts_file(empty)


def test_prompts_sha256_stable_and_sensitive():
    assert bench.prompts_sha256(["a", "b"]) == bench.prompts_sha256(["a", "b"])
    assert bench.prompts_sha256(["a", "b"]) != bench.prompts_sha256(["b", "a"])


# -------------------------------------------------------------- CLI wiring

def test_cli_bench_help_and_registration():
    from toaster_loop import cli

    parser = cli.build_parser()
    ns = parser.parse_args(["bench", "--runs", "3", "--noise-floor"])
    assert ns.command == "bench"
    assert ns.runs == 3
    assert ns.noise_floor is True


def test_cli_bench_attach_end_to_end(bench_home, capsys):
    from fake_server import FakeOpenAIServer

    _BenchHandler.requests = []
    with FakeOpenAIServer(handler_cls=_BenchHandler) as srv:
        rc = cli_main(["bench", "--attach", srv.base_url, "--runs", "2",
                       "--label", "cli-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "toaster_loop bench" in out
    assert "median=" in out
    assert "spread=" in out
    assert "token identity: OK" in out
    # persisted under the overridden dir
    files = list(bench_home.glob("bench-*-cli-run.json"))
    assert len(files) == 1


def test_cli_bench_noise_floor_mode_and_list(bench_home, capsys):
    from fake_server import FakeOpenAIServer

    with FakeOpenAIServer(handler_cls=_BenchHandler) as srv:
        assert cli_main(["bench", "--attach", srv.base_url, "--runs", "2",
                         "--noise-floor", "--label", "floor"]) == 0
    capsys.readouterr()
    assert list(bench_home.glob("noise_floor-*-floor.json"))
    assert cli_main(["bench", "--list"]) == 0
    out = capsys.readouterr().out
    assert "noise_floor" in out and "floor" in out


def test_cli_bench_compare_valid_and_invalid(bench_home, capsys):
    from fake_server import FakeOpenAIServer

    # two identical runs -> valid comparison, tokens unchanged
    with FakeOpenAIServer(handler_cls=_BenchHandler) as srv:
        assert cli_main(["bench", "--attach", srv.base_url, "--runs", "2",
                         "--label", "base"]) == 0
        assert cli_main(["bench", "--attach", srv.base_url, "--runs", "2",
                         "--label", "cand"]) == 0
    capsys.readouterr()
    assert cli_main(["bench", "--compare", "base", "cand"]) == 0
    out = capsys.readouterr().out
    assert "VALID" in out and "INVALID" not in out


def test_cli_bench_compare_missing_record_exits_2(bench_home, capsys):
    from toaster_loop import cli

    rc = cli.main(["bench", "--compare", "nope-a", "nope-b"])
    assert rc == 2
    assert "no persisted bench record" in capsys.readouterr().err


def test_cli_bench_dead_endpoint_clear_error(capsys):
    # No attach, no gguf configured -> clear endpoint error, no spawn attempts.
    from toaster_loop import cli

    rc = cli.main(["bench", "--attach", "http://127.0.0.1:9/v1"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "error" in err

# ------------------------------------------------------------------------
# from tests/test_remote_peers.py
# ------------------------------------------------------------------------

@pytest.fixture()
def home__test_remote_peers(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(h))
    return h


@pytest.fixture()
def app__test_remote_peers(home__test_remote_peers):
    return DashboardApp(ToasterLoopConfig(), home=home__test_remote_peers)


def _ok_snapshot(url="http://peer:8765"):
    return {"ok": True, "probed_at": "2026-01-01T00:00:00+00:00",
            "version": "0.9.9", "peer_home": "/home/other/.toaster-loop",
            "servers": [{"name": "default", "role": "custom",
                         "running": True, "mode": "spawn",
                         "model_name": "remote-model"}],
            "agents": [{"name": "coder", "default": False}],
            "default_profile": None,
            "jobs": [{"job_id": "j1", "kind": "train", "state": "running",
                      "model_id": "m", "created_at": "t"}],
            "traces": [{"trace_id": "t1", "profile": "coder",
                        "status": "success", "created_at": "t",
                        "model_name": "remote-model"}],
            "server": {"running": True}}


def _fail_snapshot(err="ConnectError: boom"):
    return {"ok": False, "probed_at": "2026-01-02T00:00:00+00:00",
            "error": err}


# ------------------------------------------------------------ url handling

def test_normalize_url():
    assert peers_mod.normalize_url("192.168.1.5:8765") == \
        "http://192.168.1.5:8765"
    assert peers_mod.normalize_url("https://box.local:9/") == \
        "https://box.local:9"
    assert peers_mod.normalize_url("http://h/prefix/") == "http://h/prefix"
    with pytest.raises(ValueError):
        peers_mod.normalize_url("")
    with pytest.raises(ValueError):
        peers_mod.normalize_url("ftp://nope")
    with pytest.raises(ValueError):
        peers_mod.normalize_url("http:///noserver")


# ------------------------------------------------------------ persistence

def test_upsert_remove_roundtrip(home__test_remote_peers):
    assert peers_mod.load_peers(home__test_remote_peers) == []
    rec = peers_mod.upsert_peer(home__test_remote_peers, "http://a:1", "tokA", name="alpha")
    assert rec["id"] and rec["name"] == "alpha"
    # re-add same URL updates instead of duplicating; empty token keeps old
    rec2 = peers_mod.upsert_peer(home__test_remote_peers, "http://a:1", None)
    assert rec2["id"] == rec["id"]
    assert len(peers_mod.load_peers(home__test_remote_peers)) == 1
    stored = peers_mod.get_peer(home__test_remote_peers, rec["id"])
    assert stored is not None and stored["token"] == "tokA"
    assert peers_mod.remove_peer(home__test_remote_peers, rec["id"]) is True
    assert peers_mod.remove_peer(home__test_remote_peers, rec["id"]) is False
    assert peers_mod.load_peers(home__test_remote_peers) == []


def test_peer_view_never_leaks_token(home__test_remote_peers):
    rec = peers_mod.upsert_peer(home__test_remote_peers, "http://a:1", "secret-token")
    view = peers_mod.peer_view(rec)
    assert view["has_token"] is True
    assert "secret-token" not in str(view)


# ------------------------------------------------------------------- API

def test_probe_remembers_then_lists(app__test_remote_peers, home__test_remote_peers, monkeypatch):
    calls = []

    def fake_probe(url, token):
        calls.append((url, token))
        return _ok_snapshot()

    monkeypatch.setattr(peers_mod, "probe", fake_probe)
    r = app__test_remote_peers.handle("POST", "/api/remote/probe",
                   body={"url": "http://peer:8765", "token": "tk"})
    assert r.status == 200, r.body
    assert calls == [("http://peer:8765", "tk")]
    pid = r.body["peer"]["id"]

    lst = app__test_remote_peers.handle("GET", "/api/remote/peers").body
    peers = {p["id"]: p for p in lst["peers"]}
    assert pid in peers
    p = peers[pid]
    assert p["healthy"] is True and p["last_ok_at"]
    assert p["snapshot"]["agents"][0]["name"] == "coder"
    assert p["snapshot"]["jobs"][0]["state"] == "running"
    assert p["snapshot"]["traces"][0]["trace_id"] == "t1"
    assert "tk" not in str(lst)  # token never round-trips to the UI


def test_failed_probe_remembers_nothing(app__test_remote_peers, monkeypatch):
    monkeypatch.setattr(peers_mod, "probe",
                        lambda url, token: _fail_snapshot())
    r = app__test_remote_peers.handle("POST", "/api/remote/probe", body={"url": "http://x:1"})
    assert r.status == 400
    assert r.body["error"]["code"] == "peer_unreachable"
    assert app__test_remote_peers.handle("GET", "/api/remote/peers").body["peers"] == []


def test_bad_url_is_clean_400(app__test_remote_peers):
    r = app__test_remote_peers.handle("POST", "/api/remote/probe", body={"url": "ftp://x"})
    assert r.status == 400 and r.body["error"]["code"] == "bad_request"


def test_live_reprobe_keeps_last_ok_on_failure(app__test_remote_peers, home__test_remote_peers, monkeypatch):
    monkeypatch.setattr(peers_mod, "probe",
                        lambda url, token: _ok_snapshot())
    pid = app__test_remote_peers.handle("POST", "/api/remote/probe",
                     body={"url": "http://peer:8765"}).body["peer"]["id"]
    stored = peers_mod.get_peer(home__test_remote_peers, pid)
    assert stored is not None and stored["last_ok_at"]

    monkeypatch.setattr(peers_mod, "probe",
                        lambda url, token: _fail_snapshot("down"))
    r = app__test_remote_peers.handle("GET", f"/api/remote/peers/{pid}/live")
    assert r.status == 200
    p = r.body["peer"]
    assert p["healthy"] is False
    assert p["snapshot"]["error"] == "down"
    # last good timestamp survives the failed re-probe (failure honesty)
    assert p["last_ok_at"]

    assert app__test_remote_peers.handle("GET", "/api/remote/peers/nope123abcd/live").status == 404


def test_delete_peer(app__test_remote_peers, monkeypatch):
    monkeypatch.setattr(peers_mod, "probe",
                        lambda url, token: _ok_snapshot())
    pid = app__test_remote_peers.handle("POST", "/api/remote/probe",
                     body={"url": "http://peer:8765"}).body["peer"]["id"]
    assert app__test_remote_peers.handle("DELETE", f"/api/remote/peers/{pid}").status == 200
    assert app__test_remote_peers.handle("DELETE", f"/api/remote/peers/{pid}").status == 404


def test_probe_snapshot_shape(home__test_remote_peers, monkeypatch):
    """The real probe() against a fake transport: shape + auth header."""
    import httpx

    seen_headers = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.append(request.headers.get("X-Toaster-Loop-Token"))
        path = request.url.path
        payloads = {
            "/api/status": {"version": "1.2.3", "home": "/h",
                            "server": {"running": True}},
            "/api/servers": {"servers": [
                {"name": "default", "running": True, "mode": "spawn",
                 "model_name": "m", "role": "custom"}]},
            "/api/profiles": {"profiles": [{"name": "a"}],
                              "default_profile": "a"},
            "/api/jobs": {"jobs": []},
            "/api/traces": {"traces": []},
        }
        if path in payloads:
            return httpx.Response(200, json=payloads[path])
        return httpx.Response(404, json={"error": "nope"})

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    def patched_client(**kw):
        kw["transport"] = transport
        return real_client(**kw)

    monkeypatch.setattr(httpx, "Client", patched_client)
    snap = peers_mod.probe("http://peer:8765", "tok123")
    assert snap["ok"] is True
    assert snap["version"] == "1.2.3"
    assert snap["servers"][0]["model_name"] == "m"
    assert snap["default_profile"] == "a"
    assert all(h == "tok123" for h in seen_headers) and seen_headers

    bad = peers_mod.probe("http://peer:8765", None)  # headers now None-tuple
    # MockTransport persists; second call without token still ok
    assert bad["ok"] is True

# ------------------------------------------------------------------------
# from tests/test_shadow.py
# ------------------------------------------------------------------------

@pytest.fixture()
def home__test_shadow(tmp_path, monkeypatch):
    h = tmp_path / "uf-shadow"
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(h))
    (h / "buffer").mkdir(parents=True)
    return h


@pytest.fixture()
def store__test_shadow(home__test_shadow):
    s = TraceStore(home__test_shadow / "buffer" / "toaster_loop.db")
    yield s
    s.close()


# ------------------------------------------------------------------ seeding



def seed_trace(store, home, tid, *, ts, profile="coder", status="success",
               score: Optional[float] = 0.9, failure_reason=None, user_fb=None,
               task=None, reply="the old answer"):
    p = home / "buffer" / f"{tid}.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        f.write(json.dumps({"kind": "trace_header"}) + "\n")
        f.write(json.dumps({"kind": "message", "message": {
            "role": "system", "content": "sp"}}) + "\n")
        f.write(json.dumps({"kind": "message", "message": {
            "role": "user", "content": task or f"task {tid}"}}) + "\n")
        f.write(json.dumps({"kind": "message", "message": {
            "role": "assistant", "content": reply}}) + "\n")
    store.insert_trace(TraceRow(
        trace_id=tid, session_id=None, timestamp=ts, agent_profile=profile,
        model_alias="m", model_version=None, endpoint_label=None,
        task_summary="s", status=status, outcome_score=score,
        num_tool_calls=0, tokens_in=None, tokens_out=None, duration_s=None,
        jsonl_path=str(p), failure_reason=failure_reason,
    ))
    if user_fb is not None:
        store.add_feedback(tid, user_fb, None, source="user", timestamp=ts)


# ------------------------------------------------------- replay-set selection

def test_replay_set_picks_positive_recent_deterministically(store__test_shadow, home__test_shadow):
    seed_trace(store__test_shadow, home__test_shadow, "t1", ts="2026-09-01T00:00:00+00:00", score=0.9)
    seed_trace(store__test_shadow, home__test_shadow, "t2", ts="2026-09-02T00:00:00+00:00", score=0.1)
    seed_trace(store__test_shadow, home__test_shadow, "t3", ts="2026-09-03T00:00:00+00:00", score=None,
               user_fb=0.8)  # positive via USER feedback despite NULL heuristic
    seed_trace(store__test_shadow, home__test_shadow, "t4", ts="2026-09-04T00:00:00+00:00",
               profile="other", score=0.9)

    rs = build_replay_set(store__test_shadow, "coder", n=8)
    assert [r["trace_id"] for r in rs] == ["t3", "t1"]  # ts desc; t2 low-score
    assert rs[0]["task"] == "task t3"
    assert rs[0]["reference_response"] == "the old answer"

    # deterministic across calls + n cap
    assert build_replay_set(store__test_shadow, "coder", n=8) == rs
    assert [r["trace_id"] for r in build_replay_set(store__test_shadow, "coder", n=1)] == ["t3"]


def test_replay_set_excludes_infra_and_failures(store__test_shadow, home__test_shadow):
    seed_trace(store__test_shadow, home__test_shadow, "ok", ts="2026-09-05T00:00:00+00:00")
    seed_trace(store__test_shadow, home__test_shadow, "infra", ts="2026-09-06T00:00:00+00:00",
               status="partial", failure_reason="backend_error:connection")
    seed_trace(store__test_shadow, home__test_shadow, "fail", ts="2026-09-07T00:00:00+00:00",
               status="failure")
    rs = build_replay_set(store__test_shadow, "coder", n=8)
    assert [r["trace_id"] for r in rs] == ["ok"]


def test_replay_set_honest_empty(store__test_shadow, home__test_shadow):
    seed_trace(store__test_shadow, home__test_shadow, "t1", ts="2026-09-01T00:00:00+00:00", score=0.05)
    assert build_replay_set(store__test_shadow, "coder", n=8) == []
    assert build_replay_set(store__test_shadow, "nobody", n=8) == []


# ------------------------------------------------------------ compare math

def test_compare_scores_math_and_threshold():
    per = [{"candidate_score": 0.9, "baseline_score": 0.5},
           {"candidate_score": 0.7, "baseline_score": 0.6}]
    c = compare_scores(per)
    assert c["candidate_mean"] == pytest.approx(0.8)
    assert c["baseline_mean"] == pytest.approx(0.55)
    assert c["improvement"] == pytest.approx(0.25)
    assert c["is_regression"] is False
    assert c["regression_threshold"] == pytest.approx(-0.05)

    reg = compare_scores([{"candidate_score": 0.4, "baseline_score": 0.5}])
    assert reg["improvement"] == pytest.approx(-0.1)
    assert reg["is_regression"] is True  # -0.1 < -0.05 (default 5%)

    # knob override: 20% tolerance makes the same delta a non-regression
    loose = compare_scores([{"candidate_score": 0.4, "baseline_score": 0.5}],
                           max_regression_pct=20.0)
    assert loose["is_regression"] is False

    # None scores are skipped, means stay honest (never coerced to zero)
    mixed = compare_scores([{"candidate_score": None, "baseline_score": 0.5},
                            {"candidate_score": 0.3, "baseline_score": None}])
    assert mixed["skipped"] == 2
    assert mixed["improvement"] is None
    assert mixed["is_regression"] is False

    empty = compare_scores([])
    assert empty["n"] == 0 and empty["candidate_mean"] is None


# --------------------------------------------------------------- e2e job

def _stub_judge_client_for(score_map):
    """judge_client_for seam: score by substring of the judged transcript."""
    def client_for(base_url, model_name):
        def client(messages, model):
            text = json.dumps(messages)
            for needle, sc in score_map.items():
                if needle in text:
                    return json.dumps({"score": sc, "reason": "stub"})
            return json.dumps({"score": 0.5, "reason": "stub-mid"})
        return client
    return client_for


def test_run_shadow_eval_end_to_end_stores_comparison(home__test_shadow):
    store__test_shadow = TraceStore(home__test_shadow / "buffer" / "toaster_loop.db")
    reg = ModelRegistry(home__test_shadow / "buffer" / "toaster_loop.db")
    seed_trace(store__test_shadow, home__test_shadow, "t1", ts="2026-09-01T00:00:00+00:00")
    seed_trace(store__test_shadow, home__test_shadow, "t2", ts="2026-09-02T00:00:00+00:00")

    base = reg.register_model("Base/Model")            # v1 -> active
    reg.set_validation(base.model_id, {"ok": True}, passed=True)
    reg.promote(base.model_id)
    cand = reg.register_model("Base/Model")            # v2 candidate

    # two fake servers responding differently per side (spec requirement);
    # the judge stub grades "candidate reply" high and "baseline reply" low.
    cand_srv = FakeOpenAIServer(handler_cls=make_scripted_handler(
        {"*": "candidate reply: excellent grounded answer"})).start()
    base_srv = FakeOpenAIServer(handler_cls=make_scripted_handler(
        {"*": "baseline reply: weak unhelpful mush"})).start()

    @contextmanager
    def endpoint_for(model):
        srv = cand_srv if model.model_id == cand.model_id else base_srv
        yield srv.base_url, model.model_id

    try:
        result = run_shadow_eval(
            reg, store__test_shadow, candidate_id=cand.model_id, profile="coder", n=8,
            home=home__test_shadow, endpoint_for=endpoint_for,
            judge_client_for=_stub_judge_client_for(
                {"excellent grounded": 0.9, "weak unhelpful": 0.2}),
        )
    finally:
        cand_srv.stop()
        base_srv.stop()

    assert result["baseline_id"] == base.model_id
    assert result["n"] == 2
    assert result["candidate_mean"] == pytest.approx(0.9)
    assert result["baseline_mean"] == pytest.approx(0.2)
    assert result["improvement"] == pytest.approx(0.7)
    assert result["is_regression"] is False
    # judge calls bounded: 2 per task => exactly n candidate + n baseline rows
    assert len(result["per_task"]) == 2

    row = reg.get_model(cand.model_id)
    assert row is not None
    stored = row.shadow
    assert stored is not None and stored["baseline_id"] == base.model_id
    assert set(stored) >= {"ts", "baseline_id", "n", "candidate_mean",
                           "baseline_mean", "improvement", "is_regression",
                           "per_task", "judge_model"}

    # verdict storing is ledgered (pass4 #31 actor plumbing)
    ops = [e["op"] for e in reg.history(limit=10)]
    assert "set_shadow" in ops
    store__test_shadow.close()
    reg.close()


def test_run_shadow_eval_regression_flag_and_honest_skips(home__test_shadow):
    store__test_shadow = TraceStore(home__test_shadow / "buffer" / "toaster_loop.db")
    reg = ModelRegistry(home__test_shadow / "buffer" / "toaster_loop.db")
    seed_trace(store__test_shadow, home__test_shadow, "t1", ts="2026-09-01T00:00:00+00:00")
    base = reg.register_model("Base/Model")
    reg.set_validation(base.model_id, {"ok": True}, passed=True)
    reg.promote(base.model_id)
    cand = reg.register_model("Base/Model")

    @contextmanager
    def endpoint_for(model):
        yield "http://127.0.0.1:1/v1", model.model_id

    gen_calls = {"n": 0}

    def generate(base_url, model_name, task):
        gen_calls["n"] += 1
        return "CANDIDATE" if "v2" in model_name else "BASELINE"

    result = run_shadow_eval(
        reg, store__test_shadow, candidate_id=cand.model_id, profile="coder",
        home=home__test_shadow, endpoint_for=endpoint_for, generate_fn=generate,
        judge_client_for=_stub_judge_client_for({"CANDIDATE": 0.2,
                                                 "BASELINE": 0.9}),
    )
    assert result["is_regression"] is True
    # bounded: exactly one generate + one judge per side per task = 2xN
    assert gen_calls["n"] == 2

    # no replay set -> honest skip, nothing stored
    reg2_home = home__test_shadow.parent / "empty-home"
    reg2_home.mkdir()
    (reg2_home / "buffer").mkdir()
    store2 = TraceStore(reg2_home / "buffer" / "toaster_loop.db")
    reg2 = ModelRegistry(reg2_home / "buffer" / "toaster_loop.db")
    m = reg2.register_model("Base/Model")
    skip = run_shadow_eval(reg2, store2, candidate_id=m.model_id,
                           profile="coder", home=reg2_home)
    assert skip == {"skipped": True, "reason": NO_REPLAY_SET}
    row_m = reg2.get_model(m.model_id)
    assert row_m is not None and row_m.shadow is None

    # no baseline (candidate itself active) -> honest skip too
    reg2.set_validation(m.model_id, {"ok": True}, passed=True)
    reg2.promote(m.model_id)
    skip2 = run_shadow_eval(reg2, store2, candidate_id=m.model_id,
                            profile="coder", home=reg2_home,
                            endpoint_for=endpoint_for)
    assert skip2["skipped"] is True
    store__test_shadow.close(); reg.close(); store2.close(); reg2.close()


def test_run_shadow_eval_all_skipped_raises(home__test_shadow):
    """Zero scored tasks must be an error, never a fake 'no regression'."""
    store__test_shadow = TraceStore(home__test_shadow / "buffer" / "toaster_loop.db")
    reg = ModelRegistry(home__test_shadow / "buffer" / "toaster_loop.db")
    seed_trace(store__test_shadow, home__test_shadow, "t1", ts="2026-09-01T00:00:00+00:00")
    base = reg.register_model("Base/Model")
    reg.set_validation(base.model_id, {"ok": True}, passed=True)
    reg.promote(base.model_id)
    cand = reg.register_model("Base/Model")

    @contextmanager
    def endpoint_for(model):  # never reached: generation fails first
        yield "http://127.0.0.1:1/v1", model.model_id

    with pytest.raises(ShadowError, match="zero scored"):
        run_shadow_eval(reg, store__test_shadow, candidate_id=cand.model_id,
                        profile="coder", home=home__test_shadow, endpoint_for=endpoint_for,
                        generate_fn=lambda *a: None)  # every reply fails
    store__test_shadow.close()
    reg.close()


# --------------------------------------------------------- scheduler path

def test_scheduler_claims_shadow_eval_respects_vram_gate(home__test_shadow, monkeypatch):
    reg = ModelRegistry(home__test_shadow / "buffer" / "toaster_loop.db")
    ran = []

    def runner(job):
        ran.append(job.job_id)
        return {"ok": True}

    sched = Scheduler(registry=reg, cfg=TrainingConfig(min_free_vram_mib=1),
                      inprocess_runners={"shadow_eval": runner})
    job = sched.submit("shadow_eval", model_id="v9", enqueue_actor="cli")
    # enqueue itself is ledgered when an actor is passed (#32)
    hist = reg.history(limit=5)
    assert hist[0]["op"] == "job_queued"

    # VRAM gate closed -> job stays queued with the reason recorded
    monkeypatch.setattr("toaster_loop.training.scheduler.gpu_free_mib", lambda: 1)
    sched.cfg.min_free_vram_mib = 100_000
    with pytest.raises(SchedulerError, match="resource gate"):
        sched.run_job(job)
    jrow = reg.get_job(job.job_id)
    assert jrow is not None and jrow.state == "queued"
    assert ran == []

    # gate open -> in-process runner executes, job completes
    monkeypatch.setattr("toaster_loop.training.scheduler.gpu_free_mib", lambda: 10**7)
    done = sched.run_pending()
    assert [j.state for j in done] == ["done"]
    assert ran == [job.job_id]
    reg.close()


def test_scheduler_shadow_without_runner_fails_honestly(home__test_shadow, monkeypatch):
    monkeypatch.setattr("toaster_loop.training.scheduler.gpu_free_mib", lambda: None)
    reg = ModelRegistry(home__test_shadow / "buffer" / "toaster_loop.db")
    sched = Scheduler(registry=reg, cfg=TrainingConfig(),
                      inprocess_runners={})  # no shadow_eval runner
    # also disable the lazy default by pointing the module import at nothing:
    job = sched.submit("shadow_eval", model_id="v9")
    import toaster_loop.evaluation.shadow as sh
    monkeypatch.setattr(sh, "make_shadow_job_runner",
                        lambda *a, **k: (_ for _ in ()).throw(
                            ImportError("no store")))
    done = sched.run_pending()
    assert done[0].state == "error"
    reg.close()


# ------------------------------------------------------------ promotion gate

def test_shadow_promotion_check_semantics():
    assert shadow_promotion_check(None) == {"warned": False,
                                            "blocked": False, "message": None}
    assert shadow_promotion_check({"skipped": True})["warned"] is False
    good = {"is_regression": False, "baseline_id": "v1", "n": 4,
            "improvement": 0.1, "regression_threshold": -0.05}
    assert shadow_promotion_check(good)["message"] is None

    bad = {"is_regression": True, "baseline_id": "v1", "n": 4,
           "improvement": -0.2, "regression_threshold": -0.05}
    warn = shadow_promotion_check(bad)
    assert warn["warned"] is True and warn["blocked"] is False
    assert "v1" in warn["message"]

    strict = shadow_promotion_check(bad, blocks=True)
    assert strict["blocked"] is True


def test_iterate_promote_warns_then_blocks_with_strict_flag(home__test_shadow, tmp_path,
                                                            monkeypatch):
    """Regression verdict: default warn still promotes; strict flag blocks."""
    from toaster_loop.curate import CurationConfig
    from toaster_loop.training.pipeline import run_iterate
    from toaster_loop.training.validate import ValidationConfig

    stub = tmp_path / "stub.py"
    stub.write_text(
        "import argparse, json, sys\n"
        "from pathlib import Path\n"
        "ap = argparse.ArgumentParser()\n"
        "for f in (('--job', True), ('--kind', True)):\n"
        "    ap.add_argument(f[0], required=f[1])\n"
        "for f in ('--config','--dataset-dir','--base-model','--log-path',"
        "'--adapter-dir','--gguf-out'):\n"
        "    ap.add_argument(f, default=None)\n"
        "ap.add_argument('--quant', default='q4_k_m')\n"
        "ap.add_argument('--dry-run', action='store_true')\n"
        "a = ap.parse_args()\n"
        "if a.kind in ('sft','dpo'):\n"
        "    out = Path(a.config).parent\n"
        "    (out/'adapter').mkdir(parents=True, exist_ok=True)\n"
        "    (out/'adapter'/'adapter_config.json').write_text('{}')\n"
        "elif a.kind == 'export':\n"
        "    gg = Path(a.gguf_out); gg.parent.mkdir(parents=True, exist_ok=True)\n"
        "    gg.write_bytes(b'GGUF-stub')\n")
    store__test_shadow = TraceStore(home__test_shadow / "buffer" / "toaster_loop.db")
    reg = ModelRegistry(home__test_shadow / "buffer" / "toaster_loop.db")
    seed_trace(store__test_shadow, home__test_shadow, "t1", ts="2026-09-01T00:00:00+00:00",
               profile="fallback")

    def starter(gguf_path):
        srv = FakeOpenAIServer(handler_cls=make_scripted_handler(
            {"*": "ok-answer"}))
        srv.start()
        return srv

    verdict = {"is_regression": True, "baseline_id": "v0", "n": 3,
               "improvement": -0.2, "regression_threshold": -0.05}
    monkeypatch.setattr(
        "toaster_loop.evaluation.shadow.shadow_promotion_check",
        lambda shadow, *, blocks=False: {
            "warned": not blocks, "blocked": blocks,
            "message": "shadow regressed (test)"})

    def go(strict):
        # The stored-shadow plumbing is covered by the e2e test above; here we
        # drive the promote-stage gate directly: shadow_promotion_check is
        # imported inside the stage, so patching the source module works.
        sched = Scheduler(registry=reg, cfg=TrainingConfig(base_model="B/M"),
                          worker_command=[sys.executable, str(stub)])
        return run_iterate(
            home=home__test_shadow, registry=reg, scheduler=sched, store=store__test_shadow,
            curation_cfg=CurationConfig(min_score=0.5),
            training_cfg=TrainingConfig(base_model="B/M"),
            validation_cfg=ValidationConfig(
                min_pass_rate=1.0,
                golden_tasks=[{"prompt": "q", "expect_contains": ["ok-answer"]}],
                shadow_blocks_promotion=strict),
            name="sh", server_starter=starter)

    r = go(strict=False)
    assert r.ok and r.promoted, r.to_dict()
    assert (r.validation or {}).get("shadow_gate", {}).get("warned") is True

    # strict: next candidate blocked at promote
    seed_trace(store__test_shadow, home__test_shadow, "t2", ts="2026-09-02T00:00:00+00:00",
               profile="fallback")
    r2 = go(strict=True)
    assert not r2.ok and r2.stage == "promote"
    assert "shadow regressed" in (r2.error or "")
    store__test_shadow.close()
    reg.close()


# ---------------------------------------------------------------- migration

def test_shadow_migration_on_old_db_shapes(tmp_path):
    db = tmp_path / "old.db"
    reg = ModelRegistry(db)
    reg.register_model("B/M")
    reg.close()

    # Simulate a pre-#32 DB: drop the shadow column and rebuild jobs with the
    # old kind CHECK (as _migrate_job_kinds-era tables looked).
    conn = sqlite3.connect(str(db))
    conn.execute("ALTER TABLE models DROP COLUMN shadow_json")
    conn.execute("ALTER TABLE jobs RENAME TO jobs_old_x")
    conn.execute(
        "CREATE TABLE jobs (job_id TEXT PRIMARY KEY,"
        " kind TEXT NOT NULL CHECK (kind IN"
        " ('sft','dpo','grpo','export','validate','curate')),"
        " dataset_id TEXT, model_id TEXT, state TEXT NOT NULL CHECK (state IN"
        " ('queued','running','done','error','cancelled')), log_path TEXT,"
        " pid INTEGER, queued_at TEXT NOT NULL, started_at TEXT, finished_at TEXT,"
        " error TEXT, params_json TEXT)")
    # Explicit column list (not SELECT *): the live table may carry newer
    # columns (#43 depends_on_json) than this simulated pre-#32 shape.
    conn.execute(
        "INSERT INTO jobs (job_id, kind, dataset_id, model_id, state,"
        " log_path, pid, queued_at, started_at, finished_at, error,"
        " params_json)"
        " SELECT job_id, kind, dataset_id, model_id, state, log_path, pid,"
        " queued_at, started_at, finished_at, error, params_json"
        " FROM jobs_old_x")
    conn.execute("DROP TABLE jobs_old_x")
    conn.commit()
    conn.close()

    reg2 = ModelRegistry(db)  # runs _migrate_shadow_column + kind rebuild
    try:
        m = reg2.get_model("v1")
        assert m is not None and m.shadow is None  # tolerant read
        job = reg2.enqueue_job("shadow_eval", model_id="v1")  # CHECK accepts
        assert job.state == "queued"
        reg2.set_shadow("v1", {"is_regression": False, "n": 1})
        row = reg2.get_model("v1")
        assert row is not None and row.shadow is not None
        assert row.shadow["n"] == 1
    finally:
        reg2.close()

# ------------------------------------------------------------------------
# from tests/test_portability_p0p2.py
# ------------------------------------------------------------------------

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
    from toaster_loop.training.scheduler import TrainingConfig, resource_gate_ok

    calls = []

    def fake_free():
        calls.append(None)
        return 10_000

    monkeypatch.setattr("toaster_loop.training.scheduler.gpu_free_mib", fake_free)
    assert resource_gate_ok(TrainingConfig(min_free_vram_mib=5000))["ok"]
    assert calls == [None]


def test_resource_gate_passes_device_when_pinned(monkeypatch):
    from toaster_loop.training.scheduler import TrainingConfig, resource_gate_ok

    seen = []

    def fake_free(device=None):
        seen.append(device)
        return 10_000 if device == "1" else 100

    monkeypatch.setattr("toaster_loop.training.scheduler.gpu_free_mib", fake_free)
    # Device 1 has room => admits even though GPU 0 (global min) is full.
    ok = resource_gate_ok(TrainingConfig(min_free_vram_mib=5000), device="1")
    assert ok["ok"] and seen == ["1"]


def test_scheduler_gate_device_none_without_gpu_ids():
    """Unpinned config (the default) => the global-minimum probe, unchanged."""
    from toaster_loop.training.scheduler import Scheduler, TrainingConfig

    sched = Scheduler.__new__(Scheduler)
    sched.cfg = TrainingConfig()
    assert sched._gate_device(0) is None

    sched.cfg = TrainingConfig(gpu_ids=[1, 3])
    assert sched._gate_device(0) == "1"
    assert sched._gate_device(1) == "3"


def test_nvidia_free_mib_per_device(monkeypatch):
    import toaster_loop.training.scheduler as sched

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

    import toaster_loop.training.scheduler as sched

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
    from toaster_loop.core import gpumem as _gm
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
    from toaster_loop.serving.watchdog import (
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
    import toaster_loop.serving.watchdog as wd

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
    from toaster_loop import doctor

    p = tmp_path / "meminfo"
    p.write_text("MemTotal: 1000000 kB\nMemAvailable: 500000 kB\n")
    assert doctor.read_meminfo(str(p)) == 500_000


def test_read_meminfo_darwin(monkeypatch):
    from toaster_loop import doctor

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
    from toaster_loop import doctor

    # This box may itself BE a unified-pool machine; the chain under test
    # starts at the vendor probes.
    from toaster_loop.core import gpumem as _gm
    monkeypatch.setattr(_gm, "unified_pool", lambda **kw: None)
    monkeypatch.setattr(doctor, "free_vram_mib_via_smi", lambda: None)
    monkeypatch.setattr("toaster_loop.training.scheduler._rocm_free_mib",
                        lambda device=None: 8192)
    assert doctor.free_vram_mib() == 8192

    monkeypatch.setattr("toaster_loop.training.scheduler._rocm_free_mib",
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
    monkeypatch.setenv("TOASTER_LOOP_LLAMA_CPP", str(tmp_path / "llama.cpp"))
    dirs = runtime.search_dirs()
    assert tmp_path / "llama.cpp" / "build" / "bin" / "Release" in dirs
    assert any("homebrew" in str(d) for d in dirs)

# ------------------------------------------------------------------------
# from tests/test_doctor_wp7.py
# ------------------------------------------------------------------------

# --------------------------------------------------------------------- helpers

class FakeUsage:
    def __init__(self, free: int, total: int = 100 * 1024 ** 3):
        self.total = total
        self.free = free
        self.used = total - free


def _write_gguf(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.truncate(size)


def _registry_with_model(home: Path, model_id: str, gguf: Path) -> None:
    db = home / "buffer" / "toaster_loop.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE models (model_id TEXT, gguf_path TEXT)")
    conn.execute("INSERT INTO models VALUES (?, ?)", (model_id, str(gguf)))
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------- disk

def test_disk_check_flags_low_space_on_export_paths(tmp_path):
    paths = [tmp_path / "home", tmp_path / "tmp"]
    for p in paths:
        p.mkdir()
    required = 10 * 1024 ** 3  # 10 GiB largest model
    # need = required + max(2 GiB, 10%) = 11 GiB; report 10 GiB free -> FAIL.
    ok_free = 12 * 1024 ** 3

    def usage_fn(p):
        return FakeUsage(free=ok_free if "home" in p else 10 * 1024 ** 3)

    res = doctor.check_disk(paths, required, "m-1", disk_usage_fn=usage_fn)
    assert res.status == "problem"
    joined = "\n".join(res.lines)
    assert "BELOW REQUIREMENT" in joined and "OK" in joined
    assert any("janitor" in c or "free space" in c for c in res.next_commands)


def test_disk_check_ok_when_free_covers_model_plus_margin(tmp_path):
    p = tmp_path / "home"
    p.mkdir()
    required = 10 * 1024 ** 3
    res = doctor.check_disk([p], required, "m-1",
                            disk_usage_fn=lambda _p: FakeUsage(
                                free=required + 2 * 1024 ** 3))
    assert res.status == "ok"


def test_largest_registered_model_reads_registry(tmp_path):
    # No DB yet -> zero.
    assert doctor.largest_registered_model_bytes(tmp_path) == (0, None)
    big = tmp_path / "models" / "big.gguf"
    _write_gguf(big, 3 * 1024 ** 2)
    small = tmp_path / "models" / "small.gguf"
    _write_gguf(small, 1024 ** 2)
    db = tmp_path / "buffer" / "toaster_loop.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE models (model_id TEXT, gguf_path TEXT)")
    conn.execute("INSERT INTO models VALUES ('m-small', ?)", (str(small),))
    conn.execute("INSERT INTO models VALUES ('m-big', ?)", (str(big),))
    conn.commit()
    conn.close()
    size, mid = doctor.largest_registered_model_bytes(tmp_path)
    assert size == 3 * 1024 ** 2
    assert mid == "m-big"


def test_export_relevant_paths_includes_home_and_training_output(
        tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    out = tmp_path / "outputs"
    cfg = SimpleNamespace(training={"output_dir": str(out)})
    # Fake distinct devices so the same-filesystem dedup keeps all three.
    devs = {}

    def fake_stat(path, *a, **kw):
        devs.setdefault(str(path), len(devs) + 7)
        return SimpleNamespace(st_dev=devs[str(path)])

    monkeypatch.setattr(doctor.os, "stat", fake_stat)
    paths = doctor.export_relevant_paths(cfg, home)
    assert home in paths and out in paths


def test_export_relevant_paths_dedupes_same_filesystem(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    out = tmp_path / "outputs"
    cfg = SimpleNamespace(training={"output_dir": str(out)})
    monkeypatch.setattr(doctor.os, "stat",
                        lambda path, *a, **kw: SimpleNamespace(st_dev=42))
    paths = doctor.export_relevant_paths(cfg, home)
    assert paths == [home]  # one filesystem probed once


# -------------------------------------------------------------------- memory

MEMINFO_FIXTURE = """MemTotal:       32760192 kB
MemFree:         1048576 kB
MemAvailable:   16777216 kB
Buffers:          123456 kB
"""


def test_parse_meminfo_available():
    assert doctor.parse_meminfo_available_kib(MEMINFO_FIXTURE) == 16777216
    assert doctor.parse_meminfo_available_kib("MemTotal: 1 kB") is None


def test_ram_preset_math_on_fixed_inputs():
    # 48 GiB RAM -> context tier 16384; 20 GiB VRAM -> ngl 999 cap 16384.
    p = doctor.presets_from_measurements(48 * 1024, 20 * 1024)
    assert p["serve_context"] == 16384
    assert p["serve_ngl"] == 999
    # batch = 49152 // 16384 = 3
    assert p["train_batch_size"] == 3

    # 80 GiB RAM, 30 GiB VRAM -> 32768 ctx (both tiers allow), batch cap 8? no:
    # 81920 // 16384 = 5.
    p = doctor.presets_from_measurements(80 * 1024, 30 * 1024)
    assert p["serve_context"] == 32768
    assert p["train_batch_size"] == 5

    # Tiny box: 4 GiB RAM, no VRAM -> 4096 ctx, CPU offload unknown.
    p = doctor.presets_from_measurements(4 * 1024, None)
    assert p["serve_context"] == 4096
    assert p["serve_ngl"] is None
    assert p["train_batch_size"] == 1

    # VRAM cap binds below the RAM tier: 80 GiB RAM but only 5 GiB VRAM.
    p = doctor.presets_from_measurements(80 * 1024, 5 * 1024)
    assert p["serve_context"] == 8192   # min(32768, cap 8192)
    assert p["serve_ngl"] == 24         # partial-offload tier

    # Nothing measured -> honest None presets + explanation lines.
    p = doctor.presets_from_measurements(None, None)
    assert p["serve_context"] is None and p["train_batch_size"] is None
    assert any("unavailable" in ln for ln in p["lines"])


def test_free_vram_via_smi_absent_binary(monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda _n: None)
    assert doctor.free_vram_mib_via_smi() is None


def test_free_vram_via_smi_parses_output(monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda _n: "/usr/bin/nvidia-smi")

    def fake_run(cmd, **kw):
        return SimpleNamespace(returncode=0, stdout="24000\n12000\n")

    assert doctor.free_vram_mib_via_smi(run=fake_run) == 12000


# ---------------------------------------------------------- next-command logic

def test_noise_floor_missing_suggests_bench(tmp_path):
    res = doctor.check_noise_floor(tmp_path, load_fn=lambda **kw: None)
    assert res.status == "warn"
    assert res.next_commands == ["run: toaster_loop bench --noise-floor"]


def test_noise_floor_present_is_ok(tmp_path):
    res = doctor.check_noise_floor(
        tmp_path, load_fn=lambda **kw: {"label": "base"})
    assert res.status == "ok" and not res.next_commands


def test_calibration_auto_without_profile_suggests_calibrate(tmp_path):
    cfg = SimpleNamespace(serve={"calibration": "auto"})
    res = doctor.check_calibration_profile(
        cfg, tmp_path, mode_fn=lambda c: "auto", list_fn=lambda **kw: [])
    assert res.status == "warn"
    assert res.next_commands == ["run: toaster_loop calibrate --quick"]


def test_calibration_off_needs_nothing(tmp_path):
    res = doctor.check_calibration_profile(
        SimpleNamespace(), tmp_path, mode_fn=lambda c: "off",
        list_fn=lambda **kw: [])
    assert res.status == "ok" and not res.next_commands


def test_llama_binary_missing_suggests_setup():
    def boom():
        raise RuntimeError("no binary")

    res = doctor.check_llama_binary(boom)
    assert res.status == "problem"
    assert res.next_commands == ["run: toaster_loop setup"]


# --------------------------------------------------------------------- ports

def test_port_check_busy_foreign_is_problem():
    res = doctor.check_ports([8080, 8081], [8081], bind_fn=lambda p: False)
    assert res.status == "problem"
    joined = "\n".join(res.lines)
    assert "held by a live toaster_loop instance (OK)" in joined
    assert "BUSY" in joined


def test_port_check_free_is_ok():
    assert doctor.check_ports([9999], [], bind_fn=lambda p: True).status == "ok"


# --------------------------------------------------------------- stale state

def _mk_state(home: Path, iid: str, body: dict) -> Path:
    d = home / "serving" / "state"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{iid}.json"
    p.write_text(json.dumps(body), encoding="utf-8")
    return p


def test_stale_state_reported_but_not_deleted_by_default(tmp_path):
    dead = _mk_state(tmp_path, "a", {"pid": 424242})
    foreign = _mk_state(tmp_path, "b", {"pid": 424243})
    attach = _mk_state(tmp_path, "c", {"mode": "attach",
                                       "base_url": "http://127.0.0.1:1/v1"})

    def alive_fn(pid):
        return None if pid == 424243 else False  # b: not ours to know

    res = doctor.check_stale_state(tmp_path, alive_fn=alive_fn)
    assert res.status == "problem"
    assert dead.is_file() and foreign.is_file() and attach.is_file()
    joined = "\n".join(res.lines)
    assert "424242" in joined and "424243" not in joined  # foreign never listed
    assert any("--fix" in c for c in res.next_commands)


def test_stale_state_fix_removes_only_verified_gone(tmp_path):
    dead = _mk_state(tmp_path, "a", {"pid": 424242})
    foreign = _mk_state(tmp_path, "b", {"pid": 424243})
    live = _mk_state(tmp_path, "c", {"pid": 424244})

    def alive_fn(pid):
        return {424242: False, 424243: None, 424244: True}[pid]

    res = doctor.check_stale_state(tmp_path, alive_fn=alive_fn, fix=True)
    assert not dead.is_file()          # deterministically gone -> removed
    assert foreign.is_file()           # ownership rule: never touched
    assert live.is_file()              # still running -> kept
    assert res.status == "ok"


def test_stale_state_ignores_garbage_files(tmp_path):
    d = tmp_path / "serving" / "state"
    d.mkdir(parents=True)
    (d / "broken.json").write_text("{not json", encoding="utf-8")
    res = doctor.check_stale_state(tmp_path, alive_fn=lambda p: False)
    assert res.status == "ok"


# -------------------------------------------------------------------- config

def test_config_error_passthrough_exit_3(tmp_path, monkeypatch):
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(tmp_path))
    boom_msg = ("invalid configuration — refusing to guess (2 problems):\n"
                "  - port: expected an integer, found 'abc' (str)\n"
                "  - gguf_path: expected a string, found 3 (int)")

    def bad_load(_path=None):
        raise ConfigError(boom_msg)

    lines: list[str] = []
    rc = doctor.collect_checks(load_config_fn=bad_load, out=lines.append,
                               binary_probe=lambda: "/x/llama-server",
                               meminfo_kib=1024 * 1024, vram_mib=None,
                               alive_fn=lambda p: True)
    results, code = rc
    assert code == EXIT_CONFIG_INVALID
    # The consolidated report is surfaced VERBATIM (every field at once).
    text = "\n".join(lines)
    assert "port: expected an integer" in text
    assert "gguf_path: expected a string" in text


def test_config_ok_when_parses(tmp_path, monkeypatch):
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(tmp_path))
    res = doctor.check_config(None, out=lambda s: None,
                              load_config_fn=lambda _p=None: SimpleNamespace())
    assert res.status == "ok"


# ------------------------------------------------------------- full runner

def test_collect_checks_green_with_injected_probes(tmp_path, monkeypatch):
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(tmp_path))
    # Persisted noise floor + calibration profile so no warn->next-command.
    bench = tmp_path / "bench"
    bench.mkdir()
    (bench / "noise_floor-x.json").write_text(
        json.dumps({"kind": "noise_floor", "label": ""}), encoding="utf-8")

    results, rc = doctor.collect_checks(
        load_config_fn=lambda _p=None: ToasterLoopConfig(),
        disk_usage_fn=lambda _p: FakeUsage(free=64 * 1024 ** 3),
        meminfo_kib=32 * 1024 * 1024,   # 32 GiB available
        vram_mib=13000,
        binary_probe=lambda: "/fake/llama-server",
        alive_fn=lambda _pid: True,
    )
    assert rc == EXIT_OK
    by_name = {r.name: r for r in results}
    assert by_name["memory"].status == "ok"
    assert by_name["noise-floor"].status == "ok"
    assert by_name["llama-server"].lines == ["/fake/llama-server"]


def test_collect_checks_problems_yield_exit_1(tmp_path, monkeypatch):
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(tmp_path))

    def dead(_pid):
        return False

    _mk_state(tmp_path, "a", {"pid": 999999})
    results, rc = doctor.collect_checks(
        load_config_fn=lambda _p=None: ToasterLoopConfig(),
        disk_usage_fn=lambda _p: FakeUsage(free=64 * 1024 ** 3),
        meminfo_kib=32 * 1024 * 1024, vram_mib=None,
        binary_probe=lambda: "/fake/llama-server",
        alive_fn=dead,
    )
    assert rc == EXIT_ERROR
    assert any(r.name == "state" and r.status == "problem" for r in results)


def test_run_doctor_prints_report_and_next_commands(tmp_path, monkeypatch):
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(tmp_path))
    lines: list[str] = []
    rc = doctor.run_doctor(
        out=lines.append,
        load_config_fn=lambda _p=None: ToasterLoopConfig(),
        disk_usage_fn=lambda _p: FakeUsage(free=64 * 1024 ** 3),
        meminfo_kib=8 * 1024 * 1024, vram_mib=None,
        binary_probe=lambda: "/fake/llama-server",
        alive_fn=lambda _pid: True,
    )
    text = "\n".join(lines)
    assert "toaster_loop doctor" in text and "next commands:" in text
    # No noise floor persisted in tmp home -> the bench suggestion appears.
    assert "toaster_loop bench --noise-floor" in text
    assert rc == EXIT_OK


def test_check_python_bounds():
    assert doctor.check_python((3, 12)).status == "ok"
    assert doctor.check_python((3, 9)).status == "problem"
    assert doctor.check_python((3, 14)).status == "problem"


def test_check_deps_missing_core_dep():
    res = doctor.check_deps(find_spec=lambda m: None if m == "httpx"
                            else object())
    assert res.status == "problem"
    assert any("pip install" in c for c in res.next_commands)


def test_home_writable_check(tmp_path):
    ok = doctor.check_home_writable(tmp_path / "h")
    assert ok.status == "ok" and (tmp_path / "h").is_dir()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))

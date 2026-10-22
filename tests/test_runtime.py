"""Runtime detection + llama.cpp auto-setup wiring (no GPU, no network).

detect_runtime() must never raise regardless of what vendor probes exist, and
launch args must follow the config knob: explicit gpu_layers wins, otherwise
full offload on GPU backends and nothing on CPU-only machines.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from sable.core.config import SableConfig
from sable.serving import runtime
from sable.serving.supervisor import ServeError, find_llama_server


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
    monkeypatch.delenv("SABLE_UNIFIED_MEMORY", raising=False)
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
    monkeypatch.delenv("SABLE_UNIFIED_MEMORY", raising=False)
    monkeypatch.setattr(runtime, "_is_jetson", lambda: False)
    monkeypatch.setattr(runtime, "_is_strix_halo", lambda: True)
    assert runtime.is_unified_memory(
        _um_info(["AMD GPU [1002:1586]"])) is True


def test_unified_memory_jetson(monkeypatch):
    monkeypatch.delenv("SABLE_UNIFIED_MEMORY", raising=False)
    monkeypatch.setattr(runtime, "_is_jetson", lambda: True)
    assert runtime.is_unified_memory(_um_info([], backend="cuda")) is True


def test_unified_memory_mac_excluded(monkeypatch):
    monkeypatch.delenv("SABLE_UNIFIED_MEMORY", raising=False)
    monkeypatch.setattr(runtime, "_is_jetson", lambda: False)
    monkeypatch.setattr(runtime.sys, "platform", "darwin")
    assert runtime.is_unified_memory(_um_info([("apple M4")], backend="metal")) is False


def test_unified_memory_override(monkeypatch):
    monkeypatch.setenv("SABLE_UNIFIED_MEMORY", "0")
    assert runtime.is_unified_memory(_um_info(["gfx1151"])) is False
    monkeypatch.setenv("SABLE_UNIFIED_MEMORY", "1")
    assert runtime.is_unified_memory(_um_info(["gfx1030"])) is True


def test_unified_memory_env_value(monkeypatch):
    monkeypatch.delenv("SABLE_UNIFIED_MEMORY", raising=False)
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

def test_search_dirs_include_sable_and_env_root(tmp_path, monkeypatch):
    monkeypatch.setenv("SABLE_LLAMA_CPP", str(tmp_path / "llama.cpp"))
    dirs = runtime.search_dirs()
    assert tmp_path / "llama.cpp" / "build" / "bin" in dirs
    # sable is standalone: it must never depend on (or probe) the separate
    # unrelated per-installer directories.
    homes = [d for d in dirs if ".unrelated-installer" in str(d)]
    assert not homes, "sable must not search foreign installer dirs"


def test_discover_external_server_url_opt_in_only(monkeypatch):
    # No explicit env var => no probing at all (no default-port scan).
    from sable.core.config import discover_external_server_url

    monkeypatch.delenv("SABLE_EXTERNAL_SERVER_URL", raising=False)
    assert discover_external_server_url() is None


def test_sable_llama_root_uses_home_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SABLE_HOME", str(tmp_path))
    assert runtime.sable_llama_root() == tmp_path / "llama.cpp"


# ------------------------------------------------------------ auto_setup

def test_auto_setup_enabled_default(monkeypatch):
    monkeypatch.delenv("SABLE_NO_AUTO_SETUP", raising=False)
    assert runtime.auto_setup_enabled(SableConfig()) is True


def test_auto_setup_env_optout(monkeypatch):
    monkeypatch.setenv("SABLE_NO_AUTO_SETUP", "1")
    assert runtime.auto_setup_enabled(SableConfig()) is False


def test_auto_setup_config_optout(monkeypatch):
    monkeypatch.delenv("SABLE_NO_AUTO_SETUP", raising=False)
    assert runtime.auto_setup_enabled(SableConfig(llama_auto_setup=False)) is False


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
    with pytest.raises(ServeError, match="sable setup"):
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
    from sable.core.config import load_config

    monkeypatch.setenv("SABLE_GPU_LAYERS", "17")
    assert load_config().gpu_layers == 17


def test_auto_setup_env_override(monkeypatch):
    from sable.core.config import load_config

    monkeypatch.setenv("SABLE_LLAMA_AUTO_SETUP", "false")
    assert load_config().llama_auto_setup is False


def test_llama_source_env_overrides(monkeypatch):
    from sable.core.config import load_config

    monkeypatch.setenv("SABLE_LLAMA_REPO", "https://example.com/fork/llama.cpp")
    monkeypatch.setenv("SABLE_LLAMA_REF", "gfx1151")
    monkeypatch.setenv("SABLE_LLAMA_BACKEND", "vulkan")
    cfg = load_config()
    assert cfg.llama_repo == "https://example.com/fork/llama.cpp"
    assert cfg.llama_ref == "gfx1151"
    assert cfg.llama_backend == "vulkan"


# ------------------------------------------------- install state + updates

def test_install_state_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("SABLE_HOME", str(tmp_path))
    st = {"repo": "https://example.com/fork", "ref": "b1", "backend": "vulkan",
          "commit": "abc1234", "built_at": "2026-09-15T00:00:00Z"}
    runtime._write_install_state(st)
    assert runtime.read_install_state() == st
    # State lives OUTSIDE the checkout (a git clean can't fake it).
    assert runtime.state_path() == tmp_path / runtime.STATE_FILENAME


def test_check_llama_update_not_installed(tmp_path, monkeypatch):
    monkeypatch.setenv("SABLE_HOME", str(tmp_path))
    res = runtime.check_llama_update()
    assert res["installed"] is False and res["update_available"] is False


def test_check_llama_update_behind(tmp_path, monkeypatch):
    # Fake checkout + state; remote probe faked to a different sha.
    root = tmp_path / "llama.cpp"
    (root / ".git").mkdir(parents=True)
    (root / "CMakeLists.txt").write_text("project")
    monkeypatch.setenv("SABLE_HOME", str(tmp_path))
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
    monkeypatch.setenv("SABLE_HOME", str(tmp_path))
    runtime._write_install_state({"repo": "r", "ref": "m", "commit": "a"})
    monkeypatch.setattr(runtime, "git_head", lambda r: "a")
    monkeypatch.setattr(runtime, "remote_ref_commit", lambda repo, ref: None)
    res = runtime.check_llama_update()
    assert res["update_available"] is False and "offline" in res["reason"]


def test_uninstall_only_touches_managed_install(tmp_path, monkeypatch):
    monkeypatch.setenv("SABLE_HOME", str(tmp_path))
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
    monkeypatch.setenv("SABLE_HOME", str(tmp_path))
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
    monkeypatch.setenv("SABLE_HOME", str(tmp_path))
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

def test_sable_llama_root_per_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("SABLE_HOME", str(tmp_path))
    assert runtime.sable_llama_root() == tmp_path / "llama.cpp"
    assert runtime.sable_llama_root("vulkan") == tmp_path / "llama.cpp-vulkan"
    assert runtime.sable_llama_root("ROCM") == tmp_path / "llama.cpp-rocm"


def test_install_root_for_routing(tmp_path, monkeypatch):
    """Primary slot hosts the first backend; a *different* backend routes to
    its own llama.cpp-<backend> checkout instead of overwriting it."""
    monkeypatch.setenv("SABLE_HOME", str(tmp_path))
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
    monkeypatch.setenv("SABLE_HOME", str(tmp_path))
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
    monkeypatch.setenv("SABLE_HOME", str(tmp_path))
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
    monkeypatch.setenv("SABLE_HOME", str(tmp_path))
    runtime._write_install_state({"backend": "rocm"}, backend=None)
    runtime._write_install_state({"backend": "vulkan"}, backend="vulkan")
    dirs = [str(p) for p in runtime.search_dirs()]
    assert str(tmp_path / "llama.cpp" / "build" / "bin") in dirs
    assert str(tmp_path / "llama.cpp-vulkan" / "build" / "bin") in dirs


def test_uninstall_backend_slot_keeps_primary(tmp_path, monkeypatch):
    monkeypatch.setenv("SABLE_HOME", str(tmp_path))
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
    monkeypatch.setenv("SABLE_HOME", str(tmp_path))
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
    from sable.core.config import SableConfig
    from sable.dashboard.app import DashboardApp

    monkeypatch.setenv("SABLE_HOME", str(tmp_path))
    app = DashboardApp(SableConfig(), home=tmp_path)

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

    from sable.core.config import SableConfig
    from sable.dashboard.app import DashboardApp

    monkeypatch.setenv("SABLE_HOME", str(tmp_path))
    app = DashboardApp(SableConfig(), home=tmp_path)

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

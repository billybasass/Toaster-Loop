"""Installable engine catalog tests (serving/engine_installs.py + dashboard API).

Hermetic: tmp SABLE_HOME (fixture), no GPU, no network, no subprocesses —
every git/pip/cmake invocation is faked by monkeypatching the module's
``_stream``/``_stream_out``/``_require`` seams. Background install threads are
driven through a fake ``install_recipe`` and polled to completion.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sable.core.config import SableConfig  # noqa: E402
from sable.dashboard.app import DashboardApp  # noqa: E402
from sable.serving import engine_installs as ei  # noqa: E402
from sable.serving import engines as eng_mod  # noqa: E402


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "uf-installs"
    monkeypatch.setenv("SABLE_HOME", str(h))
    return h


@pytest.fixture()
def app(home):
    return DashboardApp(SableConfig(), home=home)


def err_code(r) -> str:
    return (r.body.get("error") or {}).get("code", "")


def fake_host(system="linux", backend="cpu", gpus=()):
    """Host profile shaped like ei.detect_host()'s output. Passing it
    explicitly keeps every test hermetic — no vendor CLI probes."""
    return {"system": system, "machine": "x86_64", "backend": backend,
            "gpus": list(gpus)}


@pytest.fixture(autouse=True)
def cached_host(monkeypatch):
    """Pin the process-level host cache so install_recipe()'s detect_host()
    never shells out to nvidia-smi/rocminfo/vulkaninfo in tests. Individual
    tests override by passing host= or re-patching ei.detect_host."""
    monkeypatch.setattr(ei, "_HOST_CACHE", fake_host())


# ------------------------------------------------------------------ catalog

def test_catalog_contains_the_four_engines():
    ids = ei.catalog_ids()
    assert ids == ["vllm", "airllm", "strata", "sd-cpp"]
    vllm = ei.get_recipe("vllm")
    assert vllm["kind"] == "openai"
    assert "{model}" in " ".join(vllm["args_template"])
    assert "127.0.0.1" not in " ".join(vllm["args_template"])  # host via {host}
    # library / cli engines are inventory-only (never spawned by sable)
    for rid in ("airllm", "strata", "sd-cpp"):
        assert ei.get_recipe(rid)["kind"] == "other"
    with pytest.raises(KeyError):
        ei.get_recipe("nope")


def test_catalog_recipes_validate_as_engine_bodies():
    """Every recipe must produce a body the engine registry accepts."""
    for r in ei.CATALOG:
        st = {"binary": "/opt/x/bin/thing", "installed_at": "2026-01-01T00:00:00Z"}
        body = ei.engine_body_for(r["id"], st)
        eng = eng_mod.validate_engine_body(body)
        assert eng["name"] == r["name"]
        assert eng["kind"] == r["kind"]


def test_paths_under_home(home):
    assert str(ei.venv_dir(home, "vllm")).startswith(str(home))
    assert str(ei.src_dir(home, "sd-cpp")).startswith(str(home))
    assert str(ei.state_path(home, "airllm")).startswith(str(home))
    with pytest.raises(KeyError):
        ei.engine_dir(home, "escape")


# ------------------------------------------------------------ state file

def test_state_roundtrip_and_installed_probe(home):
    assert ei.read_state(home, "vllm") is None
    assert ei.recipe_installed(home, "vllm") is False

    ei.write_state(home, "vllm", {"binary": str(home / "missing"), "x": 1})
    st = ei.read_state(home, "vllm")
    assert st is not None and st["x"] == 1
    # state recorded but the binary vanished => not installed
    assert ei.recipe_installed(home, "vllm") is False

    b = home / "real-binary"
    b.write_text("#!/bin/sh\n")
    ei.write_state(home, "vllm", {"binary": str(b)})
    assert ei.recipe_installed(home, "vllm") is True


def test_list_catalog_probes_without_subprocess(home, monkeypatch):
    def _boom(*a, **k):  # listing must never shell out
        raise AssertionError("listing endpoint ran a subprocess")
    monkeypatch.setattr(ei.subprocess, "Popen", _boom)
    monkeypatch.setattr(ei.subprocess, "run", _boom)
    rows = ei.list_catalog(home)
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
def fake_shell(home, monkeypatch):
    fs = FakeShell(home)
    monkeypatch.setattr(ei, "_stream", fs.stream)
    monkeypatch.setattr(ei, "_stream_out", fs.stream_out)
    monkeypatch.setattr(ei, "_require", fs.require)
    return fs


def test_install_pip_recipe_creates_venv_and_state(home, fake_shell):
    st = ei.install_recipe("vllm", home=home, log_fn=lambda _s: None)
    assert st["method"] == "pip"
    assert Path(st["binary"]).is_file()  # faked venv python exists
    assert "venv" in st["binary"] and "engines" in st["binary"]
    assert ei.recipe_installed(home, "vllm") is True
    joined = "\n".join(" ".join(c) for c in fake_shell.cmds)
    assert "-m venv" in joined
    assert "pip" in joined and "install" in joined and "vllm" in joined
    # pip NEVER runs against the sable interpreter — only the venv python
    pip_cmds = [c for c in fake_shell.cmds if "pip" in c]
    assert pip_cmds
    venv_py = str(ei.venv_python(home, "vllm"))
    for c in pip_cmds:
        assert c[0] == venv_py, f"pip ran outside the private venv: {c}"
    assert sys.executable not in [c[0] for c in pip_cmds]


def test_install_pip_failure_raises_install_error(home, fake_shell, monkeypatch):
    # only the pip step fails; venv creation succeeds
    fs = FakeShell(home, fail_when=lambda c: "pip" in c)
    monkeypatch.setattr(ei, "_stream", fs.stream)
    with pytest.raises(ei.InstallError, match="pip install"):
        ei.install_recipe("vllm", home=home, log_fn=lambda _s: None)
    assert ei.recipe_installed(home, "vllm") is False
    assert ei.read_state(home, "vllm") is None  # no state on failure


def test_install_git_pip_recipe_clones_editable(home, fake_shell, monkeypatch):
    # A pip-installable checkout (setuptools setup.py) still installs editable.
    monkeypatch.setattr(ei, "_stream", _app_style_clone(
        fake_shell, "strata",
        "from setuptools import setup\nsetup(name='strata')\n"))
    st = ei.install_recipe("strata", home=home, log_fn=lambda _s: None)
    joined = "\n".join(" ".join(c) for c in fake_shell.cmds)
    assert "git clone" in joined and "niko1221/strata" in joined
    assert "pip" in joined and "-e" in joined
    assert st["commit"] == "deadbee"
    assert st["variant"] == "editable"
    assert Path(st["binary"]).is_file()


def test_install_git_pip_requirements_file_uses_dash_r(home, fake_shell,
                                                       monkeypatch):
    """A requirements.txt inside the checkout must install via `-r <file>`.

    Regression: passing the path as a positional spec makes pip fail with
    "Invalid requirement: … Expected package name at the start of dependency
    specifier" (the exact strata install failure)."""
    req = ei.src_dir(home, "strata") / "requirements.txt"

    def stream_with_req(cmd, log, cwd=None):
        rc = fake_shell.stream(cmd, log, cwd=cwd)
        if "clone" in cmd:  # clone materialized the checkout — add its reqs
            req.parent.mkdir(parents=True, exist_ok=True)
            req.write_text("numpy\n")
        return rc
    monkeypatch.setattr(ei, "_stream", stream_with_req)

    st = ei.install_recipe("strata", home=home, log_fn=lambda _s: None)
    pip_cmds = [c for c in fake_shell.cmds if "pip" in c and "install" in c]
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


def test_is_pip_installable_detects_app_style_setup_py(home):
    sdir = ei.src_dir(home, "strata")
    sdir.mkdir(parents=True)
    assert ei._is_pip_installable(sdir) is False          # nothing there
    (sdir / "setup.py").write_text(APP_STYLE_SETUP_PY)
    assert ei._is_pip_installable(sdir) is False          # argparse-only CLI
    (sdir / "pyproject.toml").write_text("[project]\nname='x'\n")
    assert ei._is_pip_installable(sdir) is True           # pyproject wins


def test_install_app_style_repo_skips_editable(home, fake_shell, monkeypatch):
    """strata regression: `pip install -e .` on an application setup.py dies
    with `setup.py: error: unrecognized arguments: egg_info`. Install must
    succeed as checkout+deps instead."""
    monkeypatch.setattr(ei, "_stream", _app_style_clone(fake_shell, "strata",
                                                        APP_STYLE_SETUP_PY))
    st = ei.install_recipe("strata", home=home, log_fn=lambda _s: None)
    joined = "\n".join(" ".join(c) for c in fake_shell.cmds)
    assert "-e" not in joined  # editable install never attempted
    assert st["method"] == "git_pip" and st["variant"] == "checkout"
    assert Path(st["binary"]).is_file()
    assert ei.recipe_installed(home, "strata") is True


def test_install_editable_egg_info_failure_downgrades(home, fake_shell,
                                                      monkeypatch):
    """Even when detection passes (pyproject present), an egg_info failure at
    pip runtime downgrades to checkout instead of erroring."""
    def egg_failing_stream(cmd, log, cwd=None):
        rc = fake_shell.stream(cmd, log, cwd=cwd)
        if "clone" in cmd:  # pip-installable metadata on disk
            (Path(cmd[-1]) / "pyproject.toml").write_text(
                "[build-system]\nrequires=['setuptools']\n")
        if "-e" in [str(c) for c in cmd]:  # exact arg, not a path substring
            log("setup.py: error: unrecognized arguments: egg_info")
            return 1
        return rc
    monkeypatch.setattr(ei, "_stream", egg_failing_stream)

    st = ei.install_recipe("strata", home=home, log_fn=lambda _s: None)
    assert st["variant"] == "checkout"
    assert ei.recipe_installed(home, "strata") is True


def test_install_editable_other_failure_still_raises(home, fake_shell,
                                                     monkeypatch):
    """A normal pip failure (network, build error…) must NOT be swallowed by
    the egg_info downgrade."""
    def failing_stream(cmd, log, cwd=None):
        rc = fake_shell.stream(cmd, log, cwd=cwd)
        if "clone" in cmd:
            (Path(cmd[-1]) / "pyproject.toml").write_text("[project]\n")
        if "-e" in [str(c) for c in cmd]:  # exact arg, not a path substring
            log("error: connection timed out")
            return 1
        return rc
    monkeypatch.setattr(ei, "_stream", failing_stream)

    with pytest.raises(ei.InstallError, match="pip install"):
        ei.install_recipe("strata", home=home, log_fn=lambda _s: None)


def test_install_git_cmake_recipe_builds_binary(home, fake_shell):
    st = ei.install_recipe("sd-cpp", home=home, log_fn=lambda _s: None)
    joined = "\n".join(" ".join(c) for c in fake_shell.cmds)
    assert "--recursive" in joined  # ggml submodule
    assert "cmake" in joined and "--build" in joined
    assert st["binary"].endswith("sd-cli" + (".exe" if os.name == "nt" else ""))
    assert ei.recipe_installed(home, "sd-cpp") is True


def test_install_git_cmake_missing_binary_errors(home, monkeypatch):
    monkeypatch.setattr(ei, "_stream", lambda cmd, log, cwd=None: 0)
    monkeypatch.setattr(ei, "_require", lambda tool, hint: f"/usr/bin/{tool}")
    monkeypatch.setattr(ei, "_git_commit", lambda dest: None)
    with pytest.raises(ei.InstallError, match="sd-cli"):
        ei.install_recipe("sd-cpp", home=home, log_fn=lambda _s: None)


def test_install_update_reuses_checkout(home, fake_shell):
    ei.install_recipe("strata", home=home, log_fn=lambda _s: None)
    fake_shell.cmds.clear()
    st2 = ei.install_recipe("strata", home=home, update=True,
                            log_fn=lambda _s: None)
    joined = "\n".join(" ".join(c) for c in fake_shell.cmds)
    assert "git clone" not in joined          # reused, not re-cloned
    assert "fetch" in joined and "reset" in joined
    assert st2["method"] == "git_pip"


# ---------------------------------------------------------- dashboard API

def test_catalog_routes_registered(app):
    paths = {(m, p) for (m, p, _) in app.ROUTES}
    for m, p in [("GET", "/api/engine-catalog"),
                 ("POST", "/api/engine-catalog/<recipe_id>/install"),
                 ("GET", "/api/engine-catalog/<recipe_id>/install")]:
        assert (m, p) in paths, f"{m} {p} missing"


def test_catalog_endpoint_lists_all_not_installed(app, home):
    r = app.handle("GET", "/api/engine-catalog")
    assert r.status == 200
    ids = [c["id"] for c in r.body["catalog"]]
    assert ids == ["vllm", "airllm", "strata", "sd-cpp"]
    assert all(c["installed"] is False for c in r.body["catalog"])
    assert all(c["installing"] is False for c in r.body["catalog"])


def test_install_status_unknown_recipe_404(app, home):
    r = app.handle("GET", "/api/engine-catalog/ghost/install")
    assert r.status == 404 and err_code(r) == "unknown_recipe"
    r2 = app.handle("POST", "/api/engine-catalog/ghost/install", body={})
    assert r2.status == 404 and err_code(r2) == "unknown_recipe"


def _wait_done(app, rid, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = app.handle("GET", f"/api/engine-catalog/{rid}/install").body
        if s.get("state") in ("done", "error", "idle"):
            return s
        time.sleep(0.02)
    raise AssertionError(f"install {rid} never finished")


def test_install_endpoint_runs_and_registers_engine(app, home, monkeypatch):
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
    r = app.handle("POST", "/api/engine-catalog/vllm/install", body={})
    assert r.status == 200 and r.body["state"] == "running"
    done = _wait_done(app, "vllm")
    assert done["state"] == "done", done.get("error")
    assert calls == ["vllm"]

    # engine got registered with the recipe id and install binary
    lst = app.handle("GET", "/api/engines").body["engines"]
    row = next(e for e in lst if e["id"] == "vllm")
    assert row["kind"] == "openai"
    assert row["installed"] is True
    assert row["binary"].endswith("vllm")
    # vLLM spawn argv: `vllm serve {model} --host {host} --port {port}`
    cmd = eng_mod.build_engine_cmd(row, row["binary"], model="m.gguf",
                                   host="127.0.0.1", port=8001)
    assert cmd[:2] == [row["binary"], "serve"] and "8001" in cmd


def test_install_endpoint_reports_error(app, home, monkeypatch):
    def boom(recipe_id, *, home, ref=None, update=False, backend=None,
             force=False, log_fn=None):
        raise ei.InstallError("cmake missing")

    monkeypatch.setattr(ei, "install_recipe", boom)
    r = app.handle("POST", "/api/engine-catalog/sd-cpp/install", body={})
    assert r.status == 200
    done = _wait_done(app, "sd-cpp")
    assert done["state"] == "error" and "cmake missing" in done["error"]
    ids = [e["id"] for e in app.handle("GET", "/api/engines").body["engines"]]
    assert "sd-cpp" not in ids  # failed install registers nothing


def test_install_endpoint_conflict_while_running(app, home):
    app._engine_installs["vllm"] = {"state": "running", "log": []}
    r = app.handle("POST", "/api/engine-catalog/vllm/install", body={})
    assert r.status == 409 and err_code(r) == "already_running"


def test_install_register_false_skips_registry(app, home, monkeypatch):
    def fake_install(recipe_id, *, home, ref=None, update=False, backend=None,
                     force=False, log_fn=None):
        return {"recipe": recipe_id, "method": "pip",
                "binary": "/nowhere/airllm-python"}

    monkeypatch.setattr(ei, "install_recipe", fake_install)
    r = app.handle("POST", "/api/engine-catalog/airllm/install",
                   body={"register": False})
    assert r.status == 200
    done = _wait_done(app, "airllm")
    assert done["state"] == "done"
    ids = [e["id"] for e in app.handle("GET", "/api/engines").body["engines"]]
    assert "airllm" not in ids


# ------------------------------------------------- hardware / OS awareness

def test_vllm_blocked_on_windows_and_macos():
    """docs.vllm.ai: Linux only (Windows via WSL). Hard block, not warning."""
    for system in ("windows", "darwin"):
        c = ei.recipe_compat("vllm", fake_host(system=system))
        assert c["ok"] is False and system in c["blocked"]
    ok = ei.recipe_compat("vllm", fake_host(system="linux"))
    assert ok["ok"] is True


def test_vllm_variant_follows_gpu_vendor():
    nvidia = [{"vendor": "nvidia", "name": "RTX 4090"}]
    amd = [{"vendor": "amd", "name": "gfx1100 (Radeon RX 7900)"}]
    c_n = ei.recipe_compat("vllm", fake_host(backend="cuda", gpus=nvidia))
    assert c_n["variant"] == "cuda" and not c_n["warnings"]
    c_a = ei.recipe_compat("vllm", fake_host(backend="rocm", gpus=amd))
    assert c_a["variant"] == "rocm"
    c_none = ei.recipe_compat("vllm", fake_host())  # no GPUs
    assert c_none["ok"] is True and c_none["warnings"]  # advisory only


def test_vllm_rocm_install_uses_rocm_wheel_index(home, fake_shell, monkeypatch):
    """AMD host => pip must get --extra-index-url wheels.vllm.ai/rocm/."""
    monkeypatch.setattr(ei, "detect_host", lambda: fake_host(
        backend="rocm", gpus=[{"vendor": "amd", "name": "gfx942"}]))
    ei.install_recipe("vllm", home=home, log_fn=lambda _s: None)
    pip_cmds = [c for c in fake_shell.cmds if "pip" in c and "install" in c]
    assert any("wheels.vllm.ai/rocm/" in " ".join(c) for c in pip_cmds)
    st = ei.read_state(home, "vllm")
    assert st is not None and st["variant"] == "rocm"


def test_vllm_cuda_install_uses_plain_pypi(home, fake_shell, monkeypatch):
    monkeypatch.setattr(ei, "detect_host", lambda: fake_host(
        backend="cuda", gpus=[{"vendor": "nvidia", "name": "RTX 4090"}]))
    ei.install_recipe("vllm", home=home, log_fn=lambda _s: None)
    pip_cmds = [c for c in fake_shell.cmds if "pip" in c and "install" in c]
    assert pip_cmds and not any("--extra-index-url" in c for c in pip_cmds)


def test_install_blocked_on_incompatible_os(home, fake_shell, monkeypatch):
    monkeypatch.setattr(ei, "detect_host", lambda: fake_host(system="windows"))
    with pytest.raises(ei.InstallError, match="does not support windows"):
        ei.install_recipe("vllm", home=home, log_fn=lambda _s: None)
    assert fake_shell.cmds == []          # nothing ran — no network touched
    # force overrides the gate (user knows best; upstream will fail later)
    st = ei.install_recipe("vllm", home=home, force=True,
                           log_fn=lambda _s: None)
    assert st["method"] == "pip"


def test_sd_cpp_backend_flags_per_host(home, fake_shell):
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


def test_sd_cpp_install_passes_detected_backend(home, monkeypatch):
    cfg_cmds = []

    def rec(cmd, log, cwd=None):
        if "-S" in cmd:
            cfg_cmds.append([str(c) for c in cmd])
        return FakeShell.stream(fs, cmd, log, cwd=cwd)
    fs = FakeShell(home)
    monkeypatch.setattr(ei, "_stream", rec)
    monkeypatch.setattr(ei, "_stream_out", fs.stream_out)
    monkeypatch.setattr(ei, "_require", fs.require)
    monkeypatch.setattr(ei, "detect_host", lambda: fake_host(
        backend="vulkan", gpus=[{"vendor": "intel", "name": "Arc A770"}]))
    st = ei.install_recipe("sd-cpp", home=home, log_fn=lambda _s: None)
    assert any("-DSD_VULKAN=ON" in c for c in cfg_cmds)
    assert st["backend"] == "vulkan"


def test_sd_cpp_backend_override(home, monkeypatch):
    cfg_cmds = []

    def rec(cmd, log, cwd=None):
        if "-S" in cmd:
            cfg_cmds.append([str(c) for c in cmd])
        return FakeShell.stream(fs, cmd, log, cwd=cwd)
    fs = FakeShell(home)
    monkeypatch.setattr(ei, "_stream", rec)
    monkeypatch.setattr(ei, "_stream_out", fs.stream_out)
    monkeypatch.setattr(ei, "_require", fs.require)
    ei.install_recipe("sd-cpp", home=home, backend="cuda",
                      log_fn=lambda _s: None)
    assert any("-DSD_CUDA=ON" in c for c in cfg_cmds)


def test_airllm_warns_without_nvidia():
    c = ei.recipe_compat("airllm", fake_host())
    assert c["ok"] is True and c["warnings"]
    c2 = ei.recipe_compat("airllm", fake_host(
        backend="cuda", gpus=[{"vendor": "nvidia", "name": "T4"}]))
    assert not c2["warnings"]


def test_amd_gfx_target_parsing():
    h = fake_host(gpus=[{"vendor": "amd", "name": "gfx942 (MI300X)"}])
    assert ei.amd_gfx_target(h) == "gfx942"
    assert ei.amd_gfx_target(fake_host()) is None


def test_catalog_endpoint_includes_compat_and_host(app, monkeypatch):
    monkeypatch.setattr(ei, "detect_host", lambda: fake_host(system="windows"))
    r = app.handle("GET", "/api/engine-catalog")
    assert r.status == 200
    assert r.body["host"]["system"] == "windows"
    by_id = {c["id"]: c for c in r.body["catalog"]}
    assert by_id["vllm"]["compat"]["ok"] is False
    assert by_id["sd-cpp"]["compat"]["ok"] is True


def test_install_endpoint_rejects_incompatible_host(app, home, monkeypatch):
    monkeypatch.setattr(ei, "detect_host", lambda: fake_host(system="windows"))
    r = app.handle("POST", "/api/engine-catalog/vllm/install", body={})
    assert r.status == 400 and err_code(r) == "incompatible_host"
    # force goes through (install itself is faked here)
    monkeypatch.setattr(ei, "install_recipe", lambda *a, **k: {
        "recipe": "vllm", "method": "pip", "binary": "/x"})
    r2 = app.handle("POST", "/api/engine-catalog/vllm/install",
                    body={"force": True})
    assert r2.status == 200


def test_reregister_preserves_user_edits(app, home):
    """Re-running an install refreshes binary/args but keeps user notes."""
    eng = eng_mod.validate_engine_body({"name": "vLLM", "binary": "/old/vllm",
                                        "kind": "openai"})
    eng["id"] = "vllm"
    eng["notes"] = "my custom notes"
    eng_mod.save_engine(home, eng)

    app._register_installed_engine("vllm", {
        "binary": "/new/venv/bin/vllm", "installed_at": "2026-02-02T00:00:00Z"})
    again = eng_mod.get_engine(home, "vllm")
    assert again is not None
    assert again["binary"] == "/new/venv/bin/vllm"
    assert again["notes"] == "my custom notes"

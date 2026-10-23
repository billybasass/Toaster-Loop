"""Installable engine catalog (opt-in; Hardware page "Install engines").

The inference-engine registry (:mod:`sable.serving.engines`) lets users
*register* engines they already have. This module adds the other half: a small
curated catalog of known local engines sable can *install for you*, managed
entirely under ``<SABLE_HOME>/engines/<id>/`` — a private venv per Python
engine (never the sable process's own interpreter environment, so installing
vLLM can never touch core deps) and a git checkout per source-built engine.

Cardinal rule #1: nothing here runs on import, on boot, or without an explicit
user action. The catalog listing is pure data + stat() probes; the install
runner only ever executes when POST /api/engine-catalog/<id>/install fires
(the dashboard's Install button). Network use (git clone / pip install) is an
explicit user action exactly like ``sable setup`` already is.

Kinds follow the registry: ``openai`` engines get an args_template with
{model}/{host}/{port} placeholders so sable can spawn them loopback-only;
library/CLI engines (AirLLM, Strata, stable-diffusion.cpp) register as
``other`` — inventory + bookkeeping, never spawned.

Licensing (D-12): these projects are *installed side-by-side*, never imported;
sable only records paths and spawn argv for them.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("sable.serving.engine_installs")


class InstallError(Exception):
    """Actionable install failure (prerequisites, clone/build/pip errors)."""


# ------------------------------------------------------------------ catalog
# Each recipe: id, name, homepage, kind (engine kind), method
# (pip | git_pip | git_cmake), plus method-specific fields:
#   pip      -> packages: [pypi specs], variants?: {cuda|rocm: {extra_index_url?,
#               python_note?}}  (hardware-aware wheel index selection)
#   git_pip  -> repo, ref?, requirements? (file name inside repo)
#   git_cmake-> repo, ref?, recursive?, binary_name, cmake_backends?
#               (backend -> extra cmake flags; auto-picked from detected hw)
# binary_name: executable the install must produce (venv bin dir for pip
# methods, build/bin for cmake). args_template only for kind=openai.
# os_support drives recipe_compat() — the OS/hardware gate the dashboard shows
# and install_recipe enforces before touching the network.

CATALOG: List[Dict[str, Any]] = [
    {
        "id": "vllm",
        "name": "vLLM",
        "homepage": "https://github.com/vllm-project/vllm",
        "kind": "openai",
        "method": "pip",
        "packages": ["vllm"],
        "binary_name": "vllm",
        # `vllm serve` exposes an OpenAI-compatible endpoint; sable always
        # substitutes a loopback host.
        "args_template": ["serve", "{model}", "--host", "{host}",
                          "--port", "{port}"],
        # Verified against docs.vllm.ai (2026-10): OS Linux only (Windows via
        # WSL), Python 3.11–3.14; NVIDIA wheels are the PyPI default, AMD ROCm
        # wheels come from wheels.vllm.ai/rocm (ROCm 6.3+; prebuilt wheels are
        # Python 3.12; gfx90a/gfx942/gfx950/gfx1100/1101). Intel XPU and Apple
        # Silicon go through separate community plugins — not offered here.
        "os_support": {"linux": True, "darwin": False, "windows": False},
        "variants": {
            "cuda": {},
            "rocm": {
                "extra_index_url": "https://wheels.vllm.ai/rocm/",
                "python_note": "ROCm prebuilt wheels are Python 3.12 only",
            },
        },
        "requires": "Linux (Windows: use WSL); NVIDIA CUDA or AMD ROCm GPU; "
                    "large download (torch + GPU wheels)",
        "notes": "High-throughput OpenAI-compatible server. Installed in a "
                 "private venv under SABLE_HOME/engines/vllm; the wheel "
                 "variant (CUDA vs ROCm) is picked from the detected GPU.",
    },
    {
        "id": "airllm",
        "name": "AirLLM",
        "homepage": "https://github.com/lyogavin/airllm",
        "kind": "other",
        "method": "pip",
        "packages": ["airllm"],
        "binary_name": "python",  # library: run via the venv interpreter
        "args_template": [],
        # Verified against the airllm README (2026-10): plain `pip install
        # airllm`; Windows documented as same-install-as-linux; the
        # quantization stack (bitsandbytes, flash-attn) is CUDA-centric.
        "os_support": {"linux": True, "darwin": True, "windows": True},
        "requires": "Python 3.x; NVIDIA CUDA GPU for real use "
                    "(bitsandbytes/flash-attn); runs huge models on low VRAM",
        "notes": "Memory-efficient inference library (4-bit loading of "
                 "100B+ models). A Python library, not an OpenAI server — "
                 "inventory-only entry; run it from the venv "
                 "(SABLE_HOME/engines/airllm/venv/bin/python).",
    },
    {
        "id": "strata",
        "name": "Strata",
        "homepage": "https://github.com/niko1221/strata",
        # Strata's serve/server.py exposes an OpenAI-compatible endpoint
        # (/v1/chat/completions, /v1/models), so sable CAN spawn it — but it
        # launches from its own run-config JSON (written by strata setup), NOT
        # a GGUF path. The instance "model" is therefore the path to that
        # config file ({model} below); the Serve page shows a config-path
        # picker for this engine (model_kind json).
        "kind": "openai",
        "model_kind": "json",
        "method": "git_pip",
        "repo": "https://github.com/niko1221/strata",
        "ref": None,  # repo default branch
        "binary_name": "python",
        # run as `python -m serve.server` from the checkout (cwd below)
        "run_from_checkout": True,
        "args_template": ["-m", "serve.server", "--engine", "strata",
                          "--config", "{model}", "--host", "{host}",
                          "--port", "{port}"],
        "os_support": {"linux": True, "darwin": True, "windows": True},
        "requires": "research codebase; inspect README before installing; "
                    "needs a run-config.json written by strata's own setup",
        "notes": "Strata inference server (OpenAI-compatible). Spawned as "
                 "`python -m serve.server --engine strata --config <run-"
                 "config.json>` from the checkout; create instances by "
                 "pointing the model field at a run-config.json.",
    },
    {
        "id": "sd-cpp",
        "name": "stable-diffusion.cpp",
        "homepage": "https://github.com/leejet/stable-diffusion.cpp",
        # sd-server is an HTTP image-generation server (endpoints / +
        # /sdcpp/v1/*). Spawnable as kind "diffusion": sable health-checks
        # the bare root (no /v1) and appends per-instance image_settings
        # (--steps/--cfg-scale/-W/-H/...) to the command.
        "kind": "diffusion",
        "model_kind": "gguf",
        "method": "git_cmake",
        "repo": "https://github.com/leejet/stable-diffusion.cpp",
        "ref": None,
        "recursive": True,  # vendored ggml submodule
        # the build produces both CLIs; sd-server is what sable spawns,
        # sd-cli stays as a fallback for older/newer upstream layouts
        "binary_name": "sd-server",
        "fallback_binary_names": ["sd-cli"],
        "args_template": ["--model", "{model}", "--listen-ip", "{host}",
                          "--listen-port", "{port}"],
        # Verified against the repo README/docs/build.md (2026-10): platforms
        # Linux/macOS/Windows; backends CPU (AVX/AVX2/AVX512), CUDA
        # (-DSD_CUDA=ON), Vulkan (-DSD_VULKAN=ON, needs libvulkan-dev + glslc),
        # Metal (-DSD_METAL=ON), AMD via HIPBLAS (-DSD_HIPBLAS=ON with
        # GPU_TARGETS/AMDGPU_TARGETS=gfxNNNN). sable picks the backend from
        # detected hardware (same detection as llama.cpp) unless overridden.
        "os_support": {"linux": True, "darwin": True, "windows": True},
        "cmake_backends": {
            "cuda": ["-DSD_CUDA=ON"],
            "metal": ["-DSD_METAL=ON"],
            "vulkan": ["-DSD_VULKAN=ON"],
            # rocm flags are built dynamically (gfx target); see _sd_cmake_flags
            "rocm": None,
            "cpu": [],
        },
        "requires": "C/C++ compiler + cmake (same toolchain as llama.cpp); "
                    "GPU backend auto-picked (CUDA/Vulkan/Metal/HIP)",
        "notes": "Image generation (SD1/SD2/SDXL/Flux/Qwen-Image …) in C/C++. "
                 "Spawned via sd-server (loopback HTTP); image defaults "
                 "(steps/CFG/size/sampler) are per-instance image_settings, "
                 "not chat sampling. Models are GGUF files picked from the "
                 "models folder.",
    },
]


def get_recipe(recipe_id: str) -> Dict[str, Any]:
    """Catalog entry by id. Raises KeyError for unknown ids."""
    for r in CATALOG:
        if r["id"] == recipe_id:
            return r
    raise KeyError(f"unknown engine recipe {recipe_id!r}")


def catalog_ids() -> List[str]:
    return [r["id"] for r in CATALOG]


# ------------------------------------------------------- hardware awareness

def detect_host() -> Dict[str, Any]:
    """Cheap host profile for compat checks (the SAME stdlib probes the
    llama.cpp runtime detection uses). Never raises — degrades to cpu/other.
    Cached per process: GPU probing shells out to vendor CLIs and a polling
    dashboard endpoint must not re-run it every few seconds."""
    global _HOST_CACHE
    if _HOST_CACHE is not None:
        return _HOST_CACHE
    try:
        from . import runtime
        info = runtime.detect_runtime()
        host = {
            "system": info.system,
            "machine": info.machine,
            "backend": info.backend,
            "gpus": [vars(g) for g in info.gpus],
        }
    except Exception:  # pragma: no cover - defensive (probes are best-effort)
        import platform as _p
        host = {"system": _p.system().lower(), "machine": _p.machine(),
                "backend": "cpu", "gpus": []}
    _HOST_CACHE = host
    return host


_HOST_CACHE: Optional[Dict[str, Any]] = None


def amd_gfx_target(host: Dict[str, Any]) -> Optional[str]:
    """gfxNNNN target from detected AMD GPU names (HIPBLAS builds need it);
    None when no vendor name parses as a gfx id."""
    import re as _re
    for g in host.get("gpus") or []:
        m = _re.search(r"gfx[0-9a-fA-F]+(?:-[a-z0-9]+)*",
                       str(g.get("name") or ""))
        if m:
            return m.group(0)
    return None


def recipe_compat(recipe_id_or_recipe: Any,
                  host: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Hardware/OS verdict for one recipe on this (or the given) host.

    Returns {ok, blocked, warnings, variant, backend}. ``blocked`` is a hard
    stop (e.g. vLLM on Windows — upstream does not support it); ``warnings``
    are advisory (no GPU detected, python != 3.12 for ROCm wheels…). Pure
    function of recipe + host dict; detect_host() only runs when the caller
    does not pass one."""
    recipe = (get_recipe(recipe_id_or_recipe)
              if isinstance(recipe_id_or_recipe, str) else recipe_id_or_recipe)
    host = host or detect_host()
    system = str(host.get("system") or "").lower()
    backend = str(host.get("backend") or "cpu").lower()
    gpus = host.get("gpus") or []
    vendors = {str(g.get("vendor") or "") for g in gpus}
    out: Dict[str, Any] = {"ok": True, "blocked": None, "warnings": [],
                           "variant": None, "backend": None}

    os_map = recipe.get("os_support") or {}
    if os_map and os_map.get(system) is False:
        extra = (" (Windows: use WSL — upstream does not support it "
                 "natively)" if system == "windows" else "")
        out["ok"] = False
        out["blocked"] = f"{recipe['name']} does not support {system}{extra}"
        return out

    rid = recipe["id"]
    if rid == "vllm":
        # Wheel variant follows the detected vendor (docs.vllm.ai: NVIDIA via
        # PyPI default, AMD via wheels.vllm.ai/rocm).
        if "nvidia" in vendors:
            out["variant"], out["backend"] = "cuda", "cuda"
        elif "amd" in vendors:
            out["variant"], out["backend"] = "rocm", "rocm"
            note = (recipe.get("variants", {}).get("rocm", {})
                    .get("python_note"))
            if note and sys.version_info[:2] != (3, 12):
                out["warnings"].append(
                    f"{note}; the install venv would use "
                    f"{sys.version_info[0]}.{sys.version_info[1]} — pip may "
                    "fall back to building from source (slow) or fail")
        else:
            out["variant"], out["backend"] = "cuda", "cpu"
            out["warnings"].append(
                "no NVIDIA/AMD GPU detected — the default CUDA wheel will "
                "install but vLLM has no usable backend on this host")
    elif rid == "airllm":
        if "nvidia" not in vendors:
            out["warnings"].append(
                "AirLLM's quantization stack (bitsandbytes/flash-attn) is "
                "CUDA-centric; without an NVIDIA GPU it installs but has "
                "little practical use")
    elif rid == "sd-cpp":
        be = backend if backend in (recipe.get("cmake_backends") or {}) else "cpu"
        if be == "rocm" and not amd_gfx_target(host):
            out["warnings"].append(
                "AMD GPU without a parsable gfx target — falling back to the "
                "CPU build (backend=vulkan uses the GPU if libvulkan-dev and "
                "glslc are installed)")
            be = "cpu"
        out["backend"] = be
    return out


def _sd_cmake_flags(backend: str, host: Dict[str, Any]) -> List[str]:
    """cmake flags for stable-diffusion.cpp on this host (flag names verified
    against docs/build.md: SD_CUDA / SD_METAL / SD_VULKAN / SD_HIPBLAS +
    GPU_TARGETS/AMDGPU_TARGETS)."""
    if backend == "rocm":
        gfx = amd_gfx_target(host)
        if not gfx:
            return []  # CPU build
        return ["-DSD_HIPBLAS=ON", f"-DGPU_TARGETS={gfx}",
                f"-DAMDGPU_TARGETS={gfx}"]
    recipe = get_recipe("sd-cpp")
    flags = (recipe.get("cmake_backends") or {}).get(backend)
    return list(flags) if flags else []


# -------------------------------------------------------------------- paths

def engines_root(home: Path) -> Path:
    return Path(home) / "engines"


def engine_dir(home: Path, recipe_id: str) -> Path:
    get_recipe(recipe_id)  # validates id
    return engines_root(home) / recipe_id


def venv_dir(home: Path, recipe_id: str) -> Path:
    return engine_dir(home, recipe_id) / "venv"


def src_dir(home: Path, recipe_id: str) -> Path:
    return engine_dir(home, recipe_id) / "src"


def state_path(home: Path, recipe_id: str) -> Path:
    get_recipe(recipe_id)
    return (Path(home) / "serving" / "engine_installs"
            / f"{recipe_id}.json")


def read_state(home: Path, recipe_id: str) -> Optional[Dict[str, Any]]:
    p = state_path(home, recipe_id)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def write_state(home: Path, recipe_id: str, state: Dict[str, Any]) -> None:
    p = state_path(home, recipe_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    os.replace(tmp, p)


def _utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _exe(name: str) -> str:
    return name + (".exe" if os.name == "nt" else "")


def venv_python(home: Path, recipe_id: str) -> Path:
    v = venv_dir(home, recipe_id)
    bindir = "Scripts" if os.name == "nt" else "bin"
    return v / bindir / _exe("python")


def _find_built_binary(root: Path, binary_name: str) -> Optional[Path]:
    """cmake multi-config safe lookup for a freshly built binary."""
    cand_dirs = (root / "build" / "bin", root / "build" / "bin" / "Release",
                 root / "bin")
    for d in cand_dirs:
        p = d / _exe(binary_name)
        if p.is_file() and os.access(p, os.X_OK):
            return p
    return None


# ------------------------------------------------------------------ status

def recipe_installed(home: Path, recipe_id: str) -> bool:
    """State file exists AND its recorded binary is still on disk."""
    st = read_state(home, recipe_id)
    if not st:
        return False
    b = str(st.get("binary") or "")
    return bool(b) and Path(b).is_file()


def list_catalog(home: Path) -> List[Dict[str, Any]]:
    """Catalog rows for the dashboard: recipe + install status. Pure stat()
    probes — never runs subprocesses (a listing endpoint must stay cheap)."""
    rows = []
    for r in CATALOG:
        st = read_state(home, r["id"])
        rows.append({
            **{k: r.get(k) for k in ("id", "name", "homepage", "kind",
                                     "method", "requires", "notes")},
            "installed": recipe_installed(home, r["id"]),
            "install_state": st,
        })
    return rows


# ------------------------------------------------------------- subprocess

def _stream(cmd: List[str], log: Callable[[str], None],
            cwd: Optional[Path] = None) -> int:
    """Run a long command streaming combined output through log(). rc back."""
    logger.debug("engine install exec: %s", " ".join(cmd))
    try:
        proc = subprocess.Popen(cmd, cwd=str(cwd) if cwd else None,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                bufsize=1)
    except OSError as e:
        raise InstallError(f"could not run {cmd[0]!r}: {e}") from e
    assert proc.stdout is not None
    for line in proc.stdout:
        log(line.rstrip())
    return proc.wait()


def _check(cmd: List[str], log: Callable[[str], None], what: str,
           cwd: Optional[Path] = None) -> None:
    rc = _stream(cmd, log, cwd=cwd)
    if rc != 0:
        raise InstallError(f"{what} failed (rc={rc}) — see install log")


def _require(tool: str, hint: str) -> str:
    import shutil
    found = shutil.which(tool)
    if not found:
        raise InstallError(f"{tool!r} is required: {hint}")
    return found


# ----------------------------------------------------------------- steps

def _make_venv(home: Path, rid: str, log: Callable[[str], None]) -> Path:
    py = venv_python(home, rid)
    if py.is_file():
        log(f"reusing existing venv at {venv_dir(home, rid)}")
        return py
    venv_dir(home, rid).parent.mkdir(parents=True, exist_ok=True)
    log(f"creating private venv at {venv_dir(home, rid)} …")
    _check([sys.executable, "-m", "venv", str(venv_dir(home, rid))], log,
           "python -m venv")
    if not py.is_file():  # pragma: no cover - exotic venv layouts
        raise InstallError(f"venv created but {py} missing")
    return py


def _pip_install(packages: List[str], log: Callable[[str], None],
                 py: Path, cwd: Optional[Path] = None,
                 editable: bool = False,
                 extra_index_url: Optional[str] = None,
                 requirements: Optional[List[Any]] = None) -> None:
    """pip install into the private venv. ``packages`` are positional specs;
    ``requirements`` are requirement FILES and must go through ``-r`` (a bare
    path makes pip fail with "Invalid requirement: … Expected package name")."""
    log("upgrading pip … (best effort)")
    _stream([str(py), "-m", "pip", "install", "--upgrade", "pip"], log,
            cwd=cwd)
    cmd = [str(py), "-m", "pip", "install"]
    if extra_index_url:
        cmd += ["--extra-index-url", extra_index_url]
    for rf in requirements or []:
        cmd += ["-r", str(rf)]
    if editable:
        cmd.append("-e")
    cmd += packages
    shown = " ".join(["-r " + str(r) for r in requirements or []]
                     + (["-e"] if editable else []) + list(packages))
    log(f"pip install {shown} … (this can download many GB)")
    _check(cmd, log, "pip install", cwd=cwd)


def _git_clone(repo: str, ref: Optional[str], dest: Path, recursive: bool,
               log: Callable[[str], None]) -> None:
    git = _require("git", "Install git (apt/brew/winget).")
    dest.parent.mkdir(parents=True, exist_ok=True)
    # --depth 1 + --branch works for branches; tags/some refs fall back to a
    # full clone (same trick as runtime.setup_llama_cpp).
    shallow = [git, "clone"] + (["--recursive"] if recursive else []) \
        + ["--depth", "1"] + (["--branch", ref] if ref else []) \
        + [repo, str(dest)]
    log(f"cloning {repo}" + (f" @ {ref}" if ref else "") + f" into {dest} …")
    rc = _stream(shallow, log)
    if rc != 0:
        log("shallow clone failed; retrying full clone …")
        full = [git, "clone"] + (["--recursive"] if recursive else []) \
            + [repo, str(dest)]
        if ref:
            full += ["--branch", ref]
        _check(full, log, "git clone")
    if recursive:
        _stream([git, "-C", str(dest), "submodule", "update", "--init",
                 "--recursive"], log)


def _git_update(ref: Optional[str], dest: Path, recursive: bool,
                log: Callable[[str], None]) -> None:
    git = _require("git", "Install git (apt/brew/winget).")
    if ref:
        target = f"origin/{ref}"
    else:
        branch = _stream_out([git, "-C", str(dest), "rev-parse",
                              "--abbrev-ref", "HEAD"])
        target = f"origin/{branch}" if branch and branch != "HEAD" else None
    log(f"fetching + resetting {dest} to {target or 'current HEAD'}")
    _check([git, "-C", str(dest), "fetch", "--all", "--tags", "--force"],
           log, "git fetch")
    if ref:
        _stream([git, "-C", str(dest), "checkout", str(ref)], log)
    if target:
        _check([git, "-C", str(dest), "reset", "--hard", target], log,
               "git reset")
    if recursive:
        _stream([git, "-C", str(dest), "submodule", "update", "--init",
                 "--recursive", "--force"], log)


def _is_pip_installable(sdir: Path) -> bool:
    """Can this checkout be ``pip install``-ed at all?

    Repos that ship an *application* setup.py (an argparse CLI script with no
    setuptools call — strata is one) make pip die with
    ``setup.py: error: unrecognized arguments: egg_info``, because pip's
    metadata hook runs ``setup.py egg_info`` and the script's parser rejects
    it. Detect that here and install requirements-only instead of failing."""
    if (sdir / "pyproject.toml").is_file():
        return True
    try:
        cfg = sdir / "setup.cfg"
        if cfg.is_file() and "[metadata]" in cfg.read_text(
                encoding="utf-8", errors="replace"):
            return True
        sp = sdir / "setup.py"
        if sp.is_file():
            txt = sp.read_text(encoding="utf-8", errors="replace")
            return "setuptools" in txt or "distutils" in txt
    except OSError:  # pragma: no cover - unreadable file => treat as not
        return False
    return False


def _stream_out(cmd: List[str]) -> Optional[str]:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return (out.stdout or "").strip() if out.returncode == 0 else None
    except Exception:
        return None


def _git_commit(dest: Path) -> Optional[str]:
    import shutil
    git = shutil.which("git")
    if not git:
        return None
    return _stream_out([git, "-C", str(dest), "rev-parse", "--short", "HEAD"])


def _cmake_build(root: Path, log: Callable[[str], None],
                 extra_flags: Optional[List[str]] = None) -> None:
    cmake = _require("cmake", "Install cmake (>=3.18): apt/brew/winget or "
                              "pip install cmake.")
    jobs = str(max(1, min(8, os.cpu_count() or 4)))
    cfg = [cmake, "-S", str(root), "-B", str(root / "build"),
           "-DCMAKE_BUILD_TYPE=Release"] + list(extra_flags or [])
    _check(cfg, log, "cmake configure")
    log(f"building (-j{jobs}) — this can take a while …")
    _check([cmake, "--build", str(root / "build"), "--config", "Release",
            "-j", jobs], log, "cmake build")


# ---------------------------------------------------------------- install

def install_recipe(recipe_id: str, *, home: Path, ref: Optional[str] = None,
                   update: bool = False, backend: Optional[str] = None,
                   force: bool = False,
                   log_fn: Optional[Callable[[str], None]] = None,
                   ) -> Dict[str, Any]:
    """Install (or update) one catalog engine. Long-running — call from a
    background thread and stream progress via ``log_fn``. Writes the install
    state file on success and returns it. Raises InstallError with actionable
    messages; never touches anything outside <SABLE_HOME>/engines/<id>/.

    Hardware/OS awareness: the host profile (same probes as llama.cpp runtime
    detection) selects the vLLM wheel variant (CUDA vs ROCm index) and the
    stable-diffusion.cpp cmake backend. ``backend`` overrides the auto-pick;
    ``force`` installs even when recipe_compat() reports a hard OS block."""
    log = log_fn or (lambda line: logger.info("%s", line))
    recipe = get_recipe(recipe_id)  # KeyError for unknown ids
    home = Path(home)
    method = recipe["method"]

    host = detect_host()
    compat = recipe_compat(recipe, host)
    if backend:
        compat["backend"] = backend
    for w in compat.get("warnings") or []:
        log(f"warning: {w}")
    if not compat["ok"] and not force:
        raise InstallError(
            f"{compat['blocked']} — pass force to install anyway "
            "(upstream will likely fail to build/run)")

    prefix: Optional[Path] = None
    binary: Optional[str] = None
    commit: Optional[str] = None
    variant: Optional[str] = None
    used_ref = ref or recipe.get("ref")

    if method in ("pip", "git_pip"):
        py = _make_venv(home, recipe_id, log)
        prefix = venv_dir(home, recipe_id)
        if method == "git_pip":
            sdir = src_dir(home, recipe_id)
            if not (sdir / ".git").is_dir():
                _git_clone(recipe["repo"], used_ref, sdir, False, log)
            elif update:
                _git_update(used_ref, sdir, False, log)
            else:
                log(f"reusing existing checkout at {sdir}")
            req = sdir / str(recipe.get("requirements") or "requirements.txt")
            if req.is_file():
                _pip_install([], log, py, cwd=sdir, requirements=[req])
            if _is_pip_installable(sdir):
                # Buffer the pip output so an egg_info-style failure (app-
                # style setup.py that predates detection, or a pyproject that
                # still routes metadata through argparse setup.py) can be
                # recognised and downgraded instead of killing the install.
                tail: List[str] = []

                def _pip_log(line: str, _log=log) -> None:
                    tail.append(line)
                    del tail[:-120]
                    _log(line)

                try:
                    _pip_install(["."], _pip_log, py, cwd=sdir, editable=True)
                    variant = "editable"
                except InstallError:
                    joined_tail = "\n".join(tail)
                    if ("egg_info" not in joined_tail
                            and "unrecognized arguments" not in joined_tail):
                        raise
                    log("setup.py rejects pip's metadata hook (egg_info) — "
                        "this repo ships an application script, not a "
                        "pip-installable package; keeping checkout + deps")
                    variant = "checkout"
            else:
                # Application-style repo (argparse setup.py, no setuptools
                # entry point): `pip install -e .` dies on `egg_info`. Keep the
                # checkout + its dependencies — that is all an inventory-only
                # engine needs; run it via the venv python in this dir.
                log("no pip-installable package metadata "
                    "(setup.py without setuptools / no pyproject.toml) — "
                    "skipping `pip install -e .`; the checkout and its "
                    f"requirements are installed under {sdir}")
                variant = "checkout"
            commit = _git_commit(sdir)
        else:
            # Hardware-aware wheel variant (vLLM: ROCm index on AMD GPUs).
            variant = compat.get("variant")
            vconf = (recipe.get("variants") or {}).get(variant or "") or {}
            eix = vconf.get("extra_index_url")
            if eix:
                log(f"using {variant} wheel index {eix}")
            _pip_install(list(recipe["packages"]), log, py,
                         extra_index_url=eix)
        binary = str(venv_python(home, recipe_id))
        if recipe.get("run_from_checkout"):
            prefix = src_dir(home, recipe_id)

    elif method == "git_cmake":
        sdir = src_dir(home, recipe_id)
        recursive = bool(recipe.get("recursive"))
        if not (sdir / ".git").is_dir():
            _git_clone(recipe["repo"], used_ref, sdir, recursive, log)
        elif update:
            _git_update(used_ref, sdir, recursive, log)
        else:
            log(f"reusing existing checkout at {sdir}")
        be = compat.get("backend") or "cpu"
        flags = _sd_cmake_flags(be, host) if recipe_id == "sd-cpp" else []
        log(f"cmake backend: {be} {' '.join(flags) if flags else '(CPU default)'}")
        _cmake_build(sdir, log, extra_flags=flags)
        found = _find_built_binary(sdir, recipe["binary_name"])
        for fb in recipe.get("fallback_binary_names") or []:
            if found is not None:
                break
            found = _find_built_binary(sdir, fb)
        if found is None:
            raise InstallError(
                f"build finished but no {recipe['binary_name']!r} under "
                f"{sdir / 'build' / 'bin'}")
        prefix, binary = sdir, str(found)
        commit = _git_commit(sdir)

    else:  # pragma: no cover - catalog is static data
        raise InstallError(f"unknown install method {method!r}")

    state = {
        "recipe": recipe_id,
        "method": method,
        "ref": used_ref,
        "commit": commit,
        "variant": variant,
        "backend": compat.get("backend"),
        "prefix": str(prefix) if prefix else None,
        "binary": binary,
        "installed_at": _utcnow(),
    }
    write_state(home, recipe_id, state)
    log(f"installed {recipe['name']}: binary at {binary}")
    return state


# ------------------------------------------------- engine registration glue

def engine_body_for(recipe_id: str, state: Dict[str, Any],
                    home: Optional[Path] = None) -> Dict[str, Any]:
    """Engine-registry body for a completed install (re-register / update).

    ``home`` lets checkout-run engines resolve their cwd from the canonical
    install layout (<SABLE_HOME>/engines/<id>/src) instead of trusting the
    recorded prefix: state files written before a recipe gained
    ``run_from_checkout`` carry the *venv* dir as prefix, and spawning
    ``python -m <pkg>`` from there dies with ModuleNotFoundError (strata's
    serve/server.py lives in the checkout, not the venv)."""
    recipe = get_recipe(recipe_id)
    body = {
        "name": recipe["name"],
        "kind": recipe["kind"],
        "binary": str(state.get("binary") or ""),
        "args_template": list(recipe.get("args_template") or []),
        "notes": (recipe.get("notes") or "")
                 + f" [installed by sable {state.get('installed_at') or ''}]"
                   .strip(),
    }
    cwd = ""
    if recipe.get("run_from_checkout"):
        if home is not None:
            src = src_dir(Path(home), recipe_id)
            if (src / ".git").is_dir():
                cwd = str(src)
        if not cwd:  # no checkout on disk to re-derive from => recorded value
            cwd = str(state.get("prefix") or "")
    if cwd:
        body["cwd"] = cwd
    mk = str(recipe.get("model_kind") or "").strip().lower()
    if mk:
        body["model_kind"] = mk
    return body


def sync_installed_recipes(home: Path) -> List[str]:
    """Re-derive the registry bodies of catalog-managed engines (id == recipe
    id) so recipe upgrades reach existing installs — e.g. stable-diffusion.cpp
    moving from inventory-only ``other`` to spawnable ``diffusion`` (with its
    sd-server binary), or strata gaining its serve args. Returns refreshed
    engine ids. Only touches engines whose id matches a recipe AND whose
    binary still matches the recorded install state (a hand-registered engine
    that merely shares the name is left alone). Cheap: stat() + two JSON reads
    per recipe, and after the first refresh nothing re-writes."""
    from . import engines as engines_mod

    refreshed: List[str] = []
    for r in CATALOG:
        rid = str(r["id"])
        st = read_state(home, rid)
        if not st or not recipe_installed(home, rid):
            continue
        eng = engines_mod.get_engine(home, rid)
        if eng is None or eng.get("builtin"):
            continue
        body = engine_body_for(rid, st, home=home)
        want_kind = str(r.get("kind") or "")
        want_bin_name = str(r.get("binary_name") or "")
        if want_kind == "diffusion" and body["binary"] and \
                Path(body["binary"]).name != want_bin_name:
            # Installs recorded before the server binary existed (sd-cli
            # only): promote to the server sitting next to it; when there is
            # none, leave the entry inventory-only rather than promoting a
            # CLI that cannot answer the healthcheck.
            sib = Path(body["binary"]).parent / _exe(want_bin_name)
            if sib.is_file():
                body["binary"] = str(sib)
            else:
                continue
        # Only touch records this install actually owns: the recorded state
        # binary or its promoted sibling. A hand-registered engine that just
        # happens to share the recipe id keeps its own binary.
        cur_bin = str(eng.get("binary") or "")
        if cur_bin not in (str(st.get("binary") or ""), body["binary"]):
            continue
        if str(eng.get("kind") or "") == want_kind and \
                str(eng.get("binary") or "") == body["binary"] and \
                list(eng.get("args_template") or []) == body["args_template"] \
                and str(eng.get("cwd") or "") == str(body.get("cwd") or ""):
            continue  # already current
        merged = dict(eng)
        user_notes = str(eng.get("notes") or "").strip()
        merged.update(body)
        if user_notes and "[installed by sable" not in user_notes:
            merged["notes"] = user_notes  # keep genuine user edits
        merged["id"] = rid
        try:
            new_eng = engines_mod.validate_engine_body(merged, existing=eng)
        except Exception:  # defensive: a broken record must not break listing
            continue
        new_eng["id"] = rid
        engines_mod.save_engine(home, new_eng)
        refreshed.append(rid)
    return refreshed

"""Serving Supervisor v0 — spawn/health-check/stop a llama-server GGUF process.

Design (docs/DECISIONS.md D-4): sable owns one llama-server process bound to
loopback on a dynamic port, no auth. Alternative: attach mode points at an
already-running OpenAI-compatible server without
spawning anything.

Safety rules enforced here:
- Bind address is always 127.0.0.1 (loopback-only).
- Port 0 means "pick a free port ourselves" — we never assume :8888 is ours;
  an external server may own it. We never kill
  processes we did not spawn.
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import os
import shutil
import signal
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from ..core.config import SableConfig
from . import runtime

logger = logging.getLogger("sable.serving")


class ServeError(RuntimeError):
    pass


def find_llama_server(
    explicit: Optional[str] = None,
    *,
    auto_setup: bool = True,
    log_fn=None,
    cfg: Optional[SableConfig] = None,
) -> str:
    """Locate a llama-server binary: explicit path > PATH > common locations.

    When nothing is found and ``auto_setup`` is on, sable builds llama.cpp
    itself (clone + cmake for the detected GPU backend) into
    ``<SABLE_HOME>/llama.cpp`` — so a fresh machine with git/cmake/compiler
    gets a working server without manual steps. Set SABLE_NO_AUTO_SETUP=1 or
    config ``llama_auto_setup: false`` to keep discovery read-only.
    """
    if explicit:
        p = Path(explicit).expanduser()
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
        raise ServeError(f"llama-server not found at {p}")
    found = shutil.which("llama-server")
    if found:
        return found
    for d in runtime.search_dirs():
        cand = d / ("llama-server.exe" if os.name == "nt" else "llama-server")
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)

    if not auto_setup:
        raise ServeError(
            "Could not find llama-server on PATH or in common locations "
            f"({', '.join(str(d) for d in runtime.search_dirs())}). "
            "Install llama.cpp, run `sable setup`, set SABLE_LLAMA_SERVER, "
            "or use --attach mode."
        )

    # Nothing found: build llama.cpp ourselves for this machine's hardware.
    log = log_fn or (lambda line: logger.info("%s", line))
    if runtime.setup_in_progress():
        raise ServeError("llama.cpp setup already running in this process; retry shortly")
    with runtime._setup_lock:
        info = runtime.detect_runtime()
        backend = (getattr(cfg, "llama_backend", None) or "").strip() or info.backend
        log(
            f"no llama-server found — auto-setting up llama.cpp "
            f"(backend={backend}, gpus={len(info.gpus)})"
        )
        try:
            runtime.setup_llama_cpp(
                backend=backend,
                log_fn=log,
                repo=getattr(cfg, "llama_repo", None),
                ref=getattr(cfg, "llama_ref", None),
            )
        except runtime.SetupError as e:
            raise ServeError(f"llama.cpp auto-setup failed: {e}") from e
    found = shutil.which("llama-server")
    if found:
        return found
    for d in runtime.search_dirs():
        cand = d / ("llama-server.exe" if os.name == "nt" else "llama-server")
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
    raise ServeError("llama.cpp setup finished but llama-server is still not runnable")


def _free_port(host: str = "127.0.0.1") -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def _auth_headers(api_key: Optional[str]) -> dict:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def healthcheck(base_url: str, timeout: float = 5.0, api_key: Optional[str] = None) -> bool:
    """True when GET {base_url}/models answers 200 (OpenAI-compatible).

    `api_key` adds a bearer token for servers that require auth (e.g. an
    external external OpenAI-compatible server); supervisor-owned llama-server needs none.
    """
    url = base_url.rstrip("/") + "/models"
    try:
        r = httpx.get(url, timeout=timeout, headers=_auth_headers(api_key))
        return r.status_code == 200
    except Exception:
        return False


@dataclass
class ServerHandle:
    """A running (or attached) OpenAI-compatible endpoint."""

    base_url: str  # e.g. http://127.0.0.1:43210/v1
    model_name: str
    owned: bool  # False for attach mode: never kill/stop
    process: Optional[subprocess.Popen] = field(default=None, repr=False)
    api_key: Optional[str] = field(default=None, repr=False)  # attach-mode bearer

    def stop(self, grace: float = 5.0) -> None:
        """Terminate the spawned child process (attach-mode servers untouched)."""
        if not self.owned or self.process is None:
            return
        if self.process.poll() is not None:
            return
        logger.info("Stopping llama-server (pid=%s)…", self.process.pid)
        try:
            self.process.send_signal(signal.SIGTERM)
            self.process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            logger.warning("llama-server did not exit; sending SIGKILL")
            self.process.kill()
            with contextlib.suppress(Exception):
                self.process.wait(timeout=grace)
        except Exception:
            with contextlib.suppress(Exception):
                self.process.kill()


class ServingSupervisor:
    """Owns the llama-server lifecycle for one active GGUF (Phase 1, single model)."""

    def __init__(self, cfg: SableConfig):
        self.cfg = cfg
        self._handle: Optional[ServerHandle] = None
        self._stopping = threading.Event()

    # ------------------------------------------------------------------ attach
    def attach(self) -> ServerHandle:
        """Point at an already-running OpenAI-compatible server; spawn nothing."""
        base = self.cfg.effective_attach_url
        if not base:
            raise ServeError("attach mode requested but no attach URL configured")
        logger.info("Attaching to existing OpenAI-compatible server at %s", base)
        model = (
            self._discover_model(base, self.cfg.attach_api_key)
            or self.cfg.model_name
            or "attached-model"
        )
        handle = ServerHandle(
            base_url=base, model_name=model, owned=False, api_key=self.cfg.attach_api_key
        )
        self._handle = handle
        return handle

    def _discover_model(
        self, base_url: str, api_key: Optional[str] = None
    ) -> Optional[str]:
        try:
            r = httpx.get(
                base_url.rstrip("/") + "/models",
                timeout=10.0,
                headers=_auth_headers(api_key),
            )
            if r.status_code == 200:
                data = r.json().get("data") or []
                if data and isinstance(data[0], dict):
                    return data[0].get("id")
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------ spawn
    def start(self, ready_timeout: float = 300.0,
              log_path: Optional[Path] = None) -> ServerHandle:
        """Spawn llama-server for the configured GGUF and wait until healthy.

        Consolidation (review #11): this legacy single-server entry point is a
        thin adapter over the canonical Phase-6 instance machinery — the
        config maps onto the reserved "default" instance and spawning goes
        through ``start_instance_server``. The owned child therefore lands in
        the shared ownership registry (visible to ``stop_all_servers()`` /
        at-exit cleanup) and inherits the crash-loop budget (#28), so a
        flapping GGUF stops respawning after CRASH_LOOP_MAX_FAILURES failed
        cycles instead of hammering forever. Explicit user starts (`sable
        serve`) clear the refusal before calling in, as before."""
        if self._handle is not None:
            return self._handle

        cfg = self.cfg
        if not cfg.gguf_path:
            raise ServeError(
                "No GGUF configured. Pass --gguf /path/to/model.gguf, set "
                "SABLE_GGUF, or use --attach <url> for an existing server."
            )

        if log_path is None:
            # Same log location the dashboard's default instance uses, so CLI
            # and dashboard runs of the default server share one log file.
            from ..store.paths import resolve_sable_home

            log_path = Path(resolve_sable_home(cfg)) / "logs" / "server-default.log"

        from . import instances as srv_inst  # lazy: instances imports us at top level

        inst = srv_inst.default_instance(cfg)
        if cfg.model_name:
            inst["model_name"] = cfg.model_name
        handle = start_instance_server(
            cfg, inst, log_path=log_path, ready_timeout=ready_timeout,
            cancel=self._stopping)
        self._handle = handle
        logger.info("llama-server healthy at %s (model=%s)",
                    handle.base_url, handle.model_name)
        return handle

    # ------------------------------------------------------------------ misc
    @property
    def handle(self) -> Optional[ServerHandle]:
        return self._handle

    def stop(self) -> None:
        self._stopping.set()
        handle, self._handle = self._handle, None
        if handle is None:
            return
        if handle.owned:
            # Drop the shared-registry entry too (stop_instance pops + stops;
            # a no-op when stop_all_servers already removed it).
            stop_instance("default")
        handle.stop()  # idempotent; attach mode never kills anything

    def __enter__(self) -> "ServingSupervisor":
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


# ===========================================================================
# Phase 6: multi-instance serving — N named instances in ONE process.
#
# This section is the CANONICAL spawn machinery: the dashboard runs many
# instances side by side through it, and the legacy `ServingSupervisor.start`
# (CLI `serve` / shadow eval) now routes through it as well (review #11), so
# every owned child — however it was started — shares one ownership registry
# and one crash-loop budget, and ONE shutdown entry point (`stop_all_servers`)
# reaches all of them while also folding in the Phase 5 embedding servers —
# the embed server is just another managed serving instance now.
# ===========================================================================

#: Instance handles THIS process spawned, keyed by instance id. Only entries
#: here are ever terminated in-process; attach-URL instances never appear
#: (they own nothing), and foreign pids are never killed (see instances.py).
_owned_handles: Dict[str, ServerHandle] = {}
_owned_lock = threading.Lock()
_multi_atexit_registered = False


# ---------------------------------------------------------------------------
# Crash-loop budget (inspiration pass 4 #28)
#
# A GGUF that OOMs or a broken binary turns into an endless respawn cycle:
# start -> healthcheck fails -> stop -> autostart tries again. CRASH_LOOP_-
# MAX_FAILURES failed start/healthcheck cycles inside CRASH_LOOP_WINDOW_S put
# the instance in a refused `crash_loop` state: further AUTO starts (the
# dashboard autostart thread) raise instead of respawning. An explicit
# user-initiated start (dashboard Start button, `sable serve`) calls
# clear_crash_loop() and retries normally. Only failed spawn/healthcheck
# cycles count; a server that goes healthy resets its history.
# ---------------------------------------------------------------------------

CRASH_LOOP_MAX_FAILURES = 3
CRASH_LOOP_WINDOW_S = 600.0  # 10 minutes

_crash_lock = threading.Lock()
#: instance_key -> list of (monotonic, wall_epoch) failure stamps
_crash_failures: Dict[str, List[tuple]] = {}


def record_start_failure(instance_key: str) -> None:
    """Note one failed start/healthcheck cycle for the budget."""
    with _crash_lock:
        stamps = _crash_failures.setdefault(instance_key, [])
        stamps.append((time.monotonic(), time.time()))
        # keep only the window
        cutoff = time.monotonic() - CRASH_LOOP_WINDOW_S
        _crash_failures[instance_key] = [s for s in stamps if s[0] >= cutoff]


def crash_loop_refused(instance_key: str) -> bool:
    """True when the instance is over budget and must not auto-respawn."""
    with _crash_lock:
        stamps = _crash_failures.get(instance_key) or []
        cutoff = time.monotonic() - CRASH_LOOP_WINDOW_S
        return sum(1 for s in stamps if s[0] >= cutoff) >= CRASH_LOOP_MAX_FAILURES


def clear_crash_loop(instance_key: str) -> bool:
    """Explicit user action: drop the failure history. True when it was open."""
    with _crash_lock:
        had = len(_crash_failures.get(instance_key) or []) > 0
        _crash_failures.pop(instance_key, None)
        return had


def crash_loop_status(instance_key: str) -> Optional[Dict[str, Any]]:
    """Budget view for status surfaces; None when nothing recorded."""
    with _crash_lock:
        stamps = [s for s in (_crash_failures.get(instance_key) or [])
                  if s[0] >= time.monotonic() - CRASH_LOOP_WINDOW_S]
        if not stamps:
            return None
        return {
            "failures": len(stamps),
            "refused": len(stamps) >= CRASH_LOOP_MAX_FAILURES,
            "window_s": CRASH_LOOP_WINDOW_S,
            "max_failures": CRASH_LOOP_MAX_FAILURES,
            "last_failure": stamps[-1][1],
        }


def _register_handle(instance_id: str, handle: ServerHandle) -> None:
    global _multi_atexit_registered
    with _owned_lock:
        _owned_handles[instance_id] = handle
        if not _multi_atexit_registered:
            atexit.register(stop_all_servers)
            _multi_atexit_registered = True


def owned_instance_ids() -> List[str]:
    """Ids of the server processes THIS process spawned (still running)."""
    with _owned_lock:
        return list(_owned_handles)


def stop_all_servers() -> None:
    """Stop every serving process this process owns — instances + embed servers.

    Phase 6 shutdown entry point: replaces scattered cleanup calls. Only ever
    terminates children we spawned ourselves (in-process handles); attach-mode
    endpoints and foreign processes are never touched. The Phase 5 embedding
    server registry is folded in here so one call stops everything sable
    manages (lazy import avoids a supervisor<->indexer cycle).
    """
    with _owned_lock:
        items = list(_owned_handles.items())
    for iid, handle in items:
        try:
            handle.stop()
        except Exception:
            pass
        with _owned_lock:
            _owned_handles.pop(iid, None)
    try:
        from ..core.indexer import stop_all_embed_servers
        stop_all_embed_servers()
    except Exception:
        pass


def require_loopback(url: str) -> str:
    """Validate an attach URL is loopback-only (project constraint).

    Phase 6 enforces this for EVERY attach target (the spawn path already
    binds 127.0.0.1 unconditionally): hostname must be 127.0.0.1, localhost
    or ::1. Returns the normalized URL; raises ServeError otherwise."""
    from urllib.parse import urlparse

    u = urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise ServeError(f"attach URL must be http(s) with a host: {url!r}")
    if u.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise ServeError(
            "attach URLs must be loopback-only (127.0.0.1/localhost) — "
            f"refusing {url!r}"
        )
    return url


def build_instance_cmd(binary: str, cfg, inst: dict, *, port: int,
                       model_name: str, rt=None) -> List[str]:
    """llama-server argv for one instance record.

    Named instances run with THEIR OWN run_settings only — the global
    `run_settings:` section is the *default server's* configuration and is
    never inherited by new instances (per-instance settings are captured on
    the Serve page when the instance is created). The reserved "default"
    instance keeps inheriting the global section, mirroring the legacy
    single-server path. Context/gpu offload still fall back to cfg defaults;
    precedence chain inside the command is documented in run_args.py."""
    from .run_args import run_settings_args

    rt = rt or runtime.detect_runtime()
    # Same shard/blob safety as the spawn wrapper: point at shard 1 through its
    # visible snapshot name (a hash-named blob breaks split-GGUF loading).
    from ..core.model_discovery import resolve_gguf_shards

    gguf = resolve_gguf_shards(str(inst.get("gguf") or ""))
    is_default = inst.get("default") or inst.get("id") == "default"
    # WP3 calibration defaults — EMPTY unless serve.calibration: auto opts in
    # and a profile matches this machine+model (serving/calibrate.py). Explicit
    # config/instance run_settings below ALWAYS override calibrated values.
    from .calibrate import apply_calibration

    rs = dict(apply_calibration(cfg, gguf))
    if is_default:
        rs.update(getattr(cfg, "run_settings", None) or {})
    rs.update(inst.get("run_settings") or {})
    ctx = inst.get("context_length") or cfg.context_length
    cmd = [
        binary, "-m", gguf, "--alias", model_name,
        "--host", "127.0.0.1", "--port", str(port),
        "-c", str(int(ctx)),
    ]
    if inst.get("role") == "embedding":
        cmd.append("--embedding")
    ngl_cfg = getattr(cfg, "gpu_layers", None)
    cmd += runtime.launch_args(ngl_cfg, rt)
    cmd += run_settings_args(rs, gpu_layers=runtime.gpu_layers(ngl_cfg, rt),
                             gguf_path=gguf)
    # GPU pinning device selection (opt-in: no-op unless the instance defines
    # gpu_id, so every existing command line stays byte-identical).
    cmd += _gpu_pin_args(inst, rs, rt)
    cmd += list(getattr(cfg, "extra_llama_args", None) or [])
    return cmd


def _gpu_pin_args(inst: dict, rs: dict, rt) -> List[str]:
    """llama-server args required by an instance ``gpu_id`` pin.

    CUDA/HIP pinning rides the environment (CUDA/HIP_VISIBLE_DEVICES — see
    instances.instance_device_env), so nothing is needed here. Vulkan has no
    *_VISIBLE_DEVICES equivalent: the adapter is selected with repeatable
    ``--device <id>`` flags instead. And tensor_parallel across a pinned
    multi-GPU set only actually splits when llama.cpp is told to split by
    layer (the default row-split keeps everything on device 0)."""
    gid = inst.get("gpu_id")
    if gid is None:
        return []
    ids = [int(x) for x in (gid if isinstance(gid, list) else [gid])]
    args: List[str] = []
    backend = getattr(rt, "backend", None)
    if backend == runtime.BACKEND_VULKAN:
        for d in ids:
            args += ["--device", str(d)]
    elif len(ids) > 1 and rs.get("tensor_parallel"):
        args += ["--split-mode", "layer"]
    return args


def start_instance_server(cfg, inst: dict, *, log_path: Optional[Path] = None,
                          ready_timeout: float = 300.0,
                          cancel: Optional[threading.Event] = None) -> ServerHandle:
    """Spawn wrapper enforcing the crash-loop budget (#28).

    Refuses while the instance is over budget (explicit user starts clear the
    state first — see dashboard h_server_start); every failed spawn/health
    cycle counts, a healthy server resets its history. ``cancel`` is an
    optional shutdown event polled during the health wait (the legacy
    ServingSupervisor.stop() sets its own mid-start)."""
    crash_key = str(inst.get("id") or inst.get("gguf") or "default")
    if crash_loop_refused(crash_key):
        raise ServeError(
            f"instance {crash_key!r} is in a crash loop "
            f"({CRASH_LOOP_MAX_FAILURES} failed starts within "
            f"{CRASH_LOOP_WINDOW_S / 60:.0f} min) — refusing to respawn. "
            "Start it explicitly from the dashboard or `sable serve` to "
            "clear the crash-loop state."
        )
    try:
        handle = _start_instance_server(
            cfg, inst, log_path=log_path, ready_timeout=ready_timeout,
            cancel=cancel)
    except ServeError:
        record_start_failure(crash_key)
        raise
    clear_crash_loop(crash_key)  # healthy: clean slate for the next cycle
    return handle


def _engine_spawn_cmd(cfg, inst: dict, engine_ref: str, *, gguf: str,
                      port: int):
    """(argv, extra_env, cwd, kind) for an instance bound to a registered engine.

    OpenAI-compatible and diffusion engines are spawnable (sable health-checks
    openai kinds through /v1/models, diffusion kinds on the bare HTTP root);
    kind "other" is inventory-only and refuses here rather than pretending a
    healthcheck will pass. A missing binary raises with the fix path
    (register/repair it on the Hardware page). Diffusion instances append their
    per-instance image_settings (steps/cfg/size/sampler) to the command —
    llama.cpp run_settings do not apply to sd-server."""
    from ..store.paths import resolve_sable_home
    from . import engines as engines_mod

    home = Path(resolve_sable_home(cfg))
    eng = engines_mod.get_engine(home, engine_ref)
    if eng is None:
        raise ServeError(
            f"unknown inference engine {engine_ref!r} — register it on the "
            "Hardware page first")
    kind = str(eng.get("kind") or "openai")
    if kind not in ("openai", "diffusion"):
        raise ServeError(
            f"engine {eng.get('name')!r} is kind={kind!r}: only OpenAI-"
            "compatible and diffusion engines can be spawned — attach it "
            "manually instead")
    probe = engines_mod.probe(eng)
    if not probe["found"]:
        raise ServeError(
            f"engine {eng.get('name')!r} binary not found "
            f"({eng.get('binary')!r}) — install it or fix the path on the "
            "Hardware page")
    cmd = engines_mod.build_engine_cmd(eng, probe["path"], model=gguf,
                                       host="127.0.0.1", port=port)
    if kind == "diffusion":
        from .run_args import image_settings_args

        cmd += image_settings_args(inst.get("image_settings"))
    return (cmd, engines_mod.engine_env(eng),
            str(eng.get("cwd") or "") or None, kind)


def _start_instance_server(cfg, inst: dict, *,
                           log_path: Optional[Path] = None,
                           ready_timeout: float = 300.0,
                           cancel: Optional[threading.Event] = None) -> ServerHandle:
    """Spawn the engine server for one instance record; wait until healthy.

    Registers an owned handle (keyed by instance id) so shutdown stops it —
    and ONLY it. Attach-mode instances never reach this function. Without an
    ``engine`` binding this is exactly the llama-server path as before; with
    one, the argv comes from the registered engine's template (opt-in only —
    serving/engines.py)."""
    from ..core.model_discovery import resolve_gguf_shards

    # Multi-part GGUF safety: always launch against shard 1 (llama.cpp loads
    # the remaining parts itself); the alias keeps the user-visible name.
    gguf = Path(resolve_gguf_shards(str(inst.get("gguf") or ""))).expanduser()
    if not gguf.is_file():
        raise ServeError(f"GGUF file not found: {gguf}")

    port = int(inst.get("port") or 0) or _free_port("127.0.0.1")
    from ..core.model_discovery import shard_group as _shard_group

    _grp = _shard_group(gguf.name)
    model_name = str(inst.get("model_name")
                     or (_grp[0] if _grp else gguf.stem))

    engine_ref = str(inst.get("engine") or "").strip()
    from . import engines as engines_mod
    eng_cwd: Optional[str] = None
    eng_kind: Optional[str] = None
    if engine_ref and not engines_mod.is_builtin_ref(engine_ref):
        # Registered engine: argv + env from the engine record. No llama.cpp
        # discovery/auto-setup, no run_settings translation (the engine owns
        # its own flags; diffusion kinds get their image_settings instead).
        cmd, engine_env, eng_cwd, eng_kind = _engine_spawn_cmd(
            cfg, inst, engine_ref, gguf=str(gguf), port=port)
        child_env_extra = dict(engine_env)
    else:
        binary = find_llama_server(
            cfg.llama_server_binary,
            auto_setup=runtime.auto_setup_enabled(cfg),
            cfg=cfg,
        )
        rt = runtime.detect_runtime()
        cmd = build_instance_cmd(binary, cfg, inst, port=port,
                                 model_name=model_name, rt=rt)
        child_env_extra = dict(runtime.unified_memory_env(rt))

    out = subprocess.DEVNULL
    logf = None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logf = open(log_path, "ab")
        out = logf
    # gpu_id pinning (opt-in): {} unless the instance defines a device, so
    # default spawns keep the ambient environment byte-identically.
    from .instances import instance_device_env

    child_env = {**os.environ, **child_env_extra,
                 **instance_device_env(inst)}
    logger.info("Spawning instance %r: %s", inst.get("id"), " ".join(cmd))
    try:
        proc = subprocess.Popen(
            cmd, stdout=out, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, env=child_env,
            cwd=eng_cwd,  # None => inherit (byte-identical for llama.cpp)
            **runtime.detached_popen_kwargs(),
        )
    except OSError as e:
        raise ServeError(f"Failed to spawn llama-server: {e}") from e
    finally:
        # fork/exec handed the log fd to the child; close OUR copy (#9). Left
        # open, every spawn pins one fd for the lifetime of the Popen/traceback
        # frame — a slow leak toward EMFILE on long-lived dashboards.
        if logf is not None:
            logf.close()

    # Diffusion servers (sd-server) have no /v1 namespace — their base URL is
    # the bare root and readiness is any HTTP response from "/" (the server
    # answers index/404 as soon as it is listening).
    if eng_kind == "diffusion":
        base_url = f"http://127.0.0.1:{port}"
        health_url = base_url + "/"
    else:
        base_url = f"http://127.0.0.1:{port}/v1"
        health_url = base_url + "/models"
    deadline = time.monotonic() + ready_timeout
    last_err = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise ServeError(
                f"llama-server exited early with code {proc.returncode}; "
                f"see {log_path or 'the server log'}")
        if cancel is not None and cancel.is_set():
            proc.terminate()
            raise ServeError("Shutdown requested while server was starting")
        try:
            r = httpx.get(health_url, timeout=3.0)
            if r.status_code == 200:
                break
            last_err = f"HTTP {r.status_code}"
        except Exception as e:  # connection refused while booting is normal
            last_err = str(e)
        time.sleep(0.5)
    else:
        try:
            proc.terminate()
        except Exception:
            pass
        raise ServeError(
            f"llama-server did not become healthy within {ready_timeout:.0f}s "
            f"(last error: {last_err})")

    handle = ServerHandle(base_url=base_url, model_name=model_name,
                          owned=True, process=proc)
    _register_handle(str(inst["id"]), handle)
    return handle


def stop_instance(instance_id: str, grace: float = 8.0) -> bool:
    """Stop the in-process child we spawned for this instance id. True=stopped.

    Never used for attach instances (they have no owned handle) and never for
    pids this process did not spawn — cross-process stop goes through
    serving.instances which re-verifies ownership via the state file."""
    with _owned_lock:
        handle = _owned_handles.pop(instance_id, None)
    if handle is None:
        return False
    handle.stop(grace=grace)
    return True


def install_signal_handlers(supervisor: ServingSupervisor) -> None:
    """Make SIGINT/SIGTERM stop the supervisor (and its child) cleanly."""

    def _handler(signum, frame):  # noqa: ANN001
        logger.info("Received signal %s — shutting down", signal.Signals(signum).name)
        supervisor.stop()
        # Restore default handler and re-raise so the CLI unwinds normally.
        signal.signal(signum, signal.SIG_DFL)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):  # not main thread etc.
            pass

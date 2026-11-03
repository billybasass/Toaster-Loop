"""Hot-swap / deployment helpers (Phase 4).

v0 is daemonless but stateful: when sable spawns a llama-server it records
``<home>/serve_state.json`` {pid, host, port, base_url, model_name, gguf_path,
model_id}. ``sable deploy <model_id>`` then:

  a) stops the sable-owned llama-server if running (verified by pid liveness
     AND that we wrote the state file — never kill processes we didn't start),
  b) starts a detached llama-server on the candidate GGUF (dynamic port),
  c) prints the switch instruction and calls ``runner.switch_model(...)`` when
     an in-process runner is supplied (daemon mode).

Full always-on daemon orchestration is intentionally rough here (Phase 5); the
contract that matters — registry active pointer flips, a healthy server serves
the new GGUF, live runners can be re-pointed via switch_model — holds.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

from ..core.config import SableConfig
from ..registry.registry import ModelRegistry
from ..store.paths import resolve_sable_home
from . import runtime
from .run_args import run_settings_args
from .supervisor import ServeError, find_llama_server, healthcheck, _free_port

logger = logging.getLogger("sable.deploy")

STATE_FILENAME = "serve_state.json"


def state_path(home: Path) -> Path:
    return Path(home) / STATE_FILENAME


def write_serve_state(home: Path, state: Dict[str, Any]) -> None:
    p = state_path(home)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    # The state may embed attach_api_key — restrict before it becomes visible
    # at its final path (chmod the tmp first so there is no world-readable
    # window, then rename atomically).
    os.chmod(tmp, 0o600)
    os.replace(tmp, p)


def read_serve_state(home: Path) -> Optional[Dict[str, Any]]:
    p = state_path(home)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def clear_serve_state(home: Path) -> None:
    try:
        state_path(home).unlink()
    except FileNotFoundError:
        pass


def stop_owned_server(home: Path, grace: float = 8.0) -> bool:
    """Stop the llama-server recorded in serve_state.json (if alive). True=stopped.

    Cross-platform via runtime.terminate_owned_pid — the shared verify-then-kill
    helper (WP4 extraction; os.kill(pid, 0) would terminate on Windows and
    SIGKILL doesn't exist there). A pid that is alive but NOT ours (stale state
    file / recycled pid) is left alone and the stale state cleared. Clears BOTH
    views of the default instance's state (legacy file + per-instance mirror)
    via the merged accessor, so no stale mirror outlives the server."""
    from .instances import DEFAULT_ID, clear_state_merged

    st = read_serve_state(home)
    if not st or not st.get("pid"):
        return False
    pid = int(st["pid"])
    verdict = runtime.terminate_owned_pid(pid, grace=grace)
    if verdict == "foreign":
        # Alive but not ours (state file stale or reused pid) — do NOT kill.
        logger.warning(
            "serve_state pid %s exists but is not owned by us; leaving it alone",
            pid,
        )
        clear_state_merged(home, DEFAULT_ID)
        return False
    if verdict == "gone":
        clear_state_merged(home, DEFAULT_ID)
        return False
    clear_state_merged(home, DEFAULT_ID)
    return True


def start_server_detached(
    cfg: SableConfig, gguf_path: str, model_name: Optional[str] = None,
    model_id: Optional[str] = None, home: Optional[Path] = None,
) -> Dict[str, Any]:
    """Spawn llama-server detached from the CLI; record serve_state. Returns it.

    Crash-loop budget (#28): every failed spawn/healthcheck cycle records via
    ``record_start_failure`` under the reserved "default" key — this path owns
    ``serve_state.json``, i.e. the default instance's legacy view (metrics
    ``sable_instances_up`` and the router pool read it). We deliberately do
    NOT refuse to spawn into an open crash loop: like ``sable serve`` and
    the dashboard Start button, an explicit CLI invocation IS the user-intent
    signal, so the budget is cleared up front and only re-armed on failure —
    dashboard visibility stays honest (``crash_loop_status("default")`` shows
    the latest failed attempt) while the CLI keeps its detach-by-design
    semantics. Auto-respawn refusal stays where it belongs: inside
    ``start_instance_server`` (the autostart-facing wrapper).

    The recorded state mirrors dashboard ``_write_instance_state``'s spawn
    shape (``mode``/``instance_id`` included) so every reader — router pool
    building, ``instances.state_alive``, metrics pid probe — treats a
    CLI-detached server exactly like a dashboard-started default instance.
    """
    from .supervisor import clear_crash_loop, record_start_failure

    crash_key = "default"  # == instances.DEFAULT_ID (avoid the module import)
    clear_crash_loop(crash_key)  # explicit user action (see docstring)
    try:
        return _start_server_detached_once(
            cfg, gguf_path, model_name=model_name, model_id=model_id, home=home)
    except (ServeError, OSError):
        # ServeError = early exit / unhealthy; OSError = spawn failure —
        # same pair supervisor.start_instance_server counts as one cycle.
        record_start_failure(crash_key)
        raise


def _start_server_detached_once(
    cfg: SableConfig, gguf_path: str, model_name: Optional[str] = None,
    model_id: Optional[str] = None, home: Optional[Path] = None,
) -> Dict[str, Any]:
    """One spawn/healthcheck attempt for start_server_detached (no budget)."""
    from ..core.model_discovery import resolve_gguf_shards, shard_group

    # Split GGUF safety: launch against shard 1 (see supervisor.py).
    gguf = Path(resolve_gguf_shards(gguf_path)).expanduser()
    if not gguf.is_file():
        raise ServeError(f"GGUF file not found: {gguf}")
    binary = find_llama_server(
        cfg.llama_server_binary,
        auto_setup=runtime.auto_setup_enabled(cfg),
        cfg=cfg,
    )
    port = cfg.port or _free_port(cfg.host)
    _grp = shard_group(gguf.name)
    name = model_name or cfg.model_name or (_grp[0] if _grp else gguf.stem)

    rt = runtime.detect_runtime()
    # WP3 calibration defaults — EMPTY unless serve.calibration: auto opts in
    # and a profile matches this machine+model (serving/calibrate.py); explicit
    # cfg.run_settings merge on top, so manual values always win.
    from .calibrate import apply_calibration

    rs = {**apply_calibration(cfg, str(gguf)), **(cfg.run_settings or {})}
    cmd = [
        binary, "-m", str(gguf), "--alias", name,
        "--host", cfg.host, "--port", str(port),
        "-c", str(cfg.context_length),
    ] + runtime.launch_args(cfg.gpu_layers, rt) \
        + run_settings_args(rs,
                            gpu_layers=runtime.gpu_layers(cfg.gpu_layers, rt),
                            gguf_path=str(gguf)) \
        + list(cfg.extra_llama_args)

    # Honor the configured home (SABLE_HOME env > config `home:` > ~/.sable)
    # so serve_state.json and logs land in the same root as everything else.
    root = Path(home) if home is not None else resolve_sable_home(cfg)
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "llama-server.log"
    logf = open(log_path, "ab")

    try:
        proc = subprocess.Popen(
            cmd, stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            env={**os.environ, **runtime.unified_memory_env(rt)},
            **runtime.detached_popen_kwargs(),  # survive the CLI exiting (any OS)
        )
    finally:
        # fork/exec handed the log fd to the child; close OUR copy (#9) —
        # leaving it open pins one fd per spawn for the Popen's lifetime.
        logf.close()

    base_url = f"http://{cfg.host}:{port}/v1"
    deadline = time.monotonic() + 300.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise ServeError(
                f"llama-server exited early (rc={proc.returncode}); "
                f"see {log_path}"
            )
        if healthcheck(base_url):
            break
        time.sleep(0.5)
    else:
        try:
            proc.terminate()
        except Exception:
            pass
        raise ServeError(f"llama-server did not become healthy at {base_url}")

    # Shape parity with dashboard _write_instance_state's spawn record (same
    # keys): mode/instance_id let router pool building and metrics treat a
    # CLI-detached default server exactly like a dashboard-started one.
    from .instances import DEFAULT_ID, write_state_merged

    state = {
        "mode": "spawn",
        "pid": proc.pid,
        "host": cfg.host,
        "port": port,
        "base_url": base_url,
        "model_name": name,
        "model_id": model_id,
        "gguf_path": str(gguf),
        "instance_id": DEFAULT_ID,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    # Merged write: legacy serve_state.json AND the per-instance mirror, so a
    # CLI-detached default server is visible to every Phase-6 reader and the
    # merged read never prefers a stale mirror over this fresher record.
    write_state_merged(log_dir.parent, DEFAULT_ID, state)
    return state


def deploy_model(
    home: Path,
    registry: ModelRegistry,
    model_id: str,
    cfg: SableConfig,
    runner: Optional[Any] = None,
    *,
    stop_first: bool = True,
) -> Dict[str, Any]:
    """Deploy a registered model (must have a gguf_path); hot-swap runners.

    Returns {model_id, base_url, model_name, switched, previous}. Does NOT
    change the registry active pointer — that is `models promote`'s job;
    deploy just makes the chosen artifact live.
    """
    model = registry.get_model(model_id)
    if model is None:
        raise ServeError(f"no model {model_id} in registry")
    # Backlog item #16: archived models must never be served by accident.
    # (Their artifact columns are cleared on archive, but fail loudly with a
    # restore hint rather than the generic "no GGUF" error.)
    if model.status == "archived":
        raise ServeError(
            f"model {model_id} is archived — run `sable models restore "
            f"{model_id}` first"
        )
    if not model.gguf_path:
        raise ServeError(
            f"model {model_id} has no GGUF artifact "
            "(run the export step / `sable iterate` first)"
        )

    previous = read_serve_state(home)
    stopped = stop_owned_server(home) if stop_first else False

    state = start_server_detached(
        cfg, model.gguf_path, model_name=model.model_id or gguf_stem(model),
        model_id=model.model_id, home=home,
    )
    # Idea adapted from PIASystem (MIT) app/core/adapter_manager.py — stamp the
    # activation so age-based archival policy sees actual use. Once per deploy,
    # never per token; record_activation never raises.
    registry.record_activation(model.model_id)

    switched = False
    if runner is not None and hasattr(runner, "switch_model"):
        try:
            switched = bool(runner.switch_model(
                state["base_url"], state["model_name"],
                api_key=cfg.attach_api_key,
            ))
        except Exception as e:  # daemon path must never crash on swap
            logger.warning("runner.switch_model raised %s: %s", type(e).__name__, e)

    return {
        "model_id": model.model_id,
        "base_url": state["base_url"],
        "model_name": state["model_name"],
        "gguf_path": state["gguf_path"],
        "stopped_previous": stopped,
        "previous": previous,
        "switched": switched,
    }


def gguf_stem(model: Any) -> str:
    from pathlib import Path as _P

    return _P(str(model.gguf_path)).stem if model.gguf_path else model.model_id

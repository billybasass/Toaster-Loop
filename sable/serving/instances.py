"""Multi-instance serving registry (feature-wave Phase 6).

N named llama-server instances instead of one active GGUF. Design choices:

* **Persistence**: instance *definitions* live as one JSON file per instance
  at ``<SABLE_HOME>/serving/instances/<id>.json`` — the same file-per-record
  pattern the projects/chat threads already use (atomic tmp+replace writes).
  Runtime state (pid/base_url) lives separately at
  ``<SABLE_HOME>/serving/state/<id>.json``, mirroring the legacy
  ``serve_state.json`` shape per instance. Config YAML stays user-editable
  config; these are program state, so they don't bloat sable.yaml.
* **Backward compatibility**: the reserved id ``"default"`` has no file — it
  is synthesized from the legacy single-instance config keys (``gguf_path``,
  ``attach_url``, ``port``, ...) with runtime state read from the legacy
  ``serve_state.json``. Old endpoints keep writing that file, so old and new
  views of the default instance never diverge.
* **Ownership**: spawning goes through ``supervisor.start_instance_server``
  which registers an in-process handle; cross-process stop re-reads the
  per-instance state file and verifies the pid is ours via
  ``runtime.pid_alive`` (None => not ours => never kill). Attach-URL
  instances spawn nothing and are never killed — detaching only clears the
  state file.
* **Loopback**: spawned servers bind 127.0.0.1; attach URLs are validated
  loopback-only (``supervisor.require_loopback``) to respect the project
  constraint that sable never talks to (or proxies out to) non-local
  endpoints.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.config import SableConfig
from . import runtime
from .supervisor import ServeError, healthcheck, require_loopback

logger = logging.getLogger("sable.serving.instances")

#: Reserved id for the legacy single-instance serve (config-derived).
DEFAULT_ID = "default"

#: Instance roles (free-form beyond these; role is a tag, not a switch —
#: except "embedding", which spawns llama-server with --embedding).
ROLES = ("orchestrator", "coding", "embedding", "custom")

_INSTANCE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


# ------------------------------------------------------------------ paths

def instances_dir(home: Path) -> Path:
    return Path(home) / "serving" / "instances"


def state_dir(home: Path) -> Path:
    return Path(home) / "serving" / "state"


def instance_path(home: Path, instance_id: str) -> Path:
    if not _INSTANCE_ID_RE.match(instance_id):
        raise ServeError(f"invalid instance id: {instance_id!r}")
    return instances_dir(home) / f"{instance_id}.json"


def instance_state_path(home: Path, instance_id: str) -> Path:
    return state_dir(home) / f"{instance_id}.json"


# ------------------------------------------------------------- definitions

def _read_json(p: Path) -> Optional[Dict[str, Any]]:
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def default_instance(cfg: SableConfig) -> Dict[str, Any]:
    """Synthesize the reserved "default" instance from legacy config keys.

    Old configs carry a single active-GGUF setup (gguf_path / attach_url /
    port / context_length); that maps 1:1 onto an instance definition.
    ``engine`` comes from serve.default_engine (lenient read; unset/blank =>
    None => the built-in llama.cpp path, byte-identical to pre-engine
    behavior)."""
    engine = str((getattr(cfg, "serve", None) or {})
                 .get("default_engine") or "").strip() or None
    return {
        "id": DEFAULT_ID,
        "name": "default",
        "gguf": cfg.gguf_path or None,
        "attach_url": cfg.effective_attach_url or None,
        "port": int(cfg.port or 0),
        "role": "custom",
        "context_length": cfg.context_length,
        "engine": engine,
        "run_settings": {},   # inherits the global run_settings:
        "sampling": {},       # inherits the global sampling:
        "image_settings": {},
        "autostart": False,
        "created": None,
        "updated": None,
        "default": True,
    }


def load_instances(home: Path, cfg: Optional[SableConfig] = None) -> List[Dict[str, Any]]:
    """All instance definitions, default first. The reserved "default" row is
    always present (config-derived unless a file overrides it)."""
    out: List[Dict[str, Any]] = []
    d = _read_json(instances_dir(home) / f"{DEFAULT_ID}.json") if \
        instances_dir(home).is_dir() else None
    if d is not None:
        out.append(d)
    elif cfg is not None:
        out.append(default_instance(cfg))
    dir_ = instances_dir(home)
    if dir_.is_dir():
        for p in sorted(dir_.glob("*.json")):
            if p.stem == DEFAULT_ID:
                continue
            inst = _read_json(p)
            if inst and inst.get("id"):
                out.append(inst)
    return out


def get_instance(home: Path, instance_id: str,
                 cfg: Optional[SableConfig] = None) -> Optional[Dict[str, Any]]:
    if instance_id == DEFAULT_ID:
        d = _read_json(instance_path(home, DEFAULT_ID))
        if d is not None:
            return d
        if cfg is not None:
            return default_instance(cfg)
        return None
    return _read_json(instance_path(home, instance_id))


def save_instance(home: Path, inst: Dict[str, Any]) -> None:
    p = instance_path(home, str(inst["id"]))
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(inst, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    os.replace(tmp, p)


def delete_instance_file(home: Path, instance_id: str) -> bool:
    removed = False
    for p in (instance_path(home, instance_id),
              instance_state_path(home, instance_id)):
        try:
            p.unlink()
            removed = True
        except FileNotFoundError:
            pass
    return removed


def _utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def validate_instance_body(body: Dict[str, Any],
                           existing: Optional[Dict[str, Any]] = None,
                           ) -> Dict[str, Any]:
    """Validate/normalize a create-or-update body into an instance record.

    Rules: name required; exactly one of gguf / attach_url (gguf wins when
    both given is NOT allowed — ambiguous target); attach URLs must be
    loopback; port 0/absent = dynamic. run_settings/sampling are stored as
    override maps (validated upstream where the spawn/chat builds them)."""
    existing = existing or {}
    name = str(body.get("name", existing.get("name")) or "").strip()
    if not name:
        raise ServeError("name required (non-empty string)")
    gguf = str(body.get("gguf", existing.get("gguf")) or "").strip() or None
    attach = str(body.get("attach_url", existing.get("attach_url"))
                 or "").strip() or None
    if gguf and attach:
        raise ServeError("an instance is either a GGUF spawn or an attach "
                         "URL, not both — clear one")
    if not gguf and not attach:
        raise ServeError("provide gguf (path) or attach_url (loopback endpoint)")
    if attach:
        attach = require_loopback(attach)
        if not attach.rstrip("/").endswith("/v1"):
            attach = attach.rstrip("/") + "/v1"
    role = str(body.get("role", existing.get("role")) or "custom").strip() \
        or "custom"
    try:
        port = int(body.get("port", existing.get("port")) or 0)
    except (TypeError, ValueError):
        raise ServeError("port must be an integer (0 = dynamic)") from None
    if not (0 <= port <= 65535):
        raise ServeError("port must be 0..65535")
    ctx_raw = body.get("context_length", existing.get("context_length"))
    context_length: Optional[int] = None
    if ctx_raw not in (None, "", 0):
        try:
            context_length = int(ctx_raw)
        except (TypeError, ValueError):
            raise ServeError("context_length must be an integer") from None
        if context_length < 128:
            raise ServeError("context_length must be >= 128")
    rs = body.get("run_settings", existing.get("run_settings")) or {}
    sp = body.get("sampling", existing.get("sampling")) or {}
    im = body.get("image_settings", existing.get("image_settings")) or {}
    if not isinstance(rs, dict) or not isinstance(sp, dict) \
            or not isinstance(im, dict):
        raise ServeError("run_settings, sampling and image_settings must "
                         "be objects")
    # Instance run settings are their OWN (new instances do not inherit the
    # global run_settings: section — see supervisor.build_instance_cmd), so
    # validate them here with the same rules the config API applies.
    from .run_args import sanitize_run_settings, sanitize_sampling

    try:
        rs = sanitize_run_settings(rs)
    except ValueError as e:
        raise ServeError(f"run_settings: {e}") from e
    # Drop None values so the stored record only carries explicit choices.
    rs = {k: v for k, v in rs.items() if v is not None}
    # Per-instance sampling + thinking overrides (reference style): same
    # validation the config API applies to the global `sampling:` section.
    try:
        sp = sanitize_sampling(sp)
    except ValueError as e:
        raise ServeError(f"sampling: {e}") from e
    sp = {k: v for k, v in sp.items() if v is not None}
    # Diffusion engines (sd-server) carry generation defaults instead of the
    # chat knobs above: steps / cfg_scale / width / height / sampler / ...
    from .run_args import sanitize_image_settings

    try:
        im = sanitize_image_settings(im)
    except ValueError as e:
        raise ServeError(f"image_settings: {e}") from e
    im = {k: v for k, v in im.items() if v is not None}
    autostart = bool(body.get("autostart", existing.get("autostart") or False))
    # Optional GPU pinning (multi-GPU boxes): int or list[int] of physical
    # device ids. Absent/None => spawn with the ambient environment exactly
    # like before (cardinal rule: opt-in, byte-identical default).
    gpu_id = _normalize_gpu_id(
        body.get("gpu_id", existing.get("gpu_id")))
    # Optional inference-engine binding (serving/engines.py). Blank/absent =>
    # None => the built-in llama.cpp spawn path, byte-identically. Existence
    # of a non-builtin id is checked by the caller (needs SABLE_HOME).
    engine = str(body.get("engine", existing.get("engine")) or "").strip() \
        or None
    if engine == "llama-cpp":  # canonical name for the built-in => default path
        engine = None
    return {
        "id": str(existing.get("id") or body.get("id") or ""),
        "name": name,
        "gguf": gguf,
        "attach_url": attach,
        "port": port,
        "role": role if role in ROLES else "custom",
        "role_tag": role,
        "context_length": context_length,
        "engine": engine,
        "run_settings": rs,
        "sampling": sp,
        "image_settings": im,
        "autostart": autostart,
        "gpu_id": gpu_id,
        "created": existing.get("created") or _utcnow(),
        "updated": _utcnow(),
    }


def _normalize_gpu_id(raw: Any) -> Optional[Any]:
    """Validate/normalize an instance ``gpu_id``: int | list[int] | None.

    Invalid values raise ServeError (config-API style: refuse loudly rather
    than silently ignoring a device pin that would then be silently ignored)."""
    if raw is None or raw == "" or raw == []:
        return None
    if isinstance(raw, bool):  # bool is an int subclass — never a device id
        raise ServeError("gpu_id must be an integer or list of integers")
    if isinstance(raw, int):
        if raw < 0:
            raise ServeError("gpu_id must be >= 0")
        return int(raw)
    if isinstance(raw, (list, tuple)):
        out = []
        for x in raw:
            if isinstance(x, bool) or not isinstance(x, int):
                raise ServeError("gpu_id list entries must be integers")
            if x < 0:
                raise ServeError("gpu_id must be >= 0")
            out.append(int(x))
        return out or None
    # Numeric strings from form fields are acceptable ("1"), anything else no.
    if isinstance(raw, str) and raw.strip().isdigit():
        return int(raw.strip())
    raise ServeError("gpu_id must be an integer or list of integers")


def instance_device_env(inst: Dict[str, Any]) -> Dict[str, str]:
    """Child-process env pinning this instance to its gpu_id, or {} when the
    instance has no pin (spawn stays byte-identical).

    CUDA/HIP_VISIBLE_DEVICES are set together (torch/llama.cpp both honor
    their respective one; setting both is harmless on the other vendor). The
    value uses the canonical comma-separated device list."""
    gid = inst.get("gpu_id")
    if gid is None:
        return {}
    ids = gid if isinstance(gid, list) else [gid]
    val = ",".join(str(int(x)) for x in ids)
    return {"CUDA_VISIBLE_DEVICES": val, "HIP_VISIBLE_DEVICES": val}


def new_instance_id(name: str, taken: List[str]) -> str:
    base = re.sub(r"[^A-Za-z0-9_-]", "-", name.strip()).strip("-")[:48] or "server"
    cid = base
    n = 2
    while cid in taken:
        cid = f"{base}-{n}"
        n += 1
    return cid


# -------------------------------------------------------------- runtime state

def read_state(home: Path, instance_id: str) -> Optional[Dict[str, Any]]:
    return _read_json(instance_state_path(home, instance_id))


def write_state(home: Path, instance_id: str, state: Dict[str, Any]) -> None:
    p = instance_state_path(home, instance_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    # State may embed attach_api_key — restrict on-disk access before the
    # atomic rename makes it visible at its final path.
    os.chmod(tmp, 0o600)
    os.replace(tmp, p)


def clear_state(home: Path, instance_id: str) -> None:
    try:
        instance_state_path(home, instance_id).unlink()
    except FileNotFoundError:
        pass


def state_alive(st: Optional[Dict[str, Any]]) -> bool:
    """Is this recorded state pointing at a live endpoint?"""
    if not st:
        return False
    if st.get("mode") == "attach":
        return healthcheck(str(st.get("base_url", "")), timeout=2.0,
                           api_key=st.get("attach_api_key"))
    pid = st.get("pid")
    if not pid:
        return False
    try:
        pid_i = int(pid)
    except ValueError:
        return False
    # Windows-safe liveness (A1): raw os.kill(pid, 0) TERMINATES on NT.
    # None = alive but not ours to inspect -> same as the old
    # PermissionError branch (consider it running, never kill it).
    return runtime.pid_alive(pid_i) is not False


def stop_recorded(home: Path, instance_id: str, grace: float = 8.0) -> bool:
    """Stop the process recorded in the per-instance state file, if WE own it.

    Same verification chain as deploy.stop_owned_server — both now share the
    WP4-extracted ``runtime.terminate_owned_pid`` helper (the single
    verify-then-kill sequence): ``pid_alive`` returning None (alive but not
    ours / stale pid reuse) means leave it alone and clear the stale state.
    Attach-mode states have no pid and are just cleared by the caller — never
    killed.

    In-process children spawned by THIS process should be stopped through
    ``supervisor.stop_instance`` first (that path owns the Popen handle)."""
    st = read_state(home, instance_id)
    if not st or not st.get("pid"):
        return False
    pid = int(st["pid"])
    verdict = runtime.terminate_owned_pid(pid, grace=grace)
    if verdict == "foreign":
        logger.warning("instance %s pid %s exists but is not owned by us; "
                       "leaving it alone", instance_id, pid)
        clear_state(home, instance_id)
        return False
    if verdict == "gone":
        clear_state(home, instance_id)
        return False
    logger.info("Stopped sable-owned llama-server for instance %s pid=%s",
                instance_id, pid)
    clear_state(home, instance_id)
    return True


# ----------------------------------------------------------- port management

def ports_in_use(home: Path, cfg: Optional[SableConfig] = None,
                 exclude_id: Optional[str] = None) -> List[int]:
    """Every port sable knows about: configured + running + embed servers.

    Shared free-port allocation (Phase 6): spawn-time checks consult this so a
    new instance never grabs the port of another managed server, including
    the indexer's embedding server."""
    used: List[int] = []
    for inst in load_instances(home, cfg=cfg):
        iid = str(inst.get("id"))
        if iid == exclude_id:
            continue
        p = inst.get("port")
        if p:
            used.append(int(p))
        st = read_state(home, iid)
        bp = (st or {}).get("port")
        if bp:
            used.append(int(bp))
    # Phase 5 embedding servers spawned in-process (their ports aren't files).
    try:
        import threading as _th  # noqa: F401  (lock lives in indexer)
        from ..core.indexer import _spawn_lock, _spawned_clients
        from urllib.parse import urlparse
        with _spawn_lock:
            clients = list(_spawned_clients)
        for c in clients:
            url = getattr(c, "url", None)
            if url:
                port = urlparse(url).port
                if port:
                    used.append(int(port))
    except Exception:
        pass
    return sorted(set(used))


def pick_free_port(home: Path, cfg: SableConfig,
                   exclude_id: Optional[str] = None) -> int:
    """Dynamic port allocation that avoids every known in-use port."""
    from .supervisor import _free_port

    reserved = set(ports_in_use(home, cfg=cfg, exclude_id=exclude_id))
    for _ in range(50):
        p = _free_port("127.0.0.1")
        if p not in reserved:
            return p
    raise ServeError("could not allocate a free port")


def check_port_conflict(home: Path, cfg: SableConfig, port: int,
                        exclude_id: Optional[str] = None) -> bool:
    """True when `port` is already claimed by another managed instance."""
    if not port:
        return False
    return port in set(ports_in_use(home, cfg=cfg, exclude_id=exclude_id))


# ------------------------------------------------------------- VRAM estimate

#: Rough llama.cpp resident-memory heuristic multipliers (Phase 6 warning —
#: advisory only, never a hard block). Weights dominate; KV cache grows with
#: the context window (~1 MiB per 1k tokens at typical small-model configs).
GGUF_HEADROOM = 1.2          # weights + compute buffers
KV_MIB_PER_1K_CTX = 1.0      # crude KV estimate


def estimate_model_vram_mib(gguf_path: str,
                            context_length: Optional[int] = None) -> Optional[int]:
    """Estimated VRAM (MiB) to serve a GGUF; None when the file is missing.

    Split (multi-part) GGUFs count ALL shards — a shard-1-only stat would
    underestimate a 3-part model by two thirds."""
    from ..core.model_discovery import gguf_total_size

    total = gguf_total_size(str(gguf_path))
    if not total:
        return None
    size_mib = total / (1024 * 1024)
    est = size_mib * GGUF_HEADROOM
    if context_length:
        est += (int(context_length) / 1024.0) * KV_MIB_PER_1K_CTX
    return int(est) + 1


def free_vram_mib(device: Optional[str] = None) -> Optional[int]:
    """Free VRAM right now; None when no GPU telemetry exists (skip check).

    Running sable instances already hold VRAM, so this live reading is the
    "free accounting for other running instances" figure the over-commit
    warning compares against. ``device`` restricts the reading to one
    physical device id (pinned instances); None keeps the global minimum.
    Tests monkeypatch this to stub hardware."""
    from ..training.scheduler import gpu_free_mib

    return gpu_free_mib(device) if device is not None else gpu_free_mib()


def vram_advisory(inst: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Over-commit advisory for a spawn instance: {needed_mib, free_mib,
    message} when the estimate exceeds currently free VRAM; None when the
    check is not applicable (attach instance, missing file, no telemetry).

    A pinned instance (``gpu_id``) is judged against ITS device's free VRAM,
    not the global minimum — otherwise N instances on a multi-GPU box all
    compare against whichever GPU happens to be emptiest."""
    if inst.get("attach_url") or not inst.get("gguf"):
        return None
    needed = estimate_model_vram_mib(str(inst["gguf"]),
                                     inst.get("context_length"))
    if needed is None:
        return None
    gid = inst.get("gpu_id")
    device = None
    if gid is not None:
        first = gid[0] if isinstance(gid, list) else gid
        device = str(int(first)) if not isinstance(first, bool) else None
    # Unpinned instances keep the exact zero-arg historical probe call.
    free = free_vram_mib(device) if device is not None else free_vram_mib()
    if free is None or needed <= free:
        return None
    where = f" on GPU {device}" if device is not None else ""
    return {
        "needed_mib": needed,
        "free_mib": int(free),
        "message": (f"model needs ~{needed} MiB but only {int(free)} MiB of "
                    f"VRAM is free{where} after other running servers — start "
                    "may OOM or thrash (pass force to override)"),
    }

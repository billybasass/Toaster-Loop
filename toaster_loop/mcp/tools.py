"""MCP tool registry for toaster_loop (WP6).

Every tool mirrors an EXISTING CLI/dashboard code path — the MCP server adds
no new authority. Read-only tools call the same in-process handlers the
dashboard uses (``DashboardApp.handle``) or read the same files the CLI
reads. Mutating tools are wrapped by the server's plan-then-confirm flow
before their executor runs.

Classification table (also in README "toaster-loop mcp"):

  tool              class     mirrors
  ────────────────  ────────  ─────────────────────────────────────────────
  status            read      GET /api/status  (`toaster_loop` overview chips)
  instances_list    read      GET /api/servers (`toaster-loop instances status`)
  models_list       read      GET /api/models  (`toaster-loop models list`)
  logs_tail         read      <home>/logs/server-<id>.log (bounded tail)
  instance_start    MUTATING  POST /api/servers/<id>/start
  instance_stop     MUTATING  POST /api/servers/<id>/stop
  instance_load     MUTATING  POST /api/servers/<id>/load   (WP5)
  instance_unload   MUTATING  POST /api/servers/<id>/unload (WP5)
  curate_enqueue    MUTATING  POST /api/curate  (`toaster-loop curate`, queued)
  train_enqueue     MUTATING  POST /api/train   (`toaster-loop train start`)
  validate          MUTATING  training.validate.validate_model (spawns a
                              llama-server + occupies the GPU: side effects,
                              even though no registry/buffer state is written)
  promote           MUTATING  POST /api/models/promote (`toaster-loop models promote`)
  bench             MUTATING  `toaster-loop bench` measure mode (may spawn a
                              server; persists a record under <home>/bench/)

`validate` classification rationale: it does not mutate the registry, but its
code path spawns llama-server and replays golden tasks against it — that is
a real process + GPU side effect, so policy treats it as mutating. Same for
`bench` (spawns endpoints, persists measurement records).

No executor ever receives a filesystem path from the caller (ids only,
validated against the instance-id pattern), and nothing is passed to a
shell: executors import and call Python functions directly.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .schema import validate_args  # noqa: F401  (re-export convenience)


class ToolError(Exception):
    """Execution failure surfaced to the MCP client as an isError result."""


@dataclass
class Tool:
    name: str
    description: str
    #: Fixed JSON Schema for arguments (object, additionalProperties:false).
    input_schema: Dict[str, Any]
    mutating: bool
    #: executor(ctx: "ToolContext", args: dict) -> JSON-serializable result.
    executor: Callable[["ToolContext", Dict[str, Any]], Any]
    #: Human-readable plan text for the confirm flow (mutating tools).
    effect: str = ""
    risks: List[str] = field(default_factory=list)
    #: arg name whose value names the plan target (for the PLAN envelope).
    target_arg: Optional[str] = None
    #: Optional capability family. "plans" tools exist ONLY when config
    #: plans.enabled is true — the server filters them out of tools/list and
    #: refuses calls when the knob is off (documented choice over a cfg-aware
    #: build_tools(): the server owns config/app lifecycle, so it can check
    #: cheaply at list time; tests monkeypatch app.cfg directly).
    family: str = ""


@dataclass
class ToolContext:
    """Everything executors need; one per MCP server process."""

    #: DashboardApp instance (lazy — constructed on first use by the server).
    app: Any
    config_path: Optional[str] = None

    @property
    def home(self) -> Path:
        return Path(self.app.home)


# --------------------------------------------------------------- helpers

def _api(ctx: ToolContext, method: str, path: str,
         body: Optional[Dict[str, Any]] = None) -> Any:
    """Call the dashboard API in-process; raise ToolError on any error status.

    This is the SAME ``DashboardApp.handle`` entry the HTTP layer adapts to
    sockets — identical validation, conflict semantics and side effects, no
    network involved.
    """
    resp = ctx.app.handle(method, path, body=dict(body or {}))
    status = getattr(resp, "status", 200)
    payload = getattr(resp, "body", None)
    if status >= 400:
        err = (payload or {}).get("error") if isinstance(payload, dict) else None
        msg = (err or {}).get("message") if isinstance(err, dict) else str(payload)
        code = (err or {}).get("code") if isinstance(err, dict) else None
        suffix = f" ({code})" if code else ""
        raise ToolError(f"{method} {path} -> HTTP {status}{suffix}: {msg}")
    return payload


# --------------------------------------------------------------- read tools

def _exec_status(ctx: ToolContext, args: Dict[str, Any]) -> Any:
    return _api(ctx, "GET", "/api/status")


def _exec_instances_list(ctx: ToolContext, args: Dict[str, Any]) -> Any:
    return _api(ctx, "GET", "/api/servers")


def _exec_models_list(ctx: ToolContext, args: Dict[str, Any]) -> Any:
    return _api(ctx, "GET", "/api/models")


#: Hard cap for logs_tail so a client can never ask for an unbounded read.
LOGS_TAIL_MAX_LINES = 500
LOGS_TAIL_DEFAULT_LINES = 100


def _exec_logs_tail(ctx: ToolContext, args: Dict[str, Any]) -> Any:
    """Bounded tail of one instance's server log, contained to <home>/logs.

    The caller supplies only an instance id (validated against the same
    pattern serving/instances.py enforces); the path is constructed here, so
    no path component is ever taken from the client. ``validate_within_dir``
    (core/path_security.py) is a defense-in-depth containment check that
    also rejects ``..`` traversal and symlink escapes before the file opens.
    """
    from ..core.path_security import has_traversal_component, validate_within_dir
    from ..serving.instances import _INSTANCE_ID_RE

    instance_id = str(args["instance_id"])
    lines = int(args.get("lines", LOGS_TAIL_DEFAULT_LINES))
    if not _INSTANCE_ID_RE.match(instance_id) or \
            has_traversal_component(instance_id):
        raise ToolError(f"invalid instance id {instance_id!r}")
    logs_dir = ctx.home / "logs"
    log_path = logs_dir / f"server-{instance_id}.log"
    denied = validate_within_dir(log_path, logs_dir)
    if denied is not None:
        raise ToolError(f"refused: {denied}")
    if not log_path.is_file():
        raise ToolError(f"no log file for instance {instance_id!r} "
                        f"({log_path})")
    try:
        content = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        raise ToolError(f"cannot read log: {e}") from e
    all_lines = content.splitlines()
    return {
        "instance_id": instance_id,
        "log_path": str(log_path),
        "total_lines": len(all_lines),
        "returned": min(lines, len(all_lines)),
        "lines": all_lines[-lines:],
    }


# ----------------------------------------------------------- mutating tools

def _exec_instance_start(ctx: ToolContext, args: Dict[str, Any]) -> Any:
    body: Dict[str, Any] = {}
    for k in ("port", "context", "force"):
        if args.get(k) is not None:
            body[k] = args[k]
    return _api(ctx, "POST", f"/api/servers/{args['id']}/start", body)


def _exec_instance_stop(ctx: ToolContext, args: Dict[str, Any]) -> Any:
    return _api(ctx, "POST", f"/api/servers/{args['id']}/stop")


def _exec_instance_load(ctx: ToolContext, args: Dict[str, Any]) -> Any:
    return _api(ctx, "POST", f"/api/servers/{args['id']}/load")


def _exec_instance_unload(ctx: ToolContext, args: Dict[str, Any]) -> Any:
    return _api(ctx, "POST", f"/api/servers/{args['id']}/unload")


def _exec_curate_enqueue(ctx: ToolContext, args: Dict[str, Any]) -> Any:
    # DashboardApp.submit_curate validates filters synchronously (bad body =>
    # immediate error) and queues a kind="curate" job; the response carries
    # {job_id, state:"queued"} exactly like the dashboard.
    return _api(ctx, "POST", "/api/curate", dict(args))


def _exec_train_enqueue(ctx: ToolContext, args: Dict[str, Any]) -> Any:
    # submit_train enforces single-flight: while a job runs it answers
    # 409 already_running. _api turns that into a ToolError, so the MCP
    # client sees an error — never a silent queue-behind.
    return _api(ctx, "POST", "/api/train", dict(args))


def _exec_promote(ctx: ToolContext, args: Dict[str, Any]) -> Any:
    body = {"model_id": args["model_id"]}
    if args.get("intent"):
        body["intent"] = args["intent"]
    return _api(ctx, "POST", "/api/models/promote", body)


def _exec_validate(ctx: ToolContext, args: Dict[str, Any]) -> Any:
    """Replay the golden suite against one registry model's GGUF.

    Same code path `toaster-loop iterate` uses for its validation gate
    (training.validate.validate_model + cli._validation_config). Spawns a
    temporary llama-server; writes no registry/buffer state — classified
    mutating because of the process/GPU side effect.
    """
    from ..cli import _validation_config
    from ..registry import ModelRegistry
    from ..training.validate import validate_model

    reg = ModelRegistry(ctx.home / "buffer" / "toaster_loop.db")
    try:
        m = reg.get_model(args["model_id"])
        if m is None:
            raise ToolError(f"no model {args['model_id']!r} in the registry")
        if not m.gguf_path:
            raise ToolError(f"model {m.model_id} has no GGUF artifact to "
                            "validate")
        gguf = str(m.gguf_path)
        name = m.model_id
    finally:
        reg.close()

    vcfg = _validation_config(ctx.app.cfg, ctx.config_path)
    report = validate_model(gguf, vcfg, model_name=name)
    return report


def _exec_bench(ctx: ToolContext, args: Dict[str, Any]) -> Any:
    """`toaster-loop bench` measure mode (WP1 harness), same functions.

    Endpoint resolution mirrors cmd_bench: an --attach URL (loopback-only,
    enforced here before anything else) or the configured/spawned
    llama-server. Persists a record under <home>/bench/ like the CLI does.
    """
    from ..cli import _resolve_endpoint
    from ..serving import bench as bench_mod
    from ..serving.supervisor import ServeError

    cfg = ctx.app.cfg
    attach = args.get("attach")
    if attach:
        # Loopback-only rule, enforced up front (the CLI honors the same
        # invariant via supervisor.require_loopback on the spawn path).
        _require_loopback_url(attach)
        cfg = copy.copy(cfg)
        cfg.attach_url = attach

    supervisor = None
    try:
        base_url, model, supervisor = _resolve_endpoint(cfg, spawn_if_needed=True)
        record = bench_mod.run_bench(
            base_url, model,
            runs=args.get("runs") or bench_mod.DEFAULT_RUNS,
            max_tokens=args.get("max_tokens") or bench_mod.DEFAULT_MAX_TOKENS,
            label=str(args.get("label") or ""),
            kind=(bench_mod.KIND_NOISE_FLOOR if args.get("noise_floor")
                  else bench_mod.KIND_BENCH),
        )
    except (ServeError, bench_mod.BenchError) as e:
        raise ToolError(str(e)) from e
    finally:
        if supervisor is not None:
            supervisor.stop()

    saved = bench_mod.save_result(record)
    ti = record.get("token_identity") or {}
    return {
        "record": record,
        "saved_path": str(saved) if saved else None,
        "token_identity": ti.get("status"),
        "valid_speed_claims": ti.get("status") == "ok",
    }


# ------------------------------------------------------------ plan tools
# WP12: the MCP mirror of the dashboard /api/plans handlers — every executor
# calls the SAME op-routed endpoints via _api() (no new authority, no paths:
# project ids + task ids only, pattern-validated). Reads execute immediately;
# everything that writes, spawns agents, or burns the GPU is mutating with the
# full plan-then-confirm flow. When config plans.enabled is off these tools
# are absent from tools/list AND refuse if called anyway (server-side family
# gate).

#: Project ids are file names under <home>/plans/ — same pattern as the
#: projects API; "default" addresses the no-project plan.
_PROJECT_ID_SCHEMA = {
    "type": "string",
    "pattern": "^[A-Za-z0-9._-]{1,64}$",
    "description": ("Project id (see projects_list); 'default' is the "
                    "no-project plan."),
}

_TASK_ID_SCHEMA = {
    "type": "string",
    "pattern": "^t[0-9]{1,6}$",
    "description": "Plan task id like t3 (see plans_list).",
}

#: deps: real string list via the WP12 array validator (scalar items only,
#: pattern-checked, hard-capped at 64).
_DEPS_ARG = {
    "type": "array",
    "items": {"type": "string", "pattern": "^t[0-9]{1,6}$"},
    "maxItems": 32,
    "description": "Task ids this task depends on (must exist in the plan).",
}


def _plans_api(ctx: ToolContext, project_id: str, method: str = "POST",
               body: Optional[Dict[str, Any]] = None) -> Any:
    return _api(ctx, method, f"/api/plans/{project_id}", body)


def _exec_plans_list(ctx: ToolContext, args: Dict[str, Any]) -> Any:
    return _plans_api(ctx, str(args["project"]), "GET")


def _exec_plan_loop_status(ctx: ToolContext, args: Dict[str, Any]) -> Any:
    data = _plans_api(ctx, str(args["project"]), "GET")
    return {"project": str(args["project"]),
            "loop": (data or {}).get("loop")}


def _exec_projects_list(ctx: ToolContext, args: Dict[str, Any]) -> Any:
    # Mirrors GET /api/projects so clients can DISCOVER project ids to use
    # with the plan tools (ids only ever leave as ids).
    return _api(ctx, "GET", "/api/projects")


def _exec_plan_add(ctx: ToolContext, args: Dict[str, Any]) -> Any:
    body = {"op": "add"}
    for k in ("title", "description", "details", "test_strategy",
              "priority", "deps", "parent"):
        if args.get(k) is not None:
            body[k] = args[k]
    return _plans_api(ctx, str(args["project"]), body=body)


def _exec_plan_update(ctx: ToolContext, args: Dict[str, Any]) -> Any:
    body = {"op": "update", "task_id": args["task_id"]}
    for k in ("title", "description", "details", "test_strategy",
              "priority"):
        if args.get(k) is not None:
            body[k] = args[k]
    return _plans_api(ctx, str(args["project"]), body=body)


def _exec_plan_set_status(ctx: ToolContext, args: Dict[str, Any]) -> Any:
    body = {"op": "set-status", "task_id": args["task_id"],
            "status": args["status"]}
    if args.get("note") is not None:
        body["note"] = args["note"]
    return _plans_api(ctx, str(args["project"]), body=body)


def _exec_plan_depends(ctx: ToolContext, args: Dict[str, Any]) -> Any:
    body = {"op": "depends", "task_id": args["task_id"],
            "dep_id": args["dep_id"]}
    if args.get("remove"):
        body["remove"] = True
    return _plans_api(ctx, str(args["project"]), body=body)


def _exec_plan_import(ctx: ToolContext, args: Dict[str, Any]) -> Any:
    body = {"op": "import", "text": args["text"],
            "use_ai": bool(args.get("use_ai"))}
    if args.get("num_tasks") is not None:
        body["num_tasks"] = int(args["num_tasks"])
    return _plans_api(ctx, str(args["project"]), body=body)


def _exec_plan_expand(ctx: ToolContext, args: Dict[str, Any]) -> Any:
    body = {"op": "expand", "task_id": args["task_id"]}
    if args.get("num") is not None:
        body["num"] = int(args["num"])
    if args.get("prompt") is not None:
        body["prompt"] = args["prompt"]
    return _plans_api(ctx, str(args["project"]), body=body)


def _exec_plan_loop_start(ctx: ToolContext, args: Dict[str, Any]) -> Any:
    body: Dict[str, Any] = {"op": "loop-start"}
    if args.get("max_tasks") is not None:
        body["max_tasks"] = int(args["max_tasks"])
    if args.get("resume"):
        body["resume"] = True
    return _plans_api(ctx, str(args["project"]), body=body)


def _exec_plan_loop_cancel(ctx: ToolContext, args: Dict[str, Any]) -> Any:
    return _plans_api(ctx, str(args["project"]),
                      body={"op": "loop-cancel"})


def _require_loopback_url(url: str) -> None:
    """Refuse any attach target whose host is not loopback."""
    from urllib.parse import urlparse

    try:
        u = urlparse(url)
    except ValueError as e:
        raise ToolError(f"invalid attach URL: {e}") from e
    if u.scheme not in ("http", "https") or not u.hostname:
        raise ToolError(f"invalid attach URL {url!r} (need http(s)://host:port)")
    host = u.hostname.strip("[]")
    loopback = host in ("127.0.0.1", "localhost", "::1") or \
        host.startswith("127.")
    if not loopback:
        raise ToolError(f"attach target {url!r} is not loopback; toaster_loop "
                        "never talks to non-loopback endpoints")


# --------------------------------------------------------------- registry

def _obj(props: Dict[str, Any],
         required: Optional[List[str]] = None) -> Dict[str, Any]:
    """Fixed-object schema fragment: closed properties, no free-form bags."""
    s: Dict[str, Any] = {
        "type": "object",
        "properties": props,
        "additionalProperties": False,
    }
    if required:
        s["required"] = list(required)
    return s


#: The optional confirmation argument every MUTATING tool's schema carries.
_CONFIRM_ARG = {
    "confirm_token": {
        "type": "string",
        "description": ("Opaque token from this tool's PLAN response; pass it "
                        "back unchanged to execute the plan (single use)."),
    },
}

_ID_SCHEMA = {
    "type": "string",
    "pattern": "^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$",
    "description": "Managed instance id (see instances_list).",
}


def build_tools() -> List[Tool]:
    """The full tool table. Fresh list per server; monkeypatchable in tests."""
    return [
        # ------------------------------------------------------------ read
        Tool(
            name="status",
            description=("toaster_loop overview: version, home, owned-server state, "
                         "buffer stats, active model, job counts (mirrors the "
                         "dashboard /api/status and `instances status` chips)."),
            input_schema=_obj({}),
            mutating=False,
            executor=_exec_status,
        ),
        Tool(
            name="instances_list",
            description=("Every managed serving instance with its load state "
                         "(loaded|unloaded|unloading|stopped) — mirrors "
                         "`toaster-loop instances status`."),
            input_schema=_obj({}),
            mutating=False,
            executor=_exec_instances_list,
        ),
        Tool(
            name="models_list",
            description=("Model registry rows + lineage + active pointer — "
                         "mirrors `toaster-loop models list`."),
            input_schema=_obj({}),
            mutating=False,
            executor=_exec_models_list,
        ),
        Tool(
            name="logs_tail",
            description=(f"Bounded tail (max {LOGS_TAIL_MAX_LINES} lines) of "
                         "one instance's server log under <home>/logs. Only "
                         "instance ids are accepted — never paths."),
            input_schema=_obj({
                "instance_id": _ID_SCHEMA,
                "lines": {"type": "integer", "minimum": 1,
                          "maximum": LOGS_TAIL_MAX_LINES,
                          "description": f"Default {LOGS_TAIL_DEFAULT_LINES}."},
            }, required=["instance_id"]),
            mutating=False,
            executor=_exec_logs_tail,
        ),
        # -------------------------------------------------------- mutating
        Tool(
            name="instance_start",
            description=("Start (spawn or attach) one managed llama-server "
                         "instance — mirrors POST /api/servers/<id>/start. "
                         "Plan-then-confirm: first call returns a plan."),
            input_schema=_obj({**{
                "id": _ID_SCHEMA,
                "port": {"type": "integer", "minimum": 0, "maximum": 65535,
                         "description": "0 = dynamic (default)."},
                "context": {"type": "integer", "minimum": 1,
                            "description": "Context length override (-c)."},
                "force": {"type": "boolean",
                          "description": "Stop a running instance first."},
            }, **_CONFIRM_ARG}, required=["id"]),
            mutating=True,
            executor=_exec_instance_start,
            effect="Launches a llama-server process for this instance on "
                   "127.0.0.1 and records its state.",
            risks=["Occupies GPU/VRAM while running",
                   "may stop an existing server when force=true"],
            target_arg="id",
        ),
        Tool(
            name="instance_stop",
            description=("Stop one managed instance (only processes toaster_loop "
                         "owns are ever stopped) — mirrors POST "
                         "/api/servers/<id>/stop. Plan-then-confirm."),
            input_schema=_obj({**{"id": _ID_SCHEMA}, **_CONFIRM_ARG},
                              required=["id"]),
            mutating=True,
            executor=_exec_instance_stop,
            effect="Gracefully stops the instance's llama-server process "
                   "(never a foreign process).",
            risks=["In-flight requests on this endpoint are interrupted"],
            target_arg="id",
        ),
        Tool(
            name="instance_load",
            description=("(Re)launch one instance via the WP5 coexistence "
                         "engine — mirrors POST /api/servers/<id>/load. "
                         "Plan-then-confirm."),
            input_schema=_obj({**{"id": _ID_SCHEMA}, **_CONFIRM_ARG},
                              required=["id"]),
            mutating=True,
            executor=_exec_instance_load,
            effect="Loads (launches) the instance through the coexistence "
                   "engine's canonical spawn path.",
            risks=["Occupies GPU/VRAM; may evict per coexist_policy"],
            target_arg="id",
        ),
        Tool(
            name="instance_unload",
            description=("Gracefully stop one instance and mark it unloaded "
                         "(lazy_load may restart it) — mirrors POST "
                         "/api/servers/<id>/unload. Plan-then-confirm."),
            input_schema=_obj({**{"id": _ID_SCHEMA}, **_CONFIRM_ARG},
                              required=["id"]),
            mutating=True,
            executor=_exec_instance_unload,
            effect="Stops the instance and records load_state=unloaded.",
            risks=["In-flight requests cause a retryable conflict refusal"],
            target_arg="id",
        ),
        Tool(
            name="curate_enqueue",
            description=("Queue a curation job over the experience buffer — "
                         "mirrors `toaster-loop curate` / POST /api/curate. The "
                         "job runs in-process (no GPU slot). "
                         "Plan-then-confirm."),
            input_schema=_obj({**{
                "name": {"type": "string",
                         "description": "Dataset id suffix."},
                "min_score": {"type": "number",
                              "description": "Minimum effective score."},
                "since": {"type": "string",
                          "description": "ISO8601 lower bound."},
                "profile": {"type": "string"},
                # WP12: project id filter (mirrors `toaster-loop curate --project`);
                # the dashboard handler already honors body["project"].
                "project": {"type": "string",
                            "description": ("Only curate traces stamped with "
                                            "this project id (see "
                                            "projects_list).")},
                "model_alias": {"type": "string"},
                "include_failures": {"type": "boolean"},
                "limit": {"type": "integer", "minimum": 1},
                "formats": {"type": "string",
                            "enum": ["sft", "dpo", "both"]},
                "rebuild": {"type": "string",
                            "description": "Dataset id to re-curate."},
            }, **_CONFIRM_ARG}),
            mutating=True,
            executor=_exec_curate_enqueue,
            effect="Queues a curate job: claims matching traces (curated=1) "
                   "and writes a dataset directory under <home>/datasets/.",
            risks=["Claimed traces are excluded from later curation runs"],
        ),
        Tool(
            name="train_enqueue",
            description=("Queue an SFT/DPO/GRPO training job for a curated "
                         "dataset — mirrors `toaster-loop train start` / POST "
                         "/api/train. Single-flight: while one job runs this "
                         "returns an error (HTTP 409 semantics), it never "
                         "queues silently. Plan-then-confirm."),
            input_schema=_obj({**{
                "dataset": {"type": "string",
                            "description": "Curated dataset id."},
                "kind": {"type": "string", "enum": ["sft", "dpo", "grpo"]},
                "dry_run": {"type": "boolean",
                            "description": "Validate without GPU."},
                "continue_from": {
                    "type": "string",
                    "description": ("Continual training: registry model id "
                                    "(or adapter directory path) to build on "
                                    "instead of starting fresh from the base "
                                    "model.")},
            }, **_CONFIRM_ARG}, required=["dataset"]),
            mutating=True,
            executor=_exec_train_enqueue,
            effect="Queues a training job (kind selected) for the dataset; "
                   "starts when the GPU frees.",
            risks=["Occupies the single GPU slot while running",
                   "refused with a conflict error if another job runs"],
            target_arg="dataset",
        ),
        Tool(
            name="validate",
            description=("Replay the golden validation suite against one "
                         "registry model's GGUF (same gate `toaster-loop iterate` "
                         "uses). Classified MUTATING: it spawns a temporary "
                         "llama-server and occupies the GPU, though it writes "
                         "no registry state. Plan-then-confirm."),
            input_schema=_obj({**{
                "model_id": {"type": "string",
                             "description": "Registry model id to validate."},
            }, **_CONFIRM_ARG}, required=["model_id"]),
            mutating=True,
            executor=_exec_validate,
            effect="Spawns a temporary llama-server for the model's GGUF and "
                   "replays golden tasks; no registry state is written.",
            risks=["Occupies GPU while running",
                   "may conflict with a running server or training job"],
            target_arg="model_id",
        ),
        Tool(
            name="promote",
            description=("Promote a validated model to the active pointer — "
                         "mirrors `toaster-loop models promote` / POST "
                         "/api/models/promote. Plan-then-confirm."),
            input_schema=_obj({**{
                "model_id": {"type": "string"},
                "intent": {"type": "string",
                           "description": "One-liner recorded on the ledger."},
            }, **_CONFIRM_ARG}, required=["model_id"]),
            mutating=True,
            executor=_exec_promote,
            effect="Flips the registry's active model pointer (previous "
                   "model superseded; hash-chained ledger entry).",
            risks=["serve --auto / deploy will pick up the new model"],
            target_arg="model_id",
        ),
        Tool(
            name="bench",
            description=("Measure prompt/generation tokens/sec (WP1 harness; "
                         "mirrors `toaster-loop bench` measure mode). May spawn a "
                         "server and persists a record under <home>/bench/. "
                         "Plan-then-confirm."),
            input_schema=_obj({**{
                "runs": {"type": "integer", "minimum": 1, "maximum": 50},
                "max_tokens": {"type": "integer", "minimum": 1,
                               "maximum": 4096},
                "label": {"type": "string",
                          "description": "Label for the persisted record."},
                "noise_floor": {"type": "boolean",
                                "description": "Record as noise-floor "
                                               "baseline."},
                "attach": {"type": "string",
                           "description": "Loopback-only attach URL; default "
                                          "is the configured/spawned server."},
            }, **_CONFIRM_ARG}),
            mutating=True,
            executor=_exec_bench,
            effect="Runs N temperature-0 measurement passes and persists a "
                   "bench record under <home>/bench/.",
            risks=["May spawn llama-server (GPU) if none is running"],
        ),
        # ------------------------------------------------------- plan tools
        # WP12 mirror of /api/plans (family="plans": listed ONLY while
        # config plans.enabled — the server filters + refuses otherwise).
        Tool(
            name="projects_list",
            description=("Every project id/name/workspace (mirrors GET "
                         "/api/projects) — discover ids for the plan tools."),
            input_schema=_obj({}),
            mutating=False,
            executor=_exec_projects_list,
            family="plans",
        ),
        Tool(
            name="plans_list",
            description=("One project's task graph: summary counts, every "
                         "task (status/priority/deps/labels), the cycle/"
                         "dangling validate report, and loop state (mirrors "
                         "GET /api/plans/<project>)."),
            input_schema=_obj({"project": _PROJECT_ID_SCHEMA},
                              required=["project"]),
            mutating=False,
            executor=_exec_plans_list,
            family="plans",
        ),
        Tool(
            name="plan_loop_status",
            description=("Autonomous-loop state for a project: running flag "
                         "+ the iteration/task progress record (mirrors GET "
                         "/api/plans/<project> loop field)."),
            input_schema=_obj({"project": _PROJECT_ID_SCHEMA},
                              required=["project"]),
            mutating=False,
            executor=_exec_plan_loop_status,
            family="plans",
        ),
        Tool(
            name="plan_add",
            description=("Add a task to a project's persistent plan (mirrors "
                         "POST /api/plans/<project> op=add). "
                         "Plan-then-confirm."),
            input_schema=_obj({**{
                "project": _PROJECT_ID_SCHEMA,
                "title": {"type": "string", "description": "Task title."},
                "description": {"type": "string"},
                "details": {"type": "string"},
                "test_strategy": {"type": "string"},
                "priority": {"type": "string",
                             "enum": ["high", "medium", "low"]},
                "deps": _DEPS_ARG,
                "parent": _TASK_ID_SCHEMA,
            }, **_CONFIRM_ARG}, required=["project", "title"]),
            mutating=True,
            executor=_exec_plan_add,
            effect="Appends one task to the project's persistent plan graph.",
            risks=["Agents and loops may later execute this task under "
                   "their profile grants"],
            target_arg="project",
            family="plans",
        ),
        Tool(
            name="plan_update",
            description=("Edit a task's title/description/details/"
                         "test_strategy/priority (mirrors op=update). "
                         "Plan-then-confirm."),
            input_schema=_obj({**{
                "project": _PROJECT_ID_SCHEMA,
                "task_id": _TASK_ID_SCHEMA,
                "title": {"type": "string"},
                "description": {"type": "string"},
                "details": {"type": "string"},
                "test_strategy": {"type": "string"},
                "priority": {"type": "string",
                             "enum": ["high", "medium", "low"]},
            }, **_CONFIRM_ARG}, required=["project", "task_id"]),
            mutating=True,
            executor=_exec_plan_update,
            effect="Overwrites editable fields of one plan task.",
            risks=["Changes what future runs/loops are told to do"],
            target_arg="project",
            family="plans",
        ),
        Tool(
            name="plan_set_status",
            description=("Set a task's status (pending|in_progress|blocked|"
                         "done|cancelled; done cascades to subtasks) — "
                         "mirrors op=set-status. Plan-then-confirm."),
            input_schema=_obj({**{
                "project": _PROJECT_ID_SCHEMA,
                "task_id": _TASK_ID_SCHEMA,
                "status": {"type": "string",
                           "enum": ["pending", "in_progress", "blocked",
                                    "done", "cancelled"]},
                "note": {"type": "string",
                         "description": "Timestamped line appended to the "
                                        "task details."},
            }, **_CONFIRM_ARG}, required=["project", "task_id", "status"]),
            mutating=True,
            executor=_exec_plan_set_status,
            effect="Flips one task's status (done cascades to its subtasks) "
                   "and unblocks/blocks dependents.",
            risks=["Marking done releases dependent tasks for execution"],
            target_arg="project",
            family="plans",
        ),
        Tool(
            name="plan_depends",
            description=("Add (or with remove=true drop) a dependency edge "
                         "task->dep; cycles and self-deps are refused — "
                         "mirrors op=depends. Plan-then-confirm."),
            input_schema=_obj({**{
                "project": _PROJECT_ID_SCHEMA,
                "task_id": _TASK_ID_SCHEMA,
                "dep_id": _TASK_ID_SCHEMA,
                "remove": {"type": "boolean"},
            }, **_CONFIRM_ARG},
                required=["project", "task_id", "dep_id"]),
            mutating=True,
            executor=_exec_plan_depends,
            effect="Wires (or removes) one dependency edge in the graph.",
            risks=["Reorders what agents/loops may run next"],
            target_arg="project",
            family="plans",
        ),
        Tool(
            name="plan_import",
            description=("Import a brief into the plan: deterministic line "
                         "format by default, use_ai=true asks the served "
                         "model for a structured task list (mirrors "
                         "op=import). Plan-then-confirm."),
            input_schema=_obj({**{
                "project": _PROJECT_ID_SCHEMA,
                "text": {"type": "string",
                         "description": "Pasted brief (plain text lines or "
                                        "a JSON task array)."},
                "use_ai": {"type": "boolean",
                           "description": "Structure the brief with the "
                                          "served model (one bounded call)."},
                "num_tasks": {"type": "integer", "minimum": 1,
                              "maximum": 32},
            }, **_CONFIRM_ARG}, required=["project", "text"]),
            mutating=True,
            executor=_exec_plan_import,
            effect="Appends N parsed/generated tasks to the project plan.",
            risks=["use_ai=true occupies the model/GPU for one bounded call"],
            target_arg="project",
            family="plans",
        ),
        Tool(
            name="plan_expand",
            description=("Decompose one task into subtasks with the served "
                         "model (parent set; parent status untouched; "
                         "mirrors op=expand). Plan-then-confirm."),
            input_schema=_obj({**{
                "project": _PROJECT_ID_SCHEMA,
                "task_id": _TASK_ID_SCHEMA,
                "num": {"type": "integer", "minimum": 1, "maximum": 12},
                "prompt": {"type": "string",
                           "description": "Extra guidance for the model."},
            }, **_CONFIRM_ARG}, required=["project", "task_id"]),
            mutating=True,
            executor=_exec_plan_expand,
            effect="Appends generated subtasks (parent=task) to the plan.",
            risks=["Occupies the model/GPU for one bounded call"],
            target_arg="project",
            family="plans",
        ),
        Tool(
            name="plan_loop_start",
            description=("Start the guarded autonomous loop over a project's "
                         "plan (mirrors op=loop-start). Single-flight: a "
                         "second start returns a conflict error. Tasks close "
                         "only when the agent explicitly marks them done. "
                         "Plan-then-confirm."),
            input_schema=_obj({**{
                "project": _PROJECT_ID_SCHEMA,
                "max_tasks": {"type": "integer", "minimum": 1,
                              "maximum": 50,
                              "description": "Overrides plans.loop.max_tasks "
                                             "for this run."},
                "resume": {"type": "boolean",
                           "description": "Continue from the progress file."},
            }, **_CONFIRM_ARG}, required=["project"]),
            mutating=True,
            executor=_exec_plan_loop_start,
            effect="Autonomously runs agents that write files in the "
                   "project workspace under that profile's grants, one "
                   "unblocked task at a time.",
            risks=["Agents execute with the profile's tool subset (no "
                   "permission bypass); GPU busy until done/blocked/cancel"],
            target_arg="project",
            family="plans",
        ),
        Tool(
            name="plan_loop_cancel",
            description=("Cancel a project's running autonomous loop at the "
                         "current task boundary (mirrors op=loop-cancel). "
                         "Plan-then-confirm."),
            input_schema=_obj({**{"project": _PROJECT_ID_SCHEMA},
                               **_CONFIRM_ARG}, required=["project"]),
            mutating=True,
            executor=_exec_plan_loop_cancel,
            effect="Requests cancellation of the running plan loop.",
            risks=["The in-flight task run finishes or stays in_progress "
                   "(no forced kill)"],
            target_arg="project",
            family="plans",
        ),
    ]

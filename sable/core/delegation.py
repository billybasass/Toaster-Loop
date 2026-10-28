"""Agent discovery + delegation (feature-wave Phase 2).

Lets one agent profile hand a subtask to another profile on the SAME project.
Response shapes (success / timeout-with-diagnostics / error payloads) follow
modeled on the delegation tool contract in
spirit; the execution model follows sable's own synchronous tool loop.

Design decision — SYNC WITH TIMEOUT (see docs/DECISIONS.md D-11):
sable executes tools synchronously inside ``SableAgentClient.run()``
(``content = impl(**args)`` blocks the loop), so an async enqueue/poll model
would add a background scheduler for no benefit. ``delegate_task`` therefore
runs the child to completion on a worker thread and waits up to a bounded
timeout; on timeout it returns a structured timeout payload instead of
hanging the parent. Delegation records still live in an in-process registry,
so ``check_delegation`` can poll status/result of any delegation made from
this process (and re-reads terminal state after the child finished).

Execution reuse: the child runs through ``core.agent_runner.create_runner`` —
the target profile's own system prompt / tool subset / workspace override —
so guardrails, compaction, spill and trace recording behave exactly like a
normal run. The child gets its own trace whose header metadata links
``delegation_id``, ``parent_run_id`` and the project (the parent's trace id
does not exist yet while the parent is mid-run — traces are recorded
post-run — so the linkage key is the parent run id; see D-11).

Lineage: when the tool executes with thread context (dashboard chat passes
the current thread id), an entry {agent, delegation_id, status, created} is
appended to the thread JSON's ``delegations`` list at <home>/chat/*.json —
the shape Phase 1's explorer-tree lineage chips already tolerate. Without
thread context (CLI) lineage is skipped and everything else still works.

Fault posture mirrors the rest of core: tools never raise into the loop;
errors come back as JSON payloads with an ``error`` field (which
core/failure_types.classify_tool_result correctly scores as a failure).
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

MAX_RESULT_CHARS = 8_000        # child answer kept in the payload
MIN_TIMEOUT_S = 0.1             # floor keeps a positive wait (tests use short ones)
MAX_TIMEOUT_S = 3_600.0
DELEGATION_BLOCKED_TOOLS_NOTE = "nested delegation is not allowed (depth limit 1)"

_THREAD_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# In-process registries (one dashboard/CLI process owns its own runs)
# ---------------------------------------------------------------------------

_registry_lock = threading.Lock()
#: run_id -> {profile, project, started} for every in-flight runner run
#: (populated by SableAgentClient.run when a delegation context exists).
_active_runs: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
#: delegation_id -> record dict, bounded FIFO (check_delegation source).
_delegations: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_DELEGATIONS_CAP = 500


def register_run(profile: Optional[str], project: Optional[str] = None,
                 run_id: Optional[str] = None) -> str:
    """Mark a runner execution in-flight; returns the registry key."""
    key = run_id or uuid.uuid4().hex[:12]
    with _registry_lock:
        _active_runs[key] = {"profile": profile, "project": project,
                             "started": _utcnow()}
    return key


def unregister_run(key: str) -> None:
    with _registry_lock:
        _active_runs.pop(key, None)


def active_runs() -> List[Dict[str, Any]]:
    """Snapshot of in-flight runs (list of {profile, project, started})."""
    with _registry_lock:
        return [dict(r) for r in _active_runs.values()]


def running_count_for(profile: str) -> int:
    with _registry_lock:
        return sum(1 for r in _active_runs.values() if r.get("profile") == profile)


def _store_delegation(record: Dict[str, Any]) -> None:
    with _registry_lock:
        _delegations[record["delegation_id"]] = record
        while len(_delegations) > _DELEGATIONS_CAP:
            _delegations.pop(next(iter(_delegations)))


def get_delegation(delegation_id: str) -> Optional[Dict[str, Any]]:
    with _registry_lock:
        rec = _delegations.get(delegation_id)
        return dict(rec) if rec else None


def reset_state() -> None:
    """Clear registries (tests)."""
    with _registry_lock:
        _active_runs.clear()
        _delegations.clear()


# ---------------------------------------------------------------------------
# Profiles / projects lookup (config + <home>/projects/*.json)
# ---------------------------------------------------------------------------

def _profiles(cfg: Any) -> List[Dict[str, Any]]:
    profs = getattr(cfg, "profiles", None) or []
    return [p for p in profs if isinstance(p, dict)]


def find_profile(cfg: Any, name: str) -> Optional[Dict[str, Any]]:
    for p in _profiles(cfg):
        if p.get("name") == name:
            return p
    return None


def profile_description(prof: Dict[str, Any]) -> str:
    desc = str(prof.get("description") or "").strip()
    if desc:
        return desc.splitlines()[0][:160]
    prompt = str(prof.get("system_prompt") or "").strip()
    return prompt.splitlines()[0][:160] if prompt else "(no description)"


def load_project(cfg: Any, project_id: str) -> Optional[Dict[str, Any]]:
    """Read <home>/projects/<id>.json (same file layout as the dashboard)."""
    if not project_id or not _THREAD_ID_RE.match(project_id):
        return None
    from ..store.paths import resolve_sable_home

    p = resolve_sable_home(cfg) / "projects" / f"{project_id}.json"
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def project_agent_names(cfg: Any, project_id: Optional[str]) -> List[str]:
    """Agents attached to the active project; all profiles when no project."""
    if project_id:
        proj = load_project(cfg, project_id) or {}
        agents = proj.get("agents")
        return [str(a) for a in agents] if isinstance(agents, list) else []
    return [str(p.get("name")) for p in _profiles(cfg) if p.get("name")]


# ---------------------------------------------------------------------------
# Thread lineage (<home>/chat/<thread>.json "delegations" list)
# ---------------------------------------------------------------------------

def _thread_path(cfg: Any, thread_id: str) -> Optional[Path]:
    if not thread_id or not _THREAD_ID_RE.match(thread_id):
        return None
    from ..store.paths import resolve_sable_home

    return resolve_sable_home(cfg) / "chat" / f"{thread_id}.json"


def thread_record_delegation(cfg: Any, thread_id: Optional[str],
                             record: Dict[str, Any]) -> None:
    """Append/update the delegation entry on the thread JSON. Best-effort:
    lineage failures never break a delegation (or the agent loop)."""
    p = _thread_path(cfg, thread_id or "")
    if p is None:
        return  # no/invalid thread context (e.g. CLI) -> lineage skipped
    try:
        thread: Dict[str, Any] = {}
        if p.is_file():
            try:
                loaded = json.loads(p.read_text())
                if isinstance(loaded, dict):
                    thread = loaded
            except (OSError, json.JSONDecodeError):
                thread = {}
        entry = {
            "agent": record.get("agent"),
            "delegation_id": record.get("delegation_id"),
            "status": record.get("status"),
            "created": record.get("created"),
        }
        if record.get("child_trace_id"):
            entry["trace_id"] = record["child_trace_id"]
        delegs = thread.get("delegations")
        if not isinstance(delegs, list):
            delegs = []
        for i, d in enumerate(delegs):
            if isinstance(d, dict) and d.get("delegation_id") == entry["delegation_id"]:
                delegs[i] = entry
                break
        else:
            delegs.append(entry)
        thread.setdefault("id", p.stem)
        thread.setdefault("created", record.get("created"))
        thread["delegations"] = delegs
        thread["updated"] = _utcnow()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(thread, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        os.replace(tmp, p)
    except Exception:  # lineage is observability, never fatal
        pass


# ---------------------------------------------------------------------------
# Payload helpers (reference-shaped responses; JSON strings for the tool result)
# ---------------------------------------------------------------------------

def _payload(**fields: Any) -> str:
    return json.dumps(fields, ensure_ascii=False, default=str)


def _error_payload(message: str, **extra: Any) -> str:
    return _payload(status="failed", error=message, **extra)


# ---------------------------------------------------------------------------
# delegate_task execution
# ---------------------------------------------------------------------------

def _resolve_timeout(cfg: Any, timeout_seconds: Any) -> float:
    raw = timeout_seconds
    if raw is None:
        raw = getattr(cfg, "delegation_timeout_seconds", 300)
    try:
        val = float(raw)
    except (TypeError, ValueError):
        val = 300.0
    return max(MIN_TIMEOUT_S, min(val, MAX_TIMEOUT_S))


def _run_child(cfg: Any, prof: Dict[str, Any], task: str, ctx: Dict[str, Any],
               delegation_id: str, out: Dict[str, Any],
               plan_task: Optional[str] = None) -> None:
    """Execute the child run in-place (called on the worker thread). Reuses
    create_runner so guardrails/compaction/recording are identical to a
    normal run. Results land in `out` (the caller's result box) so a timed-out
    parent never races with a still-running child's registry record.

    ``plan_task`` (WP12): task id whose prompt text was already resolved by
    execute_delegation; it stamps the child trace (additive plan_task field)
    so the traces panel links child runs to graph nodes."""
    from .agent_runner import create_runner
    from ..store.recorder import make_recorder

    base_url = ctx.get("base_url") or ""
    model = prof.get("model_name") or ctx.get("model") or ""
    api_key = ctx.get("api_key") if "api_key" in ctx \
        else getattr(cfg, "attach_api_key", None)
    project_id = ctx.get("project")
    proj = load_project(cfg, project_id) if project_id else None

    child_ctx: Dict[str, Any] = {
        "cfg": cfg, "base_url": base_url, "model": model,
        "profile": prof.get("name"), "project": project_id,
        "thread": None,                 # child never writes thread lineage itself
        "run_id": delegation_id,
        "depth": int(ctx.get("depth", 0)) + 1,
    }
    metadata = {
        "delegation_id": delegation_id,
        # Parent trace does not exist while the parent runs (recorded
        # post-run) — link by run id instead; see DECISIONS D-11.
        "parent_run_id": ctx.get("run_id"),
        "parent_trace_id": ctx.get("trace_id"),
        "delegated_from": ctx.get("profile") or "default",
    }
    recorder = make_recorder(cfg, project=project_id, metadata=metadata,
                             plan_task=plan_task)
    tools = prof.get("tools")
    runner = create_runner(
        cfg, base_url, model, api_key=api_key,
        recorder=recorder,
        system_prompt=str(prof.get("system_prompt") or "") or None,
        enabled_tool_names=list(tools) if isinstance(tools, list) else None,
        workspace_root_override=prof.get("workspace_root"),
        project_workspace_dir=(proj or {}).get("workspace_dir"),
        delegation_context=child_ctx,
        # Projects: the child sees the same '## Project' section as its
        # parent (description + full agent roster with roles), so a
        # delegated agent knows the project and every teammate's job.
        project_context=proj,
        # Per-agent capabilities: the CHILD profile governs its own reach-out
        # (merged over global agent.capabilities inside create_runner).
        capabilities=prof.get("capabilities"),
        # Phase 5: child inherits the parent's project for index routing.
        project_id=project_id,
        # Pass 4 #38: the child profile selects its playbooks (no-op unless
        # playbooks.enabled).
        profile_name=prof.get("name"))
    res = runner.run(task)
    out["child_trace_id"] = res.trace_id
    out["completed"] = bool(res.completed)
    out["failure_reason"] = res.failure_reason
    out["result"] = (res.final_response or "")[:MAX_RESULT_CHARS]


def execute_delegation(agent: str, task: str, context: str = "",
                       timeout_seconds: Any = None,
                       plan_task: Optional[str] = None,
                       _ctx: Optional[Dict[str, Any]] = None) -> str:
    """Core of the delegate_task tool. Returns a JSON payload string.

    ``plan_task`` (WP12, optional): id of a task in the active project's
    persistent plan; when set the child prompt is built from that task with
    build_task_prompt (title/description/details/test-strategy + completed-
    dependency context) and the child trace is stamped with plan_task. The
    free-text ``task`` still rides along as extra instruction."""
    ctx = dict(_ctx or {})
    cfg = ctx.get("cfg")
    if cfg is None:
        return _error_payload("delegation unavailable: no configuration context "
                              "(runner built without create_runner)")
    if not getattr(cfg, "delegation_enabled", True):
        return _error_payload("delegation disabled (config delegation.enabled=false)")
    if int(ctx.get("depth", 0)) >= 1:
        return _error_payload(DELEGATION_BLOCKED_TOOLS_NOTE)
    agent = str(agent or "").strip()
    task = str(task or "").strip()
    if not agent:
        return _error_payload("agent is required (target profile name)")
    if not task:
        return _error_payload("task is required (non-empty subtask description)")

    # Target must be an agent of the active project (or any profile when no
    # project is set) AND exist in the profile list.
    allowed = project_agent_names(cfg, ctx.get("project"))
    prof = find_profile(cfg, agent)
    if prof is None:
        known = ", ".join(allowed) or "(none configured)"
        return _error_payload(f"unknown agent {agent!r}; available: {known}",
                              agent=agent)
    if ctx.get("project") and agent not in allowed:
        return _error_payload(
            f"agent {agent!r} is not attached to the active project; "
            f"available: {', '.join(allowed) or '(none)'}", agent=agent)
    parent_profile = ctx.get("profile")
    if parent_profile and agent == parent_profile:
        return _error_payload(f"cannot delegate to self (profile {agent!r} is "
                              "already running this task)", agent=agent)

    timeout = _resolve_timeout(cfg, timeout_seconds)
    record: Dict[str, Any] = {
        "delegation_id": uuid.uuid4().hex[:12],
        "agent": agent,
        "task": task[:300],
        "status": "running",
        "created": _utcnow(),
        "project": ctx.get("project"),
        "thread": ctx.get("thread"),
        "parent_run_id": ctx.get("run_id"),
        "timeout_seconds": timeout,
        "started_mono": time.monotonic(),
        "result": None,
        "child_trace_id": None,
        "error": None,
    }
    _store_delegation(record)

    def _finish(payload_fields: Dict[str, Any]) -> str:
        record["updated"] = _utcnow()
        thread_record_delegation(cfg, record.get("thread"), record)
        cb = ctx.get("on_update")
        if callable(cb):
            try:
                cb({k: record.get(k) for k in
                    ("delegation_id", "agent", "status", "project", "thread",
                     "child_trace_id", "parent_run_id")})
            except Exception:  # observers never break the delegation
                pass
        return _payload(**payload_fields)

    # Lineage entry appears as soon as the child starts (status running).
    thread_record_delegation(cfg, record.get("thread"), record)

    full_task = task if not context else f"{task}\n\nContext:\n{context}"
    # WP12: a plan-task id upgrades the child prompt to the full task graph
    # context (bounded); the free-text instruction stays appended. A bad/
    # unknown id degrades to the plain prompt with a note — delegation must
    # never fail just because the graph moved under us.
    if plan_task:
        try:
            from .plans import build_task_prompt, load_plan

            pstore = load_plan(ctx.get("project"))
            full_task = (build_task_prompt(pstore, str(plan_task))
                         + f"\n\nExtra instruction:\n{full_task}")
        except Exception as e:  # noqa: BLE001 - degrade, never break delegation
            full_task = (f"{full_task}\n\n(Note: plan task {plan_task!r} "
                         f"could not be expanded: {e})")
    box: Dict[str, Any] = {}

    def _worker() -> None:
        try:
            _run_child(cfg, prof, full_task, ctx, record["delegation_id"],
                       box, plan_task=str(plan_task or "") or None)
        except Exception as e:  # child blew up; surfaced below as failed
            box["exception"] = f"{type(e).__name__}: {e}"

    executor = ThreadPoolExecutor(max_workers=1,
                                  thread_name_prefix=f"delegate-{record['delegation_id']}")
    future = executor.submit(_worker)
    try:
        future.result(timeout=timeout)
    except FuturesTimeoutError:
        elapsed = round(time.monotonic() - record["started_mono"], 2)
        record["status"] = "timed_out"
        record["error"] = (f"child agent {agent!r} timed out after {timeout}s "
                           "without returning")
        # Daemon-style abandonment: never wait on a wedged child thread.
        executor.shutdown(wait=False)
        return _finish({
            "status": "timed_out",
            "delegation_id": record["delegation_id"],
            "agent": agent,
            "error": record["error"],
            "timeout_seconds": timeout,
            "elapsed_s": elapsed,
            "partial_output": None,
            "diagnostics": {
                "child_agent": agent,
                "project": record.get("project"),
                "task_preview": task[:200],
                "note": "child thread abandoned; its trace (if any) is still recorded",
            },
        })
    except Exception as e:
        record["status"] = "failed"
        record["error"] = f"{type(e).__name__}: {e}"
        executor.shutdown(wait=False)
        return _finish({
            "status": "failed",
            "delegation_id": record["delegation_id"],
            "agent": agent,
            "error": record["error"],
        })
    executor.shutdown(wait=False)

    if box.get("exception"):
        record["status"] = "failed"
        record["error"] = box["exception"]
        return _finish({
            "status": "failed",
            "delegation_id": record["delegation_id"],
            "agent": agent,
            "error": box["exception"],
            "child_trace_id": box.get("child_trace_id"),
        })

    record.update({k: box.get(k) for k in
                   ("child_trace_id", "completed", "failure_reason", "result")})
    if box.get("completed"):
        record["status"] = "completed"
        return _finish({
            "status": "completed",
            "delegation_id": record["delegation_id"],
            "agent": agent,
            "result": record.get("result"),
            "child_trace_id": record.get("child_trace_id"),
            "duration_s": round(time.monotonic() - record["started_mono"], 2),
        })
    record["status"] = "failed"
    record["error"] = f"child run did not complete ({box.get('failure_reason') or 'unknown reason'})"
    return _finish({
        "status": "failed",
        "delegation_id": record["delegation_id"],
        "agent": agent,
        "error": record["error"],
        "failure_reason": box.get("failure_reason"),
        "partial_output": record.get("result"),
        "child_trace_id": record.get("child_trace_id"),
    })


def check_delegation_status(delegation_id: str) -> str:
    rec = get_delegation(str(delegation_id or "").strip())
    if rec is None:
        return _error_payload(
            f"unknown delegation id {delegation_id!r} "
            "(ids are tracked in-process for the current session)")
    fields = {
        "status": rec.get("status"),
        "delegation_id": rec.get("delegation_id"),
        "agent": rec.get("agent"),
        "created": rec.get("created"),
    }
    if rec.get("status") in ("completed", "failed", "timed_out"):
        fields.update({
            "result": rec.get("result"),
            "child_trace_id": rec.get("child_trace_id"),
            "error": rec.get("error"),
        })
    return _payload(**fields)


def list_agents_text(_ctx: Optional[Dict[str, Any]] = None) -> str:
    """Core of the list_agents tool: JSON list of candidate agents."""
    ctx = dict(_ctx or {})
    cfg = ctx.get("cfg")
    if cfg is None:
        return _error_payload("delegation unavailable: no configuration context")
    names = project_agent_names(cfg, ctx.get("project"))
    out: List[Dict[str, Any]] = []
    for name in names:
        prof = find_profile(cfg, name) or {"name": name}
        running = running_count_for(name)
        out.append({
            "agent": name,
            "description": profile_description(prof),
            "running": running > 0,
            "running_runs": running,
            "is_self": bool(ctx.get("profile")) and name == ctx.get("profile"),
        })
    return _payload(
        status="ok",
        project=ctx.get("project"),
        scope=("project agents" if ctx.get("project") else "all configured profiles"),
        agents=out)

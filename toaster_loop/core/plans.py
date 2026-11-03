"""WP12 "Plans": persistent per-project task graphs — pure-logic store + DAG.

Zero LLM calls live here: this module is the deterministic core (storage,
validation, dependency math, prompt rendering). The agent tool family
(``core/tools.py``), the dashboard API (``dashboard/api/plans.py``), the CLI
(``toaster_loop plan ...``) and the MCP control plane all call into these functions,
so every surface agrees on exactly one store.

Storage layout::

    <TOASTER_LOOP_HOME>/plans/<project_id>.json      one graph per project
    <TOASTER_LOOP_HOME>/plans/default.json           the no-project graph

Design ideas adapted from the open-source project Taskmaster
(github.com/eyaltoledano/claude-task-master) — inspiration ONLY, re-implemented
inline per D-3/D-12; nothing is imported from it.

Fault posture mirrors the projects API + todo store: atomic tmp+os.replace
writes under a module-level lock; a corrupt file reads back as an EMPTY plan
with one warning (never raises into callers); unknown keys are dropped on
write; ids are strings validated with the same pattern the projects API uses.

Opt-in gate: every caller checks ``plans_config(cfg)["enabled"]`` BEFORE
touching this module; with the default config nothing here is ever called, so
existing behavior stays byte-identical (AGENTS.md cardinal rule #1).
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("toaster_loop.plans")

SCHEMA_VERSION = 1

#: Same pattern as dashboard/api/projects.py::_PROJECT_ID_RE — ids (project and
#: task) are validated identically everywhere they cross a boundary.
_PROJECT_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
TASK_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

TASK_STATUSES = ("pending", "in_progress", "blocked", "done", "cancelled")
TASK_PRIORITIES = ("high", "medium", "low")
_PRIORITY_RANK = {"high": 0, "medium": 1, "low": 2}

#: Statuses that satisfy a dependency (a cancelled dep unblocks work too —
#: the decision was made not to do it; blocking forever would deadlock).
DEPS_SATISFYING = ("done", "cancelled")

#: Bounds keeping plans (and their prompt injections) small.
MAX_TASKS = 256
MAX_TITLE_CHARS = 200
MAX_TEXT_CHARS = 4000
MAX_LABELS = 16
MAX_RUNS_PER_TASK = 20

DEFAULT_PROJECT = "default"

# Module-level lock: one toaster_loop process at a time is the project assumption,
# but dashboard threads + CLI + MCP can interleave inside it. The lock covers
# read-modify-write cycles so two concurrent adds never lose an update.
# Re-entrant: compound writers hold it across load/modify/save (via the public
# ``lock()`` below) and may still call helpers that re-enter. Save itself also
# takes it, so even a caller that forgot the context manager cannot interleave
# two file writes.
_LOCK = threading.RLock()


@contextmanager
def lock():
    """Hold the store lock across a compound load -> modify -> save sequence.

    Every writer that reads a plan and later saves it MUST wrap the whole
    sequence (``with plans.lock(): ...``), otherwise a concurrent writer —
    another dashboard thread, a loop task closing via plan_progress, or the
    runner's finalize linkage — can overwrite the interleaving update (the
    atomic os.replace in save_plan only prevents torn files, not lost
    updates). Re-entrant, so nesting is safe."""
    with _LOCK:
        yield


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def plans_config(cfg: Any) -> Dict[str, Any]:
    """Lenient read of the raw ``plans:`` YAML section (empty => defaults).

    Recognized keys (all optional; documented in config.example.yaml):
      enabled: bool (default False — master switch, OFF = zero behavior change)
      inject_summary: bool (default True — append the bounded plan section)
      max_inject_chars: int (default 800 — hard cap on that section)
      loop: {require_verification: true, max_tasks: 10,
             max_consecutive_failures: 1}   (task runs have no timeout)
    """
    sec = getattr(cfg, "plans", None)
    return dict(sec) if isinstance(sec, dict) else {}


def _int_cfg(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# ------------------------------------------------------------------ paths

def plans_dir(home: Optional[Path] = None) -> Path:
    if home is not None:
        return Path(home) / "plans"
    from ..store.paths import resolve_toaster_loop_home

    return resolve_toaster_loop_home() / "plans"


#: A project id ending in ".json" would collide on disk with the plan file of
#: the same name minus the suffix ("foo.json" and "foo" -> foo.json), silently
#: sharing one graph. Reject it at the single chokepoint instead.
_JSON_SUFFIX_RE = re.compile(r"\.json$", re.I)


def plan_path(project_id: Optional[str], home: Optional[Path] = None) -> Path:
    """File for one project's plan. Invalid/None project => default.json.

    Raises ValueError on a malformed id (callers at API boundaries map that to
    400; internal callers pass ids already validated by the projects layer)."""
    pid = str(project_id or "").strip()
    if not pid:
        return plans_dir(home) / f"{DEFAULT_PROJECT}.json"
    if not _PROJECT_ID_RE.match(pid):
        raise ValueError(f"invalid project id {pid!r}")
    if _JSON_SUFFIX_RE.search(pid):
        raise ValueError(f"invalid project id {pid!r} (may not end in .json)")
    return plans_dir(home) / f"{pid}.json"


# ------------------------------------------------------------- validation

def _clean_text(value: Any, cap: int) -> str:
    return str(value or "").strip()[:cap]


def _validate_task(raw: Any, seen_ids: Optional[set] = None) -> Optional[Dict[str, Any]]:
    """Coerce one task dict to the schema v1 shape; None when unusable.

    Unknown keys are dropped; ids must be strings matching TASK_ID_RE (a
    missing id is minted by the caller). ``seen_ids`` (when given) marks ids
    that already exist so a fresh task colliding with one is rejected."""
    if not isinstance(raw, dict):
        return None
    tid = raw.get("id")
    if not isinstance(tid, str) or not TASK_ID_RE.match(tid.strip()):
        return None
    tid = tid.strip()
    if seen_ids is not None and tid in seen_ids:
        return None
    title = _clean_text(raw.get("title"), MAX_TITLE_CHARS)
    if not title:
        return None
    status = str(raw.get("status") or "pending").strip().lower()
    if status not in TASK_STATUSES:
        status = "pending"
    priority = str(raw.get("priority") or "medium").strip().lower()
    if priority not in TASK_PRIORITIES:
        priority = "medium"
    deps_raw = raw.get("deps")
    deps: List[str] = []
    if isinstance(deps_raw, list):
        for d in deps_raw:
            if isinstance(d, str) and TASK_ID_RE.match(d.strip()) \
                    and d.strip() != tid and d.strip() not in deps:
                deps.append(d.strip())
    parent = raw.get("parent")
    if not (isinstance(parent, str) and TASK_ID_RE.match(parent.strip())):
        parent = None
    else:
        parent = parent.strip()
    labels_raw = raw.get("labels")
    labels: List[str] = []
    if isinstance(labels_raw, list):
        for l in labels_raw[:MAX_LABELS]:
            ls = _clean_text(l, 64)
            if ls and ls not in labels:
                labels.append(ls)
    runs_raw = raw.get("runs")
    runs: List[Dict[str, Any]] = []
    if isinstance(runs_raw, list):
        for r in runs_raw[-MAX_RUNS_PER_TASK:]:
            if not isinstance(r, dict):
                continue
            runs.append({
                "trace_id": _clean_text(r.get("trace_id"), 64) or None,
                "thread": _clean_text(r.get("thread"), 64) or None,
                "at": _clean_text(r.get("at"), 40) or _utcnow(),
            })
    created = _clean_text(raw.get("created"), 40) or _utcnow()
    updated = _clean_text(raw.get("updated"), 40) or created
    return {
        "id": tid,
        "title": title,
        "description": _clean_text(raw.get("description"), MAX_TEXT_CHARS),
        "details": _clean_text(raw.get("details"), MAX_TEXT_CHARS),
        "test_strategy": _clean_text(raw.get("test_strategy"), MAX_TEXT_CHARS),
        "status": status,
        "priority": priority,
        "deps": deps,
        "parent": parent,
        "profile": _clean_text(raw.get("profile"), 64) or None,
        "labels": labels,
        "created": created,
        "updated": updated,
        "runs": runs,
    }


def validate_store(data: Any) -> Dict[str, Any]:
    """Coerce arbitrary loaded JSON into a valid schema-v1 store (never raises).

    Unknown keys dropped at both levels; tasks that fail validation are
    dropped with a warning. Returns a fresh dict."""
    out: Dict[str, Any] = {"version": SCHEMA_VERSION, "project": None,
                           "updated": _utcnow(), "tasks": []}
    if not isinstance(data, dict):
        return out
    proj = data.get("project")
    if isinstance(proj, str) and _PROJECT_ID_RE.match(proj.strip()):
        out["project"] = proj.strip()
    tasks_raw = data.get("tasks")
    if isinstance(tasks_raw, list):
        seen: set = set()
        for raw in tasks_raw[:MAX_TASKS]:
            t = _validate_task(raw, seen_ids=seen)
            if t is None:
                logger.warning("plans: dropping invalid task entry %.80r", raw)
                continue
            seen.add(t["id"])
            out["tasks"].append(t)
        # Dangling parents drop the link (task survives); dangling deps are
        # reported by validate(), kept in the store so the user can fix them.
        ids = {t["id"] for t in out["tasks"]}
        for t in out["tasks"]:
            if t["parent"] is not None and t["parent"] not in ids:
                t["parent"] = None
    return out


# ------------------------------------------------------------------ load/save

def load_plan(project_id: Optional[str] = None, *,
              home: Optional[Path] = None) -> Dict[str, Any]:
    """Read one plan store. Missing/corrupt file => empty store (with warning),
    never raises — the same tolerant posture as the todo/projects stores."""
    try:
        p = plan_path(project_id, home=home)
    except ValueError:
        return validate_store({"project": None})
    if not p.is_file():
        return validate_store({"project": project_id or None})
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("plans: %s unreadable (%s); starting empty", p, e)
        return validate_store({"project": project_id or None})
    store = validate_store(data)
    if store["project"] is None:
        store["project"] = str(project_id) if project_id else None
    return store


def save_plan(store: Dict[str, Any], *, home: Optional[Path] = None) -> None:
    """Atomic tmp + os.replace write of a validated store (raises OSError on
    real I/O failure — callers decide how loud to be).

    Takes the module lock so two writers in this process can never interleave
    file writes even if a caller forgot ``with lock():`` around its
    read-modify-write; the tmp file is mkstemp-unique so concurrent saves
    (this or another process) cannot clobber each other's partial temp."""
    clean = validate_store(store)
    clean["version"] = SCHEMA_VERSION
    clean["updated"] = _utcnow()
    pid = clean.get("project")
    p = plan_path(pid, home=home)
    with _LOCK:
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=".plan-",
                                   suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(clean, f, ensure_ascii=False, indent=2)
            os.replace(tmp, p)
        except OSError:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise


def next_task_id(store: Dict[str, Any]) -> str:
    """Mint the next "t<N>" id (smallest unused)."""
    used = {t["id"] for t in store.get("tasks") or []}
    n = 1
    while f"t{n}" in used:
        n += 1
    return f"t{n}"


# --------------------------------------------------------------- mutations

class PlanError(ValueError):
    """User-facing plan operation failure (bad ids, cycles, unknown tasks)."""


def get_task(store: Dict[str, Any], task_id: str) -> Optional[Dict[str, Any]]:
    for t in store.get("tasks") or []:
        if t["id"] == task_id:
            return t
    return None


def add_task(store: Dict[str, Any], fields: Dict[str, Any]) -> Dict[str, Any]:
    """Add a task to an in-memory store (caller saves). Mints an id when the
    given one is missing; raises PlanError on collision/cap."""
    raw = dict(fields or {})
    if not raw.get("id"):
        raw["id"] = next_task_id(store)
    existing = {t["id"] for t in store.get("tasks") or []}
    task = _validate_task(raw, seen_ids=existing)
    if task is None:
        raise PlanError(f"invalid task (need a unique string id matching "
                        f"{TASK_ID_RE.pattern} and a non-empty title)")
    tasks = store.setdefault("tasks", [])
    if len(tasks) >= MAX_TASKS:
        raise PlanError(f"plan is full ({MAX_TASKS} tasks)")
    # deps referencing unknown ids are allowed here (validate() reports them);
    # self-dep dropped by _validate_task.
    tasks.append(task)
    return task


_UPDATABLE = ("title", "description", "details", "test_strategy", "priority",
              "profile")


def update_task(store: Dict[str, Any], task_id: str,
                fields: Dict[str, Any]) -> Dict[str, Any]:
    """Patch mutable text/priority fields of one task (status goes through
    set_status; deps through add_dependency/remove_dependency)."""
    task = get_task(store, task_id)
    if task is None:
        raise PlanError(f"no such task {task_id!r}")
    for k in _UPDATABLE:
        if k in (fields or {}):
            cap = MAX_TITLE_CHARS if k == "title" else MAX_TEXT_CHARS
            val = _clean_text(fields[k], cap)
            if k == "title" and not val:
                raise PlanError("title must be non-empty")
            task[k] = val
    if "labels" in (fields or {}):
        labels_raw = fields["labels"]
        if not isinstance(labels_raw, list):
            raise PlanError("labels must be a list of strings")
        labels: List[str] = []
        for l in labels_raw[:MAX_LABELS]:
            ls = _clean_text(l, 64)
            if ls and ls not in labels:
                labels.append(ls)
        task["labels"] = labels
    task["updated"] = _utcnow()
    return task


def set_status(store: Dict[str, Any], task_id: str, status: str, *,
               note: Optional[str] = None) -> Dict[str, Any]:
    """Set a task's status with parent->subtask cascade on done.

    ``note`` (optional) appends "[iso] note" to details — the agent tool uses
    this to leave a completion trail. Cascade rule: flipping a parent to done
    flips its non-terminal subtasks to done too (the parent being finished
    means its decomposition is moot); cancelled parents do NOT cascade."""
    if status not in TASK_STATUSES:
        raise PlanError(f"invalid status {status!r} (choose from "
                        f"{TASK_STATUSES})")
    task = get_task(store, task_id)
    if task is None:
        raise PlanError(f"no such task {task_id!r}")
    task["status"] = status
    if note:
        stamp = _clean_text(note, MAX_TEXT_CHARS)
        task["details"] = (f"{task['details']}\n[{_utcnow()}] {stamp}"
                           .strip())[:MAX_TEXT_CHARS]
    if status == "done":
        # Cascade recursively: a finished parent makes its whole decomposition
        # moot (subtasks AND their subtasks). Terminal children are skipped,
        # and an already-done ancestor stops re-walking that subtree.
        stack = [task_id]
        seen = {task_id}
        while stack:
            cur = stack.pop()
            for t in store.get("tasks") or []:
                if t.get("parent") == cur and t["id"] not in seen:
                    seen.add(t["id"])
                    if t["status"] not in ("done", "cancelled"):
                        t["status"] = "done"
                        t["updated"] = _utcnow()
                    stack.append(t["id"])
    task["updated"] = _utcnow()
    return task


def record_run(store: Dict[str, Any], task_id: str, *,
               trace_id: Optional[str] = None,
               thread: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Append a run linkage entry (trace/thread) to a task's runs list."""
    task = get_task(store, task_id)
    if task is None:
        return None
    task["runs"].append({"trace_id": trace_id, "thread": thread,
                         "at": _utcnow()})
    task["runs"] = task["runs"][-MAX_RUNS_PER_TASK:]
    task["updated"] = _utcnow()
    return task


# ------------------------------------------------------------------- DAG

def _dep_graph(store: Dict[str, Any]) -> Tuple[Dict[str, List[str]], set]:
    ids = {t["id"] for t in store.get("tasks") or []}
    graph: Dict[str, List[str]] = {}
    for t in store.get("tasks") or []:
        graph[t["id"]] = [d for d in t["deps"] if d in ids and d != t["id"]]
    return graph, ids


def find_cycles(store: Dict[str, Any]) -> List[List[str]]:
    """All dependency cycles (each reported once, smallest rotation)."""
    graph, _ids = _dep_graph(store)
    cycles: List[List[str]] = []
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in graph}
    stack: List[str] = []

    def visit(node: str) -> None:
        color[node] = GRAY
        stack.append(node)
        for dep in graph.get(node, []):
            if color.get(dep, WHITE) == GRAY:
                i = stack.index(dep)
                cyc = stack[i:]
                k = cyc.index(min(cyc))
                cyc = cyc[k:] + cyc[:k]
                if cyc not in cycles:
                    cycles.append(cyc)
            elif color.get(dep, WHITE) == WHITE:
                visit(dep)
        stack.pop()
        color[node] = BLACK

    for n in sorted(graph):
        if color[n] == WHITE:
            visit(n)
    return cycles


def validate(store: Dict[str, Any]) -> Dict[str, Any]:
    """Machine-readable report: cycles + dangling dep/parent references.

    Shape: {"ok": bool, "cycles": [[ids]], "dangling_deps":
    [{"task": id, "dep": id}], "duplicate_ids": [..]}. Used by the CLI
    (`toaster_loop plan validate`), the dashboard GET, and MCP plans_list.

    ``duplicate_ids`` is defensive only: ``validate_store`` already drops a
    second task with an existing id on every load/save, so well-formed stores
    can never report one. Mutations that need "did THIS change break things"
    should diff :func:`issues` before/after instead of trusting global ok."""
    tasks = store.get("tasks") or []
    ids = {t["id"] for t in tasks}
    dupes: List[str] = []
    seen: set = set()
    for t in tasks:
        if t["id"] in seen and t["id"] not in dupes:
            dupes.append(t["id"])
        seen.add(t["id"])
    dangling = [{"task": t["id"], "dep": d} for t in tasks
                for d in t["deps"] if d not in ids]
    cycles = find_cycles(store)
    return {"ok": not (cycles or dangling), "cycles": cycles,
            "dangling_deps": dangling, "duplicate_ids": dupes}


def issues(store: Dict[str, Any]) -> set:
    """Normalized snapshot of every graph problem (cycles + dangling deps).

    Used to scope validation on mutations: a mutation is acceptable when it
    introduces NO NEW problems. Validating global ``ok`` instead would let one
    pre-existing dangling dep — reachable via manual file edits or the
    deliberate keep-on-load policy — brick every future add forever. Cycles
    are canonicalized (smallest rotation) so the same cycle never reads as
    "new" after an unrelated edit shifts traversal order."""
    rep = validate(store)
    out = set()
    for cyc in rep["cycles"]:
        k = cyc.index(min(cyc))
        out.add(("cycle", tuple(cyc[k:] + cyc[:k])))
    for d in rep["dangling_deps"]:
        out.add(("dangling", d["task"], d["dep"]))
    return out


def add_dependency(store: Dict[str, Any], task_id: str,
                   dep_id: str) -> Dict[str, Any]:
    """Add task depends-on dep. Refuses self-dep and any cycle up front."""
    if task_id == dep_id:
        raise PlanError("a task cannot depend on itself")
    task = get_task(store, task_id)
    dep = get_task(store, dep_id)
    if task is None:
        raise PlanError(f"no such task {task_id!r}")
    if dep is None:
        raise PlanError(f"no such dependency task {dep_id!r}")
    if dep_id in task["deps"]:
        return task
    task["deps"].append(dep_id)
    cycles = find_cycles(store)
    if cycles:
        task["deps"].remove(dep_id)
        joined = " -> ".join(cycles[0] + [cycles[0][0]])
        raise PlanError(f"refused: that dependency creates a cycle ({joined})")
    task["updated"] = _utcnow()
    return task


def remove_dependency(store: Dict[str, Any], task_id: str,
                      dep_id: str) -> Dict[str, Any]:
    task = get_task(store, task_id)
    if task is None:
        raise PlanError(f"no such task {task_id!r}")
    if dep_id not in task["deps"]:
        raise PlanError(f"task {task_id} does not depend on {dep_id}")
    task["deps"].remove(dep_id)
    task["updated"] = _utcnow()
    return task


def remove_task(store: Dict[str, Any], task_id: str) -> Tuple[bool, List[str]]:
    """Remove a task; rewrites deps of other tasks that referenced it.

    Returns (removed, warned_about). Subtasks keep existing but their parent
    link is cleared (they were already promoted-orphaned conceptually)."""
    tasks = store.get("tasks") or []
    task = get_task(store, task_id)
    if task is None:
        return False, []
    store["tasks"] = [t for t in tasks if t["id"] != task_id]
    warned: List[str] = []
    for t in store["tasks"]:
        if task_id in t["deps"]:
            t["deps"].remove(task_id)
            t["updated"] = _utcnow()
            warned.append(t["id"])
        if t.get("parent") == task_id:
            t["parent"] = None
    return True, warned


def blockers(store: Dict[str, Any], task_id: str) -> List[Dict[str, Any]]:
    """Unsatisfied deps of a task (deps not done|cancelled)."""
    by_id = {t["id"]: t for t in store.get("tasks") or []}
    task = by_id.get(task_id)
    if task is None:
        return []
    return [by_id[d] for d in task["deps"]
            if d in by_id and by_id[d]["status"] not in DEPS_SATISFYING]


def ready(store: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Tasks that could run right now: pending or in_progress, no unsatisfied
    deps. Ordered like next_task (priority > fewer blockers > oldest id)."""
    report = validate(store)
    broken = {e["task"] for e in report["dangling_deps"]}
    cycled: set = set()
    for cyc in report["cycles"]:
        cycled.update(cyc)
    out = []
    for t in store.get("tasks") or []:
        if t["status"] not in ("pending", "in_progress"):
            continue
        if t["id"] in broken or t["id"] in cycled:
            continue  # a dangling/cyclic dep is not merely unsatisfied
        if blockers(store, t["id"]):
            continue
        out.append(t)
    return sorted(out, key=lambda t: (_PRIORITY_RANK.get(t["priority"], 2),
                                      len(blockers(store, t["id"])), t["id"]))


def next_task(store: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The single best task to work on next (None when nothing is unblocked)."""
    r = ready(store)
    return r[0] if r else None


# -------------------------------------------------------------- rendering

def counts(store: Dict[str, Any]) -> Dict[str, int]:
    tasks = store.get("tasks") or []
    c = {"total": len(tasks), "done": 0, "in_progress": 0, "blocked": 0,
         "pending": 0, "cancelled": 0}
    for t in tasks:
        c[t["status"]] = c.get(t["status"], 0) + 1
    return c


def summary(store: Dict[str, Any], max_chars: int = 800) -> str:
    """Bounded human/SSE render: done/total + next ready ids+titles + the
    in-progress line. Hard-capped at ``max_chars`` (never sliced mid-id: if a
    line does not fit whole, it is dropped, then the block gets an ellipsis)."""
    c = counts(store)
    lines = ["## Project plan",
             f"{c['done']}/{c['total']} tasks done"]
    nxt = ready(store)[:5]
    if nxt:
        lines.append("Ready next:")
        for t in nxt:
            lines.append(f"- {t['id']}: {t['title']}")
    prog = [t for t in store.get("tasks") or []
            if t["status"] == "in_progress"]
    if prog:
        lines.append("In progress: " + ", ".join(
            f"{t['id']} ({t['title'][:40]})" for t in prog[:5]))
    out = "\n".join(lines)
    if len(out) <= max_chars:
        return out
    # Drop trailing lines until the block fits whole.
    keep: List[str] = []
    used = 0
    for line in lines:
        add = len(line) + (1 if keep else 0)
        if used + add + 1 > max_chars - 1:  # room for "\u2026"
            break
        keep.append(line)
        used += add
    return ("\n".join(keep) + "\n\u2026")[:max_chars]


#: Cap on the "Completed dependencies:" block inside a task prompt.
DEP_CONTEXT_CAP = 1500


def build_task_prompt(store: Dict[str, Any], task_id: str) -> str:
    """Full working prompt for one task: title + description + details +
    test_strategy + a bounded "Completed dependencies:" context block."""
    task = get_task(store, task_id)
    if task is None:
        raise PlanError(f"no such task {task_id!r}")
    parts = [f"# Task {task['id']}: {task['title']}"]
    if task["description"]:
        parts.append(task["description"])
    if task["details"]:
        parts.append(f"## Details\n{task['details']}")
    if task["test_strategy"]:
        parts.append(f"## How to verify\n{task['test_strategy']}")
    done_deps = []
    by_id = {t["id"]: t for t in store.get("tasks") or []}
    for d in task["deps"]:
        dt = by_id.get(d)
        if dt is not None and dt["status"] in DEPS_SATISFYING:
            done_deps.append(dt)
    if done_deps:
        block = ["## Completed dependencies:"]
        used = len(block[0])
        for dt in done_deps:
            first = (dt["details"].splitlines() or [""])[0] if dt["details"] \
                else ""
            line = f"- {dt['id']} ({dt['status']}): {dt['title']}"
            if first:
                line += f" — {first[:120]}"
            if used + len(line) + 1 > DEP_CONTEXT_CAP:
                block.append("- …(dependency context truncated)")
                break
            block.append(line)
            used += len(line) + 1
        parts.append("\n".join(block))
    parts.append(
        "\nWhen this task is complete, mark it done with the plan_progress "
        f"tool (task_id={task['id']}, status=done).")
    return "\n\n".join(parts)

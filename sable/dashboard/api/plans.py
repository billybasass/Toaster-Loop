"""WP12 plan endpoints: GET/POST /api/plans/<project_id>.

One store, three surfaces (CLI, dashboard, MCP): everything here goes
through core/plans.py — the same functions `sable plan ...` uses — so the
dashboard and CLI can never disagree. Mutations publish SSE ``plan.updated``
{project, task_ids}; the autonomous loop publishes ``plan.loop`` per
iteration (injected bus publish; see core/plan_loop.py).

With config plans.enabled off (the default) every endpoint answers 404
code="plans_disabled" — no store access, mirroring "no plan tools anywhere"
at the agent/MCP layer. Unknown project ids are 404 unknown_project (same
pattern as the projects API); the literal id "default" addresses the
no-project plan and needs no project record.

POST body: {"op": ...}. Read-only op is "validate"; everything else mutates
(or starts a run/loop) and therefore publishes:

  add | update | set-status | depends   -> store edits (PlanError => 400)
  import | expand                       -> core/plan_llm via the judge seam
                                           (one bounded model call; use_ai flag)
  start                                 -> "Start task": new thread seeded with
                                           build_task_prompt + the existing chat
                                           run path + status=in_progress
  loop-start | loop-cancel              -> guarded autonomous loop (409 when a
                                           loop is already running: single-flight)
"""

from __future__ import annotations

import re
import uuid
from typing import Any, Dict, Optional

from .base import ApiContext, HttpError


class PlansRoutes(ApiContext):
    """Project-scoped task-graph endpoints (WP12)."""

    _PLAN_OPS = ("add", "update", "set-status", "depends", "validate",
                 "import", "expand", "start", "loop-start", "loop-cancel")

    # ------------------------------------------------------------------ gate

    def _plans_enabled(self) -> bool:
        from ...core.plans import plans_config

        return bool(plans_config(self.cfg).get("enabled", False))

    def _require_plans(self) -> None:
        if not self._plans_enabled():
            raise HttpError(404, "plans are disabled (config plans.enabled)",
                            code="plans_disabled")

    def _require_project(self, project_id: str) -> Optional[str]:
        """Validate the path id; returns the store key (None = default plan).

        'default' is the no-project plan and always allowed; any other id
        must be an existing project record (404 unknown_project otherwise —
        same contract as chat threads pointing at deleted projects)."""
        pid = str(project_id or "").strip()
        if not re.match(r"^[A-Za-z0-9._-]{1,64}$", pid):
            raise HttpError(400, "invalid project id")
        if pid == "default":
            return None
        if self._load_project(pid) is None:
            raise HttpError(404, f"no project {pid}", code="unknown_project")
        return pid

    # ------------------------------------------------------------------- GET

    def h_plan_get(self, q: Dict[str, str], b: Dict[str, Any],
                   project_id: str) -> Dict[str, Any]:
        self._require_plans()
        key = self._require_project(project_id)
        from ...core import plans as P

        store = P.load_plan(key)
        return {
            "project": key or "default",
            "updated": store.get("updated"),
            "counts": P.counts(store),
            "summary": P.summary(store, max_chars=2000),
            "tasks": store["tasks"],
            "validate": P.validate(store),
            "loop": self._loop_progress(key),
        }

    def _loop_progress(self, key: Optional[str]) -> Dict[str, Any]:
        from ...core.plan_loop import current_loop, read_loop_progress
        from ...store.paths import resolve_sable_home

        prog = read_loop_progress(key, resolve_sable_home(self.cfg)) or {}
        live = current_loop()
        running = bool(live is not None
                       and str(live.project_id or "") == str(key or ""))
        return {"running": running, "progress": prog}

    # ------------------------------------------------------------------ POST

    def h_plan_post(self, q: Dict[str, str], b: Dict[str, Any],
                    project_id: str) -> Dict[str, Any]:
        self._require_plans()
        key = self._require_project(project_id)
        op = str(b.get("op") or "").strip()
        if op not in self._PLAN_OPS:
            raise HttpError(400, f"unknown op {op!r} (want one of "
                                 f"{', '.join(self._PLAN_OPS)})")
        handler = getattr(self, f"_op_{op.replace('-', '_')}")
        return handler(key, b)

    # -- store edits ---------------------------------------------------------

    def _publish_updated(self, key: Optional[str], task_ids: list) -> None:
        try:
            self.bus.publish("plan.updated",
                             {"project": key or "default",
                              "task_ids": list(task_ids)})
        except Exception:  # noqa: BLE001 - SSE is best-effort
            pass

    def _op_add(self, key: Optional[str], b: Dict[str, Any]) -> Dict[str, Any]:
        from ...core import plans as P
        from ...core.plans import PlanError

        fields = {k: b.get(k) for k in
                  ("title", "description", "details", "test_strategy",
                   "priority", "deps", "parent", "labels")
                  if b.get(k) is not None}
        store = P.load_plan(key)
        try:
            t = P.add_task(store, fields)
            report = P.validate(store)
            if not report["ok"]:
                raise PlanError(
                    f"plan invalid after add ({report['cycles']} "
                    f"{report['dangling_deps']}); task NOT saved")
            P.save_plan(store)
        except PlanError as e:
            raise HttpError(400, str(e)) from e
        self._publish_updated(key, [t["id"]])
        return {"ok": True, "task": t}

    def _op_update(self, key: Optional[str], b: Dict[str, Any]) -> Dict[str, Any]:
        from ...core import plans as P
        from ...core.plans import PlanError

        store = P.load_plan(key)
        try:
            t = P.update_task(store, str(b.get("task_id") or ""),
                              {k: b[k] for k in
                               ("title", "description", "details",
                                "test_strategy", "priority", "labels")
                               if k in b})
            P.save_plan(store)
        except PlanError as e:
            status = 404 if "no such task" in str(e) else 400
            raise HttpError(status, str(e)) from e
        self._publish_updated(key, [t["id"]])
        return {"ok": True, "task": t}

    def _op_set_status(self, key: Optional[str],
                       b: Dict[str, Any]) -> Dict[str, Any]:
        from ...core import plans as P
        from ...core.plans import PlanError

        store = P.load_plan(key)
        try:
            t = P.set_status(store, str(b.get("task_id") or ""),
                             str(b.get("status") or ""),
                             note=str(b.get("note") or "") or None)
            P.save_plan(store)
        except PlanError as e:
            status = 404 if "no such task" in str(e) else 400
            raise HttpError(status, str(e)) from e
        self._publish_updated(key, [t["id"]])
        return {"ok": True, "task": t}

    def _op_depends(self, key: Optional[str], b: Dict[str, Any]) -> Dict[str, Any]:
        from ...core import plans as P
        from ...core.plans import PlanError

        store = P.load_plan(key)
        task_id = str(b.get("task_id") or "")
        dep_id = str(b.get("dep_id") or "")
        try:
            if b.get("remove"):
                t = P.remove_dependency(store, task_id, dep_id)
            else:
                t = P.add_dependency(store, task_id, dep_id)
            P.save_plan(store)
        except PlanError as e:
            status = 404 if "no such task" in str(e) else 400
            raise HttpError(status, str(e)) from e
        self._publish_updated(key, [task_id])
        return {"ok": True, "task": t}

    def _op_validate(self, key: Optional[str],
                     b: Dict[str, Any]) -> Dict[str, Any]:
        from ...core import plans as P

        return {"ok": True, "validate": P.validate(P.load_plan(key))}

    # -- LLM ops (judge seam; one bounded call each) --------------------------

    def _ai_client(self):
        """(client_fn, model) for the served endpoint — same resolution the
        dashboard's other AI touchpoints use (cli._resolve_endpoint attaches
        to a running server before considering a spawn)."""
        from ...curate.judge import make_chat_client

        try:
            from ...cli import _resolve_endpoint

            base_url, model, sup = _resolve_endpoint(self.cfg,
                                                     spawn_if_needed=True)
        except Exception as e:  # noqa: BLE001 - surface as clean 503
            raise HttpError(503, f"no serving endpoint for AI op ({e})",
                            code="no_endpoint") from e
        if sup is not None:
            try:
                sup.stop()  # attach-only here; the chat path owns servers
            except Exception:  # noqa: BLE001
                pass
        client = make_chat_client(base_url,
                                  api_key=getattr(self.cfg, "attach_api_key",
                                                  None) or None,
                                  max_tokens=1200)
        return client, model

    def _op_import(self, key: Optional[str], b: Dict[str, Any]) -> Dict[str, Any]:
        from ...core.plan_llm import PlanImportError, plan_import
        from ...core.plans import PlanError

        text = b.get("text")
        if not isinstance(text, str) or not text.strip():
            raise HttpError(400, "text required (paste the brief)")
        use_ai = bool(b.get("use_ai"))
        client = model = None
        if use_ai:
            client, model = self._ai_client()
        try:
            rep = plan_import(key, text, use_ai=use_ai,
                              num_tasks=int(b.get("num_tasks") or 8),
                              client_fn=client, model=model)
        except PlanImportError as e:
            raise HttpError(400, str(e)) from e
        except PlanError as e:
            raise HttpError(400, str(e)) from e
        if rep["ok"]:
            self._publish_updated(
                key, [t["id"] for t in rep["added"]])
        return {"ok": bool(rep["ok"]), "source": rep["source"],
                "added": rep["added"], "warnings": rep["warnings"]}

    def _op_expand(self, key: Optional[str], b: Dict[str, Any]) -> Dict[str, Any]:
        from ...core.plan_llm import PlanImportError, plan_expand
        from ...core.plans import PlanError

        task_id = str(b.get("task_id") or "")
        if not task_id:
            raise HttpError(400, "task_id required")
        client, model = self._ai_client()
        try:
            rep = plan_expand(key, task_id, num=int(b.get("num") or 3),
                              prompt=str(b.get("prompt") or "") or None,
                              client_fn=client, model=model)
        except PlanImportError as e:
            status = 404 if "no such task" in str(e) else 400
            raise HttpError(status, str(e)) from e
        except PlanError as e:
            raise HttpError(400, str(e)) from e
        if rep["ok"]:
            self._publish_updated(
                key, [t["id"] for t in rep["added"]])
        return {"ok": bool(rep["ok"]), "source": rep["source"],
                "added": rep["added"], "warnings": rep["warnings"]}

    # -- start task (thread + run + in_progress) ------------------------------

    def _op_start(self, key: Optional[str], b: Dict[str, Any]) -> Dict[str, Any]:
        """Chat-page "Start task": create a thread seeded with the task's
        full prompt, launch it through the existing chat run path (profile
        grants apply unchanged), stamp plan_task on the run + thread linkage,
        and flip the task to in_progress."""
        from ...core import plans as P
        from ...core.plans import PlanError

        task_id = str(b.get("task_id") or "")
        store = P.load_plan(key)
        task = P.get_task(store, task_id)
        if task is None:
            raise HttpError(404, f"no such task {task_id!r}",
                            code="unknown_task")
        prompt = P.build_task_prompt(store, task_id)
        extra = str(b.get("instruction") or "").strip()
        if extra:
            prompt = f"{prompt}\n\nExtra instruction:\n{extra}"

        thread_id = uuid.uuid4().hex[:12]
        prof = str(task.get("profile") or b.get("profile") or "").strip() \
            or None
        # Thread record carries the plan task ("task" key, additive like
        # "project": readers .get(), legacy threads simply lack it).
        self.h_thread_put({}, {
            "title": task["title"],
            "messages": [{"role": "user", "content": prompt}],
            "project": key,
            "profile": prof,
            "task": task_id,
        }, thread_id)

        run_body: Dict[str, Any] = {
            "prompt": prompt,
            "thread": thread_id,
            "plan_task": task_id,
        }
        if key:
            run_body["project"] = key
        if prof:
            run_body["profile"] = prof
        # Kick the existing chat run path. Its worker thread owns the run;
        # this handler is only a reader (dropping the stream never affects
        # the run — same contract as a navigating-away browser).
        resp = self.h_chat_stream({}, run_body)
        if getattr(resp, "stream", None) is not None:
            try:
                next(iter(resp.stream))  # starts the worker; frames unread
            except StopIteration:
                pass

        try:
            store = P.load_plan(key)
            t = P.set_status(store, task_id, "in_progress",
                             note=f"started from plan (thread {thread_id})")
            P.record_run(store, task_id, thread=thread_id)
            P.save_plan(store)
        except PlanError as e:  # run may have raced ahead; report the task state
            raise HttpError(409, f"task not startable: {e}") from e
        self._publish_updated(key, [task_id])
        return {"ok": True, "thread": thread_id, "task": t}

    # -- autonomous loop -------------------------------------------------------

    def _op_loop_start(self, key: Optional[str],
                       b: Dict[str, Any]) -> Dict[str, Any]:
        from ...core.plan_loop import LoopConflictError, start_plan_loop

        try:
            loop = start_plan_loop(
                self.cfg, key,
                max_tasks=(int(b["max_tasks"]) if b.get("max_tasks")
                           else None),
                publish=lambda kind, payload: self.bus.publish(kind, payload),
                resume=bool(b.get("resume")))
        except LoopConflictError as e:
            # Single-flight: a second start is a conflict, never queued.
            raise HttpError(409, str(e), code="loop_running") from e
        return {"ok": True, "project": key or "default",
                "iteration": loop.progress.get("iteration")}

    def _op_loop_cancel(self, key: Optional[str],
                        b: Dict[str, Any]) -> Dict[str, Any]:
        from ...core.plan_loop import cancel_plan_loop

        cancelled = cancel_plan_loop(key)
        return {"ok": True, "cancelled": cancelled}

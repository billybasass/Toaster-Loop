"""Project records (<home>/projects/*.json) CRUD.

Record fields: id, name, workspace_dir, agents (profile names),
description (free text shown to every agent working on the project),
agent_roles ({profile name: that agent's job on this project}),
created, updated. description/agent_roles reach the model through
``agent_runner.create_runner(project_context=...)``; the delete endpoint
removes the project + its per-project index DB and NEVER touches the
workspace directory.
"""

# Domain slice of the former monolithic dashboard/app.py: a pure move,
# no behavior change. Handlers resolve through DashboardApp attribute
# lookup (see ROUTES in app.py).

from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, Optional


from .base import (ApiContext, HttpError, PROJECTS_DIRNAME, ApiResponse,
                   _utcnow)


class ProjectsRoutes(ApiContext):
    """Project records (<home>/projects/*.json) CRUD."""

    _PROJECT_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

    def _project_path(self, project_id: str) -> Path:
        if not self._PROJECT_ID_RE.match(project_id):
            raise HttpError(400, "invalid project id")
        return self.home / PROJECTS_DIRNAME / f"{project_id}.json"

    def _load_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        try:
            p = self._project_path(project_id)
        except HttpError:
            return None
        if not p.is_file():
            return None
        try:
            data = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    def _save_project(self, project: Dict[str, Any]) -> None:
        p = self._project_path(str(project["id"]))
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(project, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        os.replace(tmp, p)

    def _validate_project_body(self, b: Dict[str, Any],
                               existing: Optional[Dict[str, Any]] = None,
                               ) -> Dict[str, Any]:
        """Validate a create/update body; returns the mutable field set."""
        name = b.get("name", (existing or {}).get("name"))
        if not isinstance(name, str) or not name.strip():
            raise HttpError(400, "name required (non-empty string)")
        # Duplicate names make projects indistinguishable in the thread tree,
        # the trace filter and the explorer title — reject them (case-
        # insensitive; another project keeping the same name is fine).
        want = name.strip().casefold()
        d = self.home / PROJECTS_DIRNAME
        if d.is_dir():
            for f in d.glob("*.json"):
                other_id = f.stem
                if existing is not None and other_id == existing.get("id"):
                    continue
                try:
                    other = json.loads(f.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                if (isinstance(other, dict)
                        and str(other.get("name") or "").strip().casefold()
                        == want):
                    raise HttpError(
                        409, f"a project named {name.strip()!r} already exists "
                        f"(id {other_id})", code="duplicate_project_name")
        ws_raw = b.get("workspace_dir",
                       (existing or {}).get("workspace_dir"))
        if not isinstance(ws_raw, str) or not ws_raw.strip():
            raise HttpError(400, "workspace_dir required (absolute path)")
        ws = Path(ws_raw).expanduser()
        if not ws.is_absolute():
            raise HttpError(400, "workspace_dir must be an absolute path")
        if not ws.is_dir():
            raise HttpError(400, f"workspace_dir does not exist: {ws}")
        agents = b.get("agents", (existing or {}).get("agents") or [])
        if not isinstance(agents, list) or any(
                not isinstance(a, str) for a in agents):
            raise HttpError(400, "agents must be a list of profile names")
        # Project description: free text injected into every agent run on
        # this project (see agent_runner._with_project_context).
        description = b.get("description",
                            (existing or {}).get("description"))
        if description is None:
            description = ""
        if not isinstance(description, str):
            raise HttpError(400, "description must be a string")
        # agent_roles: {profile name: that agent's job on this project}.
        # Keys outside the final agent list are dropped (roles belong to
        # attached agents; removing an agent prunes its role), as are
        # blank roles. An omitted key keeps the existing map (PUT fallback).
        roles_raw = b.get("agent_roles",
                          (existing or {}).get("agent_roles") or {})
        if not isinstance(roles_raw, dict) or any(
                not isinstance(k, str) or not isinstance(v, str)
                for k, v in roles_raw.items()):
            raise HttpError(400, "agent_roles must be an object mapping "
                                 "profile names to role descriptions")
        agents = sorted(set(agents))
        agent_roles = {}
        for k, v in roles_raw.items():
            k, v = k.strip(), v.strip()
            if k and v and k in agents:
                agent_roles[k] = v
        return {"name": name.strip(), "workspace_dir": str(ws),
                "agents": agents, "description": description.strip(),
                "agent_roles": agent_roles}

    def h_projects_list(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        d = self.home / PROJECTS_DIRNAME
        projects = []
        if d.is_dir():
            for f in sorted(d.glob("*.json")):
                try:
                    p = json.loads(f.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                if isinstance(p, dict):
                    projects.append(p)
        return {"projects": projects}

    def h_projects_create(self, q: Dict[str, str], b: Dict[str, Any]) -> ApiResponse:
        fields = self._validate_project_body(b)
        project_id = uuid.uuid4().hex[:12]
        now = _utcnow()
        project = {"id": project_id, "created": now, "updated": now, **fields}
        self._save_project(project)
        # A new project enrolls in codebase indexing immediately: with
        # indexing.enabled, its workspace_dir becomes the root of its OWN
        # sqlite DB (<home>/index/<project_id>.db) and a background build
        # starts — other projects' indexes are untouched.
        self._kick_project_index(project_id)
        return ApiResponse(201, {"ok": True, "project": project})

    def h_project_put(self, q: Dict[str, str], b: Dict[str, Any],
                      project_id: str) -> Dict[str, Any]:
        existing = self._load_project(project_id)
        if existing is None:
            raise HttpError(404, f"no project {project_id}")
        fields = self._validate_project_body(b, existing=existing)
        project = {**existing, **fields, "id": project_id,
                   "updated": _utcnow()}
        self._save_project(project)
        # Re-enroll: builds are incremental, and get_index() rebuilds the
        # cached instance in place when workspace_dir changed, so an edit
        # keeps the project's own DB pointed at the right directory.
        self._kick_project_index(project_id)
        return {"ok": True, "project": project}

    def _kick_project_index(self, project_id: str) -> None:
        """Start this project's index build when indexing is enabled.

        Best-effort by design: the project record is already saved, and an
        index hiccup (no embedding model, bus failure, ...) must never fail
        or roll back a create/update. The build runs in the same background
        thread + index.build SSE stream the manual "build index now"
        button uses, scoped to this project's DB only."""
        ix = getattr(self.cfg, "indexing", None) or {}
        if not ix.get("enabled"):
            return
        try:
            self.h_index_build({}, {"project": project_id})
        except Exception:
            pass  # indexing is best-effort; the project exists regardless

    def h_project_delete(self, q: Dict[str, str], b: Dict[str, Any],
                         project_id: str) -> Dict[str, Any]:
        p = self._project_path(project_id)
        if not p.is_file():
            raise HttpError(404, f"no project {project_id}")
        # Cancel any in-flight build FIRST and drop the cached instance — an
        # uncancelled build would recreate <home>/index/<id>.db through
        # _db()'s mkdir/connect after delete_project_index removed it.
        try:
            from ...core.indexer import drop_index_instance, get_index

            try:
                get_index(self.home, cfg=self.cfg,
                          project_id=project_id).cancel_build()
            except Exception:
                pass  # unknown/invalid id: nothing to cancel
            drop_index_instance(self.home, project_id)
        except Exception:
            pass
        p.unlink()
        # Phase 5: delete the project's index DB (and only that file — the
        # default DB and other projects' indexes are never touched).
        index_removed = False
        try:
            from ...core.indexer import delete_project_index

            index_removed = delete_project_index(self.home, project_id)
        except Exception:
            pass  # index cleanup is best-effort; the project file is gone
        # Threads intentionally untouched: their stale project ids read as
        # Unfiled (the tree groups unknown/absent ids under Unfiled).
        return {"ok": True, "index_removed": index_removed}

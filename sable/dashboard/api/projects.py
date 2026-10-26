"""Project records (<home>/projects/*.json) CRUD."""

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
        return {"name": name.strip(), "workspace_dir": str(ws),
                "agents": sorted(set(agents))}

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
        return {"ok": True, "project": project}

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

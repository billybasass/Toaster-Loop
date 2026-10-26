"""Program control (shutdown/logs) + codebase index endpoints
(semantic search).
"""

# Domain slice of the former monolithic dashboard/app.py: a pure move,
# no behavior change. Handlers resolve through DashboardApp attribute
# lookup (see ROUTES in app.py).

from __future__ import annotations

import os
import re
import signal
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


from ...serving.deploy import (
    stop_owned_server,
)
from .base import (ApiContext, HttpError, PROJECTS_DIRNAME, _utcnow)


class ControlRoutes(ApiContext):
    """Program control (shutdown/logs) + codebase index endpoints
(semantic search).
    """

    def h_shutdown(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        """Close the program: stop the owned llama-server, then exit ourselves.

        Only meaningful for `sable dashboard`/`launch` processes; a graceful
        self-terminate via SIGTERM so the serve_dashboard signal handler runs
        (server socket closed, 'dashboard stopped.' printed). The owned
        llama-server child is stopped first so nothing is orphaned.
        """

        # Run lifecycle step 3: let in-flight chat runs cancel cleanly BEFORE
        # the serving backends go away, so each runner records its own
        # score-neutralized cancelled trace instead of dying with the process.
        try:
            self.shutdown_chat_runs()
        except Exception:
            pass

        # Phase 6: stop every serving process this process owns — all
        # multi-instance children AND the Phase 5 embed servers. Foreign /
        # attach processes are never touched (registry holds only ours).
        try:
            from ...serving.supervisor import stop_all_servers
            stop_all_servers()
        except Exception:
            pass
        def _die() -> None:
            try:
                stop_owned_server(self.home)
            except Exception:
                pass
            try:
                # Phase 5: stop embedding servers WE spawned (never foreign).
                from ...core.indexer import stop_all_embed_servers
                stop_all_embed_servers()
            except Exception:
                pass
            self.bus.publish("app.state", {"event": "shutdown"})
            time.sleep(0.4)  # let the SSE frame flush + response return
            os.kill(os.getpid(), signal.SIGTERM)

        threading.Thread(target=_die, daemon=True, name="shutdown").start()
        return {"ok": True, "note": "shutting down — stop_owned_server ran; "
                                    "dashboard process exiting"}

    def h_logs(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        """Console log tail: dashboard.log + llama-server.log (program console).

        Every llama-server sable spawns writes its own file: the legacy
        default server to ``llama-server.log``, named instances to
        ``server-<id>.log`` (see h_server_start) and the indexer's embedding
        server to ``embed-server.log``. We enumerate the logs dir so new
        instances show up in the console without any registration step."""
        tail = max(10, min(int(q.get("tail", "400") or 400), 5000))
        logs_dir = self.home / "logs"
        files: Dict[str, Path] = {
            "dashboard": logs_dir / "dashboard.log",
            "llama-server": logs_dir / "llama-server.log",
            "embed-server": logs_dir / "embed-server.log",
        }
        try:
            for p in sorted(logs_dir.glob("server-*.log")):
                name = p.stem  # "server-<instance-id>"
                if name not in files:
                    files[name] = p
        except OSError:
            pass
        out: Dict[str, Any] = {}
        for name, p in files.items():
            lines: list[str] = []
            exists = p.is_file()
            if exists:
                try:
                    with p.open("r", errors="replace") as fh:
                        # cheap tail: read the last ~256 KB
                        size = p.stat().st_size
                        if size > 262_144:
                            fh.seek(size - 262_144)
                            fh.readline()
                        lines = fh.read().splitlines()[-tail:]
                except OSError:
                    exists = False
            out[name] = {"exists": exists, "lines": lines}
        return out

    def _default_root_is_home(self, idx) -> bool:
        """True when the DEFAULT (no-project) index would index SABLE_HOME.

        That fallback root is the process cwd, and the dashboard is normally
        launched from ~/.sable — so the default index silently swallowed
        llama.cpp build trees, vendored engines and sable.yaml itself (9k
        files / 185 MB observed) instead of any code. An explicit
        paths_override is honoured; the implicit home-rooted default is not."""
        try:
            if getattr(idx, "project_id", None) is not None:
                return False
            if idx.index_cfg.get("paths_override") or idx.index_cfg.get("paths"):
                return False
            return bool(idx.roots) and \
                idx.roots[0].resolve() == Path(self.home).resolve()
        except OSError:
            return False

    def _index(self, project_id: Optional[str] = None):
        from ...core.indexer import get_index

        proj_ws = None
        if project_id:
            proj = self._load_project(project_id)
            if proj is None:
                raise HttpError(404, f"no project {project_id}",
                                code="unknown_project")
            proj_ws = str(proj.get("workspace_dir") or "") or None
        return get_index(self.home, cfg=self.cfg, project_id=project_id,
                         workspace_dir=proj_ws)

    def h_index_status(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        pid = str(q.get("project") or "").strip() or None
        return self._index(pid).status()

    def h_index_build(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        """Build/refresh the codebase index in a background thread.

        Phase 5: `project` (query or body) selects the project DB; omitted
        => default DB. Uses the spawn-managed embedding server — when the
        configured model can't serve, the build fails loudly (index.build
        error event) instead of silently falling back to lexical.

        Progress streams as `index.build` SSE events; result also pollable via
        GET /api/index (building flag).
        """
        pid = (str(b.get("project") or "").strip()
               or str(q.get("project") or "").strip() or None)
        idx = self._index(pid)
        if self._default_root_is_home(idx):
            # Refuse loudly instead of silently building a 185 MB index of
            # SABLE_HOME (build trees + vendored engines + sable.yaml).
            raise HttpError(
                409,
                "the default index would index SABLE_HOME itself "
                f"({self.home}); open a project (projects index their "
                "workspace_dir) or set indexing.paths_override to pick roots",
                code="index_root_is_sable_home") from None

        record = {"state": "running", "started": _utcnow()}

        def _log(line: str) -> None:
            self.bus.publish("index.build", {"state": "running", "line": str(line)})

        def _run() -> None:
            try:
                res = idx.build(progress=_log)
                record["state"] = "done"
                record.update(res)
                self.bus.publish("index.build", {"state": "done", **res})
            except Exception as e:
                record["state"] = "error"
                record["error"] = f"{type(e).__name__}: {e}"
                self.bus.publish("index.build", {"state": "error",
                                                 "error": record["error"]})

        threading.Thread(target=_run, daemon=True, name="index-build").start()
        return {"ok": True, "state": "running", "project": pid,
                "note": "index build started; watch index.build events"}

    def h_index_search(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        from ...core.indexer import IndexRebuildRequired

        query = str(b.get("query") or "").strip()
        if not query:
            raise HttpError(400, "query required")
        pid = (str(b.get("project") or "").strip() or None)
        idx = self._index(pid)
        try:
            results = idx.search(query,
                                 top_k=int(b["top_k"]) if b.get("top_k") else None,
                                 mode=str(b.get("mode") or "") or None)
        except IndexRebuildRequired as e:
            # Client-actionable state (dim/model mismatch), not a server bug:
            # 409 + code lets the UI offer "rebuild now" instead of a raw 500.
            raise HttpError(409, str(e), code="index_rebuild_required") from e
        except Exception as e:
            raise HttpError(500, str(e)) from e
        return {"results": results}

    def h_index_stop(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        """Stop the embedding server(s) this process spawned (foreign
        processes are never touched — sable only kills its own children)."""
        from ...core.indexer import stop_all_embed_servers

        stop_all_embed_servers()
        return {"ok": True}

    def startup_indexing(self) -> None:
        """Dashboard startup hook (Phase 5): while indexing is enabled, kick
        a non-blocking refresh of the default DB and every project DB. The
        first embed call spawns the owned embedding server; failures land in
        index.build error events, never blocking or crashing startup."""
        ix = getattr(self.cfg, "indexing", None) or {}
        if not ix.get("enabled"):
            return

        def _run() -> None:

            targets: List[Optional[str]] = [None]
            d = self.home / PROJECTS_DIRNAME
            if d.is_dir():
                for f in sorted(d.glob("*.json")):
                    pid = f.stem
                    if _PROJECT_ID_RE.match(pid):
                        targets.append(pid)
            for pid in targets:
                try:
                    idx = self._index(pid)
                    if not idx.roots or not any(r.is_dir() for r in idx.roots):
                        continue
                    # Never auto-index SABLE_HOME itself: the default
                    # (no-project) index falls back to the process cwd, and
                    # for a dashboard launched from ~/.sable that meant
                    # startup silently built a 9k-file index of llama.cpp
                    # build trees, vendored engines and the config file —
                    # noise at best, a config leak at worst. Only an
                    # explicit indexing.paths_override may target it.
                    if pid is None and self._default_root_is_home(idx):
                        self.bus.publish("index.build", {
                            "state": "done", "project": None,
                            "line": "default index skipped: its root is "
                                    "SABLE_HOME; open a project or set "
                                    "indexing.paths_override to index code"})
                        continue
                    self.bus.publish("index.build", {
                        "state": "running", "project": pid,
                        "line": f"startup index refresh ({pid or 'default'})…"})
                    res = idx.build(progress=lambda line, pid=pid: (
                        self.bus.publish(
                            "index.build", {"state": "running", "project": pid,
                                            "line": str(line)})))
                    self.bus.publish("index.build", {
                        "state": "done", "project": pid, **res})
                except Exception as e:
                    self.bus.publish("index.build", {
                        "state": "error", "project": pid,
                        "error": f"{type(e).__name__}: {e}"})

        threading.Thread(target=_run, daemon=True,
                         name="index-startup").start()



#: Same shape rule as ProjectsRoutes._PROJECT_ID_RE (startup index
#: scan walks <home>/projects/*.json directly).
_PROJECT_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

"""Self-update endpoints (core/updater.py) — status, re-check, apply.

GET  /api/update            cached-or-fresh check result (never blocks long)
POST /api/update/check      force a fresh network check (body: {})
POST /api/update/apply      git pull + reinstall on a background thread;
                            progress as `update.apply` SSE events + poll GET
                            /api/update/apply. Refuses while running (409).

Applying is always an explicit user action (dashboard button / CLI); the
GET path only ever reads the throttled cache, so status polling never hits
GitHub. The apply slot is a BackgroundTask: one live update per process.
"""

from __future__ import annotations

from typing import Any, Dict

from ...core import updater
from .base import ApiContext


class UpdateRoutes(ApiContext):
    """Self-update status/apply endpoints."""

    def h_update_get(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        """Update status for the banner: cached check unless ?check=1 forces."""
        force = q.get("check") == "1"
        status = updater.check_for_update(self.cfg, home=self.home, force=force)
        task_rec = getattr(self, "_update_task", None)
        if task_rec is not None and task_rec.record:
            rec = task_rec.record
            status["apply"] = {"state": rec.get("state"),
                               "error": rec.get("error")}
        return status

    def h_update_check(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        """Force a fresh check (user clicked 'Check now')."""
        return updater.check_for_update(self.cfg, home=self.home, force=True)

    def h_update_apply(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        """Start the git pull + reinstall in the background (explicit click)."""

        def _work(record: Dict[str, Any], log) -> None:
            result = updater.apply_update(self.cfg, log_fn=log)
            record["state"] = "done"
            record["result"] = result

        self._update_task.try_start(  # type: ignore[attr-defined]
            _work, conflict_msg="an update is already applying",
            done_extra=lambda r: {"result": r.get("result")})
        return {"ok": True, "state": "running",
                "note": "update running in background; watch update.apply "
                        "events or poll GET /api/update/apply"}

    def h_update_apply_status(self, q: Dict[str, str],
                              b: Dict[str, Any]) -> Dict[str, Any]:
        task = getattr(self, "_update_task", None)
        rec = task.record if task else None  # type: ignore[union-attr]
        if not rec:
            return {"state": "idle"}
        tail = int(q.get("tail", "40"))
        return {**rec, "log": rec["log"][-tail:]}

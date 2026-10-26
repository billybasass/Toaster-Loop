"""Prompt-evolution proposals: gated apply/dismiss (pass 4 #39)."""

# Domain slice of the former monolithic dashboard/app.py: a pure move,
# no behavior change. Handlers resolve through DashboardApp attribute
# lookup (see ROUTES in app.py).

from __future__ import annotations

from typing import Any, Dict, Optional


from ...core.config import load_config
from .base import ApiContext, HttpError, _utcnow


class EvolveRoutes(ApiContext):
    """Prompt-evolution proposals: gated apply/dismiss (pass 4 #39)."""

    def h_evolve_latest(self, q: Dict[str, str],
                        b: Dict[str, Any]) -> Dict[str, Any]:
        """Latest finished evolve proposal for one profile (Profiles panel).

        Compact payload only — lineage stays in the job row. Dismissed or
        already-applied proposals read as flags so the UI can hide/annotate
        them without deleting history.
        """
        from ...evaluation.evolve import (APPLIED_KEY, DISMISSED_KEY, RESULT_KEY)

        profile = str(q.get("profile") or b.get("profile") or "")
        if not profile:
            raise HttpError(400, "profile query parameter required")
        reg = self._registry()
        try:
            candidates = [j for j in reg.list_jobs()
                          if j.kind == "evolve" and j.state == "done"
                          and isinstance((j.params or {}).get(RESULT_KEY), dict)
                          and (j.params or {}).get("profile") == profile]
        finally:
            reg.close()
        if not candidates:
            return {"proposal": None}
        job = candidates[-1]  # list_jobs is queued_at-ordered; newest wins
        res = job.params[RESULT_KEY]
        baseline = res.get("baseline") or {}
        best = res.get("best") or {}

        def _fit(entry: Dict[str, Any]) -> Optional[float]:
            m = entry.get("judge_mean")
            return float(m) if isinstance(m, (int, float)) else None

        return {
            "proposal": {
                "job_id": job.job_id,
                "ts": res.get("ts"),
                "improved": bool(res.get("improved")),
                "diff": res.get("diff") or "",
                "best_variant_id": best.get("variant_id"),
                "best_direction": best.get("direction"),
                "fitness_current": _fit(baseline),
                "fitness_proposed": _fit(best),
                "golden_pass_current": baseline.get("golden_pass"),
                "golden_pass_proposed": best.get("golden_pass"),
                "calls_used": res.get("calls_used"),
                "call_budget": res.get("call_budget"),
                "partial": bool(res.get("partial")),
                "dismissed": bool((job.params or {}).get(DISMISSED_KEY)),
                "applied_at": (job.params or {}).get(APPLIED_KEY),
            }
        }

    def h_evolve_apply(self, q: Dict[str, str],
                       b: Dict[str, Any]) -> Dict[str, Any]:
        """Gated apply: profile config changes ONLY through this explicit user
        action (UI confirm dialog mirrors the promote flow)."""
        from ...evaluation.evolve import (APPLIED_KEY, EvolveError, RESULT_KEY,
                                         apply_prompt_to_config)

        job_id = str(b.get("job_id") or "")
        if not job_id:
            raise HttpError(400, "job_id required")
        reg = self._registry()
        try:
            job = reg.get_job(job_id)
            if job is None or job.kind != "evolve":
                raise HttpError(404, f"no evolve job {job_id}")
            if job.state != "done":
                raise HttpError(409, f"job {job_id} is {job.state}; only "
                                     "finished proposals can be applied")
            res = (job.params or {}).get(RESULT_KEY)
            if not isinstance(res, dict):
                raise HttpError(409, "job has no stored proposal")
            if (job.params or {}).get(APPLIED_KEY):
                raise HttpError(409, "proposal already applied")
            best = res.get("best") or {}
            if not res.get("improved") or not best.get("text"):
                raise HttpError(409, "no improving variant to apply")
            profile_name = str((job.params or {}).get("profile") or "")
            path = self._config_path()
            try:
                out = apply_prompt_to_config(path, profile_name, best["text"])
            except EvolveError as e:
                raise HttpError(404, str(e)) from e
            reg.set_job_param(job_id, APPLIED_KEY, _utcnow())
        finally:
            reg.close()
        # Reflect the new profile prompt in this app instance (same reload the
        # config PUT does; keeps /api/profiles honest right after apply).
        try:
            self.cfg = load_config(str(path))
        except Exception:
            pass
        return {"ok": True, "profile": out.get("profile"),
                "backup": out.get("backup")}

    def h_evolve_dismiss(self, q: Dict[str, str],
                         b: Dict[str, Any]) -> Dict[str, Any]:
        """Flag a proposal dismissed on the job row — never deletes history."""
        from ...evaluation.evolve import DISMISSED_KEY, RESULT_KEY

        job_id = str(b.get("job_id") or "")
        if not job_id:
            raise HttpError(400, "job_id required")
        reg = self._registry()
        try:
            job = reg.get_job(job_id)
            if job is None or job.kind != "evolve":
                raise HttpError(404, f"no evolve job {job_id}")
            if not isinstance((job.params or {}).get(RESULT_KEY), dict):
                raise HttpError(409, "job has no stored proposal")
            reg.set_job_param(job_id, DISMISSED_KEY, _utcnow())
        finally:
            reg.close()
        return {"ok": True, "dismissed": job_id}

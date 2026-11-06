"""Model registry rows/detail/promote/rollback, the discovered
model catalog (#15) and vision-projector (mmproj) endpoints.
"""

# Domain slice of the former monolithic dashboard/app.py: a pure move,
# no behavior change. Handlers resolve through DashboardApp attribute
# lookup (see ROUTES in app.py).

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


from .base import ApiContext, HttpError


class ModelsRoutes(ApiContext):
    """Model registry rows/detail/promote/rollback, the discovered
model catalog (#15) and vision-projector (mmproj) endpoints.
    """

    def h_model_catalog(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        """Models discovered on disk under the base model folder (config
        `models_dir` / TOASTER_LOOP_MODELS_DIR / HF hub cache fallback). Optional
        ?dir= overrides the folder for this request only.

        When nothing explicit is configured, also scan <home>/models — that is
        where the Resource Hub downloads land when no models_dir/HF cache
        exists (ModelHub.base_dir fallback), and discovery used to miss them,
        so the Train/Serve pickers showed less than the Hub's On Device tab.
        Results are merged and de-duplicated by path."""
        from ...core.model_discovery import discover_models

        override = q.get("dir") or self.cfg.models_dir
        result = discover_models(override)
        if not override:
            extra = discover_models(str(self.home / "models"))
            if extra.get("exists"):
                have = {m["path"] for m in result["gguf"]}
                result["gguf"].extend(
                    m for m in extra["gguf"] if m["path"] not in have)
                have_b = {m["path"] for m in result["base_models"]}
                result["base_models"].extend(
                    m for m in extra["base_models"]
                    if m["path"] not in have_b)
                if not result.get("exists"):
                    result.update({"exists": True,
                                   "models_dir": extra.get("models_dir")})
        return result

    def h_model_base_status(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve the HF training base behind a GGUF (core/gguf_base.py):
        {gguf, basename, arch, source_repo, local[], remote[], reason}.

        The Train picker uses this to turn a serve-only GGUF entry into an
        actionable one: ``local`` already trains today, ``remote`` is one hub
        download away (POST /api/model_base/download). Nothing converts GGUF
        weights — the GGUF stays a serving artifact and training always runs
        on real HF safetensors."""
        from ...core.gguf_base import resolve_gguf_base

        gguf = (q.get("gguf") or "").strip()
        if not gguf:
            raise HttpError(400, "gguf query param required")
        from pathlib import Path as _P

        if not _P(gguf).expanduser().is_file():
            raise HttpError(404, f"GGUF not found: {gguf}")
        return resolve_gguf_base(gguf, home=self.home,
                                 models_dir=self.cfg.models_dir)

    def h_model_base_download(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        """Start a hub download of the HF base repo for a GGUF (full safetensors
        variant). Poll GET /api/hub/downloads/<job_id> — same job machinery as
        the Resource Hub tab; lands in the hub cache where discovery picks it
        up automatically."""
        from ...core.modelhub import HubError, validate_repo_id

        repo = str(b.get("repo") or "").strip()
        if not repo:
            raise HttpError(400, "repo required")
        try:
            validate_repo_id(repo)
        except HubError as e:
            raise HttpError(400, str(e)) from e
        hub = getattr(self, "_hub", None)
        if hub is None:
            from ...core.modelhub import ModelHub

            hub = ModelHub(models_dir=self.cfg.models_dir, home=self.home)
            self._hub = hub
        try:
            # variant "" -> the repo's default (non-GGUF repos expose one
            # pseudo-variant "full" covering every file).
            job_id = hub.start_download(repo, "", "models")
        except HubError as e:
            raise HttpError(400, str(e)) from e
        return {"ok": True, "job_id": job_id, "repo": repo,
                "note": "poll GET /api/hub/downloads/<job_id>; the base model "
                        "appears in the Train picker once downloaded"}

    def h_mmproj_status(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        """Vision-projector state for one GGUF: {gguf, local, repo_id,
        remote: [{rfilename,size}], downloadable}."""
        from ...core import mmproj

        gguf = (q.get("gguf") or "").strip()
        if not gguf:
            raise HttpError(400, "gguf query param required")
        from pathlib import Path as _P

        if not _P(gguf).expanduser().is_file():
            raise HttpError(404, f"GGUF not found: {gguf}")
        local = mmproj.find_local(gguf)
        repo_id = mmproj.source_repo_for(gguf)
        remote: list = []
        if local is None and repo_id:
            try:
                remote = mmproj.remote_mmproj_candidates(repo_id)
            except Exception:
                remote = []  # hub unreachable: UI just hides the download button
        return {"gguf": gguf, "local": local, "repo_id": repo_id,
                "remote": remote, "downloadable": local is None and bool(remote)}

    def h_mmproj_download(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        """Download the mmproj companion for a GGUF into the GGUF's own folder
        (background thread; poll GET /api/mmproj?gguf=... until local)."""
        from ...core import mmproj

        gguf = str(b.get("gguf") or "").strip()
        if not gguf:
            raise HttpError(400, "gguf required")
        rfilename = str(b.get("rfilename") or "").strip() or None
        repo_id = str(b.get("repo_id") or "").strip() or None

        def _run() -> None:
            try:
                info = mmproj.download_for(gguf, repo_id=repo_id,
                                           rfilename=rfilename)
                self.bus.publish("mmproj.downloaded", {"gguf": gguf, **info})
            except Exception as e:
                self.bus.publish("mmproj.error",
                                 {"gguf": gguf, "error": str(e)})

        threading.Thread(target=_run, daemon=True, name="mmproj-dl").start()
        return {"ok": True, "note": "mmproj download started; poll "
                                    "GET /api/mmproj?gguf= for completion"}

    def h_models_list(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        reg = self._registry()
        try:
            models = [m.to_dict() for m in reg.list_models()]
            active = reg.active_model()
            jobs = [j.to_dict() for j in reg.list_jobs()]
        finally:
            reg.close()
        # Lineage edges for the UI graph: parent_adapter -> model_id.
        edges = [
            {"from": m["parent_adapter"], "to": m["model_id"]}
            for m in models
            if m.get("parent_adapter")
        ]
        # Backlog item #15: catalog-style join enrichment (additive —
        # existing keys unchanged). Dataset title from the manifest, final step
        # count + duration from the model's last finished train job, losses
        # already carried on the row. `detail` is composed server-side so the
        # panel renders one string; every missing piece is omitted tolerantly.
        train_jobs: Dict[str, Dict[str, Any]] = {}
        for j in jobs:
            if j.get("kind") in ("sft", "dpo", "grpo") and j.get("model_id"):
                train_jobs[j["model_id"]] = j  # later (finished) rows win
        for m in models:
            job = train_jobs.get(m["model_id"])
            title = None
            ds_id = m.get("dataset_id")
            if ds_id:
                mf = self.home / "datasets" / ds_id / "manifest.json"
                try:
                    man = json.loads(mf.read_text())
                    title = man.get("title") or man.get("dataset_id") or ds_id
                except (OSError, ValueError):
                    title = ds_id  # manifest missing/unreadable -> id itself
            steps = _job_final_steps(job.get("log_path")) if job else None
            duration = _job_duration_s(job) if job else None
            m["dataset_title"] = title
            m["train_steps"] = steps
            m["train_duration_s"] = duration
            # All rows here have a registry row (local GGUFs are not listed by
            # this endpoint), so grouping is trivially "trained" — kept as an
            # explicit field for the panel's trained-vs-local grouping.
            m["origin"] = "trained"
            m["detail"] = _model_detail_line(
                title=title, steps=steps, created_at=m.get("created_at"),
                train_loss=m.get("train_loss"), eval_loss=m.get("eval_loss"))
        # Pass 4 #41: derived capability tier per model — computed at read
        # time from evidence (registry/tiers.py), never stored. Display
        # context only; promote/rollback stay permissive.
        self._attach_tiers(models)
        return {
            "models": models,
            "edges": edges,
            "active_model": active.model_id if active else None,
        }

    def _attach_tiers(self, models: list) -> None:
        """Add tier/tier_name/tier_hint to each model dict (pass 4 #41)."""
        from ...registry.tiers import compute_served_stats, derive_tier, \
            requirements_hint, tier_name

        store = self._store()
        try:
            for m in models:
                since = m.get("promoted_at") or m.get("created_at")
                stats = compute_served_stats(
                    store, str(m.get("model_id") or ""), since=since)
                t = derive_tier(m, stats)
                m["tier"] = t
                m["tier_name"] = tier_name(t)
                m["tier_hint"] = requirements_hint(m, stats)
        finally:
            store.close()

    def h_model_get(self, q: Dict[str, str], b: Dict[str, Any],
                    model_id: str) -> Dict[str, Any]:
        reg = self._registry()
        try:
            m = reg.get_model(model_id)
        finally:
            reg.close()
        if m is None:
            raise HttpError(404, f"no model {model_id}")
        d = m.to_dict()
        return {
            "model": d,
            "validation": d.get("validation"),
            "adapter_path": d.get("adapter_dir"),
            "gguf_path": d.get("gguf_path"),
        }

    def h_model_promote(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        model_id = str(b.get("model_id", ""))
        if not model_id:
            raise HttpError(400, "model_id required")
        # Pass 4 #36: optional one-line intent ("promoted X expecting Y");
        # ledgered with the op and reviewable via insights intent_review.
        intent = b.get("intent") or None
        reg = self._registry()
        try:
            try:
                m = reg.promote(model_id, actor="dashboard", intent=intent)
            except Exception as e:
                raise HttpError(409, str(e)) from e
        finally:
            reg.close()
        self.bus.publish("model.promoted", {"model_id": m.model_id,
                                            "base_model": m.base_model})
        return {"ok": True, "model": m.to_dict()}

    def h_model_rollback(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        base = b.get("base_model") or None
        intent = b.get("intent") or None  # pass 4 #36 optional "why"
        reg = self._registry()
        try:
            m = reg.rollback(base_model=base, actor="dashboard", intent=intent)
        finally:
            reg.close()
        self.bus.publish("model.rolled_back",
                         {"model_id": m.model_id if m else None})
        return {"ok": True, "active_model": m.to_dict() if m else None}


# ------------------------------------------- module-level helpers

def _job_final_steps(log_path: Optional[str]) -> Optional[int]:
    """Final step count from a finished train job log (backlog item #15).

    Tolerant parse: prefers an explicit ``steps``/``total`` on the terminal
    sft_done/dpo_done line (batch-#4 protocol carries them on progress
    events); falls back to the largest observed ``step``. Any unreadable or
    malformed log yields None (piece omitted from the detail line).
    """
    if not log_path:
        return None
    try:
        lines = Path(log_path).read_text(errors="replace").splitlines()
    except OSError:
        return None
    best: Optional[int] = None
    for line in lines:
        if '"event"' not in line:
            continue
        try:
            rec = json.loads(line)
        except (ValueError, TypeError):
            continue
        ev = rec.get("event")
        if ev in ("sft_done", "dpo_done"):
            for k in ("steps", "total", "step"):
                v = rec.get(k)
                if isinstance(v, (int, float)) and v >= 0:
                    best = int(v)
                    break
        elif ev == "progress":
            v = rec.get("step")
            if isinstance(v, (int, float)) and (best is None or v > best):
                best = int(v)
    return best

def _job_duration_s(job: Dict[str, Any]) -> Optional[float]:
    """Finished-minus-started seconds for a job row; None when incomplete."""
    started, finished = job.get("started_at"), job.get("finished_at")
    if not started or not finished:
        return None
    try:
        t0 = datetime.fromisoformat(str(started))
        t1 = datetime.fromisoformat(str(finished))
    except ValueError:
        return None
    delta = (t1 - t0).total_seconds()
    return round(delta, 1) if delta >= 0 else None

def _model_detail_line(*, title: Optional[str], steps: Optional[int],
                       created_at: Optional[str], train_loss: Optional[float],
                       eval_loss: Optional[float]) -> str:
    """Compose the catalog-style detail line (backlog item #15).

    ``dataset · N steps · YYYY-MM-DD · loss 0.42/0.51`` — pieces missing from
    older runs are simply omitted; never a placeholder like 'None'.
    """
    parts: List[str] = []
    if title:
        parts.append(str(title))
    if steps is not None:
        parts.append(f"{steps} steps")
    if created_at:
        parts.append(str(created_at)[:10])
    if train_loss is not None:
        loss = f"loss {float(train_loss):.2f}"
        if eval_loss is not None:
            loss += f"/{float(eval_loss):.2f}"
        parts.append(loss)
    return " · ".join(parts)

"""Job queue views + curation/training submission (curate/train/
rft/mine_goldens) and the train check/estimate endpoints.
"""

# Domain slice of the former monolithic dashboard/app.py: a pure move,
# no behavior change. Handlers resolve through DashboardApp attribute
# lookup (see ROUTES in app.py).

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional


from ...curate.curater import Curater, CurationConfig, curation_config_from_yaml
from .base import (ApiContext, HttpError, ApiResponse, logger)


class JobsRoutes(ApiContext):
    """Job queue views + curation/training submission (curate/train/
rft/mine_goldens) and the train check/estimate endpoints.
    """

    def h_jobs_list(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        tail = int(q.get("tail", str(DEFAULT_JOB_LOG_TAIL)) or 0)
        reg = self._registry()
        try:
            jobs = [j.to_dict() for j in reg.list_jobs()]
        finally:
            reg.close()
        if tail > 0:
            for j in jobs:
                lp = j.get("log_path")
                if lp and Path(lp).is_file():
                    try:
                        lines = Path(lp).read_text(errors="replace").splitlines()
                        j["log_tail"] = lines[-tail:]
                    except OSError:
                        j["log_tail"] = []
        return {"jobs": jobs}

    def h_job_get(self, q: Dict[str, str], b: Dict[str, Any],
                  job_id: str) -> Dict[str, Any]:
        reg = self._registry()
        try:
            job = reg.get_job(job_id)
        finally:
            reg.close()
        if job is None:
            raise HttpError(404, f"no job {job_id}")
        offset = max(0, int(q.get("offset", "0") or 0))
        limit = min(5000, max(1, int(q.get("limit", "500") or 500)))
        lines: List[str] = []
        total = 0
        if job.log_path and Path(job.log_path).is_file():
            try:
                all_lines = Path(job.log_path).read_text(errors="replace").splitlines()
                total = len(all_lines)
                lines = all_lines[offset:offset + limit]
            except OSError:
                pass
        return {"job": job.to_dict(),
                "log": {"lines": lines, "offset": offset, "total": total}}

    def h_job_log(self, q: Dict[str, str], b: Dict[str, Any],
                  job_id: str) -> Dict[str, Any]:
        tail = int(q.get("tail", str(DEFAULT_JOB_LOG_TAIL)) or 0)
        reg = self._registry()
        try:
            job = reg.get_job(job_id)
        finally:
            reg.close()
        if job is None:
            raise HttpError(404, f"no job {job_id}")
        lines: List[str] = []
        if job.log_path and Path(job.log_path).is_file():
            try:
                all_lines = Path(job.log_path).read_text(errors="replace").splitlines()
                lines = all_lines[-tail:] if tail > 0 else all_lines
            except OSError:
                pass
        return {"job_id": job_id, "state": job.state, "lines": lines}

    def h_job_cancel(self, q: Dict[str, str], b: Dict[str, Any]) -> ApiResponse:
        job_id = str(b.get("job_id", ""))
        if not job_id:
            raise HttpError(400, "job_id required")
        reg = self._registry()
        try:
            sched = self._scheduler(reg)
            job = sched.cancel(job_id, actor="dashboard")
        finally:
            reg.close()
        if job is None:
            raise HttpError(404, f"no job {job_id}")
        return ApiResponse(200, {"ok": True, "job": job.to_dict()})

    def h_curate(self, q: Dict[str, str], b: Dict[str, Any]) -> ApiResponse:
        # 5d contract change: curate is now a queued job (kind="curate"), not
        # a synchronous request — large buffers must not block the HTTP thread.
        # 202 {ok, job_id, state:"queued"}; follow via job.state events or
        # GET /api/jobs/<id>. On success the job log carries a final line:
        #   {"event": "done", "job": <id>, "result": {dataset_id, sft_count,
        #    dpo_count, claimed, warnings}}
        return ApiResponse(202, self.submit_curate(b))

    def submit_curate(self, b: Dict[str, Any]) -> Dict[str, Any]:
        # Validate filters synchronously so bad bodies still get an immediate
        # 400 before anything is queued.
        self._curation_cfg(b)

        reg = self._registry()
        try:
            sched = self._scheduler(reg)
            job = sched.submit("curate", log_dir=self.home / "curate_runs",
                               params=dict(b))
        finally:
            reg.close()

        def _run(job_id: str, log_path: Optional[str]) -> None:
            # In-process execution (no worker subprocess): curate touches only
            # the buffer DB + filesystem, so it neither needs the GPU nor takes
            # the scheduler's single-flight slot.
            reg = self._registry()
            try:
                if not reg.mark_running(job_id, os.getpid(), actor="dashboard"):
                    return  # cancelled while queued — keep the terminal state
                self._job_observer(job_id, "running")
                try:
                    res = self.curate(b)
                except Exception as e:
                    msg = f"{type(e).__name__}: {e}"
                    if log_path:
                        try:
                            with open(log_path, "a", encoding="utf-8") as f:
                                f.write(json.dumps(
                                    {"event": "error", "job": job_id,
                                     "error": msg}) + "\n")
                        except OSError:
                            pass
                    reg.finish_job(job_id, "error", error=msg, actor="dashboard")
                    self._job_observer(job_id, "error")
                    return
                if log_path:
                    try:
                        with open(log_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(
                                {"event": "done", "job": job_id,
                                 "result": {k: res.get(k) for k in
                                            ("dataset_id", "sft_count",
                                             "dpo_count", "claimed",
                                             "short_claims", "warnings")}},
                                ensure_ascii=False) + "\n")
                    except OSError:
                        pass
                reg.finish_job(job_id, "done", actor="dashboard")
                self._job_observer(job_id, "done")
            except Exception:  # pragma: no cover - defensive
                try:
                    reg.finish_job(job_id, "error", error="curate runner failed",
                                   actor="dashboard")
                except Exception:
                    pass
            finally:
                reg.close()

        threading.Thread(target=_run, args=(job.job_id, job.log_path),
                         daemon=True).start()
        return {"ok": True, "job_id": job.job_id, "state": "queued"}

    def curate(self, b: Dict[str, Any]) -> Dict[str, Any]:
        cc = self._curation_cfg(b)
        store = self._store()
        try:
            curater = Curater(store, self.home)
            res = curater.curate(cc, name=str(b.get("name") or "ui"),
                                 rebuild_id=b.get("rebuild") or None)
        finally:
            store.close()
        return {
            "ok": True,
            "dataset_id": res.dataset_id,
            "sft_count": res.sft_count,
            "dpo_count": res.dpo_count,
            "claimed": res.claimed,
            "short_claims": res.short_claims,
            "warnings": res.warnings,
        }

    def _curation_cfg(self, b: Dict[str, Any]) -> CurationConfig:
        cc = curation_config_from_yaml({"curation": self.cfg.curation or {}})
        if b.get("min_score") is not None:
            cc.min_score = float(b["min_score"])
        if b.get("since"):
            cc.since = str(b["since"])
        if b.get("profile"):
            cc.agent_profile = str(b["profile"])
        if b.get("project"):
            cc.project = str(b["project"])
        if b.get("include_failures"):
            cc.include_failures = True
        if b.get("model_alias"):
            cc.model_alias = str(b["model_alias"])
        if b.get("formats"):
            fmt = str(b["formats"]).lower()
            if fmt not in ("sft", "dpo", "both"):
                raise HttpError(400, "formats must be sft, dpo or both")
            cc.formats = fmt
        if b.get("limit") is not None:
            cc.limit = int(b["limit"])
        if self.cfg.gguf_path and not cc.base_model:
            cc.base_model = self.cfg.gguf_path
        return cc

    def h_train(self, q: Dict[str, str], b: Dict[str, Any]) -> ApiResponse:
        return ApiResponse(202, self.submit_train(b))

    def submit_train(self, b: Dict[str, Any]) -> Dict[str, Any]:
        dataset_id = str(b.get("dataset") or b.get("dataset_id") or "")
        if not dataset_id:
            raise HttpError(400, "dataset required")
        kind = str(b.get("kind") or "sft")
        if kind not in ("sft", "dpo", "grpo"):
            raise HttpError(400, "kind must be sft, dpo or grpo")
        dry_run = bool(b.get("dry_run"))

        tc = self._training_config()
        # Per-request base-model override (the Train wizard's discovered-models
        # dropdown) beats the configured training.base_model.
        if b.get("base_model"):
            tc.base_model = str(b["base_model"])
        reg = self._registry()
        max_par = max(1, int(tc.max_parallel_jobs or 1))
        try:
            # Capacity pre-check (5d) — ADVISORY only. A dead worker's phantom
            # `running` row must not hold every future submit at 409, so reap
            # stale rows before counting; and because the check races any other
            # submitter, the AUTHORITATIVE capacity decision is the atomic
            # claim + device-slot acquisition inside the execution thread (and
            # in the dispatcher). A submit that slips past a just-raced check
            # stays queued and runs when capacity frees — never rejected,
            # never stranded without explanation.
            self._scheduler(reg).reap_stale()
            running = reg.running_jobs()
            if len(running) >= max_par:
                raise HttpError(
                    409,
                    f"job {running[0].job_id} is already running "
                    f"({running[0].kind}); one GPU, one job"
                    if max_par == 1 else
                    f"{len(running)} jobs already running "
                    "(training.max_parallel_jobs reached)",
                    code="already_running",
                )
            sched = self._scheduler(reg)
            job = sched.submit(
                kind, dataset_id=dataset_id, log_dir=self.home / "train_runs",
                params={
                    "train_yaml": str(self.home / "datasets" / dataset_id / "train.yaml"),
                    "dataset_dir": str(self.home / "datasets" / dataset_id),
                    "base_model": tc.base_model or "",
                    "dry_run": dry_run,
                },
            )
        finally:
            reg.close()

        def _run(job_id: str) -> None:
            # Ownership-safe start: claim THIS job with an atomic queued ->
            # running flip under our pid (guarded UPDATE — a losing claim can
            # never double-run it). When capacity is full the job simply stays
            # queued; this thread's drain below picks it up when capacity
            # frees, and the dispatcher tick is the backstop for anything that
            # arrives after this thread exits.
            reg = self._registry()
            try:
                sched = self._scheduler(reg)
                j = reg.get_job(job_id)
                if (j is not None and j.state == "queued"
                        and len(reg.running_jobs()) < max_par):
                    if reg.mark_running(job_id, os.getpid(), actor="dashboard"):
                        try:
                            sched.run_claimed(reg.get_job(job_id) or j)
                        except Exception:
                            # Gate/slot refusal requeues the claim; job state
                            # already reflects any real failure.
                            pass
                # Holder duty: drain remaining queued work instead of assuming
                # another execution will — nothing else would until the next
                # dispatcher tick.
                try:
                    sched.drain_queue()
                except Exception:
                    pass
            except Exception:  # pragma: no cover - defensive
                logger.exception("train execution thread failed for %s", job_id)
            finally:
                reg.close()

        threading.Thread(target=_run, args=(job.job_id,), daemon=True).start()
        return {"ok": True, "job_id": job.job_id, "state": "queued"}

    def h_rft(self, q: Dict[str, str], b: Dict[str, Any]) -> ApiResponse:
        return ApiResponse(202, self.submit_rft(b))

    def submit_rft(self, b: Dict[str, Any]) -> Dict[str, Any]:
        """Queue an rft (rejection-sampling) job through the SAME atomic submit
        path as submit_train — reap pre-check, sched.submit, then a holder
        thread that claims (mark_running) + runs in-process and drains. The
        rft master switch (rft.enabled) is enforced by the job runner itself,
        so a disabled config refuses at run time with an honest job error."""
        tc = self._training_config()
        reg = self._registry()
        max_par = max(1, int(tc.max_parallel_jobs or 1))
        try:
            # Advisory capacity pre-check (identical semantics to submit_train):
            # reap phantoms first; the authoritative decision is the atomic
            # claim inside the execution thread / dispatcher.
            self._scheduler(reg).reap_stale()
            running = reg.running_jobs()
            if len(running) >= max_par:
                raise HttpError(
                    409,
                    f"job {running[0].job_id} is already running "
                    f"({running[0].kind}); one GPU, one job"
                    if max_par == 1 else
                    f"{len(running)} jobs already running "
                    "(training.max_parallel_jobs reached)",
                    code="already_running",
                )
            params: Dict[str, Any] = {}
            for k in ("seeds", "goldens", "golden_file", "name"):
                if b.get(k):
                    params[k] = b[k]
            for k in ("sample_k", "max_prompts"):
                if b.get(k) is not None:
                    params[k] = int(b[k])
            for k in ("judge_min_score", "temperature"):
                if b.get(k) is not None:
                    params[k] = float(b[k])
            sched = self._scheduler(reg)
            job = sched.submit("rft", log_dir=self.home / "train_runs",
                               params=params, enqueue_actor="dashboard")
        finally:
            reg.close()

        def _run(job_id: str) -> None:
            # Ownership-safe start — same shape as submit_train's thread: claim
            # THIS job with an atomic queued -> running flip under our pid; the
            # scheduler runs it IN-PROCESS (kind rft, like shadow/evolve). A
            # gate/slot refusal keeps the job queued for the dispatcher tick.
            reg = self._registry()
            try:
                sched = self._scheduler(reg)
                j = reg.get_job(job_id)
                if (j is not None and j.state == "queued"
                        and len(reg.running_jobs()) < max_par):
                    if reg.mark_running(job_id, os.getpid(), actor="dashboard"):
                        try:
                            sched.run_claimed(reg.get_job(job_id) or j)
                        except Exception:
                            pass  # refusal requeued the claim; state is truth
                try:
                    sched.drain_queue()
                except Exception:
                    pass
            except Exception:  # pragma: no cover - defensive
                logger.exception("rft execution thread failed for %s", job_id)
            finally:
                reg.close()

        threading.Thread(target=_run, args=(job.job_id,), daemon=True).start()
        return {"ok": True, "job_id": job.job_id, "state": "queued"}

    def h_mine_goldens(self, q: Dict[str, str], b: Dict[str, Any]) -> ApiResponse:
        return ApiResponse(202, self.submit_mine_goldens(b))

    def submit_mine_goldens(self, b: Dict[str, Any]) -> Dict[str, Any]:
        """Queue a mine_goldens job through the SAME atomic submit path as
        submit_rft (reap pre-check, sched.submit, holder thread that claims +
        runs in-process and drains). The goldens_mine master switch is enforced
        by the job runner itself, so a disabled config refuses at run time with
        an honest job error."""
        tc = self._training_config()
        reg = self._registry()
        max_par = max(1, int(tc.max_parallel_jobs or 1))
        try:
            self._scheduler(reg).reap_stale()
            running = reg.running_jobs()
            if len(running) >= max_par:
                raise HttpError(
                    409,
                    f"job {running[0].job_id} is already running "
                    f"({running[0].kind}); one GPU, one job"
                    if max_par == 1 else
                    f"{len(running)} jobs already running "
                    "(training.max_parallel_jobs reached)",
                    code="already_running",
                )
            params: Dict[str, Any] = {}
            if b.get("max_per_run") is not None:
                params["max_per_run"] = int(b["max_per_run"])
            if b.get("use_judge"):
                params["use_judge"] = True
            sched = self._scheduler(reg)
            job = sched.submit("mine_goldens", log_dir=self.home / "train_runs",
                               params=params, enqueue_actor="dashboard")
        finally:
            reg.close()

        def _run(job_id: str) -> None:
            reg = self._registry()
            try:
                sched = self._scheduler(reg)
                j = reg.get_job(job_id)
                if (j is not None and j.state == "queued"
                        and len(reg.running_jobs()) < max_par):
                    if reg.mark_running(job_id, os.getpid(), actor="dashboard"):
                        try:
                            sched.run_claimed(reg.get_job(job_id) or j)
                        except Exception:
                            pass  # refusal requeued the claim; state is truth
                try:
                    sched.drain_queue()
                except Exception:
                    pass
            except Exception:  # pragma: no cover - defensive
                logger.exception("mine_goldens execution thread failed for %s",
                                 job_id)
            finally:
                reg.close()

        threading.Thread(target=_run, args=(job.job_id,), daemon=True).start()
        return {"ok": True, "job_id": job.job_id, "state": "queued"}

    def h_train_estimate(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        """Predicted VRAM footprint for a would-be training job (#20).

        GET /api/train/estimate?dataset=<id>&base_model=<path> — the Train
        wizard calls this before submit so the user sees estimate + headroom.
        ``estimated: null`` when the base model's dims are not resolvable
        locally (HF id without cache, GGUF, missing train.yaml): the scheduler
        then falls back to the static floor, same as before.
        """
        from ...core.vram import estimate_from_train_yaml
        from ...training.scheduler import gpu_free_mib

        dataset_id = str(q.get("dataset") or q.get("dataset_id") or "")
        if not dataset_id:
            raise HttpError(400, "dataset required")
        train_yaml = self.home / "datasets" / dataset_id / "train.yaml"
        base_override = str(q.get("base_model") or "").strip() or None
        tc = self._training_config()
        est = estimate_from_train_yaml(
            str(train_yaml), base_override or tc.base_model or None)
        free = gpu_free_mib()
        out: Dict[str, Any] = {
            "estimated": None, "free_mib": free,
            "fits": None, "headroom_mib": None,
        }
        if est is not None:
            out["estimated"] = est.as_dict()
            if free is not None:
                out["headroom_mib"] = free - est.total_mib
                out["fits"] = free >= est.total_mib
        return out

    def h_train_check(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        from ...training.scheduler import (cron_trigger_status,
                                           resource_gate_ok, should_auto_train)

        tc = self._training_config()
        store = self._store()
        try:
            d = should_auto_train(store, min_traces=tc.auto_min_traces,
                                  min_score=tc.auto_min_score)
        finally:
            store.close()
        # 5d: surface the scheduler resource gate (VRAM / AC power) alongside
        # the buffer thresholds so the Train + Hardware panels can explain
        # *why* training would not start.
        gate = resource_gate_ok(tc)
        return {
            "enough_traces": bool(d["trigger"]),
            "unclaimed": d["unclaimed_traces"],
            "mean_score": d["mean_score"],
            "threshold": {"min_traces": d["min_traces"],
                          "min_score": d["min_score"]},
            "gate": gate,
            # Whether ANYTHING will ever call auto-check. The dashboard runs
            # no ticker by design, so thresholds alone never train: without a
            # cron entry the loop is armed but never fired, which looked like
            # "training never starts despite easy training being on".
            "scheduler": cron_trigger_status(),
        }


# ------------------------------------------- module-level helpers

DEFAULT_JOB_LOG_TAIL = 40

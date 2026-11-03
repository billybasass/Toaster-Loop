"""Iterate pipeline submission + persisted iterate run records."""

# Domain slice of the former monolithic dashboard/app.py: a pure move,
# no behavior change. Handlers resolve through DashboardApp attribute
# lookup (see ROUTES in app.py).

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


from ...core.config import ToasterLoopConfig
from ...curate.curater import curation_config_from_yaml
from ...store.paths import resolve_toaster_loop_home
from ...training.easy import (
    acquire_loop_lock,
    release_loop_lock,
)
from .base import (ApiContext, HttpError, ITERATE_RUNS_DIRNAME,
                   ApiResponse, logger, _utcnow)


class IterateRoutes(ApiContext):
    """Iterate pipeline submission + persisted iterate run records."""

    def h_iterate(self, q: Dict[str, str], b: Dict[str, Any]) -> ApiResponse:
        return ApiResponse(202, self.submit_iterate(b))

    def submit_iterate(self, b: Dict[str, Any], *,
                       on_done: Optional[Callable[[Dict[str, Any]], None]] = None,
                       ) -> Dict[str, Any]:
        """Queue a full iterate run in a background thread.

        ``b["profile"]`` scopes the curation stage to one agent profile's
        traces (easy training); absent => whole buffer. ``on_done`` fires with
        the finished run record after the persisted state is saved (the easy
        loop uses it for auto-deploy bookkeeping)."""
        kind = str(b.get("kind") or "sft")
        if kind not in ("sft", "dpo", "grpo"):
            raise HttpError(400, "kind must be sft, dpo or grpo")
        name = b.get("name")
        dataset_id = b.get("dataset") or None
        dry_run = bool(b.get("dry_run"))
        # Per-request base-model override (discovered-models dropdown), same
        # contract as submit_train.
        base_model_override = str(b["base_model"]) if b.get("base_model") else None
        # Curation scope (easy training): only traces stamped with this agent
        # profile feed the dataset. None => unscoped, same as before.
        curate_profile = str(b["profile"]) if b.get("profile") else None

        # Cross-process guard: cron (`train auto-check --trigger` -> cmd_iterate)
        # and a manual click here are separate processes and both curate the
        # shared buffer; _exec_lock only serializes inside this process. The
        # advisory lockfile in <home>/easy_training/ serializes across them so
        # traces can never be double-claimed. Dry runs validate without
        # claiming, so they skip it. (Released when the worker finishes.)
        loop_handle = None
        if not dry_run:
            loop_handle = acquire_loop_lock(self.home)
            if loop_handle is None:
                raise HttpError(
                    409, "another training loop run is active (dashboard or "
                         "cron) — wait for it to finish", code="loop_busy")
        run_id = f"iter-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        runs_dir = self.home / ITERATE_RUNS_DIRNAME
        runs_dir.mkdir(parents=True, exist_ok=True)
        record: Dict[str, Any] = {
            "run_id": run_id,
            "started": _utcnow(),
            "finished": None,
            "state": "running",
            "name": str(name or "ui"),
            "kind": kind,
            "dataset": dataset_id,
            "dry_run": dry_run,
            "steps": [],
            "result": None,
        }
        _iter_tc = self._training_config()
        if base_model_override:
            _iter_tc.base_model = base_model_override
        record_path = runs_dir / f"{run_id}.json"

        def _save() -> None:
            tmp = record_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2),
                           encoding="utf-8")
            os.replace(tmp, record_path)

        _save()

        def _on_stage(stage: str, detail: str) -> None:
            state = ("failed" if detail.startswith("FAILED")
                     else "running" if detail == "running" else "done")
            record["steps"].append({"name": stage, "state": state,
                                    "detail": detail})
            _save()
            self.bus.publish("iterate.step", {"run_id": run_id, "step": stage,
                                              "state": state, "detail": detail})

        def _run() -> None:
            if not self._exec_lock.acquire(blocking=False):
                record["state"] = "error"
                record["steps"].append({"name": "queue", "state": "failed",
                                        "detail": "another execution is running"})
                record["finished"] = _utcnow()
                _save()
                release_loop_lock(loop_handle)
                return
            store = self._store()
            reg = self._registry()
            try:
                from ...training.pipeline import run_iterate

                cc = curation_config_from_yaml({"curation": self.cfg.curation or {}})
                if self.cfg.gguf_path and not cc.base_model:
                    cc.base_model = self.cfg.gguf_path
                if curate_profile:
                    cc.agent_profile = curate_profile
                res = run_iterate(
                    home=self.home, registry=reg,
                    scheduler=self._scheduler(reg),
                    store=store,
                    curation_cfg=cc,
                    training_cfg=_iter_tc,
                    validation_cfg=_validation_cfg_from(self.cfg, home=self.home),
                    name=str(name or "ui"), kind=kind, dataset_id=dataset_id,
                    dry_run=dry_run, on_stage=_on_stage,
                )
                record["result"] = res.to_dict()
                record["state"] = "done" if res.ok else "failed"
            except Exception as e:
                record["state"] = "error"
                record["result"] = {"error": f"{type(e).__name__}: {e}"}
            finally:
                record["finished"] = _utcnow()
                _save()
                self.bus.publish("iterate.step", {"run_id": run_id,
                                                  "step": "finished",
                                                  "state": record["state"],
                                                  "detail": ""})
                store.close()
                reg.close()
                self._exec_lock.release()
                if on_done:
                    try:
                        on_done(record)
                    except Exception:  # completion hooks never break the run
                        logger.exception("iterate: on_done hook failed for %s",
                                         run_id)
                # Released only after the completion hook (easy auto-deploy
                # runs there), so the loop stays owned until fully finished.
                release_loop_lock(loop_handle)

        threading.Thread(target=_run, daemon=True).start()
        return {"ok": True, "run_id": run_id,
                "note": "iterate running in background; poll /api/iterate/runs "
                        "and listen for iterate.step events"}

    def h_iterate_runs(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        runs_dir = self.home / ITERATE_RUNS_DIRNAME
        runs: List[Dict[str, Any]] = []
        if runs_dir.is_dir():
            for f in sorted(runs_dir.glob("iter-*.json"), reverse=True):
                try:
                    runs.append(json.loads(f.read_text()))
                except (OSError, json.JSONDecodeError):
                    continue
        return {"runs": runs}


# ------------------------------------------- module-level helpers

def _validation_cfg_from(cfg: ToasterLoopConfig, home: Optional[Path] = None):
    from ...core.config import resolve_config_path
    from ...curate.mine_goldens import merge_mined_into_tasks
    from ...training.validate import ValidationConfig

    data = cfg.validation or {}
    if not data:
        from ...training.scheduler import load_yaml_section

        # Same resolution as load_config: TOASTER_LOOP_CONFIG > <home>/toaster_loop.yaml.
        path = resolve_config_path()
        data = load_yaml_section(str(path) if path else None, "validation")
    vcfg = ValidationConfig.from_yaml({"validation": data})
    # Auto-mined goldens (curate/mine_goldens.py) join the promotion gate —
    # same merge as cli._validation_config. No mined file -> unchanged list.

    vcfg.golden_tasks = merge_mined_into_tasks(
        vcfg.golden_tasks, home or resolve_toaster_loop_home(cfg))
    return vcfg

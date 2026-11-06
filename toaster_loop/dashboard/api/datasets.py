"""Dataset manifests, row previews, version diff (idea #6),
WP9 analytics and guarded export.
"""

# Domain slice of the former monolithic dashboard/app.py: a pure move,
# no behavior change. Handlers resolve through DashboardApp attribute
# lookup (see ROUTES in app.py).

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from ...curate.curater import Curater
from .base import ApiContext, ApiResponse, HttpError

logger = logging.getLogger("toaster_loop.dashboard.datasets")

#: WP3b one-click install guard: one datasets-install at a time per process.
_ENV_INSTALL: Dict[str, Any] = {"lock": threading.Lock(), "running": False,
                                "result": None}


class DatasetsRoutes(ApiContext):
    """Dataset manifests, row previews, version diff (idea #6),
WP9 analytics and guarded export.
    """

    def h_datasets_list(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        store = self._store()
        try:
            datasets = Curater(store, self.home).list_datasets()
        finally:
            store.close()
        for m in datasets:
            self._attach_lint_info(m)
        return {"datasets": datasets}

    def _attach_lint_info(self, manifest: Dict[str, Any]) -> None:
        """Pass 4 #42: trust + flagged-row summary from the gate sidecars.

        `trust` defaults to "imported" on old manifests (matches the gate).
        `lint` is None for datasets without a flags.json sidecar — honest
        "unlinted" state; the panel offers a lint hint instead of numbers.
        Cheap: two small JSON reads per dataset, no row scans.
        """
        from ...curate.ingest_gate import dataset_trust, load_flags_sidecar

        ds_id = str(manifest.get("dataset_id", ""))
        if not ds_id or manifest.get("error"):
            return
        ds_dir = self.home / "datasets" / ds_id
        if not (ds_dir / "manifest.json").is_file():
            return
        manifest["trust"] = dataset_trust(ds_dir)
        flags = load_flags_sidecar(ds_dir)
        if flags is None:
            manifest["lint"] = None
            return
        per_row = flags.get("flags") or {}
        reasons: Dict[str, int] = {}
        for row_reasons in per_row.values():
            for r in row_reasons if isinstance(row_reasons, list) else []:
                key = str(r).split(":", 1)[0]
                reasons[key] = reasons.get(key, 0) + 1
        manifest["lint"] = {"flagged_rows": len(per_row), "reasons": reasons}

    def h_dataset_get(self, q: Dict[str, str], b: Dict[str, Any],
                      dataset_id: str) -> Dict[str, Any]:
        m = self.get_dataset(dataset_id)
        if m is None:
            raise HttpError(404, "no such dataset")
        self._attach_lint_info(m)
        return m

    def h_dataset_rows(self, q: Dict[str, str], b: Dict[str, Any],
                       dataset_id: str) -> Dict[str, Any]:
        fkey = (q.get("file") or "sft").lower()
        if fkey not in ("sft", "dpo"):
            raise HttpError(400, "file must be sft or dpo")
        offset = max(0, int(q.get("offset", "0") or 0))
        limit = min(200, max(1, int(q.get("limit", "25") or 25)))

        ds_dir = self.home / "datasets" / dataset_id
        path = ds_dir / f"{fkey}.jsonl"
        if not (ds_dir / "manifest.json").is_file():
            raise HttpError(404, "no such dataset")
        rows: List[Dict[str, Any]] = []
        total = 0
        truncated = False
        if path.is_file():
            with open(path, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if not line.strip():
                        continue
                    if i >= offset and len(rows) < limit:
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            row = {"error": "unparseable row"}
                        row, was_trunc = _truncate_row(row)
                        truncated = truncated or was_trunc
                        rows.append(row)
                    total += 1
        return {"file": fkey, "offset": offset, "limit": limit, "total": total,
                "truncated": truncated, "rows": rows}

    def h_dataset_delete(self, q: Dict[str, str], b: Dict[str, Any],
                         dataset_id: str) -> Dict[str, Any]:
        """DELETE /api/datasets/<id> — remove a dataset directory for good.

        Releases the dataset's trace claims first (the same release the
        rebuild path uses), so traces claimed from the buffer become curatable
        again instead of orphaned forever; source-derived datasets hold no
        claims and just vanish. Refuses ids that could escape the datasets
        root, and 404s when there is nothing to delete."""
        import re as _re
        import shutil

        if not dataset_id or not _re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*",
                                               dataset_id):
            raise HttpError(400, "invalid dataset id")
        ds_dir = (self.home / "datasets").resolve() / dataset_id
        root = (self.home / "datasets").resolve()
        if root not in ds_dir.parents:
            raise HttpError(400, "invalid dataset id")
        if not (ds_dir / "manifest.json").is_file():
            raise HttpError(404, "no such dataset")
        store = self._store()
        try:
            released = store.reset_claims(dataset_id)
        except Exception:  # noqa: BLE001 — claims are best-effort on delete
            released = 0
        finally:
            store.close()
        shutil.rmtree(ds_dir)
        logger.info("dataset %s deleted (%d trace claims released)",
                    dataset_id, released)
        return {"ok": True, "dataset_id": dataset_id,
                "claims_released": released}

    def h_dataset_diff(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        """Idea #6: diff two dataset versions by manifest + content hashes.

        GET /api/datasets/diff?a=<id>&b=<id>. Missing datasets -> 404,
        malformed manifests / missing params -> 400 (honest errors from the
        shared curate/dataset_diff module, also used by `toaster_loop datasets diff`).
        """
        from ...curate.dataset_diff import DatasetDiffError, diff_datasets

        a_id = str(q.get("a") or "").strip()
        b_id = str(q.get("b") or "").strip()
        if not a_id or not b_id:
            raise HttpError(400, "query params a and b (dataset ids) required")
        try:
            return diff_datasets(self.home / "datasets", a_id, b_id)
        except DatasetDiffError as e:
            raise HttpError(e.status, e.message) from e

    def h_analytics_curation(self, q: Dict[str, str],
                             b: Dict[str, Any]) -> Dict[str, Any]:
        """WP9a: curation funnel history + latest dataset diff summary.

        GET /api/analytics/curation[?dataset=<id>][&limit=N]. History is the
        append-only JSONL written by Curater.curate() after each run (see
        store/analytics.py). ``latest_diff`` reuses curate/dataset_diff over
        the two newest distinct dataset ids in the filtered history — the
        diff logic is never duplicated here; a missing/unreadable pair is
        reported honestly as {error} rather than failing the endpoint.
        """
        from ...curate.dataset_diff import DatasetDiffError, diff_datasets
        from ...store.analytics import read_curation_funnels

        dataset = str(q.get("dataset") or "").strip() or None
        try:
            limit = max(1, min(1000, int(q.get("limit", "200") or 200)))
        except ValueError:
            raise HttpError(400, "limit must be an integer") from None

        history = read_curation_funnels(self.home, dataset=dataset,
                                        limit=limit)
        # newest distinct dataset ids (oldest-first history -> reverse walk)
        ordered_ids: List[str] = []
        for rec in reversed(history):
            did = str(rec.get("dataset_id") or "")
            if did and did not in ordered_ids:
                ordered_ids.append(did)
        latest_diff: Optional[Dict[str, Any]] = None
        if len(ordered_ids) >= 2:
            b_id, a_id = ordered_ids[0], ordered_ids[1]  # newest pair, older=a
            try:
                latest_diff = diff_datasets(self.home / "datasets", a_id, b_id)
            except DatasetDiffError as e:
                latest_diff = {"error": e.message, "status": e.status}
        return {"dataset": dataset, "history": history,
                "latest_diff": latest_diff}

    def h_analytics_validation(self, q: Dict[str, str],
                               b: Dict[str, Any]) -> Dict[str, Any]:
        """WP9b: validation-gate history + promotion/rollback ledger events.

        GET /api/analytics/validation[?model=<id>][&limit=N]. Validation runs
        come from the append-only analytics JSONL (written by the training
        pipeline after each gate); promote/rollback events come from the
        hash-chained registry ledger. Both are merged chronologically into a
        single ``timeline`` so pass-rate trends and their promotion outcomes
        read as one story per model.
        """
        from ...store.analytics import read_validation_runs

        model = str(q.get("model") or "").strip() or None
        try:
            limit = max(1, min(1000, int(q.get("limit", "200") or 200)))
        except ValueError:
            raise HttpError(400, "limit must be an integer") from None

        runs = read_validation_runs(self.home, model=model, limit=limit)

        # Promotion/rollback events from the registry ledger (the durable
        # source of truth — already hash-chained; analytics never re-writes it).
        events: List[Dict[str, Any]] = []
        reg = self._registry()
        try:
            for op in reg.history(limit=1000):
                if op.get("op") not in ("promote", "rollback"):
                    continue
                if model and str(op.get("target_id") or "") != model:
                    continue
                events.append({
                    "ts": op.get("ts"),
                    "kind": "promotion",
                    "op": op.get("op"),
                    "model_id": op.get("target_id"),
                    "actor": op.get("actor"),
                    "intent": op.get("intent"),
                })
        finally:
            reg.close()

        timeline = sorted(
            [{"kind": "validation", **r} for r in runs] + events,
            key=lambda e: str(e.get("ts") or ""))
        return {"model": model, "history": runs, "events": events,
                "timeline": timeline[-limit:]}

    def h_dataset_export(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        dataset_id = str(b.get("dataset_id", ""))
        to = str(b.get("to", ""))
        if not dataset_id or not to:
            raise HttpError(400, "dataset_id and to required")
        # 5d containment guard: refuse destinations outside TOASTER_LOOP_HOME
        # unless the caller explicitly forces it (UI confirms first).
        force = bool(b.get("force"))
        if not force:
            home_r = self.home.expanduser().resolve(strict=False)
            dest_r = Path(to).expanduser().resolve(strict=False)
            if dest_r != home_r and home_r not in dest_r.parents:
                raise HttpError(
                    409,
                    f"destination {dest_r} is outside TOASTER_LOOP_HOME ({home_r}); "
                    "re-send with force=true to export anyway",
                    code="outside_home",
                )
        store = self._store()
        try:
            try:
                target = Curater(store, self.home).export_dataset(
                    dataset_id, Path(to))
            except FileNotFoundError as e:
                raise HttpError(404, str(e)) from e
            except FileExistsError as e:
                raise HttpError(409, str(e), code="target_exists") from e
        finally:
            store.close()
        return {"ok": True, "path": str(target)}

    def get_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        store = self._store()
        try:
            return Curater(store, self.home).load_manifest(dataset_id)
        finally:
            store.close()

    # ------------------------------------------- WP3c/d: source curation

    def h_dataset_import_hub(self, q: Dict[str, str],
                             b: Dict[str, Any]) -> Dict[str, Any]:
        """WP3c: materialize a downloaded hub dataset into <home>/datasets/.

        POST /api/datasets/import_hub {repo_id, name?}. The snapshot is already
        on disk (Hub page download), so this is pure local work: importer.py
        auto-detection + ingest gate over the snapshot files -> a normal
        dataset record (provenance "hub" + source_repo) that the Train panel
        picker consumes with zero special cases. Synchronous by design (file
        scan only; no network, seconds at worst)."""
        from ...core.modelhub import HubError
        from ...curate.importer import ImportError_ as CurateImportError
        from ...curate.sources import import_hub_snapshot

        repo_id = str(b.get("repo_id") or "").strip()
        if not repo_id:
            raise HttpError(400, "repo_id required")
        try:
            res = import_hub_snapshot(self.home, repo_id,
                                      name=str(b.get("name") or "") or None)
        except CurateImportError as e:
            # Not-found snapshots are 404s; unusable content is a 422 — the
            # hub UI shows the message verbatim.
            status = 404 if "no downloaded hub snapshot" in str(e) else 422
            raise HttpError(status, str(e)) from e
        except HubError as e:
            raise HttpError(400, str(e)) from e   # bad repo id (validate_repo_id)
        return {"ok": True, "dataset_id": res.dataset_id,
                "rows": res.rows, "warnings": res.warnings}

    def h_curate_source(self, q: Dict[str, str],
                        b: Dict[str, Any]) -> "ApiResponse":
        """WP3d: queue a local-folder or website curation job.

        POST /api/datasets/curate_source
          {source: "local_folder", path: "/some/dir", name?, server?}
          {source: "website", url: "https://docs.example.org/", name?,
           max_pages?, max_depth?, server?}
        ``server`` names a serving instance (Phase 6) to drive synthesis;
        omitted = the default serve endpoint, nothing running = heading
        fallback. The job log streams {"event":"progress", message} lines.
        Same 202+poll contract as /api/curate (kind="curate_source"): large
        folders/crawls plus LLM synthesis must never block the HTTP thread.
        The job's final log line carries {"event": "done", "result":
        {dataset_id, rows, warnings}}."""
        from ...curate.sources import CurateCancelled
        from .base import ApiResponse

        source = str(b.get("source") or "").strip().lower()
        if source not in ("local_folder", "website"):
            raise HttpError(400, 'source must be "local_folder" or "website"')
        if source == "local_folder":
            path = str(b.get("path") or "").strip()
            if not path:
                raise HttpError(400, "path required for local_folder")
            if not Path(path).expanduser().is_dir():
                raise HttpError(400, f"not a directory: {path}")
        else:
            url = str(b.get("url") or "").strip()
            if not url:
                raise HttpError(400, "url required for website")
            if urlparse(url).scheme not in ("http", "https"):
                raise HttpError(400, "only http(s) URLs can be crawled")

        reg = self._registry()
        try:
            sched = self._scheduler(reg)
            job = sched.submit("curate_source",
                               log_dir=self.home / "curate_runs",
                               params=dict(b))
        finally:
            reg.close()

        def _run(job_id: str, log_path: Optional[str]) -> None:
            # In-process execution (like submit_curate): the crawl/folder scan
            # plus per-chunk synthesis through the live serve endpoint is CPU/
            # IO work with no GPU dependency; the scheduler slot machinery for
            # training jobs does not apply.

            def _log(rec: Dict[str, Any]) -> None:
                if not log_path:
                    return
                try:
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                except OSError:
                    pass

            def _progress(msg: str) -> None:
                # Progress lines land in the job log so long folder/crawl
                # builds are visible live (Jobs panel + curateJob poller)
                # instead of running silent until completion.
                _log({"event": "progress", "job": job_id, "message": msg})

            reg = self._registry()
            try:
                if not reg.mark_running(job_id, os.getpid(), actor="dashboard"):
                    return  # cancelled while queued
                self._job_observer(job_id, "running")

                def _should_cancel() -> bool:
                    # Cooperative stop: the Jobs page cancels by stamping the
                    # registry; the crawl/synthesis loop observes it here and
                    # aborts within one page/chunk instead of running to the
                    # page cap.
                    try:
                        j = reg.get_job(job_id)
                        return bool(j and j.state == "cancelled")
                    except Exception:  # noqa: BLE001 — probe never fatal
                        return False

                try:
                    res = self._curate_source_run(b, progress=_progress,
                                                  should_cancel=_should_cancel)
                except CurateCancelled:
                    _log({"event": "cancelled", "job": job_id,
                          "result": {"message":
                                     "curation cancelled — partial rows were "
                                     "not saved"}})
                    reg.finish_job(job_id, "cancelled", actor="dashboard")
                    self._job_observer(job_id, "cancelled")
                    return
                except Exception as e:  # noqa: BLE001 — job errors are data
                    msg = f"{type(e).__name__}: {e}"
                    if log_path:
                        try:
                            with open(log_path, "a", encoding="utf-8") as f:
                                f.write(json.dumps(
                                    {"event": "error", "job": job_id,
                                     "error": msg}) + "\n")
                        except OSError:
                            pass
                    reg.finish_job(job_id, "error", error=msg,
                                   actor="dashboard")
                    self._job_observer(job_id, "error")
                    return
                if log_path:
                    try:
                        with open(log_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(
                                {"event": "done", "job": job_id,
                                 "result": {"dataset_id": res.dataset_id,
                                            "rows": res.rows,
                                            "warnings": res.warnings[:20]}},
                                ensure_ascii=False) + "\n")
                    except OSError:
                        pass
                reg.finish_job(job_id, "done", actor="dashboard")
                self._job_observer(job_id, "done")
            except Exception:  # noqa: BLE001 — pragma: no cover defensive
                try:
                    reg.finish_job(job_id, "error",
                                   error="curate_source runner failed",
                                   actor="dashboard")
                except Exception:
                    pass
            finally:
                reg.close()

        threading.Thread(target=_run, args=(job.job_id, job.log_path),
                         daemon=True).start()
        return ApiResponse(202, {"ok": True, "job_id": job.job_id,
                                 "state": "queued"})

    def _curate_source_run(self, b: Dict[str, Any], *,
                           progress: Optional[Any] = None,
                           should_cancel: Optional[Any] = None):
        """Execute a curate_source body (folder or website) with live-model
        synthesis when a server is up. Split out so tests can call it inline.

        ``server`` optionally names a serving instance (Phase 6) whose
        chat-completions drive the doc->ShareGPT synthesis; omitted/"default"
        uses the legacy serve endpoint, and nothing serving falls back to the
        deterministic heading synth. ``lang`` restricts website crawls to one
        language (auto-detected from the environment when omitted; "*" = keep
        everything)."""
        from ...curate.sources import crawl_website, ingest_folder, \
            live_synth_fn

        source = str(b.get("source") or "").strip().lower()
        server_id = str(b.get("server") or "").strip() or None
        synth = live_synth_fn(self, server_id)
        if progress is not None:
            progress("synthesis: " + (f"live model via {server_id or 'default'} "
                     "server" if synth else
                     "heading fallback (no server running)"))
        name = str(b.get("name") or "") or None
        if source == "local_folder":
            return ingest_folder(self.home, str(b.get("path") or ""),
                                 name=name, synth_fn=synth,
                                 progress=progress,
                                 should_cancel=should_cancel)
        kwargs: Dict[str, Any] = {}
        if b.get("max_pages") is not None:
            kwargs["max_pages"] = int(b["max_pages"])
        if b.get("max_depth") is not None:
            kwargs["max_depth"] = int(b["max_depth"])
        if str(b.get("lang") or "").strip():
            kwargs["lang"] = str(b["lang"]).strip()
        # Optional crawl scope override (path subtree on the same host). The
        # default derives from the entry page's directory; "" = whole origin.
        if b.get("scope") is not None:
            kwargs["path_prefix"] = str(b["scope"])
        return crawl_website(self.home, str(b.get("url") or ""), name=name,
                             synth_fn=synth, progress=progress,
                             should_cancel=should_cancel, **kwargs)

    # --------------------------------------------- WP3b: datasets env gate

    def h_datasets_env(self, q: Dict[str, str],
                       b: Dict[str, Any]) -> Dict[str, Any]:
        """GET /api/datasets/env — does `datasets` import in the training
        interpreter? The Datasets page shows an install button when not."""
        from ...core.datasetenv import preflight

        return preflight(self.cfg)

    def h_datasets_env_install(self, q: Dict[str, str],
                               b: Dict[str, Any]) -> "ApiResponse":
        """POST /api/datasets/env/install — one-click `datasets` install into
        the training interpreter (explicit user action; subprocess pip on a
        worker thread, poll GET /api/datasets/env). Deliberately without
        bitsandbytes so it works even where setup.ps1 skipped [training]."""
        from .base import ApiResponse

        reg_state = _ENV_INSTALL
        with reg_state["lock"]:
            if reg_state["running"]:
                return ApiResponse(409, {"ok": False,
                                         "error": "install already running"})
            reg_state["running"] = True

        def _worker() -> None:
            from ...core.datasetenv import install_datasets

            try:
                res = install_datasets()
                reg_state["result"] = res
            except Exception as e:  # noqa: BLE001 — surfaced via env GET
                reg_state["result"] = {"ok": False, "error": str(e)}
            finally:
                reg_state["running"] = False

        threading.Thread(target=_worker, daemon=True).start()
        return ApiResponse(202, {"ok": True, "state": "installing",
                                 "note": "poll GET /api/datasets/env"})


# ------------------------------------------- module-level helpers

ROW_PREVIEW_CHARS = 800          # per-string truncation cap in row previews

def _truncate_row(row: Any) -> Tuple[Any, bool]:
    """Cap long strings in a preview row; returns (row, was_truncated)."""
    trunc = False

    def walk(v: Any) -> Any:
        nonlocal trunc
        if isinstance(v, str):
            if len(v) > ROW_PREVIEW_CHARS:
                trunc = True
                return v[:ROW_PREVIEW_CHARS] + "…"
            return v
        if isinstance(v, dict):
            return {k: walk(x) for k, x in v.items()}
        if isinstance(v, list):
            return [walk(x) for x in v]
        return v

    return walk(row), trunc

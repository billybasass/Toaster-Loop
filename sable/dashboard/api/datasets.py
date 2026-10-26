"""Dataset manifests, row previews, version diff (idea #6),
WP9 analytics and guarded export.
"""

# Domain slice of the former monolithic dashboard/app.py: a pure move,
# no behavior change. Handlers resolve through DashboardApp attribute
# lookup (see ROUTES in app.py).

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


from ...curate.curater import Curater
from .base import ApiContext, HttpError


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

    def h_dataset_diff(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        """Idea #6: diff two dataset versions by manifest + content hashes.

        GET /api/datasets/diff?a=<id>&b=<id>. Missing datasets -> 404,
        malformed manifests / missing params -> 400 (honest errors from the
        shared curate/dataset_diff module, also used by `sable datasets diff`).
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
        # 5d containment guard: refuse destinations outside SABLE_HOME
        # unless the caller explicitly forces it (UI confirms first).
        force = bool(b.get("force"))
        if not force:
            home_r = self.home.expanduser().resolve(strict=False)
            dest_r = Path(to).expanduser().resolve(strict=False)
            if dest_r != home_r and home_r not in dest_r.parents:
                raise HttpError(
                    409,
                    f"destination {dest_r} is outside SABLE_HOME ({home_r}); "
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

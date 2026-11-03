"""WP9 curation/validation analytics — append-only JSONL history under home.

Storage design (documented choice, WP9): two append-only JSONL files under
``<home>/analytics/``, one record per run:

    <home>/analytics/curation_funnels.jsonl   one funnel snapshot per curate run
    <home>/analytics/validation_runs.jsonl    one record per validation-gate run

Why JSONL and not a table in the buffer DB: the curation funnel is produced by
``Curater.curate()`` which already treats the filesystem (dataset dirs +
manifest.json) as its artifact store, and validation reports land on the
registry row plus the ledger. A schema migration for two small append-only
history streams would touch ``store/db.py``'s versioned schema for no query
benefit — history sizes are personal-loop scale (hundreds of lines), reads are
"tail the file", and append-only JSONL is crash-safe the same way the trace
JSONL transcripts are (line-granular writes, readers skip malformed tails).
History accumulates across runs; nothing here is ever rewritten in place.

Every write goes through :func:`append_record` which creates the directory on
first use and appends exactly one JSON line. Readers tolerate missing files
(empty history) and corrupt lines (skipped, honest partial reads).

WP9 metric-honesty rule carried into the schema: any rate a consumer derives
from a snapshot must have its denominator in the same record — funnel records
carry every stage count (traces_scanned -> eligible -> claimed -> curated plus
dedup_rejects/quality_rejects/short_claims), and score histograms carry
``scored`` beside ``unscored`` so "mean over 3 of 400" cannot masquerade as a
coverage claim (WP2c precedent, dashboard/metrics.py).
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ANALYTICS_DIRNAME = "analytics"
CURATION_FILE = "curation_funnels.jsonl"
VALIDATION_FILE = "validation_runs.jsonl"

# Score histogram: 10 bins over [0, 1] (WP9 spec). Bin i covers
# [i/10, (i+1)/10); score == 1.0 lands in the last bin. Scores outside [0, 1]
# clamp to the edge bins (heuristic scores are documented 0..1; judge verdicts
# too — clamping keeps a weird row visible instead of silently dropped).
SCORE_BINS = 10

_write_lock = threading.Lock()


def analytics_dir(home: Path) -> Path:
    return Path(home) / ANALYTICS_DIRNAME


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_record(home: Path, filename: str, record: Dict[str, Any]) -> bool:
    """Append one JSON line to ``<home>/analytics/<filename>``.

    Returns True on success; False (with a debug-grade warning) on any I/O
    failure — analytics persistence must NEVER break the run that produced
    it (same posture as the ledger-optional playbook flips, but here even the
    warning is only logged)."""
    path = analytics_dir(home) / filename
    line = json.dumps(record, ensure_ascii=False, default=str)
    try:
        with _write_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
        return True
    except OSError:
        import logging
        logging.getLogger("toaster_loop.analytics").warning(
            "could not persist analytics record to %s (continuing)", path,
            exc_info=True)
        return False


def read_records(home: Path, filename: str,
                 limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Oldest-first records from one analytics JSONL file.

    Missing file -> []; malformed lines are skipped (an interrupted write must
    never poison history reads). ``limit`` keeps the newest N records."""
    path = analytics_dir(home) / filename
    if not path.is_file():
        return []
    out: List[Dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn tail / manual edit — skip honestly
                if isinstance(rec, dict):
                    out.append(rec)
    except OSError:
        return []
    if limit is not None and limit > 0:
        out = out[-limit:]
    return out


# ------------------------------------------------------------- score histogram

def score_histogram(scores: List[Optional[float]]) -> Dict[str, Any]:
    """10-bin [0,1] histogram of effective scores + honest denominators.

    ``scored`` + ``unscored`` travel with the buckets (WP2c: a distribution is
    only meaningful over its stated coverage; NULL scores are counted, never
    coerced to 0)."""
    bins = [0] * SCORE_BINS
    scored = unscored = 0
    for s in scores:
        if s is None:
            unscored += 1
            continue
        scored += 1
        idx = int(float(s) * SCORE_BINS)
        if idx < 0:
            idx = 0
        if idx >= SCORE_BINS:
            idx = SCORE_BINS - 1
        bins[idx] += 1
    return {
        "bins": bins,
        "bin_edges": [round(i / SCORE_BINS, 1) for i in range(SCORE_BINS + 1)],
        "scored": scored,          # denominator of any derived distribution rate
        "unscored": unscored,      # NULL scores — never coerced to a bin
        "total": scored + unscored,
    }


# ------------------------------------------------------------- curation funnel

def record_curation_funnel(home: Path, snapshot: Dict[str, Any]) -> bool:
    """Persist one post-run funnel snapshot (append-only; additive history)."""
    rec = {"ts": _utcnow()}
    rec.update(snapshot)
    return append_record(home, CURATION_FILE, rec)


def read_curation_funnels(home: Path, dataset: Optional[str] = None,
                          limit: Optional[int] = 200) -> List[Dict[str, Any]]:
    """Funnel snapshots oldest-first; optional exact dataset-id filter."""
    recs = read_records(home, CURATION_FILE,
                        limit=None if dataset else limit)
    if dataset:
        recs = [r for r in recs if r.get("dataset_id") == dataset]
        if limit is not None and limit > 0:
            recs = recs[-limit:]
    return recs


# --------------------------------------------------------- validation telemetry

def record_validation_run(home: Path, snapshot: Dict[str, Any]) -> bool:
    """Persist one validation-gate outcome (append-only)."""
    rec = {"ts": _utcnow()}
    rec.update(snapshot)
    return append_record(home, VALIDATION_FILE, rec)


def read_validation_runs(home: Path, model: Optional[str] = None,
                         limit: Optional[int] = 200) -> List[Dict[str, Any]]:
    """Validation-gate snapshots oldest-first; optional model-id filter."""
    recs = read_records(home, VALIDATION_FILE,
                        limit=None if model else limit)
    if model:
        recs = [r for r in recs if r.get("model_id") == model]
        if limit is not None and limit > 0:
            recs = recs[-limit:]
    return recs


def summarize_validation_report(report: Dict[str, Any], model_id: str,
                                source: str = "pipeline",
                                iteration: Optional[int] = None) -> Dict[str, Any]:
    """Compact telemetry record from a ``validate_model`` report dict.

    Keeps the per-task pass/fail summary (passed/failed/skipped + overall
    pass_rate + gate verdict ok/min_pass_rate) but drops reply previews — the
    full report already lives on the registry row; analytics is the trend
    line, not the archive. ``skipped`` counts tasks that never produced a
    verdict because of transport errors (error set, no checks list)."""
    results = report.get("results") or []
    passed = failed = skipped = 0
    for r in results:
        if not isinstance(r, dict):
            continue
        if r.get("error") and not r.get("checks"):
            skipped += 1
        elif r.get("pass"):
            passed += 1
        else:
            failed += 1
    total = report.get("total")
    total = int(total) if isinstance(total, (int, float)) else len(results)
    denom = passed + failed
    return {
        "model_id": str(model_id),
        "source": source,
        "iteration": iteration,
        "started_at": report.get("started_at"),
        "finished_at": report.get("finished_at"),
        "gate_verdict": "pass" if report.get("ok") else "fail",
        "error": report.get("error"),
        "total": total,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        # Honest pass rate: denominator is the tasks that actually ran
        # (passed+failed); skipped tasks are reported separately so a suite
        # where everything errored transport-side reads as 0 ran, not 0%.
        "pass_rate": report.get("pass_rate"),
        "min_pass_rate": report.get("min_pass_rate"),
        "ran_tasks": denom,
    }

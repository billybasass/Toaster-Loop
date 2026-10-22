"""Batch trace generation with content-based resume (backlog item #8).

Batch design built in
(attribution: atomic checkpoint writes à la ``_save_checkpoint``, the
content-matching resume of ``_scan_completed_prompts_by_content`` — matching
on prompt *text*, not line index, so reordering/duplicating the input file
never re-runs finished work — and deliberately NOT checkpointing infra
failures so they retry on resume). Multiprocessing Pool is dropped:
sable drives a single local GPU, so batch tasks run sequentially through
the same runner/recorder path as one-shot ``sable run`` calls. Output
traces are ordinary buffer records; curation picks them up unchanged.

Resume semantics (the valuable part):
  * A prompt line is DONE when a trace header exists whose recorded task
    CONTENT matches it and whose ``failure_reason`` is NOT infra-ish
    (core/failure_types.is_infra_reason). Infra faults (connection /
    timeout / model_not_found) auto-retry on resume; model faults
    (max_iterations, guardrail_halt, …) count as done — they are training
    signal, not transport noise.
  * The JSONL buffer is the source of truth, never the SQLite index: the
    indexed ``task_summary`` is redacted AND truncated to 300 chars, so it
    cannot identify a task exactly. Trace headers carry the full (redacted)
    task, and matching compares ``task_key()`` on both sides — i.e. equality
    means: identical text after secret-redaction and whitespace trimming.
    Redaction is idempotent, so comparing redact_text(prompt) against the
    stored header form is exact whether or not the prompt contains secrets.
  * The checkpoint at ``<home>/batches/<run_name>.json`` is STATS ONLY: when
    it disagrees with the trace scan, the scan wins (principle:
    content is truth). It exists to name the run, remember ``started_at``
    (so the scan can ignore unrelated older traces) and report per-task
    stats. The schema reserves a ``discarded`` tombstone field for a future
    --discard path; v0 never writes one (documented deviation).

The loop is factored into :func:`run_batch`, which takes an injected
``run_fn(task) -> (trace_id, failure_reason)`` so tests drive it directly —
no subprocesses. The CLI wires ``run_fn`` to a real runner+recorder.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .failure_types import is_infra_reason
from .redact import redact_text

CHECKPOINT_SCHEMA_VERSION = 1
TASK_PREVIEW_CHARS = 60

#: run_fn(task) -> (trace_id or None, failure_reason or None)
RunFn = Callable[[str], Tuple[Optional[str], Optional[str]]]


class BatchInputError(Exception):
    """Malformed batch input file; the message names the offending line."""


# --------------------------------------------------------------------- input

def load_tasks(path: Path | str) -> List[str]:
    """Parse a batch task file into a list of task strings.

    Schema (documented in ``sable run --help`` / README): one JSON value
    per line — either ``{"task": "..."}`` (extra keys are ignored in v0) or
    a bare JSON string. Blank/whitespace-only lines are tolerated. Anything
    else raises :class:`BatchInputError` with the 1-based line number.
    """
    path = Path(path)
    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        raise BatchInputError(f"batch file not found: {path}") from None
    except OSError as e:
        raise BatchInputError(f"cannot read batch file {path}: {e}") from None

    tasks: List[str] = []
    for lineno, line in enumerate(raw_lines, 1):
        if not line.strip():
            continue  # tolerate blank lines
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise BatchInputError(
                f"{path}:{lineno}: invalid JSON line: {e.msg} "
                "(each line must be a JSON object with a \"task\" key or a "
                "bare JSON string)"
            ) from None
        if isinstance(obj, str):
            text = obj.strip()
        elif isinstance(obj, dict):
            task = obj.get("task")
            if not isinstance(task, str) or not task.strip():
                raise BatchInputError(
                    f"{path}:{lineno}: object line must have a non-empty "
                    "\"task\" string"
                )
            text = task.strip()
        else:
            raise BatchInputError(
                f"{path}:{lineno}: expected an object with \"task\" or a "
                f"bare string, got {type(obj).__name__}"
            )
        if text:
            tasks.append(text)
    return tasks


# ------------------------------------------------------------- match keys

def task_key(text: str) -> str:
    """Normalized identity of a task for content matching.

    Equality means: identical after ``redact_text`` (the same transform the
    recorder applies to the durable header — it is idempotent, so re-running
    it over an already-redacted stored task is a no-op) and whitespace
    trimming. Truncation never enters the comparison because matching reads
    full trace *headers* from the JSONL buffer, not the SQLite 300-char
    ``task_summary``.
    """
    return redact_text((text or "").strip())


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_iso(ts: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None


# ----------------------------------------------------------- trace scanning

def scan_completed(home: Path | str,
                   since: Optional[str] = None) -> Dict[str, dict]:
    """Content-scan the JSONL buffer for completed tasks.

    Reads only each trace's first line (``trace_header`` — the exact stored
    task + failure_reason live there). A task counts as completed when at
    least one matching trace has a non-infra ``failure_reason`` (success or
    model fault). Infra-only matches are absent from the result, so those
    prompts re-run on resume.

    ``since`` (ISO-8601, normally the checkpoint's ``started_at``) restricts
    the scan to traces recorded at/after that instant — set only when a real
    prior run exists, so an interactive trace from last month with identical
    text does not silently suppress a fresh batch task.

    Returns {task_key: {"trace_id": ..., "failure_reason": ...}}.
    """
    completed: Dict[str, dict] = {}
    buffer_dir = Path(home) / "buffer"
    if not buffer_dir.is_dir():
        return completed
    since_dt = _parse_iso(since) if since else None

    for jsonl in sorted(buffer_dir.rglob("*.jsonl")):
        try:
            with open(jsonl, encoding="utf-8") as f:
                header = json.loads(f.readline())
        except (OSError, json.JSONDecodeError, ValueError):
            continue  # a corrupt trace must never block resume
        if not isinstance(header, dict) or header.get("kind") != "trace_header":
            continue
        if since_dt is not None:
            ts = _parse_iso(header.get("timestamp") or "")
            if ts is not None and ts < since_dt:
                continue
        reason = header.get("failure_reason") or ""
        if is_infra_reason(reason):
            continue  # transport noise: retry on resume
        key = task_key(header.get("task") or "")
        if not key:
            continue
        completed[key] = {
            "trace_id": header.get("trace_id"),
            "failure_reason": header.get("failure_reason"),
        }
    return completed


# --------------------------------------------------------------- checkpoint

def checkpoint_path(home: Path | str, run_name: str) -> Path:
    return Path(home) / "batches" / f"{run_name}.json"


def load_checkpoint(path: Path | str) -> Optional[dict]:
    """Read a batch checkpoint; None when missing or unreadable (stats only —
    a corrupt checkpoint must degrade to a full content scan, never crash)."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def save_checkpoint(path: Path | str, data: dict) -> None:
    """Atomic checkpoint write (tmp + os.replace) via _save_checkpoint.

    Schema v1: {schema_version, run_name, input, started_at, last_updated,
    tasks: [{task, prompt_hash, trace_id, completed_at, failure_reason,
    discarded?}...]}. The ``discarded`` tombstone is schema-reserved for a
    future --discard path; run_batch never writes it in v0.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(data)
    data["last_updated"] = _now_iso()
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


# --------------------------------------------------------------- the loop

def run_batch(tasks: List[str], *, home: Path | str, run_name: str = "default",
              resume: bool = False, run_fn: RunFn, input_path: Optional[str] = None,
              print_fn: Optional[Callable[..., None]] = None,
              model_id: Optional[str] = None) -> int:
    """Execute ``tasks`` sequentially with per-task checkpointing.

    Each task goes through ``run_fn`` exactly once unless skipped by resume;
    the CLI wires it to ``runner.run(task)`` so every trace is an ordinary
    buffer record. The checkpoint is rewritten atomically after EVERY task,
    so a crash mid-run loses at most the in-flight task (which has no
    checkpoint record and no terminal trace — failed prompts are not
    checkpointed, they retry on resume).

    ``model_id`` (pass 4 #29): when the batch explicitly targets a registry
    model, its activation is stamped ONCE at run start (cheap UPDATE; never
    per task or per token). Failures to stamp are ignored — activation
    tracking must never break a batch.

    Returns the process exit code: 0 when every task ends with a terminal
    trace (model failures included), 1 when any infra failure remains.
    """
    out = print_fn if print_fn is not None else (
        lambda *a, **k: print(*a, flush=True, **k))
    home = Path(home)
    cp_path = checkpoint_path(home, run_name)

    if model_id:
        # Idea adapted from PIASystem (MIT) app/core/adapter_manager.py —
        # age-based archival policy should see actual use, not just dates.
        try:
            from ..registry.registry import ModelRegistry

            reg = ModelRegistry(home / "buffer" / "sable.db")
            try:
                reg.record_activation(model_id)
            finally:
                reg.close()
        except Exception as e:  # tracking is best-effort, never fatal
            out(f"batch: activation stamp for {model_id} skipped ({e})")

    checkpoint = load_checkpoint(cp_path) if resume else None
    if checkpoint is not None and checkpoint.get("run_name") != run_name:
        checkpoint = None  # foreign run name -> fresh stats
    started_at = (checkpoint or {}).get("started_at") or _now_iso()
    # Without --resume: fresh intent, no skipping. With --resume: a genuine
    # prior run bounds the scan to its started_at (so an unrelated older
    # trace with identical text cannot suppress a fresh task); with no
    # checkpoint at all we scan the whole history (content is truth).
    since = started_at if checkpoint else None
    completed: Dict[str, dict] = scan_completed(home, since=since) if resume else {}

    records_by_key: Dict[str, dict] = {}
    for rec in (checkpoint or {}).get("tasks", []) or []:
        key = task_key(rec.get("task") or "")
        if key:
            records_by_key[key] = rec

    n_total = len(tasks)
    ran = skipped = retried_infra = failed_model = failed_infra = 0
    t0 = time.monotonic()

    for i, task in enumerate(tasks, 1):
        key = task_key(task)
        preview = " ".join(task.split())[:TASK_PREVIEW_CHARS]
        if key in completed:
            skipped += 1
            out(f"batch {i}/{n_total}: {preview} ... skipped(resume)")
            continue

        prior = records_by_key.get(key)
        if prior and is_infra_reason(prior.get("failure_reason") or ""):
            retried_infra += 1  # re-running an infra fault from a prior run
        ran += 1

        try:
            trace_id, failure_reason = run_fn(task)
        except KeyboardInterrupt:
            out(f"batch {i}/{n_total}: {preview} ... interrupted")
            return 130

        records_by_key[key] = {
            "task": task,
            "prompt_hash": hashlib.sha256(key.encode()).hexdigest(),
            "trace_id": trace_id,
            "completed_at": _now_iso(),
            "failure_reason": failure_reason,
        }
        save_checkpoint(cp_path, {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "run_name": run_name,
            "input": str(input_path) if input_path else None,
            "started_at": started_at,
            "last_updated": None,  # stamped by save_checkpoint
            "tasks": list(records_by_key.values()),
        })

        if failure_reason and is_infra_reason(failure_reason):
            failed_infra += 1
            out(f"batch {i}/{n_total}: {preview} ... failed({failure_reason})")
            # NOT marked completed -> content scan (non-infra only) will offer
            # it again on resume. A later duplicate line retries it, which is
            # the desired behaviour for transport noise.
        else:
            if failure_reason:
                failed_model += 1
            out(f"batch {i}/{n_total}: {preview} ... "
                + (f"failed({failure_reason})" if failure_reason else "ok"))
            completed[key] = {"trace_id": trace_id,
                              "failure_reason": failure_reason}

    elapsed = time.monotonic() - t0
    out(f"batch '{run_name}' finished in {elapsed:.1f}s: ran={ran} "
        f"skipped={skipped} retried-infra={retried_infra} "
        f"failed-model={failed_model} failed-infra={failed_infra} "
        f"(checkpoint: {cp_path})")
    return 1 if failed_infra else 0

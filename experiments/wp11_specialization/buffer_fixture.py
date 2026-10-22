"""Fabricate an Sable experience buffer for WP11 (documented stand-in).

Calibration data for the specialization experiment comes from the SABLE
EXPERIENCE BUFFER ONLY — so the fixture writes traces through the real
``sable.store.TraceRecorder`` (JSONL under ``<home>/buffer/YYYY/MM/`` plus
the SQLite index) into a caller-provided tmp ``SABLE_HOME``, and reads them
back through ``TraceStore.list_traces`` + the JSONL transcripts, exactly like
dashboard/curation code does. Nothing else feeds calibration.

Realism model (documented stand-in for a real user's buffer): each fabricated
trace is one completed agent run whose task line embeds a prefix to extend —
``"nextchar <4-char prefix>"`` — plus a minimal user/assistant message pair,
mirroring how sable traces carry the task text in ``task_summary`` and the
transcript in the JSONL. The domain skew (70% domain A) simulates one user's
behavioral distribution; the split discipline lives in testbed.split_tasks.

This module imports no torch — it is pure buffer plumbing (stdlib + sable
store), so tests can exercise it without the training extra.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from sable.store import TraceRecorder, TraceStore

TASK_PREFIX = "nextchar "


def seed_buffer(home: Path, calib_tasks: List[Dict[str, Any]], *,
                model_alias: str = "wp11-spike-model") -> int:
    """Write one trace per calibration task via the REAL recorder.

    Returns the number of traces recorded (recorder is best-effort and
    returns None on failure — callers must notice a short count).
    """
    rec = TraceRecorder(home=home)
    n = 0
    try:
        for t in calib_tasks:
            prompt = t["prompt"]
            messages = [
                {"role": "user", "content": TASK_PREFIX + prompt},
                # assistant reply is the (buffer-observed) completion; the
                # spike consumes only the task line, but a realistic trace
                # carries the exchange.
                {"role": "assistant", "content": t["expect_contains"][0]},
            ]
            row = rec.record(
                task=TASK_PREFIX + prompt,
                messages=messages,
                backend="spike",
                model_alias=model_alias,
                base_url="http://127.0.0.1:9/v1",  # loopback label only
                completed=True,
            )
            if row is not None:
                n += 1
    finally:
        rec.close()
    return n


def parse_prompt(task_summary: Optional[str]) -> Optional[str]:
    """Extract the 4-char prefix from a buffered task line (None if foreign)."""
    if not task_summary or not task_summary.startswith(TASK_PREFIX):
        return None
    rest = task_summary[len(TASK_PREFIX):].strip()
    return rest.split()[0] if rest else None


def load_calibration_prompts(home: Path) -> List[str]:
    """Read calibration prompts back OUT OF THE BUFFER (index + JSONL).

    Uses the SQLite index for ordering, then re-reads each trace's user turn
    from its JSONL transcript — the same two-layer read curation does. If the
    index and transcripts disagree, the transcript wins (it is the durable
    record); foreign traces (not matching our task prefix) are skipped, which
    also proves the reader tolerates a mixed buffer.
    """
    db = TraceStore(home / "buffer" / "sable.db")
    try:
        rows = db.list_traces()  # newest first; order irrelevant for stats
    finally:
        db.close()
    prompts: List[str] = []
    for row in rows:
        p = parse_prompt(row.task_summary)
        if p is None:
            continue
        # transcript-level confirmation (durable copy wins over index)
        try:
            lines = [json.loads(l) for l in
                     Path(row.jsonl_path).read_text(encoding="utf-8").splitlines()]
        except (OSError, json.JSONDecodeError):
            continue
        user_lines = [l["message"].get("content", "") for l in lines
                      if l.get("kind") == "message"
                      and l.get("message", {}).get("role") == "user"]
        parsed = [parse_prompt(str(c)) for c in user_lines]
        parsed = [p2 for p2 in parsed if p2]
        if parsed:
            prompts.append(parsed[0])
    return sorted(set(prompts))  # dedupe; stats are frequency-based anyway

"""Trace recorder: turn an agent run into a durable buffer record.

Backlog item #1 (secret redaction): every string that lands in the durable
JSONL buffer or the SQLite index passes through core/redact.redact_text /
redact_messages first. The live in-memory conversation returned by the runner
is never mutated — only this durable copy is scrubbed, so credentials stay out
of traces -> datasets -> weights while the agent still sees raw tool output.
Gate: `recording.redact_secrets` (default true); explicitly disabling it logs
one loud warning at recorder init.

One trace = one `runner.run(task)` execution. Output:

  1. Append-only JSONL at <home>/buffer/YYYY/MM/<trace_id>.jsonl containing:
       - line 1: {"kind": "trace_header", ...}    metadata incl. schema_version
       - next N: {"kind": "message", ...}         every message (redacted)
                 (system/user/assistant/tool, tool_calls included)
       - last:   {"kind": "trace_footer", ...}    status/score/tokens/duration
  2. One SQLite index row (store.db.TraceStore).

Fault posture (ROADMAP Phase 2 acceptance): recording is best-effort and must
never break an agent conversation. Every public method here swallows its own
exceptions after logging — a broken disk or schema bug degrades to "no trace",
not "crashed agent".

Capture mechanism: post-run, from the `messages` list returned by the runner
(see core/agent_runner.py). No live hook is registered in the chat loop — the
completed message list is the single canonical trace source.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.failure_types import classify_tool_result, is_cancelled_reason
from ..core.redact import redact_messages, redact_text
from .db import SCHEMA_VERSION, TraceRow, TraceStore
from .paths import resolve_toaster_loop_home

logger = logging.getLogger("toaster_loop.store")

# Message content longer than this is truncated in the *summary* only; the
# JSONL keeps full content.
TASK_SUMMARY_CHARS = 300


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def label_endpoint(base_url: str) -> str:
    """Stable, non-reversible label for an endpoint (hash of host:port path)."""
    h = hashlib.sha256(base_url.encode()).hexdigest()[:12]
    return f"sha256:{h}"


def heuristic_score(messages: List[Dict[str, Any]], completed: bool) -> float:
    """Outcome score v0 — deliberately dumb and transparent.

    Rules (additive, clamped to [-1, 1]):
      +0.5  run completed (agent produced a final answer within limits)
      -0.5  run did NOT complete (max iterations / aborted)
      -0.25 per failed tool result (capped at -0.75), as judged by the shared
            classifier core/failure_types.classify_tool_result (backlog #7 —
            superset of the old startswith("error")/"traceback"/"exception"
            checks: also catches error JSON bodies and nonzero exit codes).
    No tool calls at all is neutral (many tasks don't need them).
    """
    score = 0.5 if completed else -0.5
    err_penalties = 0.0
    for m in messages:
        if m.get("role") != "tool":
            continue
        content = m.get("content")
        if not isinstance(content, str):
            content = json.dumps(content, default=str)
        failed, _kind = classify_tool_result(m.get("name", ""), content)
        if failed:
            err_penalties += 0.25
    score -= min(err_penalties, 0.75)
    return max(-1.0, min(1.0, round(score, 3)))


def count_tool_calls(messages: List[Dict[str, Any]]) -> int:
    n = 0
    for m in messages:
        if m.get("role") == "assistant":
            n += len(m.get("tool_calls") or [])
    return n


class TraceRecorder:
    """Creates trace records for agent runs. Construct one per run() call."""

    def __init__(self, cfg: Any = None, home: Optional[Path] = None,
                 on_record: Optional[Any] = None,
                 project: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 profile: Optional[str] = None,
                 plan_task: Optional[str] = None):
        self.home = Path(home) if home is not None else resolve_toaster_loop_home(cfg)
        # Projects: id stamped onto every trace this recorder writes (None =
        # unfiled). The dashboard builds one recorder per chat run with the
        # active project id from the run context.
        self.project = project or None
        # WP12 plans: task id a run was started from ("Start task" / plan
        # loop / delegate_task plan_task=). Nullable; old traces lack it and
        # readers use tolerant .get() — same additive pattern as `project`.
        self.plan_task = str(plan_task or "").strip() or None
        # Easy training: when set, this name is written into
        # traces.agent_profile INSTEAD of the runner backend name, so
        # curation / readiness checks scope to one agent profile's sessions
        # (query_for_curation(agent_profile=...)). None => legacy behavior
        # (agent_profile = backend).
        self.profile = profile or None
        # Phase 2 (delegation): optional link metadata stamped verbatim into
        # the trace header ("delegation" key): {delegation_id, parent_run_id,
        # parent_trace_id, delegated_from}. Nullable; readers use .get().
        self.metadata = metadata or None
        self.buffer_dir = self.home / "buffer"
        # Backlog #1: secret redaction of the durable copy (default on).
        # `recording.redact_secrets: false` is an escape hatch, not a
        # recommendation — traces feed training data permanently.
        # Flat config field recording_redact_secrets (see core/config.py).
        self.redact_secrets = bool(getattr(cfg, "recording_redact_secrets", True))
        if not self.redact_secrets:
            logger.warning(
                "SECURITY: trace secret redaction is DISABLED "
                "(recording.redact_secrets=false). API keys, tokens and "
                "credentials seen by the agent will be written verbatim to "
                "the buffer and can end up in trained model weights.")
        # Optional callback fired with the TraceRow after a successful record
        # (the dashboard uses it to publish `trace.recorded` on its event bus;
        # keeping it a plain callable keeps this module UI-agnostic).
        self.on_record = on_record
        # Lazily-opened SQLite index reused across records (REPL loops record
        # many traces per process); reset on failure, closed via close().
        self._store: Optional[TraceStore] = None

    def _index(self) -> TraceStore:
        if self._store is None:
            self._store = TraceStore(self.buffer_dir / "toaster_loop.db")
        return self._store

    def close(self) -> None:
        """Close the reused index connection (safe to call more than once)."""
        if self._store is not None:
            try:
                self._store.close()
            except Exception:  # pragma: no cover - defensive
                logger.debug("closing trace index failed (ignored)", exc_info=True)
            self._store = None

    def __del__(self) -> None:  # pragma: no cover - GC timing
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ paths

    def jsonl_path_for(self, trace_id: str, when: datetime) -> Path:
        return self.buffer_dir / f"{when:%Y}" / f"{when:%m}" / f"{trace_id}.jsonl"

    # ------------------------------------------------------------------ write

    def record(
        self,
        *,
        task: str,
        messages: List[Dict[str, Any]],
        backend: str,
        model_alias: str,
        base_url: str,
        completed: bool,
        status: Optional[str] = None,
        tokens_in: Optional[int] = None,
        tokens_out: Optional[int] = None,
        duration_s: Optional[float] = None,
        session_id: Optional[str] = None,
        model_version: Optional[str] = None,
        failure_reason: Optional[str] = None,
        conf_mean: Optional[float] = None,
        conf_p90_negentropy: Optional[float] = None,
    ) -> Optional[TraceRow]:
        """Write JSONL + index row for one completed run. Never raises.

        ``failure_reason`` (backlog #2/#7) is the runner-level failure mode
        ("max_iterations" | "backend_error:<reason>" | "guardrail_halt") or
        None on success; it lands in header+footer and the index row so
        curation can exclude infra faults from DPO negatives.

        ``conf_mean`` / ``conf_p90_negentropy`` (pass 4 #40): the run's two
        confidence aggregates from llama.cpp logprobs (formulas at
        core/agent_runner._confidence_aggregate), or None when no logprob data
        exists. Two floats only — per-token arrays are never stored. They land
        on the index row AND the footer JSON: insights reads traces through
        ``TraceStore.list_traces`` (index rows, same path as #35's timeline
        metadata columns), while the footer keeps the transcript self-
        describing for direct JSONL readers.

        Returns the TraceRow on success, None if recording was skipped/failed.
        """
        try:
            return self._record(
                task=task, messages=messages, backend=backend,
                model_alias=model_alias, base_url=base_url, completed=completed,
                status=status, tokens_in=tokens_in, tokens_out=tokens_out,
                duration_s=duration_s, session_id=session_id,
                model_version=model_version, failure_reason=failure_reason,
                conf_mean=conf_mean, conf_p90_negentropy=conf_p90_negentropy,
            )
        except Exception:  # recording must never break the agent (ARCHITECTURE §6)
            logger.exception("trace recording failed (ignored)")
            return None

    def _record(
        self,
        *,
        task: str,
        messages: List[Dict[str, Any]],
        backend: str,
        model_alias: str,
        base_url: str,
        completed: bool,
        status: Optional[str],
        tokens_in: Optional[int],
        tokens_out: Optional[int],
        duration_s: Optional[float],
        session_id: Optional[str],
        model_version: Optional[str],
        failure_reason: Optional[str] = None,
        conf_mean: Optional[float] = None,
        conf_p90_negentropy: Optional[float] = None,
    ) -> TraceRow:
        trace_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc)
        ts = now.isoformat(timespec="seconds")
        jsonl_path = self.jsonl_path_for(trace_id, now)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)

        # Backlog #1: redact the durable copy only — `messages` (the live
        # in-memory conversation owned by the runner) is never mutated. The
        # footer has no free-text fields (status/score/tokens/coded reason),
        # so header task + messages + index task_summary are the scrub points.
        if self.redact_secrets:
            task = redact_text(task or "")
            messages = redact_messages(messages)

        score: Optional[float] = heuristic_score(messages, completed)
        if status is None:
            # heuristic_score always returns a float; the None branch only
            # exists for the cancelled-run NULLing below.
            status = "success" if (completed and (score or 0) >= 0) else (
                "partial" if completed else "failure"
            )

        # HIGH #4: cancelled runs are score-NEUTRAL. A human pressing Stop says
        # nothing about model quality, so the -0.5 non-completion penalty must
        # never enter aggregate means (should_auto_train gate, stats, insights)
        # or score-band curation. Enforced at WRITE time by NULLing the score:
        # every consumer already treats NULL as "unscored" (AVG skips NULLs;
        # should_auto_train / insights filter None; borderline selection skips
        # None), so no read-side special case is needed. The legacy stamped
        # -0.5 rows are neutralized by the idempotent migration in store/db.py.
        if is_cancelled_reason(failure_reason or ""):
            score = None

        header = {
            "kind": "trace_header",
            "schema_version": SCHEMA_VERSION,
            "trace_id": trace_id,
            "session_id": session_id,
            "timestamp": ts,
            "agent_profile": self.profile or backend,
            "model_alias": model_alias,
            "model_version": model_version,
            "endpoint_label": label_endpoint(base_url) if base_url else None,
            "task": task,
            # nullable; old traces lack it and readers use tolerant .get()
            "failure_reason": failure_reason,
            # v3 (projects): nullable project id; same tolerant-read rule.
            "project": self.project,
            # WP12 plans: nullable plan task id the run was started from.
            "plan_task": self.plan_task,
            # Phase 2 (delegation): nullable linkage block; absent/None on
            # ordinary runs. Readers use .get() — never a schema migration.
            "delegation": self.metadata,
        }

        with open(jsonl_path, "x", encoding="utf-8") as f:
            f.write(json.dumps(header, ensure_ascii=False, default=str) + "\n")
            for m in messages:
                f.write(json.dumps({"kind": "message", "message": m},
                                   ensure_ascii=False, default=str) + "\n")
            footer = {
                "kind": "trace_footer",
                "schema_version": SCHEMA_VERSION,
                "status": status,
                "completed": completed,
                "outcome_score": score,
                "num_tool_calls": count_tool_calls(messages),
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "duration_s": duration_s,
                "failure_reason": failure_reason,
                # Pass 4 #40: confidence aggregates (nullable; None when the
                # logprobs knob was off or the backend lacks support).
                "conf_mean": conf_mean,
                "conf_p90_negentropy": conf_p90_negentropy,
            }
            f.write(json.dumps(footer, ensure_ascii=False, default=str) + "\n")

        row = TraceRow(
            trace_id=trace_id,
            session_id=session_id,
            timestamp=ts,
            agent_profile=self.profile or backend,
            model_alias=model_alias,
            model_version=model_version,
            endpoint_label=label_endpoint(base_url) if base_url else None,
            task_summary=(task or "")[:TASK_SUMMARY_CHARS],
            status=status,
            outcome_score=score,
            num_tool_calls=count_tool_calls(messages),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            duration_s=duration_s,
            jsonl_path=str(jsonl_path),
            schema_version=SCHEMA_VERSION,
            failure_reason=failure_reason,
            project=self.project,
            plan_task=self.plan_task,
            conf_mean=conf_mean,
            conf_p90_negentropy=conf_p90_negentropy,
        )
        try:
            store = self._index()
            store.insert_trace(row)
            # Record the auto-heuristic as a feedback row too, so Phase 3 can
            # distinguish auto-score provenance from explicit user feedback.
            store.add_feedback(trace_id, score, comment="heuristic v0",
                               source="heuristic", timestamp=ts)
        except Exception:
            # A broken connection must not poison later records: drop it so the
            # next _index() reopens (the outer record() still swallows+logs).
            self.close()
            raise
        if self.on_record is not None:
            try:
                self.on_record(row)
            except Exception:  # observers must never break recording
                logger.exception("on_record observer failed (ignored)")
        logger.info("recorded trace %s (%s, score=%s)", trace_id, status, score)
        return row


class NullRecorder:
    """No-op recorder used when recording.enabled is false."""

    def record(self, **kwargs: Any) -> None:  # noqa: D401
        return None


def make_recorder(cfg: Any, on_record: Optional[Any] = None,
                  project: Optional[str] = None,
                  metadata: Optional[Dict[str, Any]] = None,
                  profile: Optional[str] = None,
                  plan_task: Optional[str] = None) -> Any:
    """Recorder honoring config `recording.enabled` (default True)."""
    if not getattr(cfg, "recording_enabled", True):
        return NullRecorder()
    return TraceRecorder(cfg, on_record=on_record, project=project,
                         metadata=metadata, profile=profile,
                         plan_task=plan_task)

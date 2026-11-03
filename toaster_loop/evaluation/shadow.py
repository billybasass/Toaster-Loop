# Adapted from PIASystem (MIT) app/core/evaluation_loop.py
"""Shadow-baseline evaluation before promotion (inspiration pass 4 #32).

A candidate model can win on ``eval_loss`` and regress on the tasks actually
run. Nothing in the promote path catches that — the pass-3 #22 gate only
compares eval losses. This module closes the loop: replay a held-out set of
recent high-feedback traces against BOTH the active (baseline) model and the
candidate, score both sides with the existing LLM judge
(``curate/judge.py``), store the comparison JSON on the registry row next to
``eval_loss``, and flag regressions (warn-only by config).

PIASystem runs candidate and baseline over a benchmark set inside an eval
loop and emits ``ShadowBaselineComparison{improvement, is_regression,
flags}`` with history. Ported shape here: one comparison per job, persisted
on the model row (``models.shadow_json``); "history" stays honest — the jobs
table already keeps every shadow_eval run, we do not duplicate it.

Scoring choice (documented): ABSOLUTE per side, not pairwise. The existing
judge rubric grades one trace against task completion / tool correctness /
grounding and normalizes to 0..1; a pairwise rubric would be a second prompt
to maintain and calibrate. So each side's reply is judged independently:
``improvement = candidate_mean - baseline_mean`` on the judge's 0..1 scale,
and ``is_regression = improvement < -(max_shadow_regression_pct/100 * 1.0)``
— the scale factor is exactly 1.0 because scores are already absolute on a
unit range (5% of the score range default). Total judge calls are bounded at
2×N (one per side per replay task); unparseable/failed judge replies skip
that task rather than fabricating a score, same as curate's judge path.

Bias caveat (loud, inherited from curate/judge.py): each side is judged by
the served model reached through its own endpoint when no external judge
endpoint is configured — self-grading is biased toward its own style. Shadow
verdicts are warn-only tripwires, not ground truth. Judge transport uses
``make_chat_client`` so the pass-4 #28 per-backend circuit breaker applies
automatically (same base_url => same breaker).

Execution shape: IN-PROCESS scheduler job kind (like ``curate``, unlike the
GPU worker kinds). Shadow eval is pure stdlib + httpx — it needs this
process's TraceStore/ModelRegistry handles and serving attach/start; a
subprocess would need to rebuild all of that for no benefit. The scheduler
still claims a device slot and runs the VRAM gate before invoking it, so
"one GPU, one job" holds.

Never mutates serving state beyond what validate already does: if the model
is ALREADY served (serve_state.json matches + healthy) we attach read-only;
otherwise we start a temporary supervisor-owned server and stop it again —
respecting the #28 crash-loop budget, refusing honestly when over budget.
Stdlib only (+ httpx like the rest of the serving/judge paths).
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger("toaster_loop.evaluation.shadow")

#: Judge scores live on the judge's absolute 0..1 scale (parse_judge_reply
#: clamps), so the regression scale below is exactly 1.0 — documented here
#: because the config knob is expressed as a percentage of that range.
SCORE_SCALE = 1.0
DEFAULT_REPLAY_N = 8
DEFAULT_MAX_SHADOW_REGRESSION_PCT = 5.0
#: The honest-empty reason callers log when nothing replayable exists (#32).
NO_REPLAY_SET = "no replay set available"


def no_replay_set_for(profile: Optional[str]) -> str:
    """Transfer-probe failure reason for a profile with no replay set (#37):
    an explicit job error, never a fabricated verdict."""
    return f"no replay set for profile {profile!r}"


class ShadowError(RuntimeError):
    """Expected shadow-eval failure (no baseline, all judge calls skipped…)."""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------- replay-set selection

def build_replay_set(store: Any, profile: Optional[str], n: int = DEFAULT_REPLAY_N,
                     *, min_score: float = 0.5,
                     scan_limit: int = 200) -> List[Dict[str, Any]]:
    """N recent high-feedback traces for ``profile``, as replay tasks.

    The reverse of ``select_borderline`` (which surfaces near-threshold cases
    for the judge to adjudicate): here we want CONFIDENT good outcomes —
    user-positive feedback or effective score >= ``min_score`` — because they
    define "what the model was already good at", and a regression against
    them is what promotion must not hide.

    Selection rules (all deterministic):
      * recent finished traces for the profile (status success/partial),
        newest first;
      * infra faults excluded via ``core.failure_types.is_infra_reason`` —
        they say nothing about model quality (same rule batch resume uses);
      * positive feedback signal: user feedback > 0 OR effective score
        >= min_score (effective = user > judge > heuristic precedence, via
        curater.load_trace — NULL scores are never treated as zero);
      * task input = the FIRST user turn; reference response = the final
        assistant text (context only — never scored against);
      * ordering ts desc then trace_id, capped at ``n``.

    Honest-empty: returns [] when nothing qualifies — callers skip shadow
    eval with reason ``NO_REPLAY_SET`` and NEVER fabricate replay tasks.
    """
    from ..core.failure_types import is_infra_reason
    from ..curate.curater import load_trace

    rows = store.list_traces(agent_profile=profile, limit=max(1, int(scan_limit)))
    # list_traces orders ts DESC already; stable sort makes the tiebreak
    # explicit (ts desc, then trace_id asc) per spec.
    rows = sorted(rows, key=lambda r: (r.timestamp, ), reverse=True)
    ordered: List[Any] = []
    seen_ts: Dict[str, List[Any]] = {}
    for r in rows:
        seen_ts.setdefault(r.timestamp, []).append(r)
    for ts in sorted(seen_ts, reverse=True):
        ordered.extend(sorted(seen_ts[ts], key=lambda r: r.trace_id))

    out: List[Dict[str, Any]] = []
    for row in ordered:
        if len(out) >= max(0, int(n)):
            break
        if row.status not in ("success", "partial"):
            continue  # unfinished/failed runs are not proven-good behavior
        if row.failure_reason and is_infra_reason(row.failure_reason):
            continue  # transport noise, not a task the model handled well
        loaded = load_trace(store, row)
        if loaded is None:
            continue  # transcript unreadable — honest skip, no placeholder
        positive = bool(loaded.has_positive_user_feedback) or (
            loaded.effective_score is not None
            and float(loaded.effective_score) >= float(min_score))
        if not positive:
            continue
        task = ""
        for m in loaded.messages:
            if m.get("role") == "user":
                c = m.get("content")
                task = c if isinstance(c, str) else json.dumps(
                    c, ensure_ascii=False, default=str)
                break
        if not task.strip():
            continue  # no usable prompt -> nothing to replay
        reference = ""
        for m in reversed(loaded.messages):
            if m.get("role") == "assistant":
                c = m.get("content")
                if isinstance(c, str) and c.strip():
                    reference = c
                    break
        out.append({
            "trace_id": row.trace_id,
            "task": task,
            "reference_response": reference,  # context only — never scored
            "ts": row.timestamp,
        })
    return out


# ------------------------------------------------------------- comparison math

def compare_scores(per_task: List[Dict[str, Any]], *,
                   max_regression_pct: float = DEFAULT_MAX_SHADOW_REGRESSION_PCT,
                   ) -> Dict[str, Any]:
    """Aggregate per-task judge scores into the comparison payload.

    Means are PAIRED: only tasks where BOTH sides produced a judgeable score
    enter them (a task whose baseline reply failed to generate must not drag
    the comparison via asymmetric samples). Tasks missing either score count
    in ``skipped``; a mean over zero paired tasks is None, never 0.0.

    Verdict semantics (documented choice: absolute per-side scores on the
    judge's 0..1 scale):
        improvement   = candidate_mean - baseline_mean
        is_regression = improvement < -(max_shadow_regression_pct/100 * SCORE_SCALE)
    with SCORE_SCALE == 1.0 because the score range IS the unit interval.
    """
    paired = [(float(t["candidate_score"]), float(t["baseline_score"]))
              for t in per_task
              if isinstance(t.get("candidate_score"), (int, float))
              and isinstance(t.get("baseline_score"), (int, float))]
    candidate_mean = (sum(c for c, _ in paired) / len(paired)) if paired else None
    baseline_mean = (sum(b for _, b in paired) / len(paired)) if paired else None
    improvement = ((candidate_mean - baseline_mean)
                   if paired and candidate_mean is not None
                   and baseline_mean is not None else None)
    threshold = -(float(max_regression_pct) / 100.0 * SCORE_SCALE)
    is_regression = (improvement is not None and improvement < threshold)
    return {
        "n": len(per_task),
        "candidate_mean": candidate_mean,
        "baseline_mean": baseline_mean,
        "improvement": improvement,
        "regression_threshold": threshold,
        "is_regression": bool(is_regression),
        "skipped": sum(1 for t in per_task
                       if not isinstance(t.get("candidate_score"), (int, float))
                       or not isinstance(t.get("baseline_score"), (int, float))),
        "per_task": per_task,
    }


# --------------------------------------------------------------- serving glue

@contextmanager
def _model_endpoint(model: Any, *, home: Path) -> Iterator[Tuple[str, str]]:
    """Yield (base_url, model_name) for a registry model; never leak servers.

    Attach-first: if serve_state.json already points at this exact model and
    healthchecks, we read-replay against the running server and mutate
    nothing. Otherwise spawn a temporary supervisor-owned llama-server and
    stop it on exit — guarded by the pass-4 #28 crash-loop budget (a model
    that keeps failing to start is refused with an honest error instead of a
    respawn storm).
    """
    from ..core.config import ToasterLoopConfig
    from ..serving.deploy import read_serve_state
    from ..serving.supervisor import (ServingSupervisor, crash_loop_refused, record_start_failure,
                                      clear_crash_loop, healthcheck)

    st = read_serve_state(home) or {}
    if (st.get("model_id") == model.model_id and st.get("base_url")
            and healthcheck(st["base_url"])):
        yield st["base_url"], (st.get("model_name") or model.model_id)
        return

    if not getattr(model, "gguf_path", None):
        raise ShadowError(
            f"model {model.model_id} is not served and has no GGUF artifact — "
            "cannot replay against it")
    crash_key = f"shadow:{model.model_id}"
    if crash_loop_refused(crash_key):
        raise ShadowError(
            f"refusing to start {model.model_id}: crash-loop budget exhausted "
            "(pass 4 #28) — investigate `toaster_loop serve` before shadow eval")
    cfg = ToasterLoopConfig(gguf_path=str(model.gguf_path), port=0,
                        model_name=model.model_id)
    sup = ServingSupervisor(cfg)
    try:
        handle = sup.start()
    except Exception as e:
        record_start_failure(crash_key)
        raise ShadowError(
            f"could not start a temporary server for {model.model_id}: {e}") from e
    clear_crash_loop(crash_key)  # healthy: clean slate (mirrors #28 wrapper)
    try:
        yield handle.base_url, (handle.model_name or model.model_id)
    finally:
        try:
            sup.stop()
        except Exception:  # pragma: no cover - defensive
            logger.warning("shadow eval: stopping temporary server failed",
                           exc_info=True)


# ------------------------------------------------------------------ execution

def run_shadow_eval(
    registry: Any,
    store: Any,
    *,
    candidate_id: str,
    profile: Optional[str] = None,
    n: int = DEFAULT_REPLAY_N,
    max_regression_pct: float = DEFAULT_MAX_SHADOW_REGRESSION_PCT,
    judge_model: Optional[str] = None,
    home: Optional[Path] = None,
    endpoint_for: Optional[Callable[[Any], Any]] = None,
    generate_fn: Optional[Callable[..., Optional[str]]] = None,
    judge_client_for: Optional[Callable[[str, str], Any]] = None,
    actor: str = "worker",
    store_as: Optional[str] = None,
    baseline_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Replay the held-out set against candidate AND baseline; store verdict.

    Returns the comparison dict (also stored via ``registry.set_shadow`` on
    the candidate row), or ``{"skipped": True, "reason": ...}`` when honest
    emptiness applies (no replay set / no baseline). Raises ShadowError for
    expected failures that should land as a job error.

    Transfer-probe mode (pass 4 #37, ``store_as="transfer"``): the same
    machinery against a FOREIGN profile's replay set, with different storage
    and honesty rules — idea adapted from PIASystem (MIT)
    app/core/transfer_bridge.py (shadow simulation before a cross-domain
    transfer is trusted). Differences from promotion shadow eval:
      * baseline = ``baseline_id`` when given, else the registry's CURRENT
        active model ("would this adapter beat what that profile runs today?")
        — never the candidate itself;
      * no active baseline -> CANDIDATE-ONLY run: absolute scores stored with
        ``baseline_id: None`` and ``is_regression: None`` (no fake comparison);
      * empty replay set -> ShadowError 'no replay set for profile P' (a probe
        asked a question; skipping silently would hide the answer "nothing to
        test against");
      * verdict lands on ``registry.record_transfer`` (per-profile history in
        models.transfers_json), NOT shadow_json — that column stays the
        promotion verdict for the model's own profile.

    Injectable seams (all default to production paths) keep this testable
    without GPU: ``endpoint_for(model_row)`` -> contextmanager yielding
    (base_url, model_name); ``generate_fn(base_url, model_name, task)``;
    ``judge_client_for(base_url, model_name)`` -> judge client_fn.
    """
    from ..curate.judge import judge_trace, make_chat_client
    from ..training.validate import _post_task

    transfer = store_as == "transfer"
    home_root: Path = Path(home) if home is not None else Path(registry.home)
    cand = registry.get_model(candidate_id)
    if cand is None:
        raise ShadowError(f"no model {candidate_id} in registry")

    replay = build_replay_set(store, profile, n=n)
    if not replay:
        if transfer:
            # Honest-empty as a JOB FAILURE (#37): nothing is stored.
            raise ShadowError(no_replay_set_for(profile))
        logger.info("shadow eval %s: %s — skipping", candidate_id, NO_REPLAY_SET)
        return {"skipped": True, "reason": NO_REPLAY_SET}

    # Baseline resolution. Transfer probes take an explicit baseline or fall
    # back to the current active model; promotion shadow eval always uses the
    # active model for this base model AT RUN TIME (claim time), never the
    # candidate itself. No baseline -> transfer runs candidate-only,
    # promotion skips honestly instead of assuming a win.
    if transfer:
        if baseline_id:
            baseline = registry.get_model(baseline_id)
            if baseline is None:
                raise ShadowError(f"no model {baseline_id} in registry "
                                  "(explicit --baseline)")
            if baseline.model_id == cand.model_id:
                raise ShadowError("baseline must differ from the candidate")
        else:
            baseline = registry.active_model()
            if baseline is not None and baseline.model_id == cand.model_id:
                baseline = None  # comparing a model to itself proves nothing
    else:
        baseline = registry.active_model(base_model=cand.base_model)
        if baseline is None or baseline.model_id == cand.model_id:
            logger.info("shadow eval %s: no active baseline for base model %r "
                        "— skipping", candidate_id, cand.base_model)
            return {"skipped": True,
                    "reason": f"no active baseline for {cand.base_model!r}"}

    _endpoint_for = endpoint_for or (lambda m: _model_endpoint(m,
                                                              home=home_root))

    def _generate(base_url: str, model_name: str, task: str) -> Optional[str]:
        message, err = _post_task(
            base_url, model_name, {"prompt": task},
            timeout=120.0, max_tokens=256, temperature=0.0, headers={})
        if err is not None:
            logger.warning("shadow replay request failed (%s): %s", model_name, err)
            return None
        return (message or {}).get("content")

    generate = generate_fn or _generate

    def _client_for(base_url: str, model_name: str):
        return make_chat_client(base_url)

    judge_client = judge_client_for or _client_for

    def _side_scores(model: Any) -> Dict[str, Optional[float]]:
        """Run + judge every replay task once against one model."""
        scores: Dict[str, Optional[float]] = {}
        with _endpoint_for(model) as (base_url, model_name):
            client = judge_client(base_url, judge_model or model_name)
            for item in replay:
                reply = generate(base_url, model_name, item["task"])
                if reply is None:
                    scores[item["trace_id"]] = None
                    continue
                verdict = judge_trace(
                    [{"role": "user", "content": item["task"]},
                     {"role": "assistant", "content": reply}],
                    client, judge_model or model_name)
                scores[item["trace_id"]] = (
                    verdict["score"] if verdict else None)
        return scores

    cand_scores = _side_scores(cand)
    base_scores = _side_scores(baseline) if baseline is not None else {}

    per_task: List[Dict[str, Any]] = []
    for item in replay:
        tid = item["trace_id"]
        per_task.append({
            "trace_id": tid,
            "task_preview": item["task"][:120],
            "candidate_score": cand_scores.get(tid),
            "baseline_score": base_scores.get(tid),
        })

    comparison = compare_scores(per_task,
                                max_regression_pct=max_regression_pct)
    if baseline is None:
        # Candidate-only transfer probe (#37): no comparison exists, so the
        # absolute candidate mean is stored and regression stays honest-None.
        scored_c = [float(t["candidate_score"]) for t in per_task
                    if isinstance(t.get("candidate_score"), (int, float))]
        if not scored_c:
            raise ShadowError(
                "transfer probe produced zero scored tasks (all generation or "
                "judge calls failed/skipped) — refusing to record a verdict")
        comparison = {
            "n": len(per_task),
            "candidate_mean": sum(scored_c) / len(scored_c),
            "baseline_mean": None,
            "improvement": None,
            "regression_threshold": None,
            "is_regression": None,  # documented: no baseline -> no comparison
            "skipped": len(per_task) - len(scored_c),
            "per_task": per_task,
        }
    else:
        scored = comparison["n"] - comparison["skipped"]
        if scored == 0:
            # Every task skipped (transport/judge failures): an empty verdict
            # must never masquerade as "no regression found".
            raise ShadowError(
                "shadow eval produced zero scored tasks (all generation or "
                "judge calls failed/skipped) — refusing to record a verdict")

    comparison = {
        "ts": _utcnow(),
        "baseline_id": baseline.model_id if baseline is not None else None,
        "profile": profile,
        "judge_model": judge_model,
        **comparison,
    }
    if transfer:
        registry.record_transfer(candidate_id, comparison, actor=actor)
        logger.info("transfer probe %s on profile %r vs %s: improvement=%s",
                    candidate_id, profile,
                    comparison["baseline_id"] or "(candidate-only)",
                    comparison["improvement"])
        return comparison
    registry.set_shadow(candidate_id, comparison, actor=actor)
    if comparison["is_regression"]:
        logger.warning(
            "shadow eval %s vs %s: REGRESSION improvement=%s (threshold %s)",
            candidate_id, comparison["baseline_id"],
            f"{comparison['improvement']:.3f}",
            f"{comparison['regression_threshold']:.3f}")
    else:
        logger.info("shadow eval %s vs %s: improvement=%s",
                    candidate_id, comparison["baseline_id"],
                    comparison["improvement"])
    return comparison


# ------------------------------------------------------------- promotion gate

def shadow_promotion_check(shadow: Optional[Dict[str, Any]], *,
                           blocks: bool = False) -> Dict[str, Any]:
    """Consult a stored shadow verdict before promoting (pipeline #32).

    Warn-only by default: a regression logs + returns a message for the
    report/ledger-visible note; ``blocks=True``
    (validation.shadow_blocks_promotion) turns it into a promotion veto.
    Absent or skipped verdicts never block anything — honest absence is not
    evidence of quality either way.
    """
    res: Dict[str, Any] = {"warned": False, "blocked": False, "message": None}
    if not shadow or shadow.get("skipped"):
        return res
    if not shadow.get("is_regression"):
        return res
    imp = shadow.get("improvement")
    thr = shadow.get("regression_threshold")
    res["message"] = (
        "shadow eval regressed vs baseline "
        f"{shadow.get('baseline_id')}: improvement "
        + (f"{imp:.3f}" if isinstance(imp, (int, float)) else str(imp))
        + " < threshold "
        + (f"{thr:.3f}" if isinstance(thr, (int, float)) else str(thr))
        + f" on {shadow.get('n')} replay task(s) "
          "(validation.max_shadow_regression_pct)")
    if blocks:
        res["blocked"] = True
    else:
        res["warned"] = True
    return res


# ------------------------------------------------------- scheduler job runner

def make_shadow_job_runner(registry: Any, store: Any, *, home: Path,
                           defaults: Optional[Dict[str, Any]] = None,
                           endpoint_for: Optional[Callable[[Any], Any]] = None,
                           ) -> Callable[[Any], Dict[str, Any]]:
    """Closure the scheduler invokes in-process for ``shadow_eval`` jobs.

    Reads per-job inputs from ``job.params`` (candidate model id, profile, n,
    judge model) overlaid on config defaults; finishes by returning the
    comparison dict (the scheduler turns exceptions into job errors).

    Pass 4 #37 reuses this job kind for cross-profile transfer probes: params
    ``store_as: "transfer"`` (+ optional ``baseline_id``) switch storage to
    ``record_transfer`` and honesty rules to probe semantics — no new job kind.
    """
    d = dict(defaults or {})

    def runner(job: Any) -> Dict[str, Any]:
        p = job.params if isinstance(job.params, dict) else {}
        candidate_id = p.get("model_id") or job.model_id
        if not candidate_id:
            raise ShadowError("shadow_eval job has no candidate model id")
        result = run_shadow_eval(
            registry, store,
            candidate_id=str(candidate_id),
            profile=p.get("profile") or None,
            n=int(p.get("n") or d.get("replay_n") or DEFAULT_REPLAY_N),
            max_regression_pct=float(
                p["max_regression_pct"]
                if p.get("max_regression_pct") is not None
                else d.get("max_shadow_regression_pct",
                           DEFAULT_MAX_SHADOW_REGRESSION_PCT)),
            judge_model=p.get("judge_model") or d.get("judge_model") or None,
            home=home,
            endpoint_for=endpoint_for,
            store_as=p.get("store_as") or None,
            baseline_id=p.get("baseline_id") or None,
        )
        if result.get("skipped"):
            logger.info("shadow_eval job %s skipped: %s", job.job_id,
                        result.get("reason"))
        return result

    return runner

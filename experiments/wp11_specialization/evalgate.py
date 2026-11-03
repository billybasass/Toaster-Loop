"""Downstream golden-task eval for WP11 — PRODUCTION gate semantics.

Mapping to training/validate.py (documented per WP11 spec):

  validate.py                       this spike
  -------------------------------   ----------------------------------------
  served GGUF, chat completion      greedy_completion() of the tiny model
                                    (argmax = temperature-0 decode; no HTTP)
  task {prompt, expect_contains}    identical dict shape (testbed.make_task)
  _check_task(message, task)        THE SAME FUNCTION, imported from
                                    toaster_loop.training.validate — the substring
                                    oracle is not re-implemented (D-12 spirit:
                                    reuse own code, no forked semantics)
  run_golden_suite -> {total,       run_eval() -> same verdict shape plus
    passed, pass_rate, results}       per-domain breakdown for reporting
  min_pass_rate gate (ok =          gate_passed uses the identical rule:
    pass_rate >= min_pass_rate)       pass_rate >= min_pass_rate

Verdict rule (cardinal rule 2): ONLY this eval judges specialization. Weight
MSE, router-shift, and any other proxy are deliberately not computed as
adoption evidence anywhere in the experiment.
"""

from __future__ import annotations

from typing import Any, Dict, List

from toaster_loop.training.validate import _check_task  # production oracle, D-12-safe

from .testbed import greedy_completion


def run_eval(params: Dict[str, Any], eval_tasks: List[Dict[str, Any]], *,
             keep: Any = None, act_thresholds: Any = None,
             min_pass_rate: float = 1.0) -> Dict[str, Any]:
    """Replay the golden suite through the production check semantics.

    Each task: model reply = greedy completion; scored by validate.py's own
    ``_check_task`` (a "serving failure" is impossible offline — a crash in
    scoring would be a bug, not a model fault, so it propagates).
    """
    results: List[Dict[str, Any]] = []
    for t in eval_tasks:
        reply = greedy_completion(params, t, keep=keep,
                                  act_thresholds=act_thresholds)
        # message dict shaped like an OpenAI chat-completion message so the
        # production oracle sees exactly what it would see over HTTP.
        entry = _check_task({"content": reply}, t)
        entry["domain"] = t["domain"]
        results.append(entry)
    total = len(results)
    passed = sum(1 for r in results if r.get("pass"))
    rate = (passed / total) if total else 0.0
    per_domain: Dict[str, Dict[str, Any]] = {}
    for d in sorted({r["domain"] for r in results}):
        sub = [r for r in results if r["domain"] == d]
        p = sum(1 for r in sub if r.get("pass"))
        per_domain[d] = {"total": len(sub), "passed": p,
                         "pass_rate": p / len(sub) if sub else 0.0}
    return {
        "total": total,
        "passed": passed,
        "pass_rate": rate,
        "min_pass_rate": min_pass_rate,
        "gate_passed": rate >= min_pass_rate,  # validate.py's ok= rule
        "per_domain": per_domain,
        "results": results,
    }


def variant_eval(params: Dict[str, Any], eval_tasks: List[Dict[str, Any]], *,
                 keep: Any = None, act_thresholds: Any = None) -> float:
    """Convenience: pass-rate only (the metric the noise floor measures)."""
    return run_eval(params, eval_tasks, keep=keep,
                    act_thresholds=act_thresholds)["pass_rate"]

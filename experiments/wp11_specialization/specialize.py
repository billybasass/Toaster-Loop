"""Specialization methods for WP11 — calibration-set statistics ONLY.

Two variants, both computed exclusively from the experience-buffer
calibration prompts (never eval, never labels):

  A) ``expert_prune``     — mean router load per expert over the calibration
     set; drop the least-loaded expert(s); router renormalized over
     survivors at inference (sparse_transformers-style expert-pruning
     methodology, re-implemented inline, D-12).
  B) ``activation_sparse``— per-unit activation quantile thresholds over the
     calibration set; zero sub-threshold expert-hidden units at inference
     (calibrated top-k-style activation sparsity, same citation discipline).

The "per-user" story: the buffer is skewed toward one user's domain mix, so
the hypothesis under test is that statistics from THAT buffer specialize the
mixture toward the user's distribution — measurable only downstream.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .testbed import N_EXPERTS, encode


def _torch():
    import torch

    return torch


# ----------------------------------------------------------------- statistics

def calibration_stats(params: Dict[str, Any],
                      calib_prompts_as_tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Router load + per-unit activation quantiles over calibration prompts.

    Returns {"router_mean": [E], "act_q25": [[H]x E], "n": int}. Pure forward
    statistics — no labels touched, no gradient, deterministic.
    """
    torch = _torch()
    if not calib_prompts_as_tasks:
        raise ValueError("calibration set is empty — cannot specialize")
    X, _ = encode(calib_prompts_as_tasks)
    router_acc = torch.zeros(N_EXPERTS)
    act_min = [None] * N_EXPERTS
    with torch.no_grad():
        r = torch.softmax(X @ params["Wr"].T, dim=-1)  # [B,E]
        router_acc += r.mean(dim=0)
        for e in range(N_EXPERTS):
            z = torch.relu(X @ params["W1"][e].T + params["b1"][e])  # [B,H]
            act_min[e] = z
    n = X.shape[0]
    # per-unit quantiles across the calibration batch (q=0.25 chosen as a
    # moderate sparsity level; fixed, not tuned on eval)
    q25 = [torch.quantile(z.clamp(min=1e-12), 0.25, dim=0).detach()
           for z in act_min]
    return {
        "router_mean": [float(v) for v in router_acc],
        "act_q25": [q.detach() for q in q25],
        "n": int(n),
    }


# ------------------------------------------------------------------ variants

def expert_prune(params: Dict[str, Any], calib_tasks: List[Dict[str, Any]],
                 *, prune_count: int = 1,
                 stats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Variant A: least-loaded-on-calibration experts get pruned.

    Returns {"keep": bool tensor [E], "pruned": [expert ids]}. Renormalized
    routing is applied by testbed.forward(keep=...).
    """
    torch = _torch()
    st = stats or calibration_stats(params, calib_tasks)
    order = sorted(range(N_EXPERTS), key=lambda e: st["router_mean"][e])
    pruned = order[:max(0, prune_count)]
    keep = torch.ones(N_EXPERTS, dtype=torch.bool)
    for e in pruned:
        keep[e] = False
    return {"keep": keep, "pruned": pruned}


def activation_sparse(params: Dict[str, Any],
                      calib_tasks: List[Dict[str, Any]], *,
                      stats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Variant B: calibrated per-unit thresholds from the calibration set.

    Returns {"act_thresholds": float tensor [E,H]}; sub-threshold expert
    hidden units are zeroed at inference by testbed.forward(...).
    """
    torch = _torch()
    st = stats or calibration_stats(params, calib_tasks)
    return {"act_thresholds": torch.stack(st["act_q25"])}

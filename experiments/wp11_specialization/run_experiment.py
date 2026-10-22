"""WP11 experiment driver — R-replicate baseline vs specialization variants.

Protocol (docs/SPECIALIZATION-WP11.md):

  1. Fabricate the experience buffer through the REAL TraceRecorder into a
     throwaway SABLE_HOME under a tmpdir (NEVER the developer's ~/.sable),
     read calibration prompts back out of the buffer, map them to tasks.
  2. Two optimization regimes are measured (both real, reported separately):
       full    — train to convergence (production-realistic; ceiling risk)
       limited — few steps, baseline mid-band (where specialization could
                 plausibly help and the noise floor is non-degenerate)
     For replicate r = 0..R-1 in each regime: train one fresh tiny MoE
     (identical recipe, init seed r); then evaluate — PAIRED on the same
     trained model — baseline (no specialization), variant A (expert prune,
     keep-3-of-4 + renormalized router), variant B (calibrated activation
     sparsity). All stats from calibration prompts only. Eval is the
     downstream golden suite through validate.py's own _check_task.
  3. Noise floor per regime: spread of the R baseline pass rates (fixed
     config, seed variation) via the WP1 compute_stats definition. Verdict
     per variant: paired relative delta vs max(3%, floor).

Output: JSON report to stdout (and --out file), findings-doc table on stderr.
Run: python -m experiments.wp11_specialization.run_experiment --replicates 13
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

from .buffer_fixture import load_calibration_prompts, seed_buffer
from .evalgate import run_eval
from .noise_floor import (estimate_eval_floor, persisted_floor,
                          summarize_variant)
from .specialize import activation_sparse, calibration_stats, expert_prune
from .testbed import make_params, split_tasks, train_model

DEFAULT_REPLICATES = 13

#: Optimization regimes, both measured (docs/SPECIALIZATION-WP11.md):
#:   full    — production-realistic: train to convergence. If the baseline
#:             saturates near 100% there is a CEILING EFFECT: wins become
#:             impossible, only regressions measurable. Reported for honesty.
#:   limited — deliberately optimization-limited (few steps) so the baseline
#:             sits in a mid band with real replicate-to-replicate variance:
#:             the regime where specialization COULD plausibly help and
#:             where the measured noise floor is non-degenerate.
REGIMES = {"full": 600, "limited": 8}


def _tasks_from_prompts(prompts: List[str],
                        train_tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Map buffered prompts back to task dicts (labels come from the rules,
    but ONLY as calibration inputs — labels never enter the eval path)."""
    by_prompt = {t["prompt"]: t for t in train_tasks}
    return [by_prompt[p] for p in prompts if p in by_prompt]


def run_replicate(rep: int, splits: Dict[str, List[Dict[str, Any]]],
                  calib_tasks: List[Dict[str, Any]], *,
                  steps: int) -> Dict[str, Any]:
    """Train one model and evaluate baseline + both variants on it (paired)."""
    params = make_params(seed=1000 + rep)
    train_model(params, splits["train"], steps=steps, lr=0.05)

    st = calibration_stats(params, calib_tasks)
    a = expert_prune(params, calib_tasks, prune_count=1, stats=st)
    b = activation_sparse(params, calib_tasks, stats=st)

    base = run_eval(params, splits["eval"])
    va = run_eval(params, splits["eval"], keep=a["keep"])
    vb = run_eval(params, splits["eval"], act_thresholds=b["act_thresholds"])
    return {
        "rep": rep,
        "final_loss": params.get("_final_loss"),
        "router_mean": st["router_mean"],
        "pruned_experts": a["pruned"],
        "baseline": base,
        "expert_prune": va,
        "activation_sparse": vb,
    }


def run_regime(name: str, steps: int, replicates: int,
               splits: Dict[str, List[Dict[str, Any]]],
               calib_tasks: List[Dict[str, Any]],
               persisted: Any, verbose: bool) -> Dict[str, Any]:
    reps: List[Dict[str, Any]] = []
    t0 = time.time()
    for r in range(replicates):
        reps.append(run_replicate(r, splits, calib_tasks, steps=steps))
        if verbose:
            b = reps[-1]["baseline"]["pass_rate"]
            print(f"[wp11:{name}] rep {r}: baseline eval pass-rate {b:.3f}",
                  file=sys.stderr)
    elapsed = time.time() - t0

    base_rates = [x["baseline"]["pass_rate"] for x in reps]
    floor = estimate_eval_floor(base_rates)
    variants = {}
    for vname in ("expert_prune", "activation_sparse"):
        rates = [x[vname]["pass_rate"] for x in reps]
        variants[vname] = summarize_variant(vname, base_rates, rates, floor)

    return {
        "regime": name,
        "train_steps": steps,
        "protocol": {
            "replicates": replicates,
            "paired": True,
            "eval": "downstream golden suite via "
                    "sable.training.validate._check_task (substring)",
            "calibration_source": "sable experience buffer (TraceRecorder "
                                  "-> JSONL+index -> read back)",
            "noise_floor_method": floor["method"],
            "persisted_bench_noise_floor": persisted,
        },
        "noise_floor": floor,
        "replicates_detail": reps,
        "variants": variants,
        "elapsed_s": round(elapsed, 2),
    }


def run_experiment(replicates: int = DEFAULT_REPLICATES,
                   regimes: Dict[str, int] | None = None,
                   verbose: bool = True) -> Dict[str, Any]:
    tmp_home = Path(tempfile.mkdtemp(prefix="wp11-sable-home-"))
    os.environ["SABLE_HOME"] = str(tmp_home)  # belt & braces: buffer never
    os.environ.pop("SABLE_BENCH_DIR", None)   # lands in the real ~ dir

    try:
        splits = split_tasks()
        n_seeded = seed_buffer(tmp_home, splits["calib"])
        prompts = load_calibration_prompts(tmp_home)
        calib_tasks = _tasks_from_prompts(prompts, splits["train"])
        if len(calib_tasks) < 8:
            raise RuntimeError(
                f"buffer round-trip too small: seeded={n_seeded} "
                f"recovered={len(prompts)} usable={len(calib_tasks)}")

        persisted = persisted_floor()  # reference only; offline floor is used

        regimes = regimes or REGIMES
        regime_reports = {
            name: run_regime(name, steps, replicates, splits, calib_tasks,
                             persisted, verbose)
            for name, steps in regimes.items()
        }

        return {
            "work_package": "WP11",
            "splits": {k: len(v) for k, v in splits.items()},
            "buffer": {"home": str(tmp_home), "seeded": n_seeded,
                       "recovered_prompts": len(prompts),
                       "calib_tasks_used": len(calib_tasks)},
            "regimes": regime_reports,
        }
    finally:
        shutil.rmtree(tmp_home, ignore_errors=True)


def render_table(report: Dict[str, Any]) -> str:
    lines = ["", f"WP11 results — R={report['splits']['eval']} eval goldens"]
    for name, rr in report["regimes"].items():
        floor = rr["noise_floor"]
        sp = floor["spread_pct"]
        lines += [
            "",
            f"regime={name} (train steps={rr['train_steps']}, "
            f"{rr['elapsed_s']}s) R={floor['R']}",
            f"  baseline pass-rates: median={floor['median_pass_rate']:.3f} "
            f"spread={'n/a' if sp is None else f'{sp:.1f}%'} of median "
            f"({floor['spread_pp']:.1f} pp abs)",
            f"  {'variant':<20} {'median pass':>11} {'Δ vs base':>10} "
            f"{'bar':>7} {'verdict':>12}",
        ]
        for vname, v in rr["variants"].items():
            lines.append(
                f"  {vname:<20} {v['variant_median_pass_rate']:>11.3f} "
                f"{v['relative_delta_pct']:>+9.1f}% "
                f"{v['adoption_bar_pct']:>6.1f}% {v['verdict']:>12}")
    return "\n".join(lines)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replicates", "-R", type=int, default=DEFAULT_REPLICATES)
    ap.add_argument("--out", type=str, default=None,
                    help="write the JSON report to this path")
    args = ap.parse_args(argv)
    report = run_experiment(replicates=args.replicates)
    text = json.dumps(report, indent=2, sort_keys=True, default=str)
    print(text)
    print(render_table(report), file=sys.stderr)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

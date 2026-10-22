"""Noise-floor protocol for WP11 — WP1 methodology on the EVAL metric.

Two sources, tried in order (documented in the findings doc):

1. ``serving.bench.load_noise_floor()`` — a persisted ``kind=noise_floor``
   record from ``sable bench --noise-floor``. If present it is recorded for
   reference, BUT it measures tokens/sec spread of the llama.cpp server, not
   this offline eval; it cannot bound an eval pass-rate delta, so…
2. …the experiment ALWAYS also estimates the floor offline on the eval metric
   itself, per the WP11 spec: run the eval R >= 11 times at fixed config with
   seed variation (each replicate = a fresh model-init training run under the
   identical recipe), take the baseline pass-rate distribution, and reuse the
   WP1 spread definition — ``sable.serving.bench.compute_stats``'s
   ``spread_pct = (max - min) / median * 100`` — verbatim. That spread is the
   noise floor for adoption decisions in THIS experiment.

Adoption language rule (cardinal rule 2): a variant may be called a WIN only
when its mean delta vs baseline clears ``max(3.0, measured_floor_pct)``
percentage points of pass rate; anything inside is NOISE / inconclusive.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from sable.serving.bench import compute_stats, load_noise_floor

#: The fixed 3% floor from the WP11 spec; the measured floor overrides when
#: larger (adopts strictly: never a smaller bar than the program minimum).
MIN_BAR_PCT = 3.0


def _f(v: Optional[float], ctx: str) -> float:
    """compute_stats returns Optional floats; our inputs are always non-empty
    so they are present — assert once, keep mypy/pyright quiet."""
    if v is None:
        raise ValueError(f"compute_stats produced no value for {ctx}")
    return float(v)


def persisted_floor() -> Optional[Dict[str, Any]]:
    """The persisted bench noise-floor record if one exists (reference only).

    Returns the full record; callers extract generation/prompt spread_pct for
    the findings doc. None when ``sable bench --noise-floor`` was never run
    (this experiment's box: none — documented in docs/SPECIALIZATION-WP11.md).
    """
    return load_noise_floor()


def estimate_eval_floor(baseline_pass_rates: Sequence[float]) -> Dict[str, Any]:
    """WP1-style floor from R baseline eval replicates at fixed config.

    Reuses bench.compute_stats so the spread definition is literally WP1's
    (median/min/max + (max-min)/median*100). Guards: R >= 11 per spec; a
    degenerate all-equal distribution gives spread_pct = 0, in which case the
    MIN_BAR_PCT still applies (3% bar never weakens to zero).
    """
    vals = [float(v) for v in baseline_pass_rates]
    if len(vals) < 11:
        raise ValueError(
            f"noise-floor estimation needs R >= 11 replicates, got {len(vals)}")
    st = compute_stats(vals)
    return {
        "method": "offline eval-metric replicates (WP1 spread definition via "
                  "serving.bench.compute_stats)",
        "R": len(vals),
        "pass_rates": vals,
        "median_pass_rate": _f(st["median"], "median"),
        # WP1 spread: (max-min)/median*100 — relative units, same convention
        # as bench.compare_records' fractional speedup vs spread_pct.
        "spread_pct": st["spread_pct"],
        # absolute percentage-point spread of the pass rate, for reporting.
        "spread_pp": (_f(st["max"], "max") - _f(st["min"], "min")) * 100.0,
        "floor_pct": st["spread_pct"],
    }


def adoption_bar_pct(measured_floor_pct: Optional[float]) -> float:
    """The bar a delta must clear: max(3%, measured noise floor)."""
    f = measured_floor_pct if isinstance(measured_floor_pct, (int, float)) else 0.0
    return max(MIN_BAR_PCT, float(f))


def verdict(delta_pp: float, bar_pct: float) -> str:
    """WIN only above the bar; inside => noise. delta_pp in percentage points.

    Note on units: pass-rate deltas are converted to percentage-of-baseline
    (relative), matching bench.compare_records' fractional speedup convention,
    so the same max(3, floor_pct) bar applies consistently. A negative delta
    beyond the bar is a REGRESSION (still "clears" the floor as an effect).
    """
    if abs(delta_pp) < bar_pct:
        return "noise"
    return "win" if delta_pp > 0 else "regression"


def summarize_variant(name: str, baseline_rates: Sequence[float],
                      variant_rates: Sequence[float],
                      floor: Dict[str, Any]) -> Dict[str, Any]:
    """Paired per-replicate comparison of one variant vs the baseline.

    Paired (same replicate = same trained model) so training-init variance —
    the dominant noise source here — cancels in the delta; the floor then
    judges whether any residual mean effect clears it. Relative-percent
    convention: delta = (variant - baseline) / baseline * 100, matching
    bench.compare_records' fractional speedup. Baseline-zero edge case: with a
    zero baseline pass rate relative deltas are undefined; fall back to
    absolute percentage points and mark the comparison ``relative=False``.
    """
    if len(baseline_rates) != len(variant_rates) or not baseline_rates:
        raise ValueError("baseline/variant replicate lists must match, non-empty")
    base_med = _f(compute_stats(baseline_rates)["median"], "base median")
    var_med = _f(compute_stats(variant_rates)["median"], "variant median")
    rel: List[float] = []
    for b, v in zip(baseline_rates, variant_rates):
        rel.append(((v - b) / b * 100.0) if b > 0 else (v - b) * 100.0)
    rel_stats = compute_stats(rel)
    bar = adoption_bar_pct(floor.get("floor_pct"))
    return {
        "variant": name,
        "baseline_median_pass_rate": base_med,
        "variant_median_pass_rate": var_med,
        "median_delta_pp_points": var_med - base_med,
        "relative_delta_pct": _f(rel_stats["median"], "relative delta"),
        "relative_delta_spread_pct": rel_stats["spread_pct"],
        "noise_floor_pct": floor.get("floor_pct"),
        "adoption_bar_pct": bar,
        "verdict": verdict(_f(rel_stats["median"], "relative delta"), bar),
    }

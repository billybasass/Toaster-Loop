"""Evaluation subpackage (inspiration pass 4 #32).

Shadow-baseline evaluation before promotion: replay a held-out set of recent
high-feedback traces against BOTH the active (baseline) model and a candidate,
score both with the existing LLM judge, and store the comparison on the
registry row next to ``eval_loss``. Runs as a normal scheduler job kind
(``shadow_eval``) so VRAM gating and device slots apply.
"""

from .shadow import (  # noqa: F401
    DEFAULT_MAX_SHADOW_REGRESSION_PCT,
    DEFAULT_REPLAY_N,
    NO_REPLAY_SET,
    ShadowError,
    build_replay_set,
    compare_scores,
    run_shadow_eval,
    shadow_promotion_check,
)

__all__ = [
    "DEFAULT_MAX_SHADOW_REGRESSION_PCT",
    "DEFAULT_REPLAY_N",
    "NO_REPLAY_SET",
    "ShadowError",
    "build_replay_set",
    "compare_scores",
    "run_shadow_eval",
    "shadow_promotion_check",
]

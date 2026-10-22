# Idea adapted from PIASystem (MIT) app/core/capability_tiers.py — graduated
# autonomy tiers with evidenced requirements. Their request/approval ceremony
# is deliberately skipped (pass-4 #41): tiers here are DERIVED, never hand-set,
# so there is nothing to approve. T4 "audited" stays reserved and unimplemented
# — we do not fake evidence for a tier nobody audits.
"""Derived capability tiers for registry models (inspiration pass 4 #41).

Pure derivation over evidence the registry/buffer already carry — no stored
tier column, no DB writes here (the one read helper below only reads traces).
Deriving on read keeps the tier honest and migration-free: it can never drift
from the evidence it is computed from.

Ladder (cumulative — a tier implies every lower tier's evidence):

    T0 experimental     fresh train; nothing below satisfied.
    T1 validated        validation report present AND passing. "Passing" is
                        read honestly from what ``set_validation`` stores: the
                        ``validate_model`` report dict, whose contract key is
                        ``ok: true`` (train/export failures store error-only
                        payloads without an ``ok`` key — those never count).
                        Models with no validation at all stay T0.
    T2 shadow-proven    latest shadow verdict (#32) present, not skipped, and
                        explicitly ``is_regression is False``. Absent or
                        skipped verdicts are honest absence — not evidence.
    T3 field-proven     at least ``MIN_SERVED_OK`` (default 10) served traces
                        since promotion with positive-or-neutral outcome AND
                        zero negative-feedback traces. "Served trace" uses the
                        same matching + score definition insights uses:
                        ``model_alias``/``model_version`` == model_id, quality
                        = ``traces.outcome_score`` (user feedback synced;
                        judge excluded). Unscored traces count toward neither
                        side. Floor timestamp is ``promoted_at`` (rows never
                        promoted fall back to ``created_at`` — same trace
                        window insights would look at). No recency guard is
                        applied, so derivation needs no clock and stays
                        deterministic; ``last_used_at`` is deliberately NOT
                        consulted.
    T4 audited          RESERVED — not implemented this pass. Cap is T3.

Policy hooks (exactly two, per backlog #41):
  1. automatic promotion (pipeline stage 6, reached by ``iterate`` and the
     ``train auto-check --trigger`` cron path) requires tier >=
     ``validation.min_tier_auto_deploy`` (0 disables);
  2. manual CLI/dashboard promote/rollback stay permissive — tier is shown as
     context only, never a block.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

#: Successful-served-traces threshold for T3 (pass-4 #41 default N).
MIN_SERVED_OK = 10

#: Highest implemented tier. T4 ("audited") is reserved: reaching it would
#: require an audit trail sable does not produce yet — never fake it.
MAX_TIER = 3

TIER_NAMES: Dict[int, str] = {
    0: "experimental",
    1: "validated",
    2: "shadow-proven",
    3: "field-proven",
}


def tier_name(tier: int) -> str:
    return TIER_NAMES.get(int(tier), "unknown")


def _get(model: Any, key: str) -> Any:
    """Read a field from a ModelRow or a plain dict (to_dict output)."""
    if isinstance(model, dict):
        return model.get(key)
    return getattr(model, key, None)


def validation_passing(model: Any) -> bool:
    """True when the stored validation report says it passed.

    ``set_validation`` stores whatever report the caller had; only the
    ``validate_model`` contract's ``ok: true`` counts as passing evidence —
    error payloads ({"error": ...}) and absent reports do not.
    """
    v = _get(model, "validation")
    return isinstance(v, dict) and v.get("ok") is True


def shadow_proven(model: Any) -> bool:
    """True when the latest shadow verdict (#32) explicitly says non-regression.

    Skipped verdicts ({"skipped": true, reason}) and absent ones are honest
    absence — never evidence of quality. A regression caps the tier below T2.
    """
    s = _get(model, "shadow")
    return (isinstance(s, dict) and not s.get("skipped")
            and s.get("is_regression") is False)


def derive_tier(model: Any, served_stats: Optional[Dict[str, int]] = None) -> int:
    """Highest tier whose evidence is present (cumulative ladder).

    Pure + deterministic given the inputs. ``served_stats`` comes from
    :func:`compute_served_stats` (or a caller-supplied dict in tests); when it
    is None the T3 rung simply cannot be satisfied — absence is not evidence,
    so row-only reads cap at T2.
    """
    if model is None:
        return 0
    tier = 0
    if validation_passing(model):
        tier = 1
        if shadow_proven(model):
            tier = 2
            stats = served_stats or {}
            ok = int(stats.get("served_ok", 0) or 0)
            neg = int(stats.get("negative_feedback", 0) or 0)
            if ok >= MIN_SERVED_OK and neg == 0:
                tier = 3
    return min(tier, MAX_TIER)


def requirements_hint(model: Any,
                      served_stats: Optional[Dict[str, int]] = None) -> str:
    """Human-readable missing evidence for the NEXT tier (UI/CLI honesty).

    Only ever describes checks actually run against the row/stats it was
    given — never claims evidence that is not present.
    """
    if model is None:
        return "no registry row — nothing to derive from"
    v = _get(model, "validation")
    s = _get(model, "shadow")
    if not validation_passing(model):
        if isinstance(v, dict) and v.get("ok") is False:
            return ("T1 needs a passing validation report — the stored one "
                    f"failed (pass_rate {v.get('pass_rate', '?')})")
        return ("T1 needs a passing validation report "
                "(`sable train validate` golden checks) — none recorded")
    if not shadow_proven(model):
        if isinstance(s, dict) and s.get("skipped"):
            return (f"T2 needs a non-regression shadow verdict — the last one "
                    f"was skipped ({s.get('reason', 'no reason')})")
        if isinstance(s, dict) and s.get("is_regression") is True:
            imp = s.get("improvement")
            return ("T2 needs a non-regression shadow verdict — the last one "
                    f"flagged a regression (improvement {imp} vs "
                    f"{s.get('baseline_id', 'unknown baseline')})")
        return ("T2 needs a shadow-baseline verdict with is_regression=false "
                "(`sable evaluate shadow`) — none recorded")
    stats = served_stats or {}
    ok = int(stats.get("served_ok", 0) or 0)
    neg = int(stats.get("negative_feedback", 0) or 0)
    if not (ok >= MIN_SERVED_OK and neg == 0):  # T3 rung not yet reached
        if neg > 0:
            return (f"T3 needs {MIN_SERVED_OK} served traces without negative "
                    f"feedback since promotion — {neg} negative-feedback "
                    f"trace(s) block it ({ok} clean so far)")
        return (f"T3 needs {MIN_SERVED_OK} served traces with positive-or-"
                f"neutral outcome since promotion — {ok} so far")
    return ("T4 (audited) is reserved and not implemented — no evidence path "
            "exists for it yet")


def compute_served_stats(store: Any, model_id: str, *,
                         since: Optional[str] = None) -> Dict[str, int]:
    """Read-only served-trace counts for one model (cheap single read).

    Returns {served_ok, negative_feedback}: finished traces attributed to the
    model (``model_alias``/``model_version`` == model_id, insights' matching
    rule) with ``timestamp > since``, scored by ``outcome_score`` — >= 0 is
    positive-or-neutral, < 0 is negative feedback, None counts toward neither
    (unscored traces are not evidence). Never raises: an unreadable store
    reads as zeros (honest absence).
    """
    stats = {"served_ok": 0, "negative_feedback": 0}
    try:
        rows = store.list_traces()
    except Exception:  # noqa: BLE001 — tier derivation must never crash a read
        return stats
    for r in rows:
        if model_id not in (r.model_alias, r.model_version):
            continue
        if since and (r.timestamp or "") <= since:
            continue
        score = r.outcome_score
        if score is None:
            continue
        if score < 0:
            stats["negative_feedback"] += 1
        else:
            stats["served_ok"] += 1
    return stats


def served_stats_for(store: Any, model: Any) -> Dict[str, int]:
    """compute_served_stats with the promotion-window floor read off the row
    (``promoted_at``, falling back to ``created_at``). Convenience for the
    dashboard/CLI surfaces so every caller uses the identical window."""
    since = _get(model, "promoted_at") or _get(model, "created_at")
    model_id = _get(model, "model_id")
    if not model_id:
        return {"served_ok": 0, "negative_feedback": 0}
    return compute_served_stats(store, str(model_id), since=since)

"""Capability-tier-aware auto-routing at serve time (review idea #2).

The registry derives evidence-based capability tiers for every model
(:mod:`sable.registry.tiers`: T0 experimental .. T3 field-proven) and the
Phase-6 serving layer can run N instances concurrently
(:mod:`sable.serving.instances`). Until now every request went to whichever
single endpoint the caller resolved. This module turns a multi-instance setup
into an automatic small→big cascade: pick the LOWEST-tier running instance
that meets the request's complexity requirement, and escalate to the
next-higher tier only when a lower-tier attempt fails at the run boundary.

Everything here is deterministic and cheap — no LLM classifier. The feature
is OFF by default (``routing.enabled: false``); with it off nothing in this
module runs and endpoint resolution behaves exactly as before.

Design constraints (deliberate, do not "fix" without reading twice):
  * **Per-run decisions only.** A routing decision is made once per run at
    endpoint-resolution time; escalation happens between whole-run attempts
    (the existing retry/forced-compaction paths finish inside one runner).
    Mid-stream per-message switching would fragment a conversation across
    models with different tokenizers/templates — never do that here.
  * **Explicit wins over routing.** A profile bound to a ``server_instance``
    or an explicit model choice bypasses the router entirely.
  * **Breakers are shared, not duplicated.** The router consults the same
    :func:`sable.core.circuit.get_breaker` view keyed by base_url that the
    agent runner already maintains — it never keeps a private breaker
    registry, so a wedged backend is wedged for everyone exactly once.

Endpoint pool
-------------
``build_pool`` enumerates instance definitions (canonical Phase-6 files),
keeps only those whose runtime state is alive, skips ``embedding`` role
instances (not chat endpoints), and joins each with its registry row to get
the derived tier. Join key: the instance's ``gguf`` path matched against
``ModelRow.gguf_path`` (exact normalized path first, then basename). An
instance whose model has no registry row — or whose state carries no gguf at
all (attach URLs) — falls back to ``UNTIERED_TIER`` (0): absence of evidence
is not evidence, so an unknown model is treated exactly like a T0
"experimental" one. It will still serve tier-agnostic/low requests (that is
where small models belong) but never satisfies a requirement >= 1 unless it
is the *only* thing running (highest-available fallback below).

Request classification rule table (``classify_request``)
--------------------------------------------------------
Evaluated in order; the final requirement is the MAX of everything that
fired. ``None`` means tier-agnostic — any running instance may serve, and
selection still prefers the lowest tier.

  ==========================  ===========================================
  signal                      effect
  ==========================  ===========================================
  explicit hint ``"high"``    requirement = 3 (best available) — direct
  explicit hint ``"low"``     requirement = 0 — direct
  profile hint (config        same mapping as the explicit hint, applied
  ``profile_hints``)          when the run's profile name matches
  est. prompt tokens >        requirement >= 2 (long context wants a
  ``high_context_tokens``     proven model; default 8192)
  tool-heavy profile          requirement >= 1 (default: >= 6 enabled
  (``enabled_tool_names``)    tools — small models mangle tool loops)
  nothing fired               None (tier-agnostic passthrough)
  ==========================  ===========================================

Estimated prompt tokens use chars/4 — the same cheap heuristic the runner's
compression gate falls back to. Callers with a TokenCounter may pass an
``estimate_fn`` and get exact counts instead.

Routing decision (``AutoRouter.select``)
----------------------------------------
With ``prefer: lowest_tier`` (the point of the feature): pick the LOWEST-tier
breaker-allowed instance meeting the requirement. If nothing at/above the
requirement runs, use the HIGHEST available and record a note on the target
(honest degradation, never a silent lie). Candidates whose circuit breaker is
OPEN are skipped; if every candidate is open (or the pool is empty) a
``RoutingError`` names the tiers it wanted and why each instance was skipped.

Escalation semantics (``run_routed``)
-------------------------------------
A whole-run retry driver: run against the selected target; escalate to the
next-higher tier only when the attempt failed for an INFRA reason — a
transport/backend error (``failure_reason`` prefixed ``backend_error:``), an
open circuit, or (opt-in, config-listed markers, empty by default) a
refusal-shaped final response. Model-behavior failures (cancelled,
max_iterations, guardrail halt, verification nudge) never escalate — a bigger
model would repeat them. Each failed attempt records one failure on that
backend's shared breaker so repeated routing away from a wedged instance is
automatic. ``escalate_on_failure: false`` disables escalation but keeps
initial tier selection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..core.circuit import get_breaker
from ..core.failure_types import (
    FAILURE_BACKEND_CIRCUIT_OPEN,
    FAILURE_BACKEND_ERROR_PREFIX,
)
from . import instances as srv_inst

logger = logging.getLogger("sable.serving.router")

#: Fallback tier for instances whose model has no registry row (see module
#: docstring): honest absence of evidence reads as T0 experimental.
UNTIERED_TIER = 0

PREFER_LOWEST = "lowest_tier"
PREFER_HIGHEST = "highest_tier"

#: Hint values understood by classify_request (anything else is ignored).
HINTS = ("auto", "low", "high")


class RoutingError(RuntimeError):
    """No routable endpoint: pool empty or every candidate circuit-open."""


# ------------------------------------------------------------------ config

@dataclass
class RoutingConfig:
    """`routing:` YAML section (serving/router.py). OFF by default — with
    ``enabled`` false no routing code runs and behavior is byte-identical to
    pre-feature sable."""

    enabled: bool = False
    escalate_on_failure: bool = True   # retry a failed run at the next tier
    prefer: str = PREFER_LOWEST        # "lowest_tier" | "highest_tier"
    high_context_tokens: int = 8192    # prompt-token threshold for req >= 2
    tool_heavy_min_tools: int = 6      # enabled tools counting as tool-heavy
    #: per-profile hints: {profile_name: "low"|"high"|"auto"} — same mapping
    #: as the per-run routing_hint.
    profile_hints: Dict[str, str] = field(default_factory=dict)
    #: refusal-shaped response markers that trigger escalation (checked
    #: case-insensitively against the final response). EMPTY by default:
    #: self-reported inability is weak evidence and must be opted into.
    refusal_markers: List[str] = field(default_factory=list)

    @staticmethod
    def from_yaml(data: Dict[str, Any]) -> "RoutingConfig":
        sec = data.get("routing") if isinstance(data, dict) else None
        if not isinstance(sec, dict):
            return RoutingConfig()
        known = {f for f in RoutingConfig.__dataclass_fields__}
        unknown = sorted(k for k in sec if k not in known)
        if unknown:
            logger.warning(
                "routing config: ignoring unknown key(s) %s (known: %s)",
                ", ".join(unknown), ", ".join(sorted(known)))
        kwargs = {k: v for k, v in sec.items() if k in known and v is not None}
        cfg = RoutingConfig(**kwargs)
        if cfg.prefer not in (PREFER_LOWEST, PREFER_HIGHEST):
            logger.warning("routing config: prefer %r unknown, using %s",
                           cfg.prefer, PREFER_LOWEST)
            cfg.prefer = PREFER_LOWEST
        return cfg


def routing_config(cfg: Any) -> RoutingConfig:
    """RoutingConfig from a loaded SableConfig (raw dict section)."""
    return RoutingConfig.from_yaml({"routing": getattr(cfg, "routing", None)})


# --------------------------------------------------------------- pool view

@dataclass
class Candidate:
    """One routable chat endpoint with its derived capability tier."""

    instance_id: str
    tier: int
    base_url: str
    model_name: str
    api_key: Optional[str] = None
    #: "registry" — tier derived from a matched registry row;
    #: "untiered"   — no registry row matched (honest T0 fallback).
    tier_source: str = "registry"


def _norm_path(p: str) -> str:
    try:
        return str(Path(str(p)).expanduser().resolve())
    except OSError:
        return str(p)


def _registry_rows_by_gguf(home: Path) -> List[Any]:
    """Registry rows for tier join; [] when no registry DB exists yet (a
    routing read must never create one)."""
    db = Path(home) / "buffer" / "sable.db"
    if not db.is_file():
        return []
    try:
        from ..registry.registry import ModelRegistry

        reg = ModelRegistry(db)
        try:
            return list(reg.list_models())
        finally:
            reg.close()
    except Exception as e:  # noqa: BLE001 — routing never crashes over a read
        logger.debug("router: registry read failed: %s", e)
        return []


def _tier_for_instance(inst: Dict[str, Any], rows: List[Any]) -> Tuple[int, str]:
    """(tier, tier_source) for one instance definition.

    Join order: exact normalized gguf path, then basename (archive/restore or
    a moved models dir keeps names stable). No match => UNTIERED_TIER."""
    gguf = inst.get("gguf")
    if not gguf or not rows:
        return UNTIERED_TIER, "untiered"
    from ..registry.tiers import derive_tier

    want = _norm_path(str(gguf))
    base = Path(want).name
    row = None
    for r in rows:
        rp = getattr(r, "gguf_path", None)
        if rp and _norm_path(str(rp)) == want:
            row = r
            break
    if row is None and base:
        for r in rows:
            rp = getattr(r, "gguf_path", None)
            if rp and Path(str(rp)).name == base:
                row = r
                break
    if row is None:
        return UNTIERED_TIER, "untiered"
    # Row-only derivation caps at T2 (served-trace stats would need the
    # TraceStore; per-run routing stays a single cheap registry read).
    return derive_tier(row), "registry"


def build_pool(home: Path, cfg: Any = None, *,
               registry_rows: Optional[List[Any]] = None) -> List[Candidate]:
    """Running chat instances joined with tier info: [Candidate].

    Enumeration is the canonical Phase-6 view (instance definition files +
    per-instance runtime state; the reserved "default" folds in the legacy
    serve_state.json). ``embedding``-role instances are excluded — they answer
    /embeddings, not chat. Liveness uses ``instances.state_alive`` (pid check
    for spawns, healthcheck for attaches), so a dead state file never becomes
    a candidate."""
    home = Path(home)
    out: List[Candidate] = []
    try:
        defs = srv_inst.load_instances(home, cfg=cfg)
    except Exception as e:  # noqa: BLE001 — pool build must not crash a run
        logger.warning("router: instance enumeration failed: %s", e)
        return out
    rows = registry_rows if registry_rows is not None \
        else _registry_rows_by_gguf(home)
    for inst in defs:
        iid = str(inst.get("id") or "")
        if not iid or inst.get("role") == "embedding":
            continue
        st = srv_inst.read_state(home, iid)
        if st is None and iid == srv_inst.DEFAULT_ID:
            from .deploy import read_serve_state

            st = read_serve_state(home)
        if not srv_inst.state_alive(st):
            continue
        base_url = str((st or {}).get("base_url") or "").strip()
        if not base_url:
            continue
        tier, source = _tier_for_instance(inst, rows)
        out.append(Candidate(
            instance_id=iid,
            tier=tier,
            base_url=base_url.rstrip("/"),
            model_name=str((st or {}).get("model_name")
                           or inst.get("name") or "model"),
            api_key=(st or {}).get("attach_api_key"),
            tier_source=source,
        ))
    return out


# ------------------------------------------------------------- classification

def estimate_prompt_tokens(prompt: str, history: Optional[List[dict]] = None,
                           *, estimate_fn: Optional[Callable[[str], int]] = None
                           ) -> int:
    """Cheap deterministic size signal: chars/4 over prompt + history text.

    Same heuristic the runner's compression gate uses as fallback; callers
    with a warm TokenCounter pass ``estimate_fn`` for exact counts."""
    parts = [prompt or ""]
    for m in history or []:
        c = m.get("content")
        if isinstance(c, str):
            parts.append(c)
    text = "\n".join(parts)
    if estimate_fn is not None:
        try:
            return max(0, int(estimate_fn(text)))
        except Exception:  # noqa: BLE001 — a dead counter falls back, never fails
            pass
    return len(text) // 4


def classify_request(rc: RoutingConfig, *, prompt: str = "",
                     history: Optional[List[dict]] = None,
                     routing_hint: Optional[str] = None,
                     profile_name: Optional[str] = None,
                     enabled_tool_names: Optional[List[str]] = None,
                     estimate_fn: Optional[Callable[[str], int]] = None,
                     ) -> Optional[int]:
    """Tier requirement for one run per the module-docstring rule table.

    Returns ``None`` for tier-agnostic (nothing fired). Explicit per-run
    hints ("low"/"high") are direct answers — human intent outranks the
    heuristics. "auto" is an explicit no-op."""
    hint = str(routing_hint or "").strip().lower()
    if hint == "high":
        return 3
    if hint == "low":
        return 0
    if not hint and profile_name:
        ph = str((rc.profile_hints or {}).get(str(profile_name)) or "").strip() \
            .lower()
        if ph == "high":
            return 3
        if ph == "low":
            return 0

    requirement: Optional[int] = None

    def _bump(min_tier: int) -> None:
        nonlocal requirement
        requirement = min_tier if requirement is None \
            else max(requirement, min_tier)

    est = estimate_prompt_tokens(prompt, history, estimate_fn=estimate_fn)
    if est > int(rc.high_context_tokens):
        _bump(2)
    if enabled_tool_names is not None \
            and len(enabled_tool_names) >= int(rc.tool_heavy_min_tools):
        _bump(1)
    return requirement


# ---------------------------------------------------------------- selection

def _breaker_allows(base_url: str) -> bool:
    """False only when the shared breaker for this backend is OPEN (a
    half-open breaker passes — the runner's own guard grants the probe)."""
    try:
        return get_breaker(base_url).state != "open"
    except Exception:  # noqa: BLE001 — a breaker we can't read never blocks
        return True


@dataclass
class RouteDecision:
    target: Candidate
    #: requirement honored ("met"), downgraded to highest available
    #: ("highest_fallback"), or pool-wide note strings.
    note: Optional[str] = None


def select_target(pool: List[Candidate], requirement: Optional[int], *,
                  prefer: str = PREFER_LOWEST) -> RouteDecision:
    """Pick the instance for one run. Raises RoutingError when nothing can
    serve, naming the wanted tiers and why each candidate was skipped."""
    if not pool:
        raise RoutingError(
            "no running serving instances to route to"
            + (f" (request wants tier >= {requirement})"
               if requirement else "")
            + " — start a server from the Serve panel or `sable serve`")
    allowed = [c for c in pool if _breaker_allows(c.base_url)]
    if not allowed:
        raise RoutingError(
            "every running instance is circuit-open (backend failures); "
            "refusing to route: " + ", ".join(
                f"{c.instance_id}@tier{c.tier}" for c in pool))
    meeting = [c for c in allowed
               if requirement is None or c.tier >= requirement]
    if not meeting:
        best = max(allowed, key=lambda c: (c.tier, c.instance_id))
        return RouteDecision(target=best, note=(
            f"no running instance at tier >= {requirement}; using highest "
            f"available ({best.instance_id} at tier {best.tier})"))
    # lowest_tier: cheapest model that still meets the requirement (the point
    # of the feature); highest_tier: strongest available. Ties break on
    # instance id so the choice is deterministic.
    meeting.sort(key=lambda c: ((c.tier, c.instance_id)
                                if prefer != PREFER_HIGHEST
                                else (-c.tier, c.instance_id)))
    return RouteDecision(target=meeting[0])


def next_escalation(pool: List[Candidate], tried: List[Candidate]
                    ) -> Optional[Candidate]:
    """Next-higher-tier breaker-allowed instance above every tier already
    tried this run; None when the ladder is exhausted."""
    if not tried:
        return None
    floor = max(c.tier for c in tried)
    ids = {c.instance_id for c in tried}
    higher = [c for c in pool
              if c.instance_id not in ids and c.tier > floor
              and _breaker_allows(c.base_url)]
    if not higher:
        return None
    higher.sort(key=lambda c: (c.tier, c.instance_id))
    return higher[0]


# ------------------------------------------------------------- run driver

def _looks_refusal(rc: RoutingConfig, final_response: str) -> bool:
    if not rc.refusal_markers:
        return False
    low = (final_response or "").lower()
    return any(str(m).lower() in low for m in rc.refusal_markers if m)


@dataclass
class RoutingOutcome:
    result: Any                       # AgentResult from the final attempt
    target: Candidate                 # instance that produced it
    requirement: Optional[int] = None
    attempts: List[Tuple[str, int]] = field(default_factory=list)  # (id,tier)
    note: Optional[str] = None        # selection note (highest_fallback etc.)


def run_routed(router: "AutoRouter", make_runner: Callable[[Candidate], Any],
               task: str, *, history: Optional[List[dict]] = None,
               previous_summary: Optional[str] = None) -> RoutingOutcome:
    """Run one task with per-run tier routing + whole-run escalation.

    ``make_runner(candidate)`` must build a ready runner bound to that
    candidate (base_url/model/api_key). Escalation fires only on INFRA-class
    failures — backend errors (incl. context-overflow 400s surfacing as
    AgentBackendError), circuit-open, or config-listed refusal markers — and
    never mid-stream: each escalation is a fresh whole-run attempt at the
    next tier. Model-behavior failures (cancelled/max_iterations/guardrail)
    return as-is; a bigger model would only repeat them."""
    rc = router.config
    requirement = classify_request(
        rc, prompt=task, history=history,
        routing_hint=router.routing_hint, profile_name=router.profile_name,
        enabled_tool_names=router.enabled_tool_names)
    decision = select_target(router.pool, requirement, prefer=rc.prefer)
    attempts: List[Tuple[str, int]] = []
    tried: List[Candidate] = []
    target = decision.target
    while True:
        runner = make_runner(target)
        exc: Optional[Exception] = None
        res = None
        try:
            if previous_summary is not None:
                res = runner.run(task, history=history,
                                 previous_summary=previous_summary)
            else:
                res = runner.run(task, history=history)
        except Exception as e:  # noqa: BLE001 — classified below
            exc = e
        attempts.append((target.instance_id, target.tier))

        infra_reason = False
        refusal = False
        if exc is not None:
            name = type(exc).__name__
            reason_text = str(exc)
            # AgentBackendError / CircuitOpenError from the runner surface as
            # exceptions when raised outside the run loop; both are infra.
            infra_reason = ("AgentBackendError" in name
                            or "CircuitOpenError" in name
                            or FAILURE_BACKEND_ERROR_PREFIX in reason_text)
        else:
            reason = str(getattr(res, "failure_reason", "") or "")
            infra_reason = (reason.startswith(FAILURE_BACKEND_ERROR_PREFIX)
                            or reason == FAILURE_BACKEND_CIRCUIT_OPEN)
            # Opt-in refusal-shaped self-report (config markers, empty by
            # default): escalate to a bigger model, but do NOT count it as a
            # breaker failure — the backend answered fine, the model bailed.
            refusal = (not infra_reason and res is not None
                       and getattr(res, "completed", True)
                       and _looks_refusal(rc, str(getattr(
                           res, "final_response", "") or "")))

        if not (infra_reason or refusal):
            if res is not None and \
                    getattr(res, "completed", True) and exc is None:
                router.note_success(target)
            return RoutingOutcome(result=res if res is not None else exc,
                                  target=target, requirement=requirement,
                                  attempts=attempts, note=decision.note)

        if infra_reason:
            # Failed at the run boundary on infra: one failure on the shared
            # breaker (the runner records its own too; consecutive counting
            # makes double-counting converge, never diverge).
            router.note_failure(target)
        nxt = next_escalation(router.pool, tried + [target]) \
            if rc.escalate_on_failure else None
        if nxt is None:
            outcome_result: Any = res if res is not None else exc
            tail = f"escalation exhausted at {target.instance_id}"
            note = f"{decision.note}; {tail}" if decision.note else tail
            return RoutingOutcome(result=outcome_result, target=target,
                                  requirement=requirement, attempts=attempts,
                                  note=note)
        logger.info("routing: escalating run from %s (tier %d) to %s (tier %d)",
                    target.instance_id, target.tier, nxt.instance_id, nxt.tier)
        tried.append(target)
        target = nxt


# ------------------------------------------------------------------ facade

class AutoRouter:
    """Drop-in endpoint provider for one run family.

    Constructed at endpoint-resolution time (per run); the pool snapshot is
    taken then — a run never re-routes mid-stream. ``routing_hint`` /
    ``profile_name`` / ``enabled_tool_names`` feed classify_request."""

    def __init__(self, config: RoutingConfig, pool: List[Candidate], *,
                 routing_hint: Optional[str] = None,
                 profile_name: Optional[str] = None,
                 enabled_tool_names: Optional[List[str]] = None) -> None:
        self.config = config
        self.pool = list(pool)
        self.routing_hint = routing_hint
        self.profile_name = profile_name
        self.enabled_tool_names = enabled_tool_names

    @classmethod
    def from_home(cls, home: Path, cfg: Any, **kw: Any) -> "AutoRouter":
        rc = routing_config(cfg)
        return cls(rc, build_pool(home, cfg=cfg), **kw)

    def select(self) -> RouteDecision:
        requirement = classify_request(
            self.config, routing_hint=self.routing_hint,
            profile_name=self.profile_name,
            enabled_tool_names=self.enabled_tool_names)
        return select_target(self.pool, requirement,
                             prefer=self.config.prefer)

    @staticmethod
    def note_success(target: Candidate) -> None:
        get_breaker(target.base_url).record_success()

    @staticmethod
    def note_failure(target: Candidate) -> None:
        get_breaker(target.base_url).record_failure()

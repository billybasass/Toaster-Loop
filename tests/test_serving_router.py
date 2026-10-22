"""Capability-tier-aware auto-routing (review idea #2) — hermetic tests.

Pool construction joins fake instance state files with registry-derived
tiers; classification/selection/escalation are exercised against in-process
stubs and the shared circuit-breaker registry (reset per test by conftest).
The default-off proof is the full suite staying green plus the explicit
no-touch assertions below.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from sable.core.config import SableConfig, load_config
from sable.registry.registry import ModelRegistry
from sable.serving import instances as srv_inst
from sable.serving import router as R


# ------------------------------------------------------------------ helpers

def _mk_registry_model(home: Path, gguf: str, *, validated: bool = False,
                       shadow_ok: bool = False) -> str:
    db = home / "buffer" / "sable.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    reg = ModelRegistry(db)
    try:
        row = reg.register_model(base_model="base")
        reg.set_artifacts(row.model_id, gguf_path=gguf)
        if validated:
            reg.set_validation(row.model_id, {"ok": True, "pass_rate": 1.0},
                               passed=True)
        if shadow_ok:
            reg.set_shadow(row.model_id, {"is_regression": False})
        return row.model_id
    finally:
        reg.close()


def _mk_running_instance(home: Path, iid: str, *, gguf: str | None = None,
                         base_url: str, role: str = "custom",
                         alive: bool = True) -> None:
    inst = {"id": iid, "name": iid, "gguf": gguf, "attach_url": None,
            "port": 0, "role": role, "context_length": None,
            "run_settings": {}, "sampling": {}, "autostart": False}
    srv_inst.save_instance(home, inst)
    st = {"mode": "spawn", "pid": 999_999_999 if not alive else None,
          "base_url": base_url, "model_name": f"{iid}-model"}
    if alive:
        import os
        st["pid"] = os.getpid()  # ours => state_alive True without a server
    srv_inst.write_state(home, iid, st)


def _two_tier_pool(home: Path):
    """small.gguf -> T0 (no evidence), big.gguf -> T2 (validated+shadow)."""
    _mk_registry_model(home, "/models/small.gguf")
    _mk_registry_model(home, "/models/big.gguf", validated=True, shadow_ok=True)
    _mk_running_instance(home, "small", gguf="/models/small.gguf",
                         base_url="http://127.0.0.1:4001/v1")
    _mk_running_instance(home, "big", gguf="/models/big.gguf",
                         base_url="http://127.0.0.1:4002/v1")


class _StubResult:
    def __init__(self, completed=True, failure_reason=None, response="ok"):
        self.completed = completed
        self.failure_reason = failure_reason
        self.final_response = response


class _StubRunner:
    """Scripted runner: each queued behavior is either a result or an exc."""

    def __init__(self, behaviors):
        self._behaviors = list(behaviors)

    def run(self, task, history=None, previous_summary=None):
        b = self._behaviors.pop(0) if self._behaviors else _StubResult()
        if isinstance(b, Exception):
            raise b
        return b


# -------------------------------------------------------------------- config

def test_routing_config_defaults_off():
    rc = R.RoutingConfig()
    assert rc.enabled is False
    assert rc.escalate_on_failure is True
    assert rc.prefer == "lowest_tier"
    assert rc.refusal_markers == []


def test_routing_config_parses_section_and_ignores_unknown(caplog):
    rc = R.RoutingConfig.from_yaml({"routing": {
        "enabled": True, "prefer": "highest_tier", "bogus_key": 1}})
    assert rc.enabled is True and rc.prefer == "highest_tier"
    assert "bogus_key" in caplog.text  # warned, not crashed


def test_load_config_routing_section_no_warning(tmp_path, caplog):
    p = tmp_path / "sable.yaml"
    p.write_text(
        "routing:\n"
        "  enabled: true\n"
        "  escalate_on_failure: false\n"
        "  prefer: lowest_tier\n"
        "  high_context_tokens: 4096\n"
        "  tool_heavy_min_tools: 8\n"
        "  profile_hints: {coding: high}\n"
        "  refusal_markers: ['i cannot']\n",
        encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        cfg = load_config(str(p))
    assert not [r for r in caplog.records if "unknown key" in r.getMessage()]
    assert cfg.routing["enabled"] is True
    assert R.routing_config(cfg).escalate_on_failure is False
    assert R.routing_config(cfg).profile_hints == {"coding": "high"}


def test_bad_prefer_falls_back_with_warning(caplog):
    rc = R.RoutingConfig.from_yaml({"routing": {"prefer": "sideways"}})
    assert rc.prefer == "lowest_tier"
    assert "sideways" in caplog.text


# -------------------------------------------------------------- pool + tiers

def test_pool_joins_registry_tiers(tmp_path):
    _two_tier_pool(tmp_path)
    pool = R.build_pool(tmp_path)
    by_id = {c.instance_id: c for c in pool}
    assert set(by_id) == {"small", "big"}
    assert by_id["small"].tier == 0 and by_id["small"].tier_source == "registry"
    assert by_id["big"].tier == 2
    assert by_id["big"].base_url == "http://127.0.0.1:4002/v1"


def test_pool_skips_dead_and_embedding(tmp_path):
    _mk_running_instance(tmp_path, "dead", gguf=None,
                         base_url="http://127.0.0.1:1/v1", alive=False)
    _mk_running_instance(tmp_path, "embed", gguf=None,
                         base_url="http://127.0.0.1:2/v1", role="embedding")
    assert R.build_pool(tmp_path) == []


def test_untiered_fallback_when_no_registry_row(tmp_path):
    _mk_running_instance(tmp_path, "mystery", gguf="/models/none.gguf",
                         base_url="http://127.0.0.1:5/v1")
    pool = R.build_pool(tmp_path)
    assert len(pool) == 1
    assert pool[0].tier == R.UNTIERED_TIER
    assert pool[0].tier_source == "untiered"


# ------------------------------------------------------------ classification

def test_classify_rule_table():
    rc = R.RoutingConfig()
    # nothing fired -> tier-agnostic passthrough
    assert R.classify_request(rc, prompt="hi") is None
    # explicit hints are direct answers
    assert R.classify_request(rc, prompt="x" * 200_000,
                              routing_hint="low") == 0
    assert R.classify_request(rc, prompt="hi", routing_hint="high") == 3
    assert R.classify_request(rc, prompt="hi", routing_hint="auto") is None
    # long context -> >= 2
    long_prompt = "x" * (4 * rc.high_context_tokens + 8)
    assert R.classify_request(rc, prompt=long_prompt) == 2
    # tool-heavy profile -> >= 1
    tools = ["a", "b", "c", "d", "e", "f"]
    assert R.classify_request(rc, prompt="hi", enabled_tool_names=tools) == 1
    # both fire -> max wins
    assert R.classify_request(rc, prompt=long_prompt,
                              enabled_tool_names=tools) == 2
    # per-profile hint only applies when no explicit hint
    rc2 = R.RoutingConfig(profile_hints={"coding": "high"})
    assert R.classify_request(rc2, prompt="hi", profile_name="coding") == 3
    assert R.classify_request(rc2, prompt="hi", profile_name="other") is None
    assert R.classify_request(rc2, prompt="hi", profile_name="coding",
                              routing_hint="low") == 0


# ----------------------------------------------------------------- selection

def _cand(iid: str, tier: int) -> R.Candidate:
    return R.Candidate(instance_id=iid, tier=tier,
                       base_url=f"http://127.0.0.1/{iid}", model_name=iid)


def test_select_lowest_tier_meeting_requirement():
    pool = [_cand("t2", 2), _cand("t0", 0), _cand("t1", 1)]
    assert R.select_target(pool, None).target.instance_id == "t0"
    assert R.select_target(pool, 1).target.instance_id == "t1"
    assert R.select_target(pool, 2).target.instance_id == "t2"
    d = R.select_target(pool, 3)  # nothing meets it -> highest + honest note
    assert d.target.instance_id == "t2"
    assert d.note is not None and "highest available" in d.note
    hi = R.select_target(pool, None, prefer="highest_tier")
    assert hi.target.instance_id == "t2"


def test_select_empty_pool_error_names_tiers():
    with pytest.raises(R.RoutingError) as e:
        R.select_target([], 2)
    assert "tier >= 2" in str(e.value)
    with pytest.raises(R.RoutingError):
        R.select_target([], None)


def test_select_skips_circuit_open_instances():
    pool = [_cand("small", 0), _cand("big", 2)]
    from sable.core.circuit import get_breaker

    br = get_breaker(pool[0].base_url)
    for _ in range(br.failure_threshold):
        br.record_failure()
    assert br.state == "open"
    d = R.select_target(pool, None)  # small is skipped even though cheaper
    assert d.target.instance_id == "big"
    with pytest.raises(R.RoutingError, match="circuit-open"):
        R.select_target([pool[0]], None)


# ---------------------------------------------------------------- escalation

def _routed_pool():
    return [_cand("small", 0), _cand("mid", 1), _cand("big", 2)]


def test_run_routed_escalates_on_backend_error_result():
    pool = _routed_pool()
    router = R.AutoRouter(R.RoutingConfig(), pool)
    made = []

    def make(cand):
        made.append(cand.instance_id)
        if cand.instance_id == "small":
            return _StubRunner([_StubResult(
                completed=False, failure_reason="backend_error:connection")])
        return _StubRunner([_StubResult(response="big answered")])

    out = R.run_routed(router, make, "do a thing")
    assert made == ["small", "mid"]  # next-higher tier above the failed one
    assert out.attempts == [("small", 0), ("mid", 1)]
    assert out.result.final_response == "big answered"
    assert out.target.instance_id == "mid"
    # breaker recorded exactly one failure on the failed backend only
    from sable.core.circuit import get_breaker

    assert get_breaker("http://127.0.0.1/small").failures == 1
    assert get_breaker("http://127.0.0.1/mid").failures == 0


def test_run_routed_escalates_on_backend_error_exception():
    from sable.core.agent_runner import AgentBackendError

    pool = _routed_pool()
    router = R.AutoRouter(R.RoutingConfig(), pool)

    def make(cand):
        if cand.instance_id == "small":
            # context-overflow 400 surfaces as AgentBackendError in the runner
            return _StubRunner([AgentBackendError(
                "Chat completion failed: HTTP 400 — context_size_exceeded")])
        return _StubRunner([_StubResult()])

    out = R.run_routed(router, make, "long task")
    assert [a[0] for a in out.attempts] == ["small", "mid"]
    assert out.result.completed is True


def test_run_routed_no_escalation_on_model_behavior_failure():
    pool = _routed_pool()
    router = R.AutoRouter(R.RoutingConfig(), pool)

    def make(cand):
        return _StubRunner([_StubResult(
            completed=False, failure_reason="max_iterations")])

    out = R.run_routed(router, make, "task")
    assert [a[0] for a in out.attempts] == ["small"]  # bigger model repeats it
    assert out.result.failure_reason == "max_iterations"


def test_run_routed_escalation_disabled_keeps_first_failure():
    pool = _routed_pool()
    router = R.AutoRouter(R.RoutingConfig(escalate_on_failure=False), pool)

    def make(cand):
        return _StubRunner([_StubResult(
            completed=False, failure_reason="backend_error:connection")])

    out = R.run_routed(router, make, "task")
    assert [a[0] for a in out.attempts] == ["small"]
    assert "escalation exhausted" in (out.note or "")


def test_run_routed_refusal_markers_opt_in():
    pool = _routed_pool()
    # off by default: refusal-shaped answer is returned untouched
    router = R.AutoRouter(R.RoutingConfig(), pool)
    out = R.run_routed(router, lambda c: _StubRunner(
        [_StubResult(response="Sorry, I cannot do that.")]), "task")
    assert [a[0] for a in out.attempts] == ["small"]

    router2 = R.AutoRouter(R.RoutingConfig(refusal_markers=["i cannot"]), pool)
    out2 = R.run_routed(router2, lambda c: _StubRunner(
        [_StubResult(response="Sorry, I cannot do that.")
         if c.instance_id == "small" else _StubResult(response="done")]),
        "task")
    assert [a[0] for a in out2.attempts] == ["small", "mid"]
    # refusal is NOT an infra fault: no breaker failure recorded
    from sable.core.circuit import get_breaker

    assert get_breaker("http://127.0.0.1/small").failures == 0


def test_escalation_skips_circuit_open_higher_tier():
    pool = _routed_pool()
    from sable.core.circuit import get_breaker

    br = get_breaker("http://127.0.0.1/mid")
    for _ in range(br.failure_threshold):
        br.record_failure()
    router = R.AutoRouter(R.RoutingConfig(), pool)

    def make(cand):
        if cand.instance_id == "big":
            return _StubRunner([_StubResult()])
        return _StubRunner([_StubResult(
            completed=False, failure_reason="backend_error:connection")])

    out = R.run_routed(router, make, "task", history=[{"role": "user",
                                                       "content": "x" * 40000}])
    # requirement 2 -> big directly; small/mid never tried
    assert [a[0] for a in out.attempts] == ["big"]


def test_run_routed_single_instance_unchanged():
    """Single-instance pool: routed run hits exactly that instance once —
    behavior identical to an unrouted call."""
    pool = [_cand("only", 1)]
    router = R.AutoRouter(R.RoutingConfig(), pool)
    seen = []

    def make(cand):
        seen.append((cand.instance_id, cand.base_url))
        return _StubRunner([_StubResult(response="solo")])

    out = R.run_routed(router, make, "task")
    assert seen == [("only", "http://127.0.0.1/only")]
    assert out.attempts == [("only", 1)]
    assert out.result.final_response == "solo"


# ------------------------------------------------------- dashboard seam (opt-in)

def _dashboard_app(tmp_path, routing: dict):
    from sable.dashboard.app import DashboardApp

    return DashboardApp(SableConfig(routing=routing), home=tmp_path)


def test_seam_default_off_never_touches_router(tmp_path, monkeypatch):
    # Legacy serve_state resolution must be used untouched when disabled.
    (tmp_path / "serve_state.json").write_text(
        '{"mode": "spawn", "pid": %d, "base_url": "http://127.0.0.1:7/v1",'
        ' "model_name": "legacy"}' % __import__("os").getpid(),
        encoding="utf-8")

    def _boom(*a, **k):
        raise AssertionError("router must not run when disabled")

    monkeypatch.setattr(R, "build_pool", _boom)
    app = _dashboard_app(tmp_path, {})  # routing absent => off
    base, key, model = app._resolve_chat_target(None)
    assert base == "http://127.0.0.1:7/v1" and model == "legacy"


def test_seam_routes_to_lowest_tier_when_enabled(tmp_path):
    _two_tier_pool(tmp_path)
    app = _dashboard_app(tmp_path, {"enabled": True})
    base, _key, model = app._resolve_chat_target(None)
    assert base == "http://127.0.0.1:4001/v1"  # small tier wins by default
    assert model == "small-model"

    base_hi, _, model_hi = app._resolve_chat_target(None, routing_hint="high")
    assert base_hi == "http://127.0.0.1:4002/v1"  # hint escalates selection
    assert model_hi == "big-model"


def test_seam_explicit_binding_bypasses_routing(tmp_path):
    _two_tier_pool(tmp_path)
    app = _dashboard_app(tmp_path, {"enabled": True})
    # Explicit instance choice wins even though routing prefers small.
    base, _key, model = app._resolve_chat_target("big")
    assert base == "http://127.0.0.1:4002/v1"


def test_seam_routing_error_falls_back_to_legacy_503(tmp_path):
    from sable.dashboard.app import HttpError

    app = _dashboard_app(tmp_path, {"enabled": True})
    with pytest.raises(HttpError) as e:  # honest legacy error, no crash
        app._resolve_chat_target(None)
    assert e.value.status == 503

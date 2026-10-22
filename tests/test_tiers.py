"""Pass 4 #41 — derived capability tiers (registry/tiers.py) + policy hooks.

Three groups:
  * derivation ladder: synthetic ModelRows + served_stats produce T0..T3
    exactly per the documented rules (regression shadow caps below T2; row-
    only reads never claim T3).
  * policy hook 1: unattended auto-promotion (run_iterate auto=True) skips
    with an honest reason when tier < validation.min_tier_auto_deploy; knob 0
    disables; manual paths stay permissive. Uses the same stub-worker +
    fake-server harness as tests/test_iterate.py.
  * requirements_hint honesty: never claims evidence that is not present.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sable.curate import CurationConfig  # noqa: E402
from sable.registry.registry import ModelRow  # noqa: E402
from sable.registry.tiers import (  # noqa: E402
    MIN_SERVED_OK, compute_served_stats, derive_tier, requirements_hint,
    served_stats_for, tier_name)
from sable.store.db import TraceRow, TraceStore  # noqa: E402


def row(**kw):
    base: dict = dict(
        model_id="v1", base_model="B/M", parent_adapter=None, dataset_id=None,
        manifest_hash=None, adapter_dir=None, gguf_path=None, quant=None,
        status="candidate", created_at="2026-09-01T00:00:00+00:00",
    )
    base.update(kw)
    return ModelRow(**base)


PASSING = {"ok": True, "pass_rate": 1.0, "total": 2, "passed": 2}
FAILING = {"ok": False, "pass_rate": 0.5, "total": 2, "passed": 1}
SHADOW_OK = {"is_regression": False, "baseline_id": "v0", "n": 4,
             "improvement": 0.1}
SHADOW_REG = {"is_regression": True, "baseline_id": "v0", "n": 4,
              "improvement": -0.2}

GOOD_STATS = {"served_ok": MIN_SERVED_OK, "negative_feedback": 0}


# ------------------------------------------------------------- derivation

def test_ladder_T0_fresh_train():
    assert derive_tier(row()) == 0
    # error-only validation payload (train failure) is not a passing report
    assert derive_tier(row(validation={"error": "boom"})) == 0
    # failing validation stays T0
    assert derive_tier(row(validation=FAILING)) == 0


def test_ladder_T1_validation_passing():
    m = row(validation=PASSING, status="validated")
    assert derive_tier(m) == 1
    # even with perfect served stats, no shadow -> never T2/T3
    assert derive_tier(m, GOOD_STATS) == 1


def test_ladder_T2_shadow_not_regression():
    m = row(validation=PASSING, status="promoted", shadow=SHADOW_OK)
    assert derive_tier(m) == 2
    # regression caps below T2 (i.e. at T1)
    assert derive_tier(row(validation=PASSING, shadow=SHADOW_REG)) == 1
    assert derive_tier(row(validation=PASSING, shadow=SHADOW_REG),
                       GOOD_STATS) == 1
    # skipped verdict is honest absence, not evidence
    skip = row(validation=PASSING, shadow={"skipped": True, "reason": "none"})
    assert derive_tier(skip) == 1
    assert derive_tier(skip, GOOD_STATS) == 1


def test_ladder_T3_served_traces():
    m = row(validation=PASSING, status="promoted", shadow=SHADOW_OK,
            promoted_at="2026-09-02T00:00:00+00:00")
    assert derive_tier(m, GOOD_STATS) == 3
    # one short of N -> T2
    assert derive_tier(
        m, {"served_ok": MIN_SERVED_OK - 1, "negative_feedback": 0}) == 2
    # any negative feedback blocks T3 even past N
    assert derive_tier(
        m, {"served_ok": MIN_SERVED_OK + 5, "negative_feedback": 1}) == 2


def test_T4_reserved_cap():
    # row-only reads (no stats) cap at T2 — absence of stats is not evidence
    m = row(validation=PASSING, shadow=SHADOW_OK)
    assert derive_tier(m) == 2
    assert derive_tier(m, {"served_ok": 10_000, "negative_feedback": 0}) == 3
    assert tier_name(3) == "field-proven"
    assert tier_name(4) == "unknown"


def test_derivation_accepts_dicts_and_none():
    assert derive_tier(None) == 0
    assert derive_tier({"validation": PASSING}) == 1
    assert derive_tier({"validation": PASSING, "shadow": SHADOW_OK},
                       GOOD_STATS) == 3


# ------------------------------------------------------- served stats query

def _trace(store, tid, ts, alias=None, version=None,
           score: Optional[float] = 0.5):
    store.insert_trace(TraceRow(
        trace_id=tid, session_id=None, timestamp=ts, agent_profile="p",
        model_alias=alias, model_version=version, endpoint_label=None,
        task_summary="t", status="success", outcome_score=score,
        num_tool_calls=0, tokens_in=None, tokens_out=None, duration_s=None,
        jsonl_path=f"/tmp/{tid}.jsonl"))


def test_compute_served_stats(tmp_path):
    store = TraceStore(tmp_path / "sable.db")
    _trace(store, "a", "2026-09-03T00:00:00+00:00", alias="v1", score=0.8)
    _trace(store, "b", "2026-09-03T01:00:00+00:00", alias="v1", score=0.0)
    _trace(store, "c", "2026-09-03T02:00:00+00:00", alias="v1", score=-0.4)
    _trace(store, "d", "2026-09-03T03:00:00+00:00", alias="v1", score=None)
    _trace(store, "e", "2026-09-03T04:00:00+00:00", version="v1", score=0.2)
    _trace(store, "f", "2026-09-03T05:00:00+00:00", alias="other", score=0.9)
    # before the `since` floor -> excluded
    _trace(store, "g", "2026-09-01T00:00:00+00:00", alias="v1", score=0.9)

    stats = compute_served_stats(store, "v1", since="2026-09-02T00:00:00+00:00")
    # 0.0 counts positive-or-neutral; None counts toward neither; <0 negative
    assert stats == {"served_ok": 3, "negative_feedback": 1}
    # served_stats_for reads the floor off the row (promoted_at)
    m = row(promoted_at="2026-09-02T00:00:00+00:00")
    assert served_stats_for(store, m) == stats
    store.close()


def test_compute_served_stats_never_raises():
    class Boom:
        def list_traces(self):
            raise RuntimeError("db gone")

    assert compute_served_stats(Boom(), "v1") == {"served_ok": 0,
                                                  "negative_feedback": 0}


# --------------------------------------------------- requirements_hint honesty

def test_requirements_hint_matches_evidence():
    # T0: missing validation, must NOT talk about shadow/field evidence
    h = requirements_hint(row())
    assert "T1" in h and "validation" in h.lower()
    assert "shadow" not in h

    # failing validation names the failure, not a made-up pass
    h = requirements_hint(row(validation=FAILING))
    assert "failed" in h

    # T1: next step is shadow, no false claims about existing verdicts
    h = requirements_hint(row(validation=PASSING))
    assert "T2" in h and "shadow" in h.lower()

    # regression hint quotes the actual stored numbers
    h = requirements_hint(row(validation=PASSING, shadow=SHADOW_REG))
    assert "-0.2" in h and "v0" in h

    # skipped hint carries the stored reason
    h = requirements_hint(row(validation=PASSING,
                              shadow={"skipped": True, "reason": "no replay set"}))
    assert "no replay set" in h

    # T2: next step is served traces; counts come from the stats passed in
    m = row(validation=PASSING, shadow=SHADOW_OK)
    h = requirements_hint(m, {"served_ok": 3, "negative_feedback": 0})
    assert f"{MIN_SERVED_OK}" in h and "3" in h
    h = requirements_hint(m, {"served_ok": 12, "negative_feedback": 2})
    assert "negative" in h.lower()

    # T3: T4 reserved — never invents an evidence path
    h = requirements_hint(m, GOOD_STATS)
    assert "T4" in h and "reserved" in h


# ------------------------------------------------------------- to_dict tier

def test_to_dict_exposes_derived_tier(tmp_path):
    from sable.registry import ModelRegistry

    reg = ModelRegistry(tmp_path / "sable.db")
    m = reg.register_model("B/M")
    d = m.to_dict()
    assert d["tier"] == 0 and d["tier_name"] == "experimental"
    reg.set_validation(m.model_id, PASSING, passed=True)
    reg.set_shadow(m.model_id, SHADOW_OK)
    fetched = reg.get_model(m.model_id)
    assert fetched is not None
    d = fetched.to_dict()
    assert d["tier"] == 2 and d["tier_name"] == "shadow-proven"
    # row-only to_dict caps at T2 (served stats need the trace store)
    cols = {r[1] for r in reg._conn.execute("PRAGMA table_info(models)")}
    assert "tier" not in cols  # derivation-on-read, no stored column
    reg.close()


# ------------------------------------------------ policy hook (pipeline path)

STUB = r'''
import argparse, json, sys
from pathlib import Path
ap = argparse.ArgumentParser()
for f in (("--job", True), ("--kind", True), ("--config", False),
          ("--dataset-dir", False), ("--base-model", False),
          ("--log-path", False), ("--adapter-dir", False),
          ("--gguf-out", False)):
    ap.add_argument(f[0], required=f[1], default=None)
ap.add_argument("--quant", default="q4_k_m")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

def emit(ev, **kw):
    line = json.dumps({"event": ev, "job": a.job, **kw})
    print(line, flush=True)
    if a.log_path:
        Path(a.log_path).parent.mkdir(parents=True, exist_ok=True)
        open(a.log_path, "a").write(line + "\n")

emit("start")
if a.kind in ("sft", "dpo"):
    out = Path(a.config).parent if a.config else Path(".")
    (out / "adapter").mkdir(parents=True, exist_ok=True)
    (out / "adapter" / "adapter_config.json").write_text(json.dumps(
        {"base_model": a.base_model or "stub", "r": 8}))
elif a.kind == "export":
    gg = Path(a.gguf_out)
    gg.parent.mkdir(parents=True, exist_ok=True)
    gg.write_bytes(b"GGUF-stub")
emit("done")
'''

GOLDEN = [{"prompt": "anything", "expect_contains": ["ok-answer"]}]


@pytest.fixture()
def env(tmp_path, monkeypatch):
    from fake_server import FakeOpenAIServer, make_scripted_handler

    home = tmp_path / "uf-tiers"
    monkeypatch.setenv("SABLE_HOME", str(home))
    stub = tmp_path / "stub.py"
    stub.write_text(STUB)

    def starter(gguf_path):
        srv = FakeOpenAIServer(handler_cls=make_scripted_handler(
            {"*": "ok-answer"}))
        srv.start()
        return srv
    return home, stub, starter


def seed(store, home, i):
    ts = f"2026-09-01T1{i:09d}+00:00"[:25]
    p = home / "buffer" / "2026" / "09" / f"tier-{i}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        f.write(json.dumps({"kind": "trace_header"}) + "\n")
        for m in ({"role": "user", "content": f"task {i}"},
                  {"role": "assistant", "content": f"done {i} result=ok"}):
            f.write(json.dumps({"kind": "message", "message": m}) + "\n")
        f.write(json.dumps({"kind": "trace_footer", "status": "success"}) + "\n")
    store.insert_trace(TraceRow(
        trace_id=f"tier-{i}", session_id=None, timestamp=ts,
        agent_profile="fallback", model_alias="base-model",
        model_version=None, endpoint_label=None, task_summary="t",
        status="success", outcome_score=0.9, num_tool_calls=0,
        tokens_in=None, tokens_out=None, duration_s=None, jsonl_path=str(p)))


def _run(home, stub, starter, reg, store, auto, min_tier):
    from sable.training.pipeline import run_iterate
    from sable.training.scheduler import Scheduler, TrainingConfig
    from sable.training.validate import ValidationConfig

    sched = Scheduler(registry=reg, cfg=TrainingConfig(
        base_model="Base/HF-Model"), worker_command=[sys.executable, str(stub)])
    return run_iterate(
        home=home, registry=reg, scheduler=sched, store=store,
        curation_cfg=CurationConfig(min_score=0.5),
        training_cfg=TrainingConfig(base_model="Base/HF-Model"),
        validation_cfg=ValidationConfig(min_pass_rate=1.0, golden_tasks=GOLDEN,
                                        min_tier_auto_deploy=min_tier),
        name="run", kind="sft", server_starter=starter, auto=auto,
    )


def test_auto_promotion_skips_below_min_tier(env):
    """Fresh train = T1 after validation (no shadow) -> auto path skips with
    an honest reason; the model stays validated for manual promotion."""
    home, stub, starter = env
    store = TraceStore(home / "buffer" / "sable.db")
    reg = _registry_for(home)
    for i in range(3):
        seed(store, home, i)
    r = _run(home, stub, starter, reg, store, auto=True, min_tier=2)
    assert not r.ok and not r.promoted
    assert r.stage == "promote"
    assert r.error is not None
    assert "auto-promotion skipped" in r.error and "tier T1" in r.error
    assert r.model_id is not None
    m = reg.get_model(r.model_id)
    assert m is not None and m.status == "validated"  # not promoted
    store.close()
    reg.close()


def test_auto_promotion_knob_zero_disables_policy(env):
    home, stub, starter = env
    store = TraceStore(home / "buffer" / "sable.db")
    reg = _registry_for(home)
    for i in range(3):
        seed(store, home, i)
    r = _run(home, stub, starter, reg, store, auto=True, min_tier=0)
    assert r.ok and r.promoted, r.to_dict()
    store.close()
    reg.close()


def test_manual_iterate_and_promote_permissive_at_low_tier(env):
    """auto=False (manual iterate) promotes at T1; registry.promote itself was
    never tier-aware and stays that way."""
    home, stub, starter = env
    store = TraceStore(home / "buffer" / "sable.db")
    reg = _registry_for(home)
    for i in range(3):
        seed(store, home, i)
    r = _run(home, stub, starter, reg, store, auto=False, min_tier=2)
    assert r.ok and r.promoted, r.to_dict()
    from sable.registry.tiers import derive_tier
    assert r.model_id is not None
    m = reg.get_model(r.model_id)
    assert m is not None
    assert derive_tier(m) < 2  # promoted via a manual path below T2 — allowed
    store.close()
    reg.close()


def _registry_for(home):
    from sable.registry import ModelRegistry

    return ModelRegistry(home / "buffer" / "sable.db")


def test_validation_config_knob_parses():
    from sable.training.validate import ValidationConfig

    vc = ValidationConfig.from_yaml({"validation": {"min_tier_auto_deploy": 0}})
    assert vc.min_tier_auto_deploy == 0
    assert ValidationConfig().min_tier_auto_deploy == 2

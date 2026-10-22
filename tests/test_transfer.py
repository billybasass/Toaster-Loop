"""Pass 4 #37 — cross-profile transfer probe (adapted from PIASystem (MIT)
app/core/transfer_bridge.py).

Covers: transfer run via the reused shadow_eval machinery stores an entry in
models.transfers_json (profile + improvement math, baseline default = the
registry's current active model), candidate-only storage when there is no
active baseline (is_regression null, no fake comparison), honest-empty failure
('no replay set for profile P', nothing stored), the newest-first cap of 20
(21st entry evicts the oldest), tolerant hydration of corrupt transfers_json,
ledger op record_transfer + chain verification, migration from a pre-#37 DB,
and a CLI smoke for `sable evaluate transfer`.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fake_server import FakeOpenAIServer, make_scripted_handler  # noqa: E402
from sable.evaluation.shadow import (  # noqa: E402
    ShadowError,
    no_replay_set_for,
    run_shadow_eval,
)
from sable.registry import ModelRegistry  # noqa: E402
from sable.registry.registry import TRANSFERS_CAP  # noqa: E402
from sable.store.db import TraceRow, TraceStore  # noqa: E402


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "uf-transfer"
    monkeypatch.setenv("SABLE_HOME", str(h))
    (h / "buffer").mkdir(parents=True)
    return h


# ------------------------------------------------------------------ seeding

def seed_trace(store, home, tid, *, ts, profile="writer", status="success",
               score: Optional[float] = 0.9, task=None):
    p = home / "buffer" / f"{tid}.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        f.write(json.dumps({"kind": "trace_header"}) + "\n")
        f.write(json.dumps({"kind": "message", "message": {
            "role": "user", "content": task or f"task {tid}"}}) + "\n")
        f.write(json.dumps({"kind": "message", "message": {
            "role": "assistant", "content": "the old answer"}}) + "\n")
    store.insert_trace(TraceRow(
        trace_id=tid, session_id=None, timestamp=ts, agent_profile=profile,
        model_alias="m", model_version=None, endpoint_label=None,
        task_summary="s", status=status, outcome_score=score,
        num_tool_calls=0, tokens_in=None, tokens_out=None, duration_s=None,
        jsonl_path=str(p), failure_reason=None,
    ))


def _stub_judge_client_for(score_map):
    def client_for(base_url, model_name):
        def client(messages, model):
            text = json.dumps(messages)
            for needle, sc in score_map.items():
                if needle in text:
                    return json.dumps({"score": sc, "reason": "stub"})
            return json.dumps({"score": 0.5, "reason": "stub-mid"})
        return client
    return client_for


# ------------------------------------------------- transfer run (with baseline)

def test_transfer_probe_stores_entry_with_active_baseline(home):
    """Baseline default = the registry's CURRENT active model; improvement math
    and per-profile storage on transfers_json (not shadow_json)."""
    store = TraceStore(home / "buffer" / "sable.db")
    reg = ModelRegistry(home / "buffer" / "sable.db")
    seed_trace(store, home, "w1", ts="2026-09-01T00:00:00+00:00")
    seed_trace(store, home, "w2", ts="2026-09-02T00:00:00+00:00")

    base = reg.register_model("Base/Model")          # v1 -> active
    reg.set_validation(base.model_id, {"ok": True}, passed=True)
    reg.promote(base.model_id)
    cand = reg.register_model("Base/Model")          # v2 candidate

    cand_srv = FakeOpenAIServer(handler_cls=make_scripted_handler(
        {"*": "candidate reply: strong"})).start()
    base_srv = FakeOpenAIServer(handler_cls=make_scripted_handler(
        {"*": "baseline reply: weak"})).start()

    @contextmanager
    def endpoint_for(model):
        srv = cand_srv if model.model_id == cand.model_id else base_srv
        yield srv.base_url, model.model_id

    try:
        result = run_shadow_eval(
            reg, store, candidate_id=cand.model_id, profile="writer", n=6,
            home=home, endpoint_for=endpoint_for,
            judge_client_for=_stub_judge_client_for(
                {"strong": 0.8, "weak": 0.4}),
            store_as="transfer",
        )
    finally:
        cand_srv.stop()
        base_srv.stop()

    assert result["baseline_id"] == base.model_id
    assert result["profile"] == "writer"
    assert result["candidate_mean"] == pytest.approx(0.8)
    assert result["baseline_mean"] == pytest.approx(0.4)
    assert result["improvement"] == pytest.approx(0.4)
    assert result["is_regression"] is False

    row = reg.get_model(cand.model_id)
    assert row is not None
    assert row.shadow is None  # shadow_json untouched — own-profile verdict only
    assert len(row.transfers) == 1
    t = row.transfers[0]
    assert set(t) >= {"ts", "profile", "baseline_id", "n", "candidate_mean",
                      "baseline_mean", "improvement", "is_regression"}
    assert t["profile"] == "writer" and t["improvement"] == pytest.approx(0.4)

    # ledger records the op and the chain still verifies (#31)
    ops = [e["op"] for e in reg.history(limit=20)]
    assert "record_transfer" in ops
    ok, bad, _ = reg.verify_ledger_chain()
    assert ok, f"chain broken at {bad}"
    store.close()
    reg.close()


def test_transfer_probe_explicit_baseline_and_regression(home):
    store = TraceStore(home / "buffer" / "sable.db")
    reg = ModelRegistry(home / "buffer" / "sable.db")
    seed_trace(store, home, "w1", ts="2026-09-01T00:00:00+00:00")
    base = reg.register_model("Base/Model")          # v1 (NOT active)
    cand = reg.register_model("Base/Model")          # v2

    gen = lambda base_url, model_name, task: (
        "CANDIDATE" if model_name == cand.model_id else "BASELINE")

    @contextmanager
    def endpoint_for(model):
        yield "http://127.0.0.1:1/v1", model.model_id

    result = run_shadow_eval(
        reg, store, candidate_id=cand.model_id, profile="writer",
        home=home, endpoint_for=endpoint_for, generate_fn=gen,
        judge_client_for=_stub_judge_client_for({"CANDIDATE": 0.2,
                                                 "BASELINE": 0.9}),
        store_as="transfer", baseline_id=base.model_id,
    )
    assert result["baseline_id"] == base.model_id
    assert result["improvement"] == pytest.approx(-0.7)
    assert result["is_regression"] is True
    row = reg.get_model(cand.model_id)
    assert row is not None
    assert row.transfers[0]["is_regression"] is True
    store.close()
    reg.close()


# ------------------------------------------- candidate-only (no active model)

def test_transfer_probe_candidate_only_when_no_active(home):
    """No active model -> run candidate, store absolute scores with
    baseline_id null and is_regression null — no fabricated comparison."""
    store = TraceStore(home / "buffer" / "sable.db")
    reg = ModelRegistry(home / "buffer" / "sable.db")
    seed_trace(store, home, "w1", ts="2026-09-01T00:00:00+00:00")
    cand = reg.register_model("Base/Model")  # never promoted -> no active

    @contextmanager
    def endpoint_for(model):
        yield "http://127.0.0.1:1/v1", model.model_id

    result = run_shadow_eval(
        reg, store, candidate_id=cand.model_id, profile="writer",
        home=home, endpoint_for=endpoint_for,
        generate_fn=lambda *a: "CANDIDATE",
        judge_client_for=_stub_judge_client_for({"CANDIDATE": 0.7}),
        store_as="transfer",
    )
    assert result["baseline_id"] is None
    assert result["candidate_mean"] == pytest.approx(0.7)
    assert result["baseline_mean"] is None
    assert result["improvement"] is None
    assert result["is_regression"] is None  # documented: no comparison exists

    row = reg.get_model(cand.model_id)
    assert row is not None
    assert row.transfers[0]["baseline_id"] is None
    assert row.transfers[0]["candidate_mean"] == pytest.approx(0.7)
    store.close()
    reg.close()


def test_transfer_probe_candidate_skips_self_as_active(home):
    """The candidate itself being active proves nothing -> candidate-only."""
    store = TraceStore(home / "buffer" / "sable.db")
    reg = ModelRegistry(home / "buffer" / "sable.db")
    seed_trace(store, home, "w1", ts="2026-09-01T00:00:00+00:00")
    m = reg.register_model("Base/Model")
    reg.set_validation(m.model_id, {"ok": True}, passed=True)
    reg.promote(m.model_id)  # candidate IS the active model

    @contextmanager
    def endpoint_for(model):
        yield "http://127.0.0.1:1/v1", model.model_id

    result = run_shadow_eval(
        reg, store, candidate_id=m.model_id, profile="writer", home=home,
        endpoint_for=endpoint_for, generate_fn=lambda *a: "CANDIDATE",
        judge_client_for=_stub_judge_client_for({"CANDIDATE": 0.6}),
        store_as="transfer",
    )
    assert result["baseline_id"] is None and result["improvement"] is None
    store.close()
    reg.close()


# ------------------------------------------------------------- honest-empty

def test_transfer_probe_honest_empty_fails_and_stores_nothing(home):
    """Profile with no replay set -> job failure 'no replay set for profile P',
    nothing stored (a probe asked a question; silence would hide the answer)."""
    store = TraceStore(home / "buffer" / "sable.db")
    reg = ModelRegistry(home / "buffer" / "sable.db")
    m = reg.register_model("Base/Model")

    with pytest.raises(ShadowError) as ei:
        run_shadow_eval(reg, store, candidate_id=m.model_id,
                        profile="ghost", home=home, store_as="transfer")
    assert str(ei.value) == no_replay_set_for("ghost")
    assert "no replay set for profile" in str(ei.value)
    row = reg.get_model(m.model_id)
    assert row is not None
    assert row.transfers == [] and row.shadow is None

    # zero scored tasks must also fail, never store an empty verdict
    seed_trace(store, home, "w1", ts="2026-09-01T00:00:00+00:00")

    @contextmanager
    def endpoint_for(model):
        yield "http://127.0.0.1:1/v1", model.model_id

    with pytest.raises(ShadowError, match="zero scored"):
        run_shadow_eval(reg, store, candidate_id=m.model_id, profile="writer",
                        home=home, endpoint_for=endpoint_for,
                        generate_fn=lambda *a: None, store_as="transfer")
    row = reg.get_model(m.model_id)
    assert row is not None and row.transfers == []
    store.close()
    reg.close()


def test_transfer_probe_rejects_bad_baseline(home):
    store = TraceStore(home / "buffer" / "sable.db")
    reg = ModelRegistry(home / "buffer" / "sable.db")
    seed_trace(store, home, "w1", ts="2026-09-01T00:00:00+00:00")
    cand = reg.register_model("Base/Model")
    with pytest.raises(ShadowError, match="no model v9"):
        run_shadow_eval(reg, store, candidate_id=cand.model_id,
                        profile="writer", home=home, store_as="transfer",
                        baseline_id="v9")
    with pytest.raises(ShadowError, match="must differ"):
        run_shadow_eval(reg, store, candidate_id=cand.model_id,
                        profile="writer", home=home, store_as="transfer",
                        baseline_id=cand.model_id)
    store.close()
    reg.close()


# ------------------------------------------------------------------ cap

def test_record_transfer_cap_evicts_oldest(home):
    reg = ModelRegistry(home / "buffer" / "sable.db")
    m = reg.register_model("Base/Model")
    for i in range(TRANSFERS_CAP + 1):
        reg.record_transfer(m.model_id, {
            "ts": f"2026-09-{(i % 28) + 1:02d}T00:00:00+00:00",
            "profile": f"p{i}", "baseline_id": None, "n": 1,
            "candidate_mean": i / 100.0, "baseline_mean": None,
            "improvement": None, "is_regression": None})
    row = reg.get_model(m.model_id)
    assert row is not None
    assert len(row.transfers) == TRANSFERS_CAP
    # newest first: entry p20 on top, the very first (p0) evicted
    assert row.transfers[0]["profile"] == f"p{TRANSFERS_CAP}"
    assert all(t["profile"] != "p0" for t in row.transfers)
    with pytest.raises(Exception):
        reg.record_transfer(m.model_id, {"no_profile": True})
    reg.close()


def test_transfers_json_tolerant_on_read(home):
    reg = ModelRegistry(home / "buffer" / "sable.db")
    m = reg.register_model("Base/Model")
    reg.record_transfer(m.model_id, {"profile": "a", "n": 1})
    # corrupt the blob behind the API's back — hydration must never crash
    conn = sqlite3.connect(str(reg.db_path))
    conn.execute("UPDATE models SET transfers_json = '{not json' "
                 "WHERE model_id = ?", (m.model_id,))
    conn.commit()
    conn.close()
    row = reg.get_model(m.model_id)
    assert row is not None and row.transfers == []
    # and writing again works from the honest empty state
    reg.record_transfer(m.model_id, {"profile": "b", "n": 2})
    row2 = reg.get_model(m.model_id)
    assert row2 is not None
    assert [t["profile"] for t in row2.transfers] == ["b"]
    reg.close()


# -------------------------------------------------------------- migration

def test_transfers_migration_on_old_db(home):
    db = home / "buffer" / "sable.db"
    reg = ModelRegistry(db)
    reg.register_model("B/M")
    reg.close()

    # Simulate a pre-#37 DB: drop the transfers column entirely.
    conn = sqlite3.connect(str(db))
    conn.execute("ALTER TABLE models DROP COLUMN transfers_json")
    conn.commit()
    conn.close()

    reg2 = ModelRegistry(db)  # _migrate_transfers_column re-adds it
    try:
        m = reg2.get_model("v1")
        assert m is not None and m.transfers == []  # tolerant read pre-write
        reg2.record_transfer("v1", {"profile": "p", "n": 3})
        row = reg2.get_model("v1")
        assert row is not None and row.transfers[0]["profile"] == "p"
    finally:
        reg2.close()


# ------------------------------------------------------------------ CLI smoke

def test_cli_evaluate_transfer_smoke(home, monkeypatch, capsys):
    """`sable evaluate transfer v2 --profile writer` queues the reused
    shadow_eval job with store_as=transfer and reports the stored entry."""
    import sable.cli as cli_mod
    import sable.evaluation.shadow as sh
    from sable.training import scheduler as sched_mod

    monkeypatch.setattr(sched_mod, "gpu_free_mib", lambda: 10**7)

    reg = ModelRegistry(home / "buffer" / "sable.db")
    m = reg.register_model("Base/Model")
    reg.close()

    seen = {}

    def fake_run(registry, store, *, candidate_id, profile=None, n=8,
                 max_regression_pct=5.0, judge_model=None, home=None,
                 endpoint_for=None, generate_fn=None, judge_client_for=None,
                 actor="worker", store_as=None, baseline_id=None):
        seen.update(store_as=store_as, profile=profile, n=n,
                    baseline_id=baseline_id)
        registry.record_transfer(candidate_id, {
            "ts": "2026-09-24T00:00:00+00:00", "profile": profile,
            "baseline_id": baseline_id, "n": n, "candidate_mean": 0.7,
            "baseline_mean": None, "improvement": None,
            "is_regression": None}, actor="cli")
        return {"skipped": False, "profile": profile, "baseline_id": baseline_id,
                "candidate_mean": 0.7, "improvement": None}

    monkeypatch.setattr(sh, "run_shadow_eval", fake_run)

    rc = cli_mod.main(["evaluate", "transfer", m.model_id, "--profile",
                       "writer", "--n", "4"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert seen["store_as"] == "transfer" and seen["profile"] == "writer"
    assert seen["n"] == 4
    assert "transfer probe on 'writer'" in out
    assert "candidate-only" in out

    reg = ModelRegistry(home / "buffer" / "sable.db")
    row = reg.get_model(m.model_id)
    assert row is not None
    assert row.transfers and row.transfers[0]["profile"] == "writer"
    reg.close()


def test_cli_evaluate_transfer_unknown_model(home, capsys):
    from sable.cli import main as cli_main

    rc = cli_main(["evaluate", "transfer", "v9", "--profile", "writer"])
    assert rc == 1
    assert "no model v9" in capsys.readouterr().err


def test_cli_evaluate_transfer_unknown_baseline(home, capsys):
    from sable.cli import main as cli_main

    reg = ModelRegistry(home / "buffer" / "sable.db")
    m = reg.register_model("Base/Model")
    reg.close()
    rc = cli_main(["evaluate", "transfer", m.model_id, "--profile", "writer",
                   "--baseline", "v9"])
    assert rc == 1
    assert "no baseline model v9" in capsys.readouterr().err

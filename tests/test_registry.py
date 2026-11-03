"""Registry-domain tests: CRUD/lineage, hash-chained ledger, capability
tiers, archive-don't-delete lifecycle + disk janitor.

Consolidated from tests/test_registry.py, test_ledger.py, test_tiers.py
and test_archive.py; section banners mark the origin of each block.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from toaster_loop.cli import main as cli_main  # noqa: E402
from toaster_loop.curate import CurationConfig  # noqa: E402
from toaster_loop.registry import (  # noqa: E402
    ModelRegistry, RegistryError, days_since_last_use, recent_ops, record,
    verify_ledger)
from toaster_loop.registry import janitor as janitor_mod  # noqa: E402
from toaster_loop.registry.janitor import load_state, run_janitor  # noqa: E402
from toaster_loop.registry.ledger import GENESIS_HASH, normalize_intent  # noqa: E402
from toaster_loop.registry.registry import ModelRow  # noqa: E402
from toaster_loop.registry.tiers import (  # noqa: E402
    MIN_SERVED_OK, compute_served_stats, derive_tier, requirements_hint,
    served_stats_for, tier_name)
from toaster_loop.serving.supervisor import ServeError  # noqa: E402
from toaster_loop.store.db import TraceRow, TraceStore  # noqa: E402


@pytest.fixture()
def home(tmp_path):
    """Fake TOASTER_LOOP_HOME with a buffer dir (matches conftest's autouse env)."""
    h = tmp_path / ".toaster-loop-test-home"
    (h / "buffer").mkdir(parents=True)
    return h


@pytest.fixture()
def registry(home):
    reg = ModelRegistry(home / "buffer" / "toaster_loop.db")
    yield reg
    reg.close()


def test_register_auto_ids_and_candidate_status(registry):
    m1 = registry.register_model("base-a", dataset_id="d1", manifest_hash="h1",
                                 quant="q4_k_m")
    m2 = registry.register_model("base-a", dataset_id="d2", manifest_hash="h2")
    assert m1.model_id == "v1" and m2.model_id == "v2"
    assert m1.status == "candidate"
    assert m2.parent_adapter is None  # nothing promoted yet -> no lineage edge


def test_artifacts_and_validation_storage(registry):
    m = registry.register_model("base-a")
    registry.set_artifacts(m.model_id, adapter_dir="/tmp/a", gguf_path="/tmp/a.gguf",
                           quant="q5_k_m")
    got = registry.get_model(m.model_id)
    assert got.adapter_dir == "/tmp/a"
    assert got.gguf_path == "/tmp/a.gguf"
    assert got.quant == "q5_k_m"

    registry.set_validation(m.model_id, {"pass_rate": 1.0}, passed=True)
    got = registry.get_model(m.model_id)
    assert got.status == "validated"
    assert got.validation == {"pass_rate": 1.0}


def test_validate_fail_marks_failed(registry):
    m = registry.register_model("base-a")
    registry.set_validation(m.model_id, {"pass_rate": 0.2}, passed=False)
    assert registry.get_model(m.model_id).status == "failed"


def test_promote_requires_validated(registry):
    m = registry.register_model("base-a")
    with pytest.raises(RegistryError):
        registry.promote(m.model_id)  # still candidate


def test_promote_sets_active_pointer_and_supersedes(registry):
    m1 = registry.register_model("base-a")
    registry.set_validation(m1.model_id, {}, passed=True)
    registry.promote(m1.model_id)
    assert registry.get_model("v1").status == "promoted"
    assert registry.active_model().model_id == "v1"

    # second model gains lineage to the active one
    m2 = registry.register_model("base-a")
    assert m2.parent_adapter == "v1"
    registry.set_validation(m2.model_id, {}, passed=True)
    registry.promote(m2.model_id)

    assert registry.get_model("v1").status == "superseded"
    assert registry.get_model("v2").status == "promoted"
    assert registry.active_model().model_id == "v2"


def test_rollback_restores_previous(registry):
    m1 = registry.register_model("base-a")
    registry.set_validation(m1.model_id, {}, passed=True)
    registry.promote("v1")
    m2 = registry.register_model("base-a")
    registry.set_validation(m2.model_id, {}, passed=True)
    registry.promote("v2")

    active = registry.rollback()
    assert active is not None and active.model_id == "v1"
    assert active.status == "promoted"
    assert registry.get_model("v2").status == "rolled_back"
    assert registry.active_model().model_id == "v1"


def test_rollback_with_no_history_clears_pointer(registry):
    registry.register_model("base-a")
    registry.set_validation("v1", {}, passed=True)
    registry.promote("v1")
    active = registry.rollback()
    assert active is None
    assert registry.active_model() is None


def test_rollback_can_repromote_rolled_back(registry):
    registry.register_model("base-a")
    registry.set_validation("v1", {}, passed=True)
    registry.promote("v1")
    registry.rollback()
    # rolled_back models are promotable again (re-deploy prior champion)
    m = registry.promote("v1")
    assert m.status == "promoted"
    assert registry.active_model().model_id == "v1"


def test_per_base_active_pointers(registry):
    registry.register_model("base-a")
    registry.register_model("base-b")
    registry.set_validation("v1", {}, passed=True)
    registry.set_validation("v2", {}, passed=True)
    registry.promote("v1")
    registry.promote("v2")
    assert registry.active_model("base-a").model_id == "v1"
    assert registry.active_model("base-b").model_id == "v2"
    rb = registry.rollback(base_model="base-b")
    assert rb is None  # no previous for base-b
    assert registry.active_model("base-a").model_id == "v1"


def test_list_models_orders_numerically(registry):
    for _ in range(10):
        registry.register_model("base-a")
    ids = [m.model_id for m in registry.list_models()]
    assert ids == [f"v{i}" for i in range(1, 11)]
    # numeric ordering: v9 before v10
    assert ids.index("v9") < ids.index("v10")


# ------------------------------------------------------------------ jobs table

def test_job_enqueue_and_fifo(registry):
    j1 = registry.enqueue_job("sft", dataset_id="d")
    j2 = registry.enqueue_job("export", model_id="v1")
    assert j1.state == "queued" and j2.state == "queued"
    nxt = registry.next_queued()
    # same-second timestamps: insertion order via job_id tiebreak
    assert nxt.job_id == j1.job_id


def test_job_lifecycle_and_cancel(registry):
    j = registry.enqueue_job("sft")
    registry.mark_running(j.job_id, pid=999999)
    assert registry.get_job(j.job_id).state == "running"
    registry.finish_job(j.job_id, "done")
    assert registry.get_job(j.job_id).state == "done"

    j2 = registry.enqueue_job("sft")
    assert registry.cancel_job(j2.job_id).state == "cancelled"
    # finish after cancel must not resurrect
    registry.finish_job(j2.job_id, "done")
    assert registry.get_job(j2.job_id).state == "cancelled"


def test_job_bad_kind_rejected(registry):
    with pytest.raises(RegistryError):
        registry.enqueue_job("banana")


# ------------------------------------------- pass 4 #29: activation tracking

def test_record_activation_updates_row(registry):
    m = registry.register_model("base-a")
    assert m.last_used_at is None
    assert registry.record_activation(m.model_id) is True
    got = registry.get_model(m.model_id)
    assert got.last_used_at is not None
    # re-activation refreshes the stamp (backdate, then bump: second-granular
    # timestamps make direct string comparison flaky, so compare ages instead)
    registry._conn.execute(
        "UPDATE models SET last_used_at = ? WHERE model_id = ?",
        ((datetime.now(timezone.utc) - timedelta(days=5))
         .isoformat(timespec="seconds"), m.model_id),
    )
    registry._conn.commit()
    aged = days_since_last_use(registry.get_model(m.model_id))
    assert aged is not None and 4.9 < aged < 5.2
    assert registry.record_activation(m.model_id) is True
    refreshed = registry.get_model(m.model_id)
    d0 = days_since_last_use(refreshed)
    assert refreshed is not None and d0 is not None and d0 < 1.0


def test_record_activation_unknown_id_safe(registry):
    # Must not raise: called on serve-attach paths where failures cascade.
    assert registry.record_activation("v99") is False


def test_days_since_last_use_semantics(registry):
    m = registry.register_model("base-a")
    # never used -> None ("no usage signal" for the janitor fallback)
    assert days_since_last_use(registry.get_model(m.model_id)) is None

    registry._conn.execute(
        "UPDATE models SET last_used_at = ? WHERE model_id = ?",
        ((datetime.now(timezone.utc) - timedelta(days=3, hours=12))
         .isoformat(timespec="seconds"), m.model_id),
    )
    registry._conn.commit()
    d = days_since_last_use(registry.get_model(m.model_id))
    assert d is not None and 3.4 < d < 3.6

    # unparseable timestamps read as None, never raise
    registry._conn.execute(
        "UPDATE models SET last_used_at = 'not-a-timestamp' WHERE model_id = ?",
        (m.model_id,),
    )
    registry._conn.commit()
    assert days_since_last_use(registry.get_model(m.model_id)) is None


def test_migration_adds_last_used_at_column(tmp_path):
    db = tmp_path / "pre29.db"
    conn = sqlite3.connect(db)
    # Hand-built pre-#29 table: every column up to provenance_json, no last_used_at.
    conn.execute("""
        CREATE TABLE models (
            model_id        TEXT PRIMARY KEY,
            base_model      TEXT NOT NULL,
            parent_adapter  TEXT,
            dataset_id      TEXT,
            manifest_hash   TEXT,
            adapter_dir     TEXT,
            gguf_path       TEXT,
            quant           TEXT,
            status          TEXT NOT NULL CHECK (status IN
              ('candidate','validated','promoted','rolled_back','failed','superseded','archived')),
            created_at      TEXT NOT NULL,
            validation_json TEXT,
            train_loss      REAL,
            eval_loss       REAL,
            promoted_at     TEXT,
            pinned          INTEGER NOT NULL DEFAULT 0,
            pre_archive_paths TEXT,
            provenance_json TEXT
        )""")
    conn.execute(
        "INSERT INTO models (model_id, base_model, status, created_at) "
        "VALUES ('v1','base-a','superseded','2026-01-01T00:00:00+00:00')")
    conn.commit()
    conn.close()

    reg = ModelRegistry(db)
    try:
        cols = {r[1] for r in reg._conn.execute("PRAGMA table_info(models)")}
        assert "last_used_at" in cols
        m = reg.get_model("v1")
        assert m is not None and m.last_used_at is None  # tolerant read
        assert reg.record_activation("v1") is True
        after = reg.get_model("v1")
        assert after is not None and after.last_used_at is not None
    finally:
        reg.close()

    # idempotent: reopening an already-migrated DB must not fail
    reg2 = ModelRegistry(db)
    try:
        again = reg2.get_model("v1")
        assert again is not None and again.last_used_at is not None
    finally:
        reg2.close()

# ------------------------------------------------------------------------
# from tests/test_ledger.py
# ------------------------------------------------------------------------


def _entry(r: sqlite3.Row) -> dict:
    return {"seq": r[0], "ts": r[1], "actor": r[2], "op": r[3],
            "target_id": r[4], "before_json": r[5], "after_json": r[6],
            "intent": r[7]}


def _rows(conn):
    return conn.execute(
        "SELECT seq, ts, actor, op, target_id, before_json, after_json,"
        " intent FROM ledger ORDER BY seq").fetchall()


# ------------------------------------------------------------- chain math

def test_genesis_entry_uses_documented_formula(tmp_path):
    reg = ModelRegistry(tmp_path / "reg.db")
    try:
        h = record(reg._conn, "cli", "promote", "v1",
                   before={"status": "validated"}, after={"status": "promoted"})
        reg._conn.commit()
        r = _rows(reg._conn)[0]
        assert r[0] == 1
        # prev_hash of the first entry is the literal genesis marker.
        assert reg._conn.execute(
            "SELECT prev_hash FROM ledger WHERE seq = 1").fetchone()[0] \
            == GENESIS_HASH
        canonical = json.dumps(_entry(r), sort_keys=True, separators=(",", ":"))
        expect = hashlib.sha256((GENESIS_HASH + canonical).encode()).hexdigest()
        assert h == expect
    finally:
        reg.close()


def test_sequential_entries_chain_and_verify(tmp_path):
    reg = ModelRegistry(tmp_path / "reg.db")
    try:
        for i in range(3):
            record(reg._conn, "cli", f"op{i}", f"t{i}")
        reg._conn.commit()
        ok, bad, total = verify_ledger(reg._conn)
        assert (ok, bad, total) == (True, None, 3)
        # each entry's prev_hash equals its predecessor's hash
        hashes = [row[0] for row in
                  reg._conn.execute("SELECT hash FROM ledger ORDER BY seq")]
        prevs = [row[0] for row in
                 reg._conn.execute("SELECT prev_hash FROM ledger ORDER BY seq")]
        assert prevs[0] == GENESIS_HASH
        assert prevs[1:] == hashes[:-1]
    finally:
        reg.close()


def test_tampered_mid_entry_fails_at_right_seq(tmp_path):
    reg = ModelRegistry(tmp_path / "reg.db")
    try:
        for i in range(4):
            record(reg._conn, "cli", f"op{i}", f"t{i}")
        reg._conn.commit()
        assert verify_ledger(reg._conn)[0] is True
        # direct SQL tamper with the middle entry (test-only mutation)
        reg._conn.execute("UPDATE ledger SET actor = 'evil' WHERE seq = 2")
        reg._conn.commit()
        ok, bad, total = verify_ledger(reg._conn)
        assert ok is False and bad == 2 and total == 4
    finally:
        reg.close()


def test_tampered_hash_breaks_next_link(tmp_path):
    reg = ModelRegistry(tmp_path / "reg.db")
    try:
        for i in range(3):
            record(reg._conn, "cli", f"op{i}", f"t{i}")
        reg._conn.commit()
        reg._conn.execute("UPDATE ledger SET hash = '0' * 64 WHERE seq = 1")
        reg._conn.commit()
        ok, bad, _ = verify_ledger(reg._conn)
        # seq 1's stored hash no longer matches; first mismatch is seq 1
        assert ok is False and bad == 1
    finally:
        reg.close()


def test_two_handles_do_not_fork_the_chain(tmp_path):
    db = tmp_path / "reg.db"
    reg_a = ModelRegistry(db)
    reg_b = ModelRegistry(db)
    try:
        # interleaved writes from two handles on the same DB file
        for i in range(5):
            record(reg_a._conn, "cli", f"a{i}", f"ta{i}")
            reg_a._conn.commit()
            record(reg_b._conn, "scheduler", f"b{i}", f"tb{i}")
            reg_b._conn.commit()
        ok, bad, total = verify_ledger(reg_a._conn)
        assert (ok, bad, total) == (True, None, 10)
    finally:
        reg_a.close()
        reg_b.close()


def test_intent_column_stored_and_hashed(tmp_path):
    reg = ModelRegistry(tmp_path / "reg.db")
    try:
        record(reg._conn, "cli", "promote", "v1", intent="expected: +5% pass rate")
        reg._conn.commit()
        ok, _, _ = verify_ledger(reg._conn)
        assert ok is True
        ops = recent_ops(reg._conn, limit=5)
        assert ops[0]["intent"] == "expected: +5% pass rate"
    finally:
        reg.close()


# ------------------------------------------------------- instrumentation

def test_promote_records_exactly_one_entry(registry):
    registry.register_model("base-a")
    registry.set_validation("v1", {}, passed=True)
    n_before = len(recent_ops(registry._conn, limit=100))
    registry.promote("v1", actor="dashboard")
    ops = recent_ops(registry._conn, limit=100)
    assert len(ops) == n_before + 1
    e = ops[0]
    assert (e["op"], e["actor"], e["target_id"]) == ("promote", "dashboard", "v1")
    assert e["before"] == {"status": "validated", "previous_active": None}
    assert e["after"] == {"status": "promoted"}


def test_promote_default_actor_is_cli(registry):
    registry.register_model("base-a")
    registry.set_validation("v1", {}, passed=True)
    registry.promote("v1")
    assert recent_ops(registry._conn, limit=5)[0]["actor"] == "cli"


def test_archive_and_restore_each_record_one_entry(registry, tmp_path):
    gguf = tmp_path / "artifacts" / "v1.gguf"
    gguf.parent.mkdir()
    gguf.write_text("x")
    registry.register_model("base-a")
    registry.set_artifacts("v1", gguf_path=str(gguf))
    registry.set_validation("v1", {}, passed=False)  # -> failed (archivable)
    n = len(recent_ops(registry._conn, limit=100))
    registry.archive_model("v1", home=tmp_path / "home", actor="scheduler")
    ops = recent_ops(registry._conn, limit=100)
    assert len(ops) == n + 1
    assert (ops[0]["op"], ops[0]["actor"]) == ("archive", "scheduler")
    assert ops[0]["before"] == {"status": "failed"}
    assert ops[0]["after"] == {"status": "archived"}

    registry.restore_model("v1")
    ops = recent_ops(registry._conn, limit=100)
    assert len(ops) == n + 2
    assert (ops[0]["op"], ops[0]["actor"]) == ("restore", "cli")
    assert ops[0]["after"] == {"status": "failed"}


def test_set_losses_records_before_after_diff(registry):
    registry.register_model("base-a")
    registry.set_losses("v1", train_loss=1.5, eval_loss=0.5, actor="worker")
    e = recent_ops(registry._conn, limit=5)[0]
    assert (e["op"], e["actor"], e["target_id"]) == ("set_losses", "worker", "v1")
    assert e["before"] == {"train_loss": None, "eval_loss": None}
    assert e["after"] == {"train_loss": 1.5, "eval_loss": 0.5}

    registry.set_losses("v1", train_loss=1.2)
    e = recent_ops(registry._conn, limit=5)[0]
    assert e["before"] == {"train_loss": 1.5, "eval_loss": 0.5}
    assert e["after"] == {"train_loss": 1.2, "eval_loss": 0.5}


def test_pin_unpin_record_entries(registry):
    registry.register_model("base-a")
    registry.set_pinned("v1", True)
    e = recent_ops(registry._conn, limit=5)[0]
    assert (e["op"], e["target_id"]) == ("pin", "v1")
    assert e["before"] == {"pinned": False} and e["after"] == {"pinned": True}
    registry.set_pinned("v1", False, actor="dashboard")
    e = recent_ops(registry._conn, limit=5)[0]
    assert (e["op"], e["actor"]) == ("unpin", "dashboard")


def test_job_transitions_record_entries(registry):
    job = registry.enqueue_job("sft")
    n = len(recent_ops(registry._conn, limit=100))  # enqueue itself: not ledgered
    assert n == 0
    registry.mark_running(job.job_id, pid=4242)
    e = recent_ops(registry._conn, limit=5)[0]
    assert (e["op"], e["actor"]) == ("job_running", "scheduler")
    registry.finish_job(job.job_id, "done")
    e = recent_ops(registry._conn, limit=5)[0]
    assert (e["op"], e["target_id"]) == ("job_done", job.job_id)
    assert e["before"] == {"state": "running"}
    assert e["after"] == {"state": "done", "error": None}


def test_cancel_job_records_entry(registry):
    job = registry.enqueue_job("sft")
    registry.cancel_job(job.job_id, actor="dashboard")
    e = recent_ops(registry._conn, limit=5)[0]
    assert (e["op"], e["actor"]) == ("job_cancelled", "dashboard")


def test_history_and_verify_helpers(registry):
    registry.register_model("base-a")
    registry.set_validation("v1", {}, passed=True)
    registry.promote("v1")
    ops = registry.history(limit=10)
    assert ops[0]["op"] == "promote"  # newest first: register/validate/promote
    assert registry.verify_ledger_chain() == (True, None, 3)


def test_failed_mutation_leaves_no_ledger_entry(tmp_path):
    # Ledger shares the mutation transaction: a refused promote writes nothing.
    reg = ModelRegistry(tmp_path / "reg.db")
    try:
        reg.register_model("base-a")
        with pytest.raises(Exception):
            reg.promote("v1")  # candidate -> RegistryError, rolls back too
        ops = recent_ops(reg._conn, limit=5)
        assert [o["op"] for o in ops] == ["register"]  # no promote entry
    finally:
        reg.close()


# ------------------------------------------------------------- CLI smoke

def test_cli_history_and_verify(tmp_path, monkeypatch, capsys):
    import os


    db = Path(os.environ["TOASTER_LOOP_HOME"]) / "buffer" / "toaster_loop.db"
    reg = ModelRegistry(db)
    try:
        reg.register_model("base-a")
        reg.set_validation("v1", {}, passed=True)
        reg.promote("v1")
    finally:
        reg.close()

    assert cli_main(["history"]) == 0
    out = capsys.readouterr().out
    assert "promote" in out and "cli" in out and "v1" in out

    assert cli_main(["history", "--verify"]) == 0
    # register + validate + promote all ledgered
    assert "chain OK (3 entries verified)" in capsys.readouterr().out

    # tamper -> verify reports the break and exits non-zero
    conn = sqlite3.connect(str(db))
    conn.execute("UPDATE ledger SET op = 'lie' WHERE seq = 2")
    conn.commit()
    conn.close()
    assert cli_main(["history", "--verify"]) == 1
    err = capsys.readouterr().err
    assert "chain BROKEN at seq 2" in err


def test_cli_history_empty_home(capsys):

    assert cli_main(["history", "--verify"]) == 0
    out = capsys.readouterr().out
    assert "no operations recorded yet" in out
    assert "chain OK (0 entries verified)" in out


# ------------------------------------------------- pass 4 #36: intent capture

def test_normalize_intent_caps_collapses_and_empties():
    assert normalize_intent(None) is None
    assert normalize_intent("   ") is None          # absent, not ''
    assert normalize_intent("fix  tool\ncalls") == "fix tool calls"
    long = "x" * 500
    capped = normalize_intent(long) or ""
    assert len(capped) == 200 and capped.endswith("\u2026")
    assert normalize_intent("y" * 200) == "y" * 200  # exactly at cap untouched


def test_promote_intent_survives_hashing_and_verify(registry):
    registry.register_model("base-a")
    registry.set_validation("v1", {}, passed=True)
    long_intent = "expect " + "z" * 400  # truncated before hashing
    registry.promote("v1", intent=long_intent)
    ok, bad, _ = verify_ledger(registry._conn)       # chain intact w/ intent
    assert (ok, bad) == (True, None)
    e = recent_ops(registry._conn, limit=5)[0]
    assert e["intent"] == "expect " + "z" * 192 + "\u2026"
    assert len(e["intent"]) == 200
    # tampering the intent breaks the chain (it participates in the hash)
    registry._conn.execute("UPDATE ledger SET intent = 'lied' WHERE seq = 3")
    registry._conn.commit()
    assert verify_ledger(registry._conn)[0] is False


def test_promote_without_intent_stays_null_and_rollback_records_intent(
        registry):
    registry.register_model("base-a")
    registry.set_validation("v1", {}, passed=True)
    registry.promote("v1")
    assert recent_ops(registry._conn, limit=5)[0]["intent"] is None
    registry.register_model("base-a")               # v2 -> v1 superseded
    registry.set_validation("v2", {}, passed=True)
    registry.promote("v2")
    registry.rollback(intent="v2 hallucinates tools")
    e = recent_ops(registry._conn, limit=5)[0]
    assert (e["op"], e["intent"]) == ("rollback", "v2 hallucinates tools")


def test_cli_promote_intent_then_history_intents(tmp_path, capsys):
    import os


    db = Path(os.environ["TOASTER_LOOP_HOME"]) / "buffer" / "toaster_loop.db"
    reg = ModelRegistry(db)
    try:
        reg.register_model("base-a")
        reg.set_validation("v1", {}, passed=True)
    finally:
        reg.close()

    assert cli_main(["models", "promote", "v1",
                     "--intent", "fix tool-call regressions"]) == 0
    capsys.readouterr()

    assert cli_main(["history", "--intents"]) == 0
    out = capsys.readouterr().out
    assert "fix tool-call regressions" in out and "promote" in out
    assert "INTENT" in out  # table header

    # the recorded intent survived hashing: chain still verifies
    assert cli_main(["history", "--verify"]) == 0
    assert "chain OK" in capsys.readouterr().out


def test_cli_history_intents_honest_empty(capsys):

    assert cli_main(["history", "--intents"]) == 0
    out = capsys.readouterr().out
    assert "no recorded intents" in out
    # honesty: the hint is a hint, not a fabricated row
    assert "INTENT" not in out


def test_dashboard_promote_accepts_optional_intent(tmp_path):
    from toaster_loop.core.config import ToasterLoopConfig
    from toaster_loop.dashboard.app import DashboardApp

    home = tmp_path / "dash-home"
    db = home / "buffer" / "toaster_loop.db"
    reg = ModelRegistry(db)
    try:
        reg.register_model("base-a")
        reg.set_validation("v1", {}, passed=True)
    finally:
        reg.close()

    app = DashboardApp(ToasterLoopConfig(), home=home)
    r = app.handle("POST", "/api/models/promote",
                   body={"model_id": "v1", "intent": "fewer tool errors"})
    assert r.status == 200
    reg = ModelRegistry(db)
    try:
        assert recent_ops(reg._conn, limit=5)[0]["intent"] == \
            "fewer tool errors"
        # empty intent stays frictionless -> NULL, not ''
        reg.register_model("base-a")
        reg.set_validation("v2", {}, passed=True)
    finally:
        reg.close()
    r = app.handle("POST", "/api/models/promote", body={"model_id": "v2"})
    assert r.status == 200
    reg = ModelRegistry(db)
    try:
        assert recent_ops(reg._conn, limit=5)[0]["intent"] is None
        assert verify_ledger(reg._conn)[0] is True
    finally:
        reg.close()

# ------------------------------------------------------------------------
# from tests/test_tiers.py
# ------------------------------------------------------------------------


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
    store = TraceStore(tmp_path / "toaster_loop.db")
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

    reg = ModelRegistry(tmp_path / "toaster_loop.db")
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
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(home))
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
    from toaster_loop.training.pipeline import run_iterate
    from toaster_loop.training.scheduler import Scheduler, TrainingConfig
    from toaster_loop.training.validate import ValidationConfig

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
    store = TraceStore(home / "buffer" / "toaster_loop.db")
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
    store = TraceStore(home / "buffer" / "toaster_loop.db")
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
    store = TraceStore(home / "buffer" / "toaster_loop.db")
    reg = _registry_for(home)
    for i in range(3):
        seed(store, home, i)
    r = _run(home, stub, starter, reg, store, auto=False, min_tier=2)
    assert r.ok and r.promoted, r.to_dict()
    assert r.model_id is not None
    m = reg.get_model(r.model_id)
    assert m is not None
    assert derive_tier(m) < 2  # promoted via a manual path below T2 — allowed
    store.close()
    reg.close()


def _registry_for(home):

    return ModelRegistry(home / "buffer" / "toaster_loop.db")


def test_validation_config_knob_parses():
    from toaster_loop.training.validate import ValidationConfig

    vc = ValidationConfig.from_yaml({"validation": {"min_tier_auto_deploy": 0}})
    assert vc.min_tier_auto_deploy == 0
    assert ValidationConfig().min_tier_auto_deploy == 2

# ------------------------------------------------------------------------
# from tests/test_archive.py
# ------------------------------------------------------------------------


def _iso(days_ago: float) -> str:
    ts = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return ts.isoformat(timespec="seconds")


def _make_artifacts(home, registry, model_id, gguf_bytes=b"G" * 100,
                    adapter_files=None):
    """Fake GGUF + adapter dir under home/models, recorded on the row."""
    gguf = home / "models" / "gguf" / f"{model_id}-q4_k_m.gguf"
    gguf.parent.mkdir(parents=True, exist_ok=True)
    gguf.write_bytes(gguf_bytes)
    adapter = home / "models" / "adapters" / model_id
    adapter.mkdir(parents=True, exist_ok=True)
    files = adapter_files or {"adapter_model.bin": b"A" * 50}
    for name, data in files.items():
        (adapter / name).write_bytes(data)
    registry.set_artifacts(model_id, adapter_dir=str(adapter),
                           gguf_path=str(gguf))
    return adapter, gguf


def _age(registry, model_id, days_ago, promoted=True):
    """Backdate a row so janitor age rules see it as old."""
    col = "promoted_at" if promoted else "created_at"
    registry._conn.execute(
        f"UPDATE models SET {col} = ? WHERE model_id = ?",
        (_iso(days_ago), model_id),
    )
    registry._conn.commit()


def _set_last_used(registry, model_id, days_ago):
    """Stamp last_used_at directly (days_ago None -> never used)."""
    val = None if days_ago is None else _iso(days_ago)
    registry._conn.execute(
        "UPDATE models SET last_used_at = ? WHERE model_id = ?",
        (val, model_id),
    )
    registry._conn.commit()


def _superseded(registry, base="base-a"):
    """Register v_n, promote twice so the older row ends superseded."""
    n = len(registry.list_models()) + 1
    registry.register_model(base)
    mid = f"v{n}"
    registry.set_validation(mid, {}, passed=True)
    registry.promote(mid)
    n2 = n + 1
    registry.register_model(base)
    mid2 = f"v{n2}"
    registry.set_validation(mid2, {}, passed=True)
    registry.promote(mid2)
    assert registry.get_model(mid).status == "superseded"
    return mid, mid2


# --------------------------------------------------------- archive/restore

def test_archive_restore_roundtrip_bytes_identical(home, registry):
    mid, _ = _superseded(registry)
    adapter, gguf = _make_artifacts(home, registry, mid)
    before_gguf = gguf.read_bytes()
    before_adapter = {f.name: f.read_bytes() for f in adapter.iterdir()}

    m = registry.archive_model(mid, home=home)
    assert m.status == "archived"
    assert m.gguf_path is None and m.adapter_dir is None  # unserveable
    assert not gguf.exists() and not adapter.exists()
    archive_dir = Path(m.pre_archive["archive_dir"])
    assert (archive_dir / gguf.name).read_bytes() == before_gguf

    back = registry.restore_model(mid)
    assert back.status == "superseded"  # previous terminal status restored
    assert back.gguf_path == str(gguf) and back.adapter_dir == str(adapter)
    assert gguf.read_bytes() == before_gguf
    assert {f.name: f.read_bytes() for f in adapter.iterdir()} == before_adapter
    assert back.pre_archive is None  # record consumed
    assert not archive_dir.exists()  # empty per-model dir tidied


def test_archive_records_previous_status_and_paths(home, registry):
    mid, _ = _superseded(registry)
    adapter, gguf = _make_artifacts(home, registry, mid)
    m = registry.archive_model(mid, home=home)
    rec = m.pre_archive
    assert rec["previous_status"] == "superseded"
    assert rec["paths"][str(adapter)] == str(Path(rec["archive_dir"]) / adapter.name)
    assert rec["adapter_dir"] == str(adapter)  # column snapshot for restore
    assert rec["gguf_path"] == str(gguf)
    assert rec["archived_at"]


def test_archive_collision_suffixes(home, registry):
    """Same artifact filenames from different dirs coexist in the archive."""
    m1 = home / "run-a" / "model.gguf"
    m2 = home / "run-b" / "model.gguf"
    m1.parent.mkdir(parents=True); m2.parent.mkdir(parents=True)
    m1.write_bytes(b"one"); m2.write_bytes(b"two")

    registry.register_model("base-a")
    registry.set_validation("v1", {}, passed=False)  # failed -> archivable
    registry.set_artifacts("v1", gguf_path=str(m1))
    registry.register_model("base-a")
    registry.set_validation("v2", {}, passed=False)
    registry.set_artifacts("v2", gguf_path=str(m2))

    a1 = registry.archive_model("v1", home=home)
    a2 = registry.archive_model("v2", home=home)
    assert a1.pre_archive["archive_dir"] != a2.pre_archive["archive_dir"]
    assert a2.pre_archive["archive_dir"].endswith("v2-2") or \
        Path(a2.pre_archive["archive_dir"]).name == "v2"
    # distinct archived copies, contents intact
    dests = {p for p in a1.pre_archive["paths"].values()} | \
            {p for p in a2.pre_archive["paths"].values()}
    assert len(dests) == 2
    contents = sorted(Path(d).read_bytes() for d in dests)
    assert contents == [b"one", b"two"]


def test_archive_same_id_twice_after_restore(home, registry):
    mid, _ = _superseded(registry)
    _, gguf = _make_artifacts(home, registry, mid)
    first = registry.archive_model(mid, home=home)
    # Keep the first archive dir non-empty (foreign file) so restore cannot
    # tidy it and the second archive must collide-suffix.
    (Path(first.pre_archive["archive_dir"]) / "stray").write_bytes(b"x")
    registry.restore_model(mid)
    second = registry.archive_model(mid, home=home)
    assert first.pre_archive["archive_dir"] != second.pre_archive["archive_dir"]
    assert Path(second.pre_archive["archive_dir"]).name == f"{mid}-2"
    assert (Path(first.pre_archive["archive_dir"]) / "stray").exists()


def test_archive_skips_gguf_nested_in_adapter_dir(home, registry):
    mid, _ = _superseded(registry)
    adapter, gguf = _make_artifacts(home, registry, mid)
    nested = adapter / "nested.gguf"
    nested.write_bytes(b"N")
    registry.set_artifacts(mid, gguf_path=str(nested))
    m = registry.archive_model(mid, home=home)
    # only the adapter dir moved; the nested gguf rode along inside it
    assert len(m.pre_archive["paths"]) == 1
    archived_adapter = Path(next(iter(m.pre_archive["paths"].values())))
    assert (archived_adapter / "nested.gguf").read_bytes() == b"N"


def test_archive_refusals(home, registry):
    # candidate
    registry.register_model("base-a")
    with pytest.raises(RegistryError, match="candidate"):
        registry.archive_model("v1", home=home)

    # validated (not terminal)
    registry.set_validation("v1", {}, passed=True)
    with pytest.raises(RegistryError, match="supersede"):
        registry.archive_model("v1", home=home)

    # promoted / active
    registry.promote("v1")
    with pytest.raises(RegistryError, match="promoted"):
        registry.archive_model("v1", home=home)

    # pinned (terminal but protected)
    registry.register_model("base-a")
    registry.set_validation("v2", {}, passed=True)
    registry.promote("v2")  # v1 -> superseded, archivable now
    registry.set_pinned("v1", True)
    with pytest.raises(RegistryError, match="pinned"):
        registry.archive_model("v1", home=home)

    # unknown id / already archived
    with pytest.raises(RegistryError, match="no model v99"):
        registry.archive_model("v99", home=home)
    registry.set_pinned("v1", False)
    registry.archive_model("v1", home=home)
    with pytest.raises(RegistryError, match="already archived"):
        registry.archive_model("v1", home=home)


def test_archive_missing_artifact_refused(home, registry):
    mid, _ = _superseded(registry)
    registry.set_artifacts(mid, gguf_path=str(home / "nowhere.gguf"))
    with pytest.raises(RegistryError, match="missing"):
        registry.archive_model(mid, home=home)
    assert registry.get_model(mid).status == "superseded"  # row untouched


def test_restore_refusals(home, registry):
    mid, _ = _superseded(registry)
    _, gguf = _make_artifacts(home, registry, mid)

    # not archived yet
    with pytest.raises(RegistryError, match="only archived"):
        registry.restore_model(mid)

    registry.archive_model(mid, home=home)

    # target path occupied -> refuse, archive stays intact
    gguf.parent.mkdir(parents=True, exist_ok=True)
    gguf.write_bytes(b"impostor")
    with pytest.raises(RegistryError, match="already exists"):
        registry.restore_model(mid)
    assert Path(registry.get_model(mid).pre_archive["paths"][str(gguf)]).exists()

    # tampered archive (file gone) -> refuse
    gguf.unlink()
    archived = Path(json.loads(
        registry._conn.execute("SELECT pre_archive_paths FROM models "
                               "WHERE model_id = ?", (mid,)).fetchone()[0]
    )["paths"][str(gguf)])
    archived.unlink()
    with pytest.raises(RegistryError, match="missing"):
        registry.restore_model(mid)


# ------------------------------------------------------------------ pin

def test_pin_unpin_roundtrip(registry):
    registry.register_model("base-a")
    assert registry.get_model("v1").pinned is False
    m = registry.set_pinned("v1", True)
    assert m.pinned is True
    m = registry.set_pinned("v1", False)
    assert m.pinned is False
    with pytest.raises(RegistryError, match="no model"):
        registry.set_pinned("v99", True)


# ---------------------------------------------------------------- janitor

def test_janitor_archives_only_old_terminal_unpinned(home, registry):
    def new_model(base):
        m = registry.register_model(base)
        return m.model_id

    # old superseded (promoted_at aged) -> archive
    s1, _ = _superseded(registry)
    _make_artifacts(home, registry, s1)
    _age(registry, s1, 40)

    # old failed (created_at aged, never promoted) -> archive
    f1 = new_model("base-a")
    registry.set_validation(f1, {}, passed=False)
    _age(registry, f1, 40, promoted=False)
    _make_artifacts(home, registry, f1)

    # old but pinned -> skip
    s2, _ = _superseded(registry, base="base-b")
    _make_artifacts(home, registry, s2)
    _age(registry, s2, 40)
    registry.set_pinned(s2, True)

    # recent superseded -> skip
    s3, _ = _superseded(registry, base="base-c")
    _make_artifacts(home, registry, s3)
    _age(registry, s3, 5)

    # promoted / candidate never touched even when ancient
    p1 = new_model("base-d")
    registry.set_validation(p1, {}, passed=True)
    registry.promote(p1)
    _age(registry, p1, 400)
    c1 = new_model("base-e")
    _age(registry, c1, 400, promoted=False)

    summary = run_janitor(home, days=30, registry=registry)
    assert sorted(summary["archived"]) == sorted([s1, f1])
    assert summary["freed_bytes"] > 0 and not summary["errors"]
    statuses = {m.model_id: m.status for m in registry.list_models()}
    assert statuses[s2] != "archived" and statuses[s3] != "archived"
    assert statuses[p1] == "promoted" and statuses[c1] == "candidate"


def test_janitor_skips_recently_used_models(home, registry):
    """Pass 4 #29: among equally-old terminal models, a recently-used one is
    spared while the never/long-unused one is archived."""
    # two equally old superseded models, same base so both age identically
    used, _ = _superseded(registry)              # older row ends superseded
    unused, _ = _superseded(registry, base="base-b")
    _make_artifacts(home, registry, used)
    _make_artifacts(home, registry, unused)
    _age(registry, used, 40)
    _age(registry, unused, 40)
    # 'used' was activated yesterday (inside the 30-day window); 'unused'
    # was last touched before the window -> only it gets archived.
    _set_last_used(registry, used, 1)
    _set_last_used(registry, unused, 60)

    summary = run_janitor(home, days=30, registry=registry)
    assert summary["archived"] == [unused]
    statuses = {m.model_id: m.status for m in registry.list_models()}
    assert statuses[used] == "superseded"
    assert statuses[unused] == "archived"


def test_janitor_never_used_model_still_archives(home, registry):
    """last_used_at None means 'no usage signal' -> age rules decide (#29)."""
    mid, _ = _superseded(registry)
    _make_artifacts(home, registry, mid)
    _age(registry, mid, 40)
    assert registry.get_model(mid).last_used_at is None

    summary = run_janitor(home, days=30, registry=registry)
    assert summary["archived"] == [mid]


def test_record_activation_spares_model_from_janitor(home, registry):
    """End-to-end: record_activation on an aged terminal model keeps it live."""
    mid, _ = _superseded(registry)
    _make_artifacts(home, registry, mid)
    _age(registry, mid, 40)

    assert run_janitor(home, days=30, dry_run=True,
                       registry=registry)["archived"] == [mid]
    assert registry.record_activation(mid) is True
    assert run_janitor(home, days=30, registry=registry)["archived"] == []
    assert registry.get_model(mid).status == "superseded"


def test_janitor_state_file_run_count(home, registry):
    mid, _ = _superseded(registry)
    _make_artifacts(home, registry, mid)
    _age(registry, mid, 40)

    state_path = home / "janitor_state.json"
    assert not state_path.exists()
    run_janitor(home, days=30, registry=registry)
    st = load_state(home)
    assert st["run_count"] == 1 and st["last_run_at"]
    assert set(st["last_summary"]) == {"archived", "freed_bytes"}
    run_janitor(home, days=30, registry=registry)
    assert load_state(home)["run_count"] == 2


def test_janitor_dry_run_touches_nothing(home, registry):
    mid, _ = _superseded(registry)
    adapter, gguf = _make_artifacts(home, registry, mid)
    _age(registry, mid, 40)

    summary = run_janitor(home, days=30, dry_run=True, registry=registry)
    assert summary["archived"] == [mid] and summary["freed_bytes"] > 0
    assert registry.get_model(mid).status == "superseded"
    assert gguf.exists() and adapter.exists()
    assert not (home / "archive").exists()
    assert not (home / "janitor_state.json").exists()


def test_janitor_config_parsing():
    jc = janitor_mod.janitor_config_from_yaml(None)
    assert jc.enabled is False and jc.retention_days == 30  # conservative default
    jc = janitor_mod.janitor_config_from_yaml(
        {"enabled": True, "retention_days": "14"})
    assert jc.enabled is True and jc.retention_days == 14
    jc = janitor_mod.janitor_config_from_yaml({"retention_days": "bogus"})
    assert jc.retention_days == 30


# ------------------------------------------------------------ deploy guard

def test_deploy_rejects_archived_model(home, registry):
    from toaster_loop.serving.deploy import deploy_model
    from toaster_loop.core.config import ToasterLoopConfig

    mid, _ = _superseded(registry)
    _, gguf = _make_artifacts(home, registry, mid)
    registry.set_artifacts(mid, gguf_path=str(gguf))
    registry.archive_model(mid, home=home)

    with pytest.raises(ServeError, match="archived"):
        deploy_model(home, registry, mid, ToasterLoopConfig())


# ------------------------------------------------- auto-check cron hook

def _auto_check(tmp_path, monkeypatch, janitor_cfg):
    """Run `toaster_loop train auto-check` with a config carrying a janitor section."""
    import toaster_loop.registry.janitor as jm

    calls: list = []

    def fake_run_janitor(home, days=30, dry_run=False, registry=None,
                         actor="cli", disk_days=90):
        calls.append({"home": home, "days": days, "actor": actor,
                      "disk_days": disk_days})
        return {"archived": ["vX"], "freed_bytes": 1, "dry_run": False, "errors": []}

    monkeypatch.setattr(jm, "run_janitor", fake_run_janitor)
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text("janitor:\n" + janitor_cfg)
    rc = cli_main(["train", "auto-check", "--config", str(cfg_file)])
    return calls, rc


def test_auto_check_runs_janitor_when_enabled(tmp_path, monkeypatch):
    calls, _ = _auto_check(tmp_path, monkeypatch, "  enabled: true\n  retention_days: 7\n")
    assert len(calls) == 1 and calls[0]["days"] == 7
    # pass4 #31: scheduled archives stamp the ledger as scheduler-driven
    assert calls[0]["actor"] == "scheduler"


def test_auto_check_skips_janitor_when_disabled(tmp_path, monkeypatch):
    calls, _ = _auto_check(tmp_path, monkeypatch, "  enabled: false\n")
    assert calls == []


def test_auto_check_survives_janitor_crash(tmp_path, monkeypatch):
    import toaster_loop.registry.janitor as jm

    def boom(*a, **k):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(jm, "run_janitor", boom)
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text("janitor:\n  enabled: true\n")
    # must not raise out of the cron entrypoint
    cli_main(["train", "auto-check", "--config", str(cfg_file)])


# --------------------------------------------------------------- CLI

def test_cli_archive_restore_pin_flow(home, registry, capsys):
    mid, _ = _superseded(registry)
    _, gguf = _make_artifacts(home, registry, mid)
    registry.set_artifacts(mid, gguf_path=str(gguf))

    assert cli_main(["models", "pin", mid]) == 0
    assert "pinned" in capsys.readouterr().out
    # pinned -> archive refused with non-zero exit
    assert cli_main(["models", "archive", mid]) == 1
    assert "pinned" in capsys.readouterr().err

    assert cli_main(["models", "unpin", mid]) == 0
    assert cli_main(["models", "archive", mid]) == 0
    out = capsys.readouterr().out
    assert "archived v1" in out and "restore" in out
    assert not gguf.exists()

    assert cli_main(["models", "restore", mid]) == 0
    assert gguf.exists()

    # janitor CLI dry-run lists nothing left to do, exits 0
    assert cli_main(["models", "janitor", "--dry-run"]) == 0


def test_cli_models_list_shows_pinned_marker(home, registry, capsys):
    registry.register_model("base-a")
    registry.set_pinned("v1", True)
    assert cli_main(["models", "list"]) == 0
    assert "[pinned]" in capsys.readouterr().out


# ------------------------------------------------------ migration v4 -> v5

def test_migration_v4_to_v5(home):
    db = home / "buffer" / "toaster_loop.db"
    conn = sqlite3.connect(db)
    # Hand-built pre-v5 table: no pinned/pre_archive_paths, old CHECK.
    conn.execute("""
        CREATE TABLE models (
            model_id        TEXT PRIMARY KEY,
            base_model      TEXT NOT NULL,
            parent_adapter  TEXT,
            dataset_id      TEXT,
            manifest_hash   TEXT,
            adapter_dir     TEXT,
            gguf_path       TEXT,
            quant           TEXT,
            status          TEXT NOT NULL CHECK (status IN
              ('candidate','validated','promoted','rolled_back','failed','superseded')),
            created_at      TEXT NOT NULL,
            validation_json TEXT,
            train_loss      REAL,
            eval_loss       REAL,
            promoted_at     TEXT
        )""")
    conn.execute(
        "INSERT INTO models (model_id, base_model, status, created_at, "
        "promoted_at) VALUES ('v1','base-a','superseded',?,?)",
        (_iso(100), _iso(90)),
    )
    conn.commit()
    conn.close()

    reg = ModelRegistry(db)
    try:
        m = reg.get_model("v1")
        assert m is not None and m.pinned is False and m.pre_archive is None
        # CHECK constraint rebuilt: 'archived' now writable end-to-end
        gguf = home / "old.gguf"
        gguf.write_bytes(b"legacy")
        reg.set_artifacts("v1", gguf_path=str(gguf))
        archived = reg.archive_model("v1", home=home)
        assert archived.status == "archived"
        restored = reg.restore_model("v1")
        assert restored.status == "superseded" and gguf.exists()
    finally:
        reg.close()

"""Pass 4 #31 — hash-chained operation ledger: chain math, tamper detection,
mutator instrumentation and the CLI history smoke test.

Chain convention (see registry/ledger.py): entry = all fields but `hash`,
canonicalized with sort_keys + tight separators; hash = sha256(prev_hash +
canonical); seq 1 chains from GENESIS_HASH ("genesis").
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sable.registry import ModelRegistry, recent_ops, record, verify_ledger
from sable.registry.ledger import GENESIS_HASH, normalize_intent


@pytest.fixture()
def registry(tmp_path):
    reg = ModelRegistry(tmp_path / "reg.db")
    yield reg
    reg.close()


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

    from sable.cli import main as cli_main

    db = Path(os.environ["SABLE_HOME"]) / "buffer" / "sable.db"
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
    from sable.cli import main as cli_main

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

    from sable.cli import main as cli_main

    db = Path(os.environ["SABLE_HOME"]) / "buffer" / "sable.db"
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
    from sable.cli import main as cli_main

    assert cli_main(["history", "--intents"]) == 0
    out = capsys.readouterr().out
    assert "no recorded intents" in out
    # honesty: the hint is a hint, not a fabricated row
    assert "INTENT" not in out


def test_dashboard_promote_accepts_optional_intent(tmp_path):
    from sable.core.config import SableConfig
    from sable.dashboard.app import DashboardApp

    home = tmp_path / "dash-home"
    db = home / "buffer" / "sable.db"
    reg = ModelRegistry(db)
    try:
        reg.register_model("base-a")
        reg.set_validation("v1", {}, passed=True)
    finally:
        reg.close()

    app = DashboardApp(SableConfig(), home=home)
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

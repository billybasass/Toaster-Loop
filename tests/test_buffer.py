"""Phase 2 tests — experience buffer (trace store, recorder, feedback CLI).

All hermetic: SABLE_HOME points at a tmp dir; nothing touches ~/.sable.
Run: pytest tests/ -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fake_server import FakeOpenAIServer  # noqa: E402

import sable.cli as cli  # noqa: E402
from sable.core.agent_runner import SableAgentClient  # noqa: E402
from sable.core.config import load_config  # noqa: E402
from sable.store import TraceRecorder, TraceStore, heuristic_score  # noqa: E402
from sable.store.db import SCHEMA_VERSION  # noqa: E402
from sable.store.paths import resolve_sable_home  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sable.core.config import SableConfig


@pytest.fixture()
def sable_home(tmp_path, monkeypatch):
    """Redirect the state root into a tmp dir for every test in this module."""
    home = tmp_path / "uf-home"
    monkeypatch.setenv("SABLE_HOME", str(home))
    monkeypatch.delenv("SABLE_CONFIG", raising=False)
    return home


@pytest.fixture(scope="module")
def fake_server():
    with FakeOpenAIServer() as srv:
        yield srv


# ------------------------------------------------------------------ paths/config

def test_home_resolution_precedence(tmp_path, monkeypatch):
    monkeypatch.delenv("SABLE_HOME", raising=False)

    class Cfg:
        home = str(tmp_path / "from-config")

    cfg = cast("SableConfig", Cfg())
    assert resolve_sable_home(cfg) == tmp_path / "from-config"
    monkeypatch.setenv("SABLE_HOME", str(tmp_path / "from-env"))
    assert resolve_sable_home(cfg) == tmp_path / "from-env"  # env wins
    assert resolve_sable_home(None) == tmp_path / "from-env"


def test_config_recording_yaml_and_env(tmp_path, monkeypatch):
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text("recording:\n  enabled: false\n")
    monkeypatch.delenv("SABLE_RECORDING", raising=False)
    assert load_config(str(cfg_file)).recording_enabled is False

    monkeypatch.setenv("SABLE_RECORDING", "true")  # env wins over file
    assert load_config(str(cfg_file)).recording_enabled is True

    cfg_file2 = tmp_path / "cfg2.yaml"
    cfg_file2.write_text("home: ~/somewhere\n")
    monkeypatch.delenv("SABLE_HOME", raising=False)  # exercise the file value
    assert load_config(str(cfg_file2)).home == "~/somewhere"


# -------------------------------------------------------------------- recorder

def test_fallback_run_writes_jsonl_and_index(sable_home, fake_server):
    """Full task via fallback client => JSONL trace + SQLite row under SABLE_HOME."""
    rec = TraceRecorder(home=sable_home)
    client = SableAgentClient(
        base_url=fake_server.base_url, model="fake-model", recorder=rec
    )
    result = client.run("read the fake server source")
    assert result.completed is True
    assert result.trace_id, "run must return the buffer trace id"

    # JSONL under buffer/YYYY/MM/<trace_id>.jsonl
    matches = list((sable_home / "buffer").rglob(f"{result.trace_id}.jsonl"))
    assert len(matches) == 1
    lines = [json.loads(l) for l in matches[0].read_text().splitlines()]
    kinds = [l["kind"] for l in lines]
    assert kinds[0] == "trace_header"
    assert kinds[-1] == "trace_footer"
    assert kinds.count("message") == len(result.messages)
    header = lines[0]
    assert header["schema_version"] == SCHEMA_VERSION
    assert header["model_alias"] == "fake-model"
    # backlog #2/#7: failure_reason is nullable and None on clean runs
    assert header.get("failure_reason") is None
    assert lines[-1].get("failure_reason") is None
    assert header["task"] == "read the fake server source"
    footer = lines[-1]
    assert footer["status"] == "success"
    assert footer["num_tool_calls"] == 1

    # SQLite index row
    store = TraceStore(sable_home / "buffer" / "sable.db")
    try:
        row = store.get_trace(result.trace_id)
        assert row is not None
        assert row.status == "success"
        assert row.num_tool_calls == 1
        assert row.outcome_score == pytest.approx(0.5)  # clean completion
        assert row.tokens_in == 2 and row.tokens_out == 2  # from fake usage
        assert Path(row.jsonl_path).is_file()
        fb = store.get_feedback(result.trace_id)
        assert any(f["source"] == "heuristic" for f in fb)
    finally:
        store.close()


def test_recording_failure_does_not_break_run(sable_home, fake_server, monkeypatch):
    """ROADMAP acceptance: a trace write failure cannot break a conversation."""
    rec = TraceRecorder(home=sable_home / "\0invalid")  # unrepresentable path
    client = SableAgentClient(
        base_url=fake_server.base_url, model="fake-model", recorder=rec
    )
    result = client.run("read the fake server source")
    assert result.completed is True
    assert "Done." in result.final_response
    assert result.trace_id is None


def test_recording_disabled_writes_nothing(sable_home, fake_server):
    from sable.store.recorder import NullRecorder, make_recorder

    cfg = load_config(None)
    object.__setattr__(cfg, "recording_enabled", False)
    rec = make_recorder(cfg)
    assert isinstance(rec, NullRecorder)

    client = SableAgentClient(
        base_url=fake_server.base_url, model="fake-model", recorder=rec
    )
    result = client.run("read the fake server source")
    assert result.completed is True
    assert result.trace_id is None
    assert not (sable_home / "buffer").exists()


# ---------------------------------------------------------------- v2 migration

def test_db_v1_to_v2_migration(tmp_path):
    """Pre-failure_reason DBs migrate in place and stay readable (backlog #7)."""
    import sqlite3

    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.executescript("""
CREATE TABLE traces (
    trace_id TEXT PRIMARY KEY, session_id TEXT, timestamp TEXT NOT NULL,
    agent_profile TEXT, model_alias TEXT, model_version TEXT, endpoint_label TEXT,
    task_summary TEXT, status TEXT NOT NULL, outcome_score REAL,
    num_tool_calls INTEGER NOT NULL DEFAULT 0, tokens_in INTEGER, tokens_out INTEGER,
    duration_s REAL, jsonl_path TEXT NOT NULL, schema_version INTEGER NOT NULL DEFAULT 1,
    curated INTEGER NOT NULL DEFAULT 0, dataset_id TEXT);
PRAGMA schema_version = 1;
INSERT INTO traces (trace_id,timestamp,status,num_tool_calls,jsonl_path)
VALUES ('old1','2026-01-01T00:00:00+00:00','success',0,'/dev/null');
""")
    con.commit()
    con.close()

    from sable.store.db import TraceRow as _TR
    store = TraceStore(db)
    try:
        old = store.get_trace("old1")           # tolerant read of v1 row
        assert old is not None and old.failure_reason is None
        store.insert_trace(_TR(
            trace_id="new1", session_id=None, timestamp="t", agent_profile=None,
            model_alias=None, model_version=None, endpoint_label=None,
            task_summary=None, status="failure", outcome_score=-1.0,
            num_tool_calls=0, tokens_in=None, tokens_out=None, duration_s=None,
            jsonl_path="/dev/null", failure_reason="backend_error:connection"))
        new = store.get_trace("new1")
        assert new is not None
        assert new.failure_reason == "backend_error:connection"
    finally:
        store.close()
    # Reopen: migration must be idempotent (no duplicate ALTER crash).
    store = TraceStore(db)
    try:
        old2 = store.get_trace("old1")
        new2 = store.get_trace("new1")
        assert old2 is not None and old2.failure_reason is None
        assert new2 is not None
        assert new2.failure_reason == "backend_error:connection"
    finally:
        store.close()


# ------------------------------------------------------ duplicate insert (#15)

def test_insert_trace_duplicate_skipped_preserves_curation_state(tmp_path):
    """ON CONFLICT(trace_id) DO NOTHING: a colliding insert must never clobber
    load-bearing curation state (curated/dataset_id/outcome_score). Returns
    False on skip, True on insert."""
    from sable.store.db import TraceRow as _TR

    def row(summary):
        return _TR(
            trace_id="dup1", session_id=None, timestamp="2026-09-01T00:00:00+00:00",
            agent_profile="p", model_alias=None, model_version=None,
            endpoint_label=None, task_summary=summary, status="success",
            outcome_score=0.5, num_tool_calls=0, tokens_in=None, tokens_out=None,
            duration_s=1.0, jsonl_path="/dev/null", schema_version=SCHEMA_VERSION,
            curated=False, dataset_id=None)

    store = TraceStore(tmp_path / "dup.db")
    try:
        assert store.insert_trace(row("original")) is True
        # simulate the curater having claimed + a feedback score having landed
        assert store.claim_traces(["dup1"], "ds-7") == 1
        store.add_feedback("dup1", -0.25, "user says bad", source="user")

        assert store.insert_trace(row("collision")) is False  # skip signal
        kept = store.get_trace("dup1")
        assert kept is not None
        assert kept.task_summary == "original"   # row untouched
        assert kept.curated is True              # curation state preserved
        assert kept.dataset_id == "ds-7"
        assert kept.outcome_score == pytest.approx(-0.25)
    finally:
        store.close()


# ------------------------------------------------- failure remediation notes
# Pass 4 #33 — idea adapted from PIASystem (MIT) app/core/failure_taxonomy.py.

def test_failure_notes_set_get_and_fallback(tmp_path):
    store = TraceStore(tmp_path / "notes.db")
    try:
        # implicit open: no row -> None everywhere, empty listing.
        assert store.get_failure_note("timeout") is None
        assert store.get_failure_note("timeout", profile="agentx") is None
        assert store.list_failure_notes() == []

        store.set_failure_note("timeout", "in_progress", note="retry wiring")
        n = store.get_failure_note("timeout")
        assert n is not None
        assert n["status"] == "in_progress" and n["profile"] is None
        assert n["note"] == "retry wiring" and n["ts"]

        # profile-specific lookup falls back to the global row...
        fb = store.get_failure_note("timeout", profile="agentx")
        assert fb is not None and fb["status"] == "in_progress"
        # ...until a profile-scoped row exists, which then wins.
        store.set_failure_note("timeout", "resolved", note="fixed",
                               profile="agentx")
        scoped = store.get_failure_note("timeout", profile="agentx")
        assert scoped is not None and scoped["status"] == "resolved"
        glob = store.get_failure_note("timeout")
        assert glob is not None and glob["status"] == "in_progress"
        assert len(store.list_failure_notes()) == 2

        # upsert: one row per (reason, scope), status replaced in place.
        store.set_failure_note("timeout", "acknowledged", note="seen",
                               profile="agentx")
        rows = [r for r in store.list_failure_notes() if r["profile"] == "agentx"]
        assert len(rows) == 1 and rows[0]["status"] == "acknowledged"

        # reset-to-open with an empty note deletes the row (back to implicit
        # open / untracked); open-with-note stays tracked.
        store.set_failure_note("timeout", "open")
        assert store.get_failure_note("timeout") is None
        kept = store.get_failure_note("timeout", profile="agentx")
        assert kept is not None and kept["status"] == "acknowledged"
        store.set_failure_note("timeout", "open", note="wontfix")
        n = store.get_failure_note("timeout")
        assert n is not None
        assert n["status"] == "open" and n["note"] == "wontfix"

        # status validation against the fixed set.
        with pytest.raises(ValueError):
            store.set_failure_note("timeout", "bogus")
    finally:
        store.close()


def test_failure_notes_survive_reopen(tmp_path):
    db = tmp_path / "notes2.db"
    store = TraceStore(db)
    try:
        store.set_failure_note("context_overflow", "resolved", note="cap raised")
    finally:
        store.close()
    store = TraceStore(db)  # idempotent CREATE TABLE IF NOT EXISTS migration
    try:
        n = store.get_failure_note("context_overflow")
        assert n is not None and n["status"] == "resolved"
    finally:
        store.close()


# ------------------------------------------------------------------- heuristic

def test_heuristic_score_rules():
    clean = [{"role": "user", "content": "hi"},
             {"role": "assistant", "content": "ok"}]
    assert heuristic_score(clean, completed=True) == 0.5

    err_tools = [
        {"role": "tool", "content": "error: bad arguments for read_file"},
        {"role": "tool", "content": "Traceback (most recent call last): ..."},
    ]
    assert heuristic_score(err_tools, completed=True) == pytest.approx(0.0)

    # error penalties cap at -0.75
    many_errs = [{"role": "tool", "content": f"error {i}"} for i in range(10)]
    assert heuristic_score(many_errs, completed=True) == pytest.approx(-0.25)

    assert heuristic_score([], completed=False) == -0.5


# ------------------------------------------------- HIGH #4 cancelled neutrality

def _seed_row(store, tid, *, status="failure", score=None, failure_reason=None):
    from datetime import datetime, timedelta, timezone

    from sable.store.db import TraceRow as _TR
    p = store.db_path.parent / f"{tid}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("")
    # Relative-to-now stamp: consumers include stats_daily's REAL calendar
    # cutoff (test_stats_and_daily_means_exclude_cancelled), so a frozen
    # literal here is a time-bomb. days_ago=1 keeps rows inside any window.
    ts = (datetime.now(timezone.utc) - timedelta(days=1)
          ).isoformat(timespec="seconds")
    store.insert_trace(_TR(
        trace_id=tid, session_id=None, timestamp=ts,
        agent_profile="fallback", model_alias="m", model_version=None,
        endpoint_label=None, task_summary="s", status=status,
        outcome_score=score, num_tool_calls=0, tokens_in=None,
        tokens_out=None, duration_s=None, jsonl_path=str(p),
        failure_reason=failure_reason,
    ))


def test_cancelled_run_recorded_score_neutral(sable_home):
    """Recorder NULLs outcome_score for cancelled runs (HIGH #4)."""
    rec = TraceRecorder(home=sable_home)
    row = rec.record(task="t", messages=[{"role": "user", "content": "hi"}],
                     backend="b", model_alias="m", base_url="http://x",
                     completed=False, failure_reason="cancelled")
    assert row is not None
    assert row.outcome_score is None
    # status derivation unchanged (non-completed -> failure bucket)
    assert row.status == "failure"
    # footer carries the neutral score too
    footer = json.loads(Path(row.jsonl_path).read_text().splitlines()[-1])
    assert footer["outcome_score"] is None
    assert footer["failure_reason"] == "cancelled"
    rec.close()

    # composed runner form behaves identically
    rec2 = TraceRecorder(home=sable_home)
    row2 = rec2.record(task="t", messages=[], backend="b", model_alias="m",
                       base_url="http://x", completed=False,
                       failure_reason="backend_error:cancelled")
    assert row2 is not None and row2.outcome_score is None
    rec2.close()


def test_non_cancelled_scoring_unchanged(sable_home):
    """Non-cancelled runs keep the exact pre-fix heuristic scores."""
    rec = TraceRecorder(home=sable_home)
    done = rec.record(task="t", messages=[{"role": "user", "content": "hi"}],
                      backend="b", model_alias="m", base_url="http://x",
                      completed=True)
    assert done is not None and done.outcome_score == pytest.approx(0.5)
    failed = rec.record(task="t", messages=[], backend="b", model_alias="m",
                        base_url="http://x", completed=False,
                        failure_reason="max_iterations")
    assert failed is not None and failed.outcome_score == pytest.approx(-0.5)
    rec.close()


def test_legacy_cancelled_rows_neutralized_on_open(tmp_path):
    """Migration: pre-fix rows stamped -0.5 + cancelled NULL out on reopen."""
    db = tmp_path / "legacy.db"
    store = TraceStore(db)
    _seed_row(store, "c1", score=-0.5, failure_reason="cancelled")
    _seed_row(store, "c2", score=-0.5, failure_reason="backend_error:cancelled")
    _seed_row(store, "m1", score=-0.5, failure_reason="max_iterations")
    _seed_row(store, "ok", status="success", score=0.5)
    store.close()

    store = TraceStore(db)  # idempotent migration runs on open
    try:
        c1 = store.get_trace("c1")
        c2 = store.get_trace("c2")
        m1 = store.get_trace("m1")
        ok = store.get_trace("ok")
        assert c1 is not None and c2 is not None
        assert m1 is not None and ok is not None
        assert c1.outcome_score is None
        assert c2.outcome_score is None
        # non-cancelled rows untouched
        assert m1.outcome_score == pytest.approx(-0.5)
        assert ok.outcome_score == pytest.approx(0.5)
    finally:
        store.close()


def test_stats_and_daily_means_exclude_cancelled(tmp_path):
    """stats()/stats_daily means skip NULL-scored cancelled rows (never zero)."""
    db = tmp_path / "means.db"
    store = TraceStore(db)
    _seed_row(store, "ok1", status="success", score=0.5)
    _seed_row(store, "ok2", status="success", score=0.5)
    _seed_row(store, "cx", score=None, failure_reason="cancelled")
    try:
        st = store.stats()
        assert st["scored_traces"] == 2
        assert st["avg_outcome_score"] == pytest.approx(0.5)
        daily = {d["date"]: d for d in store.stats_daily(days=3650)}
        cx = store.get_trace("cx")
        assert cx is not None
        today = daily[cx.timestamp[:10]]
        assert today["traces"] == 3  # cancelled row still counted as a trace
        assert today["avg_score"] == pytest.approx(0.5)  # but not in the mean
    finally:
        store.close()


# ------------------------------------------------------------------- feedback CLI

def _record_one(sable_home, fake_server) -> str:
    rec = TraceRecorder(home=sable_home)
    client = SableAgentClient(
        base_url=fake_server.base_url, model="fake-model", recorder=rec
    )
    result = client.run("read the fake server source")
    assert result.trace_id is not None
    return result.trace_id


def test_feedback_cli_by_id_and_last(sable_home, fake_server, capsys):
    trace_id = _record_one(sable_home, fake_server)

    assert cli.main(["feedback", trace_id, "--score", "0.8", "--note", "nice"]) == 0
    store = TraceStore(sable_home / "buffer" / "sable.db")
    try:
        fb = store.get_feedback(trace_id)
        user_rows = [f for f in fb if f["source"] == "user"]
        assert len(user_rows) == 1
        assert user_rows[0]["score"] == pytest.approx(0.8)
        assert user_rows[0]["comment"] == "nice"
        # trace row outcome_score synced with explicit feedback
        row = store.get_trace(trace_id)
        assert row is not None
        assert row.outcome_score == pytest.approx(0.8)
    finally:
        store.close()

    # --last targets most recent; --fail shorthand => -1
    assert cli.main(["feedback", "--last", "--fail"]) == 0
    store = TraceStore(sable_home / "buffer" / "sable.db")
    try:
        row = store.get_trace(trace_id)
        assert row is not None
        assert row.outcome_score == pytest.approx(-1.0)
        latest = store.latest_trace()
        assert latest is not None
        assert latest.trace_id == trace_id
    finally:
        store.close()

    # --ok shorthand => +1
    assert cli.main(["feedback", "--last", "--ok"]) == 0
    store = TraceStore(sable_home / "buffer" / "sable.db")
    try:
        row = store.get_trace(trace_id)
        assert row is not None
        assert row.outcome_score == pytest.approx(1.0)
    finally:
        store.close()


def test_feedback_cli_validation(sable_home, fake_server, capsys):
    trace_id = _record_one(sable_home, fake_server)
    assert cli.main(["feedback", trace_id, "--score", "5"]) != 0   # out of range
    assert cli.main(["feedback", trace_id]) != 0                    # nothing to record
    assert cli.main(["feedback", "deadbeefcafe"]) != 0              # unknown id


# ------------------------------------------------------------------- traces CLI

def test_traces_list_show_stats(sable_home, fake_server, capsys):
    trace_id = _record_one(sable_home, fake_server)

    assert cli.main(["traces", "list"]) == 0
    out = capsys.readouterr().out
    assert trace_id in out
    assert "success" in out

    assert cli.main(["traces", "list", "--status", "failure"]) == 0
    out = capsys.readouterr().out
    assert "no traces recorded yet" in out

    assert cli.main(["traces", "show", trace_id]) == 0
    out = capsys.readouterr().out
    assert trace_id in out
    assert "transcript:" in out
    assert "assistant" in out and "tool" in out

    assert cli.main(["traces", "stats"]) == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["total_traces"] == 1
    assert stats["by_status"] == {"success": 1}
    assert stats["tool_calls_total"] == 1

    assert cli.main(["traces", "show", "nonexistent"]) == 1


# ------------------------------------------------------- built-in capture path

def test_builtin_capture_equivalent_jsonl(sable_home, fake_server):
    """Built-in runtime: post-run capture produces a complete trace."""
    rec = TraceRecorder(home=sable_home)
    runner = SableAgentClient(
        base_url=fake_server.base_url, model="fake-model", recorder=rec
    )

    result = runner.run("read the fake server file")
    assert result.completed is True
    assert result.trace_id, "built-in runtime must record a trace"

    matches = list((sable_home / "buffer").rglob(f"{result.trace_id}.jsonl"))
    assert len(matches) == 1
    lines = [json.loads(l) for l in matches[0].read_text().splitlines()]
    kinds = [l["kind"] for l in lines]
    assert kinds[0] == "trace_header" and kinds[-1] == "trace_footer"
    # Equivalent capture: one JSONL message line per captured conversation message.
    assert kinds.count("message") == len(result.messages)
    header = lines[0]
    assert header["agent_profile"] == "sable"

    store = TraceStore(sable_home / "buffer" / "sable.db")
    try:
        row = store.get_trace(result.trace_id)
        assert row is not None
        assert row.agent_profile == "sable"
        assert row.num_tool_calls >= 1  # fake server forces one tool call
    finally:
        store.close()


def test_stats_daily_respects_calendar_window(tmp_path):
    """stats_daily(days=N) must exclude days older than N real days ago."""
    from datetime import datetime, timedelta, timezone

    from sable.store.db import TraceRow

    store = TraceStore(tmp_path / "idx.db")
    try:
        def row(tid, ts):
            return TraceRow(trace_id=tid, session_id=None, timestamp=ts,
                            agent_profile="p", model_alias="m",
                            model_version=None, endpoint_label=None,
                            task_summary=tid, status="success",
                            outcome_score=0.5, num_tool_calls=0,
                            tokens_in=None, tokens_out=None, duration_s=None,
                            jsonl_path="/dev/null")

        now = datetime.now(timezone.utc)
        store.insert_trace(row("recent", (now - timedelta(days=1))
                               .isoformat(timespec="seconds")))
        store.insert_trace(row("ancient", (now - timedelta(days=90))
                               .isoformat(timespec="seconds")))
        days = {d["date"] for d in store.stats_daily(days=30)}
        recent_day = (now - timedelta(days=1)).strftime("%Y-%m-%d")
        ancient_day = (now - timedelta(days=90)).strftime("%Y-%m-%d")
        assert recent_day in days
        assert ancient_day not in days
    finally:
        store.close()

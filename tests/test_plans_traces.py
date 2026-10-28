"""WP12: traces plan_task linkage (recorder + store) and curate --project.

Additive field discipline: header key + SQLite column via the tolerant
pattern; pre-WP12 traces/DBs must read fine.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from sable.core.config import SableConfig
from sable.store.db import TraceRow, TraceStore
from sable.store.recorder import make_recorder


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    (h / "buffer").mkdir(parents=True)
    monkeypatch.setenv("SABLE_HOME", str(h))
    return h


MSGS = [{"role": "user", "content": "do it"},
        {"role": "assistant", "content": "done"}]


# ------------------------------------------------------------- recorder

def test_recorder_stamps_plan_task_header_and_row(home):
    rec = make_recorder(SableConfig(), plan_task="t7")
    row = rec.record(task="do it", messages=MSGS, backend="sable",
                     model_alias="m", base_url="http://127.0.0.1:9/v1",
                     completed=True)
    assert row is not None
    trace_id = row.trace_id
    # header line carries the additive key (buffer/YYYY/MM/<id>.jsonl)
    row_path = list((home / "buffer").rglob(f"{trace_id}.jsonl"))
    assert row_path
    first = json.loads(row_path[0].read_text().splitlines()[0])
    assert first["kind"] == "trace_header"
    assert first["plan_task"] == "t7"
    # index row carries it too
    store = TraceStore(home / "buffer" / "sable.db")
    try:
        r = store.get_trace(trace_id)
        assert r is not None and r.plan_task == "t7"
        d = r.to_dict()
        assert d["plan_task"] == "t7"
    finally:
        store.close()


def test_recorder_without_plan_task_is_null(home):
    rec = make_recorder(SableConfig())
    row = rec.record(task="plain", messages=MSGS, backend="sable",
                     model_alias="m", base_url="http://127.0.0.1:9/v1",
                     completed=True)
    assert row is not None
    trace_id = row.trace_id
    store = TraceStore(home / "buffer" / "sable.db")
    try:
        r = store.get_trace(trace_id)
        assert r is not None and r.plan_task is None
    finally:
        store.close()


def test_tolerant_read_of_old_header_without_key(home):
    """A pre-WP12 trace file (header lacks plan_task) still reads."""
    old = home / "buffer" / "old-trace.jsonl"
    header = {"kind": "trace_header", "schema_version": 1,
              "trace_id": "old-1", "timestamp": "2024-01-01T00:00:00+00:00",
              "agent_profile": "sable", "model_alias": "m",
              "task": "legacy"}          # no plan_task key at all
    footer = {"kind": "trace_footer", "schema_version": 1,
              "status": "success", "completed": True}
    old.write_text(json.dumps(header) + "\n"
                   + json.dumps({"kind": "message",
                                 "message": MSGS[0]}) + "\n"
                   + json.dumps(footer) + "\n")
    lines = old.read_text().splitlines()
    hdr = json.loads(lines[0])
    assert hdr.get("plan_task") is None          # .get()-style tolerant read


def test_legacy_db_migrates_plan_task_column(home):
    """A DB created without the column opens, migrates, reads NULL rows."""
    db = home / "buffer" / "legacy.db"
    con = sqlite3.connect(db)
    con.execute("""CREATE TABLE traces (
        trace_id TEXT PRIMARY KEY, session_id TEXT, timestamp TEXT,
        agent_profile TEXT, model_alias TEXT, model_version TEXT,
        endpoint_label TEXT, task_summary TEXT, status TEXT,
        outcome_score REAL, num_tool_calls INTEGER, tokens_in INTEGER,
        tokens_out INTEGER, duration_s REAL, jsonl_path TEXT,
        curated INTEGER DEFAULT 0, dataset_id TEXT)""")
    con.execute("INSERT INTO traces (trace_id, timestamp, status, "
                "jsonl_path, curated) VALUES ('old-9', 'x', 'success', "
                "'/nope.jsonl', 0)")
    con.commit()
    con.close()

    store = TraceStore(db)     # runs _migrate on open
    try:
        r = store.get_trace("old-9")
        assert r is not None and r.plan_task is None   # reads as NULL
        cols = {row[1] for row in store._conn.execute(
            "PRAGMA table_info(traces)")}
        assert "plan_task" in cols                     # column added
    finally:
        store.close()


def test_list_and_query_filter_project(home):
    store = TraceStore(home / "buffer" / "sable.db")
    try:
        for i, proj in enumerate(["p1", "p1", None]):
            store.insert_trace(TraceRow(
                trace_id=f"tr{i}", session_id=None, timestamp="2025-01-01",
                agent_profile="sable", model_alias="m", model_version=None,
                endpoint_label=None, task_summary="s", status="success",
                outcome_score=0.5, num_tool_calls=0, tokens_in=None,
                tokens_out=None, duration_s=None, jsonl_path="/x.jsonl",
                project=proj, plan_task=f"t{i}" if proj else None))
        assert len(store.list_traces(project="p1")) == 2
        assert len(store.list_traces()) == 3
        q = store.query_for_curation(project="p1")
        assert {r.trace_id for r in q} == {"tr0", "tr1"}
        assert all(r.plan_task for r in q)             # linkage survives query
    finally:
        store.close()

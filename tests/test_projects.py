"""Phase 1 (projects) tests — project CRUD, thread filing, trace propagation.

Covers:
  * /api/projects CRUD round-trip incl. validation errors (empty name,
    relative workspace_dir, missing directory).
  * Threads gain an optional nullable "project" field; legacy chat/*.json
    files without the key keep listing (backward compatibility, no migration).
  * Deleting a project leaves thread files intact and renders them Unfiled.
  * Traces record the active project id and list/count filter by ?project=.
  * resolve_workspace_roots precedence: profile > project > config > cwd.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sable.core.config import SableConfig  # noqa: E402
from sable.dashboard.app import DashboardApp  # noqa: E402


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "uf-proj"
    monkeypatch.setenv("SABLE_HOME", str(h))
    return h


@pytest.fixture()
def app(home):
    return DashboardApp(SableConfig(), home=home)


# ------------------------------------------------------------- project CRUD

def test_project_crud_roundtrip(app, home, tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()

    # create
    r = app.handle("POST", "/api/projects",
                   body={"name": "alpha", "workspace_dir": str(ws),
                         "agents": ["coder", "reviewer"]})
    assert r.status == 201, r.body
    p = r.body["project"]
    pid = p["id"]
    assert p["name"] == "alpha" and p["workspace_dir"] == str(ws)
    assert p["agents"] == ["coder", "reviewer"]
    assert p["created"] and p["updated"]
    # one JSON file per project under <home>/projects/
    assert (home / "projects" / f"{pid}.json").is_file()

    # list
    r = app.handle("GET", "/api/projects")
    assert r.status == 200
    assert [x["id"] for x in r.body["projects"]] == [pid]

    # update: bump updated, change name/agents/workspace_dir
    ws2 = tmp_path / "ws2"
    ws2.mkdir()
    r = app.handle("PUT", f"/api/projects/{pid}",
                   body={"name": "beta", "workspace_dir": str(ws2),
                         "agents": ["coder"]})
    assert r.status == 200
    assert r.body["project"]["name"] == "beta"
    assert r.body["project"]["workspace_dir"] == str(ws2)
    assert r.body["project"]["agents"] == ["coder"]

    # delete
    r = app.handle("DELETE", f"/api/projects/{pid}")
    assert r.status == 200 and r.body["ok"]
    assert app.handle("GET", "/api/projects").body["projects"] == []
    assert app.handle("PUT", f"/api/projects/{pid}",
                      body={"name": "x"}).status == 404
    assert app.handle("DELETE", f"/api/projects/{pid}").status == 404


def test_project_validation_errors(app, tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()

    r = app.handle("POST", "/api/projects",
                   body={"name": "", "workspace_dir": str(ws)})
    assert r.status == 400 and r.body["error"]["code"] == "bad_request"

    r = app.handle("POST", "/api/projects",
                   body={"name": "  ", "workspace_dir": str(ws)})
    assert r.status == 400

    # relative path rejected
    r = app.handle("POST", "/api/projects",
                   body={"name": "rel", "workspace_dir": "relative/dir"})
    assert r.status == 400
    assert "absolute" in r.body["error"]["message"]

    # absolute but missing dir rejected
    r = app.handle("POST", "/api/projects",
                   body={"name": "gone",
                         "workspace_dir": str(tmp_path / "nope")})
    assert r.status == 400
    assert "does not exist" in r.body["error"]["message"]

    # agents must be a list of strings
    r = app.handle("POST", "/api/projects",
                   body={"name": "ok", "workspace_dir": str(ws),
                         "agents": [{"bad": 1}]})
    assert r.status == 400


# ------------------------------------------------------- threads <-> project

def test_thread_with_project_lists_under_it(app):
    ws = Path("/tmp")  # exists, absolute
    pid = app.handle("POST", "/api/projects",
                     body={"name": "p1", "workspace_dir": str(ws)}
                     ).body["project"]["id"]

    r = app.handle("PUT", "/api/chat/threads/t-proj",
                   body={"title": "filed", "messages": [], "project": pid})
    assert r.status == 200
    assert r.body["thread"]["project"] == pid

    lst = app.handle("GET", "/api/chat/threads").body["threads"]
    row = next(t for t in lst if t["id"] == "t-proj")
    assert row["project"] == pid

    # move between projects via update; explicit null -> Unfiled
    pid2 = app.handle("POST", "/api/projects",
                      body={"name": "p2", "workspace_dir": str(ws)}
                      ).body["project"]["id"]
    r = app.handle("PUT", "/api/chat/threads/t-proj",
                   body={"project": pid2})
    assert r.body["thread"]["project"] == pid2
    r = app.handle("PUT", "/api/chat/threads/t-proj",
                   body={"project": None})
    assert r.body["thread"]["project"] is None

    # unknown project id rejected on thread write
    assert app.handle("PUT", "/api/chat/threads/t-proj",
                      body={"project": "does-not-exist"}).status == 400


def test_legacy_thread_without_project_key_still_lists(app, home):
    """Raw pre-projects chat file (no "project" key) lists fine = Unfiled."""
    d = home / "chat"
    d.mkdir(parents=True, exist_ok=True)
    legacy = {"id": "old1", "title": "legacy", "created": "2026-01-01T00:00:00+00:00",
              "updated": "2026-01-01T00:00:00+00:00", "model_alias": "m",
              "messages": [{"role": "user", "content": "hi"}],
              "trace_ids": ["t9"], "summary": None}
    (d / "old1.json").write_text(json.dumps(legacy))

    lst = app.handle("GET", "/api/chat/threads").body["threads"]
    assert [t["id"] for t in lst] == ["old1"]
    assert lst[0]["project"] is None          # absent key -> Unfiled
    assert lst[0]["message_count"] == 1

    # full detail read still works, and a re-save keeps it projectless
    det = app.handle("GET", "/api/chat/threads/old1")
    assert det.status == 200 and det.body["thread"]["title"] == "legacy"
    r = app.handle("PUT", "/api/chat/threads/old1", body={"title": "renamed"})
    assert r.status == 200 and r.body["thread"]["project"] is None


def test_delete_project_leaves_threads_unfiled(app):
    ws = Path("/tmp")
    pid = app.handle("POST", "/api/projects",
                     body={"name": "doomed", "workspace_dir": str(ws)}
                     ).body["project"]["id"]
    app.handle("PUT", "/api/chat/threads/keep-me",
               body={"title": "kept", "messages": [], "project": pid})

    assert app.handle("DELETE", f"/api/projects/{pid}").status == 200

    # thread file untouched; its dangling id reads as Unfiled in the tree
    lst = app.handle("GET", "/api/chat/threads").body["threads"]
    row = next(t for t in lst if t["id"] == "keep-me")
    assert row["project"] == pid  # raw value kept; UI groups unknown ids here
    known = {p["id"] for p in app.handle("GET", "/api/projects").body["projects"]}
    assert row["project"] not in known  # => renders under Unfiled


# ------------------------------------------------- trace project propagation

def test_trace_recording_carries_project_and_filters(app, home):
    from sable.store.recorder import TraceRecorder

    ws = Path("/tmp")
    pid = app.handle("POST", "/api/projects",
                     body={"name": "trace-proj", "workspace_dir": str(ws)}
                     ).body["project"]["id"]

    rec = TraceRecorder(home=home, project=pid)
    row = rec.record(task="do the thing", messages=[
        {"role": "user", "content": "do the thing"},
        {"role": "assistant", "content": "done"}],
        backend="sable", model_alias="m", base_url="http://x/v1",
        completed=True)
    assert row is not None and row.project == pid
    rec2 = TraceRecorder(home=home)  # no project => unfiled trace
    row2 = rec2.record(task="other task", messages=[
        {"role": "user", "content": "other"},
        {"role": "assistant", "content": "ok"}],
        backend="sable", model_alias="m", base_url="http://x/v1",
        completed=True)
    assert row2 is not None and row2.project is None
    rec.close(); rec2.close()

    # list + filter
    r = app.handle("GET", "/api/traces", query={"project": pid})
    assert [t["trace_id"] for t in r.body["traces"]] == [row.trace_id]
    assert r.body["total"] == 1

    r = app.handle("GET", "/api/traces")
    ids = {t["trace_id"] for t in r.body["traces"]}
    assert ids == {row.trace_id, row2.trace_id}

    # detail exposes project too
    det = app.handle("GET", f"/api/traces/{row.trace_id}")
    assert det.body["trace"]["project"] == pid


def test_trace_project_survives_legacy_db(tmp_path, monkeypatch):
    """A v2 DB (no project column) migrates on open and reads old rows."""
    import sqlite3
    from sable.store.db import TraceStore

    monkeypatch.setenv("SABLE_HOME", str(tmp_path / "h"))
    db = tmp_path / "buffer" / "sable.db"
    db.parent.mkdir(parents=True)
    # hand-build a minimal v2-shaped traces table (no project column)
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE traces (trace_id TEXT PRIMARY KEY, session_id TEXT,"
        " timestamp TEXT NOT NULL, agent_profile TEXT, model_alias TEXT,"
        " model_version TEXT, endpoint_label TEXT, task_summary TEXT,"
        " status TEXT NOT NULL, outcome_score REAL, num_tool_calls INTEGER"
        " NOT NULL DEFAULT 0, tokens_in INTEGER, tokens_out INTEGER,"
        " duration_s REAL, jsonl_path TEXT NOT NULL, schema_version INTEGER"
        " NOT NULL DEFAULT 2, curated INTEGER NOT NULL DEFAULT 0,"
        " dataset_id TEXT, failure_reason TEXT)")
    conn.execute(
        "INSERT INTO traces (trace_id, timestamp, status, jsonl_path)"
        " VALUES ('old', '2026-01-01T00:00:00+00:00', 'success', '/x.jsonl')")
    conn.commit()
    conn.close()

    store = TraceStore(db)  # runs the guarded ALTER TABLE ADD COLUMN
    try:
        row = store.get_trace("old")
        assert row is not None and row.project is None
        cols = {r[1] for r in store._conn.execute(
            "PRAGMA table_info(traces)").fetchall()}
        assert "project" in cols
    finally:
        store.close()


# ------------------------------------------------ workspace roots precedence

def test_resolve_workspace_roots_precedence(tmp_path, monkeypatch):
    import sable.core.agent_runner as ar
    from sable.core.agent_runner import resolve_workspace_roots

    fake_cwd = tmp_path / "cwd"
    fake_cwd.mkdir()
    monkeypatch.setattr(ar.os, "getcwd", lambda: str(fake_cwd))

    proj_ws = tmp_path / "proj-ws"
    proj_ws.mkdir()
    prof_ws = tmp_path / "prof-ws"
    prof_ws.mkdir()
    cfg_ws = tmp_path / "cfg-ws"
    cfg_ws.mkdir()

    # config level only -> config wins over cwd
    cfg = SableConfig(agent={"workspace_root": str(cfg_ws)})
    roots = resolve_workspace_roots(cfg)
    assert roots is not None and roots[0] == cfg_ws

    # project beats config
    roots = resolve_workspace_roots(cfg, project_workspace_dir=str(proj_ws))
    assert roots is not None and roots[0] == proj_ws

    # profile beats project (and config)
    roots = resolve_workspace_roots(
        cfg, workspace_root_override=str(prof_ws),
        project_workspace_dir=str(proj_ws))
    assert roots is not None and roots[0] == prof_ws

    # no config, no project -> cwd (behavior unchanged when no project)
    roots = resolve_workspace_roots(SableConfig())
    assert roots is not None and roots[0] == fake_cwd

    # project with no config still beats the cwd default
    roots = resolve_workspace_roots(SableConfig(),
                                    project_workspace_dir=str(proj_ws))
    assert roots is not None and roots[0] == proj_ws


def test_chat_stream_rejects_unknown_project(app):
    """Run context channel validates the project id before streaming."""
    r = app.handle("POST", "/api/chat/stream",
                   body={"prompt": "hi", "project": "nope"})
    assert r.status == 404
    assert r.body["error"]["code"] == "unknown_project"

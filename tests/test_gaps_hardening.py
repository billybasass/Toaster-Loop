"""Regression tests for the deep-dive review fixes (indexing / explorer / projects).

Covers:
  1. Shared embed client: get_embed_client returns ONE client per
     (model_path, ubatch); different models stay separate; lexical-only
     indexes never touch the shared registry.
  2. get_index invalidates a cached instance when the project's
     workspace_dir changes (roots follow the project record).
  3. cancel_build aborts a running build; drop_index_instance removes the
     cache entry (project-delete flow can't resurrect the DB via cache).
  4. load_config strips the never-implemented auto_retrieve_threshold key.
  5. /api/fs/read preview: happy path, binary sniff, size truncation, and
     the same containment + sensitive-path rules as /api/fs/tree.
  6. /api/fs/tree entry cap -> truncated flag.
  7. Duplicate project names rejected (409); renaming to its own name OK.
  8. Index search maps IndexRebuildRequired to HTTP 409 (not 500).
  9. A successful search clears a stale last_error in /api/index status.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sable.core.config import SableConfig, load_config  # noqa: E402
from sable.core import indexer as ix  # noqa: E402
from sable.dashboard.app import DashboardApp  # noqa: E402


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "uf-hard"
    monkeypatch.setenv("SABLE_HOME", str(h))
    return h


@pytest.fixture()
def app(home):
    return DashboardApp(SableConfig(), home=home)


@pytest.fixture(autouse=True)
def _clean_registries():
    """Keep the process-global index/embed registries from leaking."""
    yield
    ix._instances.clear()
    ix._embed_registry.clear()


# ------------------------------------------------- 1. shared embed client

def test_embed_client_shared_per_model():
    a = ix.get_embed_client("/models/x.gguf")
    b = ix.get_embed_client("/models/x.gguf")
    assert a is b
    c = ix.get_embed_client("/models/y.gguf")
    assert c is not a
    # different ubatch -> separate client (spawn args differ)
    d = ix.get_embed_client("/models/x.gguf", ubatch=1024)
    assert d is not a


def test_lexical_index_uses_no_shared_client(tmp_path):
    idx = ix.CodebaseIndex(tmp_path, index_cfg={"enabled": True})
    emb = idx.embedder()
    assert not emb.available()
    assert ix._embed_registry == {}  # nothing registered for the lexical path


# ------------------------------------- 2. workspace-change invalidation

def test_get_index_rebuilds_on_workspace_change(home):
    ws_a, ws_b = home / "a", home / "b"
    ws_a.mkdir(parents=True); ws_b.mkdir(parents=True)
    i1 = ix.get_index(home, cfg=None, project_id="p1",
                      workspace_dir=str(ws_a))
    assert i1.roots == [ws_a]
    i2 = ix.get_index(home, cfg=None, project_id="p1",
                      workspace_dir=str(ws_b))
    assert i2 is not i1
    assert i2.roots == [ws_b]
    # unchanged workspace keeps the cached instance
    assert ix.get_index(home, cfg=None, project_id="p1",
                        workspace_dir=str(ws_b)) is i2


# ------------------------------------------- 3. cancel + drop instance

def test_cancel_build_aborts(home, monkeypatch):
    ws = home / "ws"
    ws.mkdir(parents=True)
    for n in range(50):
        (ws / f"f{n}.py").write_text("x = 1\n" * 80)
    idx = ix.CodebaseIndex(home, index_cfg={"enabled": True,
                                            "paths_override": [str(ws)]})

    # Deterministic: the cancel flag flips during iteration of the FIRST
    # file (build() clears it at start, so racing a thread would be flaky).
    real_iter = ix.iter_files

    def _iter_then_cancel(roots, max_files=20_000):
        first = True
        for p in real_iter(roots, max_files=max_files):
            if first:
                idx._cancel.set()
            first = False
            yield p

    monkeypatch.setattr(ix, "iter_files", _iter_then_cancel)
    with pytest.raises(RuntimeError, match="cancelled"):
        idx.build()


def test_drop_index_instance(home):
    ix.get_index(home, cfg=None, project_id="dropme")
    assert ("dropme" in [k[1] for k in ix._instances])
    assert ix.drop_index_instance(home, "dropme") is True
    assert ix.drop_index_instance(home, "dropme") is False


# --------------------------------------- 4. phantom config key stripped

def test_auto_retrieve_threshold_stripped(tmp_path, monkeypatch):
    p = tmp_path / "cfg.yaml"
    p.write_text("indexing:\n  enabled: true\n"
                 "  auto_retrieve_threshold: 0.7\n")
    monkeypatch.setenv("SABLE_CONFIG", str(p))
    cfg = load_config()
    assert "auto_retrieve_threshold" not in cfg.indexing
    assert cfg.indexing.get("enabled") is True


# ------------------------------------------------- 5. /api/fs/read

@pytest.fixture()
def workspace(tmp_path):
    ws = tmp_path / "ws"
    (ws / "src").mkdir(parents=True)
    (ws / "src" / "app.py").write_text("print('hi')\n")
    (ws / "blob.bin").write_bytes(b"\x00\x01\x02binary")
    (ws / ".env").write_text("SECRET=1\n")
    return ws


def _mkproject(app, ws, name="p"):
    r = app.handle("POST", "/api/projects",
                   body={"name": name, "workspace_dir": str(ws)})
    assert r.status == 201, r.body
    return r.body["project"]["id"]


def test_fs_read_happy(app, workspace):
    pid = _mkproject(app, workspace)
    r = app.handle("GET", "/api/fs/read",
                   query={"project": pid, "path": "src/app.py"})
    assert r.status == 200, r.body
    assert r.body["content"] == "print('hi')\n"
    assert r.body["binary"] is False and r.body["truncated"] is False


def test_fs_read_binary_sniff(app, workspace):
    pid = _mkproject(app, workspace)
    r = app.handle("GET", "/api/fs/read",
                   query={"project": pid, "path": "blob.bin"})
    assert r.status == 200
    assert r.body["binary"] is True and r.body["content"] == ""


def test_fs_read_sensitive_and_traversal(app, workspace):
    pid = _mkproject(app, workspace)
    r = app.handle("GET", "/api/fs/read",
                   query={"project": pid, "path": ".env"})
    assert r.status == 400
    r = app.handle("GET", "/api/fs/read",
                   query={"project": pid, "path": "../escape.txt"})
    assert r.status == 400
    r = app.handle("GET", "/api/fs/read",
                   query={"project": pid, "path": "/etc/passwd"})
    assert r.status == 400
    r = app.handle("GET", "/api/fs/read",
                   query={"project": pid, "path": ""})
    assert r.status == 400


def test_fs_read_size_truncation(app, workspace, monkeypatch):
    import sable.dashboard.app as app_mod
    monkeypatch.setattr(app_mod, "FS_READ_MAX_CHARS", 10)
    # between the char cap and the hard size gate (10 < 30 <= 40 bytes) so
    # the content-level truncation branch runs rather than the metadata-only one
    (workspace / "big.txt").write_text("x" * 30)
    pid = _mkproject(app, workspace)
    r = app.handle("GET", "/api/fs/read",
                   query={"project": pid, "path": "big.txt"})
    assert r.status == 200
    assert r.body["truncated"] is True
    assert len(r.body["content"]) == 10


# ------------------------------------------------- 6. tree entry cap

def test_fs_tree_truncated_flag(app, tmp_path, monkeypatch):
    # dedicated workspace (fixture's .env would be hidden-filtered after the
    # cap slice, making the exact entry count layout-dependent)
    ws = tmp_path / "ws-cap"
    ws.mkdir()
    import sable.dashboard.app as app_mod
    monkeypatch.setattr(app_mod, "FS_TREE_MAX_ENTRIES", 2)
    for n in range(10):
        (ws / f"n{n}.txt").write_text("x")
    pid = _mkproject(app, ws)
    r = app.handle("GET", "/api/fs/tree", query={"project": pid, "path": ""})
    assert r.status == 200
    assert r.body["truncated"] is True
    assert len(r.body["entries"]) == 2


# ------------------------------------------- 7. duplicate project names

def test_duplicate_project_names_rejected(app):
    ws = Path("/tmp")  # any existing dir
    r1 = app.handle("POST", "/api/projects",
                    body={"name": "Website", "workspace_dir": str(ws)})
    assert r1.status == 201
    r2 = app.handle("POST", "/api/projects",
                    body={"name": "website ", "workspace_dir": str(ws)})
    assert r2.status == 409
    # renaming a project to its OWN name still works (id excluded)
    pid = r1.body["project"]["id"]
    r3 = app.handle("PUT", f"/api/projects/{pid}",
                    body={"name": "WEBSITE"})
    assert r3.status == 200


# ------------------------------- 8/9. 409 rebuild + stale error cleared

def test_index_search_409_on_rebuild_required(app, home, monkeypatch):
    # create a project so _index routes there
    ws = Path("/tmp")
    pid = _mkproject(app, ws, name="rebuildme")

    class Boom:
        enabled = True
        cfg = None  # get_index invalidation checks read .cfg/.index_cfg
        index_cfg: dict = {}
        roots = [Path("/tmp")]  # workspace-change check reads .roots

        def search(self, query, top_k=None, mode=None):
            raise ix.IndexRebuildRequired("dim mismatch")

        def status(self):
            return {}

    monkeypatch.setitem(ix._instances, (str(home), pid), Boom())
    r = app.handle("POST", "/api/index/search",
                   body={"query": "q", "project": pid})
    assert r.status == 409
    assert r.body["error"]["code"] == "index_rebuild_required"


def test_search_clears_stale_last_error(home):
    ws = home / "ws-clear"
    ws.mkdir(parents=True)
    (ws / "a.py").write_text("def alpha():\n    pass\n")
    idx = ix.CodebaseIndex(home, index_cfg={"enabled": True,
                                            "paths_override": [str(ws)]})
    idx.build()
    idx.last_error = "old failure"
    idx.search("alpha")
    assert idx.last_error is None


# ------------------------------------------- 10. search_codebase top_k

def test_search_codebase_schema_has_no_top_k():
    from sable.core.tools import TOOL_SCHEMAS

    sc = next(s for s in TOOL_SCHEMAS
              if s.get("function", {}).get("name") == "search_codebase")
    props = sc["function"]["parameters"]["properties"]
    assert "top_k" not in props
    assert "query" in props

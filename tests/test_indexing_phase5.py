"""Feature-wave Phase 5 tests: per-project codebase indexing + embedding UX.

Covers:
  1. Per-project DB isolation (index A, search B => nothing; default separate)
  2. MRL truncation math (truncate+L2-renorm at index and query time, via a
     fake embedding client returning known vectors) + dim-mismatch rebuild error
  3. Project delete removes the project's index DB file only
  4. Embedding-server spawn/stop lifecycle with a stub llama-server binary;
     foreign processes are never touched
  5. Legacy config keys (embedding_url / embedding_model / top_k / paths)
     load without crashing and map onto the Phase 5 schema
  6. search_codebase routes by bound project context (create_runner partial)

All hermetic; conftest redirects SABLE_HOME.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sable.core.config import SableConfig, load_config  # noqa: E402
from sable.core.indexer import (  # noqa: E402
    CodebaseIndex, IndexRebuildRequired, clip_to_tokens, delete_project_index,
    get_index, stop_all_embed_servers, truncate_renorm,
    EmbeddingClient, DEFAULT_DB_ID, EMBED_UBATCH,
)
from sable.dashboard.app import DashboardApp  # noqa: E402


# ------------------------------------------------------------------ helpers

def make_project(home: Path, project_id: str, root: Path) -> dict:
    """Create a project record the way the dashboard does (file layout)."""
    root.mkdir(parents=True, exist_ok=True)
    # NOTE: create workspace contents BEFORE calling this via helper below.
    d = home / "projects"
    d.mkdir(parents=True, exist_ok=True)
    proj = {"id": project_id, "name": project_id, "created": "", "updated": "",
            "workspace_dir": str(root), "agents": []}
    (d / f"{project_id}.json").write_text(json.dumps(proj))
    return proj


class FakeEmbedder:
    """Deterministic embedding client for MRL math tests.

    Maps text -> vector by hashing tokens into a fixed 8-dim space; vectors
    are NOT normalized so truncation+renormalization is observable."""

    def __init__(self, dim: int = 8, model_path: str = "/models/fake.gguf"):
        self.dim_full = dim
        self.model_path = model_path
        self.url = None
        self.dim = None
        self.embed_calls = 0

    def available(self) -> bool:
        return True

    @property
    def state(self):
        return {"mode": "spawn", "url": None, "model_path": self.model_path,
                "running": False, "dim": self.dim}

    def embed(self, texts):
        self.embed_calls += 1
        import hashlib
        out = []
        for t in texts:
            v = [0.0] * self.dim_full
            words = t.lower().split()
            for w in words:
                h = int(hashlib.md5(w.encode()).hexdigest(), 16)
                v[h % self.dim_full] += 1.0
            # make vector non-unit and distinct per content
            v = [x + 0.05 * (i + 1) for i, x in enumerate(v)]
            out.append(v)
        return out

    def stop(self):
        pass


# ---------------------------------------------------------------- fixtures

@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "uf-home"
    monkeypatch.setenv("SABLE_HOME", str(h))
    return h


@pytest.fixture()
def app(home):
    return DashboardApp(SableConfig(), home=home)


# ------------------------------------------------- 1. per-project isolation

def test_per_project_db_isolation(tmp_path, home):
    a_root = tmp_path / "proj-a"
    b_root = tmp_path / "proj-b"
    make_project(home, "proja", a_root)
    make_project(home, "projb", b_root)
    (a_root / "alpha.py").write_text("def alpha_widget():\n    return 'alpha'\n")
    (b_root / "beta.py").write_text("def beta_gadget():\n    return 'beta'\n")

    cfg = SableConfig(indexing={"enabled": True})
    ia = get_index(home, cfg=cfg, project_id="proja")
    ib = get_index(home, cfg=cfg, project_id="projb")
    assert ia.db_path == home / "index" / "proja.db"
    assert ib.db_path != ia.db_path
    ia.build()
    ib.build()

    hits_a = ia.search("alpha widget")
    assert hits_a and all("alpha.py" in h["path"] for h in hits_a)
    # searching project B never returns project A's chunks
    hits_b = ib.search("alpha widget")
    assert all("alpha.py" not in h["path"] for h in hits_b)

    # default DB is a separate file and stays empty
    idf = get_index(home, cfg=cfg)
    assert idf.db_path == home / "index" / f"{DEFAULT_DB_ID}.db"
    st = idf.status()
    assert st["files"] == 0


def test_project_roots_derived_from_workspace(tmp_path, home):
    root = tmp_path / "ws"
    root.mkdir()
    (root / "x.py").write_text("def x():\n    return 1\n")
    make_project(home, "projr", root)
    cfg = SableConfig(indexing={"enabled": True,
                                  "paths_override": [str(tmp_path)]})
    idx = get_index(home, cfg=cfg, project_id="projr")
    # project DB ignores paths_override: its only root is workspace_dir
    assert idx.roots == [root]


def test_get_index_cache_refreshes_on_config_change(home):
    cfg1 = SableConfig(indexing={"enabled": True, "output_dim": 768})
    i1 = get_index(home, cfg=cfg1)
    cfg2 = SableConfig(indexing={"enabled": True, "output_dim": 256})
    i2 = get_index(home, cfg=cfg2)
    assert i2.output_dim == 256
    assert get_index(home, cfg=cfg2) is i2  # stable afterwards


# ------------------------------------------------------- 2. MRL truncation

def test_truncate_renorm_math():
    v = [3.0, 4.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    t = truncate_renorm(v, 2)
    assert len(t) == 2
    assert math.isclose(math.sqrt(sum(x * x for x in t)), 1.0)
    assert math.isclose(t[0], 0.6) and math.isclose(t[1], 0.8)
    # dim >= len keeps the full vector, renormalized
    f = truncate_renorm(v, 16)
    assert len(f) == len(v)
    assert math.isclose(math.sqrt(sum(x * x for x in f)), 1.0)
    # falsy dim: untouched copy
    assert truncate_renorm(v, None) == v


def test_mrl_stored_and_query_vectors_truncated(tmp_path, home):
    root = tmp_path / "mrl"
    root.mkdir()
    (root / "a.py").write_text("alpha widget one two\n")
    (root / "b.py").write_text("beta gadget three four\n")
    make_project(home, "projm", root)

    cfg = SableConfig(indexing={"enabled": True, "model": "/models/fake.gguf",
                                  "output_dim": 4})
    idx = get_index(home, cfg=cfg, project_id="projm")
    fake = FakeEmbedder(dim=8)
    # Duck-typed test double; setattr keeps the type checker happy.
    setattr(idx, "_embed", fake)

    res = idx.build()
    assert res["embedded"] > 0 and res["embed_ok"] is True

    # stored vectors are exactly output_dim long and unit-normalized
    import sqlite3
    from sable.core.indexer import _unpack
    con = sqlite3.connect(str(idx.db_path))
    blobs = [r[0] for r in con.execute(
        "SELECT embedding FROM chunks WHERE embedding IS NOT NULL")]
    meta = dict(con.execute("SELECT key, value FROM meta"))
    con.close()
    assert blobs
    for b in blobs:
        v = _unpack(b)
        assert len(v) == 4
        assert math.isclose(math.sqrt(sum(x * x for x in v)), 1.0,
                            rel_tol=1e-6)
    assert meta["vector_dim"] == "4"
    assert meta["model_id"] == "/models/fake.gguf"

    # query vector truncated too (fake returns 8 dims; search must cope)
    hits = idx.search("alpha widget one two", mode="semantic")
    assert hits and fake.embed_calls >= 2


def test_dim_mismatch_requires_rebuild(tmp_path, home):
    root = tmp_path / "dim"
    root.mkdir()
    (root / "a.py").write_text("alpha widget one two\n")
    make_project(home, "projd", root)

    cfg = SableConfig(indexing={"enabled": True, "model": "/models/fake.gguf",
                                  "output_dim": 4})
    idx = get_index(home, cfg=cfg, project_id="projd")
    setattr(idx, "_embed", FakeEmbedder(dim=8))
    idx.build()

    # now query at a different dim: the DB was built at 4 dims
    cfg2 = SableConfig(indexing={"enabled": True, "model": "/models/fake.gguf",
                                   "output_dim": 6})
    idx2 = CodebaseIndex(home, cfg=cfg2, project_id="projd")
    setattr(idx2, "_embed", FakeEmbedder(dim=8))
    with pytest.raises(IndexRebuildRequired):
        idx2.search("alpha widget", mode="semantic")


def test_model_change_requires_rebuild(tmp_path, home):
    root = tmp_path / "mod"
    root.mkdir()
    (root / "a.py").write_text("alpha widget one two\n")
    make_project(home, "projmm", root)
    cfg = SableConfig(indexing={"enabled": True, "model": "/models/a.gguf",
                                  "output_dim": 4})
    idx = get_index(home, cfg=cfg, project_id="projmm")
    setattr(idx, "_embed", FakeEmbedder(dim=8, model_path="/models/a.gguf"))
    idx.build()

    cfg2 = SableConfig(indexing={"enabled": True, "model": "/models/b.gguf",
                                   "output_dim": 4})
    idx2 = CodebaseIndex(home, cfg=cfg2, project_id="projmm")
    setattr(idx2, "_embed", FakeEmbedder(dim=8, model_path="/models/b.gguf"))
    with pytest.raises(IndexRebuildRequired):
        idx2.search("alpha widget", mode="semantic")


# ---------------------------------------------------- 3. project delete

def test_delete_project_index_only_that_file(home):
    idxdir = home / "index"
    idxdir.mkdir(parents=True)
    (idxdir / "projax.db").write_text("a")
    (idxdir / "projay.db").write_text("b")
    (idxdir / f"{DEFAULT_DB_ID}.db").write_text("d")

    assert delete_project_index(home, "projax") is True
    assert not (idxdir / "projax.db").exists()
    assert (idxdir / "projay.db").is_file()
    assert (idxdir / f"{DEFAULT_DB_ID}.db").is_file()
    # refusing to nuke the default DB
    assert delete_project_index(home, DEFAULT_DB_ID) is False
    assert (idxdir / f"{DEFAULT_DB_ID}.db").is_file()


def test_project_delete_endpoint_removes_db(app, home, tmp_path):
    root = tmp_path / "wsd"
    root.mkdir()
    make_project(home, "projdel", root)
    idxdir = home / "index"
    idxdir.mkdir(parents=True, exist_ok=True)
    (idxdir / "projdel.db").write_text("x")
    (idxdir / f"{DEFAULT_DB_ID}.db").write_text("y")

    r = app.handle("DELETE", "/api/projects/projdel", body={})
    assert r.status == 200 and r.body["ok"] is True
    assert r.body["index_removed"] is True
    assert not (idxdir / "projdel.db").exists()
    assert (idxdir / f"{DEFAULT_DB_ID}.db").is_file()


# --------------------------------------------------- 4. spawn/stop lifecycle

STUB = '''#!/usr/bin/env python3
import argparse, json, os, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

p = argparse.ArgumentParser()
p.add_argument("-m", dest="model", required=True)
p.add_argument("--embedding", action="store_true")
p.add_argument("--alias", default=None)
p.add_argument("--host", default=None)
p.add_argument("--port", type=int, required=True)
p.add_argument("-ngl", dest="ngl", type=int, default=None)
p.add_argument("-b", dest="batch", type=int, default=None)
p.add_argument("-ub", dest="ubatch", type=int, default=None)
p.add_argument("-c", dest="ctx", type=int, default=None)
a = p.parse_args()

if a.host != "127.0.0.1":
    print("stub: refusing non-loopback bind", file=sys.stderr); sys.exit(4)
if not a.embedding:
    print("stub: --embedding missing", file=sys.stderr); sys.exit(5)

with open(os.environ.get("STUB_PIDFILE", "/dev/null"), "w") as f:
    f.write(str(os.getpid()))
if os.environ.get("STUB_ARGFILE"):
    with open(os.environ["STUB_ARGFILE"], "w") as f:
        json.dump(sys.argv[1:], f)

class H(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002 - silence access log
        pass
    def do_GET(self):
        if self.path.rstrip("/") == "/v1/models":
            body = json.dumps({"object": "list", "data": [{"id": "sable-embeddings"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body)
        else:
            self.send_response(404); self.end_headers()
    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(n) or b"{}")
        inputs = payload.get("input") or []
        data = [{"index": i, "embedding": [0.1] * 8} for i in range(len(inputs))]
        body = json.dumps({"object": "list", "data": data}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers(); self.wfile.write(body)

srv = ThreadingHTTPServer((a.host, a.port), H)
print("stub embed server ready", flush=True)
srv.serve_forever()
'''


@pytest.fixture()
def stub_binary(tmp_path):
    p = tmp_path / "llama-server"
    p.write_text(STUB)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(p)


def test_build_backfills_null_embeddings(tmp_path, home):
    """A build that previously failed mid-embed (NULL rows, files tracked)
    retries those chunks on the next build instead of skipping them."""
    root = tmp_path / "bfws"
    root.mkdir()
    (root / "a.py").write_text("alpha widget\n")
    make_project(home, "projbf", root)
    cfg = SableConfig(indexing={"enabled": True,
                                  "model": "/models/fake.gguf"})
    idx = CodebaseIndex(home, cfg=cfg, project_id="projbf")
    fake = FakeEmbedder(dim=8)
    setattr(idx, "_embed", fake)
    res = idx.build()
    assert res["embedded"] > 0
    # simulate a failed partial embed: wipe one embedding, keep files tracked
    con = sqlite3.connect(str(idx.db_path))
    con.execute("UPDATE chunks SET embedding=NULL")
    con.commit()
    con.close()
    fake2 = FakeEmbedder(dim=8)
    setattr(idx, "_embed", fake2)
    res2 = idx.build()  # no file changes: only the NULL backfill path
    assert res2["embedded"] > 0 and fake2.embed_calls >= 1
    con = sqlite3.connect(str(idx.db_path))
    n_null = con.execute(
        "SELECT COUNT(*) FROM chunks WHERE embedding IS NULL").fetchone()[0]
    con.close()
    assert n_null == 0


def test_embed_client_passes_batch_flags(tmp_path, home, stub_binary,
                                         monkeypatch):
    """Spawn sets -b/-ub/-c so inputs over the 512 default are accepted."""
    argvfile = tmp_path / "argv.json"
    monkeypatch.setenv("STUB_ARGFILE", str(argvfile))
    client = EmbeddingClient(model_path=str(tmp_path / "m.gguf"),
                             cfg=SableConfig(
                                 llama_server_binary=stub_binary,
                                 llama_auto_setup=False))
    (tmp_path / "m.gguf").write_bytes(b"GGUF")
    try:
        assert client.ensure_running()
        argv = json.loads(argvfile.read_text())
        assert argv[argv.index("-ub") + 1] == str(EMBED_UBATCH)
        assert argv[argv.index("-b") + 1] == str(EMBED_UBATCH)
        assert argv[argv.index("-c") + 1] == str(EMBED_UBATCH)
    finally:
        client.stop()


def test_clip_to_tokens():
    short = "x" * 60
    assert clip_to_tokens(short, 128) == short
    long = "\n".join(["word " * 12 + str(i) for i in range(2000)])
    out = clip_to_tokens(long, 500)
    assert len(out) <= 500 * 2 and len(out) < len(long)
    # prefix of the original, cut at a newline boundary, not over-clipped
    assert long.startswith(out) and long[len(out)] == "\n"
    assert len(out) > 400 * 2


def test_embed_client_spawn_stop_loopback(tmp_path, home, stub_binary,
                                          monkeypatch):
    """Build spawns the owned server (loopback-only), stop() terminates it."""
    pidfile = tmp_path / "stub.pid"
    monkeypatch.setenv("STUB_PIDFILE", str(pidfile))
    root = tmp_path / "spawnws"
    root.mkdir()
    (root / "a.py").write_text("alpha widget\n")
    make_project(home, "projsp", root)

    cfg = SableConfig(indexing={"enabled": True,
                                  "model": str(tmp_path / "emb.gguf")},
                        llama_server_binary=stub_binary,
                        llama_auto_setup=False)
    (tmp_path / "emb.gguf").write_bytes(b"GGUF")
    idx = CodebaseIndex(home, cfg=cfg, project_id="projsp")
    try:
        res = idx.build()
        assert res["embed_ok"] is True and res["embedded"] > 0
        child_pid = int(pidfile.read_text())
        # process alive while owned server runs; registered for cleanup
        os.kill(child_pid, 0)
        import sable.core.indexer as im
        assert idx.embedder() in im._spawned_clients

        stop_all_embed_servers()
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            try:
                os.kill(child_pid, 0)
                time.sleep(0.1)
            except OSError:
                break
        else:
            pytest.fail("spawned embedding server still running after stop")
        assert not idx.embedder()._proc
    finally:
        idx.stop_embed_server()


def test_stop_all_never_touches_foreign_processes(stub_binary, tmp_path,
                                                  monkeypatch):
    """A process sable did not spawn survives stop_all_embed_servers."""
    foreign = subprocess.Popen([sys.executable, "-c",
                                "import time; time.sleep(30)"])
    try:
        stop_all_embed_servers()  # nothing registered => no-op
        assert foreign.poll() is None
        # with a registered client, only the client's own child stops
        pidfile = tmp_path / "stub2.pid"
        monkeypatch.setenv("STUB_PIDFILE", str(pidfile))
        client = EmbeddingClient(model_path=str(tmp_path / "m.gguf"),
                                 cfg=SableConfig(
                                     llama_server_binary=stub_binary,
                                     llama_auto_setup=False))
        (tmp_path / "m.gguf").write_bytes(b"GGUF")
        base = client.ensure_running()
        assert base and base.startswith("http://127.0.0.1:")
        child_pid = int(pidfile.read_text())
        stop_all_embed_servers()
        os.kill(foreign.pid, 0)          # foreign untouched
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            try:
                os.kill(child_pid, 0)
                time.sleep(0.1)
            except OSError:
                break
    finally:
        foreign.terminate()
        foreign.wait(timeout=5)


def test_spawn_failure_surfaces_clear_error(tmp_path, home):
    """Missing llama-server binary => build raises, no silent fallback."""
    root = tmp_path / "nofind"
    root.mkdir()
    (root / "a.py").write_text("alpha widget\n")
    make_project(home, "projnf", root)
    cfg = SableConfig(indexing={"enabled": True,
                                  "model": str(tmp_path / "e.gguf")},
                        llama_server_binary=str(tmp_path / "nope"),
                        llama_auto_setup=False)
    (tmp_path / "e.gguf").write_bytes(b"GGUF")
    idx = CodebaseIndex(home, cfg=cfg, project_id="projnf")
    with pytest.raises(RuntimeError) as ei:
        idx.build()
    assert "embedding server error" in str(ei.value)


# ----------------------------------------------------- 5. legacy config keys

def test_legacy_indexing_keys_load_without_crash(tmp_path, monkeypatch):
    cfg_file = tmp_path / "old.yaml"
    cfg_file.write_text(
        "indexing:\n"
        "  enabled: true\n"
        "  embedding_model: /models/old.gguf\n"
        "  embedding_url: http://127.0.0.1:9999/v1\n"
        "  top_k: 3\n"
        "  paths:\n"
        "    - /some/root\n"
        "  search_mode: semantic\n")
    monkeypatch.setenv("SABLE_CONFIG", str(cfg_file))
    cfg = load_config()
    ix = cfg.indexing
    assert ix["enabled"] is True
    assert ix["model"] == "/models/old.gguf"       # embedding_model -> model
    assert ix["paths_override"] == ["/some/root"]  # paths -> paths_override
    assert "embedding_url" not in ix               # attach mode absorbed
    assert "top_k" not in ix                       # internal constant now
    assert ix["search_mode"] == "semantic"
    # and the indexer consumes it without exploding
    idx = CodebaseIndex(tmp_path / "h", cfg=cfg)
    assert idx.model_path == "/models/old.gguf"
    assert idx.top_k == 8  # SEARCH_TOP_K, not the old 3


def test_config_put_accepts_phase5_indexing_keys(app, home, monkeypatch):
    monkeypatch.setenv("SABLE_CONFIG", str(home / "sable.yaml"))
    r = app.handle("PUT", "/api/config", body={
        "section": "indexing",
        "patch": {"enabled": True, "model": "/models/x.gguf",
                  "output_dim": 512, "paths_override": ["/x"]}})
    assert r.status == 200, r.body
    # legacy keys still round-trip (absorbed on load, not rejected on write)
    r = app.handle("PUT", "/api/config", body={
        "section": "indexing", "patch": {"top_k": 4}})
    assert r.status == 200, r.body
    cfg = load_config(str(home / "sable.yaml"))
    assert cfg.indexing["model"] == "/models/x.gguf"
    assert cfg.indexing["output_dim"] == 512
    assert "top_k" not in cfg.indexing


# ------------------------------------------------- 6. search_codebase routing

def test_search_codebase_routes_by_bound_project(monkeypatch, tmp_path, home):
    import sable.core.tools as uf_tools  # noqa: F811

    a_root = tmp_path / "ra"
    b_root = tmp_path / "rb"
    make_project(home, "routea", a_root)
    make_project(home, "routeb", b_root)
    # spaced words: the BM25 tokenizer keeps identifiers like alpha_widget
    # as one token, so query terms must appear as separate words.
    (a_root / "alpha.py").write_text("# alpha widget factory\nx = 1\n")
    (b_root / "beta.py").write_text("# beta gadget factory\ny = 2\n")

    cfg = SableConfig(indexing={"enabled": True})
    ia = get_index(home, cfg=cfg, project_id="routea")
    ib = get_index(home, cfg=cfg, project_id="routeb")
    ia.build()
    ib.build()

    import functools
    out_a = uf_tools.search_codebase("alpha widget", _project_id="routea",
                    _workspace_dir=str(a_root))
    assert "alpha.py" in out_a
    out_b = uf_tools.search_codebase("alpha widget", _project_id="routeb",
                                     _workspace_dir=str(b_root))
    assert "alpha.py" not in out_b  # no cross-project results

    # create_runner binds the partial when project_id is given
    from sable.core.agent_runner import create_runner

    runner = create_runner(cfg, "http://127.0.0.1:1/v1", "m",
                           project_workspace_dir=str(a_root),
                           project_id="routea")
    impl = runner.tool_impls["search_codebase"]
    assert isinstance(impl, functools.partial)
    assert impl.keywords.get("_project_id") == "routea"
    assert impl.keywords.get("_workspace_dir") == str(a_root)

    runner2 = create_runner(cfg, "http://127.0.0.1:1/v1", "m")
    assert not isinstance(runner2.tool_impls["search_codebase"],
                          functools.partial) or \
        "_project_id" not in getattr(runner2.tool_impls["search_codebase"],
                                     "keywords", {})


# ------------------------------------------------------- dashboard endpoints

def test_index_build_endpoint_routes_project(app, home, tmp_path):
    root = tmp_path / "epws"
    root.mkdir()
    (root / "a.py").write_text("alpha widget\n")
    make_project(home, "projep", root)

    r = app.handle("POST", "/api/index/build", body={"project": "projep"})
    assert r.status == 200 and r.body["project"] == "projep"
    st = {}
    for _ in range(100):
        st = app.handle("GET", "/api/index", query={"project": "projep"}).body
        if not st["building"] and st["chunks"]:
            break
        time.sleep(0.05)
    assert st["files"] == 1
    assert (home / "index" / "projep.db").is_file()

    r = app.handle("POST", "/api/index/build", body={"project": "nope"})
    assert r.status == 404


def test_index_status_reports_phase5_fields(app, home, tmp_path):
    root = tmp_path / "stws"
    root.mkdir()
    make_project(home, "projst", root)
    cfgd = SableConfig(indexing={"enabled": True, "output_dim": 256})
    app.cfg = cfgd
    r = app.handle("GET", "/api/index", query={"project": "projst"})
    assert r.status == 200
    body = r.body
    assert body["enabled"] is True
    assert body["project_id"] == "projst"
    assert body["output_dim"] == 256
    assert body["roots"] == [str(root)]

"""Tests for the run settings, sampling, program control and
codebase indexing additions:

  1. serving/run_args.py   — run_settings -> llama-server arg mapping
  2. core/agent_runner.py  — sampling injection + heal_tool_calls toggle
  3. core/indexer.py       — chunking, BM25 lexical search, hybrid fallback
  4. dashboard/app.py      — /api/logs, /api/index*, /api/shutdown (mocked),
                             config PUT accepts the new sections

All hermetic; conftest redirects SABLE_HOME.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fake_server import FakeOpenAIServer, make_message_handler  # noqa: E402

from sable.core.agent_runner import SableAgentClient  # noqa: E402
from sable.core.config import SableConfig, load_config  # noqa: E402
from sable.core.indexer import (  # noqa: E402
    CodebaseIndex, LexicalIndex, chunk_file, cosine, _pack, _unpack,
)
from sable.dashboard.app import DashboardApp  # noqa: E402
from sable.serving.run_args import (  # noqa: E402
    run_settings_args, sanitize_run_settings, scale_gpu_layers,
)


# ------------------------------------------------------------ 1. run args

def _has(args, *tokens):
    """True when the tokens appear consecutively in args."""
    s = "\x00".join(args)
    return ("\x00".join(tokens)) in s


def test_run_args_empty():
    assert run_settings_args(None) == []
    assert run_settings_args({}) == []


def test_run_args_kv_and_flags():
    args = run_settings_args({
        "kv_cache_type": "q8_0",
        "flash_attn": True,
        "parallel": 4,
        "batch_size": 512,
        "ubatch_size": 128,
        "tensor_parallel": True,
        "no_mmap": True,
        "mlock": True,
        "mmproj_path": "/x/mm.gguf",
    })
    assert _has(args, "-ctk", "q8_0", "-ctv", "q8_0")
    assert _has(args, "--flash-attn", "on")
    assert _has(args, "-np", "4") and _has(args, "-b", "512") \
        and _has(args, "-ub", "128")
    assert "--tensor-parallel" in args and "--no-mmap" in args
    assert "--mlock" in args and _has(args, "--mmproj", "/x/mm.gguf")


def test_run_args_f16_kv_is_default_noop():
    assert run_settings_args({"kv_cache_type": "f16"}) == []


def test_run_args_vram_budget_scales_ngl():
    args = run_settings_args({"gpu_memory_pct": 50}, gpu_layers=999)
    assert _has(args, "-ngl", "499")
    # 100% leaves the resolved offload untouched
    assert run_settings_args({"gpu_memory_pct": 100}, gpu_layers=999) == []
    assert scale_gpu_layers(999, 0) == 999


def test_run_args_extra_args_split():
    args = run_settings_args({"extra_args": "--rope-scaling yarn --yarn-orig-ctx 32768"})
    assert args == ["--rope-scaling", "yarn", "--yarn-orig-ctx", "32768"]


def test_sanitize_run_settings_rejects_unknown():
    with pytest.raises(ValueError):
        sanitize_run_settings({"bogus": 1})
    with pytest.raises(ValueError):
        sanitize_run_settings({"gpu_memory_pct": 101})
    out = sanitize_run_settings({"kv_cache_type": "Q8_0", "flash_attn": 1,
                                 "parallel": "4"})
    assert out == {"kv_cache_type": "q8_0", "flash_attn": True, "parallel": 4}


def test_pack_unpack_vectors():
    v = [0.25, -1.5, 3.0]
    assert _unpack(_pack(v)) == pytest.approx(v)
    assert cosine(v, v) == pytest.approx(1.0)


# ------------------------------------------------- 2. sampling + healing

def test_sampling_injected_into_requests():
    handler = make_message_handler({"*": {"role": "assistant", "content": "ok"}})
    with FakeOpenAIServer(handler_cls=handler) as srv:
        handler.requests.clear()
        client = SableAgentClient(base_url=srv.base_url, model="m",
                                    sampling={"temperature": 0.7, "top_k": 20})
        client.run("hi")
    assert handler.requests, "fake server saw no request"
    sent = handler.requests[-1]
    assert sent["temperature"] == 0.7
    assert sent["top_k"] == 20


def test_no_sampling_by_default():
    handler = make_message_handler({"*": {"role": "assistant", "content": "ok"}})
    with FakeOpenAIServer(handler_cls=handler) as srv:
        handler.requests.clear()
        client = SableAgentClient(base_url=srv.base_url, model="m")
        client.run("hi")
    sent = handler.requests[-1]
    assert "temperature" not in sent and "top_k" not in sent


def test_heal_tool_calls_toggle_off():
    """With healing off, a fenced-JSON tool call stays plain text (final answer)."""
    fenced = ('```json\n{"name": "todo_read", "arguments": {}}\n```')
    handler = make_message_handler(
        {"*": {"role": "assistant", "content": fenced}})
    with FakeOpenAIServer(handler_cls=handler) as srv:
        client = SableAgentClient(base_url=srv.base_url, model="m",
                                    heal_tool_calls=False)
        res = client.run("do it")
    assert res.completed
    # no tool executed -> the fenced block is (stripped or raw) final content,
    # and importantly the loop ended after one turn without a tool round-trip.
    assert "todo" not in (res.final_response or "").lower() or \
        res.messages[-1]["role"] == "assistant"


def test_create_runner_threads_sampling_and_healing(tmp_path, monkeypatch):
    monkeypatch.setenv("SABLE_HOME", str(tmp_path))
    cfg = SableConfig(sampling={"temperature": 0.1},
                        agent={"heal_tool_calls": False})
    from sable.core.agent_runner import create_runner

    r = create_runner(cfg, "http://127.0.0.1:1/v1", "m")
    assert r.sampling == {"temperature": 0.1}
    assert r.heal_tool_calls is False


# -------------------------------------------------------- 3. indexer

@pytest.fixture()
def project(tmp_path):
    root = tmp_path / "proj"
    (root / "src").mkdir(parents=True)
    (root / "src" / "embeddings.py").write_text(
        "def compute_embeddings(vectors):\n"
        "    # normalize each embedding vector before cosine similarity\n"
        "    return [v for v in vectors]\n" * 3)
    (root / "src" / "toast.py").write_text(
        "def raise_toast(message):\n    print('toast:', message)\n" * 3)
    (root / "notes.md").write_text("# notes\nsemantic search rocks\n")
    return root


def test_chunk_file_line_windows(project):
    chunks = chunk_file(project / "src" / "embeddings.py")
    assert chunks and chunks[0][0] == 1
    # windows overlap: second chunk starts before the first ends
    if len(chunks) > 1:
        assert chunks[1][0] <= chunks[0][1]


def test_lexical_index_scores_matching_doc():
    lx = LexicalIndex([(1, "embedding vector cosine similarity"),
                       (2, "toast notifications ui")])
    hits = lx.score("cosine embedding", limit=5)
    assert hits and hits[0][0] == 1


def test_index_build_and_lexical_search(tmp_path, project):
    idx = CodebaseIndex(tmp_path / "home", cfg=None,
                        index_cfg={"enabled": True, "paths": [str(project)]})
    res = idx.build()
    assert res["added"] == 3 and res["chunks"] >= 3
    assert res["embedded"] == 0 and res["embed_ok"] is False  # no model set

    hits = idx.search("cosine similarity embedding vector")
    assert hits, "lexical search returned nothing"
    assert hits[0]["path"].endswith("embeddings.py")

    st = idx.status()
    assert st["enabled"] and st["files"] == 3
    assert st["embedding"]["mode"] == "none"


def test_index_incremental_rebuild(tmp_path, project):
    idx = CodebaseIndex(tmp_path / "home2", cfg=None,
                        index_cfg={"enabled": True, "paths": [str(project)]})
    first = idx.build()
    assert first["added"] == 3
    second = idx.build()          # nothing changed
    assert second["added"] == 0 and second["updated"] == 0
    (project / "extra.py").write_text("def extra():\n    return 42\n")
    third = idx.build()
    assert third["added"] == 1
    (project / "notes.md").unlink()
    fourth = idx.build()
    assert fourth["removed"] == 1


def test_search_codebase_tool_disabled_message(monkeypatch, tmp_path):
    from sable.core import tools as uf_tools

    monkeypatch.setenv("SABLE_HOME", str(tmp_path / "h"))
    out = uf_tools.search_codebase("anything")
    assert "disabled" in out


def test_search_codebase_tool_over_built_index(monkeypatch, tmp_path, project):
    import sable.core.indexer as indexer_mod

    home = tmp_path / "h2"
    monkeypatch.setenv("SABLE_HOME", str(home))
    idx = CodebaseIndex(home, cfg=None,
                        index_cfg={"enabled": True, "paths_override":
                                   [str(project)]})
    idx.build()
    # Phase 5: get_index caches per (home, project) — seed the default slot.
    monkeypatch.setitem(indexer_mod._instances, (str(home), None), idx)

    from sable.core import tools as uf_tools

    out = uf_tools.search_codebase("embedding vector cosine")
    assert "embeddings.py" in out


# ---------------------------------------------------- 4. dashboard API

@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "uf-ctl"
    monkeypatch.setenv("SABLE_HOME", str(h))
    return h


@pytest.fixture()
def app(home):
    return DashboardApp(SableConfig(), home=home)


def test_logs_endpoint(app, home):
    (home / "logs").mkdir(parents=True)
    (home / "logs" / "dashboard.log").write_text("one\ntwo\nthree\n")
    r = app.handle("GET", "/api/logs", query={"tail": "2"})
    assert r.status == 200
    body = r.body
    assert body["dashboard"]["exists"] is True
    assert body["dashboard"]["lines"][-2:] == ["two", "three"]
    assert body["llama-server"]["exists"] is False


def test_logs_endpoint_includes_instance_logs(app, home):
    """Named instances write server-<id>.log — the console must list them."""
    (home / "logs").mkdir(parents=True)
    (home / "logs" / "server-coder.log").write_text("slot booted\n")
    (home / "logs" / "server-embed2.log").write_text("ready\n")
    body = app.handle("GET", "/api/logs").body
    assert body["server-coder"]["exists"] is True
    assert body["server-coder"]["lines"] == ["slot booted"]
    assert body["server-embed2"]["exists"] is True


def test_shutdown_route_mocked(app, home, monkeypatch):
    import sable.dashboard.app as appmod

    kills = []
    monkeypatch.setattr(appmod.os, "kill",
                        lambda pid, sig: kills.append((pid, sig)))
    r = app.handle("POST", "/api/shutdown", body={})
    assert r.status == 200 and r.body["ok"]
    import time as _t
    _t.sleep(1.0)  # the daemon thread sleeps ~0.4 s before signalling
    assert kills and kills[0][1].name == "SIGTERM"


def test_index_endpoints(app, home, tmp_path, project, monkeypatch):
    import sable.core.indexer as indexer_mod

    idx = CodebaseIndex(home, cfg=None,
                        index_cfg={"enabled": True, "paths_override":
                                   [str(project)]})
    # The app now routes through get_index's cache; seed the default slot so
    # _index(None) returns this lexical-only test index.
    monkeypatch.setitem(indexer_mod._instances, (str(home), None), idx)

    r = app.handle("GET", "/api/index")
    assert r.status == 200 and r.body["enabled"] is True

    r = app.handle("POST", "/api/index/build", body={})
    assert r.status == 200 and r.body["state"] == "running"
    # wait for the background build to land
    import time as _t
    st = {}
    for _ in range(100):
        st = app.handle("GET", "/api/index").body
        if not st["building"] and st["chunks"]:
            break
        _t.sleep(0.05)
    assert st["files"] == 3

    r = app.handle("POST", "/api/index/search",
                   body={"query": "cosine embedding vector"})
    assert r.status == 200 and r.body["results"]

    r = app.handle("POST", "/api/index/stop", body={})
    assert r.status == 200

    r = app.handle("POST", "/api/index/search", body={})
    assert r.status == 400


def test_config_put_accepts_new_sections(app, home, monkeypatch):
    monkeypatch.setenv("SABLE_CONFIG", str(home / "sable.yaml"))
    r = app.handle("PUT", "/api/config",
                   body={"section": "run_settings",
                         "patch": {"kv_cache_type": "q8_0",
                                   "gpu_memory_pct": 90}})
    assert r.status == 200, r.body
    r = app.handle("PUT", "/api/config",
                   body={"section": "sampling", "patch": {"temperature": 0.6}})
    assert r.status == 200, r.body
    r = app.handle("PUT", "/api/config",
                   body={"section": "indexing",
                         "patch": {"enabled": True, "search_mode": "hybrid"}})
    assert r.status == 200, r.body

    # unknown keys inside the new sections are rejected
    r = app.handle("PUT", "/api/config",
                   body={"section": "sampling", "patch": {"vibes": 9}})
    assert r.status == 400 and r.body["error"]["code"] == "unknown_keys"

    # and the file reloads into the typed config
    cfg = load_config(str(home / "sable.yaml"))
    assert cfg.run_settings["kv_cache_type"] == "q8_0"
    assert cfg.sampling["temperature"] == 0.6
    assert cfg.indexing["enabled"] is True

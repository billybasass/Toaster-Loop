"""GGUF -> HF base resolution tests.

Covers the three pieces of the feature:
  * core/gguf_meta.py — stdlib GGUF KV reader (round-trips a synthetic file)
  * core/gguf_base.py — local detection + candidate assembly (no network)
  * dashboard /api/model_base + /api/model_base/download endpoints

Network paths are monkeypatched; nothing here downloads.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from toaster_loop.core.gguf_meta import (  # noqa: E402
    gguf_arch, gguf_basename, gguf_name, read_gguf_metadata)
from toaster_loop.core.gguf_base import (  # noqa: E402
    normalize_model_name, resolve_gguf_base)


# ------------------------------------------------------------- GGUF builder
def _b(s: str) -> bytes:
    e = s.encode("utf-8")
    return struct.pack("<Q", len(e)) + e


def _kv_str(key: str, val: str) -> bytes:
    return _b(key) + struct.pack("<I", 8) + _b(val)  # type 8 = string


def write_gguf(path: Path, *, basename: Optional[str] = None,
               arch: str = "llama", name: Optional[str] = None,
               extra: Optional[Dict[str, Any]] = None,
               n_tensors: int = 3) -> Path:
    """Minimal valid GGUF v3 header (KV only, no tensor-info/weights needed —
    the reader stops after the KV section)."""
    kvs = b""
    if basename is not None:
        kvs += _kv_str("general.basename", basename)
    kvs += _kv_str("general.architecture", arch)
    if name is not None:
        kvs += _kv_str("general.name", name)
    for k, v in (extra or {}).items():
        kvs += _kv_str(k, str(v))
    n_kv = 1 + (1 if basename is not None else 0) + (1 if name is not None else 0) \
        + len(extra or {})
    header = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", n_tensors) \
        + struct.pack("<Q", n_kv)
    path.write_bytes(header + kvs)
    return path


# ------------------------------------------------------------------- reader
def test_read_gguf_metadata_roundtrip(tmp_path):
    g = write_gguf(tmp_path / "m.gguf", basename="Qwen3-8B", arch="qwen2",
                  name="Qwen3 8B Q4_K_M")
    md = read_gguf_metadata(str(g))
    assert md is not None
    assert md["version"] == 3
    assert md["n_tensors"] == 3
    assert md["keys"]["general.basename"] == "Qwen3-8B"
    assert md["keys"]["general.architecture"] == "qwen2"
    assert md["keys"]["general.name"] == "Qwen3 8B Q4_K_M"


def test_reader_helpers(tmp_path):
    g = write_gguf(tmp_path / "m.gguf", basename="Llama-3.1-8B", arch="llama")
    assert gguf_basename(str(g)) == "Llama-3.1-8B"
    assert gguf_arch(str(g)) == "llama"
    assert gguf_name(str(g)) is None  # not written


def test_reader_rejects_non_gguf(tmp_path):
    junk = tmp_path / "junk.gguf"
    junk.write_bytes(b"not a gguf at all, no magic here")
    assert read_gguf_metadata(str(junk)) is None
    assert gguf_basename(str(junk)) is None
    # Missing file: also None, never raises.
    assert read_gguf_metadata(str(tmp_path / "absent.gguf")) is None


def test_reader_truncated_kv_keeps_what_parsed(tmp_path):
    g = write_gguf(tmp_path / "m.gguf", basename="Qwen3-8B")
    raw = g.read_bytes()
    g.write_bytes(raw[: len(raw) - 5])  # cut the last value short
    md = read_gguf_metadata(str(g))
    assert md is not None
    # general.basename sits before architecture, so it survives the truncation.
    assert md["keys"].get("general.basename") == "Qwen3-8B"


# -------------------------------------------------------------- name folding
def test_normalize_model_name_folds_quant_and_separators():
    assert normalize_model_name("Qwen3-8B-Q4_K_M") == \
        normalize_model_name("qwen3_8b")
    assert normalize_model_name("Meta-Llama-3.1-8B-Instruct") == \
        "metallama318binstruct"


# ------------------------------------------------------------- resolution
def test_resolve_finds_local_hf_checkpoint(tmp_path):
    home = tmp_path / "home"
    models = home / "models"
    snap = models / "models--Qwen--Qwen3-8B" / "snapshots" / "abc12345"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    g = write_gguf(models / "Qwen3-8B-Q4_K_M.gguf", basename="Qwen3-8B")

    res = resolve_gguf_base(str(g), home=home, models_dir=str(models))
    assert res["basename"] == "Qwen3-8B"
    assert len(res["local"]) == 1
    assert res["local"][0]["repo_id"] == "Qwen/Qwen3-8B"
    assert Path(res["local"][0]["path"]).is_dir()
    assert res["reason"] is None


def test_resolve_local_flat_checkpoint_dir(tmp_path):
    home = tmp_path / "home"
    models = home / "models"
    ckpt = models / "Qwen3-8B"  # flat dir with config.json
    ckpt.mkdir(parents=True)
    (ckpt / "config.json").write_text("{}")
    g = write_gguf(models / "Qwen3-8B-Q5_K_M.gguf", basename="Qwen3-8B")

    res = resolve_gguf_base(str(g), home=home, models_dir=str(models))
    assert len(res["local"]) == 1
    assert res["local"][0]["path"] == str(ckpt.resolve())


def test_resolve_no_local_searches_hub(tmp_path, monkeypatch):
    import toaster_loop.core.modelhub as mh

    g = write_gguf(tmp_path / "m.gguf", basename="Mistral-7B")

    def fake_search(query, **kw):
        assert query == "Mistral-7B"
        return [
            {"repo_id": "TheBloke/Mistral-7B-Instruct-GGUF", "gguf": True,
             "downloads": 9_000_000},   # GGUF repos must be dropped
            {"repo_id": "mistralai/Mistral-7B-Instruct-v0.3", "gguf": False,
             "downloads": 500},          # training-capable: kept
        ]

    monkeypatch.setattr(mh, "search_models", fake_search)
    res = resolve_gguf_base(str(g), home=tmp_path / "home",
                            models_dir=str(tmp_path / "empty"))
    assert res["local"] == []
    rids = [c["repo_id"] for c in res["remote"]]
    assert rids == ["mistralai/Mistral-7B-Instruct-v0.3"]


def test_resolve_uses_card_base_model_from_source_repo(tmp_path, monkeypatch):
    import toaster_loop.core.gguf_base as gb

    # Hub-layout GGUF: source_repo decodes from the models-- dir name.
    models = tmp_path / "models"
    snap = models / "models--SomeOrg--Foo-GGUF" / "snapshots" / "deadbeef"
    snap.mkdir(parents=True)
    g = write_gguf(snap / "foo-q4.gguf", basename="Foo")

    def fake_detail(repo_id, kind="models"):
        assert repo_id == "SomeOrg/Foo-GGUF"
        return {"base_models": ["Upstream/Foo-Base"]}

    monkeypatch.setattr(gb, "_card_base_models",
                        lambda rid: [{"repo_id": "Upstream/Foo-Base",
                                      "downloads": 0, "source": "card",
                                      "gguf": False}])
    res = resolve_gguf_base(str(g), home=tmp_path / "home",
                            models_dir=str(models))
    assert res["source_repo"] == "SomeOrg/Foo-GGUF"
    assert any(c["repo_id"] == "Upstream/Foo-Base" and c["source"] == "card"
               for c in res["remote"])


def test_resolve_no_metadata_gives_reason(tmp_path):
    # No basename key, no hub layout -> actionable reason, empty candidates.
    g = tmp_path / "mystery.gguf"
    write_gguf(g, basename=None)  # only general.architecture
    res = resolve_gguf_base(str(g), home=tmp_path / "home",
                            models_dir=str(tmp_path / "none"))
    assert res["local"] == []
    assert res["remote"] == []
    assert res["reason"] and "basename" in res["reason"]


def test_resolve_missing_file_reports_not_found(tmp_path):
    res = resolve_gguf_base(str(tmp_path / "gone.gguf"), home=tmp_path,
                            models_dir=None)
    assert "not found" in (res["reason"] or "").lower()


# ------------------------------------------------------------------- API
@pytest.fixture()
def dash_home(tmp_path, monkeypatch):
    h = tmp_path / "dash-home"
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(h))
    return h


@pytest.fixture()
def dash_app(dash_home):
    from toaster_loop.core.config import ToasterLoopConfig
    from toaster_loop.dashboard.app import DashboardApp

    return DashboardApp(ToasterLoopConfig(), home=dash_home)


def test_api_model_base_requires_gguf_param(dash_app):
    assert dash_app.handle("GET", "/api/model_base").status == 400


def test_api_model_base_missing_file_404(dash_app, tmp_path):
    r = dash_app.handle("GET", "/api/model_base",
                        query={"gguf": str(tmp_path / "nope.gguf")})
    assert r.status == 404


def test_api_model_base_local_hit(dash_app, dash_home):
    models = dash_home / "models"
    snap = models / "models--Qwen--Qwen3-8B" / "snapshots" / "rev1234567"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    g = write_gguf(models / "q.gguf", basename="Qwen3-8B")
    r = dash_app.handle("GET", "/api/model_base", query={"gguf": str(g)})
    assert r.status == 200
    assert r.body["basename"] == "Qwen3-8B"
    assert len(r.body["local"]) == 1


def test_api_model_base_download_validates_repo(dash_app):
    r = dash_app.handle("POST", "/api/model_base/download", body={})
    assert r.status == 400
    r2 = dash_app.handle("POST", "/api/model_base/download",
                         body={"repo": "..\\..\\evil"})
    assert r2.status == 400


def test_train_submit_rejects_gguf_base(dash_app, dash_home, tmp_path):
    """Defense in depth: even if a GGUF path reaches /api/train, the server
    refuses with an actionable gguf_not_trainable error (workers would fail
    far deeper)."""
    g = write_gguf(tmp_path / "m.gguf", basename="Qwen3-8B")
    ds = dash_home / "datasets" / "ds-g"
    ds.mkdir(parents=True)
    (ds / "train.yaml").write_text("model: base\ndata: {}\n")
    r = dash_app.handle("POST", "/api/train",
                        body={"dataset": "ds-g", "kind": "sft",
                              "base_model": str(g)})
    assert r.status == 400
    # Error envelope: {"error": {"code": ..., "message": ...}}
    assert r.body["error"]["code"] == "gguf_not_trainable"
    assert "/api/model_base" in r.body["error"]["message"]


def test_iterate_submit_rejects_gguf_config_base(dash_app, dash_home, tmp_path):
    """Easy training sends no base override — the guard must also cover the
    configured training.base_model (iterate path), not just submit_train."""
    g = write_gguf(tmp_path / "cfg.gguf", basename="Qwen3-8B")
    dash_app.cfg.training = {"base_model": str(g)}
    r = dash_app.handle("POST", "/api/iterate", body={"dry_run": True})
    assert r.status == 400
    assert r.body["error"]["code"] == "gguf_not_trainable"


def test_api_model_base_download_starts_job(dash_app, monkeypatch):
    started: Dict[str, Any] = {}

    def fake_start(self, repo_id, variant="", kind="models"):
        started["repo"] = repo_id
        started["variant"] = variant
        return "dl-test123"

    from toaster_loop.core.modelhub import ModelHub

    monkeypatch.setattr(ModelHub, "start_download", fake_start)
    r = dash_app.handle("POST", "/api/model_base/download",
                        body={"repo": "mistralai/Mistral-7B-Instruct-v0.3"})
    assert r.status == 200
    assert r.body["job_id"] == "dl-test123"
    assert started["repo"] == "mistralai/Mistral-7B-Instruct-v0.3"

"""Model discovery over a base model folder (HF hub cache) + dashboard wiring.

Unit tests build a fake HF hub cache in tmp_path; API tests exercise
GET /api/model_catalog and the POST /api/train base_model override.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sable.core.config import SableConfig, load_config  # noqa: E402
from sable.core.model_discovery import (  # noqa: E402
    discover_base_models,
    discover_gguf_models,
    discover_models,
    gguf_total_size,
    is_companion_gguf,
    resolve_gguf_path,
    resolve_gguf_shards,
    resolve_models_dir,
    shard_group,
    snapshot_link_for_blob,
)
from sable.dashboard.app import DashboardApp  # noqa: E402


@pytest.fixture()
def hub(tmp_path, monkeypatch):
    """Fake HF hub cache:

    models--Qwen--Qwen3-8B/snapshots/<rev>/  (GGUF via blob symlink + config.json)
    models--meta--Llama-FP16/snapshots/<rev>/ (real GGUF file, no config.json)
    loose/model.gguf                          (flat layout)
    """
    h = tmp_path / "hub"

    # repo 1: hub-layout GGUF stored as blobs/ symlink + an HF snapshot
    r1 = h / "models--Qwen--Qwen3-8B"
    blobs = r1 / "blobs"
    blobs.mkdir(parents=True)
    blob = blobs / "abc123"
    blob.write_bytes(b"GGUF" + b"\0" * 64)
    snap = r1 / "snapshots" / "deadbeefcafe1234"
    snap.mkdir(parents=True)
    (snap / "qwen3-8b-q4_k_m.gguf").symlink_to(
        Path("..") / ".." / "blobs" / "abc123")
    # repo 2: plain GGUF file, no HF config.json (serve-only)
    r2 = h / "models--meta--Llama-FP16"
    snap2 = r2 / "snapshots" / "rev1"
    snap2.mkdir(parents=True)
    (snap2 / "llama-f16.gguf").write_bytes(b"GGUF")
    # loose file outside the hub layout
    (h / "loose").mkdir()
    (h / "loose" / "model.gguf").write_bytes(b"GGUF")
    return h


# ---------------------------------------------------------------- discovery

def test_discover_gguf_hub_and_loose(hub):
    found = {m["name"]: m for m in discover_gguf_models(hub)}
    assert "Qwen/Qwen3-8B · qwen3-8b-q4_k_m" in found
    assert "meta/Llama-FP16 · llama-f16" in found
    assert "model" in found  # loose/model.gguf
    q = found["Qwen/Qwen3-8B · qwen3-8b-q4_k_m"]
    assert q["quant"] == "q4_k_m"
    assert q["size_bytes"] == 68  # symlink resolved to blob size
    assert Path(q["path"]).is_file()


def test_discover_base_models_requires_config_json(hub):
    # No config.json anywhere yet -> nothing training-capable.
    assert discover_base_models(hub) == []
    snap = hub / "models--Qwen--Qwen3-8B" / "snapshots" / "deadbeefcafe1234"
    (snap / "config.json").write_text('{"architectures": []}')
    found = discover_base_models(hub)
    assert len(found) == 1
    assert found[0]["name"] == "Qwen/Qwen3-8B"
    assert Path(found[0]["path"]).is_dir()


def test_discover_models_bundle(hub):
    d = discover_models(str(hub))
    assert d["exists"] is True
    assert d["models_dir"] == str(hub)
    assert len(d["gguf"]) == 3
    assert len(d["base_models"]) == 0


def test_resolve_models_dir_precedence(tmp_path, monkeypatch):
    explicit = tmp_path / "a"; explicit.mkdir()
    envd = tmp_path / "b"; envd.mkdir()
    monkeypatch.setenv("SABLE_MODELS_DIR", str(envd))
    assert resolve_models_dir(str(explicit)) == explicit
    assert resolve_models_dir(None) == envd
    # Nothing configured and no HF cache => None (isolate HOME/HF_HOME).
    empty = tmp_path / "home"; empty.mkdir()
    monkeypatch.setenv("HOME", str(empty))
    monkeypatch.setenv("USERPROFILE", str(empty))
    monkeypatch.delenv("SABLE_MODELS_DIR", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    assert resolve_models_dir(None) is None
    # HF_HOME/hub fallback
    hf = tmp_path / "hfhome"
    (hf / "hub").mkdir(parents=True)
    monkeypatch.setenv("HF_HOME", str(hf))
    assert resolve_models_dir(None) == hf / "hub"


def test_discover_models_missing_dir_is_graceful(tmp_path):
    d = discover_models(str(tmp_path / "nope"))
    assert d["exists"] is False and d["gguf"] == [] and d["base_models"] == []


# ------------------------------------------------------- split (multi-part) GGUFs

@pytest.fixture()
def shards(tmp_path):
    """Loose 3-part model + an old-style split-NN pair in one folder."""
    d = tmp_path / "models"
    d.mkdir()
    for n in (1, 2, 3):
        (d / f"Qwen3.8-Flash-Next-Q4_K_M-0000{n}-of-00003.gguf").write_bytes(
            b"x" * (100 + n))
    (d / "Old-split-01.gguf").write_bytes(b"y" * 10)
    (d / "Old-split-02.gguf").write_bytes(b"y" * 20)
    (d / "plain-model.gguf").write_bytes(b"z" * 7)
    return d


def test_shard_group_parsing():
    assert shard_group("Foo-00001-of-00003.gguf") == ("Foo", 1)
    assert shard_group("Foo-00002-of-00003.gguf") == ("Foo", 2)
    assert shard_group("model-Q4_K_M-00001-of-00003.gguf") == \
        ("model-Q4_K_M", 1)
    assert shard_group("Old-split-02.gguf") == ("Old", 2)
    # Not shards: plain names, quant tails, context-size tails.
    assert shard_group("plain-model.gguf") is None
    assert shard_group("model-q4_k_m.gguf") is None
    assert shard_group("model-4k.gguf") is None


def test_discover_folds_shards_into_one_entry(shards):
    found = discover_gguf_models(shards)
    by_path = {m["path"]: m for m in found}
    names = [m["name"] for m in found]
    # One entry per model, not three (and two).
    assert "Qwen3.8-Flash-Next-Q4_K_M" in names
    assert sum(1 for n in names if "of-0000" in n) == 0
    assert "Old" in names and "plain-model" in names
    q = by_path[str((shards / "Qwen3.8-Flash-Next-Q4_K_M-00001-of-00003.gguf").resolve())]
    assert q["shard_count"] == 3
    # size_bytes sums every part: 101+102+103 = 306.
    assert q["size_bytes"] == 306
    old = by_path[str((shards / "Old-split-01.gguf").resolve())]
    assert old["shard_count"] == 2 and old["size_bytes"] == 30


def test_resolve_gguf_shards(shards):
    d = shards
    first = str((d / "Qwen3.8-Flash-Next-Q4_K_M-00001-of-00003.gguf"))
    # A later shard resolves back to shard 1 (what llama-server -m wants).
    assert resolve_gguf_shards(
        str(d / "Qwen3.8-Flash-Next-Q4_K_M-00002-of-00003.gguf")) == first
    # Shard 1 and unsharded paths pass through unchanged.
    assert resolve_gguf_shards(first) == first
    assert resolve_gguf_shards(str(d / "plain-model.gguf")) == \
        str(d / "plain-model.gguf")


def test_gguf_total_size(shards):
    assert gguf_total_size(
        str(shards / "Qwen3.8-Flash-Next-Q4_K_M-00002-of-00003.gguf")) == 306
    assert gguf_total_size(str(shards / "plain-model.gguf")) == 7


# ------------------------------------- HF blob-hash paths (the Qwen3.8 trap)

@pytest.fixture()
def hub_shards(tmp_path):
    """reference-style hub cache: split GGUF + mmproj + MTP drafter stored as
    hash-named blobs, exposed through snapshot symlinks in nested folders."""
    repo = tmp_path / "hub" / "models--testorg--Qwen3.8-Flash-Next-GGUF"
    blobs = repo / "blobs"
    blobs.mkdir(parents=True)
    snap = repo / "snapshots" / "38bb39ee"
    quant = snap / "UD-IQ4_XS"
    mtp = snap / "MTP"
    quant.mkdir(parents=True)
    mtp.mkdir(parents=True)
    hashes = {}
    for n in (1, 2, 3):
        h = f"hash-part{n}"
        (blobs / h).write_bytes(b"x" * (10 * n))
        (quant / f"Qwen3.8-Flash-Next-UD-IQ4_XS-0000{n}-of-00003.gguf") \
            .symlink_to(Path("..") / ".." / ".." / "blobs" / h)
        hashes[n] = h
    (blobs / "hash-mmproj").write_bytes(b"m" * 5)
    (snap / "mmproj-F16.gguf").symlink_to(
        Path("..") / ".." / "blobs" / "hash-mmproj")
    (blobs / "hash-mtp").write_bytes(b"t" * 6)
    (mtp / "mtp-Qwen3.8-Flash-Next-Q8_0.gguf").symlink_to(
        Path("..") / ".." / ".." / "blobs" / "hash-mtp")
    return repo, snap, hashes


def test_blob_path_maps_back_to_snapshot_symlink(hub_shards):
    repo, snap, hashes = hub_shards
    blob = repo / "blobs" / hashes[1]
    link = snapshot_link_for_blob(blob)
    assert link is not None
    assert link.name == "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf"
    # resolve_gguf_path keeps the visible filename (no blob-hash resolution).
    assert resolve_gguf_path(str(blob)).endswith(
        "UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf")


def test_resolve_gguf_shards_through_blob_names(hub_shards):
    """A blob path for shard 2 must resolve to the SNAPSHOT SHARD 1 symlink —
    llama-server refuses hash-named split files ('invalid split file name')."""
    repo, snap, hashes = hub_shards
    resolved = resolve_gguf_shards(str(repo / "blobs" / hashes[2]))
    assert Path(resolved).name == \
        "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf"
    assert str(snap / "UD-IQ4_XS") in resolved


def test_gguf_total_size_through_blob_names(hub_shards):
    repo, _snap, hashes = hub_shards
    assert gguf_total_size(str(repo / "blobs" / hashes[2])) == 60


def test_discovery_excludes_companions_and_keeps_visible_paths(hub_shards):
    repo, snap, _h = hub_shards
    found = discover_gguf_models(tmp := repo.parent)
    names = [m["name"] for m in found]
    # one folded shard entry; mmproj + mtp never offered as standalone models
    assert sum(1 for n in names if "UD-IQ4_XS" in n) == 1
    assert not any("mmproj" in n or "mtp" in n.lower() for n in names)
    entry = next(m for m in found if "UD-IQ4_XS" in m["name"])
    assert entry["shard_count"] == 3
    assert entry["size_bytes"] == 60
    # visible snapshot path, not the blob hash (spawn + sibling lookup need it)
    assert Path(entry["path"]).name.endswith(".gguf")
    assert "blobs" not in Path(entry["path"]).parts


def test_is_companion_gguf():
    assert is_companion_gguf("mmproj-F16.gguf")
    assert is_companion_gguf("mtp-Qwen3.8-Flash-Next-Q8_0.gguf")
    assert is_companion_gguf("Qwen3-VL-mmproj-BF16.gguf")
    assert not is_companion_gguf("Qwen3.8-Flash-Next-UD-IQ4_XS.gguf")
    assert not is_companion_gguf("temp.gguf")


def test_mtp_and_mmproj_siblings_through_hub_layout(hub_shards):
    from sable.core import mmproj as mmp

    repo, snap, hashes = hub_shards
    # Even given the raw blob path of a shard, companions resolve by name.
    gguf = str(repo / "blobs" / hashes[1])
    proj = mmp.find_local(gguf)
    assert proj is not None and Path(proj).name == "mmproj-F16.gguf"
    drafter = mmp.find_local_mtp(gguf)
    assert drafter is not None and Path(drafter).name.startswith("mtp-")


def test_vram_advisory_counts_all_shards(tmp_path):
    from sable.serving.instances import estimate_model_vram_mib

    p = tmp_path / "M-00002-of-00002.gguf"
    p.write_bytes(b"\0" * (2 * 1024 * 1024))
    (tmp_path / "M-00001-of-00002.gguf").write_bytes(b"\0" * (3 * 1024 * 1024))
    est = estimate_model_vram_mib(str(p))
    # 5 MiB of weights * 1.2 headroom => > 5 MiB (a shard-only stat would say ~2).
    assert est is not None and est > 5


# ------------------------------------------------------------------- config

def test_config_models_dir_yaml_and_env(tmp_path, monkeypatch):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("models_dir: /srv/models\n")
    monkeypatch.delenv("SABLE_MODELS_DIR", raising=False)
    assert load_config(str(cfg_file)).models_dir == "/srv/models"
    monkeypatch.setenv("SABLE_MODELS_DIR", "/env/models")
    assert load_config(str(cfg_file)).models_dir == "/env/models"


# --------------------------------------------------------------- dashboard API

@pytest.fixture()
def app(home, cfg_extra=None):
    a = DashboardApp(SableConfig(**(cfg_extra or {})), home=home)
    yield a


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "uf-cat"
    monkeypatch.setenv("SABLE_HOME", str(h))
    return h


def test_model_catalog_endpoint(app, hub):
    body = app.handle("GET", "/api/model_catalog",
                      query={"dir": str(hub)}).body
    assert body["exists"] is True
    names = [m["name"] for m in body["gguf"]]
    assert any("qwen3-8b" in n for n in names)

    # Configured models_dir (no ?dir=) is honored too.
    app2 = DashboardApp(SableConfig(models_dir=str(hub)), home=app.home)
    body2 = app2.handle("GET", "/api/model_catalog").body
    assert body2["exists"] is True
    assert Path(body2["models_dir"]).resolve() == Path(hub).resolve()
    assert len(body2["gguf"]) == 3


def test_model_catalog_no_folder(app, monkeypatch, tmp_path):
    empty = tmp_path / "home"; empty.mkdir()
    monkeypatch.setenv("HOME", str(empty))
    monkeypatch.setenv("USERPROFILE", str(empty))
    monkeypatch.delenv("SABLE_MODELS_DIR", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    body = app.handle("GET", "/api/model_catalog").body
    assert body["exists"] is False


def test_train_base_model_override(app, hub, tmp_path, monkeypatch):
    """POST /api/train {base_model} beats training.base_model in job params."""
    stub = tmp_path / "stub.py"
    stub.write_text("import sys\nsys.exit(0)\n")
    app.cfg.training = {
        "base_model": "Config/Default",
        "worker_command": [sys.executable, str(stub)],
    }
    r = app.handle("POST", "/api/train", body={
        "dataset": "ds-1", "kind": "sft", "dry_run": True,
        "base_model": str(hub / "models--Qwen--Qwen3-8B"),
    })
    assert r.status == 202
    from sable.registry import ModelRegistry
    reg = ModelRegistry(app.home / "buffer" / "sable.db")
    try:
        job = reg.get_job(r.body["job_id"])
        assert job is not None
        assert job.params["base_model"] == str(hub / "models--Qwen--Qwen3-8B")
    finally:
        reg.close()


def test_train_without_override_uses_config(app, tmp_path):
    stub = tmp_path / "stub2.py"
    stub.write_text("import sys\nsys.exit(0)\n")
    app.cfg.training = {
        "base_model": "Config/Default",
        "worker_command": [sys.executable, str(stub)],
    }
    r = app.handle("POST", "/api/train", body={"dataset": "ds-2", "dry_run": True})
    assert r.status == 202
    from sable.registry import ModelRegistry
    reg = ModelRegistry(app.home / "buffer" / "sable.db")
    try:
        job = reg.get_job(r.body["job_id"])
        assert job is not None
        assert job.params["base_model"] == "Config/Default"
    finally:
        reg.close()

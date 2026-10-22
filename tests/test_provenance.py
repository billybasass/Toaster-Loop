"""Artifact provenance attestation (inspiration pass 3, item #23).

core/provenance hashing + blocks; importer manifest carries resource hashes;
registry round-trips provenance_json (including migration from a pre-column
DB); pipeline parses the worker's resource_provenance log event.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sable.core.provenance import (  # noqa: E402
    LOAD_MODE_PREQUANTIZED,
    LOAD_MODE_RUNTIME_4BIT,
    LOAD_MODE_UNQUANTIZED,
    PROVENANCE_VERSION,
    attest_hub_repo,
    attest_path,
    file_entry,
    hash_file,
    parse_worker_provenance,
    worker_provenance_event,
)


# ------------------------------------------------------------- core module

def test_hash_file_matches_hashlib(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"hello weights")
    assert hash_file(f) == hashlib.sha256(b"hello weights").hexdigest()
    assert hash_file(tmp_path / "missing") is None


def test_file_entry_size_and_hash(tmp_path):
    f = tmp_path / "a.jsonl"
    f.write_text("row\n")
    e = file_entry("a.jsonl", f)
    assert e["relpath"] == "a.jsonl"
    assert e["size"] == 4
    assert e["sha256"] == hashlib.sha256(b"row\n").hexdigest()


def test_attest_path_file(tmp_path):
    f = tmp_path / "src.jsonl"
    f.write_text("data")
    b = attest_path(f)
    assert b["version"] == PROVENANCE_VERSION
    assert b["status"] == "attested"
    assert len(b["files"]) == 1 and b["files"][0]["sha256"]


def test_attest_path_directory_sorted_recursive(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "b.txt").write_text("b")
    (tmp_path / "sub" / "a.txt").write_text("aa")
    b = attest_path(tmp_path)
    assert b["status"] == "attested"
    assert [e["relpath"] for e in b["files"]] == ["b.txt", "sub/a.txt"]


def test_attest_path_missing_is_incomplete_with_reason(tmp_path):
    b = attest_path(tmp_path / "gone")
    assert b["status"] == "incomplete"
    assert "missing" in b["reason"]
    assert b["files"] == []


def test_attest_hub_repo_honest_incomplete():
    b = attest_hub_repo("org/model", filename="model.gguf", etag="abc123")
    assert b["status"] == "incomplete"
    assert b["source"] == "hf:org/model"
    assert b["files"][0]["relpath"] == "model.gguf"
    assert b["files"][0]["etag"] == "abc123"


def test_attest_path_load_mode_recorded(tmp_path):
    f = tmp_path / "m.gguf"
    f.write_text("w")
    b = attest_path(f, load_mode=LOAD_MODE_PREQUANTIZED)
    assert b["load_mode"] == LOAD_MODE_PREQUANTIZED


# ---------------------------------------------------- worker event + parse

def test_worker_event_shape_and_parse(tmp_path):
    log = tmp_path / "job.jsonl"
    ev = worker_provenance_event(LOAD_MODE_RUNTIME_4BIT, "org/base",
                                 extra={"requested_4bit": True})
    lines = [
        json.dumps({"event": "phase", "phase": "load_model"}),
        json.dumps(ev),
        json.dumps({"event": "sft_done", "train_loss": 1.0}),
    ]
    log.write_text("\n".join(lines) + "\n")
    got = parse_worker_provenance(str(log))
    assert got is not None
    assert got["load_mode"] == LOAD_MODE_RUNTIME_4BIT
    assert got["source"] == "org/base"
    assert got["requested_4bit"] is True


def test_parse_tolerant_shapes(tmp_path):
    assert parse_worker_provenance(None) is None
    assert parse_worker_provenance(str(tmp_path / "none.jsonl")) is None
    # Garbage + unrelated events -> None (older workers).
    log = tmp_path / "old.jsonl"
    log.write_text("not json\n" + json.dumps({"event": "sft_done"}) + "\n")
    assert parse_worker_provenance(str(log)) is None
    # Last occurrence wins.
    log2 = tmp_path / "two.jsonl"
    log2.write_text("\n".join([
        json.dumps(worker_provenance_event(LOAD_MODE_UNQUANTIZED, "a")),
        json.dumps(worker_provenance_event(LOAD_MODE_RUNTIME_4BIT, "b")),
    ]) + "\n")
    got2 = parse_worker_provenance(str(log2))
    assert got2 is not None and got2["source"] == "b"


# ---------------------------------------------------------------- registry

def _fresh_db(path: Path) -> sqlite3.Connection:
    """A models table WITHOUT provenance_json (simulates a pre-pass3 DB)."""
    conn = sqlite3.connect(str(path))
    conn.execute("""
        CREATE TABLE models (
            model_id TEXT PRIMARY KEY, base_model TEXT NOT NULL,
            parent_adapter TEXT, dataset_id TEXT, manifest_hash TEXT,
            adapter_dir TEXT, gguf_path TEXT, quant TEXT,
            status TEXT NOT NULL CHECK (status IN
              ('candidate','validated','promoted','rolled_back','failed','superseded','archived')),
            created_at TEXT NOT NULL, validation_json TEXT,
            train_loss REAL, eval_loss REAL, promoted_at TEXT,
            pinned INTEGER NOT NULL DEFAULT 0, pre_archive_paths TEXT
        )""")
    conn.execute(
        "INSERT INTO models (model_id, base_model, status, created_at) "
        "VALUES ('v1','base','promoted','2026-01-01T00:00:00+00:00')")
    conn.commit()
    return conn


def test_registry_migrates_and_roundtrips(tmp_path):
    from sable.registry.registry import ModelRegistry

    db = tmp_path / "registry.db"
    conn = _fresh_db(db)
    conn.close()

    reg = ModelRegistry(db)
    row = reg.get_model("v1")
    assert row is not None and row.provenance is None  # tolerant read

    block = attest_hub_repo("org/base")
    reg.set_provenance("v1", block)
    row2 = reg.get_model("v1")
    assert row2 is not None and row2.provenance is not None
    assert row2.provenance["source"] == "hf:org/base"

    # Empty provenance never clobbers prior attestation.
    reg.set_provenance("v1", None)
    row3 = reg.get_model("v1")
    assert row3 is not None and row3.provenance is not None
    reg.close()


def test_registry_status_check_rebuild_keeps_provenance(tmp_path):
    """The #16 CHECK-rebuild path must survive the new last column (#23)."""
    from sable.registry.registry import ModelRegistry

    db = tmp_path / "registry.db"
    conn = _fresh_db(db)
    # Rewrite stored DDL with a pre-'archived' CHECK to trigger the rebuild.
    conn.execute("DROP TABLE models")
    conn.execute("""
        CREATE TABLE models (
            model_id TEXT PRIMARY KEY, base_model TEXT NOT NULL,
            parent_adapter TEXT, dataset_id TEXT, manifest_hash TEXT,
            adapter_dir TEXT, gguf_path TEXT, quant TEXT,
            status TEXT NOT NULL CHECK (status IN
              ('candidate','validated','promoted','rolled_back','failed','superseded')),
            created_at TEXT NOT NULL, validation_json TEXT,
            train_loss REAL, eval_loss REAL, promoted_at TEXT,
            pinned INTEGER NOT NULL DEFAULT 0, pre_archive_paths TEXT
        )""")
    conn.execute(
        "INSERT INTO models (model_id, base_model, status, created_at) "
        "VALUES ('v1','base','promoted','2026-01-01T00:00:00+00:00')")
    conn.commit()
    conn.close()

    reg = ModelRegistry(db)          # provenance ALTER + CHECK rebuild run
    reg.set_provenance("v1", attest_hub_repo("org/m"))
    row = reg.get_model("v1")
    assert row is not None and row.provenance is not None
    assert row.provenance["status"] == "incomplete"
    reg.close()


# ---------------------------------------------------------------- importer

ALPACA_ROW = json.dumps({
    "instruction": "say hi", "input": "", "output": "hi"}) + "\n"


def test_import_manifest_carries_source_and_dataset_hashes(tmp_path, monkeypatch):
    monkeypatch.setenv("SABLE_HOME", str(tmp_path / "home"))
    from sable.curate.importer import import_dataset

    src = tmp_path / "data.jsonl"
    src.write_text(ALPACA_ROW * 3)
    res = import_dataset(src, home=tmp_path / "home")
    mf = json.loads(
        (tmp_path / "home" / "datasets" / res.dataset_id / "manifest.json")
        .read_text())
    rp = mf["resource_provenance"]
    assert rp["source"]["status"] == "attested"
    assert rp["source"]["files"][0]["sha256"] == \
        hashlib.sha256((ALPACA_ROW * 3).encode()).hexdigest()
    assert rp["dataset"]["status"] == "attested"
    assert rp["dataset"]["files"][0]["relpath"] == "sft.jsonl"


# ------------------------------------------------- pipeline glue (unit-level)

def test_dataset_file_inventory(tmp_path):
    from sable.training.pipeline import _dataset_file_inventory

    (tmp_path / "sft.jsonl").write_text("a\n")
    (tmp_path / "train.yaml").write_text("x")   # not hashed
    inv = _dataset_file_inventory(tmp_path)
    assert [e["relpath"] for e in inv] == ["sft.jsonl"]
    assert inv[0]["sha256"] == hashlib.sha256(b"a\n").hexdigest()

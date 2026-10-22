"""Pass 4 #42 — dataset ingestion quality gate (trust, dedupe, conflicts).

Pre-commit gate in the importer (idea adapted from PIASystem (MIT)
app/core/knowledge_ingestion.py): exact + normalized dedupe against existing
datasets via hashes.json sidecars, trust-tier conflict detection, flag-only
semantics (rows are NEVER dropped), flags.json sidecar keeping sft.jsonl
byte-canonical ShareGPT, `sable datasets lint` reporting/refreshing, and
trust on manifests (curated / judge_scored / imported; legacy = imported).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sable.cli as cli  # noqa: E402
from sable.curate.importer import import_dataset  # noqa: E402
from sable.curate.ingest_gate import (  # noqa: E402
    dataset_trust,
    lint_datasets,
    load_flags_sidecar,
    load_hashes_sidecar,
    normalize_compare_key,
    trust_rank,
)


@pytest.fixture()
def sable_home(tmp_path, monkeypatch):
    home = tmp_path / "uf-gate"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SABLE_HOME", str(home))
    monkeypatch.delenv("SABLE_CONFIG", raising=False)
    return home


def write_jsonl(path: Path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


def sft_rows(ds_dir: Path):
    return [json.loads(l) for l in (ds_dir / "sft.jsonl").read_text()
            .splitlines() if l.strip()]


def flags_of(ds_dir: Path):
    data = load_flags_sidecar(ds_dir)
    assert data is not None, "flags.json sidecar missing"
    return data["flags"]


def set_trust(ds_dir: Path, trust: Optional[str]):
    mf = json.loads((ds_dir / "manifest.json").read_text())
    if trust is None:
        mf.pop("trust", None)
    else:
        mf["trust"] = trust
    (ds_dir / "manifest.json").write_text(json.dumps(mf))


def gate_of(res):
    """The importer always attaches a GateReport; narrow for the type checker."""
    assert res.gate is not None
    return res.gate


# ------------------------------------------------------------ exact dedupe

def test_exact_dupe_across_two_imports_flagged_not_dropped(sable_home):
    a = write_jsonl(sable_home / "a.jsonl", [
        {"instruction": "What is the capital of France?", "output": "Paris."},
    ])
    res1 = import_dataset(a, name="one", home=sable_home)
    assert res1.rows == 1
    # first import: sidecars written, nothing flagged
    assert load_hashes_sidecar(res1.dataset_dir) is not None
    assert flags_of(res1.dataset_dir) == {}

    b = write_jsonl(sable_home / "b.jsonl", [
        {"instruction": "What is the capital of France?", "output": "Paris."},
    ])
    res2 = import_dataset(b, name="two", home=sable_home)
    # flagged, NOT dropped: the row still lands in sft.jsonl
    assert res2.rows == 1 and len(sft_rows(res2.dataset_dir)) == 1
    assert gate_of(res2).exact_dupes == 1
    fl = flags_of(res2.dataset_dir)
    assert list(fl) == ["0"]
    assert fl["0"][0].startswith("exact_duplicate:")
    assert f"{res1.dataset_id}:0" in fl["0"][0]


def test_exact_dupe_within_batch(sable_home):
    a = write_jsonl(sable_home / "a.jsonl", [
        {"instruction": "same", "output": "same"},
        {"instruction": "same", "output": "same"},
        {"instruction": "other", "output": "answer"},
    ])
    res = import_dataset(a, home=sable_home)
    assert res.rows == 3  # nothing dropped
    assert gate_of(res).exact_dupes == 1
    fl = flags_of(res.dataset_dir)
    assert set(fl) == {"1"} and "exact_duplicate:batch:0" in fl["1"][0]


# --------------------------------------------------------- normalized dupes

def test_normalized_variant_flagged_near_duplicate(sable_home):
    a = write_jsonl(sable_home / "a.jsonl", [
        {"instruction": "Add 2 and 2.", "output": "4"},
    ])
    res1 = import_dataset(a, name="norm", home=sable_home)
    assert gate_of(res1).exact_dupes == 0

    b = write_jsonl(sable_home / "b.jsonl", [
        # case + whitespace + punctuation differ -> exact miss, norm hit
        {"instruction": "  ADD 2  and 2 ?! ", "output": "4"},
    ])
    res2 = import_dataset(b, name="norm2", home=sable_home)
    assert res2.rows == 1  # kept
    assert gate_of(res2).exact_dupes == 0
    assert gate_of(res2).near_dupes == 1
    fl = flags_of(res2.dataset_dir)
    assert fl["0"][0].startswith("near_duplicate:")

    assert normalize_compare_key("  Add 2,  and\t2!! ") == "add 2 and 2"


def test_distinct_rows_no_false_flags(sable_home):
    a = write_jsonl(sable_home / "a.jsonl", [
        {"instruction": f"question number {i}", "output": f"answer {i}"}
        for i in range(5)
    ])
    res = import_dataset(a, name="distinct", home=sable_home)
    assert res.rows == 5
    assert gate_of(res).exact_dupes == 0
    assert gate_of(res).near_dupes == 0
    assert gate_of(res).conflicts == 0 and gate_of(res).notes == 0
    assert flags_of(res.dataset_dir) == {}


# --------------------------------------------------------------- conflicts

def test_conflict_against_higher_trust_row_flagged_with_reference(sable_home):
    a = write_jsonl(sable_home / "a.jsonl", [
        {"instruction": "What is two plus two?", "output": "four"},
    ])
    res1 = import_dataset(a, name="truth", home=sable_home)
    set_trust(res1.dataset_dir, "curated")  # pretend internal curation

    b = write_jsonl(sable_home / "b.jsonl", [
        {"instruction": "what is two plus two?", "output": "seventeen, obviously"},
    ])
    res2 = import_dataset(b, name="doubt", home=sable_home)
    assert res2.rows == 1  # kept — flag-only gate
    assert gate_of(res2).conflicts == 1 and gate_of(res2).notes == 0
    fl = flags_of(res2.dataset_dir)
    assert fl["0"] == [f"conflicts_with:{res1.dataset_id}:0"]


def test_conflict_equal_trust_informational_only(sable_home):
    a = write_jsonl(sable_home / "a.jsonl", [
        {"instruction": "What is two plus two?", "output": "four"},
    ])
    res1 = import_dataset(a, name="src1", home=sable_home)

    b = write_jsonl(sable_home / "b.jsonl", [
        {"instruction": "What is two plus two?", "output": "seventeen, obviously"},
    ])
    res2 = import_dataset(b, name="src2", home=sable_home)
    assert res2.rows == 1
    assert gate_of(res2).conflicts == 0
    assert gate_of(res2).notes == 1
    fl = flags_of(res2.dataset_dir)
    assert fl["0"] == [f"conflict_note:{res1.dataset_id}:0"]


def test_same_input_similar_output_not_a_conflict(sable_home):
    a = write_jsonl(sable_home / "a.jsonl", [
        {"instruction": "spell color", "output": "colour"},
    ])
    import_dataset(a, name="c1", home=sable_home)
    b = write_jsonl(sable_home / "b.jsonl", [
        {"instruction": "spell color", "output": "colouur"},  # near-identical
    ])
    res2 = import_dataset(b, name="c2", home=sable_home)
    assert gate_of(res2).conflicts == 0 and gate_of(res2).notes == 0


# ------------------------------------------------------------ trust tiers

def test_trust_precedence_ordering():
    assert trust_rank("curated") > trust_rank("judge_scored") \
        > trust_rank("imported")
    assert trust_rank(None) == trust_rank("imported")
    assert trust_rank("bogus") == trust_rank("imported")


def test_legacy_manifest_defaults_to_imported(sable_home):
    a = write_jsonl(sable_home / "a.jsonl", [
        {"instruction": "i", "output": "o"}])
    res = import_dataset(a, name="t", home=sable_home)
    assert json.loads((res.dataset_dir / "manifest.json")
                      .read_text())["trust"] == "imported"
    set_trust(res.dataset_dir, None)  # simulate a pre-#42 manifest
    assert dataset_trust(res.dataset_dir) == "imported"


def test_curater_writes_curated_and_judge_scored(sable_home):
    from sable.curate import CurationConfig, Curater
    from sable.store.db import TraceRow, TraceStore

    store = TraceStore(sable_home / "buffer" / "sable.db")
    try:
        def seed(tid: str, judge=None):
            p = sable_home / "buffer" / f"{tid}.jsonl"
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w") as f:
                f.write(json.dumps({"kind": "trace_header"}) + "\n")
                f.write(json.dumps({"kind": "message", "message": {
                    "role": "user", "content": f"task {tid}"}}) + "\n")
                f.write(json.dumps({"kind": "message", "message": {
                    "role": "assistant", "content": f"done {tid} result=ok"}})
                    + "\n")
            store.insert_trace(TraceRow(
                trace_id=tid, session_id=None,
                timestamp="2026-09-01T10:00:00+00:00",
                agent_profile="fallback", model_alias="base-model",
                model_version=None, endpoint_label=None, task_summary=tid,
                status="success", outcome_score=0.8, num_tool_calls=0,
                tokens_in=10, tokens_out=20, duration_s=1.0,
                jsonl_path=str(p), failure_reason=None, curated=False,
                dataset_id=None, project=None))
            if judge is not None:
                store.add_feedback(tid, score=judge, comment="judge: ok",
                                   source="judge",
                                   timestamp="2026-09-01T10:01:00+00:00")

        seed("plain")
        res = Curater(store, sable_home).curate(CurationConfig(), name="c")
        mf = json.loads((res.dataset_dir / "manifest.json").read_text())
        assert mf["trust"] == "curated"

        seed("judged", judge=0.9)
        res2 = Curater(store, sable_home).curate(CurationConfig(), name="j")
        mf2 = json.loads((res2.dataset_dir / "manifest.json").read_text())
        assert mf2["trust"] == "judge_scored"
    finally:
        store.close()


# ------------------------------------------------- curater/train byte-compat

def test_gated_dataset_stays_canonical_sharegpt(sable_home):
    a = write_jsonl(sable_home / "a.jsonl", [
        {"instruction": "q1", "output": "a1"},
        {"instruction": "q1", "output": "a1"},   # in-batch dupe -> flagged
    ])
    res = import_dataset(a, name="compat", home=sable_home)
    rows = sft_rows(res.dataset_dir)
    assert len(rows) == 2 and gate_of(res).exact_dupes == 1
    for row in rows:
        # exactly the canonical shape — no flag columns leak into the data
        assert set(row.keys()) == {"conversations"}
        for turn in row["conversations"]:
            assert set(turn.keys()) == {"from", "value"}
    # sidecars are separate files; sft.jsonl untouched by the gate
    assert (res.dataset_dir / "flags.json").is_file()
    assert (res.dataset_dir / "hashes.json").is_file()


# ------------------------------------------------------------------- lint

def test_datasets_lint_idempotent_and_refreshes_sidecars(sable_home):
    a = write_jsonl(sable_home / "a.jsonl", [
        {"instruction": "capital of France?", "output": "Paris."},
    ])
    res1 = import_dataset(a, name="fr", home=sable_home)
    b = write_jsonl(sable_home / "b.jsonl", [
        {"instruction": "CAPITAL OF FRANCE", "output": "Paris."},  # near-dupe
    ])
    res2 = import_dataset(b, name="fr2", home=sable_home)
    assert gate_of(res2).near_dupes == 1

    # simulate legacy datasets: no sidecars at all
    for d in (res1.dataset_dir, res2.dataset_dir):
        (d / "hashes.json").unlink()
        (d / "flags.json").unlink()
        assert load_hashes_sidecar(d) is None

    reports = lint_datasets(sable_home / "datasets")
    by_id = {r["dataset_id"]: r for r in reports}
    assert set(by_id) == {res1.dataset_id, res2.dataset_id}
    # lint re-checks every dataset against ALL others: the normalized pair is
    # mutual, so both sides report the collision (import-time flagging only
    # flags the incoming batch — documented asymmetry).
    assert by_id[res2.dataset_id]["near_dupes"] == 1
    assert by_id[res1.dataset_id]["near_dupes"] == 1
    assert f"{res2.dataset_id}:0" in by_id[res1.dataset_id]["details"][0]

    # sidecars refreshed on the legacy datasets
    for d in (res1.dataset_dir, res2.dataset_dir):
        sc = load_hashes_sidecar(d)
        assert sc is not None and sc["row_count"] == 1
        assert load_flags_sidecar(d) is not None

    # idempotent: a second lint reports exactly the same thing
    first = {r["dataset_id"]: (r["exact_dupes"], r["near_dupes"],
                               r["conflicts"], r["notes"], r["details"])
             for r in lint_datasets(sable_home / "datasets")}
    again = {r["dataset_id"]: (r["exact_dupes"], r["near_dupes"],
                               r["conflicts"], r["notes"], r["details"])
             for r in lint_datasets(sable_home / "datasets")}
    assert first == again
    assert first[res2.dataset_id][1] == 1


def test_datasets_lint_name_filter(sable_home):
    a = write_jsonl(sable_home / "a.jsonl", [
        {"instruction": "one", "output": "uno"}])
    res = import_dataset(a, name="alpha", home=sable_home)
    reports = lint_datasets(sable_home / "datasets", name="alpha")
    assert [r["dataset_id"] for r in reports] == [res.dataset_id]
    assert lint_datasets(sable_home / "datasets", name="zzz") == []


def test_cli_import_prints_gate_summary_and_lint(sable_home, capsys):
    src = write_jsonl(sable_home / "seed.jsonl", [
        {"instruction": "hello there", "output": "hi"},
    ])
    assert cli.main(["datasets", "import", str(src), "--name", "g"]) == 0
    out = capsys.readouterr().out
    assert "gate:" in out and "n_rows=1" in out and "exact_dupes=0" in out

    dup = write_jsonl(sable_home / "dup.jsonl", [
        {"instruction": "hello there", "output": "hi"},
    ])
    assert cli.main(["datasets", "import", str(dup), "--name", "g2"]) == 0
    out2 = capsys.readouterr().out
    assert "exact_dupes=1" in out2 and "flag:" in out2

    assert cli.main(["datasets", "lint"]) == 0
    out3 = capsys.readouterr().out
    assert "trust=imported" in out3 and "linted 2 dataset(s)" in out3


# ------------------------------------------------------------ dashboard API

def test_dashboard_datasets_expose_trust_and_lint(sable_home):
    from sable.core.config import SableConfig
    from sable.dashboard.app import DashboardApp

    a = write_jsonl(sable_home / "a.jsonl", [
        {"instruction": "dash q", "output": "dash a"},
    ])
    res = import_dataset(a, name="ds", home=sable_home)
    app = DashboardApp(SableConfig(), home=sable_home)

    listed = app.handle("GET", "/api/datasets").body["datasets"]
    mine = [d for d in listed if d["dataset_id"] == res.dataset_id]
    assert len(mine) == 1
    assert mine[0]["trust"] == "imported"
    assert mine[0]["lint"] == {"flagged_rows": 0, "reasons": {}}

    single = app.handle("GET", f"/api/datasets/{res.dataset_id}").body
    assert single["trust"] == "imported" and single["lint"]["flagged_rows"] == 0

    # legacy dataset without sidecars -> honest unlinted (lint: null)
    (res.dataset_dir / "flags.json").unlink()
    listed = app.handle("GET", "/api/datasets").body["datasets"]
    mine = [d for d in listed if d["dataset_id"] == res.dataset_id][0]
    assert mine["lint"] is None

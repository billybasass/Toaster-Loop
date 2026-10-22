"""Phase 3 tests — dataset curation (SFT/DPO export, dedup, claiming).

All hermetic: SABLE_HOME points at a tmp dir; traces are seeded directly via
the store API (no agent runs needed except where the fake server is handy).
Run: pytest tests/ -v
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sable.cli as cli  # noqa: E402
from sable.curate import (  # noqa: E402
    Curater,
    CurationConfig,
    messages_to_sharegpt,
)
from sable.store.db import TraceRow, TraceStore  # noqa: E402


@pytest.fixture()
def sable_home(tmp_path, monkeypatch):
    home = tmp_path / "uf-home"
    monkeypatch.setenv("SABLE_HOME", str(home))
    monkeypatch.delenv("SABLE_CONFIG", raising=False)
    return home


@pytest.fixture()
def store(sable_home):
    st = TraceStore(sable_home / "buffer" / "sable.db")
    yield st
    st.close()


# ------------------------------------------------------------------- seeding

def seed_trace(
    store: TraceStore,
    home: Path,
    *,
    trace_id: str,
    messages,
    status: str = "success",
    score: float | None = 0.5,
    timestamp: str = "2026-09-01T10:00:00+00:00",
    model_alias: str = "test-model",
    agent_profile: str = "fallback",
    user_feedback: float | None = None,
) -> TraceRow:
    """Write a JSONL transcript + index row exactly like the Phase 2 recorder."""
    jsonl_path = home / "buffer" / "2026" / "09" / f"{trace_id}.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "x", encoding="utf-8") as f:
        f.write(json.dumps({
            "kind": "trace_header", "schema_version": 1, "trace_id": trace_id,
            "session_id": None, "timestamp": timestamp,
            "agent_profile": agent_profile, "model_alias": model_alias,
            "model_version": None, "endpoint_label": None,
            "task": (messages[0].get("content") if messages else "") or "",
        }) + "\n")
        for m in messages:
            f.write(json.dumps({"kind": "message", "message": m}) + "\n")
        f.write(json.dumps({
            "kind": "trace_footer", "schema_version": 1, "status": status,
            "completed": status == "success", "outcome_score": score,
            "num_tool_calls": sum(len(m.get("tool_calls") or []) for m in messages),
            "tokens_in": None, "tokens_out": None, "duration_s": None,
        }) + "\n")

    row = TraceRow(
        trace_id=trace_id, session_id=None, timestamp=timestamp,
        agent_profile=agent_profile, model_alias=model_alias,
        model_version=None, endpoint_label=None,
        task_summary="seed", status=status, outcome_score=score,
        num_tool_calls=0, tokens_in=None, tokens_out=None, duration_s=None,
        jsonl_path=str(jsonl_path),
    )
    store.insert_trace(row)
    if score is not None:
        store.add_feedback(trace_id, score, comment="heuristic v0",
                           source="heuristic", timestamp=timestamp)
    if user_feedback is not None:
        store.add_feedback(trace_id, user_feedback, comment=None,
                           source="user", timestamp=timestamp)
    return row


SIMPLE_OK = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"},
]

TOOLCALL_TRACE = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "list files"},
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_1", "type": "function",
         "function": {"name": "shell", "arguments": '{"cmd": "ls"}'}},
    ]},
    {"role": "tool", "tool_call_id": "call_1", "content": "a.txt\nb.txt"},
    {"role": "assistant", "content": "Files: a.txt and b.txt"},
]


# ------------------------------------------------------- ShareGPT rendering

def test_sharegpt_basic_roles():
    conv = messages_to_sharegpt(SIMPLE_OK)
    assert conv == [
        {"from": "system", "value": "You are helpful."},
        {"from": "human", "value": "What is 2+2?"},
        {"from": "gpt", "value": "< think>\n< /think>\n4"},
    ]


def test_sharegpt_tool_call_rendering_round_trip():
    """Tool-call rendering matches the format conventions; payloads round-trip."""
    conv = messages_to_sharegpt(TOOLCALL_TRACE)
    assert [t["from"] for t in conv] == ["system", "human", "gpt", "tool", "gpt"]

    # gpt turn with tool call: think block + < tool_call> JSON payload.
    tc_turn = conv[2]["value"]
    assert tc_turn.startswith("< think>\n< /think>\n")
    marker_open, marker_close = "< tool_call>\n", "\n< /tool_call>"
    assert marker_open in tc_turn and marker_close in tc_turn
    payload_txt = tc_turn.split(marker_open, 1)[1].split(marker_close, 1)[0]
    payload = json.loads(payload_txt)
    assert payload == {"name": "shell", "arguments": {"cmd": "ls"}}

    # tool turn: one < tool_response> block with tool_call_id/name/content;
    # JSON-looking content is parsed to an object.
    tr_turn = conv[3]["value"]
    assert tr_turn.startswith("< tool_response>\n")
    assert tr_turn.endswith("\n< /tool_response>")
    body = json.loads(tr_turn[len("< tool_response>\n"): -len("\n< /tool_response>")])
    assert body == {"tool_call_id": "call_1", "name": "shell",
                    "content": "a.txt\nb.txt"}

    # final gpt answer keeps a think block.
    assert conv[4]["value"] == "< think>\n< /think>\nFiles: a.txt and b.txt"


def test_sharegpt_multiple_tool_calls_collapse_into_one_tool_turn():
    msgs = [
        {"role": "user", "content": "do two"},
        {"role": "assistant", "tool_calls": [
            {"id": "c1", "function": {"name": "a", "arguments": "{}"}},
            {"id": "c2", "function": {"name": "b", "arguments": '{"x": 1}'}},
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": "r1"},
        {"role": "tool", "tool_call_id": "c2", "content": '{"ok": true}'},
        {"role": "assistant", "content": "done"},
    ]
    conv = messages_to_sharegpt(msgs)
    assert [t["from"] for t in conv] == ["human", "gpt", "tool", "gpt"]
    blocks = conv[2]["value"].split("\n< /tool_response>\n< tool_response>\n")
    assert len(blocks) == 2
    b0 = json.loads(blocks[0][len("< tool_response>\n"):])
    b1 = json.loads(blocks[1][: -len("\n< /tool_response>")])
    assert (b0["name"], b1["name"]) == ("a", "b")
    assert b1["content"] == {"ok": True}  # JSON content parsed


# ----------------------------------------------------------------- dedup

def test_dedup_prefix_overlap_keeps_longest(store, sable_home):
    base = SIMPLE_OK[:2]
    seed_trace(store, sable_home, trace_id="short", messages=SIMPLE_OK)
    long_msgs = base + [
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "and squared?"},
        {"role": "assistant", "content": "16"},
    ]
    seed_trace(store, sable_home, trace_id="long", messages=long_msgs,
               timestamp="2026-09-01T11:00:00+00:00")
    # Different conversation -> kept separately.
    seed_trace(store, sable_home, trace_id="other", messages=[
        {"role": "user", "content": "totally different"},
        {"role": "assistant", "content": "ok"},
    ])

    res = cli.main(["curate", "--name", "dd"])
    assert res == 0
    rows = [json.loads(l) for l in
            next((sable_home / "datasets").rglob("sft.jsonl")).read_text().splitlines()]
    tids = {r["trace_id"] for r in rows}
    assert tids == {"long", "other"}  # 'short' collapsed into its continuation


# ------------------------------------------------------------------ DPO

def test_dpo_pairs_mined_from_group(store, sable_home):
    good = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "same task"},
        {"role": "assistant", "content": "great answer"},
    ]
    bad = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "same task"},
        {"role": "assistant", "content": "terrible answer"},
    ]
    seed_trace(store, sable_home, trace_id="good", messages=good, score=1.0)
    seed_trace(store, sable_home, trace_id="bad", messages=bad,
               status="failure", score=-1.0,
               timestamp="2026-09-01T09:00:00+00:00")

    # failure traces only enter the pool as DPO negatives when requested
    assert cli.main(["curate", "--name", "dp", "--include-failures"]) == 0
    dpo_path = next((sable_home / "datasets").rglob("dpo.jsonl"))
    rows = [json.loads(l) for l in dpo_path.read_text().splitlines()]
    assert len(rows) == 1
    r = rows[0]
    assert r["chosen"] == "great answer"
    assert r["rejected"] == "terrible answer"
    # TRL PreferenceDataset shape: prompt is a role/content message list.
    assert all(set(m) >= {"role", "content"} for m in r["prompt"])
    assert r["prompt"][-1] == {"role": "user", "content": "same task"}
    assert r["metadata"]["chosen_trace_id"] == "good"
    assert r["metadata"]["rejected_trace_id"] == "bad"


def test_dpo_empty_case_warns_and_emits_valid_empty_file(store, sable_home):
    seed_trace(store, sable_home, trace_id="solo", messages=SIMPLE_OK)
    assert cli.main(["curate", "--name", "empty"]) == 0
    dpo_path = next((sable_home / "datasets").rglob("dpo.jsonl"))
    assert dpo_path.read_text() == ""  # valid empty JSONL
    manifest = json.loads((dpo_path.parent / "manifest.json").read_text())
    assert any("no DPO preference pairs" in w for w in manifest["warnings"])


def test_dpo_tie_yields_no_pairs(store, sable_home):
    a = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    b = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "b"}]
    seed_trace(store, sable_home, trace_id="t1", messages=a, score=0.5)
    seed_trace(store, sable_home, trace_id="t2", messages=b, score=0.5)
    assert cli.main(["curate"]) == 0
    dpo_path = next((sable_home / "datasets").rglob("dpo.jsonl"))
    assert dpo_path.read_text() == ""


# ------------------------------------------- remediation weighting (pass 4 #33)
# Idea adapted from PIASystem (MIT) app/core/failure_taxonomy.py — a RESOLVED
# failure note newer than the trace means the bug behind that failure_reason
# was fixed after the trace ran, so it must not be mined as bad-data DPO
# negative material. Other statuses stay display-only.

def _seed_failure_pair(store, sable_home, *, reason="max_iterations",
                       agent_profile="fallback"):
    """Same-conversation good/bad pair; the failure carries a runner reason."""
    good = [{"role": "user", "content": "same task"},
            {"role": "assistant", "content": "great answer"}]
    bad = [{"role": "user", "content": "same task"},
           {"role": "assistant", "content": "terrible answer"}]
    seed_trace(store, sable_home, trace_id="good", messages=good, score=1.0)
    row = seed_trace(store, sable_home, trace_id="bad", messages=bad,
                     status="failure", score=-1.0,
                     timestamp="2026-09-01T09:00:00+00:00",
                     agent_profile=agent_profile)
    # Direct UPDATE: insert_trace is INSERT ... ON CONFLICT DO NOTHING (#15),
    # so a re-insert no longer patches the existing row.
    store._conn.execute(
        "UPDATE traces SET failure_reason = ? WHERE trace_id = ?",
        (reason, "bad"))
    store._conn.commit()
    return row


def _dpo_pairs(store, sable_home):
    """Curate once and return the DPO rows; releases claims afterwards so a
    second call with different notes sees the same candidate pool. Dataset
    ids are sortable timestamps (with -N suffixes), so the newest is last."""
    assert cli.main(["curate", "--name", "remed", "--include-failures"]) == 0
    newest = sorted(d.name for d in (sable_home / "datasets").iterdir())[-1]
    dpo_path = sable_home / "datasets" / newest / "dpo.jsonl"
    rows = [json.loads(l) for l in dpo_path.read_text().splitlines()]
    for ds_id in store.claimed_dataset_ids():
        store.reset_claims(ds_id)
    return rows


def test_resolved_note_newer_than_trace_not_down_weighted(store, sable_home):
    _seed_failure_pair(store, sable_home)
    # Resolved AFTER the failure trace was recorded -> bug fixed after the
    # fact. set_failure_note stamps ts=now(); traces are seeded 2026-09-01,
    # so any wall-clock "now" is newer than the trace date.
    store.set_failure_note("max_iterations", "resolved", note="prompt fix")
    assert store.get_failure_note("max_iterations")["status"] == "resolved"
    pairs = _dpo_pairs(store, sable_home)
    # the failure is no longer rejected material -> no pair survives
    assert pairs == []


def test_open_or_older_resolved_note_still_down_weights(store, sable_home):
    _seed_failure_pair(store, sable_home)
    store.set_failure_note("max_iterations", "acknowledged", note="todo")
    pairs = _dpo_pairs(store, sable_home)
    assert len(pairs) == 1 and pairs[0]["metadata"]["rejected_trace_id"] == "bad"

    # RESOLVED but stamped BEFORE the trace -> still bad data (conservative).
    store.set_failure_note("max_iterations", "resolved", note="old fix")
    # rewrite ts to predate the trace via direct row surgery on the note
    conn = store._conn
    conn.execute("UPDATE failure_notes SET ts = ? WHERE reason = ?",
                 ("2025-01-01T00:00:00+00:00", "max_iterations"))
    conn.commit()
    pairs = _dpo_pairs(store, sable_home)
    assert len(pairs) == 1 and pairs[0]["metadata"]["rejected_trace_id"] == "bad"


def test_resolved_note_profile_scoping(store, sable_home):
    _seed_failure_pair(store, sable_home, agent_profile="fallback")
    # note scoped to a different profile must not rescue this trace
    store.set_failure_note("max_iterations", "resolved", note="other agent",
                           profile="other-profile")
    pairs = _dpo_pairs(store, sable_home)
    assert len(pairs) == 1 and pairs[0]["metadata"]["rejected_trace_id"] == "bad"

    # global note rescues everywhere; matching-profile note rescues its own
    store.set_failure_note("max_iterations", "resolved", note="fixed")
    pairs = _dpo_pairs(store, sable_home)
    assert pairs == []


# --------------------------------------------------------------- claiming

def test_claiming_idempotency_and_rebuild(store, sable_home):
    seed_trace(store, sable_home, trace_id="t1", messages=SIMPLE_OK)

    assert cli.main(["curate", "--name", "first"]) == 0
    row = store.get_trace("t1")
    assert row.curated is True and row.dataset_id is not None
    ds_id = row.dataset_id

    # Second run: nothing new to claim, dataset has zero rows.
    assert cli.main(["curate", "--name", "second"]) == 0
    datasets = sorted((sable_home / "datasets").iterdir())
    assert len(datasets) == 2
    second_sft = (sable_home / "datasets" / datasets[1].name / "sft.jsonl")
    assert second_sft.read_text() == ""

    # --rebuild releases the claim and rewrites the original dataset.
    assert cli.main(["curate", "--rebuild", ds_id]) == 0
    row = store.get_trace("t1")
    assert row.curated is True and row.dataset_id == ds_id
    first_sft = (sable_home / "datasets" / ds_id / "sft.jsonl")
    rows = [json.loads(l) for l in first_sft.read_text().splitlines()]
    assert [r["trace_id"] for r in rows] == ["t1"]


def test_trace_appears_in_at_most_one_dataset(store, sable_home):
    seed_trace(store, sable_home, trace_id="t1", messages=SIMPLE_OK)
    seed_trace(store, sable_home, trace_id="t2", messages=[
        {"role": "user", "content": "other"},
        {"role": "assistant", "content": "ok"},
    ])
    assert cli.main(["curate"]) == 0
    assert cli.main(["curate"]) == 0
    all_ids = []
    for mf in (sable_home / "datasets").rglob("manifest.json"):
        m = json.loads(mf.read_text())
        all_ids.extend(m["files"]["sft.jsonl"]["trace_ids"])
    assert len(all_ids) == len(set(all_ids)) == 2


# ------------------------------------------------------------- selection

def test_min_score_and_user_feedback_selection(store, sable_home):
    def msg(c):
        return [{"role": "user", "content": c}, {"role": "assistant", "content": "a"}]

    seed_trace(store, sable_home, trace_id="low", messages=msg("lo"), score=0.2)
    seed_trace(store, sable_home, trace_id="mid", messages=msg("mi"), score=0.6)
    seed_trace(store, sable_home, trace_id="rescued", messages=msg("rs"),
               score=0.1, user_feedback=0.9)
    seed_trace(store, sable_home, trace_id="fail", messages=msg("fa"),
               status="failure", score=-1.0)

    assert cli.main(["curate"]) == 0
    sft = next((sable_home / "datasets").rglob("sft.jsonl"))
    ids = {json.loads(l)["trace_id"] for l in sft.read_text().splitlines()}
    assert ids == {"mid", "rescued"}  # low below threshold, failure excluded


def test_include_failures_and_filters(store, sable_home):
    def msg(c):
        return [{"role": "user", "content": c}, {"role": "assistant", "content": "a"}]

    seed_trace(store, sable_home, trace_id="f1", messages=msg("x"),
               status="failure", score=-1.0)
    seed_trace(store, sable_home, trace_id="old", messages=msg("y"),
               timestamp="2026-01-01T00:00:00+00:00")
    seed_trace(store, sable_home, trace_id="old2", messages=msg("w"),
               timestamp="2026-02-01T00:00:00+00:00")

    assert cli.main(["curate", "--include-failures"]) == 0
    sft = next((sable_home / "datasets").rglob("sft.jsonl"))
    ids = {json.loads(l)["trace_id"] for l in sft.read_text().splitlines()}
    # failure with score -1 still fails the min-score gate even when included
    assert "f1" not in ids

    # seed AFTER the first run so the profile/time filters see fresh material
    seed_trace(store, sable_home, trace_id="hm", messages=msg("z"),
               agent_profile="legacy")

    assert cli.main(["curate", "--since", "2026-06-01", "--profile", "legacy"]) == 0
    dirs = sorted((sable_home / "datasets").iterdir(), reverse=True)
    rows = [json.loads(l) for l in
            (dirs[0] / "sft.jsonl").read_text().splitlines()]
    assert {r["trace_id"] for r in rows} == {"hm"}


def test_null_score_not_treated_as_zero(store, sable_home):
    # NULL score + clean success status => included; NULL + errors => excluded.
    seed_trace(store, sable_home, trace_id="nulls", messages=SIMPLE_OK, score=None)
    seed_trace(store, sable_home, trace_id="nullerr", messages=[
        {"role": "user", "content": "q"},
        {"role": "assistant", "tool_calls": [
            {"id": "c", "function": {"name": "t", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c", "content": "error: boom"},
        {"role": "assistant", "content": "gave up"},
    ], score=None)
    assert cli.main(["curate"]) == 0
    sft = next((sable_home / "datasets").rglob("sft.jsonl"))
    ids = {json.loads(l)["trace_id"] for l in sft.read_text().splitlines()}
    assert ids == {"nulls"}


def test_limit_caps_rows(store, sable_home):
    for i in range(4):
        seed_trace(store, sable_home, trace_id=f"t{i}", messages=[
            {"role": "user", "content": f"q{i}"},
            {"role": "assistant", "content": f"a{i}"},
        ], timestamp=f"2026-09-01T10:0{i}:00+00:00")
    assert cli.main(["curate", "--limit", "2"]) == 0
    sft = next((sable_home / "datasets").rglob("sft.jsonl"))
    rows = [json.loads(l) for l in sft.read_text().splitlines()]
    assert len(rows) == 2


# ------------------------------------------------------- manifest & yaml

def test_manifest_completeness_and_train_yaml(store, sable_home):
    seed_trace(store, sable_home, trace_id="t1", messages=TOOLCALL_TRACE)
    assert cli.main(["curate", "--name", "mf"]) == 0
    mf_path = next((sable_home / "datasets").rglob("manifest.json"))
    m = json.loads(mf_path.read_text())

    for key in ("dataset_id", "created_at", "dataset_schema_version",
                "trace_schema_version", "filters", "files", "format_hints"):
        assert key in m
    assert m["files"]["sft.jsonl"]["count"] == 1
    assert m["files"]["sft.jsonl"]["trace_ids"] == ["t1"]
    assert m["files"]["sft.jsonl"]["format"] == "sharegpt"
    assert m["format_hints"]["sft_format_type"] == "sharegpt"
    assert m["filters"]["min_score"] == pytest.approx(0.5)

    tyaml = mf_path.parent / "train.yaml"
    doc = yaml.safe_load(tyaml.read_text())  # must parse as YAML
    for section in ("model", "data", "training", "lora", "logging"):
        assert section in doc
    assert doc["data"]["format_type"] == "sharegpt"
    assert str(doc["data"]["local_dataset"][0]).endswith("sft.jsonl")
    assert Path(doc["data"]["local_dataset"][0]).is_file()


def test_datasets_cli_list_show_export(store, sable_home, capsys):
    seed_trace(store, sable_home, trace_id="t1", messages=SIMPLE_OK)
    assert cli.main(["datasets", "list"]) == 0
    assert "no datasets curated yet" in capsys.readouterr().out

    assert cli.main(["curate", "--name", "exp"]) == 0
    ds_id = store.get_trace("t1").dataset_id

    assert cli.main(["datasets", "list"]) == 0
    out = capsys.readouterr().out
    assert ds_id in out and "sft=1" in out

    assert cli.main(["datasets", "show", ds_id]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["dataset_id"] == ds_id

    dest = sable_home.parent / "export-dest"
    assert cli.main(["datasets", "export", ds_id, "--to", str(dest)]) == 0
    assert (dest / ds_id / "sft.jsonl").is_file()
    assert (dest / ds_id / "manifest.json").is_file()
    # re-export refuses to clobber
    assert cli.main(["datasets", "export", ds_id, "--to", str(dest)]) == 1
    assert cli.main(["datasets", "show", "nosuchdataset"]) == 1


# ------------------------------------------------------------ golden fixture

GOLDEN_SFT_ROW = {
    "conversations": [
        {"from": "system", "value": "You are helpful."},
        {"from": "human", "value": "list files"},
        {"from": "gpt", "value": (
            "< think>\n< /think>\n"
            "< tool_call>\n"
            + json.dumps({"name": "shell", "arguments": {"cmd": "ls"}},
                         ensure_ascii=False)
            + "\n< /tool_call>"
        )},
        {"from": "tool", "value": (
            "< tool_response>\n"
            + json.dumps({"tool_call_id": "call_1", "name": "shell",
                          "content": "a.txt\nb.txt"}, ensure_ascii=False)
            + "\n< /tool_response>"
        )},
        {"from": "gpt", "value": "< think>\n< /think>\nFiles: a.txt and b.txt"},
    ],
}


def test_sft_golden_fixture(store, sable_home):
    seed_trace(store, sable_home, trace_id="gold", messages=TOOLCALL_TRACE)
    assert cli.main(["curate"]) == 0
    sft = next((sable_home / "datasets").rglob("sft.jsonl"))
    rows = [json.loads(l) for l in sft.read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["conversations"] == GOLDEN_SFT_ROW["conversations"]
    # provenance extras present but conversations is the training payload
    assert row["trace_id"] == "gold"
    assert row["model"] == "test-model"
    assert row["completed"] is True


# ------------------------------------------------- claim race (finding #8)

def test_concurrent_curates_produce_disjoint_manifests(store, sable_home):
    """Two curates racing over one candidate pool never share a trace ID.

    Claim-before-write makes this structural: each trace flips curated=0->1
    atomically (BEGIN IMMEDIATE + guarded UPDATE), so exactly one of the two
    datasets can list it. Threads use separate TraceStore handles on the same
    DB file — the realistic two-process shape under WAL."""
    n = 12
    for i in range(n):
        seed_trace(store, sable_home, trace_id=f"t{i}", messages=[
            {"role": "user", "content": f"question {i}"},
            {"role": "assistant", "content": f"answer {i}"},
        ], timestamp=f"2026-09-01T10:00:{i:02d}+00:00")

    db = sable_home / "buffer" / "sable.db"
    store_a = TraceStore(db)
    store_b = TraceStore(db)
    barrier = threading.Barrier(2)
    results: list = []
    errors: list = []

    def run(st):
        try:
            barrier.wait(timeout=10)
            results.append(Curater(st, sable_home).curate(
                CurationConfig(formats="sft"), name="race"))
        except Exception as e:  # noqa: BLE001 - surfaced below
            errors.append(e)

    threads = [threading.Thread(target=run, args=(s,)) for s in (store_a, store_b)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    store_a.close()
    store_b.close()
    assert not errors, f"concurrent curate raised: {errors!r}"
    assert len(results) == 2

    ids_a = set(results[0].sft_trace_ids)
    ids_b = set(results[1].sft_trace_ids)
    assert ids_a & ids_b == set()          # disjoint: no trace in both
    assert ids_a | ids_b == {f"t{i}" for i in range(n)}  # every trace shipped once

    # The manifests on disk agree with the returned results.
    mf_ids = []
    for res in results:
        mf = json.loads((res.dataset_dir / "manifest.json").read_text())
        assert mf["short_claims"] + len(mf["files"]["sft.jsonl"]["trace_ids"]) \
            == n  # won + lost = selected, exactly once each
        mf_ids.append(set(mf["files"]["sft.jsonl"]["trace_ids"]))
    assert mf_ids[0] & mf_ids[1] == set()


def test_short_claim_is_warned_and_reported(store, sable_home, caplog,
                                            monkeypatch):
    """Traces lost to a concurrent claim: WARNING log + result/manifest field."""
    seed_trace(store, sable_home, trace_id="mine", messages=[
        {"role": "user", "content": "q1"}, {"role": "assistant", "content": "a"}])
    seed_trace(store, sable_home, trace_id="theirs", messages=[
        {"role": "user", "content": "q2"}, {"role": "assistant", "content": "b"}])

    # Deterministic interleaving of two curates: 'theirs' is selected fine
    # (still unclaimed at query time), then a competing run wins the atomic
    # claim just before ours lands. The wrapper plays that competitor inside
    # claim_traces_owned -- exactly the window BEGIN IMMEDIATE serializes.
    real_claim = store.claim_traces_owned

    def racing_claim(trace_ids, dataset_id):
        if "theirs" in trace_ids and dataset_id != "other-ds":
            assert real_claim(["theirs"], "other-ds") == ["theirs"]
        return real_claim(trace_ids, dataset_id)

    monkeypatch.setattr(store, "claim_traces_owned", racing_claim)

    with caplog.at_level(logging.WARNING, logger="sable.curate"):
        res = Curater(store, sable_home).curate(CurationConfig(), name="sc")

    assert res.short_claims == 1
    assert res.claimed == 1
    assert res.sft_trace_ids == ["mine"]           # 'theirs' excluded entirely
    assert any("short claim" in w for w in res.warnings)
    assert any("short claim" in r.getMessage() and "1 of 2" in r.getMessage()
               for r in caplog.records if r.levelno >= logging.WARNING)

    mf = json.loads((res.dataset_dir / "manifest.json").read_text())
    assert mf["short_claims"] == 1
    assert "theirs" not in mf["files"]["sft.jsonl"]["trace_ids"]
    # the winner's claim is untouched by our release/cleanup paths
    assert store.get_trace("theirs").dataset_id == "other-ds"


def test_write_failure_releases_claims(store, sable_home, monkeypatch):
    """An error after claiming returns every trace to the buffer."""
    seed_trace(store, sable_home, trace_id="t1", messages=SIMPLE_OK)

    def boom(*a, **k):
        raise RuntimeError("disk on fire")

    # Fails inside the write block, after claims landed and manifest bytes
    # were prepared — exactly the window finding #8 is about.
    monkeypatch.setattr("sable.curate.curater.render_train_yaml", boom)
    with pytest.raises(RuntimeError, match="disk on fire"):
        Curater(store, sable_home).curate(CurationConfig(), name="boom")
    monkeypatch.undo()  # the re-curation below must render for real

    row = store.get_trace("t1")
    assert row.curated is False and row.dataset_id is None
    # partial dataset dir cleaned up; nothing half-written is discoverable
    assert not any((sable_home / "datasets").rglob("manifest.json"))

    # ...and the trace is curatable again.
    res = Curater(store, sable_home).curate(CurationConfig(), name="again")
    assert res.sft_trace_ids == ["t1"]
    assert res.claimed == 1
    assert (res.dataset_dir / "train.yaml").is_file()

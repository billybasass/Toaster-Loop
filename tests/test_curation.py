"""Curation/evaluation domain tests.

Consolidated from 10 former test modules; section banners mark origins.
"""

from __future__ import annotations

import sys
from pathlib import Path
import json
import logging
import threading
import yaml
from typing import Optional
import argparse
import re
from typing import Any, Dict, Optional
import shutil
import sqlite3
from contextlib import contextmanager
import hashlib
from typing import Dict, Optional
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sable.cli as cli  # noqa: E402
from sable.curate import Curater, CurationConfig, messages_to_sharegpt  # noqa: E402
from sable.store.db import TraceRow, TraceStore  # noqa: E402
from sable.curate.importer import ImportError_, detect_format, import_dataset  # noqa: E402
from sable.curate.importer import import_dataset  # noqa: E402
from sable.curate.ingest_gate import dataset_trust, lint_datasets, load_flags_sidecar, load_hashes_sidecar, normalize_compare_key, trust_rank  # noqa: E402
from fake_server import FakeOpenAIServer, make_message_handler  # noqa: E402
from sable.curate.judge import JUDGE_RUBRIC, TOOL_VIEW_CAP, build_judge_messages, judge_trace, parse_judge_reply, select_borderline  # noqa: E402
from sable.core.config import GoldensMineConfig, load_config  # noqa: E402
from sable.curate.mine_goldens import MineGoldensError, load_mined_goldens, merge_mined_into_tasks, mined_path, run_mine_goldens, task_pair  # noqa: E402
from sable.registry import ModelRegistry  # noqa: E402
from sable.training.rejection import make_rft_job_runner, select_prompts  # noqa: E402
from sable.curate.mine_goldens import make_mine_goldens_job_runner  # noqa: E402
from sable.training.scheduler import JobDispatcher, Scheduler, TrainingConfig  # noqa: E402
from sable.training.validate import run_golden_suite  # noqa: E402
from sable.core.config import SableConfig  # noqa: E402
from sable.curate.dataset_diff import diff_datasets  # noqa: E402
from sable.dashboard.app import DashboardApp  # noqa: E402
from sable.registry.registry import ModelRegistry  # noqa: E402
from sable.training.pipeline import eval_regression_check, parse_worker_losses  # noqa: E402
from sable.training.worker import EVAL_MIN_ROWS, EVAL_MIN_TOTAL_ROWS, eval_event, find_named_eval_files, find_resumable_checkpoint, make_protocol_callback, maybe_eval_split, plan_cadence, progress_event, resolve_resume_path  # noqa: E402
from sable.evaluation.evolve import APPLIED_KEY, DISMISSED_KEY, MUTATION_DIRECTIONS, RESULT_KEY, EvolveError, apply_prompt_to_config, length_cap_for, make_evolve_job_runner, run_evolution, validate_variant  # noqa: E402
from sable.training.scheduler import Scheduler, TrainingConfig  # noqa: E402
from sable.core.provenance import LOAD_MODE_PREQUANTIZED, LOAD_MODE_RUNTIME_4BIT, LOAD_MODE_UNQUANTIZED, PROVENANCE_VERSION, attest_hub_repo, attest_path, file_entry, hash_file, parse_worker_provenance, worker_provenance_event  # noqa: E402
from sable.curate import Curater, CurationConfig  # noqa: E402
from sable.store import analytics  # noqa: E402

# ------------------------------------------------------------------------
# from tests/test_curate.py
# ------------------------------------------------------------------------

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

# ------------------------------------------------------------------------
# from tests/test_importer.py
# ------------------------------------------------------------------------

@pytest.fixture()
def sable_home__test_importer(tmp_path, monkeypatch):
    home = tmp_path / "uf-import"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SABLE_HOME", str(home))
    monkeypatch.delenv("SABLE_CONFIG", raising=False)
    return home


def write_jsonl(path: Path, rows, raw: str | None = None):
    if raw is not None:
        path.write_text(raw, encoding="utf-8")
    else:
        path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


# ------------------------------------------------------------------- detection

def test_detect_sharegpt():
    assert detect_format([{"conversations": [{"from": "human", "value": "x"},
                                             {"from": "gpt", "value": "y"}]}]) \
        == "sharegpt"


def test_detect_alpaca():
    assert detect_format([{"instruction": "i", "input": "", "output": "o"}]) \
        == "alpaca"


def test_detect_chatml():
    assert detect_format([{"messages": [{"role": "user", "content": "x"},
                                        {"role": "assistant", "content": "y"}]}]) \
        == "chatml"


def test_detect_ambiguity_error_shows_keys_seen():
    with pytest.raises(ImportError_) as e:
        detect_format([{"prompt": "p", "completion": "c"}])
    msg = str(e.value)
    assert "completion" in msg and "prompt" in msg  # keys seen, clearly shown


def test_detect_empty_raises():
    with pytest.raises(ImportError_):
        detect_format([])


# -------------------------------------------------------------- normalization

def _sft_rows(ds_dir: Path):
    return [json.loads(l) for l in
            (ds_dir / "sft.jsonl").read_text().splitlines()]


def test_golden_alpaca_pair(sable_home__test_importer):
    src = write_jsonl(sable_home__test_importer / "a.jsonl", [
        {"instruction": "Add 2 and 2.", "input": "", "output": "4"},
    ])
    res = import_dataset(src, home=sable_home__test_importer)
    (row,) = _sft_rows(res.dataset_dir)
    # exact internal shape: ShareGPT turns with the gpt think-block invariant
    assert row == {"conversations": [
        {"from": "human", "value": "Add 2 and 2."},
        {"from": "gpt", "value": "< think>\n< /think>\n4"},
    ]}


def test_golden_alpaca_with_input_prefix(sable_home__test_importer):
    src = write_jsonl(sable_home__test_importer / "a.jsonl", [
        {"instruction": "Translate.", "input": "Hello", "output": "Hola"},
    ])
    res = import_dataset(src, home=sable_home__test_importer)
    (row,) = _sft_rows(res.dataset_dir)
    assert row["conversations"][0]["value"] == "Hello\n\nTranslate."


def test_golden_chatml_passthrough(sable_home__test_importer):
    src = write_jsonl(sable_home__test_importer / "c.jsonl", [
        {"messages": [{"role": "system", "content": "s"},
                      {"role": "user", "content": "u"},
                      {"role": "assistant", "content": "a"}]},
    ])
    res = import_dataset(src, home=sable_home__test_importer)
    (row,) = _sft_rows(res.dataset_dir)
    assert row["conversations"] == [
        {"from": "system", "value": "s"},
        {"from": "human", "value": "u"},
        {"from": "gpt", "value": "< think>\n< /think>\na"},
    ]


def test_golden_sharegpt_internal_keys(sable_home__test_importer):
    src = write_jsonl(sable_home__test_importer / "s.jsonl", [
        {"conversations": [{"from": "human", "value": "q"},
                           {"from": "gpt", "value": "< think>\n< /think>\na"}]},
    ])
    res = import_dataset(src, home=sable_home__test_importer)
    (row,) = _sft_rows(res.dataset_dir)
    # from/value round-trips byte-identically through the internal builder
    assert row["conversations"] == [
        {"from": "human", "value": "q"},
        {"from": "gpt", "value": "< think>\n< /think>\na"},
    ]


def test_sharegpt_role_content_keys_accepted(sable_home__test_importer):
    src = write_jsonl(sable_home__test_importer / "s.jsonl", [
        {"conversations": [{"role": "user", "content": "q"},
                           {"role": "assistant", "content": "a"}]},
    ])
    res = import_dataset(src, home=sable_home__test_importer)
    (row,) = _sft_rows(res.dataset_dir)
    assert [t["from"] for t in row["conversations"]] == ["human", "gpt"]


def test_chatml_bad_role_row_skipped(sable_home__test_importer):
    src = write_jsonl(sable_home__test_importer / "c.jsonl", [
        {"messages": [{"role": "wizard", "content": "?"}]},
        {"messages": [{"role": "user", "content": "u"},
                      {"role": "assistant", "content": "a"}]},
    ])
    res = import_dataset(src, home=sable_home__test_importer)
    assert res.rows == 1 and res.skipped_rows == 1


# ------------------------------------------------------- malformed + manifest

def test_malformed_rows_skipped_and_counted(sable_home__test_importer):
    src = sable_home__test_importer / "m.jsonl"
    src.write_text(
        '{"instruction": "ok", "output": "y"}\n'
        'not json at all\n'
        '{"instruction": "", "output": ""}\n'   # parses, normalizes to nothing
        '"just a string"\n'
        '{"instruction": "ok2", "output": "y2"}\n',
        encoding="utf-8")
    res = import_dataset(src, home=sable_home__test_importer)
    assert res.rows == 2
    assert res.skipped_rows == 3  # bad json + empty pair + bare string
    mf = json.loads((res.dataset_dir / "manifest.json").read_text())
    assert mf["skipped_rows"] == 3


def test_manifest_provenance_fields(sable_home__test_importer):
    src = write_jsonl(sable_home__test_importer / "pub.jsonl", [
        {"instruction": "i", "output": "o"}])
    res = import_dataset(src, name="pub", home=sable_home__test_importer)
    mf = json.loads((res.dataset_dir / "manifest.json").read_text())
    assert mf["provenance"] == "imported"
    assert mf["source_file"].endswith("pub.jsonl")
    assert mf["source_format"] == "alpaca"
    assert mf["dataset_id"] == res.dataset_id and "--pub" in res.dataset_id
    assert mf["created_at"]
    assert mf["files"]["sft.jsonl"]["count"] == 1
    assert mf["files"]["sft.jsonl"]["trace_ids"] == []  # no trace provenance
    assert mf["files"]["dpo.jsonl"]["count"] == 0
    assert (res.dataset_dir / "train.yaml").is_file()


def test_json_array_input_accepted(sable_home__test_importer):
    src = sable_home__test_importer / "arr.json"
    src.write_text(json.dumps([{"instruction": "i", "output": "o"}]))
    res = import_dataset(src, home=sable_home__test_importer)
    assert res.rows == 1


def test_unreadable_and_undetectable_raise(sable_home__test_importer):
    with pytest.raises(ImportError_):
        import_dataset(sable_home__test_importer / "missing.jsonl", home=sable_home__test_importer)
    bad = sable_home__test_importer / "bad.jsonl"
    write_jsonl(bad, [{"foo": 1}])
    with pytest.raises(ImportError_):
        import_dataset(bad, home=sable_home__test_importer)


# -------------------------------------------------------------------- CLI

def test_cli_import_and_list(sable_home__test_importer):
    src = write_jsonl(sable_home__test_importer / "seed.jsonl", [
        {"instruction": "i1", "output": "o1"},
        {"garbage": True},
    ])
    assert cli.main(["datasets", "import", str(src), "--name", "seed"]) == 0
    out = cli.main(["datasets", "list"])
    assert out == 0


def test_cli_import_undetectable_exits_nonzero(sable_home__test_importer, capsys):
    bad = sable_home__test_importer / "bad.jsonl"
    write_jsonl(bad, [{"weird": 1}])
    assert cli.main(["datasets", "import", str(bad)]) != 0
    assert "error:" in capsys.readouterr().err


# ------------------------------------- e2e: imported dataset trains via stub

STUB = r'''
import argparse, json, os, sys
from pathlib import Path
ap = argparse.ArgumentParser()
for f in (("--job", True), ("--kind", True), ("--config", False),
          ("--dataset-dir", False), ("--base-model", False),
          ("--log-path", False), ("--adapter-dir", False),
          ("--gguf-out", False)):
    ap.add_argument(f[0], required=f[1], default=None)
ap.add_argument("--quant", default="q4_k_m")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

def emit(ev, **kw):
    line = json.dumps({"event": ev, "job": a.job, **kw})
    print(line, flush=True)
    if a.log_path:
        Path(a.log_path).parent.mkdir(parents=True, exist_ok=True)
        open(a.log_path, "a").write(line + "\n")

emit("start")
if a.kind in ("sft", "dpo"):
    # prove the imported sft.jsonl contract is what the worker receives
    rows = (Path(a.dataset_dir) / "sft.jsonl").read_text().splitlines()
    assert rows, "imported dataset produced no rows for the worker"
    first = json.loads(rows[0])
    assert first["conversations"][0]["from"] in ("human", "system")
    out = Path(a.config).parent if a.config else Path(".")
    (out / "adapter").mkdir(parents=True, exist_ok=True)
    (out / "adapter" / "adapter_config.json").write_text(json.dumps(
        {"base_model": a.base_model or "stub", "r": 8}))
elif a.kind == "export":
    gg = Path(a.gguf_out)
    gg.parent.mkdir(parents=True, exist_ok=True)
    fixture = os.environ.get("STUB_FIXTURE_GGUF")
    if fixture and Path(fixture).is_file():
        gg.write_bytes(Path(fixture).read_bytes())
    else:
        gg.write_bytes(b"GGUF-stub")
emit("done")
'''


def test_imported_dataset_trains_end_to_end(sable_home__test_importer, tmp_path, monkeypatch):
    from fake_server import FakeOpenAIServer, make_scripted_handler
    from sable.curate import CurationConfig
    from sable.registry import ModelRegistry
    from sable.store.db import TraceStore
    from sable.training.pipeline import run_iterate
    from sable.training.scheduler import Scheduler, TrainingConfig
    from sable.training.validate import ValidationConfig

    stub = tmp_path / "stub.py"
    stub.write_text(STUB)
    fixture_gguf = tmp_path / "tiny.gguf"
    fixture_gguf.write_bytes(b"GGUF" + b"\x00" * 32)
    monkeypatch.setenv("STUB_FIXTURE_GGUF", str(fixture_gguf))

    src = write_jsonl(sable_home__test_importer / "seed.jsonl", [
        {"instruction": f"task {i}", "output": f"result {i} ok"}
        for i in range(3)
    ])
    res = import_dataset(src, name="e2e", home=sable_home__test_importer)

    store = TraceStore(sable_home__test_importer / "buffer" / "sable.db")
    reg = ModelRegistry(sable_home__test_importer / "buffer" / "sable.db")

    def starter(gguf_path):
        srv = FakeOpenAIServer(handler_cls=make_scripted_handler(
            {"*": "ok-answer"}))
        srv.start()
        return srv

    try:
        sched = Scheduler(registry=reg, cfg=TrainingConfig(),
                          worker_command=[sys.executable, str(stub)])
        result = run_iterate(
            home=sable_home__test_importer, registry=reg, scheduler=sched, store=store,
            curation_cfg=CurationConfig(),
            training_cfg=TrainingConfig(base_model="Base/HF-Model"),
            validation_cfg=ValidationConfig(
                min_pass_rate=1.0,
                golden_tasks=[{"prompt": "x", "expect_contains": ["ok-answer"]}]),
            dataset_id=res.dataset_id,   # skip curate: train the IMPORT
            server_starter=starter,
        )
    finally:
        store.close()
        reg.close()

    assert result.ok and result.promoted, result.to_dict()
    assert result.dataset_id == res.dataset_id
    assert "train" in result.stages and "export" in result.stages

# ------------------------------------------------------------------------
# from tests/test_ingest_gate.py
# ------------------------------------------------------------------------

@pytest.fixture()
def sable_home__test_ingest_gate(tmp_path, monkeypatch):
    home = tmp_path / "uf-gate"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SABLE_HOME", str(home))
    monkeypatch.delenv("SABLE_CONFIG", raising=False)
    return home


def write_jsonl__test_ingest_gate(path: Path, rows):
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

def test_exact_dupe_across_two_imports_flagged_not_dropped(sable_home__test_ingest_gate):
    a = write_jsonl__test_ingest_gate(sable_home__test_ingest_gate / "a.jsonl", [
        {"instruction": "What is the capital of France?", "output": "Paris."},
    ])
    res1 = import_dataset(a, name="one", home=sable_home__test_ingest_gate)
    assert res1.rows == 1
    # first import: sidecars written, nothing flagged
    assert load_hashes_sidecar(res1.dataset_dir) is not None
    assert flags_of(res1.dataset_dir) == {}

    b = write_jsonl__test_ingest_gate(sable_home__test_ingest_gate / "b.jsonl", [
        {"instruction": "What is the capital of France?", "output": "Paris."},
    ])
    res2 = import_dataset(b, name="two", home=sable_home__test_ingest_gate)
    # flagged, NOT dropped: the row still lands in sft.jsonl
    assert res2.rows == 1 and len(sft_rows(res2.dataset_dir)) == 1
    assert gate_of(res2).exact_dupes == 1
    fl = flags_of(res2.dataset_dir)
    assert list(fl) == ["0"]
    assert fl["0"][0].startswith("exact_duplicate:")
    assert f"{res1.dataset_id}:0" in fl["0"][0]


def test_exact_dupe_within_batch(sable_home__test_ingest_gate):
    a = write_jsonl__test_ingest_gate(sable_home__test_ingest_gate / "a.jsonl", [
        {"instruction": "same", "output": "same"},
        {"instruction": "same", "output": "same"},
        {"instruction": "other", "output": "answer"},
    ])
    res = import_dataset(a, home=sable_home__test_ingest_gate)
    assert res.rows == 3  # nothing dropped
    assert gate_of(res).exact_dupes == 1
    fl = flags_of(res.dataset_dir)
    assert set(fl) == {"1"} and "exact_duplicate:batch:0" in fl["1"][0]


# --------------------------------------------------------- normalized dupes

def test_normalized_variant_flagged_near_duplicate(sable_home__test_ingest_gate):
    a = write_jsonl__test_ingest_gate(sable_home__test_ingest_gate / "a.jsonl", [
        {"instruction": "Add 2 and 2.", "output": "4"},
    ])
    res1 = import_dataset(a, name="norm", home=sable_home__test_ingest_gate)
    assert gate_of(res1).exact_dupes == 0

    b = write_jsonl__test_ingest_gate(sable_home__test_ingest_gate / "b.jsonl", [
        # case + whitespace + punctuation differ -> exact miss, norm hit
        {"instruction": "  ADD 2  and 2 ?! ", "output": "4"},
    ])
    res2 = import_dataset(b, name="norm2", home=sable_home__test_ingest_gate)
    assert res2.rows == 1  # kept
    assert gate_of(res2).exact_dupes == 0
    assert gate_of(res2).near_dupes == 1
    fl = flags_of(res2.dataset_dir)
    assert fl["0"][0].startswith("near_duplicate:")

    assert normalize_compare_key("  Add 2,  and\t2!! ") == "add 2 and 2"


def test_distinct_rows_no_false_flags(sable_home__test_ingest_gate):
    a = write_jsonl__test_ingest_gate(sable_home__test_ingest_gate / "a.jsonl", [
        {"instruction": f"question number {i}", "output": f"answer {i}"}
        for i in range(5)
    ])
    res = import_dataset(a, name="distinct", home=sable_home__test_ingest_gate)
    assert res.rows == 5
    assert gate_of(res).exact_dupes == 0
    assert gate_of(res).near_dupes == 0
    assert gate_of(res).conflicts == 0 and gate_of(res).notes == 0
    assert flags_of(res.dataset_dir) == {}


# --------------------------------------------------------------- conflicts

def test_conflict_against_higher_trust_row_flagged_with_reference(sable_home__test_ingest_gate):
    a = write_jsonl__test_ingest_gate(sable_home__test_ingest_gate / "a.jsonl", [
        {"instruction": "What is two plus two?", "output": "four"},
    ])
    res1 = import_dataset(a, name="truth", home=sable_home__test_ingest_gate)
    set_trust(res1.dataset_dir, "curated")  # pretend internal curation

    b = write_jsonl__test_ingest_gate(sable_home__test_ingest_gate / "b.jsonl", [
        {"instruction": "what is two plus two?", "output": "seventeen, obviously"},
    ])
    res2 = import_dataset(b, name="doubt", home=sable_home__test_ingest_gate)
    assert res2.rows == 1  # kept — flag-only gate
    assert gate_of(res2).conflicts == 1 and gate_of(res2).notes == 0
    fl = flags_of(res2.dataset_dir)
    assert fl["0"] == [f"conflicts_with:{res1.dataset_id}:0"]


def test_conflict_equal_trust_informational_only(sable_home__test_ingest_gate):
    a = write_jsonl__test_ingest_gate(sable_home__test_ingest_gate / "a.jsonl", [
        {"instruction": "What is two plus two?", "output": "four"},
    ])
    res1 = import_dataset(a, name="src1", home=sable_home__test_ingest_gate)

    b = write_jsonl__test_ingest_gate(sable_home__test_ingest_gate / "b.jsonl", [
        {"instruction": "What is two plus two?", "output": "seventeen, obviously"},
    ])
    res2 = import_dataset(b, name="src2", home=sable_home__test_ingest_gate)
    assert res2.rows == 1
    assert gate_of(res2).conflicts == 0
    assert gate_of(res2).notes == 1
    fl = flags_of(res2.dataset_dir)
    assert fl["0"] == [f"conflict_note:{res1.dataset_id}:0"]


def test_same_input_similar_output_not_a_conflict(sable_home__test_ingest_gate):
    a = write_jsonl__test_ingest_gate(sable_home__test_ingest_gate / "a.jsonl", [
        {"instruction": "spell color", "output": "colour"},
    ])
    import_dataset(a, name="c1", home=sable_home__test_ingest_gate)
    b = write_jsonl__test_ingest_gate(sable_home__test_ingest_gate / "b.jsonl", [
        {"instruction": "spell color", "output": "colouur"},  # near-identical
    ])
    res2 = import_dataset(b, name="c2", home=sable_home__test_ingest_gate)
    assert gate_of(res2).conflicts == 0 and gate_of(res2).notes == 0


# ------------------------------------------------------------ trust tiers

def test_trust_precedence_ordering():
    assert trust_rank("curated") > trust_rank("judge_scored") \
        > trust_rank("imported")
    assert trust_rank(None) == trust_rank("imported")
    assert trust_rank("bogus") == trust_rank("imported")


def test_legacy_manifest_defaults_to_imported(sable_home__test_ingest_gate):
    a = write_jsonl__test_ingest_gate(sable_home__test_ingest_gate / "a.jsonl", [
        {"instruction": "i", "output": "o"}])
    res = import_dataset(a, name="t", home=sable_home__test_ingest_gate)
    assert json.loads((res.dataset_dir / "manifest.json")
                      .read_text())["trust"] == "imported"
    set_trust(res.dataset_dir, None)  # simulate a pre-#42 manifest
    assert dataset_trust(res.dataset_dir) == "imported"


def test_curater_writes_curated_and_judge_scored(sable_home__test_ingest_gate):
    from sable.curate import CurationConfig, Curater
    from sable.store.db import TraceRow, TraceStore

    store = TraceStore(sable_home__test_ingest_gate / "buffer" / "sable.db")
    try:
        def seed(tid: str, judge=None):
            p = sable_home__test_ingest_gate / "buffer" / f"{tid}.jsonl"
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
        res = Curater(store, sable_home__test_ingest_gate).curate(CurationConfig(), name="c")
        mf = json.loads((res.dataset_dir / "manifest.json").read_text())
        assert mf["trust"] == "curated"

        seed("judged", judge=0.9)
        res2 = Curater(store, sable_home__test_ingest_gate).curate(CurationConfig(), name="j")
        mf2 = json.loads((res2.dataset_dir / "manifest.json").read_text())
        assert mf2["trust"] == "judge_scored"
    finally:
        store.close()


# ------------------------------------------------- curater/train byte-compat

def test_gated_dataset_stays_canonical_sharegpt(sable_home__test_ingest_gate):
    a = write_jsonl__test_ingest_gate(sable_home__test_ingest_gate / "a.jsonl", [
        {"instruction": "q1", "output": "a1"},
        {"instruction": "q1", "output": "a1"},   # in-batch dupe -> flagged
    ])
    res = import_dataset(a, name="compat", home=sable_home__test_ingest_gate)
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

def test_datasets_lint_idempotent_and_refreshes_sidecars(sable_home__test_ingest_gate):
    a = write_jsonl__test_ingest_gate(sable_home__test_ingest_gate / "a.jsonl", [
        {"instruction": "capital of France?", "output": "Paris."},
    ])
    res1 = import_dataset(a, name="fr", home=sable_home__test_ingest_gate)
    b = write_jsonl__test_ingest_gate(sable_home__test_ingest_gate / "b.jsonl", [
        {"instruction": "CAPITAL OF FRANCE", "output": "Paris."},  # near-dupe
    ])
    res2 = import_dataset(b, name="fr2", home=sable_home__test_ingest_gate)
    assert gate_of(res2).near_dupes == 1

    # simulate legacy datasets: no sidecars at all
    for d in (res1.dataset_dir, res2.dataset_dir):
        (d / "hashes.json").unlink()
        (d / "flags.json").unlink()
        assert load_hashes_sidecar(d) is None

    reports = lint_datasets(sable_home__test_ingest_gate / "datasets")
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
             for r in lint_datasets(sable_home__test_ingest_gate / "datasets")}
    again = {r["dataset_id"]: (r["exact_dupes"], r["near_dupes"],
                               r["conflicts"], r["notes"], r["details"])
             for r in lint_datasets(sable_home__test_ingest_gate / "datasets")}
    assert first == again
    assert first[res2.dataset_id][1] == 1


def test_datasets_lint_name_filter(sable_home__test_ingest_gate):
    a = write_jsonl__test_ingest_gate(sable_home__test_ingest_gate / "a.jsonl", [
        {"instruction": "one", "output": "uno"}])
    res = import_dataset(a, name="alpha", home=sable_home__test_ingest_gate)
    reports = lint_datasets(sable_home__test_ingest_gate / "datasets", name="alpha")
    assert [r["dataset_id"] for r in reports] == [res.dataset_id]
    assert lint_datasets(sable_home__test_ingest_gate / "datasets", name="zzz") == []


def test_cli_import_prints_gate_summary_and_lint(sable_home__test_ingest_gate, capsys):
    src = write_jsonl__test_ingest_gate(sable_home__test_ingest_gate / "seed.jsonl", [
        {"instruction": "hello there", "output": "hi"},
    ])
    assert cli.main(["datasets", "import", str(src), "--name", "g"]) == 0
    out = capsys.readouterr().out
    assert "gate:" in out and "n_rows=1" in out and "exact_dupes=0" in out

    dup = write_jsonl__test_ingest_gate(sable_home__test_ingest_gate / "dup.jsonl", [
        {"instruction": "hello there", "output": "hi"},
    ])
    assert cli.main(["datasets", "import", str(dup), "--name", "g2"]) == 0
    out2 = capsys.readouterr().out
    assert "exact_dupes=1" in out2 and "flag:" in out2

    assert cli.main(["datasets", "lint"]) == 0
    out3 = capsys.readouterr().out
    assert "trust=imported" in out3 and "linted 2 dataset(s)" in out3


# ------------------------------------------------------------ dashboard API

def test_dashboard_datasets_expose_trust_and_lint(sable_home__test_ingest_gate):
    from sable.core.config import SableConfig
    from sable.dashboard.app import DashboardApp

    a = write_jsonl__test_ingest_gate(sable_home__test_ingest_gate / "a.jsonl", [
        {"instruction": "dash q", "output": "dash a"},
    ])
    res = import_dataset(a, name="ds", home=sable_home__test_ingest_gate)
    app = DashboardApp(SableConfig(), home=sable_home__test_ingest_gate)

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

# ------------------------------------------------------------------------
# from tests/test_judge.py
# ------------------------------------------------------------------------

@pytest.fixture()
def sable_home__test_judge(tmp_path, monkeypatch):
    home = tmp_path / "uf-judge"
    monkeypatch.setenv("SABLE_HOME", str(home))
    monkeypatch.delenv("SABLE_CONFIG", raising=False)
    return home


@pytest.fixture()
def store__test_judge(sable_home__test_judge):
    st = TraceStore(sable_home__test_judge / "buffer" / "sable.db")
    yield st
    st.close()


def seed_trace__test_judge(store, home, *, trace_id, score=0.5, n_msgs=3,
               timestamp="2026-09-01T10:00:00+00:00"):
    jsonl = home / "buffer" / "2026" / "09" / f"{trace_id}.jsonl"
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(n_msgs - 1):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": f"{trace_id} turn {i}"})
    msgs[-1] = {"role": "assistant", "content": f"{trace_id} final answer"}
    with open(jsonl, "w", encoding="utf-8") as f:
        f.write(json.dumps({"kind": "trace_header", "trace_id": trace_id}) + "\n")
        for m in msgs:
            f.write(json.dumps({"kind": "message", "message": m}) + "\n")
        f.write(json.dumps({"kind": "trace_footer", "status": "success",
                            "outcome_score": score}) + "\n")
    store.insert_trace(TraceRow(
        trace_id=trace_id, session_id=None, timestamp=timestamp,
        agent_profile="fallback", model_alias="m", model_version=None,
        endpoint_label=None, task_summary="t", status="success",
        outcome_score=score, num_tool_calls=0, tokens_in=None,
        tokens_out=None, duration_s=None, jsonl_path=str(jsonl),
    ))
    if score is not None:
        store.add_feedback(trace_id, score, comment="heuristic v0",
                           source="heuristic", timestamp=timestamp)


# ------------------------------------------------------- bounded prompt build

def test_prompt_bounded_by_msg_count_and_char_caps():
    msgs = [{"role": "user", "content": f"m{i} " + "x" * 5000}
            for i in range(40)]
    built = build_judge_messages(msgs, max_msgs=10, per_msg_cap=800)
    assert len(built) == 2  # system rubric + one user excerpt turn
    assert built[0]["content"] == JUDGE_RUBRIC
    excerpt = built[1]["content"]
    # exactly the LAST 10 messages survive, oldest-first
    assert "m39" in excerpt and "m30" in excerpt and "m29" not in excerpt
    assert excerpt.count("[user]") == 10
    # per-message cap enforced (cap + role tag + truncation marker)
    for line in excerpt.splitlines():
        if line.startswith("[user]"):
            assert len(line) <= 800 + len("[user] ") + 10


def test_tool_outputs_trimmed_harder_in_judge_view():
    msgs = [{"role": "tool", "name": "shell", "content": "y" * 4000},
            {"role": "user", "content": "hi"}]
    excerpt = build_judge_messages(msgs)[1]["content"]
    tool_line = next(l for l in excerpt.splitlines() if l.startswith("[tool:"))
    assert len(tool_line) <= TOOL_VIEW_CAP + 40
    assert tool_line.endswith("[…]")


# ------------------------------------------------------------ forced-JSON parse

@pytest.mark.parametrize("reply,expected", [
    ('{"score": 0.8, "reason": "solid"}', (0.8, "solid")),
    ('```json\n{"score": 0.42, "reason": "fenced"}\n```', (0.42, "fenced")),
    ('Sure! Here is my verdict:\n{"score": 1.0, "reason": "tail"}\nHope that helps.',
     (1.0, "tail")),
])
def test_parse_clean_fenced_and_prose_variants(reply, expected):
    got = parse_judge_reply(reply)
    assert got is not None and (got["score"], got["reason"]) == expected


def test_parse_clamps_and_rejects_garbage():
    clamped = parse_judge_reply('{"score": 5, "reason": "hype"}')
    assert clamped is not None and clamped["score"] == 1.0
    assert parse_judge_reply("no json at all") is None
    assert parse_judge_reply('{"score": "high", "reason": "bad type"}') is None
    assert parse_judge_reply("") is None
    assert parse_judge_reply(None) is None


def test_judge_trace_unparseable_is_skip_not_crash():
    def client(msgs, model):
        return "the model rambles with no JSON"
    assert judge_trace([{"role": "user", "content": "t"}], client, "m") is None

    def boom(msgs, model):
        raise ConnectionError("server down")
    assert judge_trace([{"role": "user", "content": "t"}], boom, "m") is None


# ------------------------------------------------- storage + precedence (#12)

def test_judge_feedback_row_written_and_outcome_score_untouched(store__test_judge):
    seed_trace__test_judge(store__test_judge, Path(store__test_judge.db_path).parent.parent, trace_id="t1", score=0.4)
    store__test_judge.add_feedback("t1", 0.9, comment="judge: good", source="judge",
                       timestamp="2026-09-01T10:01:00+00:00")
    fb = store__test_judge.get_feedback("t1")
    assert [f["source"] for f in fb] == ["heuristic", "judge"]
    # judge never clobbers the record-time heuristic baseline
    assert store__test_judge.get_trace("t1").outcome_score == 0.4


def test_precedence_user_over_judge_over_heuristic(store__test_judge, sable_home__test_judge):
    from sable.curate.curater import load_trace

    seed_trace__test_judge(store__test_judge, sable_home__test_judge, trace_id="h-only", score=0.4)
    seed_trace__test_judge(store__test_judge, sable_home__test_judge, trace_id="judge-wins", score=0.4)
    store__test_judge.add_feedback("judge-wins", 0.8, comment="j", source="judge",
                       timestamp="2026-09-01T10:02:00+00:00")
    seed_trace__test_judge(store__test_judge, sable_home__test_judge, trace_id="user-wins", score=0.4)
    store__test_judge.add_feedback("user-wins", 0.8, comment="j", source="judge",
                       timestamp="2026-09-01T10:02:00+00:00")
    store__test_judge.add_feedback("user-wins", -1.0, comment="u", source="user",
                       timestamp="2026-09-01T10:03:00+00:00")

    eff = {}
    for tid in ("h-only", "judge-wins", "user-wins"):
        row = store__test_judge.get_trace(tid)
        assert row is not None
        loaded = load_trace(store__test_judge, row)
        assert loaded is not None
        eff[tid] = loaded.effective_score
    assert eff["h-only"] == 0.4        # no judge rows -> byte-identical today
    assert eff["judge-wins"] == 0.8    # judge overrides heuristic
    assert eff["user-wins"] == -1.0    # user overrides judge


def test_curater_selection_uses_judge_score(sable_home__test_judge, store__test_judge):
    """A trace below min_score but above it by judge score enters SFT."""
    seed_trace__test_judge(store__test_judge, sable_home__test_judge, trace_id="below", score=0.4)
    seed_trace__test_judge(store__test_judge, sable_home__test_judge, trace_id="above-by-judge", score=0.4)
    store__test_judge.add_feedback("above-by-judge", 0.9, comment="j", source="judge",
                       timestamp="2026-09-01T10:05:00+00:00")
    assert cli.main(["curate", "--name", "jsel"]) == 0
    ds = next((sable_home__test_judge / "datasets").rglob("sft.jsonl"))
    tids = {json.loads(l)["trace_id"] for l in
            ds.read_text().splitlines()}
    assert "above-by-judge" in tids and "below" not in tids


# ----------------------------------------------------------- borderline select

class _T:
    def __init__(self, tid, eff):
        self.effective_score = eff
        self.row = type("R", (), {"trace_id": tid})()


def test_select_borderline_band_and_topn():
    traces = [_T("a", 0.5), _T("b", 0.41), _T("c", 0.62), _T("d", 0.9),
              _T("e", None)]
    picked = [t.row.trace_id for t in select_borderline(traces, 0.5, 0.15, 2)]
    # closest-to-threshold first; d outside band; e NULL never borderline
    assert picked == ["a", "b"]
    assert select_borderline(traces, 0.5, 0.15, 9) and len(
        select_borderline(traces, 0.5, 0.15, 9)) == 3


# ------------------------------------------------------------- CLI --judge N

VERDICT = json.dumps({"score": 0.9, "reason": "helpful"})


def _fake_judge_server(reply=VERDICT):
    handler = make_message_handler({"*": {"role": "assistant", "content": reply}})
    srv = FakeOpenAIServer(handler_cls=handler)
    srv.start()
    return srv, handler


def test_curate_judge_flag_scores_borderline_only(sable_home__test_judge, store__test_judge,
                                                  monkeypatch, capsys):
    seed_trace__test_judge(store__test_judge, sable_home__test_judge, trace_id="border0", score=0.5)
    seed_trace__test_judge(store__test_judge, sable_home__test_judge, trace_id="border1", score=0.4)
    seed_trace__test_judge(store__test_judge, sable_home__test_judge, trace_id="far", score=0.9)

    srv, handler = _fake_judge_server()
    monkeypatch.setattr(cli, "_resolve_endpoint",
                        lambda cfg, spawn_if_needed: (srv.base_url, "judge-model", None))
    try:
        assert cli.main(["curate", "--name", "j", "--judge", "2"]) == 0
    finally:
        srv.stop()

    judged = {r["messages"][-1]["content"] for r in handler.requests}
    assert len(handler.requests) == 2  # top-N=2 only, 'far' untouched
    tids_judged = set()
    for f in ("border0", "border1", "far"):
        srcs = [fb["source"] for fb in store__test_judge.get_feedback(f)]
        if "judge" in srcs:
            tids_judged.add(f)
    assert tids_judged == {"border0", "border1"}
    out = capsys.readouterr().out
    assert "biased tiebreaker" in out  # loud caveat printed once per run
    assert "judge summary: 2 scored, 0 skipped" in out


def test_curate_without_judge_flag_makes_zero_llm_calls(sable_home__test_judge, store__test_judge,
                                                        monkeypatch):
    seed_trace__test_judge(store__test_judge, sable_home__test_judge, trace_id="t", score=0.5)
    srv, handler = _fake_judge_server()

    def spy(*a, **k):
        raise AssertionError("_resolve_endpoint must not be called")
    monkeypatch.setattr(cli, "_resolve_endpoint", spy)
    try:
        assert cli.main(["curate", "--name", "nojudge"]) == 0
        assert handler.requests == []  # fake server untouched
    finally:
        srv.stop()
    for f in ("t",):
        assert [fb["source"] for fb in store__test_judge.get_feedback(f)] == ["heuristic"]


def test_curate_judge_unparseable_reply_skips_not_crashes(sable_home__test_judge, store__test_judge,
                                                          monkeypatch, capsys):
    seed_trace__test_judge(store__test_judge, sable_home__test_judge, trace_id="s", score=0.5)
    srv, handler = _fake_judge_server(reply="I think it went pretty well!")
    monkeypatch.setattr(cli, "_resolve_endpoint",
                        lambda cfg, spawn_if_needed: (srv.base_url, "m", None))
    try:
        assert cli.main(["curate", "--name", "skip", "--judge", "1"]) == 0
    finally:
        srv.stop()
    assert [fb["source"] for fb in store__test_judge.get_feedback("s")] == ["heuristic"]
    out = capsys.readouterr().out
    assert "SKIPPED" in out and "judge summary: 0 scored, 1 skipped" in out

# ------------------------------------------------------------------------
# from tests/test_mine_goldens.py
# ------------------------------------------------------------------------

@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "umine"
    monkeypatch.setenv("SABLE_HOME", str(h))
    return h


@pytest.fixture()
def store__test_mine_goldens(home):
    st = TraceStore(home / "buffer" / "sable.db")
    yield st
    st.close()


def seed_trace__test_mine_goldens(store, home, *, trace_id, messages, status="success",
               thumb=True, timestamp="2026-09-01T10:00:00+00:00"):
    """JSONL transcript + index row + optional user 👍 (recorder shape)."""
    jsonl_path = home / "buffer" / "2026" / "09" / f"{trace_id}.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "x", encoding="utf-8") as f:
        f.write(json.dumps({
            "kind": "trace_header", "schema_version": 1, "trace_id": trace_id,
            "session_id": None, "timestamp": timestamp,
            "agent_profile": "fallback", "model_alias": "test-model",
            "model_version": None, "endpoint_label": None, "task": "seed",
        }) + "\n")
        for m in messages:
            f.write(json.dumps({"kind": "message", "message": m}) + "\n")
        f.write(json.dumps({
            "kind": "trace_footer", "schema_version": 1, "status": status,
            "completed": status == "success", "outcome_score": 0.5,
            "num_tool_calls": 0, "tokens_in": None, "tokens_out": None,
            "duration_s": None,
        }) + "\n")
    store.insert_trace(TraceRow(
        trace_id=trace_id, session_id=None, timestamp=timestamp,
        agent_profile="fallback", model_alias="test-model", model_version=None,
        endpoint_label=None, task_summary="seed", status=status,
        outcome_score=0.5, num_tool_calls=0, tokens_in=None, tokens_out=None,
        duration_s=None, jsonl_path=str(jsonl_path),
    ))
    if thumb:
        store.add_feedback(trace_id, score=1.0, comment=None, source="user",
                           timestamp=timestamp)


def cfg(**kw) -> GoldensMineConfig:
    base = dict(enabled=True)
    base.update(kw)
    return GoldensMineConfig(**base)


GOOD_CONTENT = ("To rename the current branch use git branch -m main.\n"
                "Verify with git branch --show-current afterwards.")

BRANCH_TRACE = [
    {"role": "user", "content": "How do I rename my current git branch to main?"},
    {"role": "assistant", "content": GOOD_CONTENT},
]


def mined_tasks(home):
    p = mined_path(home)
    if not p.is_file():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


# ------------------------------------------------------------- happy path

def test_thumbed_trace_mined_into_suite_and_replayed(home, store__test_mine_goldens):
    """👍 trace -> golden persisted with provenance; merged task list feeds
    run_golden_suite and the golden passes against a server replaying the
    source answer."""
    seed_trace__test_mine_goldens(store__test_mine_goldens, home, trace_id="t1", messages=BRANCH_TRACE)
    res = run_mine_goldens(store__test_mine_goldens, home, cfg())
    assert res.mined == 1 and not res.dropped

    tasks = mined_tasks(home)
    assert len(tasks) == 1
    g = tasks[0]
    assert g["prompt"] == BRANCH_TRACE[0]["content"]
    assert "git branch -m main" in "".join(g["expect_contains"])
    assert g["enabled"] is True
    assert g["_mine"]["source_trace_id"] == "t1"
    assert g["_mine"]["trust"] == "mined"
    assert "expect_contains" in g["_mine"]["verifiers"]

    # ledger finalized
    row = store__test_mine_goldens._conn.execute(
        "SELECT status, enabled FROM mined_goldens WHERE trace_id='t1'"
    ).fetchone()
    assert (row["status"], row["enabled"]) == ("mined", 1)

    # appears in the merged suite task list and passes replay
    merged = merge_mined_into_tasks([], home)
    assert [t["prompt"] for t in merged] == [g["prompt"]]
    handler = make_message_handler({"rename": {"role": "assistant",
                                               "content": GOOD_CONTENT}})
    with FakeOpenAIServer(handler_cls=handler) as srv:
        suite = run_golden_suite(srv.base_url, "fake-model", merged,
                                 temperature=0.0)
    assert suite["total"] == 1 and suite["passed"] == 1


def test_rft_select_prompts_picks_up_mined_golden(home, store__test_mine_goldens):
    """Mined goldens join RFT's golden prompt source (job-runner merge path)."""
    seed_trace__test_mine_goldens(store__test_mine_goldens, home, trace_id="t1", messages=BRANCH_TRACE)
    run_mine_goldens(store__test_mine_goldens, home, cfg())

    from sable.core.config import RFTConfig as _RC
    rcfg = _RC(enabled=True, sample_k=1, max_prompts=10)
    prompts, _ = select_prompts(store__test_mine_goldens, rcfg,
                                goldens=load_mined_goldens(home))
    assert len(prompts) == 1
    assert prompts[0].is_golden and prompts[0].source == "golden"

    # and the rft job runner (the real consumer) samples + verifies it:
    reg = ModelRegistry(home / "buffer" / "sable.db")
    try:
        sampler = lambda msgs: {"role": "assistant", "content": GOOD_CONTENT}
        runner = make_rft_job_runner(
            reg, home=home, defaults=_RC(enabled=True, sample_k=1),
            endpoint_for=lambda: _ctx(("http://127.0.0.1:1/v1", "m")),
            sampler_factory=lambda *a, **k: sampler)

        class J:
            job_id = "jr"
            params = {}

        out = runner(J())
        assert out["rows_kept"] == 1
    finally:
        reg.close()


class _ctx:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self.value

    def __exit__(self, *exc):
        return False


# ------------------------------------------------------------ drop + gates

def test_verifier_less_candidate_dropped_with_reason(home, store__test_mine_goldens):
    """Reply too short to yield any deterministic check -> dropped (a 👍 with
    nothing machine-checkable must not mint a vacuous golden)."""
    seed_trace__test_mine_goldens(store__test_mine_goldens, home, trace_id="t2", messages=[
        {"role": "user", "content": "What is the mood today?"},
        {"role": "assistant", "content": "OK"},
    ])
    res = run_mine_goldens(store__test_mine_goldens, home, cfg())
    assert res.mined == 0 and len(res.dropped) == 1
    assert "deterministic" in res.dropped[0]["reason"]
    row = store__test_mine_goldens._conn.execute(
        "SELECT status, reason FROM mined_goldens WHERE trace_id='t2'"
    ).fetchone()
    assert row["status"] == "dropped" and "deterministic" in row["reason"]
    assert not mined_tasks(home)


def test_judge_may_only_propose_self_passing_substrings(home, store__test_mine_goldens):
    """Judge proposals join ONLY if the source output passes them; a fabricated
    phrase is discarded, and no golden is ever judge-only."""
    seed_trace__test_mine_goldens(store__test_mine_goldens, home, trace_id="t3", messages=BRANCH_TRACE)

    def judge(msgs, model):
        return json.dumps({"expect_contains": [
            "git branch --show-current",   # present -> adopted
            "definitely not in the reply",  # absent -> rejected by self-check
        ]})

    res = run_mine_goldens(store__test_mine_goldens, home, cfg(use_judge=True),
                           judge_client_fn=judge, judge_model="j")
    assert res.mined == 1
    g = mined_tasks(home)[0]
    joined = "".join(g["expect_contains"])
    assert "git branch --show-current" in joined or len(g["expect_contains"]) >= 1
    assert all("definitely not" not in s for s in g["expect_contains"])


def test_tool_call_verifier_derived(home, store__test_mine_goldens):
    seed_trace__test_mine_goldens(store__test_mine_goldens, home, trace_id="t4", messages=[
        {"role": "user", "content": "List the files in src/"},
        {"role": "assistant", "content": "Listing now for you.",
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "list_files",
                                      "arguments": "{\"path\": \"src/\"}"}}]},
    ])
    res = run_mine_goldens(store__test_mine_goldens, home, cfg())
    assert res.mined == 1
    g = mined_tasks(home)[0]
    assert g["expect_tool_call"] == {"name": "list_files"}
    assert "tool_call" in "".join(g["_mine"]["verifiers"])


# ------------------------------------------------------------------ dedup

def test_same_trace_mined_once(home, store__test_mine_goldens):
    seed_trace__test_mine_goldens(store__test_mine_goldens, home, trace_id="t5", messages=BRANCH_TRACE)
    r1 = run_mine_goldens(store__test_mine_goldens, home, cfg())
    r2 = run_mine_goldens(store__test_mine_goldens, home, cfg())
    assert r1.mined == 1
    assert r2.candidates == 0 and r2.mined == 0
    assert len(mined_tasks(home)) == 1


def test_prompt_hash_collision_with_manual_golden_skipped(home, store__test_mine_goldens):
    seed_trace__test_mine_goldens(store__test_mine_goldens, home, trace_id="t6", messages=BRANCH_TRACE)
    manual = [{"prompt": BRANCH_TRACE[0]["content"], "expect_contains": ["x"]}]
    res = run_mine_goldens(store__test_mine_goldens, home, cfg(), existing_goldens=manual)
    assert res.skipped_dedup == 1 and res.mined == 0
    assert not mined_tasks(home)
    # merge dedups too: manual golden wins, no duplicate in the suite list
    p = mined_path(home)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"prompt": BRANCH_TRACE[0]["content"],
                             "expect_contains": ["y"]}) + "\n")
    merged = merge_mined_into_tasks(manual, home)
    assert len(merged) == 1 and merged[0]["expect_contains"] == ["x"]


def test_untthumbd_trace_never_a_candidate(home, store__test_mine_goldens):
    seed_trace__test_mine_goldens(store__test_mine_goldens, home, trace_id="t7", messages=BRANCH_TRACE, thumb=False)
    res = run_mine_goldens(store__test_mine_goldens, home, cfg())
    assert res.candidates == 0 and res.mined == 0


# ---------------------------------------------------------------- redaction

def test_secret_in_source_output_never_persisted(home, store__test_mine_goldens):
    """A raw token in the (unredacted-at-rest) assistant output is masked and
    never persisted. When masking is ALL the reply has to offer, the candidate
    is dropped entirely: a substring carrying [REDACTED] passes the self-check
    against the redacted message but could never match real replay output — an
    always-failing golden would poison the promotion gate."""
    secret = "ghp_abc1234567890abcdefghij"  # matches redact.py GitHub PAT family
    seed_trace__test_mine_goldens(store__test_mine_goldens, home, trace_id="t8", messages=[
        {"role": "user",
         "content": f"Store my token {secret} somewhere safe"},
        {"role": "assistant",
         "content": f"Saved. The token {secret} is now configured globally."},
    ])
    res = run_mine_goldens(store__test_mine_goldens, home, cfg())
    assert res.mined == 0  # every derived line carries a redaction marker
    assert res.dropped and "no deterministic verifier" in res.dropped[0]["reason"]
    raw_file = mined_path(home).read_text() if mined_path(home).is_file() else ""
    assert secret not in raw_file
    assert "[REDACTED" not in raw_file


def test_mixed_reply_keeps_clean_verifiers_drops_masked_lines(home, store__test_mine_goldens):
    """Reply with one clean line + one secret-bearing line: the golden mines
    with ONLY the clean substring — no [REDACTED] verifier can slip through."""
    secret = "ghp_abc1234567890abcdefghij"
    seed_trace__test_mine_goldens(store__test_mine_goldens, home, trace_id="t9", messages=[
        {"role": "user", "content": "Configure my token please"},
        {"role": "assistant", "content":
            "Set the deploy key.\nThe token " + secret + " is now configured."},
    ])
    res = run_mine_goldens(store__test_mine_goldens, home, cfg())
    assert res.mined == 1
    task = res.goldens[0]
    subs = task.get("expect_contains") or []
    assert any("deploy key" in s for s in subs)
    assert not any("[REDACTED" in s for s in subs)
    assert secret not in json.dumps(task)


# ------------------------------------------------- ledger atomicity + cap

def test_ledger_claim_is_atomic(home, store__test_mine_goldens):
    assert store__test_mine_goldens.claim_mined_goldens(["a", "b"]) == ["a", "b"]
    assert store__test_mine_goldens.claim_mined_goldens(["b", "c"]) == ["c"]  # 'b' already owned


def test_stale_claims_are_reclaimed_finalized_never(home, store__test_mine_goldens):
    """Crash recovery: an unfinalized 'claimed' row older than the stale
    window is released and claimable again; mined/dropped rows never are."""
    assert store__test_mine_goldens.claim_mined_goldens(["x", "y", "z"]) == ["x", "y", "z"]
    store__test_mine_goldens.record_mined_golden("y", prompt_key="ky", status="mined",
                              enabled=True, task_json="{}")
    store__test_mine_goldens.record_mined_golden("z", prompt_key="", status="dropped",
                              reason="no verifier")
    # backdate all claims so 'x' (still claimed) looks orphaned
    store__test_mine_goldens._conn.execute(
        "UPDATE mined_goldens SET claimed_at = '2000-01-01T00:00:00+00:00'")
    store__test_mine_goldens._conn.commit()
    assert store__test_mine_goldens.claim_mined_goldens(["x", "y", "z"]) == ["x"]


def test_fresh_claims_are_not_reclaimed(home, store__test_mine_goldens):
    assert store__test_mine_goldens.claim_mined_goldens(["n1"]) == ["n1"]
    # immediate re-claim: still in flight, must NOT be released
    assert store__test_mine_goldens.claim_mined_goldens(["n1"]) == []


# ---------------------------------------------------- list + toggle (UI API)

def test_list_and_toggle_mined_goldens(home, store__test_mine_goldens):
    from sable.curate.mine_goldens import (
        list_mined_goldens, set_mined_enabled)

    seed_trace__test_mine_goldens(store__test_mine_goldens, home, trace_id="tg1", messages=BRANCH_TRACE)
    res = run_mine_goldens(store__test_mine_goldens, home, cfg())
    assert res.mined == 1

    rows = list_mined_goldens(home)
    assert len(rows) == 1 and rows[0]["enabled"] is True and rows[0]["key"]

    key = rows[0]["key"]
    assert set_mined_enabled(home, key, False) is True
    # disabled goldens vanish from the enabled loader but stay listable
    assert load_mined_goldens(home) == []
    assert list_mined_goldens(home)[0]["enabled"] is False
    # ...and merge (promotion gate) skips them too
    assert merge_mined_into_tasks([], home) == []

    assert set_mined_enabled(home, key, True) is True
    assert len(load_mined_goldens(home)) == 1
    assert set_mined_enabled(home, "nope", True) is False


def test_max_per_run_cap(home, store__test_mine_goldens):
    for i in range(3):
        seed_trace__test_mine_goldens(store__test_mine_goldens, home, trace_id=f"c{i}", messages=[
            {"role": "user", "content": f"question number {i} about widgets"},
            {"role": "assistant",
             "content": f"answer {i}: widgets need the exact phrase number {i}."},
        ])
    res = run_mine_goldens(store__test_mine_goldens, home, cfg(max_per_run=2))
    assert res.candidates == 2 and res.mined == 2
    assert len(mined_tasks(home)) == 2


# ----------------------------------------------------------- disabled path

def test_disabled_master_switch_refuses(home, store__test_mine_goldens):
    with pytest.raises(MineGoldensError, match="goldens_mine.enabled is false"):
        run_mine_goldens(store__test_mine_goldens, home, GoldensMineConfig())


def test_runner_refuses_when_master_switch_off(home):
    reg = ModelRegistry(home / "buffer" / "sable.db")
    try:
        runner = make_mine_goldens_job_runner(reg, home=home,
                                              defaults=GoldensMineConfig())

        class J:
            job_id = "j1"
            params = {}

        with pytest.raises(MineGoldensError, match="goldens_mine.enabled"):
            runner(J())
    finally:
        reg.close()


# ------------------------------------------------------------ dispatcher e2e

def test_mine_goldens_job_flows_through_dispatcher(home):
    """queued -> running -> done under a JobDispatcher tick; _result recorded;
    the in-process kind survives reap_stale (INPROCESS_JOB_KINDS)."""
    st = TraceStore(home / "buffer" / "sable.db")
    seed_trace__test_mine_goldens(st, home, trace_id="d1", messages=BRANCH_TRACE)
    st.close()

    reg = ModelRegistry(home / "buffer" / "sable.db")
    try:
        runner = make_mine_goldens_job_runner(reg, home=home, defaults=cfg())
        tc = TrainingConfig()
        factory = lambda: Scheduler(registry=ModelRegistry(
            home / "buffer" / "sable.db"), cfg=tc,
            inprocess_runners={"mine_goldens": runner})
        sched = factory()
        job = sched.submit("mine_goldens", log_dir=home / "train_runs",
                           params={}, enqueue_actor="test")
        assert job.state == "queued"
        d = JobDispatcher(db_path=reg.db_path, scheduler_factory=factory,
                          tick_s=0.2)
        d.start()
        try:
            import time
            deadline = time.time() + 15
            while time.time() < deadline:
                cur = reg.get_job(job.job_id)
                assert cur is not None
                if cur.state in ("done", "error"):
                    break
                assert cur.state in ("queued", "running")
                time.sleep(0.1)
        finally:
            d.stop()
        cur = reg.get_job(job.job_id)
        assert cur is not None
        assert cur.state == "done", cur.error
        assert cur.params is not None
        result = cur.params["_result"]
        assert result["mined"] == 1
        assert len(mined_tasks(home)) == 1
    finally:
        reg.close()


# ------------------------------------------------------------------ config

def test_goldens_mine_config_defaults_and_yaml_parsing():
    d = GoldensMineConfig()
    assert d.enabled is False and d.max_per_run == 8 and d.use_judge is False
    parsed = GoldensMineConfig.from_yaml(
        {"goldens_mine": {"enabled": True, "max_per_run": 3}})
    assert parsed.enabled is True and parsed.max_per_run == 3
    quiet = GoldensMineConfig.from_yaml({"goldens_mine": {"nonsense": 1}})
    assert quiet.enabled is False


def test_load_config_recognizes_goldens_mine_without_warnings(tmp_path, caplog):
    import logging

    p = tmp_path / "c.yaml"
    p.write_text("goldens_mine:\n  enabled: true\n  max_per_run: 4\n")
    with caplog.at_level(logging.WARNING):
        c = load_config(str(p))
    assert c.goldens_mine == {"enabled": True, "max_per_run": 4}
    assert not [r for r in caplog.records if "unknown" in r.getMessage()]


def test_cli_mine_goldens_parses_and_refuses_when_disabled(tmp_path, monkeypatch):
    from sable.cli import build_parser, cmd_mine_goldens

    args = build_parser().parse_args(["mine-goldens", "--max", "3"])
    assert args.command == "mine-goldens" and args.max == 3

    monkeypatch.setenv("SABLE_HOME", str(tmp_path / "h"))
    c = tmp_path / "c.yaml"
    c.write_text("goldens_mine:\n  enabled: false\n")
    ns = argparse.Namespace(config=str(c), verbose=False, max=None,
                            use_judge=False, wait=False)
    assert cmd_mine_goldens(ns) == 2  # master switch off -> refuse before queue


# ------------------------------------------------------------------ helpers

def test_task_pair_shapes():
    p, a = task_pair(BRANCH_TRACE)
    assert p == BRANCH_TRACE[0]["content"]
    assert a is not None and a["role"] == "assistant"
    assert task_pair([{"role": "assistant", "content": "x"}]) == (None, None)
    # last user turn wins; assistant before it is ignored
    msgs = [{"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer with plenty of text"},
            {"role": "user", "content": "new question here please"}]
    p, a = task_pair(msgs)
    assert p == "new question here please" and a is None

# ------------------------------------------------------------------------
# from tests/test_metrics_and_diff.py
# ------------------------------------------------------------------------

@pytest.fixture()
def home__test_metrics_and_diff(tmp_path, monkeypatch):
    h = tmp_path / "uf-md"
    monkeypatch.setenv("SABLE_HOME", str(h))
    return h


@pytest.fixture()
def app__test_metrics_and_diff(home__test_metrics_and_diff):
    return DashboardApp(SableConfig(), home=home__test_metrics_and_diff)


# ============================================================ text-format gate

# Prometheus text exposition spec patterns (metric/label names + sample
# syntax). Family names here are sable_* so [a-zA-Z_:] covers them.
_RE_NAME = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*$")
_RE_HELP = re.compile(r"^# HELP [a-zA-Z_:][a-zA-Z0-9_:]* .+$")
_RE_TYPE = re.compile(
    r"^# TYPE [a-zA-Z_:][a-zA-Z0-9_:]* "
    r"(counter|gauge|summary|histogram|untyped)$")
_RE_LABELS = (r"\{(?:[a-zA-Z_][a-zA-Z0-9_]*=\"(?:[^\"\\]|\\.)*\")"
              r"(?:,[a-zA-Z_][a-zA-Z0-9_]*=\"(?:[^\"\\]|\\.)*\")*\}")
_RE_SAMPLE = re.compile(
    r"^[a-zA-Z_:][a-zA-Z0-9_:]*(?:" + _RE_LABELS + r")?"
    r" -?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?$")


def assert_valid_exposition(text: str) -> Dict[str, str]:
    """Validate the whole body against the text-format grammar.

    Returns {family: type}. Raises AssertionError with the offending line.
    """
    assert text.endswith("\n"), "exposition must end with a newline"
    types: Dict[str, str] = {}
    pending: Optional[str] = None  # family awaiting its TYPE line
    for i, line in enumerate(text.splitlines()):
        if not line.strip():
            continue
        if line.startswith("# HELP "):
            assert _RE_HELP.match(line), f"line {i}: bad HELP: {line!r}"
            pending = line.split()[2]
            assert pending not in types, f"duplicate HELP for {pending}"
            continue
        if line.startswith("# TYPE "):
            m = _RE_TYPE.match(line)
            assert m, f"line {i}: bad TYPE: {line!r}"
            fam = line.split()[2]
            assert pending == fam, \
                f"line {i}: TYPE for {fam} without an adjacent HELP"
            types[fam] = m.group(1)
            pending = None
            continue
        assert line.startswith("#") is False or line.startswith("# EOF"), \
            f"line {i}: unknown comment form: {line!r}"
        m = _RE_SAMPLE.match(line)
        assert m, f"line {i}: bad sample line: {line!r}"
        family = line.split("{")[0].split(" ")[0]
        # summary/histogram sub-samples (_sum/_count/quantile/bucket) belong
        # to the parent family declared by TYPE.
        root = family
        for suffix in ("_sum", "_count", "_bucket"):
            if root.endswith(suffix) and root[: -len(suffix)] in types:
                root = root[: -len(suffix)]
        assert root in types, f"line {i}: sample {family} has no TYPE"
    return types


def _seed_rich_home__test_metrics_and_diff(home):
    """Traces + jobs (+gate reason) + promotion/rollback + rft/mined ledgers."""
    store = TraceStore(Path(home) / "buffer" / "sable.db")
    for n, status in ((5, "success"), (2, "failure"), (1, "partial")):
        for i in range(n):
            tid = f"{status[0]}{i}"
            p = Path(store.db_path).parent / f"{tid}.jsonl"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"kind": "trace_header"}) + "\n")
            store.insert_trace(TraceRow(
                trace_id=tid, session_id=None,
                timestamp="2026-09-20T10:00:00+00:00",
                agent_profile="p", model_alias="m", model_version=None,
                endpoint_label=None, task_summary="t", status=status,
                outcome_score=0.5, num_tool_calls=0, tokens_in=7,
                tokens_out=10, duration_s=float(i + 1), jsonl_path=str(p)))
    # rft ledger: one kept, two sampled-but-unkept
    keys = [("k1", "seed", None), ("k2", "seed", None), ("k3", "seed", None)]
    store.claim_rft_prompts(keys)
    store.mark_rft_prompts_emitted(["k1", "k2", "k3"], "ds-x", kept_keys=["k1"])
    # mined goldens: one mined, one dropped
    store.claim_mined_goldens(["g1", "g2"])
    store.record_mined_golden("g1", prompt_key="p1", status="mined",
                              enabled=True)
    store.record_mined_golden("g2", prompt_key="p2", status="dropped",
                              reason="no stable lines")
    # gate events (v9): one durable block episode — seeds the cumulative
    # counter; the job param below is the current-state gauge side.
    store.record_gate_event("seed-blocked-1", "disk gate")
    store.close()

    reg = ModelRegistry(Path(home) / "buffer" / "sable.db")
    j1 = reg.enqueue_job("sft")
    reg.claim_next_queued(pid=4242)          # -> running, started_at set
    reg.enqueue_job("curate")                # stays queued
    reg.set_job_param(j1.job_id, "_gate_reason", "disk gate")
    m1 = reg.register_model("base-a")
    reg.set_validation(m1.model_id, {"ok": True}, passed=True)
    reg.promote(m1.model_id)                 # ledger op promote
    m2 = reg.register_model("base-a")
    reg.set_validation(m2.model_id, {"ok": True}, passed=True)
    reg.promote(m2.model_id)
    reg.rollback(base_model="base-a")        # ledger op rollback
    reg.close()


def _metrics__test_metrics_and_diff(app, refresh=True):
    """GET /api/metrics through the router (query goes in as a dict —
    DashboardApp.handle strips nothing; server.py parses the query string)."""
    q = {"refresh": "1"} if refresh else {}
    resp = app.handle("GET", "/api/metrics", query=q)
    assert resp.status == 200, resp.body
    return resp


def _sample_value(text: str, prefix: str, labels: str = "") -> Optional[str]:
    """Value of one exact sample line (prefix + optional label set)."""
    needle = prefix + (labels or "")
    for line in text.splitlines():
        if line.startswith(needle + " ") or line.startswith(needle + "{"):
            if labels and not line.startswith(prefix + labels):
                continue
            return line.rsplit(" ", 1)[1]
    return None


# ---------------------------------------------------------------- metrics: A

def test_metrics_exposition_valid_and_complete(app__test_metrics_and_diff, home__test_metrics_and_diff):
    _seed_rich_home__test_metrics_and_diff(home__test_metrics_and_diff)
    resp = _metrics__test_metrics_and_diff(app__test_metrics_and_diff)
    text = resp.raw.decode("utf-8")
    types = assert_valid_exposition(text)
    expected = {
        "sable_traces_total": "counter",
        "sable_outcome_score_mean": "gauge",
        "sable_tokens_in_total": "counter",
        "sable_tokens_out_total": "counter",
        "sable_tokens_per_second": "summary",
        "sable_request_latency_seconds": "summary",
        "sable_jobs_total": "counter",
        "sable_job_queue_wait_seconds": "summary",
        "sable_gate_blocks_total": "counter",
        "sable_jobs_gate_blocked": "gauge",
        "sable_promotions_total": "counter",
        "sable_promotion_pass_rate": "gauge",
        "sable_instances_up": "gauge",
        "sable_rft_samples_kept_total": "counter",
        "sable_rft_samples_dropped_total": "counter",
        "sable_mined_goldens_total": "counter",
    }
    assert expected == {k: types[k] for k in expected}, sorted(types)
    # every family has HELP and TYPE (grammar check above already enforced
    # adjacency; this enforces presence per family).
    helps = {l.split()[2] for l in text.splitlines() if l.startswith("# HELP")}
    tys = {l.split()[2] for l in text.splitlines() if l.startswith("# TYPE")}
    assert helps == tys == set(types)


def test_metrics_values_match_seed(app__test_metrics_and_diff, home__test_metrics_and_diff):
    _seed_rich_home__test_metrics_and_diff(home__test_metrics_and_diff)
    text = _metrics__test_metrics_and_diff(app__test_metrics_and_diff).raw.decode("utf-8")
    assert _sample_value(text, "sable_traces_total",
                         '{status="success"}') == "5"
    assert _sample_value(text, "sable_traces_total",
                         '{status="failure"}') == "2"
    assert _sample_value(text, "sable_traces_total",
                         '{status="partial"}') == "1"
    assert _sample_value(text, "sable_traces_total",
                         '{status="unknown"}') == "0"
    # 8 traces x tokens_in=7 / tokens_out=10
    assert _sample_value(text, "sable_tokens_in_total") == "56"
    assert _sample_value(text, "sable_tokens_out_total") == "80"
    assert _sample_value(text, "sable_outcome_score_mean") == "0.5"
    # durations: success 1..5, failure 1..2, partial 1 -> count 8.
    assert _sample_value(text, "sable_request_latency_seconds_count") == "8"
    # throughput 10/d sorted [2, 2.5, 3.3333, 5, 5, 10, 10, 10] -> p50 rank 4 = 5
    assert _sample_value(
        text, "sable_tokens_per_second{quantile=\"0.5\"}") == "5"
    # jobs: one running sft + one queued curate (state counts)
    assert _sample_value(text, "sable_jobs_total",
                         '{kind="sft",state="running"}') == "1"
    assert _sample_value(text, "sable_jobs_total",
                         '{kind="curate",state="queued"}') == "1"
    # gate blocks: counter = durable block episodes (one seeded event);
    # gauge = jobs currently carrying a non-null _gate_reason (the same one)
    assert _sample_value(text, "sable_gate_blocks_total") == "1"
    assert _sample_value(text, "sable_jobs_gate_blocked") == "1"
    # promotions: 2 promote ops, 1 rollback op in the ledger -> pass-rate 2/3
    assert _sample_value(text, "sable_promotions_total",
                         '{result="pass"}') == "2"
    assert _sample_value(text, "sable_promotions_total",
                         '{result="fail"}') == "1"
    rate_s = _sample_value(text, "sable_promotion_pass_rate")
    assert rate_s is not None
    rate = float(rate_s)
    assert abs(rate - 2 / 3) < 1e-6
    # ledgers
    assert _sample_value(text, "sable_rft_samples_kept_total") == "1"
    assert _sample_value(text, "sable_rft_samples_dropped_total") == "2"
    assert _sample_value(text, "sable_mined_goldens_total",
                         '{status="mined"}') == "1"
    assert _sample_value(text, "sable_mined_goldens_total",
                         '{status="dropped"}') == "1"
    # default instance exists (config-derived) but is not running
    assert _sample_value(text, "sable_instances_up",
                         '{id="default"}') == "0"


def test_metrics_empty_home_is_honest(app__test_metrics_and_diff):
    resp = _metrics__test_metrics_and_diff(app__test_metrics_and_diff)
    text = resp.raw.decode("utf-8")
    types = assert_valid_exposition(text)
    assert "sable_traces_total" in types
    # zero-valued status counters still present (valid empty family)
    assert _sample_value(text, "sable_traces_total",
                         '{status="success"}') == "0"
    # score mean honest-absent when nothing is scored
    assert "sable_outcome_score_mean " not in text


def test_metrics_content_type(app__test_metrics_and_diff):
    resp = _metrics__test_metrics_and_diff(app__test_metrics_and_diff)
    assert resp.content_type == "text/plain; version=0.0.4"


def test_metrics_ttl_cache(app__test_metrics_and_diff, home__test_metrics_and_diff):
    _seed_rich_home__test_metrics_and_diff(home__test_metrics_and_diff)
    body1 = _metrics__test_metrics_and_diff(app__test_metrics_and_diff).raw.decode()
    # New data lands in the DB *after* the first scrape...
    store = TraceStore(Path(home__test_metrics_and_diff) / "buffer" / "sable.db")
    p = Path(store.db_path).parent / "late.jsonl"
    p.write_text(json.dumps({"kind": "trace_header"}) + "\n")
    store.insert_trace(TraceRow(
        trace_id="late", session_id=None,
        timestamp="2026-09-21T10:00:00+00:00", agent_profile="p",
        model_alias="m", model_version=None, endpoint_label=None,
        task_summary="t", status="success", outcome_score=0.5,
        num_tool_calls=0, tokens_in=1, tokens_out=1, duration_s=1.0,
        jsonl_path=str(p)))
    store.close()
    # ...still served from the TTL cache...
    body2 = _metrics__test_metrics_and_diff(app__test_metrics_and_diff, refresh=False).raw.decode()
    assert body2 == body1
    # ...until an explicit refresh bypasses it.
    body3 = _metrics__test_metrics_and_diff(app__test_metrics_and_diff).raw.decode()
    assert _sample_value(body3, "sable_traces_total",
                         '{status="success"}') == "6"
    assert _sample_value(body2, "sable_traces_total",
                         '{status="success"}') == "5"


def test_metrics_percentile_matches_insights(app__test_metrics_and_diff, home__test_metrics_and_diff):
    """Quantiles reuse insights.percentile (ceil nearest-rank)."""
    from sable.store.insights import percentile

    store = TraceStore(Path(home__test_metrics_and_diff) / "buffer" / "sable.db")
    durs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    for i, d in enumerate(durs):
        tid = f"p{i}"
        p = Path(store.db_path).parent / f"{tid}.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"kind": "trace_header"}) + "\n")
        store.insert_trace(TraceRow(
            trace_id=tid, session_id=None,
            timestamp="2026-09-20T10:00:00+00:00", agent_profile="p",
            model_alias="m", model_version=None, endpoint_label=None,
            task_summary="t", status="success", outcome_score=None,
            num_tool_calls=0, tokens_in=None, tokens_out=None,
            duration_s=d, jsonl_path=str(p)))
    store.close()
    text = _metrics__test_metrics_and_diff(app__test_metrics_and_diff).raw.decode()
    srt = sorted(durs)
    assert _sample_value(
        text, "sable_request_latency_seconds{quantile=\"0.5\"}") \
        == _fmt(percentile(srt, 50))
    assert _sample_value(
        text, "sable_request_latency_seconds{quantile=\"0.9\"}") \
        == _fmt(percentile(srt, 90))
    assert _sample_value(
        text, "sable_request_latency_seconds{quantile=\"0.95\"}") \
        == _fmt(percentile(srt, 95))


def _fmt(v) -> str:
    from sable.dashboard.metrics import _fmt_num
    return _fmt_num(v)


# ------------------------------------------------------------------ auth: A

def _start_server_with_token__test_metrics_and_diff(home, token=None):
    from sable.dashboard.server import BIND_HOST, make_server

    httpd = make_server(SableConfig(), port=0, home=home, token=token)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, f"http://{BIND_HOST}:{httpd.server_address[1]}"


def test_metrics_token_auth_parity(home__test_metrics_and_diff):
    """No backdoor: with a token configured, /api/metrics demands it exactly
    like /api/status (header or GET query param)."""
    httpd, base = _start_server_with_token__test_metrics_and_diff(home__test_metrics_and_diff, token="scr4pe")
    try:
        import httpx

        assert httpx.get(base + "/api/metrics", timeout=5).status_code == 401
        r = httpx.get(base + "/api/metrics", timeout=5,
                      headers={"X-Sable-Token": "scr4pe"})
        assert r.status_code == 200
        assert r.headers["content-type"] == "text/plain; version=0.0.4"
        # GET query token allowed for scrapers (same rule as SSE/EventSource)
        assert httpx.get(base + "/api/metrics?token=scr4pe",
                         timeout=5).status_code == 200
    finally:
        httpd.shutdown()
        httpd.server_close()


# ================================================================ dataset diff

def _make_dataset(root: Path, ds_id: str, *, sft_ids=None, dpo_ids=None,
                  created_at="2026-09-01T00:00:00+00:00", source=None,
                  sft_lines=None, extra_lines=None):
    """Write a dataset dir with manifest (+optional raw-file overrides)."""
    d = root / ds_id
    d.mkdir(parents=True, exist_ok=True)
    files: Dict[str, Any] = {
        "sft.jsonl": {"count": len(sft_ids or []),
                      "trace_ids": list(sft_ids or []), "format": "sharegpt"},
        "dpo.jsonl": {"count": len(dpo_ids or []),
                      "trace_ids": list(dpo_ids or []),
                      "format": "trl-preference"},
    }
    manifest = {"dataset_id": ds_id, "created_at": created_at,
                "files": files}
    if source:
        manifest["source"] = source
    (d / "manifest.json").write_text(json.dumps(manifest))
    if sft_lines is not None:
        (d / "sft.jsonl").write_text("".join(sft_lines))
    elif sft_ids:
        (d / "sft.jsonl").write_text(
            "".join(json.dumps({"conversations": [], "trace_id": t}) + "\n"
                    for t in sft_ids))
    if dpo_ids:
        (d / "dpo.jsonl").write_text(
            "".join(json.dumps({"metadata": {"chosen_trace_id": t}}) + "\n"
                    for t in dpo_ids))
    if extra_lines:
        (d / "extra.jsonl").write_text("".join(extra_lines))
    return d


def _pair__test_metrics_and_diff(home):
    """Two datasets: overlap {t2,t3}, unique a:{t1}, unique b:{t4,t5}; one
    modified file (sft content differs), identical dpo file."""
    root = Path(home) / "datasets"
    _make_dataset(root, "ds-a", sft_ids=["t1", "t2", "t3"], dpo_ids=["d1"],
                  created_at="2026-09-01T00:00:00+00:00")
    _make_dataset(root, "ds-b", sft_ids=["t2", "t3", "t4", "t5"],
                  dpo_ids=["d1"], created_at="2026-09-08T00:00:00+00:00")
    # make ds-a's sft.jsonl content differ from what its ids would produce
    (root / "ds-a" / "sft.jsonl").write_text(
        json.dumps({"conversations": [], "trace_id": "t1"}) + "\n"
        + json.dumps({"conversations": [{"from": "human", "value": "old"}],
                      "trace_id": "t2"}) + "\n"
        + json.dumps({"conversations": [], "trace_id": "t3"}) + "\n")
    return root


def test_dataset_diff_sets_and_hashes(app__test_metrics_and_diff, home__test_metrics_and_diff):
    _pair__test_metrics_and_diff(home__test_metrics_and_diff)
    resp = app__test_metrics_and_diff.handle("GET", "/api/datasets/diff",
                      query={"a": "ds-a", "b": "ds-b"})
    assert resp.status == 200
    body = resp.body
    td = body["trace_diff"]["sft.jsonl"]
    assert td["added_count"] == 2 and sorted(td["added_sample"]) == ["t4", "t5"]
    assert td["removed_count"] == 1 and td["removed_sample"] == ["t1"]
    assert td["unchanged_count"] == 2 and td["unchanged_sample"] == ["t2", "t3"]
    # hash verdicts: sft differs (modified file), dpo equal
    assert body["files"]["sft.jsonl"]["hash_equal"] is False
    assert body["files"]["dpo.jsonl"]["hash_equal"] is True
    # count deltas: sft 3 -> 4, dpo 1 -> 1
    assert body["counts"]["sft_delta"] == 1
    assert body["counts"]["dpo_delta"] == 0
    # row-level content diff sees the modified t2 row + added t5-side rows
    rows = body["rows"]["sft.jsonl"]
    assert rows["content_equal"] is False
    assert rows["rows_added"] >= 1 and rows["rows_removed"] >= 1
    # metadata: both curate-side (no source) -> equal; created delta one week
    assert body["source_equal"] is True
    assert body["created_at_delta_s"] == 7 * 86400.0
    assert body["identical"] is False


def test_dataset_diff_source_field(app__test_metrics_and_diff, home__test_metrics_and_diff):
    root = Path(home__test_metrics_and_diff) / "datasets"
    _make_dataset(root, "cur", sft_ids=["t1"])
    _make_dataset(root, "rft", sft_ids=["t1"], source="rft")
    body = app__test_metrics_and_diff.handle("GET", "/api/datasets/diff",
                      query={"a": "cur", "b": "rft"}).body
    assert body["source_equal"] is False
    assert body["b"]["source"] == "rft" and body["a"]["source"] is None


def test_dataset_diff_identical(app__test_metrics_and_diff, home__test_metrics_and_diff):
    root = Path(home__test_metrics_and_diff) / "datasets"
    _make_dataset(root, "x1", sft_ids=["t1", "t2"], dpo_ids=["d1"])
    _make_dataset(root, "x2", sft_ids=["t1", "t2"], dpo_ids=["d1"])
    body = app__test_metrics_and_diff.handle("GET", "/api/datasets/diff",
                      query={"a": "x1", "b": "x2"}).body
    assert body["identical"] is True
    td = body["trace_diff"]["sft.jsonl"]
    assert td["added_count"] == 0 and td["removed_count"] == 0
    assert td["unchanged_count"] == 2
    assert all(f["hash_equal"] is True for f in body["files"].values())
    # same-id compare is trivially identical
    same = app__test_metrics_and_diff.handle("GET", "/api/datasets/diff",
                      query={"a": "x1", "b": "x1"}).body
    assert same["identical"] is True


def test_dataset_diff_missing_ids_404(app__test_metrics_and_diff, home__test_metrics_and_diff):
    root = _pair__test_metrics_and_diff(home__test_metrics_and_diff)
    r = app__test_metrics_and_diff.handle("GET", "/api/datasets/diff",
                   query={"a": "ds-a", "b": "nope"})
    assert r.status == 404
    assert "no dataset" in r.body["error"]["message"]
    # a directory without manifest.json is not a dataset -> 404 as well
    (root / "hollow").mkdir()
    r2 = app__test_metrics_and_diff.handle("GET", "/api/datasets/diff",
                    query={"a": "hollow", "b": "ds-a"})
    assert r2.status == 404


def test_dataset_diff_malformed_manifest_400(app__test_metrics_and_diff, home__test_metrics_and_diff):
    root = _pair__test_metrics_and_diff(home__test_metrics_and_diff)
    (root / "bad" ).mkdir()
    (root / "bad" / "manifest.json").write_text("{not json")
    r = app__test_metrics_and_diff.handle("GET", "/api/datasets/diff",
                   query={"a": "bad", "b": "ds-a"})
    assert r.status == 400
    assert r.body["error"]["code"] == "bad_request"


def test_dataset_diff_missing_params_400(app__test_metrics_and_diff, home__test_metrics_and_diff):
    _pair__test_metrics_and_diff(home__test_metrics_and_diff)
    assert app__test_metrics_and_diff.handle("GET", "/api/datasets/diff",
                      query={"a": "ds-a"}).status == 400
    assert app__test_metrics_and_diff.handle("GET", "/api/datasets/diff", query={}).status == 400


def test_dataset_diff_library_and_cli(home__test_metrics_and_diff, monkeypatch, capsys):
    root = _pair__test_metrics_and_diff(home__test_metrics_and_diff)
    d = diff_datasets(root, "ds-a", "ds-b")
    assert d["trace_diff"]["sft.jsonl"]["added_count"] == 2
    # deterministic: two runs byte-equal payloads
    assert json.dumps(d, sort_keys=True) == json.dumps(
        diff_datasets(root, "ds-a", "ds-b"), sort_keys=True)

    from sable.cli import cmd_datasets_diff
    import argparse
    monkeypatch.setenv("SABLE_HOME", str(home__test_metrics_and_diff))
    ns = argparse.Namespace(verbose=False, config=None, a="ds-a", b="ds-b")
    assert cmd_datasets_diff(ns) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["counts"]["sft_delta"] == 1
    # missing dataset -> exit 1 with an error line
    ns2 = argparse.Namespace(verbose=False, config=None, a="ds-a", b="gone")
    assert cmd_datasets_diff(ns2) == 1


def test_datasets_cli_parser_has_diff():
    from sable.cli import build_parser

    ns = build_parser().parse_args(["datasets", "diff", "a1", "b2"])
    assert ns.func.__name__ == "cmd_datasets_diff"
    assert (ns.a, ns.b) == ("a1", "b2")

# ------------------------------------------------------------------------
# from tests/test_eval_logging.py
# ------------------------------------------------------------------------

def _mk_ckpt(root: Path, step: int, *, weights=True, state=True,
             global_step=None, max_steps=None):
    """Create a checkpoint-<step> dir with the requested contents."""
    ck = root / f"checkpoint-{step}"
    ck.mkdir(parents=True)
    if weights:
        (ck / "adapter_model.safetensors").write_text("w")
    if state:
        gs = step if global_step is None else global_step
        ms = (step * 2) if max_steps is None else max_steps
        (ck / "trainer_state.json").write_text(
            json.dumps({"global_step": gs, "max_steps": ms}))
    return ck


# ------------------------------------------------------------- eval split

def test_no_split_for_tiny_datasets():
    rows = list(range(19))  # below the 20-row floor
    train, ev = maybe_eval_split(rows, 0.1, 42)
    assert ev is None
    assert sorted(train) == rows


def test_no_split_when_eval_side_too_small():
    rows = list(range(25))
    train, ev = maybe_eval_split(rows, 0.05, 42)  # 1 eval row < min 2
    assert ev is None
    train2, ev2 = maybe_eval_split(rows, 0.1, 42)  # 2 eval rows -> splits
    assert ev2 is not None and len(ev2) == 2 and len(train2) == 23


def test_no_split_when_fraction_zero_or_negative():
    rows = list(range(100))
    train, ev = maybe_eval_split(rows, 0.0, 42)
    assert ev is None and len(train) == 100


def test_split_is_deterministic_and_disjoint():
    rows = [{"i": i} for i in range(50)]
    t1, e1 = maybe_eval_split(rows, 0.2, seed=42)
    t2, e2 = maybe_eval_split(rows, 0.2, seed=42)
    assert e1 is not None and e2 is not None
    assert [r["i"] for r in t1] == [r["i"] for r in t2]
    assert [r["i"] for r in e1] == [r["i"] for r in e2]
    ids_t = {r["i"] for r in t1}
    ids_e = {r["i"] for r in e1}
    assert not (ids_t & ids_e)              # disjoint
    assert ids_t | ids_e == set(range(50))  # complete
    assert len(e1) == 10                    # 20% of 50


def test_split_seed_changes_the_split():
    rows = list(range(50))
    _, e1 = maybe_eval_split(rows, 0.2, seed=42)
    _, e2 = maybe_eval_split(rows, 0.2, seed=7)
    assert e1 is not None and e2 is not None
    assert sorted(e1) != sorted(e2)


def test_split_preserves_row_order_in_eval():
    rows = list(range(40))
    _, ev = maybe_eval_split(rows, 0.25, seed=42)
    assert ev is not None
    assert ev == sorted(ev)


def test_split_floor_policy_constants_are_declared():
    # Inspiration pass 3 #22: the floors are declared policy, and the default
    # arguments of maybe_eval_split ARE those constants (one source of truth).
    import inspect
    sig = inspect.signature(maybe_eval_split)
    assert sig.parameters["min_total"].default == EVAL_MIN_TOTAL_ROWS
    assert sig.parameters["min_eval"].default == EVAL_MIN_ROWS
    assert EVAL_MIN_TOTAL_ROWS > EVAL_MIN_ROWS > 0


# ------------------------------------------------- named eval splits (#22)

def test_named_eval_autodetect_next_to_train_files(tmp_path):
    train = tmp_path / "sft.jsonl"
    train.write_text("{}\n")
    ev = tmp_path / "eval.jsonl"
    ev.write_text("{}\n")
    assert find_named_eval_files({}, [str(train)]) == [str(ev)]


def test_named_eval_explicit_data_sec_wins(tmp_path):
    train = tmp_path / "sft.jsonl"
    train.write_text("{}\n")
    (tmp_path / "eval.jsonl").write_text("{}\n")
    holdout = tmp_path / "holdout.jsonl"
    holdout.write_text("{}\n")
    out = find_named_eval_files({"eval_dataset": str(holdout)}, [str(train)])
    assert out == [str(holdout)]  # explicit list beats auto-detection
    out2 = find_named_eval_files({"eval_dataset": [str(holdout), "missing.jsonl"]},
                                 [str(train)])
    assert out2 == [str(holdout)]  # missing entries dropped


def test_named_eval_never_becomes_a_train_file(tmp_path):
    train = tmp_path / "sft.jsonl"
    (tmp_path / "eval.jsonl").write_text("{}\n")
    # A dataset that (wrongly) lists eval.jsonl as a TRAINING file must not
    # get it back as its own holdout.
    assert find_named_eval_files({}, [str(train),
                                      str(tmp_path / "eval.jsonl")]) == []


def test_named_eval_empty_when_nothing_exists(tmp_path):
    train = tmp_path / "sft.jsonl"
    train.write_text("{}\n")
    assert find_named_eval_files({}, [str(train)]) == []


# ------------------------------------------------- checkpoint resume (#21)

def test_resume_requires_weights_and_state(tmp_path):
    _mk_ckpt(tmp_path, 10, weights=False)          # state only -> skip
    _mk_ckpt(tmp_path, 20, state=False)            # weights only -> skip
    assert find_resumable_checkpoint(tmp_path) is None
    good = _mk_ckpt(tmp_path, 30)
    assert find_resumable_checkpoint(tmp_path) == str(good)


def test_resume_picks_highest_step_not_lexical(tmp_path):
    _mk_ckpt(tmp_path, 9)
    _mk_ckpt(tmp_path, 100)   # lexically "checkpoint-100" < "checkpoint-9"
    _mk_ckpt(tmp_path, 50)
    assert find_resumable_checkpoint(tmp_path) == str(tmp_path / "checkpoint-100")


def test_resume_skips_finished_runs(tmp_path):
    # output_dir is shared across runs of a dataset: a checkpoint whose
    # trainer state shows the run completed must never auto-resume.
    done = _mk_ckpt(tmp_path, 40, global_step=60, max_steps=60)
    assert find_resumable_checkpoint(tmp_path) is None
    interrupted = _mk_ckpt(tmp_path, 20, global_step=20, max_steps=60)
    assert find_resumable_checkpoint(tmp_path) == str(interrupted)
    # Zero-step checkpoint: nothing trained yet -> start clean instead.
    shutil.rmtree(interrupted)
    _mk_ckpt(tmp_path, 5, global_step=0, max_steps=60)
    assert find_resumable_checkpoint(tmp_path) is None


def test_resume_unreadable_state_counts_unfinished(tmp_path):
    ck = tmp_path / "checkpoint-7"
    ck.mkdir()
    (ck / "adapter_model.safetensors").write_text("w")
    (ck / "trainer_state.json").write_text("{trunc")  # crashed mid-save
    assert find_resumable_checkpoint(tmp_path) == str(ck)


def test_resume_ignores_junk_dirs(tmp_path):
    (tmp_path / "checkpoint-abc").mkdir()
    (tmp_path / "not-a-checkpoint").mkdir()
    (tmp_path / "outputs").mkdir()          # no recursion into subdirs
    assert find_resumable_checkpoint(tmp_path) is None
    assert find_resumable_checkpoint(tmp_path / "missing") is None


def test_resolve_resume_knob_semantics(tmp_path):
    ck = _mk_ckpt(tmp_path, 12)
    # auto (the default): discover
    assert resolve_resume_path({}, tmp_path) == str(ck)
    # explicit false: never
    assert resolve_resume_path({"resume": False}, tmp_path) is None
    assert resolve_resume_path({"resume": "off"}, tmp_path) is None
    # explicit valid path: used verbatim
    assert resolve_resume_path({"resume": str(ck)}, tmp_path) == str(ck)
    # explicit bogus path: refuse loudly, never silently restart
    with pytest.raises(ValueError):
        resolve_resume_path({"resume": str(tmp_path / "nope")}, tmp_path)


def test_resolve_resume_auto_with_nothing_to_resume(tmp_path):
    assert resolve_resume_path({}, tmp_path) is None


# ------------------------------------------------------------- cadence plan

def test_cadence_steps_when_run_longer_than_save_interval():
    c = plan_cadence(1000, batch_size=2, grad_accum=4, epochs=3, save_steps=0)
    assert c["save_strategy"] == "steps" and c["save_steps"] == 50
    assert c["eval_strategy"] == "steps" and c["eval_steps"] == 50


def test_cadence_honors_explicit_save_steps():
    c = plan_cadence(1000, save_steps=25)
    assert c["save_steps"] == 25 and c["eval_steps"] == 25


def test_cadence_epoch_when_run_shorter_than_save_interval():
    c = plan_cadence(40, batch_size=2, grad_accum=4, epochs=1, save_steps=50)
    assert c["save_strategy"] == "epoch" and c["eval_strategy"] == "epoch"
    assert "save_steps" not in c


def test_cadence_max_steps_respected():
    c = plan_cadence(10_000, max_steps=10, save_steps=50)
    assert c["save_strategy"] == "epoch"  # tiny run: 10 steps <= 50
    c2 = plan_cadence(10_000, max_steps=200, save_steps=50)
    assert c2["save_strategy"] == "steps"


def test_cadence_empty_dataset_no_crash():
    c = plan_cadence(0)
    assert c["save_strategy"] == "no" and c["eval_strategy"] == "no"


# --------------------------------------------------------- event formatting

def test_progress_event_shape():
    ev = progress_event(12, 120, 1.234)
    assert ev == {"event": "progress", "step": 12, "total": 120, "loss": 1.234}
    # backward-compatible: None fields omitted entirely
    assert progress_event(3) == {"event": "progress", "step": 3}


def test_eval_event_shape():
    assert eval_event(20, 0.5) == {"event": "eval", "step": 20, "eval_loss": 0.5}
    assert eval_event(20, None) == {"event": "eval", "step": 20}


class FakeState:
    def __init__(self, step=10, max_steps=100, log_history=None):
        self.global_step = step
        self.max_steps = max_steps
        self.log_history = log_history or []


class FakeArgs:
    max_steps = 100


def test_protocol_callback_emits_json_lines(tmp_path, capsys):
    log = tmp_path / "job.log"
    cb = make_protocol_callback(str(log))

    cb.on_log(FakeArgs(), FakeState(step=10), logs={"loss": 1.5})
    cb.on_evaluate(FakeArgs(), FakeState(step=20), metrics={"eval_loss": 0.75})

    lines = [json.loads(l) for l in log.read_text().splitlines()]
    prog = lines[0]
    assert prog["event"] == "progress" and prog["step"] == 10
    assert prog["total"] == 100 and prog["loss"] == pytest.approx(1.5)
    ev = lines[1]
    assert ev["event"] == "eval" and ev["step"] == 20
    assert ev["eval_loss"] == pytest.approx(0.75)
    # stdout mirror (the scheduler tails stdout too)
    out = capsys.readouterr().out.splitlines()
    assert json.loads(out[0])["event"] == "progress"
    assert cb.last_eval_loss == pytest.approx(0.75)


def test_protocol_callback_loss_from_log_history_fallback(tmp_path):
    log = tmp_path / "job.log"
    cb = make_protocol_callback(str(log))
    state = FakeState(step=30, log_history=[{"loss": 2.0}, {"loss": 1.75}])
    cb.on_log(FakeArgs(), state, logs={"lr": 1e-4})  # no loss in this log dict
    rec = json.loads(log.read_text().splitlines()[0])
    assert rec["loss"] == pytest.approx(1.75)


# ---------------------------------------------------- registry losses + gate

def test_registry_migration_adds_loss_columns(tmp_path):
    db = tmp_path / "sable.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    # Simulate a pre-#4 models table (no train_loss/eval_loss columns).
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE models ("
        " model_id TEXT PRIMARY KEY, base_model TEXT NOT NULL,"
        " parent_adapter TEXT, dataset_id TEXT, manifest_hash TEXT,"
        " adapter_dir TEXT, gguf_path TEXT, quant TEXT, status TEXT NOT NULL,"
        " created_at TEXT NOT NULL, validation_json TEXT)")
    con.execute(
        "INSERT INTO models (model_id, base_model, status, created_at)"
        " VALUES ('v1', 'B/M', 'promoted', '2026-01-01T00:00:00+00:00')")
    con.commit()
    con.close()

    reg = ModelRegistry(db)
    try:
        # Tolerant read of the legacy row: losses default to None.
        m = reg.get_model("v1")
        assert m is not None and m.train_loss is None and m.eval_loss is None
        reg.set_losses("v1", train_loss=1.25, eval_loss=0.9)
        m = reg.get_model("v1")
        assert m is not None
        assert m.train_loss == pytest.approx(1.25)
        assert m.eval_loss == pytest.approx(0.9)
    finally:
        reg.close()

    # Reopen: values persisted; migration is idempotent.
    reg2 = ModelRegistry(db)
    try:
        m = reg2.get_model("v1")
        assert m is not None
        assert m.eval_loss == pytest.approx(0.9)
    finally:
        reg2.close()


def test_set_losses_only_writes_provided_fields(tmp_path):
    reg = ModelRegistry(tmp_path / "r.db")
    try:
        reg.register_model("B/M")
        reg.set_losses("v1", train_loss=2.0)
        m = reg.get_model("v1")
        assert m is not None
        assert m.train_loss == pytest.approx(2.0) and m.eval_loss is None
        reg.set_losses("v1")  # no-op, must not clear anything
        m_after = reg.get_model("v1")
        assert m_after is not None
        assert m_after.train_loss == pytest.approx(2.0)
    finally:
        reg.close()


def test_parse_worker_losses(tmp_path):
    log = tmp_path / "job.log"
    log.write_text(
        json.dumps({"event": "start"}) + "\n"
        + json.dumps({"event": "progress", "step": 1, "loss": 2.0}) + "\n"
        + json.dumps({"event": "sft_done", "train_loss": 0.42,
                      "eval_loss": 0.5}) + "\n")
    assert parse_worker_losses(str(log)) == {"train_loss": 0.42,
                                             "eval_loss": 0.5}
    # older/stub worker logs without the fields -> both None
    log.write_text(json.dumps({"event": "done"}) + "\n")
    assert parse_worker_losses(str(log)) == {"train_loss": None,
                                             "eval_loss": None}
    assert parse_worker_losses(None) == {"train_loss": None, "eval_loss": None}
    assert parse_worker_losses(str(tmp_path / "missing.log"))["train_loss"] is None


# ---------------------------------------------------------- promotion gate

def test_gate_no_op_without_both_losses():
    for cand, parent in [(None, 1.0), (1.0, None), (None, None), (1.0, 0.0)]:
        res = eval_regression_check(cand, parent, gate="block")
        assert not res["blocked"] and not res["warned"]
        assert res["regression_pct"] is None


def test_gate_warn_default():
    # 2.0 vs parent 1.0 = +100% regression, limit 25% -> warn only by default
    res = eval_regression_check(2.0, 1.0)
    assert res["warned"] and not res["blocked"]
    assert res["regression_pct"] == pytest.approx(100.0)
    assert "eval_loss regressed" in res["message"]


def test_gate_block():
    res = eval_regression_check(1.3, 1.0, gate="block", max_pct=25.0)
    assert res["blocked"] and not res["warned"]
    # improvement / within tolerance never fires
    assert not eval_regression_check(0.9, 1.0, gate="block")["blocked"]
    assert not eval_regression_check(1.2, 1.0, gate="block", max_pct=25.0)["blocked"]


# ------------------------------------------------ pipeline wiring (stubbed)

STUB__test_eval_logging = r'''
import argparse, json, os, sys
from pathlib import Path
ap = argparse.ArgumentParser()
for f in (("--job", True), ("--kind", True), ("--config", False),
          ("--dataset-dir", False), ("--base-model", False),
          ("--log-path", False), ("--adapter-dir", False),
          ("--gguf-out", False)):
    ap.add_argument(f[0], required=f[1], default=None)
ap.add_argument("--quant", default="q4_k_m")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

def emit(ev, **kw):
    line = json.dumps({"event": ev, "job": a.job, **kw})
    print(line, flush=True)
    if a.log_path:
        Path(a.log_path).parent.mkdir(parents=True, exist_ok=True)
        open(a.log_path, "a").write(line + "\n")

emit("start")
if a.kind in ("sft", "dpo"):
    out = Path(a.config).parent if a.config else Path(".")
    (out / "adapter").mkdir(parents=True, exist_ok=True)
    (out / "adapter" / "adapter_config.json").write_text("{}")
    # backlog #4: terminal event carries the losses
    emit("sft_done", adapter_dir=str(out / "adapter"),
         train_loss=float(os.environ.get("STUB_TRAIN_LOSS", "1.0")),
         eval_loss=float(os.environ.get("STUB_EVAL_LOSS", "-1")) or None)
elif a.kind == "export":
    gg = Path(a.gguf_out)
    gg.parent.mkdir(parents=True, exist_ok=True)
    gg.write_bytes(b"GGUF-stub")
emit("done")
'''


@pytest.fixture()
def env(tmp_path, monkeypatch):
    home = tmp_path / "uf-eval"
    monkeypatch.setenv("SABLE_HOME", str(home))
    stub = tmp_path / "stub.py"
    stub.write_text(STUB__test_eval_logging)
    yield home, stub


def seed_trace__test_eval_logging(store, home, i, score=0.9):
    ts = f"2026-09-01T1{i:09d}+00:00"[:25]
    p = home / "buffer" / "2026" / "09" / f"ev-{i}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    msgs = [
        {"role": "user", "content": f"task {i}"},
        {"role": "assistant", "content": f"done task {i} result=ok"},
    ]
    with open(p, "w") as f:
        f.write(json.dumps({"kind": "trace_header"}) + "\n")
        for m in msgs:
            f.write(json.dumps({"kind": "message", "message": m}) + "\n")
        f.write(json.dumps({"kind": "trace_footer", "status": "success"}) + "\n")
    from sable.store.db import TraceRow
    store.insert_trace(TraceRow(
        trace_id=f"ev-{i}", session_id=None, timestamp=ts,
        agent_profile="fallback", model_alias="base-model",
        model_version=None, endpoint_label=None, task_summary="t",
        status="success", outcome_score=score, num_tool_calls=0,
        tokens_in=None, tokens_out=None, duration_s=None, jsonl_path=str(p),
    ))


def _starter(reply="ok-answer"):
    from fake_server import FakeOpenAIServer, make_scripted_handler

    def starter(gguf_path):
        srv = FakeOpenAIServer(handler_cls=make_scripted_handler({"*": reply}))
        srv.start()
        return srv
    return starter


def _run(home, stub, reg, store, monkeypatch, *, eval_loss, gate):
    from sable.curate import CurationConfig
    from sable.training.pipeline import run_iterate
    from sable.training.scheduler import Scheduler, TrainingConfig
    from sable.training.validate import ValidationConfig

    monkeypatch.setenv("STUB_EVAL_LOSS", str(eval_loss))
    sched = Scheduler(registry=reg, cfg=TrainingConfig(base_model="B/M"),
                      worker_command=[sys.executable, str(stub)])
    return run_iterate(
        home=home, registry=reg, scheduler=sched, store=store,
        curation_cfg=CurationConfig(min_score=0.5),
        training_cfg=TrainingConfig(base_model="B/M", eval_gate=gate,
                                    max_eval_regression_pct=25.0),
        validation_cfg=ValidationConfig(
            min_pass_rate=1.0,
            golden_tasks=[{"prompt": "q", "expect_contains": ["ok-answer"]}]),
        name="ev", server_starter=_starter(),
    )


def test_pipeline_stores_losses_and_gate_blocks(tmp_path, env, monkeypatch):
    from sable.store.db import TraceStore

    home, stub = env
    store = TraceStore(home / "buffer" / "sable.db")
    reg = ModelRegistry(home / "buffer" / "sable.db")
    for i in range(2):
        seed_trace__test_eval_logging(store, home, i)

    r1 = _run(home, stub, reg, store, monkeypatch, eval_loss=1.0, gate="warn")
    assert r1.ok and r1.promoted, r1.to_dict()
    m1 = reg.get_model("v1")
    assert m1 is not None
    assert m1.train_loss == pytest.approx(1.0)
    assert m1.eval_loss == pytest.approx(1.0)

    # second run: parent eval 1.0 -> candidate 2.0 (+100%)
    for i in range(2, 4):
        seed_trace__test_eval_logging(store, home, i)
    r2 = _run(home, stub, reg, store, monkeypatch, eval_loss=2.0, gate="warn")
    assert r2.ok and r2.promoted  # warn mode: promotion proceeds
    m2 = reg.get_model("v2")
    assert m2 is not None
    assert m2.eval_loss == pytest.approx(2.0)
    assert m2.validation is not None
    assert m2.validation["eval_gate"]["blocked"] is False

    store.close()
    reg.close()


def test_pipeline_gate_block_refuses_promotion(env, monkeypatch):
    from sable.store.db import TraceStore

    home, stub = env
    store = TraceStore(home / "buffer" / "sable.db")
    reg = ModelRegistry(home / "buffer" / "sable.db")
    for i in range(2):
        seed_trace__test_eval_logging(store, home, i)
    _run(home, stub, reg, store, monkeypatch, eval_loss=1.0, gate="warn")

    for i in range(2, 4):
        seed_trace__test_eval_logging(store, home, i)
    r2 = _run(home, stub, reg, store, monkeypatch, eval_loss=5.0, gate="block")
    assert not r2.ok and r2.stage == "promote"
    assert "eval gate blocked" in (r2.error or "")
    m2 = reg.get_model("v2")
    assert m2 is not None and m2.status != "promoted"
    active = reg.active_model()
    assert active is not None and active.model_id == "v1"  # parent stays live

    store.close()
    reg.close()


def test_pipeline_gate_skipped_without_eval_data(env, monkeypatch):
    """Stub worker without eval_loss -> gate silently skipped (compat)."""
    from sable.store.db import TraceStore

    home, stub = env
    monkeypatch.setenv("STUB_EVAL_LOSS", "0")  # -> None via `or None`
    store = TraceStore(home / "buffer" / "sable.db")
    reg = ModelRegistry(home / "buffer" / "sable.db")
    for i in range(2):
        seed_trace__test_eval_logging(store, home, i)
    r = _run(home, stub, reg, store, monkeypatch, eval_loss=0, gate="block")
    assert r.ok and r.promoted
    m = reg.get_model("v1")
    assert m is not None
    assert m.eval_loss is None

    store.close()
    reg.close()

# ------------------------------------------------------------------------
# from tests/test_evolve.py
# ------------------------------------------------------------------------

# Idea adapted from PIASystem (MIT) app/core/skill_evolution.py





BASE = "You are a careful assistant.\n\n## Tools\nUse tools wisely.\n"


def golden(prompt="say hi", contains="hi"):
    return {"prompt": prompt, "expect_contains": [contains]}


def replay(tid="r1", task="do the thing"):
    return {"task_id": tid, "task": task}


# ------------------------------------------------------------ constraint rules

def test_length_cap_formula():
    assert length_cap_for("x" * 100) == max(150, 900)  # base+800 wins small
    assert length_cap_for("x" * 2000) == 3000          # 1.5x wins large


def test_validate_rejects_over_length():
    cap = length_cap_for(BASE)
    ok, reason = validate_variant("y" * (cap + 1), base_text=BASE, cap=cap,
                                  must_keep_sections=[], seen={})
    assert not ok and "length cap" in (reason or "")


def test_validate_rejects_missing_must_keep_header():
    mutated = BASE.replace("## Tools", "## Instruments")
    ok, reason = validate_variant(mutated, base_text=BASE,
                                  cap=length_cap_for(BASE),
                                  must_keep_sections=["## Tools"], seen={})
    assert not ok and "must-keep" in (reason or "")


def test_validate_rejects_near_duplicate():
    mutated = BASE + "\n\n  \n"  # whitespace-only change -> ratio ~1.0
    ok, reason = validate_variant(mutated, base_text=BASE,
                                  cap=length_cap_for(BASE),
                                  must_keep_sections=[], seen={})
    assert not ok and "near-duplicate" in (reason or "")


def test_validate_rejects_duplicate_of_seen_variant():
    v1 = BASE + "\nAdd step ordering: 1) plan 2) act 3) verify carefully.\n"
    # near-identical to the recorded variant text (one char differs):
    twin = v1.replace("verify", "Verify")
    ok, reason = validate_variant(twin, base_text=BASE,
                                  cap=length_cap_for(BASE),
                                  must_keep_sections=[],
                                  seen={_norm(v1): "v7"})
    assert not ok and "v7" in (reason or "")


def test_validate_rejects_empty_accepts_clean():
    ok, reason = validate_variant("   ", base_text=BASE,
                                  cap=length_cap_for(BASE),
                                  must_keep_sections=[], seen={})
    assert not ok and "empty" in (reason or "")
    clean = BASE + "\nAlways verify results before reporting success.\n"
    ok, reason = validate_variant(clean, base_text=BASE,
                                  cap=length_cap_for(BASE),
                                  must_keep_sections=["## Tools"], seen={})
    assert ok and reason is None


def _norm(t):
    import re

    return re.sub(r"\s+", " ", t).strip()


# ----------------------------------------------------------------- search loop

class Stub:
    """Scripted world: mutate_fn pops variants; run/judge fns count calls."""

    def __init__(self, variants=None, golden_pass=("*",), judge_scores=None):
        self.variants = list(variants or [])
        self.directions_seen = []
        self.run_calls = []       # (prompt_head, task_prompt)
        self.judge_calls = 0
        self.golden_pass = golden_pass  # prompts whose replies pass checks
        self.judge_scores = judge_scores or {}

    def mutate(self, direction, prompt, cap, must_keep=None):
        self.directions_seen.append(direction)
        if not self.variants:
            return None
        head, text = self.variants.pop(0)
        # tag so golden/judge stubs can key off the variant identity
        return text + f"\n<!--{head}-->\n"

    def run_task(self, system_prompt, task):
        self.run_calls.append((system_prompt, task.get("prompt", "")))
        # A reply passes the substring golden check when the prompt carries a
        # marker from `golden_pass` (variant identity tags).
        passes = any(pat in system_prompt for pat in self.golden_pass)
        return {"content": "hi" if passes else "wrong answer"}

    def judge(self, system_prompt, rt, message):
        self.judge_calls += 1
        for key, score in self.judge_scores.items():
            if key in system_prompt:
                return score
        return 0.5


def test_search_golden_first_elimination_no_judge_calls_for_hard_fails():
    """A variant failing hard golden checks never reaches the judge."""
    stub = Stub(
        variants=[("bad", "Assistant.\n\n## Tools\nUse tools.\nbad")],
        golden_pass=("<!--good-->",),
    )
    result = run_evolution(
        BASE, [golden()], [replay()], k=1, generations=1, call_budget=50,
        mutate_fn=stub.mutate, run_task_fn=stub.run_task,
        judge_fn=stub.judge)
    bad = [e for e in result["lineage"] if e.get("direction")]
    assert bad and bad[0]["golden_pass"] is False
    # hard-failed variant: zero judge calls total (baseline had no replay pass
    # either because baseline itself fails the stubbed golden).
    assert stub.judge_calls == 0


def test_search_top2_selection_and_tie_prefers_shorter():
    v_short = ("You are an assistant. Plan first, then act; verify every "
               "result before reporting success. <!--s-->")
    v_long = v_short + "\n" + ("Additional padding sentence for length. " * 8) \
        .replace("<!--s-->", "") + "<!--l-->"
    stub = Stub(variants=[("s", v_short), ("l", v_long)],
                golden_pass=("<!--",))
    # both pass goldens; equal judge scores -> shorter must rank first
    result = run_evolution(
        BASE, [golden()], [replay()], k=2, generations=1, call_budget=60,
        mutate_fn=stub.mutate, run_task_fn=stub.run_task,
        judge_fn=lambda sp, rt, m: 0.5)
    scored = [e for e in result["lineage"] if e.get("golden_pass")]
    best = result["best"]
    assert best is not None
    assert "<!--s-->" in best["text"]  # tie -> shorter variant wins
    assert len(scored) >= 2


def test_search_lineage_and_direction_rotation():
    stub = Stub(variants=[(str(i), f"You are assistant number {i}. First make "
                                  f"a plan; then execute it step by step and "
                                  f"verify. unique{i} <!--v{i}-->")
                          for i in range(6)],
                golden_pass=("<!--",))
    result = run_evolution(
        BASE, [golden()], [], k=3, generations=1, call_budget=60,
        mutate_fn=stub.mutate, run_task_fn=stub.run_task,
        judge_fn=lambda sp, rt, m: 0.5)
    dirs = [e.get("direction") for e in result["lineage"]
            if e.get("direction")]
    assert dirs[:3] == list(MUTATION_DIRECTIONS[:3])
    ids = {e["variant_id"] for e in result["lineage"]}
    assert "v0" in ids and all(e["parent_id"] in ids
                               for e in result["lineage"])
    # every lineage entry carries the full shape
    for e in result["lineage"]:
        assert {"variant_id", "parent_id", "direction", "golden_pass",
                "judge_mean", "n_scored"} <= set(e)


def test_search_call_budget_stops_honestly_with_partial():
    stub = Stub(variants=[(str(i), f"You are assistant {i}. Plan, act, "
                                  f"verify; never skip verification. "
                                  f"<!--x{i}-->") for i in range(10)],
                golden_pass=("<!--",))
    result = run_evolution(
        BASE, [golden(), golden("again", "again")], [replay()],
        k=3, generations=2, call_budget=3,
        mutate_fn=stub.mutate, run_task_fn=stub.run_task,
        judge_fn=lambda sp, rt, m: 0.5)
    assert result["partial"] is True
    assert result["calls_used"] <= result["call_budget"] == 3


def test_search_no_goldens_raises_honestly():
    with pytest.raises(EvolveError, match="no golden tasks"):
        run_evolution(BASE, [], [replay()], mutate_fn=lambda **k: None,
                      run_task_fn=lambda *a: None,
                      judge_fn=lambda *a: None)


def test_search_improved_flag_and_diff():
    stub = Stub(variants=[("good", "You are an improved assistant. Always "
                                   "plan before acting and verify results "
                                   "against the request. <!--good-->")],
                golden_pass=("<!--",),
                judge_scores={"<!--good-->": 0.9})
    result = run_evolution(
        BASE, [golden()], [replay()], k=1, generations=1, call_budget=30,
        mutate_fn=stub.mutate, run_task_fn=stub.run_task,
        judge_fn=lambda sp, rt, m: 0.9 if "<!--good-->" in sp else 0.1)
    assert result["improved"] is True
    assert "--- current_prompt" in result["diff"]
    assert result["best"]["variant_id"] != "v0"


# ------------------------------------------------------------------ job kind

@contextmanager
def _endpoint():
    yield "http://127.0.0.1:9", "stub-model"


@pytest.fixture()
def home__test_evolve(tmp_path, monkeypatch):
    h = tmp_path / ".sable-test-home"
    monkeypatch.setenv("SABLE_CONFIG", str(h / "sable.yaml"))
    (h / "buffer").mkdir(parents=True)
    return h


def _job_params(**over):
    p = {"profile": "coder", "tasks": [golden()], "k": 1, "generations": 1,
         "call_budget": 10}
    p.update(over)
    return p


def test_evolve_job_runs_via_scheduler_and_stores_result(home__test_evolve, monkeypatch):
    monkeypatch.setattr("sable.training.scheduler.gpu_free_mib",
                        lambda: None)
    reg = ModelRegistry(home__test_evolve / "buffer" / "sable.db")
    stub = Stub(variants=[("good", "Assistant.\n\n## Tools\ntools <!--good-->")],
                golden_pass=("<!--",))

    def mutate(direction, prompt, cap, must_keep=None):
        return stub.mutate(direction, prompt, cap, must_keep)

    runner = make_evolve_job_runner(
        reg, home=home__test_evolve, defaults={"enabled": True, "max_call_budget": 60},
        endpoint_for=_endpoint,
        # production run/judge paths need HTTP; inject via mutate + monkeypatch
        )
    # Replace the heavy execution with the pure search over stubs:
    import sable.evaluation.evolve as ev

    monkeypatch.setattr(
        ev, "run_prompt_evolution",
        lambda base, goldens, replays, **kw: run_evolution(
            base, goldens, replays, k=kw.get("k", 1),
            generations=kw.get("generations", 1),
            call_budget=kw.get("call_budget", 10),
            mutate_fn=stub.mutate, run_task_fn=stub.run_task,
            judge_fn=lambda sp, rt, m: 0.8))

    sched = Scheduler(registry=reg, cfg=TrainingConfig(),
                      inprocess_runners={"evolve": runner})
    job = sched.submit("evolve", params=_job_params(base_prompt=BASE),
                       enqueue_actor="cli")
    done = sched.run_pending()
    assert [j.state for j in done] == ["done"]
    row = reg.get_job(job.job_id)
    assert row is not None
    res = row.params.get(RESULT_KEY)
    assert isinstance(res, dict)
    assert {"lineage", "best", "diff", "improved", "calls_used",
            "call_budget", "partial"} <= set(res)
    # NO registry mutation of the profile: proposal only.
    reg.close()


def test_evolve_job_refuses_when_disabled(home__test_evolve, monkeypatch):
    monkeypatch.setattr("sable.training.scheduler.gpu_free_mib",
                        lambda: None)
    reg = ModelRegistry(home__test_evolve / "buffer" / "sable.db")
    runner = make_evolve_job_runner(
        reg, home=home__test_evolve, defaults={"enabled": False},
        endpoint_for=_endpoint)
    sched = Scheduler(registry=reg, cfg=TrainingConfig(),
                      inprocess_runners={"evolve": runner})
    job = sched.submit("evolve", params=_job_params(base_prompt=BASE))
    done = sched.run_pending()
    assert done[0].state == "error"
    assert "evolution.enabled" in (done[0].error or "")
    reg.close()


def test_evolve_job_honest_empty_no_goldens(home__test_evolve, monkeypatch):
    monkeypatch.setattr("sable.training.scheduler.gpu_free_mib",
                        lambda: None)
    reg = ModelRegistry(home__test_evolve / "buffer" / "sable.db")
    runner = make_evolve_job_runner(
        reg, home=home__test_evolve, defaults={"enabled": True}, endpoint_for=_endpoint)
    sched = Scheduler(registry=reg, cfg=TrainingConfig(),
                      inprocess_runners={"evolve": runner})
    p = _job_params(base_prompt=BASE)
    p.pop("tasks")
    job = sched.submit("evolve", params=p)
    done = sched.run_pending()
    assert done[0].state == "error"
    assert "golden" in (done[0].error or "").lower()
    reg.close()


def test_evolve_job_kind_accepted_by_registry_and_migration(home__test_evolve):
    reg = ModelRegistry(home__test_evolve / "buffer" / "sable.db")
    job = reg.enqueue_job("evolve", params={"profile": "coder"})
    assert job.state == "queued"
    reg.close()
    # reopen: CHECK already contains 'evolve' -> migration is a no-op
    reg2 = ModelRegistry(home__test_evolve / "buffer" / "sable.db")
    assert reg2.get_job(job.job_id) is not None
    reg2.close()


# ----------------------------------------------------------------- apply flow

def _write_cfg(path: Path, prompt="old prompt text"):
    path.write_text(yaml.safe_dump({
        "profiles": [{"name": "coder", "system_prompt": prompt,
                      "tools": ["read_file"]}],
        "default_profile": "coder",
    }, sort_keys=False))


def test_apply_updates_profile_and_keeps_backup(home__test_evolve):
    cfg = home__test_evolve / "sable.yaml"
    _write_cfg(cfg)
    out = apply_prompt_to_config(cfg, "coder", "new evolved prompt")
    data = yaml.safe_load(cfg.read_text())
    prof = data["profiles"][0]
    assert prof["system_prompt"] == "new evolved prompt"
    assert prof["system_prompt_prev"] == "old prompt text"
    assert prof["prompt_evolved_at"]
    assert Path(out["backup"]).is_file()


def test_apply_missing_profile_raises(home__test_evolve):
    cfg = home__test_evolve / "sable.yaml"
    _write_cfg(cfg)
    with pytest.raises(EvolveError, match="no profile"):
        apply_prompt_to_config(cfg, "ghost", "x")


def test_dismiss_and_applied_flags_persist_without_deleting_history(home__test_evolve):
    reg = ModelRegistry(home__test_evolve / "buffer" / "sable.db")
    job = reg.enqueue_job("evolve", params={"profile": "coder"})
    reg.finish_job(job.job_id, "done")
    reg.set_job_param(job.job_id, RESULT_KEY, {"improved": True})
    reg.set_job_param(job.job_id, DISMISSED_KEY, "2026-01-01T00:00:00+00:00")
    row = reg.get_job(job.job_id)
    assert row is not None
    assert row.params[DISMISSED_KEY].startswith("2026")
    assert row.params[RESULT_KEY] == {"improved": True}  # history intact
    reg.set_job_param(job.job_id, APPLIED_KEY, "2026-01-02T00:00:00+00:00")
    row = reg.get_job(job.job_id)
    assert row is not None
    assert row.params[APPLIED_KEY] and row.params[RESULT_KEY]
    reg.close()


def test_plain_run_never_touches_profile_config(home__test_evolve, monkeypatch):
    """Guardrail: an evolve job NEVER mutates the config file."""
    monkeypatch.setattr("sable.training.scheduler.gpu_free_mib",
                        lambda: None)
    cfg = home__test_evolve / "sable.yaml"
    _write_cfg(cfg)
    before = cfg.read_text()

    reg = ModelRegistry(home__test_evolve / "buffer" / "sable.db")
    stub = Stub(variants=[("good", "Assistant.\n\n## Tools\ntools <!--g-->")],
                golden_pass=("<!--",))
    import sable.evaluation.evolve as ev

    monkeypatch.setattr(
        ev, "run_prompt_evolution",
        lambda base, goldens, replays, **kw: run_evolution(
            base, goldens, replays, k=1, generations=1, call_budget=10,
            mutate_fn=stub.mutate, run_task_fn=stub.run_task,
            judge_fn=lambda sp, rt, m: 0.9))
    runner = make_evolve_job_runner(
        reg, home=home__test_evolve, defaults={"enabled": True}, endpoint_for=_endpoint)
    sched = Scheduler(registry=reg, cfg=TrainingConfig(),
                      inprocess_runners={"evolve": runner})
    sched.submit("evolve", params=_job_params(base_prompt=BASE))
    done = sched.run_pending()
    assert done[0].state == "done"
    assert cfg.read_text() == before  # proposal only — config untouched
    reg.close()


# ------------------------------------------------------------- dashboard API

def _dash_app__test_evolve(home):
    from sable.core.config import SableConfig
    from sable.dashboard.app import DashboardApp

    return DashboardApp(SableConfig(), home=home)


def test_dashboard_evolve_endpoints(home__test_evolve, monkeypatch):
    cfg = home__test_evolve / "sable.yaml"
    _write_cfg(cfg)
    reg = ModelRegistry(home__test_evolve / "buffer" / "sable.db")
    job = reg.enqueue_job("evolve", params={"profile": "coder"})
    reg.set_job_param(job.job_id, RESULT_KEY, {
        "ts": "2026-01-01T00:00:00+00:00", "improved": True,
        "diff": "--- a\n+++ b\n+new", "calls_used": 5, "call_budget": 60,
        "partial": False,
        "baseline": {"variant_id": "v0", "golden_pass": True,
                     "judge_mean": 0.4},
        "best": {"variant_id": "v1", "direction": "tighten instructions",
                 "golden_pass": True, "judge_mean": 0.8,
                 "text": "brand new prompt"},
    })
    reg.finish_job(job.job_id, "done")
    reg.close()

    app = _dash_app__test_evolve(home__test_evolve)
    r = app.handle("GET", "/api/evolve", query={"profile": "coder"})
    assert r.status == 200
    prop = r.body["proposal"]
    assert prop["job_id"] == job.job_id and prop["improved"]
    assert prop["fitness_current"] == 0.4
    assert prop["fitness_proposed"] == 0.8

    # apply is gated: explicit POST -> profile prompt replaced + backup
    r = app.handle("POST", "/api/evolve/apply", body={"job_id": job.job_id})
    assert r.status == 200 and r.body["ok"]
    data = yaml.safe_load(cfg.read_text())
    assert data["profiles"][0]["system_prompt"] == "brand new prompt"
    assert data["profiles"][0]["system_prompt_prev"] == "old prompt text"

    # second apply refused (409), history untouched
    r = app.handle("POST", "/api/evolve/apply", body={"job_id": job.job_id})
    assert r.status == 409

    # dismiss flag on another proposal
    job2 = reg2 = None
    reg2 = ModelRegistry(home__test_evolve / "buffer" / "sable.db")
    j2 = reg2.enqueue_job("evolve", params={"profile": "coder"})
    reg2.set_job_param(j2.job_id, RESULT_KEY, {"improved": False})
    reg2.finish_job(j2.job_id, "done")
    reg2.close()
    r = app.handle("POST", "/api/evolve/dismiss", body={"job_id": j2.job_id})
    assert r.status == 200 and r.body["dismissed"] == j2.job_id

    assert app.handle("GET", "/api/evolve").status == 400
    assert app.handle("POST", "/api/evolve/dismiss",
                      body={"job_id": "nope"}).status == 404


# ----------------------------------------------------------------- CLI smoke

def test_cli_evolve_refuses_when_disabled(home__test_evolve, monkeypatch, capsys):
    from sable.cli import main

    cfg = home__test_evolve / "sable.yaml"
    _write_cfg(cfg)
    monkeypatch.setenv("SABLE_CONFIG", str(cfg))
    rc = main(["evolve", "coder", "--goldens", "x.json"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "evolution.enabled" in err


def test_cli_evolve_apply_updates_config(home__test_evolve, monkeypatch, capsys):
    from sable.cli import main

    cfg = home__test_evolve / "sable.yaml"
    _write_cfg(cfg)
    monkeypatch.setenv("SABLE_CONFIG", str(cfg))
    reg = ModelRegistry(home__test_evolve / "buffer" / "sable.db")
    job = reg.enqueue_job("evolve", params={"profile": "coder"})
    reg.set_job_param(job.job_id, RESULT_KEY, {
        "improved": True, "best": {"variant_id": "v1",
                                   "text": "cli evolved prompt"}})
    reg.finish_job(job.job_id, "done")
    reg.close()

    rc = main(["evolve", "apply", job.job_id])
    assert rc == 0
    data = yaml.safe_load(cfg.read_text())
    assert data["profiles"][0]["system_prompt"] == "cli evolved prompt"
    assert data["profiles"][0]["system_prompt_prev"] == "old prompt text"

    # applying twice refuses (already applied) — gated, single-shot
    rc = main(["evolve", "apply", job.job_id])
    assert rc == 1
    assert "already applied" in capsys.readouterr().err


def test_cli_evolve_dismiss_flags_job(home__test_evolve, monkeypatch):
    from sable.cli import main

    cfg = home__test_evolve / "sable.yaml"
    _write_cfg(cfg)
    monkeypatch.setenv("SABLE_CONFIG", str(cfg))
    reg = ModelRegistry(home__test_evolve / "buffer" / "sable.db")
    job = reg.enqueue_job("evolve", params={"profile": "coder"})
    reg.set_job_param(job.job_id, RESULT_KEY, {"improved": False})
    reg.finish_job(job.job_id, "done")
    reg.close()

    assert main(["evolve", "dismiss", job.job_id]) == 0
    reg2 = ModelRegistry(home__test_evolve / "buffer" / "sable.db")
    row = reg2.get_job(job.job_id)
    assert row is not None
    assert row.params[DISMISSED_KEY] and row.params[RESULT_KEY]
    reg2.close()

# ------------------------------------------------------------------------
# from tests/test_provenance.py
# ------------------------------------------------------------------------

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

# ------------------------------------------------------------------------
# from tests/test_wp9_analytics.py
# ------------------------------------------------------------------------

@pytest.fixture()
def home__test_wp9_analytics(tmp_path, monkeypatch):
    h = tmp_path / "uf-wp9"
    monkeypatch.setenv("SABLE_HOME", str(h))
    return h


@pytest.fixture()
def app__test_wp9_analytics(home__test_wp9_analytics):
    return DashboardApp(SableConfig(), home=home__test_wp9_analytics)


# ------------------------------------------------------------------- seeding

def seed_trace__test_wp9_analytics(store: TraceStore, home: Path, *, trace_id: str, messages,
               status: str = "success", score: Optional[float] = 0.8,
               timestamp: str = "2026-09-01T10:00:00+00:00") -> None:
    """Minimal transcript + index row (same shape as tests/test_curate.py)."""
    jsonl_path = home / "buffer" / "2026" / "09" / f"{trace_id}.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "x", encoding="utf-8") as f:
        f.write(json.dumps({
            "kind": "trace_header", "schema_version": 1, "trace_id": trace_id,
            "session_id": None, "timestamp": timestamp,
            "agent_profile": "fallback", "model_alias": "test-model",
            "model_version": None, "endpoint_label": None,
            "task": (messages[0].get("content") if messages else "") or "",
        }) + "\n")
        for m in messages:
            f.write(json.dumps({"kind": "message", "message": m}) + "\n")
        f.write(json.dumps({
            "kind": "trace_footer", "schema_version": 1, "status": status,
            "completed": status == "success", "outcome_score": score,
            "num_tool_calls": 0, "tokens_in": None, "tokens_out": None,
            "duration_s": None,
        }) + "\n")
    store.insert_trace(TraceRow(
        trace_id=trace_id, session_id=None, timestamp=timestamp,
        agent_profile="fallback", model_alias="test-model",
        model_version=None, endpoint_label=None, task_summary="seed",
        status=status, outcome_score=score, num_tool_calls=0, tokens_in=None,
        tokens_out=None, duration_s=None, jsonl_path=str(jsonl_path),
    ))


OK_MSGS = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"},
]
DUP_MSGS = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "four"},
    {"role": "user", "content": "sure?"},
    {"role": "assistant", "content": "yes, four"},
]
OTHER_MSGS = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Capital of France?"},
    {"role": "assistant", "content": "Paris"},
]


# ================================================================== 9a funnel

def test_funnel_snapshot_persisted_with_honest_denominators(home__test_wp9_analytics):
    store = TraceStore(home__test_wp9_analytics / "buffer" / "sable.db")
    try:
        seed_trace__test_wp9_analytics(store, home__test_wp9_analytics, trace_id="t1", messages=OK_MSGS, score=0.9)
        seed_trace__test_wp9_analytics(store, home__test_wp9_analytics, trace_id="t2", messages=OTHER_MSGS, score=None)
        res = Curater(store, home__test_wp9_analytics).curate(CurationConfig(min_score=0.5),
                                          name="wp9")
    finally:
        store.close()

    recs = analytics.read_curation_funnels(home__test_wp9_analytics)
    assert len(recs) == 1
    r = recs[0]
    assert r["dataset_id"] == res.dataset_id
    assert r["traces_scanned"] == 2
    assert r["loaded"] == 2
    assert r["curated"] == res.claimed
    # honest denominators: histogram carries scored/unscored beside buckets
    hist = r["score_histogram"]
    assert hist["scored"] == 1 and hist["unscored"] == 1
    assert hist["total"] == 2
    assert sum(hist["bins"]) == hist["scored"]
    # 0.9 -> bin index 9
    assert hist["bins"][9] == 1
    assert r["ts"] and r["created_at"]


def test_funnel_history_accumulates_over_runs(home__test_wp9_analytics):
    store = TraceStore(home__test_wp9_analytics / "buffer" / "sable.db")
    try:
        seed_trace__test_wp9_analytics(store, home__test_wp9_analytics, trace_id="a1", messages=OK_MSGS)
        Curater(store, home__test_wp9_analytics).curate(CurationConfig(), name="one")
        seed_trace__test_wp9_analytics(store, home__test_wp9_analytics, trace_id="b1", messages=OTHER_MSGS,
                   timestamp="2026-09-01T11:00:00+00:00")
        Curater(store, home__test_wp9_analytics).curate(CurationConfig(), name="two")
    finally:
        store.close()
    recs = analytics.read_curation_funnels(home__test_wp9_analytics)
    assert len(recs) == 2  # append-only history
    ids = [r["dataset_id"] for r in recs]
    assert len(set(ids)) == 2
    # dataset filter returns only that dataset's runs
    one = analytics.read_curation_funnels(home__test_wp9_analytics, dataset=ids[0])
    assert [r["dataset_id"] for r in one] == [ids[0]]


def test_funnel_dedup_rejects_counted_on_dup_fixture(home__test_wp9_analytics):
    store = TraceStore(home__test_wp9_analytics / "buffer" / "sable.db")
    try:
        # same first-user turn + profile -> one conversation lineage; the
        # longer (DUP) trace wins, the shorter one is a dedup reject.
        seed_trace__test_wp9_analytics(store, home__test_wp9_analytics, trace_id="d1", messages=OK_MSGS)
        seed_trace__test_wp9_analytics(store, home__test_wp9_analytics, trace_id="d2", messages=DUP_MSGS,
                   timestamp="2026-09-01T10:30:00+00:00")
        res = Curater(store, home__test_wp9_analytics).curate(CurationConfig(), name="dupes")
    finally:
        store.close()
    r = analytics.read_curation_funnels(home__test_wp9_analytics)[-1]
    assert r["traces_scanned"] == 2
    assert r["dedup_rejects"] == 1
    assert r["curated"] == res.claimed == 1


def test_funnel_api_returns_history_and_latest_diff(app__test_wp9_analytics, home__test_wp9_analytics):
    store = TraceStore(home__test_wp9_analytics / "buffer" / "sable.db")
    try:
        seed_trace__test_wp9_analytics(store, home__test_wp9_analytics, trace_id="x1", messages=OK_MSGS)
        Curater(store, home__test_wp9_analytics).curate(CurationConfig(), name="r1")
        seed_trace__test_wp9_analytics(store, home__test_wp9_analytics, trace_id="x2", messages=OTHER_MSGS,
                   timestamp="2026-09-02T10:00:00+00:00")
        Curater(store, home__test_wp9_analytics).curate(CurationConfig(), name="r2")
    finally:
        store.close()

    resp = app__test_wp9_analytics.handle("GET", "/api/analytics/curation")
    assert resp.status == 200
    body = resp.body
    assert len(body["history"]) == 2
    # latest_diff reuses dataset_diff over the two newest datasets
    diff = body["latest_diff"]
    assert diff is not None and "error" not in diff
    assert set(diff) >= {"trace_diff", "counts", "identical"}

    resp1 = app__test_wp9_analytics.handle("GET", "/api/analytics/curation",
                       query={"dataset": body["history"][0]["dataset_id"]})
    assert resp1.status == 200
    assert len(resp1.body["history"]) == 1
    # single dataset -> no pair to diff, honest null (not an error)
    assert resp1.body["latest_diff"] is None


# ============================================================ 9b validation

def _fake_gate_report(passed: int, total: int, ok: bool) -> Dict:
    results = [{"prompt": f"p{i}", "pass": i < passed, "checks": [
        {"kind": "substring", "pass": i < passed}]} for i in range(total)]
    return {"ok": ok, "pass_rate": passed / total if total else 0.0,
            "min_pass_rate": 1.0, "total": total, "passed": passed,
            "results": results, "started_at": "2026-10-01T10:00:00+00:00",
            "finished_at": "2026-10-01T10:01:00+00:00"}


def test_validation_summary_counts_and_skips():
    rep = _fake_gate_report(1, 3, False)
    rep["results"][2] = {"prompt": "p2", "pass": False,
                         "error": "ConnectionError: boom"}
    rec = analytics.summarize_validation_report(rep, "v7", iteration=2)
    assert rec["model_id"] == "v7" and rec["iteration"] == 2
    assert rec["passed"] == 1 and rec["failed"] == 1 and rec["skipped"] == 1
    assert rec["gate_verdict"] == "fail"
    assert rec["ran_tasks"] == 2  # transport-skips excluded from the ran set


def test_validation_api_merges_promotion_events_chronologically(app__test_wp9_analytics, home__test_wp9_analytics):
    # two fake gate runs, explicitly timestamped for deterministic ordering
    analytics.record_validation_run(home__test_wp9_analytics, {
        "ts": "2026-10-01T10:00:00+00:00", **analytics.summarize_validation_report(
            _fake_gate_report(1, 2, False), "v1")})
    analytics.record_validation_run(home__test_wp9_analytics, {
        "ts": "2026-10-03T10:00:00+00:00", **analytics.summarize_validation_report(
            _fake_gate_report(2, 2, True), "v1")})

    # registry: v1 validated -> promoted (ledger records both ops)
    reg = ModelRegistry(home__test_wp9_analytics / "buffer" / "sable.db")
    try:
        reg.register_model("base-a", dataset_id=None)
        reg.set_validation("v1", {"ok": True}, passed=True, actor="test")
        reg.promote("v1", actor="test", intent="wp9 test promote")
    finally:
        reg.close()

    resp = app__test_wp9_analytics.handle("GET", "/api/analytics/validation", query={"model": "v1"})
    assert resp.status == 200
    body = resp.body
    assert [r["pass_rate"] for r in body["history"]] == [0.5, 1.0]
    kinds = [e["op"] for e in body["events"]]
    assert "promote" in kinds
    tl = body["timeline"]
    ts = [e["ts"] for e in tl]
    assert ts == sorted(ts)  # chronological merge
    assert {e["kind"] for e in tl} == {"validation", "promotion"}


# ================================================================ 9c metrics

def _metrics_text__test_wp9_analytics(app) -> str:
    return app.handle("GET", "/api/metrics").raw.decode("utf-8")


def test_metrics_exposition_parses_back_with_help_type(app__test_wp9_analytics, home__test_wp9_analytics):
    # seed one trace so counters are non-trivial
    store = TraceStore(home__test_wp9_analytics / "buffer" / "sable.db")
    try:
        seed_trace__test_wp9_analytics(store, home__test_wp9_analytics, trace_id="m1", messages=OK_MSGS)
    finally:
        store.close()
    text = _metrics_text__test_wp9_analytics(app__test_wp9_analytics)
    assert text.endswith("\n")
    # HELP immediately followed by TYPE for every family that has samples
    lines = text.splitlines()
    helped = set()
    typed = set()
    for i, line in enumerate(lines):
        if line.startswith("# HELP "):
            name = line.split()[2]
            assert lines[i + 1].startswith(f"# TYPE {name} "), \
                f"HELP {name} not followed by TYPE"
            helped.add(name)
        elif line.startswith("# TYPE "):
            typed.add(line.split()[2])
    assert helped == typed
    # every sample line's family (minus _sum/_count/quantile suffixes) is known
    for line in lines:
        if line.startswith("#") or not line:
            continue
        head = line.split("{")[0].split(" ")[0]
        base = head
        for suffix in ("_sum", "_count"):
            if head.endswith(suffix):
                base = head[:-len(suffix)]
                break
        assert base in helped or head in helped, f"orphan sample {head}"


def test_metrics_label_escaping_quotes_backslash(app__test_wp9_analytics, home__test_wp9_analytics):
    """Instance id with quote/backslash/newline-free edge chars must be escaped
    per spec: \\\\ , \\" inside label values; the raw line stays parseable."""
    inst_dir = home__test_wp9_analytics / "serving" / "instances"
    inst_dir.mkdir(parents=True)
    (inst_dir / "weird.json").write_text(json.dumps(
        {"id": 'a"b\\c', "name": "weird", "role": "custom"}), encoding="utf-8")
    text = _metrics_text__test_wp9_analytics(app__test_wp9_analytics)
    # the escaped form appears; the raw quote inside a value never does
    assert '{id="a\\"b\\\\c"}' in text
    # every label value on that line is well-formed: value chars are either
    # non-quote/non-backslash or valid escapes
    for line in text.splitlines():
        if 'a\\"b' in line:
            assert re.fullmatch(
                r'sable_instances_up\{id="(?:[^"\\]|\\.)*"\} [0-9]+', line), line
            break
    else:
        pytest.fail("escaped instance sample not found")


def test_metrics_vllm_mirrors_only_where_semantics_match(app__test_wp9_analytics, home__test_wp9_analytics):
    # one scored trace so the WP2c coverage ratio family is emitted at all,
    # and one promotion decision so the pass-rate denominator pair exists
    store = TraceStore(home__test_wp9_analytics / "buffer" / "sable.db")
    try:
        seed_trace__test_wp9_analytics(store, home__test_wp9_analytics, trace_id="mm1", messages=OK_MSGS, score=0.5)
    finally:
        store.close()
    reg = ModelRegistry(home__test_wp9_analytics / "buffer" / "sable.db")
    try:
        reg.register_model("base-a")
        reg.set_validation("v1", {"ok": True}, passed=True, actor="test")
        reg.promote("v1", actor="test")
    finally:
        reg.close()
    text = _metrics_text__test_wp9_analytics(app__test_wp9_analytics)
    fams = {l.split()[2] for l in text.splitlines() if l.startswith("# TYPE")}
    # semantics match -> mirrors present
    assert "vllm:prompt_tokens_total" in fams
    assert "vllm:generation_tokens_total" in fams
    # semantics differ / not measurable -> deliberately absent
    assert "vllm:num_requests_total" not in fams
    assert "vllm:time_to_first_token_seconds" not in fams
    assert "vllm:num_requests_waiting" not in fams
    # canonical sable families intact (WP2 honest variants included)
    for fam in ("sable_traces_total", "sable_traces_scored_total",
                "sable_score_coverage_ratio", "sable_promotion_decisions_total",
                "sable_instance_up_strength", "sable_requests_total",
                "sable_jobs_queued"):
        assert fam in fams, f"missing family {fam}"


def test_metrics_token_auth_parity_on_metrics_route(home__test_wp9_analytics):
    """GET /metrics enforces the dashboard token exactly like /api/metrics —
    no unauthenticated backdoor on the scrape alias."""
    from sable.dashboard.server import BIND_HOST, make_server

    httpd = make_server(SableConfig(), port=0, home=home__test_wp9_analytics, token="scr4pe")
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    base = f"http://{BIND_HOST}:{httpd.server_address[1]}"
    try:
        import httpx
        assert httpx.get(base + "/metrics", timeout=5).status_code == 401
        r = httpx.get(base + "/metrics", timeout=5,
                      headers={"X-Sable-Token": "scr4pe"})
        assert r.status_code == 200
        assert r.headers["content-type"] == "text/plain; version=0.0.4"
        assert "# HELP" in r.text
        # GET query token allowed (scrapers cannot set headers)
        assert httpx.get(base + "/metrics?token=scr4pe",
                         timeout=5).status_code == 200
        # cross-site Origin still blocked on the alias
        assert httpx.get(base + "/metrics", timeout=5,
                         headers={"Origin": "http://evil.example"},
                         ).status_code == 403
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_metrics_route_200_content_type_unauth_disabled(home__test_wp9_analytics):
    """HTTP-level happy path: token disabled -> /metrics is scrapeable."""
    from sable.dashboard.server import BIND_HOST, make_server

    httpd = make_server(SableConfig(), port=0, home=home__test_wp9_analytics, token="")
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    base = f"http://{BIND_HOST}:{httpd.server_address[1]}"
    try:
        import httpx
        r = httpx.get(base + "/metrics", timeout=5)
        assert r.status_code == 200
        assert r.headers["content-type"] == "text/plain; version=0.0.4"
        # /api/metrics still works identically (pre-existing route unchanged)
        r2 = httpx.get(base + "/api/metrics", timeout=5)
        assert r2.status_code == 200
        assert r2.text == r.text  # same TTL-cached body
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_metrics_no_rate_without_denominator(app__test_wp9_analytics):
    """WP9 honesty rule over the rendered text: every *_ratio / *_rate gauge
    must be accompanied by its denominator counter family."""
    text = _metrics_text__test_wp9_analytics(app__test_wp9_analytics)
    fams = {l.split()[2] for l in text.splitlines() if l.startswith("# TYPE")}
    pairs = {
        "sable_score_coverage_ratio": "sable_traces_scored_total",
        "sable_promotion_pass_rate": "sable_promotion_decisions_total",
    }
    for rate, denom in pairs.items():
        if rate in text:  # only emitted when non-empty
            assert f"# TYPE {denom} counter" in text, \
                f"rate {rate} rendered without denominator {denom}"
    assert fams  # sanity: something was parsed

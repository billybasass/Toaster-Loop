"""GRPO training kind: prompt-row derivation, rule rewards, worker dry-run,
and the grpo job kind end-to-end through the registry. No torch needed —
dry-run paths and pure functions only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sable.training.worker import (  # noqa: E402
    grpo_rows_from_dataset,
    main as worker_main,
    reward_format,
    reward_length,
    reward_solution_overlap,
)


def _write_jsonl(p: Path, rows):
    p.write_text("\n".join(json.dumps(r) for r in rows))


# ---------------------------------------------------------- row derivation

def test_rows_from_explicit_grpo_file(tmp_path):
    _write_jsonl(tmp_path / "grpo.jsonl", [
        {"prompt": [{"role": "user", "content": "hi"}], "solution": "yo"},
        {"prompt": [], "solution": "dropped (empty prompt)"},
    ])
    rows = grpo_rows_from_dataset(tmp_path)
    assert len(rows) == 1
    assert rows[0]["solution"] == "yo"


def test_rows_fall_back_to_dpo_chosen_as_solution(tmp_path):
    _write_jsonl(tmp_path / "dpo.jsonl", [{
        "prompt": [{"role": "user", "content": "q"}],
        "chosen": "good", "rejected": "bad",
    }])
    rows = grpo_rows_from_dataset(tmp_path)
    assert rows == [{"prompt": [{"role": "user", "content": "q"}],
                     "solution": "good"}]


def test_rows_fall_back_to_sft_last_turn(tmp_path):
    _write_jsonl(tmp_path / "sft.jsonl", [{
        "conversations": [
            {"role": "user", "value": "q"},
            {"role": "assistant", "value": "a"},
        ]}])
    rows = grpo_rows_from_dataset(tmp_path)
    assert len(rows) == 1
    assert rows[0]["solution"] == "a"
    assert rows[0]["prompt"] == [{"role": "user", "value": "q"}]


def test_rows_missing_dataset_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        grpo_rows_from_dataset(tmp_path)


# ------------------------------------------------------------------ rewards

def test_reward_format_scores():
    scores = reward_format(["a real answer with words", "", "x x x x x x x x x x" * 5])
    assert scores[0] == 1.0
    assert scores[1] == 0.0
    assert scores[2] < 1.0  # degenerate repetition penalized


def test_reward_format_accepts_chat_completion_shape():
    scores = reward_format([[{"role": "assistant", "content": "hello there"}]])
    assert scores == [1.0]


def test_reward_length_decay():
    scores = reward_length(["short", "long" * 4000], max_chars=100)
    assert scores[0] == 1.0
    assert scores[1] == 0.0  # way over budget -> floor at 0


def test_reward_solution_overlap_f1():
    scores = reward_solution_overlap(
        ["alpha beta gamma"], solutions=["alpha beta delta"])
    # intersection 2, precision 2/3, recall 2/3 -> F1 2/3
    assert scores[0] == pytest.approx(2 / 3)
    assert reward_solution_overlap(["anything"], solutions=[""])[0] == 0.0


# ------------------------------------------------------- worker dry-run CLI

def _train_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "train.yaml"
    p.write_text("model: some/base\ntraining:\n  output_dir: outputs\n")
    return p


def test_worker_grpo_dry_run(tmp_path):
    ds = tmp_path / "ds"
    ds.mkdir()
    _write_jsonl(ds / "grpo.jsonl", [
        {"prompt": [{"role": "user", "content": "q1"}], "solution": "a1"},
        {"prompt": [{"role": "user", "content": "q2"}], "solution": "a2"},
    ])
    log = tmp_path / "job.log"
    rc = worker_main([
        "--job", "j-grpo", "--kind", "grpo",
        "--config", str(_train_yaml(tmp_path)),
        "--dataset-dir", str(ds),
        "--log-path", str(log),
        "--dry-run",
    ])
    assert rc == 0
    events = [json.loads(l) for l in log.read_text().splitlines()]
    kinds = [e.get("event") for e in events]
    assert "dry_run_ok" in kinds
    dry = next(e for e in events if e.get("event") == "dry_run_ok")
    assert dry["kind"] == "grpo" and dry["rows"] == 2
    assert kinds[-1] == "done"


def test_registry_accepts_grpo_kind(tmp_path):
    from sable.registry import ModelRegistry

    reg = ModelRegistry(tmp_path / "r.db")
    try:
        job = reg.enqueue_job("grpo", dataset_id="ds1")
        assert job.kind == "grpo" and job.state == "queued"
    finally:
        reg.close()


def test_registry_set_job_param_merges(tmp_path):
    from sable.registry import ModelRegistry

    reg = ModelRegistry(tmp_path / "r.db")
    try:
        job = reg.enqueue_job("sft", params={"a": 1})
        reg.set_job_param(job.job_id, "_device", 2)
        got = reg.get_job(job.job_id)
        assert got is not None
        assert got.params == {"a": 1, "_device": 2}
    finally:
        reg.close()

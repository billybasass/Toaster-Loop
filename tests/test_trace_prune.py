"""Buffer retention: TraceStore.prune_traces + `sable traces prune` CLI.

Covers the production-readiness gap "the append-only buffer grows forever":
  1. dry_run deletes nothing and reports counts;
  2. real runs delete index rows AND transcript JSONL files;
  3. curated traces survive by default, go with keep_curated=False;
  4. feedback rows cascade with the trace;
  5. CLI: no window configured -> exit 1; --older-than + --yes prunes;
     config recording.retention_days supplies the implicit window.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sable.cli import main  # noqa: E402
from sable.store.db import TraceRow, TraceStore  # noqa: E402


def _mk(store: TraceStore, home: Path, tid: str, ts: str,
        curated: bool = False) -> TraceRow:
    jp = home / "buffer" / f"{tid}.jsonl"
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text("x" * 100, encoding="utf-8")
    row = TraceRow(trace_id=tid, session_id="s", timestamp=ts,
                   agent_profile="p", model_alias="m", model_version=None,
                   endpoint_label=None, task_summary="t", status="success",
                   outcome_score=0.5, num_tool_calls=0, tokens_in=1,
                   tokens_out=1, duration_s=1.0, jsonl_path=str(jp),
                   curated=curated)
    store.insert_trace(row)
    store.add_feedback(tid, 0.9, "nice", source="user", timestamp=ts)
    return row


@pytest.fixture()
def store(tmp_path):
    s = TraceStore(tmp_path / "buffer" / "sable.db")
    yield s, tmp_path
    s.close()


def test_prune_dry_run_keeps_everything(store):
    s, home = store
    _mk(s, home, "old1", "2025-01-01T00:00:00+00:00")
    summ = s.prune_traces("2026-01-01T00:00:00+00:00", dry_run=True)
    assert summ["dry_run"] and summ["deleted"] == 1
    assert (home / "buffer" / "old1.jsonl").exists()
    assert s.get_trace("old1") is not None


def test_prune_deletes_rows_transcripts_and_feedback(store):
    s, home = store
    _mk(s, home, "old1", "2025-01-01T00:00:00+00:00")
    _mk(s, home, "new1", "2026-10-01T00:00:00+00:00")
    summ = s.prune_traces("2026-01-01T00:00:00+00:00")
    assert summ["deleted"] == 1 and summ["freed_bytes"] >= 100
    assert not (home / "buffer" / "old1.jsonl").exists()
    assert s.get_trace("old1") is None
    assert s.get_trace("new1") is not None
    assert s.get_feedback("old1") == []


def test_prune_keeps_curated_unless_forced(store):
    s, home = store
    _mk(s, home, "oldc", "2025-01-01T00:00:00+00:00", curated=True)
    summ = s.prune_traces("2026-01-01T00:00:00+00:00")
    assert summ["deleted"] == 0 and summ["kept_curated"] == 1
    summ2 = s.prune_traces("2026-01-01T00:00:00+00:00", keep_curated=False)
    assert summ2["deleted"] == 1
    assert s.get_trace("oldc") is None


def test_prune_missing_transcript_still_deletes_row(store):
    s, home = store
    row = _mk(s, home, "old1", "2025-01-01T00:00:00+00:00")
    Path(row.jsonl_path).unlink()
    summ = s.prune_traces("2026-01-01T00:00:00+00:00")
    assert summ["deleted"] == 1 and not summ["errors"]
    assert s.get_trace("old1") is None


# ------------------------------------------------------------------ CLI


def _run_cli(tmp_path, monkeypatch, argv, keep_config: bool = False):
    monkeypatch.setenv("SABLE_HOME", str(tmp_path))
    if not keep_config:
        monkeypatch.delenv("SABLE_CONFIG", raising=False)
    return main(argv)


def test_cli_prune_requires_window(tmp_path, monkeypatch, capsys):
    rc = _run_cli(tmp_path, monkeypatch, ["traces", "prune"])
    assert rc == 1
    assert "--older-than" in capsys.readouterr().err


def test_cli_prune_dry_then_yes(tmp_path, monkeypatch, capsys):
    db = tmp_path / "buffer" / "sable.db"
    s = TraceStore(db)
    _mk(s, tmp_path, "old1", "2025-01-01T00:00:00+00:00")
    s.close()
    rc = _run_cli(tmp_path, monkeypatch, ["traces", "prune", "--older-than", "30"])
    assert rc == 0
    assert "dry run" in capsys.readouterr().out
    assert (tmp_path / "buffer" / "old1.jsonl").exists()
    rc = _run_cli(tmp_path, monkeypatch,
                  ["traces", "prune", "--older-than", "30", "--yes"])
    assert rc == 0
    assert not (tmp_path / "buffer" / "old1.jsonl").exists()


def test_cli_prune_uses_config_retention_days(tmp_path, monkeypatch, capsys):
    cfg = tmp_path / "sable.yaml"
    cfg.write_text("recording:\n  retention_days: 45\n", encoding="utf-8")
    db = tmp_path / "buffer" / "sable.db"
    s = TraceStore(db)
    _mk(s, tmp_path, "old1", "2025-01-01T00:00:00+00:00")
    s.close()
    monkeypatch.setenv("SABLE_CONFIG", str(cfg))
    rc = _run_cli(tmp_path, monkeypatch, ["traces", "prune"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "older than 45d" in out

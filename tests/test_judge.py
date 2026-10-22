"""Backlog #12 tests — LLM-as-judge trace scoring (opt-in, biased tiebreaker).

Prompt bounding, forced-JSON parsing, feedback storage (source="judge"),
effective-score precedence user > judge > heuristic, borderline selection in
`curate --judge N`, and the zero-call guarantee when the flag is absent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_server import FakeOpenAIServer, make_message_handler  # noqa: E402

import sable.cli as cli  # noqa: E402
from sable.curate.judge import (  # noqa: E402
    JUDGE_RUBRIC,
    TOOL_VIEW_CAP,
    build_judge_messages,
    judge_trace,
    parse_judge_reply,
    select_borderline,
)
from sable.store.db import TraceRow, TraceStore  # noqa: E402


@pytest.fixture()
def sable_home(tmp_path, monkeypatch):
    home = tmp_path / "uf-judge"
    monkeypatch.setenv("SABLE_HOME", str(home))
    monkeypatch.delenv("SABLE_CONFIG", raising=False)
    return home


@pytest.fixture()
def store(sable_home):
    st = TraceStore(sable_home / "buffer" / "sable.db")
    yield st
    st.close()


def seed_trace(store, home, *, trace_id, score=0.5, n_msgs=3,
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

def test_judge_feedback_row_written_and_outcome_score_untouched(store):
    seed_trace(store, Path(store.db_path).parent.parent, trace_id="t1", score=0.4)
    store.add_feedback("t1", 0.9, comment="judge: good", source="judge",
                       timestamp="2026-09-01T10:01:00+00:00")
    fb = store.get_feedback("t1")
    assert [f["source"] for f in fb] == ["heuristic", "judge"]
    # judge never clobbers the record-time heuristic baseline
    assert store.get_trace("t1").outcome_score == 0.4


def test_precedence_user_over_judge_over_heuristic(store, sable_home):
    from sable.curate.curater import load_trace

    seed_trace(store, sable_home, trace_id="h-only", score=0.4)
    seed_trace(store, sable_home, trace_id="judge-wins", score=0.4)
    store.add_feedback("judge-wins", 0.8, comment="j", source="judge",
                       timestamp="2026-09-01T10:02:00+00:00")
    seed_trace(store, sable_home, trace_id="user-wins", score=0.4)
    store.add_feedback("user-wins", 0.8, comment="j", source="judge",
                       timestamp="2026-09-01T10:02:00+00:00")
    store.add_feedback("user-wins", -1.0, comment="u", source="user",
                       timestamp="2026-09-01T10:03:00+00:00")

    eff = {}
    for tid in ("h-only", "judge-wins", "user-wins"):
        row = store.get_trace(tid)
        assert row is not None
        loaded = load_trace(store, row)
        assert loaded is not None
        eff[tid] = loaded.effective_score
    assert eff["h-only"] == 0.4        # no judge rows -> byte-identical today
    assert eff["judge-wins"] == 0.8    # judge overrides heuristic
    assert eff["user-wins"] == -1.0    # user overrides judge


def test_curater_selection_uses_judge_score(sable_home, store):
    """A trace below min_score but above it by judge score enters SFT."""
    seed_trace(store, sable_home, trace_id="below", score=0.4)
    seed_trace(store, sable_home, trace_id="above-by-judge", score=0.4)
    store.add_feedback("above-by-judge", 0.9, comment="j", source="judge",
                       timestamp="2026-09-01T10:05:00+00:00")
    assert cli.main(["curate", "--name", "jsel"]) == 0
    ds = next((sable_home / "datasets").rglob("sft.jsonl"))
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


def test_curate_judge_flag_scores_borderline_only(sable_home, store,
                                                  monkeypatch, capsys):
    seed_trace(store, sable_home, trace_id="border0", score=0.5)
    seed_trace(store, sable_home, trace_id="border1", score=0.4)
    seed_trace(store, sable_home, trace_id="far", score=0.9)

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
        srcs = [fb["source"] for fb in store.get_feedback(f)]
        if "judge" in srcs:
            tids_judged.add(f)
    assert tids_judged == {"border0", "border1"}
    out = capsys.readouterr().out
    assert "biased tiebreaker" in out  # loud caveat printed once per run
    assert "judge summary: 2 scored, 0 skipped" in out


def test_curate_without_judge_flag_makes_zero_llm_calls(sable_home, store,
                                                        monkeypatch):
    seed_trace(store, sable_home, trace_id="t", score=0.5)
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
        assert [fb["source"] for fb in store.get_feedback(f)] == ["heuristic"]


def test_curate_judge_unparseable_reply_skips_not_crashes(sable_home, store,
                                                          monkeypatch, capsys):
    seed_trace(store, sable_home, trace_id="s", score=0.5)
    srv, handler = _fake_judge_server(reply="I think it went pretty well!")
    monkeypatch.setattr(cli, "_resolve_endpoint",
                        lambda cfg, spawn_if_needed: (srv.base_url, "m", None))
    try:
        assert cli.main(["curate", "--name", "skip", "--judge", "1"]) == 0
    finally:
        srv.stop()
    assert [fb["source"] for fb in store.get_feedback("s")] == ["heuristic"]
    out = capsys.readouterr().out
    assert "SKIPPED" in out and "judge summary: 0 scored, 1 skipped" in out

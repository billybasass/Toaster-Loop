"""Pass 4 #38 — trajectory → playbook distillation (core/playbooks.py).

Covers: chain mining (counts/window/failures/order), template distillation,
lifecycle incl. judge-gate stubs and correlational demotion, prompt injection
caps + byte-identical disabled path, export metadata, CLI smoke.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from sable.core.config import SableConfig
from sable.core.playbooks import (
    DEMOTE_NORM_THRESHOLD,
    GateUnavailable,
    PlaybookStore,
    assess_playbook,
    distill_recipe,
    export_playbook,
    find_recurring_chains,
    promote,
    prompt_section,
    run_promotion_gate,
)
from sable.store.db import TraceRow, TraceStore


# ------------------------------------------------------------------ fixtures

def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def _write_transcript(path: Path, tool_names, first_user="fix the bug",
                      fail_tools=()) -> None:
    """Minimal recorder-shaped JSONL: assistant tool_calls + tool results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    recs = [{
        "kind": "message",
        "message": {"role": "user", "content": first_user},
    }]
    for i, name in enumerate(tool_names):
        recs.append({
            "kind": "message",
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {"name": name,
                                 "arguments": json.dumps({"path": "x.py"})},
                }],
            },
        })
        content = ("error: boom" if name in fail_tools
                   else f"{name} ok")
        recs.append({
            "kind": "message",
            "message": {"role": "tool", "tool_call_id": f"call_{i}",
                        "name": name, "content": content},
        })
    recs.append({"kind": "message",
                 "message": {"role": "assistant", "content": "done"}})
    path.write_text("\n".join(json.dumps(r) for r in recs) + "\n")


@pytest.fixture()
def store(tmp_path):
    s = TraceStore(tmp_path / "buffer" / "sable.db")
    yield s
    s.close()


@pytest.fixture()
def pb_store(tmp_path):
    ps = PlaybookStore(tmp_path / "buffer" / "sable.db")
    yield ps
    ps.close()


def _must(pb_store, pb_id):
    """get() + assert-not-None (typing convenience for the assertions)."""
    pb = pb_store.get(pb_id)
    assert pb is not None
    return pb


def _add_trace(store, tmp_path, tid, tools, *, profile="coder",
               status="success", when=None, score=0.5, fail_tools=(),
               first_user="fix the bug"):
    when = when or datetime.now(timezone.utc)
    jp = tmp_path / "traces" / f"{tid}.jsonl"
    _write_transcript(jp, tools, first_user=first_user,
                      fail_tools=fail_tools)
    store.insert_trace(TraceRow(
        trace_id=tid, session_id=None, timestamp=_iso(when),
        agent_profile=profile, model_alias="m", model_version=None,
        endpoint_label=None, task_summary="t", status=status,
        outcome_score=score, num_tool_calls=len(tools), tokens_in=None,
        tokens_out=None, duration_s=None, jsonl_path=str(jp)))


# ------------------------------------------------------------- chain mining

def test_scan_finds_recurring_chain_with_counts(store, tmp_path):
    for i in range(3):
        _add_trace(store, tmp_path, f"t{i}",
                   ["read_file", "search_files", "patch"])
    _add_trace(store, tmp_path, "x1", ["list_dir"])  # too short: no chain

    chains = find_recurring_chains(store, "coder", min_occurrences=3)
    got = {tuple(c["chain"]): c["count"] for c in chains}
    assert ("read_file", "search_files", "patch") in got
    assert got[("read_file", "search_files", "patch")] == 3
    # contiguous subsequences of length 2 also recur
    assert got[("read_file", "search_files")] == 3
    assert all(c["count"] >= 3 for c in chains)


def test_scan_excludes_failures(store, tmp_path):
    _add_trace(store, tmp_path, "ok1", ["a", "b"], status="success")
    _add_trace(store, tmp_path, "ok2", ["a", "b"], status="partial")
    _add_trace(store, tmp_path, "bad1", ["a", "b"], status="failure")
    _add_trace(store, tmp_path, "ok3", ["a", "b"], status="success")

    # the failing trace never counts: exactly the 3 non-failures do
    chains = find_recurring_chains(store, "coder", min_occurrences=3)
    hit = [c for c in chains if c["chain"] == ["a", "b"]]
    assert hit and hit[0]["count"] == 3  # success + partial count
    assert set(hit[0]["trace_ids"]) == {"ok1", "ok2", "ok3"}
    # and with four required occurrences nothing qualifies
    assert [c for c in find_recurring_chains(store, "coder",
                                             min_occurrences=4)
            if c["chain"] == ["a", "b"]] == []


def test_scan_respects_window(store, tmp_path):
    old = datetime.now(timezone.utc) - timedelta(days=45)
    for i in range(2):
        _add_trace(store, tmp_path, f"old{i}", ["a", "b"], when=old)
    for i in range(3):
        _add_trace(store, tmp_path, f"new{i}", ["a", "b"])

    chains = find_recurring_chains(store, "coder", min_occurrences=3,
                                   window_days=30)
    hit = [c for c in chains if c["chain"] == ["a", "b"]]
    assert hit and hit[0]["count"] == 3  # the two old traces are out of window


def test_scan_order_deterministic(store, tmp_path):
    for i in range(4):
        _add_trace(store, tmp_path, f"a{i}", ["read_file", "edit_file"])
    for i in range(2):
        _add_trace(store, tmp_path, f"b{i}", ["aaa_tool", "zzz_tool"])

    def order():
        return list(find_recurring_chains(store, "coder",
                                          min_occurrences=2))

    first = order()
    # count desc: the 4x chain precedes any 2x chain; ties asc by join
    counts = [c["count"] for c in find_recurring_chains(
        store, "coder", min_occurrences=2)]
    assert counts == sorted(counts, reverse=True)
    # ties break ascending on the joined chain
    seen = {}
    for c in first:
        seen.setdefault(c["count"], []).append("\x00".join(c["chain"]))
    for chains_at_n in seen.values():
        assert chains_at_n == sorted(chains_at_n)
    # repeated calls identical
    assert order() == first


def test_occurrence_deduped_per_trace(store, tmp_path):
    # chain appears twice inside ONE trace => still one occurrence
    _add_trace(store, tmp_path, "d1", ["a", "b", "c", "a", "b"])
    chains = find_recurring_chains(store, "coder", min_occurrences=2)
    assert [c for c in chains if c["chain"] == ["a", "b"]] == []


# ------------------------------------------------------------- distillation

def test_distill_template_contains_facts_no_placeholders(store, tmp_path):
    from sable.curate.curater import load_trace

    for i in range(3):
        _add_trace(store, tmp_path, f"t{i}", ["read_file", "patch"],
                   first_user=f"refactor module {i}", fail_tools=("patch",))
    rows = [store.get_trace(f"t{i}") for i in range(3)]
    examples = [load_trace(store, r) for r in rows]

    text = distill_recipe(["read_file", "patch"], examples, occurrences=3,
                          profile="coder")
    assert "## Workflow: read_file → patch" in text
    assert "3 recorded run(s)" in text
    assert "Typical task:" in text
    assert "1. read_file" in text and "2. patch" in text
    assert "args seen: path" in text
    assert "Gotchas:" in text and "patch returned an error-shaped result" in text
    # no leftover template artifacts
    for junk in ("{", "}", "<%", "TODO", "PLACEHOLDER"):
        assert junk not in text


def test_distill_redacts_secrets(tmp_path):
    class FakeT:
        row = None

        def __init__(self, msgs):
            self.messages = msgs

    t = FakeT([{"role": "user",
                "content": "use key sk-abcdefghijklmnopqrstuvwxyz012345"}])
    text = distill_recipe(["a", "b"], [t], occurrences=1)
    assert "sk-abcdefghijklmnopqrstuvwxyz012345" not in text
    assert "[REDACTED" in text


# ---------------------------------------------------------------- lifecycle

def test_repromotion_stamps_fresh_era(pb_store):
    """Retire → re-promote starts a NEW era: promoted_ts is re-stamped so
    assess_playbook compares the right before/after boundary (the first
    promotion's timestamp must not survive across a retirement)."""
    pb_id, _ = pb_store.create_candidate(["a", "b"], "r")
    assert promote(pb_store, pb_id)["promoted"]
    era1 = _must(pb_store, pb_id).promoted_ts
    assert era1
    pb_store.set_status(pb_id, "retired")
    assert promote(pb_store, pb_id)["promoted"]
    pb = _must(pb_store, pb_id)
    assert pb.status == "active" and pb.promoted_ts >= era1
    # monotonic seconds: re-promotion within the same second can tie; never
    # may it be OLDER than the first era.
    assert pb.promoted_ts is not None


def test_assess_cli_reports_and_retires(tmp_path, monkeypatch, capsys):
    from sable.cli import main

    home = tmp_path / "home"
    monkeypatch.setenv("SABLE_HOME", str(home))
    monkeypatch.delenv("SABLE_CONFIG", raising=False)
    st = TraceStore(home / "buffer" / "sable.db")
    ps = PlaybookStore(home / "buffer" / "sable.db")
    try:
        pb_id, _ = ps.create_candidate(["a", "b"], "r", profile="coder")
        promote(ps, pb_id)
        # era boundary then a clear regression over enough samples
        old = datetime.now(timezone.utc) - timedelta(days=5)
        for i in range(3):
            _add_trace(st, tmp_path, f"pre{i}", ["a", "b"], score=0.8,
                       when=old)
        for i in range(3):
            _add_trace(st, tmp_path, f"post{i}", ["a", "b"], score=-0.9)
        rc = main(["playbooks", "assess"])
        assert rc == 0
        out = capsys.readouterr().out
        assert pb_id in out and "RETIRED" in out
        assert _must(ps, pb_id).status == "retired"
    finally:
        ps.close()
        st.close()


def test_candidate_idempotent_and_status_transitions(pb_store):
    pb_id, created = pb_store.create_candidate(["a", "b"], "recipe",
                                               profile="coder")
    assert created
    pb_id2, created2 = pb_store.create_candidate(["a", "b"], "other",
                                                 profile="coder")
    assert pb_id == pb_id2 and not created2  # deterministic id, no clobber

    assert pb_store.set_status(pb_id, "active")
    pb = pb_store.get(pb_id)
    assert pb.status == "active" and pb.promoted_ts
    assert pb_store.set_status(pb_id, "retired")
    pb = pb_store.get(pb_id)
    assert pb.status == "retired"
    assert pb.promoted_ts  # era boundary kept
    assert pb_store.get(pb_id) is not None  # retired rows KEPT


def test_promote_gate_pass_and_fail(pb_store, store, tmp_path):
    # replay set: positive-score success traces for the profile
    for i in range(3):
        _add_trace(store, tmp_path, f"r{i}", ["a", "b"], score=0.8)
    pb_id, _ = pb_store.create_candidate(["a", "b"], "recipe text",
                                         profile="coder")
    pb = pb_store.get(pb_id)

    def gate_stub(mean):
        return {"ts": "t", "n": 3, "scored": 3, "skipped": 0,
                "judge_mean": mean, "threshold": 0.7,
                "passed": mean is not None and mean >= 0.7, "reason": None}

    # fail: candidate untouched, history recorded
    res = promote(pb_store, pb_id, check=True, gate=gate_stub(0.4))
    assert not res["ok"] and not res.get("promoted")
    assert pb_store.get(pb_id).status == "candidate"
    hist = pb_store.get(pb_id).score_history
    assert hist and hist[-1]["judge_mean"] == 0.4

    # pass: promoted, promoted_ts stamped
    res = promote(pb_store, pb_id, check=True, gate=gate_stub(0.9))
    assert res["ok"] and res["promoted"]
    assert pb_store.get(pb_id).status == "active"


def test_promote_without_check_is_explicit_confirmation(pb_store):
    pb_id, _ = pb_store.create_candidate(["a", "b"], "r")
    res = promote(pb_store, pb_id)  # no gate: the call itself confirms
    assert res["ok"] and res["promoted"]
    assert pb_store.get(pb_id).status == "active"


def test_promotion_gate_requires_replay_set(store, tmp_path):
    pb = PlaybookStore(tmp_path / "pb.db")
    try:
        pb_id, _ = pb.create_candidate(["a"], "r", profile="nobody")
        with pytest.raises(GateUnavailable):
            run_promotion_gate(store, _must(pb, pb_id),
                               base_url="http://127.0.0.1:1", model="m")
    finally:
        pb.close()


def test_promotion_gate_bounded_calls_and_zero_scored_fails(
        store, tmp_path):
    for i in range(2):
        _add_trace(store, tmp_path, f"g{i}", ["a", "b"], score=0.9)
    pb = PlaybookStore(tmp_path / "pb.db")
    try:
        pb_id, _ = pb.create_candidate(["a", "b"], "r", profile="coder")
        pbo = _must(pb, pb_id)
        calls = {"gen": 0}

        def gen(task):
            calls["gen"] += 1
            return None  # every generation fails => zero scored

        verdict = run_promotion_gate(store, pbo, base_url="http://x/v1",
                                     model="m", n=6, generate_fn=gen,
                                     judge_client_for=lambda u, m:
                                     (lambda msgs, mdl: None))
        assert not verdict["passed"] and verdict["scored"] == 0
        assert calls["gen"] == 2  # bounded by replay size, no extras

        def gen_ok(task):
            return "answer"
        verdict = run_promotion_gate(
            store, pbo, base_url="http://x/v1", model="m", n=6,
            generate_fn=gen_ok,
            judge_client_for=lambda u, m: (lambda msgs, mdl:
                                           '{"score": 0.95, "reason": "ok"}'))
        assert verdict["passed"] and verdict["judge_mean"] == 0.95
    finally:
        pb.close()


def test_assess_demotes_on_regression_kept(store, tmp_path):
    pb = PlaybookStore(tmp_path / "buffer" / "sable.db")
    try:
        # pre-era traces score well; post-era matching traces collapse
        old = datetime.now(timezone.utc) - timedelta(days=5)
        for i in range(3):
            _add_trace(store, tmp_path, f"pre{i}", ["a", "b"], when=old,
                       score=0.5)
        pb_id, _ = pb.create_candidate(["a", "b"], "r", profile="coder")
        pb.set_status(pb_id, "active")
        boundary = _must(pb, pb_id).promoted_ts
        for i in range(3):
            _add_trace(store, tmp_path, f"post{i}", ["a", "b"], score=-0.9)
        # ensure post rows are strictly after the promoted boundary
        row = store.get_trace("post0")
        assert row.timestamp >= boundary

        res = assess_playbook(store, pb, _must(pb, pb_id))
        assert res["ok"] and res["demoted"]
        assert res["mean_without"] > 0 > res["mean_with"]
        assert (res["mean_with"] + 1) / 2 <= DEMOTE_NORM_THRESHOLD
        got = _must(pb, pb_id)
        assert got.status == "retired"          # demoted...
        assert got.score_history[-1]["n"] >= 6  # ...with the reading stored
    finally:
        pb.close()


def test_assess_no_regression_stays_active(store, tmp_path):
    pb = PlaybookStore(tmp_path / "buffer" / "sable.db")
    try:
        old = datetime.now(timezone.utc) - timedelta(days=5)
        for i in range(3):
            _add_trace(store, tmp_path, f"pre{i}", ["a", "b"], when=old,
                       score=-0.2)
        pb_id, _ = pb.create_candidate(["a", "b"], "r", profile="coder")
        pb.set_status(pb_id, "active")
        for i in range(3):
            _add_trace(store, tmp_path, f"post{i}", ["a", "b"], score=0.5)
        res = assess_playbook(store, pb, _must(pb, pb_id))
        assert res["ok"] and not res["demoted"]
        assert _must(pb, pb_id).status == "active"
    finally:
        pb.close()


# ---------------------------------------------------------------- injection

def _cfg_with(tmp_path, **playbooks):
    return SableConfig(home=str(tmp_path / "home"), playbooks=playbooks)


def _seed_active(pb_store, chain, text, profile=None, promoted_ts="b"):
    pb_id, _ = pb_store.create_candidate(chain, text, profile=profile)
    pb_store.set_status(pb_id, "active")
    with pb_store._lock:
        pb_store._conn.execute(
            "UPDATE playbooks SET promoted_ts = ? WHERE id = ?",
            (promoted_ts, pb_id))
        pb_store._conn.commit()
    return pb_id


def test_injection_enabled_contains_section(tmp_path, monkeypatch):
    monkeypatch.delenv("SABLE_HOME", raising=False)  # cfg.home resolution
    home = tmp_path / "home"
    (home / "buffer").mkdir(parents=True)
    pb = PlaybookStore(home / "buffer" / "sable.db")
    try:
        _seed_active(pb, ["read_file", "patch"], "## Workflow: r → p\nbody")
    finally:
        pb.close()
    cfg = _cfg_with(tmp_path, enabled=True)

    section = prompt_section(cfg, None)
    assert section.startswith("## Playbooks\n")
    assert "## Workflow: r → p" in section


def test_injection_caps_omit_whole_recipes(tmp_path, monkeypatch):
    monkeypatch.delenv("SABLE_HOME", raising=False)  # cfg.home resolution
    home = tmp_path / "home"
    (home / "buffer").mkdir(parents=True)
    pb = PlaybookStore(home / "buffer" / "sable.db")
    try:
        big = "x" * 900  # > per_recipe_cap 800 -> omitted whole
        ok1 = "A" * 300
        ok2 = "B" * 300
        _seed_active(pb, ["a"], big, promoted_ts="2026-01-01T00:00:00+00:00")
        _seed_active(pb, ["b"], ok1, promoted_ts="2026-01-02T00:00:00+00:00")
        _seed_active(pb, ["c"], ok2, promoted_ts="2026-01-03T00:00:00+00:00")
        cfg = SableConfig(home=str(home), playbooks={
            "enabled": True, "total_cap_chars": 400,
            "per_recipe_cap_chars": 800})
        section = prompt_section(cfg, None)
        assert "BBB" in section          # most-recently-promoted first
        assert "AAA" not in section      # does not fit remaining budget
        assert "xxx" not in section      # over per-recipe cap
        assert len(section) <= 400 + len("## Playbooks\n")
    finally:
        pb.close()


def test_injection_disabled_byte_identical_prompt(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / "buffer").mkdir(parents=True)
    pb = PlaybookStore(home / "buffer" / "sable.db")
    try:
        _seed_active(pb, ["a", "b"], "SHOULD NOT APPEAR")
    finally:
        pb.close()
    monkeypatch.delenv("SABLE_HOME", raising=False)
    cfg = SableConfig(home=str(home), playbooks={})  # disabled default

    from sable.core.agent_runner import create_runner

    captured = {}

    class Spy(Exception):
        pass

    def spy_client(*args, **kwargs):
        captured["system_prompt"] = kwargs.get("system_prompt")
        raise Spy()

    monkeypatch.setattr("sable.core.agent_runner.SableAgentClient",
                        spy_client)
    with pytest.raises(Spy):
        create_runner(cfg, "http://127.0.0.1:1", "m",
                      system_prompt="BASE PROMPT", profile_name="coder")
    assert captured["system_prompt"] == "BASE PROMPT"  # untouched

    # and enabled flips it (same fixture DB has an active playbook)
    cfg.playbooks = {"enabled": True}
    with pytest.raises(Spy):
        create_runner(cfg, "http://127.0.0.1:1", "m",
                      system_prompt="BASE PROMPT", profile_name="coder")
    assert "## Playbooks" in captured["system_prompt"]
    assert captured["system_prompt"].startswith("BASE PROMPT")


# ------------------------------------------------------------------- export

def test_export_writes_metadata_not_rows(tmp_path, pb_store):
    pb_id, _ = pb_store.create_candidate(["a", "b"], "recipe", profile="coder",
                                         source_trace_ids=["t1", "t2"])
    pb_store.set_status(pb_id, "active")
    out = export_playbook(pb_store.get(pb_id), home=tmp_path)
    payload = json.loads(Path(out).read_text())
    assert payload["not_training_rows"] is True
    assert payload["chain"] == ["a", "b"]
    assert payload["recipe"] == "recipe"
    assert payload["source_trace_ids"] == ["t1", "t2"]
    assert payload["kind"] == "sable.playbook_export"
    assert "conversations" not in payload  # no fabricated rows
    assert out.parent.name == "_playbooks"


# --------------------------------------------------------------- db schema

def test_playbooks_table_via_tracestore(tmp_path):
    """The playbooks table is part of the buffer schema (SCHEMA_VERSION 6)."""
    s = TraceStore(tmp_path / "x.db")
    try:
        cols = {r[1] for r in s._conn.execute("PRAGMA table_info(playbooks)")}
        assert {"id", "profile", "chain_json", "text", "status",
                "created_ts", "promoted_ts",
                "score_history_json"} <= cols
    finally:
        s.close()


# -------------------------------------------------------------- CLI smoke

def test_cli_playbooks_end_to_end(tmp_path, monkeypatch, capsys):
    from sable.cli import main

    home = tmp_path / "clihome"
    monkeypatch.setenv("SABLE_HOME", str(home))
    st = TraceStore(home / "buffer" / "sable.db")
    try:
        for i in range(3):
            _add_trace(st, tmp_path / "cli", f"c{i}", ["read_file", "patch"],
                       profile="coder")
    finally:
        st.close()

    assert main(["playbooks", "scan", "--profile", "coder",
                 "--create"]) == 0
    out = capsys.readouterr().out
    assert "read_file → patch" in out and "created candidate pb-" in out

    assert main(["playbooks", "candidates"]) == 0
    out = capsys.readouterr().out
    pb_id = [tok for tok in out.split() if tok.startswith("pb-")][0]

    assert main(["playbooks", "show", pb_id]) == 0
    assert "candidate" in capsys.readouterr().out

    # explicit confirmation promote (no --check, no endpoint needed)
    assert main(["playbooks", "promote", pb_id]) == 0
    assert "promoted" in capsys.readouterr().out

    assert main(["playbooks", "list"]) == 0
    assert "active" in capsys.readouterr().out

    assert main(["playbooks", "export", pb_id]) == 0
    out = capsys.readouterr().out
    assert "NOT training rows" in out
    exported = home / "datasets" / "_playbooks" / f"{pb_id}.json"
    assert exported.is_file()

    assert main(["playbooks", "retire", pb_id]) == 0
    assert "retired" in capsys.readouterr().out

    # retired row kept (show still works)
    assert main(["playbooks", "show", pb_id]) == 0
    assert "retired" in capsys.readouterr().out


def test_cli_promote_check_without_endpoint_refuses(tmp_path, monkeypatch,
                                                    capsys):
    from sable.cli import main

    home = tmp_path / "clihome2"
    monkeypatch.setenv("SABLE_HOME", str(home))
    monkeypatch.delenv("SABLE_ATTACH_URL", raising=False)
    ps = PlaybookStore(home / "buffer" / "sable.db")
    try:
        pb_id, _ = ps.create_candidate(["a", "b"], "r", profile="coder")
    finally:
        ps.close()
    assert main(["playbooks", "promote", pb_id, "--check"]) == 1
    assert "served endpoint" in capsys.readouterr().err

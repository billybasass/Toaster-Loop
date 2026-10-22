"""Golden-mining tests (curate/mine_goldens.py).

All hermetic: fake OpenAI server only where an endpoint is involved (RFT
replay / suite replay); mining itself is pure store+filesystem. Covers the
feature contract: 👍 trace -> self-checked golden in the real suite store;
mined goldens appear in the merged run_golden_suite task list and are picked
up by RFT's golden prompt source; non-self-passing / verifier-less candidates
dropped with a reason; deterministic-verifier requirement (judge may only
propose); trace + prompt-hash dedup; redaction proof; ledger claim atomicity;
cap enforcement; the mine_goldens job kind end to end through the dispatcher;
master switch off by default everywhere.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).parent))

from fake_server import FakeOpenAIServer, make_message_handler  # noqa: E402

from sable.core.config import GoldensMineConfig, load_config  # noqa: E402
from sable.curate.mine_goldens import (  # noqa: E402
    MineGoldensError,
    load_mined_goldens,
    merge_mined_into_tasks,
    mined_path,
    run_mine_goldens,
    task_pair,
)
from sable.registry import ModelRegistry  # noqa: E402
from sable.store.db import TraceRow, TraceStore  # noqa: E402
from sable.training.rejection import (  # noqa: E402
    make_rft_job_runner,
    select_prompts,
)
from sable.curate.mine_goldens import make_mine_goldens_job_runner  # noqa: E402
from sable.training.scheduler import (  # noqa: E402
    JobDispatcher,
    Scheduler,
    TrainingConfig,
)
from sable.training.validate import run_golden_suite  # noqa: E402


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "umine"
    monkeypatch.setenv("SABLE_HOME", str(h))
    return h


@pytest.fixture()
def store(home):
    st = TraceStore(home / "buffer" / "sable.db")
    yield st
    st.close()


def seed_trace(store, home, *, trace_id, messages, status="success",
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

def test_thumbed_trace_mined_into_suite_and_replayed(home, store):
    """👍 trace -> golden persisted with provenance; merged task list feeds
    run_golden_suite and the golden passes against a server replaying the
    source answer."""
    seed_trace(store, home, trace_id="t1", messages=BRANCH_TRACE)
    res = run_mine_goldens(store, home, cfg())
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
    row = store._conn.execute(
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


def test_rft_select_prompts_picks_up_mined_golden(home, store):
    """Mined goldens join RFT's golden prompt source (job-runner merge path)."""
    seed_trace(store, home, trace_id="t1", messages=BRANCH_TRACE)
    run_mine_goldens(store, home, cfg())

    from sable.core.config import RFTConfig as _RC
    rcfg = _RC(enabled=True, sample_k=1, max_prompts=10)
    prompts, _ = select_prompts(store, rcfg,
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

def test_verifier_less_candidate_dropped_with_reason(home, store):
    """Reply too short to yield any deterministic check -> dropped (a 👍 with
    nothing machine-checkable must not mint a vacuous golden)."""
    seed_trace(store, home, trace_id="t2", messages=[
        {"role": "user", "content": "What is the mood today?"},
        {"role": "assistant", "content": "OK"},
    ])
    res = run_mine_goldens(store, home, cfg())
    assert res.mined == 0 and len(res.dropped) == 1
    assert "deterministic" in res.dropped[0]["reason"]
    row = store._conn.execute(
        "SELECT status, reason FROM mined_goldens WHERE trace_id='t2'"
    ).fetchone()
    assert row["status"] == "dropped" and "deterministic" in row["reason"]
    assert not mined_tasks(home)


def test_judge_may_only_propose_self_passing_substrings(home, store):
    """Judge proposals join ONLY if the source output passes them; a fabricated
    phrase is discarded, and no golden is ever judge-only."""
    seed_trace(store, home, trace_id="t3", messages=BRANCH_TRACE)

    def judge(msgs, model):
        return json.dumps({"expect_contains": [
            "git branch --show-current",   # present -> adopted
            "definitely not in the reply",  # absent -> rejected by self-check
        ]})

    res = run_mine_goldens(store, home, cfg(use_judge=True),
                           judge_client_fn=judge, judge_model="j")
    assert res.mined == 1
    g = mined_tasks(home)[0]
    joined = "".join(g["expect_contains"])
    assert "git branch --show-current" in joined or len(g["expect_contains"]) >= 1
    assert all("definitely not" not in s for s in g["expect_contains"])


def test_tool_call_verifier_derived(home, store):
    seed_trace(store, home, trace_id="t4", messages=[
        {"role": "user", "content": "List the files in src/"},
        {"role": "assistant", "content": "Listing now for you.",
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "list_files",
                                      "arguments": "{\"path\": \"src/\"}"}}]},
    ])
    res = run_mine_goldens(store, home, cfg())
    assert res.mined == 1
    g = mined_tasks(home)[0]
    assert g["expect_tool_call"] == {"name": "list_files"}
    assert "tool_call" in "".join(g["_mine"]["verifiers"])


# ------------------------------------------------------------------ dedup

def test_same_trace_mined_once(home, store):
    seed_trace(store, home, trace_id="t5", messages=BRANCH_TRACE)
    r1 = run_mine_goldens(store, home, cfg())
    r2 = run_mine_goldens(store, home, cfg())
    assert r1.mined == 1
    assert r2.candidates == 0 and r2.mined == 0
    assert len(mined_tasks(home)) == 1


def test_prompt_hash_collision_with_manual_golden_skipped(home, store):
    seed_trace(store, home, trace_id="t6", messages=BRANCH_TRACE)
    manual = [{"prompt": BRANCH_TRACE[0]["content"], "expect_contains": ["x"]}]
    res = run_mine_goldens(store, home, cfg(), existing_goldens=manual)
    assert res.skipped_dedup == 1 and res.mined == 0
    assert not mined_tasks(home)
    # merge dedups too: manual golden wins, no duplicate in the suite list
    p = mined_path(home)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"prompt": BRANCH_TRACE[0]["content"],
                             "expect_contains": ["y"]}) + "\n")
    merged = merge_mined_into_tasks(manual, home)
    assert len(merged) == 1 and merged[0]["expect_contains"] == ["x"]


def test_untthumbd_trace_never_a_candidate(home, store):
    seed_trace(store, home, trace_id="t7", messages=BRANCH_TRACE, thumb=False)
    res = run_mine_goldens(store, home, cfg())
    assert res.candidates == 0 and res.mined == 0


# ---------------------------------------------------------------- redaction

def test_secret_in_source_output_never_persisted(home, store):
    """A raw token in the (unredacted-at-rest) assistant output is masked and
    never persisted. When masking is ALL the reply has to offer, the candidate
    is dropped entirely: a substring carrying [REDACTED] passes the self-check
    against the redacted message but could never match real replay output — an
    always-failing golden would poison the promotion gate."""
    secret = "ghp_abc1234567890abcdefghij"  # matches redact.py GitHub PAT family
    seed_trace(store, home, trace_id="t8", messages=[
        {"role": "user",
         "content": f"Store my token {secret} somewhere safe"},
        {"role": "assistant",
         "content": f"Saved. The token {secret} is now configured globally."},
    ])
    res = run_mine_goldens(store, home, cfg())
    assert res.mined == 0  # every derived line carries a redaction marker
    assert res.dropped and "no deterministic verifier" in res.dropped[0]["reason"]
    raw_file = mined_path(home).read_text() if mined_path(home).is_file() else ""
    assert secret not in raw_file
    assert "[REDACTED" not in raw_file


def test_mixed_reply_keeps_clean_verifiers_drops_masked_lines(home, store):
    """Reply with one clean line + one secret-bearing line: the golden mines
    with ONLY the clean substring — no [REDACTED] verifier can slip through."""
    secret = "ghp_abc1234567890abcdefghij"
    seed_trace(store, home, trace_id="t9", messages=[
        {"role": "user", "content": "Configure my token please"},
        {"role": "assistant", "content":
            "Set the deploy key.\nThe token " + secret + " is now configured."},
    ])
    res = run_mine_goldens(store, home, cfg())
    assert res.mined == 1
    task = res.goldens[0]
    subs = task.get("expect_contains") or []
    assert any("deploy key" in s for s in subs)
    assert not any("[REDACTED" in s for s in subs)
    assert secret not in json.dumps(task)


# ------------------------------------------------- ledger atomicity + cap

def test_ledger_claim_is_atomic(home, store):
    assert store.claim_mined_goldens(["a", "b"]) == ["a", "b"]
    assert store.claim_mined_goldens(["b", "c"]) == ["c"]  # 'b' already owned


def test_stale_claims_are_reclaimed_finalized_never(home, store):
    """Crash recovery: an unfinalized 'claimed' row older than the stale
    window is released and claimable again; mined/dropped rows never are."""
    assert store.claim_mined_goldens(["x", "y", "z"]) == ["x", "y", "z"]
    store.record_mined_golden("y", prompt_key="ky", status="mined",
                              enabled=True, task_json="{}")
    store.record_mined_golden("z", prompt_key="", status="dropped",
                              reason="no verifier")
    # backdate all claims so 'x' (still claimed) looks orphaned
    store._conn.execute(
        "UPDATE mined_goldens SET claimed_at = '2000-01-01T00:00:00+00:00'")
    store._conn.commit()
    assert store.claim_mined_goldens(["x", "y", "z"]) == ["x"]


def test_fresh_claims_are_not_reclaimed(home, store):
    assert store.claim_mined_goldens(["n1"]) == ["n1"]
    # immediate re-claim: still in flight, must NOT be released
    assert store.claim_mined_goldens(["n1"]) == []


# ---------------------------------------------------- list + toggle (UI API)

def test_list_and_toggle_mined_goldens(home, store):
    from sable.curate.mine_goldens import (
        list_mined_goldens, set_mined_enabled)

    seed_trace(store, home, trace_id="tg1", messages=BRANCH_TRACE)
    res = run_mine_goldens(store, home, cfg())
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


def test_max_per_run_cap(home, store):
    for i in range(3):
        seed_trace(store, home, trace_id=f"c{i}", messages=[
            {"role": "user", "content": f"question number {i} about widgets"},
            {"role": "assistant",
             "content": f"answer {i}: widgets need the exact phrase number {i}."},
        ])
    res = run_mine_goldens(store, home, cfg(max_per_run=2))
    assert res.candidates == 2 and res.mined == 2
    assert len(mined_tasks(home)) == 2


# ----------------------------------------------------------- disabled path

def test_disabled_master_switch_refuses(home, store):
    with pytest.raises(MineGoldensError, match="goldens_mine.enabled is false"):
        run_mine_goldens(store, home, GoldensMineConfig())


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
    seed_trace(st, home, trace_id="d1", messages=BRANCH_TRACE)
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

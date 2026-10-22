"""RFT (rejection-sampling fine-tuning / STaR) tests.

All hermetic: fake OpenAI server for K-sampling transport, direct judge-client
lambdas for verdict shaping, stub token counters. Covers the feature contract:
K-sampling keeps only verified rows; honest-empty leaves no dataset dir; ledger
dedup stops re-runs resampling; the curate/rft trace-claim overlap invariant
holds both directions; the `rft` job kind flows through the dispatcher end to
end; strict judge thresholds; sampling-payload routing through
request_sampling_payload; config parsing without unknown-key warnings.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).parent))

from fake_server import FakeOpenAIServer, make_message_handler  # noqa: E402

from sable.core.config import RFTConfig, load_config  # noqa: E402
from sable.registry import ModelRegistry  # noqa: E402
from sable.store.db import TraceRow, TraceStore  # noqa: E402
from sable.training.rejection import (  # noqa: E402
    RftError,
    context_prefix,
    make_rft_job_runner,
    make_sampler,
    prompt_key,
    run_rft,
    verify_judged,
)
from sable.training.scheduler import JobDispatcher, Scheduler, TrainingConfig  # noqa: E402


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "ufrft"
    monkeypatch.setenv("SABLE_HOME", str(h))
    return h


@pytest.fixture()
def store(home):
    st = TraceStore(home / "buffer" / "sable.db")
    yield st
    st.close()


def seed_trace(store, home, *, trace_id, messages, status="success",
               timestamp="2026-09-01T10:00:00+00:00"):
    """JSONL transcript + index row exactly like the recorder (test_curate shape)."""
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


def cfg(**kw) -> RFTConfig:
    base: dict = dict(enabled=True, sample_k=4, max_prompts=10)
    base.update(kw)
    return RFTConfig(**base)


def ds_dir(res) -> Path:
    """Narrow run_rft's Optional dataset_dir for tests that assert a dataset
    was actually produced."""
    assert res.dataset_dir is not None
    return res.dataset_dir


GOLDEN_FRANCE = {
    "prompt": "capital of France",
    "expect_contains": ["Paris"],
}

JUDGE_OK = lambda msgs, model: json.dumps({"score": 0.9, "reason": "good"})


def cycle(responses):
    """Callable reply: pops scripted messages per request (last one repeats)."""
    state = {"i": 0}

    def reply(payload):
        r = responses[min(state["i"], len(responses) - 1)]
        state["i"] += 1
        return r
    return reply


def sft_rows(ds_dir: Path):
    return [json.loads(l) for l in
            (ds_dir / "sft.jsonl").read_text().splitlines() if l.strip()]


# ------------------------------------------------------------------ K-sampling

def test_ksampling_keeps_only_verified_row(home, store):
    """K=4 candidates; the first golden-verified one ships, later ones don't."""
    handler = make_message_handler({
        "capital": cycle([
            {"role": "assistant", "content": "nope"},
            {"role": "assistant", "content": "It is Paris, France."},
            {"role": "assistant", "content": "extra wrong"},
            {"role": "assistant", "content": "extra wrong 2"},
        ]),
    })
    with FakeOpenAIServer(handler_cls=handler) as srv:
        res = run_rft(store, home, cfg(sample_k=4),
                      sampler=make_sampler(srv.base_url, "fake-model",
                                           temperature=0.9, max_tokens=64),
                      goldens=[dict(GOLDEN_FRANCE)], name="rftk")
    assert res.dataset_id and res.rows_kept == 1
    d = ds_dir(res)
    rows = sft_rows(d)
    assert len(rows) == 1
    assert "Paris" in rows[0]["conversations"][-1]["value"]
    assert rows[0]["source"] == "rft"
    mf = json.loads((d / "manifest.json").read_text())
    assert mf["source"] == "rft"
    assert mf["trust"] == "curated"          # deterministic golden checks
    assert mf["files"]["sft.jsonl"]["count"] == 1
    assert mf["rft"]["prompts_sampled"] == 1
    assert (d / "train.yaml").is_file()


def test_empty_case_leaves_no_dataset_dir(home, store):
    """No verified candidate -> no-op; NO dataset dir survives (curater parity)."""
    handler = make_message_handler({
        "*": {"role": "assistant", "content": "always wrong"},
    })
    with FakeOpenAIServer(handler_cls=handler) as srv:
        res = run_rft(store, home, cfg(sample_k=3),
                      sampler=make_sampler(srv.base_url, "fake-model",
                                           temperature=0.9, max_tokens=64),
                      goldens=[dict(GOLDEN_FRANCE)], name="rftempty")
    assert res.dataset_id is None and res.rows_kept == 0
    assert any("no candidate passed" in w for w in res.warnings)
    assert not list(home.rglob("manifest.json"))


# ----------------------------------------------------------- dedup/provenance

def test_second_run_skips_sampled_prompts(home, store):
    handler = make_message_handler({
        "*": {"role": "assistant", "content": "Paris is the capital."},
    })
    sampler = lambda msgs: {"role": "assistant",
                            "content": "Paris is the capital."}
    with FakeOpenAIServer(handler_cls=handler):
        res1 = run_rft(store, home, cfg(sample_k=1), sampler=sampler,
                       judge_client_fn=JUDGE_OK,
                       seeds=["what's up?"], name="rftd")
        assert res1.rows_kept == 1
        res2 = run_rft(store, home, cfg(sample_k=1), sampler=sampler,
                       judge_client_fn=JUDGE_OK,
                       seeds=["what's up?"], name="rftd2")
    assert res2.dataset_id is None
    assert res2.prompts_sampled == 0
    assert any("already sampled" in w for w in res2.warnings)


def test_ledger_claim_is_atomic_and_stamped(home, store):
    key = prompt_key("hello there")
    assert store.claim_rft_prompts([(key, "seed", None)]) == [key]
    assert store.claim_rft_prompts([(key, "seed", None)]) == []  # race loser
    assert store.rft_prompt_seen(key)
    store.mark_rft_prompts_emitted([key], "ds-1", kept_keys=[key])
    row = store._conn.execute(
        "SELECT dataset_id, kept FROM rft_prompts WHERE prompt_key = ?",
        (key,)).fetchone()
    assert (row["dataset_id"], row["kept"]) == ("ds-1", 1)
    assert store.list_rft_prompt_keys() == [key]


# --------------------------------------------------------- overlap invariant #8

def test_curate_claimed_trace_never_becomes_rft_prompt(home, store):
    seed_trace(store, home, trace_id="t1", messages=[
        {"role": "user", "content": "capital of France"},
        {"role": "assistant", "content": "Paris"},
    ])
    assert store.claim_traces_owned(["t1"], "curate-ds") == ["t1"]
    sampler = lambda msgs: {"role": "assistant", "content": "Paris"}
    res = run_rft(store, home, cfg(), sampler=sampler)
    # The trace is excluded at selection (query_for_curation include_curated=0):
    # no prompt, no sampling, no ledger claim.
    assert res.prompts_sampled == 0 and res.dataset_id is None
    assert store.list_rft_prompt_keys() == []


def test_rft_claimed_trace_unavailable_to_other_claims(home, store):
    seed_trace(store, home, trace_id="t2", messages=[
        {"role": "user", "content": "capital of France"},
        {"role": "assistant", "content": "old answer"},
    ])
    sampler = lambda msgs: {"role": "assistant", "content": "Paris is it."}
    res = run_rft(store, home, cfg(), sampler=sampler, judge_client_fn=JUDGE_OK)
    assert res.rows_kept == 1 and res.claimed == 1
    row = store.get_trace("t2")
    assert row.curated == 1 and row.dataset_id == res.dataset_id
    # Same primitive the curater uses: a second claim (curate or rft) loses.
    assert store.claim_traces_owned(["t2"], "another-ds") == []


# ------------------------------------------------------- judge threshold gate

def test_verify_judged_strict_boundary():
    prompt = [{"role": "user", "content": "q"}]
    msg = {"role": "assistant", "content": "a"}
    ok_at = verify_judged(prompt, msg, lambda m, mo: json.dumps(
        {"score": 0.85, "reason": "x"}), "j", 0.85)
    assert ok_at[0] is True
    ok_below = verify_judged(prompt, msg, lambda m, mo: json.dumps(
        {"score": 0.849, "reason": "x"}), "j", 0.85)
    assert ok_below[0] is False
    # Unparseable / missing verdicts are FAILS, never passes.
    assert verify_judged(prompt, msg, lambda m, mo: "not json", "j", 0.85)[0] \
        is False
    assert verify_judged(prompt, None, JUDGE_OK, "j", 0.85)[0] is False


# ------------------------------------------------------ sampling payload shape

def test_sampler_routes_thinking_keys_via_request_sampling_payload():
    handler = make_message_handler({
        "*": {"role": "assistant", "content": "ok"},
    })
    with FakeOpenAIServer(handler_cls=handler) as srv:
        sampler = make_sampler(
            srv.base_url, "fake-model", temperature=0.9, max_tokens=128,
            sampling={"enable_thinking": False, "temperature": 0.3,
                      "top_k": 40})
        assert sampler([{"role": "user", "content": "hi"}]) is not None
    body = handler.requests[-1]
    # Thinking controls reach llama.cpp via the template-controls map AND stay
    # top-level for vLLM-style servers (request_sampling_payload contract).
    assert body["chat_format_template_controls"] == {"enable_thinking": False}
    assert body["enable_thinking"] is False
    assert body["top_k"] == 40
    # RFT's elevated temperature overrides the default section AFTER merge.
    assert body["temperature"] == 0.9
    assert body["max_tokens"] == 128


# ------------------------------------------------------------- token budget

class StubCounter:
    def __init__(self, n):
        self.n = n

    def count_messages(self, messages):
        return self.n


def test_budget_gate_skips_long_prompts_without_claiming(home, store):
    sampler = lambda msgs: {"role": "assistant", "content": "Paris"}
    long_prompt = "x" * 400
    res = run_rft(store, home, cfg(max_prompt_tokens=100), sampler=sampler,
                  judge_client_fn=JUDGE_OK,
                  seeds=[long_prompt], counter=StubCounter(500))
    assert res.skipped_budget == 1 and res.dataset_id is None
    # Skipped prompts stay UNclaimed so a future (compressed) run may retry.
    assert store.list_rft_prompt_keys() == []

    res2 = run_rft(store, home, cfg(max_prompt_tokens=100), sampler=sampler,
                   judge_client_fn=JUDGE_OK,
                   seeds=[long_prompt], counter=StubCounter(50))
    assert res2.prompts_sampled == 1


def test_context_prefix_cuts_after_last_user():
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "tool", "content": "t"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]
    prefix = context_prefix(msgs)
    assert prefix is not None
    assert [m["role"] for m in prefix] == ["system", "user", "user"]
    assert context_prefix([{"role": "assistant", "content": "x"}]) is None


# ------------------------------------------------------------ dispatcher e2e

def test_rft_job_flows_through_dispatcher(home, monkeypatch):
    """queued -> running -> done under a JobDispatcher tick; _result recorded;
    reap_stale never touches the in-process kind (INPROCESS_JOB_KINDS)."""
    handler = make_message_handler({
        "capital": {"role": "assistant", "content": "It is Paris."},
    })
    reg = ModelRegistry(home / "buffer" / "sable.db")
    try:
        with FakeOpenAIServer(handler_cls=handler) as srv:
            runner = make_rft_job_runner(
                reg, home=home, defaults=cfg(),
                endpoint_for=lambda: _ctx((srv.base_url, "fake-model")))
            tc = TrainingConfig()
            factory = lambda: Scheduler(registry=ModelRegistry(
                home / "buffer" / "sable.db"), cfg=tc,
                inprocess_runners={"rft": runner})
            sched = factory()
            job = sched.submit("rft", log_dir=home / "train_runs",
                               params={"goldens": [dict(GOLDEN_FRANCE)],
                                       "name": "rftdisp"},
                               enqueue_actor="test")
            assert job.state == "queued"
            d = JobDispatcher(db_path=reg.db_path, scheduler_factory=factory,
                              tick_s=0.2)
            d.start()
            try:
                deadline = time.time() + 15
                while time.time() < deadline:
                    cur = reg.get_job(job.job_id)
                    assert cur is not None
                    if cur.state in ("done", "error"):
                        break
                    # reap-safety probe mid-flight: an in-process kind must
                    # survive reaping even with a foreign pid.
                    assert cur.state in ("queued", "running")
                    time.sleep(0.1)
            finally:
                d.stop()
            cur = reg.get_job(job.job_id)
            assert cur is not None
            assert cur.state == "done", cur.error
            result = cur.params["_result"]
            assert result["rows_kept"] == 1 and result["dataset_id"]
            ds_dir = Path(result["dataset_dir"])
            mf = json.loads((ds_dir / "manifest.json").read_text())
            assert mf["source"] == "rft"
    finally:
        reg.close()


class _ctx:
    """Minimal context manager for the endpoint_for seam."""

    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self.value

    def __exit__(self, *exc):
        return False


def test_runner_refuses_when_master_switch_off(home):
    reg = ModelRegistry(home / "buffer" / "sable.db")
    try:
        runner = make_rft_job_runner(
            reg, home=home, defaults=RFTConfig(),  # enabled=False default
            endpoint_for=lambda: _ctx(("http://127.0.0.1:1/v1", "m")))

        class J:
            job_id = "j1"
            params = {}

        with pytest.raises(RftError, match="rft.enabled is false"):
            runner(J())
    finally:
        reg.close()


# -------------------------------------------------------------------- config

def test_rft_config_defaults_and_yaml_parsing():
    d = RFTConfig()
    assert d.enabled is False and d.sample_k == 4 and d.judge_min_score == 0.85
    parsed = RFTConfig.from_yaml({"rft": {"enabled": True, "sample_k": 8,
                                          "judge_min_score": 0.9}})
    assert parsed.enabled is True and parsed.sample_k == 8
    assert parsed.judge_min_score == 0.9
    # Unknown keys inside the section are ignored (with a warning), never crash.
    quiet = RFTConfig.from_yaml({"rft": {"nonsense": 1}})
    assert quiet.enabled is False


def test_load_config_recognizes_rft_section_without_warnings(tmp_path, caplog):
    import logging

    p = tmp_path / "c.yaml"
    p.write_text("rft:\n  enabled: true\n  sample_k: 6\n")
    with caplog.at_level(logging.WARNING):
        c = load_config(str(p))
    assert c.rft == {"enabled": True, "sample_k": 6}
    assert not [r for r in caplog.records if "unknown" in r.getMessage()]


# ------------------------------------------------------- DPO from RFT failures

def dpo_rows(ds_dir: Path):
    p = ds_dir / "dpo.jsonl"
    if not p.is_file():
        return None
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def seq_replies(responses):
    """Direct sampler reply: scripted assistant messages, last one repeats."""
    state = {"i": 0}

    def sample(messages):
        r = responses[min(state["i"], len(responses) - 1)]
        state["i"] += 1
        return {"role": "assistant", "content": r}
    return sample


def test_dpo_pairs_emitted_from_failed_candidates(home, store):
    """K=4 with 1 verified + 3 failed -> 1 SFT row AND 3 DPO pairs (<= cap),
    manifest files.dpo.jsonl mirrors the curater shape, and the worker's DPO
    loader consumes the file verbatim."""
    # best-of-K breaks at the first verified candidate -> put it last so all
    # three failures are collected as negative material.
    sampler = seq_replies(["wrong a", "wrong b", "wrong c", "It is Paris."])
    res = run_rft(store, home, cfg(sample_k=4), sampler=sampler,
                  goldens=[dict(GOLDEN_FRANCE)], name="rftdpo")
    assert res.rows_kept == 1 and res.dpo_pairs == 3
    d = ds_dir(res)
    assert len(sft_rows(d)) == 1

    rows = dpo_rows(d)
    assert rows is not None and len(rows) == 3
    for r in rows:
        # TRL preference schema, same context on both sides.
        assert r["prompt"] == [{"role": "user", "content": "capital of France"}]
        assert "Paris" in r["chosen"]
        assert r["rejected"] in ("wrong a", "wrong b", "wrong c")
        assert r["metadata"]["source"] == "rft"
        assert r["metadata"]["verifier"] == "golden"
    # Golden prompt -> no source trace; trace_ids stays empty (tolerated).
    mf = json.loads((d / "manifest.json").read_text())
    assert mf["files"]["dpo.jsonl"] == {"count": 3, "trace_ids": [],
                                        "format": "trl-preference"}
    assert mf["rft"]["dpo_pairs_emitted"] == 3

    # Consumable by the pipeline's DPO loader (worker dry-run: schema check,
    # no torch needed).
    from sable.training.worker import run_dpo
    out = run_dpo({"_dataset_dir": str(res.dataset_dir)}, None, True)
    assert out["dry_run"] is True and out["rows"] == 3


def test_all_fail_prompt_emits_no_orphan_pairs(home, store):
    """Two prompts: one yields a kept row, the other all-K fail. The failed
    prompt contributes NO DPO pairs (no orphan negatives)."""
    def sampler(messages):
        if "capital" in messages[-1]["content"]:
            return {"role": "assistant", "content": "It is Paris."}
        return {"role": "assistant", "content": "never right"}
    res = run_rft(store, home, cfg(sample_k=3), sampler=sampler,
                  goldens=[dict(GOLDEN_FRANCE),
                           {"prompt": "capital of Mars",
                            "expect_contains": ["Olympus Mons"]}],
                  name="rftorphan")
    assert res.rows_kept == 1 and res.dpo_pairs == 0
    rows = dpo_rows(ds_dir(res))
    assert rows is not None and rows == []  # curater parity: empty file, kept
    mf = json.loads((ds_dir(res) / "manifest.json").read_text())
    assert mf["files"]["dpo.jsonl"]["count"] == 0


def test_emit_dpo_off_is_byte_identical_to_legacy(home, store):
    """emit_dpo=False -> no dpo.jsonl at all and the manifest carries neither a
    files.dpo.jsonl entry nor rft.dpo_pairs_emitted (pre-feature shape)."""
    sampler = seq_replies(["wrong", "It is Paris."])
    res = run_rft(store, home, cfg(sample_k=2, emit_dpo=False), sampler=sampler,
                  goldens=[dict(GOLDEN_FRANCE)], name="rftoff")
    assert res.rows_kept == 1 and res.dpo_pairs == 0
    d = ds_dir(res)
    assert not (d / "dpo.jsonl").exists()
    mf = json.loads((d / "manifest.json").read_text())
    assert set(mf["files"]) == {"sft.jsonl"}
    assert "dpo_pairs_emitted" not in mf["rft"]


def test_dpo_pair_cap_prefers_diverse_failure_reasons(home, store):
    """Cap per prompt; distinct failure reasons are picked before duplicates
    of the same reason even when the dupes come first in document order."""
    # Judge scores: two identical fails (0.10), one distinct fail (0.20), keep.
    scores = iter([0.1, 0.1, 0.2, 0.9])
    judge = lambda msgs, model: json.dumps(
        {"score": next(scores), "reason": "x"})
    sampler = seq_replies(["bad one", "bad two", "bad three", "good answer"])
    res = run_rft(store, home, cfg(sample_k=4, max_dpo_pairs_per_prompt=2),
                  sampler=sampler, judge_client_fn=judge,
                  seeds=["how do I greet a user?"], name="rftcap")
    assert res.rows_kept == 1 and res.dpo_pairs == 2
    rows = dpo_rows(ds_dir(res))
    assert rows is not None
    reasons = [r["metadata"]["rejected_reason"] for r in rows]
    # Document order would have paired two identical "judge 0.10" fails;
    # diversity picks one of each distinct reason.
    assert len(set(reasons)) == 2
    assert all(r["chosen"] == "good answer" for r in rows)
    mf = json.loads((ds_dir(res) / "manifest.json").read_text())
    assert mf["files"]["dpo.jsonl"]["count"] == 2


def test_dpo_pairs_redact_rejected_and_prompt(home, store):
    """Fresh model output may echo prompt secrets: chosen/rejected/prompt all
    pass core/redact before the durable write."""
    secret = "sk-abcdefghijklmnop1234567890"
    sampler = seq_replies([f"the key is {secret} but wrong",
                           "Paris, and here is a key: " + secret])
    res = run_rft(store, home, cfg(sample_k=2), sampler=sampler,
                  goldens=[dict(GOLDEN_FRANCE)], name="rftredact")
    assert res.dpo_pairs == 1
    rows = dpo_rows(ds_dir(res))
    assert rows is not None
    row = rows[0]
    assert secret not in row["rejected"] and "[REDACTED" in row["rejected"]
    assert secret not in row["chosen"] and "[REDACTED" in row["chosen"]


def test_dpo_manifest_lists_trace_derived_kept_side(home, store):
    """Trace-derived kept row -> files.dpo.jsonl.trace_ids carries the source
    trace id (curater shape); sampling-derived negatives add nothing."""
    seed_trace(store, home, trace_id="tdpo", messages=[
        {"role": "user", "content": "capital of France"},
        {"role": "assistant", "content": "old answer"},
    ])
    sampler = seq_replies(["wrong", "It is Paris."])
    # Content-aware judge: the first candidate must actually FAIL or there is
    # no negative to pair.
    judge = lambda msgs, model: json.dumps(
        {"score": 0.1 if "wrong" in (msgs[-1].get("content") or "") else 0.9,
         "reason": "x"})
    res = run_rft(store, home, cfg(sample_k=2), sampler=sampler,
                  judge_client_fn=judge, name="rfttr")
    assert res.rows_kept == 1 and res.dpo_pairs == 1
    mf = json.loads((ds_dir(res) / "manifest.json").read_text())
    assert mf["files"]["dpo.jsonl"] == {"count": 1, "trace_ids": ["tdpo"],
                                        "format": "trl-preference"}


def test_dpo_run_preserves_ledger_and_claim_invariants(home, store):
    """DPO emission rides the existing write path: prompt ledger dedup and the
    claim/release invariants are untouched (re-run still skips)."""
    sampler = seq_replies(["wrong", "It is Paris."])
    judge = lambda msgs, model: json.dumps(
        {"score": 0.1 if "wrong" in (msgs[-1].get("content") or "") else 0.9,
         "reason": "x"})
    res1 = run_rft(store, home, cfg(sample_k=2), sampler=sampler,
                   judge_client_fn=judge, seeds=["cap q"], name="rftinv")
    assert res1.dpo_pairs == 1
    key = prompt_key("cap q")
    row = store._conn.execute(
        "SELECT dataset_id, kept FROM rft_prompts WHERE prompt_key = ?",
        (key,)).fetchone()
    assert (row["dataset_id"], row["kept"]) == (res1.dataset_id, 1)
    res2 = run_rft(store, home, cfg(sample_k=2), sampler=sampler,
                   judge_client_fn=JUDGE_OK, seeds=["cap q"], name="rftinv2")
    assert res2.dataset_id is None and res2.prompts_sampled == 0


def test_rft_config_dpo_knobs():
    d = RFTConfig()
    assert d.emit_dpo is True and d.max_dpo_pairs_per_prompt == 3
    parsed = RFTConfig.from_yaml({"rft": {"emit_dpo": False,
                                          "max_dpo_pairs_per_prompt": 1}})
    assert parsed.emit_dpo is False and parsed.max_dpo_pairs_per_prompt == 1


def test_cli_rft_subcommand_parses_and_refuses_when_disabled(tmp_path,
                                                             monkeypatch):
    import argparse

    from sable.cli import build_parser, cmd_rft

    args = build_parser().parse_args(["rft", "--k", "8", "--seeds", "hi"])
    assert args.command == "rft" and args.k == 8 and args.seeds == ["hi"]

    monkeypatch.setenv("SABLE_HOME", str(tmp_path / "h"))
    c = tmp_path / "c.yaml"
    c.write_text("rft:\n  enabled: false\n")
    ns = argparse.Namespace(config=str(c), verbose=False, seeds=None,
                            goldens=None, k=None, max_prompts=None,
                            name=None, wait=False)
    assert cmd_rft(ns) == 2  # master switch off -> refuse before queueing

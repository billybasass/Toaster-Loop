"""Backlog #6 tests — Compression v2: tool-output prune pass + iterative summaries.

Covers: deterministic prune pre-pass (placement, size floor, tail protection,
idempotency, placeholder char count), prune-alone-suffices (no LLM call),
iterative-update vs from-scratch summarizer prompts, structured template /
historical heading in summary placement, extractive-digest previous-summary
awareness, and runner threading of the rolling summary across runs.
"""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fake_server import FakeOpenAIServer, make_message_handler  # noqa: E402
from sable.core.agent_runner import SableAgentClient  # noqa: E402
from sable.core.compression import (  # noqa: E402
    SUMMARY_PREFIX,
    _PRUNE_MIN_CHARS,
    compress_messages,
    estimate_tokens,
    prune_old_tool_results,
    summary_from_messages,
)


def _msg(role, content, **extra):
    return {"role": role, "content": content, **extra}


def _tool(content, name="read_file"):
    return {"role": "tool", "name": name, "tool_call_id": "c",
            "content": content}


# ------------------------------------------------------------- prune pre-pass

def test_prune_replaces_long_tool_msgs_outside_tail():
    long1, long2 = "a" * 500, "b" * 400
    msgs = [_msg("user", "q"), _tool(long1, name="terminal"),
            _msg("assistant", "a"), _tool(long2),
            _msg("user", "tail q"), _msg("assistant", "tail a")]
    out, n = prune_old_tool_results(msgs, tail_start_index=4)
    assert n == 2
    assert out[1]["content"].startswith("[pruned tool output:")
    assert "terminal" in out[1]["content"]
    assert f"{len(long1)} chars" in out[1]["content"]  # char count present
    assert f"{len(long2)} chars" in out[3]["content"]
    # untouched rows keep identity; input never mutated
    assert out[0] is msgs[0] and out[4] is msgs[4]
    assert msgs[1]["content"] == long1


def test_prune_keeps_short_and_inside_tail():
    short = "x" * _PRUNE_MIN_CHARS  # floor is strict >
    inside = "y" * 500
    msgs = [_tool("z" * 300), _tool(short), _msg("user", "q"), _tool(inside)]
    out, n = prune_old_tool_results(msgs, tail_start_index=3)
    assert n == 1
    assert out[0]["content"].startswith("[pruned tool output:")
    assert out[1]["content"] == short       # <= floor untouched
    assert out[3]["content"] == inside      # inside protected tail untouched


def test_prune_idempotent_second_pass_noop():
    msgs = [_msg("user", "q"), _tool("a" * 500), _msg("assistant", "a")]
    once, n1 = prune_old_tool_results(msgs, tail_start_index=3)
    twice, n2 = prune_old_tool_results(once, tail_start_index=3)
    assert n1 == 1 and n2 == 0
    assert twice == once


def test_compress_returns_without_llm_when_prune_suffices():
    # Fat tool results push the estimate over threshold; pruning alone brings
    # it back under — the summarizer must never be invoked. Ends on an
    # assistant text turn so the tail boundary is always prunable-safe.
    msgs = [_msg("system", "S")]
    for i in range(4):
        msgs.append(_msg("user", f"q{i}"))
        msgs.append({"role": "assistant", "content": None,
                     "tool_calls": [{"id": f"c{i}", "type": "function",
                                     "function": {"name": "list_dir",
                                                  "arguments": "{}"}}]})
        msgs.append(_tool("t" * 4000))
        msgs.append(_msg("assistant", f"noted {i}"))
    threshold = estimate_tokens(msgs) - 100
    assert estimate_tokens(msgs) > threshold

    def boom(prompt):
        raise AssertionError("summarizer must not be called when pruning suffices")

    out = compress_messages(msgs, boom, threshold_tokens=threshold,
                            keep_recent_tokens=256)
    assert estimate_tokens(out) <= threshold
    pruned = [m for m in out if m.get("role") == "tool"
              and str(m["content"]).startswith("[pruned tool output:")]
    assert pruned  # pruning actually happened


# --------------------------------------------------- iterative / structured

def _head_msgs():
    msgs = [_msg("system", "S")]
    for i in range(20):
        msgs.append(_msg("user", f"question {i} " + "u" * 300))
        msgs.append(_msg("assistant", f"answer {i} " + "v" * 300))
    return msgs


def test_scratch_prompt_has_structured_template():
    prompts = []

    def fake_summarize(prompt):
        prompts.append(prompt)
        return "## What has been accomplished\n- did things"

    out = compress_messages(_head_msgs(), fake_summarize, threshold_tokens=10,
                            keep_recent_tokens=256)
    assert len(prompts) == 1
    p = prompts[0]
    assert "PREVIOUS SUMMARY" not in p
    # structured template sections offered to the summarizer
    for section in ("## What has been accomplished",
                    "## Key decisions & files touched",
                    "## Open threads & pending work"):
        assert section in p
    # summary placed under the historical heading prefix (not live intent)
    summary_msg = next(m for m in out
                       if str(m["content"]).startswith(SUMMARY_PREFIX))
    assert summary_msg["content"].startswith("[Summary of earlier conversation")
    assert "not new instructions" in SUMMARY_PREFIX


def test_previous_summary_uses_update_prompt():
    prompts = []

    def fake_summarize(prompt):
        prompts.append(prompt)
        return "updated summary body"

    out = compress_messages(_head_msgs(), fake_summarize, threshold_tokens=10,
                            keep_recent_tokens=256,
                            previous_summary="OLD SUMMARY TEXT 42")
    assert len(prompts) == 1
    p = prompts[0]
    assert "PREVIOUS SUMMARY:" in p          # iterative update framing
    assert "NEW TURNS TO FOLD IN:" in p
    assert "OLD SUMMARY TEXT 42" in p        # old summary fed forward verbatim
    assert "updated summary body" in out[-1]["content"] or any(
        "updated summary body" in str(m["content"]) for m in out)


def test_extractive_fallback_carries_previous_summary():
    def boom(prompt):
        raise RuntimeError("endpoint down")

    out = compress_messages(_head_msgs(), boom, threshold_tokens=10,
                            keep_recent_tokens=256,
                            previous_summary="PRIOR DIGEST MARKER")
    summary_msg = next(m for m in out
                       if str(m["content"]).startswith(SUMMARY_PREFIX))
    assert "user asked" in summary_msg["content"]  # extractive shape kept
    assert "PRIOR DIGEST MARKER" in summary_msg["content"]


def test_summary_from_messages_roundtrip():
    msgs = [_msg("system", "S")]
    for i in range(20):
        msgs.append(_msg("user", f"q{i} " + "u" * 300))
        msgs.append(_msg("assistant", f"a{i} " + "v" * 300))
    assert summary_from_messages(msgs) is None
    out = compress_messages(msgs, lambda p: "ROLLING SUMMARY BODY",
                            threshold_tokens=10, keep_recent_tokens=256)
    assert summary_from_messages(out) == "ROLLING SUMMARY BODY"


# --------------------------------------------------------- runner threading

def test_runner_forwards_summary_across_compactions(monkeypatch, tmp_path):
    monkeypatch.setenv("SABLE_HOME", str(tmp_path / "uf"))
    handler = make_message_handler({"*": {"role": "assistant",
                                          "content": "final answer"}})
    prompts: list[str] = []

    def fake_summarize(self, prompt):  # captures summarizer prompts, no LLM
        prompts.append(prompt)
        return f"SUMMARY-v{len(prompts)}"

    monkeypatch.setattr(SableAgentClient, "_summarize", fake_summarize)
    seed_history = []
    for i in range(6):
        seed_history.append(_msg("user", f"old question {i} " + "h" * 400))
        seed_history.append(_msg("assistant", f"old answer {i} " + "g" * 400))
    with FakeOpenAIServer(handler_cls=handler) as srv:
        client = SableAgentClient(base_url=srv.base_url, model="m",
                                    compress_threshold_tokens=400,
                                    keep_recent_tokens=256)
        r1 = client.run("first task", history=list(seed_history))
        assert len(prompts) == 1
        assert "PREVIOUS SUMMARY" not in prompts[0]  # first compaction: cold
        assert r1.summary == "SUMMARY-v1"

        # Non-system remainder of run 1 becomes run 2's history; the new task
        # pushes it back over the threshold for a second compaction.
        history = [m for m in r1.messages if m.get("role") != "system"]
        r2 = client.run("second task " + "y" * 1500, history=history,
                        previous_summary=r1.summary)
        assert len(prompts) == 2
        assert "PREVIOUS SUMMARY:" in prompts[1]     # second: iterative update
        assert "SUMMARY-v1" in prompts[1]
        assert r2.summary == "SUMMARY-v2"


def test_runner_backward_compatible_call(monkeypatch, tmp_path):
    """Old call signature (no previous_summary) still valid; summary None."""
    monkeypatch.setenv("SABLE_HOME", str(tmp_path / "uf"))
    handler = make_message_handler({"*": {"role": "assistant",
                                          "content": "all good"}})
    with FakeOpenAIServer(handler_cls=handler) as srv:
        client = SableAgentClient(base_url=srv.base_url, model="m")
        res = client.run("hello there")  # compression off (threshold 0)
    assert res.completed
    assert res.summary is None

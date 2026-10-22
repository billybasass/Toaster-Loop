"""Tests for the built-in runtime features:

  1. Text tool-call fallback parsing (core/tool_calls.py)
  2. Context compression (core/compression.py)
  3. todo_write / todo_read tools
  4. search_traces tool over the experience buffer
  + response-only label masking helpers (training/worker.py, no torch needed)

All hermetic; nothing touches ~/.sable (conftest redirects SABLE_HOME).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, cast

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fake_server import FakeOpenAIServer  # noqa: E402

from sable.core import tools as uf_tools  # noqa: E402
from sable.core.agent_runner import SableAgentClient  # noqa: E402
from sable.core.compression import (  # noqa: E402
    SUMMARY_PREFIX,
    compress_messages,
    estimate_tokens,
    split_for_compression,
)
from sable.core.tool_calls import (  # noqa: E402
    parse_text_tool_calls,
    strip_tool_call_markup,
)
from sable.training.worker import (  # noqa: E402
    assistant_char_regions,
    sharegpt_to_messages,
    tokenize_with_labels,
)

LT = "<"  # avoid editor/template tooling mangling tag-like test fixtures


@pytest.fixture(scope="module")
def fake_server():
    with FakeOpenAIServer() as srv:
        yield srv


# ------------------------------------------------------ 1. text tool calls

def test_parse_fenced_json_call():
    content = (
        "Sure!\n```json\n"
        + json.dumps({"name": "read_file", "arguments": {"path": "a.py"}})
        + "\n```\n"
    )
    calls = parse_text_tool_calls(content)
    assert len(calls) == 1
    fn = calls[0]["function"]
    assert fn["name"] == "read_file"
    assert json.loads(fn["arguments"]) == {"path": "a.py"}
    assert calls[0]["id"].startswith("call_text_")
    # prose survives, block is stripped
    assert strip_tool_call_markup(content) == "Sure!"


def test_parse_openai_nested_and_list():
    content = ("```json\n"
               + json.dumps([{"function": {"name": "list_dir",
                                           "arguments": '{"path": "."}'}}])
               + "\n```")
    calls = parse_text_tool_calls(content)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "list_dir"
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "."}


def test_parse_xmlish_call():
    content = (f"Let me check.\n{LT}tool_call>read_file"
               + json.dumps({"path": "x.txt"}) + f"{LT}/tool_call>")
    calls = parse_text_tool_calls(content)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "read_file"


def test_parse_negatives():
    assert parse_text_tool_calls(None) == []
    assert parse_text_tool_calls("Just a normal answer.") == []
    # non-tool JSON is left alone
    assert parse_text_tool_calls(f"```json\n{json.dumps({'foo': 1})}\n```") == []
    # leaked chat-template markup => never rewritten
    im = LT + "|"
    assert parse_text_tool_calls(
        f"{im}im_start{im}assistant hi {im}im_end{im}") == []


def test_agent_loop_executes_text_tool_call():
    """End-to-end: a server that returns a fenced JSON call gets it executed."""
    from fake_server import _Handler

    class TextCallHandler(_Handler):
        def _completion_payload(self, payload):  # noqa: N802
            messages = payload.get("messages", [])
            saw_tool_result = any(m.get("role") == "tool" for m in messages)
            if not saw_tool_result:
                msg = {"role": "assistant", "content":
                       'Working.\n```json\n{"name": "list_dir", '
                       '"arguments": {"path": "."}}\n```'}
                finish = "stop"  # NOTE: no native tool_calls field at all
            else:
                msg = {"role": "assistant",
                       "content": "Saw the directory listing."}
                finish = "stop"
            return {
                "id": "cmpl-text", "object": "chat.completion",
                "model": payload.get("model", "fake-model"),
                "choices": [{"index": 0, "message": msg,
                             "finish_reason": finish}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                          "total_tokens": 2},
            }

    with FakeOpenAIServer(handler_cls=TextCallHandler) as fs:
        client = SableAgentClient(base_url=fs.base_url, model="fake-model")
        result = client.run("list stuff")
    assert result.completed
    roles = [m.get("role") for m in result.messages]
    assert "tool" in roles  # the text call was actually executed
    assert "Saw the directory listing." in result.final_response


# ------------------------------------------------------ 2. compression

def _msg(role, content):
    return {"role": role, "content": content}


def test_estimate_tokens_scales_with_size():
    small = estimate_tokens([_msg("user", "hi")])
    big = estimate_tokens([_msg("user", "x" * 4000)])
    assert big > small * 10


def test_split_and_compress_replaces_head():
    msgs = [_msg("system", "SYS")]
    for i in range(30):
        msgs.append(_msg("user", f"question {i} " + "y" * 200))
        msgs.append(_msg("assistant", f"answer {i} " + "z" * 200))

    parts = split_for_compression(msgs[1:], keep_recent_tokens=300)
    assert parts is not None
    head, tail = parts
    assert head and tail
    assert tail[0]["role"] in ("user", "assistant")

    out = compress_messages(msgs, summarize=None, threshold_tokens=10,
                            keep_recent_tokens=300)
    # system prompt kept first, exactly one summary message, tail preserved
    assert out[0]["content"] == "SYS"
    summaries = [m for m in out if str(m["content"]).startswith(SUMMARY_PREFIX)]
    assert len(summaries) == 1
    assert out[-len(tail):] == tail
    # idempotent: recompressing under threshold is a no-op
    again = compress_messages(out, summarize=None, threshold_tokens=10 ** 9)
    assert again == out


def test_compress_disabled_and_short_noop():
    msgs = [_msg("system", "S"), _msg("user", "hi")]
    assert compress_messages(msgs, None, threshold_tokens=0) == msgs
    assert compress_messages(msgs, None, threshold_tokens=10_000) == msgs


def test_llm_summarizer_failure_falls_back_to_digest():
    msgs = [_msg("system", "S")]
    for i in range(20):
        msgs.append(_msg("user", f"q{i} " + "u" * 300))
        msgs.append(_msg("assistant", f"a{i} " + "v" * 300))

    def boom(prompt):
        raise RuntimeError("endpoint down")

    out = compress_messages(msgs, boom, threshold_tokens=10,
                            keep_recent_tokens=200)
    summary = next(m for m in out
                   if str(m["content"]).startswith(SUMMARY_PREFIX))
    assert "user asked" in summary["content"]  # extractive digest shape


def test_tool_call_pairing_preserved():
    """Tail must never start with an orphan tool message."""
    msgs: list = [_msg("system", "S")]
    for i in range(10):
        msgs.append(_msg("user", f"q{i}"))
        msgs.append({"role": "assistant", "content": None,
                     "tool_calls": [{"id": f"c{i}", "type": "function",
                                     "function": {"name": "list_dir",
                                                  "arguments": "{}"}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}",
                     "content": "x" * 300})
    out = compress_messages(msgs, None, threshold_tokens=10,
                            keep_recent_tokens=250)
    body = [m for m in out if not str(m["content"] or "").startswith(SUMMARY_PREFIX)
            and m.get("role") != "system"]
    assert body[0]["role"] != "tool"


# ---------------------------------------------------------- 3. todo tools

@pytest.fixture()
def uf_home(tmp_path, monkeypatch):
    monkeypatch.setenv("SABLE_HOME", str(tmp_path / "uf"))
    return tmp_path / "uf"


def test_todo_roundtrip_and_merge(uf_home):
    assert uf_tools.todo_read() == "(todo list is empty)"
    out = uf_tools.todo_write(todos=[
        {"id": "a", "content": "first", "status": "pending"},
        {"id": "b", "content": "second"},
    ])
    assert "[ ] a: first" in out and "[?] b: second" not in out  # b defaults pending
    out = uf_tools.todo_write(todos=[{"id": "a", "content": "first",
                                      "status": "in_progress"}], merge=True)
    assert "[~] a: first" in out and "[ ] b: second" in out
    # replace semantics drop old items
    out = uf_tools.todo_write(todos=[{"id": "z", "content": "only",
                                      "status": "done"}])
    assert out.strip() == "[x] z: only"


def test_todo_validation(uf_home):
    # deliberately wrong type: the tool must reject non-list input gracefully
    assert uf_tools.todo_write(todos=cast(Any, "nope")).startswith("error")
    assert uf_tools.todo_write(todos=[{"content": ""}]).startswith("error")


def test_todo_salvages_markdown_strings(uf_home):
    """Weak models emit '- [ ] text' checklist lines instead of objects."""
    out = uf_tools.todo_write(todos=["- [ ] map source tree",
                                     "[x] done thing",
                                     "plain item"])
    assert "map source tree" in out and "done thing" in out
    assert "plain item" in out
    # status marker parsed from the checkbox -> line renders as "[x] …"
    assert any(l.startswith("[x] ") and "done thing" in l
               for l in out.splitlines())
    # oversized list refused, existing state untouched
    many = [{"content": f"t{i}"} for i in range(uf_tools.TODO_MAX_ITEMS + 1)]
    assert uf_tools.todo_write(todos=many).startswith("error")


# ------------------------------------------------------- 4. search_traces

def test_search_traces_over_buffer(uf_home):
    from sable.store import TraceRecorder

    rec = TraceRecorder(home=uf_home)
    row = rec.record(
        task="rename the ffmpeg output files",
        messages=[{"role": "user", "content": "rename the ffmpeg output files"},
                  {"role": "assistant", "content": "used mmv to glob-rename"}],
        backend="sable", model_alias="m", base_url="http://x/v1",
        completed=True,
    )
    rec.close()
    assert row is not None

    out = uf_tools.search_traces(pattern="ffmpeg")
    assert row.trace_id in out and "[success]" in out
    assert "no matching traces" in uf_tools.search_traces(pattern="zzz-nope-zzz")
    assert uf_tools.search_traces(status="bogus").startswith("error")
    # empty pattern lists recent
    assert row.trace_id in uf_tools.search_traces()


def test_search_traces_empty_buffer(uf_home):
    assert "no traces" in uf_tools.search_traces(pattern="anything")


def test_tool_schemas_expose_new_tools():
    names = {s["function"]["name"] for s in uf_tools.TOOL_SCHEMAS}
    assert {"todo_write", "todo_read", "search_traces"} <= names
    # Every impl is either a default schema or capability-gated; the
    # capability tools must NOT be in the default set (per-agent opt-in).
    assert set(names) | set(uf_tools.CAPABILITY_TOOLS) == set(uf_tools.TOOL_IMPLS)
    assert not names & set(uf_tools.CAPABILITY_TOOLS)
    assert set(uf_tools.CAPABILITY_TOOL_SCHEMAS) == set(uf_tools.CAPABILITY_TOOLS)


# ------------------------------------- 5. masking helpers (no torch needed)

class FakeTok:
    """Minimal chat-template tokenizer over a fixed unigram vocab."""

    pad_token_id = 0
    eos_token = " "
    is_fast = True

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False):
        out = "".join(f"[{m['role']}]{m['content']}" for m in messages)
        if add_generation_prompt:
            out += "[assistant]"
        return out

    def __call__(self, text, truncation=False, max_length=None,
                 return_offsets_mapping=False, add_special_tokens=True):
        # tokens = each char; offsets = exact char positions
        toks = list(text)
        enc: dict = {"input_ids": [ord(c) for c in toks],
                     "attention_mask": [1] * len(toks)}
        if return_offsets_mapping:
            enc["offset_mapping"] = [(i, i + 1) for i in range(len(toks))]
        return enc


def test_assistant_char_regions_and_labels():
    tok = FakeTok()
    convs = [
        {"from": "system", "value": "sys"},
        {"from": "human", "value": "hello"},
        {"from": "gpt", "value": "WORLD"},
        {"from": "human", "value": "again"},
        {"from": "gpt", "value": "END"},
    ]
    regions = assistant_char_regions(tok, convs)
    assert regions is not None and len(regions) == 2
    full = tok.apply_chat_template(sharegpt_to_messages(convs), tokenize=False)
    # The generation-prompt header stays in the prefix; the supervised region
    # is exactly the assistant content.
    assert {full[s:e] for s, e in regions} == {"WORLD", "END"}
    for start, _end in regions:
        assert full[:start].endswith("[assistant]")

    enc = tokenize_with_labels(tok, convs, max_length=10_000,
                               mask_instructions=True)
    assert enc is not None
    labels = enc["labels"]
    assert len(labels) == len(enc["input_ids"])
    # the response letters are supervised...
    text = full
    for i, (tok_id, label) in enumerate(zip(enc["input_ids"], labels)):
        ch = chr(tok_id)
        if ch in "WORLD" and text[i] == ch:
            assert label != -100
    # ...and instruction text is not
    for i, ch in enumerate(text):
        if ch == "h" and text[i - 1 : i + 5] == "[human":
            assert labels[i] == -100


def test_tokenize_with_labels_noncomposing_template():
    class WeirdTok(FakeTok):
        def apply_chat_template(self, messages, tokenize=False,
                                add_generation_prompt=False):
            base = super().apply_chat_template(messages, tokenize,
                                               add_generation_prompt)
            return f"{len(messages)}::{base}"  # prefix length varies

    convs = [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "yo"}]
    assert assistant_char_regions(WeirdTok(), convs) is None
    assert tokenize_with_labels(WeirdTok(), convs, 100, True) is None

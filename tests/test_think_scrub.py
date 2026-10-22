"""Streaming think/reasoning scrubber (inspiration pass 3, item #17).

Covers:
  1. StreamingThinkScrubber unit semantics — closed pairs (even inline),
     unterminated blocks at block boundaries, prose that merely *mentions* a
     tag mid-line, tags split across delta boundaries, orphan close tags,
     flush inside/outside a block, reset between turns.
  2. strip_think_blocks whole-string pass (idempotent).
  3. Agent runner integration — a streaming backend leaking inline
     <thinking> into plain content yields scrubbed message.delta events AND a
     scrubbed final message (what history/trace record); non-streaming backends
     get the same via the finalize chokepoint.

All hermetic: local fake servers only.
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fake_server import FakeOpenAIServer  # noqa: E402
from sable.core.agent_runner import SableAgentClient  # noqa: E402
from sable.core.think_scrub import (  # noqa: E402
    THINK_TAG_NAMES,
    StreamingThinkScrubber,
    strip_think_blocks,
)


def run_stream(deltas):
    s = StreamingThinkScrubber()
    out = "".join(s.feed(d) for d in deltas)
    return out + s.flush()


# ------------------------------------------------------------- 1. streaming

def test_closed_pair_removed_inline():
    assert run_stream(["Here is ", "<thinking>", "secret plan", "</thinking>",
                       "the answer."]) == "Here is the answer."


def test_closed_pair_tag_split_across_deltas():
    # Tag names split across delta boundaries must still resolve.
    assert run_stream(["a<thi", "nking>hidden</thi", "nking>b"]) == "ab"


def test_unterminated_block_at_boundary_suppressed_to_flush():
    out = run_stream(["<think>", "pondering ", "and pondering", " more"])
    assert out == ""  # never leaks, flush discards the unterminated tail


def test_prose_mentioning_tag_midline_survives():
    text = ("Docs say to wrap chains like <thinking>your plan</thinking>? "
            "No — inline mentions of <thinking> stay.")
    # Closed pairs are *always* stripped (deliberate), but an unterminated
    # mention mid-line must never suppress the rest of the line.
    assert "<thinking>your plan" not in strip_think_blocks(text)
    solo = "Use the <thinking> tag to enable chains."
    assert strip_think_blocks(solo) == solo


def test_open_tag_midline_not_a_block_start():
    out = run_stream(["prose <think", "ing> still visible"])
    # '<' + 'ing>' never forms a boundary block: everything survives.
    assert "<thinking>" in out or "<thinking" in out


def test_orphan_close_tags_dropped():
    assert run_stream(["visible ", "</thinking>", " text"]) == "visible  text"


def test_partial_tag_tail_flushed_as_prose():
    # A trailing '<thi' that never becomes a tag is released at flush.
    assert run_stream(["ends with <thi"]) == "ends with <thi"


def test_reset_between_turns():
    s = StreamingThinkScrubber()
    assert s.feed("<thinking>hidden") == ""
    s.reset()
    assert s.feed("clean turn") == "clean turn"
    assert s.flush() == ""


def test_block_after_newline_is_boundary():
    # Closed block after a newline: suppressed, following prose resumes.
    out = run_stream(["answer line\n", "<thinking>", "hidden", "</thinking>",
                      "tail visible"])
    assert out == "answer line\ntail visible"


def test_unterminated_block_swallows_to_end():
    # Deliberate: once an unterminated block opens at a boundary, everything
    # after it is treated as reasoning (leaking beats truncating).
    out = run_stream(["answer line\n", "<thinking>", "hidden", "\n",
                      "tail visible"])
    assert out == "answer line\n"


def test_central_tag_list_bound_everywhere():
    for name in THINK_TAG_NAMES:
        tag = f"<{name.lower()}>"
        assert run_stream([tag, "hidden", f"</{name.lower()}>"]) == ""


# ------------------------------------------------------ 2. whole-string pass

def test_strip_think_blocks_idempotent():
    raw = "a<thinking>x</thinking>b"
    once = strip_think_blocks(raw)
    assert once == "ab"
    assert strip_think_blocks(once) == once


def test_strip_think_blocks_noop_plain_text():
    assert strip_think_blocks("plain prose, no tags") == "plain prose, no tags"
    assert strip_think_blocks("") == ""


# --------------------------------------------------- 3. runner integration

class InlineThinkSSEHandler(BaseHTTPRequestHandler):
    """Streams a reply whose plain content carries inline thinking."""

    def log_message(self, format, *args):  # noqa: A002 - silence access log
        pass

    def do_POST(self):
        chunks = [
            {"delta": {"role": "assistant"}},
            {"delta": {"content": "<thinking>"}},
            {"delta": {"content": "secret chain of thought"}},
            {"delta": {"content": "</thinking>"}},
            {"delta": {"content": "The answer is 42."}},
            {"delta": {}, "finish_reason": "stop"},
        ]
        body = "".join(
            "data: " + json.dumps({"id": "x", "object": "chat.completion.chunk",
                                   "model": "m",
                                   "choices": [dict(index=0, **c)]}) + "\n\n"
            for c in chunks) + "data: [DONE]\n\n"
        raw = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(raw)


def test_runner_stream_scrubs_deltas_and_final_message():
    events = []
    with FakeOpenAIServer(handler_cls=InlineThinkSSEHandler) as fs:
        client = SableAgentClient(fs.base_url, "m", tools=[], tool_impls={})
        client.on_event = lambda kind, data: events.append((kind, data))
        res = client.run("what is the answer?")
    streamed = "".join(d["text"] for k, d in events if k == "message.delta")
    assert streamed == "The answer is 42."
    assert "secret chain of thought" not in streamed
    # The finalized result (what history + trace record) is scrubbed too.
    assert res.final_response == "The answer is 42."
    last = res.messages[-1]
    assert last["role"] == "assistant" and last["content"] == "The answer is 42."


class InlineThinkJSONHandler(BaseHTTPRequestHandler):
    """Non-streaming backend leaking inline thinking in plain content."""

    def log_message(self, format, *args):  # noqa: A002 - silence access log
        pass

    def do_POST(self):
        body = json.dumps({
            "id": "x", "object": "chat.completion", "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant",
                                     "content": "<thinking>hidden work</thinking>"
                                                "done."}}],
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_runner_non_streaming_final_scrubbed():
    with FakeOpenAIServer(handler_cls=InlineThinkJSONHandler) as fs:
        client = SableAgentClient(fs.base_url, "m", tools=[], tool_impls={})
        res = client.run("go")
    assert res.final_response == "done."
    assert res.messages[-1]["content"] == "done."

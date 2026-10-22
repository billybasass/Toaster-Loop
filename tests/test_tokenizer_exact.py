"""Improvement #4 tests — exact token accounting via llama.cpp /tokenize.

Covers: TokenCounter hitting POST /tokenize at the server ROOT (base_url's
/v1 stripped), LRU cache hits avoiding repeat HTTP, graceful chars/4 fallback
on non-200 / refused connection, circuit breaker stop-probing + cooldown
recovery (+ single WARNING), degraded count_messages == estimate_tokens
byte-for-byte, compress_messages threshold decisions driven by exact counts
where chars/4 disagrees (incl. CJK-heavy content at the trigger boundary),
and runner wiring (injected counter, lazy construction, switch_model reset).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fake_server import (  # noqa: E402
    FakeOpenAIServer,
    fake_tokenize_count,
    make_tokenize_handler,
)
from sable.core.agent_runner import SableAgentClient  # noqa: E402
from sable.core.compression import (  # noqa: E402
    SUMMARY_PREFIX,
    compress_messages,
    estimate_tokens,
)
from sable.core.tokenizer import TokenCounter  # noqa: E402

DEAD_URL = "http://127.0.0.1:9/v1"  # discard port: instant connection refused


def _msg(role, content, **extra):
    return {"role": role, "content": content, **extra}


@pytest.fixture()
def tok_server():
    handler = make_tokenize_handler({"*": "final answer"})
    with FakeOpenAIServer(handler_cls=handler) as srv:
        handler.tokenize_requests.clear()
        yield srv, handler


# ------------------------------------------------------------ TokenCounter

def test_count_uses_tokenize_endpoint(tok_server):
    srv, handler = tok_server
    tc = TokenCounter(srv.base_url)
    text = "hello exact tokenizer world"  # fake counts ceil(len/2), not len/4
    n = tc.count(text)
    assert n == fake_tokenize_count(text)
    assert n != len(text) // 4  # proves the exact path, not the fallback
    assert handler.tokenize_requests == [text]


def test_lru_cache_hit_avoids_second_http_call(tok_server):
    srv, handler = tok_server
    tc = TokenCounter(srv.base_url)
    a = tc.count("cache me please")
    b = tc.count("cache me please")
    assert a == b
    assert len(handler.tokenize_requests) == 1  # second call served from LRU


def test_lru_cache_is_bounded(tok_server):
    srv, _ = tok_server
    tc = TokenCounter(srv.base_url, cache_size=2)
    for i in range(5):
        tc.count(f"entry {i}")
    assert len(tc._cache) == 2
    # oldest entries were evicted: re-counting them hits the network again
    assert "entry 4" in tc._cache and "entry 3" in tc._cache


def test_fallback_on_non_200(tok_server):
    srv, handler = tok_server
    handler.tokenize_status = 500
    tc = TokenCounter(srv.base_url)
    text = "x" * 100
    assert tc.count(text) == len(text) // 4


def test_fallback_on_connection_refused():
    tc = TokenCounter(DEAD_URL)
    text = "no server here at all"
    assert tc.count(text) == len(text) // 4
    assert tc.exact_available  # single failure: circuit still closed


def test_degraded_count_messages_equals_estimate_tokens(tok_server):
    srv, handler = tok_server
    handler.tokenize_status = 404
    tc = TokenCounter(srv.base_url, failure_threshold=100)  # keep probing
    msgs = [
        _msg("system", "S"),
        _msg("user", "question " * 30),
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "read_file",
                                      "arguments": '{"path": "a.txt"}'}}]},
        _msg("tool", "result text " * 40),
        _msg("assistant", None),  # content None, no tool_calls -> 0 both ways
    ]
    assert tc.count_messages(msgs) == estimate_tokens(msgs)


def test_circuit_breaker_stops_probing_and_recovers(tok_server, caplog):
    srv, handler = tok_server
    now = [1000.0]
    handler.tokenize_status = 500
    with caplog.at_level(logging.WARNING, logger="sable.tokenizer"):
        tc = TokenCounter(srv.base_url, failure_threshold=2,
                          cooldown_seconds=10.0, clock=lambda: now[0])
        text = "probe text"
        assert tc.count(text) == len(text) // 4  # failure 1
        assert tc.count("other") == 5 // 4       # failure 2 -> circuit opens
        assert len(handler.tokenize_requests) == 2
        assert not tc.exact_available

        # Circuit open: no further probing at all, silent chars/4 fallback.
        assert tc.count(text) == len(text) // 4
        assert tc.count("a longer fallback string") == 26 // 4
        assert len(handler.tokenize_requests) == 2  # nothing left the process
        warnings = [r for r in caplog.records
                    if "tokenize endpoint" in r.getMessage()]
        assert len(warnings) == 1  # logged once, not per call

    # Recovery: endpoint heals, cooldown passes -> half-open probe succeeds.
    handler.tokenize_status = 200
    now[0] += 11.0
    assert tc.count(text) == fake_tokenize_count(text)
    assert tc.exact_available
    assert len(handler.tokenize_requests) == 3


def test_base_url_strips_v1_suffix():
    tc = TokenCounter("http://example.test:8080/v1/")
    assert tc.tokenize_url == "http://example.test:8080/tokenize"


# ------------------------------------------------- compress_messages gate

class _FakeCounter:
    """Duck-typed counter with a fixed chars/2 policy for boundary proofs."""

    def __init__(self):
        self.calls = 0

    def count_messages(self, messages):
        self.calls += 1
        total = 0
        for m in messages:
            content = m.get("content")
            if isinstance(content, str):
                total += -(-len(content) // 2) + 4
            elif content is not None:
                total += 4
            for _tc in m.get("tool_calls") or []:
                total += 4
        return total


def test_gate_differs_from_chars4_with_exact_counter():
    # Content sized so chars/4 says "under budget" while the exact counter
    # (2 chars/token) says "over": same messages, opposite gate outcomes.
    msgs = [_msg("system", "S")]
    for i in range(6):
        msgs.append(_msg("user", f"q{i} " + "u" * 100))
        msgs.append(_msg("assistant", f"a{i} " + "v" * 100))
    est = estimate_tokens(msgs)
    fake = _FakeCounter()
    exact = fake.count_messages(msgs)
    assert est < exact  # chars/4 underestimates this content
    threshold = (est + exact) // 2

    out_plain = compress_messages(msgs, None, threshold_tokens=threshold,
                                  keep_recent_tokens=64)
    assert out_plain is msgs  # estimate path: no compression

    out_exact = compress_messages(msgs, None, threshold_tokens=threshold,
                                  keep_recent_tokens=64, counter=fake)
    assert out_exact is not msgs
    assert any(str(m.get("content", "")).startswith(SUMMARY_PREFIX)
               for m in out_exact)
    assert fake.calls >= 1


def test_cjk_boundary_compression_with_real_server(tok_server):
    # CJK-heavy history: llama-style tokenizers are ~1 char/token here, our
    # fake uses ceil(len/2), and chars/4 badly underestimates — the exact
    # counter triggers compression where the estimate never would.
    srv, handler = tok_server
    msgs = [_msg("system", "S")]
    for i in range(6):
        msgs.append(_msg("user", "汉" * 200))
        msgs.append(_msg("assistant", "字" * 200))
    est = estimate_tokens(msgs)
    tc = TokenCounter(srv.base_url)
    exact = tc.count_messages(msgs)
    assert est < exact
    threshold = (est + exact) // 2

    out_plain = compress_messages(msgs, None, threshold_tokens=threshold,
                                  keep_recent_tokens=64)
    assert out_plain is msgs

    out = compress_messages(msgs, None, threshold_tokens=threshold,
                            keep_recent_tokens=64, counter=tc)
    assert out is not msgs
    assert any(str(m.get("content", "")).startswith(SUMMARY_PREFIX)
               for m in out)
    # every distinct content string was tokenized exactly once (LRU dedupes
    # the six identical users / six identical assistants into one probe each)
    assert len(set(handler.tokenize_requests)) == len(handler.tokenize_requests)


# ------------------------------------------------------------ runner wiring

def test_runner_uses_injected_counter_for_compression_gate(tok_server):
    srv, handler = tok_server
    msgs = [_msg("system", "S")]
    for i in range(6):
        msgs.append(_msg("user", f"question {i} " + "u" * 100))
        msgs.append(_msg("assistant", f"answer {i} " + "v" * 100))
    est = estimate_tokens(msgs)
    exact = TokenCounter(srv.base_url).count_messages(msgs)
    threshold = (est + exact) // 2

    tc = TokenCounter(srv.base_url)
    client = SableAgentClient(base_url=srv.base_url, model="tok-model",
                                compress_threshold_tokens=threshold,
                                keep_recent_tokens=64, token_counter=tc)
    out, _summary = client._maybe_compress(msgs)
    assert out is not msgs  # exact count crossed the gate chars/4 did not
    assert any(str(m.get("content", "")).startswith(SUMMARY_PREFIX)
               for m in out)
    assert handler.tokenize_requests  # the endpoint actually answered


def test_runner_compression_off_never_probes_tokenize(tok_server):
    srv, handler = tok_server
    client = SableAgentClient(base_url=srv.base_url, model="tok-model",
                                compress_threshold_tokens=0)
    out, _ = client._maybe_compress([_msg("user", "x" * 10_000)])
    assert out and handler.tokenize_requests == []


def test_runner_lazy_counter_binds_to_endpoint_and_resets_on_swap(tok_server):
    srv, _ = tok_server
    client = SableAgentClient(base_url=srv.base_url, model="tok-model",
                                compress_threshold_tokens=100)
    assert client._token_counter is None  # lazy: nothing built at init
    tc = client._tokens()
    assert isinstance(tc, TokenCounter)
    assert tc.tokenize_url == srv.base_url.replace("/v1", "") + "/tokenize"
    assert client._token_counter is tc  # memoized

    ok = client.switch_model("http://127.0.0.1:4321/v1", "other-model")
    assert ok and client._token_counter is None  # re-binds to the new endpoint

    injected = TokenCounter(DEAD_URL)
    client2 = SableAgentClient(base_url=srv.base_url, model="tok-model",
                                 token_counter=injected)
    client2.switch_model("http://127.0.0.1:4321/v1", "other")
    assert client2._token_counter is injected  # test doubles survive swaps


def test_full_run_against_tokenizing_server(tok_server):
    # End-to-end smoke: the loop's compression gate lazily builds a real
    # TokenCounter against the fake server (/tokenize at root, chat under
    # /v1) and the run completes normally.
    srv, _ = tok_server
    client = SableAgentClient(base_url=srv.base_url, model="tok-model",
                                compress_threshold_tokens=50,
                                max_iterations=3)
    result = client.run("hello there")
    assert result.completed
    assert result.final_response == "final answer"

"""Backlog #10 tests — tool-result spill instead of hard truncation.

Covers: per-result spill (atomic write, sanitized filename, stub shape with
byte count + preview, redaction on disk), trace keeps the stub not the full
output, per-turn aggregate budget (largest first, idempotent, disabled gate),
retention prune with the once-per-process sweep flag, ordering (guardrails
see real content before spilling), read_file retrieval under default
containment, and the compression prune synergy for spill stubs.
"""

from __future__ import annotations

import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fake_server import (  # noqa: E402
    FakeOpenAIServer,
    make_tokenize_handler,
)
from sable.core.agent_runner import SableAgentClient, create_runner  # noqa: E402
from sable.core.compression import prune_old_tool_results  # noqa: E402
from sable.core.config import SableConfig  # noqa: E402
from sable.core.tokenizer import TokenCounter  # noqa: E402
from sable.core.spill import (  # noqa: E402
    PERSISTED_OUTPUT_PREFIX,
    enforce_turn_budget,
    is_spillover_read,
    maybe_spill_result,
    prune_spillover,
    reset_sweep_flag_for_tests,
    spill_dir,
)


# ------------------------------------------------- spillover re-read exemption

def test_is_spillover_read(home):
    d = spill_dir(home)
    d.mkdir(parents=True, exist_ok=True)
    f = d / "abc.txt"
    f.write_text("x")
    assert is_spillover_read("read_file", {"path": str(f)}, home) is True
    # other tools / other paths / missing files are NOT exempt
    assert is_spillover_read("list_dir", {"path": str(d)}, home) is False
    assert is_spillover_read("read_file", {"path": "/etc/passwd"}, home) is False
    assert is_spillover_read("read_file", {"path": str(d / "nope.txt")},
                             home) is False
    assert is_spillover_read("read_file", {}, home) is False


def test_turn_budget_exempt_index_survives(home):
    """Exempted (spillover re-read) messages are never spilled by level 3."""
    big = "y" * 9000
    msgs = [{"role": "tool", "tool_call_id": "a", "name": "read_file",
             "content": big},
            {"role": "tool", "tool_call_id": "b", "name": "list_dir",
             "content": "z" * 9000}]
    enforce_turn_budget(msgs, 10_000, home, exempt={0})
    assert msgs[0]["content"] == big  # exempt: untouched
    assert msgs[1]["content"].startswith(PERSISTED_OUTPUT_PREFIX)


@pytest.fixture(autouse=True)
def _fresh_sweep_flag():
    """The once-per-process prune sweep must be re-armable per test."""
    reset_sweep_flag_for_tests()
    yield
    reset_sweep_flag_for_tests()


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "uf-home"
    monkeypatch.setenv("SABLE_HOME", str(h))
    return h


# ------------------------------------------------------------ per-result spill

def test_large_result_spilled_stub_shape_and_file(home):
    content = "line of output\n" * 2000  # ~28 KB, distinct tail
    stub = maybe_spill_result("search_files", "call_42", content, home,
                              threshold=8000)
    assert stub is not None
    assert stub.startswith(PERSISTED_OUTPUT_PREFIX)
    assert stub.endswith("</persisted-output>")
    path = spill_dir(home) / "call_42.txt"
    assert f"path={path}" in stub and f"bytes={len(content)}" in stub
    assert content[:500] in stub  # preview (first ~500 chars)
    assert "read_file the path to retrieve" in stub
    # full output on disk, byte-identical (no secret -> redaction is a no-op)
    assert path.read_text() == content
    # atomic write leaves no temp files behind
    assert not list(spill_dir(home).glob("*.tmp"))


def test_small_result_not_spilled(home):
    assert maybe_spill_result("read_file", "c1", "tiny", home,
                              threshold=8000) is None
    assert not spill_dir(home).exists() or not list(spill_dir(home).iterdir())


def test_stub_never_double_spilled(home):
    content = "x" * 9000
    stub = maybe_spill_result("t", "c1", content, home, threshold=100)
    assert stub is not None
    # feeding the stub back in is a no-op (idempotency guard on the prefix)
    assert maybe_spill_result("t", "c1", stub, home, threshold=1) is None


def test_tool_call_id_sanitized_into_safe_filename(home):
    content = "y" * 9000
    stub = maybe_spill_result("t", "../../evil/:*\x00id", content, home,
                              threshold=100)
    assert stub is not None
    files = list(spill_dir(home).iterdir())
    assert len(files) == 1
    name = files[0].name
    assert "/" not in name and ".." not in name and "\x00" not in name
    assert name.endswith(".txt")
    # escaped id gets a digest suffix so distinct ids never collide
    stub2 = maybe_spill_result("t", "..evil.id", content, home, threshold=100)
    assert stub2 is not None
    assert len(list(spill_dir(home).iterdir())) == 2


def test_spill_file_written_redacted(home):
    secret = "sk-abcdefghijklmnopqrstuvwx"
    content = ("report header\n" + secret + "\n" + "z" * 9000)
    stub = maybe_spill_result("t", "c1", content, home, threshold=8000)
    assert stub is not None
    disk = (spill_dir(home) / "c1.txt").read_text()
    assert secret not in disk          # durable artifact must be scrubbed
    assert "[REDACTED" in disk
    # the in-memory stub keeps raw preview; recorder redacts at record time
    assert secret in stub


# --------------------------------------------------------- turn aggregate budget

def _tool_msg(i: int, size: int) -> dict:
    return {"role": "tool", "tool_call_id": f"c{i}", "name": "t",
            "content": str(i) * size}


def test_turn_budget_spills_largest_first_until_under(home):
    # three medium results, each under the 8000 per-result threshold,
    # aggregate 23400 over a 15000 budget.
    msgs = [_tool_msg(1, 7900), _tool_msg(2, 7800), _tool_msg(3, 7700)]
    enforce_turn_budget(msgs, 15000, home)
    # spill order is strictly largest-first: c1 (7900) then c2 (7800); after
    # those two stubs the aggregate fits, so c3 stays inline.
    assert msgs[0]["content"].startswith(PERSISTED_OUTPUT_PREFIX)
    assert msgs[1]["content"].startswith(PERSISTED_OUTPUT_PREFIX)
    assert not msgs[2]["content"].startswith(PERSISTED_OUTPUT_PREFIX)
    total = sum(len(m["content"]) for m in msgs)
    assert total <= 15000
    # spilled files exist on disk with the full originals
    assert (spill_dir(home) / "c1.txt").read_text() == "1" * 7900
    assert (spill_dir(home) / "c2.txt").read_text() == "2" * 7800


def test_turn_budget_idempotent_second_pass(home):
    msgs = [_tool_msg(1, 5000), _tool_msg(2, 9000), _tool_msg(3, 7000)]
    enforce_turn_budget(msgs, 15000, home)
    after_first = [m["content"] for m in msgs]
    files_after_first = sorted(p.name for p in spill_dir(home).iterdir())
    enforce_turn_budget(msgs, 15000, home)  # no-op: under budget + stub-skipping
    assert [m["content"] for m in msgs] == after_first
    assert sorted(p.name for p in spill_dir(home).iterdir()) == files_after_first


def test_turn_budget_under_budget_untouched(home):
    msgs = [_tool_msg(1, 100), _tool_msg(2, 200)]
    enforce_turn_budget(msgs, 15000, home)
    assert [len(m["content"]) for m in msgs] == [100, 200]


# ------------------------------------------------------------------- retention

def test_retention_prune_and_once_per_process_flag(home):
    old_stub = maybe_spill_result("t", "old", "a" * 9000, home, threshold=100)
    assert old_stub is not None
    old_file = spill_dir(home) / "old.txt"
    # backdate beyond the retention window
    stale = time.time() - 8 * 86400
    os.utime(old_file, (stale, stale))
    # sweep already ran once for this process -> old file survives a new spill
    assert maybe_spill_result("t", "fresh", "b" * 9000, home, threshold=100)
    assert old_file.exists()
    # re-arming the flag lets the next spill prune it
    reset_sweep_flag_for_tests()
    assert maybe_spill_result("t", "fresh2", "c" * 9000, home, threshold=100)
    assert not old_file.exists()
    assert (spill_dir(home) / "fresh.txt").exists()
    assert (spill_dir(home) / "fresh2.txt").exists()


def test_prune_direct_call_keeps_fresh(home):
    maybe_spill_result("t", "keep", "d" * 9000, home, threshold=100)
    assert prune_spillover(home, retention_days=7) == 0
    assert (spill_dir(home) / "keep.txt").exists()


# ---------------------------------------------------------- runner integration

class OneBigResultThenAnswerHandler(BaseHTTPRequestHandler):
    """First turn: one big read_file-style result. Second turn: final answer."""

    def log_message(self, format, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        messages = payload.get("messages", [])
        if any(m.get("role") == "tool" for m in messages):
            msg = {"role": "assistant", "content": "Done."}
            finish = "stop"
        else:
            msg = {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_big", "type": "function",
                "function": {"name": "dump", "arguments": "{}"}}]}
            finish = "tool_calls"
        body = json.dumps({
            "id": "x", "object": "chat.completion",
            "model": payload.get("model", "m"),
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                      "total_tokens": 2}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


BIG_MARKER = "SABLE_BIG_OUTPUT_MARKER"


TAIL_MARKER = "SABLE_TAIL_BEYOND_PREVIEW"


def big_dump_impl(**kwargs) -> str:
    # marker at the head (lands in the stub preview) and one past char 500
    # (must NOT appear anywhere in the stub/trace).
    return (BIG_MARKER + "\n" + ("payload line\n" * 2000) + TAIL_MARKER)


def _trace_lines(home: Path, trace_id: str) -> list:
    matches = list((home / "buffer").rglob(f"{trace_id}.jsonl"))
    assert len(matches) == 1
    return [json.loads(l) for l in matches[0].read_text().splitlines()]


def test_runner_spills_before_record_trace_keeps_stub(home):
    from sable.store.recorder import TraceRecorder
    rec = TraceRecorder(home=home)
    with FakeOpenAIServer(handler_cls=OneBigResultThenAnswerHandler) as srv:
        client = SableAgentClient(base_url=srv.base_url, model="m",
                                    tool_impls={"dump": big_dump_impl},
                                    recorder=rec, spill_threshold_chars=8000)
        result = client.run("dump something big")
    assert result.completed is True
    tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    stub = tool_msgs[0]["content"]
    assert stub.startswith(PERSISTED_OUTPUT_PREFIX)
    # spill file exists under SABLE_HOME with the full original
    spill_file = spill_dir(home) / "call_big.txt"
    assert spill_file.exists()
    assert BIG_MARKER in spill_file.read_text()
    # trace JSONL carries the stub, never the full output: the head marker
    # is legitimately inside the 500-char preview; anything past it (the
    # tail marker) must be absent from the trace.
    assert result.trace_id is not None
    text = json.dumps(_trace_lines(home, result.trace_id))
    assert PERSISTED_OUTPUT_PREFIX in text
    assert TAIL_MARKER not in text
    # preview carries ~36 payload lines; the full output would carry 2000
    assert text.count("payload line") < 100


def test_runner_spill_disabled_keeps_content_inline(home):
    with FakeOpenAIServer(handler_cls=OneBigResultThenAnswerHandler) as srv:
        client = SableAgentClient(base_url=srv.base_url, model="m",
                                    tool_impls={"dump": big_dump_impl},
                                    spill_enabled=False)
        result = client.run("dump something big")
    tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
    assert BIG_MARKER in tool_msgs[0]["content"]
    assert not tool_msgs[0]["content"].startswith(PERSISTED_OUTPUT_PREFIX)


class IdenticalFailingPatchHandler(BaseHTTPRequestHandler):
    """Re-issues the byte-identical failing patch call forever."""

    def log_message(self, format, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        msg = {"role": "assistant", "content": None, "tool_calls": [{
            "id": "c", "type": "function",
            "function": {"name": "patch", "arguments": json.dumps(
                {"path": "missing.py", "old_string": "x",
                 "new_string": "y"})}}]}
        body = json.dumps({
            "id": "x", "object": "chat.completion",
            "model": payload.get("model", "m"),
            "choices": [{"index": 0, "message": msg,
                         "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                      "total_tokens": 2}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def failing_patch_impl(**kwargs) -> str:
    # > spill threshold below, so every executed result gets stubbed — yet
    # the guardrail ladder must still fire (it saw the real failure text).
    return "error: patch failed: file not found and could not be located"


def test_guardrails_see_real_content_despite_tiny_spill_threshold():
    with FakeOpenAIServer(handler_cls=IdenticalFailingPatchHandler) as srv:
        client = SableAgentClient(base_url=srv.base_url, model="m",
                                    tool_impls={"patch": failing_patch_impl},
                                    max_iterations=8,
                                    spill_threshold_chars=10)
        result = client.run("fix the file")
    tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
    # executed results are stubs in history...
    assert any(m["content"].startswith(PERSISTED_OUTPUT_PREFIX)
               for m in tool_msgs)
    # ...but the failure ladder still warned and blocked on the REAL text
    assert any("[Tool loop warning: repeated_exact_failure_warning" in m["content"]
               for m in tool_msgs)
    assert any(m["content"].startswith("error: blocked by guardrail "
                                       "(repeated_exact_failure_block)")
               for m in tool_msgs)
    assert result.failure_reason == "max_iterations"


# ------------------------------------------------- containment + read_file loop

def test_spilled_file_readable_under_default_containment(home, tmp_path):
    """The agent recovers a spill via read_file: spillover sits under
    SABLE_HOME, which resolve_workspace_roots allow-lists (batch #4)."""
    from sable.core.path_security import contain
    from sable.core.tools import TOOL_IMPLS

    cfg = SableConfig()
    roots = create_runner(cfg, base_url="http://127.0.0.1:9/v1",
                          model="m").workspace_roots
    assert roots is not None  # containment on (cwd default root)
    assert any(Path(r) == home for r in roots)  # SABLE_HOME allow-listed

    content = "recoverable payload\n" * 1000
    stub = maybe_spill_result("t", "call_recover", content, home, threshold=100)
    path = spill_dir(home) / "call_recover.txt"
    contained_read = contain(roots)(TOOL_IMPLS["read_file"])
    out = contained_read(path=str(path))
    # read_file prefixes line numbers; the spilled content must come through
    assert "error: blocked" not in out
    assert "1 | recoverable payload" in out


# ------------------------------------------------------- compression synergy

def test_prune_collapses_spill_stub_to_one_line(home):
    stub = maybe_spill_result("t", "c9", "q" * 9000, home, threshold=100)
    msg = {"role": "tool", "tool_call_id": "c9", "name": "t", "content": stub}
    pruned, n = prune_old_tool_results([msg], 1)
    assert n == 1
    assert pruned[0]["content"] == (
        f"[pruned persisted output: {spill_dir(home) / 'c9.txt'}]")
    # idempotent: the one-line form is never re-pruned
    pruned2, n2 = prune_old_tool_results(pruned, 1)
    assert n2 == 0 and pruned2[0]["content"] == pruned[0]["content"]


# --------------------------------------------------------------- config wiring

def test_create_runner_reads_tools_spill_knobs(home):
    cfg = SableConfig(tools={"spill_enabled": False,
                               "spill_threshold_chars": 1234,
                               "turn_result_budget_chars": 5678,
                               "spill_retention_days": 3})
    runner = create_runner(cfg, base_url="http://127.0.0.1:9/v1", model="m")
    assert runner.spill_enabled is False
    assert runner.spill_threshold_chars == 1234
    assert runner.turn_result_budget_chars == 5678
    assert runner.spill_retention_days == 3
    assert runner._spill_home is not None
    assert Path(runner._spill_home) == home

    default = create_runner(SableConfig(), base_url="http://127.0.0.1:9/v1",
                            model="m")
    assert default.spill_enabled is True
    # Defaults track the constants in spill.py (raised over time).
    from sable.core.spill import (
        DEFAULT_SPILL_THRESHOLD_CHARS,
        DEFAULT_TURN_RESULT_BUDGET_CHARS,
    )

    assert default.spill_threshold_chars == DEFAULT_SPILL_THRESHOLD_CHARS
    assert default.turn_result_budget_chars == DEFAULT_TURN_RESULT_BUDGET_CHARS
    assert default.spill_retention_days == 7
    # token gate is opt-in: unset knob leaves the char gate (None sentinel)
    assert default.spill_threshold_tokens is None


def test_create_runner_reads_spill_threshold_tokens(home):
    cfg = SableConfig(tools={"spill_threshold_tokens": 999})
    runner = create_runner(cfg, base_url="http://127.0.0.1:9/v1", model="m")
    assert runner.spill_threshold_tokens == 999


# ------------------------------------------------------- token-accurate gate
# Deferred item from the exact-token build (improvement #4): spill_threshold_-
# tokens drives the per-result gate via exact /tokenize counts; everything
# above (char-gate tests, untouched) proves the default-off behavior.

class _FakeCounter:
    """Duck-typed counter with a fixed ceil(len/2) policy (fake-server math)."""

    def __init__(self, exact_available: bool = True):
        self.calls = 0
        self.exact_available = exact_available

    def count(self, text: str) -> int:
        self.calls += 1
        return -(-len(text) // 2)


def test_token_gate_spills_cjk_that_passes_chars_gate(home):
    # CJK content the char gate waves through (4000 chars < 8000 threshold)
    # but the exact token gate catches — real TokenCounter against the fake
    # /tokenize server, mirroring test_tokenizer_exact's CJK boundary style.
    handler = make_tokenize_handler({"*": "unused"})
    with FakeOpenAIServer(handler_cls=handler) as srv:
        handler.tokenize_requests.clear()
        tc = TokenCounter(srv.base_url)
        content = "汉" * 4000
        exact = tc.count(content)
        assert len(content) <= 8000  # char gate would NOT spill this
        assert exact == -(-4000 // 2)  # fake tokenizer: ceil(len/2) = 2000

        stub = maybe_spill_result("t", "cjk1", content, home, threshold=8000,
                                  threshold_tokens=1500, counter=tc)
        assert stub is not None
        assert stub.startswith(PERSISTED_OUTPUT_PREFIX)
        assert "gate=tokens" in stub  # provenance: token gate fired
        assert (spill_dir(home) / "cjk1.txt").read_text() == content


def test_chars_gate_stub_has_no_gate_marker(home):
    stub = maybe_spill_result("t", "cg", "x" * 9000, home, threshold=100)
    assert stub is not None and "gate=" not in stub


def test_threshold_tokens_unset_never_counts(home):
    fake = _FakeCounter()
    # threshold_tokens unset -> char comparison only, counter never consulted.
    assert maybe_spill_result("t", "a", "x" * 100, home, threshold=8000,
                              counter=fake) is None
    assert fake.calls == 0
    stub = maybe_spill_result("t", "b", "x" * 9000, home, threshold=8000,
                              counter=fake)
    assert stub is not None and fake.calls == 0


def test_both_knobs_set_tokens_win(home):
    fake = _FakeCounter()  # ceil(len/2): 4000 chars -> 2000 tokens
    content = "y" * 4000
    # over the token gate, under the char gate -> spilled (tokens win)
    stub = maybe_spill_result("t", "w1", content, home, threshold=8000,
                              threshold_tokens=1500, counter=fake)
    assert stub is not None and "gate=tokens" in stub
    # under the token gate, over the char gate -> NOT spilled (tokens win)
    assert maybe_spill_result("t", "w2", content, home, threshold=100,
                              threshold_tokens=3000, counter=fake) is None


def test_open_circuit_falls_back_to_chars_gate(home):
    # Counter degraded (circuit open): no probing, exact chars gate decides.
    fake = _FakeCounter(exact_available=False)
    content = "汉" * 9000  # 9000 chars > 8000 char threshold
    stub = maybe_spill_result("t", "d1", content, home, threshold=8000,
                              threshold_tokens=10_000, counter=fake)
    assert stub is not None and "gate=" not in stub  # chars gate fired
    assert fake.calls == 0  # degraded: never counted


def test_tokenizing_endpoint_404_degrades_without_crash(home):
    handler = make_tokenize_handler({"*": "unused"})
    with FakeOpenAIServer(handler_cls=handler) as srv:
        handler.tokenize_requests.clear()
        handler.tokenize_status = 404
        tc = TokenCounter(srv.base_url, failure_threshold=100)  # keep probing
        content = "汉" * 4000
        # exact would say 2000 > 1500 (spill); degraded chars/4 says 1000
        # < 1500 -> estimate-driven no-spill, and crucially no exception.
        assert maybe_spill_result("t", "d2", content, home, threshold=8000,
                                  threshold_tokens=1500, counter=tc) is None


class TokenizingDumpHandler(BaseHTTPRequestHandler):
    """OneBigResultThenAnswer + a POST /tokenize (fake ceil(len/2)) endpoint."""

    tokenize_requests: list = []
    tokenize_status = 200

    def log_message(self, format, *args):
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if self.path.rstrip("/") == "/tokenize":
            content = payload.get("content")
            if isinstance(content, str):
                type(self).tokenize_requests.append(content)
            if type(self).tokenize_status != 200:
                self._send({"error": "tokenize unavailable"},
                           type(self).tokenize_status)
                return
            n = -(-len(content or "") // 2)
            self._send({"tokens": list(range(n)), "count": n})
            return
        messages = payload.get("messages", [])
        if any(m.get("role") == "tool" for m in messages):
            msg, finish = {"role": "assistant", "content": "Done."}, "stop"
        else:
            msg = {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_cjk", "type": "function",
                "function": {"name": "dump_cjk", "arguments": "{}"}}]}
            finish = "tool_calls"
        self._send({
            "id": "x", "object": "chat.completion",
            "model": payload.get("model", "m"),
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                      "total_tokens": 2}})


CJK_CONTENT = "汉" * 4000  # 4000 chars: passes an 8000-char gate, fails 1500 tok


def cjk_dump_impl(**kwargs) -> str:
    return CJK_CONTENT


def test_runner_token_gate_spills_cjk_one_http_count_per_output(home):
    # Agent-level proof: with spill_threshold_tokens set, a CJK tool output
    # that the char gate keeps inline is spilled on the exact token count —
    # and the output crosses /tokenize exactly once (decision time only).
    TokenizingDumpHandler.tokenize_requests = []
    TokenizingDumpHandler.tokenize_status = 200
    with FakeOpenAIServer(handler_cls=TokenizingDumpHandler) as srv:
        client = SableAgentClient(
            base_url=srv.base_url, model="m",
            tool_impls={"dump_cjk": cjk_dump_impl},
            spill_threshold_chars=8000, spill_threshold_tokens=1500,
            token_counter=TokenCounter(srv.base_url))
        result = client.run("dump the cjk thing")
    assert result.completed is True
    tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    stub = tool_msgs[0]["content"]
    assert stub.startswith(PERSISTED_OUTPUT_PREFIX)
    assert "gate=tokens" in stub
    assert (spill_dir(home) / "call_cjk.txt").read_text() == CJK_CONTENT
    # one HTTP count per tool output: the content was POSTed exactly once
    assert TokenizingDumpHandler.tokenize_requests == [CJK_CONTENT]


def test_runner_token_gate_degraded_run_survives_404(home):
    # /tokenize down mid-run: gate degrades (chars/4 estimate says 1000 <
    # 1500 -> inline), the run completes, nothing crashes.
    TokenizingDumpHandler.tokenize_requests = []
    TokenizingDumpHandler.tokenize_status = 404
    try:
        with FakeOpenAIServer(handler_cls=TokenizingDumpHandler) as srv:
            client = SableAgentClient(
                base_url=srv.base_url, model="m",
                tool_impls={"dump_cjk": cjk_dump_impl},
                spill_threshold_chars=8000, spill_threshold_tokens=1500,
                token_counter=TokenCounter(srv.base_url, failure_threshold=100))
            result = client.run("dump the cjk thing")
    finally:
        TokenizingDumpHandler.tokenize_status = 200
    assert result.completed is True
    tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
    assert tool_msgs[0]["content"] == CJK_CONTENT  # degraded estimate: inline
    assert TokenizingDumpHandler.tokenize_requests == [CJK_CONTENT]  # probed once

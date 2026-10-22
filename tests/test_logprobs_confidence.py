"""Pass 4 #40 — confidence calibration from llama.cpp logprobs (runner side).

Covers the record-only half of the idea: the runner requests content logprobs
behind the `sampling.logprobs` config knob, folds each run's per-token
logprobs into two floats on the trace row, and degrades to exactly today's
behavior when the backend has no logprob support.

All hermetic: local fake OpenAI servers only (stdlib http.server).
"""

from __future__ import annotations

import json
import math
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fake_server import FakeOpenAIServer  # noqa: E402
from sable.core.agent_runner import (  # noqa: E402
    SableAgentClient, _confidence_aggregate, create_runner,
)
from sable.core.config import SableConfig  # noqa: E402
from sable.store import TraceRecorder, TraceStore  # noqa: E402


# Known synthetic token distribution reused across assertions:
#   logprobs = [ln 0.5, ln 0.5, ln 0.8, ln 0.8, ln 0.8]
#   conf_mean            = (2·ln0.5 + 3·ln0.8) / 5          = -0.411145
#   conf_p90_negentropy  = exp(sorted_p90)                  = 0.8
CANNED_LPS = [math.log(0.5), math.log(0.5), math.log(0.8),
              math.log(0.8), math.log(0.8)]
EXPECT_MEAN = round(sum(CANNED_LPS) / len(CANNED_LPS), 6)


def _logprobs_payload(with_logprobs: bool):
    lp = {"content": [{"token": f"t{i}", "logprob": v, "top_logprobs": []}
                      for i, v in enumerate(CANNED_LPS)]} if with_logprobs \
        else None

    class Handler(BaseHTTPRequestHandler):
        requests: list = []

        def log_message(self, format, *args):  # silence
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
            type(self).requests.append(payload)
            choice: dict = {
                "index": 0,
                "message": {"role": "assistant", "content": "all good"},
                "finish_reason": "stop",
            }
            if lp is not None and payload.get("logprobs"):
                choice["logprobs"] = lp
            # A backend WITHOUT support answers unchanged even when asked.
            self._send({
                "id": "cmpl-lp", "object": "chat.completion",
                "model": payload.get("model", "lp-model"),
                "choices": [choice],
                "usage": {"prompt_tokens": 1, "completion_tokens": 5,
                          "total_tokens": 6},
            })

    return Handler


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "uf-logprobs"
    monkeypatch.setenv("SABLE_HOME", str(h))
    return h


# ------------------------------------------------------------- aggregate math

def test_confidence_aggregate_known_values():
    mean, p90 = _confidence_aggregate(list(CANNED_LPS))
    assert mean == pytest.approx(EXPECT_MEAN, abs=1e-6)
    # nearest-rank p90 of [l.5,l.5,l.8,l.8,l.8] -> index round(4.5)-1 = 3
    assert p90 == pytest.approx(0.8, abs=1e-6)


def test_confidence_aggregate_empty_is_none():
    assert _confidence_aggregate([]) == (None, None)


# --------------------------------------------------------------- runner path

def _run_with(home, handler_cls, logprobs: bool):
    server = FakeOpenAIServer(handler_cls).start()
    recorder = TraceRecorder(home=home)
    try:
        client = SableAgentClient(
            base_url=server.base_url, model="lp-model",
            tools=[], tool_impls={}, recorder=recorder,
            logprobs=logprobs)
        result = client.run("say hi")
        assert result.completed and result.final_response == "all good"
        assert result.trace_id is not None
        store = TraceStore(Path(home) / "buffer" / "sable.db")
        try:
            row = store.get_trace(result.trace_id)
        finally:
            store.close()
        assert row is not None
        return result, row
    finally:
        recorder.close()
        server.stop()


def test_logprobs_requested_and_aggregates_stored(home):
    handler = _logprobs_payload(with_logprobs=True)
    _, row = _run_with(home, handler, logprobs=True)
    assert row.conf_mean == pytest.approx(EXPECT_MEAN, abs=1e-6)
    assert row.conf_p90_negentropy == pytest.approx(0.8, abs=1e-6)


def test_backend_without_logprobs_degrades_to_none(home):
    # Backend ignores the flag and returns no logprobs field at all: run
    # succeeds exactly as before, aggregates stay NULL.
    handler = _logprobs_payload(with_logprobs=False)
    result, row = _run_with(home, handler, logprobs=True)
    assert result.completed is True
    assert row.conf_mean is None
    assert row.conf_p90_negentropy is None


def test_flag_off_requests_nothing_and_stores_none(home):
    handler = _logprobs_payload(with_logprobs=True)
    _, row = _run_with(home, handler, logprobs=False)
    assert row.conf_mean is None and row.conf_p90_negentropy is None
    # And the request body never carried the flag in the first place.
    assert handler.requests, "handler saw no request"
    assert all("logprobs" not in p for p in handler.requests)


# ------------------------------------------------------------ config plumbing

def test_create_runner_reads_sampling_logprobs_knob():
    cfg = SableConfig(sampling={"logprobs": True, "temperature": 0.5})
    runner = create_runner(cfg, base_url="http://127.0.0.1:1/v1",
                           model="m")
    assert runner.logprobs is True
    # The gate must NOT leak verbatim into the upstream sampling map (it would
    # otherwise be forwarded twice by request_sampling_payload).
    assert "logprobs" not in runner.sampling
    assert runner.sampling == {"temperature": 0.5}


def test_create_runner_default_off_and_absent_key_kept_out():
    runner = create_runner(SableConfig(sampling={"temperature": 0.3}),
                           base_url="http://127.0.0.1:1/v1", model="m")
    assert runner.logprobs is False

    runner2 = create_runner(SableConfig(), base_url="http://127.0.0.1:1/v1",
                            model="m")
    assert runner2.logprobs is False


def test_create_runner_logprobs_on_reaches_request_body():
    handler = _logprobs_payload(with_logprobs=True)
    server = FakeOpenAIServer(handler).start()
    try:
        cfg = SableConfig(sampling={"logprobs": True})
        runner = create_runner(cfg, base_url=server.base_url, model="m")
        result = runner.run("say hi")
        assert result.completed
        body = handler.requests[-1]
        assert body.get("logprobs") is True
    finally:
        server.stop()

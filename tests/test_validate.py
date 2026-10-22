"""Phase 4 tests — validation gate with a scripted fake server.

The golden-task suite runs against tests/fake_server.py's scripted handler
(no GPU, no llama-server). Pass-rate thresholds drive promote vs fail in the
registry; `sable iterate` consumes the same gate via an injected starter.
"""

from __future__ import annotations

import sys
import threading
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fake_server import FakeOpenAIServer, make_scripted_handler  # noqa: E402
from sable.registry import ModelRegistry  # noqa: E402
from sable.training.validate import (  # noqa: E402
    ValidationConfig,
    run_golden_suite,
    validate_model,
)

GOLDEN = [
    {"prompt": "what is 2+2?", "expect_contains": ["4"]},
    {"prompt": "say the sky's color", "expect_contains": ["blue"],
     "expect_not_contains": ["green"]},
]


def starter_for(script):
    """ServerStarter: launch scripted fake server for any gguf path."""

    def starter(gguf_path: str):
        srv = FakeOpenAIServer(handler_cls=make_scripted_handler(script))
        srv.start()
        # validate_model expects .base_url/.model_name and .stop()
        return srv

    return starter


def test_golden_suite_all_pass(tmp_path):
    # keys are substring-matched against the prompt:
    script = {"2+2": "the answer is 4", "color": "the sky is blue"}
    with FakeOpenAIServer(handler_cls=make_scripted_handler(script)) as srv:
        res = run_golden_suite(srv.base_url, "m", GOLDEN)
    assert res["pass_rate"] == 1.0 and res["passed"] == 2


def test_golden_suite_partial_fail():
    script = {"*": "4", "color": "the sky is green"}  # forbidden hit + missing
    with FakeOpenAIServer(handler_cls=make_scripted_handler(script)) as srv:
        res = run_golden_suite(srv.base_url, "m", GOLDEN)
    assert res["passed"] == 1 and res["pass_rate"] == 0.5


def test_validate_model_pass_promotes(tmp_path):
    reg = ModelRegistry(tmp_path / "r.db")
    m = reg.register_model("base-a")
    script = {"2+2": "4", "color": "blue skies"}
    vcfg = ValidationConfig(min_pass_rate=1.0, golden_tasks=GOLDEN)
    report = validate_model(
        str(tmp_path / "fake.gguf"), vcfg, server_starter=starter_for(script)
    )
    assert report["ok"] is True and report["pass_rate"] == 1.0
    reg.set_validation(m.model_id, report, passed=True)
    validated = reg.get_model(m.model_id)
    assert validated is not None and validated.status == "validated"
    reg.promote(m.model_id)
    active = reg.active_model()
    assert active is not None and active.model_id == m.model_id
    reg.close()


def test_validate_model_fail_blocks_promotion(tmp_path):
    reg = ModelRegistry(tmp_path / "r.db")
    m = reg.register_model("base-a")
    script = {"*": "nonsense"}
    vcfg = ValidationConfig(min_pass_rate=0.75, golden_tasks=GOLDEN)
    report = validate_model(
        str(tmp_path / "fake.gguf"), vcfg, server_starter=starter_for(script)
    )
    assert report["ok"] is False
    reg.set_validation(m.model_id, report, passed=False)
    got = reg.get_model(m.model_id)
    assert got is not None and got.status == "failed"
    with pytest.raises(Exception):
        reg.promote(m.model_id)
    reg.close()


def test_threshold_boundary():
    # 1/2 pass at min_pass_rate 0.5 -> ok; at 0.75 -> not ok
    script = {"2+2": "4", "color": "wrong"}
    with FakeOpenAIServer(handler_cls=make_scripted_handler(script)) as srv:
        res = run_golden_suite(srv.base_url, "m", GOLDEN)
    assert res["pass_rate"] == 0.5


def test_empty_golden_tasks_refuses_promotion(tmp_path):
    vcfg = ValidationConfig(min_pass_rate=0.0, golden_tasks=[])
    report = validate_model(
        str(tmp_path / "x.gguf"), vcfg,
        server_starter=starter_for({"*": "anything"}),
    )
    assert report["ok"] is False and "golden_tasks" in report["error"]


def test_server_failure_reported_not_raised(tmp_path):
    def broken_starter(gguf):
        raise RuntimeError("llama-server exploded")

    vcfg = ValidationConfig(min_pass_rate=1.0, golden_tasks=GOLDEN)
    report = validate_model(str(tmp_path / "x.gguf"), vcfg,
                            server_starter=broken_starter)
    assert report["ok"] is False and "exploded" in report["error"]


# ------------------------------------------------------- parallel replay (#housekeeping)

def _slow_handler(delay_s: float = 0.3, slow_key: str = "slow"):
    """Handler class sleeping ``delay_s`` on prompts containing ``slow_key``.

    Tracks concurrent in-flight requests so tests can assert real overlap
    (and that it stays bounded by the worker cap).
    """
    state = {"inflight": 0, "peak": 0}
    lock = threading.Lock()

    class SlowHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # silence
            pass

        def _send(self, obj, code=200):
            import json as _json
            body = _json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            import json as _json
            length = int(self.headers.get("Content-Length", 0))
            payload = _json.loads(self.rfile.read(length) or b"{}")
            last_user = ""
            for m in reversed(payload.get("messages", [])):
                if m.get("role") == "user" and isinstance(m.get("content"), str):
                    last_user = m["content"]
                    break
            with lock:
                state["inflight"] += 1
                state["peak"] = max(state["peak"], state["inflight"])
            try:
                if slow_key in last_user:
                    time.sleep(delay_s)
                self._send({
                    "id": "cmpl-slow", "object": "chat.completion",
                    "model": payload.get("model", "slow-model"),
                    "choices": [{"index": 0, "message": {
                        "role": "assistant", "content": f"ok {last_user}"},
                        "finish_reason": "stop"}],
                })
            finally:
                with lock:
                    state["inflight"] -= 1

    return SlowHandler, state


def _slow_tasks(n_fast: int = 5, n_slow: int = 1):
    tasks = [{"prompt": f"fast {i}", "expect_contains": ["ok"]}
             for i in range(n_fast)]
    tasks += [{"prompt": f"slow {i}", "expect_contains": ["ok"]}
              for i in range(n_slow)]
    return tasks


def test_parallel_replay_verdicts_match_serial():
    """Same fixture, same verdicts — parallelism must not change outcomes."""
    handler_cls, _ = _slow_handler(delay_s=0.0)
    tasks = _slow_tasks(n_fast=6, n_slow=2)
    with FakeOpenAIServer(handler_cls=handler_cls) as srv:
        serial = run_golden_suite(srv.base_url, "m", tasks, parallel=1)
        par = run_golden_suite(srv.base_url, "m", tasks, parallel=4)
    assert par["results"] == serial["results"]  # ordering + content identical
    assert par["passed"] == serial["passed"] == 8
    assert par["pass_rate"] == serial["pass_rate"] == 1.0


def test_parallel_replay_preserves_failures():
    handler_cls, _ = _slow_handler(delay_s=0.0)
    tasks = [
        {"prompt": "a", "expect_contains": ["ok"]},
        {"prompt": "b", "expect_contains": ["nope-missing"]},
        {"prompt": "c", "expect_not_contains": ["ok"]},
    ]
    with FakeOpenAIServer(handler_cls=handler_cls) as srv:
        res = run_golden_suite(srv.base_url, "m", tasks, parallel=3)
    assert res["passed"] == 1 and res["pass_rate"] == pytest.approx(1 / 3)
    # failure stays attached to the right (ordered) slot
    assert [r["pass"] for r in res["results"]] == [True, False, False]


def test_slow_task_does_not_block_others_beyond_worker_cap():
    """One slow task + many fast ones: wall time bounded by ceil(slow/workers)
    rounds, not the serial sum; overlap actually happens (peak > 1)."""
    delay = 0.35
    handler_cls, state = _slow_handler(delay_s=delay)
    tasks = _slow_tasks(n_fast=8, n_slow=4)  # 4 slow x 0.35 s = 1.4 s serial
    with FakeOpenAIServer(handler_cls=handler_cls) as srv:
        t0 = time.monotonic()
        res = run_golden_suite(srv.base_url, "m", tasks, parallel=4)
        dt = time.monotonic() - t0
    assert res["passed"] == 12
    # serial would cost >= 4*delay; with 4 workers expect ~delay plus slack.
    assert dt < 4 * delay, f"parallel replay took {dt:.2f}s, no pipelining?"
    assert state["peak"] > 1, "requests never overlapped"
    assert state["peak"] <= 4, f"concurrency {state['peak']} exceeded worker cap"


def test_parallel_smoke_n_workers_m_tasks():
    """Thread-safety smoke: 8 workers x 24 tasks, all verdicts correct."""
    handler_cls, _ = _slow_handler(delay_s=0.01)
    tasks = [{"prompt": f"t{i}", "expect_contains": ["ok"]} for i in range(24)]
    with FakeOpenAIServer(handler_cls=handler_cls) as srv:
        res = run_golden_suite(srv.base_url, "m", tasks, parallel=8)
    assert res["total"] == 24 and res["passed"] == 24
    assert [r["prompt"] for r in res["results"]] == [t["prompt"] for t in tasks]


def test_validate_model_uses_config_parallel(tmp_path):
    """vcfg.parallel flows through validate_model into the suite."""
    vcfg = ValidationConfig(min_pass_rate=1.0, golden_tasks=_slow_tasks(4, 2),
                            parallel=3)
    handler_cls, state = _slow_handler(delay_s=0.05)

    def starter(gguf_path):
        srv = FakeOpenAIServer(handler_cls=handler_cls)
        srv.start()
        return srv

    report = validate_model(str(tmp_path / "p.gguf"), vcfg, server_starter=starter)
    assert report["ok"] is True and report["passed"] == 6
    assert state["peak"] > 1  # the knob actually drove concurrent traffic


def test_validation_config_parallel_default_and_yaml():
    assert ValidationConfig().parallel == 2
    cfg = ValidationConfig.from_yaml({"validation": {"parallel": 5}})
    assert cfg.parallel == 5
    # absent key keeps the default
    assert ValidationConfig.from_yaml({"validation": {}}).parallel == 2

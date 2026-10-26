"""CLI-domain tests: smoke, exit-code taxonomy, golden validation,
capability tools, metric honesty, hardening gaps, dependency graph
reachability, model hub.
"""

from __future__ import annotations


# ------------------------------------------------------------------------
# from tests/test_smoke.py
# ------------------------------------------------------------------------

import sys
from pathlib import Path

import pytest


from fake_server import FakeOpenAIServer  # noqa: E402

from sable.core.agent_runner import (  # noqa: E402
    SableAgentClient,
)
from sable.core.config import SableConfig, load_config  # noqa: E402
from sable.serving.supervisor import (  # noqa: E402
    ServeError,
    ServingSupervisor,
    healthcheck,
)


@pytest.fixture(scope="module")
def fake_server():
    with FakeOpenAIServer() as srv:
        yield srv


# --------------------------------------------------------------------- config

def test_config_env_overrides_file(tmp_path, monkeypatch):
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text("port: 1234\ncontext_length: 32768\ngguf_path: /x/model.gguf\n")
    monkeypatch.setenv("SABLE_PORT", "9999")  # env wins over file
    monkeypatch.delenv("SABLE_GGUF", raising=False)
    cfg = load_config(str(cfg_file))
    assert cfg.port == 9999
    assert cfg.context_length == 32768
    assert cfg.gguf_path == "/x/model.gguf"
    assert cfg.host == "127.0.0.1"  # loopback enforced


def test_attach_url_normalization():
    cfg = SableConfig(attach_url="http://127.0.0.1:8888")
    assert cfg.effective_attach_url == "http://127.0.0.1:8888/v1"
    cfg2 = SableConfig(attach_url="http://127.0.0.1:8888/v1")
    assert cfg2.effective_attach_url == "http://127.0.0.1:8888/v1"


# ------------------------------------------------------------------- serving

def test_healthcheck(fake_server):
    assert healthcheck(fake_server.base_url) is True
    assert healthcheck("http://127.0.0.1:1/v1") is False


def test_supervisor_attach_no_spawn(fake_server):
    cfg = SableConfig(attach_url=fake_server.base_url)
    sup = ServingSupervisor(cfg)
    handle = sup.attach()
    try:
        assert handle.owned is False           # external server: never kill it
        assert handle.model_name == "fake-model"  # discovered via /v1/models
        assert handle.process is None
    finally:
        sup.stop()  # must be a no-op on the external server


def test_supervisor_spawn_requires_gguf():
    cfg = SableConfig(gguf_path=None)
    with pytest.raises(ServeError):
        ServingSupervisor(cfg).start(ready_timeout=1)


def test_supervisor_spawn_missing_gguf_file(tmp_path):
    cfg = SableConfig(gguf_path=str(tmp_path / "nope.gguf"))
    with pytest.raises(ServeError):
        ServingSupervisor(cfg).start(ready_timeout=1)


# --------------------------------------------------------------------- agent

def test_agent_tool_call_roundtrip(fake_server, tmp_path):
    """Agent loop reaches chat completion and completes a tool-call round trip."""
    client = SableAgentClient(base_url=fake_server.base_url, model="fake-model")
    result = client.run("read the fake server source")
    assert result.completed is True
    assert result.backend == "sable"
    # Sequence must include: assistant tool_call -> tool result -> final answer.
    roles = [m.get("role") for m in result.messages]
    assert "tool" in roles
    assert "Done." in result.final_response


def test_agent_connection_error_not_hang(monkeypatch):
    """Dead endpoint => fast classified failure, no hang (ROADMAP acceptance).

    Backlog #2: run() now retries transient faults with capped backoff and
    returns a completed=False result carrying failure_reason instead of
    raising; the retry sleeps are stubbed so the test stays quick.
    """
    monkeypatch.setattr(SableAgentClient, "_sleep", staticmethod(lambda s: None))
    client = SableAgentClient(
        base_url="http://127.0.0.1:9/v1", model="m", timeout=2.0
    )
    result = client.run("hello")
    assert result.completed is False
    assert result.failure_reason == "backend_error:connection"


# ------------------------------------------------------------- built-in runtime

def test_create_runner_returns_builtin(fake_server):
    """create_runner returns the built-in runtime with no external deps."""
    from sable.core.agent_runner import create_runner

    cfg = SableConfig()
    runner = create_runner(cfg, fake_server.base_url, "fake-model")
    assert runner.name == "sable"
    result = runner.run("read the fake server file")
    assert result.completed is True
    assert "Done." in result.final_response
    assert result.backend == "sable"


def test_profile_tool_subset_enforced():
    """enabled_tool_names restricts the exposed schemas/impls."""
    from sable.core.agent_runner import create_runner

    runner = create_runner(SableConfig(), "http://127.0.0.1:9/v1", "m",
                           enabled_tool_names=["list_dir"])
    names = {s["function"]["name"] for s in runner.tools}
    assert names == {"list_dir"}
    assert set(runner.tool_impls) == {"list_dir"}

# ------------------------------------------------------------------------
# from tests/test_exit_codes.py
# ------------------------------------------------------------------------

import json
import sys
from pathlib import Path

import pytest
import yaml


from sable.core.exitcodes import (  # noqa: E402
    EXIT_CONFIG_INVALID,
    EXIT_CORRUPT_ARTIFACT,
    EXIT_OK,
)
from sable.training.integrity import (  # noqa: E402
    check_gguf_file,
    export_integrity_problems,
    find_nan_losses,
)

# ---------------------------------------------------------------- unit: integrity

def test_check_gguf_ok(tmp_path):
    g = tmp_path / "m.gguf"
    g.write_bytes(b"GGUF" + b"\x00" * 32)
    assert check_gguf_file(g) is None


def test_check_gguf_missing_zero_and_bad_magic(tmp_path):
    missing = tmp_path / "nope.gguf"
    reason = check_gguf_file(missing)
    assert reason is not None and "missing" in reason

    zero = tmp_path / "zero.gguf"
    zero.write_bytes(b"")
    reason = check_gguf_file(zero)
    assert reason is not None and "zero-byte" in reason

    trunc = tmp_path / "trunc.gguf"
    trunc.write_bytes(b"GGU")          # truncated header
    reason = check_gguf_file(trunc)
    assert reason is not None and "magic" in reason

    wrong = tmp_path / "wrong.gguf"
    wrong.write_bytes(b"impostor bytes")
    reason = check_gguf_file(wrong)
    assert reason is not None and "magic" in reason


def test_find_nan_losses():
    assert find_nan_losses({"train_loss": 0.5, "eval_loss": None}) == []
    assert find_nan_losses({"train_loss": float("nan")}) == ["train_loss"]
    assert find_nan_losses({"train_loss": float("nan"),
                            "eval_loss": float("nan")}) == \
        ["eval_loss", "train_loss"]
    assert find_nan_losses({}) == []


def test_export_integrity_problems(tmp_path):
    good = tmp_path / "g.gguf"
    good.write_bytes(b"GGUFxxxxxxxx")
    assert export_integrity_problems(good) == []
    bad = tmp_path / "b.gguf"
    bad.write_bytes(b"GG")
    probs = export_integrity_problems(bad, {"train_loss": float("nan")})
    assert len(probs) == 2
    assert any("magic" in p for p in probs)
    assert any("NaN" in p for p in probs)


# ---------------------------------------------------------------- stub fixtures

_STUB_BASE = r'''
import argparse, json, os, sys
from pathlib import Path
ap = argparse.ArgumentParser()
for f in (("--job", True), ("--kind", True), ("--config", False),
          ("--dataset-dir", False), ("--base-model", False),
          ("--log-path", False), ("--adapter-dir", False),
          ("--gguf-out", False)):
    ap.add_argument(f[0], required=f[1], default=None)
ap.add_argument("--quant", default="q4_k_m")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

def emit(ev, **kw):
    line = json.dumps({"event": ev, "job": a.job, **kw})
    print(line, flush=True)
    if a.log_path:
        Path(a.log_path).parent.mkdir(parents=True, exist_ok=True)
        open(a.log_path, "a").write(line + "\n")

emit("start")
if a.kind in ("sft", "dpo"):
    out = Path(a.config).parent if a.config else Path(".")
    (out / "adapter").mkdir(parents=True, exist_ok=True)
    (out / "adapter" / "adapter_config.json").write_text(json.dumps(
        {"base_model": a.base_model or "stub", "r": 8}))
    # WP2b knob: emit NaN losses to simulate silently-corrupt numerics
    extra = {}
    if os.environ.get("STUB_NAN_LOSS") == "1":
        extra = {"train_loss": float("nan")}
    emit("sft_done", **extra)
elif a.kind == "export":
    gg = Path(a.gguf_out)
    gg.parent.mkdir(parents=True, exist_ok=True)
    mode = os.environ.get("STUB_EXPORT_MODE", "good")
    if mode == "good":
        gg.write_bytes(b"GGUF" + b"\x00" * 32)
    elif mode == "badmagic":
        gg.write_bytes(b"not a gguf at all")
    elif mode == "empty":
        gg.write_bytes(b"")
emit("done")
'''


@pytest.fixture()
def env(tmp_path, monkeypatch):
    home = tmp_path / "uf-exitcodes"
    monkeypatch.setenv("SABLE_HOME", str(home))
    stub = tmp_path / "stub.py"
    stub.write_text(_STUB_BASE)
    cfg_file = home.parent / "config.yaml"
    cfg_file.write_text(yaml.safe_dump({
        "home": str(home),
        "training": {"base_model": "Base/HF-Model",
                     "worker_command": [sys.executable, str(stub)]},
    }))
    return home, cfg_file


def _seed_dataset(home, ds_id: str):
    ds_dir = home / "datasets" / ds_id
    ds_dir.mkdir(parents=True)
    (ds_dir / "sft.jsonl").write_text(json.dumps(
        {"conversations": [{"from": "human", "value": "hi"}]}) + "\n")
    (ds_dir / "train.yaml").write_text(yaml.safe_dump({
        "model": "REPLACE_WITH_BASE_MODEL_PATH_OR_HF_ID",
        "data": {"local_dataset": [str(ds_dir / "sft.jsonl")],
                 "format_type": "sharegpt"},
    }))


def _iterate_args(cfg_file, tmp_path):
    from sable.cli import build_parser

    return build_parser().parse_args([
        "iterate", "--config", str(cfg_file), "--dry-run",
    ])


# ---------------------------------------------------------------- CLI: code 3

def test_invalid_config_exits_3_with_consolidated_report(env, capsys):
    """`sable.main()` on an invalid --config exits 3 and prints EVERY
    problem field to stderr in one listing."""
    from sable.cli import main

    bad = env[0].parent / "bad.yaml"
    bad.write_text(yaml.safe_dump({
        "port": "not-a-port",
        "context_length": {"nope": 1},
    }))
    rc = main(["traces", "list", "--config", str(bad)])
    assert rc == EXIT_CONFIG_INVALID
    err = capsys.readouterr().err
    assert "refusing to guess" in err
    assert "port" in err and "context_length" in err


def test_missing_config_file_exits_3(env, capsys):
    from sable.cli import main

    rc = main(["traces", "list",
               "--config", str(env[0] / "does-not-exist.yaml")])
    assert rc == EXIT_CONFIG_INVALID
    assert "Config file not found" in capsys.readouterr().err


def test_valid_config_still_exits_0(env, capsys):
    """Successful observable path unchanged: valid config -> exit 0."""
    from sable.cli import main

    _seed_dataset(env[0], "ds-ok")
    rc = main(["train", "start", "--config", str(env[1]),
               "--dataset", "ds-ok", "--dry-run"])
    assert rc == EXIT_OK


# ---------------------------------------------------------------- CLI: code 4

def test_train_start_nan_loss_exits_4(env, monkeypatch, capsys):
    """Worker exits 0 but records NaN train_loss -> exit 4 (corrupt)."""
    from sable.cli import build_parser, cmd_train_start

    _seed_dataset(env[0], "ds-nan")
    monkeypatch.setenv("STUB_NAN_LOSS", "1")
    args = build_parser().parse_args([
        "train", "start", "--config", str(env[1]), "--dataset", "ds-nan"])
    rc = cmd_train_start(args)
    assert rc == EXIT_CORRUPT_ARTIFACT
    assert "NaN" in capsys.readouterr().err


def test_train_start_clean_still_zero(env):
    from sable.cli import build_parser, cmd_train_start

    _seed_dataset(env[0], "ds-clean")
    args = build_parser().parse_args([
        "train", "start", "--config", str(env[1]), "--dataset", "ds-clean"])
    assert cmd_train_start(args) == EXIT_OK


def test_iterate_corrupt_export_exits_4(env, monkeypatch):
    """Full iterate loop: export stub writes a bad-magic GGUF -> the pipeline
    flags res.corrupt and cmd_iterate exits with code 4."""
    import os

    from sable.cli import build_parser, cmd_iterate

    _seed_dataset(env[0], "ds-bad")
    # seed traces so curate produces a dataset on its own is complex; reuse
    # the seeded one via --dataset and let train+export run against stubs.
    monkeypatch.setenv("STUB_EXPORT_MODE", "badmagic")
    try:
        args = build_parser().parse_args([
            "iterate", "--config", str(env[1]), "--dataset", "ds-bad"])
        rc = cmd_iterate(args)
        assert rc == EXIT_CORRUPT_ARTIFACT
    finally:
        os.environ.pop("STUB_EXPORT_MODE", None)


def test_iterate_empty_export_exits_4(env, monkeypatch):
    import os

    from sable.cli import build_parser, cmd_iterate

    _seed_dataset(env[0], "ds-empty")
    monkeypatch.setenv("STUB_EXPORT_MODE", "empty")
    try:
        args = build_parser().parse_args([
            "iterate", "--config", str(env[1]), "--dataset", "ds-empty"])
        assert cmd_iterate(args) == EXIT_CORRUPT_ARTIFACT
    finally:
        os.environ.pop("STUB_EXPORT_MODE", None)


def test_iterate_good_export_still_zero(env):
    """A sound GGUF keeps the historical success path (the validation stage
    may fail without a server starter — what matters is that we do NOT exit
    with the corrupt code)."""
    from sable.cli import build_parser, cmd_iterate

    _seed_dataset(env[0], "ds-good")
    args = build_parser().parse_args([
        "iterate", "--config", str(env[1]), "--dataset", "ds-good",
        "--dry-run"])
    assert cmd_iterate(args) == EXIT_OK


def test_taxonomy_no_collisions():
    """The finalized table: established codes untouched, new ones distinct."""
    from sable.core import exitcodes as ec

    vals = [ec.EXIT_OK, ec.EXIT_ERROR, ec.EXIT_USAGE, ec.EXIT_CONFIG_INVALID,
            ec.EXIT_CORRUPT_ARTIFACT, ec.EXIT_UPDATE_AVAILABLE,
            ec.EXIT_INTERRUPTED]
    assert len(set(vals)) == len(vals)
    assert (ec.EXIT_OK, ec.EXIT_ERROR, ec.EXIT_USAGE) == (0, 1, 2)
    # WP1 bench conventions stay coherent: bad compare operand = usage (2),
    # measurement error = generic (1) — asserted where bench maps them in cli.

# ------------------------------------------------------------------------
# from tests/test_validate.py
# ------------------------------------------------------------------------

import sys
import threading
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest


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

# ------------------------------------------------------------------------
# from tests/test_capabilities.py
# ------------------------------------------------------------------------

import sys
from pathlib import Path

import pytest


from sable.core.agent_runner import SableAgentClient, create_runner
from sable.core.config import SableConfig
from sable.core.tools import (
    CAPABILITY_TOOLS,
    resolve_capabilities,
    run_command,
    web_fetch,
)


# ------------------------------------------------------------- resolution

def test_capabilities_default_off():
    assert resolve_capabilities() == {
        "internet": False, "run_commands": False,
        "vision": False, "screenshots": False}


def test_capabilities_global_then_profile_merge():
    caps = resolve_capabilities({"capabilities": {"internet": True}},
                                {"internet": False, "run_commands": True})
    assert caps["internet"] is False and caps["run_commands"] is True


def test_capabilities_unknown_keys_ignored():
    caps = resolve_capabilities({"capabilities": {"teleport": True}},
                                {"launch_missiles": True, "internet": 1})
    assert caps["internet"] is True and caps["run_commands"] is False


# ------------------------------------------------------------- runner gate

def _tool_names(client: SableAgentClient):
    return {s.get("function", {}).get("name") for s in client.tools}


def test_default_runner_exposes_no_capability_tools():
    client = SableAgentClient(base_url="http://127.0.0.1:1/v1", model="m")
    names = _tool_names(client)
    assert "web_fetch" not in names and "run_command" not in names
    assert "read_file" in names  # sanity: the normal set is intact
    for tool in CAPABILITY_TOOLS:
        assert tool not in client.tool_impls


def test_capability_enabled_exposes_schema_and_impl():
    client = SableAgentClient(
        base_url="http://127.0.0.1:1/v1", model="m",
        capabilities={"internet": True, "run_commands": False})
    names = _tool_names(client)
    assert "web_fetch" in names
    assert "run_command" not in names
    assert "web_fetch" in client.tool_impls
    assert "run_command" not in client.tool_impls


def test_capability_tool_honors_profile_tool_subset():
    # Profile lists web_fetch but the capability is OFF => still not exposed.
    client = SableAgentClient(
        base_url="http://127.0.0.1:1/v1", model="m",
        enabled_tool_names=["read_file", "web_fetch"])
    assert "web_fetch" not in _tool_names(client)
    assert "web_fetch" not in client.tool_impls

    # Capability ON grants the tool even when the subset does not list it.
    client = SableAgentClient(
        base_url="http://127.0.0.1:1/v1", model="m",
        enabled_tool_names=["read_file"],
        capabilities={"internet": True})
    assert "web_fetch" in _tool_names(client)
    assert "web_fetch" in client.tool_impls
    # ...and unrelated tools the profile omitted stay gone.
    assert "write_file" not in _tool_names(client)


def test_create_runner_merges_global_and_profile_capabilities():
    cfg = SableConfig(agent={"capabilities": {"internet": True}})
    runner = create_runner(cfg, base_url="http://127.0.0.1:1/v1", model="m")
    assert "web_fetch" in _tool_names(runner)

    # Profile override turns the global default back off.
    runner = create_runner(cfg, base_url="http://127.0.0.1:1/v1", model="m",
                           capabilities={"internet": False})
    assert "web_fetch" not in _tool_names(runner)
    assert "web_fetch" not in runner.tool_impls


def test_create_runner_web_fetch_config_binds():
    cfg = SableConfig(tools={"web_fetch": {"allow_private_networks": True}})
    runner = create_runner(cfg, base_url="http://127.0.0.1:1/v1", model="m",
                           capabilities={"internet": True})
    impl = runner.tool_impls["web_fetch"]
    # Bound via functools.partial — the model cannot forge the knob.
    assert getattr(impl, "keywords", {}).get("allow_private_networks") is True


# ------------------------------------------------------------ web_fetch

def test_web_fetch_refuses_non_http_schemes():
    out = web_fetch("file:///etc/passwd")
    assert out.startswith("error: blocked:")
    assert "scheme" in out


def test_web_fetch_blocks_loopback_and_private_by_default():
    for url in ("http://127.0.0.1/x", "http://localhost/x",
                "http://169.254.169.254/latest/meta-data/",
                "http://10.0.0.5/x", "http://[::1]/x"):
        out = web_fetch(url)
        assert out.startswith("error: blocked:"), url


def test_web_fetch_allow_private_knob(tmp_path):
    # With the knob on, the address-space rule is relaxed (the fetch itself
    # then fails at connect time against a dead port — NOT as "blocked").
    out = web_fetch("http://127.0.0.1:9/", allow_private_networks=True)
    assert not out.startswith("error: blocked:")
    assert out.startswith("error: web_fetch failed")


# ------------------------------------------------------------ run_command

def test_run_command_captures_streams_and_exit_code():
    out = run_command("echo hello; echo oops >&2; exit 3")
    assert "exit code: 3" in out
    assert "hello" in out and "oops" in out


def test_run_command_timeout_kills_child():
    out = run_command("sleep 30", timeout_seconds=1)
    assert "timed out" in out


def test_run_command_rejects_empty():
    assert run_command("   ").startswith("error:")


# --------------------------------------------------- dashboard profiles API

def _app(home):
    from sable.dashboard.app import DashboardApp

    return DashboardApp(SableConfig(), home=home)


def test_profiles_api_validates_and_persists_capabilities(tmp_path):
    app = _app(str(tmp_path))
    r = app.handle("PUT", "/api/profiles/researcher", body={
        "system_prompt": "browse for me",
        "tools": ["read_file"],
        "capabilities": {"internet": True, "run_commands": False},
    })
    assert r.status < 400, r.body
    assert r.body["profile"]["capabilities"] == {"internet": True,
                                                 "run_commands": False}

    lst = app.handle("GET", "/api/profiles")
    profs = {p["name"]: p for p in lst.body["profiles"]}
    assert profs["researcher"]["capabilities"]["internet"] is True
    assert lst.body["capability_toggles"] == {
        "internet": "web_fetch", "run_commands": "run_command",
        "vision": "view_image", "screenshots": "take_screenshot"}
    assert lst.body["global_capabilities"] == {"internet": False,
                                               "run_commands": False,
                                               "vision": False,
                                               "screenshots": False}


def test_profiles_api_rejects_unknown_capability(tmp_path):
    app = _app(str(tmp_path))
    r = app.handle("PUT", "/api/profiles/evil", body={
        "capabilities": {"internet": True, "teleport": True}})
    assert r.status == 400
    assert "unknown capability" in r.body["error"]["message"]


def test_profiles_api_rejects_non_dict_capabilities(tmp_path):
    app = _app(str(tmp_path))
    r = app.handle("PUT", "/api/profiles/evil", body={"capabilities": ["a"]})
    assert r.status == 400


def test_profile_capabilities_survive_config_roundtrip(tmp_path):
    import yaml

    from sable.core.config import load_config

    app = _app(str(tmp_path))
    app.handle("PUT", "/api/profiles/webkit", body={
        "system_prompt": "s", "tools": [],
        "capabilities": {"internet": True}})
    cfg_file = Path(str(tmp_path)) / "sable.yaml"
    assert cfg_file.is_file()
    data = yaml.safe_load(cfg_file.read_text())
    prof = next(p for p in data["profiles"] if p["name"] == "webkit")
    assert prof["capabilities"] == {"internet": True}

    cfg = load_config(str(cfg_file))
    runner = create_runner(cfg, base_url="http://127.0.0.1:1/v1", model="m",
                           capabilities=prof["capabilities"])
    assert "web_fetch" in _tool_names(runner)
    assert "run_command" not in _tool_names(runner)


# --------------------------------------------------------------- delegation

def test_delegation_child_inherits_profile_capabilities(tmp_path, monkeypatch):
    """_run_child passes the CHILD profile's capabilities into create_runner."""
    from sable.core import delegation as deleg
    from sable.core.agent_runner import create_runner as real_create

    seen: dict = {}

    def spy(cfg, base_url, model, **kw):
        seen["capabilities"] = kw.get("capabilities")
        raise RuntimeError("stop-after-spy")

    monkeypatch.setattr("sable.core.agent_runner.create_runner", spy)
    cfg = SableConfig(profiles=[{
        "name": "surf", "tools": [], "capabilities": {"internet": True}}])
    with pytest.raises(RuntimeError, match="stop-after-spy"):
        deleg._run_child(cfg, cfg.profiles[0], "task", {"base_url": "u"},
                         "dlg1", {})
    assert seen["capabilities"] == {"internet": True}
    # sanity: real_create untouched for other tests
    assert real_create is not None

# ------------------------------------------------------------------------
# from tests/test_metric_honesty.py
# ------------------------------------------------------------------------

import json
import sys
from pathlib import Path

import pytest


from sable.core.config import SableConfig  # noqa: E402
from sable.dashboard.app import DashboardApp  # noqa: E402
from sable.dashboard.metrics import (  # noqa: E402
    liveness_strength,
    promotion_denominator,
    score_coverage,
)
from sable.registry import ModelRegistry  # noqa: E402
from sable.store.db import TraceRow, TraceStore  # noqa: E402


# ---------------------------------------------------------------- pure helpers

def test_score_coverage():
    assert score_coverage(100, 10) == 0.1
    assert score_coverage(3, 3) == 1.0
    assert score_coverage(0, 0) is None
    assert score_coverage(10, 0) == 0.0


def test_promotion_denominator():
    assert promotion_denominator(2, 1) == 3
    assert promotion_denominator(0, 0) == 0
    assert promotion_denominator(5, 2) == 7


def test_liveness_strength():
    assert liveness_strength("spawn") == "pid_probe"
    assert liveness_strength(None) == "pid_probe"
    assert liveness_strength("attach") == "state_file"


# ---------------------------------------------------------------- endpoint

@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "uf-cli"
    monkeypatch.setenv("SABLE_HOME", str(h))
    return h


@pytest.fixture()
def app(home):
    return DashboardApp(SableConfig(), home=home)


def _seed(home):
    """4 traces, only 1 scored; 1 promote op (rate would read 100%)."""
    store = TraceStore(Path(home) / "buffer" / "sable.db")
    for i, score in enumerate((0.5, None, None, None)):
        tid = f"h{i}"
        p = Path(store.db_path).parent / f"{tid}.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"kind": "trace_header"}) + "\n")
        store.insert_trace(TraceRow(
            trace_id=tid, session_id=None,
            timestamp="2026-09-20T10:00:00+00:00",
            agent_profile="p", model_alias="m", model_version=None,
            endpoint_label=None, task_summary="t", status="success",
            outcome_score=score, num_tool_calls=0, tokens_in=1,
            tokens_out=1, duration_s=1.0, jsonl_path=str(p)))
    store.close()
    reg = ModelRegistry(Path(home) / "buffer" / "sable.db")
    m = reg.register_model("base-a")
    reg.set_validation(m.model_id, {"ok": True}, passed=True)
    reg.promote(m.model_id)  # single decision -> pass rate would say 100%
    reg.close()


def _sample(text: str, prefix: str):
    for line in text.splitlines():
        if line.startswith(prefix + " "):
            return line.rsplit(" ", 1)[1]
    return None


def test_metrics_endpoint_exposes_both_variants(app, home):
    _seed(home)
    resp = app.handle("GET", "/api/metrics", query={"refresh": "1"})
    assert resp.status == 200
    text = resp.raw.decode("utf-8")

    # pair 1: score mean (rate) + its raw denominator + coverage ratio
    assert _sample(text, "sable_outcome_score_mean") is not None
    assert _sample(text, "sable_traces_scored_total") == "1"
    cov = _sample(text, "sable_score_coverage_ratio")
    assert cov is not None and abs(float(cov) - 0.25) < 1e-9

    # pair 2: promotion pass RATE + its raw decision count (1/1 decisions)
    rate = _sample(text, "sable_promotion_pass_rate")
    assert rate is not None and abs(float(rate) - 1.0) < 1e-9
    assert _sample(text, "sable_promotion_decisions_total") == "1"

    # pair 3: instance liveness + evidence strength split
    assert "sable_instances_up{id=" in text
    assert 'sable_instance_up_strength{id="default",strength=' in text


def test_metrics_empty_home_still_valid(app):
    """Honest absence: no traces -> no coverage gauge, but exposition stays
    valid and the counters exist at zero."""
    resp = app.handle("GET", "/api/metrics", query={"refresh": "1"})
    text = resp.raw.decode("utf-8")
    assert "sable_score_coverage_ratio " not in text
    assert _sample(text, "sable_traces_scored_total") == "0"
    # promotion rate AND its denominator are honest-absent with zero decisions
    assert "sable_promotion_pass_rate " not in text
    assert "sable_promotion_decisions_total " not in text


def test_status_endpoint_carries_scored_denominator(app, home):
    _seed(home)
    resp = app.handle("GET", "/api/status")
    assert resp.status == 200
    buf = resp.body["buffer"]
    # raw counts beside the avg (the JS overview renders "1/4 scored")
    assert buf["total_traces"] == 4
    assert buf["scored_traces"] == 1
    assert buf["unscored_traces"] == 3

# ------------------------------------------------------------------------
# from tests/test_gaps_hardening.py
# ------------------------------------------------------------------------

import sys
from pathlib import Path

import pytest


from sable.core.config import SableConfig, load_config  # noqa: E402
from sable.core import indexer as ix  # noqa: E402
from sable.dashboard.app import DashboardApp  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_registries():
    """Keep the process-global index/embed registries from leaking."""
    yield
    ix._instances.clear()
    ix._embed_registry.clear()


# ------------------------------------------------- 1. shared embed client

def test_embed_client_shared_per_model():
    a = ix.get_embed_client("/models/x.gguf")
    b = ix.get_embed_client("/models/x.gguf")
    assert a is b
    c = ix.get_embed_client("/models/y.gguf")
    assert c is not a
    # different ubatch -> separate client (spawn args differ)
    d = ix.get_embed_client("/models/x.gguf", ubatch=1024)
    assert d is not a


def test_lexical_index_uses_no_shared_client(tmp_path):
    idx = ix.CodebaseIndex(tmp_path, index_cfg={"enabled": True})
    emb = idx.embedder()
    assert not emb.available()
    assert ix._embed_registry == {}  # nothing registered for the lexical path


# ------------------------------------- 2. workspace-change invalidation

def test_get_index_rebuilds_on_workspace_change(home):
    ws_a, ws_b = home / "a", home / "b"
    ws_a.mkdir(parents=True); ws_b.mkdir(parents=True)
    i1 = ix.get_index(home, cfg=None, project_id="p1",
                      workspace_dir=str(ws_a))
    assert i1.roots == [ws_a]
    i2 = ix.get_index(home, cfg=None, project_id="p1",
                      workspace_dir=str(ws_b))
    assert i2 is not i1
    assert i2.roots == [ws_b]
    # unchanged workspace keeps the cached instance
    assert ix.get_index(home, cfg=None, project_id="p1",
                        workspace_dir=str(ws_b)) is i2


# ------------------------------------------- 3. cancel + drop instance

def test_cancel_build_aborts(home, monkeypatch):
    ws = home / "ws"
    ws.mkdir(parents=True)
    for n in range(50):
        (ws / f"f{n}.py").write_text("x = 1\n" * 80)
    idx = ix.CodebaseIndex(home, index_cfg={"enabled": True,
                                            "paths_override": [str(ws)]})

    # Deterministic: the cancel flag flips during iteration of the FIRST
    # file (build() clears it at start, so racing a thread would be flaky).
    real_iter = ix.iter_files

    def _iter_then_cancel(roots, max_files=20_000):
        first = True
        for p in real_iter(roots, max_files=max_files):
            if first:
                idx._cancel.set()
            first = False
            yield p

    monkeypatch.setattr(ix, "iter_files", _iter_then_cancel)
    with pytest.raises(RuntimeError, match="cancelled"):
        idx.build()


def test_drop_index_instance(home):
    ix.get_index(home, cfg=None, project_id="dropme")
    assert ("dropme" in [k[1] for k in ix._instances])
    assert ix.drop_index_instance(home, "dropme") is True
    assert ix.drop_index_instance(home, "dropme") is False


# --------------------------------------- 4. phantom config key stripped

def test_auto_retrieve_threshold_stripped(tmp_path, monkeypatch):
    p = tmp_path / "cfg.yaml"
    p.write_text("indexing:\n  enabled: true\n"
                 "  auto_retrieve_threshold: 0.7\n")
    monkeypatch.setenv("SABLE_CONFIG", str(p))
    cfg = load_config()
    assert "auto_retrieve_threshold" not in cfg.indexing
    assert cfg.indexing.get("enabled") is True


# ------------------------------------------------- 5. /api/fs/read

@pytest.fixture()
def workspace(tmp_path):
    ws = tmp_path / "ws"
    (ws / "src").mkdir(parents=True)
    (ws / "src" / "app.py").write_text("print('hi')\n")
    (ws / "blob.bin").write_bytes(b"\x00\x01\x02binary")
    (ws / ".env").write_text("SECRET=1\n")
    return ws


def _mkproject(app, ws, name="p"):
    r = app.handle("POST", "/api/projects",
                   body={"name": name, "workspace_dir": str(ws)})
    assert r.status == 201, r.body
    return r.body["project"]["id"]


def test_fs_read_happy(app, workspace):
    pid = _mkproject(app, workspace)
    r = app.handle("GET", "/api/fs/read",
                   query={"project": pid, "path": "src/app.py"})
    assert r.status == 200, r.body
    assert r.body["content"] == "print('hi')\n"
    assert r.body["binary"] is False and r.body["truncated"] is False


def test_fs_read_binary_sniff(app, workspace):
    pid = _mkproject(app, workspace)
    r = app.handle("GET", "/api/fs/read",
                   query={"project": pid, "path": "blob.bin"})
    assert r.status == 200
    assert r.body["binary"] is True and r.body["content"] == ""


def test_fs_read_sensitive_and_traversal(app, workspace):
    pid = _mkproject(app, workspace)
    r = app.handle("GET", "/api/fs/read",
                   query={"project": pid, "path": ".env"})
    assert r.status == 400
    r = app.handle("GET", "/api/fs/read",
                   query={"project": pid, "path": "../escape.txt"})
    assert r.status == 400
    r = app.handle("GET", "/api/fs/read",
                   query={"project": pid, "path": "/etc/passwd"})
    assert r.status == 400
    r = app.handle("GET", "/api/fs/read",
                   query={"project": pid, "path": ""})
    assert r.status == 400


def test_fs_read_size_truncation(app, workspace, monkeypatch):
    import sable.dashboard.api.fsbrowse as fsbrowse_mod
    monkeypatch.setattr(fsbrowse_mod, "FS_READ_MAX_CHARS", 10)
    # between the char cap and the hard size gate (10 < 30 <= 40 bytes) so
    # the content-level truncation branch runs rather than the metadata-only one
    (workspace / "big.txt").write_text("x" * 30)
    pid = _mkproject(app, workspace)
    r = app.handle("GET", "/api/fs/read",
                   query={"project": pid, "path": "big.txt"})
    assert r.status == 200
    assert r.body["truncated"] is True
    assert len(r.body["content"]) == 10


# ------------------------------------------------- 6. tree entry cap

def test_fs_tree_truncated_flag(app, tmp_path, monkeypatch):
    # dedicated workspace (fixture's .env would be hidden-filtered after the
    # cap slice, making the exact entry count layout-dependent)
    ws = tmp_path / "ws-cap"
    ws.mkdir()
    import sable.dashboard.api.fsbrowse as fsbrowse_mod
    monkeypatch.setattr(fsbrowse_mod, "FS_TREE_MAX_ENTRIES", 2)
    for n in range(10):
        (ws / f"n{n}.txt").write_text("x")
    pid = _mkproject(app, ws)
    r = app.handle("GET", "/api/fs/tree", query={"project": pid, "path": ""})
    assert r.status == 200
    assert r.body["truncated"] is True
    assert len(r.body["entries"]) == 2


# ------------------------------------------- 7. duplicate project names

def test_duplicate_project_names_rejected(app):
    ws = Path("/tmp")  # any existing dir
    r1 = app.handle("POST", "/api/projects",
                    body={"name": "Website", "workspace_dir": str(ws)})
    assert r1.status == 201
    r2 = app.handle("POST", "/api/projects",
                    body={"name": "website ", "workspace_dir": str(ws)})
    assert r2.status == 409
    # renaming a project to its OWN name still works (id excluded)
    pid = r1.body["project"]["id"]
    r3 = app.handle("PUT", f"/api/projects/{pid}",
                    body={"name": "WEBSITE"})
    assert r3.status == 200


# ------------------------------- 8/9. 409 rebuild + stale error cleared

def test_index_search_409_on_rebuild_required(app, home, monkeypatch):
    # create a project so _index routes there
    ws = Path("/tmp")
    pid = _mkproject(app, ws, name="rebuildme")

    class Boom:
        enabled = True
        cfg = None  # get_index invalidation checks read .cfg/.index_cfg
        index_cfg: dict = {}
        roots = [Path("/tmp")]  # workspace-change check reads .roots

        def search(self, query, top_k=None, mode=None):
            raise ix.IndexRebuildRequired("dim mismatch")

        def status(self):
            return {}

    monkeypatch.setitem(ix._instances, (str(home), pid), Boom())
    r = app.handle("POST", "/api/index/search",
                   body={"query": "q", "project": pid})
    assert r.status == 409
    assert r.body["error"]["code"] == "index_rebuild_required"


def test_search_clears_stale_last_error(home):
    ws = home / "ws-clear"
    ws.mkdir(parents=True)
    (ws / "a.py").write_text("def alpha():\n    pass\n")
    idx = ix.CodebaseIndex(home, index_cfg={"enabled": True,
                                            "paths_override": [str(ws)]})
    idx.build()
    idx.last_error = "old failure"
    idx.search("alpha")
    assert idx.last_error is None


# ------------------------------------------- 10. search_codebase top_k

def test_search_codebase_schema_has_no_top_k():
    from sable.core.tools import TOOL_SCHEMAS

    sc = next(s for s in TOOL_SCHEMAS
              if s.get("function", {}).get("name") == "search_codebase")
    props = sc["function"]["parameters"]["properties"]
    assert "top_k" not in props
    assert "query" in props

# ------------------------------------------------------------------------
# from tests/test_pass4_gaps.py
# ------------------------------------------------------------------------

import json
import sys
from pathlib import Path

import pytest


from sable.registry import ModelRegistry  # noqa: E402
from sable.registry.registry import RegistryError  # noqa: E402
from sable.training.pipeline import parse_worker_rank_history  # noqa: E402
from sable.training.worker import (  # noqa: E402
    RankScaler,
    rank_scaling_from_config,
)


@pytest.fixture()
def registry(home):
    reg = ModelRegistry(home / "buffer" / "sable.db")
    yield reg
    reg.close()


# ================================================================ #43 registry

def test_enqueue_rejects_unknown_dep(registry):
    with pytest.raises(RegistryError, match="unknown job"):
        registry.enqueue_job("sft", depends_on=["job-9999"])


def test_dep_blocks_next_queued_until_done(registry):
    a = registry.enqueue_job("sft")
    b = registry.enqueue_job("export", depends_on=[a.job_id])
    # A later independent job is claimable while the dependent one waits.
    c = registry.enqueue_job("dpo")
    assert registry.next_queued().job_id == a.job_id
    registry.finish_job(a.job_id, "error", error="boom")
    # b cascade-failed with the dep reason; c still runs.
    rb = registry.get_job(b.job_id)
    assert rb.state == "error"
    assert "dependency" in (rb.error or "") and a.job_id in (rb.error or "")
    assert registry.next_queued().job_id == c.job_id


def test_dep_unblocks_on_success(registry):
    a = registry.enqueue_job("sft")
    b = registry.enqueue_job("export", depends_on=[a.job_id])
    assert registry.next_queued().job_id == a.job_id
    registry.finish_job(a.job_id, "done")
    assert registry.next_queued().job_id == b.job_id
    assert registry.blocked_queued() == []


def test_blocked_queued_lists_waiting_jobs(registry):
    a = registry.enqueue_job("sft")
    b = registry.enqueue_job("export", depends_on=[a.job_id])
    blocked = registry.blocked_queued()
    assert [j.job_id for j in blocked] == [b.job_id]
    assert blocked[0].depends_on == [a.job_id]


def test_transitive_cascade(registry):
    a = registry.enqueue_job("sft")
    b = registry.enqueue_job("export", depends_on=[a.job_id])
    c = registry.enqueue_job("validate", depends_on=[b.job_id])
    registry.finish_job(a.job_id, "error", error="boom")
    for jid in (b.job_id, c.job_id):
        row = registry.get_job(jid)
        assert row.state == "error"


def test_cancel_cascades(registry):
    a = registry.enqueue_job("sft")
    b = registry.enqueue_job("export", depends_on=[a.job_id])
    registry.cancel_job(a.job_id)
    rb = registry.get_job(b.job_id)
    assert rb.state == "error"
    assert "cancelled" in (rb.error or "")


def test_depends_on_roundtrips_through_row(registry):
    a = registry.enqueue_job("sft")
    b = registry.enqueue_job("export", depends_on=[a.job_id])
    got = registry.get_job(b.job_id)
    assert got.depends_on == [a.job_id]
    d = got.to_dict()
    assert d["depends_on"] == [a.job_id]


def test_legacy_db_migration_adds_column(home):
    """Pre-#43 jobs table (no depends_on_json) upgrades in place."""
    import sqlite3

    db = home / "buffer" / "sable.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE jobs (
            job_id TEXT PRIMARY KEY,
            kind TEXT NOT NULL CHECK (kind IN ('sft','dpo','grpo','export',
              'validate','curate','shadow_eval','evolve')),
            dataset_id TEXT, model_id TEXT,
            state TEXT NOT NULL CHECK (state IN ('queued','running','done',
              'error','cancelled')),
            log_path TEXT, pid INTEGER, queued_at TEXT NOT NULL,
            started_at TEXT, finished_at TEXT, error TEXT, params_json TEXT
        );
        INSERT INTO jobs (job_id, kind, state, queued_at)
        VALUES ('job-0001', 'sft', 'done', '2026-01-01T00:00:00+00:00');
        """
    )
    conn.commit()
    conn.close()

    reg = ModelRegistry(db)
    try:
        row = reg.get_job("job-0001")
        assert row is not None and row.depends_on == []
        b = reg.enqueue_job("export", depends_on=["job-0001"])
        nxt = reg.next_queued()
        assert nxt is not None and nxt.job_id == b.job_id  # dep already done
    finally:
        reg.close()


# ============================================================ #43 scheduler

STUB_WORKER = r'''
import argparse, json, os, sys
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("--job", required=True)
ap.add_argument("--kind", required=True)
ap.add_argument("--config", default="")
ap.add_argument("--dataset-dir", default=None)
ap.add_argument("--base-model", default=None)
ap.add_argument("--log-path", default=None)
ap.add_argument("--adapter-dir", default=None)
ap.add_argument("--gguf-out", default=None)
ap.add_argument("--quant", default="q4_k_m")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()
def emit(ev, **kw):
    line = json.dumps({"event": ev, "job": a.job, **kw})
    print(line, flush=True)
    if a.log_path:
        Path(a.log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(a.log_path, "a") as f:
            f.write(line + "\n")
emit("start"); emit("done")
'''


@pytest.fixture()
def stub(tmp_path):
    p = tmp_path / "stub_worker.py"
    p.write_text(STUB_WORKER)
    return str(p)


def test_run_job_refuses_blocked_job(registry, stub):
    from sable.training.scheduler import Scheduler, SchedulerError, TrainingConfig

    sched = Scheduler(registry=registry, cfg=TrainingConfig(),
                      worker_command=[sys.executable, stub])
    a = sched.submit("sft")
    b = sched.submit("export", depends_on=[a.job_id])
    with pytest.raises(SchedulerError, match="waits on dependencies"):
        sched.run_job(registry.get_job(b.job_id))
    # Still queued (not corrupted), and runs once the dep finishes.
    assert registry.get_job(b.job_id).state == "queued"
    registry.finish_job(a.job_id, "done")
    done = sched.run_job(registry.get_job(b.job_id))
    assert done.state == "done"


def test_run_pending_respects_edges(registry, stub):
    from sable.training.scheduler import Scheduler, TrainingConfig

    sched = Scheduler(registry=registry, cfg=TrainingConfig(),
                      worker_command=[sys.executable, stub])
    a = sched.submit("sft")
    b = sched.submit("export", depends_on=[a.job_id])
    finished = sched.run_pending()
    ids = [j.job_id for j in finished]
    assert ids == [a.job_id, b.job_id]  # topological despite submit order
    assert registry.get_job(b.job_id).state == "done"


# ================================================================= #34 scaler

def test_rank_scaling_disabled_by_default():
    sc = rank_scaling_from_config({})
    assert not sc.enabled
    assert sc.observe(1, 1.0) is False
    assert sc.plan_growth() is None


def test_rank_scaling_parses_mapping_knob():
    sc = rank_scaling_from_config({"training": {"rank_scaling": {
        "max_rank": 128, "min_improvement_steps": 2,
        "min_rel_improvement": 0.05}}})
    assert sc.enabled and sc.max_rank == 128
    assert sc.min_improvement_steps == 2
    assert abs(sc.min_rel_improvement - 0.05) < 1e-9


def test_scaler_triggers_after_consecutive_improvements():
    sc = RankScaler(enabled=True, min_improvement_steps=3,
                    min_rel_improvement=0.01)
    sc.start_stage(64)
    assert sc.observe(1, 1.00) is False   # seeds best
    assert sc.observe(2, 0.98) is False   # -2% (streak 1)
    assert sc.observe(3, 0.96) is False   # streak 2
    assert sc.observe(4, 0.94) is True    # streak 3 -> trigger
    assert sc.plan_growth() == 128


def test_scaler_streak_resets_on_non_improvement():
    sc = RankScaler(enabled=True, min_improvement_steps=3)
    sc.start_stage(64)
    sc.observe(1, 1.0)
    sc.observe(2, 0.98)
    assert sc.observe(3, 0.99) is False   # streak reset
    assert sc.streak == 0
    assert sc.observe(4, 0.95) is False   # one improvement again
    assert sc.triggered is False


def test_scaler_caps_at_max_rank():
    sc = RankScaler(enabled=True, max_rank=64, min_improvement_steps=1)
    sc.start_stage(64)
    assert sc.plan_growth() is None       # already at cap
    assert sc.observe(1, 1.0) is False
    assert sc.observe(2, 0.5) is False    # trigger impossible below cap


def test_scaler_records_growth_history():
    sc = RankScaler(enabled=True, min_improvement_steps=1)
    sc.start_stage(64)
    sc.observe(1, 1.0)
    assert sc.observe(2, 0.5) is True
    entry = sc.record_growth(128, step=2)
    assert entry == {"from_r": 64, "to_r": 128, "step": 2, "best_loss": 0.5}
    assert sc.rank_history == [entry]


def test_parse_worker_rank_history(tmp_path):
    log = tmp_path / "job.log"
    lines = [
        {"event": "progress", "step": 1, "loss": 1.0},
        {"event": "sft_done", "train_loss": 0.5, "eval_loss": None,
         "rank_history": [{"from_r": 64, "to_r": 128, "step": 9,
                           "best_loss": 0.5}]},
    ]
    log.write_text("\n".join(json.dumps(r) for r in lines))
    assert parse_worker_rank_history(str(log)) == lines[1]["rank_history"]

    # No history field => [] (scaling off is the norm).
    log.write_text(json.dumps({"event": "sft_done", "train_loss": 0.5}))
    assert parse_worker_rank_history(str(log)) == []
    assert parse_worker_rank_history(None) == []
    assert parse_worker_rank_history("/nonexistent/path") == []

# ------------------------------------------------------------------------
# from tests/test_modelhub.py
# ------------------------------------------------------------------------

import gzip
import json
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


from sable.core import modelhub  # noqa: E402
from sable.core.config import SableConfig  # noqa: E402
from sable.core.modelhub import (  # noqa: E402
    HubError,
    ModelHub,
    validate_repo_id,
)
from sable.dashboard.app import DashboardApp  # noqa: E402

# ------------------------------------------------------------- fake hub API

REPO_SHA = "deadbeefcafe"
FILE_A = b"A" * (3 * 1024)          # small file
FILE_B = b"B" * (5 * 1024)          # resumed-download victim
SEARCH_RESULT = [
    {"id": "testorg/Test-GGUF", "downloads": 999, "likes": 3,
     "lastModified": "2026-01-02T00:00:00.000Z", "tags": ["gguf"]},
    {"id": "org/Plain", "downloads": 5, "likes": 0,
     "lastModified": "", "tags": ["safetensors"]},
]
DATASET_RESULT = [
    {"id": "testds/Test-Dataset", "downloads": 10, "likes": 2,
     "lastModified": "2026-04-01T00:00:00.000Z", "tags": ["size_categories:n<1K"]},
]
SPACE_RESULT = [
    {"id": "testspace/mcp-weather", "likes": 42,
     "createdAt": "2026-03-01T00:00:00.000Z", "sdk": "docker",
     "tags": ["mcp", "mcp-server"]},
]


class _FakeHubHandler(BaseHTTPRequestHandler):
    """Subset of the HF Hub REST API + resolve endpoint (Range aware)."""

    interrupt_first = False  # when True: first resolve request dies mid-stream

    def log_message(self, format, *args):  # noqa: A002 - silence access log
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/api/models":
            self._json(SEARCH_RESULT)
        elif p == "/api/datasets":
            self._json(DATASET_RESULT)
        elif p == "/api/spaces":
            self._json(SPACE_RESULT)
        elif p == "/api/datasets/testds/Test-Dataset":
            self._json({
                "id": "testds/Test-Dataset", "sha": "ds-sha",
                "author": "testds", "downloads": 10, "likes": 2,
                "lastModified": "2026-04-01T00:00:00.000Z",
                "cardData": {"license": "cc-by-4.0"},
                "siblings": [
                    {"rfilename": "train.jsonl", "size": len(FILE_A)},
                    {"rfilename": "README.md", "size": 9}],
            })
        elif p.startswith("/datasets/testds/Test-Dataset/resolve/"):
            rel = p.split("/resolve/")[1].split("/", 1)[1]
            data = FILE_A if rel == "train.jsonl" else b"# ds card"
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif p == "/datasets/testds/Test-Dataset/raw/main/README.md":
            body = b"# Test dataset\ncorpus for tests\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif p == "/api/spaces/testspace/mcp-weather":
            self._json({
                "id": "testspace/mcp-weather", "sha": "sp-sha",
                "author": "testspace", "likes": 42,
                "createdAt": "2026-03-01T00:00:00.000Z",
                "sdk": "docker", "tags": ["mcp", "mcp-server"],
                "cardData": {"license": "mit"},
                "siblings": [
                    {"rfilename": "app.py", "size": 3},
                    {"rfilename": "Dockerfile", "size": len(FILE_A)}],
            })
        elif p.startswith("/spaces/testspace/mcp-weather/resolve/"):
            rel = p.split("/resolve/")[1].split("/", 1)[1]
            data = b"py\n" if rel == "app.py" else FILE_A
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif p == "/spaces/testspace/mcp-weather/raw/main/README.md":
            body = b"# mcp-weather\nA weather MCP server.\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif p == "/testorg/Test-GGUF/raw/main/README.md":
            body = b"---\nlicense: apache-2.0\n---\n# Test card\nHello **world**\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif p == "/api/models/testorg/Test-GGUF":
            self._json({
                "id": "testorg/Test-GGUF", "sha": REPO_SHA,
                "author": "testorg", "downloads": 42, "likes": 7,
                "lastModified": "2026-05-01T00:00:00.000Z",
                "pipeline_tag": "text-generation", "gated": False,
                "cardData": {"license": "apache-2.0",
                             "base_model": "deepseek-ai/Test-Base"},
                "gguf": {"total": 123456, "architecture": "qwen3",
                         "context_length": 40960},
                "siblings": [
                    {"rfilename": "Q4_K_M/model-00001-of-00002.gguf",
                     "size": len(FILE_A)},
                    {"rfilename": "Q4_K_M/model-00002-of-00002.gguf",
                     "size": len(FILE_B)},
                    {"rfilename": "Q8_0/model.gguf", "size": 7},
                    {"rfilename": "mmproj-model.gguf", "size": 7},
                ],
            })
        elif p == "/api/models/org/Plain":
            self._json({"id": "org/Plain", "sha": "cafefood", "siblings": [
                {"rfilename": "config.json", "size": 2},
                {"rfilename": "model.safetensors", "size": 4},
                {"rfilename": "README.md", "size": len(FILE_A)}]})
        elif p.startswith("/org/Plain/resolve/"):
            # Mimic HF: small text files are served gzip'd WHEN the client
            # advertises gzip, with Content-Length describing the COMPRESSED
            # body (real-HF behaviour that killed non-GGUF downloads).
            rel = p.split("/resolve/")[1].split("/", 1)[1]
            data = {"README.md": FILE_A, "config.json": b"xy",
                    "model.safetensors": b"abcd"}.get(rel, b"")
            enc = "gzip" in (self.headers.get("Accept-Encoding") or "")
            if enc and rel == "README.md":
                data = gzip.compress(data)
                self.send_response(200)
                self.send_header("Content-Encoding", "gzip")
            else:
                self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif p.startswith("/testorg/Test-GGUF/resolve/"):
            rel = p.split("/resolve/")[1].split("/", 1)[1]
            data = FILE_A if "00001" in rel else (
                FILE_B if "00002" in rel else b"x" * 7)
            rng = self.headers.get("Range")
            start = 0
            if rng:
                start = int(rng.split("=")[1].split("-")[0])
            if start:
                self.send_response(206)
                self.send_header("Content-Range",
                                 f"bytes {start}-{len(data)-1}/{len(data)}")
            else:
                self.send_response(200)
            if type(self).interrupt_first and "model-00002" in rel \
                    and not getattr(self.server, "interrupted_once", False):
                # Abort mid-stream once (chunked + hard socket close) => the
                # client keeps the bytes already delivered in its .part.
                setattr(self.server, "interrupted_once", True)
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                body = data[start:start + 1024]
                self.wfile.write(b"%x\r\n" % len(body) + body + b"\r\n")
                self.wfile.flush()
                self.connection.setsockopt(
                    socket.SOL_SOCKET, socket.SO_LINGER,
                    b"\x01\x00\x00\x00\x00\x00\x00\x00")  # RST, no trailer
                self.close_connection = True
                return
            self.send_header("Content-Length", str(len(data) - start))
            if rel.lower().endswith(".png"):
                self.send_header("Content-Type", "image/png")
            self.end_headers()
            self.wfile.write(data[start:])
        elif p.startswith("/api/models/404"):
            self._json({"error": "not found"}, 404)
        else:
            self._json({"error": "nope"}, 404)


@pytest.fixture()
def hub_server(monkeypatch, tmp_path):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _FakeHubHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    monkeypatch.setenv("HF_ENDPOINT", f"http://127.0.0.1:{srv.server_address[1]}")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    yield srv
    srv.shutdown()


@pytest.fixture()
def hub(tmp_path, hub_server):
    # home matters: datasets/spaces roots live under <home>/hub.
    return ModelHub(models_dir=str(tmp_path / "models"),
                    home=tmp_path / "home")


# ---------------------------------------------------------------- validation

def test_validate_repo_id():
    assert validate_repo_id("testorg/Qwen3-8B-GGUF") == "testorg/Qwen3-8B-GGUF"
    # Leading/trailing slashes normalize away (harmless: paths stay anchored).
    assert validate_repo_id("/org/name/") == "org/name"
    for bad in ("", "..", "a/../b", "//", "or g/", "a\\b", "../x", ".hidden/x"):
        with pytest.raises(HubError):
            validate_repo_id(bad)


# -------------------------------------------------------------------- search

def test_search_annotates_gguf(hub_server, hub):
    out = hub.search("anything")
    assert [m["repo_id"] for m in out] == ["testorg/Test-GGUF", "org/Plain"]
    assert out[0]["gguf"] is True and out[1]["gguf"] is False
    assert out[0]["downloads"] == 999


# ------------------------------------------------------------- detail/feed

def test_detail_card_metadata(hub_server, hub):
    d = hub.detail("testorg/Test-GGUF")
    assert d["name"] == "Test-GGUF" and d["author"] == "testorg"
    assert d["architecture"] == "qwen3"
    assert d["context_length"] == 40960
    assert d["license"] == "apache-2.0"
    assert d["base_models"] == ["deepseek-ai/Test-Base"]
    assert d["pipeline_tag"] == "text-generation"
    # README fetched with the YAML frontmatter still present (client strips).
    assert "Hello **world**" in d["readme"]
    # Variants ride along for the detail pane's quant picker.
    assert {v["variant"] for v in d["variants"]} == {"Q4_K_M", "Q8_0"}


def test_latest_feed(hub_server, hub):
    out = hub.feed()
    assert [m["repo_id"] for m in out] == ["testorg/Test-GGUF", "org/Plain"]


# ------------------------------------------------------------------ variants

def test_variants_grouping(hub_server, hub):
    info = hub.variants("testorg/Test-GGUF")
    assert info["gguf"] and info["sha"] == REPO_SHA
    by_name = {v["variant"]: v for v in info["variants"]}
    q4 = by_name["Q4_K_M"]
    assert q4["shard_count"] == 2
    # mmproj is a companion, never its own variant: it rides along with EVERY
    # quant download (its size included), so served models always have the
    # projector on disk.
    assert "mmproj-model.gguf" in q4["files"]
    assert q4["size_bytes"] == len(FILE_A) + len(FILE_B) + 7
    assert "mmproj-model.gguf" in by_name["Q8_0"]["files"]
    assert by_name["Q8_0"]["size_bytes"] == 14
    assert not any("mmproj" in v["variant"] for v in info["variants"])
    # default prefers a Q4-family quant
    assert info["default_variant"] == "Q4_K_M"


def test_variants_mtp_companions_ride_along(hub_server, monkeypatch):
    """MTP drafter files (incl. an MTP/ directory) join every variant."""


    def fake_get(url, **kw):
        class R:
            status_code = 200
            def json(self):
                return {"id": "org/Mtp", "sha": "sha1", "siblings": [
                    {"rfilename": "UD-Q4/model-00001-of-00002.gguf",
                     "size": 10},
                    {"rfilename": "UD-Q4/model-00002-of-00002.gguf",
                     "size": 11},
                    {"rfilename": "mmproj-F16.gguf", "size": 3},
                    {"rfilename": "MTP/mtp-org-Mtp-Q8_0.gguf", "size": 5},
                ]}
        return R()

    monkeypatch.setattr(modelhub, "_get", fake_get)
    info = modelhub.list_variants("org/Mtp")
    assert len(info["variants"]) == 1
    v = info["variants"][0]
    assert "MTP/mtp-org-Mtp-Q8_0.gguf" in v["files"]
    assert "mmproj-F16.gguf" in v["files"]
    assert v["size_bytes"] == 10 + 11 + 3 + 5


def test_variants_non_gguf(hub_server, hub):
    info = hub.variants("org/Plain")
    assert not info["gguf"]
    assert info["variants"][0]["variant"] == "full"
    assert info["default_variant"] == "full"


def test_download_identity_encoding_non_gguf(hub_server, hub):
    """Non-GGUF download must land identity bytes.

    The fake server gzip-encodes README.md when the client advertises gzip
    (what HF really does for small text files). With iter_raw() writing the
    wire stream straight to disk, a gzip-capable request corrupts the file;
    _headers() must therefore pin Accept-Encoding: identity.
    """
    assert "identity" in modelhub._headers().get("Accept-Encoding", "")
    job = hub.download_sync("org/Plain", "full")
    assert job.state == "done", job.error
    snap = hub.snapshot_dir("org/Plain", "cafefood")
    assert (snap / "README.md").read_bytes() == FILE_A
    assert (snap / "config.json").read_bytes() == b"xy"


def test_download_rejects_gzip_part_residue(hub_server, hub):
    """A .part left by the gzip bug must not be resumed-into (append would
    concatenate identity bytes onto a gzip stream); it is restarted clean."""
    snap = hub.snapshot_dir("org/Plain", "cafefood")
    (snap / "README.md").parent.mkdir(parents=True, exist_ok=True)
    part = snap / "README.md.part"
    part.write_bytes(gzip.compress(b"stale-garbage")[:16])
    job = hub.download_sync("org/Plain", "full")
    assert job.state == "done", job.error
    assert (snap / "README.md").read_bytes() == FILE_A
    assert not part.exists()


def test_download_redownloads_gzip_poisoned_final(hub_server, hub):
    """A *final* file that is gzip poison must be re-fetched, not skipped by
    the size shortcut (real-HF corrupt .gitattributes was 1570 bytes of gzip
    that coincidentally matched its declared size).

    Two faces: size-coinciding poison (skippable -> must still re-download)
    and short poison (would fail the size check if it were a .part)."""
    snap = hub.snapshot_dir("org/Plain", "cafefood")
    snap.mkdir(parents=True, exist_ok=True)
    gz = gzip.compress(b"z" * len(FILE_A))
    # Poison whose size coincides with the declared size: size shortcut
    # would happily skip it.
    (snap / "README.md").write_bytes(gz[:len(FILE_A)].ljust(len(FILE_A), b"z"))
    job = hub.download_sync("org/Plain", "full")
    assert job.state == "done", job.error
    assert (snap / "README.md").read_bytes() == FILE_A
    # Short poison: must be purged, not resumed-into or treated as final.
    (snap / "README.md").write_bytes(gz[:16])
    job2 = hub.download_sync("org/Plain", "full")
    assert job2.state == "done", job2.error
    assert (snap / "README.md").read_bytes() == FILE_A


# ------------------------------------------------------------------ download

def test_download_sync_writes_hub_layout(hub_server, hub):
    job = hub.download_sync("testorg/Test-GGUF", "Q8_0")
    assert job.state == "done", job.error
    snap = hub.base_dir() / "models--testorg--Test-GGUF" / "snapshots" / REPO_SHA
    assert (snap / "Q8_0" / "model.gguf").read_bytes() == b"x" * 7
    assert (snap / "sable_download.json").is_file()
    assert (hub.base_dir() / "models--testorg--Test-GGUF" / "refs" / "main"
            ).read_text().strip() == REPO_SHA


def test_download_resumes_from_part(hub_server, hub):
    _FakeHubHandler.interrupt_first = True
    hub_server.interrupted_once = False
    try:
        job = hub.download_sync("testorg/Test-GGUF", "Q4_K_M")
        # First pass fails on the lied-about size; .part survives.
        assert job.state == "error"
        part = (hub.base_dir() / "models--testorg--Test-GGUF" / "snapshots"
                / REPO_SHA / "Q4_K_M" / "model-00002-of-00002.gguf.part")
        assert part.is_file() and part.stat().st_size == 1024
        # Retry resumes with Range and completes.
        job2 = hub.download_sync("testorg/Test-GGUF", "Q4_K_M")
        assert job2.state == "done", job2.error
        final = part.with_name(part.name[: -len(".part")])
        assert final.read_bytes() == FILE_B
    finally:
        _FakeHubHandler.interrupt_first = False


def test_download_unknown_variant(hub_server, hub):
    job = hub.download_sync("testorg/Test-GGUF", "Q99_NOPE")
    assert job.state == "error" and "unknown variant" in (job.error or "")


def test_download_not_found(hub_server, hub):
    job = hub.download_sync("nope/missing")
    assert job.state == "error" and "not found" in (job.error or "")


def test_start_download_job_and_single_flight(hub_server, hub):
    jid = hub.start_download("testorg/Test-GGUF", "Q8_0")
    jid2 = hub.start_download("testorg/Test-GGUF", "Q8_0")  # dedup while active
    # Wait for the worker to finish (tiny files).
    import time
    for _ in range(100):
        if hub.job(jid).state in ("done", "error"):
            break
        time.sleep(0.05)
    assert hub.job(jid).state == "done", hub.job(jid).error
    # After completion a new job id is issued (no longer in flight).
    jid3 = hub.start_download("testorg/Test-GGUF", "Q8_0")
    assert jid3 != jid
    assert len(hub.jobs()) >= 2


# ------------------------------------------------------------------ manage

def test_installed_and_delete_roundtrip(hub_server, hub):
    assert hub.installed() == []
    job = hub.download_sync("testorg/Test-GGUF", "Q8_0")
    assert job.state == "done"
    inst = hub.installed()
    assert len(inst) == 1
    assert inst[0]["repo_id"] == "testorg/Test-GGUF"
    assert inst[0]["size_bytes"] > 0
    assert "Q8_0" in inst[0]["quants"]

    out = hub.delete("testorg/Test-GGUF")
    assert out["freed_bytes"] > 0
    assert hub.installed() == []


def test_delete_missing_repo(hub):
    with pytest.raises(HubError, match="not downloaded"):
        hub.delete("never/seen")


def test_base_dir_fallback_to_home(tmp_path, monkeypatch):
    # No explicit dir + no existing HF cache anywhere => <home>/models.
    monkeypatch.delenv("SABLE_MODELS_DIR", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "nope"))  # nonexistent
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    h = ModelHub(models_dir=None, home=tmp_path / "home")
    assert h.base_dir() == tmp_path / "home" / "models"


def test_explicit_missing_models_dir_is_kept(tmp_path):
    # An explicit dir that doesn't exist yet must NOT fall back to the HF
    # cache — downloads create it on demand.
    target = tmp_path / "fresh" / "models"
    h = ModelHub(models_dir=str(target), home=tmp_path / "home")
    assert h.base_dir() == target


# ------------------------------------------------------------- dashboard API

@pytest.fixture()
def hub_app(tmp_path, hub_server):
    cfg = SableConfig(models_dir=str(tmp_path / "models"))
    return DashboardApp(cfg, home=tmp_path / ".sable")


def test_dashboard_hub_routes(hub_app):
    r = hub_app.handle("GET", "/api/hub/search", query={"q": "test"})
    assert r.status == 200
    assert r.body["results"][0]["repo_id"] == "testorg/Test-GGUF"

    r = hub_app.handle("GET", "/api/hub/variants", query={"repo": "org/Plain"})
    assert r.status == 200 and r.body["default_variant"] == "full"

    r = hub_app.handle("POST", "/api/hub/download",
                   body={"repo": "testorg/Test-GGUF", "variant": "Q8_0"})
    assert r.status == 202
    job_id = r.body["job_id"]

    import time
    j = hub_app.handle("GET", f"/api/hub/downloads/{job_id}")
    for _ in range(100):
        if j.body["state"] in ("done", "error"):
            break
        time.sleep(0.05)
        j = hub_app.handle("GET", f"/api/hub/downloads/{job_id}")
    assert j.body["state"] == "done", j.body["error"]

    r = hub_app.handle("GET", "/api/hub/installed")
    assert r.status == 200
    assert any(i["repo_id"] == "testorg/Test-GGUF" for i in r.body["installed"])

    r = hub_app.handle("POST", "/api/hub/delete", body={"repo": "testorg/Test-GGUF"})
    assert r.status == 200 and r.body["freed_bytes"] > 0


def test_dashboard_hub_detail_and_feed_routes(hub_app):
    r = hub_app.handle("GET", "/api/hub/detail", query={"repo": "testorg/Test-GGUF"})
    assert r.status == 200
    assert r.body["architecture"] == "qwen3"
    assert r.body["base_models"] == ["deepseek-ai/Test-Base"]

    r = hub_app.handle("GET", "/api/hub/feed")
    assert r.status == 200
    assert len(r.body["models"]) >= 1

    # Filtered search accepts the new params (fake hub ignores them).
    r = hub_app.handle("GET", "/api/hub/search",
                   query={"q": "x", "gguf": "1", "capability": "vision",
                          "sort": "newest"})
    assert r.status == 200


def test_hub_image_proxy_and_card_image(hub_server, hub):
    from sable.core.modelhub import HubError as _HE
    from sable.core.modelhub import _first_card_image, fetch_image

    # Repo-relative path fetched through the configured endpoint.
    data, ctype = fetch_image(repo_id="testorg/Test-GGUF", path="a.png")
    assert ctype == "image/png" and data

    # Absolute URLs: https + public hosts only (SSRF screen, no network).
    with pytest.raises(_HE):
        fetch_image(url="https://127.0.0.1:9/x.png")
    with pytest.raises(_HE):
        fetch_image(url="http://cdn.example.com/x.png")
    with pytest.raises(_HE):
        fetch_image(repo_id="testorg/Test-GGUF", path="../secrets.png")

    # Card image extraction: first non-badge markdown image wins.
    md = ("[![badge](https://img.shields.io/x.svg)](y)\n"
          "![art](assets/banner.png)\n![](other.jpg)")
    assert _first_card_image(md) == "assets/banner.png"


def test_dashboard_hub_image_route(app, hub_server):
    r = app.handle("GET", "/api/hub/image",
                   query={"repo": "testorg/Test-GGUF", "path": "a.png"})
    assert r.status == 200 and r.raw and r.content_type == "image/png"
    # Private/loopback targets stay blocked (SSRF screen); arbitrary public
    # https hosts are now allowed so card art on third-party CDNs renders.
    r = app.handle("GET", "/api/hub/image",
                   query={"url": "https://127.0.0.1:9/x.png"})
    assert r.status == 400
    r = app.handle("GET", "/api/hub/image",
                   query={"url": "http://example.com/x.png"})  # not https
    assert r.status == 400


def test_dashboard_hub_errors(hub_app):
    r = hub_app.handle("GET", "/api/hub/variants", query={"repo": "../etc"})
    assert r.status == 400
    r = hub_app.handle("GET", "/api/hub/variants", query={"repo": "api/nope"})
    assert r.status in (400, 404)
    r = hub_app.handle("GET", "/api/hub/downloads/does-not-exist")
    assert r.status == 400


# ------------------------------------------------------------- kinds: datasets

def test_normalize_kind():
    from sable.core.modelhub import normalize_kind

    assert normalize_kind("") == "models"
    assert normalize_kind("model") == "models"
    assert normalize_kind("dataset") == "datasets"
    assert normalize_kind("mcp") == "spaces"      # MCP servers live in Spaces
    with pytest.raises(HubError):
        normalize_kind("widgets")


def test_search_datasets_and_spaces(hub_server, hub):
    ds = hub.search("corpus", kind="datasets")
    assert ds[0]["repo_id"] == "testds/Test-Dataset"
    assert ds[0]["kind"] == "datasets"

    sp = hub.search("", kind="spaces")
    assert sp[0]["repo_id"] == "testspace/mcp-weather"
    assert sp[0]["sdk"] == "docker"
    # Spaces have no download counter; likes still surface.
    assert sp[0]["likes"] == 42


def test_dataset_detail_and_download(hub_server, hub, tmp_path):
    d = hub.detail("testds/Test-Dataset", kind="datasets")
    assert d["kind"] == "datasets"
    assert d["variants"][0]["variant"] == "full"
    assert "corpus for tests" in d["readme"]

    job = hub.download_sync("testds/Test-Dataset", kind="datasets")
    assert job.state == "done", job.error
    # Datasets land under <home>/hub/datasets, never the models dir.
    ds_base = hub.kind_base("datasets")
    assert ds_base == tmp_path / "home" / "hub" / "datasets"
    snap = ds_base / "datasets--testds--Test-Dataset" / "snapshots" / "ds-sha"
    assert (snap / "train.jsonl").read_bytes() == FILE_A
    # Model discovery inventory must not see datasets.
    assert hub.installed("models") == []
    inst = hub.installed("datasets")
    assert inst and inst[0]["repo_id"] == "testds/Test-Dataset"

    out = hub.delete("testds/Test-Dataset", kind="datasets")
    assert out["freed_bytes"] > 0
    assert hub.installed("datasets") == []


def test_mcp_space_detail_and_download(hub_server, hub):
    d = hub.detail("testspace/mcp-weather", kind="spaces")
    assert d["kind"] == "spaces" and d["sdk"] == "docker"
    assert "weather MCP server" in d["readme"]

    job = hub.download_sync("testspace/mcp-weather", kind="spaces")
    assert job.state == "done", job.error
    snap = (hub.kind_base("spaces") / "spaces--testspace--mcp-weather"
            / "snapshots" / "sp-sha")
    assert (snap / "Dockerfile").read_bytes() == FILE_A

    # installed_all spans every kind.
    rows = hub.installed_all()
    assert {r["kind"] for r in rows} == {"spaces"}


def test_dashboard_hub_kind_routes(hub_app):
    r = hub_app.handle("GET", "/api/hub/search", query={"q": "x", "kind": "datasets"})
    assert r.status == 200
    assert r.body["results"][0]["repo_id"] == "testds/Test-Dataset"

    r = hub_app.handle("GET", "/api/hub/feed", query={"kind": "spaces"})
    assert r.status == 200
    assert r.body["models"][0]["repo_id"] == "testspace/mcp-weather"

    r = hub_app.handle("GET", "/api/hub/detail",
                   query={"repo": "testds/Test-Dataset", "kind": "datasets"})
    assert r.status == 200 and r.body["kind"] == "datasets"

    r = hub_app.handle("POST", "/api/hub/download",
                   body={"repo": "testds/Test-Dataset", "kind": "datasets"})
    assert r.status == 202
    import time
    jid = r.body["job_id"]
    j = hub_app.handle("GET", f"/api/hub/downloads/{jid}")
    for _ in range(100):
        if j.body["state"] in ("done", "error"):
            break
        time.sleep(0.05)
        j = hub_app.handle("GET", f"/api/hub/downloads/{jid}")
    assert j.body["state"] == "done", j.body["error"]

    r = hub_app.handle("POST", "/api/hub/delete",
                   body={"repo": "testds/Test-Dataset", "kind": "datasets"})
    assert r.status == 200 and r.body["freed_bytes"] > 0

    r = hub_app.handle("GET", "/api/hub/search", query={"kind": "bogus"})
    assert r.status == 400


# ------------------------------------------------------------- PulseMCP dir

def test_mcp_dir_requires_key(monkeypatch):
    from sable.core.modelhub import HubError as _HE, search_mcp_dir

    monkeypatch.delenv("PULSEMCP_API_KEY", raising=False)
    with pytest.raises(_HE, match="PULSEMCP_API_KEY"):
        search_mcp_dir("github")


def test_mcp_dir_search_and_detail(monkeypatch):
    import httpx

    from sable.core.modelhub import mcp_dir_detail, search_mcp_dir

    monkeypatch.setenv("PULSEMCP_API_KEY", "test-key")
    seen = {}

    def fake_get(url, params=None, headers=None, **kw):
        seen["url"] = url
        seen["params"] = dict(params or {})
        seen["key"] = (headers or {}).get("X-API-Key")

        class R:
            status_code = 200

            def json(self_inner):
                if url.endswith("/servers"):
                    return {"data": [{"id": "github", "name": "GitHub",
                                      "author": "github",
                                      "description": "repos & PRs",
                                      "downloads": 1234,
                                      "github_stars": 900,
                                      "categories": ["Development"]},
                                     ]}
                return {"data": {"id": "github", "name": "GitHub",
                                 "readme": "# GitHub MCP",
                                 "transports": {"stdio": True,
                                                "streamable-http": False}}}
        return R()

    monkeypatch.setattr(httpx, "get", fake_get)
    rows = search_mcp_dir("git hub", limit=10)
    assert seen["key"] == "test-key"
    assert seen["params"]["search_query"] == "git hub"
    assert rows[0]["kind"] == "mcp-dir"
    assert rows[0]["url"].endswith("/servers/github")

    d = mcp_dir_detail("github")
    assert d["readme"] == "# GitHub MCP"
    assert d["transports"] == ["stdio"]
    with pytest.raises(modelhub.HubError):
        mcp_dir_detail("../etc/passwd")


def test_dashboard_mcp_dir_route(app, monkeypatch):
    # PulseMCP source without a key -> 400 with the signup hint; the default
    # registry source no longer touches that key at all.
    monkeypatch.delenv("PULSEMCP_API_KEY", raising=False)
    r = app.handle("GET", "/api/hub/mcp_dir",
                   query={"q": "x", "source": "pulsemcp"})
    assert r.status == 400 and "PULSEMCP_API_KEY" in r.body["error"]["message"]


def test_mcp_registry_search_and_detail(monkeypatch):
    import httpx

    from sable.core.modelhub import mcp_registry_detail, search_mcp_registry

    def fake_get(url, params=None, headers=None, **kw):
        class R:
            status_code = 200

            def json(self_inner):
                if url.endswith("/servers"):
                    return {"servers": [
                        {"server": {"name": "io.github.foo/weather",
                                    "title": "Weather",
                                    "description": "forecast server",
                                    "version": "1.2.0",
                                    "repository": {"url": "https://github.com/foo/weather"},
                                    "remotes": [{"type": "streamable-http",
                                                 "url": "https://x/mcp"}]}},
                        # older non-latest duplicate must be dropped
                        {"server": {"name": "io.github.bar/old",
                                    "version": "1.0.0"},
                         "_meta": {"io.modelcontextprotocol.registry/official":
                                   {"isLatest": False}}},
                    ]}
                return {"server": {"name": "io.github.foo/weather",
                                   "description": "forecast server",
                                   "version": "1.2.0",
                                   "packageVersions": [
                                       {"registryType": "npm",
                                        "identifier": "@foo/weather"}],
                                   "repository": {
                                       "url": "https://github.com/foo/weather"}}}
        return R()

    monkeypatch.setattr(httpx, "get", fake_get)
    rows = search_mcp_registry("weather", limit=10)
    assert len(rows) == 1  # non-latest row filtered
    assert rows[0]["kind"] == "mcp-registry"
    assert rows[0]["github_url"] == "https://github.com/foo/weather"
    assert rows[0]["transports"] == ["streamable-http"]

    d = mcp_registry_detail("io.github.foo/weather")
    assert "@foo/weather" in d["readme"]
    with pytest.raises(modelhub.HubError):
        mcp_registry_detail("../etc/passwd")


def test_dashboard_mcp_registry_route(app, monkeypatch):
    import httpx

    def fake_get(url, params=None, headers=None, **kw):
        class R:
            status_code = 200

            def json(self_inner):
                return {"servers": [{"server": {
                    "name": "io.github.foo/weather", "version": "1.0",
                    "description": "d"}}]}
        return R()

    monkeypatch.setattr(httpx, "get", fake_get)
    r = app.handle("GET", "/api/hub/mcp_dir", query={"q": "weather"})
    assert r.status == 200
    assert r.body["results"][0]["repo_id"] == "io.github.foo/weather"

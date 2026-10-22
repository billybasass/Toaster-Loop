"""Phase 1 smoke tests — no real model or GGUF download required.

Run: pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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

"""WP12 plans: CLI round-trips (`toaster-loop plan ...`, run --plan-task)."""

from __future__ import annotations

import json

import pytest
import yaml

from toaster_loop.cli import main


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """Config with plans enabled + an attach endpoint stub (fake server)."""
    home = tmp_path / "plan-home"
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(home))
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump({
        "home": str(home),
        "plans": {"enabled": True},
    }))
    return home, cfg_file


def _cli(cfg_file, *argv):
    return main([*argv, "--config", str(cfg_file)])


# ---------------------------------------------------------------- gating

def test_plan_commands_require_enabled(tmp_path, monkeypatch):
    home = tmp_path / "off-home"
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(home))
    cfg_file = tmp_path / "off.yaml"
    cfg_file.write_text(yaml.safe_dump({"home": str(home)}))
    assert _cli(cfg_file, "plan", "list") == 1


# ------------------------------------------------------------ round-trip

def test_add_list_show_next_roundtrip(env, capsys):
    home, cfg = env
    assert _cli(cfg, "plan", "add", "first thing", "--priority",
                "high") == 0
    assert "added t1" in capsys.readouterr().out
    assert _cli(cfg, "plan", "add", "second", "--deps", "t1") == 0
    assert _cli(cfg, "plan", "list") == 0
    out = capsys.readouterr().out
    assert "[ ] t1 [high] first thing" in out
    assert "(deps: t1)" in out

    assert _cli(cfg, "plan", "next") == 0
    assert "t1 [high] first thing" in capsys.readouterr().out

    assert _cli(cfg, "plan", "show", "t2") == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["deps"] == ["t1"]

    # t2 is blocked until t1 closes
    assert _cli(cfg, "plan", "set-status", "t1", "done") == 0
    assert _cli(cfg, "plan", "next") == 0
    assert "t2" in capsys.readouterr().out


def test_depends_validate_set_status(env, capsys):
    _, cfg = env
    _cli(cfg, "plan", "add", "a")
    _cli(cfg, "plan", "add", "b")
    capsys.readouterr()
    assert _cli(cfg, "plan", "depends", "t2", "t1") == 0
    assert "t2 now depends on t1" in capsys.readouterr().out
    # cycle refused (exit 1)
    assert _cli(cfg, "plan", "depends", "t1", "t2") == 1
    assert _cli(cfg, "plan", "validate") == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    # argparse choice gate: invalid status never reaches the store
    with pytest.raises(SystemExit) as excinfo:
        _cli(cfg, "plan", "set-status", "t1", "shipped")
    assert excinfo.value.code == 2
    assert _cli(cfg, "plan", "set-status", "t1", "blocked",
                "--note", "waiting on API") == 0
    out = capsys.readouterr().out
    assert "t1 -> blocked" in out


def test_remove_rewires(env, capsys):
    _, cfg = env
    _cli(cfg, "plan", "add", "a")
    _cli(cfg, "plan", "add", "b", "--deps", "t1")
    capsys.readouterr()
    assert _cli(cfg, "plan", "remove", "t1") == 0
    out = capsys.readouterr().out
    assert "removed t1" in out and "rewired deps of t2" in out


def test_export_import_roundtrip(env, tmp_path, capsys):
    _, cfg = env
    _cli(cfg, "plan", "add", "alpha")
    _cli(cfg, "plan", "add", "beta", "--deps", "t1")
    capsys.readouterr()
    assert _cli(cfg, "plan", "export") == 0
    exported = json.loads(capsys.readouterr().out)
    assert exported["version"] == 1 and len(exported["tasks"]) == 2

    # text brief import (deterministic mode)
    brief = tmp_path / "brief.txt"
    brief.write_text("Set up repo [priority=high]\nWrite code [deps=t1]\n")
    assert _cli(cfg, "plan", "import", str(brief)) == 0
    out = capsys.readouterr().out
    assert "imported t3: Set up repo" in out
    assert "(text mode)" in out

    # JSON brief file
    jfile = tmp_path / "brief.json"
    jfile.write_text(json.dumps(
        [{"title": "gamma"}, {"title": "delta", "deps": ["t1"]}]))
    assert _cli(cfg, "plan", "import", str(jfile)) == 0
    out = capsys.readouterr().out
    assert "imported t5: gamma" in out


def test_import_binary_rejected(env, tmp_path, capsys):
    _, cfg = env
    blob = tmp_path / "blob.bin"
    blob.write_bytes(b"\x00\x01\x02\xff" * 32)
    assert _cli(cfg, "plan", "import", str(blob)) == 1
    err = capsys.readouterr().err.lower()
    assert "binary" in err or "not text" in err


def test_project_scoping(env, capsys):
    _, cfg = env
    assert _cli(cfg, "plan", "add", "p task", "--project", "px") == 0
    capsys.readouterr()
    assert _cli(cfg, "plan", "list") == 0
    assert "(plan is empty)" in capsys.readouterr().out   # default untouched
    assert _cli(cfg, "plan", "list", "--project", "px") == 0
    assert "p task" in capsys.readouterr().out


# ------------------------------------------------------------- run --task

def test_run_plan_task_seeds_prompt(env, monkeypatch, capsys):
    from fake_server import FakeOpenAIServer

    _, cfg = env
    _cli(cfg, "plan", "add", "ship the thing", "--description", "desc here")
    capsys.readouterr()

    captured = {}

    class StubRunner:
        name = "stub"

        def __init__(self, *_a, **kw):
            self.recorder = kw.get("recorder")

        def run(self, msg, history=None):
            captured["msg"] = msg
            from toaster_loop.core.agent_runner import AgentResult
            return AgentResult(final_response="ok", messages=[],
                               backend="stub", tokens_in=1, tokens_out=1)

    monkeypatch.setattr("toaster_loop.cli.create_runner", StubRunner)
    with FakeOpenAIServer() as fs:
        rc = main(["run", "--config", str(cfg), "--attach",
                   fs.base_url.rsplit("/v1", 1)[0], "--plan-task", "t1",
                   "--task", "extra instruction"])
    assert rc == 0
    prompt = captured["msg"]
    assert prompt.startswith("# Task t1: ship the thing")
    assert "desc here" in prompt
    assert "extra instruction" in prompt          # rides along


def test_run_plan_task_requires_enabled(tmp_path, monkeypatch):
    home = tmp_path / "off2"
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(home))
    cfg_file = tmp_path / "off2.yaml"
    cfg_file.write_text(yaml.safe_dump({"home": str(home)}))
    rc = main(["run", "--config", str(cfg_file),
               "--attach", "http://127.0.0.1:9", "--plan-task", "t1"])
    assert rc == 1


def test_run_plan_task_unknown_id(env, capsys):
    from fake_server import FakeOpenAIServer

    _, cfg = env
    with FakeOpenAIServer() as fs:
        rc = main(["run", "--config", str(cfg), "--attach",
                   fs.base_url.rsplit("/v1", 1)[0], "--plan-task", "t9"])
    assert rc == 1
    assert "no task" in capsys.readouterr().err


# ------------------------------------------------------------- loop CLI

def test_plan_loop_status_empty(env, capsys):
    _, cfg = env
    assert _cli(cfg, "plan", "loop-status") == 0
    assert "(no loop progress file" in capsys.readouterr().out


def test_plan_loop_single_flight_conflict(env, monkeypatch, capsys):
    """Second `toaster-loop plan loop` while one runs => conflict exit 1, no queue."""
    import threading

    from toaster_loop.core import plan_loop as pl

    _, cfg = env
    _cli(cfg, "plan", "add", "solo")
    capsys.readouterr()

    gate = threading.Event()
    started = threading.Event()

    def slow_run(prompt, task, profile):
        started.set()
        gate.wait(5)
        return {"ok": False}

    monkeypatch.setattr(pl, "_default_run_fn", lambda cfg_, p: slow_run)
    loop = pl.PlanLoop(__import__("toaster_loop.core.config",
                                   fromlist=["ToasterLoopConfig"]
                                   ).ToasterLoopConfig(plans={"enabled": True}),
                       None, run_fn=slow_run).start()
    started.wait(5)
    try:
        rc = _cli(cfg, "plan", "loop")
        err = capsys.readouterr().err
        assert rc == 1 and "already running" in err
    finally:
        loop.cancel()
        gate.set()
        loop.wait(timeout=5)

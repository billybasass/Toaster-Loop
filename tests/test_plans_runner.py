"""WP12 plans: agent surface (tools + agent_runner wiring) and delegation.

Byte-identical-off contract, bounded injection, forgery-proof `_plan_ctx`,
plan_progress note/run recording, and delegate_task(plan_task=...).
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from fake_server import FakeOpenAIServer  # noqa: E402

from sable.core import delegation
from sable.core.agent_runner import SableAgentClient, create_runner  # noqa: E402
from sable.core.config import SableConfig  # noqa: E402
from sable.core.tools import (TOOL_SCHEMAS, PLAN_TOOL_NAMES, plan_add,
                              plan_next, plan_progress, plan_view)

PLAN_TOOLS = set(PLAN_TOOL_NAMES)


def _tool_names(client) -> set:
    return {s.get("function", {}).get("name") for s in client.tools}


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("SABLE_HOME", str(h))
    return h


# --------------------------------------------------------------- gating

def test_disabled_zero_plan_tools_byte_identical_prompt(home):
    """plans off: no plan tools registered, prompt untouched."""
    ref = create_runner(SableConfig(), "http://127.0.0.1:1", "m",
                        system_prompt="BASE")
    assert not (PLAN_TOOLS & _tool_names(ref))
    assert ref.system_prompt == "BASE"
    # plan tool names are also absent from the exposed schema table copy the
    # runner builds (they exist in TOOL_SCHEMAS but must be stripped here)
    for s in ref.tools:
        assert s["function"]["name"] not in PLAN_TOOLS


def test_enabled_tools_present_and_section_injected_capped(home):
    from sable.core.plans import add_task, load_plan, save_plan

    store = load_plan(None)
    for i in range(40):
        add_task(store, {"title": "a fairly long plan task title " + str(i)})
    save_plan(store)

    cfg = SableConfig(plans={"enabled": True, "max_inject_chars": 300})
    runner = create_runner(cfg, "http://127.0.0.1:1", "m",
                           system_prompt="BASE")
    assert PLAN_TOOLS <= _tool_names(runner)
    prompt = runner.system_prompt
    assert prompt.startswith("BASE")
    assert "## Project plan" in prompt
    # the injected section itself is capped (allow the base + separator slack)
    section = prompt.split("## Project plan", 1)[1]
    assert len(section) <= 300


def test_enabled_empty_plan_prompt_identical(home):
    cfg = SableConfig(plans={"enabled": True})
    runner = create_runner(cfg, "http://127.0.0.1:1", "m",
                           system_prompt="BASE")
    assert runner.system_prompt == "BASE"   # no tasks => nothing to inject


def test_inject_summary_false_skips_section(home):
    from sable.core.plans import add_task, load_plan, save_plan

    store = load_plan(None)
    add_task(store, {"title": "x"})
    save_plan(store)
    cfg = SableConfig(plans={"enabled": True, "inject_summary": False})
    runner = create_runner(cfg, "http://127.0.0.1:1", "m",
                           system_prompt="BASE")
    assert runner.system_prompt == "BASE"


def test_profile_tool_subset_can_drop_plan_tools(home):
    cfg = SableConfig(plans={"enabled": True})
    runner = create_runner(cfg, "http://127.0.0.1:1", "m",
                           enabled_tool_names=["read_file"])
    names = _tool_names(runner)
    assert "read_file" in names
    assert not (PLAN_TOOLS & names)     # profile subsets apply like any tool


# ---------------------------------------------------------------- forgery

def test_forged_plan_ctx_refused(home):
    """A model-supplied `_plan_ctx` must never reach the impl (schema gate)."""
    executed = {"hit": False}

    def spy_next(**kwargs):
        executed["hit"] = True
        return "SHOULD NOT RUN"

    script_msgs = [
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "c1", "type": "function", "function": {
             "name": "plan_next",
             "arguments": json.dumps(
                 {"_plan_ctx": {"project": "victim"}})}}]},
        {"role": "assistant", "content": "done"},
    ]

    class H(BaseHTTPRequestHandler):
        n = 0

        def log_message(self, *a):
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            msg = script_msgs[min(type(self).n, len(script_msgs) - 1)]
            type(self).n += 1
            body = json.dumps({"id": "x", "object": "chat.completion",
                               "model": "m",
                               "choices": [{"index": 0, "finish_reason": "stop",
                                            "message": msg}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    with FakeOpenAIServer(handler_cls=H) as fs:
        client = SableAgentClient(
            base_url=fs.base_url, model="m",
            tools=[s for s in TOOL_SCHEMAS
                   if s["function"]["name"] == "plan_next"],
            tool_impls={"plan_next": spy_next},
            plans_enabled=True)   # keep the family registered (gate test)
        result = client.run("what is next")
    assert not executed["hit"], "impl ran despite forged _plan_ctx"
    tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
    assert tool_msgs, "forged call produced no tool message"
    body = tool_msgs[0]["content"].lower()
    assert "_plan_ctx" in body or "unknown arg" in body, body[:120]


# ------------------------------------------------------- plan tools direct

def test_plan_tools_via_bound_ctx(home):
    from sable.core.plans import add_task, get_task, load_plan, save_plan

    store = load_plan(None)
    add_task(store, {"title": "first"})
    save_plan(store)
    ctx = {"project": None, "thread": "th-1", "_touched": []}

    out = plan_view("all", _plan_ctx=ctx)
    assert "t1" in out and "first" in out
    nxt = plan_next(_plan_ctx=ctx)
    assert "t1" in nxt
    prog = plan_progress("t1", "in_progress", note="hands on",
                         _plan_ctx=ctx)
    assert prog.startswith("task t1 -> in_progress")
    added = plan_add("follow-up", deps=["t1"], priority="high",
                     _plan_ctx=ctx)
    assert "added task t2" in added
    store = load_plan(None)
    t1 = get_task(store, "t1")
    assert "] hands on" in t1["details"]          # timestamped note appended
    assert any(r.get("thread") == "th-1" for r in t1["runs"])
    assert ctx["_touched"] == ["t1"]
    assert get_task(store, "t2")["deps"] == ["t1"]


def test_plan_progress_invalid_status_is_error_string(home):
    from sable.core.plans import add_task, load_plan, save_plan

    store = load_plan(None)
    add_task(store, {"title": "x"})
    save_plan(store)
    out = plan_progress("t1", "shipped", _plan_ctx={"_touched": []})
    assert out.startswith("error:")


def test_plan_add_cycle_refused(home):
    from sable.core.plans import add_task, load_plan, save_plan

    store = load_plan(None)
    add_task(store, {"title": "a"})                # t1
    add_task(store, {"title": "b", "deps": ["t1"]})  # t2
    save_plan(store)
    out = plan_add("c", deps=["t2"], _plan_ctx={"_touched": []})
    assert "added task t3" in out
    # now force a cycle attempt: t1 depends on t3 -> refused by validation
    store = load_plan(None)
    from sable.core.plans import PlanError, add_dependency

    with pytest.raises(PlanError):
        add_dependency(store, "t1", "t3")


# --------------------------------------------------------- delegate task

def test_delegate_task_builds_prompt_from_task(home, tmp_path, monkeypatch):
    """delegate_task(plan_task=...) => child prompt = build_task_prompt +
    the free-text instruction; child trace stamped with plan_task."""
    from sable.core.plans import add_task, load_plan, save_plan, set_status

    ws = tmp_path / "ws"
    ws.mkdir()
    (home / "projects").mkdir()
    (home / "projects" / "p1.json").write_text(json.dumps(
        {"id": "p1", "name": "p1", "workspace_dir": str(ws),
         "agents": ["coder"], "description": "", "agent_roles": {}}))
    store = load_plan("p1")
    add_task(store, {"title": "dep task", "details": "dep detail line"})  # t1
    set_status(store, "t1", "done")
    add_task(store, {"title": "main task", "description": "do it",
                     "deps": ["t1"], "test_strategy": "run tests"})       # t2
    save_plan(store)

    cfg = SableConfig(attach_url="http://127.0.0.1:9/v1")
    cfg.profiles = [{"name": "coder", "system_prompt": "code",
                     "tools": ["read_file"]}]
    seen = {}

    def fake_run_child(cfg_, prof, task, ctx, delegation_id, out,
                       plan_task=None):
        seen["task"] = task
        seen["plan_task"] = plan_task
        out.update({"completed": True, "result": "ok",
                    "child_trace_id": None, "failure_reason": None})

    monkeypatch.setattr(delegation, "_run_child", fake_run_child)
    payload = delegation.execute_delegation(
        "coder", "please proceed", plan_task="t2",
        _ctx={"cfg": cfg, "base_url": "http://127.0.0.1:9/v1",
              "model": "m", "project": "p1", "depth": 0})
    data = json.loads(payload)
    assert data.get("status") in ("completed", "ok"), payload
    prompt = seen["task"]
    assert "# Task t2: main task" in prompt
    assert "Completed dependencies:" in prompt and "dep task" in prompt
    assert "run tests" in prompt
    assert "please proceed" in prompt          # free text rides along
    assert seen["plan_task"] == "t2"


def test_delegate_unknown_plan_task_degrades(home, tmp_path, monkeypatch):
    """Bad plan_task never breaks delegation: plain prompt + note."""
    ws = tmp_path / "ws"
    ws.mkdir()
    (home / "projects").mkdir()
    (home / "projects" / "p1.json").write_text(json.dumps(
        {"id": "p1", "name": "p1", "workspace_dir": str(ws),
         "agents": ["coder"], "description": "", "agent_roles": {}}))
    seen = {}

    def fake_run_child(cfg_, prof, task, ctx, delegation_id, out,
                       plan_task=None):
        seen["task"] = task
        out.update({"completed": True, "result": "ok",
                    "child_trace_id": None, "failure_reason": None})

    monkeypatch.setattr(delegation, "_run_child", fake_run_child)
    cfg = SableConfig(attach_url="http://127.0.0.1:9/v1")
    cfg.profiles = [{"name": "coder", "system_prompt": "code",
                     "tools": ["read_file"]}]
    payload = delegation.execute_delegation(
        "coder", "just do it", plan_task="t99",
        _ctx={"cfg": cfg, "base_url": "http://127.0.0.1:9/v1",
              "model": "m", "project": "p1", "depth": 0})
    data = json.loads(payload)
    assert data.get("status") in ("completed", "ok"), payload
    assert seen["task"].startswith("just do it")
    assert "could not be expanded" in seen["task"]

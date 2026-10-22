"""Feature-wave Phase 2 (delegation) tests — list_agents / delegate_task /
check_delegation.

Covers:
  * list_agents returns project agents with descriptions + mid-run flag
    (in-flight run simulated via the registry).
  * delegate_task happy path: child trace recorded with delegation metadata
    (delegation_id, parent_run_id) + project; id returned; check_delegation
    reports completed with result text.
  * unknown agent -> clean error payload; self-delegation rejected.
  * timeout path: stubbed slow child returns timed_out payload quickly.
  * thread JSON gains a delegations lineage entry; no-thread (CLI) context
    degrades gracefully.
  * guardrails apply inside the child run (same-tool failure spiral halts).
  * KNOWN_TOOLS exposes the three tools; config section + env overrides.
"""

from __future__ import annotations

import json
import sys
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fake_server import FakeOpenAIServer  # noqa: E402
from sable.core import delegation  # noqa: E402
from sable.core.config import SableConfig, load_config  # noqa: E402
from sable.core.tools import TOOL_IMPLS, TOOL_SCHEMAS  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_registry():
    delegation.reset_state()
    yield
    delegation.reset_state()


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "uf-deleg"
    monkeypatch.setenv("SABLE_HOME", str(h))
    return h


def _cfg(profiles=None, project=None, home=None, **kw):
    cfg = SableConfig(home=str(home) if home else None, **kw)
    cfg.profiles = profiles or []
    return cfg


def _make_project(home_dir, agents):
    """Write a <home>/projects/<id>.json the way the dashboard would."""
    d = Path(home_dir) / "projects"
    d.mkdir(parents=True, exist_ok=True)
    pid = "proj-1"
    (d / f"{pid}.json").write_text(json.dumps({
        "id": pid, "name": "wave", "workspace_dir": str(Path.cwd()),
        "agents": agents, "created": "x", "updated": "x"}))
    return pid


def _ctx(cfg, base_url, profile=None, project=None, thread=None, run_id="run-parent"):
    return {"cfg": cfg, "base_url": base_url, "model": "m",
            "profile": profile, "project": project, "thread": thread,
            "run_id": run_id, "depth": 0}


PROFILES = [
    {"name": "researcher", "system_prompt": "Research things.\nBe thorough.",
     "tools": ["read_file"], "model_name": None},
    {"name": "coder", "system_prompt": "Write code.", "description": "Code writer",
     "tools": None, "model_name": None},
]


# ------------------------------------------------------------------ list_agents

def test_list_agents_project_scope_descriptions_and_midrun(home):
    pid = _make_project(home, ["researcher", "coder"])
    cfg = _cfg(PROFILES, home=home)
    # simulate an in-flight coder run (registry is how mid-run is tracked)
    delegation.register_run(profile="coder", project=pid, run_id="r-coder")

    out = json.loads(TOOL_IMPLS["list_agents"](
        _ctx=_ctx(cfg, "http://127.0.0.1:1/v1", profile="researcher",
                  project=pid)))
    assert out["status"] == "ok" and out["project"] == pid
    by_name = {a["agent"]: a for a in out["agents"]}
    assert set(by_name) == {"researcher", "coder"}
    # description: explicit field wins, else first line of system prompt
    assert by_name["coder"]["description"] == "Code writer"
    assert by_name["researcher"]["description"] == "Research things."
    assert by_name["coder"]["running"] is True
    assert by_name["researcher"]["running"] is False
    # scope marker: with a project, only project agents list
    assert out["scope"] == "project agents"

    # no project -> all configured profiles
    out2 = json.loads(TOOL_IMPLS["list_agents"](
        _ctx=_ctx(cfg, "http://127.0.0.1:1/v1")))
    assert {a["agent"] for a in out2["agents"]} == {"researcher", "coder"}


# --------------------------------------------------------------- delegate happy

def test_delegate_task_records_linked_child_trace_and_completes(home):
    pid = _make_project(home, ["coder"])
    cfg = _cfg(PROFILES, home=home)
    with FakeOpenAIServer() as srv:  # scripted read_file -> final answer
        payload = json.loads(TOOL_IMPLS["delegate_task"](
            agent="coder", task="summarize the fake server file",
            _ctx=_ctx(cfg, srv.base_url, profile="researcher", project=pid)))
    assert payload["status"] == "completed", payload
    did = payload["delegation_id"]
    assert did and payload["agent"] == "coder"
    assert "Done" in (payload["result"] or "")
    child_trace = payload["child_trace_id"]
    assert child_trace

    # child trace header carries the delegation linkage + project (v3 column)
    from sable.store.db import TraceStore

    store = TraceStore(Path(home) / "buffer" / "sable.db")
    try:
        row = store.get_trace(child_trace)
    finally:
        store.close()
    assert row is not None and row.project == pid
    header = json.loads(Path(row.jsonl_path).read_text().splitlines()[0])
    meta = header["delegation"]
    assert meta["delegation_id"] == did
    assert meta["parent_run_id"] == "run-parent"
    assert meta["delegated_from"] == "researcher"

    # check_delegation transitions to completed with the result text
    poll = json.loads(TOOL_IMPLS["check_delegation"](id=did))
    assert poll["status"] == "completed"
    assert "Done" in poll["result"] and poll["child_trace_id"] == child_trace


def test_check_delegation_unknown_id(home):
    out = json.loads(TOOL_IMPLS["check_delegation"](id="nope"))
    assert out["status"] == "failed" and "unknown delegation" in out["error"]


# ------------------------------------------------------------- rejection paths

def test_delegate_unknown_agent_clean_error(home):
    cfg = _cfg(PROFILES, home=home)
    out = json.loads(TOOL_IMPLS["delegate_task"](
        agent="ghost", task="x",
        _ctx=_ctx(cfg, "http://127.0.0.1:1/v1", profile="researcher")))
    assert out["status"] == "failed"
    assert "unknown agent" in out["error"] and "ghost" in out["error"]


def test_delegate_to_self_rejected(home):
    cfg = _cfg(PROFILES, home=home)
    out = json.loads(TOOL_IMPLS["delegate_task"](
        agent="coder", task="x",
        _ctx=_ctx(cfg, "http://127.0.0.1:1/v1", profile="coder")))
    assert out["status"] == "failed" and "self" in out["error"]


def test_delegate_agent_outside_project_rejected(home):
    pid = _make_project(home, ["researcher"])
    cfg = _cfg(PROFILES, home=home)
    out = json.loads(TOOL_IMPLS["delegate_task"](
        agent="coder", task="x",
        _ctx=_ctx(cfg, "http://127.0.0.1:1/v1", profile="researcher",
                  project=pid)))
    assert out["status"] == "failed" and "not attached to the active project" in out["error"]


def test_delegate_disabled_by_config(home):
    cfg = _cfg(PROFILES, home=home)
    cfg.delegation_enabled = False
    out = json.loads(TOOL_IMPLS["delegate_task"](
        agent="coder", task="x",
        _ctx=_ctx(cfg, "http://127.0.0.1:1/v1", profile="researcher")))
    assert out["status"] == "failed" and "disabled" in out["error"]


def test_nesting_depth_rejected(home):
    cfg = _cfg(PROFILES, home=home)
    ctx = _ctx(cfg, "http://127.0.0.1:1/v1", profile="researcher")
    ctx["depth"] = 1
    out = json.loads(TOOL_IMPLS["delegate_task"](agent="coder", task="x", _ctx=ctx))
    assert out["status"] == "failed" and "depth" in out["error"]


def test_delegate_without_context_degrades(home):
    # hand-built client (no bound ctx): clean payload, never an exception
    out = json.loads(TOOL_IMPLS["delegate_task"](agent="coder", task="x"))
    assert out["status"] == "failed" and "configuration context" in out["error"]


# -------------------------------------------------------------------- timeout

def test_delegate_timeout_returns_diagnostics_payload(home, monkeypatch):
    cfg = _cfg(PROFILES, home=home)

    def slow_child(cfg_, prof, task, ctx, delegation_id, out):
        time.sleep(5)  # far longer than the 0.3s cap below

    monkeypatch.setattr(delegation, "_run_child", slow_child)
    t0 = time.monotonic()
    with FakeOpenAIServer() as srv:
        payload = json.loads(TOOL_IMPLS["delegate_task"](
            agent="coder", task="take forever", timeout_seconds=0.3,
            _ctx=_ctx(cfg, srv.base_url, profile="researcher")))
    elapsed = time.monotonic() - t0
    assert elapsed < 3.0, "timeout payload must not wait for the child"
    assert payload["status"] == "timed_out"
    assert payload["delegation_id"] and payload["timeout_seconds"] == pytest.approx(0.3)
    assert payload["diagnostics"]["child_agent"] == "coder"
    assert "timed out" in payload["error"]
    # registry keeps the terminal state for check_delegation
    poll = json.loads(TOOL_IMPLS["check_delegation"](id=payload["delegation_id"]))
    assert poll["status"] == "timed_out"


# ---------------------------------------------------------------- lineage

def test_thread_json_gets_delegations_entry(home):
    pid = _make_project(home, ["coder"])
    d = Path(home) / "chat"
    d.mkdir(parents=True, exist_ok=True)
    (d / "t-lin.json").write_text(json.dumps(
        {"id": "t-lin", "title": "lin", "messages": []}))
    cfg = _cfg(PROFILES, home=home)
    with FakeOpenAIServer() as srv:
        payload = json.loads(TOOL_IMPLS["delegate_task"](
            agent="coder", task="read it",
            _ctx=_ctx(cfg, srv.base_url, profile="researcher", project=pid,
                      thread="t-lin")))
    thread = json.loads((d / "t-lin.json").read_text())
    assert len(thread["delegations"]) == 1
    entry = thread["delegations"][0]
    assert entry["agent"] == "coder"
    assert entry["delegation_id"] == payload["delegation_id"]
    assert entry["status"] == "completed"
    assert entry["created"] and entry.get("trace_id") == payload["child_trace_id"]


def test_no_thread_context_degrades_gracefully(home):
    cfg = _cfg(PROFILES, home=home)
    with FakeOpenAIServer() as srv:
        payload = json.loads(TOOL_IMPLS["delegate_task"](
            agent="coder", task="read it",
            _ctx=_ctx(cfg, srv.base_url, profile="researcher")))  # thread=None
    assert payload["status"] == "completed"
    assert not (Path(home) / "chat").exists()  # no lineage files invented


# ---------------------------------------------------------------- guardrails

class AlwaysFailingPatchHandler(BaseHTTPRequestHandler):
    """Child model that keeps issuing patch calls with varying args; the
    real patch impl fails every time (missing files) -> same-tool failure
    spiral. Mirrors test_guardrails.AlwaysFailingPatchHandler."""

    n = 0

    def log_message(self, format, *args):
        pass

    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        AlwaysFailingPatchHandler.n += 1
        msg = {
            "role": "assistant", "content": None,
            "tool_calls": [{
                "id": f"call_{AlwaysFailingPatchHandler.n}", "type": "function",
                "function": {"name": "patch", "arguments": json.dumps({
                    "path": f"missing_{AlwaysFailingPatchHandler.n}.py",
                    "old_string": "x", "new_string": "y"})}}],
        }
        self._send({"id": "c", "object": "chat.completion", "model": "m",
                    "choices": [{"index": 0, "message": msg,
                                 "finish_reason": "tool_calls"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    def do_GET(self):
        self._send({"object": "list", "data": [{"id": "m", "object": "model"}]})


def test_guardrails_apply_to_child_runs(home, monkeypatch):
    AlwaysFailingPatchHandler.n = 0
    profiles = [{"name": "patcher", "system_prompt": "patch things",
                 "tools": ["patch"], "model_name": None}] + PROFILES
    cfg = _cfg(profiles, home=home)
    with FakeOpenAIServer(handler_cls=AlwaysFailingPatchHandler) as srv:
        payload = json.loads(TOOL_IMPLS["delegate_task"](
            agent="patcher", task="patch everything",
            _ctx=_ctx(cfg, srv.base_url, profile="researcher")))
    assert payload["status"] == "failed"
    # the halt came from the child's OWN guardrail controller, not max_iters
    assert payload["failure_reason"] == "guardrail_halt:same_tool_failure_halt"
    assert AlwaysFailingPatchHandler.n < 12


# ------------------------------------------------- registration / config glue

def test_tools_registered_and_in_known_tools():
    from sable.dashboard.app import DashboardApp

    names = {s["function"]["name"] for s in TOOL_SCHEMAS}
    assert {"list_agents", "delegate_task", "check_delegation"} <= names
    for n in ("list_agents", "delegate_task", "check_delegation"):
        assert callable(TOOL_IMPLS[n])
    for t in ("list_agents", "delegate_task", "check_delegation"):
        assert t in DashboardApp.KNOWN_TOOLS


def test_recorder_metadata_and_nullrecorder_passthrough(home):
    from sable.store.recorder import make_recorder

    cfg = _cfg(home=home)
    rec = make_recorder(cfg, project="p9",
                        metadata={"delegation_id": "d1", "parent_run_id": "r1"})
    row = rec.record(task="t", messages=[{"role": "user", "content": "t"}],
                     backend="sable", model_alias="m", base_url="",
                     completed=True)
    header = json.loads(Path(row.jsonl_path).read_text().splitlines()[0])
    assert header["delegation"]["delegation_id"] == "d1"
    assert header["project"] == "p9"
    # plain runs keep the key nullable/absent-safe
    rec2 = make_recorder(cfg)
    row2 = rec2.record(task="t", messages=[{"role": "user", "content": "t"}],
                       backend="sable", model_alias="m", base_url="",
                       completed=True)
    header2 = json.loads(Path(row2.jsonl_path).read_text().splitlines()[0])
    assert header2.get("delegation") is None


def test_config_section_and_env_overrides(tmp_path, monkeypatch):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(
        "delegation:\n  enabled: false\n  timeout_seconds: 42\n")
    monkeypatch.delenv("SABLE_DELEGATION", raising=False)
    monkeypatch.delenv("SABLE_DELEGATION_TIMEOUT_S", raising=False)
    cfg = load_config(str(cfg_file))
    assert cfg.delegation_enabled is False
    assert cfg.delegation_timeout_seconds == 42.0

    monkeypatch.setenv("SABLE_DELEGATION", "true")
    monkeypatch.setenv("SABLE_DELEGATION_TIMEOUT_S", "77")
    cfg2 = load_config(str(cfg_file))
    assert cfg2.delegation_enabled is True
    assert cfg2.delegation_timeout_seconds == 77.0

    defaults = SableConfig()
    assert defaults.delegation_enabled is True
    assert defaults.delegation_timeout_seconds == 300.0


def test_chat_stream_passes_delegation_context_and_thread(home, monkeypatch):
    """h_chat/stream threads {thread, run_id, project...} into create_runner
    and h_thread_put keeps server-written delegations on client saves."""
    from sable.dashboard.app import DashboardApp

    app = DashboardApp(SableConfig(home=home), home=home)
    seen: dict = {}

    class StubRunner:
        name = "stub"

        def __init__(self, cfg, base_url, model, api_key=None, recorder=None,
                     **kwargs):
            seen.update(kwargs.get("delegation_context") or {})
            self.recorder = recorder

        def run(self, task, history=None, previous_summary=None):
            from sable.core.agent_runner import AgentResult
            return AgentResult(final_response="ok", messages=[
                {"role": "user", "content": task},
                {"role": "assistant", "content": "ok"}], backend="stub")

    monkeypatch.setattr("sable.core.agent_runner.create_runner", StubRunner)

    # a serving target is required to reach the runner (attach-mode state,
    # same shape as tests/test_dashboard.py::_fake_serve_state)
    from fake_server import FakeOpenAIServer
    with FakeOpenAIServer() as fs:
        home.mkdir(parents=True, exist_ok=True)
        (home / "serve_state.json").write_text(json.dumps({
            "mode": "attach", "base_url": fs.base_url,
            "model_name": "fake-model",
            "attach_url": fs.base_url.rsplit("/v1", 1)[0]}))
        r = app.handle("POST", "/api/chat/stream",
                       body={"prompt": "go", "thread": "t-ctx"})
        assert r.status == 200 and r.stream is not None
        for _ in r.stream:  # drain the generator
            pass

    assert seen.get("thread") == "t-ctx"
    assert seen.get("run_id") and seen.get("cfg") is not None
    assert seen.get("depth") == 0

    # thread PUT without a delegations key keeps stored lineage
    app.handle("PUT", "/api/chat/threads/t-keep", body={"title": "k"})
    tdir = Path(home) / "chat"
    (tdir / "t-keep.json").write_text(json.dumps({
        "id": "t-keep", "title": "k", "messages": [],
        "delegations": [{"agent": "coder", "delegation_id": "d9",
                         "status": "completed", "created": "x"}]}))
    r = app.handle("PUT", "/api/chat/threads/t-keep", body={"title": "k2"})
    assert r.status == 200
    lst = app.handle("GET", "/api/chat/threads").body["threads"]
    row = next(t for t in lst if t["id"] == "t-keep")
    assert [d["delegation_id"] for d in row["delegations"]] == ["d9"]

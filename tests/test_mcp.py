"""MCP/agent-tooling domain tests.

Consolidated from 11 former test modules; section banners mark origins.
"""

from __future__ import annotations

import sys
from pathlib import Path
import json
import re
from typing import Any, Dict, List, Optional
import time
from http.server import BaseHTTPRequestHandler
import logging
import os
import threading
import stat
import yaml
import sqlite3
from contextlib import contextmanager
from typing import Optional
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from toaster_loop.cli import build_parser  # noqa: E402
from toaster_loop.mcp.confirm import ConfirmStore, args_fingerprint  # noqa: E402
from toaster_loop.mcp.schema import validate_args  # noqa: E402
from toaster_loop.mcp.server import PROTOCOL_VERSION, McpServer, serve_stdio  # noqa: E402
from fake_server import FakeOpenAIServer  # noqa: E402
from toaster_loop.core import delegation  # noqa: E402
from toaster_loop.core.config import ToasterLoopConfig, load_config  # noqa: E402
from toaster_loop.core.tools import TOOL_IMPLS, TOOL_SCHEMAS  # noqa: E402
from toaster_loop.core.config import ToasterLoopConfig  # noqa: E402
from toaster_loop.dashboard.app import DashboardApp  # noqa: E402
from toaster_loop.registry import ModelRegistry  # noqa: E402
from toaster_loop.training.scheduler import DISPATCH_TICK_S, DispatcherAlreadyRunning, JobDispatcher, Scheduler, TrainingConfig, _dispatcher_lock_path  # noqa: E402
from toaster_loop.core.agent_runner import ToasterLoopAgentClient, create_runner  # noqa: E402
from toaster_loop.core.failure_types import is_infra_reason  # noqa: E402
from toaster_loop.core.guardrails import IDEMPOTENT_TOOL_NAMES, MUTATING_TOOL_NAMES, REPETITION_MARKER, ToolCallSignature, ToolLoopGuardrails, detect_repetition, find_loop, truncate_repetition  # noqa: E402
from toaster_loop.core import tools  # noqa: E402
from toaster_loop.core.config import load_config  # noqa: E402
from toaster_loop.core.tools import SYNTAX_FAIL_MARK, SYNTAX_OK_MARK, patch, write_file  # noqa: E402
from toaster_loop.core.presets import ENABLE_ALL_SWITCHES, enable_all_diff, enable_all_patches  # noqa: E402
from toaster_loop.store.db import TraceRow, TraceStore  # noqa: E402
from fake_server import FakeOpenAIServer, make_message_handler  # noqa: E402
from toaster_loop.core.agent_runner import ToasterLoopAgentClient  # noqa: E402
from toaster_loop.core.batch import BatchInputError, checkpoint_path, load_checkpoint, load_tasks, run_batch, scan_completed, task_key  # noqa: E402
from toaster_loop.store.recorder import TraceRecorder  # noqa: E402
from fake_server import FakeOpenAIServer, make_scripted_handler  # noqa: E402
from toaster_loop.curate import CurationConfig  # noqa: E402
from toaster_loop.training.pipeline import run_iterate  # noqa: E402
from toaster_loop.training.scheduler import Scheduler, TrainingConfig  # noqa: E402
from toaster_loop.training.validate import ValidationConfig  # noqa: E402
from toaster_loop.core.verification import VerificationLedger, is_code_path  # noqa: E402
from toaster_loop.evaluation.shadow import ShadowError, no_replay_set_for, run_shadow_eval  # noqa: E402
from toaster_loop.registry.registry import TRANSFERS_CAP  # noqa: E402

# ------------------------------------------------------------------------
# from tests/test_mcp_wp6.py
# ------------------------------------------------------------------------

MCP_SRC = Path(__file__).resolve().parents[1] / "toaster_loop" / "mcp"


# ------------------------------------------------------------------ helpers

class FakeResponse:
    def __init__(self, status: int, body: Any = None):
        self.status = status
        self.body = body


class FakeApp:
    """Stand-in for DashboardApp: records handled calls, canned responses."""

    def __init__(self, home: Path, routes: Optional[Dict[str, FakeResponse]]
                 = None):
        self.home = home
        self.cfg = None  # executors needing cfg are not exercised here
        self.calls: List[tuple] = []
        self.routes = routes or {}

    def handle(self, method: str, path: str,
               body: Optional[Dict[str, Any]] = None) -> FakeResponse:
        self.calls.append((method, path, dict(body or {})))
        key = f"{method} {path}"
        if key in self.routes:
            return self.routes[key]
        return FakeResponse(200, {"ok": True, "echo": key})


def make_server(home: Path, *, routes=None, clock=None, ttl_s=300.0):
    app = FakeApp(home, routes=routes)
    srv = McpServer(app=app, ttl_s=ttl_s, clock=clock)
    return srv, app


def rpc(srv: McpServer, n: int, method: str,
        params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Send one request, parse the response envelope."""
    line = json.dumps({"jsonrpc": "2.0", "id": n, "method": method,
                       **({"params": params} if params is not None else {})})
    raw = srv.feed_line(line)
    assert raw is not None, f"expected a response to {method}"
    return json.loads(raw)


def init(srv: McpServer) -> Dict[str, Any]:
    return rpc(srv, 0, "initialize",
               {"protocolVersion": PROTOCOL_VERSION, "capabilities": {},
                "clientInfo": {"name": "pytest", "version": "0"}})


def call_tool(srv: McpServer, n: int, name: str,
              args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """tools/call and unpack the MCP content envelope."""
    resp = rpc(srv, n, "tools/call",
               {"name": name, "arguments": args or {}})
    if "error" in resp:
        return {"_rpc_error": resp["error"]}
    payload = json.loads(resp["result"]["content"][0]["text"])
    payload["_isError"] = resp["result"]["isError"]
    return payload


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "uf-mcp"
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(h))
    return h


# -------------------------------------------------------------- handshake

def test_initialize_handshake(home):
    srv, _ = make_server(home)
    resp = init(srv)
    result = resp["result"]
    assert result["protocolVersion"] == PROTOCOL_VERSION
    assert result["serverInfo"]["name"] == "toaster_loop"
    assert "tools" in result["capabilities"]


def test_tools_before_initialize_refused(home):
    srv, _ = make_server(home)
    resp = rpc(srv, 1, "tools/list")
    assert resp["error"]["code"] == -32600


def test_tools_list_schema_completeness(home):
    srv, _ = make_server(home)
    init(srv)
    tools = rpc(srv, 2, "tools/list")["result"]["tools"]
    names = {t["name"] for t in tools}
    assert names == {
        "status", "instances_list", "models_list", "logs_tail",
        "instance_start", "instance_stop", "instance_load",
        "instance_unload", "curate_enqueue", "train_enqueue",
        "validate", "promote", "bench",
    }
    for t in tools:
        assert t["name"] and isinstance(t["description"], str)
        assert t["description"]
        schema = t["inputSchema"]
        assert schema["type"] == "object"
        # Fixed schemas: closed property sets, no free-form bags.
        assert schema.get("additionalProperties") is False
        for prop in schema["properties"].values():
            assert prop.get("type") in ("string", "integer", "number",
                                        "boolean")


def test_unknown_method_and_tool(home):
    srv, _ = make_server(home)
    init(srv)
    assert rpc(srv, 3, "bogus/method")["error"]["code"] == -32601
    resp = rpc(srv, 4, "tools/call", {"name": "nope", "arguments": {}})
    assert resp["error"]["code"] == -32601


def test_notification_gets_no_reply(home):
    srv, _ = make_server(home)
    init(srv)
    assert srv.feed_line(json.dumps(
        {"jsonrpc": "2.0", "method": "notifications/initialized"})) is None


# ------------------------------------------------------- schema validation

def test_schema_unknown_arg_rejected(home):
    srv, app = make_server(home)
    init(srv)
    resp = call_tool(srv, 5, "status", {"sudo": True})
    assert "_rpc_error" in resp
    assert resp["_rpc_error"]["code"] == -32602
    assert "unknown argument" in resp["_rpc_error"]["message"]
    assert app.calls == []


def test_schema_missing_required_rejected(home):
    srv, _ = make_server(home)
    init(srv)
    resp = call_tool(srv, 6, "train_enqueue", {"kind": "sft"})
    assert resp["_rpc_error"]["code"] == -32602
    assert "missing required" in resp["_rpc_error"]["message"]


def test_schema_wrong_type_rejected(home):
    srv, _ = make_server(home)
    init(srv)
    # runs is integer; a string must be refused without coercion.
    resp = call_tool(srv, 7, "bench", {"runs": "10"})
    assert resp["_rpc_error"]["code"] == -32602
    assert "integer" in resp["_rpc_error"]["message"]
    # bool is not an integer either.
    resp = call_tool(srv, 8, "bench", {"runs": True})
    assert "_rpc_error" in resp


def test_schema_enum_violation_rejected(home):
    srv, _ = make_server(home)
    init(srv)
    resp = call_tool(srv, 9, "train_enqueue",
                     {"dataset": "d1", "kind": "rlhf"})
    assert resp["_rpc_error"]["code"] == -32602
    assert "must be one of" in resp["_rpc_error"]["message"]


def test_schema_unit_validator():
    schema = {"type": "object", "additionalProperties": False,
              "properties": {"n": {"type": "number"},
                             "s": {"type": "string"}},
              "required": ["n"]}
    assert validate_args(schema, {"n": 3}) is None       # int satisfies number
    assert validate_args(schema, {"n": 3.5}) is None
    assert validate_args(schema, {}) is not None         # missing required
    assert validate_args(schema, {"n": "3"}) is not None  # no str coercion
    assert validate_args(schema, {"n": 1, "x": 2}) is not None


# -------------------------------------------------------- read-only tools

def test_readonly_executes_immediately(home):
    srv, app = make_server(home)
    init(srv)
    out = call_tool(srv, 10, "status")
    assert out["_isError"] is False
    assert out["status"] == "ok"
    assert ("GET", "/api/status", {}) in app.calls


def test_models_and_instances_list(home):
    srv, app = make_server(home)
    init(srv)
    call_tool(srv, 11, "models_list")
    call_tool(srv, 12, "instances_list")
    paths = [c[1] for c in app.calls]
    assert paths == ["/api/models", "/api/servers"]


def test_logs_tail_bounded(home):
    logs = home / "logs"
    logs.mkdir(parents=True)
    (logs / "server-default.log").write_text(
        "\n".join(f"line {i}" for i in range(1, 201)))
    srv, _ = make_server(home)
    init(srv)
    out = call_tool(srv, 13, "logs_tail", {"instance_id": "default",
                                           "lines": 5})
    assert out["_isError"] is False
    body = out["result"]
    assert body["lines"] == ["line 196", "line 197", "line 198", "line 199",
                             "line 200"]
    assert body["total_lines"] == 200


def test_logs_tail_rejects_traversal(home):
    srv, _ = make_server(home)
    init(srv)
    for bad in ("../../etc/passwd", "..", "a/../../b", "../secret"):
        out = call_tool(srv, 14, "logs_tail", {"instance_id": bad})
        assert out["_isError"] is True, f"traversal id {bad!r} accepted"


def test_logs_tail_path_stays_in_log_dir(home):
    # Even a syntactically odd-but-valid-looking id resolves inside logs/.
    logs = home / "logs"
    logs.mkdir(parents=True)
    outside = home / "secret.log"
    outside.write_text("do not read")
    srv, _ = make_server(home)
    init(srv)
    out = call_tool(srv, 15, "logs_tail", {"instance_id": "missing"})
    assert out["_isError"] is True  # no such log; nothing outside read
    assert not any("do not read" in json.dumps(o) for o in [out])


# ------------------------------------------------------- plan/confirm flow

def test_mutating_plan_then_confirm(home):
    srv, app = make_server(home)
    init(srv)
    # Phase 1: plan only — executor must NOT run.
    out = call_tool(srv, 20, "instance_stop", {"id": "alpha"})
    assert out["_isError"] is False
    assert out["status"] == "plan"
    token = out["confirm_token"]
    assert token and isinstance(token, str)
    plan = out["plan"]
    assert plan["action"] == "instance_stop"
    assert plan["target"] == "alpha"
    assert plan["exact_effect"]
    assert plan["risks"]
    assert plan["params"] == {"id": "alpha"}
    assert app.calls == []  # NO side effect executed yet

    # Phase 2: same args + token executes exactly once.
    out2 = call_tool(srv, 21, "instance_stop",
                     {"id": "alpha", "confirm_token": token})
    assert out2["_isError"] is False
    assert out2["status"] == "ok"
    assert app.calls == [("POST", "/api/servers/alpha/stop", {})]

    # Phase 3: token reuse refused (single use).
    out3 = call_tool(srv, 22, "instance_stop",
                     {"id": "alpha", "confirm_token": token})
    assert out3["_isError"] is True
    assert out3["status"] == "refused"
    assert len(app.calls) == 1  # still exactly one execution


def test_confirm_token_expired(home):
    now = [1000.0]
    srv, app = make_server(home, clock=lambda: now[0], ttl_s=60.0)
    init(srv)
    out = call_tool(srv, 30, "promote", {"model_id": "v2"})
    token = out["confirm_token"]
    now[0] += 61.0  # TTL elapses
    out2 = call_tool(srv, 31, "promote",
                     {"model_id": "v2", "confirm_token": token})
    assert out2["_isError"] is True
    assert "expired" in out2["reason"]
    assert app.calls == []


def test_confirm_token_wrong_tool(home):
    srv, app = make_server(home)
    init(srv)
    # Token issued for instance_stop must not execute instance_load...
    out = call_tool(srv, 40, "instance_stop", {"id": "alpha"})
    token = out["confirm_token"]
    out2 = call_tool(srv, 41, "instance_load",
                     {"id": "alpha", "confirm_token": token})
    assert out2["_isError"] is True
    assert "does not match" in out2["reason"]
    # ...nor the same tool with different args.
    out3 = call_tool(srv, 42, "instance_stop",
                     {"id": "beta", "confirm_token": token})
    assert out3["_isError"] is True
    assert app.calls == []


def test_confirm_unknown_token(home):
    srv, _ = make_server(home)
    init(srv)
    out = call_tool(srv, 50, "promote",
                    {"model_id": "v1", "confirm_token": "hunter2"})
    assert out["_isError"] is True
    assert out["status"] == "refused"


def test_confirm_store_unit():
    now = [0.0]
    store = ConfirmStore(ttl_s=10.0, clock=lambda: now[0])
    tok = store.issue("promote", {"model_id": "v1"})
    assert store.consume(tok, "promote", {"model_id": "v1"}) is None
    # single use
    assert store.consume(tok, "promote", {"model_id": "v1"}) is not None
    # binding ignores dict ordering (canonical fingerprint)
    tok2 = store.issue("t", {"a": 1, "b": 2})
    assert store.consume(tok2, "t", {"b": 2, "a": 1}) is None
    assert args_fingerprint("t", {"a": 1}) == args_fingerprint("t", {"a": 1})
    assert args_fingerprint("t", {"a": 1}) != args_fingerprint("u", {"a": 1})


def test_train_enqueue_conflict_surfaces_as_error(home, monkeypatch):
    """Single-flight 409 (the existing dashboard conflict) => tool error.

    DashboardApp.handle converts HttpError(409) into an error-envelope
    ApiResponse; the fake reproduces exactly that contract for the train
    route (submit_train raises HttpError 409 already_running while busy).
    """
    from toaster_loop.dashboard.app import DashboardApp

    a = DashboardApp.__new__(DashboardApp)  # do not construct threads etc.

    def fake_handle(method, path, query=None, body=None):
        if method == "POST" and path == "/api/train":
            return FakeResponse(409, {"error": {
                "code": "already_running",
                "message": "job job-123 is already running (sft); "
                           "one GPU, one job"}})
        return FakeResponse(200, {"ok": True})

    a.handle = fake_handle  # type: ignore[assignment]
    a.home = home
    srv = McpServer(app=a)
    init(srv)
    out = call_tool(srv, 60, "train_enqueue", {"dataset": "d1"})
    token = out["confirm_token"]
    out2 = call_tool(srv, 61, "train_enqueue",
                     {"dataset": "d1", "confirm_token": token})
    assert out2["_isError"] is True
    assert out2["status"] == "error"
    assert "already running" in out2["reason"]
    assert "409" in out2["reason"]


def test_curate_enqueue_queues_via_api(home):
    srv, app = make_server(home)
    init(srv)
    out = call_tool(srv, 70, "curate_enqueue", {"name": "mcpset"})
    token = out["confirm_token"]
    out2 = call_tool(srv, 71, "curate_enqueue",
                     {"name": "mcpset", "confirm_token": token})
    assert out2["_isError"] is False
    assert ("POST", "/api/curate", {"name": "mcpset"}) in app.calls


def test_curate_enqueue_project_passthrough(home):
    """WP12: curate_enqueue accepts the optional project id filter (mirrors
    `toaster_loop curate --project`) and forwards it to POST /api/curate."""
    srv, app = make_server(home)
    init(srv)
    out = call_tool(srv, 78, "curate_enqueue",
                    {"name": "projset", "project": "alpha123"})
    assert out["status"] == "plan"
    token = out["confirm_token"]
    out2 = call_tool(srv, 79, "curate_enqueue",
                     {"name": "projset", "project": "alpha123",
                      "confirm_token": token})
    assert out2["_isError"] is False
    assert ("POST", "/api/curate",
            {"name": "projset", "project": "alpha123"}) in app.calls


# ---------------------------------------------------------- no-shell rules

def test_no_shell_execution_anywhere_in_mcp():
    """No subprocess/shell/os.system anywhere in toaster_loop/mcp/**."""
    offenders = []
    pat = re.compile(r"shell\s*=\s*True|os\.system|subprocess|\bos\.popen\b|"
                     r"pty\.spawn|eval\(|exec\(")
    for py in sorted(MCP_SRC.rglob("*.py")):
        for i, line in enumerate(py.read_text().splitlines(), 1):
            m = pat.search(line)
            if m:
                offenders.append(f"{py.name}:{i}: {line.strip()!r}")
    assert not offenders, f"shell/exec patterns found: {offenders}"


def test_mcp_package_is_stdlib_only():
    """The MCP package imports only stdlib + toaster_loop (no MCP SDK)."""
    import ast

    allowed_roots = set(sys.stdlib_module_names) | {"toaster_loop", "yaml", "httpx"}
    offenders = []
    for py in sorted(MCP_SRC.rglob("*.py")):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module \
                    and node.level == 0:
                names = [node.module]
            if not names:
                continue
            assert isinstance(node, (ast.Import, ast.ImportFrom))
            for n in names:
                if n.split(".")[0] not in allowed_roots:
                    offenders.append(f"{py.name}:{node.lineno}: {n}")
    assert not offenders, f"non-stdlib imports: {offenders}"


# ------------------------------------------------------------- CLI wiring

def test_cli_mcp_help():
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["mcp", "--help"])
    assert exc.value.code == 0


def test_stdio_loop_survives_malformed_json(home):
    """serve_stdio over pipes: parse error response, then clean EOF exit 0."""
    import io

    stdin = io.StringIO("this is { not json\n"
                        '{"jsonrpc": "2.0", "id": 1, "method": "ping"}\n')
    stdout = io.StringIO()
    rc = serve_stdio(stdin=stdin, stdout=stdout)
    assert rc == 0
    lines = [json.loads(l) for l in stdout.getvalue().splitlines() if l]
    assert lines[0]["error"]["code"] == -32700
    # ping is answered even before initialize (liveness method).
    assert lines[1]["result"] == {}


def test_stdio_banishes_nothing_to_stdout_on_diagnostics(home, capsys):
    import io

    stdin = io.StringIO('{"jsonrpc": "2.0", "id": 1, "method": "ping"}\n')
    stdout = io.StringIO()
    serve_stdio(stdin=stdin, stdout=stdout)
    # The banner went to stderr (capsys), stdout holds exactly one JSON line.
    assert len(stdout.getvalue().strip().splitlines()) == 1
    assert "toaster_loop mcp" in capsys.readouterr().err

# ------------------------------------------------------------------------
# from tests/test_delegation.py
# ------------------------------------------------------------------------

@pytest.fixture()
def _clean_registry():
    delegation.reset_state()
    yield
    delegation.reset_state()


@pytest.fixture()
def home__test_delegation(tmp_path, monkeypatch):
    h = tmp_path / "uf-deleg"
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(h))
    return h


def _cfg(profiles=None, project=None, home=None, **kw):
    cfg = ToasterLoopConfig(home=str(home) if home else None, **kw)
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

def test_list_agents_project_scope_descriptions_and_midrun(home__test_delegation, _clean_registry):
    pid = _make_project(home__test_delegation, ["researcher", "coder"])
    cfg = _cfg(PROFILES, home=home__test_delegation)
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

def test_delegate_task_records_linked_child_trace_and_completes(home__test_delegation, _clean_registry):
    pid = _make_project(home__test_delegation, ["coder"])
    cfg = _cfg(PROFILES, home=home__test_delegation)
    with FakeOpenAIServer() as srv:  # scripted read_file -> final answer
        payload = json.loads(TOOL_IMPLS["delegate_task"](
            agent="coder", task="summarize the fake server file", wait=True,
            _ctx=_ctx(cfg, srv.base_url, profile="researcher", project=pid)))
    assert payload["status"] == "completed", payload
    did = payload["delegation_id"]
    assert did and payload["agent"] == "coder"
    assert "Done" in (payload["result"] or "")
    child_trace = payload["child_trace_id"]
    assert child_trace

    # child trace header carries the delegation linkage + project (v3 column)
    from toaster_loop.store.db import TraceStore

    store = TraceStore(Path(home__test_delegation) / "buffer" / "toaster_loop.db")
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


def test_delegate_child_trace_stamped_with_profile_in_easy_loop(home__test_delegation, _clean_registry):
    """Audit §3: when easy training covers the CHILD profile, its trace is
    stamped agent_profile=<child name> (not the backend "toaster_loop") so scoped
    curation/readiness count it — same rule as the dashboard chat path."""
    pid = _make_project(home__test_delegation, ["coder"])
    cfg = _cfg(PROFILES, home=home__test_delegation)
    cfg.easy_training = {"enabled": True, "profiles": ["researcher", "coder"]}
    with FakeOpenAIServer() as srv:
        payload = json.loads(TOOL_IMPLS["delegate_task"](
            agent="coder", task="do the easy thing", wait=True,
            _ctx=_ctx(cfg, srv.base_url, profile="researcher", project=pid)))
    assert payload["status"] == "completed", payload

    from toaster_loop.store.db import TraceStore

    store = TraceStore(Path(home__test_delegation) / "buffer" / "toaster_loop.db")
    try:
        row = store.get_trace(payload["child_trace_id"])
    finally:
        store.close()
    assert row is not None and row.agent_profile == "coder"


def test_delegate_child_trace_keeps_backend_when_profile_outside_easy_loop(home__test_delegation, _clean_registry):
    """Easy training enabled but the child profile NOT in all_profiles keeps the
    legacy backend stamp (chat-parity: profile= only when covered)."""
    pid = _make_project(home__test_delegation, ["coder"])
    cfg = _cfg(PROFILES, home=home__test_delegation)
    cfg.easy_training = {"enabled": True, "profile": "someone-else"}
    with FakeOpenAIServer() as srv:
        payload = json.loads(TOOL_IMPLS["delegate_task"](
            agent="coder", task="do the thing", wait=True,
            _ctx=_ctx(cfg, srv.base_url, profile="researcher", project=pid)))
    assert payload["status"] == "completed", payload

    from toaster_loop.store.db import TraceStore

    store = TraceStore(Path(home__test_delegation) / "buffer" / "toaster_loop.db")
    try:
        row = store.get_trace(payload["child_trace_id"])
    finally:
        store.close()
    assert row is not None and row.agent_profile == "toaster_loop"


def test_run_child_passes_recorder_profile(monkeypatch, home__test_delegation):
    """_run_child hands make_recorder the child profile name iff easy training
    covers it (spy on make_recorder; runner creation short-circuits after)."""
    seen = {}

    def spy_recorder(cfg, **kw):
        seen["profile"] = kw.get("profile")
        raise RuntimeError("stop-after-recorder")

    monkeypatch.setattr("toaster_loop.store.recorder.make_recorder", spy_recorder)
    cfg = _cfg(PROFILES, home=home__test_delegation)
    cfg.easy_training = {"enabled": True, "profile": "coder"}
    with pytest.raises(RuntimeError, match="stop-after-recorder"):
        delegation._run_child(cfg, {"name": "coder"}, "task",
                              {"base_url": "u"}, "dlg1", {})
    assert seen["profile"] == "coder"

    # easy disabled -> no profile stamp (legacy backend behavior)
    seen.clear()
    cfg.easy_training = {}
    with pytest.raises(RuntimeError, match="stop-after-recorder"):
        delegation._run_child(cfg, {"name": "coder"}, "task",
                              {"base_url": "u"}, "dlg1", {})
    assert seen["profile"] is None


# ------------------------------------------------------- remote peer delegation
# Audit §1 / D-16: `delegate_task("tim@host")` resolves through the peers
# registry; the child loop runs HERE with inference on the peer's gateway.

REMOTE_PROF = {"name": "tim@192.0.2.9",
               "system_prompt": "You are Tim on the render node.",
               "model_name": "Qwen3-27B", "remote": True,
               "peer_url": "http://192.0.2.9:55429", "peer_token": "peer-tok",
               "remote_instance": "tim"}


def _fake_run_child(seen):
    def fake(cfg_, prof, task, ctx, delegation_id, out, plan_task=None,
             cancel_event=None):
        seen["prof"] = prof
        out.update({"completed": True, "result": "ok",
                    "child_trace_id": None, "failure_reason": None})
    return fake


def test_delegate_peer_name_resolves_via_peers(home__test_delegation,
                                               monkeypatch, _clean_registry):
    """Local profile miss + `@` name -> peers.resolve_remote_agent (not the
    plain unknown-agent error); healthcheck passes; child runs."""
    seen = {}
    calls = []

    def fake_resolve(home_, full_name):
        calls.append(full_name)
        return dict(REMOTE_PROF)

    monkeypatch.setattr("toaster_loop.dashboard.peers.resolve_remote_agent",
                        fake_resolve)
    monkeypatch.setattr("toaster_loop.serving.supervisor.healthcheck",
                        lambda *a, **k: True)
    monkeypatch.setattr(delegation, "_run_child", _fake_run_child(seen))
    cfg = _cfg(PROFILES, home=home__test_delegation)
    payload = json.loads(delegation.execute_delegation(
        "tim@192.0.2.9", "render the frame", wait=True,
        _ctx=_ctx(cfg, "http://127.0.0.1:1/v1", profile="researcher")))
    assert payload["status"] == "completed", payload
    assert calls == ["tim@192.0.2.9"]
    assert seen["prof"]["remote"] is True
    # child trace attribution stays the FULL peer name (§3 stamping)
    assert seen["prof"]["name"] == "tim@192.0.2.9"

    # no-@ unknown keeps the exact legacy message (unchanged error path)
    out = json.loads(delegation.execute_delegation(
        "nobody", "x", _ctx=_ctx(cfg, "http://127.0.0.1:1/v1")))
    assert out["status"] == "failed" and "unknown agent 'nobody'" in out["error"]
    assert "peer offline" not in out["error"]


def test_delegate_peer_unreachable_error_leaves_no_record(
        home__test_delegation, monkeypatch, _clean_registry):
    """Resolved peer that fails the gateway healthcheck -> peer_unreachable
    payload with chat's not-serving sentence, and NO phantom registry record."""
    monkeypatch.setattr("toaster_loop.dashboard.peers.resolve_remote_agent",
                        lambda h, n: dict(REMOTE_PROF))
    monkeypatch.setattr("toaster_loop.serving.supervisor.healthcheck",
                        lambda *a, **k: False)
    cfg = _cfg(PROFILES, home=home__test_delegation)
    payload = json.loads(delegation.execute_delegation(
        "tim@192.0.2.9", "render", _ctx=_ctx(cfg, "http://127.0.0.1:1/v1")))
    assert payload["status"] == "failed"
    assert payload["code"] == "peer_unreachable"
    assert "remote toaster_loop at 192.0.2.9:55429 is not serving right now" \
        in payload["error"]
    assert len(delegation._delegations) == 0  # no phantom running record

    # unresolvable peer name (offline / never probed / agent gone)
    monkeypatch.setattr("toaster_loop.dashboard.peers.resolve_remote_agent",
                        lambda h, n: None)
    out = json.loads(delegation.execute_delegation(
        "tim@192.0.2.9", "render", _ctx=_ctx(cfg, "http://127.0.0.1:1/v1")))
    assert out["code"] == "peer_unreachable"
    assert "peer offline, never probed, or no such agent there" \
        in out["error"]
    assert len(delegation._delegations) == 0


# ------------------------------------------------- audit §6: pruned-eviction ring

def _settled_record(did: str, **over) -> Dict[str, Any]:
    rec = {"delegation_id": did, "agent": "coder", "task": "t",
           "status": "completed", "created": "2026-01-01T00:00:00+00:00",
           "project": None, "thread": None, "parent_run_id": "r",
           "timeout_seconds": 300.0, "started_mono": 0.0,
           "result": "the answer", "child_trace_id": f"tr-{did}",
           "error": None, "last_activity_ts": "2026-01-01T00:00:05+00:00",
           "cancel_event": threading.Event(), "_settled": True}
    rec.update(over)
    return rec


def test_check_delegation_pruned_eviction_returns_summary(
        home__test_delegation, _clean_registry):
    """Audit §6: after the FIFO cap evicts a SETTLED record, check_delegation
    answers with its terminal summary + a (pruned) marker instead of the
    confusing 'unknown delegation id'."""
    old = _settled_record("old0001")
    delegation._store_delegation(old)
    # fill to exactly the cap so the next insert evicts `old` first
    n_rest = delegation._DELEGATIONS_CAP - 1
    for i in range(n_rest):
        delegation._store_delegation(_settled_record(f"fill{i:05d}"))
    assert len(delegation._delegations) == delegation._DELEGATIONS_CAP
    delegation._store_delegation(_settled_record("newest1"))

    assert "old0001" not in delegation._delegations
    poll = json.loads(TOOL_IMPLS["check_delegation"](id="old0001"))
    assert poll["status"] == "completed"
    assert poll["agent"] == "coder"
    assert poll["result"] == "the answer"
    assert poll["child_trace_id"] == "tr-old0001"
    assert poll.get("pruned") is True
    assert "pruned from the in-memory registry" in poll["note"]
    assert "Traces store" in poll["note"]

    # an id this process never saw keeps the legacy unknown-id error
    unk = json.loads(TOOL_IMPLS["check_delegation"](id="never-seen"))
    assert unk["status"] == "failed" and "unknown delegation id" in unk["error"]


def test_check_delegation_pruned_eviction_skips_unsettled_records(
        home__test_delegation, _clean_registry):
    """Only SETTLED evictions enter the ring — a still-running record that got
    evicted has no trustworthy summary, so the unknown-id error stays."""
    running = _settled_record("run00001", status="running", result=None,
                              child_trace_id=None, error=None,
                              _settled=False)
    delegation._store_delegation(running)
    for i in range(delegation._DELEGATIONS_CAP - 1):
        delegation._store_delegation(_settled_record(f"fill{i:05d}"))
    delegation._store_delegation(_settled_record("newest1"))

    assert "run00001" not in delegation._delegations
    unk = json.loads(TOOL_IMPLS["check_delegation"](id="run00001"))
    assert unk["status"] == "failed" and "unknown delegation id" in unk["error"]


# ------------------------------------------- audit §6: schema + timeout floor

def test_delegate_schema_mentions_depth_limit(_clean_registry):
    """Audit §6: the depth-1 limit must be visible to the model in the tool
    description, not only in the runtime error."""
    schema = next(s for s in TOOL_SCHEMAS
                  if s["function"]["name"] == "delegate_task")
    desc = schema["function"]["description"]
    assert "depth limit is 1" in desc
    assert "cannot itself delegate" in desc


def test_delegate_has_no_timeout_knobs():
    """D-18: the timeout machinery (clamps, config knob, tool arg) is gone —
    children run as long as they need."""
    assert not hasattr(delegation, "_resolve_timeout")
    assert not hasattr(delegation, "MIN_TIMEOUT_S")
    assert not hasattr(delegation, "MAX_TIMEOUT_S")
    schema = next(s for s in TOOL_SCHEMAS
                  if s["function"]["name"] == "delegate_task")
    assert "timeout_seconds" not in schema["function"]["parameters"]["properties"]


# --------------------------------------- audit §6: list_agents peer busy state

def test_list_agents_shows_peer_serving_state(home__test_delegation,
                                              _clean_registry):
    """Peer agent lines carry a serving state built ONLY from cached probe
    snapshots in peers.json (no network calls): serving / idle / unknown."""
    from toaster_loop.dashboard import peers as peers_mod

    home = home__test_delegation
    pid = _make_project(home, ["coder", "tim@192.0.2.9", "ana@192.0.2.10",
                               "zed@192.0.2.11"])
    # healthy + serving (one running instance)
    rec_serving = peers_mod.upsert_peer(home, "http://192.0.2.9:55429", None)
    peers_mod.set_snapshot(home, rec_serving["id"], {
        "ok": True, "probed_at": "2026-01-01T00:00:00+00:00",
        "servers": [{"name": "s1", "running": True, "model_name": "m"}],
        "agents": [{"name": "tim", "system_prompt": "You are Tim."}],
        "server": {"running": True}})
    # healthy but idle (no running instance)
    rec_idle = peers_mod.upsert_peer(home, "http://192.0.2.10:8765", None)
    peers_mod.set_snapshot(home, rec_idle["id"], {
        "ok": True, "probed_at": "2026-01-01T00:00:00+00:00",
        "servers": [{"name": "s1", "running": False, "model_name": None}],
        "agents": [{"name": "ana", "system_prompt": "You are Ana."}],
        "server": {"running": False}})
    # remembered but never successfully probed -> unknown
    peers_mod.upsert_peer(home, "http://192.0.2.11:8765", None)

    cfg = _cfg(PROFILES, home=home)
    out = json.loads(TOOL_IMPLS["list_agents"](
        _ctx=_ctx(cfg, "http://127.0.0.1:1/v1", project=pid)))
    by_name = {a["agent"]: a for a in out["agents"]}
    assert by_name["tim@192.0.2.9"]["peer_serving"] == "serving"
    assert by_name["ana@192.0.2.10"]["peer_serving"] == "idle"
    assert by_name["zed@192.0.2.11"]["peer_serving"] == "unknown"
    # local profiles carry no peer_serving key at all
    assert "peer_serving" not in by_name["coder"]


def test_run_child_remote_wiring(home__test_delegation, monkeypatch):
    """_run_child overrides the parent endpoint for remote profiles: peer
    gateway base_url / token / model + X-Toaster-Loop-Instance header."""
    from toaster_loop.core.agent_runner import AgentResult

    captured = {}

    class StubRunner:
        def __init__(self, cfg, base_url, model, api_key=None, **kwargs):
            captured["base_url"] = base_url
            captured["model"] = model
            captured["api_key"] = api_key
            captured["extra_headers"] = kwargs.get("extra_headers")

        def run(self, task, history=None, **kw):
            return AgentResult(final_response="ok", messages=[
                {"role": "user", "content": task},
                {"role": "assistant", "content": "ok"}],
                backend="stub", tokens_in=1, tokens_out=1)

    monkeypatch.setattr("toaster_loop.core.agent_runner.create_runner", StubRunner)
    pushed = []
    monkeypatch.setattr(
        "toaster_loop.dashboard.peers.push_remote_trace",
        lambda prof, task, res, t0: pushed.append((prof["name"], task)))

    cfg = _cfg(PROFILES, home=home__test_delegation)
    out = {}
    delegation._run_child(cfg, dict(REMOTE_PROF), "render",
                          {"base_url": "http://127.0.0.1:9/v1",
                           "model": "parent-model"}, "dlg-remote", out)
    assert captured["base_url"] == "http://192.0.2.9:55429/remote/v1"
    assert captured["api_key"] == "peer-tok"
    assert captured["model"] == "Qwen3-27B"
    assert captured["extra_headers"] == {"X-Toaster-Loop-Instance": "tim"}
    assert out["completed"] is True
    # trace flow-back fired once for the remote child
    assert pushed == [("tim@192.0.2.9", "render")]


def test_run_child_local_wiring_unchanged(home__test_delegation, monkeypatch):
    """Local children keep the parent-inherited endpoint and never push."""
    from toaster_loop.core.agent_runner import AgentResult

    captured = {}

    class StubRunner:
        def __init__(self, cfg, base_url, model, api_key=None, **kwargs):
            captured["base_url"] = base_url
            captured["extra_headers"] = kwargs.get("extra_headers")

        def run(self, task, history=None, **kw):
            return AgentResult(final_response="ok", messages=[],
                               backend="stub", tokens_in=1, tokens_out=1)

    monkeypatch.setattr("toaster_loop.core.agent_runner.create_runner", StubRunner)
    pushed = []
    monkeypatch.setattr(
        "toaster_loop.dashboard.peers.push_remote_trace",
        lambda *a, **k: pushed.append(a))
    cfg = _cfg(PROFILES, home=home__test_delegation)
    delegation._run_child(cfg, {"name": "coder"}, "task",
                          {"base_url": "http://127.0.0.1:9/v1",
                           "model": "m"}, "dlg-local", {})
    assert captured["base_url"] == "http://127.0.0.1:9/v1"
    assert captured["extra_headers"] is None
    assert pushed == []  # never for local children


def test_run_child_flowback_failure_never_breaks_child(home__test_delegation,
                                                       monkeypatch):
    """A raising trace push leaves the child's terminal state untouched."""
    from toaster_loop.core.agent_runner import AgentResult

    class StubRunner:
        def __init__(self, cfg, base_url, model, api_key=None, **kwargs):
            pass

        def run(self, task, history=None, **kw):
            return AgentResult(final_response="done", messages=[],
                               backend="stub", tokens_in=1, tokens_out=1)

    def boom(*a, **k):
        raise RuntimeError("peer is down")

    monkeypatch.setattr("toaster_loop.core.agent_runner.create_runner", StubRunner)
    monkeypatch.setattr("toaster_loop.dashboard.peers.push_remote_trace", boom)
    cfg = _cfg(PROFILES, home=home__test_delegation)
    out = {}
    delegation._run_child(cfg, dict(REMOTE_PROF), "render",
                          {"base_url": "http://127.0.0.1:9/v1"}, "dlg-x", out)
    assert out["completed"] is True and out["result"] == "done"


def test_check_delegation_unknown_id(home__test_delegation, _clean_registry):
    out = json.loads(TOOL_IMPLS["check_delegation"](id="nope"))
    assert out["status"] == "failed" and "unknown delegation" in out["error"]


# ------------------------------------------------------------- rejection paths

def test_delegate_unknown_agent_clean_error(home__test_delegation, _clean_registry):
    cfg = _cfg(PROFILES, home=home__test_delegation)
    out = json.loads(TOOL_IMPLS["delegate_task"](
        agent="ghost", task="x",
        _ctx=_ctx(cfg, "http://127.0.0.1:1/v1", profile="researcher")))
    assert out["status"] == "failed"
    assert "unknown agent" in out["error"] and "ghost" in out["error"]


def test_delegate_to_self_rejected(home__test_delegation, _clean_registry):
    cfg = _cfg(PROFILES, home=home__test_delegation)
    out = json.loads(TOOL_IMPLS["delegate_task"](
        agent="coder", task="x",
        _ctx=_ctx(cfg, "http://127.0.0.1:1/v1", profile="coder")))
    assert out["status"] == "failed" and "self" in out["error"]


def test_delegate_agent_outside_project_rejected(home__test_delegation, _clean_registry):
    pid = _make_project(home__test_delegation, ["researcher"])
    cfg = _cfg(PROFILES, home=home__test_delegation)
    out = json.loads(TOOL_IMPLS["delegate_task"](
        agent="coder", task="x",
        _ctx=_ctx(cfg, "http://127.0.0.1:1/v1", profile="researcher",
                  project=pid)))
    assert out["status"] == "failed" and "not attached to the active project" in out["error"]


def test_delegate_disabled_by_config(home__test_delegation, _clean_registry):
    cfg = _cfg(PROFILES, home=home__test_delegation)
    cfg.delegation_enabled = False
    out = json.loads(TOOL_IMPLS["delegate_task"](
        agent="coder", task="x",
        _ctx=_ctx(cfg, "http://127.0.0.1:1/v1", profile="researcher")))
    assert out["status"] == "failed" and "disabled" in out["error"]


def test_nesting_depth_rejected(home__test_delegation, _clean_registry):
    cfg = _cfg(PROFILES, home=home__test_delegation)
    ctx = _ctx(cfg, "http://127.0.0.1:1/v1", profile="researcher")
    ctx["depth"] = 1
    out = json.loads(TOOL_IMPLS["delegate_task"](agent="coder", task="x", _ctx=ctx))
    assert out["status"] == "failed" and "depth" in out["error"]


def test_delegate_without_context_degrades(home__test_delegation, _clean_registry):
    # hand-built client (no bound ctx): clean payload, never an exception
    out = json.loads(TOOL_IMPLS["delegate_task"](agent="coder", task="x"))
    assert out["status"] == "failed" and "configuration context" in out["error"]


def _cfg_ctx(cfg, base_url):
    return _ctx(cfg, base_url, profile="researcher")


def test_delegate_wait_blocks_until_slow_child_settles(
        home__test_delegation, monkeypatch, _clean_registry):
    """D-18: wait=true has NO deadline — a slow child finishes on its own and
    the parent returns its real terminal state (no timed_out payload)."""
    cfg = _cfg(PROFILES, home=home__test_delegation)

    def slow_then_done(cfg_, prof, task, ctx, delegation_id, out,
                       plan_task=None, cancel_event=None):
        time.sleep(1.0)  # any duration: nothing cuts the wait short anymore
        out["completed"] = True
        out["result"] = "slow but done"
        out["child_trace_id"] = "tr-late"

    monkeypatch.setattr(delegation, "_run_child", slow_then_done)
    with FakeOpenAIServer() as srv:
        payload = json.loads(TOOL_IMPLS["delegate_task"](
            agent="coder", task="finish late", wait=True,
            _ctx=_cfg_ctx(cfg, srv.base_url)))
    assert payload["status"] == "completed", payload
    assert "slow but done" in payload["result"]
    assert payload["child_trace_id"] == "tr-late"
    poll = json.loads(TOOL_IMPLS["check_delegation"](id=payload["delegation_id"]))
    assert poll["status"] == "completed", poll


def test_delegate_parent_cancel_stops_child_and_settles_cancelled(
        home__test_delegation, monkeypatch, _clean_registry):
    """Audit §2 bug B: the parent run's cancel Event (chat Stop) flips the
    delegation's cooperative cancel; the child observes it, stops looping and
    the record settles to terminal state 'cancelled'."""
    cfg = _cfg(PROFILES, home=home__test_delegation)
    seen = {"loops": 0}

    def looping_child(cfg_, prof, task, ctx, delegation_id, out,
                      plan_task=None, cancel_event=None):
        # Emulates a runner polling cancel_event at loop boundaries: exits
        # with failure_reason="cancelled" exactly like ToasterLoopAgentClient does.
        for _ in range(200):  # cap: never outlive the test
            seen["loops"] += 1
            if cancel_event is not None and cancel_event.is_set():
                from toaster_loop.core.failure_types import FAILURE_CANCELLED
                out["completed"] = False
                out["failure_reason"] = FAILURE_CANCELLED
                out["result"] = "(stopped: cancelled by user)"
                return
            time.sleep(0.02)
        out["completed"] = False
        out["failure_reason"] = "test-child-never-cancelled"

    monkeypatch.setattr(delegation, "_run_child", looping_child)
    parent_cancel = threading.Event()
    ctxd = _ctx(cfg, "http://127.0.0.1:1/v1", profile="researcher")
    ctxd["cancel_event"] = parent_cancel

    result: Dict[str, Any] = {}

    def run_delegate():
        result["payload"] = json.loads(TOOL_IMPLS["delegate_task"](
            agent="coder", task="loop forever", wait=True,
            _ctx=ctxd))

    t = threading.Thread(target=run_delegate, daemon=True)
    t.start()
    time.sleep(0.4)          # let the child start looping
    parent_cancel.set()      # chat Stop -> parent run cancelled
    t.join(timeout=5)
    assert not t.is_alive(), "execute_delegation must return promptly on cancel"

    payload = result["payload"]
    assert payload["status"] == "cancelled", payload
    did = payload["delegation_id"]

    # The record settles to cancelled once the child notices the event.
    deadline = time.monotonic() + 4.0
    poll: Dict[str, Any] = {}
    while time.monotonic() < deadline:
        poll = json.loads(TOOL_IMPLS["check_delegation"](id=did))
        if poll["status"] == "cancelled":
            break
        time.sleep(0.05)
    assert poll["status"] == "cancelled", poll
    assert "cancel" in (poll.get("error") or "").lower()

    # Child actually stopped looping promptly after the cancel was set.
    loops_at_stop = seen["loops"]
    time.sleep(0.3)
    assert seen["loops"] - loops_at_stop <= 2, "child must stop on cancel"


def test_cancel_delegation_stops_running_child(home__test_delegation,
                                               monkeypatch, _clean_registry):
    """Audit §2 bug B: the registry-level cancel_delegation(id) hook sets the
    record's event; a running child cooperatively terminates 'cancelled'."""
    cfg = _cfg(PROFILES, home=home__test_delegation)

    def looping_child(cfg_, prof, task, ctx, delegation_id, out,
                      plan_task=None, cancel_event=None):
        for _ in range(200):
            if cancel_event is not None and cancel_event.is_set():
                from toaster_loop.core.failure_types import FAILURE_CANCELLED
                out["completed"] = False
                out["failure_reason"] = FAILURE_CANCELLED
                return
            time.sleep(0.02)
        out["completed"] = False
        out["failure_reason"] = "test-child-never-cancelled"

    monkeypatch.setattr(delegation, "_run_child", looping_child)

    result: Dict[str, Any] = {}

    def run_delegate():
        result["payload"] = json.loads(TOOL_IMPLS["delegate_task"](
            agent="coder", task="loop forever", wait=True,
            _ctx=_cfg_ctx(cfg, "http://127.0.0.1:1/v1")))

    t = threading.Thread(target=run_delegate, daemon=True)
    t.start()
    # find the record once it exists
    did = ""
    deadline = time.monotonic() + 4.0
    while time.monotonic() < deadline and not did:
        with delegation._registry_lock:
            if delegation._delegations:
                did = next(iter(delegation._delegations))
        time.sleep(0.02)
    assert did, "delegation record should exist while the child runs"

    assert delegation.cancel_delegation(did) is True
    t.join(timeout=5)
    assert not t.is_alive()
    assert result["payload"]["status"] == "cancelled", result["payload"]

    poll = json.loads(TOOL_IMPLS["check_delegation"](id=did))
    assert poll["status"] == "cancelled", poll
    # already-settled ids report False (no double cancel)
    assert delegation.cancel_delegation(did) is False


# ---------------------------------------------------------------- lineage

def test_thread_json_gets_delegations_entry(home__test_delegation, _clean_registry):
    pid = _make_project(home__test_delegation, ["coder"])
    d = Path(home__test_delegation) / "chat"
    d.mkdir(parents=True, exist_ok=True)
    (d / "t-lin.json").write_text(json.dumps(
        {"id": "t-lin", "title": "lin", "messages": []}))
    cfg = _cfg(PROFILES, home=home__test_delegation)
    with FakeOpenAIServer() as srv:
        payload = json.loads(TOOL_IMPLS["delegate_task"](
            agent="coder", task="read it", wait=True,
            _ctx=_ctx(cfg, srv.base_url, profile="researcher", project=pid,
                      thread="t-lin")))
    thread = json.loads((d / "t-lin.json").read_text())
    assert len(thread["delegations"]) == 1
    entry = thread["delegations"][0]
    assert entry["agent"] == "coder"
    assert entry["delegation_id"] == payload["delegation_id"]
    assert entry["status"] == "completed"
    assert entry["created"] and entry.get("trace_id") == payload["child_trace_id"]


def test_no_thread_context_degrades_gracefully(home__test_delegation, _clean_registry):
    cfg = _cfg(PROFILES, home=home__test_delegation)
    with FakeOpenAIServer() as srv:
        payload = json.loads(TOOL_IMPLS["delegate_task"](
            agent="coder", task="read it", wait=True,
            _ctx=_ctx(cfg, srv.base_url, profile="researcher")))  # thread=None
    assert payload["status"] == "completed"
    assert not (Path(home__test_delegation) / "chat").exists()  # no lineage files invented


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


def test_guardrails_apply_to_child_runs(home__test_delegation, monkeypatch, _clean_registry):
    AlwaysFailingPatchHandler.n = 0
    profiles = [{"name": "patcher", "system_prompt": "patch things",
                 "tools": ["patch"], "model_name": None}] + PROFILES
    cfg = _cfg(profiles, home=home__test_delegation)
    with FakeOpenAIServer(handler_cls=AlwaysFailingPatchHandler) as srv:
        payload = json.loads(TOOL_IMPLS["delegate_task"](
            agent="patcher", task="patch everything", wait=True,
            _ctx=_ctx(cfg, srv.base_url, profile="researcher")))
    assert payload["status"] == "failed"
    # the halt came from the child's OWN guardrail controller, not max_iters
    assert payload["failure_reason"] == "guardrail_halt:same_tool_failure_halt"
    assert AlwaysFailingPatchHandler.n < 12


# ------------------------------------------------- registration / config glue

def test_tools_registered_and_in_known_tools(_clean_registry):
    from toaster_loop.dashboard.app import DashboardApp

    names = {s["function"]["name"] for s in TOOL_SCHEMAS}
    assert {"list_agents", "delegate_task", "check_delegation"} <= names
    for n in ("list_agents", "delegate_task", "check_delegation"):
        assert callable(TOOL_IMPLS[n])
    for t in ("list_agents", "delegate_task", "check_delegation"):
        assert t in DashboardApp.KNOWN_TOOLS


def test_recorder_metadata_and_nullrecorder_passthrough(home__test_delegation, _clean_registry):
    from toaster_loop.store.recorder import make_recorder

    cfg = _cfg(home=home__test_delegation)
    rec = make_recorder(cfg, project="p9",
                        metadata={"delegation_id": "d1", "parent_run_id": "r1"})
    row = rec.record(task="t", messages=[{"role": "user", "content": "t"}],
                     backend="toaster_loop", model_alias="m", base_url="",
                     completed=True)
    header = json.loads(Path(row.jsonl_path).read_text().splitlines()[0])
    assert header["delegation"]["delegation_id"] == "d1"
    assert header["project"] == "p9"
    # plain runs keep the key nullable/absent-safe
    rec2 = make_recorder(cfg)
    row2 = rec2.record(task="t", messages=[{"role": "user", "content": "t"}],
                       backend="toaster_loop", model_alias="m", base_url="",
                       completed=True)
    header2 = json.loads(Path(row2.jsonl_path).read_text().splitlines()[0])
    assert header2.get("delegation") is None


def test_config_section_and_env_overrides(tmp_path, monkeypatch, _clean_registry):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("delegation:\n  enabled: false\n")
    monkeypatch.delenv("TOASTER_LOOP_DELEGATION", raising=False)
    cfg = load_config(str(cfg_file))
    assert cfg.delegation_enabled is False

    monkeypatch.setenv("TOASTER_LOOP_DELEGATION", "true")
    cfg2 = load_config(str(cfg_file))
    assert cfg2.delegation_enabled is True

    defaults = ToasterLoopConfig()
    assert defaults.delegation_enabled is True
    # D-18: no timeout knob anymore.
    assert not hasattr(defaults, "delegation_timeout_seconds")


def test_chat_stream_passes_delegation_context_and_thread(home__test_delegation, monkeypatch, _clean_registry):
    """h_chat/stream threads {thread, run_id, project...} into create_runner
    and h_thread_put keeps server-written delegations on client saves."""
    from toaster_loop.dashboard.app import DashboardApp

    app = DashboardApp(ToasterLoopConfig(home=home__test_delegation), home=home__test_delegation)
    seen: dict = {}

    class StubRunner:
        name = "stub"

        def __init__(self, cfg, base_url, model, api_key=None, recorder=None,
                     **kwargs):
            seen.update(kwargs.get("delegation_context") or {})
            self.recorder = recorder

        def run(self, task, history=None, previous_summary=None):
            from toaster_loop.core.agent_runner import AgentResult
            return AgentResult(final_response="ok", messages=[
                {"role": "user", "content": task},
                {"role": "assistant", "content": "ok"}], backend="stub")

    monkeypatch.setattr("toaster_loop.core.agent_runner.create_runner", StubRunner)

    # a serving target is required to reach the runner (attach-mode state,
    # same shape as tests/test_dashboard.py::_fake_serve_state)
    from fake_server import FakeOpenAIServer
    with FakeOpenAIServer() as fs:
        home__test_delegation.mkdir(parents=True, exist_ok=True)
        (home__test_delegation / "serve_state.json").write_text(json.dumps({
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
    tdir = Path(home__test_delegation) / "chat"
    (tdir / "t-keep.json").write_text(json.dumps({
        "id": "t-keep", "title": "k", "messages": [],
        "delegations": [{"agent": "coder", "delegation_id": "d9",
                         "status": "completed", "created": "x"}]}))
    r = app.handle("PUT", "/api/chat/threads/t-keep", body={"title": "k2"})
    assert r.status == 200
    lst = app.handle("GET", "/api/chat/threads").body["threads"]
    row = next(t for t in lst if t["id"] == "t-keep")
    assert [d["delegation_id"] for d in row["delegations"]] == ["d9"]

# ------------------------------------------------------------------------
# from tests/test_dispatcher.py
# ------------------------------------------------------------------------

STUB_WORKER = r'''
import argparse, json, os, sys, time
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

mode = os.environ.get("STUB_MODE", "ok")
if mode == "sleep":
    emit("start")
    time.sleep(float(os.environ.get("STUB_SLEEP", "30")))
    emit("done")
else:
    emit("start")
    emit("done")
'''


@pytest.fixture()
def home__test_dispatcher(tmp_path, monkeypatch):
    h = tmp_path / "ufdisp"
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(h))
    return h


@pytest.fixture()
def registry(home__test_dispatcher):
    reg = ModelRegistry(home__test_dispatcher / "buffer" / "toaster_loop.db")
    yield reg
    reg.close()


@pytest.fixture()
def stub(tmp_path):
    p = tmp_path / "stub_worker.py"
    p.write_text(STUB_WORKER)
    return str(p)


@pytest.fixture()
def _reset_slot_bookkeeping():
    """Per-test isolation for the module-global device-slot state.

    _INFLIGHT_CLAIMS is process-global (same family as conftest's
    _reset_runtime_counters): a daemon execution thread from an earlier test
    can still hold slot 0 briefly; without this reset the next test's claim
    would refuse against a phantom in-flight owner."""
    import toaster_loop.training.scheduler as sched_mod

    with sched_mod._SLOT_LOCK:
        sched_mod._INFLIGHT_CLAIMS.clear()
    yield
    with sched_mod._SLOT_LOCK:
        sched_mod._INFLIGHT_CLAIMS.clear()


def make_sched(registry, stub):
    return Scheduler(registry=registry, cfg=TrainingConfig(),
                     worker_command=[sys.executable, stub])


def wait_for(fn, timeout=15.0, interval=0.05):
    """Poll fn() until truthy; returns its value (last on timeout)."""
    deadline = time.time() + timeout
    val = None
    while time.time() < deadline:
        val = fn()
        if val:
            return val
        time.sleep(interval)
    return val


# ------------------------------------------------- (a) queue drain, no stranding

def test_drain_queue_runs_every_queued_job(registry, stub, _reset_slot_bookkeeping):
    """Jobs enqueued without any submit thread still execute when a holder
    drains — the exact 'nothing ever drains the queue' defect."""
    sched = make_sched(registry, stub)
    j1 = registry.enqueue_job("sft")
    j2 = registry.enqueue_job("export")
    finished = sched.drain_queue()
    assert {j.job_id for j in finished} == {j1.job_id, j2.job_id}
    states = [registry.get_job(j.job_id).state for j in (j1, j2)]
    assert states == ["done", "done"]


def test_dispatcher_loop_drains_without_submit_thread(home__test_dispatcher, registry, stub, _reset_slot_bookkeeping):
    """A JobDispatcher tick picks up a queued job nobody is waiting on."""
    def factory():
        return make_sched(ModelRegistry(home__test_dispatcher / "buffer" / "toaster_loop.db"), stub)

    j = registry.enqueue_job("sft")
    d = JobDispatcher(db_path=registry.db_path, scheduler_factory=factory,
                      tick_s=0.2)
    d.start()
    try:
        ok = wait_for(lambda: registry.get_job(j.job_id).state == "done")
        assert ok, f"job stranded in {registry.get_job(j.job_id).state}"
    finally:
        d.stop()


def test_two_rapid_submits_both_execute(home__test_dispatcher, monkeypatch, stub, _reset_slot_bookkeeping):
    """(a) POST /api/train twice fast with two slots: both jobs execute,
    neither strands in `queued`."""
    monkeypatch.setenv("TOASTER_LOOP_WORKER_COMMAND", f"{sys.executable} {stub}")
    app = DashboardApp(ToasterLoopConfig(training={"max_parallel_jobs": 2}),
                       home=home__test_dispatcher)
    for ds in ("ds-a", "ds-b"):
        d = home__test_dispatcher / "datasets" / ds
        d.mkdir(parents=True)
        (d / "train.yaml").write_text("model: base\ndata: {}\n")

    r1 = app.handle("POST", "/api/train", body={"dataset": "ds-a"})
    r2 = app.handle("POST", "/api/train", body={"dataset": "ds-b"})
    assert r1.status == 202 and r2.status == 202
    ids = [r1.body["job_id"], r2.body["job_id"]]

    reg = ModelRegistry(home__test_dispatcher / "buffer" / "toaster_loop.db")

    def _state(job_id):
        cur = reg.get_job(job_id)
        assert cur is not None
        return cur.state

    try:
        ok = wait_for(lambda: all(
            _state(i) in ("done", "error", "cancelled")
            for i in ids))
        assert ok, [_state(i) for i in ids]
        # both actually ran (not cancelled by a race) and nothing is stranded
        assert {_state(i) for i in ids} == {"done"}
        assert reg.list_jobs("queued") == []
    finally:
        reg.close()


# ------------------------------------------- (b) phantom running rows get reaped

def test_phantom_running_row_reaped_at_startup(home__test_dispatcher, registry, _reset_slot_bookkeeping):
    """A dead worker's `running` row must not survive boot holding submits."""
    j = registry.enqueue_job("sft")
    registry.mark_running(j.job_id, pid=999_999)  # dead pid -> phantom

    app = DashboardApp(ToasterLoopConfig(), home=home__test_dispatcher)
    try:
        d = app.startup_dispatcher()
        assert d is not None
        row = wait_for(lambda: registry.get_job(j.job_id).state == "error")
        assert row, registry.get_job(j.job_id).state
        assert "reaped" in (registry.get_job(j.job_id).error or "")
    finally:
        app.stop_dispatcher()


def test_phantom_row_no_longer_returns_409(home__test_dispatcher, registry, monkeypatch, stub, _reset_slot_bookkeeping):
    """(b) With the phantom reaped by the submit-path pre-check, POST /api/train
    is accepted instead of 409 already_running forever."""
    monkeypatch.setenv("TOASTER_LOOP_WORKER_COMMAND", f"{sys.executable} {stub}")
    j = registry.enqueue_job("sft")
    registry.mark_running(j.job_id, pid=999_999)

    app = DashboardApp(ToasterLoopConfig(), home=home__test_dispatcher)
    ds = home__test_dispatcher / "datasets" / "ds-p"
    ds.mkdir(parents=True)
    (ds / "train.yaml").write_text("model: base\ndata: {}\n")

    r = app.handle("POST", "/api/train", body={"dataset": "ds-p"})
    assert r.status == 202, r.body
    ok = wait_for(lambda: registry.get_job(j.job_id).state == "error")
    assert ok  # phantom reaped; new job accepted


def test_live_running_row_still_returns_409(home__test_dispatcher, registry, _reset_slot_bookkeeping):
    """The reap must not weaken the real single-flight guard: a LIVE running
    row (our own pid) still refuses with 409 already_running."""
    app = DashboardApp(ToasterLoopConfig(), home=home__test_dispatcher)
    j = registry.enqueue_job("sft")
    registry.mark_running(j.job_id, os.getpid())  # alive -> not reaped

    r = app.handle("POST", "/api/train", body={"dataset": "d1"})
    assert r.status == 409 and r.body["error"]["code"] == "already_running"
    assert j.job_id in r.body["error"]["message"]


# --------------------------------------- (c) concurrent claimers, one owner

def test_concurrent_claimers_never_double_claim(home__test_dispatcher, _reset_slot_bookkeeping):
    """(c) N threads racing claim_next_queued over M rows: every row claimed at
    most once (guarded UPDATE + IMMEDIATE transaction)."""
    reg_a = ModelRegistry(home__test_dispatcher / "buffer" / "toaster_loop.db")
    ids = [reg_a.enqueue_job("sft").job_id for _ in range(12)]

    claimed: list = []
    claim_lock = threading.Lock()

    def racer():
        reg = ModelRegistry(home__test_dispatcher / "buffer" / "toaster_loop.db")
        try:
            while True:
                job = reg.claim_next_queued(os.getpid())
                if job is None:
                    return
                with claim_lock:
                    claimed.append(job.job_id)
        finally:
            reg.close()

    threads = [threading.Thread(target=racer) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert sorted(claimed) == sorted(ids), "every row claimed exactly once"
    # all rows landed running under the claim
    reg_b = ModelRegistry(home__test_dispatcher / "buffer" / "toaster_loop.db")
    try:
        for i in ids:
            cur = reg_b.get_job(i)
            assert cur is not None and cur.state == "running"
        assert reg_b.list_jobs("queued") == []
    finally:
        reg_a.close()
        reg_b.close()


def test_requeue_only_releases_own_unspawned_claim(registry, _reset_slot_bookkeeping):
    """requeue_job is pid-guarded: it releases this process's placeholder claim
    but never another holder's running row or a terminal state."""
    mine = registry.enqueue_job("sft")
    claimed = registry.claim_next_queued(os.getpid())
    assert claimed is not None and claimed.job_id == mine.job_id
    assert registry.requeue_job(mine.job_id, os.getpid()) is True
    assert registry.get_job(mine.job_id).state == "queued"

    # a genuinely running row (different pid) cannot be requeued by us
    theirs = registry.enqueue_job("sft")
    registry.mark_running(theirs.job_id, pid=999_998)
    assert registry.requeue_job(theirs.job_id, os.getpid()) is False
    assert registry.get_job(theirs.job_id).state == "running"


def test_claim_skips_dependents_and_kinds(registry, _reset_slot_bookkeeping):
    """claim_next_queued honors #43 dependency edges and skip_kinds — same
    queue-skipping semantics as next_queued."""
    a = registry.enqueue_job("sft")
    b = registry.enqueue_job("export", depends_on=[a.job_id])
    c = registry.enqueue_job("curate")

    first = registry.claim_next_queued(os.getpid(), skip_kinds=("curate",))
    assert first is not None and first.job_id == a.job_id  # b blocked, c skipped
    registry.finish_job(a.job_id, "done")

    nxt = registry.claim_next_queued(os.getpid(), skip_kinds=("curate",))
    assert nxt is not None and nxt.job_id == b.job_id
    assert registry.claim_next_queued(os.getpid(),
                                      skip_kinds=("curate",)) is None


# ------------------------------------------- (d) gate-blocked jobs get retried

def test_gate_blocked_job_retried_when_gate_passes(registry, stub, monkeypatch, _reset_slot_bookkeeping):
    """(d) A job blocked by the VRAM gate stays queued with its _gate_reason;
    once the gate passes, a later drain runs it. The gate itself is never
    weakened — while it fails, the claim is simply released back to queued."""
    import toaster_loop.training.scheduler as sched_mod

    monkeypatch.setattr(sched_mod, "gpu_free_mib", lambda: 100)
    sched = Scheduler(registry=registry,
                      cfg=TrainingConfig(min_free_vram_mib=99_999),
                      worker_command=[sys.executable, stub])
    j = sched.submit("sft")

    assert sched.drain_queue() == []  # gate refuses -> claim released
    row = registry.get_job(j.job_id)
    assert row.state == "queued", "blocked job must not strand claimed/running"
    assert "free memory" in (row.params.get("_gate_reason") or "")

    # gate still closed: another drain attempt keeps it queued, reason intact
    assert sched.drain_queue() == []
    assert registry.get_job(j.job_id).state == "queued"

    monkeypatch.setattr(sched_mod, "gpu_free_mib", lambda: 10 ** 7)
    finished = sched.drain_queue()  # gate passes -> retried and completed
    assert [x.job_id for x in finished] == [j.job_id]
    row = registry.get_job(j.job_id)
    assert row.state == "done"
    assert row.params.get("_gate_reason") is None  # cleared on successful start


def test_gate_block_events_dedup_per_episode(registry, stub, monkeypatch, _reset_slot_bookkeeping):
    """Gate-block events are cumulative but episode-deduped: repeated blocked
    drain attempts (the dispatcher retries a queued job every tick) with the
    SAME reason record ONE event; unblocking and re-blocking records a second.
    This is what makes toaster_loop_gate_blocks_total a true counter while the
    jobs-side _gate_reason stays a cleared-on-unblock gauge."""
    import sqlite3

    import toaster_loop.training.scheduler as sched_mod

    def gate_events():
        conn = sqlite3.connect(str(registry.db_path))
        try:
            return conn.execute(
                "SELECT job_id, reason FROM gate_events ORDER BY id").fetchall()
        finally:
            conn.close()

    monkeypatch.setattr(sched_mod, "gpu_free_mib", lambda: 100)
    sched = Scheduler(registry=registry,
                      cfg=TrainingConfig(min_free_vram_mib=99_999),
                      worker_command=[sys.executable, stub])
    j = sched.submit("sft")

    # several blocked drain attempts (dispatcher-tick loop shape): the same
    # gate reason every time -> exactly one episode event.
    for _ in range(4):
        assert sched.drain_queue() == []
    assert registry.get_job(j.job_id).state == "queued"
    assert len(gate_events()) == 1

    # unblock (job runs; reason cleared) then re-block on a fresh episode:
    monkeypatch.setattr(sched_mod, "gpu_free_mib", lambda: 10 ** 7)
    sched.drain_queue()
    assert registry.get_job(j.job_id).state == "done"

    j2 = sched.submit("sft")
    monkeypatch.setattr(sched_mod, "gpu_free_mib", lambda: 100)
    for _ in range(3):
        assert sched.drain_queue() == []
    ev = gate_events()
    assert len(ev) == 2, "second block episode must add exactly one event"
    assert ev[1][0] == j2.job_id


def test_run_claimed_releases_gate_refusal(registry, stub, monkeypatch, _reset_slot_bookkeeping):
    """run_claimed turns a SchedulerError (gate/slot busy) into a requeued job
    owned by nobody, so the next tick can retry it."""
    import toaster_loop.training.scheduler as sched_mod

    monkeypatch.setattr(sched_mod, "gpu_free_mib", lambda: 100)
    sched = Scheduler(registry=registry,
                      cfg=TrainingConfig(min_free_vram_mib=99_999),
                      worker_command=[sys.executable, stub])
    j = registry.enqueue_job("sft")
    claimed = registry.claim_next_queued(os.getpid())
    assert claimed is not None and claimed.job_id == j.job_id

    with pytest.raises(sched_mod.SchedulerError):
        sched.run_claimed(claimed)
    assert registry.get_job(j.job_id).state == "queued"


# ------------------------------- (e) one dispatcher per DB: pid-lockfile

def test_second_dispatcher_refuses_while_lock_held(home__test_dispatcher, registry, stub, _reset_slot_bookkeeping):
    """(e) Contention fails loudly naming the holder pid and lock path; after
    the holder stops, a new dispatcher starts."""
    def factory():
        return make_sched(ModelRegistry(home__test_dispatcher / "buffer" / "toaster_loop.db"), stub)

    d1 = JobDispatcher(db_path=registry.db_path, scheduler_factory=factory,
                       tick_s=0.2)
    d1.start()
    try:
        lock_path = _dispatcher_lock_path(registry.db_path)
        assert lock_path.is_file()
        assert lock_path.read_text().strip() == str(os.getpid())

        d2 = JobDispatcher(db_path=registry.db_path,
                           scheduler_factory=factory, tick_s=0.2)
        with pytest.raises(DispatcherAlreadyRunning) as ei:
            d2.start()
        msg = str(ei.value)
        assert str(lock_path) in msg and str(os.getpid()) in msg
    finally:
        d1.stop()

    # lock released -> a fresh dispatcher starts (stale pid reclaimed)
    d3 = JobDispatcher(db_path=registry.db_path, scheduler_factory=factory,
                       tick_s=0.2)
    d3.start()
    try:
        assert d3.is_running()
    finally:
        d3.stop()


def test_stale_lock_from_dead_pid_is_reclaimed(home__test_dispatcher, registry, stub, _reset_slot_bookkeeping):
    """A lockfile left behind by a dead process (no flock held) never wedges
    the queue: start succeeds and rewrites the recorded pid."""
    def factory():
        return make_sched(ModelRegistry(home__test_dispatcher / "buffer" / "toaster_loop.db"), stub)

    lock_path = _dispatcher_lock_path(registry.db_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("999999")  # dead holder's stale pid, no flock

    d = JobDispatcher(db_path=registry.db_path, scheduler_factory=factory,
                      tick_s=0.2)
    d.start()
    try:
        assert d.is_running()
        assert lock_path.read_text().strip() == str(os.getpid())
    finally:
        d.stop()


def test_app_ensure_dispatcher_contention(home__test_dispatcher, _reset_slot_bookkeeping):
    """Two DashboardApps over one home: the second ensure_dispatcher raises,
    and succeeds again once the first app stops its dispatcher."""
    app1 = DashboardApp(ToasterLoopConfig(), home=home__test_dispatcher)
    app2 = DashboardApp(ToasterLoopConfig(), home=home__test_dispatcher)
    try:
        assert app1.ensure_dispatcher() is not None
        with pytest.raises(DispatcherAlreadyRunning):
            app2.ensure_dispatcher()
    finally:
        app1.stop_dispatcher()
    try:
        assert app2.ensure_dispatcher() is not None
    finally:
        app2.stop_dispatcher()


# ------------------------------------------------ startup workspace-root log

def test_workspace_root_fallback_logged(caplog, home__test_dispatcher, _reset_slot_bookkeeping):
    """agent.workspace_root unset -> the cwd fallback (often $HOME) is logged
    loudly at WARNING on the startup path."""
    app = DashboardApp(ToasterLoopConfig(), home=home__test_dispatcher)
    try:
        with caplog.at_level(logging.WARNING, logger="toaster_loop.dashboard"):
            app.startup_dispatcher()
        assert any(
            "workspace root" in r.getMessage()
            and str(Path(os.getcwd())) in r.getMessage()
            and r.levelno == logging.WARNING
            for r in caplog.records)
    finally:
        app.stop_dispatcher()


def test_workspace_root_explicit_logged_at_info(caplog, home__test_dispatcher, _reset_slot_bookkeeping):
    """An explicitly configured root is logged at INFO (resolved value), no
    fallback warning."""
    ws = home__test_dispatcher / "ws"
    ws.mkdir(parents=True)
    app = DashboardApp(ToasterLoopConfig(agent={"workspace_root": str(ws)}),
                       home=home__test_dispatcher)
    with caplog.at_level(logging.INFO, logger="toaster_loop.dashboard"):
        app._log_workspace_root()
    msgs = [r for r in caplog.records if "workspace root" in r.getMessage()]
    assert any(str(ws) in r.getMessage() and r.levelno == logging.INFO
               for r in msgs)
    assert not any(r.levelno == logging.WARNING for r in msgs)


def test_workspace_root_null_warns_containment_disabled(caplog, home__test_dispatcher, _reset_slot_bookkeeping):
    """Explicit null (UNSET sentinel) -> containment-disabled warning."""
    from toaster_loop.core.config import UNSET

    app = DashboardApp(ToasterLoopConfig(agent={"workspace_root": UNSET}),
                       home=home__test_dispatcher)
    with caplog.at_level(logging.WARNING, logger="toaster_loop.dashboard"):
        app._log_workspace_root()
    assert any("DISABLED" in r.getMessage() for r in caplog.records
               if "workspace root" in r.getMessage())


# --------------------------------------------------------------- constants

def test_dispatch_tick_constant_sane(_reset_slot_bookkeeping):
    """The documented tick interval exists and is a sane cadence next to
    TrainingConfig.poll_interval_s (1 s)."""
    assert 1.0 <= DISPATCH_TICK_S <= 60.0

# ------------------------------------------------------------------------
# from tests/test_guardrails.py
# ------------------------------------------------------------------------

@pytest.fixture()
def toaster_loop_home(tmp_path, monkeypatch):
    home = tmp_path / "uf-home"
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(home))
    return home


# ------------------------------------------------------------- signatures

def test_signature_ignores_arg_key_order(_reset_handler_counters):
    a = ToolCallSignature.from_call("patch", {"path": "x", "old_string": "a"})
    b = ToolCallSignature.from_call("patch", {"old_string": "a", "path": "x"})
    assert a == b


def test_signature_accepts_json_string_args(_reset_handler_counters):
    a = ToolCallSignature.from_call("patch", '{"b": 2, "a": 1}')
    b = ToolCallSignature.from_call("patch", {"a": 1, "b": 2})
    assert a == b
    # malformed / non-mapping args collapse to {} deterministically
    assert (ToolCallSignature.from_call("patch", "{oops")
            == ToolCallSignature.from_call("patch", None))


def test_signature_differs_per_tool_and_args(_reset_handler_counters):
    base = ToolCallSignature.from_call("patch", {"a": 1})
    assert base != ToolCallSignature.from_call("write_file", {"a": 1})
    assert base != ToolCallSignature.from_call("patch", {"a": 2})


# --------------------------------------------------------- threshold ladder

def test_exact_failure_ladder_warn_then_block_then_halt(_reset_handler_counters):
    g = ToolLoopGuardrails()
    args = {"path": "f.py", "old_string": "x"}

    actions = [g.after_call("patch", args, "error: nope").action for _ in range(5)]
    # call 1 allow; calls 2..5 warn (exact_failure_warn_after=2)
    assert actions == ["allow", "warn", "warn", "warn", "warn"]

    # after 5 identical failures the 6th identical call is blocked before exec
    d = g.before_call("patch", args)
    assert d.action == "block" and d.code == "repeated_exact_failure_block"

    # if the model keeps re-failing the same tool (counted anyway), the
    # same-tool counter halts at 8
    decisions = [g.after_call("patch", {"path": f"f{i}.py"}, "error: nope")
                 for i in range(3)]
    assert decisions[-1].action == "halt"
    assert decisions[-1].code == "same_tool_failure_halt"


def test_same_tool_varied_args_warn_then_halt(_reset_handler_counters):
    g = ToolLoopGuardrails()
    acts = [g.after_call("patch", {"path": f"f{i}.py"}, "error: boom").action
            for i in range(8)]
    # same_tool_failure_warn_after=3 -> warn from the 3rd failure on
    assert acts[0] == "allow" and acts[1] == "allow"
    assert acts[2] == "warn"
    assert acts[-1] == "halt"


def test_success_clears_failure_streaks(_reset_handler_counters):
    g = ToolLoopGuardrails()
    args = {"path": "f.py"}
    g.after_call("patch", args, "error: one")
    g.after_call("patch", args, "error: two")  # would warn on next failure
    d = g.after_call("patch", args, "ok: written")
    assert d.action == "allow"
    d2 = g.after_call("patch", args, "error: three")
    assert d2.action == "allow"  # counters were reset by the success


# ------------------------------------------------------------ no progress

def test_idempotent_no_progress_warn_then_block(_reset_handler_counters):
    g = ToolLoopGuardrails()
    args = {"path": "a.txt"}
    acts = [g.after_call("read_file", args, "same content").action
            for _ in range(5)]
    assert acts == ["allow", "warn", "warn", "warn", "warn"]
    d = g.before_call("read_file", args)
    assert d.action == "block" and d.code == "idempotent_no_progress_block"


def test_idempotent_changed_result_resets_no_progress(_reset_handler_counters):
    g = ToolLoopGuardrails()
    args = {"path": "a.txt"}
    g.after_call("read_file", args, "v1")
    g.after_call("read_file", args, "v2")  # changed -> streak resets
    d = g.after_call("read_file", args, "v3")
    assert d.action == "allow"


def test_mutating_tools_never_no_progress_blocked(_reset_handler_counters):
    g = ToolLoopGuardrails()
    args = {"path": "a.txt", "content": "x"}
    for _ in range(6):  # identical successful writes: never flagged
        d = g.after_call("write_file", args, "ok: wrote a.txt")
        assert d.action == "allow"
    assert g.before_call("write_file", args).action == "allow"


def test_unknown_tools_treated_as_mutating(_reset_handler_counters):
    g = ToolLoopGuardrails()
    for _ in range(6):
        assert g.after_call("mystery_tool", {"a": 1}, "same").action == "allow"
    assert g.before_call("mystery_tool", {"a": 1}).action == "allow"


def test_tool_sets_match_spec(_reset_handler_counters):
    # WP12: plan reads join the idempotent set, plan writes the mutating set.
    assert IDEMPOTENT_TOOL_NAMES == {
        "read_file", "list_dir", "search_files", "todo_read", "search_traces",
        "plan_next", "plan_view"}
    assert MUTATING_TOOL_NAMES == {"write_file", "patch", "todo_write",
                                   "plan_progress", "plan_add"}


# ------------------------------------------------------------------- off

def test_disabled_controller_always_allows(_reset_handler_counters):
    g = ToolLoopGuardrails(enabled=False)
    for _ in range(10):
        assert g.after_call("patch", {"a": 1}, "error: boom").action == "allow"
        assert g.before_call("patch", {"a": 1}).action == "allow"


# ------------------------------------------------------- repetition guard

def _loop_text(prefix: str = "", repeats: int = 12) -> str:
    unit = "the quick brown fox jumps over the lazy dog again and again.\n"  # 60 chars
    return prefix + unit * repeats


def test_detect_repetition_flags_loop(_reset_handler_counters):
    text = _loop_text()
    assert len(text) >= 400
    assert detect_repetition(text) is True
    # loop starts at index 0 (no prefix in this fixture)
    assert find_loop(text) == 0


def test_detect_repetition_ignores_normal_text(_reset_handler_counters):
    normal = "toaster_loop records traces to ~/.toaster-loop/buffer. " * 3
    assert detect_repetition(normal) is False
    assert detect_repetition("short loop loop loop") is False
    assert detect_repetition("") is False


def test_truncate_keeps_prefix_and_marks(_reset_handler_counters):
    text = _loop_text(prefix="Here is the answer you asked for.\n")
    out = truncate_repetition(text)
    assert out.startswith("Here is the answer you asked for.")
    assert out.endswith(REPETITION_MARKER)
    assert len(out) < len(text)


def test_truncate_clean_text_unchanged(_reset_handler_counters):
    text = "a perfectly normal response.\n" * 3
    assert truncate_repetition(text) == text


# ------------------------------------------------------ runner integration

class AlwaysFailingPatchHandler__test_guardrails(BaseHTTPRequestHandler):
    """Always requests the same patch call; args vary per request (counter)
    so the exact-failure block never pre-empts the same-tool halt."""

    n = 0

    def log_message(self, format, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        type(self).n += 1
        msg = {"role": "assistant", "content": None, "tool_calls": [{
            "id": f"c{type(self).n}", "type": "function",
            "function": {"name": "patch", "arguments": json.dumps(
                {"path": f"missing{type(self).n}.py",
                 "old_string": "x", "new_string": "y"})}}]}
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
    return "error: patch failed: file not found"


@pytest.fixture()
def _reset_handler_counters():
    AlwaysFailingPatchHandler__test_guardrails.n = 0
    yield


def test_run_halts_before_max_iterations_with_guardrail_reason(toaster_loop_home, _reset_handler_counters):
    from toaster_loop.store.recorder import TraceRecorder
    rec = TraceRecorder(home=toaster_loop_home)
    with FakeOpenAIServer(handler_cls=AlwaysFailingPatchHandler__test_guardrails) as srv:
        client = ToasterLoopAgentClient(base_url=srv.base_url, model="m",
                                    tool_impls={"patch": failing_patch_impl},
                                    max_iterations=12, recorder=rec)
        result = client.run("fix the file")
    assert result.completed is False
    assert result.failure_reason == "guardrail_halt:same_tool_failure_halt"
    assert is_infra_reason(result.failure_reason) is False  # model-ish failure
    assert AlwaysFailingPatchHandler__test_guardrails.n < 12  # halted before max_iterations
    assert result.trace_id is not None

    footer = _footer_for(toaster_loop_home, result.trace_id)
    assert footer["failure_reason"] == "guardrail_halt:same_tool_failure_halt"
    assert footer["completed"] is False


def test_run_warn_and_block_surfaced_in_tool_results(_reset_handler_counters):
    with FakeOpenAIServer(handler_cls=IdenticalFailingPatchHandler) as srv:
        client = ToasterLoopAgentClient(base_url=srv.base_url, model="m",
                                    tool_impls={"patch": failing_patch_impl},
                                    max_iterations=8)
        result = client.run("fix the file")
    tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
    # warn nudge appended to a tool result
    assert any("[Tool loop warning: repeated_exact_failure_warning" in m["content"]
               for m in tool_msgs)
    # block -> synthetic error tool result, execution skipped, loop continued
    assert any(m["content"].startswith("error: blocked by guardrail "
                                       "(repeated_exact_failure_block)")
               for m in tool_msgs)
    # identical-call spiral ends at max_iterations (blocks never halt)
    assert result.failure_reason == "max_iterations"


def test_final_answer_repetition_truncated(_reset_handler_counters):
    loop = _loop_text(prefix="Answer:\n")

    class LoopAnswerHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            body = json.dumps({
                "id": "x", "object": "chat.completion",
                "model": payload.get("model", "m"),
                "choices": [{"index": 0,
                             "message": {"role": "assistant", "content": loop},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                          "total_tokens": 2}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    with FakeOpenAIServer(handler_cls=LoopAnswerHandler) as srv:
        client = ToasterLoopAgentClient(base_url=srv.base_url, model="m")
        result = client.run("tell me a story")
    assert result.completed is True
    assert result.final_response.endswith(REPETITION_MARKER)
    assert result.final_response.startswith("Answer:")
    assert len(result.final_response) < len(loop)


def test_summarize_output_repetition_truncated(_reset_handler_counters):
    loop = _loop_text(prefix="Summary:\n")

    class LoopSummaryHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_POST(self):
            body = json.dumps({
                "id": "x", "object": "chat.completion", "model": "m",
                "choices": [{"index": 0,
                             "message": {"role": "assistant", "content": loop},
                             "finish_reason": "stop"}],
                "usage": {}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    with FakeOpenAIServer(handler_cls=LoopSummaryHandler) as srv:
        client = ToasterLoopAgentClient(base_url=srv.base_url, model="m")
        out = client._summarize("summarize please")
    assert out is not None
    assert out.endswith(REPETITION_MARKER)
    assert len(out) < len(loop)


def test_guardrails_off_runs_to_max_iterations(toaster_loop_home, _reset_handler_counters):
    with FakeOpenAIServer(handler_cls=AlwaysFailingPatchHandler__test_guardrails) as srv:
        client = ToasterLoopAgentClient(base_url=srv.base_url, model="m",
                                    tool_impls={"patch": failing_patch_impl},
                                    max_iterations=4, guardrails_enabled=False)
        result = client.run("fix the file")
    assert result.failure_reason == "max_iterations"
    assert AlwaysFailingPatchHandler__test_guardrails.n == 4


def test_create_runner_config_knob(_reset_handler_counters):
    cfg_on = ToasterLoopConfig()
    assert create_runner(cfg_on, base_url="http://127.0.0.1:9/v1",
                         model="m").guardrails_enabled is True
    cfg_off = ToasterLoopConfig(agent={"guardrails": False})
    assert create_runner(cfg_off, base_url="http://127.0.0.1:9/v1",
                         model="m").guardrails_enabled is False


# ------------------------------------------------------------------ utils

def _footer_for(home: Path, trace_id: str) -> dict:
    matches = list((home / "buffer").rglob(f"{trace_id}.jsonl"))
    assert len(matches) == 1
    lines = [json.loads(l) for l in matches[0].read_text().splitlines()]
    return lines[-1]

# ------------------------------------------------------------------------
# from tests/test_post_write_checks.py
# ------------------------------------------------------------------------

# ------------------------------------------------------------- tier 1: python

def test_broken_python_write_reports_error_and_still_writes(tmp_path):
    p = tmp_path / "broken.py"
    res = write_file(str(p), "def f(:\n    pass\n")
    # file WAS written (established semantics: never reject the write)
    assert p.is_file() and p.read_text().startswith("def f(")
    assert res.startswith(f"wrote {len('def f(:' + chr(10) + '    pass' + chr(10))} chars")
    assert SYNTAX_FAIL_MARK in res
    assert "fix it next turn" in res
    assert str(p) in res
    # parser error text carries the line number of the break
    assert ":1:" in res or "line 1" in res


def test_valid_python_write_passes_with_check_ok(tmp_path):
    p = tmp_path / "ok.py"
    res = write_file(str(p), "x = 1\n")
    assert res.endswith(SYNTAX_OK_MARK)
    assert SYNTAX_FAIL_MARK not in res


def test_unknown_extension_skipped_silently(tmp_path):
    p = tmp_path / "notes.dat"
    res = write_file(str(p), "totally (( unbalanced content")
    assert res == f"wrote {len('totally (( unbalanced content')} chars to {p}"
    assert SYNTAX_OK_MARK not in res and SYNTAX_FAIL_MARK not in res


# ----------------------------------------------------------------- tier 1: js

def test_broken_js_write_reports_node_error(tmp_path):
    if tools.shutil.which("node") is None:
        pytest.skip("node not installed")
    p = tmp_path / "bad.js"
    res = write_file(str(p), "const x = ;\n")
    assert p.is_file()
    assert SYNTAX_FAIL_MARK in res
    assert "SyntaxError" in res


def test_valid_js_write_passes(tmp_path):
    if tools.shutil.which("node") is None:
        pytest.skip("node not installed")
    p = tmp_path / "ok.js"
    res = write_file(str(p), "const x = 1;\n")
    assert res.endswith(SYNTAX_OK_MARK)


def test_node_missing_degrades_to_skip(tmp_path, monkeypatch):
    monkeypatch.setattr(tools.shutil, "which", lambda name: None)
    p = tmp_path / "x.js"
    res = write_file(str(p), "const x = ;\n")  # broken, but node is "missing"
    assert SYNTAX_FAIL_MARK not in res and SYNTAX_OK_MARK not in res
    status, _ = tools._check_node_syntax(str(p), "const x = ;\n")
    assert status == "skipped"


# ------------------------------------------------------------ delta semantics

def test_patch_into_pre_broken_file_not_blamed(tmp_path):
    """Pre-existing syntax error must be annotated as inherited, not new."""
    p = tmp_path / "pre.py"
    write_file(str(p), "def g(:\n    pass\n", lint_on_write=False)
    res = patch(str(p), "pass", "pass  # touched")
    assert p.is_file() and "# touched" in p.read_text()
    assert SYNTAX_FAIL_MARK in res
    assert "didn't introduce a new one" in res


def test_patch_introducing_new_error_is_reported(tmp_path):
    p = tmp_path / "clean.py"
    write_file(str(p), "x = 1\n", lint_on_write=False)
    res = patch(str(p), "x = 1", "x = (")
    assert SYNTAX_FAIL_MARK in res
    assert "didn't introduce a new one" not in res


def test_tier1_delta_filters_pre_existing_error_lines(monkeypatch):
    """Direct old-vs-new diff logic: shared error lines drop out, new ones
    stay (pre-edit baseline line-set difference)."""
    def fake_check(path, content):
        if "NEWBREAK" in content:
            return "error", "f.py:1: SyntaxError: old\nf.py:9: SyntaxError: new"
        return "error", "f.py:1: SyntaxError: old"

    monkeypatch.setattr(tools, "_tier1_check", fake_check)
    post_err = "f.py:1: SyntaxError: old\nf.py:9: SyntaxError: new"
    report, text = tools._tier1_delta("f.py", "OLD", post_err)
    assert report is True
    assert "New syntax errors" in text and "new" in text
    assert "old" not in text.split("introduced by this edit")[1]

    # every post error pre-existing => still-broken annotation, no blame
    monkeypatch.setattr(tools, "_tier1_check",
                        lambda path, content: ("error", "f.py:1: same"))
    report, text = tools._tier1_delta("f.py", "A", "f.py:1: same")
    assert report is True
    assert "didn't introduce a new one" in text

    # clean pre-edit baseline => full post error reported as-is
    monkeypatch.setattr(tools, "_tier1_check", lambda path, content: ("ok", ""))
    report, text = tools._tier1_delta("f.py", "clean-before", "f.py:2: boom")
    assert text == "f.py:2: boom"


# --------------------------------------------------------------- gates/config

def test_lint_on_write_false_disables_everything(tmp_path):
    p = tmp_path / "off.py"
    res = write_file(str(p), "def f(:\n", lint_on_write=False)
    assert res == f"wrote {len('def f(:' + chr(10))} chars to {p}"
    p2 = tmp_path / "c.py"
    write_file(str(p2), "x = 1\n", lint_on_write=False)
    res2 = patch(str(p2), "x = 1", "x = (", lint_on_write=False)
    assert SYNTAX_FAIL_MARK not in res2


def test_lsp_disabled_by_default(tmp_path):
    # no lsp_config at all => zero LSP text even on a syntax failure
    p = tmp_path / "d.py"
    res = write_file(str(p), "def f(:\n")
    assert "LSP" not in res
    assert tools._maybe_lsp_suffix(str(p), None, "x", {}) == ""


# ------------------------------------------------------------- tier 2: LSP

_STUB_LSP = """#!/usr/bin/env python3
import json, re, sys

def send(msg):
    body = json.dumps(msg).encode()
    sys.stdout.buffer.write(b"Content-Length: %d\\r\\n\\r\\n" % len(body) + body)
    sys.stdout.buffer.flush()

def read():
    headers = b""
    while b"\\r\\n\\r\\n" not in headers:
        ch = sys.stdin.buffer.read(1)
        if not ch:
            return None
        headers += ch
    n = int(re.search(rb"Content-Length: (\\d+)", headers).group(1))
    return json.loads(sys.stdin.buffer.read(n))

def diag(msg, line):
    return {"range": {"start": {"line": line, "character": 0},
                      "end": {"line": line, "character": 1}},
            "severity": 1, "source": "stub", "code": "E1", "message": msg}

while True:
    m = read()
    if m is None:
        break
    method = m.get("method")
    if method == "initialize":
        send({"jsonrpc": "2.0", "id": m["id"], "result": {"capabilities": {}}})
    elif method == "shutdown":
        send({"jsonrpc": "2.0", "id": m["id"], "result": None})
    elif method == "exit":
        break
    elif method == "textDocument/didOpen":
        uri = m["params"]["textDocument"]["uri"]
        send({"jsonrpc": "2.0", "method": "textDocument/publishDiagnostics",
              "params": {"uri": uri, "diagnostics": [diag("old error", 0)]}})
    elif method == "textDocument/didChange":
        uri = m["params"]["textDocument"]["uri"]
        send({"jsonrpc": "2.0", "method": "textDocument/publishDiagnostics",
              "params": {"uri": uri,
                         "diagnostics": [diag("old error", 0),
                                         diag("new error", 1)]}})
"""


@pytest.fixture()
def stub_server(tmp_path):
    """Executable stub LSP server fixture (baseline 'old error'; after the
    didChange it also pushes 'new error' — only the latter may be reported)."""
    s = tmp_path / "stub-langserver"
    s.write_text(_STUB_LSP)
    s.chmod(s.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP)
    return str(s)


def test_lsp_detected_server_reports_only_new_diagnostics(tmp_path, stub_server):
    p = tmp_path / "svc.py"
    cfg = {"enabled": True,
           "servers": {".py": stub_server},
           "startup_timeout_seconds": 10,
           "check_timeout_seconds": 15}
    # clean pre-edit content (written without checks), then a patch whose
    # edit makes the stub push baseline + new diagnostics.
    write_file(str(p), "x = 1\n", lint_on_write=False)
    res = patch(str(p), "x = 1", "x = 2\ny = 3\n", lsp_config=cfg)
    assert "NEW LSP DIAGNOSTICS" in res
    assert "new error" in res
    assert "old error" not in res  # baseline delta filtered it out


def test_lsp_graceful_skip_when_server_not_detected(tmp_path, monkeypatch):
    monkeypatch.setattr(tools.shutil, "which", lambda name: None)
    cfg = {"enabled": True}
    assert tools._maybe_lsp_suffix(
        str(tmp_path / "a.py"), "x=1", "x=2", cfg) == ""


def test_lsp_probe_never_outlives_its_timeout(tmp_path, stub_server):
    # check_timeout 0.5s must return quickly with no verdict rather than hang
    diags = tools._lsp_diagnostics(str(tmp_path / "t.py"), "a", "b",
                                   [stub_server], startup_timeout=0.3,
                                   check_timeout=0.5)
    assert diags is None or isinstance(diags, list)


# ------------------------------------------------------------ config plumbing

def test_config_parses_tools_lint_and_lsp(tmp_path, monkeypatch):
    monkeypatch.delenv("TOASTER_LOOP_LINT_ON_WRITE", raising=False)
    monkeypatch.delenv("TOASTER_LOOP_LSP", raising=False)
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(yaml.safe_dump({
        "tools": {"lint_on_write": False,
                  "lsp": {"enabled": True, "startup_timeout_seconds": 9}}}))
    cfg = load_config(str(cfg_file))
    assert cfg.tools["lint_on_write"] is False
    assert cfg.tools["lsp"]["enabled"] is True
    assert cfg.tools["lsp"]["startup_timeout_seconds"] == 9


def test_config_defaults_lint_on_true_lsp_absent(tmp_path, monkeypatch):
    monkeypatch.delenv("TOASTER_LOOP_LINT_ON_WRITE", raising=False)
    monkeypatch.delenv("TOASTER_LOOP_LSP", raising=False)
    cfg = load_config()
    assert "lint_on_write" not in cfg.tools  # default applied at read site
    assert bool((cfg.tools.get("lsp") or {}).get("enabled")) is False


def test_env_override_lint_on_write(tmp_path, monkeypatch):
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(yaml.safe_dump({"tools": {"lint_on_write": True}}))
    monkeypatch.setenv("TOASTER_LOOP_LINT_ON_WRITE", "false")
    assert load_config(str(cfg_file)).tools["lint_on_write"] is False
    monkeypatch.setenv("TOASTER_LOOP_LINT_ON_WRITE", "1")
    assert load_config(str(cfg_file)).tools["lint_on_write"] is True


def test_env_override_lsp_enabled(tmp_path, monkeypatch):
    monkeypatch.delenv("TOASTER_LOOP_LINT_ON_WRITE", raising=False)
    monkeypatch.setenv("TOASTER_LOOP_LSP", "true")
    cfg = load_config()
    assert cfg.tools["lsp"]["enabled"] is True


def test_create_runner_binds_lint_flags(tmp_path, monkeypatch):
    """create_runner must thread tools.lint_on_write/lsp into write_file/patch
    (bound invisibly; the model cannot toggle them)."""
    from toaster_loop.core.agent_runner import create_runner

    monkeypatch.delenv("TOASTER_LOOP_LINT_ON_WRITE", raising=False)
    monkeypatch.delenv("TOASTER_LOOP_LSP", raising=False)
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(yaml.safe_dump({"tools": {"lint_on_write": False}}))
    cfg = load_config(str(cfg_file))
    client = create_runner(cfg, base_url="http://127.0.0.1:1/v1", model="m")
    p = tmp_path / "bound.py"
    res = client.tool_impls["write_file"](str(p), "def f(:\n")
    assert SYNTAX_FAIL_MARK not in res  # lint disabled through the runner

# ------------------------------------------------------------------------
# from tests/test_presets_enable_all.py
# ------------------------------------------------------------------------

@pytest.fixture()
def home__test_presets_enable_all(tmp_path, monkeypatch):
    h = tmp_path / "toaster_loop-presets"
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(h))
    return h


@pytest.fixture()
def app(home__test_presets_enable_all):
    a = DashboardApp(ToasterLoopConfig(), home=home__test_presets_enable_all)
    yield a


# --------------------------------------------------------------- patch list

def test_patches_cover_known_sections_and_bool_switches():
    for section, switches in ENABLE_ALL_SWITCHES.items():
        assert all(isinstance(v, bool) for v in switches.values())
    # The headline features must be in the list.
    assert ENABLE_ALL_SWITCHES["skills"]["enabled"] is True
    assert ENABLE_ALL_SWITCHES["playbooks"]["enabled"] is True
    assert ENABLE_ALL_SWITCHES["routing"]["enabled"] is True


def test_easy_training_and_janitor_never_in_preset():
    # Cardinal: the loop and auto-archive are decisions, not switches.
    assert "easy_training" not in ENABLE_ALL_SWITCHES
    assert "janitor" not in ENABLE_ALL_SWITCHES


def test_conditional_switches_off_without_env(home__test_presets_enable_all, monkeypatch):
    # No language server on PATH, no calibration profile => neither appears.
    monkeypatch.setattr("shutil.which", lambda *a, **k: None)
    sections = [p["section"] for p in enable_all_patches(ToasterLoopConfig(), home=str(home__test_presets_enable_all))]
    tools = [p for p in enable_all_patches(ToasterLoopConfig(), home=str(home__test_presets_enable_all))
             if p["section"] == "tools"]
    assert not tools  # lsp patch absent
    # serve only ever carries watchdog (a plain switch), never calibration here
    for p in enable_all_patches(ToasterLoopConfig(), home=str(home__test_presets_enable_all)):
        if p["section"] == "serve":
            assert set(p["patch"]) == {"watchdog_enabled"}
    assert "serve" in sections  # watchdog itself is unconditional


def test_lsp_included_when_server_on_path(home__test_presets_enable_all, monkeypatch):
    monkeypatch.setattr("shutil.which",
                        lambda name, **k: "/usr/bin/" + name
                        if name == "pyright-langserver" else None)
    tools = [p for p in enable_all_patches(ToasterLoopConfig(), home=str(home__test_presets_enable_all))
             if p["section"] == "tools"]
    assert tools and tools[0]["patch"]["lsp"]["enabled"] is True


def test_diff_reports_only_differing_keys():
    cfg = ToasterLoopConfig()
    cfg.skills = {"enabled": True}  # already on => absent from diff
    d = {(x["section"], x["key"]): x for x in enable_all_diff(cfg)}
    assert ("skills", "enabled") not in d
    assert d[("routing", "enabled")]["to"] is True
    assert d[("validation", "shadow_enabled")]["to"] is True


# ---------------------------------------------------------------- endpoints

def test_preview_writes_nothing(app, home__test_presets_enable_all):
    cfg_file = home__test_presets_enable_all / "toaster_loop.yaml"
    assert not cfg_file.exists()
    r = app.handle("POST", "/api/features/enable_all", body={"preview": True})
    assert r.status == 200
    assert isinstance(r.body["diff"], list) and r.body["diff"]
    assert not cfg_file.exists()
    # live config untouched
    assert not (app.cfg.skills or {}).get("enabled")


def test_apply_flips_switches_and_preserves_unrelated_keys(app, home__test_presets_enable_all):
    app.handle("PUT", "/api/config", body={
        "section": "skills", "patch": {"index_total_cap_chars": 999}})
    app.handle("PUT", "/api/config", body={
        "section": "easy_training", "patch": {"enabled": False}})
    r = app.handle("POST", "/api/features/enable_all", body={})
    assert r.status == 200, r.body
    cfg = load_config(str(home__test_presets_enable_all / "toaster_loop.yaml"))
    assert (cfg.skills or {}).get("enabled") is True
    assert (cfg.skills or {}).get("index_total_cap_chars") == 999  # merge kept it
    assert (cfg.playbooks or {}).get("enabled") is True
    assert (cfg.routing or {}).get("enabled") is True
    assert (cfg.serve or {}).get("watchdog_enabled") is True
    assert (cfg.validation or {}).get("shadow_enabled") is True
    assert (cfg.rft or {}).get("enabled") is True
    assert (cfg.evolution or {}).get("enabled") is True
    assert (cfg.goldens_mine or {}).get("enabled") is True
    # deliberate exclusions untouched
    assert not (cfg.easy_training or {}).get("enabled")
    assert not (cfg.janitor or {}).get("enabled")


def test_apply_idempotent_empty_diff_second_time(app, home__test_presets_enable_all):
    assert app.handle("POST", "/api/features/enable_all", body={}).status == 200
    r = app.handle("POST", "/api/features/enable_all",
                   body={"preview": True})
    assert r.status == 200 and r.body["diff"] == []


def test_rollback_on_failed_patch(app, home__test_presets_enable_all, monkeypatch):
    # Poison the SECOND section-patch so the first write lands and then must
    # revert. handle() converts PUT failures to ApiResponse (never raises),
    # which is exactly the contract h_features_enable_all rolls back on.
    called = {"n": 0}
    orig = DashboardApp.handle

    def flaky(self, method, path, query=None, body=None):
        if method == "PUT" and path == "/api/config":
            called["n"] += 1
            if called["n"] == 2:
                from toaster_loop.dashboard.app import ApiResponse
                return ApiResponse(500, {"error": {
                    "code": "test", "message": "boom"}})
        return orig(self, method, path, query, body)

    monkeypatch.setattr(DashboardApp, "handle", flaky)
    # handle() converts the handler's HttpError into an error ApiResponse.
    r = orig(app, "POST", "/api/features/enable_all", body={})
    assert r.status >= 400
    assert r.body["error"]["message"] == "boom"
    assert called["n"] == 2  # first landed, second failed, then rollback
    # After rollback neither the live config nor the file has any switch on.
    assert not (app.cfg.skills or {}).get("enabled")
    assert not (app.cfg.playbooks or {}).get("enabled")
    cfg_file = home__test_presets_enable_all / "toaster_loop.yaml"
    if cfg_file.is_file():
        cfg = load_config(str(cfg_file))
        assert not (cfg.skills or {}).get("enabled")
        assert not (cfg.playbooks or {}).get("enabled")

# ------------------------------------------------------------------------
# from tests/test_profiles_schedule.py
# ------------------------------------------------------------------------

STATIC = Path(__file__).resolve().parents[1] / "toaster_loop" / "dashboard" / "static"


@pytest.fixture()
def home__test_profiles_schedule(tmp_path, monkeypatch):
    h = tmp_path / "uf-5e"
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(h))
    return h


@pytest.fixture()
def app__test_profiles_schedule(home__test_profiles_schedule):
    return DashboardApp(ToasterLoopConfig(), home=home__test_profiles_schedule)


def seed_trace(store: TraceStore, tid: str, score=0.8, curated=False):
    from datetime import datetime, timedelta, timezone

    p = Path(store.db_path).parent / f"{tid}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        f.write(json.dumps({"kind": "trace_header"}) + "\n")
        f.write(json.dumps({"kind": "message", "message": {
            "role": "user", "content": f"task {tid}"}}) + "\n")
        f.write(json.dumps({"kind": "message", "message": {
            "role": "assistant", "content": f"done {tid}"}}) + "\n")
    # relative-to-now stamp (time-bomb sweep): never ages out of windowed
    # assertions (stats_daily / insights calendar cutoffs).
    ts = (datetime.now(timezone.utc) - timedelta(days=1)
          ).isoformat(timespec="seconds")
    store.insert_trace(TraceRow(
        trace_id=tid, session_id=None, timestamp=ts,
        agent_profile="fallback", model_alias="base-model",
        model_version=None, endpoint_label=None, task_summary=f"t {tid}",
        status="success", outcome_score=score, num_tool_calls=0,
        tokens_in=10, tokens_out=20, duration_s=1.0, jsonl_path=str(p),
        curated=bool(curated),
    ))


# ---------------------------------------------------------------- profiles CRUD

def test_profiles_crud_and_default(app__test_profiles_schedule, home__test_profiles_schedule):
    r = app__test_profiles_schedule.handle("GET", "/api/profiles")
    assert r.status == 200
    assert r.body["profiles"] == [] and r.body["default_profile"] is None
    assert "read_file" in r.body["known_tools"]

    # create
    r = app__test_profiles_schedule.handle("PUT", "/api/profiles/coder", body={
        "system_prompt": "You are a careful coder.",
        "tools": ["read_file"], "model_name": None})
    assert r.status == 200 and r.body["profile"]["name"] == "coder"

    # persisted to the config file (profiles: section)
    raw = (home__test_profiles_schedule / "toaster_loop.yaml").read_text()
    assert "coder" in raw and "careful coder" in raw

    # upsert same name updates in place
    app__test_profiles_schedule.handle("PUT", "/api/profiles/coder", body={
        "system_prompt": "v2 prompt", "tools": ["read_file", "list_dir"]})
    lst = app__test_profiles_schedule.handle("GET", "/api/profiles").body["profiles"]
    assert len(lst) == 1 and lst[0]["system_prompt"] == "v2 prompt"
    assert lst[0]["tools"] == ["list_dir", "read_file"]

    # default: unknown name rejected, known accepted, cleared with null
    assert app__test_profiles_schedule.handle("PUT", "/api/profiles/default",
                      body={"default": "ghost"}).status == 404
    r = app__test_profiles_schedule.handle("PUT", "/api/profiles/default", body={"default": "coder"})
    assert r.status == 200
    assert app__test_profiles_schedule.handle("GET", "/api/profiles").body["default_profile"] == "coder"

    # deleting the default clears it
    assert app__test_profiles_schedule.handle("DELETE", "/api/profiles/coder").status == 200
    d = app__test_profiles_schedule.handle("GET", "/api/profiles").body
    assert d["profiles"] == [] and d["default_profile"] is None
    assert app__test_profiles_schedule.handle("DELETE", "/api/profiles/coder").status == 404


def test_profile_validation(app__test_profiles_schedule):
    assert app__test_profiles_schedule.handle("PUT", "/api/profiles/bad name!", body={}).status == 400
    assert app__test_profiles_schedule.handle("PUT", "/api/profiles/ok",
                      body={"tools": "notalist"}).status == 400
    assert app__test_profiles_schedule.handle("PUT", "/api/profiles/ok",
                      body={"system_prompt": 5}).status == 400


# ------------------------------------------------------- chat stream + profile

class _RecordingHandler(BaseHTTPRequestHandler):
    """Answers plain-text completions and records every request payload."""
    payloads: list = []

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        body = json.dumps({"object": "list",
                           "data": [{"id": "fake-model", "object": "model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if self.path.rstrip("/") == "/tokenize":
            # Improvement #4: exact-token probes hit the server ROOT, not
            # /v1/chat/completions. Answer 404 (counter degrades to chars/4)
            # and keep them out of `payloads`: assertions below read chat
            # completion bodies.
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        type(self).payloads.append(payload)
        body = json.dumps({
            "id": "cmpl", "object": "chat.completion", "model": "fake-model",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "ok done"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_chat_stream_honors_profile(app__test_profiles_schedule, home__test_profiles_schedule, monkeypatch):
    """Profile system prompt + tool subset reach the wire (fallback client)."""
    from fake_server import FakeOpenAIServer

    _RecordingHandler.payloads = []

    app__test_profiles_schedule.handle("PUT", "/api/profiles/coder", body={
        "system_prompt": "PROFILE-PROMPT-MARKER",
        "tools": ["list_dir"],
    })
    with FakeOpenAIServer(handler_cls=_RecordingHandler) as fs:
        home__test_profiles_schedule.mkdir(parents=True, exist_ok=True)
        (home__test_profiles_schedule / "serve_state.json").write_text(json.dumps(
            {"mode": "attach", "base_url": fs.base_url,
             "model_name": "fake-model"}))
        r = app__test_profiles_schedule.handle("POST", "/api/chat/stream",
                       body={"prompt": "say hi", "profile": "coder"})
        assert r.status == 200 and r.stream is not None
        text = "".join(r.stream)

    assert "run.done" in text
    # unknown profile => clean 404 envelope, not a stream error
    bad = app__test_profiles_schedule.handle("POST", "/api/chat/stream",
                     body={"prompt": "x", "profile": "ghost"})
    assert bad.status == 404 and bad.body["error"]["code"] == "unknown_profile"

    payloads = _RecordingHandler.payloads
    assert payloads, "fake server never called"
    first = payloads[0]
    sys_msgs = [m for m in first["messages"] if m.get("role") == "system"]
    assert sys_msgs and sys_msgs[0]["content"] == "PROFILE-PROMPT-MARKER"
    tool_names = {t["function"]["name"] for t in first.get("tools", [])}
    assert tool_names == {"list_dir"}  # read_file subsetted away


# ------------------------------------------------------ config unset semantics

def test_config_patch_null_unsets_key(app__test_profiles_schedule, home__test_profiles_schedule):
    home__test_profiles_schedule.mkdir(parents=True, exist_ok=True)
    (home__test_profiles_schedule / "toaster_loop.yaml").write_text(
        "attach_url: http://127.0.0.1:9/v1\nport: 1234\n")
    r = app__test_profiles_schedule.handle("PUT", "/api/config", body={
        "section": "root", "patch": {"attach_url": None}})
    assert r.status == 200
    import yaml

    data = yaml.safe_load((home__test_profiles_schedule / "toaster_loop.yaml").read_text())
    assert "attach_url" not in data and data["port"] == 1234


# --------------------------------------------------- auto-check --trigger CLI

STUB_WORKER__test_profiles_schedule = r'''
import argparse, json, sys
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("--job", required=True); ap.add_argument("--kind", required=True)
ap.add_argument("--config", default=""); ap.add_argument("--dataset-dir", default=None)
ap.add_argument("--base-model", default=None); ap.add_argument("--log-path", default=None)
ap.add_argument("--adapter-dir", default=None); ap.add_argument("--gguf-out", default=None)
ap.add_argument("--quant", default="q4_k_m"); ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()
line = json.dumps({"event": "done", "job": a.job})
print(line, flush=True)
if a.log_path:
    Path(a.log_path).parent.mkdir(parents=True, exist_ok=True)
    open(a.log_path, "a").write(line + "\n")
'''


def _run_auto_check(monkeypatch, home, trigger=False, dry_run=False):
    import toaster_loop.cli as cli

    monkeypatch.setenv("TOASTER_LOOP_HOME", str(home))
    monkeypatch.setenv("TOASTER_LOOP_CONFIG", str(home / "toaster_loop.yaml"))
    argv = ["train", "auto-check"]
    if trigger:
        argv.append("--trigger")
    if dry_run:
        argv.append("--dry-run")
    return cli.main(argv)


def test_auto_check_trigger_only_when_gate_ok(monkeypatch, home__test_profiles_schedule, tmp_path, capsys):
    # thresholds low enough that a couple of good traces suffice
    home__test_profiles_schedule.mkdir(parents=True, exist_ok=True)
    (home__test_profiles_schedule / "toaster_loop.yaml").write_text(
        "training:\n  auto_min_traces: 2\n  auto_min_score: 0.5\n")
    stub = tmp_path / "stub_worker.py"
    stub.write_text(STUB_WORKER__test_profiles_schedule)
    monkeypatch.setenv("TOASTER_LOOP_WORKER_COMMAND", f"{sys.executable} {stub}")

    # 1) not enough traces -> no trigger, exit 0 with --trigger, no jobs
    rc = _run_auto_check(monkeypatch, home__test_profiles_schedule, trigger=True, dry_run=True)
    assert rc == 0
    out = capsys.readouterr().out
    assert "no trigger" in out and "iterate" not in out.lower().split("no trigger")[1]

    # 2) enough good traces -> --trigger runs iterate (dry-run, stub worker)
    store = TraceStore(home__test_profiles_schedule / "buffer" / "toaster_loop.db")
    seed_trace(store, "t1", score=0.9)
    seed_trace(store, "t2", score=0.8)
    seed_trace(store, "t3", score=0.7)
    store.close()
    rc = _run_auto_check(monkeypatch, home__test_profiles_schedule, trigger=True, dry_run=True)
    out = capsys.readouterr().out
    assert rc == 0 and "starting iterate loop" in out
    assert '"stage": "train"' in out or "train" in out  # pipeline report printed
    # an iterate job actually ran through the scheduler/registry
    from toaster_loop.registry import ModelRegistry

    reg = ModelRegistry(home__test_profiles_schedule / "buffer" / "toaster_loop.db")
    try:
        kinds = [j.kind for j in reg.list_jobs()]
        states = [j.state for j in reg.list_jobs()]
    finally:
        reg.close()
    assert "sft" in kinds and "done" in states

    # 3) resource gate blocked -> trigger reports and exits without running
    (home__test_profiles_schedule / "toaster_loop.yaml").write_text(
        "training:\n  auto_min_traces: 2\n  auto_min_score: 0.5\n"
        "  min_free_vram_mib: 999999999\n")
    monkeypatch.setattr("toaster_loop.training.scheduler.gpu_free_mib",
                        lambda: 1, raising=True)
    store = TraceStore(home__test_profiles_schedule / "buffer" / "toaster_loop.db")
    seed_trace(store, "t4", score=0.9)
    seed_trace(store, "t5", score=0.9)
    # unclaim the earlier ones so curation has material again
    for tid in ("t1", "t2", "t3"):
        store._conn.execute("UPDATE traces SET curated=0 WHERE trace_id=?", (tid,))
    store._conn.commit()
    store.close()
    rc = _run_auto_check(monkeypatch, home__test_profiles_schedule, trigger=True, dry_run=True)
    out = capsys.readouterr().out
    assert rc == 0 and "gate blocked" in out
    assert "starting iterate loop" not in out


# ------------------------------------------------------------- static hygiene

def test_stub_panel_removed_and_nav_complete():
    assert not (STATIC / "js" / "panels" / "stub.js").exists(), \
        "stub panel must be gone after 5e"
    app_js = (STATIC / "app.js").read_text()
    assert "stub.js" not in app_js
    for panel in ("overview", "chat", "serve", "models", "jobs", "traces",
                  "datasets", "train", "iterate", "schedule", "profiles",
                  "settings", "hardware"):
        assert f'["{panel}",' in app_js, f"{panel} missing from nav"
    # every registered panel file exists and is imported
    import re

    for m in re.finditer(r'import \w+ from "\./js/panels/(\w+)\.js"', app_js):
        assert (STATIC / "js" / "panels" / f"{m.group(1)}.js").is_file()


def test_profiles_routes_registered(app__test_profiles_schedule):
    paths = {(m, p) for (m, p, _) in app__test_profiles_schedule.ROUTES}
    assert ("GET", "/api/profiles") in paths
    assert ("PUT", "/api/profiles/<name>") in paths
    assert ("DELETE", "/api/profiles/<name>") in paths
    assert ("PUT", "/api/profiles/default") in paths

# ------------------------------------------------------------------------
# from tests/test_batch_run.py
# ------------------------------------------------------------------------

@pytest.fixture()
def _no_sleeps(monkeypatch):
    """Retry backoff must not slow the suite down (cf. test_resilience)."""
    from toaster_loop.core import agent_runner as ar
    monkeypatch.setattr(ar.ToasterLoopAgentClient, "_sleep",
                        staticmethod(lambda s: None))


# ----------------------------------------------------------------- helpers

OK_MSG = {"role": "assistant", "content": "all done"}


def make_run_fn(home: Path, base_url: str, max_iterations: int = 3):
    """run_fn(contract) -> real runner+recorder per task (like the CLI)."""

    def run_fn(task):
        rec = TraceRecorder(home=home)
        client = ToasterLoopAgentClient(base_url=base_url, model="fake-model",
                                    recorder=rec, max_iterations=max_iterations)
        result = client.run(task)
        return getattr(result, "trace_id", None), result.failure_reason

    return run_fn


def write_tasks(path: Path, lines: list) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def seed_trace__test_batch_run(home: Path, task: str, failure_reason):
    """Record one trace with a given failure_reason (no server involved)."""
    rec = TraceRecorder(home=home)
    row = rec.record(task=task, messages=[{"role": "user", "content": task}],
                     backend="toaster_loop", model_alias="m",
                     base_url="http://127.0.0.1:1/v1", completed=False,
                     failure_reason=failure_reason)
    assert row is not None
    return row


def cp_tasks(home: Path, run_name: str) -> list:
    cp = load_checkpoint(checkpoint_path(home, run_name))
    return (cp or {}).get("tasks", [])


# -------------------------------------------------------------- input parsing

def test_load_tasks_object_bare_and_blank_lines(tmp_path, _no_sleeps):
    p = tmp_path / "tasks.jsonl"
    p.write_text(
        '{"task": "alpha"}\n'
        "\n   \n"
        '"beta bare string"\n'
        '{"task": "gamma", "extra": "ignored in v0"}\n',
        encoding="utf-8",
    )
    assert load_tasks(p) == ["alpha", "beta bare string", "gamma"]


def test_load_tasks_malformed_json_names_line(tmp_path, _no_sleeps):
    p = tmp_path / "tasks.jsonl"
    write_tasks(p, ['{"task": "ok"}', "{not json", '{"task": "x"}'])
    with pytest.raises(BatchInputError) as ei:
        load_tasks(p)
    assert ":2:" in str(ei.value)


def test_load_tasks_object_without_task_names_line(tmp_path, _no_sleeps):
    p = tmp_path / "tasks.jsonl"
    write_tasks(p, ["42"])
    with pytest.raises(BatchInputError) as ei:
        load_tasks(p)
    assert ":1:" in str(ei.value)


def test_load_tasks_missing_file(_no_sleeps):
    with pytest.raises(BatchInputError, match="not found"):
        load_tasks(Path("/nonexistent/tasks.jsonl"))


# ------------------------------------------------------------------ happy path

def test_batch_happy_path_traces_and_incremental_checkpoint(toaster_loop_home, _no_sleeps):
    tasks = ["task alpha", "task beta", "task gamma"]
    handler = make_message_handler({"*": OK_MSG})
    counts = []  # checkpoint record count observed after each task

    def run_fn(task):
        tid, reason = make_run_fn(toaster_loop_home, srv.base_url)(task)
        counts.append(len(cp_tasks(toaster_loop_home, "run1")))
        return tid, reason

    with FakeOpenAIServer(handler_cls=handler) as srv:
        lines = []
        rc = run_batch(tasks, home=toaster_loop_home, run_name="run1",
                       run_fn=run_fn, print_fn=lambda *a: lines.append(a[0]))

    assert rc == 0
    # traces: three ordinary buffer records, one per task
    traces = list((toaster_loop_home / "buffer").rglob("*.jsonl"))
    assert len(traces) == 3
    # checkpoint grew incrementally: 1, 2, 3 (written AFTER each task)
    assert counts == [0, 1, 2]  # observed before this task's own save
    final = cp_tasks(toaster_loop_home, "run1")
    assert len(final) == 3
    assert {r["task"] for r in final} == set(tasks)
    assert all(r["trace_id"] and not r["failure_reason"] for r in final)
    # progress lines + summary
    assert any("batch 2/3: task beta ... ok" in l for l in lines)
    assert "ran=3 skipped=0 retried-infra=0 failed-model=0 failed-infra=0" \
        in lines[-1]


def test_batch_no_resume_ignores_prior_traces(toaster_loop_home, _no_sleeps):
    seed_trace__test_batch_run(toaster_loop_home, "already done", None)
    seen = []

    def run_fn(task):
        seen.append(task)
        return "t1", None

    rc = run_batch(["already done"], home=toaster_loop_home, run_name="r",
                   resume=False, run_fn=run_fn, print_fn=lambda *a: None)
    assert rc == 0
    assert seen == ["already done"]  # fresh intent: nothing skipped


# ------------------------------------------------------------------- resume

def test_resume_after_crash_skips_completed_by_content(toaster_loop_home, _no_sleeps):
    tasks = ["crash one", "crash two", "crash three"]
    handler = make_message_handler({"*": OK_MSG})

    def crashy_run_fn(task):
        if task == "crash three":
            raise SystemExit(9)  # simulate process death mid-task
        return make_run_fn(toaster_loop_home, srv.base_url)(task)

    with FakeOpenAIServer(handler_cls=handler) as srv:
        with pytest.raises(SystemExit):
            run_batch(tasks, home=toaster_loop_home, run_name="crashrun",
                      run_fn=crashy_run_fn, print_fn=lambda *a: None)
        # crash left exactly the two finished tasks checkpointed
        assert len(cp_tasks(toaster_loop_home, "crashrun")) == 2

        handler.requests.clear()
        lines = []
        rc = run_batch(tasks, home=toaster_loop_home, run_name="crashrun",
                       resume=True,
                       run_fn=make_run_fn(toaster_loop_home, srv.base_url),
                       print_fn=lambda *a: lines.append(a[0]))

    assert rc == 0
    # only task 3's request hit the server this time (one chat call)
    last_users = [
        next((m["content"] for m in reversed(p["messages"])
              if m.get("role") == "user"), "")
        for p in handler.requests
    ]
    assert last_users == ["crash three"]
    assert any("skipped(resume)" in l and "crash one" in l for l in lines)
    assert any("skipped(resume)" in l and "crash two" in l for l in lines)
    assert len(cp_tasks(toaster_loop_home, "crashrun")) == 3


def test_resume_matches_reordered_and_duplicated_lines(toaster_loop_home, _no_sleeps):
    handler = make_message_handler({"*": OK_MSG})
    with FakeOpenAIServer(handler_cls=handler) as srv:
        run_batch(["task A", "task B"], home=toaster_loop_home, run_name="ord",
                  run_fn=make_run_fn(toaster_loop_home, srv.base_url),
                  print_fn=lambda *a: None)

        handler.requests.clear()
        # reordered + duplicated input; content matching, not index
        lines = []
        rc = run_batch(["task B", "task A", "task B"], home=toaster_loop_home,
                       run_name="ord", resume=True,
                       run_fn=make_run_fn(toaster_loop_home, srv.base_url),
                       print_fn=lambda *a: lines.append(a[0]))
    assert rc == 0
    assert sum("skipped(resume)" in l for l in lines) == 3
    # no new requests reached the server on resume
    assert handler.requests == []


def test_infra_failure_retried_on_resume(toaster_loop_home, _no_sleeps):
    seed_trace__test_batch_run(toaster_loop_home, "flaky task", "backend_error:connection")
    completed = scan_completed(toaster_loop_home)
    assert task_key("flaky task") not in completed  # infra != done

    seen = []

    def ok_run_fn(task):
        seen.append(task)
        return "t-new", None

    rc = run_batch(["flaky task"], home=toaster_loop_home, run_name="infra",
                   resume=True, run_fn=ok_run_fn, print_fn=lambda *a: None)
    assert seen == ["flaky task"]  # re-run, not skipped
    assert rc == 0


def test_infra_failure_exits_1_and_is_not_checkpointed_done(toaster_loop_home, _no_sleeps):
    def dead_run_fn(task):
        return make_run_fn(toaster_loop_home, "http://127.0.0.1:9/v1")(task)

    lines = []
    rc = run_batch(["will fail infra"], home=toaster_loop_home, run_name="dead",
                   resume=False, run_fn=dead_run_fn,
                   print_fn=lambda *a: lines.append(a[0]))
    assert rc == 1  # infra failure remaining -> exit 1
    assert any("failed(backend_error:connection)" in l for l in lines)
    # the trace exists but is infra-marked -> a resume scan sees it as undone
    assert task_key("will fail infra") not in scan_completed(toaster_loop_home)


def test_model_failure_counts_as_done_not_retried(toaster_loop_home, _no_sleeps):
    for reason in ("guardrail_halt", "max_iterations"):
        seed_trace__test_batch_run(toaster_loop_home, f"model-ish {reason}", reason)
    completed = scan_completed(toaster_loop_home)
    assert task_key("model-ish guardrail_halt") in completed
    assert task_key("model-ish max_iterations") in completed

    seen = []

    def run_fn(task):
        seen.append(task)
        return None, None

    lines = []
    rc = run_batch(["model-ish guardrail_halt", "model-ish max_iterations"],
                   home=toaster_loop_home, run_name="modelfail", resume=True,
                   run_fn=run_fn, print_fn=lambda *a: lines.append(a[0]))
    assert seen == []  # signal, not transport noise -> done
    assert rc == 0
    assert sum("skipped(resume)" in l for l in lines) == 2


def test_model_failure_via_real_runner_skipped_on_resume(toaster_loop_home, _no_sleeps):
    """End-to-end: a never-converging server yields max_iterations; resume
    treats the task as done."""
    handler = make_message_handler(
        {"*": {"role": "assistant", "content": None,
               "tool_calls": [{"id": "c", "type": "function",
                               "function": {"name": "list_dir",
                                            "arguments": "{}"}}]}})
    tasks = ["loop task"]
    with FakeOpenAIServer(handler_cls=handler) as srv:
        lines = []
        rc = run_batch(tasks, home=toaster_loop_home, run_name="looprun",
                       run_fn=make_run_fn(toaster_loop_home, srv.base_url,
                                          max_iterations=2),
                       print_fn=lambda *a: lines.append(a[0]))
        assert rc == 0  # model failure is terminal, not an error exit
        assert any("failed(max_iterations)" in l for l in lines)

        handler.requests.clear()
        lines = []
        rc = run_batch(tasks, home=toaster_loop_home, run_name="looprun",
                       resume=True,
                       run_fn=make_run_fn(toaster_loop_home, srv.base_url),
                       print_fn=lambda *a: lines.append(a[0]))
    assert rc == 0
    assert handler.requests == []
    assert any("skipped(resume)" in l for l in lines)


# --------------------------------------------------------------- checkpoint

def test_checkpoint_atomic_write_and_schema(toaster_loop_home, _no_sleeps):
    def run_fn(task):
        return "trace-xyz", None

    run_batch(["one"], home=toaster_loop_home, run_name="cp", resume=False,
              run_fn=run_fn, input_path="/tmp/tasks.jsonl",
              print_fn=lambda *a: None)
    cp_file = checkpoint_path(toaster_loop_home, "cp")
    data = json.loads(cp_file.read_text())
    assert data["schema_version"] == 1
    assert data["run_name"] == "cp"
    assert data["input"] == "/tmp/tasks.jsonl"
    assert data["started_at"] and data["last_updated"]
    rec = data["tasks"][0]
    assert rec["task"] == "one"
    assert rec["trace_id"] == "trace-xyz"
    assert rec["failure_reason"] is None
    assert rec["completed_at"]
    assert len(rec["prompt_hash"]) == 64
    assert not list(cp_file.parent.glob("*.tmp"))  # tmp file replaced


def test_corrupt_checkpoint_degrades_to_content_scan(toaster_loop_home, _no_sleeps):
    cp = checkpoint_path(toaster_loop_home, "bad")
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text("{not json", encoding="utf-8")
    seed_trace__test_batch_run(toaster_loop_home, "seeded task", None)

    seen = []

    def run_fn(task):
        seen.append(task)
        return "t", None

    rc = run_batch(["seeded task"], home=toaster_loop_home, run_name="bad",
                   resume=True, run_fn=run_fn, print_fn=lambda *a: None)
    assert rc == 0
    assert seen == []  # scan still wins: content is truth


def test_resume_bounds_scan_to_checkpoint_started_at(toaster_loop_home, _no_sleeps):
    """An old trace (pre-batch) with identical text must not suppress a fresh
    batch task when a checkpoint bounds the scan window."""
    row = seed_trace__test_batch_run(toaster_loop_home, "classic task", None)
    # backdate the seeded trace header by a year
    path = next((toaster_loop_home / "buffer").rglob(f"{row.trace_id}.jsonl"))
    lines = path.read_text().splitlines()
    header = json.loads(lines[0])
    header["timestamp"] = "2020-01-01T00:00:00+00:00"
    lines[0] = json.dumps(header)
    path.write_text("\n".join(lines) + "\n")

    # checkpoint claiming a recent started_at -> old trace out of window
    from toaster_loop.core.batch import save_checkpoint
    save_checkpoint(checkpoint_path(toaster_loop_home, "win"), {
        "schema_version": 1, "run_name": "win", "input": None,
        "started_at": "2025-01-01T00:00:00+00:00", "tasks": [],
    })

    seen = []

    def run_fn(task):
        seen.append(task)
        return "t2", None

    rc = run_batch(["classic task"], home=toaster_loop_home, run_name="win",
                   resume=True, run_fn=run_fn, print_fn=lambda *a: None)
    assert rc == 0
    assert seen == ["classic task"]  # re-run: prior trace predates the run

# ------------------------------------------------------------------------
# from tests/test_iterate.py
# ------------------------------------------------------------------------

# Stub worker: same contract as tests/test_scheduler.py plus export GGUF copy.
STUB = r'''
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

# fail knobs via env file (subprocess env is inherited; tests write markers)
marker = os.environ.get("STUB_FAIL_FILE")
if marker and Path(marker).is_file():
    emit("error", error="stub failure")
    sys.exit(2)

emit("start")
if a.kind in ("sft", "dpo"):
    out = Path(a.config).parent if a.config else Path(".")
    (out / "adapter").mkdir(parents=True, exist_ok=True)
    (out / "adapter" / "adapter_config.json").write_text(json.dumps(
        {"base_model": a.base_model or "stub", "r": 8}))
elif a.kind == "export":
    gg = Path(a.gguf_out)
    gg.parent.mkdir(parents=True, exist_ok=True)
    # "convert" by copying the fixture GGUF the test points at
    fixture = os.environ.get("STUB_FIXTURE_GGUF")
    if fixture and Path(fixture).is_file():
        gg.write_bytes(Path(fixture).read_bytes())
    else:
        gg.write_bytes(b"GGUF-stub")
emit("done")
'''


@pytest.fixture()
def env(tmp_path, monkeypatch):
    home = tmp_path / "uf-iterate"
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(home))
    stub = tmp_path / "stub.py"
    stub.write_text(STUB)
    fixture_gguf = tmp_path / "tiny.gguf"
    fixture_gguf.write_bytes(b"GGUF" + b"\x00" * 32)
    monkeypatch.setenv("STUB_FIXTURE_GGUF", str(fixture_gguf))
    yield home, stub, fixture_gguf


def seed_trace__test_iterate(store, home, i, messages, score=0.9):
    ts = f"2026-09-01T1{i:09d}+00:00"[:25]
    p = home / "buffer" / "2026" / "09" / f"iter-{i}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        f.write(json.dumps({"kind": "trace_header"}) + "\n")
        for m in messages:
            f.write(json.dumps({"kind": "message", "message": m}) + "\n")
        f.write(json.dumps({"kind": "trace_footer", "status": "success"}) + "\n")
    store.insert_trace(TraceRow(
        trace_id=f"iter-{i}", session_id=None, timestamp=ts,
        agent_profile="fallback", model_alias="base-model",
        model_version=None, endpoint_label=None, task_summary="t",
        status="success", outcome_score=score, num_tool_calls=0,
        tokens_in=None, tokens_out=None, duration_s=None, jsonl_path=str(p),
    ))


def good_trace(n):
    return [
        {"role": "system", "content": "helpful"},
        {"role": "user", "content": f"task number {n}"},
        {"role": "assistant", "content": f"done with task {n} result=ok"},
    ]


GOLDEN = [{"prompt": "anything", "expect_contains": ["ok-answer"]}]


def starter_factory(reply="ok-answer"):
    def starter(gguf_path):
        srv = FakeOpenAIServer(handler_cls=make_scripted_handler(
            {"*": reply}))
        srv.start()
        return srv
    return starter


_starter_replies = starter_factory


def run_loop(home, stub, registry, store, kind="sft", dataset_id=None,
             min_pass=1.0, fail_file=None):
    import os

    if fail_file:
        os.environ["STUB_FAIL_FILE"] = str(fail_file)
    try:
        sched = Scheduler(registry=registry, cfg=TrainingConfig(
            base_model="Base/HF-Model", gguf_quant="q4_k_m"),
            worker_command=[sys.executable, str(stub)])
        return run_iterate(
            home=home, registry=registry, scheduler=sched, store=store,
            curation_cfg=CurationConfig(min_score=0.5),
            training_cfg=TrainingConfig(base_model="Base/HF-Model"),
            validation_cfg=ValidationConfig(min_pass_rate=min_pass,
                                            golden_tasks=GOLDEN),
            name="run", kind=kind, dataset_id=dataset_id,
            server_starter=starter_factory(),
        )
    finally:
        if fail_file:
            os.environ.pop("STUB_FAIL_FILE", None)


def test_iterate_full_loop_v1_then_v2_then_rollback(env):
    home, stub, fixture = env
    store = TraceStore(home / "buffer" / "toaster_loop.db")
    reg = ModelRegistry(home / "buffer" / "toaster_loop.db")
    for i in range(3):
        seed_trace__test_iterate(store, home, i, good_trace(i))

    r1 = run_loop(home, stub, reg, store)
    assert r1.ok and r1.promoted, r1.to_dict()
    assert r1.model_id == "v1"
    assert r1.stages == ["curate", "register", "train", "export",
                         "validate", "promote"]
    # artifacts exist
    assert r1.gguf_path is not None
    gguf = Path(r1.gguf_path)
    assert gguf.is_file() and gguf.read_bytes().startswith(b"GGUF")
    assert r1.adapter_dir is not None
    assert Path(r1.adapter_dir, "adapter_config.json").is_file()
    active = reg.active_model()
    assert active is not None and active.model_id == "v1"
    assert r1.validation is not None
    assert r1.validation["pass_rate"] == 1.0

    # second iterate -> v2 with lineage to v1 (needs fresh traces: claims are
    # consumed; seed two more)
    for i in range(3, 5):
        seed_trace__test_iterate(store, home, i, good_trace(i))
    r2 = run_loop(home, stub, reg, store)
    assert r2.ok and r2.promoted
    assert r2.model_id == "v2"
    m2 = reg.get_model("v2")
    assert m2 is not None and m2.parent_adapter == "v1"
    active = reg.active_model()
    assert active is not None and active.model_id == "v2"
    m1 = reg.get_model("v1")
    assert m1 is not None and m1.status == "superseded"

    # rollback -> active pointer back to v1
    rb = reg.rollback()
    assert rb is not None and rb.model_id == "v1"
    active = reg.active_model()
    assert active is not None and active.model_id == "v1"
    store.close()
    reg.close()


def test_iterate_validation_failure_blocks_promotion(env, tmp_path):
    home, stub, fixture = env
    store = TraceStore(home / "buffer" / "toaster_loop.db")
    reg = ModelRegistry(home / "buffer" / "toaster_loop.db")
    for i in range(2):
        seed_trace__test_iterate(store, home, i, good_trace(i))

    # golden task that the scripted server can never satisfy
    sched = Scheduler(registry=reg, cfg=TrainingConfig(base_model="B/M"),
                      worker_command=[sys.executable, str(stub)])
    res = run_iterate(
        home=home, registry=reg, scheduler=sched, store=store,
        curation_cfg=CurationConfig(min_score=0.5),
        training_cfg=TrainingConfig(base_model="B/M"),
        validation_cfg=ValidationConfig(
            min_pass_rate=1.0,
            golden_tasks=[{"prompt": "q", "expect_contains": ["never-served"]}],
        ),
        name="bad", server_starter=_starter_replies("ok"),
    )
    assert not res.ok and res.stage == "validate"
    assert res.model_id is not None
    failed = reg.get_model(res.model_id)
    assert failed is not None and failed.status == "failed"
    assert reg.active_model() is None  # nothing promoted
    store.close()
    reg.close()


def test_iterate_train_failure_keeps_candidate_unpromoted(env):
    home, stub, fixture = env
    store = TraceStore(home / "buffer" / "toaster_loop.db")
    reg = ModelRegistry(home / "buffer" / "toaster_loop.db")
    for i in range(2):
        seed_trace__test_iterate(store, home, i, good_trace(i))
    fail = tmp_marker = home / "FAIL"
    fail.write_text("")
    r = run_loop(home, stub, reg, store, fail_file=fail)
    assert not r.ok and r.stage == "train"
    assert r.model_id is not None
    m = reg.get_model(r.model_id)
    assert m is not None and m.status == "failed"  # validation error recorded by pipeline
    assert reg.active_model() is None
    store.close()
    reg.close()


def test_iterate_export_failure_keeps_adapter(env, monkeypatch):
    home, stub, fixture = env
    store = TraceStore(home / "buffer" / "toaster_loop.db")
    reg = ModelRegistry(home / "buffer" / "toaster_loop.db")
    for i in range(2):
        seed_trace__test_iterate(store, home, i, good_trace(i))

    # worker that trains fine but fails export
    bad_stub = home.parent / "bad_export.py"
    bad_stub.write_text(STUB.replace('elif a.kind == "export":',
                                     'elif a.kind == "export":\n    sys.exit(7)'))
    sched = Scheduler(registry=reg, cfg=TrainingConfig(base_model="B/M"),
                      worker_command=[sys.executable, str(bad_stub)])
    res = run_iterate(
        home=home, registry=reg, scheduler=sched, store=store,
        curation_cfg=CurationConfig(min_score=0.5),
        training_cfg=TrainingConfig(base_model="B/M"),
        validation_cfg=ValidationConfig(min_pass_rate=1.0,
                                        golden_tasks=GOLDEN),
        name="x", server_starter=starter_factory(),
    )
    assert not res.ok and res.stage == "export"
    # adapter artifact kept; model still candidate (data fine, GGUF missing)
    assert res.adapter_dir is not None
    assert Path(res.adapter_dir).is_dir()
    assert res.model_id is not None
    cand = reg.get_model(res.model_id)
    assert cand is not None and cand.status == "candidate"
    store.close()
    reg.close()


def test_iterate_dry_run_stops_before_training_artifacts(env):
    home, stub, fixture = env
    store = TraceStore(home / "buffer" / "toaster_loop.db")
    reg = ModelRegistry(home / "buffer" / "toaster_loop.db")
    for i in range(2):
        seed_trace__test_iterate(store, home, i, good_trace(i))
    sched = Scheduler(registry=reg, cfg=TrainingConfig(base_model="B/M"),
                      worker_command=[sys.executable, str(stub)])
    res = run_iterate(
        home=home, registry=reg, scheduler=sched, store=store,
        curation_cfg=CurationConfig(min_score=0.5),
        training_cfg=TrainingConfig(base_model="B/M"),
        validation_cfg=ValidationConfig(min_pass_rate=1.0,
                                        golden_tasks=GOLDEN),
        name="dry", dry_run=True, server_starter=starter_factory(),
    )
    assert res.ok and not res.promoted
    assert res.stages == ["curate", "register", "train"]
    assert res.gguf_path is None
    store.close()
    reg.close()

# ------------------------------------------------------------------------
# from tests/test_verification_nudge.py
# ------------------------------------------------------------------------

# ------------------------------------------------------------ path policy

@pytest.mark.parametrize("path,want", [
    ("src/app.py", True),
    ("pkg/mod.js", True),
    ("x.html", True),
    ("Makefile", True),
    ("dockerfile", True),
    ("README.md", False),
    ("docs/guide.mdx", False),
    ("data/rows.jsonl", False),
    ("config.yaml", False),
    ("LICENSE", False),
    ("CHANGELOG", False),
    ("notes.txt", False),
    ("", False),
])
def test_is_code_path(path, want):
    assert is_code_path(path) is want


# ---------------------------------------------------------------- ledger

def test_ledger_edit_marks_dirty_and_nudge_lists_paths():
    led = VerificationLedger()
    led.observe("write_file", {"path": "a.py"}, "wrote a.py\n✓ syntax ok")
    assert led.pending_paths() == []          # fresh passing check
    led.observe("write_file", {"path": "b.py"}, "wrote b.py\n✗ SYNTAX ERROR x")
    assert led.pending_paths() == ["b.py"]
    nudge = led.build_nudge()
    assert nudge is not None and "b.py" in nudge
    assert led.build_nudge() is None          # exactly one nudge


def test_ledger_unverdict_edit_is_unverified():
    led = VerificationLedger()
    led.observe("write_file", {"path": "notes.md"}, "wrote notes.md")
    assert led.pending_paths() == []          # prose exempt
    led.observe("patch", {"path": "c.ts"}, "patched c.ts")  # no ✓/✗ verdict
    assert led.pending_paths() == ["c.ts"]


def test_ledger_failed_edit_leaves_no_trace():
    led = VerificationLedger()
    led.observe("write_file", {"path": "d.py"}, "error: workspace containment")
    assert led.pending_paths() == []


def test_ledger_run_command_clears_evidence():
    led = VerificationLedger()
    led.observe("patch", {"path": "e.py"}, "patched e.py\n✗ SYNTAX ERROR x")
    assert led.pending_paths() == ["e.py"]
    led.observe("run_command", {"command": "pytest"}, "all tests passed")
    assert led.pending_paths() == []
    # A FAILED command is not evidence.
    led.observe("patch", {"path": "f.py"}, "patched f.py")
    led.observe("run_command", {"command": "pytest"}, "error: boom")
    assert led.pending_paths() == ["f.py"]


# ------------------------------------------------------------ runner loop

def _tool_call_msg(name, args):
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_1", "type": "function",
         "function": {"name": name, "arguments": json.dumps(args)}}]}


class ReplySequenceHandler(BaseHTTPRequestHandler):
    """Replies per request index from a class-level script (then repeats last).

    Also records the messages of every request so tests can inspect whether
    the nudge was injected and where."""

    script: list = []
    requests: list = []

    def log_message(self, format, *args):  # noqa: A002 - silence access log
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        idx = len(type(self).requests)
        type(self).requests.append(payload.get("messages") or [])
        msg = type(self).script[min(idx, len(type(self).script) - 1)]
        finish = "tool_calls" if msg.get("tool_calls") else "stop"
        body = json.dumps({
            "id": "x", "object": "chat.completion", "model": "m",
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _client(script, tmp_path, monkeypatch, **kw):
    """Client with a stubbed write_file impl (no real FS needed)."""

    def fake_write(path, content, **_):
        # Mirrors the real post-write checker contract: a plain write carries
        # no verdict (dirty until something verifies it); writing the literal
        # content "VERIFIED" simulates a run whose check passed (✓ mark).
        if content == "VERIFIED":
            return f"wrote {path}\n✓ syntax ok"
        return f"wrote {path}"

    ReplySequenceHandler.script = script
    ReplySequenceHandler.requests = []
    fs = FakeOpenAIServer(handler_cls=ReplySequenceHandler)
    fs.start()
    client = ToasterLoopAgentClient(
        fs.base_url, "m",
        tools=[{"type": "function", "function": {
            "name": "write_file", "description": "w",
            "parameters": {"type": "object",
                           "properties": {"path": {"type": "string"},
                                          "content": {"type": "string"}}}}}],
        tool_impls={"write_file": fake_write},
        max_iterations=6, **kw)
    return fs, client


def test_edit_then_done_gets_one_nudge_then_verified_finish(tmp_path):
    script = [
        _tool_call_msg("write_file", {"path": "a.py", "content": "x"}),
        {"role": "assistant", "content": "all done!"},          # unverified
        _tool_call_msg("write_file", {"path": "a.py",
                                      "content": "VERIFIED"}),  # fresh ✓
        {"role": "assistant", "content": "verified via re-write"},
    ]
    fs, client = _client(script, tmp_path, None)
    try:
        res = client.run("edit a.py")
    finally:
        fs.stop()
    assert res.completed and res.failure_reason is None
    # The nudge went in as a synthetic user message after the first "done".
    nudges = [m for m in res.messages
              if m.get("role") == "user" and "verification" in
              str(m.get("content", "")).lower()]
    assert len(nudges) == 1
    assert "a.py" in nudges[0]["content"]
    assert res.final_response == "verified via re-write"


def test_second_unverified_finish_records_failure_reason(tmp_path):
    script = [
        _tool_call_msg("write_file", {"path": "b.py", "content": "x"}),
        {"role": "assistant", "content": "done!"},   # unverified -> nudge
        {"role": "assistant", "content": "really done"},  # still unverified
    ]
    fs, client = _client(script, tmp_path, None)
    try:
        res = client.run("edit b.py")
    finally:
        fs.stop()
    assert res.completed is True
    assert res.failure_reason == "verification_nudge"
    # Exactly one nudge in history — no nagging on the third turn.
    nudges = [m for m in res.messages if m.get("role") == "user"
              and "[System:" in str(m.get("content", ""))]
    assert len(nudges) == 1


def test_prose_edit_never_nudges(tmp_path):
    script = [
        _tool_call_msg("write_file", {"path": "README.md", "content": "x"}),
        {"role": "assistant", "content": "docs updated"},
    ]
    fs, client = _client(script, tmp_path, None)
    try:
        res = client.run("edit the readme")
    finally:
        fs.stop()
    assert res.failure_reason is None
    assert all("[System:" not in str(m.get("content", ""))
               for m in res.messages if m.get("role") == "user"
               and m["content"] != "edit the readme")


def test_nudge_disabled_by_flag(tmp_path):
    script = [
        _tool_call_msg("write_file", {"path": "c.py", "content": "x"}),
        {"role": "assistant", "content": "done!"},
    ]
    fs, client = _client(script, tmp_path, None, verification_nudge=False)
    try:
        res = client.run("edit c.py")
    finally:
        fs.stop()
    assert res.failure_reason is None
    assert res.final_response == "done!"

# ------------------------------------------------------------------------
# from tests/test_transfer.py
# ------------------------------------------------------------------------

@pytest.fixture()
def home__test_transfer(tmp_path, monkeypatch):
    h = tmp_path / "uf-transfer"
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(h))
    (h / "buffer").mkdir(parents=True)
    return h


# ------------------------------------------------------------------ seeding

def seed_trace__test_transfer(store, home, tid, *, ts, profile="writer", status="success",
               score: Optional[float] = 0.9, task=None):
    p = home / "buffer" / f"{tid}.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        f.write(json.dumps({"kind": "trace_header"}) + "\n")
        f.write(json.dumps({"kind": "message", "message": {
            "role": "user", "content": task or f"task {tid}"}}) + "\n")
        f.write(json.dumps({"kind": "message", "message": {
            "role": "assistant", "content": "the old answer"}}) + "\n")
    store.insert_trace(TraceRow(
        trace_id=tid, session_id=None, timestamp=ts, agent_profile=profile,
        model_alias="m", model_version=None, endpoint_label=None,
        task_summary="s", status=status, outcome_score=score,
        num_tool_calls=0, tokens_in=None, tokens_out=None, duration_s=None,
        jsonl_path=str(p), failure_reason=None,
    ))


def _stub_judge_client_for(score_map):
    def client_for(base_url, model_name):
        def client(messages, model):
            text = json.dumps(messages)
            for needle, sc in score_map.items():
                if needle in text:
                    return json.dumps({"score": sc, "reason": "stub"})
            return json.dumps({"score": 0.5, "reason": "stub-mid"})
        return client
    return client_for


# ------------------------------------------------- transfer run (with baseline)

def test_transfer_probe_stores_entry_with_active_baseline(home__test_transfer):
    """Baseline default = the registry's CURRENT active model; improvement math
    and per-profile storage on transfers_json (not shadow_json)."""
    store = TraceStore(home__test_transfer / "buffer" / "toaster_loop.db")
    reg = ModelRegistry(home__test_transfer / "buffer" / "toaster_loop.db")
    seed_trace__test_transfer(store, home__test_transfer, "w1", ts="2026-09-01T00:00:00+00:00")
    seed_trace__test_transfer(store, home__test_transfer, "w2", ts="2026-09-02T00:00:00+00:00")

    base = reg.register_model("Base/Model")          # v1 -> active
    reg.set_validation(base.model_id, {"ok": True}, passed=True)
    reg.promote(base.model_id)
    cand = reg.register_model("Base/Model")          # v2 candidate

    cand_srv = FakeOpenAIServer(handler_cls=make_scripted_handler(
        {"*": "candidate reply: strong"})).start()
    base_srv = FakeOpenAIServer(handler_cls=make_scripted_handler(
        {"*": "baseline reply: weak"})).start()

    @contextmanager
    def endpoint_for(model):
        srv = cand_srv if model.model_id == cand.model_id else base_srv
        yield srv.base_url, model.model_id

    try:
        result = run_shadow_eval(
            reg, store, candidate_id=cand.model_id, profile="writer", n=6,
            home=home__test_transfer, endpoint_for=endpoint_for,
            judge_client_for=_stub_judge_client_for(
                {"strong": 0.8, "weak": 0.4}),
            store_as="transfer",
        )
    finally:
        cand_srv.stop()
        base_srv.stop()

    assert result["baseline_id"] == base.model_id
    assert result["profile"] == "writer"
    assert result["candidate_mean"] == pytest.approx(0.8)
    assert result["baseline_mean"] == pytest.approx(0.4)
    assert result["improvement"] == pytest.approx(0.4)
    assert result["is_regression"] is False

    row = reg.get_model(cand.model_id)
    assert row is not None
    assert row.shadow is None  # shadow_json untouched — own-profile verdict only
    assert len(row.transfers) == 1
    t = row.transfers[0]
    assert set(t) >= {"ts", "profile", "baseline_id", "n", "candidate_mean",
                      "baseline_mean", "improvement", "is_regression"}
    assert t["profile"] == "writer" and t["improvement"] == pytest.approx(0.4)

    # ledger records the op and the chain still verifies (#31)
    ops = [e["op"] for e in reg.history(limit=20)]
    assert "record_transfer" in ops
    ok, bad, _ = reg.verify_ledger_chain()
    assert ok, f"chain broken at {bad}"
    store.close()
    reg.close()


def test_transfer_probe_explicit_baseline_and_regression(home__test_transfer):
    store = TraceStore(home__test_transfer / "buffer" / "toaster_loop.db")
    reg = ModelRegistry(home__test_transfer / "buffer" / "toaster_loop.db")
    seed_trace__test_transfer(store, home__test_transfer, "w1", ts="2026-09-01T00:00:00+00:00")
    base = reg.register_model("Base/Model")          # v1 (NOT active)
    cand = reg.register_model("Base/Model")          # v2

    gen = lambda base_url, model_name, task: (
        "CANDIDATE" if model_name == cand.model_id else "BASELINE")

    @contextmanager
    def endpoint_for(model):
        yield "http://127.0.0.1:1/v1", model.model_id

    result = run_shadow_eval(
        reg, store, candidate_id=cand.model_id, profile="writer",
        home=home__test_transfer, endpoint_for=endpoint_for, generate_fn=gen,
        judge_client_for=_stub_judge_client_for({"CANDIDATE": 0.2,
                                                 "BASELINE": 0.9}),
        store_as="transfer", baseline_id=base.model_id,
    )
    assert result["baseline_id"] == base.model_id
    assert result["improvement"] == pytest.approx(-0.7)
    assert result["is_regression"] is True
    row = reg.get_model(cand.model_id)
    assert row is not None
    assert row.transfers[0]["is_regression"] is True
    store.close()
    reg.close()


# ------------------------------------------- candidate-only (no active model)

def test_transfer_probe_candidate_only_when_no_active(home__test_transfer):
    """No active model -> run candidate, store absolute scores with
    baseline_id null and is_regression null — no fabricated comparison."""
    store = TraceStore(home__test_transfer / "buffer" / "toaster_loop.db")
    reg = ModelRegistry(home__test_transfer / "buffer" / "toaster_loop.db")
    seed_trace__test_transfer(store, home__test_transfer, "w1", ts="2026-09-01T00:00:00+00:00")
    cand = reg.register_model("Base/Model")  # never promoted -> no active

    @contextmanager
    def endpoint_for(model):
        yield "http://127.0.0.1:1/v1", model.model_id

    result = run_shadow_eval(
        reg, store, candidate_id=cand.model_id, profile="writer",
        home=home__test_transfer, endpoint_for=endpoint_for,
        generate_fn=lambda *a: "CANDIDATE",
        judge_client_for=_stub_judge_client_for({"CANDIDATE": 0.7}),
        store_as="transfer",
    )
    assert result["baseline_id"] is None
    assert result["candidate_mean"] == pytest.approx(0.7)
    assert result["baseline_mean"] is None
    assert result["improvement"] is None
    assert result["is_regression"] is None  # documented: no comparison exists

    row = reg.get_model(cand.model_id)
    assert row is not None
    assert row.transfers[0]["baseline_id"] is None
    assert row.transfers[0]["candidate_mean"] == pytest.approx(0.7)
    store.close()
    reg.close()


def test_transfer_probe_candidate_skips_self_as_active(home__test_transfer):
    """The candidate itself being active proves nothing -> candidate-only."""
    store = TraceStore(home__test_transfer / "buffer" / "toaster_loop.db")
    reg = ModelRegistry(home__test_transfer / "buffer" / "toaster_loop.db")
    seed_trace__test_transfer(store, home__test_transfer, "w1", ts="2026-09-01T00:00:00+00:00")
    m = reg.register_model("Base/Model")
    reg.set_validation(m.model_id, {"ok": True}, passed=True)
    reg.promote(m.model_id)  # candidate IS the active model

    @contextmanager
    def endpoint_for(model):
        yield "http://127.0.0.1:1/v1", model.model_id

    result = run_shadow_eval(
        reg, store, candidate_id=m.model_id, profile="writer", home=home__test_transfer,
        endpoint_for=endpoint_for, generate_fn=lambda *a: "CANDIDATE",
        judge_client_for=_stub_judge_client_for({"CANDIDATE": 0.6}),
        store_as="transfer",
    )
    assert result["baseline_id"] is None and result["improvement"] is None
    store.close()
    reg.close()


# ------------------------------------------------------------- honest-empty

def test_transfer_probe_honest_empty_fails_and_stores_nothing(home__test_transfer):
    """Profile with no replay set -> job failure 'no replay set for profile P',
    nothing stored (a probe asked a question; silence would hide the answer)."""
    store = TraceStore(home__test_transfer / "buffer" / "toaster_loop.db")
    reg = ModelRegistry(home__test_transfer / "buffer" / "toaster_loop.db")
    m = reg.register_model("Base/Model")

    with pytest.raises(ShadowError) as ei:
        run_shadow_eval(reg, store, candidate_id=m.model_id,
                        profile="ghost", home=home__test_transfer, store_as="transfer")
    assert str(ei.value) == no_replay_set_for("ghost")
    assert "no replay set for profile" in str(ei.value)
    row = reg.get_model(m.model_id)
    assert row is not None
    assert row.transfers == [] and row.shadow is None

    # zero scored tasks must also fail, never store an empty verdict
    seed_trace__test_transfer(store, home__test_transfer, "w1", ts="2026-09-01T00:00:00+00:00")

    @contextmanager
    def endpoint_for(model):
        yield "http://127.0.0.1:1/v1", model.model_id

    with pytest.raises(ShadowError, match="zero scored"):
        run_shadow_eval(reg, store, candidate_id=m.model_id, profile="writer",
                        home=home__test_transfer, endpoint_for=endpoint_for,
                        generate_fn=lambda *a: None, store_as="transfer")
    row = reg.get_model(m.model_id)
    assert row is not None and row.transfers == []
    store.close()
    reg.close()


def test_transfer_probe_rejects_bad_baseline(home__test_transfer):
    store = TraceStore(home__test_transfer / "buffer" / "toaster_loop.db")
    reg = ModelRegistry(home__test_transfer / "buffer" / "toaster_loop.db")
    seed_trace__test_transfer(store, home__test_transfer, "w1", ts="2026-09-01T00:00:00+00:00")
    cand = reg.register_model("Base/Model")
    with pytest.raises(ShadowError, match="no model v9"):
        run_shadow_eval(reg, store, candidate_id=cand.model_id,
                        profile="writer", home=home__test_transfer, store_as="transfer",
                        baseline_id="v9")
    with pytest.raises(ShadowError, match="must differ"):
        run_shadow_eval(reg, store, candidate_id=cand.model_id,
                        profile="writer", home=home__test_transfer, store_as="transfer",
                        baseline_id=cand.model_id)
    store.close()
    reg.close()


# ------------------------------------------------------------------ cap

def test_record_transfer_cap_evicts_oldest(home__test_transfer):
    reg = ModelRegistry(home__test_transfer / "buffer" / "toaster_loop.db")
    m = reg.register_model("Base/Model")
    for i in range(TRANSFERS_CAP + 1):
        reg.record_transfer(m.model_id, {
            "ts": f"2026-09-{(i % 28) + 1:02d}T00:00:00+00:00",
            "profile": f"p{i}", "baseline_id": None, "n": 1,
            "candidate_mean": i / 100.0, "baseline_mean": None,
            "improvement": None, "is_regression": None})
    row = reg.get_model(m.model_id)
    assert row is not None
    assert len(row.transfers) == TRANSFERS_CAP
    # newest first: entry p20 on top, the very first (p0) evicted
    assert row.transfers[0]["profile"] == f"p{TRANSFERS_CAP}"
    assert all(t["profile"] != "p0" for t in row.transfers)
    with pytest.raises(Exception):
        reg.record_transfer(m.model_id, {"no_profile": True})
    reg.close()


def test_transfers_json_tolerant_on_read(home__test_transfer):
    reg = ModelRegistry(home__test_transfer / "buffer" / "toaster_loop.db")
    m = reg.register_model("Base/Model")
    reg.record_transfer(m.model_id, {"profile": "a", "n": 1})
    # corrupt the blob behind the API's back — hydration must never crash
    conn = sqlite3.connect(str(reg.db_path))
    conn.execute("UPDATE models SET transfers_json = '{not json' "
                 "WHERE model_id = ?", (m.model_id,))
    conn.commit()
    conn.close()
    row = reg.get_model(m.model_id)
    assert row is not None and row.transfers == []
    # and writing again works from the honest empty state
    reg.record_transfer(m.model_id, {"profile": "b", "n": 2})
    row2 = reg.get_model(m.model_id)
    assert row2 is not None
    assert [t["profile"] for t in row2.transfers] == ["b"]
    reg.close()


# -------------------------------------------------------------- migration

def test_transfers_migration_on_old_db(home__test_transfer):
    db = home__test_transfer / "buffer" / "toaster_loop.db"
    reg = ModelRegistry(db)
    reg.register_model("B/M")
    reg.close()

    # Simulate a pre-#37 DB: drop the transfers column entirely.
    conn = sqlite3.connect(str(db))
    conn.execute("ALTER TABLE models DROP COLUMN transfers_json")
    conn.commit()
    conn.close()

    reg2 = ModelRegistry(db)  # _migrate_transfers_column re-adds it
    try:
        m = reg2.get_model("v1")
        assert m is not None and m.transfers == []  # tolerant read pre-write
        reg2.record_transfer("v1", {"profile": "p", "n": 3})
        row = reg2.get_model("v1")
        assert row is not None and row.transfers[0]["profile"] == "p"
    finally:
        reg2.close()


# ------------------------------------------------------------------ CLI smoke

def test_cli_evaluate_transfer_smoke(home__test_transfer, monkeypatch, capsys):
    """`toaster_loop evaluate transfer v2 --profile writer` queues the reused
    shadow_eval job with store_as=transfer and reports the stored entry."""
    import toaster_loop.cli as cli_mod
    import toaster_loop.evaluation.shadow as sh
    from toaster_loop.training import scheduler as sched_mod

    monkeypatch.setattr(sched_mod, "gpu_free_mib", lambda: 10**7)

    reg = ModelRegistry(home__test_transfer / "buffer" / "toaster_loop.db")
    m = reg.register_model("Base/Model")
    reg.close()

    seen = {}

    def fake_run(registry, store, *, candidate_id, profile=None, n=8,
                 max_regression_pct=5.0, judge_model=None, home=None,
                 endpoint_for=None, generate_fn=None, judge_client_for=None,
                 actor="worker", store_as=None, baseline_id=None):
        seen.update(store_as=store_as, profile=profile, n=n,
                    baseline_id=baseline_id)
        registry.record_transfer(candidate_id, {
            "ts": "2026-09-24T00:00:00+00:00", "profile": profile,
            "baseline_id": baseline_id, "n": n, "candidate_mean": 0.7,
            "baseline_mean": None, "improvement": None,
            "is_regression": None}, actor="cli")
        return {"skipped": False, "profile": profile, "baseline_id": baseline_id,
                "candidate_mean": 0.7, "improvement": None}

    monkeypatch.setattr(sh, "run_shadow_eval", fake_run)

    rc = cli_mod.main(["evaluate", "transfer", m.model_id, "--profile",
                       "writer", "--n", "4"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert seen["store_as"] == "transfer" and seen["profile"] == "writer"
    assert seen["n"] == 4
    assert "transfer probe on 'writer'" in out
    assert "candidate-only" in out

    reg = ModelRegistry(home__test_transfer / "buffer" / "toaster_loop.db")
    row = reg.get_model(m.model_id)
    assert row is not None
    assert row.transfers and row.transfers[0]["profile"] == "writer"
    reg.close()


def test_cli_evaluate_transfer_unknown_model(home__test_transfer, capsys):
    from toaster_loop.cli import main as cli_main

    rc = cli_main(["evaluate", "transfer", "v9", "--profile", "writer"])
    assert rc == 1
    assert "no model v9" in capsys.readouterr().err


def test_cli_evaluate_transfer_unknown_baseline(home__test_transfer, capsys):
    from toaster_loop.cli import main as cli_main

    reg = ModelRegistry(home__test_transfer / "buffer" / "toaster_loop.db")
    m = reg.register_model("Base/Model")
    reg.close()
    rc = cli_main(["evaluate", "transfer", m.model_id, "--profile", "writer",
                   "--baseline", "v9"])
    assert rc == 1
    assert "no baseline model v9" in capsys.readouterr().err

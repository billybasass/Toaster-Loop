"""WP6 tests — MCP control plane (`sable mcp`).

Hermetic: the server is driven IN-PROCESS by feeding JSON-RPC lines to
``McpServer.feed_line`` and parsing the returned response lines. No sockets,
no subprocesses for the protocol tests; mutating executors are exercised
against a fake DashboardApp so nothing spawns and no state is written.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sable.cli import build_parser  # noqa: E402
from sable.mcp.confirm import ConfirmStore, args_fingerprint  # noqa: E402
from sable.mcp.schema import validate_args  # noqa: E402
from sable.mcp.server import (PROTOCOL_VERSION, McpServer,  # noqa: E402
                                serve_stdio)

MCP_SRC = Path(__file__).resolve().parents[1] / "sable" / "mcp"


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
    monkeypatch.setenv("SABLE_HOME", str(h))
    return h


# -------------------------------------------------------------- handshake

def test_initialize_handshake(home):
    srv, _ = make_server(home)
    resp = init(srv)
    result = resp["result"]
    assert result["protocolVersion"] == PROTOCOL_VERSION
    assert result["serverInfo"]["name"] == "sable"
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
    from sable.dashboard.app import DashboardApp

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


# ---------------------------------------------------------- no-shell rules

def test_no_shell_execution_anywhere_in_mcp():
    """No subprocess/shell/os.system anywhere in sable/mcp/**."""
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
    """The MCP package imports only stdlib + sable (no MCP SDK)."""
    import ast

    allowed_roots = set(sys.stdlib_module_names) | {"sable", "yaml", "httpx"}
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
    assert "sable mcp" in capsys.readouterr().err

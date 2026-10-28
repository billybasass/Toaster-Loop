"""MCP server: JSON-RPC 2.0 line-delimited over stdio (WP6).

Transport choice — **stdio, no sockets**: MCP's canonical local transport.
The MCP client spawns ``sable mcp`` and speaks newline-delimited JSON on
stdin/stdout. Nothing binds a port, so the project's loopback-only rule is
satisfied trivially (there is no listener to protect), zero cloud, zero API
keys. A single-threaded synchronous loop is used deliberately: MCP stdio
clients pipeline requests but one-in-flight-at-a-time semantics are fully
spec-compliant, keep the confirm store trivially safe, and match the rest of
sable's defensive posture (documented in README "sable mcp").

Protocol surface (MCP basics): ``initialize`` / ``notifications/initialized``
/ ``ping`` / ``tools/list`` / ``tools/call``. Responses carry
``protocolVersion``, ``serverInfo`` and ``capabilities.tools``.

STDLIB-ONLY (D-12): json/sys/io/re only — no MCP SDK. The JSON-RPC framing
is re-implemented inline; the control-plane-over-tools pattern is inspired
by Strata's MCP design.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Optional, TextIO

from .confirm import DEFAULT_TTL_S, ConfirmStore
from .schema import validate_args
from .tools import Tool, ToolContext, ToolError, build_tools

#: MCP protocol version this server implements (basics subset).
PROTOCOL_VERSION = "2024-11-05"

SERVER_INFO = {"name": "sable", "version": "0.1.0"}

# JSON-RPC 2.0 standard error codes.
_PARSE_ERROR = -32700
_INVALID_REQUEST = -32600
_METHOD_NOT_FOUND = -32601
_INVALID_PARAMS = -32602
_INTERNAL_ERROR = -32603


class McpServer:
    """Line-delimited JSON-RPC 2.0 server exposing sable tools.

    ``feed_line`` processes exactly one request line and returns the response
    line (or None for notifications) — this is what tests drive in-process,
    without any pipes or child processes. ``serve_stdio`` wraps it over real
    stdin/stdout.
    """

    def __init__(self, config_path: Optional[str] = None, *,
                 app: Any = None, ttl_s: float = DEFAULT_TTL_S,
                 clock: Any = None) -> None:
        self.config_path = config_path
        self.tools: List[Tool] = build_tools()
        self._by_name = {t.name: t for t in self.tools}
        self.confirm = ConfirmStore(ttl_s=ttl_s, clock=clock)
        self._initialized = False
        # DashboardApp is heavyweight-ish (threads on first use only); lazy
        # construction means `sable mcp --help`-style usage and tools/list
        # never touch the home directory. Tests inject a prepared app.
        self._app = app
        self._ctx: Optional[ToolContext] = None

    # ------------------------------------------------------------- context

    def _context(self) -> ToolContext:
        if self._ctx is None:
            if self._app is None:
                from ..core.config import load_config
                from ..dashboard.app import DashboardApp

                cfg = load_config(self.config_path)
                self._app = DashboardApp(cfg)
            self._ctx = ToolContext(app=self._app,
                                    config_path=self.config_path)
        return self._ctx

    # ---------------------------------------------------------- dispatch

    def feed_line(self, line: str) -> Optional[str]:
        """Handle one newline-delimited JSON-RPC message; return response.

        Returns the serialized response line, or None for notifications
        (messages without an ``id``). Never raises on malformed input —
        parse and protocol errors become JSON-RPC error responses.
        """
        try:
            msg = json.loads(line)
        except (ValueError, RecursionError):
            return _serialize(_error(None, _PARSE_ERROR,
                                     "Parse error: invalid JSON"))
        if not isinstance(msg, dict):
            return _serialize(_error(None, _INVALID_REQUEST,
                                     "Request must be a JSON object"))
        mid = msg.get("id")
        # Responses without an id are notifications: no reply per spec.
        is_notification = "id" not in msg
        method = msg.get("method")

        if not isinstance(method, str):
            if is_notification:
                return None
            return _serialize(_error(mid, _INVALID_REQUEST,
                                     "Missing method name"))

        try:
            result = self._dispatch(method, msg)
        except _RpcError as e:
            if is_notification:
                return None
            return _serialize(_error(mid, e.code, e.message))
        except ToolError as e:
            # tools/call reaches here via _call_tool only for pre-execution
            # refusals that opted into ToolError; execution errors go into a
            # result envelope instead (isError). Defensive net.
            if is_notification:
                return None
            return _serialize(_error(mid, _INTERNAL_ERROR, str(e)))
        except Exception as e:  # noqa: BLE001 - never crash the loop
            if is_notification:
                return None
            return _serialize(_error(mid, _INTERNAL_ERROR,
                                     f"{type(e).__name__}: {e}"))
        if is_notification or result is _NO_REPLY:
            return None
        return _serialize({"jsonrpc": "2.0", "id": mid, "result": result})

    def _dispatch(self, method: str,
                  msg: Dict[str, Any]) -> Any:
        if method == "initialize":
            return self._initialize(msg)
        if method in ("notifications/initialized", "initialized"):
            return _NO_REPLY
        if method == "ping":
            return {}
        # MCP requires initialization first; answer protocol errors cleanly.
        if not self._initialized and method != "initialize":
            raise _RpcError(_INVALID_REQUEST,
                            "server not initialized — call initialize first")
        if method == "tools/list":
            return {"tools": [self._tool_spec(t)
                              for t in self.tools
                              if self._tool_visible(t)]}
        if method == "tools/call":
            return self._call_tool(msg.get("params"))
        raise _RpcError(_METHOD_NOT_FOUND, f"unknown method: {method}")

    # -------------------------------------------------------- family gating

    def _plans_enabled(self) -> bool:
        """Config gate for the WP12 plan tool family. Cheap + never raises:
        reads the live app config when one exists; loads config from the
        server's own path otherwise. Any read failure => disabled (the gate
        fails CLOSED, matching "absent from tools/list AND refused if called
        anyway")."""
        try:
            from ..core.plans import plans_config

            cfg = None
            if self._app is not None:
                cfg = getattr(self._app, "cfg", None)
            elif self._ctx is not None:
                cfg = getattr(self._ctx.app, "cfg", None)
            if cfg is None:
                from ..core.config import load_config

                cfg = load_config(self.config_path)
            return bool(plans_config(cfg).get("enabled", False))
        except Exception:  # noqa: BLE001 - gate fails closed
            return False

    def _tool_visible(self, t: Tool) -> bool:
        """WP12: family="plans" tools exist ONLY while plans.enabled (spec:
        absent from tools/list AND refused if called anyway)."""
        if t.family == "plans":
            return self._plans_enabled()
        return True

    # ----------------------------------------------------------- handlers

    def _initialize(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        params = msg.get("params") or {}
        if not isinstance(params, dict):
            raise _RpcError(_INVALID_PARAMS, "initialize params must be an object")
        self._initialized = True
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": SERVER_INFO,
        }

    def _tool_spec(self, t: Tool) -> Dict[str, Any]:
        return {
            "name": t.name,
            "description": t.description,
            "inputSchema": t.input_schema,
        }

    def _call_tool(self, params: Any) -> Dict[str, Any]:
        if not isinstance(params, dict):
            raise _RpcError(_INVALID_PARAMS,
                            "tools/call params must be an object")
        name = params.get("name")
        if not isinstance(name, str) or name not in self._by_name:
            known = ", ".join(sorted(self._by_name))
            raise _RpcError(_METHOD_NOT_FOUND,
                            f"unknown tool {name!r}; known tools: {known}")
        tool = self._by_name[name]
        args = params.get("arguments")
        if args is None:
            args = {}

        # WP12 family gate: a plans tool called while plans.enabled is off is
        # as good as nonexistent (it never appeared in tools/list). Refuse as
        # unknown — same shape as any other unknown tool, no plan machinery
        # runs and no pending-plan state is touched.
        if not self._tool_visible(tool):
            raise _RpcError(_METHOD_NOT_FOUND,
                            f"unknown tool {name!r} (plans are disabled; "
                            "enable config plans.enabled)")

        # ---- strict schema validation (before any plan or execution) ----
        err = validate_args(tool.input_schema, args)
        if err is not None:
            raise _RpcError(_INVALID_PARAMS, err)

        # ---- read-only tools execute immediately, no token needed ------
        if not tool.mutating:
            return self._execute(tool, args)

        # ---- mutating: plan-then-confirm --------------------------------
        token = args.get("confirm_token")
        exec_args = {k: v for k, v in args.items() if k != "confirm_token"}
        if token is None:
            plan = self._plan(tool, exec_args)
            new_token = self.confirm.issue(tool.name, exec_args)
            return _result_envelope({
                "status": "plan",
                "plan": plan,
                "confirm_token": new_token,
                "next": (f"Call {tool.name} again with the same arguments "
                         "plus \"confirm_token\": \"<token>\" to execute."),
            }, isError=False)
        bad = self.confirm.consume(str(token), tool.name, exec_args)
        if bad is not None:
            # Refusal travels as a tool error, not a protocol crash.
            return _result_envelope({"status": "refused",
                                     "reason": bad}, isError=True)
        return self._execute(tool, exec_args)

    def _plan(self, tool: Tool, args: Dict[str, Any]) -> Dict[str, Any]:
        target = None
        if tool.target_arg:
            target = args.get(tool.target_arg)
        return {
            "action": tool.name,
            "target": target,
            "exact_effect": tool.effect,
            "risks": list(tool.risks),
            "params": args,
        }

    def _execute(self, tool: Tool, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            out = tool.executor(self._context(), args)
        except ToolError as e:
            return _result_envelope({"status": "error",
                                     "reason": str(e)}, isError=True)
        except Exception as e:  # noqa: BLE001 - tool errors, not crashes
            return _result_envelope(
                {"status": "error",
                 "reason": f"{type(e).__name__}: {e}"}, isError=True)
        return _result_envelope({"status": "ok", "result": out},
                                isError=False)


# ------------------------------------------------------------ envelopes

_NO_REPLY = object()


def _result_envelope(payload: Any, *, isError: bool) -> Dict[str, Any]:
    """MCP tools/call result: JSON text content + isError flag."""
    return {
        "content": [{
            "type": "text",
            "text": json.dumps(payload, ensure_ascii=False, default=str),
        }],
        "isError": isError,
    }


class _RpcError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _error(mid: Any, code: int, message: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": mid,
            "error": {"code": code, "message": message}}


def _serialize(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


# ------------------------------------------------------------------ loop

def serve_stdio(config_path: Optional[str] = None, *,
                stdin: Optional[TextIO] = None,
                stdout: Optional[TextIO] = None) -> int:
    """Run the MCP server over stdio until EOF. Returns process exit code.

    stdout is reserved for JSON-RPC; diagnostics go to stderr only (MCP
    requirement). Exits 0 on clean stdin EOF, 130 on Ctrl-C — matching the
    project's exit-code taxonomy (core/exitcodes.py).
    """
    fin = stdin if stdin is not None else sys.stdin
    fout = stdout if stdout is not None else sys.stdout
    server = McpServer(config_path)
    print("sable mcp: stdio JSON-RPC server ready "
          "(loopback-only by construction: no socket)", file=sys.stderr)
    try:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            response = server.feed_line(line)
            if response is not None:
                fout.write(response + "\n")
                fout.flush()
    except KeyboardInterrupt:
        return 130
    return 0

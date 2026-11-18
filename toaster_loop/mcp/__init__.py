"""toaster_loop.mcp — optional MCP control plane over stdio (WP6).

Exposes toaster_loop's existing CLI/dashboard capabilities as MCP tools using
JSON-RPC 2.0 line-delimited over **stdio** — the canonical MCP transport,
chosen deliberately:

  * No socket is ever opened by this module, so the project's loopback-only
    networking rule is satisfied trivially (there is no listener at all).
  * The MCP client owns the process lifecycle (spawns ``toaster-loop mcp``,
    writes requests to stdin, reads responses from stdout).
  * Single-threaded request/response loop: one message in, one message out.

STDLIB-ONLY (D-12): this package imports only the Python standard library
plus toaster_loop's own modules — no MCP SDK, no third-party framework. The
JSON-RPC framing is re-implemented inline here; the pattern (a control plane
exposing the app's own operations as typed tools) is inspired by Strata's
MCP control-plane design.

Tools never invent authority: every tool calls the same in-process code the
dashboard/CLI call (``DashboardApp.handle`` and the CLI helper functions).
Mutating tools use a plan-then-confirm flow (see confirm.py): the first call
returns a PLAN plus an opaque single-use token bound to (tool, args-hash);
only a second call presenting that token executes.
"""

from __future__ import annotations

from .server import PROTOCOL_VERSION, McpServer, serve_stdio

__all__ = ["McpServer", "PROTOCOL_VERSION", "serve_stdio"]

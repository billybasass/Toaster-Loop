"""Dashboard HTTP server — stdlib only, bound to 127.0.0.1 (loopback) always.

Serves the static UI from ``dashboard/static/`` and delegates /api/* to
DashboardApp. No external assets: the browser fetches nothing beyond this
server, so the dashboard works fully offline.

Config (`dashboard:` YAML section, all optional):
    port: 0            # 0 => dynamic free port on loopback
    token: ""          # optional shared secret (also SABLE_DASHBOARD_TOKEN)

Phase 5a: PUT/DELETE verbs, SSE streaming responses (chunked, no
Content-Length), optional constant-time token auth on /api/* (header
``X-Sable-Token`` or ``?token=``), CSP + frame-blocking headers.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

from ..core.config import SableConfig
from .app import DashboardApp, api_max_body_bytes

logger = logging.getLogger("sable.dashboard")

STATIC_DIR = Path(__file__).resolve().parent / "static"

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".json": "application/json",
}

# Loopback-only constraint: this is the single bind address. Not configurable.
BIND_HOST = "127.0.0.1"

#: Admission control for SSE streams (chat runs + event bus). ThreadingHTTPServer
#: spawns one thread per connection with no cap; a reconnecting-EventSource
#: storm (many tabs, wedged clients) would grow threads without bound. Every
#: SSE response passes through _stream_sse, so one bounded semaphore there is
#: the ceiling: over-cap streams get 503 + Retry-After and the client's natural
#: reconnect picks them up when a slot frees. Loopback single-user defaults to
#: plenty of headroom; dashboard.max_sse in config overrides.
DEFAULT_MAX_SSE = 16


def _max_sse_connections(cfg: SableConfig) -> int:
    sec = getattr(cfg, "dashboard", None) or {}
    v = sec.get("max_sse") if isinstance(sec, dict) else None
    return int(v) if isinstance(v, int) and v > 0 else DEFAULT_MAX_SSE

#: Hostnames a browser may address us by (Host header) or come from
#: (Origin/Referer). Anything else means either DNS rebinding (a page at an
#: attacker domain resolving to 127.0.0.1 and hitting this loopback server)
#: or a cross-site request — both refused on /api/*.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})


def _host_is_loopback(value: str) -> bool:
    """True when a Host/Origin URL value names a loopback host (port ignored)."""
    v = (value or "").strip()
    if not v:
        return False
    # Origin/Referer values are URLs; Host values are host[:port].
    if "://" in v:
        rest = v.split("://", 1)[1]
        v = rest.split("/", 1)[0]
    else:
        v = v.split("/", 1)[0]
    if v.startswith("["):          # IPv6 [::1]:port
        host = v.split("]", 1)[0].lstrip("[")
    else:
        host = v.split(":", 1)[0]
    return host in _LOOPBACK_HOSTS


def _free_port(host: str = BIND_HOST) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def dashboard_port(cfg: SableConfig, override: Optional[int] = None) -> int:
    if override is not None:
        return int(override)
    env = os.environ.get("SABLE_DASHBOARD_PORT")
    if env and env.strip().isdigit():
        return int(env)
    sec = cfg.dashboard or {}
    p = sec.get("port")
    return int(p) if isinstance(p, int) and p >= 0 else 0


def make_server(
    cfg: SableConfig, port: Optional[int] = None, home: Optional[Path] = None,
    token: Optional[str] = None,
) -> ThreadingHTTPServer:
    """Build (but do not serve) the loopback HTTP server.

    ``token`` is the CLI override: a non-empty secret requires it, the empty
    string explicitly DISABLES auth (``--no-token``), and None falls through
    to config/env/the generated-and-persisted default token.
    """
    app = DashboardApp(cfg, home=home)
    if token is not None:  # CLI override wins over config/env ("" disables)
        app.cfg.dashboard = dict(app.cfg.dashboard or {})
        app.cfg.dashboard["token"] = token
    p = dashboard_port(cfg, override=port) or _free_port()
    # Shared across all handler threads (closure over make_server scope).
    sse_slots = threading.Semaphore(_max_sse_connections(app.cfg))

    class Handler(BaseHTTPRequestHandler):
        server_version = "sable-dashboard"
        sys_version = ""  # do not leak the Python version
        protocol_version = "HTTP/1.1"  # needed for chunked SSE responses

        # -- helpers ------------------------------------------------------
        def _security_headers(self) -> None:
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self'",
            )

        def _send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            # Loopback-only UI; keep it locked down.
            self._security_headers()
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _send_json(self, status: int, obj: Any) -> None:
            self._send(status, json.dumps(obj, ensure_ascii=False, default=str).encode(),
                       "application/json")

        def _stream_sse(self, it) -> None:
            """Write an SSE response from a str iterator; chunked transfer.

            Admission control: each stream occupies one slot of the shared
            semaphore for its whole lifetime (a wedged client holds its slot —
            that is exactly what the cap protects against). Over-cap requests
            answer 503 + Retry-After; EventSource reconnects on its own.
            """
            if not sse_slots.acquire(blocking=False):
                try:
                    self.send_response(503)
                    self.send_header("Retry-After", "2")
                    self.send_header("Content-Type", "application/json")
                    body = json.dumps({"error": "too many event streams",
                                       "code": "sse_capacity"}).encode()
                    self.send_header("Content-Length", str(len(body)))
                    self._security_headers()
                    self.end_headers()
                    self.wfile.write(body)
                except OSError:
                    pass
                return
            try:
                self.send_response(200)
                self.send_header("Content-Type",
                                 "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                # We hand-write chunk framing below; declare it so clients decode.
                self.send_header("Transfer-Encoding", "chunked")
                self._security_headers()
                self.end_headers()
                try:
                    for chunk in it:
                        if isinstance(chunk, str):
                            chunk = chunk.encode("utf-8")
                        self.wfile.write(b"%x\r\n" % len(chunk) + chunk
                                         + b"\r\n")
                        self.wfile.flush()
                    self.wfile.write(b"0\r\n\r\n")  # terminating chunk
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass  # client went away; iterators clean up in their finally
            finally:
                sse_slots.release()
            self.close_connection = True

        def _static(self, path: str) -> None:
            rel = path.lstrip("/") or "index.html"
            if rel in ("", "index.html"):
                rel = "index.html"
            candidate = (STATIC_DIR / rel).resolve()
            # Path traversal guard: must stay inside static/ (is_relative_to
            # compares path components, so sibling-prefix collisions like
            # /tmp/static-evil can never slip through a string prefix check).
            if not candidate.is_relative_to(STATIC_DIR.resolve()) \
                    or not candidate.is_file():
                self._send_json(404, {"error": "not found"})
                return
            ctype = _CONTENT_TYPES.get(candidate.suffix, "application/octet-stream")
            self._send(200, candidate.read_bytes(), ctype)

        # -- verbs ----------------------------------------------------------
        def do_GET(self) -> None:  # noqa: N802
            self._handle("GET")

        def do_HEAD(self) -> None:  # noqa: N802
            self._handle("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._handle("POST")

        def do_PUT(self) -> None:  # noqa: N802
            self._handle("PUT")

        def do_DELETE(self) -> None:  # noqa: N802
            self._handle("DELETE")

        def _authorized(self, query: Dict[str, str], method: str) -> bool:
            if app.check_token(None):  # no token configured
                return True
            provided = self.headers.get("X-Sable-Token")
            if not provided and method == "GET":
                # Query-string tokens are a convenience for EventSource (which
                # cannot set headers) — GET only, so a mutation can never smuggle
                # the token through a URL that leaks into logs/history.
                provided = query.get("token")
            return app.check_token(provided)

        def _cross_site_blocked(self, method: str) -> bool:
            """CSRF / DNS-rebinding guard for /api/*.

            A same-origin browser request always sends either Origin (fetch,
            XHR, EventSource) or Referer (same-origin default policy); curl and
            our own tests send neither, so they stay allowed. When either IS
            present it must name a loopback origin — anything else is a page on
            another site driving this local API. The Host header is checked too:
            a rebinding attack reaches 127.0.0.1 under an attacker hostname.
            """
            host = self.headers.get("Host") or ""
            if not _host_is_loopback(host):
                return True
            origin = self.headers.get("Origin")
            referer = self.headers.get("Referer")
            for value in (origin, referer):
                if value and not _host_is_loopback(value):
                    return True
            return False

        def _handle(self, method: str) -> None:
            parsed = urlparse(self.path)
            path = parsed.path
            # WP9c: Prometheus scrape endpoint — GET /metrics is an alias of
            # GET /api/metrics on this SAME loopback server (no new listener,
            # no new port). Standard scrape configs default to /metrics and
            # cannot always set headers, so ?token= works like every other
            # GET; when a dashboard token is configured this route enforces
            # it exactly like /api/* (no unauthenticated backdoor), including
            # the DNS-rebinding/CSRF gate.
            if path == "/metrics" and method in ("GET", "HEAD"):
                query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
                if self._cross_site_blocked(method):
                    self._send_json(403, {"error": {
                        "code": "cross_site_blocked",
                        "message": "Origin/Referer/Host must be loopback"}})
                    return
                if not self._authorized(query, method):
                    self._send_json(401, {"error": {
                        "code": "unauthorized",
                        "message": "missing or invalid X-Sable-Token"}})
                    return
                resp = app.handle("GET", "/api/metrics", query=query, body={})
                raw = getattr(resp, "raw", None) or b""
                self._send(resp.status, raw,
                           str(getattr(resp, "content_type", None)
                               or "text/plain; version=0.0.4"))
                return
            if path.startswith("/api/"):
                query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
                if self._cross_site_blocked(method):
                    self._send_json(403, {"error": {
                        "code": "cross_site_blocked",
                        "message": "Origin/Referer/Host must be loopback"}})
                    return
                if not self._authorized(query, method):
                    self._send_json(401, {"error": {
                        "code": "unauthorized",
                        "message": "missing or invalid X-Sable-Token"}})
                    return
                body: Dict[str, Any] = {}
                if method in ("POST", "PUT", "DELETE"):
                    length = int(self.headers.get("Content-Length") or 0)
                    cap = api_max_body_bytes(path)
                    if length > cap:
                        self._send_json(413, {"error": {
                            "code": "payload_too_large",
                            "message": f"body too large (cap {cap} bytes)"}})
                        return
                    raw = self.rfile.read(length) if length else b""
                    if raw:
                        try:
                            body = json.loads(raw)
                        except json.JSONDecodeError:
                            self._send_json(400, {"error": {
                                "code": "bad_request",
                                "message": "invalid JSON body"}})
                            return
                        if not isinstance(body, dict):
                            self._send_json(400, {"error": {
                                "code": "bad_request",
                                "message": "body must be a JSON object"}})
                            return
                resp = app.handle(method, path, query=query, body=body)
                if resp.stream is not None:
                    self._stream_sse(resp.stream)
                    return
                raw = getattr(resp, "raw", None)
                if raw is not None:
                    self._send(resp.status, raw,
                               str(getattr(resp, "content_type", None)
                                   or "application/octet-stream"))
                    return
                self._send(resp.status, json.dumps(resp.body, ensure_ascii=False,
                                                   default=str).encode(),
                           "application/json")
                return
            if method == "GET":
                self._static(path)
                return
            self._send_json(405, {"error": {"code": "method_not_allowed",
                                            "message": "method not allowed"}})

        # Quiet, secret-free access log (no query strings, no bodies).
        def log_message(self, format: str, *args: Any) -> None:
            logger.debug("%s %s", self.address_string(), format % args)

    httpd = ThreadingHTTPServer((BIND_HOST, p), Handler)
    httpd.daemon_threads = True
    httpd.sable_app = app  # type: ignore[attr-defined]
    return httpd


def serve_dashboard(
    cfg: SableConfig, port: Optional[int] = None, home: Optional[Path] = None,
    block: bool = True, token: Optional[str] = None,
) -> ThreadingHTTPServer:
    """Start the dashboard. Returns the running server (blocks when block=True)."""
    httpd = make_server(cfg, port=port, home=home, token=token)
    url = f"http://{BIND_HOST}:{httpd.server_address[1]}/"
    tok = httpd.sable_app.token  # type: ignore[attr-defined]
    # Queue dispatcher: reap stale workers NOW (a phantom `running` row from a
    # crash must not hold submits at 409), then drain queued/gate-blocked jobs
    # on its tick. Contention with another dashboard's dispatcher over one DB
    # is fatal — two drivers with independent in-process state is the bug this
    # lock exists to prevent; abort boot with a message naming holder+path.
    from ..training.scheduler import DispatcherAlreadyRunning

    try:
        httpd.sable_app.startup_dispatcher()  # type: ignore[attr-defined]
    except DispatcherAlreadyRunning as e:
        httpd.server_close()
        raise SystemExit(f"sable dashboard refused to start: {e}")
    except Exception:
        logger.exception("queue dispatcher failed to start")
    # Phase 5: while indexing is enabled, refresh indexes in the background
    # (non-blocking; embedding server spawn happens lazily on first embed).
    try:
        httpd.sable_app.startup_indexing()  # type: ignore[attr-defined]
    except Exception:
        logger.exception("startup index refresh failed to schedule")
    # Phase 6: autostart-flagged server instances boot in the background.
    try:
        httpd.sable_app.startup_autostart()  # type: ignore[attr-defined]
    except Exception:
        logger.exception("server autostart failed to schedule")
    # WP4 engine watchdog: strictly OPT-IN (serve.watchdog_enabled) — with no
    # config changes start_watchdog returns None immediately and nothing here
    # runs. When enabled it watches the managed llama-server children for the
    # frozen signature (request in flight + zero CPU + zero IO) over /proc,
    # kills only ownership-verified pids, and publishes serve.watchdog events
    # on the dashboard bus.
    watchdog = None
    try:
        from ..serving.watchdog import start_watchdog

        watchdog = start_watchdog(
            httpd.sable_app.cfg,  # type: ignore[attr-defined]
            httpd.sable_app.home,  # type: ignore[attr-defined]
            emit=httpd.sable_app.bus.publish)  # type: ignore[attr-defined]
    except Exception:
        logger.exception("engine watchdog failed to start")
    # WP5 idle-unload loop: strictly OPT-IN (serve.idle_unload_seconds > 0) —
    # disabled, start_idle_loop returns False and no thread exists. When on it
    # gracefully unloads instances idle past the window (never mid-request),
    # publishes serve.coexist events, and registers the engine process-wide so
    # the agent runner's connection-failure hook can lazy-reload.
    coex = getattr(httpd.sable_app, "coexist", None)  # type: ignore[attr-defined]
    try:
        if coex is not None:
            from ..serving.coexist import set_active as _set_coexist_active

            _set_coexist_active(coex)
            coex.start_idle_loop()
    except Exception:
        logger.exception("coexistence idle loop failed to start")
    if block:
        print(f"sable dashboard: {url}")
        if tok:
            # Machine-readable line `sable launch` parses so the browser is
            # opened already authenticated (api.js moves ?token= into
            # sessionStorage and strips it from history).
            print(f"sable dashboard authed: {url}?token={tok}")
            print(f"token auth ON — open {url}?token={tok} or set the "
                  "X-Sable-Token header on /api/* calls")
        print("Loopback-only. Press Ctrl-C to stop.")
        stop = threading.Event()

        def _sig(signum, frame):  # graceful shutdown
            stop.set()

        old_int = signal.signal(signal.SIGINT, _sig)
        old_term = signal.signal(signal.SIGTERM, _sig)
        try:
            t = threading.Thread(target=httpd.serve_forever, daemon=True)
            t.start()
            stop.wait()
        finally:
            signal.signal(signal.SIGINT, old_int)
            signal.signal(signal.SIGTERM, old_term)
            httpd.shutdown()
            httpd.server_close()
            # Stop the queue dispatcher (releases its pid-lockfile).
            try:
                httpd.sable_app.stop_dispatcher()  # type: ignore[attr-defined]
            except Exception:
                pass
            # Run lifecycle step 3: give in-flight chat runs a chance to cancel
            # cleanly (recording their cancelled traces) BEFORE the serving
            # backends they talk to are stopped. Daemon runner threads would
            # otherwise die mid-turn with no trace at all.
            try:
                httpd.sable_app.shutdown_chat_runs()  # type: ignore[attr-defined]
            except Exception:
                pass
            # WP4: stop the watchdog before its watched children go away.
            try:
                if watchdog is not None:
                    watchdog.stop()
            except Exception:
                pass
            # WP5: stop the idle-unload loop before its instances do.
            try:
                if coex is not None:
                    coex.stop()
            except Exception:
                pass
            # Phase 6: stop every serving process this process owns —
            # multi-instance llama-server children AND the Phase 5 embedding
            # servers (the registry holds only processes we spawned; foreign
            # and attach-mode processes are never touched).
            try:
                from ..serving.supervisor import stop_all_servers
                stop_all_servers()
            except Exception:
                pass
            print("dashboard stopped.")
    return httpd

"""Shared API primitives: response envelope, errors, route compilation,
redaction helpers and the BackgroundTask slot used by the long-running
hardware/tuning endpoints.

Extracted verbatim from the former monolithic ``dashboard/app.py`` (pure move,
no behavior change). ``app.py`` re-exports everything here for backward
compatibility with existing imports and test monkeypatching.
"""

from __future__ import annotations

import hmac  # noqa: F401  (re-export convenience for callers/tests)
import json
import logging
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (TYPE_CHECKING, Any, Callable, Dict, Iterator, List,
                    Optional, Tuple)

logger = logging.getLogger("sable.dashboard")


class ApiContext:
    """Type-only contract for the DashboardApp members the route mixins use.

    The mixins are composed into ``DashboardApp`` (which owns the real
    attributes and cross-domain helpers); analyzed standalone, a type checker
    cannot see them. Everything here lives under ``TYPE_CHECKING`` so it adds
    nothing at runtime — the class is an empty base whose sole purpose is to
    make ``self.home``/``self.bus``/``self._store()``& friends resolve in each
    api/*.py file. Keep signatures loose: they exist for navigation, not
    checking; the real definitions are authoritative on DashboardApp.
    """

    if TYPE_CHECKING:
        # --- composed state (set in DashboardApp.__init__) -----------------
        bus: Any
        cfg: Any
        coexist: Any
        hardware: Any
        home: Any
        token: Any
        _bench_task: Any
        _calibrate_task: Any
        _chat_runs: Any
        _chat_runs_lock: Any
        _doctor_task: Any
        _engine_install_lock: Any
        _engine_install_tasks: Any
        _exec_lock: Any
        _metrics_lock: Any
        _runtime_task: Any

        # --- cross-domain helpers (DashboardApp or sibling mixins) ---------
        def handle(self, *args: Any, **kwargs: Any) -> Any: ...
        def shutdown_chat_runs(self, *args: Any, **kwargs: Any) -> Any: ...
        def submit_iterate(self, *args: Any, **kwargs: Any) -> Any: ...
        def _config_path(self, *args: Any, **kwargs: Any) -> Any: ...
        def _easy_cfg(self, *args: Any, **kwargs: Any) -> Any: ...
        def _instance_row(self, *args: Any, **kwargs: Any) -> Any: ...
        def _instance_state(self, *args: Any, **kwargs: Any) -> Any: ...
        def _job_observer(self, *args: Any, **kwargs: Any) -> Any: ...
        def _load_project(self, *args: Any, **kwargs: Any) -> Any: ...
        def _profiles(self, *args: Any, **kwargs: Any) -> Any: ...
        def _registry(self, *args: Any, **kwargs: Any) -> Any: ...
        def _resolve_profile(self, *args: Any, **kwargs: Any) -> Any: ...
        def _scheduler(self, *args: Any, **kwargs: Any) -> Any: ...
        def _serve_alive(self, *args: Any, **kwargs: Any) -> Any: ...
        def _stop_instance(self, *args: Any, **kwargs: Any) -> Any: ...
        def _store(self, *args: Any, **kwargs: Any) -> Any: ...
        def _training_config(self, *args: Any, **kwargs: Any) -> Any: ...
        def _validate_engine_ref(self, *args: Any, **kwargs: Any) -> Any: ...
        def _write_instance_state(self, *args: Any, **kwargs: Any) -> Any: ...


@dataclass
class ApiResponse:
    status: int
    body: Any = None
    # When set, the response is a chunked stream (SSE): server.py iterates
    # this iterator and writes each str chunk. `body` is ignored then.
    stream: Optional[Iterator[str]] = field(default=None, repr=False)
    # When set, the response body is raw bytes of this content type (e.g. the
    # hub image proxy). `body` is ignored then.
    raw: Optional[bytes] = field(default=None, repr=False)
    content_type: str = "application/json"

    def json(self) -> str:
        return json.dumps(self.body, ensure_ascii=False, default=str)


class HttpError(Exception):
    """Raise from handlers to produce the JSON error envelope."""

    def __init__(self, status: int, message: str, code: Optional[str] = None,
                 extra: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.code = code or _DEFAULT_ERROR_CODES.get(status, "error")
        # Machine-readable detail fields merged into the error envelope body
        # (e.g. the active run_id on a 409 run_in_flight).
        self.extra = dict(extra or {})


_DEFAULT_ERROR_CODES = {
    400: "bad_request",
    401: "unauthorized",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    413: "payload_too_large",
    500: "internal_error",
    503: "unavailable",
}

#: Default request-body cap for /api/* (JSON control traffic stays small).
API_MAX_BODY_BYTES = 1_000_000

#: Per-route body caps for endpoints that legitimately carry large payloads.
#: Chat image attachments: up to 4 images x core.images.MAX_IMAGE_BYTES,
#: base64-inflated (~4/3) plus JSON envelope headroom. The server.py handler
#: consults this table before reading the body so oversized uploads fail with
#: a clean 413 instead of being rejected by the generic cap (contract fix:
#: the old flat 1 MB limit made core/images' 8 MB per-image cap unreachable).
API_MAX_BODY_BYTES_BY_ROUTE = {
    "/api/chat/stream": 4 * 8_000_000 * 4 // 3 + 200_000,
}


def api_max_body_bytes(path: str) -> int:
    """Body cap for an /api/* path (per-route override, else default)."""
    return API_MAX_BODY_BYTES_BY_ROUTE.get(path, API_MAX_BODY_BYTES)


#: SSE heartbeat cadence (comment frames keep proxies/browsers warm).
SSE_HEARTBEAT_S = 15.0

#: Shared <home> subdirectory names used by more than one domain module.
ITERATE_RUNS_DIRNAME = "iterate_runs"
PROJECTS_DIRNAME = "projects"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------ route matcher

_PARAM_RE = re.compile(r"<([a-zA-Z_][a-zA-Z0-9_]*)>")


def _compile_route(pattern: str) -> "re.Pattern[str]":
    """`/api/traces/<trace_id>` -> anchored regex with named groups."""
    parts = _PARAM_RE.split(pattern)
    rx: List[str] = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            rx.append(re.escape(part))
        else:
            rx.append(f"(?P<{part}>[^/]+)")
    return re.compile("^" + "".join(rx) + "$")


Route = Tuple[str, "re.Pattern[str]", str]  # (method, regex, handler attr name)

_SECRET_KEY_RE = re.compile(r"(api_key|apikey|token|secret|password)", re.I)


def _redact(obj: Any) -> Any:
    """Recursively mask values under secret-looking keys."""
    if isinstance(obj, dict):
        return {
            k: ("***" if _SECRET_KEY_RE.search(str(k)) else _redact(v))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_redact(v) for v in obj]
    return obj


def _merge_patch(target: Dict[str, Any], patch: Dict[str, Any]) -> None:
    """Merge ``patch`` into ``target`` in place. A ``None`` patch value
    explicitly *deletes* the key (5e unset semantics — the plain dict.update
    merge could only overwrite, never clear)."""
    for k, v in patch.items():
        if v is None:
            target.pop(k, None)
        else:
            target[k] = v


class BackgroundTask:
    """One lock-guarded background run: record + daemon thread + bus events.

    Replaces the five hand-rolled "lock, check running, seed record, spawn
    thread, publish on finish" blocks (runtime setup, engine installs, bench,
    calibrate, doctor). The slot holds at most one live run; ``try_start``
    raises 409 while one is in flight. ``work(record, log)`` executes on the
    daemon thread and owns the final ``record["state"]`` (done/diverged/...);
    an uncaught exception becomes ``state=error`` with the message recorded.
    On finish the task publishes ``{state, error}`` plus (when ``final_lines``)
    the log tail and any ``done_extra(record)`` fields, so each endpoint's SSE
    contract stays byte-identical to the old hand-rolled blocks.

    Usage::

        self._bench = BackgroundTask(bus, "bench.run", log_cap=200,
                                     final_lines=False)
        rec = self._bench.try_start(_work, conflict_msg="...", initial={...},
                                    done_extra=lambda r: {"result": ...})
    """

    def __init__(self, bus: Any, event: str, *, log_cap: Optional[int] = 500,
                 final_lines: bool = True,
                 line_extra: Optional[Callable[[Dict[str, Any]],
                                               Dict[str, Any]]] = None):
        self.bus = bus
        self.event = event
        self.log_cap = log_cap
        #: Include the last-5 log lines in the *finish* event payload (the
        #: old blocks disagreed: runtime setup did, bench/calibrate didn't).
        self.final_lines = final_lines and log_cap is not None
        #: Extra fields merged into every running-line event (e.g. recipe id).
        self.line_extra = line_extra
        self.lock = threading.Lock()
        #: Current/last run record (None until first start). Pollable by the
        #: status endpoints; consumers treat missing as "idle".
        self.record: Optional[Dict[str, Any]] = None

    def running(self) -> bool:
        return (self.record or {}).get("state") == "running"

    def try_start(self, work: Callable[[Dict[str, Any], Callable[[str], None]],
                                        None],
                  *, conflict_msg: str,
                  initial: Optional[Dict[str, Any]] = None,
                  done_extra: Optional[Callable[[Dict[str, Any]],
                                                Dict[str, Any]]] = None,
                  ) -> Dict[str, Any]:
        """Claim the slot and run ``work`` on a daemon thread.

        Raises HttpError(409, code=already_running) when a run is live.
        Returns the freshly seeded record (state=running).
        """
        with self.lock:
            if self.running():
                raise HttpError(409, conflict_msg, code="already_running")
            record: Dict[str, Any] = {
                "state": "running", "started": _utcnow(), "finished": None,
                "error": None,
            }
            if self.log_cap is not None:
                record["log"] = []
            record.update(initial or {})
            self.record = record

        def _log(line: str) -> None:
            if self.log_cap is None:
                return
            record["log"].append(str(line))
            if len(record["log"]) > self.log_cap:
                del record["log"][: len(record["log"]) - self.log_cap]
            payload: Dict[str, Any] = {
                "state": "running", "line": str(line),
                "lines": record["log"][-5:],
            }
            if self.line_extra is not None:
                payload.update(self.line_extra(record))
            self.bus.publish(self.event, payload)

        def _run() -> None:
            try:
                work(record, _log)
            except Exception as e:  # a hung probe surfaces, never hangs us
                record["state"] = "error"
                record["error"] = f"{type(e).__name__}: {e}"
            finally:
                record["finished"] = _utcnow()
                payload: Dict[str, Any] = {"state": record["state"],
                                           "error": record["error"]}
                if self.final_lines:
                    payload["lines"] = record["log"][-5:]
                if done_extra is not None:
                    payload.update(done_extra(record))
                self.bus.publish(self.event, payload)

        threading.Thread(target=_run, daemon=True).start()
        return record

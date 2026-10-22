"""Dashboard control API — transport-agnostic JSON request handler.

Every mutation goes through the same core objects the CLI uses (TraceStore,
ModelRegistry, Scheduler, Curater) so the UI can never corrupt state beyond
what the CLI could; reads are direct DB queries. The HTTP layer in server.py
just adapts this to sockets, which keeps the whole API unit-testable by
calling ``DashboardApp.handle(method, path, query, body)`` directly.

Phase 5a: route-pattern matcher (``<param>`` segments, GET/POST/PUT/DELETE),
consistent error envelope ``{"error": {"code", "message"}}``, an in-process
EventBus (SSE at ``GET /api/events``), serve/deploy control, hardware
sampler, config read/write, chat proxy + agent stream, dataset row preview,
train/check and persisted iterate run records.

Endpoint catalog:

  GET    /api/status                        overview chips
  GET    /api/events                        SSE global bus (stream)
  GET    /api/serve/status                  serve detail + liveness
  POST   /api/serve/start                   {gguf|model_id|auto, context?, port?, force?}
  POST   /api/serve/stop                    stop owned server only
  POST   /api/serve/attach                  {url, api_key?}
  POST   /api/deploy                        {model_id} (alias: /api/serve/deploy)
  POST   /api/remote/probe                  {url, token?} -> probe + remember a
                                            LAN sable peer dashboard
  GET    /api/remote/peers                  remembered peers + live snapshots
  GET    /api/remote/peers/<id>/live        re-probe one remembered peer
  DELETE /api/remote/peers/<id>             forget a peer (never signals it)
  GET    /api/hardware                      sampler snapshot (2 s cache)
  GET    /api/config                        merged config (secrets redacted) + raw
  PUT    /api/config                        {yaml: str} | {section, patch} -> validated write
  POST   /api/proxy/chat                    raw passthrough (SSE when stream=true)
   POST   /api/chat/stream                   agent run -> SSE event protocol (stream)
   POST   /api/chat/cancel                   {run_id}
   GET    /api/chat/runs                     in-flight runs (adopt-orphans)  GET    /api/chat/threads                  thread summaries
  GET    /api/chat/threads/<id>             full thread JSON
  PUT    /api/chat/threads/<id>             upsert thread JSON
  DELETE /api/chat/threads/<id>             delete thread
  GET    /api/traces?limit&status&since&q=  index rows (q = text search)
  GET    /api/traces/stats?days=30          daily trend series
  GET    /api/traces/<id>                   detail + feedback + transcript
                                            (+ pass 4 #35 `timeline` segments)
  POST   /api/feedback                      {trace_id|last, score?, note?}
  POST   /api/traces/<id>/feedback          same body, id from path
  GET    /api/datasets                      manifests
  POST   /api/curate                        queue curation job -> 202 {job_id}
                                            (kind="curate", runs in-process;
                                            follow via job.state events /
                                            GET /api/jobs/<id>; result JSON
                                            line lands in the job log)
  POST   /api/datasets/export               {dataset_id, to, force?} -> guarded
                                            by home containment (409
                                            outside_home unless force=true)
  GET    /api/datasets/<id>                 manifest
  GET    /api/datasets/<id>/rows            ?file=sft|dpo&offset=&limit=
  GET    /api/models                        rows + lineage edges + active
  GET    /api/model_catalog                 models discovered under the base
                                            model folder (?dir= override)
  GET    /api/hub/search?q=                 HF Hub text search (popular first)
                                            ?gguf=1&capability=&sort= filters
  GET    /api/hub/feed        latest GGUF models rail (discover page)
  GET    /api/hub/detail?repo=              card metadata + variants + README
  GET    /api/hub/variants?repo=            GGUF quant variants for a repo
  GET    /api/hub/installed                 downloaded repos + download jobs
  POST   /api/hub/download                  {repo, variant?} -> 202 {job_id}
  GET    /api/hub/downloads                 all hub download jobs
  GET    /api/hub/downloads/<job_id>        one job (progress polling)
  POST   /api/hub/cancel                    {job_id}
  POST   /api/hub/delete                    {repo} -> remove downloaded repo
  POST   /api/models/promote                {model_id}
  POST   /api/models/rollback               {base_model?}
  GET    /api/models/<id>                   one model (+validation report)
  GET    /api/jobs?tail=N                   queue rows (+log tail)
  POST   /api/jobs/cancel                   {job_id}
  GET    /api/jobs/<id>                     one job + log window
  GET    /api/jobs/<id>/log?tail=N          log tail shorthand
  POST   /api/train                         {dataset, kind?, dry_run?} -> queued job
  GET    /api/train/check                   auto-train thresholds verdict
  POST   /api/iterate                       {name?, kind?, dataset?, dry_run?}
  GET    /api/iterate/runs                  persisted iterate run records
  GET    /api/profiles                      agent profiles + default + tool set
  PUT    /api/profiles/<name>               upsert one profile (validated)
  DELETE /api/profiles/<name>               delete a profile
  PUT    /api/profiles/default              {default: name|null} set default
  GET    /api/projects                      project records (<home>/projects/*.json)
  POST   /api/projects                      {name, workspace_dir, agents?} -> 201
  PUT    /api/projects/<project_id>         update name/agents/workspace_dir
  DELETE /api/projects/<project_id>         delete project (threads -> Unfiled)
  GET    /api/fs/tree?project=<id>&path=    read-only workspace listing, one
                                            lazy dir level (?hidden=1 shows
                                            dotfiles; names/metadata only)
  GET    /api/fs/read?project=<id>&path=    read-only file preview (text only,
                                            size-capped; same containment rules)

Agent profiles (Phase 5e) live in the config file: `profiles:` is a list of
{name, system_prompt, tools[], model_name?} and `default_profile` names one.
Dedicated endpoints wrap config read/merge/atomic-write so the UI never has to
patch raw YAML; PUT /api/config keeps its {section, patch} merge semantics
(`null` values in a patch now delete keys — explicit unset).

Long-running work (train/iterate) runs on daemon threads; state transitions
publish on the bus *and* remain pollable via /api/jobs and /api/iterate/runs.
A process-wide lock keeps submit+run serialized, and the registry's
single-flight rule is the hard backstop.

Chat streaming: agent runs execute on a worker thread that forwards the
built-in runtime's token stream live — message.delta / reasoning.delta
arrives per generated token (llama.cpp `reasoning_content` for thinking
models), plus tool.call / tool.result / context.usage as the loop progresses.
Runners without streaming support still work: the protocol is then
synthesized from the completed message list. The event contract
(UI-ARCHITECTURE §5) is unchanged; only the producer gained fidelity.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import re
import secrets
import signal
import threading
import time
import uuid
from dataclasses import dataclass, field, replace as _dc_replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple
from urllib.parse import unquote

import yaml

from ..core.config import SableConfig, load_config
from ..curate.curater import Curater, CurationConfig, curation_config_from_yaml
from ..registry.registry import ModelRegistry
from ..serving import instances as srv_inst
from ..serving.deploy import (
    clear_serve_state,
    deploy_model,
    read_serve_state,
    stop_owned_server,
    write_serve_state,
)
from ..serving.supervisor import ServeError, healthcheck
from ..store.db import TraceStore
from ..store.paths import resolve_sable_home
from ..training.easy import (
    OPTIMAL_CURATION,
    OPTIMAL_GOLDENS_MINE,
    OPTIMAL_TRAINING,
    OPTIMAL_VALIDATION,
    EasyTrainingConfig,
    acquire_loop_lock,
    evaluate_readiness,
    load_state as easy_load_state,
    mine_goldens_pass,
    optimal_config_patches,
    release_loop_lock,
    save_state as easy_save_state,
    validate_profile as easy_validate_profile,
)
from ..training.scheduler import TrainingConfig
from .events import EventBus, format_sse, sse_frame
from .hardware import HardwareSampler

logger = logging.getLogger("sable.dashboard")

DEFAULT_JOB_LOG_TAIL = 40
ROW_PREVIEW_CHARS = 800          # per-string truncation cap in row previews
SSE_HEARTBEAT_S = 15.0
ITERATE_RUNS_DIRNAME = "iterate_runs"
CHAT_DIRNAME = "chat"
PROJECTS_DIRNAME = "projects"

#: /api/fs/tree per-level entry cap (huge dirs must not serialize wholesale)
FS_TREE_MAX_ENTRIES = 2000
#: /api/fs/read preview cap (characters of text returned)
FS_READ_MAX_CHARS = 64_000

_SECRET_KEY_RE = re.compile(r"(api_key|apikey|token|secret|password)", re.I)


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


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------- models-panel catalog helpers (#15)

def _job_final_steps(log_path: Optional[str]) -> Optional[int]:
    """Final step count from a finished train job log (backlog item #15).

    Tolerant parse: prefers an explicit ``steps``/``total`` on the terminal
    sft_done/dpo_done line (batch-#4 protocol carries them on progress
    events); falls back to the largest observed ``step``. Any unreadable or
    malformed log yields None (piece omitted from the detail line).
    """
    if not log_path:
        return None
    try:
        lines = Path(log_path).read_text(errors="replace").splitlines()
    except OSError:
        return None
    best: Optional[int] = None
    for line in lines:
        if '"event"' not in line:
            continue
        try:
            rec = json.loads(line)
        except (ValueError, TypeError):
            continue
        ev = rec.get("event")
        if ev in ("sft_done", "dpo_done"):
            for k in ("steps", "total", "step"):
                v = rec.get(k)
                if isinstance(v, (int, float)) and v >= 0:
                    best = int(v)
                    break
        elif ev == "progress":
            v = rec.get("step")
            if isinstance(v, (int, float)) and (best is None or v > best):
                best = int(v)
    return best


def _job_duration_s(job: Dict[str, Any]) -> Optional[float]:
    """Finished-minus-started seconds for a job row; None when incomplete."""
    started, finished = job.get("started_at"), job.get("finished_at")
    if not started or not finished:
        return None
    try:
        t0 = datetime.fromisoformat(str(started))
        t1 = datetime.fromisoformat(str(finished))
    except ValueError:
        return None
    delta = (t1 - t0).total_seconds()
    return round(delta, 1) if delta >= 0 else None


# ------------------------------------------------------- trace timeline (#35)
# Segment-kinds for the traces-panel swimlane (backlog #35,
# concept adapted from PIASystem (MIT) app/core/rollout_visualizer.py). Fixed
# vocabulary; the frontend maps each kind to a design token.
TIMELINE_KINDS = ("llm", "tool", "retry", "scrub", "other")

#: Above this many raw segments, adjacent same-kind segments are merged so the
#: swimlane stays cheap to render (spec cap ~200).
TIMELINE_MAX_SEGMENTS = 200


def _timeline_ts(value: Any) -> Optional[float]:
    """Epoch seconds from an ISO string / epoch number; None when unusable."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        return v if v >= 0 else None
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except (TypeError, ValueError):
        return None


def _timeline_kind(m: Dict[str, Any]) -> Tuple[str, str]:
    """(kind, label) for one trace message; kind always in TIMELINE_KINDS."""
    role = str(m.get("role") or "")
    if role == "tool":
        name = str(m.get("name") or "tool")
        return ("retry", name) if name.startswith("retry_") else ("tool", name)
    if role == "assistant":
        return ("llm", "assistant")
    if role == "system":
        # Compression inserts exactly one rolling-summary system message whose
        # content starts with compression.SUMMARY_PREFIX (core/compression.py).
        try:
            from ..core.compression import SUMMARY_PREFIX
            if str(m.get("content") or "").startswith(SUMMARY_PREFIX):
                return ("scrub", "compression summary")
        except Exception:  # pragma: no cover - defensive; kind stays 'other'
            pass
        return ("other", "system")
    if role == "user":
        return ("other", "user")
    return ("other", role or "unknown")


def build_trace_timeline(records: List[Dict[str, Any]],
                         messages: Optional[List[Dict[str, Any]]] = None,
                         ) -> List[Dict[str, Any]]:
    """Derive swimlane segments from raw trace JSONL records (read-time only).

    Stored traces are never mutated; durations are computed here from
    per-event timestamps. Timestamp lookup order per message record: the
    record's own ``ts``/``timestamp``, then the embedded message's fields —
    today's recorder writes none of these, so a plain trace yields only the
    single run-wide segment anchored at the header timestamp and closed by
    footer ``duration_s`` (the "everything was one llm/tool phase" fallback is
    deliberately honest: zero-duration event streams render nothing client-side).

    Each returned entry: {start_ts, duration_ms, kind, label, message_index}
    where kind is in TIMELINE_KINDS and message_index indexes ``messages``
    (the delivered transcript; defaults to the messages found in ``records``).
    Degenerate inputs yield [] — never raises.
    """
    header: Optional[Dict[str, Any]] = None
    footer: Optional[Dict[str, Any]] = None
    msg_records: List[Tuple[Optional[float], Dict[str, Any]]] = []
    for rec in records or []:
        if not isinstance(rec, dict):
            continue
        kind = rec.get("kind")
        if kind == "trace_header":
            header = rec
        elif kind == "trace_footer":
            footer = rec
        elif kind == "message" and isinstance(rec.get("message"), dict):
            ts = _timeline_ts(rec.get("ts"))
            if ts is None:
                ts = _timeline_ts(rec.get("timestamp"))
            if ts is None:
                ts = _timeline_ts(rec["message"].get("ts"))
            if ts is None:
                ts = _timeline_ts(rec["message"].get("timestamp"))
            msg_records.append((ts, rec["message"]))

    # message_index must anchor into the *delivered* transcript (the detail
    # view may have q-filtered it): match by object identity first, then fall
    # back to positional order over the unfiltered records.
    id_index: Dict[int, int] = {}
    if messages is not None:
        for i, m in enumerate(messages):
            id_index.setdefault(id(m), i)

    def _anchor(i: int, m: Dict[str, Any]) -> Optional[int]:
        if messages is None:
            return i
        # Identity match only: when the transcript was q-filtered, a message
        # that did not survive filtering has no valid index -> skip its segment.
        return id_index.get(id(m))

    raw_start = _timeline_ts((header or {}).get("timestamp"))
    dur_s = (footer or {}).get("duration_s")
    end_ts: Optional[float] = None
    if isinstance(dur_s, (int, float)) and not isinstance(dur_s, bool):
        if raw_start is not None:
            end_ts = raw_start + max(0.0, float(dur_s))
        else:
            end_ts = max(0.0, float(dur_s))

    timed = [(ts, i, m) for i, (ts, m) in enumerate(msg_records) if ts is not None]
    timed.sort(key=lambda t: t[0])

    segs: List[Dict[str, Any]] = []
    if timed:
        # Per-event durations: each message owns the gap until the next event.
        for n, (ts, i, m) in enumerate(timed):
            nxt = timed[n + 1][0] if n + 1 < len(timed) else end_ts
            duration_ms = int(round(((nxt - ts) * 1000.0) if (nxt is not None
                                                               and nxt >= ts) else 0))
            kind, label = _timeline_kind(m)
            mi = _anchor(i, m)
            if mi is None:
                continue
            segs.append({"start_ts": round(ts, 3), "duration_ms": duration_ms,
                         "kind": kind, "label": label, "message_index": mi})
    elif raw_start is not None or end_ts is not None:
        # No per-event timestamps (current recorder): one run-wide segment
        # spanning header timestamp -> header + footer.duration_s, colored by
        # the dominant activity so slow runs are still visible as one bar.
        kinds = [(_timeline_kind(m)[0], m) for _, m in msg_records]
        if any(k == "tool" for k, _ in kinds):
            kind, label = "tool", "run (tools)"
        elif any(k == "llm" for k, _ in kinds):
            kind, label = "llm", "run"
        else:
            kind, label = "other", "run"
        start = raw_start if raw_start is not None else 0.0
        duration_ms = int(round(((end_ts - start) * 1000.0)
                                if (end_ts is not None and end_ts >= start)
                                else 0))
        segs.append({"start_ts": round(start, 3), "duration_ms": duration_ms,
                     "kind": kind, "label": label, "message_index": 0})

    # Cap: merge adjacent same-kind segments (keep first anchor) until under
    # the limit; if merging cannot help (all kinds alternate) keep the first
    # TIMELINE_MAX_SEGMENTS so rendering stays bounded.
    def _merge_once(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for s in items:
            if out and out[-1]["kind"] == s["kind"]:
                out[-1] = dict(out[-1], duration_ms=out[-1]["duration_ms"]
                               + s["duration_ms"], merged=True)
            else:
                out.append(dict(s))
        return out

    while len(segs) > TIMELINE_MAX_SEGMENTS:
        merged = _merge_once(segs)
        if len(merged) >= len(segs):
            segs = segs[:TIMELINE_MAX_SEGMENTS]
            break
        segs = merged
    return segs


def _model_detail_line(*, title: Optional[str], steps: Optional[int],
                       created_at: Optional[str], train_loss: Optional[float],
                       eval_loss: Optional[float]) -> str:
    """Compose the catalog-style detail line (backlog item #15).

    ``dataset · N steps · YYYY-MM-DD · loss 0.42/0.51`` — pieces missing from
    older runs are simply omitted; never a placeholder like 'None'.
    """
    parts: List[str] = []
    if title:
        parts.append(str(title))
    if steps is not None:
        parts.append(f"{steps} steps")
    if created_at:
        parts.append(str(created_at)[:10])
    if train_loss is not None:
        loss = f"loss {float(train_loss):.2f}"
        if eval_loss is not None:
            loss += f"/{float(eval_loss):.2f}"
        parts.append(loss)
    return " · ".join(parts)


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


class DashboardApp:
    """JSON API over the sable home. One instance per server process."""

    def __init__(self, cfg: SableConfig, home: Optional[Path] = None):
        self.cfg = cfg
        self.home = Path(home) if home else resolve_sable_home(cfg)
        # Serialize local submit+run executions (registry single-flight is the
        # cross-process backstop).
        self._exec_lock = threading.Lock()
        # Queue dispatcher + stale-job reaper (started by startup_dispatcher /
        # lazily by ensure_dispatcher; one per process over this home's DB,
        # guarded across processes by its pid-lockfile).
        self._dispatcher: Optional[Any] = None
        # llama.cpp setup/update background task (one at a time per process).
        self._runtime_lock = threading.Lock()
        self._runtime_setup: Optional[Dict[str, Any]] = None
        # Engine-catalog installs (Hardware page "Install engines"): one
        # background install per recipe id; each gets its own record so two
        # different engines can install concurrently but one cannot double-run.
        self._engine_install_lock = threading.Lock()
        self._engine_installs: Dict[str, Dict[str, Any]] = {}
        # Tuning panel: bench + calibrate background runs (one of each; both
        # drive a llama-server of their own and must not fight over the GPU).
        self._bench_lock = threading.Lock()
        self._bench_run: Optional[Dict[str, Any]] = None
        self._calibrate_lock = threading.Lock()
        self._calibrate_run: Optional[Dict[str, Any]] = None
        # Doctor run (WP7 checks). Background like bench/calibrate: rocm-smi
        # on a busy GPU can block in the kernel past its subprocess timeout,
        # which would hang the HTTP handler indefinitely.
        self._doctor_lock = threading.Lock()
        self._doctor_run: Optional[Dict[str, Any]] = None
        self.bus = EventBus()
        self.hardware = HardwareSampler(self.home)
        self._chat_runs: Dict[str, Dict[str, Any]] = {}
        # Guards the single-flight scan-and-insert below (two concurrent POSTs
        # for the same thread must not both pass the in-flight check).
        self._chat_runs_lock = threading.Lock()
        # /api/metrics TTL cache: (expires_at_monotonic, exposition_text).
        # Scrapes arrive every ~15 s; recomputation is cheap but this keeps
        # bursty scrapers off the DB entirely. Guarded by _metrics_lock.
        self._metrics_cache: Optional[Tuple[float, str]] = None
        self._metrics_lock = threading.Lock()
        # WP5 GPU coexistence (idle-unload / lazy-load / train handshake).
        # Constructing the engine is inert: every automatic behavior is gated
        # by its own serve.* key (all default off), so with no config changes
        # nothing runs and observable behavior matches pre-WP5. The explicit
        # load/unload endpoints work regardless — they are user actions, not
        # background policy.
        self.coexist: Optional[Any] = None
        try:
            from ..serving.coexist import Coexistence, coexist_config

            # EmitFn is typed -> None; EventBus.publish returns the event dict
            # (test convenience), so adapt instead of passing it directly.
            def _emit(kind: str, payload: Dict[str, Any]) -> None:
                self.bus.publish(kind, payload)

            self.coexist = Coexistence(coexist_config(cfg), parent_cfg=cfg,
                                       home=self.home, emit=_emit)
        except Exception:  # pragma: no cover - defensive
            logger.debug("coexistence engine unavailable", exc_info=True)
        self._routes: List[Route] = [
            (m, _compile_route(p), h) for (m, p, h) in self.ROUTES
        ]

    # ------------------------------------------------------------- routing

    ROUTES: List[Tuple[str, str, str]] = [
        # status & events
        ("GET", "/api/status", "h_status"),
        ("GET", "/api/events", "h_events"),
        # Prometheus text exposition (review idea #5) — same auth/loopback
        # rules as every other /api/* route (enforced in server.py).
        ("GET", "/api/metrics", "h_metrics"),
        # serve / deploy
        ("GET", "/api/serve/status", "h_serve_status"),
        ("POST", "/api/serve/start", "h_serve_start"),
        ("POST", "/api/serve/stop", "h_serve_stop"),
        ("POST", "/api/serve/attach", "h_serve_attach"),
        ("POST", "/api/deploy", "h_deploy"),
        ("POST", "/api/serve/deploy", "h_deploy"),
        # remote sable peers (LAN): probe + remember another machine's
        # dashboard so its agents/servers/training/traces show up here
        ("POST", "/api/remote/probe", "h_remote_probe"),
        ("GET", "/api/remote/peers", "h_remote_peers_list"),
        ("GET", "/api/remote/peers/<peer_id>/live", "h_remote_peer_live"),
        ("DELETE", "/api/remote/peers/<peer_id>", "h_remote_peer_delete"),
        # multi-instance serving (Phase 6) — static path before <id> ones
        ("GET", "/api/servers", "h_servers_list"),
        ("POST", "/api/servers", "h_servers_create"),
        ("GET", "/api/servers/<server_id>/status", "h_server_status"),
        ("POST", "/api/servers/<server_id>/start", "h_server_start"),
        ("POST", "/api/servers/<server_id>/stop", "h_server_stop"),
        # WP5 explicit coexistence control (load/unload by name)
        ("POST", "/api/servers/<server_id>/load", "h_server_load"),
        ("POST", "/api/servers/<server_id>/unload", "h_server_unload"),
        ("PUT", "/api/servers/<server_id>", "h_server_put"),
        ("DELETE", "/api/servers/<server_id>", "h_server_delete"),
        # inference-engine registry (opt-in; Hardware page manages it)
        ("GET", "/api/engines", "h_engines_list"),
        ("POST", "/api/engines", "h_engines_create"),
        ("PUT", "/api/engines/<engine_id>", "h_engine_put"),
        ("DELETE", "/api/engines/<engine_id>", "h_engine_delete"),
        # installable engine catalog (opt-in; Hardware page installs it)
        ("GET", "/api/engine-catalog", "h_engine_catalog"),
        ("POST", "/api/engine-catalog/<recipe_id>/install",
         "h_engine_install_start"),
        ("GET", "/api/engine-catalog/<recipe_id>/install",
         "h_engine_install_status"),
        # hardware / config
        ("GET", "/api/hardware", "h_hardware"),
        ("GET", "/api/config", "h_config_get"),
        ("PUT", "/api/config", "h_config_put"),
        # runtime (llama.cpp install/update)
        ("GET", "/api/runtime", "h_runtime_get"),
        ("POST", "/api/runtime/setup", "h_runtime_setup"),
        ("GET", "/api/runtime/setup", "h_runtime_setup_status"),
        ("POST", "/api/runtime/sync", "h_runtime_sync"),
        # tuning panel: doctor --env, bench (WP1), calibrate (WP3)
        ("GET", "/api/doctor/env", "h_doctor_env"),
        ("POST", "/api/doctor/run", "h_doctor_run"),
        ("POST", "/api/doctor/env", "h_doctor_env_fix"),
        ("GET", "/api/bench", "h_bench_list"),
        ("GET", "/api/bench/targets", "h_bench_targets"),
        ("POST", "/api/bench/run", "h_bench_run"),
        ("GET", "/api/bench/run", "h_bench_run_status"),
        ("GET", "/api/calibrate", "h_calibrate_show"),
        ("POST", "/api/calibrate/run", "h_calibrate_run"),
        ("GET", "/api/calibrate/run", "h_calibrate_run_status"),
        ("POST", "/api/calibrate/clear", "h_calibrate_clear"),
        # chat
        ("POST", "/api/proxy/chat", "h_proxy_chat"),
        ("POST", "/api/chat/stream", "h_chat_stream"),
        ("POST", "/api/chat/cancel", "h_chat_cancel"),
        ("GET", "/api/chat/runs", "h_chat_runs"),
        ("GET", "/api/chat/threads", "h_threads_list"),
        ("GET", "/api/chat/threads/<thread_id>", "h_thread_get"),
        ("PUT", "/api/chat/threads/<thread_id>", "h_thread_put"),
        ("DELETE", "/api/chat/threads/<thread_id>", "h_thread_delete"),
        # traces & feedback (static paths before <param> ones)
        ("GET", "/api/traces", "h_traces_list"),
        ("GET", "/api/traces/stats", "h_traces_stats"),
        ("GET", "/api/insights", "h_insights"),
        ("GET", "/api/traces/<trace_id>", "h_trace_detail"),
        ("POST", "/api/feedback", "h_feedback"),
        ("POST", "/api/traces/<trace_id>/feedback", "h_trace_feedback"),
        # Pass 4 #33: failure remediation lifecycle (status cycle per reason).
        ("POST", "/api/failures/note", "h_failure_note_set"),
        # Pass 4 #38: playbooks (list / promote / retire) — recipes distilled
        # from recurring tool chains, managed from the Profiles panel.
        ("GET", "/api/playbooks", "h_playbooks_list"),
        ("POST", "/api/playbooks/promote", "h_playbook_promote"),
        ("POST", "/api/playbooks/retire", "h_playbook_retire"),
        ("POST", "/api/playbooks/assess", "h_playbook_assess"),
        # auto-mined golden suite (curate/mine_goldens.py): list + toggle
        ("GET", "/api/goldens/mined", "h_goldens_mined_list"),
        ("POST", "/api/goldens/mined/toggle", "h_goldens_mined_toggle"),
        # datasets & curation
        ("GET", "/api/datasets", "h_datasets_list"),
        ("POST", "/api/curate", "h_curate"),
        ("POST", "/api/datasets/export", "h_dataset_export"),
        # static path before <dataset_id> (idea #6: version diff)
        ("GET", "/api/datasets/diff", "h_dataset_diff"),
        ("GET", "/api/datasets/<dataset_id>", "h_dataset_get"),
        ("GET", "/api/datasets/<dataset_id>/rows", "h_dataset_rows"),
        # WP9 analytics: curation funnel history + validation-gate telemetry
        ("GET", "/api/analytics/curation", "h_analytics_curation"),
        ("GET", "/api/analytics/validation", "h_analytics_validation"),
        # models registry
        ("GET", "/api/models", "h_models_list"),
        # discovered models on disk (base folder / HF hub cache)
        ("GET", "/api/model_catalog", "h_model_catalog"),
        # vision projector (mmproj): status + download into the GGUF's folder
        ("GET", "/api/mmproj", "h_mmproj_status"),
        ("POST", "/api/mmproj/download", "h_mmproj_download"),
        # model hub: search / download / manage (core/modelhub.py)
        ("GET", "/api/hub/search", "h_hub_search"),
        ("GET", "/api/hub/feed", "h_hub_feed"),
        ("GET", "/api/hub/detail", "h_hub_detail"),
        ("GET", "/api/hub/image", "h_hub_image"),
        ("GET", "/api/hub/variants", "h_hub_variants"),
        ("GET", "/api/hub/installed", "h_hub_installed"),
        ("POST", "/api/hub/download", "h_hub_download"),
        ("GET", "/api/hub/downloads", "h_hub_downloads"),
        ("GET", "/api/hub/downloads/<job_id>", "h_hub_download_get"),
        ("POST", "/api/hub/cancel", "h_hub_cancel"),
        ("POST", "/api/hub/delete", "h_hub_delete"),
        ("POST", "/api/models/promote", "h_model_promote"),
        ("POST", "/api/models/rollback", "h_model_rollback"),
        ("GET", "/api/models/<model_id>", "h_model_get"),
        # jobs & training
        ("GET", "/api/jobs", "h_jobs_list"),
        ("POST", "/api/jobs/cancel", "h_job_cancel"),
        ("GET", "/api/jobs/<job_id>", "h_job_get"),
        ("GET", "/api/jobs/<job_id>/log", "h_job_log"),
        ("POST", "/api/train", "h_train"),
        ("GET", "/api/train/check", "h_train_check"),
        ("GET", "/api/train/estimate", "h_train_estimate"),
        # rejection-sampling fine-tuning (STaR) — queued job, same submit path
        ("POST", "/api/rft", "h_rft"),
        # auto-mine golden tasks from 👍 traces — queued job, same submit path
        ("POST", "/api/mine_goldens", "h_mine_goldens"),
        # one-shot feature presets (core/presets.py) — "enable everything
        # worth enabling" with a reviewable diff first (Options panel button)
        ("POST", "/api/features/enable_all", "h_features_enable_all"),
        # easy training (turnkey loop, one or more agent profiles) — static
        # paths first
        ("GET", "/api/easy", "h_easy_get"),
        ("POST", "/api/easy/enable", "h_easy_enable"),
        ("POST", "/api/easy/disable", "h_easy_disable"),
        ("POST", "/api/easy/add", "h_easy_add"),
        ("POST", "/api/easy/remove", "h_easy_remove"),
        ("POST", "/api/easy/run", "h_easy_run"),
        # iterate pipeline
        ("POST", "/api/iterate", "h_iterate"),
        ("GET", "/api/iterate/runs", "h_iterate_runs"),
        # program control
        ("POST", "/api/shutdown", "h_shutdown"),
        ("GET", "/api/logs", "h_logs"),
        # codebase index (semantic search)
        ("GET", "/api/index", "h_index_status"),
        ("POST", "/api/index/build", "h_index_build"),
        ("POST", "/api/index/search", "h_index_search"),
        ("POST", "/api/index/stop", "h_index_stop"),
        # prompt evolution proposals (pass 4 #39) — gated apply/dismiss
        ("GET", "/api/evolve", "h_evolve_latest"),
        ("POST", "/api/evolve/apply", "h_evolve_apply"),
        # persistent skills (core/skills.py) — human review surface: the
        # documented mechanism is editing these files, so the dashboard must
        # at least list/read them; delete goes through core.skills.delete_skill.
        ("GET", "/api/skills", "h_skills_list"),
        ("GET", "/api/skills/<skill_name>", "h_skill_get"),
        ("PUT", "/api/skills/<skill_name>", "h_skill_put"),
        ("DELETE", "/api/skills/<skill_name>", "h_skill_delete"),
        ("POST", "/api/evolve/dismiss", "h_evolve_dismiss"),
        # agent profiles (static path before <param> one)
        ("GET", "/api/profiles", "h_profiles_list"),
        ("PUT", "/api/profiles/default", "h_profiles_set_default"),
        ("PUT", "/api/profiles/<name>", "h_profile_put"),
        ("DELETE", "/api/profiles/<name>", "h_profile_delete"),
        # projects (static paths before the <param> ones)
        ("GET", "/api/projects", "h_projects_list"),
        ("POST", "/api/projects", "h_projects_create"),
        ("PUT", "/api/projects/<project_id>", "h_project_put"),
        ("DELETE", "/api/projects/<project_id>", "h_project_delete"),
        # workspace file explorer (read-only lazy tree + file preview)
        ("GET", "/api/fs/tree", "h_fs_tree"),
        ("GET", "/api/fs/read", "h_fs_read"),
    ]

    def handle(
        self,
        method: str,
        path: str,
        query: Optional[Dict[str, str]] = None,
        body: Optional[Dict[str, Any]] = None,
    ) -> ApiResponse:
        query = query or {}
        body = body or {}
        method = method.upper()
        try:
            return self._route(method, path.rstrip("/") or "/", query, body)
        except HttpError as e:
            return ApiResponse(e.status, {
                "error": {"code": e.code, "message": e.message, **e.extra}
            })
        except Exception as e:  # never leak a traceback to the UI
            return ApiResponse(500, {
                "error": {"code": "internal_error",
                          "message": f"{type(e).__name__}: {e}"}
            })

    def _route(self, method: str, path: str, q: Dict[str, str],
               b: Dict[str, Any]) -> ApiResponse:
        for m, rx, handler_name in self._routes:
            match = rx.match(path)
            if not match:
                continue
            if m != method:
                continue  # verb mismatch => keep scanning => 404 (legacy shape)
            handler = getattr(self, handler_name)
            out = handler(q=q, b=b, **{k: unquote(v)
                                       for k, v in match.groupdict().items()})
            if isinstance(out, ApiResponse):
                return out
            return ApiResponse(200, out)
        raise HttpError(404, f"no such endpoint: {method} {path}")

    # ------------------------------------------------------------- plumbing

    def _store(self) -> TraceStore:
        return TraceStore(self.home / "buffer" / "sable.db")

    def _registry(self) -> ModelRegistry:
        return ModelRegistry(self.home / "buffer" / "sable.db")

    def _training_config(self) -> TrainingConfig:
        tc = TrainingConfig.from_yaml({"training": self.cfg.training or {}})
        if not tc.base_model and self.cfg.training.get("base_model"):
            tc.base_model = self.cfg.training["base_model"]
        for k, v in (self.cfg.training or {}).items():
            if hasattr(tc, k) and getattr(tc, k) in (None, 0, False):
                setattr(tc, k, v)
        return tc

    def _job_observer(self, job_id: str, state: str) -> None:
        """Scheduler observer: mirror transitions onto the event bus."""
        self.bus.publish("job.state", {"job_id": job_id, "state": state})
        reg = self._registry()
        try:
            job = reg.get_job(job_id)
        finally:
            reg.close()
        if job and job.log_path and Path(job.log_path).is_file():
            try:
                lines = Path(job.log_path).read_text(errors="replace").splitlines()
            except OSError:
                lines = []
            self.bus.publish("job.log", {"job_id": job_id, "lines": lines[-5:]})

    def _scheduler(self, reg: ModelRegistry):
        from ..training.scheduler import Scheduler

        # WP5 handshake wiring: rescue_gate self-gates on serve.* keys
        # (disabled => returns False immediately and the refusal path is the
        # pre-WP5 one), so passing the engine here changes nothing by default.
        return Scheduler(registry=reg, cfg=self._training_config(),
                         observer=self._job_observer,
                         coexist=self.coexist)

    # ---- queue dispatcher --------------------------------------------------

    def ensure_dispatcher(self):
        """Start (once per process) the queue dispatcher + stale-job reaper.

        Returns the running JobDispatcher. Raises DispatcherAlreadyRunning
        when another live process holds this DB's dispatcher lockfile — two
        dashboards must not drive one queue with independent in-process state;
        callers surface it loudly (serve_dashboard aborts boot). Other OSErrors
        (unwritable home) are logged and return None: a missing dispatcher
        degrades to submit-thread execution instead of killing the dashboard.
        """
        from ..training.scheduler import (DispatcherAlreadyRunning,
                                          JobDispatcher)

        if self._dispatcher is not None and self._dispatcher.is_running():
            return self._dispatcher
        db_path = self.home / "buffer" / "sable.db"

        def _factory():
            # One registry per dispatcher thread (SQLite connections are
            # created-and-used on the same thread; check_same_thread=False is
            # only a workaround for the request threads, not an invitation to
            # share handles across the loop and us).
            return self._scheduler(self._registry())

        d = JobDispatcher(db_path=db_path, scheduler_factory=_factory)
        try:
            d.start()
        except DispatcherAlreadyRunning:
            raise  # contention is a hard refusal — caller fails loudly
        except OSError as e:  # pragma: no cover - defensive (unwritable home)
            logger.error("queue dispatcher failed to start: %s", e)
            return None
        self._dispatcher = d
        return d

    def stop_dispatcher(self) -> None:
        d, self._dispatcher = self._dispatcher, None
        if d is not None:
            try:
                d.stop()
            except Exception:  # pragma: no cover - defensive
                logger.debug("dispatcher stop failed", exc_info=True)

    def _log_workspace_root(self) -> None:
        """Log the RESOLVED agent workspace root at startup.

        `agent.workspace_root` unset silently defaults to the process cwd —
        usually $HOME when the dashboard is launched from a desktop entry —
        which quietly changes what the file tools can touch. Make the resolved
        value (and the fallback) visible in the log."""
        from ..core.config import UNSET

        agent_cfg = getattr(self.cfg, "agent", None) or {}
        raw = agent_cfg.get("workspace_root") if "workspace_root" in agent_cfg \
            else None
        if "workspace_root" not in agent_cfg:
            resolved = Path(os.getcwd())
            logger.warning(
                "workspace root: agent.workspace_root is unset — falling back "
                "to the dashboard's current working directory %s. Chat file "
                "tools are contained to it (plus SABLE_HOME and temp); set "
                "agent.workspace_root in the config to pin it explicitly.",
                resolved)
        elif raw is UNSET or raw is None:
            logger.warning(
                "workspace root: agent.workspace_root is null — workspace "
                "containment is DISABLED for chat runs (see core/agent_runner).")
        else:
            logger.info("workspace root: %s", Path(str(raw)).expanduser())

    def startup_dispatcher(self):
        """Dashboard boot hook: reap stale jobs immediately, run the periodic
        dispatcher, and log the resolved workspace root. Safe to call twice;
        returns the running dispatcher (raises DispatcherAlreadyRunning on
        lockfile contention — serve_dashboard aborts loudly on that)."""
        self._log_workspace_root()
        return self.ensure_dispatcher()

    @property
    def token(self) -> Optional[str]:
        """Dashboard auth token — ON by default (production hardening).

        Resolution order:
          1. ``dashboard.token`` in config (the CLI ``--token`` flag injects
             here; an explicit empty string DISABLES auth entirely);
          2. env ``SABLE_DASHBOARD_TOKEN`` (empty value also disables);
          3. a generated secret persisted at ``<home>/dashboard.token`` —
             created on first dashboard start, stable across restarts, so
             /api/* is never wide open by default even with zero config.

        Loopback binding + Origin checks stop remote and DNS-rebinding
        attackers, but any local process could previously drive the full
        mutation surface (promote models, edit config, run tasks). The token
        closes that; ``dashboard.token: ""`` or ``--no-token`` opts out for
        users who accept single-user-machine exposure.
        """
        sec = self.cfg.dashboard or {}
        if "token" in sec:
            # "" and null both mean "explicitly disabled"; any other value
            # is the required secret (never stringified through str(None)).
            tok = sec["token"]
            return str(tok).strip() if tok else None
        env = os.environ.get("SABLE_DASHBOARD_TOKEN")
        if env is not None:
            return env.strip() or None
        # Default: generated + persisted (0600, atomic create).
        p = self.home / "dashboard.token"
        try:
            tok = p.read_text(encoding="utf-8").strip()
            if tok:
                return tok
        except FileNotFoundError:
            pass
        except OSError as e:  # unreadable: fall through to regenerate-in-RAM
            logger.warning("dashboard token file unreadable (%s); using an "
                           "ephemeral token for this process", e)
            return secrets.token_urlsafe(32)
        tok = secrets.token_urlsafe(32)
        try:
            self.home.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(tok + "\n")
            os.replace(tmp, p)
        except OSError as e:  # pragma: no cover - home unwritable is rare
            logger.warning("could not persist dashboard token (%s); using an "
                           "ephemeral token — restart will print a new one", e)
        return tok

    def check_token(self, provided: Optional[str]) -> bool:
        expected = self.token
        if not expected:
            return True
        if not provided:
            return False
        return hmac.compare_digest(str(provided), expected)

    # --------------------------------------------------------------- events

    def h_events(self, q: Dict[str, str], b: Dict[str, Any]) -> ApiResponse:
        """SSE global bus. Heartbeat comments every ~15 s; clean disconnect."""

        def gen() -> Iterator[str]:
            sub = self.bus.subscribe()
            try:
                yield ": sable events\nretry: 3000\n\n"
                while True:
                    ev = sub.poll(timeout=SSE_HEARTBEAT_S)
                    if ev is None:
                        if sub.closed:
                            return
                        yield ": ping\n\n"  # keep proxies/browsers warm
                        continue
                    yield format_sse(ev)
            finally:
                sub.close()

        return ApiResponse(200, stream=gen())

    # --------------------------------------------------------------- status

    def h_status(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        from .. import __version__

        store = self._store()
        try:
            stats = store.stats()
        finally:
            store.close()
        # WP2c metric honesty: avg_outcome_score is computed only over SCORED
        # traces; surface the raw denominator beside the rate so the overview
        # can show "avg 0.9 (3/400 scored)" instead of a bare average.
        _tot = stats.get("total_traces") or 0
        _scored = stats.get("scored_traces") or 0
        stats["unscored_traces"] = max(0, int(_tot) - int(_scored))

        serve_state = read_serve_state(self.home)
        server_live = False
        if serve_state and serve_state.get("pid"):
            # Windows-safe liveness (A1): os.kill(pid, 0) TERMINATES on NT.
            from ..serving.runtime import pid_alive as _pid_alive

            try:
                _ppid = int(serve_state["pid"])
            except (ValueError, TypeError):
                _ppid = 0
            # None ("alive but not ours") keeps the old PermissionError branch.
            server_live = bool(_ppid) and _pid_alive(_ppid) is not False

        reg = self._registry()
        try:
            active = reg.active_model()
            models = reg.list_models()
            jobs = reg.list_jobs()
            # Pass4 #31: compact operation-ledger lineage for the overview.
            recent_ops = reg.history(limit=10)
        finally:
            reg.close()

        return {
            "version": __version__,
            "home": str(self.home),
            "server": {
                "running": server_live,
                "state": _redact(serve_state) if serve_state else None,
            },
            "buffer": stats,
            "active_model": active.model_id if active else None,
            "model_count": len(models),
            "jobs": {
                "queued": sum(1 for j in jobs if j.state == "queued"),
                "running": sum(1 for j in jobs if j.state == "running"),
                "total": len(jobs),
            },
            "recent_ops": recent_ops,
        }

    # ------------------------------------------------------------ serving

    def _serve_alive(self, st: Dict[str, Any]) -> bool:
        mode = st.get("mode", "spawn")
        if mode == "attach":
            return healthcheck(
                str(st.get("base_url", "")), timeout=2.0,
                api_key=st.get("attach_api_key"),
            )
        pid = st.get("pid")
        if not pid:
            return False
        # Windows-safe liveness (A1): os.kill(pid, 0) TERMINATES on NT.
        from ..serving.runtime import pid_alive

        try:
            pid_i = int(pid)
        except (ValueError, TypeError):
            return False
        # None ("alive but not ours") keeps the old PermissionError branch.
        return pid_alive(pid_i) is not False

    def h_serve_status(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        from ..serving.supervisor import crash_loop_status

        st = read_serve_state(self.home)
        # Pass 4 #28: per-backend circuit-breaker states (runner/judge side),
        # keyed by base_url — the dashboard's view of "this backend has been
        # failing and is being failed-fast". Only breakers with activity.
        from ..core.circuit import all_breakers
        breakers = {k: v for k, v in all_breakers().items()
                    if v.get("state") != "closed" or v.get("failures")}
        if not st:
            return {"running": False, "mode": "none",
                    "crash_loop": crash_loop_status(srv_inst.DEFAULT_ID)}
        running = self._serve_alive(st)
        out = {
            "running": running,
            "mode": st.get("mode", "spawn"),
            "pid": st.get("pid"),
            "base_url": st.get("base_url") if running else None,
            "model_name": st.get("model_name"),
            "gguf_path": st.get("gguf_path"),
            "model_id": st.get("model_id"),
            "attach_url": st.get("attach_url"),
            "started_at": st.get("started_at"),
            "crash_loop": crash_loop_status(srv_inst.DEFAULT_ID),
            "circuits": breakers,
        }
        return out

    def h_serve_start(self, q: Dict[str, str], b: Dict[str, Any]) -> ApiResponse:
        from ..serving.supervisor import clear_crash_loop, start_instance_server

        if b.get("attach_url"):
            return self._serve_attach(b)
        # User-initiated start of the default server clears any crash-loop
        # refusal (#28). Mirrors h_server_start: internal/boot callers may set
        # autostart=true to keep the refusal in force (it then surfaces as the
        # wrapper's honest 500 instead of silently respawning).
        if not b.get("autostart"):
            clear_crash_loop(srv_inst.DEFAULT_ID)
        force = bool(b.get("force"))
        st = read_serve_state(self.home)
        if st and self._serve_alive(st) and not force:
            raise HttpError(409, "a server is already running; stop it first "
                                 "or pass force", code="already_serving")

        gguf = b.get("gguf") or None
        model_id = b.get("model_id") or None
        auto = bool(b.get("auto"))
        model_name = None

        if not gguf and (auto or model_id):
            reg = self._registry()
            try:
                if model_id:
                    m = reg.get_model(str(model_id))
                    if m is None:
                        raise HttpError(404, f"no model {model_id}")
                else:
                    m = reg.active_model()
                    if m is None:
                        raise HttpError(409, "no active model to serve "
                                             "(auto=true)", code="no_active_model")
                if not m.gguf_path:
                    raise HttpError(409, f"model {m.model_id} has no GGUF artifact",
                                    code="no_gguf")
                gguf = m.gguf_path
                model_id = m.model_id
                model_name = m.model_id
            finally:
                reg.close()
        if not gguf:
            gguf = self.cfg.gguf_path
        if not gguf:
            raise HttpError(400, "provide gguf, model_id, auto=true or attach_url")

        cfg = self.cfg
        overrides: Dict[str, Any] = {}
        if b.get("context"):
            overrides["context_length"] = int(b["context"])
        if b.get("port") is not None:
            overrides["port"] = int(b["port"])
        if overrides:
            cfg = _dc_replace(cfg, **overrides)

        if st and self._serve_alive(st):  # force path: stop owned first
            self._stop_instance(srv_inst.DEFAULT_ID)

        # Phase-6 consolidation: the legacy detached spawn (deploy.start_server_-
        # detached) is replaced by the canonical instance machinery. Starting
        # through start_instance_server with the config-derived reserved
        # "default" record gives this endpoint the crash-loop budget (#28,
        # accounted inside the wrapper), registry visibility (_owned_handles ->
        # owned_instance_ids, so stop_all_servers/shutdown reaches it) and
        # per-instance state files — while _write_instance_state keeps
        # serve_state.json in sync for every legacy reader (shadow eval,
        # status panels, CLI). Log stays at logs/server-default.log.
        run_inst = dict(srv_inst.default_instance(cfg))
        run_inst["gguf"] = str(gguf)
        # This branch always SPAWNS (attach requests returned above): never let
        # a config-level attach_url leak into the spawn record.
        run_inst["attach_url"] = None
        # Preserve the legacy alias precedence: explicit model > cfg.model_name
        # > shard group / GGUF stem (handled inside _start_instance_server).
        run_inst["model_name"] = model_name or cfg.model_name or None
        try:
            handle = start_instance_server(
                cfg, run_inst,
                log_path=self.home / "logs" / "server-default.log")
        except ServeError as e:
            raise HttpError(500, str(e)) from e

        state = {
            "mode": "spawn",
            "pid": handle.process.pid if handle.process else None,
            "host": "127.0.0.1",
            "port": int(handle.base_url.rsplit(":", 1)[1].split("/")[0]),
            "base_url": handle.base_url,
            "model_name": handle.model_name,
            "model_id": model_id,
            "gguf_path": str(gguf),
            "instance_id": srv_inst.DEFAULT_ID,
            "started_at": _utcnow(),
        }
        self._write_instance_state(srv_inst.DEFAULT_ID, state)
        self.bus.publish("serve.state", {"event": "started", **{
            k: state.get(k) for k in ("base_url", "model_name", "model_id",
                                      "gguf_path", "pid")}})
        return ApiResponse(202, {"ok": True, "base_url": state["base_url"],
                                 "model_name": state["model_name"]})

    def _serve_attach(self, b: Dict[str, Any]) -> ApiResponse:
        url = str(b.get("attach_url") or b.get("url") or "").strip()
        if not url:
            raise HttpError(400, "attach requires a url")
        api_key = b.get("api_key") or None
        base = url.rstrip("/")
        if not base.endswith("/v1"):
            base = base + "/v1"
        if not healthcheck(base, timeout=5.0, api_key=api_key):
            raise HttpError(400, f"attach target {base} is not healthy "
                                 "(GET /models failed)")
        st = read_serve_state(self.home)
        if st and self._serve_alive(st) and st.get("mode") != "attach":
            stop_owned_server(self.home)
        model_name = None
        try:
            import httpx

            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            r = httpx.get(base + "/models", timeout=5.0, headers=headers)
            data = (r.json().get("data") or []) if r.status_code == 200 else []
            if data and isinstance(data[0], dict):
                model_name = data[0].get("id")
        except Exception:
            pass
        state = {
            "mode": "attach",
            "base_url": base,
            "attach_url": url,
            "attach_api_key": api_key,
            "model_name": model_name or self.cfg.model_name or "attached-model",
            "started_at": _utcnow(),
        }
        write_serve_state(self.home, state)
        self.bus.publish("serve.state", {"event": "attached", "base_url": base,
                                         "model_name": state["model_name"]})
        return ApiResponse(202, {"ok": True, "base_url": base,
                                 "model_name": state["model_name"]})

    def h_serve_attach(self, q: Dict[str, str], b: Dict[str, Any]) -> ApiResponse:
        return self._serve_attach(b)

    def h_serve_stop(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        st = read_serve_state(self.home)
        stopped = False
        if st and st.get("mode", "spawn") == "attach":
            clear_serve_state(self.home)  # never kill an attached foreign server
        elif st:
            # _stop_instance prefers the in-process owned handle (spawned via
            # start_instance_server) and falls back to the recorded pid — so
            # servers started here AND legacy-started servers both stop.
            stopped = self._stop_instance(srv_inst.DEFAULT_ID)
        self.bus.publish("serve.state", {"event": "stopped", "stopped": stopped})
        return {"ok": True, "stopped": stopped}

    def h_deploy(self, q: Dict[str, str], b: Dict[str, Any]) -> ApiResponse:
        model_id = str(b.get("model_id", ""))
        if not model_id:
            raise HttpError(400, "model_id required")
        reg = self._registry()
        try:
            m = reg.get_model(model_id)
            if m is None:
                raise HttpError(404, f"no model {model_id}")
            if not m.gguf_path:
                raise HttpError(409, f"model {model_id} has no GGUF artifact",
                                code="no_gguf")
            try:
                result = deploy_model(self.home, reg, model_id, self.cfg)
            except ServeError as e:
                raise HttpError(500, str(e)) from e
        finally:
            reg.close()
        self.bus.publish("serve.state", {"event": "deployed", **{
            k: result.get(k) for k in ("model_id", "base_url", "model_name")}})
        return ApiResponse(202, {"ok": True, "base_url": result["base_url"],
                                 "model_name": result["model_name"]})

    # --------------------------------------------------- remote LAN peers
    # Connect to ANOTHER machine running sable on the local network: probe
    # its dashboard (status/servers/profiles/jobs/traces), remember it, and
    # surface what that box is serving / what agents it has active / its
    # training + trace activity. Outbound-only GET polling with the peer's
    # dashboard token; nothing is ever pushed or spawned remotely. See
    # dashboard/peers.py for the record format and probe contract.

    def h_remote_probe(self, q: Dict[str, str],
                       b: Dict[str, Any]) -> ApiResponse:
        from . import peers as peers_mod

        raw = str(b.get("url") or "").strip()
        token = str(b.get("token") or "").strip() or None
        try:
            base = peers_mod.normalize_url(raw)
        except ValueError as e:
            raise HttpError(400, str(e)) from None
        snap = peers_mod.probe(base, token)
        if not snap.get("ok"):
            # A failed probe never remembers anything: the user fixes the
            # URL/token and retries (a dead peer in the list is noise).
            raise HttpError(400, f"remote sable at {base} is unreachable: "
                                 f"{snap.get('error')}", code="peer_unreachable")
        rec = peers_mod.upsert_peer(self.home, base, token,
                                    name=str(b.get("name") or "").strip() or None)
        peers_mod.set_snapshot(self.home, str(rec["id"]), snap)
        self.bus.publish("peers.changed", {"event": "added",
                                           "url": base})
        return ApiResponse(200, {"ok": True, "peer": rec and {
            "id": rec["id"], "url": base, "name": rec.get("name")},
            "snapshot": snap})

    def h_remote_peers_list(self, q: Dict[str, str],
                            b: Dict[str, Any]) -> Dict[str, Any]:
        from . import peers as peers_mod

        return {"peers": [peers_mod.peer_view(p)
                          for p in peers_mod.load_peers(self.home)]}

    def h_remote_peer_live(self, q: Dict[str, str], b: Dict[str, Any],
                           peer_id: str) -> Dict[str, Any]:
        """Re-probe one remembered peer and store the fresh snapshot."""
        from . import peers as peers_mod

        rec = peers_mod.get_peer(self.home, peer_id)
        if rec is None:
            raise HttpError(404, f"no peer {peer_id}", code="unknown_peer")
        snap = peers_mod.probe(str(rec["url"]), rec.get("token"))
        peers_mod.set_snapshot(self.home, peer_id, snap)
        fresh = peers_mod.get_peer(self.home, peer_id) or rec
        return {"peer": peers_mod.peer_view(fresh)}

    def h_remote_peer_delete(self, q: Dict[str, str], b: Dict[str, Any],
                             peer_id: str) -> Dict[str, Any]:
        from . import peers as peers_mod

        if not peers_mod.remove_peer(self.home, peer_id):
            raise HttpError(404, f"no peer {peer_id}", code="unknown_peer")
        self.bus.publish("peers.changed", {"event": "removed",
                                           "peer_id": peer_id})
        return {"ok": True}

    # ------------------------------------------------- multi-instance serving
    # Phase 6: N named server instances. The reserved id "default" mirrors the
    # legacy single-instance serve (config keys + serve_state.json) so old
    # panels/endpoints keep working; new instances persist as JSON files under
    # <home>/serving/instances/ with runtime state under <home>/serving/state/.

    def _instance_circuits(self, running: bool,
                           st: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Pass 4 #28: breaker snapshots for one instance's live base_url."""
        if not running or not st or not st.get("base_url"):
            return {}
        from ..core.circuit import all_breakers
        br = all_breakers().get(str(st["base_url"]))
        return {str(st["base_url"]): br} if br else {}

    def _instance_row(self, inst: Dict[str, Any]) -> Dict[str, Any]:
        from ..serving.supervisor import crash_loop_status

        iid = str(inst.get("id"))
        if iid == srv_inst.DEFAULT_ID:
            st = srv_inst.read_state(self.home, iid) or read_serve_state(self.home)
        else:
            st = srv_inst.read_state(self.home, iid)
        running = srv_inst.state_alive(st)
        # mode follows the live state when present (the reserved default's
        # definition is config-derived; serve_state.json knows the truth).
        mode = None
        if st:
            mode = st.get("mode") or ("attach" if inst.get("attach_url")
                                      else "spawn")
        return {
            "id": iid,
            "name": inst.get("name"),
            "role": inst.get("role") or "custom",
            "gguf": inst.get("gguf"),
            # Registered inference engine binding (serving/engines.py); None
            # => built-in llama.cpp. Additive field, old readers ignore it.
            "engine": inst.get("engine") or None,
            "attach_url": inst.get("attach_url") or (st or {}).get("attach_url"),
            "port": (st or {}).get("port") if running else inst.get("port"),
            "autostart": bool(inst.get("autostart")),
            "context_length": inst.get("context_length"),
            "run_settings": inst.get("run_settings") or {},
            "sampling": inst.get("sampling") or {},
            "created": inst.get("created"),
            "updated": inst.get("updated"),
            "default": iid == srv_inst.DEFAULT_ID,
            "running": running,
            "mode": mode,
            "base_url": (st or {}).get("base_url") if running else None,
            "model_name": (st or {}).get("model_name"),
            "pid": (st or {}).get("pid") if running else None,
            "started_at": (st or {}).get("started_at"),
            # Pass 4 #28: crash-loop budget view (None = nothing recorded).
            "crash_loop": crash_loop_status(iid),
            # Pass 4 #28: circuit-breaker snapshot for this instance's live
            # backend URL (runner/judge-side fail-fast state; {} when none).
            "circuits": self._instance_circuits(running, st),
            # WP5 coexistence load-state view: loaded | unloaded | unloading |
            # stopped. A pure derivation from the state file + engine flags;
            # with no WP5 config nothing is ever marked `unloaded`, so every
            # pre-WP5 row reads "loaded"/"stopped" exactly as its `running`
            # flag already implied (additive field, no behavior change).
            "load_state": self._load_state_of(iid, running, st),
        }

    def _load_state_of(self, server_id: str, running: bool,
                       st: Optional[Dict[str, Any]]) -> str:
        """WP5 load-state label (see _instance_row)."""
        if self.coexist is not None and server_id in getattr(
                self.coexist, "_unloading", ()):
            return "unloading"
        if running:
            return "loaded"
        if st and st.get("unloaded"):
            return "unloaded"
        return "stopped"

    def _load_instance_or_404(self, server_id: str) -> Dict[str, Any]:
        inst = srv_inst.get_instance(self.home, server_id, cfg=self.cfg)
        if inst is None:
            raise HttpError(404, f"no server instance {server_id}",
                            code="unknown_server")
        return inst

    def h_servers_list(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        rows = [self._instance_row(i)
                for i in srv_inst.load_instances(self.home, cfg=self.cfg)]
        return {"servers": rows}

    def h_server_status(self, q: Dict[str, str], b: Dict[str, Any],
                        server_id: str) -> Dict[str, Any]:
        inst = self._load_instance_or_404(server_id)
        return self._instance_row(inst)

    def _validate_new_server(self, b: Dict[str, Any]) -> Dict[str, Any]:
        existing_rows = srv_inst.load_instances(self.home, cfg=self.cfg)
        taken_ids = [str(i.get("id")) for i in existing_rows]
        taken_names = [str(i.get("name") or "").lower() for i in existing_rows]
        name = str(b.get("name") or "").strip()
        if not name:
            raise HttpError(400, "name required (non-empty string)")
        if name.lower() in taken_names:
            raise HttpError(409, f"a server named {name!r} already exists",
                            code="duplicate_name")
        try:
            inst = srv_inst.validate_instance_body(b)
        except ServeError as e:
            raise HttpError(400, str(e)) from e
        self._validate_engine_ref(inst)
        iid = srv_inst.new_instance_id(name, taken_ids)
        inst["id"] = iid
        port = int(inst.get("port") or 0)
        if port and srv_inst.check_port_conflict(self.home, self.cfg, port):
            raise HttpError(409, f"port {port} is already used by another "
                                 "managed server", code="port_conflict")
        return inst

    def h_servers_create(self, q: Dict[str, str], b: Dict[str, Any]) -> ApiResponse:
        inst = self._validate_new_server(b)
        srv_inst.save_instance(self.home, inst)
        self.bus.publish("servers.changed", {"event": "created",
                                             "id": inst["id"]})
        return ApiResponse(201, {"ok": True, "server": self._instance_row(inst)})

    def h_server_put(self, q: Dict[str, str], b: Dict[str, Any],
                     server_id: str) -> Dict[str, Any]:
        existing = self._load_instance_or_404(server_id)
        if server_id == srv_inst.DEFAULT_ID and "run_settings" in b:
            # The reserved default mirrors the legacy config; its run settings
            # live in the global run_settings: section (Settings page). A
            # default.json override would shadow config changes divergently.
            raise HttpError(400, "the default instance's run settings are the "
                                 "Settings page's Model run settings section",
                            code="reserved_instance")
        rows = srv_inst.load_instances(self.home, cfg=self.cfg)
        name = str(b.get("name") or existing.get("name") or "").strip()
        for other in rows:
            if str(other.get("id")) != server_id and \
                    str(other.get("name") or "").lower() == name.lower():
                raise HttpError(409, f"a server named {name!r} already exists",
                                code="duplicate_name")
        try:
            inst = srv_inst.validate_instance_body(b, existing=existing)
        except ServeError as e:
            raise HttpError(400, str(e)) from e
        self._validate_engine_ref(inst)
        port = int(inst.get("port") or 0)
        if port and srv_inst.check_port_conflict(self.home, self.cfg, port,
                                                 exclude_id=server_id):
            raise HttpError(409, f"port {port} is already used by another "
                                 "managed server", code="port_conflict")
        srv_inst.save_instance(self.home, inst)
        self.bus.publish("servers.changed", {"event": "updated",
                                             "id": server_id})
        return {"ok": True, "server": self._instance_row(inst)}

    def h_server_delete(self, q: Dict[str, str], b: Dict[str, Any],
                        server_id: str) -> Dict[str, Any]:
        if server_id == srv_inst.DEFAULT_ID:
            raise HttpError(400, "the reserved default instance cannot be "
                                 "deleted (it mirrors the legacy serve config)",
                            code="reserved_instance")
        self._load_instance_or_404(server_id)
        # Stop first when owned; attach instances are only forgotten.
        self._stop_instance(server_id, forget=True)
        srv_inst.delete_instance_file(self.home, server_id)
        self.bus.publish("servers.changed", {"event": "deleted",
                                             "id": server_id})
        return {"ok": True}

    def _instance_state(self, server_id: str) -> Optional[Dict[str, Any]]:
        if server_id == srv_inst.DEFAULT_ID:
            return srv_inst.read_state(self.home, server_id) \
                or read_serve_state(self.home)
        return srv_inst.read_state(self.home, server_id)

    def _write_instance_state(self, server_id: str,
                              state: Dict[str, Any]) -> None:
        srv_inst.write_state(self.home, server_id, state)
        if server_id == srv_inst.DEFAULT_ID:
            # Keep the legacy view in sync (old panels read serve_state.json).
            write_serve_state(self.home, state)

    def _clear_instance_state(self, server_id: str) -> None:
        srv_inst.clear_state(self.home, server_id)
        if server_id == srv_inst.DEFAULT_ID:
            clear_serve_state(self.home)

    def _stop_instance(self, server_id: str, forget: bool = False) -> bool:
        """Stop one instance: in-process child first, then recorded pid.
        Attach instances are never killed — only their state is cleared."""
        from ..serving.supervisor import stop_instance as _stop_inproc

        stopped = _stop_inproc(server_id)
        if not stopped:
            stopped = srv_inst.stop_recorded(self.home, server_id)
        if not stopped and server_id == srv_inst.DEFAULT_ID:
            # Legacy single-instance serve (serve_state.json) ownership path.
            st = read_serve_state(self.home)
            if st and st.get("mode", "spawn") != "attach":
                stopped = stop_owned_server(self.home)
        self._clear_instance_state(server_id)
        return stopped

    def h_server_load(self, q: Dict[str, str], b: Dict[str, Any],
                      server_id: str) -> ApiResponse:
        """WP5 explicit load: (re)launch one named instance through the
        coexistence engine's canonical spawn path. Already-loaded is an
        idempotent no-op (200); a fresh launch answers 202."""
        self._load_instance_or_404(server_id)
        if self.coexist is None:
            raise HttpError(500, "coexistence engine unavailable",
                            code="coexist_unavailable")
        was_stopped = self.coexist.load_state_of(server_id) != "loaded"
        try:
            st = self.coexist.load_instance(server_id, reason="manual")
        except Exception as e:
            raise HttpError(500, str(e), code="load_failed") from e
        return ApiResponse(202 if was_stopped else 200,
                           {"ok": True, "id": server_id,
                            "state": self.coexist.load_state_of(server_id),
                            "base_url": (st or {}).get("base_url")})

    def h_server_unload(self, q: Dict[str, str], b: Dict[str, Any],
                        server_id: str) -> Dict[str, Any]:
        """WP5 explicit unload: graceful stop + `unloaded` state marker so a
        later load (manual or lazy) can restart it. In flight => 409 retryable
        (the documented race policy; nothing is stopped)."""
        self._load_instance_or_404(server_id)
        if self.coexist is None:
            raise HttpError(500, "coexistence engine unavailable",
                            code="coexist_unavailable")
        from ..serving.watchdog import inflight_count

        st = self._instance_state(server_id) or {}
        url = str(st.get("base_url") or "")
        if url and inflight_count(url) > 0:
            raise HttpError(409, f"server instance {server_id} has a request "
                                 "in flight — retry the unload shortly",
                            code="request_in_flight")
        unloaded = self.coexist.unload_instance(server_id, reason="manual")
        return {"ok": True, "id": server_id, "unloaded": unloaded,
                "state": self.coexist.load_state_of(server_id)}

    def h_server_start(self, q: Dict[str, str], b: Dict[str, Any],
                       server_id: str) -> ApiResponse:
        from ..serving.supervisor import (
            ServeError as _ServeErr, clear_crash_loop, start_instance_server)

        # Explicit user action clears the crash-loop refusal (#28); the boot
        # autostart thread marks its requests so it can never self-clear.
        if not b.get("autostart"):
            clear_crash_loop(server_id)

        inst = self._load_instance_or_404(server_id)
        force = bool(b.get("force"))
        st = self._instance_state(server_id)
        if srv_inst.state_alive(st) and not force:
            raise HttpError(409, "this server is already running; stop it "
                                 "first or pass force", code="already_serving")
        if srv_inst.state_alive(st) and force:
            self._stop_instance(server_id)

        # Attach instance: validate loopback + health, record — spawn nothing.
        if inst.get("attach_url"):
            base = str(inst["attach_url"])
            try:
                srv_inst.require_loopback(base)
            except _ServeErr as e:
                raise HttpError(400, str(e)) from e
            api_key = b.get("api_key") or None
            if not healthcheck(base, timeout=5.0, api_key=api_key):
                raise HttpError(400, f"attach target {base} is not healthy "
                                     "(GET /models failed)",
                                code="attach_unhealthy")
            model_name = None
            try:
                import httpx

                r = httpx.get(base + "/models", timeout=5.0)
                data = (r.json().get("data") or []) if r.status_code == 200 \
                    else []
                if data and isinstance(data[0], dict):
                    model_name = data[0].get("id")
            except Exception:
                pass
            state = {
                "mode": "attach", "base_url": base,
                "attach_url": base, "attach_api_key": api_key,
                "model_name": model_name or inst.get("name") or "attached-model",
                "instance_id": server_id,
                "started_at": _utcnow(),
            }
            self._write_instance_state(server_id, state)
            self.bus.publish("serve.state", {"event": "attached",
                                             "base_url": base,
                                             "id": server_id})
            return ApiResponse(202, {"ok": True, "base_url": base,
                                     "model_name": state["model_name"],
                                     "warning": None})

        # Spawn instance: VRAM advisory (warn, don't block; force bypasses).
        gguf = inst.get("gguf")
        if not gguf:
            raise HttpError(400, "instance has neither gguf nor attach_url",
                            code="bad_instance")
        warning = None
        if not force:
            advisory = srv_inst.vram_advisory(inst)
            if advisory:
                warning = advisory["message"]

        # Port: honor configured, else shared free-port allocation; conflict
        # against other managed servers (incl. the embed server).
        port = int(b.get("port") or inst.get("port") or 0)
        if port and srv_inst.check_port_conflict(self.home, self.cfg, port,
                                                 exclude_id=server_id):
            raise HttpError(409, f"port {port} is already used by another "
                                 "managed server", code="port_conflict")
        if not port:
            try:
                port = srv_inst.pick_free_port(self.home, self.cfg,
                                               exclude_id=server_id)
            except _ServeErr as e:
                raise HttpError(500, str(e)) from e

        run_inst = dict(inst)
        run_inst["port"] = port
        if b.get("context"):
            run_inst["context_length"] = int(b["context"])
        try:
            handle = start_instance_server(
                self.cfg, run_inst,
                log_path=self.home / "logs" / f"server-{server_id}.log",
                ready_timeout=float(b.get("ready_timeout", 300.0)))
        except _ServeErr as e:
            raise HttpError(500, str(e)) from e

        state = {
            "mode": "spawn", "pid": handle.process.pid if handle.process else None,
            "host": "127.0.0.1", "port": port, "base_url": handle.base_url,
            "model_name": handle.model_name, "gguf_path": str(gguf),
            "instance_id": server_id, "role": inst.get("role"),
            "started_at": _utcnow(),
        }
        self._write_instance_state(server_id, state)
        self.bus.publish("serve.state", {"event": "started", "id": server_id,
                                         "base_url": handle.base_url,
                                         "model_name": handle.model_name})
        return ApiResponse(202, {"ok": True, "base_url": handle.base_url,
                                 "model_name": handle.model_name,
                                 "warning": warning})

    def startup_autostart(self) -> None:
        """Start autostart instances that aren't running (dashboard boot).

        Best-effort and non-blocking: failures land in a serve.state error
        event, never crash the dashboard. VRAM advisory is skipped here — an
        explicit autostart flag IS the user's force."""
        def _run() -> None:
            for inst in srv_inst.load_instances(self.home, cfg=self.cfg):
                if not inst.get("autostart") or inst.get("id") == \
                        srv_inst.DEFAULT_ID:
                    continue
                iid = str(inst["id"])
                try:
                    if srv_inst.state_alive(self._instance_state(iid)):
                        continue
                    # autostart=True: never clears the crash-loop refusal —
                    # only an explicit user Start does (#28).
                    self.handle("POST", f"/api/servers/{iid}/start",
                                body={"force": True, "autostart": True})
                except Exception as e:  # pragma: no cover - defensive
                    self.bus.publish("serve.state", {
                        "event": "autostart_error", "id": iid,
                        "error": f"{type(e).__name__}: {e}"})

        threading.Thread(target=_run, daemon=True,
                         name="servers-autostart").start()

    def h_server_stop(self, q: Dict[str, str], b: Dict[str, Any],
                      server_id: str) -> Dict[str, Any]:
        inst = self._load_instance_or_404(server_id)
        st = self._instance_state(server_id)
        if st and st.get("mode") == "attach":
            self._clear_instance_state(server_id)  # never kill a foreign server
            stopped = False
        else:
            stopped = self._stop_instance(server_id)
        _ = inst
        self.bus.publish("serve.state", {"event": "stopped", "id": server_id,
                                         "stopped": stopped})
        return {"ok": True, "stopped": stopped}

    # ------------------------------------------------------------- hardware

    # ------------------------------------------------- inference-engine registry
    # Opt-in registry (serving/engines.py): register extra local
    # OpenAI-compatible engines (vLLM, llama-swap, …) so named instances — and
    # the default server via serve.default_engine — can be spawned by them.
    # The built-in llama.cpp engine is synthesized and immutable.

    def _engine_row(self, eng: Dict[str, Any]) -> Dict[str, Any]:
        from ..serving import engines as engines_mod

        row = dict(eng)
        try:
            row["installed"] = bool(engines_mod.probe(eng)["found"])
        except Exception:  # defensive: probe must never break listing
            row["installed"] = False
        row["in_use_by"] = (engines_mod.engines_in_use(self.home, str(eng["id"]))
                            if not eng.get("builtin") else [])
        return row

    def h_engines_list(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        from ..serving import engines as engines_mod

        rows = [self._engine_row(e) for e in engines_mod.list_engines(self.home)]
        default_engine = str((self.cfg.serve or {}).get("default_engine")
                             or "").strip() or None
        return {"engines": rows, "default_engine": default_engine}

    def h_engines_create(self, q: Dict[str, str], b: Dict[str, Any]) -> ApiResponse:
        from ..serving import engines as engines_mod
        from ..serving.supervisor import ServeError as _ServeErr

        try:
            eng = engines_mod.validate_engine_body(b)
        except _ServeErr as e:
            raise HttpError(400, str(e)) from e
        taken = [str(e.get("id")) for e in engines_mod.list_engines(self.home)]
        eng["id"] = engines_mod.new_engine_id(eng["name"], taken)
        engines_mod.save_engine(self.home, eng)
        self.bus.publish("engines.changed", {"event": "created",
                                             "id": eng["id"]})
        return ApiResponse(201, {"ok": True, "engine": self._engine_row(eng)})

    def h_engine_put(self, q: Dict[str, str], b: Dict[str, Any],
                     engine_id: str) -> Dict[str, Any]:
        from ..serving import engines as engines_mod
        from ..serving.supervisor import ServeError as _ServeErr

        existing = engines_mod.get_engine(self.home, engine_id)
        if existing is None:
            raise HttpError(404, f"no engine {engine_id}", code="unknown_engine")
        if existing.get("builtin"):
            raise HttpError(400, "the built-in llama.cpp engine cannot be "
                                 "modified", code="reserved_engine")
        b = dict(b)
        b["id"] = engine_id
        try:
            eng = engines_mod.validate_engine_body(b, existing=existing)
        except _ServeErr as e:
            raise HttpError(400, str(e)) from e
        eng["id"] = engine_id
        engines_mod.save_engine(self.home, eng)
        self.bus.publish("engines.changed", {"event": "updated",
                                             "id": engine_id})
        return {"ok": True, "engine": self._engine_row(eng)}

    def h_engine_delete(self, q: Dict[str, str], b: Dict[str, Any],
                        engine_id: str) -> Dict[str, Any]:
        from ..serving import engines as engines_mod

        if engine_id == engines_mod.BUILTIN_ID:
            raise HttpError(400, "the built-in llama.cpp engine cannot be "
                                 "deleted", code="reserved_engine")
        if engines_mod.get_engine(self.home, engine_id) is None:
            raise HttpError(404, f"no engine {engine_id}", code="unknown_engine")
        used = engines_mod.engines_in_use(self.home, engine_id)
        if used:
            raise HttpError(409, f"engine is used by instance(s): "
                                 f"{', '.join(used)}", code="in_use")
        engines_mod.delete_engine_file(self.home, engine_id)
        removed_build = False
        # A sable-managed llama.cpp backend build (llama-cpp-<backend>):
        # deleting the engine also removes its dedicated checkout under
        # SABLE_HOME so the two views can never disagree.
        if engine_id.startswith("llama-cpp-"):
            try:
                from ..serving import runtime as _rt
                removed_build = bool(_rt.uninstall_llama_cpp(
                    backend=engine_id[len("llama-cpp-"):]))
            except Exception:  # defensive: registration already deleted
                removed_build = False
        self.bus.publish("engines.changed", {"event": "deleted",
                                             "id": engine_id})
        return {"ok": True, "removed_build": removed_build}

    # ------------------------------------------- installable engine catalog

    def h_engine_catalog(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        """Installable-engine catalog with per-recipe install status + host
        compatibility verdicts. stat() probes only; the GPU probe behind
        detect_host() is process-cached, so this stays cheap to poll."""
        from ..serving import engine_installs as ei

        host = ei.detect_host()
        rows = []
        for row in ei.list_catalog(self.home):
            rid = str(row.get("id"))
            run = self._engine_installs.get(rid) or {}
            row["installing"] = run.get("state") == "running"
            try:
                row["compat"] = ei.recipe_compat(rid, host)
            except Exception as e:  # defensive: verdict is display-only
                row["compat"] = {"ok": True, "blocked": None,
                                 "warnings": [f"compat check failed: {e}"],
                                 "variant": None, "backend": None}
            rows.append(row)
        return {"catalog": rows, "host": host}

    def h_engine_install_start(self, q: Dict[str, str], b: Dict[str, Any],
                               recipe_id: str) -> Dict[str, Any]:
        """Start (or re-run/update) a catalog engine install in the background.

        Body: {ref?: str, update?: bool, register?: bool (default True)}.
        Progress streams as `engine.install` SSE events + GET the same path.
        On success the engine is registered in the engines registry too, so it
        shows up in the Inference-engines card / Serve page immediately.
        """
        from ..serving import engine_installs as ei

        try:
            recipe = ei.get_recipe(recipe_id)
        except KeyError:
            raise HttpError(404, f"unknown engine recipe {recipe_id!r}",
                            code="unknown_recipe") from None

        ref = (b.get("ref") or "").strip() or None
        do_update = bool(b.get("update"))
        do_register = b.get("register") is not False  # default True
        backend = (b.get("backend") or "").strip() or None
        force = bool(b.get("force"))
        name = str(recipe["name"])

        # Fail fast on a known-incompatible host (e.g. vLLM on Windows) so the
        # user sees the reason instead of a doomed pip/build in the log. The
        # install itself re-checks; force skips both gates.
        if not force:
            try:
                compat = ei.recipe_compat(recipe_id)
            except Exception:  # defensive: never block on a broken probe
                compat = {"ok": True}
            if not compat.get("ok"):
                raise HttpError(400, str(compat.get("blocked")
                                         or "incompatible host"),
                                code="incompatible_host")

        with self._engine_install_lock:
            cur = self._engine_installs.get(recipe_id)
            if cur and cur.get("state") == "running":
                raise HttpError(409, f"an install of {name} is already running",
                                code="already_running")
            record: Dict[str, Any] = {
                "recipe": recipe_id, "state": "running", "ref": ref,
                "started": _utcnow(), "finished": None, "log": [],
                "error": None, "state_file": None,
            }
            self._engine_installs[recipe_id] = record

        def _log(line: str) -> None:
            record["log"].append(str(line))
            if len(record["log"]) > 500:
                del record["log"][: len(record["log"]) - 500]
            self.bus.publish("engine.install", {
                "recipe": recipe_id, "state": "running",
                "line": str(line), "lines": record["log"][-5:],
            })

        def _run() -> None:
            try:
                st = ei.install_recipe(recipe_id, home=self.home, ref=ref,
                                       update=do_update, backend=backend,
                                       force=force, log_fn=_log)
                record["state_file"] = st
                if do_register:
                    self._register_installed_engine(recipe_id, st)
                record["state"] = "done"
            except ei.InstallError as e:
                record["state"] = "error"
                record["error"] = str(e)
            except Exception as e:
                record["state"] = "error"
                record["error"] = f"{type(e).__name__}: {e}"
            finally:
                record["finished"] = _utcnow()
                self.bus.publish("engine.install", {
                    "recipe": recipe_id, "state": record["state"],
                    "error": record["error"], "lines": record["log"][-5:],
                })

        threading.Thread(target=_run, daemon=True).start()
        return {"ok": True, "state": "running",
                "note": f"{name} install running in background; watch "
                        "engine.install events or poll the same URL"}

    def _register_installed_engine(self, recipe_id: str,
                                   state: Dict[str, Any]) -> None:
        """Create-or-refresh the engines-registry record for an install. The
        registry entry keeps the same id as the recipe so instances can bind
        to it; user edits of other fields survive (only binary/args/kind/notes
        track the install)."""
        from ..serving import engine_installs as ei
        from ..serving import engines as engines_mod

        body = ei.engine_body_for(recipe_id, state)
        existing = engines_mod.get_engine(self.home, recipe_id)
        if existing is not None and not existing.get("builtin"):
            merged = dict(existing)
            user_notes = str(existing.get("notes") or "").strip()
            merged.update(body)
            if user_notes:  # re-install refreshes binary/args, keeps notes
                merged["notes"] = user_notes
            merged["id"] = recipe_id
            eng = engines_mod.validate_engine_body(merged, existing=existing)
            eng["id"] = recipe_id
        else:
            eng = engines_mod.validate_engine_body(body)
            taken = [str(e.get("id")) for e in engines_mod.list_engines(self.home)]
            # Stable id: the recipe id when free, else the usual dedup.
            eng["id"] = recipe_id if recipe_id not in taken \
                else engines_mod.new_engine_id(body["name"], taken)
        engines_mod.save_engine(self.home, eng)
        self.bus.publish("engines.changed", {"event": "installed",
                                             "id": eng["id"]})

    def h_engine_install_status(self, q: Dict[str, str], b: Dict[str, Any],
                                recipe_id: str) -> Dict[str, Any]:
        from ..serving import engine_installs as ei

        try:
            ei.get_recipe(recipe_id)
        except KeyError:
            raise HttpError(404, f"unknown engine recipe {recipe_id!r}",
                            code="unknown_recipe") from None
        rec = self._engine_installs.get(recipe_id)
        try:
            tail = max(1, min(int(q.get("tail") or 40), 500))
        except (TypeError, ValueError):
            tail = 40
        if not rec:
            return {"state": "idle",
                    "installed": ei.recipe_installed(self.home, recipe_id)}
        return {**rec, "log": rec["log"][-tail:],
                "installed": ei.recipe_installed(self.home, recipe_id)}

    def _validate_engine_ref(self, inst: Dict[str, Any]) -> None:
        """Reject instance bodies binding an unregistered engine id."""
        from ..serving import engines as engines_mod

        ref = str(inst.get("engine") or "").strip()
        if ref and not engines_mod.is_builtin_ref(ref) and \
                engines_mod.get_engine(self.home, ref) is None:
            raise HttpError(400, f"unknown inference engine {ref!r} — "
                                 "register it on the Hardware page first",
                            code="unknown_engine")

    def h_hardware(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        return self.hardware.snapshot()

    # ------------------------------------------------------- runtime (llama.cpp)

    _BACKENDS = ("cuda", "rocm", "metal", "vulkan", "cpu")

    def h_runtime_get(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        """GPU + llama.cpp install state for the Hardware panel runtime card.

        The remote-commit check is a network probe; skip with ?check=0 (the
        panel polls, so only the explicit user refresh pays for it).
        """
        from ..serving import runtime

        info = runtime.detect_runtime()
        out: Dict[str, Any] = {
            "system": info.system,
            "machine": info.machine,
            "backend": info.backend,
            "backend_notes": info.backend_notes,
            "gpus": [vars(g) for g in info.gpus],
            "offload_ngl": runtime.gpu_layers(self.cfg.gpu_layers, info),
            "configured": {
                "repo": self.cfg.llama_repo or runtime.LLAMA_CPP_REPO,
                "ref": self.cfg.llama_ref or runtime.LLAMA_CPP_REF,
                "backend": self.cfg.llama_backend,
                "auto_setup": runtime.auto_setup_enabled(self.cfg),
            },
            "install": runtime.read_install_state(),
            # every managed install (primary + per-backend slots like
            # llama.cpp-vulkan) so the panel can show/label them individually
            "installs": runtime.list_installs(),
            "backends": list(self._BACKENDS),
        }
        if q.get("check", "1") != "0":
            try:
                out["update"] = runtime.check_llama_update(
                    self.cfg, backend=q.get("backend") or None)
            except Exception as e:  # network probes must never break the card
                out["update"] = {"installed": False, "update_available": False,
                                 "reason": f"check failed: {e}"}
        return out

    def h_runtime_setup(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        """Start (or restart) the managed llama.cpp build in a background thread.

        Body: {backend?: cuda|rocm|metal|vulkan|cpu, repo?, ref?, update?}.
        Progress streams as `runtime.setup` SSE events + GET /api/runtime/setup.
        """
        from ..serving import runtime

        backend = b.get("backend") or None
        if backend == "auto":
            backend = None
        if backend is not None and backend not in self._BACKENDS:
            raise HttpError(400, f"unknown backend {backend!r}", code="bad_request")
        repo = (b.get("repo") or "").strip() or None
        ref = (b.get("ref") or "").strip() or None
        do_update = bool(b.get("update"))

        with self._runtime_lock:
            if self._runtime_setup and self._runtime_setup.get("state") == "running":
                raise HttpError(409, "a llama.cpp setup is already running",
                                code="already_running")
            record = {
                "state": "running", "backend": backend, "repo": repo, "ref": ref,
                "started": _utcnow(), "finished": None, "log": [], "error": None,
            }
            self._runtime_setup = record

        def _log(line: str) -> None:
            record["log"].append(str(line))
            if len(record["log"]) > 500:
                del record["log"][: len(record["log"]) - 500]
            self.bus.publish("runtime.setup", {
                "state": "running", "line": str(line),
                "lines": record["log"][-5:],
            })

        def _run() -> None:
            try:
                runtime.setup_llama_cpp(backend=backend, log_fn=_log,
                                        repo=repo, ref=ref, update=do_update)
                record["state"] = "done"
                if backend:
                    # Each backend install is also a first-class engine entry
                    # (llama-cpp-<backend>) so Vulkan/ROCm builds appear side by
                    # side in the Inference-engines card with update/delete.
                    try:
                        self._register_llama_backend_engine(backend)
                    except Exception as e:  # registration must not fail a build
                        _log(f"warning: engine registration failed: {e}")
            except Exception as e:
                record["state"] = "error"
                record["error"] = f"{type(e).__name__}: {e}"
            finally:
                record["finished"] = _utcnow()
                self.bus.publish("runtime.setup", {
                    "state": record["state"],
                    "error": record["error"],
                    "lines": record["log"][-5:],
                })

        threading.Thread(target=_run, daemon=True).start()
        return {"ok": True, "state": "running",
                "note": "llama.cpp build running in background; watch "
                        "runtime.setup events or poll GET /api/runtime/setup"}

    def h_runtime_setup_status(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        rec = self._runtime_setup
        if not rec:
            return {"state": "idle"}
        tail = int(q.get("tail", "40"))
        return {**rec, "log": rec["log"][-tail:]}

    def _register_llama_backend_engine(self, backend: str) -> None:
        """Create-or-refresh the engines-registry record for one managed
        llama.cpp backend build (id ``llama-cpp-<backend>``). The binary is the
        built llama-server path; an empty args_template means sable builds the
        full argv like it does for the built-in engine. Refuses to touch the
        built-in slot when the build *is* the primary auto-setup install — that
        one stays represented by the immutable built-in engine row."""
        from ..serving import engines as engines_mod
        from ..serving import runtime

        root, state_be = runtime.install_root_for(backend)
        st = runtime.read_install_state(backend=state_be)
        if not st:
            return
        if state_be is None:
            return  # primary slot == the built-in engine row (immutable);
                    # a duplicate entry could delete the auto-setup install
        exe = runtime.llama_server_path(root)
        if exe is None:
            return  # nothing built yet — nothing to register
        eng_id = f"llama-cpp-{str(backend).lower()}"
        taken = [str(e.get("id")) for e in engines_mod.list_engines(self.home)]
        existing = engines_mod.get_engine(self.home, eng_id)
        body = {
            "name": f"llama.cpp ({str(backend).lower()})",
            "kind": "openai",
            "binary": str(exe),
            # default llama-server argv (--model/--host/--port); richer
            # run_settings translation stays a built-in-engine feature
            "args_template": [],
            "notes": (f"sable-managed llama.cpp {str(backend).lower()} build "
                      f"({st.get('repo', '?')} @ {st.get('ref', '?')} "
                      f"commit {st.get('commit', '?')})"),
        }
        if existing is not None:
            merged = dict(existing)
            merged.update(body)
            merged["id"] = eng_id
            eng = engines_mod.validate_engine_body(merged, existing=existing)
            eng["id"] = eng_id
        else:
            eng = engines_mod.validate_engine_body(body)
            eng["id"] = eng_id if eng_id not in taken \
                else engines_mod.new_engine_id(body["name"], taken)
        engines_mod.save_engine(self.home, eng)
        self.bus.publish("engines.changed", {"event": "installed",
                                             "id": eng["id"]})

    def h_runtime_sync(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        """Register every managed llama.cpp backend build as an engine entry.
        Idempotent; covers installs that predate per-backend registration."""
        from ..serving import runtime

        synced = []
        for rec in runtime.list_installs():
            be = str(rec.get("backend") or "").strip()
            if not be:
                continue
            try:
                self._register_llama_backend_engine(be)
                synced.append(be)
            except Exception:  # defensive: one bad slot must not break sync
                continue
        return {"ok": True, "synced": synced}

    # ------------------------------------------------- tuning (doctor/bench/calibrate)

    def h_doctor_env(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        """Latest doctor report WITHOUT running anything (panel bootstrap).

        Returns {state: idle|running|done|error, ...}. The checks only run
        when the user clicks Run — never on page open — because vendor probes
        (rocm-smi under GPU load) can block in the kernel past their
        subprocess timeout and must not stall either the browser or a server
        worker thread."""
        rec = self._doctor_run
        if not rec:
            return {"state": "idle", "calibration_mode":
                    self._calibration_mode()}
        return {**rec, "calibration_mode": self._calibration_mode()}

    def _start_doctor(self, fix: bool) -> Dict[str, Any]:
        """Kick off collect_checks in a background thread (one at a time)."""
        with self._doctor_lock:
            if (self._doctor_run or {}).get("state") == "running":
                raise HttpError(409, "a doctor run is already in progress",
                                code="already_running")
            record: Dict[str, Any] = {
                "state": "running", "fix": fix, "started": _utcnow(),
                "finished": None, "error": None, "checks": [],
                "exit_code": None,
            }
            self._doctor_run = record

        def _run() -> None:
            from dataclasses import asdict

            try:
                from ..doctor import collect_checks

                results, rc = collect_checks(fix=fix, out=lambda _s: None)
                record["checks"] = [asdict(r) for r in results]
                record["exit_code"] = rc
                record["state"] = "done"
            except Exception as e:  # a hung probe surfaces, never hangs us
                record["state"] = "error"
                record["error"] = f"{type(e).__name__}: {e}"
            finally:
                record["finished"] = _utcnow()
                self.bus.publish("doctor.run", {
                    "state": record["state"], "error": record["error"],
                    "exit_code": record["exit_code"],
                })

        threading.Thread(target=_run, daemon=True).start()
        return {"ok": True, "state": "running"}

    def h_doctor_run(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        """POST /api/doctor/run — start the WP7 checks (user action only)."""
        return self._start_doctor(fix=bool(b.get("fix")))

    def h_doctor_env_fix(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        """POST /api/doctor/env — legacy alias of /api/doctor/run with
        fix=True (removes stale instance state files whose pid is verified
        gone; foreign pids are never touched — ownership rule lives in the
        doctor/serving code)."""
        return self._start_doctor(fix=bool(b.get("fix", True)))

    def _calibration_mode(self) -> str:
        from ..serving.calibrate import calibration_mode

        try:
            return calibration_mode(self.cfg)
        except Exception:  # defensive: display-only
            return "off"

    def h_bench_list(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        """Persisted bench records (newest first) + the active noise floor."""
        from ..serving import bench as bench_mod

        recs = []
        for r in bench_mod.load_results():
            gen = (r.get("generation") or {}).get("median")
            recs.append({
                "timestamp": r.get("timestamp"),
                "kind": r.get("kind"),
                "label": r.get("label") or "",
                "model": r.get("model"),
                "gen_median_tps": gen,
                "gen_spread_pct": (r.get("generation") or {}).get("spread_pct"),
                "token_identity": (r.get("token_identity") or {}).get("status"),
                "path": r.get("_path"),
            })
        nf = bench_mod.load_noise_floor()
        return {
            "records": recs,
            "noise_floor": {
                "spread_pct": (nf or {}).get("generation", {}).get("spread_pct")
                if nf else None,
                "label": (nf or {}).get("label"),
            } if nf else None,
        }

    def h_bench_run(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        """Start a bench run in the background (WP1 semantics identical to
        `sable bench`: temperature 0, token-identity check, persisted).

        Body: {runs?: int, max_tokens?: int, label?: str, noise_floor?: bool}.
        Progress: SSE `bench.run` events + GET /api/bench/run. A running
        calibration refuses the GPU (409) and vice versa."""
        from ..serving import bench as bench_mod

        runs = b.get("runs") or bench_mod.DEFAULT_RUNS
        try:
            runs = max(1, min(int(runs), 50))
        except (TypeError, ValueError):
            raise HttpError(400, "runs must be an integer") from None
        max_tokens = b.get("max_tokens") or bench_mod.DEFAULT_MAX_TOKENS
        try:
            max_tokens = max(1, min(int(max_tokens), 4096))
        except (TypeError, ValueError):
            raise HttpError(400, "max_tokens must be an integer") from None
        label = str(b.get("label") or "")[:80]
        noise_floor = bool(b.get("noise_floor"))
        # Multi-instance aware: bench a RUNNING server instance by id (spawn
        # mode attaches to its endpoint — never spawns or stops anything it
        # does not own). Without server_id: legacy behavior (resolve from
        # config, spawn the default if needed).
        server_id = str(b.get("server_id") or "").strip() or None

        target = None
        if server_id:
            inst = next((i for i in srv_inst.load_instances(self.home,
                                                            cfg=self.cfg)
                         if str(i.get("id")) == server_id), None)
            if inst is None:
                raise HttpError(404, f"no such server instance: {server_id}",
                                code="not_found")
            row = self._instance_row(inst)
            if not row.get("running") or not row.get("base_url"):
                raise HttpError(
                    409, f"server instance {server_id} is not running — start "
                         "it on the Serve panel first", code="not_running")
            target = {"base_url": str(row["base_url"]),
                      "model": str(row.get("model_name") or "instance-model")}

        if (self._calibrate_run or {}).get("state") == "running":
            raise HttpError(409, "a calibration sweep is running",
                            code="already_running")
        with self._bench_lock:
            if (self._bench_run or {}).get("state") == "running":
                raise HttpError(409, "a bench run is already in progress",
                                code="already_running")
            record = {
                "state": "running", "kind": "noise_floor" if noise_floor
                else "bench", "label": label, "runs": runs,
                "server_id": server_id,
                "started": _utcnow(), "finished": None, "log": [],
                "error": None, "result": None,
            }
            self._bench_run = record

        def _log(line: str) -> None:
            record["log"].append(str(line))
            if len(record["log"]) > 200:
                del record["log"][: len(record["log"]) - 200]
            self.bus.publish("bench.run", {
                "state": "running", "line": str(line),
                "lines": record["log"][-5:],
            })

        def _run() -> None:
            supervisor = None
            try:
                if target is not None:
                    base_url, model = (target["base_url"],
                                       target["model"])
                else:
                    from ..cli import _resolve_endpoint
                    from ..core.config import load_config as _load

                    cfg = _load(None)
                    base_url, model, supervisor = _resolve_endpoint(
                        cfg, spawn_if_needed=True)
                _log(f"endpoint {base_url} model={model}")
                _log(f"measuring {runs} run(s), max_tokens={max_tokens}…")
                res = bench_mod.run_bench(
                    base_url, model, runs=runs, max_tokens=max_tokens,
                    label=label,
                    kind=(bench_mod.KIND_NOISE_FLOOR if noise_floor
                          else bench_mod.KIND_BENCH))
                saved = bench_mod.save_result(res)
                record["result"] = {
                    "generation": res.get("generation"),
                    "prompt_processing": res.get("prompt_processing"),
                    "token_identity": res.get("token_identity"),
                    "model": res.get("model"),
                    "path": str(saved),
                }
                ti = (res.get("token_identity") or {}).get("status")
                record["state"] = "done" if ti == "ok" else "diverged"
            except Exception as e:
                record["state"] = "error"
                record["error"] = f"{type(e).__name__}: {e}"
            finally:
                if supervisor is not None:
                    try:
                        supervisor.stop()
                    except Exception:  # noqa: BLE001 - best-effort cleanup
                        pass
                record["finished"] = _utcnow()
                self.bus.publish("bench.run", {
                    "state": record["state"], "error": record["error"],
                    "result": record["result"],
                })

        threading.Thread(target=_run, daemon=True).start()
        return {"ok": True, "state": "running"}

    def h_bench_targets(self, q: Dict[str, str],
                        b: Dict[str, Any]) -> Dict[str, Any]:
        """Benchable targets for the Tuning panel: running server instances
        (any agent may be assigned to them; bench just measures the endpoint)
        + whether a default spawn is possible (gguf_path configured)."""
        rows = []
        try:
            insts = srv_inst.load_instances(self.home, cfg=self.cfg)
        except Exception:  # defensive: panel must still render
            insts = []
        for inst in insts:
            row = self._instance_row(inst)
            if not isinstance(row, dict):
                continue
            rows.append({
                "id": row.get("id"), "name": row.get("name"),
                "running": bool(row.get("running")),
                "model_name": row.get("model_name"),
                "base_url": row.get("base_url") if row.get("running") else None,
            })
        return {"instances": rows,
                "can_spawn_default": bool(self.cfg.gguf_path)}

    def h_bench_run_status(self, q: Dict[str, str],
                           b: Dict[str, Any]) -> Dict[str, Any]:
        rec = self._bench_run
        if not rec:
            return {"state": "idle"}
        tail = int(q.get("tail", "40"))
        return {**rec, "log": rec["log"][-tail:]}

    def _calibrate_model_ref(self) -> Optional[str]:
        """GGUF to calibrate: configured gguf_path when readable (same rule
        as the CLI's _calibrate_model_ref without the --model override)."""
        from ..cli import _calibrate_model_ref as cli_ref

        try:
            return cli_ref(self.cfg, None)
        except Exception:  # noqa: BLE001 - display-only resolution
            return None

    def h_calibrate_show(self, q: Dict[str, str],
                         b: Dict[str, Any]) -> Dict[str, Any]:
        """Profiles for this machine (newest first) + the active one for the
        configured model + serve.calibration mode."""
        from ..serving import calibrate as cal_mod

        profs = cal_mod.list_profiles()
        cur_mid = cal_mod.machine_id()
        gguf = self._calibrate_model_ref()
        mhash = cal_mod.model_hash(gguf) if gguf else None
        active_path = None
        if mhash:
            active = cal_mod.load_profile(cur_mid, mhash)
            if active:
                active_path = str(active.get("_path"))
        out = []
        for p in profs:
            out.append({
                "timestamp": p.get("timestamp"),
                "machine_id": p.get("machine_id"),
                "active_machine": p.get("machine_id") == cur_mid,
                "model_hash": (p.get("model_hash") or "")[:12],
                "model_path": p.get("model_path"),
                "adopted_settings": p.get("adopted_settings") or {},
                "threshold_pct": p.get("threshold_pct"),
                "noise_floor_pct": (p.get("noise_floor") or {}).get(
                    "spread_pct"),
                "is_active_profile": str(p.get("_path")) == active_path,
                "path": p.get("_path"),
            })
        # The sweep spawns its OWN llama-server per candidate; anything else
        # holding the GPU skews every measurement. Surface them for the UI.
        busy = []
        try:
            for inst in srv_inst.load_instances(self.home, cfg=self.cfg):
                row = self._instance_row(inst)
                if isinstance(row, dict) and row.get("running"):
                    busy.append({"id": row.get("id"), "name": row.get("name")})
        except Exception:  # defensive: display-only
            pass
        return {"profiles": out, "calibration_mode": self._calibration_mode(),
                "model": gguf, "running_instances": busy}

    def h_calibrate_run(self, q: Dict[str, str],
                        b: Dict[str, Any]) -> Dict[str, Any]:
        """Start a WP3 autotune sweep in the background (identical adoption
        rules to `sable calibrate`: gain > max(3%, noise floor) AND token
        identity). Body: {quick?: bool, runs?: int, knobs?: "a,b"}.

        Progress: SSE `calibrate.run` events + GET /api/calibrate/run."""
        from ..serving import calibrate as cal_mod

        quick = bool(b.get("quick", True))
        runs = b.get("runs") or cal_mod.DEFAULT_SWEEP_RUNS
        try:
            runs = max(1, min(int(runs), 20))
        except (TypeError, ValueError):
            raise HttpError(400, "runs must be an integer") from None
        knobs = None
        if b.get("knobs"):
            knobs = [k.strip() for k in str(b["knobs"]).split(",") if k.strip()]

        gguf = self._calibrate_model_ref()
        if not gguf:
            raise HttpError(
                400, "no GGUF to calibrate — set gguf_path (Settings or "
                     "config) or download a model first", code="no_model")
        if (self._bench_run or {}).get("state") == "running":
            raise HttpError(409, "a bench run is in progress",
                            code="already_running")
        with self._calibrate_lock:
            if (self._calibrate_run or {}).get("state") == "running":
                raise HttpError(409, "a calibration sweep is already running",
                                code="already_running")
            record = {
                "state": "running", "quick": quick, "runs": runs,
                "model": gguf, "started": _utcnow(), "finished": None,
                "log": [], "error": None, "result": None,
            }
            self._calibrate_run = record

        def _log(line: str) -> None:
            record["log"].append(str(line))
            if len(record["log"]) > 400:
                del record["log"][: len(record["log"]) - 400]
            self.bus.publish("calibrate.run", {
                "state": "running", "line": str(line),
                "lines": record["log"][-5:],
            })

        def _run() -> None:
            try:
                profile = cal_mod.run_sweep(
                    self.cfg, gguf_path=gguf, runs=runs, knobs=knobs,
                    quick=quick, log=_log)
                record["result"] = {
                    "adopted_settings": profile.get("adopted_settings") or {},
                    "threshold_pct": profile.get("threshold_pct"),
                    "noise_floor_pct": (profile.get("noise_floor") or {}).get(
                        "spread_pct"),
                    "knobs": profile.get("knobs"),
                }
                record["state"] = "done"
            except Exception as e:
                record["state"] = "error"
                record["error"] = f"{type(e).__name__}: {e}"
            finally:
                record["finished"] = _utcnow()
                self.bus.publish("calibrate.run", {
                    "state": record["state"], "error": record["error"],
                    "adopted": (record.get("result") or {}).get(
                        "adopted_settings"),
                })

        threading.Thread(target=_run, daemon=True).start()
        return {"ok": True, "state": "running", "model": gguf}

    def h_calibrate_run_status(self, q: Dict[str, str],
                               b: Dict[str, Any]) -> Dict[str, Any]:
        rec = self._calibrate_run
        if not rec:
            return {"state": "idle"}
        tail = int(q.get("tail", "40"))
        return {**rec, "log": rec["log"][-tail:]}

    def h_calibrate_clear(self, q: Dict[str, str],
                          b: Dict[str, Any]) -> Dict[str, Any]:
        """Remove stored calibration profiles (same as `calibrate --clear`)."""
        from ..serving import calibrate as cal_mod

        if (self._calibrate_run or {}).get("state") == "running":
            raise HttpError(409, "a calibration sweep is running",
                            code="already_running")
        n = cal_mod.clear_profiles()
        self.bus.publish("app.state", {"what": "calibrate.cleared",
                                       "removed": n})
        return {"ok": True, "removed": n}

    # --------------------------------------------------------------- config

    _CONFIG_PATH_SECRETS = ("token", "attach_api_key")

    def _config_path(self) -> Path:
        env = os.environ.get("SABLE_CONFIG", "").strip()
        if env:
            return Path(env).expanduser()
        return self.home / "sable.yaml"

    def h_config_get(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        import dataclasses

        cfg_dict = dataclasses.asdict(self.cfg)
        raw = ""
        path = self._config_path()
        if path.is_file():
            try:
                raw = path.read_text(encoding="utf-8")
            except OSError:
                raw = ""
        return {
            "config": _redact(cfg_dict),
            "raw": raw,
            "path": str(path),
            "token_configured": bool(self.token),
        }

    def h_config_put(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        from ..training.scheduler import TrainingConfig as _TC
        from ..training.validate import ValidationConfig as _VC

        known_sections = {
            "training": set(_TC.__dataclass_fields__),
            "curation": set(CurationConfig.__dataclass_fields__),
            "validation": set(_VC.__dataclass_fields__),
            "dashboard": {"port", "token"},
            # Golden mining (curate/mine_goldens.py).
            "goldens_mine": {"enabled", "max_per_run", "max_candidates",
                             "min_feedback_score", "use_judge"},
            # Playbooks (core/playbooks.py) — injection switch + caps. Without
            # this entry, section patches to playbooks: skip field validation
            # entirely (typos land silently while goldens_mine typos 400).
            "playbooks": {"enabled", "total_cap_chars",
                          "per_recipe_cap_chars"},
            # Capability-tier auto-routing (serving/router.py) — review idea #2.
            "routing": {"enabled", "escalate_on_failure", "prefer",
                        "high_context_tokens", "tool_heavy_min_tools",
                        "profile_hints", "refusal_markers"},
            # Built-in toolset policy (backlog #1/#10): raw `tools:` section.
            # `lsp.servers` (per-extension overrides) is raw-YAML territory —
            # the flat LSP keys below cover the common case.
            "tools": {"block_sensitive_files", "run_command_argv_only",
                      "web_fetch", "spill_enabled", "spill_threshold_chars",
                      "spill_threshold_tokens", "turn_result_budget_chars",
                      "spill_retention_days", "lint_on_write",
                      "lsp"},
            # Prompt evolution (pass 4 #39) master switch + call budget.
            "evolution": {"enabled", "max_call_budget"},
            # Delegation (feature-wave Phase 2): flat fields populated from
            # the `delegation:` section by the loader.
            "delegation": {"enabled", "timeout_seconds"},
            # Agent runtime (core/agent_runner.py): raw `agent:` section.
            # Keys the runner reads; `capabilities`/`vision` are nested
            # dicts/lists handled opaquely here (Profiles owns per-profile).
            "agent": {"context_token_budget", "keep_recent_tokens",
                      "workspace_root", "heal_tool_calls", "verify_on_stop",
                      "max_iterations", "capabilities", "vision"},
            # Disk janitor (backlog #16): raw `janitor:` section.
            "janitor": {"enabled", "retention_days", "disk_retention_days"},
            # Rejection-sampling fine-tuning (training/rejection.py): raw
            # `rft:` section.
            "rft": {"enabled", "sample_k", "judge_min_score", "max_prompts",
                    "temperature", "max_tokens", "max_prompt_tokens", "emit_dpo",
                    "max_dpo_pairs_per_prompt", "agent_profile", "project",
                    "base_model"},
            # Logging: flat keys the logging handler accepts (core/config.py
            # carries the section raw).
            "logging": {"level", "file", "max_bytes", "backup_count",
                        "module_levels"},
            # redact_secrets was loadable but un-patchable through the
            # dashboard validator (latent gap); retention_days is the
            # `sable traces prune` default window.
            "recording": {"enabled", "redact_secrets", "retention_days"},
            # Persistent skills (core/skills.py) — agent-authored procedures;
            # this section toggles exposure + index cap. The skill list
            # itself lives under /api/skills (Settings panel Skills card).
            "skills": {"enabled", "index_total_cap_chars"},
            # Easy training (turnkey loop) — see training/easy.py.
            # `profiles` is the multi-agent list; `profile` stays the primary.
            "easy_training": {"enabled", "profile", "profiles", "auto_deploy",
                              "min_traces", "min_score"},
            "run_settings": {
                "kv_cache_type", "flash_attn", "parallel", "batch_size",
                "ubatch_size", "tensor_parallel", "no_mmap", "mlock",
                "mmproj_path", "vision", "speculative_type",
                "spec_draft_n_max", "mtp_path", "gpu_memory_pct",
                "extra_args",
            },
            "sampling": {
                "temperature", "top_p", "top_k", "min_p", "repeat_penalty",
                "presence_penalty", "frequency_penalty", "penalty_free_repeat",
                "max_tokens",
                # Thinking / reasoning controls (reference style): sent
                # with every request; llama.cpp applies them through
                # chat_format_template_controls (serving/run_args).
                "enable_thinking", "reasoning_effort", "preserve_thinking",
            },
            "indexing": {
                # Phase 5 schema
                "enabled", "model", "output_dim", "search_mode",
                "paths_override", "auto_retrieve_threshold",
                # legacy keys stay writable so old configs round-trip;
                # load_config absorbs them (core/config._normalize_indexing)
                "paths", "embedding_model", "embedding_url", "top_k",
            },
            # Serving-launch behavior (WP3 calibration / WP4 watchdog / WP5
            # coexistence). Without this entry the Tuning panel could not
            # toggle serve.calibration through a section patch.
            "serve": {
                "calibration", "watchdog_enabled", "watchdog_window_seconds",
                "watchdog_sample_interval_seconds", "watchdog_cpu_threshold_ms",
                "watchdog_io_threshold_bytes", "watchdog_kill_grace_seconds",
                "watchdog_restart_backoff_base_seconds", "watchdog_max_restarts",
                "watchdog_restart_window_seconds", "idle_unload_seconds",
                "lazy_load", "min_free_vram_mib", "coexist_policy",
                # Inference-engine registry: default server's engine binding
                # (serving/engines.py). Blank/unset = built-in llama.cpp.
                "default_engine",
            },
        }
        # Top-level keys the write path accepts. Superset of the loader's own
        # recognition (core/config.py _NESTED_SECTIONS): delegation/agent/tools
        # and the other sections below are all loadable config, so a section
        # patch must never 400 on a key the loader would happily read.
        known_top = set(SableConfig.__dataclass_fields__) | {
            "recording", "delegation", "agent", "tools", "janitor",
            "training", "curation", "validation", "dashboard",
        }

        if "yaml" in b:
            text = b.get("yaml")
            if not isinstance(text, str):
                raise HttpError(400, "yaml must be a string")
        elif "section" in b:
            # section-patch mode: merge into current raw.
            # 5d: section "root" patches top-level keys (gguf_path, port, ...)
            # so the Settings form can edit serve defaults too.
            section = str(b["section"])
            patch = b.get("patch")
            if not isinstance(patch, dict):
                raise HttpError(400, "patch must be an object")
            path = self._config_path()
            current: Dict[str, Any] = {}
            if path.is_file():
                try:
                    loaded = yaml.safe_load(path.read_text()) or {}
                    current = loaded if isinstance(loaded, dict) else {}
                except yaml.YAMLError:
                    pass
            if section in ("root", "_top"):
                unknown = [k for k in patch if k not in known_top]
                if unknown:
                    raise HttpError(
                        400,
                        "unknown config keys: " + ", ".join(sorted(map(str, unknown))),
                        code="unknown_keys",
                    )
                _merge_patch(current, patch)
            else:
                sec = current.get(section)
                merged = dict(sec) if isinstance(sec, dict) else {}
                _merge_patch(merged, patch)
                current[section] = merged
            text = yaml.safe_dump(current, sort_keys=False)
        else:
            raise HttpError(400, "body must contain yaml or section+patch")

        # ---- validate before touching the file ------------------------------
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as e:
            msg = str(e).replace("\n", " ")
            raise HttpError(400, f"invalid YAML: {msg}") from e
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise HttpError(400, "config must be a YAML mapping")
        unknown_top = [k for k in data if k not in known_top]
        if unknown_top:
            raise HttpError(
                400,
                "unknown config keys: " + ", ".join(sorted(map(str, unknown_top))),
                code="unknown_keys",
            )
        for section, fields in known_sections.items():
            sec = data.get(section)
            if sec is None:
                continue
            if not isinstance(sec, dict):
                raise HttpError(400, f"config section {section!r} must be a mapping")
            unknown = [k for k in sec if k not in fields]
            if unknown:
                raise HttpError(
                    400,
                    f"unknown keys in {section}: " + ", ".join(map(str, sorted(unknown))),
                    code="unknown_keys",
                )

        # ---- write with backup ----------------------------------------------
        path = self._config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file():
            import shutil

            shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
        # pid-suffixed tmp: two concurrent writers must not clobber each
        # other's shared "<file>.tmp" mid-write (lost-update race).
        tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)

        # Reflect the new config in this app instance (env overrides still win).
        try:
            self.cfg = load_config(str(path))
        except Exception:
            pass  # keep serving with the previous cfg; file is valid regardless
        return {"ok": True, "path": str(path), "backup": str(
            path.with_suffix(path.suffix + ".bak")) if (
            path.with_suffix(path.suffix + ".bak")).is_file() else None}

    # ------------------------------------------------------- program control

    def h_shutdown(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        """Close the program: stop the owned llama-server, then exit ourselves.

        Only meaningful for `sable dashboard`/`launch` processes; a graceful
        self-terminate via SIGTERM so the serve_dashboard signal handler runs
        (server socket closed, 'dashboard stopped.' printed). The owned
        llama-server child is stopped first so nothing is orphaned.
        """
        from ..serving.deploy import stop_owned_server

        # Run lifecycle step 3: let in-flight chat runs cancel cleanly BEFORE
        # the serving backends go away, so each runner records its own
        # score-neutralized cancelled trace instead of dying with the process.
        try:
            self.shutdown_chat_runs()
        except Exception:
            pass

        # Phase 6: stop every serving process this process owns — all
        # multi-instance children AND the Phase 5 embed servers. Foreign /
        # attach processes are never touched (registry holds only ours).
        try:
            from ..serving.supervisor import stop_all_servers
            stop_all_servers()
        except Exception:
            pass
        def _die() -> None:
            try:
                stop_owned_server(self.home)
            except Exception:
                pass
            try:
                # Phase 5: stop embedding servers WE spawned (never foreign).
                from ..core.indexer import stop_all_embed_servers
                stop_all_embed_servers()
            except Exception:
                pass
            self.bus.publish("app.state", {"event": "shutdown"})
            time.sleep(0.4)  # let the SSE frame flush + response return
            os.kill(os.getpid(), signal.SIGTERM)

        threading.Thread(target=_die, daemon=True, name="shutdown").start()
        return {"ok": True, "note": "shutting down — stop_owned_server ran; "
                                    "dashboard process exiting"}

    def h_logs(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        """Console log tail: dashboard.log + llama-server.log (program console).

        Every llama-server sable spawns writes its own file: the legacy
        default server to ``llama-server.log``, named instances to
        ``server-<id>.log`` (see h_server_start) and the indexer's embedding
        server to ``embed-server.log``. We enumerate the logs dir so new
        instances show up in the console without any registration step."""
        tail = max(10, min(int(q.get("tail", "400") or 400), 5000))
        logs_dir = self.home / "logs"
        files: Dict[str, Path] = {
            "dashboard": logs_dir / "dashboard.log",
            "llama-server": logs_dir / "llama-server.log",
            "embed-server": logs_dir / "embed-server.log",
        }
        try:
            for p in sorted(logs_dir.glob("server-*.log")):
                name = p.stem  # "server-<instance-id>"
                if name not in files:
                    files[name] = p
        except OSError:
            pass
        out: Dict[str, Any] = {}
        for name, p in files.items():
            lines: list[str] = []
            exists = p.is_file()
            if exists:
                try:
                    with p.open("r", errors="replace") as fh:
                        # cheap tail: read the last ~256 KB
                        size = p.stat().st_size
                        if size > 262_144:
                            fh.seek(size - 262_144)
                            fh.readline()
                        lines = fh.read().splitlines()[-tail:]
                except OSError:
                    exists = False
            out[name] = {"exists": exists, "lines": lines}
        return out

    # ------------------------------------------------------------ codebase index
    # Phase 5: one sqlite DB per project (<home>/index/<project_id>.db) plus
    # a `default` fallback DB. Endpoints accept ?project= / body "project"
    # to route; omitted => default DB (legacy/config-driven behavior).

    def _index(self, project_id: Optional[str] = None):
        from ..core.indexer import get_index

        proj_ws = None
        if project_id:
            proj = self._load_project(project_id)
            if proj is None:
                raise HttpError(404, f"no project {project_id}",
                                code="unknown_project")
            proj_ws = str(proj.get("workspace_dir") or "") or None
        return get_index(self.home, cfg=self.cfg, project_id=project_id,
                         workspace_dir=proj_ws)

    def h_index_status(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        pid = str(q.get("project") or "").strip() or None
        return self._index(pid).status()

    def h_index_build(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        """Build/refresh the codebase index in a background thread.

        Phase 5: `project` (query or body) selects the project DB; omitted
        => default DB. Uses the spawn-managed embedding server — when the
        configured model can't serve, the build fails loudly (index.build
        error event) instead of silently falling back to lexical.

        Progress streams as `index.build` SSE events; result also pollable via
        GET /api/index (building flag).
        """
        pid = (str(b.get("project") or "").strip()
               or str(q.get("project") or "").strip() or None)
        idx = self._index(pid)

        record = {"state": "running", "started": _utcnow()}

        def _log(line: str) -> None:
            self.bus.publish("index.build", {"state": "running", "line": str(line)})

        def _run() -> None:
            try:
                res = idx.build(progress=_log)
                record["state"] = "done"
                record.update(res)
                self.bus.publish("index.build", {"state": "done", **res})
            except Exception as e:
                record["state"] = "error"
                record["error"] = f"{type(e).__name__}: {e}"
                self.bus.publish("index.build", {"state": "error",
                                                 "error": record["error"]})

        threading.Thread(target=_run, daemon=True, name="index-build").start()
        return {"ok": True, "state": "running", "project": pid,
                "note": "index build started; watch index.build events"}

    def h_index_search(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        from ..core.indexer import IndexRebuildRequired

        query = str(b.get("query") or "").strip()
        if not query:
            raise HttpError(400, "query required")
        pid = (str(b.get("project") or "").strip() or None)
        idx = self._index(pid)
        try:
            results = idx.search(query,
                                 top_k=int(b["top_k"]) if b.get("top_k") else None,
                                 mode=str(b.get("mode") or "") or None)
        except IndexRebuildRequired as e:
            # Client-actionable state (dim/model mismatch), not a server bug:
            # 409 + code lets the UI offer "rebuild now" instead of a raw 500.
            raise HttpError(409, str(e), code="index_rebuild_required") from e
        except Exception as e:
            raise HttpError(500, str(e)) from e
        return {"results": results}

    def h_index_stop(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        """Stop the embedding server(s) this process spawned (foreign
        processes are never touched — sable only kills its own children)."""
        from ..core.indexer import stop_all_embed_servers

        stop_all_embed_servers()
        return {"ok": True}

    def startup_indexing(self) -> None:
        """Dashboard startup hook (Phase 5): while indexing is enabled, kick
        a non-blocking refresh of the default DB and every project DB. The
        first embed call spawns the owned embedding server; failures land in
        index.build error events, never blocking or crashing startup."""
        ix = getattr(self.cfg, "indexing", None) or {}
        if not ix.get("enabled"):
            return

        def _run() -> None:

            targets: List[Optional[str]] = [None]
            d = self.home / PROJECTS_DIRNAME
            if d.is_dir():
                for f in sorted(d.glob("*.json")):
                    pid = f.stem
                    if DashboardApp._PROJECT_ID_RE.match(pid):
                        targets.append(pid)
            for pid in targets:
                try:
                    idx = self._index(pid)
                    if not idx.roots or not any(r.is_dir() for r in idx.roots):
                        continue
                    self.bus.publish("index.build", {
                        "state": "running", "project": pid,
                        "line": f"startup index refresh ({pid or 'default'})…"})
                    res = idx.build(progress=lambda line, pid=pid: (
                        self.bus.publish(
                            "index.build", {"state": "running", "project": pid,
                                            "line": str(line)})))
                    self.bus.publish("index.build", {
                        "state": "done", "project": pid, **res})
                except Exception as e:
                    self.bus.publish("index.build", {
                        "state": "error", "project": pid,
                        "error": f"{type(e).__name__}: {e}"})

        threading.Thread(target=_run, daemon=True,
                         name="index-startup").start()

    # ----------------------------------------------------------------- chat

    def _resolve_chat_target(self, server_instance: Optional[str] = None,
                             routing_hint: Optional[str] = None,
                             ) -> Tuple[str, Optional[str], str]:
        """(base_url, api_key, model_name) of the live serving endpoint.

        Phase 6: ``server_instance`` (from an agent profile's binding) targets
        that managed instance instead of the default serve endpoint — e.g. an
        orchestrator profile on one model while a coding profile runs another,
        concurrently. Unset/"default" keeps the legacy behavior (serve_state
        then config attach). The bound instance must be running; unknown ids
        are a clean 404, stopped ones a 503 naming the instance.

        Capability-tier auto-routing (review idea #2, opt-in): with
        ``routing.enabled: true`` AND no explicit instance binding, the run is
        routed per-run to the tier-appropriate running instance
        (serving/router.py). An explicit binding ALWAYS wins over routing —
        this block sits below it for exactly that reason. With routing off
        (the default) the code path is byte-identical to before: one config
        attribute read, no pool build."""
        if server_instance and str(server_instance).strip() not in (
                "", "default", "none"):
            iid = str(server_instance).strip()
            inst = srv_inst.get_instance(self.home, iid, cfg=self.cfg)
            if inst is None:
                raise HttpError(404, f"no server instance {iid}",
                                code="unknown_server")
            st = self._instance_state(iid)
            if st is None or not srv_inst.state_alive(st):
                # WP5 lazy load: an idle/coexist-UNLOADED instance reloads
                # transparently here (blocking; cold-start accepted). With
                # lazy_load off the user gets the explicit guidance message;
                # a never-loaded stopped instance keeps its old 503 verbatim.
                if self.coexist is not None and st is not None \
                        and st.get("unloaded"):
                    if self.coexist.cfg.lazy_load:
                        try:
                            st = self.coexist.ensure_loaded(iid) or st
                        except Exception as e:
                            raise HttpError(503, str(e),
                                            code="lazy_load_failed") from e
                    else:
                        raise HttpError(
                            503, self.coexist.unloaded_guidance(iid)
                            or f"server instance {iid} is unloaded",
                            code="instance_unloaded") from None
                if st is None or not srv_inst.state_alive(st):
                    raise HttpError(503, f"server instance {iid} is not "
                                         "running (start it from the Serve "
                                         "panel)",
                                    code="not_serving")
            return (str(st["base_url"]), st.get("attach_api_key"),
                    str(st.get("model_name") or inst.get("name") or "model"))
        # Routing seam: per-run tier selection across running instances. An
        # empty/unroutable pool falls through to the legacy resolution below,
        # whose honest 503 ("nothing is serving") stays the user-facing error.
        try:
            from ..serving.router import routing_config as _routing_cfg
            _rc = _routing_cfg(self.cfg)
        except Exception:  # noqa: BLE001 — a broken section never breaks runs
            _rc = None
        if _rc is not None and _rc.enabled:
            try:
                from ..serving.router import AutoRouter
                router = AutoRouter.from_home(
                    self.home, self.cfg, routing_hint=routing_hint)
                decision = router.select()
                t = decision.target
                logger.info("routing: run -> instance %s (tier %d)%s",
                            t.instance_id, t.tier,
                            f" [{decision.note}]" if decision.note else "")
                return (t.base_url, t.api_key, t.model_name)
            except Exception as e:  # noqa: BLE001 — fall through to legacy
                logger.debug("routing unavailable (%s); "
                             "using default endpoint resolution", e)
        st = read_serve_state(self.home)
        if st and self._serve_alive(st):
            return (str(st["base_url"]), st.get("attach_api_key"),
                    str(st.get("model_name") or "model"))
        # WP5: the default instance idle/coexist-unloaded (tomb state with a
        # dead pid). Lazy reload when enabled; clear guidance otherwise. The
        # untouched legacy "nothing is serving" 503 below stays for states
        # that were never loaded.
        if st and st.get("unloaded") and self.coexist is not None:
            if self.coexist.cfg.lazy_load:
                try:
                    st2 = self.coexist.ensure_loaded(srv_inst.DEFAULT_ID)
                except Exception as e:
                    raise HttpError(503, str(e),
                                    code="lazy_load_failed") from e
                if st2 and self._serve_alive(st2):
                    return (str(st2["base_url"]), st2.get("attach_api_key"),
                            str(st2.get("model_name") or "model"))
            else:
                raise HttpError(
                    503, self.coexist.unloaded_guidance(srv_inst.DEFAULT_ID)
                    or "default server is unloaded",
                    code="instance_unloaded") from None
        attach = self.cfg.effective_attach_url
        if attach:
            return (attach, self.cfg.attach_api_key,
                    self.cfg.model_name or "model")
        raise HttpError(503, "nothing is serving — start a server "
                             "(POST /api/serve/start) or attach one",
                        code="not_serving")

    def _instance_context_length(self, server_instance: Optional[str]
                                 ) -> int:
        """Context window for the chat context tracker: the bound instance's
        context_length when set, else the global config value."""
        if not server_instance or str(server_instance).strip() in (
                "", "default", "none"):
            return int(self.cfg.context_length)
        inst = srv_inst.get_instance(self.home, str(server_instance).strip(),
                                     cfg=self.cfg) or {}
        try:
            return int(inst.get("context_length") or self.cfg.context_length)
        except (TypeError, ValueError):
            return int(self.cfg.context_length)

    def _instance_sampling(self, server_instance: Optional[str],
                           ) -> Dict[str, Any]:
        """Per-instance sampling override (Phase 6) for profile-bound runs."""
        if not server_instance or str(server_instance).strip() in (
                "", "default", "none"):
            return {}
        inst = srv_inst.get_instance(self.home, str(server_instance).strip(),
                                     cfg=self.cfg)
        sp = (inst or {}).get("sampling")
        return dict(sp) if isinstance(sp, dict) else {}

    def h_proxy_chat(self, q: Dict[str, str], b: Dict[str, Any]) -> ApiResponse:
        import httpx

        # WP4 watchdog: this proxy is one of the two paths that put requests
        # in flight against a served endpoint (the other is the agent runner's
        # chat calls). The tally is a cheap dict counter — inert unless
        # serve.watchdog_enabled is on.
        from ..serving.watchdog import track_request

        base, key, model = self._resolve_chat_target()
        payload = dict(b)
        payload.setdefault("model", model)
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        url = base.rstrip("/") + "/chat/completions"

        if not payload.get("stream"):
            try:
                with track_request(base):
                    r = httpx.post(url, json=payload, headers=headers,
                                   timeout=300.0)
            except httpx.HTTPError as e:
                raise HttpError(502, f"upstream request failed: {e}") from e
            try:
                return ApiResponse(r.status_code, r.json())
            except ValueError:
                return ApiResponse(r.status_code, {"raw": r.text[:4000]})

        def gen() -> Iterator[str]:
            try:
                with track_request(base), \
                        httpx.Client(timeout=httpx.Timeout(600.0,
                                                           connect=10.0)) as c:
                    with c.stream("POST", url, json=payload,
                                  headers=headers) as upstream:
                        if upstream.status_code != 200:
                            detail = upstream.read().decode(errors="replace")[:2000]
                            yield sse_frame("proxy.error", {
                                "status": upstream.status_code, "detail": detail})
                            return
                        for line in upstream.iter_lines():
                            if not line:
                                continue
                            # Verbatim SSE passthrough.
                            yield line + "\n"
            except Exception as e:  # surface upstream failures as an event
                yield sse_frame("proxy.error", {"message": f"{type(e).__name__}: {e}"})

        return ApiResponse(200, stream=gen())

    def h_chat_stream(self, q: Dict[str, str], b: Dict[str, Any]) -> ApiResponse:
        prompt = str(b.get("prompt") or "").strip()
        # Image support: `images` is an optional list of base64 data URLs
        # the user attached in chat. They are saved under <home>/uploads and
        # referenced by attach markers — the runner expands them into native
        # image parts per-request when the served model has vision, while the
        # thread/trace keep lean text (core/images.py).
        imgs_in = b.get("images")
        if imgs_in:
            from ..core.images import attach_marker, save_data_url

            if not isinstance(imgs_in, list) or len(imgs_in) > 4:
                raise HttpError(400, "images must be a list of up to 4 data URLs")
            for du in imgs_in:
                saved, err = save_data_url(str(du), self.home / "uploads")
                if err is not None or not saved:
                    raise HttpError(400, err or "image upload failed")
                prompt = f"{prompt}\n{attach_marker(saved)}"
        if not prompt:
            raise HttpError(400, "prompt required")
        history = b.get("history")
        if history is not None and not isinstance(history, list):
            raise HttpError(400, "history must be a list of messages")
        # Backlog #6: rolling compaction summary persisted with the thread;
        # threaded back into the runner so the next compaction updates it.
        previous_summary = b.get("previous_summary")
        if previous_summary is not None and not isinstance(previous_summary, str):
            raise HttpError(400, "previous_summary must be a string")

        # 5e: optional agent profile (name | null). Resolved *before* the
        # stream starts so an unknown name is a clean 404, not a run.error.
        prof = self._resolve_profile(b.get("profile") or None)
        prof_tools = None
        if prof and isinstance(prof.get("tools"), list):
            prof_tools = [str(t) for t in prof["tools"]]
        # Projects: optional active-project id on the run context. Resolved
        # up-front (same rationale as the profile): unknown id => clean 404.
        # It reaches traces via the recorder and the workspace via the runner.
        project_id = str(b["project"]).strip() if b.get("project") else None
        proj: Optional[Dict[str, Any]] = None
        if project_id:
            proj = self._load_project(project_id)
            if proj is None:
                raise HttpError(404, f"no project {project_id}",
                                code="unknown_project")
        # Phase 6: a profile can bind chat runs to a named server instance
        # (orchestrator on one model, coding profile on another, concurrently).
        # routing_hint ("auto"|"low"|"high", run-settings field) is the
        # per-request signal for capability-tier auto-routing (review idea
        # #2) — ignored unless routing.enabled and no explicit binding.
        _routing_hint = str(b.get("routing_hint") or "").strip().lower() \
            or None
        if _routing_hint not in (None, "auto", "low", "high"):
            _routing_hint = None
        # Only pass the kwarg when set: callers/stubs written against the
        # pre-routing signature keep working untouched (default path call
        # shape is byte-identical to before).
        base, key, model = self._resolve_chat_target(
            (prof or {}).get("server_instance"), **({"routing_hint": _routing_hint}
                                                    if _routing_hint else {}))
        # Profile model alias override (falls back to the served model).
        if prof and prof.get("model_name"):
            model = str(prof["model_name"])
        instance_sampling = self._instance_sampling(
            (prof or {}).get("server_instance"))
        # Per-profile sampling/thinking override (reference style): wins
        # over the bound instance's defaults, which win over global sampling:.
        prof_sampling = (prof or {}).get("sampling")
        if not isinstance(prof_sampling, dict):
            prof_sampling = {}
        sampling_override = {**instance_sampling, **prof_sampling}
        # Context window for the chat UI's context tracker (per-instance
        # override first, then the global config).
        ctx_window = self._instance_context_length(
            (prof or {}).get("server_instance"))
        # Phase 2 (delegation): optional thread id on the run context — tools
        # append lineage entries to that thread JSON. Invalid ids degrade to
        # no-thread (CLI-like) behavior instead of failing the run.
        thread_id = str(b.get("thread") or "").strip() or None
        if thread_id and not self._THREAD_ID_RE.match(thread_id):
            thread_id = None

        # Single-flight per thread: a second POST for a thread with an
        # in-flight run gets a clean 409 carrying the active run_id, instead
        # of silently spawning a duplicate agent (the orphan-duplicate cascade
        # — user resends a message they believed was stopped).
        if thread_id:
            with self._chat_runs_lock:
                for rid, r in self._chat_runs.items():
                    if not r.get("done") and r.get("thread") == thread_id:
                        raise HttpError(409,
                                        "a run is already in flight for this "
                                        "thread — stop it first",
                                        code="run_in_flight",
                                        extra={"run_id": rid})
        run_id = uuid.uuid4().hex[:12]
        cancel = threading.Event()
        # Registry entry is created INSIDE gen() (below), not here: an SSE
        # admission rejection (server.py sse_slots cap) never starts the
        # generator, and an eagerly-created entry would then never be marked
        # done — permanently 409-ing the thread via the single-flight check.

        from ..store.recorder import make_recorder

        def _on_record(row: Any) -> None:
            self.bus.publish("trace.recorded", {
                "trace_id": row.trace_id, "status": row.status,
                "task_summary": row.task_summary,
            })

        def _on_delegation_update(rec: Dict[str, Any]) -> None:
            # Live status flips for the chat renderer's nested delegation card.
            self.bus.publish("delegation.update", rec)

        # Easy training: when the loop is enabled for THIS profile, stamp its
        # name into traces.agent_profile so curation/readiness scope to it.
        # Multi-agent loops stamp any profile in ec.all_profiles — each agent's
        # traces feed that agent's slice of the loop. Otherwise agent_profile
        # keeps the backend name (legacy behavior).
        _easy = self._easy_cfg()
        _prof_name = str(prof.get("name") or "") if prof else ""
        rec_profile = (_prof_name
                       if _easy.enabled and _prof_name
                       and _prof_name in _easy.all_profiles else None)
        recorder = make_recorder(self.cfg, on_record=_on_record,
                                 project=project_id, profile=rec_profile)

        def gen() -> Iterator[str]:
            import queue as _queue

            # First iteration = the stream was admitted: register now so a
            # rejected (never-started) stream leaves no phantom run behind.
            with self._chat_runs_lock:
                self._chat_runs[run_id] = {
                    "cancel": cancel, "done": False, "thread": thread_id,
                    "model": model, "profile": prof.get("name") if prof else None,
                    "status": "running",
                }
            t0 = time.monotonic()
            yield sse_frame("run.started", {"run_id": run_id, "model": model,
                                            "profile": prof.get("name") if prof
                                            else None,
                                            # Context tracker: the window the
                                            # chat UI renders usage against.
                                            "context_window": ctx_window})
            if cancel.is_set():
                yield sse_frame("run.error", {"message": "cancelled"})
                return

            # reference-style live streaming: the runner executes on a
            # worker thread and forwards token/reasoning/tool events through
            # this queue while it runs; the generator drains it so deltas
            # reach the browser as they are generated instead of after the
            # whole run. Runners without streaming support (test stubs, CLI
            # shims) produce no live events — we synthesize the protocol from
            # the returned message list afterwards, exactly like before.
            evq: "_queue.Queue[Any]" = _queue.Queue()
            live = {"n": 0}

            def _sink(kind: str, data: Dict[str, Any]) -> None:
                live["n"] += 1
                evq.put(("event", kind, data))

            def _worker() -> None:
                try:
                    res = (runner.run(prompt, history=history,
                                      previous_summary=previous_summary)
                           if previous_summary else
                           runner.run(prompt, history=history))
                    evq.put(("done", res))
                except Exception as e:  # surfaced as run.error by the drainer
                    evq.put(("error", e))

            def _make_runner():
                from ..core.agent_runner import create_runner

                return create_runner(self.cfg, base, model, api_key=key,
                                     recorder=recorder,
                                     system_prompt=(prof or {}).get(
                                         "system_prompt") or None,
                                     enabled_tool_names=prof_tools,
                                     # Phase 5: route search_codebase to the
                                     # active project's index DB.
                                     project_id=project_id,
                                     # Backlog #9: per-profile workspace
                                     # override (widens/narrows the global
                                     # agent.workspace_root for this chat).
                                     workspace_root_override=(prof or {}).get(
                                         "workspace_root"),
                                     # Projects: active project's workspace
                                     # dir becomes the containment root with
                                     # precedence below profile, above config.
                                     project_workspace_dir=(proj or {}).get(
                                         "workspace_dir"),
                                     # Phase 2 (delegation): run metadata for
                                     # list_agents/delegate_task/check_delegation.
                                     delegation_context={
                                         "cfg": self.cfg,
                                         "base_url": base, "model": model,
                                         "profile": prof.get("name") if prof else None,
                                         "project": project_id,
                                         "thread": thread_id,
                                         "run_id": run_id,
                                         "depth": 0,
                                         "on_update": _on_delegation_update,
                                     },
                                     # Phase 6 + per-profile thinking knobs:
                                     # profile `sampling:` wins over the bound
                                     # instance's, which wins over global.
                                     sampling_override=sampling_override or None,
                                     # Per-agent capabilities (internet /
                                     # run_commands): profile toggles merged
                                     # over global agent.capabilities.
                                     capabilities=(prof or {}).get(
                                         "capabilities"),
                                      # Pass 4 #38: playbook injection selects
                                      # rows by profile name (no-op unless
                                      # playbooks.enabled).
                                      profile_name=prof.get("name")
                                      if prof else None,
                                      # Run lifecycle: h_chat_cancel sets this
                                      # event; the runner polls it at loop
                                      # boundaries and stops cleanly with
                                      # failure_reason="cancelled".
                                      cancel_event=cancel)
            try:
                runner = _make_runner()
                # Attach the live sink (plain attribute on the built-in
                # runner; harmless for stubs that never call it).
                try:
                    runner.on_event = _sink
                except Exception:
                    pass
            except Exception as e:
                yield sse_frame("run.error", {"message": f"{type(e).__name__}: {e}"})
                self._chat_runs[run_id]["done"] = True
                self._chat_runs[run_id]["status"] = "completed"
                return

            threading.Thread(target=_worker, daemon=True).start()
            res = None
            run_exc: Optional[Exception] = None
            try:
                while True:
                    try:
                        item = evq.get(timeout=0.25)
                    except _queue.Empty:
                        continue
                    tag = item[0]
                    if tag == "event":
                        yield sse_frame(item[1], item[2])
                    elif tag == "done":
                        res = item[1]
                        break
                    else:  # ("error", exc)
                        run_exc = item[1]
                        break

                if run_exc is not None:
                    yield sse_frame("run.error", {
                        "message": f"{type(run_exc).__name__}: {run_exc}"})
                elif res is not None:
                    if live["n"] == 0:
                        # Runner without streaming support (stubs/CLI shims):
                        # synthesize the event protocol from the message list.
                        for m in res.messages:
                            role = m.get("role")
                            if role == "assistant":
                                content = m.get("content")
                                if isinstance(content, str) and content:
                                    yield sse_frame("message.delta",
                                                    {"text": content})
                                for tc in (m.get("tool_calls") or []):
                                    fn = tc.get("function") or {}
                                    try:
                                        args = json.loads(
                                            fn.get("arguments") or "{}")
                                    except json.JSONDecodeError:
                                        args = {"raw": fn.get("arguments")}
                                    yield sse_frame("tool.call", {
                                        "id": tc.get("id", ""),
                                        "name": fn.get("name", ""),
                                        "args": args})
                            elif role == "tool":
                                content = m.get("content")
                                if not isinstance(content, str):
                                    content = json.dumps(content, default=str)
                                preview = content[:8000]
                                yield sse_frame("tool.result", {
                                    "id": m.get("tool_call_id", ""),
                                    "name": m.get("name"),
                                    "ok": not _looks_like_error(content),
                                    "preview": preview,
                                    "truncated": len(content) > 8000,
                                })
                    yield sse_frame("turn.done", {"completed": res.completed})
                    yield sse_frame("run.done", {
                        "trace_id": res.trace_id,
                        "tokens_in": res.tokens_in,
                        "tokens_out": res.tokens_out,
                        "duration_s": round(time.monotonic() - t0, 3),
                        "backend": res.backend,
                        # Why a controlled stop happened ("max_iterations",
                        # "guardrail_halt:<code>", "backend_error:<reason>");
                        # null on success. The chat footer renders this so the
                        # user never has to guess why the agent stopped.
                        "failure_reason": res.failure_reason,
                        # Backlog #6: rolling compaction summary (None when no
                        # compression ran); the client persists it with the thread.
                        "summary": res.summary,
                        # Context tracker: final window usage for the footer.
                        "context_window": ctx_window,
                    })
            finally:
                self._chat_runs[run_id]["done"] = True
                if self._chat_runs[run_id].get("status") != "cancelling":
                    self._chat_runs[run_id]["status"] = "completed"

        return ApiResponse(200, stream=gen())

    def h_chat_cancel(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        run_id = str(b.get("run_id", ""))
        run = self._chat_runs.get(run_id)
        if run is None:
            raise HttpError(404, f"no chat run {run_id}")
        run["cancel"].set()
        run["status"] = "cancelling"
        return {"ok": True, "cancelling": not run["done"]}

    def h_chat_runs(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        """In-flight chat run registry (run lifecycle, step 2).

        The chat panel calls this on load: if the active thread has an
        in-flight run (e.g. the user navigated away mid-run and came back —
        the run kept executing on its worker thread, the old panel just
        stopped reading it), the panel adopts it: Stop button reappears,
        pointing at the real run_id. Done runs are pruned here so the
        registry does not grow forever.
        """
        with self._chat_runs_lock:
            # Prune finished runs (their state is no longer needed — the
            # trace/summary already recorded everything).
            for rid in [r for r, v in self._chat_runs.items()
                        if v.get("done")]:
                del self._chat_runs[rid]
            runs = [{
                "run_id": rid,
                "thread": v.get("thread"),
                "model": v.get("model"),
                "profile": v.get("profile"),
                "status": v.get("status", "running"),
            } for rid, v in self._chat_runs.items()
                    if not v.get("done")]
        return {"runs": runs}

    def shutdown_chat_runs(self, grace_s: float = 5.0) -> int:
        """Cooperative shutdown for in-flight chat runs (run lifecycle, step 3).

        The runner threads are daemons: without help they die mid-turn with no
        trace and a thread whose last save lacks the partial turn. On dashboard
        exit we set every in-flight cancel event — the agent runner polls it at
        loop boundaries and stops cleanly, recording its own score-neutralized
        cancelled trace (store/recorder HIGH #4) — then wait up to ``grace_s``
        for them to drain. Runs still unfinished after the grace window are
        marked interrupted; their SSE clients die with the socket anyway.

        MUST be called BEFORE stop_all_servers(): a runner cancelling while its
        backend is being killed records an error trace instead of a clean one.
        Returns the number of runs that were in flight at call time.
        """
        with self._chat_runs_lock:
            inflight = [(rid, r) for rid, r in self._chat_runs.items()
                        if not r.get("done")]
            for _, r in inflight:
                r["cancel"].set()
                r["status"] = "cancelling"
        deadline = time.monotonic() + grace_s
        while time.monotonic() < deadline:
            if all(r.get("done") for _, r in inflight):
                break
            time.sleep(0.1)
        for rid, r in inflight:
            if not r.get("done"):
                logging.getLogger("sable.dashboard").warning(
                    "chat run %s still in flight at shutdown (grace expired); "
                    "it dies with this process — no trace will be recorded",
                    rid)
                r["status"] = "interrupted"
        return len(inflight)

    # -- thread persistence (<home>/chat/<id>.json) ---------------------------

    _THREAD_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

    def _thread_path(self, thread_id: str) -> Path:
        if not self._THREAD_ID_RE.match(thread_id):
            raise HttpError(400, "invalid thread id")
        return self.home / CHAT_DIRNAME / f"{thread_id}.json"

    def h_thread_get(self, q: Dict[str, str], b: Dict[str, Any],
                     thread_id: str) -> Dict[str, Any]:
        p = self._thread_path(thread_id)
        if not p.is_file():
            raise HttpError(404, f"no thread {thread_id}")
        try:
            return {"thread": json.loads(p.read_text())}
        except (OSError, json.JSONDecodeError) as e:
            raise HttpError(500, f"unreadable thread file: {e}") from e

    def h_threads_list(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        d = self.home / CHAT_DIRNAME
        threads = []
        if d.is_dir():
            for f in sorted(d.glob("*.json")):
                try:
                    t = json.loads(f.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                threads.append({
                    "id": t.get("id", f.stem),
                    "title": t.get("title") or f.stem,
                    # Title provenance (pass 3 #18): derived | llm | user;
                    # legacy threads without the key read as un-sourced.
                    "title_source": t.get("title_source"),
                    "created": t.get("created"),
                    "updated": t.get("updated"),
                    "model_alias": t.get("model_alias"),
                    "trace_ids": t.get("trace_ids") or [],
                    "message_count": len(t.get("messages") or []),
                    # Projects: null/absent = Unfiled. Legacy chat files
                    # predate the key entirely — .get() keeps them readable
                    # (no migration needed; they simply list as unfiled).
                    "project": t.get("project"),
                    # Last agent profile used on this thread (null = default
                    # built-in); shown in the explorer tree's thread rows.
                    "profile": t.get("profile"),
                    # Phase 2 placeholder: delegation lineage chip data when a
                    # future writer stores it; absent tolerated everywhere.
                    "delegations": t.get("delegations") or [],
                })
        return {"threads": threads}

    def h_thread_put(self, q: Dict[str, str], b: Dict[str, Any],
                     thread_id: str) -> Dict[str, Any]:
        p = self._thread_path(thread_id)
        existing: Dict[str, Any] = {}
        if p.is_file():
            try:
                existing = json.loads(p.read_text())
            except (OSError, json.JSONDecodeError):
                existing = {}

        # Two-stage titles (pass 3 #18). An explicit title from the client is a
        # USER rename — authoritative, never auto-overwritten. Otherwise stage 1
        # derives instantly from the opening message (cannot fail), and stage 2
        # may later upgrade it off this request via an LLM call.
        from ..core.titles import DERIVED, USER, derive_title
        existing_source = existing.get("title_source")
        if "title" in b and str(b.get("title") or "").strip():
            title_val = str(b["title"]).strip()
            title_src = USER
        else:
            title_val = existing.get("title")
            title_src = existing_source
            msgs_for_title = (b["messages"]
                              if isinstance(b.get("messages"), list)
                              else existing.get("messages") or [])
            if not title_val and msgs_for_title:
                title_val = derive_title(msgs_for_title)
                title_src = DERIVED
        thread = {
            "id": thread_id,
            # Single source of truth: the block above already resolved
            # explicit-client-title vs existing vs derived into title_val.
            "title": title_val,
            "title_source": title_src,
            "created": existing.get("created") or _utcnow(),
            "updated": _utcnow(),
            "model_alias": b.get("model_alias", existing.get("model_alias")),
            "messages": b["messages"] if isinstance(b.get("messages"), list)
            else existing.get("messages", []),
            "trace_ids": sorted(set((existing.get("trace_ids") or [])
                                    + list(b.get("trace_ids") or []))),
            # Backlog #6: rolling compaction summary persisted alongside the
            # thread history so a reopened thread keeps its compression state.
            # Explicit null clears it; absent key keeps the stored one.
            "summary": (b["summary"] if "summary" in b
                        and isinstance(b["summary"], str) else None)
            if "summary" in b else existing.get("summary"),
            # Projects: explicit null moves the thread to Unfiled; absent key
            # keeps the stored project. A new thread may be created directly
            # with a project id (moving between projects = update). Unknown
            # ids rejected so the tree never points at a deleted project file.
            "project": ((b["project"] if b.get("project") else None)
                        if "project" in b else existing.get("project")),
            # Last agent profile used on this thread (display metadata only).
            "profile": b.get("profile", existing.get("profile")),
            # Phase 2 (delegation): lineage entries are appended server-side by
            # the delegation tools; a client save that omits the key must not
            # wipe them. An explicit list replaces (client-authoritative edit).
            "delegations": (b["delegations"]
                            if isinstance(b.get("delegations"), list)
                            else existing.get("delegations") or []),
        }
        if thread["project"] and self._load_project(thread["project"]) is None:
            raise HttpError(400, f"no project {thread['project']}",
                            code="unknown_project")
        p.parent.mkdir(parents=True, exist_ok=True)
        # Atomic durable write (tmp + os.replace), matching every other
        # persisted JSON in the codebase — a crash mid-write must never leave
        # a partial file that the read path would treat as an empty thread.
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(thread, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        # Transcripts can quote secrets the user pasted into a prompt —
        # owner-only on disk (same hardening as the serve-state writers).
        os.chmod(tmp, 0o600)
        os.replace(tmp, p)

        # Stage 2 (off the critical path): a freshly derived title on a thread
        # that has an assistant reply gets one bounded LLM upgrade attempt.
        # Fire-and-forget daemon thread; failure leaves the stage-1 label.
        if (title_src == DERIVED and thread.get("messages")
                and any(m.get("role") == "assistant"
                        for m in thread["messages"] if isinstance(m, dict))):
            self._schedule_llm_title(thread_id)

        return {"ok": True, "thread": thread}

    def _schedule_llm_title(self, thread_id: str) -> None:
        """Stage-2 title upgrade: one small-model call in a daemon thread.

        Re-reads the thread file at execution time (the client may have renamed
        it in the meantime — the provenance gate is applied AGAIN against the
        then-current source, so a user rename always wins). Never raises; any
        error just keeps the derived title.
        """
        def _work() -> None:
            try:
                from ..core.titles import LLM, generate_title, should_apply_title

                p = self._thread_path(thread_id)
                if not p.is_file():
                    return
                t = json.loads(p.read_text())
                if not should_apply_title(t.get("title_source"), LLM):
                    return  # user renamed (or llm already set) while we queued

                base, key, model = self._resolve_chat_target()
                import httpx

                def client_fn(prompt_msgs, _model):
                    headers = {"Authorization": f"Bearer {key}"} if key else {}
                    r = httpx.post(
                        base.rstrip("/") + "/chat/completions",
                        json={"model": _model, "messages": prompt_msgs,
                              "max_tokens": 48, "temperature": 0.2},
                        headers=headers, timeout=60.0)
                    if r.status_code != 200:
                        return None
                    msg = ((r.json().get("choices") or [{}])[0]).get("message") or {}
                    return msg.get("content")

                title = generate_title(client_fn, model, t.get("messages") or [])
                if not title:
                    return
                # Re-check under the write (rename race during the LLM call).
                t2 = json.loads(p.read_text())
                if not should_apply_title(t2.get("title_source"), LLM):
                    return
                t2["title"] = title
                t2["title_source"] = LLM
                tmp = p.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(t2, ensure_ascii=False, indent=2),
                               encoding="utf-8")
                os.replace(tmp, p)
            except Exception:  # a title upgrade must never surface anywhere
                import logging
                logging.getLogger("sable.dashboard").debug(
                    "llm title upgrade failed for %s", thread_id, exc_info=True)

        threading.Thread(target=_work, daemon=True).start()

    def h_thread_delete(self, q: Dict[str, str], b: Dict[str, Any],
                        thread_id: str) -> Dict[str, Any]:
        p = self._thread_path(thread_id)
        if not p.is_file():
            raise HttpError(404, f"no thread {thread_id}")
        p.unlink()
        return {"ok": True}

    # --------------------------------------------------------------- traces

    def h_traces_list(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        limit = int(q.get("limit", "50") or 50)
        offset = max(0, int(q.get("offset", "0") or 0))
        curated_raw = q.get("curated")
        curated: Optional[bool] = None
        if curated_raw not in (None, ""):
            curated = curated_raw.lower() in ("1", "true", "yes")
        store = self._store()
        try:
            rows = store.list_traces(
                limit=min(limit, 500),
                status=q.get("status") or None,
                since=q.get("since") or None,
                q=q.get("q") or None,
                agent_profile=q.get("profile") or None,
                model_alias=q.get("model_alias") or None,
                curated=curated,
                offset=offset,
                project=q.get("project") or None,
            )
            # 5d: total across the whole filtered set (not just this window)
            # so pagers can render "page X of Y".
            total = store.count_traces(
                status=q.get("status") or None,
                since=q.get("since") or None,
                q=q.get("q") or None,
                agent_profile=q.get("profile") or None,
                model_alias=q.get("model_alias") or None,
                curated=curated,
                project=q.get("project") or None,
            )
        finally:
            store.close()
        return {"traces": [r.to_dict() for r in rows], "total": total,
                "offset": offset, "limit": min(limit, 500)}

    def h_traces_stats(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        days = int(q.get("days", "30") or 30)
        store = self._store()
        try:
            series = store.stats_daily(days=days)
        finally:
            store.close()
        return {"series": series}

    def h_insights(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        """Backlog #11: insights report over the buffer DB (read-only)."""
        from ..store.insights import generate_insights

        try:
            days = int(q.get("days", "30") or 30)
        except ValueError:
            raise HttpError(400, "days must be an integer") from None
        return generate_insights(self.home / "buffer" / "sable.db",
                                 days=days, profile=q.get("profile") or None)

    def h_metrics(self, q: Dict[str, str], b: Dict[str, Any]) -> ApiResponse:
        """Prometheus text exposition over the buffer/registry DB (idea #5).

        Raw-text ApiResponse; server.py writes it with the declared content
        type. Auth is NOT handled here — every /api/* request passes the same
        token + loopback/CSRF gate in server.py first, so scrapers send
        X-Sable-Token (or ?token=) exactly like any other endpoint. The
        rendered body is memoized for METRICS_TTL_S seconds (scrape-friendly;
        force refresh with ?refresh=1).
        """
        from .metrics import CONTENT_TYPE, METRICS_TTL_S, collect

        now = time.monotonic()
        stale = bool(q.get("refresh"))
        with self._metrics_lock:
            if not stale and self._metrics_cache is not None \
                    and now < self._metrics_cache[0]:
                body = self._metrics_cache[1]
            else:
                body = collect(self.home, cfg=self.cfg)
                self._metrics_cache = (now + METRICS_TTL_S, body)
        return ApiResponse(200, raw=body.encode("utf-8"),
                           content_type=CONTENT_TYPE)

    def h_trace_detail(self, q: Dict[str, str], b: Dict[str, Any],
                       trace_id: str) -> Dict[str, Any]:
        store = self._store()
        try:
            row = store.get_trace(trace_id)
            feedback = store.get_feedback(trace_id) if row else []
        finally:
            store.close()
        if row is None:
            raise HttpError(404, f"no trace {trace_id}")
        messages: List[Dict[str, Any]] = []
        records: List[Dict[str, Any]] = []
        try:
            with open(row.jsonl_path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    records.append(rec)
                    if rec.get("kind") == "message":
                        messages.append(rec["message"])
        except (OSError, json.JSONDecodeError):
            pass  # transcript file may be pruned; index still valid
        out: Dict[str, Any] = {"trace": row.to_dict(), "feedback": feedback,
                               "messages": messages}
        qq = q.get("q")
        if qq:
            needle = qq.lower()
            out["messages"] = [
                m for m in messages
                if needle in json.dumps(m, ensure_ascii=False, default=str).lower()
            ]
        # Pass 4 #35: derived swimlane segments (read-time only; stored trace
        # untouched). Anchored into the *delivered* (possibly q-filtered) list.
        out["timeline"] = build_trace_timeline(records, out["messages"])
        return out

    def h_feedback(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        return self.add_feedback(b)

    def h_trace_feedback(self, q: Dict[str, str], b: Dict[str, Any],
                         trace_id: str) -> Dict[str, Any]:
        body = dict(b)
        body.pop("trace_id", None)
        body.pop("last", None)
        body["trace_id"] = trace_id
        return self.add_feedback(body)

    def add_feedback(self, b: Dict[str, Any]) -> Dict[str, Any]:
        score = b.get("score")
        note = b.get("note")
        if score is not None:
            try:
                score = float(score)
            except (TypeError, ValueError):
                raise HttpError(400, "score must be a number") from None
            if not -1.0 <= score <= 1.0:
                raise HttpError(400, "score must be within [-1, 1]")
        if score is None and not note:
            raise HttpError(400, "provide score and/or note")

        store = self._store()
        try:
            row = store.latest_trace() if b.get("last") else store.get_trace(
                str(b.get("trace_id", ""))
            )
            if row is None:
                raise HttpError(404, "no such trace")
            store.add_feedback(
                row.trace_id, score=score, comment=note, source="user",
                timestamp=_utcnow(),
            )
        finally:
            store.close()
        self.bus.publish("trace.feedbacked", {"trace_id": row.trace_id})
        return {"ok": True, "trace_id": row.trace_id}

    # ------------------------------------------------- failure remediation notes
    # Idea adapted from PIASystem (MIT) app/core/failure_taxonomy.py —
    # update_remediation_status() exposed as the single dashboard endpoint.
    # Current notes ride along inside /api/insights' failure_reasons section,
    # so badges render without extra round-trips; this POST only mutates.

    def h_failure_note_set(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        reason = str(b.get("reason") or "").strip()
        if not reason:
            raise HttpError(400, "reason required")
        status = str(b.get("status") or "").strip().lower()
        from ..store.db import FAILURE_NOTE_STATUSES
        if status not in FAILURE_NOTE_STATUSES:
            raise HttpError(
                400, f"status must be one of {', '.join(FAILURE_NOTE_STATUSES)}")
        note = str(b.get("note") or "")
        profile = (str(b.get("profile")).strip() or None) if b.get("profile") else None
        store = self._store()
        try:
            store.set_failure_note(reason, status, note=note, profile=profile)
            current = store.get_failure_note(reason, profile=profile)
        finally:
            store.close()
        return {"ok": True, "reason": reason, "profile": profile,
                "status": status, "note": current}

    # ------------------------------------------------- playbooks (pass 4 #38)
    # Idea adapted from PIASystem (MIT) app/core/skill_creator.py — recipes
    # distilled from recurring tool chains. The dashboard surfaces stored
    # rows + a live scan; promote/retire are explicit user actions (the CLI
    # gate machinery with judge calls stays CLI-side, out of the panel).

    def _playbook_store(self):
        from ..core.playbooks import PlaybookStore

        return PlaybookStore(self.home / "buffer" / "sable.db")

    # Live chain scanning reads up to MAX_SCAN_TRACES JSONL transcripts per
    # call; running it on every Profiles-panel render is pure waste. Short TTL
    # cache per profile — candidates appear minutes, not milliseconds, apart.
    _CHAIN_SCAN_TTL_SECS = 120.0

    def _cached_chain_scan(self, profile):
        import time as _time

        now = _time.monotonic()
        cache = self.__dict__.setdefault("_chain_scan_cache", {})
        hit = cache.get(profile)
        if hit and now - hit[0] < self._CHAIN_SCAN_TTL_SECS:
            return hit[1]
        store = self._store()
        try:
            from ..core.playbooks import find_recurring_chains

            chains = find_recurring_chains(store, profile)[:10]
        finally:
            store.close()
        cache[profile] = (now, chains)
        return chains

    def h_playbooks_list(self, q: Dict[str, str],
                         b: Dict[str, Any]) -> Dict[str, Any]:
        from ..core.playbooks import playbooks_config

        profile = (str(q.get("profile")).strip() or None) \
            if q.get("profile") else None
        pb_store = self._playbook_store()
        try:
            rows = pb_store.list(profile=profile)
        finally:
            pb_store.close()
        chains = self._cached_chain_scan(profile)
        return {
            "enabled": bool(playbooks_config(self.cfg).get("enabled", False)),
            "playbooks": [p.to_dict() for p in rows],
            "chains": chains,
        }

    def h_playbook_assess(self, q: Dict[str, str],
                          b: Dict[str, Any]) -> Dict[str, Any]:
        """Correlational regression check (core/playbooks.assess_playbook):
        mean outcome score of matching-chain traces after vs before promotion;
        a clear regression retires the playbook. Explicit user action — never
        automatic from the panel."""
        from ..core.playbooks import Playbook, assess_playbook

        pb_id = str(b.get("id") or "").strip()
        pb_store = self._playbook_store()
        store = self._store()
        try:
            targets: List[Playbook] = []
            if pb_id:
                found = pb_store.get(pb_id)
                if found is None:
                    raise HttpError(404, f"no playbook {pb_id}")
                targets.append(found)
            else:
                targets.extend(pb_store.list(status="active"))
            results = []
            for pb in targets:
                res = assess_playbook(store, pb_store, pb)
                results.append({"id": pb.id, "chain": pb.chain, **res})
        finally:
            store.close()
            pb_store.close()
        return {"ok": True, "results": results}

    # ------------------------------------------- mined golden suite (UI view)

    def h_goldens_mined_list(self, q: Dict[str, str],
                             b: Dict[str, Any]) -> Dict[str, Any]:
        """Every mined golden (enabled + disabled) with provenance — the
        operator view that makes the per-line enabled flag usable."""
        from ..curate.mine_goldens import list_mined_goldens

        return {"goldens": list_mined_goldens(self.home)}

    def h_goldens_mined_toggle(self, q: Dict[str, str],
                               b: Dict[str, Any]) -> Dict[str, Any]:
        from ..curate.mine_goldens import set_mined_enabled

        key = str(b.get("key") or "").strip()
        if not key:
            raise HttpError(400, "key required")
        enabled = bool(b.get("enabled"))
        if not set_mined_enabled(self.home, key, enabled):
            raise HttpError(404, f"no mined golden {key}")
        return {"ok": True, "key": key, "enabled": enabled}

    # ---- persistent skills (core/skills.py) ---------------------------------

    def h_skills_list(self, q: Dict[str, str],
                      b: Dict[str, Any]) -> Dict[str, Any]:
        from ..core.skills import all_skills, skills_config, skills_dir

        scfg = skills_config(self.cfg)
        return {
            "enabled": bool(scfg.get("enabled", False)),
            "dir": str(skills_dir()),
            "index_total_cap_chars": int(scfg.get("index_total_cap_chars", 1500)),
            "skills": [{"name": s.name, "description": s.description,
                        "updated": s.updated_ts, "path": s.path,
                        "chars": len(s.content)} for s in all_skills()],
        }

    def h_skill_get(self, q: Dict[str, str], b: Dict[str, Any], *,
                    skill_name: str) -> Dict[str, Any]:
        from ..core.skills import load_skill

        s = load_skill((skill_name or "").strip().lower())
        if s is None:
            raise HttpError(404, f"no such skill: {skill_name}")
        return {"name": s.name, "description": s.description,
                "updated": s.updated_ts, "path": s.path, "content": s.content}

    def h_skill_put(self, q: Dict[str, str], b: Dict[str, Any], *,
                    skill_name: str) -> Dict[str, Any]:
        from ..core.skills import save_skill

        name = (skill_name or "").strip().lower()
        description = b.get("description")
        content = b.get("content")
        if not isinstance(description, str) or not isinstance(content, str):
            raise HttpError(400, "description and content must be strings")
        # Same path the agent tool uses: validation + secret redaction +
        # atomic write, so a human edit can't land unredacted either.
        res = save_skill(name=name, description=description, content=content)
        if not res.get("ok"):
            raise HttpError(400, res.get("error") or "save failed")
        return {"ok": True, "name": name, "path": res["path"]}

    def h_skill_delete(self, q: Dict[str, str], b: Dict[str, Any], *,
                       skill_name: str) -> Dict[str, Any]:
        from ..core.skills import delete_skill

        res = delete_skill((skill_name or "").strip().lower())
        if not res.get("ok"):
            raise HttpError(404, res.get("error") or "delete failed")
        return res

    def h_playbook_promote(self, q: Dict[str, str],
                           b: Dict[str, Any]) -> Dict[str, Any]:
        from ..core.playbooks import promote

        pb_id = str(b.get("id") or "").strip()
        if not pb_id:
            raise HttpError(400, "id required")
        pb_store = self._playbook_store()
        try:
            # Dashboard promote is the EXPLICIT user confirmation path (no
            # judge gate here — --check lives in the CLI where an endpoint
            # can be resolved honestly).
            res = promote(pb_store, pb_id, check=False, actor="dashboard")
        finally:
            pb_store.close()
        if not res["ok"]:
            raise HttpError(404, res.get("error") or res.get("reason")
                            or "promote failed")
        return res

    def h_playbook_retire(self, q: Dict[str, str],
                          b: Dict[str, Any]) -> Dict[str, Any]:
        pb_id = str(b.get("id") or "").strip()
        if not pb_id:
            raise HttpError(400, "id required")
        pb_store = self._playbook_store()
        try:
            ok = pb_store.set_status(pb_id, "retired", actor="dashboard")
        finally:
            pb_store.close()
        if not ok:
            raise HttpError(404, f"no playbook {pb_id}")
        return {"ok": True, "id": pb_id, "status": "retired"}

    # ------------------------------------------------------------- datasets

    def h_datasets_list(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        store = self._store()
        try:
            datasets = Curater(store, self.home).list_datasets()
        finally:
            store.close()
        for m in datasets:
            self._attach_lint_info(m)
        return {"datasets": datasets}

    def _attach_lint_info(self, manifest: Dict[str, Any]) -> None:
        """Pass 4 #42: trust + flagged-row summary from the gate sidecars.

        `trust` defaults to "imported" on old manifests (matches the gate).
        `lint` is None for datasets without a flags.json sidecar — honest
        "unlinted" state; the panel offers a lint hint instead of numbers.
        Cheap: two small JSON reads per dataset, no row scans.
        """
        from ..curate.ingest_gate import dataset_trust, load_flags_sidecar

        ds_id = str(manifest.get("dataset_id", ""))
        if not ds_id or manifest.get("error"):
            return
        ds_dir = self.home / "datasets" / ds_id
        if not (ds_dir / "manifest.json").is_file():
            return
        manifest["trust"] = dataset_trust(ds_dir)
        flags = load_flags_sidecar(ds_dir)
        if flags is None:
            manifest["lint"] = None
            return
        per_row = flags.get("flags") or {}
        reasons: Dict[str, int] = {}
        for row_reasons in per_row.values():
            for r in row_reasons if isinstance(row_reasons, list) else []:
                key = str(r).split(":", 1)[0]
                reasons[key] = reasons.get(key, 0) + 1
        manifest["lint"] = {"flagged_rows": len(per_row), "reasons": reasons}

    def h_dataset_get(self, q: Dict[str, str], b: Dict[str, Any],
                      dataset_id: str) -> Dict[str, Any]:
        m = self.get_dataset(dataset_id)
        if m is None:
            raise HttpError(404, "no such dataset")
        self._attach_lint_info(m)
        return m

    def h_dataset_rows(self, q: Dict[str, str], b: Dict[str, Any],
                       dataset_id: str) -> Dict[str, Any]:
        fkey = (q.get("file") or "sft").lower()
        if fkey not in ("sft", "dpo"):
            raise HttpError(400, "file must be sft or dpo")
        offset = max(0, int(q.get("offset", "0") or 0))
        limit = min(200, max(1, int(q.get("limit", "25") or 25)))

        ds_dir = self.home / "datasets" / dataset_id
        path = ds_dir / f"{fkey}.jsonl"
        if not (ds_dir / "manifest.json").is_file():
            raise HttpError(404, "no such dataset")
        rows: List[Dict[str, Any]] = []
        total = 0
        truncated = False
        if path.is_file():
            with open(path, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if not line.strip():
                        continue
                    if i >= offset and len(rows) < limit:
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            row = {"error": "unparseable row"}
                        row, was_trunc = _truncate_row(row)
                        truncated = truncated or was_trunc
                        rows.append(row)
                    total += 1
        return {"file": fkey, "offset": offset, "limit": limit, "total": total,
                "truncated": truncated, "rows": rows}

    def h_dataset_diff(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        """Idea #6: diff two dataset versions by manifest + content hashes.

        GET /api/datasets/diff?a=<id>&b=<id>. Missing datasets -> 404,
        malformed manifests / missing params -> 400 (honest errors from the
        shared curate/dataset_diff module, also used by `sable datasets diff`).
        """
        from ..curate.dataset_diff import DatasetDiffError, diff_datasets

        a_id = str(q.get("a") or "").strip()
        b_id = str(q.get("b") or "").strip()
        if not a_id or not b_id:
            raise HttpError(400, "query params a and b (dataset ids) required")
        try:
            return diff_datasets(self.home / "datasets", a_id, b_id)
        except DatasetDiffError as e:
            raise HttpError(e.status, e.message) from e

    # ------------------------------------------------------------- analytics

    def h_analytics_curation(self, q: Dict[str, str],
                             b: Dict[str, Any]) -> Dict[str, Any]:
        """WP9a: curation funnel history + latest dataset diff summary.

        GET /api/analytics/curation[?dataset=<id>][&limit=N]. History is the
        append-only JSONL written by Curater.curate() after each run (see
        store/analytics.py). ``latest_diff`` reuses curate/dataset_diff over
        the two newest distinct dataset ids in the filtered history — the
        diff logic is never duplicated here; a missing/unreadable pair is
        reported honestly as {error} rather than failing the endpoint.
        """
        from ..curate.dataset_diff import DatasetDiffError, diff_datasets
        from ..store.analytics import read_curation_funnels

        dataset = str(q.get("dataset") or "").strip() or None
        try:
            limit = max(1, min(1000, int(q.get("limit", "200") or 200)))
        except ValueError:
            raise HttpError(400, "limit must be an integer") from None

        history = read_curation_funnels(self.home, dataset=dataset,
                                        limit=limit)
        # newest distinct dataset ids (oldest-first history -> reverse walk)
        ordered_ids: List[str] = []
        for rec in reversed(history):
            did = str(rec.get("dataset_id") or "")
            if did and did not in ordered_ids:
                ordered_ids.append(did)
        latest_diff: Optional[Dict[str, Any]] = None
        if len(ordered_ids) >= 2:
            b_id, a_id = ordered_ids[0], ordered_ids[1]  # newest pair, older=a
            try:
                latest_diff = diff_datasets(self.home / "datasets", a_id, b_id)
            except DatasetDiffError as e:
                latest_diff = {"error": e.message, "status": e.status}
        return {"dataset": dataset, "history": history,
                "latest_diff": latest_diff}

    def h_analytics_validation(self, q: Dict[str, str],
                               b: Dict[str, Any]) -> Dict[str, Any]:
        """WP9b: validation-gate history + promotion/rollback ledger events.

        GET /api/analytics/validation[?model=<id>][&limit=N]. Validation runs
        come from the append-only analytics JSONL (written by the training
        pipeline after each gate); promote/rollback events come from the
        hash-chained registry ledger. Both are merged chronologically into a
        single ``timeline`` so pass-rate trends and their promotion outcomes
        read as one story per model.
        """
        from ..store.analytics import read_validation_runs

        model = str(q.get("model") or "").strip() or None
        try:
            limit = max(1, min(1000, int(q.get("limit", "200") or 200)))
        except ValueError:
            raise HttpError(400, "limit must be an integer") from None

        runs = read_validation_runs(self.home, model=model, limit=limit)

        # Promotion/rollback events from the registry ledger (the durable
        # source of truth — already hash-chained; analytics never re-writes it).
        events: List[Dict[str, Any]] = []
        reg = self._registry()
        try:
            for op in reg.history(limit=1000):
                if op.get("op") not in ("promote", "rollback"):
                    continue
                if model and str(op.get("target_id") or "") != model:
                    continue
                events.append({
                    "ts": op.get("ts"),
                    "kind": "promotion",
                    "op": op.get("op"),
                    "model_id": op.get("target_id"),
                    "actor": op.get("actor"),
                    "intent": op.get("intent"),
                })
        finally:
            reg.close()

        timeline = sorted(
            [{"kind": "validation", **r} for r in runs] + events,
            key=lambda e: str(e.get("ts") or ""))
        return {"model": model, "history": runs, "events": events,
                "timeline": timeline[-limit:]}

    def h_dataset_export(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        dataset_id = str(b.get("dataset_id", ""))
        to = str(b.get("to", ""))
        if not dataset_id or not to:
            raise HttpError(400, "dataset_id and to required")
        # 5d containment guard: refuse destinations outside SABLE_HOME
        # unless the caller explicitly forces it (UI confirms first).
        force = bool(b.get("force"))
        if not force:
            home_r = self.home.expanduser().resolve(strict=False)
            dest_r = Path(to).expanduser().resolve(strict=False)
            if dest_r != home_r and home_r not in dest_r.parents:
                raise HttpError(
                    409,
                    f"destination {dest_r} is outside SABLE_HOME ({home_r}); "
                    "re-send with force=true to export anyway",
                    code="outside_home",
                )
        store = self._store()
        try:
            try:
                target = Curater(store, self.home).export_dataset(
                    dataset_id, Path(to))
            except FileNotFoundError as e:
                raise HttpError(404, str(e)) from e
            except FileExistsError as e:
                raise HttpError(409, str(e), code="target_exists") from e
        finally:
            store.close()
        return {"ok": True, "path": str(target)}

    def get_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        store = self._store()
        try:
            return Curater(store, self.home).load_manifest(dataset_id)
        finally:
            store.close()

    # --------------------------------------------------------------- models

    def h_model_catalog(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        """Models discovered on disk under the base model folder (config
        `models_dir` / SABLE_MODELS_DIR / HF hub cache fallback). Optional
        ?dir= overrides the folder for this request only."""
        from ..core.model_discovery import discover_models

        return discover_models(q.get("dir") or self.cfg.models_dir)

    # ------------------------------------------------------- vision (mmproj)
    # Catalog pattern: the user never types an mmproj path. We detect a
    # projector GGUF next to the model, or offer to download the companion
    # from the model's hub repo into that same folder. The UI renders a load
    # toggle (run_settings.vision) when one exists.

    def h_mmproj_status(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        """Vision-projector state for one GGUF: {gguf, local, repo_id,
        remote: [{rfilename,size}], downloadable}."""
        from ..core import mmproj

        gguf = (q.get("gguf") or "").strip()
        if not gguf:
            raise HttpError(400, "gguf query param required")
        from pathlib import Path as _P

        if not _P(gguf).expanduser().is_file():
            raise HttpError(404, f"GGUF not found: {gguf}")
        local = mmproj.find_local(gguf)
        repo_id = mmproj.source_repo_for(gguf)
        remote: list = []
        if local is None and repo_id:
            try:
                remote = mmproj.remote_mmproj_candidates(repo_id)
            except Exception:
                remote = []  # hub unreachable: UI just hides the download button
        return {"gguf": gguf, "local": local, "repo_id": repo_id,
                "remote": remote, "downloadable": local is None and bool(remote)}

    def h_mmproj_download(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        """Download the mmproj companion for a GGUF into the GGUF's own folder
        (background thread; poll GET /api/mmproj?gguf=... until local)."""
        from ..core import mmproj

        gguf = str(b.get("gguf") or "").strip()
        if not gguf:
            raise HttpError(400, "gguf required")
        rfilename = str(b.get("rfilename") or "").strip() or None
        repo_id = str(b.get("repo_id") or "").strip() or None

        def _run() -> None:
            try:
                info = mmproj.download_for(gguf, repo_id=repo_id,
                                           rfilename=rfilename)
                self.bus.publish("mmproj.downloaded", {"gguf": gguf, **info})
            except Exception as e:
                self.bus.publish("mmproj.error",
                                 {"gguf": gguf, "error": str(e)})

        threading.Thread(target=_run, daemon=True, name="mmproj-dl").start()
        return {"ok": True, "note": "mmproj download started; poll "
                                    "GET /api/mmproj?gguf= for completion"}

    # ------------------------------------------------------------- model hub
    # Search / download / manage remote models (core/modelhub.py). One shared
    # ModelHub per dashboard process so background download jobs + progress
    # survive across requests (download-registry pattern).

    def _modelhub(self):
        from ..core.modelhub import ModelHub

        hub = getattr(self, "_hub", None)
        if hub is None:
            hub = ModelHub(models_dir=self.cfg.models_dir, home=self.home)
            self._hub = hub
        return hub

    @staticmethod
    def _hub_call(fn, *args, **kwargs):
        """Run a hub op, translating HubError -> HttpError for the envelope."""
        from ..core.modelhub import HubError

        try:
            return fn(*args, **kwargs)
        except HubError as e:
            msg = str(e)
            status = 404 if "not found" in msg else \
                401 if ("authorization" in msg or "HF_TOKEN" in msg) else 400
            raise HttpError(status, msg) from e

    def h_hub_search(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        limit = 30
        try:
            limit = int(q.get("limit", "30"))
        except ValueError:
            pass
        results = self._hub_call(
            self._modelhub().search, q.get("q", ""), limit,
            gguf_only=q.get("gguf", "") in ("1", "true", "yes"),
            capability=q.get("capability", ""),
            sort=q.get("sort", "downloads"))
        return {"query": q.get("q", ""), "results": results}

    def h_hub_feed(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        limit = 15
        try:
            limit = int(q.get("limit", "15"))
        except ValueError:
            pass
        return {"models": self._hub_call(self._modelhub().feed, limit)}

    def h_hub_detail(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        repo = q.get("repo", "")
        return self._hub_call(self._modelhub().detail, repo)

    def h_hub_image(self, q: Dict[str, str], b: Dict[str, Any]) -> ApiResponse:
        """Same-origin image proxy for model cards (CSP is default-src 'self').
        ?repo=<id>&path=<repo-relative> or ?url=<allow-listed https url>."""
        from ..core.modelhub import fetch_image

        data, ctype = self._hub_call(
            fetch_image, repo_id=q.get("repo", ""), path=q.get("path", ""),
            url=q.get("url", ""))
        return ApiResponse(200, raw=data, content_type=ctype)

    def h_hub_variants(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        repo = q.get("repo", "")
        return self._hub_call(self._modelhub().variants, repo)

    def h_hub_installed(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        hub = self._modelhub()
        hub.prune_finished()
        return {"models_dir": str(hub.base_dir()),
                "installed": self._hub_call(hub.installed),
                "downloads": hub.jobs()}

    def h_hub_download(self, q: Dict[str, str], b: Dict[str, Any]) -> ApiResponse:
        # 202 + poll (studio contract): the worker thread writes the HF hub
        # cache layout; progress via GET /api/hub/downloads/<job_id>.
        repo = str(b.get("repo") or "")
        variant = str(b.get("variant") or "")
        job_id = self._hub_call(self._modelhub().start_download, repo, variant)
        return ApiResponse(202, {"ok": True, "job_id": job_id,
                                 "state": "queued"})

    def h_hub_downloads(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        return {"downloads": self._modelhub().jobs()}

    def h_hub_download_get(self, q: Dict[str, str], b: Dict[str, Any],
                           job_id: str) -> Dict[str, Any]:
        return self._hub_call(lambda: self._modelhub().job(job_id).to_dict())

    def h_hub_cancel(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        job_id = str(b.get("job_id") or "")
        return self._hub_call(self._modelhub().cancel, job_id)

    def h_hub_delete(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        repo = str(b.get("repo") or "")
        out = self._hub_call(self._modelhub().delete, repo)
        self.bus.publish("model.deleted", out)
        return out

    def h_models_list(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        reg = self._registry()
        try:
            models = [m.to_dict() for m in reg.list_models()]
            active = reg.active_model()
            jobs = [j.to_dict() for j in reg.list_jobs()]
        finally:
            reg.close()
        # Lineage edges for the UI graph: parent_adapter -> model_id.
        edges = [
            {"from": m["parent_adapter"], "to": m["model_id"]}
            for m in models
            if m.get("parent_adapter")
        ]
        # Backlog item #15: catalog-style join enrichment (additive —
        # existing keys unchanged). Dataset title from the manifest, final step
        # count + duration from the model's last finished train job, losses
        # already carried on the row. `detail` is composed server-side so the
        # panel renders one string; every missing piece is omitted tolerantly.
        train_jobs: Dict[str, Dict[str, Any]] = {}
        for j in jobs:
            if j.get("kind") in ("sft", "dpo", "grpo") and j.get("model_id"):
                train_jobs[j["model_id"]] = j  # later (finished) rows win
        for m in models:
            job = train_jobs.get(m["model_id"])
            title = None
            ds_id = m.get("dataset_id")
            if ds_id:
                mf = self.home / "datasets" / ds_id / "manifest.json"
                try:
                    man = json.loads(mf.read_text())
                    title = man.get("title") or man.get("dataset_id") or ds_id
                except (OSError, ValueError):
                    title = ds_id  # manifest missing/unreadable -> id itself
            steps = _job_final_steps(job.get("log_path")) if job else None
            duration = _job_duration_s(job) if job else None
            m["dataset_title"] = title
            m["train_steps"] = steps
            m["train_duration_s"] = duration
            # All rows here have a registry row (local GGUFs are not listed by
            # this endpoint), so grouping is trivially "trained" — kept as an
            # explicit field for the panel's trained-vs-local grouping.
            m["origin"] = "trained"
            m["detail"] = _model_detail_line(
                title=title, steps=steps, created_at=m.get("created_at"),
                train_loss=m.get("train_loss"), eval_loss=m.get("eval_loss"))
        # Pass 4 #41: derived capability tier per model — computed at read
        # time from evidence (registry/tiers.py), never stored. Display
        # context only; promote/rollback stay permissive.
        self._attach_tiers(models)
        return {
            "models": models,
            "edges": edges,
            "active_model": active.model_id if active else None,
        }

    def _attach_tiers(self, models: list) -> None:
        """Add tier/tier_name/tier_hint to each model dict (pass 4 #41)."""
        from ..registry.tiers import compute_served_stats, derive_tier, \
            requirements_hint, tier_name

        store = self._store()
        try:
            for m in models:
                since = m.get("promoted_at") or m.get("created_at")
                stats = compute_served_stats(
                    store, str(m.get("model_id") or ""), since=since)
                t = derive_tier(m, stats)
                m["tier"] = t
                m["tier_name"] = tier_name(t)
                m["tier_hint"] = requirements_hint(m, stats)
        finally:
            store.close()

    def h_model_get(self, q: Dict[str, str], b: Dict[str, Any],
                    model_id: str) -> Dict[str, Any]:
        reg = self._registry()
        try:
            m = reg.get_model(model_id)
        finally:
            reg.close()
        if m is None:
            raise HttpError(404, f"no model {model_id}")
        d = m.to_dict()
        return {
            "model": d,
            "validation": d.get("validation"),
            "adapter_path": d.get("adapter_dir"),
            "gguf_path": d.get("gguf_path"),
        }

    def h_model_promote(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        model_id = str(b.get("model_id", ""))
        if not model_id:
            raise HttpError(400, "model_id required")
        # Pass 4 #36: optional one-line intent ("promoted X expecting Y");
        # ledgered with the op and reviewable via insights intent_review.
        intent = b.get("intent") or None
        reg = self._registry()
        try:
            try:
                m = reg.promote(model_id, actor="dashboard", intent=intent)
            except Exception as e:
                raise HttpError(409, str(e)) from e
        finally:
            reg.close()
        self.bus.publish("model.promoted", {"model_id": m.model_id,
                                            "base_model": m.base_model})
        return {"ok": True, "model": m.to_dict()}

    def h_model_rollback(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        base = b.get("base_model") or None
        intent = b.get("intent") or None  # pass 4 #36 optional "why"
        reg = self._registry()
        try:
            m = reg.rollback(base_model=base, actor="dashboard", intent=intent)
        finally:
            reg.close()
        self.bus.publish("model.rolled_back",
                         {"model_id": m.model_id if m else None})
        return {"ok": True, "active_model": m.to_dict() if m else None}

    # ----------------------------------------------------------------- jobs

    def h_jobs_list(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        tail = int(q.get("tail", str(DEFAULT_JOB_LOG_TAIL)) or 0)
        reg = self._registry()
        try:
            jobs = [j.to_dict() for j in reg.list_jobs()]
        finally:
            reg.close()
        if tail > 0:
            for j in jobs:
                lp = j.get("log_path")
                if lp and Path(lp).is_file():
                    try:
                        lines = Path(lp).read_text(errors="replace").splitlines()
                        j["log_tail"] = lines[-tail:]
                    except OSError:
                        j["log_tail"] = []
        return {"jobs": jobs}

    def h_job_get(self, q: Dict[str, str], b: Dict[str, Any],
                  job_id: str) -> Dict[str, Any]:
        reg = self._registry()
        try:
            job = reg.get_job(job_id)
        finally:
            reg.close()
        if job is None:
            raise HttpError(404, f"no job {job_id}")
        offset = max(0, int(q.get("offset", "0") or 0))
        limit = min(5000, max(1, int(q.get("limit", "500") or 500)))
        lines: List[str] = []
        total = 0
        if job.log_path and Path(job.log_path).is_file():
            try:
                all_lines = Path(job.log_path).read_text(errors="replace").splitlines()
                total = len(all_lines)
                lines = all_lines[offset:offset + limit]
            except OSError:
                pass
        return {"job": job.to_dict(),
                "log": {"lines": lines, "offset": offset, "total": total}}

    def h_job_log(self, q: Dict[str, str], b: Dict[str, Any],
                  job_id: str) -> Dict[str, Any]:
        tail = int(q.get("tail", str(DEFAULT_JOB_LOG_TAIL)) or 0)
        reg = self._registry()
        try:
            job = reg.get_job(job_id)
        finally:
            reg.close()
        if job is None:
            raise HttpError(404, f"no job {job_id}")
        lines: List[str] = []
        if job.log_path and Path(job.log_path).is_file():
            try:
                all_lines = Path(job.log_path).read_text(errors="replace").splitlines()
                lines = all_lines[-tail:] if tail > 0 else all_lines
            except OSError:
                pass
        return {"job_id": job_id, "state": job.state, "lines": lines}

    def h_job_cancel(self, q: Dict[str, str], b: Dict[str, Any]) -> ApiResponse:
        job_id = str(b.get("job_id", ""))
        if not job_id:
            raise HttpError(400, "job_id required")
        reg = self._registry()
        try:
            sched = self._scheduler(reg)
            job = sched.cancel(job_id, actor="dashboard")
        finally:
            reg.close()
        if job is None:
            raise HttpError(404, f"no job {job_id}")
        return ApiResponse(200, {"ok": True, "job": job.to_dict()})

    # -------------------------------------------------------------- actions

    def h_curate(self, q: Dict[str, str], b: Dict[str, Any]) -> ApiResponse:
        # 5d contract change: curate is now a queued job (kind="curate"), not
        # a synchronous request — large buffers must not block the HTTP thread.
        # 202 {ok, job_id, state:"queued"}; follow via job.state events or
        # GET /api/jobs/<id>. On success the job log carries a final line:
        #   {"event": "done", "job": <id>, "result": {dataset_id, sft_count,
        #    dpo_count, claimed, warnings}}
        return ApiResponse(202, self.submit_curate(b))

    def submit_curate(self, b: Dict[str, Any]) -> Dict[str, Any]:
        # Validate filters synchronously so bad bodies still get an immediate
        # 400 before anything is queued.
        self._curation_cfg(b)

        reg = self._registry()
        try:
            sched = self._scheduler(reg)
            job = sched.submit("curate", log_dir=self.home / "curate_runs",
                               params=dict(b))
        finally:
            reg.close()

        def _run(job_id: str, log_path: Optional[str]) -> None:
            # In-process execution (no worker subprocess): curate touches only
            # the buffer DB + filesystem, so it neither needs the GPU nor takes
            # the scheduler's single-flight slot.
            reg = self._registry()
            try:
                if not reg.mark_running(job_id, os.getpid(), actor="dashboard"):
                    return  # cancelled while queued — keep the terminal state
                self._job_observer(job_id, "running")
                try:
                    res = self.curate(b)
                except Exception as e:
                    msg = f"{type(e).__name__}: {e}"
                    if log_path:
                        try:
                            with open(log_path, "a", encoding="utf-8") as f:
                                f.write(json.dumps(
                                    {"event": "error", "job": job_id,
                                     "error": msg}) + "\n")
                        except OSError:
                            pass
                    reg.finish_job(job_id, "error", error=msg, actor="dashboard")
                    self._job_observer(job_id, "error")
                    return
                if log_path:
                    try:
                        with open(log_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(
                                {"event": "done", "job": job_id,
                                 "result": {k: res.get(k) for k in
                                            ("dataset_id", "sft_count",
                                             "dpo_count", "claimed",
                                             "short_claims", "warnings")}},
                                ensure_ascii=False) + "\n")
                    except OSError:
                        pass
                reg.finish_job(job_id, "done", actor="dashboard")
                self._job_observer(job_id, "done")
            except Exception:  # pragma: no cover - defensive
                try:
                    reg.finish_job(job_id, "error", error="curate runner failed",
                                   actor="dashboard")
                except Exception:
                    pass
            finally:
                reg.close()

        threading.Thread(target=_run, args=(job.job_id, job.log_path),
                         daemon=True).start()
        return {"ok": True, "job_id": job.job_id, "state": "queued"}

    def curate(self, b: Dict[str, Any]) -> Dict[str, Any]:
        cc = self._curation_cfg(b)
        store = self._store()
        try:
            curater = Curater(store, self.home)
            res = curater.curate(cc, name=str(b.get("name") or "ui"),
                                 rebuild_id=b.get("rebuild") or None)
        finally:
            store.close()
        return {
            "ok": True,
            "dataset_id": res.dataset_id,
            "sft_count": res.sft_count,
            "dpo_count": res.dpo_count,
            "claimed": res.claimed,
            "short_claims": res.short_claims,
            "warnings": res.warnings,
        }

    def _curation_cfg(self, b: Dict[str, Any]) -> CurationConfig:
        cc = curation_config_from_yaml({"curation": self.cfg.curation or {}})
        if b.get("min_score") is not None:
            cc.min_score = float(b["min_score"])
        if b.get("since"):
            cc.since = str(b["since"])
        if b.get("profile"):
            cc.agent_profile = str(b["profile"])
        if b.get("project"):
            cc.project = str(b["project"])
        if b.get("include_failures"):
            cc.include_failures = True
        if b.get("model_alias"):
            cc.model_alias = str(b["model_alias"])
        if b.get("formats"):
            fmt = str(b["formats"]).lower()
            if fmt not in ("sft", "dpo", "both"):
                raise HttpError(400, "formats must be sft, dpo or both")
            cc.formats = fmt
        if b.get("limit") is not None:
            cc.limit = int(b["limit"])
        if self.cfg.gguf_path and not cc.base_model:
            cc.base_model = self.cfg.gguf_path
        return cc

    def h_train(self, q: Dict[str, str], b: Dict[str, Any]) -> ApiResponse:
        return ApiResponse(202, self.submit_train(b))

    def submit_train(self, b: Dict[str, Any]) -> Dict[str, Any]:
        dataset_id = str(b.get("dataset") or b.get("dataset_id") or "")
        if not dataset_id:
            raise HttpError(400, "dataset required")
        kind = str(b.get("kind") or "sft")
        if kind not in ("sft", "dpo", "grpo"):
            raise HttpError(400, "kind must be sft, dpo or grpo")
        dry_run = bool(b.get("dry_run"))

        tc = self._training_config()
        # Per-request base-model override (the Train wizard's discovered-models
        # dropdown) beats the configured training.base_model.
        if b.get("base_model"):
            tc.base_model = str(b["base_model"])
        reg = self._registry()
        max_par = max(1, int(tc.max_parallel_jobs or 1))
        try:
            # Capacity pre-check (5d) — ADVISORY only. A dead worker's phantom
            # `running` row must not hold every future submit at 409, so reap
            # stale rows before counting; and because the check races any other
            # submitter, the AUTHORITATIVE capacity decision is the atomic
            # claim + device-slot acquisition inside the execution thread (and
            # in the dispatcher). A submit that slips past a just-raced check
            # stays queued and runs when capacity frees — never rejected,
            # never stranded without explanation.
            self._scheduler(reg).reap_stale()
            running = reg.running_jobs()
            if len(running) >= max_par:
                raise HttpError(
                    409,
                    f"job {running[0].job_id} is already running "
                    f"({running[0].kind}); one GPU, one job"
                    if max_par == 1 else
                    f"{len(running)} jobs already running "
                    "(training.max_parallel_jobs reached)",
                    code="already_running",
                )
            sched = self._scheduler(reg)
            job = sched.submit(
                kind, dataset_id=dataset_id, log_dir=self.home / "train_runs",
                params={
                    "train_yaml": str(self.home / "datasets" / dataset_id / "train.yaml"),
                    "dataset_dir": str(self.home / "datasets" / dataset_id),
                    "base_model": tc.base_model or "",
                    "dry_run": dry_run,
                },
            )
        finally:
            reg.close()

        def _run(job_id: str) -> None:
            # Ownership-safe start: claim THIS job with an atomic queued ->
            # running flip under our pid (guarded UPDATE — a losing claim can
            # never double-run it). When capacity is full the job simply stays
            # queued; this thread's drain below picks it up when capacity
            # frees, and the dispatcher tick is the backstop for anything that
            # arrives after this thread exits.
            reg = self._registry()
            try:
                sched = self._scheduler(reg)
                j = reg.get_job(job_id)
                if (j is not None and j.state == "queued"
                        and len(reg.running_jobs()) < max_par):
                    if reg.mark_running(job_id, os.getpid(), actor="dashboard"):
                        try:
                            sched.run_claimed(reg.get_job(job_id) or j)
                        except Exception:
                            # Gate/slot refusal requeues the claim; job state
                            # already reflects any real failure.
                            pass
                # Holder duty: drain remaining queued work instead of assuming
                # another execution will — nothing else would until the next
                # dispatcher tick.
                try:
                    sched.drain_queue()
                except Exception:
                    pass
            except Exception:  # pragma: no cover - defensive
                logger.exception("train execution thread failed for %s", job_id)
            finally:
                reg.close()

        threading.Thread(target=_run, args=(job.job_id,), daemon=True).start()
        return {"ok": True, "job_id": job.job_id, "state": "queued"}

    def h_rft(self, q: Dict[str, str], b: Dict[str, Any]) -> ApiResponse:
        return ApiResponse(202, self.submit_rft(b))

    def submit_rft(self, b: Dict[str, Any]) -> Dict[str, Any]:
        """Queue an rft (rejection-sampling) job through the SAME atomic submit
        path as submit_train — reap pre-check, sched.submit, then a holder
        thread that claims (mark_running) + runs in-process and drains. The
        rft master switch (rft.enabled) is enforced by the job runner itself,
        so a disabled config refuses at run time with an honest job error."""
        tc = self._training_config()
        reg = self._registry()
        max_par = max(1, int(tc.max_parallel_jobs or 1))
        try:
            # Advisory capacity pre-check (identical semantics to submit_train):
            # reap phantoms first; the authoritative decision is the atomic
            # claim inside the execution thread / dispatcher.
            self._scheduler(reg).reap_stale()
            running = reg.running_jobs()
            if len(running) >= max_par:
                raise HttpError(
                    409,
                    f"job {running[0].job_id} is already running "
                    f"({running[0].kind}); one GPU, one job"
                    if max_par == 1 else
                    f"{len(running)} jobs already running "
                    "(training.max_parallel_jobs reached)",
                    code="already_running",
                )
            params: Dict[str, Any] = {}
            for k in ("seeds", "goldens", "golden_file", "name"):
                if b.get(k):
                    params[k] = b[k]
            for k in ("sample_k", "max_prompts"):
                if b.get(k) is not None:
                    params[k] = int(b[k])
            for k in ("judge_min_score", "temperature"):
                if b.get(k) is not None:
                    params[k] = float(b[k])
            sched = self._scheduler(reg)
            job = sched.submit("rft", log_dir=self.home / "train_runs",
                               params=params, enqueue_actor="dashboard")
        finally:
            reg.close()

        def _run(job_id: str) -> None:
            # Ownership-safe start — same shape as submit_train's thread: claim
            # THIS job with an atomic queued -> running flip under our pid; the
            # scheduler runs it IN-PROCESS (kind rft, like shadow/evolve). A
            # gate/slot refusal keeps the job queued for the dispatcher tick.
            reg = self._registry()
            try:
                sched = self._scheduler(reg)
                j = reg.get_job(job_id)
                if (j is not None and j.state == "queued"
                        and len(reg.running_jobs()) < max_par):
                    if reg.mark_running(job_id, os.getpid(), actor="dashboard"):
                        try:
                            sched.run_claimed(reg.get_job(job_id) or j)
                        except Exception:
                            pass  # refusal requeued the claim; state is truth
                try:
                    sched.drain_queue()
                except Exception:
                    pass
            except Exception:  # pragma: no cover - defensive
                logger.exception("rft execution thread failed for %s", job_id)
            finally:
                reg.close()

        threading.Thread(target=_run, args=(job.job_id,), daemon=True).start()
        return {"ok": True, "job_id": job.job_id, "state": "queued"}

    def h_mine_goldens(self, q: Dict[str, str], b: Dict[str, Any]) -> ApiResponse:
        return ApiResponse(202, self.submit_mine_goldens(b))

    def submit_mine_goldens(self, b: Dict[str, Any]) -> Dict[str, Any]:
        """Queue a mine_goldens job through the SAME atomic submit path as
        submit_rft (reap pre-check, sched.submit, holder thread that claims +
        runs in-process and drains). The goldens_mine master switch is enforced
        by the job runner itself, so a disabled config refuses at run time with
        an honest job error."""
        tc = self._training_config()
        reg = self._registry()
        max_par = max(1, int(tc.max_parallel_jobs or 1))
        try:
            self._scheduler(reg).reap_stale()
            running = reg.running_jobs()
            if len(running) >= max_par:
                raise HttpError(
                    409,
                    f"job {running[0].job_id} is already running "
                    f"({running[0].kind}); one GPU, one job"
                    if max_par == 1 else
                    f"{len(running)} jobs already running "
                    "(training.max_parallel_jobs reached)",
                    code="already_running",
                )
            params: Dict[str, Any] = {}
            if b.get("max_per_run") is not None:
                params["max_per_run"] = int(b["max_per_run"])
            if b.get("use_judge"):
                params["use_judge"] = True
            sched = self._scheduler(reg)
            job = sched.submit("mine_goldens", log_dir=self.home / "train_runs",
                               params=params, enqueue_actor="dashboard")
        finally:
            reg.close()

        def _run(job_id: str) -> None:
            reg = self._registry()
            try:
                sched = self._scheduler(reg)
                j = reg.get_job(job_id)
                if (j is not None and j.state == "queued"
                        and len(reg.running_jobs()) < max_par):
                    if reg.mark_running(job_id, os.getpid(), actor="dashboard"):
                        try:
                            sched.run_claimed(reg.get_job(job_id) or j)
                        except Exception:
                            pass  # refusal requeued the claim; state is truth
                try:
                    sched.drain_queue()
                except Exception:
                    pass
            except Exception:  # pragma: no cover - defensive
                logger.exception("mine_goldens execution thread failed for %s",
                                 job_id)
            finally:
                reg.close()

        threading.Thread(target=_run, args=(job.job_id,), daemon=True).start()
        return {"ok": True, "job_id": job.job_id, "state": "queued"}

    def h_train_estimate(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        """Predicted VRAM footprint for a would-be training job (#20).

        GET /api/train/estimate?dataset=<id>&base_model=<path> — the Train
        wizard calls this before submit so the user sees estimate + headroom.
        ``estimated: null`` when the base model's dims are not resolvable
        locally (HF id without cache, GGUF, missing train.yaml): the scheduler
        then falls back to the static floor, same as before.
        """
        from ..core.vram import estimate_from_train_yaml
        from ..training.scheduler import gpu_free_mib

        dataset_id = str(q.get("dataset") or q.get("dataset_id") or "")
        if not dataset_id:
            raise HttpError(400, "dataset required")
        train_yaml = self.home / "datasets" / dataset_id / "train.yaml"
        base_override = str(q.get("base_model") or "").strip() or None
        tc = self._training_config()
        est = estimate_from_train_yaml(
            str(train_yaml), base_override or tc.base_model or None)
        free = gpu_free_mib()
        out: Dict[str, Any] = {
            "estimated": None, "free_mib": free,
            "fits": None, "headroom_mib": None,
        }
        if est is not None:
            out["estimated"] = est.as_dict()
            if free is not None:
                out["headroom_mib"] = free - est.total_mib
                out["fits"] = free >= est.total_mib
        return out

    def h_train_check(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        from ..training.scheduler import resource_gate_ok, should_auto_train

        tc = self._training_config()
        store = self._store()
        try:
            d = should_auto_train(store, min_traces=tc.auto_min_traces,
                                  min_score=tc.auto_min_score)
        finally:
            store.close()
        # 5d: surface the scheduler resource gate (VRAM / AC power) alongside
        # the buffer thresholds so the Train + Hardware panels can explain
        # *why* training would not start.
        gate = resource_gate_ok(tc)
        return {
            "enough_traces": bool(d["trigger"]),
            "unclaimed": d["unclaimed_traces"],
            "mean_score": d["mean_score"],
            "threshold": {"min_traces": d["min_traces"],
                          "min_score": d["min_score"]},
            "gate": gate,
        }

    # ------------------------------------------------------- easy training
    # One-profile turnkey loop (training/easy.py): the user picks a profile
    # already bound to a server instance; enabling writes the optimal preset
    # into curation:/training: and turns on profile-stamped trace recording.
    # From then on chat with that profile feeds the buffer, this page tells
    # them when training is due, and "Train now" runs the full iterate loop
    # scoped to the profile — optionally deploying the promoted model back
    # onto the profile's server.

    def _easy_cfg(self) -> EasyTrainingConfig:
        return EasyTrainingConfig.from_yaml(
            (self.cfg.easy_training if isinstance(self.cfg.easy_training, dict)
             else {}) or {})

    def _easy_readiness(self, ec: EasyTrainingConfig) -> Dict[str, Any]:
        from ..training.scheduler import resource_gate_ok

        tc = self._training_config()
        gate = resource_gate_ok(tc)
        store = self._store()
        try:
            return evaluate_readiness(store, ec, tc, gate)
        finally:
            store.close()

    def _easy_deploy(self, model_id: str, server_instance: str) -> Dict[str, Any]:
        """Hot-swap the easy profile's bound server to a new GGUF (best-effort).

        Spawn instances are restarted on the promoted model's GGUF (the
        instance definition is updated first so the swap survives restarts);
        attach instances are left alone — an external server is not ours to
        re-point. Never raises: a failed deploy is reported, not fatal, since
        the model is already promoted and the loop's data is intact.
        """
        out: Dict[str, Any] = {"attempted": True, "ok": False,
                               "server": server_instance, "model_id": model_id,
                               "detail": None}
        try:
            reg = self._registry()
            try:
                m = reg.get_model(model_id)
            finally:
                reg.close()
            if m is None or not m.gguf_path:
                out["detail"] = "promoted model has no GGUF on disk"
                return out
            inst = srv_inst.get_instance(self.home, server_instance,
                                         cfg=self.cfg)
            if inst is None:
                out["detail"] = f"server instance {server_instance} vanished"
                return out
            if inst.get("attach_url"):
                out["detail"] = ("attached server is external — deploy "
                                 f"{model_id} there manually")
                return out
            # Update the definition, then bounce the running process (or just
            # record the new gguf when it wasn't running).
            inst = dict(inst)
            inst["gguf"] = str(m.gguf_path)
            srv_inst.save_instance(self.home, inst)
            st = self._instance_state(server_instance)
            if srv_inst.state_alive(st):
                self._stop_instance(server_instance)
                start = self.handle("POST",
                                    f"/api/servers/{server_instance}/start",
                                    body={"force": True})
                # A failed restart must NOT report success: the definition is
                # updated (so the next manual start uses the new GGUF), but
                # the profile's server is down right now — say so.
                if getattr(start, "status", 500) >= 400:
                    err = {}
                    if isinstance(getattr(start, "body", None), dict):
                        err = start.body.get("error") or {}
                    out["detail"] = (
                        f"definition updated to {model_id} but restart failed"
                        + (f": {err.get('message')}" if err.get("message")
                           else " — start it from the Serve panel"))
                    return out
            out["ok"] = True
            out["detail"] = f"{server_instance} now serves {model_id}"
            self.bus.publish("easy_training.deployed",
                             {"model_id": model_id, "server": server_instance})
        except Exception as e:  # deploy is best-effort by design
            out["detail"] = f"{type(e).__name__}: {e}"
        return out

    def _easy_after_run(self, run_id: str, record: Dict[str, Any],
                        ec: EasyTrainingConfig,
                        profile: Dict[str, Any]) -> None:
        """Easy-loop completion hook: ledger the result and auto-deploy.

        Called from the iterate worker thread AFTER the run record is saved
        (wired through submit_iterate's on_done)."""
        try:
            result = record.get("result") or {}
            state = record.get("state")
            patch: Dict[str, Any] = {
                "last_run": {
                    "run_id": run_id, "state": state,
                    "finished": record.get("finished"),
                    "model_id": result.get("model_id"),
                    "dataset_id": result.get("dataset_id"),
                    "promoted": bool(result.get("promoted")),
                },
            }
            deploy_info: Optional[Dict[str, Any]] = None
            if (state == "done" and result.get("promoted")
                    and ec.auto_deploy and result.get("model_id")):
                si = str(profile.get("server_instance") or "").strip()
                if si:
                    deploy_info = self._easy_deploy(str(result["model_id"]), si)
                    patch["last_run"]["deploy"] = deploy_info
            easy_save_state(self.home, patch)
            self.bus.publish("easy_training.run.finished", {
                "run_id": run_id, "state": state,
                "profile": ec.profile,
                "model_id": result.get("model_id"),
                "promoted": bool(result.get("promoted")),
                "deploy": deploy_info,
            })
        except Exception:  # completion bookkeeping must never break the run
            logger.exception("easy training: completion hook failed for %s",
                             run_id)

    def _easy_agent_entry(self, ec: EasyTrainingConfig, name: str,
                          profiles: List[Dict[str, Any]],
                          servers: Dict[str, Any],
                          run_recs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Per-agent view for one profile in the easy loop.

        Same shape as the legacy top-level fields (profile/server/problem/
        readiness/active_run/runs) so the panel renders each agent card with
        the code it already has; `name` lets it key them apart."""
        match = next((p for p in profiles if p.get("name") == name), None)
        problem = None
        server = None
        if match is None:
            problem = (f"configured profile “{name}” no longer exists — "
                       "remove it or pick one below")
        else:
            si = str(match.get("server_instance") or "").strip()
            server = servers.get(si)
            if not si:
                problem = ("the configured profile is not bound to a "
                           "server instance")
        server_state = (self._instance_state(str(server.get("id")))
                        if server else None)
        # Readiness scoped to THIS agent (single-profile projection).
        readiness = (self._easy_readiness(ec.for_profile(name))
                     if ec.enabled and match else None)
        runs = [r for r in run_recs if r.get("name") == f"easy:{name}"]
        active = next((r for r in runs if r.get("state") == "running"), None)
        return {
            "profile": name,
            "bound": match is not None,
            "server": ({"id": server.get("id"), "name": server.get("name"),
                        "model_name": (server_state or {}).get("model_name")
                        or server.get("model_name"),
                        "running": srv_inst.state_alive(server_state)}
                       if server else None),
            "problem": problem,
            "readiness": readiness,
            "active_run": active,
            "runs": runs[:10],
        }

    def h_easy_get(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        ec = self._easy_cfg()
        profiles, _default = self._profiles()
        servers = {str(i.get("id")): i
                   for i in srv_inst.load_instances(self.home, cfg=self.cfg)}
        # All iterate records once (newest first); per-agent views filter by
        # their `easy:<profile>` run name without re-reading the runs dir.
        run_recs: List[Dict[str, Any]] = []
        runs_dir = self.home / ITERATE_RUNS_DIRNAME
        if runs_dir.is_dir():
            for f in sorted(runs_dir.glob("iter-*.json"), reverse=True)[:40]:
                try:
                    run_recs.append(json.loads(f.read_text()))
                except (OSError, json.JSONDecodeError):
                    continue
        agents = [self._easy_agent_entry(ec, n, profiles, servers, run_recs)
                  for n in ec.all_profiles]
        # Legacy top-level fields mirror the PRIMARY agent (ec.profile, or the
        # first configured one) so existing readers keep working unchanged.
        primary_name = ec.profile or (ec.all_profiles[0]
                                      if ec.all_profiles else None)
        primary = next((a for a in agents
                        if a["profile"] == primary_name), None)
        prof = server = readiness = problem = None
        active = None
        runs: List[Dict[str, Any]] = []
        if primary:
            problem = primary["problem"]
            server = primary["server"]
            readiness = primary["readiness"]
            active = primary["active_run"]
            runs = primary["runs"]
            if primary["bound"]:
                prof = next((p for p in profiles
                             if p.get("name") == primary_name), None)
        # Stale stamps: unclaimed traces whose agent_profile matches no known
        # profile (typical after a profile rename — old chats keep the old
        # name and silently stop counting toward readiness). 'sable' is the
        # legacy backend stamp (pre-easy traces) and never stale. Reported so
        # the panel can nudge instead of showing an unexplained zero gauge.
        stale_stamps: List[Dict[str, Any]] = []
        if ec.enabled and agents:
            try:
                names = {str(p.get("name")) for p in profiles}
                store = self._store()
                try:
                    counts: Dict[str, int] = {}
                    for r in store.query_for_curation(include_curated=False):
                        ap = str(r.agent_profile or "").strip()
                        if ap and ap not in names and ap != "sable":
                            counts[ap] = counts.get(ap, 0) + 1
                    stale_stamps = [{"profile": k, "traces": v} for k, v
                                    in sorted(counts.items())]
                finally:
                    store.close()
            except Exception:  # purely informational — never break GET
                stale_stamps = []
        state = easy_load_state(self.home)
        return {
            "config": ec.as_dict(),
            "agents": agents,
            "profile": prof,
            "server": server,
            "problem": problem,
            "stale_stamps": stale_stamps,
            "readiness": readiness,
            "active_run": active,
            "runs": runs[:10],
            "state": state,
        }

    def _easy_preset_diff(self) -> List[Dict[str, Any]]:
        """What the optimal preset WOULD change in curation:/training:/
        validation:/goldens_mine:, as [{section, key, from, to}] — only keys
        whose value actually differs. The enable dialog shows this so a user
        who tuned those sections sees exactly which knobs get overwritten
        before consenting."""
        diff: List[Dict[str, Any]] = []
        for section, patch in (("curation", OPTIMAL_CURATION),
                               ("training", OPTIMAL_TRAINING),
                               ("validation", OPTIMAL_VALIDATION),
                               ("goldens_mine", OPTIMAL_GOLDENS_MINE)):
            cur = getattr(self.cfg, section, None)
            cur = cur if isinstance(cur, dict) else {}
            for key, to in patch.items():
                if cur.get(key) != to:
                    diff.append({"section": section, "key": key,
                                 "from": cur.get(key), "to": to})
        return diff

    def h_easy_enable(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        name = str(b.get("profile") or "").strip()
        profiles, _default = self._profiles()
        try:
            prof = easy_validate_profile(profiles, name)
        except ValueError as e:
            raise HttpError(400, str(e)) from e
        # Optional extra agents (multi-agent loop): every name must exist and
        # be server-bound, same bar as the primary. `profile` stays the primary.
        extras_raw = b.get("profiles")
        extras: List[str] = []
        if extras_raw is not None:
            if not isinstance(extras_raw, list):
                raise HttpError(400, "profiles must be a list of profile names")
            for extra in extras_raw:
                ename = str(extra or "").strip()
                if not ename or ename == name:
                    continue
                try:
                    easy_validate_profile(profiles, ename)
                except ValueError as e:
                    raise HttpError(400, str(e)) from e
                if ename not in extras:
                    extras.append(ename)
        # Preview mode: report the preset diff without writing anything.
        if b.get("preview"):
            return {"ok": True, "diff": self._easy_preset_diff()}
        auto_deploy = bool(b.get("auto_deploy"))
        apply_preset = bool(b.get("apply_preset", True))
        loop_all = [str(prof["name"])] + extras
        ec = EasyTrainingConfig(enabled=True, profile=str(prof["name"]),
                                profiles=loop_all, auto_deploy=auto_deploy)
        patches = optimal_config_patches(ec) if apply_preset else [
            {"section": "easy_training", "patch": ec.as_dict()}]
        # All-or-nothing: snapshot the config file and restore it if any
        # patch fails, so a partial enable (enabled=true with a half-applied
        # preset) can never persist.
        cfg_path = self._config_path()
        snapshot = (cfg_path.read_bytes() if cfg_path.is_file() else None)

        def _rollback() -> None:
            try:
                if snapshot is None:
                    cfg_path.unlink(missing_ok=True)
                    self.cfg = SableConfig()
                else:
                    cfg_path.write_bytes(snapshot)
                    self.cfg = load_config(str(cfg_path))
            except Exception:  # last resort: keep serving, file may be mixed
                logger.exception("easy training: config rollback failed")

        for body in patches:
            r = self.handle("PUT", "/api/config", body=body)
            if r.status >= 400:
                _rollback()
                err = r.body.get("error", {}) if isinstance(r.body, dict) else {}
                raise HttpError(r.status,
                                str(err.get("message", "config write failed")),
                                code=str(err.get("code", "internal_error")))
        self.bus.publish("easy_training.enabled",
                         {"profile": ec.profile, "auto_deploy": auto_deploy})
        return {"ok": True, "config": ec.as_dict(),
                "diff": self._easy_preset_diff() if apply_preset else [],
                "note": ("Easy training on: chat with this profile feeds the "
                         "buffer; run it until the readiness gauge fills, "
                         "then Train now closes the loop.")}

    def h_easy_disable(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        # Section-patch (not root): keeps profile/auto_deploy keys intact so a
        # later re-enable remembers them.
        r = self.handle("PUT", "/api/config", body={
            "section": "easy_training",
            "patch": {"enabled": False}})
        if r.status >= 400:
            err = r.body.get("error", {}) if isinstance(r.body, dict) else {}
            raise HttpError(r.status,
                            str(err.get("message", "config write failed")),
                            code=str(err.get("code", "internal_error")))
        self.bus.publish("easy_training.disabled", {})
        return {"ok": True}

    # ----------------------------------------------- one-shot feature presets

    def h_features_enable_all(self, q: Dict[str, str],
                              b: Dict[str, Any]) -> Dict[str, Any]:
        """Turn on every safe-and-useful opt-in feature (Options panel button).

        Same contract as h_easy_enable: `{"preview": true}` writes nothing and
        returns the diff; an apply is all-or-nothing (config file snapshot +
        restore on any failed patch, so a half-enabled preset can never
        persist). Each patch goes through the normal PUT /api/config path, so
        validation, .bak backup and per-section merge semantics are identical
        to flipping each card by hand. What is covered vs deliberately skipped
        lives in core/presets.py (easy_training loop, janitor, unmeasured GPU
        knobs, and conditional LSP/calibration)."""
        from ..core.presets import enable_all_diff, enable_all_patches

        if b.get("preview"):
            return {"ok": True,
                    "diff": enable_all_diff(self.cfg, home=str(self.home))}

        patches = enable_all_patches(self.cfg, home=str(self.home))
        cfg_path = self._config_path()
        snapshot = (cfg_path.read_bytes() if cfg_path.is_file() else None)

        def _rollback() -> None:
            # Snapshot restore AND in-memory reload must both happen even
            # when no config file existed before (fresh install): load_config
            # raises FileNotFoundError on a missing path, which used to leave
            # self.cfg stale-with-half-the-preset while the file was gone.
            try:
                if snapshot is None:
                    cfg_path.unlink(missing_ok=True)
                    self.cfg = SableConfig()
                else:
                    cfg_path.write_bytes(snapshot)
                    self.cfg = load_config(str(cfg_path))
            except Exception:  # last resort: keep serving, file may be mixed
                logger.exception("enable-all: config rollback failed")

        diff = enable_all_diff(self.cfg, home=str(self.home))
        for body in patches:
            r = self.handle("PUT", "/api/config", body=body)
            if r.status >= 400:
                _rollback()
                err = r.body.get("error", {}) if isinstance(r.body, dict) else {}
                raise HttpError(r.status,
                                str(err.get("message", "config write failed")),
                                code=str(err.get("code", "internal_error")))
        self.bus.publish("features.enabled_all", {"changed": len(diff)})
        return {"ok": True, "changed": diff,
                "note": ("Features enabled: knowledge (skills/playbooks/"
                         "indexing), learning pipeline (RFT, evolution, "
                         "golden mining, shadow eval), routing, watchdog"
                         + (" — restart running servers to pick up "
                            "watchdog/routing changes." if diff else "."))}

    # ------------------------------------------------- add / remove one agent
    # The loop is a list: `profile` stays the primary, `profiles` holds every
    # agent (primary included once there are multiple). These endpoints patch
    # just that list — enabling/disabling the whole loop stays with
    # enable/disable, and the optimal preset is already applied from the first
    # enable, so adding an agent never re-writes curation/training sections.

    def _easy_write_loop(self, names: List[str], auto_deploy: Optional[bool]
                         = None) -> EasyTrainingConfig:
        """Persist a new agent list; returns the resulting config.

        Single-agent lists collapse back to the legacy shape (profile only, no
        profiles key) so removing everyone but one leaves the same config file
        an original single-profile enable would have written."""
        primary = names[0] if names else None
        patch: Dict[str, Any] = {"profile": primary}
        patch["profiles"] = list(names) if len(names) > 1 else None
        if auto_deploy is not None:
            patch["auto_deploy"] = bool(auto_deploy)
        r = self.handle("PUT", "/api/config", body={
            "section": "easy_training", "patch": patch})
        if r.status >= 400:
            err = r.body.get("error", {}) if isinstance(r.body, dict) else {}
            raise HttpError(r.status,
                            str(err.get("message", "config write failed")),
                            code=str(err.get("code", "internal_error")))
        return self._easy_cfg()

    def h_easy_add(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        """Add one agent profile to the easy loop (must exist + be server-bound).

        Works while the loop is enabled; when disabled it just grows the saved
        list so a later enable picks everyone up. No preset re-application —
        quality machinery is section-global and already armed by the first
        enable (enable it fresh if it was never on)."""
        name = str(b.get("profile") or "").strip()
        profiles, _default = self._profiles()
        try:
            easy_validate_profile(profiles, name)
        except ValueError as e:
            raise HttpError(400, str(e)) from e
        ec = self._easy_cfg()
        current = ec.all_profiles
        if name in current:
            return {"ok": True, "config": ec.as_dict(),
                    "note": f"“{name}” is already in the easy loop"}
        new_ec = self._easy_write_loop(current + [name])
        self.bus.publish("easy_training.agent_added",
                         {"profile": name, "profiles": new_ec.all_profiles})
        return {"ok": True, "config": new_ec.as_dict(),
                "note": (f"“{name}” joined the easy loop: chat with it to "
                         "feed its slice; Train now runs per-agent loops.")}

    def h_easy_remove(self, q: Dict[str, str], b: Dict[str, Any]
                      ) -> Dict[str, Any]:
        """Remove one agent from the easy loop (traces stay; only scoping ends)."""
        name = str(b.get("profile") or "").strip()
        ec = self._easy_cfg()
        current = ec.all_profiles
        if name not in current:
            raise HttpError(400, f"“{name}” is not in the easy loop")
        remaining = [p for p in current if p != name]
        new_ec = self._easy_write_loop(remaining)
        self.bus.publish("easy_training.agent_removed",
                         {"profile": name, "profiles": new_ec.all_profiles})
        return {"ok": True, "config": new_ec.as_dict()}

    def h_easy_run(self, q: Dict[str, str], b: Dict[str, Any]) -> ApiResponse:
        ec = self._easy_cfg()
        if not ec.enabled or not ec.all_profiles:
            raise HttpError(400, "easy training is not enabled — pick a "
                                 "profile and enable it first",
                            code="easy_disabled")
        # Which agent to train: explicit body.profile (multi-agent panels run
        # one loop per agent) else the primary — legacy callers unchanged.
        target = str(b.get("profile") or "").strip() or ec.profile \
            or ec.all_profiles[0]
        if target not in ec.all_profiles:
            raise HttpError(
                400, f"“{target}” is not in the easy loop "
                f"({', '.join(ec.all_profiles)})", code="unknown_profile")
        profiles, _default = self._profiles()
        try:
            prof = easy_validate_profile(profiles, target)
        except ValueError as e:
            raise HttpError(400, str(e)) from e
        dry_run = bool(b.get("dry_run"))
        if not dry_run:
            readiness = self._easy_readiness(ec.for_profile(target))
            reasons = []
            if not readiness["trigger"]:
                th = {"min_traces": readiness["min_traces"],
                      "min_score": readiness["min_score"]}
                if readiness["unclaimed"] < th["min_traces"]:
                    reasons.append(
                        f"only {readiness['unclaimed']} of "
                        f"{th['min_traces']} traces — chat more with this "
                        "profile (or lower the threshold in Settings)")
                ms = readiness["mean_score"]
                if ms is None:
                    reasons.append("no scored traces yet — rate a few chats")
                elif ms < th["min_score"]:
                    reasons.append(
                        f"mean score {ms:.2f} below {th['min_score']}")
            if readiness["gate"].get("ok") is False:
                reasons.extend(readiness["gate"].get("reasons") or [])
            if reasons:
                raise HttpError(409, "; ".join(reasons), code="not_ready")
        # Reuse the iterate engine with easy-loop semantics: profile-scoped
        # curation, sft kind (the preset curates both formats; sft trains from
        # sft.jsonl and dpo pairs ride along in the dataset). On completion
        # _easy_after_run ledgers the result and hot-deploys when configured.
        # Before training: one best-effort golden-mining pass so 👍-thumbed
        # chats since the last run gate the model about to be promoted
        # (goldens_mine.enabled gates it; stdlib-only, never breaks the run —
        # failures are logged inside mine_goldens_pass and recorded here).
        mine = mine_goldens_pass(self.cfg, self.home)
        easy_save_state(self.home, {"last_mine": {
            "ts": _utcnow(), "profile": target, **mine}})
        # Single-agent projection so the completion hook (auto-deploy to THIS
        # agent's server, bus event naming) behaves exactly like the legacy
        # one-profile loop. The cross-process loop lock still serializes runs
        # globally — per-agent loops queue behind each other, never double-
        # claim the shared buffer.
        run = self.submit_iterate(
            {
                "name": f"easy:{target}",
                "kind": str(b.get("kind") or "sft"),
                "dry_run": dry_run,
                "profile": target,
            },
            on_done=lambda record: self._easy_after_run(
                record.get("run_id", ""), record, ec.for_profile(target), prof),
        )
        easy_save_state(self.home, {"last_start": {
            "run_id": run["run_id"], "ts": _utcnow(), "profile": target}})
        return ApiResponse(202, run)

    def h_iterate(self, q: Dict[str, str], b: Dict[str, Any]) -> ApiResponse:
        return ApiResponse(202, self.submit_iterate(b))

    def submit_iterate(self, b: Dict[str, Any], *,
                       on_done: Optional[Callable[[Dict[str, Any]], None]] = None,
                       ) -> Dict[str, Any]:
        """Queue a full iterate run in a background thread.

        ``b["profile"]`` scopes the curation stage to one agent profile's
        traces (easy training); absent => whole buffer. ``on_done`` fires with
        the finished run record after the persisted state is saved (the easy
        loop uses it for auto-deploy bookkeeping)."""
        kind = str(b.get("kind") or "sft")
        if kind not in ("sft", "dpo", "grpo"):
            raise HttpError(400, "kind must be sft, dpo or grpo")
        name = b.get("name")
        dataset_id = b.get("dataset") or None
        dry_run = bool(b.get("dry_run"))
        # Per-request base-model override (discovered-models dropdown), same
        # contract as submit_train.
        base_model_override = str(b["base_model"]) if b.get("base_model") else None
        # Curation scope (easy training): only traces stamped with this agent
        # profile feed the dataset. None => unscoped, same as before.
        curate_profile = str(b["profile"]) if b.get("profile") else None

        # Cross-process guard: cron (`train auto-check --trigger` -> cmd_iterate)
        # and a manual click here are separate processes and both curate the
        # shared buffer; _exec_lock only serializes inside this process. The
        # advisory lockfile in <home>/easy_training/ serializes across them so
        # traces can never be double-claimed. Dry runs validate without
        # claiming, so they skip it. (Released when the worker finishes.)
        loop_handle = None
        if not dry_run:
            loop_handle = acquire_loop_lock(self.home)
            if loop_handle is None:
                raise HttpError(
                    409, "another training loop run is active (dashboard or "
                         "cron) — wait for it to finish", code="loop_busy")
        run_id = f"iter-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        runs_dir = self.home / ITERATE_RUNS_DIRNAME
        runs_dir.mkdir(parents=True, exist_ok=True)
        record: Dict[str, Any] = {
            "run_id": run_id,
            "started": _utcnow(),
            "finished": None,
            "state": "running",
            "name": str(name or "ui"),
            "kind": kind,
            "dataset": dataset_id,
            "dry_run": dry_run,
            "steps": [],
            "result": None,
        }
        _iter_tc = self._training_config()
        if base_model_override:
            _iter_tc.base_model = base_model_override
        record_path = runs_dir / f"{run_id}.json"

        def _save() -> None:
            tmp = record_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2),
                           encoding="utf-8")
            os.replace(tmp, record_path)

        _save()

        def _on_stage(stage: str, detail: str) -> None:
            state = ("failed" if detail.startswith("FAILED")
                     else "running" if detail == "running" else "done")
            record["steps"].append({"name": stage, "state": state,
                                    "detail": detail})
            _save()
            self.bus.publish("iterate.step", {"run_id": run_id, "step": stage,
                                              "state": state, "detail": detail})

        def _run() -> None:
            if not self._exec_lock.acquire(blocking=False):
                record["state"] = "error"
                record["steps"].append({"name": "queue", "state": "failed",
                                        "detail": "another execution is running"})
                record["finished"] = _utcnow()
                _save()
                release_loop_lock(loop_handle)
                return
            store = self._store()
            reg = self._registry()
            try:
                from ..training.pipeline import run_iterate

                cc = curation_config_from_yaml({"curation": self.cfg.curation or {}})
                if self.cfg.gguf_path and not cc.base_model:
                    cc.base_model = self.cfg.gguf_path
                if curate_profile:
                    cc.agent_profile = curate_profile
                res = run_iterate(
                    home=self.home, registry=reg,
                    scheduler=self._scheduler(reg),
                    store=store,
                    curation_cfg=cc,
                    training_cfg=_iter_tc,
                    validation_cfg=_validation_cfg_from(self.cfg, home=self.home),
                    name=str(name or "ui"), kind=kind, dataset_id=dataset_id,
                    dry_run=dry_run, on_stage=_on_stage,
                )
                record["result"] = res.to_dict()
                record["state"] = "done" if res.ok else "failed"
            except Exception as e:
                record["state"] = "error"
                record["result"] = {"error": f"{type(e).__name__}: {e}"}
            finally:
                record["finished"] = _utcnow()
                _save()
                self.bus.publish("iterate.step", {"run_id": run_id,
                                                  "step": "finished",
                                                  "state": record["state"],
                                                  "detail": ""})
                store.close()
                reg.close()
                self._exec_lock.release()
                if on_done:
                    try:
                        on_done(record)
                    except Exception:  # completion hooks never break the run
                        logger.exception("iterate: on_done hook failed for %s",
                                         run_id)
                # Released only after the completion hook (easy auto-deploy
                # runs there), so the loop stays owned until fully finished.
                release_loop_lock(loop_handle)

        threading.Thread(target=_run, daemon=True).start()
        return {"ok": True, "run_id": run_id,
                "note": "iterate running in background; poll /api/iterate/runs "
                        "and listen for iterate.step events"}

    def h_iterate_runs(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        runs_dir = self.home / ITERATE_RUNS_DIRNAME
        runs: List[Dict[str, Any]] = []
        if runs_dir.is_dir():
            for f in sorted(runs_dir.glob("iter-*.json"), reverse=True):
                try:
                    runs.append(json.loads(f.read_text()))
                except (OSError, json.JSONDecodeError):
                    continue
        return {"runs": runs}

    # ------------------------------------------------- evolve proposals (#39)

    def h_evolve_latest(self, q: Dict[str, str],
                        b: Dict[str, Any]) -> Dict[str, Any]:
        """Latest finished evolve proposal for one profile (Profiles panel).

        Compact payload only — lineage stays in the job row. Dismissed or
        already-applied proposals read as flags so the UI can hide/annotate
        them without deleting history.
        """
        from ..evaluation.evolve import (APPLIED_KEY, DISMISSED_KEY, RESULT_KEY)

        profile = str(q.get("profile") or b.get("profile") or "")
        if not profile:
            raise HttpError(400, "profile query parameter required")
        reg = self._registry()
        try:
            candidates = [j for j in reg.list_jobs()
                          if j.kind == "evolve" and j.state == "done"
                          and isinstance((j.params or {}).get(RESULT_KEY), dict)
                          and (j.params or {}).get("profile") == profile]
        finally:
            reg.close()
        if not candidates:
            return {"proposal": None}
        job = candidates[-1]  # list_jobs is queued_at-ordered; newest wins
        res = job.params[RESULT_KEY]
        baseline = res.get("baseline") or {}
        best = res.get("best") or {}

        def _fit(entry: Dict[str, Any]) -> Optional[float]:
            m = entry.get("judge_mean")
            return float(m) if isinstance(m, (int, float)) else None

        return {
            "proposal": {
                "job_id": job.job_id,
                "ts": res.get("ts"),
                "improved": bool(res.get("improved")),
                "diff": res.get("diff") or "",
                "best_variant_id": best.get("variant_id"),
                "best_direction": best.get("direction"),
                "fitness_current": _fit(baseline),
                "fitness_proposed": _fit(best),
                "golden_pass_current": baseline.get("golden_pass"),
                "golden_pass_proposed": best.get("golden_pass"),
                "calls_used": res.get("calls_used"),
                "call_budget": res.get("call_budget"),
                "partial": bool(res.get("partial")),
                "dismissed": bool((job.params or {}).get(DISMISSED_KEY)),
                "applied_at": (job.params or {}).get(APPLIED_KEY),
            }
        }

    def h_evolve_apply(self, q: Dict[str, str],
                       b: Dict[str, Any]) -> Dict[str, Any]:
        """Gated apply: profile config changes ONLY through this explicit user
        action (UI confirm dialog mirrors the promote flow)."""
        from ..evaluation.evolve import (APPLIED_KEY, EvolveError, RESULT_KEY,
                                         apply_prompt_to_config)

        job_id = str(b.get("job_id") or "")
        if not job_id:
            raise HttpError(400, "job_id required")
        reg = self._registry()
        try:
            job = reg.get_job(job_id)
            if job is None or job.kind != "evolve":
                raise HttpError(404, f"no evolve job {job_id}")
            if job.state != "done":
                raise HttpError(409, f"job {job_id} is {job.state}; only "
                                     "finished proposals can be applied")
            res = (job.params or {}).get(RESULT_KEY)
            if not isinstance(res, dict):
                raise HttpError(409, "job has no stored proposal")
            if (job.params or {}).get(APPLIED_KEY):
                raise HttpError(409, "proposal already applied")
            best = res.get("best") or {}
            if not res.get("improved") or not best.get("text"):
                raise HttpError(409, "no improving variant to apply")
            profile_name = str((job.params or {}).get("profile") or "")
            path = self._config_path()
            try:
                out = apply_prompt_to_config(path, profile_name, best["text"])
            except EvolveError as e:
                raise HttpError(404, str(e)) from e
            reg.set_job_param(job_id, APPLIED_KEY, _utcnow())
        finally:
            reg.close()
        # Reflect the new profile prompt in this app instance (same reload the
        # config PUT does; keeps /api/profiles honest right after apply).
        try:
            self.cfg = load_config(str(path))
        except Exception:
            pass
        return {"ok": True, "profile": out.get("profile"),
                "backup": out.get("backup")}

    def h_evolve_dismiss(self, q: Dict[str, str],
                         b: Dict[str, Any]) -> Dict[str, Any]:
        """Flag a proposal dismissed on the job row — never deletes history."""
        from ..evaluation.evolve import DISMISSED_KEY, RESULT_KEY

        job_id = str(b.get("job_id") or "")
        if not job_id:
            raise HttpError(400, "job_id required")
        reg = self._registry()
        try:
            job = reg.get_job(job_id)
            if job is None or job.kind != "evolve":
                raise HttpError(404, f"no evolve job {job_id}")
            if not isinstance((job.params or {}).get(RESULT_KEY), dict):
                raise HttpError(409, "job has no stored proposal")
            reg.set_job_param(job_id, DISMISSED_KEY, _utcnow())
        finally:
            reg.close()
        return {"ok": True, "dismissed": job_id}

    # -------------------------------------------------------- agent profiles

    # Known tool names offered as checkboxes in the Profiles panel: the
    # built-in toolset (core/tools.py). Unknown names are still *stored* but
    # flagged by the UI as unrecognized (they simply won't be exposed).
    # Keep in sync with TOOL_SCHEMAS + CAPABILITY_TOOL_SCHEMAS in
    # core/tools.py; test_dashboard pins the parity. The skill tools appear
    # here too: profiles may still restrict them, and when skills.enabled is
    # off the runner strips them regardless (the union in agent_runner keeps
    # them available to profiles that omit them while the feature is on).
    KNOWN_TOOLS = ["read_file", "list_dir", "write_file", "patch",
                   "search_files", "todo_write", "todo_read",
                   "search_traces", "search_codebase",
                   "list_agents", "delegate_task", "check_delegation",
                   "save_skill", "read_skill", "list_skills", "delete_skill"]

    # Per-agent capabilities (reach outside the machine / use other programs):
    # key -> tool it gates. Off by default; global defaults live in config
    # `agent.capabilities:`, per-profile overrides in the profile dict. The
    # Profiles panel renders these as toggles separate from the tool grid —
    # enabling one exposes web_fetch/run_command to that agent only.
    CAPABILITY_TOGGLES = {
        "internet": "web_fetch",
        "run_commands": "run_command",
        "vision": "view_image",
        "screenshots": "take_screenshot",
    }

    _PROFILE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

    def _profiles(self) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Current (profiles, default_profile) from the live config object."""
        profs = self.cfg.profiles if isinstance(self.cfg.profiles, list) else []
        default = self.cfg.default_profile
        return [p for p in profs if isinstance(p, dict)], (
            str(default) if default else None)

    def _write_profiles(self, profiles: List[Dict[str, Any]],
                        default: Optional[str]) -> Dict[str, Any]:
        patch: Dict[str, Any] = {"profiles": profiles}
        # `default_profile: null` deletes the key (unset semantics).
        patch["default_profile"] = default
        r = self.handle("PUT", "/api/config", body={"section": "root",
                                                    "patch": patch})
        if r.status >= 400:
            err = r.body.get("error", {}) if isinstance(r.body, dict) else {}
            raise HttpError(r.status, str(err.get("message", "config write failed")),
                            code=str(err.get("code", "internal_error")))
        return r.body if isinstance(r.body, dict) else {"ok": True}

    def _validate_profile(self, name: str, b: Dict[str, Any]) -> Dict[str, Any]:
        if not self._PROFILE_NAME_RE.match(name):
            raise HttpError(
                400, "profile name must match [A-Za-z0-9][A-Za-z0-9_-]{0,63}")
        prompt = b.get("system_prompt", "")
        if not isinstance(prompt, str):
            raise HttpError(400, "system_prompt must be a string")
        tools = b.get("tools", [])
        if not isinstance(tools, list) or any(not isinstance(t, str) for t in tools):
            raise HttpError(400, "tools must be a list of tool names")
        model_name = b.get("model_name")
        if model_name is not None and not isinstance(model_name, str):
            raise HttpError(400, "model_name must be a string or null")
        # Backlog #9: optional per-profile workspace containment override —
        # widens/narrows the global agent.workspace_root for chats using this
        # profile (see core/agent_runner.resolve_workspace_roots).
        workspace_root = b.get("workspace_root")
        if workspace_root is not None and not isinstance(workspace_root, str):
            raise HttpError(400, "workspace_root must be a string or null")
        # Phase 6: optional server-instance binding — chat runs for this
        # profile target that managed instance's endpoint instead of the
        # default serve target ("" | "default" => unset).
        server_instance = b.get("server_instance")
        if server_instance is not None and not isinstance(server_instance, str):
            raise HttpError(400, "server_instance must be a string or null")
        # Per-agent capabilities: {internet?: bool, run_commands?: bool}.
        # Only recognized keys are stored; absent => inherit global defaults.
        caps_in = b.get("capabilities")
        caps: Dict[str, bool] = {}
        if caps_in is not None:
            if not isinstance(caps_in, dict):
                raise HttpError(400, "capabilities must be an object or null")
            for key, val in caps_in.items():
                if key not in self.CAPABILITY_TOGGLES:
                    raise HttpError(
                        400, f"unknown capability {key!r} "
                        f"(known: {', '.join(self.CAPABILITY_TOGGLES)})")
                caps[str(key)] = bool(val)
        # Per-profile sampling + thinking overrides (reference style):
        # merged over the bound instance's / global `sampling:` sections for
        # chat runs using this profile (see h_chat_stream).
        samp_in = b.get("sampling")
        samp: Dict[str, Any] = {}
        if samp_in is not None:
            from ..serving.run_args import sanitize_sampling

            if not isinstance(samp_in, dict):
                raise HttpError(400, "sampling must be an object or null")
            try:
                samp = {k: v for k, v in sanitize_sampling(samp_in).items()
                        if v is not None}
            except ValueError as e:
                raise HttpError(400, f"sampling: {e}") from e
        prof: Dict[str, Any] = {
            "name": name,
            "system_prompt": prompt,
            "tools": sorted(set(tools)),
            "model_name": model_name or None,
        }
        if workspace_root:
            prof["workspace_root"] = workspace_root
        si = (server_instance or "").strip()
        if si and si != "default":
            prof["server_instance"] = si
        if caps:
            prof["capabilities"] = caps
        if samp:
            prof["sampling"] = samp
        return prof

    def h_profiles_list(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        profiles, default = self._profiles()
        # Global capability defaults (config `agent.capabilities:`) so the UI
        # can show each toggle's effective state when a profile is silent.
        agent_cfg = self.cfg.agent if isinstance(self.cfg.agent, dict) else {}
        global_caps = agent_cfg.get("capabilities")
        return {"profiles": profiles, "default_profile": default,
                "known_tools": list(self.KNOWN_TOOLS),
                "capability_toggles": dict(self.CAPABILITY_TOGGLES),
                "global_capabilities": {
                    k: bool((global_caps or {}).get(k, False))
                    for k in self.CAPABILITY_TOGGLES}}

    def h_profile_put(self, q: Dict[str, str], b: Dict[str, Any],
                      name: str) -> Dict[str, Any]:
        prof = self._validate_profile(name, b)
        profiles, default = self._profiles()
        existing = {p.get("name"): i for i, p in enumerate(profiles)}
        if name in existing:
            profiles[existing[name]] = prof
        else:
            profiles.append(prof)
        self._write_profiles(profiles, default)
        return {"ok": True, "profile": prof}

    def h_profile_delete(self, q: Dict[str, str], b: Dict[str, Any],
                         name: str) -> Dict[str, Any]:
        profiles, default = self._profiles()
        if not any(p.get("name") == name for p in profiles):
            raise HttpError(404, f"no profile {name}")
        profiles = [p for p in profiles if p.get("name") != name]
        if default == name:
            default = None
        self._write_profiles(profiles, default)
        return {"ok": True}

    def h_profiles_set_default(self, q: Dict[str, str],
                               b: Dict[str, Any]) -> Dict[str, Any]:
        default = b.get("default")
        profiles, _cur = self._profiles()
        if default is not None:
            default = str(default)
            if not any(p.get("name") == default for p in profiles):
                raise HttpError(404, f"no profile {default}")
        self._write_profiles(profiles, default)
        return {"ok": True, "default_profile": default}

    def _resolve_profile(self, name: Optional[str]) -> Optional[Dict[str, Any]]:
        """Look up a profile for /api/chat/stream. Empty/'default' => config
        default; unknown name => 404."""
        profiles, default = self._profiles()
        if not name or name in ("default", "none"):
            name = default
            if not name:
                return None
        for p in profiles:
            if p.get("name") == name:
                return p
        raise HttpError(404, f"no profile {name}", code="unknown_profile")

    # --------------------------------------------------------------- projects
    # One JSON file per project at <home>/projects/<id>.json — same shape and
    # atomic-write style as the thread files in <home>/chat/. Threads carry an
    # optional nullable "project" id (null = Unfiled); deleting a project never
    # touches threads (their ids dangle and render under Unfiled).

    _PROJECT_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

    def _project_path(self, project_id: str) -> Path:
        if not self._PROJECT_ID_RE.match(project_id):
            raise HttpError(400, "invalid project id")
        return self.home / PROJECTS_DIRNAME / f"{project_id}.json"

    def _load_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        try:
            p = self._project_path(project_id)
        except HttpError:
            return None
        if not p.is_file():
            return None
        try:
            data = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    def _save_project(self, project: Dict[str, Any]) -> None:
        p = self._project_path(str(project["id"]))
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(project, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        os.replace(tmp, p)

    def _validate_project_body(self, b: Dict[str, Any],
                               existing: Optional[Dict[str, Any]] = None,
                               ) -> Dict[str, Any]:
        """Validate a create/update body; returns the mutable field set."""
        name = b.get("name", (existing or {}).get("name"))
        if not isinstance(name, str) or not name.strip():
            raise HttpError(400, "name required (non-empty string)")
        # Duplicate names make projects indistinguishable in the thread tree,
        # the trace filter and the explorer title — reject them (case-
        # insensitive; another project keeping the same name is fine).
        want = name.strip().casefold()
        d = self.home / PROJECTS_DIRNAME
        if d.is_dir():
            for f in d.glob("*.json"):
                other_id = f.stem
                if existing is not None and other_id == existing.get("id"):
                    continue
                try:
                    other = json.loads(f.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                if (isinstance(other, dict)
                        and str(other.get("name") or "").strip().casefold()
                        == want):
                    raise HttpError(
                        409, f"a project named {name.strip()!r} already exists "
                        f"(id {other_id})", code="duplicate_project_name")
        ws_raw = b.get("workspace_dir",
                       (existing or {}).get("workspace_dir"))
        if not isinstance(ws_raw, str) or not ws_raw.strip():
            raise HttpError(400, "workspace_dir required (absolute path)")
        ws = Path(ws_raw).expanduser()
        if not ws.is_absolute():
            raise HttpError(400, "workspace_dir must be an absolute path")
        if not ws.is_dir():
            raise HttpError(400, f"workspace_dir does not exist: {ws}")
        agents = b.get("agents", (existing or {}).get("agents") or [])
        if not isinstance(agents, list) or any(
                not isinstance(a, str) for a in agents):
            raise HttpError(400, "agents must be a list of profile names")
        return {"name": name.strip(), "workspace_dir": str(ws),
                "agents": sorted(set(agents))}

    def h_projects_list(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        d = self.home / PROJECTS_DIRNAME
        projects = []
        if d.is_dir():
            for f in sorted(d.glob("*.json")):
                try:
                    p = json.loads(f.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                if isinstance(p, dict):
                    projects.append(p)
        return {"projects": projects}

    def h_projects_create(self, q: Dict[str, str], b: Dict[str, Any]) -> ApiResponse:
        fields = self._validate_project_body(b)
        project_id = uuid.uuid4().hex[:12]
        now = _utcnow()
        project = {"id": project_id, "created": now, "updated": now, **fields}
        self._save_project(project)
        return ApiResponse(201, {"ok": True, "project": project})

    def h_project_put(self, q: Dict[str, str], b: Dict[str, Any],
                      project_id: str) -> Dict[str, Any]:
        existing = self._load_project(project_id)
        if existing is None:
            raise HttpError(404, f"no project {project_id}")
        fields = self._validate_project_body(b, existing=existing)
        project = {**existing, **fields, "id": project_id,
                   "updated": _utcnow()}
        self._save_project(project)
        return {"ok": True, "project": project}

    def h_project_delete(self, q: Dict[str, str], b: Dict[str, Any],
                         project_id: str) -> Dict[str, Any]:
        p = self._project_path(project_id)
        if not p.is_file():
            raise HttpError(404, f"no project {project_id}")
        # Cancel any in-flight build FIRST and drop the cached instance — an
        # uncancelled build would recreate <home>/index/<id>.db through
        # _db()'s mkdir/connect after delete_project_index removed it.
        try:
            from ..core.indexer import drop_index_instance, get_index

            try:
                get_index(self.home, cfg=self.cfg,
                          project_id=project_id).cancel_build()
            except Exception:
                pass  # unknown/invalid id: nothing to cancel
            drop_index_instance(self.home, project_id)
        except Exception:
            pass
        p.unlink()
        # Phase 5: delete the project's index DB (and only that file — the
        # default DB and other projects' indexes are never touched).
        index_removed = False
        try:
            from ..core.indexer import delete_project_index

            index_removed = delete_project_index(self.home, project_id)
        except Exception:
            pass  # index cleanup is best-effort; the project file is gone
        # Threads intentionally untouched: their stale project ids read as
        # Unfiled (the tree groups unknown/absent ids under Unfiled).
        return {"ok": True, "index_removed": index_removed}

    # ------------------------------------------------------------ fs explorer
    # GET /api/fs/tree?project=<id>&path=<rel> — strictly read-only workspace
    # listing for the chat-page file explorer. One directory level per call
    # (the client lazy-loads on expand); names/metadata only, never contents.
    # Containment reuses core/path_security (resolve()-based, symlink-safe: a
    # path whose realpath leaves the workspace is rejected exactly like the
    # agent tools do) and the sensitive-path deny list is imported straight
    # from core/tools so the explorer blocks what the agent cannot read.

    def h_fs_tree(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        from ..core.path_security import has_traversal_component, validate_within_dir
        from ..core.tools import _sensitive_read_reason

        project_id = str(q.get("project") or "").strip()
        if not project_id:
            raise HttpError(400, "project parameter required",
                            code="bad_request")
        proj = self._load_project(project_id)
        if proj is None:
            raise HttpError(404, f"no project {project_id}",
                            code="unknown_project")
        ws_raw = str(proj.get("workspace_dir") or "")
        try:
            ws_root = Path(ws_raw).expanduser().resolve()
        except OSError:
            ws_root = None
        if ws_root is None or not ws_root.is_dir():
            raise HttpError(404, f"project workspace missing: {ws_raw}",
                            code="unknown_project")

        rel = str(q.get("path") or "").strip()
        if rel in ("", ".", "./"):
            target = ws_root
        else:
            # Absolute paths and '..' components are rejected up front (same
            # cheap pre-check the contain() decorator runs), then resolve()-
            # based containment catches symlink escapes to existing targets.
            # %2e%2e-style encoded traversal: the HTTP layer's parse_qs already
            # decodes query values, but re-check the unquoted form here too so
            # direct handle() callers get the same 400.
            if (Path(rel).is_absolute() or rel.startswith("~")
                    or Path(unquote(rel)).is_absolute()):
                raise HttpError(400, "path must be relative to the workspace",
                                code="bad_request")
            if (has_traversal_component(rel)
                    or has_traversal_component(unquote(rel))):
                raise HttpError(400, "'..' path traversal not allowed",
                                code="bad_request")
            candidate = ws_root / rel
            reason = validate_within_dir(candidate, ws_root)
            if reason is not None:
                raise HttpError(400, f"path escapes workspace: {rel}",
                                code="bad_request")
            target = candidate

        # A sensitive directory (.ssh/.gnupg/... or .aws credentials) is
        # blocked as a listing root too — listing it is itself the leak.
        if _sensitive_read_reason(str(target)) is not None:
            raise HttpError(400, "path blocked (sensitive credential path)",
                            code="bad_request")
        if not target.is_dir():
            raise HttpError(404, f"no such directory: {rel}",
                            code="not_found")

        show_hidden = str(q.get("hidden") or "") in ("1", "true", "yes")
        entries: List[Dict[str, Any]] = []
        truncated = False
        try:
            children = list(target.iterdir())
        except OSError as e:
            raise HttpError(404, f"unreadable directory: {rel}",
                            code="not_found") from e
        # Entry cap: a node_modules-scale directory must not serialize every
        # child into one response. Alphabetical pre-sort keeps the visible
        # slice deterministic; `truncated` tells the client more exists.
        if len(children) > FS_TREE_MAX_ENTRIES:
            children = sorted(children, key=lambda c: c.name.lower()) \
                [:FS_TREE_MAX_ENTRIES]
            truncated = True
        for child in children:
            name = child.name
            if not show_hidden and name.startswith("."):
                continue
            try:
                # Deny-list first (lexical, like tools.py), then stat.
                if _sensitive_read_reason(str(child)) is not None:
                    continue
                is_dir = child.is_dir()  # follows symlinks
                st = child.stat()
                # Symlink escapes never appear in the listing at all: the
                # target lives outside the workspace, so expanding it would
                # be blocked anyway — omit rather than show a dead node.
                if validate_within_dir(child, ws_root) is not None:
                    continue
                child_rel = (str(rel) + "/" if rel and rel != "." else "") + name
                entry: Dict[str, Any] = {
                    "name": name,
                    "path": child_rel,
                    "type": "dir" if is_dir else "file",
                }
                if not is_dir:
                    entry["size"] = st.st_size
                entry["mtime"] = datetime.fromtimestamp(
                    st.st_mtime, timezone.utc).isoformat(timespec="seconds")
                entries.append(entry)
            except OSError:
                continue  # unreadable entries are silently omitted
        entries.sort(key=lambda e: (e["type"] != "dir",
                                    e["name"].lower()))
        return {"project": project_id, "path": rel or ".", "entries": entries,
                "truncated": truncated}

    # GET /api/fs/read?project=<id>&path=<rel> — read-only file preview for
    # the explorer: same containment + sensitive-path rules as /api/fs/tree,
    # plus a hard size cap and a binary sniff (binary files return metadata
    # only). Strictly text; never follows out of the workspace.
    def h_fs_read(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        from ..core.path_security import has_traversal_component, validate_within_dir
        from ..core.tools import _sensitive_read_reason

        project_id = str(q.get("project") or "").strip()
        if not project_id:
            raise HttpError(400, "project parameter required",
                            code="bad_request")
        proj = self._load_project(project_id)
        if proj is None:
            raise HttpError(404, f"no project {project_id}",
                            code="unknown_project")
        ws_raw = str(proj.get("workspace_dir") or "")
        try:
            ws_root = Path(ws_raw).expanduser().resolve()
        except OSError:
            ws_root = None
        if ws_root is None or not ws_root.is_dir():
            raise HttpError(404, f"project workspace missing: {ws_raw}",
                            code="unknown_project")

        rel = str(q.get("path") or "").strip()
        if (not rel or rel in (".", "./")
                or Path(rel).is_absolute() or rel.startswith("~")
                or Path(unquote(rel)).is_absolute()):
            raise HttpError(400, "path must be a relative file path",
                            code="bad_request")
        if (has_traversal_component(rel)
                or has_traversal_component(unquote(rel))):
            raise HttpError(400, "'..' path traversal not allowed",
                            code="bad_request")
        target = ws_root / rel
        if validate_within_dir(target, ws_root) is not None:
            raise HttpError(400, f"path escapes workspace: {rel}",
                            code="bad_request")
        if _sensitive_read_reason(str(target)) is not None:
            raise HttpError(400, "path blocked (sensitive credential path)",
                            code="bad_request")
        try:
            if not target.is_file():
                raise HttpError(404, f"no such file: {rel}", code="not_found")
            size = target.stat().st_size
        except OSError as e:
            raise HttpError(404, f"unreadable file: {rel}",
                            code="not_found") from e
        # Size gate before touching contents: big files get metadata only.
        if size > FS_READ_MAX_CHARS * 4:
            return {"project": project_id, "path": rel, "size": size,
                    "binary": False, "truncated": True, "content": "",
                    "note": f"file too large to preview (>{FS_READ_MAX_CHARS * 4} bytes)"}
        try:
            raw = target.read_bytes()
        except OSError as e:
            raise HttpError(404, f"unreadable file: {rel}",
                            code="not_found") from e
        if b"\x00" in raw[:8192]:  # cheap binary sniff on the head window
            return {"project": project_id, "path": rel, "size": size,
                    "binary": True, "truncated": False, "content": ""}
        text = raw.decode("utf-8", errors="replace")
        truncated = len(text) > FS_READ_MAX_CHARS
        if truncated:
            text = text[:FS_READ_MAX_CHARS]
        return {"project": project_id, "path": rel, "size": size,
                "binary": False, "truncated": truncated, "content": text}


# ------------------------------------------------------------------ helpers

def _merge_patch(target: Dict[str, Any], patch: Dict[str, Any]) -> None:
    """Merge ``patch`` into ``target`` in place. A ``None`` patch value
    explicitly *deletes* the key (5e unset semantics — the plain dict.update
    merge could only overwrite, never clear)."""
    for k, v in patch.items():
        if v is None:
            target.pop(k, None)
        else:
            target[k] = v


def _validation_cfg_from(cfg: SableConfig, home: Optional[Path] = None):
    from ..core.config import resolve_config_path
    from ..curate.mine_goldens import merge_mined_into_tasks
    from ..training.validate import ValidationConfig

    data = cfg.validation or {}
    if not data:
        from ..training.scheduler import load_yaml_section

        # Same resolution as load_config: SABLE_CONFIG > <home>/sable.yaml.
        path = resolve_config_path()
        data = load_yaml_section(str(path) if path else None, "validation")
    vcfg = ValidationConfig.from_yaml({"validation": data})
    # Auto-mined goldens (curate/mine_goldens.py) join the promotion gate —
    # same merge as cli._validation_config. No mined file -> unchanged list.
    from ..store.paths import resolve_sable_home

    vcfg.golden_tasks = merge_mined_into_tasks(
        vcfg.golden_tasks, home or resolve_sable_home(cfg))
    return vcfg


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


def _truncate_row(row: Any) -> Tuple[Any, bool]:
    """Cap long strings in a preview row; returns (row, was_truncated)."""
    trunc = False

    def walk(v: Any) -> Any:
        nonlocal trunc
        if isinstance(v, str):
            if len(v) > ROW_PREVIEW_CHARS:
                trunc = True
                return v[:ROW_PREVIEW_CHARS] + "…"
            return v
        if isinstance(v, dict):
            return {k: walk(x) for k, x in v.items()}
        if isinstance(v, list):
            return [walk(x) for x in v]
        return v

    return walk(row), trunc


def _looks_like_error(content: str) -> bool:
    low = content.lstrip().lower()
    return (low.startswith("error") or "traceback" in low or "exception" in low)



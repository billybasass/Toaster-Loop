"""Dashboard control API — transport-agnostic JSON request handler.

Every mutation goes through the same core objects the CLI uses (TraceStore,
ModelRegistry, Scheduler, Curater) so the UI can never corrupt state beyond
what the CLI could; reads are direct DB queries. The HTTP layer in server.py
just adapts this to sockets, which keeps the whole API unit-testable by
calling ``DashboardApp.handle(method, path, query, body)`` directly.

Layout: this module is the composition root — routing (ROUTES + ``handle``)
and dependency wiring (stores, scheduler, dispatcher, auth token) only. The
route handlers live one domain per file under ``dashboard/api/`` as mixin
classes (chat, servers, hardwareops, jobs, hub, ...); ``api/base.py`` holds
the shared primitives (ApiResponse, HttpError, route compilation, the
BackgroundTask run-slot helper). Handlers are resolved by name through
attribute lookup, so a domain file can be read and tested on its own while
dispatch stays untouched.

The endpoint catalog in DashboardApp's docstring is GENERATED from ROUTES at
import time (first docstring line of each handler becomes the description),
so it can never drift from the table again.

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
import json  # noqa: F401  (back-compat re-export surface)
import logging
import os
import secrets
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote  # noqa: F401

from ..core.config import ToasterLoopConfig
from ..registry.registry import ModelRegistry
from ..serving import instances as srv_inst  # noqa: F401 (test anchor)
from ..store.db import TraceStore
from ..store.paths import resolve_toaster_loop_home
from ..training.scheduler import TrainingConfig
from .api.base import (API_MAX_BODY_BYTES,  # noqa: F401
                       API_MAX_BODY_BYTES_BY_ROUTE, ApiResponse, BackgroundTask,
                       HttpError, Route, SSE_HEARTBEAT_S, _compile_route,
                       _merge_patch, _redact, _utcnow, api_max_body_bytes)
from .api.chat import (CHAT_DIRNAME, MAX_CHAT_RUN_FRAMES,  # noqa: F401
                       _follow_chat_run, _looks_like_error)
from .api.datasets import ROW_PREVIEW_CHARS, _truncate_row  # noqa: F401
from .api.evolve import EvolveRoutes
from .api.config import ConfigRoutes
from .api.control import ControlRoutes
from .api.chat import ChatRoutes
from .api.datasets import DatasetsRoutes
from .api.easy import EasyRoutes
from .api.fsbrowse import FS_READ_MAX_CHARS, FS_TREE_MAX_ENTRIES  # noqa: F401
from .api.fsbrowse import FsBrowseRoutes
from .api.hardwareops import HardwareOpsRoutes
from .api.hub import HubRoutes
from .api.iterate import ITERATE_RUNS_DIRNAME, _validation_cfg_from  # noqa: F401
from .api.iterate import IterateRoutes
from .api.jobs import DEFAULT_JOB_LOG_TAIL, JobsRoutes  # noqa: F401
from .api.models import ModelsRoutes
from .api.overview import OverviewRoutes
from .api.plans import PlansRoutes
from .api.playbooks import PlaybooksRoutes
from .api.profiles import ProfilesRoutes
from .api.projects import PROJECTS_DIRNAME, ProjectsRoutes  # noqa: F401
from .api.remote import RemoteRoutes
from .api.serve import ServeRoutes
from .api.servers import ServersRoutes
from .api.traces import TIMELINE_KINDS, build_trace_timeline  # noqa: F401
from .api.traces import TracesRoutes
from .events import EventBus
from .hardware import HardwareSampler

logger = logging.getLogger("toaster_loop.dashboard")


class DashboardApp(OverviewRoutes, ServeRoutes, RemoteRoutes, ServersRoutes, HardwareOpsRoutes, ConfigRoutes, ControlRoutes, ChatRoutes, TracesRoutes, PlaybooksRoutes, DatasetsRoutes, ModelsRoutes, HubRoutes, JobsRoutes, EasyRoutes, IterateRoutes, EvolveRoutes, ProfilesRoutes, ProjectsRoutes, FsBrowseRoutes, PlansRoutes):
    """JSON API over the toaster_loop home. One instance per server process.

    Domains are mixins under ``dashboard/api/``; this class owns routing and
dependency wiring only. Route handlers resolve by name via getattr.
    """

    def __init__(self, cfg: ToasterLoopConfig, home: Optional[Path] = None):
        self.cfg = cfg
        self.home = Path(home) if home else resolve_toaster_loop_home(cfg)
        # Serialize local submit+run executions (registry single-flight is the
        # cross-process backstop).
        self._exec_lock = threading.Lock()
        # Queue dispatcher + stale-job reaper (started by startup_dispatcher /
        # lazily by ensure_dispatcher; one per process over this home's DB,
        # guarded across processes by its pid-lockfile).
        self._dispatcher: Optional[Any] = None
        self.bus = EventBus()
        # llama.cpp setup/update background task (one at a time per process).
        self._runtime_task = BackgroundTask(self.bus, "runtime.setup")
        # Engine-catalog installs (Hardware page "Install engines"): one
        # background install per recipe id; each gets its own slot so two
        # different engines can install concurrently but one cannot double-run.
        self._engine_install_lock = threading.Lock()
        self._engine_install_tasks: Dict[str, BackgroundTask] = {}
        # Tuning panel: bench + calibrate background runs (one of each; both
        # drive a llama-server of their own and must not fight over the GPU).
        self._bench_task = BackgroundTask(self.bus, "bench.run", log_cap=200,
                                          final_lines=False)
        self._calibrate_task = BackgroundTask(self.bus, "calibrate.run",
                                              log_cap=400, final_lines=False)
        # Doctor run (WP7 checks). Background like bench/calibrate: rocm-smi
        # on a busy GPU can block in the kernel past its subprocess timeout,
        # which would hang the HTTP handler indefinitely. No log buffer — the
        # report IS the result.
        self._doctor_task = BackgroundTask(self.bus, "doctor.run",
                                           log_cap=None)
        self.hardware = HardwareSampler(self.home)
        self._chat_runs: Dict[str, Dict[str, Any]] = {}
        # Guards the single-flight scan-and-insert below (two concurrent POSTs
        # for the same thread must not both pass the in-flight check).
        self._chat_runs_lock = threading.Lock()
        # WP2: delegated child runs stream through this app's chat-run
        # machinery (run registry + frame buffer + adopt/re-attach) so a
        # sub-thread opens live mid-run exactly like a main thread. Re-setting
        # the bridge per instance is safe: one DashboardApp per process.
        self.register_child_bridge()
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

    ROUTES: List[Tuple[str, str, str]] = [
        # status & events
        ("GET", "/api/status", "h_status"),
        ("GET", "/api/events", "h_events"),
        # Prometheus text exposition (review idea #5) — same auth/loopback
        # rules as every other /api/* route (enforced in server.py).
        ("GET", "/api/metrics", "h_metrics"),
        # serve / deploy — start/attach moved to /api/servers/<id>/start (the
        # reserved default instance covers the legacy single-server flow).
        ("GET", "/api/serve/status", "h_serve_status"),
        ("POST", "/api/serve/stop", "h_serve_stop"),
        ("POST", "/api/deploy", "h_deploy"),
        ("POST", "/api/serve/deploy", "h_deploy"),
        # remote toaster_loop peers (LAN): probe + remember another machine's
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
        # D-17 Phase 2: force an inbox delivery attempt now (spec §4.5)
        ("POST", "/api/chat/wake", "h_chat_wake"),
        ("GET", "/api/chat/runs", "h_chat_runs"),
        ("GET", "/api/chat/runs/<run_id>/stream", "h_chat_run_stream"),
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
        # WP3b: `datasets` package preflight + one-click install (static
        # paths must precede <dataset_id> like the diff route below).
        ("GET", "/api/datasets/env", "h_datasets_env"),
        ("POST", "/api/datasets/env/install", "h_datasets_env_install"),
        # WP3c/d: hub-snapshot import + queued folder/website curation.
        ("POST", "/api/datasets/import_hub", "h_dataset_import_hub"),
        ("POST", "/api/datasets/curate_source", "h_curate_source"),
        # static path before <dataset_id> (idea #6: version diff)
        ("GET", "/api/datasets/diff", "h_dataset_diff"),
        ("GET", "/api/datasets/<dataset_id>", "h_dataset_get"),
        ("DELETE", "/api/datasets/<dataset_id>", "h_dataset_delete"),
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
        # resource hub: models/datasets/MCP search, download, manage
        # (core/modelhub.py)
        ("GET", "/api/hub/search", "h_hub_search"),
        ("GET", "/api/hub/feed", "h_hub_feed"),
        ("GET", "/api/hub/mcp_dir", "h_hub_mcp_dir"),
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
        # WP12 plans: per-project task graphs (summary+tasks+validate report;
        # POST op-routed: add/update/set-status/depends/validate/import/
        # expand/start/loop-start/loop-cancel). 404 when plans.enabled is off.
        ("GET", "/api/plans/<project_id>", "h_plan_get"),
        ("POST", "/api/plans/<project_id>", "h_plan_post"),
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

    def _store(self) -> TraceStore:
        return TraceStore(self.home / "buffer" / "toaster_loop.db")

    def _registry(self) -> ModelRegistry:
        return ModelRegistry(self.home / "buffer" / "toaster_loop.db")

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
        db_path = self.home / "buffer" / "toaster_loop.db"

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
                "tools are contained to it (plus TOASTER_LOOP_HOME and temp); set "
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
          2. env ``TOASTER_LOOP_DASHBOARD_TOKEN`` (empty value also disables);
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
        env = os.environ.get("TOASTER_LOOP_DASHBOARD_TOKEN")
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


# ------------------------------------------------------- endpoint catalog
# Generated from ROUTES at import time (kills the hand-maintained doc table).

def _endpoint_catalog(routes: List[Tuple[str, str, str]],
                      handlers: Dict[str, Any]) -> str:
    """Render the endpoint catalog: METHOD path — first docstring line."""
    rows: List[Tuple[str, str, str]] = []
    seen = set()
    for method, path, handler_name in routes:
        key = (method, path)
        if key in seen:
            continue  # alias routes (e.g. /api/serve/deploy) share a handler
        seen.add(key)
        doc = (getattr(handlers.get(handler_name), "__doc__") or "").strip()
        desc = doc.splitlines()[0].rstrip(".") if doc else ""
        rows.append((method, path, desc))
    width_m = max(len(r[0]) for r in rows)
    width_p = max(len(r[1]) for r in rows)
    out = ["Endpoint catalog (generated from ROUTES at import time):", ""]
    for method, path, desc in rows:
        line = f"  {method:<{width_m}} {path:<{width_p}}"
        if desc:
            line += f"  {desc}"
        out.append(line.rstrip())
    return "\n".join(out)


DashboardApp.__doc__ = (
    (DashboardApp.__doc__ or "").rstrip("\n")
    + "\n\n" + _endpoint_catalog(DashboardApp.ROUTES,
                                  {name: getattr(DashboardApp, name, None)
                                   for _, _, name in DashboardApp.ROUTES})
    + "\n"
)

"""Chat: raw proxy passthrough, the agent SSE stream with run
lifecycle (adopt-orphans / re-attach), and thread persistence
(<home>/chat/<id>.json).
"""

# Domain slice of the former monolithic dashboard/app.py: a pure move,
# no behavior change. Handlers resolve through DashboardApp attribute
# lookup (see ROUTES in app.py).

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple


from ...serving import instances as srv_inst
from ..events import sse_frame
from .base import (ApiContext, HttpError, SSE_HEARTBEAT_S, ApiResponse,
                   logger, _utcnow)

#: Cap on the per-run SSE frame buffer: a re-attaching reader that falls
#: further behind than this skips frames (announced via a ``stream.note``
#: event in _follow_chat_run — finding #11).
MAX_CHAT_RUN_FRAMES = 5000

#: Directory under home holding thread JSON files.
CHAT_DIRNAME = "chat"

#: Grace window before a registered-but-never-admitted run (its SSE stream
#: was rejected by the server's admission control before gen() ever ran) is
#: evicted by h_chat_runs, so it can never wedge a thread via single-flight.
_STALE_UNADMITTED_S = 300.0


class ChatRoutes(ApiContext):
    """Chat: raw proxy passthrough, the agent SSE stream with run
lifecycle (adopt-orphans / re-attach), and thread persistence
(<home>/chat/<id>.json).
    """

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
            from ...serving.router import routing_config as _routing_cfg
            _rc = _routing_cfg(self.cfg)
        except Exception:  # noqa: BLE001 — a broken section never breaks runs
            _rc = None
        if _rc is not None and _rc.enabled:
            try:
                from ...serving.router import AutoRouter
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
        # Merged default-instance view (legacy serve_state.json folded in).
        st = srv_inst.read_state_merged(self.home, srv_inst.DEFAULT_ID)
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
                             "(POST /api/servers/default/start) or attach one",
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

    def _push_remote_trace(self, prof: Dict[str, Any], prompt: str,
                           res: Any, t0: float) -> None:
        """Send one finished remote-agent run back to the model's owner.

        Thin wrapper over ``peers.push_remote_trace`` (shared with delegated
        children — see core/delegation.py). Best-effort by contract: any
        failure is logged and swallowed by the caller."""
        from .. import peers as peers_mod

        peers_mod.push_remote_trace(prof, prompt, res, t0)

    def h_proxy_chat(self, q: Dict[str, str], b: Dict[str, Any]) -> ApiResponse:
        import httpx

        # WP4 watchdog: this proxy is one of the two paths that put requests
        # in flight against a served endpoint (the other is the agent runner's
        # chat calls). The tally is a cheap dict counter — inert unless
        # serve.watchdog_enabled is on.
        from ...serving.watchdog import track_request

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
            # Quiet-upstream heartbeat (parity with _follow_chat_run): a slow
            # model keeps intermediate proxies from timing the connection out.
            last_beat = time.monotonic()
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
                            if time.monotonic() - last_beat >= SSE_HEARTBEAT_S:
                                last_beat = time.monotonic()
                                yield ": ping\n\n"
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
            from ...core.images import attach_marker, save_data_url

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
        # Remote LAN agents (`<agent>@<host>`): inference runs on the peer's
        # GPU through its dashboard's /remote/v1 passthrough (authed with the
        # peer token from peers.json). The agent loop — tools, workspace,
        # project files, traces — stays entirely local; only completions
        # cross the network. A never-probed/offline peer is a clean 503 here.
        # Extra headers merged into every runner request. For remote agents
        # this carries X-Toaster-Loop-Instance so a multi-instance peer routes the
        # completion to the SAME instance the snapshot named (the selector
        # rides a header because the runner builds URLs as
        # f"{base_url}/chat/completions" — a ?instance= query would land
        # mid-path). None keeps every existing call byte-identical.
        extra_headers: Optional[Dict[str, str]] = None
        if prof and prof.get("remote"):
            from urllib.parse import urlparse as _urlparse

            from ...serving.supervisor import healthcheck as _hc

            purl = str(prof.get("peer_url") or "")
            base = purl.rstrip("/") + "/remote/v1"
            key = prof.get("peer_token") or None
            model = str(prof.get("model_name") or "model")
            remote_inst = str(prof.get("remote_instance") or "").strip()
            if remote_inst:
                extra_headers = {"X-Toaster-Loop-Instance": remote_inst}
            if not purl or not _hc(base, timeout=5.0, api_key=key):
                raise HttpError(
                    503, f"remote toaster_loop at {_urlparse(purl).netloc or purl} is "
                    "not serving right now (its dashboard must be up with "
                    "expose_lan on, and a model running)",
                    code="peer_unreachable")
        else:
            # Only pass the kwarg when set: callers/stubs written against the
            # pre-routing signature keep working untouched (default path call
            # shape is byte-identical to before).
            base, key, model = self._resolve_chat_target(
                (prof or {}).get("server_instance"),
                **({"routing_hint": _routing_hint} if _routing_hint else {}))
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

        # WP12 plans: an optional plan task id on the run body stamps every
        # trace this run records (additive plan_task field), so traces link
        # task<->trace both ways. Absent => None => identical to before.
        _plan_task = str(b.get("plan_task") or "").strip() or None

        # D-17 Phase 2: the run-spawn body below (single-flight check through
        # SSE stream admission) is extracted VERBATIM into _spawn_chat_run so
        # the inbox wake path can reuse it — pure refactor, no behavior
        # change. drain_inbox=True implements the "user wins" rule (§4.3):
        # a real user message consumes pending delegation-inbox entries into
        # its own prompt block instead of waiting on a wake run.
        return self._spawn_chat_run(
            prompt=prompt, prof=prof, base=base, key=key, model=model,
            thread_id=thread_id, history=history,
            previous_summary=previous_summary, prof_tools=prof_tools,
            project_id=project_id, proj=proj,
            sampling_override=sampling_override, ctx_window=ctx_window,
            extra_headers=extra_headers, plan_task=_plan_task,
            drain_inbox=True)

    def _spawn_chat_run(self, *, prompt: str,
                        prof: Optional[Dict[str, Any]],
                        base: str, key: Optional[str], model: str,
                        thread_id: Optional[str],
                        history: Any = None,
                        previous_summary: Optional[str] = None,
                        prof_tools: Optional[List[str]] = None,
                        project_id: Optional[str] = None,
                        proj: Optional[Dict[str, Any]] = None,
                        sampling_override: Optional[Dict[str, Any]] = None,
                        ctx_window: int = 0,
                        extra_headers: Optional[Dict[str, str]] = None,
                        plan_task: Optional[str] = None,
                        run_id: Optional[str] = None,
                        inbox_wake: bool = False,
                        drain_inbox: bool = False,
                        register_now: bool = False) -> ApiResponse:
        """Spawn one chat run and return the streaming SSE response.

        Extracted verbatim from ``h_chat_stream`` (D-17 Phase 2, spec §4.1:
        "factored by EXTRACTING the existing run-spawn body ... used by both
        handlers"). Both the normal user-message path and the inbox wake path
        (``start_inbox_wake_run``) call this; every preserved invariant lives
        here: the per-thread single-flight 409 check, SSE frame buffering with
        deferred registration, the cancel Event registered as
        ``delegation_context["cancel_event"]``, end-of-run transcript
        persistence, and one ``runner.run()`` = one trace.

        Wake-only knobs (defaults keep the user path byte-identical):
        ``run_id`` pre-selects the id (so delivered_run_id can be stamped
        before the worker starts), ``inbox_wake`` marks
        ``delegation_context["inbox_wake"] = True`` (trace metadata only), and
        ``drain_inbox`` folds any pending inbox entries into the prompt.
        Raises HttpError(409) when the thread already has an in-flight run —
        wake callers catch it and leave their entries pending.
        """
        if run_id is None:
            run_id = uuid.uuid4().hex[:12]

        def _new_run() -> Dict[str, Any]:
            return {
                "cancel": cancel, "done": False, "thread": thread_id,
                "model": model, "profile": prof.get("name") if prof else None,
                # Re-attaching panels render the in-flight user turn from the
                # registry (the transcript is only persisted when the run
                # ends), so the prompt travels with the run record.
                "prompt": prompt,
                "status": "running",
                "frames": [], "dropped": 0, "cond": threading.Condition(),
                # gen() flips this once the stream is admitted and its worker
                # starts (both paths). An entry that stays un-admitted long
                # after spawn — rejected stream, crashed wake caller — gets
                # evicted by h_chat_runs so it can never wedge single-flight.
                "admitted": False, "registered_at": time.monotonic(),
            }

        cancel = threading.Event()
        run: Optional[Dict[str, Any]] = None
        # Single-flight per thread: a second POST for a thread with an
        # in-flight run gets a clean 409 carrying the active run_id, instead
        # of silently spawning a duplicate agent (the orphan-duplicate cascade
        # — user resends a message they believed was stopped). With
        # register_now the scan AND the registry insert happen under ONE lock
        # hold: the wake path's entries are consumed outside this function,
        # so leaving a check->insert gap would let a concurrent user POST
        # start a second run on the same thread (finding #2).
        if thread_id:
            with self._chat_runs_lock:
                for rid, r in self._chat_runs.items():
                    if not r.get("done") and r.get("thread") == thread_id:
                        raise HttpError(409,
                                        "a run is already in flight for this "
                                        "thread — stop it first",
                                        code="run_in_flight",
                                        extra={"run_id": rid})
                if register_now:
                    run = _new_run()
                    self._chat_runs[run_id] = run
        # D-17 §4.3 row 1 ("user wins"): a real user message on a thread with
        # pending inbox entries drains them into THIS turn's prompt (the user
        # never waits on a wake). Runs only after the single-flight check, so
        # entries are consumed by exactly one consumer.
        drained = False
        if drain_inbox and thread_id:
            prompt, drained = self._drain_inbox_into_prompt(
                thread_id, prompt, run_id)
            if drained and run is not None:
                run["prompt"] = prompt
        # Ping-pong guard (spec §10 risk #2): any real user-initiated run on
        # the thread clears its consecutive-wake streak.
        if thread_id and not inbox_wake:
            try:
                from ...core import delegation as _delegation

                _delegation.reset_wake_streak(thread_id)
            except Exception:  # noqa: BLE001 — bookkeeping never breaks a run
                pass
        # Wake hook registration (spec §4.1): the dashboard owns run startup,
        # so it registers itself lazily per active thread as the wake
        # deliverer. Hook-less threads (CLI, no dashboard yet) simply keep
        # their entries pending until a user message drains them.
        if thread_id:
            try:
                from ...core import delegation as _delegation

                # Plain parameter (not the `_t=thread_id` default trick): the
                # dispatcher calls hook(thread_id) with exactly this id, and a
                # defaulted lambda is un-inferable for mypy.
                _delegation.register_wake_hook(
                    thread_id,
                    lambda _tid: self.start_inbox_wake_run(_tid))
            except Exception:  # noqa: BLE001 — wake wiring never breaks a run
                pass
        # Registry entry is created INSIDE gen() (below), not here: an SSE
        # admission rejection (server.py sse_slots cap) never starts the
        # generator, and an eagerly-created entry would then never be marked
        # done — permanently 409-ing the thread via the single-flight check.

        from ...store.recorder import make_recorder

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
        # Phase 3 (spec §8): wake runs carry an `inbox_wake` marker in the
        # trace header's delegation block so the traces panel can badge them.
        recorder = make_recorder(
            self.cfg, on_record=_on_record, project=project_id,
            profile=rec_profile, plan_task=plan_task,
            metadata=({"inbox_wake": True} if inbox_wake else None))

        def gen() -> Iterator[str]:
            # First iteration = the stream was admitted: register now so a
            # rejected (never-started) stream leaves no phantom run behind
            # (the deferred path). The wake path (register_now) already
            # inserted its entry under the single-flight lock; here we only
            # flip `admitted`.
            # Run lifecycle: the registry entry owns a frame buffer + a
            # condition variable. The worker thread publishes every SSE frame
            # into it; readers (this generator, or a panel re-attaching via
            # GET /api/chat/runs/<id>/stream) follow from any cursor and may
            # come and go freely — a reader leaving NEVER stops the run and
            # never marks it done. Only the worker marks done, after it has
            # persisted the complete transcript.
            nonlocal run
            if run is None:
                run = _new_run()
                with self._chat_runs_lock:
                    self._chat_runs[run_id] = run
            # gen() running at all == the stream was admitted. Wake-path
            # entries briefly carry admitted=False between spawn and this
            # first read; that window is exactly what the stale eviction in
            # h_chat_runs protects (a wake caller that never reads).
            run["admitted"] = True
            entry: Dict[str, Any] = run

            def _emit(frame: str) -> None:
                with entry["cond"]:
                    frames = entry["frames"]
                    frames.append(frame)
                    if len(frames) > MAX_CHAT_RUN_FRAMES:
                        frames.pop(0)
                        entry["dropped"] += 1
                    entry["cond"].notify_all()

            def _finish() -> None:
                with entry["cond"]:
                    if entry.get("status") != "cancelling":
                        entry["status"] = "completed"
                    entry["done"] = True
                    entry["cond"].notify_all()

            t0 = time.monotonic()
            _emit(sse_frame("run.started",
                            {"run_id": run_id, "model": model,
                             "profile": prof.get("name") if prof
                             else None,
                             # Context tracker: the window the
                             # chat UI renders usage against.
                             "context_window": ctx_window,
                             # Authoritative user turn (finding #7): a
                             # "user wins" drain prepends the inbox block to
                             # the typed prompt before anything reaches this
                             # frame, and wake turns are synthetic entirely.
                             # The client replaces its locally-pushed raw
                             # message with this so its in-memory copy and
                             # the server-persisted transcript never diverge.
                             "prompt": prompt,
                             "inbox": bool(inbox_wake or drained)}))

            # reference-style live streaming: the runner executes on a worker
            # thread and forwards token/reasoning/tool events through _sink.
            # Navigation resilience: the whole turn is accumulated here (the
            # sink runs on the worker thread, so no locking is needed) so the
            # transcript can be persisted server-side when the run finishes —
            # regardless of who is still reading the stream. Runners without
            # streaming support (test stubs, CLI shims) produce no live
            # events; the worker synthesizes the protocol from the returned
            # message list, exactly like before.
            live = {"n": 0}
            acc: Dict[str, Any] = {"text": "", "reasoning": "", "tools": []}

            def _sink(kind: str, data: Dict[str, Any]) -> None:
                live["n"] += 1
                try:
                    if kind == "message.delta":
                        acc["text"] += str(data.get("text") or "")
                    elif kind == "message.reset":
                        # stream->non-stream fallback: the authoritative text
                        # replaces everything streamed so far.
                        acc["text"] = str(data.get("text") or "")
                    elif kind == "reasoning.delta":
                        acc["reasoning"] += str(data.get("text") or "")
                    elif kind == "tool.call":
                        acc["tools"].append({"id": data.get("id", ""),
                                             "name": data.get("name", ""),
                                             "args": data.get("args") or {},
                                             "result": None})
                    elif kind == "tool.result":
                        for t in reversed(acc["tools"]):
                            if t["id"] == data.get("id"):
                                t["result"] = {"ok": bool(data.get("ok")),
                                               "preview": str(
                                                   data.get("preview") or ""),
                                               "truncated": bool(
                                                   data.get("truncated"))}
                                break
                except Exception:  # accumulation must never kill the run
                    pass
                _emit(sse_frame(kind, data))

            def _worker() -> None:
                try:
                    runner = _make_runner()
                    # Attach the live sink (plain attribute on the built-in
                    # runner; harmless for stubs that never call it).
                    try:
                        runner.on_event = _sink
                    except Exception:
                        pass
                    res = (runner.run(prompt, history=history,
                                      previous_summary=previous_summary)
                           if previous_summary else
                           runner.run(prompt, history=history))
                    # Persist BEFORE announcing done: an adoption poll (or a
                    # thread reload) that sees the run leave the registry can
                    # then re-read a complete transcript immediately.
                    if thread_id:
                        try:
                            self._persist_run_turn(
                                thread_id, prompt, acc, res,
                                {"trace_id": res.trace_id,
                                 "tokens_in": res.tokens_in,
                                 "tokens_out": res.tokens_out,
                                 "duration_s": round(time.monotonic() - t0, 3),
                                 "backend": res.backend,
                                 "failure_reason": res.failure_reason},
                                prof.get("name") if prof else None,
                                inbox=bool(inbox_wake or drained),
                                run_entry=entry)
                        except Exception as e:  # persist must never eat the run
                            # WP1c: WARNING (not debug) with the path — a lost
                            # transcript must be diagnosable from logs alone.
                            logging.getLogger("toaster_loop.dashboard").warning(
                                "chat transcript persist failed for thread %s "
                                "(%s/%s/%s.json): %s",
                                thread_id, self.home, CHAT_DIRNAME, thread_id,
                                e, exc_info=True)
                    # Training flow-back: when the run used a REMOTE agent's
                    # model, that model's owner trains on it — push the
                    # transcript to the peer gateway's trace import (best
                    # effort; a dead peer never breaks the local run). The
                    # profile name is mapped back to the peer's local name
                    # ("tim@host" -> "tim") so their easy-training scoping
                    # matches, exactly like one of their own chats.
                    if prof and prof.get("remote"):
                        try:
                            self._push_remote_trace(prof, prompt, res, t0)
                        except Exception:  # noqa: BLE001 — flow-back is best-effort
                            logging.getLogger("toaster_loop.dashboard").debug(
                                "remote trace push failed (ignored)",
                                exc_info=True)
                    if live["n"] == 0:
                        # Runner without streaming support (stubs/CLI shims):
                        # synthesize the event protocol from the message list.
                        for m in res.messages:
                            role = m.get("role")
                            if role == "assistant":
                                content = m.get("content")
                                if isinstance(content, str) and content:
                                    _emit(sse_frame("message.delta",
                                                    {"text": content}))
                                for tc in (m.get("tool_calls") or []):
                                    fn = tc.get("function") or {}
                                    try:
                                        args = json.loads(
                                            fn.get("arguments") or "{}")
                                    except json.JSONDecodeError:
                                        args = {"raw": fn.get("arguments")}
                                    _emit(sse_frame("tool.call", {
                                        "id": tc.get("id", ""),
                                        "name": fn.get("name", ""),
                                        "args": args}))
                            elif role == "tool":
                                content = m.get("content")
                                if not isinstance(content, str):
                                    content = json.dumps(content, default=str)
                                preview = content[:8000]
                                _emit(sse_frame("tool.result", {
                                    "id": m.get("tool_call_id", ""),
                                    "name": m.get("name"),
                                    "ok": not _looks_like_error(content),
                                    "preview": preview,
                                    "truncated": len(content) > 8000,
                                }))
                    _emit(sse_frame("turn.done",
                                    {"completed": res.completed}))
                    _emit(sse_frame("run.done", {
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
                    }))
                except Exception as e:  # surfaced to every reader as run.error
                    _emit(sse_frame("run.error", {
                        "message": f"{type(e).__name__}: {e}"}))
                finally:
                    _finish()
                    # D-17 §4.3: a run ending frees the thread's single-flight
                    # slot — nudge the wake watcher so queued inbox entries
                    # are delivered now instead of at the next retry tick.
                    if thread_id:
                        try:
                            from ...core import delegation as _delegation

                            _delegation.nudge_delivery()
                        except Exception:  # noqa: BLE001 — never eat the run
                            pass

            def _make_runner():
                from ...core.agent_runner import create_runner

                return create_runner(self.cfg, base, model, api_key=key,
                                     recorder=recorder,
                                     system_prompt=(prof or {}).get(
                                         "system_prompt") or None,
                                     enabled_tool_names=prof_tools,
                                     # Phase 5: route search_codebase to the
                                     # active project's index DB.
                                     project_id=project_id,
                                     # Projects: the record (description +
                                     # agent roster/roles) injected into the
                                     # system prompt so the agent knows the
                                     # project and every agent's job.
                                     project_context=proj,
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
                                        # Audit §2 bug B: parent-run cancel
                                        # propagation — h_chat_cancel /
                                        # shutdown_chat_runs set this event;
                                        # execute_delegation watches it and
                                        # flips the child's cooperative
                                        # cancel so a Stop on the parent also
                                        # stops in-flight delegations.
                                        "cancel_event": cancel,
                                        # D-17 Phase 2: trace metadata only —
                                        # True when this run was started by
                                        # the inbox wake watcher.
                                        "inbox_wake": bool(inbox_wake),
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
                                      cancel_event=cancel,
                                      # Remote agents: X-Toaster-Loop-Instance so a
                                      # multi-instance peer routes to the
                                      # snapshot's instance. None otherwise.
                                      extra_headers=extra_headers)
            # The worker owns the run's whole lifecycle (create -> stream ->
            # persist -> finish). This generator is ONLY a reader: dropping it
            # (client navigated away, broken pipe) never affects the run.
            threading.Thread(target=_worker, daemon=True).start()
            yield from _follow_chat_run(run)

        return ApiResponse(200, stream=gen())

    # --------------------------------------------------- D-17 Phase 2: inbox

    def _drain_inbox_into_prompt(self, thread_id: str, prompt: str,
                                 run_id: Optional[str] = None,
                                 ) -> Tuple[str, bool]:
        """Consume any pending delegation-inbox entries for the thread into a
        context block prepended to the user's prompt (the "user wins" rule,
        D-17 §4.3 / spec §4.4 — same header as a wake turn). Marks them
        delivered against this run id. Returns ``(prompt, drained)`` so the
        caller can mark the persisted turn as an inbox turn without content
        sniffing (finding #6/#7). Best-effort: store failures leave the prompt
        untouched (a broken inbox must never break a user message)."""
        try:
            from ...core import delegation as _delegation

            rows = _delegation.delegation_store.take_pending_inbox(
                thread_id, cfg=self.cfg)
        except Exception:  # noqa: BLE001
            return prompt, False
        if not rows:
            return prompt, False
        try:
            _delegation.delegation_store.mark_delivered_run(
                [str(r.get("delegation_id")) for r in rows],
                run_id or "user-turn", cfg=self.cfg)
        except Exception:  # noqa: BLE001 — provenance is best-effort
            pass
        return (_delegation.format_inbox_message(rows) + "\n\n" + prompt, True)

    def start_inbox_wake_run(self, thread_id: str) -> Dict[str, Any]:
        """D-17 wake path: start a FRESH chat run on the thread whose user
        message is the synthetic "delegation results arrived" block built from
        ``take_pending_inbox``. The wake run is an ordinary run in every
        respect (own run_id + cancel Event, transcript persistence, one
        trace) carrying ``delegation_context["inbox_wake"] = True``.

        Single-flight is obeyed exactly like a user message: if the thread
        already has an in-flight run NOTHING is consumed and this returns
        ``{started: False, busy: True}`` — entries stay pending for the next
        watcher tick (default ``delegation.wake_retry_seconds``) or are
        drained by a real user message. A wake never 409s outward (nobody
        POSTed); it just doesn't start yet.

        On success publishes the additive bus event ``delegation.wake``
        ``{"thread", "run_id", "delegation_ids"}`` and stamps
        ``delivered_run_id`` on each consumed entry. Returns a plain dict;
        never raises (the watcher thread must survive anything)."""
        from ...core import delegation as _delegation

        with self._chat_runs_lock:
            for rid, r in self._chat_runs.items():
                if not r.get("done") and r.get("thread") == thread_id:
                    return {"started": False, "busy": True, "run_id": rid}
        try:
            rows = _delegation.delegation_store.take_pending_inbox(
                thread_id, cfg=self.cfg)
        except Exception as e:  # noqa: BLE001
            return {"started": False, "error": f"store unavailable: {e}"}
        if not rows:
            return {"started": False, "empty": True}

        def _requeue() -> None:
            for r in rows:
                try:
                    _delegation.delegation_store.update_delegation(
                        str(r.get("delegation_id")), cfg=self.cfg,
                        inbox_state="pending")
                except Exception:  # noqa: BLE001 — best-effort rollback
                    pass

        prompt = _delegation.format_inbox_message(rows)
        run_id = uuid.uuid4().hex[:12]
        try:
            prof = self._resolve_profile(None)
            base, key, model = self._resolve_chat_target(
                (prof or {}).get("server_instance"))
            if prof and prof.get("model_name"):
                model = str(prof["model_name"])
            _inst_sampling = self._instance_sampling(
                (prof or {}).get("server_instance"))
            _prof_sampling = (prof or {}).get("sampling")
            if not isinstance(_prof_sampling, dict):
                _prof_sampling = {}
            resp = self._spawn_chat_run(
                prompt=prompt, prof=prof, base=base, key=key, model=model,
                thread_id=thread_id, run_id=run_id, inbox_wake=True,
                sampling_override={**_inst_sampling, **_prof_sampling},
                ctx_window=self._instance_context_length(
                    (prof or {}).get("server_instance")),
                # Register inside the single-flight lock: entries are already
                # consumed here, so a check->insert gap would let a concurrent
                # user POST start a second run on this thread.
                register_now=True)
        except HttpError as e:
            # Lost the single-flight race to a real user message (or nothing
            # is serving): nothing started, so nothing was consumed.
            _requeue()
            return {"started": False, "busy": e.code == "run_in_flight",
                    "error": e.message}
        except Exception as e:  # noqa: BLE001 — watcher must never die
            _requeue()
            return {"started": False, "error": f"{type(e).__name__}: {e}"}
        # Admit the stream (first frame): this starts the worker thread.
        # Nobody reads this synthetic stream live; frames buffer for any
        # attached panel exactly like a user run (re-attach via
        # /api/chat/runs/<id>/stream still works). If admission itself blows
        # up before the worker ever ran, drop the registry entry (it would
        # otherwise wedge the thread's single-flight slot until the stale-run
        # sweep) and requeue: delivered-but-never-started must not eat inbox
        # entries.
        stream = resp.stream
        try:
            if stream is not None:
                next(iter(stream))
        except (StopIteration, ValueError):  # pragma: no cover - defensive
            pass
        except Exception as e:  # noqa: BLE001 — watcher must never die
            with self._chat_runs_lock:
                r = self._chat_runs.pop(run_id, None)
            if r is not None and not r.get("done"):
                _requeue()
                return {"started": False,
                        "error": f"admission failed: {type(e).__name__}: {e}"}
        ids = [str(r.get("delegation_id")) for r in rows]
        try:
            _delegation.delegation_store.mark_delivered_run(ids, run_id,
                                                            cfg=self.cfg)
        except Exception:  # noqa: BLE001 — provenance is best-effort
            pass
        self.bus.publish("delegation.wake", {
            "thread": thread_id, "run_id": run_id, "delegation_ids": ids})
        return {"started": True, "run_id": run_id, "delegation_ids": ids}

    def h_chat_wake(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        """POST /api/chat/wake {"thread": id}: force a delivery attempt now
        (spec §4.5) — a deterministic trigger for tests and the UI instead of
        sleeping on the watcher interval. Returns {started, run_id?}."""
        thread_id = str(b.get("thread") or "").strip()
        if not thread_id or not self._THREAD_ID_RE.match(thread_id):
            raise HttpError(400, "thread (valid id) required")
        return self.start_inbox_wake_run(thread_id)

    def h_chat_run_stream(self, q: Dict[str, str], b: Dict[str, Any],
                          run_id: str = "") -> ApiResponse:
        """Re-attach to an in-flight chat run's live SSE stream.

        The chat panel calls this when it reopens a thread whose run is still
        executing (the user navigated away mid-run). The response replays the
        frames buffered so far — the viewer sees everything produced while
        nobody was watching — then follows live until the run finishes.
        Multiple readers may follow one run concurrently; leaving again never
        touches the run itself.
        """
        run = self._chat_runs.get(str(run_id or ""))
        if run is None:
            raise HttpError(404, f"no chat run {run_id}", code="no_such_run")
        return ApiResponse(200, stream=_follow_chat_run(run))

    def h_chat_cancel(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        run_id = str(b.get("run_id", ""))
        # Read + mutate under the registry lock: h_chat_runs prunes done runs
        # under the same lock, so an unlocked get here could race a prune and
        # report a successful cancel for an entry that no longer exists.
        with self._chat_runs_lock:
            run = self._chat_runs.get(run_id)
            if run is None:
                raise HttpError(404, f"no chat run {run_id}")
            cancelling = not run.get("done")
            if cancelling:
                run["cancel"].set()
                run["status"] = "cancelling"
        return {"ok": True, "cancelling": cancelling}

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
            # trace/summary already recorded everything). Also evict a
            # register_now entry whose stream was never admitted after the
            # grace window: its worker never started, so without this it
            # would wedge the thread's single-flight slot forever. Entries
            # predating the flag (test seeds, legacy shapes) are untouched.
            now = time.monotonic()
            for rid in [r for r, v in self._chat_runs.items()
                        if v.get("done")
                        or (v.get("admitted") is False
                            and now - float(v.get("registered_at") or 0.0)
                            > _STALE_UNADMITTED_S)]:
                del self._chat_runs[rid]
            runs = [{
                "run_id": rid,
                "thread": v.get("thread"),
                "model": v.get("model"),
                "profile": v.get("profile"),
                # The re-attaching panel renders the in-flight user turn from
                # this (the transcript file only gets it when the run ends).
                "prompt": v.get("prompt"),
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
                logging.getLogger("toaster_loop.dashboard").warning(
                    "chat run %s still in flight at shutdown (grace expired); "
                    "it dies with this process — no trace will be recorded",
                    rid)
                r["status"] = "interrupted"
        return len(inflight)

    def _persist_run_turn(self, thread_id: str, prompt: str, acc: Dict[str, Any],
                          res: Any, footer: Dict[str, Any],
                          profile_name: Optional[str], *, inbox: bool = False,
                          run_entry: Optional[Dict[str, Any]] = None) -> None:
        """Server-side transcript persistence for a finished chat run.

        Persistence used to be client-only: the starting panel saved the thread
        after its SSE stream ended. Navigating away aborted that stream, so a
        finished run's reply was never written to the thread file (the "agent
        stopped responding" symptom when returning to chat). The worker now
        calls this when the run completes, regardless of who is still reading
        the stream.

        Merge rules keep the connected-client save (which still happens and
        overwrites with equivalent content) and an aborted panel's partial
        state from producing duplicates: everything after the LAST user turn
        equal to this run's prompt is dropped — anything there is either that
        panel's partial rendering of THIS run (mid-stream abort) or the bare
        prompt itself. When no matching user turn exists, the prompt is
        appended before the authoritative assistant turn.

        Runners without live events (test stubs) never filled the accumulator;
        for those the assistant text falls back to ``res.final_response``.

        ``inbox`` stamps the persisted user turn with ``{"inbox": True}`` so
        the UI renders its "⧗ inbox" chip from data instead of regex-sniffing
        content a user could accidentally type (finding #6). ``run_entry`` is
        the run's registry dict: if the thread was deleted while this run was
        in flight (h_thread_delete flips ``thread_deleted``), persisting would
        resurrect the deleted file — skip instead.
        """
        if run_entry is not None and run_entry.get("thread_deleted"):
            return
        p = self._thread_path(thread_id)
        t: Dict[str, Any] = {}
        if p.is_file():
            try:
                t = json.loads(p.read_text())
            except (OSError, json.JSONDecodeError):
                t = {}
        msgs = [m for m in (t.get("messages") or []) if isinstance(m, dict)]

        text = str(acc.get("text") or "")
        tools = list(acc.get("tools") or [])
        if not text and not tools:
            text = str(getattr(res, "final_response", "") or "").strip()
        if not text and not tools:
            return  # the run produced nothing at all — persist no phantom turn

        last_user = next(
            (i for i in range(len(msgs) - 1, -1, -1)
             if msgs[i].get("role") == "user"
             and str(msgs[i].get("content") or "") == prompt), None)
        if last_user is not None:
            # Drop this run's partial leftovers (an aborted panel's
            # half-rendered reply); keep the matched user turn itself and
            # stamp its inbox provenance so the chip survives a reload
            # without content sniffing.
            del msgs[last_user + 1:]
            if inbox:
                msgs[last_user]["inbox"] = True
        else:
            user_msg: Dict[str, Any] = {"role": "user", "content": prompt}
            if inbox:
                user_msg["inbox"] = True
            msgs.append(user_msg)

        assistant: Dict[str, Any] = {"role": "assistant", "content": text}
        reasoning = str(acc.get("reasoning") or "")
        if reasoning:
            assistant["reasoning"] = reasoning
        if tools:
            assistant["tools"] = tools
        clean_footer = {k: v for k, v in (footer or {}).items()
                        if v is not None and k != "summary"}
        if clean_footer:
            assistant["footer"] = clean_footer
        msgs.append(assistant)

        t.setdefault("id", thread_id)
        t["messages"] = msgs
        t["updated"] = _utcnow()
        t.setdefault("created", t["updated"])
        # Last agent profile used on this thread (drives the dropdown on
        # reopen — previously it always fell back to default). None is a
        # legitimate value here: a later default run must clear a stale one.
        t["profile"] = profile_name
        tid = getattr(res, "trace_id", None)
        if tid:
            t["trace_ids"] = sorted(set((t.get("trace_ids") or []) + [str(tid)]))
        summary = getattr(res, "summary", None)
        if isinstance(summary, str) and summary:
            # Backlog #6 parity with the client save path: a run that finished
            # while nobody was reading still keeps its compaction state.
            t["summary"] = summary
        if not t.get("title"):
            from ...core.titles import DERIVED, derive_title
            title = derive_title(msgs)
            if title:
                t["title"] = title
                t["title_source"] = DERIVED

        # Second tombstone check right before the write: a thread deleted
        # during the run must stay deleted.
        if run_entry is not None and run_entry.get("thread_deleted"):
            return
        # WP1: unique-tmp + retried replace (Windows AV/indexer handles and
        # interleaved delegation-wake saves made the old fixed ".json.tmp"
        # collide with PermissionError [WinError 32] — silent thread loss).
        from ...store.atomic import atomic_write_json

        atomic_write_json(p, t)

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
        """List thread summaries. Optional ?limit=N&offset=M paginate the
        listing (each row is one file read; hundreds of threads should not
        have to wait on all of them). Absent params keep the classic
        return-everything behavior byte-identical."""
        d = self.home / CHAT_DIRNAME
        threads = []
        # Phase 3 (D-17): per-thread undelivered inbox counts drive the chat
        # UI's "N waiting" badge / restart banner. One indexed GROUP BY; a
        # store failure degrades to no badges (never breaks the listing).
        try:
            from ...core import delegation as _delegation

            pending = _delegation.delegation_store.pending_inbox_counts(
                self.cfg)
        except Exception:  # noqa: BLE001
            pending = {}
        if d.is_dir():
            files = sorted(d.glob("*.json"))
            # Slice ONLY when a caller passes limit/offset; the default path
            # stays byte-identical (read every file, return everything).
            try:
                off = int(q.get("offset") or 0)
                lim = int(q.get("limit") or 0)
            except (TypeError, ValueError):
                off, lim = 0, 0
            if off > 0 or lim > 0:
                files = files[off: (off + lim) if lim > 0 else None]
            for f in files:
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
                    # Phase 3 (D-17): undelivered delegation-inbox entries —
                    # the chat UI renders this as a small "N waiting" badge.
                    "pending_inbox": pending.get(
                        t.get("id", f.stem), 0),
                    # WP2: delegated child threads carry parent_thread /
                    # delegation_id / agent (additive keys; legacy files read
                    # as None and list exactly like before). The explorer
                    # nests rows with a parent_thread under their parent and
                    # hides them from the top-level count via these keys.
                    "parent_thread": t.get("parent_thread"),
                    "delegation_id": t.get("delegation_id"),
                    "agent": t.get("agent"),
                    "delegation_status": t.get("delegation_status"),
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
        from ...core.titles import DERIVED, USER, derive_title
        existing_source = existing.get("title_source")
        # Both may stay None (thread with no prior title and no client title):
        # annotate so the later derive_title() assignment is legal.
        title_val: Optional[str]
        title_src: Optional[str]
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
            # wipe them. An explicit list replaces (client-authoritative edit),
            # but only dict-shaped entries survive — a stray string/number in
            # the array used to render as "?" owner chips forever.
            "delegations": ([d for d in b["delegations"]
                             if isinstance(d, dict)]
                            if isinstance(b.get("delegations"), list)
                            else existing.get("delegations") or []),
        }
        # WP12 plans: optional plan-task linkage on a thread ("task": "<id>").
        # Written ONLY when set (by the plan "start" op) so ordinary threads
        # stay byte-identical to pre-WP12 records; readers use .get(). An
        # existing task key is preserved across later saves (the chat writer
        # round-trips the whole record). Explicit null clears it.
        if "task" in b:
            if b["task"]:
                thread["task"] = str(b["task"])
        elif existing.get("task"):
            thread["task"] = existing["task"]
        # WP2 additive keys survive client round-trips exactly like "task":
        # a save against a child thread must never orphan it (the bridge and
        # delegation writers own these keys; clients never send them).
        for _k in ("parent_thread", "delegation_id", "agent",
                   "delegation_status"):
            if existing.get(_k) is not None:
                thread[_k] = existing[_k]
        if thread["project"] and self._load_project(thread["project"]) is None:
            raise HttpError(400, f"no project {thread['project']}",
                            code="unknown_project")
        # Atomic durable write (unique tmp + retried os.replace), matching
        # every other persisted JSON in the codebase — a crash mid-write must
        # never leave a partial file that the read path would treat as an
        # empty thread. WP1: the tmp name is per-write unique and the replace
        # retries transient Windows sharing violations (WinError 32) instead
        # of losing the save silently. WARNING on failure with the path so a
        # lost thread is diagnosable from logs alone (WP1c).
        from ...store.atomic import atomic_write_json

        try:
            atomic_write_json(p, thread)
        except OSError as e:
            logger.warning("thread save failed for %s: %s", p, e)
            raise

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
                from ...core.titles import LLM, generate_title, should_apply_title

                p = self._thread_path(thread_id)
                if not p.is_file():
                    return
                t = json.loads(p.read_text())
                if not should_apply_title(t.get("title_source"), LLM):
                    return  # user renamed (or llm already set) while we queued

                # Don't spend a GPU slot while a run is in flight anywhere:
                # the title can wait for the next save or the quiet moment
                # after the run (a busy single-GPU box should never contend
                # with the user's agent for this cosmetic call).
                with self._chat_runs_lock:
                    busy = any(not r.get("done")
                               for r in self._chat_runs.values())
                if busy:
                    return

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
                # WP1: same unique-tmp retried write as the main save path;
                # a concurrent client save used to collide on the fixed tmp.
                from ...store.atomic import atomic_write_json

                atomic_write_json(p, t2)
            except Exception:  # a title upgrade must never surface anywhere
                import logging
                logging.getLogger("toaster_loop.dashboard").debug(
                    "llm title upgrade failed for %s", thread_id, exc_info=True)

        threading.Thread(target=_work, daemon=True).start()

    def h_thread_delete(self, q: Dict[str, str], b: Dict[str, Any],
                        thread_id: str) -> Dict[str, Any]:
        p = self._thread_path(thread_id)
        if not p.is_file():
            raise HttpError(404, f"no thread {thread_id}")
        # Deleting a thread with an in-flight run: ask the run to stop and
        # tombstone its registry entry so _persist_run_turn does not recreate
        # the file we are about to remove (a finished run would otherwise
        # resurrect a deleted transcript).
        with self._chat_runs_lock:
            for r in self._chat_runs.values():
                if r.get("thread") == thread_id and not r.get("done"):
                    r["cancel"].set()
                    r["status"] = "cancelling"
                    r["thread_deleted"] = True
        try:
            from ...core import delegation as _delegation

            _delegation.unregister_wake_hook(thread_id)
        except Exception:  # noqa: BLE001 — cleanup never blocks the delete
            pass
        p.unlink()
        # WP2 lifecycle: deleting a parent deletes its child threads (the
        # explorer would otherwise show orphans pointing at a dead parent).
        # Children are found by their additive parent_thread key; each delete
        # is best-effort. ?keep_children=1 opts out (future UI confirmation).
        if not str(q.get("keep_children") or "").strip():
            deleted = [thread_id]
            d = self.home / CHAT_DIRNAME
            try:
                for f in d.glob("*.json"):
                    try:
                        t = json.loads(f.read_text())
                    except (OSError, json.JSONDecodeError):
                        continue
                    if isinstance(t, dict) and t.get("parent_thread") in deleted:
                        cid = str(t.get("id") or f.stem)
                        with self._chat_runs_lock:
                            for r in self._chat_runs.values():
                                if (r.get("thread") == cid
                                        and not r.get("done")):
                                    r["cancel"].set()
                                    r["status"] = "cancelling"
                                    r["thread_deleted"] = True
                        try:
                            f.unlink()
                            deleted.append(cid)
                        except OSError:  # already gone / locked: skip
                            pass
            except OSError:  # noqa: BLE001 — chat dir vanished mid-delete
                pass
        return {"ok": True}

    # ------------------------------------------------- WP2 child-run bridge

    def register_child_bridge(self) -> None:
        """Wire delegated child runs into this app's chat-run machinery.

        Called once from DashboardApp.__init__ (and re-safely on each new
        app so the most recent instance owns the bridge). The core dispatcher
        calls these hooks around a child run that has a parent chat thread:

          started(info) -> child run id   (registers a run entry exactly like
                                           _spawn_chat_run's _new_run, keyed
                                           to the CHILD thread so single-flight
                                           and adopt/re-attach work unchanged)
          frame(rid, kind, data)          (publishes into the run's frame
                                           buffer -> live SSE for readers)
          finished(rid, payload)          (persists the transcript to the
                                           child thread file via the WP1
                                           atomic writer + closes the stream)

        Threadless/CLI dispatches never reach the bridge; every hook is
        best-effort (core swallows exceptions here regardless)."""
        from ...core import delegation as _delegation

        def _started(info: Dict[str, Any]) -> Optional[str]:
            child_thread = str(info.get("child_thread") or "")
            if not child_thread or not self._THREAD_ID_RE.match(child_thread):
                return None
            run_id = "dlg-" + str(info.get("delegation_id") or "")[:16]
            cancel = threading.Event()
            entry: Dict[str, Any] = {
                "cancel": cancel, "done": False, "thread": child_thread,
                "model": None, "profile": info.get("agent"),
                "prompt": info.get("task") or "",
                "status": "running",
                "frames": [], "dropped": 0, "cond": threading.Condition(),
                "admitted": True, "registered_at": time.monotonic(),
                # WP2 markers: the bridge owns this entry's lifecycle (no SSE
                # POST ever created it), and readers may attach at any time.
                "child_delegation": info.get("delegation_id"),
            }
            with self._chat_runs_lock:
                # Adopt-race guard: a previous run for the same child thread
                # that is somehow still present wins (one run per thread).
                for _rid, r in self._chat_runs.items():
                    if (not r.get("done")
                            and r.get("thread") == child_thread):
                        return None
                self._chat_runs[run_id] = entry

            def _emit(frame: str) -> None:
                with entry["cond"]:
                    frames = entry["frames"]
                    frames.append(frame)
                    if len(frames) > MAX_CHAT_RUN_FRAMES:
                        frames.pop(0)
                        entry["dropped"] += 1
                    entry["cond"].notify_all()

            entry["_emit"] = _emit
            _emit(sse_frame("run.started", {
                "run_id": run_id, "model": None,
                "profile": info.get("agent"),
                "prompt": info.get("task") or "",
                "child_thread": child_thread,
                "parent_thread": info.get("parent_thread"),
                "delegation_id": info.get("delegation_id")}))
            return run_id

        def _frame(run_id: str, kind: str, data: Dict[str, Any]) -> None:
            entry = self._chat_runs.get(str(run_id))
            emit = entry.get("_emit") if entry else None
            if callable(emit):
                emit(sse_frame(kind, data))

        def _finished(run_id: str, payload: Dict[str, Any]) -> None:
            entry = self._chat_runs.get(str(run_id))
            if entry is None:
                return
            acc = payload.get("acc") or {}
            text = str(acc.get("text") or "") or \
                str(payload.get("result") or "")
            child_thread = str(entry.get("thread") or "")
            # Persist the final transcript to the child thread file (WP1
            # atomic path) so a finished sub-thread opens from disk exactly
            # like a main thread. Merge rules mirror _persist_run_turn: drop
            # anything after the seeded task turn, then append the reply.
            if child_thread and self._THREAD_ID_RE.match(child_thread):
                try:
                    p = self._thread_path(child_thread)
                    t: Dict[str, Any] = {}
                    if p.is_file():
                        try:
                            t = json.loads(p.read_text())
                        except (OSError, json.JSONDecodeError):
                            t = {}
                    msgs = [m for m in (t.get("messages") or [])
                            if isinstance(m, dict)]
                    last_user = next(
                        (i for i in range(len(msgs) - 1, -1, -1)
                         if msgs[i].get("role") == "user"), None)
                    if last_user is not None:
                        del msgs[last_user + 1:]
                    assistant: Dict[str, Any] = {"role": "assistant",
                                                 "content": text}
                    reasoning = str(acc.get("reasoning") or "")
                    if reasoning:
                        assistant["reasoning"] = reasoning
                    if acc.get("tools"):
                        assistant["tools"] = list(acc["tools"])
                    footer = {k: v for k, v in {
                        "trace_id": payload.get("child_trace_id"),
                        "delegation_status": payload.get("status")}.items()
                        if v is not None}
                    if footer:
                        assistant["footer"] = footer
                    msgs.append(assistant)
                    t["messages"] = msgs
                    t["updated"] = _utcnow()
                    tid = payload.get("child_trace_id")
                    if tid:
                        t["trace_ids"] = sorted(set(
                            (t.get("trace_ids") or []) + [str(tid)]))
                    from ...store.atomic import atomic_write_json
                    atomic_write_json(p, t)
                except OSError as e:
                    logger.warning("child thread persist failed for %s: %s",
                                   child_thread, e)
            emit = entry.get("_emit")
            if callable(emit):
                emit(sse_frame("run.done", {
                    "trace_id": payload.get("child_trace_id"),
                    "delegation_status": payload.get("status")}))
            with entry["cond"]:
                if entry.get("status") != "cancelling":
                    entry["status"] = str(payload.get("status") or "completed")
                entry["done"] = True
                entry["cond"].notify_all()

        _delegation.register_child_run_bridge(
            {"started": _started, "frame": _frame, "finished": _finished})


# ------------------------------------------- module-level helpers

def _follow_chat_run(run: Dict[str, Any]) -> Iterator[str]:
    """Reader over a chat run's frame buffer, from a fresh cursor.

    Replays every buffered frame (the oldest may have been dropped past
    MAX_CHAT_RUN_FRAMES), then blocks on the run's condition variable until
    new frames arrive or the run finishes. Generators dropped mid-follow
    (client navigated away / broken pipe) simply stop reading — the run and
    its buffer are untouched, so another reader can attach later from zero.

    Entries without a live buffer (seeded stubs in tests, legacy shapes) end
    the stream immediately.
    """
    cond = run.get("cond")
    frames = run.get("frames")
    if not isinstance(cond, threading.Condition) or frames is None:
        return
    cursor = 0
    idle_s = 0.0
    noted = False
    while True:
        with cond:
            base = int(run.get("dropped") or 0)
            if cursor < base:      # buffer rolled over: skip what we missed
                cursor = base
            batch = list(frames[cursor - base:])
            if batch:
                cursor += len(batch)
            finished = bool(run.get("done"))
            if not batch and not finished:
                cond.wait(timeout=1.0)
                idle_s += 1.0
        for frame in batch:
            yield frame
        if base and not noted:
            # Honest replay (finding #11): the reader skipped frames the ring
            # buffer evicted. Announce it exactly once, right after replaying
            # what survived, so a re-attaching panel can say "N frames
            # dropped" instead of silently showing a gapped transcript.
            noted = True
            yield sse_frame("stream.note", {
                "note": f"{base} buffered frame(s) were dropped",
                "dropped": base})
        if batch:
            idle_s = 0.0
        elif not finished and idle_s >= SSE_HEARTBEAT_S:
            idle_s = 0.0
            yield ": ping\n\n"    # keep the connection warm on quiet runs
        # Exit only after the final frames (run.done / run.error) were read.
        if finished and not batch:
            return

def _looks_like_error(content: str) -> bool:
    """Heuristic error flag for the no-stream synthesis path only. Anchored:
    a successful search/read whose body merely *mentions* "exception" (e.g.
    grepping Python code) must not render as a failed tool call, so the loose
    keywords are checked only in the first line."""
    first_line = content.lstrip().split("\n", 1)[0].lower()
    return (first_line.startswith("error") or "traceback" in first_line
            or "exception" in first_line)

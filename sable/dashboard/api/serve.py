"""Serve/deploy control: single-server start/stop/attach/deploy."""

# Domain slice of the former monolithic dashboard/app.py: a pure move,
# no behavior change. Handlers resolve through DashboardApp attribute
# lookup (see ROUTES in app.py).

from __future__ import annotations

from dataclasses import replace as _dc_replace
from typing import Any, Dict


from ...serving import instances as srv_inst
from ...serving.deploy import (
    clear_serve_state,
    deploy_model,
    read_serve_state,
    stop_owned_server,
    write_serve_state,
)
from ...serving.supervisor import ServeError, healthcheck
from .base import ApiContext, ApiResponse, HttpError, _utcnow


class ServeRoutes(ApiContext):
    """Serve/deploy control: single-server start/stop/attach/deploy."""

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
        from ...serving.runtime import pid_alive

        try:
            pid_i = int(pid)
        except (ValueError, TypeError):
            return False
        # None ("alive but not ours") keeps the old PermissionError branch.
        return pid_alive(pid_i) is not False

    def h_serve_status(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        from ...serving.supervisor import crash_loop_status

        st = read_serve_state(self.home)
        # Pass 4 #28: per-backend circuit-breaker states (runner/judge side),
        # keyed by base_url — the dashboard's view of "this backend has been
        # failing and is being failed-fast". Only breakers with activity.
        from ...core.circuit import all_breakers
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
        from ...serving.supervisor import clear_crash_loop, start_instance_server

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

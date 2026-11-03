"""Serve/deploy control: legacy status/stop + model hot-swap deploy.

Start and attach live on the multi-instance API (POST /api/servers/<id>/start;
the reserved "default" instance covers the legacy single-server flow), so the
old /api/serve/start + /api/serve/attach endpoints were removed once no UI
caller remained. /api/serve/stop stays: the Serve panel's status card uses it
to stop/detach whatever the legacy state file points at."""

# Domain slice of the former monolithic dashboard/app.py: a pure move,
# no behavior change. Handlers resolve through DashboardApp attribute
# lookup (see ROUTES in app.py).

from __future__ import annotations

from typing import Any, Dict


from ...serving import instances as srv_inst
from ...serving.deploy import deploy_model
from ...serving.supervisor import ServeError, healthcheck
from .base import ApiContext, ApiResponse, HttpError


class ServeRoutes(ApiContext):
    """Serve/deploy control: legacy status/stop + deploy hot-swap."""

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

        # Merged view: per-instance state file with the legacy serve_state.json
        # folded in (the seam lives in serving/instances.py).
        st = srv_inst.read_state_merged(self.home, srv_inst.DEFAULT_ID)
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

    def h_serve_stop(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        st = srv_inst.read_state_merged(self.home, srv_inst.DEFAULT_ID)
        stopped = False
        if st and st.get("mode", "spawn") == "attach":
            # never kill an attached foreign server — clear both views
            srv_inst.clear_state_merged(self.home, srv_inst.DEFAULT_ID)
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

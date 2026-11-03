"""Multi-instance serving (Phase 6): N named server instances.
The reserved id default mirrors the legacy single-server
serve state so pre-Phase-6 setups keep working untouched.
"""

# Domain slice of the former monolithic dashboard/app.py: a pure move,
# no behavior change. Handlers resolve through DashboardApp attribute
# lookup (see ROUTES in app.py).

from __future__ import annotations

import threading
from typing import Any, Dict, Optional


from ...serving import instances as srv_inst
from ...serving.deploy import read_serve_state, stop_owned_server
from ...serving.supervisor import ServeError, healthcheck
from .base import ApiContext, ApiResponse, HttpError, _utcnow


class ServersRoutes(ApiContext):
    """Multi-instance serving (Phase 6): N named server instances.
The reserved id default mirrors the legacy single-server
serve state so pre-Phase-6 setups keep working untouched.
    """

    def _instance_circuits(self, running: bool,
                           st: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Pass 4 #28: breaker snapshots for one instance's live base_url."""
        if not running or not st or not st.get("base_url"):
            return {}
        from ...core.circuit import all_breakers
        br = all_breakers().get(str(st["base_url"]))
        return {str(st["base_url"]): br} if br else {}

    def _instance_row(self, inst: Dict[str, Any]) -> Dict[str, Any]:
        from ...serving.supervisor import crash_loop_status

        iid = str(inst.get("id"))
        st = srv_inst.read_state_merged(self.home, iid)
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
            # Diffusion engines (sd-server): generation defaults instead of
            # the chat knobs above. Additive field, old readers ignore it.
            "image_settings": inst.get("image_settings") or {},
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
        return srv_inst.read_state_merged(self.home, server_id)

    def _write_instance_state(self, server_id: str,
                              state: Dict[str, Any]) -> None:
        # Merged accessor keeps serve_state.json in sync for the reserved
        # default (old panels and CLI readers still consult it).
        srv_inst.write_state_merged(self.home, server_id, state)

    def _clear_instance_state(self, server_id: str) -> None:
        srv_inst.clear_state_merged(self.home, server_id)

    def _stop_instance(self, server_id: str, forget: bool = False) -> bool:
        """Stop one instance: in-process child first, then recorded pid.
        Attach instances are never killed — only their state is cleared."""
        from ...serving.supervisor import stop_instance as _stop_inproc

        stopped = _stop_inproc(server_id)
        if not stopped:
            stopped = srv_inst.stop_recorded(self.home, server_id)
        if not stopped and server_id == srv_inst.DEFAULT_ID:
            # Legacy single-instance serve (serve_state.json) ownership path.
            st = read_serve_state(self.home)
            if st and st.get("mode", "spawn") != "attach":
                # stop_owned_server operates on the legacy file itself, so it
                # stays a direct deploy call rather than a merged accessor.
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
        from ...serving.watchdog import inflight_count

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
        from ...serving.supervisor import (
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

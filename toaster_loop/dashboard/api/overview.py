"""Overview/status endpoints, the global SSE bus, /api/metrics."""

# Domain slice of the former monolithic dashboard/app.py: a pure move,
# no behavior change. Handlers resolve through DashboardApp attribute
# lookup (see ROUTES in app.py).

from __future__ import annotations

import time
from typing import Any, Dict, Iterator, Optional, Tuple


from ...serving import instances as srv_inst
from ..events import format_sse
from .base import ApiContext, ApiResponse, SSE_HEARTBEAT_S, _redact


class OverviewRoutes(ApiContext):
    """Overview/status endpoints, the global SSE bus, /api/metrics."""

    # Set by DashboardApp.__init__ (this mixin is composed into it); declared
    # here so mypy can type the accesses in h_metrics.
    _metrics_cache: Optional[Tuple[float, str]]

    def h_events(self, q: Dict[str, str], b: Dict[str, Any]) -> ApiResponse:
        """SSE global bus. Heartbeat comments every ~15 s; clean disconnect."""

        def gen() -> Iterator[str]:
            sub = self.bus.subscribe()
            try:
                yield ": toaster_loop events\nretry: 3000\n\n"
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

    def h_status(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        from ... import __version__

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

        # Merged default-instance view (legacy serve_state.json folded in).
        serve_state = srv_inst.read_state_merged(self.home,
                                                 srv_inst.DEFAULT_ID)
        server_live = False
        if serve_state and serve_state.get("pid"):
            # Windows-safe liveness (A1): os.kill(pid, 0) TERMINATES on NT.
            from ...serving.runtime import pid_alive as _pid_alive

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

        # Peer-gateway address for the Serve page: when expose_lan is on,
        # this dashboard serves the OpenAI-compatible /remote/v1 passthrough,
        # so other machines' toaster_loop can attach our models over the LAN. The
        # token is deliberately NOT included (the UI never needs it; peers
        # store it at connect time).
        gateway = None
        try:
            from ..server import _lan_ip, expose_lan

            if expose_lan(self.cfg) and self.token:
                lan_ip = _lan_ip()
                dport = getattr(self, "dashboard_port", None)
                if lan_ip and dport:
                    gateway = f"http://{lan_ip}:{dport}/remote/v1"
        except Exception:  # noqa: BLE001 — status never breaks on this
            gateway = None

        return {
            "version": __version__,
            "home": str(self.home),
            "peer_gateway": gateway,
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

    def h_metrics(self, q: Dict[str, str], b: Dict[str, Any]) -> ApiResponse:
        """Prometheus text exposition over the buffer/registry DB (idea #5).

        Raw-text ApiResponse; server.py writes it with the declared content
        type. Auth is NOT handled here — every /api/* request passes the same
        token + loopback/CSRF gate in server.py first, so scrapers send
        X-Toaster-Loop-Token (or ?token=) exactly like any other endpoint. The
        rendered body is memoized for METRICS_TTL_S seconds (scrape-friendly;
        force refresh with ?refresh=1).
        """
        from ..metrics import CONTENT_TYPE, METRICS_TTL_S, collect

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

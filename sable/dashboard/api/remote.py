"""Remote LAN sable peers: probe + remember another machine's
dashboard so its agents/servers/training/traces show up here.
Outbound-only GET polling; see dashboard/peers.py.
"""

# Domain slice of the former monolithic dashboard/app.py: a pure move,
# no behavior change. Handlers resolve through DashboardApp attribute
# lookup (see ROUTES in app.py).

from __future__ import annotations

from typing import Any, Dict


from .base import ApiContext, ApiResponse, HttpError


class RemoteRoutes(ApiContext):
    """Remote LAN sable peers: probe + remember another machine's
dashboard so its agents/servers/training/traces show up here.
Outbound-only GET polling; see dashboard/peers.py.
    """

    def h_remote_probe(self, q: Dict[str, str],
                       b: Dict[str, Any]) -> ApiResponse:
        from .. import peers as peers_mod

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
        from .. import peers as peers_mod

        return {"peers": [peers_mod.peer_view(p)
                          for p in peers_mod.load_peers(self.home)]}

    def h_remote_peer_live(self, q: Dict[str, str], b: Dict[str, Any],
                           peer_id: str) -> Dict[str, Any]:
        """Re-probe one remembered peer and store the fresh snapshot."""
        from .. import peers as peers_mod

        rec = peers_mod.get_peer(self.home, peer_id)
        if rec is None:
            raise HttpError(404, f"no peer {peer_id}", code="unknown_peer")
        snap = peers_mod.probe(str(rec["url"]), rec.get("token"))
        peers_mod.set_snapshot(self.home, peer_id, snap)
        fresh = peers_mod.get_peer(self.home, peer_id) or rec
        return {"peer": peers_mod.peer_view(fresh)}

    def h_remote_peer_delete(self, q: Dict[str, str], b: Dict[str, Any],
                             peer_id: str) -> Dict[str, Any]:
        from .. import peers as peers_mod

        if not peers_mod.remove_peer(self.home, peer_id):
            raise HttpError(404, f"no peer {peer_id}", code="unknown_peer")
        self.bus.publish("peers.changed", {"event": "removed",
                                           "peer_id": peer_id})
        return {"ok": True}

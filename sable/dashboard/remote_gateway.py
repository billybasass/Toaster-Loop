"""Peer-facing OpenAI-compatible passthrough — ``/remote/*`` on the dashboard.

Why: a peer machine's sable agents need to use THIS box's served model, but
llama-server binds 127.0.0.1 on the peer by design (the loopback invariant in
serving/supervisor.py is deliberate and stays). The dashboard is already the
sanctioned, token-authed network surface when ``dashboard.expose_lan`` is on,
so inference for remote agents rides through it instead of opening a second
unauthenticated port.

Routes (mounted by server.py; dead 404s unless expose_lan is on):

    POST /remote/v1/chat/completions   -> the serve endpoint's chat route
    GET  /remote/v1/models             -> the serve endpoint's model list
    POST /remote/tokenize              -> engine-root /tokenize
    POST /remote/v1/traces/import      -> re-record a peer's trace HERE

Trace import (training flow-back): when someone chats with ``tim@peer`` from
another box, the agent loop and its traces live there. Easy training on THIS
machine needs that data, so the pushing side posts the finished transcript to
this route; it lands in our buffer via TraceRecorder with the local agent
name stamped (``tim@peer`` -> ``tim``), exactly like a local run — the
readiness counters and curation scope pick it up untouched.

Auth: ``Authorization: Bearer <dashboard token>`` (what the agent runner
sends) or ``X-Sable-Token``. Query-string tokens are NOT accepted here —
inference bodies and URLs end up in logs; keep the secret in headers.

Target selection: the requested body's ``model`` names a running server
instance (matched against each instance state's model_name); no match falls
back to the default serve state. A request with NO model (healthchecks,
/v1/models listings) prefers the default state but also falls back to the
first live named instance — a peer may serve only named instances, and the
probe must see that as "serving". The engine's own attach key (if any) is
injected toward upstream; the caller's dashboard token never leaves this box.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("sable.dashboard.remote_gateway")


def _root(base_url: str) -> str:
    """Strip a trailing /vN from an OpenAI-style base URL (engine root)."""
    b = str(base_url).rstrip("/")
    if b.endswith("/v1"):
        b = b[:-3]
    return b.rstrip("/")


def _state_base(st: Optional[Dict[str, Any]]) -> Optional[Tuple[str, Optional[str]]]:
    """(base_url, attach_api_key) of a live instance state file entry."""
    from ..serving import instances as srv_inst

    if not st or not srv_inst.state_alive(st):
        return None
    base = st.get("base_url")
    if not base:
        return None
    return str(base), st.get("attach_api_key")


def resolve_upstream(app: Any, subpath: str,
                     model: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """Upstream URL + engine api_key for a /remote request.

    ``subpath`` is the part after the /v1 marker ("chat/completions",
    "models") or "tokenize" for the root form. Raises LookupError with a
    user-facing message when nothing serving matches.
    """
    from pathlib import Path

    from ..serving import instances as srv_inst
    from ..serving.deploy import read_serve_state

    home = Path(app.home)
    want = str(model or "").strip().lower()

    def _from_state(st: Optional[Dict[str, Any]]) -> Optional[Tuple[str, Optional[str]]]:
        got = _state_base(st)
        if got is None:
            return None
        base, key = got
        if subpath == "tokenize":
            return _root(base) + "/tokenize", key
        return base.rstrip("/") + "/" + subpath.lstrip("/"), key

    # Named-instance match by served model name (the runner sends the peer's
    # snapshot model_name as the request model).
    if want and want != "model":
        try:
            for inst in srv_inst.load_instances(home, cfg=app.cfg):
                iid = str(inst.get("id"))
                st = srv_inst.read_state(home, iid)
                if iid == srv_inst.DEFAULT_ID and not st:
                    st = read_serve_state(home)
                if st and str(st.get("model_name") or "").strip().lower() == want:
                    hit = _from_state(st)
                    if hit:
                        return hit
        except Exception:  # noqa: BLE001 — fall through to the default target
            logger.debug("remote gateway: instance scan failed", exc_info=True)

    # Default serve endpoint (legacy single-server state, else default row).
    # For requests WITHOUT a model (healthchecks, /v1/models listings) the
    # default endpoint is preferred, but a live *named* instance is a valid
    # serving target too — the peer may run its model only there. A request
    # WITH a model must still hit the default endpoint: a random named
    # instance may serve a different model than the one named.
    st = read_serve_state(home) or srv_inst.read_state(home, srv_inst.DEFAULT_ID)
    hit = _from_state(st)
    if hit:
        return hit
    if not want:
        try:
            for inst in srv_inst.load_instances(home, cfg=app.cfg):
                iid = str(inst.get("id"))
                st_i = (srv_inst.read_state(home, iid)
                        if iid != srv_inst.DEFAULT_ID
                        else st)
                hit = _from_state(st_i)
                if hit:
                    logger.info("remote gateway: no default endpoint serving, "
                                "falling back to named instance %s", iid)
                    return hit
        except Exception:  # noqa: BLE001 — honest 503 below, debug the why
            logger.debug("remote gateway: named-fallback scan failed",
                         exc_info=True)
    raise LookupError("no model is serving on this sable — start a server "
                      "instance first")


def import_trace(app: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Re-record a peer-pushed transcript into THIS box's trace buffer.

    Called after gateway auth for POST /remote/v1/traces/import. ``profile``
    is the LOCAL agent name the peer run belonged to (the pushing side maps
    ``tim@thishost`` back to ``tim``), so easy-training readiness counters and
    curation scope pick the row up exactly like a local run. Raises ValueError
    on a bad payload (-> 400 at the route)."""
    from ..store.recorder import TraceRecorder

    messages = payload.get("messages")
    task = str(payload.get("task") or "").strip()
    if not isinstance(messages, list) or not messages or not task:
        raise ValueError("messages (non-empty list) and task required")
    if len(messages) > 400:
        raise ValueError("too many messages (cap 400)")
    profile = str(payload.get("profile") or "").strip()[:80] or None
    rec = TraceRecorder(app.cfg, home=app.home, profile=profile)
    try:
        row = rec.record(
            task=task, messages=messages,
            backend=str(payload.get("backend") or "sable-remote")[:40],
            model_alias=str(payload.get("model_name") or "model")[:80],
            base_url=str(payload.get("base_url") or "peer")[:200],
            completed=bool(payload.get("completed")),
            tokens_in=payload.get("tokens_in"),
            tokens_out=payload.get("tokens_out"),
            duration_s=payload.get("duration_s"))
    finally:
        rec.close()
    if row is None:
        raise RuntimeError("trace recording failed")
    logger.info("remote gateway: imported peer trace %s (profile=%s)",
                row.trace_id, profile)
    return {"ok": True, "trace_id": row.trace_id}

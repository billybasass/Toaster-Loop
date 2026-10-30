"""Remote sable peers — remember and poll other sable dashboards on the
local network (Serve page "Connect remote sable").

Design:

* **Persistence**: one JSON file ``<SABLE_HOME>/peers.json`` holding a list
  of peer records ``{id, url, name, token, added_at, last_ok_at, snapshot}``.
  Atomic tmp+replace write, mode 0600 (records may carry the peer's dashboard
  token). Same file-per-state pattern the instances registry uses.
* **Direction**: this module only ever *reads* from the remote dashboard via
  its public GET endpoints (``/api/status``, ``/api/servers``, ``/api/profiles``,
  ``/api/jobs``, ``/api/traces``) with the peer's token in the standard
  ``X-Sable-Token`` header. Nothing is pushed, nothing is spawned remotely,
  and sable never sends credentials anywhere except the exact URL the user
  entered. This is the explicit, user-initiated exception to the loopback-only
  rule: the *listener* stays loopback; an outbound poll of a LAN machine the
  user chose to connect is the feature.
* **Failure honesty**: a failed probe never destroys the last good snapshot —
  it records ``snapshot.ok = false`` with the error string and keeps
  ``last_ok_at`` from the previous success. The UI shows "was healthy at …".
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

logger = logging.getLogger("sable.dashboard.peers")

PEERS_FILENAME = "peers.json"

#: Per-request timeout for every probe sub-call. A dead LAN host costs at most
#: this long (connect failures return immediately).
PROBE_TIMEOUT_S = 5.0

#: How many recent traces / jobs a snapshot carries.
SNAPSHOT_LIMIT = 6

_ID_RE = re.compile(r"^[a-f0-9]{12}$")
_lock = threading.Lock()


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------- url handling

def normalize_url(raw: str) -> str:
    """Validate + canonicalize a peer dashboard URL. Raises ValueError with a
    user-facing message on anything unusable. http(s) only; default scheme
    http; trailing slash stripped."""
    v = (raw or "").strip()
    if not v:
        raise ValueError("enter the remote dashboard URL "
                         "(e.g. http://192.168.1.42:8765)")
    if "://" not in v:
        v = "http://" + v
    try:
        p = urlparse(v)
    except ValueError as e:  # pragma: no cover - urlparse rarely raises
        raise ValueError(f"invalid URL: {e}") from None
    if p.scheme not in ("http", "https"):
        raise ValueError("URL must be http(s)")
    if not p.hostname:
        raise ValueError("URL must include a host")
    # Strip query/fragment; keep path (a peer behind a path-prefix proxy).
    base = f"{p.scheme}://{p.netloc}" + (p.path.rstrip("/") or "")
    return base


def peer_id(base_url: str) -> str:
    """Stable id for a peer: first 12 hex chars of its normalized URL hash."""
    return hashlib.sha256(base_url.encode("utf-8")).hexdigest()[:12]


# ------------------------------------------------------------- persistence

def peers_path(home: Path) -> Path:
    return Path(home) / PEERS_FILENAME


def load_peers(home: Path) -> List[Dict[str, Any]]:
    p = peers_path(home)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (OSError, json.JSONDecodeError):
        logger.warning("peers.json unreadable; starting from an empty list")
        return []
    if not isinstance(data, list):
        return []
    return [d for d in data if isinstance(d, dict) and _ID_RE.match(
        str(d.get("id", "")))]


def save_peers(home: Path, peers: List[Dict[str, Any]]) -> None:
    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(home), prefix=".peers-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(peers, f, ensure_ascii=False, indent=1)
        os.chmod(tmp, 0o600)  # may contain peer dashboard tokens
        os.replace(tmp, peers_path(home))
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def upsert_peer(home: Path, base_url: str, token: Optional[str],
                name: Optional[str] = None) -> Dict[str, Any]:
    """Add or update a peer record (url/token/name). Returns the record."""
    pid = peer_id(base_url)
    with _lock:
        peers = load_peers(home)
        rec = next((p for p in peers if p.get("id") == pid), None)
        if rec is None:
            rec = {"id": pid, "url": base_url, "token": token or None,
                   "name": name or _default_name(base_url),
                   "added_at": _utcnow(), "last_ok_at": None,
                   "snapshot": None}
            peers.append(rec)
        else:
            rec["url"] = base_url
            if token is not None and token != "":
                rec["token"] = token
            if name:
                rec["name"] = name
        save_peers(home, peers)
    return rec


def remove_peer(home: Path, pid: str) -> bool:
    with _lock:
        peers = load_peers(home)
        keep = [p for p in peers if p.get("id") != pid]
        if len(keep) == len(peers):
            return False
        save_peers(home, keep)
    return True


def set_snapshot(home: Path, pid: str, snapshot: Dict[str, Any]) -> None:
    """Store a probe result on the record (last_ok_at advances only on ok)."""
    with _lock:
        peers = load_peers(home)
        for p in peers:
            if p.get("id") != pid:
                continue
            p["snapshot"] = snapshot
            if snapshot.get("ok"):
                p["last_ok_at"] = snapshot.get("probed_at") or _utcnow()
        save_peers(home, peers)


def get_peer(home: Path, pid: str) -> Optional[Dict[str, Any]]:
    return next((p for p in load_peers(home) if p.get("id") == pid), None)


def _default_name(base_url: str) -> str:
    p = urlparse(base_url)
    return p.hostname or base_url


# --------------------------------------------------------------- probing

def _get_json(client, base: str, path: str, token: Optional[str],
              params: Optional[Dict[str, str]] = None) -> Any:
    headers = {"X-Sable-Token": token} if token else {}
    r = client.get(base.rstrip("/") + path, headers=headers, params=params,
                   timeout=PROBE_TIMEOUT_S)
    if r.status_code != 200:
        raise RuntimeError(f"{path} -> HTTP {r.status_code}")
    return r.json()


def probe(base_url: str, token: Optional[str]) -> Dict[str, Any]:
    """Poll one remote dashboard and return a normalized snapshot.

    Snapshot keys (ok=True): ok, probed_at, version, home, server, agents,
    servers, jobs, traces. On any transport/HTTP failure the snapshot is
    {ok: False, error, probed_at} — partial data is dropped rather than shown
    half-truths, and the caller keeps the previous good snapshot's timestamp.
    """
    import httpx

    probed_at = _utcnow()
    try:
        with httpx.Client(timeout=PROBE_TIMEOUT_S,
                          follow_redirects=True) as c:
            status = _get_json(c, base_url, "/api/status", token)
            servers_j = _get_json(c, base_url, "/api/servers", token)
            profiles_j = _get_json(c, base_url, "/api/profiles", token)
            jobs_j = _get_json(c, base_url, "/api/jobs",
                               token, params={"tail": str(SNAPSHOT_LIMIT)})
            traces_j = _get_json(c, base_url, "/api/traces",
                                 token, params={"limit": str(SNAPSHOT_LIMIT)})
    except Exception as e:  # httpx errors + JSON issues — one honest verdict
        return {"ok": False, "probed_at": probed_at,
                "error": f"{type(e).__name__}: {e}"}

    servers = (servers_j or {}).get("servers") or []
    jobs = (jobs_j or {}).get("jobs") or []
    traces = (traces_j or {}).get("traces") or []
    profiles = (profiles_j or {}).get("profiles") or []
    server_state = (status or {}).get("server") or {}
    return {
        "ok": True,
        "probed_at": probed_at,
        # identity
        "version": (status or {}).get("version"),
        "peer_home": (status or {}).get("home"),
        # serving: name/role/state/model per instance
        "servers": [
            {"name": s.get("name"), "role": s.get("role"),
             "running": bool(s.get("running")), "mode": s.get("mode"),
             "model_name": s.get("model_name")}
            for s in servers if isinstance(s, dict)],
        # agents: profiles (the remote's agent definitions) + what is busy.
        # The persona (system_prompt) travels so chat AND delegated children
        # run the peer's actual persona; capped at 8000 chars to keep
        # peers.json prompt-sized. Peer *tool grants* never travel.
        "agents": [
            {"name": p.get("name") or p.get("id"),
             "default": bool(p.get("default")),
             "system_prompt": str(p.get("system_prompt") or "")[:8000]}
            for p in profiles if isinstance(p, dict)],
        "default_profile": (profiles_j or {}).get("default_profile"),
        # training + activity: queue rows and the newest traces ride along so
        # "training/traces on" on the remote shows up here.
        "jobs": [
            {"job_id": j.get("job_id"), "kind": j.get("kind"),
             "state": j.get("state"), "model_id": j.get("model_id"),
             "created_at": j.get("created_at")}
            for j in jobs if isinstance(j, dict)][:SNAPSHOT_LIMIT],
        "traces": [
            {"trace_id": t.get("trace_id"), "profile": t.get("profile"),
             "status": t.get("status"), "created_at": t.get("created_at"),
             "model_name": t.get("model_name")}
            for t in traces if isinstance(t, dict)][:SNAPSHOT_LIMIT],
        # the remote default server summary (running bool + state blob)
        "server": {"running": bool(server_state.get("running"))},
    }


def remote_agent_entries(home: Path) -> List[Dict[str, Any]]:
    """Profile-shaped entries for agents on remembered, currently-healthy
    peers, named ``<agent>@<host>`` so two machines with same-named agents
    never collide. DISPLAY shape: never carries the peer token — chat
    resolution re-reads the record server-side via resolve_remote_agent."""
    out: List[Dict[str, Any]] = []
    seen = set()
    for p in load_peers(home):
        snap = p.get("snapshot") or {}
        if not snap.get("ok"):
            continue
        url = str(p.get("url") or "")
        host = urlparse(url).hostname or url
        # The model/instance a remote run lands on: first running server.
        model, instance = None, None
        for s in snap.get("servers") or []:
            if isinstance(s, dict) and s.get("running"):
                model = s.get("model_name")
                instance = s.get("name")
                break
        for a in snap.get("agents") or []:
            an = str((a or {}).get("name") or "").strip()
            full = f"{an}@{host}"
            if not an or full in seen:
                continue
            seen.add(full)
            out.append({"name": full, "system_prompt": "", "tools": [],
                        "model_name": model, "remote": True,
                        "peer_host": host})
    return out


def resolve_remote_agent(home: Path,
                         full_name: str) -> Optional[Dict[str, Any]]:
    """Server-side lookup of a ``<agent>@<host>`` name in peers.json.

    Returns a profile-shaped dict WITH the peer credentials (peer_url /
    peer_token / remote_instance) for h_chat_stream to point the runner at
    the peer dashboard's /remote/v1 passthrough, or None when the name is
    not a remote agent or its peer is unhealthy. Never returned to the UI:
    h_profiles_list uses remote_agent_entries (token-free) instead."""
    full = str(full_name or "").strip()
    agent, sep, host = full.rpartition("@")
    if not sep or not agent or not host:
        return None
    for p in load_peers(home):
        url = str(p.get("url") or "")
        u = urlparse(url)
        if (u.hostname or "") != host and (u.netloc or "") != host:
            continue
        snap = p.get("snapshot") or {}
        if not snap.get("ok"):
            continue
        match = next((a for a in (snap.get("agents") or [])
                      if str((a or {}).get("name") or "") == agent), None)
        if match is None:
            continue
        model, instance = None, None
        for s in snap.get("servers") or []:
            if isinstance(s, dict) and s.get("running"):
                model = s.get("model_name")
                instance = s.get("name")
                break
        # No `tools` key on purpose: an absent list means "all default tools"
        # downstream (agent_runner treats [] as drop-everything). The peer's
        # own tool grants stay on the peer — its inference answers whatever
        # this machine's agent loop asks, under this machine's guardrails.
        # The persona DOES travel (snapshot system_prompt): chat and delegated
        # children run the peer's actual persona.
        return {"name": full,
                "system_prompt": str(match.get("system_prompt") or ""),
                "model_name": model, "remote": True,
                "peer_url": url, "peer_token": p.get("token") or None,
                "remote_instance": instance}
    return None


def push_remote_trace(prof: Dict[str, Any], task: str, res: Any,
                      t0: float) -> None:
    """Send one finished remote-agent run back to the model's owner.

    POSTs the transcript to the peer dashboard's /remote/v1/traces/import
    (same bearer auth as inference). The profile name is mapped back to
    the peer's local agent name so their easy-training scoping sees it as
    one of its own traces. Best-effort by contract: any failure is logged
    and swallowed by the caller (chat stream, delegated children)."""
    import httpx

    purl = str(prof.get("peer_url") or "")
    if not purl:
        return
    full = str(prof.get("name") or "")
    local_name = full.rpartition("@")[0] or None
    body = {
        "task": task[:8000],
        "messages": [m for m in (getattr(res, "messages", None) or [])
                     if isinstance(m, dict)],
        "profile": local_name,
        "model_name": str(getattr(res, "model_alias", None)
                          or prof.get("model_name") or "model"),
        "base_url": str(getattr(res, "base_url", None) or ""),
        "completed": bool(getattr(res, "completed", False)),
        "tokens_in": getattr(res, "tokens_in", None),
        "tokens_out": getattr(res, "tokens_out", None),
        "duration_s": round(time.monotonic() - t0, 3),
    }
    headers = {"X-Sable-Token": prof["peer_token"]} \
        if prof.get("peer_token") else {}
    r = httpx.post(purl.rstrip("/") + "/remote/v1/traces/import",
                   json=body, headers=headers, timeout=15.0)
    if r.status_code >= 400:
        logger.info("remote trace push to %s failed: HTTP %s",
                    purl, r.status_code)


def peer_view(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Public shape for one peer record — token never leaves the backend."""
    snap = rec.get("snapshot") or None
    return {
        "id": rec.get("id"),
        "url": rec.get("url"),
        "name": rec.get("name"),
        "added_at": rec.get("added_at"),
        "last_ok_at": rec.get("last_ok_at"),
        "has_token": bool(rec.get("token")),
        "snapshot": snap,
        "healthy": bool(snap and snap.get("ok")),
    }


def stale_after_s() -> int:
    """UI hint: snapshots older than this render dimmed."""
    return 30

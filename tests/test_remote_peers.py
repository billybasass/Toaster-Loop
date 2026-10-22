"""Remote LAN sable peers (Serve page "Connect remote sable").

Hermetic: tmp SABLE_HOME via the conftest `home` fixture pattern, and the
network probe is monkeypatched — no sockets beyond localhost-free unit calls.
Covers URL normalization, record persistence, the probe/list/live/delete API,
token non-leakage, and failure honesty (failed probe never clobbers a good
snapshot's last_ok_at, failed first probe remembers nothing).
"""

from __future__ import annotations

import pytest

from sable.core.config import SableConfig
from sable.dashboard import peers as peers_mod
from sable.dashboard.app import DashboardApp


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("SABLE_HOME", str(h))
    return h


@pytest.fixture()
def app(home):
    return DashboardApp(SableConfig(), home=home)


def _ok_snapshot(url="http://peer:8765"):
    return {"ok": True, "probed_at": "2026-01-01T00:00:00+00:00",
            "version": "0.9.9", "peer_home": "/home/other/.sable",
            "servers": [{"name": "default", "role": "custom",
                         "running": True, "mode": "spawn",
                         "model_name": "remote-model"}],
            "agents": [{"name": "coder", "default": False}],
            "default_profile": None,
            "jobs": [{"job_id": "j1", "kind": "train", "state": "running",
                      "model_id": "m", "created_at": "t"}],
            "traces": [{"trace_id": "t1", "profile": "coder",
                        "status": "success", "created_at": "t",
                        "model_name": "remote-model"}],
            "server": {"running": True}}


def _fail_snapshot(err="ConnectError: boom"):
    return {"ok": False, "probed_at": "2026-01-02T00:00:00+00:00",
            "error": err}


# ------------------------------------------------------------ url handling

def test_normalize_url():
    assert peers_mod.normalize_url("192.168.1.5:8765") == \
        "http://192.168.1.5:8765"
    assert peers_mod.normalize_url("https://box.local:9/") == \
        "https://box.local:9"
    assert peers_mod.normalize_url("http://h/prefix/") == "http://h/prefix"
    with pytest.raises(ValueError):
        peers_mod.normalize_url("")
    with pytest.raises(ValueError):
        peers_mod.normalize_url("ftp://nope")
    with pytest.raises(ValueError):
        peers_mod.normalize_url("http:///noserver")


# ------------------------------------------------------------ persistence

def test_upsert_remove_roundtrip(home):
    assert peers_mod.load_peers(home) == []
    rec = peers_mod.upsert_peer(home, "http://a:1", "tokA", name="alpha")
    assert rec["id"] and rec["name"] == "alpha"
    # re-add same URL updates instead of duplicating; empty token keeps old
    rec2 = peers_mod.upsert_peer(home, "http://a:1", None)
    assert rec2["id"] == rec["id"]
    assert len(peers_mod.load_peers(home)) == 1
    assert peers_mod.get_peer(home, rec["id"])["token"] == "tokA"
    assert peers_mod.remove_peer(home, rec["id"]) is True
    assert peers_mod.remove_peer(home, rec["id"]) is False
    assert peers_mod.load_peers(home) == []


def test_peer_view_never_leaks_token(home):
    rec = peers_mod.upsert_peer(home, "http://a:1", "secret-token")
    view = peers_mod.peer_view(rec)
    assert view["has_token"] is True
    assert "secret-token" not in str(view)


# ------------------------------------------------------------------- API

def test_probe_remembers_then_lists(app, home, monkeypatch):
    calls = []

    def fake_probe(url, token):
        calls.append((url, token))
        return _ok_snapshot()

    monkeypatch.setattr(peers_mod, "probe", fake_probe)
    r = app.handle("POST", "/api/remote/probe",
                   body={"url": "http://peer:8765", "token": "tk"})
    assert r.status == 200, r.body
    assert calls == [("http://peer:8765", "tk")]
    pid = r.body["peer"]["id"]

    lst = app.handle("GET", "/api/remote/peers").body
    peers = {p["id"]: p for p in lst["peers"]}
    assert pid in peers
    p = peers[pid]
    assert p["healthy"] is True and p["last_ok_at"]
    assert p["snapshot"]["agents"][0]["name"] == "coder"
    assert p["snapshot"]["jobs"][0]["state"] == "running"
    assert p["snapshot"]["traces"][0]["trace_id"] == "t1"
    assert "tk" not in str(lst)  # token never round-trips to the UI


def test_failed_probe_remembers_nothing(app, monkeypatch):
    monkeypatch.setattr(peers_mod, "probe",
                        lambda url, token: _fail_snapshot())
    r = app.handle("POST", "/api/remote/probe", body={"url": "http://x:1"})
    assert r.status == 400
    assert r.body["error"]["code"] == "peer_unreachable"
    assert app.handle("GET", "/api/remote/peers").body["peers"] == []


def test_bad_url_is_clean_400(app):
    r = app.handle("POST", "/api/remote/probe", body={"url": "ftp://x"})
    assert r.status == 400 and r.body["error"]["code"] == "bad_request"


def test_live_reprobe_keeps_last_ok_on_failure(app, home, monkeypatch):
    monkeypatch.setattr(peers_mod, "probe",
                        lambda url, token: _ok_snapshot())
    pid = app.handle("POST", "/api/remote/probe",
                     body={"url": "http://peer:8765"}).body["peer"]["id"]
    assert peers_mod.get_peer(home, pid)["last_ok_at"]

    monkeypatch.setattr(peers_mod, "probe",
                        lambda url, token: _fail_snapshot("down"))
    r = app.handle("GET", f"/api/remote/peers/{pid}/live")
    assert r.status == 200
    p = r.body["peer"]
    assert p["healthy"] is False
    assert p["snapshot"]["error"] == "down"
    # last good timestamp survives the failed re-probe (failure honesty)
    assert p["last_ok_at"]

    assert app.handle("GET", "/api/remote/peers/nope123abcd/live").status == 404


def test_delete_peer(app, monkeypatch):
    monkeypatch.setattr(peers_mod, "probe",
                        lambda url, token: _ok_snapshot())
    pid = app.handle("POST", "/api/remote/probe",
                     body={"url": "http://peer:8765"}).body["peer"]["id"]
    assert app.handle("DELETE", f"/api/remote/peers/{pid}").status == 200
    assert app.handle("DELETE", f"/api/remote/peers/{pid}").status == 404


def test_probe_snapshot_shape(home, monkeypatch):
    """The real probe() against a fake transport: shape + auth header."""
    import httpx

    seen_headers = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.append(request.headers.get("X-Sable-Token"))
        path = request.url.path
        payloads = {
            "/api/status": {"version": "1.2.3", "home": "/h",
                            "server": {"running": True}},
            "/api/servers": {"servers": [
                {"name": "default", "running": True, "mode": "spawn",
                 "model_name": "m", "role": "custom"}]},
            "/api/profiles": {"profiles": [{"name": "a"}],
                              "default_profile": "a"},
            "/api/jobs": {"jobs": []},
            "/api/traces": {"traces": []},
        }
        if path in payloads:
            return httpx.Response(200, json=payloads[path])
        return httpx.Response(404, json={"error": "nope"})

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    def patched_client(**kw):
        kw["transport"] = transport
        return real_client(**kw)

    monkeypatch.setattr(httpx, "Client", patched_client)
    snap = peers_mod.probe("http://peer:8765", "tok123")
    assert snap["ok"] is True
    assert snap["version"] == "1.2.3"
    assert snap["servers"][0]["model_name"] == "m"
    assert snap["default_profile"] == "a"
    assert all(h == "tok123" for h in seen_headers) and seen_headers

    bad = peers_mod.probe("http://peer:8765", None)  # headers now None-tuple
    # MockTransport persists; second call without token still ok
    assert bad["ok"] is True

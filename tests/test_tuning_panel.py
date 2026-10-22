"""Tuning panel API tests — doctor --env, bench (WP1) and calibrate (WP3)
endpoints on the dashboard.

Hermetic: tmp SABLE_HOME (fixture), no GPU, no network. The doctor runs in
the background (a hung vendor probe must never hang the HTTP handler); these
tests poll its status endpoint with a generous deadline. The bench run test
monkeypatches the endpoint resolution + run_bench seam; the calibrate sweep
stays behind its own guard (no gguf_path => 400 before any spawn).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sable.core.config import SableConfig  # noqa: E402
from sable.dashboard.app import DashboardApp  # noqa: E402


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "uf-tuning"
    monkeypatch.setenv("SABLE_HOME", str(h))
    return h


@pytest.fixture()
def app(home):
    return DashboardApp(SableConfig(), home=home)


def err_code(r) -> str:
    return (r.body.get("error") or {}).get("code", "")


def wait_state(app, path: str, timeout: float = 30.0,
               ) -> Dict[str, Any]:
    """Poll a run-status endpoint until it leaves "running"."""
    s: Dict[str, Any] = {"state": "running"}
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = app.handle("GET", path).body
        if s.get("state") != "running":
            return s
        time.sleep(0.05)
    return s


# ------------------------------------------------------------------ routes

def test_tuning_routes_registered(app):
    paths = {(m, p) for (m, p) in ((m, p) for (m, p, _) in app.ROUTES)}
    for m, p in [
        ("GET", "/api/doctor/env"), ("POST", "/api/doctor/run"),
        ("POST", "/api/doctor/env"),
        ("GET", "/api/bench"), ("GET", "/api/bench/targets"),
        ("POST", "/api/bench/run"), ("GET", "/api/bench/run"),
        ("GET", "/api/calibrate"), ("POST", "/api/calibrate/run"),
        ("GET", "/api/calibrate/run"), ("POST", "/api/calibrate/clear"),
    ]:
        assert (m, p) in paths, f"{m} {p} missing"


# -------------------------------------------------------------------- doctor

def test_doctor_env_idle_without_running(app, home):
    """Bootstrap must NOT execute any checks (user clicks to activate)."""
    r = app.handle("GET", "/api/doctor/env")
    assert r.status == 200
    assert r.body["state"] == "idle"
    assert r.body.get("checks") in (None, [])
    assert r.body["calibration_mode"] in ("off", "auto")


def test_doctor_run_background_structured(app, home):
    r = app.handle("POST", "/api/doctor/run", body={})
    assert r.status == 200 and r.body["state"] == "running"
    s = wait_state(app, "/api/doctor/env")
    assert s["state"] == "done", s.get("error")
    assert s["exit_code"] in (0, 1, 3)
    names = {c["name"] for c in s["checks"]}
    # The WP7 check set must be visible to the panel.
    assert {"python", "deps", "config", "home", "disk"} <= names
    statuses = {c["status"] for c in s["checks"]}
    assert statuses <= {"ok", "warn", "problem"}


def test_doctor_run_fix_flag(app, home):
    r = app.handle("POST", "/api/doctor/env", body={"fix": True})
    assert r.status == 200 and r.body["state"] == "running"
    s = wait_state(app, "/api/doctor/env")
    assert s["state"] == "done", s.get("error")
    assert any(c["name"] == "state" for c in s["checks"])


def test_doctor_run_single_flight(app, home, monkeypatch):
    """A second click while measuring gets 409 already_running."""
    import sable.doctor as doctor_mod

    started = __import__("threading").Event()
    release = __import__("threading").Event()

    def slow_collect(*a, **k):
        started.set()
        release.wait(timeout=5)
        return [], 0

    monkeypatch.setattr(doctor_mod, "collect_checks", slow_collect)
    r = app.handle("POST", "/api/doctor/run", body={})
    assert r.status == 200
    assert started.wait(timeout=5)
    busy = app.handle("POST", "/api/doctor/run", body={})
    assert busy.status == 409 and err_code(busy) == "already_running"
    release.set()
    wait_state(app, "/api/doctor/env")


# --------------------------------------------------------------------- bench

def test_bench_list_empty(app, home):
    r = app.handle("GET", "/api/bench")
    assert r.status == 200
    assert r.body["records"] == []
    assert r.body["noise_floor"] is None


def test_bench_targets_lists_instances(app, home):
    """Multi-instance awareness: the panel learns which server instances are
    running (benchable endpoints) and whether a default spawn is possible."""
    from sable.serving import instances as srv_inst

    srv_inst.save_instance(home, {"id": "sable", "name": "Sable",
                                  "port": 8124})
    r = app.handle("GET", "/api/bench/targets")
    assert r.status == 200
    ids = {i["id"] for i in r.body["instances"]}
    assert {"default", "sable"} <= ids
    sable = next(i for i in r.body["instances"] if i["id"] == "sable")
    assert sable["running"] is False  # no state file => not running
    assert r.body["can_spawn_default"] == bool(app.cfg.gguf_path)


def test_bench_run_unknown_server_404(app, home):
    r = app.handle("POST", "/api/bench/run", body={"server_id": "ghost"})
    assert r.status == 404 and err_code(r) == "not_found"


def test_bench_run_stopped_server_409(app, home):
    from sable.serving import instances as srv_inst

    srv_inst.save_instance(home, {"id": "sable", "name": "Sable",
                                  "port": 8124})
    r = app.handle("POST", "/api/bench/run", body={"server_id": "sable"})
    assert r.status == 409 and err_code(r) == "not_running"


def _fake_record(**kw) -> Dict[str, Any]:
    return {
        "schema_version": 1, "kind": kw.get("kind", "bench"),
        "label": kw.get("label", ""), "model": "fake-model",
        "timestamp": "2026-01-01T00:00:00Z", "runs": 1,
        "prompt_processing": {"median": 10.0, "min": 10.0, "max": 10.0,
                              "spread_pct": 0.0},
        "generation": {"median": 20.0, "min": 20.0, "max": 20.0,
                       "spread_pct": 0.5},
        "token_identity": {"status": "ok", "sha256": "deadbeef"},
    }


def test_bench_list_shows_persisted_record(app, home, monkeypatch):
    from sable.serving import bench as bench_mod

    monkeypatch.setenv("SABLE_BENCH_DIR", str(home / "bench"))
    bench_mod.save_result(_fake_record(kind="noise_floor", label="base"))
    r = app.handle("GET", "/api/bench")
    assert r.status == 200
    assert len(r.body["records"]) == 1
    rec = r.body["records"][0]
    assert rec["kind"] == "noise_floor" and rec["label"] == "base"
    assert rec["gen_median_tps"] == 20.0
    # noise floor surfaced for the panel chip
    assert r.body["noise_floor"]["spread_pct"] == 0.5


def test_bench_run_background_success(app, home, monkeypatch):
    """POST /api/bench/run drives the WP1 modules through the same code path
    as the CLI; with a stubbed endpoint it completes and persists."""
    from sable.serving import bench as bench_mod

    monkeypatch.setenv("SABLE_BENCH_DIR", str(home / "bench"))

    def fake_resolve(cfg, spawn_if_needed=True):
        return "http://127.0.0.1:1/v1", "fake-model", None

    monkeypatch.setattr("sable.cli._resolve_endpoint", fake_resolve)
    monkeypatch.setattr(bench_mod, "run_bench",
                        lambda *a, **k: _fake_record(label="ui"))

    r = app.handle("POST", "/api/bench/run", body={"runs": 1, "label": "ui"})
    assert r.status == 200 and r.body["state"] == "running"

    s = wait_state(app, "/api/bench/run")
    assert s["state"] == "done", s.get("error")
    assert s["result"]["token_identity"]["status"] == "ok"
    # persisted -> visible through the list endpoint
    lst = app.handle("GET", "/api/bench").body
    assert len(lst["records"]) == 1


def test_bench_run_error_state(app, home, monkeypatch):
    """Endpoint resolution failure lands as state=error (not a 500)."""
    from sable.serving.supervisor import ServeError

    def boom(cfg, spawn_if_needed=True):
        raise ServeError("no endpoint available")

    monkeypatch.setattr("sable.cli._resolve_endpoint", boom)
    r = app.handle("POST", "/api/bench/run", body={"runs": 1})
    assert r.status == 200
    s = wait_state(app, "/api/bench/run")
    assert s["state"] == "error" and "no endpoint" in s["error"]


def test_bench_run_rejects_bad_runs(app):
    r = app.handle("POST", "/api/bench/run", body={"runs": "many"})
    assert r.status == 400


# ----------------------------------------------------------------- calibrate

def test_calibrate_show_empty(app, home):
    r = app.handle("GET", "/api/calibrate")
    assert r.status == 200
    assert r.body["profiles"] == []
    assert r.body["calibration_mode"] == "off"
    assert r.body["running_instances"] == []


def test_calibrate_run_without_model_400(app, home):
    r = app.handle("POST", "/api/calibrate/run", body={"quick": True})
    assert r.status == 400 and err_code(r) == "no_model"


def test_calibrate_clear_empty(app, home):
    r = app.handle("POST", "/api/calibrate/clear", body={})
    assert r.status == 200 and r.body["removed"] == 0


def test_calibrate_run_rejects_bad_runs(app, home):
    # runs is validated before the model resolution, so no gguf needed.
    r = app.handle("POST", "/api/calibrate/run", body={"runs": "lots"})
    assert r.status == 400


# ------------------------------------------------- serve.calibration toggle

def test_config_put_serve_section(app, home):
    """The Tuning panel toggles serve.calibration through a section patch."""
    r = app.handle("PUT", "/api/config", body={
        "section": "serve", "patch": {"calibration": "auto"}})
    assert r.status == 200, r.body
    raw = (home / "sable.yaml").read_text()
    assert "calibration: auto" in raw

    bad = app.handle("PUT", "/api/config", body={
        "section": "serve", "patch": {"nonsense_key": 1}})
    assert bad.status == 400 and err_code(bad) == "unknown_keys"

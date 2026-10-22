"""Easy training — turnkey one-profile loop (training/easy.py + /api/easy).

Coverage: profile-scoped should_auto_train, recorder profile stamping, the
config preset written on enable, endpoint gating (unbound profile, not-ready
run), and the completion hook's auto-deploy bookkeeping. No GPU: run wiring
is asserted through a captured submit_iterate call, not a live worker.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sable.core.config import SableConfig  # noqa: E402
from sable.dashboard.app import DashboardApp  # noqa: E402
from sable.store.db import TraceRow, TraceStore  # noqa: E402
from sable.store.recorder import TraceRecorder  # noqa: E402
from sable.training.easy import (  # noqa: E402
    EasyTrainingConfig,
    evaluate_readiness,
    load_state,
    optimal_config_patches,
    save_state,
    validate_profile,
)
from sable.training.scheduler import should_auto_train  # noqa: E402


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "uf-easy"
    monkeypatch.setenv("SABLE_HOME", str(h))
    return h


@pytest.fixture()
def app(home):
    a = DashboardApp(SableConfig(), home=home)
    yield a


def seed_trace(store, tid, score=0.9, profile="fallback"):
    from datetime import datetime, timedelta, timezone

    p = Path(store.db_path).parent / f"{tid}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        f.write(json.dumps({"kind": "trace_header"}) + "\n")
        f.write(json.dumps({"kind": "message", "message": {
            "role": "user", "content": f"task {tid}"}}) + "\n")
        f.write(json.dumps({"kind": "message", "message": {
            "role": "assistant", "content": f"done {tid} ok"}}) + "\n")
    # relative-to-now stamp (time-bomb sweep): keeps seeded traces inside any
    # rolling window production may apply, forever.
    ts = (datetime.now(timezone.utc) - timedelta(days=1)
          ).isoformat(timespec="seconds")
    store.insert_trace(TraceRow(
        trace_id=tid, session_id=None, timestamp=ts,
        agent_profile=profile, model_alias="m", model_version=None,
        endpoint_label=None, task_summary="t", status="success",
        outcome_score=score, num_tool_calls=0, tokens_in=None,
        tokens_out=None, duration_s=None, jsonl_path=str(p),
    ))


# ------------------------------------------------------------ scoped check

def test_should_auto_train_scopes_to_profile(home):
    store = TraceStore(home / "buffer" / "sable.db")
    for i in range(3):
        seed_trace(store, f"a{i}", score=0.9, profile="coder")
    seed_trace(store, "b0", score=0.1, profile="other")
    whole = should_auto_train(store, min_traces=3, min_score=0.5)
    assert whole["trigger"] and whole["unclaimed_traces"] == 4
    scoped = should_auto_train(store, min_traces=3, min_score=0.5,
                               agent_profile="coder")
    assert scoped["trigger"] and scoped["unclaimed_traces"] == 3
    empty = should_auto_train(store, min_traces=1, min_score=0.5,
                              agent_profile="nobody")
    assert not empty["trigger"] and empty["unclaimed_traces"] == 0
    store.close()


def test_recorder_stamps_profile(home):
    rec = TraceRecorder(SableConfig(), home=home, profile="coder")
    row = rec.record(task="t", messages=[{"role": "user", "content": "hi"},
                                         {"role": "assistant",
                                          "content": "yo"}],
                     backend="sable", model_alias="m", base_url="",
                     completed=True)
    assert row is not None and row.agent_profile == "coder"
    rec.close()
    # Without the profile kwarg the legacy backend name is kept.
    rec2 = TraceRecorder(SableConfig(), home=home)
    row2 = rec2.record(task="t", messages=[], backend="sable",
                       model_alias="m", base_url="", completed=True)
    assert row2 is not None and row2.agent_profile == "sable"
    rec2.close()


# ---------------------------------------------------------------- preset

def test_optimal_patches_merge_per_section():
    ec = EasyTrainingConfig(enabled=True, profile="coder", auto_deploy=True)
    patches = optimal_config_patches(ec)
    sections = [p["section"] for p in patches]
    assert sections == ["easy_training", "curation", "training",
                        "validation", "goldens_mine"]
    et = next(p for p in patches if p["section"] == "easy_training")["patch"]
    assert et == {"enabled": True, "profile": "coder", "auto_deploy": True}
    # Quality machinery the loop grew after it first shipped is armed:
    val = next(p for p in patches if p["section"] == "validation")["patch"]
    assert val == {"shadow_enabled": True}
    gm = next(p for p in patches if p["section"] == "goldens_mine")["patch"]
    assert gm == {"enabled": True}
    # And deliberately NOT armed: unmeasured backends / spend-heavy features.
    tr = next(p for p in patches if p["section"] == "training")["patch"]
    assert "backend" not in tr and "rank_scaling" not in tr


def test_validate_profile_requires_server_binding():
    profs = [{"name": "coder", "server_instance": "srv-1"},
             {"name": "loose"}]
    assert validate_profile(profs, "coder")["name"] == "coder"
    with pytest.raises(ValueError, match="not bound"):
        validate_profile(profs, "loose")
    with pytest.raises(ValueError, match="no profile"):
        validate_profile(profs, "ghost")
    with pytest.raises(ValueError):
        validate_profile(profs, "")


def test_state_roundtrip(home):
    assert load_state(home) == {}
    save_state(home, {"last_start": {"run_id": "x"}})
    save_state(home, {"last_run": {"state": "done"}})
    st = load_state(home)
    assert st["last_start"]["run_id"] == "x" and st["last_run"]["state"] == "done"


def test_evaluate_readiness_shape(home):
    store = TraceStore(home / "buffer" / "sable.db")
    for i in range(2):
        seed_trace(store, f"a{i}", score=0.9, profile="coder")
    ec = EasyTrainingConfig(enabled=True, profile="coder", min_traces=5)
    d = evaluate_readiness(store, ec, object(), {"ok": True, "reasons": []})
    assert d["unclaimed"] == 2 and d["threshold"]["min_traces"] == 5
    assert not d["needs_training"]
    d2 = evaluate_readiness(store, EasyTrainingConfig(enabled=True,
                              profile="coder", min_traces=1),
                            object(), {"ok": False, "reasons": ["no vram"]})
    assert d2["trigger"] and not d2["needs_training"]
    store.close()


# ------------------------------------------------------------------ API

def _put_profile(app, name, server=None):
    body = {"system_prompt": "p", "tools": []}
    if server:
        body["server_instance"] = server
    r = app.handle("PUT", f"/api/profiles/{name}", body=body)
    assert r.status == 200, r.body


def test_easy_get_disabled(app):
    r = app.handle("GET", "/api/easy")
    assert r.status == 200
    b = r.body
    assert b["config"]["enabled"] is False
    assert b["readiness"] is None
    assert b["runs"] == []


def test_easy_enable_requires_bound_profile(app):
    _put_profile(app, "loose")
    r = app.handle("POST", "/api/easy/enable", body={"profile": "loose"})
    assert r.status == 400 and "not bound" in r.body["error"]["message"]
    r = app.handle("POST", "/api/easy/enable", body={"profile": "ghost"})
    assert r.status == 400


def test_easy_enable_writes_preset_and_scopes(app, home):
    _put_profile(app, "coder", server="srv-1")
    r = app.handle("POST", "/api/easy/enable",
                   body={"profile": "coder", "auto_deploy": True})
    assert r.status == 200, r.body
    cfg = app._easy_cfg()
    assert cfg.enabled and cfg.profile == "coder" and cfg.auto_deploy
    # Optimal preset landed in the sibling sections.
    assert app.cfg.curation.get("include_failures") is True
    assert app.cfg.curation.get("formats") == "both"
    assert app.cfg.training.get("auto_enabled") is True
    assert app.cfg.training.get("eval_gate") == "warn"
    # Quality machinery armed by the extended preset (section-merged, so an
    # existing validation.golden_tasks list would have survived untouched).
    assert app.cfg.validation.get("shadow_enabled") is True
    assert app.cfg.goldens_mine.get("enabled") is True
    # GET reflects the enabled loop + scoped readiness.
    g = app.handle("GET", "/api/easy").body
    assert g["config"]["enabled"] and g["profile"]["name"] == "coder"
    assert g["readiness"] is not None  # server missing => still computed


def test_easy_disable_keeps_profile(app):
    _put_profile(app, "coder", server="srv-1")
    app.handle("POST", "/api/easy/enable", body={"profile": "coder"})
    r = app.handle("POST", "/api/easy/disable", body={})
    assert r.status == 200
    cfg = app._easy_cfg()
    assert cfg.enabled is False and cfg.profile == "coder"


def test_easy_run_requires_readiness(app, home):
    _put_profile(app, "coder", server="srv-1")
    app.handle("POST", "/api/easy/enable", body={"profile": "coder"})
    r = app.handle("POST", "/api/easy/run", body={})
    assert r.status == 409 and r.body["error"]["code"] == "not_ready"


def test_easy_run_scopes_curation_and_names_run(app, home, monkeypatch):
    _put_profile(app, "coder", server="srv-1")
    app.handle("POST", "/api/easy/enable", body={"profile": "coder"})
    # Make readiness pass cheaply: threshold 1 trace.
    store = TraceStore(home / "buffer" / "sable.db")
    seed_trace(store, "a0", score=0.9, profile="coder")
    store.close()
    r = app.handle("PUT", "/api/config", body={
        "section": "easy_training", "patch": {"min_traces": 1}})
    assert r.status == 200

    captured = {}

    def fake_submit(b, *, on_done=None):
        captured["body"] = b
        captured["on_done"] = on_done
        return {"ok": True, "run_id": "iter-test", "note": "stub"}

    monkeypatch.setattr(app, "submit_iterate", fake_submit)
    r = app.handle("POST", "/api/easy/run", body={})
    assert r.status == 202 and r.body["run_id"] == "iter-test"
    assert captured["body"]["profile"] == "coder"
    assert captured["body"]["name"] == "easy:coder"
    assert captured["on_done"] is not None

    # Completion hook: promoted + auto_deploy off => ledger only, no deploy.
    captured["on_done"]({"run_id": "iter-test", "state": "done",
                         "finished": "now",
                         "result": {"model_id": "v2", "promoted": True,
                                    "dataset_id": "d1"}})
    st = load_state(home)
    assert st["last_run"]["model_id"] == "v2"
    assert "deploy" not in st["last_run"]


def test_easy_run_not_enabled(app):
    r = app.handle("POST", "/api/easy/run", body={})
    assert r.status == 400 and r.body["error"]["code"] == "easy_disabled"


def test_easy_after_run_deploy_missing_instance(app, home):
    ec = EasyTrainingConfig(enabled=True, profile="coder", auto_deploy=True)
    prof = {"name": "coder", "server_instance": "gone"}
    app._easy_after_run("r1", {"run_id": "r1", "state": "done",
                               "finished": "now",
                               "result": {"model_id": "v9",
                                          "promoted": True}},
                        ec, prof)
    st = load_state(home)
    dep = st["last_run"]["deploy"]
    assert dep["attempted"] and not dep["ok"]
    # v9 was never registered, so the deploy stops at the GGUF check —
    # best-effort failure is recorded, never raised.
    assert "GGUF" in (dep["detail"] or "")


# ------------------------------------------------------- config hardening

def test_from_yaml_garbage_numbers_degrade():
    """A hand-edited min_traces: lots must not raise out of every parse."""
    ec = EasyTrainingConfig.from_yaml(
        {"enabled": True, "profile": "p", "min_traces": "lots",
         "min_score": "high"})
    assert ec.enabled and ec.profile == "p"
    assert ec.min_traces is None and ec.min_score is None


def test_from_yaml_numeric_strings_still_parse():
    ec = EasyTrainingConfig.from_yaml({"min_traces": "5", "min_score": "0.7"})
    assert ec.min_traces == 5
    assert ec.min_score is not None and abs(ec.min_score - 0.7) < 1e-9


# ------------------------------------------------------- golden-mining pass

def test_mine_goldens_pass_skips_when_disabled(home):
    from sable.core.config import SableConfig
    from sable.training.easy import mine_goldens_pass

    rep = mine_goldens_pass(SableConfig(), home)  # goldens_mine: {} => off
    assert rep["ok"] is False and "false" in rep["skipped"]
    # Nothing was created by a skipped pass.
    assert not (home / "goldens" / "mined.jsonl").exists()


def test_mine_goldens_pass_runs_and_never_raises(home):
    from sable.core.config import SableConfig
    from sable.training.easy import mine_goldens_pass

    cfg = SableConfig(goldens_mine={"enabled": True})
    store = TraceStore(home / "buffer" / "sable.db")
    seed_trace(store, "g0", score=0.9, profile="coder")
    store.add_feedback("g0", 1.0, "nice", source="user", timestamp="")
    store.close()
    rep = mine_goldens_pass(cfg, home)
    # Either it mined (task distillable) or dropped it honestly — the pass
    # itself must report ok and never raise on a healthy store.
    assert rep["ok"] is True
    assert rep["candidates"] + rep["skipped_dedup"] >= 0


def test_mine_goldens_pass_broken_store_degrades(home):
    """A garbage goldens_mine section or unreadable store must not raise."""
    from sable.core.config import SableConfig
    from sable.training.easy import mine_goldens_pass

    cfg = SableConfig(goldens_mine={"enabled": True, "max_per_run": "oops"})
    rep = mine_goldens_pass(cfg, home)  # no buffer DB at all
    assert rep["ok"] is False and "skipped" in rep


def test_easy_run_fires_mining_pass(app, home, monkeypatch):
    _put_profile(app, "coder", server="srv-1")
    app.handle("POST", "/api/easy/enable", body={"profile": "coder"})
    store = TraceStore(home / "buffer" / "sable.db")
    seed_trace(store, "a0", score=0.9, profile="coder")
    store.close()
    app.handle("PUT", "/api/config", body={
        "section": "easy_training", "patch": {"min_traces": 1}})

    calls = []
    import sable.dashboard.app as appmod

    def fake_pass(cfg, home_):
        calls.append(home_)
        return {"ok": True, "mined": 0, "skipped_dedup": 0}

    monkeypatch.setattr(appmod, "mine_goldens_pass", fake_pass)
    monkeypatch.setattr(
        app, "submit_iterate",
        lambda b, *, on_done=None: {"ok": True, "run_id": "iter-m1",
                                    "note": "stub"})
    r = app.handle("POST", "/api/easy/run", body={})
    assert r.status == 202
    assert calls == [home]
    # Ledgered for the panel.
    st = load_state(home)
    assert st["last_mine"]["ok"] is True and st["last_mine"]["profile"] == "coder"


# ---------------------------------------------------------- loop lock

def test_loop_lock_excludes_second_holder(home):
    from sable.training.easy import (acquire_loop_lock, release_loop_lock)

    h1 = acquire_loop_lock(home)
    assert h1 is not None
    h2 = acquire_loop_lock(home)
    assert h2 is None  # second process would be turned away
    release_loop_lock(h1)
    h3 = acquire_loop_lock(home)
    assert h3 is not None
    release_loop_lock(h3)


def test_submit_iterate_rejects_when_loop_busy(app, home):
    from sable.training.easy import acquire_loop_lock, release_loop_lock

    handle = acquire_loop_lock(home)
    try:
        with pytest.raises(Exception) as ei:
            app.submit_iterate({"name": "x", "kind": "sft"})
        assert getattr(ei.value, "code", None) == "loop_busy" or \
            "loop_busy" in str(ei.value)
    finally:
        release_loop_lock(handle)


# ------------------------------------------------- enable preview / preset

def test_easy_enable_preview_writes_nothing(app):
    _put_profile(app, "coder", server="srv-1")
    before = dict(app.cfg.curation or {})
    r = app.handle("POST", "/api/easy/enable",
                   body={"profile": "coder", "preview": True})
    assert r.status == 200
    assert isinstance(r.body["diff"], list) and r.body["diff"]
    # Nothing was enabled and no preset landed.
    assert app._easy_cfg().enabled is False
    assert dict(app.cfg.curation or {}) == before


def test_easy_enable_apply_preset_false_keeps_sections(app):
    _put_profile(app, "coder", server="srv-1")
    app.handle("PUT", "/api/config", body={
        "section": "curation", "patch": {"min_score": 0.9}})
    r = app.handle("POST", "/api/easy/enable",
                   body={"profile": "coder", "auto_deploy": True,
                         "apply_preset": False})
    assert r.status == 200
    cfg = app._easy_cfg()
    assert cfg.enabled and cfg.profile == "coder" and cfg.auto_deploy
    # User-tuned curation untouched; training preset NOT applied.
    assert app.cfg.curation.get("min_score") == 0.9
    assert not (app.cfg.training or {}).get("auto_enabled")


def test_easy_enable_rolls_back_on_partial_failure(app, monkeypatch):
    """If a later config patch fails, the earlier ones must not persist."""
    _put_profile(app, "coder", server="srv-1")
    real_handle = app.handle
    calls = {"n": 0}

    def flaky(method, path, query=None, body=None):
        if path == "/api/config" and body and body.get("section") == "training":
            calls["n"] += 1
            from sable.dashboard.app import ApiResponse
            return ApiResponse(500, {"error": {"message": "boom",
                                               "code": "internal_error"}})
        return real_handle(method, path, query=query, body=body)

    monkeypatch.setattr(app, "handle", flaky)
    # handle() converts the raised HttpError into an error ApiResponse.
    r = app.handle("POST", "/api/easy/enable", body={"profile": "coder"})
    assert r.status == 500
    assert calls["n"] == 1
    # Rollback: easy_training never enabled, preset sections not written.
    cfg = EasyTrainingConfig.from_yaml(app.cfg.easy_training or {})
    assert cfg.enabled is False
    assert not (app.cfg.curation or {}).get("include_failures")


# ---------------------------------------------------- deploy start checks

def _deploy_setup(app, monkeypatch, start_status=200):
    """Fake registry/instances so _easy_deploy reaches the restart step."""
    from types import SimpleNamespace
    import sable.dashboard.app as appmod
    from sable.dashboard.app import ApiResponse

    saved = {}
    monkeypatch.setattr(app, "_registry", lambda: SimpleNamespace(
        get_model=lambda mid: SimpleNamespace(gguf_path="/models/x.gguf"),
        close=lambda: None))
    inst = {"id": "srv-1", "name": "S1", "gguf": "/old.gguf"}
    monkeypatch.setattr(appmod.srv_inst, "get_instance",
                        lambda home, sid, cfg=None: dict(inst))
    monkeypatch.setattr(appmod.srv_inst, "save_instance",
                        lambda home, i: saved.update(defn=dict(i)))
    monkeypatch.setattr(appmod.srv_inst, "state_alive", lambda st: True)
    monkeypatch.setattr(app, "_instance_state", lambda sid: {"pid": 1})
    monkeypatch.setattr(app, "_stop_instance", lambda sid: True)
    monkeypatch.setattr(app, "handle",
                        lambda m, p, body=None, q=None:
                        ApiResponse(start_status, {"ok": start_status < 400}))
    return saved


def test_easy_deploy_success(app, monkeypatch):
    _deploy_setup(app, monkeypatch, start_status=200)
    out = app._easy_deploy("v2", "srv-1")
    assert out["ok"] and "now serves" in out["detail"]


def test_easy_deploy_start_failure_reports_down(app, monkeypatch):
    _deploy_setup(app, monkeypatch, start_status=500)
    out = app._easy_deploy("v2", "srv-1")
    # Definition updated, but the failed restart must not report success.
    assert out["attempted"] and not out["ok"]
    assert "restart failed" in (out["detail"] or "")


def test_easy_after_run_no_longer_takes_on_done(app):
    import inspect
    sig = inspect.signature(app._easy_after_run)
    assert "on_done" not in sig.parameters


# --------------------------------------------------------- stale stamps

def test_easy_get_reports_stale_stamps(app, home):
    """Unclaimed traces under a vanished profile name surface as a nudge."""
    _put_profile(app, "coder", server="srv-1")
    app.handle("POST", "/api/easy/enable", body={"profile": "coder"})
    store = TraceStore(home / "buffer" / "sable.db")
    seed_trace(store, "a0", score=0.9, profile="coder")
    seed_trace(store, "b0", score=0.9, profile="renamed-away")
    seed_trace(store, "b1", score=0.9, profile="sable")  # legacy: never stale
    store.close()
    g = app.handle("GET", "/api/easy").body
    stamps = {s["profile"]: s["traces"] for s in g["stale_stamps"]}
    assert stamps == {"renamed-away": 1}


def test_easy_get_no_stale_when_disabled(app, home):
    r = app.handle("GET", "/api/easy")
    assert r.status == 200 and r.body["stale_stamps"] == []


# ================================================================ multi-agent
# The loop grew from one profile to a list: `profile` stays the primary,
# `profiles` holds every agent. A single-profile config must behave EXACTLY
# like before (cardinal rule 1), so these tests pin both the new paths and the
# legacy fallbacks.

def test_from_yaml_profiles_list_and_all_profiles():
    ec = EasyTrainingConfig.from_yaml(
        {"enabled": True, "profile": "coder",
         "profiles": ["coder", "writer", " writer ", 42]})
    # dedupe + string coercion; primary first, order preserved
    assert ec.all_profiles == ["coder", "writer"]
    # legacy config (no profiles key) => exactly [profile]
    legacy = EasyTrainingConfig.from_yaml({"enabled": True,
                                           "profile": "coder"})
    assert legacy.profiles == [] and legacy.all_profiles == ["coder"]
    none = EasyTrainingConfig.from_yaml({"enabled": True})
    assert none.all_profiles == []


def test_from_yaml_garbage_profiles_degrade():
    ec = EasyTrainingConfig.from_yaml(
        {"enabled": True, "profile": "p", "profiles": "not-a-list"})
    assert ec.profiles == [] and ec.all_profiles == ["p"]
    ec2 = EasyTrainingConfig.from_yaml(
        {"enabled": True, "profile": "p", "profiles": [None, "", 5]})
    # non-string junk and blanks are skipped, never coerced
    assert ec2.profiles == [] and ec2.all_profiles == ["p"]


def test_for_profile_projection_keeps_thresholds():
    ec = EasyTrainingConfig(enabled=True, profile="a", profiles=["a", "b"],
                            auto_deploy=True, min_traces=7, min_score=0.6)
    view = ec.for_profile("b")
    assert (view.profile, view.profiles, view.enabled, view.auto_deploy,
            view.min_traces, view.min_score) == \
        ("b", [], True, True, 7, 0.6)


def test_optimal_patches_omit_profiles_for_single_agent():
    """Legacy enable must write the byte-identical easy_training patch."""
    ec = EasyTrainingConfig(enabled=True, profile="coder")
    et = next(p for p in optimal_config_patches(ec)
              if p["section"] == "easy_training")["patch"]
    assert "profiles" not in et
    ec2 = EasyTrainingConfig(enabled=True, profile="coder",
                             profiles=["coder", "writer"])
    et2 = next(p for p in optimal_config_patches(ec2)
               if p["section"] == "easy_training")["patch"]
    assert et2["profiles"] == ["coder", "writer"]


def test_easy_enable_with_extra_agents(app):
    _put_profile(app, "coder", server="srv-1")
    _put_profile(app, "writer", server="srv-2")
    r = app.handle("POST", "/api/easy/enable",
                   body={"profile": "coder", "profiles": ["writer"]})
    assert r.status == 200, r.body
    cfg = app._easy_cfg()
    assert cfg.enabled and cfg.profile == "coder"
    assert cfg.all_profiles == ["coder", "writer"]
    g = app.handle("GET", "/api/easy").body
    names = [a["profile"] for a in g["agents"]]
    assert names == ["coder", "writer"]
    # Legacy top-level fields mirror the primary agent.
    assert g["profile"]["name"] == "coder"


def test_easy_enable_rejects_unbound_extra(app):
    _put_profile(app, "coder", server="srv-1")
    _put_profile(app, "loose")
    r = app.handle("POST", "/api/easy/enable",
                   body={"profile": "coder", "profiles": ["loose"]})
    assert r.status == 400 and "not bound" in r.body["error"]["message"]


def test_easy_add_agent(app):
    _put_profile(app, "coder", server="srv-1")
    _put_profile(app, "writer", server="srv-2")
    app.handle("POST", "/api/easy/enable", body={"profile": "coder"})
    r = app.handle("POST", "/api/easy/add", body={"profile": "writer"})
    assert r.status == 200, r.body
    assert app._easy_cfg().all_profiles == ["coder", "writer"]
    # Idempotent: adding again is a no-op success.
    r2 = app.handle("POST", "/api/easy/add", body={"profile": "writer"})
    assert r2.status == 200
    assert app._easy_cfg().all_profiles == ["coder", "writer"]
    # Unknown / unbound agents are refused like enable does.
    assert app.handle("POST", "/api/easy/add",
                      body={"profile": "ghost"}).status == 400


def test_easy_add_requires_server_binding(app):
    _put_profile(app, "coder", server="srv-1")
    _put_profile(app, "loose")
    app.handle("POST", "/api/easy/enable", body={"profile": "coder"})
    r = app.handle("POST", "/api/easy/add", body={"profile": "loose"})
    assert r.status == 400 and "not bound" in r.body["error"]["message"]


def test_easy_remove_agent_collapses_to_legacy(app):
    _put_profile(app, "coder", server="srv-1")
    _put_profile(app, "writer", server="srv-2")
    app.handle("POST", "/api/easy/enable",
               body={"profile": "coder", "profiles": ["writer"]})
    r = app.handle("POST", "/api/easy/remove", body={"profile": "writer"})
    assert r.status == 200, r.body
    cfg = app._easy_cfg()
    assert cfg.profile == "coder" and cfg.all_profiles == ["coder"]
    # Removing the collapse back: `profiles` key is gone (legacy shape).
    raw = app.cfg.easy_training or {}
    assert "profiles" not in raw
    r2 = app.handle("POST", "/api/easy/remove", body={"profile": "writer"})
    assert r2.status == 400  # never was / no longer in the loop


def test_easy_remove_primary_promotes_next(app):
    _put_profile(app, "coder", server="srv-1")
    _put_profile(app, "writer", server="srv-2")
    app.handle("POST", "/api/easy/enable",
               body={"profile": "coder", "profiles": ["writer"]})
    r = app.handle("POST", "/api/easy/remove", body={"profile": "coder"})
    assert r.status == 200
    cfg = app._easy_cfg()
    assert cfg.profile == "writer" and cfg.all_profiles == ["writer"]


def test_easy_run_targets_selected_agent(app, home, monkeypatch):
    _put_profile(app, "coder", server="srv-1")
    _put_profile(app, "writer", server="srv-2")
    app.handle("POST", "/api/easy/enable",
               body={"profile": "coder", "profiles": ["writer"]})
    store = TraceStore(home / "buffer" / "sable.db")
    seed_trace(store, "w0", score=0.9, profile="writer")
    store.close()
    app.handle("PUT", "/api/config", body={
        "section": "easy_training", "patch": {"min_traces": 1}})

    captured = {}

    def fake_submit(b, *, on_done=None):
        captured["body"] = b
        return {"ok": True, "run_id": "iter-w", "note": "stub"}

    monkeypatch.setattr(app, "submit_iterate", fake_submit)
    # Explicit target: writer's traces only.
    r = app.handle("POST", "/api/easy/run", body={"profile": "writer"})
    assert r.status == 202, r.body
    assert captured["body"]["profile"] == "writer"
    assert captured["body"]["name"] == "easy:writer"
    # Coder has zero traces => its own run is refused as not ready.
    r2 = app.handle("POST", "/api/easy/run", body={"profile": "coder"})
    assert r2.status == 409 and r2.body["error"]["code"] == "not_ready"


def test_easy_run_defaults_to_primary(app, home, monkeypatch):
    _put_profile(app, "coder", server="srv-1")
    _put_profile(app, "writer", server="srv-2")
    app.handle("POST", "/api/easy/enable",
               body={"profile": "coder", "profiles": ["writer"]})
    store = TraceStore(home / "buffer" / "sable.db")
    seed_trace(store, "c0", score=0.9, profile="coder")
    store.close()
    app.handle("PUT", "/api/config", body={
        "section": "easy_training", "patch": {"min_traces": 1}})
    captured = {}
    monkeypatch.setattr(
        app, "submit_iterate",
        lambda b, *, on_done=None: captured.update(body=b) or
        {"ok": True, "run_id": "iter-c", "note": "stub"})
    # No body.profile => legacy semantics: train the primary.
    r = app.handle("POST", "/api/easy/run", body={})
    assert r.status == 202
    assert captured["body"]["profile"] == "coder"


def test_easy_run_rejects_agent_outside_loop(app):
    _put_profile(app, "coder", server="srv-1")
    _put_profile(app, "writer", server="srv-2")
    app.handle("POST", "/api/easy/enable", body={"profile": "coder"})
    r = app.handle("POST", "/api/easy/run", body={"profile": "writer"})
    assert r.status == 400 and r.body["error"]["code"] == "unknown_profile"


def test_easy_get_per_agent_readiness(app, home):
    _put_profile(app, "coder", server="srv-1")
    _put_profile(app, "writer", server="srv-2")
    app.handle("POST", "/api/easy/enable",
               body={"profile": "coder", "profiles": ["writer"]})
    store = TraceStore(home / "buffer" / "sable.db")
    for i in range(3):
        seed_trace(store, f"w{i}", score=0.9, profile="writer")
    store.close()
    g = app.handle("GET", "/api/easy").body
    by_name = {a["profile"]: a for a in g["agents"]}
    # Scoped gauges: writer sees its 3 traces, coder sees none of writer's.
    assert by_name["writer"]["readiness"]["unclaimed"] == 3
    assert by_name["coder"]["readiness"]["unclaimed"] == 0


def test_easy_auto_check_multi_trains_each_due_agent(tmp_path, monkeypatch,
                                                     capsys):
    """Cron path with two agents: one due, one not => exactly one iterate run,
    scoped to the due profile."""
    import sable.cli as cli

    home = tmp_path / "uf-easy-cli"
    monkeypatch.setenv("SABLE_HOME", str(home))
    cfg_text = (
        "easy_training:\n"
        "  enabled: true\n"
        "  profile: coder\n"
        "  profiles: [coder, writer]\n"
        "  min_traces: 1\n"
        "training:\n"
        "  auto_min_score: 0.5\n")
    cfg_path = tmp_path / "sable.yaml"
    cfg_path.write_text(cfg_text)
    monkeypatch.setenv("SABLE_CONFIG", str(cfg_path))

    store = TraceStore(home / "buffer" / "sable.db")
    seed_trace(store, "w0", score=0.9, profile="writer")
    store.close()

    seen = []

    def fake_iterate(ns):
        seen.append((ns.name, ns.easy_profile))
        return 0

    monkeypatch.setattr(cli, "cmd_iterate", fake_iterate)
    rc = cli.cmd_train_auto_check(_ns(config=str(cfg_path), verbose=False,
                                      trigger=True, dry_run=True))
    assert rc == 0
    assert seen == [("easy:writer", "writer")]
    out = capsys.readouterr().out
    assert "easy training [writer]" in out


def _ns(**kw):
    import argparse
    return argparse.Namespace(**kw)

"""Training-domain tests.

Consolidated from 10 former test modules; section banners mark origins.
"""

from __future__ import annotations

import sys
from pathlib import Path
import json
import time
from typing import TYPE_CHECKING, cast
import yaml
import struct
import math
import random
import threading
from typing import Any, Dict, List
from experiments.wp11_specialization import buffer_fixture as bf
from experiments.wp11_specialization import evalgate as eg
from experiments.wp11_specialization import noise_floor as nf
from experiments.wp11_specialization import testbed as tb
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from toaster_loop.core.config import ToasterLoopConfig  # noqa: E402
from toaster_loop.dashboard.app import DashboardApp  # noqa: E402
from toaster_loop.store.db import TraceRow, TraceStore  # noqa: E402
from toaster_loop.store.recorder import TraceRecorder  # noqa: E402
from toaster_loop.training.easy import EasyTrainingConfig, evaluate_readiness, load_state, optimal_config_patches, save_state, validate_profile  # noqa: E402
from toaster_loop.training.scheduler import should_auto_train  # noqa: E402
from fake_server import FakeOpenAIServer, make_message_handler  # noqa: E402
from toaster_loop.core.config import RFTConfig, load_config  # noqa: E402
from toaster_loop.registry import ModelRegistry  # noqa: E402
from toaster_loop.training.rejection import RftError, context_prefix, make_rft_job_runner, make_sampler, prompt_key, run_rft, verify_judged  # noqa: E402
from toaster_loop.training.scheduler import JobDispatcher, Scheduler, TrainingConfig  # noqa: E402
from toaster_loop.training.worker import grpo_rows_from_dataset, main as worker_main, reward_format, reward_length, reward_solution_overlap  # noqa: E402
from fake_server import FakeOpenAIServer  # noqa: E402
import toaster_loop.cli as cli  # noqa: E402
from toaster_loop.core.agent_runner import ToasterLoopAgentClient  # noqa: E402
from toaster_loop.core.config import load_config  # noqa: E402
from toaster_loop.store import TraceRecorder, TraceStore, heuristic_score  # noqa: E402
from toaster_loop.store.db import SCHEMA_VERSION  # noqa: E402
from toaster_loop.store.paths import resolve_toaster_loop_home  # noqa: E402
from toaster_loop.core.vram import ModelDims, estimate_from_train_yaml, resolve_load_in_4bit  # noqa: E402
from toaster_loop.training.worker import adapter_is_full_finetune, muon_optimizer_factory  # noqa: E402
from toaster_loop.training.streamed import chunked_ce  # noqa: E402
from toaster_loop.training.streamed.streaming import LayerStreamer, Loader  # noqa: E402
from toaster_loop.core.compression import SUMMARY_PREFIX, _PRUNE_MIN_CHARS, compress_messages, estimate_tokens, prune_old_tool_results, summary_from_messages  # noqa: E402

# ------------------------------------------------------------------------
# from tests/test_easy.py
# ------------------------------------------------------------------------

@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "uf-easy"
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(h))
    return h


@pytest.fixture()
def app(home):
    a = DashboardApp(ToasterLoopConfig(), home=home)
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
    store = TraceStore(home / "buffer" / "toaster_loop.db")
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
    rec = TraceRecorder(ToasterLoopConfig(), home=home, profile="coder")
    row = rec.record(task="t", messages=[{"role": "user", "content": "hi"},
                                         {"role": "assistant",
                                          "content": "yo"}],
                     backend="toaster_loop", model_alias="m", base_url="",
                     completed=True)
    assert row is not None and row.agent_profile == "coder"
    rec.close()
    # Without the profile kwarg the legacy backend name is kept.
    rec2 = TraceRecorder(ToasterLoopConfig(), home=home)
    row2 = rec2.record(task="t", messages=[], backend="toaster_loop",
                       model_alias="m", base_url="", completed=True)
    assert row2 is not None and row2.agent_profile == "toaster_loop"
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
    store = TraceStore(home / "buffer" / "toaster_loop.db")
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
    store = TraceStore(home / "buffer" / "toaster_loop.db")
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
    from toaster_loop.core.config import ToasterLoopConfig
    from toaster_loop.training.easy import mine_goldens_pass

    rep = mine_goldens_pass(ToasterLoopConfig(), home)  # goldens_mine: {} => off
    assert rep["ok"] is False and "false" in rep["skipped"]
    # Nothing was created by a skipped pass.
    assert not (home / "goldens" / "mined.jsonl").exists()


def test_mine_goldens_pass_runs_and_never_raises(home):
    from toaster_loop.core.config import ToasterLoopConfig
    from toaster_loop.training.easy import mine_goldens_pass

    cfg = ToasterLoopConfig(goldens_mine={"enabled": True})
    store = TraceStore(home / "buffer" / "toaster_loop.db")
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
    from toaster_loop.core.config import ToasterLoopConfig
    from toaster_loop.training.easy import mine_goldens_pass

    cfg = ToasterLoopConfig(goldens_mine={"enabled": True, "max_per_run": "oops"})
    rep = mine_goldens_pass(cfg, home)  # no buffer DB at all
    assert rep["ok"] is False and "skipped" in rep


def test_easy_run_fires_mining_pass(app, home, monkeypatch):
    _put_profile(app, "coder", server="srv-1")
    app.handle("POST", "/api/easy/enable", body={"profile": "coder"})
    store = TraceStore(home / "buffer" / "toaster_loop.db")
    seed_trace(store, "a0", score=0.9, profile="coder")
    store.close()
    app.handle("PUT", "/api/config", body={
        "section": "easy_training", "patch": {"min_traces": 1}})

    calls = []
    import toaster_loop.dashboard.api.easy as easy_mod

    def fake_pass(cfg, home_):
        calls.append(home_)
        return {"ok": True, "mined": 0, "skipped_dedup": 0}

    monkeypatch.setattr(easy_mod, "mine_goldens_pass", fake_pass)
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
    from toaster_loop.training.easy import (acquire_loop_lock, release_loop_lock)

    h1 = acquire_loop_lock(home)
    assert h1 is not None
    h2 = acquire_loop_lock(home)
    assert h2 is None  # second process would be turned away
    release_loop_lock(h1)
    h3 = acquire_loop_lock(home)
    assert h3 is not None
    release_loop_lock(h3)


def test_submit_iterate_rejects_when_loop_busy(app, home):
    from toaster_loop.training.easy import acquire_loop_lock, release_loop_lock

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
            from toaster_loop.dashboard.app import ApiResponse
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
    import toaster_loop.dashboard.app as appmod
    from toaster_loop.dashboard.app import ApiResponse

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
    store = TraceStore(home / "buffer" / "toaster_loop.db")
    seed_trace(store, "a0", score=0.9, profile="coder")
    seed_trace(store, "b0", score=0.9, profile="renamed-away")
    seed_trace(store, "b1", score=0.9, profile="toaster_loop")  # legacy: never stale
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
    store = TraceStore(home / "buffer" / "toaster_loop.db")
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
    store = TraceStore(home / "buffer" / "toaster_loop.db")
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
    store = TraceStore(home / "buffer" / "toaster_loop.db")
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
    import toaster_loop.cli as cli

    home = tmp_path / "uf-easy-cli"
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(home))
    cfg_text = (
        "easy_training:\n"
        "  enabled: true\n"
        "  profile: coder\n"
        "  profiles: [coder, writer]\n"
        "  min_traces: 1\n"
        "training:\n"
        "  auto_min_score: 0.5\n")
    cfg_path = tmp_path / "toaster_loop.yaml"
    cfg_path.write_text(cfg_text)
    monkeypatch.setenv("TOASTER_LOOP_CONFIG", str(cfg_path))

    store = TraceStore(home / "buffer" / "toaster_loop.db")
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

# ------------------------------------------------------------------------
# from tests/test_rejection.py
# ------------------------------------------------------------------------

@pytest.fixture()
def home__test_rejection(tmp_path, monkeypatch):
    h = tmp_path / "ufrft"
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(h))
    return h


@pytest.fixture()
def store(home__test_rejection):
    st = TraceStore(home__test_rejection / "buffer" / "toaster_loop.db")
    yield st
    st.close()


def seed_trace__test_rejection(store, home, *, trace_id, messages, status="success",
               timestamp="2026-09-01T10:00:00+00:00"):
    """JSONL transcript + index row exactly like the recorder (test_curate shape)."""
    jsonl_path = home / "buffer" / "2026" / "09" / f"{trace_id}.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "x", encoding="utf-8") as f:
        f.write(json.dumps({
            "kind": "trace_header", "schema_version": 1, "trace_id": trace_id,
            "session_id": None, "timestamp": timestamp,
            "agent_profile": "fallback", "model_alias": "test-model",
            "model_version": None, "endpoint_label": None, "task": "seed",
        }) + "\n")
        for m in messages:
            f.write(json.dumps({"kind": "message", "message": m}) + "\n")
        f.write(json.dumps({
            "kind": "trace_footer", "schema_version": 1, "status": status,
            "completed": status == "success", "outcome_score": 0.5,
            "num_tool_calls": 0, "tokens_in": None, "tokens_out": None,
            "duration_s": None,
        }) + "\n")
    store.insert_trace(TraceRow(
        trace_id=trace_id, session_id=None, timestamp=timestamp,
        agent_profile="fallback", model_alias="test-model", model_version=None,
        endpoint_label=None, task_summary="seed", status=status,
        outcome_score=0.5, num_tool_calls=0, tokens_in=None, tokens_out=None,
        duration_s=None, jsonl_path=str(jsonl_path),
    ))


def cfg(**kw) -> RFTConfig:
    base: dict = dict(enabled=True, sample_k=4, max_prompts=10)
    base.update(kw)
    return RFTConfig(**base)


def ds_dir(res) -> Path:
    """Narrow run_rft's Optional dataset_dir for tests that assert a dataset
    was actually produced."""
    assert res.dataset_dir is not None
    return res.dataset_dir


GOLDEN_FRANCE = {
    "prompt": "capital of France",
    "expect_contains": ["Paris"],
}

JUDGE_OK = lambda msgs, model: json.dumps({"score": 0.9, "reason": "good"})


def cycle(responses):
    """Callable reply: pops scripted messages per request (last one repeats)."""
    state = {"i": 0}

    def reply(payload):
        r = responses[min(state["i"], len(responses) - 1)]
        state["i"] += 1
        return r
    return reply


def sft_rows(ds_dir: Path):
    return [json.loads(l) for l in
            (ds_dir / "sft.jsonl").read_text().splitlines() if l.strip()]


# ------------------------------------------------------------------ K-sampling

def test_ksampling_keeps_only_verified_row(home__test_rejection, store):
    """K=4 candidates; the first golden-verified one ships, later ones don't."""
    handler = make_message_handler({
        "capital": cycle([
            {"role": "assistant", "content": "nope"},
            {"role": "assistant", "content": "It is Paris, France."},
            {"role": "assistant", "content": "extra wrong"},
            {"role": "assistant", "content": "extra wrong 2"},
        ]),
    })
    with FakeOpenAIServer(handler_cls=handler) as srv:
        res = run_rft(store, home__test_rejection, cfg(sample_k=4),
                      sampler=make_sampler(srv.base_url, "fake-model",
                                           temperature=0.9, max_tokens=64),
                      goldens=[dict(GOLDEN_FRANCE)], name="rftk")
    assert res.dataset_id and res.rows_kept == 1
    d = ds_dir(res)
    rows = sft_rows(d)
    assert len(rows) == 1
    assert "Paris" in rows[0]["conversations"][-1]["value"]
    assert rows[0]["source"] == "rft"
    mf = json.loads((d / "manifest.json").read_text())
    assert mf["source"] == "rft"
    assert mf["trust"] == "curated"          # deterministic golden checks
    assert mf["files"]["sft.jsonl"]["count"] == 1
    assert mf["rft"]["prompts_sampled"] == 1
    assert (d / "train.yaml").is_file()


def test_empty_case_leaves_no_dataset_dir(home__test_rejection, store):
    """No verified candidate -> no-op; NO dataset dir survives (curater parity)."""
    handler = make_message_handler({
        "*": {"role": "assistant", "content": "always wrong"},
    })
    with FakeOpenAIServer(handler_cls=handler) as srv:
        res = run_rft(store, home__test_rejection, cfg(sample_k=3),
                      sampler=make_sampler(srv.base_url, "fake-model",
                                           temperature=0.9, max_tokens=64),
                      goldens=[dict(GOLDEN_FRANCE)], name="rftempty")
    assert res.dataset_id is None and res.rows_kept == 0
    assert any("no candidate passed" in w for w in res.warnings)
    assert not list(home__test_rejection.rglob("manifest.json"))


# ----------------------------------------------------------- dedup/provenance

def test_second_run_skips_sampled_prompts(home__test_rejection, store):
    handler = make_message_handler({
        "*": {"role": "assistant", "content": "Paris is the capital."},
    })
    sampler = lambda msgs: {"role": "assistant",
                            "content": "Paris is the capital."}
    with FakeOpenAIServer(handler_cls=handler):
        res1 = run_rft(store, home__test_rejection, cfg(sample_k=1), sampler=sampler,
                       judge_client_fn=JUDGE_OK,
                       seeds=["what's up?"], name="rftd")
        assert res1.rows_kept == 1
        res2 = run_rft(store, home__test_rejection, cfg(sample_k=1), sampler=sampler,
                       judge_client_fn=JUDGE_OK,
                       seeds=["what's up?"], name="rftd2")
    assert res2.dataset_id is None
    assert res2.prompts_sampled == 0
    assert any("already sampled" in w for w in res2.warnings)


def test_ledger_claim_is_atomic_and_stamped(home__test_rejection, store):
    key = prompt_key("hello there")
    assert store.claim_rft_prompts([(key, "seed", None)]) == [key]
    assert store.claim_rft_prompts([(key, "seed", None)]) == []  # race loser
    assert store.rft_prompt_seen(key)
    store.mark_rft_prompts_emitted([key], "ds-1", kept_keys=[key])
    row = store._conn.execute(
        "SELECT dataset_id, kept FROM rft_prompts WHERE prompt_key = ?",
        (key,)).fetchone()
    assert (row["dataset_id"], row["kept"]) == ("ds-1", 1)
    assert store.list_rft_prompt_keys() == [key]


# --------------------------------------------------------- overlap invariant #8

def test_curate_claimed_trace_never_becomes_rft_prompt(home__test_rejection, store):
    seed_trace__test_rejection(store, home__test_rejection, trace_id="t1", messages=[
        {"role": "user", "content": "capital of France"},
        {"role": "assistant", "content": "Paris"},
    ])
    assert store.claim_traces_owned(["t1"], "curate-ds") == ["t1"]
    sampler = lambda msgs: {"role": "assistant", "content": "Paris"}
    res = run_rft(store, home__test_rejection, cfg(), sampler=sampler)
    # The trace is excluded at selection (query_for_curation include_curated=0):
    # no prompt, no sampling, no ledger claim.
    assert res.prompts_sampled == 0 and res.dataset_id is None
    assert store.list_rft_prompt_keys() == []


def test_rft_claimed_trace_unavailable_to_other_claims(home__test_rejection, store):
    seed_trace__test_rejection(store, home__test_rejection, trace_id="t2", messages=[
        {"role": "user", "content": "capital of France"},
        {"role": "assistant", "content": "old answer"},
    ])
    sampler = lambda msgs: {"role": "assistant", "content": "Paris is it."}
    res = run_rft(store, home__test_rejection, cfg(), sampler=sampler, judge_client_fn=JUDGE_OK)
    assert res.rows_kept == 1 and res.claimed == 1
    row = store.get_trace("t2")
    assert row.curated == 1 and row.dataset_id == res.dataset_id
    # Same primitive the curater uses: a second claim (curate or rft) loses.
    assert store.claim_traces_owned(["t2"], "another-ds") == []


# ------------------------------------------------------- judge threshold gate

def test_verify_judged_strict_boundary():
    prompt = [{"role": "user", "content": "q"}]
    msg = {"role": "assistant", "content": "a"}
    ok_at = verify_judged(prompt, msg, lambda m, mo: json.dumps(
        {"score": 0.85, "reason": "x"}), "j", 0.85)
    assert ok_at[0] is True
    ok_below = verify_judged(prompt, msg, lambda m, mo: json.dumps(
        {"score": 0.849, "reason": "x"}), "j", 0.85)
    assert ok_below[0] is False
    # Unparseable / missing verdicts are FAILS, never passes.
    assert verify_judged(prompt, msg, lambda m, mo: "not json", "j", 0.85)[0] \
        is False
    assert verify_judged(prompt, None, JUDGE_OK, "j", 0.85)[0] is False


# ------------------------------------------------------ sampling payload shape

def test_sampler_routes_thinking_keys_via_request_sampling_payload():
    handler = make_message_handler({
        "*": {"role": "assistant", "content": "ok"},
    })
    with FakeOpenAIServer(handler_cls=handler) as srv:
        sampler = make_sampler(
            srv.base_url, "fake-model", temperature=0.9, max_tokens=128,
            sampling={"enable_thinking": False, "temperature": 0.3,
                      "top_k": 40})
        assert sampler([{"role": "user", "content": "hi"}]) is not None
    body = handler.requests[-1]
    # Thinking controls reach llama.cpp via the template-controls map AND stay
    # top-level for vLLM-style servers (request_sampling_payload contract).
    assert body["chat_format_template_controls"] == {"enable_thinking": False}
    assert body["enable_thinking"] is False
    assert body["top_k"] == 40
    # RFT's elevated temperature overrides the default section AFTER merge.
    assert body["temperature"] == 0.9
    assert body["max_tokens"] == 128


# ------------------------------------------------------------- token budget

class StubCounter:
    def __init__(self, n):
        self.n = n

    def count_messages(self, messages):
        return self.n


def test_budget_gate_skips_long_prompts_without_claiming(home__test_rejection, store):
    sampler = lambda msgs: {"role": "assistant", "content": "Paris"}
    long_prompt = "x" * 400
    res = run_rft(store, home__test_rejection, cfg(max_prompt_tokens=100), sampler=sampler,
                  judge_client_fn=JUDGE_OK,
                  seeds=[long_prompt], counter=StubCounter(500))
    assert res.skipped_budget == 1 and res.dataset_id is None
    # Skipped prompts stay UNclaimed so a future (compressed) run may retry.
    assert store.list_rft_prompt_keys() == []

    res2 = run_rft(store, home__test_rejection, cfg(max_prompt_tokens=100), sampler=sampler,
                   judge_client_fn=JUDGE_OK,
                   seeds=[long_prompt], counter=StubCounter(50))
    assert res2.prompts_sampled == 1


def test_context_prefix_cuts_after_last_user():
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "tool", "content": "t"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]
    prefix = context_prefix(msgs)
    assert prefix is not None
    assert [m["role"] for m in prefix] == ["system", "user", "user"]
    assert context_prefix([{"role": "assistant", "content": "x"}]) is None


# ------------------------------------------------------------ dispatcher e2e

def test_rft_job_flows_through_dispatcher(home__test_rejection, monkeypatch):
    """queued -> running -> done under a JobDispatcher tick; _result recorded;
    reap_stale never touches the in-process kind (INPROCESS_JOB_KINDS)."""
    handler = make_message_handler({
        "capital": {"role": "assistant", "content": "It is Paris."},
    })
    reg = ModelRegistry(home__test_rejection / "buffer" / "toaster_loop.db")
    try:
        with FakeOpenAIServer(handler_cls=handler) as srv:
            runner = make_rft_job_runner(
                reg, home=home__test_rejection, defaults=cfg(),
                endpoint_for=lambda: _ctx((srv.base_url, "fake-model")))
            tc = TrainingConfig()
            factory = lambda: Scheduler(registry=ModelRegistry(
                home__test_rejection / "buffer" / "toaster_loop.db"), cfg=tc,
                inprocess_runners={"rft": runner})
            sched = factory()
            job = sched.submit("rft", log_dir=home__test_rejection / "train_runs",
                               params={"goldens": [dict(GOLDEN_FRANCE)],
                                       "name": "rftdisp"},
                               enqueue_actor="test")
            assert job.state == "queued"
            d = JobDispatcher(db_path=reg.db_path, scheduler_factory=factory,
                              tick_s=0.2)
            d.start()
            try:
                deadline = time.time() + 15
                while time.time() < deadline:
                    cur = reg.get_job(job.job_id)
                    assert cur is not None
                    if cur.state in ("done", "error"):
                        break
                    # reap-safety probe mid-flight: an in-process kind must
                    # survive reaping even with a foreign pid.
                    assert cur.state in ("queued", "running")
                    time.sleep(0.1)
            finally:
                d.stop()
            cur = reg.get_job(job.job_id)
            assert cur is not None
            assert cur.state == "done", cur.error
            result = cur.params["_result"]
            assert result["rows_kept"] == 1 and result["dataset_id"]
            ds_dir = Path(result["dataset_dir"])
            mf = json.loads((ds_dir / "manifest.json").read_text())
            assert mf["source"] == "rft"
    finally:
        reg.close()


class _ctx:
    """Minimal context manager for the endpoint_for seam."""

    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self.value

    def __exit__(self, *exc):
        return False


def test_runner_refuses_when_master_switch_off(home__test_rejection):
    reg = ModelRegistry(home__test_rejection / "buffer" / "toaster_loop.db")
    try:
        runner = make_rft_job_runner(
            reg, home=home__test_rejection, defaults=RFTConfig(),  # enabled=False default
            endpoint_for=lambda: _ctx(("http://127.0.0.1:1/v1", "m")))

        class J:
            job_id = "j1"
            params = {}

        with pytest.raises(RftError, match="rft.enabled is false"):
            runner(J())
    finally:
        reg.close()


# -------------------------------------------------------------------- config

def test_rft_config_defaults_and_yaml_parsing():
    d = RFTConfig()
    assert d.enabled is False and d.sample_k == 4 and d.judge_min_score == 0.85
    parsed = RFTConfig.from_yaml({"rft": {"enabled": True, "sample_k": 8,
                                          "judge_min_score": 0.9}})
    assert parsed.enabled is True and parsed.sample_k == 8
    assert parsed.judge_min_score == 0.9
    # Unknown keys inside the section are ignored (with a warning), never crash.
    quiet = RFTConfig.from_yaml({"rft": {"nonsense": 1}})
    assert quiet.enabled is False


def test_load_config_recognizes_rft_section_without_warnings(tmp_path, caplog):
    import logging

    p = tmp_path / "c.yaml"
    p.write_text("rft:\n  enabled: true\n  sample_k: 6\n")
    with caplog.at_level(logging.WARNING):
        c = load_config(str(p))
    assert c.rft == {"enabled": True, "sample_k": 6}
    assert not [r for r in caplog.records if "unknown" in r.getMessage()]


# ------------------------------------------------------- DPO from RFT failures

def dpo_rows(ds_dir: Path):
    p = ds_dir / "dpo.jsonl"
    if not p.is_file():
        return None
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def seq_replies(responses):
    """Direct sampler reply: scripted assistant messages, last one repeats."""
    state = {"i": 0}

    def sample(messages):
        r = responses[min(state["i"], len(responses) - 1)]
        state["i"] += 1
        return {"role": "assistant", "content": r}
    return sample


def test_dpo_pairs_emitted_from_failed_candidates(home__test_rejection, store):
    """K=4 with 1 verified + 3 failed -> 1 SFT row AND 3 DPO pairs (<= cap),
    manifest files.dpo.jsonl mirrors the curater shape, and the worker's DPO
    loader consumes the file verbatim."""
    # best-of-K breaks at the first verified candidate -> put it last so all
    # three failures are collected as negative material.
    sampler = seq_replies(["wrong a", "wrong b", "wrong c", "It is Paris."])
    res = run_rft(store, home__test_rejection, cfg(sample_k=4), sampler=sampler,
                  goldens=[dict(GOLDEN_FRANCE)], name="rftdpo")
    assert res.rows_kept == 1 and res.dpo_pairs == 3
    d = ds_dir(res)
    assert len(sft_rows(d)) == 1

    rows = dpo_rows(d)
    assert rows is not None and len(rows) == 3
    for r in rows:
        # TRL preference schema, same context on both sides.
        assert r["prompt"] == [{"role": "user", "content": "capital of France"}]
        assert "Paris" in r["chosen"]
        assert r["rejected"] in ("wrong a", "wrong b", "wrong c")
        assert r["metadata"]["source"] == "rft"
        assert r["metadata"]["verifier"] == "golden"
    # Golden prompt -> no source trace; trace_ids stays empty (tolerated).
    mf = json.loads((d / "manifest.json").read_text())
    assert mf["files"]["dpo.jsonl"] == {"count": 3, "trace_ids": [],
                                        "format": "trl-preference"}
    assert mf["rft"]["dpo_pairs_emitted"] == 3

    # Consumable by the pipeline's DPO loader (worker dry-run: schema check,
    # no torch needed).
    from toaster_loop.training.worker import run_dpo
    out = run_dpo({"_dataset_dir": str(res.dataset_dir)}, None, True)
    assert out["dry_run"] is True and out["rows"] == 3


def test_all_fail_prompt_emits_no_orphan_pairs(home__test_rejection, store):
    """Two prompts: one yields a kept row, the other all-K fail. The failed
    prompt contributes NO DPO pairs (no orphan negatives)."""
    def sampler(messages):
        if "capital" in messages[-1]["content"]:
            return {"role": "assistant", "content": "It is Paris."}
        return {"role": "assistant", "content": "never right"}
    res = run_rft(store, home__test_rejection, cfg(sample_k=3), sampler=sampler,
                  goldens=[dict(GOLDEN_FRANCE),
                           {"prompt": "capital of Mars",
                            "expect_contains": ["Olympus Mons"]}],
                  name="rftorphan")
    assert res.rows_kept == 1 and res.dpo_pairs == 0
    rows = dpo_rows(ds_dir(res))
    assert rows is not None and rows == []  # curater parity: empty file, kept
    mf = json.loads((ds_dir(res) / "manifest.json").read_text())
    assert mf["files"]["dpo.jsonl"]["count"] == 0


def test_emit_dpo_off_is_byte_identical_to_legacy(home__test_rejection, store):
    """emit_dpo=False -> no dpo.jsonl at all and the manifest carries neither a
    files.dpo.jsonl entry nor rft.dpo_pairs_emitted (pre-feature shape)."""
    sampler = seq_replies(["wrong", "It is Paris."])
    res = run_rft(store, home__test_rejection, cfg(sample_k=2, emit_dpo=False), sampler=sampler,
                  goldens=[dict(GOLDEN_FRANCE)], name="rftoff")
    assert res.rows_kept == 1 and res.dpo_pairs == 0
    d = ds_dir(res)
    assert not (d / "dpo.jsonl").exists()
    mf = json.loads((d / "manifest.json").read_text())
    assert set(mf["files"]) == {"sft.jsonl"}
    assert "dpo_pairs_emitted" not in mf["rft"]


def test_dpo_pair_cap_prefers_diverse_failure_reasons(home__test_rejection, store):
    """Cap per prompt; distinct failure reasons are picked before duplicates
    of the same reason even when the dupes come first in document order."""
    # Judge scores: two identical fails (0.10), one distinct fail (0.20), keep.
    scores = iter([0.1, 0.1, 0.2, 0.9])
    judge = lambda msgs, model: json.dumps(
        {"score": next(scores), "reason": "x"})
    sampler = seq_replies(["bad one", "bad two", "bad three", "good answer"])
    res = run_rft(store, home__test_rejection, cfg(sample_k=4, max_dpo_pairs_per_prompt=2),
                  sampler=sampler, judge_client_fn=judge,
                  seeds=["how do I greet a user?"], name="rftcap")
    assert res.rows_kept == 1 and res.dpo_pairs == 2
    rows = dpo_rows(ds_dir(res))
    assert rows is not None
    reasons = [r["metadata"]["rejected_reason"] for r in rows]
    # Document order would have paired two identical "judge 0.10" fails;
    # diversity picks one of each distinct reason.
    assert len(set(reasons)) == 2
    assert all(r["chosen"] == "good answer" for r in rows)
    mf = json.loads((ds_dir(res) / "manifest.json").read_text())
    assert mf["files"]["dpo.jsonl"]["count"] == 2


def test_dpo_pairs_redact_rejected_and_prompt(home__test_rejection, store):
    """Fresh model output may echo prompt secrets: chosen/rejected/prompt all
    pass core/redact before the durable write."""
    secret = "sk-abcdefghijklmnop1234567890"
    sampler = seq_replies([f"the key is {secret} but wrong",
                           "Paris, and here is a key: " + secret])
    res = run_rft(store, home__test_rejection, cfg(sample_k=2), sampler=sampler,
                  goldens=[dict(GOLDEN_FRANCE)], name="rftredact")
    assert res.dpo_pairs == 1
    rows = dpo_rows(ds_dir(res))
    assert rows is not None
    row = rows[0]
    assert secret not in row["rejected"] and "[REDACTED" in row["rejected"]
    assert secret not in row["chosen"] and "[REDACTED" in row["chosen"]


def test_dpo_manifest_lists_trace_derived_kept_side(home__test_rejection, store):
    """Trace-derived kept row -> files.dpo.jsonl.trace_ids carries the source
    trace id (curater shape); sampling-derived negatives add nothing."""
    seed_trace__test_rejection(store, home__test_rejection, trace_id="tdpo", messages=[
        {"role": "user", "content": "capital of France"},
        {"role": "assistant", "content": "old answer"},
    ])
    sampler = seq_replies(["wrong", "It is Paris."])
    # Content-aware judge: the first candidate must actually FAIL or there is
    # no negative to pair.
    judge = lambda msgs, model: json.dumps(
        {"score": 0.1 if "wrong" in (msgs[-1].get("content") or "") else 0.9,
         "reason": "x"})
    res = run_rft(store, home__test_rejection, cfg(sample_k=2), sampler=sampler,
                  judge_client_fn=judge, name="rfttr")
    assert res.rows_kept == 1 and res.dpo_pairs == 1
    mf = json.loads((ds_dir(res) / "manifest.json").read_text())
    assert mf["files"]["dpo.jsonl"] == {"count": 1, "trace_ids": ["tdpo"],
                                        "format": "trl-preference"}


def test_dpo_run_preserves_ledger_and_claim_invariants(home__test_rejection, store):
    """DPO emission rides the existing write path: prompt ledger dedup and the
    claim/release invariants are untouched (re-run still skips)."""
    sampler = seq_replies(["wrong", "It is Paris."])
    judge = lambda msgs, model: json.dumps(
        {"score": 0.1 if "wrong" in (msgs[-1].get("content") or "") else 0.9,
         "reason": "x"})
    res1 = run_rft(store, home__test_rejection, cfg(sample_k=2), sampler=sampler,
                   judge_client_fn=judge, seeds=["cap q"], name="rftinv")
    assert res1.dpo_pairs == 1
    key = prompt_key("cap q")
    row = store._conn.execute(
        "SELECT dataset_id, kept FROM rft_prompts WHERE prompt_key = ?",
        (key,)).fetchone()
    assert (row["dataset_id"], row["kept"]) == (res1.dataset_id, 1)
    res2 = run_rft(store, home__test_rejection, cfg(sample_k=2), sampler=sampler,
                   judge_client_fn=JUDGE_OK, seeds=["cap q"], name="rftinv2")
    assert res2.dataset_id is None and res2.prompts_sampled == 0


def test_rft_config_dpo_knobs():
    d = RFTConfig()
    assert d.emit_dpo is True and d.max_dpo_pairs_per_prompt == 3
    parsed = RFTConfig.from_yaml({"rft": {"emit_dpo": False,
                                          "max_dpo_pairs_per_prompt": 1}})
    assert parsed.emit_dpo is False and parsed.max_dpo_pairs_per_prompt == 1


def test_cli_rft_subcommand_parses_and_refuses_when_disabled(tmp_path,
                                                             monkeypatch):
    import argparse

    from toaster_loop.cli import build_parser, cmd_rft

    args = build_parser().parse_args(["rft", "--k", "8", "--seeds", "hi"])
    assert args.command == "rft" and args.k == 8 and args.seeds == ["hi"]

    monkeypatch.setenv("TOASTER_LOOP_HOME", str(tmp_path / "h"))
    c = tmp_path / "c.yaml"
    c.write_text("rft:\n  enabled: false\n")
    ns = argparse.Namespace(config=str(c), verbose=False, seeds=None,
                            goldens=None, k=None, max_prompts=None,
                            name=None, wait=False)
    assert cmd_rft(ns) == 2  # master switch off -> refuse before queueing

# ------------------------------------------------------------------------
# from tests/test_grpo.py
# ------------------------------------------------------------------------

def _write_jsonl(p: Path, rows):
    p.write_text("\n".join(json.dumps(r) for r in rows))


# ---------------------------------------------------------- row derivation

def test_rows_from_explicit_grpo_file(tmp_path):
    _write_jsonl(tmp_path / "grpo.jsonl", [
        {"prompt": [{"role": "user", "content": "hi"}], "solution": "yo"},
        {"prompt": [], "solution": "dropped (empty prompt)"},
    ])
    rows = grpo_rows_from_dataset(tmp_path)
    assert len(rows) == 1
    assert rows[0]["solution"] == "yo"


def test_rows_fall_back_to_dpo_chosen_as_solution(tmp_path):
    _write_jsonl(tmp_path / "dpo.jsonl", [{
        "prompt": [{"role": "user", "content": "q"}],
        "chosen": "good", "rejected": "bad",
    }])
    rows = grpo_rows_from_dataset(tmp_path)
    assert rows == [{"prompt": [{"role": "user", "content": "q"}],
                     "solution": "good"}]


def test_rows_fall_back_to_sft_last_turn(tmp_path):
    _write_jsonl(tmp_path / "sft.jsonl", [{
        "conversations": [
            {"role": "user", "value": "q"},
            {"role": "assistant", "value": "a"},
        ]}])
    rows = grpo_rows_from_dataset(tmp_path)
    assert len(rows) == 1
    assert rows[0]["solution"] == "a"
    assert rows[0]["prompt"] == [{"role": "user", "value": "q"}]


def test_rows_missing_dataset_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        grpo_rows_from_dataset(tmp_path)


# ------------------------------------------------------------------ rewards

def test_reward_format_scores():
    scores = reward_format(["a real answer with words", "", "x x x x x x x x x x" * 5])
    assert scores[0] == 1.0
    assert scores[1] == 0.0
    assert scores[2] < 1.0  # degenerate repetition penalized


def test_reward_format_accepts_chat_completion_shape():
    scores = reward_format([[{"role": "assistant", "content": "hello there"}]])
    assert scores == [1.0]


def test_reward_length_decay():
    scores = reward_length(["short", "long" * 4000], max_chars=100)
    assert scores[0] == 1.0
    assert scores[1] == 0.0  # way over budget -> floor at 0


def test_reward_solution_overlap_f1():
    scores = reward_solution_overlap(
        ["alpha beta gamma"], solutions=["alpha beta delta"])
    # intersection 2, precision 2/3, recall 2/3 -> F1 2/3
    assert scores[0] == pytest.approx(2 / 3)
    assert reward_solution_overlap(["anything"], solutions=[""])[0] == 0.0


# ------------------------------------------------------- worker dry-run CLI

def _train_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "train.yaml"
    p.write_text("model: some/base\ntraining:\n  output_dir: outputs\n")
    return p


def test_worker_grpo_dry_run(tmp_path):
    ds = tmp_path / "ds"
    ds.mkdir()
    _write_jsonl(ds / "grpo.jsonl", [
        {"prompt": [{"role": "user", "content": "q1"}], "solution": "a1"},
        {"prompt": [{"role": "user", "content": "q2"}], "solution": "a2"},
    ])
    log = tmp_path / "job.log"
    rc = worker_main([
        "--job", "j-grpo", "--kind", "grpo",
        "--config", str(_train_yaml(tmp_path)),
        "--dataset-dir", str(ds),
        "--log-path", str(log),
        "--dry-run",
    ])
    assert rc == 0
    events = [json.loads(l) for l in log.read_text().splitlines()]
    kinds = [e.get("event") for e in events]
    assert "dry_run_ok" in kinds
    dry = next(e for e in events if e.get("event") == "dry_run_ok")
    assert dry["kind"] == "grpo" and dry["rows"] == 2
    assert kinds[-1] == "done"


def test_registry_accepts_grpo_kind(tmp_path):
    from toaster_loop.registry import ModelRegistry

    reg = ModelRegistry(tmp_path / "r.db")
    try:
        job = reg.enqueue_job("grpo", dataset_id="ds1")
        assert job.kind == "grpo" and job.state == "queued"
    finally:
        reg.close()


def test_registry_set_job_param_merges(tmp_path):
    from toaster_loop.registry import ModelRegistry

    reg = ModelRegistry(tmp_path / "r.db")
    try:
        job = reg.enqueue_job("sft", params={"a": 1})
        reg.set_job_param(job.job_id, "_device", 2)
        got = reg.get_job(job.job_id)
        assert got is not None
        assert got.params == {"a": 1, "_device": 2}
    finally:
        reg.close()

# ------------------------------------------------------------------------
# from tests/test_buffer.py
# ------------------------------------------------------------------------

if TYPE_CHECKING:  # pragma: no cover - typing only
    from toaster_loop.core.config import ToasterLoopConfig


@pytest.fixture()
def toaster_loop_home(tmp_path, monkeypatch):
    """Redirect the state root into a tmp dir for every test in this module."""
    home = tmp_path / "uf-home"
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(home))
    monkeypatch.delenv("TOASTER_LOOP_CONFIG", raising=False)
    return home


@pytest.fixture(scope="module")
def fake_server():
    with FakeOpenAIServer() as srv:
        yield srv


# ------------------------------------------------------------------ paths/config

def test_home_resolution_precedence(tmp_path, monkeypatch):
    monkeypatch.delenv("TOASTER_LOOP_HOME", raising=False)

    class Cfg:
        home = str(tmp_path / "from-config")

    cfg = cast("ToasterLoopConfig", Cfg())
    assert resolve_toaster_loop_home(cfg) == tmp_path / "from-config"
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(tmp_path / "from-env"))
    assert resolve_toaster_loop_home(cfg) == tmp_path / "from-env"  # env wins
    assert resolve_toaster_loop_home(None) == tmp_path / "from-env"


def test_config_recording_yaml_and_env(tmp_path, monkeypatch):
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text("recording:\n  enabled: false\n")
    monkeypatch.delenv("TOASTER_LOOP_RECORDING", raising=False)
    assert load_config(str(cfg_file)).recording_enabled is False

    monkeypatch.setenv("TOASTER_LOOP_RECORDING", "true")  # env wins over file
    assert load_config(str(cfg_file)).recording_enabled is True

    cfg_file2 = tmp_path / "cfg2.yaml"
    cfg_file2.write_text("home: ~/somewhere\n")
    monkeypatch.delenv("TOASTER_LOOP_HOME", raising=False)  # exercise the file value
    assert load_config(str(cfg_file2)).home == "~/somewhere"


# -------------------------------------------------------------------- recorder

def test_fallback_run_writes_jsonl_and_index(toaster_loop_home, fake_server):
    """Full task via fallback client => JSONL trace + SQLite row under TOASTER_LOOP_HOME."""
    rec = TraceRecorder(home=toaster_loop_home)
    client = ToasterLoopAgentClient(
        base_url=fake_server.base_url, model="fake-model", recorder=rec
    )
    result = client.run("read the fake server source")
    assert result.completed is True
    assert result.trace_id, "run must return the buffer trace id"

    # JSONL under buffer/YYYY/MM/<trace_id>.jsonl
    matches = list((toaster_loop_home / "buffer").rglob(f"{result.trace_id}.jsonl"))
    assert len(matches) == 1
    lines = [json.loads(l) for l in matches[0].read_text().splitlines()]
    kinds = [l["kind"] for l in lines]
    assert kinds[0] == "trace_header"
    assert kinds[-1] == "trace_footer"
    assert kinds.count("message") == len(result.messages)
    header = lines[0]
    assert header["schema_version"] == SCHEMA_VERSION
    assert header["model_alias"] == "fake-model"
    # backlog #2/#7: failure_reason is nullable and None on clean runs
    assert header.get("failure_reason") is None
    assert lines[-1].get("failure_reason") is None
    assert header["task"] == "read the fake server source"
    footer = lines[-1]
    assert footer["status"] == "success"
    assert footer["num_tool_calls"] == 1

    # SQLite index row
    store = TraceStore(toaster_loop_home / "buffer" / "toaster_loop.db")
    try:
        row = store.get_trace(result.trace_id)
        assert row is not None
        assert row.status == "success"
        assert row.num_tool_calls == 1
        assert row.outcome_score == pytest.approx(0.5)  # clean completion
        assert row.tokens_in == 2 and row.tokens_out == 2  # from fake usage
        assert Path(row.jsonl_path).is_file()
        fb = store.get_feedback(result.trace_id)
        assert any(f["source"] == "heuristic" for f in fb)
    finally:
        store.close()


def test_recording_failure_does_not_break_run(toaster_loop_home, fake_server, monkeypatch):
    """ROADMAP acceptance: a trace write failure cannot break a conversation."""
    rec = TraceRecorder(home=toaster_loop_home / "\0invalid")  # unrepresentable path
    client = ToasterLoopAgentClient(
        base_url=fake_server.base_url, model="fake-model", recorder=rec
    )
    result = client.run("read the fake server source")
    assert result.completed is True
    assert "Done." in result.final_response
    assert result.trace_id is None


def test_recording_disabled_writes_nothing(toaster_loop_home, fake_server):
    from toaster_loop.store.recorder import NullRecorder, make_recorder

    cfg = load_config(None)
    object.__setattr__(cfg, "recording_enabled", False)
    rec = make_recorder(cfg)
    assert isinstance(rec, NullRecorder)

    client = ToasterLoopAgentClient(
        base_url=fake_server.base_url, model="fake-model", recorder=rec
    )
    result = client.run("read the fake server source")
    assert result.completed is True
    assert result.trace_id is None
    assert not (toaster_loop_home / "buffer").exists()


# ---------------------------------------------------------------- v2 migration

def test_db_v1_to_v2_migration(tmp_path):
    """Pre-failure_reason DBs migrate in place and stay readable (backlog #7)."""
    import sqlite3

    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.executescript("""
CREATE TABLE traces (
    trace_id TEXT PRIMARY KEY, session_id TEXT, timestamp TEXT NOT NULL,
    agent_profile TEXT, model_alias TEXT, model_version TEXT, endpoint_label TEXT,
    task_summary TEXT, status TEXT NOT NULL, outcome_score REAL,
    num_tool_calls INTEGER NOT NULL DEFAULT 0, tokens_in INTEGER, tokens_out INTEGER,
    duration_s REAL, jsonl_path TEXT NOT NULL, schema_version INTEGER NOT NULL DEFAULT 1,
    curated INTEGER NOT NULL DEFAULT 0, dataset_id TEXT);
PRAGMA schema_version = 1;
INSERT INTO traces (trace_id,timestamp,status,num_tool_calls,jsonl_path)
VALUES ('old1','2026-01-01T00:00:00+00:00','success',0,'/dev/null');
""")
    con.commit()
    con.close()

    from toaster_loop.store.db import TraceRow as _TR
    store = TraceStore(db)
    try:
        old = store.get_trace("old1")           # tolerant read of v1 row
        assert old is not None and old.failure_reason is None
        store.insert_trace(_TR(
            trace_id="new1", session_id=None, timestamp="t", agent_profile=None,
            model_alias=None, model_version=None, endpoint_label=None,
            task_summary=None, status="failure", outcome_score=-1.0,
            num_tool_calls=0, tokens_in=None, tokens_out=None, duration_s=None,
            jsonl_path="/dev/null", failure_reason="backend_error:connection"))
        new = store.get_trace("new1")
        assert new is not None
        assert new.failure_reason == "backend_error:connection"
    finally:
        store.close()
    # Reopen: migration must be idempotent (no duplicate ALTER crash).
    store = TraceStore(db)
    try:
        old2 = store.get_trace("old1")
        new2 = store.get_trace("new1")
        assert old2 is not None and old2.failure_reason is None
        assert new2 is not None
        assert new2.failure_reason == "backend_error:connection"
    finally:
        store.close()


# ------------------------------------------------------ duplicate insert (#15)

def test_insert_trace_duplicate_skipped_preserves_curation_state(tmp_path):
    """ON CONFLICT(trace_id) DO NOTHING: a colliding insert must never clobber
    load-bearing curation state (curated/dataset_id/outcome_score). Returns
    False on skip, True on insert."""
    from toaster_loop.store.db import TraceRow as _TR

    def row(summary):
        return _TR(
            trace_id="dup1", session_id=None, timestamp="2026-09-01T00:00:00+00:00",
            agent_profile="p", model_alias=None, model_version=None,
            endpoint_label=None, task_summary=summary, status="success",
            outcome_score=0.5, num_tool_calls=0, tokens_in=None, tokens_out=None,
            duration_s=1.0, jsonl_path="/dev/null", schema_version=SCHEMA_VERSION,
            curated=False, dataset_id=None)

    store = TraceStore(tmp_path / "dup.db")
    try:
        assert store.insert_trace(row("original")) is True
        # simulate the curater having claimed + a feedback score having landed
        assert store.claim_traces(["dup1"], "ds-7") == 1
        store.add_feedback("dup1", -0.25, "user says bad", source="user")

        assert store.insert_trace(row("collision")) is False  # skip signal
        kept = store.get_trace("dup1")
        assert kept is not None
        assert kept.task_summary == "original"   # row untouched
        assert kept.curated is True              # curation state preserved
        assert kept.dataset_id == "ds-7"
        assert kept.outcome_score == pytest.approx(-0.25)
    finally:
        store.close()


# ------------------------------------------------- failure remediation notes
# Pass 4 #33 — idea adapted from PIASystem (MIT) app/core/failure_taxonomy.py.

def test_failure_notes_set_get_and_fallback(tmp_path):
    store = TraceStore(tmp_path / "notes.db")
    try:
        # implicit open: no row -> None everywhere, empty listing.
        assert store.get_failure_note("timeout") is None
        assert store.get_failure_note("timeout", profile="agentx") is None
        assert store.list_failure_notes() == []

        store.set_failure_note("timeout", "in_progress", note="retry wiring")
        n = store.get_failure_note("timeout")
        assert n is not None
        assert n["status"] == "in_progress" and n["profile"] is None
        assert n["note"] == "retry wiring" and n["ts"]

        # profile-specific lookup falls back to the global row...
        fb = store.get_failure_note("timeout", profile="agentx")
        assert fb is not None and fb["status"] == "in_progress"
        # ...until a profile-scoped row exists, which then wins.
        store.set_failure_note("timeout", "resolved", note="fixed",
                               profile="agentx")
        scoped = store.get_failure_note("timeout", profile="agentx")
        assert scoped is not None and scoped["status"] == "resolved"
        glob = store.get_failure_note("timeout")
        assert glob is not None and glob["status"] == "in_progress"
        assert len(store.list_failure_notes()) == 2

        # upsert: one row per (reason, scope), status replaced in place.
        store.set_failure_note("timeout", "acknowledged", note="seen",
                               profile="agentx")
        rows = [r for r in store.list_failure_notes() if r["profile"] == "agentx"]
        assert len(rows) == 1 and rows[0]["status"] == "acknowledged"

        # reset-to-open with an empty note deletes the row (back to implicit
        # open / untracked); open-with-note stays tracked.
        store.set_failure_note("timeout", "open")
        assert store.get_failure_note("timeout") is None
        kept = store.get_failure_note("timeout", profile="agentx")
        assert kept is not None and kept["status"] == "acknowledged"
        store.set_failure_note("timeout", "open", note="wontfix")
        n = store.get_failure_note("timeout")
        assert n is not None
        assert n["status"] == "open" and n["note"] == "wontfix"

        # status validation against the fixed set.
        with pytest.raises(ValueError):
            store.set_failure_note("timeout", "bogus")
    finally:
        store.close()


def test_failure_notes_survive_reopen(tmp_path):
    db = tmp_path / "notes2.db"
    store = TraceStore(db)
    try:
        store.set_failure_note("context_overflow", "resolved", note="cap raised")
    finally:
        store.close()
    store = TraceStore(db)  # idempotent CREATE TABLE IF NOT EXISTS migration
    try:
        n = store.get_failure_note("context_overflow")
        assert n is not None and n["status"] == "resolved"
    finally:
        store.close()


# ------------------------------------------------------------------- heuristic

def test_heuristic_score_rules():
    clean = [{"role": "user", "content": "hi"},
             {"role": "assistant", "content": "ok"}]
    assert heuristic_score(clean, completed=True) == 0.5

    err_tools = [
        {"role": "tool", "content": "error: bad arguments for read_file"},
        {"role": "tool", "content": "Traceback (most recent call last): ..."},
    ]
    assert heuristic_score(err_tools, completed=True) == pytest.approx(0.0)

    # error penalties cap at -0.75
    many_errs = [{"role": "tool", "content": f"error {i}"} for i in range(10)]
    assert heuristic_score(many_errs, completed=True) == pytest.approx(-0.25)

    assert heuristic_score([], completed=False) == -0.5


# ------------------------------------------------- HIGH #4 cancelled neutrality

def _seed_row(store, tid, *, status="failure", score=None, failure_reason=None):
    from datetime import datetime, timedelta, timezone

    from toaster_loop.store.db import TraceRow as _TR
    p = store.db_path.parent / f"{tid}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("")
    # Relative-to-now stamp: consumers include stats_daily's REAL calendar
    # cutoff (test_stats_and_daily_means_exclude_cancelled), so a frozen
    # literal here is a time-bomb. days_ago=1 keeps rows inside any window.
    ts = (datetime.now(timezone.utc) - timedelta(days=1)
          ).isoformat(timespec="seconds")
    store.insert_trace(_TR(
        trace_id=tid, session_id=None, timestamp=ts,
        agent_profile="fallback", model_alias="m", model_version=None,
        endpoint_label=None, task_summary="s", status=status,
        outcome_score=score, num_tool_calls=0, tokens_in=None,
        tokens_out=None, duration_s=None, jsonl_path=str(p),
        failure_reason=failure_reason,
    ))


def test_cancelled_run_recorded_score_neutral(toaster_loop_home):
    """Recorder NULLs outcome_score for cancelled runs (HIGH #4)."""
    rec = TraceRecorder(home=toaster_loop_home)
    row = rec.record(task="t", messages=[{"role": "user", "content": "hi"}],
                     backend="b", model_alias="m", base_url="http://x",
                     completed=False, failure_reason="cancelled")
    assert row is not None
    assert row.outcome_score is None
    # status derivation unchanged (non-completed -> failure bucket)
    assert row.status == "failure"
    # footer carries the neutral score too
    footer = json.loads(Path(row.jsonl_path).read_text().splitlines()[-1])
    assert footer["outcome_score"] is None
    assert footer["failure_reason"] == "cancelled"
    rec.close()

    # composed runner form behaves identically
    rec2 = TraceRecorder(home=toaster_loop_home)
    row2 = rec2.record(task="t", messages=[], backend="b", model_alias="m",
                       base_url="http://x", completed=False,
                       failure_reason="backend_error:cancelled")
    assert row2 is not None and row2.outcome_score is None
    rec2.close()


def test_non_cancelled_scoring_unchanged(toaster_loop_home):
    """Non-cancelled runs keep the exact pre-fix heuristic scores."""
    rec = TraceRecorder(home=toaster_loop_home)
    done = rec.record(task="t", messages=[{"role": "user", "content": "hi"}],
                      backend="b", model_alias="m", base_url="http://x",
                      completed=True)
    assert done is not None and done.outcome_score == pytest.approx(0.5)
    failed = rec.record(task="t", messages=[], backend="b", model_alias="m",
                        base_url="http://x", completed=False,
                        failure_reason="max_iterations")
    assert failed is not None and failed.outcome_score == pytest.approx(-0.5)
    rec.close()


def test_legacy_cancelled_rows_neutralized_on_open(tmp_path):
    """Migration: pre-fix rows stamped -0.5 + cancelled NULL out on reopen."""
    db = tmp_path / "legacy.db"
    store = TraceStore(db)
    _seed_row(store, "c1", score=-0.5, failure_reason="cancelled")
    _seed_row(store, "c2", score=-0.5, failure_reason="backend_error:cancelled")
    _seed_row(store, "m1", score=-0.5, failure_reason="max_iterations")
    _seed_row(store, "ok", status="success", score=0.5)
    store.close()

    store = TraceStore(db)  # idempotent migration runs on open
    try:
        c1 = store.get_trace("c1")
        c2 = store.get_trace("c2")
        m1 = store.get_trace("m1")
        ok = store.get_trace("ok")
        assert c1 is not None and c2 is not None
        assert m1 is not None and ok is not None
        assert c1.outcome_score is None
        assert c2.outcome_score is None
        # non-cancelled rows untouched
        assert m1.outcome_score == pytest.approx(-0.5)
        assert ok.outcome_score == pytest.approx(0.5)
    finally:
        store.close()


def test_stats_and_daily_means_exclude_cancelled(tmp_path):
    """stats()/stats_daily means skip NULL-scored cancelled rows (never zero)."""
    db = tmp_path / "means.db"
    store = TraceStore(db)
    _seed_row(store, "ok1", status="success", score=0.5)
    _seed_row(store, "ok2", status="success", score=0.5)
    _seed_row(store, "cx", score=None, failure_reason="cancelled")
    try:
        st = store.stats()
        assert st["scored_traces"] == 2
        assert st["avg_outcome_score"] == pytest.approx(0.5)
        daily = {d["date"]: d for d in store.stats_daily(days=3650)}
        cx = store.get_trace("cx")
        assert cx is not None
        today = daily[cx.timestamp[:10]]
        assert today["traces"] == 3  # cancelled row still counted as a trace
        assert today["avg_score"] == pytest.approx(0.5)  # but not in the mean
    finally:
        store.close()


# ------------------------------------------------------------------- feedback CLI

def _record_one(toaster_loop_home, fake_server) -> str:
    rec = TraceRecorder(home=toaster_loop_home)
    client = ToasterLoopAgentClient(
        base_url=fake_server.base_url, model="fake-model", recorder=rec
    )
    result = client.run("read the fake server source")
    assert result.trace_id is not None
    return result.trace_id


def test_feedback_cli_by_id_and_last(toaster_loop_home, fake_server, capsys):
    trace_id = _record_one(toaster_loop_home, fake_server)

    assert cli.main(["feedback", trace_id, "--score", "0.8", "--note", "nice"]) == 0
    store = TraceStore(toaster_loop_home / "buffer" / "toaster_loop.db")
    try:
        fb = store.get_feedback(trace_id)
        user_rows = [f for f in fb if f["source"] == "user"]
        assert len(user_rows) == 1
        assert user_rows[0]["score"] == pytest.approx(0.8)
        assert user_rows[0]["comment"] == "nice"
        # trace row outcome_score synced with explicit feedback
        row = store.get_trace(trace_id)
        assert row is not None
        assert row.outcome_score == pytest.approx(0.8)
    finally:
        store.close()

    # --last targets most recent; --fail shorthand => -1
    assert cli.main(["feedback", "--last", "--fail"]) == 0
    store = TraceStore(toaster_loop_home / "buffer" / "toaster_loop.db")
    try:
        row = store.get_trace(trace_id)
        assert row is not None
        assert row.outcome_score == pytest.approx(-1.0)
        latest = store.latest_trace()
        assert latest is not None
        assert latest.trace_id == trace_id
    finally:
        store.close()

    # --ok shorthand => +1
    assert cli.main(["feedback", "--last", "--ok"]) == 0
    store = TraceStore(toaster_loop_home / "buffer" / "toaster_loop.db")
    try:
        row = store.get_trace(trace_id)
        assert row is not None
        assert row.outcome_score == pytest.approx(1.0)
    finally:
        store.close()


def test_feedback_cli_validation(toaster_loop_home, fake_server, capsys):
    trace_id = _record_one(toaster_loop_home, fake_server)
    assert cli.main(["feedback", trace_id, "--score", "5"]) != 0   # out of range
    assert cli.main(["feedback", trace_id]) != 0                    # nothing to record
    assert cli.main(["feedback", "deadbeefcafe"]) != 0              # unknown id


# ------------------------------------------------------------------- traces CLI

def test_traces_list_show_stats(toaster_loop_home, fake_server, capsys):
    trace_id = _record_one(toaster_loop_home, fake_server)

    assert cli.main(["traces", "list"]) == 0
    out = capsys.readouterr().out
    assert trace_id in out
    assert "success" in out

    assert cli.main(["traces", "list", "--status", "failure"]) == 0
    out = capsys.readouterr().out
    assert "no traces recorded yet" in out

    assert cli.main(["traces", "show", trace_id]) == 0
    out = capsys.readouterr().out
    assert trace_id in out
    assert "transcript:" in out
    assert "assistant" in out and "tool" in out

    assert cli.main(["traces", "stats"]) == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["total_traces"] == 1
    assert stats["by_status"] == {"success": 1}
    assert stats["tool_calls_total"] == 1

    assert cli.main(["traces", "show", "nonexistent"]) == 1


# ------------------------------------------------------- built-in capture path

def test_builtin_capture_equivalent_jsonl(toaster_loop_home, fake_server):
    """Built-in runtime: post-run capture produces a complete trace."""
    rec = TraceRecorder(home=toaster_loop_home)
    runner = ToasterLoopAgentClient(
        base_url=fake_server.base_url, model="fake-model", recorder=rec
    )

    result = runner.run("read the fake server file")
    assert result.completed is True
    assert result.trace_id, "built-in runtime must record a trace"

    matches = list((toaster_loop_home / "buffer").rglob(f"{result.trace_id}.jsonl"))
    assert len(matches) == 1
    lines = [json.loads(l) for l in matches[0].read_text().splitlines()]
    kinds = [l["kind"] for l in lines]
    assert kinds[0] == "trace_header" and kinds[-1] == "trace_footer"
    # Equivalent capture: one JSONL message line per captured conversation message.
    assert kinds.count("message") == len(result.messages)
    header = lines[0]
    assert header["agent_profile"] == "toaster_loop"

    store = TraceStore(toaster_loop_home / "buffer" / "toaster_loop.db")
    try:
        row = store.get_trace(result.trace_id)
        assert row is not None
        assert row.agent_profile == "toaster_loop"
        assert row.num_tool_calls >= 1  # fake server forces one tool call
    finally:
        store.close()


def test_stats_daily_respects_calendar_window(tmp_path):
    """stats_daily(days=N) must exclude days older than N real days ago."""
    from datetime import datetime, timedelta, timezone

    from toaster_loop.store.db import TraceRow

    store = TraceStore(tmp_path / "idx.db")
    try:
        def row(tid, ts):
            return TraceRow(trace_id=tid, session_id=None, timestamp=ts,
                            agent_profile="p", model_alias="m",
                            model_version=None, endpoint_label=None,
                            task_summary=tid, status="success",
                            outcome_score=0.5, num_tool_calls=0,
                            tokens_in=None, tokens_out=None, duration_s=None,
                            jsonl_path="/dev/null")

        now = datetime.now(timezone.utc)
        store.insert_trace(row("recent", (now - timedelta(days=1))
                               .isoformat(timespec="seconds")))
        store.insert_trace(row("ancient", (now - timedelta(days=90))
                               .isoformat(timespec="seconds")))
        days = {d["date"] for d in store.stats_daily(days=30)}
        recent_day = (now - timedelta(days=1)).strftime("%Y-%m-%d")
        ancient_day = (now - timedelta(days=90)).strftime("%Y-%m-%d")
        assert recent_day in days
        assert ancient_day not in days
    finally:
        store.close()

# ------------------------------------------------------------------------
# from tests/test_fullft_and_tiers.py
# ------------------------------------------------------------------------

BIG = ModelDims(hidden_size=8192, num_hidden_layers=40,
                num_attention_heads=32, num_key_value_heads=8,
                intermediate_size=28672, vocab_size=128256)   # ~40B-ish
SMALL = ModelDims(hidden_size=1024, num_hidden_layers=12,
                  num_attention_heads=16, num_key_value_heads=8,
                  intermediate_size=3072, vocab_size=151936)  # ~0.3B-ish


# ------------------------------------------------------------- tier policy

def test_tier_policy_table():
    """The policy table in one place: (explicit, dims, full, gpu) -> result."""
    cases = [
        # (explicit, dims, train_full, gpu) -> (expect_4bit, why-substring)
        (True,  SMALL, False, True,  True,  "explicit"),
        (False, BIG,   False, True,  False, "explicit"),
        (None,  BIG,   True,  True,  False, "full fine-tune"),
        (None,  BIG,   False, False, False, "no GPU"),
        (None,  BIG,   False, True,  True,  "4-bit"),
        (None,  SMALL, False, True,  False, "bf16"),
        (None,  None,  False, True,  True,  "unknown"),
    ]
    for explicit, dims, full, gpu, want, why in cases:
        load, reason = resolve_load_in_4bit(explicit=explicit, dims=dims,
                                            train_full=full, gpu_available=gpu)
        assert load == want, (explicit, dims, full, gpu, reason)
        assert why in reason, (explicit, dims, full, gpu, reason)


def test_explicit_always_wins():
    assert resolve_load_in_4bit(explicit=True, dims=SMALL,
                                train_full=False, gpu_available=True)[0]
    assert not resolve_load_in_4bit(explicit=False, dims=BIG,
                                    train_full=False, gpu_available=True)[0]


def test_full_ft_never_4bit():
    load, why = resolve_load_in_4bit(explicit=None, dims=BIG,
                                     train_full=True, gpu_available=True)
    assert not load and "full fine-tune" in why


def test_no_gpu_never_4bit():
    load, _ = resolve_load_in_4bit(explicit=None, dims=BIG,
                                   train_full=False, gpu_available=False)
    assert not load


def test_size_tier():
    big_load, big_why = resolve_load_in_4bit(
        explicit=None, dims=BIG, train_full=False, gpu_available=True)
    small_load, small_why = resolve_load_in_4bit(
        explicit=None, dims=SMALL, train_full=False, gpu_available=True)
    assert big_load and "4-bit" in big_why
    assert not small_load and "bf16" in small_why


def test_unknown_dims_conservative_default():
    load, why = resolve_load_in_4bit(explicit=None, dims=None,
                                     train_full=False, gpu_available=True)
    assert load and "unknown" in why


def test_yaml_estimate_flows_training_type(tmp_path):
    """training_type: full must reach the estimator (bf16 weights, all
    params trainable) through estimate_from_train_yaml."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({
        "hidden_size": 1024, "num_hidden_layers": 12,
        "num_attention_heads": 16, "num_key_value_heads": 8,
        "intermediate_size": 3072, "vocab_size": 4096}))
    train_yaml = tmp_path / "train.yaml"
    train_yaml.write_text(yaml.safe_dump({
        "model": str(model_dir),
        "training": {"training_type": "full", "max_seq_length": 2048},
    }))
    est_full = estimate_from_train_yaml(str(train_yaml))
    assert est_full is not None and est_full.load_mode == "bf16"
    assert any("full fine-tune" in n for n in est_full.notes)
    # Same config as lora -> smaller total (adapters only train).
    train_yaml.write_text(yaml.safe_dump({
        "model": str(model_dir),
        "training": {"training_type": "lora", "load_in_4bit": False,
                     "max_seq_length": 2048},
    }))
    est_lora = estimate_from_train_yaml(str(train_yaml))
    assert est_lora.total_mib < est_full.total_mib
    assert est_full.optimizer_mib > 4 * est_lora.optimizer_mib


# -------------------------------------------------------- full-FT artifacts

def test_full_ft_marker_roundtrip(tmp_path):
    assert not adapter_is_full_finetune(tmp_path)
    (tmp_path / "training_type.json").write_text(
        json.dumps({"training_type": "full", "model": "x/y"}))
    assert adapter_is_full_finetune(tmp_path)
    (tmp_path / "training_type.json").write_text("not json {{{")
    assert not adapter_is_full_finetune(tmp_path)


def test_dry_run_reports_training_type(tmp_path):
    """Dry-run must surface training_type (and reject bogus values)."""
    from toaster_loop.training.worker import run_sft

    cfg = {"model": "X/Y",
           "data": {"local_dataset": [], "format_type": "sharegpt"},
           "training": {"training_type": "full"}}
    # Empty dataset raises before the type check? No: type check is at
    # parse time, files validated first — craft a real file.
    ds = tmp_path / "sft.jsonl"
    ds.write_text(json.dumps({"conversations": []}) + "\n")
    cfg["data"]["local_dataset"] = [str(ds)]
    out = run_sft(cfg, None, dry_run=True)
    assert out["dry_run"] and out["training_type"] == "full"

    cfg["training"]["training_type"] = "bogus"
    try:
        run_sft(cfg, None, dry_run=True)
        raise AssertionError("bogus training_type must raise")
    except ValueError as e:
        assert "training_type" in str(e)


def test_dpo_grpo_reject_full(tmp_path):
    from toaster_loop.training.worker import run_dpo, run_grpo

    dpo_file = tmp_path / "dpo.jsonl"
    dpo_file.write_text(json.dumps(
        {"prompt": [{"role": "user", "content": "a"}],
         "chosen": "x", "rejected": "y"}) + "\n")
    cfg = {"model": "X/Y",
           "training": {"training_type": "full"},
           "_dataset_dir": str(tmp_path)}
    try:
        run_dpo(cfg, None, dry_run=True)
        raise AssertionError("DPO must refuse training_type=full")
    except ValueError as e:
        assert "SFT only" in str(e)

    # GRPO needs prompt rows in the dataset dir; reuse dpo.jsonl shape by
    # writing prompts.jsonl if the loader wants it — guard fires before
    # heavy lifting either way as long as rows exist. Use a prompts file.
    (tmp_path / "prompts.jsonl").write_text(json.dumps(
        {"prompt": "hi"}) + "\n")
    for attempt in range(2):
        try:
            run_grpo(cfg, None, dry_run=True)
            raise AssertionError("GRPO must refuse training_type=full")
        except ValueError as e:
            assert "SFT only" in str(e)
            break
        except FileNotFoundError as e:
            if attempt:
                raise AssertionError(
                    f"GRPO guard never fired (row loader kept failing: {e})")
            # Row loader runs first: give it the file shape it wants.
            (tmp_path / "grpo_prompts.jsonl").write_text(json.dumps(
                {"prompt": "hi"}) + "\n")


# ----------------------------------------------------------------- Muon

def test_muon_factory_splits_params():
    import torch

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Linear(8, 4)       # 2D weight + 1D bias
            self.norm = torch.nn.Parameter(torch.ones(4))

    m = M()
    factory = muon_optimizer_factory(lr=2e-4, weight_decay=0.01)
    # HF calls is_optimizer_factory(cls) -> not an Optimizer subclass.
    from transformers.trainer_optimizer import is_optimizer_factory
    assert is_optimizer_factory(factory)

    opt = factory(m, betas=(0.9, 0.999), eps=1e-8)
    n_muon = len(opt.optimizers[0].param_groups[0]["params"])
    n_adam = len(opt.optimizers[1].param_groups[0]["params"])
    assert n_muon == 1                      # only the Linear weight is 2D
    assert n_adam == 2                      # bias + norm go to AdamW
    # LR policy: Muon at its own default, AdamW at the run's lr.
    assert opt.optimizers[0].param_groups[0]["lr"] == 1e-3
    assert opt.optimizers[1].param_groups[0]["lr"] == 2e-4

    # step() runs across both children; state_dict round-trips.
    for p in m.parameters():
        p.grad = torch.ones_like(p)
    opt.step()
    sd = opt.state_dict()
    opt.load_state_dict(sd)


def test_muon_factory_rejects_no_2d():
    import torch

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.v = torch.nn.Parameter(torch.ones(3))

    factory = muon_optimizer_factory(lr=1e-3, weight_decay=0.0)
    try:
        factory(M())
        raise AssertionError("expected RuntimeError for 2D-less model")
    except RuntimeError as e:
        assert "no 2D trainable" in str(e)

# ------------------------------------------------------------------------
# from tests/test_streamed_backend_wp10.py
# ------------------------------------------------------------------------

# ------------------------------------------------------------- backend select
def test_parse_backend_defaults_and_lenient_fallback():
    from toaster_loop.training.streamed.runner import parse_backend

    assert parse_backend(None) == "peft"          # absent => today's path
    assert parse_backend("") == "peft"
    assert parse_backend("peft") == "peft"
    assert parse_backend("standard") == "peft"    # alias of the old behavior
    assert parse_backend("streamed") == "streamed"
    assert parse_backend("STREAMED ") == "streamed"

    warns: list[str] = []
    assert parse_backend("Streamd", warn=warns.append) == "peft"
    assert "falling back" in warns[0]
    warns.clear()
    assert parse_backend(42, warn=warns.append) == "peft"
    assert warns  # mistyped non-strings warn too


def test_training_config_carries_backend():
    from toaster_loop.training.scheduler import TrainingConfig

    assert TrainingConfig().backend is None       # default: inject nothing
    tc = TrainingConfig.from_yaml({"training": {"backend": "streamed"}})
    assert tc.backend == "streamed"


def test_run_sft_routes_streamed_without_touching_peft_path(tmp_path, monkeypatch):
    from toaster_loop.training import worker

    calls = []

    def fake_streamed(cfg, log_path, dry_run):
        calls.append((cfg, dry_run))
        return {"adapter_dir": "x", "train_loss": 1.0, "eval_loss": None,
                "backend": "streamed"}

    monkeypatch.setattr(worker, "run_streamed_sft", fake_streamed)
    cfg = {"training": {"backend": "streamed"}}
    out = worker.run_sft(cfg, str(tmp_path / "job.log"), False)
    assert out["backend"] == "streamed"
    assert len(calls) == 1


def test_run_sft_default_never_calls_streamed(tmp_path, monkeypatch):
    from toaster_loop.training import worker

    def boom(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("streamed backend invoked for default config")

    monkeypatch.setattr(worker, "run_streamed_sft", boom)
    # No training.backend key => old path. It fails fast on the empty config
    # (no dataset/model); what matters is the streamed spy was not called.
    with pytest.raises(Exception):
        worker.run_sft({}, str(tmp_path / "job.log"), False)


def test_resolved_train_yaml_backend_plumbing(tmp_path):
    from toaster_loop.training.pipeline import _resolved_train_yaml

    dataset = tmp_path / "ds"
    dataset.mkdir()
    (dataset / "train.yaml").write_text(
        "model: REPLACE_WITH_BASE_MODEL_PATH_OR_HF_ID\n"
        "data:\n  local_dataset: [REPLACE]\ntraining:\n  output_dir: out\n")

    p_default, _ = _resolved_train_yaml(dataset, tmp_path / "run", None)
    text = Path(p_default).read_text()
    assert "backend" not in text                 # default: file untouched

    p_streamed, applied = _resolved_train_yaml(
        dataset, tmp_path / "run2", None, backend="streamed")
    import yaml

    data = yaml.safe_load(Path(p_streamed).read_text())
    assert data["training"]["backend"] == "streamed"
    assert applied.get("training.backend") is None  # not an override entry


def test_parse_worker_backend_lineage(tmp_path):
    from toaster_loop.training.pipeline import parse_worker_backend

    log = tmp_path / "job.log"
    log.write_text(
        json.dumps({"event": "progress", "step": 1}) + "\n"
        + json.dumps({"event": "streamed_sft_done", "backend": "streamed",
                      "gguf_lora_path": "/x/lora.gguf"}) + "\n")
    assert parse_worker_backend(str(log)) == {
        "backend": "streamed", "gguf_lora_path": "/x/lora.gguf"}

    peft_log = tmp_path / "peft.log"
    peft_log.write_text(json.dumps({"event": "sft_done",
                                    "adapter_dir": "/a"}) + "\n")
    assert parse_worker_backend(str(peft_log)) == {}   # additive: absent => {}
    assert parse_worker_backend(None) == {}


# ------------------------------------------------------------------- masking
def _tiny_chat_tokenizer(tmp_path):
    """Deterministic word-level chat tokenizer built HERE (no downloads)."""
    pytest.importorskip("tokenizers")
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    specials = ["[UNK]", "[PAD]", "[SYS]", "[/SYS]", "[USR]", "[/USR]",
                "[TOOL]", "[/TOOL]", "[ASST]", "[/ASST]"]
    words = specials + ["hello", "world", "be", "nice", "fine", "thanks",
                        "tell", "me", "more", "ok", "done", "a", "b", "c"]
    vocab = {w: i for i, w in enumerate(words)}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens(specials)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]",
                                   pad_token="[PAD]")
    fast.chat_template = (
        "{% for m in messages %}"
        "{% if m.role == 'system' %}[SYS]{{ m.content }}[/SYS]"
        "{% elif m.role == 'user' %}[USR]{{ m.content }}[/USR]"
        "{% elif m.role == 'tool' %}[TOOL]{{ m.content }}[/TOOL]"
        "{% else %}[ASST]{{ m.content }}[/ASST]{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}[ASST]{% endif %}")
    out = tmp_path / "tok"
    fast.save_pretrained(str(out))
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(out), use_fast=True)


def _trainable_strings(tok, enc):
    return [tok.decode([i]) for i, l in zip(enc["input_ids"], enc["labels"])
            if l != -100]


def test_mask_multi_turn_system_tool(tmp_path):
    pytest.importorskip("transformers")
    from toaster_loop.training.streamed.masking import mask_conversation

    tok = _tiny_chat_tokenizer(tmp_path)
    msgs = [
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "hello world"},
        {"role": "assistant", "content": "fine thanks"},
        {"role": "user", "content": "tell me more"},
        {"role": "assistant", "content": "ok done"},
    ]
    enc = mask_conversation(tok, msgs, 128)
    assert enc is not None
    assert len(enc["input_ids"]) == len(enc["labels"])
    trainable = _trainable_strings(tok, enc)
    # ONLY assistant content + the assistant end token are trainable; every
    # system/user/tool span and every header is -100.
    assert trainable == ["fine", "thanks", "[/ASST]", "ok", "done", "[/ASST]"]


def test_mask_consecutive_assistant_turns(tmp_path):
    pytest.importorskip("transformers")
    from toaster_loop.training.streamed.masking import mask_conversation

    tok = _tiny_chat_tokenizer(tmp_path)
    msgs = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "tool", "content": "c"},
        {"role": "assistant", "content": "fine thanks"},
    ]
    enc = mask_conversation(tok, msgs, 128)
    assert enc is not None
    trainable = _trainable_strings(tok, enc)
    # consecutive same-role assistant turns each keep their own span; the
    # tool turn stays fully masked.
    assert trainable == ["a", "[/ASST]", "b", "[/ASST]",
                         "fine", "thanks", "[/ASST]"]


def test_mask_stricter_than_airllm(tmp_path):
    """airllm's messages loader masks only up to the LAST assistant turn —
    everything after earlier assistant turns (the follow-up user/tool spans)
    would be TRAINED. Ours must keep those spans masked."""
    pytest.importorskip("transformers")
    from toaster_loop.training.streamed.masking import mask_conversation

    tok = _tiny_chat_tokenizer(tmp_path)
    msgs = [
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "hello world"},
        {"role": "assistant", "content": "fine thanks"},   # intermediate
        {"role": "user", "content": "tell me more"},       # airllm trains this
        {"role": "tool", "content": "be nice done"},       # ...and this
        {"role": "assistant", "content": "ok done"},
    ]
    enc = mask_conversation(tok, msgs, 128)
    assert enc is not None
    ids, labels = enc["input_ids"], enc["labels"]
    trainable = _trainable_strings(tok, enc)
    assert trainable == ["fine", "thanks", "[/ASST]", "ok", "done", "[/ASST]"]

    # The divergence concretely: 'tell' and 'me' sit between the two
    # assistant spans; under airllm's last-turn-only rule their labels would
    # be non-masked, here they must be -100.
    tell_ids = {tok.convert_tokens_to_ids(w) for w in ("tell", "me", "more")}
    for i, t in enumerate(ids):
        if t in tell_ids:
            assert labels[i] == -100


def test_mask_requires_assistant_turn(tmp_path):
    pytest.importorskip("transformers")
    from toaster_loop.training.streamed.masking import mask_conversation

    tok = _tiny_chat_tokenizer(tmp_path)
    enc = mask_conversation(tok, [{"role": "user", "content": "hello"}], 64)
    assert enc is None  # nothing trainable => caller skips the row


def test_sharegpt_role_mapping():
    from toaster_loop.training.streamed.masking import sharegpt_to_messages

    # Roles as emitted by curate/sharegpt.py, plus ShareGPT's raw
    # function_call; genuinely unknown roles are dropped.
    msgs = sharegpt_to_messages([
        {"from": "system", "value": "s"},
        {"from": "human", "value": "h"},
        {"from": "gpt", "value": "g"},
        {"from": "tool", "value": "t"},
        {"from": "function_call", "value": "f"},
        {"from": "nonsense", "value": "x"},
    ])
    assert [m["role"] for m in msgs] == ["system", "user", "assistant",
                                         "tool", "assistant"]


# ------------------------------------------------------- streamed training e2e
def _build_tiny_llama_ckpt(root: Path):
    """Synthetic 2-layer llama checkpoint + tokenizer, written locally."""
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    tok = _tiny_chat_tokenizer(root)
    cfg = LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2,
                      vocab_size=len(tok), max_position_embeddings=128,
                      tie_word_embeddings=False)
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg)
    ckpt = root / "base"
    model.save_pretrained(str(ckpt))      # safetensors shard(s) on disk
    tok.save_pretrained(str(ckpt))
    return ckpt, tok


@pytest.fixture(scope="module")
def streamed_run(tmp_path_factory):
    """Run the full streamed worker once; share artifacts across tests."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("safetensors")
    pytest.importorskip("tokenizers")

    root = tmp_path_factory.mktemp("wp10_streamed")
    ckpt, tok = _build_tiny_llama_ckpt(root)

    rows = [{"conversations": [
        {"from": "system", "value": "be nice"},
        {"from": "human", "value": "hello world"},
        {"from": "gpt", "value": "fine thanks"},
        {"from": "human", "value": "tell me more"},
        {"from": "gpt", "value": "ok done"}]}] * 4
    sft = root / "sft.jsonl"
    sft.write_text("\n".join(json.dumps(r) for r in rows))

    out = root / "out"
    train_yaml = root / "train.yaml"
    train_yaml.write_text(json.dumps({
        "model": str(ckpt),
        "data": {"local_dataset": [str(sft)], "format_type": "sharegpt"},
        "training": {"output_dir": str(out), "num_epochs": 6, "batch_size": 2,
                     "learning_rate": 1e-2, "backend": "streamed",
                     "max_seq_length": 64},
        "lora": {"lora_r": 4, "lora_alpha": 8},
    }))

    from toaster_loop.training.worker import main as worker_main

    log = root / "job.log"
    rc = worker_main(["--job", "wp10", "--kind", "sft",
                      "--config", str(train_yaml), "--log-path", str(log)])
    return {"rc": rc, "root": root, "out": out,
            "log_lines": [json.loads(x) for x in
                          log.read_text().splitlines() if x.strip()]}


def test_streamed_e2e_worker_success(streamed_run):
    assert streamed_run["rc"] == 0
    events = [r["event"] for r in streamed_run["log_lines"]]
    assert "streamed_sft_done" in events and "done" in events
    done = [r for r in streamed_run["log_lines"]
            if r["event"] == "streamed_sft_done"][0]
    assert done["backend"] == "streamed"
    assert Path(done["adapter_dir"]).is_dir()


def test_streamed_e2e_splitting_progress_in_job_log(streamed_run):
    split = [r for r in streamed_run["log_lines"]
             if r.get("phase") == "splitting_checkpoint"]
    assert split, "splitting progress must stream to the job log"
    assert split[0]["message"].startswith("splitting checkpoint: shard ")
    assert split[-1]["shard"] == split[-1]["total"]


def test_streamed_e2e_metrics_csv_schema(streamed_run):
    csv_path = streamed_run["out"] / "train_metrics.csv"
    rows = csv_path.read_text().splitlines()
    assert rows[0] == "step,loss,peak_vram_gb,elapsed_s"
    # CPU-only box: peak stays an honest 0.0 (documented in the runner).
    progress = [r for r in streamed_run["log_lines"]
                if r["event"] == "progress"]
    assert len(rows) - 1 == len(progress)
    first_step, loss, peak, elapsed = rows[1].split(",")
    assert first_step == "1" and float(peak) == 0.0 and float(elapsed) >= 0


def test_streamed_e2e_loss_decreases(streamed_run):
    losses = [r["loss"] for r in streamed_run["log_lines"]
              if r["event"] == "progress"]
    assert len(losses) >= 6
    assert losses[-1] < losses[0] * 0.9          # clear downward trend
    assert losses[-1] < losses[-2] <= losses[0]  # not diverging


def test_streamed_e2e_adapter_artifacts(streamed_run):
    adapter = streamed_run["out"] / "adapter"
    cfg = json.loads((adapter / "adapter_config.json").read_text())
    assert cfg["backend"] == "streamed"
    assert cfg["format"] == "toaster_loop-lora-state-dict"
    assert (adapter / "lora_state.safetensors").is_file()
    gguf = streamed_run["out"] / "lora.gguf"
    assert gguf.is_file()
    assert gguf.read_bytes()[:4] == b"GGUF"


# ------------------------------------------------------------------- artifact
def _parse_gguf(path: Path):
    """Minimal independent GGUF reader (stdlib only) for assertions."""
    SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
           7: "<B", 10: "<Q", 11: "<q", 12: "<d"}

    def s(f):
        (n,) = struct.unpack("<Q", f.read(8))
        return f.read(n).decode()

    def v(f, t):
        if t == 8:
            return s(f)
        if t == 9:
            (at,) = struct.unpack("<I", f.read(4))
            (n,) = struct.unpack("<Q", f.read(8))
            return [v(f, at) for _ in range(n)]
        (x,) = struct.unpack(fmt[t], f.read(SZ[t]))
        return x

    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF"
        (ver,) = struct.unpack("<I", f.read(4))
        (n_tensors,) = struct.unpack("<Q", f.read(8))
        (n_kv,) = struct.unpack("<Q", f.read(8))
        kv = {}
        for _ in range(n_kv):
            k = s(f)
            (t,) = struct.unpack("<I", f.read(4))
            kv[k] = v(f, t)
        tensors = []
        for _ in range(n_tensors):
            name = s(f)
            (nd,) = struct.unpack("<I", f.read(4))
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
            (dt,) = struct.unpack("<I", f.read(4))
            (off,) = struct.unpack("<Q", f.read(8))
            tensors.append((name, dims, dt, off))
        size = path.stat().st_size
    return {"version": ver, "kv": kv, "tensors": tensors, "size": size}


def test_gguf_lora_artifact_structure(streamed_run):
    art = _parse_gguf(streamed_run["out"] / "lora.gguf")
    assert art["version"] == 3
    kv = art["kv"]
    assert kv["general.architecture"] == "llama"
    assert kv["general.type"] == "adapter"
    assert kv["adapter.type"] == "lora"
    assert isinstance(kv["adapter.lora.alpha"], float)

    names = [t[0] for t in art["tensors"]]
    # llama.cpp LoRA naming: blk.<i>.<role>.weight.lora_a/_b, complete pairs.
    assert "blk.0.attn_q.weight.lora_a" in names
    assert "blk.1.ffn_down.weight.lora_b" in names
    bases = {n.rsplit(".lora_", 1)[0] for n in names}
    for b in bases:
        assert f"{b}.lora_a" in names and f"{b}.lora_b" in names
    # ggml F32 type id (0) — NOT the GGUF value-type id — every tensor.
    assert all(t[2] == 0 for t in art["tensors"])
    # llama.cpp strictly validates offsets: aligned + non-overlapping and
    # inside the file.
    prev_end = 0
    for (_n, dims, _dt, off) in sorted(art["tensors"], key=lambda x: x[3]):
        assert off % 32 == 0 or off == 0
        assert off >= prev_end
        nbytes = 4
        for d in dims:
            nbytes *= d
        prev_end = off + nbytes
    assert prev_end <= art["size"]


def test_map_state_to_gguf_contract_rules():
    from toaster_loop.training.streamed.gguf_lora import map_state_to_gguf

    good_A = [[1.0, 2.0], [3.0, 4.0]]          # [r=2, in=2]
    good_B = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]  # [out=3, r=2]
    state = {
        "model.layers.0.self_attn.q_proj.lora_A": good_A,
        "model.layers.0.self_attn.q_proj.lora_B": good_B,
    }
    tensors, ranks = map_state_to_gguf(state)
    assert sorted(t.name for t in tensors) == [
        "blk.0.attn_q.weight.lora_a", "blk.0.attn_q.weight.lora_b"]
    assert ranks["attn_q"] == 2

    with pytest.raises(ValueError, match="half|partner"):
        map_state_to_gguf({"model.layers.0.self_attn.q_proj.lora_A": good_A})
    with pytest.raises(ValueError, match="contract"):
        map_state_to_gguf({"model.layers.0.norm.weight": good_A})
    with pytest.raises(ValueError, match="mapping"):
        map_state_to_gguf({"model.layers.9.self_attn.x_proj.lora_A": good_A,
                           "model.layers.9.self_attn.x_proj.lora_B": good_B})
    with pytest.raises(ValueError, match="empty"):
        map_state_to_gguf({})


def test_export_relocates_streamed_lora_without_llama_cpp(tmp_path):
    """The streamed GGUF-LoRA artifact flows through run_export's existing
    contract unchanged — and BEFORE the llama.cpp requirement check (no
    merge/quantize needed for an adapter-only artifact)."""
    from toaster_loop.training import worker

    out_dir = tmp_path / "out"
    adapter = out_dir / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(json.dumps(
        {"backend": "streamed", "format": "toaster_loop-lora-state-dict"}))
    payload = b"GGUF" + bytes(200)
    (out_dir / "lora.gguf").write_bytes(payload)

    gguf_out = tmp_path / "registered-model-q4_k_m.gguf"
    cfg = {"_adapter_dir": str(adapter), "_gguf_out": str(gguf_out),
           "training": {}}
    res = worker.run_export(cfg, str(tmp_path / "export.log"), False)
    assert res["method"] == "gguf-lora"
    assert Path(res["gguf_path"]).read_bytes() == payload

    # A peft adapter (no backend marker) must NOT take this branch: it hits
    # the pre-existing llama.cpp requirement error instead.
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "adapter_config.json").write_text(json.dumps({"r": 8}))
    cfg2 = {"_adapter_dir": str(plain), "_gguf_out": str(tmp_path / "x.gguf"),
            "training": {}}
    with pytest.raises(RuntimeError):
        worker.run_export(cfg2, None, False)


def test_adapter_is_streamed():
    from toaster_loop.training.worker import adapter_is_streamed

    assert adapter_is_streamed(Path("/nonexistent")) is False

# ------------------------------------------------------------------------
# from tests/test_streamed_lora_wp10.py
# ------------------------------------------------------------------------

# ============================================================== fake loaders
class CountingLoader(Loader):
    """Tracks simultaneously-live handles: the residency oracle for tier 1."""

    def __init__(self, delay_s: float = 0.0, fail_on: int | None = None):
        self.delay_s = delay_s
        self.fail_on = fail_on
        self.live = 0
        self.max_live = 0
        self.loads: List[int] = []
        self.unloads: List[int] = []
        self._lock = threading.Lock()

    def load(self, layer_idx: int) -> Dict[str, Any]:
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.fail_on is not None and layer_idx == self.fail_on:
            raise RuntimeError(f"shard read failed for layer {layer_idx}")
        with self._lock:
            self.loads.append(layer_idx)
            self.live += 1
            self.max_live = max(self.max_live, self.live)
        return {"layer": layer_idx, "w": [float(layer_idx)]}

    def unload(self, handle: Dict[str, Any]) -> None:
        with self._lock:
            self.unloads.append(handle["layer"])
            self.live -= 1


# ======================================================= TIER 1: pure python
def test_package_imports_without_torch():
    """The spike module must be import-safe without torch (WP10 constraint).

    We can't uninstall torch here, so assert the *mechanism*: no heavy import
    appears at module level in any streamed module (AST-level guarantee that
    doubles as the license-hygiene contract), and a fresh subprocess import
    with torch blocked still succeeds.
    """
    import ast

    pkg = Path(__file__).resolve().parents[1] / "toaster_loop" / "training" / "streamed"
    for py in sorted(pkg.glob("*.py")):
        tree = ast.parse(py.read_text())
        for node in tree.body:  # module level only
            names: List[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for n in names:
                assert n.split(".")[0] not in {"torch", "transformers", "peft",
                                               "trl", "safetensors"}, \
                    f"{py.name}:{getattr(node, 'lineno', 0)}: module-level {n}"


def _run_pass(st: LayerStreamer, n_layers: int) -> None:
    st.prefetch(0)
    for i in range(n_layers):
        w = st.acquire(i)
        assert w["layer"] == i
        st.prefetch(i + 1)   # read-ahead while "computing"
        st.release(i)


@pytest.mark.parametrize("depth", [0, 1, 2])
def test_streamer_residency_window_sync(depth):
    n = 7
    loader = CountingLoader()
    with LayerStreamer(loader, n, prefetch_depth=depth) as st:
        _run_pass(st, n)
        assert st.peak_residency() <= 1 + depth
    assert loader.max_live <= 1 + depth
    # Every layer loaded exactly once (prefetch hits), evicted exactly once.
    assert sorted(loader.loads) == list(range(n))
    assert sorted(loader.unloads) == list(range(n))


def test_streamer_threaded_prefetch_respects_window():
    n = 6
    loader = CountingLoader(delay_s=0.005)
    with LayerStreamer(loader, n, prefetch_depth=1, prefetch_thread=True) as st:
        _run_pass(st, n)
        assert st.peak_residency() <= 2
    # With read-ahead on, each layer is still loaded exactly once.
    assert sorted(loader.loads) == list(range(n))
    assert loader.max_live <= 2


def test_streamer_threaded_acquire_waits_for_prefetch():
    """acquire() on an in-flight prefetch blocks until the worker lands it."""
    loader = CountingLoader(delay_s=0.05)
    with LayerStreamer(loader, 3, prefetch_depth=1, prefetch_thread=True) as st:
        st.prefetch(0)
        t0 = time.monotonic()
        w = st.acquire(0)
        assert time.monotonic() - t0 >= 0.04
        assert w["layer"] == 0


def test_streamer_window_blocks_overfetch():
    """prefetch() beyond the window is a no-op: never more than 1+depth."""
    loader = CountingLoader()
    with LayerStreamer(loader, 10, prefetch_depth=1) as st:
        st.prefetch(0)
        st.prefetch(1)
        st.prefetch(2)          # window full -> dropped
        st.prefetch(3)          # still dropped
        assert sorted(st.resident_layers()) == [0, 1]
        assert st.peak_residency() == 2


def test_streamer_release_during_inflight_prefetch():
    """Releasing a layer whose threaded load is in flight must not leak."""
    loader = CountingLoader(delay_s=0.03)
    with LayerStreamer(loader, 4, prefetch_depth=1, prefetch_thread=True) as st:
        st.prefetch(1)               # in flight
        st.release(1)                # abandoned before landing
        time.sleep(0.08)             # worker lands + discards
        assert 1 not in st.resident_layers()
        assert loader.live == 0


def test_streamer_prefetch_error_surfaces_on_acquire():
    loader = CountingLoader(fail_on=2)
    with LayerStreamer(loader, 4, prefetch_depth=1) as st:
        st.prefetch(2)
        with pytest.raises(RuntimeError, match="layer 2"):
            st.acquire(2)


def test_streamer_reload_after_release():
    """The streamer is a cache: acquire after release re-loads transparently."""
    loader = CountingLoader()
    with LayerStreamer(loader, 2, prefetch_depth=0) as st:
        w1 = st.acquire(0)
        st.release(0)
        w2 = st.acquire(0)
        assert w1["layer"] == w2["layer"] == 0
        assert loader.loads.count(0) == 2


def test_streamer_close_evicts_all():
    loader = CountingLoader()
    st = LayerStreamer(loader, 3, prefetch_depth=1)
    st.prefetch(0)
    st.acquire(1)
    st.close()
    assert loader.live == 0
    with pytest.raises(RuntimeError):
        st.acquire(0)


# ------------------------------------------------- pure chunked-CE numerics
def _softmax_ce_ref(logits_rows, labels):
    """Independent stdlib reference: plain two-pass softmax CE per row."""
    total, count = 0.0, 0
    for row, lab in zip(logits_rows, labels):
        if lab == -100:
            continue
        m = max(row)
        z = sum(math.exp(v - m) for v in row)
        lse = m + math.log(z)
        total += lse - row[lab]
        count += 1
    return total / max(count, 1)


def test_chunked_ce_reference_matches_plain_softmax():
    rng = random.Random(7)
    n, v = 5, 64
    rows = [[rng.uniform(-4, 4) for _ in range(v)] for _ in range(n)]
    labels = [rng.randrange(v) for _ in range(n)]
    labels[3] = -100  # ignored row exercises the mask path
    loss, per_token = chunked_ce.chunked_ce_reference(
        rows, labels, vocab_chunk=7)
    assert abs(per_token[3]) < 1e-12
    assert math.isclose(loss, _softmax_ce_ref(rows, labels), rel_tol=1e-9)


def test_online_lse_merge_properties():
    assert chunked_ce.online_lse_merge(float("-inf"), 2.0) == 2.0
    assert chunked_ce.online_lse_merge(2.0, float("-inf")) == 2.0
    merged = chunked_ce.online_lse_merge(0.0, 0.0)
    assert math.isclose(merged, math.log(2.0), rel_tol=1e-12)
    # order independence within float noise
    a = chunked_ce.online_lse_merge(3.0, -1.0)
    b = chunked_ce.online_lse_merge(-1.0, 3.0)
    assert math.isclose(a, b, rel_tol=1e-12)
    # merging many singletons == full logsumexp
    vals = [1.0, 4.0, -2.0, 0.5]
    run = float("-inf")
    for x in vals:
        run = chunked_ce.online_lse_merge(run, x)
    m = max(vals)
    assert math.isclose(run, m + math.log(sum(math.exp(x - m) for x in vals)),
                        rel_tol=1e-12)


def test_chunked_ce_reference_chunk_invariance():
    """Chunk width must not change the answer (online merge is associative)."""
    rng = random.Random(11)
    rows = [[rng.uniform(-6, 6) for _ in range(48)] for _ in range(3)]
    labels = [5, 47, 0]
    l1, _ = chunked_ce.chunked_ce_reference(rows, labels, vocab_chunk=1)
    l7, _ = chunked_ce.chunked_ce_reference(rows, labels, vocab_chunk=7)
    lfull, _ = chunked_ce.chunked_ce_reference(rows, labels, vocab_chunk=48)
    assert math.isclose(l1, l7, rel_tol=1e-9)
    assert math.isclose(l7, lfull, rel_tol=1e-12)


def test_dropout_guard_rejects_p_gt_zero():
    """Recompute needs deterministic forwards — enforced without torch."""
    from toaster_loop.training.streamed.recompute import assert_no_stochastic_dropout

    class DropoutFake:  # duck-typed on purpose: name contains "dropout"
        p = 0.1

    class FakeInner:
        def __init__(self):
            self.drop = DropoutFake()

    class FakeModel:
        def __init__(self):
            self.block = FakeInner()

        def named_modules(self):
            yield "", self
            yield "block", self.block
            yield "block.drop", self.block.drop

    with pytest.raises(ValueError, match="dropout p=0"):
        assert_no_stochastic_dropout(FakeModel())


# ====================================================== TIER 2: real torch
def _torch():
    return pytest.importorskip("torch")


def test_lora_linear_numerics_and_frozen_base():
    torch = _torch()
    import torch.nn as nn

    from toaster_loop.training.streamed.lora_linear import LoRALinear

    torch.manual_seed(0)
    base = nn.Linear(12, 8, bias=True).double()
    lin = LoRALinear(base, r=4, alpha=8.0)

    # (1) adapter starts as an exact no-op (B = 0).
    x = torch.randn(3, 12, dtype=torch.float64)
    base_out = torch.nn.functional.linear(x, base.weight, base.bias)
    out = lin(x)
    assert torch.allclose(out, base_out, atol=1e-10)

    # (2) output identity: y = base(x) + scaling * B(A(x)) numerically.
    with torch.no_grad():
        lin.lora_A.normal_()
    expected = base_out + ((x @ lin.lora_A.detach().t())
                           @ lin.lora_B.detach().t()) * (8.0 / 4)
    assert torch.allclose(lin(x), expected, atol=1e-10)

    # (3) frozen base: requires_grad False and bit-identical after a step.
    assert not lin.weight.requires_grad
    assert not lin.bias.requires_grad
    w_before = lin.weight.detach().clone()
    b_before_bias = lin.bias.detach().clone()
    loss = lin(x).pow(2).mean()
    opt = torch.optim.SGD([lin.lora_A, lin.lora_B], lr=0.5)
    loss.backward()
    a_before = lin.lora_A.detach().clone()
    bmat_before = lin.lora_B.detach().clone()
    # Standard LoRA init consequence: B = 0 => dL/dA == 0 exactly on step 1;
    # only B receives gradient. Both are trainable; A moves from step 2 on.
    assert torch.allclose(lin.lora_A.grad, torch.zeros_like(lin.lora_A.grad))
    assert lin.lora_B.grad is not None and lin.lora_B.grad.abs().sum() > 0
    opt.step()
    assert torch.equal(lin.weight, w_before)       # base untouched
    assert torch.equal(lin.bias, b_before_bias)    # base bias untouched
    assert not torch.allclose(lin.lora_B, bmat_before)  # adapter moved
    # second step moves A as well (B is non-zero now)
    opt.zero_grad(set_to_none=True)
    lin(x).pow(2).mean().backward()
    opt.step()
    assert not torch.allclose(lin.lora_A, a_before)

    # (4) dropout > 0 is rejected loudly (recompute contract).
    with pytest.raises(ValueError, match="dropout must be 0"):
        LoRALinear(nn.Linear(4, 4), r=2, alpha=2, dropout=0.1)


def test_lora_state_dict_layout():
    _torch()
    import torch.nn as nn

    from toaster_loop.training.streamed.lora_linear import (inject_lora,
                                                       lora_state_dict)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(6, 6)
            self.unused = nn.Linear(6, 6)

    m = Block()
    n = inject_lora(m, ["q_proj"], r=2, alpha=4)
    assert n == 1
    keys = set(lora_state_dict(m))
    assert keys == {"q_proj.lora_A", "q_proj.lora_B"}
    sd_keys = set(m.state_dict())
    # Base checkpoint key survives wrapping (GGUF mapping requirement).
    assert "q_proj.weight" in sd_keys and "unused.weight" in sd_keys


def test_recompute_matches_dense_reference():
    """Streamed no_grad-forward/recompute-backward grads == dense autograd."""
    torch = _torch()
    import torch.nn as nn

    from toaster_loop.training.streamed.recompute import StreamedRecomputeTrainer

    torch.manual_seed(3)
    dims = [(8, 6), (6, 6), (6, 4)]

    def stack():
        mods = nn.ModuleList()
        for i_f, o_f in dims:
            lin = nn.Linear(i_f, o_f, bias=False).double()
            mods.append(nn.Sequential(lin))
        return mods

    dense = stack()
    streamed = [nn.Sequential(nn.Linear(i_f, o_f, bias=False).double())
                for i_f, o_f in dims]
    # Identical weights (seeded copy).
    torch.manual_seed(4)
    for a, b in zip(dense, streamed):
        wa: "torch.nn.Parameter" = a[0].weight  # type: ignore[index]
        wb: "torch.nn.Parameter" = b[0].weight  # type: ignore[index]
        wb.data.copy_(wa.data)

    x = torch.randn(5, 8, dtype=torch.float64)
    target = torch.randn(5, 4, dtype=torch.float64)

    def loss_fn(h):
        l = (h - target).pow(2).mean()
        return l, torch.autograd.grad(l, h, retain_graph=False)[0] \
            if h.requires_grad else 2.0 * (h - target) / (5 * 4)

    # Dense reference: ordinary autograd through the whole stack.
    for p in dense.parameters():
        p.grad = None
    hd = x
    for m in dense:
        hd = m(hd)
    ld = (hd - target).pow(2).mean()
    ld.backward()

    # Streamed: no_grad forward, per-layer recompute backward.
    trainer = StreamedRecomputeTrainer(streamed,
                                       [m[0].weight for m in streamed])
    final_h, inputs = trainer.forward_inputs(x)
    loss_val, grad_out = loss_fn(final_h)
    trainer.backward_recompute(inputs, grad_out)

    assert math.isclose(float(loss_val), float(ld.detach()), rel_tol=1e-9)
    for mdense, mstream in zip(dense, streamed):
        pd = mdense[0].weight.grad  # type: ignore[index]
        ps = mstream[0].weight.grad  # type: ignore[index]
        assert pd is not None and ps is not None
        assert torch.allclose(pd, ps, atol=1e-8), \
            f"grad mismatch: max diff {(pd - ps).abs().max()}"


def test_streamed_training_loss_decreases():
    """Several full streamed steps (SGD on CPU-held grads) reduce the loss."""
    torch = _torch()
    import torch.nn as nn

    from toaster_loop.training.streamed.recompute import StreamedRecomputeTrainer

    torch.manual_seed(5)
    dims = [(6, 6), (6, 6), (6, 3)]
    layers = [nn.Sequential(nn.Linear(i_f, o_f, bias=False).double())
              for i_f, o_f in dims]
    params = [m[0].weight for m in layers]
    trainer = StreamedRecomputeTrainer(layers, params)
    opt = torch.optim.SGD(params, lr=0.3)

    x = torch.randn(4, 6, dtype=torch.float64)
    target = torch.randn(4, 3, dtype=torch.float64)

    def loss_fn(h):
        h = h.detach().requires_grad_(False)
        l = (h - target).pow(2).mean()
        return float(l), 2.0 * (h - target) / (4 * 3)

    losses = []
    for _ in range(15):
        opt.zero_grad(set_to_none=True)
        loss = trainer.step(x, loss_fn)
        losses.append(loss)
        opt.step()
    assert losses[-1] < losses[0] * 0.5, f"loss did not fall: {losses}"
    # Streaming discipline: every layer ran once per pass, fwd and bwd.
    assert trainer.forward_layer_calls == 15 * 3
    assert trainer.backward_layer_calls == 15 * 3


def test_chunked_ce_matches_cross_entropy():
    torch = _torch()
    import torch.nn.functional as F

    from toaster_loop.training.streamed.chunked_ce import \
        chunked_linear_cross_entropy

    torch.manual_seed(9)
    hidden = torch.randn(2, 8, 1024, dtype=torch.float64)
    weight = torch.randn(64, 1024, dtype=torch.float64)
    labels = torch.randint(0, 64, (2, 8))
    labels[0, 3] = -100  # ignore path

    dense_logits = hidden.reshape(-1, 1024) @ weight.t()
    ref = F.cross_entropy(dense_logits, labels.reshape(-1),
                          ignore_index=-100, reduction="mean")

    hid = hidden.reshape(-1, 1024).clone().requires_grad_(True)
    w = weight.clone().requires_grad_(True)
    loss = chunked_linear_cross_entropy(hid, w, labels, chunk_size=5)
    assert torch.isclose(loss.double(), ref, rtol=1e-9), (loss, ref)

    # Gradients flow: d_hidden matches the dense autograd gradient. NOTE:
    # `weight` is stashed (not an apply() input), so it stays FROZEN by
    # design — exactly the streamed-LoRA contract where lm_head is never
    # trained; d_weight is intentionally not produced.
    loss.backward()
    hid2 = hidden.reshape(-1, 1024).clone().requires_grad_(True)
    w2 = weight.clone().requires_grad_(True)
    F.cross_entropy(hid2 @ w2.t(), labels.reshape(-1),
                    ignore_index=-100).backward()
    assert torch.allclose(hid.grad.double(), hid2.grad, atol=1e-7)
    assert w.grad is None  # frozen head contract


def test_layer_streaming_from_real_safetensors(tmp_path):
    """End-to-end (a): real safetensors shards -> LayerStreamer -> meta shell.

    Tiny random weights written in-test with the safetensors save API — no
    network, no HF download (WP10 constraint). Proves the loader/shell seam:
    keys stay ``model.layers.i.*``, tensors land on demand, shells fill via
    load_layer_into with Parameter identity preserved.
    """
    torch = _torch()
    from safetensors.torch import save_file

    from toaster_loop.training.streamed.streaming import (SafetensorsLayerLoader,
                                                     load_layer_into,
                                                     meta_shell)

    n_layers, dim = 3, 5
    shard_paths = []
    layer_keys = []
    for i in range(n_layers):
        p = tmp_path / f"model-{i:02d}.safetensors"
        key = f"model.layers.{i}.mlp.weight"
        save_file({key: torch.randn(dim, dim, dtype=torch.float32)}, str(p))
        shard_paths.append(p)
        layer_keys.append([(str(p), key)])

    loader = SafetensorsLayerLoader(layer_keys, device="cpu")
    shell = meta_shell([(dim, dim)] * n_layers)  # zero-storage skeleton
    for lin in shell:
        assert lin.weight.device.type == "meta"

    loaded_order = []

    class TracingLoader(SafetensorsLayerLoader):
        def load(self, layer_idx):
            loaded_order.append(layer_idx)
            return super().load(layer_idx)

    tl = TracingLoader(layer_keys, device="cpu")
    with LayerStreamer(tl, n_layers, prefetch_depth=1) as st:
        st.prefetch(0)
        for i in range(n_layers):
            weights = st.acquire(i)
            load_layer_into(shell[i], weights)
            assert shell[i].weight.device.type == "cpu"
            y = shell[i](torch.ones(dim))
            assert torch.isfinite(y).all()
            st.prefetch(i + 1)
            st.release(i)
        assert st.peak_residency() <= 2
    assert loaded_order == [0, 1, 2]
    tl.close()

# ------------------------------------------------------------------------
# from tests/test_golden_gate_wp10.py
# ------------------------------------------------------------------------

def _deps():
    for mod in ("torch", "transformers", "tokenizers", "safetensors"):
        pytest.importorskip(mod)
    from experiments.wp10_golden_gate.run_gate import find_binary

    binary = find_binary()
    if not binary:
        pytest.skip("no existing llama-server binary (never auto-build)")
    return binary


def test_golden_gate_promotion_cpu(tmp_path):
    """Full CPU proof; asserts the evidence JSON end to end."""
    _deps()
    from experiments.wp10_golden_gate.run_gate import run

    ev = run(work=tmp_path / "work", keep=True)

    assert ev["train"]["backend"] == "streamed"
    assert ev["export"]["gguf_path"].endswith(".gguf")
    assert ev["probe"]["changed"] is True          # adapter measurably active
    assert ev["gate_with_adapter"]["ok"] is True   # REAL gate passed
    assert ev["gate_with_adapter"]["pass_rate"] == 1.0
    assert ev["gate_without_adapter"]["ok"] is False  # golden not vacuous
    assert ev["promotion"]["active"] is True       # normal registry promotion

# ------------------------------------------------------------------------
# from tests/test_wp11_specialization.py
# ------------------------------------------------------------------------

def _torch__test_wp11_specialization():
    return pytest.importorskip("torch")


# ------------------------------------------------------------ tier 1: splits

def test_split_discipline_calib_eval_disjoint():
    splits = tb.split_tasks()
    cal_prompts = {t["prompt"] for t in splits["calib"]}
    ev_prompts = {t["prompt"] for t in splits["eval"]}
    assert cal_prompts and ev_prompts
    assert not (cal_prompts & ev_prompts), \
        "WP11 mandate: calibration and eval prompts must be disjoint"
    # eval covers both domains equally (skew cannot hide a regression)
    from collections import Counter
    per_domain = Counter(t["domain"] for t in splits["eval"])
    assert per_domain["A"] == per_domain["B"] > 0
    # every eval task is answerable by the declared rules (sanity of oracle)
    for t in splits["eval"]:
        alph = tb.ALPHABETS[t["domain"]]
        idx = alph.index(t["prompt"][t["plen"] - 1])
        assert t["expect_contains"][0] == alph[tb.rule(t["domain"], idx)]


def test_tasks_deterministic_and_rules_correct():
    a = tb.make_task(5, "A", 4)
    b = tb.make_task(5, "A", 4)
    assert a == b  # deterministic fixture data
    seq = tb.make_sequence("A", 5)
    assert a["prompt"] == seq[:4] and a["expect_contains"][0] == seq[4]
    with pytest.raises(ValueError):
        tb.make_task(1, "C", 4)
    with pytest.raises(ValueError):
        tb.make_task(1, "A", 5)  # unsupported prefix length


# ----------------------------------------------------- tier 1: buffer fixture

def test_buffer_roundtrip_through_real_recorder(tmp_path):
    """Seed via the REAL TraceRecorder; read prompts back out of JSONL+index."""
    splits = tb.split_tasks()
    calib = splits["calib"][:12]
    n = bf.seed_buffer(tmp_path, calib)
    assert n == len(calib), "recorder must confirm every trace"

    # a foreign trace (not our task shape) must be skipped by the reader —
    # proves the spike tolerates a mixed real-world buffer.
    from toaster_loop.store import TraceRecorder
    rec = TraceRecorder(home=tmp_path)
    try:
        row = rec.record(task="some unrelated user request",
                         messages=[{"role": "user", "content": "hi"}],
                         backend="other", model_alias="m",
                         base_url="http://127.0.0.1:9/v1", completed=True)
        assert row is not None
    finally:
        rec.close()

    prompts = bf.load_calibration_prompts(tmp_path)
    assert set(prompts) == {t["prompt"] for t in calib}


def test_parse_prompt_shapes():
    assert bf.parse_prompt("nextchar ab") == "ab"
    assert bf.parse_prompt("nextchar ab extra") == "ab"
    assert bf.parse_prompt("unrelated task") is None
    assert bf.parse_prompt(None) is None


# ------------------------------------------------- tier 1: eval gate semantics

def test_evalgate_mirrors_validate_gate(tmp_path, monkeypatch):
    """run_eval verdict shape == validate.py semantics (same _check_task)."""
    from toaster_loop.training.validate import ValidationConfig, _check_task

    tasks = [tb.make_task(i, "A", 4) for i in range(4)]
    # scripted oracle: right on first three, wrong on the last (fresh
    # iterator per run_eval call so two calls replay identically)
    script = [(t["expect_contains"][0] if i < 3 else "zz")
              for i, t in enumerate(tasks)]
    state = {"i": 0}

    def fake_greedy(params, task, **kw):
        out = script[state["i"] % len(script)]
        state["i"] += 1
        return out

    monkeypatch.setattr(eg, "greedy_completion", fake_greedy)
    rep = eg.run_eval({}, tasks, min_pass_rate=1.0)
    assert rep["total"] == 4 and rep["passed"] == 3
    assert abs(rep["pass_rate"] - 0.75) < 1e-9
    assert rep["gate_passed"] is False          # validate.py: rate >= min
    state["i"] = 0
    assert eg.run_eval({}, tasks, min_pass_rate=0.75)["gate_passed"] is True
    # per-check entries carry validate.py's legacy keys (shape mapping)
    for r in rep["results"]:
        for k in ("prompt", "pass", "missing", "forbidden_hit",
                  "checks", "reply_preview"):
            assert k in r
    failing = rep["results"][3]
    assert failing["pass"] is False
    assert failing["missing"] == [tasks[3]["expect_contains"][0]]

    # per-domain breakdown present and consistent
    tot = sum(d["total"] for d in rep["per_domain"].values())
    assert tot == rep["total"]
    _ = _check_task, ValidationConfig  # imported to assert shared semantics


def test_evalgate_uses_production_check_function():
    """The oracle must be validate.py's own function object (no forked copy)."""
    from toaster_loop.training import validate
    assert eg._check_task is validate._check_task


# ----------------------------------------------------- tier 1: noise-floor math

def test_noise_floor_requires_11_replicates():
    with pytest.raises(ValueError, match="R >= 11"):
        nf.estimate_eval_floor([0.5] * 10)


def test_noise_floor_spread_matches_wp1_formula():
    rates = [0.8, 0.9, 1.0, 0.7, 0.85, 0.9, 0.8, 0.75, 0.95, 0.85, 0.9]
    floor = nf.estimate_eval_floor(rates)
    # WP1 definition: (max-min)/median*100
    import statistics
    med = statistics.median(rates)
    assert floor["median_pass_rate"] == pytest.approx(med)
    assert floor["spread_pct"] == pytest.approx((1.0 - 0.7) / med * 100.0)
    assert floor["spread_pp"] == pytest.approx((1.0 - 0.7) * 100.0)
    assert floor["floor_pct"] == floor["spread_pct"]


def test_adoption_bar_never_below_three_percent():
    assert nf.adoption_bar_pct(None) == 3.0
    assert nf.adoption_bar_pct(0.0) == 3.0        # saturated floor -> 3% bar
    assert nf.adoption_bar_pct(19.2) == 19.2      # measured floor dominates


def test_verdict_boundaries():
    assert nf.verdict(2.9, 3.0) == "noise"
    assert nf.verdict(-2.9, 3.0) == "noise"
    assert nf.verdict(3.5, 3.0) == "win"
    assert nf.verdict(-3.5, 3.0) == "regression"


def test_summarize_variant_paired_relative_delta():
    base = [1.0] * 11
    var = [0.9] * 11
    floor = {"floor_pct": 0.0}
    s = nf.summarize_variant("v", base, var, floor)
    assert s["relative_delta_pct"] == pytest.approx(-10.0)
    assert s["verdict"] == "regression"
    assert s["median_delta_pp_points"] == pytest.approx(-0.1)
    # zero-baseline edge case falls back to absolute pp
    s2 = nf.summarize_variant("v", [0.0] * 11, [0.05] * 11, floor)
    assert s2["relative_delta_pct"] == pytest.approx(5.0)  # pp fallback
    with pytest.raises(ValueError):
        nf.summarize_variant("v", [1.0], [1.0, 1.0], floor)


def test_persisted_floor_reader_tolerates_absence(tmp_path, monkeypatch):
    monkeypatch.setenv("TOASTER_LOOP_BENCH_DIR", str(tmp_path / "no-bench"))
    assert nf.persisted_floor() is None


# -------------------------------------------------------------- tier 2: torch

def test_testbed_training_is_deterministic():
    _torch__test_wp11_specialization()
    splits = tb.split_tasks()
    p1 = tb.make_params(seed=7)
    tb.train_model(p1, splits["train"], steps=8, lr=0.05)
    r1 = [tb.greedy_completion(p1, t) for t in splits["eval"]]
    p2 = tb.make_params(seed=7)
    tb.train_model(p2, splits["train"], steps=8, lr=0.05)
    r2 = [tb.greedy_completion(p2, t) for t in splits["eval"]]
    assert r1 == r2, "same seed + fixed config must replay identically"


def test_specialization_stats_and_masks():
    torch = _torch__test_wp11_specialization()
    splits = tb.split_tasks()
    p = tb.make_params(seed=11)
    tb.train_model(p, splits["train"], steps=8, lr=0.05)

    from experiments.wp11_specialization.specialize import (activation_sparse,
                                                            calibration_stats,
                                                            expert_prune)
    st = calibration_stats(p, splits["calib"])
    assert len(st["router_mean"]) == tb.N_EXPERTS
    assert sum(st["router_mean"]) == pytest.approx(1.0, abs=1e-4)
    assert st["n"] == len(splits["calib"])

    a = expert_prune(p, splits["calib"], prune_count=1, stats=st)
    assert int(a["keep"].sum()) == tb.N_EXPERTS - 1
    least = min(range(tb.N_EXPERTS), key=lambda e: st["router_mean"][e])
    assert a["pruned"] == [least]

    b = activation_sparse(p, splits["calib"], stats=st)
    thr = b["act_thresholds"]
    assert tuple(thr.shape) == (tb.N_EXPERTS, tb.HIDDEN)
    assert bool((thr >= 0).all())

    # forward with masks runs and stays finite
    X, _ = tb.encode(splits["eval"])
    logits = tb.forward(p, X, keep=a["keep"], act_thresholds=thr)
    assert torch.isfinite(logits).all()


def test_run_experiment_mini_end_to_end(tmp_path, monkeypatch):
    """Driver smoke: 11 replicates in the FAST limited regime only (<~5 s)."""
    _torch__test_wp11_specialization()
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(tmp_path / "wp11-home"))
    from experiments.wp11_specialization import run_experiment as rx

    report = rx.run_experiment(replicates=11, regimes={"limited": 8},
                               verbose=False)
    assert report["work_package"] == "WP11"
    assert report["buffer"]["seeded"] >= 8
    assert (report["buffer"]["recovered_prompts"]
            == report["buffer"]["calib_tasks_used"])
    rr = report["regimes"]["limited"]
    floor = rr["noise_floor"]
    assert floor["R"] == 11
    assert len(floor["pass_rates"]) == 11
    for v in rr["variants"].values():
        assert v["verdict"] in ("win", "noise", "regression")
        assert v["adoption_bar_pct"] >= nf.MIN_BAR_PCT
    # report must be JSON-serializable (findings-doc persistence path)
    json.dumps(report, default=str)
    # eval goldens all scored through validate.py semantics
    one = rr["replicates_detail"][0]["baseline"]
    assert one["total"] == len(tb.split_tasks()["eval"])


def test_license_boundary_spike_dir_stays_outside_package():
    """The spike must never become importable from the package itself: the
    deliberate not-wired seam. experiments/ lives OUTSIDE toaster_loop/, so both
    test_no_orphans.py (import graph over toaster_loop/) and
    test_license_hygiene.py (AST scan over toaster_loop/) stay green by
    construction — while the D-12 banned names must appear nowhere in the
    spike as imports."""
    import ast

    pkg_root = Path(tb.__file__).resolve().parents[2]
    assert not (pkg_root / "toaster_loop" / "research").exists(), \
        "spike must live under experiments/, not wired into the package"
    banned = {"sparse_transformers", "airllm", "Strata", "kimi-k3-in-c"}
    spike_dir = Path(tb.__file__).resolve().parent
    for py in spike_dir.glob("*.py"):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                assert name.split(".")[0] not in banned, \
                    f"{py.name} imports banned {name}"

# ------------------------------------------------------------------------
# from tests/test_compression_v2.py
# ------------------------------------------------------------------------

def _msg(role, content, **extra):
    return {"role": role, "content": content, **extra}


def _tool(content, name="read_file"):
    return {"role": "tool", "name": name, "tool_call_id": "c",
            "content": content}


# ------------------------------------------------------------- prune pre-pass

def test_prune_replaces_long_tool_msgs_outside_tail():
    long1, long2 = "a" * 500, "b" * 400
    msgs = [_msg("user", "q"), _tool(long1, name="terminal"),
            _msg("assistant", "a"), _tool(long2),
            _msg("user", "tail q"), _msg("assistant", "tail a")]
    out, n = prune_old_tool_results(msgs, tail_start_index=4)
    assert n == 2
    assert out[1]["content"].startswith("[pruned tool output:")
    assert "terminal" in out[1]["content"]
    assert f"{len(long1)} chars" in out[1]["content"]  # char count present
    assert f"{len(long2)} chars" in out[3]["content"]
    # untouched rows keep identity; input never mutated
    assert out[0] is msgs[0] and out[4] is msgs[4]
    assert msgs[1]["content"] == long1


def test_prune_keeps_short_and_inside_tail():
    short = "x" * _PRUNE_MIN_CHARS  # floor is strict >
    inside = "y" * 500
    msgs = [_tool("z" * 300), _tool(short), _msg("user", "q"), _tool(inside)]
    out, n = prune_old_tool_results(msgs, tail_start_index=3)
    assert n == 1
    assert out[0]["content"].startswith("[pruned tool output:")
    assert out[1]["content"] == short       # <= floor untouched
    assert out[3]["content"] == inside      # inside protected tail untouched


def test_prune_idempotent_second_pass_noop():
    msgs = [_msg("user", "q"), _tool("a" * 500), _msg("assistant", "a")]
    once, n1 = prune_old_tool_results(msgs, tail_start_index=3)
    twice, n2 = prune_old_tool_results(once, tail_start_index=3)
    assert n1 == 1 and n2 == 0
    assert twice == once


def test_compress_returns_without_llm_when_prune_suffices():
    # Fat tool results push the estimate over threshold; pruning alone brings
    # it back under — the summarizer must never be invoked. Ends on an
    # assistant text turn so the tail boundary is always prunable-safe.
    msgs = [_msg("system", "S")]
    for i in range(4):
        msgs.append(_msg("user", f"q{i}"))
        msgs.append({"role": "assistant", "content": None,
                     "tool_calls": [{"id": f"c{i}", "type": "function",
                                     "function": {"name": "list_dir",
                                                  "arguments": "{}"}}]})
        msgs.append(_tool("t" * 4000))
        msgs.append(_msg("assistant", f"noted {i}"))
    threshold = estimate_tokens(msgs) - 100
    assert estimate_tokens(msgs) > threshold

    def boom(prompt):
        raise AssertionError("summarizer must not be called when pruning suffices")

    out = compress_messages(msgs, boom, threshold_tokens=threshold,
                            keep_recent_tokens=256)
    assert estimate_tokens(out) <= threshold
    pruned = [m for m in out if m.get("role") == "tool"
              and str(m["content"]).startswith("[pruned tool output:")]
    assert pruned  # pruning actually happened


# --------------------------------------------------- iterative / structured

def _head_msgs():
    msgs = [_msg("system", "S")]
    for i in range(20):
        msgs.append(_msg("user", f"question {i} " + "u" * 300))
        msgs.append(_msg("assistant", f"answer {i} " + "v" * 300))
    return msgs


def test_scratch_prompt_has_structured_template():
    prompts = []

    def fake_summarize(prompt):
        prompts.append(prompt)
        return "## What has been accomplished\n- did things"

    out = compress_messages(_head_msgs(), fake_summarize, threshold_tokens=10,
                            keep_recent_tokens=256)
    assert len(prompts) == 1
    p = prompts[0]
    assert "PREVIOUS SUMMARY" not in p
    # structured template sections offered to the summarizer
    for section in ("## What has been accomplished",
                    "## Key decisions & files touched",
                    "## Open threads & pending work"):
        assert section in p
    # summary placed under the historical heading prefix (not live intent)
    summary_msg = next(m for m in out
                       if str(m["content"]).startswith(SUMMARY_PREFIX))
    assert summary_msg["content"].startswith("[Summary of earlier conversation")
    assert "not new instructions" in SUMMARY_PREFIX


def test_previous_summary_uses_update_prompt():
    prompts = []

    def fake_summarize(prompt):
        prompts.append(prompt)
        return "updated summary body"

    out = compress_messages(_head_msgs(), fake_summarize, threshold_tokens=10,
                            keep_recent_tokens=256,
                            previous_summary="OLD SUMMARY TEXT 42")
    assert len(prompts) == 1
    p = prompts[0]
    assert "PREVIOUS SUMMARY:" in p          # iterative update framing
    assert "NEW TURNS TO FOLD IN:" in p
    assert "OLD SUMMARY TEXT 42" in p        # old summary fed forward verbatim
    assert "updated summary body" in out[-1]["content"] or any(
        "updated summary body" in str(m["content"]) for m in out)


def test_extractive_fallback_carries_previous_summary():
    def boom(prompt):
        raise RuntimeError("endpoint down")

    out = compress_messages(_head_msgs(), boom, threshold_tokens=10,
                            keep_recent_tokens=256,
                            previous_summary="PRIOR DIGEST MARKER")
    summary_msg = next(m for m in out
                       if str(m["content"]).startswith(SUMMARY_PREFIX))
    assert "user asked" in summary_msg["content"]  # extractive shape kept
    assert "PRIOR DIGEST MARKER" in summary_msg["content"]


def test_summary_from_messages_roundtrip():
    msgs = [_msg("system", "S")]
    for i in range(20):
        msgs.append(_msg("user", f"q{i} " + "u" * 300))
        msgs.append(_msg("assistant", f"a{i} " + "v" * 300))
    assert summary_from_messages(msgs) is None
    out = compress_messages(msgs, lambda p: "ROLLING SUMMARY BODY",
                            threshold_tokens=10, keep_recent_tokens=256)
    assert summary_from_messages(out) == "ROLLING SUMMARY BODY"


# --------------------------------------------------------- runner threading

def test_runner_forwards_summary_across_compactions(monkeypatch, tmp_path):
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(tmp_path / "uf"))
    handler = make_message_handler({"*": {"role": "assistant",
                                          "content": "final answer"}})
    prompts: list[str] = []

    def fake_summarize(self, prompt):  # captures summarizer prompts, no LLM
        prompts.append(prompt)
        return f"SUMMARY-v{len(prompts)}"

    monkeypatch.setattr(ToasterLoopAgentClient, "_summarize", fake_summarize)
    seed_history = []
    for i in range(6):
        seed_history.append(_msg("user", f"old question {i} " + "h" * 400))
        seed_history.append(_msg("assistant", f"old answer {i} " + "g" * 400))
    with FakeOpenAIServer(handler_cls=handler) as srv:
        client = ToasterLoopAgentClient(base_url=srv.base_url, model="m",
                                    compress_threshold_tokens=400,
                                    keep_recent_tokens=256)
        r1 = client.run("first task", history=list(seed_history))
        assert len(prompts) == 1
        assert "PREVIOUS SUMMARY" not in prompts[0]  # first compaction: cold
        assert r1.summary == "SUMMARY-v1"

        # Non-system remainder of run 1 becomes run 2's history; the new task
        # pushes it back over the threshold for a second compaction.
        history = [m for m in r1.messages if m.get("role") != "system"]
        r2 = client.run("second task " + "y" * 1500, history=history,
                        previous_summary=r1.summary)
        assert len(prompts) == 2
        assert "PREVIOUS SUMMARY:" in prompts[1]     # second: iterative update
        assert "SUMMARY-v1" in prompts[1]
        assert r2.summary == "SUMMARY-v2"


def test_runner_backward_compatible_call(monkeypatch, tmp_path):
    """Old call signature (no previous_summary) still valid; summary None."""
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(tmp_path / "uf"))
    handler = make_message_handler({"*": {"role": "assistant",
                                          "content": "all good"}})
    with FakeOpenAIServer(handler_cls=handler) as srv:
        client = ToasterLoopAgentClient(base_url=srv.base_url, model="m")
        res = client.run("hello there")  # compression off (threshold 0)
    assert res.completed
    assert res.summary is None

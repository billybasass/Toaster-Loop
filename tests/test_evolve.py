# Idea adapted from PIASystem (MIT) app/core/skill_evolution.py
"""Prompt-evolution tests (inspiration pass 4 #39).

Pure search logic runs with fully injected seams (no HTTP, no GPU); the job
path mirrors test_shadow.py's scheduler harness; apply/dismiss run against a
real config YAML + registry.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest
import yaml

from sable.evaluation.evolve import (
    APPLIED_KEY,
    DISMISSED_KEY,
    MUTATION_DIRECTIONS,
    RESULT_KEY,
    EvolveError,
    apply_prompt_to_config,
    length_cap_for,
    make_evolve_job_runner,
    run_evolution,
    validate_variant,
)
from sable.registry.registry import ModelRegistry
from sable.training.scheduler import Scheduler, TrainingConfig

BASE = "You are a careful assistant.\n\n## Tools\nUse tools wisely.\n"


def golden(prompt="say hi", contains="hi"):
    return {"prompt": prompt, "expect_contains": [contains]}


def replay(tid="r1", task="do the thing"):
    return {"task_id": tid, "task": task}


# ------------------------------------------------------------ constraint rules

def test_length_cap_formula():
    assert length_cap_for("x" * 100) == max(150, 900)  # base+800 wins small
    assert length_cap_for("x" * 2000) == 3000          # 1.5x wins large


def test_validate_rejects_over_length():
    cap = length_cap_for(BASE)
    ok, reason = validate_variant("y" * (cap + 1), base_text=BASE, cap=cap,
                                  must_keep_sections=[], seen={})
    assert not ok and "length cap" in (reason or "")


def test_validate_rejects_missing_must_keep_header():
    mutated = BASE.replace("## Tools", "## Instruments")
    ok, reason = validate_variant(mutated, base_text=BASE,
                                  cap=length_cap_for(BASE),
                                  must_keep_sections=["## Tools"], seen={})
    assert not ok and "must-keep" in (reason or "")


def test_validate_rejects_near_duplicate():
    mutated = BASE + "\n\n  \n"  # whitespace-only change -> ratio ~1.0
    ok, reason = validate_variant(mutated, base_text=BASE,
                                  cap=length_cap_for(BASE),
                                  must_keep_sections=[], seen={})
    assert not ok and "near-duplicate" in (reason or "")


def test_validate_rejects_duplicate_of_seen_variant():
    v1 = BASE + "\nAdd step ordering: 1) plan 2) act 3) verify carefully.\n"
    # near-identical to the recorded variant text (one char differs):
    twin = v1.replace("verify", "Verify")
    ok, reason = validate_variant(twin, base_text=BASE,
                                  cap=length_cap_for(BASE),
                                  must_keep_sections=[],
                                  seen={_norm(v1): "v7"})
    assert not ok and "v7" in (reason or "")


def test_validate_rejects_empty_accepts_clean():
    ok, reason = validate_variant("   ", base_text=BASE,
                                  cap=length_cap_for(BASE),
                                  must_keep_sections=[], seen={})
    assert not ok and "empty" in (reason or "")
    clean = BASE + "\nAlways verify results before reporting success.\n"
    ok, reason = validate_variant(clean, base_text=BASE,
                                  cap=length_cap_for(BASE),
                                  must_keep_sections=["## Tools"], seen={})
    assert ok and reason is None


def _norm(t):
    import re

    return re.sub(r"\s+", " ", t).strip()


# ----------------------------------------------------------------- search loop

class Stub:
    """Scripted world: mutate_fn pops variants; run/judge fns count calls."""

    def __init__(self, variants=None, golden_pass=("*",), judge_scores=None):
        self.variants = list(variants or [])
        self.directions_seen = []
        self.run_calls = []       # (prompt_head, task_prompt)
        self.judge_calls = 0
        self.golden_pass = golden_pass  # prompts whose replies pass checks
        self.judge_scores = judge_scores or {}

    def mutate(self, direction, prompt, cap, must_keep=None):
        self.directions_seen.append(direction)
        if not self.variants:
            return None
        head, text = self.variants.pop(0)
        # tag so golden/judge stubs can key off the variant identity
        return text + f"\n<!--{head}-->\n"

    def run_task(self, system_prompt, task):
        self.run_calls.append((system_prompt, task.get("prompt", "")))
        # A reply passes the substring golden check when the prompt carries a
        # marker from `golden_pass` (variant identity tags).
        passes = any(pat in system_prompt for pat in self.golden_pass)
        return {"content": "hi" if passes else "wrong answer"}

    def judge(self, system_prompt, rt, message):
        self.judge_calls += 1
        for key, score in self.judge_scores.items():
            if key in system_prompt:
                return score
        return 0.5


def test_search_golden_first_elimination_no_judge_calls_for_hard_fails():
    """A variant failing hard golden checks never reaches the judge."""
    stub = Stub(
        variants=[("bad", "Assistant.\n\n## Tools\nUse tools.\nbad")],
        golden_pass=("<!--good-->",),
    )
    result = run_evolution(
        BASE, [golden()], [replay()], k=1, generations=1, call_budget=50,
        mutate_fn=stub.mutate, run_task_fn=stub.run_task,
        judge_fn=stub.judge)
    bad = [e for e in result["lineage"] if e.get("direction")]
    assert bad and bad[0]["golden_pass"] is False
    # hard-failed variant: zero judge calls total (baseline had no replay pass
    # either because baseline itself fails the stubbed golden).
    assert stub.judge_calls == 0


def test_search_top2_selection_and_tie_prefers_shorter():
    v_short = ("You are an assistant. Plan first, then act; verify every "
               "result before reporting success. <!--s-->")
    v_long = v_short + "\n" + ("Additional padding sentence for length. " * 8) \
        .replace("<!--s-->", "") + "<!--l-->"
    stub = Stub(variants=[("s", v_short), ("l", v_long)],
                golden_pass=("<!--",))
    # both pass goldens; equal judge scores -> shorter must rank first
    result = run_evolution(
        BASE, [golden()], [replay()], k=2, generations=1, call_budget=60,
        mutate_fn=stub.mutate, run_task_fn=stub.run_task,
        judge_fn=lambda sp, rt, m: 0.5)
    scored = [e for e in result["lineage"] if e.get("golden_pass")]
    best = result["best"]
    assert best is not None
    assert "<!--s-->" in best["text"]  # tie -> shorter variant wins
    assert len(scored) >= 2


def test_search_lineage_and_direction_rotation():
    stub = Stub(variants=[(str(i), f"You are assistant number {i}. First make "
                                  f"a plan; then execute it step by step and "
                                  f"verify. unique{i} <!--v{i}-->")
                          for i in range(6)],
                golden_pass=("<!--",))
    result = run_evolution(
        BASE, [golden()], [], k=3, generations=1, call_budget=60,
        mutate_fn=stub.mutate, run_task_fn=stub.run_task,
        judge_fn=lambda sp, rt, m: 0.5)
    dirs = [e.get("direction") for e in result["lineage"]
            if e.get("direction")]
    assert dirs[:3] == list(MUTATION_DIRECTIONS[:3])
    ids = {e["variant_id"] for e in result["lineage"]}
    assert "v0" in ids and all(e["parent_id"] in ids
                               for e in result["lineage"])
    # every lineage entry carries the full shape
    for e in result["lineage"]:
        assert {"variant_id", "parent_id", "direction", "golden_pass",
                "judge_mean", "n_scored"} <= set(e)


def test_search_call_budget_stops_honestly_with_partial():
    stub = Stub(variants=[(str(i), f"You are assistant {i}. Plan, act, "
                                  f"verify; never skip verification. "
                                  f"<!--x{i}-->") for i in range(10)],
                golden_pass=("<!--",))
    result = run_evolution(
        BASE, [golden(), golden("again", "again")], [replay()],
        k=3, generations=2, call_budget=3,
        mutate_fn=stub.mutate, run_task_fn=stub.run_task,
        judge_fn=lambda sp, rt, m: 0.5)
    assert result["partial"] is True
    assert result["calls_used"] <= result["call_budget"] == 3


def test_search_no_goldens_raises_honestly():
    with pytest.raises(EvolveError, match="no golden tasks"):
        run_evolution(BASE, [], [replay()], mutate_fn=lambda **k: None,
                      run_task_fn=lambda *a: None,
                      judge_fn=lambda *a: None)


def test_search_improved_flag_and_diff():
    stub = Stub(variants=[("good", "You are an improved assistant. Always "
                                   "plan before acting and verify results "
                                   "against the request. <!--good-->")],
                golden_pass=("<!--",),
                judge_scores={"<!--good-->": 0.9})
    result = run_evolution(
        BASE, [golden()], [replay()], k=1, generations=1, call_budget=30,
        mutate_fn=stub.mutate, run_task_fn=stub.run_task,
        judge_fn=lambda sp, rt, m: 0.9 if "<!--good-->" in sp else 0.1)
    assert result["improved"] is True
    assert "--- current_prompt" in result["diff"]
    assert result["best"]["variant_id"] != "v0"


# ------------------------------------------------------------------ job kind

@contextmanager
def _endpoint():
    yield "http://127.0.0.1:9", "stub-model"


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / ".sable-test-home"
    monkeypatch.setenv("SABLE_CONFIG", str(h / "sable.yaml"))
    (h / "buffer").mkdir(parents=True)
    return h


def _job_params(**over):
    p = {"profile": "coder", "tasks": [golden()], "k": 1, "generations": 1,
         "call_budget": 10}
    p.update(over)
    return p


def test_evolve_job_runs_via_scheduler_and_stores_result(home, monkeypatch):
    monkeypatch.setattr("sable.training.scheduler.gpu_free_mib",
                        lambda: None)
    reg = ModelRegistry(home / "buffer" / "sable.db")
    stub = Stub(variants=[("good", "Assistant.\n\n## Tools\ntools <!--good-->")],
                golden_pass=("<!--",))

    def mutate(direction, prompt, cap, must_keep=None):
        return stub.mutate(direction, prompt, cap, must_keep)

    runner = make_evolve_job_runner(
        reg, home=home, defaults={"enabled": True, "max_call_budget": 60},
        endpoint_for=_endpoint,
        # production run/judge paths need HTTP; inject via mutate + monkeypatch
        )
    # Replace the heavy execution with the pure search over stubs:
    import sable.evaluation.evolve as ev

    monkeypatch.setattr(
        ev, "run_prompt_evolution",
        lambda base, goldens, replays, **kw: run_evolution(
            base, goldens, replays, k=kw.get("k", 1),
            generations=kw.get("generations", 1),
            call_budget=kw.get("call_budget", 10),
            mutate_fn=stub.mutate, run_task_fn=stub.run_task,
            judge_fn=lambda sp, rt, m: 0.8))

    sched = Scheduler(registry=reg, cfg=TrainingConfig(),
                      inprocess_runners={"evolve": runner})
    job = sched.submit("evolve", params=_job_params(base_prompt=BASE),
                       enqueue_actor="cli")
    done = sched.run_pending()
    assert [j.state for j in done] == ["done"]
    row = reg.get_job(job.job_id)
    assert row is not None
    res = row.params.get(RESULT_KEY)
    assert isinstance(res, dict)
    assert {"lineage", "best", "diff", "improved", "calls_used",
            "call_budget", "partial"} <= set(res)
    # NO registry mutation of the profile: proposal only.
    reg.close()


def test_evolve_job_refuses_when_disabled(home, monkeypatch):
    monkeypatch.setattr("sable.training.scheduler.gpu_free_mib",
                        lambda: None)
    reg = ModelRegistry(home / "buffer" / "sable.db")
    runner = make_evolve_job_runner(
        reg, home=home, defaults={"enabled": False},
        endpoint_for=_endpoint)
    sched = Scheduler(registry=reg, cfg=TrainingConfig(),
                      inprocess_runners={"evolve": runner})
    job = sched.submit("evolve", params=_job_params(base_prompt=BASE))
    done = sched.run_pending()
    assert done[0].state == "error"
    assert "evolution.enabled" in (done[0].error or "")
    reg.close()


def test_evolve_job_honest_empty_no_goldens(home, monkeypatch):
    monkeypatch.setattr("sable.training.scheduler.gpu_free_mib",
                        lambda: None)
    reg = ModelRegistry(home / "buffer" / "sable.db")
    runner = make_evolve_job_runner(
        reg, home=home, defaults={"enabled": True}, endpoint_for=_endpoint)
    sched = Scheduler(registry=reg, cfg=TrainingConfig(),
                      inprocess_runners={"evolve": runner})
    p = _job_params(base_prompt=BASE)
    p.pop("tasks")
    job = sched.submit("evolve", params=p)
    done = sched.run_pending()
    assert done[0].state == "error"
    assert "golden" in (done[0].error or "").lower()
    reg.close()


def test_evolve_job_kind_accepted_by_registry_and_migration(home):
    reg = ModelRegistry(home / "buffer" / "sable.db")
    job = reg.enqueue_job("evolve", params={"profile": "coder"})
    assert job.state == "queued"
    reg.close()
    # reopen: CHECK already contains 'evolve' -> migration is a no-op
    reg2 = ModelRegistry(home / "buffer" / "sable.db")
    assert reg2.get_job(job.job_id) is not None
    reg2.close()


# ----------------------------------------------------------------- apply flow

def _write_cfg(path: Path, prompt="old prompt text"):
    path.write_text(yaml.safe_dump({
        "profiles": [{"name": "coder", "system_prompt": prompt,
                      "tools": ["read_file"]}],
        "default_profile": "coder",
    }, sort_keys=False))


def test_apply_updates_profile_and_keeps_backup(home):
    cfg = home / "sable.yaml"
    _write_cfg(cfg)
    out = apply_prompt_to_config(cfg, "coder", "new evolved prompt")
    data = yaml.safe_load(cfg.read_text())
    prof = data["profiles"][0]
    assert prof["system_prompt"] == "new evolved prompt"
    assert prof["system_prompt_prev"] == "old prompt text"
    assert prof["prompt_evolved_at"]
    assert Path(out["backup"]).is_file()


def test_apply_missing_profile_raises(home):
    cfg = home / "sable.yaml"
    _write_cfg(cfg)
    with pytest.raises(EvolveError, match="no profile"):
        apply_prompt_to_config(cfg, "ghost", "x")


def test_dismiss_and_applied_flags_persist_without_deleting_history(home):
    reg = ModelRegistry(home / "buffer" / "sable.db")
    job = reg.enqueue_job("evolve", params={"profile": "coder"})
    reg.finish_job(job.job_id, "done")
    reg.set_job_param(job.job_id, RESULT_KEY, {"improved": True})
    reg.set_job_param(job.job_id, DISMISSED_KEY, "2026-01-01T00:00:00+00:00")
    row = reg.get_job(job.job_id)
    assert row is not None
    assert row.params[DISMISSED_KEY].startswith("2026")
    assert row.params[RESULT_KEY] == {"improved": True}  # history intact
    reg.set_job_param(job.job_id, APPLIED_KEY, "2026-01-02T00:00:00+00:00")
    row = reg.get_job(job.job_id)
    assert row is not None
    assert row.params[APPLIED_KEY] and row.params[RESULT_KEY]
    reg.close()


def test_plain_run_never_touches_profile_config(home, monkeypatch):
    """Guardrail: an evolve job NEVER mutates the config file."""
    monkeypatch.setattr("sable.training.scheduler.gpu_free_mib",
                        lambda: None)
    cfg = home / "sable.yaml"
    _write_cfg(cfg)
    before = cfg.read_text()

    reg = ModelRegistry(home / "buffer" / "sable.db")
    stub = Stub(variants=[("good", "Assistant.\n\n## Tools\ntools <!--g-->")],
                golden_pass=("<!--",))
    import sable.evaluation.evolve as ev

    monkeypatch.setattr(
        ev, "run_prompt_evolution",
        lambda base, goldens, replays, **kw: run_evolution(
            base, goldens, replays, k=1, generations=1, call_budget=10,
            mutate_fn=stub.mutate, run_task_fn=stub.run_task,
            judge_fn=lambda sp, rt, m: 0.9))
    runner = make_evolve_job_runner(
        reg, home=home, defaults={"enabled": True}, endpoint_for=_endpoint)
    sched = Scheduler(registry=reg, cfg=TrainingConfig(),
                      inprocess_runners={"evolve": runner})
    sched.submit("evolve", params=_job_params(base_prompt=BASE))
    done = sched.run_pending()
    assert done[0].state == "done"
    assert cfg.read_text() == before  # proposal only — config untouched
    reg.close()


# ------------------------------------------------------------- dashboard API

def _dash_app(home):
    from sable.core.config import SableConfig
    from sable.dashboard.app import DashboardApp

    return DashboardApp(SableConfig(), home=home)


def test_dashboard_evolve_endpoints(home, monkeypatch):
    cfg = home / "sable.yaml"
    _write_cfg(cfg)
    reg = ModelRegistry(home / "buffer" / "sable.db")
    job = reg.enqueue_job("evolve", params={"profile": "coder"})
    reg.set_job_param(job.job_id, RESULT_KEY, {
        "ts": "2026-01-01T00:00:00+00:00", "improved": True,
        "diff": "--- a\n+++ b\n+new", "calls_used": 5, "call_budget": 60,
        "partial": False,
        "baseline": {"variant_id": "v0", "golden_pass": True,
                     "judge_mean": 0.4},
        "best": {"variant_id": "v1", "direction": "tighten instructions",
                 "golden_pass": True, "judge_mean": 0.8,
                 "text": "brand new prompt"},
    })
    reg.finish_job(job.job_id, "done")
    reg.close()

    app = _dash_app(home)
    r = app.handle("GET", "/api/evolve", query={"profile": "coder"})
    assert r.status == 200
    prop = r.body["proposal"]
    assert prop["job_id"] == job.job_id and prop["improved"]
    assert prop["fitness_current"] == 0.4
    assert prop["fitness_proposed"] == 0.8

    # apply is gated: explicit POST -> profile prompt replaced + backup
    r = app.handle("POST", "/api/evolve/apply", body={"job_id": job.job_id})
    assert r.status == 200 and r.body["ok"]
    data = yaml.safe_load(cfg.read_text())
    assert data["profiles"][0]["system_prompt"] == "brand new prompt"
    assert data["profiles"][0]["system_prompt_prev"] == "old prompt text"

    # second apply refused (409), history untouched
    r = app.handle("POST", "/api/evolve/apply", body={"job_id": job.job_id})
    assert r.status == 409

    # dismiss flag on another proposal
    job2 = reg2 = None
    reg2 = ModelRegistry(home / "buffer" / "sable.db")
    j2 = reg2.enqueue_job("evolve", params={"profile": "coder"})
    reg2.set_job_param(j2.job_id, RESULT_KEY, {"improved": False})
    reg2.finish_job(j2.job_id, "done")
    reg2.close()
    r = app.handle("POST", "/api/evolve/dismiss", body={"job_id": j2.job_id})
    assert r.status == 200 and r.body["dismissed"] == j2.job_id

    assert app.handle("GET", "/api/evolve").status == 400
    assert app.handle("POST", "/api/evolve/dismiss",
                      body={"job_id": "nope"}).status == 404


# ----------------------------------------------------------------- CLI smoke

def test_cli_evolve_refuses_when_disabled(home, monkeypatch, capsys):
    from sable.cli import main

    cfg = home / "sable.yaml"
    _write_cfg(cfg)
    monkeypatch.setenv("SABLE_CONFIG", str(cfg))
    rc = main(["evolve", "coder", "--goldens", "x.json"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "evolution.enabled" in err


def test_cli_evolve_apply_updates_config(home, monkeypatch, capsys):
    from sable.cli import main

    cfg = home / "sable.yaml"
    _write_cfg(cfg)
    monkeypatch.setenv("SABLE_CONFIG", str(cfg))
    reg = ModelRegistry(home / "buffer" / "sable.db")
    job = reg.enqueue_job("evolve", params={"profile": "coder"})
    reg.set_job_param(job.job_id, RESULT_KEY, {
        "improved": True, "best": {"variant_id": "v1",
                                   "text": "cli evolved prompt"}})
    reg.finish_job(job.job_id, "done")
    reg.close()

    rc = main(["evolve", "apply", job.job_id])
    assert rc == 0
    data = yaml.safe_load(cfg.read_text())
    assert data["profiles"][0]["system_prompt"] == "cli evolved prompt"
    assert data["profiles"][0]["system_prompt_prev"] == "old prompt text"

    # applying twice refuses (already applied) — gated, single-shot
    rc = main(["evolve", "apply", job.job_id])
    assert rc == 1
    assert "already applied" in capsys.readouterr().err


def test_cli_evolve_dismiss_flags_job(home, monkeypatch):
    from sable.cli import main

    cfg = home / "sable.yaml"
    _write_cfg(cfg)
    monkeypatch.setenv("SABLE_CONFIG", str(cfg))
    reg = ModelRegistry(home / "buffer" / "sable.db")
    job = reg.enqueue_job("evolve", params={"profile": "coder"})
    reg.set_job_param(job.job_id, RESULT_KEY, {"improved": False})
    reg.finish_job(job.job_id, "done")
    reg.close()

    assert main(["evolve", "dismiss", job.job_id]) == 0
    reg2 = ModelRegistry(home / "buffer" / "sable.db")
    row = reg2.get_job(job.job_id)
    assert row is not None
    assert row.params[DISMISSED_KEY] and row.params[RESULT_KEY]
    reg2.close()

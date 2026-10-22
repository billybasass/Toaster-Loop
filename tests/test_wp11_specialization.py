"""WP11 research-spike tests (experiments/wp11_specialization/).

TIER 1 (no torch): split discipline, buffer round-trip through the REAL
recorder, eval-gate semantics vs validate.py, noise-floor math.
TIER 2 (pytest.importorskip("torch")): testbed determinism, specialization
statistic shapes, mini end-to-end driver run (limited regime only — fast).

All hermetic: buffer lands in tmp SABLE_HOME; nothing touches ~/.sable;
no network. Findings doc: docs/SPECIALIZATION-WP11.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.wp11_specialization import buffer_fixture as bf  # noqa: E402
from experiments.wp11_specialization import evalgate as eg  # noqa: E402
from experiments.wp11_specialization import noise_floor as nf  # noqa: E402
from experiments.wp11_specialization import testbed as tb  # noqa: E402


def _torch():
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
    from sable.store import TraceRecorder
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
    from sable.training.validate import ValidationConfig, _check_task

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
    from sable.training import validate
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
    monkeypatch.setenv("SABLE_BENCH_DIR", str(tmp_path / "no-bench"))
    assert nf.persisted_floor() is None


# -------------------------------------------------------------- tier 2: torch

def test_testbed_training_is_deterministic():
    _torch()
    splits = tb.split_tasks()
    p1 = tb.make_params(seed=7)
    tb.train_model(p1, splits["train"], steps=8, lr=0.05)
    r1 = [tb.greedy_completion(p1, t) for t in splits["eval"]]
    p2 = tb.make_params(seed=7)
    tb.train_model(p2, splits["train"], steps=8, lr=0.05)
    r2 = [tb.greedy_completion(p2, t) for t in splits["eval"]]
    assert r1 == r2, "same seed + fixed config must replay identically"


def test_specialization_stats_and_masks():
    torch = _torch()
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
    _torch()
    monkeypatch.setenv("SABLE_HOME", str(tmp_path / "wp11-home"))
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
    deliberate not-wired seam. experiments/ lives OUTSIDE sable/, so both
    test_no_orphans.py (import graph over sable/) and
    test_license_hygiene.py (AST scan over sable/) stay green by
    construction — while the D-12 banned names must appear nowhere in the
    spike as imports."""
    import ast

    pkg_root = Path(tb.__file__).resolve().parents[2]
    assert not (pkg_root / "sable" / "research").exists(), \
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

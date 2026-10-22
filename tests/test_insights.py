"""Backlog item #11 tests — insights engine over the buffer DB.

Synthetic traces are seeded through the real TraceRecorder (with a frozen
datetime so timestamps vary deterministically), then each report section is
asserted against hand-computed numbers. Also covers: empty-DB zero structure,
--days/--profile filters, promotion deltas via the real registry, CLI --json
round-trip and the GET /api/insights endpoint.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sable.store.recorder as recorder_mod  # noqa: E402
from sable.core.config import SableConfig  # noqa: E402
from sable.dashboard.app import DashboardApp  # noqa: E402
from sable.registry import ModelRegistry  # noqa: E402
from sable.store import TraceRecorder  # noqa: E402
from sable.store.insights import generate_insights  # noqa: E402


# --------------------------------------------------------------------- seeds

class _FrozenDatetime(datetime):
    """datetime.now() shim so the recorder writes controlled timestamps."""
    frozen: datetime = datetime(2026, 9, 1, tzinfo=timezone.utc)

    @classmethod
    def now(cls, tz=None):
        return cls.frozen if tz is None else cls.frozen.astimezone(tz)


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "uf-insights"
    monkeypatch.setenv("SABLE_HOME", str(h))
    monkeypatch.setattr(recorder_mod, "datetime", _FrozenDatetime)
    return h


def _tool_messages(name: str, fail: bool) -> list:
    """One assistant tool_call + its tool result (pass or fail)."""
    content = "error: boom" if fail else "ok output"
    return [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": f"call_{name}", "type": "function",
             "function": {"name": name, "arguments": "{}"}}]},
        {"role": "tool", "name": name, "tool_call_id": f"call_{name}",
         "content": content},
    ]


def seed(home, *, when: datetime, profile="sable", model_alias="base-model",
         model_version=None, completed=True, fail_tool=False, tool=None,
         tokens_in=None, tokens_out=None, duration_s=None,
         failure_reason=None, conf_mean=None, conf_p90_negentropy=None):
    """Record one synthetic trace at a frozen timestamp."""
    _FrozenDatetime.frozen = when
    msgs = [{"role": "user", "content": "do the thing"}]
    if tool:
        msgs += _tool_messages(tool, fail_tool)
    msgs.append({"role": "assistant", "content": "done"})
    rec = TraceRecorder(home=home)
    try:
        row = rec.record(
            task="do the thing", messages=msgs, backend=profile,
            model_alias=model_alias, base_url="http://127.0.0.1:9/v1",
            completed=completed, tokens_in=tokens_in, tokens_out=tokens_out,
            duration_s=duration_s, model_version=model_version,
            failure_reason=failure_reason,
            conf_mean=conf_mean, conf_p90_negentropy=conf_p90_negentropy,
        )
    finally:
        rec.close()
    assert row is not None
    return row


def now():
    return datetime.now(timezone.utc)


# ------------------------------------------------------------- zero structure

def test_empty_db_returns_well_formed_zero_structure(home):
    r = generate_insights(home, days=30)
    assert r["empty"] is True
    assert r["overview"]["traces"] == 0
    assert r["overview"]["avg_score"] is None
    assert r["score_trend"]["overall"]["points"] == []
    assert r["score_trend"]["overall"]["half_mean_delta"] is None
    assert r["tool_reliability"]["tools"] == []
    assert r["failure_reasons"]["total"] == 0
    assert r["latency"]["duration_s"]["p50"] is None
    assert r["activity"]["traces_per_day"] == []
    assert r["promotions"]["available"] is False


# ------------------------------------------------------------- overview/trend

def test_overview_and_score_trend_exact(home):
    # day 1: scores 0.5, 0.25 (avg 0.375) · day 2: score 0.5 -> trend up
    seed(home, when=now() - timedelta(days=2), completed=True)          # 0.5
    seed(home, when=now() - timedelta(days=2), completed=False)         # -0.5
    seed(home, when=now() - timedelta(days=1), completed=True)          # 0.5

    r = generate_insights(home, days=30)
    ov = r["overview"]
    assert ov["traces"] == 3
    assert ov["avg_score"] == pytest.approx((0.5 - 0.5 + 0.5) / 3, abs=1e-4)
    assert ov["status_counts"]["failure"] == 1
    assert ov["status_counts"]["success"] == 2

    pts = r["score_trend"]["overall"]["points"]
    assert [p["traces"] for p in pts] == [2, 1]
    assert pts[0]["avg_score"] == pytest.approx(0.0)   # (0.5 + -0.5)/2
    assert pts[1]["avg_score"] == pytest.approx(0.5)
    # two scored buckets: second-half mean (0.5) - first-half mean (0.0)
    assert r["score_trend"]["overall"]["half_mean_delta"] == pytest.approx(0.5)
    assert r["score_trend"]["overall"]["direction"] == "up"

    prof = r["score_trend"]["by_profile"]
    assert len(prof) == 1 and prof[0]["traces"] == 3


def test_score_trend_grouped_by_model(home):
    seed(home, when=now() - timedelta(days=1), model_alias="v1")
    seed(home, when=now() - timedelta(days=1), model_alias="v2",
         model_version="v2")
    r = generate_insights(home)
    names = {g["name"] for g in r["score_trend"]["by_model"]}
    assert names == {"v1", "v2"}  # model_version wins over alias


# ------------------------------------------------------------ tool reliability

def test_tool_failure_rates_from_jsonl_scan(home):
    seed(home, when=now() - timedelta(days=1), tool="read_file")
    seed(home, when=now() - timedelta(days=1), tool="read_file", fail_tool=True)
    seed(home, when=now() - timedelta(days=1), tool="shell", fail_tool=True)

    tr = generate_insights(home)["tool_reliability"]
    assert tr["available"] is True and tr["traces_scanned"] == 3
    by_name = {t["tool"]: t for t in tr["tools"]}
    assert by_name["read_file"]["calls"] == 2
    assert by_name["read_file"]["failures"] == 1
    assert by_name["read_file"]["failure_rate"] == pytest.approx(0.5)
    assert by_name["shell"]["calls"] == 1
    assert by_name["shell"]["failure_rate"] == pytest.approx(1.0)


def test_tool_reliability_tolerates_pruned_transcripts(home):
    row = seed(home, when=now() - timedelta(days=1), tool="shell")
    Path(row.jsonl_path).unlink()  # transcript pruned -> skipped silently
    tr = generate_insights(home)["tool_reliability"]
    assert tr["traces_scanned"] == 0
    assert tr["tools"] == []


# ----------------------------------------------------------- failure reasons

def test_failure_reason_histogram_splits_infra_model(home):
    seed(home, when=now() - timedelta(days=1), completed=False,
         failure_reason="backend_error:timeout")     # infra
    seed(home, when=now() - timedelta(days=1), completed=False,
         failure_reason="max_iterations")            # model
    seed(home, when=now() - timedelta(days=1), completed=False,
         failure_reason="max_iterations")            # model
    seed(home, when=now() - timedelta(days=1))       # no reason

    fr = generate_insights(home)["failure_reasons"]
    assert fr["total"] == 3
    assert fr["infra_total"] == 1 and fr["model_total"] == 2
    top = fr["by_reason"][0]
    assert top["reason"] == "max_iterations" and top["count"] == 2
    assert top["kind"] == "model"
    infra = next(b for b in fr["infra"] if b["reason"] == "backend_error:timeout")
    assert infra["count"] == 1


# ---------------------------------------------------------------- percentiles

def test_duration_token_percentiles(home):
    for d, t in ((1.0, 100), (2.0, 200), (3.0, 300), (4.0, 400)):
        seed(home, when=now() - timedelta(days=1), duration_s=d,
             tokens_in=t // 2, tokens_out=t // 2)
    lat = generate_insights(home)["latency"]
    ds, tk = lat["duration_s"], lat["tokens_total"]
    assert ds["count"] == 4 and ds["max"] == 4.0
    # nearest rank: p50 of [1,2,3,4] -> index 1 (2.0); p90 -> index 3 (4.0)
    assert ds["p50"] == 2.0 and ds["p90"] == 4.0
    assert tk["count"] == 4 and tk["p50"] == 200.0 and tk["max"] == 400.0


# ------------------------------------------------- percentile parity (#16)
#
# insights.percentile and agent_runner._confidence_aggregate both claim
# nearest-rank semantics; the old banker's-round() index disagreed with
# ceil(p*n) at exact .5 boundaries (e.g. p50 @ n=15, p90 @ n=5/16/25/36,
# p95 @ n=11..19/30..39). Pin the canonical rank across n=1..40 on random
# and adversarial samples so the two can never silently diverge again.

def _rank_samples(n: int) -> list:
    import random

    rng = random.Random(1234 + n)
    return [
        sorted(rng.uniform(-10, 10) for _ in range(n)),      # random
        [7.0] * n,                                           # all-equal
        sorted(rng.choice([1.0, 1.0, 2.0, 3.0]) for _ in range(n)),  # dup-heavy
        [float(i) for i in range(n)],                        # strictly increasing
    ]


@pytest.mark.parametrize("pct", [50, 90, 95])
@pytest.mark.parametrize("n", list(range(1, 41)))
def test_percentile_matches_nearest_rank_ceiling(pct, n):
    from sable.store.insights import percentile

    for vals in _rank_samples(n):
        expected = vals[math.ceil(pct * n / 100) - 1]   # canonical nearest rank
        got = percentile(vals, pct)
        assert got == pytest.approx(round(expected, 4)), (pct, n, vals[:5])


@pytest.mark.parametrize("n", list(range(1, 41)))
def test_confidence_aggregate_p90_matches_insights_percentile(n):
    """agent_runner._confidence_aggregate's p90 picks the SAME sorted element
    as insights.percentile(..., 90) (its docstring promises this parity).
    Compared via exp() over logprob-domain (<= 0) samples so rounding stays
    bounded; the previous test pins percentile itself to ceil-rank."""
    import random

    from sable.core.agent_runner import _confidence_aggregate

    rng = random.Random(99 + n)
    for vals in (sorted(-rng.random() * 20 for _ in range(n)),   # random lp
                 [math.log(0.5)] * n,                            # all-equal
                 sorted(rng.choice([math.log(0.1), math.log(0.1),
                                    math.log(0.9)]) for _ in range(n))):
        _, p90 = _confidence_aggregate(vals)
        expected = math.exp(vals[math.ceil(90 * n / 100) - 1])
        assert p90 == pytest.approx(expected, abs=1e-3), (n, vals[:5])


# ------------------------------------------------------------------- activity

def test_activity_hour_profiles_daily(home):
    seed(home, when=now().replace(hour=9), tool="shell")
    seed(home, when=now().replace(hour=9))
    seed(home, when=now().replace(hour=3), profile="coder")

    act = generate_insights(home)["activity"]
    assert act["busiest_hour"] == "09"
    hist = {h["hour"]: h["traces"] for h in act["hour_histogram"]}
    assert hist["09"] == 2 and hist["03"] == 1 and len(hist) == 24
    profiles = {p["profile"]: p for p in act["profiles"]}
    assert set(profiles) == {"sable", "coder"}
    assert profiles["sable"]["traces"] == 2
    assert act["traces_per_day"][-1]["traces"] == 3


# ------------------------------------------------------------------- filters

def test_days_and_profile_filters(home):
    seed(home, when=now() - timedelta(days=3), profile="sable")
    seed(home, when=now() - timedelta(days=40), profile="sable")
    seed(home, when=now() - timedelta(days=1), profile="coder")

    r30 = generate_insights(home, days=30)
    assert r30["overview"]["traces"] == 2          # the 40-day-old one drops
    r_p = generate_insights(home, days=30, profile="coder")
    assert r_p["overview"]["traces"] == 1
    assert all(p["profile"] == "coder" for p in r_p["activity"]["profiles"])


# --------------------------------------------------------------- promotions

def test_promotion_delta_before_after(home):
    db = home / "buffer" / "sable.db"
    reg = ModelRegistry(db)
    try:
        m1 = reg.register_model("base-x")           # v1
        reg.set_validation(m1.model_id, {"tasks": []}, passed=True)
        reg.promote(m1.model_id)
        m2 = reg.register_model("base-x")           # v2, parent v1
        assert m2.parent_adapter == "v1"
        reg.set_validation(m2.model_id, {"tasks": []}, passed=True)
        reg.promote(m2.model_id)
        # Backdate both promotions so symmetric windows contain seeded traces.
        with reg._lock:
            for mid, days_ago in (("v1", 6), ("v2", 3)):
                ts = (now() - timedelta(days=days_ago)).isoformat(
                    timespec="seconds")
                reg._conn.execute("UPDATE models SET promoted_at = ? "
                                  "WHERE model_id = ?", (ts, mid))
            reg._conn.commit()
    finally:
        reg.close()

    # History edges at -4d/-2d make the symmetric window around the v2
    # promotion (-3d) ±1d: one v1 trace before it, one failing v2 trace after.
    seed(home, when=now() - timedelta(days=4), model_alias="v1")
    seed(home, when=now() - timedelta(days=2), model_alias="v2",
         completed=False)

    promo = generate_insights(home)["promotions"]
    assert promo["available"] is True

    v2 = next(d for d in promo["deltas"] if d["model_id"] == "v2")
    assert v2["previous_model_id"] == "v1"
    assert v2["match"] == "model_id"          # traces carry model_alias=vN
    assert v2["window_days"] == pytest.approx(1.0)
    assert v2["before_avg_score"] == pytest.approx(0.5)   # v1 traces before
    assert v2["after_avg_score"] == pytest.approx(-0.5)   # v2 traces after
    assert v2["delta"] == pytest.approx(-1.0)
    assert v2["before_traces"] == 1 and v2["after_traces"] == 1

    # v1 (first promotion, no parent): entry exists; before-window falls back
    # to all traces in the window since there is no previous model to match.
    v1 = next(d for d in promo["deltas"] if d["model_id"] == "v1")
    assert v1["previous_model_id"] is None
    assert v1["promoted_at"] is not None


def test_promotion_section_degrades_without_history(home):
    seed(home, when=now() - timedelta(days=1))
    promo = generate_insights(home)["promotions"]
    # No registry tables in a fresh buffer DB -> unavailable with a reason.
    assert promo["available"] is False
    assert promo["reason"]


# ---------------------------------------------------- intent review (#36)

def test_intent_review_pairs_promote_intent_with_delta(home):
    """Promote-with-intent gets the same before/after window as promotions."""
    db = home / "buffer" / "sable.db"
    reg = ModelRegistry(db)
    try:
        m1 = reg.register_model("base-x")            # v1
        reg.set_validation(m1.model_id, {"tasks": []}, passed=True)
        reg.promote(m1.model_id)
        m2 = reg.register_model("base-x")            # v2, parent v1
        reg.set_validation(m2.model_id, {"tasks": []}, passed=True)
        reg.promote(m2.model_id, intent="expect fewer tool errors")
        # Backdate the ledger entry to when the promotion "happened", mirroring
        # promoted_at (the review anchors windows on the entry's own ts).
        ts = (now() - timedelta(days=3)).isoformat(timespec="seconds")
        with reg._lock:
            reg._conn.execute(
                "UPDATE ledger SET ts = ? WHERE op = 'promote' AND seq = "
                "(SELECT MAX(seq) FROM ledger WHERE op = 'promote')", (ts,))
            reg._conn.commit()
    finally:
        reg.close()

    # Same edges as the promotions test: ±1d window around -3d.
    seed(home, when=now() - timedelta(days=4), model_alias="v1")
    seed(home, when=now() - timedelta(days=2), model_alias="v2",
         completed=False)

    ir = generate_insights(home)["intent_review"]
    assert ir["available"] is True
    # Only the intented promote shows up; v1's promote had none.
    assert len(ir["rows"]) == 1
    r = ir["rows"][0]
    assert r["op"] == "promote" and r["target"] == "v2"
    assert r["intent"] == "expect fewer tool errors"
    assert r["before_score"] == pytest.approx(0.5)   # v1 traces before
    assert r["after_score"] == pytest.approx(-0.5)   # failing v2 traces after
    assert r["delta"] == pytest.approx(-1.0)         # expectation missed
    assert r["window_days"] == pytest.approx(1.0)
    assert r["match"] == "model_id"


def test_intent_review_honest_empty_no_fabrication(home):
    seed(home, when=now() - timedelta(days=1))       # traces but no intents
    ir = generate_insights(home)["intent_review"]
    assert ir["available"] is True                   # readable, just empty
    assert ir["rows"] == []                          # zero synthetic rows
    assert ir["hint"] == "no recorded intents"


def test_intent_review_ignores_ops_without_intent(home):
    db = home / "buffer" / "sable.db"
    reg = ModelRegistry(db)
    try:
        reg.register_model("base-x")
        reg.set_validation("v1", {"tasks": []}, passed=True)
        reg.promote("v1")                            # no intent recorded
    finally:
        reg.close()
    seed(home, when=now() - timedelta(days=1))
    ir = generate_insights(home)["intent_review"]
    assert ir["available"] is True and ir["rows"] == []


def test_intent_review_cli_output(home, capsys):
    from sable.cli import main as cli_main

    db = home / "buffer" / "sable.db"
    reg = ModelRegistry(db)
    try:
        reg.register_model("base-x")
        reg.set_validation("v1", {"tasks": []}, passed=True)
    finally:
        reg.close()
    assert cli_main(["models", "promote", "v1", "--intent",
                     "halve the timeout failures"]) == 0
    capsys.readouterr()
    assert cli_main(["insights", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    ir = report["intent_review"]
    assert ir["available"] is True and len(ir["rows"]) == 1
    assert ir["rows"][0]["intent"] == "halve the timeout failures"


# ------------------------------------------------- confidence calibration (#40)

def test_calibration_buckets_and_confidently_wrong(home):
    """High-confidence-but-failing top bucket flips the divergence flag."""
    day = now() - timedelta(days=1)
    # very_high bucket (conf >= -0.1): three FAILING traces (score -0.5 each)
    for _ in range(3):
        seed(home, when=day, completed=False, conf_mean=-0.05,
             conf_p90_negentropy=0.95)
    # low bucket ([-2.0,-1.0)): three succeeding traces (score +0.5 each)
    for _ in range(3):
        seed(home, when=day, completed=True, conf_mean=-1.5,
             conf_p90_negentropy=0.22)
    # no confidence data -> excluded from every bucket
    seed(home, when=day, completed=True)

    cal = generate_insights(home)["calibration"]
    assert cal["available"] is True
    by_label = {r["bucket_label"]: r for r in cal["rows"]}
    assert sum(r["n"] for r in cal["rows"]) == 6      # conf-less trace excluded
    top = by_label["very_high (>=-0.1)"]
    assert top["n"] == 3
    assert top["mean_conf"] == pytest.approx(-0.05, abs=1e-4)
    assert top["mean_score"] == pytest.approx(-0.5)
    low = by_label["low ([-2.0,-1.0))"]
    assert low["n"] == 3 and low["mean_score"] == pytest.approx(0.5)
    # pooled others (+0.5) minus top (-0.5) = 1.0 > 0.15 gap threshold
    assert cal["confidently_wrong"] is True


def test_calibration_no_divergence_when_confidence_tracks_quality(home):
    day = now() - timedelta(days=1)
    for _ in range(3):
        seed(home, when=day, completed=True, conf_mean=-0.05)   # top: +0.5
    for _ in range(3):
        seed(home, when=day, completed=False, conf_mean=-1.5)   # low: -0.5
    cal = generate_insights(home)["calibration"]
    assert cal["confidently_wrong"] is False


def test_calibration_honest_empty(home):
    seed(home, when=now() - timedelta(days=1))       # traces, no conf data
    cal = generate_insights(home)["calibration"]
    assert cal["available"] is True
    assert cal["rows"] == []
    assert cal["hint"] == "no traces with confidence data yet"
    assert cal["confidently_wrong"] is False


def test_calibration_cli_output(home, capsys):
    seed(home, when=now() - timedelta(days=1), conf_mean=-0.3)
    from sable.cli import build_parser

    args = build_parser().parse_args(["insights"])
    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert "Confidence calibration" in out
    assert "high ([-0.5,-0.1))" in out


# ------------------------------------------------------------------------ CLI

def test_cli_insights_json_round_trip(home, capsys):
    seed(home, when=now() - timedelta(days=1), tool="shell", fail_tool=True)
    from sable.cli import build_parser

    args = build_parser().parse_args(["insights", "--days", "7", "--json"])
    assert args.command == "insights" and args.days == 7
    assert args.func(args) == 0
    out = capsys.readouterr().out
    report = json.loads(out)
    assert report["days"] == 7
    assert report["overview"]["traces"] == 1
    assert report["tool_reliability"]["tools"][0]["tool"] == "shell"


def test_cli_insights_human_output(home, capsys):
    seed(home, when=now() - timedelta(days=1), completed=False,
         failure_reason="max_iterations")
    from sable.cli import build_parser

    args = build_parser().parse_args(["insights"])
    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert "Insights — last 30 days" in out
    assert "max_iterations" in out


# ------------------------------------------------------------------- API

def test_api_insights_endpoint(home):
    seed(home, when=now() - timedelta(days=1), profile="sable")
    app = DashboardApp(SableConfig(), home=home)
    r = app.handle("GET", "/api/insights", query={"days": "30"})
    assert r.status == 200
    assert r.body["overview"]["traces"] == 1
    assert r.body["score_trend"]["overall"]["points"]

    rp = app.handle("GET", "/api/insights",
                    query={"days": "30", "profile": "nope"})
    assert rp.status == 200 and rp.body["empty"] is True


def test_api_insights_bad_days(home):
    app = DashboardApp(SableConfig(), home=home)
    r = app.handle("GET", "/api/insights", query={"days": "abc"})
    assert r.status == 400

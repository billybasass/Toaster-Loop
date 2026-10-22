"""WP3 — `sable calibrate` tests (sweep logic, profiles, opt-in serving).

All hermetic: no GPU and no real llama-server. The sweep is driven through the
injected ``measure_fn`` seam returning WP1-shaped bench records; serving
integration is checked against ``supervisor.build_instance_cmd`` argv. Machine
id is mocked via SABLE_MACHINE_ID; dirs via SABLE_CALIBRATE_DIR /
SABLE_BENCH_DIR (tmp_path).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sable.serving import bench
from sable.serving import calibrate as cal


# ------------------------------------------------------------------ fixtures

@pytest.fixture()
def cal_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("SABLE_CALIBRATE_DIR", str(tmp_path / "cal"))
    monkeypatch.setenv("SABLE_BENCH_DIR", str(tmp_path / "bench"))
    monkeypatch.setenv("SABLE_MACHINE_ID", "test-machine")
    return tmp_path


@pytest.fixture()
def gguf(tmp_path):
    g = tmp_path / "model.gguf"
    g.write_bytes(b"GGUF" + b"\x00" * 500)
    return g


def fake_rec(gen_median, streams="tok-base", spread=1.0):
    """Minimal WP1-shaped bench record for the measure seam."""
    return {
        "kind": bench.KIND_BENCH,
        "config": {"prompts_sha256": "psha"},
        "token_identity": {"status": "ok", "streams_sha256": streams},
        "generation": {"median": gen_median, "spread_pct": spread},
        "prompt_processing": {"median": gen_median * 4, "spread_pct": spread},
    }


def perf_measure(perf_by_knob, base=100.0, drift_keys=(), error_keys=()):
    """measure_fn backed by a static perf table.

    ``perf_by_knob``: {knob: {value: gen_median}}; anything unlisted measures
    at ``base``. Keys in drift_keys answer with a DIFFERENT token stream;
    keys in error_keys raise (server failed to start / crashed)."""

    def measure(settings, runs, label):  # noqa: ARG001
        med = base
        streams = "tok-base"
        for knob, table in perf_by_knob.items():
            value = settings.get(knob)
            if value is not None and value in table:
                key = (knob, str(value))
                if key in error_keys:
                    raise RuntimeError("server died mid-bench")
                if key in drift_keys:
                    streams = "tok-drift"
                med *= table[value] / 100.0
        return fake_rec(med, streams=streams)

    return measure


def persist_noise_floor(spread_pct):
    bench.save_result({
        "kind": bench.KIND_NOISE_FLOOR,
        "timestamp": "2026-10-11T00:00:00Z",
        "label": "floor",
        "generation": {"median": 100.0, "spread_pct": spread_pct},
    })


def row_for(profile, knob, value):
    for row in profile["knobs"][knob]["candidates"]:
        if row["value"] == value:
            return row
    raise AssertionError(f"no row for {knob}={value}")


# ------------------------------------------------------------- sweep adoption

def test_sweep_adopts_only_gains_above_threshold(cal_dirs, gguf):
    persist_noise_floor(1.0)  # noise floor 1% -> threshold max(3%, 1%) = 3%
    measure = perf_measure({
        "kv_cache_type": {"bf16": 102.0,   # +2%  -> rejected (below 3%)
                          "q8_0": 110.0,   # +10% -> adopted
                          "q4_0": 95.0},   # slower -> rejected
    })
    prof = cal.run_sweep(object(), gguf_path=str(gguf), runs=2,
                         knobs=["kv_cache_type"], measure_fn=measure)

    assert row_for(prof, "kv_cache_type", "bf16")["verdict"] == "rejected"
    assert row_for(prof, "kv_cache_type", "q8_0")["verdict"] == "adopted"
    assert row_for(prof, "kv_cache_type", "q4_0")["verdict"] == "rejected"
    # default f16 equals the current value -> recorded as baseline, not measured
    assert row_for(prof, "kv_cache_type", "f16")["verdict"] == "baseline"
    assert prof["knobs"]["kv_cache_type"]["adopted"] == "q8_0"
    assert prof["adopted_settings"] == {"kv_cache_type": "q8_0"}
    # delta recorded against the running best (110 vs 100 = +10%)
    assert row_for(prof, "kv_cache_type", "q8_0")["delta_pct"] == \
        pytest.approx(10.0, abs=0.01)


def test_sweep_noise_floor_overrides_three_percent(cal_dirs, gguf):
    persist_noise_floor(12.0)  # threshold max(3%, 12%) = 12%
    measure = perf_measure({"kv_cache_type": {"q8_0": 110.0}})  # +10% < 12%
    prof = cal.run_sweep(object(), gguf_path=str(gguf), runs=2,
                         knobs=["kv_cache_type"], measure_fn=measure)
    row = row_for(prof, "kv_cache_type", "q8_0")
    assert row["verdict"] == "rejected"
    assert "threshold" in row.get("reason", "")
    assert prof["adopted_settings"] == {}


def test_sweep_token_drift_is_invalid_never_adopted(cal_dirs, gguf):
    persist_noise_floor(1.0)
    measure = perf_measure({"kv_cache_type": {"q8_0": 150.0}},
                           drift_keys={("kv_cache_type", "q8_0")})
    prof = cal.run_sweep(object(), gguf_path=str(gguf), runs=2,
                         knobs=["kv_cache_type"], measure_fn=measure)
    row = row_for(prof, "kv_cache_type", "q8_0")
    assert row["verdict"] == "invalid"  # +50% but tokens changed -> never a win
    assert prof["knobs"]["kv_cache_type"]["adopted"] is None
    assert prof["adopted_settings"] == {}


def test_sweep_estimates_and_persists_noise_floor_when_absent(cal_dirs, gguf):
    assert bench.load_noise_floor() is None
    measure = perf_measure({"kv_cache_type": {"q8_0": 110.0}})
    prof = cal.run_sweep(object(), gguf_path=str(gguf), runs=2,
                         knobs=["kv_cache_type"], measure_fn=measure)
    nf = bench.load_noise_floor()
    assert nf is not None, "a noise-floor record must be persisted on first use"
    assert prof["noise_floor"]["spread_pct"] == \
        nf["generation"]["spread_pct"]


def test_sweep_error_candidate_skipped_and_sweep_continues(cal_dirs, gguf):
    persist_noise_floor(1.0)
    measure = perf_measure(
        {"flash_attn": {True: 120.0}, "kv_cache_type": {"q8_0": 115.0}},
        error_keys={("flash_attn", "True")})
    prof = cal.run_sweep(object(), gguf_path=str(gguf), runs=2,
                         knobs=["flash_attn", "kv_cache_type"],
                         measure_fn=measure)
    assert row_for(prof, "flash_attn", True)["verdict"] == "error"
    # sweep continued past the failing candidate into the next knob
    assert row_for(prof, "kv_cache_type", "q8_0")["verdict"] == "adopted"
    assert prof["adopted_settings"] == {"kv_cache_type": "q8_0"}


def test_sweep_unmeasurable_baseline_raises(cal_dirs, gguf):
    persist_noise_floor(1.0)

    def dead(settings, runs, label):  # noqa: ARG001
        raise RuntimeError("no server reachable")

    with pytest.raises(cal.CalibrateError, match="nothing could be measured"):
        cal.run_sweep(object(), gguf_path=str(gguf), runs=2,
                      knobs=["flash_attn"], measure_fn=dead)


def test_sweep_spec_mtp_without_drafter_records_tested_as(cal_dirs, gguf):
    persist_noise_floor(1.0)
    measure = perf_measure({"speculative_type": {"ngram": 110.0,
                                                 "mtp": 130.0}})
    prof = cal.run_sweep(object(), gguf_path=str(gguf), runs=2,
                         knobs=["speculative_type"], measure_fn=measure)
    mtp_row = row_for(prof, "speculative_type", "mtp")
    # No mtp*.gguf next to the model -> run_args fell back to ngram-mod.
    assert mtp_row["tested_as"] == "ngram (mtp unavailable)"
    ngram_row = row_for(prof, "speculative_type", "ngram")
    assert ngram_row.get("tested_as") == "ngram"


def test_sweep_coordinate_wise_best_carries_across_knobs(cal_dirs, gguf):
    persist_noise_floor(1.0)
    # kv q8_0 wins (+10% -> best 110); flash True adds a multiplicative +4%,
    # i.e. +4% over the NEW best (110*1.04 = 114.4) -> adopted too.
    measure = perf_measure({
        "kv_cache_type": {"bf16": 102, "q8_0": 110, "q4_0": 95},
        "flash_attn": {True: 104.0},
    })
    prof = cal.run_sweep(object(), gguf_path=str(gguf), runs=2,
                         knobs=["kv_cache_type", "flash_attn"],
                         measure_fn=measure)
    assert prof["adopted_settings"] == {"kv_cache_type": "q8_0",
                                        "flash_attn": True}


def test_sweep_quick_preset_is_smaller_and_overridable():
    full = cal.resolve_grid()
    quick = cal.resolve_grid(quick=True)
    assert set(quick) < set(full)
    subset = cal.resolve_grid(["kv_cache_type", "flash_attn"])
    assert set(subset) == {"kv_cache_type", "flash_attn"}
    with pytest.raises(ValueError, match="unknown knob"):
        cal.resolve_grid(["bogus_knob"])


# ----------------------------------------------------------- profile storage

def test_profile_save_load_roundtrip_keyed_by_machine_and_model(cal_dirs, gguf):
    prof = {
        "profile_version": cal.PROFILE_VERSION,
        "machine_id": cal.machine_id(),
        "model_hash": cal.model_hash(gguf),
        "timestamp": "2026-10-11T00:00:00Z",
        "adopted_settings": {"kv_cache_type": "q8_0"},
        "knobs": {},
    }
    p = cal.save_profile(prof)
    assert p.is_file() and str(cal_dirs / "cal") in str(p)
    loaded = cal.load_profile(cal.machine_id(), cal.model_hash(gguf))
    assert loaded["adopted_settings"] == {"kv_cache_type": "q8_0"}
    # different machine or model -> no match
    assert cal.load_profile("other-machine", cal.model_hash(gguf)) is None
    other = Path(cal_dirs / "other.gguf")
    other.write_bytes(b"different bytes entirely")
    assert cal.load_profile(cal.machine_id(), cal.model_hash(other)) is None
    listing = cal.list_profiles()
    assert len(listing) == 1 and listing[0]["machine_id"] == "test-machine"
    assert cal.clear_profiles() == 1
    assert cal.list_profiles() == []


def test_model_hash_stable_and_content_sensitive(gguf, tmp_path):
    h1 = cal.model_hash(gguf)
    assert h1 and h1 == cal.model_hash(gguf)
    g2 = tmp_path / "b.gguf"
    g2.write_bytes(b"GGUX" + b"\x00" * 500)  # first byte differs
    assert cal.model_hash(g2) != h1
    g3 = tmp_path / "c.gguf"
    g3.write_bytes(b"GGUF" + b"\x00" * 499)  # size differs
    assert cal.model_hash(g3) != h1
    assert cal.model_hash(tmp_path / "missing.gguf") is None


def test_machine_id_env_and_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv("SABLE_MACHINE_ID", "boxed-42")
    assert cal.machine_id() == "boxed-42"
    monkeypatch.delenv("SABLE_MACHINE_ID")
    mid = cal.machine_id()  # /etc/machine-id or hostname fallback
    assert mid and isinstance(mid, str)


# ----------------------------------------------- serving integration (opt-in)

class _Cfg:
    def __init__(self, serve=None, run_settings=None):
        self.serve = serve or {}
        self.run_settings = run_settings or {}
        self.context_length = 4096
        self.gpu_layers = None
        self.extra_llama_args = []


def _store_profile(gguf, adopted):
    cal.save_profile({
        "profile_version": cal.PROFILE_VERSION,
        "machine_id": cal.machine_id(),
        "model_hash": cal.model_hash(gguf),
        "timestamp": "2026-10-11T00:00:00Z",
        "knobs": {k: {"adopted": v} for k, v in adopted.items()},
    })


def test_apply_calibration_precedence_matrix(cal_dirs, gguf):
    _store_profile(gguf, {"kv_cache_type": "q8_0", "flash_attn": True})

    # 1. calibration OFF (default) -> {} — today's behavior exactly.
    assert cal.apply_calibration(_Cfg(), str(gguf)) == {}
    # unknown mode value also stays off (never guesses).
    assert cal.apply_calibration(_Cfg(serve={"calibration": "yolo"}),
                                 str(gguf)) == {}
    # 2. opt-in + matching profile -> adopted values.
    got = cal.apply_calibration(_Cfg(serve={"calibration": "auto"}), str(gguf))
    assert got == {"kv_cache_type": "q8_0", "flash_attn": True}
    # 3. opt-in but no profile for this model -> {}.
    other = gguf.parent / "other.gguf"
    other.write_bytes(b"unrelated model bytes")
    assert cal.apply_calibration(_Cfg(serve={"calibration": "auto"}),
                                 str(other)) == {}
    # 4. opt-in but unreadable model -> {}.
    assert cal.apply_calibration(_Cfg(serve={"calibration": "auto"}),
                                 str(gguf.parent / "nope.gguf")) == {}


def test_build_instance_cmd_default_path_unchanged(cal_dirs, gguf):
    """No serve.calibration key: argv is exactly what pre-WP3 produced."""
    from sable.serving import supervisor
    from sable.serving.runtime import RuntimeInfo

    _store_profile(gguf, {"kv_cache_type": "q8_0"})
    rt = RuntimeInfo(system="linux", machine="x86_64", backend="cpu")
    inst = {"id": "default", "default": True, "gguf": str(gguf),
            "run_settings": {}}
    cmd = supervisor.build_instance_cmd("llama-server", _Cfg(), inst,
                                        port=1234, model_name="m", rt=rt)
    assert "-ctk" not in cmd, "calibrated values must NOT leak without opt-in"


def test_build_instance_cmd_opt_in_adopts_and_manual_wins(cal_dirs, gguf):
    from sable.serving import supervisor
    from sable.serving.runtime import RuntimeInfo

    _store_profile(gguf, {"kv_cache_type": "q8_0", "parallel": 4})
    rt = RuntimeInfo(system="linux", machine="x86_64", backend="cpu")

    # opt-in: calibrated values become defaults for the default instance.
    cfg = _Cfg(serve={"calibration": "auto"})
    inst = {"id": "default", "default": True, "gguf": str(gguf),
            "run_settings": {}}
    cmd = supervisor.build_instance_cmd("llama-server", cfg, inst,
                                        port=1234, model_name="m", rt=rt)
    assert "-ctk" in cmd and cmd[cmd.index("-ctk") + 1] == "q8_0"
    assert "-np" in cmd and cmd[cmd.index("-np") + 1] == "4"

    # manual explicit config ALWAYS overrides the calibrated value.
    cfg2 = _Cfg(serve={"calibration": "auto"},
                run_settings={"kv_cache_type": "q4_0"})
    cmd2 = supervisor.build_instance_cmd("llama-server", cfg2, inst,
                                         port=1234, model_name="m", rt=rt)
    assert cmd2[cmd2.index("-ctk") + 1] == "q4_0"


# ------------------------------------------------------------------ CLI wiring

def cli_main(argv):
    from sable import cli

    return cli.main(argv)


def test_cli_calibrate_help_and_flags():
    from sable import cli

    ns = cli.build_parser().parse_args(
        ["calibrate", "--quick", "--knobs", "flash_attn", "--runs", "2"])
    assert ns.command == "calibrate"
    assert ns.quick and ns.knobs == "flash_attn" and ns.runs == 2


def test_cli_list_show_clear_roundtrip(cal_dirs, gguf, capsys):
    assert cli_main(["calibrate", "--list"]) == 0
    assert "no calibration profiles" in capsys.readouterr().out

    _store_profile(gguf, {"kv_cache_type": "q8_0"})
    assert cli_main(["calibrate", "--list"]) == 0
    out = capsys.readouterr().out
    assert "test-machine" in out and "active" in out

    assert cli_main(["calibrate", "--show", "--model", str(gguf)]) == 0
    out = capsys.readouterr().out
    assert "machine_id:  test-machine" in out
    assert '"kv_cache_type": "q8_0"' in out

    assert cli_main(["calibrate", "--clear"]) == 0
    assert "removed 1" in capsys.readouterr().out
    assert cal.list_profiles() == []


def test_cli_show_without_profile_exits_1(cal_dirs, gguf, capsys):
    assert cli_main(["calibrate", "--show", "--model", str(gguf)]) == 1
    assert "no calibration profile" in capsys.readouterr().out


def test_cli_sweep_without_model_exits_2(cal_dirs, capsys):
    assert cli_main(["calibrate"]) == 2
    assert "no GGUF to calibrate" in capsys.readouterr().err


def test_cli_unknown_knob_exits_2(cal_dirs, gguf, capsys):
    assert cli_main(["calibrate", "--model", str(gguf),
                     "--knobs", "not_a_knob"]) == 2
    err = capsys.readouterr().err
    assert "unknown knob" in err


def test_cli_dead_endpoint_exits_1(cal_dirs, gguf, capsys):
    # Attach to a dead loopback port: nothing can be measured -> exit 1.
    rc = cli_main(["calibrate", "--model", str(gguf), "--runs", "1",
                   "--knobs", "flash_attn",
                   "--attach", "http://127.0.0.1:9/v1"])
    assert rc == 1
    assert "error" in capsys.readouterr().err


def test_cli_nonloopback_attach_exits_2(cal_dirs, gguf, capsys):
    rc = cli_main(["calibrate", "--model", str(gguf),
                   "--attach", "http://example.com/v1"])
    assert rc == 2
    assert "loopback" in capsys.readouterr().err


def test_cli_sweep_attach_end_to_end(cal_dirs, gguf, capsys, monkeypatch):
    """Full CLI sweep through the attach seam against a fake server.

    The candidate knobs are NOT applied to an attached foreign server (honest
    mode: profile records endpoint_mode=attach), so this exercises wiring +
    rendering, not real autotune — spawn mode is covered via run_sweep above."""
    from fake_server import FakeOpenAIServer
    from test_bench import _BenchHandler

    _BenchHandler.requests = []
    with FakeOpenAIServer(handler_cls=_BenchHandler) as srv:
        rc = cli_main(["calibrate", "--model", str(gguf), "--runs", "1",
                       "--quick", "--attach", srv.base_url])
    out = capsys.readouterr().out
    assert rc == 0
    assert "sable calibrate" in out
    assert "adoption threshold" in out
    profs = cal.list_profiles()
    assert len(profs) == 1
    assert profs[0]["endpoint_mode"] == "attach"


def test_rendering_smoke(cal_dirs, gguf):
    persist_noise_floor(2.0)
    measure = perf_measure({"kv_cache_type": {"q8_0": 110.0}})
    prof = cal.run_sweep(object(), gguf_path=str(gguf), runs=2,
                         knobs=["kv_cache_type"], measure_fn=measure)
    table = cal.format_summary(prof)
    assert "knob" in table and "verdict" in table
    assert "adopted: kv_cache_type=q8_0" in table
    shown = cal.format_profile_show(prof)
    assert "machine_id:" in shown and "noise floor" in shown

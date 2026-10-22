"""WP1 — bench harness tests (stats, persistence, token identity, CLI wiring).

All hermetic: the existing FakeOpenAIServer harness answers chat
completions; no GPU and no real llama-server. SABLE_HOME is redirected by
conftest; the bench dir additionally honors SABLE_BENCH_DIR (tmp_path).
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler

import pytest

from sable.serving import bench


# ------------------------------------------------------------- fake servers

class _BenchHandler(BaseHTTPRequestHandler):
    """Deterministic chat-completions server: replies are a pure function of
    the prompt, so repeated runs produce identical token streams. Optionally
    emits llama.cpp-style ``timings`` when TIMINGS is set on the class."""

    REPLY = "the answer"
    # e.g. {"prompt_per_second": 100.0, "predicted_per_second": 20.0}
    TIMINGS: dict | None = None
    requests: list = []

    def log_message(self, format, *args):  # silence
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") == "/v1/models":
            self._send({"object": "list",
                        "data": [{"id": "bench-model", "object": "model"}]})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        if not self.path.startswith("/v1/chat/completions"):
            self._send({"error": "not found"}, 404)
            return
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        type(self).requests.append(payload)
        last_user = ""
        for m in reversed(payload.get("messages", [])):
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                last_user = m["content"]
                break
        data = {
            "id": "cmpl-bench",
            "object": "chat.completion",
            "model": payload.get("model", "bench-model"),
            "choices": [{"index": 0,
                         "message": {"role": "assistant",
                                     "content": self.REPLY + "|" + last_user[:8]},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 7,
                      "total_tokens": 19},
        }
        timings = type(self).TIMINGS
        if timings is not None:
            data["timings"] = dict(timings)
        self._send(data)


class _DriftHandler(_BenchHandler):
    """Nondeterministic server: reply changes on every request, so repeated
    temperature-0 runs produce DIVERGING token streams."""

    counter = 0

    def do_POST(self):
        type(self).counter += 1
        self.REPLY = f"drift-{type(self).counter}"
        super().do_POST()


@pytest.fixture()
def bench_home(tmp_path, monkeypatch):
    d = tmp_path / "benchhome"
    monkeypatch.setenv("SABLE_BENCH_DIR", str(d))
    return d


# ---------------------------------------------------------------- statistics

def test_compute_stats_known_values():
    s = bench.compute_stats([10.0, 20.0, 30.0])
    assert s["median"] == 20.0
    assert s["min"] == 10.0
    assert s["max"] == 30.0
    # spread = (30-10)/20 * 100 = 100%
    assert s["spread_pct"] == pytest.approx(100.0)


def test_compute_stats_even_count_median():
    s = bench.compute_stats([2.0, 4.0, 6.0, 8.0])
    assert s["median"] == 5.0
    assert s["spread_pct"] == pytest.approx((8.0 - 2.0) / 5.0 * 100.0)


def test_compute_stats_zero_median_spread_undefined():
    s = bench.compute_stats([0.0, 0.0])
    assert s["median"] == 0.0
    assert s["spread_pct"] is None


def test_compute_stats_empty_raises():
    with pytest.raises(ValueError):
        bench.compute_stats([])


# ------------------------------------------------------------- measurement

def test_run_bench_repeat_runs_and_identity_ok():
    from fake_server import FakeOpenAIServer

    _BenchHandler.requests = []
    with FakeOpenAIServer(handler_cls=_BenchHandler) as srv:
        rec = bench.run_bench(srv.base_url, "bench-model",
                              prompts=["alpha", "beta"], runs=3)
    assert rec["schema_version"] == bench.SCHEMA_VERSION
    assert rec["kind"] == bench.KIND_BENCH
    assert len(rec["per_run"]) == 3
    assert all(r["completion_tokens"] == 14 for r in rec["per_run"])
    gen = [r["gen_toks_per_s"] for r in rec["per_run"]]
    assert all(g and g > 0 for g in gen)
    # stats block consistent with per-run values
    assert rec["generation"]["min"] == min(gen)
    assert rec["generation"]["max"] == max(gen)
    # deterministic server => identical streams across runs
    assert rec["token_identity"]["status"] == "ok"
    assert rec["token_identity"]["divergent_runs"] == []
    # temperature 0 is what we actually sent
    assert all(r["temperature"] == 0 for r in _BenchHandler.requests)


def test_run_bench_uses_server_timings_when_present():
    from fake_server import FakeOpenAIServer

    _BenchHandler.TIMINGS = {"prompt_per_second": 100.0,
                             "predicted_per_second": 20.0}
    try:
        with FakeOpenAIServer(handler_cls=_BenchHandler) as srv:
            rec = bench.run_bench(srv.base_url, "bench-model",
                                  prompts=["one"], runs=2)
    finally:
        _BenchHandler.TIMINGS = None
    assert rec["timings_source"] == "server"
    assert rec["generation"]["median"] == pytest.approx(20.0)
    assert rec["prompt_processing"]["median"] == pytest.approx(100.0)


def test_run_bench_token_identity_drift_detected():
    from fake_server import FakeOpenAIServer

    _DriftHandler.counter = 0
    with FakeOpenAIServer(handler_cls=_DriftHandler) as srv:
        rec = bench.run_bench(srv.base_url, "bench-model",
                              prompts=["alpha"], runs=3)
    ti = rec["token_identity"]
    assert ti["status"] == "drift"
    # runs 2 and 3 differ from run 1's stream
    assert ti["divergent_runs"] == [2, 3]


def test_run_bench_connection_failure_clear_message():
    with pytest.raises(bench.BenchError, match="llama-server running"):
        bench.run_bench("http://127.0.0.1:9/v1", "m", prompts=["x"], runs=1)


# ------------------------------------------------------------- persistence

def test_bench_dir_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("SABLE_BENCH_DIR", str(tmp_path / "custom"))
    assert bench.bench_dir() == tmp_path / "custom"


def test_bench_dir_under_home():
    from sable.store.paths import resolve_sable_home

    assert bench.bench_dir() == resolve_sable_home() / "bench"


def test_save_load_roundtrip(bench_home):
    rec = {"schema_version": bench.SCHEMA_VERSION, "kind": bench.KIND_BENCH,
           "timestamp": "2026-10-11T12:00:00Z", "label": "exp-a",
           "generation": {"median": 42.0}}
    p = bench.save_result(rec)
    assert p.is_file()
    assert str(bench_home) in str(p)
    loaded = bench.load_results()
    assert len(loaded) == 1
    assert loaded[0]["label"] == "exp-a"
    assert loaded[0]["generation"]["median"] == 42.0


def test_save_result_filename_keyed_timestamp_label(bench_home):
    rec = {"kind": bench.KIND_NOISE_FLOOR,
           "timestamp": "2026-10-11T12:34:56Z", "label": "floor one"}
    p = bench.save_result(rec)
    assert p.name == "noise_floor-20261011T123456Z-floor-one.json"


def test_load_results_filters_and_skips_corrupt(bench_home):
    bench_home.mkdir(parents=True)
    (bench_home / "a.json").write_text(json.dumps(
        {"kind": bench.KIND_BENCH, "label": "x",
         "timestamp": "2026-10-10T00:00:00Z"}))
    (bench_home / "b.json").write_text(json.dumps(
        {"kind": bench.KIND_NOISE_FLOOR, "label": "floor",
         "timestamp": "2026-10-11T00:00:00Z"}))
    (bench_home / "junk.json").write_text("{not json")
    assert len(bench.load_results()) == 2
    assert [r["label"] for r in bench.load_results(
        kind=bench.KIND_NOISE_FLOOR)] == ["floor"]
    assert [r["label"] for r in bench.load_results(label="x")] == ["x"]


def test_noise_floor_roundtrip_and_newest_wins(bench_home):
    old = {"kind": bench.KIND_NOISE_FLOOR, "label": "",
           "timestamp": "2026-10-01T00:00:00Z",
           "generation": {"median": 10.0, "spread_pct": 5.0}}
    new = dict(old, timestamp="2026-10-11T00:00:00Z",
               generation={"median": 12.0, "spread_pct": 3.0})
    bench.save_result(old)
    bench.save_result(new)
    nf = bench.load_noise_floor()
    assert nf is not None
    assert nf["timestamp"] == "2026-10-11T00:00:00Z"
    assert nf["generation"]["spread_pct"] == 3.0


# ------------------------------------------------------- identity rule / cmp

def _mk_record(sha, gen_median, spread=4.0):
    return {
        "kind": bench.KIND_BENCH,
        "config": {"prompts_sha256": sha},
        "token_identity": {"status": "ok", "streams_sha256": sha + "-tok"},
        "generation": {"median": gen_median, "spread_pct": spread},
        "prompt_processing": {"median": gen_median * 4, "spread_pct": spread},
    }


def test_compare_valid_speedup_and_noise_floor_flag():
    a = _mk_record("sha1", 100.0, spread=4.0)
    b = _mk_record("sha1", 110.0, spread=4.0)  # +10% > 4% floor
    c = _mk_record("sha1", 102.0, spread=4.0)  # +2% < 4% floor
    cmp_ok = bench.compare_records(a, b)
    assert cmp_ok["valid"] is True
    assert cmp_ok["verdict"] == "VALID"
    assert cmp_ok["generation"]["speedup"] == pytest.approx(0.10)
    assert cmp_ok["generation"]["within_noise_floor"] is False

    cmp_noise = bench.compare_records(a, c)
    assert cmp_noise["valid"] is True
    assert cmp_noise["generation"]["within_noise_floor"] is True


def test_compare_invalid_when_tokens_differ():
    a = _mk_record("sha1", 100.0)
    b = _mk_record("sha1", 150.0)
    b["token_identity"]["streams_sha256"] = "different-tokens"
    cmp_res = bench.compare_records(a, b)
    assert cmp_res["valid"] is False
    assert cmp_res["verdict"] == "INVALID"
    assert any("INVALID" in r for r in cmp_res["reasons"])


def test_compare_invalid_on_different_prompt_sets():
    a = _mk_record("sha1", 100.0)
    b = _mk_record("sha2", 150.0)
    b["token_identity"]["streams_sha256"] = "sha2-tok"
    cmp_res = bench.compare_records(a, b)
    assert cmp_res["valid"] is False
    assert any("prompts" in r for r in cmp_res["reasons"])


def test_compare_invalid_on_within_run_drift():
    a = _mk_record("sha1", 100.0)
    b = _mk_record("sha1", 150.0)
    b["token_identity"]["status"] = "drift"
    cmp_res = bench.compare_records(a, b)
    assert cmp_res["valid"] is False


# ------------------------------------------------------------------ prompts

def test_load_prompts_file(tmp_path):
    f = tmp_path / "p.txt"
    f.write_text("# comment\n\none prompt\ntwo prompt\n")
    assert bench.load_prompts_file(f) == ["one prompt", "two prompt"]
    empty = tmp_path / "e.txt"
    empty.write_text("# only comments\n")
    with pytest.raises(bench.BenchError):
        bench.load_prompts_file(empty)


def test_prompts_sha256_stable_and_sensitive():
    assert bench.prompts_sha256(["a", "b"]) == bench.prompts_sha256(["a", "b"])
    assert bench.prompts_sha256(["a", "b"]) != bench.prompts_sha256(["b", "a"])


# -------------------------------------------------------------- CLI wiring

def test_cli_bench_help_and_registration():
    from sable import cli

    parser = cli.build_parser()
    ns = parser.parse_args(["bench", "--runs", "3", "--noise-floor"])
    assert ns.command == "bench"
    assert ns.runs == 3
    assert ns.noise_floor is True


def test_cli_bench_attach_end_to_end(bench_home, capsys):
    from fake_server import FakeOpenAIServer

    _BenchHandler.requests = []
    with FakeOpenAIServer(handler_cls=_BenchHandler) as srv:
        rc = cli_main(["bench", "--attach", srv.base_url, "--runs", "2",
                       "--label", "cli-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "sable bench" in out
    assert "median=" in out
    assert "spread=" in out
    assert "token identity: OK" in out
    # persisted under the overridden dir
    files = list(bench_home.glob("bench-*-cli-run.json"))
    assert len(files) == 1


def test_cli_bench_noise_floor_mode_and_list(bench_home, capsys):
    from fake_server import FakeOpenAIServer

    with FakeOpenAIServer(handler_cls=_BenchHandler) as srv:
        assert cli_main(["bench", "--attach", srv.base_url, "--runs", "2",
                         "--noise-floor", "--label", "floor"]) == 0
    capsys.readouterr()
    assert list(bench_home.glob("noise_floor-*-floor.json"))
    assert cli_main(["bench", "--list"]) == 0
    out = capsys.readouterr().out
    assert "noise_floor" in out and "floor" in out


def test_cli_bench_compare_valid_and_invalid(bench_home, capsys):
    from fake_server import FakeOpenAIServer

    # two identical runs -> valid comparison, tokens unchanged
    with FakeOpenAIServer(handler_cls=_BenchHandler) as srv:
        assert cli_main(["bench", "--attach", srv.base_url, "--runs", "2",
                         "--label", "base"]) == 0
        assert cli_main(["bench", "--attach", srv.base_url, "--runs", "2",
                         "--label", "cand"]) == 0
    capsys.readouterr()
    assert cli_main(["bench", "--compare", "base", "cand"]) == 0
    out = capsys.readouterr().out
    assert "VALID" in out and "INVALID" not in out


def test_cli_bench_compare_missing_record_exits_2(bench_home, capsys):
    from sable import cli

    rc = cli.main(["bench", "--compare", "nope-a", "nope-b"])
    assert rc == 2
    assert "no persisted bench record" in capsys.readouterr().err


def test_cli_bench_dead_endpoint_clear_error(capsys):
    # No attach, no gguf configured -> clear endpoint error, no spawn attempts.
    from sable import cli

    rc = cli.main(["bench", "--attach", "http://127.0.0.1:9/v1"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "error" in err


def cli_main(argv):
    from sable import cli

    return cli.main(argv)

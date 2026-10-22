"""WP7 doctor tests — hermetic: tmp SABLE_HOME + injected probes only.

The doctor is importable as a module with pure-ish functions (every
environment probe injectable), so these tests never spawn subprocesses,
never touch real hardware and never write outside tmp dirs.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from sable import doctor
from sable.core.config import ConfigError, SableConfig
from sable.core.exitcodes import EXIT_CONFIG_INVALID, EXIT_ERROR, EXIT_OK


# --------------------------------------------------------------------- helpers

class FakeUsage:
    def __init__(self, free: int, total: int = 100 * 1024 ** 3):
        self.total = total
        self.free = free
        self.used = total - free


def _write_gguf(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.truncate(size)


def _registry_with_model(home: Path, model_id: str, gguf: Path) -> None:
    db = home / "buffer" / "sable.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE models (model_id TEXT, gguf_path TEXT)")
    conn.execute("INSERT INTO models VALUES (?, ?)", (model_id, str(gguf)))
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------- disk

def test_disk_check_flags_low_space_on_export_paths(tmp_path):
    paths = [tmp_path / "home", tmp_path / "tmp"]
    for p in paths:
        p.mkdir()
    required = 10 * 1024 ** 3  # 10 GiB largest model
    # need = required + max(2 GiB, 10%) = 11 GiB; report 10 GiB free -> FAIL.
    ok_free = 12 * 1024 ** 3

    def usage_fn(p):
        return FakeUsage(free=ok_free if "home" in p else 10 * 1024 ** 3)

    res = doctor.check_disk(paths, required, "m-1", disk_usage_fn=usage_fn)
    assert res.status == "problem"
    joined = "\n".join(res.lines)
    assert "BELOW REQUIREMENT" in joined and "OK" in joined
    assert any("janitor" in c or "free space" in c for c in res.next_commands)


def test_disk_check_ok_when_free_covers_model_plus_margin(tmp_path):
    p = tmp_path / "home"
    p.mkdir()
    required = 10 * 1024 ** 3
    res = doctor.check_disk([p], required, "m-1",
                            disk_usage_fn=lambda _p: FakeUsage(
                                free=required + 2 * 1024 ** 3))
    assert res.status == "ok"


def test_largest_registered_model_reads_registry(tmp_path):
    # No DB yet -> zero.
    assert doctor.largest_registered_model_bytes(tmp_path) == (0, None)
    big = tmp_path / "models" / "big.gguf"
    _write_gguf(big, 3 * 1024 ** 2)
    small = tmp_path / "models" / "small.gguf"
    _write_gguf(small, 1024 ** 2)
    db = tmp_path / "buffer" / "sable.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE models (model_id TEXT, gguf_path TEXT)")
    conn.execute("INSERT INTO models VALUES ('m-small', ?)", (str(small),))
    conn.execute("INSERT INTO models VALUES ('m-big', ?)", (str(big),))
    conn.commit()
    conn.close()
    size, mid = doctor.largest_registered_model_bytes(tmp_path)
    assert size == 3 * 1024 ** 2
    assert mid == "m-big"


def test_export_relevant_paths_includes_home_and_training_output(
        tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    out = tmp_path / "outputs"
    cfg = SimpleNamespace(training={"output_dir": str(out)})
    # Fake distinct devices so the same-filesystem dedup keeps all three.
    devs = {}

    def fake_stat(path, *a, **kw):
        devs.setdefault(str(path), len(devs) + 7)
        return SimpleNamespace(st_dev=devs[str(path)])

    monkeypatch.setattr(doctor.os, "stat", fake_stat)
    paths = doctor.export_relevant_paths(cfg, home)
    assert home in paths and out in paths


def test_export_relevant_paths_dedupes_same_filesystem(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    out = tmp_path / "outputs"
    cfg = SimpleNamespace(training={"output_dir": str(out)})
    monkeypatch.setattr(doctor.os, "stat",
                        lambda path, *a, **kw: SimpleNamespace(st_dev=42))
    paths = doctor.export_relevant_paths(cfg, home)
    assert paths == [home]  # one filesystem probed once


# -------------------------------------------------------------------- memory

MEMINFO_FIXTURE = """MemTotal:       32760192 kB
MemFree:         1048576 kB
MemAvailable:   16777216 kB
Buffers:          123456 kB
"""


def test_parse_meminfo_available():
    assert doctor.parse_meminfo_available_kib(MEMINFO_FIXTURE) == 16777216
    assert doctor.parse_meminfo_available_kib("MemTotal: 1 kB") is None


def test_ram_preset_math_on_fixed_inputs():
    # 48 GiB RAM -> context tier 16384; 20 GiB VRAM -> ngl 999 cap 16384.
    p = doctor.presets_from_measurements(48 * 1024, 20 * 1024)
    assert p["serve_context"] == 16384
    assert p["serve_ngl"] == 999
    # batch = 49152 // 16384 = 3
    assert p["train_batch_size"] == 3

    # 80 GiB RAM, 30 GiB VRAM -> 32768 ctx (both tiers allow), batch cap 8? no:
    # 81920 // 16384 = 5.
    p = doctor.presets_from_measurements(80 * 1024, 30 * 1024)
    assert p["serve_context"] == 32768
    assert p["train_batch_size"] == 5

    # Tiny box: 4 GiB RAM, no VRAM -> 4096 ctx, CPU offload unknown.
    p = doctor.presets_from_measurements(4 * 1024, None)
    assert p["serve_context"] == 4096
    assert p["serve_ngl"] is None
    assert p["train_batch_size"] == 1

    # VRAM cap binds below the RAM tier: 80 GiB RAM but only 5 GiB VRAM.
    p = doctor.presets_from_measurements(80 * 1024, 5 * 1024)
    assert p["serve_context"] == 8192   # min(32768, cap 8192)
    assert p["serve_ngl"] == 24         # partial-offload tier

    # Nothing measured -> honest None presets + explanation lines.
    p = doctor.presets_from_measurements(None, None)
    assert p["serve_context"] is None and p["train_batch_size"] is None
    assert any("unavailable" in ln for ln in p["lines"])


def test_free_vram_via_smi_absent_binary(monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda _n: None)
    assert doctor.free_vram_mib_via_smi() is None


def test_free_vram_via_smi_parses_output(monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda _n: "/usr/bin/nvidia-smi")

    def fake_run(cmd, **kw):
        return SimpleNamespace(returncode=0, stdout="24000\n12000\n")

    assert doctor.free_vram_mib_via_smi(run=fake_run) == 12000


# ---------------------------------------------------------- next-command logic

def test_noise_floor_missing_suggests_bench(tmp_path):
    res = doctor.check_noise_floor(tmp_path, load_fn=lambda **kw: None)
    assert res.status == "warn"
    assert res.next_commands == ["run: sable bench --noise-floor"]


def test_noise_floor_present_is_ok(tmp_path):
    res = doctor.check_noise_floor(
        tmp_path, load_fn=lambda **kw: {"label": "base"})
    assert res.status == "ok" and not res.next_commands


def test_calibration_auto_without_profile_suggests_calibrate(tmp_path):
    cfg = SimpleNamespace(serve={"calibration": "auto"})
    res = doctor.check_calibration_profile(
        cfg, tmp_path, mode_fn=lambda c: "auto", list_fn=lambda **kw: [])
    assert res.status == "warn"
    assert res.next_commands == ["run: sable calibrate --quick"]


def test_calibration_off_needs_nothing(tmp_path):
    res = doctor.check_calibration_profile(
        SimpleNamespace(), tmp_path, mode_fn=lambda c: "off",
        list_fn=lambda **kw: [])
    assert res.status == "ok" and not res.next_commands


def test_llama_binary_missing_suggests_setup():
    def boom():
        raise RuntimeError("no binary")

    res = doctor.check_llama_binary(boom)
    assert res.status == "problem"
    assert res.next_commands == ["run: sable setup"]


# --------------------------------------------------------------------- ports

def test_port_check_busy_foreign_is_problem():
    res = doctor.check_ports([8080, 8081], [8081], bind_fn=lambda p: False)
    assert res.status == "problem"
    joined = "\n".join(res.lines)
    assert "held by a live sable instance (OK)" in joined
    assert "BUSY" in joined


def test_port_check_free_is_ok():
    assert doctor.check_ports([9999], [], bind_fn=lambda p: True).status == "ok"


# --------------------------------------------------------------- stale state

def _mk_state(home: Path, iid: str, body: dict) -> Path:
    d = home / "serving" / "state"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{iid}.json"
    p.write_text(json.dumps(body), encoding="utf-8")
    return p


def test_stale_state_reported_but_not_deleted_by_default(tmp_path):
    dead = _mk_state(tmp_path, "a", {"pid": 424242})
    foreign = _mk_state(tmp_path, "b", {"pid": 424243})
    attach = _mk_state(tmp_path, "c", {"mode": "attach",
                                       "base_url": "http://127.0.0.1:1/v1"})

    def alive_fn(pid):
        return None if pid == 424243 else False  # b: not ours to know

    res = doctor.check_stale_state(tmp_path, alive_fn=alive_fn)
    assert res.status == "problem"
    assert dead.is_file() and foreign.is_file() and attach.is_file()
    joined = "\n".join(res.lines)
    assert "424242" in joined and "424243" not in joined  # foreign never listed
    assert any("--fix" in c for c in res.next_commands)


def test_stale_state_fix_removes_only_verified_gone(tmp_path):
    dead = _mk_state(tmp_path, "a", {"pid": 424242})
    foreign = _mk_state(tmp_path, "b", {"pid": 424243})
    live = _mk_state(tmp_path, "c", {"pid": 424244})

    def alive_fn(pid):
        return {424242: False, 424243: None, 424244: True}[pid]

    res = doctor.check_stale_state(tmp_path, alive_fn=alive_fn, fix=True)
    assert not dead.is_file()          # deterministically gone -> removed
    assert foreign.is_file()           # ownership rule: never touched
    assert live.is_file()              # still running -> kept
    assert res.status == "ok"


def test_stale_state_ignores_garbage_files(tmp_path):
    d = tmp_path / "serving" / "state"
    d.mkdir(parents=True)
    (d / "broken.json").write_text("{not json", encoding="utf-8")
    res = doctor.check_stale_state(tmp_path, alive_fn=lambda p: False)
    assert res.status == "ok"


# -------------------------------------------------------------------- config

def test_config_error_passthrough_exit_3(tmp_path, monkeypatch):
    monkeypatch.setenv("SABLE_HOME", str(tmp_path))
    boom_msg = ("invalid configuration — refusing to guess (2 problems):\n"
                "  - port: expected an integer, found 'abc' (str)\n"
                "  - gguf_path: expected a string, found 3 (int)")

    def bad_load(_path=None):
        raise ConfigError(boom_msg)

    lines: list[str] = []
    rc = doctor.collect_checks(load_config_fn=bad_load, out=lines.append,
                               binary_probe=lambda: "/x/llama-server",
                               meminfo_kib=1024 * 1024, vram_mib=None,
                               alive_fn=lambda p: True)
    results, code = rc
    assert code == EXIT_CONFIG_INVALID
    # The consolidated report is surfaced VERBATIM (every field at once).
    text = "\n".join(lines)
    assert "port: expected an integer" in text
    assert "gguf_path: expected a string" in text


def test_config_ok_when_parses(tmp_path, monkeypatch):
    monkeypatch.setenv("SABLE_HOME", str(tmp_path))
    res = doctor.check_config(None, out=lambda s: None,
                              load_config_fn=lambda _p=None: SimpleNamespace())
    assert res.status == "ok"


# ------------------------------------------------------------- full runner

def test_collect_checks_green_with_injected_probes(tmp_path, monkeypatch):
    monkeypatch.setenv("SABLE_HOME", str(tmp_path))
    # Persisted noise floor + calibration profile so no warn->next-command.
    bench = tmp_path / "bench"
    bench.mkdir()
    (bench / "noise_floor-x.json").write_text(
        json.dumps({"kind": "noise_floor", "label": ""}), encoding="utf-8")

    results, rc = doctor.collect_checks(
        load_config_fn=lambda _p=None: SableConfig(),
        disk_usage_fn=lambda _p: FakeUsage(free=64 * 1024 ** 3),
        meminfo_kib=32 * 1024 * 1024,   # 32 GiB available
        vram_mib=13000,
        binary_probe=lambda: "/fake/llama-server",
        alive_fn=lambda _pid: True,
    )
    assert rc == EXIT_OK
    by_name = {r.name: r for r in results}
    assert by_name["memory"].status == "ok"
    assert by_name["noise-floor"].status == "ok"
    assert by_name["llama-server"].lines == ["/fake/llama-server"]


def test_collect_checks_problems_yield_exit_1(tmp_path, monkeypatch):
    monkeypatch.setenv("SABLE_HOME", str(tmp_path))

    def dead(_pid):
        return False

    _mk_state(tmp_path, "a", {"pid": 999999})
    results, rc = doctor.collect_checks(
        load_config_fn=lambda _p=None: SableConfig(),
        disk_usage_fn=lambda _p: FakeUsage(free=64 * 1024 ** 3),
        meminfo_kib=32 * 1024 * 1024, vram_mib=None,
        binary_probe=lambda: "/fake/llama-server",
        alive_fn=dead,
    )
    assert rc == EXIT_ERROR
    assert any(r.name == "state" and r.status == "problem" for r in results)


def test_run_doctor_prints_report_and_next_commands(tmp_path, monkeypatch):
    monkeypatch.setenv("SABLE_HOME", str(tmp_path))
    lines: list[str] = []
    rc = doctor.run_doctor(
        out=lines.append,
        load_config_fn=lambda _p=None: SableConfig(),
        disk_usage_fn=lambda _p: FakeUsage(free=64 * 1024 ** 3),
        meminfo_kib=8 * 1024 * 1024, vram_mib=None,
        binary_probe=lambda: "/fake/llama-server",
        alive_fn=lambda _pid: True,
    )
    text = "\n".join(lines)
    assert "sable doctor" in text and "next commands:" in text
    # No noise floor persisted in tmp home -> the bench suggestion appears.
    assert "sable bench --noise-floor" in text
    assert rc == EXIT_OK


def test_check_python_bounds():
    assert doctor.check_python((3, 12)).status == "ok"
    assert doctor.check_python((3, 9)).status == "problem"
    assert doctor.check_python((3, 14)).status == "problem"


def test_check_deps_missing_core_dep():
    res = doctor.check_deps(find_spec=lambda m: None if m == "httpx"
                            else object())
    assert res.status == "problem"
    assert any("pip install" in c for c in res.next_commands)


def test_home_writable_check(tmp_path):
    ok = doctor.check_home_writable(tmp_path / "h")
    assert ok.status == "ok" and (tmp_path / "h").is_dir()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))

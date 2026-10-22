"""WP2b — CLI exit-code taxonomy + corrupt-artifact integrity checks.

  0 success | 1 generic error | 2 usage | 3 invalid config (consolidated
  report on stderr) | 4 finished-but-corrupt artifact | 10 update available
  | 130 interrupted   (core/exitcodes.py, README "Exit codes")

Fabricated corruptions: a stub export worker that writes a truncated /
bad-magic GGUF, and a train stub that records NaN in its loss line. Normal
stub runs must still exit 0.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sable.core.exitcodes import (  # noqa: E402
    EXIT_CONFIG_INVALID,
    EXIT_CORRUPT_ARTIFACT,
    EXIT_OK,
)
from sable.training.integrity import (  # noqa: E402
    check_gguf_file,
    export_integrity_problems,
    find_nan_losses,
)

# ---------------------------------------------------------------- unit: integrity

def test_check_gguf_ok(tmp_path):
    g = tmp_path / "m.gguf"
    g.write_bytes(b"GGUF" + b"\x00" * 32)
    assert check_gguf_file(g) is None


def test_check_gguf_missing_zero_and_bad_magic(tmp_path):
    missing = tmp_path / "nope.gguf"
    reason = check_gguf_file(missing)
    assert reason is not None and "missing" in reason

    zero = tmp_path / "zero.gguf"
    zero.write_bytes(b"")
    reason = check_gguf_file(zero)
    assert reason is not None and "zero-byte" in reason

    trunc = tmp_path / "trunc.gguf"
    trunc.write_bytes(b"GGU")          # truncated header
    reason = check_gguf_file(trunc)
    assert reason is not None and "magic" in reason

    wrong = tmp_path / "wrong.gguf"
    wrong.write_bytes(b"impostor bytes")
    reason = check_gguf_file(wrong)
    assert reason is not None and "magic" in reason


def test_find_nan_losses():
    assert find_nan_losses({"train_loss": 0.5, "eval_loss": None}) == []
    assert find_nan_losses({"train_loss": float("nan")}) == ["train_loss"]
    assert find_nan_losses({"train_loss": float("nan"),
                            "eval_loss": float("nan")}) == \
        ["eval_loss", "train_loss"]
    assert find_nan_losses({}) == []


def test_export_integrity_problems(tmp_path):
    good = tmp_path / "g.gguf"
    good.write_bytes(b"GGUFxxxxxxxx")
    assert export_integrity_problems(good) == []
    bad = tmp_path / "b.gguf"
    bad.write_bytes(b"GG")
    probs = export_integrity_problems(bad, {"train_loss": float("nan")})
    assert len(probs) == 2
    assert any("magic" in p for p in probs)
    assert any("NaN" in p for p in probs)


# ---------------------------------------------------------------- stub fixtures

_STUB_BASE = r'''
import argparse, json, os, sys
from pathlib import Path
ap = argparse.ArgumentParser()
for f in (("--job", True), ("--kind", True), ("--config", False),
          ("--dataset-dir", False), ("--base-model", False),
          ("--log-path", False), ("--adapter-dir", False),
          ("--gguf-out", False)):
    ap.add_argument(f[0], required=f[1], default=None)
ap.add_argument("--quant", default="q4_k_m")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

def emit(ev, **kw):
    line = json.dumps({"event": ev, "job": a.job, **kw})
    print(line, flush=True)
    if a.log_path:
        Path(a.log_path).parent.mkdir(parents=True, exist_ok=True)
        open(a.log_path, "a").write(line + "\n")

emit("start")
if a.kind in ("sft", "dpo"):
    out = Path(a.config).parent if a.config else Path(".")
    (out / "adapter").mkdir(parents=True, exist_ok=True)
    (out / "adapter" / "adapter_config.json").write_text(json.dumps(
        {"base_model": a.base_model or "stub", "r": 8}))
    # WP2b knob: emit NaN losses to simulate silently-corrupt numerics
    extra = {}
    if os.environ.get("STUB_NAN_LOSS") == "1":
        extra = {"train_loss": float("nan")}
    emit("sft_done", **extra)
elif a.kind == "export":
    gg = Path(a.gguf_out)
    gg.parent.mkdir(parents=True, exist_ok=True)
    mode = os.environ.get("STUB_EXPORT_MODE", "good")
    if mode == "good":
        gg.write_bytes(b"GGUF" + b"\x00" * 32)
    elif mode == "badmagic":
        gg.write_bytes(b"not a gguf at all")
    elif mode == "empty":
        gg.write_bytes(b"")
emit("done")
'''


@pytest.fixture()
def env(tmp_path, monkeypatch):
    home = tmp_path / "uf-exitcodes"
    monkeypatch.setenv("SABLE_HOME", str(home))
    stub = tmp_path / "stub.py"
    stub.write_text(_STUB_BASE)
    cfg_file = home.parent / "config.yaml"
    cfg_file.write_text(yaml.safe_dump({
        "home": str(home),
        "training": {"base_model": "Base/HF-Model",
                     "worker_command": [sys.executable, str(stub)]},
    }))
    return home, cfg_file


def _seed_dataset(home, ds_id: str):
    ds_dir = home / "datasets" / ds_id
    ds_dir.mkdir(parents=True)
    (ds_dir / "sft.jsonl").write_text(json.dumps(
        {"conversations": [{"from": "human", "value": "hi"}]}) + "\n")
    (ds_dir / "train.yaml").write_text(yaml.safe_dump({
        "model": "REPLACE_WITH_BASE_MODEL_PATH_OR_HF_ID",
        "data": {"local_dataset": [str(ds_dir / "sft.jsonl")],
                 "format_type": "sharegpt"},
    }))


def _iterate_args(cfg_file, tmp_path):
    from sable.cli import build_parser

    return build_parser().parse_args([
        "iterate", "--config", str(cfg_file), "--dry-run",
    ])


# ---------------------------------------------------------------- CLI: code 3

def test_invalid_config_exits_3_with_consolidated_report(env, capsys):
    """`sable.main()` on an invalid --config exits 3 and prints EVERY
    problem field to stderr in one listing."""
    from sable.cli import main

    bad = env[0].parent / "bad.yaml"
    bad.write_text(yaml.safe_dump({
        "port": "not-a-port",
        "context_length": {"nope": 1},
    }))
    rc = main(["traces", "list", "--config", str(bad)])
    assert rc == EXIT_CONFIG_INVALID
    err = capsys.readouterr().err
    assert "refusing to guess" in err
    assert "port" in err and "context_length" in err


def test_missing_config_file_exits_3(env, capsys):
    from sable.cli import main

    rc = main(["traces", "list",
               "--config", str(env[0] / "does-not-exist.yaml")])
    assert rc == EXIT_CONFIG_INVALID
    assert "Config file not found" in capsys.readouterr().err


def test_valid_config_still_exits_0(env, capsys):
    """Successful observable path unchanged: valid config -> exit 0."""
    from sable.cli import main

    _seed_dataset(env[0], "ds-ok")
    rc = main(["train", "start", "--config", str(env[1]),
               "--dataset", "ds-ok", "--dry-run"])
    assert rc == EXIT_OK


# ---------------------------------------------------------------- CLI: code 4

def test_train_start_nan_loss_exits_4(env, monkeypatch, capsys):
    """Worker exits 0 but records NaN train_loss -> exit 4 (corrupt)."""
    from sable.cli import build_parser, cmd_train_start

    _seed_dataset(env[0], "ds-nan")
    monkeypatch.setenv("STUB_NAN_LOSS", "1")
    args = build_parser().parse_args([
        "train", "start", "--config", str(env[1]), "--dataset", "ds-nan"])
    rc = cmd_train_start(args)
    assert rc == EXIT_CORRUPT_ARTIFACT
    assert "NaN" in capsys.readouterr().err


def test_train_start_clean_still_zero(env):
    from sable.cli import build_parser, cmd_train_start

    _seed_dataset(env[0], "ds-clean")
    args = build_parser().parse_args([
        "train", "start", "--config", str(env[1]), "--dataset", "ds-clean"])
    assert cmd_train_start(args) == EXIT_OK


def test_iterate_corrupt_export_exits_4(env, monkeypatch):
    """Full iterate loop: export stub writes a bad-magic GGUF -> the pipeline
    flags res.corrupt and cmd_iterate exits with code 4."""
    import os

    from sable.cli import build_parser, cmd_iterate

    _seed_dataset(env[0], "ds-bad")
    # seed traces so curate produces a dataset on its own is complex; reuse
    # the seeded one via --dataset and let train+export run against stubs.
    monkeypatch.setenv("STUB_EXPORT_MODE", "badmagic")
    try:
        args = build_parser().parse_args([
            "iterate", "--config", str(env[1]), "--dataset", "ds-bad"])
        rc = cmd_iterate(args)
        assert rc == EXIT_CORRUPT_ARTIFACT
    finally:
        os.environ.pop("STUB_EXPORT_MODE", None)


def test_iterate_empty_export_exits_4(env, monkeypatch):
    import os

    from sable.cli import build_parser, cmd_iterate

    _seed_dataset(env[0], "ds-empty")
    monkeypatch.setenv("STUB_EXPORT_MODE", "empty")
    try:
        args = build_parser().parse_args([
            "iterate", "--config", str(env[1]), "--dataset", "ds-empty"])
        assert cmd_iterate(args) == EXIT_CORRUPT_ARTIFACT
    finally:
        os.environ.pop("STUB_EXPORT_MODE", None)


def test_iterate_good_export_still_zero(env):
    """A sound GGUF keeps the historical success path (the validation stage
    may fail without a server starter — what matters is that we do NOT exit
    with the corrupt code)."""
    from sable.cli import build_parser, cmd_iterate

    _seed_dataset(env[0], "ds-good")
    args = build_parser().parse_args([
        "iterate", "--config", str(env[1]), "--dataset", "ds-good",
        "--dry-run"])
    assert cmd_iterate(args) == EXIT_OK


def test_taxonomy_no_collisions():
    """The finalized table: established codes untouched, new ones distinct."""
    from sable.core import exitcodes as ec

    vals = [ec.EXIT_OK, ec.EXIT_ERROR, ec.EXIT_USAGE, ec.EXIT_CONFIG_INVALID,
            ec.EXIT_CORRUPT_ARTIFACT, ec.EXIT_UPDATE_AVAILABLE,
            ec.EXIT_INTERRUPTED]
    assert len(set(vals)) == len(vals)
    assert (ec.EXIT_OK, ec.EXIT_ERROR, ec.EXIT_USAGE) == (0, 1, 2)
    # WP1 bench conventions stay coherent: bad compare operand = usage (2),
    # measurement error = generic (1) — asserted where bench maps them in cli.

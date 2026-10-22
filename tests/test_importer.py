"""Backlog #14 tests — dataset import with format auto-detection.

Detection per format, golden normalization against the exact internal row
shape curate emits, ambiguity errors, malformed-row skip+count, manifest
provenance, and an end-to-end `datasets list` + train pipeline stub run
consuming an imported dataset (no GPU, no torch).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sable.cli as cli  # noqa: E402
from sable.curate.importer import (  # noqa: E402
    ImportError_,
    detect_format,
    import_dataset,
)


@pytest.fixture()
def sable_home(tmp_path, monkeypatch):
    home = tmp_path / "uf-import"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SABLE_HOME", str(home))
    monkeypatch.delenv("SABLE_CONFIG", raising=False)
    return home


def write_jsonl(path: Path, rows, raw: str | None = None):
    if raw is not None:
        path.write_text(raw, encoding="utf-8")
    else:
        path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


# ------------------------------------------------------------------- detection

def test_detect_sharegpt():
    assert detect_format([{"conversations": [{"from": "human", "value": "x"},
                                             {"from": "gpt", "value": "y"}]}]) \
        == "sharegpt"


def test_detect_alpaca():
    assert detect_format([{"instruction": "i", "input": "", "output": "o"}]) \
        == "alpaca"


def test_detect_chatml():
    assert detect_format([{"messages": [{"role": "user", "content": "x"},
                                        {"role": "assistant", "content": "y"}]}]) \
        == "chatml"


def test_detect_ambiguity_error_shows_keys_seen():
    with pytest.raises(ImportError_) as e:
        detect_format([{"prompt": "p", "completion": "c"}])
    msg = str(e.value)
    assert "completion" in msg and "prompt" in msg  # keys seen, clearly shown


def test_detect_empty_raises():
    with pytest.raises(ImportError_):
        detect_format([])


# -------------------------------------------------------------- normalization

def _sft_rows(ds_dir: Path):
    return [json.loads(l) for l in
            (ds_dir / "sft.jsonl").read_text().splitlines()]


def test_golden_alpaca_pair(sable_home):
    src = write_jsonl(sable_home / "a.jsonl", [
        {"instruction": "Add 2 and 2.", "input": "", "output": "4"},
    ])
    res = import_dataset(src, home=sable_home)
    (row,) = _sft_rows(res.dataset_dir)
    # exact internal shape: ShareGPT turns with the gpt think-block invariant
    assert row == {"conversations": [
        {"from": "human", "value": "Add 2 and 2."},
        {"from": "gpt", "value": "< think>\n< /think>\n4"},
    ]}


def test_golden_alpaca_with_input_prefix(sable_home):
    src = write_jsonl(sable_home / "a.jsonl", [
        {"instruction": "Translate.", "input": "Hello", "output": "Hola"},
    ])
    res = import_dataset(src, home=sable_home)
    (row,) = _sft_rows(res.dataset_dir)
    assert row["conversations"][0]["value"] == "Hello\n\nTranslate."


def test_golden_chatml_passthrough(sable_home):
    src = write_jsonl(sable_home / "c.jsonl", [
        {"messages": [{"role": "system", "content": "s"},
                      {"role": "user", "content": "u"},
                      {"role": "assistant", "content": "a"}]},
    ])
    res = import_dataset(src, home=sable_home)
    (row,) = _sft_rows(res.dataset_dir)
    assert row["conversations"] == [
        {"from": "system", "value": "s"},
        {"from": "human", "value": "u"},
        {"from": "gpt", "value": "< think>\n< /think>\na"},
    ]


def test_golden_sharegpt_internal_keys(sable_home):
    src = write_jsonl(sable_home / "s.jsonl", [
        {"conversations": [{"from": "human", "value": "q"},
                           {"from": "gpt", "value": "< think>\n< /think>\na"}]},
    ])
    res = import_dataset(src, home=sable_home)
    (row,) = _sft_rows(res.dataset_dir)
    # from/value round-trips byte-identically through the internal builder
    assert row["conversations"] == [
        {"from": "human", "value": "q"},
        {"from": "gpt", "value": "< think>\n< /think>\na"},
    ]


def test_sharegpt_role_content_keys_accepted(sable_home):
    src = write_jsonl(sable_home / "s.jsonl", [
        {"conversations": [{"role": "user", "content": "q"},
                           {"role": "assistant", "content": "a"}]},
    ])
    res = import_dataset(src, home=sable_home)
    (row,) = _sft_rows(res.dataset_dir)
    assert [t["from"] for t in row["conversations"]] == ["human", "gpt"]


def test_chatml_bad_role_row_skipped(sable_home):
    src = write_jsonl(sable_home / "c.jsonl", [
        {"messages": [{"role": "wizard", "content": "?"}]},
        {"messages": [{"role": "user", "content": "u"},
                      {"role": "assistant", "content": "a"}]},
    ])
    res = import_dataset(src, home=sable_home)
    assert res.rows == 1 and res.skipped_rows == 1


# ------------------------------------------------------- malformed + manifest

def test_malformed_rows_skipped_and_counted(sable_home):
    src = sable_home / "m.jsonl"
    src.write_text(
        '{"instruction": "ok", "output": "y"}\n'
        'not json at all\n'
        '{"instruction": "", "output": ""}\n'   # parses, normalizes to nothing
        '"just a string"\n'
        '{"instruction": "ok2", "output": "y2"}\n',
        encoding="utf-8")
    res = import_dataset(src, home=sable_home)
    assert res.rows == 2
    assert res.skipped_rows == 3  # bad json + empty pair + bare string
    mf = json.loads((res.dataset_dir / "manifest.json").read_text())
    assert mf["skipped_rows"] == 3


def test_manifest_provenance_fields(sable_home):
    src = write_jsonl(sable_home / "pub.jsonl", [
        {"instruction": "i", "output": "o"}])
    res = import_dataset(src, name="pub", home=sable_home)
    mf = json.loads((res.dataset_dir / "manifest.json").read_text())
    assert mf["provenance"] == "imported"
    assert mf["source_file"].endswith("pub.jsonl")
    assert mf["source_format"] == "alpaca"
    assert mf["dataset_id"] == res.dataset_id and "--pub" in res.dataset_id
    assert mf["created_at"]
    assert mf["files"]["sft.jsonl"]["count"] == 1
    assert mf["files"]["sft.jsonl"]["trace_ids"] == []  # no trace provenance
    assert mf["files"]["dpo.jsonl"]["count"] == 0
    assert (res.dataset_dir / "train.yaml").is_file()


def test_json_array_input_accepted(sable_home):
    src = sable_home / "arr.json"
    src.write_text(json.dumps([{"instruction": "i", "output": "o"}]))
    res = import_dataset(src, home=sable_home)
    assert res.rows == 1


def test_unreadable_and_undetectable_raise(sable_home):
    with pytest.raises(ImportError_):
        import_dataset(sable_home / "missing.jsonl", home=sable_home)
    bad = sable_home / "bad.jsonl"
    write_jsonl(bad, [{"foo": 1}])
    with pytest.raises(ImportError_):
        import_dataset(bad, home=sable_home)


# -------------------------------------------------------------------- CLI

def test_cli_import_and_list(sable_home):
    src = write_jsonl(sable_home / "seed.jsonl", [
        {"instruction": "i1", "output": "o1"},
        {"garbage": True},
    ])
    assert cli.main(["datasets", "import", str(src), "--name", "seed"]) == 0
    out = cli.main(["datasets", "list"])
    assert out == 0


def test_cli_import_undetectable_exits_nonzero(sable_home, capsys):
    bad = sable_home / "bad.jsonl"
    write_jsonl(bad, [{"weird": 1}])
    assert cli.main(["datasets", "import", str(bad)]) != 0
    assert "error:" in capsys.readouterr().err


# ------------------------------------- e2e: imported dataset trains via stub

STUB = r'''
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
    # prove the imported sft.jsonl contract is what the worker receives
    rows = (Path(a.dataset_dir) / "sft.jsonl").read_text().splitlines()
    assert rows, "imported dataset produced no rows for the worker"
    first = json.loads(rows[0])
    assert first["conversations"][0]["from"] in ("human", "system")
    out = Path(a.config).parent if a.config else Path(".")
    (out / "adapter").mkdir(parents=True, exist_ok=True)
    (out / "adapter" / "adapter_config.json").write_text(json.dumps(
        {"base_model": a.base_model or "stub", "r": 8}))
elif a.kind == "export":
    gg = Path(a.gguf_out)
    gg.parent.mkdir(parents=True, exist_ok=True)
    fixture = os.environ.get("STUB_FIXTURE_GGUF")
    if fixture and Path(fixture).is_file():
        gg.write_bytes(Path(fixture).read_bytes())
    else:
        gg.write_bytes(b"GGUF-stub")
emit("done")
'''


def test_imported_dataset_trains_end_to_end(sable_home, tmp_path, monkeypatch):
    from fake_server import FakeOpenAIServer, make_scripted_handler
    from sable.curate import CurationConfig
    from sable.registry import ModelRegistry
    from sable.store.db import TraceStore
    from sable.training.pipeline import run_iterate
    from sable.training.scheduler import Scheduler, TrainingConfig
    from sable.training.validate import ValidationConfig

    stub = tmp_path / "stub.py"
    stub.write_text(STUB)
    fixture_gguf = tmp_path / "tiny.gguf"
    fixture_gguf.write_bytes(b"GGUF" + b"\x00" * 32)
    monkeypatch.setenv("STUB_FIXTURE_GGUF", str(fixture_gguf))

    src = write_jsonl(sable_home / "seed.jsonl", [
        {"instruction": f"task {i}", "output": f"result {i} ok"}
        for i in range(3)
    ])
    res = import_dataset(src, name="e2e", home=sable_home)

    store = TraceStore(sable_home / "buffer" / "sable.db")
    reg = ModelRegistry(sable_home / "buffer" / "sable.db")

    def starter(gguf_path):
        srv = FakeOpenAIServer(handler_cls=make_scripted_handler(
            {"*": "ok-answer"}))
        srv.start()
        return srv

    try:
        sched = Scheduler(registry=reg, cfg=TrainingConfig(),
                          worker_command=[sys.executable, str(stub)])
        result = run_iterate(
            home=sable_home, registry=reg, scheduler=sched, store=store,
            curation_cfg=CurationConfig(),
            training_cfg=TrainingConfig(base_model="Base/HF-Model"),
            validation_cfg=ValidationConfig(
                min_pass_rate=1.0,
                golden_tasks=[{"prompt": "x", "expect_contains": ["ok-answer"]}]),
            dataset_id=res.dataset_id,   # skip curate: train the IMPORT
            server_starter=starter,
        )
    finally:
        store.close()
        reg.close()

    assert result.ok and result.promoted, result.to_dict()
    assert result.dataset_id == res.dataset_id
    assert "train" in result.stages and "export" in result.stages

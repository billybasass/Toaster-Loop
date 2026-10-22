"""Backlog item #16 tests — archive-don't-delete lifecycle + disk janitor.

Covers: archive/restore round-trips with real files, collision handling,
refusal matrix (promoted/candidate/pinned/double-archive), restore refusal on
target-exists / tampered archive, pin/unpin, janitor eligibility + state file,
dry-run inertness, the deploy guard for archived models, the auto-check cron
hook (enabled/disabled) and the v4→v5 registry migration.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sable.cli import main as cli_main  # noqa: E402
from sable.registry import ModelRegistry, RegistryError  # noqa: E402
from sable.registry import janitor as janitor_mod  # noqa: E402
from sable.registry.janitor import load_state, run_janitor  # noqa: E402
from sable.serving.supervisor import ServeError  # noqa: E402


@pytest.fixture()
def home(tmp_path):
    h = tmp_path / ".sable-test-home"
    (h / "buffer").mkdir(parents=True)
    return h


@pytest.fixture()
def registry(home):
    reg = ModelRegistry(home / "buffer" / "sable.db")
    yield reg
    reg.close()


def _iso(days_ago: float) -> str:
    ts = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return ts.isoformat(timespec="seconds")


def _make_artifacts(home, registry, model_id, gguf_bytes=b"G" * 100,
                    adapter_files=None):
    """Fake GGUF + adapter dir under home/models, recorded on the row."""
    gguf = home / "models" / "gguf" / f"{model_id}-q4_k_m.gguf"
    gguf.parent.mkdir(parents=True, exist_ok=True)
    gguf.write_bytes(gguf_bytes)
    adapter = home / "models" / "adapters" / model_id
    adapter.mkdir(parents=True, exist_ok=True)
    files = adapter_files or {"adapter_model.bin": b"A" * 50}
    for name, data in files.items():
        (adapter / name).write_bytes(data)
    registry.set_artifacts(model_id, adapter_dir=str(adapter),
                           gguf_path=str(gguf))
    return adapter, gguf


def _age(registry, model_id, days_ago, promoted=True):
    """Backdate a row so janitor age rules see it as old."""
    col = "promoted_at" if promoted else "created_at"
    registry._conn.execute(
        f"UPDATE models SET {col} = ? WHERE model_id = ?",
        (_iso(days_ago), model_id),
    )
    registry._conn.commit()


def _set_last_used(registry, model_id, days_ago):
    """Stamp last_used_at directly (days_ago None -> never used)."""
    val = None if days_ago is None else _iso(days_ago)
    registry._conn.execute(
        "UPDATE models SET last_used_at = ? WHERE model_id = ?",
        (val, model_id),
    )
    registry._conn.commit()


def _superseded(registry, base="base-a"):
    """Register v_n, promote twice so the older row ends superseded."""
    n = len(registry.list_models()) + 1
    registry.register_model(base)
    mid = f"v{n}"
    registry.set_validation(mid, {}, passed=True)
    registry.promote(mid)
    n2 = n + 1
    registry.register_model(base)
    mid2 = f"v{n2}"
    registry.set_validation(mid2, {}, passed=True)
    registry.promote(mid2)
    assert registry.get_model(mid).status == "superseded"
    return mid, mid2


# --------------------------------------------------------- archive/restore

def test_archive_restore_roundtrip_bytes_identical(home, registry):
    mid, _ = _superseded(registry)
    adapter, gguf = _make_artifacts(home, registry, mid)
    before_gguf = gguf.read_bytes()
    before_adapter = {f.name: f.read_bytes() for f in adapter.iterdir()}

    m = registry.archive_model(mid, home=home)
    assert m.status == "archived"
    assert m.gguf_path is None and m.adapter_dir is None  # unserveable
    assert not gguf.exists() and not adapter.exists()
    archive_dir = Path(m.pre_archive["archive_dir"])
    assert (archive_dir / gguf.name).read_bytes() == before_gguf

    back = registry.restore_model(mid)
    assert back.status == "superseded"  # previous terminal status restored
    assert back.gguf_path == str(gguf) and back.adapter_dir == str(adapter)
    assert gguf.read_bytes() == before_gguf
    assert {f.name: f.read_bytes() for f in adapter.iterdir()} == before_adapter
    assert back.pre_archive is None  # record consumed
    assert not archive_dir.exists()  # empty per-model dir tidied


def test_archive_records_previous_status_and_paths(home, registry):
    mid, _ = _superseded(registry)
    adapter, gguf = _make_artifacts(home, registry, mid)
    m = registry.archive_model(mid, home=home)
    rec = m.pre_archive
    assert rec["previous_status"] == "superseded"
    assert rec["paths"][str(adapter)] == str(Path(rec["archive_dir"]) / adapter.name)
    assert rec["adapter_dir"] == str(adapter)  # column snapshot for restore
    assert rec["gguf_path"] == str(gguf)
    assert rec["archived_at"]


def test_archive_collision_suffixes(home, registry):
    """Same artifact filenames from different dirs coexist in the archive."""
    m1 = home / "run-a" / "model.gguf"
    m2 = home / "run-b" / "model.gguf"
    m1.parent.mkdir(parents=True); m2.parent.mkdir(parents=True)
    m1.write_bytes(b"one"); m2.write_bytes(b"two")

    registry.register_model("base-a")
    registry.set_validation("v1", {}, passed=False)  # failed -> archivable
    registry.set_artifacts("v1", gguf_path=str(m1))
    registry.register_model("base-a")
    registry.set_validation("v2", {}, passed=False)
    registry.set_artifacts("v2", gguf_path=str(m2))

    a1 = registry.archive_model("v1", home=home)
    a2 = registry.archive_model("v2", home=home)
    assert a1.pre_archive["archive_dir"] != a2.pre_archive["archive_dir"]
    assert a2.pre_archive["archive_dir"].endswith("v2-2") or \
        Path(a2.pre_archive["archive_dir"]).name == "v2"
    # distinct archived copies, contents intact
    dests = {p for p in a1.pre_archive["paths"].values()} | \
            {p for p in a2.pre_archive["paths"].values()}
    assert len(dests) == 2
    contents = sorted(Path(d).read_bytes() for d in dests)
    assert contents == [b"one", b"two"]


def test_archive_same_id_twice_after_restore(home, registry):
    mid, _ = _superseded(registry)
    _, gguf = _make_artifacts(home, registry, mid)
    first = registry.archive_model(mid, home=home)
    # Keep the first archive dir non-empty (foreign file) so restore cannot
    # tidy it and the second archive must collide-suffix.
    (Path(first.pre_archive["archive_dir"]) / "stray").write_bytes(b"x")
    registry.restore_model(mid)
    second = registry.archive_model(mid, home=home)
    assert first.pre_archive["archive_dir"] != second.pre_archive["archive_dir"]
    assert Path(second.pre_archive["archive_dir"]).name == f"{mid}-2"
    assert (Path(first.pre_archive["archive_dir"]) / "stray").exists()


def test_archive_skips_gguf_nested_in_adapter_dir(home, registry):
    mid, _ = _superseded(registry)
    adapter, gguf = _make_artifacts(home, registry, mid)
    nested = adapter / "nested.gguf"
    nested.write_bytes(b"N")
    registry.set_artifacts(mid, gguf_path=str(nested))
    m = registry.archive_model(mid, home=home)
    # only the adapter dir moved; the nested gguf rode along inside it
    assert len(m.pre_archive["paths"]) == 1
    archived_adapter = Path(next(iter(m.pre_archive["paths"].values())))
    assert (archived_adapter / "nested.gguf").read_bytes() == b"N"


def test_archive_refusals(home, registry):
    # candidate
    registry.register_model("base-a")
    with pytest.raises(RegistryError, match="candidate"):
        registry.archive_model("v1", home=home)

    # validated (not terminal)
    registry.set_validation("v1", {}, passed=True)
    with pytest.raises(RegistryError, match="supersede"):
        registry.archive_model("v1", home=home)

    # promoted / active
    registry.promote("v1")
    with pytest.raises(RegistryError, match="promoted"):
        registry.archive_model("v1", home=home)

    # pinned (terminal but protected)
    registry.register_model("base-a")
    registry.set_validation("v2", {}, passed=True)
    registry.promote("v2")  # v1 -> superseded, archivable now
    registry.set_pinned("v1", True)
    with pytest.raises(RegistryError, match="pinned"):
        registry.archive_model("v1", home=home)

    # unknown id / already archived
    with pytest.raises(RegistryError, match="no model v99"):
        registry.archive_model("v99", home=home)
    registry.set_pinned("v1", False)
    registry.archive_model("v1", home=home)
    with pytest.raises(RegistryError, match="already archived"):
        registry.archive_model("v1", home=home)


def test_archive_missing_artifact_refused(home, registry):
    mid, _ = _superseded(registry)
    registry.set_artifacts(mid, gguf_path=str(home / "nowhere.gguf"))
    with pytest.raises(RegistryError, match="missing"):
        registry.archive_model(mid, home=home)
    assert registry.get_model(mid).status == "superseded"  # row untouched


def test_restore_refusals(home, registry):
    mid, _ = _superseded(registry)
    _, gguf = _make_artifacts(home, registry, mid)

    # not archived yet
    with pytest.raises(RegistryError, match="only archived"):
        registry.restore_model(mid)

    registry.archive_model(mid, home=home)

    # target path occupied -> refuse, archive stays intact
    gguf.parent.mkdir(parents=True, exist_ok=True)
    gguf.write_bytes(b"impostor")
    with pytest.raises(RegistryError, match="already exists"):
        registry.restore_model(mid)
    assert Path(registry.get_model(mid).pre_archive["paths"][str(gguf)]).exists()

    # tampered archive (file gone) -> refuse
    gguf.unlink()
    archived = Path(json.loads(
        registry._conn.execute("SELECT pre_archive_paths FROM models "
                               "WHERE model_id = ?", (mid,)).fetchone()[0]
    )["paths"][str(gguf)])
    archived.unlink()
    with pytest.raises(RegistryError, match="missing"):
        registry.restore_model(mid)


# ------------------------------------------------------------------ pin

def test_pin_unpin_roundtrip(registry):
    registry.register_model("base-a")
    assert registry.get_model("v1").pinned is False
    m = registry.set_pinned("v1", True)
    assert m.pinned is True
    m = registry.set_pinned("v1", False)
    assert m.pinned is False
    with pytest.raises(RegistryError, match="no model"):
        registry.set_pinned("v99", True)


# ---------------------------------------------------------------- janitor

def test_janitor_archives_only_old_terminal_unpinned(home, registry):
    def new_model(base):
        m = registry.register_model(base)
        return m.model_id

    # old superseded (promoted_at aged) -> archive
    s1, _ = _superseded(registry)
    _make_artifacts(home, registry, s1)
    _age(registry, s1, 40)

    # old failed (created_at aged, never promoted) -> archive
    f1 = new_model("base-a")
    registry.set_validation(f1, {}, passed=False)
    _age(registry, f1, 40, promoted=False)
    _make_artifacts(home, registry, f1)

    # old but pinned -> skip
    s2, _ = _superseded(registry, base="base-b")
    _make_artifacts(home, registry, s2)
    _age(registry, s2, 40)
    registry.set_pinned(s2, True)

    # recent superseded -> skip
    s3, _ = _superseded(registry, base="base-c")
    _make_artifacts(home, registry, s3)
    _age(registry, s3, 5)

    # promoted / candidate never touched even when ancient
    p1 = new_model("base-d")
    registry.set_validation(p1, {}, passed=True)
    registry.promote(p1)
    _age(registry, p1, 400)
    c1 = new_model("base-e")
    _age(registry, c1, 400, promoted=False)

    summary = run_janitor(home, days=30, registry=registry)
    assert sorted(summary["archived"]) == sorted([s1, f1])
    assert summary["freed_bytes"] > 0 and not summary["errors"]
    statuses = {m.model_id: m.status for m in registry.list_models()}
    assert statuses[s2] != "archived" and statuses[s3] != "archived"
    assert statuses[p1] == "promoted" and statuses[c1] == "candidate"


def test_janitor_skips_recently_used_models(home, registry):
    """Pass 4 #29: among equally-old terminal models, a recently-used one is
    spared while the never/long-unused one is archived."""
    # two equally old superseded models, same base so both age identically
    used, _ = _superseded(registry)              # older row ends superseded
    unused, _ = _superseded(registry, base="base-b")
    _make_artifacts(home, registry, used)
    _make_artifacts(home, registry, unused)
    _age(registry, used, 40)
    _age(registry, unused, 40)
    # 'used' was activated yesterday (inside the 30-day window); 'unused'
    # was last touched before the window -> only it gets archived.
    _set_last_used(registry, used, 1)
    _set_last_used(registry, unused, 60)

    summary = run_janitor(home, days=30, registry=registry)
    assert summary["archived"] == [unused]
    statuses = {m.model_id: m.status for m in registry.list_models()}
    assert statuses[used] == "superseded"
    assert statuses[unused] == "archived"


def test_janitor_never_used_model_still_archives(home, registry):
    """last_used_at None means 'no usage signal' -> age rules decide (#29)."""
    mid, _ = _superseded(registry)
    _make_artifacts(home, registry, mid)
    _age(registry, mid, 40)
    assert registry.get_model(mid).last_used_at is None

    summary = run_janitor(home, days=30, registry=registry)
    assert summary["archived"] == [mid]


def test_record_activation_spares_model_from_janitor(home, registry):
    """End-to-end: record_activation on an aged terminal model keeps it live."""
    mid, _ = _superseded(registry)
    _make_artifacts(home, registry, mid)
    _age(registry, mid, 40)

    assert run_janitor(home, days=30, dry_run=True,
                       registry=registry)["archived"] == [mid]
    assert registry.record_activation(mid) is True
    assert run_janitor(home, days=30, registry=registry)["archived"] == []
    assert registry.get_model(mid).status == "superseded"


def test_janitor_state_file_run_count(home, registry):
    mid, _ = _superseded(registry)
    _make_artifacts(home, registry, mid)
    _age(registry, mid, 40)

    state_path = home / "janitor_state.json"
    assert not state_path.exists()
    run_janitor(home, days=30, registry=registry)
    st = load_state(home)
    assert st["run_count"] == 1 and st["last_run_at"]
    assert set(st["last_summary"]) == {"archived", "freed_bytes"}
    run_janitor(home, days=30, registry=registry)
    assert load_state(home)["run_count"] == 2


def test_janitor_dry_run_touches_nothing(home, registry):
    mid, _ = _superseded(registry)
    adapter, gguf = _make_artifacts(home, registry, mid)
    _age(registry, mid, 40)

    summary = run_janitor(home, days=30, dry_run=True, registry=registry)
    assert summary["archived"] == [mid] and summary["freed_bytes"] > 0
    assert registry.get_model(mid).status == "superseded"
    assert gguf.exists() and adapter.exists()
    assert not (home / "archive").exists()
    assert not (home / "janitor_state.json").exists()


def test_janitor_config_parsing():
    jc = janitor_mod.janitor_config_from_yaml(None)
    assert jc.enabled is False and jc.retention_days == 30  # conservative default
    jc = janitor_mod.janitor_config_from_yaml(
        {"enabled": True, "retention_days": "14"})
    assert jc.enabled is True and jc.retention_days == 14
    jc = janitor_mod.janitor_config_from_yaml({"retention_days": "bogus"})
    assert jc.retention_days == 30


# ------------------------------------------------------------ deploy guard

def test_deploy_rejects_archived_model(home, registry):
    from sable.serving.deploy import deploy_model
    from sable.core.config import SableConfig

    mid, _ = _superseded(registry)
    _, gguf = _make_artifacts(home, registry, mid)
    registry.set_artifacts(mid, gguf_path=str(gguf))
    registry.archive_model(mid, home=home)

    with pytest.raises(ServeError, match="archived"):
        deploy_model(home, registry, mid, SableConfig())


# ------------------------------------------------- auto-check cron hook

def _auto_check(tmp_path, monkeypatch, janitor_cfg):
    """Run `sable train auto-check` with a config carrying a janitor section."""
    import sable.registry.janitor as jm

    calls: list = []

    def fake_run_janitor(home, days=30, dry_run=False, registry=None,
                         actor="cli", disk_days=90):
        calls.append({"home": home, "days": days, "actor": actor,
                      "disk_days": disk_days})
        return {"archived": ["vX"], "freed_bytes": 1, "dry_run": False, "errors": []}

    monkeypatch.setattr(jm, "run_janitor", fake_run_janitor)
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text("janitor:\n" + janitor_cfg)
    rc = cli_main(["train", "auto-check", "--config", str(cfg_file)])
    return calls, rc


def test_auto_check_runs_janitor_when_enabled(tmp_path, monkeypatch):
    calls, _ = _auto_check(tmp_path, monkeypatch, "  enabled: true\n  retention_days: 7\n")
    assert len(calls) == 1 and calls[0]["days"] == 7
    # pass4 #31: scheduled archives stamp the ledger as scheduler-driven
    assert calls[0]["actor"] == "scheduler"


def test_auto_check_skips_janitor_when_disabled(tmp_path, monkeypatch):
    calls, _ = _auto_check(tmp_path, monkeypatch, "  enabled: false\n")
    assert calls == []


def test_auto_check_survives_janitor_crash(tmp_path, monkeypatch):
    import sable.registry.janitor as jm

    def boom(*a, **k):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(jm, "run_janitor", boom)
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text("janitor:\n  enabled: true\n")
    # must not raise out of the cron entrypoint
    cli_main(["train", "auto-check", "--config", str(cfg_file)])


# --------------------------------------------------------------- CLI

def test_cli_archive_restore_pin_flow(home, registry, capsys):
    mid, _ = _superseded(registry)
    _, gguf = _make_artifacts(home, registry, mid)
    registry.set_artifacts(mid, gguf_path=str(gguf))

    assert cli_main(["models", "pin", mid]) == 0
    assert "pinned" in capsys.readouterr().out
    # pinned -> archive refused with non-zero exit
    assert cli_main(["models", "archive", mid]) == 1
    assert "pinned" in capsys.readouterr().err

    assert cli_main(["models", "unpin", mid]) == 0
    assert cli_main(["models", "archive", mid]) == 0
    out = capsys.readouterr().out
    assert "archived v1" in out and "restore" in out
    assert not gguf.exists()

    assert cli_main(["models", "restore", mid]) == 0
    assert gguf.exists()

    # janitor CLI dry-run lists nothing left to do, exits 0
    assert cli_main(["models", "janitor", "--dry-run"]) == 0


def test_cli_models_list_shows_pinned_marker(home, registry, capsys):
    registry.register_model("base-a")
    registry.set_pinned("v1", True)
    assert cli_main(["models", "list"]) == 0
    assert "[pinned]" in capsys.readouterr().out


# ------------------------------------------------------ migration v4 -> v5

def test_migration_v4_to_v5(home):
    db = home / "buffer" / "sable.db"
    conn = sqlite3.connect(db)
    # Hand-built pre-v5 table: no pinned/pre_archive_paths, old CHECK.
    conn.execute("""
        CREATE TABLE models (
            model_id        TEXT PRIMARY KEY,
            base_model      TEXT NOT NULL,
            parent_adapter  TEXT,
            dataset_id      TEXT,
            manifest_hash   TEXT,
            adapter_dir     TEXT,
            gguf_path       TEXT,
            quant           TEXT,
            status          TEXT NOT NULL CHECK (status IN
              ('candidate','validated','promoted','rolled_back','failed','superseded')),
            created_at      TEXT NOT NULL,
            validation_json TEXT,
            train_loss      REAL,
            eval_loss       REAL,
            promoted_at     TEXT
        )""")
    conn.execute(
        "INSERT INTO models (model_id, base_model, status, created_at, "
        "promoted_at) VALUES ('v1','base-a','superseded',?,?)",
        (_iso(100), _iso(90)),
    )
    conn.commit()
    conn.close()

    reg = ModelRegistry(db)
    try:
        m = reg.get_model("v1")
        assert m is not None and m.pinned is False and m.pre_archive is None
        # CHECK constraint rebuilt: 'archived' now writable end-to-end
        gguf = home / "old.gguf"
        gguf.write_bytes(b"legacy")
        reg.set_artifacts("v1", gguf_path=str(gguf))
        archived = reg.archive_model("v1", home=home)
        assert archived.status == "archived"
        restored = reg.restore_model("v1")
        assert restored.status == "superseded" and gguf.exists()
    finally:
        reg.close()

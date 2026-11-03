"""WP1: atomic JSON writes survive transient Windows sharing violations.

The chat-thread-loss bug was Windows-only because the fixed "<file>.json.tmp"
name collided under concurrent writers / AV handles (PermissionError
[WinError 32]) while Linux happily renames over open files. These tests pin
the fix: unique tmp names, a bounded retry that recovers when os.replace is
refused once and then succeeds, immediate propagation of non-transient
errors (crash-mid-write stays loud), and the doctor's stale-tmp diagnostic.
"""

from __future__ import annotations

import json
import os
import re
import threading

import pytest

from toaster_loop.doctor import check_chat_tmp, find_stale_chat_tmp
from toaster_loop.store.atomic import (atomic_write_json, atomic_write_text,
                                replace_with_retry, tmp_path_for)


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "uf-atomic"
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(h))
    return h


# ------------------------------------------------------------- unit: helper

def test_tmp_names_are_unique_per_write(tmp_path):
    p = tmp_path / "rec.json"
    names = {tmp_path_for(p).name for _ in range(20)}
    assert len(names) == 20  # no fixed name, no collisions
    n = tmp_path_for(p).name
    assert n.startswith("rec.json.") and n.endswith(".tmp")
    assert re.match(r"^rec\.json\.\d+\.[0-9a-f]{8}\.tmp$", n)


def test_replace_retry_recovers_from_one_permission_error(tmp_path,
                                                          monkeypatch):
    """Acceptance: os.replace raising PermissionError once then succeeding
    proves the retry loop recovers (the WinError-32 shape)."""
    src, dst = tmp_path / "a.tmp", tmp_path / "b.json"
    src.write_text("payload", encoding="utf-8")

    real_replace = os.replace
    calls = {"n": 0}

    def flaky(s, d):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermissionError(32, "The process cannot access the file "
                                      "because it is being used by another "
                                      "process")
        real_replace(s, d)

    monkeypatch.setattr(os, "replace", flaky)
    slept = []
    replace_with_retry(src, dst, sleep=slept.append)
    assert dst.read_text() == "payload"
    assert calls["n"] == 2 and slept == [0.05]


def test_replace_retry_gives_up_after_attempts(tmp_path, monkeypatch):
    src, dst = tmp_path / "a.tmp", tmp_path / "b.json"
    src.write_text("x", encoding="utf-8")

    def always_locked(s, d):
        raise PermissionError(32, "locked forever")

    monkeypatch.setattr(os, "replace", always_locked)
    slept = []
    with pytest.raises(PermissionError):
        replace_with_retry(src, dst, attempts=3, sleep=slept.append)
    assert len(slept) == 2  # backoff between, none after the last attempt


def test_replace_no_retry_on_non_transient(tmp_path, monkeypatch):
    """A genuine OSError (disk full etc.) must surface immediately — the
    existing crash-mid-write posture stays intact."""
    src, dst = tmp_path / "a.tmp", tmp_path / "b.json"
    src.write_text("x", encoding="utf-8")
    calls = {"n": 0}

    def boom(s, d):
        calls["n"] += 1
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        replace_with_retry(src, dst, sleep=lambda s: None)
    assert calls["n"] == 1


# ------------------------------------------------------------ unit: writers

def test_atomic_write_json_roundtrip_and_chmod(tmp_path):
    p = tmp_path / "sub" / "rec.json"
    atomic_write_json(p, {"a": "ünicode", "b": [1, 2]})
    assert json.loads(p.read_text(encoding="utf-8")) == {"a": "ünicode",
                                                          "b": [1, 2]}
    # tmp cleaned up even on success
    assert list(tmp_path.glob("*.tmp")) == []
    if os.name != "nt":
        assert p.stat().st_mode & 0o777 == 0o600


def test_atomic_write_text_leaves_no_tmp_on_failure(tmp_path, monkeypatch):
    def boom(s, d):
        raise PermissionError(32, "locked")

    monkeypatch.setattr(os, "replace", boom)
    p = tmp_path / "rec.json"
    with pytest.raises(PermissionError):
        atomic_write_text(p, "content")
    # The failed write's own tmp is cleaned up (doctor only sees true orphans)
    assert list(tmp_path.glob("*.tmp")) == []


def test_concurrent_writers_never_clobber(tmp_path):
    """20 threads hammering the same record: unique tmp names mean every
    replace lands; the final file is one complete JSON document."""
    p = tmp_path / "shared.json"
    atomic_write_json(p, {"i": -1})

    def writer(i):
        atomic_write_json(p, {"i": i, "blob": "x" * 2000})

    ts = [threading.Thread(target=writer, args=(i,)) for i in range(20)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    final = json.loads(p.read_text(encoding="utf-8"))
    assert isinstance(final.get("i"), int)
    assert list(tmp_path.glob("*.tmp")) == []


# ------------------------------------------------------- integration: chat

def test_thread_put_recovers_from_locked_replace(home, monkeypatch):
    """The WP1 acceptance shape, end to end through the real handler: the
    first os.replace is refused with WinError 32 (AV/indexer handle), the
    retry succeeds and the thread file is written."""
    from toaster_loop.core.config import ToasterLoopConfig
    from toaster_loop.dashboard.app import DashboardApp

    app = DashboardApp(ToasterLoopConfig(), home=home)
    r = app.handle("PUT", "/api/chat/threads/w1",
                   body={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status == 200

    real_replace = os.replace
    calls = {"n": 0}

    def flaky(s, d):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermissionError(32, "being used by another process")
        real_replace(s, d)

    monkeypatch.setattr(os, "replace", flaky)
    r = app.handle("PUT", "/api/chat/threads/w1",
                   body={"messages": [{"role": "user", "content": "second"}]})
    assert r.status == 200
    assert calls["n"] == 2
    t = app.handle("GET", "/api/chat/threads/w1").body["thread"]
    assert t["messages"][0]["content"] == "second"


# ------------------------------------------------------------ doctor checks

def test_doctor_flags_stale_chat_tmp(tmp_path):
    chat = tmp_path / "chat"
    chat.mkdir()
    (chat / "t1.json").write_text("{}")
    fresh = chat / "t2.json.1234.abcd1234.tmp"
    fresh.write_text("in-flight, fine")
    stale = chat / "old.json.tmp"
    stale.write_text("orphan")
    old = stale.stat().st_mtime - 3600
    os.utime(stale, (old, old))

    found = find_stale_chat_tmp(tmp_path)
    assert found == [stale]  # fresh in-flight tmp excluded by age

    res = check_chat_tmp(tmp_path)
    assert res.status == "warn"
    assert any("old.json.tmp" in ln for ln in res.lines)

    stale.unlink()
    assert check_chat_tmp(tmp_path).status == "ok"

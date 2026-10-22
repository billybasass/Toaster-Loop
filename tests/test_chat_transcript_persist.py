"""Server-side chat transcript persistence (the "reply vanished on navigation" bug).

Transcript persistence used to be client-only: the starting panel saved the
thread after its SSE stream ended. Navigating away aborted that stream, so a
finished run's reply was never written to <home>/chat/<id>.json — returning to
chat showed an empty thread while the server had long since finished ("the
agent stopped responding but the server was still running"). The worker now
persists the complete turn itself when the run finishes.

Covers: a completed streamed run writes user+assistant turns (with tool cards)
to the thread file even though no client ever PUT it; the stored profile drives
the dropdown on reopen; re-running the same prompt replaces rather than
duplicates the turn (an aborted panel may have saved the bare prompt or a
partial assistant rendering first); and runs without a thread id persist
nothing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fake_server import FakeOpenAIServer  # noqa: E402
from sable.core.config import SableConfig  # noqa: E402
from sable.dashboard.app import DashboardApp  # noqa: E402


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "uf-persist"
    monkeypatch.setenv("SABLE_HOME", str(h))
    return h


def make_app(home):
    return DashboardApp(SableConfig(), home=home)


def _fake_serve_state(home, url):
    home.mkdir(parents=True, exist_ok=True)
    (home / "serve_state.json").write_text(json.dumps({
        "mode": "attach", "base_url": url, "model_name": "fake-model",
        "attach_url": url.rsplit("/v1", 1)[0],
    }))


def _events(text):
    out = []
    for frame in text.split("\n\n"):
        kind = None
        data = None
        for line in frame.splitlines():
            if line.startswith("event: "):
                kind = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        if kind:
            out.append((kind, data or {}))
    return out


def _drain(r):
    """Consume the whole SSE stream (no client-side save follows)."""
    assert r.stream is not None
    return "".join(r.stream)


def _thread(home, tid):
    p = home / "chat" / f"{tid}.json"
    assert p.is_file(), "the finished run must persist the thread itself"
    return json.loads(p.read_text())


# ---------------------------------------------------------------- happy path

def test_finished_run_persists_transcript_without_client_save(home):
    """Drain the stream to completion and never PUT the thread: the file must
    still carry both turns (fake server answers with a tool loop + text)."""
    app = make_app(home)
    with FakeOpenAIServer() as fs:
        _fake_serve_state(home, fs.base_url)
        r = app.handle("POST", "/api/chat/stream",
                       body={"prompt": "read the file", "thread": "t-p1"})
        assert r.stream is not None
        text = "".join(r.stream)  # client fully drains; NO client save follows
    evs = dict((k, d) for k, d in _events(text))
    final_text = "".join(d.get("text", "")
                         for k, d in _events(text) if k == "message.delta")

    t = _thread(home, "t-p1")
    roles = [m["role"] for m in t["messages"]]
    assert roles == ["user", "assistant"], \
        f"expected exactly one user+assistant pair, got {roles}"
    assert t["messages"][0]["content"] == "read the file"
    assistant = t["messages"][1]
    assert final_text and final_text in assistant["content"]
    # Tool cards survive into the stored transcript.
    assert assistant.get("tools"), "tool calls must persist with the turn"
    assert evs["run.done"]["trace_id"] in (t.get("trace_ids") or [])


def test_persisted_profile_matches_run_selection(home):
    """The thread records the profile used so reopening restores the dropdown
    to that agent instead of default."""
    app = make_app(home)
    pr = app.handle("PUT", "/api/profiles/sable",
                    body={"system_prompt": "you are sable", "tools": []})
    assert pr.status == 200
    with FakeOpenAIServer() as fs:
        _fake_serve_state(home, fs.base_url)
        r = app.handle("POST", "/api/chat/stream",
                       body={"prompt": "hi sable", "thread": "t-prof",
                             "profile": "sable"})
        _drain(r)
    assert _thread(home, "t-prof").get("profile") == "sable"


# ------------------------------------------------------------ idempotent merge

def test_partial_client_save_is_not_duplicated(home):
    """An aborted panel may have saved the prompt (or a partial assistant turn
    marked partial:true) before navigating away. The server-side write for the
    same run must REPLACE that tail, not append a second copy."""
    app = make_app(home)
    tid = "t-dup"
    # Simulate the aborted panel's save: prompt + half-rendered partial reply.
    app.handle("PUT", f"/api/chat/threads/{tid}", body={"messages": [
        {"role": "user", "content": "read the file"},
        {"role": "assistant", "content": "half a re", "partial": True},
    ]})
    with FakeOpenAIServer() as fs:
        _fake_serve_state(home, fs.base_url)
        r = app.handle("POST", "/api/chat/stream",
                       body={"prompt": "read the file", "thread": tid})
        _drain(r)
    t = _thread(home, tid)
    roles = [m["role"] for m in t["messages"]]
    assert roles == ["user", "assistant"], f"duplicate turns: {roles}"
    assert not t["messages"][1].get("partial")
    assert "half a re" != t["messages"][1]["content"]


def test_prompt_only_save_replaced_in_place(home):
    """Panel saved just the user prompt before aborting: the server appends
    its assistant turn after that same user message (no second user msg)."""
    app = make_app(home)
    tid = "t-prompt"
    app.handle("PUT", f"/api/chat/threads/{tid}", body={"messages": [
        {"role": "user", "content": "read the file"}]})
    with FakeOpenAIServer() as fs:
        _fake_serve_state(home, fs.base_url)
        r = app.handle("POST", "/api/chat/stream",
                       body={"prompt": "read the file", "thread": tid})
        _drain(r)
    t = _thread(home, tid)
    assert [m["role"] for m in t["messages"]] == ["user", "assistant"]


def test_threadless_run_persists_nothing(home):
    app = make_app(home)
    with FakeOpenAIServer() as fs:
        _fake_serve_state(home, fs.base_url)
        _drain(app.handle("POST", "/api/chat/stream", body={"prompt": "go"}))
    assert not (home / "chat").exists() or not list(
        (home / "chat").glob("*.json"))

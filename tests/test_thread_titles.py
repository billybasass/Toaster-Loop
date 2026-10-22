"""Two-stage chat-thread titles (inspiration pass 3, item #18).

Stage 1 (derive_title) is pure and deterministic; stage 2 (generate_title)
takes an injected client fn; the provenance gate encodes derived < llm < user.
API level: PUT /api/chat/threads/<id> auto-derives, explicit titles are user
renames, and the daemon upgrade replaces a derived title when a completion
endpoint answers — but never over a user rename.
"""

from __future__ import annotations

import json
import sys
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fake_server import FakeOpenAIServer  # noqa: E402
from sable.core.titles import (  # noqa: E402
    DERIVED,
    LLM,
    USER,
    derive_title,
    generate_title,
    should_apply_title,
)
from sable.core.config import SableConfig  # noqa: E402
from sable.dashboard.app import DashboardApp  # noqa: E402


# ------------------------------------------------------------ stage 1

def test_derive_from_first_user_message():
    msgs = [{"role": "user", "content": "Fix the scheduler VRAM gate please"}]
    assert derive_title(msgs) == "Fix the scheduler VRAM gate please"


def test_derive_strips_pleasantries_and_markers():
    # Multiple stacked openers ("hey can you") collapse in repeated passes.
    assert derive_title([{"role": "user", "content": "hey can you  explain\nthis"}]) \
        == "explain this"
    assert derive_title([{"role": "user", "content": "please - fix the bug"}]) \
        == "fix the bug"
    assert derive_title([{"role": "user", "content": "- item one two three"}]) \
        == "item one two three"


def test_derive_truncates_on_word_boundary():
    long = "word " * 40
    t = derive_title([{"role": "user", "content": long}])
    assert len(t) <= 61 and t.endswith("…")
    assert not t.endswith(" …")


def test_derive_never_fails():
    assert derive_title([]) == "Untitled chat"
    assert derive_title([{"role": "user", "content": "   "}]) == "Untitled chat"
    assert derive_title([{"role": "assistant", "content": "hi"}]) == "Untitled chat"
    # A pure code-block opener still yields something stable.
    assert derive_title([{"role": "user", "content": "```\nprint(1)\n```"}]) \
        == "Untitled chat"


def test_derive_skips_nonstring_content():
    assert derive_title([{"role": "user", "content": None},
                         {"role": "user", "content": "real question here"}]) \
        == "real question here"


# ------------------------------------------------------------ stage 2

def _client(reply):
    return lambda msgs, model: reply


def test_generate_title_parses_json_object():
    out = generate_title(
        _client('{"title": "scheduler vram gating"}'), "m",
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}])
    assert out == "scheduler vram gating"


def test_generate_title_cleans_plain_reply():
    clean = lambda r: generate_title(  # noqa: E731
        _client(r), "m", [{"role": "user", "content": "hi"},
                          {"role": "assistant", "content": "ok"}])
    assert clean('Title: "VRAM gate tuning"') == "VRAM gate tuning"
    assert clean("```\nvram gate tuning\n```") == "vram gate tuning"
    assert clean('"quoted title"') == "quoted title"


def test_generate_title_failure_shapes():
    empty_msgs = generate_title(_client('{"title": "x"}'), "m", [])
    assert empty_msgs is None                       # nothing to title
    assert generate_title(_client(None), "m",
                          [{"role": "user", "content": "hi"}]) is None
    assert generate_title(_client('{"title": "  "}'), "m",
                          [{"role": "user", "content": "hi"}]) is None

    def boom(msgs, model):
        raise RuntimeError("endpoint down")
    assert generate_title(boom, "m",
                          [{"role": "user", "content": "hi"}]) is None


# ------------------------------------------------------------- precedence

def test_precedence_derived_llm_user():
    assert should_apply_title(None, DERIVED)
    assert should_apply_title(DERIVED, LLM)
    assert should_apply_title(LLM, LLM)        # refresh allowed
    assert not should_apply_title(LLM, DERIVED)  # never downgrade
    assert not should_apply_title(USER, LLM)     # user rename is final
    assert not should_apply_title(USER, DERIVED)
    assert should_apply_title(USER, USER)


# ------------------------------------------------------------------- API

@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "uf-titles"
    monkeypatch.setenv("SABLE_HOME", str(h))
    return h


def make_app(home) -> DashboardApp:
    return DashboardApp(SableConfig(llama_auto_setup=False), home=home)


def _fake_serve_state(home, url):
    home.mkdir(parents=True, exist_ok=True)
    (home / "serve_state.json").write_text(json.dumps({
        "mode": "attach", "base_url": url, "model_name": "fake-model",
        "attach_url": url.rsplit("/v1", 1)[0],
    }))


class TitleJSONHandler(BaseHTTPRequestHandler):
    """Answers the stage-2 call with a JSON title object.

    Also serves GET /v1/models so the app's serve-alive probe (which gates
    _resolve_chat_target) treats this fake as a live endpoint.
    """

    def log_message(self, format, *args):  # noqa: A002 - silence access log
        pass

    def do_GET(self):
        if self.path.rstrip("/") == "/v1/models":
            body = json.dumps({"object": "list",
                               "data": [{"id": "fake-model", "object": "model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        body = json.dumps({
            "id": "x", "object": "chat.completion", "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant",
                                     "content": '{"title": "vram gate design"}'}}],
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _thread_file(home, tid):
    return json.loads((home / "chat" / f"{tid}.json").read_text())


def test_put_derives_title_instantly(home):
    app = make_app(home)
    r = app.handle("PUT", "/api/chat/threads/t1", body={
        "messages": [{"role": "user", "content": "how do I quantize a 70B model"}]})
    assert r.status == 200
    t = r.body["thread"]
    assert t["title"] == "how do I quantize a 70B model"
    assert t["title_source"] == "derived"


def test_thread_file_is_owner_only(home):
    """The durable thread file lands 0600 regardless of umask (transcripts can
    contain pasted secrets; matches the serve-state hardening)."""
    app = make_app(home)
    app.handle("PUT", "/api/chat/threads/t7", body={
        "messages": [{"role": "user", "content": "here is my key sk-abc123"}]})
    mode = (home / "chat" / "t7.json").stat().st_mode & 0o777
    assert mode == 0o600


def test_list_rows_carry_title_source(home):
    app = make_app(home)
    app.handle("PUT", "/api/chat/threads/t2", body={
        "messages": [{"role": "user", "content": "quantize a model tonight"}]})
    rows = app.handle("GET", "/api/chat/threads").body["threads"]
    row = [x for x in rows if x["id"] == "t2"][0]
    assert row["title"] == "quantize a model tonight"
    assert row["title_source"] == "derived"


def test_explicit_title_is_user_sourced(home):
    app = make_app(home)
    app.handle("PUT", "/api/chat/threads/t3", body={
        "messages": [{"role": "user", "content": "first words"}]})
    r = app.handle("PUT", "/api/chat/threads/t3", body={"title": "My important thread"})
    t = r.body["thread"]
    assert t["title"] == "My important thread"
    assert t["title_source"] == "user"


def test_stage2_upgrade_replaces_derived_title(home):
    app = make_app(home)
    with FakeOpenAIServer(handler_cls=TitleJSONHandler) as fs:
        _fake_serve_state(home, fs.base_url)
        app.handle("PUT", "/api/chat/threads/t4", body={
            "messages": [{"role": "user", "content": "tell me about vram gates"},
                         {"role": "assistant", "content": "sure"}]})
        # Daemon thread: poll briefly for the upgrade.
        t = None
        for _ in range(50):
            t = _thread_file(home, "t4")
            if t.get("title_source") == "llm":
                break
            time.sleep(0.1)
    assert t is not None
    assert t["title"] == "vram gate design"
    assert t["title_source"] == "llm"


def test_stage2_never_overrides_user_rename(home):
    app = make_app(home)
    with FakeOpenAIServer(handler_cls=TitleJSONHandler) as fs:
        _fake_serve_state(home, fs.base_url)
        # First save derives + schedules an upgrade; rename immediately.
        app.handle("PUT", "/api/chat/threads/t5", body={
            "messages": [{"role": "user", "content": "tell me about vram"},
                         {"role": "assistant", "content": "ok"}]})
        app.handle("PUT", "/api/chat/threads/t5", body={"title": "keep mine"})
        time.sleep(1.0)  # give any scheduled upgrade rope to hang itself
    t = _thread_file(home, "t5")
    assert t["title"] == "keep mine"
    assert t["title_source"] == "user"


def test_stage2_skipped_without_assistant_reply(home):
    app = make_app(home)
    with FakeOpenAIServer(handler_cls=TitleJSONHandler) as fs:
        _fake_serve_state(home, fs.base_url)
        app.handle("PUT", "/api/chat/threads/t6", body={
            "messages": [{"role": "user", "content": "one-shot question"}]})
        time.sleep(0.5)
    t = _thread_file(home, "t6")
    assert t["title_source"] == "derived"


# ------------------------------------- regression: client saves vs titles (#1)

def test_message_only_save_preserves_user_rename(home):
    """A save that omits `title` (the chat panel's auto-save shape) must never
    revert a double-click rename — the old client sent the first-message
    snippet on every save, silently clobbering renames."""
    app = make_app(home)
    app.handle("PUT", "/api/chat/threads/r1", body={
        "messages": [{"role": "user", "content": "original question"}]})
    app.handle("PUT", "/api/chat/threads/r1", body={"title": "renamed by hand"})
    # Auto-save with more messages, NO title key (post-fix chat.js shape).
    r = app.handle("PUT", "/api/chat/threads/r1", body={
        "messages": [{"role": "user", "content": "original question"},
                     {"role": "assistant", "content": "answer"},
                     {"role": "user", "content": "follow-up"}]})
    t = r.body["thread"]
    assert t["title"] == "renamed by hand"
    assert t["title_source"] == "user"
    assert len(t["messages"]) == 3


def test_message_only_save_preserves_llm_title(home):
    """Same contract for the stage-2 LLM title: an omitted title key leaves an
    llm-sourced title intact (previously overwritten + mislabeled USER)."""
    app = make_app(home)
    # Seed a thread whose title already carries llm provenance.
    d = home / "chat"
    d.mkdir(parents=True, exist_ok=True)
    (d / "r2.json").write_text(json.dumps({
        "id": "r2", "title": "vram gate design", "title_source": "llm",
        "created": "2026-01-01T00:00:00+00:00", "updated": "2026-01-01T00:00:00+00:00",
        "messages": [{"role": "user", "content": "tell me about vram gates"}],
    }))
    r = app.handle("PUT", "/api/chat/threads/r2", body={
        "messages": [{"role": "user", "content": "tell me about vram gates"},
                     {"role": "assistant", "content": "sure"}]})
    t = r.body["thread"]
    assert t["title"] == "vram gate design"
    assert t["title_source"] == "llm"


def test_message_only_save_preserves_derived_title(home):
    """Even a derived title stays put when messages change (no re-derivation
    churn once a thread has a label)."""
    app = make_app(home)
    r1 = app.handle("PUT", "/api/chat/threads/r3", body={
        "messages": [{"role": "user", "content": "first question here"}]})
    assert r1.body["thread"]["title"] == "first question here"
    r2 = app.handle("PUT", "/api/chat/threads/r3", body={
        "messages": [{"role": "user", "content": "totally different opener"},
                     {"role": "assistant", "content": "ok"}]})
    assert r2.body["thread"]["title"] == "first question here"


# ------------------------------------- regression: rename preserves messages (#1)

def test_rename_only_put_preserves_messages(home):
    """The double-click rename sends {title} alone; the thread body must ride
    through untouched (messages, trace ids, summary)."""
    app = make_app(home)
    app.handle("PUT", "/api/chat/threads/r4", body={
        "messages": [{"role": "user", "content": "hello there"},
                     {"role": "assistant", "content": "hi"}],
        "trace_ids": ["abc123"],
        "summary": "rolling summary text"})
    r = app.handle("PUT", "/api/chat/threads/r4", body={"title": "renamed"})
    t = r.body["thread"]
    assert t["title"] == "renamed" and t["title_source"] == "user"
    assert len(t["messages"]) == 2
    assert t["trace_ids"] == ["abc123"]
    assert t["summary"] == "rolling summary text"


# ------------------------------------- regression: atomic thread writes (#2)

def test_thread_write_is_atomic(home, monkeypatch):
    """A crash mid-write must never leave a partial file at the thread path:
    the write goes to *.json.tmp and lands via os.replace. We make os.replace
    raise (simulating death during the rename) and assert the previous file is
    byte-intact."""
    app = make_app(home)
    app.handle("PUT", "/api/chat/threads/a1", body={
        "messages": [{"role": "user", "content": "durable original"}]})
    good = (home / "chat" / "a1.json").read_bytes()

    import os as _os

    def boom(src, dst):  # crash exactly between tmp write and rename
        raise OSError("simulated crash mid-write")

    monkeypatch.setattr(_os, "replace", boom)
    # handle() funnels every handler exception into a 500 (no traceback leak).
    r = app.handle("PUT", "/api/chat/threads/a1", body={
        "messages": [{"role": "user", "content": "x" * 5000}]})
    assert r.status == 500
    # Original untouched; the half-written tmp never reached the read path.
    assert (home / "chat" / "a1.json").read_bytes() == good
    t = app.handle("GET", "/api/chat/threads/a1").body["thread"]
    assert t["messages"][0]["content"] == "durable original"

"""Chat run re-attachment — leave the page, the agent keeps going (the bug).

Previously the SSE generator owned the run: navigating away aborted the
stream, the generator's finally marked the run done, and returning to chat
showed a dead thread that "stopped responding". Runs are now worker-owned:
the HTTP stream is just one reader over a frame buffer. Leaving drops the
reader; GET /api/chat/runs/<id>/stream re-attaches and replays everything
produced while nobody was watching, then follows live. Multiple threads may
run concurrently (single-flight is per thread).

Covers: abandoning the starting stream mid-run does not stop the run (it
completes and persists); the in-flight registry carries the prompt for the
re-attaching view; re-attach replays buffered frames + live tail through
run.done; unknown run -> 404 no_such_run; two different threads can run at
the same time.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fake_server import FakeOpenAIServer  # noqa: E402
from sable.core.agent_runner import AgentResult  # noqa: E402
from sable.core.config import SableConfig  # noqa: E402
from sable.dashboard.app import DashboardApp  # noqa: E402


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "uf-reattach"
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


class GatedRunner:
    """create_runner stub: streams deltas only after `release` is set, so a
    test controls exactly when the run finishes relative to re-attachment."""

    def __init__(self, cfg, base_url, model, **kw):
        self.on_event = None          # dashboard attaches its sink here
        self.release = threading.Event()
        self.cancel = kw.get("cancel_event")

    def run(self, prompt, history=None, previous_summary=None):
        if self.on_event:
            self.on_event("message.delta", {"text": "half-"})
        self.release.wait(timeout=10)
        if self.on_event:
            self.on_event("message.delta", {"text": "done"})
        return AgentResult(final_response="half-done",
                           messages=[{"role": "assistant",
                                      "content": "half-done"}],
                           trace_id="tr-gate")


@pytest.fixture()
def gated(monkeypatch):
    made: list = []

    def _create(cfg, base_url, model, **kw):
        r = GatedRunner(cfg, base_url, model, **kw)
        made.append(r)
        return r

    monkeypatch.setattr("sable.core.agent_runner.create_runner", _create)
    return made


def _start(app, thread_id, prompt="go"):
    """POST /api/chat/stream and iterate ONE frame (stream admitted/run
    registered); returns the still-suspended generator."""
    resp = app.handle("POST", "/api/chat/stream",
                      body={"prompt": prompt, "thread": thread_id})
    assert resp.status == 200 and resp.stream is not None
    it = iter(resp.stream)
    first = next(it)          # run.started — registry entry now live
    assert "run.started" in first
    return it


# ------------------------------------------------ leave-mid-run resilience

def test_abandoning_the_stream_does_not_kill_the_run(home, gated):
    """Close the starting reader mid-run: the worker keeps going, finishes,
    persists the transcript, and only THEN marks the registry entry done."""
    app = make_app(home)
    with FakeOpenAIServer() as fs:
        _fake_serve_state(home, fs.base_url)
        it = _start(app, "t-leave")
        run_id = app.handle("GET", "/api/chat/runs").body["runs"][0]["run_id"]

        # Reader goes away mid-run (navigation): pure reader drop.
        it.close()

        runs = app.handle("GET", "/api/chat/runs").body["runs"]
        assert [r["run_id"] for r in runs] == [run_id], \
            "the run must still be in flight after its reader left"
        # The registry carries the prompt so a re-attaching view can render
        # the user turn (the transcript file only gets it at completion).
        assert runs[0]["prompt"] == "go"

        gated[0].release.set()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if app._chat_runs.get(run_id, {}).get("done"):
                break
            time.sleep(0.02)
        assert app._chat_runs[run_id]["done"] is True

        t = json.loads((home / "chat" / "t-leave.json").read_text())
        roles = [m["role"] for m in t["messages"]]
        assert roles == ["user", "assistant"]
        assert t["messages"][1]["content"] == "half-done"


def test_reattach_replays_buffered_frames_and_follows_live(home, gated):
    """A second reader (the returning panel) sees the deltas streamed while
    away, then the live tail, then run.done — and can watch again after."""
    app = make_app(home)
    with FakeOpenAIServer() as fs:
        _fake_serve_state(home, fs.base_url)
        it = _start(app, "t-back")
        it.close()                                # original reader leaves
        run_id = app.handle("GET", "/api/chat/runs").body["runs"][0]["run_id"]

        # Some deltas may have streamed while nobody was watching; the
        # re-attaching reader replays them from the buffer.
        att = app.handle("GET", f"/api/chat/runs/{run_id}/stream")
        assert att.status == 200 and att.stream is not None
        reader = iter(att.stream)

        def collect_until_done():
            frames = []
            for frame in reader:
                frames.append(frame)
                if "event: run.done" in frame or "event: run.error" in frame:
                    break
            return "".join(frames)

        # Release while the re-attached reader is parked on the cond var:
        # it must receive BOTH buffered and post-attach frames.
        threading.Timer(0.1, gated[0].release.set).start()
        text = collect_until_done()
        kinds = [k for k, _ in _events(text)]
        assert kinds[0] == "run.started"          # replay from the beginning
        deltas = "".join(d["text"] for k, d in _events(text)
                         if k == "message.delta")
        assert deltas == "half-done"              # buffered + live tail
        assert kinds[-1] == "run.done"

        # A second re-attach (another tab) still gets the finished buffer.
        again = app.handle("GET", f"/api/chat/runs/{run_id}/stream")
        assert again.status == 200 and again.stream is not None
        text2 = "".join(again.stream)
        assert "event: run.done" in text2


def test_reattach_unknown_run_is_404(home):
    app = make_app(home)
    r = app.handle("GET", "/api/chat/runs/nope/stream")
    assert r.status == 404
    assert json.loads(r.json())["error"]["code"] == "no_such_run"


# ------------------------------------------------------ concurrent threads

def test_two_threads_run_concurrently(home, gated):
    """Single-flight is PER thread: a second thread starts while the first
    is still executing; both stay in the registry."""
    app = make_app(home)
    with FakeOpenAIServer() as fs:
        _fake_serve_state(home, fs.base_url)
        it1 = _start(app, "t-a", prompt="first")
        it2 = _start(app, "t-b", prompt="second")   # must NOT 409
        assert it2 is not None
        runs = {r["thread"]: r for r in
                app.handle("GET", "/api/chat/runs").body["runs"]}
        assert set(runs) == {"t-a", "t-b"}
        assert runs["t-a"]["prompt"] == "first"
        assert runs["t-b"]["prompt"] == "second"

        # A THIRD post on a busy thread is still refused (one run per thread).
        busy = app.handle("POST", "/api/chat/stream",
                          body={"prompt": "x", "thread": "t-a"})
        assert busy.status == 409
        assert json.loads(busy.json())["error"]["code"] == "run_in_flight"

        for r in gated:
            r.release.set()
        it1.close()
        it2.close()

"""Image support (vision input) for the built-in agent.

Covers core/images.py encoding + reactive expansion, the view_image /
take_screenshot capability tools, runner-level multimodal wiring (markers
expand on the OUTBOUND request only — history/trace stay text), and the
dashboard chat upload path. All hermetic; conftest redirects SABLE_HOME.
"""

from __future__ import annotations

import base64
import struct
import sys
import zlib
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fake_server import FakeOpenAIServer  # noqa: E402

from sable.core import images as uf_images  # noqa: E402
from sable.core.agent_runner import (  # noqa: E402
    SableAgentClient,
    _resolve_vision_mode,
)
from sable.core.config import SableConfig  # noqa: E402
from sable.core.tools import view_image, take_screenshot  # noqa: E402


def png_bytes(w: int = 2, h: int = 2) -> bytes:
    """Smallest valid PNG (solid color), stdlib only."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data)))

    raw = b"".join(b"\x00" + b"\x7f\xff\x00" * w for _ in range(h))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b""))


@pytest.fixture()
def png(tmp_path) -> Path:
    p = tmp_path / "pic.png"
    p.write_bytes(png_bytes())
    return p


# ------------------------------------------------------------ encoding

def test_sniff_mime(png):
    assert uf_images.sniff_image_mime(png.read_bytes()) == "image/png"
    assert uf_images.sniff_image_mime(b"\xff\xd8\xffhello") == "image/jpeg"
    assert uf_images.sniff_image_mime(b"not an image at all") is None


def test_encode_image_file_roundtrip(png):
    url, err = uf_images.encode_image_file(str(png))
    assert err is None
    assert url is not None and url.startswith("data:image/png;base64,")
    raw = base64.b64decode(url.split(",", 1)[1])
    assert raw == png.read_bytes()


def test_encode_image_rejects_non_image(tmp_path):
    bad = tmp_path / "x.png"
    bad.write_text("plain text pretending to be png")
    url, err = uf_images.encode_image_file(str(bad))
    assert url is None
    assert err is not None and "not a recognized image" in err


def test_save_data_url_validates_and_writes(tmp_path):
    du = "data:image/png;base64," + base64.b64encode(png_bytes()).decode()
    path, err = uf_images.save_data_url(du, tmp_path / "uploads")
    assert err is None and path
    assert Path(path).read_bytes() == png_bytes()

    _p, err2 = uf_images.save_data_url("data:text/plain;base64,aGk=", tmp_path)
    assert err2 is not None
    _p, err3 = uf_images.save_data_url("not-a-data-url", tmp_path)
    assert err3 is not None


# ----------------------------------------------------------- expansion

def test_expand_message_images_rewrites_only_on_markers(png):
    plain = {"role": "user", "content": "hello"}
    out, n = uf_images.expand_message_images(plain)
    assert out is plain and n == 0

    marked = {"role": "tool",
              "content": f"look: {uf_images.attach_marker(str(png))}"}
    out, n = uf_images.expand_message_images(marked)
    assert n == 1
    assert isinstance(out["content"], list)
    assert out["content"][0]["type"] == "text"
    assert out["content"][1]["type"] == "image_url"
    assert out["content"][1]["image_url"]["url"].startswith(
        "data:image/png;base64,")
    # Original message untouched (history/trace keep the stub).
    assert isinstance(marked["content"], str)


def test_expand_degrades_on_missing_file():
    marked = {"role": "tool",
              "content": uf_images.attach_marker("/nonexistent/x.png")}
    out, n = uf_images.expand_message_images(marked)
    assert n == 0 and out is marked


# -------------------------------------------------------- vision tools

def test_view_image_returns_attach_marker(png):
    out = view_image(str(png), note="reference")
    assert out.startswith("[image attached:")
    assert "(see image)" in out and "reference" in out


def test_view_image_rejects_non_image(tmp_path):
    assert view_image(str(tmp_path)).startswith("error:")


def test_take_screenshot_graceful_without_tool(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _n: None)
    out = take_screenshot()
    assert out.startswith("error: no screenshot backend succeeded")
    assert "tried:" in out


# --------------------------------------------------------- capability gate

def _names(client):
    return {s.get("function", {}).get("name") for s in client.tools}


def test_vision_tools_gated_by_capabilities():
    from sable.core.agent_runner import SableAgentClient as C

    c = C(base_url="http://127.0.0.1:1/v1", model="m")
    assert not {"view_image", "take_screenshot"} & _names(c)

    c = C(base_url="http://127.0.0.1:1/v1", model="m",
          capabilities={"vision": True, "screenshots": True})
    assert {"view_image", "take_screenshot"} <= _names(c)
    assert {"view_image", "take_screenshot"} <= set(c.tool_impls)


class _FakeResp:
    def __init__(self, payload, status=200):
        self._p, self.status_code = payload, status

    def json(self):
        return self._p


class _FakeClient:
    def __init__(self, payload):
        self.payload = payload

    def get(self, url):
        return _FakeResp(self.payload)


def test_vision_probe_detects_multimodal_capability():
    """Current llama.cpp advertises the projector as capabilities
    ["completion","multimodal"] (not an mmproj meta key) — both shapes work."""
    uf_images.reset_vision_cache()
    modern = {"data": [{"id": "m", "capabilities": ["completion",
                                                    "multimodal"],
                        "meta": {}}]}
    assert uf_images.detect_vision_support(
        _FakeClient(modern), "http://x/v1") is True
    uf_images.reset_vision_cache()
    legacy = {"data": [{"id": "m", "meta": {"mmproj": "/x/mmproj.gguf"}}]}
    assert uf_images.detect_vision_support(
        _FakeClient(legacy), "http://x/v1") is True
    uf_images.reset_vision_cache()
    text_only = {"data": [{"id": "m", "capabilities": ["completion"],
                           "meta": {}}]}
    assert uf_images.detect_vision_support(
        _FakeClient(text_only), "http://x/v1") is False
    uf_images.reset_vision_cache()


def test_vision_mode_resolution():
    cfg = SableConfig()
    assert _resolve_vision_mode({}, cfg) == "auto"
    assert _resolve_vision_mode({"vision": True}, cfg) == "true"
    assert _resolve_vision_mode({"vision": False}, cfg) == "false"
    assert _resolve_vision_mode({"vision": "off"}, cfg) == "off"
    cfg2 = SableConfig(run_settings={"mmproj_path": "/x/mmproj.gguf"})
    assert _resolve_vision_mode({}, cfg2) == "true"


# --------------------------------------- runner end-to-end (fake server)

def test_runner_expands_markers_only_on_the_wire(tmp_path, monkeypatch):
    """With vision forced on, the REQUEST carries image parts while the
    recorded message list keeps the plain text marker."""
    png = tmp_path / "shot.png"
    png.write_bytes(png_bytes())
    captured: dict = {}

    with FakeOpenAIServer() as srv:
        client = SableAgentClient(
            base_url=srv.base_url, model="fake-model", vision="true")
        orig_post = client._client.post

        def spy(url, json=None, **kw):  # httpx passthrough spy
            if url.endswith("/chat/completions") and json is not None:
                captured["messages"] = json.get("messages")
            return orig_post(url, json=json, **kw)

        client._client.post = spy
        res = client.run(
            f"look at this {uf_images.attach_marker(str(png))}")
        assert res.completed
        # Wire: multimodal parts present.
        wire_msgs = captured["messages"]
        assert any(isinstance(m.get("content"), list)
                   and any(p.get("type") == "image_url" for p in m["content"])
                   for m in wire_msgs)
        # History/trace: still plain strings.
        assert all(isinstance(m.get("content"), str) or m.get("content") is None
                   for m in res.messages if m.get("role") != "tool"
                   or isinstance(m.get("content"), str))


def test_runner_text_only_when_vision_off(fake_server_unused=None):
    """Default (vision off / unreachable probe) => markers stay text."""
    with FakeOpenAIServer() as srv:
        client = SableAgentClient(base_url=srv.base_url,
                                    model="fake-model", vision="false")
        res = client.run("hello [image attached: /nope.png]")
        assert res.completed
        assert all(isinstance(m.get("content"), str) or m.get("content") is None
                   for m in res.messages)


# -------------------------------------------------- dashboard uploads

def test_chat_stream_saves_uploaded_images(tmp_path):
    from sable.dashboard.app import DashboardApp

    app = DashboardApp(SableConfig(), home=Path(str(tmp_path)))
    du = "data:image/png;base64," + base64.b64encode(png_bytes()).decode()
    # The handler validates + saves before touching the runner; with no
    # server running the run itself fails, but uploads must already exist.
    resp = None
    try:
        resp = app.handle("POST", "/api/chat/stream",
                          body={"prompt": "what is in this image?",
                                "images": [du]})
    except Exception:
        pass  # stream setup may raise without a live endpoint — fine here
    uploads = list((Path(str(tmp_path)) / "uploads").glob("upload-*.png"))
    assert uploads, "uploaded image was not persisted"
    assert uploads[0].read_bytes() == png_bytes()

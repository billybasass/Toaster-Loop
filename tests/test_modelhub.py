"""Model Hub (core/modelhub.py): search, variants, resumable download,
inventory and guarded deletion — against a local fake HF-hub server so the
real httpx paths run with zero network. Also covers dashboard route wiring.
"""

from __future__ import annotations

import gzip
import json
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sable.core import modelhub  # noqa: E402
from sable.core.config import SableConfig  # noqa: E402
from sable.core.modelhub import (  # noqa: E402
    HubError,
    ModelHub,
    validate_repo_id,
)
from sable.dashboard.app import DashboardApp  # noqa: E402

# ------------------------------------------------------------- fake hub API

REPO_SHA = "deadbeefcafe"
FILE_A = b"A" * (3 * 1024)          # small file
FILE_B = b"B" * (5 * 1024)          # resumed-download victim
SEARCH_RESULT = [
    {"id": "testorg/Test-GGUF", "downloads": 999, "likes": 3,
     "lastModified": "2026-01-02T00:00:00.000Z", "tags": ["gguf"]},
    {"id": "org/Plain", "downloads": 5, "likes": 0,
     "lastModified": "", "tags": ["safetensors"]},
]
DATASET_RESULT = [
    {"id": "testds/Test-Dataset", "downloads": 10, "likes": 2,
     "lastModified": "2026-04-01T00:00:00.000Z", "tags": ["size_categories:n<1K"]},
]
SPACE_RESULT = [
    {"id": "testspace/mcp-weather", "likes": 42,
     "createdAt": "2026-03-01T00:00:00.000Z", "sdk": "docker",
     "tags": ["mcp", "mcp-server"]},
]


class _FakeHubHandler(BaseHTTPRequestHandler):
    """Subset of the HF Hub REST API + resolve endpoint (Range aware)."""

    interrupt_first = False  # when True: first resolve request dies mid-stream

    def log_message(self, format, *args):  # noqa: A002 - silence access log
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/api/models":
            self._json(SEARCH_RESULT)
        elif p == "/api/datasets":
            self._json(DATASET_RESULT)
        elif p == "/api/spaces":
            self._json(SPACE_RESULT)
        elif p == "/api/datasets/testds/Test-Dataset":
            self._json({
                "id": "testds/Test-Dataset", "sha": "ds-sha",
                "author": "testds", "downloads": 10, "likes": 2,
                "lastModified": "2026-04-01T00:00:00.000Z",
                "cardData": {"license": "cc-by-4.0"},
                "siblings": [
                    {"rfilename": "train.jsonl", "size": len(FILE_A)},
                    {"rfilename": "README.md", "size": 9}],
            })
        elif p.startswith("/datasets/testds/Test-Dataset/resolve/"):
            rel = p.split("/resolve/")[1].split("/", 1)[1]
            data = FILE_A if rel == "train.jsonl" else b"# ds card"
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif p == "/datasets/testds/Test-Dataset/raw/main/README.md":
            body = b"# Test dataset\ncorpus for tests\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif p == "/api/spaces/testspace/mcp-weather":
            self._json({
                "id": "testspace/mcp-weather", "sha": "sp-sha",
                "author": "testspace", "likes": 42,
                "createdAt": "2026-03-01T00:00:00.000Z",
                "sdk": "docker", "tags": ["mcp", "mcp-server"],
                "cardData": {"license": "mit"},
                "siblings": [
                    {"rfilename": "app.py", "size": 3},
                    {"rfilename": "Dockerfile", "size": len(FILE_A)}],
            })
        elif p.startswith("/spaces/testspace/mcp-weather/resolve/"):
            rel = p.split("/resolve/")[1].split("/", 1)[1]
            data = b"py\n" if rel == "app.py" else FILE_A
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif p == "/spaces/testspace/mcp-weather/raw/main/README.md":
            body = b"# mcp-weather\nA weather MCP server.\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif p == "/testorg/Test-GGUF/raw/main/README.md":
            body = b"---\nlicense: apache-2.0\n---\n# Test card\nHello **world**\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif p == "/api/models/testorg/Test-GGUF":
            self._json({
                "id": "testorg/Test-GGUF", "sha": REPO_SHA,
                "author": "testorg", "downloads": 42, "likes": 7,
                "lastModified": "2026-05-01T00:00:00.000Z",
                "pipeline_tag": "text-generation", "gated": False,
                "cardData": {"license": "apache-2.0",
                             "base_model": "deepseek-ai/Test-Base"},
                "gguf": {"total": 123456, "architecture": "qwen3",
                         "context_length": 40960},
                "siblings": [
                    {"rfilename": "Q4_K_M/model-00001-of-00002.gguf",
                     "size": len(FILE_A)},
                    {"rfilename": "Q4_K_M/model-00002-of-00002.gguf",
                     "size": len(FILE_B)},
                    {"rfilename": "Q8_0/model.gguf", "size": 7},
                    {"rfilename": "mmproj-model.gguf", "size": 7},
                ],
            })
        elif p == "/api/models/org/Plain":
            self._json({"id": "org/Plain", "sha": "cafefood", "siblings": [
                {"rfilename": "config.json", "size": 2},
                {"rfilename": "model.safetensors", "size": 4},
                {"rfilename": "README.md", "size": len(FILE_A)}]})
        elif p.startswith("/org/Plain/resolve/"):
            # Mimic HF: small text files are served gzip'd WHEN the client
            # advertises gzip, with Content-Length describing the COMPRESSED
            # body (real-HF behaviour that killed non-GGUF downloads).
            rel = p.split("/resolve/")[1].split("/", 1)[1]
            data = {"README.md": FILE_A, "config.json": b"xy",
                    "model.safetensors": b"abcd"}.get(rel, b"")
            enc = "gzip" in (self.headers.get("Accept-Encoding") or "")
            if enc and rel == "README.md":
                data = gzip.compress(data)
                self.send_response(200)
                self.send_header("Content-Encoding", "gzip")
            else:
                self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif p.startswith("/testorg/Test-GGUF/resolve/"):
            rel = p.split("/resolve/")[1].split("/", 1)[1]
            data = FILE_A if "00001" in rel else (
                FILE_B if "00002" in rel else b"x" * 7)
            rng = self.headers.get("Range")
            start = 0
            if rng:
                start = int(rng.split("=")[1].split("-")[0])
            if start:
                self.send_response(206)
                self.send_header("Content-Range",
                                 f"bytes {start}-{len(data)-1}/{len(data)}")
            else:
                self.send_response(200)
            if type(self).interrupt_first and "model-00002" in rel \
                    and not getattr(self.server, "interrupted_once", False):
                # Abort mid-stream once (chunked + hard socket close) => the
                # client keeps the bytes already delivered in its .part.
                setattr(self.server, "interrupted_once", True)
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                body = data[start:start + 1024]
                self.wfile.write(b"%x\r\n" % len(body) + body + b"\r\n")
                self.wfile.flush()
                self.connection.setsockopt(
                    socket.SOL_SOCKET, socket.SO_LINGER,
                    b"\x01\x00\x00\x00\x00\x00\x00\x00")  # RST, no trailer
                self.close_connection = True
                return
            self.send_header("Content-Length", str(len(data) - start))
            if rel.lower().endswith(".png"):
                self.send_header("Content-Type", "image/png")
            self.end_headers()
            self.wfile.write(data[start:])
        elif p.startswith("/api/models/404"):
            self._json({"error": "not found"}, 404)
        else:
            self._json({"error": "nope"}, 404)


@pytest.fixture()
def hub_server(monkeypatch, tmp_path):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _FakeHubHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    monkeypatch.setenv("HF_ENDPOINT", f"http://127.0.0.1:{srv.server_address[1]}")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    yield srv
    srv.shutdown()


@pytest.fixture()
def hub(tmp_path, hub_server):
    # home matters: datasets/spaces roots live under <home>/hub.
    return ModelHub(models_dir=str(tmp_path / "models"),
                    home=tmp_path / "home")


# ---------------------------------------------------------------- validation

def test_validate_repo_id():
    assert validate_repo_id("testorg/Qwen3-8B-GGUF") == "testorg/Qwen3-8B-GGUF"
    # Leading/trailing slashes normalize away (harmless: paths stay anchored).
    assert validate_repo_id("/org/name/") == "org/name"
    for bad in ("", "..", "a/../b", "//", "or g/", "a\\b", "../x", ".hidden/x"):
        with pytest.raises(HubError):
            validate_repo_id(bad)


# -------------------------------------------------------------------- search

def test_search_annotates_gguf(hub_server, hub):
    out = hub.search("anything")
    assert [m["repo_id"] for m in out] == ["testorg/Test-GGUF", "org/Plain"]
    assert out[0]["gguf"] is True and out[1]["gguf"] is False
    assert out[0]["downloads"] == 999


# ------------------------------------------------------------- detail/feed

def test_detail_card_metadata(hub_server, hub):
    d = hub.detail("testorg/Test-GGUF")
    assert d["name"] == "Test-GGUF" and d["author"] == "testorg"
    assert d["architecture"] == "qwen3"
    assert d["context_length"] == 40960
    assert d["license"] == "apache-2.0"
    assert d["base_models"] == ["deepseek-ai/Test-Base"]
    assert d["pipeline_tag"] == "text-generation"
    # README fetched with the YAML frontmatter still present (client strips).
    assert "Hello **world**" in d["readme"]
    # Variants ride along for the detail pane's quant picker.
    assert {v["variant"] for v in d["variants"]} == {"Q4_K_M", "Q8_0"}


def test_latest_feed(hub_server, hub):
    out = hub.feed()
    assert [m["repo_id"] for m in out] == ["testorg/Test-GGUF", "org/Plain"]


# ------------------------------------------------------------------ variants

def test_variants_grouping(hub_server, hub):
    info = hub.variants("testorg/Test-GGUF")
    assert info["gguf"] and info["sha"] == REPO_SHA
    by_name = {v["variant"]: v for v in info["variants"]}
    q4 = by_name["Q4_K_M"]
    assert q4["shard_count"] == 2
    # mmproj is a companion, never its own variant: it rides along with EVERY
    # quant download (its size included), so served models always have the
    # projector on disk.
    assert "mmproj-model.gguf" in q4["files"]
    assert q4["size_bytes"] == len(FILE_A) + len(FILE_B) + 7
    assert "mmproj-model.gguf" in by_name["Q8_0"]["files"]
    assert by_name["Q8_0"]["size_bytes"] == 14
    assert not any("mmproj" in v["variant"] for v in info["variants"])
    # default prefers a Q4-family quant
    assert info["default_variant"] == "Q4_K_M"


def test_variants_mtp_companions_ride_along(hub_server, monkeypatch):
    """MTP drafter files (incl. an MTP/ directory) join every variant."""


    def fake_get(url, **kw):
        class R:
            status_code = 200
            def json(self):
                return {"id": "org/Mtp", "sha": "sha1", "siblings": [
                    {"rfilename": "UD-Q4/model-00001-of-00002.gguf",
                     "size": 10},
                    {"rfilename": "UD-Q4/model-00002-of-00002.gguf",
                     "size": 11},
                    {"rfilename": "mmproj-F16.gguf", "size": 3},
                    {"rfilename": "MTP/mtp-org-Mtp-Q8_0.gguf", "size": 5},
                ]}
        return R()

    monkeypatch.setattr(modelhub, "_get", fake_get)
    info = modelhub.list_variants("org/Mtp")
    assert len(info["variants"]) == 1
    v = info["variants"][0]
    assert "MTP/mtp-org-Mtp-Q8_0.gguf" in v["files"]
    assert "mmproj-F16.gguf" in v["files"]
    assert v["size_bytes"] == 10 + 11 + 3 + 5


def test_variants_non_gguf(hub_server, hub):
    info = hub.variants("org/Plain")
    assert not info["gguf"]
    assert info["variants"][0]["variant"] == "full"
    assert info["default_variant"] == "full"


def test_download_identity_encoding_non_gguf(hub_server, hub):
    """Non-GGUF download must land identity bytes.

    The fake server gzip-encodes README.md when the client advertises gzip
    (what HF really does for small text files). With iter_raw() writing the
    wire stream straight to disk, a gzip-capable request corrupts the file;
    _headers() must therefore pin Accept-Encoding: identity.
    """
    assert "identity" in modelhub._headers().get("Accept-Encoding", "")
    job = hub.download_sync("org/Plain", "full")
    assert job.state == "done", job.error
    snap = hub.snapshot_dir("org/Plain", "cafefood")
    assert (snap / "README.md").read_bytes() == FILE_A
    assert (snap / "config.json").read_bytes() == b"xy"


def test_download_rejects_gzip_part_residue(hub_server, hub):
    """A .part left by the gzip bug must not be resumed-into (append would
    concatenate identity bytes onto a gzip stream); it is restarted clean."""
    snap = hub.snapshot_dir("org/Plain", "cafefood")
    (snap / "README.md").parent.mkdir(parents=True, exist_ok=True)
    part = snap / "README.md.part"
    part.write_bytes(gzip.compress(b"stale-garbage")[:16])
    job = hub.download_sync("org/Plain", "full")
    assert job.state == "done", job.error
    assert (snap / "README.md").read_bytes() == FILE_A
    assert not part.exists()


def test_download_redownloads_gzip_poisoned_final(hub_server, hub):
    """A *final* file that is gzip poison must be re-fetched, not skipped by
    the size shortcut (real-HF corrupt .gitattributes was 1570 bytes of gzip
    that coincidentally matched its declared size).

    Two faces: size-coinciding poison (skippable -> must still re-download)
    and short poison (would fail the size check if it were a .part)."""
    snap = hub.snapshot_dir("org/Plain", "cafefood")
    snap.mkdir(parents=True, exist_ok=True)
    gz = gzip.compress(b"z" * len(FILE_A))
    # Poison whose size coincides with the declared size: size shortcut
    # would happily skip it.
    (snap / "README.md").write_bytes(gz[:len(FILE_A)].ljust(len(FILE_A), b"z"))
    job = hub.download_sync("org/Plain", "full")
    assert job.state == "done", job.error
    assert (snap / "README.md").read_bytes() == FILE_A
    # Short poison: must be purged, not resumed-into or treated as final.
    (snap / "README.md").write_bytes(gz[:16])
    job2 = hub.download_sync("org/Plain", "full")
    assert job2.state == "done", job2.error
    assert (snap / "README.md").read_bytes() == FILE_A


# ------------------------------------------------------------------ download

def test_download_sync_writes_hub_layout(hub_server, hub):
    job = hub.download_sync("testorg/Test-GGUF", "Q8_0")
    assert job.state == "done", job.error
    snap = hub.base_dir() / "models--testorg--Test-GGUF" / "snapshots" / REPO_SHA
    assert (snap / "Q8_0" / "model.gguf").read_bytes() == b"x" * 7
    assert (snap / "sable_download.json").is_file()
    assert (hub.base_dir() / "models--testorg--Test-GGUF" / "refs" / "main"
            ).read_text().strip() == REPO_SHA


def test_download_resumes_from_part(hub_server, hub):
    _FakeHubHandler.interrupt_first = True
    hub_server.interrupted_once = False
    try:
        job = hub.download_sync("testorg/Test-GGUF", "Q4_K_M")
        # First pass fails on the lied-about size; .part survives.
        assert job.state == "error"
        part = (hub.base_dir() / "models--testorg--Test-GGUF" / "snapshots"
                / REPO_SHA / "Q4_K_M" / "model-00002-of-00002.gguf.part")
        assert part.is_file() and part.stat().st_size == 1024
        # Retry resumes with Range and completes.
        job2 = hub.download_sync("testorg/Test-GGUF", "Q4_K_M")
        assert job2.state == "done", job2.error
        final = part.with_name(part.name[: -len(".part")])
        assert final.read_bytes() == FILE_B
    finally:
        _FakeHubHandler.interrupt_first = False


def test_download_unknown_variant(hub_server, hub):
    job = hub.download_sync("testorg/Test-GGUF", "Q99_NOPE")
    assert job.state == "error" and "unknown variant" in (job.error or "")


def test_download_not_found(hub_server, hub):
    job = hub.download_sync("nope/missing")
    assert job.state == "error" and "not found" in (job.error or "")


def test_start_download_job_and_single_flight(hub_server, hub):
    jid = hub.start_download("testorg/Test-GGUF", "Q8_0")
    jid2 = hub.start_download("testorg/Test-GGUF", "Q8_0")  # dedup while active
    # Wait for the worker to finish (tiny files).
    import time
    for _ in range(100):
        if hub.job(jid).state in ("done", "error"):
            break
        time.sleep(0.05)
    assert hub.job(jid).state == "done", hub.job(jid).error
    # After completion a new job id is issued (no longer in flight).
    jid3 = hub.start_download("testorg/Test-GGUF", "Q8_0")
    assert jid3 != jid
    assert len(hub.jobs()) >= 2


# ------------------------------------------------------------------ manage

def test_installed_and_delete_roundtrip(hub_server, hub):
    assert hub.installed() == []
    job = hub.download_sync("testorg/Test-GGUF", "Q8_0")
    assert job.state == "done"
    inst = hub.installed()
    assert len(inst) == 1
    assert inst[0]["repo_id"] == "testorg/Test-GGUF"
    assert inst[0]["size_bytes"] > 0
    assert "Q8_0" in inst[0]["quants"]

    out = hub.delete("testorg/Test-GGUF")
    assert out["freed_bytes"] > 0
    assert hub.installed() == []


def test_delete_missing_repo(hub):
    with pytest.raises(HubError, match="not downloaded"):
        hub.delete("never/seen")


def test_base_dir_fallback_to_home(tmp_path, monkeypatch):
    # No explicit dir + no existing HF cache anywhere => <home>/models.
    monkeypatch.delenv("SABLE_MODELS_DIR", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "nope"))  # nonexistent
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    h = ModelHub(models_dir=None, home=tmp_path / "home")
    assert h.base_dir() == tmp_path / "home" / "models"


def test_explicit_missing_models_dir_is_kept(tmp_path):
    # An explicit dir that doesn't exist yet must NOT fall back to the HF
    # cache — downloads create it on demand.
    target = tmp_path / "fresh" / "models"
    h = ModelHub(models_dir=str(target), home=tmp_path / "home")
    assert h.base_dir() == target


# ------------------------------------------------------------- dashboard API

@pytest.fixture()
def app(tmp_path, hub_server):
    cfg = SableConfig(models_dir=str(tmp_path / "models"))
    return DashboardApp(cfg, home=tmp_path / ".sable")


def test_dashboard_hub_routes(app):
    r = app.handle("GET", "/api/hub/search", query={"q": "test"})
    assert r.status == 200
    assert r.body["results"][0]["repo_id"] == "testorg/Test-GGUF"

    r = app.handle("GET", "/api/hub/variants", query={"repo": "org/Plain"})
    assert r.status == 200 and r.body["default_variant"] == "full"

    r = app.handle("POST", "/api/hub/download",
                   body={"repo": "testorg/Test-GGUF", "variant": "Q8_0"})
    assert r.status == 202
    job_id = r.body["job_id"]

    import time
    j = app.handle("GET", f"/api/hub/downloads/{job_id}")
    for _ in range(100):
        if j.body["state"] in ("done", "error"):
            break
        time.sleep(0.05)
        j = app.handle("GET", f"/api/hub/downloads/{job_id}")
    assert j.body["state"] == "done", j.body["error"]

    r = app.handle("GET", "/api/hub/installed")
    assert r.status == 200
    assert any(i["repo_id"] == "testorg/Test-GGUF" for i in r.body["installed"])

    r = app.handle("POST", "/api/hub/delete", body={"repo": "testorg/Test-GGUF"})
    assert r.status == 200 and r.body["freed_bytes"] > 0


def test_dashboard_hub_detail_and_feed_routes(app):
    r = app.handle("GET", "/api/hub/detail", query={"repo": "testorg/Test-GGUF"})
    assert r.status == 200
    assert r.body["architecture"] == "qwen3"
    assert r.body["base_models"] == ["deepseek-ai/Test-Base"]

    r = app.handle("GET", "/api/hub/feed")
    assert r.status == 200
    assert len(r.body["models"]) >= 1

    # Filtered search accepts the new params (fake hub ignores them).
    r = app.handle("GET", "/api/hub/search",
                   query={"q": "x", "gguf": "1", "capability": "vision",
                          "sort": "newest"})
    assert r.status == 200


def test_hub_image_proxy_and_card_image(hub_server, hub):
    from sable.core.modelhub import HubError as _HE
    from sable.core.modelhub import _first_card_image, fetch_image

    # Repo-relative path fetched through the configured endpoint.
    data, ctype = fetch_image(repo_id="testorg/Test-GGUF", path="a.png")
    assert ctype == "image/png" and data

    # Absolute URLs: https + public hosts only (SSRF screen, no network).
    with pytest.raises(_HE):
        fetch_image(url="https://127.0.0.1:9/x.png")
    with pytest.raises(_HE):
        fetch_image(url="http://cdn.example.com/x.png")
    with pytest.raises(_HE):
        fetch_image(repo_id="testorg/Test-GGUF", path="../secrets.png")

    # Card image extraction: first non-badge markdown image wins.
    md = ("[![badge](https://img.shields.io/x.svg)](y)\n"
          "![art](assets/banner.png)\n![](other.jpg)")
    assert _first_card_image(md) == "assets/banner.png"


def test_dashboard_hub_image_route(app, hub_server):
    r = app.handle("GET", "/api/hub/image",
                   query={"repo": "testorg/Test-GGUF", "path": "a.png"})
    assert r.status == 200 and r.raw and r.content_type == "image/png"
    # Private/loopback targets stay blocked (SSRF screen); arbitrary public
    # https hosts are now allowed so card art on third-party CDNs renders.
    r = app.handle("GET", "/api/hub/image",
                   query={"url": "https://127.0.0.1:9/x.png"})
    assert r.status == 400
    r = app.handle("GET", "/api/hub/image",
                   query={"url": "http://example.com/x.png"})  # not https
    assert r.status == 400


def test_dashboard_hub_errors(app):
    r = app.handle("GET", "/api/hub/variants", query={"repo": "../etc"})
    assert r.status == 400
    r = app.handle("GET", "/api/hub/variants", query={"repo": "api/nope"})
    assert r.status in (400, 404)
    r = app.handle("GET", "/api/hub/downloads/does-not-exist")
    assert r.status == 400


# ------------------------------------------------------------- kinds: datasets

def test_normalize_kind():
    from sable.core.modelhub import normalize_kind

    assert normalize_kind("") == "models"
    assert normalize_kind("model") == "models"
    assert normalize_kind("dataset") == "datasets"
    assert normalize_kind("mcp") == "spaces"      # MCP servers live in Spaces
    with pytest.raises(HubError):
        normalize_kind("widgets")


def test_search_datasets_and_spaces(hub_server, hub):
    ds = hub.search("corpus", kind="datasets")
    assert ds[0]["repo_id"] == "testds/Test-Dataset"
    assert ds[0]["kind"] == "datasets"

    sp = hub.search("", kind="spaces")
    assert sp[0]["repo_id"] == "testspace/mcp-weather"
    assert sp[0]["sdk"] == "docker"
    # Spaces have no download counter; likes still surface.
    assert sp[0]["likes"] == 42


def test_dataset_detail_and_download(hub_server, hub, tmp_path):
    d = hub.detail("testds/Test-Dataset", kind="datasets")
    assert d["kind"] == "datasets"
    assert d["variants"][0]["variant"] == "full"
    assert "corpus for tests" in d["readme"]

    job = hub.download_sync("testds/Test-Dataset", kind="datasets")
    assert job.state == "done", job.error
    # Datasets land under <home>/hub/datasets, never the models dir.
    ds_base = hub.kind_base("datasets")
    assert ds_base == tmp_path / "home" / "hub" / "datasets"
    snap = ds_base / "datasets--testds--Test-Dataset" / "snapshots" / "ds-sha"
    assert (snap / "train.jsonl").read_bytes() == FILE_A
    # Model discovery inventory must not see datasets.
    assert hub.installed("models") == []
    inst = hub.installed("datasets")
    assert inst and inst[0]["repo_id"] == "testds/Test-Dataset"

    out = hub.delete("testds/Test-Dataset", kind="datasets")
    assert out["freed_bytes"] > 0
    assert hub.installed("datasets") == []


def test_mcp_space_detail_and_download(hub_server, hub):
    d = hub.detail("testspace/mcp-weather", kind="spaces")
    assert d["kind"] == "spaces" and d["sdk"] == "docker"
    assert "weather MCP server" in d["readme"]

    job = hub.download_sync("testspace/mcp-weather", kind="spaces")
    assert job.state == "done", job.error
    snap = (hub.kind_base("spaces") / "spaces--testspace--mcp-weather"
            / "snapshots" / "sp-sha")
    assert (snap / "Dockerfile").read_bytes() == FILE_A

    # installed_all spans every kind.
    rows = hub.installed_all()
    assert {r["kind"] for r in rows} == {"spaces"}


def test_dashboard_hub_kind_routes(app):
    r = app.handle("GET", "/api/hub/search", query={"q": "x", "kind": "datasets"})
    assert r.status == 200
    assert r.body["results"][0]["repo_id"] == "testds/Test-Dataset"

    r = app.handle("GET", "/api/hub/feed", query={"kind": "spaces"})
    assert r.status == 200
    assert r.body["models"][0]["repo_id"] == "testspace/mcp-weather"

    r = app.handle("GET", "/api/hub/detail",
                   query={"repo": "testds/Test-Dataset", "kind": "datasets"})
    assert r.status == 200 and r.body["kind"] == "datasets"

    r = app.handle("POST", "/api/hub/download",
                   body={"repo": "testds/Test-Dataset", "kind": "datasets"})
    assert r.status == 202
    import time
    jid = r.body["job_id"]
    j = app.handle("GET", f"/api/hub/downloads/{jid}")
    for _ in range(100):
        if j.body["state"] in ("done", "error"):
            break
        time.sleep(0.05)
        j = app.handle("GET", f"/api/hub/downloads/{jid}")
    assert j.body["state"] == "done", j.body["error"]

    r = app.handle("POST", "/api/hub/delete",
                   body={"repo": "testds/Test-Dataset", "kind": "datasets"})
    assert r.status == 200 and r.body["freed_bytes"] > 0

    r = app.handle("GET", "/api/hub/search", query={"kind": "bogus"})
    assert r.status == 400


# ------------------------------------------------------------- PulseMCP dir

def test_mcp_dir_requires_key(monkeypatch):
    from sable.core.modelhub import HubError as _HE, search_mcp_dir

    monkeypatch.delenv("PULSEMCP_API_KEY", raising=False)
    with pytest.raises(_HE, match="PULSEMCP_API_KEY"):
        search_mcp_dir("github")


def test_mcp_dir_search_and_detail(monkeypatch):
    import httpx

    from sable.core.modelhub import mcp_dir_detail, search_mcp_dir

    monkeypatch.setenv("PULSEMCP_API_KEY", "test-key")
    seen = {}

    def fake_get(url, params=None, headers=None, **kw):
        seen["url"] = url
        seen["params"] = dict(params or {})
        seen["key"] = (headers or {}).get("X-API-Key")

        class R:
            status_code = 200

            def json(self_inner):
                if url.endswith("/servers"):
                    return {"data": [{"id": "github", "name": "GitHub",
                                      "author": "github",
                                      "description": "repos & PRs",
                                      "downloads": 1234,
                                      "github_stars": 900,
                                      "categories": ["Development"]},
                                     ]}
                return {"data": {"id": "github", "name": "GitHub",
                                 "readme": "# GitHub MCP",
                                 "transports": {"stdio": True,
                                                "streamable-http": False}}}
        return R()

    monkeypatch.setattr(httpx, "get", fake_get)
    rows = search_mcp_dir("git hub", limit=10)
    assert seen["key"] == "test-key"
    assert seen["params"]["search_query"] == "git hub"
    assert rows[0]["kind"] == "mcp-dir"
    assert rows[0]["url"].endswith("/servers/github")

    d = mcp_dir_detail("github")
    assert d["readme"] == "# GitHub MCP"
    assert d["transports"] == ["stdio"]
    with pytest.raises(modelhub.HubError):
        mcp_dir_detail("../etc/passwd")


def test_dashboard_mcp_dir_route(app, monkeypatch):
    # PulseMCP source without a key -> 400 with the signup hint; the default
    # registry source no longer touches that key at all.
    monkeypatch.delenv("PULSEMCP_API_KEY", raising=False)
    r = app.handle("GET", "/api/hub/mcp_dir",
                   query={"q": "x", "source": "pulsemcp"})
    assert r.status == 400 and "PULSEMCP_API_KEY" in r.body["error"]["message"]


def test_mcp_registry_search_and_detail(monkeypatch):
    import httpx

    from sable.core.modelhub import mcp_registry_detail, search_mcp_registry

    def fake_get(url, params=None, headers=None, **kw):
        class R:
            status_code = 200

            def json(self_inner):
                if url.endswith("/servers"):
                    return {"servers": [
                        {"server": {"name": "io.github.foo/weather",
                                    "title": "Weather",
                                    "description": "forecast server",
                                    "version": "1.2.0",
                                    "repository": {"url": "https://github.com/foo/weather"},
                                    "remotes": [{"type": "streamable-http",
                                                 "url": "https://x/mcp"}]}},
                        # older non-latest duplicate must be dropped
                        {"server": {"name": "io.github.bar/old",
                                    "version": "1.0.0"},
                         "_meta": {"io.modelcontextprotocol.registry/official":
                                   {"isLatest": False}}},
                    ]}
                return {"server": {"name": "io.github.foo/weather",
                                   "description": "forecast server",
                                   "version": "1.2.0",
                                   "packageVersions": [
                                       {"registryType": "npm",
                                        "identifier": "@foo/weather"}],
                                   "repository": {
                                       "url": "https://github.com/foo/weather"}}}
        return R()

    monkeypatch.setattr(httpx, "get", fake_get)
    rows = search_mcp_registry("weather", limit=10)
    assert len(rows) == 1  # non-latest row filtered
    assert rows[0]["kind"] == "mcp-registry"
    assert rows[0]["github_url"] == "https://github.com/foo/weather"
    assert rows[0]["transports"] == ["streamable-http"]

    d = mcp_registry_detail("io.github.foo/weather")
    assert "@foo/weather" in d["readme"]
    with pytest.raises(modelhub.HubError):
        mcp_registry_detail("../etc/passwd")


def test_dashboard_mcp_registry_route(app, monkeypatch):
    import httpx

    def fake_get(url, params=None, headers=None, **kw):
        class R:
            status_code = 200

            def json(self_inner):
                return {"servers": [{"server": {
                    "name": "io.github.foo/weather", "version": "1.0",
                    "description": "d"}}]}
        return R()

    monkeypatch.setattr(httpx, "get", fake_get)
    r = app.handle("GET", "/api/hub/mcp_dir", query={"q": "weather"})
    assert r.status == 200
    assert r.body["results"][0]["repo_id"] == "io.github.foo/weather"

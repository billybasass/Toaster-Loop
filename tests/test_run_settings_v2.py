"""Run-settings wave 2: speculative decoding, vision (mmproj) auto-detection +
download, and per-instance run settings that do NOT inherit the global
default-server section.

Covers:
  1. serving/run_args.py     — speculative_type -> --spec-type mapping,
                               sanitize rules for the new keys, mmproj
                               auto-detection via gguf_path + vision toggle.
  2. core/mmproj.py          — sibling detection, repo provenance (hub layout
                               + manifests), remote candidates + download into
                               the GGUF's own folder (fake hub server).
  3. serving/supervisor.py   — build_instance_cmd: named instances use ONLY
                               their own run_settings; "default" inherits the
                               global section.
  4. dashboard/app.py        — instance create validates run_settings; PUT on
                               the reserved default is rejected; /api/mmproj
                               status endpoint.

All hermetic: fake hub HTTP server + tmp dirs (conftest redirects SABLE_HOME).
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sable.core import mmproj  # noqa: E402
from sable.core.config import SableConfig  # noqa: E402
from sable.dashboard.app import DashboardApp  # noqa: E402
from sable.serving.run_args import (  # noqa: E402
    canonical_spec_mode, run_settings_args, sanitize_run_settings,
    speculative_args,
)
from sable.serving.supervisor import build_instance_cmd  # noqa: E402


def _has(args, *tokens):
    s = "\x00".join(args)
    return ("\x00".join(tokens)) in s


# ------------------------------------------------- 1. speculative decoding

def test_canonical_spec_mode_aliases():
    assert canonical_spec_mode("auto") == "auto"
    assert canonical_spec_mode("default") == "auto"
    assert canonical_spec_mode("none") == "off"
    assert canonical_spec_mode("draft-mtp") == "mtp"
    assert canonical_spec_mode("ngram-mod") == "ngram"
    assert canonical_spec_mode("mtp+ngram") == "ngram+mtp"
    assert canonical_spec_mode("ngram-mod,draft-mtp") == "ngram+mtp"
    assert canonical_spec_mode(None) is None
    assert canonical_spec_mode("garbage") is None


def test_speculative_args_emissions(tmp_path):
    drafter = tmp_path / "mtp-model-Q8_0.gguf"
    drafter.write_bytes(b"GGUF")
    assert speculative_args({}) == []
    assert speculative_args({"speculative_type": "off"}) == []
    assert speculative_args({"speculative_type": "auto"}) == \
        ["--spec-type", "ngram-mod"]
    # MTP modes need a drafter: explicit mtp_path -> --model-draft.
    a = speculative_args({"speculative_type": "mtp",
                          "mtp_path": str(drafter)})
    assert _has(a, "--spec-type", "draft-mtp")
    assert _has(a, "--model-draft", str(drafter))
    b = speculative_args({"speculative_type": "ngram+mtp",
                          "mtp_path": str(drafter),
                          "spec_draft_n_max": 3})
    assert _has(b, "--spec-type", "ngram-mod,draft-mtp")
    assert _has(b, "--model-draft", str(drafter))
    assert _has(b, "--spec-draft-n-max", "3")
    # No drafter anywhere (no mtp_path, no gguf sibling): degrade to ngram so
    # llama-server never exits early on a missing draft model.
    assert speculative_args({"speculative_type": "mtp"}) == \
        ["--spec-type", "ngram-mod"]
    assert speculative_args({"speculative_type": "mtp",
                             "mtp_path": str(tmp_path / "nope.gguf")}) == \
        ["--spec-type", "ngram-mod"]


def test_speculative_mtp_autodetects_sibling(tmp_path):
    gguf = tmp_path / "model-Q4_K_M.gguf"
    gguf.write_bytes(b"GGUF")
    mtp = tmp_path / "mtp-model-Q8_0.gguf"
    mtp.write_bytes(b"GGUF")
    a = speculative_args({"speculative_type": "mtp"}, gguf_path=str(gguf))
    assert _has(a, "--spec-type", "draft-mtp")
    assert _has(a, "--model-draft", str(mtp.resolve()))


def test_run_settings_args_includes_spec():
    args = run_settings_args({"speculative_type": "ngram"})
    assert _has(args, "--spec-type", "ngram-mod")


def test_sanitize_new_run_setting_keys():
    out = sanitize_run_settings({
        "vision": False, "speculative_type": "DRAFT-MTP",
        "spec_draft_n_max": 4})
    assert out["vision"] is False
    assert out["speculative_type"] == "mtp"
    assert out["spec_draft_n_max"] == 4
    with pytest.raises(ValueError):
        sanitize_run_settings({"speculative_type": "quantum"})
    with pytest.raises(ValueError):
        sanitize_run_settings({"spec_draft_n_max": 99})


# ------------------------------------------------- 2. mmproj detection

def test_find_local_sibling(tmp_path):
    gguf = tmp_path / "model-Q4_K_M.gguf"
    gguf.write_bytes(b"GGUF")
    assert mmproj.find_local(str(gguf)) is None
    proj = tmp_path / "mmproj-model-F16.gguf"
    proj.write_bytes(b"GGUF")
    assert mmproj.find_local(str(gguf)) == str(proj.resolve())
    # non-mmproj ggufs and other extensions don't count
    (tmp_path / "other.gguf").write_bytes(b"x")
    found = mmproj.find_local(str(gguf))
    assert found is not None and Path(found).name == proj.name


def test_is_mmproj_name():
    assert mmproj.is_mmproj_name("mmproj.gguf")
    assert mmproj.is_mmproj_name("mmproj-F16.gguf")
    assert mmproj.is_mmproj_name("Qwen3-VL-mmproj-BF16.gguf")
    assert not mmproj.is_mmproj_name("model-Q4_K_M.gguf")
    assert not mmproj.is_mmproj_name("mmproj.safetensors")


def test_resolve_mmproj_toggle_precedence(tmp_path):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"G")
    proj = tmp_path / "mmproj.gguf"
    proj.write_bytes(b"P")
    # default (no vision key): auto-detect the sibling
    assert mmproj.resolve_mmproj(str(gguf), {}) == str(proj.resolve())
    # explicit off wins
    assert mmproj.resolve_mmproj(str(gguf), {"vision": False}) is None
    # explicit manual path overrides detection
    other = tmp_path / "manual-mmproj.gguf"
    other.write_bytes(b"M")
    assert mmproj.resolve_mmproj(
        str(gguf), {"mmproj_path": str(other)}) == str(other.resolve())
    # dead manual path => absent, not a crash
    assert mmproj.resolve_mmproj(
        str(gguf), {"mmproj_path": str(tmp_path / "gone.gguf")}) is None


def test_source_repo_for_hub_layout_and_manifest(tmp_path):
    snap = (tmp_path / "models--testorg--Test-GGUF" / "snapshots" / "abc123")
    snap.mkdir(parents=True)
    g = snap / "model.gguf"
    g.write_bytes(b"G")
    assert mmproj.source_repo_for(str(g)) == "testorg/Test-GGUF"
    # loose file with a manifest
    loose = tmp_path / "loose"
    loose.mkdir()
    (loose / "m.gguf").write_bytes(b"G")
    (loose / "sable_download.json").write_text(
        json.dumps({"repo_id": "org/From-Manifest"}))
    assert mmproj.source_repo_for(str(loose / "m.gguf")) == "org/From-Manifest"
    # loose file, no provenance
    (tmp_path / "plain.gguf").write_bytes(b"G")
    assert mmproj.source_repo_for(str(tmp_path / "plain.gguf")) is None


# ---------------------------------------------------- fake hub + download

class _HubHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002 - silence access log
        pass

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/api/models/org/Vision-GGUF":
            body = json.dumps({
                "id": "org/Vision-GGUF", "sha": "cafe1234",
                "siblings": [
                    {"rfilename": "vision-Q4_K_M.gguf", "size": 4},
                    {"rfilename": "mmproj-BF16.gguf", "size": 5},
                ]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif p == "/org/Vision-GGUF/resolve/cafe1234/mmproj-BF16.gguf":
            body = b"MMPRJ"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()


@pytest.fixture()
def fake_hub(monkeypatch):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _HubHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    monkeypatch.setenv("HF_ENDPOINT", f"http://127.0.0.1:{port}")
    yield srv
    srv.shutdown()


def test_remote_candidates_and_download(tmp_path, fake_hub):
    snap = (tmp_path / "models--org--Vision-GGUF" / "snapshots" / "cafe1234")
    snap.mkdir(parents=True)
    gguf = snap / "vision-Q4_K_M.gguf"
    gguf.write_bytes(b"GGUF")

    cands = mmproj.remote_mmproj_candidates("org/Vision-GGUF")
    assert [c["rfilename"] for c in cands] == ["mmproj-BF16.gguf"]

    info = mmproj.download_for(str(gguf))
    # lands NEXT TO the GGUF (expected launcher behavior)
    assert Path(info["path"]).parent == snap.resolve()
    assert Path(info["path"]).name == "mmproj-BF16.gguf"
    assert Path(info["path"]).read_bytes() == b"MMPRJ"
    # now detection finds it
    assert mmproj.find_local(str(gguf)) == info["path"]


def test_download_requires_provenance(tmp_path):
    (tmp_path / "plain.gguf").write_bytes(b"G")
    with pytest.raises(ValueError):
        mmproj.download_for(str(tmp_path / "plain.gguf"))


# ------------------------------------- 3. per-instance non-inheritance

class _Cfg:
    run_settings = {"flash_attn": True, "kv_cache_type": "q8_0"}
    context_length = 4096
    gpu_layers = None
    extra_llama_args = []


def test_named_instance_does_not_inherit_global_run_settings():
    rt_gpus_none = None
    inst = {"id": "coding", "gguf": "/x/model.gguf",
            "run_settings": {"parallel": 4}}
    cmd = build_instance_cmd("llama-server", _Cfg(), inst, port=1234,
                             model_name="m", rt=rt_gpus_none)
    assert "--flash-attn" not in cmd          # global setting NOT inherited
    assert "-ctk" not in cmd
    assert _has(cmd, "-np", "4")              # own settings applied


def test_default_instance_inherits_global_run_settings():
    for ident in ({"id": "default"}, {"default": True}):
        inst = {**ident, "gguf": "/x/model.gguf"}
        cmd = build_instance_cmd("llama-server", _Cfg(), inst, port=1234,
                                 model_name="m", rt=None)
        assert "--flash-attn" in cmd


def test_instance_cmd_vision_toggle_and_spec(tmp_path):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"G")
    (tmp_path / "mmproj.gguf").write_bytes(b"P")
    inst = {"id": "vision", "gguf": str(gguf), "run_settings": {}}
    cmd = build_instance_cmd("llama-server", _Cfg(), inst, port=1,
                             model_name="m", rt=None)
    assert _has(cmd, "--mmproj", str((tmp_path / "mmproj.gguf").resolve()))
    inst2 = {"id": "textonly", "gguf": str(gguf),
             "run_settings": {"vision": False}}
    cmd2 = build_instance_cmd("llama-server", _Cfg(), inst2, port=1,
                              model_name="m", rt=None)
    assert "--mmproj" not in cmd2


# ------------------------------------------ 4. dashboard API behavior

@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "uf-rs2"
    monkeypatch.setenv("SABLE_HOME", str(h))
    return h


def make_app(home) -> DashboardApp:
    return DashboardApp(SableConfig(llama_auto_setup=False), home=home)


def test_instance_create_validates_run_settings(home):
    app = make_app(home)
    gguf = Path(home).parent / "m.gguf"
    gguf.write_bytes(b"G")
    r = app.handle("POST", "/api/servers", body={
        "name": "ok", "gguf": str(gguf),
        "run_settings": {"speculative_type": "auto", "vision": True}})
    assert r.status == 201
    row = next(s for s in app.handle("GET", "/api/servers").body["servers"]
               if s["name"] == "ok")
    assert row["run_settings"]["speculative_type"] == "auto"
    bad = app.handle("POST", "/api/servers", body={
        "name": "bad", "gguf": str(gguf),
        "run_settings": {"bogus_key": 1}})
    assert bad.status == 400


def test_default_instance_run_settings_put_rejected(home):
    app = make_app(home)
    r = app.handle("PUT", "/api/servers/default",
                   body={"run_settings": {"flash_attn": True}})
    assert r.status == 400


def test_mmproj_status_endpoint(tmp_path, monkeypatch):
    h = tmp_path / "uf-mm"
    monkeypatch.setenv("SABLE_HOME", str(h))
    app = DashboardApp(SableConfig(llama_auto_setup=False), home=h)
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"G")
    r = app.handle("GET", "/api/mmproj", query={"gguf": str(gguf)})
    assert r.body["local"] is None
    (tmp_path / "mmproj-x.gguf").write_bytes(b"P")
    r = app.handle("GET", "/api/mmproj", query={"gguf": str(gguf)})
    assert r.body["local"].endswith("mmproj-x.gguf")
    assert app.handle("GET", "/api/mmproj", query={}).status == 400


def test_config_put_accepts_new_run_setting_keys(tmp_path, monkeypatch):
    cfgp = tmp_path / "sable.yaml"
    monkeypatch.setenv("SABLE_CONFIG", str(cfgp))
    h = tmp_path / "uf-cfg"
    app = DashboardApp(SableConfig(llama_auto_setup=False), home=h)
    r = app.handle("PUT", "/api/config", body={
        "section": "run_settings",
        "patch": {"speculative_type": "auto", "vision": True,
                  "spec_draft_n_max": 3}})
    assert r.status == 200 and r.body["ok"]
    import yaml
    data = yaml.safe_load(cfgp.read_text())
    assert data["run_settings"]["speculative_type"] == "auto"
    assert data["run_settings"]["vision"] is True

"""WP3: datasets page — hub imports, folder/website curation, env gate.

Covers the acceptance shapes:
  * importer round-trips for a local folder (jsonl passthrough + doc->ShareGPT
    synthesis via the deterministic fallback) and HTML->rows crawling against
    a local fake site (robots.txt honored, same-origin bound);
  * an imported hub dataset appears in GET /api/datasets and is submittable
    to POST /api/train with the dataset paths flowing into job params;
  * the `datasets` env preflight/install commands target the training
    interpreter WITHOUT bitsandbytes (Windows [training]-extra independence).

The 429-then-200 hub retry test lives in tests/test_cli.py alongside the fake
hub server (test_download_retries_429_then_succeeds).
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from sable.core.config import SableConfig
from sable.curate import sources as src
from sable.curate.importer import ImportError_ as CurateImportError


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("SABLE_HOME", str(h))
    return h


# --------------------------------------------------------------- doc synth

def test_chunk_text_deterministic_paragraphs():
    text = "para one\n\npara two\n\n" + ("x" * 5000)
    chunks = src.chunk_text(text, chunk_chars=1000, overlap=100)
    assert chunks[0].startswith("para one")
    assert any(set(c) == {"x"} for c in chunks)
    # deterministic: same input+params -> identical chunks
    assert src.chunk_text(text, chunk_chars=1000, overlap=100) == chunks


def test_fallback_synth_heading_pairs():
    convs, warns = src.docs_to_rows(
        [("guide.md", "# T\n\n## Install\n" + "install steps here. " * 10)],
        synth_fn=None)
    assert convs and warns == []
    row = convs[0]
    assert row[0]["from"] == "human" and "Install" in row[0]["value"]
    assert row[1]["from"] == "gpt" and "install steps" in row[1]["value"]


def test_synth_failure_degrades_with_warning():
    def boom(name, chunk):
        raise RuntimeError("model down")

    convs, warns = src.docs_to_rows(
        [("a.md", "# H\n\n## S\n" + "content body. " * 10)],
        synth_fn=boom)
    assert convs, "fallback must still produce rows"
    assert any("model synthesis failed" in w for w in warns)


def test_parse_qa_blocks():
    text = "**Q:** What?\nA: This.\n\nQ: Two?\nA: **Bold** ok"
    pairs = src._parse_qa(text)
    assert pairs == [("What?", "This."), ("Two?", "**Bold** ok")]


# ---------------------------------------------------------------- folder

def test_ingest_folder_roundtrip(home, tmp_path):
    """Acceptance: folder import round-trips jsonl data files unchanged and
    converts prose docs through the synthesis pipeline into one dataset."""
    folder = tmp_path / "docs"
    (folder / "sub").mkdir(parents=True)
    # data file: sharegpt rows pass through importer.py untouched
    (folder / "data.jsonl").write_text(json.dumps(
        {"conversations": [{"from": "human", "value": "hi"},
                           {"from": "gpt", "value": "hello"}]}) + "\n")
    # prose doc: heading fallback synthesis
    (folder / "sub" / "guide.md").write_text(
        "# Guide\n\n## Setup\n" + "Run launch.ps1 to start the server. " * 5)

    res = src.ingest_folder(home, str(folder))
    assert res.rows == 2, "one passthrough row + one synthesized pair"

    ds_dir = home / "datasets" / res.dataset_id
    manifest = json.loads((ds_dir / "manifest.json").read_text())
    assert manifest["provenance"] == "local_folder"
    assert manifest["source_path"] == str(folder.resolve())
    assert manifest["files"]["sft.jsonl"]["count"] == 2
    rows = [json.loads(ln) for ln in
            (ds_dir / "sft.jsonl").read_text().splitlines()]
    # Passthrough row round-trips through the importer's normalize path
    # (messages_to_sharegpt may decorate values; content must survive).
    assert any("hello" in r["conversations"][1]["value"] for r in rows)
    assert any("Setup" in r["conversations"][0]["value"] for r in rows)
    # train.yaml materialized -> Train panel picker consumes it zero-config
    assert (ds_dir / "train.yaml").is_file()


def test_ingest_folder_rejects_missing(home):
    with pytest.raises(CurateImportError):
        src.ingest_folder(home, "/nonexistent/nope/dir")


# ---------------------------------------------------------------- website

PAGES = {
    "/": ("<html><head><title>Docs</title></head><body>"
          "<h1>Welcome</h1><p>" + "Comfy docs intro paragraph. " * 8
          + "</p><a href='/guide.html'>guide</a>"
          "<a href='https://elsewhere.example/x.html'>ext</a>"
          "</body></html>"),
    "/guide.html": ("<html><body><h2>Install</h2><p>"
                    + "Install steps for the toolkit. " * 8
                    + "</p></body></html>"),
}


class _SiteHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002 - silence access log
        pass

    def do_GET(self):
        p = self.path.split("?")[0]
        if p in PAGES:
            body = PAGES[p].encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif p == "/robots.txt":
            body = b"User-agent: *\nDisallow: /private\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()


@pytest.fixture()
def site():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _SiteHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_crawl_website_html_to_rows(home, site):
    """Acceptance: HTML -> text -> ShareGPT rows, same-origin bounded crawl,
    provenance records the crawled URLs (private net allowed ONLY because the
    test opts in explicitly — default posture refuses 127.0.0.1)."""
    res = src.crawl_website(home, site + "/", max_pages=5, max_depth=1,
                            sleep_s=0.0, allow_private_networks=True)
    ds_dir = home / "datasets" / res.dataset_id
    manifest = json.loads((ds_dir / "manifest.json").read_text())
    assert manifest["provenance"] == "website"
    assert manifest["source_url"].endswith("/")
    crawled = manifest["crawled_urls"]
    assert any(u.endswith("/guide.html") for u in crawled)
    assert not any("elsewhere.example" in u for u in crawled), \
        "same-origin bound enforced"
    rows = [json.loads(ln) for ln in
            (ds_dir / "sft.jsonl").read_text().splitlines()]
    assert rows, "synthesized rows from both pages"
    assert (ds_dir / "train.yaml").is_file()


def test_crawl_respects_robots_disallow(home, site, monkeypatch):
    # Entry page under a disallowed prefix -> refuse before fetching.
    with pytest.raises(CurateImportError) as ei:
        src.crawl_website(home, site + "/private/secret.html", sleep_s=0.0,
                          allow_private_networks=True)
    assert "robots.txt" in str(ei.value)


def test_crawl_refuses_private_by_default(home, site):
    with pytest.raises(CurateImportError):
        src.crawl_website(home, site + "/", sleep_s=0.0)


# ------------------------------------------------------------- hub import

def _make_hub_snapshot(home):
    snap = (home / "hub" / "datasets--testorg--Test-Dataset"
            / "snapshots" / "abc123")
    snap.mkdir(parents=True)
    # One file per format: importer.py sniffs a single format per file, so a
    # realistic hub snapshot keeps sharegpt and alpaca rows in separate files.
    (snap / "train.jsonl").write_text(json.dumps(
        {"conversations": [{"from": "human", "value": "q1"},
                           {"from": "gpt", "value": "a1"}]}) + "\n")
    (snap / "extra.jsonl").write_text(json.dumps(
        {"instruction": "q2", "input": "", "output": "a2"}) + "\n")
    return snap


def test_hub_import_lists_and_trains(home, monkeypatch):
    """Acceptance: import_hub snapshot -> listed by GET /api/datasets with hub
    provenance -> POST /api/train accepts it and the job params point at the
    materialized dataset dir (train.yaml + dataset_dir)."""
    from sable.dashboard.app import DashboardApp

    _make_hub_snapshot(home)
    app = DashboardApp(SableConfig(), home=home)

    r = app.handle("POST", "/api/datasets/import_hub",
                   body={"repo_id": "testorg/Test-Dataset"})
    assert r.status == 200, r.body
    ds_id = r.body["dataset_id"]
    assert r.body["rows"] == 2

    listed = {d["dataset_id"]: d for d in
              app.handle("GET", "/api/datasets").body["datasets"]}
    assert ds_id in listed
    assert listed[ds_id]["provenance"] == "hub"
    assert listed[ds_id]["source_repo"] == "testorg/Test-Dataset"

    # Train submission: capture what the scheduler would spawn with, without
    # a GPU worker (the params ARE the train/iterate contract).
    captured = {}

    def fake_run_claimed(self, job):
        captured["job"] = job
        return job

    monkeypatch.setattr("sable.training.scheduler.Scheduler.run_claimed",
                        fake_run_claimed)
    tr = app.handle("POST", "/api/train", body={"dataset": ds_id})
    assert tr.status == 202
    import time
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and "job" not in captured:
        time.sleep(0.02)
    job = captured["job"]
    params = json.loads(json.dumps(job.params))
    assert params["dataset_dir"].endswith(ds_id)
    assert (home / "datasets" / ds_id / "train.yaml").is_file()


def test_hub_import_missing_snapshot(home):
    from sable.dashboard.app import DashboardApp

    app = DashboardApp(SableConfig(), home=home)
    r = app.handle("POST", "/api/datasets/import_hub",
                   body={"repo_id": "org/never-downloaded"})
    assert r.status == 404


def test_hub_import_bad_repo_id(home):
    from sable.dashboard.app import DashboardApp

    app = DashboardApp(SableConfig(), home=home)
    r = app.handle("POST", "/api/datasets/import_hub",
                   body={"repo_id": "../escape"})
    assert r.status == 400


# ------------------------------------------------ curate_source endpoint

def test_curate_source_endpoint_queues_folder_job(home, monkeypatch):
    """202+poll contract (same as /api/curate); the done log line carries the
    dataset id so the UI can jump straight to it."""
    import time as _t

    from sable.dashboard.app import DashboardApp

    folder = home / "in"
    folder.mkdir()
    (folder / "note.md").write_text("# N\n\n## Sec\n" + "body words. " * 12)

    # No live server -> live_synth_fn None -> fallback synth inline.
    monkeypatch.setattr(src, "live_synth_fn", lambda app: None)
    app = DashboardApp(SableConfig(), home=home)

    r = app.handle("POST", "/api/datasets/curate_source",
                   body={"source": "local_folder", "path": str(folder)})
    assert r.status == 202
    job_id = r.body["job_id"]

    from sable.registry import ModelRegistry

    reg = ModelRegistry(home / "buffer" / "sable.db")
    try:
        deadline = _t.monotonic() + 15
        state = None
        while _t.monotonic() < deadline:
            job = reg.get_job(job_id)
            state = job.state if job else None
            if state in ("done", "error", "cancelled"):
                break
            _t.sleep(0.05)
        assert state == "done", f"job ended {state}"
    finally:
        reg.close()


def test_curate_source_validates(home):
    from sable.dashboard.app import DashboardApp

    app = DashboardApp(SableConfig(), home=home)
    assert app.handle("POST", "/api/datasets/curate_source",
                      body={"source": "ftp"}).status == 400
    assert app.handle("POST", "/api/datasets/curate_source",
                      body={"source": "local_folder",
                            "path": "/no/such/dir"}).status == 400
    assert app.handle("POST", "/api/datasets/curate_source",
                      body={"source": "website",
                            "url": "ftp://x"}).status == 400


# --------------------------------------------------------------- env gate

def test_datasets_env_preflight_and_install(monkeypatch, tmp_path):
    from sable.core import datasetenv

    calls = []

    def fake_run(cmd, **kw):
        calls.append(list(cmd))

        class R:
            returncode = 1 if "find_spec" in cmd[-1] else 0
        return R()

    py = str(tmp_path / "venv" / "bin" / "python")
    monkeypatch.setattr(datasetenv.os.path, "isabs", lambda p: True)
    st = datasetenv.preflight(type("C", (), {"training": {"python": py}})())
    assert st["datasets"] is False
    assert st["install_command"].startswith(py)
    assert "datasets" in st["install_command"]

    res = datasetenv.install_datasets(py, runner=fake_run)
    assert res["ok"]
    assert calls[-1][:5] == [py, "-m", "pip", "install", "-U"]
    assert calls[-1][5:] == ["datasets"], \
        "must NOT drag bitsandbytes (Windows independence)"

    # pip failure surfaces with tail detail
    def bad_run(cmd, **kw):
        class R:
            returncode, stdout, stderr = 2, "", "No matching distribution"
        return R()

    with pytest.raises(datasetenv.DatasetEnvError) as ei:
        datasetenv.install_datasets(py, runner=bad_run)
    assert "exit 2" in str(ei.value)


def test_doctor_datasets_check(monkeypatch):
    from sable import doctor
    from sable.core import datasetenv

    monkeypatch.setattr(datasetenv, "preflight",
                        lambda cfg: {"python": "py", "datasets": False,
                                     "install_command": "py -m pip install datasets",
                                     "note": "missing"})
    res = doctor.check_datasets_env(None)
    assert res.status == "warn"
    assert any("pip install datasets" in ln or
               "install datasets" in ln for ln in res.next_commands)

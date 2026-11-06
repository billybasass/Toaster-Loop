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

from toaster_loop.core.config import ToasterLoopConfig
from toaster_loop.curate import sources as src
from toaster_loop.curate.importer import ImportError_ as CurateImportError


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(h))
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


# ------------------------------------------------------- quality screening

def test_looks_like_data_separates_prose_from_json():
    prose = ("Toaster Loop curates datasets from local folders and websites. "
             "Start a model on the Serve page for higher-quality pairs. " * 4)
    assert not src._looks_like_data(prose)
    payload = json.dumps({
        "routes": [{"path": f"/{i}", "label": f"L{i}"} for i in range(30)],
        "ga4": {"measurementId": "G-X"}, "contextual": {"options": ["copy"]}})
    assert src._looks_like_data(payload)


def test_docs_to_rows_drops_machine_data_pairs():
    """A synth that returns a JSON-blob answer must not contribute rows — the
    'may hurt a model' failure mode from SPA docs payloads."""
    def jsony(name, chunk):
        return [("What is this?", '{"href":"https://x","label":"y",'
                                  '"ga4":{"id":"G-1"},"seo":{"a":1},'
                                  '"routes":[{"p":"/a"},{"p":"/b"}]}')]

    convs, warns = src.docs_to_rows([("page", "Some prose intro here. " * 6)],
                                    synth_fn=jsony)
    assert convs == []
    assert any("machine-data" in w for w in warns)


def test_scrub_machine_lines_keeps_prose():
    text = ('Real documentation sentence about installing the toolkit.\n'
            '{"href":"/terms","label":"Privacy"},{"search":{"prompt":"hi"}}\n'
            'Another fine prose line with normal punctuation.')
    out = src._scrub_machine_lines(text)
    assert "installing the toolkit" in out
    assert "Another fine prose" in out
    assert '"href"' not in out


def test_docs_to_rows_reports_progress():
    seen = []
    src.docs_to_rows(
        [("a.md", "# H\n\n## S\n" + "content body. " * 10),
         ("b.md", "# H2\n\n## T\n" + "more content. " * 10)],
        synth_fn=None, progress=seen.append)
    assert seen and "files)" in seen[-1]


def test_crawl_strips_script_json_links(home, site):
    """Links inside <script> JSON payloads must not be crawled (phantom 404s)."""
    res = src.crawl_website(home, site + "/scripty.html", max_pages=10,
                            max_depth=1, sleep_s=0.0,
                            allow_private_networks=True)
    manifest = json.loads(
        (home / "datasets" / res.dataset_id / "manifest.json").read_text())
    # /from-script-payload only appears inside the page's JSON state object;
    # it must never have been fetched (no 404 warning for it).
    assert not any("from-script-payload" in w for w in manifest["warnings"])


def test_live_synth_fn_dead_instance_falls_back(home, monkeypatch):
    """A named-but-not-running server falls back to the default serve state."""
    from toaster_loop.serving import instances as srv_inst

    calls = {}

    def fake_http(base, model):
        calls["base"] = base
        return ("synth", base, model)

    monkeypatch.setattr(src, "_http_synth", fake_http)
    # dead named instance -> default view (empty here) -> None
    assert src.live_synth_fn(type("A", (), {"home": home})(), "coder") is None
    assert calls == {}

    # live default state -> synth bound to it (write_state_merged keeps the
    # legacy serve_state.json view that live_synth_fn reads for default).
    srv_inst.write_state_merged(home, "default",
                                {"base_url": "http://127.0.0.1:9",
                                 "model_name": "m", "mode": "attach"})
    res = src.live_synth_fn(type("A", (), {"home": home})())
    assert res and calls["base"] == "http://127.0.0.1:9"


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
    # i18n hub page: links into localized mirrors plus the English guide.
    "/i18n.html": ("<html><body><h2>Docs</h2><p>"
                   + "Pick your docs language here. " * 6
                   + "</p>"
                   "<a href='/guide.html'>en guide</a>"
                   "<a href='/zh/guide.html'>中文</a>"
                   "<a href='/ja/guide.html'>日本語</a>"
                   "</body></html>"),
    "/zh/guide.html": ("<html><body><h2>安装</h2><p>"
                       + "工具包的安装步骤说明。" * 10
                       + "</p></body></html>"),
    "/ja/guide.html": ("<html><body><h2>インストール</h2><p>"
                       + "ツールキットのインストール手順です。" * 10
                       + "</p></body></html>"),
    # SPA-style page: navigable link in real markup, phantom route ONLY inside
    # a <script> body (quoted href= form the crawl regex would otherwise hit).
    "/scripty.html": (
        "<html><body><h2>Scripty</h2><p>"
        + "Docs body text for the scripty page. " * 6
        + "</p><a href='/guide.html'>guide</a>"
        + "<script>var state = '<ul>';"
        + "state += '<li><a href=\"/from-script-payload\">phantom</a></li>';"
        + "</script></body></html>"),
    # docs-tree pages for the subtree-scope tests: index links to its own
    # tree AND to a sibling forum section that must NOT be crawled by default.
    "/docs/index.html": ("<html><body><h2>Docs Home</h2><p>"
                         + "The documentation home page text. " * 6
                         + "</p><a href='/docs/page2.html'>next</a>"
                         + "<a href='/forum/post.html'>forum</a>"
                         + "</body></html>"),
    "/docs/page2.html": ("<html><body><h2>Second Doc</h2><p>"
                         + "Second page of the docs tree here. " * 6
                         + "</p></body></html>"),
    "/forum/post.html": ("<html><body><h2>Forum Post</h2><p>"
                         + "Community chatter not part of the docs. " * 6
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


def test_crawl_language_filter(home, site):
    """lang='en' keeps un-prefixed pages and drops /zh/ + /ja/ mirrors; the
    skipped count surfaces as a warning."""
    res = src.crawl_website(home, site + "/i18n.html", max_pages=10,
                            max_depth=1, sleep_s=0.0, lang="en",
                            allow_private_networks=True)
    manifest = json.loads(
        (home / "datasets" / res.dataset_id / "manifest.json").read_text())
    crawled = manifest["crawled_urls"]
    assert any(u.endswith("/guide.html") and "/zh/" not in u
               and "/ja/" not in u for u in crawled)
    assert not any("/zh/" in u or "/ja/" in u for u in crawled)
    assert any("other languages" in w for w in manifest["warnings"])


def test_crawl_language_all_keeps_everything(home, site):
    res = src.crawl_website(home, site + "/i18n.html", max_pages=10,
                            max_depth=1, sleep_s=0.0, lang="*",
                            allow_private_networks=True)
    manifest = json.loads(
        (home / "datasets" / res.dataset_id / "manifest.json").read_text())
    crawled = manifest["crawled_urls"]
    assert any("/zh/guide.html" in u for u in crawled)


def test_crawl_entry_page_language_wins(home, site):
    """Entering the crawl AT a /ja/ mirror crawls Japanese even when the
    machine language is English (auto-detect must not fight the entry)."""
    res = src.crawl_website(home, site + "/ja/guide.html", max_pages=10,
                            max_depth=1, sleep_s=0.0, lang="*",
                            allow_private_networks=True)
    manifest = json.loads(
        (home / "datasets" / res.dataset_id / "manifest.json").read_text())
    assert any("/ja/guide.html" in u for u in manifest["crawled_urls"])


def test_crawl_cancellation_stops_early(home, site):
    """should_cancel firing mid-crawl aborts with CurateCancelled before the
    remaining pages are fetched."""
    flags = {"n": 0}

    def _cancel():
        flags["n"] += 1
        return flags["n"] > 2   # let the first page or two through

    with pytest.raises(src.CurateCancelled):
        src.crawl_website(home, site + "/", max_pages=30, max_depth=2,
                          sleep_s=0.0, lang="*", should_cancel=_cancel,
                          allow_private_networks=True)


def test_crawl_scope_defaults_to_entry_dir(home, site):
    """A crawl started on /docs/index.html stays inside /docs/ — the forum
    link on the same host is never fetched (the Unreal Engine docs complaint)."""
    res = src.crawl_website(home, site + "/docs/index.html", max_pages=10,
                            max_depth=2, sleep_s=0.0, lang="*",
                            allow_private_networks=True)
    manifest = json.loads(
        (home / "datasets" / res.dataset_id / "manifest.json").read_text())
    crawled = manifest["crawled_urls"]
    assert any(u.endswith("/docs/page2.html") for u in crawled)
    assert not any("/forum/" in u for u in crawled)
    assert any("outside the docs subtree" in w for w in manifest["warnings"])


def test_crawl_scope_star_crawls_whole_origin(home, site):
    """path_prefix='' opts back into whole-same-origin crawling."""
    res = src.crawl_website(home, site + "/docs/index.html", max_pages=10,
                            max_depth=2, sleep_s=0.0, lang="*",
                            path_prefix="", allow_private_networks=True)
    crawled = json.loads(
        (home / "datasets" / res.dataset_id / "manifest.json").read_text()
    )["crawled_urls"]
    assert any("/forum/post.html" in u for u in crawled)


# ---------------------------------------------------- multi-dataset training

def test_train_merges_multiple_datasets(home, monkeypatch):
    """POST /api/train {datasets: [a, b]} materializes a merged dataset whose
    rows are the concatenation, and the job params point at the merge."""
    import time as _t

    from toaster_loop.dashboard.app import DashboardApp

    def _mk_folder(name, body):
        f = home / name
        f.mkdir()
        (f / "d.md").write_text("# H\n\n## S\n" + body * 12)
        return f

    a = src.ingest_folder(home, str(_mk_folder("fa", "alpha doc content. ")))
    b = src.ingest_folder(home, str(_mk_folder("fb", "beta doc content. ")))

    captured = {}

    def fake_run_claimed(self, job):
        captured["job"] = job
        return job

    monkeypatch.setattr("toaster_loop.training.scheduler.Scheduler.run_claimed",
                        fake_run_claimed)
    app = DashboardApp(ToasterLoopConfig(), home=home)
    r = app.handle("POST", "/api/train",
                   body={"datasets": [a.dataset_id, b.dataset_id],
                         "kind": "sft", "dry_run": True})
    assert r.status == 202, r.body
    deadline = _t.monotonic() + 5
    while _t.monotonic() < deadline and "job" not in captured:
        _t.sleep(0.02)
    job = captured["job"]
    merged_id = job.dataset_id
    assert merged_id not in (a.dataset_id, b.dataset_id)
    mdir = home / "datasets" / merged_id
    manifest = json.loads((mdir / "manifest.json").read_text())
    assert manifest["provenance"] == "merged"
    assert manifest["source_ids"] == [a.dataset_id, b.dataset_id]
    n_a = int(json.loads((home / "datasets" / a.dataset_id / "manifest.json")
                         .read_text())["files"]["sft.jsonl"]["count"])
    n_b = int(json.loads((home / "datasets" / b.dataset_id / "manifest.json")
                         .read_text())["files"]["sft.jsonl"]["count"])
    assert manifest["files"]["sft.jsonl"]["count"] == n_a + n_b
    # originals untouched
    assert (home / "datasets" / a.dataset_id / "manifest.json").is_file()
    assert job.params["dataset_dir"].endswith(merged_id)


def test_train_merge_rejects_missing_dataset(home):
    from toaster_loop.dashboard.app import DashboardApp

    app = DashboardApp(ToasterLoopConfig(), home=home)
    r = app.handle("POST", "/api/train",
                   body={"datasets": ["nope-a", "nope-b"]})
    assert r.status == 404


def test_model_name_reaches_job_params(home):
    """model_name (trained-model naming) rides on the job params."""
    from toaster_loop.dashboard.app import DashboardApp

    folder = home / "f"
    folder.mkdir()
    (folder / "d.md").write_text("# H\n\n## S\n" + "content words. " * 12)
    res = src.ingest_folder(home, str(folder))

    captured = {}
    import toaster_loop.training.scheduler as sched_mod

    def fake_run_claimed(self, job):
        captured["job"] = job
        return job

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(sched_mod.Scheduler, "run_claimed", fake_run_claimed)
    try:
        app = DashboardApp(ToasterLoopConfig(), home=home)
        r = app.handle("POST", "/api/train",
                       body={"dataset": res.dataset_id, "dry_run": True,
                             "model_name": "my-special-model"})
        assert r.status == 202
        import time as _t
        deadline = _t.monotonic() + 5
        while _t.monotonic() < deadline and "job" not in captured:
            _t.sleep(0.02)
        assert captured["job"].params.get("model_name") == "my-special-model"
    finally:
        monkeypatch.undo()


def test_docs_to_rows_cancellation():
    seen = {"n": 0}

    def _cancel():
        seen["n"] += 1
        return seen["n"] > 2

    with pytest.raises(src.CurateCancelled):
        src.docs_to_rows(
            [(f"f{i}.md", "# H\n\n## S\n" + "content body. " * 10)
             for i in range(6)],
            synth_fn=None, should_cancel=_cancel)


# ----------------------------------------------------------- delete endpoint

def test_dataset_delete_endpoint(home):
    from toaster_loop.dashboard.app import DashboardApp

    _make_hub_snapshot(home)
    app = DashboardApp(ToasterLoopConfig(), home=home)
    ds_id = app.handle("POST", "/api/datasets/import_hub",
                       body={"repo_id": "testorg/Test-Dataset"}
                       ).body["dataset_id"]
    assert (home / "datasets" / ds_id / "manifest.json").is_file()

    r = app.handle("DELETE", "/api/datasets/" + ds_id)
    assert r.status == 200, r.body
    assert not (home / "datasets" / ds_id).exists()
    # gone from the list, and a second delete is an honest 404
    ids = [d["dataset_id"] for d in
           app.handle("GET", "/api/datasets").body["datasets"]]
    assert ds_id not in ids
    assert app.handle("DELETE", "/api/datasets/" + ds_id).status == 404


def test_dataset_delete_rejects_bad_ids(home):
    from toaster_loop.dashboard.app import DashboardApp

    app = DashboardApp(ToasterLoopConfig(), home=home)
    assert app.handle("DELETE", "/api/datasets/..%2Fescape").status in (400, 404)
    assert app.handle("DELETE", "/api/datasets/nope-not-here").status == 404


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
    from toaster_loop.dashboard.app import DashboardApp

    _make_hub_snapshot(home)
    app = DashboardApp(ToasterLoopConfig(), home=home)

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

    monkeypatch.setattr("toaster_loop.training.scheduler.Scheduler.run_claimed",
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
    from toaster_loop.dashboard.app import DashboardApp

    app = DashboardApp(ToasterLoopConfig(), home=home)
    r = app.handle("POST", "/api/datasets/import_hub",
                   body={"repo_id": "org/never-downloaded"})
    assert r.status == 404


def test_hub_import_bad_repo_id(home):
    from toaster_loop.dashboard.app import DashboardApp

    app = DashboardApp(ToasterLoopConfig(), home=home)
    r = app.handle("POST", "/api/datasets/import_hub",
                   body={"repo_id": "../escape"})
    assert r.status == 400


# ------------------------------------------------ curate_source endpoint

def test_curate_source_endpoint_queues_folder_job(home, monkeypatch):
    """202+poll contract (same as /api/curate); the done log line carries the
    dataset id so the UI can jump straight to it."""
    import time as _t

    from toaster_loop.dashboard.app import DashboardApp

    folder = home / "in"
    folder.mkdir()
    (folder / "note.md").write_text("# N\n\n## Sec\n" + "body words. " * 12)

    # No live server -> live_synth_fn None -> fallback synth inline.
    monkeypatch.setattr(src, "live_synth_fn",
                        lambda app, server_id=None: None)
    app = DashboardApp(ToasterLoopConfig(), home=home)

    r = app.handle("POST", "/api/datasets/curate_source",
                   body={"source": "local_folder", "path": str(folder)})
    assert r.status == 202
    job_id = r.body["job_id"]

    from toaster_loop.registry import ModelRegistry

    reg = ModelRegistry(home / "buffer" / "toaster_loop.db")
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
    from toaster_loop.dashboard.app import DashboardApp

    app = DashboardApp(ToasterLoopConfig(), home=home)
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
    from toaster_loop.core import datasetenv

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
    from toaster_loop import doctor
    from toaster_loop.core import datasetenv

    monkeypatch.setattr(datasetenv, "preflight",
                        lambda cfg: {"python": "py", "datasets": False,
                                     "install_command": "py -m pip install datasets",
                                     "note": "missing"})
    res = doctor.check_datasets_env(None)
    assert res.status == "warn"
    assert any("pip install datasets" in ln or
               "install datasets" in ln for ln in res.next_commands)

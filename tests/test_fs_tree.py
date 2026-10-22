"""Phase 3 (file explorer) tests — GET /api/fs/tree.

Covers:
  * happy path: root listing dirs-first, one level only (nested contents are
    not expanded), relative paths + size/mtime metadata.
  * containment: `..` traversal, absolute paths and %2e%2e-encoded traversal
    rejected with 400; a symlink pointing outside the workspace is neither
    listed nor traversable.
  * sensitive paths from the core/tools.py deny list (.ssh/ dir, .env file)
    omitted from listings; blocked as a listing root too.
  * hidden dotfiles omitted unless ?hidden=1.
  * errors: unknown project 404, missing project param 400.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sable.core.config import SableConfig  # noqa: E402
from sable.dashboard.app import DashboardApp  # noqa: E402


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "uf-fs"
    monkeypatch.setenv("SABLE_HOME", str(h))
    return h


@pytest.fixture()
def app(home):
    return DashboardApp(SableConfig(), home=home)


@pytest.fixture()
def workspace(tmp_path):
    ws = tmp_path / "ws"
    (ws / "src").mkdir(parents=True)
    (ws / "src" / "app.py").write_text("print('hi')\n")
    (ws / "README.md").write_text("# hi\n")
    (ws / ".hidden-file").write_text("x")
    (ws / ".config").mkdir()
    return ws


def _mkproject(app, ws):
    r = app.handle("POST", "/api/projects",
                   body={"name": "p", "workspace_dir": str(ws)})
    assert r.status == 201
    return r.body["project"]["id"]


# ------------------------------------------------------------------ happy path

def test_fs_tree_root_dirs_first_one_level(app, workspace):
    pid = _mkproject(app, workspace)
    r = app.handle("GET", "/api/fs/tree", query={"project": pid, "path": ""})
    assert r.status == 200, r.body
    entries = r.body["entries"]
    names = [e["name"] for e in entries]
    # dirs first, then files, each group sorted; dotfiles hidden by default
    assert names == ["src", "README.md"]
    src = entries[0]
    assert src["type"] == "dir" and src["path"] == "src"
    readme = entries[1]
    assert readme["type"] == "file" and readme["path"] == "README.md"
    assert readme["size"] == len("# hi\n")
    assert readme["mtime"]
    # one level only: nested dir contents are NOT expanded
    assert all("src/" not in e["path"] for e in entries)

    # expanding the subdir returns its single level with rel paths prefixed
    r = app.handle("GET", "/api/fs/tree", query={"project": pid, "path": "src"})
    assert r.status == 200
    assert [(e["name"], e["path"], e["type"]) for e in r.body["entries"]] == \
        [("app.py", "src/app.py", "file")]


# ------------------------------------------------------------------- containment

def test_fs_tree_rejects_traversal(app, workspace):
    pid = _mkproject(app, workspace)
    outside = workspace.parent / "evil"
    outside.mkdir()
    (outside / "secret.txt").write_text("boo")

    for bad in ("../evil", "..", "src/../../evil", "a/../b"):
        r = app.handle("GET", "/api/fs/tree",
                       query={"project": pid, "path": bad})
        assert r.status == 400, (bad, r.body)
        assert r.body["error"]["code"] == "bad_request"


def test_fs_tree_rejects_absolute_paths(app, workspace):
    pid = _mkproject(app, workspace)
    for bad in ("/etc", str(workspace.parent), "~/.ssh"):
        r = app.handle("GET", "/api/fs/tree",
                       query={"project": pid, "path": bad})
        assert r.status == 400, (bad, r.body)


def test_fs_tree_rejects_encoded_traversal(app, workspace):
    pid = _mkproject(app, workspace)
    # The HTTP layer decodes query strings via parse_qs; the handler re-checks
    # the unquoted form too, so both shapes are rejected.
    for enc in ("%2e%2e/evil", "..%2fevil", "%2E%2E"):
        r = app.handle("GET", "/api/fs/tree",
                       query={"project": pid, "path": enc})
        assert r.status == 400, (enc, r.body)
        # and after transport-level decoding (what parse_qs delivers) likewise
        from urllib.parse import unquote
        r2 = app.handle("GET", "/api/fs/tree",
                        query={"project": pid, "path": unquote(enc)})
        assert r2.status == 400, (unquote(enc), r2.body)


@pytest.mark.skipif(os.name != "posix", reason="symlinks are posix here")
def test_fs_tree_symlink_escape_not_listed_or_traversed(app, workspace, tmp_path):
    pid = _mkproject(app, workspace)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "private.txt").write_text("secret")
    link = workspace / "link-out"
    link.symlink_to(outside)

    # not listed at all (target resolves outside the workspace root)
    r = app.handle("GET", "/api/fs/tree", query={"project": pid, "path": ""})
    assert r.status == 200
    assert "link-out" not in [e["name"] for e in r.body["entries"]]

    # and cannot be traversed into either
    r = app.handle("GET", "/api/fs/tree",
                   query={"project": pid, "path": "link-out"})
    assert r.status == 400, r.body


# ---------------------------------------------------------------- sensitive paths

def test_fs_tree_omits_sensitive_entries(app, workspace):
    pid = _mkproject(app, workspace)
    # real cases from core/tools.py deny lists: .ssh/ dir + .env file
    (workspace / ".ssh").mkdir()
    (workspace / ".ssh" / "id_rsa").write_text("key")
    (workspace / ".env").write_text("SECRET=1")

    r = app.handle("GET", "/api/fs/tree",
                   query={"project": pid, "path": "", "hidden": "1"})
    assert r.status == 200
    names = [e["name"] for e in r.body["entries"]]
    assert ".ssh" not in names and ".env" not in names
    assert "src" in names  # control: normal entries still listed


def test_fs_tree_sensitive_dir_blocked_as_root(app, workspace):
    pid = _mkproject(app, workspace)
    (workspace / ".ssh").mkdir()
    r = app.handle("GET", "/api/fs/tree",
                   query={"project": pid, "path": ".ssh"})
    assert r.status == 400
    assert "sensitive" in r.body["error"]["message"]


# ----------------------------------------------------------------------- hidden

def test_fs_tree_hidden_dotfiles_opt_in(app, workspace):
    pid = _mkproject(app, workspace)
    r = app.handle("GET", "/api/fs/tree", query={"project": pid})
    names = [e["name"] for e in r.body["entries"]]
    assert ".hidden-file" not in names and ".config" not in names

    r = app.handle("GET", "/api/fs/tree",
                   query={"project": pid, "hidden": "1"})
    names = [e["name"] for e in r.body["entries"]]
    assert ".hidden-file" in names and ".config" in names


# ------------------------------------------------------------------------ errors

def test_fs_tree_unknown_project_404(app):
    r = app.handle("GET", "/api/fs/tree", query={"project": "does-not-exist"})
    assert r.status == 404
    assert r.body["error"]["code"] == "unknown_project"


def test_fs_tree_requires_project_param(app):
    r = app.handle("GET", "/api/fs/tree", query={})
    assert r.status == 400
    r = app.handle("GET", "/api/fs/tree", query={"project": "  "})
    assert r.status == 400


def test_fs_tree_missing_workspace_dir_404(app, home, tmp_path):
    ws = tmp_path / "gone"
    ws.mkdir()
    pid = _mkproject(app, ws)
    ws.rmdir()  # workspace deleted after the project was created
    r = app.handle("GET", "/api/fs/tree", query={"project": pid})
    assert r.status == 404


def test_fs_tree_missing_subdir_404(app, workspace):
    pid = _mkproject(app, workspace)
    r = app.handle("GET", "/api/fs/tree",
                   query={"project": pid, "path": "nope"})
    assert r.status == 404


def test_fs_tree_endpoint_is_read_only(app, workspace):
    # listing must never mutate anything on disk
    pid = _mkproject(app, workspace)
    before = sorted(os.listdir(workspace))
    app.handle("GET", "/api/fs/tree", query={"project": pid})
    assert sorted(os.listdir(workspace)) == before

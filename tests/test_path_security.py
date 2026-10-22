"""Backlog #9 tests — workspace containment in the tool dispatcher.

Unit coverage for core/path_security.py (traversal detection, inside/outside
root, symlink escape, nonexistent-but-inside paths, multi-root allow-list)
plus runner integration through create_runner/SableAgentClient: refused
writes outside the workspace and through symlink escapes, normal writes
inside allowed, todo_write still working via the SABLE_HOME allow-list,
temp-dir allow-list keeping tmp_path flows green, and explicit-null
unrestricted mode logging a startup warning.
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fake_server import FakeOpenAIServer  # noqa: E402
from sable.core import agent_runner as ar  # noqa: E402
from sable.core import path_security as ps  # noqa: E402
from sable.core.agent_runner import (  # noqa: E402
    create_runner,
    resolve_workspace_roots,
)
from sable.core.config import UNSET, SableConfig, load_config  # noqa: E402


@pytest.fixture()
def sable_home(tmp_path, monkeypatch):
    home = tmp_path / "uf-home"
    monkeypatch.setenv("SABLE_HOME", str(home))
    return home


@pytest.fixture(autouse=True)
def _tempdir_outside_workspace(tmp_path, monkeypatch):
    """Move the tempdir allow-list root out from under tmp_path.

    pytest's tmp_path lives *inside* the real system temp dir, which backlog
    #9 always allow-lists — without this, "outside workspace" assertions would
    trivially pass through the temp-dir exception. The substitute is an empty
    sibling dir that no test writes into.
    """
    fake_tmp = tmp_path.parent / f"fake-tempdir-{tmp_path.name}"
    fake_tmp.mkdir(exist_ok=True)
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(fake_tmp))
    yield fake_tmp


# --------------------------------------------------------------- unit: checks

def test_traversal_detection():
    assert ps.has_traversal_component("a/../../etc/passwd")
    assert ps.has_traversal_component("../sibling.txt")
    assert not ps.has_traversal_component("a/b/c.txt")
    assert not ps.has_traversal_component("..hidden/file")  # name, not component
    assert not ps.has_traversal_component("")


def test_validate_within_dir_inside_outside(tmp_path):
    inside = tmp_path / "sub" / "file.txt"  # nonexistent is fine (strict=False)
    assert ps.validate_within_dir(inside, tmp_path) is None
    assert ps.validate_within_dir(tmp_path, tmp_path) is None
    outside = tmp_path.parent / "elsewhere.txt"
    err = ps.validate_within_dir(outside, tmp_path)
    assert err and "escapes" in err


def test_validate_within_dir_symlink_escape(tmp_path):
    (tmp_path / "real").mkdir()
    link = tmp_path / "link"
    link.symlink_to(tmp_path.parent)  # link/ -> parent of the workspace
    err = ps.validate_within_dir(link / "secret.txt", tmp_path)
    assert err and "escapes" in err


def test_within_any_dir_multiroot(tmp_path):
    root_a = tmp_path / "ws"
    root_b = tmp_path / "home-state"
    root_a.mkdir()
    root_b.mkdir()
    assert ps.within_any_dir(root_a / "a.txt", [root_a, root_b]) is None
    assert ps.within_any_dir(root_b / "b.txt", [root_a, root_b]) is None
    # relative paths resolve against the FIRST (primary) root
    assert ps.within_any_dir("sub/a.txt", [root_a, root_b]) is None
    assert ps.within_any_dir(tmp_path / "outside.txt", [root_a, root_b])


# ------------------------------------------------------------- unit: decorator

def test_contain_blocks_write_outside_root(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()

    @ps.contain([ws])
    def write_file(path: str, content: str = "") -> str:
        return f"executed {path}"

    out = write_file(path=str(ws / "ok.txt"))
    assert "executed" in out  # inside: runs
    blocked = write_file(path=str(tmp_path / "outside.txt"))
    assert blocked.startswith("error: blocked: path outside workspace")
    assert str(ws) in blocked


def test_contain_blocks_traversal_even_inside_root(tmp_path):
    ws = tmp_path / "ws"
    (ws / "sub").mkdir(parents=True)

    @ps.contain([ws])
    def read_file(path: str) -> str:
        return f"executed {path}"

    # normalizes back inside the workspace, but '..' is refused up front
    out = read_file(path=str(ws / "sub" / ".." / "a.txt"))
    assert out.startswith("error: blocked: '..' path traversal")


def test_contain_blocks_symlink_escape_write(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    victim_dir = tmp_path / "outside"
    victim_dir.mkdir()
    link = ws / "escape"
    link.symlink_to(victim_dir)  # existing target -> resolve() follows it

    @ps.contain([ws])
    def write_file(path: str, content: str = "") -> str:
        Path(path).write_text(content)
        return f"executed {path}"

    out = write_file(path=str(link / "bashrc"), content="rm -rf /")
    assert out.startswith("error: blocked: path outside workspace")
    assert not (victim_dir / "bashrc").exists()


def test_contain_allows_nonexistent_inside_and_tempdir(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()  # parent exists, file does not

    @ps.contain([ws, Path(tempfile.gettempdir())])
    def write_file(path: str, content: str = "") -> str:
        return f"executed {path}"

    assert "executed" in write_file(path=str(ws / "new" / "deep.txt"))
    assert "executed" in write_file(
        path=str(Path(tempfile.gettempdir()) / "sable-tmp-check.txt"))


def test_contain_ignores_non_path_and_missing_args(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()

    @ps.contain([ws])
    def todo_write(todos=None, **_):
        return "ok"

    assert todo_write(todos=[{"content": "x"}]) == "ok"


def test_contain_checks_file_path_alias(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()

    @ps.contain([ws])
    def some_tool(file_path: str = "") -> str:
        return f"executed {file_path}"

    assert some_tool(file_path=str(ws.parent / "x")).startswith("error: blocked")


# ------------------------------------------------------ config / root resolution

def test_roots_default_to_cwd_when_unset(monkeypatch, sable_home):
    monkeypatch.setattr(ar.os, "getcwd", lambda: "/tmp/fake-cwd")
    roots = resolve_workspace_roots(SableConfig())
    assert roots is not None
    assert roots[0] == Path("/tmp/fake-cwd")
    assert Path(str(sable_home)) in [Path(str(r)) for r in roots[1:]]
    assert Path(tempfile.gettempdir()) in [Path(str(r)) for r in roots[1:]]


def test_roots_explicit_path_and_profile_override(tmp_path, sable_home):
    cfg = SableConfig(agent={"workspace_root": str(tmp_path / "ws")})
    roots = resolve_workspace_roots(cfg)
    assert roots is not None and roots[0] == tmp_path / "ws"
    # per-profile override widens/narrows the global primary root
    roots2 = resolve_workspace_roots(cfg, workspace_root_override=str(tmp_path))
    assert roots2 is not None and roots2[0] == tmp_path


def test_explicit_null_is_unrestricted_with_warning(tmp_path, caplog, monkeypatch):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("agent:\n  workspace_root: null\n")
    cfg = load_config(str(cfg_file))
    assert cfg.agent["workspace_root"] is UNSET  # explicit-null preserved
    monkeypatch.setattr(ar, "_UNRESTRICTED_WARNED", False)
    with caplog.at_level(logging.WARNING, logger="sable.agent"):
        assert resolve_workspace_roots(cfg) is None
    assert any("containment is DISABLED" in r.message for r in caplog.records)

    # unset (key absent entirely) => cwd default, NOT unrestricted
    cfg_file.write_text("agent:\n  keep_recent_tokens: 10\n")
    cfg2 = load_config(str(cfg_file))
    assert "workspace_root" not in cfg2.agent
    assert resolve_workspace_roots(cfg2) is not None


def test_create_runner_applies_containment(tmp_path, sable_home):
    ws = tmp_path / "ws"
    ws.mkdir()
    cfg = SableConfig(agent={"workspace_root": str(ws)})
    runner = create_runner(cfg, base_url="http://127.0.0.1:9/v1", model="m")
    assert runner.workspace_roots and runner.workspace_roots[0] == ws
    out = runner.tool_impls["write_file"](path=str(tmp_path / "x.txt"),
                                          content="boom")
    assert out.startswith("error: blocked: path outside workspace")
    ok = runner.tool_impls["write_file"](path=str(ws / "ok.txt"), content="hi")
    assert ok.startswith("wrote")


# ------------------------------------------------------------ runner integration

def _tool_loop_handler(tool: str, args: dict):
    """Handler class that issues one tool call, then finishes."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args_):
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            messages = payload.get("messages", [])
            if any(m.get("role") == "tool" for m in messages):
                msg = {"role": "assistant", "content": "done"}
                finish = "stop"
            else:
                msg = {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "c1", "type": "function",
                    "function": {"name": tool,
                                 "arguments": json.dumps(args)}}]}
                finish = "tool_calls"
            body = json.dumps({
                "id": "x", "object": "chat.completion",
                "model": payload.get("model", "m"),
                "choices": [{"index": 0, "message": msg,
                             "finish_reason": finish}],
                "usage": {}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def _run_tool(tmp_path, sable_home, tool: str, args: dict,
              workspace_root: Path) -> tuple:
    """Run one tool call through a contained runner; return (result, content)."""
    cfg = SableConfig(agent={"workspace_root": str(workspace_root)})
    with FakeOpenAIServer(handler_cls=_tool_loop_handler(tool, args)) as srv:
        runner = create_runner(cfg, base_url=srv.base_url, model="m")
        result = runner.run("go")
    tool_msgs = [m["content"] for m in result.messages if m.get("role") == "tool"]
    return result, tool_msgs


def test_runner_refuses_write_outside_workspace(tmp_path, sable_home):
    ws = tmp_path / "ws"
    ws.mkdir()
    outside = ws.parent / "outside.txt"
    result, contents = _run_tool(
        tmp_path, sable_home, "write_file",
        {"path": str(ws / ".." / "outside.txt"), "content": "x"}, ws)
    assert any(c.startswith("error: blocked: '..' path traversal")
               for c in contents)
    assert not outside.exists()

    # absolute outside path: blocked by containment, file never created
    result, contents = _run_tool(
        tmp_path, sable_home, "write_file",
        {"path": str(outside), "content": "x"}, ws)
    assert any("path outside workspace" in c for c in contents)
    assert not outside.exists()


def test_runner_blocks_symlink_escape(tmp_path, sable_home):
    ws = tmp_path / "ws"
    ws.mkdir()
    victim = tmp_path / "victim"
    victim.mkdir()
    link = ws / "escape"
    link.symlink_to(victim)
    result, contents = _run_tool(
        tmp_path, sable_home, "write_file",
        {"path": str(link / ".bashrc"), "content": "evil"}, ws)
    assert any("path outside workspace" in c for c in contents)
    assert not (victim / ".bashrc").exists()


def test_runner_allows_inside_and_todo_write(tmp_path, sable_home):
    ws = tmp_path / "ws"
    ws.mkdir()
    result, contents = _run_tool(
        tmp_path, sable_home, "write_file",
        {"path": str(ws / "notes" / "a.txt"), "content": "hello"}, ws)
    assert any(c.startswith("wrote") for c in contents)
    assert (ws / "notes" / "a.txt").read_text() == "hello"

    # todo_write lives under SABLE_HOME — allowed via the home allow-list
    result, contents = _run_tool(
        tmp_path, sable_home, "todo_write",
        {"todos": [{"id": "t1", "content": "ship it", "status": "pending"}]}, ws)
    assert any("[ ] t1: ship it" in c for c in contents)


def test_runner_tempdir_allowlist(tmp_path, sable_home, _tempdir_outside_workspace):
    ws = tmp_path / "ws"
    ws.mkdir()
    td = Path(tempfile.gettempdir()) / f"sable-allow-{tmp_path.name}.txt"
    try:
        result, contents = _run_tool(
            tmp_path, sable_home, "write_file",
            {"path": str(td), "content": "tmp"}, ws)
        assert any(c.startswith("wrote") for c in contents)
        assert td.read_text() == "tmp"
    finally:
        td.unlink(missing_ok=True)


def test_runner_list_dir_sensitive_and_containment(tmp_path, sable_home):
    ws = tmp_path / "ws"
    ws.mkdir()
    # sensitive deny list now covers list_dir roots too (inside the
    # workspace, so containment passes and the deny list is what refuses)
    fake_ssh = ws / ".ssh"
    fake_ssh.mkdir()
    result, contents = _run_tool(
        tmp_path, sable_home, "list_dir", {"path": str(fake_ssh)}, ws)
    assert any("sensitive credential path" in c for c in contents)
    # and containment covers it as an outside path when not sensitive-flagged:
    result, contents = _run_tool(
        tmp_path, sable_home, "list_dir", {"path": str(tmp_path)}, ws)
    assert any("path outside workspace" in c for c in contents)


def test_runner_unrestricted_when_workspace_root_null(tmp_path, sable_home,
                                                      monkeypatch):
    monkeypatch.setattr(ar, "_UNRESTRICTED_WARNED", False)
    ws = tmp_path / "ws"
    ws.mkdir()
    target = tmp_path / "free.txt"
    cfg = SableConfig(agent={"workspace_root": None})
    with FakeOpenAIServer(
            handler_cls=_tool_loop_handler(
                "write_file", {"path": str(target), "content": "ok"})) as srv:
        runner = create_runner(cfg, base_url=srv.base_url, model="m")
        assert runner.workspace_roots is None
        runner.run("go")
    assert target.read_text() == "ok"


# ------------------------------------- config-takeover carve-out (deny_writes)

def test_contain_deny_writes_blocks_config_write_allows_read(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    cfg_file = home / "sable.yaml"
    cfg_file.write_text("agent: {}\n")

    @ps.contain([ws, home], deny_writes=[cfg_file])
    def write_file(path: str, content: str = "") -> str:
        return f"executed {path}"

    blocked = write_file(path=str(cfg_file), content="pwned")
    assert blocked.startswith("error: blocked:")
    assert "cannot write the sable config" in blocked
    # reads of the same path stay allowed through the allow-list
    @ps.contain([ws, home], deny_writes=[cfg_file], read_only=True)
    def read_file(path: str) -> str:
        return f"executed {path}"

    assert read_file(path=str(cfg_file)) == f"executed {cfg_file}"


def test_runner_blocks_config_takeover(tmp_path, sable_home):
    """End-to-end: agent-mode write_file at <home>/sable.yaml is refused and
    the file is untouched; other SABLE_HOME paths still work (allow-list)."""
    ws = tmp_path / "ws"
    ws.mkdir()
    sable_home.mkdir(parents=True, exist_ok=True)
    cfg_file = sable_home / "sable.yaml"
    cfg_file.write_text("agent:\n  workspace_root: keep-me\n")

    result, contents = _run_tool(
        tmp_path, sable_home, "write_file",
        {"path": str(cfg_file), "content": "capabilities: {run_commands: true}"},
        ws)
    assert any("cannot write the sable config" in c for c in contents)
    assert cfg_file.read_text().startswith("agent:")

    # a sibling path under SABLE_HOME is still writable (allow-list intact)
    ok_target = sable_home / "scratch.txt"
    result, contents = _run_tool(
        tmp_path, sable_home, "write_file",
        {"path": str(ok_target), "content": "fine"}, ws)
    assert ok_target.is_file()

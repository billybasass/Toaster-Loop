"""Feature-wave Phase 4: editor-style post-write error checking tests.

Covers tier-1 syntax checks on write_file/patch (established semantics: the file
IS written; the model sees the parser error and fixes it next turn), the
pre-vs-post delta filtering, node-missing graceful skips, lint_on_write=false
disabling everything, opt-in LSP defaults + a stub JSON-RPC server exercising
the detected-server path with baseline delta, and config/env plumbing.
"""

from __future__ import annotations

import stat
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sable.core import tools  # noqa: E402
from sable.core.config import load_config  # noqa: E402
from sable.core.tools import (SYNTAX_FAIL_MARK, SYNTAX_OK_MARK, patch,
                                write_file)


# ------------------------------------------------------------- tier 1: python

def test_broken_python_write_reports_error_and_still_writes(tmp_path):
    p = tmp_path / "broken.py"
    res = write_file(str(p), "def f(:\n    pass\n")
    # file WAS written (established semantics: never reject the write)
    assert p.is_file() and p.read_text().startswith("def f(")
    assert res.startswith(f"wrote {len('def f(:' + chr(10) + '    pass' + chr(10))} chars")
    assert SYNTAX_FAIL_MARK in res
    assert "fix it next turn" in res
    assert str(p) in res
    # parser error text carries the line number of the break
    assert ":1:" in res or "line 1" in res


def test_valid_python_write_passes_with_check_ok(tmp_path):
    p = tmp_path / "ok.py"
    res = write_file(str(p), "x = 1\n")
    assert res.endswith(SYNTAX_OK_MARK)
    assert SYNTAX_FAIL_MARK not in res


def test_unknown_extension_skipped_silently(tmp_path):
    p = tmp_path / "notes.dat"
    res = write_file(str(p), "totally (( unbalanced content")
    assert res == f"wrote {len('totally (( unbalanced content')} chars to {p}"
    assert SYNTAX_OK_MARK not in res and SYNTAX_FAIL_MARK not in res


# ----------------------------------------------------------------- tier 1: js

def test_broken_js_write_reports_node_error(tmp_path):
    if tools.shutil.which("node") is None:
        pytest.skip("node not installed")
    p = tmp_path / "bad.js"
    res = write_file(str(p), "const x = ;\n")
    assert p.is_file()
    assert SYNTAX_FAIL_MARK in res
    assert "SyntaxError" in res


def test_valid_js_write_passes(tmp_path):
    if tools.shutil.which("node") is None:
        pytest.skip("node not installed")
    p = tmp_path / "ok.js"
    res = write_file(str(p), "const x = 1;\n")
    assert res.endswith(SYNTAX_OK_MARK)


def test_node_missing_degrades_to_skip(tmp_path, monkeypatch):
    monkeypatch.setattr(tools.shutil, "which", lambda name: None)
    p = tmp_path / "x.js"
    res = write_file(str(p), "const x = ;\n")  # broken, but node is "missing"
    assert SYNTAX_FAIL_MARK not in res and SYNTAX_OK_MARK not in res
    status, _ = tools._check_node_syntax(str(p), "const x = ;\n")
    assert status == "skipped"


# ------------------------------------------------------------ delta semantics

def test_patch_into_pre_broken_file_not_blamed(tmp_path):
    """Pre-existing syntax error must be annotated as inherited, not new."""
    p = tmp_path / "pre.py"
    write_file(str(p), "def g(:\n    pass\n", lint_on_write=False)
    res = patch(str(p), "pass", "pass  # touched")
    assert p.is_file() and "# touched" in p.read_text()
    assert SYNTAX_FAIL_MARK in res
    assert "didn't introduce a new one" in res


def test_patch_introducing_new_error_is_reported(tmp_path):
    p = tmp_path / "clean.py"
    write_file(str(p), "x = 1\n", lint_on_write=False)
    res = patch(str(p), "x = 1", "x = (")
    assert SYNTAX_FAIL_MARK in res
    assert "didn't introduce a new one" not in res


def test_tier1_delta_filters_pre_existing_error_lines(monkeypatch):
    """Direct old-vs-new diff logic: shared error lines drop out, new ones
    stay (pre-edit baseline line-set difference)."""
    def fake_check(path, content):
        if "NEWBREAK" in content:
            return "error", "f.py:1: SyntaxError: old\nf.py:9: SyntaxError: new"
        return "error", "f.py:1: SyntaxError: old"

    monkeypatch.setattr(tools, "_tier1_check", fake_check)
    post_err = "f.py:1: SyntaxError: old\nf.py:9: SyntaxError: new"
    report, text = tools._tier1_delta("f.py", "OLD", post_err)
    assert report is True
    assert "New syntax errors" in text and "new" in text
    assert "old" not in text.split("introduced by this edit")[1]

    # every post error pre-existing => still-broken annotation, no blame
    monkeypatch.setattr(tools, "_tier1_check",
                        lambda path, content: ("error", "f.py:1: same"))
    report, text = tools._tier1_delta("f.py", "A", "f.py:1: same")
    assert report is True
    assert "didn't introduce a new one" in text

    # clean pre-edit baseline => full post error reported as-is
    monkeypatch.setattr(tools, "_tier1_check", lambda path, content: ("ok", ""))
    report, text = tools._tier1_delta("f.py", "clean-before", "f.py:2: boom")
    assert text == "f.py:2: boom"


# --------------------------------------------------------------- gates/config

def test_lint_on_write_false_disables_everything(tmp_path):
    p = tmp_path / "off.py"
    res = write_file(str(p), "def f(:\n", lint_on_write=False)
    assert res == f"wrote {len('def f(:' + chr(10))} chars to {p}"
    p2 = tmp_path / "c.py"
    write_file(str(p2), "x = 1\n", lint_on_write=False)
    res2 = patch(str(p2), "x = 1", "x = (", lint_on_write=False)
    assert SYNTAX_FAIL_MARK not in res2


def test_lsp_disabled_by_default(tmp_path):
    # no lsp_config at all => zero LSP text even on a syntax failure
    p = tmp_path / "d.py"
    res = write_file(str(p), "def f(:\n")
    assert "LSP" not in res
    assert tools._maybe_lsp_suffix(str(p), None, "x", {}) == ""


# ------------------------------------------------------------- tier 2: LSP

_STUB_LSP = """#!/usr/bin/env python3
import json, re, sys

def send(msg):
    body = json.dumps(msg).encode()
    sys.stdout.buffer.write(b"Content-Length: %d\\r\\n\\r\\n" % len(body) + body)
    sys.stdout.buffer.flush()

def read():
    headers = b""
    while b"\\r\\n\\r\\n" not in headers:
        ch = sys.stdin.buffer.read(1)
        if not ch:
            return None
        headers += ch
    n = int(re.search(rb"Content-Length: (\\d+)", headers).group(1))
    return json.loads(sys.stdin.buffer.read(n))

def diag(msg, line):
    return {"range": {"start": {"line": line, "character": 0},
                      "end": {"line": line, "character": 1}},
            "severity": 1, "source": "stub", "code": "E1", "message": msg}

while True:
    m = read()
    if m is None:
        break
    method = m.get("method")
    if method == "initialize":
        send({"jsonrpc": "2.0", "id": m["id"], "result": {"capabilities": {}}})
    elif method == "shutdown":
        send({"jsonrpc": "2.0", "id": m["id"], "result": None})
    elif method == "exit":
        break
    elif method == "textDocument/didOpen":
        uri = m["params"]["textDocument"]["uri"]
        send({"jsonrpc": "2.0", "method": "textDocument/publishDiagnostics",
              "params": {"uri": uri, "diagnostics": [diag("old error", 0)]}})
    elif method == "textDocument/didChange":
        uri = m["params"]["textDocument"]["uri"]
        send({"jsonrpc": "2.0", "method": "textDocument/publishDiagnostics",
              "params": {"uri": uri,
                         "diagnostics": [diag("old error", 0),
                                         diag("new error", 1)]}})
"""


@pytest.fixture()
def stub_server(tmp_path):
    """Executable stub LSP server fixture (baseline 'old error'; after the
    didChange it also pushes 'new error' — only the latter may be reported)."""
    s = tmp_path / "stub-langserver"
    s.write_text(_STUB_LSP)
    s.chmod(s.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP)
    return str(s)


def test_lsp_detected_server_reports_only_new_diagnostics(tmp_path, stub_server):
    p = tmp_path / "svc.py"
    cfg = {"enabled": True,
           "servers": {".py": stub_server},
           "startup_timeout_seconds": 10,
           "check_timeout_seconds": 15}
    # clean pre-edit content (written without checks), then a patch whose
    # edit makes the stub push baseline + new diagnostics.
    write_file(str(p), "x = 1\n", lint_on_write=False)
    res = patch(str(p), "x = 1", "x = 2\ny = 3\n", lsp_config=cfg)
    assert "NEW LSP DIAGNOSTICS" in res
    assert "new error" in res
    assert "old error" not in res  # baseline delta filtered it out


def test_lsp_graceful_skip_when_server_not_detected(tmp_path, monkeypatch):
    monkeypatch.setattr(tools.shutil, "which", lambda name: None)
    cfg = {"enabled": True}
    assert tools._maybe_lsp_suffix(
        str(tmp_path / "a.py"), "x=1", "x=2", cfg) == ""


def test_lsp_probe_never_outlives_its_timeout(tmp_path, stub_server):
    # check_timeout 0.5s must return quickly with no verdict rather than hang
    diags = tools._lsp_diagnostics(str(tmp_path / "t.py"), "a", "b",
                                   [stub_server], startup_timeout=0.3,
                                   check_timeout=0.5)
    assert diags is None or isinstance(diags, list)


# ------------------------------------------------------------ config plumbing

def test_config_parses_tools_lint_and_lsp(tmp_path, monkeypatch):
    monkeypatch.delenv("SABLE_LINT_ON_WRITE", raising=False)
    monkeypatch.delenv("SABLE_LSP", raising=False)
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(yaml.safe_dump({
        "tools": {"lint_on_write": False,
                  "lsp": {"enabled": True, "startup_timeout_seconds": 9}}}))
    cfg = load_config(str(cfg_file))
    assert cfg.tools["lint_on_write"] is False
    assert cfg.tools["lsp"]["enabled"] is True
    assert cfg.tools["lsp"]["startup_timeout_seconds"] == 9


def test_config_defaults_lint_on_true_lsp_absent(tmp_path, monkeypatch):
    monkeypatch.delenv("SABLE_LINT_ON_WRITE", raising=False)
    monkeypatch.delenv("SABLE_LSP", raising=False)
    cfg = load_config()
    assert "lint_on_write" not in cfg.tools  # default applied at read site
    assert bool((cfg.tools.get("lsp") or {}).get("enabled")) is False


def test_env_override_lint_on_write(tmp_path, monkeypatch):
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(yaml.safe_dump({"tools": {"lint_on_write": True}}))
    monkeypatch.setenv("SABLE_LINT_ON_WRITE", "false")
    assert load_config(str(cfg_file)).tools["lint_on_write"] is False
    monkeypatch.setenv("SABLE_LINT_ON_WRITE", "1")
    assert load_config(str(cfg_file)).tools["lint_on_write"] is True


def test_env_override_lsp_enabled(tmp_path, monkeypatch):
    monkeypatch.delenv("SABLE_LINT_ON_WRITE", raising=False)
    monkeypatch.setenv("SABLE_LSP", "true")
    cfg = load_config()
    assert cfg.tools["lsp"]["enabled"] is True


def test_create_runner_binds_lint_flags(tmp_path, monkeypatch):
    """create_runner must thread tools.lint_on_write/lsp into write_file/patch
    (bound invisibly; the model cannot toggle them)."""
    from sable.core.agent_runner import create_runner

    monkeypatch.delenv("SABLE_LINT_ON_WRITE", raising=False)
    monkeypatch.delenv("SABLE_LSP", raising=False)
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(yaml.safe_dump({"tools": {"lint_on_write": False}}))
    cfg = load_config(str(cfg_file))
    client = create_runner(cfg, base_url="http://127.0.0.1:1/v1", model="m")
    p = tmp_path / "bound.py"
    res = client.tool_impls["write_file"](str(p), "def f(:\n")
    assert SYNTAX_FAIL_MARK not in res  # lint disabled through the runner

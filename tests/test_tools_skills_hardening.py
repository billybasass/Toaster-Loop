"""Regression tests for the tools/skills deep-review hardening pass.

Covers (per fix from the review):
  1. SECURITY: model-supplied tool arguments that are not in the exposed
     schema are REJECTED at the runner chokepoint before the impl runs —
     private functools.partial knobs (block_sensitive, allow_private_networks,
     lsp_config, _ctx) cannot be forged through tool args.
  2. TOOL SCHEMA PARITY: every TOOL_IMPLS entry has exactly one schema
     (TOOL_SCHEMAS or CAPABILITY_TOOL_SCHEMAS) and vice versa; the dashboard's
     KNOWN_TOOLS list stays a subset of real tool names.
  3. read_file pages past the old 256 KB whole-file window (offset beyond it
     returns real lines instead of "no lines in range").
  4. patch reports FUZZY matching when the exact match failed; search_files
     refuses a missing root in content mode like name mode does.
  5. Skills: delete_skill tool + back-end, atomic save (no torn frontmatter),
     quoted frontmatter values parse, index orders most-recent-first.
  6. Dashboard skills API: list/get/put/delete round-trip through App.handle.

Hermetic: fake OpenAI server only; SABLE_HOME isolated per test by conftest.
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fake_server import FakeOpenAIServer  # noqa: E402
from sable.core.agent_runner import SableAgentClient  # noqa: E402
from sable.core import tools as uf_tools  # noqa: E402


class _Silent(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002
        pass


def _completion(message, finish="stop"):
    return json.dumps({
        "id": "x", "object": "chat.completion", "model": "m",
        "choices": [{"index": 0, "finish_reason": finish, "message": message}],
    }).encode()


def _scripted_server(script):
    """Fake server answering one scripted completion per POST.

    `script` is a list of message dicts; the last entry repeats if exhausted."""
    state = {"n": 0}

    class Handler(_Silent):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            msg = script[min(state["n"], len(script) - 1)]
            state["n"] += 1
            body = _completion(msg)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def _tool_call(cid, name, args):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


# ------------------------------------------------ 1. schema-args security gate

def test_forged_block_sensitive_is_rejected_not_executed(tmp_path):
    """read_file with block_sensitive=false must NOT reach the impl."""
    secret = tmp_path / ".env"
    secret.write_text("API_KEY=super-secret\n")
    executed = {"hit": False}

    def spy_read(path, **kwargs):
        executed["hit"] = True
        return uf_tools.read_file(path, **kwargs)

    script = [
        {"role": "assistant", "content": None,
         "tool_calls": [_tool_call("c1", "read_file",
                                   {"path": str(secret),
                                    "block_sensitive": False})]},
        {"role": "assistant", "content": "done"},
    ]
    with FakeOpenAIServer(handler_cls=_scripted_server(script)) as fs:
        client = SableAgentClient(base_url=fs.base_url, model="m",
                                    tool_impls={"read_file": spy_read})
        result = client.run("read the secret please")
    assert not executed["hit"], "impl ran despite forged kwarg"
    tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
    assert tool_msgs and tool_msgs[0]["content"].startswith(
        "error: bad arguments")
    assert "block_sensitive" in tool_msgs[0]["content"]
    assert "super-secret" not in json.dumps(result.messages)


def test_forged_private_network_and_lsp_config_rejected():
    """web_fetch/run_command/lsp knobs cannot be forged either."""
    script = [
        {"role": "assistant", "content": None,
         "tool_calls": [_tool_call("c1", "write_file",
                                   {"path": "x.py", "content": "print(1)",
                                    "lsp_config": {"enabled": True}})]},
        {"role": "assistant", "content": "done"},
    ]
    with FakeOpenAIServer(handler_cls=_scripted_server(script)) as fs:
        client = SableAgentClient(base_url=fs.base_url, model="m")
        result = client.run("write and lsp me")
    tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
    assert tool_msgs and tool_msgs[0]["content"].startswith(
        "error: bad arguments")


def test_valid_args_still_execute():
    """The gate must not break normal calls."""
    script = [
        {"role": "assistant", "content": None,
         "tool_calls": [_tool_call("c1", "todo_write",
                                   {"todos": [{"content": "step one"}]})]},
        {"role": "assistant", "content": "done"},
    ]
    with FakeOpenAIServer(handler_cls=_scripted_server(script)) as fs:
        client = SableAgentClient(base_url=fs.base_url, model="m")
        result = client.run("plan it")
    tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
    assert tool_msgs and not tool_msgs[0]["content"].startswith("error:")
    assert "step one" in tool_msgs[0]["content"]


def test_unschemaed_custom_impls_pass_through():
    """Custom tools without a discoverable schema keep working (escape hatch)."""
    script = [
        {"role": "assistant", "content": None,
         "tool_calls": [_tool_call("c1", "custom_tool",
                                   {"anything": 1})]},
        {"role": "assistant", "content": "done"},
    ]
    with FakeOpenAIServer(handler_cls=_scripted_server(script)) as fs:
        client = SableAgentClient(
            base_url=fs.base_url, model="m", tools=[],
            tool_impls={"custom_tool": lambda **kw: f"ran with {kw}"})
        result = client.run("go")
    tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
    assert tool_msgs and "ran with" in tool_msgs[0]["content"]


# --------------------------------------------------------- 2. schema parity

def test_tool_impls_and_schemas_have_exact_parity():
    impls = set(uf_tools.TOOL_IMPLS)
    schemas = ({s["function"]["name"] for s in uf_tools.TOOL_SCHEMAS}
               | set(uf_tools.CAPABILITY_TOOL_SCHEMAS))
    assert impls == schemas, (sorted(impls - schemas), sorted(schemas - impls))


def test_dashboard_known_tools_subset_of_real_tools():
    from sable.dashboard.app import DashboardApp

    real = set(uf_tools.TOOL_IMPLS)
    unknown = [t for t in DashboardApp.KNOWN_TOOLS if t not in real]
    assert not unknown, f"KNOWN_TOOLS drifted: {unknown}"


# ------------------------------------------------------------ 3. read paging

def test_read_file_pages_beyond_old_256kb_window(tmp_path):
    p = tmp_path / "big.txt"
    # ~400 KB of short lines: line 9000 sits far past the old 256 KB window.
    p.write_text("\n".join(f"line-{i}" for i in range(1, 10_001)) + "\n")
    out = uf_tools.read_file(str(p), offset=9000, limit=3)
    assert "line-9000" in out and "line-9002" in out
    assert "no lines" not in out


def test_read_file_offset_past_eof_reports_clearly(tmp_path):
    p = tmp_path / "small.txt"
    p.write_text("a\nb\nc\n")
    out = uf_tools.read_file(str(p), offset=99)
    assert "shorter" in out


def test_read_file_more_lines_hint_offsets_forward(tmp_path):
    p = tmp_path / "lines.txt"
    p.write_text("\n".join(f"L{i}" for i in range(1, 201)) + "\n")
    out = uf_tools.read_file(str(p), offset=1, limit=10)
    assert "continue with offset=11" in out


# ---------------------------------------------------------- 4. patch / search

def test_patch_fuzzy_match_is_annotated(tmp_path):
    p = tmp_path / "f.py"
    p.write_text("def f():\n    return 1\n")
    # Multi-line old_string whose lines match only whitespace-stripped:
    # exact substring fails, the fuzzy line-window hits => annotated.
    out = uf_tools.patch(str(p), "def f():\n        return 1",
                         "def f():\n    return 2")
    assert "FUZZY" in out
    assert "return 2" in p.read_text()


def test_patch_exact_match_not_annotated(tmp_path):
    p = tmp_path / "g.py"
    p.write_text("x = 1\n")
    out = uf_tools.patch(str(p), "x = 1", "x = 2")
    assert "FUZZY" not in out and out.startswith("patched")


def test_search_files_missing_root_errors(tmp_path):
    out = uf_tools.search_files("needle", str(tmp_path / "nope"))
    assert out.startswith("error: not a directory")


# --------------------------------------------------------------- 5. skills

def test_delete_skill_tool_and_backend():
    from sable.core.skills import load_skill, save_skill

    assert save_skill("stale-recipe", "old", "wrong facts")["ok"]
    assert load_skill("stale-recipe") is not None
    out = uf_tools.delete_skill(name="stale-recipe")
    assert out.startswith("skill deleted")
    assert load_skill("stale-recipe") is None
    assert "error:" in uf_tools.delete_skill(name="stale-recipe")


def test_save_skill_atomic_no_tmp_leftover():
    from sable.core.skills import save_skill, skills_dir

    assert save_skill("atomic", "d", "body")["ok"]
    leftovers = list(skills_dir().glob("*.tmp"))
    assert not leftovers, f"temp file left behind: {leftovers}"


def test_frontmatter_quoted_values_parse():
    from sable.core.skills import save_skill, load_skill

    res = save_skill("quoted", 'when x happens: do y', "body")  # colon desc
    assert res["ok"]
    s = load_skill("quoted")
    assert s is not None and ":" in s.description


def test_index_orders_most_recent_first_under_cap():
    from sable.core.config import SableConfig
    from sable.core.skills import index_section, save_skill

    # Same timestamp within a fast loop would tie; force ordering by writing
    # updated fields directly.
    save_skill("aaa-old", "old entry", "b")
    save_skill("zzz-new", "new entry", "b")
    from sable.core.skills import skills_dir
    (skills_dir() / "aaa-old.md").write_text(
        "---\nname: aaa-old\ndescription: old entry\n"
        "updated: 2020-01-01T00:00:00+00:00\n---\n\nb\n")
    (skills_dir() / "zzz-new.md").write_text(
        "---\nname: zzz-new\ndescription: new entry\n"
        "updated: 2099-01-01T00:00:00+00:00\n---\n\nb\n")
    # Cap fits exactly one "- name: desc" line (~21 chars each).
    sec = index_section(SableConfig(skills={"enabled": True,
                                              "index_total_cap_chars": 35}))
    assert "zzz-new" in sec          # recent survives the cap
    assert "aaa-old" not in sec      # stale entry omitted whole


# ------------------------------------------------------ 6. dashboard API

def _app():
    from sable.core.config import SableConfig
    from sable.dashboard.app import DashboardApp

    return DashboardApp(SableConfig())


def test_skills_api_roundtrip():
    app = _app()
    r = app.handle("GET", "/api/skills")
    assert r.status == 200
    assert r.body["skills"] == []

    r = app.handle("PUT", "/api/skills/api-made", body={
        "description": "made via API", "content": "# steps\n1. do it"})
    assert r.status == 200, r.body

    r = app.handle("GET", "/api/skills")
    names = [s["name"] for s in r.body["skills"]]
    assert "api-made" in names

    r = app.handle("GET", "/api/skills/api-made")
    assert r.status == 200 and "do it" in r.body["content"]

    r = app.handle("DELETE", "/api/skills/api-made")
    assert r.status == 200

    r = app.handle("GET", "/api/skills/api-made")
    assert r.status == 404


def test_skills_api_validation_errors():
    app = _app()
    r = app.handle("PUT", "/api/skills/Bad Name!", body={
        "description": "d", "content": "c"})
    assert r.status == 400  # invalid slug (also name check lowercases first)
    r = app.handle("PUT", "/api/skills/ok-name", body={
        "description": "d"})  # content missing
    assert r.status == 400
    r = app.handle("DELETE", "/api/skills/nothing-here")
    assert r.status == 404


# ------------------------------------------------- run_command argv hardening

def test_run_command_argv_mode_no_shell():
    """tools.run_command_argv_only: no shell — pipes/redirects become literal
    data, expansion tokens are refused, plain commands still run."""
    from sable.core.tools import run_command

    # default (shell): the pipeline executes
    r = run_command("echo hi | wc -w")
    assert "exit code: 0" in r and "stdout:\n1" in r
    # argv mode: `| wc -w` is literal echo data, never a second process
    r = run_command("echo hi | wc -w", argv_mode=True)
    assert "hi | wc -w" in r
    # expansion tokens refused rather than passed as silent literals
    assert "forbids shell expansion" in run_command("echo $HOME", argv_mode=True)
    assert "forbids shell expansion" in run_command("echo `id`", argv_mode=True)
    # plain argv commands work; unbalanced quotes fail cleanly
    assert "hello world" in run_command("echo hello world", argv_mode=True)
    assert "unbalanced quotes" in run_command('echo "unclosed', argv_mode=True)


def test_run_command_argv_binding_not_forgeable():
    """The bound argv knob is private: a model-supplied argv_mode arg must be
    refused by the schema gate, not override the binding."""
    script = [
        {"role": "assistant", "content": None,
         "tool_calls": [_tool_call("c1", "run_command",
                                   {"command": "echo hi",
                                    "argv_mode": False})]},
        {"role": "assistant", "content": "done"},
    ]
    with FakeOpenAIServer(handler_cls=_scripted_server(script)) as fs:
        client = SableAgentClient(
            base_url=fs.base_url, model="m",
            capabilities={"run_commands": True},
            run_command_argv_only=True)
        result = client.run("try to escape argv mode")
    tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
    assert tool_msgs and tool_msgs[0]["content"].startswith(
        "error: bad arguments")

"""WP12 plans: MCP control-plane mirror (schema array support + gating).

Reuses the harness from tests/test_mcp.py (FakeApp records in-process
DashboardApp.handle calls; injectable clock for TTL tests).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_mcp import (FakeApp, FakeResponse, call_tool, init,  # noqa: E402
                      make_server, rpc)

from sable.mcp.schema import MAX_ARRAY_ITEMS, validate_args
from sable.mcp.tools import build_tools

PLAN_TOOLS = {"plans_list", "plan_loop_status", "projects_list", "plan_add",
              "plan_update", "plan_set_status", "plan_depends",
              "plan_import", "plan_expand", "plan_loop_start",
              "plan_loop_cancel"}
MUTATING_PLAN_TOOLS = PLAN_TOOLS - {"plans_list", "plan_loop_status",
                                    "projects_list"}


class _Cfg:
    def __init__(self, plans):
        self.plans = plans


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("SABLE_HOME", str(h))
    return h


def _enabled_app(home, routes=None):
    srv, app = make_server(home, routes=routes)
    app.cfg = _Cfg({"enabled": True})       # gate reads app.cfg
    return srv, app


# ----------------------------------------------------------- tool table

def test_plan_tools_in_table_with_full_metadata():
    tools = {t.name: t for t in build_tools()}
    assert PLAN_TOOLS <= set(tools)
    for name in PLAN_TOOLS:
        t = tools[name]
        assert t.family == "plans", name
        props = t.input_schema["properties"]
        # closed schema everywhere
        assert t.input_schema.get("additionalProperties") is False, name
        if name in MUTATING_PLAN_TOOLS:
            assert t.mutating, name
            assert t.effect and t.risks and t.target_arg, name
            assert "confirm_token" in props, name
        else:
            assert not t.mutating, name
        # no paths as arguments anywhere: only ids/text/enums/numbers
        for arg, spec in props.items():
            if arg == "confirm_token":
                continue
            assert spec.get("type") in ("string", "integer", "boolean",
                                        "array"), (name, arg)


def test_deps_arg_is_real_closed_array():
    tools = {t.name: t for t in build_tools()}
    deps = tools["plan_add"].input_schema["properties"]["deps"]
    assert deps["type"] == "array"
    assert deps["items"]["type"] == "string"
    assert "pattern" in deps["items"]
    assert deps["maxItems"] <= MAX_ARRAY_ITEMS


# ---------------------------------------------------- array validator unit

_ARR = {"type": "object", "additionalProperties": False,
        "properties": {"deps": {
            "type": "array",
            "items": {"type": "string", "pattern": "^t[0-9]{1,6}$"},
            "maxItems": 4}},
        "required": []}


def test_array_valid():
    assert validate_args(_ARR, {"deps": ["t1", "t2"]}) is None
    assert validate_args(_ARR, {"deps": []}) is None


def test_array_wrong_item_type_rejected():
    err = validate_args(_ARR, {"deps": ["t1", 5]})
    assert err and "must be of type string" in err


def test_array_max_items_enforced():
    err = validate_args(_ARR, {"deps": [f"t{i}" for i in range(1, 6)]})
    assert err and "maximum is 4" in err


def test_array_item_pattern_enforced():
    err = validate_args(_ARR, {"deps": ["../etc/passwd"]})
    assert err and "pattern" in err


def test_array_not_a_list_rejected():
    err = validate_args(_ARR, {"deps": "t1"})
    assert err and "must be of type array" in err


def test_array_missing_items_schema_is_schema_error():
    schema = {"type": "object", "additionalProperties": False,
              "properties": {"x": {"type": "array"}}}
    err = validate_args(schema, {"x": ["a"]})
    assert err and "schema error" in err


def test_array_object_items_refused():
    schema = {"type": "object", "additionalProperties": False,
              "properties": {"x": {"type": "array",
                                   "items": {"type": "object"}}}}
    err = validate_args(schema, {"x": [{"a": 1}]})
    assert err and "schema error" in err


def test_array_max_items_type_checked():
    schema = {"type": "object", "additionalProperties": False,
              "properties": {"x": {"type": "array",
                                   "items": {"type": "string"},
                                   "maxItems": "4"}}}
    err = validate_args(schema, {"x": ["a"]})
    assert err and "schema error" in err


def test_scalar_behavior_untouched():
    """Pins pre-WP12 scalar semantics (the hygiene the spec demands)."""
    s = {"type": "object", "additionalProperties": False,
         "properties": {"a": {"type": "string"},
                        "n": {"type": "integer"},
                        "b": {"type": "boolean"}},
         "required": ["a"]}
    assert validate_args(s, {"a": "x", "n": 3, "b": True}) is None
    assert validate_args(s, {}) is not None                 # missing req
    assert validate_args(s, {"a": "x", "zzz": 1}) is not None  # unknown
    assert validate_args(s, {"a": 5}) is not None           # wrong type
    assert validate_args(s, {"a": "x", "n": True}) is not None


# ------------------------------------------------------------- gating

def test_plan_tools_absent_when_disabled(home):
    srv, app = make_server(home)          # FakeApp cfg=None => load fails -> off
    init(srv)
    names = {t["name"] for t in rpc(srv, 1, "tools/list")["result"]["tools"]}
    assert not (PLAN_TOOLS & names)


def test_plan_tools_listed_when_enabled(home):
    srv, _ = _enabled_app(home)
    init(srv)
    names = {t["name"] for t in rpc(srv, 1, "tools/list")["result"]["tools"]}
    assert PLAN_TOOLS <= names


def test_disabled_call_refused_unknown(home):
    srv, app = make_server(home)
    init(srv)
    out = call_tool(srv, 2, "plans_list", {"project": "default"})
    blob = json.dumps(out).lower()
    assert out.get("_rpc_error") is not None or out.get("_isError") is True
    assert "unknown tool" in blob or "disabled" in blob
    assert app.calls == []


import json  # noqa: E402  (used above)


# ------------------------------------------------- reads mirror the API

def test_read_tools_execute_immediately(home):
    srv, app = _enabled_app(home)
    init(srv)
    out = call_tool(srv, 3, "plans_list", {"project": "default"})
    assert out["_isError"] is False
    assert ("GET", "/api/plans/default", {}) in app.calls
    out = call_tool(srv, 4, "projects_list", {})
    assert out["_isError"] is False
    assert ("GET", "/api/projects", {}) in app.calls


# ------------------------------------- mutating plan->confirm matrix

_MATRIX = [
    # (tool, args)
    ("plan_add", {"project": "p1", "title": "x"}),
    ("plan_update", {"project": "p1", "task_id": "t1", "title": "y"}),
    ("plan_set_status", {"project": "p1", "task_id": "t1",
                         "status": "done"}),
    ("plan_depends", {"project": "p1", "task_id": "t2", "dep_id": "t1"}),
    ("plan_import", {"project": "p1", "text": "do things"}),
    ("plan_expand", {"project": "p1", "task_id": "t1"}),
    ("plan_loop_start", {"project": "p1"}),
    ("plan_loop_cancel", {"project": "p1"}),
]


@pytest.mark.parametrize("tool,args", _MATRIX)
def test_mutating_plan_confirm_execute_once_reuse_refused(home, tool, args):
    srv, app = _enabled_app(home)
    init(srv)
    # phase 1: plan only — nothing executed
    out = call_tool(srv, 10, tool, args)
    assert out["_isError"] is False and out["status"] == "plan"
    token = out["confirm_token"]
    assert out["plan"]["action"] == tool
    assert out["plan"]["exact_effect"] and out["plan"]["risks"]
    assert app.calls == []

    # phase 2: same args + token -> exactly one API call
    out2 = call_tool(srv, 11, tool, {**args, "confirm_token": token})
    assert out2["_isError"] is False and out2["status"] == "ok"
    assert len(app.calls) == 1

    # phase 3: reuse refused; still exactly one execution
    out3 = call_tool(srv, 12, tool, {**args, "confirm_token": token})
    assert out3["_isError"] is True and out3["status"] == "refused"
    assert len(app.calls) == 1


def test_plan_refusal_does_not_burn_pending_plan(home):
    """A wrong-token guess leaves the legitimate pending plan intact."""
    srv, app = _enabled_app(home)
    init(srv)
    out = call_tool(srv, 20, "plan_add", {"project": "p1", "title": "x"})
    token = out["confirm_token"]
    bad = call_tool(srv, 21, "plan_add",
                    {"project": "p1", "title": "x",
                     "confirm_token": "guessed"})
    assert bad["_isError"] is True
    good = call_tool(srv, 22, "plan_add",
                     {"project": "p1", "title": "x", "confirm_token": token})
    assert good["_isError"] is False and good["status"] == "ok"


def test_token_bound_to_args_deps_change_refused(home):
    srv, app = _enabled_app(home)
    init(srv)
    args = {"project": "p1", "title": "x", "deps": ["t1"]}
    out = call_tool(srv, 30, "plan_add", args)
    token = out["confirm_token"]
    # mutate the deps => fingerprint mismatch, refused...
    changed = {**args, "deps": ["t1", "t2"], "confirm_token": token}
    bad = call_tool(srv, 31, "plan_add", changed)
    assert bad["_isError"] is True
    assert "does not match" in bad["reason"]
    assert app.calls == []            # nothing executed
    # ...and the ORIGINAL plan still confirms fine (pending survived)
    ok = call_tool(srv, 32, "plan_add", {**args, "confirm_token": token})
    assert ok["_isError"] is False and ok["status"] == "ok"


def test_ttl_expiry_with_injected_clock(home):
    now = [1000.0]
    app = FakeApp(home)
    app.cfg = _Cfg({"enabled": True})
    from sable.mcp.server import McpServer

    srv = McpServer(app=app, ttl_s=60.0, clock=lambda: now[0])
    init(srv)
    out = call_tool(srv, 40, "plan_loop_start", {"project": "p1"})
    token = out["confirm_token"]
    now[0] += 61.0
    out2 = call_tool(srv, 41, "plan_loop_start",
                     {"project": "p1", "confirm_token": token})
    assert out2["_isError"] is True and "expired" in out2["reason"]
    assert app.calls == []


def test_api_error_maps_to_is_error(home):
    srv, _ = _enabled_app(
        home, routes={"GET /api/plans/p1": FakeResponse(404, {"error": {
            "code": "unknown_project", "message": "no project p1"}})})
    init(srv)
    out = call_tool(srv, 50, "plans_list", {"project": "p1"})
    assert out["_isError"] is True


def test_id_schemas_declare_patterns(home):
    """Defense-in-depth contract: every project/task arg schema carries an
    id pattern (the API layer enforces it; scalar MCP validation stays
    unchanged per spec, so enforcement lives one layer down)."""
    tools = {t.name: t for t in build_tools()}
    for name in PLAN_TOOLS:
        props = tools[name].input_schema["properties"]
        if "project" in props:
            assert "pattern" in props["project"], name
        for arg in ("task_id", "dep_id", "parent"):
            if arg in props:
                assert "pattern" in props[arg], (name, arg)
    # the pattern itself rejects paths
    import re

    rx = re.compile(
        tools["plans_list"].input_schema["properties"]["project"]["pattern"])
    assert not rx.search("../etc") and not rx.search("a/b")
    assert rx.fullmatch("proj-a")

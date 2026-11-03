"""WP12 plan_llm: deterministic + AI import, expand, salvage parsing.

The "served model" is a fake client_fn (same seam curate/judge.py uses);
no network, no GPU.
"""

from __future__ import annotations

import pytest

from toaster_loop.core.plan_llm import (PlanImportError, clean_text_input,
                                 parse_task_array, parse_task_lines,
                                 plan_expand, plan_import)
from toaster_loop.core.plans import add_task, get_task, load_plan, save_plan


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(h))
    return h


def _client(reply):
    """Judge-style client_fn(messages, model[, max_tokens]) -> fixed reply."""
    calls = {"n": 0}

    def fn(messages, model, max_tokens=None):
        calls["n"] += 1
        return reply
    fn.calls = calls
    return fn


# ------------------------------------------------------------------ input

def test_binary_input_rejected(home):
    with pytest.raises(PlanImportError) as e:
        plan_import(None, b"\x00\x01\x02binary\xff" * 50)
    assert "binary" in str(e.value).lower() or "not text" in str(e.value)


def test_latin1ish_bytes_decode_or_error_not_crash(home):
    # UTF-8 bytes are accepted; invalid UTF-8 reports a clean error.
    rep = plan_import(None, "Build the thing\nShip it\n".encode("utf-8"))
    assert rep["ok"] and len(rep["added"]) == 2
    with pytest.raises(PlanImportError):
        plan_import(None, b"\xff\xfe\x00bad")


# ------------------------------------------------------- deterministic text

def test_text_line_format_with_markers(home):
    brief = ("Set up auth [priority=high]\n"
             "- Add login form [deps=t1]\n"
             "Write tests [deps=t1,t2] [priority=low]\n"
             "# comment line ignored\n")
    rep = plan_import(None, brief)
    assert rep["ok"] and rep["source"] == "text"
    ids = [t["id"] for t in rep["added"]]
    assert ids == ["t1", "t2", "t3"]
    store = load_plan(None)
    assert get_task(store, "t2")["deps"] == ["t1"]
    assert get_task(store, "t3")["deps"] == ["t1", "t2"]
    assert get_task(store, "t1")["priority"] == "high"
    assert get_task(store, "t3")["priority"] == "low"


def test_text_num_tasks_cap(home):
    brief = "\n".join(f"task {i}" for i in range(50))
    rep = plan_import(None, brief, num_tasks=5)
    assert len(rep["added"]) == 5
    assert any("cap" in w or "truncat" in w.lower()
               for w in rep["warnings"])


def test_json_array_input(home):
    payload = ('[{"title": "alpha", "priority": "high"},'
               ' {"title": "beta", "deps": ["t1"]}]')
    rep = plan_import(None, payload)
    assert rep["ok"] and len(rep["added"]) == 2
    store = load_plan(None)
    assert get_task(store, "t2")["deps"] == ["t1"]


def test_deps_remap_to_minted_ids(home):
    """Intra-import deps reference the brief's own line numbering (tN)."""
    brief = "one\ntwo [deps=t1]\nthree [deps=t1,t2]\n"
    rep = plan_import(None, brief)
    assert rep["ok"]
    store = load_plan(None)
    assert get_task(store, "t2")["deps"] == ["t1"]
    assert get_task(store, "t3")["deps"] == ["t1", "t2"]


def test_ai_alias_ids_remapped(home):
    """Model-declared ids (t1..tN or s1..) remap to minted ids."""
    reply = '[{"id": "a", "title": "first"}, ' \
            '{"id": "b", "title": "second", "deps": ["a"]}]'
    rep = plan_import(None, "brief", use_ai=True, client_fn=_client(reply),
                      model="m")
    assert rep["ok"]
    store = load_plan(None)
    assert get_task(store, "t2")["deps"] == ["t1"]


# ------------------------------------------------------------------ AI mode

def test_ai_strict_json_reply(home):
    reply = ('Here you go:\n```json\n[{"title": "Design API", '
             '"priority": "high"}, {"title": "Implement", '
             '"deps": ["t1"]}]\n```\nEnjoy!')
    client = _client(reply)
    rep = plan_import(None, "some brief text", use_ai=True,
                      client_fn=client, model="m")
    assert rep["ok"] and rep["source"] == "ai"
    assert [t["title"] for t in rep["added"]] == ["Design API", "Implement"]
    assert client.calls["n"] == 1          # single-shot


def test_ai_malformed_output_salvages(home):
    """Trailing comma + prose around the array: salvage, never crash."""
    reply = ('Sure! [{"title": "one"}, {"title": "two",},]\n'
             'Let me know if you need more.')
    rep = plan_import(None, "brief", use_ai=True, client_fn=_client(reply),
                      model="m")
    assert rep["ok"], rep["warnings"]
    assert [t["title"] for t in rep["added"]] == ["one", "two"]


def test_ai_unparseable_returns_salvage_report_not_crash(home):
    rep = plan_import(None, "brief", use_ai=True,
                      client_fn=_client("I cannot help with that."),
                      model="m")
    assert not rep["ok"]
    assert rep["added"] == [] and rep["warnings"]   # report explains


def test_ai_item_salvage_skips_bad_items(home):
    reply = '[{"title": "good"}, {"no_title": 1}, ' \
            '{"title": "also good", "priority": "WAT"}]'
    rep = plan_import(None, "brief", use_ai=True, client_fn=_client(reply),
                      model="m")
    assert rep["ok"]
    titles = [t["title"] for t in rep["added"]]
    assert titles == ["good", "also good"]           # bad item dropped
    assert any("priority" in w.lower() or "default" in w.lower()
               for w in rep["warnings"])             # warned + defaulted


def test_ai_num_tasks_cap_applies(home):
    items = ", ".join(f'{{"title": "t{i}"}}' for i in range(20))
    rep = plan_import(None, "brief", use_ai=True, num_tasks=3,
                      client_fn=_client(f"[{items}]"), model="m")
    assert len(rep["added"]) == 3


def test_ai_requires_client(home):
    with pytest.raises(PlanImportError):
        plan_import(None, "brief", use_ai=True)      # no client_fn


def test_ai_transport_failure_never_crashes(home):
    def boom(messages, model, max_tokens=None):
        raise ConnectionError("server down")

    rep = plan_import(None, "brief", use_ai=True, client_fn=boom, model="m")
    assert not rep["ok"]
    assert any("model call failed" in w for w in rep["warnings"])


# ----------------------------------------------------------------- expand

def test_expand_attaches_subtasks_parent_kept(home):
    store = load_plan(None)
    add_task(store, {"title": "big thing", "description": "huge"})
    from toaster_loop.core.plans import set_status as _ss
    _ss(store, "t1", "in_progress")
    save_plan(store)

    reply = ('[{"title": "step one"}, {"title": "step two", '
             '"deps": ["s1"]}]')
    rep = plan_expand(None, "t1", num=3, client_fn=_client(reply), model="m")
    assert rep["ok"] and len(rep["added"]) == 2
    store = load_plan(None)
    parent = get_task(store, "t1")
    assert parent["status"] == "in_progress"          # untouched
    subs = [t for t in store["tasks"] if t["parent"] == "t1"]
    assert len(subs) == 2


def test_expand_unknown_task(home):
    with pytest.raises(PlanImportError):
        plan_expand(None, "t9", client_fn=_client("[]"), model="m")


# ------------------------------------------------------- parse primitives

def test_parse_task_lines_markers_table():
    lines, warnings = parse_task_lines(
        "plain\n"
        "with deps [deps=t1,t2]\n"
        "prio [priority=HIGH]\n"
        "bad prio [priority=nope]\n")
    assert [f["title"] for f in lines] == ["plain", "with deps", "prio",
                                           "bad prio"]
    assert lines[1]["deps"] == ["t1", "t2"]
    assert lines[2]["priority"] == "high"
    assert lines[3]["priority"] == "medium"       # bad value -> default+warn
    assert warnings


def test_parse_task_array_recovers_fenced_object_stream():
    text = ('{"title": "a"}\n'
            '{"title": "b", "details": "x"}')
    items, _w = parse_task_array(text)
    assert [o["title"] for o in items] == ["a", "b"]


def test_clean_text_input_rejects_nul():
    with pytest.raises(PlanImportError):
        clean_text_input("has\x00nul")

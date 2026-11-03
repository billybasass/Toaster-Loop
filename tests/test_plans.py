"""WP12 plans: core store + DAG logic (core/plans.py).

Hermetic: every helper passes an explicit tmp `home`; the autouse TOASTER_LOOP_HOME
fixture in conftest.py keeps stray resolution out of the real home.
"""

from __future__ import annotations

import json
import threading

import pytest

from toaster_loop.core.plans import (PlanError, add_dependency, add_task,
                              blockers, build_task_prompt, counts,
                              find_cycles, get_task, load_plan, next_task,
                              next_task_id, plan_path, ready, record_run,
                              remove_dependency, remove_task, save_plan,
                              set_status, summary, validate, validate_store)


@pytest.fixture()
def home(tmp_path):
    h = tmp_path / "toaster_loop-home"
    h.mkdir()
    return h


def _add(store, title, **kw):
    fields = {"title": title}
    fields.update(kw)
    return add_task(store, fields)


# ---------------------------------------------------------------- storage

def test_store_round_trip_atomic_default_project(home):
    store = load_plan(None, home=home)
    assert store["version"] == 1 and store["tasks"] == []
    _add(store, "first")
    save_plan(store, home=home)
    again = load_plan(None, home=home)
    assert [t["title"] for t in again["tasks"]] == ["first"]
    # file lives at <home>/plans/default.json; no tmp litter left behind
    p = plan_path(None, home=home)
    assert p.is_file() and p.parent.name == "plans"
    assert not list(p.parent.glob("*.tmp"))


def test_two_projects_independent(home):
    a = load_plan("proj-a", home=home)
    _add(a, "a task")
    save_plan(a, home=home)
    b = load_plan("proj-b", home=home)
    assert b["tasks"] == []          # separate files, no bleed-through
    _add(b, "b task")
    save_plan(b, home=home)
    assert [t["title"] for t in load_plan("proj-a", home=home)["tasks"]] == \
        ["a task"]
    assert [t["title"] for t in load_plan("proj-b", home=home)["tasks"]] == \
        ["b task"]


def test_corrupt_file_reads_empty_with_warning(home, caplog):
    p = plan_path("p1", home=home)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{not json at all")
    import logging

    with caplog.at_level(logging.WARNING):
        store = load_plan("p1", home=home)
    assert store["tasks"] == []       # never raises into callers
    assert any("corrupt" in r.message.lower() or "unreadable" in r.message.lower()
               for r in caplog.records)


def test_unknown_keys_dropped_on_validation(home):
    data = {"version": 1, "project": None, "updated": "x",
            "tasks": [{"id": "t1", "title": "ok", "bogus_key": "drop me"}]}
    store = validate_store(data)
    t = store["tasks"][0]
    assert "bogus_key" not in t
    assert t["status"] == "pending" and t["deps"] == []


def test_non_string_ids_rejected(home):
    data = {"version": 1, "tasks": [
        {"id": 5, "title": "numeric id"},      # string ids only
        {"id": "t2", "title": "fine"}]}
    store = validate_store(data)
    assert [t["id"] for t in store["tasks"]] == ["t2"]


def test_two_threads_add_loses_nothing(home):
    """Concurrent add_task+save_plan under the module lock: both land."""
    start = threading.Barrier(2)

    def worker(n):
        start.wait()
        for i in range(10):
            store = load_plan(f"p{n}", home=home)
            _add(store, f"t{n}-{i}")
            save_plan(store, home=home)

    ts = [threading.Thread(target=worker, args=(n,)) for n in ("a", "b")]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    # Same project from two threads (contention on one file):
    start2 = threading.Barrier(2)

    def same(n):
        start2.wait()
        for i in range(10):
            with __import__("toaster_loop.core.plans", fromlist=["_LOCK"])._LOCK:
                store = load_plan("shared", home=home)
                _add(store, f"s{n}-{i}")
                save_plan(store, home=home)

    ts2 = [threading.Thread(target=same, args=(n,)) for n in ("x", "y")]
    for t in ts2:
        t.start()
    for t in ts2:
        t.join()
    shared = load_plan("shared", home=home)
    assert len(shared["tasks"]) == 20


def test_invalid_project_id_rejected(home):
    with pytest.raises(ValueError):
        plan_path("../evil", home=home)
    with pytest.raises(ValueError):
        plan_path("has space", home=home)


# ---------------------------------------------------------------- DAG logic

def test_next_task_priority_then_oldest_id(home):
    store = load_plan(None, home=home)
    _add(store, "med", priority="medium")            # t1
    _add(store, "high", priority="high")             # t2
    _add(store, "low", priority="low")               # t3
    assert next_task(store)["id"] == "t2"            # priority first
    set_status(store, "t2", "done")
    assert next_task(store)["id"] == "t1"            # then medium
    # equal priority: oldest id (lowest t-number) wins
    store2 = load_plan("p2", home=home)
    _add(store2, "b-first", priority="high")         # t1
    _add(store2, "c-second", priority="high")        # t2
    assert next_task(store2)["id"] == "t1"


def test_blocked_tasks_never_ready(home):
    """A task with unsatisfied deps is excluded from ready()/next_task."""
    store = load_plan(None, home=home)
    _add(store, "dep-a")                             # t1
    _add(store, "blocked-high", priority="high", deps=["t1"])  # t2
    _add(store, "free-low", priority="low")          # t3
    r = ready(store)
    assert [t["id"] for t in r] == ["t1", "t3"]      # t2 never listed
    assert next_task(store)["id"] == "t1"


def test_deps_satisfied_only_by_done_or_cancelled(home):
    store = load_plan(None, home=home)
    _add(store, "a")                                 # t1
    _add(store, "b", deps=["t1"])                    # t2
    assert [t["id"] for t in ready(store)] == ["t1"]
    set_status(store, "t1", "in_progress")           # NOT satisfying
    assert [t["id"] for t in ready(store)] == ["t1"]  # only t1 itself
    set_status(store, "t1", "cancelled")             # cancelled satisfies
    assert [t["id"] for t in ready(store)] == ["t2"]


def test_parent_done_cascades_to_subtasks(home):
    store = load_plan(None, home=home)
    _add(store, "parent")                            # t1
    _add(store, "child", parent="t1")                # t2
    _add(store, "grandchild", parent="t2")           # t3
    set_status(store, "t1", "done")
    assert get_task(store, "t2")["status"] == "done"     # children cascade
    assert get_task(store, "t3")["status"] == "done"     # ... recursively


def test_cancelled_parent_does_not_cascade(home):
    store = load_plan(None, home=home)
    _add(store, "parent")                            # t1
    _add(store, "child", parent="t1")                # t2
    set_status(store, "t1", "cancelled")
    assert get_task(store, "t2")["status"] == "pending"


def test_blocking_lists_unfinished_deps(home):
    store = load_plan(None, home=home)
    _add(store, "a")                                 # t1
    _add(store, "b")                                 # t2
    _add(store, "c", deps=["t1", "t2"])              # t3
    set_status(store, "t1", "done")
    bl = blockers(store, "t3")
    assert [t["id"] for t in bl] == ["t2"]


def test_cycle_detection_and_refusal(home):
    store = load_plan(None, home=home)
    _add(store, "a")                                 # t1
    _add(store, "b")                                 # t2
    add_dependency(store, "t1", "t2")
    with pytest.raises(PlanError):                   # cycle refused up front
        add_dependency(store, "t2", "t1")
    with pytest.raises(PlanError):                   # self-dep refused
        add_dependency(store, "t1", "t1")
    get_task(store, "t2")["deps"].append("t1")       # force a cycle in raw
    report = validate(store)
    assert not report["ok"] and report["cycles"]


def test_validate_dangling_refs(home):
    store = load_plan(None, home=home)
    _add(store, "a", deps=["t99"])                   # dangling dep
    _add(store, "b", parent="t98")                   # dangling parent
    report = validate(store)
    assert not report["ok"]
    kinds = {d.get("kind") for d in report["dangling_deps"]} | \
            {d.get("kind") for d in report.get("dangling_parents", [])}
    assert "dep" in kinds or any("t99" in json.dumps(d) for d in
                                 report["dangling_deps"])


def test_remove_task_rewrites_dependents(home):
    store = load_plan(None, home=home)
    _add(store, "a")                                 # t1
    _add(store, "b", deps=["t1"])                    # t2
    _add(store, "c", parent="t1")                    # t3
    removed, rewired = remove_task(store, "t1")
    assert removed and set(rewired) >= {"t2"}
    assert get_task(store, "t2")["deps"] == []
    assert get_task(store, "t3")["parent"] is None   # orphaned subtask


def test_remove_dependency(home):
    store = load_plan(None, home=home)
    _add(store, "a")                                 # t1
    _add(store, "b", deps=["t1"])                    # t2
    remove_dependency(store, "t2", "t1")
    assert get_task(store, "t2")["deps"] == []


def test_set_status_valid_transitions_only(home):
    store = load_plan(None, home=home)
    _add(store, "a")
    with pytest.raises(PlanError):
        set_status(store, "t1", "shipped")
    with pytest.raises(PlanError):
        set_status(store, "nope", "done")


def test_set_status_note_appends_timestamped_line(home):
    store = load_plan(None, home=home)
    _add(store, "a", details="base detail")
    t = set_status(store, "t1", "in_progress", note="started work")
    assert "base detail" in t["details"]
    assert "] started work" in t["details"]          # "[iso] note" shape


def test_record_run_capped(home):
    store = load_plan(None, home=home)
    _add(store, "a")
    for i in range(25):
        record_run(store, "t1", trace_id=f"tr{i}")
    assert len(get_task(store, "t1")["runs"]) == 20   # MAX_RUNS_PER_TASK
    assert get_task(store, "t1")["runs"][-1]["trace_id"] == "tr24"


def test_counts(home):
    store = load_plan(None, home=home)
    _add(store, "a")                                 # t1 pending
    _add(store, "b")                                 # t2
    set_status(store, "t1", "done")
    c = counts(store)
    assert c["total"] == 2 and c["done"] == 1 and c["pending"] == 1


# ------------------------------------------------------------- rendering

def test_summary_hard_cap(home):
    store = load_plan(None, home=home)
    for i in range(30):
        _add(store, "a very long task title that just keeps going " + str(i))
    s = summary(store, max_chars=200)
    assert len(s) <= 200
    assert s.startswith("## Project plan")


def test_build_task_prompt_sections_and_dep_block(home):
    store = load_plan(None, home=home)
    _add(store, "dep task", description="dep desc",
         details="first detail line\nsecond line")   # t1
    set_status(store, "t1", "done")
    _add(store, "main", description="do the thing", details="the how",
         test_strategy="run pytest")                 # t2 deps none yet
    add_dependency(store, "t2", "t1")
    prompt = build_task_prompt(store, "t2")
    assert "# Task t2: main" in prompt
    assert "do the thing" in prompt and "the how" in prompt
    assert "## How to verify\nrun pytest" in prompt
    assert "Completed dependencies:" in prompt
    assert "dep task — first detail line" in prompt
    assert "plan_progress" in prompt                 # closing instruction


def test_build_task_prompt_dep_block_capped(home):
    store = load_plan(None, home=home)
    for i in range(40):
        _add(store, f"dep {i}", details="x" * 100)
        set_status(store, f"t{i + 1}", "done")
    _add(store, "big", deps=[f"t{i}" for i in range(1, 41)])
    prompt = build_task_prompt(store, "t41")
    assert "Completed dependencies:" in prompt
    # cap enforced: the truncation marker appears and total stays bounded
    assert "truncated" in prompt
    assert len(prompt) < 6000


def test_build_task_prompt_unknown(home):
    store = load_plan(None, home=home)
    with pytest.raises(PlanError):
        build_task_prompt(store, "t9")


def test_next_task_id_sequence(home):
    store = load_plan(None, home=home)
    assert next_task_id(store) == "t1"
    _add(store, "x")                                 # takes t1
    assert next_task_id(store) == "t2"
    store["tasks"].append({"id": "t7", "title": "y", "status": "pending",
                           "priority": "medium", "deps": [], "parent": None,
                           "description": "", "details": "",
                           "test_strategy": "", "profile": None,
                           "labels": [], "created": "", "updated": "",
                           "runs": []})
    # documented rule: smallest unused id (t2 is free) — never collides
    assert next_task_id(store) == "t2"
    _add(store, "z")                                  # add mints t2
    assert get_task(store, "t2") is not None
    ids = [t["id"] for t in store["tasks"]]
    assert len(ids) == len(set(ids))                  # unique ids kept


def test_find_cycles_empty_when_clean(home):
    store = load_plan(None, home=home)
    _add(store, "a")                                 # t1
    _add(store, "b", deps=["t1"])                    # t2
    add_dependency(store, "t2", "t1") if False else None
    assert find_cycles(store) == []

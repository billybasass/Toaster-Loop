"""Backlog #8 — batch trace generation with content-based resume.

Covers: happy path (N tasks -> N traces + incremental checkpoint), crash +
--resume skipping by CONTENT (not index, survives reordering/duplication),
infra failures retried on resume while model failures count as done, input
parsing (bare strings / blank lines tolerated, malformed JSON names the
line). The loop is driven through core.batch.run_batch with a run_fn wired
to a real SableAgentClient + TraceRecorder against FakeOpenAIServer — no
subprocesses needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fake_server import FakeOpenAIServer, make_message_handler  # noqa: E402
from sable.core.agent_runner import SableAgentClient  # noqa: E402
from sable.core.batch import (  # noqa: E402
    BatchInputError,
    checkpoint_path,
    load_checkpoint,
    load_tasks,
    run_batch,
    scan_completed,
    task_key,
)
from sable.store.recorder import TraceRecorder  # noqa: E402


@pytest.fixture()
def sable_home(tmp_path, monkeypatch):
    home = tmp_path / "uf-home"
    monkeypatch.setenv("SABLE_HOME", str(home))
    return home


@pytest.fixture(autouse=True)
def _no_sleeps(monkeypatch):
    """Retry backoff must not slow the suite down (cf. test_resilience)."""
    from sable.core import agent_runner as ar
    monkeypatch.setattr(ar.SableAgentClient, "_sleep",
                        staticmethod(lambda s: None))


# ----------------------------------------------------------------- helpers

OK_MSG = {"role": "assistant", "content": "all done"}


def make_run_fn(home: Path, base_url: str, max_iterations: int = 3):
    """run_fn(contract) -> real runner+recorder per task (like the CLI)."""

    def run_fn(task):
        rec = TraceRecorder(home=home)
        client = SableAgentClient(base_url=base_url, model="fake-model",
                                    recorder=rec, max_iterations=max_iterations)
        result = client.run(task)
        return getattr(result, "trace_id", None), result.failure_reason

    return run_fn


def write_tasks(path: Path, lines: list) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def seed_trace(home: Path, task: str, failure_reason):
    """Record one trace with a given failure_reason (no server involved)."""
    rec = TraceRecorder(home=home)
    row = rec.record(task=task, messages=[{"role": "user", "content": task}],
                     backend="sable", model_alias="m",
                     base_url="http://127.0.0.1:1/v1", completed=False,
                     failure_reason=failure_reason)
    assert row is not None
    return row


def cp_tasks(home: Path, run_name: str) -> list:
    cp = load_checkpoint(checkpoint_path(home, run_name))
    return (cp or {}).get("tasks", [])


# -------------------------------------------------------------- input parsing

def test_load_tasks_object_bare_and_blank_lines(tmp_path):
    p = tmp_path / "tasks.jsonl"
    p.write_text(
        '{"task": "alpha"}\n'
        "\n   \n"
        '"beta bare string"\n'
        '{"task": "gamma", "extra": "ignored in v0"}\n',
        encoding="utf-8",
    )
    assert load_tasks(p) == ["alpha", "beta bare string", "gamma"]


def test_load_tasks_malformed_json_names_line(tmp_path):
    p = tmp_path / "tasks.jsonl"
    write_tasks(p, ['{"task": "ok"}', "{not json", '{"task": "x"}'])
    with pytest.raises(BatchInputError) as ei:
        load_tasks(p)
    assert ":2:" in str(ei.value)


def test_load_tasks_object_without_task_names_line(tmp_path):
    p = tmp_path / "tasks.jsonl"
    write_tasks(p, ["42"])
    with pytest.raises(BatchInputError) as ei:
        load_tasks(p)
    assert ":1:" in str(ei.value)


def test_load_tasks_missing_file():
    with pytest.raises(BatchInputError, match="not found"):
        load_tasks(Path("/nonexistent/tasks.jsonl"))


# ------------------------------------------------------------------ happy path

def test_batch_happy_path_traces_and_incremental_checkpoint(sable_home):
    tasks = ["task alpha", "task beta", "task gamma"]
    handler = make_message_handler({"*": OK_MSG})
    counts = []  # checkpoint record count observed after each task

    def run_fn(task):
        tid, reason = make_run_fn(sable_home, srv.base_url)(task)
        counts.append(len(cp_tasks(sable_home, "run1")))
        return tid, reason

    with FakeOpenAIServer(handler_cls=handler) as srv:
        lines = []
        rc = run_batch(tasks, home=sable_home, run_name="run1",
                       run_fn=run_fn, print_fn=lambda *a: lines.append(a[0]))

    assert rc == 0
    # traces: three ordinary buffer records, one per task
    traces = list((sable_home / "buffer").rglob("*.jsonl"))
    assert len(traces) == 3
    # checkpoint grew incrementally: 1, 2, 3 (written AFTER each task)
    assert counts == [0, 1, 2]  # observed before this task's own save
    final = cp_tasks(sable_home, "run1")
    assert len(final) == 3
    assert {r["task"] for r in final} == set(tasks)
    assert all(r["trace_id"] and not r["failure_reason"] for r in final)
    # progress lines + summary
    assert any("batch 2/3: task beta ... ok" in l for l in lines)
    assert "ran=3 skipped=0 retried-infra=0 failed-model=0 failed-infra=0" \
        in lines[-1]


def test_batch_no_resume_ignores_prior_traces(sable_home):
    seed_trace(sable_home, "already done", None)
    seen = []

    def run_fn(task):
        seen.append(task)
        return "t1", None

    rc = run_batch(["already done"], home=sable_home, run_name="r",
                   resume=False, run_fn=run_fn, print_fn=lambda *a: None)
    assert rc == 0
    assert seen == ["already done"]  # fresh intent: nothing skipped


# ------------------------------------------------------------------- resume

def test_resume_after_crash_skips_completed_by_content(sable_home):
    tasks = ["crash one", "crash two", "crash three"]
    handler = make_message_handler({"*": OK_MSG})

    def crashy_run_fn(task):
        if task == "crash three":
            raise SystemExit(9)  # simulate process death mid-task
        return make_run_fn(sable_home, srv.base_url)(task)

    with FakeOpenAIServer(handler_cls=handler) as srv:
        with pytest.raises(SystemExit):
            run_batch(tasks, home=sable_home, run_name="crashrun",
                      run_fn=crashy_run_fn, print_fn=lambda *a: None)
        # crash left exactly the two finished tasks checkpointed
        assert len(cp_tasks(sable_home, "crashrun")) == 2

        handler.requests.clear()
        lines = []
        rc = run_batch(tasks, home=sable_home, run_name="crashrun",
                       resume=True,
                       run_fn=make_run_fn(sable_home, srv.base_url),
                       print_fn=lambda *a: lines.append(a[0]))

    assert rc == 0
    # only task 3's request hit the server this time (one chat call)
    last_users = [
        next((m["content"] for m in reversed(p["messages"])
              if m.get("role") == "user"), "")
        for p in handler.requests
    ]
    assert last_users == ["crash three"]
    assert any("skipped(resume)" in l and "crash one" in l for l in lines)
    assert any("skipped(resume)" in l and "crash two" in l for l in lines)
    assert len(cp_tasks(sable_home, "crashrun")) == 3


def test_resume_matches_reordered_and_duplicated_lines(sable_home):
    handler = make_message_handler({"*": OK_MSG})
    with FakeOpenAIServer(handler_cls=handler) as srv:
        run_batch(["task A", "task B"], home=sable_home, run_name="ord",
                  run_fn=make_run_fn(sable_home, srv.base_url),
                  print_fn=lambda *a: None)

        handler.requests.clear()
        # reordered + duplicated input; content matching, not index
        lines = []
        rc = run_batch(["task B", "task A", "task B"], home=sable_home,
                       run_name="ord", resume=True,
                       run_fn=make_run_fn(sable_home, srv.base_url),
                       print_fn=lambda *a: lines.append(a[0]))
    assert rc == 0
    assert sum("skipped(resume)" in l for l in lines) == 3
    # no new requests reached the server on resume
    assert handler.requests == []


def test_infra_failure_retried_on_resume(sable_home):
    seed_trace(sable_home, "flaky task", "backend_error:connection")
    completed = scan_completed(sable_home)
    assert task_key("flaky task") not in completed  # infra != done

    seen = []

    def ok_run_fn(task):
        seen.append(task)
        return "t-new", None

    rc = run_batch(["flaky task"], home=sable_home, run_name="infra",
                   resume=True, run_fn=ok_run_fn, print_fn=lambda *a: None)
    assert seen == ["flaky task"]  # re-run, not skipped
    assert rc == 0


def test_infra_failure_exits_1_and_is_not_checkpointed_done(sable_home):
    def dead_run_fn(task):
        return make_run_fn(sable_home, "http://127.0.0.1:9/v1")(task)

    lines = []
    rc = run_batch(["will fail infra"], home=sable_home, run_name="dead",
                   resume=False, run_fn=dead_run_fn,
                   print_fn=lambda *a: lines.append(a[0]))
    assert rc == 1  # infra failure remaining -> exit 1
    assert any("failed(backend_error:connection)" in l for l in lines)
    # the trace exists but is infra-marked -> a resume scan sees it as undone
    assert task_key("will fail infra") not in scan_completed(sable_home)


def test_model_failure_counts_as_done_not_retried(sable_home):
    for reason in ("guardrail_halt", "max_iterations"):
        seed_trace(sable_home, f"model-ish {reason}", reason)
    completed = scan_completed(sable_home)
    assert task_key("model-ish guardrail_halt") in completed
    assert task_key("model-ish max_iterations") in completed

    seen = []

    def run_fn(task):
        seen.append(task)
        return None, None

    lines = []
    rc = run_batch(["model-ish guardrail_halt", "model-ish max_iterations"],
                   home=sable_home, run_name="modelfail", resume=True,
                   run_fn=run_fn, print_fn=lambda *a: lines.append(a[0]))
    assert seen == []  # signal, not transport noise -> done
    assert rc == 0
    assert sum("skipped(resume)" in l for l in lines) == 2


def test_model_failure_via_real_runner_skipped_on_resume(sable_home):
    """End-to-end: a never-converging server yields max_iterations; resume
    treats the task as done."""
    handler = make_message_handler(
        {"*": {"role": "assistant", "content": None,
               "tool_calls": [{"id": "c", "type": "function",
                               "function": {"name": "list_dir",
                                            "arguments": "{}"}}]}})
    tasks = ["loop task"]
    with FakeOpenAIServer(handler_cls=handler) as srv:
        lines = []
        rc = run_batch(tasks, home=sable_home, run_name="looprun",
                       run_fn=make_run_fn(sable_home, srv.base_url,
                                          max_iterations=2),
                       print_fn=lambda *a: lines.append(a[0]))
        assert rc == 0  # model failure is terminal, not an error exit
        assert any("failed(max_iterations)" in l for l in lines)

        handler.requests.clear()
        lines = []
        rc = run_batch(tasks, home=sable_home, run_name="looprun",
                       resume=True,
                       run_fn=make_run_fn(sable_home, srv.base_url),
                       print_fn=lambda *a: lines.append(a[0]))
    assert rc == 0
    assert handler.requests == []
    assert any("skipped(resume)" in l for l in lines)


# --------------------------------------------------------------- checkpoint

def test_checkpoint_atomic_write_and_schema(sable_home):
    def run_fn(task):
        return "trace-xyz", None

    run_batch(["one"], home=sable_home, run_name="cp", resume=False,
              run_fn=run_fn, input_path="/tmp/tasks.jsonl",
              print_fn=lambda *a: None)
    cp_file = checkpoint_path(sable_home, "cp")
    data = json.loads(cp_file.read_text())
    assert data["schema_version"] == 1
    assert data["run_name"] == "cp"
    assert data["input"] == "/tmp/tasks.jsonl"
    assert data["started_at"] and data["last_updated"]
    rec = data["tasks"][0]
    assert rec["task"] == "one"
    assert rec["trace_id"] == "trace-xyz"
    assert rec["failure_reason"] is None
    assert rec["completed_at"]
    assert len(rec["prompt_hash"]) == 64
    assert not list(cp_file.parent.glob("*.tmp"))  # tmp file replaced


def test_corrupt_checkpoint_degrades_to_content_scan(sable_home):
    cp = checkpoint_path(sable_home, "bad")
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text("{not json", encoding="utf-8")
    seed_trace(sable_home, "seeded task", None)

    seen = []

    def run_fn(task):
        seen.append(task)
        return "t", None

    rc = run_batch(["seeded task"], home=sable_home, run_name="bad",
                   resume=True, run_fn=run_fn, print_fn=lambda *a: None)
    assert rc == 0
    assert seen == []  # scan still wins: content is truth


def test_resume_bounds_scan_to_checkpoint_started_at(sable_home):
    """An old trace (pre-batch) with identical text must not suppress a fresh
    batch task when a checkpoint bounds the scan window."""
    row = seed_trace(sable_home, "classic task", None)
    # backdate the seeded trace header by a year
    path = next((sable_home / "buffer").rglob(f"{row.trace_id}.jsonl"))
    lines = path.read_text().splitlines()
    header = json.loads(lines[0])
    header["timestamp"] = "2020-01-01T00:00:00+00:00"
    lines[0] = json.dumps(header)
    path.write_text("\n".join(lines) + "\n")

    # checkpoint claiming a recent started_at -> old trace out of window
    from sable.core.batch import save_checkpoint
    save_checkpoint(checkpoint_path(sable_home, "win"), {
        "schema_version": 1, "run_name": "win", "input": None,
        "started_at": "2025-01-01T00:00:00+00:00", "tasks": [],
    })

    seen = []

    def run_fn(task):
        seen.append(task)
        return "t2", None

    rc = run_batch(["classic task"], home=sable_home, run_name="win",
                   resume=True, run_fn=run_fn, print_fn=lambda *a: None)
    assert rc == 0
    assert seen == ["classic task"]  # re-run: prior trace predates the run

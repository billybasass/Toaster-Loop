"""Hermeticity for the whole test suite.

Autouse fixture: every test runs with TOASTER_LOOP_HOME pointed at its own tmp dir,
so the experience buffer never touches the developer's real ~/.toaster-loop.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_toaster_loop_home(tmp_path, monkeypatch):
    monkeypatch.setenv("TOASTER_LOOP_HOME", str(tmp_path / ".toaster-loop-test-home"))
    yield


@pytest.fixture(autouse=True)
def _reset_runtime_counters():
    """Per-test isolation for in-process reliability state (pass 4 #28).

    The circuit-breaker registry and the supervisor crash-loop budget are
    module globals; without this reset a test that fails a backend URL five
    times would open a breaker that poisons later tests reusing that key."""
    from toaster_loop.core import circuit
    from toaster_loop.serving import supervisor

    def _clear():
        with circuit._registry_lock:
            circuit._registry.clear()
        with supervisor._crash_lock:
            supervisor._crash_failures.clear()

    _clear()
    yield
    _clear()

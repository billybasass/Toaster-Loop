"""Hermeticity for the whole test suite.

Autouse fixture: every test runs with SABLE_HOME pointed at its own tmp dir,
so the experience buffer never touches the developer's real ~/.sable.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_sable_home(tmp_path, monkeypatch):
    monkeypatch.setenv("SABLE_HOME", str(tmp_path / ".sable-test-home"))
    yield


@pytest.fixture(autouse=True)
def _reset_runtime_counters():
    """Per-test isolation for in-process reliability state (pass 4 #28).

    The circuit-breaker registry and the supervisor crash-loop budget are
    module globals; without this reset a test that fails a backend URL five
    times would open a breaker that poisons later tests reusing that key."""
    from sable.core import circuit
    from sable.serving import supervisor

    def _clear():
        with circuit._registry_lock:
            circuit._registry.clear()
        with supervisor._crash_lock:
            supervisor._crash_failures.clear()

    _clear()
    yield
    _clear()

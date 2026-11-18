"""Experience buffer (Phase 2): durable, outcome-scored trace records.

Layout (root = TOASTER_LOOP_HOME env > config `home:` > ~/.toaster-loop):

    <home>/
      buffer/
        toaster_loop.db                 # SQLite index: traces + feedback
        YYYY/MM/<trace_id>.jsonl   # append-only full conversation per trace
Full conversation content lives only in the JSONL files; SQLite holds index /
metadata so Phase 3 curation can query without touching blobs. Every JSONL
record carries `schema_version` for future migrations.
"""

from .paths import resolve_toaster_loop_home
from .db import (SCHEMA_VERSION, SchemaTooNewError, TraceStore, TraceRow,
                 STATUS_VALUES)
from .recorder import TraceRecorder, heuristic_score
from .insights import generate_insights

__all__ = [
    "resolve_toaster_loop_home",
    "SCHEMA_VERSION",
    "SchemaTooNewError",
    "TraceStore",
    "TraceRow",
    "STATUS_VALUES",
    "TraceRecorder",
    "heuristic_score",
    "generate_insights",
]

"""Tool-result spill: persist oversized outputs instead of dropping them.

Backlog item #10. Patterns built in
``tools/tool_result_storage.py`` (attribution): the three-level defense model
(per-tool cap -> per-result spill -> per-turn aggregate budget) reduced to
levels 2-3 — sable has no sandbox, so there is no path translation or
remote-write fallback; everything lives host-side under SABLE_HOME.

Levels:
  * ``maybe_spill_result`` (level 2): after a tool returns, output longer
    than ``tools.spill_threshold_chars`` is written in full to
    ``<SABLE_HOME>/cache/spillover/<tool_call_id>.txt`` and the in-context
    content becomes a small ``<persisted-output path=… bytes=N>`` stub with a
    preview. The agent can re-read the file with ``read_file`` (the spillover
    dir sits under SABLE_HOME, which workspace containment already
    allow-lists — see core/agent_runner.resolve_workspace_roots).
  * ``enforce_turn_budget`` (level 3): when the aggregate tool output of one
    assistant turn exceeds ``tools.turn_result_budget_chars``, the largest
    results are spilled first until the total fits. Already-stubbed content
    is skipped, so both entry points are idempotent.

Durability / secrets: spill files live under SABLE_HOME for provenance and
are written through ``core/redact.redact_text`` first — like the trace JSONL,
they are durable artifacts, so secrets must not land on disk unredacted (the
in-memory conversation stays raw; only durable copies are scrubbed). The
stub's ``bytes=N`` reports the ORIGINAL content size (pre-redaction); the
preview in the stub is raw because it stays in memory and is redacted again
by the recorder at record time.

Retention: files older than ``tools.spill_retention_days`` are pruned by a
best-effort sweep that runs at most once per process (first spill), mirroring
``_prune_spillover_once``; prune never raises on permission errors.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .redact import redact_text

logger = logging.getLogger("sable.spill")

#: Prefix of a spill stub; doubles as the idempotency guard (never re-spill).
PERSISTED_OUTPUT_PREFIX = "<persisted-output"
_SPILL_CLOSING_TAG = "</persisted-output>"
SPILLOVER_SUBDIR = "cache/spillover"

# Raised from 8000/24000 (2025-07): the 8k threshold spilled ordinary
# file reads into the cache, which forced agents into small-chunk reading
# dances and made the spillover stubs look like failures to anyone
# eyeballing them. 48k covers normal single-file reads whole; the turn
# budget must sit well above that so a few large results in one turn
# don't immediately re-spill.
DEFAULT_SPILL_THRESHOLD_CHARS = 48_000
DEFAULT_TURN_RESULT_BUDGET_CHARS = 120_000
DEFAULT_SPILL_RETENTION_DAYS = 7
_PREVIEW_CHARS = 500

# filename safety for tool_call_id (unsafe-filename sanitizer)
_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")
_MAX_FILENAME_STEM = 120
_PATH_IN_STUB_RE = re.compile(r"<persisted-output path=(\S+) bytes=")

_sweep_lock = threading.Lock()
_swept_once = False


def reset_sweep_flag_for_tests() -> None:
    """Re-arm the once-per-process prune sweep (tests only)."""
    global _swept_once
    with _sweep_lock:
        _swept_once = False


def spill_dir(home: Optional[Union[str, Path]] = None) -> Path:
    """Return ``<SABLE_HOME>/cache/spillover`` (not created here)."""
    if home is None:
        from ..store.paths import resolve_sable_home  # lazy: avoid cycles

        home = resolve_sable_home()
    return Path(home).expanduser() / SPILLOVER_SUBDIR


def prune_spillover(home: Optional[Union[str, Path]] = None,
                    retention_days: int = DEFAULT_SPILL_RETENTION_DAYS) -> int:
    """Delete spillover files older than *retention_days*. Best-effort."""
    cutoff = time.time() - retention_days * 86400
    removed = 0
    try:
        entries = list(spill_dir(home).iterdir())
    except OSError:
        return 0
    for f in entries:
        try:
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink()
                removed += 1
        except OSError:
            continue
    if removed:
        logger.info("pruned %d expired spillover file(s)", removed)
    return removed


def _sweep_once(home: Optional[Union[str, Path]],
                retention_days: int = DEFAULT_SPILL_RETENTION_DAYS) -> None:
    """Best-effort age-based prune, at most once per process)."""
    global _swept_once
    with _sweep_lock:
        if _swept_once:
            return
        _swept_once = True
    try:
        prune_spillover(home, retention_days)
    except Exception as exc:  # pruning must never break a tool call
        logger.debug("spillover prune failed: %s", exc)


def _safe_filename(tool_call_id: str) -> str:
    """Sanitize a tool_call_id into one safe ``<stem>.txt`` filename."""
    raw = str(tool_call_id or "tool_result")
    stem = _UNSAFE_FILENAME_CHARS.sub("_", raw).strip("._-")
    if not stem:
        stem = "tool_result"
    if stem != raw or len(stem) > _MAX_FILENAME_STEM:
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
        stem = f"{stem[:_MAX_FILENAME_STEM].rstrip('._-') or 'tool_result'}_{digest}"
    return f"{stem}.txt"


def _build_stub(content: str, path: str,
                gate_unit: Optional[str] = None) -> str:
    """Build the <persisted-output …> replacement stub.

    ``gate_unit="tokens"`` adds a cheap provenance marker recording that the
    token-accurate gate (not the char gate) fired; char-gate stubs stay
    byte-identical to every previous version.
    """
    nbytes = len(content.encode("utf-8"))
    gate_attr = f" gate={gate_unit}" if gate_unit else ""
    preview = content[:_PREVIEW_CHARS]
    return (f"{PERSISTED_OUTPUT_PREFIX} path={path} bytes={nbytes}{gate_attr}>\n"
            f"{preview}\n"
            "…[full output persisted; read_file the path to retrieve]\n"
            f"{_SPILL_CLOSING_TAG}")


def _write_spill(content: str, filename: str,
                 home: Optional[Union[str, Path]]) -> Optional[str]:
    """Atomically write *content* (redacted) into the spillover dir.

    Returns the absolute path string, or None on failure (caller then keeps
    the content inline — losing spill is acceptable, losing data is not).
    """
    try:
        d = spill_dir(home)
        d.mkdir(parents=True, exist_ok=True)
        path = d / filename
        fd, tmp = tempfile.mkstemp(dir=str(d), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8", errors="replace") as fh:
                # Secrets must not land on disk unredacted (spill files are
                # durable like the trace — see module docstring).
                fh.write(redact_text(content))
            os.replace(tmp, path)  # atomic within the same dir
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except OSError as exc:
        logger.warning("spillover write failed for %s: %s", filename, exc)
        return None
    _sweep_once(home)
    return str(path)


def is_spillover_read(tool_name: str, args: Dict[str, Any],
                      home: Optional[Union[str, Path]] = None) -> bool:
    """True when this call re-reads an already-persisted spill file.

    Re-spilling those results creates a doom loop: the agent follows the
    stub's "read_file the path" instruction, the read comes back over
    threshold, gets spilled again as a *new* stub, and the model can never
    actually retrieve its content. Reads aimed at the spillover dir are
    exempt from both spill levels."""
    if tool_name != "read_file":
        return False
    path = str((args or {}).get("path") or "").strip()
    if not path:
        return False
    try:
        resolved = Path(path).expanduser().resolve()
        return resolved.is_file() and resolved.parent == spill_dir(home).resolve()
    except OSError:
        return False


def maybe_spill_result(tool_name: str, tool_call_id: str, content: str,
                       home: Optional[Union[str, Path]] = None, *,
                       threshold: int = DEFAULT_SPILL_THRESHOLD_CHARS,
                       retention_days: int = DEFAULT_SPILL_RETENTION_DAYS,
                       threshold_tokens: Optional[int] = None,
                       counter: Optional[Any] = None,
                       ) -> Optional[str]:
    """Level 2 (backlog #10): spill one oversized tool result to disk.

    Returns a stub string when the content was persisted, or None when no
    spill was needed / was impossible (caller keeps the original content).
    Never raises; idempotent — content that is already a stub is skipped.

    Token-accurate gate (deferred item from improvement #4): when
    ``threshold_tokens`` (> 0) and a duck-typed ``counter`` (a
    core/tokenizer.TokenCounter exposing ``count(text) -> int``, plus the
    optional ``exact_available`` flag) are both supplied, the gate compares
    the exact token count against ``threshold_tokens`` instead of the char
    comparison — chars/4 badly misestimates CJK/code-heavy tool output.
    Precedence: explicit tokens gate wins; when either piece is missing, or
    the counter reports itself degraded (circuit open), or counting raises,
    the decision falls back to EXACTLY the old ``len(content) > threshold``
    char comparison — exactness degrades gracefully, never correctness. One
    count per call, at decision time only. Stubs spilled through the token
    gate carry a ``gate=tokens`` provenance marker; char-gate stubs stay
    byte-identical to before.
    """
    if not isinstance(content, str):
        return None
    if content.startswith(PERSISTED_OUTPUT_PREFIX):  # already a stub
        return None
    measured, gate, unit = len(content), threshold, "chars"
    if (threshold_tokens is not None and int(threshold_tokens) > 0
            and counter is not None
            and getattr(counter, "exact_available", True)):
        try:
            measured = int(counter.count(content))
            gate, unit = int(threshold_tokens), "tokens"
        except Exception:  # defensive: a broken counter never breaks a run
            logger.debug("spill token count failed; using chars gate",
                         exc_info=True)
            measured, gate, unit = len(content), threshold, "chars"
    if measured <= gate:
        return None
    path = _write_spill(content, _safe_filename(tool_call_id), home)
    if path is None:
        return None
    logger.info("spilled tool result %s (%s, %d %s > %d -> %s)",
                tool_name, tool_call_id, measured, unit, gate, path)
    return _build_stub(content, path,
                       gate_unit="tokens" if unit == "tokens" else None)


def enforce_turn_budget(messages_for_turn: List[Dict[str, Any]],
                        budget_chars: int,
                        home: Optional[Union[str, Path]] = None, *,
                        retention_days: int = DEFAULT_SPILL_RETENTION_DAYS,
                        exempt: Optional[set] = None,
                        ) -> List[Dict[str, Any]]:
    """Level 3 (backlog #10): cap the aggregate tool output of one turn.

    ``messages_for_turn`` holds the tool-result message dicts produced by a
    single assistant turn. When their combined content exceeds
    *budget_chars*, the largest results are spilled first (threshold 0) until
    the total fits. Mutates the message dicts in place and returns the list;
    stub-wrapped content is skipped, so re-running is a no-op. ``exempt`` is a
    set of indices never spilled (spillover re-reads — see is_spillover_read).
    """
    if budget_chars <= 0:
        return messages_for_turn
    total = 0
    exempt = exempt or set()
    candidates: List[tuple] = []  # (size, index) descending
    for i, msg in enumerate(messages_for_turn):
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, str):
            continue
        total += len(content)
        if i not in exempt and not content.startswith(PERSISTED_OUTPUT_PREFIX):
            candidates.append((len(content), i))
    if total <= budget_chars:
        return messages_for_turn
    candidates.sort(reverse=True)
    for size, i in candidates:
        if total <= budget_chars:
            break
        msg = messages_for_turn[i]
        stub = maybe_spill_result(
            msg.get("name") or "tool", str(msg.get("tool_call_id") or f"budget_{i}"),
            msg["content"], home, threshold=0, retention_days=retention_days)
        if stub is not None:
            total += len(stub) - size
            msg["content"] = stub
            logger.info("turn budget enforced: spilled tool result %s (%d chars)",
                        msg.get("tool_call_id", i), size)
    return messages_for_turn

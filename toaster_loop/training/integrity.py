"""Post-run artifact integrity checks (WP2b).

The failure mode this guards against: a training/export flow whose process
exits 0 while the *artifact* it produced is silently corrupt — a truncated or
zero-byte GGUF, a file with no ``GGUF`` magic, or a loss history containing
NaN. Those outcomes must be reported with a DISTINCT exit code
(``EXIT_CORRUPT_ARTIFACT = 4``, see core/exitcodes.py) instead of blending
into "the run errored" (1): automation needs to tell "the pipeline failed
loudly" from "the pipeline finished and quietly produced garbage"
(kimi-k3-in-c's corrupt-output exit-code idea; re-implemented inline, no
imports per D-12).

Checks are deliberately cheap and side-effect free: stat + first bytes of the
file, plus an isnan scan over already-parsed loss floats. Deep GGUF graph
validation belongs to llama.cpp/validate, not here.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional

#: GGUF container magic (file starts with these four bytes; version follows).
GGUF_MAGIC = b"GGUF"


def check_gguf_file(path: Any) -> Optional[str]:
    """Return None when ``path`` looks like a plausible GGUF, else the reason.

    Honest-minimal gate: exists, non-zero size, starts with the GGUF magic.
    Never raises — an unreadable file is itself a corruption report."""
    p = Path(str(path))
    if not p.is_file():
        return f"artifact missing: {p}"
    try:
        size = p.stat().st_size
    except OSError as e:
        return f"artifact stat failed for {p}: {e}"
    if size == 0:
        return f"zero-byte artifact: {p}"
    try:
        head = p.open("rb").read(4)
    except OSError as e:
        return f"artifact unreadable ({p}): {e}"
    if not head.startswith(GGUF_MAGIC):
        return (f"bad GGUF magic in {p}: expected 'GGUF', found "
                f"{head!r} (truncated or wrong-format file)")
    return None


def find_nan_losses(losses: Dict[str, Any]) -> List[str]:
    """Names of loss entries that are NaN (silently-corrupt numerics).

    Accepts the parse_worker_losses shape ({train_loss, eval_loss}) or any
    dict of optional floats. None entries are honest-absence, not corruption;
    non-numeric entries are ignored."""
    bad: List[str] = []
    for k, v in (losses or {}).items():
        if isinstance(v, float) and math.isnan(v):
            bad.append(str(k))
    return sorted(bad)


def export_integrity_problems(gguf_path: Any,
                              losses: Optional[Dict[str, Any]] = None,
                              ) -> List[str]:
    """All integrity problems of one finished export, as human-readable
    strings ([] when the artifact is sound)."""
    problems: List[str] = []
    reason = check_gguf_file(gguf_path) if gguf_path else "no gguf path"
    if reason:
        problems.append(reason)
    for name in find_nan_losses(losses or {}):
        problems.append(f"NaN recorded in loss history ({name})")
    return problems

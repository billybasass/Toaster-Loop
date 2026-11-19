"""ML-stack version window — "one TRL minor bump must not break DPO silently".

Why this exists: the training worker carries API-drift scar tissue (SFTTrainer
moved transformers->trl, `optimizer_cls_and_kwargs` churn, DPO dataset schema
changes). The [training] extra in pyproject.toml declares bounded ranges, but
nothing ever verified what is ACTUALLY loaded at runtime: a hand-run
``pip install -U trl`` inside the venv silently walks off the tested window and
DPO/GRPO break mid-run (or worse, train subtly wrong).

This module is the single source of truth for the supported window, mirroring
the pyproject ``[training]`` bounds. It is stdlib-only (reads version METADATA
via importlib.metadata — never imports torch; see tests/test_hygiene.py) so it
can run at worker startup AND inside ``doctor``'s deps check with zero cost.

Enforcement points:

  * worker.main() calls :func:`worker_gate` before any training kind runs:
    out-of-window => hard stop with an actionable message (the job log gets a
    JSON error event like every other failure). Escape hatch:
    ``TOASTER_LOOP_ALLOW_UNPINNED_MLSTACK=1`` downgrades the stop to a logged
    warning, for users deliberately running ahead of the tested window.
  * doctor's deps check calls :func:`check_function` and reports each package's
    verdict line.

Keep SUPPORTED in sync with pyproject.toml [training] (a test asserts the two
bounds agree so they cannot drift).
"""

from __future__ import annotations

import os
import re
from typing import Callable, Dict, List, Optional, Tuple

#: Supported version window per ML-stack package: (min_inclusive, max_exclusive)
#: as "major.minor" strings. MUST mirror pyproject.toml [training] — enforced by
#: tests/test_prod_readiness.py::test_mlstack_window_matches_pyproject.
SUPPORTED: Dict[str, Tuple[str, str]] = {
    "torch": ("2.4", "3"),
    "transformers": ("4.45", "6"),
    "peft": ("0.13", "1"),
    "trl": ("0.12", "2"),
    "datasets": ("3.0", "6"),
    "accelerate": ("0.34", "2"),
}

#: Env var that downgrades the worker's hard stop to a warning (users who
#: deliberately run outside the tested window accept the drift risk).
ALLOW_OVERRIDE_ENV = "TOASTER_LOOP_ALLOW_UNPINNED_MLSTACK"


def parse_version(v: str) -> Optional[Tuple[int, int, int]]:
    """Leading numeric triple of a PEP 440 version, tolerating local tags and
    pre-release suffixes: '2.13.0+rocm7.2' -> (2, 13, 0); '5.0rc1' -> (5, 0, 0).
    Returns None for garbage (caller treats unknown as 'skip', never 'fail')."""
    m = re.match(r"^\s*v?(\d+)(?:\.(\d+))?(?:\.(\d+))?", str(v or ""))
    if not m:
        return None
    return tuple(int(g) if g else 0 for g in m.groups())  # type: ignore[return-value]


def _bound(s: str) -> Tuple[int, int, int]:
    nums = [int(g) for g in re.findall(r"\d+", str(s))[:3]]
    nums = (nums + [0, 0, 0])[:3]
    return (nums[0], nums[1], nums[2])


def window_problems(versions: Dict[str, str]) -> List[str]:
    """Pure core: given {package: loaded_version}, return actionable problem
    lines for every package outside its SUPPORTED window. Packages absent from
    ``versions`` are NOT problems here (an uninstalled extra is a different
    check — doctor reports presence separately). Unparseable versions are
    reported as such rather than guessed."""
    problems: List[str] = []
    for pkg, (lo_s, hi_s) in SUPPORTED.items():
        got = versions.get(pkg)
        if got is None or str(got).strip() == "":
            continue
        v = parse_version(got)
        lo, hi = _bound(lo_s), _bound(hi_s)
        if v is None:
            problems.append(
                f"{pkg}: version {got!r} is unparseable — cannot verify it "
                f"lies in the supported window >={lo_s},<{hi_s}")
            continue
        if not (lo <= v < hi):
            problems.append(
                f"{pkg} {got} is outside the supported window "
                f">={lo_s},<{hi_s} (toaster_loop's training stack is tested "
                "against this range; API drift outside it has broken "
                "SFT/DPO/GRPO before)")
    return problems


def installed_versions(version_fn: Optional[Callable[[str], str]] = None
                       ) -> Dict[str, str]:
    """{package: installed version} for SUPPORTED packages, read from wheel
    METADATA only (torch is never imported). Missing dists are omitted."""
    if version_fn is None:
        from importlib.metadata import version as _dist_version

        version_fn = _dist_version
    out: Dict[str, str] = {}
    for pkg in SUPPORTED:
        try:
            out[pkg] = str(version_fn(pkg))
        except Exception:
            continue
    return out


def check_function() -> Tuple[bool, List[str]]:
    """Doctor probe: (ok, lines). One line per installed ML-stack package with
    its window verdict; ok=False when any installed package is out of window."""
    vers = installed_versions()
    if not vers:
        return True, ["ml-stack window: no [training] extras installed "
                      "(nothing to check)"]
    problems = window_problems(vers)
    lines = [f"ml-stack window: {len(SUPPORTED) - len(problems)}"
             f"/{len(vers)} installed packages in range"]
    for pkg, v in sorted(vers.items()):
        lo, hi = SUPPORTED[pkg]
        mark = "in-range" if not any(pkg in p for p in problems) else "OUT"
        lines.append(f"  {pkg} {v}: {mark} (>= {lo}, < {hi})")
    lines.extend(f"  PROBLEM: {p}" for p in problems)
    return (not problems), lines


def allow_override() -> bool:
    val = os.environ.get(ALLOW_OVERRIDE_ENV, "").strip().lower()
    return val not in ("", "0", "false", "no")


def worker_gate() -> Optional[str]:
    """Worker startup gate. Returns None when the loaded ML stack is inside
    the supported window (or the override env var silences the gate), else a
    full actionable error message for the caller to raise/emit."""
    problems = window_problems(installed_versions())
    if not problems:
        return None
    if allow_override():
        return None
    fixes = "; ".join(
        f"'{p}>={lo},<{hi}'" for p, (lo, hi) in SUPPORTED.items()
        if any(p in prob for prob in problems))
    return (
        "ML-stack version window violated — refusing to train with an "
        "untested library combination:\n  - " + "\n  - ".join(problems) +
        f"\nFix: pip install {fixes}  (or re-run ./setup.sh), or set "
        f"{ALLOW_OVERRIDE_ENV}=1 to train anyway at your own risk.")

"""`datasets` package preflight + install into the TRAINING interpreter.

Why this exists: the `datasets` package is only pulled by the optional
[training] extra (pyproject), and on Windows setup.ps1 may skip that extra
entirely (--no-training / bitsandbytes build failures). The training worker
then dies at `from datasets import load_dataset` with an ImportError whose fix
is not obvious. This module makes the gap detectable (doctor + a dashboard
preflight endpoint) and fixable in one click: it installs `datasets` — WITHOUT
bitsandbytes, which is what breaks on Windows — into exactly the interpreter
the scheduler spawns workers with (training.python, else this process's).

Pattern follows serving/engine_installs.py (explicit user action; subprocess
pip; never runs on import) but targets an EXISTING interpreter rather than a
private venv: dataset loading must land where worker.py imports it.
"""

from __future__ import annotations

import logging
import os.path
import subprocess
import sys
from typing import Any, Callable, List, Optional

logger = logging.getLogger("toaster_loop.core.datasetenv")

#: Minimal package set for loading datasets in the worker (worker.py imports
#: `datasets`; its peers pyarrow/huggingface-hub arrive as dependencies — we
#: pin only the direct need so pip's resolver stays free). NOT bitsandbytes:
#: that is a quantization-training dep and the Windows build-breaker; keeping
#: it out of this list is what makes datasets installable without [training].
DATASETS_PACKAGES = ["datasets"]

#: Import probe run inside the target interpreter. Cheap (-c), no torch.
PROBE_CODE = ("import importlib.util, sys; "
              "sys.exit(0 if importlib.util.find_spec('datasets') else 1)")


class DatasetEnvError(RuntimeError):
    """Actionable datasets-install failure (bad interpreter, pip error)."""


def training_python(cfg: Any) -> str:
    """Interpreter the scheduler spawns training workers with.

    Same resolution order as training/scheduler.py: an explicit
    `training.python` config wins; otherwise this process's sys.executable.
    Accepts a ToasterLoopConfig, a TrainingConfig (has .python directly), or None."""
    if cfg is None:
        return sys.executable
    # Scheduler-style config carries .python directly.
    py = str(getattr(cfg, "python", "") or "").strip()
    if py:
        return py
    training = getattr(cfg, "training", None)
    if isinstance(training, dict):
        py = str(training.get("python") or "").strip()
    elif training is not None:
        py = str(getattr(training, "python", "") or "").strip()
    return py or sys.executable


def datasets_importable(python: Optional[str] = None, *,
                        run_fn: Optional[Callable[..., Any]] = None,
                        timeout: float = 60.0) -> bool:
    """True when `datasets` resolves inside the training interpreter.

    Uses find_spec (no actual import): fast and side-effect free. run_fn is a
    subprocess.run seam for tests."""
    import shutil as _shutil

    py = python or sys.executable
    if not os.path.isabs(py) and _shutil.which(py) is None:
        return False
    run = run_fn or subprocess.run
    try:
        r = run([py, "-c", PROBE_CODE], capture_output=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return False
    return getattr(r, "returncode", 1) == 0


def pip_command(python: str, packages: Optional[List[str]] = None
                ) -> List[str]:
    """The exact command install_datasets runs (also the doctor's next step —
    one string the user can paste; -U keeps an existing broken install fixable)."""
    return [python, "-m", "pip", "install", "-U",
           *(packages or DATASETS_PACKAGES)]


def install_datasets(python: Optional[str] = None, *,
                     runner: Optional[Callable[..., Any]] = None,
                     timeout: float = 1800.0) -> dict:
    """pip-install `datasets` into the training interpreter (explicit action).

    Deliberately WITHOUT bitsandbytes / torch: those come from [training] or
    the torch flow; datasets alone unblocks loading (worker.py gates torch at
    its own import site). Returns {ok, command, tail}; raises DatasetEnvError
    on a non-zero pip exit with the last log lines attached."""
    py = python or training_python(None)
    cmd = pip_command(py)
    run = runner or subprocess.run
    logger.info("installing datasets into training env: %s", " ".join(cmd))
    try:
        r = run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise DatasetEnvError(
            f"pip install timed out after {timeout:.0f}s: {' '.join(cmd)}") from e
    except OSError as e:
        raise DatasetEnvError(f"cannot run pip with {py}: {e}") from e
    rc = getattr(r, "returncode", 1)
    out = str(getattr(r, "stdout", "") or "") + \
        str(getattr(r, "stderr", "") or "")
    tail = [ln for ln in out.splitlines() if ln.strip()][-12:]
    if rc != 0:
        detail = ": " + " | ".join(tail[-4:]) if tail else ""
        raise DatasetEnvError(f"pip install datasets failed (exit {rc})"
                              + detail)
    return {"ok": True, "command": cmd, "tail": tail}


def preflight(cfg: Any) -> dict:
    """Status payload for the dashboard/datasets page: where we looked, what
    it found, and the exact next command when missing."""
    py = training_python(cfg)
    ok = datasets_importable(py)
    return {"python": py, "datasets": ok,
            "install_command": None if ok else " ".join(pip_command(py)),
            "note": ("datasets imports in the training interpreter" if ok else
                     "the `datasets` package is missing from the training "
                     "interpreter — dataset-backed curation/training will "
                     "fail until it is installed (one click, or run the "
                     "command shown)")}

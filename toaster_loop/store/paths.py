"""Resolve the toaster_loop state root.

Precedence (consistent with the existing TOASTER_LOOP_* env pattern in
core/config.py): ``TOASTER_LOOP_HOME`` env var > config key ``home:`` > ~/.toaster-loop.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..core.config import ToasterLoopConfig

ENV_HOME = "TOASTER_LOOP_HOME"
DEFAULT_DIRNAME = ".toaster-loop"
_LEGACY_DIRNAME = ".sable"  # pre-rename state dir (kept working on upgrade)


def resolve_toaster_loop_home(cfg: Optional["ToasterLoopConfig"] = None) -> Path:
    """Return the toaster_loop state root (not created here)."""
    env = os.environ.get(ENV_HOME, "").strip()
    if env:
        return Path(env).expanduser()
    home = getattr(cfg, "home", None) if cfg is not None else None
    if home:
        return Path(str(home)).expanduser()
    default = Path.home() / DEFAULT_DIRNAME
    legacy = Path.home() / _LEGACY_DIRNAME
    # Upgrade path: users who ran the pre-rename build keep their existing
    # state in ~/.sable until they explicitly opt into the new location.
    if not default.exists() and legacy.is_dir():
        return legacy
    return default

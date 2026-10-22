"""Resolve the sable state root.

Precedence (consistent with the existing SABLE_* env pattern in
core/config.py): ``SABLE_HOME`` env var > config key ``home:`` > ~/.sable.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..core.config import SableConfig

ENV_HOME = "SABLE_HOME"
DEFAULT_DIRNAME = ".sable"


def resolve_sable_home(cfg: Optional["SableConfig"] = None) -> Path:
    """Return the sable state root (not created here)."""
    env = os.environ.get(ENV_HOME, "").strip()
    if env:
        return Path(env).expanduser()
    home = getattr(cfg, "home", None) if cfg is not None else None
    if home:
        return Path(str(home)).expanduser()
    return Path.home() / DEFAULT_DIRNAME

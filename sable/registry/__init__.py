"""Model registry (Phase 4): lineage, aliases, promotion/rollback.

Backlog item #16 adds the archive-don't-delete lifecycle: archive_model /
restore_model / set_pinned on ModelRegistry plus the disk janitor
(janitor.run_janitor) that auto-archives only aged, unpinned, terminal models.
"""

from .registry import (
    ARCHIVABLE_STATUSES,
    JOB_KINDS,
    JOB_STATES,
    MODEL_STATUSES,
    REGISTRY_SCHEMA_VERSION,
    JobRow,
    ModelRegistry,
    ModelRow,
    RegistryError,
    days_since_last_use,
)
from .janitor import janitor_config_from_yaml, run_janitor
from .ledger import GENESIS_HASH, recent_ops, record, verify_ledger

__all__ = [
    "ARCHIVABLE_STATUSES",
    "GENESIS_HASH",
    "JOB_KINDS",
    "JOB_STATES",
    "MODEL_STATUSES",
    "REGISTRY_SCHEMA_VERSION",
    "JobRow",
    "ModelRegistry",
    "ModelRow",
    "RegistryError",
    "days_since_last_use",
    "janitor_config_from_yaml",
    "recent_ops",
    "record",
    "run_janitor",
    "verify_ledger",
]

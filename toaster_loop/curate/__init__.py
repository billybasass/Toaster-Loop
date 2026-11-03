"""Dataset curation (Phase 3): experience buffer -> SFT/DPO exports.

Public surface:
    Curater            — orchestration (selection, dedup, export, claiming)
    CurationConfig     — all knobs (config `curation:` section + CLI flags)
    CurationResult     — what one curate run produced
    messages_to_sharegpt — OpenAI messages -> ShareGPT turns (trajectory-compatible)
    judge_trace        — LLM-as-judge scoring of one trace (backlog #12)
    import_dataset     — external dataset import w/ format detection (#14)
"""

from .curater import (
    Curater,
    CurationConfig,
    CurationResult,
    DPOPair,
    curation_config_from_yaml,
    dedup_traces,
    mine_dpo_pairs,
    render_train_yaml,
)
from .importer import ImportResult, detect_format, import_dataset
from .judge import judge_trace
from .sharegpt import final_assistant_text, messages_to_sharegpt

__all__ = [
    "Curater",
    "CurationConfig",
    "CurationResult",
    "DPOPair",
    "ImportResult",
    "curation_config_from_yaml",
    "dedup_traces",
    "detect_format",
    "import_dataset",
    "judge_trace",
    "mine_dpo_pairs",
    "render_train_yaml",
    "final_assistant_text",
    "messages_to_sharegpt",
]

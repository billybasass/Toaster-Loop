"""Dataset curater: buffered traces -> training-ready datasets on disk.

Reads the Phase 2 experience buffer (SQLite index + JSONL transcripts) and
writes a dataset directory under ``<home>/datasets/<dataset_id>/``:

    sft.jsonl        ShareGPT rows ({"conversations": [...], ...}) — one per
                     deduped, quality-gated trace. Consumable by trainers with
                     format_type=sharegpt.
    dpo.jsonl        TRL PreferenceDataset-style rows:
                     {"prompt": [{"role","content"},...],   # messages up to
                                                                (excl.) the
                                                                final answer
                      "chosen": "<text>",                    # best final answer
                      "rejected": "<text>",                  # worse/failed one
                      "metadata": {...}}                     # provenance
    manifest.json    dataset_id, created_at, per-file source trace_ids +
                     counts, filter params, schema versions, format hints.
    train.yaml       TEMPLATE for the ``train --config`` contract
                     (model/data/training/lora/logging; a documented file
                     format, not an import). Phase 4 executes it.

Selection / claiming semantics (Phase 3 spec):
    * candidates = unclaimed traces matching status/time/profile filters;
      a trace passes if effective score >= min_score OR it has explicit
      positive user feedback. ``failure`` traces are excluded unless
      include_failures (they remain eligible as DPO negatives only).
    * selected traces are claimed with curated=1, dataset_id=<id>; a trace
      appears in at most one dataset version. Re-running without --rebuild
      skips claimed traces; --rebuild <dataset_id> releases that dataset's
      claims first and re-curates from scratch.
    * finding #8: claiming happens BEFORE any dataset file is written, via
      the atomic store.claim_traces_owned primitive (guarded UPDATE +
      rowcount accounting in one BEGIN IMMEDIATE transaction), so two
      concurrent curates can never both list the same trace. Only
      successfully-claimed IDs reach sft.jsonl/dpo.jsonl/manifest.json; a
      short claim is logged as a WARNING and reported in CurationResult.
      Any write failure after claiming releases this dataset's claims so the
      traces become curatable again.

Effective score: user feedback overrides the heuristic row (Phase 2 keeps
both); NULL scores stay None — never coerced to 0 (Phase 2 note #1 applies
to scores too). Backlog item #12 adds an optional LLM-judge layer: precedence
is user > judge > heuristic, and with no judge rows present the selection is
byte-identical to before. A lightweight verifier pass re-scans tool messages
for error-shaped content that the record-time heuristic may have missed
(Phase 2 note #2).
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..core.failure_types import classify_tool_result, is_infra_reason
from ..store import analytics
from ..store.db import TraceRow, TraceStore
from .sharegpt import final_assistant_text, messages_to_sharegpt

logger = logging.getLogger("toaster_loop.curate")

DATASET_SCHEMA_VERSION = 1
# v2: default training output_dir moved from <dataset>/outputs to a local
# repo-style folder inside the HF hub cache (see adapter_output_dir).
TRAIN_YAML_TEMPLATE_VERSION = 2


def adapter_output_dir(dataset_id: str) -> Optional[str]:
    """Default adapter output for a dataset: a *local* folder inside the
    Hugging Face hub cache, in hub layout so standard tooling sees it as a
    repo directory:

        <hub>/models--toaster-loop--<dataset-id>/snapshots/<UTC-stamp>/

    The hub root is resolved like everything else (core/model_discovery:
    TOASTER_LOOP_MODELS_DIR > $HF_HOME/hub > ~/.cache/huggingface/hub). When
    no hub cache exists (or the path can't be built) returns None and callers
    keep the legacy <dataset>/outputs default — training never breaks because
    of a cache location. Datasets stay self-contained (train.yaml + jsonl);
    only the bulky adapter/checkpoints move out to the model cache where the
    big files belong.
    """
    import re

    from ..core.model_discovery import resolve_models_dir

    base = resolve_models_dir()
    if base is None:
        return None
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(dataset_id or "").strip())
    slug = slug.strip("-._") or "dataset"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    try:
        return str(base / f"models--toaster-loop--{slug}"
                   / "snapshots" / stamp)
    except (OSError, ValueError):
        return None


# Idea adapted from PIASystem (MIT) app/core/failure_taxonomy.py — the
# remediation lifecycle (#33) feeding curation weighting. Narrow rule, and
# RESOLVED is the ONLY status that matters here (acknowledged/in_progress are
# display-only): a trace whose failure_reason carries a RESOLVED note stamped
# *after* the trace was recorded failed because of a bug that has since been
# fixed, so it must not be mined as "bad data" DPO negative material — same
# treatment infra faults already get. Profile-scoped notes apply only to
# matching agent_profile rows; global (profile NULL) notes apply everywhere.
def resolved_note_covers(row: TraceRow,
                         notes: List[Dict[str, Any]]) -> bool:
    """True when a RESOLVED failure note is newer than this trace's timestamp."""
    if not row.failure_reason:
        return False
    note = None
    for n in notes:  # profile-scoped match first, then global (profile None)
        if n.get("reason") != row.failure_reason:
            continue
        if n.get("profile") == row.agent_profile:
            note = n
            break
        if n.get("profile") is None and note is None:
            note = n
    if note is None or note.get("status") != "resolved":
        return False
    try:
        note_ts = datetime.fromisoformat(str(note["ts"]))
        trace_ts = datetime.fromisoformat(row.timestamp)
    except (KeyError, TypeError, ValueError):
        return False  # unparseable stamps never change weighting (conservative)
    if note_ts.tzinfo is None:
        note_ts = note_ts.replace(tzinfo=timezone.utc)
    if trace_ts.tzinfo is None:
        trace_ts = trace_ts.replace(tzinfo=timezone.utc)
    return note_ts > trace_ts


# --------------------------------------------------------------------- config

@dataclass
class CurationConfig:
    """All knobs; defaults per Phase 3 spec. YAML `curation:` section maps here."""

    min_score: float = 0.5
    include_failures: bool = False       # failures only ever serve as DPO negatives
    # Backlog #12 (LLM-as-judge): knobs only — judging itself is purely
    # flag-driven (`curate --judge N`); no config key turns it on.
    judge_band: float = 0.15             # |score - min_score| < band = borderline
    judge_model: Optional[str] = None    # default: the served model alias
    since: Optional[str] = None          # ISO-8601 lower bound on trace timestamp
    model_alias: Optional[str] = None    # traces' model_alias filter
    agent_profile: Optional[str] = None  # backend name, e.g. 'toaster_loop'
    project: Optional[str] = None        # projects: trace project-id filter
    limit: Optional[int] = None          # cap rows per output file
    formats: str = "both"                # sft | dpo | both

    # Quality gates
    min_turns: int = 2                   # ShareGPT turns after conversion
    max_turns: int = 200
    max_est_tokens: int = 16384          # chars/4 heuristic (documented, rough)
    drop_tool_error_only: bool = True    # every tool result looks like an error

    # train.yaml template inputs (Phase 4 consumes; toaster_loop just fills them in)
    base_model: Optional[str] = None     # GGUF path or HF id for training
    output_dir: Optional[str] = None     # adapter output dir for the trainer
    gguf_quant: str = "q4_k_m"           # Phase 4 GGUF export parameter hint


def curation_config_from_yaml(data: Dict[str, Any]) -> CurationConfig:
    """Build from the optional `curation:` mapping in toaster_loop's config file."""
    sec = data.get("curation") if isinstance(data, dict) else None
    if not isinstance(sec, dict):
        return CurationConfig()
    known = {f for f in CurationConfig.__dataclass_fields__}
    unknown = sorted(k for k in sec if k not in known)
    if unknown:  # typo'd knobs used to vanish silently (same as training:)
        logger.warning("curation config: ignoring unknown key(s) %s "
                       "(known: %s)", ", ".join(unknown),
                       ", ".join(sorted(known)))
    kwargs = {k: v for k, v in sec.items() if k in known and v is not None}
    return CurationConfig(**kwargs)


# ------------------------------------------------------------------ transcript

@dataclass
class LoadedTrace:
    row: TraceRow
    messages: List[Dict[str, Any]]       # verbatim OpenAI-format messages
    effective_score: Optional[float]     # user feedback overrides heuristic; None stays None
    has_positive_user_feedback: bool = False
    tool_error_count: int = 0            # verifier-pass count of error-shaped tool results
    tool_result_count: int = 0
    # Pass 4 #42: which source decided effective_score ("user" | "judge" |
    # "heuristic") — drives the dataset manifest trust tier.
    score_source: str = "heuristic"


def _tool_content_str(m: Dict[str, Any]) -> str:
    c = m.get("content")
    if isinstance(c, str):
        return c
    return json.dumps(c, ensure_ascii=False, default=str)


def verifier_tool_errors(messages: List[Dict[str, Any]]) -> int:
    """Lightweight curate-time re-scan for error-shaped tool results.

    Backlog #7: delegates to the shared core/failure_types classifier so the
    record-time heuristic and this verifier can never disagree on what a
    failed tool result looks like (error tag / traceback / exception /
    error-JSON body / nonzero exit code / empty).
    """
    n = 0
    for m in messages:
        if m.get("role") != "tool":
            continue
        failed, _kind = classify_tool_result(m.get("name", ""),
                                             _tool_content_str(m))
        if failed:
            n += 1
    return n


def load_trace(store: TraceStore, row: TraceRow) -> Optional[LoadedTrace]:
    """Read one trace's JSONL transcript + feedback. None if file unreadable."""
    try:
        with open(row.jsonl_path, encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
    except (OSError, json.JSONDecodeError):
        logger.warning("skipping trace %s: transcript unreadable (%s)",
                       row.trace_id, row.jsonl_path)
        return None
    messages = [rec["message"] for rec in lines if rec.get("kind") == "message"]

    effective = row.outcome_score  # already synced with latest user feedback by store
    positive_user = False
    try:
        fb = store.get_feedback(row.trace_id)
    except Exception:
        fb = []
    # Backlog #12 precedence: user > judge > heuristic. Feedback rows come back
    # ordered by id; the latest row of the winning source wins. With only
    # heuristic rows (or none) this resolves to row.outcome_score exactly as
    # before — default behavior unchanged.
    user_rows = [f for f in fb if f.get("source") == "user"]
    judge_rows = [f for f in fb if f.get("source") == "judge"
                  and f.get("score") is not None]
    if judge_rows:
        effective = judge_rows[-1].get("score", effective)
        score_source = "judge"
    else:
        score_source = "heuristic"
    if user_rows:
        # latest explicit user feedback wins (matches store sync semantics)
        effective = user_rows[-1].get("score", effective)
        positive_user = any(
            (s := f.get("score")) is not None and s > 0 for f in user_rows
        )
        score_source = "user"

    return LoadedTrace(
        row=row,
        messages=messages,
        effective_score=effective,
        has_positive_user_feedback=positive_user,
        tool_error_count=verifier_tool_errors(messages),
        tool_result_count=sum(1 for m in messages if m.get("role") == "tool"),
        score_source=score_source,
    )


# ------------------------------------------------------------------ dedup

def conversation_group_key(messages: List[Dict[str, Any]], agent_profile: Optional[str]) -> str:
    """Group key for prefix-overlap dedup: hash of first user message + profile.

    REPL sessions replay the growing history in each subsequent trace (Phase 2
    note #3); all traces sharing a first user turn and agent profile belong to
    one conversation lineage, and only the longest (then most recent) is kept.
    """
    first_user = ""
    for m in messages:
        if m.get("role") == "user":
            c = m.get("content")
            first_user = c if isinstance(c, str) else json.dumps(c, ensure_ascii=False, default=str)
            break
    h = hashlib.sha256(f"{agent_profile}\x00{first_user}".encode()).hexdigest()[:16]
    return h


def dedup_traces(traces: List[LoadedTrace]) -> List[LoadedTrace]:
    """Collapse prefix-overlapping traces (REPL continuations) per group.

    Keeps the LONGEST message list per group key; ties break toward the more
    recent timestamp. Order of returned list follows the input order.
    """
    best: Dict[str, LoadedTrace] = {}
    for t in traces:
        key = conversation_group_key(t.messages, t.row.agent_profile)
        cur = best.get(key)
        if cur is None:
            best[key] = t
            continue
        cand_rank = (len(t.messages), t.row.timestamp)
        cur_rank = (len(cur.messages), cur.row.timestamp)
        if cand_rank > cur_rank:
            best[key] = t
    kept_ids = {id(t) for t in best.values()}
    return [t for t in traces if id(t) in kept_ids]


# ------------------------------------------------------------- quality gates

def passes_quality_gates(t: LoadedTrace, conv: List[Dict[str, str]],
                         cfg: CurationConfig) -> bool:
    if not conv:
        return False
    if len(conv) < cfg.min_turns or len(conv) > cfg.max_turns:
        return False
    # Token-length estimate: chars/4 (rough heuristic; a real tokenizer pass
    # belongs in the Phase 4 worker, not here).
    # TODO(exact-tokens): core/tokenizer.py TokenCounter now makes exact
    # counts available (improvement #4), but curation runs offline — no
    # llama.cpp endpoint is guaranteed live at this point — so this gate
    # deliberately keeps the chars/4 heuristic rather than depending on a
    # server. Flip here if the Phase 4 worker ever gains a serving endpoint.
    est = sum(len(turn["value"]) for turn in conv) // 4
    if est > cfg.max_est_tokens:
        return False
    if (
        cfg.drop_tool_error_only
        and t.tool_result_count > 0
        and t.tool_error_count == t.tool_result_count
        and not (t.effective_score is not None and t.effective_score >= cfg.min_score)
        and not t.has_positive_user_feedback
    ):
        # Every tool result looks like an error and nothing positive says
        # otherwise -> low-value even if the run "completed".
        return False
    return True


def passes_selection(t: LoadedTrace, cfg: CurationConfig) -> bool:
    """Score gate for SFT-eligible traces (failures already filtered by query)."""
    if t.row.status == "failure" and not cfg.include_failures:
        return False
    if t.has_positive_user_feedback:
        return True
    if t.effective_score is None:
        # NULL score: keep only clean non-failure traces (never treat missing
        # as zero; status already excludes failure unless requested).
        return t.row.status in ("success", "partial") and t.tool_error_count == 0
    return t.effective_score >= cfg.min_score


# ------------------------------------------------------------------ DPO miner

@dataclass
class DPOPair:
    prompt_messages: List[Dict[str, str]]   # OpenAI-format up to (excl.) final answer
    chosen: str
    rejected: str
    chosen_trace_id: str
    rejected_trace_id: str


def _prompt_prefix(messages: List[Dict[str, Any]]) -> Optional[List[Dict[str, str]]]:
    """OpenAI-format prompt messages preceding the final plain assistant answer."""
    idx = None
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m.get("role") == "assistant" and not m.get("tool_calls"):
            c = m.get("content")
            if isinstance(c, str) and c.strip():
                idx = i
                break
    if idx is None:
        return None
    prompt: List[Dict[str, str]] = []
    for m in messages[:idx]:
        role = m.get("role")
        if role in ("system", "user"):
            c = m.get("content")
            prompt.append({"role": role,
                           "content": c if isinstance(c, str)
                           else json.dumps(c, ensure_ascii=False, default=str)})
        elif role == "assistant":
            # Render tool-call turns as ShareGPT-style text so the prompt is a
            # plain message list (TRL PreferenceDataset expects role/content).
            conv = messages_to_sharegpt([m])
            text = "\n".join(turn["value"] for turn in conv)
            prompt.append({"role": "assistant", "content": text})
        elif role == "tool":
            prompt.append({"role": "user", "content": _tool_content_str(m)})
    return prompt


def mine_dpo_pairs(groups: Dict[str, List[LoadedTrace]],
                   cfg: CurationConfig,
                   failure_notes: Optional[List[Dict[str, Any]]] = None,
                   ) -> List[DPOPair]:
    """Pair best-vs-worst final answers among traces sharing a conversation group.

    For each first-user-message group with at least two distinct scores, pair
    the highest-scored trace's final answer (chosen) against the
    lowest-scored/failure trace's final answer (rejected). Score ties do not
    produce pairs (no signal). NULL scores rank below any real score.

    Backlog #2/#7: traces whose runner-level ``failure_reason`` is infra-ish
    (connection/timeout/model_not_found — an llama-server hiccup, not a model
    fault) are excluded from being rejected material; model-failure traces
    (max_iterations, context overflow, plain bad answers) stay valid negatives.

    Pass 4 #33: ``failure_notes`` (the remediation rows from failure_notes)
    extend the exclusion — a trace whose reason has a RESOLVED note newer
    than the trace was recorded failed due to an already-fixed bug, so it is
    not "bad data" either. Without notes this is byte-identical to before.
    """
    notes = failure_notes or []
    pairs: List[DPOPair] = []
    for _key, members in groups.items():
        ranked: List[tuple] = []
        for t in members:
            answer = final_assistant_text(t.messages)
            if answer is None:
                continue
            # failure traces rank lowest regardless of score; NULL below real.
            base = -2.0 if t.row.status == "failure" else (
                t.effective_score if t.effective_score is not None else -1.0
            )
            infra = (is_infra_reason(t.row.failure_reason or "")
                     or resolved_note_covers(t.row, notes))
            ranked.append((base, t.row.timestamp, answer, t, infra))
        if len(ranked) < 2:
            continue
        ranked.sort(key=lambda x: (x[0], x[1]))
        best = ranked[-1]
        # negative candidates: strictly-below-best, excluding infra faults.
        negatives = [x for x in ranked[:-1] if not x[4]]
        if not negatives:
            continue
        worst = negatives[0]
        if best[0] <= worst[0]:
            continue  # tie or single effective score -> no preference signal
        prompt = _prompt_prefix(best[3].messages)
        if prompt is None:
            continue
        pairs.append(DPOPair(
            prompt_messages=prompt,
            chosen=best[2],
            rejected=worst[2],
            chosen_trace_id=best[3].row.trace_id,
            rejected_trace_id=worst[3].row.trace_id,
        ))
    return pairs


# ------------------------------------------------------------------ dataset IO

def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def render_train_yaml(dataset_dir: Path, cfg: CurationConfig,
                      dataset_id: str) -> str:
    """Render the train.yaml TEMPLATE (`train --config` contract).

    Field names mirror the documented contract (we
    generate a compatible file but never import it): top-level model / data /
    training / lora / logging. Phase 4's scheduler resolves the placeholders.
    """
    doc: Dict[str, Any] = {
        # --- train config (model/data/training/lora/logging) ---
        "model": cfg.base_model or "REPLACE_WITH_BASE_MODEL_PATH_OR_HF_ID",
        "data": {
            "local_dataset": [str(dataset_dir / "sft.jsonl")],
            "format_type": "sharegpt",
        },
        "training": {
            "training_type": "lora",
            "max_seq_length": min(16384, max(2048, cfg.max_est_tokens)),
            # load_in_4bit is AUTO-TIRED by model size (worker/core.vram:
            # >=3B params -> nf4 QLoRA base, smaller -> bf16). Pin it here
            # only to force a specific choice.
            # Template v2: adapters default into the HF hub cache (local
            # models--toaster-loop--<id> repo folder) instead of living under
            # the dataset dir; explicit curation.output_dir still wins, and an
            # unavailable hub falls back to <dataset>/outputs unchanged.
            "output_dir": (cfg.output_dir or adapter_output_dir(dataset_id)
                           or str(dataset_dir / "outputs")),
            "num_epochs": 3,
            "learning_rate": 2e-4,
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
            "warmup_steps": 5,
            "max_steps": 0,
            "save_steps": 0,
            "weight_decay": 0.01,
            "random_seed": 3407,
            "packing": False,
            # Response-only loss masking: the worker labels only assistant
            # spans (see training/worker.py tokenize_with_labels).
            "train_on_completions": True,
            "gradient_checkpointing": True,
        },
        "lora": {
            "lora_r": 64,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "target_modules": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        },
        "logging": {
            "enable_wandb": False,
            "wandb_project": f"toaster_loop-{dataset_id}",
            "enable_tensorboard": False,
            "tensorboard_dir": "runs",
        },
        # Keep this file to exactly the five contract sections above: the trainer's
        # pydantic Config ignores unknown keys, but Phase 4 may re-validate it
        # strictly, and manifest.json already carries toaster_loop metadata.
    }
    header = (
        f"# toaster-loop train.yaml template (v{TRAIN_YAML_TEMPLATE_VERSION}) for "
        f"dataset {dataset_id}\n"
        "# Contract: model/data/training/lora/logging.\n"
        "# Phase 4 (training scheduler) resolves REPLACE_* placeholders and may\n"
        "# override knobs before invoking the training worker.\n"
        "# GGUF export hint for Phase 4: quantization="
        f"{cfg.gguf_quant}; adapter output -> {doc['training']['output_dir']}\n"
    )
    return header + yaml.safe_dump(doc, sort_keys=False, default_flow_style=False)


@dataclass
class CurationResult:
    dataset_id: str
    dataset_dir: Path
    sft_count: int = 0
    dpo_count: int = 0
    sft_trace_ids: List[str] = field(default_factory=list)
    dpo_trace_ids: List[str] = field(default_factory=list)
    claimed: int = 0
    # Finding #8: traces selected for this dataset that lost the atomic claim
    # race to a concurrent curate and were dropped from every output. Zero for
    # uncontended runs; also summarized in `warnings` (flows to CLI + API).
    short_claims: int = 0
    warnings: List[str] = field(default_factory=list)


class Curater:
    """Turns buffered traces into a dataset directory. One instance per run."""

    def __init__(self, store: TraceStore, home: Path):
        self.store = store
        self.home = Path(home)
        self.datasets_root = self.home / "datasets"

    # ------------------------------------------------------------------ paths

    def dataset_dir(self, dataset_id: str) -> Path:
        return self.datasets_root / dataset_id

    def list_datasets(self) -> List[Dict[str, Any]]:
        """Manifests of all datasets on disk, oldest first.

        Sorted by the manifest created_at (ids are name-first since the
        readable-id change, so directory order is no longer chronological).
        Manifests without a usable created_at keep their directory order at
        the end."""
        out: List[Dict[str, Any]] = []
        if not self.datasets_root.is_dir():
            return out
        for d in sorted(self.datasets_root.iterdir()):
            mf = d / "manifest.json"
            if mf.is_file():
                try:
                    out.append(json.loads(mf.read_text()))
                except (OSError, json.JSONDecodeError):
                    out.append({"dataset_id": d.name, "error": "manifest unreadable"})
        out.sort(key=lambda m: str(m.get("created_at") or "\uffff"))
        return out

    def load_manifest(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        mf = self.dataset_dir(dataset_id) / "manifest.json"
        if not mf.is_file():
            return None
        try:
            return json.loads(mf.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def export_dataset(self, dataset_id: str, dest: Path) -> Path:
        src = self.dataset_dir(dataset_id)
        if not (src / "manifest.json").is_file():
            raise FileNotFoundError(f"no dataset with id {dataset_id} at {src}")
        dest = Path(dest).expanduser()
        target = dest / dataset_id
        if target.exists():
            raise FileExistsError(f"target already exists: {target}")
        shutil.copytree(src, target)
        return target

    # ------------------------------------------------------------------ curate

    def curate(self, cfg: CurationConfig, name: Optional[str] = None,
               rebuild_id: Optional[str] = None) -> CurationResult:
        """Run one curation pass; returns counts/paths of the written dataset.

        ``rebuild_id`` (--rebuild <dataset_id>): release that dataset's trace
        claims first and rewrite the same dataset id in place (its directory
        is removed and regenerated from the current buffer state). Without it,
        a fresh timestamped dataset id is created and only currently-unclaimed
        traces are considered.
        """
        warnings: List[str] = []

        if rebuild_id:
            # A source-derived dataset (hub/local_folder/website/imported) has
            # no trace claims — rebuilding it through the TRACE pipeline would
            # silently wipe its rows and emit an empty dataset in place (the
            # exact failure a user hit when a plain curate targeted one of
            # those ids). Refuse; source datasets are re-created from their
            # sources, not rebuilt from the buffer.
            existing = self.load_manifest(rebuild_id) or {}
            prov = str(existing.get("provenance") or "curated")
            if existing and prov != "curated":
                raise ValueError(
                    f"dataset {rebuild_id} was curated from a "
                    f"{prov} source, not buffered traces — rebuilding it here "
                    "would erase its rows. Re-create it from the Datasets "
                    "page (Ingest folder / Crawl site / Import hub) instead.")
            released = self.store.reset_claims(rebuild_id)
            logger.info("--rebuild %s: released %d claims", rebuild_id, released)
            dataset_id = rebuild_id
            ds_dir = self.dataset_dir(dataset_id)
        else:
            # Reserve the directory BEFORE claiming (finding #8): an empty dir
            # is not a dataset until manifest.json lands, and reserving via
            # mkdir(exist_ok=False) makes concurrent runs pick distinct ids
            # instead of both racing the same mkdir later.
            dataset_id, ds_dir = self._reserve_dataset_dir(name)

        # ---- selection ------------------------------------------------------
        statuses = ["success", "partial", "unknown"]
        if cfg.include_failures:
            statuses.append("failure")
        rows = self.store.query_for_curation(
            include_curated=False,
            statuses=statuses,
            since=cfg.since,
            model_alias=cfg.model_alias,
            agent_profile=cfg.agent_profile,
            project=cfg.project,
        )

        loaded: List[LoadedTrace] = []
        for r in rows:
            t = load_trace(self.store, r)
            if t is not None:
                loaded.append(t)

        # Failures are DPO-negative material only: keep them aside; they never
        # enter SFT rows unless include_failures AND they pass the score gate.
        sft_pool = [t for t in loaded
                    if (t.row.status != "failure" or cfg.include_failures)
                    and passes_selection(t, cfg)]
        dpo_pool = list(loaded)  # groups need both good and bad members

        # ---- dedup (prefix-overlap) -----------------------------------------
        sft_deduped = dedup_traces(sft_pool)
        # WP9 funnel telemetry: prefix-overlap collapses (REPL continuations).
        dedup_rejects = len(sft_pool) - len(sft_deduped)

        # ---- quality gates + SFT rows ----------------------------------------
        sft_rows: List[tuple] = []  # (trace, conversations)
        for t in sft_deduped:
            conv = messages_to_sharegpt(t.messages)
            if passes_quality_gates(t, conv, cfg):
                sft_rows.append((t, conv))
        quality_rejects = len(sft_deduped) - len(sft_rows)
        limit_dropped = 0
        if cfg.limit is not None and len(sft_rows) > cfg.limit:
            # Deterministic cap: keep the highest-scored (NULL last), then most
            # recent, then id order; re-sort output by timestamp for stability.
            def _rank(item):
                sc = item[0].effective_score
                return (-1e9 if sc is None else sc, item[0].row.timestamp,
                        item[0].row.trace_id)
            sft_rows.sort(key=_rank, reverse=True)
            dropped = len(sft_rows) - cfg.limit
            limit_dropped = dropped
            warnings.append(f"limit={cfg.limit}: dropped {dropped} SFT rows")
            sft_rows = sft_rows[: cfg.limit]
        sft_rows.sort(key=lambda item: (item[0].row.timestamp, item[0].row.trace_id))

        # ---- DPO mining ------------------------------------------------------
        groups: Dict[str, List[LoadedTrace]] = {}
        for t in dpo_pool:
            groups.setdefault(
                conversation_group_key(t.messages, t.row.agent_profile), []
            ).append(t)
        # Pass 4 #33: remediation notes so already-fixed failure reasons stop
        # being mined as bad-data negatives (empty list -> old behavior).
        try:
            failure_notes = self.store.list_failure_notes()
        except Exception:  # fresh/odd DB shapes never break curation
            failure_notes = []
        pairs = mine_dpo_pairs(groups, cfg, failure_notes=failure_notes)
        if cfg.limit is not None and len(pairs) > cfg.limit:
            warnings.append(f"limit={cfg.limit}: dropped {len(pairs) - cfg.limit} DPO pairs")
            pairs = pairs[: cfg.limit]
        if not pairs:
            warnings.append(
                "no DPO preference pairs found (need >=2 traces per conversation "
                "group with differing scores); emitting empty dpo.jsonl"
            )

        # ---- claim before write (finding #8) ----------------------------------
        # Claim atomically BEFORE any file exists: store.claim_traces_owned
        # flips only curated=0 rows with rowcount accounting inside one BEGIN
        # IMMEDIATE transaction (the registry.claim_next_queued pattern), so two
        # concurrent curates can never both own a trace. Only IDs in `owned`
        # may appear in sft.jsonl/dpo.jsonl/manifest.json, so the manifests of
        # racing datasets are disjoint by construction.
        want_sft = cfg.formats in ("sft", "both")
        want_dpo = cfg.formats in ("dpo", "both")
        claim_ids = list(dict.fromkeys(
            ([t.row.trace_id for t, _ in sft_rows] if want_sft else [])
            + (sorted({tid for p in pairs
                       for tid in (p.chosen_trace_id, p.rejected_trace_id)})
               if want_dpo else [])
        ))
        owned: set = set()
        short_claims = 0
        try:
            owned = set(self.store.claim_traces_owned(claim_ids, dataset_id))
            lost = [tid for tid in claim_ids if tid not in owned]
            short_claims = len(lost)
            if short_claims:
                sample = ", ".join(lost[:5]) + (", ..." if short_claims > 5 else "")
                logger.warning(
                    "dataset %s: short claim -- dropped %d of %d selected traces "
                    "already claimed by a concurrent curate (%s)",
                    dataset_id, short_claims, len(claim_ids), sample)
                warnings.append(
                    f"short claim: {short_claims} of {len(claim_ids)} selected "
                    f"traces lost to a concurrent claim and were excluded "
                    f"({sample})")
                # A DPO pair needs BOTH sides owned -- one missing side is no
                # preference signal. The surviving side, if claimed here and
                # then unused, is released back to the buffer after the write.
                sft_rows = [(t, conv) for t, conv in sft_rows
                            if t.row.trace_id in owned]
                pairs = [p for p in pairs
                         if p.chosen_trace_id in owned
                         and p.rejected_trace_id in owned]

            # Deterministic output lists (claims applied): the manifest and
            # result describe exactly what the files below will contain.
            sft_trace_ids: List[str] = (
                [t.row.trace_id for t, _ in sft_rows] if want_sft else [])
            dpo_trace_ids: List[str] = (
                sorted({tid for p in pairs
                        for tid in (p.chosen_trace_id, p.rejected_trace_id)})
                if want_dpo else [])

            manifest = {
            "dataset_id": dataset_id,
            "title": name or dataset_id,   # human label for the UI list
            "created_at": _utcnow(),
            "dataset_schema_version": DATASET_SCHEMA_VERSION,
            "trace_schema_version": 1,
            "train_yaml_template_version": TRAIN_YAML_TEMPLATE_VERSION,
            # Pass 4 #42 trust tier (precedence curated > judge_scored >
            # imported). Honest per-row flow: a judge verdict only decides a
            # trace's effective score via load_trace's user > judge >
            # heuristic precedence, so the dataset is judge_scored exactly
            # when some selected SFT row was decided by a judge verdict.
            "trust": ("judge_scored"
                      if any(t.score_source == "judge" for t, _ in sft_rows)
                      else "curated"),
            "filters": {
                "min_score": cfg.min_score,
                "include_failures": cfg.include_failures,
                "since": cfg.since,
                "model_alias": cfg.model_alias,
                "agent_profile": cfg.agent_profile,
                "project": cfg.project,
                "limit": cfg.limit,
                "formats": cfg.formats,
                "min_turns": cfg.min_turns,
                "max_turns": cfg.max_turns,
                "max_est_tokens": cfg.max_est_tokens,
            },
            "files": {
                "sft.jsonl": {
                    "count": len(sft_trace_ids) if want_sft else 0,
                    "trace_ids": sft_trace_ids,
                    "format": "sharegpt",
                },
                "dpo.jsonl": {
                    "count": len(pairs) if want_dpo else 0,
                    "trace_ids": sorted(set(dpo_trace_ids)) if want_dpo else [],
                    "format": "trl-preference",
                },
            },
            "format_hints": {
                "sft_format_type": "sharegpt",
                "train_yaml": "train.yaml",
                "note": (
                    "sft.jsonl is format_type=sharegpt compatible "
                    "(conversations: [{from,value}]); dpo.jsonl targets TRL "
                    "PreferenceDataset via the DPO worker."
                ),
            },
            # Finding #8: traces dropped from this dataset by a concurrent
            # claim; zero for uncontended runs.
            "short_claims": short_claims,
            "warnings": warnings,
            }

            if rebuild_id and ds_dir.exists():
                shutil.rmtree(ds_dir)
            # Fresh ids already exist from the pre-claim reservation; a
            # rebuild recreates the dir it just removed.
            ds_dir.mkdir(parents=True, exist_ok=True)

            if want_sft:
                with open(ds_dir / "sft.jsonl", "w", encoding="utf-8") as f:
                    for t, conv in sft_rows:
                        row = {
                            "conversations": conv,
                            # provenance extras (HF datasets tolerate extra
                            # columns; format_type=sharegpt reads only)
                            "trace_id": t.row.trace_id,
                            "model": t.row.model_alias,
                            "completed": t.row.status == "success",
                        }
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")

            if want_dpo:
                with open(ds_dir / "dpo.jsonl", "w", encoding="utf-8") as f:
                    for p in pairs:
                        row = {
                            "prompt": p.prompt_messages,
                            "chosen": p.chosen,
                            "rejected": p.rejected,
                            "metadata": {
                                "chosen_trace_id": p.chosen_trace_id,
                                "rejected_trace_id": p.rejected_trace_id,
                            },
                        }
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")

            (ds_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            (ds_dir / "train.yaml").write_text(
                render_train_yaml(ds_dir, cfg, dataset_id), encoding="utf-8"
            )
        except Exception:
            # Failure cleanup (finding #8): janitor.py only archives terminal
            # *models* -- nothing reaps stale trace claims -- so every claim
            # this run holds is released back to the buffer here and the
            # partial dataset directory removed. The traces are curatable
            # again; the exception propagates to the caller unchanged.
            try:
                released = self.store.release_claims(sorted(owned), dataset_id)
                logger.warning("dataset %s: write failed after claiming; "
                               "released %d claims", dataset_id, released)
            except Exception:  # pragma: no cover - defensive
                logger.exception("dataset %s: could not release claims",
                                 dataset_id)
            shutil.rmtree(ds_dir, ignore_errors=True)
            raise

        # Claimed-but-unshipped traces (e.g. the chosen side of a DPO pair
        # dropped above because its counterpart was lost) return to the
        # buffer -- preserving the pre-fix "claim only what shipped" semantics
        # now that claiming happens before writing.
        used_ids = set(sft_trace_ids) | set(dpo_trace_ids)
        unused = sorted(owned - used_ids)
        if unused:
            self.store.release_claims(unused, dataset_id)
            owned -= set(unused)
        claimed = len(owned)

        # ---- WP9 funnel analytics (append-only; never breaks a curate) ------
        # Snapshot of the selection funnel this run just executed. Honest
        # denominators per WP2c: every stage count travels in the same record,
        # and the score histogram carries scored/unscored beside its buckets.
        try:
            analytics.record_curation_funnel(self.home, {
                "dataset_id": dataset_id,
                "name": name,
                "rebuild": bool(rebuild_id),
                "created_at": manifest.get("created_at"),
                "trust": manifest.get("trust"),
                "project": cfg.project,
                "model_alias": cfg.model_alias,
                "agent_profile": cfg.agent_profile,
                # funnel stages (each is a true denominator for the next)
                "traces_scanned": len(rows),
                "loaded": len(loaded),
                "eligible": len(sft_pool),
                "dedup_rejects": dedup_rejects,
                "quality_rejects": quality_rejects,
                "limit_dropped": limit_dropped,
                "claim_race_lost": short_claims,
                "curated": claimed,
                "sft_count": len(sft_trace_ids) if want_sft else 0,
                "dpo_pairs": len(pairs) if want_dpo else 0,
                # judge/heuristic score distribution over everything loaded
                # this run (effective scores after user>judge>heuristic
                # precedence); unscored counted honestly, never coerced to 0.
                "score_histogram": analytics.score_histogram(
                    [t.effective_score for t in loaded]),
            })
        except Exception:  # pragma: no cover - analytics must never break curate
            logger.debug("could not persist curation funnel snapshot",
                         exc_info=True)

        # Warnings are surfaced by the caller (CLI prints them); log at debug
        # to avoid double-printing.
        for w in warnings:
            logger.debug(w)

        return CurationResult(
            dataset_id=dataset_id,
            dataset_dir=ds_dir,
            sft_count=len(sft_trace_ids) if want_sft else 0,
            dpo_count=len(pairs) if want_dpo else 0,
            sft_trace_ids=sft_trace_ids,
            dpo_trace_ids=sorted(set(dpo_trace_ids)) if want_dpo else [],
            claimed=claimed,
            short_claims=short_claims,
            warnings=warnings,
        )

    # ---------------------------------------------------------------- helpers

    def _reserve_dataset_dir(self, name: Optional[str]) -> "tuple":
        """<UTC timestamp>--[name] — sortable, unique per run, RESERVED early.

        Replaces the old _make_dataset_id (finding #8): the directory is
        created here with ``mkdir(exist_ok=False)``, which acts as the
        cross-process lock behind claim-before-write — two curates in the same
        second can never reserve the same id, so each run's claims are scoped
        to a directory nobody else will write. The empty dir is not a dataset
        until manifest.json lands (``list_datasets`` skips it) and any failure
        path rmtree's it. A numeric suffix is appended on collision (rapid
        successive runs). Returns (id, dir)."""
        from .sources import alloc_dataset_id
        ds_dir, dataset_id = alloc_dataset_id(self.home, name)
        return dataset_id, ds_dir

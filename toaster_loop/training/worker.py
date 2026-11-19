"""Training worker — runs in a SUBPROCESS (`python -m toaster_loop.training.worker`).

Why a subprocess: GPU-heavy imports (torch/transformers/peft/trl) stay out of
the core daemon. All training functionality is built into toaster_loop on top of
standard open-source libraries (Apache/BSD ecosystems); no external trainer
framework or CLI is imported, and nothing outside this worker's declared
extras is needed at serve/curate time.

Job kinds (mirrors registry JOB_KINDS):

  sft   LoRA fine-tune via transformers SFTTrainer + peft:
          AutoModelForCausalLM.from_pretrained(...) + bf16/4-bit options
          get_peft_model(...)                          # LoRA knobs
          sharegpt conversations -> tokenizer.apply_chat_template text
          SFTTrainer(args=..., train_dataset=ds)
          trainer.train() ; model.save_pretrained(adapter_dir)

  dpo   TRL DPOTrainer over dpo.jsonl ({prompt[], chosen, rejected}).

  export  adapter -> GGUF, fully built-in:
       1. Merge the LoRA adapter into the base checkpoint (peft
          merge_and_unload) into <output_dir>/merged.
       2. llama.cpp convert_hf_to_gguf.py + llama-quantize as subprocesses
          (plain open-source tooling; discovered via the toaster_loop-managed
          <TOASTER_LOOP_HOME>/llama.cpp, /opt/llama.cpp or $TOASTER_LOOP_LLAMA_CPP).
       No converter available => actionable error, adapter artifact kept.

  validate  handled in-process by training/validate.py (no GPU imports); the
        scheduler records a jobs row for it but does not spawn this worker.

Progress protocol: one JSON line per event appended to --log-path:
    {"ts": ..., "job": ..., "event": "start|progress|phase|done|error", ...}
Backlog item #4 extends (never breaks) the protocol with live training events:
    {"event": "progress", "step": N, "total": M, "loss": x}   per logging step
    {"event": "eval",     "step": N, "eval_loss": y}          per evaluation
The scheduler tails this file for display; both lines are plain additive JSON —
older readers only look at start/done/error and stay compatible.

--dry-run validates config + dataset load and exits 0 without touching GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# WP10 step 2: streamed-LoRA backend seam. This import is torch-free (the
# streamed package keeps every heavy touch lazy inside functions), so the
# worker CLI still starts without the GPU stack; run_streamed_sft below is
# the only entry that crosses into it, and only when training.backend=
# streamed selects it.
from .streamed.runner import parse_backend, run_streamed_sft

PLACEHOLDER = "REPLACE_WITH_BASE_MODEL_PATH_OR_HF_ID"


# --------------------------------------------------------------- progress log

def _emit(log_path: Optional[str], event: str, **fields: Any) -> None:
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pid": os.getpid(),
        "event": event,
        **fields,
    }
    line = json.dumps(rec, ensure_ascii=False)
    print(line, flush=True)  # flush: the scheduler tails our stdout promptly
    if log_path:
        try:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass


# ------------------------------------------------- progress/eval events (#4)

def progress_event(step: int, total: Optional[int] = None,
                   loss: Optional[float] = None) -> Dict[str, Any]:
    """One training-progress event (backlog item #4). None fields are omitted."""
    ev: Dict[str, Any] = {"event": "progress", "step": int(step)}
    if total is not None:
        ev["total"] = int(total)
    if loss is not None:
        ev["loss"] = float(loss)
    return ev


def eval_event(step: int, eval_loss: Optional[float]) -> Dict[str, Any]:
    """One evaluation-pass event (backlog item #4)."""
    ev = {"event": "eval", "step": int(step)}
    if eval_loss is not None:
        ev["eval_loss"] = float(eval_loss)
    return ev


# ------------------------- adaptive LoRA rank growth (inspiration pass 4 #34)
# Idea adapted from PIASystem (MIT) app/core/training_engine.py — RankScaler:
# when loss keeps improving steadily, capacity (not data) may be the limit.
# HF Trainer can't hot-swap adapter shapes, so the honest toaster_loop shape is a
# STAGE-CHAINED resume: watch the loss stream; on trigger, stop at the current
# checkpoint, rebuild LoraConfig at 2×r seeded from the trained adapter where
# shapes allow (zero-padded rank growth preserves the learned delta exactly),
# and continue training. `training.rank_scaling` (default false) gates all of
# it — adopt only once shadow eval (#32) can prove growth helps.

class RankScaler:
    """Pure loss-stream watcher: trigger after N consecutive improvements.

    Torch-free and clock-free so unit tests drive it directly. ``observe``
    returns True exactly once per stage when ``min_improvement_steps``
    consecutive logged losses each improved on the running best by at least
    ``min_rel_improvement`` (PIASystem semantics: 3 steps, 1%), and only while
    growth is still possible below ``max_rank``.
    """

    def __init__(self, *, enabled: bool = False, max_rank: int = 256,
                 min_improvement_steps: int = 3,
                 min_rel_improvement: float = 0.01) -> None:
        self.enabled = bool(enabled)
        self.max_rank = int(max_rank)
        self.min_improvement_steps = max(1, int(min_improvement_steps))
        self.min_rel_improvement = float(min_rel_improvement)
        self.current_r: Optional[int] = None  # set per stage by run_sft
        self.best_loss: Optional[float] = None
        self.streak = 0
        self.triggered = False
        #: [{from_r, to_r, step, best_loss}] — recorded on each growth.
        self.rank_history: List[Dict[str, Any]] = []

    def start_stage(self, current_r: int) -> None:
        """Begin (or re-begin) watching for a stage trained at ``current_r``."""
        self.current_r = int(current_r)
        self.best_loss = None
        self.streak = 0
        self.triggered = False

    def plan_growth(self) -> Optional[int]:
        """Next rank on trigger: double, capped at max_rank; None if pinned."""
        if not self.enabled or self.current_r is None:
            return None
        new_r = min(2 * int(self.current_r), self.max_rank)
        return new_r if new_r > self.current_r else None

    def observe(self, step: int, loss: Optional[float]) -> bool:
        """Feed one logged loss; True when the stage should stop and grow."""
        if not self.enabled or loss is None:
            return False
        loss = float(loss)
        if self.best_loss is None:
            self.best_loss = loss
            return False
        if loss <= self.best_loss * (1.0 - self.min_rel_improvement):
            self.streak += 1
            self.best_loss = loss
        else:
            self.streak = 0
            self.best_loss = min(self.best_loss, loss)
        if (self.streak >= self.min_improvement_steps and not self.triggered
                and self.plan_growth() is not None):
            self.triggered = True
            return True
        return False

    def record_growth(self, to_r: int, step: int) -> Dict[str, Any]:
        entry = {"from_r": int(self.current_r or 0), "to_r": int(to_r),
                 "step": int(step),
                 "best_loss": self.best_loss}
        self.rank_history.append(entry)
        return entry


def rank_scaling_from_config(cfg: Dict[str, Any]) -> RankScaler:
    """Parse `training.rank_scaling` (pass4 #34): false/absent => disabled;
    true => defaults (max_rank 256, 3 consecutive >=1% improvements); a mapping
    {max_rank?, min_improvement_steps?, min_rel_improvement?} tunes it."""
    raw = (cfg.get("training") or {}).get("rank_scaling", False)
    if isinstance(raw, dict):
        return RankScaler(
            enabled=True,
            max_rank=int(raw.get("max_rank", 256)),
            min_improvement_steps=int(raw.get("min_improvement_steps", 3)),
            min_rel_improvement=float(raw.get("min_rel_improvement", 0.01)),
        )
    return RankScaler(enabled=bool(raw))


def grow_lora_state(model: Any, full_state: Dict[str, Any], *,
                    old_name: str, new_name: str) -> tuple:
    """Seed a freshly added peft adapter from a trained smaller-rank one.

    Multi-adapter peft keys look like ``....<old_name>.lora_A.weight`` vs the
    new adapter's ``...<new_name>.lora_A.weight`` at the same module path.
    Shapes: lora_A is [r, in_features], lora_B is [out_features, r]; copying
    into the top-left block leaves the extra rank rows/columns zero, so the
    composed delta B@A of the seeded values is EXACTLY the old adapter's —
    stage N+1 starts where stage N left off with new capacity to fill.

    Returns (seeded_count, candidate_count); a shape surprise for any key
    leaves that tensor at peft's init rather than failing the growth."""
    import torch  # lazy: GPU env only

    seeded = 0
    candidates = 0
    params = dict(model.named_parameters())
    with torch.no_grad():
        for key, old in full_state.items():
            if f".{old_name}.lora_" not in key:
                continue
            new_key = key.replace(f".{old_name}.", f".{new_name}.")
            new = params.get(new_key)
            if new is None or tuple(new.shape) == tuple(old.shape):
                continue
            candidates += 1
            if ".lora_A." in key and new.dim() == 2 \
                    and new.shape[1] == old.shape[1] and new.shape[0] > old.shape[0]:
                new[: old.shape[0], :] = old
                seeded += 1
            elif ".lora_B." in key and new.dim() == 2 \
                    and new.shape[0] == old.shape[0] and new.shape[1] > old.shape[1]:
                new[:, : old.shape[1]] = old
                seeded += 1
    return seeded, candidates


def make_protocol_callback(log_path: Optional[str], scaler: Optional["RankScaler"] = None):
    """TrainerCallback duck-type emitting the progress protocol (backlog #4).

    When transformers is importable (the GPU env) the class subclasses
    TrainerCallback; otherwise it stays a plain object with the same
    on_log/on_evaluate signatures so unit tests can drive it without torch.
    HF's CallbackHandler invokes callbacks structurally, so both work.

    ``scaler`` (pass4 #34): when given, each logged loss is fed to the
    RankScaler; on trigger the callback asks the Trainer to save a checkpoint
    and stop (control.should_save/should_stop) — run_sft then grows the rank
    and starts the next stage.
    """
    base: Any = object
    try:  # lazy heavy import — allowed only in worker.py (license hygiene)
        from transformers import TrainerCallback  # type: ignore[import-not-found]

        base = TrainerCallback
    except Exception:
        pass

    class ProtocolLogCallback(base):  # type: ignore[misc,valid-type]
        def __init__(self) -> None:
            super().__init__()
            self.last_eval_loss: Optional[float] = None
            #: Deep-dive fix: mid-run NaN divergence detection. Set when a
            #: logged loss is NaN/inf — the run stops at once instead of
            #: burning hours of GPU on a numerically dead model. The caller
            #: (run_sft/run_dpo/run_grpo) fails the job honestly on this flag.
            self.nan_abort_step: Optional[int] = None

        def on_log(self, args, state, control=None, **kwargs):  # noqa: ANN001
            logs = kwargs.get("logs") or {}
            loss = logs.get("loss")
            if loss is None:
                for entry in reversed(getattr(state, "log_history", None) or []):
                    if isinstance(entry, dict) and entry.get("loss") is not None:
                        loss = entry["loss"]
                        break
            total = getattr(state, "max_steps", None) or None
            if not total:
                t2 = getattr(args, "max_steps", None)
                total = t2 if (t2 and int(t2) > 0) else None
            _emit(log_path, **progress_event(
                getattr(state, "global_step", 0), total, loss))
            # Deep-dive fix: NaN/inf loss => abort the run immediately. The
            # post-run find_nan_losses gate stays (defense in depth), but it
            # only fires hours later; this one saves the GPU window.
            if loss is not None and control is not None \
                    and self.nan_abort_step is None:
                try:
                    _lv = float(loss)
                except (TypeError, ValueError):
                    _lv = None
                if _lv is not None and math.isnan(_lv):
                    self.nan_abort_step = int(getattr(state, "global_step", 0))
                    _emit(log_path, "warning", message=(
                        f"NaN loss at step {self.nan_abort_step} — aborting "
                        "run early (diverged training); checkpoint kept for "
                        "inspection, job will be marked failed"))
                    control.should_save = False
                    control.should_stop = True
            # pass4 #34: rank-growth trigger — stop at a checkpoint so the
            # stage loop can rebuild LoraConfig at a higher rank.
            if scaler is not None and control is not None \
                    and scaler.observe(getattr(state, "global_step", 0), loss):
                control.should_save = True
                control.should_stop = True

        def on_evaluate(self, args, state, control=None, **kwargs):  # noqa: ANN001
            metrics = kwargs.get("metrics") or {}
            el = metrics.get("eval_loss")
            if el is not None:
                try:
                    self.last_eval_loss = float(el)
                except (TypeError, ValueError):
                    pass
            _emit(log_path, **eval_event(getattr(state, "global_step", 0), el))

    return ProtocolLogCallback()


# --------------------------------------------- eval split + cadence (#4)

# Eval-split floors as DECLARED POLICY (inspiration pass 3 #22). Studio's
# eval_dataset.py uses MIN_EVAL_ROWS=16 / MIN_TOTAL=32; ours stay lower on
# purpose: toaster_loop trains on dozens of curated traces, and disabling eval
# below 32 rows would disable it nearly always. The principle we adopt is the
# *shape*: fixed seed + explicit floors + "disabled deliberately rather than
# degraded" — never a 1-row eval split whose noise could gate promotion.
EVAL_MIN_TOTAL_ROWS = 20
EVAL_MIN_ROWS = 2

# Named-split candidates checked next to the training files (imported datasets
# that already carry their own holdout; inspiration pass 3 #22).
EVAL_SPLIT_FILENAMES = ("eval.jsonl", "validation.jsonl", "val.jsonl")


def find_named_eval_files(data_sec: Dict[str, Any],
                          train_files: List[str]) -> List[str]:
    """Locate an explicit eval split for a dataset (inspiration pass 3 #22).

    Order: `data.eval_dataset` from the train.yaml (string or list) wins; else
    auto-detect ``eval.jsonl`` / ``validation.jsonl`` / ``val.jsonl`` sitting
    next to any training file. A named split means the dataset author already
    carved the holdout — the worker uses it verbatim instead of splitting the
    training rows (floors become warnings, not vetoes: author intent wins).
    Returns existing files only; [] => fall back to maybe_eval_split.
    """
    named = data_sec.get("eval_dataset")
    if isinstance(named, str):
        named = [named]
    candidates: List[str] = []
    if isinstance(named, list):
        candidates.extend(str(x) for x in named if str(x).strip())
    else:
        seen = set()
        for f in train_files:
            for fname in EVAL_SPLIT_FILENAMES:
                p = Path(f).expanduser().parent / fname
                if str(p) not in seen:
                    seen.add(str(p))
                    candidates.append(str(p))
    out: List[str] = []
    train_set = {str(Path(f).expanduser()) for f in train_files}
    for c in candidates:
        p = Path(c).expanduser()
        if str(p) in train_set or not p.is_file():
            continue  # never let the eval file double as training data
        if str(p) not in out:
            out.append(str(p))
    return out


def maybe_eval_split(rows: List[Any], fraction: float = 0.1, seed: int = 42,
                     min_total: int = EVAL_MIN_TOTAL_ROWS,
                     min_eval: int = EVAL_MIN_ROWS) -> Tuple[List[Any], Optional[List[Any]]]:
    """Deterministically carve an eval split off `rows` (backlog item #4).

    Tiny datasets are the norm in this loop, so splitting by fraction alone
    would starve training: we only split when there are at least `min_total`
    rows overall AND the eval side would receive at least `min_eval` rows;
    otherwise eval is skipped entirely (returns (rows, None)) and training
    keeps its previous no-eval behavior. The shuffle uses random.Random(seed)
    so reruns on the same dataset produce the identical split.
    """
    n = len(rows)
    if n == 0 or not fraction or float(fraction) <= 0 or n < min_total:
        return list(rows), None
    n_eval = int(n * float(fraction))
    if n_eval < min_eval:
        return list(rows), None
    idx = list(range(n))
    random.Random(int(seed)).shuffle(idx)
    eval_idx = set(idx[:n_eval])
    train = [r for i, r in enumerate(rows) if i not in eval_idx]
    ev = [rows[i] for i in sorted(eval_idx)]
    return train, ev


# ------------------------------------------- checkpoint resume (pass 3 #21)

#: What `training.resume` accepts besides an explicit checkpoint path.
RESUME_AUTO = "auto"
RESUME_NEVER = ("", "false", "no", "off", "none")


def _checkpoint_is_unfinished(ckpt: Path) -> bool:
    """True unless trainer_state.json proves the run already reached its end.

    Matters because `output_dir` is shared between runs of one dataset (the
    curater template points it at <dataset>/outputs), so a previous COMPLETED
    run leaves checkpoints behind that must never be resumed from by default.
    HF writes global_step + max_steps into trainer_state.json; unreadable or
    partial state counts as unfinished (conservative: an interrupted save is
    exactly the case resume exists for).
    """
    try:
        state = json.loads((ckpt / "trainer_state.json").read_text())
    except (OSError, ValueError):
        return True
    try:
        step = int(state.get("global_step") or 0)
        total = int(state.get("max_steps") or 0)
    except (TypeError, ValueError):
        return True
    if step < 1:
        return False          # nothing trained yet — start clean instead
    return total <= 0 or step < total


def find_resumable_checkpoint(outputs_root) -> Optional[str]:
    """Highest resumable ``checkpoint-N`` under a run's outputs root, or None.

    A checkpoint counts as resumable only when it holds BOTH adapter weights
    (``adapter_model.safetensors`` / ``.bin``) AND trainer state
    (``trainer_state.json``): a directory with just one of the two is either an
    in-flight save or a mid-crash write, and handing it to
    ``trainer.train(resume_from_checkpoint=...)`` dies with a confusing error.
    Checkpoints whose trainer state shows a finished run are skipped (see
    ``_checkpoint_is_unfinished``).

    The scan is deliberately shallow (one level under ``outputs_root``, name
    must match ``checkpoint-<int>``) so a stray path can never point training
    outside the run dir. Newest step number wins; ties broken lexically.
    """
    root = Path(str(outputs_root)).expanduser()
    if not root.is_dir():
        return None
    best: Optional[tuple] = None  # (step, name, path)
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        name = entry.name
        if not name.startswith("checkpoint-"):
            continue
        step_part = name[len("checkpoint-"):]
        if not step_part.isdigit():
            continue
        has_weights = ((entry / "adapter_model.safetensors").is_file()
                       or (entry / "adapter_model.bin").is_file())
        has_state = (entry / "trainer_state.json").is_file()
        if not (has_weights and has_state):
            continue
        if not _checkpoint_is_unfinished(entry):
            continue
        cand = (int(step_part), name, str(entry))
        if best is None or cand[:2] > best[:2]:
            best = cand
    return best[2] if best else None


def resolve_resume_path(tr: Dict[str, Any], outputs_root,
                        log_path: Optional[str] = None,
                        fingerprint: Optional[str] = None) -> Optional[str]:
    """Turn the `training.resume` knob into a concrete checkpoint path (or None).

      * unset / ``auto``  -> auto-discover in the run dir (crash / cancel /
        reboot resumes where it stopped; nothing found => train from scratch).
      * explicit path     -> use it when resumable; refuse loudly otherwise
        (an explicit request that silently starts over is worse than an error).
      * ``false``         -> never resume, even with checkpoints present.
    """
    raw = tr.get("resume", RESUME_AUTO)
    if isinstance(raw, bool):
        token = "true" if raw else "false"
    else:
        token = str(raw).strip()
    if raw is False or token.lower() in RESUME_NEVER:
        return None
    if token.lower() in ("auto", "true"):
        found = find_resumable_checkpoint(outputs_root)
        if found and fingerprint is not None \
                and not run_marker_matches(outputs_root, fingerprint):
            # Deep-dive fix (production readiness): auto-resume is scoped to
            # the same run identity. A checkpoint left by a DIFFERENT config
            # (different data/model/lr/rank/base_adapter) in a shared output
            # dir must never silently poison a new run — `train resume` pins
            # an explicit path for the legitimate cross-config case.
            _emit(log_path, "warning", message=(
                f"found checkpoint {found} but it belongs to a different "
                "training config (run marker fingerprint mismatch) — "
                "starting fresh; use training.resume=<checkpoint path> to "
                "resume across configs deliberately"))
            return None
        if found:
            _emit(log_path, "resume_state", checkpoint=found)
        return found
    p = Path(token).expanduser()
    if (p / "trainer_state.json").is_file() and (
            (p / "adapter_model.safetensors").is_file()
            or (p / "adapter_model.bin").is_file()):
        _emit(log_path, "resume_state", checkpoint=str(p), explicit=True)
        return str(p)
    raise ValueError(
        f"training.resume={token!r} is not a resumable checkpoint "
        "(needs adapter weights + trainer_state.json)")


def plan_cadence(n_train_rows: int, *, batch_size: int = 2, grad_accum: int = 4,
                 epochs: float = 3.0, max_steps: int = 0, save_steps: int = 0,
                 default_save_steps: int = 50,
                 logging_steps: int = 10) -> Dict[str, Any]:
    """Checkpoint/eval cadence knobs for TrainingArguments (backlog item #4).

    Honors the existing `save_steps` knob (the curater template sets it to 0,
    meaning "unset" => use `default_save_steps`). When the whole run is
    shorter than one save interval we switch to "epoch" cadence so
    checkpoints still appear; when the dataset can't even produce one
    optimizer step we keep "no" saving — never crash on degenerate datasets.
    Eval cadence mirrors save cadence (one eval per checkpoint).
    """
    if not save_steps or int(save_steps) <= 0:
        save_steps = int(default_save_steps)
    else:
        save_steps = int(save_steps)
    opt_steps_per_epoch = max(
        1, math.ceil(n_train_rows / max(1, int(batch_size) * max(1, int(grad_accum)))))
    total = (int(max_steps) if max_steps and int(max_steps) > 0
             else opt_steps_per_epoch * max(1, int(math.ceil(max(1.0, float(epochs))))))
    if n_train_rows < 1 or total < 1:
        return {"save_strategy": "no", "eval_strategy": "no",
                "logging_steps": int(logging_steps)}
    if total <= save_steps:
        return {"save_strategy": "epoch", "eval_strategy": "epoch",
                "logging_steps": int(logging_steps)}
    return {"save_strategy": "steps", "save_steps": save_steps,
            "eval_strategy": "steps", "eval_steps": save_steps,
            "logging_steps": int(logging_steps)}


# ------------------------- continual training + run identity (deep-dive fix)

#: Marker file written into a run's output_dir recording WHICH config owns the
#: checkpoints there. Auto-resume refuses foreign markers (see resolve_resume_path).
RUN_MARKER_FILENAME = ".toaster_run.json"


def run_fingerprint(cfg: Dict[str, Any]) -> str:
    """Stable fingerprint of the identity-defining parts of a train config.

    Two runs sharing an output_dir may auto-resume from each other ONLY when
    these fields agree — model, dataset files, epochs/steps, lr and LoRA rank
    (shape), and the base_adapter edge. Data-adjacent knobs (save cadence,
    batch size) are deliberately excluded: `train resume` legitimately re-runs
    with them unchanged, and a user changing e.g. batch mid-crash still gets a
    usable checkpoint chain. Pure function, no torch."""
    import hashlib

    tr = cfg.get("training") or {}
    data_sec = cfg.get("data") or {}
    lora = cfg.get("lora") or {}
    parts = {
        "model": str(cfg.get("model")),
        "files": sorted(str(f) for f in (data_sec.get("local_dataset") or [])),
        "lr": repr(tr.get("learning_rate", 2e-4)),
        "epochs": repr(tr.get("num_epochs", 3)),
        "max_steps": int(tr.get("max_steps", 0) or 0),
        "r": int(lora.get("lora_r", 64)),
        "base_adapter": str(tr.get("base_adapter") or ""),
    }
    blob = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def write_run_marker(outputs_root, fingerprint: str) -> None:
    """Best-effort: record which config owns the checkpoints in this dir."""
    try:
        root = Path(str(outputs_root)).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        (root / RUN_MARKER_FILENAME).write_text(
            json.dumps({"fingerprint": fingerprint,
                        "written_at": datetime.now(timezone.utc).isoformat(
                            timespec="seconds")}),
            encoding="utf-8")
    except OSError:
        pass


def run_marker_matches(outputs_root, fingerprint: str) -> bool:
    """True when no marker exists (legacy run — old behavior) or it matches.

    An unreadable marker counts as MATCHING only if we would otherwise refuse
    a same-process crash-resume; the conservative choice is to treat a corrupt
    marker as foreign? No — corrupt marker = unknown owner; refusing resume
    costs a restart, resuming foreign weights costs silent corruption. So:
    missing => match (legacy), unreadable/foreign => no match."""
    mf = Path(str(outputs_root)).expanduser() / RUN_MARKER_FILENAME
    if not mf.is_file():
        return True
    try:
        return str(json.loads(mf.read_text()).get("fingerprint")) == fingerprint
    except (OSError, ValueError):
        return False


def resolve_base_adapter(cfg: Dict[str, Any],
                         cli_override: Optional[str] = None) -> Optional[str]:
    """Continual training: resolve the existing adapter to continue from.

    Sources (first wins): the ``--base-adapter`` CLI flag (threaded as
    ``cfg["_base_adapter"]`` by main()) or ``training.base_adapter`` in the
    train.yaml. The value must be a directory holding a real training artifact:
    a LoRA adapter (adapter_config.json) or a full fine-tune checkpoint
    (training_type.json marker). Returns None when unset — every run without
    the knob is byte-identical to pre-fix behavior."""
    raw = str(cli_override
              or ((cfg.get("training") or {}).get("base_adapter")) or "").strip()
    if not raw:
        return None
    p = Path(raw).expanduser()
    if not p.is_dir():
        raise ValueError(
            f"base_adapter={str(p)!r} is not a directory — point it at the "
            "adapter dir of a previous training run (train_runs/<run>/adapter)")
    has_lora = (p / "adapter_config.json").is_file()
    has_full = (p / "training_type.json").is_file()
    if not (has_lora or has_full):
        raise ValueError(
            f"{p} is not a training artifact (no adapter_config.json / "
            "training_type.json) — cannot continue training from it")
    return str(p)


def load_model_hardened(model_ref: Any, tr: Dict[str, Any],
                        log_path: Optional[str], kind: str):
    """DPO/GRPO base-model load with the SFT path's hardening (deep-dive fix).

    The preference/RL paths used to load with a bare torch_dtype="auto": no
    4-bit tiering, no ROCm/bnb probe, no pre-quantized guard. A 27B DPO run
    then OOM'd or segfaulted with none of the honest diagnostics SFT earned
    the hard way. Same policy as run_sft (core.vram), compact form."""
    import torch
    from transformers import AutoModelForCausalLM

    _explicit = tr.get("load_in_4bit")
    if _explicit is not None and not isinstance(_explicit, bool):
        _explicit = bool(_explicit)
    _dims = None
    _gpu = bool(torch.cuda.is_available())
    try:
        from ..core.vram import load_model_dims, resolve_load_in_4bit
        _dims = load_model_dims(str(model_ref))
    except Exception:
        pass
    li4, why = resolve_load_in_4bit(
        explicit=_explicit, dims=_dims, train_full=False, gpu_available=_gpu)
    if li4 and torch.version.hip:
        try:
            from ..core.bnbprobe import rocm_bnb_supported
            _ok, _bnb_why = rocm_bnb_supported()
            if not _ok:
                li4 = False
                why = f"ROCm bnb probe failed, loading bf16: {_bnb_why}"
        except Exception as e:  # probe must never kill the run
            _emit(log_path, "warning", message=f"bnb probe unavailable: {e}")
    _emit(log_path, "load_tier", load_in_4bit=li4, reason=why,
          kind=kind)

    load_kwargs: Dict[str, Any] = {}
    if _gpu:
        load_kwargs["torch_dtype"] = "auto"
    # Pre-quantized checkpoints are serve-only for adapters (same guard as SFT).
    _prequantized = False
    try:
        _cfgj = Path(str(model_ref)).expanduser() / "config.json"
        if _cfgj.is_file():
            _prequantized = "quantization_config" in json.loads(
                _cfgj.read_text())
    except (OSError, ValueError):
        pass
    if _prequantized:
        raise RuntimeError(
            f"{model_ref} is a pre-quantized checkpoint (config.json "
            "quantization_config) — serve-only for LoRA training. Pick the "
            "bf16 HF base instead (the Models panel resolve step finds it).")
    if li4 and _gpu:
        try:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        except ImportError:
            _emit(log_path, "warning",
                  message="bitsandbytes missing — loading at native precision")
    return AutoModelForCausalLM.from_pretrained(str(model_ref), **load_kwargs)


# ------------------------------------------------------------------- config

def load_train_config(path: str, base_model_override: Optional[str] = None) -> Dict[str, Any]:
    """Load a train.yaml (documented config contract) and resolve placeholders.

    The curater emits `model: REPLACE_WITH_BASE_MODEL_PATH_OR_HF_ID`; the real
    base model comes from toaster_loop config `training.base_model` (HF id or local
    non-GGUF path — GGUFs are serve-only for training). The field names of the
    YAML contract stay compatible with the documented train-config contract (model/data/
    training/lora/logging) purely as a file format; nothing is imported.
    """
    data = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    model = base_model_override or data.get("model")
    if not model or model == PLACEHOLDER:
        raise ValueError(
            f"{path}: model is unresolved ({model!r}). Set training.base_model in "
            "the toaster_loop config (HF id or local HF-format path; a GGUF cannot be "
            "a training base)."
        )
    if str(model).lower().endswith(".gguf"):
        raise ValueError(
            f"training.base_model={model!r} is a GGUF — serve-only. Point it at "
            "the HF checkpoint (id or directory) used to build that GGUF."
        )
    data["model"] = model
    return data


def lora_kwargs_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Pure train.yaml -> peft.LoraConfig kwargs mapping (backlog item #13).

    Torch-free by design so unit tests can drive it without the GPU stack.
    Mirrors the train.yaml contract's LoRA knobs (field list read from
    contract mirrored from the documented file format; nothing imported):
    `lora_r`/`lora_alpha`/`lora_dropout`, comma-string or list
    `target_modules`, and the `use_rslora`/`use_dora` adapter variants
    (accepted under `lora:` or `training:`; they are direct LoraConfig args).
    The caller adds `task_type`/`bias` (peft enums live behind lazy imports).
    """
    lora = cfg.get("lora") or {}
    tr = cfg.get("training") or {}

    target_modules = lora.get("target_modules") or [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ]
    if isinstance(target_modules, str):
        target_modules = [t.strip() for t in target_modules.split(",") if t.strip()]

    return {
        "r": int(lora.get("lora_r", 64)),
        "lora_alpha": int(lora.get("lora_alpha", 16)),
        "lora_dropout": float(lora.get("lora_dropout", 0.0)),
        "target_modules": target_modules,
        # Backlog #13 pass-throughs (defaults match peft: both off).
        "use_rslora": bool(lora.get("use_rslora", tr.get("use_rslora", False))),
        "use_dora": bool(lora.get("use_dora", tr.get("use_dora", False))),
    }


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{i}: invalid JSON: {e}") from e
    return rows


# ------------------------------------------------- sharegpt -> chat template

# ShareGPT 'from' roles -> OpenAI roles (the mapping our curater emits).
_ROLE_MAP = {
    "human": "user",
    "gpt": "assistant",
    "system": "system",
    "tool": "tool",
    "function_call": "assistant",
}


def sharegpt_to_messages(conversations: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Convert one ShareGPT row ({"conversations": [{from,value}]}) to messages."""
    messages: List[Dict[str, str]] = []
    for turn in conversations or []:
        role = _ROLE_MAP.get(str(turn.get("from", "")).lower())
        if role is None:
            continue  # unknown role — skip the turn
        messages.append({"role": role, "content": turn.get("value") or ""})
    return messages


def render_conversation(tokenizer: Any, conversations: List[Dict[str, str]]) -> str:
    """Render a ShareGPT row to training text via the tokenizer's chat template.

    Falls back to plain role-prefixed concatenation when the tokenizer has no
    chat template (older/base checkpoints).
    """
    messages = sharegpt_to_messages(conversations)
    if not messages:
        return ""
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        return "".join(f"{m['role'].capitalize()}: {m['content']}\n" for m in messages) \
            + tokenizer.eos_token


# ------------------------------------------- response-only loss masking

def assistant_char_regions(tokenizer: Any,
                           conversations: List[Dict[str, str]]) -> Optional[List[tuple]]:
    """Character regions of each assistant turn inside the rendered text.

    Built by rendering growing message prefixes with the tokenizer's own chat
    template: the k-th assistant response spans
    [len(template(msgs[:k], gen_prompt=True)), len(template(msgs[:k+1]))).
    Returns None when templates don't compose (some exotic templates inject
    state) — caller then trains unmasked instead of guessing.
    """
    messages = sharegpt_to_messages(conversations)
    assistant_idx = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    if not assistant_idx:
        return None
    try:
        full = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)
        regions: List[tuple] = []
        for i in assistant_idx:
            prefix = tokenizer.apply_chat_template(
                messages[:i], tokenize=False, add_generation_prompt=True)
            upto = tokenizer.apply_chat_template(
                messages[: i + 1], tokenize=False, add_generation_prompt=False)
            if not full.startswith(prefix) or not full.startswith(upto):
                return None
            start, end = len(prefix), len(upto)
            if end > start:
                regions.append((start, end))
        return regions or None
    except Exception:
        return None


def tokenize_with_labels(tokenizer: Any, conversations: List[Dict[str, str]],
                         max_length: int, mask_instructions: bool
                         ) -> Optional[Dict[str, Any]]:
    """One ShareGPT row -> {input_ids, attention_mask, labels}.

    With mask_instructions=True every token outside the assistant regions gets
    label -100 (ignored by the causal-LM loss) — the built-in equivalent of
    the train_on_responses_only pattern, computed from the chat template rather
    than hardcoded marker strings. Requires a fast tokenizer for exact
    char->token offsets; otherwise returns None and the caller keeps the
    unmasked text path.
    """
    text = render_conversation(tokenizer, conversations)
    if not text.strip():
        return None
    enc = tokenizer(
        text, truncation=True, max_length=max_length,
        return_offsets_mapping=mask_instructions, add_special_tokens=False,
    ) if mask_instructions else tokenizer(
        text, truncation=True, max_length=max_length, add_special_tokens=False,
    )
    input_ids = enc["input_ids"]
    if not input_ids:
        return None
    labels = [tid if tid != tokenizer.pad_token_id else -100 for tid in input_ids] \
        if tokenizer.pad_token_id is not None else list(input_ids)
    if mask_instructions:
        regions = assistant_char_regions(tokenizer, conversations)
        if regions is None:
            return None  # template doesn't compose — caller falls back unmasked
        offsets = enc.get("offset_mapping") or []
        for tok_i, (a, b) in enumerate(offsets):
            if tok_i >= len(labels):
                break
            if not any(a < end and b > start for start, end in regions):
                labels[tok_i] = -100
        # never let a fully-masked row through (loss would be NaN)
        if all(l == -100 for l in labels):
            return None
    return {
        "input_ids": input_ids,
        "attention_mask": enc.get("attention_mask", [1] * len(input_ids)),
        "labels": labels,
    }


# ---------------------------------------------------------------------- SFT

def _combined_optimizer_cls():
    """Build the _CombinedOptimizer class over the real torch base (this
    module imports torch lazily; the class body needs it at definition
    time, so it is constructed on first use and cached)."""
    import torch as _t

    class _CombinedOptimizer(_t.optim.Optimizer):
        """Present N optimizers as one to the Trainer (one step/zero_grad/
        state_dict for all). Param groups are concatenated; the Trainer
        only reads param_groups on the paths we use."""

        def __init__(self, optimizers):
            self.optimizers = list(optimizers)
            params = [p for o in self.optimizers
                      for g in o.param_groups for p in g["params"]]
            _t.optim.Optimizer.__init__(self, iter(params), {})
            self.param_groups = [g for o in self.optimizers
                                 for g in o.param_groups]

        def zero_grad(self, set_to_none: bool = True):  # noqa: D102
            for o in self.optimizers:
                o.zero_grad(set_to_none=set_to_none)

        def step(self, closure=None):  # noqa: D102
            loss = None
            if closure is not None:
                with _t.enable_grad():
                    loss = closure()
            with _t.no_grad():
                for o in self.optimizers:
                    o.step()
            return loss

        def state_dict(self):  # noqa: D102
            return {"optimizers": [o.state_dict()
                                   for o in self.optimizers]}

        def load_state_dict(self, state_dict):  # noqa: D102
            for o, sd in zip(self.optimizers, state_dict["optimizers"]):
                o.load_state_dict(sd)

    return _CombinedOptimizer


def muon_optimizer_factory(*, lr: float, weight_decay: float):
    """Build an HF optimizer FACTORY for Muon + AdamW (opt-in: optim: muon).

    transformers >= ~4.57 calls a factory with the (moved, possibly
    wrapped) model and kwargs at create_optimizer() time — after device
    placement, so no pre-instantiated optimizer state can go stale. We
    split params here: ndim==2 gets torch.optim.Muon (Newton-Schulz
    orthogonalized momentum), everything else (norms, biases, embeddings,
    lm_head) gets AdamW — the published recipe (Moonlight; KellerJordan/
    Muon README). Muon's own defaults (lr 1e-3, wd 0.1, momentum 0.95)
    deliberately do NOT inherit the AdamW-tuned config: the run's lr/wd
    drive the AdamW half; Muon trains at its published defaults.

    Returns a closure — the closure object itself is not an Optimizer
    subclass, which is exactly what HF's is_optimizer_factory() detects.
    """
    def factory(model, *, betas=None, eps=None, **ignored):
        import torch as _t
        from torch.optim import AdamW

        muon_params, adam_params = [], []
        for p in model.parameters():
            if not p.requires_grad:
                continue
            (muon_params if p.ndim == 2 else adam_params).append(p)
        if not muon_params:
            raise RuntimeError(
                "optim=muon but the model exposes no 2D trainable "
                "parameters — Muon has nothing to orthogonalize")
        opt_muon = _t.optim.Muon(muon_params, lr=1e-3, weight_decay=0.1,
                                 momentum=0.95)
        opt_adam = AdamW(adam_params, lr=lr, weight_decay=weight_decay,
                         betas=tuple(betas) if betas else (0.9, 0.999),
                         eps=eps or 1e-8)
        return _combined_optimizer_cls()([opt_muon, opt_adam])
    return factory


def run_sft(cfg: Dict[str, Any], log_path: Optional[str], dry_run: bool) -> Dict[str, Any]:
    """LoRA SFT via transformers + peft. Returns {adapter_dir, ...}.

    Backend selection (WP10 step 2, opt-in): ``training.backend`` selects
    the implementation. Default (absent/invalid -> "peft") runs exactly the
    pre-WP10 path below, byte-identical. ``streamed`` routes to the
    streamed-LoRA runner (layer-streaming meta shell + PEFT-free LoRA +
    recompute-backprop) — a separate code path with the same job contract:
    it also returns {adapter_dir, train_loss, eval_loss} plus a terminal
    *_done event on the progress log.
    """
    backend = parse_backend((cfg.get("training") or {}).get("backend"),
                            warn=lambda m: _emit(log_path, "warning",
                                                 message=m))
    if backend == "streamed":
        return run_streamed_sft(cfg, log_path, dry_run)

    data_sec = cfg.get("data") or {}
    files: List[str] = list(data_sec.get("local_dataset") or [])
    fmt = data_sec.get("format_type", "sharegpt")
    if not files:
        raise ValueError("data.local_dataset is empty — nothing to train on")
    for f in files:
        if not Path(f).is_file():
            raise FileNotFoundError(f"dataset file missing: {f}")

    tr = cfg.get("training") or {}
    lora = cfg.get("lora") or {}
    out_dir = Path(tr.get("output_dir") or "outputs").expanduser()
    adapter_dir = out_dir / "adapter"
    # training_type: "lora" (default) | "full" — full fine-tune of the base
    # weights. Opt-in; every other behavior in this function is unchanged
    # for lora. Full mode: no peft wrap, adapter_dir holds a complete HF
    # checkpoint (+ marker file), export skips the merge step.
    training_type = str(tr.get("training_type", "lora")).lower().strip()
    if training_type not in ("lora", "full"):
        raise ValueError(
            f"training.training_type must be 'lora' or 'full', got {training_type!r}")
    train_full = training_type == "full"
    # Continual training (deep-dive fix): continue from an EXISTING adapter
    # instead of a fresh one on the base. `--base-adapter <dir>` (job param,
    # surfaced as cfg["_base_adapter"] in main) or training.base_adapter in
    # the train.yaml. Unset => byte-identical pre-fix behavior.
    base_adapter = resolve_base_adapter(cfg,
                                        cli_override=cfg.get("_base_adapter"))
    if base_adapter and train_full:
        raise ValueError(
            "base_adapter is a LoRA continuation — for training_type=full "
            "point training.base_model at the previous run's merged/adapter "
            "checkpoint directory instead.")

    # Dataset validation happens without GPU deps (json + line count).
    n_rows = sum(len(load_jsonl(Path(f))) for f in files)
    if n_rows == 0:
        raise ValueError("dataset(s) contain no rows")

    resolved = {
        "model": cfg["model"],
        "files": files,
        "format_type": fmt,
        "rows": n_rows,
        "output_dir": str(out_dir),
        "adapter_dir": str(adapter_dir),
        "training_type": training_type,
        "epochs": tr.get("num_epochs", 3),
        "lr": tr.get("learning_rate", 2e-4),
        "lora_r": lora.get("lora_r", 64),
        **({"base_adapter": base_adapter} if base_adapter else {}),
    }
    if dry_run:
        _emit(log_path, "dry_run_ok", config=resolved)
        return {"adapter_dir": str(adapter_dir), "dry_run": True, **resolved}

    # --- GPU path: standard open-source training stack (lazy imports). ------
    try:
        import torch
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForCausalLM,
            AutoTokenizer,
        )
        from peft import (  # type: ignore[import-not-found]
            LoraConfig,
            PeftModel,
            get_peft_model,
            TaskType,
        )
    except ImportError as e:  # pragma: no cover - env dependent
        raise RuntimeError(
            f"the training stack (torch/transformers/peft) is not importable "
            f"in this environment: {e}. Run the worker with the GPU-enabled "
            "interpreter (config training.python), e.g. "
            "pip install torch transformers peft datasets accelerate."
        ) from None

    from datasets import load_dataset  # type: ignore[import-not-found]

    _emit(log_path, "phase", phase="load_model", model=cfg["model"])
    max_seq = int(tr.get("max_seq_length") or 16384)
    # 4-bit tiering (auto unless pinned): explicit load_in_4bit wins;
    # otherwise >=3B params -> nf4 QLoRA base, smaller -> bf16, full-FT /
    # no-GPU -> never 4-bit. Same policy the VRAM gate uses (core.vram).
    from ..core.vram import resolve_load_in_4bit
    _explicit_4bit = tr.get("load_in_4bit")
    if _explicit_4bit is not None and not isinstance(_explicit_4bit, bool):
        _explicit_4bit = bool(_explicit_4bit)
    _tier_dims = None
    _tier_gpu = bool(torch.cuda.is_available())
    try:
        from ..core.vram import load_model_dims
        _tier_dims = load_model_dims(str(cfg["model"]))
    except Exception:
        pass
    load_in_4bit, _tier_why = resolve_load_in_4bit(
        explicit=_explicit_4bit, dims=_tier_dims, train_full=train_full,
        gpu_available=_tier_gpu)
    # Deep-dive fix #5: attestation records INTENT. The ROCm/bnb guards below
    # flip load_in_4bit to False before the provenance event emits — an
    # attested run would then claim 4-bit was never requested when the user
    # did request it. Snapshot the resolved intent at decision time.
    _requested_4bit = bool(load_in_4bit)
    _emit(log_path, "load_tier", load_in_4bit=load_in_4bit, reason=_tier_why)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Device attestation: name the device training will actually use, and
    # shout when the installed torch wheel can never see this machine's GPU
    # (a CUDA build on a ROCm box imports fine and crawls on CPU — the
    # silent-CPU-training incident; see core/torchenv.py).
    try:
        _gpu = torch.cuda.is_available()
        _bad = None
        try:
            from ..core import torchenv
            _bad = torchenv.installed_mismatch(torch, torchenv.detect_flavor())
        except Exception:
            pass
        _dev = (f"cuda:{torch.cuda.current_device()} "
                f"({torch.cuda.get_device_name(torch.cuda.current_device())})"
                if _gpu else "cpu")
        _emit(log_path, "device_info", device=_dev,
              torch=torch.__version__, hip=(torch.version.hip or None),
              cuda=(torch.version.cuda or None))
        if _bad:
            _emit(log_path, "warning", message=_bad)
        elif not _gpu:
            _emit(log_path, "warning", message=(
                f"training will run on CPU (torch {torch.__version__}, "
                "cuda.is_available()=False) — no accelerator visible"))
    except Exception as e:  # attestation must never kill a run
        _emit(log_path, "warning", message=f"device attestation failed: {e}")

    load_kwargs: Dict[str, Any] = {}
    if torch.cuda.is_available():
        load_kwargs["torch_dtype"] = "auto"
    if train_full and torch.cuda.is_available():
        # Full fine-tune: pin bf16 weights (matches the measured 5 GiB
        # full-FT footprint of Qwen3-0.6B; "auto" on an fp32 checkpoint
        # would train fp32 and 4x the optimizer memory).
        load_kwargs["torch_dtype"] = torch.bfloat16
    # Pre-quantized checkpoints (NVFP4/FP8/int4 compressed-tensors releases)
    # carry their own quantization_config in config.json. Stacking a
    # BitsAndBytesConfig on top makes transformers merge two quantizers and
    # die (or silently requantize); the right move is to load as-is — it is
    # already smaller than nf4 would make it.
    _prequantized = False
    try:
        import json as _json
        _cfgj = Path(str(cfg["model"])).expanduser() / "config.json"
        if _cfgj.is_file():
            _prequantized = "quantization_config" in _json.loads(
                _cfgj.read_text())
    except (OSError, ValueError):
        pass

    # bitsandbytes on ROCm used to be a blanket NO: pre-7.2 ROCm torch wheels
    # bundle an HSA/rocprofiler stack that SIGSEGVs at the FIRST kernel launch
    # on gfx1151 (pytorch/pytorch#173125), and bitsandbytes died with it — so
    # this guard was correct then. It became the bug: with a working wheel nf4
    # QLoRA trains fine on the same hardware, yet every AMD run was priced as
    # bf16-only (a 27B model => ~136 GiB, queue-blocked forever on a 120 GiB
    # unified pool that QLoRA's ~101 GiB fits). Probe the ACTUAL capability
    # instead — in a throwaway child, because a genuine SIGSEGV must never
    # take this worker down (core/bnbprobe.py). TOASTER_LOOP_BNB_ROCM still
    # forces either way.
    # (Pre-quantized checkpoints never take the runtime-nf4 path below, so the
    # probe would be pointless GPU work there.)
    if load_in_4bit and torch.version.hip and not _prequantized:
        from ..core.bnbprobe import rocm_bnb_supported

        _bnb_ok, _bnb_why = rocm_bnb_supported()
        if not _bnb_ok:
            load_in_4bit = False
            _emit(log_path, "warning", message=(
                f"ROCm torch ({torch.version.hip}) + bitsandbytes nf4 probe "
                f"failed — loading bf16 instead (needs ~2x the RAM of nf4; "
                f"lower max_seq_length if it OOMs). Probe: {_bnb_why}. "
                "A newer ROCm-built torch wheel usually fixes this "
                "(+rocm7.2 or later); set TOASTER_LOOP_BNB_ROCM=1 to force "
                "nf4 anyway when you know better than the probe."))
        else:
            _emit(log_path, "bnb_probe", ok=True, detail=_bnb_why)

    if load_in_4bit and torch.cuda.is_available() and not _prequantized:
        try:
            from transformers import BitsAndBytesConfig  # type: ignore[import-not-found]

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        except ImportError:
            _emit(log_path, "warning",
                  message="bitsandbytes missing — loading in bf16/fp16 instead of 4-bit")
    elif load_in_4bit and _prequantized:
        _emit(log_path, "warning", message=(
            "checkpoint is pre-quantized (config.json quantization_config)"
            " — loading as-is; runtime nf4 skipped"))
    elif load_in_4bit:
        _emit(log_path, "warning",
              message="no CUDA device — ignoring load_in_4bit, loading at native precision")

    # GPU sanity preflight (Strix Halo rc=-11 saga): on unified-memory AMD
    # boxes, torch/ROCm SIGSEGVs on the FIRST allocation when the GTT pool is
    # already saturated by other processes (verified: even a 4x4 tensor dies
    # while a 250k-context llama-server holds ~76 GiB). A segfault here would
    # kill the job with no traceback, so probe in a THROWAWAY CHILD and fail
    # the job honestly with who is holding memory instead.
    if torch.cuda.is_available() and not os.environ.get(
            "TOASTER_LOOP_SKIP_GPU_PROBE"):
        _probe = subprocess.run(
            [sys.executable, "-c",
             "import torch; t=torch.empty(256*1024*1024, dtype=torch.uint8,"
             " device='cuda'); print('ok', t.numel())"],
            capture_output=True, text=True, timeout=300)
        if _probe.returncode != 0:
            _diag = []
            try:
                from ..core import gpumem
                _pool = gpumem.unified_pool()
                if _pool is not None:
                    # gtt_used_mib is Optional (some vendors don't expose it)
                    _diag.append(
                        f"GPU pool: {_pool.free_mib // 1024} GiB free of "
                        f"{_pool.total_mib // 1024} GiB (gtt_used="
                        f"{(_pool.gtt_used_mib or 0) // 1024} GiB)")
            except Exception:
                pass
            try:
                _f = subprocess.run(["fuser", "-v", "/dev/dri/renderD128"],
                                    capture_output=True, text=True, timeout=10)
                _holders = " ".join(sorted(
                    {l.split()[-1] for l in _f.stdout.splitlines()[1:]
                     if len(l.split()) >= 6}))
                if _holders:
                    _diag.append(f"GPU device held by: {_holders}")
            except Exception:
                pass
            _tail = (_probe.stderr or "").strip().splitlines()[-3:]
            raise RuntimeError(
                "GPU preflight failed: torch cannot allocate even 256 MiB on "
                "cuda (child exited rc=" + str(_probe.returncode) + "). On "
                "unified-memory AMD systems this means the GPU memory pool is"
                " exhausted by OTHER processes — close the model server(s)"
                " holding it (llama-server / llama.cpp / any third-party"
                " training or inference app / Toaster Loop serving) and"
                " retry. " + " | ".join(_diag)
                + ((" | child: " + " ".join(_tail)) if _tail else ""))

    if _prequantized and not train_full:
        # Pre-quantized serving checkpoints (FP8/NVFP4 compressed-tensors)
        # are not trainable adapters-in-place: peft dies with "'Linear' object
        # has no attribute 'weight'" on the quantized modules, and where they
        # do load, transformers dequantizes to bf16 (27B -> ~50 GiB) and the
        # run OOM-kills. Fail fast with the fix instead of a segfault.
        raise RuntimeError(
            f"{cfg['model']} is a pre-quantized checkpoint (config.json"
            " quantization_config, e.g. FP8/NVFP4). Those are serve-only:"
            " LoRA cannot be injected into quantized linear layers and the"
            " dequantized weights would OOM most machines. Pick the bf16"
            " HF base instead (the Models panel's resolve step finds it) —"
            " training auto-applies 4-bit QLoRA for large models anyway.")

    model = AutoModelForCausalLM.from_pretrained(cfg["model"], **load_kwargs)
    try:
        model.config.use_cache = False  # required with gradient checkpointing
    except Exception:
        pass

    # Provenance attestation (pass 3 #23): record the load mode we ACTUALLY
    # used — load_in_4bit silently degrades without CUDA or bitsandbytes, so
    # "the config said 4-bit" is not the truth; this event is. The pipeline
    # mirrors it onto the registry row (core/provenance.py).
    from ..core.provenance import (
        LOAD_MODE_PREQUANTIZED,
        LOAD_MODE_RUNTIME_4BIT,
        LOAD_MODE_UNQUANTIZED,
        worker_provenance_event,
    )
    if "quantization_config" in load_kwargs:
        _load_mode = LOAD_MODE_RUNTIME_4BIT
    elif any(bool(getattr(model, f"is_loaded_in_{q}", False))
             for q in ("4bit", "8bit")) or getattr(model, "_hf_quantizer", None):
        # transformers exposes is_loaded_in_4bit/8bit as plain bool properties.
        _load_mode = LOAD_MODE_PREQUANTIZED
    else:
        _load_mode = LOAD_MODE_UNQUANTIZED
    # _emit's signature is (log_path, event, **fields); the provenance dict
    # carries its own "type" key (parse_worker_provenance matches on it), so
    # pass the event name positionally and splat the payload as fields.
    _prov = worker_provenance_event(
        _load_mode, str(cfg["model"]),
        extra={"requested_4bit": _requested_4bit})
    _emit(log_path, str(_prov.get("type") or "resource_provenance"), **_prov)

    lora_kwargs = lora_kwargs_from_config(cfg)  # backlog #13 (incl. rslora/DoRA)

    seed = int(tr.get("random_seed", 3407))
    import transformers as _tf  # type: ignore[import-not-found]
    _tf.set_seed(seed)

    # Inspiration pass 4 #34: adaptive LoRA rank growth (RankScaler). Disabled
    # unless `training.rank_scaling` is set. When enabled, training runs as a
    # stage loop: each logged loss feeds the scaler; on trigger the stage stops
    # at its checkpoint and the next stage wraps a 2×r adapter seeded from the
    # previous one (zero-padded growth preserves the learned delta exactly).
    scaler = rank_scaling_from_config(cfg)

    def _wrap_adapter(base: Any, r: int) -> Any:
        """Fresh peft wrapper at rank `r` + gradient checkpointing (#34 loop)."""
        kwargs = dict(lora_kwargs)
        kwargs["r"] = int(r)
        wrapped = get_peft_model(
            base,
            LoraConfig(**kwargs, task_type=TaskType.CAUSAL_LM, bias="none"),
        )
        # Gradient checkpointing + LoRA need inputs retained through the
        # adapter. Backlog #13: `gradient_checkpointing` bool honored here;
        # the train.yaml contract's compression-mode variants have no trivially
        # equivalent TrainingArguments usage and are intentionally not exposed
        # (documented).
        if tr.get("gradient_checkpointing", True):
            try:
                wrapped.enable_input_require_grads()
                wrapped.gradient_checkpointing_enable()
            except Exception as e:
                _emit(log_path, "warning",
                      message=f"gradient checkpointing disabled: {e}")
        return wrapped

    current_r = int(lora_kwargs["r"])
    if train_full:
        # Full fine-tune: no adapter wrap; all base params train. Gradient
        # checkpointing stays on (consumer-memory default).
        if scaler.enabled:
            _emit(log_path, "warning", message=(
                "rank_scaling is a LoRA feature — ignored for "
                "training_type=full"))
        model.train()
        if tr.get("gradient_checkpointing", True):
            try:
                model.gradient_checkpointing_enable()
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
            except Exception as e:
                _emit(log_path, "warning",
                      message=f"gradient checkpointing disabled: {e}")
    else:
        if base_adapter:
            # Continual training: load the previous adapter onto this base and
            # keep its LoRA trainable (no fresh random init). peft's
            # from_pretrained(is_trainable=True) makes the loaded adapter the
            # active, trainable "default" — subsequent saves overwrite the
            # same adapter shape, so vN+1 stays a drop-in for vN at serve time.
            _emit(log_path, "phase", phase="load_base_adapter",
                  base_adapter=base_adapter)
            try:
                # Continuation adapter becomes the trainable policy. The
                # rebinding through a local keeps mypy quiet (from_pretrained
                # returns PeftModel; `model` was inferred as the HF class) —
                # runtime-wise this is exactly what every peft path expects.
                _continued = PeftModel.from_pretrained(
                    model, base_adapter, is_trainable=True)
                model = _continued  # type: ignore[assignment]
            except Exception as e:
                raise RuntimeError(
                    f"could not continue-train from adapter {base_adapter} on "
                    f"base {cfg['model']}: {e}. A LoRA adapter must be loaded "
                    "onto the SAME base model it was trained on — check that "
                    "training.base_model matches the previous run's.") from None
            if tr.get("gradient_checkpointing", True):
                try:
                    model.enable_input_require_grads()
                    model.gradient_checkpointing_enable()
                except Exception as e:
                    _emit(log_path, "warning",
                          message=f"gradient checkpointing disabled: {e}")
            # Rank honesty: the loaded adapter's rank wins over the yaml's
            # lora_r (shapes are fixed by what is on disk).
            try:
                # getattr keeps mypy off peft's Module-typed attribute graph.
                loaded_r = int(getattr(
                    getattr(model, "active_peft_config", None), "r", current_r))
                if loaded_r != current_r:
                    _emit(log_path, "warning", message=(
                        f"lora.lora_r={current_r} ignored — continuing from "
                        f"adapter with r={loaded_r} (shapes are fixed by the "
                        "checkpoint on disk)"))
                current_r = loaded_r
            except Exception:
                pass
        else:
            model = _wrap_adapter(model, current_r)
        if scaler.enabled:
            if base_adapter:
                _emit(log_path, "warning", message=(
                    "rank_scaling is disabled for base_adapter continuation "
                    "runs (stage-chained growth assumes a fresh adapter)"))
            else:
                scaler.start_stage(current_r)
                _emit(log_path, "rank_scaling_enabled", r=current_r,
                      max_rank=scaler.max_rank,
                      min_improvement_steps=scaler.min_improvement_steps,
                      min_rel_improvement=scaler.min_rel_improvement)
    _emit(log_path, "phase", phase="load_dataset", files=files, format_type=fmt)
    if fmt != "sharegpt":
        raise ValueError(f"worker only supports format_type=sharegpt, got {fmt!r}")
    ds = load_dataset("json", data_files={"train": files}, split="train")

    # Response-only loss masking (config: training.train_on_completions).
    # Tokens outside assistant turns get label -100 so the LoRA only learns
    # responses, not instruction tokens. Requires a fast tokenizer; rows or
    # templates that can't be masked exactly fall back to unmasked text.
    # Deep-dive fix #9: the old guard (`mask and not is_fast`) was dead code —
    # mask already requires is_fast. Warn when masking was REQUESTED but the
    # slow tokenizer forced it off, so those users see why.
    _want_mask = bool(tr.get("train_on_completions", True))
    mask = _want_mask and tokenizer.is_fast
    if _want_mask and not tokenizer.is_fast:
        _emit(log_path, "warning",
              message="slow tokenizer — train_on_completions disabled "
                      "(training on all tokens)")

    def _format(ex: Dict[str, Any]) -> Dict[str, Any]:
        convs = ex.get("conversations") or []
        if mask:
            enc = tokenize_with_labels(tokenizer, convs, max_seq, True)
            if enc is not None:
                return enc
        return {"text": render_conversation(tokenizer, convs)}

    ds = ds.map(
        _format, remove_columns=ds.column_names, batched=False,
        desc="rendering sharegpt -> " + ("masked labels" if mask else "chat template"),
    )
    # Rows that failed exact masking come back as {"text": ...}; keep them
    # only when masking is off (with it on, drop the unmaskable minority —
    # training some rows on instructions and others on responses is worse
    # than uniformly skipping a few rows).
    if mask:
        ds = ds.filter(lambda ex: "labels" in ex)
        masked = True
    else:
        ds = ds.filter(lambda ex: bool(ex.get("text", "").strip()))
        masked = False
    if mask and len(ds) == 0:
        raise ValueError(
            "response-only masking produced no usable rows — the tokenizer's "
            "chat template does not compose over message prefixes. Set "
            "training.train_on_completions=false to train on all tokens.")

    # Eval split. Priority (inspiration pass 3 #22): a NAMED split the dataset
    # already carries (eval.jsonl next to the training files, or an explicit
    # data.eval_dataset in train.yaml) is used verbatim — author intent beats
    # our floors, which degrade to warnings there. Otherwise the deterministic
    # fractional split from backlog #4 with its min-rows floors: tiny datasets
    # keep the old no-eval behavior entirely.
    eval_fraction = float(tr.get("eval_fraction", 0.1))
    eval_seed = int(tr.get("eval_seed", 42))
    named_eval_files = find_named_eval_files(data_sec, files)
    eval_ds = None
    train_ds = ds
    if named_eval_files:
        ev_raw = load_dataset("json", data_files={"train": named_eval_files},
                              split="train")
        ev_raw = ev_raw.map(_format, remove_columns=ev_raw.column_names,
                            batched=False, desc="rendering named eval split")
        eval_ds = (ev_raw.filter(lambda ex: "labels" in ex) if mask
                   else ev_raw.filter(lambda ex: bool(ex.get("text", "").strip())))
        _emit(log_path, "named_eval_split", files=named_eval_files,
              eval_rows=len(eval_ds))
        if len(eval_ds) < EVAL_MIN_ROWS:
            _emit(log_path, "warning",
                  message=f"named eval split has {len(eval_ds)} row(s) "
                          f"(< {EVAL_MIN_ROWS}); eval-loss noise will be high")
    else:
        tr_idx, ev_idx = maybe_eval_split(list(range(len(ds))), eval_fraction,
                                          eval_seed)
        if ev_idx is not None:
            train_ds = ds.select(tr_idx)
            eval_ds = ds.select(ev_idx)
            _emit(log_path, "eval_split", train_rows=len(tr_idx),
                  eval_rows=len(ev_idx), fraction=eval_fraction, seed=eval_seed)

    # Checkpoints + eval cadence (backlog item #4). save_strategy="no" is
    # replaced with step/epoch cadence under the job's output_dir; this also
    # powers checkpoint resume (pass 3 #21: resolve_resume_path below feeds
    # trainer.train(resume_from_checkpoint=...); `toaster-loop train resume` re-queues
    # an interrupted job against these checkpoints).
    cadence = plan_cadence(
        len(train_ds),
        batch_size=int(tr.get("batch_size", 2)),
        grad_accum=int(tr.get("gradient_accumulation_steps", 4)),
        epochs=float(tr.get("num_epochs", 3)),
        max_steps=int(tr.get("max_steps", 0) or 0),
        save_steps=int(tr.get("save_steps", 0) or 0),
    )
    # Deep-dive fix: checkpoint rotation. save_strategy=steps every N with no
    # limit accumulated multi-GB checkpoints forever (the janitor only archives
    # registry artifact paths). Keep the newest few — training.save_total_limit
    # (default 3; 0 = keep all, old behavior).
    _stl = int(tr.get("save_total_limit", 3) or 0)
    args_kwargs: Dict[str, Any] = dict(
        output_dir=str(out_dir),
        num_train_epochs=float(tr.get("num_epochs", 3)),
        learning_rate=float(tr.get("learning_rate", 2e-4)),
        per_device_train_batch_size=int(tr.get("batch_size", 2)),
        gradient_accumulation_steps=int(tr.get("gradient_accumulation_steps", 4)),
        warmup_steps=int(tr.get("warmup_steps", 5)),
        weight_decay=float(tr.get("weight_decay", 0.01)),
        report_to=[],
        **({"save_total_limit": _stl} if _stl > 0 else {}),
        **cadence,
    )
    # plan_cadence mirrors save cadence into eval cadence, but Trainer hard-
    # errors (transformers >= 5) on eval_strategy != "no" without an
    # eval_dataset — small/short runs get no eval split at all.
    if eval_ds is None:
        args_kwargs["eval_strategy"] = "no"
        args_kwargs.pop("eval_steps", None)
    # Speed/VRAM knobs (opt-in; reference-style wins (no external trainer)):
    #   optim="paged_adamw_8bit" — optimizer states quantized to 8-bit, big
    #     VRAM saving on small-VRAM/unified-memory boxes (needs bitsandbytes).
    #   torch_compile=true — fuse the backward pass (~10-30% speedup; first
    #     step is slow to compile, and some ROCm builds fail — degrade warmly).
    optim = str(tr.get("optim", "") or "").strip()
    # optim: "muon" routes through the HF optimizer-FACTORY hook instead of
    # the OptimizerNames registry (torch.optim.Muon + AdamW for non-2D;
    # see muon_optimizer_factory). Everything else passes through as before.
    muon_factory = None
    if optim == "muon":
        import functools
        muon_factory = functools.partial(
            muon_optimizer_factory,
            lr=float(tr.get("learning_rate", 2e-4)),
            weight_decay=float(tr.get("weight_decay", 0.01)))
        _emit(log_path, "optimizer", name="muon+adamw",
              note="Muon (Newton-Schulz) on 2D params @ lr 1e-3/wd 0.1; "
                   "AdamW on the rest at the run's lr/wd")
    elif optim:
        args_kwargs["optim"] = optim
    if bool(tr.get("torch_compile", False)):
        args_kwargs["torch_compile"] = True
    if eval_ds is not None:
        args_kwargs["per_device_eval_batch_size"] = int(
            tr.get("eval_batch_size", 2))
    if int(tr.get("max_steps", 0) or 0) > 0:
        args_kwargs["max_steps"] = int(tr["max_steps"])

    from transformers import TrainingArguments  # type: ignore[import-not-found]

    def _build_trainer(peft_model: Any, cb: Any):
        """One Trainer over the current adapter (fresh optimizer per stage)."""
        if masked:
            # Pre-tokenized + pre-labeled rows: plain Trainer with a Seq2Seq
            # collator (pads labels with -100, preserves our mask). SFTTrainer's
            # templating path would re-tokenize and lose the labels.
            # Backlog #13: packing is a no-op on this pre-tokenized path — warn
            # and train unpacked rather than silently dropping the knob.
            if bool(tr.get("packing", False)):
                _emit(log_path, "warning",
                      message="packing ignored on the response-masked path "
                              "(pre-tokenized rows are trained unpacked)")
            from transformers import (  # type: ignore[import-not-found]
                DataCollatorForSeq2Seq,
                Trainer,
            )

            opt_kw: Dict[str, Any] = (
                {"optimizer_cls_and_kwargs": (muon_factory, {})}
                if muon_factory else {})
            return Trainer(
                model=peft_model,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                args=TrainingArguments(**args_kwargs),
                data_collator=DataCollatorForSeq2Seq(
                    tokenizer=tokenizer, padding=True, label_pad_token_id=-100),
                callbacks=[cb],
                **opt_kw,
            )
        # transformers 5.x moved SFTTrainer to trl (it was never in 4.x
        # either — the old `from transformers import SFTTrainer` form dates
        # from the trl<0.18 re-export era). Import from trl, keeping the
        # transformers try-first for any env with an ancient combined install.
        try:
            from transformers import SFTTrainer  # type: ignore[attr-defined]
        except ImportError:
            from trl import SFTTrainer  # type: ignore[assignment]

        # Backlog #13: `packing` (default off). Only the unmasked SFTTrainer
        # path can pack; older/newer APIs differ, so a rejected kwarg degrades
        # to a warning + unpacked training instead of failing the job.
        packing = bool(tr.get("packing", False))
        trainer_kwargs: Dict[str, Any] = {}
        if packing:
            trainer_kwargs["packing"] = True
        if muon_factory:
            trainer_kwargs["optimizer_cls_and_kwargs"] = (muon_factory, {})
        try:
            return SFTTrainer(
                model=peft_model,
                processing_class=tokenizer,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                args=TrainingArguments(**args_kwargs),
                max_length=max_seq,
                truncation=True,
                callbacks=[cb],
                **trainer_kwargs,
            )
        except TypeError as e:
            if not packing:
                raise
            _emit(log_path, "warning",
                  message=f"packing unsupported by this SFTTrainer API "
                          f"({e}) — training unpacked")
            fallback_kwargs = dict(trainer_kwargs)
            fallback_kwargs.pop("packing", None)
            return SFTTrainer(
                model=peft_model,
                processing_class=tokenizer,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                args=TrainingArguments(**args_kwargs),
                max_length=max_seq,
                truncation=True,
                callbacks=[cb],
                **fallback_kwargs,
            )

    # Checkpoint resume (inspiration pass 3 #21): a cancel/crash/reboot left
    # resumable checkpoints in the run dir -> continue from the newest one
    # instead of burning the whole GPU window again. `training.resume` opts
    # out (false) or pins an explicit path; default auto-discovery. Growth
    # stages (#34) never resume trainer state — adapter shapes changed; the
    # seeded weights carry the previous stage forward instead.
    first_resume = resolve_resume_path(
        tr, out_dir, log_path, fingerprint=run_fingerprint(cfg))
    if first_resume:
        _emit(log_path, "phase", phase="resume", checkpoint=first_resume)
    # Record which config owns this dir's checkpoints (deep-dive fix): the
    # NEXT run into this dir auto-resumes only when its fingerprint matches.
    write_run_marker(out_dir, run_fingerprint(cfg))

    # Stage loop (pass 4 #34): one pass per rank; exits when a stage finishes
    # without the RankScaler triggering (or growth is no longer possible).
    train_loss = None
    callback = None
    resume_from = first_resume
    while True:
        callback = make_protocol_callback(
            log_path, scaler if scaler.enabled else None)
        # Emit BEFORE the (heavy) Trainer build: Trainer construction itself
        # is a known native-crash site, and this event distinguishes "died
        # building the trainer" from "died in the first steps".
        _emit(log_path, "phase", phase="build_trainer", rank=current_r)
        trainer = _build_trainer(model, callback)
        _emit(log_path, "phase", phase="train", rank=current_r)
        result = trainer.train(resume_from_checkpoint=resume_from)
        resume_from = None  # only the first stage may use pass-3 resume

        train_loss_now = None
        try:
            train_loss_now = float(result.training_loss)
        except Exception:
            pass
        if train_loss_now is not None:
            train_loss = train_loss_now

        next_r = scaler.plan_growth() if scaler.triggered else None
        if not scaler.enabled or not scaler.triggered or next_r is None:
            break

        # Grow: add a second peft adapter at 2×r (supported multi-adapter
        # path — no re-wrapping), seed it zero-padded from the trained one
        # (composed delta B@A preserved exactly), and make it active. The old
        # adapter stays frozen and is skipped on save (save_pretrained writes
        # the active adapter).
        scaler.record_growth(next_r, int(getattr(
            trainer.state, "global_step", 0)))
        _emit(log_path, "rank_scaling_grow", **scaler.rank_history[-1])
        del trainer
        old_names = sorted({k.split(".lora_A.")[1].split(".")[0]
                            for k in model.state_dict()
                            if ".lora_A." in k}) or ["default"]
        old_name = old_names[0]
        new_name = f"rank{next_r}"
        grown_ok = True
        try:
            kwargs = dict(lora_kwargs)
            kwargs["r"] = int(next_r)
            # peft's PeftModel.add_adapter(name, config) — mypy resolves the
            # inherited transformers PreTrainedModel signature (config, name)
            # here and flags arg 2; runtime dispatch is peft's.
            model.add_adapter(
                new_name,
                LoraConfig(**kwargs, task_type=TaskType.CAUSAL_LM, bias="none"))  # type: ignore[arg-type]
            model.set_adapter(new_name)
            seeded = grow_lora_state(model, model.state_dict(),
                                     old_name=old_name, new_name=new_name)
        except Exception as e:  # peft API surprise: honest degradation
            grown_ok = False
            _emit(log_path, "warning",
                  message=f"rank growth to r={next_r} failed ({e}); keeping "
                          "current rank and finishing training normally")
        if not grown_ok:
            break
        scaler.start_stage(next_r)
        current_r = next_r
        _emit(log_path, "rank_scaling_seeded", to_r=next_r,
              seeded_tensors=seeded[0], of=seeded[1])

    # Deep-dive fix: a mid-run NaN abort is a FAILED run — do not save (and
    # let the pipeline register) an adapter derived from diverged weights.
    if callback is not None and callback.nan_abort_step is not None:
        raise RuntimeError(
            f"training diverged: NaN loss at step {callback.nan_abort_step} "
            "(run aborted early to save the GPU window). Checkpoints are kept "
            f"in {out_dir} for inspection; retry with a lower learning rate.")
    _emit(log_path, "phase", phase="save_adapter", adapter_dir=str(adapter_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    if train_full:
        # Full fine-tune: adapter_dir carries a COMPLETE HF checkpoint plus a
        # marker; export skips the peft merge (see run_export).
        (adapter_dir / "training_type.json").write_text(json.dumps(
            {"training_type": "full", "model": str(cfg["model"])}))
    try:
        tokenizer.save_pretrained(str(adapter_dir))
    except Exception:
        pass

    eval_loss = callback.last_eval_loss if callback else None  # (#4)
    _emit(log_path, "sft_done", adapter_dir=str(adapter_dir),
          train_loss=train_loss, eval_loss=eval_loss,
          training_type=("full" if train_full else "lora"),
          **({"base_adapter": base_adapter} if base_adapter else {}),
          **({"rank_history": scaler.rank_history}
             if scaler.enabled and scaler.rank_history else {}))
    out = {"adapter_dir": str(adapter_dir), "train_loss": train_loss,
           "eval_loss": eval_loss}
    if scaler.enabled and scaler.rank_history:
        out["rank_history"] = scaler.rank_history
    return out


# ---------------------------------------------------------------------- DPO

def dpo_row_for_trl(r: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a toaster_loop dpo.jsonl row to what the installed TRL expects.

    The curater emits a MIXED schema — OpenAI role/content `prompt` list with
    plain-string `chosen`/`rejected` completions (curater.py:848). TRL's
    is_conversational() inspects one arbitrary key and then concatenates
    prompt + completion, so the mixed form dies mid-tokenization with
    "can only concatenate list (not str) to list" on trl>=0.19-era builds
    (caught by scripts/ci_train_smoke.py). Conversational rows therefore get
    their completions wrapped as assistant messages; fully plain rows pass
    through untouched. Pure function — torch-free, unit-tested."""
    prompt, chosen, rejected = r["prompt"], r["chosen"], r["rejected"]
    if isinstance(prompt, list) and isinstance(chosen, str):
        chosen = [{"role": "assistant", "content": chosen}]
        rejected = ([{"role": "assistant", "content": rejected}]
                    if isinstance(rejected, str) else rejected)
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def run_dpo(cfg: Dict[str, Any], log_path: Optional[str], dry_run: bool) -> Dict[str, Any]:
    """TRL DPOTrainer over dpo.jsonl (schema {prompt[], chosen, rejected}).

    Deep-dive fix: this path now shares the SFT hardening bar — 4-bit tiering
    + ROCm/bnb probe + pre-quantized guard via load_model_hardened, seed
    setting, checkpoint resume and rotation, mid-run NaN abort, and continual
    training from an existing adapter (training.base_adapter / --base-adapter:
    the loaded LoRA becomes the trainable policy, and TRL's disabled-adapter
    reference trick still works because it compares against the same base).
    """
    dataset_dir = Path(cfg.get("_dataset_dir") or ".").expanduser()
    dpo_file = dataset_dir / "dpo.jsonl"
    if not dpo_file.is_file():
        raise FileNotFoundError(f"DPO dataset not found: {dpo_file}")
    rows = load_jsonl(dpo_file)
    for i, r in enumerate(rows):
        if not (isinstance(r.get("prompt"), list) and r.get("chosen") and r.get("rejected")):
            raise ValueError(
                f"{dpo_file}:{i}: expected {{'prompt': [...], 'chosen': str, "
                "'rejected': str}} (TRL preference schema)"
            )
    tr = cfg.get("training") or {}
    if str(tr.get("training_type", "lora")).lower().strip() == "full":
        raise ValueError(
            "training_type: full is supported for SFT only — DPO/GRPO still "
            "train LoRA adapters (TRL accepts non-PEFT models, but toaster_loop's "
            "preference pipelines are adapter-shaped; see worker run_dpo).")
    out_dir = Path(tr.get("output_dir") or "outputs").expanduser()
    adapter_dir = out_dir / "adapter"
    base_adapter = resolve_base_adapter(cfg,
                                        cli_override=cfg.get("_base_adapter"))

    if dry_run:
        _emit(log_path, "dry_run_ok", kind="dpo", rows=len(rows),
              dpo_file=str(dpo_file))
        return {"adapter_dir": str(adapter_dir), "dry_run": True, "rows": len(rows)}

    try:
        import torch  # noqa: F401 — needed by load_model_hardened's tier logic
        from datasets import Dataset  # type: ignore[import-not-found]
        from peft import LoraConfig, PeftModel, TaskType  # type: ignore[import-not-found]
        from transformers import AutoTokenizer  # type: ignore[import-not-found]
        from trl import DPOConfig, DPOTrainer  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover - env dependent
        raise RuntimeError(
            f"the DPO stack (torch/transformers/peft/trl) is not importable "
            f"here: {e} — install it in the training environment "
            "(pip install torch transformers peft trl datasets accelerate)."
        ) from None

    seed = int(tr.get("random_seed", 3407))
    import transformers as _tf  # type: ignore[import-not-found]
    _tf.set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model_hardened(cfg["model"], tr, log_path, "dpo")
    # Backlog #13: same pure kwargs builder as SFT (rslora/DoRA/string
    # target_modules included). DPO has no packing knob in TRL's DPOConfig.
    peft_cfg: Optional[LoraConfig] = LoraConfig(
        **lora_kwargs_from_config(cfg),
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    if base_adapter:
        # Continual DPO: the previous adapter becomes the policy; TRL keeps a
        # fresh LoraConfig only for shape reference, so hand it None and pass
        # the PeftModel directly (ref_model=None still uses the disabled
        # adapter as reference).
        _emit(log_path, "phase", phase="load_base_adapter",
              base_adapter=base_adapter)
        try:
            model = PeftModel.from_pretrained(model, base_adapter,
                                              is_trainable=True)
        except Exception as e:
            raise RuntimeError(
                f"could not continue-train (DPO) from adapter {base_adapter} "
                f"on base {cfg['model']}: {e}. The adapter must match the "
                "same base model.") from None
        peft_cfg = None

    ds = Dataset.from_list([dpo_row_for_trl(r) for r in rows])
    # Backlog item #4: same progress protocol + checkpoints as SFT. Eval split
    # is intentionally skipped here — DPO eval loss on pairwise preference
    # data with only a couple of held-out pairs is noise rather than signal
    # (and TRL's eval loop needs the reference model too), so DPO gets
    # step-loss logging + checkpoints and no eval gate input.
    cadence = plan_cadence(
        len(rows),
        batch_size=int(tr.get("batch_size", 2)),
        grad_accum=int(tr.get("gradient_accumulation_steps", 4)),
        epochs=float(tr.get("num_epochs", 3)),
        max_steps=int(tr.get("max_steps", 0) or 0),
        save_steps=int(tr.get("save_steps", 0) or 0),
    )
    callback = make_protocol_callback(log_path)
    # plan_cadence mirrors save cadence into eval cadence, but DPO runs no
    # eval loop (no eval_dataset — see note above), and transformers>=5
    # hard-fails the mismatch at construction. Strip the eval keys.
    cadence.pop("eval_strategy", None)
    cadence.pop("eval_steps", None)
    # Deep-dive fix: checkpoint rotation for DPO too (same knob as SFT).
    _stl_dpo = int(tr.get("save_total_limit", 3) or 0)
    if _stl_dpo > 0:
        cadence.setdefault("save_total_limit", _stl_dpo)
    dpo_kwargs: Dict[str, Any] = {}
    optim = str(tr.get("optim", "") or "").strip()
    if optim == "muon":
        raise ValueError(
            "optim: muon is wired for the SFT path only (the factory hook "
            "lives in Trainer; TRL's DPO/GRPO configs would need the same "
            "treatment — see worker.muon_optimizer_factory).")
    if optim:
        dpo_kwargs["optim"] = optim
    if bool(tr.get("torch_compile", False)):
        dpo_kwargs["torch_compile"] = True
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # PEFT: the disabled adapter acts as the reference model
        peft_config=peft_cfg,
        args=DPOConfig(
            output_dir=str(out_dir),
            # transformers>=5 defaults eval_strategy to "epoch" and hard-fails
            # a Trainer constructed without eval_dataset; DPO deliberately runs
            # no eval loop (see comment above), so pin it off explicitly.
            eval_strategy="no",
            num_train_epochs=float(tr.get("num_epochs", 3)),
            # Deep-dive fix #10: the old hardcoded `/ 10` (DPO trains best an
            # order of magnitude below SFT) is now the documented default of
            # training.dpo_lr_scale — set to 1.0 to use the run LR verbatim.
            learning_rate=float(tr.get("learning_rate", 2e-4))
            * float(tr.get("dpo_lr_scale", 0.1)),
            per_device_train_batch_size=int(tr.get("batch_size", 2)),
            gradient_accumulation_steps=int(tr.get("gradient_accumulation_steps", 4)),
            report_to=[],
            **cadence,
            **dpo_kwargs,
        ),
        train_dataset=ds,
        processing_class=tokenizer,
        callbacks=[callback],
    )
    # Deep-dive fix: DPO now honors training.resume like SFT (auto-discover in
    # the run dir, fingerprint-scoped so a foreign checkpoint never hijacks).
    resume_from = resolve_resume_path(
        tr, out_dir, log_path, fingerprint=run_fingerprint(cfg))
    if resume_from:
        _emit(log_path, "phase", phase="resume", checkpoint=resume_from)
    write_run_marker(out_dir, run_fingerprint(cfg))
    _emit(log_path, "phase", phase="train_dpo", rows=len(rows))
    result = trainer.train(resume_from_checkpoint=resume_from)
    # Deep-dive fix: NaN divergence abort (same contract as SFT).
    if callback.nan_abort_step is not None:
        raise RuntimeError(
            f"DPO diverged: NaN loss at step {callback.nan_abort_step} "
            "(run aborted early); retry with a lower learning rate.")
    out_dir.mkdir(parents=True, exist_ok=True)
    # save_model() routes to the unwrapped (Peft) model's save_pretrained —
    # same adapter-only artifact as trainer.model.save_pretrained().
    trainer.save_model(str(adapter_dir))
    train_loss = None
    try:
        train_loss = float(result.training_loss)
    except Exception:
        pass
    _emit(log_path, "dpo_done", adapter_dir=str(adapter_dir),
          train_loss=train_loss, eval_loss=None,
          **({"base_adapter": base_adapter} if base_adapter else {}))
    return {"adapter_dir": str(adapter_dir), "train_loss": train_loss,
            "eval_loss": None}


# ---------------------------------------------------------------------- grpo

def reward_format(completions, **kwargs) -> List[float]:
    """Rule reward: well-formed completion (non-empty, not degenerate).

    +1.0 for a non-empty answer without runaway repetition; scaled down for
    truncated/empty generations. Pure-python so it runs on any machine.
    """
    out: List[float] = []
    for c in completions:
        text = c if isinstance(c, str) else (
            c[0].get("content", "") if isinstance(c, list) and c else "")
        text = (text or "").strip()
        if not text:
            out.append(0.0)
            continue
        # Degenerate repetition: one token/char repeated over and over.
        low = text.lower()
        uniq = len(set(low.split())) / max(len(low.split()), 1)
        score = 1.0 if (uniq > 0.2 or len(low) < 40) else 0.3
        out.append(score)
    return out


def reward_length(completions, *, max_chars: int = 8000, **kwargs) -> List[float]:
    """Soft length shaping: full credit up to max_chars, decaying beyond."""
    out: List[float] = []
    for c in completions:
        text = c if isinstance(c, str) else (
            c[0].get("content", "") if isinstance(c, list) and c else "")
        n = len(text or "")
        out.append(1.0 if n <= max_chars else max(0.0, 1.0 - (n - max_chars) / max_chars))
    return out


def reward_solution_overlap(completions, solutions=None, **kwargs) -> List[float]:
    """Loose reference reward: word-overlap F1 between completion and the
    dataset's reference `solution` (when present). A cheap stand-in for an
    LLM judge; exact-match style tasks benefit most.
    """
    rows = list(solutions or [])
    out: List[float] = []
    for i, c in enumerate(completions):
        text = c if isinstance(c, str) else (
            c[0].get("content", "") if isinstance(c, list) and c else "")
        ref = rows[i] if i < len(rows) and isinstance(rows[i], str) else ""
        comp_words = set((text or "").lower().split())
        ref_words = set(ref.lower().split())
        if not ref_words or not comp_words:
            out.append(0.0)
            continue
        inter = len(comp_words & ref_words)
        precision = inter / len(comp_words)
        recall = inter / len(ref_words)
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        out.append(f1)
    return out


def grpo_rows_from_dataset(dataset_dir: Path) -> List[Dict[str, Any]]:
    """GRPO needs {prompt: [...], solution: str}. Prefer grpo.jsonl; otherwise
    derive prompts from dpo.jsonl (chosen as reference) or sft.jsonl (last
    assistant turn as reference). Pure-python + testable without torch.
    """
    for name in ("grpo.jsonl", "dpo.jsonl"):
        p = dataset_dir / name
        if p.is_file():
            rows = []
            for r in load_jsonl(p):
                prompt = r.get("prompt")
                if not isinstance(prompt, list) or not prompt:
                    continue
                sol = r.get("solution") or r.get("chosen") or ""
                rows.append({"prompt": prompt, "solution": sol})
            if rows:
                return rows
    sft = dataset_dir / "sft.jsonl"
    if sft.is_file():
        rows = []
        for r in load_jsonl(sft):
            convs = r.get("conversations") or []
            # prompt = everything before the final assistant turn; solution = it.
            *ctx, last = convs if len(convs) > 1 else ([],)
            if not ctx or not last:
                continue
            rows.append({
                "prompt": ctx,
                "solution": str(last.get("value", "")) if isinstance(last, dict) else "",
            })
        return rows
    raise FileNotFoundError(
        f"GRPO needs {dataset_dir}/grpo.jsonl ({{'prompt': [...], 'solution': str}}) "
        "or a dpo.jsonl/sft.jsonl to derive prompts from — none found.")


def grpo_num_generations(effective_batch: int, requested: int) -> int:
    """Divisibility-safe num_generations (deep-dive fix #7).

    TRL requires ``per_device_train_batch_size * gradient_accumulation_steps
    % num_generations == 0``. The old fallback hardcoded 2, which still
    crashes trainer construction for odd effective batches (e.g. 3 or 9).
    Pick the LARGEST divisor of ``effective_batch`` that is >= 2 and <= the
    requested value (falling back to the smallest divisor >= 2 when the
    request is below it); raise a clear ValueError only when NO valid value
    exists (effective_batch < 2), instead of an opaque TRL assertion. Pure
    function — no torch — so it is unit-testable."""
    if effective_batch < 2:
        raise ValueError(
            f"GRPO needs batch_size*gradient_accumulation_steps >= 2 "
            f"(got {effective_batch}) so at least 2 generations can share "
            "a prompt group")
    divisors = [d for d in range(2, effective_batch + 1)
                if effective_batch % d == 0]
    candidates = [d for d in divisors if d <= max(2, int(requested))]
    return max(candidates) if candidates else divisors[0]


def run_grpo(cfg: Dict[str, Any], log_path: Optional[str], dry_run: bool) -> Dict[str, Any]:
    """TRL GRPOTrainer (reinforcement fine-tuning) over prompt(+solution) rows.

    Rewards are built-in rule-based functions (format + length shaping +
    reference overlap); `training.grpo_rewards` narrows the set, e.g.
    ["format", "solution"]. num_generations comes from
    training.grpo_num_generations (default 2 — GRPO needs >=2 and it must
    divide the effective batch; raise for better advantage estimates).
    """
    dataset_dir = Path(cfg.get("_dataset_dir") or ".").expanduser()
    rows = grpo_rows_from_dataset(dataset_dir)
    tr = cfg.get("training") or {}
    if str(tr.get("training_type", "lora")).lower().strip() == "full":
        raise ValueError(
            "training_type: full is supported for SFT only — GRPO still "
            "trains LoRA adapters (full-FT RL on consumer memory risks "
            "policy collapse anyway).")
    out_dir = Path(tr.get("output_dir") or "outputs").expanduser()
    adapter_dir = out_dir / "adapter"
    base_adapter = resolve_base_adapter(cfg,
                                        cli_override=cfg.get("_base_adapter"))

    if dry_run:
        _emit(log_path, "dry_run_ok", kind="grpo", rows=len(rows),
              dataset_dir=str(dataset_dir))
        return {"adapter_dir": str(adapter_dir), "dry_run": True, "rows": len(rows)}

    try:
        import torch  # noqa: F401 — needed by load_model_hardened's tier logic
        from datasets import Dataset  # type: ignore[import-not-found]
        from peft import LoraConfig, PeftModel, TaskType  # type: ignore[import-not-found]
        from transformers import AutoTokenizer  # type: ignore[import-not-found]
        from trl import GRPOConfig, GRPOTrainer  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover - env dependent
        raise RuntimeError(
            f"the GRPO stack (torch/transformers/peft/trl>=0.16) is not "
            f"importable here: {e} — install it in the training environment "
            "(pip install -U trl).") from None

    seed = int(tr.get("random_seed", 3407))
    import transformers as _tf  # type: ignore[import-not-found]
    _tf.set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model_hardened(cfg["model"], tr, log_path, "grpo")
    peft_cfg: Optional[LoraConfig] = LoraConfig(
        **lora_kwargs_from_config(cfg),
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    if base_adapter:
        # Continual RL: previous adapter as the starting policy (fresh LoRA
        # would discard every prior training round).
        _emit(log_path, "phase", phase="load_base_adapter",
              base_adapter=base_adapter)
        try:
            model = PeftModel.from_pretrained(model, base_adapter,
                                              is_trainable=True)
        except Exception as e:
            raise RuntimeError(
                f"could not continue-train (GRPO) from adapter "
                f"{base_adapter} on base {cfg['model']}: {e}. The adapter "
                "must match the same base model.") from None
        peft_cfg = None

    ds = Dataset.from_list(rows)
    enabled = [str(x) for x in (tr.get("grpo_rewards") or ["format", "length", "solution"])]
    rewards: List[Any] = []
    if "format" in enabled:
        rewards.append(reward_format)
    if "length" in enabled:
        rewards.append(reward_length)
    if "solution" in enabled:
        rewards.append(reward_solution_overlap)
    if not rewards:
        rewards = [reward_format]

    num_gen = int(tr.get("grpo_num_generations", 2))
    batch = int(tr.get("batch_size", 2))
    grad_accum = int(tr.get("gradient_accumulation_steps", 4))
    # GRPO requires the generation batch divisible by num_generations.
    # Deep-dive fix #7: the old fallback hardcoded 2, which still fails TRL's
    # divisibility check for odd effective batches — degrade to the largest
    # valid divisor instead (grpo_num_generations, pure function).
    if (batch * grad_accum) % num_gen != 0:
        fixed = grpo_num_generations(batch * grad_accum, num_gen)
        _emit(log_path, "warning",
              message=f"grpo_num_generations={num_gen} does not divide "
                      f"batch_size*grad_accum={batch * grad_accum}; using "
                      f"{fixed}")
        num_gen = fixed

    gen_kw: Dict[str, Any] = {}
    if bool(tr.get("torch_compile", False)):
        gen_kw["torch_compile"] = True
    callback = make_protocol_callback(log_path)
    args = GRPOConfig(
        output_dir=str(out_dir),
        # Same as DPO: no eval loop here; transformers>=5 validates
        # eval_strategy against the absent eval_dataset at construction.
        eval_strategy="no",
        num_train_epochs=float(tr.get("num_epochs", 3)),
        # Same knob as DPO (fix #10): training.dpo_lr_scale, default 0.1 —
        # preference/RL objectives train best an order of magnitude below SFT.
        learning_rate=float(tr.get("learning_rate", 2e-4))
        * float(tr.get("dpo_lr_scale", 0.1)),
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=grad_accum,
        num_generations=num_gen,
        max_completion_length=int(tr.get("max_completion_length", 512)),
        report_to=[],
        **gen_kw,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=rewards,
        args=args,
        train_dataset=ds,
        peft_config=peft_cfg,
        processing_class=tokenizer,
        callbacks=[callback],
    )
    # Deep-dive fix: checkpoint rotation for GRPO (same knob as SFT/DPO).
    _stl_grpo = int(tr.get("save_total_limit", 3) or 0)
    if _stl_grpo > 0:
        args.save_total_limit = _stl_grpo
    # Deep-dive fix: GRPO now honors training.resume like SFT.
    resume_from = resolve_resume_path(
        tr, out_dir, log_path, fingerprint=run_fingerprint(cfg))
    if resume_from:
        _emit(log_path, "phase", phase="resume", checkpoint=resume_from)
    write_run_marker(out_dir, run_fingerprint(cfg))
    _emit(log_path, "phase", phase="train_grpo", rows=len(rows),
          rewards=enabled, num_generations=num_gen)
    result = trainer.train(resume_from_checkpoint=resume_from)
    # Deep-dive fix: NaN divergence abort (same contract as SFT/DPO).
    if callback.nan_abort_step is not None:
        raise RuntimeError(
            f"GRPO diverged: NaN loss at step {callback.nan_abort_step} "
            "(run aborted early); retry with a lower learning rate.")
    out_dir.mkdir(parents=True, exist_ok=True)
    # save_model() routes to the unwrapped (Peft) model's save_pretrained —
    # same adapter-only artifact as trainer.model.save_pretrained().
    trainer.save_model(str(adapter_dir))
    train_loss = None
    try:
        train_loss = float(result.training_loss)
    except Exception:
        pass
    _emit(log_path, "grpo_done", adapter_dir=str(adapter_dir),
          train_loss=train_loss, eval_loss=None,
          **({"base_adapter": base_adapter} if base_adapter else {}))
    return {"adapter_dir": str(adapter_dir), "train_loss": train_loss,
            "eval_loss": None}


# ------------------------------------------------------------------- export

def find_llama_cpp_root() -> Optional[Path]:
    """Locate a llama.cpp checkout with the conversion scripts.

    toaster_loop is standalone: checks $TOASTER_LOOP_LLAMA_CPP first, then its own
    managed build (<TOASTER_LOOP_HOME>/llama.cpp, created by `toaster-loop setup`),
    /opt, and a sibling ../llama.cpp checkout. It does NOT probe foreign
    per-installer directories.
    """
    candidates: List[Path] = []
    env = os.environ.get("TOASTER_LOOP_LLAMA_CPP")
    if env:
        candidates.append(Path(env).expanduser())
    home = os.environ.get("TOASTER_LOOP_HOME")
    managed = (Path(home).expanduser() if home
               else Path.home() / ".toaster-loop") / "llama.cpp"
    candidates += [
        managed,
        Path("/opt/llama.cpp"),
        Path(__file__).resolve().parents[3] / "llama.cpp",
    ]
    for c in candidates:
        if (c / "convert_hf_to_gguf.py").is_file() or (
            c / "conversion" / "__init__.py"
        ).is_file():
            return c
    return None


def merge_adapter(cfg: Dict[str, Any], adapter_dir: Path, merged_dir: Path,
                  log_path: Optional[str]) -> None:
    """Merge a LoRA adapter into its base checkpoint (peft merge_and_unload).

    Deep-dive fix #11: the full-precision CPU load spikes host RAM to ~2 bytes
    per param for the WHOLE base (a 40B merge needs ~80 GiB). Opt-in knobs on
    the training section: ``merge_device_map`` (e.g. "auto" to shard across
    GPU(s)) and ``merge_low_cpu_mem_usage`` (default true — incremental
    loading, roughly halves peak host RAM; requires accelerate)."""
    from peft import PeftModel  # type: ignore[import-not-found]
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]

    tr = cfg.get("training") or {}
    _emit(log_path, "phase", phase="merge_adapter", adapter=str(adapter_dir),
          base=cfg.get("model"))
    load_kwargs: Dict[str, Any] = {
        "torch_dtype": "auto",
        "device_map": tr.get("merge_device_map"),
        "low_cpu_mem_usage": bool(tr.get("merge_low_cpu_mem_usage", True)),
    }
    try:
        base = AutoModelForCausalLM.from_pretrained(cfg["model"], **load_kwargs)
    except (ImportError, ValueError) as e:
        # accelerate missing / device_map unsupported on this build -> retry
        # with the historical plain load rather than fail the export.
        _emit(log_path, "warning", message=(
            f"low-memory merge options unavailable ({e}); falling back to "
            "plain full-precision CPU load"))
        base = AutoModelForCausalLM.from_pretrained(
            cfg["model"], torch_dtype="auto", device_map=None)
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    merged = model.merge_and_unload()
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(merged_dir))
    try:
        AutoTokenizer.from_pretrained(str(adapter_dir)).save_pretrained(str(merged_dir))
    except Exception:
        try:
            AutoTokenizer.from_pretrained(cfg["model"]).save_pretrained(str(merged_dir))
        except Exception as e:
            _emit(log_path, "warning", message=f"tokenizer not saved with merged model: {e}")


def quantize_merged_hf(cfg: Dict[str, Any], merged: Path, out_dir: Path,
                       fmt: str, log_path: Optional[str]) -> Dict[str, Any]:
    """AWQ/GPTQ quantization of a merged HF checkpoint (opt-in extra path).

    Produces an HF-format quantized dir (vLLM/AWQ-Infinity compatible) instead
    of GGUF. `fmt` is "awq" or "gptq"; requires the optional autoawq / auto-gptq
    packages in the torch env — a missing dependency fails with an actionable
    error and keeps the merged artifact for a GGUF retry.
    """
    fmt = fmt.lower().strip()
    if fmt not in ("awq", "gptq"):
        raise ValueError(f"unsupported export_format {fmt!r} (gguf|awq|gptq)")
    tr = cfg.get("training") or {}
    bits = int(tr.get(f"{fmt}_bits", 4))
    quant_out = out_dir / f"quantized-{fmt}"

    if fmt == "awq":
        try:
            from awq import AutoAWQForCausalLM  # type: ignore[import-not-found]
            from transformers import AutoTokenizer  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError(
                f"AWQ export needs the autoawq package ({e}); install it in the "
                "training env (pip install autoawq) or export with "
                "training.export_format=gguf. Merged checkpoint kept at "
                f"{merged}.") from None
        _emit(log_path, "phase", phase="export_awq", merged=str(merged), bits=bits)
        tokenizer = AutoTokenizer.from_pretrained(str(merged), use_fast=True)
        model = AutoAWQForCausalLM.from_pretrained(str(merged))
        model.quantize(
            tokenizer,
            quant_path=str(quant_out),
            calib_dataset=str(tr.get("awq_calib_dataset") or "wikitext"),
            w_bit=bits,
        )
        tokenizer.save_pretrained(str(quant_out))
    else:  # gptq
        try:
            from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig  # type: ignore[import-not-found]
            from transformers import AutoTokenizer  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError(
                f"GPTQ export needs the auto-gptq package ({e}); install it in "
                "the training env (pip install auto-gptq) or export with "
                "training.export_format=gguf. Merged checkpoint kept at "
                f"{merged}.") from None
        _emit(log_path, "phase", phase="export_gptq", merged=str(merged), bits=bits)
        tokenizer = AutoTokenizer.from_pretrained(str(merged), use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        qcfg = BaseQuantizeConfig(
            bits=bits, group_size=int(tr.get("gptq_group_size", 128)),
            desc_act=bool(tr.get("gptq_desc_act", False)))
        model = AutoGPTQForCausalLM.from_pretrained(
            str(merged), qcfg, torch_dtype="auto")
        # Tiny deterministic calibration set; override with training.gptq_calib_texts.
        calib_texts = tr.get("gptq_calib_texts") or [
            "Tell me about the history of computing.",
            "Write a short poem about the sea.",
            "Explain what a neural network is in one sentence.",
        ]
        texts = [str(t) for t in calib_texts]
        samples = [
            {"input_ids": tokenizer.encode(t, add_special_tokens=False),
             "attention_mask": [1] * len(tokenizer.encode(t, add_special_tokens=False))}
            for t in texts if t.strip()
        ] or None
        model.quantize(samples, batch_size=int(tr.get("batch_size", 2)))
        model.save_quantized(str(quant_out))
        tokenizer.save_pretrained(str(quant_out))

    _emit(log_path, "export_done", gguf_path=str(quant_out), method=fmt)
    return {"gguf_path": str(quant_out), "method": fmt}


def run_export(cfg: Dict[str, Any], log_path: Optional[str], dry_run: bool) -> Dict[str, Any]:
    """adapter (+base model) -> GGUF (default) or AWQ/GPTQ HF dir, built-in.

    export_format (training.export_format, default "gguf"):
      gguf  — llama.cpp convert_hf_to_gguf.py + llama-quantize.
      awq / gptq — quantized HF checkpoint via optional autoawq/auto-gptq.

    Strategy (in order):
      1. If a merged HF dir already exists at <output_dir>/merged, use it.
      2. Merge the adapter with peft (needs the torch env — this worker's job
         when run with training.python).
      3. Fail with an actionable error; caller keeps the adapter artifact.
    """
    tr = cfg.get("training") or {}
    out_dir = Path(cfg.get("_export_dir") or tr.get("output_dir") or "outputs").expanduser()
    adapter_dir = Path(cfg.get("_adapter_dir") or (out_dir / "adapter")).expanduser()
    quant = str(cfg.get("_quant") or "q4_k_m")
    export_format = str(tr.get("export_format") or "gguf").lower().strip()
    gguf_out = Path(cfg.get("_gguf_out") or (out_dir / f"{Path(str(out_dir)).name}-{quant}.gguf"))

    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"adapter dir not found: {adapter_dir}")

    llama_root = find_llama_cpp_root()
    plan = {
        "adapter_dir": str(adapter_dir),
        "quant": quant,
        "export_format": export_format,
        "gguf_out": str(gguf_out),
        "llama_cpp_root": str(llama_root) if llama_root else None,
        "model": cfg.get("model"),
    }
    if dry_run:
        _emit(log_path, "dry_run_ok", kind="export", **plan)
        return {"gguf_path": str(gguf_out), "dry_run": True, **plan}

    # WP10 step 3: a streamed-backend adapter carries its own GGUF-LoRA
    # artifact (lora.gguf next to the adapter's parent output dir, produced
    # by the runner via the frozen lora-state-dict contract). The existing
    # export flow just relocates it — no merge, no llama.cpp converter, no
    # quantization (adapter GGUFs are F32 and -lora-loaded onto the base),
    # so this branch runs BEFORE the llama.cpp requirement check.
    streamed_lora = Path(adapter_dir).parent / "lora.gguf"
    if adapter_is_streamed(adapter_dir) and streamed_lora.is_file():
        _emit(log_path, "phase", phase="export_gguf_lora",
              source=str(streamed_lora))
        gguf_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(streamed_lora), str(gguf_out))
        _emit(log_path, "export_done", gguf_path=str(gguf_out),
              method="gguf-lora",
              verified=("parseable-here; llama.cpp -lora load proven here on "
                        "a tiny synthetic ckpt; real-model load untested"))
        return {"gguf_path": str(gguf_out), "method": "gguf-lora"}

    # AWQ/GPTQ quantize from the merged HF dir and don't need llama.cpp at
    # all; only the GGUF path requires it.
    if export_format == "gguf" and llama_root is None:
        raise RuntimeError(
            "GGUF export needs a llama.cpp checkout with convert_hf_to_gguf.py. "
            "Checked $TOASTER_LOOP_LLAMA_CPP, <TOASTER_LOOP_HOME>/llama.cpp, /opt/llama.cpp "
            "and ../llama.cpp. Run `toaster-loop setup` (or point TOASTER_LOOP_LLAMA_CPP at "
            f"it) and re-run the export job. Adapter artifact kept at {adapter_dir}."
        )

    merged = out_dir / "merged"
    if not merged.is_dir():
        if adapter_is_full_finetune(adapter_dir):
            # training_type=full: the "adapter" dir IS a complete HF
            # checkpoint — no merge step exists (or is needed).
            _emit(log_path, "phase", phase="export_full_checkpoint",
                  source=str(adapter_dir))
            merged = adapter_dir
        else:
            # Merge step needs the torch env; only when importable.
            try:
                merge_adapter(cfg, adapter_dir, merged, log_path)
            except ImportError as e:
                raise RuntimeError(
                    "GGUF export needs either a merged HF dir at "
                    f"{merged} or the torch env (peft/transformers) to merge "
                    f"the adapter first ({e}). Run the export job with the "
                    f"GPU/torch interpreter (config training.python). "
                    f"Adapter kept at {adapter_dir}."
                ) from None

    if export_format in ("awq", "gptq"):
        return quantize_merged_hf(cfg, merged, out_dir, export_format, log_path)

    assert llama_root is not None  # GGUF path guaranteed above
    convert = llama_root / "convert_hf_to_gguf.py"
    f16 = gguf_out.with_suffix(".f16.gguf")
    quantize = _find_binary("llama-quantize", llama_root)
    _emit(log_path, "phase", phase="export_gguf_llamacpp", merged=str(merged))
    subprocess.run(
        [sys.executable, str(convert), "--outfile", str(f16), str(merged)],
        check=True,
    )
    if quantize is None or quant.lower() in ("f16", "none"):
        _emit(log_path, "export_done", gguf_path=str(f16), method="llama.cpp-f16")
        return {"gguf_path": str(f16), "method": "llama.cpp"}
    subprocess.run([str(quantize), str(f16), str(gguf_out), quant], check=True)
    # Deep-dive fix #8: the intermediate f16 GGUF is model-size x 4 vs q4 and
    # used to linger in models/gguf/ forever. Unlink it once the quantized
    # artifact exists; training.keep_f16_gguf=true opts back in (e.g. to run a
    # second quant without re-converting).
    if not bool(tr.get("keep_f16_gguf", False)):
        try:
            f16.unlink()
            _emit(log_path, "phase", phase="cleanup_f16", removed=str(f16))
        except OSError as e:
            _emit(log_path, "warning",
                  message=f"could not remove intermediate {f16}: {e}")
    _emit(log_path, "export_done", gguf_path=str(gguf_out), method="llama.cpp")
    return {"gguf_path": str(gguf_out), "method": "llama.cpp"}


def adapter_is_full_finetune(adapter_dir: Path) -> bool:
    """True when a run's artifact dir is a FULL checkpoint (training_type=full),
    written by run_sft with a training_type.json marker."""
    marker = Path(adapter_dir) / "training_type.json"
    if not marker.is_file():
        return False
    try:
        return json.loads(marker.read_text()).get("training_type") == "full"
    except (OSError, ValueError):
        return False


def adapter_is_streamed(adapter_dir: Path) -> bool:
    """True when an adapter dir was written by the streamed backend."""
    cfg_f = Path(adapter_dir) / "adapter_config.json"
    if not cfg_f.is_file():
        return False
    try:
        return json.loads(cfg_f.read_text()).get("backend") == "streamed"
    except (OSError, ValueError):
        return False


def _find_binary(name: str, llama_root: Path) -> Optional[Path]:
    found = shutil.which(name)
    if found:
        return Path(found)
    for cand in (llama_root / name, llama_root / "build" / "bin" / name):
        if cand.exists():
            return cand
    return None


# --------------------------------------------------------------------- main

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="toaster_loop.training.worker")
    ap.add_argument("--job", required=True, help="Job id (registry)")
    ap.add_argument("--kind", required=True,
                    choices=["sft", "dpo", "grpo", "export"])
    # Deep-dive fix #7: --config was argparse-required, but the scheduler only
    # passes flags whose param values exist — a job submitted without
    # train_yaml died with argparse exit 2 and NO JSON error event in the log
    # ("job failed, log empty"). Accept its absence here and fail through the
    # normal error path below so the job log always tells the truth.
    ap.add_argument("--config", default=None, help="train.yaml path")
    ap.add_argument("--dataset-dir", help="Dataset directory (for dpo.jsonl etc.)")
    ap.add_argument("--base-model", help="Resolve REPLACE_WITH_* model placeholder")
    ap.add_argument("--log-path", default=None)
    ap.add_argument("--adapter-dir", help="export: adapter to convert")
    ap.add_argument("--base-adapter",
                    help="continual training: existing adapter dir to "
                         "continue from (sft/dpo/grpo)")
    ap.add_argument("--gguf-out", help="export: destination .gguf path")
    ap.add_argument("--quant", default="q4_k_m")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    # Native-crash forensics: a SIGSEGV (rc=-11) inside torch/ROCm leaves no
    # Python traceback and the job log just stops (observed repeatedly on
    # Strix Halo). faulthandler dumps the C-level-triggering Python stack —
    # but note the scheduler spawns us with stderr merged into DEVNULL, so
    # enabling onto stderr would be swallowed: point it AT THE JOB LOG file.
    # Without this these crashes are undiagnosable. Stdlib, zero cost.
    import faulthandler

    try:
        if args.log_path:
            _fh = open(args.log_path, "a", encoding="utf-8")
            faulthandler.enable(file=_fh)
        else:
            faulthandler.enable()
    except Exception:  # pragma: no cover - exotic envs
        pass

    _emit(args.log_path, "start", job=args.job, kind=args.kind, dry_run=args.dry_run)
    try:
        # ML-stack version window (production-readiness pass 3): every job kind
        # here touches the torch/transformers/peft/trl stack (export merges with
        # peft), so a loaded-but-untested combination fails fast with an
        # actionable message instead of mid-run API breakage. Read from wheel
        # metadata only — no torch import here. Escape hatch:
        # TOASTER_LOOP_ALLOW_UNPINNED_MLSTACK=1 downgrades to a logged warning.
        from ..core import mlstack

        _gate = mlstack.worker_gate()
        if _gate:
            raise RuntimeError(_gate)
        if mlstack.allow_override():
            _ov_problems = mlstack.window_problems(mlstack.installed_versions())
            if _ov_problems:
                _emit(args.log_path, "warning", message=(
                    "ML-stack window violated but "
                    f"{mlstack.ALLOW_OVERRIDE_ENV} is set — training anyway: "
                    + "; ".join(_ov_problems)))

        if not args.config:
            raise ValueError(
                "no --config train.yaml was provided for this job; the submit "
                "path must set params.train_yaml (every dashboard/CLI/pipeline "
                "submit does — a hand-inserted jobs row needs it too)")
        cfg = load_train_config(args.config, base_model_override=args.base_model)
        if args.dataset_dir:
            cfg["_dataset_dir"] = str(Path(args.dataset_dir).expanduser())
        if getattr(args, "base_adapter", None):
            cfg["_base_adapter"] = str(args.base_adapter)
        if args.kind == "sft":
            result = run_sft(cfg, args.log_path, args.dry_run)
        elif args.kind == "dpo":
            result = run_dpo(cfg, args.log_path, args.dry_run)
        elif args.kind == "grpo":
            result = run_grpo(cfg, args.log_path, args.dry_run)
        else:
            if args.adapter_dir:
                cfg["_adapter_dir"] = str(Path(args.adapter_dir).expanduser())
            if args.gguf_out:
                cfg["_gguf_out"] = str(Path(args.gguf_out).expanduser())
            cfg["_quant"] = args.quant
            result = run_export(cfg, args.log_path, args.dry_run)
        _emit(args.log_path, "done", job=args.job, result=result)
        return 0
    except Exception as e:
        import traceback

        _emit(args.log_path, "error", job=args.job, error=str(e),
              traceback=traceback.format_exc(limit=8))
        print(f"worker error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

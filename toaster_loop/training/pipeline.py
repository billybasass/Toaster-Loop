"""`toaster-loop iterate` — one-call loop closure: curate → train → export → validate.

Stage flow (each stage records its job id in the returned report):

    curate   Curater.curate()                       -> dataset_id
    train    scheduler sft|dpo job (worker subproc) -> adapter dir
    export   scheduler export job                   -> GGUF
    validate validation gate (serve + golden tasks) -> pass/fail
    promote  registry.promote on pass; else the model stays failed/rolled_back

Everything GPU-touching happens inside scheduler-spawned subprocesses. For
tests, inject ``worker_command`` (a stub script honoring the worker CLI) and
``server_starter`` (a fake OpenAI server); the whole loop then runs with no
GPU, no torch, no llama.cpp — this is the flagship loop-closure demo path.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml

from ..curate.curater import Curater, CurationConfig
from ..registry.registry import ModelRegistry
from ..store import analytics
from .scheduler import Scheduler, TrainingConfig
from .validate import ServerStarter, ValidationConfig, validate_model

logger = logging.getLogger("toaster_loop.pipeline")

# Top-level sections of the train.yaml contract (the external train-config shape,
# pattern inspiration only — never imported). --config-key overrides may only
# target these; anything else is a clear error (backlog item #13).
TRAIN_YAML_SECTIONS = ("model", "data", "training", "lora", "logging")


def parse_override_value(raw: str) -> Any:
    """Parse one --config-key scalar value (backlog item #13).

    Order: bool true/false (case-insensitive) -> int -> float -> JSON
    (covers lists/dicts like ``a.b=[1,2]`` trivially) -> raw string.
    """
    s = raw.strip()
    low = s.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    if s and s[0] in "[{":
        try:
            return json.loads(s)
        except ValueError:
            pass
    return raw


def apply_config_overrides(
    cfg: Dict[str, Any],
    overrides: List[str],
    allowed_sections: tuple = TRAIN_YAML_SECTIONS,
) -> Dict[str, Any]:
    """Apply dotted-path ``k=v`` overrides onto a config dict in place.

    Mirrors the generated-flag pattern (flatten nested config ->
    ``--lora-r 32``) but as one repeatable ``--config-key k=v`` flag so we
    keep argparse (backlog item #13). Semantics:

      * each override is ``dotted.path=value``; split on the FIRST '=';
      * values typed by parse_override_value (int/float/bool/JSON/str);
      * missing nested dicts are created on the way down;
      * the first path segment must be a top-level section of the config
        (train.yaml contract by default) -> ValueError otherwise;
      * returns {dotted_path: typed_value} for provenance logging.

    Lives in pipeline (shared, torch-free) so CLI, pipeline and tests all use
    the same applier against the resolved train.yaml.
    """
    applied: Dict[str, Any] = {}
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(
                f"invalid override {item!r}: expected dotted.path=value")
        path, raw = item.split("=", 1)
        path = path.strip()
        parts = [p for p in path.split(".") if p]
        if not parts:
            raise ValueError(f"invalid override {item!r}: empty path")
        if parts[0] not in allowed_sections:
            raise ValueError(
                f"unknown config section {parts[0]!r} in override {item!r}; "
                f"allowed: {', '.join(allowed_sections)}")
        value = parse_override_value(raw)
        if len(parts) == 1:  # top-level scalar, e.g. model=Some/Model
            cfg[parts[0]] = value
            applied[parts[0]] = value
            continue
        node = cfg
        for p in parts[:-1]:
            nxt = node.get(p)
            if not isinstance(nxt, dict):
                nxt = {}
                node[p] = nxt
            node = nxt
        node[parts[-1]] = value
        applied[".".join(parts)] = value
    return applied


def log_config_overrides(log_path: Optional[str], job_id: str,
                         applied: Dict[str, Any]) -> None:
    """Append the provenance line for applied overrides to a job log (#13).

    Plain additive JSON per the worker progress protocol — older readers only
    look at start/done/error. Also the CLI test assertion target.
    """
    if not log_path or not applied:
        return
    from datetime import datetime, timezone

    line = json.dumps({
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "event": "config_overrides", "job": job_id, "overrides": applied,
    }, ensure_ascii=False)
    try:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


@dataclass
class PipelineResult:
    dataset_id: Optional[str] = None
    model_id: Optional[str] = None
    train_job_id: Optional[str] = None
    export_job_id: Optional[str] = None
    adapter_dir: Optional[str] = None
    gguf_path: Optional[str] = None
    validation: Optional[Dict[str, Any]] = None
    promoted: bool = False
    stage: str = "curate"          # last completed/failed stage
    ok: bool = False
    error: Optional[str] = None
    stages: List[str] = field(default_factory=list)
    # WP2b: True when the flow *completed* but an artifact failed its
    # integrity check (truncated GGUF, NaN loss). The CLI maps this to the
    # distinct EXIT_CORRUPT_ARTIFACT code instead of generic 1. Deliberately
    # NOT serialized in to_dict() — existing command output stays identical.
    corrupt: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "stage": self.stage,
            "error": self.error,
            "dataset_id": self.dataset_id,
            "model_id": self.model_id,
            "train_job_id": self.train_job_id,
            "export_job_id": self.export_job_id,
            "adapter_dir": self.adapter_dir,
            "gguf_path": self.gguf_path,
            "promoted": self.promoted,
            "validation": self.validation,
            "stages_completed": self.stages,
        }


def _resolved_train_yaml(dataset_dir: Path, run_dir: Path,
                         base_model: Optional[str],
                         eval_fraction: Optional[float] = None,
                         eval_seed: Optional[int] = None,
                         config_overrides: Optional[List[str]] = None,
                         backend: Optional[str] = None,
                         filename: str = "train.resolved.yaml",
                         ) -> Tuple[Path, Dict[str, Any]]:
    """Copy the dataset train.yaml into the run dir with output_dir + model set.

    Keeps per-job artifacts under train_runs/<job>/ regardless of what the
    curater template defaulted to, and resolves the REPLACE_* placeholder from
    training.base_model when configured (the worker validates/raises if still
    unresolved — stub workers simply ignore it). Backlog item #4 additionally
    threads the eval split knobs (training.eval_fraction / eval_seed) so the
    worker splits deterministically without touching every dataset template.

    Backlog item #13: `config_overrides` (--config-key dotted.path=value from
    the CLI) are applied LAST — after output_dir/model/eval injection — so an
    explicit flag always wins. Returns (path, applied_map); the applied map is
    logged into the job log for provenance by the caller.

    Deep-dive fix #1: `filename` lets each *job* resolve into its own file.
    The export stage must never re-write `train.resolved.yaml` — that file is
    the attested record of what produced the adapter (backend, eval knobs,
    overrides). Export resolves to `export.resolved.yaml` instead.
    """
    src = dataset_dir / "train.yaml"
    data = yaml.safe_load(src.read_text()) or {} if src.is_file() else {}
    data.setdefault("training", {})
    data["training"]["output_dir"] = str(run_dir)
    if base_model:
        data["model"] = base_model
    if eval_fraction is not None:
        data["training"]["eval_fraction"] = float(eval_fraction)
    if eval_seed is not None:
        data["training"]["eval_seed"] = int(eval_seed)
    # WP10 step 2 (opt-in): carry the selected backend into the resolved
    # train.yaml so the spawned worker picks the implementation. None keeps
    # the file untouched => default "peft" path, byte-identical to pre-WP10.
    if backend:
        data["training"]["backend"] = str(backend)
    applied = apply_config_overrides(data, list(config_overrides or []))
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / filename
    out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return out, applied


def _dataset_file_inventory(dataset_dir) -> List[Dict[str, Any]]:
    """Hash inventory of the dataset's training files (pass 3 #23).

    Bounded to the jsonl files that actually feed training (sft/dpo/eval);
    sizes + hashes let lineage audits prove *which bytes* a model consumed,
    catching edits made after manifest pinning. Missing files simply yield
    fewer entries — never an error at register time.
    """
    from ..core.provenance import file_entry

    out: List[Dict[str, Any]] = []
    d = Path(str(dataset_dir))
    for fname in ("sft.jsonl", "dpo.jsonl", "eval.jsonl"):
        f = d / fname
        if f.is_file():
            out.append(file_entry(fname, f))
    return out


def parse_worker_losses(log_path: Optional[str]) -> Dict[str, Optional[float]]:
    """Extract {train_loss, eval_loss} from a finished train job log (#4).

    The worker's terminal event carries them ({"event":"sft_done"|"dpo_done"|
    "grpo_done", "train_loss":..., "eval_loss":...}); older/stub workers
    simply lack the fields -> both None. Last occurrence wins.
    """
    out: Dict[str, Optional[float]] = {"train_loss": None, "eval_loss": None}
    if not log_path:
        return out
    try:
        lines = Path(log_path).read_text().splitlines()
    except OSError:
        return out
    for line in lines:
        if ('"sft_done"' not in line and '"dpo_done"' not in line
                and '"grpo_done"' not in line):
            continue
        try:
            rec = json.loads(line)
        except (ValueError, TypeError):
            continue
        for k in out:
            if isinstance(rec.get(k), (int, float)):
                out[k] = float(rec[k])
    return out


def parse_worker_rank_history(log_path: Optional[str]) -> List[Dict[str, Any]]:
    """Extract rank_history from a finished train job log (pass 4 #34).

    The worker emits ``{"event":"sft_done", ..., "rank_history":[{from_r,
    to_r, step, best_loss}]}`` when training.rank_scaling grew the adapter
    mid-run. Absent/garbage => [] (scaling off is the default and the norm)."""
    if not log_path:
        return []
    try:
        lines = Path(log_path).read_text().splitlines()
    except OSError:
        return []
    for line in reversed(lines):
        if '"sft_done"' not in line:
            continue
        try:
            rec = json.loads(line)
        except (ValueError, TypeError):
            continue
        hist = rec.get("rank_history")
        if isinstance(hist, list):
            return [h for h in hist if isinstance(h, dict)]
    return []


def parse_worker_adapter_dir(log_path: Optional[str]) -> Optional[str]:
    """Extract the adapter_dir a finished train job reported (#train-visibility).

    The worker's terminal event carries it ({"event":"sft_done"|"dpo_done"|
    "grpo_done", "adapter_dir":...}). Last occurrence wins; absent/garbage
    returns None (stub workers, crashed runs)."""
    if not log_path:
        return None
    try:
        lines = Path(log_path).read_text(errors="replace").splitlines()
    except OSError:
        return None
    out: Optional[str] = None
    for line in lines:
        if ('"sft_done"' not in line and '"dpo_done"' not in line
                and '"grpo_done"' not in line):
            continue
        try:
            rec = json.loads(line)
        except (ValueError, TypeError):
            continue
        ad = rec.get("adapter_dir")
        if isinstance(ad, str) and ad.strip():
            out = ad.strip()
    return out


def artifact_slug(display_name: Optional[str], model_id: str) -> str:
    """Artifact-directory / GGUF-filename slug for a model row.

    A user-chosen name makes train_runs/<...> and the GGUF readable
    ("<name>-v3-q4_k_m.gguf"); the model_id suffix keeps them unique across
    runs even when names repeat. Unnamed runs are byte-identical to before."""
    import re as _re

    name_slug = _re.sub(r"[^A-Za-z0-9._-]+", "-",
                        str(display_name or "").strip()).strip("-._")[:48]
    return f"{name_slug}-{model_id}" if name_slug else model_id


def finalize_standalone_train(
    *,
    home: Path,
    registry: Any,
    scheduler: Any,
    dataset_id: str,
    log_path: Optional[str],
    base_model: str = "",
    display_name: Optional[str] = None,
    quant: Optional[str] = None,
    actor: str = "dashboard",
    submit_export: bool = True,
    adapter_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Turn a finished standalone train job into a registry model (+ GGUF).

    The Train page and `train start` run bare sft/dpo/grpo jobs without the
    iterate pipeline, so historically nothing created the registry row: the
    adapter existed on disk but was invisible in Models, un-servable and with
    no lineage. This is the missing bookkeeping half of those paths, mirroring
    what run_iterate does after its train stage:

      1. locate the worker's adapter dir (its terminal *_done event),
      2. register a candidate model row (display_name honored),
      3. mirror artifacts + losses + provenance onto the row,
      4. optionally queue AND run the GGUF export job so the model becomes
         servable; an export failure never loses the adapter or the row.

    Never raises for expected problems — they come back in the result dict
    ({ok, model_id, gguf_path, note}) so callers print an honest summary.
    """
    from ..core.provenance import parse_worker_provenance
    from .integrity import find_nan_losses

    home = Path(home)
    out: Dict[str, Any] = {"ok": False, "model_id": None, "gguf_path": None,
                           "note": None}

    if not adapter_dir:
        adapter_dir = parse_worker_adapter_dir(log_path)
    if not adapter_dir or not Path(adapter_dir).is_dir():
        # Worker contract violated (stub workers do this on purpose): nothing
        # to register. Not an error for the caller — just nothing to show.
        out["note"] = "worker reported no adapter dir; model not registered"
        return out

    # Idempotency guard: a row already tracking this exact adapter means the
    # bookkeeping ran before (e.g. CLI finalize + dashboard dispatcher both
    # observed one job) — report it instead of registering a duplicate.
    try:
        for existing in registry.list_models():
            if (existing.adapter_dir
                    and str(existing.adapter_dir) == str(adapter_dir)):
                out["ok"] = True
                out["model_id"] = existing.model_id
                out["gguf_path"] = existing.gguf_path
                out["note"] = (f"{existing.model_id} already tracks this "
                               "adapter; nothing to do")
                return out
    except Exception:  # tolerant: a broken scan must not block registration
        pass

    losses = parse_worker_losses(log_path)
    nan = find_nan_losses(losses)
    if nan:
        out["note"] = ("corrupt training result (NaN in loss history: "
                       + ", ".join(nan) + "); model not registered")
        return out

    dataset_dir = home / "datasets" / str(dataset_id or "")
    manifest_hash = None
    mf = dataset_dir / "manifest.json"
    if mf.is_file():
        import hashlib

        manifest_hash = hashlib.sha256(mf.read_bytes()).hexdigest()[:16]

    # Resolve the base model the same precedence run_iterate uses, so adopted
    # rows stay comparable across paths.
    base = str(base_model or "").strip()
    if not base and mf.is_file():
        try:
            base = str(json.loads(mf.read_text()).get("base_model") or "")
        except (OSError, json.JSONDecodeError):
            base = ""
    if not base:
        base = "unspecified-base"

    model = registry.register_model(
        actor=actor,
        base_model=base,
        dataset_id=str(dataset_id) or None,
        manifest_hash=manifest_hash,
        quant=quant,
        display_name=display_name,
    )
    out["model_id"] = model.model_id
    registry.set_artifacts(model.model_id, adapter_dir=str(adapter_dir))
    if losses["train_loss"] is not None or losses["eval_loss"] is not None:
        registry.set_losses(model.model_id, train_loss=losses["train_loss"],
                            eval_loss=losses["eval_loss"], actor=actor)
    prov_ev = parse_worker_provenance(log_path)
    rank_history = parse_worker_rank_history(log_path)
    backend_ev = parse_worker_backend(log_path)
    if prov_ev or rank_history or backend_ev:
        registry.set_provenance(model.model_id, {
            "version": 1,
            "resource": (prov_ev or {}).get("resource") or "base_model",
            "source": (prov_ev or {}).get("source"),
            "load_mode": (prov_ev or {}).get("load_mode"),
            "requested_4bit": (prov_ev or {}).get("requested_4bit"),
            "dataset_files": _dataset_file_inventory(dataset_dir),
            **({"rank_history": rank_history} if rank_history else {}),
            **backend_ev,
        })

    if not submit_export:
        out["ok"] = True
        out["note"] = f"registered {model.model_id} (adapter only, no export)"
        return out

    # ---- GGUF export so the row is servable ---------------------------------
    slug = artifact_slug(display_name, model.model_id)
    quant_eff = quant or "q4_k_m"
    gguf_dir = home / "models" / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)
    gguf_out = gguf_dir / f"{slug}-{quant_eff}.gguf"
    train_run = home / "train_runs"
    export_yaml, _ = _resolved_train_yaml(
        dataset_dir, train_run / slug, str(base_model or ""),
        filename="export.resolved.yaml")
    try:
        export_job = scheduler.submit(
            "export",
            dataset_id=str(dataset_id) or None,
            model_id=model.model_id,
            log_dir=train_run,
            params={
                "train_yaml": str(export_yaml),
                "adapter_dir": str(adapter_dir),
                "gguf_out": str(gguf_out),
                "quant": quant_eff,
                "base_model": str(base_model or ""),
            },
        )
    except Exception as e:  # registry row + adapter already exist — be honest
        out["ok"] = True
        out["note"] = (f"registered {model.model_id}; export could not be "
                       f"queued: {e}")
        return out
    try:
        finished = scheduler.run_job(export_job)
        state = finished.state
    except Exception as e:
        # Gate busy / slot refusal: the job stays queued and the dispatcher
        # (or a later drain) runs it — GGUF appears when capacity frees.
        out["ok"] = True
        out["note"] = (f"registered {model.model_id}; export job "
                       f"{export_job.job_id} stays queued ({e})")
        return out
    if state == "done" and gguf_out.is_file():
        registry.set_artifacts(model.model_id, gguf_path=str(gguf_out),
                               quant=quant_eff)
        out["gguf_path"] = str(gguf_out)
        out["ok"] = True
        out["note"] = f"registered {model.model_id}, GGUF at {gguf_out}"
    else:
        out["ok"] = True
        out["note"] = (f"registered {model.model_id}; export job "
                       f"{export_job.job_id} {state} — adapter kept, no GGUF")
    return out


def parse_worker_backend(log_path: Optional[str]) -> Dict[str, Any]:
    """Extract {backend, gguf_lora_path} from a finished train job log (WP10).

    The streamed worker emits ``{"event":"streamed_sft_done", "backend":
    "streamed", "gguf_lora_path": ...}``; the pre-WP10 paths emit none of it.
    Absent => {} so the lineage block stays exactly as before for peft runs
    (additive key only when a backend was attested).
    """
    out: Dict[str, Any] = {}
    if not log_path:
        return out
    try:
        lines = Path(log_path).read_text().splitlines()
    except OSError:
        return out
    for line in reversed(lines):
        if '"streamed_sft_done"' not in line:
            continue
        try:
            rec = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(rec.get("backend"), str):
            out["backend"] = rec["backend"]
        if isinstance(rec.get("gguf_lora_path"), str):
            out["gguf_lora_path"] = rec["gguf_lora_path"]
        break
    return out


def eval_regression_check(
    cand_eval: Optional[float], parent_eval: Optional[float], *,
    gate: str = "warn", max_pct: float = 25.0,
) -> Dict[str, Any]:
    """Soft promotion gate on eval_loss regression (backlog item #4).

    Only fires when BOTH losses exist — missing eval data (tiny datasets that
    skipped the split, DPO runs, pre-#4 parents) never blocks anything.
    Returns {blocked, warned, message, regression_pct}. gate="block" refuses
    promotion when the candidate's eval_loss exceeds the parent's by more
    than max_pct percent; the default "warn" only logs (tiny datasets are
    noisy).
    """
    res: Dict[str, Any] = {"blocked": False, "warned": False,
                           "message": None, "regression_pct": None}
    if cand_eval is None or parent_eval is None or parent_eval <= 0:
        return res
    regression_pct = (cand_eval - parent_eval) / parent_eval * 100.0
    res["regression_pct"] = regression_pct
    if regression_pct > float(max_pct):
        res["message"] = (
            f"eval_loss regressed {regression_pct:.1f}% "
            f"(parent {parent_eval:.4f} -> candidate {cand_eval:.4f}; "
            f"limit {max_pct:.0f}% via training.max_eval_regression_pct)"
        )
        if str(gate).lower() == "block":
            res["blocked"] = True
        else:
            res["warned"] = True
    return res


def run_iterate(
    *,
    home: Path,
    registry: ModelRegistry,
    scheduler: Scheduler,
    store: Any,
    curation_cfg: CurationConfig,
    training_cfg: TrainingConfig,
    validation_cfg: ValidationConfig,
    name: Optional[str] = None,
    kind: str = "sft",
    dataset_id: Optional[str] = None,
    dry_run: bool = False,
    config_overrides: Optional[List[str]] = None,
    server_starter: Optional[ServerStarter] = None,
    on_stage: Optional[Callable[[str, str], None]] = None,
    intent: Optional[str] = None,
    auto: bool = False,
    display_name: Optional[str] = None,
) -> PipelineResult:
    """Execute the full loop. Never raises for expected failures — they land
    in the result (stage/error) so the CLI prints a clean report.

    ``auto`` (pass-4 #41): True only on the unattended cron path
    (``train auto-check --trigger``). Automatic promotion then requires the
    derived capability tier >= ``validation.min_tier_auto_deploy``; attended
    runs (manual ``iterate``) stay permissive like manual promote.

    ``display_name``: optional human name for the trained model (UI Train/
    Iterate/Easy forms). It labels the registry row and flavors the artifact
    directory + GGUF filename; model_id stays the canonical key, so two runs
    with the same chosen name never collide.
    """
    home = Path(home)
    res = PipelineResult()

    def stage_done(stage: str) -> None:
        res.stages.append(stage)
        res.stage = stage
        if on_stage:
            on_stage(stage, "done")

    def fail(stage: str, msg: str) -> PipelineResult:
        res.stage = stage
        res.error = msg
        res.ok = False
        if on_stage:
            on_stage(stage, f"FAILED: {msg}")
        return res

    def fail_corrupt(stage: str, problems: List[str]) -> PipelineResult:
        """WP2b: the flow itself succeeded but an artifact is corrupt —
        distinct from an ordinary failure so the CLI exits 4, not 1."""
        res.corrupt = True
        return fail(stage, "corrupt artifact: " + "; ".join(problems))

    # ---- 1. curate ---------------------------------------------------------
    if dataset_id is None:
        if on_stage:
            on_stage("curate", "running")
        curater = Curater(store, home)
        try:
            curation = curater.curate(curation_cfg, name=name or "iterate")
        except Exception as e:
            return fail("curate", f"curation failed: {e}")
        dataset_id = curation.dataset_id
        if on_stage:
            on_stage(
                "curate",
                f"{curation.sft_count} sft rows, {curation.dpo_count} dpo pairs",
            )
    res.dataset_id = dataset_id
    stage_done("curate")

    dataset_dir = home / "datasets" / dataset_id
    if not dataset_dir.is_dir():
        return fail("curate", f"dataset directory missing: {dataset_dir}")

    # Pin the dataset manifest hash (Phase 3 note: models pin their dataset).
    manifest_hash = None
    mf = dataset_dir / "manifest.json"
    if mf.is_file():
        import hashlib

        manifest_hash = hashlib.sha256(mf.read_bytes()).hexdigest()[:16]
    quant = training_cfg.gguf_quant or "q4_k_m"
    manifest_base = None
    if mf.is_file():
        try:
            manifest_base = json.loads(mf.read_text()).get("base_model")
        except (OSError, json.JSONDecodeError):
            manifest_base = None
    base_model = (
        training_cfg.base_model
        or curation_cfg.base_model
        or manifest_base
        or "unspecified-base"
    )

    # ---- 2. register candidate ----------------------------------------------
    if on_stage:
        on_stage("register", "running")
    model = registry.register_model(
        actor="worker",  # pass4 #31: pipeline runs in the worker subprocess
        base_model=str(base_model),
        dataset_id=dataset_id,
        manifest_hash=manifest_hash,
        quant=quant,
        intent=intent,  # pass4 #36: "we trained on D expecting Y"
        display_name=display_name,
    )
    res.model_id = model.model_id
    stage_done("register")

    # Artifact slug: a user-chosen name makes train_runs/<...> and the GGUF
    # readable ("<name>-v3-q4_k_m.gguf"); the model_id suffix keeps them
    # unique across runs even when names repeat. Unnamed runs are byte-identical
    # to before (plain vN).
    import re as _re
    name_slug = _re.sub(r"[^A-Za-z0-9._-]+", "-",
                        str(display_name or "").strip()).strip("-._")[:48]
    artifact_slug = f"{name_slug}-{model.model_id}" if name_slug \
        else model.model_id

    # ---- 3. train -----------------------------------------------------------
    train_run = home / "train_runs"
    # Backlog item #13: --config-key overrides are applied onto the resolved
    # train.yaml here (the single resolution point both `iterate` and CLI
    # runs share) and logged into the job log for provenance.
    train_yaml, applied_overrides = _resolved_train_yaml(
        dataset_dir, train_run / artifact_slug, training_cfg.base_model,
        eval_fraction=training_cfg.eval_fraction,
        eval_seed=training_cfg.eval_seed,
        config_overrides=config_overrides,
        backend=getattr(training_cfg, "backend", None),
    )
    train_job = scheduler.submit(
        kind,
        dataset_id=dataset_id,
        model_id=model.model_id,
        log_dir=train_run,
        params={
            "train_yaml": str(train_yaml),
            "dataset_dir": str(dataset_dir),
            "base_model": training_cfg.base_model or "",
            "dry_run": dry_run,
        },
    )
    log_config_overrides(train_job.log_path, train_job.job_id, applied_overrides)
    res.train_job_id = train_job.job_id
    if on_stage:
        on_stage("train", f"job {train_job.job_id} queued")
    finished = scheduler.run_job(train_job)
    if dry_run:
        # Dry run validated config+dataset; stop here with everything wired.
        res.ok = True
        stage_done("train")
        return res
    if finished.state != "done":
        registry.set_validation(model.model_id, {"error": finished.error or ""},
                                passed=False, actor="worker")
        return fail("train", f"training job {train_job.job_id} {finished.state}: "
                             f"{finished.error}")
    stage_done("train")

    adapter_dir = train_run / artifact_slug / "adapter"
    if not adapter_dir.is_dir():
        # Worker contract: adapters land under the resolved output_dir.
        alt = dataset_dir / "outputs" / "adapter"
        if alt.is_dir():
            adapter_dir = alt
        else:
            return fail("train", f"worker finished but no adapter dir at "
                                 f"{adapter_dir}")
    res.adapter_dir = str(adapter_dir)
    registry.set_artifacts(model.model_id, adapter_dir=str(adapter_dir))

    # Backlog item #4: mirror the worker's final losses onto the model row.
    losses = parse_worker_losses(finished.log_path or train_job.log_path)

    # WP2b: NaN in the recorded loss history means the run "finished" while
    # its numerics were silently corrupt — refuse to continue to export.
    from .integrity import export_integrity_problems, find_nan_losses

    nan_losses = find_nan_losses(losses)
    if nan_losses:
        return fail_corrupt(
            "train", [f"NaN recorded in loss history ({n})"
                      for n in nan_losses])

    if losses["train_loss"] is not None or losses["eval_loss"] is not None:
        # actor="worker": this runs inside the worker subprocess (pass4 #31).
        registry.set_losses(model.model_id, train_loss=losses["train_loss"],
                            eval_loss=losses["eval_loss"], actor="worker")

    # Pass 3 #23: mirror the worker's resource_provenance event (base-model
    # load mode it actually used) onto the model row. Absence is tolerated —
    # older workers emit nothing and the column stays NULL ("unattested").
    from ..core.provenance import parse_worker_provenance

    prov_ev = parse_worker_provenance(finished.log_path or train_job.log_path)
    # Pass 4 #34: adaptive-rank growth history rides beside provenance — the
    # worker emits rank_history on sft_done when training.rank_scaling grew
    # the adapter mid-run ([{from_r, to_r, step, best_loss}]).
    rank_history = parse_worker_rank_history(
        finished.log_path or train_job.log_path)
    # WP10: which training BACKEND produced this adapter (streamed attests
    # itself on streamed_sft_done; peft runs stay unattested => the block is
    # byte-identical to pre-WP10 for them, additive key only when present).
    backend_ev = parse_worker_backend(finished.log_path or train_job.log_path)
    if prov_ev or rank_history or backend_ev:
        registry.set_provenance(model.model_id, {
            "version": 1,
            "resource": (prov_ev or {}).get("resource") or "base_model",
            "source": (prov_ev or {}).get("source"),
            "load_mode": (prov_ev or {}).get("load_mode"),
            "requested_4bit": (prov_ev or {}).get("requested_4bit"),
            # Dataset bytes this row was trained on: the manifest hash is
            # already pinned above; sft.jsonl's own hash catches edits under
            # the same manifest (imported rows re-hash at import time too).
            "dataset_files": _dataset_file_inventory(dataset_dir),
            **({"rank_history": rank_history} if rank_history else {}),
            **backend_ev,
        })

    # ---- 4. export -----------------------------------------------------------
    gguf_dir = home / "models" / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)
    gguf_out = gguf_dir / f"{artifact_slug}-{quant}.gguf"
    export_job = scheduler.submit(
        "export",
        dataset_id=dataset_id,
        model_id=model.model_id,
        log_dir=train_run,
        params={
            # Deep-dive fix #1: resolve into export.resolved.yaml so this
            # re-resolve (which intentionally omits eval/override/backend —
            # export only needs output_dir + model) cannot overwrite the
            # train job's attested train.resolved.yaml in the same dir.
            "train_yaml": str(_resolved_train_yaml(
                dataset_dir, train_run / artifact_slug, training_cfg.base_model,
                filename="export.resolved.yaml",
            )[0]),
            "adapter_dir": str(adapter_dir),
            "gguf_out": str(gguf_out),
            "quant": quant,
            "base_model": training_cfg.base_model or "",
        },
    )
    res.export_job_id = export_job.job_id
    if on_stage:
        on_stage("export", f"job {export_job.job_id} queued")
    finished = scheduler.run_job(export_job)
    if finished.state != "done":
        # Adapter artifact kept; model remains candidate (not failed — the data
        # is fine, only the GGUF conversion could not run).
        return fail("export", f"export job {export_job.job_id} "
                              f"{finished.state}: {finished.error} "
                              f"(adapter kept at {adapter_dir})")
    if not gguf_out.is_file():
        return fail("export", f"export job done but GGUF missing at {gguf_out}")
    # WP2b: the export job exited 0 — verify the artifact it produced is not
    # silently corrupt (zero-byte / truncated / bad-magic GGUF) before
    # promoting anything derived from it. Distinct exit code via res.corrupt.
    problems = export_integrity_problems(gguf_out)
    if problems:
        return fail_corrupt("export", problems)
    res.gguf_path = str(gguf_out)
    registry.set_artifacts(model.model_id, gguf_path=str(gguf_out), quant=quant)
    stage_done("export")

    # ---- 5. validate -----------------------------------------------------------
    if on_stage:
        on_stage("validate", "running")
    report = validate_model(
        str(gguf_out), validation_cfg, server_starter=server_starter,
        model_name=model.model_id,
    )
    res.validation = report
    registry.set_validation(model.model_id, report,
                            passed=bool(report.get("ok")), actor="worker")
    # WP9b: append-only validation-gate telemetry (pass-rate trend per model).
    # Best-effort — a failed analytics write must never change the gate verdict.
    try:
        analytics.record_validation_run(
            home,
            analytics.summarize_validation_report(report, model.model_id,
                                                  source="pipeline"))
    except Exception:  # pragma: no cover - defensive
        logger.debug("could not persist validation telemetry", exc_info=True)
    if not report.get("ok"):
        return fail(
            "validate",
            f"pass_rate {report.get('pass_rate', 0):.2f} < "
            f"{validation_cfg.min_pass_rate:.2f}"
            + (f" ({report.get('error')})" if report.get("error") else ""),
        )
    stage_done("validate")

    # ---- 5b. shadow baseline (pass 4 #32, opt-in) ------------------------------
    # validation.shadow_enabled runs the judge-scored replay against the then-
    # active model BEFORE promoting. It is a normal scheduler job (kind
    # shadow_eval), so VRAM gating + device slots apply; failures here never
    # block promotion — only a recorded regression verdict can (warn-only by
    # default via validation.shadow_blocks_promotion).
    if getattr(validation_cfg, "shadow_enabled", False):
        if on_stage:
            on_stage("shadow", "running")
        try:
            shadow_job = scheduler.submit(
                "shadow_eval",
                dataset_id=dataset_id,
                model_id=model.model_id,
                log_dir=train_run,
                params={
                    "profile": getattr(curation_cfg, "agent_profile", None),
                    "n": getattr(validation_cfg, "shadow_replay_n", 8),
                    "max_regression_pct": getattr(
                        validation_cfg, "max_shadow_regression_pct", 5.0),
                    "judge_model": getattr(
                        validation_cfg, "shadow_judge_model", None),
                },
            )
            finished = scheduler.run_job(shadow_job)
            if finished.state != "done":
                logger.warning(
                    "shadow eval job %s %s: %s (promotion proceeds — "
                    "verdict absent, not evidence of quality)",
                    shadow_job.job_id, finished.state, finished.error)
        except Exception as e:
            logger.warning("shadow eval could not run: %s "
                           "(promotion proceeds)", e)
        stage_done("shadow")

    # ---- 6. promote -------------------------------------------------------------
    if on_stage:
        on_stage("promote", "running")
    # Soft eval gate (backlog item #4): compare this candidate's eval_loss to
    # its parent adapter's recorded eval_loss. Only meaningful when both
    # exist; default mode "warn" just logs, "block" refuses promotion.
    cand = registry.get_model(model.model_id)
    parent_eval = None
    if cand is not None and cand.parent_adapter:
        parent = registry.get_model(cand.parent_adapter)
        parent_eval = parent.eval_loss if parent else None
    gate = eval_regression_check(
        losses["eval_loss"] if cand is None or cand.eval_loss is None
        else cand.eval_loss,
        parent_eval,
        gate=training_cfg.eval_gate,
        max_pct=training_cfg.max_eval_regression_pct,
    )
    if gate["regression_pct"] is not None:
        report = dict(report)
        report["eval_gate"] = {
            "regression_pct": gate["regression_pct"],
            "mode": training_cfg.eval_gate,
            "blocked": gate["blocked"],
        }
        res.validation = report
        registry.set_validation(model.model_id, report, passed=True,
                                actor="worker")
    if gate["warned"]:
        logger.warning("eval gate (warn): %s", gate["message"])
    if gate["blocked"]:
        return fail("promote", f"eval gate blocked promotion: {gate['message']}")
    # Pass 4 #32: ALSO consult the latest shadow-baseline verdict when one
    # exists on the row. Warn-only unless validation.shadow_blocks_promotion;
    # an absent verdict never blocks (honest absence, per build_replay_set).
    from ..evaluation.shadow import shadow_promotion_check

    cand_now = registry.get_model(model.model_id)
    sgate = shadow_promotion_check(
        cand_now.shadow if cand_now else None,
        blocks=bool(getattr(validation_cfg, "shadow_blocks_promotion", False)))
    if sgate["message"]:
        report = dict(report)
        report["shadow_gate"] = {
            "warned": sgate["warned"], "blocked": sgate["blocked"]}
        res.validation = report
        registry.set_validation(model.model_id, report, passed=True,
                                actor="worker")
    if sgate["warned"]:
        logger.warning("shadow gate (warn): %s", sgate["message"])
    if sgate["blocked"]:
        return fail("promote", f"shadow gate blocked promotion: "
                               f"{sgate['message']}")
    # Pass 4 #41 policy hook 1: unattended auto-promotion requires a derived
    # capability tier >= validation.min_tier_auto_deploy (0 disables). Manual
    # promote/deploy stay permissive by design — tiers gate automation, not
    # humans. Same skip shape as the eval/shadow gates above.
    min_tier = int(getattr(validation_cfg, "min_tier_auto_deploy", 2) or 0)
    if auto and min_tier > 0:
        from ..registry.tiers import (compute_served_stats, derive_tier,
                                      requirements_hint)

        row_now = registry.get_model(model.model_id)
        stats = (compute_served_stats(
            store, model.model_id,
            since=(row_now.promoted_at or row_now.created_at)
            if row_now else None)
            if store is not None else {"served_ok": 0, "negative_feedback": 0})
        tier = derive_tier(row_now, stats)
        if tier < min_tier:
            return fail(
                "promote",
                f"auto-promotion skipped: derived capability tier T{tier} < "
                f"T{min_tier} (validation.min_tier_auto_deploy; set 0 to "
                f"disable) — {requirements_hint(row_now, stats)}. Model "
                f"stays validated for manual `toaster-loop models promote "
                f"{model.model_id}`.")
    try:
        # actor="worker": the pipeline runs inside the worker subprocess (#31).
        registry.promote(model.model_id, actor="worker", intent=intent)
    except Exception as e:
        return fail("promote", str(e))
    res.promoted = True
    stage_done("promote")

    res.ok = True
    return res

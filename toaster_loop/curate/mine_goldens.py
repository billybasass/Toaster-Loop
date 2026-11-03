"""Auto-mine verifiable reward functions from 👍 traces into golden tasks.

When a user thumbs-up a trace, that conversation embodies a task the model did
WELL. This module distills such traces into golden validation tasks written in
the EXACT format ``training/validate.py`` already replays (``prompt`` +
``expect_contains`` / ``expect_not_contains`` / ``expect_tool_call``), so every
mined golden permanently joins the regression suite that gates promotions and
feeds the RFT pipeline's golden-task prompt source (``select_prompts``).

Golden store reality (learned before writing code): goldens live in the config
``validation.golden_tasks:`` YAML list — there is no DB table or task directory.
Mined goldens therefore append to an ever-growing JSONL file at
``<home>/goldens/mined.jsonl`` (one task dict per line, ``enabled`` flag +
``_mine`` provenance), and the two ValidationConfig construction points
(``cli._validation_config``, ``dashboard._validation_cfg_from``) merge enabled
mined goldens into ``golden_tasks`` via :func:`merge_mined_into_tasks`. No
file yet -> byte-identical behavior (zero change when nothing was mined).

Verifier derivation rules (STRICT, output-derived):
  * substring presence checks from stable lines of the actual successful
    assistant reply (the turn after the task-defining user message);
  * an ``expect_tool_call`` name check when the recorded assistant message
    contains native tool_calls (same oracle validate.py uses).
NEVER a verifier the source trace's own output would fail: every candidate is
self-checked with ``validate._check_task`` against the recorded (redacted)
assistant message BEFORE persisting; non-self-passing candidates are dropped
and logged. At least ONE deterministic check is required — pure-judge goldens
are refused (promotion gating must stay reproducible). The judge may only
PROPOSE additional substring checks (opt-in ``use_judge``), and every proposal
must itself pass the self-check to be kept.

Secret hygiene: mined tasks embed trace content. Prompts, verifier strings and
the self-check message all pass through ``core/redact`` before anything is
persisted — double redaction is idempotent by construction, so even though
traces are redacted at rest this closes the path for buffers recorded with
``recording.redact_secrets: false``.

Dedup + claim: the ``mined_goldens`` ledger (store/db.py, schema v8) mirrors
the ``rft_prompts`` pattern — INSERT OR IGNORE claim inside BEGIN IMMEDIATE
BEFORE distillation, so re-mining the same trace never duplicates and
concurrent runs each own a trace at most once. Prompt-hash dedup additionally
skips candidates whose prompt collides with an existing manual golden or an
earlier mined one. Dropped candidates stay in the ledger (reason recorded) so
the next run does not retry the same failure forever.

Staging/enabled semantics (documented decision): a golden lands ENABLED only if
it self-checks clean AND carries >= 1 deterministic verifier — which is exactly
the persistence precondition — so everything persisted to mined.jsonl is
enabled; rejected candidates are NOT persisted as goldens at all (they live on
as ``dropped`` ledger rows with reasons). The per-line ``enabled`` flag exists
so operators can disable a mined golden without editing it out.

Master switch: ``goldens_mine.enabled`` (default false). Job kind
``mine_goldens`` runs IN-PROCESS like rft/evolve (pure stdlib I/O; an endpoint
is touched only when ``use_judge`` is on). The 👍 hook (auto-enqueue on
feedback) is deliberately NOT wired into the recorder/feedback hot paths — CLI
+ dashboard endpoint + dispatcher kind cover the need; documented in ROADMAP.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("toaster_loop.mine_goldens")

#: Provenance trust marker written into every mined golden's ``_mine`` block.
MINED_TRUST = "mined"

#: Minimum length of a derived substring check (shorter fragments like "the"
#: would make the tripwire vacuous).
MIN_SUBSTR_LEN = 8

#: Cap on derived substring checks per golden (keeps tasks small & readable).
MAX_SUBSTRS = 3


class MineGoldensError(RuntimeError):
    """Expected mining failure (disabled switch, unreadable store...)."""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------ the store

def mined_path(home: Path) -> Path:
    """Golden-suite append target for mined tasks (matches reality: goldens
    are file/config data, not a DB table; validation reads ValidationConfig
    which merges this file in)."""
    return Path(home) / "goldens" / "mined.jsonl"


def load_mined_goldens(home: Path) -> List[Dict[str, Any]]:
    """Read enabled mined goldens (oldest first). Tolerant: malformed lines
    are skipped, never fatal — a corrupt tail must not break validation."""
    p = mined_path(home)
    if not p.is_file():
        return []
    out: List[Dict[str, Any]] = []
    seen_keys: set = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("mined goldens: skipping malformed line in %s", p)
            continue
        if not isinstance(rec, dict) or not rec.get("prompt"):
            continue
        if rec.get("enabled") is False:  # operator-disabled; absent => enabled
            continue
        from ..training.rejection import prompt_key

        k = prompt_key(str(rec["prompt"]))
        if k in seen_keys:  # defensive double-append dedup
            continue
        seen_keys.add(k)
        out.append(rec)
    return out


def list_mined_goldens(home: Path) -> List[Dict[str, Any]]:
    """ALL mined goldens (enabled AND operator-disabled), oldest first, with a
    stable ``key`` (prompt hash) for the dashboard toggle API. Malformed lines
    are skipped like load_mined_goldens."""
    p = mined_path(home)
    if not p.is_file():
        return []
    from ..training.rejection import prompt_key

    out: List[Dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(rec, dict) or not rec.get("prompt"):
            continue
        rec = dict(rec)
        rec["key"] = prompt_key(str(rec["prompt"]))
        rec.setdefault("enabled", True)
        out.append(rec)
    return out


def set_mined_enabled(home: Path, key: str, enabled: bool) -> bool:
    """Toggle one mined golden's ``enabled`` flag by prompt hash (atomic
    rewrite of mined.jsonl). The per-line flag exists so operators can retire
    a golden without editing it out; this is the API that makes that possible
    without hand-editing JSONL. Returns False when no line carries ``key``."""
    p = mined_path(home)
    if not p.is_file():
        return False
    lines = p.read_text(encoding="utf-8").splitlines()
    from ..training.rejection import prompt_key

    out_lines: List[str] = []
    found = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            rec = json.loads(stripped)
        except json.JSONDecodeError:
            out_lines.append(line)  # keep malformed lines untouched
            continue
        if (isinstance(rec, dict)
                and prompt_key(str(rec.get("prompt") or "")) == key):
            rec["enabled"] = bool(enabled)
            found = True
            out_lines.append(json.dumps(rec, ensure_ascii=False))
        else:
            out_lines.append(line)
    if not found:
        return False
    tmp = p.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    os.replace(tmp, p)
    return True


def merge_mined_into_tasks(tasks: Optional[List[Dict[str, Any]]],
                           home: Path) -> List[Dict[str, Any]]:
    """Config goldens + enabled mined goldens (prompt-hash deduped). The two
    ValidationConfig construction points call this so mined goldens gate every
    promotion and feed RFT with zero further wiring. No mined file -> the
    input list, unchanged."""
    from ..training.rejection import prompt_key

    out = [t for t in (tasks or []) if isinstance(t, dict)]
    try:
        mined = load_mined_goldens(home)
    except Exception:  # pragma: no cover - defensive: never break validation
        logger.exception("mined goldens: read failed; using config goldens only")
        return out
    if not mined:
        return out
    have = {prompt_key(str(t.get("prompt") or "")) for t in out}
    for g in mined:
        k = prompt_key(str(g.get("prompt") or ""))
        if k and k in have:
            continue  # manual golden wins over a mined duplicate
        have.add(k)
        out.append(g)
    return out


# --------------------------------------------------------------- distillation

def task_pair(messages: List[Dict[str, Any]]
              ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """(task-defining prompt, successful assistant message) from a trace: the
    LAST non-empty user turn and the last assistant turn after it carrying
    content or tool_calls. (None, None) when the shape is unusable."""
    last_user = None
    for i, m in enumerate(messages or []):
        if m.get("role") == "user" and isinstance(m.get("content"), str) \
                and m["content"].strip():
            last_user = i
    if last_user is None:
        return None, None
    assistant = None
    for m in messages[last_user + 1:]:
        if m.get("role") != "assistant":
            continue
        has_content = isinstance(m.get("content"), str) and m["content"].strip()
        if has_content or (m.get("tool_calls") or []):
            assistant = m
    return messages[last_user]["content"], assistant


def _derive_substrings(content: str) -> List[str]:
    """Stable line fragments of the recorded reply as substring checks.

    Lines shorter than MIN_SUBSTR_LEN are noise ("OK", "Sure") and would make
    a vacuous tripwire; capped at MAX_SUBSTRS to keep goldens small. Lines
    carrying redaction markers ([REDACTED… / [REDACTED_PRIVATE_KEY]) are
    dropped: the self-check replays against the SAME redacted message, so such
    a substring passes trivially here but could NEVER match real future model
    output — it would poison the promotion gate with an always-failing golden.
    The cap applies after filtering, so one masked line never crowds out a
    usable one."""
    subs: List[str] = []
    for raw in (content or "").splitlines():
        line = raw.strip().strip("-*#>").strip()
        if "[REDACTED" in line:
            continue  # can never match un-redacted replay output
        if len(line) >= MIN_SUBSTR_LEN and line not in subs:
            subs.append(line)
        if len(subs) >= MAX_SUBSTRS:
            break
    return subs


def _self_check(message: Dict[str, Any], task: Dict[str, Any]) -> bool:
    """Would the SOURCE trace's own output pass this golden? (validate.py
    semantics — the exact oracle run_golden_suite will replay against)."""
    from ..training.validate import _check_task

    try:
        return bool(_check_task(message, task).get("pass"))
    except Exception as e:  # pragma: no cover - defensive: _check_task never raises
        logger.warning("mined goldens: self-check errored: %s", e)
        return False


def distill(prompt: str, assistant: Dict[str, Any], *,
            judge_extra: Optional[List[str]] = None
            ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Build one golden task from the pair. Returns (task, None) on success or
    (None, reason) when no deterministic self-passing verifier exists.

    Redaction happens HERE for everything persisted and for the self-check
    message, so a golden can never embed a raw secret even from an unredacted
    buffer (redact_text is idempotent — double-scrubbing redacted traces is a
    no-op)."""
    from ..core.redact import redact_messages, redact_text

    prompt_r = redact_text(prompt or "").strip()
    if not prompt_r:
        return None, "empty prompt after redaction"
    msg_r = dict(redact_messages([assistant])[0])

    task: Dict[str, Any] = {"prompt": prompt_r}
    deterministic = 0

    subs = _derive_substrings(msg_r.get("content") or "")
    extra = [s for s in (judge_extra or [])
             if isinstance(s, str) and len(s.strip()) >= MIN_SUBSTR_LEN
             and "[REDACTED" not in s]  # same rule: markers never replay
    for s in extra:  # judge proposals only join after passing the self-check
        if s not in subs and _self_check(msg_r, {"prompt": prompt_r,
                                                 "expect_contains": [s]}):
            subs.append(s)
    if subs:
        task["expect_contains"] = subs[:MAX_SUBSTRS]
        deterministic += 1

    calls = [c for c in (msg_r.get("tool_calls") or []) if isinstance(c, dict)]
    if calls:
        fn = calls[0].get("function")
        name = fn.get("name") if isinstance(fn, dict) else None
        if isinstance(name, str) and name:
            task["expect_tool_call"] = {"name": name}
            deterministic += 1

    if not deterministic:
        return None, ("no deterministic verifier (empty reply, no tool calls); "
                      "pure-judge goldens are refused")
    if not _self_check(msg_r, task):
        from ..training.validate import _check_task

        why = _check_task(msg_r, task).get("error") or "self-check failed"
        return None, f"candidate failed self-check against its own output: {why}"
    return task, None


# ------------------------------------------------------------------ the engine

@dataclass
class MineResult:
    candidates: int = 0
    claimed: int = 0
    mined: int = 0
    skipped_dedup: int = 0
    dropped: List[Dict[str, Any]] = field(default_factory=list)
    goldens: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidates": self.candidates,
            "claimed": self.claimed,
            "mined": self.mined,
            "skipped_dedup": self.skipped_dedup,
            "dropped": self.dropped,
            "warnings": self.warnings,
        }


def _trace_messages(jsonl_path: str) -> Optional[List[Dict[str, Any]]]:
    try:
        with open(jsonl_path, encoding="utf-8") as f:
            recs = [json.loads(l) for l in f if l.strip()]
    except (OSError, json.JSONDecodeError):
        return None
    return [rec["message"] for rec in recs if rec.get("kind") == "message"]


def run_mine_goldens(
    store: Any,
    home: Path,
    cfg: Any,  # GoldensMineConfig
    *,
    existing_goldens: Optional[List[Dict[str, Any]]] = None,
    judge_client_fn: Optional[Callable[[List[Dict[str, str]], str],
                                       Optional[str]]] = None,
    judge_model: Optional[str] = None,
) -> MineResult:
    """One mining pass over 👍 traces. See module docstring for semantics.

    Order (invariants): select + dedup pre-filter -> ATOMIC ledger claim
    (BEGIN IMMEDIATE INSERT OR IGNORE, claim-before-mine) -> distill + redact +
    self-check gate -> append to the golden file -> finalize ledger rows.
    ``existing_goldens`` (the manual ``validation.golden_tasks`` list) is
    prompt-hash deduped against so mining never duplicates a hand-written
    golden."""
    from ..training.rejection import prompt_key

    if not getattr(cfg, "enabled", False):
        raise MineGoldensError(
            "goldens_mine.enabled is false — refusing to run (set the master "
            "switch in the config to enable golden mining)")

    result = MineResult()
    max_per_run = max(0, int(getattr(cfg, "max_per_run", 8)))
    if max_per_run == 0:
        result.warnings.append("goldens_mine.max_per_run is 0; nothing to do")
        return result

    rows = store.query_positive_feedback_traces(
        min_score=float(getattr(cfg, "min_feedback_score", 1.0)),
        limit=int(getattr(cfg, "max_candidates", 64)))

    ledger_keys = set(store.mined_golden_prompt_keys())
    manual_keys = {prompt_key(str(t.get("prompt") or ""))
                   for t in (existing_goldens or []) if isinstance(t, dict)}

    candidates: List[Tuple[Any, str, Dict[str, Any], str]] = []
    seen: set = set()
    for r in rows:
        messages = _trace_messages(r.jsonl_path)
        if messages is None:
            result.warnings.append(
                f"trace {r.trace_id}: transcript unreadable; skipped")
            continue
        prompt, assistant = task_pair(messages)
        if not prompt or assistant is None:
            continue  # nothing task-defining to mine (no claim burned)
        key = prompt_key(prompt)
        if key in ledger_keys or key in manual_keys or key in seen:
            result.skipped_dedup += 1
            continue
        seen.add(key)
        candidates.append((r, prompt, assistant, key))
        if len(candidates) >= max_per_run:
            break
    result.candidates = len(candidates)
    if not candidates:
        return result

    # Claim BEFORE mining (rft_prompts pattern): losers of the race are simply
    # absent and must be skipped, never retried into a duplicate golden.
    claimed = set(store.claim_mined_goldens([c[0].trace_id for c in candidates]))
    result.claimed = len(claimed)

    gp = mined_path(home)
    gp.parent.mkdir(parents=True, exist_ok=True)
    for (r, prompt, assistant, key) in candidates:
        if r.trace_id not in claimed:
            continue  # lost the claim race (concurrent run) — skip
        judge_extra: Optional[List[str]] = None
        if getattr(cfg, "use_judge", False) and judge_client_fn is not None:
            judge_extra = _judge_propose(prompt, assistant, judge_client_fn,
                                          judge_model or "judge")
        task, reason = distill(prompt, assistant, judge_extra=judge_extra)
        if task is None:
            store.record_mined_golden(r.trace_id, prompt_key=key,
                                      status="dropped", reason=reason)
            result.dropped.append({"trace_id": r.trace_id, "reason": reason})
            logger.info("mine goldens: dropped %s: %s", r.trace_id, reason)
            continue
        task["_mine"] = {
            "source_trace_id": r.trace_id,
            "mined_at": _utcnow(),
            "trust": MINED_TRUST,
            "verifiers": [k for k in ("expect_contains", "expect_tool_call")
                          if k in task],
        }
        task["enabled"] = True
        with open(gp, "a", encoding="utf-8") as f:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")
        store.record_mined_golden(r.trace_id, prompt_key=key, status="mined",
                                  enabled=True,
                                  task_json=json.dumps(task, ensure_ascii=False))
        result.mined += 1
        result.goldens.append(task)

    logger.info("mine goldens: claimed=%d mined=%d dropped=%d dedup=%d -> %s",
                result.claimed, result.mined, len(result.dropped),
                result.skipped_dedup, gp)
    return result


# ------------------------------------------------------ optional judge proposals

JUDGE_PROPOSAL_RUBRIC = (
    "You are hardening a regression test. Given the user request and the "
    "accepted-good answer below, propose up to 3 SHORT literal substrings "
    "(5-12 words each) that MUST appear in any future correct answer — exact "
    "phrases copied from the good answer, never paraphrases, no markdown. "
    'Respond with ONLY a JSON object: {"expect_contains": ["...", "..."]}'
)


def _judge_propose(prompt: str, assistant: Dict[str, Any],
                   client_fn: Callable[[List[Dict[str, str]], str],
                                       Optional[str]],
                   model: str) -> Optional[List[str]]:
    """Ask the judge for ADDITIONAL substring candidates (never a replacement
    for deterministic derivation — distill() self-checks every proposal)."""
    content = assistant.get("content") or ""
    reply = None
    try:
        reply = client_fn(
            [{"role": "system", "content": JUDGE_PROPOSAL_RUBRIC},
             {"role": "user",
              "content": f"Request:\n{prompt}\n\nGood answer:\n{content[:2000]}"}],
            model)
    except Exception as e:  # judge failure => deterministic-only candidate
        logger.warning("mine goldens: judge proposal call failed (skipped): %s",
                       e)
        return None
    if not reply:
        return None
    obj = None
    import re as _re
    m = _re.search(r"\{.*\}", reply, _re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            obj = None
    if not isinstance(obj, dict):
        return None
    props = obj.get("expect_contains")
    if not isinstance(props, list):
        return None
    return [p for p in props if isinstance(p, str)]


# --------------------------------------------- config + job-kind plumbing

def goldens_mine_config(config_path: Optional[str] = None) -> Any:
    """GoldensMineConfig resolved from the live config file (flipping the
    master switch takes effect for jobs queued before it — rft contract)."""
    from ..core.config import GoldensMineConfig, resolve_config_path
    from ..training.scheduler import load_yaml_section

    path = resolve_config_path(config_path)
    sec = load_yaml_section(str(path) if path else None, "goldens_mine")
    return GoldensMineConfig.from_yaml({"goldens_mine": sec})


def make_mine_goldens_job_runner(registry: Any, *, home: Path,
                                 defaults: Optional[Any] = None,
                                 ) -> Callable[[Any], Dict[str, Any]]:
    """Closure the scheduler runs in-process for ``mine_goldens`` jobs
    (mirrors make_rft_job_runner). Reads job params {max_per_run,
    min_feedback_score, use_judge} overlaid on config defaults; enforces the
    master switch HERE too. Manual goldens for dedup come from the live
    ``validation.golden_tasks`` section; the judge is contacted ONLY when
    use_judge is on (attach-style endpoint resolution, never a spawn)."""
    from ..core.config import GoldensMineConfig

    base = defaults if isinstance(defaults, GoldensMineConfig) else \
        GoldensMineConfig.from_yaml({"goldens_mine": dict(defaults or {})})

    def runner(job: Any) -> Dict[str, Any]:
        p = job.params if isinstance(job.params, dict) else {}
        if not base.enabled and not bool(p.get("enabled")):
            raise MineGoldensError(
                "goldens_mine.enabled is false — refusing to run (set the "
                "master switch in the config to enable golden mining)")

        overrides = {k: v for k, v in p.items()
                     if k in GoldensMineConfig.__dataclass_fields__
                     and v is not None}
        cfg = GoldensMineConfig(**{
            **{f: getattr(base, f) for f in GoldensMineConfig.__dataclass_fields__},
            **overrides,
        })

        # Manual goldens (dedup targets) from the live validation section.
        existing: List[Dict[str, Any]] = []
        try:
            from ..core.config import resolve_config_path
            from ..training.scheduler import load_yaml_section

            path = resolve_config_path()
            vsec = load_yaml_section(str(path) if path else None, "validation")
            gt = vsec.get("golden_tasks") if isinstance(vsec, dict) else None
            if isinstance(gt, list):
                existing = [t for t in gt if isinstance(t, dict)]
        except Exception:  # pragma: no cover - defensive
            pass

        judge_client_fn = None
        judge_model = None
        if cfg.use_judge:
            from ..training.rejection import resolve_endpoint
            from ..curate.judge import make_chat_client

            ep = resolve_endpoint(home)
            with ep as (base_url, model_name):
                judge_client_fn = make_chat_client(base_url)
                judge_model = model_name
                res = _run_with_store(registry, home, cfg, existing,
                                      judge_client_fn, judge_model)
        else:
            res = _run_with_store(registry, home, cfg, existing, None, None)

        out = res.to_dict()
        registry.set_job_param(job.job_id, "_result", out)
        logger.info("mine_goldens job %s: mined=%d dropped=%d dedup=%d",
                    job.job_id, res.mined, len(res.dropped), res.skipped_dedup)
        return out

    return runner


def _run_with_store(registry: Any, home: Path, cfg: Any,
                    existing: List[Dict[str, Any]],
                    judge_client_fn: Optional[Callable],
                    judge_model: Optional[str]) -> MineResult:
    from ..store.db import TraceStore

    store = TraceStore(registry.db_path)
    try:
        return run_mine_goldens(store, home, cfg,
                                existing_goldens=existing,
                                judge_client_fn=judge_client_fn,
                                judge_model=judge_model)
    finally:
        try:
            store.close()
        except Exception:
            pass

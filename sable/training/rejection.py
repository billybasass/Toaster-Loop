"""Rejection-sampling fine-tuning (STaR / RFT) — turn serving into a data engine.

For prompts where sable can VERIFY correctness, sample K candidate completions
from the currently served model, keep only verified-correct samples, and curate
them into a ShareGPT SFT dataset laid out EXACTLY like ``curate`` output
(sft.jsonl + manifest.json + train.yaml under ``<home>/datasets/<id>/``) so the
existing training pipeline consumes it unchanged.

Composition (every piece already existed):
  * prompt selection — buffered traces via ``store.query_for_curation``
    (already-claimed traces excluded), golden tasks from ``validation:``, or
    explicit seed prompts;
  * K-sampling — chat completions against the served endpoint with elevated
    temperature; sampling knobs route through
    ``serving/run_args.request_sampling_payload`` (thinking-key shaping is a
    recently fixed bug class — never bypass it);
  * verification — golden-style tasks reuse ``training/validate._check_task``;
    open-chat prompts go through the LLM judge (``curate/judge``) at a STRICT
    pass threshold (``rft.judge_min_score``, default 0.85). Self-consistency is
    never trusted: every kept row carries at least one verifier signal;
  * dataset emission — ShareGPT rows via ``curate/sharegpt``, manifest with
    ``source: "rft"`` provenance, train.yaml from the curater template;
  * DPO negatives (``rft.emit_dpo``, default on within RFT) — for every prompt
    that produced BOTH a verified-kept sample and verifier-failed candidates,
    emit chosen/rejected pairs into ``dpo.jsonl`` (same TRL preference shape
    curater mines: {prompt[], chosen, rejected, metadata}), so one sampling
    budget yields SFT + preference data. Orphan negatives never ship: a prompt
    with no kept row contributes neither SFT nor DPO rows;
  * dedup + overlap invariants:
      - the ``rft_prompts`` ledger (store/db.py, schema v7) records every
        sampled prompt key BEFORE sampling (INSERT OR IGNORE claim), so a
        re-run never resamples the same prompt forever;
      - trace-derived prompts are claimed with the SAME
        ``store.claim_traces_owned`` primitive curation uses (finding #8): an
        RFT dataset and a curate dataset can never contain the same trace.

Empty case mirrors the curater's honest-empty semantics: no verified samples ->
NO dataset directory is left behind, the run completes as a no-op (the ledger
rows stay, marking those prompts sampled-but-unkept — resampling them every run
would just burn GPU).

Job kind ``rft`` runs IN-PROCESS (like shadow_eval/evolve): pure stdlib+httpx
I/O against the served endpoint, no torch, needs this process's store/registry
handles. Under dispatcher drain that is fine — one device slot, same
serialization as its siblings. Master switch: ``rft.enabled`` (default false);
with it off nothing here runs and nothing else changes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger("sable.rft")

#: Manifest provenance marker on emitted datasets (spec: source "rft").
RFT_SOURCE = "rft"


class RftError(RuntimeError):
    pass


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------ prompt keys

def normalize_prompt(text: str) -> str:
    """Ledger identity of a prompt: whitespace-trimmed exact text.

    Deliberately NOT redacted here — the ledger stores only a hash, and two
    prompts differing in content should not collide. Trailing/leading
    whitespace differences are the only normalization (matching batch.task_key's
    spirit without coupling to trace headers).
    """
    return (text or "").strip()


def prompt_key(text: str) -> str:
    """Stable hash key for one prompt's normalized text."""
    return hashlib.sha256(normalize_prompt(text).encode()).hexdigest()[:24]


def _last_user_text(messages: List[Dict[str, Any]]) -> Optional[str]:
    for m in reversed(messages or []):
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str) and c.strip():
                return c
    return None


def context_prefix(messages: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    """Prompt context of a trace: system turns + everything up to (and incl.)
    the LAST user turn. Assistant continuations after it are exactly what RFT
    re-samples, so they are cut; a trace with no usable user turn yields None."""
    if not messages:
        return None
    last_user = None
    for i, m in enumerate(messages):
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str) and c.strip():
                last_user = i
    if last_user is None:
        return None
    return [m for m in messages[: last_user + 1]
            if m.get("role") in ("system", "user")]


# ------------------------------------------------------------- prompt selection

@dataclass
class RftPrompt:
    """One prompt candidate for K-sampling, with provenance."""

    key: str                        # ledger dedup key
    source: str                     # 'trace' | 'golden' | 'seed'
    source_id: Optional[str]        # trace_id when source == 'trace'
    messages: List[Dict[str, Any]]  # OpenAI-format prompt context
    task: Optional[Dict[str, Any]] = None   # golden task dict (verification)

    @property
    def is_golden(self) -> bool:
        return self.task is not None


def select_prompts(
    store: Any,
    cfg: Any,  # RFTConfig
    *,
    seeds: Optional[List[str]] = None,
    goldens: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[RftPrompt], List[str]]:
    """Collect prompt candidates, ledger-deduped against previous rft runs.

    Order: golden tasks, explicit seeds, then buffer traces — deterministic and
    GPU-cheap-first. Filters reuse the existing query helpers
    (agent_profile/project); ``query_for_curation(include_curated=False)`` means
    traces already claimed by a curate dataset never even become candidates.
    Returns (prompts, warnings). The atomic claim happens in run_rft.
    """
    warnings: List[str] = []
    prompts: List[RftPrompt] = []
    seen_keys: set = set()

    def _add(p: RftPrompt) -> None:
        if p.key in seen_keys:
            return
        seen_keys.add(p.key)
        prompts.append(p)

    for task in goldens or []:
        prompt_text = normalize_prompt(str(task.get("prompt") or ""))
        if not prompt_text:
            continue
        messages: List[Dict[str, Any]] = []
        sp = task.get("system_prompt")
        if isinstance(sp, str) and sp.strip():
            messages.append({"role": "system", "content": sp})
        messages.append({"role": "user", "content": prompt_text})
        _add(RftPrompt(key=prompt_key(prompt_text), source="golden",
                       source_id=None, messages=messages, task=task))

    for text in seeds or []:
        t = normalize_prompt(text)
        if not t:
            continue
        _add(RftPrompt(key=prompt_key(t), source="seed", source_id=None,
                       messages=[{"role": "user", "content": t}]))

    rows = store.query_for_curation(
        include_curated=False,
        statuses=["success", "partial"],
        agent_profile=cfg.agent_profile,
        project=cfg.project,
    )
    for r in rows:
        if len(prompts) >= int(cfg.max_prompts):
            break
        try:
            with open(r.jsonl_path, encoding="utf-8") as f:
                recs = [json.loads(l) for l in f if l.strip()]
        except (OSError, json.JSONDecodeError):
            warnings.append(f"trace {r.trace_id}: transcript unreadable; skipped")
            continue
        messages = [rec["message"] for rec in recs
                    if rec.get("kind") == "message"]
        prefix = context_prefix(messages)
        if not prefix:
            continue
        key_text = _last_user_text(prefix) or ""
        if not key_text:
            continue
        _add(RftPrompt(key=prompt_key(key_text), source="trace",
                       source_id=r.trace_id, messages=prefix))

    # Ledger dedup: drop prompts sampled by a previous rft run (the atomic
    # INSERT OR IGNORE claim in run_rft is the authority; this pre-filter
    # keeps the reported candidate list honest).
    already = set(store.list_rft_prompt_keys())
    if already:
        fresh = [p for p in prompts if p.key not in already]
        dropped = len(prompts) - len(fresh)
        if dropped:
            warnings.append(
                f"dedup: {dropped} prompt(s) already sampled by a previous "
                "rft run were skipped")
        prompts = fresh
    return prompts[: int(cfg.max_prompts)], warnings


# ------------------------------------------------------------------- sampling

SamplerFn = Callable[[List[Dict[str, Any]]], Optional[Dict[str, Any]]]


def make_sampler(
    base_url: str,
    model: str,
    *,
    temperature: float,
    max_tokens: int,
    sampling: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
    timeout: float = 120.0,
) -> SamplerFn:
    """sample(messages) -> assistant message dict | None (failure = drop).

    Sampling knobs merge through ``request_sampling_payload`` — the SAME
    routing the agent runner uses, so thinking controls reach llama.cpp via
    chat_format_template_controls and vLLM-style servers top-level. RFT's
    elevated ``temperature`` overrides any default-section temperature AFTER
    the merge (that is the point of rejection sampling: diverse candidates).
    """
    import httpx

    from ..serving.run_args import request_sampling_payload

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    frag = request_sampling_payload(sampling or {})

    def sample(messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        body.update(frag)               # thinking controls + default knobs
        body["temperature"] = float(temperature)  # RFT overrides (diversity)
        body["max_tokens"] = int(max_tokens)
        try:
            r = httpx.post(base_url.rstrip("/") + "/chat/completions",
                           json=body, timeout=timeout, headers=headers)
        except Exception as e:  # transport blip -> candidate dropped, run lives
            logger.warning("rft sample failed (dropped): %s", e)
            return None
        if r.status_code != 200:
            logger.warning("rft sample HTTP %d (dropped): %.120s",
                           r.status_code, r.text)
            return None
        message = ((r.json().get("choices") or [{}])[0].get("message")) or {}
        return message or None

    return sample


# --------------------------------------------------------------- verification

def verify_golden(task: Dict[str, Any],
                  message: Optional[Dict[str, Any]]) -> Tuple[bool, str]:
    """Golden-style deterministic check via validate._check_task (reused)."""
    from ..training.validate import _check_task

    if message is None:
        return False, "no completion"
    report = _check_task(message, task)
    if report.get("pass"):
        return True, "golden checks passed"
    return False, str(report.get("error") or "golden checks failed")


def verify_judged(
    prompt_messages: List[Dict[str, Any]],
    message: Optional[Dict[str, Any]],
    client_fn: Callable[[List[Dict[str, str]], str], Optional[str]],
    model: str,
    min_score: float,
) -> Tuple[bool, str]:
    """Strict judge gate for open-chat prompts. A missing/unparseable verdict
    is a FAIL (never a pass); the bar is ``score >= min_score``."""
    from ..curate.judge import judge_trace

    if message is None:
        return False, "no completion"
    transcript = list(prompt_messages) + [{
        "role": "assistant",
        "content": message.get("content") or "",
        **({"tool_calls": message["tool_calls"]}
           if message.get("tool_calls") else {}),
    }]
    verdict = judge_trace(transcript, client_fn, model)
    if verdict is None:
        return False, "judge produced no verdict"
    if float(verdict["score"]) >= float(min_score):
        return True, f"judge {verdict['score']:.2f} >= {min_score}"
    return False, f"judge {verdict['score']:.2f} < {min_score}"


# ------------------------------------------------------------- dataset emission

def build_dpo_pairs(kept_by_key: Dict[str, Dict[str, Any]],
                    failures_by_key: Dict[str, List[Dict[str, str]]],
                    prompts_by_key: Dict[str, "RftPrompt"],
                    cap_per_prompt: int) -> List[Dict[str, Any]]:
    """Preference pairs from kept samples vs verifier-failed candidates.

    Rules (deferred-item spec):
      * only prompts with a KEPT row contribute — no orphan negatives (a failed
        candidate from an all-rejected prompt has no verified counterfactual,
        and that prompt ships no SFT row either);
      * the kept completion is paired against EACH failed candidate of its own
        prompt (same context on both sides), capped at ``cap_per_prompt``;
      * cap selection prefers DIVERSE failures: first one per distinct failure
        reason (verifier detail) in sampling order, then remaining candidates
        in document order until the cap;
      * prompt side = the prompt messages (OpenAI role/content), matching the
        curater's TRL preference shape.

    Returns pair dicts ({prompt_messages, chosen, rejected, metadata});
    emission/redaction happens at write time.
    """
    pairs: List[Dict[str, Any]] = []
    # Deterministic order: by prompt key (kept/failures were collected in
    # prompt order; sorting keeps output stable across dict iteration).
    for pkey in sorted(kept_by_key):
        if pkey not in failures_by_key:
            continue
        kept_row = kept_by_key[pkey]
        fails = failures_by_key[pkey]
        cap = max(0, int(cap_per_prompt))
        picked: List[Dict[str, str]] = []
        seen_reasons: set = set()
        # Pass 1: distinct failure reasons first (diversity).
        for f in fails:
            if len(picked) >= cap:
                break
            if f["detail"] not in seen_reasons:
                seen_reasons.add(f["detail"])
                picked.append(f)
        # Pass 2: fill remaining slots in document order.
        for f in fails:
            if len(picked) >= cap:
                break
            if all(f is not q for q in picked):
                picked.append(f)
        p = prompts_by_key.get(pkey)
        prompt_messages = [
            {"role": m.get("role"), "content": m.get("content") or ""}
            for m in (p.messages if p else [])
            if m.get("role") in ("system", "user")
        ]
        for f in picked:
            pairs.append({
                "prompt_messages": prompt_messages,
                "chosen": kept_row["completion"],
                "rejected": f["content"],
                "metadata": {
                    "source": RFT_SOURCE,
                    "rft_prompt_key": pkey,
                    "verifier": kept_row["verifier"],
                    "rejected_reason": f["detail"],
                },
            })
    return pairs


def _render_rft_manifest(*, dataset_id: str, cfg: Any, prompts_sampled: int,
                         kept: List[Dict[str, Any]], warnings: List[str],
                         short_claims: int,
                         dpo_pairs: Optional[List[Dict[str, Any]]] = None,
                         ) -> Dict[str, Any]:
    """Manifest mirroring curater's shape (files/trace_ids/counts) so every
    existing consumer works; ``source: "rft"`` marks the provenance."""
    sft_trace_ids = [k["trace_id"] for k in kept]
    judged_any = any(k["verifier"] == "judge" for k in kept)
    # Curater's files.dpo.jsonl lists chosen/rejected source trace ids; a
    # sampling-derived negative has no source trace, so only genuine
    # trace-derived kept sides are listed (seed/golden prompts contribute
    # nothing here — consumers tolerate the empty list exactly like curater's
    # zero-DPO datasets). ``dpo_pairs is None`` = opt-out: emit no entry at
    # all, keeping the manifest byte-identical to pre-feature RFT output.
    dpo_entry: Optional[Dict[str, Any]] = None
    if dpo_pairs is not None:
        pair_keys = {pr["metadata"]["rft_prompt_key"] for pr in dpo_pairs}
        dpo_trace_ids = sorted({k["trace_id"] for k in kept
                                if k["source"] == "trace"
                                and k["prompt_key"] in pair_keys})
        dpo_entry = {
            "count": len(dpo_pairs),
            "trace_ids": dpo_trace_ids,
            "format": "trl-preference",
        }
    files: Dict[str, Any] = {
        "sft.jsonl": {
            "count": len(sft_trace_ids),
            "trace_ids": sft_trace_ids,
            "format": "sharegpt",
        },
    }
    if dpo_entry is not None:
        files["dpo.jsonl"] = dpo_entry
    return {
        "dataset_id": dataset_id,
        "created_at": _utcnow(),
        "dataset_schema_version": 1,
        "trace_schema_version": 1,
        "source": RFT_SOURCE,
        # Trust tier (ingest_gate vocabulary): golden-verified rows are
        # deterministic checks -> curated; judge-decided rows are honestly
        # labeled judge_scored.
        "trust": "judge_scored" if judged_any else "curated",
        "filters": {
            "sample_k": int(cfg.sample_k),
            "judge_min_score": float(cfg.judge_min_score),
            "max_prompts": int(cfg.max_prompts),
            "temperature": float(cfg.temperature),
            "max_tokens": int(cfg.max_tokens),
            "agent_profile": cfg.agent_profile,
            "project": cfg.project,
        },
        "rft": {
            "prompts_sampled": prompts_sampled,
            "rows_kept": len(kept),
            # DPO emission bookkeeping (absent when emit_dpo is off -> old
            # manifests stay byte-identical).
            **({"dpo_pairs_emitted": len(dpo_pairs)}
               if dpo_pairs is not None else {}),
            "per_row": [
                {"prompt_key": k["prompt_key"], "source": k["source"],
                 "verifier": k["verifier"], "detail": k["detail"]}
                for k in kept
            ],
        },
        "files": files,
        "format_hints": {
            "sft_format_type": "sharegpt",
            "train_yaml": "train.yaml",
            "note": ("rejection-sampling SFT rows (K candidates per prompt, "
                     "verified-correct only); format_type=sharegpt compatible."),
        },
        "short_claims": short_claims,
        "warnings": warnings,
    }


# ------------------------------------------------------------------- the engine

@dataclass
class RftResult:
    prompts_sampled: int = 0
    rows_kept: int = 0
    dpo_pairs: int = 0
    dataset_id: Optional[str] = None
    dataset_dir: Optional[Path] = None
    claimed: int = 0
    short_claims: int = 0
    skipped_budget: int = 0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompts_sampled": self.prompts_sampled,
            "rows_kept": self.rows_kept,
            "dpo_pairs": self.dpo_pairs,
            "dataset_id": self.dataset_id,
            "dataset_dir": str(self.dataset_dir) if self.dataset_dir else None,
            "claimed": self.claimed,
            "short_claims": self.short_claims,
            "skipped_budget": self.skipped_budget,
            "warnings": self.warnings,
        }


@contextmanager
def resolve_endpoint(home: Path) -> Iterator[Tuple[str, str]]:
    """(base_url, model_name) of the currently served endpoint — attach-only,
    NEVER spawn (RFT samples from what is serving; that is the data engine).

    Same serve_state.json healthcheck shape as evolve/shadow. No healthy
    endpoint -> honest RftError (fabricating data from a dead server is worse
    than an empty run).
    """
    from ..serving.deploy import read_serve_state
    from ..serving.supervisor import healthcheck

    st = read_serve_state(home) or {}
    if st.get("base_url") and healthcheck(st["base_url"]):
        yield st["base_url"], (st.get("model_name") or "sable")
        return
    raise RftError(
        "no healthy serving endpoint — start/attach a server before rft "
        "(rejection sampling mines the SERVED model; refusing to fabricate)")


def run_rft(
    store: Any,
    home: Path,
    cfg: Any,  # RFTConfig
    *,
    sampler: SamplerFn,
    judge_client_fn: Optional[Callable] = None,
    judge_model: Optional[str] = None,
    seeds: Optional[List[str]] = None,
    goldens: Optional[List[Dict[str, Any]]] = None,
    name: Optional[str] = None,
    counter: Optional[Any] = None,  # TokenCounter (exact budget gate if given)
) -> RftResult:
    """One rejection-sampling pass. See module docstring for semantics.

    Order matters for the invariants: select (ledger pre-filter) -> exact-token
    budget gate -> ATOMIC ledger claim -> reserve dataset dir -> sample K +
    verify -> claim traces of kept rows (claim-before-write, finding #8) ->
    write. Any honest-empty path releases/releases-and-removes so no partial
    dataset dir is ever left behind.
    """
    from ..curate.curater import Curater, CurationConfig, render_train_yaml
    from ..curate.sharegpt import messages_to_sharegpt

    result = RftResult()

    prompts, warnings = select_prompts(store, cfg, seeds=seeds, goldens=goldens)
    result.warnings.extend(warnings)
    if not prompts:
        warnings.append("no prompt candidates (buffer empty or all prompts "
                        "already sampled); nothing to do")
        return result

    # Exact-token budget gate BEFORE claiming: a skipped prompt stays
    # unclaimed so it can be reconsidered (e.g. after compression changes).
    if counter is not None:
        budget = int(cfg.max_prompt_tokens)
        fit = []
        for p in prompts:
            if counter.count_messages(p.messages) > budget:
                result.skipped_budget += 1
                continue
            fit.append(p)
        prompts = fit
        if result.skipped_budget:
            warnings.append(
                f"budget: {result.skipped_budget} prompt(s) exceed "
                f"rft.max_prompt_tokens={budget} and were skipped")
    if not prompts:
        result.warnings = warnings
        return result

    # Atomic ledger claim BEFORE sampling: INSERT OR IGNORE inside one BEGIN
    # IMMEDIATE transaction (claim_traces_owned pattern), so concurrent rft
    # runs and re-runs never both sample a prompt.
    claimed_keys = store.claim_rft_prompts(
        [(p.key, p.source, p.source_id) for p in prompts])
    claimed_set = set(claimed_keys)
    skipped = len(prompts) - len(claimed_set)
    if skipped:
        warnings.append(
            f"dedup: {skipped} prompt(s) lost the claim race "
            "(concurrent rft run) and were skipped")
    prompts = [p for p in prompts if p.key in claimed_set]
    result.prompts_sampled = len(prompts)
    if not prompts:
        return result

    judge_needed = any(not p.is_golden for p in prompts)
    if judge_needed and judge_client_fn is None:
        raise RftError(
            "open-chat prompts need a judge client for verification "
            "(never trust self-consistency alone)")

    # Reserve the dataset dir EARLY (curater's cross-process id lock); an empty
    # dir is not a dataset until manifest.json lands, and every honest-empty
    # path below rmtree's it.
    curater = Curater(store, home)
    dataset_id, ds_dir = curater._reserve_dataset_dir(name or "rft")
    owned: set = set()
    # DPO emission gate (rft.emit_dpo, default on within RFT). emit_dpo is
    # read defensively so callers passing a bare object never crash.
    emit_dpo = bool(getattr(cfg, "emit_dpo", True))
    try:
        # ---- sample K + verify ---------------------------------------------
        kept: List[Dict[str, Any]] = []
        kept_keys: set = set()
        # Verifier-failed candidates per prompt (DPO negative material). Empty
        # /transport-failed responses are not negatives — nothing to learn a
        # preference from — so they stay dropped exactly as before.
        failures: Dict[str, List[Dict[str, str]]] = {}
        for p in prompts:
            for _ in range(max(1, int(cfg.sample_k))):
                message = sampler(list(p.messages))
                if message is None or not (message.get("content") or "").strip():
                    continue
                if p.is_golden:
                    ok, detail = verify_golden(p.task or {}, message)
                else:
                    # judge_needed guard above guarantees a client here; the
                    # explicit check narrows the Optional for type-checkers.
                    if judge_client_fn is None:
                        raise RftError("judge_client_fn required")
                    ok, detail = verify_judged(p.messages, message,
                                               judge_client_fn,
                                               judge_model or "judge",
                                               cfg.judge_min_score)
                if not ok:
                    failures.setdefault(p.key, []).append(
                        {"content": message["content"], "detail": detail})
                    continue
                conv = messages_to_sharegpt(
                    list(p.messages) + [{"role": "assistant",
                                         "content": message["content"]}])
                kept.append({
                    "prompt_key": p.key,
                    "source": p.source,
                    "verifier": "golden" if p.is_golden else "judge",
                    "detail": detail,
                    "conversations": conv,
                    "completion": message["content"],
                    "trace_id": p.source_id or f"rft-{p.key[:12]}",
                })
                kept_keys.add(p.key)
                # best-of-K: one verified row per prompt is enough; extra
                # verified candidates for the same prompt add no signal.
                break

        if not kept:
            warnings.append("no candidate passed verification; no dataset "
                            "emitted (prompts stay ledger-recorded as sampled)")
            shutil.rmtree(ds_dir, ignore_errors=True)
            result.warnings = warnings
            return result

        # ---- claim-before-write (finding #8 invariant) ----------------------
        # Trace-derived rows reuse the curater's atomic claim so an RFT dataset
        # and a curate dataset can NEVER share a trace. Golden/seed rows have
        # no trace to claim. Unowned trace rows are dropped (short claim).
        trace_ids = sorted({r["trace_id"] for r in kept if r["source"] == "trace"})
        if trace_ids:
            owned = set(store.claim_traces_owned(trace_ids, dataset_id))
            lost = [t for t in trace_ids if t not in owned]
            if lost:
                result.short_claims = len(lost)
                warnings.append(
                    f"short claim: {result.short_claims} trace(s) already "
                    "claimed by a curate dataset; their RFT rows were excluded")
                kept = [r for r in kept
                        if r["source"] != "trace" or r["trace_id"] in owned]

        if not kept:
            # every trace-derived candidate lost its claim and no golden/seed
            # survivors -> honest empty; release the claims we DID win.
            warnings.append("all rows lost their claims; no dataset emitted")
            if owned:
                store.release_claims(sorted(owned), dataset_id)
                owned = set()
            shutil.rmtree(ds_dir, ignore_errors=True)
            result.warnings = warnings
            return result

        # ---- DPO pairs from failed candidates ---------------------------------
        # Built AFTER claim filtering: a kept row whose trace lost its claim
        # ships neither SFT nor DPO rows (no negatives for unshipped prompts).
        # dpo_pairs stays None when emit_dpo is off -> no file, no manifest
        # entry, byte-identical dataset to pre-feature RFT.
        dpo_pairs: Optional[List[Dict[str, Any]]] = None
        if emit_dpo:
            dpo_pairs = build_dpo_pairs(
                {k["prompt_key"]: k for k in kept}, failures,
                {p.key: p for p in prompts},
                int(getattr(cfg, "max_dpo_pairs_per_prompt", 3)))

        # ---- write -----------------------------------------------------------
        with open(ds_dir / "sft.jsonl", "w", encoding="utf-8") as f:
            for r in kept:
                row = {
                    "conversations": r["conversations"],
                    # provenance extras (HF datasets tolerate extra columns;
                    # format_type=sharegpt reads only 'conversations')
                    "trace_id": r["trace_id"],
                    "source": RFT_SOURCE,
                    "rft_prompt_key": r["prompt_key"],
                    "completed": True,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        if dpo_pairs is not None:
            # Curater parity: the file exists even when empty. Fresh model
            # output may echo prompt-embedded secrets, so every durable string
            # goes through core/redact (idempotent; mine_goldens precedent —
            # defense in depth on top of trace-level record-time redaction).
            from ..core.redact import redact_text

            with open(ds_dir / "dpo.jsonl", "w", encoding="utf-8") as f:
                for pr in dpo_pairs:
                    row = {
                        "prompt": [
                            {"role": m["role"],
                             "content": redact_text(m.get("content") or "")}
                            for m in pr["prompt_messages"]],
                        "chosen": redact_text(pr["chosen"]),
                        "rejected": redact_text(pr["rejected"]),
                        "metadata": pr["metadata"],
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

        manifest = _render_rft_manifest(
            dataset_id=dataset_id, cfg=cfg,
            prompts_sampled=result.prompts_sampled, kept=kept,
            warnings=warnings, short_claims=result.short_claims,
            dpo_pairs=dpo_pairs)
        (ds_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
        tcfg = CurationConfig()
        tcfg.base_model = cfg.base_model
        (ds_dir / "train.yaml").write_text(
            render_train_yaml(ds_dir, tcfg, dataset_id), encoding="utf-8")

        # Ledger bookkeeping LAST: kept prompts get the consuming dataset id;
        # non-kept claimed rows stay (dataset_id NULL) so a re-run never
        # resamples them forever.
        store.mark_rft_prompts_emitted(sorted(kept_keys), dataset_id,
                                       kept_keys=sorted(kept_keys))
        result.rows_kept = len(kept)
        result.dpo_pairs = len(dpo_pairs or [])
        result.dataset_id = dataset_id
        result.dataset_dir = ds_dir
        result.claimed = len(owned)
        result.warnings = warnings
    except Exception:
        # Failure cleanup mirrors the curater (finding #8): release this run's
        # trace claims and remove the partial dir; exception propagates. The
        # prompt ledger rows STAY claimed — resampling after a half-write would
        # risk duplicating whatever did land; operator can inspect.
        if owned:
            try:
                store.release_claims(sorted(owned), dataset_id)
            except Exception:  # pragma: no cover - defensive
                logger.exception("rft: could not release claims")
        shutil.rmtree(ds_dir, ignore_errors=True)
        raise

    for w in warnings:
        logger.debug(w)
    return result


# ---------------------------------------------------- config + job-kind plumbing

def rft_config(config_path: Optional[str] = None) -> Any:
    """RFTConfig resolved from the live config file (flipping the master switch
    takes effect for jobs queued before it — same contract as evolution)."""
    from ..core.config import RFTConfig, resolve_config_path
    from ..training.scheduler import load_yaml_section

    path = resolve_config_path(config_path)
    sec = load_yaml_section(str(path) if path else None, "rft")
    return RFTConfig.from_yaml({"rft": sec})


def make_rft_job_runner(registry: Any, *, home: Path,
                        defaults: Optional[Any] = None,
                        endpoint_for: Optional[Callable[[], Any]] = None,
                        sampler_factory: Optional[Callable[..., SamplerFn]]
                        = None,
                        ) -> Callable[[Any], Dict[str, Any]]:
    """Closure the scheduler runs in-process for ``rft`` jobs.

    Reads job params {seeds, goldens, golden_file, sample_k, max_prompts,
    judge_min_score, temperature, name} overlaid on config defaults; enforces
    the master switch HERE too (a job queued before the switch flipped still
    refuses). The dataset lands under ``<home>/datasets/<id>/`` and the summary
    is stored on the job row under ``params._result``.
    """
    from ..core.config import RFTConfig

    base = defaults if isinstance(defaults, RFTConfig) else \
        RFTConfig.from_yaml({"rft": dict(defaults or {})})

    def runner(job: Any) -> Dict[str, Any]:
        p = job.params if isinstance(job.params, dict) else {}
        if not base.enabled and not bool(p.get("enabled")):
            raise RftError(
                "rft.enabled is false — refusing to run (set the master "
                "switch in the config to enable rejection sampling)")

        overrides = {k: v for k, v in p.items()
                     if k in RFTConfig.__dataclass_fields__ and v is not None}
        cfg = RFTConfig(**{
            **{f: getattr(base, f) for f in RFTConfig.__dataclass_fields__},
            **overrides,
        })

        seeds = [str(s) for s in (p.get("seeds") or []) if str(s).strip()] \
            or None
        goldens = [g for g in (p.get("goldens") or [])
                   if isinstance(g, dict) and g.get("prompt")] or None
        if not goldens and p.get("golden_file"):
            from ..evaluation.evolve import load_golden_tasks

            goldens = load_golden_tasks(Path(p["golden_file"]))

        # Additive feed (curate/mine_goldens.py): enabled auto-mined goldens
        # join the golden prompt source, prompt-hash deduped. No mined file ->
        # goldens unchanged (zero behavior change for every existing caller).
        try:
            from ..curate.mine_goldens import load_mined_goldens

            mined = load_mined_goldens(home)
            if mined:
                have = {prompt_key(str(g.get("prompt") or ""))
                        for g in (goldens or [])}
                extra = [g for g in mined
                         if prompt_key(str(g.get("prompt") or "")) not in have]
                if extra:
                    goldens = list(goldens or []) + extra
        except Exception:  # pragma: no cover - defensive: never break rft
            logger.exception("rft: mined-goldens merge failed; continuing")

        ep = (endpoint_for or (lambda: resolve_endpoint(home)))
        with ep() as (base_url, model_name):
            make_s = sampler_factory or make_sampler
            sampler = make_s(
                base_url, model_name,
                temperature=cfg.temperature, max_tokens=cfg.max_tokens,
                sampling=None)
            from ..curate.judge import make_chat_client

            judge_client_fn = make_chat_client(base_url)

            from ..store.db import TraceStore

            store = TraceStore(registry.db_path)
            try:
                counter = None
                try:
                    from ..core.tokenizer import TokenCounter

                    counter = TokenCounter(base_url)
                except Exception:  # pragma: no cover - defensive
                    counter = None
                res = run_rft(
                    store, home, cfg, sampler=sampler,
                    judge_client_fn=judge_client_fn, judge_model=model_name,
                    seeds=seeds, goldens=goldens,
                    name=p.get("name") or "rft", counter=counter)
            finally:
                try:
                    store.close()
                except Exception:
                    pass

        out = res.to_dict()
        registry.set_job_param(job.job_id, "_result", out)
        logger.info("rft job %s: prompts=%d kept=%d dataset=%s",
                    job.job_id, res.prompts_sampled, res.rows_kept,
                    res.dataset_id)
        return out

    return runner

# Idea adapted from PIASystem (MIT) app/core/skill_evolution.py
"""Prompt/profile evolution loop on golden tasks (inspiration pass 4 #39).

``iterate`` runs one fixed prompt per cycle — improving a profile's system
prompt is manual trial and error. The judge + golden checks exist but nothing
*searches*. This module ports PIASystem's genetic variant search shape:

    mutate prompt -> score variants -> keep the best per generation -> repeat

with toaster_loop's safety rails bolted on:

  * Constraint validation BEFORE scoring rejects invalid mutations (length
    cap, must-keep sections verbatim, non-empty, near-duplicate detection via
    normalized whitespace + difflib ratio > 0.98) — rejected variants cost
    zero scoring calls and are recorded with a reason.
  * Fitness is DETERMINISTIC-FIRST: every variant replays the golden task
    file through the same check semantics as ``training/validate.py``
    (substring + exact tool-call validation). A variant that fails a hard
    check is eliminated WITHOUT spending any judge calls. Only variants that
    pass goldens get LLM-judge scoring over the optional replay tasks —
    determinism first, judge only for what determinism can't cover.
  * Every LLM call (mutation generation, task replay, judging) is counted
    against a hard call budget carried in job params and capped by
    ``evolution.max_call_budget``. Exhausting the budget stops the search
    HONESTLY with partial results — never a fabricated score.
  * The output is ALWAYS a proposal: lineage + best variant + unified diff,
    stored on the job row (``params._result``). Nothing here ever touches the
    profile config; applying happens only through the explicit
    ``toaster-loop evolve apply <job_id>`` command or the dashboard Apply button.

Cost bound (documented): at most ``k x G`` mutation calls,
``(1 + k x G) x |goldens|`` golden replay generations per generation wave and
``passers x |replay|`` judge scoring calls — all bounded in practice by the
call budget counter, which is the hard stop.

Mutation directions rotate through a fixed small list so runs are auditable:
tighten instructions / add explicit step ordering / clarify tool usage /
reduce ambiguity / restructure with headers.

Execution shape: IN-PROCESS scheduler job kind ``evolve`` (like shadow_eval)
— it needs this process's registry handles + serving attach, and the
scheduler's VRAM gate/device slot keeps "one GPU, one job" honest. LLM calls
go through the breaker-aware judge client (pass 4 #28); thinking is disabled
for mutation generation where the backend supports the knob (the same
``enable_thinking`` / ``chat_format_template_controls`` dual shape
serving/run_args.py documents). Serving reuse prefers whatever is already
healthy (attach-only), else the active model through shadow's attach-or-
temporary endpoint with the #28 crash-loop budget. Stdlib only at module
level; httpx/yaml arrive lazily like everywhere else in this codebase.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger("toaster_loop.evaluation.evolve")

# ------------------------------------------------------------- fixed defaults

#: Mutation directions — a FIXED small list, rotated deterministically so a
#: lineage is auditable ("variant v3 = parent v1 + 'reduce ambiguity'").
MUTATION_DIRECTIONS: Tuple[str, ...] = (
    "tighten instructions",
    "add explicit step ordering",
    "clarify tool usage",
    "reduce ambiguity",
    "restructure with headers",
)

DEFAULT_K = 3                 # mutations per parent per generation
DEFAULT_GENERATIONS = 2       # G
DEFAULT_CALL_BUDGET = 60      # hard cap when config doesn't lower it
#: normalized-whitespace similarity above which a variant is a near-duplicate
DUP_RATIO = 0.98

#: Result storage key on the job row (params_json merge; dismiss/applied flags
#: live next to it so history is never deleted, only flagged).
RESULT_KEY = "_result"
DISMISSED_KEY = "_evolve_dismissed"
APPLIED_KEY = "_evolve_applied"


class EvolveError(RuntimeError):
    """Expected evolution failure (disabled switch, no goldens, no endpoint)."""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------- constraint validation

def length_cap_for(base_prompt: str, cap: Optional[int] = None) -> int:
    """Length ceiling for variants: max(1.5x base, base + 800 chars)."""
    if cap is not None:
        return max(1, int(cap))
    return max(int(1.5 * len(base_prompt)), len(base_prompt) + 800)


def _norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def validate_variant(text: Optional[str], *, base_text: str, cap: int,
                     must_keep_sections: List[str],
                     seen: Dict[str, str]) -> Tuple[bool, Optional[str]]:
    """Constraint validation BEFORE any scoring call.

    Returns (ok, rejected_reason). Rules (PIASystem skill_evolution shape):
      * non-empty;
      * length cap (``length_cap_for`` semantics);
      * every must-keep section header survives VERBATIM;
      * not a near-duplicate of the base prompt or any already-seen variant
        (normalized whitespace + difflib ratio > ``DUP_RATIO``).

    ``seen`` maps normalized-text -> variant_id and is updated by the caller
    only when the variant is accepted for scoring.
    """
    if not text or not text.strip():
        return False, "empty mutation output"
    if len(text) > cap:
        return False, f"over length cap {cap} chars (got {len(text)})"
    for section in must_keep_sections or []:
        if section and section not in text:
            return False, f"missing must-keep section {section!r}"
    norm = _norm_ws(text)
    base_norm = _norm_ws(base_text)
    dup = difflib.SequenceMatcher(None, norm, base_norm).ratio()
    if dup > DUP_RATIO:
        return False, f"near-duplicate of the current prompt (ratio {dup:.3f})"
    for other_norm, other_id in seen.items():
        if difflib.SequenceMatcher(None, norm, other_norm).ratio() > DUP_RATIO:
            return False, f"near-duplicate of variant {other_id} " \
                          f"(ratio > {DUP_RATIO})"
    return True, None


# ------------------------------------------------------------ mutation prompts

MUTATION_META_SYSTEM = (
    "You rewrite agent system prompts. You output ONLY the rewritten prompt "
    "text — no commentary, no code fences, no explanations."
)


def build_mutation_messages(direction: str, prompt: str, cap: int,
                            must_keep_sections: List[str]) -> List[Dict[str, str]]:
    """Strict meta-prompt for one mutation call (PIASystem mutate step)."""
    keep = ""
    if must_keep_sections:
        keep = (" Preserve these MUST-KEEP sections verbatim: "
                + ", ".join(repr(s) for s in must_keep_sections) + ".")
    user = (
        f"Rewrite the following agent system prompt applying this mutation "
        f"direction: {direction}.{keep} Stay under {cap} characters. Output "
        f"only the new prompt.\n\n--- PROMPT STARTS\n{prompt}\n--- PROMPT ENDS"
    )
    return [{"role": "system", "content": MUTATION_META_SYSTEM},
            {"role": "user", "content": user}]


_FENCE = re.compile(r"^```(?:\w+)?\s*|\s*```$", re.M)


def _clean_mutation_reply(raw: Optional[str]) -> Optional[str]:
    """Strip fences/whitespace from a mutation reply; None when nothing left."""
    if not raw:
        return None
    text = _FENCE.sub("", raw).strip()
    return text or None


def make_llm_mutator(base_url: str, model: str) -> Callable[..., Optional[str]]:
    """Production mutator: one judge-model call per variant, thinking off.

    Uses the breaker-aware judge client (``make_chat_client``) with a larger
    completion cap and the documented dual thinking-off shape (top-level
    ``enable_thinking`` for vLLM-style servers +
    ``chat_format_template_controls`` for llama.cpp — serving/run_args.py).
    Transport failures raise; the search loop records them as rejections.
    """
    from ..curate.judge import make_chat_client

    client = make_chat_client(
        base_url, max_tokens=1500,
        extra_payload={"enable_thinking": False,
                       "chat_format_template_controls":
                           {"enable_thinking": False}})

    def mutate(direction: str, prompt: str, cap: int,
               must_keep_sections: Optional[List[str]] = None) -> Optional[str]:
        reply = client(build_mutation_messages(direction, prompt, cap,
                                               must_keep_sections or []),
                       model)
        return _clean_mutation_reply(reply)

    return mutate


# ------------------------------------------------------------------ the search

def _rank_key(entry: Dict[str, Any], texts: Dict[str, str]):
    """Selection key: golden-pass first, then judge mean desc, then SHORTER
    prompt wins ties (a leaner prompt at equal quality is the better
    mutation), then variant id for determinism."""
    mean = entry.get("judge_mean")
    return (0 if entry.get("golden_pass") else 1,
            -(float(mean) if isinstance(mean, (int, float)) else -1.0),
            len(texts.get(entry["variant_id"], "")),
            entry["variant_id"])


def run_evolution(
    base_prompt: str,
    golden_tasks: List[Dict[str, Any]],
    replay_tasks: Optional[List[Dict[str, Any]]] = None,
    *,
    k: int = DEFAULT_K,
    generations: int = DEFAULT_GENERATIONS,
    length_cap: Optional[int] = None,
    must_keep_sections: Optional[List[str]] = None,
    call_budget: int = DEFAULT_CALL_BUDGET,
    mutate_fn: Callable[..., Optional[str]],
    run_task_fn: Callable[[str, Dict[str, Any]], Optional[Dict[str, Any]]],
    judge_fn: Callable[[str, Dict[str, Any], Dict[str, Any]], Optional[float]],
) -> Dict[str, Any]:
    """Fixed-budget prompt search. Pure orchestration — every LLM touch goes
    through the injected ``mutate_fn`` / ``run_task_fn`` / ``judge_fn`` seams,
    each of which consumes exactly one unit of the call budget (checked here
    BEFORE invoking).

    ``golden_tasks`` use the validate_tool_calls golden format;
    ``replay_tasks`` are ``{task_id?, task}`` dicts scored by the judge.
    ``run_task_fn(system_prompt, task)`` returns the assistant message dict
    (content/tool_calls); ``judge_fn(system_prompt, replay_task, message)``
    returns a 0..1 score or None (skipped).

    Returns the proposal dict: {baseline, best, improved, diff, lineage,
    calls_used, call_budget, partial}. Never mutates any profile.
    """
    if not golden_tasks:
        raise EvolveError(
            "no golden tasks — refusing to evolve blind (add goldens in the "
            "validate_tool_calls format; see validation.golden_tasks)")
    replay = list(replay_tasks or [])
    cap = length_cap_for(base_prompt, length_cap)
    must_keep = [s for s in (must_keep_sections or []) if s]
    budget_limit = max(1, int(call_budget))
    state = {"used": 0, "partial": False}

    def _try_call() -> bool:
        if state["used"] >= budget_limit:
            state["partial"] = True
            return False
        state["used"] += 1
        return True

    texts: Dict[str, str] = {"v0": base_prompt}
    seen: Dict[str, str] = {_norm_ws(base_prompt): "v0"}
    lineage: List[Dict[str, Any]] = []
    pool: List[Dict[str, Any]] = []  # scored entries (eligible for selection)
    direction_seq = 0

    def _score(vid: str, parent_id: str, direction: Optional[str]) -> bool:
        """Golden-first fitness for one variant. Returns False when the call
        budget cut scoring short (caller stops the search)."""
        text = texts[vid]
        # 1) deterministic golden checks (validate.py semantics) — hard checks
        golden_pass, n_golden, problems = True, 0, []
        for t in golden_tasks:
            if not _try_call():
                _record(vid, parent_id, direction, golden_pass=None,
                        judge_mean=None, n_scored=n_golden,
                        note="call budget exhausted during golden checks")
                return False
            try:
                message = run_task_fn(text, t)
            except Exception as e:  # transport failure == failed hard check
                logger.warning("evolve %s: golden replay failed: %s", vid, e)
                message = None
            entry = _check_golden(message or {}, t)
            n_golden += 1
            if not entry["pass"]:
                golden_pass = False
                problems.append(entry.get("error") or "golden check failed")
        if not golden_pass:
            # Hard-fail elimination: NO judge calls spent on this variant.
            _record(vid, parent_id, direction, golden_pass=False,
                    judge_mean=None, n_scored=0,
                    note="; ".join(problems[:3]))
            return True
        # 2) judge scoring over replay tasks (only for golden passers)
        scores: List[float] = []
        n_scored = 0
        for rt in replay:
            if not _try_call():
                _record(vid, parent_id, direction, golden_pass=True,
                        judge_mean=None, n_scored=n_scored,
                        note="call budget exhausted during judge scoring")
                return False
            try:
                message = run_task_fn(text, {"prompt": rt.get("task", "")})
                score = (judge_fn(text, rt, message)
                         if message is not None else None)
            except Exception as e:
                logger.warning("evolve %s: judge scoring failed (skipped): %s",
                               vid, e)
                score = None
            n_scored += 1
            if isinstance(score, (int, float)):
                scores.append(float(score))
        mean = (sum(scores) / len(scores)) if scores else None
        _record(vid, parent_id, direction, golden_pass=True, judge_mean=mean,
                n_scored=n_scored)
        return True

    def _record(vid: str, parent_id: str, direction: Optional[str], *,
                golden_pass: Optional[bool], judge_mean: Optional[float],
                n_scored: int, note: Optional[str] = None) -> None:
        entry: Dict[str, Any] = {
            "variant_id": vid,
            "parent_id": parent_id,
            "direction": direction,
            "golden_pass": golden_pass,
            "judge_mean": judge_mean,
            "n_scored": n_scored,
        }
        if note:
            entry["note"] = note
        lineage.append(entry)
        # Breeding pool = every SCORED variant (pass or fail): a baseline that
        # fails its own goldens must still be able to mutate toward better
        # prompts — ranking (_rank_key) puts golden passers first anyway.
        # Budget-truncated entries (golden_pass None) stay out; rejected ones
        # never entered scoring at all.
        if golden_pass is not None:
            pool.append(entry)

    # Baseline (v0): score the CURRENT prompt first so "fitness vs current"
    # is measured, not assumed.
    budget_ok = _score("v0", "v0", None)
    survivors = sorted(pool, key=lambda e: _rank_key(e, texts))[:2]

    g = 0
    while g < max(1, int(generations)) and budget_ok:
        g += 1
        if not survivors:
            break  # nothing scored survived (budget or all-fail) — honest stop
        for parent in list(survivors):
            for _ in range(max(0, int(k))):
                direction = MUTATION_DIRECTIONS[direction_seq
                                                % len(MUTATION_DIRECTIONS)]
                direction_seq += 1
                if not _try_call():
                    break
                vid = f"v{len(texts)}"
                parent_text = texts[parent["variant_id"]]
                try:
                    text = mutate_fn(direction, parent_text, cap, must_keep)
                except Exception as e:
                    text = None
                    logger.warning("evolve: mutation call failed (%s)", e)
                ok, reason = validate_variant(text, base_text=base_prompt,
                                              cap=cap,
                                              must_keep_sections=must_keep,
                                              seen=seen)
                if not ok or text is None:
                    # Rejected BEFORE scoring — zero scoring calls spent.
                    lineage.append({
                        "variant_id": vid,
                        "parent_id": parent["variant_id"],
                        "direction": direction,
                        "rejected_reason": reason or "empty mutation output",
                        "golden_pass": None,
                        "judge_mean": None,
                        "n_scored": 0,
                    })
                    continue
                texts[vid] = text
                seen[_norm_ws(text)] = vid
                budget_ok = _score(vid, parent["variant_id"], direction)
                if not budget_ok:
                    break
            if not budget_ok:
                break
        survivors = sorted(pool, key=lambda e: _rank_key(e, texts))[:2]

    best_entry = survivors[0] if survivors else None
    best_text = texts.get(best_entry["variant_id"]) if best_entry else None
    diff = "\n".join(difflib.unified_diff(
        base_prompt.splitlines(), (best_text or "").splitlines(),
        fromfile="current_prompt", tofile="proposed_prompt", lineterm=""))
    improved = bool(best_entry
                    and best_entry["variant_id"] != "v0")

    return {
        "ts": _utcnow(),
        "baseline": next((e for e in lineage if e["variant_id"] == "v0"), None),
        "best": ({**best_entry, "text": best_text} if best_entry else None),
        "improved": improved,
        "diff": diff,
        "lineage": lineage,
        "calls_used": state["used"],
        "call_budget": budget_limit,
        "length_cap": cap,
        "k": int(k),
        "generations": int(generations),
        "must_keep_sections": must_keep,
        "n_goldens": len(golden_tasks),
        "n_replay": len(replay),
        "partial": bool(state["partial"]),
    }


def _check_golden(message: Dict[str, Any], task: Dict[str, Any]) -> Dict[str, Any]:
    """One golden task against a reply message — validate.py semantics."""
    from ..training.validate import _check_task

    return _check_task(message, task)


# --------------------------------------------------------------- golden files

def load_golden_tasks(path: Path) -> List[Dict[str, Any]]:
    """Read a golden task file: JSON (list of tasks, or {golden_tasks|tasks:
    [...]}) or YAML with the same shape. Raises EvolveError on anything
    unusable — honest empties never become empty searches."""
    p = Path(path)
    if not p.is_file():
        raise EvolveError(f"golden file not found: {p}")
    raw = p.read_text(encoding="utf-8")
    data: Any = None
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        try:
            import yaml

            data = yaml.safe_load(raw)
        except Exception as e:  # pragma: no cover - corrupt file
            raise EvolveError(f"golden file unreadable ({p}): {e}") from e
    if isinstance(data, dict):
        data = data.get("golden_tasks") or data.get("tasks")
    if not isinstance(data, list) or not data:
        raise EvolveError(
            f"golden file {p} has no tasks (expected a JSON/YAML list of "
            "validate_tool_calls-format golden tasks)")
    tasks = [t for t in data if isinstance(t, dict) and t.get("prompt")]
    if not tasks:
        raise EvolveError(f"golden file {p}: no task has a 'prompt' field")
    return tasks


# ------------------------------------------------------------- serving glue

@contextmanager
def _resolve_endpoint(registry: Any, home: Path) -> Iterator[Tuple[str, str]]:
    """(base_url, model_name) to run candidates against — never spawn storms.

    Attach-first: any healthy server in serve_state.json is used read-only
    (prompt evolution optimizes the PROMPT, not a model artifact, so "whatever
    is serving now" is the right target). Otherwise fall back to the active
    registry model through shadow's attach-or-temporary endpoint, which honors
    the pass-4 #28 crash-loop budget. No healthy endpoint -> honest error.
    """
    from ..serving.deploy import read_serve_state
    from ..serving.supervisor import healthcheck

    st = read_serve_state(home) or {}
    if st.get("base_url") and healthcheck(st["base_url"]):
        yield st["base_url"], (st.get("model_name") or "toaster_loop")
        return

    active = registry.active_model() if registry is not None else None
    if active is None:
        raise EvolveError(
            "no healthy serving endpoint and no active model — start/attach a "
            "server before evolving prompts (refusing to fabricate fitness)")
    from .shadow import _model_endpoint

    with _model_endpoint(active, home=home) as (base_url, model_name):
        yield base_url, model_name


# ------------------------------------------------------- production execution

def run_prompt_evolution(
    base_prompt: str,
    golden_tasks: List[Dict[str, Any]],
    replay_tasks: Optional[List[Dict[str, Any]]],
    *,
    home: Path,
    registry: Any = None,
    k: int = DEFAULT_K,
    generations: int = DEFAULT_GENERATIONS,
    length_cap: Optional[int] = None,
    must_keep_sections: Optional[List[str]] = None,
    call_budget: int = DEFAULT_CALL_BUDGET,
    endpoint_for: Optional[Callable[[], Any]] = None,
    mutate_fn: Optional[Callable[..., Optional[str]]] = None,
) -> Dict[str, Any]:
    """Wire the pure search to real endpoints (judge client + serving).

    All seams injectable for tests; production defaults resolve one endpoint
    and build the mutator / task runner / judge scorer over it.
    """
    from ..curate.judge import judge_trace
    from ..training.validate import _post_task

    ep = (endpoint_for or (lambda: _resolve_endpoint(registry, home)))
    with ep() as (base_url, model_name):
        mutate = mutate_fn or make_llm_mutator(base_url, model_name)
        judge_client = _make_judge_client(base_url, model_name)

        def run_task(system_prompt: str,
                     task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            message, err = _post_task(
                base_url, model_name, task, timeout=120.0, max_tokens=256,
                temperature=0.0, headers={}, system_prompt=system_prompt)
            if err is not None or message is None:
                return None
            return message

        def judge(system_prompt: str, rt: Dict[str, Any],
                  message: Dict[str, Any]) -> Optional[float]:
            transcript = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": rt.get("task", "")},
                {"role": "assistant", "content": message.get("content") or "",
                 **({"tool_calls": message["tool_calls"]}
                    if message.get("tool_calls") else {})},
            ]
            verdict = judge_trace(transcript, judge_client, model_name)
            return verdict["score"] if verdict else None

        return run_evolution(
            base_prompt, golden_tasks, replay_tasks,
            k=k, generations=generations, length_cap=length_cap,
            must_keep_sections=must_keep_sections, call_budget=call_budget,
            mutate_fn=mutate, run_task_fn=run_task, judge_fn=judge)


def _make_judge_client(base_url: str, model_name: str):
    from ..curate.judge import make_chat_client

    return make_chat_client(base_url)


# ---------------------------------------------------- scheduler job plumbing

def evolution_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Raw `evolution:` YAML section ({} when absent)."""
    from ..core.config import resolve_config_path
    from ..training.scheduler import load_yaml_section

    path = resolve_config_path(config_path)
    return load_yaml_section(str(path) if path else None, "evolution")


def make_evolve_job_runner(registry: Any, *, home: Path,
                           defaults: Optional[Dict[str, Any]] = None,
                           endpoint_for: Optional[Callable[[], Any]] = None,
                           mutate_fn: Optional[Callable[..., Optional[str]]]
                           = None,
                           ) -> Callable[[Any], Dict[str, Any]]:
    """Closure the scheduler invokes in-process for ``evolve`` jobs.

    Reads inputs from job params {profile, golden_file | tasks, replay_tasks,
    k, generations, call_budget, must_keep_sections} overlaid on config
    defaults. The master switch is enforced HERE too (a job queued before the
    switch flipped still refuses). The proposal lands on the job row under
    ``params._result`` — never on the profile itself.
    """
    d = dict(defaults or {})

    def runner(job: Any) -> Dict[str, Any]:
        p = job.params if isinstance(job.params, dict) else {}
        if not bool(d.get("enabled", False)):
            raise EvolveError(
                "evolution.enabled is false — refusing to run (set the master "
                "switch in the config to enable prompt evolution)")
        max_budget = int(d.get("max_call_budget", DEFAULT_CALL_BUDGET))
        budget = min(int(p.get("call_budget") or max_budget), max_budget)

        profile_name = p.get("profile")
        base_prompt = p.get("base_prompt")
        if not base_prompt:
            base_prompt = _profile_system_prompt(profile_name)
        if not base_prompt:
            raise EvolveError(
                f"profile {profile_name!r} has no system prompt to evolve "
                "(honest-empty: nothing searched, nothing proposed)")

        tasks = p.get("tasks")
        if isinstance(tasks, list) and tasks:
            goldens = [t for t in tasks if isinstance(t, dict) and t.get("prompt")]
            if not goldens:
                raise EvolveError("embedded task list has no usable tasks")
        elif p.get("golden_file"):
            goldens = load_golden_tasks(Path(p["golden_file"]))
        else:
            raise EvolveError(
                "no golden tasks — refusing to evolve blind (pass --goldens "
                "or embed params.tasks)")

        replay = [rt for rt in (p.get("replay_tasks") or [])
                  if isinstance(rt, dict) and rt.get("task")]

        result = run_prompt_evolution(
            base_prompt, goldens, replay,
            home=home, registry=registry,
            k=int(p.get("k") or DEFAULT_K),
            generations=int(p.get("generations") or DEFAULT_GENERATIONS),
            length_cap=p.get("length_cap"),
            must_keep_sections=p.get("must_keep_sections") or [],
            call_budget=budget,
            endpoint_for=endpoint_for, mutate_fn=mutate_fn)

        registry.set_job_param(job.job_id, RESULT_KEY, result)
        logger.info("evolve job %s: improved=%s calls=%d/%d partial=%s",
                    job.job_id, result["improved"], result["calls_used"],
                    result["call_budget"], result["partial"])
        return result

    return runner


def _profile_system_prompt(name: Optional[str]) -> Optional[str]:
    """Current system prompt of profile ``name`` from the live config file."""
    from ..core.config import load_config, resolve_config_path

    path = resolve_config_path(None)
    cfg = load_config(str(path)) if path else load_config()
    profiles = cfg.profiles if isinstance(cfg.profiles, list) else []
    for prof in profiles:
        if isinstance(prof, dict) and prof.get("name") == name:
            sp = prof.get("system_prompt")
            return sp if isinstance(sp, str) and sp.strip() else None
    return None


# ----------------------------------------------------------------- apply flow

def apply_prompt_to_config(config_path: Path, profile_name: str,
                           new_prompt: str) -> Dict[str, Any]:
    """Explicitly apply a proposed prompt to the profile config (with backup).

    Profiles persist in the config YAML ``profiles:`` list, so the backup is
    kept IN the profile entry: the replaced prompt moves to
    ``system_prompt_prev`` and the change timestamp to ``prompt_evolved_at``
    (documented storage choice — one level of undo travels with the profile,
    and the full lineage stays on the job row). The file itself also gets a
    ``.bak`` copy, matching h_config_put's write pattern.

    NEVER called automatically — only from `toaster-loop evolve apply` or the
    dashboard Apply button.
    """
    import yaml

    p = Path(config_path)
    if not p.is_file():
        raise EvolveError(f"config file not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise EvolveError(f"config {p} is not a YAML mapping")
    profiles = data.get("profiles")
    if not isinstance(profiles, list):
        raise EvolveError(f"config {p} has no profiles section")
    target = next((x for x in profiles
                   if isinstance(x, dict) and x.get("name") == profile_name),
                  None)
    if target is None:
        raise EvolveError(f"no profile {profile_name!r} in {p}")
    prev = target.get("system_prompt")
    target["system_prompt"] = new_prompt
    target["system_prompt_prev"] = prev
    target["prompt_evolved_at"] = _utcnow()

    import shutil

    shutil.copy2(p, p.with_suffix(p.suffix + ".bak"))
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    os_replace(tmp, p)
    return {"ok": True, "profile": profile_name, "prev_chars": len(prev or ""),
            "new_chars": len(new_prompt), "backup": str(p) + ".bak",
            "prev_prompt": prev}


def os_replace(src: Path, dst: Path) -> None:
    """os.replace kept tiny + patchable in tests."""
    import os

    os.replace(src, dst)

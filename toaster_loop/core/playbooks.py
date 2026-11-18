# Idea adapted from PIASystem (MIT) app/core/skill_creator.py + skill_registry.py
"""Trajectory → playbook distillation with promote/demote thresholds (pass 4 #38).

toaster_loop already turns traces into *training data* (curate); this module turns
recurring successful tool workflows into a *runtime artifact*: a small
markdown recipe distilled from repeated tool-signature chains, injected into
the profile system prompt under a hard cap. Deliberately NOT an agent
framework — PIASystem's skill objects (numpy scoring, version trees, replay
buffers) collapse here to one table + template text.

Chain mining (``find_recurring_chains``)
----------------------------------------
Scan finished traces with positive/neutral outcomes (status success/partial;
failures excluded) inside a time window for repeated *tool signature chains*.

Normalization rules (documented simplifications):
  * a trace's chain is the ordered list of tool names from assistant
    ``tool_calls`` in its JSONL transcript — arguments are stripped entirely;
  * candidates are CONTIGUOUS subsequences of length 2..4 of that chain.
    Prefix/suffix-aware mining (PIASystem computes per-skill complexity) is
    overkill here — contiguous windows already capture "read → search → edit"
    style recipes and keep the whole thing a dict counter;
  * an occurrence is a TRACE, not a position: one trace counts once per
    distinct chain no matter how often it appears inside it;
  * ordering is deterministic: count desc, then chain joined asc.

Distillation (``distill_recipe``)
---------------------------------
v1 is TEMPLATE-BASED — zero LLM calls, fully deterministic and cheap. The
recipe carries the chain, occurrence counts, an inferred task label from the
first user message of example traces, per-step observed argument keys, and
per-tool failure counts extracted from tool results (gotchas). Recipe text
runs through ``core.redact.redact_text`` before storage: traces are already
redacted at record time, but distillation also pulls first-user-message
labels, so re-applying the masker is cheap defense in depth. An LLM-polish
step is explicitly out of scope.

Lifecycle (PIASystem thresholds adapted: promote ≥ 0.7, demote ≤ 0.3)
---------------------------------------------------------------------
``candidate -> active | retired``. Promotion NEVER happens on vibes:
  * ``promote(..., check=True)`` runs a bounded judge gate — replay the
    profile's high-feedback tasks (evaluation/shadow.build_replay_set) with
    the recipe as system prompt and judge each reply with curate/judge;
    mean ≥ PROMOTE_JUDGE_THRESHOLD (0.7 on the 0..1 judge scale) promotes,
    anything else leaves the candidate untouched. The runner/profile plumbing
    has NO per-run extra-prompt override hook (``_post_task`` sends user
    messages only), so instead of bolting one onto shadow we run a slim
    dedicated scorer here mirroring shadow's bounded judge calls — documented
    choice, same 2-phase shape (generate → judge), N calls total.
  * without ``check`` the transition IS the explicit user confirmation
    (CLI / dashboard action) required when no replay set or endpoint exists.
Demotion is CORRELATIONAL and honest, driven by ``playbooks assess`` (CLI) or
POST /api/playbooks/assess (dashboard): ``assess_playbook`` compares the mean
recorded outcome_score of matching-chain traces before vs after the
playbook's promoted_ts. No counterfactual, no A/B — a regression there is
evidence worth acting on, not proof. Outcome scores live on [-1, 1]; they
are normalized to 0..1 via (s+1)/2 to reuse PIASystem's demote threshold of
0.3: demote when the with-era mean is strictly below the without-era mean
AND normalizes to ≤ 0.3 over at least DEMOTE_MIN_WITH_SAMPLES traces.

Injection
---------
``prompt_section(cfg, profile)`` renders ``## Playbooks\\n<recipe>...`` for a
profile — active playbooks, most-recently-promoted first, hard total char cap
(``playbooks.total_cap_chars``, default 2000) and per-recipe cap
(``playbooks.per_recipe_cap_chars``, default 800). A recipe that does not fit
whole is OMITTED whole — never sliced mid-markdown. Master switch
``playbooks.enabled`` (default false): when off, create_runner never opens
the store and the system prompt is byte-identical to before this feature.

Cross-feed scope reduction (deliberate)
---------------------------------------
The pass-4 sketch said "every promoted playbook is also exported as ShareGPT
rows". That would mean FABRICATING conversations from a recipe — refused.
``export_playbook`` writes an honest metadata file instead: the recipe
markdown, its chain, and the source trace ids, under the datasets tree,
clearly marked ``not_training_rows``. Real ShareGPT rows can come later by
running curate over real exports of real traces; nothing here invents turns.

Stdlib only (+ httpx on the optional judge-gate transport path, same as the
judge/shadow modules).
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("toaster_loop.core.playbooks")

# ---------------------------------------------------------------- constants

#: PIASystem promote threshold on the judge's absolute 0..1 scale.
PROMOTE_JUDGE_THRESHOLD = 0.7
#: PIASystem demote threshold, applied to outcome scores normalized via
#: (s + 1) / 2 (documented above).
DEMOTE_NORM_THRESHOLD = 0.3
DEMOTE_MIN_WITH_SAMPLES = 3
#: score_history_json keeps the most recent entries, capped:
SCORE_HISTORY_CAP = 20

DEFAULT_MIN_OCCURRENCES = 3
DEFAULT_WINDOW_DAYS = 30
#: bounded lazy JSONL scan (mirrors insights.MAX_TOOL_SCAN_TRACES ethos).
MAX_SCAN_TRACES = 500
#: contiguous-subsequence length bounds for chain candidates.
MIN_CHAIN_LEN, MAX_CHAIN_LEN = 2, 4

TOTAL_CAP_CHARS_DEFAULT = 2000
PER_RECIPE_CAP_CHARS_DEFAULT = 800

PLAYBOOK_STATUSES = ("candidate", "active", "retired")

#: Honest marker written into exported metadata files (scope reduction E).
EXPORT_NOTE = (
    "Playbook metadata, NOT training rows. This file records a distilled "
    "recipe and the real trace ids it was mined from; run `toaster-loop curate` "
    "over real trace exports for actual ShareGPT conversations."
)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------- chain mining

def _tool_chain_from_jsonl(path: str) -> List[str]:
    """Ordered tool-call names of one trace transcript ([] when unreadable).

    Same scan shape as store/insights._tool_stats_from_jsonl, but only the
    assistant tool_calls matter here — the recipe is about the sequence.
    """
    chain: List[str] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("kind") != "message":
                    continue
                m = rec.get("message") or {}
                if m.get("role") != "assistant":
                    continue
                for tc in m.get("tool_calls") or []:
                    fn = (tc.get("function") or {}).get("name") or "unknown"
                    chain.append(fn)
    except (OSError, ValueError, json.JSONDecodeError):
        return []  # pruned/corrupt transcript: honest skip
    return chain


def _subsequences(chain: List[str]) -> set:
    """All contiguous subsequences of length MIN..MAX_CHAIN_LEN."""
    out = set()
    n = len(chain)
    for size in range(MIN_CHAIN_LEN, MAX_CHAIN_LEN + 1):
        for start in range(0, n - size + 1):
            out.add(tuple(chain[start:start + size]))
    return out


def find_recurring_chains(store: Any, profile: Optional[str] = None, *,
                          min_occurrences: int = DEFAULT_MIN_OCCURRENCES,
                          window_days: int = DEFAULT_WINDOW_DAYS,
                          scan_limit: int = MAX_SCAN_TRACES,
                          ) -> List[Dict[str, Any]]:
    """Repeated tool-signature chains in recent successful traces.

    Returns [{chain: [names], count, trace_ids: [...]}] ordered by count
    desc then chain asc (fully deterministic). Failures are excluded — a
    recipe has to be distilled from workflows that WORKED. Honest-empty list
    when nothing recurs; no candidates are created here (see the CLI's
    ``scan --create`` for that step).
    """
    cutoff = (datetime.now(timezone.utc)
              - timedelta(days=max(1, int(window_days)))
              ).isoformat(timespec="seconds")
    rows = store.list_traces(agent_profile=profile or None, limit=None)
    rows = [r for r in rows
            if r.status in ("success", "partial") and r.timestamp >= cutoff]
    rows = sorted(rows, key=lambda r: (r.timestamp, r.trace_id),
                  reverse=True)[: max(1, int(scan_limit))]

    # chain -> set of trace ids (occurrence == trace, deduped per trace)
    hits: Dict[Tuple[str, ...], set] = {}
    for r in rows:
        chain = _tool_chain_from_jsonl(r.jsonl_path)
        if len(chain) < MIN_CHAIN_LEN:
            continue
        for sub in _subsequences(chain):
            hits.setdefault(sub, set()).add(r.trace_id)

    out: List[Dict[str, Any]] = []
    for sub, tids in hits.items():
        if len(tids) < max(2, int(min_occurrences)):
            continue
        out.append({"chain": list(sub), "count": len(tids),
                    "trace_ids": sorted(tids)})
    out.sort(key=lambda d: (-d["count"], "\u0000".join(d["chain"])))
    return out


def trace_matches_chain(store: Any, chain: List[str], *,
                        profile: Optional[str] = None,
                        scan_limit: int = MAX_SCAN_TRACES) -> List[Any]:
    """Traces whose tool sequence contains ``chain`` contiguously.

    Bounded lazy scan (newest first, ``scan_limit`` transcripts) mirroring
    find_recurring_chains' ethos — an unbounded walk over every historical
    trace would make assess_playbook's cost grow forever with the buffer."""
    target = tuple(chain)
    span = len(target)
    rows = sorted(store.list_traces(agent_profile=profile or None, limit=None),
                  key=lambda r: (r.timestamp, r.trace_id), reverse=True)
    rows = rows[: max(1, int(scan_limit))]
    out = []
    for r in rows:
        names = _tool_chain_from_jsonl(r.jsonl_path)
        for i in range(len(names) - span + 1):
            if tuple(names[i:i + span]) == target:
                out.append(r)
                break
    return out


# --------------------------------------------------------------- distill

def _first_user_text(messages: List[Dict[str, Any]]) -> str:
    for m in messages:
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str) and c.strip():
                return c.strip()
    return ""


def _clip_line(text: str, cap: int = 80) -> str:
    line = " ".join(text.split())
    return line[:cap] + ("…" if len(line) > cap else "")


def distill_recipe(chain: List[str], example_traces: List[Any], *,
                   occurrences: Optional[int] = None,
                   profile: Optional[str] = None) -> str:
    """Deterministic markdown recipe for ``chain`` from example traces.

    ``example_traces`` are curate LoadedTrace objects (need .messages and
    .row). NO LLM call — pure template + observed facts. Every section is
    conditional on real data, so the output never carries leftover
    placeholders. Text is secret-redacted before returning (defense in
    depth; traces are already redacted at record time).
    """
    from .redact import redact_text

    names = list(chain)
    n = occurrences if occurrences is not None else max(1, len(example_traces))
    lines: List[str] = []
    lines.append(f"## Workflow: {' → '.join(names)}")
    scope = f" for profile {profile!r}" if profile else ""
    lines.append(
        f"This tool sequence completed successfully in {n} recorded run(s)"
        f"{scope}. When a task resembles the examples below, follow it.")

    label = ""
    if example_traces:
        label = _clip_line(_first_user_text(example_traces[0].messages))
    if label:
        lines.append(f"Typical task: {label}")

    # Steps with observed argument keys (sorted, deduped across examples).
    arg_keys: Dict[str, set] = {}
    fail_counts: Dict[str, int] = {}
    tool_seen: Dict[str, int] = {}
    from ..core.failure_types import classify_tool_result

    for t in example_traces:
        for m in t.messages:
            if m.get("role") != "assistant":
                continue
            for tc in m.get("tool_calls") or []:
                fn = (tc.get("function") or {}).get("name") or ""
                if not fn:
                    continue
                tool_seen[fn] = tool_seen.get(fn, 0) + 1
                try:
                    args = json.loads((tc.get("function") or {})
                                      .get("arguments") or "{}")
                except (TypeError, ValueError):
                    args = {}
                if isinstance(args, dict):
                    arg_keys.setdefault(fn, set()).update(
                        str(k) for k in args.keys())
        # gotchas: per-tool failed results across the examples
        for m in t.messages:
            if m.get("role") != "tool":
                continue
            content = m.get("content")
            if not isinstance(content, str):
                content = json.dumps(content, default=str)
            failed, _kind = classify_tool_result(m.get("name", ""), content)
            if failed:
                nm = m.get("name") or "tool"
                fail_counts[nm] = fail_counts.get(nm, 0) + 1

    lines.append("Steps:")
    for i, name in enumerate(names, 1):
        keys = sorted(arg_keys.get(name, set()))[:6]
        detail = f" (args seen: {', '.join(keys)})" if keys else ""
        lines.append(f"{i}. {name}{detail}")

    gotchas = [f"- {nm} returned an error-shaped result "
               f"{c}× across the example runs — check its output before "
               f"continuing." for nm, c in sorted(fail_counts.items())]
    if gotchas:
        lines.append("Gotchas:")
        lines.extend(gotchas)

    return redact_text("\n".join(lines).strip() + "\n")


# ------------------------------------------------------------- persistence

@dataclass
class Playbook:
    id: str
    profile: Optional[str]
    chain: List[str]
    text: str
    status: str
    created_ts: str
    promoted_ts: Optional[str] = None
    score_history: List[Dict[str, Any]] = field(default_factory=list)
    source_trace_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "profile": self.profile, "chain": self.chain,
            "text": self.text, "status": self.status,
            "created_ts": self.created_ts, "promoted_ts": self.promoted_ts,
            "score_history": self.score_history,
            "source_trace_ids": self.source_trace_ids,
        }


def playbook_id(profile: Optional[str], chain: List[str]) -> str:
    """Deterministic id so re-scanning the same chain never duplicates rows."""
    key = "\x00".join([profile or "", *chain])
    return "pb-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


class PlaybookStore:
    """Lite CRUD over the buffer DB's playbooks table (see store/db.py v6).

    Own connection to the same SQLite file TraceStore uses — the playbooks
    table is part of the buffer schema, and playbook operations are rare and
    small. DDL is re-run defensively so a store opened against a pre-v6 file
    still works once that file migrates (idempotent CREATE family).
    """

    def __init__(self, db_path: Path):
        from ..store.db import PLAYBOOKS_TABLE_DDL

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path),
                                     check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            # Same DB file as TraceStore/ModelRegistry: busy_timeout makes
            # concurrent writers wait instead of raising "database is locked"
            # (both of those set it; this handle must too).
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.executescript(PLAYBOOKS_TABLE_DDL)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:  # pragma: no cover - defensive
                pass

    # ------------------------------------------------------------------ write

    def create_candidate(self, chain: List[str], text: str, *,
                         profile: Optional[str] = None,
                         source_trace_ids: Optional[List[str]] = None
                         ) -> Tuple[str, bool]:
        """Insert a candidate playbook; idempotent on (profile, chain).

        Returns (id, created). An existing row — whatever its status — is
        left untouched: re-scanning must not resurrect a retired playbook or
        clobber an active one's score history.
        """
        pb_id = playbook_id(profile, chain)
        payload = json.dumps({"chain": list(chain),
                              "source_trace_ids": sorted(source_trace_ids
                                                         or [])},
                             ensure_ascii=False)
        with self._lock:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO playbooks "
                "(id, profile, chain_json, text, status, created_ts,"
                " score_history_json) VALUES (?,?,?,?,?,?,?)",
                (pb_id, profile or None, payload, text, "candidate",
                 _utcnow(), "[]"))
            self._conn.commit()
        return pb_id, bool(cur.rowcount)

    def set_status(self, pb_id: str, status: str, *, actor: str = "cli") -> bool:
        """Transition lifecycle + ledger the flip (registry DB hosts both).

        Entering 'active' always re-stamps promoted_ts — it is the era
        boundary demotion compares against, and a retire → re-promote cycle
        starts a NEW era (the old one's scores must not bleed into the new
        comparison). Retired rows are KEPT (archive-don't-delete, backlog #16).
        Ledger failures are logged and swallowed here — documented exception
        to ledger.py's strict contract; see registry/ledger.py module doc.
        """
        if status not in PLAYBOOK_STATUSES:
            raise ValueError(f"unknown playbook status {status!r}")
        ts = _utcnow()
        with self._lock:
            row = self._conn.execute(
                "SELECT id, status FROM playbooks WHERE id = ?",
                (pb_id,)).fetchone()
            if row is None:
                return False
            params: List[Any] = [status]
            if status == "active":
                # era boundary: entering 'active' ALWAYS stamps a fresh
                # promoted_ts. Keeping the first promotion's timestamp across
                # retire -> re-promote would make assess_playbook compare the
                # new era against scores from before an intervening retirement
                # (wrong boundary); the newest activation is the honest one.
                params.append(ts)
            params.append(pb_id)
            self._conn.execute(
                "UPDATE playbooks SET status = ?,"
                " promoted_ts = ? WHERE id = ?"
                if status == "active" else
                "UPDATE playbooks SET status = ? WHERE id = ?",
                params)
            try:
                from ..registry import ledger
                ledger.record(self._conn, actor, f"playbook_{status}", pb_id,
                              before={"status": row["status"]},
                              after={"status": status})
            except Exception as e:  # ledger must never wedge lifecycle
                logger.warning("playbook ledger record failed: %s", e)
            self._conn.commit()
        return True

    def append_score_history(self, pb_id: str, entry: Dict[str, Any]) -> None:
        """Append one gate/assessment reading; keep the newest SCORE_HISTORY_CAP."""
        with self._lock:
            row = self._conn.execute(
                "SELECT score_history_json FROM playbooks WHERE id = ?",
                (pb_id,)).fetchone()
            if row is None:
                return
            try:
                hist = json.loads(row["score_history_json"] or "[]")
                if not isinstance(hist, list):
                    hist = []
            except (TypeError, ValueError):
                hist = []
            hist.append(entry)
            hist = hist[-SCORE_HISTORY_CAP:]
            self._conn.execute(
                "UPDATE playbooks SET score_history_json = ? WHERE id = ?",
                (json.dumps(hist, ensure_ascii=False), pb_id))
            self._conn.commit()

    # ------------------------------------------------------------------- read

    def get(self, pb_id: str) -> Optional[Playbook]:
        with self._lock:
            r = self._conn.execute(
                "SELECT * FROM playbooks WHERE id = ?", (pb_id,)).fetchone()
        return _row_to_playbook(r) if r else None

    def list(self, *, status: Optional[str] = None,
             profile: Optional[str] = None,
             include_profileless: bool = False) -> List[Playbook]:
        """Rows filtered by status/profile. ``include_profileless`` also
        matches global (profile IS NULL) rows — used by prompt injection."""
        conds: List[str] = []
        args: List[Any] = []
        if status:
            conds.append("status = ?")
            args.append(status)
        if profile and include_profileless:
            conds.append("(profile = ? OR profile IS NULL)")
            args.append(profile)
        elif profile:
            conds.append("profile = ?")
            args.append(profile)
        q = "SELECT * FROM playbooks"
        if conds:
            q += " WHERE " + " AND ".join(conds)
        q += (" ORDER BY CASE status WHEN 'active' THEN 0"
              " WHEN 'candidate' THEN 1 ELSE 2 END,"
              " COALESCE(promoted_ts, created_ts) DESC, id")
        with self._lock:
            rows = self._conn.execute(q, args).fetchall()
        return [_row_to_playbook(r) for r in rows]


def _row_to_playbook(r: sqlite3.Row) -> Playbook:
    d = dict(r)
    try:
        payload = json.loads(d.get("chain_json") or "{}")
    except (TypeError, ValueError):
        payload = {}
    try:
        hist = json.loads(d.get("score_history_json") or "[]")
        if not isinstance(hist, list):
            hist = []
    except (TypeError, ValueError):
        hist = []
    return Playbook(
        id=d["id"], profile=d.get("profile"),
        chain=list(payload.get("chain") or []),
        text=d.get("text") or "", status=d.get("status") or "candidate",
        created_ts=d.get("created_ts") or "",
        promoted_ts=d.get("promoted_ts"),
        score_history=hist,
        source_trace_ids=list(payload.get("source_trace_ids") or []),
    )


def open_playbook_store(cfg: Any) -> PlaybookStore:
    from ..store.paths import resolve_toaster_loop_home

    return PlaybookStore(resolve_toaster_loop_home(cfg) / "buffer" / "toaster_loop.db")


# ------------------------------------------------------- promotion gate (#38)

class GateUnavailable(RuntimeError):
    """No endpoint/replay set — the caller falls back to explicit confirm."""


def run_promotion_gate(store: Any, pb: Playbook, *, base_url: str,
                       model: str, n: int = 6,
                       judge_model: Optional[str] = None,
                       api_key: Optional[str] = None,
                       generate_fn: Optional[Callable[..., Optional[str]]]
                       = None,
                       judge_client_for: Optional[Callable[[str, str], Any]]
                       = None,
                       ) -> Dict[str, Any]:
    """Bounded replay gate: recipe-injected replies judged ≥ 0.7 mean?

    Mirrors evaluation/shadow.py's shape (replay set from the same builder,
    judge via curate/judge, failures SKIP instead of fabricating scores) but
    is a slim dedicated scorer because no runner/profile plumbing exists to
    replay whole agent runs with an extra system prompt. One generate + one
    judge call per replay task; skipped tasks never count toward the mean;
    zero scored tasks -> passed=False with an honest reason (empty evidence
    must not masquerade as a win).
    """
    from ..evaluation.shadow import build_replay_set
    from ..curate.judge import judge_trace, make_chat_client

    replay = build_replay_set(store, pb.profile, n=n)
    if not replay:
        raise GateUnavailable(
            f"no replay set for profile {pb.profile!r} — promote explicitly "
            "(user confirmation) instead of --check")

    def _generate(task: str) -> Optional[str]:
        import httpx
        headers = ({"Authorization": f"Bearer {api_key}"}
                   if api_key else {})
        body = {
            "model": model,
            # the ONLY difference from shadow's plain replay: the recipe is
            # injected as the system prompt for the with-recipe side.
            "messages": [
                {"role": "system", "content": pb.text},
                {"role": "user", "content": task},
            ],
            "max_tokens": 256,
            "temperature": 0.0,
        }
        try:
            r = httpx.post(base_url.rstrip("/") + "/chat/completions",
                           json=body, timeout=120.0, headers=headers)
            if r.status_code != 200:
                return None
            msg = ((r.json().get("choices") or [{}])[0].get("message")) or {}
            return msg.get("content")
        except Exception as e:  # transport failure => skip this task
            logger.warning("playbook gate generate failed (skipped): %s", e)
            return None

    generate = generate_fn or _generate
    client = (judge_client_for(base_url, judge_model or model)
              if judge_client_for else make_chat_client(base_url, api_key))

    scores: List[float] = []
    skipped = 0
    for item in replay:
        reply = generate(item["task"])
        if reply is None:
            skipped += 1
            continue
        verdict = judge_trace(
            [{"role": "user", "content": item["task"]},
             {"role": "assistant", "content": reply}],
            client, judge_model or model)
        if verdict is None:
            skipped += 1
            continue
        scores.append(float(verdict["score"]))

    mean = (sum(scores) / len(scores)) if scores else None
    passed = bool(scores and mean is not None
                  and mean >= PROMOTE_JUDGE_THRESHOLD)
    return {
        "ts": _utcnow(), "n": len(replay), "scored": len(scores),
        "skipped": skipped, "judge_mean": mean,
        "threshold": PROMOTE_JUDGE_THRESHOLD, "passed": passed,
        "reason": (None if scores else
                   "zero scored replay tasks (all generate/judge calls "
                   "failed or were skipped)"),
    }


def promote(pb_store: PlaybookStore, pb_id: str, *, check: bool = False,
            actor: str = "cli", gate: Optional[Dict[str, Any]] = None
            ) -> Dict[str, Any]:
    """candidate -> active. With ``check`` a judge gate must pass first;
    without it this call IS the explicit user confirmation (never invoked
    automatically). ``gate`` may carry a precomputed gate verdict."""
    pb = pb_store.get(pb_id)
    if pb is None:
        return {"ok": False, "error": f"no playbook {pb_id}"}
    if pb.status == "active":
        return {"ok": True, "promoted": False,
                "reason": "already active", "playbook": pb.to_dict()}
    if check:
        verdict = gate
        if verdict is None:
            return {"ok": False,
                    "error": "--check requires a gate verdict (no endpoint "
                             "resolved?) — promote without --check to confirm "
                             "explicitly"}
        pb_store.append_score_history(pb_id, {
            "ts": verdict.get("ts") or _utcnow(), "n": verdict.get("n"),
            "mean_with": None, "mean_without": None,
            "judge_mean": verdict.get("judge_mean")})
        if not verdict.get("passed"):
            return {"ok": False, "promoted": False, "gate": verdict,
                    "reason": f"replay gate mean "
                              f"{verdict.get('judge_mean')} < threshold "
                              f"{PROMOTE_JUDGE_THRESHOLD}"}
    pb_store.set_status(pb_id, "active", actor=actor)
    updated = pb_store.get(pb_id)
    return {"ok": True, "promoted": True, "gate": gate,
            "playbook": updated.to_dict() if updated else None}


def assess_playbook(store: Any, pb_store: PlaybookStore, pb: Playbook
                    ) -> Dict[str, Any]:
    """Correlational regression check for an ACTIVE playbook (documented
    honestly above): mean outcome_score of matching-chain traces after the
    promoted_ts era boundary vs before it. Appends a score_history entry and
    demotes candidate evidence — correlational, not causal."""
    if not pb.promoted_ts:
        return {"ok": False, "error": "playbook was never promoted; nothing "
                                      "to compare"}
    rows = trace_matches_chain(store, pb.chain, profile=pb.profile)
    with_scores = [float(r.outcome_score) for r in rows
                   if r.outcome_score is not None
                   and r.timestamp >= pb.promoted_ts]
    without_scores = [float(r.outcome_score) for r in rows
                      if r.outcome_score is not None
                      and r.timestamp < pb.promoted_ts]
    mean_with = (sum(with_scores) / len(with_scores)) if with_scores else None
    mean_without = (sum(without_scores) / len(without_scores)
                    if without_scores else None)
    entry = {"ts": _utcnow(), "n": len(with_scores) + len(without_scores),
             "mean_with": mean_with, "mean_without": mean_without,
             "judge_mean": None}
    pb_store.append_score_history(pb.id, entry)

    demoted = False
    if (mean_with is not None and mean_without is not None
            and len(with_scores) >= DEMOTE_MIN_WITH_SAMPLES):
        norm_with = (mean_with + 1.0) / 2.0
        if norm_with <= DEMOTE_NORM_THRESHOLD and mean_with < mean_without:
            pb_store.set_status(pb.id, "retired", actor="assess")
            demoted = True
    return {"ok": True, "demoted": demoted, **entry}


# ------------------------------------------------------------- injection

def playbooks_config(cfg: Any) -> Dict[str, Any]:
    sec = getattr(cfg, "playbooks", None)
    return dict(sec) if isinstance(sec, dict) else {}


def prompt_section(cfg: Any, profile: Optional[str] = None) -> str:
    """Render the '## Playbooks' block for a run's system prompt.

    Master switch off (default) => '' and NO database access at all — the
    effective prompt stays byte-identical to pre-#38 behavior. Ordering is
    most-recently-promoted first; recipes that do not fit whole (per-recipe
    cap or remaining total budget) are omitted whole, never sliced.
    """
    pcfg = playbooks_config(cfg)
    if not pcfg.get("enabled", False):
        return ""
    total_cap = int(pcfg.get("total_cap_chars", TOTAL_CAP_CHARS_DEFAULT))
    per_cap = int(pcfg.get("per_recipe_cap_chars",
                           PER_RECIPE_CAP_CHARS_DEFAULT))
    try:
        pb_store = open_playbook_store(cfg)
    except Exception as e:  # injection must never break a run
        logger.warning("playbooks: store unavailable (%s)", e)
        return ""
    try:
        rows = pb_store.list(status="active", profile=profile or None,
                             include_profileless=bool(profile))
        if not profile:
            # no profile selected: global playbooks only
            rows = [p for p in rows if p.profile is None]
    finally:
        pb_store.close()

    def _promoted_key(p: Playbook) -> str:
        return p.promoted_ts or p.created_ts

    rows.sort(key=_promoted_key, reverse=True)

    parts: List[str] = []
    used = 0
    for p in rows:
        text = p.text.strip()
        if not text or len(text) > per_cap:
            continue  # whole-recipe omission (per-recipe cap)
        cost = len(text) + (2 if parts else 0)
        if used + cost > total_cap:
            continue  # whole-recipe omission (total cap)
        parts.append(text)
        used += cost
    if not parts:
        return ""
    return "## Playbooks\n" + "\n\n".join(parts)


# ------------------------------------------------------------------ export

def export_playbook(pb: Playbook, *, home: Path,
                    out_path: Optional[Path] = None) -> Path:
    """Write honest metadata (recipe + chain + source trace ids), NOT rows.

    Deliberate scope reduction from the pass-4 sketch — see module docstring
    section 'Cross-feed scope reduction'. Lands under the datasets tree in a
    dedicated ``_playbooks`` dir (dataset listing only reads dirs carrying a
    manifest.json, so it never shows up as a dataset).
    """
    root = Path(home)
    target = out_path or (root / "datasets" / "_playbooks"
                          / f"{pb.id}.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "toaster_loop.playbook_export",
        "not_training_rows": True,
        "note": EXPORT_NOTE,
        "exported_ts": _utcnow(),
        "id": pb.id,
        "profile": pb.profile,
        "chain": pb.chain,
        "recipe": pb.text,
        "source_trace_ids": pb.source_trace_ids,
        "status": pb.status,
    }
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False)
                      + "\n", encoding="utf-8")
    return target

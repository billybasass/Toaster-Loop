"""Prometheus text-exposition metrics for ``GET /api/metrics`` (review idea #5).

Hand-rolled exposition format — no prometheus_client dependency. The output
follows the Prometheus text format spec: one ``# HELP`` + ``# TYPE`` pair per
metric family, sample lines ``name{label="v"} value``, summaries as
``name{quantile="q"}`` series plus ``_sum``/``_count``.

Data sources (all read-only, existing tables/indexes only):

- ``traces``      — status counts, token totals, outcome-score mean,
  request-latency quantiles and per-trace tokens/s throughput. Latency and
  throughput reuse ``insights.percentile`` (ceil nearest-rank) so numbers
  match the insights report exactly.
- ``jobs``        — kind/state counters, queue-wait summary (started_at -
  queued_at), plus two honest gate-block families:
  ``sable_gate_blocks_total`` counts durable block EPISODES from the
  ``gate_events`` ledger (true cumulative counter; one event per
  none->blocked transition or reason change), and
  ``sable_jobs_gate_blocked`` gauges jobs currently carrying a non-null
  ``_gate_reason`` (cleared by the scheduler on unblock).
- ``gate_events`` — durable resource-gate block-event log (schema v9,
  written by training/scheduler.py).
- ``ledger``      — promotion outcomes: op='promote' -> pass, op='rollback'
  -> fail, plus the pass-rate gauge.
- ``rft_prompts`` — kept vs sampled-but-unkept rejection-sampling prompts.
- ``mined_goldens`` — mined/dropped golden-task ledger.
- serving state files — ``sable_instances_up{id}`` from
  ``serving/state/<id>.json`` (pid liveness; attach-mode states report up=1
  when present — no network healthcheck here, scrapes must stay cheap).

Cost: every query is a small indexed/full-table read over the personal-loop
DB; the dashboard handler memoizes the rendered text for ``METRICS_TTL_S``
seconds so a 15 s scrape interval costs at most one recomputation per TTL.

Auth: none here — /api/metrics rides the same server.py token + loopback/CSRF
rules as every other /api/* route. Prometheus scrapers set the
``X-Sable-Token`` header (or ``?token=`` on GET) when a token is configured;
no unauthenticated backdoor exists.
"""

from __future__ import annotations

import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..store.insights import percentile  # shared ceil nearest-rank semantics

METRICS_TTL_S = 5.0
CONTENT_TYPE = "text/plain; version=0.0.4"

# ---------------------------------------------------------------- WP9c mapping
# vLLM-compatible metric NAMES where semantics truly match (inspiration only —
# re-implemented inline per D-12, no vLLM/prometheus_client dependency):
#
#   sable:*                     | vllm mirror?          | semantics note
#   ------------------------------|-----------------------|---------------------------
#   sable_requests_total        | NO (sable only)     | buffered agent traces via the chat proxy; one trace may span N engine calls, so this is NOT engine completions
#   sable_tokens_in_total       | vllm:prompt_tokens_total    | sum of prompt tokens processed per recorded request — matches "tokens processed" semantics
#   sable_tokens_out_total      | vllm:generation_tokens_total| sum of completion tokens processed per recorded request
#   (no TTFT measurement exists)  | NO                    | sable measures per-trace duration, never time-to-first-token; a histogram without samples would be fabrication — absence is documented instead (see HELP on sable_request_latency_seconds)
#   sable_jobs_total{state=...} | NO (sable only)     | job-queue depth (training/curate jobs), not vLLM scheduler request queue; different queue, different semantics
#
# Honesty rule: every rate family above keeps its WP2c denominator beside it
# (score coverage <-> traces_total/scored_total; pass rate <-> decisions_total).
# Scraping cost: one read-only connection + indexed counts, memoized for
# METRICS_TTL_S seconds by the dashboard handler (app.h_metrics).
_VLLM_TOKEN_HELP_IN = ("vLLM-compatible name (WP9c, inspiration only, "
                       "re-implemented inline): sum of prompt tokens across "
                       "recorded requests — mirrors sable_tokens_in_total "
                       "(same source, same value).")
_VLLM_TOKEN_HELP_OUT = ("vLLM-compatible name (WP9c, inspiration only, "
                        "re-implemented inline): sum of generation tokens "
                        "across recorded requests — mirrors "
                        "sable_tokens_out_total (same source, same value).")


# ------------------------------------------------------- metric honesty (WP2c)
# Program-brief rule: whenever a derived rate can mislead, expose BOTH the raw
# and the honest variant beside it. The full audit lives in docs/ROADMAP.md
# (WP2 entry). Pure helpers below so unit tests cover the math directly.

def score_coverage(total_traces: int, scored_traces: int) -> Optional[float]:
    """Fraction of the buffer that outcome_score_mean is actually computed
    over (None when the buffer is empty). A mean over 3 of 400 traces is not
    a quality signal unless the coverage travels with it."""
    if total_traces <= 0:
        return None
    return scored_traces / total_traces


def promotion_denominator(passes: int, fails: int) -> int:
    """Decision count behind sable_promotion_pass_rate. The rate alone is a
    cumulative ratio; without its denominator (and without any time window)
    '100% pass' can mean 'one decision ever'. Exposed as a counter so the
    dashboard/scrape side can refuse to over-read small samples."""
    return int(passes) + int(fails)


def liveness_strength(mode: Optional[str]) -> str:
    """How strong an ``up=1`` claim is for a given instance state mode.

    spawn states probe a pid (strong); attach states only assert that the
    state file exists — no network healthcheck runs in a scrape, so their
    up=1 is weaker evidence and must be distinguishable from spawn's."""
    return "state_file" if (mode or "") == "attach" else "pid_probe"

# Summary quantiles exposed (p50/p90/p95 — same set the insights latency
# section reports). Tuple: (numeric pct for percentile, label value).
QUANTILES: Tuple[Tuple[float, str], ...] = ((50.0, "0.5"), (90.0, "0.9"),
                                            (95.0, "0.95"))


def _fmt_num(v: Any) -> str:
    """Prometheus-safe float formatting: finite only, no NaN/Inf leakage."""
    f = float(v)
    if not math.isfinite(f):
        return "0"
    if f == int(f) and abs(f) < 1e15:
        return str(int(f))
    return repr(round(f, 6))


def _esc_label(v: Any) -> str:
    """Escape a label value per the text-format spec (\\ , \" , \\n)."""
    return (str(v).replace("\\", "\\\\").replace('"', '\\"')
            .replace("\n", "\\n"))


class Exposition:
    """Accumulates HELP/TYPE/sample lines for one scrape."""

    def __init__(self) -> None:
        self.lines: List[str] = []

    def _family(self, name: str, mtype: str, help_text: str) -> None:
        self.lines.append(f"# HELP {name} {help_text}")
        self.lines.append(f"# TYPE {name} {mtype}")

    def counter(self, name: str, help_text: str,
                samples: List[Tuple[Optional[Dict[str, str]], Any]]) -> None:
        self._family(name, "counter", help_text)
        for labels, value in samples:
            self.lines.append(_sample_line(name, labels, value))

    def gauge(self, name: str, help_text: str,
              samples: List[Tuple[Optional[Dict[str, str]], Any]]) -> None:
        self._family(name, "gauge", help_text)
        for labels, value in samples:
            self.lines.append(_sample_line(name, labels, value))

    def summary(self, name: str, help_text: str, values: List[float]) -> None:
        """Precomputed-quantile summary (nearest-rank via insights.percentile).

        Quantile lines appear only when the sample count is non-zero; _sum and
        _count always do, so an empty family is still valid exposition.
        """
        self._family(name, "summary", help_text)
        vals = sorted(float(v) for v in values)
        if vals:
            for pct, label in QUANTILES:
                q = percentile(vals, pct)
                if q is not None:
                    self.lines.append(
                        _sample_line(name, {"quantile": label}, q))
        self.lines.append(_sample_line(f"{name}_sum", None, sum(vals)))
        self.lines.append(_sample_line(f"{name}_count", None, len(vals)))

    def render(self) -> str:
        return "\n".join(self.lines) + "\n"


def _sample_line(name: str, labels: Optional[Dict[str, str]],
                 value: Any) -> str:
    if labels:
        lab = ",".join(f'{k}="{_esc_label(v)}"' for k, v in labels.items())
        return f"{name}{{{lab}}} {_fmt_num(value)}"
    return f"{name} {_fmt_num(value)}"


# ---------------------------------------------------------------- collectors

def _connect(db_path: Path) -> Optional[sqlite3.Connection]:
    """Read-only connection; None when the DB does not exist yet."""
    if not db_path.is_file():
        return None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error:
        return None


def _parse_iso(ts: Any) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(str(ts))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _traces_section(ex: Exposition, conn: Optional[sqlite3.Connection]) -> None:
    status_counts: Dict[str, int] = {}
    tok_in = tok_out = 0
    score_mean: Optional[float] = None
    total_count = scored_count = 0
    durations: List[float] = []
    throughput: List[float] = []
    if conn is not None:
        try:
            for r in conn.execute(
                    "SELECT status, COUNT(*) AS n FROM traces GROUP BY status"):
                status_counts[str(r["status"])] = int(r["n"])
            row = conn.execute(
                "SELECT COALESCE(SUM(tokens_in), 0) AS ti,"
                " COALESCE(SUM(tokens_out), 0) AS to_, "
                "AVG(outcome_score) AS score_mean FROM traces").fetchone()
            tok_in, tok_out, score_mean = int(row["ti"]), int(row["to_"]), row["score_mean"]
            sc = conn.execute(
                "SELECT COUNT(*) AS total,"
                " SUM(CASE WHEN outcome_score IS NOT NULL THEN 1 ELSE 0 END)"
                " AS scored FROM traces").fetchone()
            total_count = int(sc["total"])
            scored_count = int(sc["scored"] or 0)
            for r in conn.execute(
                    "SELECT duration_s, tokens_out FROM traces"):
                d = r["duration_s"]
                if d is not None and float(d) > 0:
                    durations.append(float(d))
                    t = r["tokens_out"]
                    if t is not None:
                        throughput.append(float(t) / float(d))
        except sqlite3.Error:
            pass  # no traces table yet -> honest zeros

    statuses = ["success", "failure", "partial", "unknown"] + sorted(
        s for s in status_counts if s not in ("success", "failure",
                                              "partial", "unknown"))
    ex.counter(
        "sable_traces_total",
        "Buffered traces by terminal status.",
        [({"status": s}, status_counts.get(s, 0)) for s in statuses])
    if score_mean is not None:
        ex.gauge("sable_outcome_score_mean",
                 "Mean outcome_score over scored traces (null when none).",
                 [(None, score_mean)])
    # WP9c: canonical request total. NO vllm:num_requests_total mirror —
    # sable counts chat-proxy agent runs (one trace can span many engine
    # completions), so the semantics do NOT match and mirroring would lie.
    ex.counter("sable_requests_total",
               "Buffered traces (chat-proxy requests) in total, all statuses."
               " Deliberately NOT mirrored as vllm:num_requests_total: a"
               " trace is one agent run, not one engine completion.",
               [(None, total_count)])
    ex.counter("sable_tokens_in_total",
               "Sum of prompt tokens across all buffered traces.",
               [(None, tok_in)])
    ex.counter("sable_tokens_out_total",
               "Sum of completion tokens across all buffered traces.",
               [(None, tok_out)])
    # WP9c vllm-named mirrors — semantics match (tokens processed per recorded
    # request), so both names carry the identical value from one source.
    ex.counter("vllm:prompt_tokens_total", _VLLM_TOKEN_HELP_IN,
               [(None, tok_in)])
    ex.counter("vllm:generation_tokens_total", _VLLM_TOKEN_HELP_OUT,
               [(None, tok_out)])
    # WP2c metric honesty: outcome_score_mean is computed only over SCORED
    # traces. Publish the raw denominator and the coverage ratio beside it so
    # a mean over 3 of 400 traces can never masquerade as a quality signal.
    ex.counter("sable_traces_scored_total",
               "Traces with a non-null outcome_score — WP2c honesty pairing: "
               "the raw denominator behind sable_outcome_score_mean.",
               [(None, scored_count)])
    cov = score_coverage(total_count, scored_count) if total_count else None
    if cov is not None:
        ex.gauge("sable_score_coverage_ratio",
                 "scored / total traces: how representative "
                 "sable_outcome_score_mean is (1.0 = every trace scored).",
                 [(None, cov)])
    ex.summary("sable_request_latency_seconds",
               "Per-trace wall-clock duration (nearest-rank quantiles)."
               " WP9c: NOT mirrored as vllm:time_to_first_token_seconds —"
               " sable records end-to-end trace duration only, never"
               " time-to-first-token; fabricating a TTFT histogram from"
               " durations would misreport engine behavior (documented"
               " absence, metric-honesty rule).",
               durations)
    ex.summary("sable_tokens_per_second",
               "Per-trace generation throughput tokens_out/duration_s.",
               throughput)


def _jobs_section(ex: Exposition, conn: Optional[sqlite3.Connection]) -> None:
    job_counts: List[Tuple[str, str, int]] = []
    waits: List[float] = []
    gate_blocked_now = 0
    gate_events = 0
    if conn is not None:
        try:
            for r in conn.execute(
                    "SELECT kind, state, COUNT(*) AS n FROM jobs"
                    " GROUP BY kind, state"):
                job_counts.append((str(r["kind"]), str(r["state"]), int(r["n"])))
            for r in conn.execute(
                    "SELECT queued_at, started_at FROM jobs"
                    " WHERE started_at IS NOT NULL"):
                q, s = _parse_iso(r["queued_at"]), _parse_iso(r["started_at"])
                if q is not None and s is not None:
                    waits.append(max(0.0, (s - q).total_seconds()))
        except sqlite3.Error:
            pass
        try:
            # json_extract sees the scheduler's null-clearing correctly; the
            # instr() fallback only matters on ancient SQLite without JSON1.
            gate_blocked_now = int(conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE"
                " json_extract(params_json, '$._gate_reason') IS NOT NULL"
            ).fetchone()[0])
        except sqlite3.Error:
            try:
                gate_blocked_now = int(conn.execute(
                    "SELECT COUNT(*) FROM jobs WHERE"
                    " instr(COALESCE(params_json, ''), '\"_gate_reason\"') > 0"
                    " AND instr(COALESCE(params_json, ''), '\"_gate_reason\": null') = 0"
                    " AND instr(COALESCE(params_json, ''), '\"_gate_reason\":null') = 0"
                ).fetchone()[0])
            except sqlite3.Error:
                gate_blocked_now = 0
        try:
            # Durable block-episode log (schema v9): append-only, so this is a
            # true cumulative counter. Absent table on pre-v9 DBs -> honest 0.
            gate_events = int(conn.execute(
                "SELECT COUNT(*) FROM gate_events").fetchone()[0])
        except sqlite3.Error:
            gate_events = 0

    ex.counter(
        "sable_jobs_total",
        "Queued jobs by kind and state.",
        [({"kind": k, "state": s}, n) for k, s, n in sorted(job_counts)])
    ex.summary("sable_job_queue_wait_seconds",
               "Seconds from queued_at to started_at (started jobs only).",
               waits)
    ex.counter("sable_gate_blocks_total",
               "Cumulative resource-gate block episodes recorded by the "
               "scheduler (gate_events ledger; one event per none->blocked "
               "transition or reason change — never cleared).",
               [(None, gate_events)])
    ex.gauge("sable_jobs_gate_blocked",
             "Jobs currently carrying an ingest-gate block reason "
             "(params._gate_reason; cleared by the scheduler on unblock).",
             [(None, gate_blocked_now)])
    # WP9c queue depth (job-queue semantics: training/curate jobs waiting on
    # this dispatcher — NOT the vLLM scheduler's waiting-request queue, so no
    # vllm:num_requests_waiting mirror; see the mapping table in this module).
    queued_now = sum(n for (_k, s, n) in job_counts if s == "queued")
    running_now = sum(n for (_k, s, n) in job_counts if s == "running")
    ex.gauge("sable_jobs_queued",
             "Jobs currently queued (waiting for a dispatcher slot). Job-"
             "queue depth; deliberately NOT mirrored as"
             " vllm:num_requests_waiting (different queue, different semantics).",
             [(None, queued_now)])
    ex.gauge("sable_jobs_running",
             "Jobs currently running on the dispatcher.",
             [(None, running_now)])


def _promotions_section(ex: Exposition,
                        conn: Optional[sqlite3.Connection]) -> None:
    passes = fails = 0
    if conn is not None:
        try:
            for r in conn.execute(
                    "SELECT op, COUNT(*) AS n FROM ledger"
                    " WHERE op IN ('promote', 'rollback') GROUP BY op"):
                if str(r["op"]) == "promote":
                    passes = int(r["n"])
                else:
                    fails = int(r["n"])
        except sqlite3.Error:
            pass  # no ledger table yet
    ex.counter(
        "sable_promotions_total",
        "Promotion outcomes from the operation ledger "
        "(pass = promote ops, fail = rollback ops).",
        [({"result": "pass"}, passes), ({"result": "fail"}, fails)])
    total = promotion_denominator(passes, fails)
    if total:
        # WP2c metric honesty: the pass RATE alone hides its sample size —
        # "100% pass" may mean one decision ever. Pair it with the raw
        # decision count so small denominators are visible at a glance.
        ex.gauge("sable_promotion_pass_rate",
                 "promote / (promote + rollback) ledger operations. Read "
                 "with sable_promotion_decisions_total: rates over tiny "
                 "cumulative denominators are weak evidence (WP2c).",
                 [(None, passes / total)])
        ex.counter("sable_promotion_decisions_total",
                   "Raw promote+rollback decision count behind "
                   "sable_promotion_pass_rate (WP2c honesty pairing).",
                   [(None, total)])


def _ledgers_section(ex: Exposition,
                     conn: Optional[sqlite3.Connection]) -> None:
    kept = dropped_rft = 0
    mined: Dict[str, int] = {}
    if conn is not None:
        try:
            for r in conn.execute(
                    "SELECT kept, COUNT(*) AS n FROM rft_prompts GROUP BY kept"):
                if int(r["kept"]):
                    kept += int(r["n"])
                else:
                    dropped_rft += int(r["n"])
        except sqlite3.Error:
            pass
        try:
            for r in conn.execute(
                    "SELECT status, COUNT(*) AS n FROM mined_goldens"
                    " GROUP BY status"):
                mined[str(r["status"])] = int(r["n"])
        except sqlite3.Error:
            pass
    ex.counter("sable_rft_samples_kept_total",
               "RFT prompts whose sampling produced a kept row.",
               [(None, kept)])
    ex.counter("sable_rft_samples_dropped_total",
               "RFT prompts sampled but never kept (no verified candidate).",
               [(None, dropped_rft)])
    statuses = ["mined", "dropped"] + sorted(
        s for s in mined if s not in ("mined", "dropped"))
    ex.counter(
        "sable_mined_goldens_total",
        "Golden-task mining ledger rows by status.",
        [({"status": s}, mined.get(s, 0)) for s in statuses])


def _instance_up(st: Optional[Dict[str, Any]]) -> float:
    """Cheap liveness from a state file — never touches the network.

    spawn states: pid probe (PermissionError = alive, owned elsewhere);
    attach states: presence of the state file means sable considers it
    running (a full healthcheck would make scrapes expensive and flaky).
    """
    if not st:
        return 0.0
    if st.get("mode") == "attach":
        return 1.0
    pid = st.get("pid")
    if not pid:
        return 0.0
    # Windows-safe liveness (A1): os.kill(pid, 0) TERMINATES on NT. Stdlib-only
    # constraint holds — runtime.pid_alive is pure stdlib (ctypes on NT).
    from ..serving.runtime import pid_alive

    try:
        pid_i = int(pid)
    except (ValueError, TypeError):
        return 0.0
    # None ("alive but not ours") keeps the old PermissionError branch -> up.
    return 1.0 if pid_alive(pid_i) is not False else 0.0


def _instances_section(ex: Exposition, home: Path, cfg: Any = None) -> None:
    from ..serving import instances as srv_inst
    from ..serving.deploy import read_serve_state

    samples: List[Tuple[Optional[Dict[str, str]], float]] = []
    strength: List[Tuple[Optional[Dict[str, str]], float]] = []
    try:
        for inst in srv_inst.load_instances(home, cfg=cfg):
            iid = str(inst.get("id") or "?")
            st = srv_inst.read_state(home, iid)
            if iid == srv_inst.DEFAULT_ID and not st:
                st = read_serve_state(home)
            up = _instance_up(st)
            samples.append(({"id": iid}, up))
            # WP2c metric honesty: an attach-mode up=1 only means "state
            # file present" (no healthcheck in a cheap scrape), which is
            # weaker evidence than a spawn-mode pid probe. Surface the
            # evidence strength per instance so the two can never be
            # conflated.
            strength.append(({"id": iid,
                              "strength": liveness_strength(
                                  (st or {}).get("mode"))}, up))
    except Exception:  # unreadable instance dir must never break a scrape
        pass
    ex.gauge("sable_instances_up",
             "Serving instances considered running (pid probe; attach-mode "
             "states count as up while their state file exists). See "
             "sable_instance_up_strength for how strong that claim is.",
             samples)
    if strength:
        ex.gauge("sable_instance_up_strength",
                 "sable_instances_up split by evidence strength: "
                 "strength=\"pid_probe\" (spawn mode, process verified) vs "
                 "strength=\"state_file\" (attach mode: file presence only, "
                 "no healthcheck run) — WP2c honesty pairing.",
                 strength)


def collect(home: Path, cfg: Any = None) -> str:
    """Render the full exposition body for one scrape (uncached)."""
    home = Path(home)
    conn = _connect(home / "buffer" / "sable.db")
    ex = Exposition()
    try:
        _traces_section(ex, conn)
        _jobs_section(ex, conn)
        _promotions_section(ex, conn)
        _ledgers_section(ex, conn)
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
    _instances_section(ex, home, cfg=cfg)
    return ex.render()

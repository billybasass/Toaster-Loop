"""Bench harness (WP1): tokens/sec measurement + noise floor + token identity.

Measures prompt-processing and generation throughput against a supervised
llama-server endpoint, N repeated runs, and reports per-run values plus
median / min / max / run-to-run spread ((max-min)/median as a percentage).

Measurement discipline — inspiration comment (D-12: ideas may be re-implemented
inline, NEVER imported): the repeat-runs + noise-floor methodology follows the
measurement discipline of the kimi-k3-in-c reference project (run N times, keep
the distribution not a single number, treat anything inside the measured
run-to-run spread as noise, and never trust a speed claim whose output tokens
changed). No code is imported from it.

Three rules this module encodes:

1. NOISE FLOOR — every measurement carries its own spread; a dedicated
   ``--noise-floor`` baseline run persists a record later work packages (WP3
   calibrate) read back via :func:`load_noise_floor`. Effects smaller than the
   recorded spread must be reported as noise, not wins.
2. TOKEN IDENTITY — speed is only comparable when the generated token streams
   are identical. Runs use temperature 0 over a fixed deterministic prompt set;
   if streams diverge between the runs being compared, any speedup computed
   from them is INVALID and :func:`compare_records` says so explicitly.
3. PERSISTENCE — results land as JSON under ``<SABLE_HOME>/bench/`` keyed by
   timestamp + label (same file-per-record atomic-write pattern as the serving
   instance state). ``SABLE_BENCH_DIR`` overrides the directory (tests).

Only pyyaml + httpx + stdlib (D-1); loopback endpoints only — the caller
resolves the endpoint through the existing supervisor/instances machinery, so
this module never spawns or owns a process.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import httpx

#: Env override for the bench results directory (mirrors SABLE_HOME).
ENV_BENCH_DIR = "SABLE_BENCH_DIR"

#: Record schema version — bump when the JSON shape changes.
SCHEMA_VERSION = 1

KIND_BENCH = "bench"
KIND_NOISE_FLOOR = "noise_floor"

DEFAULT_RUNS = 5
DEFAULT_MAX_TOKENS = 64

#: Small deterministic prompt fixture (temperature-0 friendly, no tools):
#: short factual/repetition prompts chosen so greedy decoding is stable. The
#: hash of this set is stored in every record so comparisons only happen
#: between runs over the SAME prompts.
DEFAULT_PROMPTS: List[str] = [
    "Repeat exactly: the quick brown fox jumps over the lazy dog",
    "List the numbers 1 to 10 separated by single spaces and nothing else.",
    "Answer with one word. What color is a clear daytime sky?",
]

_SLUG_RE = re.compile(r"[^A-Za-z0-9_.-]+")


class BenchError(RuntimeError):
    pass


# ------------------------------------------------------------------ prompts

def prompts_sha256(prompts: Sequence[str]) -> str:
    """Stable hash over the prompt set (identity of the measurement input)."""
    h = hashlib.sha256()
    for p in prompts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def load_prompts_file(path: Path) -> List[str]:
    """Read a prompt set from a file: one prompt per line (blank lines skipped,
    ``#`` comments stripped)."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as e:
        raise BenchError(f"cannot read prompts file {path}: {e}") from e
    out = [ln.strip() for ln in text.splitlines()
           if ln.strip() and not ln.lstrip().startswith("#")]
    if not out:
        raise BenchError(f"no prompts found in {path}")
    return out


# --------------------------------------------------------------- statistics

def compute_stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    """median/min/max + run-to-run spread as a percentage of the median.

    Spread definition (noise-floor methodology): ``(max - min) / median * 100``.
    Returns ``spread_pct=None`` when the median is 0 (undefined) so callers can
    distinguish "silent" from "fast". Raises ValueError on an empty input.
    """
    vals = [float(v) for v in values]
    if not vals:
        raise ValueError("compute_stats needs at least one value")
    med = statistics.median(vals)
    lo, hi = min(vals), max(vals)
    spread_pct: Optional[float] = ((hi - lo) / med * 100.0) if med else None
    return {
        "median": med,
        "min": lo,
        "max": hi,
        "spread_pct": spread_pct,
    }


# ------------------------------------------------------------- measurement

def _stream_sha(outputs: Sequence[str]) -> str:
    """Hash of the generated token streams (text proxy at temperature 0 —
    identical greedy decode => identical text => identical hash)."""
    h = hashlib.sha256()
    for o in outputs:
        h.update(o.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _one_completion(client: httpx.Client, base_url: str, model: str,
                    prompt: str, *, max_tokens: int,
                    temperature: float) -> Dict[str, Any]:
    """One chat completion; returns the raw response JSON.

    The endpoint is always resolved by the caller through the existing
    supervisor/attach machinery (loopback-only)."""
    url = base_url.rstrip("/") + "/chat/completions"
    try:
        r = client.post(
            url,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
            },
        )
    except httpx.HTTPError as e:
        raise BenchError(
            f"bench request to {url} failed: {e}. Is the supervised "
            "llama-server running? (`sable serve` / dashboard)") from e
    if r.status_code != 200:
        raise BenchError(f"bench request to {url} returned HTTP "
                         f"{r.status_code}: {r.text[:200]}")
    return r.json()


def run_bench(base_url: str, model: str, *,
              prompts: Optional[Sequence[str]] = None,
              runs: int = DEFAULT_RUNS,
              max_tokens: int = DEFAULT_MAX_TOKENS,
              temperature: float = 0.0,
              timeout: float = 120.0,
              label: str = "",
              kind: str = KIND_BENCH) -> Dict[str, Any]:
    """Run the bench set ``runs`` times against an OpenAI-compatible endpoint.

    Returns a JSON-serializable record (see :func:`save_result`). Per run we
    measure wall-clock throughput over the whole prompt set; when the server
    reports llama.cpp ``timings`` (prompt_per_second / predicted_per_second)
    those are preferred for the tok/s figures and the source is recorded, so a
    later reader knows how trustworthy the numbers are.

    Token identity: every run regenerates the same temperature-0 prompts; the
    per-prompt outputs of each run are hashed and compared against the first
    run. Divergence means greedy decode is NOT stable here (server nondeterminism,
    batching, or a non-zero-temperature config), which invalidates any speedup
    claim computed from these runs (:func:`compare_records` enforces the rule).
    """
    prompts = list(prompts) if prompts else list(DEFAULT_PROMPTS)
    if runs < 1:
        raise BenchError("runs must be >= 1")

    per_run: List[Dict[str, Any]] = []
    outputs_by_run: List[List[str]] = []
    timings_source = "wallclock"

    with httpx.Client(timeout=timeout) as client:
        for run_idx in range(1, runs + 1):
            prompt_tokens = 0
            completion_tokens = 0
            server_prompt_s: Optional[float] = None
            server_gen_s: Optional[float] = None
            outputs: List[str] = []
            t0 = time.monotonic()
            for p in prompts:
                data = _one_completion(client, base_url, model, p,
                                       max_tokens=max_tokens,
                                       temperature=temperature)
                usage = data.get("usage") or {}
                prompt_tokens += int(usage.get("prompt_tokens") or 0)
                completion_tokens += int(usage.get("completion_tokens") or 0)
                timings = data.get("timings")
                # llama-server adds {"timings": {prompt_per_second,
                # predicted_per_second, ...}} — use it when present.
                if isinstance(timings, dict):
                    pps = timings.get("prompt_per_second")
                    eps = timings.get("predicted_per_second")
                    if isinstance(pps, (int, float)) and pps > 0:
                        server_prompt_s = (server_prompt_s or 0.0) + float(pps)
                    if isinstance(eps, (int, float)) and eps > 0:
                        server_gen_s = (server_gen_s or 0.0) + float(eps)
                choices = data.get("choices") or []
                msg = (choices[0].get("message") or {}) if choices else {}
                outputs.append(str(msg.get("content") or ""))
            wall = time.monotonic() - t0

            n_prompts = len(prompts)
            gen_tps: Optional[float] = None
            prompt_tps: Optional[float] = None
            if server_gen_s is not None and server_prompt_s is not None:
                timings_source = "server"
                gen_tps = server_gen_s / n_prompts
                prompt_tps = server_prompt_s / n_prompts
            else:
                # Wall-clock fallback: divide usage token counts by the wall
                # time of the whole run (prompt processing is inside the same
                # window, so both figures are conservative approximations).
                if wall > 0:
                    gen_tps = completion_tokens / wall if completion_tokens else None
                    prompt_tps = prompt_tokens / wall if prompt_tokens else None

            per_run.append({
                "run": run_idx,
                "prompt_toks_per_s": prompt_tps,
                "gen_toks_per_s": gen_tps,
                "wall_s": round(wall, 6),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            })
            outputs_by_run.append(outputs)

    stats_prompt = compute_stats(
        [r["prompt_toks_per_s"] for r in per_run
         if r["prompt_toks_per_s"] is not None]
    ) if any(r["prompt_toks_per_s"] is not None for r in per_run) else None
    stats_gen = compute_stats(
        [r["gen_toks_per_s"] for r in per_run
         if r["gen_toks_per_s"] is not None]
    ) if any(r["gen_toks_per_s"] is not None for r in per_run) else None

    # Token identity across the repeated runs of THIS bench.
    reference = outputs_by_run[0]
    divergent_runs = [
        outputs_by_run.index(o) + 1
        for o in outputs_by_run[1:] if o != reference
    ]
    token_identity = {
        "status": "ok" if not divergent_runs else "drift",
        "streams_sha256": _stream_sha(reference),
        "divergent_runs": divergent_runs,
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "label": label or "",
        "config": {
            "base_url": base_url,
            "model": model,
            "runs": runs,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "prompts_sha256": prompts_sha256(prompts),
            "n_prompts": len(prompts),
        },
        "timings_source": timings_source,
        "per_run": per_run,
        "prompt_processing": stats_prompt,
        "generation": stats_gen,
        "token_identity": token_identity,
    }


# -------------------------------------------------------------- persistence

def bench_dir(home: Optional[Path] = None) -> Path:
    """``<SABLE_HOME>/bench/`` unless ``SABLE_BENCH_DIR`` overrides it."""
    env = os.environ.get(ENV_BENCH_DIR, "").strip()
    if env:
        return Path(env).expanduser()
    if home is not None:
        return Path(home) / "bench"
    from ..store.paths import resolve_sable_home  # lazy: avoid cycles

    return resolve_sable_home() / "bench"


def _slug(text: str, max_len: int = 48) -> str:
    s = _SLUG_RE.sub("-", (text or "").strip()).strip("-")
    return s[:max_len] or "unlabeled"


def save_result(record: Dict[str, Any], *, home: Optional[Path] = None,
                base_dir: Optional[Path] = None) -> Path:
    """Atomically persist one bench record under the bench dir.

    Filename: ``<kind>-<UTC timestamp>-<label slug>.json`` — timestamp + label
    is the key later readers (WP3) sort/filter on. Returns the written path."""
    d = Path(base_dir) if base_dir is not None else bench_dir(home)
    d.mkdir(parents=True, exist_ok=True)
    ts = re.sub(r"[:\-]", "", str(record.get("timestamp", ""))) or \
        time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    name = f"{_slug(str(record.get('kind', KIND_BENCH)))}-{ts}-" \
           f"{_slug(str(record.get('label', '')))}.json"
    p = d / name
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    os.replace(tmp, p)
    return p


def load_results(*, home: Optional[Path] = None,
                 base_dir: Optional[Path] = None,
                 kind: Optional[str] = None,
                 label: Optional[str] = None) -> List[Dict[str, Any]]:
    """All persisted records, oldest first. Corrupt files are skipped."""
    d = Path(base_dir) if base_dir is not None else bench_dir(home)
    if not d.is_dir():
        return []
    out: List[Dict[str, Any]] = []
    for f in sorted(d.glob("*.json")):
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(rec, dict):
            continue
        rec["_path"] = str(f)  # convenience for callers/CLI
        if kind is not None and rec.get("kind") != kind:
            continue
        if label is not None and rec.get("label") != label:
            continue
        out.append(rec)
    return out


def load_noise_floor(*, home: Optional[Path] = None,
                     base_dir: Optional[Path] = None,
                     label: Optional[str] = None
                     ) -> Optional[Dict[str, Any]]:
    """The most recent persisted noise-floor baseline, or None.

    WP3 (calibrate) API: call this to fetch the spread thresholds; treat any
    measured improvement smaller than ``record["generation"]["spread_pct"]``
    (and/or ``prompt_processing``) as within-noise."""
    recs = load_results(home=home, base_dir=base_dir,
                        kind=KIND_NOISE_FLOOR, label=label)
    return recs[-1] if recs else None


# ------------------------------------------------- comparison / identity rule

def compare_records(baseline: Dict[str, Any],
                    candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Speedup of ``candidate`` over ``baseline`` + the token-identity rule.

    A speedup that changes tokens is INVALID: when the two records were taken
    over different prompt sets, or their generated streams differ, every
    speedup here is reported with ``valid=False`` and a reason. When a noise
    floor is available (either record's own spread, or a baseline marked
    kind=noise_floor), speedups inside it are flagged ``within_noise_floor``."""
    reasons: List[str] = []
    cfg_a, cfg_b = baseline.get("config") or {}, candidate.get("config") or {}
    if cfg_a.get("prompts_sha256") != cfg_b.get("prompts_sha256"):
        reasons.append("different prompt sets (prompts_sha256 mismatch)")
    id_a = (baseline.get("token_identity") or {}).get("status")
    id_b = (candidate.get("token_identity") or {}).get("status")
    if id_a != "ok" or id_b != "ok":
        reasons.append("token streams not stable within a run "
                       f"(baseline={id_a}, candidate={id_b})")
    sha_a = (baseline.get("token_identity") or {}).get("streams_sha256")
    sha_b = (candidate.get("token_identity") or {}).get("streams_sha256")
    if sha_a != sha_b:
        reasons.append("generated token streams differ between the records "
                       "being compared — a speedup that changes tokens is "
                       "INVALID")

    def _speedup(key: str) -> Dict[str, Any]:
        sa = baseline.get(key) or {}
        sb = candidate.get(key) or {}
        ma, mb = sa.get("median"), sb.get("median")
        res: Dict[str, Any] = {"baseline_median": ma, "candidate_median": mb,
                               "speedup": None, "within_noise_floor": None}
        if isinstance(ma, (int, float)) and isinstance(mb, (int, float)) \
                and ma > 0:
            sp = mb / ma - 1.0  # fractional improvement in tok/s
            res["speedup"] = sp
            floor = sa.get("spread_pct")
            if isinstance(floor, (int, float)):
                res["within_noise_floor"] = abs(sp * 100.0) < floor
        return res

    return {
        "valid": not reasons,
        "reasons": reasons,
        "verdict": "VALID" if not reasons else "INVALID",
        "generation": _speedup("generation"),
        "prompt_processing": _speedup("prompt_processing"),
    }


# --------------------------------------------------------------- rendering

def format_report(record: Dict[str, Any], *, saved_path: Optional[Path] = None
                  ) -> str:
    """Human-readable stdout table for one bench record."""
    cfg = record.get("config") or {}
    lines: List[str] = []
    kind = record.get("kind", KIND_BENCH)
    label = f" label={record['label']}" if record.get("label") else ""
    lines.append(f"sable bench — model={cfg.get('model')} "
                 f"runs={cfg.get('runs')} kind={kind}{label}")
    lines.append(f"endpoint: {cfg.get('base_url')}  "
                 f"(timings: {record.get('timings_source')}, "
                 f"prompts: {cfg.get('n_prompts')})")
    lines.append("")
    lines.append(f"{'run':<5} {'prompt tok/s':>13} {'gen tok/s':>11} "
                 f"{'wall s':>9}")
    for r in record.get("per_run", []):
        def _f(v: Optional[float], w: int) -> str:
            return f"{v:>{w}.2f}" if isinstance(v, (int, float)) else \
                "-" * w
        lines.append(f"{r.get('run', '-'):<5} "
                     f"{_f(r.get('prompt_toks_per_s'), 13)} "
                     f"{_f(r.get('gen_toks_per_s'), 11)} "
                     f"{_f(r.get('wall_s'), 9)}")
    lines.append("")
    for key, title in (("prompt_processing", "prompt processing"),
                       ("generation", "generation")):
        s = record.get(key)
        if not s:
            lines.append(f"{title}: no data")
            continue
        sp = s.get("spread_pct")
        lines.append(
            f"{title}: median={s['median']:.2f} tok/s  "
            f"min={s['min']:.2f}  max={s['max']:.2f}  "
            f"spread={'n/a' if sp is None else f'{sp:.1f}%'}")
    ti = record.get("token_identity") or {}
    status = ti.get("status", "?")
    extra = ""
    if status != "ok":
        extra = (f" — DIVERGED on runs {ti.get('divergent_runs')}; speedup "
                 "claims from this record are INVALID")
    lines.append(f"token identity: {status.upper()}"
                 f" ({str(ti.get('streams_sha256', ''))[:12]}…){extra}")
    if saved_path is not None:
        lines.append(f"saved: {saved_path}")
    return "\n".join(lines)


def format_comparison(cmp: Dict[str, Any], *,
                      baseline_label: str = "baseline",
                      candidate_label: str = "candidate") -> str:
    """Human-readable verdict for :func:`compare_records`."""
    lines = [f"comparison {baseline_label} -> {candidate_label}: "
             f"{cmp['verdict']}"]
    for r in cmp.get("reasons", []):
        lines.append(f"  reason: {r}")
    for key, title in (("generation", "generation"),
                       ("prompt_processing", "prompt processing")):
        c = cmp.get(key) or {}
        sp = c.get("speedup")
        if sp is None:
            lines.append(f"  {title}: no speedup computed")
            continue
        note = ""
        if c.get("within_noise_floor"):
            note = "  (WITHIN NOISE FLOOR — not a real effect)"
        lines.append(
            f"  {title}: {sp * 100.0:+.1f}% tok/s "
            f"({c['baseline_median']:.2f} -> {c['candidate_median']:.2f})"
            + note)
    if not cmp["valid"]:
        lines.append("  => speedup INVALID: tokens changed between the runs "
                     "being compared")
    return "\n".join(lines)

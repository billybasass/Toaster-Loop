"""Per-machine autotune (WP3): `sable calibrate` sweep + profile store.

Sweeps llama-server knobs that are ALREADY mapped in :mod:`run_args`
(kv cache dtype, flash attention, -ngl via the gpu_memory_pct budget,
batch/ubatch, -np parallel slots, speculative mode) and persists an adopted
profile per (machine-id, model-hash). Serving can then re-apply the profile on
launch — strictly OPT-IN via ``serve.calibration: auto``; with the key absent
the serving path is byte-identical to before WP3.

Measurement discipline rides entirely on the WP1 bench harness
(:mod:`sable.serving.bench`): fixed deterministic prompt set, temperature 0,
N repeated runs, a persisted noise-floor record, and the token-identity rule.
Two rules this module adds on top:

1. ADOPTION THRESHOLD — a candidate value is adopted only when it beats the
   current best by MORE THAN ``max(3%, noise_floor_pct)``, where the noise
   floor comes from :func:`bench.load_noise_floor` (measured + persisted on
   first use, exactly like ``sable bench --noise-floor``). Effects inside the
   floor are recorded as rejected, never claimed as wins.
2. TOKEN IDENTITY — the candidate's generated token streams must match the
   current best's (same semantics as :func:`bench.compare_records`); a faster
   setting that changes tokens is recorded ``invalid`` and NEVER adopted.

The sweep is COORDINATE-WISE / staged (baseline -> test each candidate against
the current best, keep the winner, move to the next knob) instead of a full
cartesian product: bounded runtime (~sum of candidates, not their product),
greedy but honest about it — interactions between knobs are explicitly NOT
explored. The candidate grid is data-driven (:data:`KNOB_GRID` /
:data:`QUICK_GRID`) and the knob list is overridable (``--knobs``, ``--quick``).

Honesty for speculative decoding: when a candidate asks for ``mtp`` but no
drafter exists, :func:`run_args.speculative_args` falls back to ngram-mod; the
profile records ``tested_as: "ngram (mtp unavailable)"`` so nobody later claims
MTP was measured.

Machine + model identity (documented choices):
  * machine id: env ``SABLE_MACHINE_ID`` (test hook) > ``/etc/machine-id`` >
    sha256 over hostname+machine (stdlib only; no new deps).
  * model hash: sha256 over the first 1 MiB of the GGUF plus its size. The
    registry/ledger hash manifests, and :func:`core.provenance.hash_file`
    streams the WHOLE file — too slow to run before every launch on multi-GB
    weights. Header+size is O(1)-cheap and still catches model swaps; it is a
    profile KEY, not an attestation (provenance stays the attestation layer).

Persistence: one profile JSON per machine+model under ``<SABLE_HOME>/
calibrate/`` (env ``SABLE_CALIBRATE_DIR`` overrides — same pattern as the
bench dir), atomic write. Only pyyaml-adjacent stdlib + httpx here; loopback
endpoints only; no new outbound network paths.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import platform
import socket
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import bench

logger = logging.getLogger("sable.serving.calibrate")

#: Env override for the profile directory (mirrors SABLE_BENCH_DIR).
ENV_CALIBRATE_DIR = "SABLE_CALIBRATE_DIR"
#: Test hook: deterministic machine id without touching /etc/machine-id.
ENV_MACHINE_ID = "SABLE_MACHINE_ID"

#: Profile JSON schema version — bump when the shape changes.
PROFILE_VERSION = 1

#: Minimum gain a candidate must show to be adopted, in percent. The effective
#: threshold is max(MIN_GAIN_PCT, noise_floor_spread_pct).
MIN_GAIN_PCT = 3.0

#: Repeated bench runs per candidate during a sweep (small on purpose — the
#: noise floor carries the rest of the statistical weight).
DEFAULT_SWEEP_RUNS = 3

#: How many leading bytes of the GGUF feed the model hash (see module docstring
#: for why header+size instead of a full-file hash).
_MODEL_HASH_HEAD_BYTES = 1024 * 1024

#: Coordinate-wise candidate grid. Keys are `run_settings:` keys exactly as
#: serving/run_args.py maps them, so a candidate patch is just {knob: value}.
#: Values include the current default where meaningful (kept for visibility;
#: a candidate equal to the effective current value is recorded as "baseline"
#: and not re-measured).
KNOB_GRID: Dict[str, List[Any]] = {
    "kv_cache_type": ["f16", "bf16", "q8_0", "q4_0"],
    "flash_attn": [True, False],
    # -ngl variants via the soft VRAM budget (run_args.scale_gpu_layers):
    # 100 = full offload (the implicit default), lower fractions keep fewer
    # layers on the GPU. No-ops on CPU-only boxes — measured as noise there.
    "gpu_memory_pct": [100, 75, 50],
    "batch_size": [2048, 4096, 1024],
    "ubatch_size": [128, 256, 64],
    "parallel": [1, 2, 4],
    # spec modes: off / ngram self-speculation / mtp (recorded honestly as
    # ngram when no drafter is present — see _tested_as).
    "speculative_type": ["off", "ngram", "mtp"],
}

#: Knob evaluation order (cheap, independent knobs first; speculative last
#: because it changes the decode path most).
KNOB_ORDER: Tuple[str, ...] = (
    "kv_cache_type",
    "flash_attn",
    "gpu_memory_pct",
    "batch_size",
    "ubatch_size",
    "parallel",
    "speculative_type",
)

#: Preset for `--quick`: fewer knobs, fewer values — a bounded first pass.
QUICK_GRID: Dict[str, List[Any]] = {
    "kv_cache_type": ["f16", "q8_0"],
    "flash_attn": [True, False],
    "speculative_type": ["off", "ngram"],
}

#: Effective defaults for knobs absent from run_settings (used only to decide
#: whether a candidate equals the current value).
KNOB_DEFAULTS: Dict[str, Any] = {
    "kv_cache_type": "f16",
    "flash_attn": False,
    "gpu_memory_pct": 100,
    "batch_size": None,   # server default — only explicit values compare equal
    "ubatch_size": None,
    "parallel": 1,
    "speculative_type": "off",
}

# Verdicts recorded per candidate.
VERDICT_ADOPTED = "adopted"
VERDICT_REJECTED = "rejected"
VERDICT_INVALID = "invalid"
VERDICT_ERROR = "error"
VERDICT_BASELINE = "baseline"


class CalibrateError(RuntimeError):
    pass


# ------------------------------------------------------------- identities

def machine_id() -> str:
    """Stable id for THIS machine (stdlib only).

    ``SABLE_MACHINE_ID`` env (deterministic test hook) > ``/etc/machine-id``
    (the systemd convention) > sha256 over hostname+machine as a fallback."""
    env = os.environ.get(ENV_MACHINE_ID, "").strip()
    if env:
        return env
    try:
        mid = Path("/etc/machine-id").read_text(encoding="utf-8").strip()
        if mid:
            return mid
    except OSError:
        pass
    raw = f"{socket.gethostname()}|{platform.machine()}".encode("utf-8")
    return "host-" + hashlib.sha256(raw).hexdigest()[:32]


def model_hash(path: Any) -> Optional[str]:
    """Cheap content key for a GGUF: sha256 over the first 1 MiB + file size.

    Chosen deliberately over :func:`core.provenance.hash_file` (full streaming
    hash): this runs on every launch under ``serve.calibration: auto`` and
    multi-GB weights make a full hash unacceptable on the hot path. It is a
    profile lookup key (catches model swaps), NOT a provenance attestation."""
    if not path:
        return None
    p = Path(str(path)).expanduser()
    try:
        size = p.stat().st_size
    except OSError:
        return None
    h = hashlib.sha256()
    try:
        with open(p, "rb") as f:
            h.update(f.read(_MODEL_HASH_HEAD_BYTES))
    except OSError:
        return None
    h.update(b"|size=" + str(size).encode("utf-8"))
    return h.hexdigest()


# ---------------------------------------------------------------- storage

def calibrate_dir(home: Optional[Path] = None) -> Path:
    """``<SABLE_HOME>/calibrate/`` unless ``SABLE_CALIBRATE_DIR`` overrides."""
    env = os.environ.get(ENV_CALIBRATE_DIR, "").strip()
    if env:
        return Path(env).expanduser()
    if home is not None:
        return Path(home) / "calibrate"
    from ..store.paths import resolve_sable_home  # lazy: avoid cycles

    return resolve_sable_home() / "calibrate"


def profile_file_name(mid: str, mhash: str) -> str:
    return f"{mid}-{mhash[:16]}.json"


def save_profile(profile: Dict[str, Any], *, home: Optional[Path] = None) -> Path:
    """Atomically persist one calibration profile (keyed by machine+model)."""
    d = calibrate_dir(home)
    d.mkdir(parents=True, exist_ok=True)
    name = profile_file_name(str(profile.get("machine_id", "unknown")),
                             str(profile.get("model_hash", "0" * 16)))
    p = d / name
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    os.replace(tmp, p)
    return p


def load_profile(mid: str, mhash: str, *,
                 home: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """The stored profile for exactly this machine+model, or None."""
    p = calibrate_dir(home) / profile_file_name(mid, mhash)
    if not p.is_file():
        return None
    try:
        prof = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(prof, dict):
        return None
    prof["_path"] = str(p)
    return prof


def list_profiles(*, home: Optional[Path] = None) -> List[Dict[str, Any]]:
    """All stored profiles, oldest file first. Corrupt files are skipped."""
    d = calibrate_dir(home)
    if not d.is_dir():
        return []
    out: List[Dict[str, Any]] = []
    for f in sorted(d.glob("*.json")):
        try:
            prof = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(prof, dict) and prof.get("model_hash"):
            prof["_path"] = str(f)
            out.append(prof)
    return out


def clear_profiles(*, home: Optional[Path] = None) -> int:
    """Delete all stored profiles; returns the number removed."""
    d = calibrate_dir(home)
    if not d.is_dir():
        return 0
    n = 0
    for f in d.glob("*.json"):
        try:
            f.unlink()
            n += 1
        except OSError:
            pass
    return n


def adopted_settings(profile: Dict[str, Any]) -> Dict[str, Any]:
    """run_settings patch from a profile: only the ADOPTED candidate values."""
    out: Dict[str, Any] = {}
    for knob, entry in (profile.get("knobs") or {}).items():
        if not isinstance(entry, dict):
            continue
        adopted = entry.get("adopted")
        if adopted is not None:
            out[knob] = adopted
    return out


# ------------------------------------------------- serving integration (opt-in)

def calibration_mode(cfg: Any) -> str:
    """serve.calibration value normalized ("off" default | "auto")."""
    serve = getattr(cfg, "serve", None) or {}
    if not isinstance(serve, dict):
        return "off"
    mode = str(serve.get("calibration") or "off").strip().lower()
    return mode if mode in ("off", "auto") else "off"


def apply_calibration(cfg: Any, model_path: Any, *,
                      home: Optional[Path] = None) -> Dict[str, Any]:
    """Calibrated run_settings defaults for one launch — opt-in ONLY.

    Returns ``{}`` (today's exact behavior) unless ALL of:
      * ``serve.calibration: auto`` is set in the config, AND
      * ``model_path`` resolves to a readable GGUF, AND
      * a stored profile exists for this machine-id + model-hash.

    The returned map holds only ADOPTED knob values and is meant to be applied
    UNDERNEATH explicit config/instance run_settings — manual explicit values
    ALWAYS override calibrated ones (callers merge: calibration first, then
    config, then per-instance overrides)."""
    if calibration_mode(cfg) != "auto":
        return {}
    mhash = model_hash(model_path)
    if not mhash:
        return {}
    prof = load_profile(machine_id(), mhash, home=home)
    if not prof:
        return {}
    return adopted_settings(prof)


# ------------------------------------------------------------ measurement

def noise_floor_pct(nf_record: Optional[Dict[str, Any]]) -> Optional[float]:
    """Spread percentage of a persisted noise-floor record (gen preferred)."""
    if not nf_record:
        return None
    for key in ("generation", "prompt_processing"):
        spread = (nf_record.get(key) or {}).get("spread_pct")
        if isinstance(spread, (int, float)):
            return float(spread)
    return None


def ensure_noise_floor(measure_fn: "MeasureFn", baseline_settings: Dict[str, Any],
                       *, runs: int = DEFAULT_SWEEP_RUNS,
                       log: Callable[[str], None] = print,
                       ) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
    """The newest persisted noise floor, or a fresh estimate (persisted).

    Mirrors ``sable bench --noise-floor``: repeat the BASELINE settings and
    persist the record as kind=noise_floor so later runs (and WPs) share it."""
    nf = bench.load_noise_floor()
    if nf is not None:
        return nf, noise_floor_pct(nf)
    log("no persisted noise floor — measuring one from the baseline settings…")
    rec = measure_fn(dict(baseline_settings), max(runs, 3), "calibrate-noise-floor")
    rec["kind"] = bench.KIND_NOISE_FLOOR
    saved = bench.save_result(rec)
    pct = noise_floor_pct(rec)
    log(f"noise floor: spread={pct if pct is not None else 'n/a'}% "
        f"(saved {saved.name})")
    return rec, pct


def _tested_as(knob: str, value: Any, settings: Dict[str, Any],
               gguf_path: Optional[str]) -> Optional[str]:
    """What a speculative candidate ACTUALLY exercised.

    run_args.speculative_args() degrades mtp -> ngram-mod when no drafter GGUF
    exists; we re-derive the emitted flags to record the truth instead of
    claiming the requested mode was measured."""
    if knob != "speculative_type" or not value or str(value) == "off":
        return None
    from . import run_args

    mode = run_args.canonical_spec_mode(value)
    if not mode or "mtp" not in (mode or ""):
        return str(mode) if mode else None
    emitted = run_args.speculative_args(dict(settings), gguf_path=gguf_path)
    if "--model-draft" not in emitted and "draft-mtp" not in emitted:
        return "ngram (mtp unavailable)"
    return str(mode)


#: measure_fn(settings, runs, label) -> WP1 bench record. Injected by tests;
#: the default implementation spawns llama-server with those settings.
MeasureFn = Callable[[Dict[str, Any], int, str], Dict[str, Any]]


def _spawn_for_settings(cfg: Any, gguf_path: str,
                        settings: Dict[str, Any]) -> contextlib.AbstractContextManager:
    """Spawn an sable-owned llama-server with exactly these run_settings.

    Uses the canonical Phase-6 instance machinery under a dedicated crash-loop
    key ("calibrate") so candidate failures never poison the real default
    instance's respawn budget."""
    import dataclasses

    from . import instances as srv_inst
    from .supervisor import clear_crash_loop, start_instance_server

    @contextlib.contextmanager
    def _ctx():
        cfg2 = dataclasses.replace(cfg)
        cfg2.gguf_path = str(gguf_path)
        cfg2.attach_url = None
        cfg2.run_settings = dict(settings)
        inst = srv_inst.default_instance(cfg2)
        inst["id"] = "calibrate"
        inst["name"] = "calibrate"
        inst["default"] = False
        inst["run_settings"] = dict(settings)
        clear_crash_loop("calibrate")  # each candidate starts with a clean budget
        handle = start_instance_server(cfg2, inst)
        try:
            yield handle.base_url, handle.model_name
        finally:
            handle.stop()

    return _ctx()


def default_measure_fn(cfg: Any, gguf_path: str) -> MeasureFn:
    """Real measurement: spawn llama-server with the settings, bench it."""

    def measure(settings: Dict[str, Any], runs: int, label: str
                ) -> Dict[str, Any]:
        with _spawn_for_settings(cfg, gguf_path, settings) as (url, model):
            return bench.run_bench(url, model, runs=runs, label=label)

    return measure


# ------------------------------------------------------------------ sweep

def resolve_grid(knobs: Optional[Sequence[str]] = None, *,
                 quick: bool = False) -> Dict[str, List[Any]]:
    """Candidate grid for a run: preset (full or --quick), filtered by --knobs.

    Raises ValueError on unknown knob names (CLI maps that to exit 2)."""
    grid = dict(QUICK_GRID if quick else KNOB_GRID)
    if knobs:
        unknown = [k for k in knobs if k not in KNOB_GRID]
        if unknown:
            raise ValueError(
                "unknown knob(s): " + ", ".join(unknown)
                + " (known: " + ", ".join(KNOB_ORDER) + ")")
        grid = {k: v for k, v in grid.items() if k in knobs} or \
               {k: KNOB_GRID[k] for k in knobs}
    return {k: grid[k] for k in KNOB_ORDER if k in grid}


def run_sweep(cfg: Any, *, gguf_path: str,
              runs: int = DEFAULT_SWEEP_RUNS,
              knobs: Optional[Sequence[str]] = None,
              quick: bool = False,
              measure_fn: Optional[MeasureFn] = None,
              home: Optional[Path] = None,
              log: Callable[[str], None] = print,
              ) -> Dict[str, Any]:
    """Coordinate-wise autotune sweep; saves and returns the profile.

    ``measure_fn`` is the injection seam for tests (no GPU required): a callable
    ``(settings, runs, label) -> WP1 bench record``. The default spawns a real
    llama-server per candidate via the supervisor machinery."""
    grid = resolve_grid(knobs, quick=quick)
    mhash = model_hash(gguf_path)
    if not mhash:
        raise CalibrateError(f"cannot hash model {gguf_path!r} (unreadable?)")
    measure_fn = measure_fn or default_measure_fn(cfg, str(gguf_path))

    baseline_settings = dict(getattr(cfg, "run_settings", None) or {})

    nf_record, nf_pct = ensure_noise_floor(
        measure_fn, baseline_settings, runs=runs, log=log)
    threshold = max(MIN_GAIN_PCT, nf_pct or 0.0)

    log(f"baseline bench ({runs} runs)…")
    best_settings = dict(baseline_settings)
    try:
        best_record = measure_fn(dict(best_settings), runs, "calibrate-baseline")
    except Exception as e:
        # Nothing measurable at all (no server reachable): a failed
        # calibration, not a partial one — the CLI maps this to exit 1.
        raise CalibrateError(
            f"baseline measurement failed — nothing could be measured: {e}"
        ) from e

    knobs_out: Dict[str, Any] = {}
    for knob in grid:
        entry: Dict[str, Any] = {"candidates": [], "adopted": None}
        for value in grid[knob]:
            current = best_settings.get(knob, KNOB_DEFAULTS.get(knob))
            row: Dict[str, Any] = {"value": value}
            if value == current:
                row["verdict"] = VERDICT_BASELINE
                entry["candidates"].append(row)
                continue
            cand_settings = {**best_settings, knob: value}
            tested_as = _tested_as(knob, value, cand_settings, str(gguf_path))
            if tested_as:
                row["tested_as"] = tested_as
            try:
                rec = measure_fn(cand_settings, runs, f"calibrate-{knob}-{value}")
            except Exception as e:  # server failed to start / crashed mid-bench
                row["verdict"] = VERDICT_ERROR
                row["error"] = str(e)[:300]
                log(f"{knob}={value}: ERROR ({row['error']}) — skipping, "
                    "continuing sweep")
                entry["candidates"].append(row)
                continue
            gen_med = (rec.get("generation") or {}).get("median")
            cmp_res = bench.compare_records(best_record, rec)
            row["gen_toks_per_s"] = gen_med
            sp = (cmp_res.get("generation") or {}).get("speedup")
            if sp is None:
                sp = (cmp_res.get("prompt_processing") or {}).get("speedup")
            if sp is not None:
                row["delta_pct"] = round(sp * 100.0, 3)
            if not cmp_res["valid"]:
                # Token-identity rule: faster-but-different tokens NEVER win.
                row["verdict"] = VERDICT_INVALID
                row["reasons"] = cmp_res["reasons"]
            elif (isinstance(gen_med, (int, float))
                  and sp is not None
                  and sp * 100.0 > threshold):
                row["verdict"] = VERDICT_ADOPTED
                entry["adopted"] = value
                best_settings = cand_settings
                best_record = rec
            else:
                row["verdict"] = VERDICT_REJECTED
                if sp is not None and abs(sp * 100.0) <= threshold:
                    row["reason"] = (f"gain within threshold "
                                     f"(> max({MIN_GAIN_PCT}%, "
                                     f"{nf_pct if nf_pct is not None else 0}%)"
                                     " required)")
            entry["candidates"].append(row)
        knobs_out[knob] = entry

    profile: Dict[str, Any] = {
        "profile_version": PROFILE_VERSION,
        "machine_id": machine_id(),
        "model_hash": mhash,
        "model_path": str(gguf_path),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runs_per_candidate": runs,
        "threshold_pct": threshold,
        "noise_floor": {
            "spread_pct": nf_pct,
            "record": str((nf_record or {}).get("_path")
                          or (nf_record or {}).get("label", "")) or None,
            "source": "persisted" if (nf_record or {}).get("kind") == \
                bench.KIND_NOISE_FLOOR else "measured",
        },
        "baseline_settings": baseline_settings,
        "adopted_settings": adopted_settings(
            {"knobs": knobs_out}),
        "knobs": knobs_out,
    }
    save_profile(profile, home=home)
    return profile


# ---------------------------------------------------------------- rendering

def format_summary(profile: Dict[str, Any]) -> str:
    """Human-readable sweep table: knob | candidate | tok/s | delta | verdict."""
    nf_pct = (profile.get("noise_floor") or {}).get("spread_pct")
    thr = profile.get("threshold_pct", MIN_GAIN_PCT)
    lines = [
        f"sable calibrate — machine={str(profile.get('machine_id', ''))[:16]} "
        f"model={str(profile.get('model_hash', ''))[:12]}… "
        f"(runs/candidate={profile.get('runs_per_candidate')})",
        f"noise floor: {nf_pct if nf_pct is not None else 'n/a'}%  |  adoption "
        f"threshold: >{thr:.1f}% AND token-identical",
        "",
        f"{'knob':<18} {'candidate':<26} {'median tok/s':>12} "
        f"{'delta%':>8}  verdict",
    ]
    for knob, entry in (profile.get("knobs") or {}).items():
        for row in entry.get("candidates", []):
            value = str(row.get("value"))
            if row.get("tested_as"):
                value += f" [{row['tested_as']}]"
            med = row.get("gen_toks_per_s")
            delta = row.get("delta_pct")
            verdict = str(row.get("verdict", "?"))
            if row.get("reasons"):
                verdict += " (tokens changed)"
            if row.get("error"):
                verdict += f" ({str(row['error'])[:40]})"
            lines.append(
                f"{knob:<18} {value:<26} "
                + (f"{med:>12.2f}" if isinstance(med, (int, float))
                   else f"{'-':>12}")
                + (f"{delta:>+8.1f}" if isinstance(delta, (int, float))
                   else f"{'-':>8}")
                + f"  {verdict}")
    adopted = profile.get("adopted_settings") or {}
    lines.append("")
    if adopted:
        parts = ", ".join(f"{k}={v}" for k, v in sorted(adopted.items()))
        lines.append(f"adopted: {parts}")
    else:
        lines.append("adopted: nothing beat the noise floor — current "
                     "settings kept")
    lines.append("(serving honors this ONLY with serve.calibration: auto)")
    return "\n".join(lines)


def format_profile_show(profile: Dict[str, Any]) -> str:
    """`--show` rendering of one stored profile."""
    adopted = profile.get("adopted_settings")
    if adopted is None:
        adopted = adopted_settings(profile)
    adopted_json = json.dumps(adopted or {}, sort_keys=True)
    lines = [
        f"profile: {profile.get('_path', '(unsaved)')}",
        f"machine_id:  {profile.get('machine_id')}",
        f"model_hash:  {profile.get('model_hash')}",
        f"model_path:  {profile.get('model_path')}",
        f"timestamp:   {profile.get('timestamp')}",
        f"noise floor: {(profile.get('noise_floor') or {}).get('spread_pct')}% "
        f"(record: {(profile.get('noise_floor') or {}).get('record')})",
        f"adopted:     {adopted_json}",
    ]
    for knob, entry in (profile.get("knobs") or {}).items():
        verdicts = ", ".join(
            f"{c.get('value')}={c.get('verdict')}"
            + (f"[{c['tested_as']}]" if c.get("tested_as") else "")
            for c in entry.get("candidates", []))
        lines.append(f"  {knob}: adopted={entry.get('adopted')} "
                     f"({verdicts})")
    return "\n".join(lines)

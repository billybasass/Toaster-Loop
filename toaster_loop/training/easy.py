"""Easy training — one-profile turnkey loop (dashboard "Easy training" page).

The premise: a user has already created an agent profile and bound it to a
managed server instance (the profile's ``server_instance`` decides which model
serves it). Easy training closes the rest of the loop for exactly that
profile: chat traces are stamped with the profile name, curation scopes to
them, and one click runs the full iterate pipeline
(curate -> register -> train -> export -> validate -> promote) with optimal
settings — then optionally hot-deploys the promoted model back onto the
profile's server.

Design notes (matching the project's invariants):

* State lives in files, not daemons. Config is the ``easy_training:`` YAML
  section (see core/config.py); per-home run state is
  ``<home>/easy_training/state.json``. Autonomous triggering stays on the
  existing cron entrypoint (``toaster-loop train auto-check --trigger``), which
  this module also scopes; the dashboard only *notifies* readiness and offers
  one-click runs.
* "Optimal settings" is a documented preset, not magic: SFT+DPO curation with
  failures kept as preference negatives, the standard eval split, and the
  training.auto_* thresholds (overridable per easy config). Everything stays
  editable in Settings afterwards — enabling easy training never hides knobs.
* The preset also arms the loop's *quality machinery* (added after this module
  first shipped): ``validation.shadow_enabled`` so every run gets a shadow
  baseline comparison before promotion (warn-only; and the verdict is what the
  cron tier policy in ``validation.min_tier_auto_deploy`` needs to auto-promote
  at all), and ``goldens_mine.enabled`` so 👍-thumbed chats keep distilling
  into golden gate tasks that both validation and RFT read automatically.
* Deliberately NOT in the preset: ``training.backend: streamed`` and
  ``rank_scaling`` (GPU effects unmeasured — cardinal rule 2 forbids shipping
  them as "optimal" without a measured win), ``rft.enabled`` and
  ``janitor.enabled`` (real GPU spend / archival side effects; explicit opt-in
  stays with the human).
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, IO, List, Optional

logger = logging.getLogger("toaster_loop.easy")

EASY_DIRNAME = "easy_training"
STATE_FILENAME = "state.json"
LOOP_LOCK_FILENAME = "loop.lock"

try:  # POSIX advisory locking; absent on Windows (lock degrades to no-op).
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX dev environments
    fcntl = None  # type: ignore[assignment]


def _coerce(value: Any, cast, name: str) -> Optional[Any]:
    """Cast a YAML knob, degrading to None on garbage instead of raising.

    A hand-edited ``min_traces: lots`` must not 500 every endpoint that
    parses this section (GET /api/easy, chat trace stamping, cron check);
    log and fall back to the shared training.auto_* threshold."""
    if value is None:
        return None
    try:
        return cast(value)
    except (TypeError, ValueError):
        logger.warning("easy_training config: %s=%r is not a valid %s; "
                       "ignoring it", name, value, cast.__name__)
        return None


@dataclass
class EasyTrainingConfig:
    """`easy_training:` YAML section (all keys optional; defaults off).

    Multi-agent loops: ``profiles`` lists every agent in the loop; ``profile``
    stays the primary one (legacy key — old configs and readers keep working
    unchanged). ``from_yaml`` normalizes so a legacy config with only
    ``profile: coder`` yields ``all_profiles == ["coder"]`` exactly as before,
    and a hand-edited garbage ``profiles`` entry degrades to being skipped
    rather than raising (same philosophy as the numeric knobs above)."""

    enabled: bool = False
    profile: Optional[str] = None
    # Every agent in the loop (may include `profile`). Empty => single-agent
    # legacy mode driven by `profile` alone.
    profiles: List[str] = field(default_factory=list)
    auto_deploy: bool = False
    # Threshold overrides; None => fall back to training.auto_min_traces /
    # training.auto_min_score (one source of truth with the Schedule panel).
    min_traces: Optional[int] = None
    min_score: Optional[float] = None

    @staticmethod
    def _parse_profiles(value: Any) -> List[str]:
        """Lenient list-of-names parse: skip non-strings/blank, warn once."""
        if value is None:
            return []
        if not isinstance(value, list):
            logger.warning("easy_training config: profiles=%r is not a list; "
                           "ignoring it", value)
            return []
        out: List[str] = []
        for item in value:
            if not isinstance(item, str):
                logger.warning("easy_training config: skipping non-string "
                               "profiles entry %r", item)
                continue
            name = item.strip()
            if not name:
                continue
            if name not in out:
                out.append(name)
        return out

    @staticmethod
    def from_yaml(section: Any) -> "EasyTrainingConfig":
        sec = section if isinstance(section, dict) else {}
        known = {f for f in EasyTrainingConfig.__dataclass_fields__}
        unknown = sorted(k for k in sec if k not in known)
        if unknown:  # typo'd knobs used to vanish silently
            logger.warning("easy_training config: ignoring unknown key(s) %s "
                           "(known: %s)", ", ".join(unknown),
                           ", ".join(sorted(known)))
        mt = _coerce(sec.get("min_traces"), int, "min_traces")
        ms = _coerce(sec.get("min_score"), float, "min_score")
        return EasyTrainingConfig(
            enabled=bool(sec.get("enabled", False)),
            profile=str(sec["profile"]) if sec.get("profile") else None,
            profiles=EasyTrainingConfig._parse_profiles(sec.get("profiles")),
            auto_deploy=bool(sec.get("auto_deploy", False)),
            min_traces=mt,
            min_score=ms,
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "profile": self.profile,
            "profiles": list(self.profiles),
            "auto_deploy": self.auto_deploy,
            "min_traces": self.min_traces,
            "min_score": self.min_score,
        }

    @property
    def all_profiles(self) -> List[str]:
        """Ordered unique agent names: primary first, then the rest.

        Legacy single-profile configs (``profiles`` empty) return exactly
        ``[profile]`` — every consumer of the old ``ec.profile`` can migrate
        to this without behavior change for existing setups."""
        out: List[str] = []
        if self.profile:
            out.append(self.profile)
        for p in self.profiles:
            if p not in out:
                out.append(p)
        return out

    def for_profile(self, name: str) -> "EasyTrainingConfig":
        """Single-agent view of this config with ``profile`` bound to ``name``.

        Readiness/run submission reuse the single-profile code paths by
        projecting one agent out of the loop; thresholds and flags carry over."""
        return EasyTrainingConfig(
            enabled=self.enabled, profile=name, profiles=[],
            auto_deploy=self.auto_deploy,
            min_traces=self.min_traces, min_score=self.min_score)


# The optimal preset applied (as a config patch) when easy training is enabled.
# Training intent + curation quality + the validation/mining machinery the
# loop grew after this module first shipped — never destructive, and every key
# is a recognized section field so Settings keeps round-tripping it.
OPTIMAL_CURATION = {
    "min_score": 0.5,          # keep heuristic-good traces for SFT
    "include_failures": True,  # failures become DPO negatives only
    "formats": "both",         # emit sft.jsonl AND dpo.jsonl
}
OPTIMAL_TRAINING = {
    "auto_enabled": True,      # intent flag for the cron entrypoint
    "eval_fraction": 0.1,      # held-out eval split on every run
    "eval_gate": "warn",       # promotion regression gate (soft by default)
}
# Shadow-baseline comparison before every promotion (warn-only unless the user
# flips validation.shadow_blocks_promotion). This is also what makes the cron
# path work end-to-end: unattended auto-promotion requires derived tier >=
# validation.min_tier_auto_deploy (default T2 "shadow-proven"), so WITHOUT this
# key an easy loop driven by `train auto-check --trigger` could train forever
# and never auto-promote. Failures to run the shadow job never block
# promotion (pipeline stage 5b) — arming it is upside-only.
OPTIMAL_VALIDATION = {
    "shadow_enabled": True,
}
# 👍-thumbed chats keep distilling into golden gate tasks (deduped ledger; the
# suite feeds validation AND rft automatically). Pure stdlib I/O unless the
# user separately turns on goldens_mine.use_judge. The easy loop runs one
# mining pass before each training run (see mine_goldens_pass) so freshly
# thumbed behavior gates the model being promoted right now.
OPTIMAL_GOLDENS_MINE = {
    "enabled": True,
}


def optimal_config_patches(ec: EasyTrainingConfig) -> List[Dict[str, Any]]:
    """Ordered list of {section, patch} bodies for PUT /api/config.

    One section-patch per top-level section: the merge semantics of the
    config endpoint only deep-merge INSIDE the named section, so patching
    curation:/training:/validation:/goldens_mine: separately preserves every
    unrelated key the user already set there (notably validation.golden_tasks).
    """
    patch: Dict[str, Any] = {
        "enabled": True,
        "profile": ec.profile,
        "auto_deploy": bool(ec.auto_deploy),
    }
    # Only write `profiles` when there actually ARE multiple agents — a legacy
    # single-profile enable must produce the exact same config file as before.
    if len(ec.all_profiles) > 1:
        patch["profiles"] = list(ec.all_profiles)
    return [
        {"section": "easy_training", "patch": patch},
        {"section": "curation", "patch": dict(OPTIMAL_CURATION)},
        {"section": "training", "patch": dict(OPTIMAL_TRAINING)},
        {"section": "validation", "patch": dict(OPTIMAL_VALIDATION)},
        {"section": "goldens_mine", "patch": dict(OPTIMAL_GOLDENS_MINE)},
    ]


def mine_goldens_pass(cfg: Any, home: Path) -> Dict[str, Any]:
    """One best-effort golden-mining pass for the easy loop (never raises).

    The 👍 hook is deliberately not wired into the recorder/feedback hot paths
    (curate/mine_goldens.py), and the CLI command requires an explicit call —
    the easy loop's turnkey promise means the loop itself should keep the
    golden suite fed. Runs in-process like the job kind (stdlib JSONL +
    SQLite; the judge is contacted only when goldens_mine.use_judge is on,
    which this pass does not wire). The master switch is honored: disabled =>
    skipped without touching anything. Returns a small report dict.
    """
    from ..core.config import GoldensMineConfig

    try:
        gsec = getattr(cfg, "goldens_mine", None)
        gcfg = GoldensMineConfig.from_yaml(
            {"goldens_mine": gsec if isinstance(gsec, dict) else {}})
        if not gcfg.enabled:
            return {"ok": False, "skipped": "goldens_mine.enabled is false"}
        from ..curate.mine_goldens import run_mine_goldens
        from ..store.db import TraceStore

        vsec = getattr(cfg, "validation", None)
        vsec = vsec if isinstance(vsec, dict) else {}
        gt = vsec.get("golden_tasks")
        existing = [t for t in (gt or []) if isinstance(t, dict)]
        store = TraceStore(Path(home) / "buffer" / "toaster_loop.db")
        try:
            res = run_mine_goldens(store, Path(home), gcfg,
                                   existing_goldens=existing)
        finally:
            store.close()
        return {"ok": True, **res.to_dict()}
    except Exception as e:  # mining is a bonus, never a run-breaker
        logger.warning("easy training: golden-mining pass failed: %s", e)
        return {"ok": False, "skipped": f"{type(e).__name__}: {e}"}


# ------------------------------------------------------------------ state

def state_path(home: Path) -> Path:
    return Path(home) / EASY_DIRNAME / STATE_FILENAME


def load_state(home: Path) -> Dict[str, Any]:
    """Per-home easy-training run state; {} when none was ever written."""
    p = state_path(home)
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


# Guards the read-modify-write in save_state: the iterate worker thread
# (_easy_after_run) and request threads (h_easy_run's last_start) both patch
# the same file; interleaving would silently drop one update.
_state_lock = threading.Lock()


def save_state(home: Path, patch: Dict[str, Any]) -> Dict[str, Any]:
    """Merge ``patch`` into the state file atomically; returns new state."""
    with _state_lock:
        p = state_path(home)
        p.parent.mkdir(parents=True, exist_ok=True)
        state = load_state(home)
        state.update(patch)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2))
        os.replace(tmp, p)
        return state


# ------------------------------------------------------- loop lock (cross-proc)

def loop_lock_path(home: Path) -> Path:
    return Path(home) / EASY_DIRNAME / LOOP_LOCK_FILENAME


def acquire_loop_lock(home: Path) -> Optional[IO[str]]:
    """Non-blocking advisory lock for one easy-loop run ACROSS processes.

    The dashboard (submit_iterate thread) and the cron entrypoint
    (`train auto-check --trigger` -> cmd_iterate) are separate processes; a
    threading.Lock cannot serialize them, so both could pass readiness and
    double-claim traces. Whoever holds this file lock owns the loop; the
    handle must stay open for the run's duration (closing releases).
    Returns None when another process holds it. Windows (no fcntl) degrades
    to always-acquire — same behavior as before this guard existed.
    Contract: a file handle (possibly unlocked on non-POSIX) means ACQUIRED;
    None strictly means "another run holds the loop"."""
    p = loop_lock_path(home)
    p.parent.mkdir(parents=True, exist_ok=True)
    fh = open(p, "a+", encoding="utf-8")
    if fcntl is None:  # pragma: no cover - non-POSIX: keep handle, no lock
        return fh
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    return fh


def release_loop_lock(handle: Optional[IO[str]]) -> None:
    if handle is not None:
        try:
            handle.close()
        except OSError:
            pass


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------- validation

def validate_profile(profiles: List[Dict[str, Any]],
                     name: Optional[str]) -> Dict[str, Any]:
    """Return the profile dict or raise ValueError with a user-facing reason.

    The profile must exist AND be bound to a server instance — that binding is
    what ties easy training to "the model this profile serves", so the loop's
    output can be deployed back where the user actually works.
    """
    if not name:
        raise ValueError("pick a profile first")
    for p in profiles:
        if p.get("name") == name:
            if not str(p.get("server_instance") or "").strip():
                raise ValueError(
                    f"profile “{name}” is not bound to a server — open the "
                    "Profiles panel and select a server instance for it "
                    "first (easy training needs to know which model to "
                    "train on and deploy back to)")
            return p
    raise ValueError(f"no profile {name}")


# ------------------------------------------------------------- readiness

def evaluate_readiness(store: Any, ec: EasyTrainingConfig, tc: Any,
                       gate: Dict[str, Any]) -> Dict[str, Any]:
    """Buffer + gate readiness for the easy profile (never raises on empty).

    Mirrors should_auto_train's shape so panels reuse the same rendering, but
    scoped to the easy profile's stamped traces. ``gate`` comes from
    scheduler.resource_gate_ok (VRAM / AC power) — callers own that probe.
    """
    from .scheduler import should_auto_train

    min_traces = ec.min_traces if ec.min_traces is not None \
        else int(getattr(tc, "auto_min_traces", 50))
    min_score = ec.min_score if ec.min_score is not None \
        else float(getattr(tc, "auto_min_score", 0.5))
    d = should_auto_train(store, min_traces=min_traces, min_score=min_score,
                          agent_profile=ec.profile if ec.enabled else None)
    d["gate"] = gate
    d["needs_training"] = bool(d["trigger"] and gate.get("ok", True))
    # Panel-friendly mirror of h_train_check's shape (that endpoint calls the
    # scoped count "unclaimed"; keep one name across both readiness views).
    d["threshold"] = {"min_traces": d["min_traces"],
                      "min_score": d["min_score"]}
    d["unclaimed"] = d["unclaimed_traces"]
    return d

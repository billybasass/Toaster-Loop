"""Training scheduler — single-GPU arbiter over the jobs table (Phase 4).

Model: FIFO queue persisted in SQLite (`jobs`). One training/export job runs at
a time; a second submit while one is running just enqueues. Workers are
subprocesses (`python -m toaster_loop.training.worker` by default, injectable for
tests via ``worker_command``); the scheduler marks state transitions and
reaps crashed workers (a worker killed -9 leaves its job `running` with a dead
pid -> reaped to `error` on the next scheduler pass, keeping registry/serving
consistent per ROADMAP acceptance).

Resource gate (best-effort): when nvidia-smi (NVIDIA) or rocm-smi (AMD) exists,
refuse to start until free VRAM >= training.min_free_vram_mib AND >= the job's
predicted footprint (core/vram.py — inspiration pass 3 #20; unknown dims skip
the prediction, never guess zero). When a battery is present, require AC power
unless training.require_ac_power is false. No GPU tooling / no battery => gate
passes (dev machines, VMs).

`--auto` mode stub: `should_auto_train()` checks the buffer for >= N unclaimed
traces with mean score >= threshold; cron wiring is Phase 5+.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core import proclock  # cross-process locks: flock/msvcrt/pidfile
from ..registry.registry import INPROCESS_JOB_KINDS, JobRow, ModelRegistry
from ..store.db import TraceStore

logger = logging.getLogger("toaster_loop.training")

# Device-slot bookkeeping shared by all Scheduler instances in this process
# (the dashboard creates one per request/thread). Guards claim+persist so two
# threads can never take the same slot; entries live only while a worker runs.
_SLOT_LOCK = threading.Lock()
_INFLIGHT_CLAIMS: Dict[str, int] = {}

#: JobDispatcher tick interval in seconds. How often the dashboard's dispatcher
#: loop reaps stale workers, drains queued jobs that no submit thread picked up,
#: and re-attempts gate-blocked jobs (the queue-with-reason wait from #20).
#: Sized so a stranded job starts within one tick while keeping the periodic
#: os.kill probe + DB poll cheap next to TrainingConfig.poll_interval_s.
DISPATCH_TICK_S = 5.0


def _slot_lock_path(db_path: Path, slot: int) -> Path:
    """Sidecar lock file for one device slot (next to the daemon DB)."""
    return Path(str(db_path) + f".slot{slot}.lock")


def _try_slot_lock(db_path: Path, slot: int):
    """Non-blocking cross-process lock on a device slot, held until
    _release_slot_lock.

    Returns the held handle when acquired, or None when another process holds
    the slot (its worker is running there). Backend by platform via
    core/proclock.py: POSIX flock (kernel-dropped on death — unchanged
    semantics); Windows msvcrt byte-range locking (previously a silent no-op
    sentinel here, which let two NT processes dispatch onto one GPU); pid +
    liveness lockfile as last resort. The in-process dict bookkeeping still
    guards slots within one process on every backend.
    """
    return proclock.acquire(_slot_lock_path(db_path, slot))


def _release_slot_lock(handle) -> None:
    proclock.release(handle)


# ------------------------------------------------------------- config object

@dataclass
class TrainingConfig:
    """`training:` YAML section (config.example.yaml documents every key)."""

    base_model: Optional[str] = None          # HF id or local HF-format path
    python: Optional[str] = None              # interpreter for GPU workers
    min_free_vram_mib: int = 0                # 0 => gate disabled
    require_ac_power: bool = False            # best-effort when battery present
    poll_interval_s: float = 1.0
    # Hard cap on one worker subprocess. Default 0 = NO timeout: long LoRA
    # runs (multi-hour, e.g. a 27B at 16k context) legitimately exceed any
    # fixed cap and were being SIGTERM'd mid-training (jobs 38/39 died at
    # step 430/444 with the run healthy). A hung worker is still bounded by
    # reap_stale once its pid dies, and an operator can re-arm a cap by
    # setting a positive value: the timeout path SIGTERMs, waits a 10 s
    # grace, then SIGKILLs (see _spawn_worker).
    worker_timeout_s: float = 0.0
    # Multi-GPU: >1 runs that many training jobs concurrently, one per device
    # slot (CUDA/HIP_VISIBLE_DEVICES). Keep 1 on unified-memory single-GPU
    # boxes. gpu_ids pins the slot -> physical device mapping (e.g. [0,1]).
    max_parallel_jobs: int = 1
    gpu_ids: Optional[list] = None
    auto_enabled: bool = False
    auto_min_traces: int = 50
    auto_min_score: float = 0.5
    gguf_quant: str = "q4_k_m"
    # Backlog item #4: eval split + promotion gate. eval_fraction/eval_seed are
    # injected into the resolved train.yaml so the worker splits deterministically
    # (min-rows floor lives in worker.maybe_eval_split — tiny datasets skip eval).
    eval_fraction: float = 0.1
    eval_seed: int = 42
    # Soft promotion gate: "warn" (default) logs when eval_loss regresses more
    # than max_eval_regression_pct vs the parent model's recorded eval_loss;
    # "block" refuses promotion. Only applies when BOTH eval losses exist.
    eval_gate: str = "warn"            # warn | block
    max_eval_regression_pct: float = 25.0
    # Optional argv prefix for the worker (tests inject a stub; power users
    # can point at a venv launcher). Env TOASTER_LOOP_WORKER_COMMAND (shell-split)
    # overrides. None => [python, "-m", "toaster_loop.training.worker"].
    worker_command: Optional[list] = None
    # WP10 step 2 (opt-in): training backend forwarded into the resolved
    # train.yaml so the spawned WORKER selects the implementation. None
    # (default) injects nothing => the worker's default "peft" path runs,
    # byte-identical to pre-WP10. "streamed" routes sft jobs through the
    # streamed-LoRA runner. The scheduler spawn path itself is unchanged.
    backend: Optional[str] = None

    @staticmethod
    def from_yaml(data: Dict[str, Any]) -> "TrainingConfig":
        sec = data.get("training") if isinstance(data, dict) else None
        if not isinstance(sec, dict):
            return TrainingConfig()
        known = {f for f in TrainingConfig.__dataclass_fields__}
        # A typo'd knob (min_free_vram_mb vs _mib) used to vanish silently;
        # surface it so misconfiguration is visible instead of defaulting.
        unknown = sorted(k for k in sec if k not in known)
        if unknown:
            logger.warning(
                "training config: ignoring unknown key(s) %s (known: %s)",
                ", ".join(unknown), ", ".join(sorted(known)))
        kwargs = {k: v for k, v in sec.items() if k in known and v is not None}
        return TrainingConfig(**kwargs)


def load_yaml_section(path: Optional[str], section: str) -> Dict[str, Any]:
    """Read one top-level section from an toaster_loop YAML config (no dataclass)."""
    if not path:
        return {}
    import yaml

    p = Path(path).expanduser()
    if not p.is_file():
        return {}
    data = yaml.safe_load(p.read_text()) or {}
    sec = data.get(section) if isinstance(data, dict) else None
    return sec if isinstance(sec, dict) else {}


# ----------------------------------------------------------- resource gates

def _nvidia_free_mib(device: Optional[str] = None) -> Optional[int]:
    """Free VRAM (MiB); ``device`` pins the query to one physical index.

    nvidia-smi enumerates in the same order torch/CUDA does, so slot device
    ids map 1:1 onto ``index`` rows. Unknown index falls back to global min
    (honest "at least this much is free somewhere" never over-admits)."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        if device is None:
            out = subprocess.run(
                [exe, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if out.returncode != 0:
                return None
            values = [int(l.strip()) for l in out.stdout.splitlines() if l.strip()]
            return min(values) if values else None
        out = subprocess.run(
            [exe, "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return None
        for line in out.stdout.splitlines():
            cols = [c.strip() for c in line.split(",")]
            if len(cols) >= 2 and cols[0] == str(device) and cols[1].isdigit():
                return int(cols[1])
        return None  # device not present => unknown, gate skips (never guess)
    except Exception:
        return None


def _rocm_free_mib(device: Optional[str] = None) -> Optional[int]:
    """Free VRAM in MiB via rocm-smi (AMD).

    Uses `--showmeminfo vram --json` when available (per-GPU VRAM total/used
    in bytes), else the plain text table. Unified-memory APUs (Strix Halo)
    report the shared pool, which is exactly what the gate should watch.
    ``device`` selects one card (JSON keys are ``card0``/``card1``…, text
    rows ``GPU[0]``…); unknown device => None (never guess)."""
    exe = shutil.which("rocm-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--showmeminfo", "vram", "--json"],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode == 0 and out.stdout.strip():
            data = json.loads(out.stdout)

            def _free_of(gpu: dict) -> Optional[int]:
                tot_b = gpu.get("VRAM Total Memory (B)")
                used_b = gpu.get("VRAM Total Used Memory (B)")
                if tot_b is not None and used_b is not None:
                    return (int(tot_b) - int(used_b)) // (1024 * 1024)
                return None

            frees: List[int] = []
            for key, gpu in data.items():
                if not isinstance(gpu, dict):
                    continue
                if device is not None and key != f"card{device}":
                    continue
                f = _free_of(gpu)
                if f is not None:
                    frees.append(f)
            if device is not None:
                return frees[0] if frees else None
            return min(frees) if frees else None
    except Exception:
        pass
    # Text fallback: "GPU[0]\t\t: VRAM Total Memory (B): N" / "... Used ...: M"
    try:
        out = subprocess.run(
            [exe, "--showmeminfo", "vram"],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode != 0:
            return None
        totals: Dict[str, int] = {}
        used: Dict[str, int] = {}
        import re as _re
        pat = _re.compile(r"GPU\[(\d+)\].*?VRAM Total (Used )?Memory \(B\):\s*(\d+)")
        for m in pat.finditer(out.stdout):
            idx, is_used, val = m.group(1), m.group(2), int(m.group(3))
            (used if is_used else totals)[idx] = val
        keys = ([k for k in totals if k == str(device)] if device is not None
                else list(totals))
        frees = [(totals[k] - used.get(k, 0)) // (1024 * 1024) for k in keys]
        return min(frees) if frees else None
    except Exception:
        return None


def gpu_free_mib(device: Optional[str] = None) -> Optional[int]:
    """Free GPU memory in MiB (NVIDIA, AMD discrete, or unified pools);
    None when unavailable (skip).

    Unified-memory systems (Strix Halo & co. under TTM/carve-out setups)
    are checked FIRST via core.gpumem: rocm-smi/sysfs there report only the
    BIOS carve-out (512 MiB on a 128 GiB pool) and would refuse every job.
    The probe keys on the PCI class, so discrete-GPU machines never enter
    it and keep their honest vendor numbers.

    Tries each vendor tool in turn so a failing/slow probe on one (e.g. a
    busy rocm-smi under memory pressure) still falls through to the other,
    and the gate degrades to skip rather than blocking forever. ``device``
    restricts the reading to one physical device id (the training scheduler's
    per-slot gate); None keeps the historical min-across-all-GPUs semantics.
    """
    if device is None:
        try:
            from ..core import gpumem
            pool = gpumem.unified_pool()
            if pool is not None:
                return pool.free_mib
        except Exception:  # a probe failure must never block the gate
            logger.debug("unified pool probe failed", exc_info=True)
    for probe in (_nvidia_free_mib, _rocm_free_mib):
        # device=None keeps the exact zero-arg historical call (stub-friendly).
        free = probe() if device is None else probe(device)
        if free is not None:
            return free
    return None


def on_ac_power() -> Optional[bool]:
    """True/False from /sys/class/power_supply; None when no battery (skip)."""
    base = Path("/sys/class/power_supply")
    if not base.is_dir():
        return None
    try:
        entries = list(base.iterdir())
    except OSError:
        return None
    for e in entries:
        if (e / "type").is_file():
            try:
                t = (e / "type").read_text().strip().lower()
            except OSError:
                continue
            if t == "mains":
                try:
                    online = (e / "online").read_text().strip()
                    return online == "1"
                except OSError:
                    return True
    # no battery/mains info at all -> treat as desktop/skip
    return None


def resource_gate_ok(cfg: TrainingConfig,
                     estimated_need_mib: Optional[int] = None,
                     device: Optional[str] = None) -> Dict[str, Any]:
    """Evaluate the best-effort start gate. Returns {ok, reasons}.

    ``estimated_need_mib`` (inspiration pass 3 #20): predicted footprint of
    the job about to start. When provided it raises the VRAM bar to
    max(min_free_vram_mib, estimate) — dispatch-and-pray becomes
    queue-with-reason. None (unknown model dims / non-training kind) leaves
    the static floor semantics untouched.

    ``device`` (multi-GPU): restrict the VRAM reading to one physical device
    id instead of the min across all GPUs, so a busy GPU1 no longer blocks a
    job whose slot is GPU0. The scheduler only passes it when
    ``training.gpu_ids`` pins the slot map (opt-in); None keeps the exact
    historical global-minimum probe.
    """
    reasons: List[str] = []
    required = int(cfg.min_free_vram_mib or 0)
    if estimated_need_mib is not None:
        required = max(required, int(estimated_need_mib))
    if required > 0:
        free = gpu_free_mib(device) if device is not None else gpu_free_mib()
        if free is not None and free < required:
            what = (f"required {required} MiB"
                    f" (estimate {int(estimated_need_mib)} MiB)"
                    if estimated_need_mib is not None
                    else f"required {required} MiB")
            reasons.append(f"GPU free memory {free} MiB < {what}")
    if cfg.require_ac_power:
        ac = on_ac_power()
        if ac is False:
            reasons.append("on battery power (training.require_ac_power=true)")
    return {"ok": not reasons, "reasons": reasons}


# Job kinds whose worker consumes training VRAM (validate/curate don't).
_GPU_JOB_KINDS = frozenset({"sft", "dpo", "grpo"})


def job_vram_estimate_mib(job: JobRow,
                          python: Optional[str] = None) -> Optional[int]:
    """Predicted footprint for one queued job; None when not predictable.

    ``python`` (training.python) is forwarded to the estimator so the ROCm
    bitsandbytes capability probe runs in the interpreter that will actually
    train — the daemon itself has no torch, and probing the wrong environment
    silently downgrades every large-model estimate to bf16 pricing."""
    if job.kind not in _GPU_JOB_KINDS:
        return None
    try:
        from ..core.vram import estimate_from_train_yaml

        params = job.params if isinstance(job.params, dict) else {}
        est = estimate_from_train_yaml(
            params.get("train_yaml"), params.get("base_model") or None,
            python=python)
        return est.total_mib if est is not None else None
    except Exception:  # a broken/absent train.yaml must not block dispatch
        logger.debug("job VRAM estimate failed", exc_info=True)
        return None


# ------------------------------------------------------------- auto trigger

def should_auto_train(
    store: TraceStore,
    *,
    min_traces: int = 50,
    min_score: float = 0.5,
    agent_profile: Optional[str] = None,
) -> Dict[str, Any]:
    """`--auto` check: >= min_traces unclaimed traces with mean score >= threshold.

    Mean is over scored traces only (NULL scores never count as zero — Phase 2
    note). Returns the decision + inputs so callers can print/act on it.

    ``agent_profile`` (easy training): scope the buffer to one agent profile's
    traces (the recorder stamps the profile name into ``traces.agent_profile``
    for dashboard chat runs). None => whole buffer, same as before.
    """
    rows = store.query_for_curation(include_curated=False,
                                    agent_profile=agent_profile)
    scored = [r.outcome_score for r in rows if r.outcome_score is not None]
    mean = (sum(scored) / len(scored)) if scored else None
    enough = len(rows) >= min_traces
    good = mean is not None and mean >= min_score
    return {
        "trigger": bool(enough and good),
        "unclaimed_traces": len(rows),
        "scored_traces": len(scored),
        "mean_score": mean,
        "min_traces": min_traces,
        "min_score": min_score,
    }


# ------------------------------------------------- cron trigger visibility

def cron_line(cfg_home: Optional[str] = None) -> str:
    """A cron line that actually resolves `toaster-loop` in cron's minimal PATH.

    The panel used to suggest `cd ~ && toaster-loop train auto-check --trigger`,
    which silently fails (exit 127, "command not found") whenever toaster-loop is
    only installed in a venv — i.e. every source install. Anchor on this
    interpreter's sibling script instead.

    Paths are shlex-quoted: an install under a directory with spaces (e.g.
    `/media/x/Toaster Loop/.venv/bin/toaster-loop`) otherwise splits at the
    space in cron's shell and exits 127 exactly like the bare-name bug did."""
    import shlex

    exe_dir = Path(sys.executable).parent
    cand = exe_dir / ("toaster-loop.exe" if os.name == "nt" else "toaster-loop")
    toaster_loop = str(cand) if cand.is_file() else (shutil.which("toaster-loop") or "toaster-loop")
    log = str(Path(cfg_home) if cfg_home else (Path.home() / ".toaster-loop"))
    return (f"15 3 * * * {shlex.quote(toaster_loop)} train auto-check --trigger "
            f">> {shlex.quote(log + '/auto_iterate.log')} 2>&1")


def cron_trigger_status(cfg_home: Optional[str] = None) -> Dict[str, Any]:
    """Is the external scheduler (cron) actually wired to `auto-check`?

    Honesty fix: `training.auto_enabled: true` reads as "trains itself", but
    the dashboard runs no ticker and the ONLY caller of auto-check --trigger
    is the user's crontab. When no crontab mentions auto-check, the loop can
    never fire — this reports that instead of leaving it implicit. Read-only:
    toaster_loop never installs cron entries for you."""
    line = cron_line(cfg_home)
    out: Dict[str, Any] = {"installed": None, "line": line,
                           "note": "could not inspect the crontab"}
    if os.name == "nt":
        out["note"] = ("Windows: schedule `toaster-loop train auto-check --trigger` "
                       "in Task Scheduler (toaster_loop does not touch it)")
        return out
    try:
        r = subprocess.run(["crontab", "-l"], capture_output=True,
                           text=True, timeout=10)
        if r.returncode != 0:
            out["installed"] = False  # no crontab at all (or unreadable)
            out["note"] = ("no crontab found — nothing ever calls "
                           "`train auto-check --trigger`, so the autonomous "
                           "loop cannot fire")
            return out
        out["installed"] = "auto-check" in (r.stdout or "")
        out["note"] = ("crontab references auto-check" if out["installed"]
                       else "crontab exists but has no auto-check entry")
    except (OSError, subprocess.SubprocessError) as e:
        out["note"] = f"crontab check failed: {e}"
    return out


# ---------------------------------------------------------------- scheduler

class SchedulerError(RuntimeError):
    pass


@dataclass
class Scheduler:
    """Runs queued jobs one at a time in worker subprocesses.

    ``worker_command`` is an argv prefix; the worker args are appended. Default
    runs this package's worker with the current interpreter (or
    TrainingConfig.python when set). Tests inject a stub script — that is how
    `toaster-loop iterate` is exercised without GPU.

    ``observer`` (Phase 5a) is an optional callback ``(job_id, state)`` invoked
    after every state transition this scheduler performs (running/done/error/
    cancelled). The dashboard wires it to its event bus; exceptions from the
    observer are swallowed so UI problems can never break a training run.

    ``inprocess_runners`` (pass 4 #32) maps job kind -> callable ``(job) ->
    dict`` executed IN-PROCESS after the resource gate + device slot claim
    (same one-GPU serialization as subprocess workers, but no spawn — for
    pure-python kinds like shadow_eval that need this process's handles).
    A queued kind with no registered runner fails honestly instead of being
    handed to the GPU worker CLI.

    ``coexist`` (WP5 train+serve handshake) is an optional object exposing
    ``rescue_gate(job_id, required_mib) -> bool`` (serving/coexist.py). When
    the VRAM gate fails and the rescue returns True (served instances were
    gracefully unloaded and free VRAM was re-probed), the gate is re-evaluated
    once. None (the default — no WP5 config keys set) keeps the refusal path
    byte-identical to pre-WP5.
    """

    registry: ModelRegistry
    cfg: TrainingConfig = field(default_factory=TrainingConfig)
    worker_command: Optional[List[str]] = None
    observer: Optional[Any] = None
    inprocess_runners: Optional[Dict[str, Any]] = None
    coexist: Optional[Any] = None

    def _record_gate_block(self, job_id: str, reason: str) -> None:
        """Persist a gate-block reason and log NEW block episodes (schema v9).

        Queue-with-reason (the job stays ``queued``; the reason rides on
        params so the Jobs/Train panels can explain the wait) plus a durable
        append to the ``gate_events`` ledger so
        ``toaster_loop_gate_blocks_total`` is a true cumulative counter, not a
        gauge over params that the scheduler clears on unblock.

        Episode dedup: one event per none->blocked transition or reason
        change — repeated blocked dispatch attempts with the SAME reason
        (every dispatcher tick re-attempts the job) count ONCE. The previous
        reason is read from the row before writing, so no extra state is
        kept. Best-effort end to end: metrics honesty must never break
        dispatch (the caller still raises SchedulerError either way).
        """
        prev = None
        try:
            cur = self.registry.get_job(job_id)
            if cur is not None and isinstance(cur.params, dict):
                prev = cur.params.get("_gate_reason")
        except Exception:  # pragma: no cover - defensive
            pass
        try:
            self.registry.set_job_param(job_id, "_gate_reason", reason)
        except Exception:  # pragma: no cover - defensive
            logger.debug("gate reason persist failed", exc_info=True)
        if prev and prev == reason:
            return  # same block episode already logged — not a new one
        try:
            # Open/close per event: the dashboard builds a Scheduler per
            # request, so caching the store here would leak connections;
            # new block episodes are rare (at most one per dispatch attempt).
            store = TraceStore(self.registry.db_path)
            try:
                store.record_gate_event(job_id, reason)
            finally:
                store.close()
        except Exception:  # pragma: no cover - defensive
            logger.debug("gate event record failed", exc_info=True)

    def _notify(self, job_id: str, state: str) -> None:
        if self.observer is None:
            return
        try:
            self.observer(job_id, state)
        except Exception:  # pragma: no cover - defensive
            logger.debug("job observer failed (ignored)", exc_info=True)

    @property
    def max_parallel(self) -> int:
        return max(1, int(self.cfg.max_parallel_jobs or 1))

    def _claim_device_slot(self, job_id: str):
        """Claim the lowest free device slot for job_id, atomically.

        Returns (slot, cross_process_lock_handle) or None when every slot is
        busy. Occupancy = registry-visible running jobs (params._device) plus
        in-flight claims not yet marked running; the module-level lock makes
        claim+persist atomic across scheduler instances/threads in this
        process (the dashboard spawns one thread per submit). Cross-process,
        each slot additionally carries a non-blocking flock sidecar next to
        the daemon DB: another `toaster_loop` process running a worker on that
        device holds it, so two processes can never dispatch onto one GPU.
        The kernel drops the flock if the holder dies — no stale locks.
        Caller releases via _release_device_slot(job_id, handle).
        """
        with _SLOT_LOCK:
            taken = set(_INFLIGHT_CLAIMS.values())
            for j in self.registry.running_jobs():
                # A dispatcher claim flips THIS job to running before we get
                # here; its persisted _device from a previous gated attempt is
                # its own affinity, not another holder's — never let a job
                # block re-claiming the slot it already owns.
                if j.job_id == job_id:
                    continue
                d = j.params.get("_device") if isinstance(j.params, dict) else None
                if isinstance(d, int):
                    taken.add(d)
            for slot in range(self.max_parallel):
                if slot in taken:
                    continue
                try:
                    handle = _try_slot_lock(self.registry.db_path, slot)
                except OSError:
                    handle = None  # lock file unusable -> thread-only guard
                if handle is None:
                    continue  # another process owns this device slot
                _INFLIGHT_CLAIMS[job_id] = slot
                self.registry.set_job_param(job_id, "_device", slot)
                return slot, handle
            return None

    def _release_device_slot(self, job_id: str, handle=None) -> None:
        with _SLOT_LOCK:
            _INFLIGHT_CLAIMS.pop(job_id, None)
        _release_slot_lock(handle)

    def _device_for_slot(self, slot: int) -> str:
        ids = self.cfg.gpu_ids or []
        return str(ids[slot]) if slot < len(ids) else str(slot)

    # ---- submission ------------------------------------------------------

    def submit(
        self,
        kind: str,
        *,
        dataset_id: Optional[str] = None,
        model_id: Optional[str] = None,
        log_dir: Optional[Path] = None,
        params: Optional[Dict[str, Any]] = None,
        enqueue_actor: Optional[str] = None,
        depends_on: Optional[List[str]] = None,
    ) -> JobRow:
        # ``depends_on`` (pass4 #43): edge list over queued work — the job is
        # unclaimable until every dependency reaches `done` (registry
        # next_queued), and cascade-fails when one goes terminal-bad.
        log_path = None
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            # Provisional name; renamed to <job_id>.log once the id exists.
            # uuid4 keeps concurrent submits in one process from colliding.
            log_path = str(
                Path(log_dir) / f".{kind}-pending-{os.getpid()}-{uuid.uuid4().hex}.log"
            )
        job = self.registry.enqueue_job(
            kind, dataset_id=dataset_id, model_id=model_id,
            log_path=log_path, params=params or {}, actor=enqueue_actor,
            depends_on=depends_on,
        )
        if log_dir and job.log_path:
            final = Path(log_dir) / f"{job.job_id}.log"
            Path(job.log_path).touch(exist_ok=True)
            os.replace(job.log_path, final)
            self.registry.set_job_log_path(job.job_id, str(final))
            job.log_path = str(final)
        return job

    # ---- execution --------------------------------------------------------

    def worker_argv(self, job: JobRow) -> List[str]:
        # Precedence: injected instance command > env override > config key >
        # this interpreter's module worker.
        base: Optional[List[str]] = self.worker_command
        if base is None and os.environ.get("TOASTER_LOOP_WORKER_COMMAND"):
            import shlex

            base = shlex.split(os.environ["TOASTER_LOOP_WORKER_COMMAND"])
        if base is None and self.cfg.worker_command:
            base = list(self.cfg.worker_command)
        if base is None:
            base = [self.cfg.python or sys.executable, "-m",
                    "toaster_loop.training.worker"]
        params = dict(job.params)
        argv = list(base) + ["--job", job.job_id, "--kind", job.kind]
        # Only pass flags whose values exist — never leave a dangling option.
        optional_flags = [
            ("--config", params.get("train_yaml")),
            ("--dataset-dir", params.get("dataset_dir")),
            ("--base-model", params.get("base_model")),
            ("--log-path", job.log_path),
            ("--adapter-dir", params.get("adapter_dir")),
            ("--gguf-out", params.get("gguf_out")),
            ("--quant", params.get("quant")),
            # Continual training (deep-dive fix): existing adapter to build on.
            ("--base-adapter", params.get("base_adapter")),
        ]
        for flag, value in optional_flags:
            if value not in (None, ""):
                argv += [flag, str(value)]
        if params.get("dry_run"):
            argv.append("--dry-run")
        return argv

    def reap_stale(self) -> List[str]:
        """Fail `running` jobs whose pid is dead (kill -9 consistency).

        In-process kinds (curate) have no worker subprocess — the dashboard
        thread that owns them reports completion; reaping would race it.
        """
        from ..registry.registry import INPROCESS_JOB_KINDS

        reaped: List[str] = []
        for job in self.registry.running_jobs():
            if job.kind in INPROCESS_JOB_KINDS:
                continue
            pid = job.pid
            alive = False
            if pid:
                # Windows-safe liveness (A1): raw os.kill(pid, 0) TERMINATES on
                # NT. None ("alive but not ours") keeps the old PermissionError
                # semantics -> treat as alive, never reap what we can't own.
                from ..serving.runtime import pid_alive

                alive = pid_alive(pid) is not False
            if not alive:
                self.registry.finish_job(
                    job.job_id, "error",
                    error="worker process died without reporting completion "
                          "(reaped by scheduler)",
                )
                self._notify(job.job_id, "error")
                reaped.append(job.job_id)
        return reaped

    def run_job(self, job: JobRow, *, wait_gate: bool = False) -> JobRow:
        """Run one job to completion in a subprocess.

        Concurrency is bounded by cfg.max_parallel_jobs device slots; each
        worker sees exactly one GPU via CUDA/HIP_VISIBLE_DEVICES. With the
        default max_parallel_jobs=1 this preserves the original single-flight
        behavior (and its error message).

        Pass4 #43: a job with unmet ``depends_on`` edges refuses to run when
        claimed directly (the queue drain never selects it anyway); it stays
        queued until every dependency is done."""
        if job.depends_on:
            unmet = []
            for dep in job.depends_on:
                djob = self.registry.get_job(dep)
                if djob is None or djob.state != "done":
                    unmet.append(f"{dep}"
                                 + (f" ({djob.state})" if djob else " (missing)"))
            if unmet:
                raise SchedulerError(
                    f"job {job.job_id} waits on dependencies: "
                    f"{', '.join(unmet)} (pass 4 #43); it starts when they "
                    "all finish successfully")
        claim = self._claim_device_slot(job.job_id)  # persists params._device
        if claim is None:
            running = self.registry.running_jobs()
            who = running[0] if running else None
            raise SchedulerError(
                f"job {who.job_id} is already running "
                f"(pid={who.pid}); one GPU, one job"
                if who and self.max_parallel == 1 else
                f"{self.max_parallel} jobs already running "
                "(training.max_parallel_jobs reached)"
            )
        slot, slot_lock = claim

        try:
            self._run_gated(job, slot, wait_gate)
        finally:
            self._release_device_slot(job.job_id, slot_lock)
        return self.registry.get_job(job.job_id) or job

    def _gate_device(self, slot: int) -> Optional[str]:
        """Per-slot VRAM gate device, or None for the global-minimum probe.

        Only when ``training.gpu_ids`` explicitly pins the slot->device map
        do we know the worker's physical device is meaningful to query; with
        the default (unpinned) config the historical global min-free probe
        runs unchanged (cardinal rule: opt-in behavior change)."""
        if not self.cfg.gpu_ids:
            return None
        return self._device_for_slot(slot)

    def _gate_pass(self, job: JobRow, need: Optional[int],
                   device: Optional[str] = None) -> Dict[str, Any]:
        """Resource gate with the WP5 train+serve handshake (opt-in).

        Plain ``resource_gate_ok`` first — when it passes (or no ``coexist``
        rescue object is wired, which is the default), this is exactly the
        pre-WP5 path. On a VRAM shortfall the rescue gets one chance: under
        ``serve.coexist_policy: unload`` it gracefully unloads served
        instances and re-probes free VRAM (bounded wait); under ``refuse`` it
        only emits the explanatory coexist_refused event and returns False.
        A successful rescue re-evaluates the real gate once — an honest second
        refusal beats a fake admission. The blocking ``wait_gate`` loops
        deliberately do NOT call this (an unbounded retry loop would spam
        unload events every poll); the non-blocking dispatch path is where
        the handshake lives."""
        gate = resource_gate_ok(self.cfg, estimated_need_mib=need, device=device)
        if gate["ok"] or self.coexist is None:
            return gate
        required = max(int(self.cfg.min_free_vram_mib or 0),
                       int(need or 0))
        rescued = False
        if required > 0:
            try:
                rescued = bool(self.coexist.rescue_gate(job.job_id, required))
            except Exception:  # noqa: BLE001 — rescue never breaks dispatch
                logger.debug("coexistence gate rescue failed", exc_info=True)
        if rescued:
            gate = resource_gate_ok(self.cfg, estimated_need_mib=need,
                                    device=device)
        return gate

    def _run_gated(self, job: JobRow, slot: int, wait_gate: bool) -> None:
        if job.kind in INPROCESS_JOB_KINDS:
            self._run_inprocess(job, slot, wait_gate)
            return
        # Predict this job's footprint once per dispatch attempt (config dims
        # don't change while queued). None => static-floor behavior. The
        # training interpreter rides along so the ROCm bitsandbytes probe runs
        # where torch actually lives (see job_vram_estimate_mib).
        need = job_vram_estimate_mib(job, python=self.cfg.python)
        if need is not None:
            logger.info("job %s predicted VRAM footprint: %d MiB",
                        job.job_id, need)
        device = self._gate_device(slot)  # None unless gpu_ids pins slots
        if wait_gate:
            while True:
                gate = resource_gate_ok(self.cfg, estimated_need_mib=need,
                                        device=device)
                if gate["ok"]:
                    break
                logger.info("resource gate: %s — waiting", "; ".join(gate["reasons"]))
                time.sleep(max(1.0, self.cfg.poll_interval_s))
        else:
            gate = self._gate_pass(job, need, device=device)
            if not gate["ok"]:
                # Queue-with-reason: the job stays `queued`; the reason rides on
                # params so the Jobs/Train panels can explain the wait (and the
                # next dispatch attempt overwrites it). New block episodes also
                # land in the durable gate_events ledger (v9) — see
                # _record_gate_block for the episode-dedup semantics.
                self._record_gate_block(job.job_id, "; ".join(gate["reasons"]))
                raise SchedulerError(
                    "resource gate blocked start: " + "; ".join(gate["reasons"])
                )
        try:
            self.registry.set_job_param(job.job_id, "_gate_reason", None)
        except Exception:  # pragma: no cover - defensive
            pass
        self._spawn_worker(job, slot)

    def _run_inprocess(self, job: JobRow, slot: int, wait_gate: bool) -> None:
        """Run a pure-python job kind in this process (pass 4 #32).

        Same gate + device-slot serialization as worker subprocesses; the
        runner callable does the work and its exceptions become job errors.
        ``pid`` records this process — the reaper skips INPROCESS kinds, so it
        is informational only.
        """
        need = job_vram_estimate_mib(job, python=self.cfg.python)  # None for non-GPU kinds
        if wait_gate:
            while True:
                gate = resource_gate_ok(self.cfg, estimated_need_mib=need)
                if gate["ok"]:
                    break
                logger.info("resource gate: %s — waiting", "; ".join(gate["reasons"]))
                time.sleep(max(1.0, self.cfg.poll_interval_s))
        else:
            gate = self._gate_pass(job, need)
            if not gate["ok"]:
                self._record_gate_block(job.job_id, "; ".join(gate["reasons"]))
                raise SchedulerError(
                    "resource gate blocked start: " + "; ".join(gate["reasons"])
                )
        try:
            self.registry.set_job_param(job.job_id, "_gate_reason", None)
        except Exception:  # pragma: no cover - defensive
            pass

        runner = (self.inprocess_runners or {}).get(job.kind)
        if runner is None and job.kind == "shadow_eval":
            runner = self._default_shadow_runner()
        if runner is None and job.kind == "evolve":
            runner = self._default_evolve_runner()
        if runner is None and job.kind == "rft":
            runner = self._default_rft_runner()
        if runner is None and job.kind == "mine_goldens":
            runner = self._default_mine_goldens_runner()
        if runner is None:
            self.registry.finish_job(
                job.job_id, "error",
                error=f"no in-process runner registered for job kind "
                      f"{job.kind!r}", actor="scheduler")
            self._notify(job.job_id, "error")
            return

        logger.info("running %s job in-process (device slot %d)", job.kind, slot)
        if not self.registry.mark_running(job.job_id, os.getpid(),
                                          actor="scheduler"):
            cur = self.registry.get_job(job.job_id)
            if not (cur and cur.state == "running"
                    and cur.pid == os.getpid()):
                # Cancelled while queued — keep the terminal state; don't run.
                return
            # else: dispatcher claim (claim_next_queued already flipped this
            # row to running under our pid) — proceed with the runner.
        self._notify(job.job_id, "running")
        try:
            runner(job)
        except Exception as e:
            current = self.registry.get_job(job.job_id)
            if current and current.state == "cancelled":
                return
            self.registry.finish_job(job.job_id, "error",
                                     error=f"{type(e).__name__}: {e}",
                                     actor="scheduler")
            self._notify(job.job_id, "error")
            return
        current = self.registry.get_job(job.job_id)
        if current and current.state == "cancelled":
            return  # cancel won the race; keep cancelled
        self.registry.finish_job(job.job_id, "done", actor="scheduler")
        self._notify(job.job_id, "done")

    def _default_evolve_runner(self):
        """Lazy default evolve runner (pass 4 #39), mirroring shadow's.

        ``evolution.enabled`` + ``max_call_budget`` are read from the config
        file at run time, so flipping the master switch takes effect for jobs
        queued before it. Custom seams inject ``inprocess_runners`` instead.
        """
        try:
            from ..evaluation.evolve import (evolution_config,
                                             make_evolve_job_runner)

            cfg = evolution_config()
            return make_evolve_job_runner(self.registry,
                                          home=self.registry.home,
                                          defaults=cfg)
        except Exception as e:  # pragma: no cover - defensive
            logger.error("evolve: runner unavailable: %s", e)
            return None

    def _default_rft_runner(self):
        """Lazy default rft runner (rejection sampling), mirroring evolve's.

        ``rft:`` config is read from the config file at run time, so flipping
        the master switch takes effect for jobs queued before it. Custom seams
        inject ``inprocess_runners`` instead. Like shadow/evolve this is pure
        stdlib+httpx I/O against the served endpoint — in-process under the
        dispatcher drain is the right resource shape (one device slot, no GPU
        worker subprocess)."""
        try:
            from .rejection import make_rft_job_runner, rft_config

            return make_rft_job_runner(self.registry, home=self.registry.home,
                                       defaults=rft_config())
        except Exception as e:  # pragma: no cover - defensive
            logger.error("rft: runner unavailable: %s", e)
            return None

    def _default_mine_goldens_runner(self):
        """Lazy default mine_goldens runner (golden mining from 👍 traces),
        mirroring rft's. ``goldens_mine:`` config is read from the config file
        at run time, so flipping the master switch takes effect for jobs queued
        before it. Custom seams inject ``inprocess_runners`` instead."""
        try:
            from ..curate.mine_goldens import (goldens_mine_config,
                                               make_mine_goldens_job_runner)

            return make_mine_goldens_job_runner(
                self.registry, home=self.registry.home,
                defaults=goldens_mine_config())
        except Exception as e:  # pragma: no cover - defensive
            logger.error("mine_goldens: runner unavailable: %s", e)
            return None

    def _default_shadow_runner(self):
        """Lazy default shadow_eval runner over this registry's own store.

        The scheduler and the buffer DB live in the same process by design
        (single-writer daemon DB), so opening a TraceStore on the registry's
        db_path is the canonical wiring; callers needing custom seams inject
        ``inprocess_runners`` instead.
        """
        try:
            from ..evaluation.shadow import make_shadow_job_runner
            from ..store.db import TraceStore

            store = TraceStore(self.registry.db_path)
            return make_shadow_job_runner(self.registry, store,
                                          home=self.registry.home)
        except Exception as e:  # pragma: no cover - defensive
            logger.error("shadow_eval: runner unavailable: %s", e)
            return None

    def _spawn_worker(self, job: JobRow, slot: int) -> None:
        argv = self.worker_argv(job)
        logger.info("spawning worker (device slot %d): %s", slot, " ".join(argv))
        # Pin the worker to one physical device. torch maps both CUDA_VISIBLE_
        # DEVICES and HIP_VISIBLE_DEVICES; setting both is harmless elsewhere.
        env = dict(os.environ)
        dev = self._device_for_slot(slot)
        env["CUDA_VISIBLE_DEVICES"] = dev
        env["HIP_VISIBLE_DEVICES"] = dev
        try:
            proc = subprocess.Popen(
                argv,
                stdout=subprocess.DEVNULL if job.log_path else None,
                stderr=subprocess.STDOUT if job.log_path else None,
                stdin=subprocess.DEVNULL,
                env=env,
            )
        except OSError as e:
            self.registry.finish_job(job.job_id, "error", error=f"spawn failed: {e}")
            self._notify(job.job_id, "error")
            return
        if not self.registry.mark_running(job.job_id, proc.pid):
            # Row wasn't queued: either cancelled in the race between
            # next_queued()/claim and here, or a dispatcher claim this process
            # already holds (upgrade our placeholder pid to the worker's).
            if not self.registry.mark_worker_started(job.job_id, proc.pid):
                # Cancelled — kill the worker we just spawned and keep the
                # terminal 'cancelled' state.
                try:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                except Exception:
                    pass
                return
        self._notify(job.job_id, "running")

        timeout = self.cfg.worker_timeout_s or None
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            self.registry.finish_job(
                job.job_id, "error", error=f"worker exceeded {timeout}s timeout"
            )
            self._notify(job.job_id, "error")
            return

        current = self.registry.get_job(job.job_id)
        if current and current.state == "cancelled":
            return  # cancel won the race; keep cancelled
        if rc == 0:
            self.registry.finish_job(job.job_id, "done")
            self._notify(job.job_id, "done")
        else:
            tail = ""
            if job.log_path and Path(job.log_path).is_file():
                try:
                    lines = Path(job.log_path).read_text().splitlines()
                    errs = [l for l in lines if '"error"' in l]
                    tail = ("; last worker error: " + errs[-1][:400]) if errs else ""
                except OSError:
                    pass
            self.registry.finish_job(
                job.job_id, "error", error=f"worker exited rc={rc}{tail}"
            )
            self._notify(job.job_id, "error")

    def run_pending(self, *, wait_gate: bool = False) -> List[JobRow]:
        """Drain the FIFO queue (one at a time). Returns finished jobs."""
        results: List[JobRow] = []
        self.reap_stale()
        while True:
            if self.registry.running_jobs():
                break
            job = self.registry.next_queued()
            if job is None:
                break
            results.append(self.run_job(job, wait_gate=wait_gate))
        return results

    # ---- cancel ------------------------------------------------------------

    def cancel(self, job_id: str, actor: str = "scheduler") -> Optional[JobRow]:
        """Cancel a queued or running job; terminate the worker when running.

        ``actor`` stamps the ledger entry (pass4 #31): dashboard handlers pass
        "dashboard", CLI keeps the default.

        Cross-platform stop (A2): routes through the shared WP4 helper
        ``runtime.terminate_owned_pid`` — SIGTERM with SIGKILL escalation on
        POSIX, ``taskkill /T`` on Windows (os.kill(pid, SIGTERM) there maps to
        TerminateProcess: no grace window for checkpoint/job-log flush). The
        ownership rule applies: a pid alive but NOT ours raises before the
        registry is touched (same as the old PermissionError branch).

        Ordering note: the registry is stamped *cancelled first*, then the
        worker is terminated. ``finish_job`` refuses to resurrect a cancelled
        job, so this ordering is race-free even though terminate_owned_pid
        blocks until the process exits (a running run_job completion handler
        can no longer land an ``error`` over our ``cancelled``).

        In-process kinds (curate/shadow_eval/evolve) record THIS process's pid
        in ``job.pid`` — signalling it would kill the dashboard itself, so no
        signal is sent for them; their runner threads observe the cancelled
        state via ``finish_job``'s idempotency guard. Same protection when the
        recorded pid equals our own for any other reason.
        """
        job = self.registry.get_job(job_id)
        if job is None:
            return None
        kill_pid = 0
        if (job.state == "running" and job.pid
                and job.kind not in INPROCESS_JOB_KINDS
                and job.pid != os.getpid()):
            from ..serving.runtime import pid_alive

            if pid_alive(job.pid) is None:
                raise SchedulerError(
                    f"cannot signal worker pid {job.pid}: not ours to signal")
            kill_pid = int(job.pid)
        out = self.registry.cancel_job(job_id, actor=actor)
        if (kill_pid and out is not None and out.state == "cancelled"):
            from ..serving.runtime import terminate_owned_pid

            terminate_owned_pid(kill_pid)
        if out is not None and out.state == "cancelled":
            self._notify(job_id, "cancelled")
        return out

    # ---- progress ------------------------------------------------------------

    def tail_log(self, job: JobRow, n: int = 10) -> List[str]:
        if not job.log_path or not Path(job.log_path).is_file():
            return []
        try:
            lines = Path(job.log_path).read_text().splitlines()
        except OSError:
            return []
        return lines[-n:]

    # ---- claim-aware execution helpers --------------------------------------

    def run_claimed(self, job: JobRow) -> JobRow:
        """Run a job this process already flipped queued -> running.

        The ownership-safe companion of ``run_job`` for the dispatcher and the
        dashboard submit threads: the CALLER holds the claim (via
        ``ModelRegistry.claim_next_queued`` / ``mark_running`` with our pid),
        so no second thread or process can execute this row. A gate/slot
        refusal (SchedulerError) releases the claim back to `queued` — with
        any ``_gate_reason`` preserved — so the next dispatcher tick retries;
        the job never strands in a half-claimed state.
        """
        try:
            return self.run_job(job)
        except SchedulerError:
            try:
                if self.registry.requeue_job(job.job_id, os.getpid()):
                    logger.info("job %s claim released; dispatcher will retry",
                                job.job_id)
            except Exception:  # pragma: no cover - defensive
                logger.debug("claim release failed", exc_info=True)
            raise

    def drain_queue(self) -> List[JobRow]:
        """Claim-and-run queued jobs until the queue is empty or capacity is
        full. Returns finished jobs (empty when nothing was claimable).

        Uses ``claim_next_queued`` — a single guarded UPDATE inside an
        IMMEDIATE transaction — so concurrent drainers (submit threads, the
        dispatcher tick, a second dashboard) each run distinct rows; nobody
        executes a job twice and nobody waits on a job another holder owns.
        A gate-blocked claim ends the drain: retrying it this instant would
        just re-hit the same gate, so it stays queued for the next tick."""
        results: List[JobRow] = []
        while len(self.registry.running_jobs()) < self.max_parallel:
            job = self.registry.claim_next_queued(
                os.getpid(), skip_kinds=("curate", "curate_source"))
            if job is None:
                break
            try:
                results.append(self.run_claimed(job))
            except SchedulerError:
                break  # gate/slot busy — the dispatcher retries on its tick
        return results


# ------------------------------------------------------------- dispatcher

def _dispatcher_lock_path(db_path: Path) -> Path:
    """Pid-lockfile for the queue dispatcher (next to the daemon DB)."""
    return Path(str(db_path) + ".dispatcher.lock")


class DispatcherAlreadyRunning(RuntimeError):
    """Another live process holds the dispatcher lockfile for this DB."""


@dataclass
class JobDispatcher:
    """Background queue dispatcher + stale-job reaper (dashboard boot).

    Closes the three gaps ``run_pending`` left in production (it has no
    production callers): queued jobs stranded after a submit thread gave up,
    phantom `running` rows from dead workers blocking submits with 409, and
    gate-blocked jobs never re-attempted. Each tick:

      1. ``reap_stale()`` — fail `running` rows whose worker pid is dead;
      2. dispatch — atomically claim queued work and hand each claim to a
         per-job daemon thread (gate-blocked claims are released back to
         `queued` by the thread and retried on a later tick, so a job blocked
         by the VRAM/AC gate starts as soon as its gate passes).

    Deep-dive fix #2: claims run on PER-JOB THREADS, not inline on the tick
    loop. ``Scheduler.drain_queue`` blocks inside ``proc.wait`` for the whole
    duration of each worker (hours), which froze the tick between jobs — no
    reaping of stale rows, no second job under ``max_parallel_jobs > 1``, no
    gate retries while a long train ran. Here the tick only *claims* (a fast
    guarded UPDATE that flips queued -> running with our pid) and spawns; the
    thread owns one scheduler (own SQLite connection from
    ``scheduler_factory``), runs the job, and closes its registry when done.
    Device safety is unchanged: the per-slot flock + ``_INFLIGHT_CLAIMS``
    bookkeeping inside ``_claim_device_slot`` still serialize devices, and the
    claim itself caps in-flight dispatch at ``max_parallel`` because claimed
    rows are immediately visible as `running`.

    A tick also runs immediately at :meth:`start` (startup reap), so phantom
    rows from a previous crash never outlive boot. Exactly one dispatcher may
    run per DB file: ``start`` takes a non-blocking flock on a pid-lockfile
    beside the DB and raises :class:`DispatcherAlreadyRunning` naming the
    holder pid + path on contention. Like the device-slot locks, the kernel
    drops the flock when the holder dies — stale locks from dead pids are
    reclaimed automatically (the pid inside is informational; it is rewritten
    on every successful claim). On Windows the same lock runs through msvcrt
    byte-range locking (core/proclock.py) — previously this degraded to
    this-process-only, matching the old ``_try_slot_lock`` no-op.

    ``scheduler_factory`` is called once per thread (and once for the startup
    reap) so every worker thread owns its own SQLite connection; the
    dispatcher closes ``scheduler.registry`` when that thread's work ends.
    """

    db_path: Path
    scheduler_factory: Any  # () -> Scheduler (fresh registry on db_path)
    tick_s: float = DISPATCH_TICK_S
    _thread: Optional[threading.Thread] = field(default=None, init=False,
                                                repr=False)
    _stop: Optional[threading.Event] = field(default=None, init=False,
                                             repr=False)
    _lock_fh: Any = field(default=None, init=False, repr=False)
    _job_threads: List[threading.Thread] = field(default_factory=list,
                                                 init=False, repr=False)
    _jobs_lock: threading.Lock = field(default_factory=threading.Lock,
                                       init=False, repr=False)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ---- lockfile -----------------------------------------------------------

    def _acquire_lock(self) -> None:
        path = _dispatcher_lock_path(self.db_path)
        handle = proclock.acquire(path)
        if handle is None:
            holder = proclock.holder_pid(path)
            raise DispatcherAlreadyRunning(
                f"another toaster_loop dispatcher is running for this database "
                f"(lockfile {path}, holder pid {holder or 'unknown'}) — "
                "two dashboards must not share one DB; stop the other "
                "process first")
        # (Re)claim: truncate away any stale pid from a dead holder and
        # record ours, so the next contended start can name us.
        proclock.write_holder(handle)
        self._lock_fh = handle

    def _release_lock(self) -> None:
        fh, self._lock_fh = self._lock_fh, None
        proclock.release(fh)

    # ---- lifecycle -----------------------------------------------------------

    def start(self) -> "JobDispatcher":
        """Acquire the lockfile, reap once synchronously, spawn the tick loop.

        Raises DispatcherAlreadyRunning on contention — callers surface it
        loudly (the dashboard logs it at ERROR); never silently co-drive."""
        if self.is_running():
            return self
        self._acquire_lock()
        # Startup reap: a phantom `running` row from a crashed worker must not
        # survive into the first tick window and hold submits at 409.
        sched = self.scheduler_factory()
        try:
            reaped = sched.reap_stale()
            if reaped:
                logger.warning("dispatcher startup reaped %d stale job(s): %s",
                               len(reaped), ", ".join(reaped))
        except Exception:  # pragma: no cover - defensive
            logger.exception("dispatcher startup reap failed")
        finally:
            try:
                sched.registry.close()
            except Exception:  # pragma: no cover - defensive
                pass
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="job-dispatcher")
        self._thread.start()
        logger.info("job dispatcher started (tick %.1fs, lock %s)",
                    self.tick_s, _dispatcher_lock_path(self.db_path))
        return self

    def stop(self, timeout: Optional[float] = None) -> None:
        if self._stop is not None:
            self._stop.set()
        t, self._thread = self._thread, None
        if t is not None:
            t.join(timeout=timeout if timeout is not None else self.tick_s + 5.0)
        self._release_lock()

    def _prune_threads(self) -> int:
        """Drop finished job threads; returns how many are still alive."""
        with self._jobs_lock:
            self._job_threads = [t for t in self._job_threads if t.is_alive()]
            return len(self._job_threads)

    def _spawn_job_thread(self, job: JobRow) -> None:
        """Run one already-claimed job on its own daemon thread.

        The thread builds a dedicated Scheduler (own SQLite connection — the
        tick loop's connection is never touched from here), runs the claim,
        and closes its registry. A gate/slot refusal requeues the row via
        ``run_claimed``'s release path, so the next tick retries it; anything
        else is logged — the registry state machine already recorded the
        terminal error through ``run_job``."""
        def _runner() -> None:
            sched = None
            try:
                sched = self.scheduler_factory()
                try:
                    sched.run_claimed(job)
                    logger.info("dispatcher finished job %s", job.job_id)
                except SchedulerError:
                    pass  # claim released to queued; retried on a later tick
                except Exception:  # pragma: no cover - defensive
                    logger.exception("dispatcher job %s failed", job.job_id)
            finally:
                if sched is not None:
                    try:
                        sched.registry.close()
                    except Exception:  # pragma: no cover - defensive
                        pass

        t = threading.Thread(target=_runner, daemon=True,
                             name=f"job-dispatch-{job.job_id[:8]}")
        with self._jobs_lock:
            self._job_threads.append(t)
        t.start()

    def _dispatch(self, sched: Scheduler) -> int:
        """Claim queued work up to capacity and hand each claim to a thread.

        Mirrors ``drain_queue``'s claiming (same guarded UPDATE + skip_kinds),
        but returns as soon as capacity is reached instead of running jobs
        inline. The claim flips the row to `running` under our pid, so the
        ``running_jobs()`` count naturally stops us at ``max_parallel``."""
        started = 0
        while len(sched.registry.running_jobs()) < sched.max_parallel:
            job = sched.registry.claim_next_queued(
                os.getpid(), skip_kinds=("curate", "curate_source"))
            if job is None:
                break
            self._spawn_job_thread(job)
            started += 1
        return started

    # ---- loop ------------------------------------------------------------------

    def _loop(self) -> None:
        assert self._stop is not None
        sched = self.scheduler_factory()
        try:
            while True:
                try:
                    reaped = sched.reap_stale()
                    if reaped:
                        logger.warning("dispatcher reaped %d stale job(s): %s",
                                       len(reaped), ", ".join(reaped))
                    self._prune_threads()
                    started = self._dispatch(sched)
                    if started:
                        logger.info("dispatcher started %d queued job(s)",
                                    started)
                except Exception:  # a tick must never kill the loop
                    logger.exception("dispatcher tick failed")
                if self._stop.wait(self.tick_s):
                    break
        finally:
            try:
                sched.registry.close()
            except Exception:  # pragma: no cover - defensive
                pass
            self._release_lock()

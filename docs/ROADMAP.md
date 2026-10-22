# ROADMAP — sable

> **Status (2026-10).** Phases 1–5 all shipped, plus a Phase 6 (multi-model serving) and
> two idea passes that grew the loop: hub downloads, operation ledger + capability tiers,
> shadow/transfer evaluation, prompt evolution, RFT, golden mining, playbooks, skills,
> easy training. The outlines below are kept as delivered intent; where implementation
> diverged (fully built-in runtime and training stack — see [`DECISIONS.md`](DECISIONS.md)
> D-12; YAML config instead of TOML), [`ARCHITECTURE.md`](ARCHITECTURE.md) is the source
> of truth. The current command/panel surface lives in the README.

Phased delivery. Each phase is independently useful and ends with an acceptance test.
No phase may introduce a cloud dependency.

---

## Phase 1 — Walking skeleton: agent loop against a local server

**Scope**
- `sable` package scaffold (`pyproject.toml`, uv, `sable/` src tree, `~/.sable/` layout).
- Serving Supervisor v0: spawn llama-server with a GGUF (or explicitly attach to any
  running OpenAI-compatible server), health-check loop, clean shutdown.
- Boot the agent runtime against `http://127.0.0.1:<port>/v1` and run a tool-calling
  task end-to-end (CLI entry `sable run --task "…"`).
- Minimal tool-call robustness probe: verify native `tool_calls` work with the chosen
  GGUF; text tool-call recovery for flaky function callers.

**Deliverables**: `sable serve`, `sable run`; README quickstart; config file v0
(`config.yaml`).

**Acceptance criteria**
- A multi-turn tool-calling conversation completes against the local model with zero
  API keys set.
- Killing the server produces a clear error, not a hang.
- Works against an external OpenAI-compatible server via explicit attach config.

**Risks**
- Tool-calling quality of small GGUFs (mitigation: text-recovery fallback; pick a
  tool-calling-strong base, e.g. Qwen-family).
- Port conflicts with servers other programs run (mitigation: dynamic ports for
  supervisor-owned servers; discovery is opt-in only).

---

## Phase 2 — Experience buffer + trace logging

**Scope**
- SQLite schema + JSONL trace writer (`buffer/`): every run appended with
  `model_version`, tool calls/results, token counts, latency, outcome fields.
- Trace capture at one canonical post-run point in the runner.
- Outcome capture v0: heuristic score + per-tool success/failure; CLI to browse
  (`sable traces list|show|stats`) and explicit feedback (`sable feedback`).

**Acceptance criteria**
- After N tasks, the buffer yields valid ShareGPT JSONL loadable by HF `datasets`.
- Trace write failure cannot break a conversation (fault-injection test).
- Every trace records the exact model/endpoint that generated it.

---

## Phase 3 — Dataset curation → SFT/DPO export

**Scope**
- Curater: score traces (heuristic + user feedback), dedup prefix-overlapping REPL
  conversations, filter low-value; optional LLM-as-judge for borderline traces.
- SFT exporter: ShareGPT JSONL (`format_type: sharegpt`).
- DPO pair miner: chosen/rejected from success-vs-failure siblings; TRL preference
  schema (`prompt`, `chosen`, `rejected`).
- Claiming/idempotency, dataset manifests (hash, counts, source trace IDs), external
  dataset import.

**Acceptance criteria**
- From ≥50 logged traces, curation produces a JSONL the training pipeline resolves and
  loads without error.
- DPO output validates against TRL dataset expectations (schema-checked in tests).
- Re-running curation is deterministic given the same buffer state.

---

## Phase 4 — Training scheduler + adapter hot-swap

**Scope**
- Training worker subprocess (torch/transformers/peft/trl; SFT/DPO/GRPO), progress
  polling, cancel, resume-from-checkpoint, job dependencies.
- Scheduler: threshold triggers (`train auto-check`), resource arbiter (VRAM check,
  disk gate), FIFO single-GPU queue (multi-GPU pinning optional).
- Model registry: lineage (`base + parent + dataset_hash → vN`), active pointer, GGUF
  export step, promote/rollback; archive-don't-delete lifecycle + janitor.
- Hot-swap: serving supervisor restart on the new GGUF; golden-task replay validation
  gate (substring + tool-call grammar checks) before promotion.

**Acceptance criteria**
- Full closed loop demonstrated end-to-end (`sable iterate`); rollback restores
  previous behavior in one command.
- Training and serving never co-reside beyond verified VRAM headroom.
- Kill -9 of the training worker leaves registry/serving consistent (crash reaping).

---

## Phase 5 — UI (full-featured web dashboard)

> **History note**: the original Phase 5 outline was replaced 2026-09-03 after review:
> the UI exposes the union of local-model-management + agent-control capabilities as
> reachable through sable's own backend. The minimal dashboard (`sable/dashboard/` —
> stdlib loopback server + `DashboardApp` JSON API + vanilla static UI) was the growth
> base for everything shipped since.

**Scope (design: see the Phase 5 design docs)**
- [`UI-ARCHITECTURE.md`](UI-ARCHITECTURE.md) — tech stack (stdlib HTTP server,
  dependency-free vanilla-JS SPA, SSE + polling hybrid), app shell, chat streaming
  design, security model.
- [`UI-ROADMAP.md`](UI-ROADMAP.md) — subphases **5a** (API layer hardening + full
  endpoint coverage) → **5b** (shell + chat/serve/models/jobs) → **5c**
  (traces/datasets/feedback) → **5d** (train wizard + iterate pipeline + settings +
  hardware) → **5e** (polish, agent profiles, schedule/auto-iterate).

**Acceptance criteria**
- Per-subphase acceptance lives in [`UI-ROADMAP.md`](UI-ROADMAP.md); cross-cutting:
  a non-terminal user can chat with the agent (streaming + tool-call rendering +
  inline feedback), control serving, watch training live, curate/dataset-review,
  and promote/rollback from the UI; the UI adds no write paths beyond the core
  APIs the CLI uses, so it can never corrupt state.
- Loopback-only bind, zero external assets (offline), optional token auth; hygiene
  tests pass over the full shipped UI tree.

**Risks**
- Scope creep into re-building other products' internals (mitigation: sable ships
  only what its own backend implements).
- Stdlib server growing beyond comfort (mitigation: thin transport layer; all logic
  stays in the transport-agnostic `DashboardApp`, which keeps everything unit-testable).

---

## Cross-cutting
- Dependency-hygiene check in CI: import allowlist only (see D-3/D-12); zero external
  assets in the shipped UI.
- Everything under `~/.sable/`; uninstall = delete dir.
- Documented commands keep `--dry-run` ergonomics everywhere first.

## Phase 6 — Multi-model serving (shipped)
N named llama-server instances run concurrently (spawn a GGUF or attach a loopback
endpoint), each with role tags, autostart and per-instance sampling overrides; profiles
bind to instances; optional capability-tier-aware auto-routing cascades small→big
(`routing.enabled`, `serving/router.py`). VRAM estimates warn (never block) on
over-commit. API/contract details: README "Phase 6 API contract notes".

## Improvement Program — WP1: bench harness + noise floor (shipped)
`sable bench` (`serving/bench.py`) measures prompt-processing and generation
tokens/sec against a supervised llama-server endpoint over N repeated temperature-0
runs (default 5), reporting per-run values plus median/min/max and run-to-run
spread `((max-min)/median)` — the noise-floor methodology (measurement discipline
inspired by kimi-k3-in-c; re-implemented inline, no imports per D-12).
`--noise-floor` persists a baseline record that WP3 (calibrate) reads back via
`bench.load_noise_floor()`; any measured effect smaller than the recorded spread
must be reported as noise. **Token-identity rule:** speed comparisons run at
temperature 0 over a fixed prompt fixture; if token streams diverge between runs
(or prompt sets differ), `compare_records()` marks the speedup INVALID and the
CLI says so explicitly. Records persist under `<SABLE_HOME>/bench/` as JSON
keyed by timestamp+label (`SABLE_BENCH_DIR` overrides); llama.cpp `timings`
are preferred over wall-clock when the server reports them. No new dependencies;
no behavior change to existing commands.

## Improvement Program — WP2: config strictness + exit codes + metric honesty (shipped)

**Exit-code taxonomy** (`core/exitcodes.py`, README "Exit codes"; established
codes untouched, new ones additive):

| Code | Meaning |
|------|---------|
| 0 | success |
| 1 | generic runtime error |
| 2 | usage error (argparse + post-parse invocation mistakes; WP1 `bench compare` bad operand keeps 2) |
| 3 | invalid configuration — consolidated report on stderr (NEW, WP2) |
| 4 | finished but corrupt artifact — flow completed, integrity check failed (NEW, WP2; kimi-k3-in-c's corrupt-output code idea, re-implemented inline per D-12) |
| 10 | `llama update --check`: update available (established) |
| 130 | interrupted (established) |

**2a Config strictness** (`core/config.py`): invalid YAML raises one
`ConfigError(ValueError)` enumerating EVERY unknown/missing/mistyped field at
once — path, expected kind, found value (redacted via `core/redact`) — instead
of failing on the first error ("refuse to guess"; kimi-k3-in-c pattern, inline).
Mistypes are always errors; **unknown keys stay warn-only by default** (stray
keys in existing configs load byte-identically) and become errors only with
`SABLE_CONFIG_STRICT=1` / `load_config(strict_unknown=True)` — the choice is
stated inside the error text itself. All valid configs load to identical
objects (`tests/test_config_strictness.py`).

**2b Corrupt-artifact checks** (`training/integrity.py`): post-export GGUF gate
(exists / non-zero / `GGUF` magic) wired into `run_iterate` (NaN loss after a
"successful" train stage and corrupt export artifacts set `PipelineResult.corrupt`)
and the NaN check into `train start`'s done path; `cmd_iterate`/`cmd_train_start`
exit 4 instead of 1 for those outcomes. Success paths unchanged.

**2c Metric-honesty audit** — every rate that can mislead now ships its raw /
honest variant beside it:

| Metric | Could mislead because | Raw variant | Honest variant | Action taken |
|--------|----------------------|-------------|----------------|--------------|
| `sable_outcome_score_mean` (overview "avg" chip) | computed only over SCORED traces — a mean over 3 of 400 reads like a whole-buffer quality signal | `sable_traces_scored_total` + `total_traces` | `sable_score_coverage_ratio` (scored/total) | new counter + gauge on `/api/metrics`; `unscored_traces` on `/api/status`; overview chip now renders "avg X (n/m scored)" |
| `sable_promotion_pass_rate` | cumulative ratio hides sample size and time window — "100% pass" can mean one decision ever | promote/fail counts (`sable_promotions_total`) | `sable_promotion_decisions_total` (denominator counter) | paired denominator exposed on `/api/metrics`; HELP text warns against small-denominator reads |
| `sable_instances_up` | attach-mode up=1 is only "state file exists" (no healthcheck in a cheap scrape) yet looks identical to a pid-verified spawn up=1 | per-instance up gauge | `sable_instance_up_strength{strength="pid_probe"\|"state_file"}` | evidence-strength split exposed on `/api/metrics` (API-level surfacing; serve panel left unchanged — panel change judged too invasive for the value) |
| `sable_request_latency_seconds` mean-side reading | summary quantiles already ship p50/p90/p95 + `_count`, so bimodality is visible IF consumers use quantiles; a mean computed client-side would hide it | `_sum`/`_count` | p50/p90/p95 quantiles (already present) | audited — no change needed; exposition already gives the honest variants (nearest-rank via `insights.percentile`) |
| `sable_traces_total{status}` counts vs rates | raw status counts carry no denominator for a failure RATE | per-status counters | client-side rate needs totals; totals are derivable from the same family | audited — honest as exposed (pure counts, no fabricated ratio); dashboard traces strip already shows "n failing" over its own scanned total |

No KV-cache-hit-rate metric exists anywhere in the stack (audited: metrics.py,
insights.py, panels) — the program-brief canonical example has no sable
analogue to fix; the coverage/denominator/evidence pairs above are this codebase's
instances of the same failure class. No new dependencies; successful command
outputs byte-identical (`tests/test_exit_codes.py`, `tests/test_metric_honesty.py`).

## Improvement Program — WP3: `sable calibrate` per-machine autotune (shipped)

New module `serving/calibrate.py` + CLI `sable calibrate`: a **coordinate-wise
staged sweep** (baseline → test each candidate against the current best, keep
the winner, move on — bounded runtime, knob interactions explicitly not
explored) over run_settings knobs already mapped in `serving/run_args.py`:
`kv_cache_type` {f16,bf16,q8_0,q4_0}, `flash_attn` on/off, `gpu_memory_pct`
variants (→ scaled `-ngl`), `batch_size`/`ubatch_size`, `parallel` (`-np`), and
`speculative_type` {off,ngram,mtp}. The candidate grid is data-driven
(`KNOB_GRID`/`QUICK_GRID`) and bounded via `--knobs k1,k2` and the `--quick`
preset. Every measurement rides the WP1 bench harness (fixed prompt set,
temperature 0, default 3 runs/candidate).

**Measurement methods — how wins are judged:**

| Rule | Definition |
|------|------------|
| Adoption threshold | A candidate is adopted **only if** its median gen tok/s beats the current best by MORE THAN `max(3%, noise_floor_spread_pct)`. The noise floor is the newest persisted `kind=noise_floor` bench record (`bench.load_noise_floor()`); when none exists, calibration first repeats the baseline settings and persists a noise-floor record exactly like `sable bench --noise-floor`. Effects inside the floor are recorded `rejected`, never claimed as wins. |
| Token identity | Carried over from `bench.compare_records`: identical prompt-set hash + identical temperature-0 token streams between best and candidate. Any divergence ⇒ verdict `invalid` — a faster setting that changes tokens is NEVER adopted, only recorded. |
| Error handling | A candidate whose server fails to start or crashes mid-bench is recorded `error`, skipped, and the sweep continues (never aborts). If even the baseline cannot be measured, calibration exits 1 with nothing saved as adopted. |
| Speculative honesty | When `mtp` is requested but no drafter GGUF exists next to the model, `run_args.speculative_args()` falls back to ngram-mod; the profile records `tested_as: "ngram (mtp unavailable)"` so no later reader can claim MTP was measured. |

**Profiles**: JSON under `~/.sable/calibrate/` (`SABLE_CALIBRATE_DIR`
override), one file per machine-id + model-hash, atomic write. Fields:
`profile_version`, `machine_id`, `model_hash`, `model_path`, `timestamp`,
`runs_per_candidate`, `threshold_pct`, `noise_floor {spread_pct, record,
source}`, `baseline_settings`, `adopted_settings`, and per-knob candidate tables
(value → median gen tok/s, delta_pct vs running best, verdict
`adopted|rejected|invalid|error|baseline`, `tested_as` for spec fallback).
Machine id: `SABLE_MACHINE_ID` (test hook) > `/etc/machine-id` > sha256 of
hostname+machine. Model hash: sha256 over the first 1 MiB + file size — cheap
enough for the launch path (full-file hashing like `core.provenance.hash_file`
is too slow on multi-GB weights); it is a profile lookup key, not an
attestation.

**Serving integration — strictly opt-in**: new config section `serve:` with
`calibration: off|auto` (`off` default; unknown values fall back to off). With
the key absent the launch path is byte-identical (existing serving tests pass
unchanged). With `auto` AND a matching profile, `apply_calibration(cfg,
model_path)` supplies the adopted knob values as defaults underneath explicit
config/instance `run_settings` — manual explicit values ALWAYS win. Consumed at
the two argv assembly sites: `supervisor.build_instance_cmd` and
`deploy._start_server_detached_once`.

**CLI**: `sable calibrate [--quick] [--knobs k1,k2] [--runs N] [--model REF]
[--attach URL] [--list] [--show] [--clear]` — human-readable summary table
(knob | candidate | median tok/s | delta% | noise floor | verdict). Exit 0 on
success, 1 when nothing could be measured, 2 for usage mistakes (unknown knob,
no model, non-loopback attach). `--attach` measures a foreign server that never
receives the candidate knobs; such profiles record
`endpoint_mode: "attach"` to stay honest (harness/testing use).

**Dashboard**: intentionally NOT surfaced (decision: surfacing calibration
state in the serve panel would require new API + panel plumbing — invasive for
optional value; `sable calibrate --show` covers it).

**No real-machine numbers are claimed.** No real llama-server/GPU calibration
was executed in this environment; everything above is validated by hermetic
fake-server / injected-measurement tests (`tests/test_calibrate.py`, 24 tests)
only. The first real run on target hardware will produce the first actual
profiles and noise floors.

## Improvement Program — WP4: engine watchdog — frozen llama-server detection (shipped)

New module `serving/watchdog.py` (concept inspired by Strata's watchdog;
re-implemented inline, no imports per D-12). A supervised `llama-server` child
can wedge mid-request (deadlock / driver stall) while its pid stays alive —
`runtime.pid_alive` reads "healthy" forever and every dependent run hangs. The
watchdog samples `/proc` (stdlib file IO; **zero new network paths** — it never
probes HTTP) and recovers frozen engines automatically.

**Detection rule — three-condition AND over a window** (default 90 s, sampled
every `watchdog_sample_interval_seconds` = 5 s). An instance is FROZEN only
when ALL hold while at least one request is in flight against it:

| # | Condition | Source | "≈0" default |
|---|-----------|--------|--------------|
| a | no response progress — a request continuously in flight for the whole window, none completed | in-flight counter (see below) | — |
| b | process CPU-time delta ≈ 0 | `/proc/<pid>/stat` utime+stime (clock ticks / SC_CLK_TCK; comm parsed after the LAST `)`) | < 10 ms per sample (`watchdog_cpu_threshold_ms`) |
| c | disk-IO delta ≈ 0 | `/proc/<pid>/io` read_bytes+write_bytes; falls back to rchar+wchar when those keys are absent (older kernels); an unreadable io file means **no evidence → never kill on it** | ≤ 0 bytes per sample (`watchdog_io_threshold_bytes`) |

Why CPU+IO deltas and not just "no response": a healthy-but-slow server
genuinely computing a long prompt burns CPU steadily, so condition (b) fails —
it is **never killed**. A wedged server sits at zero CPU *and* zero IO while a
request is parked on it. Silence with no request in flight is idleness, not a
freeze: (a) gates the rule so idle servers are never touched. Missing evidence
on any axis reads "unknown", and unknown never kills.

**In-flight tracking (least-invasive correct approach)**: requests to a served
endpoint flow through exactly two places — the agent runner's chat calls
(`core/agent_runner.py::_chat`, streaming included) and the dashboard chat
proxy (`dashboard/app.py::h_proxy_chat`). Each wraps its upstream call in
`watchdog.track_request(base_url)`: a lock-guarded per-endpoint counter with
the time it left zero, giving `inflight_age()`. Streaming progress needs no
extra hooks — a streaming generation burns CPU/IO, so (b)/(c) already capture
"progressing". The tally is inert (a dict counter) unless the watchdog runs.

**Recovery**: on detection the pid is re-verified as sable-owned via
`runtime.terminate_owned_pid` — the WP4-extracted shared verify-then-kill
helper (`pid_alive`: ours → SIGTERM, SIGKILL after grace; foreign/recycled →
never killed, log + clear stale state) now also used by
`deploy.stop_owned_server` and `instances.stop_recorded`. In-process children
go through the supervisor's owned-handle registry (the strongest ownership
proof). Restart uses exponential backoff (`watchdog_restart_backoff_base_seconds`
= 2 → 2s, 4s, 8s…) capped at `watchdog_max_restarts` = 3 within
`watchdog_restart_window_seconds` = 600; a cap hit logs loudly and emits
`restart_capped`, after which further freezes for that instance are ignored —
no infinite loop. Restarts go through `start_instance_server`, so the existing
crash-loop budget (#28) is a second layer under the watchdog's own cap.

**Events** (dashboard bus kind `serve.watchdog`; the serve panel already
renders serve events — no invasive UI work): `freeze_suspect` (first stalled
sample of a window), `freeze_detected`, `instance_killed` (pid verified),
`stale_state_cleared` (foreign pid — never killed), `instance_restarted`
(attempt n, backoff s), `restart_failed`, `restart_capped`.

**Opt-in rationale**: a loop that KILLS processes must never run uninvited.
`serve.watchdog_enabled` defaults to **false**; with no config changes
`start_watchdog()` returns None immediately and every existing serving test
passes unchanged (default-off proof). Non-Linux / unreadable `/proc`: logged
once at boot, detection unavailable, nothing is ever killed.

**Config** (`serve.*`, documented in `config.example.yaml`):
`watchdog_enabled` (false), `watchdog_window_seconds` (90),
`watchdog_sample_interval_seconds` (5), `watchdog_cpu_threshold_ms` (10),
`watchdog_io_threshold_bytes` (0), `watchdog_kill_grace_seconds` (8),
`watchdog_restart_backoff_base_seconds` (2), `watchdog_max_restarts` (3),
`watchdog_restart_window_seconds` (600). Read leniently: unknown keys ignored,
invalid values fall back to defaults with a warning.

**No real frozen-llama-server repro is claimed.** Recovery timing is proven by
hermetic tests with injected clocks and fabricated `/proc` readings
(`tests/test_watchdog_wp4.py`, 21 tests): freeze→kill→restart-backoff, busy-but-slow
never killed, idle never triggers, foreign pid never killed (stale state cleared),
backoff sequence + cap with no infinite loop, and config default-off.

## Improvement Program — WP5: GPU coexistence — idle unload / lazy load (shipped)

Inspired by Strata's idle-unload/lazy-load coexistence pattern — re-implemented
inline (`serving/coexist.py`), no imports from it (D-12). The problem: a served
llama-server holds its model resident in VRAM while nobody chats with it, and
the training scheduler's resource gate then refuses jobs for lack of free VRAM.
Three opt-in mechanisms plus one handshake close the loop; **every key defaults
off** (`idle_unload_seconds: 0`, `lazy_load: false`, `min_free_vram_mib: 0`,
`coexist_policy: refuse`) and with no config changes `coexist.build()` returns
None — all pre-existing serving/dashboard/scheduler tests pass unchanged.

**Idle unload** (`serve.idle_unload_seconds > 0`): a supervisor-side sibling of
the WP4 watchdog loop (injectable clock/sleep; `step()` + `Event.wait(interval)`
— no busy-poll) audits running instances each tick. Activity reuses the WP4
in-flight registry: request starts stamp a per-endpoint last-activity clock
(`watchdog.note_activity` / `idle_seconds`), so no new hooks were needed on the
request paths. An instance idle past the window is gracefully stopped through
the existing owned-stop paths — in-process children via
`supervisor.stop_instance`, cross-process pids via `instances.stop_recorded`
(SIGTERM first; SIGKILL only via those paths' grace escalation) — and its state
file becomes an `unloaded` tomb (pid removed, reason + timestamp recorded) so
the dashboard shows `load_state: unloaded`. A tick never unloads while the
in-flight counter for that endpoint is > 0, and deliberately does not refresh
the stamp for in-flight endpoints (a server busy with one long request becomes
idle when that request ends). First sighting of an endpoint seeds the window
from now.

**Lazy load** (`serve.lazy_load: true`): the resolve path
(`dashboard/app.py::_resolve_chat_target`, both the named-instance branch and
the default serve-state branch) detects the `unloaded` tomb and transparently
re-launches through the canonical `start_instance_server` machinery (crash-loop
budget intact), blocking briefly — cold-start latency accepted by design — then
forwards. The agent runner adds a second net: a connection failure whose URL
matches an unloaded instance triggers exactly ONE reload + retry
(`agent_runner::_chat`). With `lazy_load: false`, requests to an unloaded
instance fail with a clear message naming the load command / lazy_load opt-in;
a never-loaded instance keeps its pre-WP5 503 verbatim.

**Explicit control**: `POST /api/servers/<id>/load` and
`/api/servers/<id>/unload` (unload refuses 409 `request_in_flight` while a
request is tracked), instance rows carry `load_state`
(loaded|unloaded|unloading|stopped), and the CLI gains
`sable instances status|load|unload <id>`. Serve-panel JS surfacing beyond
the JSON field was skipped (non-invasive-UI rule; the field is ready for it).

**Train+serve handshake — the algorithm** (`min_free_vram_mib > 0` AND
`idle_unload_seconds > 0`; wired via `Scheduler.coexist` → `_gate_pass`):

1. Evaluate the ordinary resource gate. Pass ⇒ done (rescue never consulted;
   byte-identical path).
2. Fail ⇒ compute `required = max(training.min_free_vram_mib, job estimate)`
   and probe live free VRAM (`scheduler.gpu_free_mib` — one truth per number).
3. If the failure is not a VRAM shortfall (free ≥ required ⇒ AC-power block)
   ⇒ refuse exactly as today.
4. Projected free = live free + Σ footprint estimates of currently-running
   spawn instances (same Phase-6 heuristic `estimate_model_vram_mib`;
   unestimable instances are excluded — never claim VRAM we cannot name).
5. Projected < required ⇒ refuse (unloading could not have helped; no event,
   no unload — even under policy=unload).
6. Policy `refuse` (default) ⇒ emit `coexist_refused`
   {free_mib, required_mib, projected_free_mib, instances} with
   `would_pass_if_unloaded: true`; refuse exactly as today. Nothing is stopped.
7. Policy `unload` ⇒ emit `coexist_unload_triggered`, gracefully unload all
   running instances (reason=coexist; in-flight endpoints skipped), then poll
   the VRAM probe (bounded by `coexist_vram_wait_timeout_seconds` = 60, every
   2 s) until free ≥ required. The caller re-evaluates the REAL gate once —
   an honest second refusal beats a fake admission.

**Race policy (documented choice)**: unload and load take one lock per instance
id, so they serialize. A request arriving mid-unload blocks on that lock; when
it proceeds, lazy_load reloads transparently and forwards, or (lazy off) gets a
clear retryable error naming the instance — no double-launch is possible (the
queued caller finds the fresh live record and no-ops; proven by a threaded
test). The scheduler's VRAM wait happens after the unload locks are released,
so a lazy reload racing a training launch can lose the VRAM race — llama-server
then fails its healthcheck with an honest ServeError rather than corrupting
anything; run `lazy_load: false` alongside `coexist_policy: unload` for strict
train-first behavior.

**Events** (dashboard bus kind `serve.coexist`): `instance_unloaded`
{reason: idle|manual|coexist}, `instance_loaded` {reason: lazy|manual},
`coexist_unload_triggered`, `coexist_refused` (headroom numbers).

**Config** (`serve.*`, documented in `config.example.yaml`):
`idle_unload_seconds` (0), `lazy_load` (false), `min_free_vram_mib` (0),
`coexist_policy` (refuse); tuning knobs `coexist_sample_interval_seconds` (5),
`coexist_unload_grace_seconds` (8), `coexist_vram_wait_timeout_seconds` (60),
`coexist_vram_poll_interval_seconds` (2). Read leniently like the WP4 keys.

**No real GPU repro is claimed.** VRAM-free timing is faked in tests: injected
clocks, scripted free-VRAM probes, monkeypatched spawn/stop paths
(`tests/test_coexist_wp5.py`) prove idle-fire-once, never-mid-request, lazy
reload + retry-once, endpoint/CLI state round-trips, the full gate matrix
(no-wiring / refuse+event / unload+admit / still-insufficient) and the no-
double-launch serialization — without hardware.

## Improvement Program — WP6: MCP control plane — `sable mcp` (shipped)

Inspired by Strata's MCP control-plane pattern — re-implemented inline with
**zero new dependencies**: no MCP SDK, stdlib-only (`json`/`sys` framing in
`sable/mcp/server.py`); `tests/test_license_hygiene.py` keeps the import
allowlist green. New opt-in command: nothing existing changes when
`sable mcp` is not invoked; defaults byte-identical.

**Transport — stdio, no socket.** JSON-RPC 2.0 line-delimited over
stdin/stdout (MCP's canonical local transport): the client spawns
`python -m sable mcp` and pipes frames. No listener exists, so the
loopback-only rule holds by construction; zero cloud, zero keys. Single-
threaded synchronous request loop (documented: spec-compliant, keeps the
confirm store trivially safe). stdout carries JSON-RPC only; diagnostics go
to stderr. Protocol surface: `initialize` (answers `protocolVersion`,
`serverInfo`, `capabilities.tools`) / `notifications/initialized` / `ping` /
`tools/list` / `tools/call`.

**Tools mirror existing code EXACTLY — no new authority.** Read tools call
the same in-process `DashboardApp.handle` the HTTP layer adapts
(`status`, `instances_list`, `models_list`) or read the same log files the
CLI reads (`logs_tail`, bounded 500 lines, ids validated + contained via
`core/path_security.py` — no path argument is ever accepted). Mutating tools
call the same endpoints/executors: instance start/stop/load/unload (WP5),
`curate_enqueue`, `train_enqueue` (single-flight 409 `already_running`
surfaces as a tool error, never a silent queue), `promote`, plus `validate`
and `bench`. Classification judgment calls documented in tools.py + README:
`validate` and `bench` write no registry state but spawn llama-server /
occupy the GPU (and bench persists records) — process side effects make
them mutating by policy.

**Plan-then-confirm for every mutating tool** (`sable/mcp/confirm.py`):
first call returns PLAN `{action, target, exact_effect, risks, params}` + an
opaque `secrets`-random token; second call with the token executes. Tokens:
in-memory store only (restart drops plans), TTL 300 s, single-use (consume
deletes atomically; reuse refused), bound to `(tool, sha256(canonical-json
of tool+args))` — replay for a different tool or different args is refused
without burning the pending plan. Unknown/expired/mismatched tokens are
clean refusals, never crashes.

**Strict fixed schemas.** Every tool declares `additionalProperties:false`
with scalar types + enums; a tiny inline validator (`mcp/schema.py`) refuses
unknown args / missing required / wrong types (no coercion beyond int-satisfies-
`number`) with JSON-RPC -32602 before any executor runs. Nothing is ever
passed to a shell — executors import and call Python directly; a test greps
the package for `shell=True`/`os.system`/subprocess/exec patterns.

**Tests** (`tests/test_mcp_wp6.py`, hermetic, in-process): handshake + full
schema completeness across the 13-tool table, every validation rejection,
plan-no-side-effect → execute-exactly-once → reuse/expiry/wrong-tool refusal
matrix (injectable clock), train-enqueue 409-as-error via the dashboard
conflict contract, logs_tail traversal refusal, no-shell + stdlib-only
source scans, `mcp --help`, and malformed-JSON → -32700 without a crash.

## Improvement Program — WP7: agent playbook + repo rules (shipped)

**Docs.** [`docs/AI_SETUP.md`](AI_SETUP.md): agent-facing install-and-operate
playbook with copy-pasteable command blocks and expected outputs (hermetic ones
actually executed; GPU/model-dependent steps describe their output shape
honestly) — prerequisites, install (`setup.sh` / `pip -e .` + `[training]`),
first-run config, the record→curate→train→export→promote→serve loop, bench/
calibrate/coexistence/watchdog features, MCP client wiring with the plan-then-
confirm flow, and a troubleshooting table keyed to the exit-code taxonomy
(exit 3 consolidated config report, exit 4 corrupt artifact, llama-server
missing → `sable setup`, port conflicts, VRAM-gate refusal → coexist/
instances unload, frozen server → watchdog, stale calibration profile,
token-identity INVALID on bench compare). [`AGENTS.md`](../AGENTS.md): the two
cardinal rules (opt-in features keep outputs byte-identical; no claims below
the measured noise floor), the D-12 licensing boundary (re-implement inline,
never import reference projects — `tests/test_license_hygiene.py` enforces),
dependency policy, test/lint commands, exit-code pointer, and how to add a
config key.

**`sable doctor --env`** (`sable/doctor.py`, stdlib-only like dashboard/
MCP; kimi-k3-doctor's measure-don't-assume pattern re-implemented inline per
D-12). Existing `sable doctor` (runtime GPU/backend diagnosis) is untouched;
`--env` opts into the measurement report (`python -m sable.doctor` runs it
directly — no new console-script surface). Checks: python version range; core
deps + `[training]` extras detection; config parse via `load_config` with the
consolidated ConfigError printed VERBATIM (exit 3); `SABLE_HOME` writability;
free disk via `shutil.disk_usage` on the SAME filesystems export/training write
to (home, configured `training.output_dir`, temp/staging dir — deduped by
device) against the largest registered GGUF + safety margin (max(2 GiB, 10%));
MemAvailable from `/proc/meminfo` → serve-context and train-batch presets with
the tier formulas printed; free VRAM via a local `nvidia-smi` call when present
(absence reported honestly) → `-ngl`/context tiers; llama-server discovery
reusing the existing runtime path; loopback bindability of configured ports
(live sable instances excuse their own ports); stale instance state files —
reported with exact cleanup, and `--fix` removes ONLY entries whose pid is
deterministically gone (ownership rule: foreign/alive-but-not-ours pids are
never touched). Findings emit copy-paste next commands (`sable bench
--noise-floor` when no persisted baseline; `sable calibrate --quick` when
`serve.calibration: auto` but no profile; `sable setup` when the binary is
missing). Exit codes per taxonomy: 0 ok (warnings allowed), 1 actionable
problems, 3 invalid config.

**Tests** (`tests/test_doctor_wp7.py`, hermetic): every probe injectable — tmp
`SABLE_HOME`, fake `disk_usage`, fixture meminfo text, stubbed pid-alive —
covering low-disk flagging on export paths, RAM/VRAM preset math on fixed
inputs, next-command suggestions for missing noise-floor/calibration states,
config-error passthrough (exit 3, verbatim report), stale-state detection with
no deletion by default and `--fix` respecting the ownership rule.

## Improvement Program — WP8: tool-call / reasoning-loop recovery hardening (shipped)

No new config keys; defaults keep every pre-existing parse path byte-identical
(the pre-WP8 recognizer is kept verbatim as `parse_legacy` and runs FIRST —
recovery only ever produces calls for inputs that previously yielded `[]`, so
nothing that parsed before can parse differently). kimi-k3's repeated-stop-
token / reasoning-loop recovery patterns and its drift-corpus + gate-ladder
validation discipline are the inspiration, re-implemented inline per D-12.

**Mechanism 1 — repeated stop-token stripping** (`strip_stop_tokens(text,
extra_tokens=...) -> str` in `core/tool_calls.py`). Iterative tail-band
stripping until a fixed point (capped at `MAX_STRIP_ITERATIONS=64`, monotone
so the cap never bites real output): pipe-delimited and bare-angle stop
families (`<|im_end|>`, `<|endoftext|>`, `<eot_id>`, …), duplicated CLOSING
role headers (`</assistant>`), separator junk interleaved between markers
(box pipes, `*_~#=-` runs), acknowledged bracketed markers (`[DONE]`,
`<STOP>`) and separator-guarded end-anchored stop words. Preservation is
structural: the right-to-left walk over a bounded 4 KB tail window only ever
strips an EOF-ANCHORED band (a stop-shaped sequence mid-content can never be
reached), junk between markers is capped (`MAX_BAND_JUNK_CHARS=24`), bare
separator atoms may not chain a run across real content (the `_` of
`todo_read` stays), and colons/commas are never eaten. A refined gate on the
RECOVERY path only: `_markup_residue_clean` lets recovery proceed when every
leaked pipe-delimited marker sits inside stop junk (removing all bracketed
band atoms leaves no template shape behind); template STRUCTURE (`im_start`
is not a band atom) keeps the legacy blanket refusal, and `parse_legacy`
itself is unchanged.

**Mechanism 2 — reasoning-loop detection** (`core/recovery.py`:
`detect_reasoning_loop(text) -> LoopVerdict(loop, repeated_unit, repeats,
start, reason)` + `truncate_at_loop`). Three conservative heuristics over the
think-scrubbed text (the ONE `strip_think_blocks` scrubber runs first, so
loops around leaked reasoning blocks are judged on visible text): exact
normalized paragraph repeats, near-duplicate laps via difflib with a
quick_ratio prefilter and greedy representative clustering (hard budget of
400 ratio calls → linear-ish), and hedging cycles ("Wait… actually… no, hold
on…"). Every heuristic requires BOTH a repeat count AND a dominance share of
the whole text. False-positive guards: paragraph granularity means lists/code
blocks/tables are single units never compared lap-by-lap; bullet/numbered
units are excluded from both repeat passes entirely (structured inventories
share long verbatim stems and differ only in an index — ratio() > 0.95 on a
healthy changelog is normal); two laps stay below threshold; short text fails
open. All detection failures fall open to `loop=False` (a missed loop beats
truncating a good answer).

**Consumer policy** (`core/agent_runner.py`): loop detected AND no tool call
parsed → truncate at the first lap (`…[reasoning loop truncated]` marker) and
REUSE the verification-nudge mechanism shape — one synthetic user turn, at
most one per run (`_loop_nudges_sent`), event `agent.loop_nudge` — not a
second retry path; if the verification nudge already owned the turn the loop
nudge never fires. Guardrails opt-out (`guardrails_enabled: false`) disables
the whole mechanism, byte-identical pre-WP8 behavior.

**Mechanism 3 — malformed-invocation repair** (`recover()` +
`parse_text_tool_calls_with_repairs(content, known_tool_names=…) ->
ParseReport`). Pass order (first producing calls wins): stop-token strip →
legacy fence spans with JSON repair → fenced payload followed by stop junk
before the fence close → unclosed fences → `<tool>`/`<function>` wrappers and
attribute/parameter-tag bodies (`<parameter name="k">v</parameter>`) → bare
balanced JSON in prose. Repair cascade (constant four linear passes, tags
accumulate): Python `True/False/None` literals → trailing commas →
single-quoted strings → truncated-close (dangling key/value trimmed at EOF —
a value cut mid-string is DROPPED, never fabricated). Every emitted call is
attributable via `ParseReport.repairs_by_call_id` (`tags_for(call)`; tags:
`stop_tokens, unclosed_fence, bare_json_prose, xml_tool_wrapper,
xml_parameters, py_literals, single_quotes, trailing_commas, truncated_close`,
plus `name_normalized`/`truncated_candidate`) — a side-channel keyed by the
unique call id, so the wire shape of the call dicts never changes. Name
routing is EXACT-normalized only (lowercase/underscore folding): unique fold
routes with the `name_normalized` tag; typos like `serch_files` are NOT
routed and ambiguous normal forms refuse — a mis-routed tool call executes
the wrong thing, a refused one just re-prompts. Attribution guards keep plain
data JSON pasted in prose unparsed (name + arguments-ish key required).
`strip_tool_call_markup(…, extended=True)` strips the new XML shapes for
recovered turns only; default stays legacy-verbatim.

**Measured recovery rates** (`tests/test_tool_call_recovery.py::test_layer3_
recovery_rate_report`, printed each run; baseline = `parse_legacy` live on
the 17-fixture parse corpus in `tests/fixtures/tool_drift/`):

    class                n baseline(pre-WP8)  new(WP8) recovered+
    malformed_json       7                 0         5         +5
    mixed                3                 0         2         +2
    stop_token_repeat    4                 1         4         +3
    xml_wrapped          3                 0         3         +3
    TOTAL               17                 1        14        +13
    recovery rate: baseline 5.9% -> new 82.4% (+76.5 percentage points)

All 14 curated positive expectations recovered (100% of targets; the 3
negatives stay refused), ZERO regressions asserted (every fixture
`parse_legacy` handled parses to identical name+arguments). The 6
reasoning-loop fixtures gate separately: all 4 loop positives detected, both
healthy negatives (list, code) clean. Gate ladder in one test file: LAYER 1
weightless corpus/manifest scan, LAYER 2 per-fixture parse + attribution +
zero-regression assertions, LAYER 3 the printed aggregate table with strict
gates (strictly-more-recovered AND new == expected). Performance sanity: a
~290 KB adversarial input (repaired-corrupt fences × 2600 + a 5000-marker
stop storm) parses/strips/detects in well under one second (measured ~0.1 s;
all candidate scans bounded — `MAX_BARE_CANDIDATES=16`, span/band/junk caps).

**Tests** (`tests/test_tool_call_recovery.py`, 57 tests, hermetic): stop-token
edge cases incl. mid-content preservation and idempotence; loop-detection TPs
(exact/near/hedging/interleaved think blocks) and TNs (30-item inventory,
repetitive code, two-lap text); each repair class with attribution; name-
matching exactness + ambiguity refusal; runner E2E nudge-once (and budget
never spirals when the model stays stuck); guardrails-off byte-identity; the
gate ladder and perf test above. All pre-existing tool-call/agent tests pass
unmodified; full suite 1711 green, ruff clean, no new dependencies.

## Improvement Program — WP9: curation analytics + validation telemetry + Prometheus /metrics (shipped)

Additive work package: no config keys, no new dependencies (the exposition
format stays hand-rolled — prometheus_client is forbidden), and every existing
command's observable output is unchanged. Analytics persistence is best-effort
(a failed write can never break the run that produced it).

**9a — curation funnel analytics.** After each `Curater.curate()` succeeds, one
funnel snapshot appends to `<home>/analytics/curation_funnels.jsonl`
(`store/analytics.py`). Storage design: append-only JSONL per run rather than
a buffer-DB table — the curater already treats the filesystem (dataset dirs +
manifest.json) as its artifact store, history sizes are personal-loop scale,
reads are "tail the file", and a schema migration for two append-only streams
would buy nothing. Record schema: `ts, dataset_id, name, rebuild, created_at,
trust, project, model_alias, agent_profile` + funnel stages `traces_scanned →
loaded → eligible → (dedup_rejects / quality_rejects / limit_dropped) →
claimed → curated` with `claim_race_lost` and `dpo_pairs`, plus a 10-bin [0,1]
`score_histogram` carrying honest denominators (`scored`/`unscored`/`total`
beside the buckets — WP2c: NULL scores counted, never coerced to 0). Two runs
accumulate; per-dataset diffs over time reuse `curate/dataset_diff.py` via
`GET /api/analytics/curation?dataset=…&limit=…` (`latest_diff` = diff of the
two newest distinct datasets in the filtered history; logic never duplicated).

**9b — validation-gate telemetry.** The training pipeline appends a compact
summary per gate run (model_id, iteration where available, passed/failed/
skipped, `ran_tasks` denominator excluding transport-skips, pass_rate, gate
verdict) to `<home>/analytics/validation_runs.jsonl`. `GET
/api/analytics/validation?model=…` merges that history chronologically with
promote/rollback events read live from the hash-chained registry ledger (the
ledger stays the source of truth; analytics never re-writes it) into one
per-model `timeline`.

**9c — Prometheus /metrics.** `GET /metrics` is an alias of the existing
`GET /api/metrics` on the dashboard's SAME loopback server (no new listener,
no new port); token auth + CSRF/DNS-rebinding gates apply identically — no
scrape backdoor. Content-Type `text/plain; version=0.0.4`; label escaping per
spec (`\\`, `\"`, `\n` inside values). Caching: the rendered body is memoized
for `METRICS_TTL_S = 5.0 s` (one recomputation per TTL at a 15 s scrape
interval; `?refresh=1` forces), computed from one read-only connection +
indexed counts — no per-scrape full scans heavier than an indexed count.

Metric name mapping (vLLM-compatible names are inspiration only, re-implemented
inline per D-12; mirrors exist ONLY where semantics truly match):

| sable:* | vllm mirror? | semantics note |
|---|---|---|
| `sable_requests_total` (new) | NO | chat-proxy agent runs, not engine completions — one trace can span N completions; mirroring as `vllm:num_requests_total` would lie |
| `sable_tokens_in_total` | `vllm:prompt_tokens_total` | prompt tokens processed — same semantics, identical value from one source |
| `sable_tokens_out_total` | `vllm:generation_tokens_total` | generation tokens processed — same semantics, identical value |
| `sable_request_latency_seconds` | NO (`vllm:time_to_first_token_seconds` deliberately absent) | sable records end-to-end duration only, never TTFT; no honest histogram buckets exist, so absence is documented in HELP instead of faking samples |
| `sable_jobs_queued` / `sable_jobs_running` (new gauges) | NO (`vllm:num_requests_waiting` absent) | training/curate job-queue depth — a different queue than the vLLM scheduler's waiting-request queue |

Honesty rule enforced by test: every rate family keeps its WP2c denominator
counter beside it (`sable_score_coverage_ratio` ↔ `sable_traces_scored_total`;
`sable_promotion_pass_rate` ↔ `sable_promotion_decisions_total`), and all
pre-existing families render unchanged.

**Dashboard surfacing (minimal, additive).** Datasets panel gains a "Curation
funnels" card (newest-first funnel table with scanned∝width / curated-fill bars
and the histogram denominator column); the Models drawer gains a "Gate trend"
block (validation pass-rate + verdict rows merged with promote/rollback events,
newest first). Both render honest hints on empty/unavailable history. Skipped
as too invasive: a dedicated analytics route/panel and sparkline canvas — the
APIs are the primary surface.

**Tests** (`tests/test_wp9_analytics.py`, 12 tests, hermetic): funnel snapshot
persistence with honest denominators; two-run accumulation + dataset filter;
dedup rejects counted on a prefix-overlap dup fixture; funnel API history +
latest diff; validation summary skip-counting; chronological merge of gate runs
with ledger promotion events; exposition re-parsed by a stdlib parser (HELP/
TYPE adjacency, no orphan samples); label escaping with a quote/backslash
instance id; vllm mirrors present only where semantics match (and absent where
they don't); no rate without denominator; HTTP-level `/metrics` 200 + content
type over the real server, TTL-body parity with `/api/metrics`, and token-auth
parity (401 unauthenticated / header / `?token=` / cross-site 403). Full suite
1723 green, ruff clean, pyproject untouched.

## Improvement Program — WP10: streamed-LoRA trainer — step 1 SPIKE (done; steps 2–3 gated)

Step-1 scope only: prove the four mechanisms of a streamed-LoRA trainer for
checkpoints that exceed VRAM, re-implemented inline (D-12; airllm read as
reference, cited in docstrings, never imported). New opt-in package
[`sable/training/streamed/`](../sable/training/streamed/) — `streaming.py`
(meta shell + bounded-window layer streaming with sync/threaded prefetch from
safetensors), `lora_linear.py` (PEFT-free LoRALinear keeping the frozen base
`weight` Parameter in place → checkpoint-shaped state_dict for the later
GGUF-LoRA mapping), `recompute.py` (no_grad forward + per-layer backward
recompute, CPU-held autograd state, dropout p>0 rejected by construction),
`chunked_ce.py` (vocab-chunked linear CE with online logsum-exp; lm_head
frozen by design). Nothing wired into `worker.py` — Step 2 is a separate gated
work package.

**Findings note**: [`docs/SPIKE-WP10.md`](SPIKE-WP10.md) — this machine has
torch 2.14.1 CPU (no CUDA), so ALL numeric proofs executed here on real
tensors: streamed grads == dense autograd reference (fp64), loss decrease over
15 streamed steps, chunked CE == `F.cross_entropy` incl. ignore_index and
d_hidden grads, LoRA frozen-base + output identity, residency ≤ 1+depth in
sync AND threaded prefetch, real safetensors shards written in-test. CUDA-only
paths (empty_cache eviction, device copies) code-reviewed-only; no perf claims
(CPU box). Two documented test exemptions: heavy-import allowlist now lists
`training/streamed/*` (function-level only, module-level still banned globally)
and `test_no_orphans.py` ALLOWLIST marks the package a deliberate not-yet-wired
seam (delete when Step 2 wires it). **GO for Steps 2–3, conditional** on:
single pinned transformers arch adapter first, GPU measurement before any
speed claim, Step 3 built against the frozen `lora_state_dict` name→tensor
contract. Full suite 1744 green, ruff clean, pyproject untouched.

## Improvement Program — WP10 steps 2–3: streamed backend integration + artifact pipeline (done; GPU acceptance gated)

Step 2 wired the spike into the worker behind an opt-in `training.backend`
(`peft` default = today's path byte-identical; `streamed` routes sft jobs to
`streamed/runner.py`). Plumbing: `TrainingConfig.backend` ->
`_resolved_train_yaml` injects `training.backend` into the resolved train.yaml
(job spec) -> the spawned worker selects the branch in `run_sft`; spawn path
untouched. Mistyped values warn and fall back to `peft` (WP2 lenient-enum
convention). Both step-1 test exemptions removed: the package is now
statically reachable from the worker root (`test_no_orphans.py` ALLOWLIST
back to empty), heavy imports stay function-level inside the worker seam.

Streamed branch (all heavy imports lazy): split an on-disk HF checkpoint dir
(NEVER downloaded) into per-layer safetensors shards with progress lines
(`splitting checkpoint: shard i/n`) streamed to the job log -> meta shell via
the pinned llama/qwen3 arch adapter -> `inject_lora` ->
`StreamedRecomputeTrainer.step` loop -> `chunked_linear_cross_entropy`.
Dataset input is the existing ShareGPT `sft.jsonl` path through
`masking.mask_conversation`: per-turn -100 masking STRICTER than airllm's
messages loader (which masks only up to the LAST assistant turn) — ALL
non-assistant turns including system/tool spans stay masked, consecutive
same-role turns each keep their own span. Per-step CSV
`train_metrics.csv` with schema exactly `step,loss,peak_vram_gb,elapsed_s`;
on CPU-only boxes peak is an honest 0.0 (documented).

Step 3 artifact: adapter saved as plain lora-state-dict (module-name ->
tensor) — the FROZEN contract (D-13) — and converted by the inline stdlib-only
GGUF-LoRA writer (`streamed/gguf_lora.py`, D-12: llama.cpp format re-implemented,
llama.cpp read as reference). `run_export` relocates the artifact BEFORE the
llama.cpp merge/quantize path (adapters are F32 and -lora-loaded, no quantize),
so registry register/export/-lora hot-swap flow unchanged; provenance records
`backend: streamed` additively (peft runs stay byte-identical).

**Proven HERE (CPU-only box)**: split->shell->train loop end-to-end on a tiny
synthetic llama-shaped checkpoint built from locally written safetensors
shards — loss decreasing 4.16 -> 3.11 over 12 steps, csv schema + splitting
job-log lines asserted; GGUF-LoRA parseable by an independent reader test;
**llama.cpp adapter load exercised for real here**: `llama-server --lora
lora.gguf` against a locally-built tiny f32 llama base GGUF loaded the adapter
and produced different greedy output with vs without it (adapter active).

**Golden-task gate: CLOSED-ON-CPU (final acceptance gap closed).** The full
criterion — "streamed backend trains a checkpoint that passes the golden task
validation gate through the normal registry promotion path" — now executes
end-to-end HERE through production code only: real streamed worker ->
`run_export` gguf-lora relocation -> `ServingSupervisor` spawns the real
llama-server for a locally-built tiny f32 llama base GGUF with the adapter
(`--lora` via `cfg.extra_llama_args`; no serving code modified) -> real
`training.validate.validate_model` golden gate -> registry
register/set_validation/**promote**. The golden is a MINIMAL DETERMINISTIC
tripwire derived from the measured adapter delta on greedy output (this tiny
adapter empties the greedy continuation; golden = `expect_not_contains` on the
base-only continuation), with a mandatory NEGATIVE CONTROL: the same golden
replayed against the base WITHOUT the adapter FAILS (`gate_without_adapter.ok
= false`, pass_rate 0.0 vs 1.0 with it), so the assertion is adapter-tied, not
vacuous. Convention + limits documented in DECISIONS.md **D-15** (acceptance
plumbing proof; never a quality claim). Evidence excerpt (verbatim run
output): `train_loss 3.2799`, probe base reply `"w41w9w23w51w36w52…"` vs
adapted `""` (changed=True), `gate_with_adapter {ok: true, pass_rate: 1.0}`,
`gate_without_adapter {ok: false, pass_rate: 0.0}`, `promotion {model_id: v1,
active: true}`. Reproduce: `python -m experiments.wp10_golden_gate.run_gate`
(prints full evidence JSON; deterministic across repeats) or
`pytest tests/test_golden_gate_wp10.py` (~5 s; skips without heavy deps or an
existing llama-server binary — never auto-builds).
**Hardware-unverified / remains for GPU acceptance**: any real-model-size run
(speed/VRAM claims — none made, CPU box; follow `bench --noise-floor` rules),
production-base `-lora` load, SEMANTIC golden validation against a served
real-model GGUF-LoRA (the gate CODE PATH is now proven here on a served
GGUF-LoRA; semantic goldens on real models remain GPU acceptance), non-pinned
archs (adapter registry rejects unmapped modules by design). Requirements
note: streamed targets LOCAL HF checkpoint dirs; may
need a transformers build with in-tree support for the pinned arch; ~1x model
size extra disk during splitting (`training.delete_original` reclaims copies
under the run dir only — user-owned checkpoints are never deleted). Packed-
expert LoRA: NOT implemented (config surface doesn't pretend it exists);
roadmap note in config.example.yaml mirrors airllm's rationale (512-expert
adapters + Adam don't fit small VRAM). Full suite 1764 green, ruff clean,
pyproject untouched.

## Improvement Program — WP11: research spike, per-user specialization (done; findings only, nothing adopted)

Final work package: an opt-in-by-nothing research spike testing whether a
downstream-eval-judged specialization pass (expert pruning with router
renormalization, and calibrated activation sparsity — sparse_transformers
methodology cited inline, never imported, D-12) using **only the Sable
experience buffer** as calibration data can beat the noise floor. Testbed: a
hand-built 4-expert tiny MoE on CPU (torch lazily imported; no downloads);
buffer seeded through the real `TraceRecorder` into a tmp `SABLE_HOME`;
downstream eval scored by `training/validate.py::_check_task` itself (same
function object — no forked gate semantics). Noise floor: no persisted bench
floor on this box (`load_noise_floor()` → None), so R=13 baseline eval
replicates at fixed config measured it offline with WP1's spread definition.

**Headline result**: NO wins. Activation sparsity was a measurable no-op
(+0.0%); expert pruning cost ~15% pass rate — clearing the 3% bar as a
**regression** in the converged regime and sitting **inside the measured 19.2%
noise floor** (reported as noise) in the limited regime. Verdict: dead-end at
this evidence level; a real-model GPU repeat through served-GGUF golden replay
is the only follow-up that could change the answer, and it is not justified by
these numbers. Nothing integrated: no config keys, worker or registry changes.
**Findings doc**: [`docs/SPECIALIZATION-WP11.md`](SPECIALIZATION-WP11.md) —
full results table, split discipline, floor numbers, limitations. Spike code
[`experiments/wp11_specialization/`](../experiments/wp11_specialization/)
(outside the package = the deliberate not-wired seam), tests
[`tests/test_wp11_specialization.py`](../tests/test_wp11_specialization.py).
Full suite green, ruff clean, pyproject untouched.

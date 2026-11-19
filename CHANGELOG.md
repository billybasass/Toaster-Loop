# Changelog

All notable changes to Toaster Loop are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Program self-update system** — `core/updater.py` + `toaster-loop upgrade
  [--check] [--yes]`: detect when a push lands on the GitHub repo and update
  the app for the user. A git-clone install diffs HEAD with origin via
  `git ls-remote` (so plain pushes are caught even without release tags);
  every install also reads the GitHub Releases API for a version number and
  notes link. Checks throttle through `<home>/update_check.json`
  (`update.interval_hours`, default 24 h) and degrade gracefully offline.
  The dashboard shows an update banner (startup background check +
  `update.available` bus event) with an Apply button running
  `git pull --ff-only` + reinstall on a background task — refusing on dirty
  worktrees/detached HEAD/non-fast-forward. Nothing ever updates silently;
  `--check` exits 10 (established informational code). New `update:` config
  section (`enabled`, `repo`, `interval_hours`) + `/api/update*` endpoints.
- **ML-stack version window** — `core/mlstack.py` declares the supported
  `torch`/`transformers`/`peft`/`trl`/`datasets`/`accelerate` range (mirroring
  pyproject `[training]`, drift-checked by a test). The worker refuses to start
  a job with an out-of-window combination loaded — actionable message in the job
  log instead of mid-run TRL API breakage — and `toaster-loop doctor`'s deps
  check reports each package's verdict. Escape hatch:
  `TOASTER_LOOP_ALLOW_UNPINNED_MLSTACK=1` downgrades the stop to a warning.
- **Real-torch CI smoke covers DPO too** — `scripts/ci_train_smoke.py` now runs
  SFT *and* DPO end-to-end through the real worker on the tiny CPU model (the
  exact paths where a TRL minor bump breaks silently), plus an honest GRPO
  dry-run (generation is skipped rather than faked). The CI `train-smoke` job
  got a generous timeout and an HF-hub model cache.
- **Windows cross-process locks** — the easy-loop lock and the scheduler's
  device-slot/dispatcher locks previously degraded to a silent no-op on NT
  (no `fcntl`), so dashboard + Task-Scheduler cron could double-train there.
  A new `core/proclock.py` gives every call site one interface over flock
  (POSIX, unchanged behavior) / `msvcrt` byte-range locking (Windows,
  kernel-dropped on process death) / a pid+liveness lockfile fallback that is
  loud about being advisory-only. `doctor` gained a `proc-locks` check.
- **gguf-lora artifacts are servable** — deploying a streamed `gguf-lora` row
  now spawns llama-server as a proper base+adapter pair (`-m <base GGUF>
  --lora <adapter>`), where the base resolves from provenance `base_gguf`,
  config `serve.lora_base_gguf`, or `TOASTER_LOOP_LORA_BASE_GGUF`. The gate
  stays strict: the adapter must parse as a GGUF LoRA adapter, architectures
  must match, and an incomplete pair still refuses (never serves unmodified
  base weights under a trained model's name).

### Fixed
- **Continual training** — a finished model can now be trained on more data.
  `--base-adapter <dir>` (worker) / `training.base_adapter` (train.yaml) loads
  an existing LoRA adapter onto its base with `is_trainable=True` instead of a
  fresh random one, for SFT/DPO/GRPO alike. Exposed as `--continue-from
  <model_id|dir>` on `toaster-loop train start`, the `continue_from` body field
  on `POST /api/train`, and the MCP `train_enqueue.continue_from` arg; the new
  model row records a real lineage edge (`parent_adapter`) and inherits the
  parent's resolved base model.

### Fixed
- **Trained adapters no longer strand in the HF hub cache** — every submit path
  (dashboard Train page, CLI `train start`, iterate) now resolves the dataset
  train.yaml through one point that pins `output_dir` under `train_runs/<run>/`,
  so finished adapters are visible to the Models panel and never depend on a
  location `huggingface-cli` cleanup deletes. (Root cause of "I had to move the
  model after it was made".)
- **Stale-checkpoint poisoning** — auto-resume is now scoped by a per-run config
  fingerprint written alongside the checkpoints; a checkpoint left by a different
  data/model/lr/rank/base_adapter in a shared output dir is refused (with a hint
  to resume explicitly) instead of silently training on foreign weights.
- **DPO/GRPO hardening** — both paths now share SFT's bar: 4-bit tiering with the
  ROCm/bnb probe, pre-quantized guard, `_tf.set_seed`, checkpoint resume and
  rotation, and mid-run NaN-loss abort.
- **Mid-run NaN abort** — a NaN loss stops the run immediately (and fails it)
  rather than burning hours of GPU before the post-hoc integrity gate.
- **Checkpoint retention** — `save_total_limit` (default 3) rotates checkpoints
  on SFT/DPO/GRPO so multi-GB saves no longer accumulate forever.
- **Worker CLI contract** — a job submitted without `train_yaml` now fails with a
  JSON error event in the job log instead of an argparse exit-2 and an empty log.
- **Cross-process id races** — model (`vN`) and job (`job-NNNN`) id allocation
  retry on PRIMARY KEY collisions, so concurrent dashboard/CLI/cron writers to one
  WAL DB no longer die with an opaque IntegrityError.
- **Unresolvable-base provenance** — `finalize_standalone_train` attests
  `needs_base_resolution` instead of silently writing a poison `unspecified-base`
  row; `toaster-loop doctor` gained a `model artifacts` check for adapters stranded
  in the hub cache, vanished artifact paths, unresolved bases, and un-servable
  `gguf-lora` exports.
- **Serving honesty** — deploying a `gguf-lora` (streamed) artifact without a
  resolvable base pair is refused with the reason, and deploys of never-validated
  models log + return an `unvalidated` flag.
- **DPO dataset schema for trl>=0.19** — the curater's mixed rows (role/content
  `prompt` + string `chosen`/`rejected`) died mid-tokenization in TRL's
  conversational concat path; `dpo_row_for_trl` now wraps completions as
  assistant messages when the prompt is conversational. Caught by the real-torch
  CI smoke.
- **transformers>=5 trainer construction** — DPO/GRPO pin `eval_strategy="no"`:
  transformers 5 defaults it to `epoch` and hard-fails a Trainer built without
  an eval dataset (both paths deliberately run no eval loop).

## [0.1.0] — 2026-11-18

First tagged release. Toaster Loop is a local, self-improving agent system:
built-in agent orchestration against locally served GGUF models, with trace
capture, curation, LoRA fine-tuning (SFT/DPO/GRPO), GGUF export and promotion
all running on one machine over loopback — zero API keys, zero cloud calls.

### Added
- **Agent runtime** — `toaster-loop run`: OpenAI-compatible chat loop with native
  tool calling (`read_file`, `list_dir`, `write_file`, `patch`, `search_files`,
  `todo_write`/`todo_read`, `search_codebase`, `search_traces`), delegation tools,
  opt-in skills, and capability-gated reach-out tools (`web_fetch`,
  `run_command`, `view_image`, `take_screenshot`) that are off by default,
  grantable per-profile, SSRF-blocked, and jailed to the workspace.
- **Serving** — `toaster-loop serve`/`deploy`/`instances`: supervised
  `llama-server` lifecycle (spawn/healthcheck/crash-loop budget/watchdog),
  multi-instance router with circuit breaker, GPU coexistence idle-unload /
  lazy-reload, engine registry incl. per-backend managed llama.cpp builds.
- **Any-GPU runtime** — `toaster-loop setup`/`update`: accelerator autodetect
  (CUDA / ROCm / Metal / Vulkan / CPU) and clone+build of llama.cpp for the
  detected backend; vendor-fork override via `--repo/--ref`.
- **Experience buffer** — every run recorded as an append-only JSONL trace +
  SQLite index with heuristic outcome scoring; `traces`, `feedback`, `insights`,
  `history` CLIs.
- **Curation** — `curate`/`datasets`: score/time/profile selection, dedup,
  ShareGPT + TRL export, optional LLM-as-judge, ingestion quality gate, hub /
  local-folder / website crawl sources, dataset diff + lint.
- **Training** — FIFO single-GPU scheduler running SFT (masked response-only
  LoRA via transformers/peft), DPO and GRPO (TRL) in subprocess workers;
  checkpoint resume, adapter-rank growth stages, streamed-LoRA backend (WP10);
  GGUF export + merge; golden-suite validation gate.
- **Closed loop** — `toaster-loop iterate` (curate → train → export → validate),
  `evolve`, shadow evaluation, promotion/rollback with the model registry,
  tiers, lineage and an archive-don't-delete janitor.
- **Dashboard** — full web UI on loopback (chat, serve, train, datasets,
  models, plans, delegation inbox, hardware ops, hub), SSE event bus, Prometheus
  `/api/metrics`; token auth ON by default with Host/Origin CSRF +
  DNS-rebinding defenses and hardened security headers.
- **MCP control plane** — `toaster-loop mcp` over stdio (no sockets),
  read/mutating tool classification with plan-then-confirm for mutations.
- **Ops** — `doctor [--env] [--fix]` measured environment report with exact
  next commands; `launch`/`reset`; exit-code taxonomy (`0/1/2/3`); atomic writes
  with Windows sharing-violation retries.
- **Guarded autonomy** — `plan` store shared by CLI/dashboard/MCP, `plan loop`
  with single-flight + stop conditions; delegation with child chat threads.

### Security posture (verified in the production-readiness audit)
- Loopback-only by default; refuses LAN exposure without a token.
- Unauthenticated `/api/*` → 401; `Host:` mismatch → 403; cross-origin POST →
  403; CSP / `X-Frame-Options` / `Referrer-Policy` present.
- Auth token file `-rw-------` (`secrets.token_urlsafe(32)`); no `shell=True`
  anywhere; capability tools default-off with private/link-local/metadata IP
  blocking and hop-by-hop redirect revalidation.

### Tooling / release engineering (production-readiness pass)
- CI executes on every push/PR: 9-OS×Python matrix, ruff, **full-package mypy
  gate** (all 134 modules clean), coverage floor (`--cov-fail-under=78`),
  `concurrency` cancellation, per-test timeout.
- CPU tiny-model **train smoke** job runs the real worker entry point against
  the locked training stack; **artifacts** job builds the wheel and installs it
  into a clean venv asserting the dashboard static ships; non-blocking
  `pip-audit` + BLE001 convention report.
- Upper bounds on `[training]` deps (transformers/trl churn between majors);
  pre-commit hooks mirror CI.
- MIT `LICENSE`; this changelog; release tags (`v<semver>`).
- Store refuses to open a database written by a newer release
  (`SchemaTooNewError`) instead of corrupting it after a rollback.
- User-facing hints use the installed command name (`toaster-loop`, not
  `toaster_loop`); generic `Server:` header; token no longer repeated in
  printed advice URLs.

### Fixed
- Unmasked SFT path imported `SFTTrainer` from `transformers`, which does not
  export it (it lives in `trl`) — the path now imports from `trl` with a
  transformers fallback for ancient installs. This would have crashed any run
  with `training.train_on_completions: false` on the current stack.
- 50 latent type errors across 21 modules (e.g. `run_args` int/str variable
  reuse, a missing retry-exhaustion return in the delegation claim loop).

### Known limitations
- The DNS-rebinding TOCTOU window on capability-gated egress is documented
  rather than closed (inherent to resolve-then-connect without a proxy).
- Streamed-LoRA (WP10) and specialization (WP11) remain gated spikes, not
  default paths.

[Unreleased]: https://github.com/billybasass/Toaster-Loop/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/billybasass/Toaster-Loop/releases/tag/v0.1.0

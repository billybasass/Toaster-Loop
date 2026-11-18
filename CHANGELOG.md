# Changelog

All notable changes to Toaster Loop are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

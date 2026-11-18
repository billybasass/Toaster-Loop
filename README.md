# Toaster Loop

Local, self-improving agent system: built-in agent orchestration against
locally served models, with LoRA fine-tuning (SFT/DPO/GRPO) and GGUF export
built in. Everything runs on one machine over loopback; **zero API keys, zero
cloud calls**.

## Quick start

```bash
git clone https://github.com/billybasass/Toaster-Loop.git
cd Toaster-Loop
./setup.sh          # venv + deps + builds llama.cpp for YOUR GPU
./launch.sh         # dashboard + browser (desktop icon added too)
```

That is the whole happy path: `setup.sh` detects your accelerator (NVIDIA /
AMD ROCm / Apple Metal / Vulkan / CPU), installs the matching torch build, and
compiles llama.cpp into `<home>/llama.cpp`. `launch.sh` starts the dashboard
and opens it. To check anything at any time:

```bash
toaster-loop doctor --env     # measured environment report + exact next commands
```

Windows: double-click `setup.cmd` / `launch.cmd`, or run the `.ps1` ports from
PowerShell. Every script command also has a pure-CLI form (`python -m
toaster_loop <cmd>`) that works on any OS. See [Install](#install) for options
(`--no-training`, manual venv, custom llama.cpp fork).

### Upgrade

```bash
git pull
./setup.sh                    # re-syncs the venv from uv.lock + rebuilds if needed
toaster-loop update --check   # optional: check for a newer managed llama.cpp
```

State (buffer, datasets, models) lives under `~/.toaster-loop/` and survives
upgrades. The store refuses to open a database written by a *newer* release —
if you roll back binaries, restore an older backup too (see below).

### Backup

Everything persistent is one directory: `~/.toaster-loop/` (override with
`TOASTER_LOOP_HOME`). Back it up while the app is stopped for a consistent
snapshot; SQLite runs in WAL mode so readers are safe but writers should quiesce.

```bash
tar czf toaster-loop-backup-$(date +%F).tgz -C ~ .toaster-loop
```

### Uninstall

```bash
toaster-loop launch --stop    # stop anything the launcher started
rm -rf ~/.toaster-loop        # ALL state: buffer, datasets, models, logs, llama.cpp build
```

Toaster Loop never writes outside `~/.toaster-loop/` (plus its own venv/repo) —
removing those leaves nothing behind. `toaster-loop reset --yes [--keep-models]`
does the same wipe from inside the app.

### Support matrix

| Platform | Status |
|---|---|
| Linux (x86_64, aarch64) | CI-tested on Python 3.11/3.12/3.13 |
| macOS (Apple Silicon) | CI-tested; Metal backend |
| Windows 10/11 | CI-tested (git-bash); `.cmd`/`.ps1` launchers |

GPU backends exercised by setup: CUDA, ROCm, Metal, Vulkan, CPU-only. Requires
Python ≥3.11,<3.14 and ~10 GiB free disk for the largest model exports.

---

## What it does (capability tour)

The sections below are organized by the capability each command serves; the
full operator reference continues under [Usage](#usage).

### Serve & run agents

Phase 1 delivered:

- `toaster-loop serve` — Serving Supervisor v0: spawn a supervised `llama-server`
  (GGUF, loopback, dynamic port; built for your GPU by `toaster-loop setup`) or
  optionally attach to any OpenAI-compatible server you point at with
  `--attach`. toaster-loop is fully standalone — it never requires or auto-connects
  to any other local program.
- `toaster-loop doctor [--env]` / `toaster-loop setup` — any-GPU runtime: detect the accelerator
  (NVIDIA CUDA / AMD ROCm / Apple Metal / Vulkan / CPU-only), report which
  llama-server would be used and how it launches (`-ngl` offload auto-sized to
  the hardware), and — when no llama.cpp exists anywhere — clone + build one
  automatically for the detected backend into `<TOASTER_LOOP_HOME>/llama.cpp`
  (opt out with `llama_auto_setup: false` / `TOASTER_LOOP_NO_AUTO_SETUP=1`).
  `--repo/--ref/--backend` (or config `llama_repo`/`llama_ref`/`llama_backend`)
  choose the source — e.g. a vendor fork like
  `Lychee-Technology/llama-cpp-for-strix-halo` — and `toaster-loop update [--check]`
  is the llama.cpp update system (remote-commit diff + rebuild in place).
  `doctor --env` (WP7) MEASURES the environment instead of assuming: free disk
  on the filesystems export writes to, MemAvailable → serve/train presets with
  printed formulas, nvidia-smi VRAM tiers when present, config parse, ports,
  stale instance state (`--fix` removes only verified-dead entries), and prints
  THE exact next command per finding ([`toaster-loop/doctor.py`](toaster_loop/doctor.py)).
- `toaster-loop launch [--stop]` — one-click program launcher: detached dashboard +
  browser, idempotent, stops only what it launched.
- `toaster-loop reset --yes [--keep-models]` — cleanup back to a fresh state: wipes
  TOASTER_LOOP_HOME (buffer, datasets, registry, logs, managed llama.cpp); dry-run
  without `--yes`; never touches foreign installs or system paths.
- `toaster-loop run` — boot the built-in agent (OpenAI-compatible chat loop with
  native tool calling: read_file/list_dir/write_file/patch/search_files) and
  run a one-shot task or interactive REPL.

Phase 2 adds the **experience buffer**: every agent execution is recorded as a
durable trace (append-only JSONL + SQLite index) with an automatic heuristic
outcome score, plus CLI for feedback and inspection:

- `toaster-loop traces list|show|stats` — browse the buffer.
- `toaster-loop feedback <trace_id>|--last [--score -1..1|--fail|--ok] [--note ...]`
  — attach explicit feedback; syncs the trace's outcome score for curation.

Phase 3 adds the **dataset curater**: select high-value traces (score/time/
profile filters), dedup prefix-overlapping REPL conversations, and export
training-ready datasets (ShareGPT + TRL formats):

- `toaster-loop curate [--name N] [--min-score F] [--since DATE] [--profile P]
  [--include-failures] [--limit N] [--format sft|dpo|both] [--rebuild ID]
  [--judge N]` — create a dataset under `<home>/datasets/<id>/`; `--judge N`
  opts in to LLM-as-judge scoring of borderline traces (biased tiebreaker —
  see [Dataset curation](#dataset-curation-phase-3)).
- `toaster-loop datasets list|show <id>|export <id> --to <dir>|import <file>`.

Phase 4 closes the **train → deploy → iterate loop**: a model registry with
lineage + active pointer, a single-GPU FIFO training scheduler that runs SFT
(transformers `SFTTrainer` + peft LoRA) and DPO (TRL) in subprocess workers,
a GGUF export step, a golden-task validation gate gating promotion, promote/rollback, and
hot-swap of the running server:

- `toaster-loop iterate [--name run1] [--kind sft|dpo|grpo] [--dry-run] [--config-key k=v]`
  — one call: curate → train → export → validate → promote-or-keep-previous.
  `grpo` is reinforcement fine-tuning (TRL GRPOTrainer) with built-in rule
  rewards (format / length / reference-overlap; pick a subset via
  `training.grpo_rewards`). Training runs can be parallelized across GPUs
  (`training.max_parallel_jobs` + `gpu_ids`; each worker is pinned to one
  device), and export supports GGUF (default) or AWQ/GPTQ HF checkpoints
  (`training.export_format: awq|gptq`, optional autoawq/auto-gptq packages).
- `toaster-loop train start --dataset <id> [--kind sft|dpo] [--dry-run] [--config-key k=v]`,
  `toaster-loop train status [job_id] [--tail N]`, `toaster-loop train cancel <job_id>`,
  `toaster-loop train auto-check [--trigger]` (≥N unclaimed traces with mean score
  ≥ threshold; `--trigger` runs the full iterate loop when the check **and**
  the resource gate pass — the cron entrypoint for autonomous retraining).
- `toaster-loop models list|show <id>|promote <id>|rollback` — plus the
  archive-don't-delete lifecycle: `models archive <id>` / `models restore <id>`
  (recoverable; artifacts move under `<home>/archive/<id>/` and back),
  `models pin <id>` / `models unpin <id>` (pinned models bypass automation) and
  `models janitor [--days N] [--dry-run]` (archives aged terminal models —
  see the model lifecycle section below).

Repeatable `--config-key dotted.path=value` (backlog #13) overrides any
resolved-train.yaml knob per run without editing YAML — e.g.
`--config-key training.lora_r=32 --config-key lora.use_dora=true`. Values are
typed (`true`/`false`, int, float, JSON list/object, else string), applied to
the resolved `train.yaml` copy after the pipeline's own injections (so explicit
flags win), and recorded in the job log for provenance. See
`config.example.yaml` for every knob the worker honors today and its default.
- `toaster-loop deploy <model_id>` — stop the toaster-loop-owned llama-server, start one
  on the model's GGUF, re-point live runners via `switch_model`.
- `toaster-loop serve --auto` — serve the ACTIVE promoted GGUF from the registry.

Phase 5 adds the **dashboard**: a loopback-only web UI over the whole loop,
served by toaster-loop itself (stdlib HTTP server + vanilla JS — no new
dependencies, no external CDN assets, works fully offline):

- `toaster-loop dashboard [--port N]` — serve the control UI on
  `http://127.0.0.1:<port>/` (dynamic port by default; Ctrl-C stops cleanly).

### Dashboard tour (current panel set)

The rail groups fifteen panels into five sections. Everything updates over a
single SSE channel (`GET /api/events`) with an online/offline pill in the
header; when the pill is grey, reconnecting EventSource retries automatically.

- **Loop**
  - **Overview** — status cards (server state + active model, buffer health
    with a link, job counts) and loop controls that deep-link into their own
    panels (Train…, Iterate…, Schedule…); Curate runs inline as one queued
    job. First-run empty states point at Chat ("No traces yet — start
    chatting to feed the loop"). Shows the latest iterate run.
  - **Chat** — threads persisted per conversation; **Agent mode** records a
    trace into the buffer (built-in runner) and carries a per-run
    **profile selector**; Playground
    mode hits the served model raw and records nothing. 👍/👎 feedback per
    assistant message. Threads live in a **project tree**: a project carries a
    name, a `workspace_dir` (the agents' containment root for its runs), a
    free-text **description** and one **job line per attached agent** — every
    agent run on that project (chat, delegated children, `toaster-loop run --project`)
    gets a `## Project` section with the description and the full roster, each
    agent's own line marked "that's you". Project nodes expose **edit** (✎) and
    **delete** (✕); deleting removes the record + its codebase index and moves
    its threads to Unfiled, and NEVER touches the workspace directory or the
    files the agents wrote.
- **Serving**
  - **Serve** — start/stop the toaster-loop-owned llama-server (model picker from
    local GGUFs), attach to an external OpenAI endpoint, live state via
    `serve_state.json` + pid liveness. Phase 6 adds a **server instances**
    table: N named servers (spawn a GGUF or attach a loopback endpoint) run
    concurrently with per-row start/stop, role tags
    (`orchestrator|coding|embedding|custom`) and an autostart flag; spawns
    consult free VRAM and warn (never block) on over-commit. Definitions live
    under `<TOASTER_LOOP_HOME>/serving/instances/<id>.json`; the reserved
    `default` instance mirrors the legacy single-server setup. On multi-GPU
    boxes an instance may pin itself to device(s) with `"gpu_id": 1` or
    `[0, 1]` (spawn then carries `CUDA/HIP_VISIBLE_DEVICES`, Vulkan gets
    `--device` flags, and the over-commit check reads that device only);
    instances without a pin behave exactly as before.
  - **Models** — registry table with lineage, validation results, derived
    capability tiers (T0 experimental … T3 field-proven), and **promote** /
    **rollback** buttons (confirmation-gated).
  - **Model Hub** — search Hugging Face, browse GGUF quant variants and
    download models into the models folder in the background (mirrors the
    `toaster-loop hub` CLI; gated repos need `HF_TOKEN`).
- **Data**
  - **Traces** — list/filter, detail drawer with transcript + feedback,
    inline 👍/👎 (same store the CLI writes), sortable columns + pager.
  - **Datasets** — curated manifests (sft/dpo counts, base model), export
    (contained to `TOASTER_LOOP_HOME` unless forced).
- **Training**
  - **Easy** — one-profile turnkey loop (`easy_training:` config): pick an
    agent profile bound to a managed instance and one click runs the full
    iterate pipeline with a documented optimal preset, optionally hot-deploying
    the promoted model back onto that profile's server. See
    [`toaster-loop/training/easy.py`](toaster_loop/training/easy.py).
  - **Train** — readiness gauges + dataset picker + dry-run; single-flight
    (`409 already_running` links to the live job).
  - **Iterate** — full loop (curate → train → export → validate → promote)
    with a live step timeline from `iterate.step` events and run history.
  - **Schedule** — auto-iterate threshold rule (mirrors `training.auto_*`),
    a live readiness card computed from `/api/train/check` (gate reasons are
    the source of truth), trigger-now, and the exact cron line for autonomous
    runs (copy-to-clipboard). See *Autonomous retraining* below.
  - **Jobs** — queue state over SSE, worker log tails (`#/jobs/<id>` deep
    links), cancel.
- **Setup**
  - **Profiles** — agent profiles: named runner configurations (system prompt,
    tool subset, optional model alias) stored in the config file; set-default
    and delete with confirm. Chat's agent mode applies the selected profile.
    Phase 6 adds an optional **server instance** binding per profile: chat
    runs for that profile target the chosen managed instance (e.g. an
    orchestrator profile on one model while a coding profile runs another,
    concurrently); unset keeps the default serve endpoint.
  - **Hardware** — CPU/RAM/GPU/disk/power cards with sparklines + the
    scheduler resource gate.
  - **Settings** — structured per-section patches + raw YAML editor; secrets
    come back `"***"` and blanks are never patched back. Also hosts the
    Skills management card (view/edit/delete over `/api/skills`, see
    [`docs/SKILLS.md`](docs/SKILLS.md)) and the codebase-index card
    (build/search/stop over `/api/index*`).

The UI is read-mostly: every mutation goes through the same core API objects
the CLI uses (`ModelRegistry`, `Scheduler`, `Curater`, `TraceStore`) behind a
JSON HTTP API under `/api/*`, each with a browser confirmation — it can never
corrupt state beyond what the CLI could. Bind address is hardcoded to
`127.0.0.1`; the config/env cannot widen it. Long-running actions (train,
iterate) execute on background threads; the UI polls `/api/jobs` for progress.

### Phase 5d API contract notes

- **`POST /api/curate` is a queued job** (contract change in 5d): returns
  `202 {ok, job_id, state:"queued"}` instead of running synchronously (large
  buffers used to block the request thread). The job has kind `curate`,
  emits normal `job.state` bus events, and on success its log ends with a
  line `{"event":"done","job":<id>,"result":{dataset_id,sft_count,dpo_count,
  claimed,warnings}}`. Invalid filter bodies still fail fast with `400`.
- **Dataset export containment guard**: `POST /api/datasets/export` refuses
  destinations outside `TOASTER_LOOP_HOME` with `409 {code:"outside_home"}` unless
  the body also contains `"force": true`. The Datasets panel surfaces a
  confirm dialog in that case before re-sending with force.
- **Per-route body caps**: generic `/api/*` request bodies cap at 1 MB
  (`413 payload_too_large`), but `POST /api/chat/stream` accepts up to
  ~43.5 MB so image attachments (up to 4 × 8 MB raw, base64-inflated — the
  per-image cap in `core/images.py`) actually reach the handler; the chat UI
  pre-checks file size client-side and names the offending file.
- **Train single-flight**: `POST /api/train` returns
  `409 {code:"already_running"}` while another job is running (message names
  the job id), so the UI can link straight to it instead of queueing blindly.
- `GET /api/traces` now includes `total`/`offset`/`limit` alongside the window
  (filtered count across all pages) for pager rendering.
- `PUT /api/config` accepts `{section:"root", patch:{...}}` to patch
  top-level keys (`gguf_path`, `port`, …); secret fields come back redacted
  as `"***"` from `GET /api/config` and must never be PUT back verbatim —
  the Settings panel omits unchanged secret inputs.
- Panels shipped in 5d: **Train** wizard (readiness gauges + dataset picker +
  dry-run), **Iterate** (live step timeline from `iterate.step` events),
  **Hardware** (CPU/RAM/GPU/disk/power cards with sparklines + scheduler
  gate), **Settings** (structured section patches + raw YAML editor).

### Phase 5e API contract notes (profiles, schedule, config unset)

- **Agent profiles** live in the config file under a `profiles:` list plus a
  `default_profile` key. REST surface:
  - `GET /api/profiles` → `{profiles: [{name, system_prompt, tools,
    model_name}], default_profile, known_tools}` (`known_tools` = built-in
    tool names for the editor's checkboxes).
  - `PUT /api/profiles/<name>` upserts (body `{system_prompt?, tools?,
    model_name?}`; name must match `[A-Za-z0-9_-]{1,40}`).
  - `DELETE /api/profiles/<name>` (deleting the default clears it too).
  - `PUT /api/profiles/default` with `{default: <name>|null}`.
- **`POST /api/chat/stream` accepts `profile: <name>`** (agent mode). The
  profile's `system_prompt` replaces the runner system prompt, `tools`
  restricts the tool set (the runner exposes only that subset), and `model_name`
  overrides the served-model alias when attached. Unknown profile → `404
  {code:"unknown_profile"}`. Traces record the profile name as
  `agent_profile`.
- **Config unset**: `PUT /api/config` section patches now treat a `null`
  patch value as *delete this key* (previously merge-only could not unset).
  The Settings panel uses this for its "clear" affordances.
- **`GET /api/train/check`** is the schedule readiness source of truth:
  `{enough_traces, unclaimed, mean_score, threshold:{min_traces,min_score},
  gate:{ok, reasons}}`. The Schedule panel renders it directly and never
  re-reads thresholds client-side.

### Phase 6 API contract notes (multi-instance serving)

- **Instances**: `GET/POST /api/servers`, `PUT/DELETE /api/servers/<id>`,
  `POST /api/servers/<id>/start|stop`, `GET /api/servers/<id>/status`. An
  instance is either a GGUF spawn (`gguf`) or an attach target
  (`attach_url`, **loopback-only** — enforced, matching the project's
  loopback constraint); never both. Definitions persist as
  `<TOASTER_LOOP_HOME>/serving/instances/<id>.json` (atomic writes), runtime state
  as `<TOASTER_LOOP_HOME>/serving/state/<id>.json`. Duplicate names → `409
  duplicate_name`; ports claimed by other managed servers (including the
  indexer's embed server) → `409 port_conflict`.
- **Reserved `default`**: synthesized from the legacy config keys
  (`gguf_path`/`attach_url`/`port` — these are now just *the default
  instance's source*; named instances are the primary model). Its runtime
  state lives at `serving/state/default.json`, with the legacy
  `serve_state.json` folded in by the single compatibility seam in
  `serving/instances.py` (`read_state_merged`/`write_state_merged`/
  `clear_state_merged`); it cannot be deleted. `/api/serve/start` and
  `/api/serve/attach` were removed — use `POST /api/servers/default/start`
  (spawn) or a named attach instance; `/api/serve/status|stop` remain as the
  legacy status-card path.
- **VRAM advisory**: `POST /api/servers/<id>/start` returns a `warning`
  field when the GGUF estimate exceeds currently free VRAM (accounting for
  other running servers). Warning only — the spawn proceeds; re-send with
  `{force: true}` to skip the estimate. No GPU telemetry → no warning.
- **Ownership**: toaster-loop only ever terminates llama-server processes it
  spawned itself (in-process registry + per-instance state verified via pid
  ownership); attach instances and foreign processes are never killed, on
  stop *and* on shutdown (`stop_all_servers` also covers the Phase 5 embed
  servers).
- **Profile binding**: profiles accept `server_instance: <id>`; chat runs
  for that profile then target that instance's endpoint (plus its per-instance
  `sampling:` override merged over the global section). Unknown id → `404
  unknown_server`; stopped instance → `503 not_serving`.

### Autonomous retraining (cron)

The dashboard deliberately contains **no scheduler thread** — firing training
from a web process is fragile and surprising. Instead the threshold rule
(`training.auto_enabled`, `auto_min_traces`, `auto_min_score`) is consumed by
`toaster-loop train auto-check`:

```bash
# crontab -e  — check nightly at 03:15; runs iterate only when thresholds
# AND the resource gate (free VRAM, disk) pass; exits 0 silently otherwise.
15 3 * * * cd ~ && toaster-loop train auto-check --trigger >> $HOME/.toaster-loop/auto_iterate.log 2>&1
```

`--trigger` runs the full loop in-process (curate → train → export → validate
→ promote, same as `toaster-loop iterate`) and honors `--dry-run`. The Schedule
panel shows this exact line with copy-to-clipboard plus a trigger-now button.

### Self-improvement toolset (passes 3–4)

Commands that grew on top of the closed loop:

- `toaster-loop hub search|variants|download|list|delete` — Hugging Face model hub:
  text search, GGUF quant-variant browsing and resumable downloads into the
  models folder (HF hub-cache layout; gated repos need `HF_TOKEN`). Mirrored
  by the dashboard Model Hub panel.
- `toaster-loop history [--limit N] [--verify] [--intents]` — append-only,
  SHA-256 hash-chained operation ledger over registry mutations (promotes,
  rollbacks, archives, pins, job transitions). `--intents` pairs recorded
  promotion/rollback intents (`promote --intent`, `iterate --intent`) with
  observed score deltas.
- `toaster-loop evaluate shadow <model_id> [--profile P] [--n 8]` — replay a
  candidate against the active model on recent high-feedback traces,
  judge-scored; warn-only regression flag stored on the registry.
  `evaluate transfer <model_id> --profile P` probes a candidate adapter
  against *another* profile's replay set before pointing that profile at it.
- `toaster-loop evolve <profile> [--goldens FILE] [--k 3 --generations 2]` —
  GEPA-style prompt evolution on golden tasks: deterministic golden checks
  first, judge scoring for passers. Output is **always a gated diff
  proposal**; `toaster-loop evolve apply|dismiss <job_id>` is the explicit
  follow-up (`evolution:` config section).
- `toaster-loop rft [--seeds ...] [--goldens FILE] [--k 4]` — rejection-sampling
  fine-tuning data (STaR): sample K completions per prompt from the served
  model, keep only deterministically-verified ones, emit a ShareGPT SFT
  dataset (`rft.enabled: true`; job kind `rft`).
- `toaster-loop mine-goldens [--max 8] [--use-judge]` — distill 👍-thumbed traces
  into golden validation tasks with self-checked deterministic verifiers;
  mined goldens join `validation.golden_tasks` and RFT selection
  (`goldens_mine.enabled: true`).
- `toaster-loop playbooks scan|candidates|list|promote|assess|retire|show|export` —
  trajectory → playbook distillation: recurring successful tool chains become
  candidate playbooks, promotion is gated (optionally by a judge replay),
  `assess` correlationally checks for regressions and auto-retires
  (`playbooks:` config section).
- Extras on existing commands: `datasets diff A B` (version-to-version trace/
  hash diff) and `datasets lint [id]` (re-run the ingestion quality gate);
  `train start --depends-on <job_id>` (repeatable job dependencies) and
  `train resume <job_id>` (re-queue from the newest checkpoint);
  `run --project <id>` (project workspace containment + index routing, plus the
  project description + agent roster as a `## Project` prompt section).

```
        ┌─────────────────────────── the loop (Phase 4) ───────────────────────────┐
 run tasks ─▶ buffer ─▶ curate ─▶ train (worker, LoRA/TRL) ─▶ export GGUF ─▶ validate
   ▲                                                                        │ pass
   │                                                                        ▼
   └───── serve --auto / deploy ◀── active pointer ◀── promote ◀────────────┘
                     (fail → model marked failed; previous stays live;
                      `toaster-loop models rollback` reverts one step anytime)
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md),
[`docs/ROADMAP.md`](docs/ROADMAP.md) (per-phase acceptance criteria), and
[`docs/DECISIONS.md`](docs/DECISIONS.md). AI agents: operate via
[`docs/AI_SETUP.md`](docs/AI_SETUP.md); repo rules in
[`AGENTS.md`](AGENTS.md).

## Install

One-click from a fresh clone — creates the venv, installs everything
(including the GPU training stack), builds llama.cpp for your GPU, and adds a
desktop launcher icon:

```bash
cd Toaster-Loop
./setup.sh            # add --no-training to skip the torch/transformers/trl extras
./launch.sh           # start front+back end + browser (or click the icon)
./cleanup.sh          # back to a freshly-downloaded state
```

Windows: the `.ps1` ports do the same (double-click the `.cmd` wrappers, or
run them from PowerShell). `bitsandbytes` is Linux-only, so pass
`--no-training` on Windows unless you have a supported CUDA toolchain;
equivalently, every script command has a pure-CLI form that works on any OS:

```powershell
cd Toaster-Loop
.\setup.ps1 --no-training      # or: python -m toaster_loop launch (no scripts at all)
.\launch.ps1                   # start front+back end + browser
.\cleanup.ps1                  # back to a freshly-downloaded state
```

`launch.sh --stop` stops only what the launcher started; `cleanup.sh
--keep-models` keeps trained models. Manual install (equivalent steps):

```bash
cd Toaster-Loop
uv venv --python 3.13 .venv          # Python >=3.11,<3.14
source .venv/bin/activate
uv pip install -e ".[dev]"
```

The agent runtime is built in — chat loop with native tool calling plus the
toolset `read_file`, `list_dir`, `write_file`, `patch`, `search_files`,
`todo_write`, `todo_read`, `search_codebase`, `search_traces`, the delegation
tools (`list_agents`, `delegate_task`, `check_delegation`) and — when
`skills.enabled` — the skills tools; plus capability-gated reach-out tools
(`web_fetch`, `run_command`, `view_image`, `take_screenshot`), all off by
default and grantable per-profile via `capabilities:` (see
[`config.example.yaml`](config.example.yaml) and
[`docs/BLENDER-AGENT.md`](docs/BLENDER-AGENT.md)); text tool-call recovery for GGUFs
with flaky function calling; and automatic context compression for long
threads (`agent:` config section). Oversized tool results are spilled to
`<TOASTER_LOOP_HOME>/cache/spillover/` behind a small `<persisted-output>` stub
the agent can re-read, keeping traces lean (`tools.spill_*` config keys).
No external agent framework is required.
For GPU training jobs, install the optional training extra in a GPU-enabled
interpreter:

```bash
uv pip install -e ".[training]"   # torch/transformers/peft/trl/datasets/bnb
```

## Usage

### Serve a GGUF with a supervised llama-server

```bash
toaster-loop serve --gguf ~/models/qwen3-8b-q4_k_m.gguf
# -> Serving ...
#    OpenAI-compatible base_url: http://127.0.0.1:41237/v1
#    model: qwen3-8b-q4_k_m
#    Press Ctrl-C to stop.
```

Context defaults to `-c 65536`; override
with `--context`. Ports are dynamic by default (`--port 0`) so a server
already running on :8888 is never disturbed. Ctrl-C (SIGINT/SIGTERM)
terminates the child llama-server cleanly.

### Run an agent

One-shot against a spawned server:

```bash
toaster-loop run --gguf ~/models/qwen3-8b-q4_k_m.gguf \
  --task "list files in . and summarize what this project is"
```

Against an already-running OpenAI-compatible server (optional attach mode —
always explicit; toaster-loop never scans for or depends on external servers):

```bash
toaster-loop run --attach http://127.0.0.1:8888 --task "list files in ~ and summarize"
```

If the attach target requires a bearer token (an external server may;
supervisor-owned llama-server never does), pass it explicitly — toaster-loop will
never kill or reconfigure an external server on `:8888`:

```bash
toaster-loop run --attach http://127.0.0.1:8888 --attach-api-key "$SOME_TOKEN" --task "hi"
# or: export TOASTER_LOOP_ATTACH_API_KEY=...
```

Interactive REPL (omit `--task`).

Batch trace generation with content-based resume — bootstraps the buffer for
a new profile/base-model without dozens of manual invocations:

```bash
# tasks.jsonl — one JSON value per line: {"task": "..."} or a bare string;
# blank lines tolerated, extra per-line keys ignored (v0)
toaster-loop run --attach http://127.0.0.1:8888 --batch tasks.jsonl [--run-name boot]
toaster-loop run --attach http://127.0.0.1:8888 --batch tasks.jsonl --run-name boot --resume
```

Tasks run **sequentially** (single-GPU FIFO) through the same runner and
recorder as one-shot runs — every output is an ordinary trace and curation
picks them up unchanged. Progress prints one line per task
(`batch 3/17: <task> ... ok|failed(<reason>)|skipped(resume)`) plus a final
ran/skipped/retried-infra/failed summary. Exit code is 0 when every task has
a terminal trace (model failures included), 1 only for remaining infra
failures or input errors.

`--resume` decides "completed" by **content**, not line index: it scans the
trace buffer for a recorded task matching the prompt text (after the same
secret-redaction the recorder applies) whose `failure_reason` is not
infra-ish. Infra faults (`connection`/`timeout`/`model_not_found`) auto-retry;
model faults (`max_iterations`, `guardrail_halt`, …) count as done — they are
training signal, not transport noise. Input reordering/duplication is safe.
The checkpoint at `<home>/batches/<run-name>.json` (default name: the input
file stem) is stats only — atomic per-task writes for `started_at` bookkeeping
and reporting; when it disagrees with the buffer, the traces win.

### Experience buffer (traces & feedback)

Every `toaster-loop run` execution (one-shot or REPL turn) appends a trace record —
full conversation incl. tool calls/results, model alias, endpoint label, token
counts, duration, heuristic outcome score — and prints its id:

```bash
toaster-loop traces list --limit 10 [--status success|failure|partial] [--since 2026-09-01]
toaster-loop traces show <trace_id>       # index row + feedback + pretty-printed JSONL
toaster-loop traces stats                 # counts by status, avg score, token totals

toaster-loop feedback 3fa8c2 --ok --note "summarized correctly"   # id prefix ok
toaster-loop feedback --last --score -0.5                          # most recent trace
toaster-loop feedback --last --fail                                # shorthand for -1
```

Recording is **on by default** and best-effort: a trace write failure can never
break a conversation (tested). Disable via `recording.enabled: false` in config
or `TOASTER_LOOP_RECORDING=0`.

Heuristic outcome score v0 (documented in `toaster-loop/store/recorder.py`):
`+0.5` completed / `-0.5` not completed, `-0.25` per error-looking tool result
(capped at `-0.75`), clamped to `[-1, 1]`. Explicit user feedback overwrites
the trace's `outcome_score`; the heuristic is also kept as a `feedback` row
with `source="heuristic"` for provenance.

### Insights (buffer analytics)

`toaster-loop insights` reads the same SQLite index + JSONL buffer and answers
"what should I curate next / what did training break?" — pure SQL/JSONL, no
new deps, no cost estimation (loopback-only):

```bash
toaster-loop insights [--days 30] [--profile toaster-loop] [--json]
# or: GET /api/insights?days=30&profile=...  -> Overview panel card
```

Sections: per-day **score trend** overall / by profile / by model (trend =
labelled first-half vs second-half mean delta), **per-tool failure rates**
(derived lazily from transcripts with a bounded scan — older/pruned traces
beyond the cap are reported via `traces_scanned`), **failure-reason
histogram** split infra vs model, **duration/token p50/p90/max**, **activity**
(traces/day, busiest hour, per-profile table) and **before/after-promotion
deltas** — mean score of traces served by each promoted model version vs its
predecessor in a symmetric window around the promotion (needs the registry's
`promoted_at` column; older histories report the section as degraded with a
reason). Empty DB returns a well-formed zero structure.

### Curation & validation analytics (WP9)

Every `toaster-loop curate` run appends a funnel snapshot to
`~/.toaster-loop/analytics/curation_funnels.jsonl` (append-only JSONL, one record
per run): traces scanned → eligible → curated with dedup/quality/limit reject
counts per stage and a 10-bin score histogram whose `scored`/`unscored`
denominators travel with the buckets. The training pipeline appends the same
way for each validation-gate run (`validation_runs.jsonl`: pass/fail/skip
counts, pass rate, gate verdict). History is cumulative; nothing rewrites old
records.

```bash
# API surface (dashboard loopback server, token-auth like every /api/* route):
curl -H "X-Toaster Loop-Token: $TOKEN" \
  "http://127.0.0.1:PORT/api/analytics/curation?dataset=<id>&limit=50"
curl -H "X-Toaster Loop-Token: $TOKEN" \
  "http://127.0.0.1:PORT/api/analytics/validation?model=v3"
```

`/api/analytics/curation` also returns `latest_diff` (the two newest datasets
in the filtered history compared with the same logic as
`toaster-loop datasets diff`); `/api/analytics/validation` merges gate runs
chronologically with promote/rollback events from the registry ledger. The
dashboard surfaces both: a funnels card on the Datasets panel and a gate-trend
block in the model drawer.

### Prometheus metrics endpoint

The dashboard's loopback server exposes standard Prometheus text exposition at
two equivalent paths — `GET /metrics` (scrapers' default) and `GET
/api/metrics` — same TTL-cached body (5 s), same token rules (`X-Toaster Loop-Token`
header or `?token=` on GET; with a dashboard token configured an unauthenticated
scrape gets 401, never a backdoor):

```yaml
# prometheus.yml (loopback scrape)
scrape_configs:
  - job_name: toaster-loop
    static_configs: [{ targets: ["127.0.0.1:PORT"] }]
    authorization: { credentials: "<dashboard token>" }
```

Families include the WP2 honest variants (`toaster_loop_score_coverage_ratio` beside
its `toaster_loop_traces_scored_total` denominator, `toaster_loop_promotion_pass_rate`
beside `toaster_loop_promotion_decisions_total` — no rate ever renders without its
denominator). vLLM-compatible names appear ONLY where semantics match:
`vllm:prompt_tokens_total` / `vllm:generation_tokens_total` mirror the toaster-loop
token counters; there is deliberately no `vllm:num_requests_total` (a trace is
one agent run, not one engine completion), no
`vllm:time_to_first_token_seconds` (TTFT is never measured — documented
absence beats fabricated buckets) and no `vllm:num_requests_waiting` (the job
queue is a different queue; see `toaster_loop_jobs_queued`). Mapping table and
caching design: [`docs/ROADMAP.md`](docs/ROADMAP.md) WP9 entry.

### Bench (throughput + noise floor)

`toaster-loop bench` measures prompt-processing and generation tokens/sec against a
supervised llama-server endpoint (spawned via the existing serve machinery or an
`--attach` loopback target — it never touches processes toaster-loop doesn't own):

```bash
toaster-loop bench [--runs 5] [--max-tokens 64] [--label exp-a] \
              [--prompts-file prompts.txt] [--noise-floor] [--attach URL]
toaster-loop bench --list                       # persisted records
toaster-loop bench --compare base cand          # speedup + token-identity verdict
```

N repeated temperature-0 runs over a fixed deterministic prompt set; the report
shows per-run values plus median / min / max and run-to-run spread
(`(max-min)/median`). `--noise-floor` persists a baseline record that later tool
calibration reads as the threshold for real effects — improvements smaller than
the measured spread are noise, not wins. Every record is written as JSON to
`~/.toaster-loop/bench/` (override with `TOASTER_LOOP_BENCH_DIR`). Token-identity rule: if
the generated token streams differ between the records being compared, the
speedup is reported **INVALID** — a speedup that changes tokens is not a speedup.

### Calibrate (per-machine autotune)

`toaster-loop calibrate` sweeps llama-server knobs already mapped in `run_settings`
(`kv_cache_type`, `flash_attn`, `gpu_memory_pct` → scaled `-ngl`,
`batch_size`/`ubatch_size`, `parallel` slots, `speculative_type`) against THIS
machine + model and stores an adopted profile:

```bash
toaster-loop calibrate [--quick] [--knobs kv_cache_type,flash_attn] [--runs 3] \
                  [--model path/or/name]          # spawn-mode sweep
toaster-loop calibrate --list                          # stored profiles
toaster-loop calibrate --show [--model REF]            # active/stored profile
toaster-loop calibrate --clear                         # delete stored profiles
```

The sweep is coordinate-wise (baseline → each candidate vs the current best →
keep winner → next knob); every measurement uses the `toaster-loop bench` harness
(fixed prompt set, temperature 0). **Adoption rule:** a candidate wins only if
it beats the current best by MORE THAN `max(3%, measured noise floor)` AND
generates identical tokens (token-identity rule — a faster setting that changes
tokens is recorded `invalid`, never adopted). The noise floor comes from the
newest persisted bench `--noise-floor` record, or is measured + persisted on
first use. Candidates whose server fails to start are recorded `error` and
skipped; the sweep continues. When `mtp` is requested but no drafter GGUF sits
next to the model it degrades to ngram and the profile records
`tested_as: "ngram (mtp unavailable)"` — never a claim that MTP was measured.

Profiles save as JSON under `~/.toaster-loop/calibrate/` keyed by machine-id +
model hash (override with `TOASTER_LOOP_CALIBRATE_DIR`). **Serving honors them only
with an explicit opt-in** — add to your config:

```yaml
serve:
  calibration: auto   # default off: launch path unchanged
```

With `auto` and a matching profile, the adopted knob values become defaults for
that instance's launch; **explicit `run_settings` values in config (or per
instance) always override calibrated ones**, and with no matching profile
behavior is exactly as before. No dashboard surface (use
`toaster-loop calibrate --show`).

### MCP control plane (`toaster-loop mcp`)

`toaster-loop mcp` serves an [MCP](https://modelcontextprotocol.io) (Model Context
Protocol) server that exposes toaster-loop's existing operations as typed tools —
opt-in: nothing else changes when you don't invoke it, and there is **no MCP
SDK** (stdlib-only JSON-RPC re-implementation, per the project's D-12 rule;
the control-plane pattern was inspired by Strata's MCP design).

**Transport: stdio.** The server speaks newline-delimited JSON-RPC 2.0 on
stdin/stdout — MCP's canonical local transport. It opens **no socket at
all**, so the loopback-only networking rule holds by construction; zero
cloud, zero API keys. The request loop is single-threaded (one message in,
one out), which keeps confirm-token handling trivially safe. Diagnostics go
to stderr only; stdout carries JSON-RPC frames exclusively.

Wire it into any MCP client config (the entry point is the module form):

```json
{
  "mcpServers": {
    "toaster-loop": {
      "command": "python",
      "args": ["-m", "toaster-loop", "mcp"]
    }
  }
}
```

(`python -m toaster_loop mcp --config path/to/toaster_loop.yaml` also works; the CLI's
standard `--config`/`--verbose` flags apply.)

**Tool table** — tools mirror existing CLI/API code paths exactly and add no
new authority. Read-only tools execute immediately; **mutating tools are
plan-then-confirm**:

| tool | class | mirrors |
|---|---|---|
| `status` | read | `GET /api/status` (overview chips) |
| `instances_list` | read | `toaster-loop instances status` / `GET /api/servers` |
| `models_list` | read | `toaster-loop models list` / `GET /api/models` |
| `logs_tail` | read | bounded tail of `<home>/logs/server-<id>.log` (≤500 lines) |
| `instance_start` | **mutating** | `POST /api/servers/<id>/start` |
| `instance_stop` | **mutating** | `POST /api/servers/<id>/stop` |
| `instance_load` | **mutating** | `POST /api/servers/<id>/load` (WP5) |
| `instance_unload` | **mutating** | `POST /api/servers/<id>/unload` (WP5) |
| `curate_enqueue` | **mutating** | `toaster-loop curate` / `POST /api/curate` (queued job) |
| `train_enqueue` | **mutating** | `toaster-loop train start` / `POST /api/train` (single-flight: a busy GPU returns an error, never a silent queue) |
| `validate` | **mutating** | the golden-suite gate `toaster-loop iterate` runs (`training.validate.validate_model`) — classified mutating because it spawns a temporary llama-server and occupies the GPU even though it writes no registry state |
| `promote` | **mutating** | `toaster-loop models promote` / `POST /api/models/promote` |
| `bench` | **mutating** | `toaster-loop bench` measure mode (may spawn a server; persists a record under `<home>/bench/`) |

**Plan/confirm flow.** Call a mutating tool without `confirm_token` and you
get a PLAN back instead of execution: `{action, target, exact_effect, risks,
params}` plus an opaque `confirm_token`. Call again with the same arguments
plus that token to execute. Tokens are random (`secrets`, 256 bits), stored
**in memory only** (a restart drops all outstanding plans), expire after
**300 s**, are **single-use** (executing consumes them; replays are refused),
and are **bound to (tool, args-hash)** — a token issued for one call cannot
execute a different tool or the same tool with different arguments.

**Schema validation.** Every tool declares a fixed JSON Schema
(`additionalProperties: false`, scalar types, enums for mode-like args). The
server refuses unknown arguments, missing required arguments, wrong types
(no coercion beyond an int satisfying `number`) and enum violations with a
JSON-RPC `-32602` error before any executor runs.

**Security posture.** No argument is ever passed to a shell (executors call
Python functions in-process; a test greps the package for shell/exec
patterns). No tool accepts a filesystem path — `logs_tail` takes an instance
id validated against the same pattern `serving/instances.py` enforces, and
the constructed path is containment-checked with
[`core/path_security.py`](toaster_loop/core/path_security.py) before opening.
Malformed input yields JSON-RPC parse errors, never a crash.

### Dataset curation (Phase 3)

```bash
toaster-loop curate --name myrun --min-score 0.5 --since 2026-09-01 \
  [--profile toaster-loop] [--include-failures] [--limit 500] [--format both]
# -> dataset 20260902T011727Z--myrun
#    dir:     ~/.toaster-loop/datasets/20260902T011727Z--myrun
#    sft:     37 rows -> sft.jsonl
#    dpo:     4 pairs -> dpo.jsonl
#    claimed: 39 traces (curated=1, dataset_id=...)

toaster-loop datasets list
toaster-loop datasets show 20260902T011727Z--myrun     # manifest JSON
toaster-loop datasets export <id> --to ~/exports        # copy out for training
```

**Selection**: unclaimed traces matching status/time/profile filters pass when
their effective score (user feedback overrides the heuristic; NULL is never
treated as 0) ≥ `curation.min_score` (default 0.5), or they carry explicit
positive user feedback. `failure` traces are excluded unless
`--include-failures`, in which case they serve only as DPO negatives. A
lightweight verifier pass re-scans tool messages for error-shaped content the
record-time heuristic may have missed; traces whose *only* tool results are
errors are dropped from SFT unless something positive says otherwise.

**Dedup**: REPL sessions replay earlier messages in each subsequent trace.
Traces are grouped by hash(first user message + agent profile) and only the
longest (then most recent) per group is exported.

**Claiming / idempotency**: selected traces are marked `curated=1,
dataset_id=<id>` — a trace appears in at most one dataset version. Re-running
skips claimed traces; `--rebuild <dataset_id>` releases that dataset's claims
and rewrites it from the current buffer state.

**LLM-as-judge (opt-in, `--judge N`)**: scores the top-N *borderline* traces
(`|score − min_score| < curation.judge_band`, default band 0.15) by asking the
served model a rubric question with forced JSON output
(`{"score": 0.0-1.0, "reason": "..."}`; bounded review context, defensive
parsing — unparseable replies count as skipped, never crash). Verdicts land as
feedback rows `source=judge`; effective-score precedence becomes **user >
judge > heuristic**. OFF by default: no flag = zero LLM calls (judging costs
GPU time). ⚠️ **Bias caveat**: the judged model grading its *own* traces is
biased — treat judge scores as a tiebreaker near the threshold, not ground
truth. Config knobs are limited to `curation.judge_band` and optional
`curation.judge_model`; no config key enables judging itself.

**Dataset import (`toaster-loop datasets import <file> [--format auto|alpaca|sharegpt|chatml] [--name N]`)**:
seeds the loop from external data. Reads `.jsonl` (or a `.json` array) and
auto-detects per row: `conversations` → sharegpt (`from`/`value` or
`role`/`content` keys), `instruction`(+`output`) → alpaca (`input` becomes a
context prefix), `messages` with role/content → chatml. Rows normalize through
the same ShareGPT builder curate uses, so the written `sft.jsonl` is contract-
identical and trains unchanged. Malformed rows are skipped and counted
(summary + manifest `skipped_rows`). The dataset dir carries manifest
`provenance: "imported"`, `source_file`, `source_format` and empty trace-id
lists (imports have no trace provenance / claiming); it shows up in normal
`datasets list` and is selectable for `train start`/`iterate --dataset` with
no further steps. Undetectable/unreadable input exits non-zero showing the
keys seen. (Follows the trainer ecosystem's `format_type` convention.)

**Outputs** (`<home>/datasets/<dataset_id>/`):

- `sft.jsonl` — one ShareGPT row per trace:
  `{"conversations": [{"from": "system|human|gpt|tool", "value": "..."}],
    "trace_id": ..., "model": ..., "completed": ...}`. Tool calls render as
  `< tool_call>{"name":...,"arguments":{...}}</ tool_call>` inside `gpt` turns
  and results as `< tool_response>{"tool_call_id","name","content"}</ tool_response>`
  blocks — the standard ShareGPT trajectory format, so
  `format_type: sharegpt` works unchanged.
- `dpo.jsonl` — TRL `PreferenceDataset`-style rows:
  `{"prompt": [{"role","content"},...], "chosen": "...", "rejected": "...",
    "metadata": {"chosen_trace_id", "rejected_trace_id"}}`. Pairs are mined
  per conversation group (highest-scored final answer vs lowest/failure one);
  ties produce no pair. No pairs → empty file + warning.
- `manifest.json` — dataset_id, created_at, schema versions, filter params,
  per-file counts + source trace_ids, format hints, warnings.
- `train.yaml` — template for the documented `train --config` contract
  (`model/data/training/lora/logging`; `data.local_dataset` points at
  `sft.jsonl`, `format_type: sharegpt`). **Phase 4 executes it; no trainer
  library is imported** (the import-allowlist hygiene test enforces this).

### State layout

Root = `TOASTER_LOOP_HOME` env > config `home:` > `~/.toaster-loop`:

```
<home>/
  buffer/
    toaster_loop.db                  # SQLite index: traces + feedback (WAL)
    YYYY/MM/<trace_id>.jsonl    # append-only full conversation per trace
                                # (header / message* / footer, schema_version 1)
  datasets/
    <dataset_id>/               # one dir per curate run
      sft.jsonl  dpo.jsonl      # ShareGPT + TRL-preference exports
      manifest.json  train.yaml # provenance + train.yaml template
  train_runs/
    <job_id>.log                # worker progress (JSON lines)
    <model_id>/train.resolved.yaml + adapter/   # per-model run artifacts
  models/
    gguf/<model_id>-<quant>.gguf# exported serving artifacts
  archive/
    <model_id>/                 # archived GGUF/adapter dirs (recoverable)
  index/
    <project_id>.db             # per-project codebase index (Phase 5);
    default.db                  # fallback DB for no-project/config-driven use
  projects/<id>.json            # project records (workspace_dir, agents,
                                # description, agent_roles {profile: their job})
  chat/                         # dashboard chat threads (one JSON per thread)
  uploads/                      # chat image attachments (janitor-pruned)
  batches/<run-name>.json       # batch-run checkpoints (stats only)
  iterate_runs/iter-*.json      # iterate run records (step timeline)
  easy_training/state.json      # Easy-training run state
  skills/<name>/                # agent skills: SKILL.md + scripts dir
                                # (skills.enabled; see docs/SKILLS.md)
  logs/dashboard.log            # launched-dashboard log (`toaster-loop launch`)
  janitor_state.json            # disk-janitor last_run_at/run_count/summary
  serve_state.json              # toaster-loop-owned llama-server pid/base_url
                                # (the reserved "default" instance)
  serving/
    instances/<id>.json         # named server instance definitions (Phase 6)
    state/<id>.json             # per-instance runtime state (pid/base_url)
  bench/                        # `toaster-loop bench` records + noise-floor baselines
                                # (TOASTER_LOOP_BENCH_DIR overrides)
  calibrate/                    # per-machine calibration profiles, one JSON
                                # per machine-id + model hash
                                # (TOASTER_LOOP_CALIBRATE_DIR overrides)
```

Registry state (models, jobs, active pointers) lives in
`buffer/toaster_loop.db` alongside the trace index (single-writer SQLite).

**Model status machine**: `candidate → validated → promoted → superseded`;
validation failure ⇒ `failed`; rollback ⇒ current becomes `rolled_back` and
the most recent `superseded` model of that base returns to `promoted`.
New models auto-number `vN`, pin `dataset_id + manifest_hash`, and record
`parent_adapter` = the active model at registration time (lineage chain).

**Archive lifecycle**: `superseded / failed / rolled_back → archived` via
`toaster-loop models archive <id>` or the disk janitor — never automatically from
any other status, and never for `pinned` models (`models pin/unpin`). Archival
*never deletes*: GGUF + adapter move under `<home>/archive/<model_id>/` and
`models restore <id>` moves them back to their exact recorded paths
(`pre_archive_paths`, registry schema v5). Archived rows have their artifact
columns cleared, and `deploy` refuses archived models outright. The janitor
archives only terminal (`superseded/failed/rolled_back`) models older than
`janitor.retention_days` (age = `promoted_at` when the model ever went live,
else `created_at`), skips pinned rows, records runs in
`<home>/janitor_state.json`, and runs automatically at
`toaster-loop train auto-check` time only when `janitor.enabled: true`
(default **false** — opt in; manual runs via `toaster-loop models janitor` always
work, `--dry-run` lists what would move without touching anything).

The janitor also prunes aged **operational files** (worker/server logs, chat
`uploads/`, spillover cache, batch checkpoints) older than
`janitor.disk_retention_days` (default 90; 0 disables). Unlike model
archives these deletions are final, so only those allow-listed directories
are ever touched — models, datasets, buffer and archive contents never are.
Preview with `toaster-loop models janitor --dry-run [--disk-days N]`.

**Job states**: `queued → running → done|error|cancelled`; FIFO on
`(queued_at, job_id)`; one job at a time (single GPU arbiter); a worker killed
-9 is reaped to `error` by the next scheduler pass; cancel SIGTERMs the worker.

### Training & validation config

`training.base_model` (config) must be an **HF checkpoint id or local HF-format
dir** — never a GGUF (GGUFs are serve-only). GPU-heavy steps run in a
subprocess (`python -m toaster_loop.training.worker`); set `training.python` to the
interpreter that has torch. `training.backend` selects the SFT implementation:
default **`peft`** (today's transformers+peft path, unchanged) or **`streamed`**
— a layer-streaming trainer for checkpoints bigger than VRAM: the checkpoint
(local dir only, never downloaded here) is split by layer on disk (~1x model
size extra disk while splitting; `training.delete_original` reclaims copies
under the run dir only), trained through a meta shell + bounded layer window
with PEFT-free LoRA and backward recompute, logs per-step
`train_metrics.csv` (`step,loss,peak_vram_gb,elapsed_s`) plus splitting
progress to the job log, and emits a GGUF-LoRA artifact that flows through the
same export/-lora hot-swap path. GPU/VRAM effects are unmeasured so far
(CPU-only development box — no speed claims). Mistyped values warn and fall
back to `peft`. The export step merges the adapter with peft and
runs llama.cpp `convert_hf_to_gguf.py` + `llama-quantize` (or emits an AWQ/GPTQ
HF checkpoint when `training.export_format: awq|gptq`); a streamed-backend run
instead ships its GGUF-LoRA directly (no merge, no quantize — adapters are
`-lora`-loaded onto the base). When neither path is
possible it fails the export job with an actionable error and
**keeps the adapter artifact**. `validation.golden_tasks` is a tiny list of
tripwire tasks — promotion requires `pass_rate >= validation.min_pass_rate`.
Two check kinds (mixable per task): `{prompt, expect_contains?,
expect_not_contains?}` substring tripwires, and `{prompt, expect_tool_call:
{name?, arg_contains?}}` tool-call grammar checks (backlog #5) that send the
prompt with the production `TOOL_SCHEMAS` and require a recoverable tool call
(native `tool_calls` field or text-emitted calls via `core/tool_calls.py`) —
closing the hole where a quantization/merge bug breaks function calling but
prose answers still pass substring checks. See
[`config.example.yaml`](config.example.yaml) for every key.

### Manual test with a real GPU (smoke)

```bash
# in an env with the [training] extra (torch/transformers/peft/trl):
toaster-loop train start --dataset <id> --dry-run   # config+dataset validation only
toaster-loop iterate --name smoke                    # full loop on the real trainer
toaster-loop models list && toaster-loop deploy v1        # go live
```

Deleting `<home>` fully resets toaster_loop. Traces are captured post-run from the
completed message list returned by the runner — one canonical capture point
for the built-in runtime.

### Config

Copy [`config.example.yaml`](config.example.yaml) → `config.yaml`, pass with
`--config path/to/config.yaml` or `TOASTER_LOOP_CONFIG`; when unset, config loads
from `<TOASTER_LOOP_HOME>/toaster_loop.yaml` (the file the dashboard Settings panel
writes). Environment variables (`TOASTER_LOOP_HOME`, `TOASTER_LOOP_GGUF`,
`TOASTER_LOOP_MODELS_DIR`, `TOASTER_LOOP_PORT`, `TOASTER_LOOP_CONTEXT`,
`TOASTER_LOOP_ATTACH_URL`, `TOASTER_LOOP_MODEL`,
`TOASTER_LOOP_LLAMA_SERVER`, `TOASTER_LOOP_RECORDING`, `TOASTER_LOOP_DASHBOARD_PORT`,
`TOASTER_LOOP_EXTERNAL_SERVER_URL`) override file values. The optional `dashboard:` config
section (see [`config.example.yaml`](config.example.yaml)) takes `port:` and an
optional shared-secret `token:` (also `--token` / `TOASTER_LOOP_DASHBOARD_TOKEN`;
when set, every `/api/*` call must carry `X-Toaster Loop-Token`) — the bind host is
always loopback.

The `delegation:` section gates the agent-discovery/delegation toolset
(`list_agents`, `delegate_task`, `check_delegation`): `enabled` (default true,
env `TOASTER_LOOP_DELEGATION`) and `timeout_seconds` (default 300, env
`TOASTER_LOOP_DELEGATION_TIMEOUT_S`) — the default cap on how long `delegate_task`
waits for a child agent run before returning a timeout payload. Children run
on the active project's attached profiles; see
[`docs/DECISIONS.md`](docs/DECISIONS.md) D-11.

The `serve:` section carries serving-launch behavior: `calibration: off|auto`
(default `off`) opts into applying `toaster-loop calibrate` profiles at launch —
with `auto`, adopted knob values for this machine-id + model-hash become
defaults underneath explicit `run_settings` (manual values always win; with no
matching profile, or the key absent, launches are byte-identical to before).
See [Calibrate (per-machine autotune)](#calibrate-per-machine-autotune).

The same section carries the **engine watchdog** (WP4) — detection and
automatic recovery of a FROZEN `llama-server` child (wedged mid-request: pid
alive but zero CPU-time growth and zero disk-IO growth from `/proc` while a
request stays in flight; a healthy-but-slow server computing at full CPU is
never killed, and an idle server never triggers). **Strictly opt-in** — the
master switch defaults to false, so with no config changes nothing runs:

```yaml
serve:
  watchdog_enabled: false            # master switch (default off)
  # watchdog_window_seconds: 90      # stall duration before a freeze call
  # watchdog_sample_interval_seconds: 5   # /proc sampling cadence
  # watchdog_cpu_threshold_ms: 10    # "≈0 CPU": max utime+stime growth/sample
  # watchdog_io_threshold_bytes: 0   # "≈0 IO": max read+write growth/sample
  # watchdog_kill_grace_seconds: 8   # SIGTERM -> SIGKILL escalation grace
  # watchdog_restart_backoff_base_seconds: 2   # base * 2^(attempt-1)
  # watchdog_max_restarts: 3         # cap per window, then give up loudly
  # watchdog_restart_window_seconds: 600
```

On detection the pid is re-verified as toaster-loop-owned before any signal is sent
(foreign/recycled pids are never killed — stale state is cleared instead),
then SIGTERM → SIGKILL after grace and a restart with exponential backoff up
to the cap. Events land on the dashboard SSE bus as `serve.watchdog`
(`freeze_suspect`, `freeze_detected`, `instance_killed`,
`stale_state_cleared`, `instance_restarted`, `restart_failed`,
`restart_capped`). Linux only: without `/proc` the watchdog logs once and
never kills. See [`docs/ROADMAP.md`](docs/ROADMAP.md) WP4 for the detection
rule and rationale.

The same section carries **GPU coexistence** (WP5) — idle-unload, lazy-load and
the train+serve handshake (inspired by Strata's idle-unload pattern, re-
implemented inline). **All keys default off**; with no config changes behavior
is byte-identical to before:

```yaml
serve:
  idle_unload_seconds: 0           # 0 = disabled; > 0 gracefully stops any
                                   # running instance idle this long (never
                                   # mid-request) and marks it `unloaded`
  lazy_load: false                 # true: requests to an unloaded instance
                                   # transparently reload it first (blocking;
                                   # cold-start accepted). false: clear error
  min_free_vram_mib: 0             # with idle_unload_seconds > 0, arms the
                                   # train+serve handshake on the scheduler's
                                   # VRAM gate
  coexist_policy: refuse           # refuse (default): a job that WOULD pass
                                   # assuming unloaded servers only emits a
                                   # `coexist_refused` event with headroom
                                   # numbers; unload: unload first, wait for
                                   # VRAM to free, then admit the job
```

Race policy: load/unload serialize per instance id — a request arriving
mid-unload blocks and then reloads transparently (`lazy_load: true`) or gets a
clear retryable error. Events land on the dashboard SSE bus as `serve.coexist`
(`instance_unloaded`, `instance_loaded`, `coexist_unload_triggered`,
`coexist_refused`). Explicit control:

```console
$ toaster-loop instances status              # every instance + load state
$ toaster-loop instances unload coder        # graceful stop, recorded unloaded
$ toaster-loop instances load coder          # (re)launch by id
```

or `POST /api/servers/<id>/unload` / `/load` on the dashboard API (instance
rows carry a `load_state` field: loaded | unloaded | unloading | stopped).
See [`docs/ROADMAP.md`](docs/ROADMAP.md) WP5 for the handshake algorithm.

The `tools:` section carries the editor-style post-write error checking for
coding agents: `lint_on_write` (default true, env `TOASTER_LOOP_LINT_ON_WRITE`)
runs a cheap tier-1 syntax check (`py_compile` for `.py`, `node --check` for
JS/TS when available) on the new content of every successful
`write_file`/`patch` — the write is never rejected, but failures append a
`✗ SYNTAX ERROR in <path> …` block (with pre-existing errors filtered out via
a delta against the pre-edit content) so the model fixes them next turn. The
optional `tools.lsp:` sub-section (`enabled`, default false, env
`TOASTER_LOOP_LSP`) adds tier-2 LSP diagnostics: a detected language server
(pyright / typescript-language-server / gopls / rust-analyzer by default;
override per extension via `servers:`) is driven over stdio JSON-RPC with a
pre-edit shadow-document baseline, so only NEW diagnostics are surfaced. All
probes are timeout-bounded and degrade to silent skips when no server exists.

The `indexing:` section drives per-project codebase semantic search (the
`search_codebase` tool + the Settings card). Phase 5 schema: `enabled`
(default false), `model` (embedding GGUF path — toaster-loop spawns and stops its
own loopback `llama-server --embedding` for it; only processes it spawned are
ever terminated), `output_dim` (`768|512|256|128` Matryoshka (MRL) truncation
applied to stored *and* query vectors with L2 renormalization; a query whose
dim/model differs from the built index reports "index needs rebuild" instead
of returning garbage), `search_mode` (`hybrid|semantic|lexical`) and the
advanced `paths_override` (manual roots for the default/no-project DB only —
each project indexes its own `workspace_dir`). One sqlite DB per project lives
at `<TOASTER_LOOP_HOME>/index/<project_id>.db` (+ `default.db`). Projects enroll
themselves: creating or editing a project from the Chat page (or the API)
starts a build of **that project's own** DB when `enabled` — nothing is ever
appended to `paths_override`, and one project's build/search never sees another
project's files. Deleting a project deletes its index DB (and only that file). Legacy keys (`embedding_url`, `embedding_model`,
`top_k`, `paths`) load without crashing: `embedding_model` maps to `model`,
`paths` to `paths_override`, and the url/`top_k` are dropped (toaster-loop owns
the server; top_k is an internal constant).

### Exit codes

Every `toaster-loop` command follows this taxonomy (source of truth:
[`toaster-loop/core/exitcodes.py`](toaster_loop/core/exitcodes.py)):

| Code | Meaning |
|------|---------|
| 0    | success |
| 1    | generic runtime error (job failed, backend error, …) |
| 2    | usage error — argparse rejects the command line, and post-parse invocation mistakes (mutually exclusive flags, bad `--config-key` shape, `bench compare` with an unreadable operand) |
| 3    | **invalid configuration** — the YAML failed validation; stderr carries ONE consolidated report listing every unknown / missing / mistyped field at once ("refuse to guess") |
| 4    | **finished but corrupt artifact** — the process flow completed, yet a produced artifact failed its integrity check (truncated / zero-byte / bad-magic GGUF, NaN loss recorded). Distinct from 1 so automation can tell "the run errored" from "the run looked fine and quietly produced garbage" |
| 10   | `toaster-loop llama update --check`: an update is available (established informational code) |
| 130  | interrupted (Ctrl-C), per shell convention |

Config strictness (WP2a): mistyped values are **always** errors with the
consolidated report; unknown keys stay warn-only by default so stray keys in
existing configs keep loading unchanged — set `TOASTER_LOOP_CONFIG_STRICT=1` to
promote them into the same error. Found values in reports pass through the
secret redactor, so a mistyped credential field never echoes its value.

### Post-write error checking (editor loop)

Tier 1 is always on (`tools.lint_on_write`): after a successful write or
patch the new content is syntax-checked in-process — Python via
`py_compile.compile(..., doraise=True)` against a throwaway temp file (no
`.pyc`, nothing executed), JS/TS via `node --check` on a temp file with a
5-second timeout (TS only when node ≥ 23 can parse it; missing node = skip).
Unknown extensions are skipped silently. On failure the tool result carries
`✗ SYNTAX ERROR in <path> (edit rejected? no — file WAS written; fix it next
turn)` plus the parser error (`path:line: Type: message`); on success a
subtle `✓ syntax ok`. Errors that already existed in the pre-edit content
are filtered out of the report. The dashboard chat renders failed checks as
a red visible block inside the tool card. Tier 2 (`tools.lsp`) is opt-in and
reports only diagnostics newly introduced by the edit.

## Tests

```bash
pytest tests/ -v
```

The tests spin up a **fake OpenAI-compatible server** (stdlib, no model
download) and assert: health-check works, attach mode reaches it, the agent
loop completes a full tool-call → tool-result → final-answer round trip, and —
all hermetically under a temp `TOASTER_LOOP_HOME` — trace JSONL +
SQLite rows are written, feedback/traces CLIs work, recording can be disabled,
a broken recorder cannot break a run, and the built-in runtime produces a
complete trace. Phase 3 adds curation tests (traces seeded via the store
API): ShareGPT golden fixture + tool-call rendering round trip, prefix-overlap
dedup, DPO pair mining (incl. empty/tie cases), claiming idempotency +
`--rebuild`, manifest completeness, and `train.yaml` YAML validity. Phase 4
adds: registry CRUD/lineage/promote/rollback transitions; scheduler FIFO,
single-flight, cancel (SIGTERM verified), crash reaping, error capture from a
failing stub worker, auto-trigger check; the validation gate against a
scripted fake server (pass/fail thresholds); and `iterate` end-to-end with a
stubbed worker + stub converter — dataset → v1 promoted → second run v2 with
parent=v1 → rollback returns the pointer to v1. Phase 5 adds dashboard tests:
API-level coverage of every endpoint (status/traces/feedback/datasets/models/
jobs/curate/train with a stub worker), a real-server test asserting loopback
bind + static serving + path-traversal rejection, and hygiene tests that the
shipped UI contains no external URLs (offline-capable). Later passes add
coverage for hub/model-discovery, indexing, skills, playbooks, evolution,
shadow/transfer eval, RFT, golden mining, ledger/tiers and multi-instance
serving. No GPU required anywhere, and no agent or trainer framework is
ever imported — training runs on transformers/peft/trl directly (enforced by
the `tests/test_hygiene.py` import allowlist).

## Manual test with a real local GGUF

1. Make sure `llama-server` exists (`toaster-loop setup` builds one under
   `~/.toaster-loop/llama.cpp`; toaster-loop also checks PATH).
2. Grab any tool-call-capable GGUF (Qwen-family recommended),
   e.g. `qwen3-8b-q4_k_m.gguf`, or download one via `toaster-loop hub`.
3. `toaster-loop serve --gguf /path/to/model.gguf -v` and confirm the printed
   base_url; `curl http://127.0.0.1:<port>/v1/models` should list the model.
4. In another shell: `toaster-loop run --attach http://127.0.0.1:<port> --task "list files in ." -v`.
5. Kill the server mid-conversation (`Ctrl-C` on serve): the agent must exit
   with a clear connection error, not hang.

## License / boundaries

- No external agent or trainer frameworks are imported — the runtime, toolset,
  training pipeline and GGUF export are built in on top of standard
  open-source libraries (httpx/pyyaml; torch/transformers/peft/trl for the
  optional `[training]` extra; llama.cpp via subprocess for conversion).
- Third-party imports are restricted to an explicit allowlist
  (`pyyaml`, `httpx`, stdlib, plus the lazy training extra) enforced by
  `tests/test_hygiene.py`. External servers remain optional attach
  targets over HTTP only.
- Loopback-only networking; no telemetry.

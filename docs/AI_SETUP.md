# AI_SETUP — agent-facing install-and-operate playbook

Everything in the command blocks below exists in [`sable/cli.py`](../sable/cli.py)
(verified by grep + execution). Expected outputs were produced by actually running
the hermetic commands; steps that need a real GPU/model describe their output
**shape** and are marked *(needs model/GPU)*.

Cardinal rules for agents working on this repo: see [AGENTS.md](../AGENTS.md).

---

## 1. Prerequisites

- Python **3.11–3.13** (pyproject `requires-python = ">=3.11,<3.14"`).
- git, cmake and a C/C++ compiler — only for building llama.cpp (`sable setup`).
- A GGUF model file for serving/training, or use `sable hub download` to fetch one.
- Zero API keys, zero cloud: everything is loopback-only by design.

## 2. Install

> **Any OS.** The bash scripts below are Linux/macOS. On Windows use the
> PowerShell equivalents (same flags, same behavior): `.\setup.ps1`,
> `.\launch.ps1`, `.\cleanup.ps1` — or skip the shims entirely and run the
> module form everywhere: `python -m sable setup|launch|...`. All CLI
> commands, env vars (`SABLE_HOME` etc.) and file layouts are identical
> across platforms; state always lives under `~/.sable/`.

One-click (creates `.venv`, installs the package + `[training]` extra, builds
llama.cpp for this GPU, adds a desktop launcher):

```bash
./setup.sh                    # full setup
./setup.sh --no-training      # skip the heavy torch training stack
./setup.sh --backend cuda|rocm|metal|vulkan|cpu   # override GPU autodetect
```

Expected output shape: venv creation lines, `pip install -e .` progress, then the
`sable setup` backend detection (`platform : Linux/x86_64`, `backend : cuda` …)
and a final launch hint. *(the llama.cpp build step needs git+cmake+compiler)*

Manual install:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e .                 # core deps only: pyyaml + httpx (+Pillow)
pip install -e '.[training]'     # torch/transformers/peft/trl — train/export jobs
```

Sanity check (hermetic — safe anywhere):

```bash
python -m sable --help
```

Expected output (verified): usage line listing the subcommand surface —
`{serve,run,traces,curate,datasets,feedback,insights,history,models,hub,train,
deploy,dashboard,iterate,evaluate,evolve,rft,mine-goldens,playbooks,doctor,setup,
update,launch,reset,bench,calibrate,instances,mcp}`.

## 3. First-run config

Config precedence: **CLI flags > env vars > YAML file > built-in defaults**.
With no `--config` and no `SABLE_CONFIG`, sable loads `<SABLE_HOME>/sable.yaml`
(the file the dashboard Settings panel writes); every knob has a default, so a
missing config is fine. [`config.example.yaml`](../config.example.yaml) documents
every key. Essentials for a first loop:

```bash
mkdir -p ~/.sable
cp config.example.yaml ~/.sable/config.yaml   # then edit the essentials below
# (or export SABLE_CONFIG=/path/to/config.yaml / pass --config per command)
```

```yaml
gguf_path: ~/models/qwen3-8b-q4_k_m.gguf  # model to serve/spawn
port: 0                                   # 0 = dynamic loopback port (recommended)
context_length: 65536                     # doctor prints a measured recommendation
recording:
  enabled: true                           # experience buffer ON by default
```

Config strictness (WP2): mistyped values **always** fail with ONE consolidated
report listing **every** problem field at once (exit 3). Unknown keys warn by
default; `SABLE_CONFIG_STRICT=1` promotes them into the same error.

State lives under `~/.sable/` (override: `SABLE_HOME`) — buffer, datasets,
registry, `bench/`, `calibrate/`, `logs/server-<id>.log`. Uninstall = delete dir.

## 4. Environment doctor (measure, don't assume)

```bash
sable doctor          # runtime diagnosis: GPU backend, llama-server, offload
sable doctor --env    # WP7 measurement report (disk/RAM/VRAM presets + next commands)
python -m sable.doctor --fix   # same as --env, plus stale-state cleanup
```

`--env` measures (never assumes): free disk on the SAME filesystems export writes
to (`shutil.disk_usage`, incl. temp/staging dir) vs the largest registered model +
safety margin; `MemAvailable` from `/proc/meminfo` → recommended serve context /
train batch presets **with the formulas printed**; free VRAM via a local
`nvidia-smi` call when present (absence reported honestly); config parse (verbatim
consolidated errors), `~/.sable` writability, llama-server discovery, loopback
port availability, stale instance state (dead pids; removed only with `--fix`,
and only when the pid is verified gone — foreign pids are never touched).

Verified example output (fresh home, CPU box):

```text
sable doctor — measured environment report
============================================
[OK  ] python
       3.13 (supported 3.11..3.13)
[OK  ] deps
       pyyaml: present
       httpx: present
       [training] extras: 4/4 present
[OK  ] config
       parses
[OK  ] home
       /tmp/tmp.XXXX writable
[OK  ] disk
       required for largest model (none registered) + 10% margin (min 2.0 GiB): 2.0 GiB
       /tmp/tmp.XXXX: free 61.3 GiB OK
[OK  ] memory
       MemAvailable: 5.0 GiB
       serve context: 5.0 GiB MemAvailable -> -c 4096 (tier rule: ...)
       train batch size: 5.0 GiB // 16.0 GiB per sample -> batch_size 1 (capped at 8)
       vram: unavailable (no nvidia-smi) — -ngl tier skipped; auto-detect stays in charge
[FAIL] llama-server
       NOT FOUND — Could not find llama-server on PATH ...
[OK  ] ports
       no static ports configured
[OK  ] state
       no stale instance state files
[WARN] noise-floor
       no persisted noise-floor baseline — effect-vs-noise judgments fall back to the 3% default threshold

next commands:
  $ run: sable setup
  $ run: sable bench --noise-floor
```

Exit codes: `0` all good (warnings allowed), `1` actionable problems found,
`3` invalid configuration (the consolidated stderr listing is printed verbatim).

## 5. Operate — the record → curate → train → export → promote → serve loop

```bash
# 0) get a model if you don't have one yet *(needs network for the download)*
sable hub search qwen3
sable hub variants <repo>              # list quant variants
sable hub download <repo> --variant <name>

# 1) build llama.cpp once (any GPU: CUDA/ROCm/Metal/Vulkan/CPU)
sable setup

# 2) serve + record: every `run` against the endpoint lands in the experience buffer
sable serve --gguf ~/models/qwen3-8b-q4_k_m.gguf        # spawn mode, loopback
sable run --task "summarize README.md"                  # one-shot agent task
sable traces list --limit 5                             # verify recording

# 3) curate buffered traces into a dataset
sable curate --name first-pass --min-score 0.6
sable datasets list

# 4) train (queue a job; --dry-run validates config+dataset without GPU)
sable train start --dataset <dataset_id> --kind sft --dry-run
sable train start --dataset <dataset_id> --kind sft
sable train status                       # poll; cancel with `train cancel <job>`

# 5) export — the GGUF export stage runs inside iterate (adapter -> merged ->
#    GGUF via llama.cpp); dataset copies for external training use:
sable datasets export <dataset_id> --to /path/outside/dir

# 6) promote (validation gate replays golden tasks BEFORE promotion passes)
sable models list
sable models promote <model_id> --intent "fix tool-call formatting"
sable models rollback                   # one command back, any time

# 7) serve the promoted model and hot-swap
sable serve --auto                      # serves ACTIVE promoted GGUF
sable deploy <model_id>                 # hot-swap a running instance

# One call for steps 3-6:
sable iterate --name loop1 --kind sft   # curate->train->export->validate->promote

# Dashboard (loopback web UI over the same APIs):
sable launch --no-browser               # idempotent, prints the URL
```

*(steps 2/4/5/7 need a real model; on a GPU box `serve` prints the resolved
`base_url` like `http://127.0.0.1:<port>/v1` and train status streams job phases.)*

## 6. Measurement commands (program features)

```bash
# WP1 bench: temperature-0 tokens/sec over N runs + persisted noise floor
sable bench --runs 5
sable bench --noise-floor               # baseline record under <home>/bench/
sable bench --compare BASELINE CANDIDATE   # by --label or JSON path
sable bench --list

# WP3 calibrate: per-machine knob sweep; adopted only beyond max(3%, noise floor)
# AND token-identical at temperature 0. Profiles under <home>/calibrate/.
sable calibrate --quick
sable calibrate --knobs flash_attn,batch_size
sable calibrate --show          # / --list / --clear

# Serving honors a profile ONLY with serve.calibration: auto (config); explicit
# run_settings always win.

# WP5 coexistence (all defaults OFF): serve.idle_unload_seconds, serve.lazy_load,
# serve.min_free_vram_mib, serve.coexist_policy: refuse|unload. Manual control:
sable instances status
sable instances load <id>
sable instances unload <id>
# dashboard equivalents: POST /api/servers/<id>/load and .../unload

# WP4 watchdog (default off): serve.watchdog_enabled: true (+ watchdog_* keys,
# see config.example.yaml) — detects a frozen llama-server via CPU/IO deltas,
# re-verifies ownership, restarts with backoff.
```

## 7. MCP wiring for agents

`sable mcp` speaks MCP over **stdio** (newline-delimited JSON-RPC 2.0 on
stdin/stdout — no socket, loopback rule holds by construction; stdlib-only).
Client config example (spawns the server as a subprocess):

```json
{
  "mcpServers": {
    "sable": {
      "command": "python",
      "args": ["-m", "sable", "mcp"]
    }
  }
}
```

13 tools: **read** (`status`, `instances_list`, `models_list`, `logs_tail`) and
**mutating** (`instance_start/stop/load/unload`, `curate_enqueue`,
`train_enqueue`, `validate`, `promote`, `bench`) — full table in the README.
Mutating tools are **plan-then-confirm**: the first call returns a plan plus a
single-use token (TTL 300 s, bound to tool+args); the second call presenting the
token executes. Reuse/expiry/wrong-tool tokens are clean refusals.

## 8. Troubleshooting

Exit-code taxonomy: [`sable/core/exitcodes.py`](../sable/core/exitcodes.py)
(0 ok · 1 runtime error · 2 usage · 3 invalid config · 4 finished-but-corrupt
artifact · 10 update available · 130 interrupt).

| Symptom | Likely cause | Exact next command |
|---|---|---|
| exit **3**, stderr lists config problems | invalid YAML — the report is consolidated: EVERY bad field at once | read every `- path: expected …` line in the stderr listing and fix ALL fields, then `sable doctor --env --config <file>` to confirm |
| exit **4** after train/iterate looked fine | finished-but-corrupt artifact (truncated/zero-byte/bad-magic GGUF, NaN loss) | re-run the export (`sable iterate` or re-queue), and check disk: `sable doctor --env` (disk check vs largest model) |
| server won't start, "Could not find llama-server" | llama.cpp backend never built / not on PATH | `sable setup` (builds for this GPU), then `sable doctor` to confirm discovery; `SABLE_LLAMA_SERVER=/path` or config `llama_server_binary:` pin it |
| port busy / bind error | loopback port conflict with another program or instance | `sable instances status` (see owned ports), then `sable serve --port 0` for a dynamic port; doctor's ports check names the conflicting port |
| train refused: VRAM gate | not enough free VRAM while servers hold it | enable coexistence in config (`serve.coexist_policy: unload`, `serve.idle_unload_seconds: 300`) or manually: `sable instances status` then `sable instances unload <id>`, re-queue |
| server frozen (request hangs, no CPU) | engine hang/freeze | enable the watchdog: `serve.watchdog_enabled: true` in `<SABLE_HOME>/config.yaml` (see config.example.yaml keys), or restart via dashboard/`instances unload <id>` + `load <id>` |
| calibration profile stale / model swapped | profile keyed by machine-id + model hash; hardware/model changed | `sable calibrate --clear && sable calibrate --quick` |
| bench compare says token identity **INVALID** | temperature != 0, or the model/endpoint changed between records — speedups over divergent tokens are meaningless | re-bench at temperature 0 on the same fixture: `sable bench --label <new> --runs 5`, and treat sub-noise-floor deltas as noise (`--noise-floor` baseline) |
| "no persisted noise-floor baseline" warning | nothing measured yet to judge effects against | `sable bench --noise-floor` |
| exit **10** from update check | informational: a llama.cpp update is available | `sable update` (rebuild the managed install) |

## 9. Env vars agents should know

| Var | Effect |
|---|---|
| `SABLE_HOME` | state root (default `~/.sable`) |
| `SABLE_CONFIG` | YAML config path (fallback `<SABLE_HOME>/sable.yaml`) |
| `SABLE_CONFIG_STRICT=1` | unknown config keys become errors, not warnings |
| `SABLE_BENCH_DIR` / `SABLE_CALIBRATE_DIR` | relocate bench/calibrate records |
| `SABLE_MACHINE_ID` | override the calibration machine key (tests/containers) |
| `SABLE_GGUF`, `SABLE_PORT`, `SABLE_CONTEXT`, `SABLE_GPU_LAYERS` | flat config overrides |

## 10. Deeper docs

- [README.md](../README.md) — full command list, exit codes, Phase 6 API contract.
- [docs/DECISIONS.md](DECISIONS.md) — ADRs incl. **D-12** (everything re-implemented
  inline; reference projects are never imported — CI-enforced).
- [docs/ARCHITECTURE.md](ARCHITECTURE.md) — source of truth for internals.
- [docs/ROADMAP.md](ROADMAP.md) — improvement-program work packages WP1–WP7.
- [AGENTS.md](../AGENTS.md) — cardinal rules, dependency policy, test commands.

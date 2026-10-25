# AGENTS.md — rules for AI agents working on sable

## The two cardinal rules

1. **New features are opt-in.** With no config changes, every existing command's
   observable output stays **byte-identical**. Ship behavior behind defaults that
   reproduce the old path exactly (see `serve.watchdog_enabled: false`,
   `routing.enabled: false`, `playbooks.enabled: false`, `serve.calibration: off`).
2. **No performance or quality claims without measurement.** Effects below the
   measured noise floor (`sable bench --noise-floor`) must be reported as
   **noise, not wins**. Speed comparisons run at temperature 0; if generated
   tokens diverge, the comparison is INVALID (`sable bench --compare` says so).

## Licensing boundary (D-12) — non-negotiable

Everything is **re-implemented inline**; NEVER import Strata / airllm /
kimi-k3-in-c / sparse_transformers (or any other reference project). Inspiration
belongs in comments, imports never. Enforced by the import allowlist in
[`tests/test_hygiene.py`](tests/test_hygiene.py) — it MUST keep
passing. See [`docs/DECISIONS.md`](docs/DECISIONS.md) (D-3, D-12).

## Dependency policy

- Core: **pyyaml + httpx** (+ Pillow, lazily imported) — nothing else at module level.
- Training stack (torch/transformers/peft/trl/datasets/accelerate/bitsandbytes):
  `[training]` extra, imported **lazily inside `sable/training/worker.py` only**.
- Dashboard, MCP server and the doctor are **stdlib-only** — no new dependencies there.

## Test & lint commands (run before finishing any change)

```bash
python -m pytest tests/ -q      # full suite must be green
ruff check sable tests        # must report "All checks passed!"
```

Hermetic-test conventions: tmp `SABLE_HOME`, injected probes / monkeypatched
hardware readers (see [`tests/test_serving.py`](tests/test_serving.py)),
no subprocess-dependent unit tests, no network.

## Exit-code taxonomy

Defined in [`sable/core/exitcodes.py`](sable/core/exitcodes.py), documented
in README "Exit codes": 0 ok · 1 runtime error · 2 usage (argparse & invocation)
· 3 invalid config (consolidated stderr report) · 4 finished-but-corrupt artifact
· 10 update available (`llama update --check`) · 130 interrupt. Never renumber;
extend additively.

## How to add a config key

Follow the `serve.*` conventions in [`sable/core/config.py`](sable/core/config.py):
add the key to the dataclass field's documented section dict (`serve:`, `tools:`,
…), read it **leniently** (invalid values fall back to the default with a warning)
or register mistypes in the WP2 validator sets so the consolidated report catches
them, document it in [`config.example.yaml`](config.example.yaml), and keep the
default byte-identical to pre-feature behavior. `test_core.py` must
stay green (every valid config loads identically).

## Agent operating playbook

Install/operate/troubleshoot commands with expected outputs:
[`docs/AI_SETUP.md`](docs/AI_SETUP.md). Environment measurement doctor:
`sable doctor --env` ([`sable/doctor.py`](sable/doctor.py)).

## Hard constraints (project-wide)

- **Loopback-only, zero cloud, zero API keys.** No listeners off 127.0.0.1;
  nothing downloads except explicit `hub download` / `setup` (git clone).
- All state under `~/.sable/` (`SABLE_HOME`); uninstall = delete the dir.
- Ownership rule: sable only ever stops processes it spawned (pid verified via
  `serving/runtime.terminate_owned_pid`); foreign/recycled pids are never touched.

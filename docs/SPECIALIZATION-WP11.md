# WP11 — Research spike: per-user specialization from the experience buffer

> **Deliverable = findings.** This is a RESEARCH SPIKE (final work package of
> the improvement program). Nothing was integrated: no config keys, no worker
> changes, no registry changes. Spike code lives in
> [`experiments/wp11_specialization/`](../experiments/wp11_specialization/)
> (outside the `sable` package by design — see "Not-wired seam" below) and
> hermetic tests in
> [`tests/test_wp11_specialization.py`](../tests/test_wp11_specialization.py).

## Hypothesis

If a downstream-eval-judged specialization pass (activation-sparsity or
expert-pruning style, methodology inspired by sparse_transformers — cited,
never imported, D-12) uses **only Sable's own experience buffer** as
calibration data, it can improve the model's downstream golden-task pass rate
by more than the measured noise floor. Adoption language is reserved for
deltas clearing `max(3%, measured floor)`; everything else is **noise, not a
win** (cardinal rule 2).

## Testbed (toy scale — honest about it)

A hand-built 4-expert top-level MoE (~22k params, float32 CPU, torch lazily
imported per the [training]-extra policy) over a 32-char vocabulary. Two task
families ("domains") with disjoint 16-char alphabets: domain A applies a
successor rule `+1 mod 16`, domain B an affine rule `3x+7 mod 16`. A task =
extend a deterministic sequence by one char given its first L chars
(L ∈ {2,3,4}; argmax decode = temperature-0 "greedy completion").

Why hand-rolled: fully inspectable router, ~2 s training on CPU, zero
downloads (`AutoModelForCausalLM.from_config` would also be offline-safe but
hides the router and adds weight for a toy). **Verified finding**: novel-start
generalization is information-theoretically impossible in this encoding
(one-hot char identities carry no order structure) — training on starts
[0,12) and evaluating [12,16) yields exactly 0%. Eval therefore follows
production validation-gate semantics: golden tasks are **regression tripwires
over trained behaviors**, not generalization tests.

**Split discipline (WP11-mandated, enforced by construction):**

* train: every domain × start × prefix length {2,3,4} — full behavior
  distribution (96 tasks).
* calib: 64 recorded buffer traces, prefix length 2 only, ~70% domain-A skew
  (the simulated per-user behavioral skew).
* eval: 32 golden tasks, prefix length 4 only, equal coverage of both domains.
* Calib prompts are 2 chars, eval prompts 4 chars → **no prompt string ever
  crosses the calibration/eval boundary** (asserted in code and tests).

## Calibration data = the experience buffer, only

Calibration inputs are fabricated into a throwaway `SABLE_HOME` through the
**real** [`sable.store.TraceRecorder`](../sable/store/recorder.py) (JSONL
under `buffer/YYYY/MM/` + SQLite index — documented stand-in for a real user's
buffer), then read back via `TraceStore.list_traces` + JSONL transcripts like
curation code does; foreign traces are skipped by the reader (mixed-buffer
tolerance, tested). 64 traces seeded → 16 unique calibration prompts after
the documented dedupe (frequency stats over unique prefixes). No other data
feeds specialization; eval labels never touch the calibration path.

## Methods compared (calibration statistics ONLY)

* **Baseline** — unmodified trained mixture, greedy decode.
* **Variant A `expert_prune`** — mean router load per expert over the buffer;
  drop the least-loaded expert (keep 3/4); router **renormalized** over
  survivors at inference.
* **Variant B `activation_sparse`** — per-unit 25th-percentile activation
  thresholds from the buffer; sub-threshold expert-hidden units zeroed at
  inference (calibrated top-k-style masking).

## Downstream eval = the production gate oracle

Scoring reuses [`training/validate.py::_check_task`](../sable/training/validate.py)
— literally the same function object (test-asserted), so substring `expect_contains`
semantics cannot drift from production. Verdict shape mirrors
`run_golden_suite` (`total/passed/pass_rate/results`) plus the `ok = rate >=
min_pass_rate` gate rule. **Weight MSE and router-shift were deliberately
never computed as adoption evidence.**

## Noise-floor measurement (this box: torch 2.14.1 CPU-only, no CUDA)

* `serving.bench.load_noise_floor()` → **None** (no persisted
  `kind=noise_floor` record exists here; recorded in the report for honesty).
* Therefore the floor was estimated **offline on the eval metric itself**, per
  the WP11 spec: R = 13 baseline replicates at fixed config with init-seed
  variation, spread computed with WP1's own `serving.bench.compute_stats`
  definition `(max−min)/median×100`.

| regime | baseline median pass-rate | measured floor (spread of eval) |
|---|---|---|
| full (600 train steps) | **1.000** | **0.0%** (saturated → 3% program bar applies) |
| limited (8 train steps) | **0.8125** | **19.2% of median** (15.6 pp absolute) |

## Results (R = 13 paired replicates per regime, 32 goldens each)

| regime | variant | eval pass-rate (median) | Δ vs baseline (relative) | noise floor / bar | verdict |
|---|---|---|---|---|---|
| full | expert_prune | 0.844 | **−15.6%** | floor 0.0%, bar 3.0% | **regression** (clears bar downward) |
| full | activation_sparse | 1.000 | +0.0% | floor 0.0%, bar 3.0% | noise (no-op: mask never bites at convergence) |
| limited | expert_prune | 0.719 | −15.4% | floor 19.2%, bar 19.2% | noise (inside measured floor) |
| limited | activation_sparse | 0.812 | +0.0% | floor 19.2%, bar 19.2% | noise |

Reading of the numbers:

* **Activation sparsity was a measurable no-op** (+0.0% in both regimes): at
  q25 thresholds on a converged model, zeroing sub-threshold hidden units did
  not change any greedy completion. At toy scale with dense-ish ReLU experts,
  calibrated masking simply didn't bite.
* **Expert pruning cost ~15% pass rate** in the converged regime — enough to
  clear the 3% program bar as a **regression**. The buffer's router load did
  not identify a genuinely dispensable expert (all four carried weight;
  measured calibration `router_mean` ≈ [0.23, 0.16, 0.33, 0.27] — no dead
  expert to prune). In the limited regime the same −15% delta sits **inside**
  the 19.2% init-to-init noise floor → correctly reported as noise, not a win
  or a proven loss: exactly the discipline the program mandates.
* The saturated full-regime floor (0.0%) shows a ceiling effect: once the
  baseline passes everything, wins are structurally impossible and only
  regressions can register — a general caution for any future golden-task
  specialization study.

## Honest limitations

1. **Toy scale.** A 4-expert linear-router MoE on 2-char rules is not a
   transformer; real expert specialization (if any) emerges in deep FFNs, and
   pruning renormalization behaves differently with softmax-gated top-k
   routing at scale.
2. **CPU-only box** (torch 2.14.1, no CUDA): no GGUF path was exercised — the
   eval is an offline mirror of validate.py's oracle, not llama-server replay;
   quantization interactions with sparsity are unexplored.
3. **Fabricated buffer.** The seeded traces stand in for a real user's
   history; domain-skew realism is synthetic by necessity (documented).
4. **One eval shape** (plen 4) per the disjointness discipline; a richer
   golden suite would tighten the floor (32 goldens → pass-rate granularity
   3.1 pp, itself near the 3% bar — the floor measurement absorbed this
   honestly rather than hiding it).
5. Noise-floor protocol used: **offline eval-metric replicates** (no persisted
   bench floor existed on this box; `load_noise_floor()` returned None — both
   facts recorded in the JSON report).

## RECOMMENDATION: **dead-end at this evidence level; needs-GPU-repeat only
if a real-model testbed is ever justified**

Nothing clears the adoption bar as a win; one variant cleared it as a
regression. Per the program rule, **this program adopts NOTHING into
production** from WP11. If per-user specialization is revisited, the honest
prerequisites are: (a) a real (≥100M-param) MoE checkpoint on GPU, (b) eval
through actual served-GGUF golden replay, (c) the same paired R≥11 protocol —
i.e. a materially more expensive experiment than this spike can justify.

## Reproduce

```bash
python -m experiments.wp11_specialization.run_experiment --replicates 13 \
    --out /tmp/wp11_report.json          # ~8 s, CPU, no network
python -m pytest tests/test_wp11_specialization.py -q   # ~2 s
```

## Not-wired seam (registration note)

The spike lives in a top-level `experiments/` tree — **outside** the
`sable` package — so `test_no_orphans.py`'s import graph and
`test_license_hygiene.py`'s AST scan never see it, which is the deliberate
not-wired seam (WP10-step-1 precedent, strengthened: no orphan-allowlist entry
needed because the package can never reach it; `tests/test_wp11_specialization.py`
asserts both that `sable/research/` does not exist and that no banned
D-12 name is imported anywhere in the spike). torch stays a lazy function-level
import even inside the spike; `pyproject.toml` untouched.

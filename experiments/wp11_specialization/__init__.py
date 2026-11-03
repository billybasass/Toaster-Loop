"""WP11 research spike — per-user specialization from the experience buffer.

NON-PRODUCTION. Nothing here is imported by the ``toaster_loop`` package; this tree
is deliberately outside the package import graph (see tests/test_no_orphans.py,
which only scans ``toaster_loop/``) and outside the license-hygiene scan for the
same reason — the D-12 rules still apply by convention: methodology
inspiration from sparse_transformers-style expert-pruning / activation-sparsity
work is cited in comments and NEVER imported.

Experiment (findings doc: docs/SPECIALIZATION-WP11.md):

* testbed.py      tiny CPU MoE (torch, lazily imported) over a 16-char vocab
* buffer_fixture.py  fabricates an experience buffer with the REAL
                     toaster_loop.store.TraceRecorder into a tmp TOASTER_LOOP_HOME and
                     reads it back through TraceStore/JSONL (calibration data
                     = Toaster Loop's own experience buffer, nothing else)
* specialize.py   two specialization methods using ONLY calibration stats:
                  (A) expert pruning with router renormalization,
                  (B) calibrated activation-sparsity masking
* evalgate.py     downstream golden-task eval that reuses the PRODUCTION gate
                  oracle toaster_loop.training.validate._check_task (expect_contains
                  substring semantics; temperature-0 style greedy decode)
* noise_floor.py  WP1-style noise-floor estimation on the EVAL metric itself
                  (run-to-run spread of baseline pass rate at fixed config,
                  reusing serving/bench.compute_stats) + verdict math
* run_experiment.py  R-replicate driver producing the findings-doc numbers

Run: python -m experiments.wp11_specialization.run_experiment --replicates 13
"""

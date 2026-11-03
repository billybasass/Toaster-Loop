"""Training subsystem (Phase 4): worker, scheduler, validation gate.

Import boundary: the GPU/training stack (torch/transformers/peft/trl/datasets)
is imported ONLY lazily inside functions in worker.py, so the daemon/CLI paths
never require GPU deps and the hygiene test can keep enforcing it. Training is
built on standard open-source libraries directly — no trainer framework is
imported.
"""

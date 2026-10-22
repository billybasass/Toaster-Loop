"""Dependency hygiene: sable is self-contained — no external agent/trainer imports.

The runtime, toolset, curation pipeline, training workers and GGUF export are
all built into sable (docs/DECISIONS.md D-12). This test locks that boundary
in place with a package-wide import allowlist:

  - Runtime deps: pyyaml, httpx, Pillow (+ stdlib). The optional training stack
    (torch/transformers/peft/trl/datasets/accelerate/bitsandbytes) may only be
    imported lazily inside functions in training/worker.py, so importing the
    daemon/CLI never needs the training environment.
  - huggingface_hub is NOT used: the model hub talks REST via httpx.
  - No external agent-framework or trainer-CLI packages of any kind.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "sable"

# Third-party packages the sable source may import at all. Everything else
# (outside stdlib) is an offender. Stdlib is derived from the running
# interpreter; keep this list in sync with pyproject.toml dependencies.
ALLOWED_THIRD_PARTY = {"yaml", "httpx", "PIL"}

# GPU/training stack: function-level imports allowed ONLY in worker.py and
# the streamed training-backend package (WP10 step 2 wired the spike into
# the worker seam; torch/transformers/safetensors stay lazily imported
# inside functions there — the same policy as worker.py itself). Module-
# level heavy imports stay banned everywhere. Reference projects that must
# NEVER be imported in any form (airllm/Strata/kimi-k3-in-c/
# sparse_transformers, D-12) are offenders via the general allowlist: they
# are neither stdlib nor listed here.
HEAVY = {"torch", "transformers", "peft", "trl", "datasets", "accelerate",
         "bitsandbytes", "awq", "auto_gptq", "safetensors"}
HEAVY_ALLOWED_FILES = {
    SRC / "training" / "worker.py",
    *sorted((SRC / "training" / "streamed").glob("*.py")),
}

STDLIB = set(sys.stdlib_module_names)


def _root(name: str) -> str:
    return name.split(".")[0]


def test_only_allowed_third_party_imports():
    offenders = []
    for py in SRC.rglob("*.py"):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module]
            for name in names:
                root = _root(name)
                if root in STDLIB or root in ALLOWED_THIRD_PARTY \
                        or root in HEAVY or root.startswith("sable"):
                    continue
                lineno = getattr(node, "lineno", 0)
                offenders.append(f"{py}:{lineno}: imports {name!r}")
    assert not offenders, (
        "Unexpected third-party import(s); sable is self-contained "
        "(runtime deps: pyyaml + httpx + Pillow only):\n" + "\n".join(offenders))


def test_heavy_imports_lazy_and_worker_only():
    """torch/transformers/peft/trl/datasets only ever imported lazily in
    worker.py and the wired streamed training backend (worker seam)."""
    offenders = []

    def is_heavy(node) -> bool:
        if isinstance(node, ast.Import):
            return any(_root(a.name) in HEAVY for a in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            return _root(node.module) in HEAVY
        return False

    for py in SRC.rglob("*.py"):
        tree = ast.parse(py.read_text())
        module_level = {id(n) for n in tree.body
                        if isinstance(n, (ast.Import, ast.ImportFrom))}
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)) or not is_heavy(node):
                continue
            if id(node) in module_level:
                offenders.append(f"{py}:{node.lineno}: module-level heavy import")
            elif py not in HEAVY_ALLOWED_FILES:
                offenders.append(f"{py}:{node.lineno}: heavy import outside "
                                 "worker.py / streamed spike")
    assert not offenders, "Heavy-import violations:\n" + "\n".join(offenders)


def test_no_legacy_env_var():
    """The historical external-server env var name is fully gone from source."""
    # Constructed so this file itself contains no literal of the old name.
    legacy = "UN" + "SLOTH"
    offenders = []
    for py in SRC.rglob("*.py"):
        text = py.read_text()
        for i, line in enumerate(text.splitlines(), 1):
            if legacy in line.upper() and "SABLE" not in line.upper():
                offenders.append(f"{py}:{i}")
    assert not offenders, f"Legacy env-var references: {offenders}"

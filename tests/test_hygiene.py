"""Hygiene tests: dependency/import allowlist (D-12 licensing boundary)
and static import-graph reachability (no orphan modules).
"""

from __future__ import annotations


# ------------------------------------------------------------------------
# from tests/test_license_hygiene.py
# ------------------------------------------------------------------------

import ast
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "toaster_loop"

# Third-party packages the toaster_loop source may import at all. Everything else
# (outside stdlib) is an offender. Stdlib is derived from the running
# interpreter; keep this list in sync with pyproject.toml dependencies.
#
# pyarrow is listed deliberately and is NOT a core runtime dep: it backs the
# curate hub importer's parquet reader (curate/sources.py), imported lazily
# inside the function with an honest "pip install pyarrow" degradation when
# absent — text/JSONL curation never touches it. (Apache-2.0, same family as
# the rest of the allowed ecosystem.)
ALLOWED_THIRD_PARTY = {"yaml", "httpx", "PIL", "pyarrow"}

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
                        or root in HEAVY or root.startswith("toaster_loop"):
                    continue
                lineno = getattr(node, "lineno", 0)
                offenders.append(f"{py}:{lineno}: imports {name!r}")
    assert not offenders, (
        "Unexpected third-party import(s); toaster_loop is self-contained "
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
            if legacy in line.upper() and "TOASTER_LOOP" not in line.upper():
                offenders.append(f"{py}:{i}")
    assert not offenders, f"Legacy env-var references: {offenders}"

# ------------------------------------------------------------------------
# from tests/test_no_orphans.py
# ------------------------------------------------------------------------

# Adapted from PIASystem (MIT) docs/ORPHANED_FILES.md — orphan tracking as a test, not a doc.


import ast
import pkgutil
from pathlib import Path

import toaster_loop

SRC = Path(str(toaster_loop.__file__)).resolve().parent


def _all_modules() -> dict[str, Path]:
    """Map dotted module name -> source file for the whole package.

    ``pkgutil.walk_packages`` enumerates (it only executes package __init__
    files, never leaf modules — and license hygiene guarantees those are
    cheap). The actual graph below is built from raw source with ast.
    """
    mods = {"toaster_loop": SRC / "__init__.py"}
    for info in pkgutil.walk_packages([str(SRC)], prefix="toaster_loop.",
                                      onerror=lambda m: None):
        # walk_packages yields both modules and subpackages; resolve each to a
        # file path relative to the package root.
        rel = info.name[len("toaster_loop."):].split(".")
        path = SRC.joinpath(*rel)
        file = path / "__init__.py" if info.ispkg else path.with_suffix(".py")
        if file.exists():
            mods[info.name] = file
    return mods


def _pkg_of(module: str, is_pkg: bool) -> list[str]:
    parts = module.split(".")
    return parts if is_pkg else parts[:-1]


def _mark_prefixes(name: str, imported: set[str], modules: dict[str, Path]) -> None:
    """Mark `toaster_loop.a.b` and all of its parent packages as imported."""
    parts = name.split(".")
    for i in range(1, len(parts) + 1):
        prefix = ".".join(parts[:i])
        if prefix in modules:
            imported.add(prefix)


def _build_graph(modules: dict[str, Path], is_pkg: set[str]) -> dict[str, set[str]]:
    graph: dict[str, set[str]] = {m: set() for m in modules}
    for name, path in modules.items():
        tree = ast.parse(path.read_text(), filename=str(path))
        edges = graph[name]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "toaster_loop" or alias.name.startswith("toaster_loop."):
                        _mark_prefixes(alias.name, edges, modules)
            elif isinstance(node, ast.ImportFrom):
                if node.level:  # relative: from . / ..core.x import y
                    base = _pkg_of(name, name in is_pkg)
                    # level==1 -> current package; level n strips n-1 parts.
                    if len(base) < node.level:
                        continue
                    base = base[: len(base) - (node.level - 1)]
                    if node.module:  # from .core import x -> append "core"
                        base = base + node.module.split(".")
                    pkg_target = ".".join(base)
                    if pkg_target in modules:
                        _mark_prefixes(pkg_target, edges, modules)
                    for alias in node.names:
                        # `from .pkg import name` — name may be a submodule.
                        cand = f"{pkg_target}.{alias.name}"
                        if cand in modules:
                            _mark_prefixes(cand, edges, modules)
                elif node.module and (
                    node.module == "toaster_loop"
                    or node.module.startswith("toaster_loop.")
                ):
                    _mark_prefixes(node.module, edges, modules)
                    # `from toaster_loop.x import name` — name may be a submodule.
                    for alias in node.names:
                        cand = f"{node.module}.{alias.name}"
                        if cand in modules:
                            _mark_prefixes(cand, edges, modules)
    return graph


# Production entry points:
#   - toaster_loop.cli        the `toaster_loop` console script (pyproject [project.scripts])
#   - toaster_loop.__main__   `python -m toaster_loop`
#   - toaster_loop.dashboard.server  started from cli.py (statically visible, listed
#     anyway because it is an entry point by charter)
#   - toaster_loop.training.worker   launched as a SUBPROCESS
#     (`python -m toaster_loop.training.worker`, see training/scheduler.py) — the
#     static graph can never see this edge, so it is a root by declaration.
# No importlib/__import__ usage exists in the package (grep-verified), so no
# other dynamic targets are needed here.
ROOTS = {
    "toaster_loop",
    "toaster_loop.cli",
    "toaster_loop.__main__",
    "toaster_loop.dashboard.server",
    "toaster_loop.training.worker",
}

# Honest exemptions, kept minimal — every entry needs a reason. If a module
# here becomes reachable again, remove it. (WP10 step 2 wired the streamed
# package into worker.py via streamed.runner, so the former spike exemption
# is deleted; the package must now be statically reachable from the worker
# root.)
ALLOWLIST: set[str] = set()


def test_no_orphan_modules():
    modules = _all_modules()
    is_pkg = {m for m in modules if modules[m].name == "__init__.py"}
    graph = _build_graph(modules, is_pkg)

    reachable: set[str] = set()
    stack = [r for r in ROOTS if r in modules]
    assert len(stack) == len(ROOTS), f"declared root missing from package: {ROOTS - set(modules)}"
    while stack:
        cur = stack.pop()
        if cur in reachable:
            continue
        reachable.add(cur)
        stack.extend(graph[cur] - reachable)

    orphans = sorted(set(modules) - reachable - ALLOWLIST)
    assert not orphans, (
        "Orphan module(s): not imported by any production entry point "
        "(roots: cli/__main__/dashboard.server/training.worker subprocess). "
        "Wire them in or delete them:\n  "
        + "\n  ".join(orphans)
    )

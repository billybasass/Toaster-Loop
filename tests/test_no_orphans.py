# Adapted from PIASystem (MIT) docs/ORPHANED_FILES.md — orphan tracking as a test, not a doc.
"""Import-graph hygiene: every module under sable/ must be reachable from a
production entry point.

Refactors strand helpers silently — a module whose last caller was deleted
just sits there rotting. PIASystem tracked ~20 such modules in a document that
nobody read; here it is a test instead. The graph is built statically: sources
are parsed with ``ast`` (no module execution), covering absolute imports
(``import sable.x.y`` / ``from sable.x import y``, where the imported name
may be a submodule or an attribute) and relative imports
(``from . import z`` / ``from ..core import w``).

Anything genuinely dynamic (subprocess ``python -m ...`` launches, future
importlib targets) is handled explicitly in ROOTS/ALLOWLIST below with a
comment explaining why the static graph cannot see it.
"""

from __future__ import annotations

import ast
import pkgutil
from pathlib import Path

import sable

SRC = Path(str(sable.__file__)).resolve().parent


def _all_modules() -> dict[str, Path]:
    """Map dotted module name -> source file for the whole package.

    ``pkgutil.walk_packages`` enumerates (it only executes package __init__
    files, never leaf modules — and license hygiene guarantees those are
    cheap). The actual graph below is built from raw source with ast.
    """
    mods = {"sable": SRC / "__init__.py"}
    for info in pkgutil.walk_packages([str(SRC)], prefix="sable.",
                                      onerror=lambda m: None):
        # walk_packages yields both modules and subpackages; resolve each to a
        # file path relative to the package root.
        rel = info.name[len("sable."):].split(".")
        path = SRC.joinpath(*rel)
        file = path / "__init__.py" if info.ispkg else path.with_suffix(".py")
        if file.exists():
            mods[info.name] = file
    return mods


def _pkg_of(module: str, is_pkg: bool) -> list[str]:
    parts = module.split(".")
    return parts if is_pkg else parts[:-1]


def _mark_prefixes(name: str, imported: set[str], modules: dict[str, Path]) -> None:
    """Mark `sable.a.b` and all of its parent packages as imported."""
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
                    if alias.name == "sable" or alias.name.startswith("sable."):
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
                    node.module == "sable"
                    or node.module.startswith("sable.")
                ):
                    _mark_prefixes(node.module, edges, modules)
                    # `from sable.x import name` — name may be a submodule.
                    for alias in node.names:
                        cand = f"{node.module}.{alias.name}"
                        if cand in modules:
                            _mark_prefixes(cand, edges, modules)
    return graph


# Production entry points:
#   - sable.cli        the `sable` console script (pyproject [project.scripts])
#   - sable.__main__   `python -m sable`
#   - sable.dashboard.server  started from cli.py (statically visible, listed
#     anyway because it is an entry point by charter)
#   - sable.training.worker   launched as a SUBPROCESS
#     (`python -m sable.training.worker`, see training/scheduler.py) — the
#     static graph can never see this edge, so it is a root by declaration.
# No importlib/__import__ usage exists in the package (grep-verified), so no
# other dynamic targets are needed here.
ROOTS = {
    "sable",
    "sable.cli",
    "sable.__main__",
    "sable.dashboard.server",
    "sable.training.worker",
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

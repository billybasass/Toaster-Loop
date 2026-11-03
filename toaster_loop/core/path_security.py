"""Workspace containment helpers for the agent file tools (backlog item #9).

Core validators built in
``tools/path_security.py`` (``validate_within_dir``: resolve() + relative_to,
symlink-safe; ``has_traversal_component``: cheap ``..`` pre-check), extended
here with multi-root validation and a ``contain()`` decorator that wraps tool
implementations at construction time. Enforcement happens in exactly one
place — core/agent_runner.create_runner — so the CLI runner and the dashboard
chat runner get identical containment.

Policy (resolved in create_runner, documented in config.example.yaml):
  * workspace root = config ``agent.workspace_root``; unset defaults to the
    current working directory at runner creation time; an explicit ``null``
    disables containment entirely (with a one-time startup warning).
  * TOASTER_LOOP_HOME (todo/trace state) and the system temp dir are always
    allow-listed alongside the workspace root so existing flows — todo_write,
    test suites using tmp_path — keep working.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Sequence, Union

PathLike = Union[str, "os.PathLike[str]"]

#: kwargs the built-in tools carry filesystem paths in (read_file/write_file/
#: patch/list_dir/search_files all use ``path``; ``file_path`` kept for
#: schema-compatible third-party tools).
DEFAULT_PATH_ARG_NAMES = ("path", "file_path")


def has_traversal_component(path_str: str) -> bool:
    """Return True if *path_str* contains ``..`` traversal components.

    Quick check for obvious traversal attempts before doing full resolution
    (built-in path guard).
    """
    try:
        parts = Path(path_str).parts
    except Exception:  # odd path strings must never crash a tool call
        return False
    return ".." in parts


def validate_within_dir(path: PathLike, root: PathLike) -> Optional[str]:
    """Ensure *path* resolves to a location within *root*.

    Returns an error message string if validation fails, or ``None`` if the
    path is safe. Uses ``Path.resolve()`` (strict=False semantics: nonexistent
    targets still check) so symlinks and ``..`` components are followed on
    existing path components before comparison — symlink escapes whose target
    exists are caught. Built-in guard; the
    non-raising shape matches toaster_loop tools, which return "error: ..." strings.
    """
    try:
        resolved = Path(path).expanduser().resolve()
        root_resolved = Path(root).expanduser().resolve()
        resolved.relative_to(root_resolved)
    except (ValueError, OSError) as exc:
        return f"Path escapes allowed directory: {exc}"
    return None


def within_any_dir(path: PathLike, roots: Sequence[PathLike]) -> Optional[str]:
    """Return None if *path* resolves inside ANY root of *roots*, else a reason.

    Relative paths are joined onto the first (primary) root before resolution,
    so relative tool arguments are interpreted workspace-relative rather than
    against an unrelated process cwd.
    """
    p = Path(path).expanduser()
    if not p.is_absolute() and roots:
        p = Path(roots[0]).expanduser() / p
    for root in roots:
        if validate_within_dir(p, root) is None:
            return None
    resolved = p.resolve() if p.is_absolute() else \
        (Path(roots[0]).expanduser().resolve() / p) if roots else p.resolve()
    return f"resolved to {resolved}, outside every allowed root"


def contain(roots: Iterable[PathLike],
            path_arg_names: Sequence[str] = DEFAULT_PATH_ARG_NAMES,
            deny_writes: Iterable[PathLike] = (),
            read_only: bool = False,
            ) -> Callable[[Callable[..., str]], Callable[..., str]]:
    """Decorator factory: refuse tool calls whose path args escape *roots*.

    Applied to each tool implementation at runner construction (see
    core/agent_runner.create_runner). The first root is the primary workspace
    root — relative paths resolve against it; the remaining roots are the
    standing allow-list (TOASTER_LOOP_HOME, temp dir). Violations return
    "error: blocked: ..." refusal strings in the existing tool-refusal style
    and never reach the wrapped implementation.

      * ``..`` traversal components are rejected up front (defense in depth,
        even when normalization would land inside the root);
      * everything else is checked with resolve()-based containment, so
        symlink escapes to existing outside targets are blocked;
      * *deny_writes* paths (the live config file) may still be READ through
        the allow-list but never written: TOASTER_LOOP_HOME is allow-listed so
        todo/trace state keeps working, and without this carve-out a
        prompt-injected agent could rewrite `toaster_loop.yaml` — flipping profile
        capabilities (`run_commands`, workspace widening, dashboard token)
        and escalating its own permissions.

    The deny check applies to mutating wrappers only; the caller marks
    read-only tools (read_file/list_dir/search_files) with *read_only=True*
    so reads of a denied path stay allowed.
    """
    root_list: List[PathLike] = [Path(r).expanduser() for r in roots]
    denied: List[Path] = [Path(d).expanduser() for d in deny_writes]

    def decorator(fn: Callable[..., str]) -> Callable[..., str]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> str:
            for name in path_arg_names:
                value = kwargs.get(name)
                if value is None or not isinstance(value, (str, os.PathLike)):
                    continue
                value_str = os.fspath(value)
                if has_traversal_component(value_str):
                    return (f"error: blocked: '..' path traversal component "
                            f"not allowed in {name}: {value_str!r}")
                reason = within_any_dir(value_str, root_list)
                if reason is not None:
                    return (f"error: blocked: path outside workspace "
                            f"({Path(root_list[0])}) — {name}={value_str!r} "
                            f"{reason}. Keep file operations under the "
                            "workspace root; a human can widen "
                            "`agent.workspace_root` in the toaster_loop config.")
                if not read_only and denied:
                    p = Path(value_str).expanduser()
                    if not p.is_absolute() and root_list:
                        p = Path(root_list[0]).expanduser() / p
                    resolved = p.resolve()
                    for d in denied:
                        try:
                            rd = d.resolve()
                        except OSError:
                            rd = d
                        if resolved == rd or resolved.is_relative_to(rd):
                            return (
                                f"error: blocked: {name} cannot write the "
                                f"toaster_loop config ({rd}) — agent runs must not "
                                "change profiles, capabilities, or dashboard "
                                "auth through file tools.")
            return fn(*args, **kwargs)

        return wrapper

    return decorator



"""Turn-end verification guard (inspiration pass 3, item #19).

Policy-only: this module never runs checks. It keeps a *passive ledger* of
code edits vs fresh verification evidence during one agent run, and at
finish time produces exactly one bounded follow-up nudge when the model tries
to end on "done!" right after editing code with no successful check since the
last edit.

Verification-stop policy built in:

  * prose/data extensions are exempt — a README edit must never demand a
    verification run;
  * at most one nudge per run (a second refusal is recorded, not nagged);
  * the nudge lists the changed paths (bounded) and explains what would
    count as evidence.

Evidence sources in sable (the runner loop feeds them via ``observe``):

  * the post-write checker's verdict on write_file/patch — ``✓ syntax ok``
    verifies that path, ``✗ SYNTAX ERROR`` keeps it dirty;
  * a successful ``run_command`` counts as fresh evidence for everything
    edited before it (the model actually ran something after the edits).

Ending a run with the ledger still dirty sets
``failure_reason="verification_nudge"`` so curation can down-weight
"claimed done, never checked" traces.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

# Prose/data extensions and extension-less prose filenames (case-insensitive):
# no verifiable runtime behavior — editing ONLY these never triggers a nudge.
NON_CODE_EXTENSIONS = frozenset({
    # prose
    ".md", ".markdown", ".mdx", ".rst", ".txt", ".text", ".adoc",
    ".asciidoc", ".org", ".log",
    # data/config formats with nothing to execute
    ".csv", ".tsv", ".json", ".jsonl", ".yaml", ".yml", ".toml", ".ini",
    ".cfg", ".env", ".xml", ".lock",
})
NON_CODE_FILENAMES = frozenset({
    "license", "licence", "notice", "authors", "contributors", "changelog",
    "codeowners", "readme",
})

MAX_PATHS_IN_NUDGE = 8


def is_code_path(raw: str) -> bool:
    """True when a changed path is code that deserves verification evidence."""
    try:
        p = Path(str(raw))
    except Exception:  # noqa: BLE001 - weird path objects are "not code"
        return False
    suffix = p.suffix.lower()
    if suffix in NON_CODE_EXTENSIONS:
        return False
    if not suffix and p.name.lower() in NON_CODE_FILENAMES:
        return False
    return bool(suffix) or p.name.lower() in {"makefile", "dockerfile"}


class VerificationLedger:
    """Passive per-run record of code edits vs fresh verification evidence.

    Feed every write/exec tool result through ``observe``; at finish time
    ``pending_paths()`` lists edited-code paths with no successful check
    since the edit, and ``build_nudge()`` renders the one allowed follow-up.
    """

    def __init__(self) -> None:
        # path -> True (dirty since last evidence). Insertion-ordered.
        self._dirty: Dict[str, bool] = {}
        self.nudges_sent = 0

    # -- observation ---------------------------------------------------------

    def observe(self, name: str, args: Dict, content: str) -> None:
        """Update the ledger from one executed tool result."""
        if name in ("write_file", "patch"):
            path = str(args.get("path") or "")
            if not path or not is_code_path(path):
                return
            if content.startswith("error:"):
                # The edit itself failed — nothing changed on disk.
                return
            from .tools import SYNTAX_FAIL_MARK, SYNTAX_OK_MARK
            if SYNTAX_OK_MARK in content:
                self._dirty.pop(path, None)   # fresh passing check
            elif SYNTAX_FAIL_MARK in content:
                self._dirty[path] = True      # edited AND known-broken
            else:
                # No verdict (checker skipped this extension): the edit is
                # unverified until something runs after it.
                self._dirty[path] = True
        elif name == "run_command":
            if not content.startswith("error:"):
                # A successful command after the edits is fresh evidence for
                # everything pending (the model ran, and it ran after).
                self._dirty.clear()

    # -- finish-time policy ----------------------------------------------------

    def pending_paths(self) -> List[str]:
        return sorted(self._dirty)

    def build_nudge(self, max_nudges: int = 1) -> Optional[str]:
        """The bounded follow-up, or None when nothing is owed.

        Returns None (silently) once ``max_nudges`` have been issued — the
        caller records ``failure_reason="verification_nudge"`` on the second
        unverified finish instead of nagging again.
        """
        paths = self.pending_paths()
        if not paths or self.nudges_sent >= max_nudges:
            return None
        self.nudges_sent += 1
        listed = [f"- {p}" for p in paths[:MAX_PATHS_IN_NUDGE]]
        if len(paths) > MAX_PATHS_IN_NUDGE:
            listed.append(f"- ... and {len(paths) - MAX_PATHS_IN_NUDGE} more")
        return (
            "[System: You edited code in this run but no successful "
            "verification has been observed since the last edit.\n\n"
            "Changed paths:\n" + "\n".join(listed) + "\n\n"
            "Run the relevant check now (tests, linter, or a focused script " \
            "exercising the change), read any failure, repair it, and "
            "summarize what passed. If verification is genuinely impossible "
            "here, state the concrete blocker instead of claiming the work "
            "is done.]"
        )

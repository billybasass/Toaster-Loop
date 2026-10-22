"""CLI exit-code taxonomy (WP2b).

Established codes are NOT renumbered; new ones extend the set:

  ==== ============================================================
  0    success
  1    generic runtime error (job failed, backend error, ...)
  2    usage error — argparse rejects the command line, and CLI-side
       invocation mistakes checked after parsing (mutually exclusive
       flags, bad ``--config-key`` override shape, missing required
       operand for a sub-operation). This matches argparse's own
       exit code, so the convention is coherent. `sable bench
       compare` with an unreadable/invalid operand also exits 2 (WP1).
  3    invalid configuration — the YAML config failed validation and
       the consolidated problem report (core/config.ConfigError,
       WP2a) is printed to stderr. "Refuse to guess" (pattern
       inspiration: kimi-k3-in-c; re-implemented inline, no imports
       per D-12).
  4    finished but CORRUPT artifact — the process flow completed yet
       a produced artifact failed its integrity check (truncated /
       zero-byte / bad-magic GGUF, NaN loss recorded). Distinct from
       1 so automation can tell "the run errored" from "the run
       looked fine and quietly produced garbage" (kimi-k3-in-c's
       corrupt-output code 4 idea; re-implemented inline).
  10   ``sable llama update --check``: an update is available
       (established informational code — do not reuse).
  130  interrupted (Ctrl-C), per shell convention.
  ==== ============================================================

Documented in README.md ("Exit codes") and docs/ROADMAP.md WP2 entry.
"""

from __future__ import annotations

EXIT_OK = 0
EXIT_ERROR = 1            # generic runtime failure
EXIT_USAGE = 2            # argparse / invocation error (argparse's own code)
EXIT_CONFIG_INVALID = 3   # consolidated config validation report (WP2a)
EXIT_CORRUPT_ARTIFACT = 4  # completed flow, integrity check failed (WP2b)
EXIT_UPDATE_AVAILABLE = 10  # llama update --check sentinel (established)
EXIT_INTERRUPTED = 130    # SIGINT convention

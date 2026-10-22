"""sable — local self-improving agent system.

Self-contained by design: the agent runtime, toolset, curation pipeline,
training workers and GGUF export are all built in. External systems
(any OpenAI-compatible server) are optional attach targets
reached over HTTP only; no external agent or trainer framework is imported.
The dependency boundary (docs/DECISIONS.md D-3) is enforced by
tests/test_license_hygiene.py: no external agent or trainer packages, ever.
"""

# Single source of truth: the installed distribution metadata (pyproject.toml).
# Falls back for bare-source checkouts without an install; keep in sync with
# [project] version in pyproject.toml if that ever changes.
try:
    from importlib.metadata import version as _dist_version

    __version__ = _dist_version("sable")
except Exception:  # noqa: BLE001 - not installed as a distribution
    __version__ = "0.1.0"

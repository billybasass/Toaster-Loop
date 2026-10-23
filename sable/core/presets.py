"""One-shot feature presets — the "enable everything worth enabling" path.

Why this exists
---------------
The opt-in philosophy (cardinal rule 1: every feature gates off by default)
means a fresh install has a long tail of master switches scattered through the
config, each documented in config.example.yaml but silently off. Easy training
solved the same problem for the *training loop* with a documented preset
(``training/easy.py::optimal_config_patches``). This module does it for the
*feature surface*: one reviewable list of {section, patch} bodies that turns
on everything judged safe-and-useful-by-default, through the exact same
PUT /api/config merge path — so backups, validation and per-section merge
semantics are identical to flipping each card by hand.

It is a PRESET, not a new feature: applying it writes ordinary config values.
Everything it turns on stays individually editable afterwards, same posture as
the easy-training preset ("enabling never hides knobs").

Deliberate exclusions (the "optimal" part; mirror training/easy.py's reasoning):
  * ``easy_training.enabled`` — the loop is a *decision*, not a switch: it
    needs a bound profile, and its own enable flow already applies its preset.
  * ``janitor.enabled`` — makes the cron entrypoint archive (and operational
    files age-delete) automatically; opt-in stays with the human.
  * ``serve.calibration: auto`` — only when a calibration profile exists for
    this machine (``sable calibrate`` hasn't been run => nothing to apply and
    launch behavior would change for no benefit).
  * ``tools.lsp.enabled`` — only when a language server is actually on PATH
    (otherwise it is pure probe overhead per write).
  * ``training.backend: streamed`` / ``rank_scaling`` — GPU effects unmeasured;
    cardinal rule 2 forbids shipping them as "optimal".
  * Anything with a real recurring GPU bill beyond the gate machinery itself
    (``rft`` IS included: runs only when explicitly queued, VRAM-gated).

Stdlib only, like core/. Functions never raise on weird config — a section
that isn't a dict simply diffs as "unset" and merges server-side.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ------------------------------------------------------------------- preset

#: Master switches turned on by "enable all", keyed by config section.
#: Values are the literal patch bodies sent to PUT /api/config; kept to
#: ``enabled`` keys only — tuned knobs are never overwritten here.
ENABLE_ALL_SWITCHES: Dict[str, Dict[str, Any]] = {
    # Knowledge & retrieval
    "skills": {"enabled": True},          # core/skills.py agent-authored recipes
    "playbooks": {"enabled": True},       # core/playbooks.py chain distillation
    "indexing": {"enabled": True},        # code intelligence (needs an embed model or lexical)
    # Learning pipeline quality machinery (each is gated further by its own
    # use — RFT/evolve/shadow only do work when a job is actually queued)
    "rft": {"enabled": True},             # training/rejection.py (explicit queue only)
    "evolution": {"enabled": True},       # proposals only; apply stays gated
    "goldens_mine": {"enabled": True},
    "validation": {"shadow_enabled": True},
    # Serving resilience
    "routing": {"enabled": True},         # per-run endpoint choice; explicit binds win
    "serve": {"watchdog_enabled": True},  # frozen-child detect/restart (Linux, opt-in kill rules)
    # Tool quality
    # NOTE: tools.lsp.enabled is conditional — see lsp_available().
}


def lsp_available(path: Optional[str] = None) -> bool:
    """True if any default LSP server binary is on PATH.

    Mirrors the default server map in core/tools.py (pyright for .py,
    typescript-language-server for TS/JS). Cheap: shutil.which per candidate.
    """
    for exe in ("pyright-langserver", "typescript-language-server"):
        if shutil.which(exe, path=path) is not None:
            return True
    return False


def calibration_ready(home: Optional[str] = None) -> bool:
    """True if at least one calibration profile is stored for this machine.

    serve.calibration:auto only means something once `sable calibrate` has
    produced (and adopted) a profile; enabling it blind changes launch code
    for zero measured benefit.
    """
    try:
        from ..serving.calibrate import calibrate_dir, machine_id

        d = calibrate_dir(Path(home) if home is not None else None)
        if not d.is_dir():
            return False
        mid = machine_id()
        return any(p.name.startswith(str(mid)) for p in d.glob("*.json"))
    except Exception:
        return False


def _section(cfg: Any, name: str) -> Dict[str, Any]:
    sec = getattr(cfg, name, None) if not isinstance(cfg, dict) \
        else cfg.get(name)
    return sec if isinstance(sec, dict) else {}


def enable_all_patches(cfg: Any, *,
                       home: Optional[str] = None) -> List[Dict[str, Any]]:
    """Ordered [{section, patch}] bodies for PUT /api/config.

    One patch per section (the config endpoint deep-merges INSIDE the named
    section), so unrelated keys the user already set are preserved.
    Conditional switches (LSP, calibration) are read from the LIVE config
    object so a preview and an apply in the same session agree.
    """
    patches: List[Dict[str, Any]] = []
    for section, patch in ENABLE_ALL_SWITCHES.items():
        patches.append({"section": section, "patch": dict(patch)})
    # Conditional: LSP only when a server could actually answer. The lsp
    # section merges as a whole dict server-side, so carry existing keys.
    if lsp_available():
        cur = _section(cfg, "tools").get("lsp")
        cur = cur if isinstance(cur, dict) else {}
        patches.append({"section": "tools",
                        "patch": {"lsp": {**cur, "enabled": True}}})
    # Conditional: calibration only once a profile exists for this machine.
    if calibration_ready(home):
        patches.append({"section": "serve", "patch": {"calibration": "auto"}})
    # Note: `serve` may appear twice (watchdog + calibration). The section
    # merge in h_config_put is idempotent per key, and both patches touch
    # disjoint keys, so ordering is irrelevant; sending them separately keeps
    # each conditional independent.
    return patches


def enable_all_diff(cfg: Any, *,
                    home: Optional[str] = None) -> List[Dict[str, Any]]:
    """What enable-all WOULD change, as [{section, key, from, to}].

    Only keys whose effective value actually differs (same shape as the
    easy-training enable preview dialog consumes). Nested dict keys are
    reported as "lsp.enabled".
    """
    diff: List[Dict[str, Any]] = []
    for body in enable_all_patches(cfg, home=home):
        sec = _section(cfg, body["section"])
        for key, to in body["patch"].items():
            if isinstance(to, dict):
                sub = sec.get(key)
                sub = sub if isinstance(sub, dict) else {}
                for k2, v2 in to.items():
                    if sub.get(k2) != v2:
                        diff.append({"section": body["section"],
                                     "key": f"{key}.{k2}",
                                     "from": sub.get(k2), "to": v2})
            elif sec.get(key) != to:
                diff.append({"section": body["section"], "key": key,
                             "from": sec.get(key), "to": to})
    return diff


def enable_all_summary() -> Tuple[str, ...]:
    """Human-readable one-liners of what is covered (for the dialog header)."""
    return (
        "knowledge: skills, playbooks, codebase indexing",
        "learning: RFT, prompt evolution (propose-only), golden mining, "
        "shadow-baseline promotion checks",
        "serving: capability-tier routing, engine watchdog",
        "tools: LSP diagnostics (only when a language server is on PATH)",
        "autotune: serve calibration (only when a profile was measured)",
        "NOT touched: easy_training loop, janitor auto-archive, unmeasured "
        "GPU knobs",
    )

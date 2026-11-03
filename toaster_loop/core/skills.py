"""Agent-authored persistent skills — procedural knowledge that outlives a run.

Why this exists (and playbooks don't cover it)
----------------------------------------------
``core/playbooks.py`` mines *recurring successful tool chains* into template
recipes; arguments are stripped entirely, so a playbook structurally cannot
carry the knowledge that actually decides GUI/app work: which screenshot path
works on this machine, that a URL bar sits at y=435, that XTest press+release
pairs never form a Ctrl+A chord, that ``steam://`` only routes on a warm
client. That knowledge is produced deliberately — often while FAILING, so
there is no successful trace to mine — and belongs to a specific program or
machine. Skills are its entry point: small markdown files the agent writes
itself under ``TOASTER_LOOP_HOME/skills/``, with an executable-script directory
beside each one.

Design goals (the "grow without jamming context" constraint)
-------------------------------------------------------------
* Standing prompt cost is ONE INDEX LINE per skill — ``name: description``
  under a hard total cap, same whole-entry-omission discipline as playbooks'
  ``prompt_section``. Full text is read on demand via the ``read_skill``
  tool only when a matching task lands.
* The distiller is the agent, deliberately, at the end of a session — NOT
  automatic trace compression (which loses exactly the failure details that
  make a recipe correct). No judge gate: fabrication risk lives in auto-
  mining; here a human can open the file and edit it (same posture as the
  ``run_commands`` capability itself).
* Skills are runtime knowledge, not training data. Nothing here exports
  ShareGPT rows; curation over real traces stays the only training path.

Storage layout
--------------
``TOASTER_LOOP_HOME/skills/``
  ``<slug>.md``   — frontmatter (name/description/updated) + markdown body
  ``<slug>/``     — optional companion dir for executable scripts the agent
                    bundles with a skill (e.g. an X11 input helper); the
                    ``save_skill`` tool result tells the agent this path so
                    ``run_command`` can drop scripts there.

Stdlib only, like the rest of core/. Tool-facing functions never raise —
errors come back as "error: ..." strings the model can recover from.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger("toaster_loop.core.skills")

# ---------------------------------------------------------------- constants

#: skill name: filesystem-safe slug (also the markdown filename stem).
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")

DESCRIPTION_MAX_CHARS = 160     # one index line per skill, kept short
CONTENT_MAX_CHARS = 24_000      # a skill body is a recipe, not a book
LIST_MAX_SKILLS = 200

#: injection caps (mirror the playbooks knobs; skills default OFF too).
INDEX_TOTAL_CAP_CHARS_DEFAULT = 1500

_FRONTMATTER_RE = re.compile(
    r"\A---\n(.*?)\n---\n?(.*)\Z", re.DOTALL)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------ config

def skills_config(cfg: Any) -> dict:
    sec = getattr(cfg, "skills", None)
    return dict(sec) if isinstance(sec, dict) else {}


def skills_dir() -> Path:
    """The skills root (created on demand by writers, never by readers)."""
    from ..store.paths import resolve_toaster_loop_home

    return resolve_toaster_loop_home() / "skills"


# ------------------------------------------------------------------- model

@dataclass
class Skill:
    name: str
    description: str
    content: str
    path: str
    updated_ts: str

    def one_liner(self) -> str:
        return f"{self.name}: {self.description}"


def _unquote(val: str) -> str:
    """Strip one layer of matching surrounding quotes (hand-edited files
    routinely quote descriptions that contain colons)."""
    if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
        return val[1:-1]
    return val


def _parse(path: Path) -> Optional[Skill]:
    """Read one skill file; None on any malformed/unreadable content."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    m = _FRONTMATTER_RE.match(raw)
    if not m:
        return None
    fields: dict = {}
    for line in m.group(1).splitlines():
        key, _, val = line.partition(":")
        if _:
            fields[key.strip()] = _unquote(val.strip())
    name = fields.get("name") or path.stem
    if not NAME_RE.match(name):
        return None
    return Skill(name=name,
                 description=fields.get("description", ""),
                 content=m.group(2).lstrip("\n"),
                 path=str(path),
                 updated_ts=fields.get("updated", ""))


def load_skill(name: str) -> Optional[Skill]:
    if not NAME_RE.match(name or ""):
        return None
    return _parse(skills_dir() / f"{name}.md")


def all_skills() -> List[Skill]:
    root = skills_dir()
    if not root.is_dir():
        return []
    out: List[Skill] = []
    for p in sorted(root.glob("*.md"))[:LIST_MAX_SKILLS * 4]:
        s = _parse(p)
        if s is not None:
            out.append(s)
    out.sort(key=lambda s: s.name)
    return out[:LIST_MAX_SKILLS]


# ------------------------------------------------------------------ writer

def save_skill(name: str, description: str, content: str) -> dict:
    """Create/overwrite a skill. Returns {ok, path?, scripts_dir?, error?}."""
    from .redact import redact_text

    name = (name or "").strip().lower()
    if not NAME_RE.match(name):
        return {"ok": False,
                "error": "invalid name: use a lowercase slug "
                         "(letters/digits/._- , starting alnum, <=64 chars)"}
    description = " ".join((description or "").split())
    if not description:
        return {"ok": False, "error": "description required (one line — "
                                      "it is what shows in the skill index)"}
    if len(description) > DESCRIPTION_MAX_CHARS:
        return {"ok": False,
                "error": f"description too long ({len(description)} > "
                         f"{DESCRIPTION_MAX_CHARS} chars)"}
    content = (content or "").strip()
    if not content:
        return {"ok": False, "error": "content required (the recipe body)"}
    if len(content) > CONTENT_MAX_CHARS:
        return {"ok": False,
                "error": f"content too long ({len(content)} > "
                         f"{CONTENT_MAX_CHARS} chars) — split into separate "
                         "skills or keep the detail in a bundled script"}

    # Defense in depth (same posture as playbooks distillation): skills land
    # on disk next to the experience buffer and may be read back into many
    # future prompts; scrub known secret shapes on the way in.
    description = redact_text(description)
    content = redact_text(content)

    root = skills_dir()
    try:
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{name}.md"
        # Atomic write (same discipline as todo state): a crash mid-write
        # must never leave malformed frontmatter behind — _parse skips such
        # files forever, so the skill would silently vanish from the index.
        tmp = path.with_suffix(".md.tmp")
        tmp.write_text(
            f"---\nname: {name}\ndescription: {description}\n"
            f"updated: {_utcnow()}\n---\n\n{content}\n",
            encoding="utf-8")
        os.replace(tmp, path)
        scripts = root / name
        scripts.mkdir(exist_ok=True)
    except OSError as e:
        return {"ok": False, "error": f"could not write skill: {e}"}
    return {"ok": True, "path": str(path), "scripts_dir": str(root / name)}


def delete_skill(name: str) -> dict:
    """Delete a skill's markdown file. Returns {ok, name?, scripts_dir?, error?}.

    The bundled scripts directory is intentionally LEFT on disk (it may hold
    artifacts the human wants); callers get its path so they can report it."""
    name = (name or "").strip().lower()
    if not NAME_RE.match(name):
        return {"ok": False, "error": "invalid skill name"}
    root = skills_dir()
    path = root / f"{name}.md"
    scripts = root / name
    if not path.is_file():
        return {"ok": False, "error": f"no skill named {name!r}"}
    try:
        path.unlink()
    except OSError as e:
        return {"ok": False, "error": f"could not delete skill: {e}"}
    return {"ok": True, "name": name,
            "scripts_dir": str(scripts) if scripts.is_dir() else None}


# --------------------------------------------------------------- injection

def index_section(cfg: Any) -> str:
    """Render the capped '## Skills' index for a run's system prompt.

    Master switch (``skills.enabled``, default false) off => '' and ZERO
    filesystem access beyond one directory check, so prompts stay identical
    to pre-feature behavior. Only names + descriptions are injected; bodies
    are pulled on demand with read_skill. Entries that do not fit the total
    cap are omitted whole, never sliced.
    """
    scfg = skills_config(cfg)
    if not scfg.get("enabled", False):
        return ""
    try:
        skills = all_skills()
    except Exception as e:  # never break a run over the index
        logger.warning("skills index skipped: %s", e)
        return ""
    if not skills:
        return ""
    total_cap = int(scfg.get("index_total_cap_chars",
                             INDEX_TOTAL_CAP_CHARS_DEFAULT))
    # Most-recently-updated first: when the cap bites, stale recipes are what
    # drop out of the prompt — not whatever happens to sort earliest.
    # (updated is an ISO-8601 string; lexicographic order == chronological.)
    skills = sorted(skills, key=lambda s: s.updated_ts, reverse=True)
    lines: List[str] = []
    used = 0
    for s in skills:
        line = f"- {s.one_liner()}"
        if used + len(line) + 1 > total_cap:
            continue  # whole-entry omission
        lines.append(line)
        used += len(line) + 1
    if not lines:
        return ""
    return ("## Skills\n"
            "Persistent procedures learned on this machine "
            "(read_skill(name) for the full recipe; save_skill to add or "
            "update one after you learn something durable):\n"
            + "\n".join(lines))

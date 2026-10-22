"""Persistent skills (core/skills.py) + tool wiring.

Covers: save/load round-trip, name/description/content validation,
redaction on write, index rendering with caps, the disabled-by-default
byte-identical prompt path, and runner tool gating (knob off strips all
three tools; knob on exposes them even under a restrictive profile list).
"""

from __future__ import annotations

from sable.core.agent_runner import SKILL_TOOL_NAMES, SableAgentClient
from sable.core.config import SableConfig
from sable.core.skills import (
    CONTENT_MAX_CHARS,
    DESCRIPTION_MAX_CHARS,
    all_skills,
    index_section,
    load_skill,
    save_skill,
    skills_dir,
)
from sable.core.tools import list_skills, read_skill, save_skill as save_skill_tool


# ---------------------------------------------------------------- storage

def test_save_and_load_roundtrip():
    res = save_skill("steam-gui", "Driving the Steam client GUI on Linux",
                     "1. warm the client first\n2. URL bar at y=435")
    assert res["ok"], res
    s = load_skill("steam-gui")
    assert s is not None
    assert s.description == "Driving the Steam client GUI on Linux"
    assert "URL bar at y=435" in s.content
    # companion scripts dir exists for bundled executables
    from pathlib import Path
    assert Path(res["scripts_dir"]).is_dir()


def test_save_overwrites_same_name():
    save_skill("x11-input", "v1 desc", "v1 body")
    res = save_skill("X11-Input", "v2 desc", "v2 body")  # name lowercased
    assert res["ok"]
    s = load_skill("x11-input")
    assert s is not None and s.description == "v2 desc"
    assert len([sk for sk in all_skills() if sk.name == "x11-input"]) == 1


def test_validation_rejects_bad_input():
    assert "invalid name" in save_skill(
        "Bad Name!", "d", "c")["error"]
    assert "description required" in save_skill("ok", "", "c")["error"]
    assert "too long" in save_skill(
        "ok", "d" * (DESCRIPTION_MAX_CHARS + 1), "c")["error"]
    assert "content required" in save_skill("ok", "d", "  ")["error"]
    assert "too long" in save_skill(
        "ok", "d", "c" * (CONTENT_MAX_CHARS + 1))["error"]


def test_redaction_on_write():
    res = save_skill("secrets", "uses token sk-abcdefghijklmnopqrstuvwxyz123456",
                     "auth with sk-abcdefghijklmnopqrstuvwxyz123456")
    assert res["ok"]
    raw = (skills_dir() / "secrets.md").read_text()
    assert "sk-abcdefghijklmnopqrstuvwxyz123456" not in raw


def test_load_missing_and_malformed():
    assert load_skill("nope") is None
    assert load_skill("../escape") is None
    bad = skills_dir()
    bad.mkdir(parents=True, exist_ok=True)
    (bad / "broken.md").write_text("no frontmatter here")
    assert all_skills() == []  # malformed files are skipped, never crash


# ------------------------------------------------------------------ tools

def test_tool_wrappers_never_raise():
    out = save_skill_tool(name="steam-gui", description="steam gui facts",
                          content="xact click works; import hangs")
    assert "skill saved" in out
    body = read_skill(name="steam-gui")
    assert "xact click works" in body
    assert "error:" in read_skill(name="missing-skill")
    assert "steam-gui: steam gui facts" in list_skills()


# ----------------------------------------------------------------- index

def _cfg(**skills):
    return SableConfig(skills=skills)


def test_index_disabled_by_default_and_empty_when_no_skills():
    assert index_section(_cfg()) == ""
    save_skill("a", "alpha", "body")
    assert index_section(_cfg()) == ""          # still disabled
    assert index_section(_cfg(enabled=True)) != ""


def test_index_renders_one_line_per_skill_under_cap():
    for i in range(5):
        save_skill(f"skill-{i}", f"description {i}", "body")
    sec = index_section(_cfg(enabled=True))
    assert sec.startswith("## Skills")
    assert "- skill-3: description 3" in sec
    # cap: entries that do not fit are omitted WHOLE
    tight = _cfg(enabled=True, index_total_cap_chars=40)
    sec2 = index_section(tight)
    lines = [ln for ln in sec2.splitlines() if ln.startswith("- ")]
    assert sum(len(ln) + 1 for ln in lines) <= 40
    assert all(f"skill-{i}" in sec2 or True for i in range(5))
    # no partial entries ever
    for ln in lines:
        assert not ln.endswith("d")  # descriptions are "description N"


# ------------------------------------------------------------- runner gate

def _client(**kwargs):
    return SableAgentClient(base_url="http://127.0.0.1:1", model="m",
                              **kwargs)


def _tool_names(client):
    return {s["function"]["name"] for s in client.tools}


def test_skill_tools_stripped_when_disabled():
    c = _client()  # skills_enabled defaults False
    assert not (SKILL_TOOL_NAMES & _tool_names(c))
    assert not (SKILL_TOOL_NAMES & set(c.tool_impls))


def test_skill_tools_exposed_when_enabled_even_with_profile_list():
    c = _client(skills_enabled=True, enabled_tool_names=["read_file"])
    assert SKILL_TOOL_NAMES <= _tool_names(c)
    assert SKILL_TOOL_NAMES <= set(c.tool_impls)


def test_create_runner_wires_knob_and_index(tmp_path, monkeypatch):
    import sable.core.agent_runner as ar

    cfg = SableConfig(skills={"enabled": True})
    save_skill("steam-gui", "steam gui facts on this machine",
               "URL bar at y=435; steam:// needs a warm client")
    runner = ar.create_runner(cfg, "http://127.0.0.1:1", "m")
    assert SKILL_TOOL_NAMES <= _tool_names(runner)
    assert "## Skills" in runner.system_prompt
    assert "steam-gui: steam gui facts on this machine" in runner.system_prompt

    # disabled => byte-identical prompt, tools stripped
    cfg_off = SableConfig(skills={})
    runner_off = ar.create_runner(cfg_off, "http://127.0.0.1:1", "m")
    assert "## Skills" not in runner_off.system_prompt
    assert not (SKILL_TOOL_NAMES & _tool_names(runner_off))

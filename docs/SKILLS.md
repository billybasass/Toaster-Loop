# Skills — teaching toaster-loop agents programs without MCP

The problem this solves: an agent asked to drive Steam, Blender or Unreal
discovers a pile of environment facts the hard way (which screenshot path
survives, where a URL bar sits, that `steam://` only routes on a warm
client, that XTest press+release pairs never form a Ctrl+A chord) — and
then loses every one of them when the run ends. Skills give that knowledge
a permanent home **outside the model's context**, so it is learned once.

## What it is

Enable in config:

```yaml
skills:
  enabled: true
```

Four tools appear for agents:

| Tool | Purpose |
|---|---|
| `save_skill(name, description, content)` | Persist a recipe (overwrites same name). Also reports a per-skill **scripts dir** under `<TOASTER_LOOP_HOME>/skills/<name>/` where the agent drops executable helpers via `run_command`. |
| `read_skill(name)` | Pull the full markdown body on demand. |
| `list_skills()` | Names + one-line descriptions. |
| `delete_skill(name)` | Remove a skill that turned out wrong (the bundled scripts dir stays). |

Storage is plain markdown with frontmatter under `<TOASTER_LOOP_HOME>/skills/`:

```
---
name: steam-gui
description: Driving the Steam Linux client (input tooling, URL routing, gotchas)
updated: 2026-10-03T23:00:00+00:00
---

1. Warm the client first: `steam` with no args; steam:// URLs are ignored
   by a cold client.
2. Screenshots: use take_screenshot — ImageMagick `import` hangs (xauth).
3. Input: xdotool is NOT installed and there is no sudo; compile an XTest
   helper once into the scripts dir. Ctrl+A needs a real chord
   (keypress → keypress → keyrelease → keyrelease), never press+release
   pairs.
...
```

You can edit or delete these files by hand at any time, or from the
dashboard (Settings → Skills card lists every skill with view / edit /
delete over `/api/skills`) — that is the review mechanism. Agent and
dashboard writes go through the same path: validation, secret redaction,
atomic write.

## Context economics (the point)

With `skills.enabled` on, a run's system prompt gains only a **`## Skills`
index**: one `name: description` line per skill under a hard cap
(`index_total_cap_chars`, default 1500; entries are ordered most-recently
updated first and ones that don't fit are omitted whole — so when the cap
bites, stale recipes drop out, not fresh ones). Bodies are read with
`read_skill` only when the task matches. Twenty skills ≈ forty tokens of
standing cost. With the knob off (default) the tools are stripped from
every runner and prompts are byte-identical to before the feature.

## Skills vs playbooks

| | Playbooks (`core/playbooks.py`) | Skills (`core/skills.py`) |
|---|---|---|
| Author | auto-mined from ≥3 recurring **successful** traces | the agent, deliberately, usually at end of session |
| Carries | tool-name chains (arguments stripped) | arguments, coordinates, environment facts, failure gotchas |
| Gate | judge gate on promote | none — human edits the file; same posture as `run_commands` |
| Artifact | prose recipe | markdown + bundled **executable scripts** |

They complement each other: playbooks capture *how the agent works*;
skills capture *how a program/machine behaves*.

## Guidance for agents (worth putting in app-profile system prompts)

1. **Headless first.** Blender: `blender --background --python script.py`
   covers most work — never drive it by clicking. Steam and Unreal Editor
   are GUI-only from the outside; budget for screenshots + input tooling
   and record what works as a skill immediately.
2. **Probe once, save once.** First session on a machine: check display
   server, working screenshot path, presence of `xdotool`/`wmctrl`, sudo.
   Save it as a `machine-env` skill before touching any app.
3. **Distill at the end, yourself.** Don't trust auto-compression with the
   failure details — "Ctrl+A never fires through press+release pairs" is
   exactly the line that saves the next run. Update the skill in the same
   session when a recipe just failed.
4. **Scripts over prose.** Anything deterministic belongs in the scripts
   dir (one `run_command` next time, fails loudly); markdown is for what
   to look at and when layouts changed.

## Limits (deliberate)

- Skills are runtime knowledge, never training data — no ShareGPT export;
  curation over real traces remains the only training path.
- Writes stay inside `<TOASTER_LOOP_HOME>/skills/` (same trust level as
  `todo_write`); scripts there are only executed if the agent also has the
  `run_commands` capability.
- No MCP servers, no per-app plugins: this is the "grow without jamming
  context" path.

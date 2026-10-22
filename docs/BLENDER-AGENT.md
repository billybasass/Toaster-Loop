# Blender Agent — connecting sable to Blender

This is the worked example for per-agent capabilities + image support: an
sable agent that drives **Blender** (a headless 3D program), looks at a
reference image the user uploads, and checks its own work by looking at
renders/screenshots.

## What the agent needs

| Capability | Tool it grants | Why Blender work needs it |
|---|---|---|
| `run_commands` | `run_command` | run `blender --background --python ...` (and any helper CLI) |
| `vision` | `view_image` | look at reference images and rendered PNGs |
| `screenshots` | `take_screenshot` | watch the live Blender window while iterating |
| `internet` *(optional)* | `web_fetch` | pull reference pictures from the web |

All four are **off by default** — enable them only for this profile.

## 1. Install Blender where the agent can find it

Blender is not on PATH on this machine (portable copy in
`~/Downloads/blender-5.2.2-linux-x64/`). Make a stable path the agent can
use, e.g.:

```bash
mkdir -p ~/tools && ln -s ~/Downloads/blender-5.2.2-linux-x64/blender ~/tools/blender
# then either add ~/tools to PATH, or always call ~/tools/blender
```

## 2. Create the profile (Profiles panel or API)

Dashboard → **Profiles → New profile**, name it `blender`, tick the four
capability checkboxes, and use a system prompt like:

```text
You are a Blender sculptor agent. You create 3D models by writing Python
scripts for Blender's headless mode and running them with run_command:

  ~/tools/blender --background --factory-startup --python /tmp/step.py

Workflow per iteration:
1. Look at the reference image (view_image) if the user attached one.
2. Write a small Python script that builds/modifies the scene in
   bpy, then saves the .blend AND renders a PNG from a good camera
   angle (bpy.ops.render.render(write_still=True) with a tempfilepath).
3. view_image the rendered PNG and compare it against the reference.
4. Repeat until it looks right; deliver the .blend path and a summary.

Keep scripts idempotent, prefer editing your previous script over
rewriting from scratch, and never delete user files.
```

Or via API (use the URL `sable dashboard` printed; if you want a fixed port
set `dashboard.port` in the config or `SABLE_DASHBOARD_PORT`):

```bash
curl -s -X PUT http://127.0.0.1:<port>/api/profiles/blender \
  -H 'Content-Type: application/json' \
  -d '{
    "system_prompt": "You are a Blender sculptor agent... (see above)",
    "tools": ["read_file", "write_file", "patch", "list_dir"],
    "capabilities": {"internet": true, "run_commands": true,
                     "vision": true, "screenshots": true}
  }'
```

Give it a workspace (`workspace_root`) so its `.blend` files and scripts
land in one project directory.

## 3. Serve a vision model

Image *analysis* only works when the served model is multimodal:

1. Get a GGUF + its projector, e.g. Qwen2.5-VL / Qwen3-VL from Hugging Face.
2. Settings → run settings → **mmproj_path** = `.../mmproj-*.gguf`
   (sable passes `--mmproj` to llama-server).
3. `agent.vision` can stay `auto` — with mmproj configured, sable knows
   the endpoint sees images and expands image attachments automatically.

Without a vision model the profile still works headlessly: `view_image`
degrades to telling the agent the file path only.

## 4. Use it

Chat (agent mode) → pick profile **blender** → attach the reference picture
with the 🖼 button → e.g.:

> Model this chair from my photo as a .blend in ~/projects/chair, low-poly,
> then render 3 angles and check them against the reference.

The agent will: `view_image` the upload → write `build.py` (bpy script) →
`run_command` Blender headless → `view_image` the render → iterate. With a
live Blender GUI open instead, `take_screenshot` (grim) lets it watch the
viewport directly.

## Minimal bpy iteration loop (what the agent generates)

```python
# /tmp/step.py — run: blender --background --factory-startup --python step.py
import bpy, os
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.mesh.primitive_cube_add(size=1)          # ... your modeling here
cam_data = bpy.data.cameras.new("Cam"); cam = bpy.data.objects.new("Cam", cam_data)
cam.location = (4, -4, 3); 
import mathutils
cam.rotation_euler = (cam.to_track_quat('-Z', 'Y')).to_euler()
bpy.context.scene.camera = cam
light = bpy.data.lights.new("Sun", 'SUN'); lo = bpy.data.objects.new("Sun", light)
lo.rotation_euler = (0.8, 0.2, 0.5); bpy.context.collection.objects.link(lo)
out = os.path.expanduser("~/projects/chair")
os.makedirs(out, exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=f"{out}/model.blend")
bpy.context.scene.render.filepath = f"{out}/render.png"
bpy.ops.render.render(write_still=True)          # the agent views this next
```

## Safety notes

- `run_commands` runs with **your** user privileges and is not sandboxed by
  workspace containment — that is what the capability means. Keep it on
  trusted profiles only (the Profiles panel warns about exactly this).
- Trace recording keeps text stubs (`[image attached: path]`), so renders
  never bloat the experience buffer; the images themselves stay on disk in
  your workspace.

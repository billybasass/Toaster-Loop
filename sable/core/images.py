"""Image support for the built-in agent (vision input).

Two jobs:

1. Encoding local images to OpenAI-style ``image_url`` content parts
   (base64 data URLs) so multimodal models served by llama-server can SEE
   them — reference pictures, Blender viewport renders, screenshots. MIME
   magic-byte sniffing follows standard practice;
   ``agent/image_routing.py`` playbook (platforms lie about content-type;
   llama.cpp sniffs too and rejects mismatches).

2. Reactive multimodal expansion in the chat loop: tool results stay plain
   text stubs (``[image attached: path]`` markers, established semantics) and are
   expanded into native image parts ON THE OUTBOUND REQUEST only, when the
   backend advertises vision support. Text-only backends keep receiving the
   stub — nothing breaks — and the durable trace keeps lean text markers
   instead of multi-megabyte base64 blobs.

Vision detection: llama-server exposes ``mmproj`` in /model metadata when a
vision projector is loaded (``run_settings.mmproj_path``). When unknown we
probe once per endpoint and cache; config ``agent.vision`` forces on/off.
"""

from __future__ import annotations

import base64
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

#: Magic-byte signatures ((offset, bytes), ...) conjunctions — reference-shaped.
_MAGIC: Tuple[Tuple[Tuple[Tuple[int, bytes], ...], str], ...] = (
    (((0, b"\x89PNG\r\n\x1a\n"),), "image/png"),
    (((0, b"\xff\xd8\xff"),), "image/jpeg"),
    (((0, b"GIF87a"),), "image/gif"),
    (((0, b"GIF89a"),), "image/gif"),
    (((0, b"RIFF"), (8, b"WEBP")), "image/webp"),
    (((0, b"BM"),), "image/bmp"),
    (((0, b"II*\x00"),), "image/tiff"),
    (((0, b"MM\x00*"),), "image/tiff"),
)

#: Accepted image extensions (view_image / screenshot writes).
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff"}

#: Raw-bytes cap for one attached image (base64 inflates ~33%; llama-side
#: bodies stay sane). Oversized images are refused with a clear message.
MAX_IMAGE_BYTES = 8_000_000

#: Marker emitted in tool results; the expansion step keys off exactly this.
IMAGE_ATTACH_PREFIX = "[image attached:"
_IMAGE_MARKER_RE = re.compile(
    r"\[image attached: ([^\]\n]+)\](?: \(see image\))?")


def sniff_image_mime(raw: bytes) -> Optional[str]:
    """Detect image MIME from magic bytes; None when not a known image."""
    for conj, mime in _MAGIC:
        if all(raw[offset:offset + len(sig)] == sig
               for offset, sig in conj):
            return mime
    return None


def encode_image_file(path: str) -> Tuple[Optional[str], Optional[str]]:
    """Encode one local image as (data_url, error). Exactly one is returned.

    Reads raw bytes, sniffs MIME from magic bytes (extension only as a tie
    break for formats without distinct signatures), and base64-encodes into
    a ``data:<mime>;base64,...`` URL ready for an image_url content part.
    """
    p = Path(path).expanduser()
    if not p.is_file():
        return None, f"error: not a file: {path}"
    try:
        raw = p.read_bytes()
    except OSError as e:
        return None, f"error reading {path}: {e}"
    if len(raw) > MAX_IMAGE_BYTES:
        return None, (f"error: {path} is {len(raw)} bytes "
                      f"(cap {MAX_IMAGE_BYTES}); downscale it first")
    mime = sniff_image_mime(raw)
    if mime is None:
        return None, f"error: {path} is not a recognized image (png/jpeg/gif/webp/bmp/tiff)"
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}", None


def make_image_part(data_url: str) -> dict:
    """OpenAI-shaped image content part."""
    return {"type": "image_url", "image_url": {"url": data_url}}


def find_image_markers(content: str) -> List[str]:
    """Image paths referenced by attach markers in a tool result."""
    return [m.strip() for m in _IMAGE_MARKER_RE.findall(content)]


def attach_marker(path: str) -> str:
    """The stub a vision-capable tool returns instead of raw bytes.

    Kept short and single-line so guardrails/spill/compression all treat it
    like ordinary text; the runner expands it per-request when the served
    model has vision.
    """
    return f"{IMAGE_ATTACH_PREFIX} {path}] (see image)"


def expand_message_images(msg: Dict[str, Any],
                          max_images_per_request: int = 4) -> Tuple[dict, int]:
    """Rewrite one message's attach markers into native multimodal content.

    Returns (message, images_attached). Without markers this is identity —
    text-only backends and marker-free turns are byte-identical to before.
    Images that fail to encode degrade to their plain-text marker (the model
    still sees the path), so a bad file never breaks the request.
    """
    content = msg.get("content")
    if not isinstance(content, str) or IMAGE_ATTACH_PREFIX not in content:
        return msg, 0
    paths = find_image_markers(content)
    if not paths:
        return msg, 0
    parts: List[dict] = [{"type": "text", "text": content}]
    attached = 0
    for p in paths[:max_images_per_request]:
        data_url, _err = encode_image_file(p)
        if data_url is not None:
            parts.append(make_image_part(data_url))
            attached += 1
    if attached == 0:
        return msg, 0
    return {**msg, "content": parts}, attached


def save_data_url(data_url: str, dest_dir: Path) -> Tuple[Optional[str], Optional[str]]:
    """Persist one base64 data URL (from the browser) as an image file.

    Returns (path, error). Magic-byte sniffing validates the payload — a
    non-image or oversize upload is refused before anything is written.
    """
    import re as _re

    m = _re.match(r"^data:(image/[a-z0-9.+-]+);base64,(.+)$",
                  data_url or "", _re.DOTALL | _re.IGNORECASE)
    if not m:
        return None, "error: expected a base64 image data URL"
    try:
        raw = base64.b64decode(m.group(2), validate=True)
    except Exception:  # noqa: BLE001 - decode errors are user input, not bugs
        return None, "error: image data could not be decoded"
    if len(raw) > MAX_IMAGE_BYTES:
        return None, (f"error: image is {len(raw)} bytes "
                      f"(cap {MAX_IMAGE_BYTES})")
    mime = sniff_image_mime(raw)
    if mime is None:
        return None, "error: uploaded data is not a recognized image"
    ext = {"image/png": ".png", "image/jpeg": ".jpg", "image/gif": ".gif",
           "image/webp": ".webp", "image/bmp": ".bmp",
           "image/tiff": ".tiff"}.get(mime, ".img")
    dest_dir.mkdir(parents=True, exist_ok=True)
    import time

    out = dest_dir / f"upload-{time.strftime('%Y%m%d-%H%M%S')}-" \
        f"{len(list(dest_dir.glob('upload-*'))):03d}{ext}"
    try:
        out.write_bytes(raw)
    except OSError as e:
        return None, f"error saving upload: {e}"
    return str(out), None


# ----------------------------------------------------------- vision probe

#: base_url (without /v1) -> bool | None ("unknown" is never cached).
_vision_cache: Dict[str, Optional[bool]] = {}


def reset_vision_cache() -> None:
    """Test hook: forget per-endpoint vision probe results."""
    _vision_cache.clear()


def detect_vision_support(client: Any, base_url: str) -> Optional[bool]:
    """Does the served model accept image parts? True/False, None unknown.

    llama-server reports its projector in /models metadata; we check
    ``mmproj``-ish keys on each model entry. Result cached per endpoint
    (vision only changes with a server restart / model swap, and switch_model
    clears the entry). Probe failures => None (NOT cached): conservative
    text-stub behavior until someone forces agent.vision.
    """
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    cached = _vision_cache.get(root)
    if cached is not None:
        return cached
    try:
        r = client.get(f"{root}/v1/models")
        if r.status_code != 200:
            return None
        body = r.json() or {}
        entries = list((body.get("data") or [])) + list(body.get("models") or [])
        for m in entries:
            meta = m.get("meta") or m.get("metadata") or {}
            blob = (json_dumps_lower(meta) + " "
                    + str(m.get("id") or m.get("name") or "").lower() + " "
                    + json_dumps_lower(m.get("capabilities") or []))
            # Current llama.cpp advertises a loaded projector as the
            # "multimodal" capability; older builds exposed mmproj in meta.
            if ("mmproj" in blob or "vision" in blob
                    or "multimodal" in blob):
                _vision_cache[root] = True
                return True
        # Endpoint answered but showed no projector => text-only (cacheable).
        _vision_cache[root] = False
        return False
    except Exception:  # noqa: BLE001 - probe must never break a run
        return None


def json_dumps_lower(obj: Any) -> str:
    import json

    try:
        return json.dumps(obj, default=str).lower()
    except Exception:  # noqa: BLE001
        return ""


# ------------------------------------------------- screenshot backends
#
# Cross-OS dispatch. Layers, tried in order, each skipped when its tool is
# absent so a machine only pays for what it has:
#
#   1. Native CLI capturers (platform-ordered). No Python deps.
#   2. Pillow ImageGrab — works out of the box on Windows/macOS and on X11
#      (root window). Fails cleanly on Wayland (X11 get_image is denied).
#   3. xdg-desktop-portal "Screenshot" (Wayland). The compositor-correct path;
#      spawned as a helper under an interpreter that has python-dbus + PyGObject
#      (the agent's own venv usually does not). NOTE: the portal returns a
#      request and then resolves it via a Response signal; it also needs a
#      FOCUSED window to show its consent dialog and will CANCEL (code 2)
#      otherwise. That is a property of the portal, not a bug here.
#
# Every failed layer is recorded so the returned error says exactly what was
# tried and why — no silent "no tool found" when a tool was present but failed.

#: file-mode CLI capturers, preference-ordered per platform. argv templates
#: use {out} for the destination path. ``grim`` additionally gets a stdout
#: fallback (older builds ignore the path arg and write to stdout).
def _cli_capturers() -> List[Tuple[str, List[str]]]:
    import sys

    if sys.platform == "darwin":
        return [("screencapture", ["screencapture", "-x", "{out}"])]
    if sys.platform in ("win32", "cygwin"):
        return []  # rely on the Pillow layer, which is native on Windows
    # Linux + other Unix: Wayland capturers first, then X11 tools.
    return [
        ("grim",               ["grim", "{out}"]),
        ("wayshot",            ["wayshot", "{out}"]),
        ("spectacle",          ["spectacle", "-b", "-n", "-o", "{out}"]),
        ("gnome-screenshot",   ["gnome-screenshot", "-f", "{out}"]),
        ("scrot",              ["scrot", "{out}"]),
        ("maim",               ["maim", "{out}"]),
        ("import",             ["import", "-window", "root", "{out}"]),
    ]


def _session_is_wayland() -> bool:
    v = os.environ.get("XDG_SESSION_TYPE", "").lower()
    if v == "wayland":
        return True
    if v in ("x11", "x-windows-cmdline", "tty"):
        return False
    return bool(os.environ.get("WAYLAND_DISPLAY"))


def _ok_file(p: Path) -> bool:
    return p.is_file() and p.stat().st_size > 0


def _try_cli(out: Path) -> Tuple[Optional[str], List[str]]:
    """Run each available CLI capturer; return (path, failures)."""
    import shutil
    import subprocess

    failures: List[str] = []
    for label, argv in _cli_capturers():
        if not shutil.which(label):
            continue
        rendered = [a.replace("{out}", str(out)) for a in argv]
        try:
            proc = subprocess.run(rendered, capture_output=True, timeout=20)
        except (OSError, subprocess.TimeoutExpired):
            failures.append(f"{label}(no-run)")
            continue
        if proc.returncode == 0 and _ok_file(out):
            return str(out), failures
        # Older grim ignores the path arg and writes the PNG to stdout.
        if label == "grim":
            try:
                proc2 = subprocess.run(["grim"], capture_output=True, timeout=20)
                if proc2.returncode == 0 and proc2.stdout:
                    out.write_bytes(proc2.stdout)
                    if _ok_file(out):
                        return str(out), failures
            except (OSError, subprocess.TimeoutExpired):
                pass
        why = f"rc={proc.returncode}"
        if proc.stderr:
            why += " " + proc.stderr.decode("utf-8", "replace").strip().splitlines()[-1][:120]
        failures.append(f"{label}({why})")
    return None, failures


def _try_pillow(out: Path) -> Tuple[Optional[str], List[str]]:
    """Pillow ImageGrab layer (Windows/macOS native; X11 root window)."""
    try:
        from PIL import ImageGrab
    except Exception as e:  # noqa: BLE001 - absent dep is a normal path
        return None, [f"pillow(no-module:{type(e).__name__})"]
    try:
        im = ImageGrab.grab()
        im.save(str(out))
    except Exception as e:  # noqa: BLE001 - capture can fail on Wayland
        return None, [f"pillow(grab:{type(e).__name__} {str(e)[:120]})"]
    if _ok_file(out):
        return str(out), []
    return None, ["pillow(empty-output)"]


#: Helper run under an interpreter that has python-dbus + PyGObject (the
#: agent's venv typically lacks both; the system python3 usually has them).
_PORTAL_HELPER = r'''
import sys, os, time, glob
out = sys.argv[1]
try:
    import dbus
    import dbus.mainloop.glib
    from gi.repository import GLib
except Exception as e:
    print("SABLE_SHOT_FAIL no-dbus-or-gi:" + type(e).__name__)
    sys.exit(3)
fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
state = {"code": None}
start = time.time()
try:
    loop = dbus.mainloop.glib.DBusGMainLoop()
    bus = dbus.SessionBus(mainloop=loop)
    obj = bus.get_object("org.freedesktop.portal.Desktop",
                         "/org/freedesktop/portal/desktop")
    portal = dbus.Interface(obj, "org.freedesktop.portal.Screenshot")
    mainloop = GLib.MainLoop()
    def on_response(response, results):
        state["code"] = int(response)
        mainloop.quit()
    req = portal.Screenshot("",
                            {"fd": dbus.types.UnixFd(fd),
                             "type": "full-screen"},
                            unix_fds=[fd])
    req_iface = dbus.Interface(
        bus.get_object("org.freedesktop.portal.Desktop", req),
        "org.freedesktop.portal.Request")
    req_iface.connect_to_signal("Response", on_response)
    GLib.timeout_add(20000, mainloop.quit)
    mainloop.run()
except Exception as e:
    print("SABLE_SHOT_FAIL portal-call:" + type(e).__name__ + " " + str(e)[:120])
    try:
        os.close(fd)
    except OSError:
        pass
    sys.exit(4)
try:
    os.close(fd)
except OSError:
    pass
size = os.path.getsize(out) if os.path.exists(out) else 0
if state["code"] == 0 and not size:
    # Some portal backends (notably GNOME Shell's) ignore the fd option
    # and write the PNG into the user's Pictures directory instead.
    # Find the file just written (newest match, mtime >= call start)
    # and move it to the requested path.
    def _pics_dir():
        env = os.environ.get("XDG_PICTURES_DIR")
        if env:
            return os.path.expandvars(os.path.expanduser(env))
        try:
            with open(os.path.expanduser("~/.config/user-dirs.dirs")) as f:
                for line in f:
                    if line.startswith("XDG_PICTURES_DIR="):
                        # user-dirs.dirs is shell syntax: paths look like
                        # "XDG_PICTURES_DIR=\"$HOME/Pictures\"" — expand.
                        return os.path.expandvars(
                            os.path.expanduser(line.split('"')[1]))
        except OSError:
            pass
        return os.path.expanduser("~/Pictures")
    import shutil as _shutil
    best, best_mt = None, 0.0
    for pat in ("Screenshot*.png", "*.png"):
        for f in glob.glob(os.path.join(_pics_dir(), pat)):
            try:
                mt = os.path.getmtime(f)
            except OSError:
                continue
            if mt >= start - 1.0 and mt > best_mt:
                best, best_mt = f, mt
        if best is not None:
            break
    if best is not None:
        try:
            _shutil.move(best, str(out))
        except OSError:
            pass
        size = os.path.getsize(out) if os.path.exists(out) else 0
if state["code"] == 0 and size:
    print("SABLE_SHOT_OK " + out)
    sys.exit(0)
# code: 0=success 1=failed 2=cancelled 3=canceled-unsupported
print("SABLE_SHOT_FAIL portal-code:" + str(state["code"]) + " size:" + str(size))
sys.exit(5)
'''


def _try_portal(out: Path) -> Tuple[Optional[str], List[str]]:
    """xdg-desktop-portal Screenshot (Wayland). Returns (path, failures)."""
    import shutil
    import subprocess
    import sys

    candidates = [sys.executable]
    for cand in ("python3", "python"):
        p = shutil.which(cand)
        if p and p not in candidates:
            candidates.append(p)

    failures: List[str] = []
    for interp in candidates:
        try:
            proc = subprocess.run([interp, "-c", _PORTAL_HELPER, str(out)],
                                  capture_output=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            failures.append("portal(no-run)")
            continue
        text = (proc.stdout or b"").decode("utf-8", "replace").strip()
        if proc.returncode == 0 and _ok_file(out):
            return str(out), failures
        reason = text.splitlines()[-1] if text else (
            "rc=%d stderr:%s" % (proc.returncode,
                                 (proc.stderr or b"").decode("utf-8", "replace").strip()[:120]))
        failures.append(f"portal[{os.path.basename(interp)}]({reason})")
    return None, failures


def take_screenshot(out_path: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """Grab the screen to a PNG; returns (path, error).

    Tries native CLI capturers, then Pillow, then (on Wayland) the
    xdg-desktop-portal, so it works across Windows / macOS / X11 / Wayland
    with only the backends the host actually has. Output lands in
    <SABLE_HOME or temp>/screenshots/ with a timestamped name unless an
    explicit path is given. On total failure the error enumerates every
    backend tried and why it failed.
    """
    import sys
    import tempfile
    import time

    if out_path:
        out = Path(out_path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
    else:
        home = os.environ.get("SABLE_HOME", "").strip()
        base = (Path(home) / "screenshots") if home else \
            (Path(tempfile.gettempdir()) / "sable-screenshots")
        base.mkdir(parents=True, exist_ok=True)
        out = base / f"screenshot-{time.strftime('%Y%m%d-%H%M%S')}.png"

    failures: List[str] = []

    got, f = _try_cli(out)
    failures += f
    if got:
        return got, None

    got, f = _try_pillow(out)
    failures += f
    if got:
        return got, None

    if _session_is_wayland():
        got, f = _try_portal(out)
        failures += f
        if got:
            return got, None

    hint = ""
    if _session_is_wayland():
        hint = (" On Wayland the xdg-desktop-portal needs a FOCUSED window to "
                "show its consent dialog (it otherwise cancels); run with the "
                "terminal focused, or install grim on wlroots/Hyprland.")
    elif sys.platform == "darwin":
        hint = " Grant screen-recording permission to the running app."
    elif sys.platform in ("win32", "cygwin"):
        hint = " Install Pillow (`pip install Pillow`) for the Windows capture path."
    else:
        hint = " On X11 install scrot, maim, or gnome-screenshot (or Pillow)."
    return None, (f"error: no screenshot backend succeeded. tried: "
                  f"{', '.join(failures) or 'nothing available'}.{hint}")

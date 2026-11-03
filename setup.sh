#!/usr/bin/env bash
# toaster-loop one-click setup — safe to double-click or run from a terminal:
#
#   ./setup.sh              # full setup (venv + package + training extras
#                           #  + llama.cpp build for this GPU + desktop icon)
#   ./setup.sh --no-training   # skip the heavy torch training stack
#   ./setup.sh --backend cuda|rocm|metal|vulkan|cpu   # override GPU autodetect
#
# What it does:
#   1. Creates ./.venv (uv if installed, else python3 -m venv) with a
#      Python between 3.11 and 3.13.
#   2. Installs toaster-loop (editable) + dev extras; adds the [training] extra
#      unless --no-training is passed.
#   3. Runs `toaster-loop setup` — detects the GPU and builds llama.cpp for it.
#   4. Installs a desktop icon ("toaster-loop") that launches the dashboard, and
#      prints how to launch from the terminal (./launch.sh).
#   5. Reports which screen-capture backends are present so you know up
#      front whether the take_screenshot tool will work (and what to install
#      if not).
#
# Idempotent: re-running updates everything in place. Requires git + a C/C++
# compiler + cmake for the llama.cpp build (skipped with a warning if absent).
set -euo pipefail

# Portable script-dir resolution: BSD/macOS readlink has no -f, so resolve
# $0 through a readlink loop when it is a symlink, else just cd to its dir.
SELF="$0"
while [ -L "$SELF" ]; do
    LINK="$(readlink "$SELF")"
    case "$LINK" in /*) SELF="$LINK" ;; *) SELF="$(dirname "$SELF")/$LINK" ;; esac
done
cd "$(dirname "$SELF")"
REPO="$PWD"
VENV="$REPO/.venv"

TRAINING=1
BACKEND_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --no-training) TRAINING=0 ;;
        --backend) shift; BACKEND_ARGS=(--backend "$1") ;;
        -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
    esac
    shift
done

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33mwarning: %s\033[0m\n' "$*" >&2; }

# ------------------------------------------------------------- 1. interpreter
say "1/5 locating a Python (3.11-3.13)"
PY=""
if command -v uv >/dev/null 2>&1; then
    # uv can find/download a matching interpreter itself.
    PY="$(uv python find '>=3.11,<3.14' 2>/dev/null || true)"
    if [ -z "$PY" ]; then
        say "no suitable interpreter found — installing one with uv"
        uv python install 3.13 >/dev/null 2>&1 || true
        PY="$(uv python find '>=3.11,<3.14')"
    fi
else
    for c in python3.13 python3.12 python3.11 python3; do
        if command -v "$c" >/dev/null 2>&1 && \
           "$c" -c 'import sys; raise SystemExit(not (3,11) <= sys.version_info < (3,14))'; then
            PY="$(command -v "$c")"; break
        fi
    done
fi
[ -n "$PY" ] || { echo "error: need Python 3.11-3.13 (or install uv: https://docs.astral.sh/uv/)" >&2; exit 1; }
echo "interpreter: $PY ($("$PY" --version 2>&1))"

# ------------------------------------------------------------------- 2. venv
say "2/5 creating venv + installing toaster-loop"
EXTRAS="dev"
[ "$TRAINING" = 1 ] && EXTRAS="dev,training"
if command -v uv >/dev/null 2>&1; then
    uv venv "$VENV" --python "$PY" >/dev/null 2>&1 || true
else
    [ -x "$VENV/bin/python" ] || "$PY" -m venv "$VENV"
    "$VENV/bin/python" -m pip install -q --upgrade pip setuptools wheel
fi

# torch build flavor must match THIS machine's accelerator: plain PyPI
# ships the CUDA wheel on Linux, which silently CPU-trains on AMD/MPS
# boxes (see toaster-loop/core/torchenv.py). Install torch from the right index
# FIRST so the [training] extras resolve against it; then the editable
# install (whose torch>=2.4 pin is already satisfied) won't touch it.
if [ "$TRAINING" = 1 ]; then
    # torchenv is stdlib-only (its runtime import is lazy), so a bare
    # PYTHONPATH probe works before toaster-loop itself is installed.
    FLAVOR_NAME="$(PYTHONPATH="$REPO" "$VENV/bin/python" -c \
        "from toaster_loop.core.torchenv import detect_flavor; print(detect_flavor().flavor)" \
        2>/dev/null || echo default)"
    FLAVOR_IDX="$(PYTHONPATH="$REPO" "$VENV/bin/python" -c \
        "from toaster_loop.core.torchenv import detect_flavor; print(detect_flavor().index_url or '')" \
        2>/dev/null || true)"
    echo "torch      : $FLAVOR_NAME build for this hardware"
    if command -v uv >/dev/null 2>&1; then
        if [ "$FLAVOR_NAME" = default ]; then
            uv pip install --python "$VENV/bin/python" -q torch \
                || { warn "torch install failed"; exit 1; }
        else
            uv pip install --python "$VENV/bin/python" -q torch \
                --torch-backend "$FLAVOR_NAME" \
                --extra-index-url https://download.pytorch.org/whl \
                || { warn "torch install failed ($FLAVOR_NAME)"; exit 1; }
        fi
    elif [ -n "$FLAVOR_IDX" ]; then
        "$VENV/bin/python" -m pip install -q torch --index-url "$FLAVOR_IDX" \
            || { warn "torch install failed (index $FLAVOR_IDX)"; exit 1; }
    else
        "$VENV/bin/python" -m pip install -q torch \
            || { warn "torch install failed"; exit 1; }
    fi
fi

if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$VENV/bin/python" -q -e "$REPO[$EXTRAS]" \
        || { warn "uv pip install failed"; exit 1; }
else
    "$VENV/bin/python" -m pip install -q -e "$REPO[$EXTRAS]" \
        || { warn "package install failed (check network/disk)"; exit 1; }
fi
[ -x "$VENV/bin/toaster-loop" ] || { echo "error: venv install failed — no .venv/bin/toaster-loop" >&2; exit 1; }
echo "installed  : $("$VENV/bin/toaster-loop" --version)"

# --------------------------------------------------------------- 3. llama.cpp
say "3/5 building llama.cpp for this machine (GPU autodetect)"
if command -v git >/dev/null 2>&1 && command -v cmake >/dev/null 2>&1 \
   && command -v cc >/dev/null 2>&1; then
    "$VENV/bin/toaster-loop" setup ${BACKEND_ARGS[@]+"${BACKEND_ARGS[@]}"} \
        || warn "llama.cpp setup failed — serving/training may be limited; re-run ./setup.sh or 'toaster-loop setup' after fixing the toolchain"
else
    warn "git/cmake/compiler missing — skipping llama.cpp build."
    warn "install them (e.g. sudo apt install git build-essential cmake) and re-run ./setup.sh"
fi

# ------------------------------------------------------------ 4. desktop icon
say "4/5 installing desktop launcher icon"
ICON_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
ICON_PNG_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor/scalable/apps"
if [ "$(uname)" = "Linux" ]; then
    mkdir -p "$ICON_DIR" "$ICON_PNG_DIR"
    cp "$REPO/assets/toaster-loop.svg" "$ICON_PNG_DIR/toaster-loop.svg" 2>/dev/null || true
    cat > "$ICON_DIR/toaster-loop.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Toaster Loop
Comment=Local self-improving agent dashboard
Exec=$REPO/launch.sh
Icon=toaster-loop
Terminal=false
Categories=Development;MachineLearning;
StartupNotify=true
EOF
    update-desktop-database "$ICON_DIR" >/dev/null 2>&1 || true
    echo "icon       : 'Toaster Loop' added to your applications menu"
fi

# ------------------------------------------------ 5. screenshot backends
# take_screenshot dispatches across backends (see core/images.py); tell the
# user up front which ones exist so they know the tool will (or won't) work
# without running it. Never fatal — the tool degrades with a clear error at
# runtime instead.
say "5/5 checking screen-capture backends for take_screenshot"
case "$(uname)" in
    Darwin)
        command -v screencapture >/dev/null 2>&1 \
            && echo "capture    : screencapture (native) — OK" \
            || warn "grant Screen Recording permission in System Settings for the running app"
        ;;
    Linux)
        if [ "${XDG_SESSION_TYPE:-}" = "wayland" ]; then
            for c in grim wayshot spectacle gnome-screenshot; do
                command -v "$c" >/dev/null 2>&1 \
                    && { echo "capture    : $c — OK"; break; }
            done
            echo "note       : Wayland. grim only works on wlroots (sway/hyprland); on"
            echo "             GNOME the xdg-desktop-portal is used, which needs the"
            echo "             terminal focused so its consent dialog can appear."
        else
            for c in scrot maim import; do
                command -v "$c" >/dev/null 2>&1 \
                    && { echo "capture    : $c — OK"; break; }
            done
        fi
        "$VENV/bin/python" -c 'import PIL.ImageGrab' 2>/dev/null \
            && echo "capture    : Pillow ImageGrab — OK (X11/Windows/macOS native)" \
            || warn "Pillow missing from venv — re-run ./setup.sh (installed via pyproject)"
        ;;
    *)
        "$VENV/bin/python" -c 'import PIL.ImageGrab' 2>/dev/null \
            && echo "capture    : Pillow ImageGrab — OK" \
            || warn "Pillow missing from venv — re-run ./setup.sh (installed via pyproject)"
        ;;
esac

say "done."
echo "  launch   : ./launch.sh   (or the toaster-loop icon in your apps menu)"
echo "  stop     : ./launch.sh --stop"
echo "  cleanup  : ./cleanup.sh  (fresh-downloaded state)"
echo "  tests    : .venv/bin/pytest"

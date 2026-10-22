#!/usr/bin/env bash
# sable cleanup — return the program to a freshly-downloaded state.
#
#   ./cleanup.sh              # interactive: shows what would be removed, asks
#   ./cleanup.sh --yes        # no prompt (also: -y)
#   ./cleanup.sh --keep-models
#
# Removes:
#   * SABLE_HOME runtime state (~/.sable by default): traces, skills, mined
#     goldens, buffer, datasets, model registry, logs, serve/launch state,
#     managed llama.cpp build (everything under the home).
#   * Legacy ~/.unforge state dir from before the rename.
#   * Local dev artifacts in this checkout: .venv/, __pycache__/, egg-info,
#     pytest caches, stray runtime dirs/dbs the repo .gitignore lists, and
#     coverage output (.coverage, htmlcov/).
#   * The installed desktop icon.
# Keeps: everything tracked by git (source, docs, tests) — i.e. exactly the
# state right after `git clone`. Never touches foreign installs.
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"
REPO="$PWD"

ASSUME_YES=0
KEEP_MODELS_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --yes|-y) ASSUME_YES=1 ;;
        --keep-models) KEEP_MODELS_ARGS=(--keep-models) ;;
        -h|--help) sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
    esac
    shift
done

SABLE_BIN="$REPO/.venv/bin/sable"

echo "This returns sable to a freshly-downloaded state:"
echo "  - wipes the runtime home (traces, datasets, models, logs, llama.cpp)"
echo "  - deletes the local .venv and build artifacts in $REPO"
echo "  - removes the desktop launcher icon"
echo "Source files under git are left untouched."
echo

if [ "$ASSUME_YES" != 1 ]; then
    read -r -p "Continue? [y/N] " ans
    case "$ans" in y|Y|yes|YES) ;; *) echo "aborted."; exit 0 ;; esac
fi

# 1b. Legacy pre-rename state dir (~/.unforge), if any survived the rename.
if [ -z "${UNFORGE_HOME:-}" ] && [ -d "$HOME/.unforge" ]; then
    echo "== removing legacy ~/.unforge state =="
    rm -rf -- "$HOME/.unforge"
fi

# 1. Runtime state (stops sable-owned server/dashboard first). Uses the venv
#    CLI when present; otherwise a bare python -c import as fallback.
if [ -x "$SABLE_BIN" ]; then
    echo "== wiping runtime home =="
    "$SABLE_BIN" reset --yes ${KEEP_MODELS_ARGS[@]+"${KEEP_MODELS_ARGS[@]}"} \
        || echo "warning: 'sable reset' failed; continuing with repo cleanup"
else
    echo "== runtime home: venv already gone — removing ~/.sable directly =="
    HOME_DIR="${SABLE_HOME:-$HOME/.sable}"
    case "$HOME_DIR" in
        "$HOME/.sable"|"$HOME"/.sable-*) rm -rf -- "$HOME_DIR" ;;
        *) echo "warning: refusing to delete unexpected SABLE_HOME=$HOME_DIR" >&2 ;;
    esac
fi

# 2. Repo-local artifacts (the ones .gitignore declares as generated).
echo "== cleaning repo checkout =="
rm -rf .venv .pytest_cache .mypy_cache .ruff_cache sable.egg-info
# Legacy egg-info from the pre-rename (unforge) era.
rm -rf unforge.egg-info
find . -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
# Runtime dirs that may have been created inside the checkout (see .gitignore).
rm -rf adapter outputs train_runs datasets models logs None
rm -f serve_state.json serve_state.json.tmp config.yaml .env
# Stray databases and coverage artifacts anywhere in the tree.
find . -path ./.venv -prune -o \( -name '*.db' -o -name '*.db-wal' -o -name '*.db-shm' \) -type f -print0 2>/dev/null | xargs -0 rm -f -- 2>/dev/null || true
rm -rf .coverage htmlcov

# 3. Desktop icon installed by setup.sh.
ICON_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
rm -f "$ICON_DIR/sable.desktop" \
      "${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor/scalable/apps/sable.svg"
update-desktop-database "$ICON_DIR" >/dev/null 2>&1 || true

echo
echo "sable is back to a freshly-downloaded state."
echo "To start using it again: ./setup.sh"

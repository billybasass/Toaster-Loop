#!/usr/bin/env bash
# sable launcher — double-click / CLI entry point.
#
#   ./launch.sh            # start the dashboard (front+back end) + open browser
#   ./launch.sh --stop     # stop the dashboard this launcher started
#   ./launch.sh --no-browser
#
# Idempotent: if a launched dashboard is already running, just re-opens its URL.
# Uses the repo venv; runs `sable launch` which supervises everything and
# records state under SABLE_HOME (default ~/.sable).
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

if [ ! -x .venv/bin/sable ]; then
    echo "sable is not set up yet — running ./setup.sh first."
    ./setup.sh || { echo "setup failed; see output above." >&2; exit 1; }
fi

exec .venv/bin/sable launch "$@"

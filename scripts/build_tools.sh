#!/usr/bin/env bash
# Build the S4 row drivers to build/<row file stem> (the tests/run.sh convention). Usage: scripts/build_tools.sh
set -euo pipefail
cd "$(dirname "$0")/.."
MOJO=${MOJO:-$HOME/Projects/mojo/mojo-cv/.venv/bin/mojo}
JSON_SRC=${JSON_SRC:-$HOME/Projects/mojo/mojo-json/src}
mkdir -p build
for stem in slicer slicer_merge line sinks smoother zone; do
  "$MOJO" build -O1 -I src -I src/drivers -I "$JSON_SRC" "src/drivers/tools_$stem.mojo" -o "build/tools_$stem" \
    2> ".work/build-tools_$stem.log" || { tail -20 ".work/build-tools_$stem.log"; echo "FAIL build tools_$stem: see .work/build-tools_$stem.log"; exit 1; }
done

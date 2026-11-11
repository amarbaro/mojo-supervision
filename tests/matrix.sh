#!/usr/bin/env bash
# Full suite on the pinned stable Mojo (.venv) and on the nightly; non-zero if either fails.
set -uo pipefail
cd "$(dirname "$0")/.."
NIGHTLY=${MOJO_NIGHTLY:-$HOME/Projects/mojo/mojo-nightly/.venv/bin}
rc=0
tag=$(basename "$PWD")
for bin in "$PWD/.venv/bin" "$NIGHTLY"; do
  v=$("$bin/mojo" --version)
  rm -rf build
  if PATH="$bin:$PATH" tests/build_all.sh > ".work/matrix-$tag-build.txt" 2>&1 && PATH="$bin:$PATH" tests/run.sh > ".work/matrix-$tag-${v// /_}.txt" 2>&1; then echo "PASS $v"
  else echo "FAIL $v: see .work/matrix-$tag-${v// /_}.txt"; rc=1; fi
done
exit $rc

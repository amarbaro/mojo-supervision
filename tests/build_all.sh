#!/usr/bin/env bash
# Build every row binary: the lane build scripts, then build/<stem> from <stem>.mojo for any stem still missing.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build .work
J=${MOJO_JSON:-$HOME/Projects/mojo/mojo-json/src}
C=${MOJO_CV:-$HOME/Projects/mojo/mojo-cv/src}
G=${MOJO_GZIP:-$HOME/Projects/mojo/mojo-gzip/src}
tests/build.sh; tests/build_metrics.sh; scripts/build_tools.sh
for f in tests/cases/*.txt; do
  s=$(basename "$f" .txt); [ -x "build/$s" ] && continue
  src=$(git ls-files "*/$s.mojo" | head -1)
  [ -n "$src" ] || { echo "FAIL build: no source for $s"; exit 1; }
  mojo build -O3 -I src -I "$J" -I "$C" -I "$G" "$src" -o "build/$s" 2>".work/build-$s.log" || { tail -20 ".work/build-$s.log"; echo "FAIL build: $s"; exit 1; }
done
echo "build_all: $(ls build | wc -l) binaries"

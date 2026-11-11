#!/usr/bin/env bash
# Run row files against their Mojo binaries. See tests/README.md.
# Usage: tests/run.sh [rowfile ...]   (default: tests/cases/*.txt)
# Env: BUILD_DIR (default build), ORACLE_PY (default .work/oracle-venv/bin/python)
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD_DIR=${BUILD_DIR:-build}
PY=${ORACLE_PY:-.work/oracle-venv/bin/python}
OUT=${BUILD_DIR:-build}/tests-out
mkdir -p "$OUT"

if [ $# -gt 0 ]; then files=("$@"); else
  shopt -s nullglob; files=(tests/cases/*.txt); shopt -u nullglob
fi
if [ ${#files[@]} -eq 0 ]; then
  echo "SUMMARY 0 row files, 0 rows (tests/cases/*.txt is empty)"
  exit 0
fi

status=0
for file in "${files[@]}"; do
  [ -f "$file" ] || { echo "FAIL $file: no such row file"; status=1; continue; }
  stem=$(basename "$file" .txt)
  bin="$BUILD_DIR/$stem"
  if [ ! -x "$bin" ]; then
    echo "FAIL $file: binary $bin missing (build it first)"; status=1; continue
  fi
  rows="$OUT/$stem.rows"; actual="$OUT/$stem.out"
  grep -v -e '^#' -e '^$' "$file" > "$rows" || true
  if ! "$bin" < "$rows" > "$actual"; then
    echo "FAIL $file: $bin exited non-zero"; status=1; continue
  fi
  "$PY" tools/rowdiff.py "$file" "$actual" || status=1
done
exit $status

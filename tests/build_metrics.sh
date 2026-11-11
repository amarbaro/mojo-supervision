#!/usr/bin/env bash
# Build the metrics row runner once and install it under each row-file stem.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build
rm -f build/metrics_rows
mojo build -I src tests/metrics_rows.mojo -o build/metrics_rows || { echo "FAIL build: mojo build tests/metrics_rows.mojo"; exit 1; }
for stem in map pr mar cm kp; do cp build/metrics_rows "build/metrics_$stem"; done

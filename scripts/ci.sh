#!/usr/bin/env bash
# CI entry point (GitHub runner or local): install, build every row driver, run every row file. MOJO=path overrides the toolchain.
# Needs mojo-json, mojo-cv and mojo-gzip sources: MOJO_JSON/MOJO_CV/MOJO_GZIP, else a sibling checkout, else a shallow clone into .work/deps.
# Ends with "PASS n/m" (or "FAIL k/m"): the CI badge step parses that line.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -z "${MOJO:-}" ]; then
  command -v uv >/dev/null || { echo "FAIL setup: uv not installed"; exit 1; }
  uv sync -q || { echo "FAIL setup: uv sync"; exit 1; }
  MOJO=$PWD/.venv/bin/mojo
fi
export PATH="$(dirname "$MOJO"):$PATH"   # tests/build*.sh call a bare `mojo`
"$MOJO" --version
dep() {  # dep VAR LIB: export VAR = LIB's src directory
  local var=$1 lib=$2 d
  [ -z "${!var:-}" ] || return 0
  for d in "../$lib-public" "../$lib" ".work/deps/$lib"; do [ -d "$d/src" ] && { export "$var=$PWD/$d/src"; return 0; }; done
  git clone -q --depth 1 "https://github.com/amarbaro/$lib.git" ".work/deps/$lib" || { echo "FAIL setup: clone $lib"; exit 1; }
  export "$var=$PWD/.work/deps/$lib/src"
}
mkdir -p .work build
dep MOJO_JSON mojo-json; dep MOJO_CV mojo-cv; dep MOJO_GZIP mojo-gzip
if [ -z "${ORACLE_PY:-}" ]; then  # the pixel rows read PNGs with opencv and numpy
  uv venv -q .work/oracle-venv && uv pip install -q --python .work/oracle-venv/bin/python numpy opencv-python-headless || { echo "FAIL setup: oracle venv"; exit 1; }
  export ORACLE_PY=$PWD/.work/oracle-venv/bin/python
fi
m=3; ok=0
gate() { local name=$1; shift; if "$@"; then ok=$((ok+1)); echo "ok $name"; else echo "FAIL $name"; echo "FAIL $((m-ok))/$m"; exit 1; fi; }
gate selftest tests/selftest.sh
gate build tests/build_all.sh
gate rows tests/run.sh
echo "PASS $ok/$m"
echo "PASS ci"

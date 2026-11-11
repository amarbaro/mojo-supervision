#!/usr/bin/env bash
# Self-test of the harness itself (tests/run.sh, tools/rowdiff.py, tools/pixdiff.py).
# Uses stub binaries (awk echoes the expected column), so no Mojo build is needed.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${ORACLE_PY:-.work/oracle-venv/bin/python}
T=.work/selftest
rm -rf "$T"; mkdir -p "$T/build" "$T/cases"
printf '#!/usr/bin/env bash\nawk -F"\\t" '\''{print $3}'\''\n' > "$T/build/echo_ok"
printf '#!/usr/bin/env bash\necho 2\n' > "$T/build/bad"
chmod +x "$T/build/echo_ok" "$T/build/bad"
fail() { echo "FAIL selftest <$1>: $2"; exit 1; }
export BUILD_DIR="$T/build"

# 1. Passing rows: float tolerance, nan, nested JSON; comments and blank lines skipped.
printf '# gen: none seed=0\n\nadd\t{"a":1}\t[1.0000001,"nan",{"k":2}]\nadd\t{"a":2}\t3.5\n' > "$T/cases/echo_ok.txt"
tests/run.sh "$T/cases/echo_ok.txt" > "$T/pass.log" || fail pass "expected exit 0"
grep -q "^PASS $T/cases/echo_ok.txt:3$" "$T/pass.log" || fail pass "line 3 not PASS"
grep -q "2 rows, 0 failed" "$T/pass.log" || fail pass "summary wrong"

# 2. A failing row exits non-zero and prints FAIL with its line number.
printf 'add\t{}\t1.001\n' > "$T/cases/bad.txt"
if tests/run.sh "$T/cases/bad.txt" > "$T/bad.log"; then fail fail "bad row exited 0"; fi
grep -q "^FAIL $T/cases/bad.txt:1" "$T/bad.log" || fail fail "no FAIL line"

# 3. Missing binary is a FAIL, never a skip.
printf 'nobin\t{}\t1\n' > "$T/cases/nobin.txt"
if tests/run.sh "$T/cases/nobin.txt" > "$T/nobin.log"; then fail missing "exited 0"; fi
grep -q "binary .* missing" "$T/nobin.log" || fail missing "no missing-binary message"

# 4. Pixel rows: identical PNG passes at byte-exact bar; one changed pixel fails it.
$PY - "$T" <<'PY_END'
import sys, numpy as np, cv2
t = sys.argv[1]
img = np.zeros((8, 8, 3), np.uint8); img[2:5, 2:5] = (10, 200, 30)
cv2.imwrite(f"{t}/ref.png", img)
cv2.imwrite(f"{t}/same.png", img)
bad = img.copy(); bad[0, 0] = (11, 200, 30)
cv2.imwrite(f"{t}/bad.png", bad)
PY_END
printf '#!/usr/bin/env bash\necho "{\\"out\\":\\"%s/same.png\\"}"\n' "$T" > "$T/build/pix"
printf '#!/usr/bin/env bash\necho "{\\"out\\":\\"%s/bad.png\\"}"\n' "$T" > "$T/build/pixbad"
chmod +x "$T/build/pix" "$T/build/pixbad"
printf 'pix\t{"input":"x","params":{}}\t{"ref":"%s/ref.png"}\n' "$T" > "$T/cases/pix.txt"
printf 'pixbad\t{"input":"x","params":{}}\t{"ref":"%s/ref.png"}\n' "$T" > "$T/cases/pixbad.txt"
tests/run.sh "$T/cases/pix.txt" > "$T/pix.log" || fail pixel "identical png failed"
if tests/run.sh "$T/cases/pixbad.txt" > "$T/pixbad.log"; then fail pixel "1-pixel diff passed byte-exact bar"; fi
grep -q "^FAIL" "$T/pixbad.log" || fail pixel "no FAIL line for pixel diff"
echo "selftest PASS: rows+float tol+nan, failing row, missing binary, pixel exact/diff"

#!/usr/bin/env bash
# Build OpenCV 5.0.0's own stb_truetype.cpp (gh api, tag 5.0.0) and check that it reproduces every glyph row of
# tests/cases/font_ttf.txt: those rows are what stbtt computes, not just what one cv2 build drew. No FMA: contraction is off.
set -euo pipefail
cd "$(dirname "$0")/.."
D=.work/stbtt_ref; mkdir -p "$D"
for f in stb_truetype.cpp stb_truetype.hpp; do
  [ -s "$D/$f" ] || gh api "repos/opencv/opencv/contents/modules/imgproc/src/$f?ref=5.0.0" -H "Accept: application/vnd.github.raw" > "$D/$f"
done
g++ -O2 -ffp-contract=off -I"$D" -o "$D/ref" tools/stbtt_ref.cpp "$D/stb_truetype.cpp" || { echo "FAIL stbtt_ref: compile"; exit 1; }
.work/oracle-venv/bin/python -I - "$D/ref" <<'PY'
import gzip, json, subprocess, sys
rows = [l.rstrip("\n").split("\t") for l in open("tests/cases/font_ttf.txt") if not l.startswith("#")]
rows = [(json.loads(r[1]), json.loads(r[2])) for r in rows if r[0] == "glyph"]
open(".work/stbtt_ref/ttf", "wb").write(gzip.open("src/supervision/_cv/font/Rubik-sans.ttf.gz").read())
inp = "".join(f'{a["cp"]} {a["size"]} {a["weight"]}\n' for a, _ in rows)
out = subprocess.run([sys.argv[1], ".work/stbtt_ref/ttf"], input=inp, capture_output=True, text=True, check=True).stdout.split("\n")
bad = 0
for i, (a, x) in enumerate(rows):
    hd = list(map(int, out[2 * i].split())); pix = list(map(int, out[2 * i + 1].split()))
    if hd != [x["xoff"], x["yoff"], x["w"], x["h"], x["adv"]] or pix != x["pix"]:
        bad += 1; print(f"FAIL stbtt_ref: glyph row {i} cp={a['cp']} size={a['size']} weight={a['weight']}")
print(f"stbtt_ref: {len(rows) - bad}/{len(rows)} glyph rows equal the C reference")
sys.exit(1 if bad else 0)
PY

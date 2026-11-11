#!/usr/bin/env python3
"""Oracle for cv2.putText / cv2.getTextSize (FONT_HERSHEY_SIMPLEX, the built-in "sans" Rubik; see src/supervision/_cv/font).

Writes tests/fixtures/pixels/text_bg.png (gradient canvas), one reference PNG per put row, and tests/cases/pixels_text.txt in
the pixel-row contract (README: args {input, params, tol}, expected {ref}) plus numeric getTextSize rows
(expected {"value": [w, h, baseline]}). Every bar is 0: max abs diff 0 and 0 % differing pixels. If the Mojo port is not exact the
measured diff is reported, the bar is not loosened.

Run: .work/oracle-venv/bin/python -I tools/gen_text.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oracle import row

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "tests" / "fixtures" / "pixels"
H, W = 120, 320
STRINGS = ["Hello 123", "A", "The quick brown fox", "jumps over 0123456789!?", "gjpqy_()[]{}",
           "%$#@&*+=<>/\\|~^", "MWmw iIl1 oO0", "",
           "Line one\nLine two\n\nLine four", "Caf\u00e9 \u00dcn\u00ef \u00e7\u00f1 \u20ac"]
SCALES = [0.3, 0.5, 0.9, 1.0, 1.5, 2.0]
COLORS = [[255, 255, 255], [20, 200, 40], [0, 0, 0], [30, 60, 250]]

y, x = np.mgrid[0:H, 0:W]
bg = np.stack([(x * 255 // W), (y * 255 // H), ((x + y) * 255 // (W + H))], axis=2).astype(np.uint8)
FIX.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(FIX / "text_bg.png"), bg)

BAR = {"max_abs": 0, "max_pct": 0.0}
lines = ["# gen: tools/gen_text.py seed=0", "# cv2.putText/getTextSize FONT_HERSHEY_SIMPLEX vs src/supervision/_cv/font; bar 0 on every row"]
n = 0
for si, text in enumerate(STRINGS):
    for scale in SCALES:
        for thickness in (1, 2):
            (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
            lines.append(row("text_size", {"params": {"text": text, "scale": scale, "thickness": thickness}, "tol": BAR},
                             {"value": [int(tw), int(th), int(base)]}))
            if not text:
                continue
            color = COLORS[(si + n) % len(COLORS)]
            org = [8, 90 if scale <= 1.0 else 105]
            img = bg.copy()
            cv2.putText(img, text, tuple(org), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
            rel = f"tests/fixtures/pixels/text_{n:03d}.png"
            cv2.imwrite(str(ROOT / rel), img)
            lines.append(row("text_put", {"input": "tests/fixtures/pixels/text_bg.png",
                                          "params": {"text": text, "org": org, "scale": scale, "color": color, "thickness": thickness},
                                          "tol": BAR}, {"ref": rel}))
            n += 1
# placement: origin partly or fully outside the canvas, so the blend clips on every side
for text, org in [("Clipped left", [-30, 60]), ("Clipped right", [260, 60]), ("Clipped top", [10, 4]), ("Clipped bottom", [10, 118]),
                  ("Off canvas", [400, 60]), ("Above", [10, -40]), ("Two\nlines", [-5, 20])]:
    for scale, thickness in ((0.7, 1), (1.2, 2)):
        img = bg.copy()
        cv2.putText(img, text, tuple(org), cv2.FONT_HERSHEY_SIMPLEX, scale, COLORS[1], thickness, cv2.LINE_AA)
        rel = f"tests/fixtures/pixels/text_{n:03d}.png"
        cv2.imwrite(str(ROOT / rel), img)
        lines.append(row("text_put", {"input": "tests/fixtures/pixels/text_bg.png",
                                      "params": {"text": text, "org": org, "scale": scale, "color": COLORS[1], "thickness": thickness},
                                      "tol": BAR}, {"ref": rel}))
        n += 1
(ROOT / "tests" / "cases" / "pixels_text.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {len(lines) - 2} rows, {n} reference PNGs", file=sys.stderr)

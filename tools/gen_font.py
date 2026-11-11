"""Oracle for the font parser rows (tests/cases/font_ttf.txt): fontTools reads cv2's built-in "sans" face
(tools/gen_font_blob.py) and states what supervision._cv.font.ttf must return.

ops: cmap (codepoint -> glyph id), hmetrics ((advance, lsb) from hmtx, default instance), vmetrics (hhea),
     shape (outline of one glyph: sorted unique on-curve endpoints, off-curve controls, bbox) at an optional wght.
The shape rows compare sets, not sequences: stbtt starts a contour at a different point than fontTools and rounds
implied on-curve midpoints with >>1, so composites and variation instances carry tol abs 1 (stbtt's integer delta pipeline).

Run: .work/oracle-venv/bin/python -I tools/gen_font.py
"""
import gzip
import io
import sys

sys.path.insert(0, "tools")
import cv2
import numpy as np
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer

from oracle import write_rows

GZ = "src/supervision/_cv/font/Rubik-sans.ttf.gz"
base = TTFont(io.BytesIO(gzip.open(GZ).read()))
order = base.getGlyphOrder()
glyf = base["glyf"]
WGHTS = [None]


def instance(wght):
    if wght is None:
        return base
    return instancer.instantiateVariableFont(TTFont(io.BytesIO(gzip.open(GZ).read())), {"wght": wght}, inplace=False)


def outline(font, gid):
    """Sorted x and y lists of the on-curve endpoints (ex, ey) and off-curve controls (cx, cy) stbtt reports, plus the bbox.
    Sorting each coordinate list separately keeps rows comparable under a +-d tolerance: a shift of at most d per point shifts
    every element of a sorted list by at most d."""
    g = font["glyf"]
    gl = g[order[gid]]
    if gl.numberOfContours == 0:
        return {"ex": [], "ey": [], "cx": [], "cy": [], "box": [0, 0, 0, 0]}
    coords, end_pts, flags = gl.getCoordinates(g)
    ends, ctrl = [], []
    start = 0
    for e in end_pts:
        pts = [(round(x), round(y), bool(f & 1)) for (x, y), f in zip(coords[start : e + 1], flags[start : e + 1])]
        start = e + 1
        n = len(pts)
        for i, (x, y, on) in enumerate(pts):
            nx, ny, non = pts[(i + 1) % n]
            if on:
                ends.append((x, y))
            else:
                ctrl.append((x, y))
                if not non:
                    ends.append(((x + nx) >> 1, (y + ny) >> 1))
    # stbtt's box is the header bbox plus the gvar deltas of the phantom points (advance variation lives there): not
    # derivable from fontTools coordinates, so variation rows leave it out; cv2's getTextSize/putText rows pin it.
    gb = base["glyf"][order[gid]]
    box = [gb.xMin, gb.yMin, gb.xMax, gb.yMax]
    return {"ex": sorted(p[0] for p in ends), "ey": sorted(p[1] for p in ends),
            "cx": sorted(p[0] for p in ctrl), "cy": sorted(p[1] for p in ctrl), "box": box}


rows = []
cmap = base.getBestCmap()
rows.append(("cmap", {"cps": list(range(0, 128))}, [base.getGlyphID(cmap[c]) if c in cmap else 0 for c in range(0, 128)]))
rows.append(("cmap", {"cps": list(range(128, 700)) + [0x20AC, 0x2022, 0x4E2D, 0x10000, 0xFFFD]},
             [base.getGlyphID(cmap[c]) if c in cmap else 0 for c in list(range(128, 700)) + [0x20AC, 0x2022, 0x4E2D, 0x10000, 0xFFFD]]))
hm = base["hmtx"]
n = len(order)
rows.append(("hmetrics", {"gids": list(range(n))}, [list(hm[order[g]]) for g in range(n)]))
h = base["hhea"]
rows.append(("vmetrics", {}, [h.ascent, h.descent, h.lineGap]))

ascii_gids = [base.getGlyphID(cmap[c]) for c in range(32, 127)]
composites = [g for g in range(n) if glyf[order[g]].isComposite()]
simple = [g for g in range(0, n, 7) if not glyf[order[g]].isComposite()]
for g in sorted(set(ascii_gids + simple[:60])):
    rows.append(("shape", {"gid": g, "wght": None}, outline(base, g)))
for g in composites[:80:2]:
    rows.append(("shape", {"gid": g, "wght": None, "tol": {"abs": 1, "rel": 0}}, outline(base, g)))

# variation instances: fontTools' float IUP + rounding vs stbtt's integer pipeline, so the bar is 2 font units (1000/em)
inst = {w: instance(w) for w in (300, 400, 600, 800, 900)}
for w, f in inst.items():
    for g in sorted(set(ascii_gids[::3] + composites[:40:5] + simple[:12])):
        o = outline(f, g)
        del o["box"]
        rows.append(("shape", {"gid": g, "wght": w, "tol": {"abs": 2, "rel": 0}}, o))



def cv_glyph(ch, size, weight):
    """cv2 draws white on black, so the blended pixel is exactly the glyph's coverage byte; the cropped non-zero box and its
    offset from the pen are what stbtt_GetGlyphBitmapSubpixelRealloc returns (xoff, yoff, w, h, pix); adv is the cvRound(adv*64)>>6 pen step."""
    fs = size * 3.7 / 100
    assert round(100 * fs / 3.7) == size
    thickness = 1 if weight == 400 else 2
    pad = size * 3 + 40
    img = np.zeros((pad, pad, 3), np.uint8)
    org = (size, size * 2)
    cv2.putText(img, ch, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), thickness, cv2.LINE_AA)
    adv = cv2.getTextSize(ch, cv2.FONT_HERSHEY_SIMPLEX, fs, thickness)[0][0] - 1
    nz = np.argwhere(img[:, :, 0] > 0)
    if len(nz) == 0:
        return {"xoff": 0, "yoff": 0, "w": 0, "h": 0, "pix": [], "adv": adv}
    (y0, x0), (y1, x1) = nz.min(0), nz.max(0)
    crop = img[y0 : y1 + 1, x0 : x1 + 1, 0]
    assert (img[:, :, 0] == img[:, :, 1]).all()
    return {"xoff": int(x0 - org[0]), "yoff": int(y0 - org[1]), "w": int(x1 - x0 + 1), "h": int(y1 - y0 + 1),
            "pix": crop.flatten().tolist(), "adv": adv}


printable = [chr(c) for c in range(32, 127)]
for size in (10, 26):
    for weight in (400, 600):
        for ch in printable:
            rows.append(("glyph", {"cp": ord(ch), "size": size, "weight": weight}, cv_glyph(ch, size, weight)))
for size in (5, 17, 40, 64, 100):
    for weight in (400, 600):
        for ch in "AgQ%@8,.":
            rows.append(("glyph", {"cp": ord(ch), "size": size, "weight": weight}, cv_glyph(ch, size, weight)))

write_rows("tests/cases/font_ttf.txt", "gen_font", 0, rows)

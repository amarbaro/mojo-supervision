#!/usr/bin/env python3
"""Oracle for the pixel rows: Python supervision and cv2 render references as PNG.

Writes tests/fixtures/pixels/ inputs and reference PNGs, plus two row files in the
BOSS CONTRACT format (`<op>\\t<args JSON>\\t<expected JSON>`):
  tests/cases/pixels_annotators.txt  supervision annotators (Python oracle)
  tests/cases/pixels_cv.txt          S5 _cv ops, one row per op vs cv2 (+ getTextSize numeric)
Args carry input, params and tol; expected carries the ref path. Every bar is stated
in the row and never loosened to pass. PNG only: no JPEG fixtures.

Run with the shared venv (H1 owns it):
    .work/oracle-venv/bin/python tools/gen_pixels.py [--seed 5]
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from supervision.key_points.annotators import VertexAnnotator
from supervision.key_points.core import KeyPoints

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "pixels"
CASES = ROOT / "tests" / "cases"
H, W = 240, 320

# Fixed scene. The same numbers go into every row's params, so the Mojo side reads them, not regenerates them.
XYXY = [[20, 30, 120, 130], [150, 60, 260, 200], [40, 150, 100, 230]]
CLASS_ID = [0, 1, 0]
CONFIDENCE = [0.9, 0.75, 0.6]
POLYGONS = [
    [[20, 30], [120, 30], [120, 130], [20, 130]],
    [[150, 60], [260, 60], [230, 200], [150, 200]],
    [[40, 150], [100, 150], [70, 230], [40, 230]],
]
LABELS = ["car", "bus", "cat"]
TRACE_FRAMES = [
    [[x0 + 4 * f, y0 + 2 * f, x1 + 4 * f, y1 + 2 * f] for x0, y0, x1, y1 in XYXY] for f in range(5)
]
KEYPOINTS = [[[60, 50], [90, 60], [75, 100], [50, 120], [100, 125]]]

# cv2 op inputs (same numbers in params).
CIRCLE = {"center": [160, 120], "radius": 60, "color": [0, 255, 255], "thickness": 3}
ELLIPSE = {"center": [200, 110], "axes": [70, 35], "angle": 25, "start": 30, "end": 280,
           "color": [255, 0, 128], "thickness": 2}
POLY_OPEN = {"pts": [[10, 10], [60, 200], [110, 20], [160, 210], [210, 30]], "closed": False,
             "color": [0, 200, 0], "thickness": 2}
POLY_CLOSED = {"pts": [[230, 20], [300, 60], [280, 130], [240, 110]], "closed": True,
               "color": [200, 0, 200], "thickness": 1}
# Non-convex (L-shape plus notch) so a convex-only fill would be visibly wrong.
FILL_NONCONVEX = {"pts": [[150, 150], [300, 150], [300, 170], [170, 170], [170, 230],
                          [150, 230]], "color": [30, 180, 255]}
TEXT = {"text": "Hello 123", "org": [12, 225], "scale": 0.9, "color": [255, 255, 255],
        "thickness": 2}
TEXT_SIZE_CASE = {"text": "Hello 123", "scale": 0.9, "thickness": 2}
ADD_WEIGHTED = {"alpha": 0.4, "beta": 0.6, "gamma": 10.0}
CONVERT_SCALE_ABS = {"alpha": 1.5, "beta": -40.0}

# Bars per primitive. Integer-geometry draws must match exactly; blended or filtered output gets a stated bar.
BARS = {
    "box": {"max_abs": 0, "max_pct": 0.0},
    "label": {"max_abs": 0, "max_pct": 0.0},
    "mask": {"max_abs": 1, "max_pct": 0.5},
    "polygon": {"max_abs": 0, "max_pct": 0.0},
    "circle": {"max_abs": 0, "max_pct": 0.0},
    "dot": {"max_abs": 0, "max_pct": 0.0},
    "halo": {"max_abs": 2, "max_pct": 1.0},
    "blur": {"max_abs": 1, "max_pct": 1.0},
    "pixelate": {"max_abs": 0, "max_pct": 0.0},
    "trace": {"max_abs": 0, "max_pct": 0.0},
    "keypoints": {"max_abs": 0, "max_pct": 0.0},
}

CV_BARS = {
    "circle_outline": {"max_abs": 0, "max_pct": 0.0},
    "ellipse_arc": {"max_abs": 0, "max_pct": 0.0},
    "polylines_open": {"max_abs": 0, "max_pct": 0.0},
    "polylines_closed": {"max_abs": 0, "max_pct": 0.0},
    "fill_nonconvex": {"max_abs": 0, "max_pct": 0.0},
    # putText is a port of cv2 5.0's stb_truetype path (src/supervision/_cv/font): byte exact.
    "put_text_aa": {"max_abs": 0, "max_pct": 0.0},
    "add_weighted": {"max_abs": 1, "max_pct": 0.5},
    "convert_scale_abs": {"max_abs": 0, "max_pct": 0.0},
    "apply_colormap": {"max_abs": 0, "max_pct": 0.0},
    "png_roundtrip": {"max_abs": 0, "max_pct": 0.0},
}

LABEL_VARIANTS = [
    ({"text_scale": 0.8, "text_thickness": 2, "text_padding": 6}, LABELS),
    ({"text_position": "CENTER"}, LABELS),
    ({"text_position": "BOTTOM_CENTER", "text_offset": [4, -3], "text_scale": 0.4}, LABELS),
    ({"text_position": "TOP_RIGHT", "text_padding": 3}, LABELS),
    ({"text_position": "BOTTOM_LEFT", "text_scale": 1.0}, ["two\nlines", "", "a\n\nb\u00e9"]),
]

KWARGS = {
    "box": {"thickness": 2},
    "label": {},
    "mask": {"opacity": 0.5},
    "polygon": {"thickness": 2},
    "circle": {"thickness": 2},
    "dot": {"radius": 4},
    "halo": {"opacity": 0.8, "kernel_size": 40},
    "blur": {"kernel_size": 15},
    "pixelate": {"pixel_size": 10},
    "trace": {"thickness": 2, "trace_length": 30},
    "keypoints": {"radius": 4},
}


def make_input(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    grad = np.stack(
        [xx * 255 // (W - 1), yy * 255 // (H - 1), (xx + yy) * 255 // (W + H - 2)], axis=-1
    )
    noise = rng.integers(0, 256, size=(H, W, 3))
    return (0.6 * grad + 0.4 * noise).astype(np.uint8)


def masks_from_polygons(polys: list) -> np.ndarray:
    out = np.zeros((len(polys), H, W), dtype=bool)
    for i, poly in enumerate(polys):
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(poly, dtype=np.int32)], 1)
        out[i] = mask.astype(bool)
    return out


def detections() -> sv.Detections:
    return sv.Detections(
        xyxy=np.array(XYXY, dtype=np.float32),
        class_id=np.array(CLASS_ID),
        confidence=np.array(CONFIDENCE, dtype=np.float32),
        mask=masks_from_polygons(POLYGONS),
    )


def render(name: str, scene: np.ndarray) -> np.ndarray:
    dets = detections()
    if name == "box":
        return sv.BoxAnnotator(thickness=2).annotate(scene.copy(), dets)
    if name == "label":
        return sv.LabelAnnotator().annotate(scene.copy(), dets, labels=LABELS)
    if name == "mask":
        return sv.MaskAnnotator(opacity=0.5).annotate(scene.copy(), dets)
    if name == "polygon":
        return sv.PolygonAnnotator(thickness=2).annotate(scene.copy(), dets)
    if name == "circle":
        return sv.CircleAnnotator(thickness=2).annotate(scene.copy(), dets)
    if name == "dot":
        return sv.DotAnnotator(radius=4).annotate(scene.copy(), dets)
    if name == "halo":
        return sv.HaloAnnotator(opacity=0.8, kernel_size=40).annotate(scene.copy(), dets)
    if name == "blur":
        return sv.BlurAnnotator(kernel_size=15).annotate(scene.copy(), dets)
    if name == "pixelate":
        return sv.PixelateAnnotator(pixel_size=10).annotate(scene.copy(), dets)
    if name == "trace":
        ann = sv.TraceAnnotator(thickness=2, trace_length=30)
        out = scene.copy()
        for frame in TRACE_FRAMES:  # trace state accumulates across calls; the last frame is the reference
            d = sv.Detections(
                xyxy=np.array(frame, dtype=np.float32),
                class_id=np.array(CLASS_ID),
                tracker_id=np.array([1, 2, 3]),
            )
            out = ann.annotate(out, d)
        return out
    if name == "keypoints":
        kp = KeyPoints(xy=np.array(KEYPOINTS, dtype=np.float32))
        return VertexAnnotator(radius=4).annotate(scene.copy(), kp)
    raise ValueError(f"unknown primitive {name}")


def params_for(name: str) -> dict:
    p = {"palette": "default", "xyxy": XYXY, "class_id": CLASS_ID, "confidence": CONFIDENCE}
    if name in ("mask", "polygon", "halo"):
        p["polygons"] = POLYGONS
    if name == "label":
        p["labels"] = LABELS
    if name == "trace":
        p["frames"] = TRACE_FRAMES
        p["tracker_id"] = [1, 2, 3]
    if name == "keypoints":
        p["xy"] = KEYPOINTS
    p["kwargs"] = KWARGS[name]
    return p


def render_cv(name: str, scene: np.ndarray, scene2: np.ndarray, gray: np.ndarray) -> tuple[np.ndarray, dict]:
    """cv2 reference for one S5 _cv op. Returns (image, params)."""
    img = scene.copy()
    if name == "circle_outline":
        c = CIRCLE
        cv2.circle(img, tuple(c["center"]), c["radius"], c["color"], c["thickness"], cv2.LINE_8)
        return img, c
    if name == "ellipse_arc":
        e = ELLIPSE
        cv2.ellipse(img, tuple(e["center"]), tuple(e["axes"]), e["angle"], e["start"], e["end"],
                    e["color"], e["thickness"], cv2.LINE_8)
        return img, e
    if name in ("polylines_open", "polylines_closed"):
        p = POLY_OPEN if name == "polylines_open" else POLY_CLOSED
        pts = np.array(p["pts"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], p["closed"], p["color"], p["thickness"], cv2.LINE_8)
        return img, p
    if name == "fill_nonconvex":
        f = FILL_NONCONVEX
        cv2.fillPoly(img, [np.array(f["pts"], dtype=np.int32)], f["color"], cv2.LINE_8)
        return img, f
    if name == "put_text_aa":
        t = TEXT
        cv2.putText(img, t["text"], tuple(t["org"]), cv2.FONT_HERSHEY_SIMPLEX, t["scale"],
                    t["color"], t["thickness"], cv2.LINE_AA)
        return img, t
    if name == "add_weighted":
        a = ADD_WEIGHTED
        out = cv2.addWeighted(scene, a["alpha"], scene2, a["beta"], a["gamma"])
        return out, a
    if name == "convert_scale_abs":
        c = CONVERT_SCALE_ABS
        return cv2.convertScaleAbs(scene, alpha=c["alpha"], beta=c["beta"]), c
    if name == "apply_colormap":
        return cv2.applyColorMap(gray, cv2.COLORMAP_JET), {"colormap": "JET", "source": "input_gray.png"}
    if name == "png_roundtrip":
        ok, buf = cv2.imencode(".png", scene, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if not ok:
            raise RuntimeError("png encode failed")
        return cv2.imdecode(buf, cv2.IMREAD_COLOR), {"compression": 9}
    raise ValueError(f"unknown cv op {name}")


def row(op: str, args_obj: dict, expected: dict) -> str:
    return "\t".join(
        [op, json.dumps(args_obj, separators=(",", ":")), json.dumps(expected, separators=(",", ":"))]
    )


def write_png(rel: str, img: np.ndarray) -> None:
    if not cv2.imwrite(str(ROOT / rel), img):
        raise RuntimeError(f"cannot write {rel}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    header = f"# gen: tools/gen_pixels.py seed={args.seed}"

    FIXTURES.mkdir(parents=True, exist_ok=True)
    CASES.mkdir(parents=True, exist_ok=True)

    input_rel = "tests/fixtures/pixels/input.png"
    input2_rel = "tests/fixtures/pixels/input2.png"
    gray_rel = "tests/fixtures/pixels/input_gray.png"
    write_png(input_rel, make_input(args.seed))
    write_png(input2_rel, make_input(args.seed + 1))
    # Render from the PNGs as written, so the input the Mojo side decodes is the one rendered here.
    scene = cv2.imread(str(ROOT / input_rel), cv2.IMREAD_COLOR)
    scene2 = cv2.imread(str(ROOT / input2_rel), cv2.IMREAD_COLOR)
    if scene is None or scene2 is None:
        raise RuntimeError("cannot read back input PNGs")
    write_png(gray_rel, cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY))
    gray = cv2.imread(str(ROOT / gray_rel), cv2.IMREAD_GRAYSCALE)

    # Annotators (supervision oracle).
    lines = [header,
             "# bar per row in tol: integer-geometry draws exact (max_abs 0); blended or filtered output has a stated bar"]
    for name in BARS:
        out = render(name, scene)
        rel = f"tests/fixtures/pixels/{name}.png"
        write_png(rel, out)
        lines.append(row(f"pixel_{name}",
                         {"input": input_rel, "params": params_for(name), "tol": BARS[name]},
                         {"ref": rel}))
        print(f"wrote {rel} shape={out.shape}")
    # LabelAnnotator variants (text lands in src/supervision/_cv/font, bar 0): scale/thickness/padding, every anchor, offsets, multi-line
    for k, (kw, labels) in enumerate(LABEL_VARIANTS, 1):
        pos = kw.get("text_position")
        pykw = dict(kw, **({"text_position": getattr(sv.Position, pos)} if pos else {}),
                    **({"text_offset": tuple(kw["text_offset"])} if "text_offset" in kw else {}))
        dets = sv.Detections(xyxy=np.array(XYXY, dtype=np.float32), class_id=np.array(CLASS_ID), confidence=np.array(CONFIDENCE, dtype=np.float32))
        out = sv.LabelAnnotator(**pykw).annotate(scene.copy(), dets, labels=labels)
        rel = f"tests/fixtures/pixels/label_{k}.png"
        write_png(rel, out)
        params = params_for("label")
        params["labels"], params["kwargs"] = labels, kw
        lines.append(row("pixel_label", {"input": input_rel, "params": params, "tol": BARS["label"]}, {"ref": rel}))
        print(f"wrote {rel} shape={out.shape}")
    (CASES / "pixels_annotators.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # S5 _cv ops (cv2 oracle), PNG only.
    cv_lines = [header,
                "# S5 _cv ops vs cv2; bar per row in tol. Integer draws exact; AA text and blends have stated bars"]
    for name in CV_BARS:
        out, params = render_cv(name, scene, scene2, gray)
        rel = f"tests/fixtures/pixels/cv_{name}.png"
        write_png(rel, out)
        second = input2_rel if name == "add_weighted" else None
        row_args = {"input": gray_rel if name == "apply_colormap" else input_rel,
                    "params": params, "tol": CV_BARS[name]}
        if second:
            row_args["input2"] = second
        cv_lines.append(row(f"cv_{name}", row_args, {"ref": rel}))
        print(f"wrote {rel} shape={out.shape}")
    # getTextSize is numeric, not pixels: expected [width, height, baseline] from cv2.
    (tw, th), base = cv2.getTextSize(TEXT_SIZE_CASE["text"], cv2.FONT_HERSHEY_SIMPLEX,
                                     TEXT_SIZE_CASE["scale"], TEXT_SIZE_CASE["thickness"])
    cv_lines.append(row("cv_text_size",
                        {"params": TEXT_SIZE_CASE, "tol": {"max_abs": 0, "max_pct": 0.0}},
                        {"value": [int(tw), int(th), int(base)]}))
    (CASES / "pixels_cv.txt").write_text("\n".join(cv_lines) + "\n", encoding="utf-8")

    print(f"wrote tests/cases/pixels_annotators.txt rows={len(BARS)}")
    print(f"wrote tests/cases/pixels_cv.txt rows={len(CV_BARS) + 1}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # FAIL <step>: <reason>, non-zero exit
        print(f"FAIL gen_pixels: {exc}", file=sys.stderr)
        sys.exit(1)

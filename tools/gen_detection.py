"""Oracle rows for detection core + utils (H2).

Writes tests/cases/detection_{iou,nms,detections,converters}.txt.
Row format (boss contract, exchange/team.md):
  <op>\t<args JSON>\t<expected JSON>; '#' lines are comments; line 1 names this file.
Floats compare abs+rel 1e-6. NaN is written as the string "nan".

Run: PYTHONPATH=.work/supervision-develop/src .work/oracle-venv/bin/python tools/gen_detection.py
Seed: SEED below (numpy default_rng). One rng drives every family in a fixed order.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

import supervision as sv
from supervision.dataset.utils import approximate_mask_with_polygons
from supervision.detection.utils.masks import contains_holes, contains_multiple_segments
from supervision.detection.utils.polygons import approximate_polygon
from supervision.detection.utils.converters import (
    mask_to_polygons,
    mask_to_xyxy,
    polygon_to_mask,
    xcycwh_to_xyxy,
    xywh_to_xyxy,
    xyxy_to_mask,
    xyxy_to_xcycarh,
    xyxy_to_xywh,
)
from supervision.detection.utils.iou_and_nms import (
    OverlapMetric,
    box_iou,
    box_iou_batch,
    box_non_max_merge,
    box_non_max_suppression,
    mask_iou_batch,
)

SEED = 1729
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "cases"
GEN = "tools/gen_detection.py"
rng = np.random.default_rng(SEED)


def jsonable(v):
    if isinstance(v, np.ndarray):
        return jsonable(v.tolist())
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: jsonable(x) for k, x in v.items()}
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return "nan" if math.isnan(f) else f
    return v


def row(op, args, expected):
    return f"{op}\t{json.dumps(jsonable(args), separators=(',', ':'))}\t{json.dumps(jsonable(expected), separators=(',', ':'))}"


def boxes(n, lo=0.0, span=40.0):
    x1 = rng.uniform(lo, lo + span, n)
    y1 = rng.uniform(lo, lo + span, n)
    w = rng.uniform(0.5, 25.0, n)
    h = rng.uniform(0.5, 25.0, n)
    return np.round(np.stack([x1, y1, x1 + w, y1 + h], axis=1), 3)


METRICS = {"IOU": OverlapMetric.IOU, "IOS": OverlapMetric.IOS}


def iou_rows():
    rows = []
    # pairwise box_iou: fixed edge cases, then random pairs in both metrics
    edge = [
        ([0, 0, 10, 10], [0, 0, 10, 10]),  # identical
        ([0, 0, 10, 10], [20, 20, 30, 30]),  # disjoint
        ([0, 0, 10, 10], [10, 0, 20, 10]),  # touching edge
        ([0, 0, 10, 10], [2, 2, 5, 5]),  # contained
        ([0, 0, 0, 0], [0, 0, 10, 10]),  # zero-area true box
        ([0, 0, 0, 0], [0, 0, 0, 0]),  # both zero-area
        ([-5, -5, 5, 5], [0, 0, 10, 10]),  # negative coords
        ([0.5, 0.5, 1.5, 1.5], [1, 1, 2, 2]),  # sub-pixel overlap
    ]
    for a, b in edge:
        for m in ("IOU", "IOS"):
            rows.append(("box_iou", {"box_true": a, "box_detection": b, "overlap_metric": m},
                         float(box_iou(a, b, METRICS[m]))))
    for _ in range(56):
        a = boxes(1)[0]
        b = boxes(1)[0]
        m = "IOU" if rng.random() < 0.5 else "IOS"
        rows.append(("box_iou", {"box_true": a, "box_detection": b, "overlap_metric": m},
                     float(box_iou(a, b, METRICS[m]))))
    # batch matrices
    for _ in range(40):
        n, k = int(rng.integers(1, 7)), int(rng.integers(1, 7))
        a, b = boxes(n), boxes(k)
        m = "IOU" if rng.random() < 0.5 else "IOS"
        rows.append(("box_iou_batch", {"boxes_true": a, "boxes_detection": b, "overlap_metric": m},
                     box_iou_batch(a, b, METRICS[m])))
    # mask IoU on small dense masks; includes empty masks (zero union)
    for i in range(44):
        h, w = int(rng.integers(4, 13)), int(rng.integers(4, 13))
        n, k = int(rng.integers(1, 5)), int(rng.integers(1, 5))
        mt = rng.random((n, h, w)) < 0.5
        md = rng.random((k, h, w)) < 0.5
        if i == 0:
            mt[:] = False
            md[:] = False
        if i == 1:
            md[0] = mt[0]
        m = "IOU" if i % 3 else "IOS"
        rows.append(("mask_iou_batch", {"masks_true": mt, "masks_detection": md, "overlap_metric": m},
                     mask_iou_batch(mt, md, METRICS[m])))
    return rows


def nms_rows():
    rows = []
    for i in range(52):
        n = int(rng.integers(1, 12))
        xyxy = boxes(n, span=30.0)
        conf = np.round(rng.random(n), 3)
        cls = rng.integers(0, 3, n).astype(float)
        pred = np.column_stack([xyxy, conf, cls])
        thr = float(rng.choice([0.3, 0.5, 0.7]))
        m = "IOU" if i % 4 else "IOS"
        rows.append(("box_non_max_suppression",
                     {"predictions": pred, "iou_threshold": thr, "overlap_metric": m},
                     box_non_max_suppression(pred, thr, METRICS[m])))
    for i in range(44):
        n = int(rng.integers(1, 10))
        xyxy = boxes(n, span=20.0)
        conf = np.round(rng.random(n), 3)
        cls = np.zeros(n)
        pred = np.column_stack([xyxy, conf, cls])
        thr = float(rng.choice([0.3, 0.5, 0.7]))
        m = "IOU" if i % 4 else "IOS"
        rows.append(("box_non_max_merge",
                     {"predictions": pred, "iou_threshold": thr, "overlap_metric": m},
                     box_non_max_merge(pred, thr, METRICS[m])))
    return rows


def detections_rows():
    rows = []
    for _ in range(12):
        n = int(rng.integers(1, 9))
        xyxy = boxes(n)
        conf = np.round(rng.random(n), 3)
        d = sv.Detections(xyxy=xyxy, confidence=conf)
        rows.append(("detections_area", {"xyxy": xyxy}, d.area))
    rows.append(("detections_area", {"xyxy": np.zeros((0, 4))}, sv.Detections.empty().area))
    for _ in range(14):
        n = int(rng.integers(1, 9))
        xyxy = boxes(n)
        conf = np.round(rng.random(n), 3)
        keep = rng.random(n) < 0.5
        d = sv.Detections(xyxy=xyxy, confidence=conf)[keep]
        rows.append(("detections_filter", {"xyxy": xyxy, "confidence": conf, "keep": keep},
                     {"xyxy": d.xyxy, "confidence": d.confidence}))
    for _ in range(12):
        parts = []
        for _ in range(int(rng.integers(1, 4))):
            k = int(rng.integers(1, 5))
            parts.append(sv.Detections(xyxy=boxes(k), confidence=np.round(rng.random(k), 3)))
        merged = sv.Detections.merge(parts)
        rows.append(("detections_merge",
                     {"parts": [{"xyxy": p.xyxy, "confidence": p.confidence} for p in parts]},
                     {"xyxy": merged.xyxy, "confidence": merged.confidence}))
    return rows


def converter_rows():
    rows = []
    for _ in range(22):
        n = int(rng.integers(1, 8))
        xyxy = boxes(n)
        xywh = xyxy_to_xywh(xyxy)
        rows.append(("xyxy_to_xywh", {"xyxy": xyxy}, xywh))
        rows.append(("xywh_to_xyxy", {"xywh": xywh}, xywh_to_xyxy(xywh)))
    for _ in range(20):
        n = int(rng.integers(1, 8))
        xyxy = boxes(n)
        cxcywh = np.column_stack([
            (xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2,
            xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]])
        rows.append(("xcycwh_to_xyxy", {"xcycwh": cxcywh}, xcycwh_to_xyxy(cxcywh)))
    for _ in range(10):
        xyxy = boxes(int(rng.integers(1, 6)))
        rows.append(("xyxy_to_xcycarh", {"xyxy": xyxy}, xyxy_to_xcycarh(xyxy)))
    # polygons -> masks -> boxes; the polygon is a random convex-ish quad
    for _ in range(24):
        w, h = int(rng.integers(8, 40)), int(rng.integers(8, 40))
        cx, cy = rng.uniform(0.3 * w, 0.7 * w), rng.uniform(0.3 * h, 0.7 * h)
        r = rng.uniform(2, 0.45 * min(w, h), 4)
        ang = np.sort(rng.uniform(0, 2 * np.pi, 4))
        poly = np.round(np.column_stack([cx + r * np.cos(ang), cy + r * np.sin(ang)]), 2)
        mask = polygon_to_mask(poly, (w, h))
        rows.append(("polygon_to_mask", {"polygon": poly, "resolution_wh": [w, h]}, mask))
        rows.append(("mask_to_xyxy", {"masks": mask[None]}, mask_to_xyxy(mask[None])))
    for _ in range(12):
        w, h = int(rng.integers(8, 30)), int(rng.integers(8, 30))
        xyxy = boxes(int(rng.integers(1, 4)), span=min(w, h) * 0.6)
        xyxy = np.round(np.clip(xyxy, 0, [w - 1, h - 1, w - 1, h - 1]), 0)
        masks = xyxy_to_mask(xyxy, (w, h))
        rows.append(("xyxy_to_mask", {"xyxy": xyxy, "resolution_wh": [w, h]}, masks))
    # masks -> polygons: random blobs (discs and boxes, some carved out), then the helpers the COCO export uses
    for k in range(36):
        w, h = int(rng.integers(10, 44)), int(rng.integers(10, 44))
        yy, xx = np.mgrid[0:h, 0:w]
        mask = np.zeros((h, w), bool)
        for _ in range(int(rng.integers(1, 4))):
            cx, cy, r = rng.uniform(2, w - 2), rng.uniform(2, h - 2), rng.uniform(1.5, 0.35 * min(w, h))
            if rng.random() < 0.5:
                mask |= (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
            else:
                mask |= (abs(xx - cx) <= r) & (abs(yy - cy) <= r * rng.uniform(0.4, 1.0))
        if k % 3 == 0:  # a hole
            cx, cy, r = w / 2, h / 2, 0.15 * min(w, h)
            mask &= ~((xx - cx) ** 2 + (yy - cy) ** 2 <= r * r)
        m8 = mask.astype(np.uint8)
        polys = mask_to_polygons(mask)
        rows.append(("mask_to_polygons", {"mask": m8}, polys))
        rows.append(("contains_holes", {"mask": m8}, bool(contains_holes(mask))))
        rows.append(("contains_multiple_segments", {"mask": m8, "connectivity": 4 if k % 2 == 0 else 8},
                     bool(contains_multiple_segments(mask, connectivity=4 if k % 2 == 0 else 8))))
        for pol in polys[:2]:
            pct = float(rng.choice([0.0, 0.25, 0.5, 0.75, 0.9]))
            rows.append(("approximate_polygon", {"polygon": pol, "percentage": pct}, approximate_polygon(pol, pct)))
        if mask.any():
            lo, hi, pct = float(rng.choice([0.0, 0.02, 0.1])), float(rng.choice([0.5, 1.0])), float(rng.choice([0.0, 0.5, 0.75]))
            rows.append(("approximate_mask_with_polygons",
                         {"mask": m8, "min_image_area_percentage": lo, "max_image_area_percentage": hi, "approximation_percentage": pct},
                         approximate_mask_with_polygons(mask, lo, hi, pct)))
    return rows


FILES = {
    "detection_iou.txt": (iou_rows, "iou"),
    "detection_nms.txt": (nms_rows, "nms"),
    "detection_detections.txt": (detections_rows, "detections"),
    "detection_converters.txt": (converter_rows, "converters"),
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    total = 0
    for name, (fn, _) in FILES.items():
        rows = fn()
        lines = [f"# gen: {GEN} seed={SEED}"]
        lines += [row(*r) for r in rows]
        (OUT / name).write_text("\n".join(lines) + "\n")
        total += len(rows)
        print(f"{name}: {len(rows)} rows", file=sys.stderr)
    print(f"total: {total} rows", file=sys.stderr)
    if total < 300:
        sys.exit("FAIL rows: fewer than 300 detection rows")


if __name__ == "__main__":
    main()

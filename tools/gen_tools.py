#!/usr/bin/env python3
"""Oracle rows for the detection tools (S4). Run: .work/oracle-venv/bin/python -I tools/gen_tools.py
Row: <op>\t<args JSON>\t<expected JSON>; line 1 = '# gen: tools/gen_tools.py seed=<n>'.
Errors: expected {"error": "<PythonExceptionName>"}."""
import json, os, random, sys

import numpy as np

import warnings

from supervision import Detections, Point, Position
from supervision.detection.line_zone import LineZone
from supervision.detection.tools.csv_sink import CSVSink
from supervision.detection.tools.inference_slicer import InferenceSlicer
from supervision.detection.tools.json_sink import JSONSink
from supervision.detection.tools.polygon_zone import PolygonZone
from supervision.detection.tools.smoother import DetectionsSmoother

warnings.simplefilter("ignore")

SEED = 4
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "cases")


def dump(x):
    return json.dumps(x, separators=(",", ":"), allow_nan=False)


def write(name, rows):
    path = os.path.join(OUT, f"{name}.txt")
    with open(path, "w") as f:
        f.write(f"# gen: tools/gen_tools.py seed={SEED}\n")
        for op, args, exp in rows:
            f.write(f"{op}\t{dump(args)}\t{dump(exp)}\n")
    print(f"{name}: {len(rows)} rows")


def tup(v):
    return [int(v[0]), int(v[1])]


def slicer_rows(rng):
    rows = []

    def add(image, sl, ov):
        args = {"image_wh": image, "slice_wh": sl, "overlap_wh": ov}
        try:
            s = InferenceSlicer(callback=lambda x: None, slice_wh=tuple(sl), overlap_wh=tuple(ov))
            off = InferenceSlicer._generate_offset(tuple(image), s.slice_wh, s.overlap_wh)
            rows.append(("slicer_offsets", args, [[int(v) for v in r] for r in off]))
        except ValueError:
            rows.append(("slicer_offsets", args, {"error": "ValueError"}))

    # hand-picked edges: smaller image, exact fit, zero overlap, overlap >= slice, last tile flush
    for image, sl, ov in [
        ([1000, 700], [640, 640], [100, 100]), ([100, 100], [640, 640], [100, 100]),
        ([640, 640], [640, 640], [0, 0]), ([641, 641], [640, 640], [0, 0]),
        ([1280, 640], [640, 640], [0, 0]), ([1280, 640], [640, 640], [639, 0]),
        ([640, 640], [640, 640], [640, 0]), ([640, 640], [640, 640], [0, 640]),
        ([1, 1], [1, 1], [0, 0]), ([5, 3], [2, 2], [1, 1]), ([7, 7], [3, 3], [2, 2]),
        ([1920, 1080], [640, 640], [128, 128]), ([1920, 1080], [960, 540], [0, 0]),
        ([10, 10], [4, 6], [3, 5]),
    ]:
        add(image, sl, ov)
    for _ in range(140):
        sw, sh = rng.randint(1, 64), rng.randint(1, 64)
        ow = rng.randint(0, sw - 1) if rng.random() < 0.95 else rng.randint(sw, sw + 3)
        oh = rng.randint(0, sh - 1) if rng.random() < 0.95 else rng.randint(sh, sh + 3)
        add([rng.randint(1, 200), rng.randint(1, 200)], [sw, sh], [ow, oh])
    for bad in ([0, 5], [5, 0], [-1, 5]):
        add([20, 20], bad, [0, 0])
    add([20, 20], [5, 5], [-1, 0])
    return rows


ALL_ANCHORS = ["TOP_LEFT", "TOP_RIGHT", "BOTTOM_LEFT", "BOTTOM_RIGHT"]
ANCHOR_POOL = ["CENTER", "CENTER_LEFT", "CENTER_RIGHT", "TOP_CENTER", "TOP_LEFT", "TOP_RIGHT",
               "BOTTOM_LEFT", "BOTTOM_CENTER", "BOTTOM_RIGHT"]


def per_class(counter):
    return [[None if k is None else int(k), int(v)] for k, v in counter.items()]


def line_case(rng, frames, start, end, anchors, thr):
    args = {"start": start, "end": end, "anchors": anchors, "min_threshold": thr, "frames": frames}
    try:
        z = LineZone(Point(*start), Point(*end), triggering_anchors=[Position[a] for a in anchors],
                     minimum_crossing_threshold=thr)
    except ValueError:
        return ("line_zone", args, {"error": "ValueError"})
    out = []
    for f in frames:
        xyxy = np.array(f["xyxy"], dtype=float).reshape(-1, 4)
        d = Detections(xyxy=xyxy,
                       tracker_id=None if f["tracker_id"] is None else np.array(f["tracker_id"], dtype=int),
                       class_id=None if f["class_id"] is None else np.array(f["class_id"], dtype=int))
        ci, co = z.trigger(d)
        out.append({"in": [bool(v) for v in ci], "out": [bool(v) for v in co]})
    out.append({"in_count": int(z.in_count), "out_count": int(z.out_count),
                "in_per_class": per_class(z._in_count_per_class),
                "out_per_class": per_class(z._out_count_per_class)})
    return ("line_zone", args, out)


def line_rows(rng):
    rows = []
    for k in range(70):
        start = [float(rng.randint(0, 100)), float(rng.randint(0, 100))]
        end = [float(rng.randint(0, 100)), float(rng.randint(0, 100))]
        if k == 0:
            end = list(start)  # zero magnitude -> error
        anchors = ALL_ANCHORS if rng.random() < 0.4 else rng.sample(ANCHOR_POOL, rng.randint(1, 3))
        thr = rng.choice([0, 1, 1, 2, 3])
        ntr = rng.randint(1, 5)
        cx, cy = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        pos = {t: [cx + rng.uniform(-40, 40), cy + rng.uniform(-40, 40)] for t in range(ntr)}
        # aim each track at a point near the line centre so most of them cross it
        vel = {t: [(cx + rng.uniform(-8, 8) - pos[t][0]) / 8 * rng.uniform(0.6, 1.4),
                   (cy + rng.uniform(-8, 8) - pos[t][1]) / 8 * rng.uniform(0.6, 1.4)] for t in range(ntr)}
        cls = {t: rng.randint(0, 2) for t in range(ntr)}
        use_cls = rng.random() < 0.7
        frames = []
        for fi in range(rng.randint(1, 30) if k % 5 else 30):
            ids = [t for t in range(ntr) if rng.random() < 0.85]
            xyxy, tids = [], []
            for t in ids:
                pos[t][0] += vel[t][0]; pos[t][1] += vel[t][1]
                w, h = rng.uniform(2, 20), rng.uniform(2, 20)
                xyxy.append([pos[t][0] - w / 2, pos[t][1] - h / 2, pos[t][0] + w / 2, pos[t][1] + h / 2])
                tids.append(-1 if rng.random() < 0.05 else t)
            no_tracker = rng.random() < 0.05
            frames.append({"xyxy": xyxy, "tracker_id": None if no_tracker else tids,
                           "class_id": [cls[t] for t in ids] if use_cls else None})
        rows.append(line_case(rng, frames, start, end, anchors, thr))
    return rows


WORDS = ["a", "cat", "x y", "a,b", 'say "hi"', "line\nbreak", "tab\tsep", "caf\u00e9", "", " pad "]


def py_value(v):
    """Args encoding -> Python: {"nd": [...]} ndarray, bare array list, else scalar."""
    if isinstance(v, dict):
        return np.array(v["nd"])
    return v


def sink_append(rng, n):
    a = {"xyxy": [[round(rng.uniform(0, 50), 3) for _ in range(4)] for _ in range(n)]}
    a["class_id"] = [rng.randint(0, 5) for _ in range(n)] if rng.random() < 0.7 else None
    a["confidence"] = [round(rng.uniform(0, 1), 4) for _ in range(n)] if rng.random() < 0.7 else None
    a["tracker_id"] = [rng.randint(0, 9) for _ in range(n)] if rng.random() < 0.5 else None
    data = {}
    for key in rng.sample(["class_name", "score2", "emb", "note"], rng.randint(0, 3)):
        if key == "class_name" or key == "note":
            data[key] = {"nd": [rng.choice(WORDS) for _ in range(n)]}
        elif key == "emb":
            data[key] = {"nd": [[round(rng.uniform(0, 9), 2) for _ in range(3)] for _ in range(n)]}
        else:
            data[key] = {"nd": [round(rng.uniform(0, 9), 3) for _ in range(n)]}
    a["data"] = data
    custom = {}
    for key in rng.sample(["frame", "src", "tag", "ids", "names", "extra", "flag", "score2"], rng.randint(0, 4)):
        kind = rng.choice(["int", "float", "str", "bool", "list", "ndn", "ndother"])
        if kind == "int": custom[key] = rng.randint(-5, 500)
        elif kind == "float": custom[key] = round(rng.uniform(-9, 9), 3)
        elif kind == "str": custom[key] = rng.choice(WORDS)
        elif kind == "bool": custom[key] = rng.random() < 0.5
        elif kind == "list": custom[key] = [rng.randint(0, 99) for _ in range(n)]
        elif kind == "ndn": custom[key] = {"nd": [rng.choice(WORDS) for _ in range(n)]}
        else: custom[key] = {"nd": [rng.randint(0, 9) for _ in range(n + 2)]}
    a["custom"] = custom
    return a


def py_detections(a):
    xyxy = np.array(a["xyxy"], dtype=float).reshape(-1, 4)
    data = {k: py_value(v) for k, v in a["data"].items()}
    return Detections(xyxy=xyxy, class_id=None if a["class_id"] is None else np.array(a["class_id"], dtype=int),
                      confidence=None if a["confidence"] is None else np.array(a["confidence"], dtype=float),
                      tracker_id=None if a["tracker_id"] is None else np.array(a["tracker_id"], dtype=int),
                      data=data)


def sink_rows(rng):
    rows = []
    os.makedirs(".work/sink-oracle", exist_ok=True)
    for k in range(60):
        op = "sink_csv" if k % 2 == 0 else "sink_json"
        appends = [sink_append(rng, rng.choice([0, 1, 1, 2, 3, 4])) for _ in range(rng.randint(0, 3))]
        path = f".work/sink-oracle/{k}"
        sink = CSVSink(path) if op == "sink_csv" else JSONSink(path)
        sink.open()
        for a in appends:
            sink.append(py_detections(a), {key: py_value(v) for key, v in a["custom"].items()})
        sink.close() if op == "sink_csv" else sink.write_and_close()
        with open(path, newline="") as f:
            rows.append((op, {"appends": appends}, f.read()))
    return rows


def obb_corners(rng, cx, cy):
    """4 corners of a rotated rectangle, starting corner and winding chosen at random."""
    w, h, a = rng.uniform(2, 12), rng.uniform(2, 12), rng.uniform(0, 6.283)
    c, s_ = np.cos(a), np.sin(a)
    pts = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    pts = [[cx + x * c - y * s_, cy + x * s_ + y * c] for x, y in pts]
    k = rng.randint(0, 3)
    pts = pts[k:] + pts[:k]
    return pts[::-1] if rng.random() < 0.5 else pts


def smoother_rows(rng):
    rows = []
    for k in range(60):
        length = rng.choice([1, 2, 3, 5, 5, 8])
        ntr = rng.randint(1, 5)
        use_cls = rng.random() < 0.7
        conf_mode = rng.choice(["all", "all", "none", "some"])
        obb_mode = rng.choice(["none", "none", "all", "some"])
        pos = {t: [rng.uniform(0, 80), rng.uniform(0, 80)] for t in range(ntr)}
        vel = {t: [rng.uniform(-3, 3), rng.uniform(-3, 3)] for t in range(ntr)}
        cls = {t: rng.randint(0, 3) for t in range(ntr)}
        frames = []
        for fi in range(rng.randint(1, 14)):
            ids = [t for t in range(ntr) if rng.random() < 0.8]
            xyxy, tids, obb = [], [], []
            for t in ids:
                pos[t][0] += vel[t][0]; pos[t][1] += vel[t][1]
                w, h = rng.uniform(4, 20), rng.uniform(4, 20)
                xyxy.append([pos[t][0], pos[t][1], pos[t][0] + w, pos[t][1] + h])
                tids.append(t)
                obb.append(obb_corners(rng, pos[t][0] + w / 2, pos[t][1] + h / 2))
            if ids and rng.random() < 0.15:  # same tracker id twice in a frame
                xyxy.append(list(xyxy[0])); tids.append(tids[0]); obb.append(obb[0])
                ids.append(ids[0])
            has_conf = conf_mode == "all" or (conf_mode == "some" and rng.random() < 0.6)
            has_obb = obb_mode == "all" or (obb_mode == "some" and rng.random() < 0.6)
            frames.append({
                "xyxy": xyxy,
                "confidence": [round(rng.uniform(0, 1), 3) for _ in ids] if has_conf else None,
                "class_id": [cls[t] for t in ids] if use_cls else None,
                "tracker_id": None if rng.random() < 0.07 else tids,
                "obb": obb if has_obb else None,
            })
        args = {"length": length, "frames": frames}
        try:
            sm = DetectionsSmoother(length)
            out = []
            for f in frames:
                n = len(f["xyxy"])
                d = Detections(
                    xyxy=np.array(f["xyxy"], dtype=float).reshape(-1, 4),
                    confidence=None if f["confidence"] is None else np.array(f["confidence"], dtype=float),
                    class_id=None if f["class_id"] is None else np.array(f["class_id"], dtype=int),
                    tracker_id=None if f["tracker_id"] is None else np.array(f["tracker_id"], dtype=int),
                    data={} if f["obb"] is None else {"xyxyxyxy": np.array(f["obb"], dtype=float).reshape(n, 4, 2)},
                )
                r = sm.update_with_detections(d)
                out.append({
                    "xyxy": r.xyxy.tolist(),
                    "confidence": None if r.confidence is None else r.confidence.tolist(),
                    "class_id": None if r.class_id is None else [int(v) for v in r.class_id],
                    "tracker_id": None if r.tracker_id is None else [int(v) for v in r.tracker_id],
                    "obb": [[v for pt in c for v in pt] for c in r.data["xyxyxyxy"]] if "xyxyxyxy" in r.data else None,
                })
            rows.append(("smoother", args, out))
        except Exception:
            rows.append(("smoother", args, {"error": "Exception"}))
    return rows


def zone_rows(rng):
    """The mask is the oracle's own (cv2.fillPoly via polygon_to_mask): these rows test trigger(), not the fill."""
    rows = []
    for k in range(60):
        npts = 2 if k == 0 else rng.randint(3, 8)
        polygon = [[rng.randint(0, 40), rng.randint(0, 40)] for _ in range(npts)]
        anchors = [] if k == 1 else rng.sample(ANCHOR_POOL[:], rng.randint(1, 3)) if rng.random() < 0.8 else ["BOTTOM_CENTER"]
        require_all = rng.random() < 0.5
        frames = []
        for _ in range(rng.randint(1, 4)):
            boxes = []
            for _ in range(rng.choice([0, 1, 3, 6, 10])):
                x, y = rng.uniform(-6, 42), rng.uniform(-6, 42)
                boxes.append([x, y, x + rng.uniform(0, 6), y + rng.uniform(0, 6)])
            frames.append(boxes)
        args = {"polygon": polygon, "anchors": anchors, "require_all": require_all, "mask": [], "frames": frames}
        try:
            z = PolygonZone(np.array(polygon), triggering_anchors=[Position[a] for a in anchors],
                            require_all_anchors=require_all)
            args["mask"] = z.mask.astype(int).tolist()
            out = []
            for boxes in frames:
                hit = z.trigger(Detections(xyxy=np.array(boxes, dtype=float).reshape(-1, 4)))
                out.append({"hits": [bool(v) for v in hit], "count": int(z.current_count)})
            rows.append(("polygon_zone", args, out))
        except ValueError:
            rows.append(("polygon_zone", args, {"error": "ValueError"}))
    return rows


def slicer_merge_rows(rng):
    """InferenceSlicer end to end with a callback that replays the row's per-slice detections in call order."""
    rows = []
    for k in range(60):
        image = [rng.randint(20, 200), rng.randint(20, 200)]
        sl = [rng.randint(10, 80), rng.randint(10, 80)]
        ov = [rng.randint(0, sl[0] - 1), rng.randint(0, sl[1] - 1)]
        filt = rng.choice(["NON_MAX_SUPPRESSION", "NON_MAX_SUPPRESSION", "NON_MAX_MERGE", "NONE"])
        metric = rng.choice(["IOU", "IOS"])
        iou = rng.choice([0.3, 0.5, 0.7])
        has_conf, has_cls = rng.random() < 0.9, rng.random() < 0.8
        offsets = InferenceSlicer._generate_offset(tuple(image), tuple(sl), tuple(ov))
        slices = []
        for o in offsets:
            w, h = int(o[2] - o[0]), int(o[3] - o[1])
            n = rng.choice([0, 1, 1, 2, 3])
            xyxy = []
            for _ in range(n):
                x, y = rng.uniform(0, w * 0.8), rng.uniform(0, h * 0.8)
                xyxy.append([x, y, min(w, x + rng.uniform(2, w * 0.6)), min(h, y + rng.uniform(2, h * 0.6))])
            slices.append({"xyxy": xyxy,
                           "confidence": [rng.uniform(0.1, 1) for _ in range(n)] if has_conf else None,  # no rounding: ties make argsort order matter
                           "class_id": [rng.randint(0, 2) for _ in range(n)] if has_cls else None})
        args = {"image_wh": image, "slice_wh": sl, "overlap_wh": ov, "filter": filt, "metric": metric, "iou": iou,
                "slices": slices}
        it = iter(slices)

        def callback(_img):
            f = next(it)
            return Detections(xyxy=np.array(f["xyxy"], dtype=float).reshape(-1, 4),
                              confidence=None if f["confidence"] is None else np.array(f["confidence"], dtype=float),
                              class_id=None if f["class_id"] is None else np.array(f["class_id"], dtype=int))

        try:
            r = InferenceSlicer(callback, slice_wh=tuple(sl), overlap_wh=tuple(ov), overlap_filter=filt,
                                iou_threshold=iou, overlap_metric=metric)(np.zeros((image[1], image[0], 3), np.uint8))
            rows.append(("slicer_merge", args, {
                "xyxy": r.xyxy.tolist(),
                "confidence": None if r.confidence is None else r.confidence.tolist(),
                "class_id": None if r.class_id is None else [int(v) for v in r.class_id]}))
        except Exception:
            rows.append(("slicer_merge", args, {"error": "ValueError"}))
    return rows


def main():
    rng = random.Random(SEED)
    os.makedirs(OUT, exist_ok=True)
    write("tools_slicer", slicer_rows(rng))
    write("tools_line", line_rows(rng))
    rows = sink_rows(rng)
    write("tools_sinks", rows)
    write("tools_smoother", smoother_rows(rng))
    write("tools_zone", zone_rows(rng))
    write("tools_slicer_merge", slicer_merge_rows(rng))


if __name__ == "__main__":
    main()

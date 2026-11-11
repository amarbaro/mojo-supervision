"""Oracle rows for the S1 surface H2 does not cover (S1).

Writes tests/cases/s1_detections.txt, s1_utils.txt. Row format: tests/README.md.
Run: PYTHONPATH=.work/supervision-develop/src .work/oracle-venv/bin/python tools/gen_s1.py
Seed: SEED (numpy default_rng). Confidences are drawn distinct, so argsort ties never decide a row.
"""

import sys
from pathlib import Path

import numpy as np

import supervision as sv
from oracle import write_rows
from supervision.classification.core import Classifications
from supervision.detection.utils.converters import mask_to_rle, rle_to_mask
from supervision.detection.utils.iou_and_nms import OverlapMetric, oriented_box_iou_batch
from supervision.detection.utils.masks import (
    calculate_masks_centroids,
    count_mask_pixels,
    mask_to_roi,
    move_masks,
    resize_masks,
)
from supervision.detection.utils.matching import match_detections
from supervision.geometry.core import Position

SEED = 20261104
rng = np.random.default_rng(SEED)
ROOT = Path(__file__).resolve().parent.parent


def boxes(n, size=40.0):
    x1 = rng.uniform(0, size, n)
    y1 = rng.uniform(0, size, n)
    w = rng.uniform(2, size / 2, n)
    h = rng.uniform(2, size / 2, n)
    return np.round(np.stack([x1, y1, x1 + w, y1 + h], axis=1), 3)


def confs(n):
    return np.round(rng.permutation(np.linspace(0.05, 0.99, n)), 4)


def masks_from(xyxy, hw):
    m = np.zeros((len(xyxy), *hw), dtype=bool)
    for i, (a, b, c, d) in enumerate(xyxy):
        m[i, int(b) : int(d) + 1, int(a) : int(c) + 1] = True
    return m


def det_json(d):
    out = {"xyxy": d.xyxy, "confidence": d.confidence}
    if d.class_id is not None:
        out["class_id"] = d.class_id
    if d.mask is not None:
        out["mask"] = d.mask
    return out


def det_rows():
    rows = []
    for k in range(60):
        n = int(rng.integers(1, 9))
        hw = (24, 24)
        xyxy = boxes(n, 20.0)
        use_mask = k % 3 == 0
        agnostic = k % 4 == 1
        obb = k % 5 == 4 and not use_mask
        metric = ["IOU", "IOS"][k % 2]
        mode = ["nms", "nmm", "soft"][k % 3]
        d = sv.Detections(
            xyxy=xyxy,
            confidence=confs(n),
            class_id=rng.integers(0, 3, n),
            mask=masks_from(xyxy, hw) if use_mask else None,
        )
        args = {
            "xyxy": d.xyxy,
            "confidence": d.confidence,
            "class_id": d.class_id,
            "mask": d.mask,
            "mode": mode,
            "threshold": float(rng.choice([0.2, 0.4, 0.5, 0.7])),
            "class_agnostic": agnostic,
            "overlap_metric": metric,
            "sigma": 0.5,
        }
        if mode == "soft" and k % 2 == 0:
            args["score_threshold"] = 0.3
        if obb:
            c = np.stack([xyxy[:, [0, 1]], xyxy[:, [2, 1]], xyxy[:, [2, 3]], xyxy[:, [0, 3]]], axis=1)
            d.data["xyxyxyxy"] = c
            args["xyxyxyxy"] = c
            args["tol"] = {"abs": 1e-4, "rel": 1e-4}
        if mode == "nms":
            out = d.with_nms(args["threshold"], agnostic, OverlapMetric.from_value(metric))
        elif mode == "nmm":
            out = d.with_nmm(args["threshold"], agnostic, OverlapMetric.from_value(metric))
        else:
            out = d.with_soft_nms(0.5, agnostic, args.get("score_threshold"))
        rows.append(("det_nms", args, det_json(out)))
    return rows


def anchor_rows():
    rows = []
    names = [p.value for p in Position]
    for k in range(40):
        n = int(rng.integers(1, 6))
        xyxy = boxes(n, 30.0)
        pos = names[k % len(names)]
        mask = masks_from(xyxy, (40, 40)) if pos == "CENTER_OF_MASS" else None
        d = sv.Detections(xyxy=xyxy, mask=mask)
        rows.append(("det_anchor", {"xyxy": xyxy, "mask": mask, "position": pos}, d.get_anchors_coordinates(Position(pos))))
    for k in range(10):  # oriented boxes
        n = int(rng.integers(1, 4))
        xyxy = boxes(n, 30.0)
        ang = rng.uniform(0, np.pi, n)
        cx, cy = (xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2
        hw_, hh_ = (xyxy[:, 2] - xyxy[:, 0]) / 2, (xyxy[:, 3] - xyxy[:, 1]) / 2
        c = np.zeros((n, 4, 2))
        for i in range(n):
            R = np.array([[np.cos(ang[i]), -np.sin(ang[i])], [np.sin(ang[i]), np.cos(ang[i])]])
            pts = np.array([[-hw_[i], -hh_[i]], [hw_[i], -hh_[i]], [hw_[i], hh_[i]], [-hw_[i], hh_[i]]]) @ R.T
            c[i] = pts + [cx[i], cy[i]]
        d = sv.Detections(xyxy=xyxy, data={"xyxyxyxy": c})
        pos = [p for p in names if p != "CENTER_OF_MASS"][k % 9]
        rows.append(("det_anchor", {"xyxy": xyxy, "xyxyxyxy": c, "position": pos}, d.get_anchors_coordinates(Position(pos))))
    return rows


def props_rows():
    rows = []
    for k in range(10):
        n = int(rng.integers(1, 6))
        xyxy = boxes(n)
        if k % 3 == 0:
            xyxy[0, 3] = xyxy[0, 1]  # zero height -> nan aspect ratio
        d = sv.Detections(xyxy=xyxy)
        rows.append(("det_props", {"xyxy": xyxy}, {"box_area": d.box_area, "aspect": d.box_aspect_ratio}))
    return rows


def merge_rows():
    rows = []
    for k in range(20):
        parts, objs = [], []
        has_mask = k % 3 == 0
        has_trk = k % 2 == 0
        hw = (16, 16)
        meta = {"source": "cam1"} if k % 4 == 0 else {}
        for _ in range(int(rng.integers(1, 4))):
            n = int(rng.integers(1, 4))
            xyxy = boxes(n, 12.0)
            obj = {
                "xyxy": xyxy,
                "confidence": confs(n),
                "class_id": rng.integers(0, 4, n),
                "data": {"score2": rng.uniform(0, 1, n).round(3), "vec": rng.uniform(0, 1, (n, 3)).round(3), "name": [f"c{i}" for i in rng.integers(0, 4, n)]},
                "metadata": meta,
            }
            if has_trk:
                obj["tracker_id"] = rng.integers(0, 50, n)
            if has_mask:
                obj["mask"] = masks_from(xyxy, hw)
            parts.append(sv.Detections(**obj))
            objs.append(obj)
        m = sv.Detections.merge(parts)
        exp = {"xyxy": m.xyxy, "confidence": m.confidence, "class_id": m.class_id, "data": m.data, "metadata": m.metadata}
        if m.tracker_id is not None:
            exp["tracker_id"] = m.tracker_id
        if m.mask is not None:
            exp["mask"] = m.mask
        rows.append(("det_merge_full", {"parts": objs}, exp))
    return rows


def select_rows():
    rows = []
    for _ in range(20):
        n = int(rng.integers(1, 7))
        obj = {
            "xyxy": boxes(n),
            "confidence": confs(n),
            "class_id": rng.integers(0, 4, n),
            "data": {"name": [f"c{i}" for i in rng.integers(0, 4, n)], "vec": rng.uniform(0, 1, (n, 2)).round(3)},
        }
        d = sv.Detections(**obj)
        idx = [int(i) for i in rng.integers(-n, n, int(rng.integers(1, 5)))]
        out = d[idx]
        rows.append(("det_select", {"det": obj, "index": idx}, {"xyxy": out.xyxy, "confidence": out.confidence, "class_id": out.class_id, "data": out.data, "metadata": out.metadata}))
    return rows


def error_rows():
    rows = []
    cases = [
        {"xyxy": [[0, 0, 1, 1]], "confidence": [0.5, 0.6]},
        {"xyxy": [[0, 0, 1, 1]], "class_id": [1, 2]},
        {"xyxy": [[0, 0, 1, 1]], "tracker_id": [1, 2, 3]},
        {"xyxy": [[0, 0, 1, 1], [1, 1, 2, 2]], "data": {"a": [1]}},
    ]
    for c in cases:
        kw = {k: (np.array(v, dtype=float) if k == "xyxy" else np.array(v) if k != "data" else {kk: np.array(vv) for kk, vv in v.items()}) for k, v in c.items()}
        try:
            sv.Detections(**kw)
        except ValueError as e:
            rows.append(("det_validate", c, {"error": str(e)}))
            continue
        raise SystemExit("expected a ValueError for " + str(c))
    return rows


def util_rows():
    rows = []
    for _ in range(20):
        n = int(rng.integers(0, 10))
        cid = rng.permutation(100)[:n]
        cf = np.round(rng.permutation(np.linspace(0.01, 0.99, max(n, 1)))[:n], 4)
        kk = int(rng.integers(0, 12))
        ids, conf = Classifications(class_id=cid, confidence=cf).get_top_k(kk)
        rows.append(("top_k", {"class_id": cid, "confidence": cf, "k": kk}, [ids, conf]))
    for k in range(25):
        na, nb = int(rng.integers(0, 7)), int(rng.integers(0, 7))
        xa, xb = boxes(na, 25.0), boxes(nb, 25.0)
        ca, cb = rng.integers(0, 3, na), rng.integers(0, 3, nb)
        thr = float(rng.choice([0.1, 0.3, 0.5]))
        ag = bool(k % 3 == 0)
        a = sv.Detections(xyxy=xa.reshape(-1, 4), class_id=ca)
        b = sv.Detections(xyxy=xb.reshape(-1, 4), class_id=cb)
        pairs, ua, ub = match_detections(a, b, thr, ag)
        rows.append(("match_detections", {"xyxy_a": xa, "class_a": ca, "xyxy_b": xb, "class_b": cb, "iou_threshold": thr, "class_agnostic": ag}, {"pairs": pairs, "ua": ua, "ub": ub}))
    for k in range(20):
        n, h, w = int(rng.integers(1, 4)), int(rng.integers(3, 14)), int(rng.integers(3, 14))
        m = rng.random((n, h, w)) < 0.35
        if k % 5 == 0:
            m[0] = False
        dx, dy = int(rng.integers(-5, 6)), int(rng.integers(-5, 6))
        res = (int(rng.integers(4, 16)), int(rng.integers(4, 16)))
        rows.append(("move_masks", {"masks": m, "offset": [dx, dy], "resolution_wh": list(res)}, move_masks(m, np.array([dx, dy]), res)))
        rows.append(("calculate_masks_centroids", {"masks": m}, calculate_masks_centroids(m)))
        rows.append(("count_mask_pixels", {"masks": m}, count_mask_pixels(m)))
        roi = mask_to_roi(m[0])
        rows.append(("mask_to_roi", {"mask": m[0]}, list(roi) if roi else None))
        rle = mask_to_rle(m[0])
        crle = mask_to_rle(m[0], compressed=True)
        rows.append(("mask_to_rle", {"mask": m[0], "compressed": False}, rle))
        rows.append(("mask_to_rle", {"mask": m[0], "compressed": True}, crle))
        rows.append(("rle_to_mask", {"rle": rle, "resolution_wh": [w, h]}, rle_to_mask(rle, (w, h))))
        rows.append(("rle_to_mask", {"rle": crle, "resolution_wh": [w, h]}, rle_to_mask(crle, (w, h))))
    for _ in range(8):
        m = rng.random((2, int(rng.integers(40, 90)), int(rng.integers(40, 90)))) < 0.4
        md = int(rng.integers(8, 30))
        rows.append(("resize_masks", {"masks": m, "max_dimension": md}, resize_masks(m, md)))
    for k in range(25):
        n, mm = int(rng.integers(1, 5)), int(rng.integers(1, 5))
        a = rng.uniform(0, 30, (n, 2)).round(2)
        b = rng.uniform(0, 30, (mm, 2)).round(2)
        a = np.stack([np.stack([x, x + [8, 0], x + [8, 6], x + [0, 6]]) for x in a])
        b = np.stack([np.stack([x, x + [7, 1], x + [6, 8], x + [-1, 7]]) for x in b])
        met = ["IOU", "IOS"][k % 2]
        rows.append(("oriented_box_iou_batch", {"boxes_true": a, "boxes_detection": b, "overlap_metric": met, "tol": {"abs": 1e-4, "rel": 1e-4}}, oriented_box_iou_batch(a, b, OverlapMetric.from_value(met))))
    return rows


def main():
    out = ROOT / "tests" / "cases"
    write_rows(out / "s1_detections.txt", "gen_s1", SEED, det_rows() + anchor_rows() + props_rows() + merge_rows() + select_rows() + error_rows())
    write_rows(out / "s1_utils.txt", "gen_s1", SEED, util_rows())


if __name__ == "__main__":
    main()

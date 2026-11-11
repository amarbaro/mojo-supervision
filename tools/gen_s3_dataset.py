#!/usr/bin/env python3
"""Stress rows for the dataset port: non-square images, float boxes, 11 images per format.

H4's fixtures are 8x8 with integer boxes, so a width/height swap or a float32 rounding bug
would pass them. This reuses tools/gen_datasets.py with other constants and writes
tests/fixtures/dataset_s3/ plus tests/cases/dataset_s3_<fmt>.txt (same ops as H4's rows).

Run: .work/oracle-venv/bin/python tools/gen_s3_dataset.py        (seed 1234)
"""
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_datasets as g
from oracle import write_rows

SEED = 1234
ROOT = g.ROOT
OUT = ROOT / "tests" / "fixtures" / "dataset_s3"


def make_boxes(rng):
    """(class_id, x1, y1, x2, y2): 0..4 boxes, float corners with 2 decimals, inside the image."""
    boxes = []
    for _ in range(rng.randrange(0, 5)):
        x1 = round(rng.uniform(0, g.W - 3), 2)
        y1 = round(rng.uniform(0, g.H - 3), 2)
        x2 = round(rng.uniform(x1 + 0.5, g.W), 2)
        y2 = round(rng.uniform(y1 + 0.5, g.H), 2)
        boxes.append((rng.randrange(len(g.CLASSES)), x1, y1, x2, y2))
    return boxes


def png_bytes(rgb):
    return g.png_bytes(rgb)


def yolo_edge(summaries):
    """YOLO label quirks: pose keypoints dropped, a trailing confidence, '1.0' class ids, a missing label file."""
    import supervision as sv
    src, dst = OUT / "yolo" / "in", OUT / "yolo_edge"
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst)
    (dst / "data.yaml").write_text("names: [zebra, ant, cat]\nkpt_shape: [2, 3]\n")
    for i, lab in enumerate(sorted((dst / "labels").glob("*.txt"))):
        lines = [l for l in lab.read_text().splitlines() if l]
        if i % 3 == 0:
            lines = [l + " 0.5 0.5 2 0.25 0.25 2" for l in lines]   # pose: 5 + 2*3 values
        elif i % 3 == 1:
            lines = [l + " 0.87" for l in lines]                      # trailing confidence
        if lines and i % 4 == 0:
            parts = lines[0].split()
            parts[0] = parts[0] + ".0"
            lines[0] = " ".join(parts)
        lab.write_text("\n".join(lines) + ("\n" if lines else ""))
    sorted((dst / "labels").glob("*.txt"))[-1].unlink()              # image without a label file
    ds = sv.DetectionDataset.from_yolo(str(dst / "images"), str(dst / "labels"), str(dst / "data.yaml"))
    return g.summarize(ds)


def main():
    g.SEED, g.W, g.H, g.N_IMAGES, g.OUT = SEED, 13, 9, 11, OUT
    g.CLASSES = ["zebra", "ant", "cat"]
    g.make_boxes = make_boxes
    shutil.rmtree(OUT, ignore_errors=True)
    g.write_inputs()
    summaries = g.run_oracle()
    g.verify_resave()
    for fmt in ("coco", "yolo", "voc"):
        base = f"tests/fixtures/dataset_s3/{fmt}"
        inp, ref = f"{base}/in", f"{base}/ref"
        params = {"format": fmt}
        rows = [
            ("load", {"input": inp, "params": params, "tol": {}}, summaries[fmt]),
            ("save_roundtrip", {"input": inp, "params": params, "tol": {}}, {"ref": f"{ref}/roundtrip.bundle"}),
        ]
        sp = dict(params, split_ratio=0.5, random_state=g.SEED, shuffle=True)
        rows.append(("split", {"input": inp, "params": sp, "tol": {}}, {"ref": f"{ref}/split.bundle"}))
        rows.append(("merge", {"input": f"{ref}/split_train", "params": dict(params, **{"with": f"{ref}/split_valid"}),
                               "tol": {}}, {"ref": f"{ref}/merge.bundle"}))
        if fmt == "yolo":
            rows.append(("load", {"input": "tests/fixtures/dataset_s3/yolo_edge", "params": params, "tol": {}},
                         yolo_edge(summaries)))
        write_rows(str(ROOT / "tests" / "cases" / f"dataset_s3_{fmt}.txt"), "gen_s3_dataset", SEED, rows)


main()

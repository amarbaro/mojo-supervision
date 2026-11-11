#!/usr/bin/env python3
"""Generate tests/fixtures/datasets/: tiny COCO, YOLO and VOC sets plus Python
supervision outputs (load -> save, split, merge) that the Mojo port must match byte-for-byte.

Step 1 (stdlib only): writes inputs with seed SEED: 4 PNG images per format, 8x8 px each,
    and annotations in that format's layout.
Step 2 (oracle, .work/oracle-venv): loads each input with Python supervision, saves it back,
    splits it, merges the halves, and writes every result under <fmt>/ref/.
Step 3 writes tests/cases/dataset_<fmt>.txt: one row each for load (value), save roundtrip,
    split and merge. File refs are bundles (`== <relpath> <nbytes>` header per file, sorted).

Run: .work/oracle-venv/bin/python tools/gen_datasets.py
Exit non-zero on any failure: `FAIL <step>: <reason>`.
"""

from __future__ import annotations

import json
import random
import shutil
import struct
import sys
import zlib
from datetime import datetime
from pathlib import Path

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "fixtures" / "datasets"
W = H = 8
N_IMAGES = 4
CLASSES = ["cat", "dog"]


def fail(step: str, reason: str) -> None:
    print(f"FAIL {step}: {reason}")
    sys.exit(1)


def png_bytes(rgb: list[tuple[int, int, int]], w: int = W, h: int = H) -> bytes:
    """Minimal RGB PNG writer (no PIL): filter byte 0 per row, zlib, CRCs."""
    raw = b"".join(b"\x00" + bytes(c for px in rgb[y * w:(y + 1) * w] for c in px) for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


def make_image(rng: random.Random) -> bytes:
    return png_bytes([(rng.randrange(256), rng.randrange(256), rng.randrange(256)) for _ in range(W * H)])


def make_boxes(rng: random.Random) -> list[tuple[int, float, float, float, float]]:
    """(class_id, x1, y1, x2, y2) in pixels, 1..2 boxes per image, inside the image."""
    boxes = []
    for _ in range(rng.randrange(1, 3)):
        x1 = rng.randrange(0, W - 2)
        y1 = rng.randrange(0, H - 2)
        x2 = rng.randrange(x1 + 1, W)
        y2 = rng.randrange(y1 + 1, H)
        boxes.append((rng.randrange(len(CLASSES)), float(x1), float(y1), float(x2), float(y2)))
    return boxes


def write_inputs() -> None:
    rng = random.Random(SEED)
    images = [(f"img_{i:03d}.png", make_image(rng), make_boxes(rng)) for i in range(N_IMAGES)]

    # COCO: one annotations.json, images/ directory
    coco_dir = OUT / "coco" / "in"
    (coco_dir / "images").mkdir(parents=True, exist_ok=True)
    coco = {"images": [], "annotations": [], "categories": []}
    for i, c in enumerate(CLASSES):
        coco["categories"].append({"id": i + 1, "name": c, "supercategory": "none"})
    ann_id = 1
    for img_id, (name, data, boxes) in enumerate(images, start=1):
        (coco_dir / "images" / name).write_bytes(data)
        coco["images"].append({"id": img_id, "file_name": name, "width": W, "height": H})
        for cls, x1, y1, x2, y2 in boxes:
            w, h = x2 - x1, y2 - y1
            coco["annotations"].append({
                "id": ann_id, "image_id": img_id, "category_id": cls + 1,
                "bbox": [x1, y1, w, h], "area": w * h, "iscrowd": 0, "segmentation": [],
            })
            ann_id += 1
    (coco_dir / "annotations.json").write_text(json.dumps(coco, indent=2) + "\n")

    # YOLO: images/, labels/, data.yaml
    yolo_dir = OUT / "yolo" / "in"
    (yolo_dir / "images").mkdir(parents=True, exist_ok=True)
    (yolo_dir / "labels").mkdir(parents=True, exist_ok=True)
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))
    (yolo_dir / "data.yaml").write_text(f"names:\n{names}\n")
    for name, data, boxes in images:
        (yolo_dir / "images" / name).write_bytes(data)
        lines = []
        for cls, x1, y1, x2, y2 in boxes:
            cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
        (yolo_dir / "labels" / (Path(name).stem + ".txt")).write_text("\n".join(lines) + "\n")

    # Pascal VOC: images/ (PNG bytes), annotations/*.xml
    voc_dir = OUT / "voc" / "in"
    (voc_dir / "images").mkdir(parents=True, exist_ok=True)
    (voc_dir / "annotations").mkdir(parents=True, exist_ok=True)
    for name, data, boxes in images:
        (voc_dir / "images" / name).write_bytes(data)
        objs = "".join(
            f"<object><name>{CLASSES[cls]}</name><pose>Unspecified</pose><truncated>0</truncated>"
            f"<difficult>0</difficult><bndbox><xmin>{int(x1)}</xmin><ymin>{int(y1)}</ymin>"
            f"<xmax>{int(x2)}</xmax><ymax>{int(y2)}</ymax></bndbox></object>"
            for cls, x1, y1, x2, y2 in boxes
        )
        xml = (f"<annotation><folder>images</folder><filename>{name}</filename>"
               f"<size><width>{W}</width><height>{H}</height><depth>3</depth></size>"
               f"<segmented>0</segmented>{objs}</annotation>\n")
        (voc_dir / "annotations" / (Path(name).stem + ".xml")).write_text(xml)


SW = SH = 24  # segmentation fixtures: 24 x 24 px images
SEG_FORMATS = ("coco_seg", "yolo_seg", "voc_seg")


def _disc(cx: float, cy: float, r: float) -> list[float]:
    """A 12-gon around (cx, cy), sub-pixel vertices (so np.round half-even cases occur)."""
    import math
    return [round(v, 2) for k in range(12) for v in (cx + r * math.cos(k * math.pi / 6), cy + r * math.sin(k * math.pi / 6))]


def write_seg_inputs() -> None:
    """tests/fixtures/datasets/{coco,yolo,voc}_seg/in: polygon and RLE segmentation, sizes 24 x 24."""
    import numpy as np
    from supervision.detection.utils.converters import mask_to_rle

    rng = random.Random(SEED + 1)
    names = [f"seg_{i:03d}.png" for i in range(4)]
    imgs = {n: png_bytes([(rng.randrange(256), rng.randrange(256), rng.randrange(256)) for _ in range(SW * SH)], SW, SH) for n in names}
    ring = np.zeros((SH, SW), bool)
    ring[4:20, 4:20] = True
    ring[9:15, 9:15] = False  # a mask with a hole
    two = np.zeros((SH, SW), bool)
    two[2:8, 2:8] = True
    two[14:22, 12:22] = True  # two separate segments
    blob = np.zeros((SH, SW), bool)
    blob[6:18, 8:14] = True
    blob[8:16, 6:18] = True
    polys = {  # image -> (class, polygon parts as flat COCO floats)
        names[0]: [(0, [_disc(8.0, 8.0, 6.3)]), (1, [[2.5, 12.5, 20.5, 12.5, 11.5, 22.5]])],
        names[1]: [(1, [[1.0, 1.0, 10.0, 1.0, 10.0, 9.0, 1.0, 9.0], [13.0, 13.0, 22.0, 13.0, 22.0, 22.0]])],
    }

    # COCO
    d = OUT / "coco_seg" / "in"
    (d / "images").mkdir(parents=True, exist_ok=True)
    coco = {"images": [], "annotations": [], "categories": [{"id": i + 1, "name": c, "supercategory": "none"} for i, c in enumerate(CLASSES)]}
    aid = 1
    for iid, n in enumerate(names, start=1):
        (d / "images" / n).write_bytes(imgs[n])
        coco["images"].append({"id": iid, "file_name": n, "width": SW, "height": SH})
    def ann(iid, cls, bbox, seg, iscrowd=None, area=None):
        nonlocal aid
        a = {"id": aid, "image_id": iid, "category_id": cls + 1, "bbox": bbox, "segmentation": seg}
        if iscrowd is not None:
            a["iscrowd"] = iscrowd
        if area is not None:
            a["area"] = area
        coco["annotations"].append(a)
        aid += 1
    for iid, n in enumerate(names[:2], start=1):
        for cls, parts in polys[n]:
            xs = [v for p in parts for v in p[0::2]]
            ys = [v for p in parts for v in p[1::2]]
            ann(iid, cls, [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)], parts, 0, 50.0)
    ann(2, 0, [0.0, 0.0, 5.0, 5.0], [])  # box only, empty segmentation: empty mask
    ann(3, 0, [4.0, 4.0, 16.0, 16.0], {"counts": mask_to_rle(ring), "size": [SH, SW]}, 1)  # uncompressed RLE, has a hole
    ann(3, 1, [2.0, 2.0, 20.0, 20.0], {"counts": mask_to_rle(two, compressed=True), "size": [SH, SW]}, 1)  # compressed RLE string
    ann(3, 1, [6.0, 6.0, 12.0, 12.0], {"counts": mask_to_rle(blob), "size": [SH, SW]})  # no iscrowd, no area
    (d / "annotations.json").write_text(json.dumps(coco, indent=2) + "\n")

    # YOLO: polygon rows, box rows, a polygon row with a trailing confidence (odd value count)
    d = OUT / "yolo_seg" / "in"
    (d / "images").mkdir(parents=True, exist_ok=True)
    (d / "labels").mkdir(parents=True, exist_ok=True)
    (d / "data.yaml").write_text("names:\n  0: cat\n  1: dog\n")
    rows = {
        names[0]: ["0 0.1 0.1 0.9 0.15 0.5 0.8", "1 0.5 0.5 0.4 0.4"],
        names[1]: ["1 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8 0.91", "0 0.3 0.3 0.2 0.2 0.5"],
        names[2]: ["0 0.5 0.5 0.4 0.4"],  # box only: no masks for this image
    }
    for n in names:
        (d / "images" / n).write_bytes(imgs[n])
        if n in rows:
            (d / "labels" / (Path(n).stem + ".txt")).write_text("\n".join(rows[n]) + "\n")

    # VOC: polygon objects (1-based pixel coordinates), a box object, an object with two polygons
    d = OUT / "voc_seg" / "in"
    (d / "images").mkdir(parents=True, exist_ok=True)
    (d / "annotations").mkdir(parents=True, exist_ok=True)
    def obj(name, box, polys_=()):
        pe = "".join("<polygon>" + "".join(f"<x{k + 1}>{x}</x{k + 1}><y{k + 1}>{y}</y{k + 1}>" for k, (x, y) in enumerate(p)) + "</polygon>" for p in polys_)
        return (f"<object><name>{name}</name><pose>Unspecified</pose><truncated>0</truncated><difficult>0</difficult>"
                f"<bndbox><xmin>{box[0]}</xmin><ymin>{box[1]}</ymin><xmax>{box[2]}</xmax><ymax>{box[3]}</ymax></bndbox>{pe}</object>")
    voc = {
        names[0]: [obj("cat", (3, 3, 15, 15), [[(3.5, 3), (15, 3), (15, 15.5), (3, 15)]]), obj("dog", (2, 2, 6, 6))],
        names[1]: [obj("dog", (1, 1, 22, 22), [[(2, 2), (10, 2), (10, 10)], [(14, 14), (22, 14), (22, 22), (14, 22)]])],
        names[2]: [obj("cat", (1, 1, 8, 8))],
    }
    for n in names:
        (d / "images" / n).write_bytes(imgs[n])
        if n in voc:
            xml = (f"<annotation><folder>images</folder><filename>{n}</filename><size><width>{SW}</width>"
                   f"<height>{SH}</height><depth>3</depth></size><segmented>1</segmented>{''.join(voc[n])}</annotation>\n")
            (d / "annotations" / (Path(n).stem + ".xml")).write_text(xml)


def bundle(root: Path, prefix: str = "") -> bytes:
    """Deterministic single-file form of a directory: for each file, sorted by path,
    a header `== <relpath> <nbytes>\\n` then the raw bytes. The Mojo side writes the same form."""
    parts = []
    for p in sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: p.as_posix()):
        data = p.read_bytes()
        parts.append(f"== {prefix}{p.relative_to(root).as_posix()} {len(data)}\n".encode() + data)
    return b"".join(parts)


def summarize(ds) -> dict:
    """Load result as plain JSON: classes plus per-image class ids and xyxy boxes, sorted by image name."""
    images = []
    for path, _img, det in ds:
        row = {
            "image": Path(path).name,
            "class_id": [int(c) for c in det.class_id],
            "xyxy": [[float(v) for v in box] for box in det.xyxy],
        }
        if det.mask is not None:  # dense masks as uncompressed COCO run lengths
            from supervision.detection.utils.converters import mask_to_rle
            row["mask_rle"] = [mask_to_rle(m) for m in det.mask]
        images.append(row)
    images.sort(key=lambda d: d["image"])
    return {"classes": list(ds.classes), "images": images}


def write_case_rows(summaries: dict) -> None:
    """tests/cases/dataset_<fmt>.txt: load (value), save roundtrip, split, merge (ref bundles)."""
    from oracle import write_rows  # tools/oracle.py, on sys.path as the script's directory

    cases = ROOT / "tests" / "cases"
    cases.mkdir(parents=True, exist_ok=True)
    for fmt in ("coco", "yolo", "voc", *SEG_FORMATS):
        base = f"tests/fixtures/datasets/{fmt}"
        inp = f"{base}/in"
        ref = f"{base}/ref"
        fmt_name = fmt.split("_")[0]
        rows = [
            ("load", {"input": inp, "params": {"format": fmt_name}, "tol": {}}, summaries[fmt]),
            ("save_roundtrip", {"input": inp, "params": {"format": fmt_name}, "tol": {}},
             {"ref": f"{ref}/roundtrip.bundle"}),
            ("split", {"input": inp, "params": {"format": fmt_name, "split_ratio": 0.5, "random_state": SEED,
                                                "shuffle": True}, "tol": {}},
             {"ref": f"{ref}/split.bundle"}),
            ("merge", {"input": f"{ref}/split_train", "params": {"format": fmt_name, "with": f"{ref}/split_valid"},
                       "tol": {}},
             {"ref": f"{ref}/merge.bundle"}),
        ]
        if fmt in ("yolo_seg", "voc_seg"):
            rows = rows[:1]
        path = cases / f"dataset_{fmt}.txt"
        write_rows(str(path), "gen_datasets", SEED, rows)
        check_case_rows(path)


def check_case_rows(path: Path) -> None:
    """Parse check: line 1 header, 3 tab columns, JSON args and expected, args carry tol."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if lines[0] != f"# gen: tools/gen_datasets.py seed={SEED}":
        fail("parse", f"{path.name} line 1 is not the # gen: header")
    for n, line in enumerate(lines[1:], start=2):
        cols = line.split("\t")
        if len(cols) != 3:
            fail("parse", f"{path.name}:{n} has {len(cols)} tab columns, want 3")
        try:
            args, expected = json.loads(cols[1]), json.loads(cols[2])
        except json.JSONDecodeError as exc:
            fail("parse", f"{path.name}:{n} bad JSON: {exc}")
        if "tol" not in args:
            fail("parse", f"{path.name}:{n} args lack tol")
        if "ref" in expected and not (ROOT / expected["ref"]).is_file():
            fail("parse", f"{path.name}:{n} ref {expected['ref']} is not a file")
    print(f"ok parse: {path.name} {len(lines) - 1} rows")


def run_oracle() -> dict:
    summaries: dict = {}
    try:
        import supervision as sv
        from supervision.dataset.formats import coco as coco_fmt
    except ImportError as e:
        fail("oracle-import", f"{e}; .work/oracle-venv needs supervision (H1 owns it)")

    # supervision stamps COCO "date_captured" with datetime.now(): pin it so regeneration is byte-stable.
    class _PinnedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            return datetime(2020, 1, 1, 0, 0, 0)

    coco_fmt.datetime = _PinnedDatetime

    def save_all(ds, fmt: str, tag: str) -> None:
        d = OUT / fmt / "ref" / tag
        shutil.rmtree(d, ignore_errors=True)
        imgs, anns = d / "images", d / "annotations"
        try:
            if fmt == "coco" or fmt == "coco_seg":
                ds.as_coco(images_directory_path=str(imgs), annotations_path=str(d / "annotations.json"))
            elif fmt == "yolo":
                ds.as_yolo(images_directory_path=str(imgs), annotations_directory_path=str(d / "labels"),
                           data_yaml_path=str(d / "data.yaml"))
            else:
                ds.as_pascal_voc(images_directory_path=str(imgs), annotations_directory_path=str(anns))
        except Exception as e:  # noqa: BLE001 - reported with the step name
            fail(f"save-{fmt}-{tag}", repr(e))

    for fmt in ("coco", "yolo", "voc", *SEG_FORMATS):
        src = OUT / fmt / "in"
        base_fmt = fmt.split("_")[0]
        try:
            if base_fmt == "coco":
                ds = sv.DetectionDataset.from_coco(str(src / "images"), str(src / "annotations.json"))
            elif base_fmt == "yolo":
                ds = sv.DetectionDataset.from_yolo(str(src / "images"), str(src / "labels"), str(src / "data.yaml"))
            else:
                ds = sv.DetectionDataset.from_pascal_voc(str(src / "images"), str(src / "annotations"))
        except Exception as e:  # noqa: BLE001
            fail(f"load-{fmt}", repr(e))
        if fmt in ("yolo_seg", "voc_seg"):  # their export bridges holes (not ported): load rows only
            summaries[fmt] = summarize(ds)
            print(f"ok {fmt}: {len(ds)} images (load only)")
            continue
        save_all(ds, fmt, "roundtrip")
        train, valid = ds.split(split_ratio=0.5, random_state=SEED, shuffle=True)
        save_all(train, fmt, "split_train")
        save_all(valid, fmt, "split_valid")
        merged = sv.DetectionDataset.merge([train, valid])
        save_all(merged, fmt, "merge")
        ref = OUT / fmt / "ref"
        # One file per op so rowdiff can cmp it: the whole output directory as a bundle.
        (ref / "roundtrip.bundle").write_bytes(bundle(ref / "roundtrip"))
        (ref / "split.bundle").write_bytes(bundle(ref / "split_train", "train/") + bundle(ref / "split_valid", "valid/"))
        (ref / "merge.bundle").write_bytes(bundle(ref / "merge"))
        summaries[fmt] = summarize(ds)
        print(f"ok {fmt}: {len(ds)} images, split {len(train)}+{len(valid)}, merged {len(merged)}")
    return summaries


def verify_resave() -> None:
    """Load each ref/roundtrip, save it again, and require byte equality with the reference."""
    import supervision as sv

    scratch = ROOT / ".work" / "H4-resave"
    shutil.rmtree(scratch, ignore_errors=True)
    for fmt in ("coco", "yolo", "voc"):
        src = OUT / fmt / "ref" / "roundtrip"
        out = scratch / fmt
        try:
            if fmt == "coco":
                ds = sv.DetectionDataset.from_coco(str(src / "images"), str(src / "annotations.json"))
                ds.as_coco(images_directory_path=str(out / "images"), annotations_path=str(out / "annotations.json"))
            elif fmt == "yolo":
                ds = sv.DetectionDataset.from_yolo(str(src / "images"), str(src / "labels"), str(src / "data.yaml"))
                ds.as_yolo(images_directory_path=str(out / "images"), annotations_directory_path=str(out / "labels"),
                           data_yaml_path=str(out / "data.yaml"))
            else:
                ds = sv.DetectionDataset.from_pascal_voc(str(src / "images"), str(src / "annotations"))
                ds.as_pascal_voc(images_directory_path=str(out / "images"), annotations_directory_path=str(out / "annotations"))
        except Exception as e:  # noqa: BLE001
            fail(f"resave-{fmt}", repr(e))
        for ref_file in sorted(p for p in src.rglob("*") if p.is_file()):
            got = out / ref_file.relative_to(src)
            if not got.is_file() or got.read_bytes() != ref_file.read_bytes():
                fail(f"resave-{fmt}", f"{ref_file.relative_to(src)} differs after load->save")
    shutil.rmtree(scratch, ignore_errors=True)
    print("ok resave: load->save of every roundtrip reference is byte-equal")


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    write_inputs()
    write_seg_inputs()
    summaries = run_oracle()
    verify_resave()
    write_case_rows(summaries)


if __name__ == "__main__":
    main()

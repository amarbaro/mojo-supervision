"""Row binary body for tests/cases/dataset_{coco,yolo,voc}.txt: load, save_roundtrip, split, merge.

save_* ops write a directory, then a bundle file (`== <relpath> <nbytes>\\n` + bytes per file, sorted)
and print {"out": <bundle path>}, which tests/run.sh compares byte for byte with the Python reference.
"""
from json import Json
from std.os import makedirs
from supervision.dataset.coco import as_coco, from_coco
from supervision.dataset.core import DetectionDataset
from supervision.dataset.fsutil import list_files_rec, read_bytes, write_bytes
from supervision.dataset.rowio import error_json, read_rows, row_args, row_op
from supervision.detection.utils.converters import mask_to_rle
from supervision.dataset.voc import as_pascal_voc, from_pascal_voc
from supervision.dataset.yolo import as_yolo, from_yolo

comptime PINNED_DATE = "01/01/2020,00:00:00"  # the reference pins supervision's datetime.now()
comptime OUT_ROOT = "build/ds-out"


def load(fmt: String, d: String) raises -> DetectionDataset:
    if fmt == "coco":
        return from_coco(d + "/images", d + "/annotations.json")
    if fmt == "yolo":
        return from_yolo(d + "/images", d + "/labels", d + "/data.yaml")
    if fmt == "voc":
        return from_pascal_voc(d + "/images", d + "/annotations")
    raise Error("unknown format " + fmt)


def save(fmt: String, ds: DetectionDataset, d: String) raises:
    if fmt == "coco":
        as_coco(ds, d + "/images", d + "/annotations.json", PINNED_DATE)
    elif fmt == "yolo":
        as_yolo(ds, d + "/images", d + "/labels", d + "/data.yaml")
    elif fmt == "voc":
        as_pascal_voc(ds, d + "/images", d + "/annotations")
    else:
        raise Error("unknown format " + fmt)


def bundle(root: String, prefix: String, mut out: List[UInt8]) raises:
    for rel in list_files_rec(root):
        var data = read_bytes(root + "/" + rel)
        var head = "== " + prefix + rel + " " + String(len(data)) + "\n"
        out.extend(head.as_bytes())
        out.extend(data^)


def summary(ds: DetectionDataset) raises -> Json:
    """classes plus per-image class ids, boxes and (with masks) run lengths, sorted by image file name."""
    var order = List[Int]()
    var names = List[String]()
    for i in range(len(ds)):
        order.append(i)
        var p = ds.image_paths[i]
        var k = p.rfind("/")
        names.append(String(p[byte = k + 1 :]) if k >= 0 else p)
    for i in range(1, len(order)):
        var j = i
        while j > 0 and names[order[j]] < names[order[j - 1]]:
            var t = order[j]
            order[j] = order[j - 1]
            order[j - 1] = t
            j -= 1
    var classes = Json.array()
    for c in ds.classes:
        classes.push(Json.of(c))
    var images = Json.array()
    for i in order:
        ref d = ds.annotations[i]
        var ids = Json.array()
        var boxes = Json.array()
        for k in range(len(d)):
            ids.push(Json.of(d.class_id[k]))
            var b = Json.array()
            for c in range(4):
                b.push(Json.of(d.xyxy[4 * k + c]))
            boxes.push(b^)
        var im = Json.object()
        im.set("image", Json.of(names[i]))
        im.set("class_id", ids^)
        im.set("xyxy", boxes^)
        if d.has_mask:  # dense masks as uncompressed COCO run lengths
            var rles = Json.array()
            var px = d.mask_h * d.mask_w
            for k in range(len(d)):
                var r = Json.array()
                for c in mask_to_rle(List[UInt8](d.mask[k * px : (k + 1) * px]), d.mask_h, d.mask_w):
                    r.push(Json.of(c))
                rles.push(r^)
            im.set("mask_rle", rles^)
        images.push(im^)
    var o = Json.object()
    o.set("classes", classes^)
    o.set("images", images^)
    return o^


def handle(op: String, args: Json, tag: String) raises -> Json:
    var fmt = args.get("params").text("format")
    var inp = args.text("input")
    var o = Json.object()
    if op == "load":
        return summary(load(fmt, inp))
    var work = OUT_ROOT + "/" + tag
    var blob = List[UInt8]()
    if op == "save_roundtrip":
        save(fmt, load(fmt, inp), work + "/out")
        bundle(work + "/out", "", blob)
    elif op == "split":
        var p = args.get("params")
        var parts = load(fmt, inp).split(p.get("split_ratio").num(), p.get("random_state").int(), p.get("shuffle").truthy())
        save(fmt, parts[0], work + "/train")
        save(fmt, parts[1], work + "/valid")
        bundle(work + "/train", "train/", blob)
        bundle(work + "/valid", "valid/", blob)
    elif op == "merge":
        var ds = List[DetectionDataset]()
        ds.append(load(fmt, inp))
        ds.append(load(fmt, args.get("params").text("with")))
        save(fmt, DetectionDataset.merge(ds), work + "/out")
        bundle(work + "/out", "", blob)
    else:
        raise Error("unknown op " + op)
    write_bytes(work + ".bundle", Span[UInt8](blob))
    o.set("out", Json.of(work + ".bundle"))
    return o^


def run(fmt_stem: String) raises:
    makedirs(OUT_ROOT, exist_ok=True)
    var n = 0
    for line in read_rows():
        n += 1
        try:
            print(handle(row_op(line), row_args(line), fmt_stem + "-" + String(n)).dumps())
        except e:
            print(error_json("FAIL " + row_op(line) + ": " + String(e)).dumps())

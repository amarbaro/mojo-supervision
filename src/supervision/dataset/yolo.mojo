"""YOLO dataset format (dataset/formats/yolo.py): boxes and polygon rows (dense masks on load); OBB rows and mask export raise."""
from std.math import isfinite
from std.os.path import exists
from supervision.dataset.core import DetectionDataset
from supervision.dataset.datayaml import class_names, pose_value_count
from supervision.dataset.fsutil import (
    image_size_wh, join, list_files_with_extensions, mkdirs, name_of, parent_of, read_lines, stem_of, write_text,
)
from supervision.dataset.numfmt import fmt_fixed, py_float
from supervision.dataset.utils import check_no_basename_collisions, py_repr
from supervision.dataset.yaml import dump_data_yaml, read_yaml_file
from mojocv.imgproc import rint
from supervision.detection.core import Detections
from supervision.detection.utils.converters import polygon_to_xyxy
from supervision.detection.utils.polygons import polygon_to_mask

def _yolo_exts() -> List[String]:
    return ["bmp", "dng", "jpg", "jpeg", "mpo", "png", "tif", "tiff", "webp"]


def _tokens(line: String) -> List[String]:
    var out = List[String]()
    for t in line.split():
        out.append(String(t))
    return out^


def _parse_class_id(v: String) raises -> Int:
    try:
        return Int(v.strip())
    except:
        pass
    try:
        var f = Float64(String(v.strip()))
        if isfinite(f) and f == Float64(Int(f)):
            return Int(f)
    except:
        pass
    raise Error("Invalid class id " + py_repr(v) + " in YOLO annotation; expected a whole number.")


def _parse_box(v: List[String]) raises -> List[Float32]:
    """[x_center, y_center, width, height] strings to float32 corners (relative)."""
    var n = List[Float64]()
    for s in v:
        n.append(py_float(s))
    for x in n:
        if not isfinite(x):
            raise Error(
                "Invalid box (" + py_repr(v[0]) + ", " + py_repr(v[1]) + ", " + py_repr(v[2]) + ", "
                + py_repr(v[3]) + ") in YOLO annotation; expected a finite center, width and height."
            )
    if n[2] < 0 or n[3] < 0:
        raise Error(
            "Invalid box extent (" + py_repr(v[2]) + ", " + py_repr(v[3])
            + ") in YOLO annotation; expected a non-negative width and height."
        )
    return [Float32(n[0] - n[2] / 2), Float32(n[1] - n[3] / 2), Float32(n[0] + n[2] / 2), Float32(n[1] + n[3] / 2)]


def yolo_lines_to_detections(lines: List[String], w: Int, h: Int) raises -> Detections:
    if len(lines) == 0:
        return Detections.empty()
    var xyxy = List[Float64]()
    var ids = List[Int]()
    var scale: List[Float32] = [Float32(w), Float32(h), Float32(w), Float32(h)]
    var with_masks = False
    for line in lines:
        with_masks = with_masks or len(_tokens(line)) > 6
    var masks = List[UInt8]()
    for line in lines:
        var v = _tokens(line)
        if len(v) < 5:
            raise Error(
                "Invalid YOLO annotation line " + py_repr(line) + "; expected at least 5 tokens "
                "(class id and four box values), got " + String(len(v)) + "."
            )
        ids.append(_parse_class_id(v[0]))
        var rel = List[Float32]()  # relative xyxy
        var poly = List[Float32]()  # relative polygon (x, y pairs)
        if len(v) <= 6:
            if len(v) == 6:
                _ = py_float(v[5])
            rel = _parse_box([v[1], v[2], v[3], v[4]])
            poly = [rel[0], rel[1], rel[2], rel[1], rel[2], rel[3], rel[0], rel[3]]
        else:
            var n = len(v) - 1
            if n % 2 == 1:
                _ = py_float(v[n])  # a trailing confidence
                n -= 1
            var fl = List[Float64]()
            for k in range(n):
                poly.append(Float32(py_float(v[1 + k])))
                fl.append(Float64(poly[k]))
            var bb = polygon_to_xyxy(fl)
            for k in range(4):
                rel.append(Float32(bb[k]))
        for k in range(4):
            xyxy.append(Float64(rel[k] * scale[k]))
        if with_masks:
            var px = List[Float64]()
            for k in range(len(poly)):
                px.append(Float64(rint(Float64(poly[k] * scale[k % 2]))))
            masks.extend(polygon_to_mask(px, w, h))
    var d = Detections(xyxy^)
    d.set_class_id(ids^)
    if with_masks:
        d.set_mask(masks^, h, w)
    d.validate()
    return d^


def _drop_keypoints(lines: List[String], pose_count: Int) -> List[String]:
    var out = List[String]()
    for line in lines:
        var v = _tokens(line)
        if len(v) == 5 + pose_count:
            out.append(" ".join(v[:5]))
        else:
            out.append(line)
    return out^


def from_yolo(images_dir: String, labels_dir: String, data_yaml: String) raises -> DetectionDataset:
    var paths = list_files_with_extensions(images_dir, _yolo_exts())
    var doc = read_yaml_file(data_yaml)
    var classes = class_names(doc, data_yaml)
    var pose = pose_value_count(doc, data_yaml)
    var anns = List[Detections]()
    for p in paths:
        var ap = labels_dir + "/" + stem_of(p) + ".txt"
        if not exists(ap):
            anns.append(Detections.empty())
            continue
        var wh = image_size_wh(p)
        var lines = read_lines(ap, True)
        if pose > 0:
            lines = _drop_keypoints(lines, pose)
        try:
            anns.append(yolo_lines_to_detections(lines, wh[0], wh[1]))
        except e:
            raise Error("Invalid YOLO annotation file '" + ap + "': " + String(e))
    return DetectionDataset(classes, paths, anns)


def _box_line(xyxy: List[Float64], i: Int, cid: Int, w: Int, h: Int) raises -> String:
    var x1 = Float32(xyxy[4 * i]) / Float32(w)
    var y1 = Float32(xyxy[4 * i + 1]) / Float32(h)
    var x2 = Float32(xyxy[4 * i + 2]) / Float32(w)
    var y2 = Float32(xyxy[4 * i + 3]) / Float32(h)
    var x_min = min(x1, x2)
    var x_max = max(x1, x2)
    var y_min = min(y1, y2)
    var y_max = max(y1, y2)
    var xc = (x_min + x_max) / 2
    var yc = (y_min + y_max) / 2
    return (
        String(cid) + " " + fmt_fixed(Float64(xc), 5) + " " + fmt_fixed(Float64(yc), 5) + " "
        + fmt_fixed(Float64(x_max - x_min), 5) + " " + fmt_fixed(Float64(y_max - y_min), 5)
    )


def save_yolo_annotations(ds: DetectionDataset, labels_dir: String) raises:
    var names = List[String]()
    for p in ds.image_paths:
        names.append(stem_of(p) + ".txt")
    check_no_basename_collisions(ds.image_paths, names, "YOLO annotation")
    mkdirs(labels_dir)
    for k in range(len(ds.image_paths)):
        ref d = ds.annotations[k]
        var wh = image_size_wh(ds.image_paths[k])
        var out = String()
        if d.has_mask:
            raise Error("YOLO export of masks (polygons with bridged holes) is not ported yet")
        for i in range(len(d)):
            if not d.has_class_id:
                raise Error("Class ID is required for YOLO annotations.")
            out += _box_line(d.xyxy, i, d.class_id[i], wh[0], wh[1]) + "\n"
        write_text(join(labels_dir, names[k]), out)


def save_data_yaml(path: String, classes: List[String]) raises:
    mkdirs(parent_of(path))
    write_text(path, dump_data_yaml(classes))


def as_yolo(ds: DetectionDataset, images_dir: String, labels_dir: String, data_yaml: String) raises:
    """DetectionDataset.as_yolo with all three outputs (images, labels, data.yaml)."""
    from supervision.dataset.core import save_dataset_images
    var names = List[String]()
    for p in ds.image_paths:
        names.append(name_of(p))
    save_dataset_images(ds, images_dir)
    save_yolo_annotations(ds, labels_dir)
    save_data_yaml(data_yaml, ds.classes)

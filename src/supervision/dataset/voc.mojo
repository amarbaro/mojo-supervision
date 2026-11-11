"""Pascal VOC dataset format (dataset/formats/pascal_voc.py): boxes and polygon objects (dense masks on load); mask export raises."""
from std.collections import Dict
from std.math import isfinite
from std.os.path import exists
from supervision.dataset.core import DetectionDataset, save_dataset_images
from supervision.dataset.fsutil import (
    image_size_wh, join, list_files_with_extensions, mkdirs, name_of, read_text, stem_of, write_text,
)
from supervision.dataset.numfmt import py_float
from supervision.dataset.utils import check_no_basename_collisions, py_repr
from supervision.dataset.xml import XmlDoc, parse_xml, xml_escape
from mojocv.imgproc import rint
from supervision.detection.core import Detections
from supervision.detection.utils.polygons import polygon_to_mask

def _voc_exts() -> List[String]:
    return ["bmp", "jpeg", "jpg", "png", "tif", "tiff", "webp"]


def _coordinate(text: String, tag: String) raises -> Float64:
    var v = Float64(0.0)
    var ok = True
    try:
        v = py_float(text)
    except:
        ok = False
    if not ok or not isfinite(v):
        raise Error("Invalid '" + tag + "' value " + py_repr(text) + " in Pascal VOC annotation.")
    return v


def _required_text(doc: XmlDoc, element: Int, tag: String) raises -> String:
    var c = doc.find(element, tag)
    if c < 0 or not doc.has_text[c]:
        raise Error("Missing '" + tag + "' in Pascal VOC annotation.")
    return doc.text[c]


def _sort_strings(mut v: List[String]):
    for i in range(1, len(v)):
        var j = i
        while j > 0 and v[j] < v[j - 1]:
            var t = v[j]
            v[j] = v[j - 1]
            v[j - 1] = t
            j -= 1


def detections_from_xml(doc: XmlDoc, mut classes: List[String], w: Int, h: Int) raises -> Detections:
    """detections_from_xml_obj for box objects; `classes` grows with the names seen."""
    var objs = doc.findall(0, "object")
    var xyxy = List[Float64]()
    var names = List[String]()
    var with_masks = False
    for o in objs:
        with_masks = with_masks or doc.find(o, "polygon") >= 0
    var masks = List[UInt8]()
    for o in objs:
        names.append(_required_text(doc, o, "name"))
        var bb = doc.find(o, "bndbox")
        if bb < 0:
            raise Error("Missing bndbox in Pascal VOC annotation.")
        var x1 = _coordinate(_required_text(doc, bb, "xmin"), "xmin")
        var y1 = _coordinate(_required_text(doc, bb, "ymin"), "ymin")
        var x2 = _coordinate(_required_text(doc, bb, "xmax"), "xmax")
        var y2 = _coordinate(_required_text(doc, bb, "ymax"), "ymax")
        var lo_x = min(x1, x2)
        var hi_x = max(x1, x2)
        var lo_y = min(y1, y2)
        var hi_y = max(y1, y2)
        # float32 array, then `xyxy -= 1` (1-indexed VOC to 0-indexed)
        xyxy.append(Float64(Float32(lo_x) - 1))
        xyxy.append(Float64(Float32(lo_y) - 1))
        xyxy.append(Float64(Float32(hi_x) - 1))
        xyxy.append(Float64(Float32(hi_y) - 1))
        var om = List[UInt8](length=w * h, fill=0)
        for pe in doc.findall(o, "polygon"):
            var coords = List[Float64]()
            for c in doc.descendants(pe):
                if not doc.has_text[c]:
                    raise Error("Missing polygon coordinate value in Pascal VOC.")
                coords.append(Float64(rint(_coordinate(doc.text[c], doc.tag[c]) - 1.0)))
            if len(coords) % 2 == 1:
                raise Error("list index out of range")
            var pm = polygon_to_mask(coords, w, h)
            for k in range(w * h):
                om[k] |= pm[k]
        if with_masks:
            masks.extend(om^)
    var uniq = List[String]()
    for nm in names:
        var have = False
        for u in uniq:
            if u == nm:
                have = True
        if not have:
            uniq.append(nm)
    _sort_strings(uniq)
    for k in uniq:
        var have = False
        for c in classes:
            if c == k:
                have = True
        if not have:
            classes.append(k)
    var ids = List[Int]()
    for nm in names:
        for ci in range(len(classes)):
            if classes[ci] == nm:
                ids.append(ci)
                break
    var d = Detections(xyxy^)
    d.set_class_id(ids^)
    if with_masks:
        d.set_mask(masks^, h, w)
    d.validate()
    return d^


def from_pascal_voc(images_dir: String, annotations_dir: String) raises -> DetectionDataset:
    var paths = list_files_with_extensions(images_dir, _voc_exts())
    _sort_strings(paths)
    var classes = List[String]()
    var anns = List[Detections]()
    for p in paths:
        var ap = annotations_dir + "/" + stem_of(p) + ".xml"
        if not exists(ap):
            anns.append(Detections.empty())
            continue
        var doc = parse_xml(read_text(ap))
        var wh = image_size_wh(p)
        anns.append(detections_from_xml(doc, classes, wh[0], wh[1]))
    return DetectionDataset(classes, paths, anns)


def pascal_voc_xml(d: Detections, classes: List[String], filename: String, w: Int, h: Int, depth: Int) raises -> String:
    """detections_to_pascal_voc for box detections, in minidom's toprettyxml layout."""
    if d.has_mask:
        raise Error("Pascal VOC export of masks (polygons with bridged holes) is not ported yet")
    var s = String('<?xml version="1.0" ?>\n<annotation>\n')
    s += "  <folder>VOC</folder>\n"
    s += "  <filename>" + xml_escape(filename) + "</filename>\n"
    s += "  <source>\n    <database>roboflow.ai</database>\n  </source>\n"
    s += "  <size>\n    <width>" + String(w) + "</width>\n    <height>" + String(h) + "</height>\n"
    s += "    <depth>" + String(depth) + "</depth>\n  </size>\n"
    s += "  <segmented>0</segmented>\n"
    for i in range(len(d)):
        if not d.has_class_id:
            raise Error("Detections must include class_id for Pascal VOC export.")
        var x1 = Float32(d.xyxy[4 * i]) + 1
        var y1 = Float32(d.xyxy[4 * i + 1]) + 1
        var x2 = Float32(d.xyxy[4 * i + 2]) + 1
        var y2 = Float32(d.xyxy[4 * i + 3]) + 1
        s += "  <object>\n    <name>" + xml_escape(classes[d.class_id[i]]) + "</name>\n    <bndbox>\n"
        s += "      <xmin>" + String(Int(min(x1, x2))) + "</xmin>\n"
        s += "      <ymin>" + String(Int(min(y1, y2))) + "</ymin>\n"
        s += "      <xmax>" + String(Int(max(x1, x2))) + "</xmax>\n"
        s += "      <ymax>" + String(Int(max(y1, y2))) + "</ymax>\n"
        s += "    </bndbox>\n  </object>\n"
    return s + "</annotation>\n"


def save_pascal_voc_annotations(ds: DetectionDataset, annotations_dir: String) raises:
    var names = List[String]()
    for p in ds.image_paths:
        names.append(stem_of(p) + ".xml")
    check_no_basename_collisions(ds.image_paths, names, "Pascal VOC annotation")
    mkdirs(annotations_dir)
    for k in range(len(ds.image_paths)):
        var wh = image_size_wh(ds.image_paths[k])
        var xml = pascal_voc_xml(ds.annotations[k], ds.classes, name_of(ds.image_paths[k]), wh[0], wh[1], 3)
        write_text(join(annotations_dir, names[k]), xml)


def as_pascal_voc(ds: DetectionDataset, images_dir: String, annotations_dir: String) raises:
    var imgs = List[String]()
    for p in ds.image_paths:
        imgs.append(name_of(p))
    var xmls = List[String]()
    for p in ds.image_paths:
        xmls.append(stem_of(p) + ".xml")
    check_no_basename_collisions(ds.image_paths, imgs, "image")
    check_no_basename_collisions(ds.image_paths, xmls, "Pascal VOC annotation")
    save_dataset_images(ds, images_dir)
    save_pascal_voc_annotations(ds, annotations_dir)

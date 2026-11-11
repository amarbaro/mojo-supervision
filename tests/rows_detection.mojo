"""Row driver for tests/cases/detection_*.txt (contract: tests/README.md).

Reads row lines (`op<TAB>args<TAB>expected`) on stdin, prints one JSON result per row. The same binary is
copied to build/detection_{converters,detections,iou,nms} by tests/build.sh.
"""

from std.ffi import external_call, c_ssize_t
from json import Json, parse, ARR, STR
from std.math import isnan
from supervision.classification.core import Classifications
from supervision.detection.core import Detections, DataCol
from supervision.detection.utils.masks import (
    calculate_masks_centroids,
    count_mask_pixels,
    mask_to_roi,
    move_masks,
    resize_masks,
)
from supervision.detection.utils.matching import match_detections
from supervision.geometry.core import Position
from supervision.detection.utils.converters import (
    mask_to_rle,
    mask_to_rle_string,
    rle_string_to_mask,
    rle_to_mask,
)
from supervision.detection.utils.iou_and_nms import oriented_box_iou_batch
from supervision.detection.utils.masks import contains_holes, contains_multiple_segments
from supervision.detection.utils.polygons import (
    approximate_mask_with_polygons, approximate_polygon, mask_to_polygons, polygon_to_mask,
)
from supervision.detection.utils.converters import (
    mask_to_xyxy,
    xcycwh_to_xyxy,
    xywh_to_xyxy,
    xyxy_to_mask,
    xyxy_to_xcycarh,
    xyxy_to_xywh,
)
from supervision.detection.utils.iou_and_nms import (
    OVERLAP_IOU,
    box_iou,
    box_iou_batch,
    box_non_max_merge,
    box_non_max_suppression,
    mask_iou_batch,
    overlap_metric_from_value,
)


def flat_f(j: Json, mut out: List[Float64]):
    if j.kind() == ARR:
        for i in range(j.len()):
            flat_f(j.at(i), out)
    else:
        out.append(j.num())


def flat_i(j: Json, mut out: List[UInt8]):
    if j.kind() == ARR:
        for i in range(j.len()):
            flat_i(j.at(i), out)
    else:
        out.append(UInt8(j.int()))


def floats(j: Json) -> List[Float64]:
    var o = List[Float64]()
    flat_f(j, o)
    return o^


def metric(a: Json) raises -> Int:
    if a.has("overlap_metric"):
        return overlap_metric_from_value(a.get("overlap_metric").str())
    return OVERLAP_IOU


def jf(v: Float64) -> Json:
    return Json.of("nan") if isnan(v) else Json.of(v)


def jrow(v: List[Float64], start: Int, n: Int) -> Json:
    var r = Json.array()
    for i in range(start, start + n):
        r.push(jf(v[i]))
    return r^


def jflat(v: List[Float64]) -> Json:
    return jrow(v, 0, len(v))


def jmat(v: List[Float64], cols: Int) -> Json:
    var r = Json.array()
    if cols == 0:
        return r^
    for i in range(len(v) // cols):
        r.push(jrow(v, i * cols, cols))
    return r^


def jmat32(v: List[Float32], cols: Int) -> Json:
    var f = List[Float64]()
    for x in v:
        f.append(Float64(x))
    return jmat(f, cols)


def jints(v: List[Int]) -> Json:
    var r = Json.array()
    for x in v:
        r.push(Json.of(x))
    return r^


def jbools(v: List[Bool]) -> Json:
    var r = Json.array()
    for x in v:
        r.push(Json.of(x))
    return r^


def jmask3(m: List[UInt8], n: Int, h: Int, w: Int, as_bool: Bool) -> Json:
    var r = Json.array()
    for i in range(n):
        var plane = Json.array()
        for y in range(h):
            var row = Json.array()
            for x in range(w):
                var v = m[i * h * w + y * w + x] != 0
                row.push(Json.of(v) if as_bool else Json.of(Int(v)))
            plane.push(row^)
        r.push(plane^)
    return r^


def parse_masks(j: Json) -> Tuple[List[UInt8], Int, Int, Int]:
    """(flat, n, h, w) of an [n][h][w] nested list."""
    var m = List[UInt8]()
    flat_i(j, m)
    var n = j.len()
    var h = j.at(0).len() if n > 0 else 0
    var w = j.at(0).at(0).len() if h > 0 else 0
    return (m^, n, h, w)


def preds(j: Json) -> Tuple[List[Float64], List[Float64], List[Int]]:
    var xyxy = List[Float64]()
    var sc = List[Float64]()
    var cats = List[Int]()
    for i in range(j.len()):
        var r = j.at(i)
        for k in range(4):
            xyxy.append(r.at(k).num())
        sc.append(r.at(4).num())
        cats.append(Int(r.at(5).num()) if r.len() > 5 else 0)
    return (xyxy^, sc^, cats^)


def det_from(a: Json) raises -> Detections:
    var d = Detections(floats(a.get("xyxy")))
    if a.has("confidence"):
        d.set_confidence(floats(a.get("confidence")))
    d.validate()
    return d^


def det_json(d: Detections) -> Json:
    var o = Json.object()
    o.set("xyxy", jmat(d.xyxy, 4))
    if d.has_confidence:
        o.set("confidence", jflat(d.confidence))
    return o^


def present(a: Json, k: String) -> Bool:
    return a.has(k) and not a.get(k).is_null()


def ints(j: Json) -> List[Int]:
    var o = List[Int]()
    for i in range(j.len()):
        o.append(j.at(i).int())
    return o^


def jstrs(v: List[String]) -> Json:
    var r = Json.array()
    for x in v:
        r.push(Json.of(x))
    return r^


def det_full(a: Json) raises -> Detections:
    """Detections from {xyxy, confidence?, class_id?, tracker_id?, mask?, data?, metadata?, xyxyxyxy?}."""
    var d = Detections(floats(a.get("xyxy")))
    if present(a, "confidence"):
        d.set_confidence(floats(a.get("confidence")))
    if present(a, "class_id"):
        d.set_class_id(ints(a.get("class_id")))
    if present(a, "tracker_id"):
        d.set_tracker_id(ints(a.get("tracker_id")))
    if present(a, "mask"):
        var m = parse_masks(a.get("mask"))
        d.set_mask(m[0].copy(), m[2], m[3])
    if present(a, "xyxyxyxy"):
        d.data.append(DataCol("xyxyxyxy", floats(a.get("xyxyxyxy")), 8))
    if present(a, "data"):
        var dj = a.get("data")
        for i in range(dj.len()):
            var name = dj.key_at(i)
            var col = dj.at(i)
            if col.len() > 0 and col.at(0).kind() == STR:
                var s = List[String]()
                for k in range(col.len()):
                    s.append(col.at(k).str())
                d.data.append(DataCol(name, s))
            elif col.len() > 0 and col.at(0).kind() == ARR:
                d.data.append(DataCol(name, floats(col), col.at(0).len()))
            else:
                d.data.append(DataCol(name, floats(col)))
    if present(a, "metadata"):
        var mj = a.get("metadata")
        for i in range(mj.len()):
            d.set_metadata(mj.key_at(i), mj.at(i).str())
    return d^


def det_out(d: Detections, with_data: Bool) -> Json:
    var o = Json.object()
    o.set("xyxy", jmat(d.xyxy, 4))
    if d.has_confidence:
        o.set("confidence", jflat(d.confidence))
    if d.has_class_id:
        o.set("class_id", jints(d.class_id))
    if d.has_tracker_id:
        o.set("tracker_id", jints(d.tracker_id))
    if d.has_mask:
        o.set("mask", jmask3(d.mask, len(d), d.mask_h, d.mask_w, True))
    if with_data:
        var dj = Json.object()
        for c in d.data:
            if c.is_str:
                dj.set(c.name, jstrs(c.s))
            elif c.width == 1:
                dj.set(c.name, jflat(c.f))
            else:
                dj.set(c.name, jmat(c.f, c.width))
        o.set("data", dj)
        var mj = Json.object()
        for i in range(len(d.meta_keys)):
            mj.set(d.meta_keys[i], Json.of(d.meta_vals[i]))
        o.set("metadata", mj)
    return o^


def mask2d(j: Json) -> Tuple[List[UInt8], Int, Int]:
    var m = List[UInt8]()
    flat_i(j, m)
    var h = j.len()
    var w = j.at(0).len() if h > 0 else 0
    return (m^, h, w)


def flat_ints(j: Json) -> List[Int]:
    var o = List[Int]()
    for f in floats(j):
        o.append(Int(f))
    return o^


def jpoly(p: List[Int]) -> Json:
    var r = Json.array()
    for i in range(len(p) // 2):
        r.push(jints([p[2 * i], p[2 * i + 1]]))
    return r^


def jpolys(v: List[List[Int]]) -> Json:
    var r = Json.array()
    for p in v:
        r.push(jpoly(p))
    return r^


def jpairs(v: List[Tuple[Int, Int]]) -> Json:
    var r = Json.array()
    for p in v:
        r.push(jints([p[0], p[1]]))
    return r^


def run2(op: String, a: Json) raises -> Json:
    if op == "det_nms":
        var d = det_full(a)
        d.validate()
        var mode = a.get("mode").str()
        var thr = a.get("threshold").num()
        var agn = a.get("class_agnostic").int() != 0
        var met = metric(a)
        if mode == "nms":
            return det_out(d.with_nms(thr, agn, met), False)
        if mode == "nmm":
            return det_out(d.with_nmm(thr, agn, met), False)
        var has_thr = present(a, "score_threshold")
        return det_out(d.with_soft_nms(a.get("sigma").num(), agn, a.get("score_threshold").num() if has_thr else 0.0, has_thr), False)
    if op == "det_anchor":
        var d = det_full(a)
        return jmat(d.get_anchors_coordinates(Position.from_name(a.get("position").str())), 2)
    if op == "det_props":
        var d = det_full(a)
        var o = Json.object()
        o.set("box_area", jflat(d.box_area()))
        o.set("aspect", jflat(d.box_aspect_ratio()))
        return o^
    if op == "det_merge_full":
        var parts = List[Detections]()
        var ps = a.get("parts")
        for i in range(ps.len()):
            parts.append(det_full(ps.at(i)))
        return det_out(Detections.merge(parts), True)
    if op == "det_select":
        var d = det_full(a.get("det"))
        d.validate()
        return det_out(d.select(ints(a.get("index"))), True)
    if op == "det_validate":
        det_full(a).validate()
        return Json.of("no error")
    if op == "top_k":
        var c = Classifications(ints(a.get("class_id")))
        c.set_confidence(floats(a.get("confidence")))
        var r = c.get_top_k(a.get("k").int())
        var o = Json.array()
        o.push(jints(r[0]))
        o.push(jflat(r[1]))
        return o^
    if op == "match_detections":
        var da = Detections(floats(a.get("xyxy_a")))
        da.set_class_id(ints(a.get("class_a")))
        var db = Detections(floats(a.get("xyxy_b")))
        db.set_class_id(ints(a.get("class_b")))
        var m = match_detections(da, db, a.get("iou_threshold").num(), a.get("class_agnostic").int() != 0)
        var o = Json.object()
        o.set("pairs", jpairs(m.pairs))
        o.set("ua", jints(m.unmatched_a))
        o.set("ub", jints(m.unmatched_b))
        return o^
    if op == "move_masks":
        var m = parse_masks(a.get("masks"))
        var off = ints(a.get("offset"))
        var res = ints(a.get("resolution_wh"))
        return jmask3(move_masks(m[0], m[1], m[2], m[3], off[0], off[1], res[0], res[1]), m[1], res[1], res[0], True)
    if op == "calculate_masks_centroids":
        var m = parse_masks(a.get("masks"))
        var c = calculate_masks_centroids(m[0], m[1], m[2], m[3])
        var r = Json.array()
        for i in range(m[1]):
            r.push(jints([c[2 * i], c[2 * i + 1]]))
        return r^
    if op == "count_mask_pixels":
        var m = parse_masks(a.get("masks"))
        return jints(count_mask_pixels(m[0], m[1], m[2] * m[3]))
    if op == "mask_to_roi":
        var m = mask2d(a.get("mask"))
        var roi = mask_to_roi(m[0], m[1], m[2])
        if not roi:
            return Json()
        var r = roi.value()
        return jints([r[0], r[1], r[2], r[3]])
    if op == "mask_to_rle":
        var m = mask2d(a.get("mask"))
        var counts = mask_to_rle(m[0], m[1], m[2])
        if a.get("compressed").int() != 0:
            return Json.of(mask_to_rle_string(m[0], m[1], m[2]))
        return jints(counts)
    if op == "rle_to_mask":
        var res = ints(a.get("resolution_wh"))
        var rle = a.get("rle")
        var m = rle_string_to_mask(rle.str(), res[0], res[1]) if rle.kind() == STR else rle_to_mask(ints(rle), res[0], res[1])
        return jmask3(m, 1, res[1], res[0], True).at(0)
    if op == "polygon_to_mask":
        var res = ints(a.get("resolution_wh"))
        return jmask3(polygon_to_mask(floats(a.get("polygon")), res[0], res[1]), 1, res[1], res[0], False).at(0)
    if op == "mask_to_polygons":
        var m = mask2d(a.get("mask"))
        return jpolys(mask_to_polygons(m[0], m[1], m[2]))
    if op == "contains_holes":
        var m = mask2d(a.get("mask"))
        return Json.of(contains_holes(m[0], m[1], m[2]))
    if op == "contains_multiple_segments":
        var m = mask2d(a.get("mask"))
        return Json.of(contains_multiple_segments(m[0], m[1], m[2], a.get("connectivity").int()))
    if op == "approximate_polygon":
        return jpoly(approximate_polygon(flat_ints(a.get("polygon")), a.get("percentage").num()))
    if op == "approximate_mask_with_polygons":
        var m = mask2d(a.get("mask"))
        return jpolys(approximate_mask_with_polygons(
            m[0], m[1], m[2], a.get("min_image_area_percentage").num(), a.get("max_image_area_percentage").num(),
            a.get("approximation_percentage").num(),
        ))
    if op == "resize_masks":
        var m = parse_masks(a.get("masks"))
        var r = resize_masks(m[0], m[1], m[2], m[3], a.get("max_dimension").int())
        return jmask3(r[0], m[1], r[1], r[2], True)
    if op == "oriented_box_iou_batch":
        var t = floats(a.get("boxes_true"))
        var d = floats(a.get("boxes_detection"))
        return jmat(oriented_box_iou_batch(t, d, metric(a)), len(d) // 8)
    raise Error("unknown op " + op)


def run(op: String, a: Json) raises -> Json:
    if op == "box_iou":
        return jf(box_iou(floats(a.get("box_true")), floats(a.get("box_detection")), metric(a)))
    if op == "box_iou_batch":
        var t = floats(a.get("boxes_true"))
        var d = floats(a.get("boxes_detection"))
        return jmat32(box_iou_batch(t, d, metric(a)), len(d) // 4)
    if op == "mask_iou_batch":
        var t = parse_masks(a.get("masks_true"))
        var d = parse_masks(a.get("masks_detection"))
        var px = t[2] * t[3] if t[1] > 0 else d[2] * d[3]
        return jmat(mask_iou_batch(t[0], d[0], px, metric(a)), d[1])
    if op == "box_non_max_suppression":
        var p = preds(a.get("predictions"))
        return jbools(box_non_max_suppression(p[0], p[1], p[2], a.get("iou_threshold").num(), metric(a)))
    if op == "box_non_max_merge":
        var p = preds(a.get("predictions"))
        var groups = box_non_max_merge(p[0], p[1], p[2], a.get("iou_threshold").num(), metric(a))
        var r = Json.array()
        for g in groups:
            r.push(jints(g))
        return r^
    if op == "xywh_to_xyxy":
        return jmat(xywh_to_xyxy(floats(a.get("xywh"))), 4)
    if op == "xyxy_to_xywh":
        return jmat(xyxy_to_xywh(floats(a.get("xyxy"))), 4)
    if op == "xcycwh_to_xyxy":
        return jmat(xcycwh_to_xyxy(floats(a.get("xcycwh"))), 4)
    if op == "xyxy_to_xcycarh":
        return jmat(xyxy_to_xcycarh(floats(a.get("xyxy"))), 4)
    if op == "mask_to_xyxy":
        var m = parse_masks(a.get("masks"))
        var excl = a.has("coordinate_convention") and a.get("coordinate_convention").str() == "exclusive"
        var b = mask_to_xyxy(m[0], m[1], m[2], m[3], excl)
        var f = List[Float64]()
        for v in b:
            f.append(Float64(v))
        return jmat(f, 4)
    if op == "xyxy_to_mask":
        var res = a.get("resolution_wh")
        var w = res.at(0).int()
        var h = res.at(1).int()
        var boxes = floats(a.get("xyxy"))
        var excl = a.has("coordinate_convention") and a.get("coordinate_convention").str() == "exclusive"
        return jmask3(xyxy_to_mask(boxes, w, h, excl), len(boxes) // 4, h, w, True)
    if op == "detections_area":
        return jflat(det_from(a).area())
    if op == "detections_filter":
        var d = det_from(a)
        var keep = List[Bool]()
        for i in range(a.get("keep").len()):
            keep.append(a.get("keep").at(i).int() != 0)
        return det_json(d.select_where(keep))
    if op == "detections_merge":
        var parts = List[Detections]()
        var ps = a.get("parts")
        for i in range(ps.len()):
            parts.append(det_from(ps.at(i)))
        return det_json(Detections.merge(parts))
    return run2(op, a)


def read_all_stdin() -> String:
    var buf = List[UInt8]()
    var chunk = List[UInt8](unsafe_uninit_length=65536)
    while True:
        var n = external_call["__read", c_ssize_t](0, chunk.unsafe_ptr(), 65536)
        if n <= 0:
            break
        for i in range(Int(n)):
            buf.append(chunk[i])
    return String(unsafe_from_utf8=buf)


def main() raises:
    var text = read_all_stdin()
    for line in text.split("\n"):
        if line.byte_length() == 0:
            continue
        var cols = line.split("\t")
        var result: Json
        try:
            result = run(String(cols[0]), parse(String(cols[1])))
        except e:
            result = Json.object()
            result.set("error", Json.of(String(e)))
        print(result.dumps())

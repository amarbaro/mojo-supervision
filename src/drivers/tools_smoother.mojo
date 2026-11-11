from json import Json, parse
from supervision.config import ORIENTED_BOX_COORDINATES
from supervision.detection.core import Detections, DataCol
from supervision.detection.tools.smoother import DetectionsSmoother
from rowio import read_rows


def nums(a: Json) -> List[Float64]:
    var out = List[Float64]()
    for i in range(a.len()):
        out.append(a.at(i).num())
    return out^


def ints(a: Json) -> List[Int]:
    var out = List[Int]()
    for i in range(a.len()):
        out.append(a.at(i).int())
    return out^


def flat(a: Json) -> List[Float64]:
    var out = List[Float64]()
    for i in range(a.len()):
        out.extend(nums(a.at(i)))
    return out^


def to_detections(f: Json) raises -> Detections:
    var d = Detections(flat(f.get("xyxy")))
    if not f.get("confidence").is_null():
        d.set_confidence(nums(f.get("confidence")))
    if not f.get("class_id").is_null():
        d.set_class_id(ints(f.get("class_id")))
    if not f.get("tracker_id").is_null():
        d.set_tracker_id(ints(f.get("tracker_id")))
    if not f.get("obb").is_null():
        var corners = List[Float64]()  # obb: n x 4 x 2
        var obb = f.get("obb")
        for i in range(obb.len()):
            corners.extend(flat(obb.at(i)))
        d.set_data(DataCol(ORIENTED_BOX_COORDINATES, corners^, 8))
    d.validate()
    return d^


def rows_of(v: List[Float64], w: Int) -> Json:
    var a = Json.array()
    for i in range(len(v) // w):
        var r = Json.array()
        for k in range(w):
            r.push(Json.of(v[i * w + k]))
        a.push(r^)
    return a^


def from_detections(d: Detections) raises -> Json:
    var o = Json.object()
    o.set("xyxy", rows_of(d.xyxy, 4))
    var c = Json()
    if d.has_confidence:
        c = Json.array()
        for v in d.confidence:
            c.push(Json.of(v))
    o.set("confidence", c^)
    var k = Json()
    if d.has_class_id:
        k = Json.array()
        for v in d.class_id:
            k.push(Json.of(v))
    o.set("class_id", k^)
    var t = Json()
    if d.has_tracker_id:
        t = Json.array()
        for v in d.tracker_id:
            t.push(Json.of(v))
    o.set("tracker_id", t^)
    var b = Json()
    if d.has_data(ORIENTED_BOX_COORDINATES):
        b = rows_of(d.get_data(ORIENTED_BOX_COORDINATES).f, 8)
    o.set("obb", b^)
    return o^


def run(args: Json) raises -> Json:
    var sm = DetectionsSmoother(args.get("length").int())
    var frames = args.get("frames")
    var out = Json.array()
    for i in range(frames.len()):
        out.push(from_detections(sm.update_with_detections(to_detections(frames.at(i)))))
    return out^


def main() raises:
    for row in read_rows():
        var cols = row.split("\t")
        var res: Json
        try:
            res = run(parse(String(cols[1])))
        except:
            res = Json.object()
            res.set("error", Json.of(String("Exception")))
        print(res.dumps())

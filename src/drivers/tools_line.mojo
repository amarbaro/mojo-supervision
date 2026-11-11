from json import Json, parse
from supervision.detection.core import Detections
from supervision.detection.tools.line_zone import LineZone, ClassCount
from supervision.geometry.core import Point, Position
from rowio import read_rows


def bools(v: List[Bool]) -> Json:
    var a = Json.array()
    for b in v:
        a.push(Json.of(b))
    return a^


def counts(c: ClassCount) -> Json:
    var a = Json.array()
    for j in range(len(c.keys)):
        var pair = Json.array()
        if c.keys[j]:
            pair.push(Json.of(c.keys[j].value()))
        else:
            pair.push(Json())
        pair.push(Json.of(c.counts[j]))
        a.push(pair^)
    return a^


def run(args: Json) raises -> Json:
    var s = args.get("start")
    var e = args.get("end")
    var names = args.get("anchors")
    var positions = List[Position]()
    for k in range(names.len()):
        positions.append(Position.from_name(names.at(k).str()))
    var zone = LineZone(Point(s.at(0).num(), s.at(1).num()), Point(e.at(0).num(), e.at(1).num()),
                        positions^, args.get("min_threshold").int())
    var frames = args.get("frames")
    var out = Json.array()
    for f in range(frames.len()):
        var fr = frames.at(f)
        var boxes = fr.get("xyxy")
        var n = boxes.len()
        var xyxy = List[Float64]()
        for i in range(n):
            for c in range(4):
                xyxy.append(boxes.at(i).at(c).num())
        var d = Detections(xyxy^)
        if not fr.get("tracker_id").is_null():
            var t = List[Int]()
            for i in range(n):
                t.append(fr.get("tracker_id").at(i).int())
            d.set_tracker_id(t^)
        if not fr.get("class_id").is_null():
            var c = List[Int]()
            for i in range(n):
                c.append(fr.get("class_id").at(i).int())
            d.set_class_id(c^)
        var r = zone.trigger(d)
        var o = Json.object()
        o.set("in", bools(r[0]))
        o.set("out", bools(r[1]))
        out.push(o^)
    var fin = Json.object()
    fin.set("in_count", Json.of(zone.in_count()))
    fin.set("out_count", Json.of(zone.out_count()))
    fin.set("in_per_class", counts(zone.in_counts))
    fin.set("out_per_class", counts(zone.out_counts))
    out.push(fin^)
    return out^


def main() raises:
    for row in read_rows():
        var cols = row.split("\t")
        var res: Json
        try:
            res = run(parse(String(cols[1])))
        except:
            res = Json.object()
            res.set("error", Json.of(String("ValueError")))
        print(res.dumps())

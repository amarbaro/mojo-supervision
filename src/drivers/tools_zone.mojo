from json import Json, parse
from supervision.detection.core import Detections
from supervision.detection.tools.polygon_zone import PolygonZone
from supervision.geometry.core import Position
from rowio import read_rows


def run(args: Json) raises -> Json:
    var poly = List[Float64]()
    var pj = args.get("polygon")
    for i in range(pj.len()):
        poly.append(pj.at(i).at(0).num())
        poly.append(pj.at(i).at(1).num())
    var names = args.get("anchors")
    var positions = List[Position]()
    for k in range(names.len()):
        positions.append(Position.from_name(names.at(k).str()))
    var mj = args.get("mask")
    var mask = List[UInt8]()
    var w = 0
    for r in range(mj.len()):
        var row = mj.at(r)
        w = row.len()
        for c in range(w):
            mask.append(UInt8(row.at(c).int()))
    var zone = PolygonZone(poly, positions^, args.get("require_all").truthy(), mask^, mj.len(), w)
    var out = Json.array()
    var frames = args.get("frames")
    for f in range(frames.len()):
        var xy = List[Float64]()
        var b = frames.at(f)
        for i in range(b.len()):
            for c in range(4):
                xy.append(b.at(i).at(c).num())
        var hit = zone.trigger(Detections(xy^))
        var o = Json.object()
        var a = Json.array()
        for h in hit:
            a.push(Json.of(h))
        o.set("hits", a^)
        o.set("count", Json.of(zone.current_count))
        out.push(o^)
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

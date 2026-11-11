from json import Json, parse
from supervision.detection.tools.slicer_offsets import generate_offsets
from rowio import read_rows


def run(args: Json) raises -> Json:
    var im = args.get("image_wh")
    var sl = args.get("slice_wh")
    var ov = args.get("overlap_wh")
    var offs = generate_offsets(im.at(0).int(), im.at(1).int(), sl.at(0).int(), sl.at(1).int(),
                                ov.at(0).int(), ov.at(1).int())
    var out = Json.array()
    for o in offs:
        var r = Json.array()
        for v in o:
            r.push(Json.of(v))
        out.push(r^)
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

from json import Json, parse
from supervision.config import ORIENTED_BOX_COORDINATES
from supervision.detection.core import Detections
from supervision.detection.tools.inference_slicer import merge_and_filter, FILTER_NONE, FILTER_NMS, FILTER_NMM
from supervision.detection.tools.slicer_offsets import generate_offsets
from supervision.detection.utils.iou_and_nms import OVERLAP_IOU, OVERLAP_IOS
from rowio import read_rows


def run(args: Json) raises -> Json:
    var im = args.get("image_wh")
    var sl = args.get("slice_wh")
    var ov = args.get("overlap_wh")
    var offs = generate_offsets(im.at(0).int(), im.at(1).int(), sl.at(0).int(), sl.at(1).int(), ov.at(0).int(), ov.at(1).int())
    var per = List[Detections]()
    var slices = args.get("slices")
    for s in range(slices.len()):
        var f = slices.at(s)
        var xyxy = List[Float64]()
        var b = f.get("xyxy")
        for i in range(b.len()):
            for c in range(4):
                xyxy.append(b.at(i).at(c).num())
        var d = Detections(xyxy^)
        if not f.get("confidence").is_null():
            var v = List[Float64]()
            for i in range(f.get("confidence").len()):
                v.append(f.get("confidence").at(i).num())
            d.set_confidence(v^)
        if not f.get("class_id").is_null():
            var v = List[Int]()
            for i in range(f.get("class_id").len()):
                v.append(f.get("class_id").at(i).int())
            d.set_class_id(v^)
        per.append(d^)
    var fname = args.get("filter").str()
    var filt = FILTER_NONE
    if fname == "NON_MAX_SUPPRESSION":
        filt = FILTER_NMS
    elif fname == "NON_MAX_MERGE":
        filt = FILTER_NMM
    var metric = OVERLAP_IOS if args.get("metric").str() == "IOS" else OVERLAP_IOU
    var r = merge_and_filter(per, offs, filt, args.get("iou").num(), metric)
    var o = Json.object()
    var xs = Json.array()
    for i in range(len(r)):
        var row = Json.array()
        for c in range(4):
            row.push(Json.of(r.xyxy[4 * i + c]))
        xs.push(row^)
    o.set("xyxy", xs^)
    var cf = Json()
    if r.has_confidence:
        cf = Json.array()
        for v in r.confidence:
            cf.push(Json.of(v))
    o.set("confidence", cf^)
    var ci = Json()
    if r.has_class_id:
        ci = Json.array()
        for v in r.class_id:
            ci.push(Json.of(v))
    o.set("class_id", ci^)
    return o^


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

# Rows: sink_csv | sink_json  <{"appends":[{xyxy,class_id,confidence,tracker_id,data,custom}...], "close": bool}> -> file text
from json import Json, parse, ARR, OBJ, STR
from supervision.detection.core import Detections, DataCol
from supervision.detection.tools.csv_sink import CSVSink
from supervision.detection.tools.json_sink import JSONSink
from rowio import read_rows


def floats(a: Json) -> List[Float64]:
    var out = List[Float64]()
    for i in range(a.len()):
        out.append(a.at(i).num())
    return out^


def ints(a: Json) -> List[Int]:
    var out = List[Int]()
    for i in range(a.len()):
        out.append(a.at(i).int())
    return out^


def detections(a: Json) raises -> Detections:
    var flat = List[Float64]()
    var xy = a.get("xyxy")
    for i in range(xy.len()):
        for c in range(4):
            flat.append(xy.at(i).at(c).num())
    var d = Detections(flat^)
    if not a.get("class_id").is_null():
        d.set_class_id(ints(a.get("class_id")))
    if not a.get("confidence").is_null():
        d.set_confidence(floats(a.get("confidence")))
    if not a.get("tracker_id").is_null():
        d.set_tracker_id(ints(a.get("tracker_id")))
    var data = a.get("data")
    for j in range(data.len()):
        var arr = data.at(j).get("nd")
        if arr.len() > 0 and arr.at(0).kind() == STR:
            var s = List[String]()
            for i in range(arr.len()):
                s.append(arr.at(i).str())
            d.set_data(DataCol(data.key_at(j), s))
        else:
            var f = List[Float64]()
            var w = 1
            for i in range(arr.len()):
                if arr.at(i).kind() == ARR:
                    w = arr.at(i).len()
                    for k in range(w):
                        f.append(arr.at(i).at(k).num())
                else:
                    f.append(arr.at(i).num())
            d.set_data(DataCol(data.key_at(j), f, w))
    d.validate()
    return d^


def read_text(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def run(op: String, args: Json, idx: Int) raises -> Json:
    var path = ".work/sink/" + op + String(idx) + (".csv" if op == "sink_csv" else ".json")
    var appends = args.get("appends")
    if op == "sink_csv":
        var sink = CSVSink(path)
        sink.open()
        for k in range(appends.len()):
            sink.append(detections(appends.at(k)), appends.at(k).get("custom"))
        sink.close()
    else:
        var sink = JSONSink(path)
        sink.open()
        for k in range(appends.len()):
            sink.append(detections(appends.at(k)), appends.at(k).get("custom"))
        sink.write_and_close()
    return Json.of(read_text(path))


def main() raises:
    var idx = 0
    for row in read_rows():
        var cols = row.split("\t")
        var res: Json
        try:
            res = run(String(cols[0]), parse(String(cols[1])), idx)
        except e:
            res = Json.object()
            res.set("error", Json.of(String("Exception")))
        idx += 1
        print(res.dumps())

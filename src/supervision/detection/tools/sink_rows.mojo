# Row building shared by CSVSink and JSONSink (port of their parse_detection_data).
# `from_detections` flattens a Detections into `SinkInput` columns (numeric data columns are Float64 in S1, so an
# int column prints as 1.0 here; pass ints through custom_data).
#
# Value encoding for `data` / `custom_data` entries (Json): {"nd": [...]} is a numpy ndarray, a bare array is a
# Python list, anything else a scalar. Array and list entries are indexed per detection when len == n.
from json import Json, ARR, OBJ
from supervision.detection.core import Detections


struct SinkInput(Copyable, Movable):
    var n: Int
    var xyxy: List[Float64]  # 4 per detection
    var class_id: Optional[List[Int]]
    var confidence: Optional[List[Float64]]
    var tracker_id: Optional[List[Int]]
    var data: Json  # object, insertion ordered
    var custom: Json  # object or null

    def __init__(out self, n: Int, var xyxy: List[Float64]):
        self.n = n
        self.xyxy = xyxy^
        self.class_id = None
        self.confidence = None
        self.tracker_id = None
        self.data = Json.object()
        self.custom = Json()


def _is_nd(v: Json) -> Bool:
    return v.kind() == OBJ and v.has("nd") and v.len() == 1


def slice_value(v: Json, i: Int, n: Int) -> Json:
    """Detections-sink `_slice_value`: ndarray or list of length n gives element i, anything else passes through."""
    if _is_nd(v):
        var a = v.get("nd")
        if a.len() != n:
            return v.copy()
        var e = a.at(i)
        if e.kind() == ARR:  # a row of a 2-D array stays an ndarray
            var w = Json.object()
            w.set("nd", e)
            return w^
        return e^
    if v.kind() == ARR and v.len() == n:
        return v.at(i)
    return v.copy()


def _plain(v: Json) -> Json:
    return v.get("nd") if _is_nd(v) else v.copy()


def base_header() -> List[String]:
    return ["x_min", "y_min", "x_max", "y_max", "class_id", "confidence", "tracker_id"]


def field_names(d: SinkInput) -> List[String]:
    """BASE_HEADER + sorted(custom keys | data keys)."""
    var keys = List[String]()
    for j in range(d.data.len()):
        keys.append(d.data.key_at(j))
    if d.custom.kind() == OBJ:
        for j in range(d.custom.len()):
            var k = d.custom.key_at(j)
            var seen = False
            for e in keys:
                if e == k:
                    seen = True
            if not seen:
                keys.append(k)
    sort(keys)
    var out = base_header()
    for k in keys:
        out.append(k)
    return out^


def parse_detection_data(d: SinkInput, as_text: Bool) -> List[Json]:
    """One object per detection. as_text: class_id, confidence, tracker_id are str() like CSVSink; else typed like JSONSink."""
    var rows = List[Json]()
    for i in range(d.n):
        var row = Json.object()
        for c in range(4):
            row.set(base_header()[c], Json.of(d.xyxy[4 * i + c]))
        if d.class_id:
            row.set("class_id", Json.of(String(d.class_id.value()[i])) if as_text else Json.of(d.class_id.value()[i]))
        else:
            row.set("class_id", Json.of(String("")))
        if d.confidence:
            var c = d.confidence.value()[i]
            row.set("confidence", Json.of(String(Json.of(c).dumps())) if as_text else Json.of(c))
        else:
            row.set("confidence", Json.of(String("")))
        if d.tracker_id:
            row.set("tracker_id", Json.of(String(d.tracker_id.value()[i])) if as_text else Json.of(d.tracker_id.value()[i]))
        else:
            row.set("tracker_id", Json.of(String("")))
        for j in range(d.data.len()):
            var cell = slice_value(d.data.at(j), i, d.n)
            row.set(d.data.key_at(j), cell^ if as_text else _plain(cell))
        if d.custom.kind() == OBJ:
            for j in range(d.custom.len()):
                var cell = slice_value(d.custom.at(j), i, d.n)
                row.set(d.custom.key_at(j), cell^ if as_text else _plain(cell))
        rows.append(row^)
    return rows^


def from_detections(d: Detections, custom: Json) raises -> SinkInput:
    var r = SinkInput(len(d), d.xyxy.copy())
    if d.has_class_id:
        r.class_id = d.class_id.copy()
    if d.has_confidence:
        r.confidence = d.confidence.copy()
    if d.has_tracker_id:
        r.tracker_id = d.tracker_id.copy()
    for col in d.data:
        var arr = Json.array()
        for i in range(col.rows()):
            if col.is_str:
                arr.push(Json.of(col.s[i]))
            elif col.width == 1:
                arr.push(Json.of(col.f[i]))
            else:
                var row = Json.array()
                for k in range(col.width):
                    row.push(Json.of(col.f[i * col.width + k]))
                arr.push(row^)
        var w = Json.object()
        w.set("nd", arr^)
        r.data.set(col.name, w^)
    r.custom = custom.copy()
    return r^

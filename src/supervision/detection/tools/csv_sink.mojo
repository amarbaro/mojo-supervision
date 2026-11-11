# Port of supervision/detection/tools/csv_sink.py CSVSink (MIT, see NOTICE-supervision-LICENSE.md).
# Output follows Python csv.writer defaults: "," delimiter, '"' quote, QUOTE_MINIMAL, "\r\n" line end.
from json import Json, ARR, OBJ, STR, INT, FLOAT, BOOL, NULL
from std.os import makedirs
from std.os.path import dirname, exists
from supervision.detection.core import Detections
from supervision.detection.tools.sink_rows import from_detections, field_names, parse_detection_data


def _csv_field(s: String) -> String:
    var need = False
    for b in s.as_bytes():
        if b == 44 or b == 34 or b == 13 or b == 10:  # , " \r \n
            need = True
    if not need:
        return s
    var out = List[UInt8]()
    out.append(34)
    for b in s.as_bytes():
        if b == 34:
            out.append(34)
        out.append(b)
    out.append(34)
    return String(unsafe_from_utf8=out)


def _cell(v: Json) raises -> String:
    """str() of a numpy/python scalar as csv.writer sees it; ndarray rows go through json.dumps like _format_cell."""
    var k = v.kind()
    if k == NULL:
        return ""
    if k == BOOL:
        return "True" if v.truthy() else "False"
    if k == STR:
        return v.str()
    if k == INT or k == FLOAT:
        return v.dumps()
    if k == OBJ and v.has("nd"):
        return v.get("nd").dumps(False)
    raise Error("csv_sink: unsupported cell (a Python list that is not per-detection)")


def _line(cells: List[String]) -> String:
    var s = String()
    for j in range(len(cells)):
        if j > 0:
            s += ","
        s += _csv_field(cells[j])
    return s + "\r\n"


struct CSVSink(Movable):
    var file_name: String
    var buf: String
    var is_open: Bool
    var header_written: Bool
    var field_names: List[String]
    var deferred_field_names: List[String]

    def __init__(out self, file_name: String = "output.csv"):
        self.file_name = file_name
        self.buf = String()
        self.is_open = False
        self.header_written = False
        self.field_names = List[String]()
        self.deferred_field_names = List[String]()

    def open(mut self) raises:
        var parent = dirname(self.file_name)
        if parent.byte_length() > 0 and not exists(parent):
            makedirs(parent, exist_ok=True)
        with open(self.file_name, "w") as f:
            pass
        self.buf = String()
        self.is_open = True
        self.header_written = False
        self.field_names = List[String]()
        self.deferred_field_names = List[String]()

    def append(mut self, detections: Detections, custom: Json = Json()) raises:
        var d = from_detections(detections, custom)
        if not self.is_open:
            raise Error("Cannot append to CSV: The file '" + self.file_name + "' is not open.")
        var names = field_names(d)
        if d.n == 0:
            if not self.header_written and len(self.deferred_field_names) == 0:
                self.deferred_field_names = names^
            return
        if not self.header_written:
            self.field_names = names^
            self.buf += _line(self.field_names)
            self.header_written = True
        # Python only logs a warning when the names differ from the header; rows still use the header's names.
        for row in parse_detection_data(d, True):
            var cells = List[String]()
            for f in self.field_names:
                cells.append(_cell(row.get(f)) if row.has(f) else String(""))
            self.buf += _line(cells)

    def close(mut self) raises:
        if not self.is_open:
            return
        self.is_open = False
        if not self.header_written and len(self.deferred_field_names) > 0:
            self.field_names = self.deferred_field_names.copy()
            self.buf += _line(self.field_names)
            self.header_written = True
        with open(self.file_name, "w") as f:
            f.write(self.buf)

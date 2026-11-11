# Port of supervision/detection/tools/json_sink.py JSONSink (MIT, see NOTICE-supervision-LICENSE.md).
from json import Json
from std.os import makedirs
from std.os.path import dirname, exists
from supervision.detection.core import Detections
from supervision.detection.tools.sink_rows import from_detections, parse_detection_data


struct JSONSink(Movable):
    var file_name: String
    var rows: Json
    var is_open: Bool

    def __init__(out self, file_name: String = "output.json"):
        self.file_name = file_name
        self.rows = Json.array()
        self.is_open = False

    def open(mut self) raises:
        var parent = dirname(self.file_name)
        if parent.byte_length() > 0 and not exists(parent):
            makedirs(parent, exist_ok=True)
        with open(self.file_name, "w") as f:  # truncates, like Python's open(..., "w")
            pass
        self.rows = Json.array()
        self.is_open = True

    def append(mut self, detections: Detections, custom: Json = Json()) raises:
        var d = from_detections(detections, custom)
        for r in parse_detection_data(d, False):
            self.rows.push(r)

    def write_and_close(mut self) raises:
        if not self.is_open:
            return
        self.is_open = False
        with open(self.file_name, "w") as f:
            f.write(self.rows.dumps(False, 4))  # json.dump(indent=4): no trailing newline

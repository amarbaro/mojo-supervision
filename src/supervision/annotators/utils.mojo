# Port of supervision/annotators/utils.py: ColorLookup, color resolution, Trace.
from supervision.geometry.core import Position
from supervision.detection.core import Detections
from supervision.draw.color import Color, ColorPalette

comptime LOOKUP_INDEX = 0
comptime LOOKUP_CLASS = 1
comptime LOOKUP_TRACK = 2
comptime PENDING_TRACK_ID = -1


def resolve_color_idx(det: Detections, i: Int, lookup: Int) raises -> Int:
    if i >= len(det):
        raise Error(String("Detection index ", i, " is out of bounds for detections of length ", len(det)))
    if lookup == LOOKUP_INDEX:
        return i
    if lookup == LOOKUP_CLASS:
        if not det.has_class_id:
            raise Error("Could not resolve color by class because Detections do not have class_id. Try color_lookup INDEX or TRACK.")
        return det.class_id[i]
    if lookup == LOOKUP_TRACK:
        if not det.has_tracker_id:
            raise Error("Could not resolve color by track because Detections do not have tracker_id.")
        return det.tracker_id[i]
    raise Error("Unsupported color lookup strategy: " + String(lookup))


def resolve_color(palette: ColorPalette, det: Detections, i: Int, lookup: Int) raises -> Color:
    var idx = resolve_color_idx(det, i, lookup)
    if lookup == LOOKUP_TRACK and idx == PENDING_TRACK_ID:
        return Color.GREY()
    return palette.by_idx(idx)


struct Trace(Movable):
    """Per-tracker anchor history over the last `max_size` frames (float32 positions, like the Python)."""

    var max_size: Int
    var anchor: Int
    var current_frame_id: Int
    var frame_id: List[Int]
    var x: List[Float32]
    var y: List[Float32]
    var tracker_id: List[Int]

    def __init__(out self, max_size: Int, anchor: Int):
        self.max_size = max_size
        self.anchor = anchor
        self.current_frame_id = 0
        self.frame_id = List[Int]()
        self.x = List[Float32]()
        self.y = List[Float32]()
        self.tracker_id = List[Int]()

    def put(mut self, det: Detections) raises:
        if len(det) > 0:
            if not det.has_tracker_id:
                raise Error("Could not put detections into Trace because Detections do not have tracker_id.")
            var xy = det.get_anchors_coordinates(Position(self.anchor))
            for i in range(len(det)):
                self.frame_id.append(self.current_frame_id)
                self.x.append(Float32(xy[2 * i]))
                self.y.append(Float32(xy[2 * i + 1]))
                self.tracker_id.append(det.tracker_id[i])
        var distinct = List[Int]()
        for f in self.frame_id:
            var seen = False
            for g in distinct:
                if g == f:
                    seen = True
            if not seen:
                distinct.append(f)
        if 0 < self.max_size and self.max_size < len(distinct):
            var lowest = self.current_frame_id - self.max_size + 1
            var fi = List[Int]()
            var xs = List[Float32]()
            var ys = List[Float32]()
            var ti = List[Int]()
            for k in range(len(self.frame_id)):
                if self.frame_id[k] >= lowest:
                    fi.append(self.frame_id[k])
                    xs.append(self.x[k])
                    ys.append(self.y[k])
                    ti.append(self.tracker_id[k])
            self.frame_id = fi^
            self.x = xs^
            self.y = ys^
            self.tracker_id = ti^
        self.current_frame_id += 1

    def get(self, tracker_id: Int, mut xs: List[Float32], mut ys: List[Float32]):
        xs.clear()
        ys.clear()
        for k in range(len(self.tracker_id)):
            if self.tracker_id[k] == tracker_id:
                xs.append(self.x[k])
                ys.append(self.y[k])

    def reset(mut self):
        self.current_frame_id = 0
        self.frame_id.clear()
        self.x.clear()
        self.y.clear()
        self.tracker_id.clear()

# Port of supervision/detection/line_zone.py LineZone (MIT, see NOTICE-supervision-LICENSE.md).
# The annotators in that file are S5's. coord_transform is not ported (CoordinatesTransformation is not in geometry yet).
from supervision.detection.core import Detections
from supervision.geometry.core import Point, Position, Vector


struct ClassCount(Copyable, Movable):
    """Insertion-ordered Counter[int | None]."""

    var keys: List[Optional[Int]]
    var counts: List[Int]

    def __init__(out self):
        self.keys = List[Optional[Int]]()
        self.counts = List[Int]()

    def add(mut self, k: Optional[Int]):
        for j in range(len(self.keys)):
            if self.keys[j] == k:
                self.counts[j] += 1
                return
        self.keys.append(k)
        self.counts.append(1)

    def total(self) -> Int:
        var t = 0
        for c in self.counts:
            t += c
        return t


struct LineZone(Movable):
    var vector: Vector
    var limit_start: Vector
    var limit_end: Vector
    var triggering_anchors: List[Position]
    var history_length: Int
    var history: Dict[Int, List[Bool]]
    var confirmed_side: Dict[Int, Bool]
    var frames_absent: Dict[Int, Int]
    var in_counts: ClassCount
    var out_counts: ClassCount

    def __init__(
        out self,
        start: Point,
        end: Point,
        var triggering_anchors: List[Position],
        minimum_crossing_threshold: Int = 1,
    ) raises:
        if len(triggering_anchors) == 0:
            raise Error("Triggering anchors cannot be empty.")
        self.triggering_anchors = triggering_anchors^
        self.vector = Vector(start.copy(), end.copy())
        var mag = self.vector.magnitude()
        if mag == 0:
            raise Error("The magnitude of the vector cannot be zero.")
        var ux = (end.x - start.x) / mag
        var uy = (end.y - start.y) / mag
        var px = -uy
        var py = ux
        self.limit_start = Vector(start.copy(), Point(start.x + px, start.y + py))
        self.limit_end = Vector(end.copy(), Point(end.x - px, end.y - py))
        self.history_length = max(2, minimum_crossing_threshold + 1)
        self.history = Dict[Int, List[Bool]]()
        self.confirmed_side = Dict[Int, Bool]()
        self.frames_absent = Dict[Int, Int]()
        self.in_counts = ClassCount()
        self.out_counts = ClassCount()

    def in_count(self) -> Int:
        return self.in_counts.total()

    def out_count(self) -> Int:
        return self.out_counts.total()

    def _evict(mut self, current: List[Int]) raises:
        var keys = List[Int]()
        for k in self.history.keys():
            keys.append(k)
        for key in keys:
            var present = False
            for c in current:
                if c == key:
                    present = True
                    break
            if present:
                _ = self.frames_absent.pop(key, 0)
            else:
                var absent = self.frames_absent.get(key, 0) + 1
                if absent >= self.history_length:
                    _ = self.history.pop(key)
                    _ = self.confirmed_side.pop(key, False)
                    _ = self.frames_absent.pop(key, 0)
                else:
                    self.frames_absent[key] = absent

    def trigger(mut self, d: Detections) raises -> Tuple[List[Bool], List[Bool]]:
        """Returns (crossed_in, crossed_out). Without tracker_id nothing counts (Python warns)."""
        var n = len(d)
        var crossed_in = List[Bool](length=n, fill=False)
        var crossed_out = List[Bool](length=n, fill=False)
        if n == 0 or not d.has_tracker_id:
            self._evict(List[Int]())
            return (crossed_in^, crossed_out^)
        var tracker_id = d.tracker_id.copy()
        var anchors = List[List[Float64]]()  # anchors[a] = flat x, y per detection
        for p in self.triggering_anchors:
            anchors.append(d.get_anchors_coordinates(p))
        var current = List[Int]()
        for i in range(n):
            if tracker_id[i] >= 0:
                current.append(tracker_id[i])
        self._evict(current)
        for i in range(n):
            var tid = tracker_id[i]
            if tid < 0:
                continue
            var in_limits = True
            var left = False
            var right = False
            for a in range(len(anchors)):
                var p = Point(anchors[a][2 * i], anchors[a][2 * i + 1])
                var c1 = self.limit_start.cross_product(p) > 0
                var c2 = self.limit_end.cross_product(p) > 0
                if c1 != c2:
                    in_limits = False
                if self.vector.cross_product(p) < 0:
                    left = True
                else:
                    right = True
            if not in_limits or (left and right):
                continue
            var state = left
            if tid not in self.history:
                self.history[tid] = List[Bool]()
            self.history[tid].append(state)
            if len(self.history[tid]) > self.history_length:
                _ = self.history[tid].pop(0)
            if tid not in self.confirmed_side:
                self.confirmed_side[tid] = state
            var confirmed = self.confirmed_side[tid]
            if len(self.history[tid]) < self.history_length:
                continue
            if state == confirmed:
                continue
            var sustained = True
            for j in range(1, len(self.history[tid])):
                if self.history[tid][j] != state:
                    sustained = False
            if not sustained:
                continue
            self.confirmed_side[tid] = state
            var cls = Optional[Int]()
            if d.has_class_id:
                cls = Optional[Int](d.class_id[i])
            if state:
                self.in_counts.add(cls)
                crossed_in[i] = True
            else:
                self.out_counts.add(cls)
                crossed_out[i] = True
        return (crossed_in^, crossed_out^)

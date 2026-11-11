# Port of supervision/detection/tools/polygon_zone.py PolygonZone (MIT, see NOTICE-supervision-LICENSE.md).
# PolygonZoneAnnotator is S5's. coord_transform is not ported (CoordinatesTransformation is not in geometry yet).
# PARTIAL: the zone mask comes from `polygon_to_mask` (S1 converters, cv2.fillPoly). Until that lands the mask is
# passed in; `PolygonZone.__init__` then fills it from the polygon.
from std.math import isfinite
from supervision.detection.core import Detections
from supervision.detection.utils.converters import MIN_POLYGON_POINT_COUNT
from supervision.geometry.core import Position
from supervision.geometry.utils import round_half_even


struct PolygonZone(Movable):
    var polygon: List[Int]  # x, y pairs, truncated to int like polygon.astype(int)
    var triggering_anchors: List[Position]
    var require_all_anchors: Bool
    var current_count: Int
    var mask: List[UInt8]  # mask_h * mask_w, row major, 0 or 1
    var mask_h: Int
    var mask_w: Int

    def __init__(
        out self,
        polygon: List[Float64],
        var triggering_anchors: List[Position],
        require_all_anchors: Bool,
        var mask: List[UInt8],
        mask_h: Int,
        mask_w: Int,
    ) raises:
        if len(polygon) % 2 != 0:
            raise Error("Polygon must have shape (N, 2); got an odd number of coordinates.")
        if len(polygon) // 2 < MIN_POLYGON_POINT_COUNT:
            raise Error("Polygon must have at least 3 vertices to enclose an area.")
        if len(triggering_anchors) == 0:
            raise Error("Triggering anchors cannot be empty.")
        if len(mask) != mask_h * mask_w:
            raise Error("mask size does not match mask_h * mask_w")
        self.polygon = List[Int]()
        for v in polygon:
            self.polygon.append(Int(v))
        self.triggering_anchors = triggering_anchors^
        self.require_all_anchors = require_all_anchors
        self.current_count = 0
        self.mask = mask^
        self.mask_h = mask_h
        self.mask_w = mask_w

    def trigger(mut self, detections: Detections) raises -> List[Bool]:
        var n = len(detections)
        if n == 0:
            self.current_count = 0
            return List[Bool]()
        var hits = List[Bool](length=n, fill=self.require_all_anchors)
        for p in self.triggering_anchors:
            var xy = detections.get_anchors_coordinates(p)
            for i in range(n):
                var inside = False
                var fx = xy[2 * i]
                var fy = xy[2 * i + 1]
                if isfinite(fx) and isfinite(fy):
                    var x = Int(round_half_even(fx))
                    var y = Int(round_half_even(fy))
                    inside = x >= 0 and y >= 0 and x < self.mask_w and y < self.mask_h and self.mask[y * self.mask_w + x] != 0
                hits[i] = (hits[i] and inside) if self.require_all_anchors else (hits[i] or inside)
        var count = 0
        for h in hits:
            if h:
                count += 1
        self.current_count = count
        return hits^

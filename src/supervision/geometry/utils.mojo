# Port of supervision/geometry/utils.py: get_polygon_center.
from std.math import floor
from supervision.geometry.core import Point


def round_half_even(v: Float64) -> Float64:
    """numpy `round` (ties to even), which Mojo's `round` (ties away from zero) is not."""
    var f = floor(v)
    var d = v - f
    if d < 0.5:
        return f
    if d > 0.5:
        return f + 1.0
    return f if Int(f) % 2 == 0 else f + 1.0


def get_polygon_center(polygon: List[Point]) raises -> Point:
    """Centre of the solid figure formed by the polygon (area-weighted triangle centroids), rounded half-even."""
    var n = len(polygon)
    if n == 0:
        raise Error("Polygon must have at least one vertex.")
    var ox = polygon[0].x
    var oy = polygon[0].y
    var area_sum = 0.0
    var areas = List[Float64](length=n, fill=0.0)
    for i in range(n):
        var j = (i + 1) % n
        var ax = polygon[i].x - ox
        var ay = polygon[i].y - oy
        var bx = polygon[j].x - ox
        var by = polygon[j].y - oy
        areas[i] = (ax * by - ay * bx) / 2
        area_sum += areas[i]
    if area_sum == 0:
        var mx = 0.0
        var my = 0.0
        for i in range(n):
            mx += polygon[i].x - ox
            my += polygon[i].y - oy
        return Point(round_half_even(mx / Float64(n) + ox), round_half_even(my / Float64(n) + oy))
    var cx = 0.0
    var cy = 0.0
    for i in range(n):
        var j = (i + 1) % n
        cx += ((polygon[i].x - ox + polygon[j].x - ox) / 3.0) * areas[i]
        cy += ((polygon[i].y - oy + polygon[j].y - oy) / 3.0) * areas[i]
    return Point(round_half_even(cx / area_sum + ox), round_half_even(cy / area_sum + oy))

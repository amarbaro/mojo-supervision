"""Box helpers (port of supervision/detection/utils/boxes.py).

A set of N boxes is a flat `List[Float64]` of length 4N (x1, y1, x2, y2 per box); oriented boxes are
a flat list of length 8N (four (x, y) corners per box). Every coordinate is a Float64: the Python
integer-overflow branches (object-dtype arithmetic) are not needed here.
"""

from std.math import sqrt
from supervision.geometry.core import Position


def clip_boxes(xyxy: List[Float64], width: Int, height: Int) -> List[Float64]:
    """Clip x to [0, width] and y to [0, height]."""
    var out = xyxy.copy()
    for i in range(len(out) // 4):
        for k in range(4):
            var hi = Float64(width) if k % 2 == 0 else Float64(height)
            var v = out[4 * i + k]
            out[4 * i + k] = 0.0 if v < 0.0 else (hi if v > hi else v)
    return out^


def pad_boxes(xyxy: List[Float64], px: Float64, py: Float64) -> List[Float64]:
    """Grow every box by px left and right and py top and bottom (negative shrinks)."""
    var out = xyxy.copy()
    for i in range(len(out) // 4):
        out[4 * i] -= px
        out[4 * i + 1] -= py
        out[4 * i + 2] += px
        out[4 * i + 3] += py
    return out^


def denormalize_boxes(xyxy: List[Float64], width: Int, height: Int, normalization_factor: Float64 = 1.0) -> List[Float64]:
    var out = xyxy.copy()
    for i in range(len(out) // 4):
        for k in range(4):
            var s = Float64(width) if k % 2 == 0 else Float64(height)
            out[4 * i + k] = out[4 * i + k] * (s / normalization_factor)
    return out^


def sort_box_corners(xyxy: List[Float64]) -> List[Float64]:
    """Order each box so that x1 <= x2 and y1 <= y2."""
    var out = xyxy.copy()
    for i in range(len(out) // 4):
        var a = xyxy[4 * i]
        var b = xyxy[4 * i + 1]
        var c = xyxy[4 * i + 2]
        var d = xyxy[4 * i + 3]
        out[4 * i] = a if a < c else c
        out[4 * i + 1] = b if b < d else d
        out[4 * i + 2] = a if a > c else c
        out[4 * i + 3] = b if b > d else d
    return out^


def move_boxes(xyxy: List[Float64], dx: Float64, dy: Float64) -> List[Float64]:
    var out = xyxy.copy()
    for i in range(len(out) // 4):
        out[4 * i] += dx
        out[4 * i + 1] += dy
        out[4 * i + 2] += dx
        out[4 * i + 3] += dy
    return out^


def move_oriented_boxes(xyxyxyxy: List[Float64], dx: Float64, dy: Float64) -> List[Float64]:
    var out = xyxyxyxy.copy()
    for i in range(len(out) // 2):
        out[2 * i] += dx
        out[2 * i + 1] += dy
    return out^


def scale_boxes(xyxy: List[Float64], factor: Float64) -> List[Float64]:
    """Scale every box around its centre."""
    var out = xyxy.copy()
    for i in range(len(out) // 4):
        var cx = (xyxy[4 * i] + xyxy[4 * i + 2]) / 2
        var cy = (xyxy[4 * i + 1] + xyxy[4 * i + 3]) / 2
        var w = (xyxy[4 * i + 2] - xyxy[4 * i]) * factor
        var h = (xyxy[4 * i + 3] - xyxy[4 * i + 1]) * factor
        out[4 * i] = cx - w / 2
        out[4 * i + 1] = cy - h / 2
        out[4 * i + 2] = cx + w / 2
        out[4 * i + 3] = cy + h / 2
    return out^


def obb_polygon_area(corners: List[Float64]) -> List[Float64]:
    """Area of every quadrilateral (shoelace, translated to its first corner like the oracle)."""
    var n = len(corners) // 8
    var out = List[Float64]()
    for i in range(n):
        var ox = corners[8 * i]
        var oy = corners[8 * i + 1]
        var s = 0.0
        for k in range(4):
            var j = (k + 1) % 4
            var xk = corners[8 * i + 2 * k] - ox
            var yk = corners[8 * i + 2 * k + 1] - oy
            var xj = corners[8 * i + 2 * j] - ox
            var yj = corners[8 * i + 2 * j + 1] - oy
            s += xk * yj - yk * xj
        out.append(0.5 * abs(s))
    return out^


def xyxyxyxy_to_xyxy(corners: List[Float64]) -> List[Float64]:
    var out = List[Float64]()
    for i in range(len(corners) // 8):
        var x0 = corners[8 * i]
        var y0 = corners[8 * i + 1]
        var x1 = x0
        var y1 = y0
        for k in range(1, 4):
            var x = corners[8 * i + 2 * k]
            var y = corners[8 * i + 2 * k + 1]
            x0 = x if x < x0 else x0
            x1 = x if x > x1 else x1
            y0 = y if y < y0 else y0
            y1 = y if y > y1 else y1
        out.append(x0)
        out.append(y0)
        out.append(x1)
        out.append(y1)
    return out^


def box_midpoint(first: Float64, second: Float64) -> Float64:
    return (first + second) / 2


def _anchor_offset(anchor: Position) raises -> Tuple[Float64, Float64]:
    if anchor == Position.CENTER:
        return (0.0, 0.0)
    if anchor == Position.CENTER_LEFT:
        return (-1.0, 0.0)
    if anchor == Position.CENTER_RIGHT:
        return (1.0, 0.0)
    if anchor == Position.TOP_CENTER:
        return (0.0, -1.0)
    if anchor == Position.BOTTOM_CENTER:
        return (0.0, 1.0)
    if anchor == Position.TOP_LEFT:
        return (-1.0, -1.0)
    if anchor == Position.TOP_RIGHT:
        return (1.0, -1.0)
    if anchor == Position.BOTTOM_LEFT:
        return (-1.0, 1.0)
    if anchor == Position.BOTTOM_RIGHT:
        return (1.0, 1.0)
    raise Error(anchor.name() + " is not supported.")


def oriented_box_anchors(corners: List[Float64], anchor: Position) raises -> List[Float64]:
    """Anchor point of every oriented box: (x, y) pairs, flat."""
    var off = _anchor_offset(anchor)
    var sx = off[0]
    var sy = off[1]
    var out = List[Float64]()
    for i in range(len(corners) // 8):
        var b = 8 * i
        var cx = (corners[b] + corners[b + 2] + corners[b + 4] + corners[b + 6]) / 4
        var cy = (corners[b + 1] + corners[b + 3] + corners[b + 5] + corners[b + 7]) / 4
        var ax = (corners[b + 2] - corners[b]) / 2
        var ay = (corners[b + 3] - corners[b + 1]) / 2
        var bx = (corners[b + 4] - corners[b + 2]) / 2
        var by = (corners[b + 5] - corners[b + 3]) / 2
        var wx = ax
        var wy = ay
        var hx = bx
        var hy = by
        if not (abs(ax) >= abs(bx)):
            wx = bx
            wy = by
            hx = ax
            hy = ay
        if wx < 0:
            wx = -wx
            wy = -wy
        if hy < 0:
            hx = -hx
            hy = -hy
        out.append(cx + sx * wx + sy * hx)
        out.append(cy + sx * wy + sy * hy)
    return out^

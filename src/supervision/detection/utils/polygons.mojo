"""polygon_to_mask of supervision/detection/utils/converters.py, on the cv2 layer (supervision._cv.draw.fill_poly)."""

from mojocv.imgproc import find_contours
from mojocv.mat import Mat
from supervision._cv.draw import fill_poly
from supervision.detection.utils.converters import MIN_POLYGON_POINT_COUNT


def polygon_to_mask(polygon: List[Float64], width: Int, height: Int) -> List[UInt8]:
    """(height x width) 0/1 mask of the polygon given as flat (x, y) pairs; coordinates truncate to int like astype(int32).
    Fewer than three points gives an empty mask."""
    var m = Mat[DType.uint8](height, width, 1)
    var n = len(polygon) // 2
    if n >= MIN_POLYGON_POINT_COUNT:
        var xs = List[Int]()
        var ys = List[Int]()
        for i in range(n):
            xs.append(Int(polygon[2 * i]))
            ys.append(Int(polygon[2 * i + 1]))
        fill_poly(m, [xs^], [ys^], [1.0])
    var out = List[UInt8]()
    for i in range(height * width):
        out.append(UInt8(m.data[i]))
    return out^


def mask_to_polygons(mask: List[UInt8], height: Int, width: Int) -> List[List[Int]]:
    """Every contour of the mask (RETR_TREE, CHAIN_APPROX_SIMPLE: holes too) with at least three points, as flat x, y ints."""
    var m = Mat[DType.uint8](height, width, 1)
    for i in range(height * width):
        m.data[i] = 1 if mask[i] != 0 else 0
    var r = find_contours(m, 3, True)
    var out = List[List[Int]]()
    var at = 0
    for k in range(r[1].rows):
        var n = Int(r[1].data[k])
        if n >= MIN_POLYGON_POINT_COUNT:
            var p = List[Int]()
            for j in range(2 * at, 2 * (at + n)):
                p.append(Int(r[0].data[j]))
            out.append(p^)
        at += n
    return out^


def polygon_area(p: List[Int]) -> Float64:
    """cv2.contourArea."""
    var n = len(p) // 2
    var a = 0.0
    for i in range(n):
        var j = (i + 1) % n
        a += Float64(p[2 * i]) * Float64(p[2 * j + 1]) - Float64(p[2 * j]) * Float64(p[2 * i + 1])
    return abs(a * 0.5)


def filter_polygons_by_area(polygons: List[List[Int]], min_area: Optional[Float64], max_area: Optional[Float64]) -> List[List[Int]]:
    var out = List[List[Int]]()
    for p in polygons:
        var a = polygon_area(p)
        if (not min_area or a >= min_area.value()) and (not max_area or a <= max_area.value()):
            out.append(p.copy())
    return out^


def _approx_poly_dp(src: List[Int], eps_in: Float64) -> List[Int]:
    """cv2.approxPolyDP on a closed integer contour (flat x, y), a port of imgproc/src/approx.cpp (Ramer-Douglas-Peucker
    from two far points, then the clean-up of nearly straight corners)."""
    var count = len(src) // 2
    var eps = eps_in * eps_in
    var dst = List[Int](length=2 * count, fill=0)
    var new_count = 0
    var stack = List[Int]()  # (start, end) pairs
    var pos = 0
    var rs = 0
    var sx = 0
    var sy = 0
    var le_eps = False
    for _ in range(3):
        var max_dist = 0.0
        pos = (pos + rs) % count
        sx = src[2 * pos]
        sy = src[2 * pos + 1]
        pos = (pos + 1) % count
        for j in range(1, count):
            var dx = Float64(src[2 * pos] - sx)
            var dy = Float64(src[2 * pos + 1] - sy)
            pos = (pos + 1) % count
            var dist = dx * dx + dy * dy
            if dist > max_dist:
                max_dist = dist
                rs = j
        le_eps = max_dist <= eps
    if not le_eps:
        var slice_start = pos % count
        var re = slice_start
        var se = (rs + slice_start) % count
        rs = se
        stack.append(rs)  # right slice: (rs, re)
        stack.append(re)
        stack.append(slice_start)  # slice: (slice_start, se)
        stack.append(se)
    else:
        dst[0] = sx
        dst[1] = sy
        new_count = 1
    while len(stack) > 0:
        var se = stack.pop()
        var ss = stack.pop()
        var ex = src[2 * se]
        var ey = src[2 * se + 1]
        pos = ss
        sx = src[2 * pos]
        sy = src[2 * pos + 1]
        pos = (pos + 1) % count
        if pos != se:
            var dx = Float64(ex - sx)
            var dy = Float64(ey - sy)
            var seg2 = dx * dx + dy * dy
            var best = 0.0
            while pos != se:
                var px = Float64(src[2 * pos])
                var py = Float64(src[2 * pos + 1])
                pos = (pos + 1) % count
                var proj = (px - Float64(sx)) * dx + (py - Float64(sy)) * dy
                var d2: Float64
                if proj < 0:
                    d2 = ((px - Float64(sx)) ** 2 + (py - Float64(sy)) ** 2) * seg2
                elif proj > seg2:
                    d2 = ((px - Float64(ex)) ** 2 + (py - Float64(ey)) ** 2) * seg2
                else:
                    var d = (py - Float64(sy)) * dx - (px - Float64(sx)) * dy
                    d2 = d * d
                if d2 > best:
                    best = d2
                    rs = (pos + count - 1) % count
            le_eps = best <= eps * seg2
        else:
            le_eps = True
            sx = src[2 * ss]
            sy = src[2 * ss + 1]
        if le_eps:
            dst[2 * new_count] = sx
            dst[2 * new_count + 1] = sy
            new_count += 1
        else:
            stack.append(rs)
            stack.append(se)
            stack.append(ss)
            stack.append(rs)
    # clean-up: drop points on [almost] straight lines
    count = new_count
    pos = count - 1
    sx = dst[2 * pos]
    sy = dst[2 * pos + 1]
    pos = (pos + 1) % count
    var wpos = pos
    var px = dst[2 * pos]
    var py = dst[2 * pos + 1]
    pos = (pos + 1) % count
    var i = 0
    while i < count and new_count > 2:
        var ex = dst[2 * pos]
        var ey = dst[2 * pos + 1]
        pos = (pos + 1) % count
        var dx = Float64(ex - sx)
        var dy = Float64(ey - sy)
        var dist = abs(Float64(px - sx) * dy - Float64(py - sy) * dx)
        var inner = Float64(px - sx) * Float64(ex - px) + Float64(py - sy) * Float64(ey - py)
        if dist * dist <= 0.5 * eps * (dx * dx + dy * dy) and dx != 0 and dy != 0 and inner >= 0:
            new_count -= 1
            sx = ex
            sy = ey
            dst[2 * wpos] = sx
            dst[2 * wpos + 1] = sy
            wpos = (wpos + 1) % count
            px = dst[2 * pos]
            py = dst[2 * pos + 1]
            pos = (pos + 1) % count
            i += 2
            continue
        sx = px
        sy = py
        dst[2 * wpos] = sx
        dst[2 * wpos + 1] = sy
        wpos = (wpos + 1) % count
        px = ex
        py = ey
        i += 1
    return List[Int](dst[: 2 * new_count])


def approximate_polygon(polygon: List[Int], percentage: Float64, epsilon_step: Float64 = 0.05) raises -> List[Int]:
    """Douglas-Peucker with a growing epsilon until at most (1 - percentage) of the points are left (never below 3)."""
    if percentage < 0 or percentage >= 1:
        raise Error("Percentage must be in the range [0, 1).")
    if epsilon_step <= 0:
        raise Error("epsilon_step must be positive.")
    var target = max(Int(Float64(len(polygon) // 2) * (1 - percentage)), 3)
    var cur = polygon.copy()
    var eps = 0.0
    while len(cur) // 2 > target:
        eps += epsilon_step
        var cand = _approx_poly_dp(polygon, eps)
        if len(cand) // 2 < 3:
            break
        cur = cand^
    return cur^


def approximate_mask_with_polygons(
    mask: List[UInt8], height: Int, width: Int, min_image_area_percentage: Float64 = 0.0,
    max_image_area_percentage: Float64 = 1.0, approximation_percentage: Float64 = 0.0,
) raises -> List[List[Int]]:
    """dataset.utils.approximate_mask_with_polygons without bridge_holes: area filter, then simplification."""
    var image_area = Float64(height * width)
    var polygons = mask_to_polygons(mask, height, width)
    if len(polygons) == 1:
        polygons = filter_polygons_by_area(polygons, Optional[Float64](), Optional[Float64](max_image_area_percentage * image_area))
    else:
        polygons = filter_polygons_by_area(
            polygons, Optional[Float64](min_image_area_percentage * image_area), Optional[Float64](max_image_area_percentage * image_area)
        )
    var out = List[List[Int]]()
    for p in polygons:
        out.append(approximate_polygon(p, approximation_percentage))
    return out^

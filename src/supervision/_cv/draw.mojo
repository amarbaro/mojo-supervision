# Port of the integer drawing paths of OpenCV 5.x modules/imgproc/src/drawing.cpp (Apache-2.0), LINE_8 only:
# circle, ellipse (+arcs), polylines, fillPoly. Line, Line2, ThickLine's rectangle, FillConvexPoly and the filled Circle come
# from mojo-cv (`mojocv.imgproc`), which already matches cv2 on rows; this file adds the fixed-point (16.16) ThickLine/PolyLine
# that ellipses need, the thin Circle, ellipse2Poly, and the edge-list polygon fill.
# Not ported: LINE_AA lines (supervision draws no AA geometry), drawMarker, multi-contour offsets.
from mojocv.mat import Mat
from mojocv.imgproc import clip_line, draw_line, rint, _fill_convex, _circle_fill, _hline, _put, _tdiv
from supervision._cv.sintable import sin_table

comptime Img = Mat[DType.uint8]
comptime XY_SHIFT = 16
comptime XY_ONE = 1 << 16
comptime INT_MAX = 2147483647


def _thick_fx(mut m: Img, p0x_in: Int, p0y_in: Int, p1x_in: Int, p1y_in: Int, color: List[Float64], thickness: Int, flags: Int):
    """ThickLine with endpoints already in 16.16 (shift == XY_SHIFT)."""
    var p0x = p0x_in
    var p0y = p0y_in
    var p1x = p1x_in
    var p1y = p1y_in
    if thickness <= 1:
        draw_line(m, (p0x + (XY_ONE >> 1)) >> XY_SHIFT, (p0y + (XY_ONE >> 1)) >> XY_SHIFT, (p1x + (XY_ONE >> 1)) >> XY_SHIFT, (p1y + (XY_ONE >> 1)) >> XY_SHIFT, color, 1)
        return
    var dx = Float64(p0x - p1x) / Float64(XY_ONE)
    var dy = Float64(p1y - p0y) / Float64(XY_ONE)
    var r = dx * dx + dy * dy
    var odd = thickness & 1
    var th = thickness << (XY_SHIFT - 1)
    if abs(r) > 2.220446049250313e-16:
        r = (Float64(th) + Float64(odd) * Float64(XY_ONE) * 0.5) / (r ** 0.5)
        var dpx = rint(dy * r)
        var dpy = rint(dx * r)
        var vx: List[Int] = [p0x + dpx, p0x - dpx, p1x - dpx, p1x + dpx]
        var vy: List[Int] = [p0y + dpy, p0y - dpy, p1y - dpy, p1y + dpy]
        _fill_convex(m, vx, vy, color)
    for i in range(2):
        if (flags & (i + 1)) != 0:
            _circle_fill(m, (p0x + (XY_ONE >> 1)) >> XY_SHIFT, (p0y + (XY_ONE >> 1)) >> XY_SHIFT, (th + (XY_ONE >> 1)) >> XY_SHIFT, color)
        p0x = p1x
        p0y = p1y


def _thick_int(mut m: Img, x1: Int, y1: Int, x2: Int, y2: Int, color: List[Float64], thickness: Int, flags: Int):
    """ThickLine with shift == 0: thick lines leaving the image are first clipped with a `thickness` margin."""
    var a = x1
    var b = y1
    var c = x2
    var d = y2
    if thickness > 1 and not (0 <= a < m.cols and 0 <= b < m.rows and 0 <= c < m.cols and 0 <= d < m.rows):
        var mg = thickness
        a += mg
        b += mg
        c += mg
        d += mg
        _ = clip_line(m.cols + 2 * mg, m.rows + 2 * mg, a, b, c, d)
        a -= mg
        b -= mg
        c -= mg
        d -= mg
    _thick_fx(m, a << XY_SHIFT, b << XY_SHIFT, c << XY_SHIFT, d << XY_SHIFT, color, thickness, flags)


def poly_line(mut m: Img, vx: List[Int], vy: List[Int], closed: Bool, color: List[Float64], thickness: Int, shift: Int):
    """cv2 PolyLine: every segment is a ThickLine; interior joints get a round cap through `flags`."""
    var count = len(vx)
    if count <= 0:
        return
    var i = count - 1 if closed else 0
    var flags = 2 + (0 if closed else 1)
    var p0x = vx[i]
    var p0y = vy[i]
    for k in range(0 if closed else 1, count):
        var px = vx[k]
        var py = vy[k]
        if shift == 0:
            _thick_int(m, p0x, p0y, px, py, color, thickness, flags)
        else:
            _thick_fx(m, p0x << (XY_SHIFT - shift), p0y << (XY_SHIFT - shift), px << (XY_SHIFT - shift), py << (XY_SHIFT - shift), color, thickness, flags)
        p0x = px
        p0y = py
        flags = 2


def ellipse2poly(cx: Float64, cy: Float64, aw: Float64, ah: Float64, angle_in: Int, arc_start_in: Int, arc_end_in: Int, delta: Int, mut px: List[Float64], mut py: List[Float64]) raises:
    """cv2 ellipse2Poly(Point2d, ...): polygon of an elliptic arc, degree steps from the float32 sine table."""
    var sin = sin_table()
    var angle = angle_in
    var arc_start = arc_start_in
    var arc_end = arc_end_in
    while angle < 0:
        angle += 360
    while angle > 360:
        angle -= 360
    if arc_start > arc_end:
        var t = arc_start
        arc_start = arc_end
        arc_end = t
    while arc_start < 0:
        arc_start += 360
        arc_end += 360
    while arc_end > 360:
        arc_end -= 360
        arc_start -= 360
    if arc_end - arc_start > 360:
        arc_start = 0
        arc_end = 360
    var a2 = angle + (360 if angle < 0 else 0)
    var alpha = Float64(sin[450 - a2])
    var beta = Float64(sin[a2])
    px.clear()
    py.clear()
    var i = arc_start
    while i < arc_end + delta:
        var a = i
        if a > arc_end:
            a = arc_end
        if a < 0:
            a += 360
        var x = aw * Float64(sin[450 - a])
        var y = ah * Float64(sin[a])
        px.append(cx + x * alpha - y * beta)
        py.append(cy + x * beta + y * alpha)
        i += delta
    if len(px) == 1:
        px.append(cx)
        py.append(cy)
        px[0] = cx
        py[0] = cy


struct _Edges(Movable):
    var y0: List[Int]
    var y1: List[Int]
    var x: List[Int]
    var dx: List[Int]

    def __init__(out self):
        self.y0 = List[Int]()
        self.y1 = List[Int]()
        self.x = List[Int]()
        self.dx = List[Int]()


def collect_poly_edges(mut m: Img, vx: List[Int], vy: List[Int], mut edges: _Edges, color: List[Float64], shift: Int, offx: Int = 0, offy: Int = 0):
    """cv2 CollectPolyEdges for LINE_8: outline with Line, record non-horizontal edges as 16.16 DDA edges."""
    var count = len(vx)
    var delta = offy + ((1 << shift) >> 1)
    var p0x = (vx[count - 1] + offx) << (XY_SHIFT - shift)
    var p0y = (vy[count - 1] + delta) >> shift
    for i in range(count):
        var p1x = (vx[i] + offx) << (XY_SHIFT - shift)
        var p1y = (vy[i] + delta) >> shift
        var pt0cx = p0x
        var pt0cy = p0y
        var pt1cx = p1x
        var pt1cy = p1y
        var t0x = (p0x + (XY_ONE >> 1)) >> XY_SHIFT
        var t1x = (p1x + (XY_ONE >> 1)) >> XY_SHIFT
        var t0y = p0y
        var t1y = p1y
        draw_line(m, t0x, t0y, t1x, t1y, color, 1)
        if t0x < 0 or t0x >= m.cols or t1x < 0 or t1x >= m.cols or t0y < 0 or t0y >= m.rows or t1y < 0 or t1y >= m.rows:
            _ = clip_line(m.cols, m.rows, t0x, t0y, t1x, t1y)
            if t0y != t1y:
                pt0cy = t0y
                pt1cy = t1y
        pt0cx = t0x << XY_SHIFT
        pt1cx = t1x << XY_SHIFT
        if p0y != p1y:
            var edx = _tdiv(pt1cx - pt0cx, pt1cy - pt0cy)
            if p0y < p1y:
                edges.y0.append(p0y)
                edges.y1.append(p1y)
                edges.x.append(pt0cx + (p0y - pt0cy) * edx)
            else:
                edges.y0.append(p1y)
                edges.y1.append(p0y)
                edges.x.append(pt1cx + (p1y - pt1cy) * edx)
            edges.dx.append(edx)
        p0x = p1x
        p0y = p1y


def fill_edge_collection(mut m: Img, edges: _Edges, color: List[Float64]):
    """cv2 FillEdgeCollection: even-odd scanline fill with a sorted active-edge list (index linked list)."""
    var total = len(edges.y0)
    if total < 2:
        return
    var y_max = -INT_MAX
    var y_min = INT_MAX
    var x_max = -(1 << 62)
    var x_min = 1 << 62
    for i in range(total):
        var x1 = edges.x[i] + (edges.y1[i] - edges.y0[i]) * edges.dx[i]
        y_min = min(y_min, edges.y0[i])
        y_max = max(y_max, edges.y1[i])
        x_min = min(x_min, min(edges.x[i], x1))
        x_max = max(x_max, max(edges.x[i], x1))
    if y_max < 0 or y_min >= m.rows or x_max < 0 or x_min >= (m.cols << XY_SHIFT):
        return
    # sort by (y0, x, dx); insertion sort, edge counts are small
    var order = List[Int]()
    for i in range(total):
        order.append(i)
    for i in range(1, total):
        var cur = order[i]
        var j = i - 1
        while j >= 0:
            var o = order[j]
            var gt = edges.y0[o] > edges.y0[cur] or (edges.y0[o] == edges.y0[cur] and (edges.x[o] > edges.x[cur] or (edges.x[o] == edges.x[cur] and edges.dx[o] > edges.dx[cur])))
            if not gt:
                break
            order[j + 1] = o
            j -= 1
        order[j + 1] = cur
    # sorted copies; slot `total` is the INT_MAX sentinel, slot `total + 1` the list head
    var ey0 = List[Int](length=total + 2, fill=0)
    var ey1 = List[Int](length=total + 2, fill=0)
    var ex = List[Int](length=total + 2, fill=0)
    var edx = List[Int](length=total + 2, fill=0)
    var nxt = List[Int](length=total + 2, fill=-1)
    for i in range(total):
        ey0[i] = edges.y0[order[i]]
        ey1[i] = edges.y1[order[i]]
        ex[i] = edges.x[order[i]]
        edx[i] = edges.dx[order[i]]
    ey0[total] = INT_MAX
    var TMP = total + 1
    var i = 0
    var e = 0
    y_max = min(y_max, m.rows)
    var y = ey0[e]
    var delta = XY_ONE - 1
    while y < y_max:
        var prelast = TMP
        var last = nxt[TMP]
        var draw = 0
        var clipline = y < 0
        while last != -1 or ey0[e] == y:
            if last != -1 and ey1[last] == y:
                nxt[prelast] = nxt[last]
                last = nxt[last]
                continue
            var keep_prelast = prelast
            if last != -1 and (ey0[e] > y or ex[last] < ex[e]):
                prelast = last
                last = nxt[last]
            elif i < total:
                nxt[prelast] = e
                nxt[e] = last
                prelast = e
                i += 1
                e = i
            else:
                break
            if draw == 1:
                if not clipline:
                    var x1: Int
                    var x2: Int
                    if ex[keep_prelast] > ex[prelast]:
                        x1 = (ex[prelast] + delta) >> XY_SHIFT
                        x2 = ex[keep_prelast] >> XY_SHIFT
                    else:
                        x1 = (ex[keep_prelast] + delta) >> XY_SHIFT
                        x2 = ex[prelast] >> XY_SHIFT
                    if x1 < m.cols and x2 >= 0:
                        _hline(m, y, max(x1, 0), min(x2, m.cols - 1), color)
                ex[keep_prelast] += edx[keep_prelast]
                ex[prelast] += edx[prelast]
            draw ^= 1
        # bubble sort the active list by x
        var keep = -1
        while True:
            var pl = TMP
            var la = nxt[TMP]
            var last_exchange = -1
            while la != keep and nxt[la] != -1:
                var te = nxt[la]
                if ex[la] > ex[te]:
                    nxt[pl] = te
                    nxt[la] = nxt[te]
                    nxt[te] = la
                    pl = te
                    last_exchange = pl
                else:
                    pl = la
                    la = te
            if last_exchange == -1:
                break
            keep = last_exchange
            if keep == nxt[TMP] or keep == TMP:
                break
        y += 1


def circle(mut m: Img, cx: Int, cy: Int, radius: Int, color: List[Float64], thickness: Int) raises:
    """cv2.circle, LINE_8, shift 0."""
    if thickness > 1:
        ellipse_ex(m, cx << XY_SHIFT, cy << XY_SHIFT, radius << XY_SHIFT, radius << XY_SHIFT, 0, 0, 360, color, thickness)
    elif thickness < 0:
        _circle_fill(m, cx, cy, radius, color)
    else:
        var err = 0
        var dx = radius
        var dy = 0
        var plus = 1
        var minus = (radius << 1) - 1
        while dx >= dy:
            _put(m, cx - dx, cy - dy, color)
            _put(m, cx - dx, cy + dy, color)
            _put(m, cx + dx, cy - dy, color)
            _put(m, cx + dx, cy + dy, color)
            _put(m, cx - dy, cy - dx, color)
            _put(m, cx - dy, cy + dx, color)
            _put(m, cx + dy, cy - dx, color)
            _put(m, cx + dy, cy + dx, color)
            dy += 1
            err += plus
            plus += 2
            var mask = 0 if err <= 0 else -1
            err -= minus & mask
            dx += mask
            minus -= mask & 2


def ellipse_ex(mut m: Img, cx: Int, cy: Int, aw_in: Int, ah_in: Int, angle: Int, arc_start: Int, arc_end: Int, color: List[Float64], thickness: Int) raises:
    """cv2 EllipseEx; center and axes in 16.16."""
    var aw = abs(aw_in)
    var ah = abs(ah_in)
    var delta = (max(aw, ah) + (XY_ONE >> 1)) >> XY_SHIFT
    delta = 90 if delta < 3 else (30 if delta < 10 else (18 if delta < 15 else 5))
    var px = List[Float64]()
    var py = List[Float64]()
    ellipse2poly(Float64(cx), Float64(cy), Float64(aw), Float64(ah), angle, arc_start, arc_end, delta, px, py)
    var vx = List[Int]()
    var vy = List[Int]()
    var prevx = -1
    var prevy = -1
    for i in range(len(px)):
        var x = rint(px[i] / Float64(XY_ONE)) << XY_SHIFT
        var y = rint(py[i] / Float64(XY_ONE)) << XY_SHIFT
        x += rint(px[i] - Float64(x))
        y += rint(py[i] - Float64(y))
        if x != prevx or y != prevy:
            vx.append(x)
            vy.append(y)
            prevx = x
            prevy = y
    if len(vx) <= 1:
        vx = [cx, cx]
        vy = [cy, cy]
    if thickness >= 0:
        poly_line(m, vx, vy, False, color, thickness, XY_SHIFT)
    elif arc_end - arc_start >= 360:
        _fill_convex(m, vx, vy, color)
    else:
        vx.append(cx)
        vy.append(cy)
        var ed = _Edges()
        collect_poly_edges(m, vx, vy, ed, color, XY_SHIFT)
        fill_edge_collection(m, ed, color)


def ellipse(mut m: Img, cx: Int, cy: Int, aw: Int, ah: Int, angle: Int, start: Int, end: Int, color: List[Float64], thickness: Int) raises:
    """cv2.ellipse(center, axes, angle, startAngle, endAngle), shift 0."""
    ellipse_ex(m, cx << XY_SHIFT, cy << XY_SHIFT, aw << XY_SHIFT, ah << XY_SHIFT, angle, start, end, color, thickness)


def polylines(mut m: Img, xs: List[Int], ys: List[Int], closed: Bool, color: List[Float64], thickness: Int):
    poly_line(m, xs, ys, closed, color, thickness, 0)


def fill_poly(mut m: Img, contours_x: List[List[Int]], contours_y: List[List[Int]], color: List[Float64]):
    """cv2.fillPoly, shift 0, no offset; several contours share one even-odd edge set."""
    var ed = _Edges()
    for c in range(len(contours_x)):
        if len(contours_x[c]) > 0:
            collect_poly_edges(m, contours_x[c], contours_y[c], ed, color, 0)
    fill_edge_collection(m, ed, color)

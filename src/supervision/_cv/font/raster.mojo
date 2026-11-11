# Glyph rasterizer: stbtt_GetGlyphBitmapSubpixelRealloc of OpenCV 5.0.0's stb_truetype fork (rasterizer v2, exact-area
# coverage), float32 step for step. The order of float additions decides the last bit of a coverage value, so the edge sort,
# the active-edge list order (new edges go to the head) and every cast follow the C source.
from std.math import ceil, floor, sqrt
from supervision._cv.font.ttf import Font, Vertex, VMOVE, VLINE, fmul


struct Bitmap(Movable):
    var w: Int
    var h: Int
    var xoff: Int
    var yoff: Int
    var advx: Float32
    var pix: List[UInt8]  # w * h, row-major, tightly cropped to the non-zero box

    def __init__(out self):
        self.w = 0
        self.h = 0
        self.xoff = 0
        self.yoff = 0
        self.advx = 0
        self.pix = List[UInt8]()


@fieldwise_init
struct _Edge(Copyable, Movable):
    var x0: Float32
    var y0: Float32
    var x1: Float32
    var y1: Float32
    var invert: Bool


def _tess_curve(mut xs: List[Float32], mut ys: List[Float32], x0: Float32, y0: Float32, x1: Float32, y1: Float32, x2: Float32, y2: Float32, flat2: Float32, n: Int):
    var mx = (x0 + 2 * x1 + x2) / 4
    var my = (y0 + 2 * y1 + y2) / 4
    var dx = (x0 + x2) / 2 - mx
    var dy = (y0 + y2) / 2 - my
    if n > 16:
        return
    if fmul(dx, dx) + fmul(dy, dy) > flat2:
        _tess_curve(xs, ys, x0, y0, (x0 + x1) / 2.0, (y0 + y1) / 2.0, mx, my, flat2, n + 1)
        _tess_curve(xs, ys, mx, my, (x1 + x2) / 2.0, (y1 + y2) / 2.0, x2, y2, flat2, n + 1)
    else:
        xs.append(x2)
        ys.append(y2)


def _flatten(v: List[Vertex], flatness: Float32, mut xs: List[Float32], mut ys: List[Float32], mut lens: List[Int]):
    """stbtt_FlattenCurves: one polyline per contour (quadratic curves only; TrueType has no cubics)."""
    var flat2 = flatness * flatness
    var x = Float32(0)
    var y = Float32(0)
    var start = 0
    for i in range(len(v)):
        if v[i].kind == VMOVE:
            if len(lens) > 0 or i > 0:
                lens.append(len(xs) - start)
            start = len(xs)
            x = Float32(v[i].x)
            y = Float32(v[i].y)
            xs.append(x)
            ys.append(y)
        elif v[i].kind == VLINE:
            x = Float32(v[i].x)
            y = Float32(v[i].y)
            xs.append(x)
            ys.append(y)
        else:
            _tess_curve(xs, ys, x, y, Float32(v[i].cx), Float32(v[i].cy), Float32(v[i].x), Float32(v[i].y), flat2, 0)
            x = Float32(v[i].x)
            y = Float32(v[i].y)
    if len(xs) > 0:
        lens.append(len(xs) - start)


def _sort_edges(mut p: List[_Edge], n: Int):
    _quicksort(p, 0, n)
    for i in range(1, n):  # insertion sort finishes what quicksort left within runs of 12
        var t = p[i].copy()
        var j = i
        while j > 0:
            if not (t.y0 < p[j - 1].y0):
                break
            p[j] = p[j - 1].copy()
            j -= 1
        if i != j:
            p[j] = t^


def _swap(mut p: List[_Edge], a: Int, b: Int):
    var t = p[a].copy()
    p[a] = p[b].copy()
    p[b] = t^


def _quicksort(mut p: List[_Edge], base_in: Int, n_in: Int):
    var base = base_in
    var n = n_in
    while n > 12:
        var m = n >> 1
        var c01 = p[base].y0 < p[base + m].y0
        var c12 = p[base + m].y0 < p[base + n - 1].y0
        if c01 != c12:
            var c = p[base].y0 < p[base + n - 1].y0
            _swap(p, base + (0 if c == c12 else n - 1), base + m)
        _swap(p, base, base + m)
        var i = 1
        var j = n - 1
        while True:
            while p[base + i].y0 < p[base].y0:
                i += 1
            while p[base].y0 < p[base + j].y0:
                j -= 1
            if i >= j:
                break
            _swap(p, base + i, base + j)
            i += 1
            j -= 1
        if j < n - i:
            _quicksort(p, base, j)
            base += i
            n -= i
        else:
            _quicksort(p, base + i, n - i)
            n = j


struct _Active:
    """Active edges as parallel lists; `nxt` links them, new edges are pushed on the head (the C order)."""
    var fx: List[Float32]
    var fdx: List[Float32]
    var fdy: List[Float32]
    var dir: List[Float32]
    var sy: List[Float32]
    var ey: List[Float32]
    var nxt: List[Int]

    def __init__(out self):
        self.fx = List[Float32]()
        self.fdx = List[Float32]()
        self.fdy = List[Float32]()
        self.dir = List[Float32]()
        self.sy = List[Float32]()
        self.ey = List[Float32]()
        self.nxt = List[Int]()


def _clipped(mut sc: List[Float32], x: Int, dir: Float32, esy: Float32, eey: Float32, x0_in: Float32, y0_in: Float32, x1_in: Float32, y1_in: Float32):
    var x0 = x0_in
    var y0 = y0_in
    var x1 = x1_in
    var y1 = y1_in
    if y0 == y1:
        return
    if y0 > eey:
        return
    if y1 < esy:
        return
    if y0 < esy:
        x0 += (x1 - x0) * (esy - y0) / (y1 - y0)
        y0 = esy
    if y1 > eey:
        x1 += (x1 - x0) * (eey - y1) / (y1 - y0)
        y1 = eey
    var fx = Float32(x)
    if x0 <= fx and x1 <= fx:
        sc[x] += fmul(dir, y1 - y0)
    elif x0 >= fx + 1 and x1 >= fx + 1:
        pass
    else:
        sc[x] += fmul(fmul(dir, y1 - y0), 1 - ((x0 - fx) + (x1 - fx)) / 2)


def _fill_edges(mut sc: List[Float32], mut fill: List[Float32], w: Int, a: _Active, head: Int, y_top: Float32):
    """stbtt__fill_active_edges_new. `fill` is scanline2: scanline_fill[x] is fill[x + 1], scanline_fill - 1 is fill itself."""
    var y_bottom = y_top + 1
    var e = head
    var wf = Float32(w)
    while e >= 0:
        var dir = a.dir[e]
        var esy = a.sy[e]
        var eey = a.ey[e]
        if a.fdx[e] == 0:
            var x0 = a.fx[e]
            if x0 < wf:
                if x0 >= 0:
                    _clipped(sc, Int(x0), dir, esy, eey, x0, y_top, x0, y_bottom)
                    _clipped(fill, Int(x0) + 1, dir, esy, eey, x0, y_top, x0, y_bottom)
                else:
                    _clipped(fill, 0, dir, esy, eey, x0, y_top, x0, y_bottom)
        else:
            var x0 = a.fx[e]
            var dx = a.fdx[e]
            var xb = x0 + dx
            var x_top: Float32
            var x_bottom: Float32
            var sy0: Float32
            var sy1: Float32
            var dy = a.fdy[e]
            if esy > y_top:
                x_top = x0 + fmul(dx, esy - y_top)
                sy0 = esy
            else:
                x_top = x0
                sy0 = y_top
            if eey < y_bottom:
                x_bottom = x0 + fmul(dx, eey - y_top)
                sy1 = eey
            else:
                x_bottom = xb
                sy1 = y_bottom
            if x_top >= 0 and x_bottom >= 0 and x_top < wf and x_bottom < wf:
                if Int(x_top) == Int(x_bottom):
                    var x = Int(x_top)
                    var height = fmul(sy1 - sy0, dir)
                    var fx1 = Float32(x) + 1.0
                    sc[x] += fmul(((fx1 - x_top) + (fx1 - x_bottom)) / 2.0, height)  # position_trapezoid_area(height, x_top, x+1, x_bottom, x+1)
                    fill[x + 1] += height
                else:
                    if x_top > x_bottom:
                        sy0 = y_bottom - (sy0 - y_top)
                        sy1 = y_bottom - (sy1 - y_top)
                        var t = sy0
                        sy0 = sy1
                        sy1 = t
                        t = x_bottom
                        x_bottom = x_top
                        x_top = t
                        dy = -dy
                        x0 = xb
                    var x1 = Int(x_top)
                    var x2 = Int(x_bottom)
                    var y_crossing = y_top + fmul(dy, Float32(x1 + 1) - x0)
                    var y_final = y_top + fmul(dy, Float32(x2) - x0)
                    if y_crossing > y_bottom:
                        y_crossing = y_bottom
                    var sign = dir
                    var area = fmul(sign, y_crossing - sy0)
                    sc[x1] += fmul(area, Float32(x1 + 1) - x_top) / 2
                    if y_final > y_bottom:
                        y_final = y_bottom
                        dy = (y_final - y_crossing) / Float32(x2 - (x1 + 1))
                    var step = fmul(sign, dy)
                    for x in range(x1 + 1, x2):
                        sc[x] += area + step / 2
                        area += step
                    var fx2 = Float32(x2)
                    sc[x2] += area + fmul(sign, fmul(((fx2 + 1.0 - fx2) + (fx2 + 1.0 - x_bottom)) / 2.0, sy1 - y_final))
                    fill[x2 + 1] += fmul(sign, sy1 - sy0)
            else:
                for x in range(w):
                    var yy0 = y_top
                    var x1 = Float32(x)
                    var x2 = Float32(x + 1)
                    var x3 = xb
                    var y3 = y_bottom
                    var y1 = (x1 - x0) / dx + y_top
                    var y2 = (x2 - x0) / dx + y_top
                    if x0 < x1 and x3 > x2:
                        _clipped(sc, x, dir, esy, eey, x0, yy0, x1, y1)
                        _clipped(sc, x, dir, esy, eey, x1, y1, x2, y2)
                        _clipped(sc, x, dir, esy, eey, x2, y2, x3, y3)
                    elif x3 < x1 and x0 > x2:
                        _clipped(sc, x, dir, esy, eey, x0, yy0, x2, y2)
                        _clipped(sc, x, dir, esy, eey, x2, y2, x1, y1)
                        _clipped(sc, x, dir, esy, eey, x1, y1, x3, y3)
                    elif x0 < x1 and x3 > x1:
                        _clipped(sc, x, dir, esy, eey, x0, yy0, x1, y1)
                        _clipped(sc, x, dir, esy, eey, x1, y1, x3, y3)
                    elif x3 < x1 and x0 > x1:
                        _clipped(sc, x, dir, esy, eey, x0, yy0, x1, y1)
                        _clipped(sc, x, dir, esy, eey, x1, y1, x3, y3)
                    elif x0 < x2 and x3 > x2:
                        _clipped(sc, x, dir, esy, eey, x0, yy0, x2, y2)
                        _clipped(sc, x, dir, esy, eey, x2, y2, x3, y3)
                    elif x3 < x2 and x0 > x2:
                        _clipped(sc, x, dir, esy, eey, x0, yy0, x2, y2)
                        _clipped(sc, x, dir, esy, eey, x2, y2, x3, y3)
                    else:
                        _clipped(sc, x, dir, esy, eey, x0, yy0, x3, y3)
        e = a.nxt[e]


def _rasterize(mut pix: List[UInt8], w: Int, h: Int, v: List[Vertex], flatness: Float32, scale: Float32, shift_x: Float32, shift_y: Float32, off_x: Int, off_y: Int):
    """stbtt_Rasterize with invert = 1 into a w x h A8 bitmap (stride w)."""
    var xs = List[Float32]()
    var ys = List[Float32]()
    var lens = List[Int]()
    _flatten(v, flatness / scale, xs, ys, lens)
    var edges = List[_Edge]()
    var m = 0
    for c in range(len(lens)):
        var j = lens[c] - 1
        for k in range(lens[c]):
            var a = m + k
            var b = m + j
            if ys[m + j] != ys[m + k]:
                var inv = ys[m + j] > ys[m + k]
                if inv:
                    a = m + j
                    b = m + k
                edges.append(_Edge(fmul(xs[a], scale) + shift_x, fmul(ys[a], -scale) + shift_y, fmul(xs[b], scale) + shift_x, fmul(ys[b], -scale) + shift_y, inv))
            j = k
        m += lens[c]
    var n = len(edges)
    _sort_edges(edges, n)
    var sc = List[Float32](length=w + 2, fill=0)
    var fill = List[Float32](length=w + 2, fill=0)
    var act = _Active()
    var head = -1
    var ei = 0
    var y = off_y
    for j in range(h):
        var top = Float32(y) + 0.0
        var bottom = Float32(y) + 1.0
        for i in range(w + 2):
            sc[i] = 0
            fill[i] = 0
        # drop finished edges, keeping list order
        var prev = -1
        var cur = head
        while cur >= 0:
            if act.ey[cur] <= top:
                if prev < 0:
                    head = act.nxt[cur]
                else:
                    act.nxt[prev] = act.nxt[cur]
            else:
                prev = cur
            cur = act.nxt[cur]
        while ei < n and edges[ei].y0 <= bottom:
            ref e = edges[ei]
            var dxdy = (e.x1 - e.x0) / (e.y1 - e.y0)
            act.fdx.append(dxdy)
            act.fdy.append(Float32(1.0) / dxdy if dxdy != 0.0 else Float32(0.0))
            act.fx.append(e.x0 + fmul(dxdy, top - e.y0) - Float32(off_x))
            act.dir.append(Float32(1.0) if e.invert else Float32(-1.0))
            act.sy.append(e.y0)
            var ey = e.y1
            if j == 0 and off_y != 0 and ey < top:
                ey = top
            act.ey.append(ey)
            act.nxt.append(head)
            head = len(act.fx) - 1
            ei += 1
        if head >= 0:
            _fill_edges(sc, fill, w, act, head, top)
        var total = Float32(0)
        for i in range(w):
            total += fill[i]
            var k = sc[i] + total
            if k < 0:
                k = -k
            k = fmul(k, 255) + 0.5
            var mm = Int(k)
            pix[j * w + i] = UInt8(255 if mm > 255 else mm)
        var cur2 = head
        while cur2 >= 0:
            act.fx[cur2] += act.fdx[cur2]
            cur2 = act.nxt[cur2]
        y += 1


def glyph_bitmap(font: Font, scale: Float32, gid: Int) -> Bitmap:
    """stbtt_GetGlyphBitmapSubpixelRealloc(scale, scale, shift 0): coverage bitmap, offset from the pen, and float advance."""
    var out = Bitmap()
    var s = font.glyph_shape(gid)
    var six0 = Int(floor(Float64(Float32(s.x0) * scale + 0)))
    var siy0 = Int(floor(Float64(Float32(-s.y1) * scale + 0)))
    var six1 = Int(ceil(Float64(Float32(s.x1) * scale + 0)))
    var siy1 = Int(ceil(Float64(Float32(-s.y0) * scale + 0)))
    var w0 = six1 - six0
    var h0 = siy1 - siy0
    var pad = max((h0 + 9) // 10, (w0 + 9) // 10) + 10
    var gw = w0 + pad * 2
    var gh = h0 + pad * 2
    var bx0 = 0
    var bx1 = 0
    var b = font.glyph_box(gid)
    if b:
        bx0 = b.value()[0]
        bx1 = b.value()[2]
    var hm = font.hmetrics(gid)
    var interspace = hm[0] - (bx1 - bx0)
    out.advx = Float32(interspace + (s.x1 - s.x0)) * scale
    if gw <= 0 or gh <= 0:
        return out^
    var pix = List[UInt8](length=gw * gh, fill=0)
    _rasterize(pix, gw, gh, s.v, 0.35, scale, 0.0 + Float32(pad), 0.0 + Float32(pad), six0, siy0)
    var ix0 = 1 << 30
    var iy0 = 1 << 30
    var ix1 = -(1 << 30)
    var iy1 = -(1 << 30)
    for y in range(gh):
        var nz = False
        for x in range(gw):
            if pix[y * gw + x] != 0:
                ix0 = min(ix0, x)
                ix1 = max(ix1, x)
                nz = True
        if nz:
            iy0 = min(iy0, y)
            iy1 = max(iy1, y)
    if ix0 == (1 << 30):
        out.xoff = six0
        out.yoff = siy0
        return out^
    out.w = ix1 - ix0 + 1
    out.h = iy1 - iy0 + 1
    out.xoff = six0 - pad + ix0
    out.yoff = siy0 - pad + iy0
    out.pix = List[UInt8](length=out.w * out.h, fill=0)
    for y in range(out.h):
        for x in range(out.w):
            out.pix[y * out.w + x] = pix[(iy0 + y) * gw + ix0 + x]
    return out^

# cv2.putText / cv2.getTextSize for FONT_HERSHEY_SIMPLEX on OpenCV 5.0.0: the legacy wrappers of drawing_text.cpp plus its
# `FontRenderEngine::putText_` for the no-HarfBuzz build (one glyph per codepoint, no shaping or kerning), left-aligned,
# top-left origin, no wrapping. Size is round(100 * scale / 3.7), weight 400 for thickness <= 1 else 600, pen advances in
# 26.6 fixed point, then an integer alpha blend. Glyphs missing from "sans" draw as '?'; cv2 would first try its italic face,
# which this port does not carry (no codepoint outside Latin that supervision labels use is affected).
from mojocv.imgproc import rint
from mojocv.mat import Mat
from supervision._cv.font.raster import Bitmap, glyph_bitmap
from supervision._cv.font.sans import load_sans
from supervision._cv.font.ttf import Font

comptime Img = Mat[DType.uint8]
comptime WGHT = 0x77676874
comptime CACHE_MAX = 2048


struct TextEngine(Movable):
    """Holds the decoded face and a glyph cache keyed by (glyph, size, weight), like cv2's FontRenderEngine."""
    var font: Font
    var index: Dict[Int, Int]
    var glyphs: List[Bitmap]
    var weight: Int

    def __init__(out self) raises:
        self.font = load_sans()
        self.index = Dict[Int, Int]()
        self.glyphs = List[Bitmap]()
        self.weight = -1

    def _glyph(mut self, gid: Int, size: Int, weight: Int) -> Int:
        var key = (((gid << 16) | size) << 12) | weight
        var hit = self.index.get(key)
        if hit:
            return hit.value()
        if len(self.glyphs) >= CACHE_MAX:
            self.index.clear()
            self.glyphs.clear()
        if weight != self.weight:
            self.font.set_instance(WGHT, weight * 65536)
            self.weight = weight
        self.glyphs.append(glyph_bitmap(self.font, self.font.scale_for_pixel_height_nodesc(Float32(size)), gid))
        self.index[key] = len(self.glyphs) - 1
        return len(self.glyphs) - 1

    def _run(mut self, mut img: Img, render: Bool, text: String, org_x: Int, org_y: Int, scale: Float64, thickness: Int, color: List[Float64]) -> List[Int]:
        """putText_ in one function. Returns [bbox_x, bbox_y, bbox_w, bbox_h] (cv2's Rect) of the text."""
        var size = rint(100 * scale / 3.7)
        var weight = 400 if thickness <= 1 else 600
        var bytes = text.as_bytes()
        var n = len(bytes)
        if n == 0:
            return [org_x, org_y, 0, 0]
        var cps = List[Int]()  # UTF-8 to UTF-32, with cv2's lenient decoding
        var i = 0
        while i < n:
            var ch = Int(bytes[i])
            if ch <= 127:
                cps.append(ch)
            elif ch <= 223 and i + 1 < n and (Int(bytes[i + 1]) & 0xC0) == 0x80:
                cps.append(((ch & 31) << 6) | (Int(bytes[i + 1]) & 63))
                i += 1
            elif ch <= 239 and i + 2 < n and (Int(bytes[i + 1]) & 0xC0) == 0x80 and (Int(bytes[i + 2]) & 0xC0) == 0x80:
                cps.append(((ch & 15) << 12) | ((Int(bytes[i + 1]) & 63) << 6) | (Int(bytes[i + 2]) & 63))
                i += 2
            elif ch <= 247 and i + 3 < n and (Int(bytes[i + 1]) & 0xC0) == 0x80 and (Int(bytes[i + 2]) & 0xC0) == 0x80 and (Int(bytes[i + 3]) & 0xC0) == 0x80:
                var v = ((ch & 15) << 18) | ((Int(bytes[i + 1]) & 63) << 12) | ((Int(bytes[i + 2]) & 63) << 6) | (Int(bytes[i + 3]) & 63)
                cps.append(65533 if v > 1114111 else v)
                i += 3
            else:
                cps.append(65533)
                while i + 1 < n and (Int(bytes[i + 1]) & 0xC0) == 0x80:
                    i += 1
            i += 1
        var scale_px = self.font.scale_for_pixel_height_nodesc(Float32(size))
        var asc = self.font.vmetrics()
        var linegap = asc[2] if asc[2] != 0 else asc[0] - asc[1]
        var q_gid = self.font.glyph_index(63)
        var pen_x = org_x
        var pen_y = org_y
        var max_dy = 0
        var max_baseline = 0
        var min_x = 1 << 31
        var max_x = -(1 << 31)
        var prev_dy = 0
        var pos = 0
        var nc = len(cps)
        while pos < nc:
            var nextline_dy = 0
            var start = pos
            min_x = min(min_x, pen_x)
            max_x = max(max_x, pen_x)
            while pos < nc and cps[pos] != 10:
                var gid = self.font.glyph_index(cps[pos])
                if gid == 0:
                    gid = q_gid
                pos += 1
                if scale_px == 0:
                    continue  # stbtt returns no bitmap for scale 0 and cv2 skips the glyph
                var gi = self._glyph(gid, size, weight)
                ref g = self.glyphs[gi]
                var dx = Int(rint(Float64(g.advx * 64))) >> 6
                nextline_dy = max(nextline_dy, Int(rint(Float64(Float32(linegap) * scale_px))))
                max_dy = max(max_dy, Int(rint(Float64(Float32(asc[0]) * scale_px))))
                max_baseline = max(max_baseline, g.h + g.yoff)
                if render:
                    _blend(img, g, pen_x + g.xoff, pen_y + g.yoff, color)
                pen_x += dx
                min_x = min(min_x, pen_x)
                max_x = max(max_x, pen_x)
            if pos == start:
                nextline_dy = prev_dy
            if pos < nc and cps[pos] == 10:
                pos += 1
            if pos < nc:
                pen_x = org_x
                pen_y += nextline_dy
                prev_dy = nextline_dy
        return [min_x, org_y - max_dy, max_x - min_x + 1, pen_y - org_y + max_dy + max_baseline]

    def put_text(mut self, mut img: Img, text: String, x: Int, y: Int, scale: Float64, color: List[Float64], thickness: Int):
        """cv2.putText(img, text, (x, y), FONT_HERSHEY_SIMPLEX, scale, color, thickness, LINE_AA) for 1, 3 or 4 channel uint8."""
        _ = self._run(img, True, text, x, y, scale, thickness, color)

    def text_size(mut self, text: String, scale: Float64, thickness: Int) -> List[Int]:
        """cv2.getTextSize: [width, height, baseline]."""
        var none = Img(0, 0, 1)
        var r = self._run(none, False, text, 0, 0, scale, thickness, List[Float64]())
        var baseline = r[1] + r[3]
        return [r[2], r[3] - baseline, baseline]


def _blend(mut img: Img, g: Bitmap, x0: Int, y0: Int, color: List[Float64]):
    """drawCharacter: per pixel (dst * (255 - a) + c * a + 127) / 255, alpha byte into channel 3 of a 4-channel image."""
    var nch = img.ch
    var c0 = _sat(color, 0)
    var c1 = _sat(color, 1)
    var c2 = _sat(color, 2)
    if x0 >= img.cols or x0 + g.w <= 0:
        return
    for dy in range(g.h):
        var y = y0 + dy
        if y < 0 or y >= img.rows:
            continue
        for dx in range(g.w):
            var x = x0 + dx
            if x < 0 or x >= img.cols:
                continue
            var a = Int(g.pix[dy * g.w + dx])
            if a == 0:
                continue
            var o = (y * img.cols + x) * nch
            img.data[o] = UInt8((Int(img.data[o]) * (255 - a) + c0 * a + 127) // 255)
            if nch >= 3:
                img.data[o + 1] = UInt8((Int(img.data[o + 1]) * (255 - a) + c1 * a + 127) // 255)
                img.data[o + 2] = UInt8((Int(img.data[o + 2]) * (255 - a) + c2 * a + 127) // 255)
            if nch == 4:
                img.data[o + 3] = UInt8(a)


def _sat(color: List[Float64], i: Int) -> Int:
    """saturate_cast<uchar>(Scalar[i]); missing entries are 0."""
    if i >= len(color):
        return 0
    var v = rint(color[i])
    return 0 if v < 0 else (255 if v > 255 else v)

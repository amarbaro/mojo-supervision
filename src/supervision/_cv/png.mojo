# PNG read/write for 8-bit gray / RGB / RGBA / palette images, non-interlaced; inflate and deflate through mojo-gzip (zlib).
# Images are BGR like cv2.imread: read converts RGB(A) to BGR(A) and keeps gray as 1 channel. No JPEG, no 16-bit, no Adam7.
from gzip import compress, decompress, crc32
from mojocv.mat import Mat

def _sig() -> List[Int]:
    return [137, 80, 78, 71, 13, 10, 26, 10]


def _be32(b: List[UInt8], at: Int) -> Int:
    return Int(b[at]) << 24 | Int(b[at + 1]) << 16 | Int(b[at + 2]) << 8 | Int(b[at + 3])


def _put32(mut out: List[UInt8], v: Int):
    out.append(UInt8((v >> 24) & 255))
    out.append(UInt8((v >> 16) & 255))
    out.append(UInt8((v >> 8) & 255))
    out.append(UInt8(v & 255))


def _paeth(a: Int, b: Int, c: Int) -> Int:
    var p = a + b - c
    var pa = abs(p - a)
    var pb = abs(p - b)
    var pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def decode_png(b: List[UInt8]) raises -> Mat[DType.uint8]:
    if len(b) < 33:
        raise Error("png: too short")
    var sig = _sig()
    for i in range(8):
        if Int(b[i]) != sig[i]:
            raise Error("png: bad signature")
    var pos = 8
    var w = 0
    var h = 0
    var ct = -1
    var idat = List[UInt8]()
    var plte = List[UInt8]()
    while pos + 8 <= len(b):
        var n = _be32(b, pos)
        var tag = String(unsafe_from_utf8=Span(b)[pos + 4 : pos + 8])
        var d = pos + 8
        if d + n + 4 > len(b):
            raise Error("png: truncated chunk " + tag)
        if tag == "IHDR":
            w = _be32(b, d)
            h = _be32(b, d + 4)
            if Int(b[d + 8]) != 8 or Int(b[d + 12]) != 0:
                raise Error("png: only 8-bit non-interlaced images are supported")
            ct = Int(b[d + 9])
        elif tag == "PLTE":
            for i in range(n):
                plte.append(b[d + i])
        elif tag == "IDAT":
            for i in range(n):
                idat.append(b[d + i])
        elif tag == "IEND":
            break
        pos = d + n + 4
    if ct != 0 and ct != 2 and ct != 3 and ct != 6:
        raise Error("png: unsupported color type " + String(ct))
    var spp = 1 if (ct == 0 or ct == 3) else (3 if ct == 2 else 4)
    var raw = decompress(Span(idat), "zlib")
    var stride = w * spp
    if len(raw) != h * (stride + 1):
        raise Error("png: bad decompressed size")
    var px = List[UInt8](length=h * stride, fill=0)
    for y in range(h):
        var f = Int(raw[y * (stride + 1)])
        for x in range(stride):
            var cur = Int(raw[y * (stride + 1) + 1 + x])
            var a = Int(px[y * stride + x - spp]) if x >= spp else 0
            var up = Int(px[(y - 1) * stride + x]) if y > 0 else 0
            var c = Int(px[(y - 1) * stride + x - spp]) if (y > 0 and x >= spp) else 0
            var v: Int
            if f == 0:
                v = cur
            elif f == 1:
                v = cur + a
            elif f == 2:
                v = cur + up
            elif f == 3:
                v = cur + ((a + up) >> 1)
            elif f == 4:
                v = cur + _paeth(a, up, c)
            else:
                raise Error("png: bad filter " + String(f))
            px[y * stride + x] = UInt8(v & 255)
    var ch = 1 if ct == 0 else (3 if ct == 3 or ct == 2 else 4)
    var m = Mat[DType.uint8](h, w, ch)
    for y in range(h):
        for x in range(w):
            if ct == 0:
                m.set(y, x, 0, px[y * stride + x])
            elif ct == 3:
                var i = Int(px[y * stride + x])
                for k in range(3):
                    m.set(y, x, k, plte[i * 3 + 2 - k])
            else:
                var o = y * stride + x * spp
                for k in range(3):
                    m.set(y, x, k, px[o + 2 - k])
                if ct == 6:
                    m.set(y, x, 3, px[o + 3])
    return m^


def encode_png(m: Mat[DType.uint8], level: Int = 6) raises -> List[UInt8]:
    """Filter 0 rows, zlib level `level`; 1 channel -> gray, 3 -> RGB, 4 -> RGBA (from BGR / BGRA)."""
    if m.ch != 1 and m.ch != 3 and m.ch != 4:
        raise Error("png: channels must be 1, 3 or 4")
    var ct = 0 if m.ch == 1 else (2 if m.ch == 3 else 6)
    var raw = List[UInt8]()
    for y in range(m.rows):
        raw.append(0)
        for x in range(m.cols):
            if m.ch == 1:
                raw.append(m.at(y, x, 0))
            else:
                raw.append(m.at(y, x, 2))
                raw.append(m.at(y, x, 1))
                raw.append(m.at(y, x, 0))
                if m.ch == 4:
                    raw.append(m.at(y, x, 3))
    var out = List[UInt8]()
    var sig = _sig()
    for i in range(8):
        out.append(UInt8(sig[i]))
    var hdr = List[UInt8]()
    _put32(hdr, m.cols)
    _put32(hdr, m.rows)
    hdr.append(8)
    hdr.append(UInt8(ct))
    hdr.append(0)
    hdr.append(0)
    hdr.append(0)
    _chunk(out, "IHDR", hdr)
    _chunk(out, "IDAT", compress(Span(raw), level, "zlib"))
    _chunk(out, "IEND", List[UInt8]())
    return out^


def _chunk(mut out: List[UInt8], tag: String, data: List[UInt8]):
    _put32(out, len(data))
    var body = List[UInt8]()
    for c in tag.as_bytes():
        body.append(c)
    for c in data:
        body.append(c)
    for c in body:
        out.append(c)
    _put32(out, Int(crc32(Span(body))))


def imread(path: String) raises -> Mat[DType.uint8]:
    return decode_png(open(path, "r").read_bytes())


def imwrite(path: String, m: Mat[DType.uint8], level: Int = 6) raises:
    var b = encode_png(m, level)
    with open(path, "w") as f:
        f.write_bytes(Span(b))

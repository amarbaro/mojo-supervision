# Row driver for tests/cases/font_ttf.txt: one stdin row in, one JSON line out. Ops: cmap, hmetrics, vmetrics, shape, glyph.
from json import Json, parse
from mojocv.imgproc import rint
from std.ffi import external_call
from supervision._cv.font.sans import load_sans
from supervision._cv.font.raster import glyph_bitmap
from supervision._cv.font.ttf import Font, VMOVE


def read_rows() raises -> List[String]:
    var buf = List[UInt8]()
    var chunk = List[UInt8](length=65536, fill=0)
    while True:
        var n = external_call["read", Int](Int32(0), chunk.unsafe_ptr(), 65536)  # fd type must match the stdlib's own read()
        if n <= 0:
            break
        buf.extend(Span(chunk)[:n])
    var rows = List[String]()
    for line in String(unsafe_from_utf8=buf).split("\n"):  # not splitlines(): it also splits at tabs
        if line.byte_length() > 0:
            rows.append(String(line))
    return rows^


def _sorted(var v: List[Int]) -> Json:
    sort(v)
    var a = Json.array()
    for x in v:
        a.push(Json.of(x))
    return a^


def run(mut font: Font, op: String, args: Json) raises -> Json:
    var out = Json.array()
    if op == "cmap":
        for i in range(args.get("cps").len()):
            out.push(Json.of(font.glyph_index(args.get("cps").at(i).int())))
    elif op == "hmetrics":
        for i in range(args.get("gids").len()):
            var m = font.hmetrics(args.get("gids").at(i).int())
            var e = Json.array()
            e.push(Json.of(m[0]))
            e.push(Json.of(m[1]))
            out.push(e)
    elif op == "vmetrics":
        var m = font.vmetrics()
        out.push(Json.of(m[0]))
        out.push(Json.of(m[1]))
        out.push(Json.of(m[2]))
    elif op == "shape":
        var wght = args.get("wght")
        font.set_instance(-1 if wght.is_null() else 0x77676874, 0 if wght.is_null() else wght.int() * 65536)
        var s = font.glyph_shape(args.get("gid").int())
        var ex = List[Int]()
        var ey = List[Int]()
        var cx = List[Int]()
        var cy = List[Int]()
        for v in s.v:
            if v.kind == VMOVE:
                continue  # the closing segment ends at the move point
            ex.append(v.x)
            ey.append(v.y)
            if v.kind == 3:
                cx.append(v.cx)
                cy.append(v.cy)
        var o = Json.object()
        o.set("ex", _sorted(ex^))
        o.set("ey", _sorted(ey^))
        o.set("cx", _sorted(cx^))
        o.set("cy", _sorted(cy^))
        if wght.is_null():
            var box = Json.array()
            box.push(Json.of(s.x0))
            box.push(Json.of(s.y0))
            box.push(Json.of(s.x1))
            box.push(Json.of(s.y1))
            o.set("box", box)
        return o^
    elif op == "glyph":
        font.set_instance(0x77676874, args.get("weight").int() * 65536)
        var sc = font.scale_for_pixel_height_nodesc(Float32(args.get("size").int()))
        var b = glyph_bitmap(font, sc, font.glyph_index(args.get("cp").int()))
        var o = Json.object()
        o.set("xoff", Json.of(b.xoff))
        o.set("yoff", Json.of(b.yoff))
        o.set("w", Json.of(b.w))
        o.set("h", Json.of(b.h))
        var px = Json.array()
        for v in b.pix:
            px.push(Json.of(Int(v)))
        o.set("pix", px)
        o.set("adv", Json.of(Int(rint(Float64(b.advx * 64))) >> 6))
        return o^
    else:
        raise Error("unknown op " + op)
    return out^


def main() raises:
    var font = load_sans()
    for row in read_rows():
        var cols = row.split("\t")
        var res: Json
        try:
            res = run(font, String(cols[0]), parse(String(cols[1])))
        except e:
            res = Json.object()
            res.set("error", Json.of(String(e)))
        print(res.dumps())

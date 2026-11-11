# S5 color unit test: expected values are the docstring examples of supervision/draw/color.py.
from supervision.draw.color import Color, ColorPalette


def check(ok: Bool, what: String) raises:
    if not ok:
        raise Error("FAIL " + what)


def raises_on_hex(s: String) -> Bool:
    try:
        _ = Color.from_hex(s)
    except:
        return True
    return False


def main() raises:
    var n = 0
    check(String(Color.from_hex("#ff00ff")) == "Color(r=255, g=0, b=255)", "hex6"); n += 1
    check(String(Color.from_hex("#f0f")) == "Color(r=255, g=0, b=255)", "hex3"); n += 1
    check(String(Color.from_hex("#ff00ff80")) == "Color(r=255, g=0, b=255, a=128)", "hex8"); n += 1
    check(String(Color.from_hex("#f0f8")) == "Color(r=255, g=0, b=255, a=136)", "hex4"); n += 1
    check(String(Color.from_hex("A351FB")) == "Color(r=163, g=81, b=251)", "no hash, uppercase"); n += 1
    check(Color(255, 255, 0).as_hex() == "#ffff00", "as_hex"); n += 1
    check(Color(255, 0, 255, 128).as_hex() == "#ff00ff80", "as_hex alpha"); n += 1
    check(Color(255, 255, 0).as_bgr()[0] == 0 and Color(255, 255, 0).as_bgr()[2] == 255, "as_bgr"); n += 1
    check(Color.from_bgr_tuple(0, 255, 255) == Color(255, 255, 0), "from_bgr"); n += 1
    check(raises_on_hex("#12") and raises_on_hex("#gg0000") and raises_on_hex("#12345"), "bad hex raises"); n += 1
    var bad = False
    try:
        _ = Color(256, 0, 0)
    except:
        bad = True
    check(bad, "range raises"); n += 1
    check(Color.ROBOFLOW() == Color(163, 81, 251), "ROBOFLOW"); n += 1
    var p = ColorPalette.DEFAULT()
    check(len(p) == 21 and p.by_idx(1) == Color(255, 64, 64) and p.by_idx(-1) == p.by_idx(20) and p.by_idx(21) == p.by_idx(0), "default palette wrap"); n += 1
    check(len(ColorPalette.LEGACY()) == 17 and ColorPalette.ROBOFLOW().by_idx(0) == Color(194, 141, 252), "other palettes"); n += 1
    print("PASS color", n)

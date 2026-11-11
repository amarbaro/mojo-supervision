# Port of supervision/draw/color.py: Color (RGBA), ColorPalette, the three built-in palettes, unify_to_bgr.
# Not ported: ColorPalette.from_matplotlib (needs matplotlib; no Mojo equivalent).

comptime DEFAULT_COLOR_PALETTE = ["A351FB", "FF4040", "FFA1A0", "FF7633", "FFB633", "D1D435", "4CFB12", "94CF1A", "40DE8A", "1B9640", "00D6C1", "2E9CAA", "00C4FF", "364797", "6675FF", "0019EF", "863AFF", "530087", "CD3AFF", "FF97CA", "FF39C9"]
comptime LEGACY_COLOR_PALETTE = ["#A351FB", "#E6194B", "#3CB44B", "#FFE119", "#0082C8", "#F58231", "#911EB4", "#46F0F0", "#F032E6", "#D2F53C", "#FABEBE", "#008080", "#E6BEFF", "#AA6E28", "#FFFAC8", "#800000", "#AAFFC3"]
comptime ROBOFLOW_COLOR_PALETTE = ["C28DFC", "A351FB", "8315F9", "6706CE", "5905B3", "4D049A"]

comptime _HEX = "0123456789abcdef"


def _validate_color_hex(color_hex: String) raises:
    var h = String(color_hex.lstrip("#"))
    for c in h.codepoint_slices():
        if String(c).lower() not in _HEX:
            raise Error("Invalid characters in color hash")
    var n = h.byte_length()
    if n != 3 and n != 4 and n != 6 and n != 8:
        raise Error("Invalid length of color hash")


def _hex2(v: Int) -> String:
    return String(_HEX[byte=v >> 4]) + String(_HEX[byte=v & 15])


struct Color(Copyable, Movable, Equatable, Writable):
    var r: Int
    var g: Int
    var b: Int
    var a: Int

    def __init__(out self, r: Int, g: Int, b: Int, a: Int = 255) raises:
        if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255 and 0 <= a <= 255):
            raise Error(String("Color values must be in range 0-255, got (", r, ", ", g, ", ", b, ", ", a, ")"))
        self.r = r
        self.g = g
        self.b = b
        self.a = a

    def __init__(out self, *, copy: Self):
        self.r = copy.r
        self.g = copy.g
        self.b = copy.b
        self.a = copy.a

    @staticmethod
    def from_hex(color_hex: String) raises -> Color:
        """3, 4, 6 or 8 hex digits, optional leading '#'; 3 and 4 double each digit."""
        _validate_color_hex(color_hex)
        var h = String(color_hex.lstrip("#"))
        if h.byte_length() == 3 or h.byte_length() == 4:
            var d = String()
            for c in h.codepoint_slices():
                d += String(c) + String(c)
            h = d^
        var v = List[Int]()
        for i in range(0, h.byte_length(), 2):
            v.append(atol(String(h[byte = i : i + 2]), base=16))
        if len(v) == 3:
            return Color(v[0], v[1], v[2])
        return Color(v[0], v[1], v[2], v[3])

    @staticmethod
    def from_rgb_tuple(r: Int, g: Int, b: Int) raises -> Color:
        if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
            raise Error(String("RGB values must be in range 0-255, got (", r, ", ", g, ", ", b, ")"))
        return Color(r, g, b)

    @staticmethod
    def from_bgr_tuple(b: Int, g: Int, r: Int) raises -> Color:
        if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
            raise Error(String("BGR values must be in range 0-255, got (", b, ", ", g, ", ", r, ")"))
        return Color(r, g, b)

    @staticmethod
    def from_rgba_tuple(r: Int, g: Int, b: Int, a: Int) raises -> Color:
        if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255 and 0 <= a <= 255):
            raise Error(String("RGBA values must be in range 0-255, got (", r, ", ", g, ", ", b, ", ", a, ")"))
        return Color(r, g, b, a)

    @staticmethod
    def from_bgra_tuple(b: Int, g: Int, r: Int, a: Int) raises -> Color:
        if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255 and 0 <= a <= 255):
            raise Error(String("BGRA values must be in range 0-255, got (", b, ", ", g, ", ", r, ", ", a, ")"))
        return Color(r, g, b, a)

    def as_hex(self) -> String:
        """`#rrggbb`, or `#rrggbbaa` when alpha is not 255 (lowercase)."""
        var s = "#" + _hex2(self.r) + _hex2(self.g) + _hex2(self.b)
        if self.a != 255:
            s += _hex2(self.a)
        return s

    def as_rgb(self) -> Tuple[Int, Int, Int]:
        return (self.r, self.g, self.b)

    def as_bgr(self) -> Tuple[Int, Int, Int]:
        return (self.b, self.g, self.r)

    def as_rgba(self) -> Tuple[Int, Int, Int, Int]:
        return (self.r, self.g, self.b, self.a)

    def as_bgra(self) -> Tuple[Int, Int, Int, Int]:
        return (self.b, self.g, self.r, self.a)

    @staticmethod
    def WHITE() raises -> Color:
        return Color.from_hex("#FFFFFF")

    @staticmethod
    def BLACK() raises -> Color:
        return Color.from_hex("#000000")

    @staticmethod
    def GREY() raises -> Color:
        return Color.from_hex("#808080")

    @staticmethod
    def RED() raises -> Color:
        return Color.from_hex("#FF0000")

    @staticmethod
    def GREEN() raises -> Color:
        return Color.from_hex("#00FF00")

    @staticmethod
    def BLUE() raises -> Color:
        return Color.from_hex("#0000FF")

    @staticmethod
    def YELLOW() raises -> Color:
        return Color.from_hex("#FFFF00")

    @staticmethod
    def ROBOFLOW() raises -> Color:
        return Color.from_hex("#A351FB")

    def __eq__(self, o: Self) -> Bool:
        return self.r == o.r and self.g == o.g and self.b == o.b and self.a == o.a

    def write_to(self, mut w: Some[Writer]):
        """Python repr: alpha shown only when not 255."""
        w.write("Color(r=", self.r, ", g=", self.g, ", b=", self.b)
        if self.a != 255:
            w.write(", a=", self.a)
        w.write(")")


struct ColorPalette(Copyable, Movable, Sized):
    var colors: List[Color]

    def __init__(out self, var colors: List[Color]):
        self.colors = colors^

    def __init__(out self, *, copy: Self):
        self.colors = copy.colors.copy()

    @staticmethod
    def from_hex(color_hex_list: List[String]) raises -> ColorPalette:
        var cs = List[Color]()
        for h in color_hex_list:
            cs.append(Color.from_hex(h))
        return ColorPalette(cs^)

    @staticmethod
    def _of[n: Int](hexes: Array[String, n]) raises -> ColorPalette:
        var cs = List[Color]()
        for i in range(n):
            cs.append(Color.from_hex(hexes[i]))
        return ColorPalette(cs^)

    @staticmethod
    def DEFAULT() raises -> ColorPalette:
        return ColorPalette._of(materialize[DEFAULT_COLOR_PALETTE]())

    @staticmethod
    def ROBOFLOW() raises -> ColorPalette:
        return ColorPalette._of(materialize[ROBOFLOW_COLOR_PALETTE]())

    @staticmethod
    def LEGACY() raises -> ColorPalette:
        return ColorPalette._of(materialize[LEGACY_COLOR_PALETTE]())

    def by_idx(self, idx: Int) raises -> Color:
        """Wraps like Python's `%`: negative and oversized indexes are fine."""
        var n = len(self.colors)
        if n == 0:
            raise Error("ColorPalette must contain at least one color")
        return self.colors[((idx % n) + n) % n].copy()

    def __len__(self) -> Int:
        return len(self.colors)


def unify_to_bgr(color: Color) -> Tuple[Int, Int, Int]:
    return color.as_bgr()

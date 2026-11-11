# Port of supervision/geometry/core.py (MIT, see NOTICE-supervision-LICENSE.md): Position, Point, Vector, Rect.
from std.math import sqrt


struct Position(ImplicitlyCopyable, Movable, Equatable):
    """Anchor position; the value is the index into `Position.NAMES` (definition order of the Python enum)."""

    var v: Int

    comptime CENTER = Position(0)
    comptime CENTER_LEFT = Position(1)
    comptime CENTER_RIGHT = Position(2)
    comptime TOP_CENTER = Position(3)
    comptime TOP_LEFT = Position(4)
    comptime TOP_RIGHT = Position(5)
    comptime BOTTOM_LEFT = Position(6)
    comptime BOTTOM_CENTER = Position(7)
    comptime BOTTOM_RIGHT = Position(8)
    comptime CENTER_OF_MASS = Position(9)

    def __init__(out self, v: Int):
        self.v = v

    def __init__(out self, *, copy: Self):
        self.v = copy.v

    def __eq__(self, o: Self) -> Bool:
        return self.v == o.v

    def name(self) -> String:
        return Position.names()[self.v]

    @staticmethod
    def names() -> List[String]:
        """All position values in definition order (Python `Position.list()`)."""
        return [
            "CENTER", "CENTER_LEFT", "CENTER_RIGHT", "TOP_CENTER", "TOP_LEFT",
            "TOP_RIGHT", "BOTTOM_LEFT", "BOTTOM_CENTER", "BOTTOM_RIGHT", "CENTER_OF_MASS",
        ]

    @staticmethod
    def from_name(s: String) raises -> Position:
        var n = Position.names()
        for i in range(len(n)):
            if n[i] == s:
                return Position(i)
        raise Error("supervision: unknown Position '" + s + "'")


struct Point(Copyable, Movable):
    var x: Float64
    var y: Float64

    def __init__(out self, x: Float64, y: Float64):
        self.x = x
        self.y = y

    def __init__(out self, *, copy: Self):
        self.x = copy.x
        self.y = copy.y

    def as_xy_int_tuple(self) -> Tuple[Int, Int]:
        """Truncation toward zero, as Python `int()`."""
        return (Int(self.x), Int(self.y))

    def as_xy_float_tuple(self) -> Tuple[Float64, Float64]:
        return (self.x, self.y)


struct Vector(Copyable, Movable):
    var start: Point
    var end: Point

    def __init__(out self, start: Point, end: Point):
        self.start = start.copy()
        self.end = end.copy()

    def __init__(out self, *, copy: Self):
        self.start = copy.start.copy()
        self.end = copy.end.copy()

    def magnitude(self) -> Float64:
        var dx = self.end.x - self.start.x
        var dy = self.end.y - self.start.y
        return sqrt(dx * dx + dy * dy)

    def center(self) -> Point:
        return Point((self.start.x + self.end.x) / 2, (self.start.y + self.end.y) / 2)

    def cross_product(self, point: Point) -> Float64:
        """Positive when `point` is left of start -> end, negative right, 0 collinear."""
        var dx_vector = self.end.x - self.start.x
        var dy_vector = self.end.y - self.start.y
        var dx_point = point.x - self.start.x
        var dy_point = point.y - self.start.y
        return (dx_vector * dy_point) - (dy_vector * dx_point)


struct Rect(Copyable, Movable):
    var x: Float64
    var y: Float64
    var width: Float64
    var height: Float64

    def __init__(out self, x: Float64, y: Float64, width: Float64, height: Float64):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def __init__(out self, *, copy: Self):
        self.x = copy.x
        self.y = copy.y
        self.width = copy.width
        self.height = copy.height

    @staticmethod
    def from_xyxy(x1: Float64, y1: Float64, x2: Float64, y2: Float64) -> Rect:
        return Rect(x1, y1, x2 - x1, y2 - y1)

    def top_left(self) -> Point:
        return Point(self.x, self.y)

    def bottom_right(self) -> Point:
        return Point(self.x + self.width, self.y + self.height)

    def pad(self, padding: Int) -> Rect:
        var p = Float64(padding)
        return Rect(self.x - p, self.y - p, self.width + 2 * p, self.height + 2 * p)

    def as_xyxy_int_tuple(self) -> Tuple[Int, Int, Int, Int]:
        return (Int(self.x), Int(self.y), Int(self.x + self.width), Int(self.y + self.height))

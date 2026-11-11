# S5 geometry unit test: values checked by hand against supervision's docstrings and numpy semantics.
from supervision.geometry.core import Point, Vector, Rect, Position
from supervision.geometry.utils import get_polygon_center, round_half_even


def check(ok: Bool, what: String) raises:
    if not ok:
        raise Error("FAIL " + what)


def main() raises:
    var n = 0
    var r = Rect(10.0, 20.0, 30.0, 40.0)
    var br = r.bottom_right()
    check(br.x == 40.0 and br.y == 60.0, "rect.bottom_right"); n += 1
    var t = r.as_xyxy_int_tuple()
    check(t[0] == 10 and t[1] == 20 and t[2] == 40 and t[3] == 60, "rect.xyxy_int"); n += 1
    var p = r.pad(5)
    check(p.x == 5.0 and p.width == 40.0 and p.height == 50.0, "rect.pad"); n += 1
    check(Point(-1.9, 2.9).as_xy_int_tuple()[0] == -1, "point int truncates toward zero"); n += 1
    var v = Vector(Point(0.0, 0.0), Point(3.0, 4.0))
    check(v.magnitude() == 5.0 and v.center().x == 1.5 and v.center().y == 2.0, "vector"); n += 1
    check(v.cross_product(Point(0.0, 1.0)) == 3.0 and v.cross_product(Point(6.0, 8.0)) == 0.0, "cross"); n += 1
    check(round_half_even(0.5) == 0.0 and round_half_even(1.5) == 2.0 and round_half_even(2.5) == 2.0 and round_half_even(-0.5) == -0.0 and round_half_even(-1.5) == -2.0, "round_half_even"); n += 1
    var sq = List[Point]()
    sq.append(Point(0.0, 0.0)); sq.append(Point(0.0, 2.0)); sq.append(Point(2.0, 2.0)); sq.append(Point(2.0, 0.0))
    var c = get_polygon_center(sq)
    check(c.x == 1.0 and c.y == 1.0, "polygon center square"); n += 1
    var line = List[Point]()
    line.append(Point(0.0, 0.0)); line.append(Point(3.0, 3.0))
    var c2 = get_polygon_center(line)   # zero area: mean (1.5, 1.5) rounds half-even to (2, 2)
    check(c2.x == 2.0 and c2.y == 2.0, "polygon center degenerate"); n += 1
    var raised = False
    try:
        _ = get_polygon_center(List[Point]())
    except:
        raised = True
    check(raised, "empty polygon raises"); n += 1
    check(Position.TOP_LEFT.name() == "TOP_LEFT" and Position.from_name("CENTER_OF_MASS") == Position.CENTER_OF_MASS and len(Position.names()) == 10, "position"); n += 1
    print("PASS geometry", n)

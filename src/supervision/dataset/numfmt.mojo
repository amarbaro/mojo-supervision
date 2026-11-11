"""Number text like Python's: `f"{x:.5f}"`, float(), int() with whole-number floats."""
from std.ffi import external_call
from std.math import isfinite


def fmt_fixed(x: Float64, digits: Int) raises -> String:
    """C `%.{digits}f`, which rounds the exact binary value as Python's format does."""
    var buf = List[UInt8](length=400, fill=0)
    var fmt: List[UInt8] = [37, 46, 42, 102, 0]  # "%.*f"
    _ = external_call["snprintf", Int32, num_fixed_args=3](buf.unsafe_ptr(), Int(400), fmt.unsafe_ptr(), Int32(digits), x)
    _ = fmt[0]
    var n = 0
    while buf[n] != 0:
        n += 1
    return String(from_utf8=Span[UInt8](buf)[:n])


def py_float(s: String) raises -> Float64:
    """float(s); the error text is Python's."""
    var t = String(s.strip())
    try:
        return Float64(t)
    except:
        raise Error("could not convert string to float: " + _repr(s))


def _repr(s: String) -> String:
    if s.find("'") >= 0 and s.find('"') < 0:
        return '"' + s + '"'
    return "'" + s.replace("'", "\\'") + "'"

"""Row-binary plumbing for tests/run.sh: stdin rows in, one JSON line per row out."""
from std.ffi import c_ssize_t, external_call
from json import Json, parse


def read_all_stdin() raises -> String:
    var buf = List[UInt8]()
    var chunk = List[UInt8](unsafe_uninit_length=65536)
    while True:
        var n = external_call["__read", c_ssize_t](0, chunk.unsafe_ptr(), 65536)
        if n <= 0:
            break
        for i in range(Int(n)):
            buf.append(chunk[i])
    return String(from_utf8=Span[UInt8](buf))


def error_json(msg: String) -> Json:
    var o = Json.object()
    o.set("error", Json.of(msg))
    return o^


def str_list(v: List[String]) -> Json:
    var a = Json.array()
    for s in v:
        a.push(Json.of(s))
    return a^


def int_list(v: List[Int]) -> Json:
    var a = Json.array()
    for x in v:
        a.push(Json.of(x))
    return a^


def strings_of(a: Json) -> List[String]:
    var out = List[String]()
    for i in range(a.len()):
        out.append(a.at(i).str())
    return out^


def read_rows() raises -> List[String]:
    """Non-empty stdin lines (each `op\targs\texpected`)."""
    var out = List[String]()
    for line in read_all_stdin().split("\n"):
        if line.byte_length() > 0:
            out.append(String(line))
    return out^


def row_op(line: String) -> String:
    return String(line.split("\t")[0])


def row_args(line: String) raises -> Json:
    return parse(String(line.split("\t")[1]))

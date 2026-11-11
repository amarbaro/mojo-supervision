"""The YAML subset YOLO `data.yaml` needs: read a block mapping, write `nc` + `names`.

Reader: top-level `key: value` pairs where a value is a scalar, a flow list, a block
list, or a one-level block mapping. Anything deeper (anchors, flow mappings, block
scalars) is kept as an opaque `K_OTHER` scalar. Scalars resolve like PyYAML (YAML 1.1).
Writer: reproduces `yaml.dump({"nc": n, "names": [...]}, sort_keys=False,
default_flow_style=None)` byte for byte, including width-80 flow wrapping.
Not supported: names containing a newline (raises).
"""

comptime K_NULL = 0
comptime K_BOOL = 1
comptime K_INT = 2
comptime K_FLOAT = 3
comptime K_STR = 4
comptime K_OTHER = 5

comptime V_SCALAR = 0
comptime V_SEQ = 1
comptime V_MAP = 2


@fieldwise_init
struct YScalar(Copyable, Movable):
    var kind: Int
    var text: String  # Python's str(value) for the resolved scalar
    var ival: Int
    var fval: Float64

    def type_name(self) -> String:
        if self.kind == K_NULL:
            return "NoneType"
        if self.kind == K_BOOL:
            return "bool"
        if self.kind == K_INT:
            return "int"
        if self.kind == K_FLOAT:
            return "float"
        return "str"


@fieldwise_init
struct YValue(Copyable, Movable):
    var kind: Int  # V_SCALAR | V_SEQ | V_MAP
    var scalar: YScalar  # V_SCALAR
    var items: List[YScalar]  # V_SEQ items, V_MAP values
    var keys: List[YScalar]  # V_MAP keys

    def type_name(self) -> String:
        if self.kind == V_SEQ:
            return "list"
        if self.kind == V_MAP:
            return "dict"
        return self.scalar.type_name()


struct YDoc(Movable):
    var root_kind: Int  # V_MAP when the root is a mapping, else what it is
    var keys: List[String]
    var values: List[YValue]
    var root_type: String

    def __init__(out self):
        self.root_kind = V_MAP
        self.keys = List[String]()
        self.values = List[YValue]()
        self.root_type = "dict"

    def find(self, key: String) -> Int:
        for i in range(len(self.keys)):
            if self.keys[i] == key:
                return i
        return -1


def _str_scalar(s: String) -> YScalar:
    return YScalar(K_STR, s, 0, 0.0)


def _is_digit(c: UInt8) -> Bool:
    return c >= 48 and c <= 57


def _all_in(s: Span[UInt8, _], lo: Int, hi: Int, digits_only: Bool, allow_us: Bool) -> Bool:
    if lo >= hi:
        return False
    for i in range(lo, hi):
        var c = s[i]
        var ok = _is_digit(c) if digits_only else (_is_digit(c) or (c >= 97 and c <= 102) or (c >= 65 and c <= 70))
        if not ok and not (allow_us and c == 95):
            return False
    return True


def _parse_radix(s: Span[UInt8, _], lo: Int, hi: Int, radix: Int) -> Int:
    var v = 0
    for i in range(lo, hi):
        var c = Int(s[i])
        if c == 95:
            continue
        var d = c - 48 if c <= 57 else (c - 87 if c >= 97 else c - 55)
        v = v * radix + d
    return v


def _in_set(s: String, words: List[String]) -> Bool:
    for w in words:
        if w == s:
            return True
    return False


def _is_timestamp(s: Span[UInt8, _]) -> Bool:
    var n = len(s)
    if n < 10:
        return False
    for i in range(10):
        var d = i == 4 or i == 7
        if d:
            if s[i] != 45:
                return False
        elif not _is_digit(s[i]):
            return False
    if n == 10:
        return True
    var c = s[10]
    if c != 84 and c != 116 and c != 32 and c != 9:
        return False
    # YYYY-M(M)-D(D)[Tt ]H(H):MM:SS... : the full-datetime form needs \d\d?:\d\d:\d\d
    var i = 11
    while i < n and (s[i] == 32 or s[i] == 9):
        i += 1
    var h = 0
    while i < n and _is_digit(s[i]) and h < 2:
        i += 1
        h += 1
    if h == 0 or i + 5 > n or s[i] != 58:
        return False
    return (
        _is_digit(s[i + 1]) and _is_digit(s[i + 2]) and s[i + 3] == 58
        and _is_digit(s[i + 4]) and _is_digit(s[i + 5])
    )


def resolve_plain(text: String) -> YScalar:
    """Resolve an unquoted scalar the way PyYAML's implicit resolvers do."""
    var s = text.as_bytes()
    var n = len(s)
    if n == 0 or text == "~" or text == "null" or text == "Null" or text == "NULL":
        return YScalar(K_NULL, "None", 0, 0.0)
    var truthy: List[String] = ["yes", "Yes", "YES", "true", "True", "TRUE", "on", "On", "ON"]
    var falsy: List[String] = ["no", "No", "NO", "false", "False", "FALSE", "off", "Off", "OFF"]
    if _in_set(text, truthy):
        return YScalar(K_BOOL, "True", 1, 0.0)
    if _in_set(text, falsy):
        return YScalar(K_BOOL, "False", 0, 0.0)
    if text == "<<" or text == "=" or _is_timestamp(s):
        return YScalar(K_OTHER, text, 0, 0.0)
    var lo = 0
    var neg = False
    if s[0] == 45 or s[0] == 43:
        neg = s[0] == 45
        lo = 1
    if lo >= n:
        return _str_scalar(text)
    # int forms
    var iv = -1
    if lo + 2 < n + 0 and s[lo] == 48 and (s[lo + 1] == 120) and _all_in(s, lo + 2, n, False, True):
        iv = _parse_radix(s, lo + 2, n, 16)
    elif lo + 2 < n + 0 and s[lo] == 48 and s[lo + 1] == 98 and _all_in(s, lo + 2, n, True, True):
        var ok = True
        for i in range(lo + 2, n):
            if s[i] != 48 and s[i] != 49 and s[i] != 95:
                ok = False
        if ok:
            iv = _parse_radix(s, lo + 2, n, 2)
    elif s[lo] == 48 and n - lo > 1 and _all_in(s, lo + 1, n, True, True):
        var ok = True
        for i in range(lo + 1, n):
            if s[i] > 55 and s[i] != 95:
                ok = False
        if ok:
            iv = _parse_radix(s, lo + 1, n, 8)
    elif s[lo] == 48 and n - lo == 1:
        iv = 0
    elif s[lo] >= 49 and s[lo] <= 57 and _all_in(s, lo, n, True, True):
        iv = _parse_radix(s, lo, n, 10)
    if iv >= 0:
        var v = -iv if neg else iv
        return YScalar(K_INT, String(v), v, Float64(v))
    # float forms
    var body = String(text[byte = lo :])
    if body == ".inf" or body == ".Inf" or body == ".INF":
        return YScalar(K_FLOAT, "-inf" if neg else "inf", 0, Float64.MAX if not neg else -Float64.MAX)
    if text == ".nan" or text == ".NaN" or text == ".NAN":
        return YScalar(K_FLOAT, "nan", 0, 0.0)
    var dot = -1
    for i in range(lo, n):
        if s[i] == 46:
            dot = i
            break
    if dot < 0:
        return _str_scalar(text)
    # [-+]?[0-9][0-9_]*\.[0-9_]*([eE][-+][0-9]+)?  |  \.[0-9][0-9_]*(...)?
    if dot == lo:
        if dot + 1 >= n or not _is_digit(s[dot + 1]):
            return _str_scalar(text)
    else:
        if not _is_digit(s[lo]):
            return _str_scalar(text)
        for i in range(lo, dot):
            if not (_is_digit(s[i]) or s[i] == 95):
                return _str_scalar(text)
    var e = dot + 1
    while e < n and (_is_digit(s[e]) or s[e] == 95):
        e += 1
    if e < n:
        if not ((s[e] == 101 or s[e] == 69) and e + 2 < n + 0 and (s[e + 1] == 45 or s[e + 1] == 43)):
            return _str_scalar(text)
        if not _all_in(s, e + 2, n, True, False):
            return _str_scalar(text)
    var clean = String()
    for i in range(n):
        if s[i] != 95:
            clean += chr(Int(s[i]))
    try:
        return YScalar(K_FLOAT, text, 0, Float64(clean))
    except:
        return _str_scalar(text)


def _decode_double(t: String) raises -> String:
    var s = t.as_bytes()
    var out = String()
    var i = 0
    var n = len(s)
    while i < n:
        var c = s[i]
        if c != 92:
            out += chr(Int(c)) if c < 128 else String(StringSlice(unsafe_from_utf8=s[i : i + 1]))
            i += 1
            continue
        i += 1
        if i >= n:
            break
        var e = s[i]
        i += 1
        if e == 110:
            out += "\n"
        elif e == 116:
            out += "\t"
        elif e == 114:
            out += "\r"
        elif e == 48:
            out += chr(0)
        elif e == 120 or e == 117 or e == 85:
            var k = 2 if e == 120 else (4 if e == 117 else 8)
            var cp = _parse_radix(s, i, min(i + k, n), 16)
            out += chr(cp)
            i += k
        else:
            out += chr(Int(e))
    return out^


def _strip_comment(line: String) -> String:
    var s = line.as_bytes()
    var q = 0  # 0 none, 39 single, 34 double
    for i in range(len(s)):
        var c = s[i]
        if q == 0:
            if c == 39 or c == 34:
                q = Int(c)
            elif c == 35 and (i == 0 or s[i - 1] == 32 or s[i - 1] == 9):
                return String(line[byte = :i])
        elif Int(c) == q:
            q = 0
    return line


def _scalar_from_token(tok: String) raises -> YScalar:
    var t = String(tok.strip())
    var s = t.as_bytes()
    if len(s) >= 2 and s[0] == 39 and s[len(s) - 1] == 39:
        return _str_scalar(String(t[byte = 1 : len(s) - 1]).replace("''", "'"))
    if len(s) >= 2 and s[0] == 34 and s[len(s) - 1] == 34:
        return _str_scalar(_decode_double(String(t[byte = 1 : len(s) - 1])))
    if len(s) > 0 and (s[0] == 38 or s[0] == 42 or s[0] == 33 or s[0] == 124 or s[0] == 62 or s[0] == 123):
        return YScalar(K_OTHER, t, 0, 0.0)
    return resolve_plain(t)


def _split_flow(inner: String) raises -> List[YScalar]:
    """Items of `a, 'b, c', "d"` (one level; nested brackets become K_OTHER)."""
    var out = List[YScalar]()
    for tok in _split_tokens(inner):
        out.append(_scalar_from_token(tok))
    return out^


def _split_tokens(inner: String) -> List[String]:
    var out = List[String]()
    var s = inner.as_bytes()
    var depth = 0
    var q = 0
    var start = 0
    for i in range(len(s) + 1):
        var at_end = i == len(s)
        if not at_end:
            var c = s[i]
            if q != 0:
                if Int(c) == q:
                    q = 0
                continue
            if c == 39 or c == 34:
                q = Int(c)
                continue
            if c == 91 or c == 123:
                depth += 1
                continue
            if c == 93 or c == 125:
                depth -= 1
                continue
        if at_end or (s[i] == 44 and depth == 0):
            var tok = String(inner[byte = start : i]).strip()
            if tok.byte_length() > 0 or not at_end:
                if tok.byte_length() > 0:
                    out.append(String(tok))
            start = i + 1
    return out^


def _flow_map(inner: String) raises -> YValue:
    var keys = List[YScalar]()
    var vals = List[YScalar]()
    for tok in _split_tokens(inner):
        var k = _split_key(tok)
        if k < 0:
            keys.append(_scalar_from_token(tok))
            vals.append(YScalar(K_NULL, "None", 0, 0.0))
        else:
            keys.append(_scalar_from_token(String(tok[byte = :k])))
            var v = String(String(tok[byte = k + 1 :]).strip())
            vals.append(_scalar_from_token(v) if v.byte_length() > 0 else YScalar(K_NULL, "None", 0, 0.0))
    return YValue(V_MAP, _str_scalar(""), vals^, keys^)


def _flow_balanced(t: String) -> Bool:
    var depth = 0
    var q = 0
    for c in t.as_bytes():
        if q != 0:
            if Int(c) == q:
                q = 0
        elif c == 39 or c == 34:
            q = Int(c)
        elif c == 91 or c == 123:
            depth += 1
        elif c == 93 or c == 125:
            depth -= 1
    return depth <= 0


def _split_key(content: String) -> Int:
    """Byte index of the `:` that ends a mapping key, or -1."""
    var s = content.as_bytes()
    var q = 0
    for i in range(len(s)):
        var c = s[i]
        if q != 0:
            if Int(c) == q:
                q = 0
        elif c == 39 or c == 34:
            q = Int(c)
        elif c == 58 and (i + 1 == len(s) or s[i + 1] == 32):
            return i
    return -1


@fieldwise_init
struct _Line(Copyable, Movable):
    var indent: Int
    var text: String


def _lines(text: String) -> List[_Line]:
    var out = List[_Line]()
    for raw in text.split("\n"):
        var l = _strip_comment(String(raw.rstrip("\r")))
        var t = String(l.rstrip(" \t"))
        var ind = 0
        var b = t.as_bytes()
        while ind < len(b) and b[ind] == 32:
            ind += 1
        if len(b) == ind:
            continue
        if ind == 0 and (t == "---" or t.startswith("--- ")):
            continue
        if ind == 0 and t == "...":
            break
        out.append(_Line(ind, String(t[byte = ind :])))
    return out^


def parse_yaml(text: String) raises -> YDoc:
    var ls = _lines(text)
    var doc = YDoc()
    if len(ls) == 0:
        doc.root_kind = V_SCALAR
        doc.root_type = "NoneType"
        return doc^
    var first = ls[0].text
    if first == "-" or first.startswith("- "):
        doc.root_kind = V_SEQ
        doc.root_type = "list"
        return doc^
    if first.startswith("["):
        doc.root_kind = V_SEQ
        doc.root_type = "list"
        return doc^
    if _split_key(first) < 0 and not first.startswith("{"):
        doc.root_kind = V_SCALAR
        doc.root_type = _scalar_from_token(first).type_name()
        return doc^
    var base = ls[0].indent
    var i = 0
    while i < len(ls):
        var ln = ls[i].copy()
        if ln.indent != base:
            i += 1
            continue
        var k = _split_key(ln.text)
        if k < 0:
            i += 1
            continue
        var key = _scalar_from_token(String(ln.text[byte = :k])).text
        var rest = String(String(ln.text[byte = k + 1 :]).strip())
        i += 1
        if rest.startswith("&"):  # anchor: the node follows on the same or the next lines
            var sp = rest.find(" ")
            rest = String(String(rest[byte = sp + 1 :]).strip()) if sp >= 0 else String()
        if rest.byte_length() > 0:
            if rest.startswith("{"):
                while not _flow_balanced(rest) and i < len(ls):
                    rest += " " + ls[i].text
                    i += 1
                var close = rest.rfind("}")
                doc.keys.append(key^)
                doc.values.append(_flow_map(String(rest[byte = 1 : close if close > 0 else rest.byte_length()])))
                continue
            if rest.startswith("["):
                while not _flow_balanced(rest) and i < len(ls):
                    rest += " " + ls[i].text
                    i += 1
                var close = rest.rfind("]")
                var inner = String(rest[byte = 1 : close if close > 0 else rest.byte_length()])
                doc.keys.append(key^)
                doc.values.append(YValue(V_SEQ, _str_scalar(""), _split_flow(inner), List[YScalar]()))
                continue
            if rest == "|" or rest == ">" or rest.startswith("|") or rest.startswith(">"):
                while i < len(ls) and ls[i].indent > base:
                    i += 1
                doc.keys.append(key^)
                doc.values.append(YValue(V_SCALAR, YScalar(K_OTHER, rest, 0, 0.0), List[YScalar](), List[YScalar]()))
                continue
            while i < len(ls) and ls[i].indent > base:
                rest += " " + ls[i].text
                i += 1
            doc.keys.append(key^)
            doc.values.append(YValue(V_SCALAR, _scalar_from_token(rest), List[YScalar](), List[YScalar]()))
            continue
        # empty value: block list at base or deeper, or block mapping deeper
        if i < len(ls) and ls[i].indent >= base and (ls[i].text == "-" or ls[i].text.startswith("- ")) and (ls[i].indent > base or True):
            var li = ls[i].indent
            var items = List[YScalar]()
            while i < len(ls) and ls[i].indent == li and (ls[i].text == "-" or ls[i].text.startswith("- ")):
                var it = String(String(ls[i].text[byte = 1 :]).strip())
                items.append(_scalar_from_token(it) if it.byte_length() > 0 else YScalar(K_NULL, "None", 0, 0.0))
                i += 1
                while i < len(ls) and ls[i].indent > li:
                    i += 1
            doc.keys.append(key^)
            doc.values.append(YValue(V_SEQ, _str_scalar(""), items^, List[YScalar]()))
            continue
        if i < len(ls) and ls[i].indent > base:
            var mi = ls[i].indent
            var mk = List[YScalar]()
            var mv = List[YScalar]()
            while i < len(ls) and ls[i].indent > base:
                if ls[i].indent == mi:
                    var kk = _split_key(ls[i].text)
                    if kk >= 0:
                        var v = String(String(ls[i].text[byte = kk + 1 :]).strip())
                        mk.append(_scalar_from_token(String(ls[i].text[byte = :kk])))
                        mv.append(_scalar_from_token(v) if v.byte_length() > 0 else YScalar(K_NULL, "None", 0, 0.0))
                i += 1
            doc.keys.append(key^)
            doc.values.append(YValue(V_MAP, _str_scalar(""), mv^, mk^))
            continue
        doc.keys.append(key^)
        doc.values.append(YValue(V_SCALAR, YScalar(K_NULL, "None", 0, 0.0), List[YScalar](), List[YScalar]()))
    return doc^


def read_yaml_file(path: String) raises -> YDoc:
    var text: String
    with open(path, "r") as f:
        text = f.read()
    return parse_yaml(text)


# ---------------------------------------------------------------- writer

comptime _WIDTH = 80


def _hex(v: Int, digits: Int) -> String:
    var out = String()
    var h = "0123456789ABCDEF"
    for i in range(digits):
        var sh = (digits - 1 - i) * 4
        out += String(h[byte = (v >> sh) & 15])
    return out


def _escape(v: Int) -> String:
    """PyYAML ESCAPE_REPLACEMENTS, then \\xXX / \\uXXXX / \\UXXXXXXXX."""
    if v == 0:
        return "\\0"
    if v == 7:
        return "\\a"
    if v == 8:
        return "\\b"
    if v == 9:
        return "\\t"
    if v == 10:
        return "\\n"
    if v == 11:
        return "\\v"
    if v == 12:
        return "\\f"
    if v == 13:
        return "\\r"
    if v == 27:
        return "\\e"
    if v == 34:
        return '\\"'
    if v == 92:
        return "\\\\"
    if v == 0x85:
        return "\\N"
    if v == 0xA0:
        return "\\_"
    if v == 0x2028:
        return "\\L"
    if v == 0x2029:
        return "\\P"
    if v <= 0xFF:
        return "\\x" + _hex(v, 2)
    if v <= 0xFFFF:
        return "\\u" + _hex(v, 4)
    return "\\U" + _hex(v, 8)


def _plain_allowed_in_flow(s: String) -> Bool:
    """PyYAML analyze_scalar: may this str be written plain inside a flow list?"""
    var b = s.as_bytes()
    var n = len(b)
    if n == 0:
        return False
    if (n >= 3 and (s.startswith("---") or s.startswith("..."))):
        return False
    if b[0] == 32 or b[n - 1] == 32:
        return False
    for i in range(n):
        var c = b[i]
        var next_ws = i + 1 >= n or b[i + 1] == 32
        var prev_ws = i == 0 or b[i - 1] == 32
        if i == 0:
            if c == 35 or c == 44 or c == 91 or c == 93 or c == 123 or c == 125 or c == 38 or c == 42 or c == 33 or c == 124 or c == 62 or c == 39 or c == 34 or c == 37 or c == 64 or c == 96:
                return False
            if c == 63 or c == 58:
                return False
            if c == 45 and next_ws:
                return False
        else:
            if c == 44 or c == 63 or c == 91 or c == 93 or c == 123 or c == 125 or c == 58:
                return False
            if c == 35 and prev_ws:
                return False
    return resolve_plain(s).kind == K_STR


struct _Emit:
    var out: String
    var column: Int

    def __init__(out self):
        self.out = String()
        self.column = 0

    def put(mut self, t: String):
        self.out += t
        self.column += t.count_codepoints()

    def newline_indent(mut self, indent: Int):
        self.out += "\n" + String(" ") * indent
        self.column = indent

    def text_wrapped(mut self, body: String, indent: Int):
        """write_plain / write_single_quoted: a lone space past the width breaks the line."""
        var b = body.as_bytes()
        var n = len(b)
        var i = 0
        var start = 0
        while i < n:
            if b[i] == 32 and (i == 0 or b[i - 1] != 32):
                var j = i
                while j < n and b[j] == 32:
                    j += 1
                if j - i == 1 and j < n and i > 0 and self.column + (i - start) > _WIDTH:
                    self.put(String(body[byte = start : i]))
                    self.newline_indent(indent)
                    start = j
                i = j
            else:
                i += 1
        self.put(String(body[byte = start :]))


def _double_quoted(mut e: _Emit, s: String, indent: Int):
    """write_double_quoted: escapes, and `\\` line continuation past the width."""
    var cps = List[Int]()
    for cp in s.codepoints():
        cps.append(Int(cp))
    var n = len(cps)
    e.put('"')
    var start = 0
    var end = 0
    while end <= n:
        var ch = cps[end] if end < n else -1
        var special = ch == 34 or ch == 92 or ch == 0x85 or ch == 0x2028 or ch == 0x2029 or ch == 0xFEFF or (ch >= 0 and (ch < 32 or ch > 126))
        if ch < 0 or special:
            if start < end:
                e.put(_cps_str(cps, start, end))
                start = end
            if ch >= 0:
                start = end + 1
                e.put(_escape(ch))
        if end > 0 and end < n - 1 and (ch == 32 or start >= end) and e.column + (end - start) > _WIDTH:
            e.put(_cps_str(cps, start, end) + "\\")
            if start < end:
                start = end
            e.newline_indent(indent)
            if cps[start] == 32:
                e.put("\\")
        end += 1
    e.put('"')


def _cps_str(cps: List[Int], lo: Int, hi: Int) -> String:
    var out = String()
    for k in range(lo, hi):
        out += chr(cps[k])
    return out


def dump_data_yaml(classes: List[String]) raises -> String:
    """`yaml.dump({"nc": n, "names": classes}, sort_keys=False, default_flow_style=None)`."""
    var e = _Emit()
    e.out = "nc: " + String(len(classes)) + "\nnames:"
    e.column = 6
    e.put(" [")
    for i in range(len(classes)):
        var c = classes[i]
        if i > 0:
            e.put(",")
        if e.column > _WIDTH:
            e.newline_indent(2)
        elif i > 0:
            e.put(" ")
        var b = c.as_bytes()
        for ch in b:
            if ch == 10 or ch == 13:
                raise Error("yaml: class names with line breaks are not supported")
        var special = False
        for ch in b:
            if ch < 32 or ch > 126:
                special = True
        if special:
            _double_quoted(e, c, 4)
        elif _plain_allowed_in_flow(c):
            e.text_wrapped(c, 4)
        else:
            e.put("'")
            e.text_wrapped(c.replace("'", "''"), 4)
            e.put("'")
    e.put("]\n")
    return e.out.copy()

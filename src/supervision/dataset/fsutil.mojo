"""Files and paths for the dataset formats: pathlib-style helpers, byte IO, image sizes."""
from std.os import listdir, makedirs
from std.os.path import exists, isdir, isfile, realpath


def norm_path(p: String) -> String:
    """str(Path(p)): collapse '//' and '.', drop a trailing '/'."""
    var absolute = p.startswith("/")
    var parts = List[String]()
    for part in p.split("/"):
        if part.byte_length() == 0 or part == ".":
            continue
        parts.append(String(part))
    var out = String()
    for i in range(len(parts)):
        out += ("/" if i > 0 else "") + parts[i]
    if absolute:
        return "/" + out
    return out if out.byte_length() > 0 else String(".")


def join(a: String, b: String) -> String:
    """Path(a) / Path(b)."""
    if b.startswith("/"):
        return norm_path(b)
    return norm_path(a + "/" + b)


def name_of(p: String) -> String:
    var q = norm_path(p)
    var i = q.rfind("/")
    return String(q[byte = i + 1 :]) if i >= 0 else q


def suffix_of(p: String) -> String:
    """Path.suffix: from the last '.', none for '.hidden' or a trailing '.'."""
    var n = name_of(p)
    var i = n.rfind(".")
    if i <= 0 or i == n.byte_length() - 1:
        return String()
    return String(n[byte = i :])


def stem_of(p: String) -> String:
    var n = name_of(p)
    var s = suffix_of(p)
    return String(n[byte = : n.byte_length() - s.byte_length()])


def parent_of(p: String) -> String:
    var q = norm_path(p)
    var i = q.rfind("/")
    if i < 0:
        return String(".")
    return String(q[byte = :i]) if i > 0 else String("/")


def lower_suffix_ok(p: String, exts: List[String]) -> Bool:
    """list_files_with_extensions: any dotted suffix, lower-cased, equals an extension."""
    var n = name_of(p)
    var parts = n.split(".")
    for k in range(1, len(parts)):
        var s = String(parts[k]).lower()
        for e in exts:
            if s == e:
                return True
    return False


def list_files_with_extensions(directory: String, exts: List[String]) raises -> List[String]:
    """Files of `directory` whose suffix matches, in readdir order (Path.glob('*'))."""
    var out = List[String]()
    for n in listdir(directory):
        var full = join(directory, String(n))
        if isfile(full) and lower_suffix_ok(full, exts):
            out.append(full)
    return out^


def mkdirs(p: String) raises:
    makedirs(p, exist_ok=True)


def read_bytes(p: String) raises -> List[UInt8]:
    with open(p, "r") as f:
        return f.read_bytes()


def read_text(p: String) raises -> String:
    var b = read_bytes(p)
    return String(from_utf8=Span[UInt8](b))


def write_bytes(p: String, b: Span[UInt8, _]) raises:
    with open(p, "w") as f:
        f.write_bytes(b)


def write_text(p: String, s: String) raises:
    write_bytes(p, s.as_bytes())


def copy_file(src: String, dst: String) raises:
    """shutil.copyfile; the same file (after realpath) is left as it is."""
    if exists(dst) and realpath(src) == realpath(dst):
        return
    var b = read_bytes(src)
    write_bytes(dst, Span[UInt8](b))


def read_lines(p: String, skip_empty: Bool) raises -> List[String]:
    """read_txt_file: lines without newline; skip_empty drops blank/whitespace-only lines."""
    var out = List[String]()
    var text = read_text(p)
    var parts = text.split("\n")
    var n = len(parts)
    if n > 0 and parts[n - 1].byte_length() == 0:
        n -= 1
    for i in range(n):
        var line = String(parts[i])
        if line.endswith("\r"):
            var cut = String(line[byte = : line.byte_length() - 1])
            line = cut^
        if skip_empty and line.strip().byte_length() == 0:
            continue
        out.append(line)
    return out^


def _be32(b: List[UInt8], i: Int) -> Int:
    return (Int(b[i]) << 24) | (Int(b[i + 1]) << 16) | (Int(b[i + 2]) << 8) | Int(b[i + 3])


def _le32(b: List[UInt8], i: Int) -> Int:
    return Int(b[i]) | (Int(b[i + 1]) << 8) | (Int(b[i + 2]) << 16) | (Int(b[i + 3]) << 24)


def image_size_wh(path: String) raises -> Tuple[Int, Int]:
    """(width, height) from the file header: PNG, JPEG (SOF), BMP, GIF. EXIF turns are not applied."""
    var b = read_bytes(path)
    var n = len(b)
    if n >= 24 and b[0] == 0x89 and b[1] == 0x50 and b[2] == 0x4E and b[3] == 0x47:
        return (_be32(b, 16), _be32(b, 20))
    if n >= 10 and b[0] == 0x47 and b[1] == 0x49 and b[2] == 0x46:
        return (Int(b[6]) | (Int(b[7]) << 8), Int(b[8]) | (Int(b[9]) << 8))
    if n >= 26 and b[0] == 0x42 and b[1] == 0x4D:
        var h = _le32(b, 22)
        if h >= (1 << 31):
            h = (1 << 32) - h
        return (_le32(b, 18), h)
    if n >= 4 and b[0] == 0xFF and b[1] == 0xD8:
        var i = 2
        while i + 9 < n:
            if b[i] != 0xFF:
                i += 1
                continue
            var m = Int(b[i + 1])
            if m == 0xFF:
                i += 1
                continue
            if m >= 0xC0 and m <= 0xCF and m != 0xC4 and m != 0xC8 and m != 0xCC:
                return ((Int(b[i + 7]) << 8) | Int(b[i + 8]), (Int(b[i + 5]) << 8) | Int(b[i + 6]))
            if m == 0xD8 or m == 0x01 or (m >= 0xD0 and m <= 0xD7):
                i += 2
                continue
            i += 2 + ((Int(b[i + 2]) << 8) | Int(b[i + 3]))
    raise Error("Could not read image size from path: " + path)


def list_files_rec(root: String, rel: String = "") raises -> List[String]:
    """Every file under root as a path relative to root, sorted by that string (bundle order)."""
    var out = List[String]()
    var here = root if rel.byte_length() == 0 else root + "/" + rel
    for n in listdir(here):
        var r = String(n) if rel.byte_length() == 0 else rel + "/" + String(n)
        var full = root + "/" + r
        if isdir(full):
            out.extend(list_files_rec(root, r))
        elif isfile(full):
            out.append(r)
    if rel.byte_length() == 0:
        for i in range(1, len(out)):
            var j = i
            while j > 0 and out[j] < out[j - 1]:
                var t = out[j]
                out[j] = out[j - 1]
                out[j - 1] = t
                j -= 1
    return out^

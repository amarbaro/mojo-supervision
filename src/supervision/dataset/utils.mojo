"""dataset/utils.py, the parts that do not touch Detections: split, class lists, collisions."""
from std.collections import Dict
from supervision.dataset.pyrandom import PyRandom


def py_repr(s: String) -> String:
    """Python repr() of a str: single quotes unless it holds ' and no \"."""
    var q = "'"
    if s.find("'") >= 0 and s.find('"') < 0:
        q = '"'
    var out = String(q)
    for cp in s.codepoints():
        var v = Int(cp)
        if v == 92:
            out += "\\\\"
        elif v == 10:
            out += "\\n"
        elif v == 13:
            out += "\\r"
        elif v == 9:
            out += "\\t"
        elif chr(v) == q:
            out += "\\" + q
        else:
            out += chr(v)
    return out + q


def train_test_split_indices(
    n: Int, train_ratio: Float64, random_state: UInt64, shuffle: Bool
) raises -> Tuple[List[Int], List[Int]]:
    """Indices of `train_test_split(list(range(n)), ...)`; seeds are non-negative ints."""
    if not (0.0 <= train_ratio and train_ratio <= 1.0):
        raise Error(
            "Expected a split ratio within the inclusive range [0, 1], got "
            + String(train_ratio) + "."
        )
    var order = List[Int]()
    if shuffle:
        var rng = PyRandom(random_state)
        order = rng.shuffle_indices(n)
    else:
        for i in range(n):
            order.append(i)
    var cut = Int(Float64(n) * train_ratio)
    var train = List[Int]()
    var test = List[Int]()
    for i in range(n):
        if i < cut:
            train.append(order[i])
        else:
            test.append(order[i])
    return (train^, test^)


def merge_class_lists(class_lists: List[List[String]]) -> List[String]:
    """Sorted union of class names (codepoint order, as Python's sorted(set))."""
    var seen = Dict[String, Bool]()
    var out = List[String]()
    for lst in class_lists:
        for name in lst:
            if name not in seen:
                seen[name] = True
                out.append(name)
    sort(out)
    return out^


def build_class_index_mapping(source: List[String], target: List[String]) raises -> List[Int]:
    """Index of each source class inside target; raises if one is missing."""
    var out = List[Int]()
    for name in source:
        var found = -1
        for j in range(len(target)):
            if target[j] == name:
                found = j
                break
        if found < 0:
            raise Error(
                "Class " + name + " not found in target classes. "
                "source_classes must be a subset of target_classes."
            )
        out.append(found)
    return out^


def find_duplicates(seq: List[String]) -> List[String]:
    """Elements seen more than once, in order of their first repeat."""
    var seen = Dict[String, Bool]()
    var dup = Dict[String, Bool]()
    var out = List[String]()
    for s in seq:
        if s in seen:
            if s not in dup:
                dup[s] = True
                out.append(s)
        else:
            seen[s] = True
    return out^


def casefold(s: String) -> String:
    """Lowercase fold (ASCII + Latin via to_lowercase; full Unicode casefold is not ported)."""
    return s.lower()


def check_no_basename_collisions(
    image_paths: List[String], output_names: List[String], output_kind: String
) raises:
    """Raise when two image paths map to the same output file (case-insensitively)."""
    var keys = Dict[String, Int]()
    for i in range(len(image_paths)):
        var k = casefold(output_names[i])
        if k in keys:
            var j = keys[k]
            raise Error(
                "Cannot export dataset: image paths " + py_repr(image_paths[j])
                + " and " + py_repr(image_paths[i]) + " both map to " + output_kind
                + " file " + py_repr(output_names[j])
                + ". Ensure all output paths are unique before exporting."
            )
        keys[k] = i

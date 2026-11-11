"""data.yaml readers of dataset/formats/yolo.py: class names and pose value count."""
from supervision.dataset.yaml import (
    YDoc, YScalar, YValue, V_MAP, V_SEQ, V_SCALAR, K_BOOL, K_INT, K_FLOAT, K_NULL, K_STR,
    read_yaml_file, parse_yaml,
)
from supervision.dataset.utils import py_repr


def _check_map(doc: YDoc, file_path: String) raises:
    if doc.root_kind != V_MAP:
        raise Error(
            "Expected mapping in data.yaml at '" + file_path + "', got " + doc.root_type + "."
        )


def _int_like(k: YScalar) -> Bool:
    if k.kind == K_BOOL:
        return False
    if k.kind == K_INT:
        return True
    if k.kind == K_STR:
        var t = String(k.text.strip())
        if t.byte_length() == 0:
            return False
        for c in t.as_bytes():
            if c < 48 or c > 57:
                return False
        return True
    return False


def _py_str_key(k: YScalar) -> String:
    return k.text


def class_names(doc: YDoc, file_path: String) raises -> List[String]:
    """`_extract_class_names` on a parsed document."""
    _check_map(doc, file_path)
    var at = doc.find("names")
    var out = List[String]()
    if at >= 0 and doc.values[at].kind == V_MAP:
        ref v = doc.values[at]
        var n = len(v.keys)
        var flags = List[Bool]()
        var any_il = False
        var all_il = True
        for k in v.keys:
            var il = _int_like(k)
            flags.append(il)
            any_il = any_il or il
            all_il = all_il and il
        if any_il and not all_il:
            var num = String("[")
            var oth = String("[")
            var nn = 0
            var no = 0
            for i in range(n):
                if flags[i] and nn < 3:
                    num += (", " if nn > 0 else "") + _repr_key(v.keys[i])
                    nn += 1
                elif (not flags[i]) and no < 3:
                    oth += (", " if no > 0 else "") + _repr_key(v.keys[i])
                    no += 1
            raise Error(
                "Expected 'names' dict in data.yaml at '" + file_path
                + "' to have either all numeric or all non-numeric keys, got a mix: "
                + "numeric " + num + "] and non-numeric " + oth + "] keys."
            )
        # stable sort of indices by int(key) or by str(key)
        var order = List[Int]()
        for i in range(n):
            order.append(i)
        for i in range(1, n):
            var j = i
            while j > 0 and _key_less(v.keys[order[j]], v.keys[order[j - 1]], all_il):
                var t = order[j]
                order[j] = order[j - 1]
                order[j - 1] = t
                j -= 1
        for i in order:
            out.append(v.items[i].text)
        return out^
    if at >= 0 and doc.values[at].kind == V_SEQ:
        for s in doc.values[at].items:
            out.append(s.text)
        return out^
    var tn = String("NoneType") if at < 0 else doc.values[at].type_name()
    raise Error(
        "Expected 'names' to be a list or dict in data.yaml at '" + file_path
        + "', got " + tn + "."
    )


def _repr_key(k: YScalar) -> String:
    if k.kind == K_STR:
        return py_repr(k.text)
    return k.text


def _key_less(a: YScalar, b: YScalar, numeric: Bool) raises -> Bool:
    if numeric:
        return _key_int(a) < _key_int(b)
    return a.text < b.text


def _key_int(k: YScalar) raises -> Int:
    if k.kind == K_INT:
        return k.ival
    return Int(k.text.strip())


def _positive_whole(s: YScalar) -> Bool:
    if s.kind == K_INT:
        return s.ival > 0
    if s.kind == K_FLOAT:
        return s.fval > 0.0 and s.fval == Float64(Int(s.fval))
    return False


def _repr_value(v: YValue) -> String:
    if v.kind == V_SEQ:
        var out = String("[")
        for i in range(len(v.items)):
            out += (", " if i > 0 else "") + _repr_key(v.items[i])
        return out + "]"
    if v.kind == V_MAP:
        return "{...}"
    return _repr_key(v.scalar)


def pose_value_count(doc: YDoc, file_path: String) raises -> Int:
    """`_extract_pose_value_count`: K * D from `kpt_shape`, 0 when absent."""
    _check_map(doc, file_path)
    var at = doc.find("kpt_shape")
    if at < 0:
        return 0
    ref v = doc.values[at]
    if (
        v.kind == V_SEQ and len(v.items) == 2 and _positive_whole(v.items[0])
        and _positive_whole(v.items[1])
        and (Int(v.items[1].fval if v.items[1].kind == K_FLOAT else Float64(v.items[1].ival)) == 2
             or Int(v.items[1].fval if v.items[1].kind == K_FLOAT else Float64(v.items[1].ival)) == 3)
    ):
        ref k = v.items[0]
        ref d = v.items[1]
        var kk = k.ival if k.kind == K_INT else Int(k.fval)
        var dd = d.ival if d.kind == K_INT else Int(d.fval)
        return kk * dd
    raise Error(
        "Expected 'kpt_shape' in data.yaml at '" + file_path
        + "' to be [number of keypoints, 2 or 3], got " + _repr_value(v)
        + ". Fix it, or remove 'kpt_shape' if the dataset has no keypoints."
    )

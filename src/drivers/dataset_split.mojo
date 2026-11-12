"""Row binary for tests/cases/dataset_split.txt: shuffle, split, class utils (build/dataset_split)."""
from json import Json
from supervision.dataset.pyrandom import PyRandom
from supervision.dataset.rowio import error_json, int_list, read_rows, row_args, row_op, str_list, strings_of
from supervision.dataset.utils import (
    build_class_index_mapping, check_no_basename_collisions, merge_class_lists, train_test_split_indices,
)


def _name(p: String) -> String:
    var i = p.rfind("/")
    return String(p[byte = i + 1 :]) if i >= 0 else p


def _stem_xml(p: String) -> String:
    var n = _name(p)
    var d = n.rfind(".")
    return (String(n[byte = :d]) if d > 0 else n) + ".xml"


def handle(op: String, args: Json) raises -> Json:
    if op == "shuffle":
        var r = PyRandom(UInt64(args.get("seed").int()))
        return int_list(r.shuffle_indices(args.get("n").int()))
    if op == "split":
        try:
            var t = train_test_split_indices(
                args.get("n").int(), args.get("ratio").num(), UInt64(args.get("seed").int()),
                args.get("shuffle").truthy(),
            )
            var o = Json.object()
            o.set("train", int_list(t[0]))
            o.set("test", int_list(t[1]))
            return o^
        except e:
            return error_json(String(e))
    if op == "merge_classes":
        var lists = List[List[String]]()
        var a = args.get("lists")
        for i in range(a.len()):
            lists.append(strings_of(a.at(i)))
        return str_list(merge_class_lists(lists))
    if op == "class_map":
        try:
            return int_list(build_class_index_mapping(strings_of(args.get("source")), strings_of(args.get("target"))))
        except e:
            return error_json(String(e))
    if op == "collision":
        var paths = strings_of(args.get("paths"))
        var names = List[String]()
        for p in paths:
            names.append(_name(p) if args.text("by") == "name" else _stem_xml(p))
        try:
            check_no_basename_collisions(paths, names, args.text("kind"))
            return Json()
        except e:
            return error_json(String(e))
    raise Error("unknown op " + op)


def main() raises:
    for line in read_rows():
        try:
            print(handle(row_op(line), row_args(line)).dumps())
        except e:
            print(error_json("FAIL " + row_op(line) + ": " + String(e)).dumps())

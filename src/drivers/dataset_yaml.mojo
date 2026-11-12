"""Row binary for tests/cases/dataset_yaml.txt: YAML subset writer/reader (build/dataset_yaml)."""
from json import Json
from supervision.dataset.datayaml import class_names, pose_value_count
from supervision.dataset.rowio import error_json, read_rows, row_args, row_op, str_list, strings_of
from supervision.dataset.yaml import dump_data_yaml, parse_yaml


def handle(op: String, args: Json) raises -> Json:
    if op == "dump":
        return Json.of(dump_data_yaml(strings_of(args.get("names"))))
    var doc = parse_yaml(args.text("text"))
    try:
        if op == "class_names":
            return str_list(class_names(doc, "data.yaml"))
        if op == "pose_count":
            return Json.of(pose_value_count(doc, "data.yaml"))
    except e:
        return error_json(String(e))
    raise Error("unknown op " + op)


def main() raises:
    for line in read_rows():
        try:
            print(handle(row_op(line), row_args(line)).dumps())
        except e:
            print(error_json("FAIL " + row_op(line) + ": " + String(e)).dumps())

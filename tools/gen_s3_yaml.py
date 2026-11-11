"""Oracle rows for dataset/yaml.mojo + datayaml.mojo -> tests/cases/dataset_yaml.txt.

Ops (all strings, byte-exact):
  dump         {"names": [...]}  -> yaml.dump text of {"nc", "names"} (save_data_yaml)
  class_names  {"text": yaml}    -> list[str] | {"error": msg}   (yolo._extract_class_names)
  pose_count   {"text": yaml}    -> int       | {"error": msg}   (yolo._extract_pose_value_count)
Run: .work/oracle-venv/bin/python -I tools/gen_s3_yaml.py     (seed 7)
"""
import io
import os
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import yaml
from oracle import write_rows
from supervision.dataset.formats import yolo

ROOT = Path(__file__).resolve().parent.parent
SEED = 7
TRICKY = ["1", "true", "null", "yes", "a: b", "a b", "-x", "", "x#y", " lead", "tr ail ",
          "it's", 'q"d', "é", "ü ber", "a,b", "[x]", "{y}", "~", "0x1f", "1e3", ".5",
          "2020-01-01", "*a", "&b", "!c", "%d", "@e", "`f", "a\tb", "x:", "-", "--- x",
          "a?b", "?a", "- a", "1.5", "0o7", "012", "off", "On", "=", "<<", "1_000", "+1",
          "dog  cat", "日本", "😀 x"]
DOCS = [
    "train: ../t\nnc: 2\nnames: ['dog', 'person']\n",
    "names:\n  0: a\n  1: b\n  10: c\n",
    "names:\n  10: c\n  9: b\n  2: a\n",
    "names:\n- a\n- b\nnc: 2  # comment\n",
    "names: [a, 'b, c', \"d\\u00e9\", 7, true, null]\nkpt_shape: [17, 3]\n",
    "names: [a,\n  b,\n  c]\nkpt_shape: [17.0, 2]\n",
    "kpt_shape: null\nnames: [a]\n",
    "kpt_shape: [17, 4]\nnames: [a]\n",
    "kpt_shape: [0, 3]\nnames: [a]\n",
    "kpt_shape: [true, 3]\nnames: [a]\n",
    "kpt_shape: [17]\nnames: [a]\n",
    "kpt_shape: 5\nnames: [a]\n",
    "- a\n- b\n",
    "just text\n",
    "names: x\n",
    "nc: 3\n",
    "names: {0: a}\nx: &anc 3\n",
    "names:\n  true: a\n  b: c\n",
    "names:\n  '3': a\n  '10': b\n  '2': c\n",
    "names:\n  z: a\n  b: c\n  m: d\n",
    "names:\n  0: a\n  x: b\n  1: c\n  y: d\n  2: e\n  w: f\n  3: g\n",
    "path: /data  # root\ntrain: images/train\nnames: &n\n  0: cat\n  1: dog\nnc: 2\n",
    "names: ['a', 'it''s', \"q\\\"d\"]\n",
    "---\nnames: [a]\nkpt_shape: [5, 2]\n",
]


def dump(names):
    s = io.StringIO()
    yaml.dump({"nc": len(names), "names": names}, s, sort_keys=False, default_flow_style=None)
    return s.getvalue()


def via_file(fn, text):
    with tempfile.TemporaryDirectory() as d:
        cwd = os.getcwd()
        os.chdir(d)
        try:
            Path("data.yaml").write_text(text, encoding="utf-8")
            try:
                return fn("data.yaml")
            except ValueError as e:
                return {"error": str(e)}
        finally:
            os.chdir(cwd)


def main():
    rng = random.Random(SEED)
    sets = [[], ["dog", "person"], ["class_%d" % i for i in range(30)], TRICKY]
    words = ["alpha", "bike", "car", "it's", "two words", "a  b", "x:y", "9", "no", "é"]
    for _ in range(40):
        sets.append([rng.choice(words) + str(rng.randint(0, 99)) * rng.randint(0, 2)
                     for _ in range(rng.randint(1, 40))])
    for _ in range(20):
        sets.append([" ".join(rng.choice(words) for _ in range(rng.randint(1, 6)))
                     for _ in range(rng.randint(1, 14))])
    rows = [("dump", {"names": n}, dump(n)) for n in sets]
    for d in DOCS:
        rows.append(("class_names", {"text": d}, via_file(yolo._extract_class_names, d)))
        rows.append(("pose_count", {"text": d}, via_file(yolo._extract_pose_value_count, d)))
    write_rows(ROOT / "tests/cases/dataset_yaml.txt", "gen_s3_yaml", SEED, rows)


main()

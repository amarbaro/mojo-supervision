"""Oracle rows for dataset/pyrandom.mojo + utils.mojo -> tests/cases/dataset_split.txt.

Ops:
  shuffle     {"seed","n"}                       -> permutation of range(n)  (random.Random(seed).shuffle)
  split       {"seed","n","ratio","shuffle"}     -> {"train":[...],"test":[...]} (train_test_split) | {"error"}
  merge_classes  {"lists": [[str]]}              -> merge_class_lists
  class_map   {"source","target"}                -> list[int] | {"error"}  (build_class_index_mapping)
  collision   {"paths","kind","by"}              -> null | {"error"}   (by: "name" or "stem_xml")
Run: .work/oracle-venv/bin/python -I tools/gen_s3_split.py     (seed 11)
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oracle import write_rows
from supervision.dataset import utils as u

ROOT = Path(__file__).resolve().parent.parent
SEED = 11


def attempt(fn):
    try:
        return fn()
    except ValueError as e:
        return {"error": str(e)}


def main():
    rng = random.Random(SEED)
    rows = []
    for seed in [0, 1, 2, 42, 123456789, 2**32 - 1, 2**32, 2**33 + 5]:
        for n in [0, 1, 2, 3, 7, 10, 33, 200]:
            l = list(range(n))
            random.Random(seed).shuffle(l)
            rows.append(("shuffle", {"seed": seed, "n": n}, l))
    for _ in range(80):
        n, seed = rng.randint(0, 60), rng.randint(0, 10**6)
        ratio = rng.choice([0.0, 1.0, 0.5, 0.8, 0.3, 0.123, rng.random()])
        shuf = rng.random() < 0.8
        tr, te = u.train_test_split(list(range(n)), ratio, seed, shuf)
        rows.append(("split", {"seed": seed, "n": n, "ratio": ratio, "shuffle": shuf},
                     {"train": tr, "test": te}))
    rows.append(("split", {"seed": 1, "n": 5, "ratio": 1.5, "shuffle": True},
                 attempt(lambda: u.train_test_split(list(range(5)), 1.5, 1, True))))
    rows.append(("split", {"seed": 1, "n": 5, "ratio": -0.1, "shuffle": True},
                 attempt(lambda: u.train_test_split(list(range(5)), -0.1, 1, True))))
    for lists in [[["dog", "person"], ["cat"]], [[], []], [["b", "a"], ["a", "é", "B", "Z"]],
                  [["x"]], [["a", "a"], ["a"]]]:
        rows.append(("merge_classes", {"lists": lists}, u.merge_class_lists(lists)))
    for src, tgt in [(["a", "b"], ["b", "a", "c"]), ([], ["a"]), (["z"], ["a"]), (["a", "a"], ["a"])]:
        rows.append(("class_map", {"source": src, "target": tgt},
                     attempt(lambda: [u.build_class_index_mapping(src, tgt)[i] for i in range(len(src))])))
    for paths in [["a/img.jpg", "b/img.jpg"], ["a/img.jpg", "b/img.png"], ["a/IMG.jpg", "b/img.jpg"],
                  ["a/x.jpg", "b/x.png"], ["d/it's.jpg", "e/it's.jpg"], []]:
        for by in ["name", "stem_xml"]:
            key = (lambda p: Path(p).name) if by == "name" else (lambda p: f"{Path(p).stem}.xml")
            rows.append(("collision", {"paths": paths, "kind": "image", "by": by},
                         attempt(lambda: u.check_no_basename_collisions(paths, key, "image"))))
    write_rows(ROOT / "tests/cases/dataset_split.txt", "gen_s3_split", SEED, rows)


main()

"""Differential check: PyRandom.shuffle_indices vs CPython random.Random(seed).shuffle.

Expected permutations come from the oracle venv (Python 3.14.9 supervision 0.32.0.dev0).
Run: mojo run -I src src/supervision/dataset/check_pyrandom.mojo
"""
from supervision.dataset.pyrandom import PyRandom


def check(seed: UInt64, n: Int, want: List[Int]) raises:
    var r = PyRandom(seed)
    var got = r.shuffle_indices(n)
    if got != want:
        raise Error("FAIL pyrandom seed=" + String(seed) + " n=" + String(n))


def main() raises:
    check(0, 10, [7, 8, 1, 5, 3, 4, 2, 0, 9, 6])
    check(42, 10, [7, 3, 2, 8, 5, 6, 9, 4, 0, 1])
    check(1, 7, [3, 6, 5, 2, 0, 4, 1])
    check(8589934597, 12, [7, 1, 0, 3, 10, 9, 6, 5, 8, 2, 4, 11])
    check(123456789, 20, [18, 16, 11, 10, 1, 4, 2, 0, 3, 13, 7, 15, 5, 8, 6, 19, 9, 12, 17, 14])
    print("PASS pyrandom 5/5")

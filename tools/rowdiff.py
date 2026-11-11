"""Compare binary output against a row file. Usage: rowdiff.py <rows.txt> <actual.jsonl>

Prints `PASS <file>:<line>` / `FAIL <file>:<line> <reason>` per row, then a SUMMARY.
Exit 0 only when every row passes and the row/output counts match.
"""
import json
import math
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable


def numeric(v):
    if v == "nan":
        return float("nan")
    return v


def same(actual, expected, tol):
    """Recursive compare. Floats: |a-b| <= abs + rel*|b| (default 1e-6 both)."""
    a, e = numeric(actual), numeric(expected)
    if isinstance(e, bool) or isinstance(a, bool):
        return a == e
    if isinstance(e, (int, float)) and isinstance(a, (int, float)):
        if math.isnan(e) or math.isnan(a):
            return math.isnan(e) and math.isnan(a)
        t_abs = tol.get("abs", 1e-6)
        t_rel = tol.get("rel", 1e-6)
        return abs(a - e) <= t_abs + t_rel * abs(e)
    if isinstance(e, list) and isinstance(a, list):
        return len(a) == len(e) and all(same(x, y, tol) for x, y in zip(a, e))
    if isinstance(e, dict) and isinstance(a, dict):
        return a.keys() == e.keys() and all(same(a[k], e[k], tol) for k in e)
    return a == e


def check_pixels(actual, expected, args):
    ref = os.path.join(ROOT, expected["ref"])
    out = actual.get("out") if isinstance(actual, dict) else None
    if out is None:
        return False, "actual has no 'out' path"
    tol = args.get("tol", {})
    max_abs = tol.get("max_abs", 0)
    max_pct = tol.get("max_pct", 0)
    if ref.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
        cmd = [PY, os.path.join(HERE, "pixdiff.py"), ref, out,
               "--max-abs", str(max_abs), "--max-pct", str(max_pct)]
    else:
        cmd = ["cmp", "-s", ref, out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    detail = (r.stdout + r.stderr).strip().splitlines()
    return r.returncode == 0, detail[-1] if detail else f"exit {r.returncode}"


def main():
    rows_path, actual_path = sys.argv[1], sys.argv[2]
    rows, lines = [], []
    with open(rows_path, encoding="utf-8") as f:
        for n, text in enumerate(f, 1):
            if not text.strip() or text.startswith("#"):
                continue
            rows.append((n, text.rstrip("\n")))
    with open(actual_path, encoding="utf-8") as f:
        actual = [ln.rstrip("\n") for ln in f if ln.strip()]
    rel = os.path.relpath(rows_path, ROOT)
    if len(actual) != len(rows):
        print(f"FAIL {rel}: binary printed {len(actual)} lines for {len(rows)} rows")
        return 1
    failed = 0
    for (n, text), got in zip(rows, actual):
        op, args_s, exp_s = text.split("\t")
        args, expected = json.loads(args_s), json.loads(exp_s)
        try:
            a = json.loads(got)
        except json.JSONDecodeError as ex:
            failed += 1
            print(f"FAIL {rel}:{n} {op}: output is not JSON ({ex})")
            continue
        if isinstance(expected, dict) and "ref" in expected:
            ok, why = check_pixels(a, expected, args)
        else:
            ok = same(a, expected, args.get("tol", {}) if isinstance(args, dict) else {})
            why = f"expected {exp_s[:200]} got {got[:200]}"
        if ok:
            print(f"PASS {rel}:{n}")
        else:
            failed += 1
            print(f"FAIL {rel}:{n} {op}: {why}")
    print(f"SUMMARY {rel}: {len(rows)} rows, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

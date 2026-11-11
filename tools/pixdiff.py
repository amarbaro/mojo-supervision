"""Pixel diff. Usage: pixdiff.py <ref.png> <actual.png> [--max-abs N] [--max-pct P]

Prints one JSON line: max_abs, diff_pct (percent of pixels differing in any channel).
Exit 0 when max_abs <= N and diff_pct <= P (both default 0 = byte-exact pixels).
"""
import argparse
import json
import sys

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("actual")
    ap.add_argument("--max-abs", type=float, default=0.0)
    ap.add_argument("--max-pct", type=float, default=0.0)
    a = ap.parse_args()
    ref = cv2.imread(a.ref, cv2.IMREAD_UNCHANGED)
    act = cv2.imread(a.actual, cv2.IMREAD_UNCHANGED)
    if ref is None or act is None:
        print(f"unreadable image: ref={ref is not None} actual={act is not None}")
        return 1
    if ref.shape != act.shape:
        print(f"shape mismatch: ref {ref.shape} actual {act.shape}")
        return 1
    d = np.abs(ref.astype(np.int32) - act.astype(np.int32))
    max_abs = int(d.max()) if d.size else 0
    per_px = d.reshape(d.shape[0], d.shape[1], -1).max(axis=2) if d.ndim == 3 else d
    diff_pct = 100.0 * float(np.count_nonzero(per_px)) / per_px.size if per_px.size else 0.0
    print(json.dumps({"max_abs": max_abs, "diff_pct": round(diff_pct, 6)}))
    ok = max_abs <= a.max_abs and diff_pct <= a.max_pct
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

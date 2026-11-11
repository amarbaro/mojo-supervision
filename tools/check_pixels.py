#!/usr/bin/env python3
"""BOSS GATE for the pixel rows: rows parse, and regenerating gives sha1-identical files.

Checks tests/cases/pixels_*.txt: line 1 is `# gen: tools/gen_pixels.py seed=<n>`, every
other non-comment line has 3 tab columns, the last two are JSON, args carry input and tol.
Then hashes tests/fixtures/pixels/* and tests/cases/pixels_*.txt, reruns the generator, and
hashes again. Any difference is FAIL. Exit non-zero on failure.

    .work/oracle-venv/bin/python tools/check_pixels.py
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CASES = sorted((ROOT / "tests" / "cases").glob("pixels_*.txt"))
FIXTURES = sorted((ROOT / "tests" / "fixtures" / "pixels").glob("*.png"))


def fail(step: str, reason: str) -> None:
    print(f"FAIL {step}: {reason}")
    sys.exit(1)


def parse_rows() -> int:
    total = 0
    for path in CASES:
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines or not lines[0].startswith("# gen: tools/gen_pixels.py seed="):
            fail("parse", f"{path.name} line 1 is not the # gen: header")
        for n, line in enumerate(lines[1:], start=2):
            if not line or line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) != 3:
                fail("parse", f"{path.name}:{n} has {len(cols)} tab columns, want 3")
            try:
                args = json.loads(cols[1])
                expected = json.loads(cols[2])
            except json.JSONDecodeError as exc:
                fail("parse", f"{path.name}:{n} bad JSON: {exc}")
            if "tol" not in args:
                fail("parse", f"{path.name}:{n} args lack tol")
            if "ref" in expected:
                if "input" not in args:
                    fail("parse", f"{path.name}:{n} pixel row lacks input")
            elif "value" not in expected:
                fail("parse", f"{path.name}:{n} expected lacks ref or value")
            total += 1
    return total


def digest() -> dict:
    out = {}
    for p in CASES + FIXTURES:
        out[str(p.relative_to(ROOT))] = hashlib.sha1(p.read_bytes()).hexdigest()
    return out


def main() -> int:
    rows = parse_rows()
    print(f"PASS parse: {rows} rows in {len(CASES)} files")
    before = digest()
    run = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "gen_pixels.py")],
        cwd=ROOT, capture_output=True, text=True,
    )
    if run.returncode != 0:
        fail("regen", f"gen_pixels.py exit {run.returncode}: {run.stderr.strip()}")
    after = digest()
    if before.keys() != after.keys():
        fail("regen", "file set changed on regeneration")
    diffs = [k for k in before if before[k] != after[k]]
    if diffs:
        fail("regen", f"{len(diffs)} files differ: {diffs[:5]}")
    print(f"PASS regen: {len(before)} files sha1-identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())

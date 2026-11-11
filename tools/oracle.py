"""Shared helpers for oracle generators (tools/gen_*.py).

Row format: see tests/README.md. One case per line:
    <op>\t<args JSON>\t<expected JSON>
"""
import json
import math
import sys


def check_oracle():
    """Import the Python oracle; return its version. Fails loudly if missing."""
    import supervision

    return supervision.__version__


def _clean(value):
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if hasattr(value, "tolist"):  # numpy arrays and scalars
        return _clean(value.tolist())
    return value


def row(op, args, expected):
    """Format one row line (no trailing newline)."""
    return "\t".join(
        [
            op,
            json.dumps(_clean(args), separators=(",", ":"), allow_nan=False),
            json.dumps(_clean(expected), separators=(",", ":"), allow_nan=False),
        ]
    )


def write_rows(path, gen, seed, rows):
    """Write a row file: line 1 is the gen header, then one row per line."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# gen: tools/{gen}.py seed={seed}\n")
        for op, args, expected in rows:
            f.write(row(op, args, expected) + "\n")
    print(f"wrote {len(rows)} rows to {path}", file=sys.stderr)

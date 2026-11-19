"""Check swadesh's emulation of the builtin ``sum()`` against this interpreter.

For random lists of floats and ints (including the sacreBLEU floor value
``-9999999999``), compare the builtin ``sum(items)`` with the Rust
``py_sum`` emulation in all three modes (``naive``, ``neumaier``, ``neumaier_ints``). Results are compared
by type and IEEE-754 bit pattern, never by ``==`` alone.

Usage: python tools/check_sum_semantics.py [--cases N] [--seed S]
"""
import argparse
import math
import random
import struct
import sys

from swadesh import _swadesh

FLOOR = -9999999999
#: Values that make compensated and naive summation disagree.
HARD = [1e16, -1e16, 1.0, -1.0, 0.1, 0.2, 0.3, 3.0, 1e-300, 2.5, -7.25]


def same(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, int):
        return a == b
    return struct.pack("<d", a) == struct.pack("<d", b) or (math.isnan(a) and math.isnan(b))


def random_items(rnd):
    n = rnd.randint(0, 8)
    items = []
    for _ in range(n):
        r = rnd.random()
        if r < 0.15:
            items.append(FLOOR)
        elif r < 0.25:
            items.append(rnd.randint(-5, 5))
        elif r < 0.55:
            items.append(rnd.choice(HARD))
        else:
            # log() of a precision in (0, 100]: the shape sacreBLEU actually sums.
            items.append(math.log(rnd.uniform(1e-3, 100.0)))
    return items


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    rnd = random.Random(args.seed)
    mismatches = {"naive": 0, "neumaier": 0, "neumaier_ints": 0, "selected": 0}
    first = {}
    for _ in range(args.cases):
        items = random_items(rnd)
        ref = sum(items)
        for mode in ("naive", "neumaier", "neumaier_ints"):
            got = _swadesh._py_sum(items, mode)
            if not same(ref, got):
                mismatches[mode] += 1
                first.setdefault(mode, (items, ref, got))
        if not same(ref, _swadesh._py_sum(items)):
            mismatches["selected"] += 1
            first.setdefault("selected", (items, ref, _swadesh._py_sum(items)))

    print(f"python      : {sys.version.split()[0]} ({sys.implementation.name})")
    print(f"selected    : {_swadesh._sum_mode()}")
    print(f"cases       : {args.cases}")
    print(f"mismatch naive    : {mismatches['naive']}")
    print(f"mismatch neumaier : {mismatches['neumaier']}")
    print(f"mismatch neumaier_ints : {mismatches['neumaier_ints']}")
    print(f"mismatch selected : {mismatches['selected']}")
    for mode, (items, ref, got) in first.items():
        print(f"  first {mode} mismatch: items={items!r} builtin={ref!r} swadesh={got!r}")
    return 0 if mismatches["selected"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

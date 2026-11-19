#!/usr/bin/env python3
"""Generate the numpy/CPython reference fixtures for swadesh-core's nprng tests.

Run with the venv that has numpy and sacrebleu installed:

    .venv/bin/python tests/gen_rng_fixtures.py

Writes plain-text fixtures to crates/swadesh-core/tests/fixtures/. Each file is
read by crates/swadesh-core/tests/nprng.rs with `include_str!`. Large outputs
(millions of draws) are stored as an order-sensitive 64-bit hash of the flat
C-order array instead of the full values. The hash is `stream_hash` below and
is reproduced exactly in Rust.
"""

import os
import random
import struct
import statistics
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "crates", "swadesh-core", "tests", "fixtures")
GOLD = 0x9E3779B97F4A7C15

# ---- helpers ---------------------------------------------------------------


def stream_hash(a) -> str:
    """Order-sensitive hash of a flat integer array (values taken as u64).

    h = sum_i splitmix64_finalizer(v_i + (i+1) * GOLD), wrapping u64.
    """
    v = np.asarray(a).ravel().astype(np.int64).view(np.uint64)
    i = np.arange(1, v.size + 1, dtype=np.uint64)
    x = v + i * np.uint64(GOLD)
    x ^= x >> np.uint64(30)
    x *= np.uint64(0xBF58476D1CE4E5B9)
    x ^= x >> np.uint64(27)
    x *= np.uint64(0x94D049BB133111EB)
    x ^= x >> np.uint64(31)
    return format(int(np.sum(x, dtype=np.uint64)), "016x")


def f64_hex(x: float) -> str:
    return format(struct.unpack(">Q", struct.pack(">d", float(x)))[0], "016x")


def f32_hex(x) -> str:
    return format(struct.unpack(">I", struct.pack(">f", float(x)))[0], "08x")


def write(name: str, lines) -> None:
    path = os.path.join(OUT, name)
    with open(path, "w") as f:
        for line in lines:
            f.write(line + "\n")
    print(f"wrote {name}: {len(lines)} lines, {os.path.getsize(path)} bytes")


# ---- 1. PCG64 raw stream, and buffered next_uint32 ------------------------

SEEDS_RAW = [12345, 0, 1, 2**63]


def gen_pcg64_raw():
    lines = []
    for seed in SEEDS_RAW:
        raw = np.random.default_rng(seed).bit_generator.random_raw(1000)
        lines.append(f"{seed} " + " ".join(str(int(v)) for v in raw))
    write("pcg64_raw.txt", lines)


def gen_u32_raw():
    # integers(0, 2**32) has range 0xFFFFFFFF, so numpy returns next_uint32 raw.
    # This checks the buffered 32-bit half-word order.
    lines = []
    for seed in [12345, 0]:
        vals = np.random.default_rng(seed).integers(0, 2**32, size=1000)
        lines.append(f"{seed} " + " ".join(str(int(v)) for v in vals))
    write("u32_raw.txt", lines)


# ---- 2. bounded integers and choice ----------------------------------------


def gen_integers():
    lines = []
    cases = [
        # (low, high, n, seed): exercises rng==0, 32-bit Lemire, raw 32-bit, 64-bit Lemire
        (0, 1, 5, 12345),
        (0, 7, 1000, 12345),
        (-5, 7, 1000, 0),
        (0, 2**32, 1000, 1),
        (0, 2**40, 1000, 12345),
        (-(2**62), 2**62, 1000, 0),
    ]
    for low, high, n, seed in cases:
        vals = np.random.default_rng(seed).integers(low, high, size=n)
        lines.append(f"int {low} {high} {n} {seed} full " + " ".join(str(int(v)) for v in vals))
    # Large draws, stored as hashes.
    for low, high, n, seed in [(0, 2**40, 2_000_000, 12345), (0, 1000003, 2_000_000, 7)]:
        vals = np.random.default_rng(seed).integers(low, high, size=n)
        lines.append(f"int {low} {high} {n} {seed} hash {stream_hash(vals)}")
    write("integers.txt", lines)


def gen_choice():
    lines = []
    for n in [1, 2, 3, 7, 1000, 2037]:
        for seed in [12345, 0, 1]:
            vals = np.random.default_rng(seed).choice(n, size=(5, n), replace=True)
            lines.append(f"choice {n} 5 {n} {seed} full " + " ".join(str(int(v)) for v in vals.ravel()))
    for n, rows, cols, seed in [(100000, 5, 100000, 12345), (2**31 + 5, 3, 1_000_000, 12345), (2**31 + 5, 3, 1_000_000, 0)]:
        vals = np.random.default_rng(seed).choice(n, size=(rows, cols), replace=True)
        lines.append(f"choice {n} {rows} {cols} {seed} hash {stream_hash(vals)}")
    write("choice.txt", lines)


def gen_bool():
    lines = []
    for rows, cols, seed in [(50, 7, 12345), (50, 7, 0), (5, 2037, 12345), (5, 2037, 1)]:
        vals = np.random.default_rng(seed).integers(2, size=(rows, cols), dtype=bool)
        lines.append(f"bool {rows} {cols} {seed} full " + " ".join("1" if v else "0" for v in vals.ravel()))
    vals = np.random.default_rng(12345).integers(2, size=(1000, 1000), dtype=bool)
    lines.append(f"bool 1000 1000 12345 hash {stream_hash(vals.astype(np.int64))}")
    write("bool.txt", lines)


# ---- 3. reductions ----------------------------------------------------------


def random_floats(rng: np.random.Generator, n: int) -> np.ndarray:
    # Mixed magnitudes, so rounding differences show up.
    return rng.standard_normal(n) * 10.0 ** rng.uniform(-3, 6, size=n)


MEAN_LENGTHS = list(range(1, 141)) + [255, 256, 257, 1000, 1023, 1024, 1025, 4095, 4096, 4097, 9999, 10000]


def gen_mean():
    rng = np.random.default_rng(777)
    lines = []
    for n in MEAN_LENGTHS:
        arr = random_floats(rng, n)
        expected = float(np.mean(arr))
        lines.append(f"{f64_hex(expected)} " + " ".join(f64_hex(x) for x in arr))
    write("mean.txt", lines)


def gen_pairwise_f32():
    rng = np.random.default_rng(778)
    lines = []
    for n in list(range(1, 141)) + [255, 256, 257, 1000, 2000]:
        vals = rng.integers(0, 2**26, size=n).astype(np.float32)
        expected = np.float32(vals.reshape(n, 1).sum(0)[0])
        lines.append(f"{f32_hex(expected)} " + " ".join(f32_hex(x) for x in vals))
    write("pairwise_f32.txt", lines)


def gen_statistics_mean():
    rnd = random.Random(4242)
    lines = []
    specials = [
        [1.0],
        [1e308, 1e308, -1e308],
        [5e-324, 5e-324, 5e-324],
        [0.1, 0.2, 0.3],
        [1.0, 2.0, 4.0, 8.0, 16.0],
        [-0.0, -0.0],
        [1.0, float("inf")],
        [float("inf"), float("-inf")],
        [2.5, float("nan")],
    ]
    cases = specials[:]
    for n in list(range(1, 31)) + [100, 1000]:
        cases.append([rnd.gauss(0, 1) * 10 ** rnd.uniform(-8, 8) for _ in range(n)])
    for data in cases:
        expected = statistics.mean(data)
        lines.append(f"{f64_hex(expected)} " + " ".join(f64_hex(x) for x in data))
    write("statistics_mean.txt", lines)


def gen_f32_rows():
    lines = []
    # Matrices of integer counts above 2**24, so float32 sums round.
    for rows, width, seed in [(2000, 10, 1), (300, 24, 2), (7, 3, 3), (50, 1, 4)]:
        mat = np.random.default_rng(seed).integers(0, 2**26, size=(rows, width))
        stats_np = np.array(mat.tolist(), dtype="float32")
        lines.append(f"mat {rows} {width} " + " ".join(str(int(v)) for v in mat.ravel()))
        for seed_sel, n_samples in [(12345, 20), (0, 20), (9, 5)]:
            idxs = np.random.default_rng(seed_sel).choice(rows, size=(n_samples, rows), replace=True)
            sums = [np.asarray(_s.sum(0), dtype=np.float32) for _s in stats_np[idxs]]
            flat = " ".join(f32_hex(x) for s in sums for x in s)
            lines.append(f"sums {seed_sel} {n_samples} {flat}")
    write("f32_rows.txt", lines)


def main():
    print("numpy", np.__version__, "python", sys.version.split()[0])
    os.makedirs(OUT, exist_ok=True)
    gen_pcg64_raw()
    gen_u32_raw()
    gen_integers()
    gen_choice()
    gen_bool()
    gen_mean()
    gen_pairwise_f32()
    gen_statistics_mean()
    gen_f32_rows()


if __name__ == "__main__":
    main()

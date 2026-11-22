# Contributing to swadesh

swadesh is a drop-in replacement for sacreBLEU 2.6.0. Most of the rules below
follow from that.

## Exactness rules

1. **Bit-exact.** Every statistic and score must be `==` to sacreBLEU 2.6.0 on the
   same interpreter. This includes sacreBLEU's quirks. Reproduce a quirk, do not
   correct it, and explain it in [docs/how-its-exact.md](docs/how-its-exact.md).
2. **Differential tests are the oracle.** A change to tokenization, n-gram
   statistics, score arithmetic, sum emulation or bootstrap needs a test that
   compares sacreBLEU and swadesh with `==` (or with `repr` for floats). Tolerances
   are not acceptable.
3. **Drop-in.** Keep the classes, constructor arguments, `Score` objects,
   signatures and CLI flags of sacreBLEU. A new option needs the same name and
   default as sacreBLEU's.
4. **Native tokenizers are opt-in.** Add a tokenizer to `NATIVE_TOKENIZERS`
   (Python) and `Tokenizer` (Rust) only after the differential tests cover it,
   including the adversarial alphabet.
5. **Measured claims only.** Speedups in the README and docs come from `bench/`
   runs on a named machine. Record the machine, the Python and NumPy versions,
   and the thread count. Truncate speedups to one decimal place.
6. **Generated tables are generated.** Do not edit
   `crates/swadesh-core/src/tables.rs` by hand. Regenerate it (below).
7. **New CPython versions.** Run `tools/check_sum_semantics.py` on the new
   interpreter before trusting its `SumMode`. Add or change a mode in
   `crates/swadesh-core/src/score.rs` only with evidence from that script.

## Build

You need a Rust toolchain (CI pins 1.90.0), Python 3.10 or newer, and maturin
1.7 or newer.

```bash
python -m venv .venv
. .venv/bin/activate
pip install "maturin>=1.7,<2" "sacrebleu==2.6.0" "numpy>=2,<3" pytest hypothesis
maturin develop --release                # builds swadesh._swadesh into the venv
cargo build --release -p swadesh-cli     # standalone binary: target/release/swadesh
```

`maturin develop` uses `crates/swadesh-py` (see `pyproject.toml`). It builds an
abi3 extension, so one build serves Python 3.10 and newer.

For a wheel, use `maturin build --release --out dist`, as the CI workflows do.

## Tests

```bash
cargo test -p swadesh-core
cargo clippy -p swadesh-core -- -D warnings

python -m pytest tests/test_differential.py -v     # hypothesis differential suite
python tools/check_sum_semantics.py                 # emulated sum() against builtin sum()

python -m pytest tests/test_cli.py -v               # needs target/release/swadesh and WMT22 data
```

`tests/test_cli.py` needs WMT22 files in `bench/data`. Fetch them with
`bench/get_data.sh`. The script downloads through sacreBLEU; its comment puts the
cache at about 400 MB. Set `SWADESH_DATA` to use another directory.

Run the differential suite on each Python version and NumPy major version that
CI covers (see `.github/workflows/ci.yml`). A change that passes on one interpreter
can still fail on another, because of `sum()` and Unicode behaviour.

## Benchmarks

```bash
python bench/bench.py --lines 20000 --json bench/results/<machine>.json
python bench/bench.py --lines 100000 --quick
```

`bench/bench.py` checks each timed swadesh result against sacreBLEU, and prints
`NO!` if they differ. Do not publish a timing from a run that printed `NO!`.
Summaries go in `bench/results/SUMMARY.md`, with the machine's CPU, OS, Python,
sacreBLEU and NumPy versions, and the free disk before the run.

## Regenerating the Unicode tables

`crates/swadesh-core/src/tables.rs` holds the character-class ranges that the
tokenizers and `str.split()` emulation use. They come from the installed Python and
`regex` module:

```bash
python tools/gen_tables.py > crates/swadesh-core/src/tables.rs
```

The file header records the Python and `regex` versions. Regenerate when either
changes, then run the full differential suite and look at the diff. The
checked-in tables reproduce byte for byte on CPython 3.11.14 with `regex`
2026.11.19.

## Licensing

swadesh is dual-licensed: AGPL-3.0-or-later for the community (see `LICENSE`) and a paid commercial
licence for uses the AGPL does not cover (see `LICENSE.commercial`). `docs/licensing.md` says what
that means in practice.

**Before your first pull request is merged you must agree to the Contributor Licence Agreement**
(`CLA.md`). It lets the copyright holder license your contribution under both licences; without it a
single outside contribution would block the commercial side permanently. Add this line to your first
pull request:

    I agree to the swadesh Contributor Licence Agreement.

Five Python modules still import sacreBLEU and remain under its Apache-2.0 notice as well:
`__init__.py`, `verify.py`, `metrics.py`, `significance.py`, `cli.py`. `NOTICE` names them.

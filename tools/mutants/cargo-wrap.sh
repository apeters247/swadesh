#!/usr/bin/env bash
# CARGO wrapper for cargo-mutants (used via CARGO=<this script>).
#
# cargo-mutants invokes "$CARGO build/test ..." inside each scratch copy of the tree.
# For `cargo test`, this wrapper additionally runs the Python differential subset
# against the mutated core, so a mutant counts as caught if either the Rust tests
# or the Python tests fail (or hang past cargo-mutants' timeout).
#
# Requires: VIRTUAL_ENV slots under $MUTANTS_SLOTS/venv.N (see docs/mutation-report.md).
set -uo pipefail

REAL_CARGO=${REAL_CARGO:-$HOME/.cargo/bin/cargo}
SLOTS=${MUTANTS_SLOTS:-$HOME/mutants-slots}
NSLOTS=${MUTANTS_NSLOTS:-8}

# Release-profile LTO dominates build time; mutation detection does not depend on it.
# Applied to every build in the run (baseline and mutants alike) for comparability.
export CARGO_PROFILE_RELEASE_LTO=off
export CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16

if [ "${1:-}" != "test" ]; then
    exec "$REAL_CARGO" "$@"
fi
for a in "$@"; do
    [ "$a" = "--no-run" ] && exec "$REAL_CARGO" "$@"
done

# 1. Rust tests (cargo-mutants passes -p swadesh-core --release via .cargo/mutants.toml).
"$REAL_CARGO" "$@" || exit $?

# 2. Python differential subset, in a slot venv not used by another concurrent mutant.
slot=""
for i in $(seq 0 $((NSLOTS - 1))); do
    exec 9>"$SLOTS/lock.$i"
    if flock -n 9; then slot=$i; break; fi
    exec 9>&-
done
if [ -z "$slot" ]; then echo "mut-cargo: no free venv slot" >&2; exit 2; fi

export VIRTUAL_ENV="$SLOTS/venv.$slot"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1

maturin develop --release -q || { echo "mut-cargo: maturin develop failed" >&2; exit 3; }
python -m pytest -q -x -p no:cacheprovider \
    tests/test_differential.py tests/test_ter.py -k "not long"

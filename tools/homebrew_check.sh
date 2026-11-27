#!/usr/bin/env bash
# Validate the Homebrew tap for swadesh. Run this on a Mac with Homebrew installed.
#
#   tools/homebrew_check.sh
#
# It taps, installs (there are no bottles, so this compiles from source), runs brew's own checks, then
# proves the installed binary works. `--build-from-source` is deliberately not passed: Homebrew 6 warns
# "building from source is not supported!" for it, while a formula with no bottle builds from source
# anyway. Every step is reported; the exit code is non-zero if any step failed, so it can
# be pasted into a release log as-is.
set -uo pipefail

TAP=apeters247/tap
FORMULA=swadesh
fail=0

command -v brew >/dev/null 2>&1 || { echo "Homebrew is not installed: https://brew.sh"; exit 2; }

run() {
  echo
  echo "=== $* ==="
  if "$@"; then
    echo "PASS: $*"
  else
    echo "FAIL: $*"
    fail=1
  fi
}

echo "host: $(sw_vers -productVersion 2>/dev/null) $(uname -m), brew $(brew --version | head -1)"

run brew tap "$TAP"
run brew install "$TAP/$FORMULA"
run brew test "$FORMULA"
run brew audit --new --strict "$FORMULA"
run brew style "$TAP"

echo
echo "=== the installed binary, as a user would run it ==="
BIN="$(brew --prefix)/bin/$FORMULA"
"$BIN" --version
work="$(mktemp -d)"
printf 'the cat sat on the mat\n' > "$work/ref.txt"
printf 'the cat is on the mat\n' > "$work/hyp.txt"
"$BIN" "$work/ref.txt" -i "$work/hyp.txt" -m bleu chrf chrf ter | head -20
rm -rf "$work"

echo
echo "keg: $(brew --prefix "$FORMULA" 2>/dev/null)"
if [ "$fail" -eq 0 ]; then
  echo "ALL HOMEBREW CHECKS PASSED"
else
  echo "SOME HOMEBREW CHECKS FAILED"
fi
exit "$fail"
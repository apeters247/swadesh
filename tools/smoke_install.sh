#!/usr/bin/env bash
# Smoke-test the release channels from the working tree, before any tag exists.
#
#   tools/smoke_install.sh            # wheel, then the sdist
#   tools/smoke_install.sh wheel      # wheel only (fast)
#
# It builds with maturin into a scratch directory, installs into a throwaway venv with `uv`, and
# checks the things a user does first: import the package as sacrebleu, score something, run the
# console script, run the standalone binary, run `swadesh.verify`, and import the MCP server.
#
# Run it from the repository root, on a machine with the Rust toolchain and network access.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHAT="${1:-all}"
WORK="$(mktemp -d)"
DIST="$WORK/dist"
VENV="$WORK/venv"
mkdir -p "$DIST"

export PATH="$HOME/.cargo/bin:$PATH"
MATURIN="$(command -v maturin || echo "$HOME/venv-ter/bin/maturin")"
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
[ -x "$MATURIN" ] || { echo "maturin not found"; exit 1; }
[ -x "$UV" ] || { echo "uv not found"; exit 1; }

REF="$WORK/ref.txt"
HYP="$WORK/hyp.txt"
printf 'the cat sat on the mat\nhello world here\nshe sells sea shells\n' > "$REF"
printf 'the cat sat on a mat\nhello there world\nshe sells sea shells\n' > "$HYP"

smoke() {  # $1 = a human name for the thing under test
  local what="$1"
  echo "--- $what ---"
  cd "$ROOT"
  export PATH="$VENV/bin:$PATH"   # the venv's console scripts must be resolvable
  "$VENV/bin/python" - "$REF" "$HYP" <<'PY'
import subprocess, sys
import swadesh as sacrebleu
score = sacrebleu.BLEU().corpus_score(
    ["the cat sat on a mat", "hello there world", "she sells sea shells"],
    [["the cat sat on the mat", "hello world here", "she sells sea shells"]]).score
assert score > 0, score
print(f"  import swadesh as sacrebleu: BLEU {score:.4f}")
assert sacrebleu.SACREBLEU_COMPAT_VERSION == "2.6.0", sacrebleu.SACREBLEU_COMPAT_VERSION
import swadesh
swadesh.patch()                       # the Hugging Face / fairseq path
from sacrebleu.metrics import BLEU as SB_BLEU
assert abs(SB_BLEU().corpus_score(
    ["the cat sat on a mat"], [["the cat sat on the mat"]]).score
    - swadesh.BLEU().corpus_score(
    ["the cat sat on a mat"], [["the cat sat on the mat"]]).score) == 0
print("  swadesh.patch(): patched sacrebleu agrees with swadesh")
rc = subprocess.run(["swadesh", "--version"], capture_output=True, text=True)
assert rc.returncode == 0 and rc.stdout.startswith("swadesh "), rc
print(f"  console script: {rc.stdout.strip()}")
rc = subprocess.run(["swadesh", sys.argv[1], "-i", sys.argv[2], "-m", "bleu", "chrf", "ter", "-b"],
                    capture_output=True, text=True)
assert rc.returncode == 0, rc.stderr
print(f"  console script scores: {rc.stdout.strip()}")
rc = subprocess.run([sys.executable, "-m", "swadesh.verify", "-r", sys.argv[1], "-i", sys.argv[2]],
                    capture_output=True, text=True)
assert rc.returncode == 0 and "identical" in rc.stdout, (rc.returncode, rc.stdout[-200:], rc.stderr[-200:])
print("  swadesh.verify: identical to sacreBLEU")
import swadesh.mcp_server  # noqa: F401  - the MCP entry point must import
print("  MCP server imports")
PY
}

build_wheel() {
  echo "=== building the wheel ==="
  cd "$ROOT"
  "$MATURIN" build --release --out "$DIST" ${MATURIN_COMPAT:---compatibility linux} >/dev/null
  ls "$DIST"/*.whl
  "$UV" venv "$VENV" >/dev/null
  "$UV" pip install --python "$VENV/bin/python" "$DIST"/*.whl numpy sacrebleu==2.6.0 mcp >/dev/null
  smoke "wheel"
}

build_sdist() {
  echo "=== building the sdist ==="
  cd "$ROOT"
  "$MATURIN" sdist --out "$DIST" >/dev/null
  ls "$DIST"/*.tar.gz
  "$UV" venv "$VENV.sdist" >/dev/null
  "$UV" pip install --python "$VENV.sdist/bin/python" "$DIST"/*.tar.gz numpy sacrebleu==2.6.0 mcp
  VENV="$VENV.sdist" smoke "sdist (built from source)"
}

case "$WHAT" in
  wheel) build_wheel ;;
  sdist) build_sdist ;;
  all) build_wheel; build_sdist ;;
  *) echo "usage: $0 [wheel|sdist|all]"; exit 2 ;;
esac
echo "smoke install: OK"
rm -rf "$WORK"
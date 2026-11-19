#!/usr/bin/env bash
# Download WMT22 references used by bench/bench.py (≈400MB sacrebleu cache).
set -euo pipefail
cd "$(dirname "$0")"; mkdir -p data
export SACREBLEU="$(pwd)/.sacrebleu"
for lp in en-de de-en en-zh zh-en en-ja en-ru cs-en; do
  sacrebleu -t wmt22 -l $lp --echo ref > data/wmt22.$lp.ref
  sacrebleu -t wmt22 -l $lp --echo src > data/wmt22.$lp.src
done

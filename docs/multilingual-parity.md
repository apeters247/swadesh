# Multilingual parity: ten language pairs

swadesh is checked exhaustively on English-German, because that is where the published baselines are.
This page records a second axis: **ten more language pairs**, weighted towards the ones metric
libraries get exercised on least - Gujarati, Kazakh, Lithuanian, Arabic in both directions, Japanese
and Korean with a non-mecab tokenizer, Turkish, and one pair with no English on either side.

Every comparison below is exact. Not "within a float": `==`.

## What was compared

For each pair, the references come from the oracle's own cache (`sacrebleu -t SET -l PAIR --echo ref`),
and the hypothesis is a deterministic perturbation of them - drop a middle word, swap two words, or
suffix the last word, on every third line - so the scores are non-trivial but the run is exactly
reproducible.

Then three separate comparisons:

1. **CLI corpus**: `-i hyp -m bleu chrf ter -f text`, byte for byte on **stdout, stderr and exit code**.
   stderr matters: this is where the tokenizer advisories live.
2. **CLI sentence level**: `-sl -m bleu` and `-sl -m chrf`, byte for byte.
3. **Python API**: corpus BLEU, chrF++ and TER with `==`, and the first 400 sentence scores with `==`.

Machine: AMD Ryzen 9 7900X, Linux x86-64, CPython 3.11.14, sacreBLEU 2.6.0, swadesh at `master`.
Japanese and Korean pass `-tok intl` on both sides because their default tokenizers are mecab-based
and mecab is not installed; every other pair uses the default tokenizer for its language.

## Results

| pair | set | segments | why it is here | BLEU | chrF++ | TER | CLI bytes | sentence diffs |
|---|---|---|---:|---:|---:|---:|---|---|
| en-gu | `wmt19/dev` | 1998 | Gujarati, Indic script | 90.450189 | 94.131779 | 5.700017 | identical | 0 / 400 |
| kk-en | `wmt19/dev` | 2066 | Kazakh, Cyrillic, agglutinative | 92.804065 | 95.654476 | 4.465470 | identical | 0 / 400 |
| lt-en | `wmt19/dev` | 2000 | Lithuanian, rich morphology | 92.527146 | 95.337772 | 4.607228 | identical | 0 / 400 |
| en-ar | `iwslt17` | 1460 | Arabic, right-to-left | 87.694881 | 92.518662 | 7.061607 | identical | 0 / 400 |
| ar-en | `iwslt17` | 1460 | Arabic to English | 90.425271 | 93.874638 | 5.950690 | identical | 0 / 400 |
| en-ja | `iwslt17` | 1452 | Japanese, no word spaces, `intl` | 67.195800 | 87.004139 | 16.482911 | identical | 0 / 400 |
| ja-en | `iwslt17` | 1452 | Japanese to English, `intl` | 90.550464 | 93.854995 | 5.960376 | identical | 0 / 400 |
| ko-en | `iwslt17` | 1429 | Korean to English, `intl` | 90.648839 | 93.906078 | 5.911330 | identical | 0 / 400 |
| de-fr | `wmt22` | 1984 | German to French, no English | 90.487422 | 94.406004 | 5.735887 | identical | 0 / 400 |
| en-tr | `wmt16` | 3000 | Turkish, agglutinative | 89.006015 | 93.926234 | 6.665912 | identical | 0 / 400 |

**10 of 10 pairs: CLI corpus identical on all three metrics, CLI sentence level identical for both
metrics, corpus scores `==`, 0 of 4,000 sentence scores differing** (400 per pair, two metrics).

## Two stderr advisories this found, and fixed

The scores were identical from the start; the **advisories** were not. sacreBLEU prints three
tokenizer-related lines that swadesh was missing, and they now match byte for byte:

- `Consider using the 'ja-mecab' or 'spm' tokenizer for Japanese.` - when BLEU is computed and the
  **target** of `-l` is `ja`; `-m chrf` alone prints nothing.
- `That's 100 lines that end in a tokenized period ('.')` plus **two** following lines - the count is
  fixed at 100 and the threshold is 100 hypothesis lines ending in ` .`; `--force` suppresses the
  block. swadesh printed the first line and the second, but not `If you insist your data is
  detokenized ...`.
- `You are turning off BLEU's internal tokenizer ...` plus `Published numbers will not be comparable
  to other papers.` - only when `-tok none` is combined with a **test set** (`-t`); with positional
  references the oracle stays silent.

The order is turning-off, then the Japanese advisory, then the detokenized block. The fix is in
`crates/swadesh-cli/src/run.rs`; the three cases are covered by
`tests/test_cli.py::test_bleu_tokenizer_advisories`.

## Reproducing

```bash
export SACREBLEU=~/.sacrebleu          # the oracle's cache; wmt19/dev is fetched on first use
~/venv311/bin/python tools/multilingual_parity.py --json /tmp/pairs.json
```

The tool prints a line per pair and writes the raw scores. It needs `target/release/swadesh`
(`cargo build --release -p swadesh-cli`) and sacreBLEU 2.6.0 on the same interpreter.

## What this does not cover

Ten pairs, one hypothesis per pair, and the default tokenizer except where noted. The option space
(tokenizers, smoothing, `-lc`, `--chrf-*`, `--ter-*`, `-ci`, paired tests) is covered by the main
suites - `tests/test_parity.py` (494 cases) and `tests/test_cli.py` (684 cases) - which run on
English-German and English-Chinese data. This page is the cross-language axis, not a replacement for
those.
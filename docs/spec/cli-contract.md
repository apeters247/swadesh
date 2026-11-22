# swadesh CLI contract

The command-line interface of `crates/swadesh-cli`. Its behaviour is defined by what it prints and
accepts, verified byte-for-byte against sacreBLEU 2.6.0's `sacrebleu` command. This document was
written before the Phase 4 work and records the contract the code is held to.

## Invocation

    swadesh [-h] [-V] [-m {bleu,chrf,ter} ...] [-i INPUT] [options] [REF ...]

References are positional files, one segment per line. Hypotheses come from stdin unless `-i` is
given. Both are read as UTF-8; other encodings are refused (`-e` accepts only `utf-8`).

## Metrics

`-m` selects one or more of `bleu`, `chrf`, `ter`; default `bleu`. Multiple metrics print in the
order given.

## BLEU options

| flag | effect | default |
|---|---|---|
| `-s` | smoothing method: `none`, `floor`, `add-k`, `exp` | `exp` |
| `-sv` | value for `floor` / `add-k` | — |
| `-tok` | tokenizer: `none`, `zh`, `13a`, `intl`, `char` | `13a` |
| `-lc` | lowercase the input | off |
| `--max-ngram-order` | highest n-gram order | 4 |
| `--force` | suppress the tokenized-input warning | off |

## chrF options

| flag | effect | default |
|---|---|---|
| `-cc` | character n-gram order | 6 |
| `-cw` | word n-gram order (2 gives chrF++) | 0 |
| `--chrf-beta` | beta | 2 |
| `--chrf-whitespace` | keep whitespace in character n-grams | off |
| `--chrf-lowercase` | lowercase the input | off |
| `--chrf-eps-smoothing` | epsilon smoothing instead of effective order | off |

## TER options

| flag | effect | default |
|---|---|---|
| `--ter-case-sensitive` | do not lowercase | off |
| `--ter-asian-support` | split CJK and fullwidth characters | off |
| `--ter-no-punct` | delete punctuation | off |
| `--ter-normalized` | normalized tokenization | off |

## Output

| flag | effect | default |
|---|---|---|
| `-f` | `json`, `text` or `latex`; `SACREBLEU_FORMAT` overrides when it names a valid format | `json` |
| `-b` | print only the score | off |
| `-w` | decimal places | 1 |
| `-sh` | shorter signature (text output) | off |
| `-sl` | sentence-level scores, text output, one metric | off |
| `-q` | suppress informational messages | off |

The JSON form is a list of objects carrying `name`, `score`, `signature` and the signature's parts,
with a `verbose_score` for BLEU and TER. Text form aligns the `=` signs across metrics. LaTeX form
is a tabular row.

Signatures are `key:value` triples joined by `|`, in a fixed order per metric, ending with
`version:2.6.0`. They are part of the output contract because tooling parses them.

## Input handling

* `-nr N` splits each reference line on tabs into `N` references.
* A reference that is `None` (a missing field in a stream) is not a reference; a segment left with
  none is the metric's error path, which differs by metric (BLEU counts it, TER divides by zero).
* `--python-version X.Y` emulates that CPython's float summation semantics. CPython changed
  `sum()` precision in 3.12 and 3.14, and BLEU's score depends on it; the default is 3.12.

## Verification

`tests/test_parity.py` runs both programs on the same inputs and compares stdout byte-for-byte,
across the flags above and the flagship cases (multi-metric, parallel, bootstrap, JSON and text
formats, stdin, multi-reference).

## Provenance

The CLI reproduces sacreBLEU's command-line behaviour as the contract users depend on. It contains
no sacreBLEU source; its scoring is swadesh's own clean-room metrics, and the rest is argument
parsing, formatting and reporting written for this interface. See `NOTICE`.
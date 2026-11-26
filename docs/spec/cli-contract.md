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

### Several systems

Two or more systems - `-i A B`, or one tab-separated field per line on stdin, named `System 1`,
`System 2`, ... - get a report rather than a single score. JSON is a list of objects carrying
`"system"` and one `"<METRIC>"` key per metric, the score as a string. Text is `tabulate`'s
`fancy_grid` and LaTeX its `latex_booktabs`; both were measured against sacreBLEU 2.6.0 and obey:

* column content width is `max(len(heading) + 2, longest cell)`, and each cell is then drawn with one
  space of padding on either side (so `BLEU` alone makes a 6-wide column, and only a value longer
  than `len(heading) + 2` widens it);
* the `System` column is right-aligned, in the heading and the body; each metric column is centred;
* the headings are the score objects' own names - `BLEU`, `chrF2`, `chrF2++` (`-cw 2`), `TER`;
* the cells are `f"{score:.{w}f}"`, as in the single-system report;
* the report ends with a blank line and the `Metric signatures` block, which `-sh` shortens;
* `-b` is ignored in this form, and `-sl` is refused.

LaTeX is the same grid with `&` between cells, `\\` at the end, and `\begin{tabular}{r...}`: `r` for
the system column and `c` per metric.

### The order of the refusals

With several systems the checks are ordered, and the order is observable when more than one would
fire. Measured, it is:

1. the paired tests name the first system the baseline and drop a later system with the same name
   (`Ignoring '<name>' as it was also given as the baseline.`);
2. duplicate names, for plain multi-system runs only (`'<name>' already used to name a system.` plus
   `Make sure to have a different basename for each system.`) - the paired tests allow a repeated
   baseline and check the rest separately;
3. `-sl` with more than one system (`Only one system can be evaluated in sentence-level mode.`),
   which beats `-ci` and beats the paired tests;
4. in the paired path only, `-sl` (`Statistical tests are unavailable in sentence-level mode.`) and
   then the count (`Paired tests require multiple input systems given to --input (-i).`);
5. `-ci` with more than one system (`Use paired tests (--paired) for multiple systems.`), even when
   `--paired-bs` was also given;
6. the stream checks: `Test set None contains no sentence`, then
   `System and reference streams have different lengths.`

All six print with sacreBLEU's own `sacreBLEU: ` prefix on stderr.

### Confidence-interval edge cases

* `--confidence-n` of **1 or less drops `-ci` silently**: no interval is printed and the `bs`/`seed`
  pairs are absent from the signature. It is not an error.
* A resampled mean of exactly `0.0` leaves sacreBLEU's interval undefined: the score is printed
  alone, but the `bs`/`seed` pairs *are* still in the signature. This is the opposite of the first
  case for the signature, so the two are separate behaviours.
* `--paired-ar-n` / `--paired-bs-n` at or below zero fall back to their defaults (10000 / 1000).

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
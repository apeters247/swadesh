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

### The paired report (`--paired-ar` / `--paired-bs`)

With several systems the first is the baseline and every other one is compared with it, per metric.
JSON is a list of objects carrying `"system"` and, per metric, an object with `"score"`, `"p_value"`,
`"mean"` and `"ci"` (`null` where an option does not produce one). Text is `fancy_grid` and -f latex
`latex_booktabs` again, with these differences from the multi-system report, all measured:

* the `System` column carries `Baseline` for the first system and the system's name for the rest;
* a metric column's heading is `<NAME> (μ ± 95% CI)` for `--paired-bs` and plain `<NAME>` for
  `--paired-ar`;
* a baseline row is `<score> (<mean> ± <ci>)` for `--paired-bs` and `<score>` for `--paired-ar`;
* every other system takes **two** grid rows: the score row, then a row whose first cell is empty
  and whose metric cells hold `(p = 0.2941)` - four decimals, with `*` appended when the p-value is
  below the 0.05 significance threshold. No row separator separates those two rows; the separator
  divides systems;
* LaTeX has one row per system, so the p-value is appended to the score cell there instead:
  `58.1 (46.8 ± 38.0) (p = 0.2941)`, and the heading's `%` is escaped as `\%`;
* the footer is a titled block whose dashes are as long as the title, then four paragraphs: the
  "Each system is pairwise compared to ." line (the baseline name is printed empty - kept, it is
  what sacreBLEU prints), a null-hypothesis paragraph, a significance-threshold paragraph that also
  mentions the `*` marker and red highlighting, and a NOTE paragraph. The title and second line
  differ between the two tests:
  `Paired bootstrap resampling test with <n> resampling trials` plus
  `Actual system score / bootstrap estimated true mean / 95% CI are provided for each metric.`, or
  `Paired approximate randomization test with <n> trials` plus
  `Actual system score is provided for each metric.`;
* the signature gains `bs:<n>` (`--paired-bs`) or `ar:<n>` (`--paired-ar`) and `seed:<seed>` between
  `nrefs` and the metric-specific pairs.

### `--citation`

`--citation` (alias `--cite`) prints the citation of a test set, so sacreBLEU has no default for it
and refuses the flag without `-t`: `sacreBLEU: I need a test set (-t).` on stderr, exit 1. The place
of that check is measured: it runs after the sentence-level one, so `--citation -m bleu chrf -sl`
reports the metric and `--citation -sl` reports the test set; it runs before the reference checks, so
bare `--citation` reports the test set rather than the missing references; and `--list`
short-circuits it entirely.

With a test set, sacreBLEU 2.6.0 **crashes** - `TypeError: argument of type 'WMTXMLDataset' is not
iterable`, from a `'citation' not in DATASETS[test_set]` test whose values are `Dataset` objects - so
the set-specific citation is not reproduced. `-t` stays unsupported and the combination is refused
with the `-t` hint, which is a documented divergence: the oracle exits 1 with a traceback where
swadesh exits 2 with the hint.

### Argument errors

An argument error prints argparse's usage block and exits 2, as sacreBLEU's do. The usage text is
transcribed from 2.6.0 at the default 80 columns (argparse wraps it to the terminal width) with the
program name substituted, which is the one intentional difference.

* The option name in the error line is `--long/-short` - the long form first, even when the short one
  was typed. `--chrf-beta`, `--max-ngram-order` and `--python-version` have no short form, so they
  appear alone.
* `-j` is the short form of `--paired-jobs`; `-pj` is not a flag.
* `--paired-ar` and `--paired-bs` are mutually exclusive, and the message names the one that came
  **second** as the argument that is not allowed with the first:
  `argument --paired-bs/-pbs: not allowed with argument --paired-ar/-par`.

Documented divergences, each pinned by a strict `xfail` test:

* with more than one token left unconsumed, argparse lists them all
  (`unrecognized arguments: -pj 1`) where swadesh stops at the first;
* with `--paired-jobs N` for N > 1 sacreBLEU prints `Launching N parallel workers.` and its fork pool
  reorders the per-system status lines; swadesh scores serially and prints them in system order. The
  scores are identical, and `-j 1`, the default, is byte-identical throughout;
* `--max-ngram-order` and `--python-version` are swadesh additions with no sacreBLEU equivalent, so
  the oracle rejects them as unrecognized where swadesh accepts them.

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
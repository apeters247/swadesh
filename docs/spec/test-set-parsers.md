# Test-set parsers: derived specification

This document records, for each **parser family**, the rule that turns a test set's `raw/` tree into
the *processed* files sacreBLEU's cache holds (`<prefix>.<pair>.<field>`). Written from black-box
measurement of sacreBLEU 2.6.0's own cache and raw trees; no sacreBLEU source was read.

Each family was measured by regenerating the processed files from `raw/` and byte-comparing them
with what the oracle itself wrote, then committed with its module (`sgml.rs`, `wmtxml.rs`,
`plain.rs`, `iwslt.rs`). This file collects the rules and, in its last section, the newly measured
IWSLT rule that the `iwslt.rs` module implements.

## Why a family split

The processed file layout is one layout (`docs/spec/user-test-sets.md`), but the raw tree differs per
publication: FakeSGML for `wmt08`..`wmt19`, WMT's XML for `wmt20`..`wmt24`, plain parallel files for
`multi30k`, and IWSLT's own TED XML for `iwslt17`. Each parser recognises only the layout it
measured, so a set a parser does not know falls through to the ordinary refusal rather than being
parsed into plausible-looking wrong data.

## The families, and the rule each measured

| Family | Sets | Raw layout | Text rule |
| --- | --- | --- | --- |
| FakeSGML | `wmt08`..`wmt19` | `<base>-<src><tgt>-{src,ref}.<lang>.sgm`, `<doc>` blocks of `<seg>` | verbatim (no entity occurs) |
| WMT XML | `wmt20`..`wmt24`, `wmt21/*` | `*.xml` with `<src>`/`<ref>` documents | entity-decoded (XML parser shape) |
| plain text | `multi30k/*` | two parallel files named by the registry | each line `rstrip`ped |
| IWSLT XML | `iwslt17*` | TED `<mteval>` XML, `<doc>` of `<seg id="N">` | whitespace-collapsed, **no** entity decoding |

Fields written are only those the set's catalogue entry declares. `src` and `ref` come from the two
files the registry's per-pair `paths` name; the metadata (`docid`, `genre`, `origlang`) is carried
from the enclosing `<doc>` onto every one of its segments.

## IWSLT XML (measured 2026-11-26, the rule `iwslt.rs` implements)

Measured on `iwslt17`, whose raw tree is small enough to fetch whole, against all twelve pairs.

* **Files.** The registry's per-pair `paths` are `[<pair>/IWSLT17.TED.tstYYYY.<pair>.<src>.xml,
  <tgt>-<src>/IWSLT17.TED.tstYYYY.<tgt>-<src>.<tgt>.xml]`. Both are located under `raw/` by base
  name. The reference lives in the *other* direction's directory (`en-fr`'s reference is the `fr`
  file inside `fr-en/`).
* **Segments.** `<seg id="N">text</seg>` in file order; `id` is not used for pairing. The source's
  `<doc docid="..." genre="...">` attributes (there is no `origlang` attribute in this family) are
  repeated onto each of its segments, exactly as the FakeSGML rule does.
* **Text.** Collapse runs of whitespace and strip: `" ".join(text.split())`. Measured to be the whole
  rule - the *plain* strip leaves 819 of `iwslt17` en-fr's 1455 source lines wrong (the file's
  segments contain doubled internal spaces), collapse leaves none.
* **No entity decoding.** The single `&`-bearing segment in the whole family
  (`... Bed Bath &amp; Beyond ...`) is written out with `&amp;` **literal**, so sacreBLEU is not
  running an XML parser over the segment text and neither may swadesh. `&lt;`/`&#NN;` forms do not
  occur, and the same rule (take the bytes between `<seg ...>` and `</seg>`) reproduces them.
* **`origlang` is empty.** This family's `<doc>` carries no `origlang`, so every `origlang` line is
  blank; the field still exists in the catalogue.

**Measurement.** A regex/`split()` reference implementation in Python reproduced the oracle's
`--echo all` output for all twelve pairs of `iwslt17` byte for byte: 1455 (en-fr), 1138 (en-de),
1459 (en-zh), 1460 (en-ar), 1452 (en-ja), 1429 (en-ko) segments per direction, 0 mismatches.

## Downloading a set: all archives, in URL order (measured)

The catalogue stores a **flat** list of URLs per set with no per-pair mapping, and the oracle does
not need one: `sacreBLEU: Downloading`/`Extracting` are printed once **per URL, in the order the
registry lists them**, for *every* URL of the set (measured: `-t iwslt17 -l en-fr` fetches all twelve
archives even though only one pair is parsed). Each archive is written as
`raw/<set prefix>.<url basename>`, its md5 checked against the parallel `md5` entry, and unpacked into
`raw/`. So the multi-archive case is a loop over `data`, not a lookup, and 27 single-archive and 31
multi-archive sets share one code path.

## Verification

* Each parser's tests regenerate the processed files into a scratch cache (symlinking or extracting
  the real `raw/`), and the CLI's `--echo`/scoring output is byte-diffed against
  `sacrebleu 2.6.0` (`tests/test_cli.py`).
* `iwslt.rs`'s check is the Python reference implementation above plus a CLI byte-diff on the fetched
  set, and the cold-cache download test that pins the per-URL `Downloading`/`Extracting` lines.

## Provenance

The processed-cache layout and the download notices are sacreBLEU-defined and measured; the parse
rules are derived from the raw files sacreBLEU itself downloads and the files it writes next to them.
No sacreBLEU Python source was read and no Moses `tokenizer.perl` text was transcribed.

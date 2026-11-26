# User test sets: derived specification

This document defines a **swadesh-only addition** to the CLI: the ability to score against a test
set the user provides, without a download and without forking the catalogue. It was written before
the implementation.

**Sourcing rule for this file.** Everything sacreBLEU defines (the refusal wording, the shape of
`--echo`, `--detail`, the language-pair error) is taken from black-box measurement of sacreBLEU
2.6.0 and is recorded in `cli-contract.md`; none of it changes here. Everything sacreBLEU does *not*
define - the resolution roots, the field discovery, the user-set `--list` - is swadesh's own design,
so the justification below is design rationale rather than measurement. The one measured input to
the design is that sacreBLEU's own processed cache is already the layout we want to read:

    ~/.sacrebleu/wmt16/wmt16.en-de.src  .ref  .docid  .origlang  .genre
    ~/.sacrebleu/wmt22/wmt22.de-fr.{src,ref-A,ref-B,docid,origlang,domain,LT22,Online-A,...}

That is the whole format. A directory of exactly those files is a usable test set, and reading it
needs no archive decoding, no HTTP client and no new parser.

## What it is for

Today a user's own parallel data can only be scored by passing reference files positionally
(`swadesh ref.txt -i hyp.txt`), which loses everything a test set gives: named references, multiple
references, per-segment `origlang`/`docid`/`genre`, `--subset` slices, `--echo`, `--detail`, and a
signature that says which set was scored. This closes that gap for local data, and it is also the
substrate the downloaded sets must land in later, so it is a step towards `-t`, not a detour.

## Non-goals

* No new file format or manifest: the layout is the one sacreBLEU already writes.
* No new dependency. A root is read with `std::fs`; nothing here needs a crate.
* No change to any behaviour sacreBLEU defines. In particular the refusal wording for an unknown test
  set or an unavailable language pair stays exactly as measured, and a name the built-in catalogue
  knows always resolves to the built-in.
* No download. Fetching is a separate phase (§ Phasing).

## Resolution

`-t NAME` resolves in this order, first match wins:

1. the built-in catalogue (the 67 sets of `datasets.rs`), so sacreBLEU-defined names are untouched;
2. each root of `$SWADESH_TEST_SETS`, colon-separated, in the order given;
3. `$SACREBLEU` (default `~/.sacrebleu`), the cache sacreBLEU itself uses.

A `NAME` containing a path separator is taken as a directory and used directly, which is what makes
`-t ./my-set` work without any environment variable. A root resolves `NAME` by looking for
`<root>/<NAME>/<NAME>.<pair>.<field>`.

Because the built-in catalogue wins, a user set cannot shadow a WMT name; a set is named something
the catalogue does not have (documented here so the rule is predictable rather than surprising).

## Fields

A language pair exists when `<set>.<pair>.src` and any reference file exist. The fields are the
file suffixes present, with these names:

| File suffix | Name in `--echo` |
|---|---|
| `.src` | `src` |
| `.ref` | `ref` |
| `.ref-A`, `.ref-B`, ... | `ref:A`, `ref:B`, ... |
| `.docid` | `docid` |
| `.origlang` | `origlang` |
| `.genre` | `genre` |
| `.domain` | `domain` |
| anything else, e.g. `.Online-A` | the suffix verbatim, `Online-A` |

`--echo ref` prints `.ref` when it exists, otherwise the first `ref:*` in sorted order, which is the
reference sacreBLEU scores against by default. Any other field is an error in sacreBLEU's wording if
the file is missing, so the caller sees the message it already knows.

`-t NAME --list` prints one line per pair for a user set, in the shape sacreBLEU uses for its own
catalogue:

    en-de: src, ref, ref:A, docid, origlang, genre

with the field order above (never alphabetical, so `ref` and `ref:A` stay adjacent and the
informational fields come last). sacreBLEU's own order for that line is catalogue-driven - wmt16
prints `src, ref, docid, genre, origlang` and wmt22 `src, ref:A, ref:B, docid, origlang, ...` - so
there is nothing to match here; a user set gets one fixed order instead of a per-set one.

## Composes with everything already implemented

| Flag | Behaviour on a user set |
|---|---|
| `-i HYP` | scores against the set's default reference; no positional references needed |
| `--echo FIELD...` | prints the fields requested, tab-separated, exactly as for a built-in set |
| `-ol/--origlang X` | keeps segments whose `origlang` field is `X`, or `non-X`; needs the `.origlang` file |
| `--subset X` | keeps segments whose `domain`/`genre` field is `X`; needs one of those files |
| `-d/--detail` | the per-`origlang` x domain breakdown, measured shape: `origlang=de domain=ALL : sentences=1499   BLEU = 100.0` |
| `--citation` | prints `<dir>/citation` when the set has one (a plain file, not a per-pair field); otherwise sacreBLEU's refusal wording |
| `-m/-s/-sv/-tok/-lc/-w/...` | unchanged; the signature names the set, as for a built-in |

A requested pair with no files raises the measured language-pair error:
`sacreBLEU: No such language pair 'en-xx'` followed by the available pairs, which for a user set are
the pairs discovered in the root.

## Phasing

* **Phase A (this specification).** Roots, resolution, field discovery, `--echo`, `--origlang`,
  `--subset`, `--detail`, scoring, `--citation` from a local file. No dependencies. The built-in
  catalogue keeps its current refusal until Phase B.
* **Phase B.** URLs, behind a `downloader` cargo feature (the crate already has
  `default = ["parallel"]`, and `crates/swadesh-wasm` depends on it with `default-features = false`).
  Downloaded sets extract into the same layout, so nothing in Phase A changes; the five archive
  parsers live here, because a built-in set's raw files are not in the layout.

## Verification

The acceptance test needs no downloader and no new data, because sacreBLEU's own cache is the format:
point `SWADESH_TEST_SETS` at `$SACREBLEU` and byte-diff our output against the oracle's for sets that
are already cached on this host -

* `-t wmt16 -l en-de --echo src|ref|docid|origlang|genre`, one field and several at once;
* `-t wmt16 -l en-de --echo src -ol en` and `-ol non-en`;
* `-t wmt16 -l en-de -i HYP` and `-m bleu chrf ter`, scored against `.ref`;
* `-t wmt22 -l de-fr --echo ref`, `ref:A`, `ref:B` and a system field such as `Online-A`;
* `-t wmt16 -l en-de -i HYP -d`, the breakdown lines;
* `-t wmt16 --list`, the discovered pairs and fields.

Each is compared on stdout, stderr and exit code. A set the catalogue does not know must still fail
with sacreBLEU's two-line refusal, and anything the root does not have must fail the way sacreBLEU
fails for a missing pair.

## Provenance

No data is added to the repository and none is redistributed: a user set is the user's own files, in
their own directories. Nothing here changes `NOTICE` or the licensing position.

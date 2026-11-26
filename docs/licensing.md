# Licensing

swadesh is dual-licensed.

| | licence | cost | file |
|---|---|---|---|
| Community | GNU Affero General Public License v3 or later | free | `LICENSE` |
| Commercial | negotiated per use; free below the revenue threshold | free / paid | `LICENSE.commercial` |
| Adapter modules | Apache License 2.0 | free | `LICENSE-Apache-2.0` |

## What AGPL means in practice

The AGPL's practical difference from the GPL is section 13: if users interact with a modified
version of the software **over a network**, the modified source must be offered to them. That is the
only place it reaches beyond what the GPL would require.

In plain terms:

**Free, no obligation beyond the licence itself**
* Running swadesh inside your own organisation to score your own data — evaluation pipelines,
  CI checks, internal tooling. This is the common case and it costs nothing.
* Using it as a build-time or analysis tool whose output you own, where swadesh is not distributed
  to your users.
* Publishing your own work under the AGPL or a compatible licence.
* Research, teaching, and benchmarking.

**Needs the commercial licence**
* Offering swadesh, or a modified version, as a network service, when you are not willing to publish
  the source of that service under the AGPL.
* Shipping swadesh inside a closed product distributed as a binary: an application, an appliance, an
  SDK.

If you are unsure which side of a line you are on, ask before you ship. The contact is in
`LICENSE.commercial`.

## The revenue threshold

The commercial licence is **free of charge** for a licensee whose total annual revenue - across the
licensee and its affiliates - is **below USD 5,000,000** in the most recent fiscal year. At or above
that figure it is paid and granted by written agreement. The figure is self-reported: there is no
audit and nothing to sign, so under the threshold `LICENSE.commercial` is the grant.

This is deliberately not a restriction on the AGPL: the AGPL is free for everyone at any size, and
the threshold only decides whether the *alternative* licence costs anything.

## Why dual licensing rather than a source-available licence

PyPI, crates.io and Hugging Face all accept the AGPL. A source-available licence that forbids
commercial use would be rejected by some of them, and would also stop the research and evaluation
use that this tool exists to serve. The AGPL keeps it genuinely free for the community while leaving
one route open for organisations that cannot take the AGPL - free for small ones, paid for large ones, so the licence meets a two-person studio and a vendor on the same day.

## What is Apache-2.0, and why

The Python metric classes are this project's own now, and `import swadesh` imports no sacreBLEU. The
parts that still touch it do so on use, and are under sacreBLEU's Apache-2.0 notice as well as this
project's: the `patch()` adapter and the module-level `corpus_*`/`sentence_*` wrappers in
`__init__.py`, `verify.py`, `cli.py`, `significance.py`'s `install()`/`uninstall()`, and the six
tokenizers that need mecab or sentencepiece. `LICENSE-Apache-2.0` retains that text and `NOTICE`
names them.

Everything in `crates/` is original to this project and carries only AGPL-3.0-or-later, with one
disclosed exception below.

### Exception: the TER shift search (2026-11-24)

One part of `crates/` is no longer clean-room. The candidate-budget semantics of the TER shift search
in `crates/swadesh-core/src/ter_clean.rs` were re-derived on 2026-11-24 **by reading** sacreBLEU's
Apache-2.0 `sacrebleu/metrics/lib_ter.py` (Copyright 2020 Memsource), on the copyright holder's
explicit instruction, after black-box probing had shown the oracle departs from the published
algorithm and no probe could recover the rule. The rule recovered is small - the 1,000-candidate
budget is tested once per matching block and before the reduction, so the search overshoots by up to
one block and refuses the shift found on the exhausting iteration - and it is expressed here in this
project's own terms and types, but the derivation is not clean-room for that file.

Consequences for the licence decision: `ter_clean.rs`'s shift search is a derivative work of an
Apache-2.0 file, so Apache-2.0's notice and attribution conditions attach to it, and the
"clean-room, dual-licensed" description must not be repeated for TER. The rest of the crate is
unaffected. The copyright holder should confirm the licence position with counsel before any
commercial licence is offered that covers TER.

### Ported module: `python/swadesh/base.py` (2026-11-25)

`python/swadesh/base.py` - the `Score`, `Signature` and `Metric` base classes the metrics derive from
- is a **direct port** of sacreBLEU's `sacrebleu/metrics/base.py` (Apache-2.0, Copyright Matt Post and
contributors), made with the copyright holder's authorisation so that the metrics stop inheriting from
the sacreBLEU package. Apache-2.0's notice and attribution conditions attach to it. It is held to
sacreBLEU's behaviour by `tests/test_base_port.py`, which diffs signatures, score formatting and the
argument-check errors against the real classes.

The module carries no sacreBLEU import: it is the first piece of the package that can run without
sacreBLEU installed. Unlike the TER budget rule (which was black-box unrecoverable), this file *could*
be re-derived from behaviour and the clean-room claim restored for it; it was ported because the
copyright holder authorised a direct port.

### Ported metric layer (2026-11-25)

The same authorisation covers the rest of the move off sacreBLEU. Ported so far, all Apache-2.0
derivatives (Copyright Matt Post and contributors):

| what | ported from |
|---|---|
| `python/swadesh/base.py` - `Score`, `Signature`, `Metric` | `sacrebleu/metrics/base.py` |
| TER's tokenizer surface (`_NativeTerTokenizer`, `metric.tokenizer`) | `sacrebleu/tokenizers/tokenizer_ter.py` (interface only) |
| `BLEU`: class constants, `__init__`, `_preprocess_segment`, `_compute_score_from_stats`, `_compute_bleu` | `sacrebleu/metrics/bleu.py` |
| `CHRF`: class constants, `__init__`, `_preprocess_segment`, `_remove_punctuation`, `_compute_f_score`, `_compute_score_from_stats` | `sacrebleu/metrics/chrf.py` |
| `BLEUSignature`, `CHRFSignature`, `TERSignature` | the same two modules |

`crates/` is unaffected: the Rust kernels are still the clean-room work, and the ports are the
Python layer that had been inheriting from sacreBLEU. The five native tokenizers work without
sacreBLEU installed; the six exotic ones still need it (they need mecab or sentencepiece in any
case) and it is imported lazily, only when one of those names is asked for.

## Spec-first: what holds, and what does not

`NOTICE` describes the Rust crate as written from published specifications and from black-box
measurement of sacreBLEU's observable output, and `docs/spec/` is offered as the record of that.
That holds for the clean-room rewrites, and it does **not** hold for every file in the tree. The
history shows three areas committed three days before the specification that now describes them, and
never rewritten since:

| area | spec committed | implementation first committed | spec first? |
|---|---|---|---|
| tokenizers: `tokenize_nonechar_clean.rs`, `13a`, `zh`, `intl` | `docs/spec/tokenizers.md` `0f77954`, 11-22 00:34 | `edfd940` 01:43, `3a420e4` 03:21, `515f3ba` 03:42, `87cbc03` 04:05 | yes |
| BLEU/chrF segment statistics: `bleu_stats_clean.rs`, `chrf_stats_clean.rs` | `docs/spec/stats.md` `c3e3445`, 11-22 04:52 | `3d8df0c` 13:04, `34ecd83` 13:09 | yes |
| TER: `ter_clean.rs` | `docs/spec/ter.md` `47539ec`, 11-22 13:39 | `db8e08b` 14:17 | yes (shift-search budget rule re-derived from source, 11-24 - see above) |
| **BLEU/chrF numeric core: `score.rs`, `ngram.rs`** | `docs/spec/stats.md`, 11-22 04:52 | `6b13050`, **11-19 04:35** | **no** |
| **significance kernels: `nprng.rs`, `resample.rs`, `pairwise.rs`** | `docs/spec/significance.md` `478e2f1`, 11-22 17:05 | `7f9b7da` and `60bf8fa` **11-19 04:45**, `5eb8843` 04:49 | **no** |
| **CLI: `crates/swadesh-cli/`** | `docs/spec/cli-contract.md` `2b13911`, 11-22 15:51 | `main.rs` `943dcf0`, **11-19 04:59** | **no** |

For the three bold rows `git log --since=<spec commit>` over those paths returns nothing, so the
specification records behaviour after the fact rather than before it. The repository also holds no
measurement record for them: the lab notebook has no entries dated 2026-11-19, and no probe file for
that date is tracked.

What this does and does not mean. It does not mean any sacreBLEU source was copied — there is no
sign of transcription anywhere in the tree (`tokenizer.perl` is named only in the specifications, in
sentences saying it was not read; `tests/test_provenance.py` guards the module boundaries). What it
means is that for those three areas the project cannot demonstrate the *spec-first* order that the
rest of the crate can, so this file does not claim it. Closing the gap means either re-deriving them
after their specifications, or recording for each one the measurement it was actually derived from.

## Contributions

To keep dual licensing possible, every contributor agrees to `CLA.md` before a pull request is
merged. The agreement lets the copyright holder license your contribution under the AGPL *and*
under the commercial licence — without it, a single outside contribution would block the commercial
side permanently.

## Third-party components

Checked at the time of writing; all are permissive and all are compatible with AGPL-3.0-or-later.

**Rust crates (108 third-party, from `cargo metadata`).** The downloader added the only
non-permissive-looking name in the set and it is permissive too: `webpki-roots` is
`CDLA-Permissive-2.0`, a data licence for the certificate roots. `ring` is `Apache-2.0 AND ISC`
and compiles C and assembly; `rustls`, `rustls-webpki` and `untrusted` are the TLS stack it
serves. `--no-default-features` drops the whole downloader and its crates.

| licence | crates |
|---|---|
| `MIT OR Apache-2.0` | 62 |
| `Unicode-3.0` | 18 |
| `Apache-2.0 OR MIT` | 7 |
| `Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT` | 3 |
| `MIT` | 3 |
| `CDLA-Permissive-2.0` | 2 |
| `ISC` | 2 |
| `MIT/Apache-2.0` | 2 |
| `(MIT OR Apache-2.0) AND Unicode-3.0` | 1 |
| `0BSD OR MIT OR Apache-2.0` | 1 |
| `Apache-2.0 AND ISC` | 1 |
| `Apache-2.0 OR ISC OR MIT` | 1 |
| `Apache-2.0 WITH LLVM-exception` | 1 |
| `BSD-3-Clause` | 1 |
| `MIT OR Zlib OR Apache-2.0` | 1 |
| `Unlicense OR MIT` | 1 |
| `Zlib` | 1 |
No third-party crate is copyleft. The only copyleft entries in `cargo metadata` are swadesh's own
four crates.

**Python**:

* NumPy 2.x — BSD (its distribution ships `LICENSE.txt`).
* sacreBLEU 2.6.0 — Apache-2.0, the five adapter modules described above.
* pytest and hypothesis are test-only and not runtime dependencies.

**Data**:

* The Unicode Character Database, under the Unicode Terms of Use, is the source for the generated
  tables in `crates/swadesh-core/src/tables.rs` (produced by `tools/gen_tables.py`).

None of these is bundled in a way that changes this project's licence.

## A note on the commercial licence text

`LICENSE.commercial` states the terms in outline and points to the copyright holder for a written
agreement. It is deliberately not a template contract: the scope, fee and warranties are agreed per
use, and that agreement is the binding document.
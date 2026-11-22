# Licensing

swadesh is dual-licensed.

| | licence | cost | file |
|---|---|---|---|
| Community | GNU Affero General Public License v3 or later | free | `LICENSE` |
| Commercial | negotiated per use | paid | `LICENSE.commercial` |
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

## Why dual licensing rather than a source-available licence

PyPI, crates.io and Hugging Face all accept the AGPL. A source-available licence that forbids
commercial use would be rejected by some of them, and would also stop the research and evaluation
use that this tool exists to serve. The AGPL keeps it genuinely free for the community while leaving
one route open for organisations that cannot take the AGPL.

## What is Apache-2.0, and why

Five Python modules still import sacreBLEU and are therefore under sacreBLEU's Apache-2.0 notice as
well as this project's: `__init__.py` (the `patch()` adapter), `verify.py`, `metrics.py`,
`significance.py` and `cli.py`. `LICENSE-Apache-2.0` retains that text. `NOTICE` names them. Replacing
them with standalone implementations is planned.

Everything in `crates/` is original to this project and carries only AGPL-3.0-or-later.

## Contributions

To keep dual licensing possible, every contributor agrees to `CLA.md` before a pull request is
merged. The agreement lets the copyright holder license your contribution under the AGPL *and*
under the commercial licence — without it, a single outside contribution would block the commercial
side permanently.

## Third-party components

Checked at the time of writing; all are permissive and all are compatible with AGPL-3.0-or-later.

**Rust crates** (26 third-party, from `cargo metadata`):

| licence | crates |
|---|---|
| `MIT OR Apache-2.0` | 20 |
| `Apache-2.0 OR MIT` | 2 |
| `(MIT OR Apache-2.0) AND Unicode-3.0` | 1 |
| `Apache-2.0 WITH LLVM-exception` | 1 |
| `Unlicense OR MIT` | 1 |

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
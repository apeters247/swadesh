# swadesh — master plan

**What this file is.** The single, nested plan an agentic coding system can execute end to end: every
workstream, its tasks, and for each task a *measurable* acceptance line. It replaces the earlier
52-line `PLAN.md` (which described the VPS worktree workflow and a tree that no longer exists) and it
supersedes `LAUNCH.md`'s "not published" note — see decision D-1.

**How to use it.** Read §1 and §2 before touching anything; they are the rules. Then pick the lowest
unfinished task in the current milestone (§5). Work the repo's loop: spec or measurement record first,
then code, then verify, then one commit, then a `docs/lab-notebook.md` entry. Update the task's status
in §10 as you finish it; a task is only `done` when its acceptance line has been *run* and its result
copied into the notebook.

Status vocabulary: `todo` · `doing` · `done` · `blocked` (with the blocking decision named) ·
`dropped` (with the reason).

---

## 1. Non-negotiables

These never bend; a change that breaks one is not a feature, it is a regression.

1. **Bit-exact parity is the product.** Every statistic, score, signature string and printed line
   compares `==` with sacreBLEU 2.6.0 on the same CPython, for the same inputs. The oracle is the
   reference implementation, called but never read (one disclosed exception, recorded in `NOTICE`).
2. **Drop-in.** Same class names, constructor arguments, `Score` objects, attribute names, signatures,
   CLI flags, stdout/stderr and exit codes. `import swadesh as sacrebleu` is the whole migration.
3. **Nothing new changes a default.** New features are opt-in: a new method, a new flag, a new
   endpoint, a new build feature. They may not alter any score, any existing flag's output, or any byte
   of the default CLI output. This includes performance work.
4. **Measured claims only.** Every number in README/docs comes from a command in this repo on a named
   machine, with the command recorded. No estimate is ever presented as a measurement.
5. **One verified slice per commit.** Spec or measurement first; then code; then the narrow suite, then
   `cargo test --release --workspace` and the Python selection; then commit and push; then a dated
   notebook entry (attempted, numbers, failures and why, decisions, next step).
6. **Tests are never loosened.** No skipping, xfailing or deleting a test to get green. A divergence
   that cannot be reproduced byte-identically becomes an explicit `xfail(strict=True)` *plus* a line in
   `docs/parity-report.md` *plus* a notebook entry — never a silent pass.
7. **Frozen oracle expectations change only by re-freezing** from sacreBLEU, with the reason recorded.
8. **Specs are not edited to match code.** A spec changes because a measurement changed, in its own
   commit, before the code.

## 2. Architecture policy (the shape every change must respect)

```
Rust core (all logic)        crates/swadesh-core     no I/O, no Python, no CLI
  └─ engine surface          crates/swadesh-cli      the CLI + report formats (owns no logic)
  └─ Python surface          crates/swadesh-py       pyo3/abi3 → python/swadesh (thin adapters)
  └─ wasm surface            crates/swadesh-wasm     the browser/Node playground
Other languages              CLI + JSON contract  |  C ABI (cdylib)  |  wasm  |  MCP server
```

* **All metric logic lives in Rust.** Tokenizers, statistics, score arithmetic, TER, the bootstrap/AR
  resamplers, aggregation, significance, and every new analysis feature. Nothing is implemented twice.
* **Python is a compatibility layer only** — the sacreBLEU-shaped classes, `patch()`, the MCP server,
  `verify.py`, the reward wrappers, and the exotic tokenizers that need Python packages. No new logic
  is added there; new logic is exposed *from* Rust and wrapped.
* **Every other language goes through one of four doors.** In order of preference: (1) the CLI with a
  documented, versioned JSON contract (works for shell, Go, Java, .NET, Ruby, PHP, R, Julia, anything);
  (2) the C ABI (`swadesh-core` as a `cdylib` + a stable header, which also gives C++, Go/cgo, JVM/JNA,
  .NET/P-Invoke, R/.Call, Ruby/FFI and PHP/FFI a one-artefact path); (3) wasm for browser and Node;
  (4) MCP for agents. No language gets its own copy of a metric.
* **Feature gates, not forks.** Optional capability (the `-t` downloader, extra formats) sits behind a
  cargo feature so a minimal, dependency-free build remains possible. `--no-default-features` must keep
  building and keep passing the core suites.

## 3. Decision log

| ID | Date | Decision | Supersedes / notes |
|---|---|---|---|
| D-1 | 2026-11-26 | **Publishing is permitted.** The public repo, PyPI, crates.io, npm, Hugging Face, a docs site, and `v*` release tags are all in scope. The AGPL + commercial dual licence is the licensing model. | Supersedes the "never publish" rule, `LAUNCH.md`, and the neutered `release.yml` ("no publish job, tags do nothing"). |
| D-2 | 2026-11-26 | **The `-t` downloader is native Rust** (`ureq`-class HTTP + a TLS provider + `flate2`/`tar`/`zip`), behind a `downloader` cargo feature. A `--no-default-features` build serves a pre-populated cache and keeps zero third-party dependencies (the old "option 3", for free). | Closes the blocked decision (was "1, 2 or 3"). |
| D-3 | 2026-11-26 | **Rust owns the logic; Python is compatibility; other languages enter through CLI+JSON, the C ABI, wasm or MCP** (§2). | New. |
| D-4 | 2026-11-26 | **New features are opt-in and off the parity default** (§1.3). | Carried forward from `docs/feature-ideas.md`. |
| D-5 | 2026-11-26 | **The compat target is sacreBLEU 2.6.0, pinned** (`swadesh.SACREBLEU_COMPAT_VERSION`). A future sacreBLEU release is a *new* target: re-freeze, re-measure, never drift silently. | New. |
| D-6 | 2026-11-26 | **One data layer, two sources.** The user-set reader (WS-1) is built first and is what the downloader feeds; `--echo`/`--origlang`/`--subset`/`--detail`/`--citation` are implemented once against it. | Orders WS-1 before WS-0's `-t`. |
| D-7 | 2026-11-26 | **The repository never redistributes third-party corpora.** Built-in sets are fetched from sacreBLEU's own catalogue URLs with the recorded md5; user sets live in the user's directories; any Hub mirror is a separate, licence-reviewed artefact that links rather than bundles. | New. |
| D-8 | 2026-11-26 | **Perf work has a budget and a watchdog**: a benchmark in CI on a named machine, and any change over ±5% on a tracked corpus must be explained in the notebook. | New. |

## 4. Where the tree stands (measured baseline, 2026-11-26, troy-1)

Everything below is a re-measurement from this session, at commit `f485aac`. Counts are floors: a
later run may be higher, never lower, and never with a failure.

| Suite | Command | Baseline |
|---|---|---|
| Rust | `cargo test --release --workspace` | 13 suites, 0 failed |
| CLI (Rust binary vs the oracle) | `SWADESH_DATA=bench/data pytest tests/test_cli.py -q` | 582 passed, 2 xfailed, 0 failed |
| Python selection | the 17 files listed in `docs/parity-report.md` | 885 passed, 60 skipped, 0 failed |
| Slow pair | `pytest tests/test_ter.py tests/test_deep.py -q` | 61 passed, 2 skipped |
| Long differential | `pytest tests/test_differential_long.py -q` | 46 passed |
| MCP + regressions + None-refs | `pytest tests/test_mcp.py tests/test_regressions.py tests/test_numrefs_none_refs.py -q` | 10 passed |
| Mutation kills | `pytest tests/test_mutation_kills.py -q` | 14 passed |
| Speed | `bench/` on troy-1 (24 cores) | 22-46x per core; 125-318x at 24 threads |
| Scale | 1M segments | swadesh ~1.0 s; sacreBLEU OOM (chrF at 19.9 GB, BLEU at 40-46 GB) |

CLI flags: everything sacreBLEU has is byte-identical **except** the test-set group (`-t`,
`--download`, `--echo`, `--origlang`, `--subset`), `--detail`, the six exotic tokenizers, `-e` beyond
UTF-8, and `.gz` input. Three divergences are documented and pinned by strict `xfail`s (the
unconsumed-token list, `-j > 1` status lines, negative `--chrf-word-order`). The full list, with the
test that pins each item, is `docs/parity-report.md` § Known remaining differences.

## 5. Milestones

Each milestone is a shippable state with exit criteria that can be *run*. Workstreams (§6) are the
labelled work; milestones are the order to do it in.

| M | Name | Workstream tasks | Exit criteria (all must be demonstrated) |
|---|---|---|---|
| **M0** | Parity complete | WS-1 all; WS-0.1-0.8 | The four end-state items of the current goal hold; `tests/test_cli.py` runs the `-t` cases (no longer skipped) and byte-diffs them; `cargo test --release --workspace` 0 failed; `docs/parity-report.md` has no undated row. |
| **M1** | Installable by strangers | WS-2.1-2.7 | Repo public; `v0.2.0` tagged; PyPI wheel (3 OS x 3 Python) + sdist; static binaries on the release page; DOI minted; `tools/smoke_install.sh` passes in a clean container for pip, binary, and docker; the three routes produce byte-identical scores. |
| **M2** | Hooks for other languages | WS-3.1-3.6 | A stable JSON contract doc; a C ABI (`libswadesh` cdylib) with a header; runnable examples in C, Go, Java, .NET, Ruby, R, all executed in CI; an npm package for the wasm build; the hooks matrix shows each route matching the CLI byte for byte. |
| **M3** | The researcher cluster | WS-4.1-4.6 | Each feature has a spec, tests, and a docs section; a worked meta-evaluation reproduces a published-style metric/judgement correlation on a real metrics-task file; a results-receipt round-trips through `swadesh verify`. |
| **M4** | The developer habit | WS-5.1-5.5 | `swadesh gate`, `rank`, `audit`, `report` shipped with docs; the GitHub Action is published and used by an example repo where a deliberate prompt change fails the gate; the MCP server exposes the agent verbs. |
| **M5** | Hugging Face native | WS-6.1-6.4 | `evaluate.load("swadesh")` works in a clean env; a lighteval task computes BLEU through swadesh; a Space runs the playground; the streaming accumulator scores a 1M-segment corpus under a stated memory ceiling. |
| **M6** | Recognition | WS-7.1-7.5 | The findings note is public with a DOI; `CITATION.cff` makes GitHub show "Cite this repository"; the WMT metrics-task submission is sent; the benchmark page and docs site are live; the "used by" list is maintained. |

Dependency order: M0 → M1 → {M2, M3, M4, M5} → M6. M2-M5 are independent of each other and may be
interleaved; each ends with the full suite green and a notebook entry.

---

## 6. Workstreams

Task schema: **ID — title.** *What to do.* **Acceptance:** the command(s) to run and the result that
counts as done. *(Effort — days.)* Tasks marked **[swadesh-only]** must not change any default output.

### WS-0 — Parity completion (finish the drop-in)

**WS0.1 — The shared dataset reader.** A `crates/swadesh-cli/src/dataset.rs` module: resolve
`<root>/<set>/<set>.<pair>.<field>`, discover pairs and fields, read a field's lines, expose the
field list. Used by both user sets (WS-1) and built-in sets (WS0.2). **Acceptance:** unit tests for
resolution order, field discovery, a missing field, and a pair with several references; the WS-1
acceptance byte-diff passes. *(1 d.)*

**WS0.2 — `-t/--test-set` for the built-in catalogue.** `-t NAME [-l PAIR]` resolves from
`datasets.rs`, downloads when the cache is cold (WS0.3), and scores. **Acceptance:**
`tests/test_cli.py -k "dataset or test_set"` byte-diffs `-t wmt16 -l en-de` scoring against the
oracle (previously skipped); `-t nonexistent-set` and `-t wmt16 -l en-xx` match the oracle's stderr
byte for byte; `--list` still matches. *(2 d.)*

**WS0.3 — The downloader.** `ureq`-class GET with redirects, the progress/notice lines the oracle
prints, md5 verification, `flate2`+`tar`+`zip` extraction into `raw/`, behind
`features = ["downloader"]`. **Acceptance:** in a scratch `SACREBLEU` directory,
`swadesh -t wmt16 -l en-de --echo src` reproduces the oracle's stdout *and* leaves the same cache
layout; a corrupted archive (flip a byte) produces the oracle's failure text and a non-zero exit;
`cargo build -p swadesh-cli --no-default-features` still builds and passes the core suites. *(3 d.)*

**WS0.4 — `--download NAME -l PAIR`.** Pre-fetch and exit, no scoring. **Acceptance:** byte-diff
against the oracle including the "already cached" path and the missing-`-l` failure. *(0.5 d.)*

**WS0.5 — `--echo FIELD...`.** src, ref, ref:A/ref:B, docid, origlang, genre, domain, a system
field, several fields tab-separated, `all`. **Acceptance:** the acceptance list in
`docs/spec/user-test-sets.md` passes byte-for-byte against the oracle, for wmt16 and wmt22 (multiple
references and a system field). *(1 d.)*

**WS0.6 — `-ol/--origlang` and `--subset`.** Segment filters on the `origlang` and `genre`/`domain`
fields, including sacreBLEU's `non-X` form and its count behaviour. **Acceptance:** byte-diff for
`-ol en`, `-ol non-en`, `--subset <name>` and both together, plus the error when the field is absent.
*(1 d.)*

**WS0.7 — `-d/--detail`.** The per-`origlang` x domain breakdown after the score line. **Acceptance:**
byte-diff of the measured shape (`origlang=de domain=ALL : sentences=1499   BLEU = 100.0`) for BLEU,
chrF and TER, for a set with and without the metadata fields. *(1.5 d.)*

**WS0.8 — The remaining small divergences.** (a) `--help`/`-h`: transcribe the oracle's 149-line help
(usage already matches); (b) `-e` with a non-UTF-8 codec; (c) `.gz` input; (d) the consumed-token list
in argparse errors; (e) `-j > 1` worker status lines; (f) negative `--chrf-word-order`; (g) re-run the
real-model verification (qwen3.8-27b / llama3.3:70b harness) against the current binary.
**Acceptance:** for (a)-(c) a byte-diff test replaces each hint; for (d)-(f) either the divergence
disappears from `docs/parity-report.md` or its strict `xfail` stays with an updated note; for (g) the
recorded 150-segment en-de comparison reports zero mismatches. *(3 d.)*

### WS-1 — The data layer and user test sets (spec: `docs/spec/user-test-sets.md`)

**WS1.1 — Resolution roots and discovery.** `$SWADESH_TEST_SETS` (colon-separated), a `-t` value
containing a separator used as a directory, built-ins first. **Acceptance:** the spec's acceptance
list, with `SWADESH_TEST_SETS=$SACREBLEU`, passes byte-for-byte. *(1 d.)*

**WS1.2 — `-t NAME --list` for a user set.** The discovered pairs and fields in the spec's fixed
order. **Acceptance:** a fixture root produces the documented line; a pair with no files produces the
oracle's language-pair error. *(0.5 d.)*

**WS1.3 — `--citation` from a user set.** A `.citation` file prints; otherwise the oracle's refusal
wording. **Acceptance:** both paths tested; the built-in path keeps its documented divergence. *(0.5 d.)*

**WS1.4 — Docs and examples.** `docs/guide.md` gains "score your own data"; `examples/` gains a
directory-based set with metadata. **Acceptance:** the example runs in CI and produces the documented
output. *(0.5 d.)*

**WS1.5 — A `swadesh-format` checker.** `swadesh check-set DIR` validates a user set: line counts
agree, no empty lines, no mojibake, `origlang`/`domain` are single tokens, references look like the
same language. **[swadesh-only]** **Acceptance:** a deliberately broken set produces one message per
problem and exit 1; a good set exits 0 with a summary. *(1 d.)* — also the substrate for WS5.3.

### WS-2 — Packaging, publishing, governance

**WS2.1 — Public repository and hygiene.** Flip `apeters247/swadesh` public; add
`CODE_OF_CONDUCT.md`, `SECURITY.md`, `CITATION.cff`, issue/PR templates, and a rewritten
`CONTRIBUTING.md` (the loop in §1.5, the DCO/CLA requirement, "parity first"). **Acceptance:** a
fresh clone shows the templates; `git log` has no secrets (`tools/secret_scan.sh`); `CITATION.cff`
validates against the schema. *(1 d.)*

**WS2.2 — Real release pipeline.** Rewrite `.github/workflows/release.yml`: on a `v*` tag, build
wheels (manylinux + macOS arm64/x86 + Windows, Python 3.10-3.14), an sdist, static musl + macOS +
Windows binaries, a Docker image, then publish to PyPI via trusted publishing and attach the
artefacts to the GitHub release; `cargo publish` for `swadesh-core` and `swadesh-cli`; `npm publish`
for the wasm package. **Acceptance:** a dry run on `v0.2.0-rc1` produces every artefact and a
`tools/smoke_install.sh` pass for each. *(2 d.)*

**WS2.3 — The stranger test.** `tools/smoke_install.sh`: in a clean container, install by pip, by
downloaded binary, and by Docker, then score a fixed tiny corpus in each and compare to a recorded
expected score byte-for-byte. **Acceptance:** it runs in CI per release and is referenced from the
README. *(0.5 d.)*

**WS2.4 — Version string and identity.** Decide and document what `--version` prints (currently
`swadesh 0.1.0 (sacreBLEU 2.6.0)`, which differs from the oracle's `sacrebleu 2.6.0`); keep the
compat version separate and machine-readable (`swadesh --compat-version`). **Acceptance:** both are
documented in `docs/spec/cli-contract.md` and pinned by a test. *(0.5 d.)*

**WS2.5 — DOI and citations.** Zenodo integration per release; `CITATION.cff`; a "Cite this
repository" section in the README; `swadesh --cite-tool` printing the tool's own BibTeX (kept
separate from `--citation`, which must stay sacreBLEU-exact). **Acceptance:** the first DOI resolves;
`--cite-tool` output pastes into a LaTeX file and compiles. *(0.5 d.)*

**WS2.6 — Contribution governance.** CLA enforcement on PRs (CLA assistant or DCO fallback), a
"parity test required" checklist item, and a policy that a PR changing a default output needs a
parity-suite delta in its description. **Acceptance:** a test PR is blocked until the CLA is signed;
the template asks for the delta. *(0.5 d.)*

**WS2.7 — Licences and notices for a public tree.** Extend `tests/test_dependency_licences.py` to all
new crates; generate a third-party notice (`cargo about` or an equivalent script) into
`docs/third-party.md`; keep `NOTICE` accurate (the Apache-2.0 disclosure and the ported Python layer
are already there). **Acceptance:** the licence test and a "notices are current" check run in CI;
`docs/licensing.md` names every dependency family and the commercial boundary. *(1 d.)*

**WS2.8 — Docs site.** A static site (mdBook or MkDocs) built from `docs/`, deployed on release, with
the guide, specs, benchmark page, and API reference. **Acceptance:** the site builds in CI and every
internal link resolves (`tools/check_links.py`). *(1 d.)*

### WS-3 — Language and system hooks

**WS3.1 — A versioned JSON contract.** Document the JSON that already exists (`-f json`, the results
receipt of WS4.5) as a *stable interface*: `docs/json-contract.md` with a schema, a version field and
a compatibility promise. **Acceptance:** a schema file validates every JSON the suite produces; the
doc states the promise and the test that enforces it. *(1 d.)*

**WS3.2 — The C ABI.** `crates/swadesh-capi`: a `cdylib`/`staticlib` exposing opaque handles
(create/destroy a metric, add segments, read a score/signature), no panics across the boundary, an
error struct with a message buffer; `include/swadesh.h`; `cargo build --release -p swadesh-capi`
produces a shared library on all three OSes. **Acceptance:** a C example links and scores identically
to the CLI; a fuzz pass calls every entry point with junk and gets errors, not aborts. *(3 d.)*

**WS3.3 — Hook examples, one per language, all in CI.** C, Go (cgo), Java (JNA or JNI), .NET
(P/Invoke), Ruby (FFI), R (`.Call`), each a ~30-line program; plus a shell/CLI example and a Node
example against the wasm build. **Acceptance:** CI builds and runs each and each prints the same
score and signature as the CLI for a fixed input; the results are in a table in
`docs/hooks.md`. *(2 d.)*

**WS3.4 — npm package for wasm.** Publish `swadesh-wasm` (bundler + Node targets) with TypeScript
types and a README. **Acceptance:** `npm install` in a clean dir then scoring a fixed corpus in Node
matches the CLI; the package is in the release pipeline. *(1 d.)*

**WS3.5 — An optional HTTP service.** `swadesh serve --port` exposing `/score`, `/compare`,
`/receipt` over the JSON contract, for pipelines that cannot exec a binary. **[swadesh-only]**
**Acceptance:** a container test scores a corpus over HTTP and matches the CLI; the service is behind
a cargo feature and not built by default. *(1.5 d.)*

**WS3.6 — MCP: the agent verbs.** Extend `python/swadesh/mcp_server.py` with `score_translation`,
`compare_answers`, `pick_best_of_n`, `is_delta_real`, `explain_score`, `audit_references`; ship a
copy-paste `SKILL.md` for Claude Code / Cursor / Copilot. **Acceptance:** each tool is exercised by
`tests/test_mcp.py`; the skill file is documented in `docs/guide.md` and the README. *(1 d.)*

### WS-4 — Researcher features

**WS4.1 — Significance done properly.** Holm-Bonferroni and Bonferroni correction over a family of
paired tests, exposed as a flag and in the JSON. **[swadesh-only]** **Acceptance:** a synthetic family
where the corrected decision differs from the raw one; the correction is documented with its
definition in `docs/spec/significance.md`. *(1.5 d.)*

**WS4.2 — Power, not just p.** `swadesh power --segments N --alpha a [--effect d]` → the minimum
detectable difference, and the segments needed for a given delta, from the bootstrap's own variance.
**[swadesh-only]** **Acceptance:** on a fixed corpus the function is monotone in N; a worked example
in the docs prints the number for a real set. *(1 d.)*

**WS4.3 — Document-level and context-aware scoring.** Group segments by `docid`; document-level BLEU,
chrF and TER; an optional sliding-window context for tokenization. **[swadesh-only]** **Acceptance:**
document-level scores match a hand-computed reference on a small fixture; the window option has a
test proving the segment-level default is unchanged. *(3 d.)*

**WS4.4 — Metric meta-evaluation.** `swadesh meta-eval --judgements FILE --systems A,B` computing
segment- and system-level Kendall tau and Spearman for each metric against human judgements, reading
the WMT metrics-task JSON shape. **[swadesh-only]** **Acceptance:** on a recorded metrics-task file the
correlations match an independently computed reference (scipy, in the test); the docs show a
one-command worked example. *(3 d.)*

**WS4.5 — Reproducibility receipt.** A results file carrying the signature, the test-set name, the
*hashes of the reference and hypothesis streams*, the command line, the tool version and the compat
version; `swadesh verify results.json` re-runs and reports any divergence. **[swadesh-only]**
**Acceptance:** a receipt round-trips; tampering with one hypothesis line makes `verify` fail with the
segment index; another tool's receipt with a different iterable order still verifies. *(2 d.)*

**WS4.6 — Error analysis export.** `swadesh worst --n 20` and `swadesh align` printing, per segment,
the score, the delta against a baseline and (for TER) the accepted shifts; TSV and JSON out; builds on
`TER.explain()` from `docs/feature-ideas.md`. **[swadesh-only]** **Acceptance:** the TSV loads in
pandas with the documented columns; the alignment for a hand-checked segment matches. *(2 d.)*

**WS4.7 — The rest of the API ideas.** `TER.sentence_scores`/`pairwise`; `n_jobs` on every corpus
call; the streaming accumulator; `compare(systems, refs)`; `strict=True` validation; seeds in every
signature. **[swadesh-only]** **Acceptance:** each item from `docs/feature-ideas.md` §1, §4, §5, §6,
§7, §8 has a test and a docs line; the parity suite is unchanged. *(3 d.)*

### WS-5 — Developer and agent features

**WS5.1 — The CI quality gate.** `swadesh gate baseline.json --fail-under 0.4 [--metric chrf]` →
exit 1 with a readable table when the candidate regresses. **[swadesh-only]** **Acceptance:** a
regressing candidate exits 1, a passing one exits 0, and both are tested; the JSON baseline is a
WS4.5 receipt. *(1 d.)*

**WS5.2 — The GitHub Action.** `action.yml` wrapping the CLI (install the binary, run the gate, post
a comment). **Acceptance:** an example repo in `examples/ci/` runs it; a deliberate prompt change
fails the job. *(1 d.)*

**WS5.3 — Reference auditing.** `swadesh audit refs.txt` flagging empty, duplicate, wrong-language,
mojibake, misaligned and degenerate segments (uses the tokenizers and the statistics).
**[swadesh-only]** **Acceptance:** a fixture of 20 hand-labelled defects is caught with the documented
codes; a clean file is silent. *(1.5 d.)*

**WS5.4 — Human-readable reporting.** `swadesh report --format md` with the score, the interval, the
per-domain breakdown, the three worst segments and a one-line verdict; a `--human` alias for the
common case. **[swadesh-only]** **Acceptance:** the Markdown renders correctly in `docs/` and the
default output is provably unchanged. *(1 d.)*

**WS5.5 — Notebook and pandas helpers.** `swadesh.pandas`: score a DataFrame column, return a
DataFrame; plus a Colab-friendly example. **[swadesh-only]** **Acceptance:** the example notebook runs
end to end in CI (nbval or an executed copy) and returns the documented columns. *(1 d.)*

**WS5.6 — Agent-first quickstart.** A README section and `docs/guide.md` chapter: "ask your agent to
score this translation" with the MCP setup, the skill file, and the gate. **Acceptance:** a
copy-paste block from the docs, executed verbatim in CI, works. *(0.5 d.)*

### WS-6 — Hugging Face integration

**WS6.1 — An `evaluate` module.** A metrics module usable as `evaluate.load("swadesh")` and
documented alongside `patch()`. **Acceptance:** in a clean env, `evaluate.load("swadesh")` scores and
matches the CLI byte-for-byte; the module is in the wheel. *(1.5 d.)*

**WS6.2 — A lighteval plugin.** A task/metric registration so HF's harness can compute BLEU/chrF/TER
through swadesh. **Acceptance:** a lighteval run on a small task prints a score identical to the CLI;
a snippet in the docs reproduces it. *(2 d.)*

**WS6.3 — Data via the Hub.** Document (and test) the route where the user fetches a set with the
`datasets`/`huggingface_hub` tooling and points `SWADESH_TEST_SETS` at the snapshot, so `-t` works
with no downloader in the binary; plus a licence review before any mirror is published (D-7).
**Acceptance:** the documented route works in a clean env; the licence note is in
`docs/licensing.md`. *(1 d.)*

**WS6.4 — A Space and a model-card snippet.** A Gradio/HF Space running the wasm playground, and a
copy-paste model-card "metrics" block carrying the full signature. **Acceptance:** the Space is live
and the snippet appears in an example model card. *(1 d.)*

### WS-7 — Research, recognition, community

**WS7.1 — The findings note.** A short paper (arXiv) on the three things reproducing sacreBLEU
exactly uncovered: the two-character `_UCODE_RANGES` entries that make U+2001-U+2A6D "Chinese",
CPython 3.12/3.14 changing `sum()` and moving 15.8% of sentence BLEUs, and the float32 `--paired-bs`
crash. Numbers from `docs/python-version-drift.md` and the notebook; the tool is the artefact.
**Acceptance:** the note is public with a DOI and cites measured numbers from this repo. *(3 d.)*

**WS7.2 — Posts and teaching material.** Publish `docs/blog-post.md`, `docs/how-its-exact.md` and the
torture report; write "evaluating MT properly with swadesh" for the HF/blog audience.
**Acceptance:** each is live and linked from the README and the docs site. *(2 d.)*

**WS7.3 — The sacreBLEU upstream conversation.** Send the drafted `docs/outreach/sacrebleu-issue.md`
(with the corrected findings) and offer the bit-exactness test corpus. **Acceptance:** the issue is
sent and its outcome recorded in the notebook, favourable or not. *(0.5 d.)*

**WS7.4 — A metrics-task submission.** Prepare and submit to the WMT metrics shared task using
swadesh's native metrics, or a paper-style evaluation of metrics with `swadesh meta-eval`.
**Acceptance:** a submission or an arXiv preprint exists, with the swadesh version and DOI recorded.
*(5 d.)*

**WS7.5 — The benchmark page and "used by" evidence.** A generated page with the head-to-head numbers
(machine named, command recorded) and a maintained list of projects that use swadesh.
**Acceptance:** `bench/plot.py` output is on the docs site; the list has entries with links. *(1 d.)*

**WS7.6 — Adoption metrics.** A small script that records weekly PyPI downloads, GitHub stars,
citation count (OpenAlex/Crossref), and Space views into `docs/adoption.md`, appended rather than
overwritten. **Acceptance:** the file has a dated row per run and is refreshed monthly. *(0.5 d.)*

### WS-8 — Quality, operations, performance

**WS8.1 — CI covers what the notebook covers.** Add jobs for the CLI parity suite (`test_cli.py` with
`SWADESH_DATA`), the frozen oracle, provenance, licence and version tests, the MCP tests, and the
three-OS binary build. **Acceptance:** CI on a PR runs every suite named in §4 and fails on any
failure; the job list is documented in `TESTING.md`. *(1 d.)*

**WS8.2 — Fuzzing.** A `cargo-fuzz` target per parser and per tokenizer, plus the existing hypothesis
suite; a nightly job with a time limit; every crash becomes a regression test. **Acceptance:** the
nightly run has been clean for a week, with the corpus committed. *(1.5 d.)*

**WS8.3 — Mutation testing as a budget.** Keep `tests/test_mutation_kills.py` and
`docs/mutation-report.md` current; a stated kill-rate floor per crate. **Acceptance:** the report
records the current rate and CI fails below the floor. *(1 d.)*

**WS8.4 — Performance budgets.** A CI benchmark job on a named runner tracking the corpora in
`bench/`, with a ±5% watchdog and a stored history. **Acceptance:** a deliberate regression fails the
job with the delta printed; the README's numbers are generated from it. *(1.5 d.)*

**WS8.5 — Memory envelope.** `bench/stress/` in CI at a stated size with an RSS ceiling, so the
streaming work (WS4.7) is measurable. **Acceptance:** the recorded envelope table is regenerated per
release and the ceiling is asserted. *(1 d.)*

**WS8.6 — Supply chain.** `cargo-deny` (advisories, licences, bans), pinned `Cargo.lock`, Dependabot
or Renovate, and the licence test extended to every new crate. **Acceptance:** CI runs `cargo deny
check`; a deliberately added GPL dependency fails it. *(0.5 d.)*

**WS8.7 — The divergence register.** Every known divergence lives in exactly one place
(`docs/parity-report.md` § Known remaining differences) with a test id, a reason and a date; a script
checks the register against the `xfail` markers in the suite. **Acceptance:** the script runs in CI
and fails if an `xfail` exists without a register row (or vice versa). *(1 d.)*

---

## 7. The measurement harness (what every milestone reports against)

**Reporting format.** Every task closes with a row in `docs/lab-notebook.md` and, where it changes
parity, a row in `docs/parity-report.md`:

```
task: WS0.2   command: pytest tests/test_cli.py -k dataset -q   result: 31 passed, 0 failed
             (was: 31 skipped, no data)                        artefact: commit 1234abc
```

**Floors.** Suite counts may only grow. `cargo test --release --workspace` reports 0 failed;
`tests/test_cli.py` reports 0 failed; the Python selection reports 0 failed and its skip count must
*fall* as the data questions are settled, never rise silently.

**Floor table (from §4).** Any run below a floor is a regression: stop, fix, or record a documented
divergence (§1.6). The three current strict `xfail`s may only disappear by making the divergence
disappear; adding a `xfail` requires a register row (WS8.7).

**Parity gates per changed area.**

| Change | Required evidence |
|---|---|
| Any scoring path | a byte-diff against the oracle, plus the frozen-oracle suite green |
| Any new CLI flag | a byte-diff for the flag *and* for its error paths, both orders of mutually exclusive flags, and the `--help`/usage text if it changes argument surface |
| Any new output field | the JSON contract test (WS3.1) and the schema file |
| Any dependency | `cargo deny`, the licence test, `docs/third-party.md`, `NOTICE` |
| Any performance change | the CI benchmark job's ±5% gate plus a notebook line naming the machine |

**Adoption yardsticks (soft, but recorded).** Weekly PyPI downloads, GitHub stars, Crossref/OpenAlex
citation count, Space views, and the "used by" list (WS7.6). Targets are set at M1
(`docs/adoption.md` gets a target column) and reviewed monthly; they never gate a release.

## 8. Risk register

| ID | Risk | Mitigation |
|---|---|---|
| R1 | A new feature silently changes a default output | The parity suite is a required CI job; every feature task names the default output it must not touch; `tests/test_cli.py` byte-diffs the CLI |
| R2 | Dependency creep erodes the standalone property | Feature gates (D-2), `cargo deny`, the licence test, and the `--no-default-features` build kept green in CI |
| R3 | A published artefact is broken or wrong | `tools/smoke_install.sh` per channel (WS2.3), a release checklist (§9), and **no `v*` tag until it passes** |
| R4 | Third-party data licensing | D-7: never redistribute corpora; fetch from the catalogue, hash-verify, and review any mirror separately |
| R5 | sacreBLEU drifts (a future release) | D-5: the compat version is pinned and reported; a new target is a new milestone with re-frozen fixtures |
| R6 | Reviewers cannot tell AGPL from the commercial path | `docs/licensing.md` states the boundary; `NOTICE` lists every third-party obligation; the README has one paragraph, not a maze |
| R7 | Maintainer bandwidth | Tasks are sized in days; a milestone is shippable at any point; community tasks (WS3.3, WS7.2) are labelled as good first issues |
| R8 | The plan rots as the tree moves | §10 is the tracker; `WS8.7`'s register script and the floors make drift visible; each milestone rewrites its own section |
| R9 | Name/trademark | The README explains the Swadesh-list name; no implication of affiliation with sacreBLEU beyond compatibility |

## 9. Release checklist (this replaces the old "never publish" note)

1. `cargo test --release --workspace` and the Python selection: 0 failed (WS8.1 job green).
2. `tests/test_cli.py` green with `SWADESH_DATA=bench/data`; every CLI parity case byte-diffed.
3. The divergence register (§8, WS8.7) has no undated or unpinned row.
4. Licence and notice currency: `cargo deny` clean, `docs/third-party.md` regenerated, `NOTICE` and
   `docs/licensing.md` accurate.
5. Benchmarks re-run on the named machine; README numbers regenerated (no hand-typed figures).
6. `CHANGELOG.md` has the version section, written from the notebook, with the compat version.
7. `tools/smoke_install.sh` passes for every channel *before* the tag exists (build from the branch,
   not the tag).
8. Tag `vX.Y.Z`, let the release workflow build and publish, then re-run the smoke test against the
   *published* artefacts (PyPI, binaries, Docker, npm, crates.io).
9. Mint the DOI (Zenodo) and record it in `CITATION.cff`, the README and `CHANGELOG.md`.
10. Append the adoption row (WS7.6) and a notebook entry naming the version and the numbers.

## 10. Task tracker

Update the status column as tasks finish; keep one task `doing` at a time. Effort is in days and is a
planning aid, not a commitment.

| ID | Milestone | Task | Est | Status |
|---|---|---|---|---|
| WS1.1 | M0 | Resolution roots and discovery | 1 | **done** |
| WS1.2 | M0 | `-t NAME --list` for a user set | 0.5 | **done** |
| WS1.3 | M0 | `--citation` from a user set | 0.5 | todo |
| WS1.4 | M0 | Docs and examples for own data | 0.5 | todo |
| WS1.5 | M0 | `swadesh check-set` validator | 1 | todo |
| WS0.1 | M0 | Shared dataset reader | 1 | **done** |
| WS0.2 | M0 | `-t` for the built-in catalogue | 2 | **part done** (cached sets; the cold path needs WS0.3) |
| WS0.3 | M0 | The downloader (feature-gated) | 3 | todo |
| WS0.4 | M0 | `--download NAME -l PAIR` | 0.5 | todo |
| WS0.5 | M0 | `--echo FIELD...` | 1 | **done** |
| WS0.6 | M0 | `-ol/--origlang`, `--subset` | 1 | **done** |
| WS0.7 | M0 | `-d/--detail` breakdown | 1.5 | **done** |
| WS0.8 | M0 | Small divergences + real-model re-verify | 3 | todo |
| WS2.1 | M1 | Public repo and hygiene | 1 | todo |
| WS2.2 | M1 | Release pipeline | 2 | todo |
| WS2.3 | M1 | Stranger test | 0.5 | todo |
| WS2.4 | M1 | Version identity | 0.5 | todo |
| WS2.5 | M1 | DOI and citations | 0.5 | todo |
| WS2.6 | M1 | Contribution governance | 0.5 | todo |
| WS2.7 | M1 | Licences and notices | 1 | todo |
| WS2.8 | M1 | Docs site | 1 | todo |
| WS3.1 | M2 | Versioned JSON contract | 1 | todo |
| WS3.2 | M2 | The C ABI | 3 | todo |
| WS3.3 | M2 | Hook examples (6 languages) | 2 | todo |
| WS3.4 | M2 | npm package for wasm | 1 | todo |
| WS3.5 | M2 | Optional HTTP service | 1.5 | todo |
| WS3.6 | M2 | MCP agent verbs + skill | 1 | todo |
| WS4.1 | M3 | Multiple-comparison correction | 1.5 | todo |
| WS4.2 | M3 | Power analysis | 1 | todo |
| WS4.3 | M3 | Document-level and context scoring | 3 | todo |
| WS4.4 | M3 | Metric meta-evaluation | 3 | todo |
| WS4.5 | M3 | Reproducibility receipt + `verify` | 2 | todo |
| WS4.6 | M3 | Error analysis export | 2 | todo |
| WS4.7 | M3 | The remaining API ideas | 3 | todo |
| WS5.1 | M4 | CI quality gate | 1 | todo |
| WS5.2 | M4 | GitHub Action | 1 | todo |
| WS5.3 | M4 | Reference auditing | 1.5 | todo |
| WS5.4 | M4 | Human-readable reporting | 1 | todo |
| WS5.5 | M4 | pandas/notebook helpers | 1 | todo |
| WS5.6 | M4 | Agent-first quickstart | 0.5 | todo |
| WS6.1 | M5 | `evaluate` module | 1.5 | todo |
| WS6.2 | M5 | lighteval plugin | 2 | todo |
| WS6.3 | M5 | Data via the Hub | 1 | todo |
| WS6.4 | M5 | Space + model-card snippet | 1 | todo |
| WS7.1 | M6 | The findings note | 3 | todo |
| WS7.2 | M6 | Posts and teaching material | 2 | todo |
| WS7.3 | M6 | sacreBLEU upstream conversation | 0.5 | todo |
| WS7.4 | M6 | Metrics-task submission | 5 | todo |
| WS7.5 | M6 | Benchmark page + used-by | 1 | todo |
| WS7.6 | M6 | Adoption metrics | 0.5 | todo |
| WS8.1 | all | CI covers the notebook | 1 | todo |
| WS8.2 | all | Fuzzing | 1.5 | todo |
| WS8.3 | all | Mutation budget | 1 | todo |
| WS8.4 | all | Performance budgets | 1.5 | todo |
| WS8.5 | all | Memory envelope | 1 | todo |
| WS8.6 | all | Supply chain | 0.5 | todo |
| WS8.7 | all | Divergence register | 1 | todo |

**Total: ~78 days of work**, of which M0 is ~14.

## 11. How an agent works here

1. **Pick** the lowest unfinished task in the current milestone (§10), or the one named by the owner.
2. **Read** the relevant spec (`docs/spec/*.md`) and `docs/parity-report.md`'s divergence list. If the
   task needs a rule that is not written down, write the spec/measurement first, in its own commit.
3. **Measure the oracle** before writing code when the task mirrors sacreBLEU. Never read sacreBLEU's
   Python for anything except the one disclosed exception already in `NOTICE`.
4. **Implement in Rust**, in `swadesh-core` unless it is I/O. Keep the CLI's report formatting in
   `swadesh-cli`. Python changes are adapters only.
5. **Verify**: the narrow suite for the area, then `cargo test --release --workspace`, then the Python
   selection (§4). For a CLI change, byte-diff. For a new flag, also its errors.
6. **Commit** one slice with a message that states what changed and the numbers; push to the private
   remote (or `origin` once WS2.1 lands). Never commit a red suite.
7. **Record** the notebook entry: attempted, measured numbers, failures and why, decisions, next step.
8. **Update** §10 and, if parity moved, `docs/parity-report.md`.

**Blocked?** If a task needs a dependency, a spec rewrite, or a sacreBLEU change not covered here:
write the reason into the notebook, mark it `blocked` in §10 with the decision needed, and move to the
next task. Do not widen scope silently.

**Where the older documents fit.** `docs/feature-ideas.md` items map to WS4.7 (§1, §4, §5, §6, §7,
§8), WS4.6 (§3) and WS3.5 (§2). `TESTING.md` keeps the criteria table and is rewritten as suites move
(WS8.1). `docs/parity-report.md` stays the single divergence register (WS8.7). `docs/spec/*.md` are
the contracts; `docs/lab-notebook.md` is the history; this file is the plan.

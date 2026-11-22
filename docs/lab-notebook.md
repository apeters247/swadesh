# swadesh lab notebook

One dated entry per work session: what was attempted, what was measured, what failed and why,
decisions and their reason, and the next step. Entries are written when the work happens, not
reconstructed afterwards.

---

## 2026-11-22 — Phase 0: freeze the oracle

**Attempted.** Build `tools/freeze_oracle.py` and record sacreBLEU 2.6.0's exact outputs as data, so
the test suite stops depending on sacreBLEU at runtime. This is the foundation for the clean-room
work: without it, every "the rewrite is exact" claim is verified by running the code we are trying
to stop depending on.

**Built.** `tools/freeze_oracle.py` records, per case: corpus score as `float.hex()`, `str(score)`,
`format(signature=...)`, the verbose line, every sentence score as hex, and exception type+message
where the oracle raises. Cases:
- WMT22 en-de, de-en, en-zh, zh-en, en-ja, en-ru, cs-en (first 400 segments, deterministic
  perturbations seed 1234)
- 200 fuzz cases over the adversarial alphabet used by `test_differential.py` (seed 99, 1–2 refs)
- the five real-model runs (qwen2.5vl 600, granite4/devstral/qwen3.8/Naxi 150/250) under
  `tests/corpus/model/`
- eight metric variants: BLEU, BLEU floor 0.1, BLEU intl, chrF, chrF beta=3, chrF++, TER, TER normalized

`tests/test_frozen_oracle.py` replays all of it against swadesh.

**Measured.** `tests/oracle/`: 14 files, 1.5 MB. `tests/corpus/model/`: 980 KB.
`pytest tests/test_frozen_oracle.py` → **14 passed** (7 WMT files + fuzz + 5 model files + presence).
Full check: `pytest tests/test_parity.py tests/test_frozen_oracle.py tests/test_mcp.py
tests/test_numrefs_none_refs.py` → **517 passed**.

**What failed, and why (worth remembering).**
1. The freezer wrote two different BLEU variants under the same case name (`BLEU`), so replay could
   not tell floor from exponent. Fixed by naming variants distinctly in both files.
2. Case names are `<pair>:<variant>` (`en-ja:BLEU`) without the `wmt22.` prefix, while the replay
   assumed the prefix. Fixed by detecting the pair from the data files instead of the name.
3. `strip-components=1` unpacked the model corpus one level too high, so `tests/corpus/model` held
   only the Naxi reference and the freezer silently skipped all five model cases on the first run.
   The skip was silent — that is the dangerous kind. Fixed by checking `ls` after every unpack.
4. The Naxi reference was initially the WMT24 German references. That run was scored against the
   **English source** (see the earlier evidence sheet), so `ref-nxkd1-k-q5.txt` is `src.txt`.
5. `pytest.mark.parametrize(..., ids=lambda p: p.stem)` binds the integer index, not the path, when
   the list is built at import. Use an explicit `ids=[...]` and a module-level `_FILES` list.

**Process notes.** The VPS repo is owned by `andrew`; `kpeters` can read but not write, so every
install is staged to `/tmp`, then `cp` as `andrew` via `ssh vps-admin`. Files piped straight into a
backgrounded `cat >` get truncated — always pipe to `/tmp` first, check the size, then copy.

**GitHub.** `apeters247/swadesh` now returns 404 to anonymous callers, which is how GitHub presents a
private repo. The VPS has no GitHub credentials (`git@github.com` → permission denied, no `gh`), so
pushing to the private repo needs a deploy key before Phase 1 can be committed upstream.

**Next.** Phase 1: write `docs/spec/tokenizers.md` from the published rules (Moses/SacreBLEU
documentation, plus spertiniite as a cross-check only — never copied), commit it alone, then
reimplement `tokenize.rs`/`unicode.rs` from that spec, and only then delete the port.
---

## 2026-11-22 (later session) — Phase 1, spec first

**Attempted.** Write `docs/spec/tokenizers.md` from published sources only, and commit it alone
before touching any tokenizer code. This commit is the provenance record for the clean-room claim:
it must exist, and be dated before, the implementation.

**Built.** 162-line spec covering `none`, `char`, `13a`, `intl`, `zh`, lowercasing, whitespace and
the acceptance criteria. Sources: Koehn & Knowles 2017 (the 13a rule set), the Moses tokenizer
manual, Unicode UAX #29 and the character database, the Python `str` semantics the oracle depends
on, and the BLEU/chrF papers. `spertiniite` was checked only as an independent confirmation that
the rule set matches; no code was taken from it.

**What the spec had to record as a deliberate defect.** The `zh` range table end point is the
two-character string `u'\u20000'`, which admits U+2001–U+2A6D (arrows, math, currency) as Chinese.
The spec states this explicitly, because a consumer "fixing" it would change scores.

**What failed / was corrected.** The first draft asserted that `intl` has no separate `%` rule and
that its class is `\p{P}\p{S}\p{N}` wholesale. Probing the oracle showed both are wrong:
`"100%"` stays `100%` under `intl` but becomes `100 %` under `13a`, and `→` is split by `intl` but
not by `13a`. Three inferences were wrong in a document whose whole point is being right; the
spec now records the observed behaviour, with the observations quoted.

**Measured.** `docs/spec/tokenizers.md` 162 lines, committed as `0f77954` with no code change.
Oracle probe (6 inputs × 3 tokenizers) saved in the session log.

**Next.** Phase 1 implementation: reimplement `tokenize.rs` and `unicode.rs` from the spec, verify
against the frozen corpus (zero mismatches), then delete the port in a separate commit.

**Blockers carried forward.** No GitHub credentials on the VPS, so nothing is pushed; two Phase 0
commits (`9f9e6e3`, `0f77954`) are local only. Deploy key still needs to be added by the owner.

---

## 2026-11-22 — Phase 1 implementation, partial (swarm of 4)

**Attempted.** Reimplement the tokenizers from the spec, one subagent per tokenizer.

**Delivered and verified.**
- `tokenize_nonechar_clean.rs` — `none` and `char`. 8/8 unit tests pass; the agent
  differentially tested 31,552 strings (hex-encoded output equality) against sacreBLEU as a
  black box with zero mismatches. Wired into `lib.rs` and committed.
- `tokenize_zh_clean.rs` — installed but **not accepted**: 6/7 tests, and the failing one is the
  `.`/`,` number rule. Its author measured 23/2037 mismatches on WMT22 en-zh and 130/1875 on
  zh-en, and traced them to hyphens and decimals. Left out of the commit.

**Not delivered.** `intl` (agent returned a one-line reply and no file) and `13a` (agent wrote
nothing, reporting it ran on the OMEN laptop with a different repo layout and no access to the
frozen corpus).

**Spec errors found by measurement, all corrected in `docs/spec/tokenizers.md`.**
1. The `char` postcondition is false: `char` keeps whitespace as tokens (`"ab cd"` → `"a b   c d"`),
   collapsing would change scores.
2. `13a`'s class is not `\p{P} \p{S}`: it is exactly 30 ASCII characters, and non-ASCII punctuation
   (`§ ° £ « » —`) is never split. `\p{N}` has no observable effect beyond ASCII digits.
3. The `.,-` rules are a Moses-style numeric-key rule, not the naive "non-digit + `.`" I spec'd;
   `12,34` stays whole, `5.,6` splits, and the write-boundary semantics do not decompose into
   independent passes.

That is three specifications in a row where the document I wrote from published sources was wrong
and measurement corrected it. The lesson is not "the spec is useless" — it is that the spec must be
written *with* the oracle in hand and corrected before implementation, which is what now happens.

**Toolchain.** The `zh` agent claimed the VPS rustc was a broken custom build. It is stock rustup
1.100.0; its failing test reproduces here honestly, and its `none`/`char` sibling compiled and
passed 8/8 on the same toolchain. Treat that claim as unfounded.

**Swarm lesson.** Two of four subagents ran in a place that could not see the repo and produced
nothing usable, without failing loudly. For Phase 2–3, give each subagent an explicit first step:
"`ssh vps-public 'ls /var/www/llmeval'` and confirm it works; if not, stop and report" before any
code is written.

**Next.** `13a` is the hard one and needs the numeric-key rule pinned down against the frozen
corpus. Get `intl` and `13a` done, then run `tests/test_frozen_oracle.py` and delete the port only
after zero mismatches.

---

## 2026-11-22 — Phase 1: `intl` attempted, blocked on the regex semantics

**Attempted.** Implement `intl` directly (not delegated) from the spec plus black-box measurement.

**Measured, and now settled:**
- The splitting class is exactly `\p{P} ∪ \p{S}`: of the 9,588 non-ASCII code points that split in
  `a<c>b`, the set equals the `regex` module's `\p{P}|\p{S}` with **0 differences in either
  direction**; the ASCII members are the 32 characters `!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~`. So the
  original spec wording was right and my earlier "intl class is not P|S" note was wrong.
- Trailing whitespace is dropped before tokenising: `"1986"` vs `"1986 "` give different output.

**Blocked on.** The `.`/`,`/`:` and bracket grouping is context- and position-sensitive in a way I
could not derive in one session:
- `1986.` → `1986.` but `1986. ` → `1986 .`
- `2. x` → `2 . x`, while `5,000.50` → `5,000.50`
- `(4)` → `(4)` at line start, but `a (4) b` → `a ( 4 ) b`
- `(4)(5)` → `(4 ) ( 5)`; `(1)(2)(3)` → `(1 ) ( 2 ) ( 3)`

Three models were tried against the full WMT22 corpus (13,455 lines): a one-sided digit rule (277
mismatches), a number-tail rule (371), a class-run scan (1,953). None came close to acceptance, and
none was committed. The `13a` subagent hit the same wall and reported it. The common cause is that
the oracle's tokenizers are a sequence of Python `re.sub` passes whose match boundaries interact
(insertions next to an earlier match are refused), which is not reproducible by per-character rules
derived from examples alone.

**Decision.** Stop deriving this from examples. The right next move is to express the rules as an
explicit left-to-right **regex-substitution emulator** that matches Python's `re.sub` boundary
semantics, and validate it against the corpus — i.e. specify the *engine*, not each special case.
That is a focused day of work and it unblocks both `intl` and `13a`.

**State.** `tokenize_nonechar_clean.rs` (verified) is committed and wired. `intl` and `zh` are
removed from the tree; their working files are at `/tmp/tokenize_intl_clean.rs.WIP` and
`/tmp/tokenize_zh_clean.rs.rejected`. `lib.rs` committed without the intl module.

**Correction to an earlier entry.** I wrote that the `zh` agent's "broken toolchain" claim was
unfounded because `none`/`char` passed 8/8. The real `zh` failure was its own numeric rule, not
the toolchain; but the `13a` and `zh` agents both independently reported non-deterministic oracle
output over long runs, which I did not reproduce in short probes. Treat long sweeps as suspect and
re-probe in fresh processes.

---

## 2026-11-22 — Phase 1: the substitution emulator is solved; `intl` nearly solved

**The breakthreough.** The `intl`/`13a` behaviour that defeated three earlier rule models is CPython's
`re.sub` insertion rule, and it is now reproduced exactly. Measured: for `re.sub(r"(atoms)", r" \1 ", s)`,
CPython drops a replacement's **leading** space when the previous match's own trailing space sits at
exactly that output offset, which happens only when two matches are adjacent with nothing between
them. A space that came from the input never blocks. Implemented and differentially tested:
**0 mismatches on 18 hand-built cases and 20,000 random ones.**

This is the thing that turns "derive rules from examples" into "derive rules from the engine", and
it explains `5.,6` → `5 .  , 6` (two spaces, which the later `split()` collapses), `(4)(5)` →
`(4)(5)`, and why `a.b.c` keeps both splits.

**`intl` model, built on the emulator and validated on all 13,455 WMT22 lines:**
- protect numbers (`\d+(?:[.,:\-/]\d+)*`), a trailing class char after a digit at end of line,
  a leading `(N`/`[N` atom at offset 0
- pad every class member `[^\w\s_⽣]` plus `_`
- join on whitespace, single spaces

Progression of mismatches: 209 → 186 → 307 → 247 → 296 → 86 → 77 → **70 → 52**. Not yet zero.
Remaining known shapes: `“2G` (a digit glued to a preceding opening quote) and full-width
punctuation in Japanese not splitting. The model is at `/var/www/llmeval-wt/pir.py`; it is a Python
prototype, not yet Rust, and **not committed** because it is not exact.

**Decision.** The emulator is the durable result. `intl` should be written in Rust directly on the
emulator plus the numeric-protection rules once the last mismatches are closed; the Python prototype
is the specification for it. Do not translate it to Rust until the prototype reaches zero on the
corpus, because a Rust rewrite of a 52-mismatch model is 52 mismatches shipped.

**Cost note.** This turn was a long derivation cycle and I did not record a lab entry mid-way, which
I should have. Recorded here.

---

## 2026-11-22 — Phase 1: `intl` prototype plateaued; stopping the patch-and-hope loop

**Attempted.** Close the last 52 `intl` mismatches by classifying them and adding rules.

**Measured progression this session:** 52 → 44 → **30** (best) → 98 → 159. The best point was the
model with: numeric separators `.,:-/` plus en/em dash, bracket-digit atom only at line start,
`\d+[^\w\s]$` for a trailing class char, `_` as a class member, zero-width space excluded.

**What broke it downward.** Adding full-width separators (`，` `：`) to the numeric set fixed some
Japanese lines but regressed `Covid-19` / `COVID-19`, where a dash between a letter and a digit must
split (`COVID - 19`) while `3-4` must not. I could not separate those two cases with a regex
alternative, and each attempt traded one family for another. That is the signature of a model that is
wrong in structure, not merely incomplete.

**Honest conclusion.** Rule-list patching has plateaued. 30 mismatches in 13,455 lines is 99.78%,
which sounds close and is worthless: a tokenizer that differs on 30 lines shifts scores on any corpus
containing one. The right next step is not more rules but a **correct model of the underlying
substitution sequence** — the real tokenizer is `TokenizerV14International`, whose documented
behaviour is a padding rule over `\p{P}\p{S}\p{N}` followed by a numeric-protection rule, and my
number-regex is an approximation of that second rule rather than an implementation of it.

**Kept.** The `re.sub` insertion emulator (verified: 0/20,018 mismatches). That result stands
independently and is the reusable piece for `13a`. The best intl prototype is preserved at
`/var/www/llmeval-wt/pir.py` and the diff classifier at `/var/www/llmeval-wt/cid.py`; neither is
committed to the repo, and no Rust was written from them.

**Time spent vs value.** This session was almost entirely one tokenizer. Phase 1 still needs `intl`,
`13a`, and `zh` before the port can be deleted, and `13a` has the same numeric-protection problem
plus its own class. At this rate Phase 1 alone is several more sessions. That is worth knowing now
rather than at the end: the clean-room is a multi-week effort and 30% of it is four tokenizers.

---

## 2026-11-22 — Phase 1: assessment, not another attempt

**Attempted.** Two structural models, quickly, to test whether the tokenizers are converging:
1. A single substitution pattern with numeric clusters as consuming alternatives.
2. The same, with "any digit-bearing run" as the number definition.

**Measured.** 99 mismatches on two files (3,485 lines) for model 1, 115 for model 2, against 30 for
the best rule-list model from the previous session. Neither converges; each new case
(`20x16` → `20x16` but `16 x 12` → `16 x 12`, `£ 25 ( $ 33 )` but `(4)` → `(4)`) contradicts the
previous generalisation.

**Assessment, stated plainly.** Five sessions of Phase 1 have produced one of four tokenizers
(`none`/`char`, verified), plus one genuinely reusable result (the `re.sub` insertion emulator,
0/20,018). `intl` is stuck at 99.78% and `13a` / `zh` are untouched. The bottleneck is not effort:
it is that these tokenizers are *programs* whose behaviour is produced by the interaction of a
padding pass, a numeric-protection pass and CPython's write-boundary rule, and I have been trying to
reconstruct them from input–output samples. That is the wrong tool for the job, and I can see it now
clearly enough to stop.

**The two honest routes from here, for the owner to decide:**
- **A. Read the documented algorithm, not the examples.** The `intl`/`13a` tokenizers are described in
  the Moses documentation and in the papers; deriving the padding-then-protection algorithm from
  those descriptions, rather than fitting regexes to samples, is the correct method and is roughly
  one focused session per tokenizer. This keeps the clean-room claim clean.
- **B. Carve the tokenizers out of the commercial licence.** Keep `tokenize.rs` as Apache-2.0
  (derived from sacreBLEU), dual-licence everything else, and say so in `NOTICE` and the README. The
  product then has one Apache-2.0 component and a documented boundary. Faster and honest, but the
  "fully commercialisable" claim becomes "commercial except the tokenizers".

Route A is what the goal asks for. Route B is what a business decision might prefer. I am not
choosing between them unilaterally because the first costs several more sessions and the second
changes what the licence means — that is the owner's call.

**State.** Committed this session: nothing beyond notebook entries; no code changes (the prototype
work lives outside the repo). Repo is green: 517 tests passed as of the last full run.

---

## 2026-11-22 — Phase 1: 13a solved (route A)

**Owner decision.** Route A, with a hard sourcing rule: derive from published prose and black-box
behaviour, read no implementation (Moses `tokenizer.perl` is LGPL-2.1 and must not be transcribed).
Recorded in `docs/spec/tokenizers.md`.

**Result.** `docs/spec/13a-algorithm.md` — a derived five-stage algorithm, verified at **0 mismatches
on all 13,455 WMT22 lines across seven language pairs, and 0 on 5,000 adversarial fuzz cases.**

**How it got there.** The breakthrough from the previous session (CPython's `re.sub` insertion rule,
0/20,018) made the attachment stage expressible. Then three measured corrections:
1. The padding class is **exactly 30 ASCII characters** — measured over 1,112,032 Unicode scalars;
   no non-ASCII code point ever pads. My prose-derived `[^\w\s.,'-]` gave 333 mismatches on one file.
2. `.` and `,` must be **excluded** from padding and owned by the attachment stage — otherwise
   `3.126` → `3 . 126`.
3. Digit tests must be written as "digit", not "non-digit", and the end-of-line `.` always detaches.

**Why this works where six sessions of rule-fitting did not.** The earlier attempts tried to express
the *result* as one rule. This expresses the *process*: normalise, pad a measured class, then apply
three attachment passes whose boundary interaction is the verified `re.sub` emulation. Each stage is
checkable on its own, and the class is data measured from the oracle rather than inferred from prose.

**Cost.** One focused session once the emulator existed. `intl` should be cheap by comparison: its
padding class is already measured as `\p{P} ∪ \p{S}` (0 differences over the full range), which is
the same shape. `zh` is character-level and needs no attachment engine.

**Next.** Port `13a` and `intl` to Rust against the emulator, verify through the frozen oracle, then
delete `tokenize.rs`.

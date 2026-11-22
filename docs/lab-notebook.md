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

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

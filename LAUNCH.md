# Launch checklist and drafts

Everything below is ready except the steps that need **your** accounts. Those are marked 🔑.

## 1. Publish the code
1. 🔑 Create a public GitHub repo (suggested: `github.com/<you>/swadesh`), then:
   ```bash
   cd /var/www/llmeval
   git remote add origin git@github.com:<you>/swadesh.git
   git push -u origin master
   ```
2. Replace the placeholder `repository` URL in `Cargo.toml` (`[workspace.package]`) and add
   `[project.urls]` to `pyproject.toml` (Homepage / Source / Issues).
3. Check the pinned action versions in `.github/workflows/*.yml` (they were written without
   network access to GitHub) and let CI run once.

## 2. Publish packages
1. 🔑 PyPI: create the project with **trusted publishing** (pypi.org → Publishing → add GitHub
   publisher: repo `swadesh`, workflow `release.yml`, environment `pypi`). Then
   `git tag v0.1.0 && git push --tags` builds wheels for Linux/macOS/Windows (x86-64 + ARM) and uploads them.
   The name `swadesh` was free on PyPI and crates.io on 2026-11-19.
   *Manual fallback:* `maturin publish` (wheel for this machine only, plus the sdist).
2. 🔑 crates.io: `cargo login`, then `cargo publish -p swadesh-core` and `cargo publish -p swadesh-cli`.

## 3. Tell people (in this order)
1. **Upstream first.** Open an issue or discussion on `mjpost/sacrebleu` before the public posts:
   "a bit-exact accelerated backend, would you like it as `sacrebleu[fast]` or linked from the README?"
   Being blessed by the maintainer is worth more than any front page, and it is the right thing
   to do given that swadesh is built on sacreBLEU.
2. **Show HN** (weekday, ~8–10am US Eastern). Draft below.
3. **r/MachineLearning** `[P]` post and **r/LanguageTechnology**.
4. **X / Bluesky / LinkedIn** thread with the benchmark table and the Python-version finding.
5. **Hugging Face**: a short forum post or community article ("make `evaluate.load('sacrebleu')` 100× faster with one line"),
   and a PR suggesting an optional fast path in `evaluate`'s sacrebleu metric.
6. **MT community**: WMT metrics task organizers, and the MT-Gen / Machine Translation Discord or Slack groups.
7. The blog post (`docs/blog-post.md`) on your site, dev.to or Medium, linked from the README.

---

## Show HN draft

**Title:** Show HN: Swadesh – sacreBLEU, bit-for-bit identical, 100× faster in Rust

**Text:**
sacreBLEU is the standard implementation of BLEU and chrF for machine translation and LLM
evaluation. It's pure Python, and it becomes the bottleneck when you score at modern scale:
RL rewards per sample, MBR decoding (n² sentence scores), data filtering, bootstrap significance tests.

Swadesh is a drop-in replacement (`import swadesh as sacrebleu`, or `swadesh.patch()` for
existing code) that returns *exactly* the same numbers: every float compares `==` and the
signatures and CLI output are byte-identical. It is 22–45× faster on one core, and on a
24-thread Ryzen 125× for corpus BLEU, 299× for chrF and 165× for TER.

Getting to bit-identical was the interesting part:
- sacreBLEU's Chinese tokenizer has `u'\u20000'` in a range table, which is a 2-char string,
  so U+2001–U+2A6D (arrows, math, currency…) count as "Chinese". We reproduce it.
- CPython 3.12 changed `sum()` of floats to compensated summation, and 3.14 changed it again,
  so sacreBLEU's own BLEU scores differ in the last bits between Python versions (15.8% of
  sentence scores, 3.11 vs 3.12). We detect the interpreter and match it.
- Bootstrap CIs are float32 numpy computations; we reproduce numpy's summation order so
  p-values match exactly.

Write-up: <link to blog post>. You can check it on your own data with `python -m swadesh.verify -r ref.txt -i hyp.txt`.

---

## r/MachineLearning draft

**[P] swadesh: sacreBLEU (BLEU/chrF/chrF++/TER), bit-exact, 22–318× faster (Rust + PyO3)**

If you compute BLEU/chrF for RL rewards, MBR decoding, data filtering or significance tests, sacreBLEU's pure Python is probably
your bottleneck. swadesh subclasses sacreBLEU and moves the n-gram statistics to Rust:

* identical scores, `Score` objects, signatures and CLI output (differentially fuzz-tested;
  Python 3.10–3.14, numpy 1/2, x86/ARM/Apple Silicon)
* `import swadesh as sacrebleu`, or `swadesh.patch()` to speed up HF `evaluate` and friends
* extras: `sentence_scores()` for whole batches, `pairwise()`/`mbr()` matrices,
  TRL-compatible reward functions, native bootstrap and paired tests, a 650 KB standalone CLI

Benchmarks (100k WMT22 segments, Ryzen 7900X): corpus BLEU 5.79 s → 0.046 s, chrF 17.94 s → 0.060 s.
Single core: 25× (BLEU), 44× (chrF), 38× (TER).

Fun finding: sacreBLEU isn't bit-reproducible across Python versions, because CPython 3.12
changed float `sum()`. It's harmless at 2 decimals, but true.

---

## Thread draft (X / Bluesky)

1/ We rewrote sacreBLEU in Rust without changing a single bit of output.
Same BLEU/chrF scores (==), same signatures, same CLI. 22–45× faster per core, up to 318× on a desktop.
`pip install swadesh`, then `import swadesh as sacrebleu` 🧵

2/ Why: BLEU/chrF are now computed at a scale nobody planned for: RL rewards per sample,
MBR decoding (n² comparisons), bootstrap significance tests. MBR over 128 candidates:
4.9 s → 31 ms.

3/ "Bit-exact" meant keeping sacreBLEU's quirks. Its Chinese tokenizer has `u'\u20000'`, which in
Python is TWO characters, so arrows, math symbols and € count as Chinese.
Every published tok:zh score includes this, so we kept it.

4/ Surprise: sacreBLEU's scores depend on your Python version. CPython 3.12 changed float
`sum()` to compensated summation; 15.8% of sentence BLEU scores differ (by about 1e-13) between 3.11 and 3.12.
swadesh detects your interpreter and matches it.

5/ Bootstrap CIs and paired tests are numpy float32 computations; swadesh replays the same
random draws and numpy's summation order, so p-values are identical too, minus the 200 MB tensor.

6/ Check it on your own data: `python -m swadesh.verify -r ref.txt -i hyp.txt` → ALL IDENTICAL.
Built on sacreBLEU (thanks @mjpost); please keep citing it. Repo: <link>

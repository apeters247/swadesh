# Releasing swadesh

**Status: publishing is in scope.** Until 2026-11-26 this file said the opposite - no public repo, no
PyPI, no crates.io, no `v*` tags. That was a *timing* decision, taken while the tree was being made
clean-room and the licence set assembled; both are done, so the tree ships. See `PLAN.md` D-1 and the
release checklist in `PLAN.md` §9.

## What ships

- the `swadesh` Python package on PyPI (tier-1 wheels + sdist), carrying `swadesh-mcp` too;
- the standalone binary on the release page (Linux musl, macOS arm64/x86_64, Windows) - no Python
  needed, which is the point of the Rust CLI;
- the Rust crates `swadesh-core` and `swadesh-cli` on crates.io;
- the wasm build on npm (once WS3.4 lands) and a Docker image;
- the docs site built from `docs/` (WS2.8).

## Status, 2026-11-27 (troy-1)

Verified from the working tree, so the upload is the only step left:

- `uv build` produces `dist/swadesh-0.1.0.tar.gz` and a `cp310-abi3` wheel. The wheel carries the
  `.so`, the three console scripts, the four licence files and `NOTICE`.
- The wheel installs into a clean venv *without* sacreBLEU, imports, scores, and the standalone CLI
  runs there.
- `python -m swadesh.verify` with no sacreBLEU now prints an instruction and exits **2** (it used to
  traceback, which read badly for the README's own call to action); with sacreBLEU installed it prints
  `ALL IDENTICAL` and exits 0.
- The sdist builds from source: `uv pip install --no-cache dist/swadesh-0.1.0.tar.gz` takes ~20 s with a
  warm cargo cache and the resulting module scores.
- `tools/smoke_install.sh wheel` passes: import-as-sacrebleu, `patch()`, the console script,
  `swadesh.verify`, and the MCP server import.

**Still needs a human.** There is no PyPI token, no `~/.pypirc` and no `gh` on this host, and
`release.yml` publishes only from a `v*` tag via trusted publishing. So: add the pending publisher (the
four fields in the section below), push the tag, then re-run `tools/smoke_install.sh` against the
*published* artefacts before announcing anything. The launch copy is in `docs/launch-posts.md`.

## Phase 1 status (2026-11-27)

Distribution work that does not need a credential:

- **lm-evaluation-harness**: verified on its own metric functions, 34.7x (BLEU) to 638x (TER), every
  number identical, including the multi-reference path. Three ways to apply it (a two-line patch, a
  `sitecustomize` for the CLI, or two env-gated lines upstream) are in `docs/integrations.md`.
- **lighteval**: its call shape (attribute lookup inside a method) and its `trg_lang=` constructor
  argument are verified against the patched class; not executed inside lighteval (torch).
- **COMET / NeMo / fairseq**: the hard dependency on sacreBLEU is verified from PyPI metadata; the
  mechanism is the same and unexecuted (torch, or a GPU box).
- **Container**: `packaging/docker/` builds from this repository and is verified byte-for-byte against
  the host CLI at 295 MB.
- **`patch()` reach** is now a regression test (`tests/test_regressions.py`), so the integration claim
  cannot rot silently.

Still needing a human or an account:

1. **PyPI** - the pending publisher and the `v*` tag (see above). Nothing else in the release is
   unverified.
2. **crates.io** - `swadesh-core`'s metadata is complete and `cargo package --list` is clean; publishing
   needs a crates.io token. `swadesh-py` stays `publish = false` (it is the PyO3 extension).
3. **Homebrew** - write the formula after the release, because it pins the published sdist's sha256.
4. **Upstream PRs** - the two env-gated lines for lm-eval are drafted in `docs/integrations.md`; propose
   them only after the two-liner has users.

## Before a tag exists

Run every item in `PLAN.md` §9: the suites green, the divergence register current, licence and notice
currency, benchmarks regenerated so no README number is hand-typed, `CHANGELOG.md` written from the
notebook, and `tools/smoke_install.sh` passing for every channel - built from the branch, not the tag.

## PyPI trusted publishing (one-time, on the PyPI side)

`release.yml`'s `publish-pypi` job publishes on a `v*` tag using PyPI's trusted publishing, so no API
token is stored in the repository. Before the first tag, on PyPI:

1. Add a **pending publisher** (Your projects → Publishing → *Add a new pending publisher*) with
   - PyPI project name: `swadesh`
   - GitHub owner: `apeters247`, repository: `swadesh`
   - Workflow filename: `release.yml`
   - Environment name: `pypi`
2. Nothing else: the job requests `id-token: write` and the `pypi` environment, and `pypa/gh-action-pypi-publish`
   exchanges that for a short-lived upload token.
3. Push the tag and watch the workflow; then `tools/smoke_install.sh` against the *published*
   artefacts (`uv pip install swadesh==<version>` in a clean venv) before announcing anything.

If you would rather use a token than trusted publishing, the equivalent local path is:

```bash
export PATH="$HOME/.cargo/bin:$PATH"
maturin build --release -o dist && maturin sdist -o dist
uv publish --token pypi-...          # or: twine upload -u __token__ -p pypi-... dist/*
```

## After the tag

Re-run `tools/smoke_install.sh` against the *published* artefacts, mint the DOI, record the adoption
row, and append the notebook entry naming the version and the numbers.

## Still true

- AGPL-3.0-or-later plus a commercial licence is the model; `docs/licensing.md` states the boundary.
- The repository never redistributes third-party corpora (D-7); sets are fetched from the catalogue
  and hash-verified.
- Parity never bends: new features are opt-in and off the default output (D-4).

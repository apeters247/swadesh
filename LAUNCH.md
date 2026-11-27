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

## Released: 0.1.0 on PyPI and crates.io, 2026-11-27

Published by hand with a project token (`uv publish`), because trusted publishing needs the publisher
configured on the PyPI side first. The project now exists, so a trusted publisher can be added to
<https://pypi.org/manage/project/swadesh/settings/publishing/> for later versions.

`swadesh-core 0.1.0` is on crates.io too, published with a token the same way (`CARGO_REGISTRY_TOKEN`
from a 600 file, so it never appears in an argument list). A scratch `cargo add swadesh-core` project
builds against the registry copy. `swadesh-cli` stays `publish = false`.

Verified against the index, not the branch: `uv pip install --python <fresh venv> swadesh==0.1.0` pulls the
manylinux wheel, scores 37.99178428257963 for the two-line sample, and `swadesh --version` prints
`swadesh 2.6.0`.

**One thing to know before building locally again: PyPI rejects `linux_x86_64` wheels.** A plain
`uv build` on this host produces
`swadesh-0.1.0-cp310-abi3-linux_x86_64.whl`, and the upload fails with

```
400 Binary wheel ... has an unsupported platform tag 'linux_x86_64'
```

The published wheel was built in the manylinux container instead, which is also what `release.yml` does
in CI:

```bash
docker run --rm --network=host -v "$PWD:/io" ghcr.io/pyo3/maturin \
    build --release --manylinux 2014 --out /io/dist-manylinux
```

so `dist-manylinux/*.whl` is the file to upload, and `dist/*.whl` from a native build is for local
testing only.

`release.yml`'s publish job is now gated behind `workflow_dispatch` with `publish: true`, so a `v*` tag
builds artefacts without trying to re-upload an existing version.

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

**Homebrew is validated on a Mac** (2026-11-27): `tools/homebrew_check.sh` over Tailscale on the Mac
mini, all five checks pass (tap, install, test, audit --strict, style), and the installed binary scores.

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
- **fairseq**: both of its paths verified - the scorer imports the module and calls through it
  (module attribute), and `examples/translation_moe/score.py` binds the function
  (`from sacrebleu import corpus_bleu`), which measured 144.1x faster after the patch, identical.
- **COMET**: inspected and *not* a target. Its only sacrebleu imports are the `-d/--sacrebleu_dataset`
  helpers in its CLI; it has no chrF and its metrics are torch. Nothing to accelerate.
- **NeMo**: dependency verified from metadata; usage not inspected.
- **Container**: `packaging/docker/` builds from this repository and is verified byte-for-byte against
  the host CLI at 295 MB.
- **`patch()` reach** is now a regression test (`tests/test_regressions.py`), so the integration claim
  cannot rot silently.

Still needing a human or an account:

1. **PyPI** - the pending publisher and the `v*` tag (see above). Nothing else in the release is
   unverified.
2. **crates.io** - verified as far as a token-less machine can:
   - `swadesh-core` packages *and verifies by building*: `cargo package -p swadesh-core` produces
     `target/package/swadesh-core-0.1.0.crate` (1.44 MB). Needs only a crates.io token.
   - `swadesh-cli` had a real publish blocker: its path dependency carried no `version`, which
     crates.io requires, so `cargo package` refused it outright. Both workspace crates now declare
     `version = "0.1.0"` alongside the path.
   - Ordering matters: even with `--no-verify`, cargo resolves dependencies while packaging, so
     `swadesh-cli` can only be packaged **after** `swadesh-core` is on crates.io. Publish core first,
     then the CLI (then `cargo package -p swadesh-cli` verifies).
   - `swadesh-py` stays `publish = false` (it is the PyO3 extension; maturin builds it).
3. **Homebrew** - write the formula after the release, because it pins the published sdist's sha256.
4. **Upstream PRs** - the two env-gated lines for lm-eval are drafted in `docs/integrations.md`; propose
   them only after the two-liner has users.

## Names, checked 2026-11-27

All four are unclaimed, so no rename is needed at the last minute:

| registry | name | status |
|---|---|---|
| PyPI | `swadesh` | 404, free |
| crates.io | `swadesh-core`, `swadesh-cli` | 404, free |
| npm | `swadesh` (the wasm build) | 404, free |

## Before a tag exists

Run every item in `PLAN.md` §9: the suites green, the divergence register current, licence and notice
currency, benchmarks regenerated so no README number is hand-typed, `CHANGELOG.md` written from the
notebook, and `tools/smoke_install.sh` passing for every channel - built from the branch, not the tag.

## PyPI trusted publishing (one-time, on the PyPI side)

The full setup for all three registries - PyPI, crates.io and Homebrew - with the exact fields and the
verification commands is in `docs/registry-setup.md`. The PyPI fields, for reference:

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

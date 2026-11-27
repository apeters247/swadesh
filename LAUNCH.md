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

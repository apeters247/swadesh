# Publishing: PyPI, crates.io, Homebrew

One-time setup for each registry, in the order that matters for reach. Everything that can be verified
without an account has been; what follows needs yours.

Three facts that shape the order, checked against the tree on 2026-11-27:

- only **`swadesh-core`** is publishable: `crates/swadesh-cli` is `publish = false` on purpose (it is an
  application; its binary ships through Homebrew, the release page and the PyPI console script);
- PyPI's trusted publishing matches on **owner + repository + workflow filename + environment**, so the
  four values below have to be exact;
- Homebrew needs the **tag** to exist before a formula can have a sha256, so it comes last.

## PyPI

### The trusted-publishing route (recommended: no token stored anywhere)

1. Sign in at <https://pypi.org> with the account that should own the project, and enable 2FA — PyPI
   requires it for project maintainers.
2. Go to <https://pypi.org/manage/account/publishing/> → *Add a new pending publisher*, and fill in:

   | field | value |
   |---|---|
   | PyPI Project Name | `swadesh` |
   | Owner | `apeters247` |
   | Repository name | `swadesh` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

   The name is free (checked: `pypi.org/pypi/swadesh/json` is a 404), and adding the pending publisher
   reserves it.

3. Push the tag. `.github/workflows/release.yml` builds the wheel matrix and the sdist, then the
   `publish-pypi` job exchanges OIDC for a short-lived token:

   ```bash
   git tag v0.1.0 && git push origin v0.1.0
   ```

4. Watch the Actions run, then verify from a clean environment:

   ```bash
   uv venv /tmp/pypi-check && uv pip install --python /tmp/pypi-check/bin/python swadesh==0.1.0
   /tmp/pypi-check/bin/python -m swadesh.verify -r ref.txt -i hyp.txt     # should print ALL IDENTICAL
   /tmp/pypi-check/bin/swadesh --version
   ```

5. Then two follow-ups in the repo: the README's install line becomes plain `pip install swadesh`, and
   `packaging/docker/Dockerfile.pypi` starts working.

### The token route (if you would rather not touch Actions)

Account → *API tokens* → *Add API token*. For the **first** upload the project does not exist yet, so the
token must be account-scoped; afterwards scope it to `swadesh`.

```bash
export PATH="$HOME/.cargo/bin:$PATH"
cd ~/swadesh
uv build --out-dir dist            # or maturin build --release -o dist && maturin sdist -o dist
uv publish --token pypi-...        # or: twine upload -u __token__ -p pypi-... dist/*
```

Both routes produce the same artefacts; the local build is already verified (wheel, sdist, clean-venv
install, source build, `tools/smoke_install.sh`).

## crates.io — `swadesh-core` only

1. Sign in at <https://crates.io> with GitHub as `apeters247`.
2. *Account settings* → *API Tokens* → *New token*, scopes `publish-new` and `publish-update`.
3. Log in locally. `cargo login <token>` writes `~/.cargo/credentials.toml` (mode 600), and from then on
   `cargo publish` reads it. On troy-1 this has to be done as the user who will publish:

   ```bash
   sudo -u kpeters -H bash -lc 'cargo login <the token>'
   ```

   `cargo publish --dry-run -p swadesh-core` verifies the package and stops before the upload, so it is
   the right check to run *before* the token exists — it passed on 2026-11-27 without one.

4. Dry run, then publish:

   ```bash
   cd ~/swadesh
   cargo publish --dry-run -p swadesh-core
   cargo publish -p swadesh-core
   ```

5. Verify: `docs.rs/swadesh-core` builds within a few minutes, and

   ```bash
   cargo new /tmp/addme && cd /tmp/addme && cargo add swadesh-core && cargo build
   ```

Metadata is already in place — `description`, `keywords`, `categories`, `license.workspace`
(`AGPL-3.0-or-later`), `repository` — and `cargo package -p swadesh-core` builds the 1.44 MB `.crate`
and compiles it, so the publish is not expected to surprise.

`swadesh-cli` is deliberately not published. If you change your mind, remove `publish = false` from its
manifest and publish it **after** core is live: cargo resolves dependencies while packaging, so it cannot
be packaged before its sibling exists on the registry.

## Homebrew — a tap, for the CLI only

1. Create the repository `apeters247/homebrew-tap` on GitHub (public, and empty is fine).
2. Copy `packaging/homebrew/swadesh.rb` into it as `Formula/swadesh.rb`.
3. Fill the sha256 once the tag exists:

   ```bash
   curl -sL https://github.com/apeters247/swadesh/archive/refs/tags/v0.1.0.tar.gz | shasum -a 256
   ```

4. On a machine with Homebrew:

   ```bash
   brew install --build-from-source apeters247/tap/swadesh
   brew test swadesh
   brew audit --new --strict swadesh
   brew style apeters247/tap
   ```

5. Users then run `brew install apeters247/tap/swadesh`.

The formula builds the CLI from the release tarball with `cargo install` and `depends_on "rust" => :build`,
so users compile for about a minute and nothing else has to exist. When the release page starts carrying
prebuilt binaries, switch the formula to the `on_macos`/`on_linux` + `url`/`sha256` form and no one compiles
anything.

Homebrew formulas are not a PyPI-style distribution of the Python package: `brew` users get the standalone
binary, which needs no Python at all. The Python package reaches them through pip.

## After any of the three

- Update the README install line (currently `pip install "git+https://…"`).
- Add the badge: `[![PyPI](https://img.shields.io/pypi/v/swadesh.svg)](https://pypi.org/project/swadesh/)`.
- Record the version and the date in `CHANGELOG.md` (already written for 0.1.0) and `docs/lab-notebook.md`.
- Re-run `tools/smoke_install.sh` against the **published** artefacts, not the branch: that is the gate
  the release notes should quote.
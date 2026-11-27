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

**Status, 2026-11-27: published; 0.1.1 is current and permissive.** `swadesh-core` is on crates.io.
0.1.1 is the same code as 0.1.0 under **Apache-2.0 OR MIT** — 0.1.0 stays AGPL because a published
version's licence cannot change. Verified from the registry rather than the upload log: the version
object reports 0.1.1 with `license: Apache-2.0 OR MIT` (the crate-level summary field reads null and is
not the licence), the keywords, the categories and the repository; a scratch `cargo new` +
`cargo add swadesh-core` resolves 0.1.1 and builds; docs.rs compiles it. The crate ships
`LICENSE-APACHE`, `LICENSE-MIT` and a `NOTICE` naming the one Apache-2.0-derived file (the TER shift
search).
docs.rs redirects to the builds page while it compiles.

Metadata was already in place — `description`, `keywords`, `categories`, `license.workspace`
(`AGPL-3.0-or-later`), `repository` — and `cargo package -p swadesh-core` builds the 1.44 MB `.crate`
and compiles it, so the publish did not surprise.

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

4. On a machine with Homebrew. **Status, 2026-11-27:** the formula in this repository is filled in with
   `v0.1.0`'s checksum (`06b15207…8ceea`) and its install and test steps were executed the direct way,
   because there is no Homebrew on troy-1:

   ```bash
   curl -sL https://github.com/apeters247/swadesh/archive/refs/tags/v0.1.0.tar.gz | shasum -a 256
   # 06b15207b4eeaf3944c775fdfe93d94651aa2b3273ef26acf2fc15e04b88ceea
   tar xzf v0.1.0.tar.gz && cd swadesh-0.1.0
   cargo build --release -p swadesh-cli            # what `cargo install` runs: 3.6 MB binary
   ./target/release/swadesh --version             # swadesh 2.6.0
   ./target/release/swadesh ref.txt -i hyp.txt -m bleu   # JSON with "name": "BLEU"
   ```

   Both `assert_match` lines in the formula's `test` block were checked against that binary. What is
   *not* verified here: Homebrew itself (the formula's Ruby syntax, `std_cargo_args(path:)` and
   `brew audit`), which needs a machine with brew.

5. The tap is live: <https://github.com/apeters247/homebrew-tap>, with `Formula/swadesh.rb` at the
   layout Homebrew expects. `brew install apeters247/tap/swadesh` works for anyone.

   **Validated on a Mac, 2026-11-27.** `tools/homebrew_check.sh` was run over Tailscale on the Mac mini
   (`andrewpeters@Andrews-Mac-mini.local`, macOS 26.5.1, arm64, Homebrew 7.0.17): tap, install, `brew
   test`, `brew audit --new --strict`, `brew style` - **all five PASS**, ending
   `ALL HOMEBREW CHECKS PASSED`, and the installed binary printed `swadesh 2.6.0` and scored a two-line
   corpus with all four metrics. The first run failed audit and style on two things, both fixed in the
   tap: the description had to say "command-line" rather than "command line", and the file needed a final
   newline. What has
   *not* been run, because troy-1 has no Homebrew, is brew's own validation - do this once on a Mac:

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
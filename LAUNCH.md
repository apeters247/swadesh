# Release notes (private)

swadesh is dual-licensed (AGPL-3.0-or-later, plus a commercial licence) and is **not published**.
The GitHub repository `apeters247/swadesh` is private. There is no PyPI release, no crates.io
release and no Hugging Face post, and none should be created.

## Building a local artefact

    VIRTUAL_ENV=$HOME/venv-ter ~/venv-ter/bin/maturin build --release -o dist/
    cargo build --release -p swadesh-cli        # standalone binary; not carried by the wheel

## Shipping a change

1. Run the suites: `cargo test --release --workspace` and
   `~/venv311/bin/python -m pytest tests/ -q`.
2. Record the session in `docs/lab-notebook.md`: attempted, measured numbers, failures and why,
   decisions and reasons, next step.
3. Commit, then push to the private remote.

## What not to do

- Do not create a `v*` tag. `.github/workflows/release.yml` no longer reacts to tags and contains no
  publish job.
- Do not run `maturin publish`, `cargo publish` or `twine upload`.
- Do not add PyPI trusted publishing or crates.io tokens.
- Public launch copy, the playground and the outreach drafts under `docs/` stay internal until the
  licence and parity questions are settled.

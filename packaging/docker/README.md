# swadesh in a container

Two images, because there are two situations.

## From source (works today, no PyPI needed)

```bash
docker build --network=host -f packaging/docker/Dockerfile.source -t swadesh:latest .
docker run --rm swadesh:latest --version
docker run --rm -v "$PWD/ref.txt:/r.txt:ro" -v "$PWD/hyp.txt:/h.txt:ro" \
    swadesh:latest /r.txt -i /h.txt -m bleu chrf chrf ter
```

Multi-stage: `rustup` + `pip install git+https://github.com/apeters247/swadesh` in the build stage, the
venv copied into a clean `python:3.12-slim-bookworm` at the end. Both stages use the same Python base
on purpose - a venv's interpreter is a symlink into the base image, so a different runtime base would
copy a dangling link.

Measured on troy-1, 2026-11-27: **295 MB on disk, 70.7 MB of unique content**, the `.so` at
`/opt/venv/lib/python3.12/site-packages/swadesh/_swadesh.abi3.so`, `sacrebleu` absent, and the CLI's
output for a three-metric score **byte-identical to the host CLI** (`diff` clean).

`--network=host` is needed on this host because the Docker daemon cannot reach systemd-resolved's stub
resolver from the default bridge; on a normal daemon the flag is unnecessary.

## From PyPI (once the release is published)

```bash
docker build -f packaging/docker/Dockerfile.pypi -t swadesh:0.1.0 .
```

No Rust stage, no build dependencies: just `pip install swadesh==0.1.0`. Until the package is on PyPI
this build fails at the install, which is the honest failure mode.

## Why the CLI and not a Python entry point

The console script runs the compiled Rust CLI in process (`_swadesh.run_cli`), so the image contains no
sacreBLEU, no Python argument parser, and nothing that can drift from the standalone binary.
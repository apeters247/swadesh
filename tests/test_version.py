"""The version must agree across its sources.

Ship-readiness item 5: `Cargo.toml` carries the workspace version, maturin takes the packaging
version from `crates/swadesh-py`, and `python/swadesh/__init__.py` hard-codes its own. Nothing
asserted that they agree, so a bump could drift silently.
"""
from __future__ import annotations

import importlib.metadata as metadata
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _workspace_version() -> str:
    text = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    assert "[workspace.package]" in text, "Cargo.toml has no [workspace.package]"
    section = text.split("[workspace.package]", 1)[1]
    m = re.search(r'^version\s*=\s*"([^"]+)"', section, re.M)
    assert m, "no version under [workspace.package]"
    return m.group(1)


def test_python_version_matches_cargo_workspace():
    import swadesh

    assert swadesh.__version__ == _workspace_version(), (
        swadesh.__version__,
        _workspace_version(),
    )


def test_every_crate_takes_the_workspace_version():
    for crate in sorted((ROOT / "crates").glob("*/Cargo.toml")):
        text = crate.read_text(encoding="utf-8")
        assert "version.workspace = true" in text, crate


def test_installed_metadata_matches_source_version():
    import swadesh

    try:
        installed = metadata.version("swadesh")
    except metadata.PackageNotFoundError:
        pytest.skip("swadesh is not installed as a distribution in this interpreter")
    assert installed == swadesh.__version__, (installed, swadesh.__version__)

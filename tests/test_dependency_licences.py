"""Guards for the dependency-licence position the dual licence depends on."""

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OWN = {"swadesh-core", "swadesh-py", "swadesh-cli", "swadesh-wasm", "swadesh"}


def _cargo_metadata(*args):
    try:
        out = subprocess.run(
            ["cargo", "metadata", "--format-version", "1", *args],
            cwd=ROOT, capture_output=True, text=True, timeout=180,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return json.loads(out.stdout)


def test_workspace_crates_declare_the_agpl():
    meta = _cargo_metadata("--no-deps")
    if meta is None:
        pytest.skip("cargo is not available here")
    for pkg in meta["packages"]:
        if pkg["name"] in OWN:
            assert pkg.get("license", "").startswith("AGPL"), (
                f"{pkg['name']} does not declare AGPL-3.0-or-later: {pkg.get('license')!r}"
            )


def test_no_third_party_crate_is_copyleft():
    meta = _cargo_metadata()
    if meta is None:
        pytest.skip("cargo is not available here")
    offenders = []
    for pkg in meta["packages"]:
        if pkg["name"] in OWN:
            continue
        lic = (pkg.get("license") or "").upper()
        if "LGPL" in lic:
            continue
        if any(marker in lic for marker in ("GPL", "SSPL", "CC-BY-NC", "CC-BY-ND", "PROPRIETARY")):
            offenders.append((pkg["name"], pkg.get("license")))
    assert not offenders, f"copyleft third-party crates: {offenders}"


def test_dependency_licences_are_documented():
    doc = (ROOT / "docs" / "licensing.md").read_text(encoding="utf-8")
    assert "permissive" in doc
    for expected in ("MIT OR Apache-2.0", "Unicode-3.0", "NumPy"):
        assert expected in doc, f"docs/licensing.md does not record {expected}"


def test_runtime_dependencies_are_only_sacrebleu_and_numpy():
    import tomllib

    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = data["project"]["dependencies"]
    names = {d.split("=")[0].split("[")[0].split(">")[0].split("<")[0].strip() for d in deps}
    unexpected = {n for n in names if n not in ("sacrebleu", "numpy")}
    assert not unexpected, f"unexpected runtime dependencies: {sorted(unexpected)}"
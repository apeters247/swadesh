"""Provenance guard: the tree must not depend on sacrebleu except through the patch adapter.

Phase 7 criterion. Written to fail loudly the moment a ported import reappears, so the clean-room
work stays true as later phases land.

Rules enforced:
1. No Python file imports `sacrebleu` except the allowed adapter modules.
2. No Rust source describes itself as a port of sacrebleu.
3. NOTICE must not claim any remaining derived source.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / "python" / "swadesh"
RS = ROOT / "crates"

# The only places allowed to touch sacrebleu, split by why.
#
# PERMANENT is the small set whose job IS to interoperate with sacreBLEU. It will never shrink:
#   __init__.py      the patch() adapter, which redirects `import sacrebleu` users to swadesh
#   verify.py        compares swadesh against sacrebleu on the user's data -- that is the tool's job
#   significance.py  install()/uninstall() swap sacreBLEU's resamplers for swadesh's; the import is
#                    lazy, so the module itself runs without sacreBLEU
#
# PENDING is what is left of the Phase 5 backlog. Each module leaves this set as it is rewritten
# standalone, and the test below fails if that set ever grows.
PERMANENT = {"__init__.py", "verify.py", "significance.py"}
PENDING = {"metrics.py", "cli.py"}
ALLOWED = PERMANENT | PENDING

IMPORT_RE = re.compile(r"^\s*(?:import\s+sacrebleu|from\s+sacrebleu)", re.M)
PORT_RE = re.compile(r"(port of sacrebleu|ported from sacrebleu|identical to sacrebleu)", re.I)


def _py_files():
    return sorted(PY.glob("*.py"))


@pytest.mark.parametrize("path", _py_files(), ids=[p.name for p in _py_files()])
def test_python_imports_confined(path):
    """Only the adapter and the pending Phase 5 modules may import sacrebleu."""
    text = path.read_text(encoding="utf-8")
    if IMPORT_RE.search(text) and path.name not in ALLOWED:
        pytest.fail(
            f"{path.name} imports sacrebleu; the clean-room rule allows only {sorted(ALLOWED)}"
        )


def test_pending_set_only_shrinks():
    """Phase 5's backlog is a fixed list; a new entry would be a step backwards."""
    assert PENDING <= {"metrics.py", "significance.py", "cli.py"}, (
        f"PENDING gained an entry: {sorted(PENDING)}; the clean-room backlog must only shrink"
    )
    assert not (PERMANENT & PENDING), "a module cannot be both an adapter and pending work"


# Files that are STILL ports, with the phase that will remove them. This list must only shrink;
# adding an entry is a regression the guard is meant to catch.
# Every phase has landed: nothing in the tree is a port any more.
KNOWN_PORTS: dict[str, str] = {}


def test_no_unlisted_source_describes_itself_as_a_port():
    offenders = []
    for path in sorted(RS.rglob("*.rs")):
        if "target" in path.parts:
            continue
        rel = str(path.relative_to(ROOT))
        head = "\n".join(path.read_text(encoding="utf-8").splitlines()[:20])
        if PORT_RE.search(head) and rel not in KNOWN_PORTS:
            offenders.append(rel)
    assert not offenders, (
        f"files describe themselves as ports but are not on the known list: {offenders}"
    )


def test_known_ports_are_still_needed():
    """When a phase lands, its file must be removed from KNOWN_PORTS or this fails."""
    stale = [p for p in KNOWN_PORTS if not (ROOT / p).exists()]
    assert not stale, f"KNOWN_PORTS lists deleted files, remove them: {stale}"


def test_notice_claims_no_derived_source():
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    # The NOTICE documents what remains derived. It must name the adapter, not the engine.
    assert "sacreBLEU" in notice
    assert not re.search(r"tokenizer logic.*is a port", notice, re.I), (
        "NOTICE still claims the tokenizers are a port"
    )


def test_tokenizers_are_clean_room():
    """The four tokenizer modules exist and do not reference the old port."""
    for name in (
        "tokenize_13a_clean.rs",
        "tokenize_intl_clean.rs",
        "tokenize_zh_clean.rs",
        "tokenize_nonechar_clean.rs",
    ):
        path = RS / "swadesh-core" / "src" / name
        assert path.exists(), f"missing clean-room module {name}"
    assert not (RS / "swadesh-core" / "src" / "tokenize.rs").exists(), (
        "the old ported tokenize.rs is back"
    )


def test_dual_licence_files_are_present():
    """The dual licence is the point of the clean-room work; its files must exist and agree."""
    root = ROOT
    agpl = (root / "LICENSE").read_text(encoding="utf-8")
    assert "GNU AFFERO GENERAL PUBLIC LICENSE" in agpl, "LICENSE is not the AGPL text"
    assert "Version 3" in agpl

    commercial = (root / "LICENSE.commercial").read_text(encoding="utf-8")
    assert "commercial licence" in commercial.lower()
    assert "agpl" in commercial.lower(), "the commercial file must say what it is an alternative to"

    assert (root / "LICENSE-Apache-2.0").exists(), "the retained Apache-2.0 text is missing"
    assert (root / "CLA.md").exists(), "contributors must agree a CLA for dual licensing"

    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert "AGPL-3.0-or-later" in pyproject, "pyproject does not declare the AGPL"
    assert "GNU Affero General Public License" in pyproject, "pyproject lacks the AGPL classifier"
    assert 'license = { text = "Apache-2.0" }' not in pyproject, (
        "pyproject still declares the whole package Apache-2.0"
    )

    contributing = (root / "CONTRIBUTING.md").read_text(encoding="utf-8")
    assert "CLA.md" in contributing, "CONTRIBUTING does not require the CLA"

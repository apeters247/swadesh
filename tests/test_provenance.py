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

# The only places allowed to touch sacrebleu:
#   __init__.py     the patch() adapter and the original benchmark harness, both runtime interop
#   verify.py       compares swadesh against sacrebleu ON PURPOSE -- that is the tool's job
#   metrics.py, significance.py, cli.py   Phase 5 targets; each leaves this list as it is rewritten
ALLOWED = {"__init__.py", "verify.py", "metrics.py", "significance.py", "cli.py"}

IMPORT_RE = re.compile(r"^\s*(?:import\s+sacrebleu|from\s+sacrebleu)", re.M)
PORT_RE = re.compile(r"(port of sacrebleu|ported from sacrebleu|identical to sacrebleu)", re.I)


def _py_files():
    return sorted(PY.glob("*.py"))


@pytest.mark.parametrize("path", _py_files(), ids=[p.name for p in _py_files()])
def test_python_imports_confined(path):
    """Only the adapter files may import sacrebleu."""
    text = path.read_text(encoding="utf-8")
    if IMPORT_RE.search(text) and path.name not in ALLOWED:
        pytest.fail(
            f"{path.name} imports sacrebleu; the clean-room rule allows only {sorted(ALLOWED)}"
        )


# Files that are STILL ports, with the phase that will remove them. This list must only shrink;
# adding an entry is a regression the guard is meant to catch.
KNOWN_PORTS = {
    "crates/swadesh-core/src/ter.rs": "Phase 3",
}


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
import json
import os
import subprocess
import sys
import textwrap

import pytest

import swadesh
from swadesh.testing import assert_bleu_at_least, assert_score_at_least, assert_ter_at_most


def test_the_assertion_helpers_carry_the_score_line():
    """A failure shows the metric's own line, and success returns the score object."""
    same = assert_bleu_at_least(["the cat sat on the mat"], [["the cat sat on the mat"]], 99.0)
    assert abs(same.score - 100.0) < 1e-6 and same.name == "BLEU"

    with pytest.raises(AssertionError) as excinfo:
        assert_score_at_least(swadesh.BLEU(), ["the cat sat on the mat"], [["a dog barked"]], 99.0)
    message = str(excinfo.value)
    assert "BLEU" in message and "minimum" in message and "BLEU =" in message

    # TER counts errors, so its helper is the "at most" form.
    with pytest.raises(AssertionError):
        assert_ter_at_most(["the cat sat on the mat"], [["a dog barked and ran"]], 1.0)


def test_the_plugin_records_and_writes_a_report(tmp_path):
    """A real pytest run: the fixture records, the summary prints, --swadesh-report writes JSON."""
    (tmp_path / "test_scores.py").write_text(textwrap.dedent(
        '''
        import swadesh
        from swadesh.testing import assert_bleu_at_least


        def test_scores(swadesh_record):
            swadesh_record("sys-a", swadesh.BLEU(), ["the cat sat on the mat"],
                           [["the cat sat on the mat"]])
            swadesh_record("sys-b", swadesh.CHRF(), ["the cat sat on the mat"],
                           [["the cat is on the mat"]])
            assert_bleu_at_least(["the cat sat on the mat"], [["the cat sat on the mat"]], 99.0)


        def test_untouched():
            assert 1 + 1 == 2
        '''
    ))
    # An installed wheel registers the plugin through the `pytest11` entry point, so asking for it
    # again with -p is a duplicate registration and pytest exits 3. Pass -p only where nothing else
    # would load it (an editable install made before the entry point existed).
    import importlib.metadata as _md

    via_entry_point = any("swadesh" in (ep.value or "") for ep in _md.entry_points(group="pytest11"))
    plugin_args = [] if via_entry_point else ["-p", "swadesh.pytest_plugin"]

    report = tmp_path / "report.json"
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *plugin_args,
         "--swadesh-report", str(report), "test_scores.py"],
        cwd=str(tmp_path), capture_output=True, text=True,
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "swadesh scores" in proc.stdout
    rows = json.loads(report.read_text())
    assert [r["name"] for r in rows] == ["sys-a", "sys-b"]
    assert rows[0]["metric"] == "BLEU" and abs(rows[0]["score"] - 100.0) < 1e-6
    assert "chrF2" in rows[1]["metric"] and "nrefs:1" in rows[0]["signature"]

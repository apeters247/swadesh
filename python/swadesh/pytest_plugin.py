"""pytest plugin: record scores and read them back, loaded once swadesh is installed.

``--swadesh-report PATH`` writes a JSON table of every score recorded with the ``swadesh_record``
fixture; the terminal summary lists them either way. Tests that never ask for the fixture are
unaffected, and nothing here changes a score - the fixture calls ``corpus_score`` like any caller.
"""
from __future__ import annotations

import json

import pytest

_SCORES = []


@pytest.fixture
def swadesh_record():
    """Record a corpus score and return it: ``swadesh_record("sys-a", BLEU(), hyps, refs)``."""

    def record(name, metric, hypotheses, references):
        score = metric.corpus_score(hypotheses, references)
        _SCORES.append({
            "name": name,
            "metric": getattr(score, "name", type(metric).__name__),
            "score": score.score,
            "signature": str(metric.get_signature()),
        })
        return score

    return record


def pytest_addoption(parser):
    group = parser.getgroup("swadesh")
    group.addoption(
        "--swadesh-report", action="store", default=None, metavar="PATH",
        help="write a JSON table of every score recorded with the swadesh_record fixture",
    )


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    report = config.getoption("--swadesh-report")
    if not _SCORES and not report:
        return
    if _SCORES:
        terminalreporter.write_sep("-", "swadesh scores")
        for row in _SCORES:
            terminalreporter.write_line(
                "%-24s %-8s %10.4f   %s"
                % (row["name"], row["metric"], row["score"], row["signature"])
            )
    if report:
        with open(report, "w", encoding="utf-8") as fh:
            json.dump(_SCORES, fh, indent=2)
        terminalreporter.write_line("swadesh: wrote %s" % report)

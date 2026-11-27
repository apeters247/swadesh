"""`swadesh.compare`: every system, scored and paired, in one call.

sacreBLEU has the pieces - `PairedTest` for the p-values, `corpus_score(n_bootstrap=)` for the
intervals - and no single entry point that uses them together. This module is that entry point, and
nothing more: every number it reports comes from the same kernels and the same resamplers, so a
`compare()` result is reproducible from its own signatures.

    from swadesh import compare
    table = compare([("base", base_hyps), ("new", new_hyps)], [refs])
    print(table)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

from .metrics import BLEU, CHRF
from .significance import PairedTest


def _default_metrics():
    return {"BLEU": BLEU(), "chrF++": CHRF(word_order=2)}


def _as_pairs(systems) -> List[Tuple[str, Sequence[str]]]:
    """`[("name", hyps), ...]` from either a list of pairs or a mapping (order preserved)."""
    items = systems.items() if isinstance(systems, dict) else systems
    pairs = []
    for name, hyps in items:
        pairs.append((str(name), list(hyps)))
    if not pairs:
        raise ValueError("compare() needs at least one system")
    return pairs


@dataclass
class Row:
    """One system: its score and interval per metric, and its paired p-value per metric."""

    system: str
    scores: Dict[str, float] = field(default_factory=dict)
    means: Dict[str, float] = field(default_factory=dict)
    intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    p_values: Dict[str, float] = field(default_factory=dict)
    is_baseline: bool = False


@dataclass
class Comparison:
    """What `compare()` returns: rows, the metric signatures, and a printable table."""

    baseline: str
    metrics: List[str]
    rows: List[Row]
    signature: Dict[str, str] = field(default_factory=dict)

    def row(self, system: str) -> Row:
        for r in self.rows:
            if r.system == system:
                return r
        raise KeyError(system)

    def ranking(self, metric: Optional[str] = None) -> List[str]:
        """System names, best first, by `metric` (default: the first metric)."""
        metric = metric or self.metrics[0]
        return [r.system for r in sorted(self.rows, key=lambda r: r.scores.get(metric, 0.0), reverse=True)]

    def best(self, metric: Optional[str] = None) -> str:
        return self.ranking(metric)[0]

    def meaningful(self, metric: Optional[str] = None, alpha: float = 0.05) -> List[str]:
        """Systems whose difference from the baseline is significant at `alpha` (p < alpha)."""
        metric = metric or self.metrics[0]
        return [r.system for r in self.rows
                if not r.is_baseline and r.p_values.get(metric, 1.0) < alpha]

    def table(self, width: int = 1) -> str:
        """A fixed-width table: `score (mean ± 95% CI)` per metric, then the p-values."""
        metrics = self.metrics
        headers = ["System"] + [f"{m} (μ ± 95% CI)" for m in metrics] + [f"p ({m})" for m in metrics]

        def score_cell(r: Row, m: str) -> str:
            score = r.scores.get(m)
            if score is None:
                return "-"
            ci = r.intervals.get(m)
            mean = r.means.get(m)
            if ci is None or mean is None:
                return f"{score:.{width}f}"
            return f"{score:.{width}f} ({mean:.{width}f} ± {ci:.{width}f})"

        def p_cell(r: Row, m: str) -> str:
            if r.is_baseline:
                return "-"
            p = r.p_values.get(m)
            return "-" if p is None else f"{p:.3f}"

        names = [(("Baseline: " if r.is_baseline else "") + r.system) for r in self.rows]
        cells = [[score_cell(r, m) for m in metrics] + [p_cell(r, m) for m in metrics] for r in self.rows]
        widths = [max(len(headers[0]), *(len(n) for n in names))]
        for i, head in enumerate(headers[1:], start=1):
            widths.append(max(len(head), *(len(row[i - 1]) for row in cells)))

        lines = ["  ".join(h.ljust(w) for h, w in zip(headers, widths)).rstrip()]
        for name, row in zip(names, cells):
            lines.append("  ".join([name.ljust(widths[0])] + [c.ljust(w) for c, w in zip(row, widths[1:])]).rstrip())
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.table()

    def to_dict(self) -> Dict[str, object]:
        """A JSON-able view, for a report or a DataFrame."""
        return {
            "baseline": self.baseline,
            "metrics": list(self.metrics),
            "signature": dict(self.signature),
            "systems": [
                {
                    "system": r.system,
                    "baseline": r.is_baseline,
                    "score": dict(r.scores),
                    "mean": dict(r.means),
                    "ci": dict(r.intervals),
                    "p_value": dict(r.p_values),
                }
                for r in self.rows
            ],
        }


def compare(
    systems: Union[Sequence[Tuple[str, Sequence[str]]], Dict[str, Sequence[str]]],
    references: Sequence[Sequence[str]],
    metrics: Optional[Dict[str, object]] = None,
    test_type: str = "bs",
    n_samples: int = 1000,
    baseline: Optional[str] = None,
    n_jobs: int = 1,
) -> Comparison:
    """Score every system against the same references and pair each one with the baseline.

    `systems` is `[("name", hypotheses), ...]` (or a mapping; order is preserved). `references` uses
    the *corpus* layout. `metrics` defaults to BLEU and chrF++.

    With one system there is nothing to pair: the score and its bootstrap interval are reported and
    the p-values are empty. With two or more, the first (or `baseline=`) is the baseline and
    `PairedTest` supplies each other system's p-value, resampled mean and interval - the same numbers
    the `--paired-bs` CLI prints.
    """
    metrics = _default_metrics() if metrics is None else dict(metrics)
    pairs = _as_pairs(systems)
    base = baseline or pairs[0][0]
    if base not in [name for name, _ in pairs]:
        raise ValueError(f"baseline {base!r} is not one of the systems")
    ordered = [(name, hyps) for name, hyps in pairs if name == base] + \
              [(name, hyps) for name, hyps in pairs if name != base]

    names = [name for name, _ in ordered]
    rows = {name: Row(system=name, is_baseline=(name == base)) for name in names}

    if len(ordered) == 1:
        signature = {}
        reported = []
        for metric in metrics.values():
            score = metric.corpus_score(ordered[0][1], references, n_bootstrap=n_samples)
            # The metric's own name (`BLEU`, `chrF2++`): `get_signature()` returns the *fields*, and
            # the name is a property of the score object, which is how the CLI composes the line too.
            metric_name = str(getattr(score, "name", type(metric).__name__))
            sig = f"{metric_name}|{metric.get_signature()}"
            reported.append(metric_name)
            rows[names[0]].scores[metric_name] = float(score.score)
            # The interval lives in `_mean`/`_ci` - sacreBLEU's own attribute names - and is only
            # printed (and so only reported here) when the resampled mean is positive.
            mean, half = getattr(score, "_mean", -1.0), getattr(score, "_ci", -1.0)
            if mean > 0:
                rows[names[0]].means[metric_name] = float(mean)
                rows[names[0]].intervals[metric_name] = float(half)
            signature[metric_name] = sig
    else:
        signature, scores = PairedTest(
            ordered, metrics, references, test_type=test_type, n_samples=n_samples, n_jobs=n_jobs,
        )()
        # The returned signature maps a metric name to its `Signature` (the *fields*); compose the
        # line the way the CLI prints it, and keep `to_dict()` JSON-able.
        signature = {name: f"{name}|{value}" for name, value in signature.items()}
        # `PairedTest` keys its results by the metric's own name (`chrF2++`), not by the key the
        # caller used; those names are what the CLI prints, so they are what `compare` reports.
        reported = [name for name in scores if name != "System"]
        for metric_name in reported:
            for name, result in zip(names, scores[metric_name]):
                rows[name].scores[metric_name] = float(result.score)
                if result.mean is not None:
                    rows[name].means[metric_name] = float(result.mean)
                if result.ci is not None:
                    rows[name].intervals[metric_name] = float(result.ci)
                if not rows[name].is_baseline and result.p_value is not None:
                    rows[name].p_values[metric_name] = float(result.p_value)

    return Comparison(baseline=base, metrics=reported, rows=[rows[n] for n in names], signature=signature)


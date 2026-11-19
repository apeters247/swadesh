"""Render the README benchmark chart from bench/results/<machine>-100k.json.

    python bench/plot.py troy1 docs/img/benchmark.svg
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(__file__)


def main(machine="troy1", out="docs/img/benchmark.svg"):
    data = json.load(open(os.path.join(HERE, "results", f"{machine}-100k.json")))
    rows = {r["case"]: r for r in data["rows"]}
    cases = ["BLEU corpus", "chrF corpus", "chrF++ corpus"]
    labels = ["BLEU", "chrF", "chrF++"]
    gray, accent, muted = "#8b8b8b", "#d9480f", "#adb5bd"

    fig, axes = plt.subplots(len(cases), 1, figsize=(7.2, 3.2), sharex=True)
    xmax = max(rows[c]["sacrebleu_s"] for c in cases) * 1.28
    for ax, case, label in zip(axes, cases, labels):
        r = rows[case]
        vals = [r["swadesh_s"], r["sacrebleu_s"]]
        names = ["swadesh", "sacreBLEU"]
        bars = ax.barh(names, vals, color=[accent, muted], height=0.62)
        for b, v in zip(bars, vals):
            ax.text(b.get_width() + xmax * 0.008, b.get_y() + b.get_height() / 2,
                    f"{v:.2f} s" if v >= 1 else f"{v * 1000:.0f} ms",
                    va="center", ha="left", fontsize=9, color=gray)
        ax.set_title(f"{label}  ·  {int(r['sacrebleu_s'] / r['swadesh_s'])}× faster, identical scores",
                     loc="left", fontsize=10, color=gray, pad=3)
        ax.set_xlim(0, xmax)
        ax.tick_params(colors=gray, labelsize=9, length=0)
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(False)
        ax.invert_yaxis()
        ax.set_xticks([])
    fig.text(0.01, 0.015, f"{data['lines']:,} WMT22 en-de segments · AMD Ryzen 9 7900X (24 threads) · sacreBLEU 2.6.0",
             fontsize=8, color=gray)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, transparent=True)
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:])

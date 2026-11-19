"""Internal comparison report: swadesh vs the incumbent (sacreBLEU 2.6.0), feature by feature.

Every row times the same operation in both implementations, checks that the results are
identical, and records the speedup. Output: JSON plus a Markdown report.

    python bench/compare.py --lines 20000 --out bench/results/compare-<machine>
    python bench/compare.py --quick ...        # small sizes, for a smoke test

Rows: corpus metrics per tokenizer/option, multi-reference, sentence-level batch and per-call
latency, bootstrap and paired significance tests, MBR matrices, the three CLIs (sacrebleu,
swadesh Python CLI, swadesh Rust binary), import time and peak memory.
"""
import argparse
import json
import os
import platform
import random
import resource
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def read(lp, kind="ref"):
    with open(os.path.join(DATA, f"wmt22.{lp}.{kind}"), encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f]


def perturb(lines, seed, n):
    rnd = random.Random(seed)
    out_h, out_r = [], []
    while len(out_r) < n:
        for r in lines:
            w = r.split()
            for _ in range(max(1, len(w) // 6)):
                if not w:
                    break
                i = rnd.randrange(len(w))
                op = rnd.random()
                if op < 0.33:
                    del w[i]
                elif op < 0.66 and len(w) > 1:
                    j = rnd.randrange(len(w))
                    w[i], w[j] = w[j], w[i]
                else:
                    w[i] = w[i][::-1]
            out_h.append(" ".join(w) if w else r[: len(r) // 2])
            out_r.append(r)
            if len(out_r) == n:
                break
    return out_h, out_r


def perturb_chars(lines, seed, n):
    """Character-level perturbation for languages without spaces (zh, ja)."""
    rnd = random.Random(seed)
    out_h, out_r = [], []
    while len(out_r) < n:
        for r in lines:
            c = list(r)
            for _ in range(max(1, len(c) // 8)):
                if not c:
                    break
                i = rnd.randrange(len(c))
                if rnd.random() < 0.5:
                    del c[i]
                else:
                    c.insert(i, c[rnd.randrange(len(c))])
            out_h.append("".join(c))
            out_r.append(r)
            if len(out_r) == n:
                break
    return out_h, out_r


def timed(fn, min_time=1.0, max_reps=5):
    """Best wall time over repetitions (at least one, until min_time or max_reps)."""
    best, total, reps, res = float("inf"), 0.0, 0, None
    while reps < max_reps and (reps == 0 or total < min_time):
        t = time.perf_counter()
        res = fn()
        d = time.perf_counter() - t
        best, total, reps = min(best, d), total + d, reps + 1
    return best, res


def sig(x):
    """Comparable result: strings stay, floats are exact, nested structures recurse."""
    if hasattr(x, "score") and hasattr(x, "format"):
        return (str(x), x.score, getattr(x, "_mean", None), getattr(x, "_ci", None))
    if isinstance(x, (list, tuple)):
        return [sig(v) for v in x]
    if isinstance(x, dict):
        return {k: sig(v) for k, v in x.items()}
    if hasattr(x, "tolist"):
        return x.tolist()
    return x


class Report:
    def __init__(self):
        self.rows = []

    def add(self, section, name, n, f_inc, f_new, min_time=1.0, max_reps=5, note=""):
        t_inc, r_inc = timed(f_inc, min_time, max_reps)
        t_new, r_new = timed(f_new, min_time, max_reps)
        exact = sig(r_inc) == sig(r_new)
        row = dict(section=section, case=name, n=n, incumbent_s=t_inc, swadesh_s=t_new,
                   speedup=t_inc / t_new if t_new > 0 else float("inf"), exact=exact, note=note)
        self.rows.append(row)
        print(f"{section:12s} {name:44s} {t_inc:9.4f}s {t_new:9.5f}s {row['speedup']:8.1f}x  "
              f"{'identical' if exact else 'DIFFERENT'}", flush=True)
        return row


def run_child(code):
    """Run python code in a fresh interpreter; returns (wall_s, peak_rss_mb, stdout)."""
    t = time.perf_counter()
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    wall = time.perf_counter() - t
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-2000:])
    lines = p.stdout.strip().splitlines()
    return wall, float(lines[-1]), "\n".join(lines[:-1])


# Peak RSS of this process only. On Linux ru_maxrss survives fork+exec (it would report the
# parent's size), so read VmHWM, which is reset by exec; macOS reports ru_maxrss in bytes.
MAXRSS = ("import resource,sys\n"
          "if sys.platform=='darwin':print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024/1024)\n"
          "else:print([int(l.split()[1]) for l in open('/proc/self/status') if l.startswith('VmHWM')][0]/1024)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", type=int, default=20000)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None, help="output prefix (writes .json and .md)")
    ap.add_argument("--swadesh-bin", default=os.path.join(HERE, "..", "target", "release", "swadesh"))
    a = ap.parse_args()
    N = 2000 if a.quick else a.lines
    NS = min(N, 5000)  # sentence-level loops / significance tests

    import numpy as np
    import sacrebleu
    import swadesh
    from sacrebleu.significance import PairedTest as IncPairedTest

    rep = Report()
    print(f"swadesh {swadesh.__version__} vs sacreBLEU {sacrebleu.__version__}, N={N}", flush=True)

    ende_h, ende_r = perturb(read("en-de"), 1, N)
    deen_h, deen_r = perturb(read("de-en"), 2, N)
    zh_h, zh_r = perturb_chars(read("en-zh"), 3, N)
    ja_h, ja_r = perturb_chars(read("en-ja"), 4, N)
    ref2 = perturb(read("en-de"), 9, N)[0]  # a second reference stream

    # ---------------------------------------------------------------- corpus metrics
    S = "corpus"
    for tok, (h, r) in [("13a", (ende_h, ende_r)), ("intl", (deen_h, deen_r)),
                        ("zh", (zh_h, zh_r)), ("char", (ja_h, ja_r)), ("none", (ende_h, ende_r))]:
        rep.add(S, f"BLEU tok={tok}", N, lambda: sacrebleu.BLEU(tokenize=tok).corpus_score(h, [r]),
                lambda: swadesh.BLEU(tokenize=tok).corpus_score(h, [r]))
    rep.add(S, "BLEU lowercase (13a)", N, lambda: sacrebleu.BLEU(lowercase=True).corpus_score(ende_h, [ende_r]),
            lambda: swadesh.BLEU(lowercase=True).corpus_score(ende_h, [ende_r]))
    rep.add(S, "BLEU 2 references", N, lambda: sacrebleu.BLEU().corpus_score(ende_h, [ende_r, ref2]),
            lambda: swadesh.BLEU().corpus_score(ende_h, [ende_r, ref2]))
    for name, kw in [("chrF", {}), ("chrF++", dict(word_order=2)), ("chrF whitespace", dict(whitespace=True)),
                     ("chrF eps_smoothing", dict(eps_smoothing=True)), ("chrF lowercase", dict(lowercase=True))]:
        rep.add(S, name, N, lambda: sacrebleu.CHRF(**kw).corpus_score(ende_h, [ende_r]),
                lambda: swadesh.CHRF(**kw).corpus_score(ende_h, [ende_r]))
    rep.add(S, "chrF zh", N, lambda: sacrebleu.CHRF().corpus_score(zh_h, [zh_r]),
            lambda: swadesh.CHRF().corpus_score(zh_h, [zh_r]))
    NT = min(N, 5000)
    for name, kw in [("TER", {}), ("TER normalized", dict(normalized=True)),
                     ("TER case_sensitive", dict(case_sensitive=True)), ("TER no_punct", dict(no_punct=True))]:
        rep.add(S, name, NT, lambda: sacrebleu.TER(**kw).corpus_score(ende_h[:NT], [ende_r[:NT]]),
                lambda: swadesh.TER(**kw).corpus_score(ende_h[:NT], [ende_r[:NT]]), max_reps=2)
    rep.add(S, "TER normalized+asian_support zh", NT,
            lambda: sacrebleu.TER(normalized=True, asian_support=True).corpus_score(zh_h[:NT], [zh_r[:NT]]),
            lambda: swadesh.TER(normalized=True, asian_support=True).corpus_score(zh_h[:NT], [zh_r[:NT]]), max_reps=2)
    rep.add(S, "corpus_bleu() compat function", N, lambda: sacrebleu.corpus_bleu(ende_h, [ende_r]),
            lambda: swadesh.corpus_bleu(ende_h, [ende_r]))

    # ---------------------------------------------------------------- sentence level
    S = "sentence"
    sh, sr = ende_h[:NS], ende_r[:NS]
    for name, mk_inc, mk_new in [
        ("BLEU", lambda: sacrebleu.BLEU(effective_order=True), lambda: swadesh.BLEU(effective_order=True)),
        ("chrF", sacrebleu.CHRF, swadesh.CHRF),
        ("chrF++", lambda: sacrebleu.CHRF(word_order=2), lambda: swadesh.CHRF(word_order=2)),
    ]:
        m_i, m_n = mk_inc(), mk_new()
        rep.add(S, f"{name} batch (sentence_scores vs loop)", NS,
                lambda: [m_i.sentence_score(h, [r]).score for h, r in zip(sh, sr)],
                lambda: m_n.sentence_scores(sh, [sr]), max_reps=2)
        rep.add(S, f"{name} sentence_score() in a loop", NS,
                lambda: [m_i.sentence_score(h, [r]).score for h, r in zip(sh, sr)],
                lambda: [m_n.sentence_score(h, [r]).score for h, r in zip(sh, sr)], max_reps=2)
    t_i, t_n = sacrebleu.TER(), swadesh.TER()
    rep.add(S, "TER sentence_score() in a loop", min(NS, 2000),
            lambda: [t_i.sentence_score(h, [r]).score for h, r in zip(sh[:2000], sr[:2000])],
            lambda: [t_n.sentence_score(h, [r]).score for h, r in zip(sh[:2000], sr[:2000])], max_reps=2)
    rep.add(S, "sentence_bleu() compat function x1000", 1000,
            lambda: [sacrebleu.sentence_bleu(h, [r]).score for h, r in zip(sh[:1000], sr[:1000])],
            lambda: [swadesh.sentence_bleu(h, [r]).score for h, r in zip(sh[:1000], sr[:1000])])

    # ---------------------------------------------------------------- significance
    S = "significance"
    bh, br = ende_h[:NS], [ende_r[:NS]]
    for name, mk_inc, mk_new in [("BLEU", sacrebleu.BLEU, swadesh.BLEU), ("chrF", sacrebleu.CHRF, swadesh.CHRF),
                                 ("TER", sacrebleu.TER, swadesh.TER)]:
        nb = 1000 if name != "TER" else 200
        nn = NS if name != "TER" else min(NS, 2000)
        rep.add(S, f"{name} bootstrap CI ({nb} resamples)", nn,
                lambda: mk_inc().corpus_score(bh[:nn], [br[0][:nn]], n_bootstrap=nb),
                lambda: mk_new().corpus_score(bh[:nn], [br[0][:nn]], n_bootstrap=nb), max_reps=2)
    sys2 = perturb(read("en-de"), 21, NS)[0]
    sys3 = perturb(read("en-de"), 22, NS)[0]
    named = [("base", bh), ("sys2", sys2), ("sys3", sys3)]

    def paired(PT, M, test_type, n):
        sigs, scores = PT(named, {"BLEU": M.BLEU(references=br), "chrF": M.CHRF(references=br)}, None,
                          test_type=test_type, n_samples=n)()
        return {k: str(v) for k, v in sigs.items()}, {k: [repr(x) for x in v] for k, v in scores.items()}

    rep.add(S, "paired bootstrap, 3 systems, BLEU+chrF", NS,
            lambda: paired(IncPairedTest, sacrebleu, "bs", 1000), lambda: paired(swadesh.PairedTest, swadesh, "bs", 1000),
            max_reps=1)
    rep.add(S, "paired approx. randomization (2000)", NS,
            lambda: paired(IncPairedTest, sacrebleu, "ar", 2000), lambda: paired(swadesh.PairedTest, swadesh, "ar", 2000),
            max_reps=1)

    # ---------------------------------------------------------------- MBR / pairwise
    S = "mbr"
    rnd = random.Random(5)
    for k in (32, 128):
        cands = rnd.sample(ende_r, k)
        for name, mk_inc, mk_new in [("chrF", sacrebleu.CHRF, swadesh.CHRF),
                                     ("BLEU", lambda: sacrebleu.BLEU(effective_order=True),
                                      lambda: swadesh.BLEU(effective_order=True))]:
            m_i, m_n = mk_inc(), mk_new()
            rep.add(S, f"{name} {k}x{k} matrix", k * k,
                    lambda: [[m_i.sentence_score(c, [r]).score for r in cands] for c in cands],
                    lambda: m_n.pairwise(cands), max_reps=2)

    # ---------------------------------------------------------------- tokenizers
    S = "tokenize"
    from sacrebleu.metrics.bleu import _get_tokenizer
    for tok, lines in [("13a", ende_r), ("intl", deen_r), ("zh", zh_r), ("char", ja_r)]:
        lines = list(dict.fromkeys(lines))  # unique lines: the corpus repeats, which would feed sacreBLEU's memo cache
        t = _get_tokenizer(tok)()
        cache = type(t).__call__  # sacreBLEU memoizes tokenization; clear it so every run is cold

        def inc(t=t, cache=cache, lines=lines):
            for fn in (cache, getattr(type(getattr(t, "_post_tokenizer", None)), "__call__", None)):
                if hasattr(fn, "cache_clear"):
                    fn.cache_clear()  # outer and inner (TokenizerRegexp) memoization
            return [t(l) for l in lines]

        rep.add(S, f"tokenize {tok} (cold cache, unique lines)", len(lines), inc, lambda lines=lines, tok=tok: swadesh.tokenize(lines, tok),
                max_reps=3)

    # ---------------------------------------------------------------- CLI + process level
    S = "cli"
    tmp = tempfile.mkdtemp()
    ref_f, hyp_f, hyp2_f = (os.path.join(tmp, x) for x in ("ref.txt", "hyp.txt", "hyp2.txt"))
    for path, lines in [(ref_f, ende_r), (hyp_f, ende_h), (hyp2_f, perturb(read("en-de"), 30, N)[0])]:
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    bindir = os.path.dirname(sys.executable)
    inc_cli = os.path.join(bindir, "sacrebleu")
    py_cli = os.path.join(bindir, "swadesh")
    rust_cli = os.path.abspath(a.swadesh_bin)

    def cli(binary, args):
        def f():
            p = subprocess.run([binary, ref_f, "-i", hyp_f, *args], capture_output=True, text=True)
            return p.returncode, p.stdout
        return f

    for args in (["-m", "bleu", "chrf"], ["-m", "ter"], ["-m", "bleu", "chrf", "ter", "-f", "text"]):
        label = " ".join(args)
        rep.add(S, f"python CLI: {label}", N, cli(inc_cli, args), cli(py_cli, args), max_reps=2)
        if os.path.exists(rust_cli):
            rep.add(S, f"rust binary: {label}", N, cli(inc_cli, args + ["--python-version", "3.12"])
                    if False else cli(inc_cli, args),
                    cli(rust_cli, args + ["--python-version", f"{sys.version_info[0]}.{sys.version_info[1]}"]),
                    max_reps=2, note="includes interpreter/process start-up")
    rep.add(S, "python CLI: --confidence (BLEU)", N, cli(inc_cli, ["-m", "bleu", "--confidence", "-f", "text"]),
            cli(py_cli, ["-m", "bleu", "--confidence", "-f", "text"]), max_reps=1)
    pargs = ["-m", "bleu", "chrf", "--paired-bs", "-f", "text"]
    rep.add(S, "python CLI: --paired-bs, 2 systems", N,
            lambda: subprocess.run([inc_cli, ref_f, "-i", hyp_f, hyp2_f, *pargs], capture_output=True, text=True).stdout,
            lambda: subprocess.run([py_cli, ref_f, "-i", hyp_f, hyp2_f, *pargs], capture_output=True, text=True).stdout,
            max_reps=1)
    shutil.rmtree(tmp, ignore_errors=True)

    # ---------------------------------------------------------------- import time and memory
    S = "process"
    imp_i = min(run_child("import sacrebleu\n" + MAXRSS)[0] for _ in range(3))
    imp_n = min(run_child("import swadesh\n" + MAXRSS)[0] for _ in range(3))
    rep.rows.append(dict(section=S, case="interpreter + import", n=1, incumbent_s=imp_i, swadesh_s=imp_n,
                         speedup=imp_i / imp_n, exact=True, note="swadesh imports sacrebleu too"))
    print(f"{S:12s} import: sacrebleu {imp_i:.3f}s swadesh {imp_n:.3f}s", flush=True)
    mem_code = ("import sys;sys.path.insert(0,{here!r});import compare as c;import {mod} as M;"
                "h,r=c.perturb(c.read('en-de'),1,{n});M.{metric}().corpus_score(h,[r]{extra})\n" + MAXRSS)
    for metric, extra, n in [("BLEU", "", N), ("CHRF", "", N), ("BLEU", ",n_bootstrap=1000", NS)]:
        mi = run_child(mem_code.format(here=HERE, mod="sacrebleu", n=n, metric=metric, extra=extra))[1]
        mn = run_child(mem_code.format(here=HERE, mod="swadesh", n=n, metric=metric, extra=extra))[1]
        rep.rows.append(dict(section="memory", case=f"peak RSS MB: {metric}{extra}", n=n, incumbent_s=mi, swadesh_s=mn,
                             speedup=mi / mn, exact=True, note="values are MB, ratio = incumbent/swadesh"))
        print(f"memory       {metric}{extra}: sacrebleu {mi:.0f} MB swadesh {mn:.0f} MB", flush=True)

    # ---------------------------------------------------------------- write
    meta = dict(machine=platform.node(), arch=platform.machine(), system=platform.platform(),
                python=sys.version.split()[0], numpy=np.__version__, sacrebleu=sacrebleu.__version__,
                swadesh=swadesh.__version__, cpus=os.cpu_count(), lines=N,
                threads=os.environ.get("RAYON_NUM_THREADS", "all"),
                load=os.getloadavg() if hasattr(os, "getloadavg") else None,
                date=time.strftime("%Y-%m-%d %H:%M:%S"))
    out = a.out or os.path.join(HERE, "results", f"compare-{platform.node()}")
    with open(out + ".json", "w") as f:
        json.dump(dict(meta=meta, rows=rep.rows), f, indent=1)
    with open(out + ".md", "w") as f:
        f.write(render(meta, rep.rows))
    n_diff = sum(not r["exact"] for r in rep.rows)
    print(f"\nwrote {out}.json and .md; {len(rep.rows)} rows, {n_diff} non-identical")
    return 1 if n_diff else 0


def render(meta, rows):
    out = [f"# swadesh vs sacreBLEU: comparison report\n",
           f"{meta['date']} · {meta['machine']} ({meta['arch']}, {meta['cpus']} CPUs, threads={meta['threads']}) · "
           f"Python {meta['python']} · numpy {meta['numpy']} · sacreBLEU {meta['sacrebleu']} · swadesh {meta['swadesh']} · "
           f"load {tuple(round(x, 2) for x in meta['load']) if meta['load'] else 'n/a'}\n",
           "Times are the best of up to 5 repetitions (1-2 for slow cases). \"Identical\" means the full results "
           "(score objects, strings, CIs, p-values, matrices, CLI stdout) compare equal. The corpora cycle the WMT22 "
           "test set, so sacreBLEU's tokenizer memo cache hits often: speedups are conservative.\n"]
    section = None
    for r in rows:
        if r["section"] != section:
            section = r["section"]
            unit = "MB" if section == "memory" else "s"
            out.append(f"\n## {section}\n\n| case | n | sacreBLEU ({unit}) | swadesh ({unit}) | ratio | identical |\n"
                       "|---|---:|---:|---:|---:|:---:|")
        fmt = (lambda v: f"{v:.0f}") if section == "memory" else (lambda v: f"{v:.4f}" if v >= 0.01 else f"{v:.5f}")
        out.append(f"| {r['case']} | {r['n']:,} | {fmt(r['incumbent_s'])} | {fmt(r['swadesh_s'])} | "
                   f"**{r['speedup']:.1f}×** | {'yes' if r['exact'] else '**NO**'} |")
    timing = [r for r in rows if r["section"] not in ("memory", "process")]
    sp = sorted(r["speedup"] for r in timing)
    out.append(f"\n**Summary:** {len(timing)} timed comparisons; speedup min {sp[0]:.1f}×, median "
               f"{sp[len(sp) // 2]:.1f}×, max {sp[-1]:.1f}×; "
               f"{sum(not r['exact'] for r in rows)} non-identical results.\n")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    sys.exit(main())

"""Driver for the stress track. Appends one JSON record per run to results/<phase>.jsonl.

    python bench/stress/run_all.py scaling
    python bench/stress/run_all.py memory
    python bench/stress/run_all.py threads
    python bench/stress/run_all.py pathological

Every run goes through `nice -n 10 timeout`; a timeout is recorded as status "timeout".
Where sacreBLEU also runs, the record gets `exact` (score repr and full str() equal).
"""
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_inputs import SIZES, WORK  # noqa: E402

RUN_ONE = os.path.join(HERE, "run_one.py")
RES = os.path.join(HERE, "results")
PY = sys.executable


def loadavg():
    with open("/proc/loadavg") as f:
        return f.read().split()[:3]


def run_one(impl, metric, hyp, refs, timeout, repeat=1, env=None, wrap=(), tag=""):
    cmd = [*wrap, "nice", "-n", "10", "timeout", "-k", "30", str(timeout), PY, RUN_ONE,
           "--impl", impl, "--metric", metric, "--hyp", hyp, "--repeat", str(repeat)]
    for r in refs:
        cmd += ["--ref", r]
    up0 = loadavg()
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **(env or {})})
    rec = dict(tag=tag, impl=impl, metric=metric, hyp=os.path.relpath(hyp, WORK), nrefs=len(refs),
               repeat=repeat, env=env or {}, wall_s=round(time.time() - t0, 3), rc=p.returncode,
               load_before=up0, load_after=loadavg())
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", p.stderr)
    if m:
        rec["time_v_maxrss_kb"] = int(m.group(1))
    if p.returncode in (124, 137):
        rec["status"] = "timeout"
    elif p.returncode != 0:
        rec["status"] = "crash"
        rec["stderr"] = p.stderr[-400:]
    else:
        out = json.loads(p.stdout.strip().splitlines()[-1])
        rec.update(out)
        rec["status"] = "error" if "error" in out else "ok"
    return rec


class Log:
    def __init__(self, phase):
        os.makedirs(RES, exist_ok=True)
        self.path = os.path.join(RES, f"{phase}.jsonl")
        self.f = open(self.path, "a")

    def add(self, rec):
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()
        t = rec.get("time_s")
        ts = f"{t:9.3f}s" if isinstance(t, float) else f"{rec.get('status'):>10s}"
        extra = rec.get("error", "")[:80]
        print(f"{rec['tag']:<26s} {rec['impl']:<9s} {rec['metric']:<8s} {rec['nrefs']:>3d}refs "
              f"{ts} rss={rec.get('rss_peak_kb', 0) // 1024:>6d}MB exact={rec.get('exact')} "
              f"{rec.get('status')} {extra}", flush=True)


def pair(log, tag, metric, hyp, refs, sre_timeout, sw_timeout=1800, repeat=1, sacre=True, env=None, wrap=()):
    w = run_one("swadesh", metric, hyp, refs, sw_timeout, repeat, env=env, wrap=wrap, tag=tag)
    if sacre:
        s = run_one("sacrebleu", metric, hyp, refs, sre_timeout, repeat, wrap=wrap, tag=tag)
        if s.get("status") == "ok" and w.get("status") == "ok":
            w["exact"] = s["str"] == w["str"] and s["score"] == w["score"]
        s["exact"] = w.get("exact") if s.get("status") == "ok" else None
        log.add(w)
        log.add(s)
        return w, s
    log.add(w)
    return w, None


def scaling(log):
    for n in SIZES:
        hyp, ref = f"{WORK}/c{n}.hyp.txt", f"{WORK}/c{n}.ref.txt"
        rep = 3 if n <= 100_000 else 1
        tag = f"scale-{n}"
        big = n >= 10_000_000
        pair(log, tag, "bleu", hyp, [ref], sre_timeout=3600 if big else 1800, repeat=rep, sacre=True)
        pair(log, tag, "chrf", hyp, [ref], sre_timeout=3600 if big else 1800, repeat=rep, sacre=n <= 1_000_000)
        pair(log, tag, "ter", hyp, [ref], sre_timeout=1800, repeat=rep, sacre=n <= 1_000_000)
        if n <= 1_000_000:
            pair(log, tag, "bleu-bs", hyp, [ref], sre_timeout=3600, repeat=rep, sacre=True)
        else:
            pair(log, tag, "bleu-bs", hyp, [ref], sre_timeout=0, repeat=1, sacre=False)


def memory(log):
    hyp, ref = f"{WORK}/c1000000.hyp.txt", f"{WORK}/c1000000.ref.txt"
    for metric in ("bleu", "bleu-bs", "chrf"):
        pair(log, "mem-1M", metric, hyp, [ref], sre_timeout=3600, wrap=("/usr/bin/time", "-v"))


def threads(log):
    hyp, ref = f"{WORK}/c1000000.hyp.txt", f"{WORK}/c1000000.ref.txt"
    for t in (1, 2, 4, 8, 16, 24):
        for metric in ("bleu", "chrf", "ter", "bleu-bs"):
            pair(log, f"threads-{t}", metric, hyp, [ref], sre_timeout=0, repeat=3,
                 env={"RAYON_NUM_THREADS": str(t)}, sacre=False)


def pathological(log):
    p = f"{WORK}/path"
    cases = [
        ("oneline_1mb", ["bleu", "chrf", "ter"], 900),
        ("nospace_100k", ["bleu", "chrf", "ter"], 900),
        ("tiny_1m", ["bleu", "chrf", "ter"], 900),
        ("ter1000w_50", ["ter"], 900),
        ("ter1000w_200", ["ter"], 1800),
        ("identical_1m", ["bleu", "chrf", "ter"], 1800),
        ("empty_hyp_1k", ["bleu", "chrf", "ter"], 300),
        ("all_empty_1k", ["bleu", "chrf", "ter"], 300),
        ("mixed_empty_1k", ["bleu", "chrf", "ter"], 300),
        ("zero_segments", ["bleu", "chrf", "ter"], 300),
        ("one_empty_line", ["bleu", "chrf", "ter"], 300),
    ]
    for name, metrics, sre_to in cases:
        for metric in metrics:
            pair(log, f"path-{name}", metric, f"{p}/{name}.hyp.txt", [f"{p}/{name}.ref.txt"], sre_timeout=sre_to)
    refs50 = [f"{p}/refs50_r{k:02d}.txt" for k in range(50)]
    for metric in ("bleu", "chrf", "ter"):
        pair(log, "path-refs50x10k", metric, f"{p}/refs50.hyp.txt", refs50, sre_timeout=900)


if __name__ == "__main__":
    phase = sys.argv[1]
    log = Log(phase)
    {"scaling": scaling, "memory": memory, "threads": threads, "pathological": pathological}[phase](log)
    print("DONE", phase, flush=True)

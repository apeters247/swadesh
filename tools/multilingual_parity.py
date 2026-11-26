"""Multilingual parity: sacreBLEU 2.6.0 vs swadesh on ten language pairs.

For each pair the references come from the oracle's own cache (`--echo ref`), the hypothesis is a
deterministic perturbation of them (drop a middle word, swap two words, or suffix the last word, on
every third line), and three things are compared:

* the CLI corpus output for BLEU, chrF and TER, byte for byte on stdout, stderr and exit code;
* the CLI sentence-level output for BLEU and for chrF, byte for byte;
* the Python API, corpus scores with `==` and the first 400 sentence scores with `==`.

Run it with the interpreter that has sacreBLEU 2.6.0:

    ~/venv311/bin/python tools/multilingual_parity.py --json /tmp/pairs.json

It writes the hypotheses to a scratch directory, so the run can be repeated exactly. The results are
recorded in `docs/multilingual-parity.md`.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORACLE = os.path.join(os.path.dirname(sys.executable), "sacrebleu")
SWADESH = os.path.join(HERE, "target", "release", "swadesh")

# The set, the pair, extra flags, and why the pair is interesting. The Japanese and Korean pairs pass
# `intl` explicitly: their default tokenizers are mecab-based and mecab is not installed here.
PAIRS = [
    ("wmt19/dev", "en-gu", [], "Gujarati (WMT19 dev), Indic script"),
    ("wmt19/dev", "kk-en", [], "Kazakh to English (WMT19 dev), Cyrillic"),
    ("wmt19/dev", "lt-en", [], "Lithuanian to English (WMT19 dev)"),
    ("iwslt17", "en-ar", [], "Arabic (IWSLT TED), right-to-left"),
    ("iwslt17", "ar-en", [], "Arabic to English (IWSLT TED)"),
    ("iwslt17", "en-ja", ["-tok", "intl"], "Japanese (IWSLT TED), intl tokenizer"),
    ("iwslt17", "ja-en", ["-tok", "intl"], "Japanese to English (IWSLT TED)"),
    ("iwslt17", "ko-en", ["-tok", "intl"], "Korean to English (IWSLT TED)"),
    ("wmt22", "de-fr", [], "German to French (WMT22), neither side English"),
    ("wmt16", "en-tr", [], "Turkish (WMT16), agglutinative"),
]

API_PROG = """
import json, sys
import sacrebleu as SB
import swadesh as SW
d = json.load(open(sys.argv[1]))
refs, hyp, tok = d["refs"], d["hyp"], d["tok"]
out = {}
for name, kw in (("BLEU", {}), ("CHRF", {"word_order": 2}), ("TER", {})):
    if name == "BLEU" and tok:
        kw["tokenize"] = tok
    a = getattr(SB, name)(**kw).corpus_score(hyp, refs).score
    b = getattr(SW, name)(**kw).corpus_score(hyp, refs).score
    out[name] = [a, b, bool(a == b)]
for name in ("BLEU", "CHRF"):
    kw = {"tokenize": tok} if (name == "BLEU" and tok) else {}
    m_sb, m_sw = getattr(SB, name)(**kw), getattr(SW, name)(**kw)
    n, diffs = min(len(hyp), 400), 0
    for h, r in zip(hyp[:n], refs[0][:n]):
        if m_sb.sentence_score(h, [r]).score != m_sw.sentence_score(h, [r]).score:
            diffs += 1
    out["sent_" + name] = [n, diffs]
print(json.dumps(out))
"""


def run(args, env, timeout=1800):
    p = subprocess.run(args, capture_output=True, text=True, env=env, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def perturb(lines):
    out = []
    for i, line in enumerate(lines):
        w = line.split()
        if not w:
            out.append("")
        elif i % 3 == 0 and len(w) > 2:
            del w[len(w) // 2]
            out.append(" ".join(w))
        elif i % 3 == 1 and len(w) > 3:
            w[1], w[2] = w[2], w[1]
            out.append(" ".join(w))
        elif i % 3 == 2 and len(w) > 1:
            w[-1] += "s"
            out.append(" ".join(w))
        else:
            out.append(line)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="write the raw results here")
    ap.add_argument("--work", default=os.path.join(tempfile.gettempdir(), "swadesh-pairs"))
    a = ap.parse_args()
    env = dict(os.environ)
    os.makedirs(a.work, exist_ok=True)
    prog_path = os.path.join(a.work, "api_prog.py")
    open(prog_path, "w", encoding="utf-8").write(API_PROG)

    rows = []
    print(f"{'pair':8s} {'set':12s} {'n':>6s}  cli-corpus  sl-bleu  sl-chrf  api  sent-diffs")
    for set_name, pair, tok, note in PAIRS:
        slug = f"{set_name.replace('/', '_')}.{pair}"
        rc, refs, err = run([ORACLE, "-t", set_name, "-l", pair, *tok, "--echo", "ref"], env)
        if rc != 0:
            rows.append(dict(set=set_name, pair=pair, note=note, error=err.strip().splitlines()[-1:]))
            print(f"{pair:8s} {set_name:12s}  FAILED: {err.strip().splitlines()[-1:]}")
            continue
        ref_lines = refs.split("\n")
        if ref_lines and ref_lines[-1] == "":
            ref_lines.pop()
        hyp = perturb(ref_lines)
        hyp_path = os.path.join(a.work, slug + ".hyp")
        open(hyp_path, "w", encoding="utf-8").write("\n".join(hyp) + "\n")

        args = ["-t", set_name, "-l", pair, *tok, "-i", hyp_path, "-m", "bleu", "chrf", "ter", "-f", "text"]
        rc_o, out_o, err_o = run([ORACLE, *args], env)
        rc_s, out_s, err_s = run([SWADESH, *args, "--python-version", f"{sys.version_info[0]}.{sys.version_info[1]}"], env)
        corpus_ok = (rc_o == rc_s) and (out_o == out_s) and (err_o == err_s)

        sent = {}
        for metric in ("bleu", "chrf"):
            a2 = ["-t", set_name, "-l", pair, *tok, "-i", hyp_path, "-m", metric, "-sl", "-f", "text"]
            r1, o1, e1 = run([ORACLE, *a2], env)
            r2, o2, e2 = run([SWADESH, *a2, "--python-version", f"{sys.version_info[0]}.{sys.version_info[1]}"], env)
            sent[metric] = (r1 == r2) and (o1 == o2) and (e1 == e2)

        data_path = os.path.join(a.work, slug + ".json")
        open(data_path, "w", encoding="utf-8").write(
            json.dumps({"refs": [ref_lines], "hyp": hyp, "tok": (tok[1] if tok else None)}))
        p = subprocess.run([sys.executable, prog_path, data_path], capture_output=True, text=True, env=env, timeout=1800)
        api = json.loads(p.stdout.strip().splitlines()[-1]) if p.stdout.strip() else {"error": p.stderr[-300:]}

        rows.append(dict(set=set_name, pair=pair, note=note, segments=len(ref_lines), corpus_ok=corpus_ok,
                         corpus=out_s.strip(), sent_bleu=sent["bleu"], sent_chrf=sent["chrf"], api=api))
        print(f"{pair:8s} {set_name:12s} {len(ref_lines):6d}  {str(corpus_ok):>10s}  {str(sent['bleu']):>7s}  "
              f"{str(sent['chrf']):>7s}  {str(api.get('BLEU', [0, 0, False])[2]):>5s}  "
              f"{api.get('sent_BLEU', [0, 0])[1]}/{api.get('sent_CHRF', [0, 0])[1]}")

    if a.json:
        open(a.json, "w", encoding="utf-8").write(json.dumps(rows, indent=1))
        print("wrote", a.json)


if __name__ == "__main__":
    main()
"""Differential tests: the `swadesh` binary must print exactly what sacreBLEU 2.6.0's CLI prints.

Every comparison is on raw stdout bytes (and the exit code). The reference is
the `sacrebleu` entry point of the Python environment that runs these tests.

Environment:
  SWADESH_BIN     path to the swadesh binary (default: target/release/swadesh)
  SACREBLEU_BIN   path to the sacrebleu CLI (default: the one next to sys.executable)
  SWADESH_DATA    directory with wmt22.*.ref / .src (default: /var/www/llmeval/bench/data)
Run the timing report with:  pytest tests/test_cli.py -k timing -s
"""

from __future__ import annotations

import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SWADESH = Path(os.environ.get("SWADESH_BIN", ROOT / "target" / "release" / "swadesh"))
SACREBLEU = Path(os.environ.get("SACREBLEU_BIN", Path(sys.executable).with_name("sacrebleu")))
DATA = Path(os.environ.get("SWADESH_DATA", "/var/www/llmeval/bench/data"))

if not SWADESH.exists():
    pytest.skip(f"swadesh binary not found at {SWADESH}; run `cargo build --release -p swadesh-cli`",
                allow_module_level=True)
if not SACREBLEU.exists():
    pytest.skip(f"sacrebleu CLI not found at {SACREBLEU}", allow_module_level=True)

ENV = {k: v for k, v in os.environ.items() if k not in ("SACREBLEU_FORMAT", "NO_COLOR")}


PYVER = f"{sys.version_info.major}.{sys.version_info.minor}"


def run(binary, args, stdin: bytes = b""):
    args = list(args)
    if binary == SWADESH and not any(a in ("--version", "-V", "-h", "--help") for a in args):
        # emulate the interpreter the reference sacreBLEU CLI runs on
        args = args + ["--python-version", PYVER]
    proc = subprocess.run([str(binary), *args], input=stdin, capture_output=True, env=ENV)
    return proc.returncode, proc.stdout, proc.stderr


def assert_same(args, stdin: bytes = b"", cwd=None):
    """Both CLIs must agree on stdout bytes and on success / failure."""
    ref_rc, ref_out, _ = run(SACREBLEU, args, stdin)
    sw_rc, sw_out, sw_err = run(SWADESH, args, stdin)
    assert sw_out == ref_out, (
        f"stdout differs for {args}\n--- sacrebleu ---\n{ref_out.decode('utf-8', 'replace')}\n"
        f"--- swadesh ---\n{sw_out.decode('utf-8', 'replace')}\nstderr: {sw_err.decode(errors='replace')}"
    )
    assert (sw_rc == 0) == (ref_rc == 0), f"exit status differs: sacrebleu {ref_rc}, swadesh {sw_rc}"
    return ref_rc


# --------------------------------------------------------------------- data

def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").split("\n")[:-1]


def perturb(refs: list[str], seed: int) -> dict[str, list[str]]:
    """Plausible and implausible hypotheses for the same references."""
    rng = random.Random(seed)
    out = {"ref": list(refs)}
    out["drop"] = [" ".join(t for t in line.split() if rng.random() > 0.3) for line in refs]
    out["short"] = [" ".join(line.split()[: max(1, len(line.split()) // 2)]) for line in refs]
    out["long"] = [line + " " + refs[(i + 1) % len(refs)] for i, line in enumerate(refs)]
    out["case"] = [line.upper() if i % 3 == 0 else line.title() for i, line in enumerate(refs)]
    out["punct"] = [line.replace(",", " ,").replace(".", " .") + (" ." if i % 4 == 0 else "")
                    for i, line in enumerate(refs)]
    out["empty"] = ["" if rng.random() < 0.2 else line for line in refs]
    out["unicode"] = [line.replace("e", "é").replace("a", "中") for line in refs]
    out["spaces"] = [line.replace(" ", "   　") for line in refs]
    return out


def wmt(pair: str, side: str, n: int | None = None) -> list[str]:
    lines = read_lines(DATA / f"wmt22.{pair}.{side}")
    return lines[:n] if n else lines


EDGE_REFS = [
    "The cat sat on the mat.",
    "",
    "Ünïcödé — ΑΣ ΟΔΟΣ İstanbul ß ẞ ǅ ﬁ",
    "中文 字符 测试 。 ， ！",
    "a  b\tc 　 d\x1c e",
    "x " * 120,
    "one",
    "Hello, world! (test) [x] {y} \"quoted\" 'single' 3.14 $5 50%",
    "emoji 😀 and 𝟎𝟏 digits",
    "trailing spaces   ",
]

EDGE_HYPS = [
    "The cat sat on the mat.",
    "",
    "ünïcödé — ας οδος i̇stanbul ß ẞ ǆ ﬁ",
    "中文 字符 。 ， ！ 测试",
    "a b c d e",
    "x x x x",
    "",
    "Hello world (test) [x] {y} quoted single 3.14 5 50",
    "emoji 😀 and 0 1 digits",
    "trailing spaces",
]


# --------------------------------------------------------------- flag matrix

FLAG_SETS = [
    [],
    ["-b"],
    ["-f", "text"],
    ["-f", "text", "-sh"],
    ["-f", "text", "-b"],
    ["-f", "latex"],
    ["-w", "0"],
    ["-w", "2"],
    ["-w", "4"],
    ["-w", "6"],
    ["-w", "20"],
    ["-w", "3", "-b"],
    ["-m", "chrf"],
    ["-m", "chrf", "-f", "text", "-sh"],
    ["-m", "chrf", "-b"],
    ["-m", "bleu", "chrf"],
    ["-m", "chrf", "bleu", "-f", "text"],
    ["-m", "bleu", "chrf", "-b"],
    ["-m", "bleu", "chrf", "-f", "text", "-sh", "-w", "2"],
    ["-s", "floor"],
    ["-s", "floor", "-sv", "0.5"],
    ["-s", "floor", "-sv", "0.125", "-w", "3", "-f", "text"],
    ["-s", "floor", "-sv", "0.005", "-f", "text", "-sh"],
    ["-s", "add-k"],
    ["-s", "add-k", "-sv", "2", "-f", "text"],
    ["-s", "none", "-f", "text"],
    ["-s", "exp", "-sv", "3"],
    ["-tok", "intl"],
    ["-tok", "char", "-f", "text"],
    ["-tok", "none"],
    ["-tok", "zh", "-f", "text", "-sh"],
    ["-lc"],
    ["-lc", "-f", "text", "-sh"],
    ["-lc", "-m", "chrf", "-f", "text"],
    ["-l", "en-zh"],
    ["-l", "en-de", "-f", "text"],
    ["-cc", "3", "-f", "text"],
    ["-cc", "1", "-cw", "0"],
    ["-cw", "2"],
    ["-m", "chrf", "-cw", "2", "-f", "text", "-sh"],
    ["-m", "chrf", "--chrf-beta", "1"],
    ["-m", "chrf", "--chrf-beta", "3", "-cw", "1", "-f", "text"],
    ["-m", "chrf", "--chrf-whitespace", "-f", "text"],
    ["-m", "chrf", "--chrf-lowercase", "-f", "text"],
    ["-m", "chrf", "--chrf-eps-smoothing"],
    ["-m", "chrf", "--chrf-eps-smoothing", "-cw", "2", "-f", "text"],
    ["-m", "bleu", "chrf", "--chrf-whitespace", "--chrf-lowercase", "-cw", "2", "-lc", "-f", "text"],
    ["-sl"],
    ["-sl", "-f", "text"],
    ["-sl", "-b"],
    ["-sl", "-m", "chrf"],
    ["-sl", "-m", "chrf", "-f", "text", "-sh"],
    ["-sl", "-tok", "intl", "-lc", "-s", "floor", "-sv", "0.1", "-w", "3"],
    ["-sl", "-s", "add-k", "-f", "text", "-sh", "-w", "2"],
    ["--force"],
    ["-q", "-f", "text"],
    ["-nc", "-f", "text", "-sh"],
    ["-f", "json", "-w", "5", "-m", "bleu", "chrf"],
    ["-f", "text", "-m", "chrf", "-cc", "6", "-cw", "2", "--chrf-beta", "2", "-w", "2"],
]

DATASETS = [("en-de", 400), ("zh-en", 400), ("en-zh", 300)]
VARIANTS = ["ref", "drop", "short", "long", "case", "punct", "empty", "unicode"]


def matrix_cases():
    cases = []
    for flags in FLAG_SETS:
        for pair, n in DATASETS:
            # Rotate the hypothesis variant across flag sets so every variant gets covered.
            variant = VARIANTS[(len(cases) * 3 + len(pair)) % len(VARIANTS)]
            cases.append(pytest.param(flags, pair, n, variant, id=f"{pair}-{variant}-{' '.join(flags) or 'default'}"))
    return cases


@pytest.fixture(scope="module")
def wmt_files(tmp_path_factory):
    base = tmp_path_factory.mktemp("wmt")
    files = {}
    for pair in {p for p, _ in DATASETS}:
        refs = wmt(pair, "ref", 400)
        refs_path = base / f"{pair}.ref"
        refs_path.write_text("\n".join(refs) + "\n", encoding="utf-8")
        files[pair] = refs_path
        for name, hyps in perturb(refs, seed=sum(map(ord, pair))).items():
            (base / f"{pair}.{name}.hyp").write_text("\n".join(hyps) + "\n", encoding="utf-8")
    return base


@pytest.mark.parametrize("flags,pair,n,variant", matrix_cases())
def test_matrix(wmt_files, flags, pair, n, variant):
    refs = wmt_files / f"{pair}.ref"
    hyp = wmt_files / f"{pair}.{variant}.hyp"
    assert assert_same([str(refs), "-i", str(hyp), *flags]) == 0


@pytest.mark.parametrize("flags", [
    ["-m", "chrf", "-f", "text"],
    ["-s", "floor", "-sv", "0.2", "-f", "text"],
    ["-tok", "intl", "-lc", "-f", "text", "-sh"],
    ["-cc", "2", "-cw", "1", "-f", "text"],
    [],
])
def test_stdin_matches_file(wmt_files, flags):
    """Hypotheses from stdin give the same result as the same file via -i."""
    refs = wmt_files / "en-de.ref"
    stdin = (wmt_files / "en-de.drop.hyp").read_bytes()
    assert assert_same([str(refs), *flags], stdin=stdin) == 0


# ---------------------------------------------------------- synthetic inputs

@pytest.fixture
def edge_files(tmp_path):
    refs = tmp_path / "edge.ref"
    refs.write_text("\n".join(EDGE_REFS) + "\n", encoding="utf-8")
    hyp = tmp_path / "edge.hyp"
    hyp.write_text("\n".join(EDGE_HYPS) + "\n", encoding="utf-8")
    return refs, hyp


EDGE_FLAGS = [
    [],
    ["-m", "bleu", "chrf", "-f", "text", "-sh"],
    ["-lc", "-tok", "intl", "-s", "floor", "-f", "text"],
    ["-s", "add-k", "-sv", "1.5", "-w", "3", "-f", "text"],
    ["-m", "chrf", "--chrf-whitespace", "-cw", "2", "-f", "text"],
    ["-m", "chrf", "--chrf-lowercase", "--chrf-eps-smoothing", "-cc", "4", "-f", "text"],
    ["-tok", "zh", "-s", "none", "-f", "text"],
    ["-tok", "char", "-f", "text", "-w", "5"],
]


@pytest.mark.parametrize("flags", EDGE_FLAGS, ids=lambda f: " ".join(f) or "default")
def test_edge_inputs_file(edge_files, flags):
    refs, hyp = edge_files
    assert assert_same([str(refs), "-i", str(hyp), *flags]) == 0


@pytest.mark.parametrize("flags", EDGE_FLAGS, ids=lambda f: " ".join(f) or "default")
def test_edge_inputs_stdin(edge_files, flags):
    refs, hyp = edge_files
    assert assert_same([str(refs), *flags], stdin=hyp.read_bytes()) == 0


@pytest.mark.parametrize("stdin", [
    b"a b c\r\nd e f\r\n",            # CRLF is a line break on stdin
    b"a b c\rd e f\r",                # lone CR: universal newlines split the line
    b"a b c\n\n\n",                   # trailing empty lines are real segments
    b"a b c",                         # no trailing newline
    b"\n",                            # a single empty segment
    b"",                              # empty stdin is one empty segment
    "ünïcödé 中文\n".encode(),
    b"a\t\n",                         # trailing tab is stripped, one field
])
@pytest.mark.parametrize("refs_text", [
    "a b c\nd e f\n",
    "a b c\r\nd e f\r\n",
    "a b c\nd e f",
    "a b c\n\n\n",
    "a b c\n",
])
def test_stdin_line_endings(tmp_path, stdin, refs_text):
    refs = tmp_path / "r.txt"
    refs.write_bytes(refs_text.encode())
    assert_same([str(refs), "-f", "text"], stdin=stdin)
    assert_same([str(refs), "-m", "chrf", "-b"], stdin=stdin)


@pytest.mark.parametrize("hyp_bytes", [b"a b c\nd e f\n", b"a b c\r\nd e f\r\n", b"a b\r\n\r\nc d e\n"])
def test_file_line_endings(tmp_path, hyp_bytes):
    refs = tmp_path / "r.txt"
    refs.write_bytes(b"a b c\nd e f\n")
    hyp = tmp_path / "h.txt"
    hyp.write_bytes(hyp_bytes)
    assert_same([str(refs), "-i", str(hyp), "-f", "text"])


# ---------------------------------------------------------- multiple references

def test_multiple_ref_files(tmp_path):
    refs = []
    for k in range(3):
        p = tmp_path / f"r{k}.txt"
        lines = wmt("en-de", "ref", 150)
        p.write_text("\n".join(line if k == 0 else line[::-1] + " " + line for line in lines) + "\n",
                     encoding="utf-8")
        refs.append(str(p))
    hyp = tmp_path / "h.txt"
    hyp.write_text("\n".join(wmt("en-de", "ref", 150)) + "\n", encoding="utf-8")
    for flags in ([], ["-m", "bleu", "chrf", "-f", "text"], ["-s", "floor", "-sv", "0.3"]):
        assert_same([*refs, "-i", str(hyp), *flags])


@pytest.mark.parametrize("nr", ["2", "3"])
@pytest.mark.parametrize("flags", [[], ["-m", "chrf", "-f", "text"], ["-f", "text", "-sh", "-s", "floor"]])
def test_num_refs_tab_separated(tmp_path, nr, flags):
    k = int(nr)
    lines = wmt("de-en", "ref", 120)
    refs_tsv = tmp_path / "refs.tsv"
    refs_tsv.write_text(
        "\n".join("\t".join([line] + [line.lower()] * (k - 1)) for line in lines) + "\n",
        encoding="utf-8",
    )
    hyp = tmp_path / "h.txt"
    hyp.write_text("\n".join(lines[::-1][:120]) + "\n", encoding="utf-8")
    assert_same([str(refs_tsv), "-nr", nr, "-i", str(hyp), *flags])


def test_num_refs_field_mismatch(tmp_path):
    refs_tsv = tmp_path / "refs.tsv"
    refs_tsv.write_text("a\tb\nc\n", encoding="utf-8")
    hyp = tmp_path / "h.txt"
    hyp.write_text("a\nc\n", encoding="utf-8")
    rc = assert_same([str(refs_tsv), "-nr", "2", "-i", str(hyp)])
    assert rc == 17


# ------------------------------------------------------------------ error paths

@pytest.mark.parametrize("args,stdin", [
    (["-i", "{hyp}"], b""),                             # no references at all
    (["{refs_short}", "-i", "{hyp}"], b""),             # length mismatch
    (["{empty}", "-i", "{hyp}"], b""),                  # empty reference file vs hypothesis
    (["{refs_short}", "-i", "{empty}"], b""),           # empty hypothesis file
    (["{refs}", "-i", "{missing}"], b""),               # missing file
    (["{refs}", "-nr", "2", "{refs}"], b""),            # -nr with two files
    (["{refs}", "-m", "chrf", "bleu", "-sl"], b"a\nb\n"),  # two metrics in sentence mode
    (["{refs}"], b"a\nb\nc\n"),                         # stdin length mismatch
    (["{refs}"], b"a\rb\rc\n"),                         # lone CR splits stdin into 3 lines
    (["{refs}", "-cc", "6", "-m", "chrf", "-w", "-1"], b"x\n"),  # negative width
])
def test_error_exit_codes_match(tmp_path, args, stdin):
    refs = tmp_path / "refs.txt"
    refs.write_text("a\nb\n", encoding="utf-8")
    refs_short = tmp_path / "refs_short.txt"
    refs_short.write_text("a\n", encoding="utf-8")
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    hyp = tmp_path / "hyp.txt"
    hyp.write_text("a\nb\n", encoding="utf-8")
    mapping = {"refs": refs, "refs_short": refs_short, "empty": empty, "hyp": hyp,
               "missing": tmp_path / "nope.txt"}
    resolved = [a.format(**{k: str(v) for k, v in mapping.items()}) for a in args]
    ref_rc, ref_out, _ = run(SACREBLEU, resolved, stdin)
    sw_rc, sw_out, _ = run(SWADESH, resolved, stdin)
    assert ref_rc != 0, "sacrebleu was expected to fail on this input"
    assert sw_out == ref_out
    assert sw_rc == ref_rc, f"exit codes differ: sacrebleu {ref_rc}, swadesh {sw_rc}"


@pytest.mark.parametrize("args", [
    ["x.txt", "-i", "y.txt", "-tok", "ja-mecab"],
    ["x.txt", "-i", "y.txt", "-tok", "ko-mecab"],
    ["x.txt", "-i", "y.txt", "-tok", "spm"],
    ["x.txt", "-i", "y.txt", "-tok", "flores101"],
    ["x.txt", "-i", "y.txt", "-tok", "flores200"],
    ["x.txt", "-i", "y.txt", "-tok", "spBLEU-1K"],
    ["x.txt", "-i", "y.txt", "-l", "en-ja"],
    ["x.txt", "-i", "y.txt", "-e", "utf-16"],
])
def test_unsupported_options_fail_with_hint(tmp_path, args):
    rc, out, err = run(SWADESH, args)
    assert rc == 2, err
    assert out == b""
    assert err.startswith(b"swadesh: error:"), err


# ------------------------------------------------------------------- .gz input

def _assert_same_error_line(args):
    """For the cases where sacreBLEU raises: exit code, stdout and the *final* stderr line.

    The traceback frames above that line differ (different entry points) and are the documented
    divergence in `docs/parity-report.md`; everything else must match.
    """
    ref_rc, ref_out, ref_err = run(SACREBLEU, args)
    sw_rc, sw_out, sw_err = run(SWADESH, args)
    assert sw_rc == ref_rc, (args, ref_rc, sw_rc)
    assert sw_out == ref_out, args
    assert sw_err.strip().splitlines()[-1] == ref_err.strip().splitlines()[-1], (args, ref_err, sw_err)


def test_gzip_input_matches_python(tmp_path):
    """`-i x.gz` decompresses exactly as sacreBLEU does, error lines included.

    Measured: a `.gz` *name* means gzip, case-sensitively (`.GZ` is read raw and fails to decode); an
    empty `.gz` is an empty stream; a file that is not gzip raises
    `gzip.BadGzipFile: Not a gzipped file (b'pl')` with the first two bytes repr'd; and a truncated
    stream raises `EOFError: Compressed file ended before the end-of-stream marker was reached`.
    """
    import gzip as gziplib

    text = "The cat sat on the mat.\nHello world.\n"
    plain = tmp_path / "lines.txt"
    plain.write_text(text, encoding="utf-8")
    good = tmp_path / "lines.txt.gz"
    good.write_bytes(gziplib.compress(text.encode("utf-8")))

    assert_same_streams([str(good), "-i", str(plain), "-f", "text"])
    assert_same_streams([str(good), "-i", str(good), "-f", "text"])
    assert_same_streams(["-e", "latin-1", str(good), "-i", str(plain), "-f", "text"])

    upper = tmp_path / "lines.GZ"
    upper.write_bytes(good.read_bytes())
    _assert_same_error_line([str(upper), "-i", str(plain), "-f", "text"])

    empty = tmp_path / "empty.gz"
    empty.write_bytes(b"")
    assert_same_streams([str(empty), "-i", str(plain), "-f", "text"])

    for payload in (b"plain\n", b"\x00\x01abc\n", gziplib.compress(text.encode("utf-8"))[:12]):
        bad = tmp_path / "bad.gz"
        bad.write_bytes(payload)
        _assert_same_error_line([str(bad), "-i", str(plain), "-f", "text"])


# --------------------------------------------------------------- -e / --encoding

# The payloads are bytes: cp1252's 0x80..=0x9F block cannot be written as a Python str.
_ENCODED = [
    ("latin-1", b"caf\xe9 noir\nT\xe9l\xe9\n"),
    ("latin1", b"caf\xe9 noir\n"),
    ("iso-8859-1", b"d\xe9j\xe0 vu\n"),
    ("cp1252", b"a\x93b\x94c \x80\nplain\n"),
    ("windows-1252", b"a\x93b\n"),
    ("utf-8-sig", b"\xef\xbb\xbfcaf\xc3\xa9 noir\n"),
    ("ascii", b"plain ascii only\n"),
]


@pytest.mark.parametrize("encoding,raw", _ENCODED)
def test_encoding_flag_decodes_like_python(tmp_path, encoding, raw):
    """`-e` names a codec, and files and stdin are decoded the way Python decodes them.

    Measured: sacreBLEU's `-e` takes any Python codec name. swadesh builds in the dependency-free set
    that MT data actually ships in - utf-8, utf-8-sig, ascii, latin-1 and cp1252, with their aliases -
    and refuses anything else with a hint (`docs/parity-report.md`).
    """
    path = tmp_path / "data.txt"
    path.write_bytes(raw)
    assert_same(["-e", encoding, str(path), "-i", str(path), "-f", "text"])
    assert_same(["-e", encoding, str(path), "-f", "text"], stdin=raw)


@pytest.mark.parametrize("encoding,raw,last", [
    ("ascii", b"high \xe9 byte\n", "'ascii' codec can't decode byte 0xe9 in position 5"),
    ("cp1252", b"bad \x81 byte\n", "'charmap' codec can't decode byte 0x81 in position 4"),
    ("utf-8", b"caf \xe9 noir\n", "'utf-8' codec can't decode byte 0xe9 in position 4"),
])
def test_encoding_flag_error_line_matches_python(tmp_path, encoding, raw, last):
    """A decode failure exits 1, prints nothing on stdout, and ends in Python's own line.

    The traceback *frames* above it are not reproduced (the entry points differ), which is the
    documented divergence in `docs/parity-report.md`.
    """
    path = tmp_path / "bad.txt"
    path.write_bytes(raw)
    ref_rc, ref_out, ref_err = run(SACREBLEU, ["-e", encoding, str(path), "-i", str(path)])
    sw_rc, sw_out, sw_err = run(SWADESH, ["-e", encoding, str(path), "-i", str(path)])
    assert sw_rc == ref_rc == 1, (ref_rc, sw_rc)
    assert sw_out == ref_out
    assert sw_err.strip().splitlines()[-1] == ref_err.strip().splitlines()[-1]
    assert last in sw_err.decode()


def test_version_reports_both(tmp_path):
    rc, out, _ = run(SWADESH, ["--version"])
    assert rc == 0
    assert b"2.6.0" in out and b"swadesh" in out
    _, ref_out, _ = run(SACREBLEU, ["--version"])
    assert b"2.6.0" in ref_out


@pytest.mark.parametrize("fmt", ["text", "json", "bogus"])
def test_sacrebleu_format_env_override(tmp_path, fmt):
    """SACREBLEU_FORMAT overrides -f (valid values only)."""
    refs = wmt("en-de", "ref", 50)
    rp = tmp_path / "r.txt"
    rp.write_text("\n".join(refs) + "\n", encoding="utf-8")
    hyp = tmp_path / "h.txt"
    hyp.write_text("\n".join(line.lower() for line in refs) + "\n", encoding="utf-8")
    env = {**ENV, "SACREBLEU_FORMAT": fmt}

    def go(binary):
        return subprocess.run([str(binary), str(rp), "-i", str(hyp), "-m", "bleu", "chrf"],
                              capture_output=True, env=env).stdout

    assert go(SWADESH) == go(SACREBLEU)


# -------------------------------------------- max n-gram order (Python API only)

@pytest.mark.parametrize("order", [1, 2, 3, 5, 9])
def test_max_ngram_order_matches_python_api(wmt_files, order):
    """sacreBLEU's CLI has no max-order flag, so compare with the BLEU class API."""
    from sacrebleu.metrics import BLEU

    refs = read_lines(wmt_files / "en-de.ref")
    hyps = read_lines(wmt_files / "en-de.drop.hyp")
    metric = BLEU(max_ngram_order=order)
    score = metric.corpus_score(hyps, [refs])
    expected = score.format(1, False, metric.get_signature().format(False), True)
    rc, out, _ = run(SWADESH, [str(wmt_files / "en-de.ref"), "-i", str(wmt_files / "en-de.drop.hyp"),
                               "--max-ngram-order", str(order)])
    assert rc == 0
    assert out.decode() == expected + "\n"


# --------------------------------------------------------------- timing report

def test_timing_20k_lines(wmt_files, capsys):
    """Correctness on 20k lines plus the wall time of both CLIs."""
    refs_base = wmt("en-de", "ref")
    hyps_base = perturb(refs_base, seed=7)["drop"]
    reps = max(1, 20000 // len(refs_base) + 1)
    refs = (refs_base * reps)[:20000]
    hyps = (hyps_base * reps)[:20000]
    tmp = wmt_files
    refs_path = tmp / "bench20k.ref"
    hyp_path = tmp / "bench20k.hyp"
    refs_path.write_text("\n".join(refs) + "\n", encoding="utf-8")
    hyp_path.write_text("\n".join(hyps) + "\n", encoding="utf-8")
    timings = {}
    outputs = {}
    for name, binary in (("sacrebleu", SACREBLEU), ("swadesh", SWADESH)):
        best = None
        for _ in range(3):
            t0 = time.perf_counter()
            rc, out, _ = run(binary, [str(refs_path), "-i", str(hyp_path), "-m", "bleu", "chrf"])
            elapsed = time.perf_counter() - t0
            assert rc == 0
            best = elapsed if best is None else min(best, elapsed)
            outputs[name] = out
        timings[name] = best
    assert outputs["swadesh"] == outputs["sacrebleu"]
    with capsys.disabled():
        print(f"\n20k lines, BLEU+chrF, best of 3 wall time: sacrebleu {timings['sacrebleu']:.2f}s, "
              f"swadesh {timings['swadesh']:.2f}s, speedup {timings['sacrebleu'] / timings['swadesh']:.1f}x")


# ------------------------------------------------- more input corner cases

@pytest.mark.parametrize("flags", [["-sl", "-f", "text"], ["-sl", "-m", "chrf", "-f", "text", "-sh"], ["-sl", "-b"]])
def test_sentence_level_multiple_refs(tmp_path, flags):
    lines = wmt("en-de", "ref", 60)
    refs = []
    for k in range(3):
        p = tmp_path / f"r{k}.txt"
        p.write_text("\n".join(line if k == 0 else line[::-1] for line in lines) + "\n", encoding="utf-8")
        refs.append(str(p))
    hyp = tmp_path / "h.txt"
    hyp.write_text("\n".join(line[: len(line) // 2 + 1] for line in lines) + "\n", encoding="utf-8")
    assert_same([*refs, "-i", str(hyp), *flags])


@pytest.mark.parametrize("hyp_bytes", [
    b"\xff\xfe bad utf-8\n",          # invalid UTF-8
    b"ok\nbad \x80 byte\n",
])
def test_invalid_utf8_fails_like_sacrebleu(tmp_path, hyp_bytes):
    refs = tmp_path / "r.txt"
    refs.write_text("ok\nok\n", encoding="utf-8")
    hyp = tmp_path / "h.txt"
    hyp.write_bytes(hyp_bytes)
    assert assert_same([str(refs), "-i", str(hyp)]) != 0
    assert assert_same([str(refs)], stdin=hyp_bytes) != 0


def test_bom_and_unicode_line_separators(tmp_path):
    """A BOM is kept as a character; rstrip removes NEL, LS, VT, FF like CPython does."""
    refs = tmp_path / "r.txt"
    refs.write_bytes("﻿a b c\u0085\nd e f \nx\x0b\n\x0cy\n".encode("utf-8"))
    hyp = tmp_path / "h.txt"
    hyp.write_bytes("﻿a b c\nd e f \nx\x0b\n\x0cy\n".encode("utf-8"))
    for flags in ([], ["-lc", "-m", "chrf", "-f", "text"], ["-tok", "char", "-f", "text"]):
        assert_same([str(refs), "-i", str(hyp), *flags])
        assert_same([str(refs), *flags], stdin=hyp.read_bytes())


@pytest.mark.parametrize("enc", ["utf-8", "UTF8"])
def test_encoding_option_accepts_utf8(tmp_path, enc):
    refs = tmp_path / "r.txt"
    refs.write_text("ä ö\nü\n", encoding="utf-8")
    hyp = tmp_path / "h.txt"
    hyp.write_text("ä ö\nü\n", encoding="utf-8")
    assert_same([str(refs), "-i", str(hyp), "-e", enc, "-f", "text"])


# ---------------------------------------------------------------------- TER

TER_FLAGS = ["--ter-case-sensitive", "--ter-asian-support", "--ter-no-punct", "--ter-normalized"]
# All 16 subsets of the four TER flags.
TER_FLAG_SUBSETS = [[f for i, f in enumerate(TER_FLAGS) if mask >> i & 1] for mask in range(16)]
TER_ALL = TER_FLAGS
TER_PAIRS = [("en-de", 200), ("zh-en", 200), ("en-zh", 200)]
TER_VARIANTS = ["ref", "reorder", "drop", "case", "punct", "unicode", "empty", "short", "long", "spaces"]


def reorder(refs: list[str], seed: int) -> list[str]:
    """Hypotheses with word blocks rotated and moved, so TER has to find shifts."""
    rng = random.Random(seed)
    out = []
    for line in refs:
        words = line.split()
        if len(words) < 4:
            out.append(line)
            continue
        k = len(words) // 3
        words = words[k:] + words[:k]  # rotate a block to the end
        for _ in range(2):  # then move one or two blocks elsewhere
            n = rng.randint(1, 3)
            a = rng.randrange(len(words) - n + 1)
            block, rest = words[a : a + n], words[:a] + words[a + n :]
            b = rng.randrange(len(rest) + 1)
            words = rest[:b] + block + rest[b:]
        out.append(" ".join(words))
    return out


@pytest.fixture(scope="module")
def ter_files(tmp_path_factory):
    """Per pair: `<pair>.ref`, a second reference `<pair>.ref2`, and one hypothesis file per variant."""
    base = tmp_path_factory.mktemp("ter")
    for pair, n in TER_PAIRS:
        refs = wmt(pair, "ref", n)
        seed = sum(map(ord, pair))
        (base / f"{pair}.ref").write_text("\n".join(refs) + "\n", encoding="utf-8")
        hyps = {**perturb(refs, seed=seed), "reorder": reorder(refs, seed)}
        (base / f"{pair}.ref2").write_text("\n".join(hyps["punct"]) + "\n", encoding="utf-8")
        for name in TER_VARIANTS:
            (base / f"{pair}.{name}.hyp").write_text("\n".join(hyps[name]) + "\n", encoding="utf-8")
    return base


def _ids(flags):
    return " ".join(flags) or "default"


@pytest.mark.parametrize("flags", TER_FLAG_SUBSETS, ids=_ids)
@pytest.mark.parametrize("fmt", [["-f", "text"], ["-f", "json"], ["-b"]], ids=["text", "json", "score-only"])
def test_ter_all_flag_subsets(ter_files, flags, fmt):
    """All 16 combinations of the four TER flags on WMT22 en-de with block moves."""
    assert assert_same([str(ter_files / "en-de.ref"), "-i", str(ter_files / "en-de.reorder.hyp"),
                        "-m", "ter", *fmt, *flags]) == 0


@pytest.mark.parametrize("flags", [[], TER_ALL], ids=["default", "all-flags"])
@pytest.mark.parametrize("variant", TER_VARIANTS)
@pytest.mark.parametrize("pair", [p for p, _ in TER_PAIRS])
def test_ter_pairs_and_variants(ter_files, pair, variant, flags):
    assert assert_same([str(ter_files / f"{pair}.ref"), "-i", str(ter_files / f"{pair}.{variant}.hyp"),
                        "-m", "ter", "-f", "text", "-sh", *flags]) == 0


@pytest.mark.parametrize("flags", [[], ["--ter-normalized", "--ter-asian-support"], ["--ter-no-punct"]],
                         ids=_ids)
def test_ter_multiple_reference_files(ter_files, flags):
    assert assert_same([str(ter_files / "en-de.ref"), str(ter_files / "en-de.ref2"),
                        "-i", str(ter_files / "en-de.reorder.hyp"), "-m", "ter", "-f", "text", *flags]) == 0


@pytest.mark.parametrize("flags", [[], TER_ALL, ["--ter-normalized"]], ids=_ids)
def test_ter_tab_separated_references(ter_files, tmp_path, flags):
    """`-nr 2`: two references per line in one tab-separated file."""
    refs = (ter_files / "en-de.ref").read_text(encoding="utf-8").split("\n")[:-1]
    ref2 = (ter_files / "en-de.ref2").read_text(encoding="utf-8").split("\n")[:-1]
    tab = tmp_path / "tab.ref"
    tab.write_text("\n".join(f"{a}\t{b}" for a, b in zip(refs, ref2)) + "\n", encoding="utf-8")
    for fmt in (["-f", "text"], ["-f", "json"], ["-sl", "-f", "text"]):
        assert assert_same([str(tab), "-nr", "2", "-i", str(ter_files / "en-de.reorder.hyp"),
                            "-m", "ter", *fmt, *flags]) == 0


@pytest.mark.parametrize("flags", [
    ["-m", "ter", "-sl", "-f", "text"],
    ["-m", "ter", "-sl", "-b"],
    ["-m", "ter", "-sl", "-sh", "-f", "text", "-w", "3"],
    ["-m", "ter", "-sl", "-f", "json"],
    ["-m", "ter", "-sl", "-b", "-w", "0"],
    ["-m", "ter", "-sl", "--ter-normalized", "--ter-asian-support", "-f", "text"],
    ["-m", "ter", "-sl", "--ter-no-punct", "--ter-case-sensitive", "-b", "-w", "2"],
    ["-sl", "-m", "ter", "--ter-normalized", "--ter-no-punct", "-sh"],
], ids=_ids)
def test_ter_sentence_level(ter_files, flags):
    assert assert_same([str(ter_files / "en-de.ref"), "-i", str(ter_files / "en-de.reorder.hyp"), *flags]) == 0


@pytest.mark.parametrize("pair", [p for p, _ in TER_PAIRS])
def test_ter_sentence_level_two_references(ter_files, pair):
    assert assert_same([str(ter_files / f"{pair}.ref"), str(ter_files / f"{pair}.ref2"),
                        "-i", str(ter_files / f"{pair}.reorder.hyp"), "-m", "ter", "-sl", "-f", "text",
                        *TER_ALL]) == 0


@pytest.mark.parametrize("flags", [
    ["-m", "ter", "bleu", "chrf", "-f", "text", "-sh"],
    ["-m", "bleu", "chrf", "ter", "-b"],
    ["-m", "ter", "chrf", "-w", "3", "-f", "text"],
    ["-m", "chrf", "ter", "-f", "text", "-w", "0"],
    ["-m", "ter", "-f", "text", "-w", "0"],
    ["-m", "ter", "-b", "-w", "4"],
    ["-m", "ter", "-f", "latex"],
    ["-m", "ter", "ter", "-f", "text"],
    ["-m", "ter", "-s", "floor", "-sv", "0.3", "-lc", "-f", "text", "--chrf-whitespace"],
    ["-m", "bleu", "chrf", "ter", "-f", "text", "-sh", "-w", "2", *TER_ALL],
    ["-m", "bleu", "--ter-normalized", "-f", "text"],
], ids=_ids)
def test_ter_with_other_metrics(ter_files, flags):
    assert assert_same([str(ter_files / "en-de.ref"), "-i", str(ter_files / "en-de.reorder.hyp"), *flags]) == 0


@pytest.mark.parametrize("flags", [TER_FLAG_SUBSETS[0], TER_ALL, ["--ter-asian-support", "--ter-normalized"]],
                         ids=_ids)
def test_ter_stdin(ter_files, flags):
    hyp = (ter_files / "zh-en.reorder.hyp").read_bytes()
    assert assert_same([str(ter_files / "zh-en.ref"), "-m", "ter", "-f", "text", *flags], stdin=hyp) == 0


@pytest.mark.parametrize("flags", TER_FLAG_SUBSETS, ids=_ids)
def test_ter_edge_inputs(edge_files, flags):
    """Empty lines, empty references, CJK, Turkish dotted I, final sigma, astral characters."""
    refs, hyp = edge_files
    assert assert_same([str(refs), "-i", str(hyp), "-m", "ter", "-f", "text", *flags]) == 0


@pytest.mark.parametrize("flags", [["-sl", "-f", "text"], ["-sl", "-b"], [*TER_ALL, "-sl", "-f", "text"]],
                         ids=_ids)
def test_ter_edge_inputs_sentence_level(edge_files, flags):
    refs, hyp = edge_files
    assert assert_same([str(refs), "-i", str(hyp), "-m", "ter", *flags]) == 0


def test_ter_all_empty_references(tmp_path):
    """Zero reference words: 100 for an insertion-only hypothesis, 0 when both are empty."""
    refs = tmp_path / "empty.ref"
    refs.write_text("\n\n\n", encoding="utf-8")
    for hyp_text in ["a b c\n\nd\n", "\n\n\n"]:
        hyp = tmp_path / "h.txt"
        hyp.write_text(hyp_text, encoding="utf-8")
        for flags in ([], ["-sl", "-b"], ["-b"]):
            assert assert_same([str(refs), "-i", str(hyp), "-m", "ter", "-f", "text", *flags]) == 0
            assert assert_same([str(refs), "-i", str(hyp), "-m", "ter", *flags]) == 0


def test_ter_sentence_level_with_other_metric_fails(ter_files):
    """Sentence-level mode accepts one metric only, TER included."""
    rc, out, err = run(SWADESH, ["-m", "ter", "chrf", "-sl", str(ter_files / "en-de.ref"),
                                 "-i", str(ter_files / "en-de.ref2")])
    assert rc == 1 and out == b"" and b"Only one metric" in err
    assert assert_same([str(ter_files / "en-de.ref"), "-m", "ter", "chrf", "-sl",
                        "-i", str(ter_files / "en-de.ref2")]) == 1


# ------------------------------------------------------------------ confidence intervals

CI_FLAG_SETS = [
    ["-m", "bleu", "-ci"],
    ["-m", "bleu", "-ci", "-f", "text"],
    ["-m", "bleu", "-ci", "-b"],
    ["-m", "bleu", "-ci", "-w", "3"],
    ["-m", "chrf", "-ci"],
    ["-m", "chrf", "-cw", "2", "-ci", "-f", "text"],
    ["-m", "ter", "-ci"],
    ["-m", "ter", "-ci", "-f", "text"],
    ["-m", "bleu", "chrf", "ter", "-ci"],
    ["-m", "bleu", "-ci", "--confidence-n", "250"],
    ["-m", "bleu", "-ci", "-f", "latex"],
]


@pytest.mark.parametrize("flags", CI_FLAG_SETS, ids=[" ".join(f) for f in CI_FLAG_SETS])
def test_confidence_interval(wmt_files, flags):
    """`-ci`: the bootstrap interval, the `bs`/`seed` signature pairs and the confidence fields."""
    refs = wmt_files / "en-de.ref"
    hyp = wmt_files / "en-de.short.hyp"
    assert assert_same([str(refs), "-i", str(hyp), *flags]) == 0


@pytest.mark.parametrize("pair", [p for p, _ in DATASETS])
def test_confidence_interval_every_pair(wmt_files, pair):
    """The interval has to match for the pairs whose tokenizers differ (zh, intl)."""
    refs = wmt_files / f"{pair}.ref"
    hyp = wmt_files / f"{pair}.short.hyp"
    assert assert_same([str(refs), "-i", str(hyp), "-m", "bleu", "chrf", "ter", "-ci"]) == 0


def test_confidence_seed_env(wmt_files, monkeypatch):
    """`SACREBLEU_SEED` moves the draw, and the signature reports it."""
    refs = wmt_files / "en-de.ref"
    hyp = wmt_files / "en-de.short.hyp"
    for seed in ("12345", "7", "999"):
        monkeypatch.setenv("SACREBLEU_SEED", seed)
        assert assert_same([str(refs), "-i", str(hyp), "-m", "bleu", "-ci", "-f", "text"]) == 0


# ------------------------------------------------------------------ multi-system mode

MULTI_CASES = [
    ["-m", "bleu"],
    ["-m", "bleu", "chrf", "ter"],
    ["-m", "bleu", "-w", "3"],
    ["-m", "chrf", "-b"],
    ["-m", "ter"],
]


@pytest.mark.parametrize("flags", MULTI_CASES, ids=[" ".join(f) for f in MULTI_CASES])
def test_multi_system_json(wmt_files, flags):
    """Several `-i` files switch sacreBLEU to its per-system report, not one score."""
    refs = wmt_files / "en-de.ref"
    a = wmt_files / "en-de.short.hyp"
    b = wmt_files / "en-de.drop.hyp"
    assert assert_same([str(refs), "-i", str(a), str(b), *flags]) == 0


def test_multi_system_stdin_tabs(tmp_path):
    """Tab-separated hypotheses read several systems from stdin, named `System 1`, `System 2`."""
    refs = tmp_path / "r.txt"
    refs.write_text("The cat sat on the mat.\nA dog barked.\nPrices rose.\n", encoding="utf-8")
    stdin = (
        "The cat sits on the mat.\tThe cat sat on the mat.\n"
        "A dog barked loudly.\tA dog barked.\n"
        "Prices fell.\tPrices rose.\n"
    ).encode("utf-8")
    assert assert_same([str(refs), "-m", "bleu", "chrf"], stdin=stdin) == 0


def test_multi_system_duplicate_name(wmt_files):
    """Two systems with the same name: sacreBLEU prints two lines and exits 1, stderr included."""
    refs = wmt_files / "en-de.ref"
    a = wmt_files / "en-de.short.hyp"
    args = [str(refs), "-i", str(a), str(a), "-m", "bleu"]
    ref_rc, ref_out, ref_err = run(SACREBLEU, args)
    sw_rc, sw_out, sw_err = run(SWADESH, args)
    assert ref_out == sw_out == b""
    assert ref_err == sw_err, f"{ref_err!r} != {sw_err!r}"
    assert (sw_rc == 0) == (ref_rc == 0)


# ------------------------------------------------------------------ paired significance

PAIRED_CASES = [
    ["-m", "bleu", "--paired-ar"],
    ["-m", "bleu", "--paired-bs"],
    ["-m", "bleu", "--paired-ar", "--paired-ar-n", "50"],
    ["-m", "bleu", "--paired-bs", "--paired-bs-n", "250"],
    ["-m", "ter", "--paired-ar"],
    ["-m", "bleu", "-par"],
    ["-m", "bleu", "-pbs"],
]


@pytest.mark.parametrize("flags", PAIRED_CASES, ids=[" ".join(f) for f in PAIRED_CASES])
def test_paired_significance(wmt_files, flags):
    """The paired tests: p_value, and mean/ci for the bootstrap one."""
    refs = wmt_files / "en-de.ref"
    a = wmt_files / "en-de.short.hyp"
    b = wmt_files / "en-de.drop.hyp"
    assert assert_same([str(refs), "-i", str(a), str(b), *flags]) == 0


def test_paired_ignores_a_repeated_baseline(wmt_files):
    """A system named like the baseline is dropped with a note, not compared with itself."""
    refs = wmt_files / "en-de.ref"
    a = wmt_files / "en-de.short.hyp"
    b = wmt_files / "en-de.drop.hyp"
    assert assert_same([str(refs), "-i", str(a), str(b), str(a), "-m", "bleu", "--paired-bs"]) == 0


def test_confidence_with_several_systems_is_refused(wmt_files):
    """`-ci` is a single-system report; sacreBLEU points at the paired tests instead."""
    refs = wmt_files / "en-de.ref"
    a = wmt_files / "en-de.short.hyp"
    b = wmt_files / "en-de.drop.hyp"
    args = [str(refs), "-i", str(a), str(b), "-m", "bleu", "-ci"]
    ref_rc, ref_out, ref_err = run(SACREBLEU, args)
    sw_rc, sw_out, sw_err = run(SWADESH, args)
    assert ref_out == sw_out == b""
    assert ref_err == sw_err, f"{ref_err!r} != {sw_err!r}"
    assert (sw_rc == 0) == (ref_rc == 0)


def test_paired_bootstrap_with_a_float32_metric(wmt_files):
    """A documented divergence: sacreBLEU cannot serialise its own interval and crashes.

    `sacrebleu ... -m chrf --paired-bs` (and TER) raises
    `TypeError: Object of type float32 is not JSON serializable` and exits non-zero with no output,
    because the result's fields are numpy `float32` and `json.dumps` refuses them. swadesh reports
    the interval instead of reproducing a Python traceback; its arithmetic is checked against the
    oracle where the oracle works, in the `-ci` parity cases and `tests/test_estimate_ci.py`.
    """
    refs = wmt_files / "en-de.ref"
    a = wmt_files / "en-de.short.hyp"
    b = wmt_files / "en-de.drop.hyp"
    args = [str(refs), "-i", str(a), str(b), "-m", "chrf", "--paired-bs"]
    ref_rc, ref_out, ref_err = run(SACREBLEU, args)
    sw_rc, sw_out, _ = run(SWADESH, args)
    assert ref_rc != 0 and ref_out == b""
    assert b"not JSON serializable" in ref_err
    assert sw_rc == 0 and sw_out.startswith(b"[") and b'"ci"' in sw_out


# ------------------------------------------------------------------ test-set catalogue

def test_list_test_sets():
    """`--list` prints the catalogue and quits before anything is read."""
    assert assert_same(["--list"]) == 0


def test_list_short_circuits_other_arguments():
    """The files given do not exist, and sacreBLEU still just lists the test sets."""
    assert assert_same(["x.txt", "-i", "y.txt", "--list"]) == 0

# ------------------------------------------------- multi-system tables and refusals

MULTI_TABLE_CASES = [
    ["-m", "bleu"],
    ["-m", "bleu", "chrf", "ter"],
    ["-m", "bleu", "-w", "3"],
    ["-m", "chrf", "-cw", "2"],
    ["-m", "bleu", "-sh"],
    ["-m", "bleu", "-b"],
    ["-m", "chrf", "bleu", "-w", "2"],
]


@pytest.mark.parametrize("fmt", ["text", "latex"])
@pytest.mark.parametrize("flags", MULTI_TABLE_CASES, ids=[" ".join(f) for f in MULTI_TABLE_CASES])
def test_multi_system_table(wmt_files, flags, fmt):
    """`-f text`/`-f latex` with several systems, drawn as tabulate's fancy_grid/latex_booktabs."""
    refs = wmt_files / "en-de.ref"
    a = wmt_files / "en-de.short.hyp"
    b = wmt_files / "en-de.drop.hyp"
    assert assert_same([str(refs), "-i", str(a), str(b), *flags, "-f", fmt]) == 0


def assert_same_streams(args, stdin=b""):
    """Like `assert_same`, but the refusal messages on stderr are compared too."""
    ref_rc, ref_out, ref_err = run(SACREBLEU, args, stdin)
    sw_rc, sw_out, sw_err = run(SWADESH, args, stdin)
    assert sw_out == ref_out, args
    assert sw_err == ref_err, (args, ref_err, sw_err)
    assert sw_rc == ref_rc, (args, ref_rc, sw_rc)
    return ref_rc


REFUSAL_CASES = [
    ["{refs}", "-i", "{h1}", "-m", "bleu", "chrf", "-sl"],            # one metric per sentence
    ["{refs}", "-i", "{h1}", "{h2}", "-m", "bleu", "-sl"],            # sentence level, 2 systems
    ["{refs}", "-i", "{h1}", "{h1}", "-m", "bleu"],                   # two systems, same name
    ["{refs}", "-i", "{h1}", "{h1}", "-m", "bleu", "-sl"],            # duplicates before -sl
    ["{refs}", "-i", "{h1}", "{h2}", "-m", "bleu", "-ci"],            # -ci with several systems
    ["{refs}", "-i", "{h1}", "-m", "bleu", "-ci"],                    # ... but fine with one
    ["{refs}", "-i", "{empty}", "-m", "bleu"],                        # a system with no lines
    ["{refs}", "-i", "{one}", "-m", "bleu"],                          # streams of different length
    ["{refs}", "-i", "{h1}", "{h2}", "{h1}", "-m", "bleu", "-pbs"],   # a repeated baseline
    ["{refs}", "-i", "{h1}", "-m", "bleu", "-pbs"],                   # too few for a paired test
    ["{refs}", "-i", "{h1}", "{h2}", "-m", "bleu", "-sl", "-pbs"],    # -sl beats the paired test
    ["{refs}", "-i", "{h1}", "{h1}", "-m", "bleu", "-sl", "-pbs"],    # ... afterwards, uniquely
    ["{refs}", "-i", "{h1}", "{h2}", "-m", "bleu", "-ci", "-pbs"],    # -ci still refused
    ["{refs}", "-i", "{h1}", "{h2}", "-m", "bleu", "-pbs", "-pbsn", "0"],
    ["{refs}", "-i", "{h1}", "{h2}", "-m", "bleu", "-par", "-parn", "-1"],
]


@pytest.mark.parametrize("args", REFUSAL_CASES, ids=lambda a: " ".join(a))
def test_refusal_messages_and_order(tmp_path, args):
    """The refusal messages, and the order sacreBLEU checks them in, match on stderr as well.

    The order is measured, not guessed: names are settled first (the paired tests drop a repeated
    baseline), then duplicates and the sentence-level refusals, then the paired test's own checks,
    then `-ci`, and only then are the streams validated.
    """
    files = {"refs": "a\nb\n", "h1": "a\nb\n", "h2": "c\nd\n", "one": "a\n", "empty": ""}
    for key, text in files.items():
        (tmp_path / f"{key}.txt").write_text(text, encoding="utf-8")
    resolved = [a.format(**{k: str(tmp_path / f"{k}.txt") for k in files}) for a in args]
    assert_same_streams(resolved)

CI_QUIRK_CASES = [
    ["-m", "bleu", "-ci"],                        # the default 1000 resamples
    ["-m", "bleu", "-ci", "-cin", "2"],
    ["-m", "bleu", "-ci", "-cin", "0"],           # 1 or less: -ci is dropped entirely
    ["-m", "bleu", "-ci", "-cin", "1"],
    ["-m", "bleu", "-ci", "-cin", "-1"],
    ["-m", "chrf", "-ci", "-cin", "3"],
    ["-m", "ter", "-ci", "-cin", "3"],
]


@pytest.mark.parametrize("fmt", ["text", "json"])
@pytest.mark.parametrize("flags", CI_QUIRK_CASES, ids=[" ".join(f) for f in CI_QUIRK_CASES])
def test_confidence_edge_cases(wmt_files, flags, fmt):
    """`--confidence-n` of 1 or less drops `-ci` silently: no interval, no `bs`/`seed` pair."""
    refs = wmt_files / "en-de.ref"
    hyp = wmt_files / "en-de.short.hyp"
    assert assert_same_streams([str(refs), "-i", str(hyp), *flags, "-f", fmt]) == 0


@pytest.mark.parametrize("metric", ["bleu", "chrf", "ter"])
@pytest.mark.parametrize("fmt", ["text", "json"])
def test_confidence_is_omitted_when_the_mean_is_zero(tmp_path, metric, fmt):
    """A resampled mean of 0.0 leaves sacreBLEU's interval undefined: it prints the score alone.

    The `bs`/`seed` pair stays in the signature, so this is not the same as `--confidence-n 1`.
    Identical copies of a two-line reference give every metric a 0.0 on one of these metrics.
    """
    refs = tmp_path / "r.txt"
    refs.write_text("a\nb\n", encoding="utf-8")
    hyp = tmp_path / "h.txt"
    hyp.write_text("zz\nqq\n" if metric != "ter" else "a\nb\n", encoding="utf-8")
    assert_same_streams([str(refs), "-i", str(hyp), "-m", metric, "-ci", "-cin", "3", "-f", fmt])

# ------------------------------------------------------- the paired report in text and latex

PAIRED_TABLE_CASES = [
    ["-m", "bleu", "--paired-bs", "--paired-bs-n", "50"],
    ["-m", "bleu", "--paired-ar", "--paired-ar-n", "50"],
    ["-m", "bleu", "chrf", "--paired-bs", "--paired-bs-n", "37"],
    ["-m", "bleu", "chrf", "ter", "--paired-ar", "--paired-ar-n", "11"],
    ["-m", "chrf", "--paired-bs", "--paired-bs-n", "25"],
    ["-m", "ter", "--paired-bs", "--paired-bs-n", "25"],
    ["-m", "bleu", "--paired-bs", "--paired-bs-n", "1"],       # p < 0.05, so a star
    ["-m", "bleu", "--paired-ar", "--paired-ar-n", "50", "-sh"],
]


@pytest.mark.parametrize("fmt", ["text", "latex"])
@pytest.mark.parametrize("flags", PAIRED_TABLE_CASES, ids=[" ".join(f) for f in PAIRED_TABLE_CASES])
def test_paired_report_table(wmt_files, flags, fmt):
    """The paired report's text and latex forms: heading, two-line cells, footer, stderr."""
    refs = wmt_files / "en-de.ref"
    a = wmt_files / "en-de.short.hyp"
    b = wmt_files / "en-de.drop.hyp"
    assert_same_streams([str(refs), "-i", str(a), str(b), *flags, "-f", fmt])


def test_paired_report_stdin_names(tmp_path):
    """Tab-separated systems are named `Baseline`, `System 1`, ... for the paired report.

    The footer's sentence about the baseline is the exception: sacreBLEU prints
    `compared to .` there, with no name at all, which is kept as measured.
    """
    refs = tmp_path / "r.txt"
    refs.write_text("The cat sat on the mat.\nA dog barked.\nPrices rose.\n", encoding="utf-8")
    stdin = (
        "The cat sits on the mat.\tThe cat sat on the mat.\tA cat sat on the mat.\n"
        "A dog barked loudly.\tA dog barked.\tA dog barks.\n"
        "Prices fell.\tPrices rose.\tPrices dropped.\n"
    ).encode("utf-8")
    for fmt in ("text", "latex", "json"):
        assert_same_streams(
            [str(refs), "-m", "bleu", "--paired-bs", "--paired-bs-n", "25", "-f", fmt],
            stdin=stdin,
        )


LATEX_NAMES = ["a_b%c&d.hyp", "a$b#c{d}e~f^g.hyp", 'lt<gt>pipe|quote".hyp', "back\\slash.hyp"]


@pytest.mark.parametrize("fmt", ["latex", "text"])
def test_latex_escapes_special_characters_in_names(tmp_path, fmt):
    """System names with LaTeX specials are escaped (and escaped after padding, so cells widen).

    Measured character by character: `\\`, `_`, `%`, `&`, `$`, `#`, `{`, `}`, `~`, `^`, `<`, `>`.
    """
    refs = tmp_path / "r.txt"
    refs.write_text("The cat sat on the mat.\nA dog barked.\n", encoding="utf-8")
    for name in LATEX_NAMES:
        (tmp_path / name).write_text("The cat sat on the mat.\nA dog barked.\n", encoding="utf-8")
    paths = [str(tmp_path / name) for name in LATEX_NAMES]
    assert_same([str(refs), "-i", *paths, "-m", "bleu", "-f", fmt]) == 0

# --------------------------------------------------------------------------- --citation

CITATION_CASES = [
    ["--citation"],
    ["--cite"],
    ["{refs}", "-i", "{hyp}", "--citation"],
    ["--citation", "-sl"],
    ["--citation", "-m", "bleu", "chrf", "-sl"],
    ["--citation", "--list"],
    ["--list", "--citation"],
    ["--citation", "-cin", "0"],
    ["{refs}", "-i", "{hyp}", "-m", "bleu", "chrf", "--citation"],
]


@pytest.mark.parametrize("args", CITATION_CASES, ids=lambda a: " ".join(a))
def test_citation(tmp_path, args):
    """`--citation` without a test set is refused, in sacreBLEU's own words, on stderr.

    The check sits after the sentence-level one and before the reference checks, which is
    observable: `--citation -m bleu chrf -sl` reports the metric, `--citation -sl` the test set,
    and `--list` short-circuits both. With a test set the oracle itself crashes, so the
    set-specific citation stays with the unsupported `-t` (see below).
    """
    (tmp_path / "refs.txt").write_text("a\nb\n", encoding="utf-8")
    (tmp_path / "hyp.txt").write_text("a\nc\n", encoding="utf-8")
    mapping = {k: str(tmp_path / f"{k}.txt") for k in ("refs", "hyp")}
    resolved = [a.format(**mapping) for a in args]
    assert_same_streams(resolved)
# ------------------------------------------------- argument errors: argparse's usage block

def _norm_argerr(raw: bytes) -> str:
    """The program name is the one intentional difference, and argparse wraps to the terminal
    width, so compare with the name masked and the whitespace collapsed."""
    text = raw.decode("utf-8", "replace").replace("sacrebleu", "PROG").replace("swadesh", "PROG")
    return " ".join(text.split())


ARGUMENT_ERRORS = [
    ["-m", "nope"],
    ["-s", "nope"],
    ["-f", "nope"],
    ["-tok", "nope"],
    ["-w", "x"],
    ["-cin", "x"],
    ["-cc", "x"],
    ["-cw", "x"],
    ["-nr", "x"],
    ["-sv", "x"],
    ["-parn", "x"],
    ["-pbsn", "x"],
    ["-j", "x"],
    ["--chrf-beta", "x"],
    ["--nope"],
    ["-Z"],
    ["-b", "--width"],
    ["--citation", "--nope"],
    ["-par", "-pbs"],
    ["-pbs", "-par"],
    ["--paired-bs", "--paired-ar"],
]


@pytest.mark.parametrize("args", ARGUMENT_ERRORS, ids=lambda a: " ".join(a))
def test_argument_errors_print_the_usage_block(args):
    """Argument errors carry argparse's usage block and exit 2, as sacreBLEU's do.

    The usage text is transcribed from sacreBLEU 2.6.0 at the default 80 columns; the comparison
    masks the program name and collapses whitespace, which is also how a resized terminal is
    tolerated.
    """
    ref_rc, ref_out, ref_err = run(SACREBLEU, args)
    sw_rc, sw_out, sw_err = run(SWADESH, args)
    assert ref_rc == sw_rc == 2, (args, ref_rc, sw_rc)
    assert sw_out == ref_out == b"", args
    assert _norm_argerr(sw_err) == _norm_argerr(ref_err), args


@pytest.fixture(scope="module")
def paired_stdin(tmp_path_factory):
    refs = tmp_path_factory.mktemp("jobs") / "r.txt"
    refs.write_text("The cat sat on the mat.\nA dog barked.\n", encoding="utf-8")
    stdin = (
        "The cat sat on a mat.\tA cat sat on the mat.\n"
        "A dog barks.\tA dog barked loudly.\n"
    ).encode("utf-8")
    return str(refs), stdin


def test_paired_jobs_short_flag(paired_stdin):
    """`-j` is `--paired-jobs` (``-pj`` is not a flag), and `-j 1` is byte-identical."""
    refs, stdin = paired_stdin
    base = [refs, "-m", "bleu", "--paired-bs", "--paired-bs-n", "20"]
    assert_same_streams([*base, "-j", "1", "-f", "json"], stdin=stdin)
    # `-pj` is not a flag: an argument error, so the program name differs by design.
    unreal = [*base, "-pj"]
    ref_rc, ref_out, ref_err = run(SACREBLEU, unreal, stdin)
    sw_rc, sw_out, sw_err = run(SWADESH, unreal, stdin)
    assert ref_rc == sw_rc == 2 and sw_out == ref_out == b""
    assert _norm_argerr(sw_err) == _norm_argerr(ref_err)


def test_paired_jobs_parallel_keeps_the_result(paired_stdin):
    """With `-j 2` the numbers are identical; only the worker status lines are not reproduced."""
    refs, stdin = paired_stdin
    args = [refs, "-m", "bleu", "--paired-bs", "--paired-bs-n", "20", "-j", "2", "-f", "json"]
    ref_rc, ref_out, _ = run(SACREBLEU, args, stdin)
    sw_rc, sw_out, _ = run(SWADESH, args, stdin)
    assert ref_rc == sw_rc == 0
    assert sw_out == ref_out


@pytest.mark.xfail(strict=True, reason=(
    "documented divergence: with -j > 1 sacreBLEU prints `Launching N parallel workers.` and its "
    "fork pool reorders the per-system status lines; swadesh scores serially and prints them in "
    "system order. The scores themselves match (test_paired_jobs_parallel_keeps_the_result)"))
def test_paired_jobs_parallel_status_lines(paired_stdin):
    refs, stdin = paired_stdin
    args = [refs, "-m", "bleu", "--paired-bs", "--paired-bs-n", "20", "-j", "2", "-f", "json"]
    ref_err = run(SACREBLEU, args, stdin)[2]
    sw_err = run(SWADESH, args, stdin)[2]
    assert sw_err == ref_err


@pytest.mark.xfail(strict=True, reason=(
    "documented divergence: argparse lists every unconsumed token (`-pj 1`), swadesh stops at the "
    "first (`-pj`); reproducing argparse's two-phase parse is out of scope for this slice"))
def test_unrecognized_arguments_lists_every_token(tmp_path):
    refs = tmp_path / "r.txt"
    refs.write_text("a\nb\n", encoding="utf-8")
    args = [str(refs), "-m", "bleu", "-pbs", "-pbsn", "5", "-pj", "1"]
    ref_rc, _, ref_err = run(SACREBLEU, args)
    sw_rc, _, sw_err = run(SWADESH, args)
    assert ref_rc == sw_rc == 2
    assert _norm_argerr(sw_err) == _norm_argerr(ref_err)
@pytest.mark.xfail(strict=True, reason=(
    "documented divergence: sacreBLEU's `-cc -1` raises IndexError from inside chrF; swadesh refuses "
    "a negative character order with a clean message and the same exit code (docs/parity-report.md)"))
def test_negative_chrf_char_order(tmp_path):
    refs = tmp_path / "r.txt"
    refs.write_text("the cat sat on the mat\nhello world here\n", encoding="utf-8")
    hyp = tmp_path / "h.txt"
    hyp.write_text("the cat sat on a mat\nhello there world\n", encoding="utf-8")
    assert_same_streams([str(refs), "-i", str(hyp), "-m", "chrf", "-cc", "-1", "-f", "text"])


def test_num_refs_field_count_message_matches(tmp_path):
    """`-nr N` splits each reference on tabs, and a line with the wrong field count is sacreBLEU's own
    diagnostic (`sacreBLEU: FATAL: line 1: expected 2 fields, but found 1.`, exit code 17) - not a
    `swadesh: error:` message. The `-nr` area had carried swadesh's prefix by accident."""
    refs = tmp_path / "r.txt"
    refs.write_text("the cat sat\nhello world\n", encoding="utf-8")
    hyp = tmp_path / "h.txt"
    hyp.write_text("the cat sat\nhello world\n", encoding="utf-8")
    for n in ("0", "2"):
        assert_same_streams(["-nr", n, str(refs), "-i", str(hyp)])


@pytest.mark.xfail(strict=True, reason=(
    "documented divergence: sacreBLEU accepts a negative --chrf-word-order (`nw:-1` in the "
    "signature, a reduced order and a different score); swadesh refuses it with a hint"))
def test_negative_chrf_word_order(tmp_path):
    refs = tmp_path / "r.txt"
    refs.write_text("the cat sat on the mat\nhello world here\n", encoding="utf-8")
    hyp = tmp_path / "h.txt"
    hyp.write_text("the cat sat on a mat\nhello there world\n", encoding="utf-8")
    assert_same_streams([str(refs), "-i", str(hyp), "-m", "chrf", "-cw", "-1", "-f", "text"])

# ------------------------------------------------------- the test-set data layer (-t / --echo)

def _have_set(name: str) -> bool:
    root = Path(os.environ.get("SACREBLEU", Path.home() / ".sacrebleu"))
    return (root / name.split("/")[0]).exists()


def _norm_field_order(text: str) -> str:
    """Field *lists* compared as sets: sacreBLEU's field order is not reproducible.

    Measured over twelve runs, `-t wmt22 --list` printed `ref:A, ref:B` six times and
    `ref:B, ref:A` six times while everything else kept its order. So a field list is compared as a
    set of names (the register row is in docs/parity-report.md); every other byte is compared.
    """
    out = []
    for line in text.splitlines():
        head, sep, rest = line.partition(": ")
        if sep and "," in rest:
            out.append(head + sep + ", ".join(sorted(f.strip() for f in rest.split(","))))
        else:
            out.append(line)
    return "\n".join(out)


def assert_same_fields(args, stdin=b""):
    """Like assert_same_streams, but the two field-list outputs are order-insensitive."""
    ref_rc, ref_out, ref_err = run(SACREBLEU, args, stdin)
    sw_rc, sw_out, sw_err = run(SWADESH, args, stdin)
    assert sw_rc == ref_rc, (args, ref_rc, sw_rc, ref_err, sw_err)
    assert _norm_field_order(sw_out.decode()) == _norm_field_order(ref_out.decode()), args
    assert _norm_field_order(sw_err.decode()) == _norm_field_order(ref_err.decode()), args


TEST_SET_CASES = [
    ["-t", "nope", "-l", "en-de", "--echo", "src"],          # unknown set
    ["-t", "", "-l", "en-de", "--echo", "src"],              # empty name
    ["-t", "wmt16", "-l", "en-xx", "--echo", "src"],          # unknown language pair
    ["-t", "wmt16", "--echo", "src"],                         # language pair missing
    ["-t", "wmt16", "r.txt", "--echo", "src"],                # -t with positional references
    ["-t", "nope", "--list"],                                 # --list of an unknown set: silent, rc 0
]


@pytest.mark.parametrize("args", TEST_SET_CASES, ids=lambda a: " ".join(a) or "empty")
def test_test_set_refusals(tmp_path, args):
    """The `-t` refusals, byte for byte, on stdout, stderr and exit code.

    None of these needs a cached set: the refusals come from the catalogue or from argument
    handling, which is why they also run where no data is present.
    """
    (tmp_path / "r.txt").write_text("a\nb\n", encoding="utf-8")
    resolved = [str(tmp_path / "r.txt") if a == "r.txt" else a for a in args]
    assert_same_streams(resolved)


@pytest.mark.parametrize("args", [
    ["-t", "wmt22", "-l", "en-de", "--echo", "ref:C"],       # a numbered reference that pair lacks
    ["-t", "wmt16", "-l", "en-de", "--echo", "nope"],        # a field no pair has
    ["-t", "wmt21/AB", "--list"],
], ids=lambda a: " ".join(a))
def test_test_set_field_lists(args):
    """`--echo` field errors and `-t X --list` carry the set's fields, compared as a set."""
    assert_same_fields(args)


ECHO_CASES = [
    ["-t", "wmt16", "-l", "en-de", "--echo", "src"],
    ["-t", "wmt16", "-l", "en-de", "--echo", "ref"],
    ["-t", "wmt16", "-l", "en-de", "--echo", "docid"],
    ["-t", "wmt16", "-l", "en-de", "--echo", "origlang"],
    ["-t", "wmt16", "-l", "en-de", "--echo", "genre"],
    ["-t", "wmt16", "-l", "en-de", "--echo", "src", "ref", "docid"],
    ["-t", "wmt16", "-l", "en-de", "--echo", "all"],
    ["-t", "wmt22", "-l", "en-de", "--echo", "ref:A"],
    ["-t", "wmt22", "-l", "en-de", "--echo", "ref:B"],
    ["-t", "wmt24", "-l", "en-de", "--echo", "ref"],
    ["-t", "wmt22", "-l", "cs-en", "--echo", "ref"],
    ["-t", "wmt21/AB", "-l", "en-de", "--echo", "ref:A"],
    ["-t", "wmt22", "-l", "de-fr", "--echo", "Online-A"],
    ["-t", "wmt22", "-l", "de-fr", "--echo", "LT22", "Online-W"],
]


@pytest.mark.parametrize("args", ECHO_CASES, ids=lambda a: " ".join(a))
def test_echo_from_a_cached_set(args):
    """`--echo` prints the set's fields exactly as sacreBLEU does, from the same cache.

    A cached set is required, so this skips (rather than failing) where none is present: the data
    lives in `$SACREBLEU`, not in `$SWADESH_DATA`.
    """
    if not _have_set("wmt16"):
        pytest.skip("no cached test set under $SACREBLEU")
    assert_same_streams(args)


SCORE_FROM_SET = [
    ["-m", "bleu", "chrf", "ter"],
    ["-m", "bleu"],
    ["-sl"],
    ["-ci", "-cin", "40"],
    ["-f", "text"],
]


@pytest.mark.parametrize("flags", SCORE_FROM_SET, ids=lambda a: " ".join(a))
@pytest.mark.parametrize("set_pair", [("wmt16", "en-de"), ("wmt22", "de-fr"), ("wmt24", "en-de")])
def test_score_from_a_cached_set(tmp_path, set_pair, flags):
    """Scoring with `-t`: the reference comes from the set, so no reference file is given."""
    name, pair = set_pair
    if not _have_set(name):
        pytest.skip("no cached test set under $SACREBLEU")
    ref = tmp_path / "hyp.txt"
    ref_rc, ref_out, ref_err = run(SACREBLEU, ["-t", name, "-l", pair, "--echo", "ref"])
    assert ref_rc == 0, ref_err
    ref.write_bytes(ref_out)
    assert_same_streams(["-t", name, "-l", pair, "-i", str(ref), *flags])


@pytest.mark.parametrize("set_pair,expected", [
    ("wmt22", "ref:A"), ("wmt24", "ref:refA"), ("wmt21/AB", "ref:A"),
], ids=lambda x: str(x))
def test_the_default_reference_is_deterministic(tmp_path, set_pair, expected):
    """swadesh's own rule, where sacreBLEU's listing order is random.

    `--echo ref` is the plain `ref` when a set has one, else the first `ref:*` alphabetically - and
    that matches the oracle's *stable* default on every set measured (wmt22 en-de `ref:A`, cs-en
    `ref:B`, wmt24 en-de `ref:refA`, wmt21/AB `ref:A`). The oracle's field *list* order is not
    reproducible, so this pins our behaviour rather than comparing the listing.
    """
    if not _have_set(set_pair.split("/")[0]):
        pytest.skip("no cached test set under $SACREBLEU")
    pair = "en-de" if set_pair != "wmt22" else "en-de"
    ours = run(SWADESH, ["-t", set_pair, "-l", pair, "--echo", "ref"])[1]
    want = run(SWADESH, ["-t", set_pair, "-l", pair, "--echo", expected])[1]
    assert ours == want, set_pair


def test_a_set_in_an_unsupported_format_still_refuses(tmp_path):
    """A set whose archives are not tar+gzip refuses with an accurate message, and never the network.

    The catalogue's URL list is flat and the fetcher treats every URL of a set alike, so a set that
    lists a `.zip` (the four `wmt20/robust/*` sets) or a plain `.gz` (`multi30k/2018`) cannot be
    fetched in part; an empty cache gets the "not in the cache" refusal instead. Multi-archive
    *tar+gzip* sets are fetchable now (see `test_cold_cache_downloads_a_multi_archive_set`): the
    oracle downloads every archive of a set, and so does swadesh.
    """
    cache = tmp_path / "empty-cache"
    cache.mkdir()
    env = dict(ENV)
    env["SACREBLEU"] = str(cache)
    proc = subprocess.run(
        [str(SWADESH), "-t", "wmt20/robust/set1", "-l", "en-de", "--echo", "src",
         "--python-version", PYVER],
        capture_output=True, env=env,
    )
    assert proc.returncode == 1
    assert proc.stdout == b""
    assert b"run `swadesh --download wmt20/robust/set1` first" in proc.stderr


@pytest.mark.skipif(os.environ.get("SWADESH_DOWNLOAD_TESTS") != "1",
                    reason="downloads 3.7 MB twice and needs the network; set SWADESH_DOWNLOAD_TESTS=1")
def test_cold_cache_downloads_and_matches(tmp_path):
    """A cold cache fetches, verifies and extracts, and the parsers do the rest: stdout, stderr and
    every generated file match the oracle performing the same cold download into its own cache.

    The two stderr lines are compared with the cache paths normalised, since the two caches are
    different directories by construction. The lock file sacreBLEU leaves beside the archive is not
    reproduced: it is invisible to every output and the parsers do not read it.
    """
    ours = tmp_path / "ours"
    theirs = tmp_path / "theirs"
    ours.mkdir()
    theirs.mkdir()
    args = ["-t", "wmt16", "-l", "en-de", "--echo", "src", "ref", "docid", "origlang", "genre"]
    ref = _run_env(SACREBLEU, args, {**ENV, "SACREBLEU": str(theirs)})
    own = _run_env(SWADESH, [*args, "--python-version", PYVER], {**ENV, "SACREBLEU": str(ours)})
    assert ref.returncode == own.returncode == 0, (ref.stderr[-300:], own.stderr[-300:])
    assert own.stdout == ref.stdout
    norm = lambda blob: blob.replace(str(ours).encode(), b"CACHE").replace(str(theirs).encode(), b"CACHE")
    assert norm(own.stderr) == norm(ref.stderr)
    for name in ("src", "ref", "docid", "origlang", "genre"):
        assert (ours / "wmt16" / f"wmt16.en-de.{name}").read_bytes() == \
            (theirs / "wmt16" / f"wmt16.en-de.{name}").read_bytes(), name


@pytest.mark.skipif(os.environ.get("SWADESH_DOWNLOAD_TESTS") != "1",
                    reason="downloads ~12 MB twice and needs the network; set SWADESH_DOWNLOAD_TESTS=1")
def test_cold_cache_downloads_a_multi_archive_set(tmp_path):
    """A multi-archive set fetches *every* archive, in registry order, and the IWSLT parser then
    rebuilds the pair - stdout, stderr and every generated file matching the oracle.

    Measured: `-t iwslt17 -l en-de` makes sacreBLEU download and extract all twelve archives even
    though only one pair is parsed, printing the two lines once per URL. swadesh follows the same
    loop, which is why the per-URL stderr matches.
    """
    ours = tmp_path / "ours"
    theirs = tmp_path / "theirs"
    ours.mkdir()
    theirs.mkdir()
    args = ["-t", "iwslt17", "-l", "en-de", "--echo", "all"]
    ref = _run_env(SACREBLEU, args, {**ENV, "SACREBLEU": str(theirs)})
    own = _run_env(SWADESH, [*args, "--python-version", PYVER], {**ENV, "SACREBLEU": str(ours)})
    assert ref.returncode == own.returncode == 0, (ref.stderr[-300:], own.stderr[-300:])
    assert own.stdout == ref.stdout
    norm = lambda blob: blob.replace(str(ours).encode(), b"CACHE").replace(str(theirs).encode(), b"CACHE")
    assert norm(own.stderr) == norm(ref.stderr)
    for name in ("src", "ref", "docid", "genre", "origlang"):
        assert (ours / "iwslt17" / f"iwslt17.en-de.{name}").read_bytes() == \
            (theirs / "iwslt17" / f"iwslt17.en-de.{name}").read_bytes(), name


@pytest.mark.skipif(os.environ.get("SWADESH_DOWNLOAD_TESTS") != "1",
                    reason="downloads 0.8 MB twice and needs the network; set SWADESH_DOWNLOAD_TESTS=1")
def test_download_subcommand_matches(tmp_path):
    """`--download NAME -l PAIR` prints the oracle's two lines and nothing else."""
    ours = tmp_path / "ours"
    theirs = tmp_path / "theirs"
    ours.mkdir()
    theirs.mkdir()
    args = ["--download", "multi30k/2017", "-l", "en-de"]
    ref = _run_env(SACREBLEU, args, {**ENV, "SACREBLEU": str(theirs)})
    own = _run_env(SWADESH, args, {**ENV, "SACREBLEU": str(ours)})
    assert ref.returncode == own.returncode == 0
    norm = lambda blob: blob.replace(str(ours).encode(), b"CACHE").replace(str(theirs).encode(), b"CACHE")
    assert norm(own.stderr) == norm(ref.stderr)


def test_citation_for_a_test_set():
    """Documented divergence: sacreBLEU 2.6.0 crashes on `--citation -t X`
    (`TypeError: argument of type 'WMTXMLDataset' is not iterable`, from its own
    `'citation' not in DATASETS[test_set]` test against Dataset objects). swadesh prints the
    catalogue's citation instead, which is the behaviour a user wants."""
    rc, out, err = run(SWADESH, ["-t", "wmt16", "--citation"])
    assert rc == 0, err
    assert out.startswith(b"@InProceedings")
    ref_rc, ref_out, ref_err = run(SACREBLEU, ["-t", "wmt16", "--citation"])
    assert ref_rc != 0 and ref_out == b"" and b"not iterable" in ref_err


def test_list_with_a_set_and_an_unknown_pair():
    """Documented divergence: sacreBLEU crashes on `-t X -l BAD --list` (traceback, non-zero);
    swadesh lists nothing for that pair and exits 0, like an unknown set."""
    rc, out, err = run(SWADESH, ["-t", "wmt16", "-l", "bogus", "--list"])
    assert rc == 0 and out == b"", (rc, out, err)
    ref_rc, ref_out, _ = run(SACREBLEU, ["-t", "wmt16", "-l", "bogus", "--list"])
    assert ref_rc != 0 and ref_out == b""

# ------------------------------------------------- -ol / --subset (segment filters)

def _hyp_from_set(tmp_path, name, pair):
    """A full-length hypothesis for a set: its own default reference, so scores are 100-ish and a
    filter that masks by position can be told from one that does not."""
    rc, out, err = run(SACREBLEU, ["-t", name, "-l", pair, "--echo", "ref"])
    assert rc == 0, err
    path = tmp_path / f"hyp-{name.replace('/', '_')}-{pair}.txt"
    path.write_bytes(out)
    return str(path)


FILTER_CASES = [
    ["-t", "wmt16", "-l", "en-de", "--echo", "src", "-ol", "en"],
    ["-t", "wmt16", "-l", "en-de", "--echo", "src", "-ol", "non-en"],
    ["-t", "wmt16", "-l", "en-de", "--echo", "src", "-ol", "xx"],        # nothing matches
    ["-t", "wmt16", "-l", "en-de", "--echo", "src", "-ol", "non-non-en"],
    ["-t", "wmt16", "-l", "en-de", "--echo", "docid", "origlang", "-ol", "en"],
    ["-t", "wmt16", "-l", "en-de", "--echo", "all", "-ol", "non-en"],
    ["-t", "wmt22", "-l", "en-de", "--echo", "src", "--subset", "conversation"],
    ["-t", "wmt22", "-l", "en-de", "--echo", "src", "--subset", "nope"],
    ["-t", "wmt22", "-l", "en-de", "--echo", "src", "-ol", "en", "--subset", "social"],
    ["-t", "wmt16", "-l", "en-de", "--list", "-ol", "en"],
]


@pytest.mark.parametrize("args", FILTER_CASES, ids=lambda a: " ".join(a))
def test_segment_filters_on_echo(args):
    """`-ol`/`--subset` filter the segments a set reports, byte for byte.

    `-ol en` keeps `origlang == en` (1500 of wmt16 en-de's 2999) and `non-en` the rest; `--subset X`
    keeps `domain == X`; a value nothing matches prints nothing and exits 0. A filter does not need
    the reference data, only the annotation fields, so most of these run with no cache at all.
    """
    if not _have_set("wmt16"):
        pytest.skip("no cached test set under $SACREBLEU")
    assert_same_streams(args)


@pytest.mark.parametrize("flags", [
    ["-ol", "en"],
    ["-ol", "non-en"],
    ["--subset", "conversation"],
    ["--subset", "zz"],                      # nothing matches: the oracle's doubled value
    ["-ol", "en", "--subset", "social"],
    ["-ol", "zz", "--subset", "zz"],
])
def test_segment_filters_when_scoring(tmp_path, flags):
    """The filter masks the reference and the system output **by position**, and a filter that keeps
    nothing is refused with the oracle's own line (`subset=zzzz`, its value printed twice)."""
    if not _have_set("wmt16"):
        pytest.skip("no cached test set under $SACREBLEU")
    hyp = _hyp_from_set(tmp_path, "wmt22", "en-de") if "--subset" in flags else _hyp_from_set(
        tmp_path, "wmt16", "en-de")
    pair = ["-t", "wmt22", "-l", "en-de"] if "--subset" in flags else ["-t", "wmt16", "-l", "en-de"]
    assert_same_streams([*pair, "-i", hyp, "-m", "bleu", *flags])


def test_filter_without_a_test_set_and_no_references():
    """Byte-identical: with no references at all the oracle prints its own catalogue message, and so
    do we - the filter is checked after the reference check, which is the measured order."""
    refs = Path("nonexistent-reference.txt")
    assert_same_streams(["-i", str(refs), "-ol", "en"])


def test_subset_without_a_domain_annotation():
    """Documented divergence: sacreBLEU raises a traceback for `--subset` on a set with no `domain`
    field (wmt16 has `genre` instead and still refuses); swadesh prints the exception's own last
    line as a message. Exit codes match."""
    args = ["-t", "wmt16", "-l", "en-de", "--echo", "src", "--subset", "news"]
    ref_rc, ref_out, ref_err = run(SACREBLEU, args)
    sw_rc, sw_out, sw_err = run(SWADESH, args)
    assert ref_rc == sw_rc == 1 and sw_out == ref_out == b""
    assert b"Traceback" in ref_err
    assert sw_err.strip() == b"Exception: No subset annotation available for test set wmt16"


def test_filter_with_references_but_no_test_set(tmp_path):
    """Documented divergence: with reference files and no `-t`, filtering makes the oracle raise a
    traceback; swadesh refuses in one line. Exit codes match."""
    refs = tmp_path / "r.txt"
    refs.write_text("a\nb\n", encoding="utf-8")
    hyp = tmp_path / "h.txt"
    hyp.write_text("a\nb\n", encoding="utf-8")
    args = [str(refs), "-i", str(hyp), "-m", "bleu", "-ol", "en"]
    ref_rc, _, ref_err = run(SACREBLEU, args)
    sw_rc, _, sw_err = run(SWADESH, args)
    assert ref_rc == sw_rc == 1
    assert b"Traceback" in ref_err
    assert b"I need a test set (-t) to filter segments" in sw_err

# ------------------------------------------------------------- -d / --detail

DETAIL_CASES = [
    ["-t", "wmt16", "-l", "en-de", "-d"],
    ["-t", "wmt16", "-l", "en-de", "-m", "bleu", "chrf", "ter", "-d"],
    ["-t", "wmt16", "-l", "en-de", "--detail"],
    ["-t", "wmt22", "-l", "en-de", "-d"],
    ["-t", "wmt22", "-l", "en-de", "-m", "bleu", "chrf", "-d"],
    ["-t", "wmt22", "-l", "en-de", "-m", "ter", "-d"],
    ["-t", "wmt22", "-l", "de-fr", "-d"],
    ["-t", "wmt24", "-l", "en-de", "-d"],
    ["-t", "wmt21", "-l", "en-de", "-d"],
    ["-t", "wmt16", "-l", "en-de", "-d", "-w", "2"],
    ["-t", "wmt16", "-l", "en-de", "-d", "-f", "json"],   # detail forces the text form
    ["-t", "wmt16", "-l", "en-de", "-d", "-b"],
    ["-t", "wmt16", "-l", "en-de", "-d", "-sh"],
    ["-t", "wmt16", "-l", "en-de", "-d", "-sl"],           # sentence level has no detail block
]


@pytest.mark.parametrize("args", DETAIL_CASES, ids=lambda a: " ".join(a))
def test_detail_breakdown(tmp_path, args):
    """`-d/--detail`: one line per (origlang, domain) group per metric, byte for byte.

    The columns are padded by measurement: the domain name to its widest value, the sentence count
    to at least four characters (wmt24's counts are all three digits and are still padded to four),
    and the metric name to its widest (`BLEU` against `chrF2`). One `ALL` row per source language
    aggregates its domains, and `-d` forces the text form (`-d -f json` prints text).
    """
    if not _have_set("wmt16"):
        pytest.skip("no cached test set under $SACREBLEU")
    name, pair = "wmt16", "en-de"
    for i, arg in enumerate(args):
        if arg == "-t":
            name, pair = args[i + 1], args[i + 3]
            break
    hyp = _hyp_from_set(tmp_path, name, pair)
    assert_same_streams([*args, "-i", hyp])


def test_detail_without_a_test_set(tmp_path):
    """No test set means nothing to group by, and the oracle says so on stdout, exit 0."""
    refs = tmp_path / "r.txt"
    refs.write_text("a\nb\n", encoding="utf-8")
    assert_same_streams([str(refs), "-i", str(refs), "-d"])


def test_detail_with_a_segment_filter(tmp_path):
    """Documented divergence: `-d` with `-ol`/`--subset` makes the oracle raise
    (`ValueError: max() arg is an empty sequence`, after printing the score line); swadesh prints
    the groups of the filtered set."""
    if not _have_set("wmt16"):
        pytest.skip("no cached test set under $SACREBLEU")
    hyp = _hyp_from_set(tmp_path, "wmt16", "en-de")
    args = ["-t", "wmt16", "-l", "en-de", "-d", "-ol", "en", "-i", hyp]
    rc, out, err = run(SWADESH, args)
    assert rc == 0, err
    assert b"origlang=en domain=ALL" in out and out.count(b"origlang=") == 1
    ref_rc, _, ref_err = run(SACREBLEU, args)
    assert ref_rc != 0 and b"ValueError" in ref_err

# ------------------------------------------- rebuilding a set from its raw tree (the SGML family)

def _run_env(binary, args, env):
    return subprocess.run([str(binary), *args], capture_output=True, env=env)


def test_sgml_set_is_rebuilt_from_its_raw_tree(tmp_path):
    """With only `raw/` present, swadesh rebuilds the processed files from the SGML - and they are
    byte-identical to sacreBLEU's own, so `--echo`, scoring and `-d` all match.

    This is the parser half of `-t` (PLAN.md WS0.3): sacreBLEU regenerates the sidecars from the raw
    archive the same way, which is why deleting one brings it back. Comparing the *generated files*
    is the strongest check available - stronger than comparing output, because it exercises every
    annotation column for every pair.
    """
    home = Path(os.environ.get("SACREBLEU", Path.home() / ".sacrebleu"))
    raw = home / "wmt16" / "raw"
    if not raw.is_dir():
        pytest.skip("the wmt16 raw tree is not cached under $SACREBLEU")
    cache = tmp_path / "cache"
    (cache / "wmt16").mkdir(parents=True)
    shutil.copytree(raw, cache / "wmt16" / "raw", dirs_exist_ok=True)
    env = dict(ENV)
    env["SACREBLEU"] = str(cache)

    columns = (["src"], ["ref"], ["docid"], ["origlang"], ["genre"], ["src", "ref", "docid"], ["all"])
    for pair in ("en-de", "de-en", "en-fi", "cs-en", "tr-en"):
        for column in columns:
            args = ["-t", "wmt16", "-l", pair, "--echo", *column]
            theirs = run(SACREBLEU, args)
            ours = _run_env(SWADESH, [*args, "--python-version", PYVER], env)
            assert ours.returncode == 0, ours.stderr
            assert ours.stdout == theirs[1], (pair, column)
        for name in ("src", "ref", "docid", "genre", "origlang"):
            mine = (cache / "wmt16" / f"wmt16.{pair}.{name}").read_bytes()
            want = (home / "wmt16" / f"wmt16.{pair}.{name}").read_bytes()
            assert mine == want, (pair, name)

    # and the rebuilt cache scores identically, including the detail breakdown and a filter
    hyp = tmp_path / "hyp.txt"
    hyp.write_bytes(run(SACREBLEU, ["-t", "wmt16", "-l", "en-de", "--echo", "ref"])[1])
    for flags in (["-m", "bleu", "chrf", "ter"], ["-d"], ["-ol", "en", "-m", "bleu"]):
        args = ["-t", "wmt16", "-l", "en-de", "-i", str(hyp), *flags]
        theirs = run(SACREBLEU, args)
        ours = _run_env(SWADESH, [*args, "--python-version", PYVER], env)
        assert ours.returncode == theirs[0], flags
        assert ours.stdout == theirs[1], flags


def test_sgml_variants_are_rebuilt_from_their_raw_tree(tmp_path):
    """`wmt16/B`, `wmt16/tworefs` and `wmt16/dev` rebuild from their registry `paths` and match.

    These are exactly the sets the old pair-guessing rule could not reach: `B` names a *second*
    reference file, `tworefs` names two of them over the fields `ref:0`/`ref:1`, and `dev` lives
    under a `dev/` directory. The registry's per-pair `paths` list - one `.sgm` per `src`/`ref*`
    field, in order - is the source of truth for all of them, and the guard that a set whose `paths`
    are not all `.sgm` (or do not line up with its `src`/`ref*` fields) keeps refusing is what stops
    `wmt17/ms`, `wmt19/google/*` and `wmt21/B` being parsed wrongly.
    """
    home = Path(os.environ.get("SACREBLEU", Path.home() / ".sacrebleu"))
    checked = 0
    for set_name, pair, marker in (
        ("wmt16/B", "en-fi", "newstestB2016-enfi-ref.fi.sgm"),
        ("wmt16/tworefs", "en-fi", "newstestB2016-enfi-ref.fi.sgm"),
        ("wmt16/dev", "en-ro", "newsdev2016-enro-ref.ro.sgm"),
    ):
        raw = home / set_name / "raw"
        if not raw.is_dir() or not any(f.name == marker for f in raw.rglob("*")):
            continue
        checked += 1
        cache = tmp_path / set_name.replace("/", "_")
        (cache / set_name).mkdir(parents=True)
        shutil.copytree(raw, cache / set_name / "raw", dirs_exist_ok=True)
        env = dict(ENV)
        env["SACREBLEU"] = str(cache)
        args = ["-t", set_name, "-l", pair, "--echo", "all"]
        theirs = run(SACREBLEU, args)
        ours = _run_env(SWADESH, [*args, "--python-version", PYVER], env)
        assert ours.returncode == 0, (set_name, ours.stderr)
        assert ours.stdout == theirs[1], set_name
        prefix = set_name.replace("/", "_")
        for f in (home / set_name).glob(f"{prefix}.{pair}.*"):
            mine = (cache / set_name / f.name).read_bytes()
            assert mine == f.read_bytes(), (set_name, f.name)
    assert checked, "no SGML variant is cached under $SACREBLEU"


def test_an_unparsable_layout_still_refuses(tmp_path):
    """A set whose layout no measured rule covers refuses rather than producing wrong data.

    It is a *user* set on purpose: a built-in name would send the fetcher to the network, while a
    user root has no catalogue entry to fetch, so this exercises the parsers' own refusal. The
    layout here is the SGML family's directory shape with files that are not SGML.
    """
    root = tmp_path / "sets"
    (root / "myset").mkdir(parents=True)
    (root / "myset" / "myset.en-de.src").write_text("not sgml\n", encoding="utf-8")
    env = dict(ENV)
    env["SWADESH_TEST_SETS"] = str(root)
    # The file exists, so `--echo src` succeeds and shows the parser was not invoked at all...
    ok = _run_env(SWADESH, ["-t", "myset", "-l", "en-de", "--echo", "src", "--python-version", PYVER], env)
    assert ok.returncode == 0 and ok.stdout == b"not sgml\n"
    # ...and a field the set does not have is refused in sacreBLEU's own words, because a user set's
    # fields are whatever files are there - nothing is invented for it.
    missing = _run_env(SWADESH, ["-t", "myset", "-l", "en-de", "--echo", "ref", "--python-version", PYVER], env)
    assert missing.returncode == 1 and missing.stdout == b""
    assert b"No such field ref in test set myset" in missing.stderr

# ------------------------------------------------- the IWSLT family (iwslt17)

def test_iwslt_set_is_rebuilt_from_its_raw_tree(tmp_path):
    """`iwslt17` rebuilds from its TED XML alone, and every generated file is byte-identical to
    sacreBLEU's own.

    The measured rules this pins down: the segment text is whitespace-*collapsed*
    (`" ".join(text.split())`) and **not** entity-decoded (`&amp;` stays literal, so a plain strip
    would also be wrong - 819 of en-fr's 1455 source lines carry doubled internal spaces); `docid`
    and `genre` are the enclosing `<doc>`'s; `origlang` is empty for the whole family; and the
    reference lives in the *other* direction's directory, which the registry's per-pair paths name.
    """
    home = Path(os.environ.get("SACREBLEU", Path.home() / ".sacrebleu"))
    raw = home / "iwslt17" / "raw"
    if not raw.is_dir():
        pytest.skip("the iwslt17 raw tree is not cached under $SACREBLEU")
    cache = tmp_path / "cache"
    (cache / "iwslt17").mkdir(parents=True)
    shutil.copytree(raw, cache / "iwslt17" / "raw", dirs_exist_ok=True)
    env = dict(ENV)
    env["SACREBLEU"] = str(cache)
    checked = 0
    for pair in ("en-fr", "en-de", "ja-en", "ar-en"):
        src_lang, trg_lang = pair.split("-")
        need = [raw / pair / f"IWSLT17.TED.tst2017.{pair}.{src_lang}.xml",
                raw / f"{trg_lang}-{src_lang}" / f"IWSLT17.TED.tst2017.{trg_lang}-{src_lang}.{trg_lang}.xml"]
        if not all(p.exists() for p in need):
            continue  # a direction whose raw tree is not cached here
        checked += 1
        args = ["-t", "iwslt17", "-l", pair, "--echo", "all"]
        theirs = run(SACREBLEU, args)
        ours = _run_env(SWADESH, [*args, "--python-version", PYVER], env)
        assert ours.returncode == 0, ours.stderr
        assert ours.stdout == theirs[1], pair
        for name in ("src", "ref", "docid", "genre", "origlang"):
            mine = (cache / "iwslt17" / f"iwslt17.{pair}.{name}").read_bytes()
            want = (home / "iwslt17" / f"iwslt17.{pair}.{name}").read_bytes()
            assert mine == want, (pair, name)
    assert checked, "no complete iwslt17 direction is cached under $SACREBLEU"


# ------------------------------------------------- the TSV family (mtnt1.1, mtnt2019)

def test_tsv_set_is_rebuilt_from_its_raw_tree(tmp_path):
    """`mtnt1.1/*` rebuilds from its TSV alone, and every generated file is byte-identical to
    sacreBLEU's own.

    The measured rules this pins: the registry path is `<col>:<file>` with a **0-based** column index
    (`1:`/`2:` over mtnt1.1's `id<TAB>src<TAB>tgt` file, `2:`/`3:` over mtnt2019's four-column one);
    the last column keeps its `\r`, so a CRLF input yields an LF `.src` and a CRLF `.ref`; and the
    read side is measured here too - a cached field is **rstripped** line by line when loaded (the
    files keep mtnt's trailing spaces, `--echo src` does not), which the `--echo all` comparison
    below exercises on lines that genuinely have them.
    """
    home = Path(os.environ.get("SACREBLEU", Path.home() / ".sacrebleu"))
    raw = home / "mtnt1.1" / "test" / "raw"
    if not (raw / "MTNT" / "test" / "test.en-fr.tsv").exists():
        pytest.skip("the mtnt1.1 raw tree is not cached under $SACREBLEU")
    cache = tmp_path / "cache"
    (cache / "mtnt1.1" / "test").mkdir(parents=True)
    shutil.copytree(raw, cache / "mtnt1.1" / "test" / "raw", dirs_exist_ok=True)
    env = dict(ENV)
    env["SACREBLEU"] = str(cache)
    for pair in ("en-fr", "fr-en", "en-ja"):
        args = ["-t", "mtnt1.1/test", "-l", pair, "--echo", "all"]
        theirs = run(SACREBLEU, args)
        ours = _run_env(SWADESH, [*args, "--python-version", PYVER], env)
        assert ours.returncode == 0, ours.stderr
        assert ours.stdout == theirs[1], pair
        for name in ("src", "ref"):
            mine = (cache / "mtnt1.1" / "test" / f"mtnt1.1_test.{pair}.{name}").read_bytes()
            want = (home / "mtnt1.1" / "test" / f"mtnt1.1_test.{pair}.{name}").read_bytes()
            assert mine == want, (pair, name)


# ------------------------------------------------- the WMT XML family (wmt20 and later)

def test_wmtxml_set_is_rebuilt_from_its_raw_tree(tmp_path):
    """wmt22/wmt24 rebuild from `raw/xml` alone, and every generated file is byte-identical to
    sacreBLEU's own.

    The measured traps this covers: the diagnostic `testsuite` documents are excluded (wmt22 en-de
    holds 17,618 of them and the oracle's files contain no `tsuite` id at all); `&amp;` is replaced
    *last*, so `A&amp;E;` becomes `A&E;`, `AT&amp;amp;T` becomes `AT&amp;T` and `&amp;#39;` becomes
    `&#39;`; whitespace collapses (`von  ihrem` becomes `von ihrem`, and a tab becomes a space,
    `wmt24.en-de.src` holding no tab); and a self-closing `<seg/>` is an *empty* segment (89 of them
    in wmt24 en-de) rather than a swallowed next segment.
    """
    home = Path(os.environ.get("SACREBLEU", Path.home() / ".sacrebleu"))
    root = tmp_path / "cache"
    env = dict(ENV)
    env["SACREBLEU"] = str(root)
    checked = 0
    for set_name, pair in (("wmt22", "en-de"), ("wmt22", "de-fr"), ("wmt24", "en-de")):
        raw = home / set_name / "raw"
        if not raw.is_dir():
            continue
        (root / set_name).mkdir(parents=True, exist_ok=True)
        link = root / set_name / "raw"
        if not link.exists():
            link.symlink_to(raw)
        for theirs in sorted((home / set_name).glob(f"{set_name}.{pair}.*")):
            field = theirs.name.split(f"{set_name}.{pair}.", 1)[1]
            wire = f"ref:{field[len('ref-'):]}" if field.startswith("ref-") else field
            args = ["-t", set_name, "-l", pair, "--echo", wire, "--python-version", PYVER]
            ours = _run_env(SWADESH, args, env)
            assert ours.returncode == 0, (set_name, pair, wire, ours.stderr)
            assert ours.stdout == theirs.read_bytes(), (set_name, pair, wire)
            generated = root / set_name / f"{set_name}.{pair}.{field}"
            assert generated.read_bytes() == theirs.read_bytes(), (set_name, pair, field)
            checked += 1
    if checked == 0:
        pytest.skip("no cached wmt22/wmt24 raw tree under $SACREBLEU")


def test_wmtxml_scores_and_breaks_down_identically(tmp_path):
    """Scoring, `-d` and a filter on the rebuilt wmt24 cache match the oracle's bytes."""
    home = Path(os.environ.get("SACREBLEU", Path.home() / ".sacrebleu"))
    raw = home / "wmt24" / "raw"
    if not raw.is_dir():
        pytest.skip("the wmt24 raw tree is not cached under $SACREBLEU")
    root = tmp_path / "cache"
    (root / "wmt24").mkdir(parents=True)
    (root / "wmt24" / "raw").symlink_to(raw)
    env = dict(ENV)
    env["SACREBLEU"] = str(root)
    hyp = tmp_path / "hyp.txt"
    hyp.write_bytes(run(SACREBLEU, ["-t", "wmt24", "-l", "en-de", "--echo", "ref"])[1])
    for flags in (["-m", "bleu", "chrf"], ["-d"], ["-ol", "en", "-m", "bleu"]):
        args = ["-t", "wmt24", "-l", "en-de", "-i", str(hyp), *flags]
        theirs = run(SACREBLEU, args)
        ours = _run_env(SWADESH, [*args, "--python-version", PYVER], env)
        assert ours.returncode == theirs[0], flags
        assert ours.stdout == theirs[1], flags

# ------------------------------------------------------ the plain-text family (multi30k)

def test_plain_text_set_is_rebuilt_from_its_raw_tree(tmp_path):
    """multi30k is the plain-text family: the processed `src`/`ref` are the two files the registry
    names for the pair, with **each line rstripped**.

    Measured as `[line.rstrip() for line in raw] == processed` for both columns of both pairs - so
    the whole rule, not a one-off: the French reference has two lines ending in a space in `raw/` and
    none in the processed file. The registry's per-pair paths are carried in `datasets.rs` for this.
    """
    home = Path(os.environ.get("SACREBLEU", Path.home() / ".sacrebleu"))
    set_name = "multi30k/2017"
    raw = home / set_name / "raw"
    if not raw.is_dir():
        pytest.skip("the multi30k/2017 raw tree is not cached under $SACREBLEU")
    root = tmp_path / "cache"
    (root / set_name).mkdir(parents=True)
    (root / set_name / "raw").symlink_to(raw)
    env = dict(ENV)
    env["SACREBLEU"] = str(root)
    for pair in ("en-de", "en-fr"):
        for field in ("src", "ref"):
            args = ["-t", set_name, "-l", pair, "--echo", field, "--python-version", PYVER]
            ours = _run_env(SWADESH, args, env)
            theirs = run(SACREBLEU, ["-t", set_name, "-l", pair, "--echo", field])
            assert ours.returncode == 0, ours.stderr
            assert ours.stdout == theirs[1], (pair, field)
            generated = root / set_name / f"multi30k_2017.{pair}.{field}"
            want = home / set_name / f"multi30k_2017.{pair}.{field}"
            assert generated.read_bytes() == want.read_bytes(), (pair, field)

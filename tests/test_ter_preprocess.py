"""The native TER preprocessor must produce exactly what sacreBLEU's tokenizer produces.

`_swadesh.ter_preprocess` is what lets the TER metric stop importing
`sacrebleu.tokenizers.tokenizer_ter`. It has to match `TercomTokenizer(cfg)(sent.rstrip())` for
every flag combination, including the lowercasing rules CPython has and Rust does not (the final
sigma).
"""

import itertools
import os

import pytest
from sacrebleu.tokenizers.tokenizer_ter import TercomTokenizer

import swadesh
from swadesh import _swadesh

FLAGS = list(itertools.product([False, True], repeat=4))
FLAG_IDS = [f"n{int(n)}-p{int(p)}-a{int(a)}-c{int(c)}" for n, p, a, c in FLAGS]

INPUTS = [
    "", " ", "   ", "a", "hello world", "Hello, World!", "the dog's bone",
    "a.b  a;b", "12-13", "3.14", "2,2", "a?b", "a&b", "a<b>c", "a/b", "don't", "'s",
    "&amp;", "&quot;", "&lt;", "&gt;", "&amp;lt;", "&amp;quot;", "&amp;amp;", "<skipped>",
    "&nbsp;", "&#39;", "&apos;", "&AMP;", "&AMP;LT;", "x&amp;y",
    "你好世界", "这是测试", "カメラです", "한국어", "日本語のテスト",
    "a\u00a0b", "a\u2000b", "a\nb", "a\tb", "a-\nb", "trailing   ",
    "Ünïcödé", "Straße", "ﬁle", "e\u0301", "\u200b", "\u3000",
    "3.14 12-13 2,2", "a-b c-d 1-2", "a;b;c", "a<b>c", "x" * 120,
]

# The published data, when the host has it: real text is the strongest check.
_DATA = os.path.join(os.path.dirname(__file__), "..", "bench", "data")
for _name in ("wmt22.en-de.ref", "wmt22.en-zh.ref", "wmt22.en-ja.ref"):
    _p = os.path.join(_DATA, _name)
    if os.path.exists(_p):
        with open(_p, encoding="utf-8") as fh:
            INPUTS += [ln.rstrip("\n") for ln in list(fh)[:40]]


@pytest.mark.parametrize("flags", FLAGS, ids=FLAG_IDS)
def test_ter_preprocess_matches_tercom_tokenizer(flags):
    norm, npunct, asian, case = flags
    tok = TercomTokenizer(normalized=norm, no_punct=npunct, asian_support=asian, case_sensitive=case)
    for text in INPUTS:
        assert _swadesh.ter_preprocess(text, norm, npunct, asian, case) == tok(text.rstrip()), repr(text)


@pytest.mark.parametrize("flags", FLAGS, ids=FLAG_IDS)
def test_ter_metric_preprocess_matches_tercom_tokenizer(flags):
    """The metric's own `_preprocess_segment` goes through the native kernel too."""
    norm, npunct, asian, case = flags
    tok = TercomTokenizer(normalized=norm, no_punct=npunct, asian_support=asian, case_sensitive=case)
    metric = swadesh.TER(normalized=norm, no_punct=npunct, asian_support=asian, case_sensitive=case)
    for text in INPUTS:
        assert metric._preprocess_segment(text) == tok(text.rstrip()), repr(text)


def test_final_sigma_is_cpython_lowering():
    """The reason this cannot be Rust's `to_lowercase`: word-final sigma."""
    assert _swadesh.ter_preprocess("ΣΊΣΥΦΟΣ", False, False, False, False) == "σίσυφος"
    assert _swadesh.ter_preprocess("ΣΊΣΥΦΟΣ", False, False, False, True) == "ΣΊΣΥΦΟΣ"

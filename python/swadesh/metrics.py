"""Drop-in, bit-exact accelerated replacements for sacreBLEU's metric classes.

The classes below *subclass* sacreBLEU's own ``BLEU`` and ``CHRF``. Only the
hot path -- tokenization, n-gram extraction and matching, i.e. the segment
level sufficient statistics -- is replaced by Rust. Everything else (score
objects, signatures, formatting, final score arithmetic) is sacreBLEU's own
code, so corpus-level results are identical by construction.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence

import numpy as np
from sacrebleu.metrics.bleu import BLEU as _SacreBLEU
from sacrebleu.metrics.chrf import CHRF as _SacreCHRF

from . import _swadesh

sacrelogger = logging.getLogger("sacrebleu")

#: Tokenizers implemented natively in Rust. Others (ja-mecab, ko-mecab, spm,
#: flores101/200, spBLEU-1K) are tokenized by sacreBLEU in Python and then
#: matched natively.
NATIVE_TOKENIZERS = frozenset({"none", "13a", "intl", "zh", "char"})


class _RefCache:
    """Raw reference streams. Truthy so that sacreBLEU's ``elif self._ref_cache``
    checks behave as with its own cache."""

    __slots__ = ("streams",)

    def __init__(self, streams):
        self.streams = streams

    def __bool__(self):
        return True

    def __len__(self):
        return min((len(s) for s in self.streams), default=0)


def _normalize_refs(references) -> List[Sequence[Optional[str]]]:
    return [list(s) if not isinstance(s, list) else s for s in references]


def _truncate(hypotheses, streams):
    """Mirror sacreBLEU's ``zip(hypotheses, zip(*references))`` truncation."""
    n = min([len(hypotheses)] + [len(s) for s in streams])
    if n != len(hypotheses):
        hypotheses = hypotheses[:n]
    if any(len(s) != n for s in streams):
        streams = [s[:n] for s in streams]
    return hypotheses, streams


def _tok_warning(count: int):
    if count >= 100:
        sacrelogger.warning("That's 100 lines that end in a tokenized period ('.')")
        sacrelogger.warning("It looks like you forgot to detokenize your test data, which may hurt your score.")
        sacrelogger.warning(
            "If you insist your data is detokenized, or don't care, you can suppress this message with the `force` parameter."
        )


def _matrix(buf: bytes, n: int, width: int) -> np.ndarray:
    return np.frombuffer(buf, dtype=np.int64).reshape(n, width).copy()


class _FastMixin:
    """Shared plumbing between the accelerated metrics."""

    def _cache_references(self, references):
        streams = _normalize_refs(references)
        # num_refs semantics identical to sacreBLEU's _cache_references
        nums = set()
        for refs in zip(*streams):
            nums.add(sum(1 for x in refs if x is not None))
        self.num_refs = nums.pop() if len(nums) == 1 else -1
        return _RefCache(streams)

    def _resolve_refs(self, references):
        if references:
            return self._cache_references(references).streams
        if self._ref_cache:
            return self._ref_cache.streams
        raise RuntimeError("No references provided and the cache is empty.")

    def _aggregate_and_compute(self, stats):
        if isinstance(stats, np.ndarray):
            if stats.ndim == 1:
                return self._compute_score_from_stats(stats.tolist())
            return self._compute_score_from_stats(stats.sum(0).tolist())
        return super()._aggregate_and_compute(stats)


class BLEU(_FastMixin, _SacreBLEU):
    """Bit-exact, accelerated ``sacrebleu.metrics.BLEU``.

    Same constructor and API. Additional method: :meth:`sentence_scores`.
    """

    def __init__(self, *args, references: Optional[Sequence[Sequence[str]]] = None, **kwargs):
        super().__init__(*args, references=None, **kwargs)
        if references is not None:
            self._ref_cache = self._cache_references(references)

    @property
    def _native_tokenizer(self) -> Optional[str]:
        sig = self.tokenizer_signature
        return sig if sig in NATIVE_TOKENIZERS else None

    def _prepare(self, hypotheses, streams):
        """Returns (hyps, streams, tokenizer_name, lowercase, tok_count)."""
        tok = self._native_tokenizer
        if tok is not None:
            return hypotheses, streams, tok, bool(self.lowercase), None
        # Python tokenizer fallback (ja-mecab, ko-mecab, spm, ...)
        tok_count = sum(1 for h in hypotheses if h.endswith(" ."))
        pp = self._preprocess_segment
        hyps = [pp(h) for h in hypotheses]
        streams = [[None if x is None else pp(x) for x in s] for s in streams]
        return hyps, streams, "none", False, tok_count

    def _extract_corpus_statistics(self, hypotheses: Sequence[str], references) -> Any:
        streams = self._resolve_refs(references)
        hypotheses, streams = _truncate(hypotheses, streams)
        hyps, streams, tok, lc, tok_count = self._prepare(hypotheses, streams)
        buf, n, width, _nmin, _nmax, native_tok_count = _swadesh.bleu_stats(
            hyps, streams, tok, lc, self.max_ngram_order
        )
        if not self._force:
            _tok_warning(native_tok_count if tok_count is None else tok_count)
        return _matrix(buf, n, width)

    def sentence_scores(self, hypotheses: Sequence[str], references: Sequence[Sequence[str]]) -> List[float]:
        """Sentence-level BLEU for every segment, computed in parallel.

        ``references`` uses the *corpus* layout (a list of reference streams).
        Returns exactly ``[self.sentence_score(h, [r[i] for r in references]).score ...]``.
        """
        streams = _normalize_refs(references)
        hypotheses, streams = _truncate(hypotheses, streams)
        hyps, streams, tok, lc, _ = self._prepare(hypotheses, streams)
        return _swadesh.bleu_sentence_scores(
            hyps, streams, tok, lc, self.max_ngram_order,
            self.smooth_method, None if self.smooth_value is None else float(self.smooth_value),
            bool(self.effective_order),
        )


class CHRF(_FastMixin, _SacreCHRF):
    """Bit-exact, accelerated ``sacrebleu.metrics.CHRF`` (chrF and chrF++).

    Same constructor and API. Additional method: :meth:`sentence_scores`.
    """

    def __init__(self, *args, references: Optional[Sequence[Sequence[str]]] = None, **kwargs):
        super().__init__(*args, references=None, **kwargs)
        if references is not None:
            self._ref_cache = self._cache_references(references)

    def _args(self):
        return (self.char_order, self.word_order, self.beta, bool(self.lowercase),
                bool(self.whitespace), bool(self.eps_smoothing))

    def _extract_corpus_statistics(self, hypotheses: Sequence[str], references) -> Any:
        streams = self._resolve_refs(references)
        hypotheses, streams = _truncate(hypotheses, streams)
        if not self._force:
            _tok_warning(sum(1 for h in hypotheses if h.endswith(" .")))
        buf, n, width, _nmin, _nmax = _swadesh.chrf_stats(hypotheses, streams, *self._args())
        return _matrix(buf, n, width)

    def sentence_scores(self, hypotheses: Sequence[str], references: Sequence[Sequence[str]]) -> List[float]:
        """Sentence-level chrF for every segment, computed in parallel."""
        streams = _normalize_refs(references)
        hypotheses, streams = _truncate(hypotheses, streams)
        return _swadesh.chrf_sentence_scores(hypotheses, streams, *self._args())

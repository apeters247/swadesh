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

    def corpus_score(self, hypotheses, references, n_bootstrap: int = 1):
        """Same as sacreBLEU's ``corpus_score``; bootstrap resampling is native."""
        self._check_corpus_score_args(hypotheses, references)
        stats = self._extract_corpus_statistics(hypotheses, references)
        actual_score = self._aggregate_and_compute(stats)
        if n_bootstrap > 1:
            from .significance import _bootstrap_resample
            self.n_bootstrap = n_bootstrap
            self.seed, bs_scores = _bootstrap_resample(stats, self, n_bootstrap)
            actual_score.estimate_ci(bs_scores)
        return actual_score

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

    def sentence_score(self, hypothesis: str, references: Sequence[str]):
        """Same as sacreBLEU's ``sentence_score`` (native statistics, no numpy round-trip)."""
        if not self.effective_order:
            sacrelogger.warning('It is recommended to enable `effective_order` for sentence-level BLEU.')
        self._check_sentence_score_args(hypothesis, references)
        tok = self._native_tokenizer
        if tok is None:
            return super().sentence_score(hypothesis, references)
        refs = [r for r in references if r is not None]
        self.num_refs = len(refs)
        stats = _swadesh.bleu_stats_one(hypothesis, refs, tok, bool(self.lowercase), self.max_ngram_order)
        return self._compute_score_from_stats(stats)

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

    def pairwise(self, candidates: Sequence[str], references: Optional[Sequence[str]] = None) -> np.ndarray:
        """Sentence-BLEU matrix ``M[i, j] = sentence_score(candidates[i], [references[j]]).score``.

        With ``references=None`` the candidates are scored against each other
        (MBR / consensus decoding). Computed natively and in parallel; n-grams
        are extracted once per sentence instead of once per pair.
        """
        refs = candidates if references is None else references
        cands, refs_l = list(candidates), list(refs)
        tok = self._native_tokenizer or "none"
        lc = bool(self.lowercase)
        if self._native_tokenizer is None:
            cands = [self._preprocess_segment(x) for x in cands]
            refs_l = [self._preprocess_segment(x) for x in refs_l]
            lc = False
        buf = _swadesh.bleu_matrix(
            cands, refs_l, tok, lc, self.max_ngram_order, self.smooth_method,
            None if self.smooth_value is None else float(self.smooth_value), bool(self.effective_order))
        return _as_matrix(buf, len(cands), len(refs_l))


def _as_matrix(buf, n, m):
    return np.frombuffer(buf, dtype=np.float64).reshape(n, m).copy()


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
        buf, n, width, _nmin, _nmax, tok_count = _swadesh.chrf_stats(hypotheses, streams, *self._args())
        if not self._force:
            _tok_warning(tok_count)
        return _matrix(buf, n, width)

    def sentence_score(self, hypothesis: str, references: Sequence[str]):
        """Same as sacreBLEU's ``sentence_score`` (native statistics, no numpy round-trip)."""
        self._check_sentence_score_args(hypothesis, references)
        refs = [r for r in references if r is not None]
        self.num_refs = len(refs)
        return self._compute_score_from_stats(_swadesh.chrf_stats_one(hypothesis, refs, *self._args()))

    def sentence_scores(self, hypotheses: Sequence[str], references: Sequence[Sequence[str]]) -> List[float]:
        """Sentence-level chrF for every segment, computed in parallel."""
        streams = _normalize_refs(references)
        hypotheses, streams = _truncate(hypotheses, streams)
        return _swadesh.chrf_sentence_scores(hypotheses, streams, *self._args())

    def pairwise(self, candidates: Sequence[str], references: Optional[Sequence[str]] = None) -> np.ndarray:
        """chrF matrix ``M[i, j] = sentence_score(candidates[i], [references[j]]).score``.

        With ``references=None`` the candidates are scored against each other
        (MBR / consensus decoding). Computed natively and in parallel.
        """
        refs = candidates if references is None else references
        cands, refs_l = list(candidates), list(refs)
        buf = _swadesh.chrf_matrix(cands, refs_l, *self._args())
        return _as_matrix(buf, len(cands), len(refs_l))


def mbr(candidates: Sequence[str], metric=None, exclude_self: bool = True):
    """Minimum Bayes risk selection: return ``(best_index, expected_utility)``
    where each candidate's utility is its mean score against all other
    candidates used as pseudo-references. ``metric`` defaults to ``CHRF()``.
    """
    metric = CHRF() if metric is None else metric
    m = metric.pairwise(candidates)
    n = m.shape[0]
    if exclude_self and n > 1:
        util = (m.sum(1) - np.diag(m)) / (n - 1)
    else:
        util = m.mean(1)
    best = int(np.argmax(util))
    return best, util

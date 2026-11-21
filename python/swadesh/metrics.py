"""Drop-in, bit-exact accelerated replacements for sacreBLEU's metric classes.

The classes below *subclass* sacreBLEU's own ``BLEU`` and ``CHRF``. Only the
hot path -- tokenization, n-gram extraction and matching, i.e. the segment
level sufficient statistics -- is replaced by Rust. Everything else (score
objects, signatures, formatting, final score arithmetic) is sacreBLEU's own
code, so corpus-level results are identical by construction.

Inputs the native kernels do not cover (invalid parameter values, non-string
items, segments with no reference) are not guessed at: they go through
sacreBLEU's own Python statistics loop (``_FastMixin._py_statistics``), so
the result or the exception is sacreBLEU's, not an artefact of the Rust layer.
"""

import logging
from typing import Any, List, Optional, Sequence



class _LazyNumpy:
    """numpy is only needed once statistics exist; importing it lazily keeps
    `import swadesh` as cheap as `import sacrebleu` (which doesn't load numpy)."""

    def __getattr__(self, name):
        import numpy

        globals()["np"] = numpy
        return getattr(numpy, name)


np = _LazyNumpy()
from sacrebleu.metrics.base import Metric as _SacreMetric  # noqa: E402
from sacrebleu.metrics.bleu import BLEU as _SacreBLEU
from sacrebleu.metrics.bleu import MAX_NGRAM_ORDER
from sacrebleu.metrics.chrf import CHRF as _SacreCHRF
from sacrebleu.metrics.ter import TER as _SacreTER

from . import _swadesh

sacrelogger = logging.getLogger("sacrebleu")

#: Tokenizers implemented natively in Rust. Others (ja-mecab, ko-mecab, spm,
#: flores101/200, spBLEU-1K) are tokenized by sacreBLEU in Python and then
#: matched natively.
NATIVE_TOKENIZERS = frozenset({"none", "13a", "intl", "zh", "char"})

#: Errors the native layer raises for inputs it cannot take (non-str items,
#: negative counts, wrong container types). The Python reference path then
#: runs and raises or returns exactly what sacreBLEU does.
_FALLBACK_ERRORS = (TypeError, AttributeError, OverflowError)


class _RefCache:
    """Raw reference streams, plus whether some segment has no reference at all.

    Its truth value matches sacreBLEU's per-segment cache list, which is empty
    (falsy) exactly when there are no segments.
    """

    def __init__(self, streams, has_empty: bool = False):
        self.streams = streams
        self.has_empty = has_empty

    def __bool__(self):
        return len(self) > 0

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


def _set_num_refs_like_loop(metric, streams, n):
    """``sentence_scores`` is defined as a loop of ``sentence_score`` calls, and each of
    those sets ``num_refs`` to the segment's number of references; leave the same state."""
    if n:
        metric.num_refs = sum(1 for s in streams if s[n - 1] is not None)


class _FastMixin:
    """Shared plumbing between the accelerated metrics."""

    def _native_ok(self) -> bool:
        """False when the parameters are outside what the native kernels implement."""
        return True

    def _cache_references(self, references):
        streams = _normalize_refs(references)
        # num_refs semantics identical to sacreBLEU's per-segment reference count. Only reference
        # lines are removed per segment, and a hypothesis None (which only reaches here through
        # _py_statistics, and still raises in the parse) counts as a reference too. So a single
        # reference column decides it by C-level counting; more columns only need comparing.
        has_empty = None in streams[0] if streams else False
        if len(streams) == 1:
            count = len(streams[0]) - streams[0].count(None)
            self.num_refs = count
        else:
            counts = {sum(x is not None for x in refs) for refs in zip(*streams)}
            self.num_refs = counts.pop() if len(counts) == 1 else -1
        return _RefCache(streams, has_empty)

    def _resolve_refs(self, references) -> _RefCache:
        """The reference cache for this call, with sacreBLEU's precedence and errors."""
        if references:
            return self._cache_references(references)
        if self._ref_cache:
            return self._ref_cache
        raise RuntimeError("No references provided and the cache is empty.")

    def _py_statistics(self, hypotheses, references) -> list:
        """sacreBLEU's own statistics loop (its own reference cache, its own warnings)."""
        if references:
            ref_cache = _SacreMetric._cache_references(self, references)
        elif self._ref_cache:
            ref_cache = _SacreMetric._cache_references(self, self._ref_cache.streams)
        else:
            raise RuntimeError("No references provided and the cache is empty.")
        stats = []
        tok_count = 0
        for hyp, ref_kwargs in zip(hypotheses, ref_cache):
            if not self._force and hyp.endswith(" ."):
                tok_count += 1
            hyp = self._preprocess_segment(hyp)
            stats.append(self._compute_segment_statistics(hyp, ref_kwargs))
        if not self._force:
            _tok_warning(tok_count)
        return stats

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


def _is_int(x) -> bool:
    return isinstance(x, int)


class BLEU(_FastMixin, _SacreBLEU):
    """Bit-exact, accelerated ``sacrebleu.metrics.BLEU``.

    Same constructor and API. Additional method: :meth:`sentence_scores`.
    """

    def __init__(self, lowercase: bool = False, force: bool = False, tokenize: Optional[str] = None,
                 smooth_method: str = 'exp', smooth_value: Optional[float] = None,
                 max_ngram_order: int = MAX_NGRAM_ORDER, effective_order: bool = False,
                 trg_lang: str = '', references: Optional[Sequence[Sequence[str]]] = None):
        super().__init__(lowercase=lowercase, force=force, tokenize=tokenize,
                         smooth_method=smooth_method, smooth_value=smooth_value,
                         max_ngram_order=max_ngram_order, effective_order=effective_order,
                         trg_lang=trg_lang, references=None)
        if references is not None:
            self._ref_cache = self._cache_references(references)

    def _native_ok(self) -> bool:
        return _is_int(self.max_ngram_order) and self.max_ngram_order >= 1

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
        cache = self._resolve_refs(references)
        if cache.has_empty or not self._native_ok():
            return self._py_statistics(hypotheses, references)
        try:
            hyps, streams = _truncate(hypotheses, cache.streams)
            hyps, streams, tok, lc, tok_count = self._prepare(hyps, streams)
            buf, n, width, _nmin, _nmax, native_tok_count = _swadesh.bleu_stats(
                hyps, streams, tok, lc, self.max_ngram_order
            )
        except _FALLBACK_ERRORS:
            return self._py_statistics(hypotheses, references)
        if not self._force:
            _tok_warning(native_tok_count if tok_count is None else tok_count)
        return _matrix(buf, n, width)

    def sentence_score(self, hypothesis: str, references: Sequence[str]):
        """Same as sacreBLEU's ``sentence_score`` (native statistics, no numpy round-trip)."""
        if not self.effective_order:
            sacrelogger.warning('It is recommended to enable `effective_order` for sentence-level BLEU.')
        self._check_sentence_score_args(hypothesis, references)
        tok = self._native_tokenizer
        refs = [r for r in references if r is not None]
        if tok is None or not refs or not self._native_ok():
            return _SacreMetric.sentence_score(self, hypothesis, references)
        self.num_refs = len(refs)
        try:
            stats = _swadesh.bleu_stats_one(hypothesis, refs, tok, bool(self.lowercase), self.max_ngram_order)
        except _FALLBACK_ERRORS:
            return _SacreMetric.sentence_score(self, hypothesis, references)
        return self._compute_score_from_stats(stats)

    def sentence_scores(self, hypotheses: Sequence[str], references: Sequence[Sequence[str]]) -> List[float]:
        """Sentence-level BLEU for every segment, computed in parallel.

        ``references`` uses the *corpus* layout (a list of reference streams).
        Returns exactly ``[self.sentence_score(h, [r[i] for r in references]).score ...]``.
        """
        streams = _normalize_refs(references)
        hypotheses, streams = _truncate(hypotheses, streams)
        hyps, streams, tok, lc, _ = self._prepare(hypotheses, streams)
        out = _swadesh.bleu_sentence_scores(
            hyps, streams, tok, lc, self.max_ngram_order,
            self.smooth_method, None if self.smooth_value is None else float(self.smooth_value),
            bool(self.effective_order),
        )
        _set_num_refs_like_loop(self, streams, len(hyps))
        return out

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

    def __init__(self, char_order: int = _SacreCHRF.CHAR_ORDER, word_order: int = _SacreCHRF.WORD_ORDER,
                 beta: int = _SacreCHRF.BETA, lowercase: bool = False, whitespace: bool = False,
                 eps_smoothing: bool = False, references: Optional[Sequence[Sequence[str]]] = None):
        super().__init__(char_order=char_order, word_order=word_order, beta=beta, lowercase=lowercase,
                         whitespace=whitespace, eps_smoothing=eps_smoothing, references=None)
        if references is not None:
            self._ref_cache = self._cache_references(references)

    def _native_ok(self) -> bool:
        return (_is_int(self.char_order) and self.char_order >= 1
                and _is_int(self.word_order) and self.word_order >= 0 and _is_int(self.beta))

    def _args(self):
        return (self.char_order, self.word_order, self.beta, bool(self.lowercase),
                bool(self.whitespace), bool(self.eps_smoothing))

    def _extract_corpus_statistics(self, hypotheses: Sequence[str], references) -> Any:
        cache = self._resolve_refs(references)
        if cache.has_empty or not self._native_ok():
            return self._py_statistics(hypotheses, references)
        try:
            hyps, streams = _truncate(hypotheses, cache.streams)
            buf, n, width, _nmin, _nmax, tok_count = _swadesh.chrf_stats(hyps, streams, *self._args())
        except _FALLBACK_ERRORS:
            return self._py_statistics(hypotheses, references)
        if not self._force:
            _tok_warning(tok_count)
        return _matrix(buf, n, width)

    def sentence_score(self, hypothesis: str, references: Sequence[str]):
        """Same as sacreBLEU's ``sentence_score`` (native statistics, no numpy round-trip)."""
        self._check_sentence_score_args(hypothesis, references)
        refs = [r for r in references if r is not None]
        if not refs or not self._native_ok():
            return _SacreMetric.sentence_score(self, hypothesis, references)
        self.num_refs = len(refs)
        try:
            stats = _swadesh.chrf_stats_one(hypothesis, refs, *self._args())
        except _FALLBACK_ERRORS:
            return _SacreMetric.sentence_score(self, hypothesis, references)
        return self._compute_score_from_stats(stats)

    def sentence_scores(self, hypotheses: Sequence[str], references: Sequence[Sequence[str]]) -> List[float]:
        """Sentence-level chrF for every segment, computed in parallel."""
        streams = _normalize_refs(references)
        hypotheses, streams = _truncate(hypotheses, streams)
        out = _swadesh.chrf_sentence_scores(hypotheses, streams, *self._args())
        _set_num_refs_like_loop(self, streams, len(hypotheses))
        return out

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


class TER(_FastMixin, _SacreTER):
    """Bit-exact, accelerated ``sacrebleu.metrics.TER``.

    Same constructor and API. Only the segment statistics (tercom shifts and the
    beam edit distance) are native. Aggregation and scoring use sacreBLEU's own
    ``sum_of_lists`` and score code on the same floats, so the results are identical.
    """

    def __init__(self, normalized: bool = False, no_punct: bool = False, asian_support: bool = False,
                 case_sensitive: bool = False, references: Optional[Sequence[Sequence[str]]] = None):
        super().__init__(normalized=normalized, no_punct=no_punct, asian_support=asian_support,
                         case_sensitive=case_sensitive, references=None)
        if references is not None:
            self._ref_cache = self._cache_references(references)

    def _extract_corpus_statistics(self, hypotheses: Sequence[str], references) -> Any:
        cache = self._resolve_refs(references)
        if cache.has_empty:
            return self._py_statistics(hypotheses, references)
        try:
            hyps, streams = _truncate(hypotheses, cache.streams)
            buf, n, width = _swadesh.ter_stats(
                hyps, streams, bool(self.normalized), bool(self.no_punct),
                bool(self.asian_support), bool(self.case_sensitive),
            )
        except _FALLBACK_ERRORS:
            return self._py_statistics(hypotheses, references)
        return np.frombuffer(buf, dtype=np.float64).reshape(n, width).copy()

    def _aggregate_and_compute(self, stats):
        # sacreBLEU's own sum_of_lists (sequential float +=) on the same values.
        if isinstance(stats, np.ndarray):
            stats = stats.tolist()
        return _SacreTER._aggregate_and_compute(self, stats)

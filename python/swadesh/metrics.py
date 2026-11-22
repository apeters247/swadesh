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

import json as _json
import logging
import statistics as _statistics
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
        # num_refs semantics identical to sacreBLEU's _cache_references: the per-segment count of
        # non-None references, the same for every segment else -1, and has_empty if any segment has
        # none. A None-free stream has that count at every segment by construction, so a C-level
        # scan replaces a generator per segment; anything with None takes sacreBLEU's own loop.
        if not any(None in s for s in streams):
            self.num_refs = len(streams)
            return _RefCache(streams, len(streams) == 0)
        nums = set()
        has_empty = False
        for refs in zip(*streams):
            n = sum(1 for x in refs if x is not None)
            nums.add(n)
            has_empty = has_empty or n == 0
        self.num_refs = nums.pop() if len(nums) == 1 else -1
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
        # `references` may be a `_RefCache` (when the caller passed none) or a raw stream list.
        streams = references.streams if isinstance(references, _RefCache) else references
        if streams:
            ref_cache = self._segment_ref_cache(streams)
        elif self._ref_cache:
            ref_cache = self._segment_ref_cache(self._ref_cache.streams)
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

    def _score_from_row(self, row):
        """Build our own score object from one summed statistics row.

        Each metric overrides this. The default falls back to sacreBLEU's aggregate so an
        unrecognised metric keeps working.
        """
        raise NotImplementedError

    def _sentence_score(self, hypothesis, references):
        """Score one segment against its references.

        The reference list may contain `None` entries; they are not references, and a segment left
        with none is the metric's own error path, which the statistics call decides.
        """
        self._check_sentence_score_args(hypothesis, references)
        stats = self._extract_corpus_statistics([hypothesis], [[r] for r in references])
        return self._aggregate_and_compute(stats)

    def _segment_ref_cache(self, references):
        """Per-segment reference info, and `num_refs` resolved from it.

        For each segment the references are gathered across the streams and the `None` entries are
        dropped *before* the count is taken — that is what makes a segment with one real reference
        and one `None` a one-reference segment, and it is why `num_refs` can be -1: the counts vary
        across segments. Each surviving line is preprocessed with this metric's segment
        preprocessor before the per-metric reference info is extracted.
        """
        cache = []
        seen = set()
        for refs in zip(*references):
            lines = [x for x in refs if x is not None]
            seen.add(len(lines))
            lines = [self._preprocess_segment(x) for x in lines]
            cache.append(self._extract_reference_info(lines))
        self.num_refs = seen.pop() if len(seen) == 1 else -1
        return cache

    def _aggregate_and_compute(self, stats):
        if isinstance(stats, np.ndarray):
            row = stats.tolist() if stats.ndim == 1 else stats.sum(0).tolist()
            return self._score_from_row(row)
        # List of rows: sum them here and build our own score object. `_compute_score_from_stats`
        # itself is left alone (it stays sacreBLEU's), because the bootstrap fast path relies on
        # recognising the untouched method; only the aggregate step builds the object.
        return self._score_from_row(sum_of_lists(list(stats)))


def _is_int(x) -> bool:
    return isinstance(x, int)




def sum_of_lists(lists):
    """Sum equal-length numeric rows element-wise, preserving the row's numeric type.

    One row is returned as-is. The accumulation order is the row order, which the score arithmetic
    depends on; seeding with a zero of the first element's type keeps integer rows integer.
    """
    if len(lists) == 1:
        return lists[0]
    size = len(lists[0])
    total = [type(lists[0][0])(0.0)] * size
    for row in lists:
        for i in range(size):
            total[i] += row[i]
    return total


class Score:
    """A metric score: the number, its name and optionally a confidence interval.

    `format` produces the three shapes the command line prints — the text line with its signature,
    the bare score, and JSON — and `__repr__` is the text form. The JSON key order matters, so keys
    are inserted in the order they appear: name, score, signature, the confidence fields, the
    verbose breakdown, then the signature parts.
    """

    def __init__(self, name: str, score: float):
        self.name = name
        self.score = score
        self._mean = -1.0
        self._ci = -1.0
        self._verbose = ""

    def format(self, width: int = 2, score_only: bool = False, signature: str = "",
               is_json: bool = False) -> str:
        d = {
            "name": self.name,
            "score": float(f"{self.score:.{width}f}"),
            "signature": signature,
        }

        sc = f"{self.score:.{width}f}"

        if self._mean > 0:
            confidence_mean = f"{self._mean:.{width}f}"
            confidence_var = f"{self._ci:.{width}f}"
            confidence_str = f"μ = {confidence_mean} ± {confidence_var}"
            sc += f" ({confidence_str})"
            if is_json:
                d["confidence_mean"] = float(confidence_mean)
                d["confidence_var"] = float(confidence_var)
                d["confidence"] = confidence_str

        full_score = f"{self.name}|{signature}" if signature else self.name
        full_score = f"{full_score} = {sc}"
        if self._verbose:
            full_score += f" {self._verbose}"
            d["verbose_score"] = self._verbose

        if score_only:
            return sc

        if is_json:
            for param in signature.split("|"):
                key, value = param.split(":")
                d[key] = value
            return _json.dumps(d, indent=1, ensure_ascii=False)

        return full_score

    def estimate_ci(self, scores):
        """Mean and 95% confidence interval from bootstrap resamples of this score."""
        raw = sorted([x.score for x in scores])
        n = len(raw)
        lower_idx = n // 40
        upper_idx = n - lower_idx - 1
        lower, upper = raw[lower_idx], raw[upper_idx]
        self._ci = 0.5 * (upper - lower)
        self._mean = _statistics.mean(raw)

    def __repr__(self):
        return self.format()



class BLEUScore(Score):
    """BLEU's score object: the precisions, the brevity penalty and the lengths."""

    def __init__(self, score: float, counts, totals, precisions, bp: float, sys_len: int, ref_len: int):
        super().__init__("BLEU", score)
        self.bp = bp
        self.counts = counts
        self.totals = totals
        self.sys_len = sys_len
        self.ref_len = ref_len
        self.precisions = precisions

        self.prec_str = "/".join([f"{p:.1f}" for p in self.precisions])
        self.ratio = self.sys_len / self.ref_len if self.ref_len else 0

        # The verbose part of the text line, which the JSON form also carries.
        self._verbose = f"{self.prec_str} (BP = {self.bp:.3f} "
        self._verbose += f"ratio = {self.ratio:.3f} hyp_len = {self.sys_len:d} "
        self._verbose += f"ref_len = {self.ref_len:d})"


class CHRFScore(Score):
    """chrF's score object. The name carries the beta and one '+' per word order."""

    def __init__(self, score: float, char_order: int, word_order: int, beta: int):
        self.beta = beta
        self.char_order = char_order
        self.word_order = word_order
        name = f"chrF{self.beta}" + "+" * self.word_order
        super().__init__(name, score)


class TERScore(Score):
    """TER's score object: the cumulative edits and the cumulative reference length."""

    def __init__(self, score: float, num_edits: float, ref_length: float):
        super().__init__("TER", score)
        self.num_edits = int(num_edits)
        self.ref_length = ref_length


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
            return self._sentence_score(hypothesis, references)
        self.num_refs = len(refs)
        try:
            stats = _swadesh.bleu_stats_one(hypothesis, refs, tok, bool(self.lowercase), self.max_ngram_order)
        except _FALLBACK_ERRORS:
            return self._sentence_score(hypothesis, references)
        return self._compute_score_from_stats(stats)

    def _score_from_row(self, row):
        """BLEU score object from `[sys_len, ref_len, correct_1..N, total_1..N]`."""
        n = self.max_ngram_order
        raw = [float(x) for x in row]
        # Degenerate configuration or a row that is not of this order: sacreBLEU's own arithmetic
        # produces its quirky empty object, which the native scorer refuses.
        if n < 1 or len(row) != 2 + 2 * n or not isinstance(self.smooth_value, (int, float, type(None))):
            return self._compute_score_from_stats(list(row))
        # A fractional row is a float32 bootstrap sum; truncating it would move the score, so those
        # also go through sacreBLEU's scorer.
        if any(float(x) != float(int(x)) for x in raw):
            return self._compute_score_from_stats(list(row))
        ints = [int(x) for x in raw]
        score, bp, precisions = _swadesh.bleu_result_i64(
            np.asarray(ints, dtype=np.int64).tobytes(),
            n,
            self.smooth_method,
            None if self.smooth_value is None else float(self.smooth_value),
            bool(self.effective_order),
        )
        counts = [ints[2 + k] for k in range(n)]
        totals = [ints[2 + n + k] for k in range(n)]
        if self.smooth_method == "add-k":
            sv = self.smooth_value if self.smooth_value is not None else 1.0
            for k in range(1, n):
                counts[k] += int(sv)
                totals[k] += int(sv)
        return BLEUScore(score, counts, totals, list(precisions), bp, ints[0], ints[1])

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
            return self._sentence_score(hypothesis, references)
        self.num_refs = len(refs)
        try:
            stats = _swadesh.chrf_stats_one(hypothesis, refs, *self._args())
        except _FALLBACK_ERRORS:
            return self._sentence_score(hypothesis, references)
        return self._compute_score_from_stats(stats)

    def _score_from_row(self, row):
        """chrF score object from `[hyp, ref, match]` per order."""
        order = self.char_order + self.word_order
        # A fractional beta is outside the native kernel's integral beta; sacreBLEU's own f-score
        # handles it, and truncating would change the score.
        if order < 1 or len(row) != 3 * order or float(self.beta) != float(int(self.beta)):
            return self._compute_score_from_stats(list(row))
        ints = [int(x) for x in row]
        if any(float(x) != float(i) for x, i in zip(row, ints)):
            return self._compute_score_from_stats(list(row))
        v = _swadesh.chrf_scores_i64(
            np.asarray(ints, dtype=np.int64).tobytes(), 3 * order, order, int(self.beta),
            bool(self.eps_smoothing),
        )
        return CHRFScore(v[0], self.char_order, self.word_order, self.beta)

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

    def _score_from_row(self, row):
        """TER score object from `[edits, summed reference lengths]`.

        Three cases, because a corpus can contain an empty reference: a positive total reference
        length divides; a zero length with edits is an empty reference against a non-empty
        hypothesis and scores 1.0; nothing at all scores 0.
        """
        total_edits, sum_ref_lengths = float(row[0]), float(row[1])
        if sum_ref_lengths > 0:
            score = total_edits / sum_ref_lengths
        elif total_edits > 0:
            score = 1.0
        else:
            score = 0.0
        return TERScore(100 * score, total_edits, sum_ref_lengths)

"""Drop-in, bit-exact accelerated replacements for sacreBLEU's metric classes.

The classes below *subclass* sacreBLEU's own ``BLEU`` and ``CHRF``. The hot path -- tokenization,
n-gram extraction and matching, i.e. the segment level sufficient statistics -- is replaced by
Rust, and the score objects, the aggregation and the fallback statistics are swadesh's own code.
The remaining inheritance supplies the constructors, the tokenizers, the degraded score arithmetic
behind ``_compute_score_from_stats`` and the signature printing, so corpus-level results are
identical by construction.

Inputs the native kernels do not cover (invalid parameter values, non-string items, segments with
no reference) are not guessed at: they go through the Python statistics loop
(``_FastMixin._py_statistics``), whose per-metric ``_extract_reference_info`` and
``_compute_segment_statistics`` reproduce sacreBLEU's measured arithmetic, so the result or the
exception is sacreBLEU's, not an artefact of the Rust layer.
"""

import json as _json
import math
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
from . import _swadesh
from .base import Metric as _NativeMetric
from .base import Signature as _NativeSignature

#: BLEU's default maximum n-gram order (sacreBLEU's `BLEU.MAX_NGRAM_ORDER`).
MAX_NGRAM_ORDER = 4


def _my_log(num: float) -> float:
    """sacreBLEU's `my_log`: `log`, floored so a zero precision does not give `-inf`."""
    if num == 0.0:
        return -9999999999
    return math.log(num)


def _compute_bleu(correct, total, sys_len: int, ref_len: int, smooth_method: str = "none",
                  smooth_value=None, effective_order: bool = False,
                  max_ngram_order: int = MAX_NGRAM_ORDER):
    """BLEU's score from its sufficient statistics, with smoothing.

    A port of sacreBLEU's `BLEU.compute_bleu`. The native scorer in `_score_from_row` covers the
    ordinary integer rows; this is the fallback it delegates to for the cases the native scorer
    refuses -- a degenerate n-gram order and a fractional (float32 bootstrap) row -- where
    sacreBLEU's own arithmetic is what has to be reproduced, quirks included. Smoothing per Boxing
    Chen and Colin Cherry, WMT 2014: `none`, `floor`, `add-k`, `exp` (NIST).
    """
    assert smooth_method in BLEU.SMOOTH_DEFAULTS.keys(), \
        "Unknown smooth_method {smooth_method!r}"

    # Fetch the default value for floor and add-k
    if smooth_value is None:
        smooth_value = BLEU.SMOOTH_DEFAULTS[smooth_method]

    # Compute brevity penalty
    bp = 1.0
    if sys_len < ref_len:
        bp = math.exp(1 - ref_len / sys_len) if sys_len > 0 else 0.0

    # n-gram precisions
    precisions = [0.0] * max_ngram_order

    # Early stop if there are no matches (#141)
    if not any(correct):
        return BLEUScore(0.0, correct, total, precisions, bp, sys_len, ref_len)

    smooth_mteval = 1.
    eff_order = max_ngram_order
    for n in range(1, len(precisions) + 1):
        if smooth_method == 'add-k' and n > 1:
            correct[n - 1] += smooth_value
            total[n - 1] += smooth_value

        if total[n - 1] == 0:
            break

        # If the system guesses no i-grams, 1 <= i <= max_ngram_order, the BLEU score is 0
        # (technically undefined). This is a problem for sentence-level BLEU or a corpus of short
        # sentences, where systems will get no credit if sentence lengths fall under the
        # max_ngram_order threshold. `effective_order` scales max_ngram_order to the observed
        # maximum order; it is off by default.
        if effective_order:
            eff_order = n

        if correct[n - 1] == 0:
            if smooth_method == 'exp':
                smooth_mteval *= 2
                precisions[n - 1] = 100. / (smooth_mteval * total[n - 1])
            elif smooth_method == 'floor':
                precisions[n - 1] = 100. * smooth_value / total[n - 1]
        else:
            precisions[n - 1] = 100. * correct[n - 1] / total[n - 1]

    # Compute BLEU score
    score = bp * math.exp(
        sum([_my_log(p) for p in precisions[:eff_order]]) / eff_order)

    return BLEUScore(score, correct, total, precisions, bp, sys_len, ref_len)

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


class TERSignature(_NativeSignature):
    """TER's signature: the base fields plus the four TER flags.

    A port of sacreBLEU's `TERSignature`. `tok` is `self.tokenizer_signature`, which is the
    constant `"tercom"` for every flag combination (measured 2026-11-25).
    """

    def __init__(self, args: dict):
        super().__init__(args)
        self._abbr.update({
            "case": "c",
            "tok": "t",
            "norm": "nr",
            "punct": "pn",
            "asian": "as",
        })
        self.info.update({
            "case": "mixed" if args["case_sensitive"] else "lc",
            "tok": args["tokenizer_signature"],
            "norm": args["normalized"],
            "punct": not args["no_punct"],
            "asian": args["asian_support"],
        })


class BLEUSignature(_NativeSignature):
    """BLEU's signature: the base fields plus case, effective order, tokenizer and smoothing.

    A port of sacreBLEU's `BLEUSignature`.
    """

    def __init__(self, args: dict):
        super().__init__(args)
        self._abbr.update({
            "case": "c",
            "eff": "e",
            "tok": "tok",
            "smooth": "s",
        })
        smooth_str = args["smooth_method"]
        smooth_def = BLEU.SMOOTH_DEFAULTS[smooth_str]
        if smooth_def is not None:
            smooth_val = args["smooth_value"]
            if smooth_val is None:
                smooth_val = smooth_def
            smooth_str += f"[{smooth_val:.2f}]"
        self.info.update({
            "case": "lc" if args["lowercase"] else "mixed",
            "eff": "yes" if args["effective_order"] else "no",
            "tok": args["tokenizer_signature"],
            "smooth": smooth_str,
        })


class CHRFSignature(_NativeSignature):
    """chrF's signature: the base fields plus case, effective order and the orders.

    A port of sacreBLEU's `CHRFSignature`.
    """

    def __init__(self, args: dict):
        super().__init__(args)
        self._abbr.update({
            "case": "c",
            "eff": "e",
            "nc": "nc",
            "nw": "nw",
            "space": "s",
        })
        self.info.update({
            "case": "lc" if args["lowercase"] else "mixed",
            "eff": "yes" if not args["eps_smoothing"] else "no",
            "nc": args["char_order"],
            "nw": args["word_order"],
            "space": "yes" if args["whitespace"] else "no",
        })


class _NativeBleuTokenizer:
    """BLEU's per-name tokenizer object, natively, for the five native names.

    sacreBLEU builds a tokenizer class per name, keeps it on the metric as `tokenizer` and its
    `signature()` is the tokenizer name; `_preprocess_segment` calls it. The native path covers
    `none`, `13a`, `intl`, `zh` and `char`. The exotic names (`ja-mecab`, `ko-mecab`, `spm`,
    `flores101`, `flores200`, `spBLEU-1K`) need extra libraries (mecab, sentencepiece) and are
    supplied by sacreBLEU when it is installed -- imported lazily, so the native path never needs
    it.
    """

    def __init__(self, name: str):
        self.name = name

    def __call__(self, sent: str) -> str:
        return _swadesh.tokenize(self.name, sent)

    def signature(self) -> str:
        return self.name


def _make_bleu_tokenizer(name: str):
    if name in NATIVE_TOKENIZERS:
        return _NativeBleuTokenizer(name)
    from sacrebleu.metrics.bleu import _get_tokenizer  # lazy: exotic tokenizers only

    return _get_tokenizer(name)()


#: The tokenizer names BLEU accepts, in sacreBLEU's order. Five of them build natively; the rest
#: need mecab or sentencepiece and are built by sacreBLEU when it is installed. sacreBLEU exposes
#: this same set as a key view (`BLEU.TOKENIZERS`), which is what `tests/test_parity.py` compares.
_BLEU_TOKENIZERS = {
    name: (lambda n=name: _make_bleu_tokenizer(n))
    for name in ("none", "zh", "13a", "intl", "char", "ja-mecab", "ko-mecab", "spm",
                 "flores101", "flores200", "spBLEU-1K")
}


class BLEU(_FastMixin, _NativeMetric):
    """Bit-exact, accelerated ``sacrebleu.metrics.BLEU``.

    Same constructor and API. Additional method: :meth:`sentence_scores`.
    """

    SMOOTH_DEFAULTS = {'none': None, 'floor': 0.1, 'add-k': 1, 'exp': None}
    TOKENIZER_DEFAULT = '13a'
    _TOKENIZER_MAP = {'zh': 'zh', 'ja': 'ja-mecab', 'ko': 'ko-mecab'}

    # A key view, as in sacreBLEU: the names BLEU accepts, not a lookup table.
    TOKENIZERS = _BLEU_TOKENIZERS.keys()

    _SIGNATURE_TYPE = BLEUSignature

    def __init__(self, lowercase: bool = False, force: bool = False, tokenize: Optional[str] = None,
                 smooth_method: str = 'exp', smooth_value: Optional[float] = None,
                 max_ngram_order: int = MAX_NGRAM_ORDER, effective_order: bool = False,
                 trg_lang: str = '', references: Optional[Sequence[Sequence[str]]] = None):
        _NativeMetric.__init__(self)
        self._force = force
        self.trg_lang = trg_lang
        self.lowercase = lowercase
        self.smooth_value = smooth_value
        self.smooth_method = smooth_method
        self.max_ngram_order = max_ngram_order
        self.effective_order = effective_order

        # Sanity check
        assert self.smooth_method in self.SMOOTH_DEFAULTS.keys(), \
            "Unknown smooth_method {self.smooth_method!r}"

        # Without an explicit tokenizer, the default is 13a except for ZH/JA/KO target languages.
        if tokenize is None:
            best_tokenizer = self.TOKENIZER_DEFAULT
            if self.trg_lang in self._TOKENIZER_MAP:
                best_tokenizer = self._TOKENIZER_MAP[self.trg_lang]
        else:
            best_tokenizer = tokenize
            if self.trg_lang == 'zh' and best_tokenizer != 'zh':
                sacrelogger.warning(
                    "Consider using the 'zh' or 'spm' tokenizer for Chinese.")
            if self.trg_lang == 'ja' and best_tokenizer != 'ja-mecab':
                sacrelogger.warning(
                    "Consider using the 'ja-mecab' or 'spm' tokenizer for Japanese.")
            if self.trg_lang == 'ko' and best_tokenizer != 'ko-mecab':
                sacrelogger.warning(
                    "Consider using the 'ko-mecab' or 'spm' tokenizer for Korean.")

        self.tokenizer = _make_bleu_tokenizer(best_tokenizer)
        self.tokenizer_signature = self.tokenizer.signature()

        if references is not None:
            # Pre-compute reference ngrams and lengths
            self._ref_cache = self._cache_references(references)

    def _preprocess_segment(self, sent: str) -> str:
        """sacreBLEU's BLEU pre-processing: optional lowercasing, then the tokenizer."""
        if self.lowercase:
            sent = sent.lower()
        return self.tokenizer(sent.rstrip())

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

    def _extract_reference_info(self, refs):
        """Reference n-gram counts and token lengths, one entry per reference.

        `refs` arrives already preprocessed -- lowercased and tokenised by the time this runs --
        so a reference is its whitespace split. The counts keep, for each n-gram, the largest
        count any single reference has of it: clipping is against the best reference, not against
        the sum of them. `ref_lens` keeps every length, because the row picks the closest one.
        """
        ref_ngrams = None
        ref_lens = []
        for ref in refs:
            tokens = ref.split()
            ref_lens.append(len(tokens))
            counts = {}
            for n in range(1, self.max_ngram_order + 1):
                for i in range(len(tokens) - n + 1):
                    ngram = tuple(tokens[i:i + n])
                    counts[ngram] = counts.get(ngram, 0) + 1
            if ref_ngrams is None:
                ref_ngrams = counts
            else:
                for ngram, count in counts.items():
                    if count > ref_ngrams.get(ngram, 0):
                        ref_ngrams[ngram] = count
        return {"ref_ngrams": ref_ngrams, "ref_lens": ref_lens}

    def _compute_segment_statistics(self, hypothesis, ref_kwargs):
        """`[sys_len, ref_len, correct_1..N, total_1..N]` for one preprocessed segment.

        `ref_len` is the reference length closest to the hypothesis, a tie going to the shorter
        reference, and -1 when the segment has no reference at all. An order's matches are the
        hypothesis n-grams clipped by the merged counts. A segment with no references has no
        merged counter, and testing its n-grams against it raises the same error the oracle's own
        loop raises for that case.
        """
        tokens = hypothesis.split()
        sys_len = len(tokens)
        ref_lens = ref_kwargs["ref_lens"]
        ref_len = min(ref_lens, key=lambda r: (abs(r - sys_len), r)) if ref_lens else -1
        ref_ngrams = ref_kwargs["ref_ngrams"]
        correct = []
        total = []
        for n in range(1, self.max_ngram_order + 1):
            counts = {}
            for i in range(sys_len - n + 1):
                ngram = tuple(tokens[i:i + n])
                counts[ngram] = counts.get(ngram, 0) + 1
            matched = 0
            for ngram, count in counts.items():
                if ngram in ref_ngrams:
                    matched += min(count, ref_ngrams[ngram])
            correct.append(matched)
            total.append(max(0, sys_len - n + 1))
        return [sys_len, ref_len] + correct + total

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

    def _compute_score_from_stats(self, stats) -> Any:
        """sacreBLEU's BLEU scorer, from `[sys_len, ref_len, correct_1..N, total_1..N]`.

        The significant thing it does that the native scorer does not is keep numpy dtypes and
        reproduce the smoothing arithmetic exactly, which the fallbacks in `_score_from_row` need.
        """
        return _compute_bleu(
            correct=stats[2:2 + self.max_ngram_order],
            total=stats[2 + self.max_ngram_order:],
            sys_len=int(stats[0]), ref_len=int(stats[1]),
            smooth_method=self.smooth_method, smooth_value=self.smooth_value,
            effective_order=self.effective_order,
            max_ngram_order=self.max_ngram_order,
        )

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
        # `add-k` shows up in the reported counts and totals as well as in the score, but only when
        # the corpus has at least one match at the first order: with none, the oracle leaves both
        # blocks raw (measured 2026-11-24, docs/lab-notebook.md). The value is added unchanged, so
        # the blocks can become fractional (`smooth_value=0.5` gives counts [2, 1.5, 0.5, 0.5]).
        if self.smooth_method == "add-k" and counts and counts[0] > 0:
            sv = self.smooth_value if self.smooth_value is not None else 1
            for k in range(1, n):
                counts[k] += sv
                totals[k] += sv
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


class CHRF(_FastMixin, _NativeMetric):
    """Bit-exact, accelerated ``sacrebleu.metrics.CHRF`` (chrF and chrF++).

    Same constructor and API. Additional method: :meth:`sentence_scores`.
    """

    # Maximum character n-gram order to take into account
    CHAR_ORDER = 6

    # chrF+ additionally takes into account some of the word n-grams
    WORD_ORDER = 0

    # Defaults to 2 (per http://www.aclweb.org/anthology/W16-2341)
    BETA = 2

    # Cache string.punctuation for chrF+' punctuation stripper
    _PUNCTS = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')

    _SIGNATURE_TYPE = CHRFSignature

    def __init__(self, char_order: int = CHAR_ORDER, word_order: int = WORD_ORDER,
                 beta: int = BETA, lowercase: bool = False, whitespace: bool = False,
                 eps_smoothing: bool = False, references: Optional[Sequence[Sequence[str]]] = None):
        _NativeMetric.__init__(self)
        self.beta = beta
        self.char_order = char_order
        self.word_order = word_order
        self.order = self.char_order + self.word_order
        self.lowercase = lowercase
        self.whitespace = whitespace
        self.eps_smoothing = eps_smoothing

        if references is not None:
            # Pre-compute reference ngrams
            self._ref_cache = self._cache_references(references)

    def _remove_punctuation(self, sent: str) -> List[str]:
        """Separates out punctuations from beginning and end of words for chrF.

        A port of sacreBLEU's method, itself adapted from m-popovic/chrF: a word longer than one
        character that ends in a punctuation mark is split before it, otherwise one that starts
        with a punctuation mark is split after it.
        """
        tokenized = []
        for w in sent.split():
            if len(w) == 1:
                tokenized.append(w)
            else:
                # NOTE: This splits '(hi)' to '(hi' and ')' (issue #124)
                if w[-1] in self._PUNCTS:
                    tokenized += [w[:-1], w[-1]]
                elif w[0] in self._PUNCTS:
                    tokenized += [w[0], w[1:]]
                else:
                    tokenized.append(w)
        return tokenized

    def _preprocess_segment(self, sent: str) -> str:
        """sacreBLEU's chrF pre-processing: optional lowercasing only."""
        return sent.lower() if self.lowercase else sent

    def _native_ok(self) -> bool:
        return (_is_int(self.char_order) and self.char_order >= 1
                and _is_int(self.word_order) and self.word_order >= 0 and _is_int(self.beta))

    def _args(self):
        return (self.char_order, self.word_order, self.beta, bool(self.lowercase),
                bool(self.whitespace), bool(self.eps_smoothing))

    def _extract_reference_info(self, refs):
        """Per-reference counts: `char_order` character orders, then `word_order` word orders.

        `refs` arrives already lowercased when that option is on. Unless `whitespace` keeps it,
        whitespace is dropped from the character sequence -- all of it, as Python's `str.isspace`
        defines it -- and word n-grams are the chrF punctuation split. The counters stay per
        reference because the segment row is later taken from the single best reference.
        """
        ref_ngrams = []
        for ref in refs:
            chars = ref if self.whitespace else "".join(ref.split())
            orders = []
            for n in range(1, self.char_order + 1):
                counts = {}
                for i in range(len(chars) - n + 1):
                    ngram = chars[i:i + n]
                    counts[ngram] = counts.get(ngram, 0) + 1
                orders.append(counts)
            if self.word_order > 0:
                words = self._remove_punctuation(ref)
                for n in range(1, self.word_order + 1):
                    counts = {}
                    for i in range(len(words) - n + 1):
                        ngram = " ".join(words[i:i + n])
                        counts[ngram] = counts.get(ngram, 0) + 1
                    orders.append(counts)
            ref_ngrams.append(orders)
        return {"ref_ngrams": ref_ngrams}

    def _compute_segment_statistics(self, hypothesis, ref_kwargs):
        """`[hyp, ref, match]` per order, as the single best-scoring reference reports them.

        Statistics are not merged across references: the row is the row of the reference with the
        highest f-score, and a tie keeps the first such reference -- the f-score the metric itself
        computes, so the choice is the one the score would make. An order's hypothesis total is
        reported only when that reference has n-grams of the order at all.
        """
        chars = hypothesis if self.whitespace else "".join(hypothesis.split())
        hyp_char = []
        for n in range(1, self.char_order + 1):
            counts = {}
            for i in range(len(chars) - n + 1):
                ngram = chars[i:i + n]
                counts[ngram] = counts.get(ngram, 0) + 1
            hyp_char.append(counts)
        hyp_word = None
        if self.word_order > 0:
            words = self._remove_punctuation(hypothesis)
            hyp_word = []
            for n in range(1, self.word_order + 1):
                counts = {}
                for i in range(len(words) - n + 1):
                    ngram = " ".join(words[i:i + n])
                    counts[ngram] = counts.get(ngram, 0) + 1
                hyp_word.append(counts)
        best, best_f = [], -1.0
        for ref_orders in ref_kwargs["ref_ngrams"]:
            counters = iter(ref_orders)
            stats = []
            for hyp_counts in hyp_char:
                ref_counts = next(counters)
                ref_total = sum(ref_counts.values())
                stats.append(sum(hyp_counts.values()) if ref_total > 0 else 0)
                stats.append(ref_total)
                stats.append(sum(min(count, ref_counts.get(ngram, 0))
                                 for ngram, count in hyp_counts.items()))
            for n in range(1, self.word_order + 1):
                hyp_counts = hyp_word[n - 1]
                ref_counts = next(counters)
                ref_total = sum(ref_counts.values())
                stats.append(sum(hyp_counts.values()) if ref_total > 0 else 0)
                stats.append(ref_total)
                stats.append(sum(min(count, ref_counts.get(ngram, 0))
                                 for ngram, count in hyp_counts.items()))
            f = self._compute_f_score(stats)
            if f > best_f:
                best, best_f = stats, f
        return best

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

    def _compute_f_score(self, statistics) -> float:
        """chrF's f_beta score from the flattened `[hyp, ref, match]` counts.

        A port of sacreBLEU's `CHRF._compute_f_score`. The native `_swadesh.chrf_f_score` covers the
        ordinary path; this is the fallback for the rows it refuses.
        """
        eps = 1e-16
        score = 0.0
        effective_order = 0
        factor = self.beta ** 2
        avg_prec, avg_rec = 0.0, 0.0

        for i in range(self.order):
            n_hyp, n_ref, n_match = statistics[3 * i: 3 * i + 3]

            # chrF++.py style EPS smoothing (also used by Moses and NLTK)
            prec = n_match / n_hyp if n_hyp > 0 else eps
            rec = n_match / n_ref if n_ref > 0 else eps

            denom = factor * prec + rec
            score += ((1 + factor) * prec * rec / denom) if denom > 0 else eps

            # sacreBLEU <2.0.0 style effective order smoothing
            if n_hyp > 0 and n_ref > 0:
                avg_prec += prec
                avg_rec += rec
                effective_order += 1

        if self.eps_smoothing:
            return 100 * score / self.order

        if effective_order == 0:
            avg_prec = avg_rec = 0.0
        else:
            avg_prec /= effective_order
            avg_rec /= effective_order

        if avg_prec + avg_rec:
            score = (1 + factor) * avg_prec * avg_rec
            score /= ((factor * avg_prec) + avg_rec)
            return 100 * score
        else:
            return 0.0

    def _compute_score_from_stats(self, stats) -> Any:
        """sacreBLEU's chrF scorer, from the flattened `[hyp, ref, match]` counts."""
        return CHRFScore(
            self._compute_f_score(stats), self.char_order, self.word_order, self.beta)

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


class _NativeTerTokenizer:
    """Stands in for sacreBLEU's ``TercomTokenizer`` on a TER instance.

    sacreBLEU stores its tokenizer on the metric object and downstream code may call it, so the
    native one is exposed under the same attribute name -- ``tests/test_parity.py`` checks that a
    TER instance has the same attribute names as sacreBLEU's. ``signature()`` is the constant
    ``"tercom"`` for every flag combination (measured 2026-11-25).
    """

    def __init__(self, normalized: bool, no_punct: bool, asian_support: bool,
                 case_sensitive: bool):
        self.normalized = normalized
        self.no_punct = no_punct
        self.asian_support = asian_support
        self.case_sensitive = case_sensitive

    def __call__(self, sent: str) -> str:
        """sacreBLEU's `TercomTokenizer(sent.rstrip())`, natively.

        The rstrip and CPython's `lower()` (final-sigma rule included) happen here; the tokenizing
        and the normalized-mode entity unescaping are the Rust kernel's. Verified byte-identical to
        `TercomTokenizer(...)` for all sixteen flag combinations by `tests/test_ter_preprocess.py`.
        """
        stripped = sent.rstrip()
        if not self.case_sensitive:
            stripped = stripped.lower()
        if not isinstance(stripped, str):
            # sacreBLEU's tokenizer ends in `" ".join(tokens)`, and that is where a non-str
            # segment (bytes, say) fails. Do it here so the exception and its message are the
            # same; the native extension would raise its own TypeError with different wording.
            " ".join([stripped])
        return _swadesh.ter_preprocess(
            stripped, bool(self.normalized), bool(self.no_punct),
            bool(self.asian_support), bool(self.case_sensitive),
        )

    def signature(self) -> str:
        return "tercom"


class TER(_FastMixin, _NativeMetric):
    """Bit-exact, accelerated ``sacrebleu.metrics.TER``.

    Same constructor and API. The tokenizer and the segment statistics (tercom shifts and the beam
    edit distance) are native, so this class no longer uses sacreBLEU's tokenizer module. What it
    still takes from the class it subclasses is the constructor, the score-object machinery and the
    signature printing.
    """

    _SIGNATURE_TYPE = TERSignature

    def __init__(self, normalized: bool = False, no_punct: bool = False, asian_support: bool = False,
                 case_sensitive: bool = False, references: Optional[Sequence[Sequence[str]]] = None):
        _NativeMetric.__init__(self)
        self.normalized = normalized
        self.no_punct = no_punct
        self.asian_support = asian_support
        self.case_sensitive = case_sensitive
        self.tokenizer = _NativeTerTokenizer(
            normalized, no_punct, asian_support, case_sensitive,
        )
        self.tokenizer_signature = self.tokenizer.signature()
        if references is not None:
            self._ref_cache = self._cache_references(references)

    def _compute_score_from_stats(self, stats) -> Any:
        """TER's score object from `[edits, summed reference lengths]`.

        Deliberately not `_score_from_row`: the significance paths call this with numpy rows
        (`significance._bootstrap_resample`, `_paired_ar_test`) and sacreBLEU carries the row's
        dtype into the score object, so an `np.float32` bootstrap row has to give an `np.float32`
        score. Coercing to Python floats there changes what the confidence intervals report.
        """
        total_edits, sum_ref_lengths = stats[0], stats[1]
        if sum_ref_lengths > 0:
            score = total_edits / sum_ref_lengths
        elif total_edits > 0:
            score = 1.0
        else:
            score = 0.0
        return TERScore(100 * score, total_edits, sum_ref_lengths)

    def _preprocess_segment(self, sent: str) -> str:
        """sacreBLEU's `TercomTokenizer(sent.rstrip())`, natively -- see `_NativeTerTokenizer`."""
        return self.tokenizer(sent)

    def _extract_corpus_statistics(self, hypotheses: Sequence[str], references) -> Any:
        cache = self._resolve_refs(references)
        if cache.has_empty:
            return self._py_statistics(hypotheses, references)
        try:
            hyps, streams = _truncate(hypotheses, cache.streams)
            # The oracle preprocesses each reference twice: once when the references are cached and
            # once when they are extracted (docs/spec/ter.md section 5). The kernel tokenizes the
            # text it is handed, so giving it already-preprocessed references is the second pass.
            streams = [[None if x is None else self._preprocess_segment(x) for x in s]
                       for s in streams]
            buf, n, width = _swadesh.ter_stats(
                hyps, streams, bool(self.normalized), bool(self.no_punct),
                bool(self.asian_support), bool(self.case_sensitive),
            )
        except _FALLBACK_ERRORS:
            return self._py_statistics(hypotheses, references)
        return np.frombuffer(buf, dtype=np.float64).reshape(n, width).copy()

    def _extract_reference_info(self, refs):
        """TER reference words: each reference is preprocessed then split."""
        return {"ref_words": [self._preprocess_segment(r).split() for r in refs]}

    def _compute_segment_statistics(self, hypothesis, ref_kwargs):
        """`[best_num_edits, average reference length]` for one segment.

        The edits are the **minimum** over references while the length is their average: the best
        reference wins, but the length counts every reference. The native kernel computes exactly
        that, so the fallback path uses it rather than a second Python implementation.
        """
        ref_words = ref_kwargs["ref_words"]
        if not ref_words:
            # A segment with no references is TER's error path: it divides by the reference count.
            raise ZeroDivisionError("division by zero")
        # The kernel takes parallel lists: one hypothesis, and that segment's references.
        refs = [" ".join(r) if not isinstance(r, str) else r for r in ref_words]
        buf, n, width = _swadesh.ter_stats(
            [hypothesis], [refs],
            bool(self.normalized), bool(self.no_punct),
            bool(self.asian_support), bool(self.case_sensitive),
        )
        return np.frombuffer(buf, dtype=np.float64).reshape(n, width)[0].tolist()



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

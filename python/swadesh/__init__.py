"""swadesh: bit-exact sacreBLEU, rewritten in Rust.

Named after the Swadesh list -- the anthropologist-linguist Morris Swadesh's
core vocabulary lists used to measure how much two languages share. BLEU and
chrF do the same thing for a translation and its reference: count what they
share.

Usage::

    import swadesh
    swadesh.BLEU().corpus_score(hyps, [refs])     # same API as sacrebleu
    swadesh.corpus_bleu(hyps, [refs])             # same as sacrebleu.corpus_bleu

    swadesh.patch()   # make every `import sacrebleu` user (HF evaluate, fairseq, ...) fast
"""

from __future__ import annotations

import types

import sacrebleu as _sacrebleu
import sacrebleu.compat as _compat
from sacrebleu.metrics.bleu import BLEUScore  # noqa: F401
from sacrebleu.metrics.chrf import CHRFScore  # noqa: F401
from sacrebleu.metrics.ter import TERScore  # noqa: F401

from .metrics import BLEU, CHRF, TER, NATIVE_TOKENIZERS, mbr  # noqa: F401

__version__ = "0.1.0"
SACREBLEU_COMPAT_VERSION = _sacrebleu.__version__


def _rebind(fn):
    g = dict(_compat.__dict__)
    g.update(BLEU=BLEU, CHRF=CHRF, TER=TER)
    new = types.FunctionType(fn.__code__, g, fn.__name__, fn.__defaults__, fn.__closure__)
    new.__kwdefaults__ = fn.__kwdefaults__
    new.__doc__ = fn.__doc__
    return new


corpus_bleu = _rebind(_compat.corpus_bleu)
raw_corpus_bleu = _rebind(_compat.raw_corpus_bleu)
sentence_bleu = _rebind(_compat.sentence_bleu)
corpus_chrf = _rebind(_compat.corpus_chrf)
sentence_chrf = _rebind(_compat.sentence_chrf)
corpus_ter = _rebind(_compat.corpus_ter)
sentence_ter = _rebind(_compat.sentence_ter)

_PATCHED = {}


def patch() -> None:
    """Route all of sacreBLEU (library *and* CLI) through swadesh.

    After ``swadesh.patch()``, code that does ``import sacrebleu`` -- including
    Hugging Face ``evaluate``, fairseq, Marian/OpenNMT scripts -- transparently
    gets the accelerated, bit-identical implementation. Undo with :func:`unpatch`.
    """
    import sacrebleu.metrics as m
    import sacrebleu.metrics.bleu as mb
    import sacrebleu.metrics.chrf as mc
    import sacrebleu.metrics.ter as mt

    if _PATCHED:
        return
    targets = [
        (_sacrebleu, "BLEU", BLEU), (_sacrebleu, "CHRF", CHRF), (_sacrebleu, "TER", TER),
        (m, "BLEU", BLEU), (m, "CHRF", CHRF), (m, "TER", TER),
        (mb, "BLEU", BLEU), (mc, "CHRF", CHRF), (mt, "TER", TER),
        (_compat, "BLEU", BLEU), (_compat, "CHRF", CHRF), (_compat, "TER", TER),
    ]
    for mod, name, new in targets:
        _PATCHED[(mod, name)] = getattr(mod, name)
        setattr(mod, name, new)
    from . import significance as _significance

    _PATCHED[("significance",)] = _significance.install()
    _PATCHED[("METRICS",)] = dict(m.METRICS)
    m.METRICS["BLEU"] = BLEU
    m.METRICS["CHRF"] = CHRF
    m.METRICS["TER"] = TER


def unpatch() -> None:
    """Restore the original sacreBLEU classes."""
    import sacrebleu.metrics as m

    if not _PATCHED:
        return
    m.METRICS.clear()
    m.METRICS.update(_PATCHED.pop(("METRICS",)))
    from . import significance as _significance

    _significance.uninstall(_PATCHED.pop(("significance",)))
    for (mod, name), old in list(_PATCHED.items()):
        setattr(mod, name, old)
    _PATCHED.clear()


def tokenize(line, tokenizer: str = "13a"):
    """Tokenize exactly like sacreBLEU's tokenizer ``tokenizer`` (13a, intl, zh, char, none).
    Pass a list of strings to tokenize a batch in parallel."""
    from . import _swadesh
    if isinstance(line, str):
        return _swadesh.tokenize(tokenizer, line)
    return _swadesh.tokenize_batch(tokenizer, list(line))


def __getattr__(name):
    # Lazy: significance testing pulls in numpy and sacrebleu.significance.
    if name == "PairedTest":
        from .significance import PairedTest

        return PairedTest
    raise AttributeError(f"module 'swadesh' has no attribute {name!r}")

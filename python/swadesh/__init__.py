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
from sacrebleu.metrics import TER  # noqa: F401  (native TER: work in progress)
from sacrebleu.metrics.bleu import BLEUScore  # noqa: F401
from sacrebleu.metrics.chrf import CHRFScore  # noqa: F401

from .metrics import BLEU, CHRF, NATIVE_TOKENIZERS  # noqa: F401
from .significance import PairedTest  # noqa: F401
from . import significance as _significance

__version__ = "0.1.0"
SACREBLEU_COMPAT_VERSION = _sacrebleu.__version__


def _rebind(fn):
    g = dict(_compat.__dict__)
    g.update(BLEU=BLEU, CHRF=CHRF)
    new = types.FunctionType(fn.__code__, g, fn.__name__, fn.__defaults__, fn.__closure__)
    new.__kwdefaults__ = fn.__kwdefaults__
    new.__doc__ = fn.__doc__
    return new


corpus_bleu = _rebind(_compat.corpus_bleu)
raw_corpus_bleu = _rebind(_compat.raw_corpus_bleu)
sentence_bleu = _rebind(_compat.sentence_bleu)
corpus_chrf = _rebind(_compat.corpus_chrf)
sentence_chrf = _rebind(_compat.sentence_chrf)
corpus_ter = _compat.corpus_ter
sentence_ter = _compat.sentence_ter

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

    if _PATCHED:
        return
    targets = [
        (_sacrebleu, "BLEU", BLEU), (_sacrebleu, "CHRF", CHRF),
        (m, "BLEU", BLEU), (m, "CHRF", CHRF),
        (mb, "BLEU", BLEU), (mc, "CHRF", CHRF),
        (_compat, "BLEU", BLEU), (_compat, "CHRF", CHRF),
    ]
    for mod, name, new in targets:
        _PATCHED[(mod, name)] = getattr(mod, name)
        setattr(mod, name, new)
    _PATCHED[("significance",)] = _significance.install()
    _PATCHED[("METRICS",)] = dict(m.METRICS)
    m.METRICS["BLEU"] = BLEU
    m.METRICS["CHRF"] = CHRF


def unpatch() -> None:
    """Restore the original sacreBLEU classes."""
    import sacrebleu.metrics as m

    if not _PATCHED:
        return
    m.METRICS.clear()
    m.METRICS.update(_PATCHED.pop(("METRICS",)))
    _significance.uninstall(_PATCHED.pop(("significance",)))
    for (mod, name), old in list(_PATCHED.items()):
        setattr(mod, name, old)
    _PATCHED.clear()

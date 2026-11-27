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

import sys
import types

from .base import COMPAT_VERSION

# The score objects and the metrics are all ours: none of the three metrics comes from
# sacreBLEU any more.
from .metrics import (  # noqa: F401
    BLEU,
    BLEUScore,
    CHRF,
    CHRFScore,
    NATIVE_TOKENIZERS,
    TER,
    TERScore,
    LengthMismatchError,
    mbr,
    set_strict,
    strict,
)

from .significance import seed, set_seed  # noqa: F401  (the bootstrap seed, in process)

__version__ = "0.1.0"
SACREBLEU_COMPAT_VERSION = COMPAT_VERSION

#: What needs sacreBLEU, and what does not. Used by the lazy importers below.
_NEEDS_SACREBLEU = (
    "this needs sacreBLEU installed (`pip install 'sacrebleu==2.6.0'`). swadesh's own metrics, "
    "significance testing and library API do not; the module-level convenience functions "
    "(corpus_bleu and friends), patch(), the command line and the mecab/sentencepiece tokenizers do."
)

#: sacreBLEU's module-level convenience functions that this package mirrors.
_COMPAT_NAMES = (
    "corpus_bleu",
    "raw_corpus_bleu",
    "sentence_bleu",
    "corpus_chrf",
    "sentence_chrf",
    "corpus_ter",
    "sentence_ter",
)
_COMPAT_CACHE: dict = {}


def _require_sacrebleu():
    """sacreBLEU, or an ImportError that says what needed it and what does not."""
    try:
        import sacrebleu
    except ImportError as e:
        raise ImportError(_NEEDS_SACREBLEU) from e
    return sacrebleu


def _rebind(fn, compat_globals):
    """`fn` with its globals replaced, so its `BLEU`/`CHRF`/`TER` are this package's."""
    g = dict(compat_globals)
    g.update(BLEU=BLEU, CHRF=CHRF, TER=TER)
    new = types.FunctionType(fn.__code__, g, fn.__name__, fn.__defaults__, fn.__closure__)
    new.__kwdefaults__ = fn.__kwdefaults__
    new.__doc__ = fn.__doc__
    # `inspect.signature` reads these, and a drop-in has to show the same signature as sacreBLEU's.
    new.__annotations__ = fn.__annotations__
    return new


def _compat_function(name):
    """sacreBLEU's `compat.<name>`, rebuilt, on first use.

    Built lazily rather than at import: `import swadesh` has to work with sacreBLEU absent, and
    these wrappers are the only part of the public surface that needs it.
    """
    fn = _COMPAT_CACHE.get(name)
    if fn is None:
        try:
            import sacrebleu.compat as _compat
        except ImportError as e:
            raise ImportError(_NEEDS_SACREBLEU) from e
        fn = _rebind(getattr(_compat, name), _compat.__dict__)
        _COMPAT_CACHE[name] = fn
        globals()[name] = fn
    return fn

_PATCHED = {}


def patch() -> None:
    """Route all of sacreBLEU (library *and* CLI) through swadesh.

    After ``swadesh.patch()``, code that does ``import sacrebleu`` -- including
    Hugging Face ``evaluate``, fairseq, Marian/OpenNMT scripts -- transparently
    gets the accelerated, bit-identical implementation. Undo with :func:`unpatch`.
    """
    sacrebleu = _require_sacrebleu()
    import sacrebleu.compat as compat
    import sacrebleu.metrics as m
    import sacrebleu.metrics.bleu as mb
    import sacrebleu.metrics.chrf as mc
    import sacrebleu.metrics.ter as mt

    if _PATCHED:
        return
    # The metrics stand alone -- they are not subclasses of sacreBLEU's -- but downstream code
    # does `isinstance(x, sacrebleu.metrics.bleu.BLEU)` and that has to keep working after a patch.
    # So the classes that get installed are built here, at patch time, as subclasses of both.
    class _BleuCompatible(BLEU, mb.BLEU):
        pass

    class _ChrfCompatible(CHRF, mc.CHRF):
        pass

    class _TerCompatible(TER, mt.TER):
        pass

    for _cls, _name in ((_BleuCompatible, "BLEU"), (_ChrfCompatible, "CHRF"), (_TerCompatible, "TER")):
        _cls.__name__ = _cls.__qualname__ = _name
    bleu, chrf, ter = _BleuCompatible, _ChrfCompatible, _TerCompatible
    targets = [
        (sacrebleu, "BLEU", bleu), (sacrebleu, "CHRF", chrf), (sacrebleu, "TER", ter),
        (m, "BLEU", bleu), (m, "CHRF", chrf), (m, "TER", ter),
        (mb, "BLEU", bleu), (mc, "CHRF", chrf), (mt, "TER", ter),
        (compat, "BLEU", bleu), (compat, "CHRF", chrf), (compat, "TER", ter),
        (sys.modules[__name__], "BLEU", bleu),
        (sys.modules[__name__], "CHRF", chrf),
        (sys.modules[__name__], "TER", ter),
    ]
    for mod, name, new in targets:
        _PATCHED[(mod, name)] = getattr(mod, name)
        setattr(mod, name, new)
    from . import significance as _significance

    _PATCHED[("significance",)] = _significance.install()
    _PATCHED[("METRICS",)] = dict(m.METRICS)
    m.METRICS["BLEU"] = bleu
    m.METRICS["CHRF"] = chrf
    m.METRICS["TER"] = ter


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
    # Lazy: the compat wrappers need sacreBLEU, significance testing pulls in numpy.
    if name in _COMPAT_NAMES:
        return _compat_function(name)
    if name == "PairedTest":
        from .significance import PairedTest

        return PairedTest
    if name in ("significance", "rewards", "verify", "mcp_server"):
        import importlib

        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module 'swadesh' has no attribute {name!r}")

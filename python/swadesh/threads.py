"""``swadesh.n_jobs(k)``: how many threads the native kernels use.

The corpus kernels - tokenization, the BLEU/chrF/TER statistics - already run in parallel on a rayon
pool sized to the machine, and a score never depends on the thread count: the kernels partition the
work, they do not change the arithmetic. ``n_jobs`` sets that pool's size, for capping CPU use or
matching a scheduler's allocation. It is not sacreBLEU's ``PairedTest(n_jobs=)``, which spawns
processes for its resampling; this one controls the threads inside one process.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

from . import _swadesh


def set_n_jobs(n: Optional[int]) -> int:
    """Use ``n`` worker threads for the native kernels; ``None`` or ``0`` restores the machine default.

    Returns the number of threads now in use. A score is identical at any count.
    """
    return _swadesh.set_num_threads(0 if not n else int(n))


def get_n_jobs() -> int:
    """How many worker threads the native kernels are using."""
    return _swadesh.num_threads()


@contextmanager
def n_jobs(n: Optional[int]) -> Iterator[None]:
    """Run a block with ``n`` worker threads, restoring the previous count on exit."""
    previous = get_n_jobs()
    set_n_jobs(n)
    try:
        yield
    finally:
        set_n_jobs(previous)

"""Shared helpers for the differential fuzzers: profile-aware budgets and case counting.

Profiles: ``HYPOTHESIS_PROFILE=deep`` makes every fuzz test in the suite run
``DEEP_EXAMPLES`` examples (its own ``max_examples`` is ignored). The default
profile keeps each test's own budget, so the default run is unchanged.
"""
import os
from collections import Counter, defaultdict

from hypothesis import HealthCheck, settings

DEEP_EXAMPLES = int(os.environ.get("SWADESH_DEEP_EXAMPLES", "1000000"))  # override for benchmarks
SUPPRESS = list(HealthCheck)

# Cases executed in this process, keyed by "<metric> <configuration>", and the
# distinct seeds among them. conftest.py dumps both when SWADESH_CASE_DIR is set.
CASES = Counter()
SEEDS_SEEN = defaultdict(set)


def profile_name():
    return os.environ.get("HYPOTHESIS_PROFILE", "default")


def budget(default_examples, deep_examples=DEEP_EXAMPLES, **kw):
    """Hypothesis settings: ``default_examples``, or ``deep_examples`` under the deep profile."""
    n = deep_examples if profile_name() == "deep" else default_examples
    kw.setdefault("deadline", None)
    kw.setdefault("suppress_health_check", SUPPRESS)
    return settings(max_examples=n, **kw)


def count(key, seed=None):
    CASES[key] += 1
    if seed is not None:
        SEEDS_SEEN[key].add(seed)

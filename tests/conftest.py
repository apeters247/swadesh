"""Hypothesis profiles and per-process case counts for the differential suites.

Select a profile with ``HYPOTHESIS_PROFILE=default`` (the default, fast) or
``HYPOTHESIS_PROFILE=deep`` (1M examples per fuzz test; see fuzzsupport.budget).
Set ``SWADESH_CASE_DIR`` to a directory to dump per-process case counts at exit.
"""
import json
import os

from hypothesis import HealthCheck, settings

import fuzzsupport

settings.register_profile("default", deadline=None, suppress_health_check=list(HealthCheck))
settings.register_profile(
    "deep",
    max_examples=fuzzsupport.DEEP_EXAMPLES,
    deadline=None,
    suppress_health_check=list(HealthCheck),
    print_blob=True,  # a failure prints @reproduce_failure, so it can be replayed
)
settings.load_profile(fuzzsupport.profile_name())


def pytest_sessionfinish(session, exitstatus):
    out = os.environ.get("SWADESH_CASE_DIR")
    if out and fuzzsupport.CASES:
        data = {k: {"cases": v, "distinct_seeds": len(fuzzsupport.SEEDS_SEEN.get(k, ()))}
                for k, v in fuzzsupport.CASES.items()}
        with open(os.path.join(out, f"{os.getpid()}.json"), "w") as f:
            json.dump(data, f, sort_keys=True)

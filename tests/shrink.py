"""Reduce a failing differential case to a minimal repro.

Usage: ``python tests/shrink.py <failure.json>`` (written by test_deep.py when
SWADESH_FAIL_DIR is set). The reducer drops reference streams and segments, then
delta-debugs each string character by character, and finally replaces characters
with ``a`` where the mismatch survives. A candidate is kept only if the same
check still raises AssertionError. Prints the minimal case as JSON and as Python.

Not collected by pytest (no ``test_`` prefix).
"""
import copy
import json
import sys

from test_deep import bleu_case, chrf_case, ter_case

CHECKS = {"bleu": bleu_case, "chrf": chrf_case, "ter": ter_case, "ter-long": ter_case}


def fails(case):
    try:
        CHECKS[case["kind"]](case["kw"], case["hyps"], case["refs"])
    except AssertionError:
        return True
    return False


def _get(case, path):
    node = case
    for p in path:
        node = node[p]
    return node


def _set(case, path, value):
    node = case
    for p in path[:-1]:
        node = node[p]
    node[path[-1]] = value


def _ddmin_str(s, still_fails):
    """Delta-debug a string: remove chunks while the mismatch survives."""
    n = 2
    while len(s) >= 2:
        chunk = max(1, len(s) // n)
        reduced = False
        for i in range(0, len(s), chunk):
            cand = s[:i] + s[i + chunk:]
            if still_fails(cand):
                s, n, reduced = cand, max(n - 1, 2), True
                break
        if not reduced:
            if chunk == 1:
                break
            n = min(len(s), n * 2)
    return s


def _simplify_chars(s, still_fails):
    out = list(s)
    for i in range(len(out)):
        if out[i] != "a":
            trial = out[:i] + ["a"] + out[i + 1:]
            if still_fails("".join(trial)):
                out = trial
    return "".join(out)


def _string_paths(case):
    for i, s in enumerate(case["hyps"]):
        yield ("hyps", i)
    for k, stream in enumerate(case["refs"]):
        for i, s in enumerate(stream):
            if s is not None:
                yield ("refs", k, i)


def reduce_case(case):
    assert fails(case), "the input case does not fail"
    changed = True
    while changed:
        changed = False
        # Drop whole reference streams (stream 0 must stay).
        for k in range(len(case["refs"]) - 1, 0, -1):
            cand = copy.deepcopy(case)
            del cand["refs"][k]
            if fails(cand):
                case, changed = cand, True
        # Drop segments everywhere at once.
        for i in range(len(case["hyps"]) - 1, -1, -1):
            if len(case["hyps"]) <= 1:
                break
            cand = copy.deepcopy(case)
            del cand["hyps"][i]
            for stream in cand["refs"]:
                del stream[i]
            if fails(cand):
                case, changed = cand, True
        # Shrink each string.
        for path in list(_string_paths(case)):
            s = _get(case, path)
            if s is None:
                continue

            def still(t, path=path):
                cand = copy.deepcopy(case)
                _set(cand, path, t)
                return fails(cand)

            new = _ddmin_str(s, still)
            new = _simplify_chars(new, still)
            if new != s:
                _set(case, path, new)
                changed = True
    return case


def main():
    with open(sys.argv[1], encoding="utf-8") as f:
        case = json.load(f)
    small = reduce_case(case)
    print(json.dumps(small, ensure_ascii=False, indent=1))
    print("PYTHON:", repr(small))


if __name__ == "__main__":
    main()

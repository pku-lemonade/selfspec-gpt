import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from selfspec_stats import accept_counts_to_stats, validate_stats_payload


def _assert_eq(got, expected, msg=""):
    if got != expected:
        raise AssertionError(f"{msg}\nGOT: {got}\nEXPECTED: {expected}")


def _assert_raises(fn, expected_substr: str):
    try:
        fn()
    except Exception as e:
        if expected_substr not in str(e):
            raise AssertionError(f"Expected error containing {expected_substr!r}, got: {e!r}") from e
        return
    raise AssertionError(f"Expected exception containing {expected_substr!r}, but no exception was raised")


def main() -> None:
    stats = accept_counts_to_stats([1, 2, 3], k=2)
    _assert_eq(stats["k"], 2, "k")
    _assert_eq(stats["histogram"], {0: 1, 1: 2, 2: 3}, "histogram")
    validate_stats_payload(stats)

    _assert_raises(
        lambda: validate_stats_payload({"k": 2, "histogram": {"0": 1, "2": 1}}),
        "missing required bins",
    )
    _assert_raises(
        lambda: validate_stats_payload({"k": 1, "histogram": {"0": 0, "1": 0}}),
        "histogram sum must be > 0",
    )
    _assert_raises(
        lambda: validate_stats_payload({"k": 1, "histogram": {"0": 1, "2": 1}}),
        "out of range",
    )
    _assert_raises(
        lambda: validate_stats_payload({"k": 1, "histogram": {"0": -1, "1": 1}}),
        "non-negative",
    )

    print("OK")


if __name__ == "__main__":
    main()


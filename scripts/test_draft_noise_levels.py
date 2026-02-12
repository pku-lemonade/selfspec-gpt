import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from draft_noise import resolve_level_based_draft_noise_stds


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
    # Broadcast (1): one level for all buckets and all layers.
    per_layer_stds, output_std = resolve_level_based_draft_noise_stds(
        draft_noise_level_stds=[0.0, 0.1],
        draft_noise_levels=[1],
        n_layer=2,
    )
    _assert_eq(per_layer_stds, [(0.1, 0.1, 0.1), (0.1, 0.1, 0.1)], "broadcast(1) per_layer_stds")
    _assert_eq(output_std, 0.1, "broadcast(1) output_std")

    # Broadcast (3): per-bucket levels (FFN, QKV, OUT) broadcast to all layers.
    per_layer_stds, output_std = resolve_level_based_draft_noise_stds(
        draft_noise_level_stds=[0.0, 0.1, 0.2],
        draft_noise_levels=[1, 0, 2],
        n_layer=3,
    )
    _assert_eq(per_layer_stds, [(0.1, 0.0, 0.2)] * 3, "broadcast(3) per_layer_stds")
    _assert_eq(output_std, 0.2, "broadcast(3) output_std")

    # Per-layer (3*n_layer): per-layer triplets in FFN/QKV/OUT order.
    per_layer_stds, output_std = resolve_level_based_draft_noise_stds(
        draft_noise_level_stds=[0.0, 0.1, 0.2],
        draft_noise_levels=[1, 0, 0, 0, 2, 0],  # layer0 then layer1
        n_layer=2,
    )
    _assert_eq(per_layer_stds, [(0.1, 0.0, 0.0), (0.0, 0.2, 0.0)], "per-layer(3*n_layer) per_layer_stds")
    _assert_eq(output_std, 0.0, "per-layer(3*n_layer) output_std (last layer OUT)")

    # Out-of-range level index.
    _assert_raises(
        lambda: resolve_level_based_draft_noise_stds(
            draft_noise_level_stds=[0.0],
            draft_noise_levels=[1],
            n_layer=1,
        ),
        "Valid range is [0, 0]",
    )

    # Missing level stds.
    _assert_raises(
        lambda: resolve_level_based_draft_noise_stds(
            draft_noise_level_stds=None,  # type: ignore[arg-type]
            draft_noise_levels=[0],
            n_layer=1,
        ),
        "draft_noise_level_stds is required",
    )

    print("OK")


if __name__ == "__main__":
    main()

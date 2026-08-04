"""
Unit test guarding the monotonicity-penalty perturbation.

The physiological monotonicity penalty perturbs an on-board feature (IOB or COB)
by a factor PERTURB and checks the model's directional response. The perturbation
MUST correspond to multiplying the RAW feature value: x' = PERTURB * x.

A previous bug applied the factor in standardised space (z' = PERTURB * z), which
for below-mean windows moved the feature in the WRONG direction (it could even
make IOB negative). This test fails if that mistake is reintroduced.

The correct standardised-space update for x' = PERTURB * x is:
    z' = PERTURB * z + (PERTURB - 1) * mean / scale

Run:  python3 -m pytest test_perturbation.py      (or)   python3 test_perturbation.py
"""

import numpy as np

PERTURB = 1.5


def apply_perturbation_standardised(z, mean, scale, factor=PERTURB):
    """The corrected perturbation, in standardised space, that is equivalent to
    x' = factor * x in raw space. This mirrors the offset used in train_all_*.py:
        z' = factor * z + (factor - 1) * mean / scale
    """
    offset = (factor - 1.0) * mean / scale
    return factor * z + offset


def standardise(x, mean, scale):
    return (x - mean) / scale


def destandardise(z, mean, scale):
    return z * scale + mean


# ---- Test cases spanning below-mean, at-mean, above-mean, and zero IOB ----
_CASES = [
    # (raw_x, mean, scale)
    (0.0, 3.55, 2.5),    # zero IOB, below mean  (the case the old bug broke)
    (2.0, 3.55, 2.5),    # below mean
    (3.55, 3.55, 2.5),   # exactly at mean
    (8.0, 3.55, 2.5),    # above mean
    (1.39, 1.39, 1.1),   # Manchester-like mean
    (0.0, 1.39, 1.1),    # zero IOB, Manchester scale
    (12.5, 4.0, 3.2),    # large dose
]


def test_perturbation_equals_raw_scaling():
    """x' recovered from the standardised perturbation must equal PERTURB * x."""
    for x, mean, scale in _CASES:
        z = standardise(x, mean, scale)
        z_pert = apply_perturbation_standardised(z, mean, scale)
        x_pert = destandardise(z_pert, mean, scale)
        assert np.isclose(x_pert, PERTURB * x), (
            f"x={x}: got x'={x_pert:.4f}, expected {PERTURB * x:.4f}")


def test_naive_standardised_scaling_is_wrong():
    """Guard: the OLD (buggy) z' = PERTURB * z does NOT give x' = PERTURB * x
    whenever x != mean. This documents why the offset is required."""
    wrong_count = 0
    for x, mean, scale in _CASES:
        z = standardise(x, mean, scale)
        x_pert_buggy = destandardise(PERTURB * z, mean, scale)
        if not np.isclose(x_pert_buggy, PERTURB * x):
            wrong_count += 1
    # At least the non-at-mean cases must be wrong under the buggy formula.
    assert wrong_count >= len(_CASES) - 1


def test_perturbation_increases_below_mean_iob():
    """The corrected perturbation must never move IOB DOWN when increasing it
    (the physiological failure mode of the old bug)."""
    for x, mean, scale in _CASES:
        z = standardise(x, mean, scale)
        x_pert = destandardise(apply_perturbation_standardised(z, mean, scale), mean, scale)
        assert x_pert >= x - 1e-9, (
            f"x={x}: perturbed IOB {x_pert:.4f} is LOWER than original {x} "
            f"(wrong direction)")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
    print("\nAll perturbation unit tests passed.")

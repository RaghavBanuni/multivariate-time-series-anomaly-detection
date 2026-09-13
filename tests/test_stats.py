"""Rolling and robust statistics."""

import math
import random

import pytest

from msad.stats import (
    MAD_TO_SIGMA,
    EWMA,
    RollingStats,
    Welford,
    mad,
    mean,
    median,
    pearson,
    quantile,
    robust_zscore,
    std,
    variance,
)


# -- Welford ---------------------------------------------------------------------------------


def test_welford_matches_the_two_pass_reference():
    rng = random.Random(1)
    values = [rng.gauss(340.0, 0.35) for _ in range(2000)]
    reference = variance(values)
    assert Welford().extend(values).variance == pytest.approx(reference, rel=1e-12)


def test_welford_is_exact_on_a_hand_checkable_case():
    # values 1..5: mean 3, sum of squared deviations 10, sample variance 10/4
    welford = Welford().extend([1.0, 2.0, 3.0, 4.0, 5.0])
    assert welford.mean == 3.0
    assert welford.variance == 2.5
    assert welford.population_variance == 2.0


def test_welford_survives_a_large_offset():
    """Unit spread on a 1e8 pedestal. The condition number of this problem is ~1e8, so a few digits
    go regardless of the algorithm; the point is that Welford keeps the rest, while ``E[x^2]-E[x]^2``
    is left with almost nothing.
    """
    rng = random.Random(2)
    spread = [rng.gauss(0.0, 1.0) for _ in range(500)]
    offset = [value + 1e8 for value in spread]
    assert Welford().extend(offset).variance == pytest.approx(variance(spread), rel=1e-6)


def test_variance_of_fewer_than_two_samples_is_zero_not_an_exception():
    """A warm-up window legitimately hits this on its first sample."""
    assert Welford().update(5.0).variance == 0.0
    assert Welford().variance == 0.0
    assert variance([7.0]) == 0.0


def test_std_is_never_nan_from_a_negative_rounding_artefact():
    welford = Welford().extend([2.0] * 50)
    assert welford.variance >= 0.0
    assert welford.std == 0.0


# -- quantiles and medians -------------------------------------------------------------------


def test_quantile_interpolates_between_order_statistics():
    # position = 0.25 * (5 - 1) = 1.0 exactly -> the second order statistic
    assert quantile([1.0, 2.0, 3.0, 4.0, 5.0], 0.25) == 2.0
    # position = 0.3 * 4 = 1.2 -> 0.8 * 2 + 0.2 * 3
    assert quantile([1.0, 2.0, 3.0, 4.0, 5.0], 0.3) == pytest.approx(2.2)


def test_quantile_endpoints_and_singletons():
    assert quantile([3.0, 1.0, 2.0], 0.0) == 1.0
    assert quantile([3.0, 1.0, 2.0], 1.0) == 3.0
    assert quantile([9.0], 0.42) == 9.0


def test_median_of_an_even_sample_averages_the_middle_pair():
    assert median([1.0, 2.0, 3.0, 10.0]) == 2.5


def test_quantile_rejects_nonsense():
    with pytest.raises(ValueError, match="empty"):
        quantile([], 0.5)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        quantile([1.0], 1.5)
    with pytest.raises(ValueError, match="empty"):
        mean([])


# -- robust scale ----------------------------------------------------------------------------


def test_the_mad_consistency_constant_recovers_sigma():
    """Without the 1.4826 factor every threshold expressed in sigmas is 48% too tight."""
    assert MAD_TO_SIGMA == pytest.approx(1.482602, rel=1e-6)
    rng = random.Random(3)
    sample = [rng.gauss(0.0, 3.0) for _ in range(200000)]
    assert mad(sample) == pytest.approx(3.0, rel=0.02)
    assert mad(sample, scaled=False) == pytest.approx(3.0 * 0.6744897, rel=0.02)


def test_the_mad_ignores_a_contaminated_minority():
    """The reason a robust baseline is used at all: the fault sits inside its own window."""
    clean = [10.0, 10.5, 9.5, 10.2, 9.8, 10.1, 9.9]
    contaminated = clean + [400.0, 500.0]
    assert std(contaminated) > 50.0
    assert mad(contaminated) < 2.0


def test_a_zero_scale_reports_infinity_rather_than_inventing_a_scale():
    """Half the window being one repeated number means a stuck sensor, not a tiny sigma."""
    assert robust_zscore(5.0, 5.0, 0.0) == 0.0
    assert robust_zscore(6.0, 5.0, 0.0) == math.inf
    assert robust_zscore(4.0, 5.0, 0.0) == -math.inf


def test_a_scale_floor_keeps_a_flat_window_from_flooding_the_alarm_list():
    assert robust_zscore(6.0, 5.0, 0.0, scale_floor=0.5) == 2.0
    assert robust_zscore(6.0, 5.0, 2.0, scale_floor=0.5) == 0.5  # the real scale still wins


# -- rolling window --------------------------------------------------------------------------


def test_the_rolling_window_forgets_in_order():
    rolling = RollingStats(size=3)
    for value in (1.0, 2.0, 3.0, 4.0):
        rolling.push(value)
    assert rolling.values() == [2.0, 3.0, 4.0]
    assert rolling.full
    assert rolling.mean() == 3.0
    assert rolling.median() == 3.0


def test_a_window_of_one_has_no_spread_and_is_refused():
    with pytest.raises(ValueError, match="at least two"):
        RollingStats(size=1)


# -- EWMA ------------------------------------------------------------------------------------


def test_the_ewma_is_debiased_from_the_first_sample():
    """Zero-initialised without the correction, the first estimate would be alpha * x."""
    ewma = EWMA(alpha=0.1).update(7.0)
    assert ewma.mean == pytest.approx(7.0)
    assert ewma.count == 1


def test_the_ewma_of_a_constant_series_is_that_constant():
    ewma = EWMA(alpha=0.05)
    for _ in range(200):
        ewma.update(4.0)
    assert ewma.mean == pytest.approx(4.0, rel=1e-9)
    assert ewma.variance == pytest.approx(0.0, abs=1e-12)


def test_the_ewma_tracks_a_step_and_lags_it():
    ewma = EWMA(alpha=0.2)
    for _ in range(100):
        ewma.update(0.0)
    for _ in range(3):
        ewma.update(10.0)
    assert 0.0 < ewma.mean < 10.0, "a lagging estimator must not jump to the new level"
    assert ewma.variance > 0.0, "the step must inflate the variance estimate"


def test_an_alpha_outside_the_unit_interval_is_refused():
    with pytest.raises(ValueError, match="alpha"):
        EWMA(alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        EWMA(alpha=1.5)


# -- correlation -----------------------------------------------------------------------------


def test_correlation_of_a_perfect_line_is_one():
    assert pearson([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)
    assert pearson([1.0, 2.0, 3.0], [-2.0, -4.0, -6.0]) == pytest.approx(-1.0)


def test_correlation_with_a_constant_channel_is_zero_not_nan():
    """A NaN here silently poisons every downstream comparison."""
    value = pearson([1.0, 2.0, 3.0], [5.0, 5.0, 5.0])
    assert value == 0.0
    assert value == value


def test_correlation_is_shift_and_scale_invariant():
    left = [1.0, 4.0, 2.0, 8.0, 5.0]
    right = [2.0, 3.0, 2.5, 6.0, 4.0]
    base = pearson(left, right)
    shifted = pearson([value + 1e6 for value in left], [value * 3.0 + 7.0 for value in right])
    assert shifted == pytest.approx(base, rel=1e-6)


def test_correlation_of_mismatched_lengths_is_an_error():
    with pytest.raises(ValueError, match="different lengths"):
        pearson([1.0, 2.0], [1.0])

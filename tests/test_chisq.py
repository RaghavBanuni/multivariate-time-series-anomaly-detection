"""Chi-square machinery against published tables.

The point of these tests is that ``theory_threshold`` means something. A Mahalanobis detector that
claims "alarm at a per-sample false-alarm rate of 1e-3" is only telling the truth if this file passes.
"""

import pytest

from msad.chisq import chi2_cdf, chi2_ppf, chi2_sf, gammainc_lower

# Standard tabulated critical values.
TABLE = [
    (0.95, 1, 3.841459),
    (0.99, 1, 6.634897),
    (0.95, 2, 5.991465),
    (0.99, 2, 9.210340),
    (0.95, 5, 11.070498),
    (0.999, 6, 22.457744),
    (0.99, 10, 23.209251),
    (0.95, 30, 43.772972),
]


@pytest.mark.parametrize("level,df,expected", TABLE)
def test_quantiles_match_the_published_table(level, df, expected):
    assert chi2_ppf(level, df) == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize("level,df,expected", TABLE)
def test_the_cdf_inverts_the_quantile(level, df, expected):
    assert chi2_cdf(expected, df) == pytest.approx(level, abs=1e-6)


def test_the_cdf_is_monotone_and_bounded():
    previous = -1.0
    for step in range(200):
        value = chi2_cdf(step * 0.25, 4)
        assert 0.0 <= value <= 1.0
        assert value >= previous
        previous = value


def test_the_two_expansions_agree_where_they_meet():
    """The series is used below ``a + 1`` and the continued fraction above it. A jump at the switch
    would mean one of them is being evaluated outside its region of good convergence."""
    a = 3.0
    left = gammainc_lower(a, a + 1.0 - 1e-9)
    right = gammainc_lower(a, a + 1.0 + 1e-9)
    assert left == pytest.approx(right, abs=1e-8)


def test_the_median_of_a_chi_square_with_one_degree_of_freedom():
    # median of chi^2_1 = (Phi^{-1}(0.75))^2 = 0.6744897^2
    assert chi2_ppf(0.5, 1) == pytest.approx(0.4549364, abs=1e-6)


def test_the_mean_is_the_degrees_of_freedom():
    """E[chi^2_k] = k, and the distribution is right-skewed, so the CDF at k sits just above 0.5."""
    assert 0.5 < chi2_cdf(6, 6) < 0.65


def test_survival_and_cdf_sum_to_one():
    for value in (0.5, 5.0, 25.0):
        assert chi2_cdf(value, 4) + chi2_sf(value, 4) == pytest.approx(1.0)


def test_extreme_tails_stay_in_range():
    assert chi2_cdf(1e6, 6) == pytest.approx(1.0)
    assert chi2_sf(1e6, 6) == pytest.approx(0.0, abs=1e-12)
    assert chi2_cdf(0.0, 6) == 0.0
    assert chi2_cdf(-3.0, 6) == 0.0


def test_a_higher_confidence_needs_a_larger_threshold():
    values = [chi2_ppf(level, 6) for level in (0.9, 0.99, 0.999, 0.9999)]
    assert values == sorted(values)


def test_more_channels_need_a_larger_threshold_at_the_same_confidence():
    """Six tags monitored jointly tolerate a larger squared distance than two, at equal risk."""
    values = [chi2_ppf(0.999, df) for df in (1, 2, 6, 12)]
    assert values == sorted(values)


def test_degenerate_arguments_are_refused():
    assert chi2_ppf(0.0, 3) == 0.0
    with pytest.raises(ValueError, match="p must be"):
        chi2_ppf(1.0, 3)
    with pytest.raises(ValueError, match="degrees of freedom"):
        chi2_ppf(0.5, 0)
    with pytest.raises(ValueError, match="a must be positive"):
        gammainc_lower(0.0, 1.0)
    with pytest.raises(ValueError, match="non-negative"):
        gammainc_lower(1.0, -1.0)

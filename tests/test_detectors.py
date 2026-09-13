"""The four detectors: contracts, guards, and the properties each one is supposed to have."""

import math
import random

import pytest

from msad.chisq import chi2_ppf
from msad.detectors import (
    EULER_GAMMA,
    IsolationDepthDetector,
    MahalanobisDetector,
    PCAReconstructionDetector,
    RobustZScoreDetector,
    harmonic_path_correction,
    training_quantile_threshold,
)
from msad.stats import quantile


def correlated_rows(n: int = 600, noise: float = 0.05, seed: int = 1) -> list[list[float]]:
    """Two channels on a line, plus a third that is a scaled copy with its own noise."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        base = rng.gauss(0.0, 1.0)
        rows.append([base, base + rng.gauss(0.0, noise), 3.0 * base + rng.gauss(0.0, noise)])
    return rows


def every_detector():
    return [
        RobustZScoreDetector(window=100),
        MahalanobisDetector(),
        PCAReconstructionDetector(),
        IsolationDepthDetector(n_trees=25, subsample=128),
    ]


# -- shared contract -------------------------------------------------------------------------


@pytest.mark.parametrize("detector", every_detector(), ids=lambda d: d.name)
def test_every_detector_fits_and_returns_one_finite_score_per_sample(detector):
    rows = correlated_rows()
    assert detector.fit(rows) is detector, "fit returns self so calls can be chained"
    series = detector.score(rows)
    assert series.name == detector.name
    assert len(series) == len(rows)
    assert all(math.isfinite(score) for score in series.scores)
    assert all(score >= 0.0 for score in series.scores)
    assert 0 <= series.valid_from < len(rows)


@pytest.mark.parametrize("detector", every_detector(), ids=lambda d: d.name)
def test_scoring_before_fitting_is_refused(detector):
    with pytest.raises(ValueError, match="fit before scoring"):
        detector.score(correlated_rows(20))


@pytest.mark.parametrize("detector", every_detector(), ids=lambda d: d.name)
def test_a_clear_outlier_outscores_the_centre(detector):
    """The weakest possible sanity property, asserted for all four so none can be silently broken."""
    rows = correlated_rows()
    detector.fit(rows)
    probe = rows[:200] + [[0.0, 0.0, 0.0], [4.0, -4.0, 12.0]]
    scores = detector.score(probe).scores
    assert scores[-1] > scores[-2]


# -- robust z --------------------------------------------------------------------------------


def test_the_robust_detector_declares_its_warm_up():
    """Scores before the rolling window has filled are not comparable and must not be thresholded."""
    detector = RobustZScoreDetector(window=100).fit(correlated_rows())
    series = detector.score(correlated_rows())
    assert series.valid_from == 100


def test_the_robust_detector_finds_a_single_channel_spike():
    rows = correlated_rows()
    detector = RobustZScoreDetector(window=100).fit(rows)
    probe = [row[:] for row in rows]
    probe[400][0] += 25.0
    scores = detector.score(probe).scores
    assert scores[400] > 10.0
    assert scores[400] > 5 * scores[399]


def test_the_robust_detector_needs_a_training_window():
    with pytest.raises(ValueError, match="training window"):
        RobustZScoreDetector().fit([[1.0, 2.0]])


# -- Mahalanobis -----------------------------------------------------------------------------


def test_the_distance_at_the_training_centre_is_zero():
    rows = correlated_rows()
    detector = MahalanobisDetector().fit(rows)
    centre = [
        math.fsum(row[column] for row in rows) / len(rows) for column in range(len(rows[0]))
    ]
    assert detector.score([centre]).scores[0] == pytest.approx(0.0, abs=1e-6)


def test_the_theoretical_threshold_is_the_chi_square_quantile():
    """This is the claim that makes the threshold interpretable, so it is pinned to the table."""
    detector = MahalanobisDetector().fit(correlated_rows())
    assert detector.degrees_of_freedom == 3
    assert detector.theory_threshold(1e-3) == pytest.approx(chi2_ppf(0.999, 3), rel=1e-9)
    assert detector.theory_threshold(0.05) == pytest.approx(chi2_ppf(0.95, 3), rel=1e-9)
    assert detector.theory_threshold(1e-4) > detector.theory_threshold(1e-3)


def test_shrinkage_and_jitter_are_reported_rather_than_hidden():
    detector = MahalanobisDetector().fit(correlated_rows(noise=1e-9))
    assert 0.0 <= detector.intensity <= 1.0
    assert detector.jitter >= 0.0


def test_the_distance_sees_a_broken_correlation_that_no_channel_range_would():
    """The core claim of the whole repository, on a two-channel toy version of the pump.

    Both channels stay inside their observed ranges; only the *relationship* is wrong.
    """
    rows = correlated_rows(noise=0.05)
    detector = MahalanobisDetector().fit(rows)
    low = min(row[0] for row in rows)
    high = max(row[0] for row in rows)

    in_range_but_decoupled = [high * 0.6, low * 0.6, 0.0]
    assert all(low <= value <= high for value in in_range_but_decoupled[:2])

    scores = detector.score([[0.0, 0.0, 0.0], in_range_but_decoupled]).scores
    assert scores[1] > 50 * max(scores[0], 1e-9)
    assert scores[1] > detector.theory_threshold(1e-3)


def test_fewer_training_samples_than_channels_is_refused():
    with pytest.raises(ValueError, match="fewer training samples than channels"):
        MahalanobisDetector().fit([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])


# -- PCA reconstruction ----------------------------------------------------------------------


def test_the_pca_detector_retains_the_components_it_needs_and_says_so():
    detector = PCAReconstructionDetector(variance_target=0.95).fit(correlated_rows())
    assert 1 <= detector.retained < 3, "a full-rank model would have no residual to measure"
    assert detector.explained >= 0.95


def test_the_reconstruction_error_measures_distance_from_the_structure():
    detector = PCAReconstructionDetector().fit(correlated_rows(noise=0.05))
    along_spe, _ = detector.statistics([1.0, 1.0, 3.0])  # on the line
    across_spe, _ = detector.statistics([1.0, -1.0, 3.0])  # same magnitudes, wrong relationship
    assert across_spe > 100 * along_spe


def test_the_hotelling_statistic_grows_along_the_structure():
    """SPE and T-squared answer different questions. A large excursion *along* the model is a T2
    event with almost no reconstruction error, which is why both are returned."""
    detector = PCAReconstructionDetector().fit(correlated_rows(noise=0.05))
    near_spe, near_t2 = detector.statistics([0.1, 0.1, 0.3])
    far_spe, far_t2 = detector.statistics([6.0, 6.0, 18.0])
    assert far_t2 > 100 * near_t2
    assert far_spe < 1.0, "an excursion along the principal direction is invisible to SPE"


def test_an_impossible_variance_target_is_refused():
    with pytest.raises(ValueError, match="variance_target"):
        PCAReconstructionDetector(variance_target=1.5).fit(correlated_rows())
    with pytest.raises(ValueError, match="variance_target"):
        PCAReconstructionDetector(variance_target=0.0).fit(correlated_rows())


def test_degenerate_training_data_is_refused():
    with pytest.raises(ValueError, match="zero total variance"):
        PCAReconstructionDetector().fit([[2.0, 2.0]] * 50)


# -- isolation depth -------------------------------------------------------------------------


def test_the_forest_is_deterministic_for_a_given_seed():
    """A monitoring system that reports different scores on a rerun cannot be audited."""
    rows = correlated_rows(300)
    first = IsolationDepthDetector(n_trees=20, seed=7).fit(rows).score(rows).scores
    second = IsolationDepthDetector(n_trees=20, seed=7).fit(rows).score(rows).scores
    assert first == second


def test_a_different_seed_gives_a_different_forest():
    rows = correlated_rows(300)
    first = IsolationDepthDetector(n_trees=20, seed=7).fit(rows).score(rows).scores
    other = IsolationDepthDetector(n_trees=20, seed=8).fit(rows).score(rows).scores
    assert first != other


def test_isolation_scores_are_bounded_probabilities():
    rows = correlated_rows(300)
    scores = IsolationDepthDetector(n_trees=20).fit(rows).score(rows).scores
    assert all(0.0 < score <= 1.0 for score in scores)


def test_the_path_length_correction_matches_its_definition():
    """``c(n) = 2 H_{n-1} - 2(n-1)/n``, with ``H_m`` taken as ``ln m + gamma`` for n >= 3.

    Without this normaliser a point sitting in a leaf that still holds thirty samples would be scored
    as though it had been isolated, and depth-limited trees would systematically under-score dense
    regions.
    """
    assert harmonic_path_correction(1) == 0.0, "a single-sample leaf needs no correction"
    assert harmonic_path_correction(2) == 1.0
    assert harmonic_path_correction(3) == pytest.approx(
        2.0 * (math.log(2.0) + EULER_GAMMA) - 2.0 * 2 / 3
    )
    assert harmonic_path_correction(256) == pytest.approx(10.244771, abs=1e-5)
    values = [harmonic_path_correction(n) for n in (2, 3, 10, 100, 1000)]
    assert values == sorted(values), "larger leaves imply longer expected paths"


def test_the_asymptotic_harmonic_is_close_to_the_exact_sum():
    """``ln m + gamma`` drops a ``1/(2m)`` term. At the subsample sizes used here that is a 0.05%
    difference, which is worth knowing about and not worth an exact harmonic sum per leaf."""
    exact = 2.0 * math.fsum(1.0 / k for k in range(1, 256)) - 2.0 * 255 / 256
    assert harmonic_path_correction(256) == pytest.approx(exact, rel=1e-3)
    assert harmonic_path_correction(256) < exact, "the approximation is a slight under-estimate"


def test_a_handful_of_rows_cannot_train_a_forest():
    with pytest.raises(ValueError, match="handful"):
        IsolationDepthDetector().fit([[1.0], [2.0]])


# -- calibration helper ----------------------------------------------------------------------


def test_the_training_threshold_is_the_quantile_of_the_valid_training_scores():
    rows = correlated_rows()
    detector = RobustZScoreDetector(window=100).fit(rows)
    series = detector.score(rows)
    value = training_quantile_threshold(series, 400, 0.99)
    assert value == pytest.approx(quantile(series.scores[100:400], 0.99))


def test_calibrating_inside_the_warm_up_is_refused_rather_than_guessed():
    rows = correlated_rows()
    detector = RobustZScoreDetector(window=500).fit(rows)
    series = detector.score(rows)
    with pytest.raises(ValueError, match="no valid training scores"):
        training_quantile_threshold(series, 100, 0.999)

"""Thresholding, hysteresis and dwell time — the layer that decides whether anyone keeps the
system switched on.
"""

import random

import pytest

from msad.threshold import (
    AdaptiveQuantileThreshold,
    Alarm,
    HysteresisPolicy,
    StaticQuantileThreshold,
    alarms_to_mask,
    exceedance_mask,
)

FLAT = [1.0] * 60


def constant(value: float, length: int) -> list[float]:
    return [value] * length


# -- static threshold ------------------------------------------------------------------------


def test_the_static_threshold_is_calibrated_once_and_never_moves():
    scores = [float(value) for value in range(1000)]
    threshold = StaticQuantileThreshold(level=0.99).fit(scores)
    assert threshold.value == pytest.approx(989.01)
    series = threshold.series(scores)
    assert len(series) == len(scores)
    assert len(set(series)) == 1


def test_calibrating_on_nothing_is_refused():
    with pytest.raises(ValueError, match="no training scores"):
        StaticQuantileThreshold().fit([])


# -- adaptive threshold ----------------------------------------------------------------------


def test_the_adaptive_threshold_follows_a_rising_baseline():
    """The intended behaviour: a plant whose noise floor creeps up does not drown the operator."""
    scores = [0.01 * index for index in range(1500)]
    series = AdaptiveQuantileThreshold(window=200, exclude_alarms=False).series(scores)
    assert series[-1] > series[300] > 0.0


def test_the_adaptive_threshold_never_falls_below_its_floor():
    """Otherwise a very quiet period makes the detector more sensitive than its calibration."""
    series = AdaptiveQuantileThreshold(window=50, floor=5.0).series(constant(0.1, 400))
    assert min(series) >= 5.0


def test_the_adaptive_baseline_bootstraps_itself():
    """Regression guard: rejecting "alarming" samples before any baseline exists rejects every
    sample, and the threshold then sits at its floor forever."""
    scores = constant(3.0, 400)
    series = AdaptiveQuantileThreshold(window=100, floor=0.0, exclude_alarms=True).series(scores)
    assert series[-1] == pytest.approx(3.0), "the baseline must fill from the early samples"


def test_an_ongoing_alarm_does_not_raise_the_threshold_that_judges_it():
    """Without ``exclude_alarms`` the fault feeds its own scores into the baseline, the threshold
    climbs to meet them, and the alarm clears itself while the fault is still running."""
    scores = constant(1.0, 400) + constant(50.0, 400)
    naive = AdaptiveQuantileThreshold(window=100, floor=2.0, exclude_alarms=False).series(scores)
    careful = AdaptiveQuantileThreshold(window=100, floor=2.0, exclude_alarms=True).series(scores)
    assert naive[-1] > 40.0, "the self-feeding baseline swallows the fault"
    assert careful[-1] == pytest.approx(2.0), "exclusion keeps the alarm standing"


def test_an_adaptive_threshold_absorbs_a_slow_drift_that_a_static_one_catches():
    """The trade-off, as a property rather than an anecdote.

    A drift of 0.0005 per sample is invisible inside a 200-sample window (0.1 of movement against a
    noise spread of ~0.25) but reaches twelve standard deviations over the full series. The static
    threshold fires for thousands of samples; the adaptive one barely notices. Both behaviours are
    correct, which is exactly why a deployment has to choose deliberately.
    """
    rng = random.Random(17)
    noise = [rng.gauss(1.0, 0.25) for _ in range(6000)]
    scores = [value + 0.0005 * index for index, value in enumerate(noise)]

    static = StaticQuantileThreshold(level=0.999).fit(scores[:600]).series(scores)
    adaptive = AdaptiveQuantileThreshold(
        window=200, level=0.999, exclude_alarms=False
    ).series(scores)

    static_hits = sum(exceedance_mask(scores, static, valid_from=600))
    adaptive_hits = sum(exceedance_mask(scores, adaptive, valid_from=600))
    assert static_hits > 1000, "a fixed line is crossed permanently once the baseline moves past it"
    assert adaptive_hits < 200, "a moving line follows the drift instead of reporting it"
    assert static_hits > 10 * adaptive_hits


def test_samples_before_the_warm_up_get_the_floor():
    series = AdaptiveQuantileThreshold(window=50, floor=3.0).series(FLAT, valid_from=10)
    assert series[:10] == [3.0] * 10


# -- hysteresis ------------------------------------------------------------------------------


def test_a_short_excursion_does_not_become_an_alarm():
    scores = constant(0.0, 10) + constant(10.0, 2) + constant(0.0, 30)
    alarms = HysteresisPolicy(min_duration=3).apply(scores, constant(1.0, len(scores)))
    assert alarms == []


def test_a_sustained_excursion_becomes_exactly_one_alarm():
    scores = constant(0.0, 10) + constant(10.0, 20) + constant(0.0, 20)
    alarms = HysteresisPolicy(min_duration=3, exit_ratio=0.7, clear_duration=6).apply(
        scores, constant(1.0, len(scores))
    )
    assert len(alarms) == 1
    alarm = alarms[0]
    assert alarm.start == 10, "the alarm is dated from the first exceedance, not from the raise"
    assert alarm.end == 35, "and closes once it has been clear for clear_duration samples"
    assert alarm.peak == 10.0
    assert len(alarm) == 26


def test_an_alarm_beginning_at_index_zero_keeps_its_start():
    """Regression guard: ``run_start or index`` would silently discard a legitimate index 0."""
    scores = constant(10.0, 20) + constant(0.0, 20)
    alarms = HysteresisPolicy(min_duration=3, clear_duration=6).apply(
        scores, constant(1.0, len(scores))
    )
    assert len(alarms) == 1
    assert alarms[0].start == 0


def test_the_hysteresis_band_keeps_a_hovering_score_from_clearing():
    """0.8 is below the threshold but inside the band, so the alarm neither clears nor re-raises."""
    scores = constant(10.0, 10) + constant(0.8, 40)
    alarms = HysteresisPolicy(min_duration=3, exit_ratio=0.7, clear_duration=6).apply(
        scores, constant(1.0, len(scores))
    )
    assert len(alarms) == 1
    assert alarms[0].end == len(scores) - 1, "an alarm still standing at the end is reported"


def test_debouncing_collapses_chatter_into_nothing_or_into_one_alarm():
    """A score oscillating across the threshold is the classic alarm flood."""
    scores = [10.0 if index % 2 == 0 else 0.0 for index in range(60)]
    thresholds = constant(1.0, len(scores))
    raw = HysteresisPolicy(min_duration=1, exit_ratio=1.0, clear_duration=1).apply(
        scores, thresholds
    )
    debounced = HysteresisPolicy(min_duration=3, exit_ratio=0.7, clear_duration=6).apply(
        scores, thresholds
    )
    assert len(raw) > 10, "per-sample alarming is what floods the operator"
    assert debounced == [], "three consecutive exceedances are never available here"


def test_a_cooldown_suppresses_an_immediate_re_raise():
    scores = constant(10.0, 10) + constant(0.0, 10) + constant(10.0, 10) + constant(0.0, 20)
    thresholds = constant(1.0, len(scores))
    without = HysteresisPolicy(min_duration=3, clear_duration=6, cooldown=0).apply(
        scores, thresholds
    )
    with_cooldown = HysteresisPolicy(min_duration=3, clear_duration=6, cooldown=40).apply(
        scores, thresholds
    )
    assert len(without) == 2
    assert len(with_cooldown) == 1


def test_scores_before_the_warm_up_cannot_raise_an_alarm():
    scores = constant(10.0, 40)
    alarms = HysteresisPolicy(min_duration=3, clear_duration=6).apply(
        scores, constant(1.0, 40), valid_from=20
    )
    assert len(alarms) == 1
    assert alarms[0].start == 20


def test_nonsense_policies_are_refused():
    with pytest.raises(ValueError, match="min_duration"):
        HysteresisPolicy(min_duration=0)
    with pytest.raises(ValueError, match="exit_ratio"):
        HysteresisPolicy(exit_ratio=0.0)
    with pytest.raises(ValueError, match="exit_ratio"):
        HysteresisPolicy(exit_ratio=1.5)


# -- masks -----------------------------------------------------------------------------------


def test_the_exceedance_mask_respects_the_warm_up():
    scores = constant(10.0, 10)
    assert exceedance_mask(scores, constant(1.0, 10), valid_from=4) == [0] * 4 + [1] * 6


def test_alarms_convert_to_a_mask_and_are_clipped_to_the_series():
    mask = alarms_to_mask([Alarm(start=2, end=4, peak=1.0), Alarm(start=8, end=99, peak=1.0)], 10)
    assert mask == [0, 0, 1, 1, 1, 0, 0, 0, 1, 1]


def test_an_alarm_knows_its_own_extent():
    alarm = Alarm(start=5, end=9, peak=3.0)
    assert len(alarm) == 5
    assert alarm.overlaps(9, 20)
    assert alarm.overlaps(0, 5)
    assert not alarm.overlaps(10, 20)
    assert not alarm.overlaps(0, 4)

"""The benchmark claims, asserted rather than asserted-in-prose.

This file is the one that would fail if the central argument of the repository were wrong: that a fault
can hide inside every individual channel's normal range, that the multivariate detectors see it, and
that a per-channel robust baseline does not.

It is the slowest file in the suite (it fits four detectors on ten days of six-channel data, and the
last test runs every CLI demonstration end to end) — expect a couple of minutes.
"""

import random
from functools import lru_cache

import pytest

from msad import simulate
from msad.cli import COMMANDS, main
from msad.detectors import training_quantile_threshold
from msad.evaluate import point_adjusted_f1, point_wise_f1, score_separation
from msad.pipeline import default_detectors, run_detector
from msad.simulate import INDEX, SAMPLES_PER_DAY


@lru_cache(maxsize=1)
def dataset():
    return simulate.build_dataset()


@lru_cache(maxsize=None)
def scored(name: str):
    """Fit on the clean window only, score everything, calibrate on the training scores."""
    data = dataset()
    detector = next(candidate for candidate in default_detectors() if candidate.name == name)
    detector.fit(data.train())
    series = detector.score(data.rows)
    return series, training_quantile_threshold(series, data.train_end, 0.999)


def ratio(name: str, kind: str) -> float:
    """Median score inside a fault, as a multiple of that detector's own alarm threshold.

    Threshold-free and hysteresis-free: it answers "can this detector see this fault at all".
    """
    data = dataset()
    event = next(candidate for candidate in data.events if candidate.kind == kind)
    series, threshold = scored(name)
    inside, _ = score_separation(series.scores, event, threshold)
    return inside / threshold


def event_of(kind: str):
    return next(candidate for candidate in dataset().events if candidate.kind == kind)


# -- the simulated plant ---------------------------------------------------------------------


def test_the_benchmark_is_reproducible():
    """A benchmark that moves between runs cannot support a comparison."""
    assert simulate.build_dataset().rows == simulate.build_dataset().rows


def test_the_shape_of_the_benchmark_is_what_the_readme_claims():
    data = dataset()
    assert len(data) == 10 * SAMPLES_PER_DAY == 2880
    assert data.width == 6
    assert data.train_end == 4 * SAMPLES_PER_DAY
    assert len(data.events) == 5
    assert {event.kind for event in data.events} == {
        "spike",
        "level_shift",
        "correlation_break",
        "slow_drift",
        "stuck_sensor",
    }


def test_the_training_window_is_clean():
    """Fitting on contaminated data teaches the detector that the fault is normal."""
    data = dataset()
    assert all(event.start >= data.train_end for event in data.events)
    assert sum(data.labels()[: data.train_end]) == 0


def test_the_faults_do_not_overlap_and_are_in_order():
    events = dataset().events
    for earlier, later in zip(events, events[1:]):
        assert earlier.end < later.start


def test_a_fault_free_dataset_has_no_labels():
    clean = simulate.build_dataset(with_faults=False)
    assert clean.events == []
    assert sum(clean.labels()) == 0


# -- the structural claim --------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind,sensor", [("correlation_break", "discharge_pressure"), ("stuck_sensor", "flow")]
)
def test_the_hidden_faults_never_leave_their_own_normal_range(kind, sensor):
    """The reason univariate monitoring is *structurally* blind here, not merely badly tuned.

    Every reading during these two faults sits inside the min/max that channel showed during clean
    operation. No per-channel limit, however carefully chosen, can separate them — the information is
    in the relationship between channels, and a univariate check cannot look there.
    """
    data = dataset()
    position = INDEX[sensor]
    training = [row[position] for row in data.train()]
    low, high = min(training), max(training)
    event = event_of(kind)
    during = [data.rows[index][position] for index in event.indices()]
    assert during, "the event must cover at least one sample"
    assert all(low <= value <= high for value in during)


def test_the_multivariate_detectors_see_the_broken_correlation():
    """Both of them, by a wide margin, using the relationship the covariance model encodes."""
    assert ratio("mahalanobis", "correlation_break") > 5.0
    assert ratio("pca-spe", "correlation_break") > 2.0


def test_the_per_channel_detector_is_far_less_sensitive_to_it():
    """A ratio below one means "never crosses its own alarm threshold"; the comparison is asserted as
    an order-of-magnitude gap so the test measures the effect and not the noise."""
    univariate = ratio("robust-z", "correlation_break")
    multivariate = ratio("mahalanobis", "correlation_break")
    assert univariate * 10 < multivariate
    assert univariate * 2 < ratio("pca-spe", "correlation_break")


def test_the_per_channel_detector_still_wins_on_the_obvious_faults():
    """It is the right tool for a spike or a level shift, and it is kept for exactly that reason."""
    assert ratio("robust-z", "spike") > 5.0
    assert ratio("robust-z", "level_shift") > 2.0


def test_the_covariance_model_catches_the_frozen_transmitter():
    """A plausible value that stops agreeing with the rest of the machine."""
    assert ratio("mahalanobis", "stuck_sensor") > 3.0


def test_the_slow_drift_is_the_hard_one_for_everybody():
    """Documented rather than hidden: a 4 K ramp over twelve hours is at the edge of what any of
    these detectors can separate from ordinary process variation at its median point."""
    ratios = {name: ratio(name, "slow_drift") for name in ("robust-z", "mahalanobis", "pca-spe")}
    assert all(value < 5.0 for value in ratios.values()), ratios


# -- end to end ------------------------------------------------------------------------------


def test_the_full_pipeline_catches_most_faults_without_flooding_the_operator():
    from msad.detectors import MahalanobisDetector

    result = run_detector(dataset(), MahalanobisDetector())
    caught = sum(1 for hit in result.events.detected.values() if hit)
    assert caught >= 4, result.events.detected
    assert result.events.detected["correlation_break"] is True
    assert result.events.detected["stuck_sensor"] is True
    assert result.events.false_alarms_per_day < 3.0
    assert result.raw_exceedances >= len(result.alarms), "debouncing cannot invent exceedances"
    assert result.static_threshold > 0.0


def test_debouncing_collapses_many_exceedances_into_few_alarms():
    from msad.detectors import MahalanobisDetector

    result = run_detector(dataset(), MahalanobisDetector())
    assert len(result.alarms) < 30, "one notification per fault, not one per sample"
    assert result.chatter_ratio > 1.0


def test_no_detector_is_evaluated_on_the_window_it_was_fitted_on():
    """Scoring the training window would flatter every detector here."""
    data = dataset()
    for detector in default_detectors():
        result = run_detector(data, detector)
        assert all(alarm.end >= data.train_end for alarm in result.alarms)


def test_point_adjusted_f1_flatters_a_coin_flip_on_this_benchmark():
    """Reproduces the inflation on real labels, which is why the README reports event-level numbers."""
    data = dataset()
    rng = random.Random(4242)
    predictions = [
        1 if index >= data.train_end and rng.random() < 0.02 else 0 for index in range(len(data))
    ]
    labels = [
        label if index >= data.train_end else 0 for index, label in enumerate(data.labels())
    ]
    honest = point_wise_f1(labels, predictions)
    inflated = point_adjusted_f1(labels, predictions, data.events)
    assert inflated.f1 > 5 * honest.f1
    assert inflated.recall > 0.5, "random noise 'detects' nearly every long event"


# -- the demonstrations ----------------------------------------------------------------------


@pytest.mark.parametrize("command", sorted(COMMANDS))
def test_every_demonstration_runs(command, capsys):
    assert main([command]) == 0
    printed = capsys.readouterr().out
    assert len(printed) > 400, f"{command} printed almost nothing"

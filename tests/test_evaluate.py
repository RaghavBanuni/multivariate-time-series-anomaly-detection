"""Evaluation: the honest metric, the operational metric, and the inflated one."""

import pytest

from msad.evaluate import (
    event_scores,
    point_adjusted_f1,
    point_wise_f1,
    score_separation,
)
from msad.threshold import Alarm
from msad.types import AnomalyEvent


def event(start: int, end: int, kind: str = "fault") -> AnomalyEvent:
    return AnomalyEvent(start=start, end=end, kind=kind, sensors=("x",), note="synthetic")


# -- point-wise ------------------------------------------------------------------------------


def test_point_wise_scores_are_the_textbook_quantities():
    labels = [0, 1, 1, 1, 0]
    predictions = [0, 1, 0, 1, 1]
    score = point_wise_f1(labels, predictions)
    assert (score.true_positives, score.false_positives, score.false_negatives) == (2, 1, 1)
    assert score.precision == pytest.approx(2 / 3)
    assert score.recall == pytest.approx(2 / 3)
    assert score.f1 == pytest.approx(2 / 3)
    assert "F1=0.667" in score.summary()


def test_an_empty_prediction_set_scores_zero_rather_than_dividing_by_zero():
    score = point_wise_f1([0, 1, 1], [0, 0, 0])
    assert (score.precision, score.recall, score.f1) == (0.0, 0.0, 0.0)


def test_a_perfect_prediction_scores_one():
    score = point_wise_f1([0, 1, 1, 0], [0, 1, 1, 0])
    assert score.f1 == 1.0


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError, match="differ in length"):
        point_wise_f1([0, 1], [0])


# -- point-adjusted --------------------------------------------------------------------------


def test_one_lucky_sample_buys_an_entire_interval():
    """The reason this repository does not draw conclusions from point-adjusted F1."""
    labels = [0] * 10 + [1] * 100 + [0] * 10
    predictions = [0] * 120
    predictions[50] = 1  # a single flagged sample inside a 100-sample fault

    honest = point_wise_f1(labels, predictions)
    inflated = point_adjusted_f1(labels, predictions, [event(10, 109)])

    assert honest.recall == pytest.approx(0.01)
    assert inflated.recall == 1.0
    assert inflated.f1 == 1.0
    assert inflated.f1 > 90 * honest.f1


def test_the_adjustment_credits_nothing_when_the_interval_was_missed():
    labels = [0] * 10 + [1] * 20 + [0] * 10
    predictions = [0] * 40
    predictions[0] = 1  # a false alarm outside the event
    adjusted = point_adjusted_f1(labels, predictions, [event(10, 29)])
    assert adjusted.recall == 0.0
    assert adjusted.false_positives == 1


def test_the_adjustment_leaves_false_positives_untouched():
    labels = [0] * 5 + [1] * 5
    predictions = [1] + [0] * 4 + [1] + [0] * 4
    adjusted = point_adjusted_f1(labels, predictions, [event(5, 9)])
    assert adjusted.false_positives == 1, "only labelled intervals are extended"
    assert adjusted.true_positives == 5


# -- event level -----------------------------------------------------------------------------


def test_an_alarm_inside_an_event_is_a_detection_with_measured_latency():
    score = event_scores(
        [event(100, 199)],
        [Alarm(start=150, end=160, peak=9.0)],
        evaluated_samples=2880,
        minutes_per_sample=5.0,
    )
    assert score.detected["fault"] is True
    assert score.latency_minutes["fault"] == pytest.approx(250.0)  # 50 samples x 5 min
    assert score.false_alarms == 0
    assert score.recall == 1.0


def test_an_alarm_already_standing_when_the_fault_starts_has_zero_latency():
    """It is neither a false alarm nor a miss: the operator was already looking at the machine."""
    score = event_scores(
        [event(100, 199)],
        [Alarm(start=80, end=120, peak=9.0)],
        evaluated_samples=2880,
        minutes_per_sample=5.0,
    )
    assert score.detected["fault"] is True
    assert score.latency_minutes["fault"] == 0.0
    assert score.false_alarms == 0


def test_a_missed_event_reports_no_latency_rather_than_a_misleading_zero():
    score = event_scores(
        [event(100, 199)],
        [Alarm(start=500, end=510, peak=9.0)],
        evaluated_samples=2880,
        minutes_per_sample=5.0,
    )
    assert score.detected["fault"] is False
    assert score.latency_minutes["fault"] is None
    assert score.false_alarms == 1


def test_false_alarms_are_counted_per_alarm_and_normalised_per_day():
    score = event_scores(
        [],
        [Alarm(start=10, end=11, peak=1.0), Alarm(start=900, end=1200, peak=1.0)],
        evaluated_samples=2880,  # 10 days at five-minute cadence
        minutes_per_sample=5.0,
    )
    assert score.false_alarms == 2, "a long spurious alarm is one interruption, not 300"
    assert score.false_alarms_per_day == pytest.approx(0.2)
    assert score.alarms == 2


def test_one_alarm_spanning_two_events_detects_both_and_is_not_a_false_alarm():
    score = event_scores(
        [event(100, 150, "first"), event(200, 250, "second")],
        [Alarm(start=90, end=260, peak=9.0)],
        evaluated_samples=1000,
        minutes_per_sample=5.0,
    )
    assert score.detected == {"first": True, "second": True}
    assert score.false_alarms == 0
    assert score.recall == 1.0


def test_the_summary_reports_what_an_operator_asks_for():
    score = event_scores(
        [event(100, 199), event(400, 499, "other")],
        [Alarm(start=110, end=120, peak=9.0)],
        evaluated_samples=2880,
        minutes_per_sample=5.0,
    )
    text = score.summary()
    assert "1/2 events" in text
    assert "latency 50 min" in text
    assert "0.00/day" in text
    assert score.recall == 0.5


# -- threshold-free separation ---------------------------------------------------------------


def test_separation_reports_the_median_score_inside_the_event():
    scores = [0.0] * 10 + [1.0, 2.0, 3.0, 100.0] + [0.0] * 10
    inside, threshold = score_separation(scores, event(10, 13), threshold=5.0)
    assert inside == pytest.approx(2.5), "the median, so one lucky sample proves nothing"
    assert threshold == 5.0


def test_separation_of_an_odd_window_takes_the_middle_value():
    scores = [0.0] * 10 + [1.0, 7.0, 3.0] + [0.0] * 10
    inside, _ = score_separation(scores, event(10, 12), threshold=1.0)
    assert inside == 3.0


def test_separation_of_an_event_beyond_the_series_is_zero_not_an_exception():
    inside, threshold = score_separation([1.0, 2.0], event(50, 60), threshold=4.0)
    assert inside == 0.0
    assert threshold == 4.0

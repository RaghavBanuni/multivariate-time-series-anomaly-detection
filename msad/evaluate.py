"""Evaluation, including the metric this field should stop reporting.

Three views of the same detections, because each answers a different question and only one of them is
the question an operator asks.

**Point-wise F1** treats every sample as an independent classification. Honest, harsh, and pessimistic
in a specific way: a detector that fires four minutes into a six-hour fault and stays on gets nearly
full credit, while one that fires reliably but only on the sharpest part of the fault is punished for
samples nobody cares about.

**Point-adjusted F1** is the metric most papers report: if *any* sample inside a ground-truth interval
is flagged, the whole interval is credited as detected. It is close to meaningless, and this module
implements it only so the demo can show why. With long events, a detector that flags 2% of samples at
random scores a point-adjusted F1 around 0.85 — while its point-wise F1 sits near 0.03. Any published
comparison relying on it is dominated by event *length*, not by detector quality. (Kim et al., "Towards
a Rigorous Evaluation of Time-series Anomaly Detection", AAAI 2022, makes this argument at length; the
test suite reproduces the effect.)

**Event-level scores with latency and false alarms per day** is what actually decides deployment. An
operator wants: did we catch the fault, how long after it began, and how many times did we cry wolf per
shift. Latency is measured in wall-clock minutes, because "7 samples" means nothing to the person
holding the radio.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .threshold import Alarm
from .types import AnomalyEvent


@dataclass(frozen=True)
class PointScore:
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int

    def summary(self) -> str:
        return (
            f"P={self.precision:.3f} R={self.recall:.3f} F1={self.f1:.3f} "
            f"(tp={self.true_positives} fp={self.false_positives} fn={self.false_negatives})"
        )


def _score(tp: int, fp: int, fn: int) -> PointScore:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return PointScore(precision, recall, f1, tp, fp, fn)


def point_wise_f1(labels: Sequence[int], predictions: Sequence[int]) -> PointScore:
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions differ in length")
    tp = sum(1 for y, p in zip(labels, predictions) if y and p)
    fp = sum(1 for y, p in zip(labels, predictions) if not y and p)
    fn = sum(1 for y, p in zip(labels, predictions) if y and not p)
    return _score(tp, fp, fn)


def point_adjusted_f1(
    labels: Sequence[int], predictions: Sequence[int], events: Sequence[AnomalyEvent]
) -> PointScore:
    """The inflated metric: one flagged sample credits an entire interval.

    Implemented faithfully so the demonstration is fair, and used nowhere in this repository's own
    conclusions.
    """
    adjusted = list(predictions)
    for event in events:
        if any(predictions[index] for index in event.indices() if index < len(predictions)):
            for index in event.indices():
                if index < len(adjusted):
                    adjusted[index] = 1
    return point_wise_f1(labels, adjusted)


@dataclass
class EventScore:
    """Per-event detection with latency, plus the false-alarm rate the alarms imply."""

    detected: dict[str, bool] = field(default_factory=dict)
    latency_minutes: dict[str, float | None] = field(default_factory=dict)
    false_alarms: int = 0
    false_alarms_per_day: float = 0.0
    alarms: int = 0

    @property
    def recall(self) -> float:
        if not self.detected:
            return 0.0
        return sum(1 for hit in self.detected.values() if hit) / len(self.detected)

    def summary(self) -> str:
        caught = sum(1 for hit in self.detected.values() if hit)
        latencies = [value for value in self.latency_minutes.values() if value is not None]
        mean_latency = sum(latencies) / len(latencies) if latencies else float("nan")
        return (
            f"{caught}/{len(self.detected)} events, mean latency {mean_latency:.0f} min, "
            f"{self.false_alarms} false alarms ({self.false_alarms_per_day:.2f}/day)"
        )


def event_scores(
    events: Sequence[AnomalyEvent],
    alarms: Sequence[Alarm],
    evaluated_samples: int,
    minutes_per_sample: float,
) -> EventScore:
    """Match alarms to events; count the rest as false alarms.

    An alarm that starts before an event and continues into it counts as a detection with zero latency
    rather than as both a false alarm and a miss. That is the operationally sensible reading: the
    operator was already looking at the machine.

    False alarms are counted per *alarm*, not per sample: one 30-minute spurious alarm is one
    interruption, and the per-sample count would rate it as six times worse than a five-minute one for
    no operational reason.
    """
    result = EventScore(alarms=len(alarms))
    matched: set[int] = set()

    for event in events:
        key = event.kind
        hits = [
            (position, alarm)
            for position, alarm in enumerate(alarms)
            if alarm.overlaps(event.start, event.end)
        ]
        result.detected[key] = bool(hits)
        if hits:
            first = min(alarm.start for _, alarm in hits)
            result.latency_minutes[key] = max(0.0, (first - event.start) * minutes_per_sample)
            matched.update(position for position, _ in hits)
        else:
            result.latency_minutes[key] = None

    result.false_alarms = len(alarms) - len(matched)
    days = evaluated_samples * minutes_per_sample / 1440.0
    result.false_alarms_per_day = result.false_alarms / days if days > 0 else 0.0
    return result


def score_separation(
    scores: Sequence[float], event: AnomalyEvent, threshold: float
) -> tuple[float, float]:
    """``(median score inside the event, threshold)`` — the threshold-free view of detectability.

    Useful when the question is "can this detector see this fault at all", independent of the
    thresholding and hysteresis stack. The median is used rather than the maximum so that one lucky
    sample cannot claim detectability the detector does not have.
    """
    window = sorted(scores[index] for index in event.indices() if index < len(scores))
    if not window:
        return 0.0, threshold
    middle = len(window) // 2
    if len(window) % 2:
        return window[middle], threshold
    return 0.5 * (window[middle - 1] + window[middle]), threshold

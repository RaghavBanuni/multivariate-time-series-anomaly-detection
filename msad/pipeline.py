"""Wiring: fit on the clean window, score, threshold, debounce, evaluate.

One function, because the order of these steps is where the mistakes live. Calibrating the threshold on
the whole series, or scoring the training window with a detector fitted on it and then reporting those
scores as evidence, are both easy to do by accident and both produce nonsense that looks like success.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .detectors import Detector, training_quantile_threshold
from .evaluate import (
    EventScore,
    PointScore,
    event_scores,
    point_adjusted_f1,
    point_wise_f1,
)
from .threshold import (
    AdaptiveQuantileThreshold,
    Alarm,
    HysteresisPolicy,
    StaticQuantileThreshold,
    alarms_to_mask,
    exceedance_mask,
)
from .types import Dataset, ScoreSeries


@dataclass
class RunResult:
    detector: str
    series: ScoreSeries
    thresholds: list[float]
    alarms: list[Alarm]
    events: EventScore
    point: PointScore
    point_adjusted: PointScore
    raw_exceedances: int
    static_threshold: float

    @property
    def chatter_ratio(self) -> float:
        """Raw threshold crossings per alarm actually raised — what debouncing is worth."""
        return self.raw_exceedances / len(self.alarms) if self.alarms else float("inf")


def run_detector(
    dataset: Dataset,
    detector: Detector,
    threshold_level: float = 0.999,
    adaptive: bool = False,
    policy: HysteresisPolicy | None = None,
) -> RunResult:
    """Fit on ``dataset.train()`` only, then score and evaluate the whole series.

    Evaluation deliberately covers the *test* portion only for false-alarm rates: counting the training
    window would flatter every detector, since it was fitted there.
    """
    policy = policy or HysteresisPolicy()
    detector.fit(dataset.train())
    series = detector.score(dataset.rows)

    static_value = training_quantile_threshold(series, dataset.train_end, threshold_level)
    if adaptive:
        thresholds = AdaptiveQuantileThreshold(
            level=threshold_level, floor=static_value
        ).series(series.scores, series.valid_from)
    else:
        thresholds = StaticQuantileThreshold(level=threshold_level).fit(
            series.scores[series.valid_from : dataset.train_end]
        ).series(series.scores, series.valid_from)

    evaluate_from = max(series.valid_from, dataset.train_end)
    alarms = [
        alarm
        for alarm in policy.apply(series.scores, thresholds, series.valid_from)
        if alarm.end >= evaluate_from
    ]

    labels = dataset.labels()
    predictions = alarms_to_mask(alarms, len(dataset))
    # Only the evaluated region contributes to point metrics; the training window is neither a
    # success nor a failure, it is where the model came from.
    masked_labels = [
        label if index >= evaluate_from else 0 for index, label in enumerate(labels)
    ]

    return RunResult(
        detector=series.name,
        series=series,
        thresholds=thresholds,
        alarms=alarms,
        events=event_scores(
            dataset.events,
            alarms,
            evaluated_samples=len(dataset) - evaluate_from,
            minutes_per_sample=dataset.step.total_seconds() / 60.0,
        ),
        point=point_wise_f1(masked_labels, predictions),
        point_adjusted=point_adjusted_f1(masked_labels, predictions, dataset.events),
        raw_exceedances=sum(
            exceedance_mask(series.scores, thresholds, evaluate_from)
        ),
        static_threshold=static_value,
    )


def default_detectors() -> list[Detector]:
    """The four detectors with the settings used throughout the demos and the README."""
    from .detectors import (
        IsolationDepthDetector,
        MahalanobisDetector,
        PCAReconstructionDetector,
        RobustZScoreDetector,
    )

    return [
        RobustZScoreDetector(window=288),
        MahalanobisDetector(),
        PCAReconstructionDetector(variance_target=0.95),
        IsolationDepthDetector(n_trees=60, subsample=256),
    ]

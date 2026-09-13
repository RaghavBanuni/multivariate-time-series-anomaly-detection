"""Multivariate anomaly detection for industrial telemetry.

The interesting failures on a pump skid are not spikes. A spike is caught by a threshold that an
instrument engineer set in 1998 and it does not need machine learning. The failures that matter are
the ones where every sensor stays inside its own limits and the *relationship between them* breaks:
discharge pressure stops tracking motor current, bearing temperature stops tracking vibration.
Univariate monitoring is structurally blind to that, no matter how carefully the limits are tuned.

So this package is built around four detectors with genuinely different blind spots
(:mod:`msad.detectors`), the linear algebra they need written out rather than imported
(:mod:`msad.linalg`, :mod:`msad.chisq`), thresholding that behaves like something an operator would
tolerate (:mod:`msad.threshold`), and an evaluation module that refuses to report the inflated metric
the literature usually reports (:mod:`msad.evaluate`).

Standard library only.
"""

from .chisq import chi2_cdf, chi2_ppf
from .detectors import (
    IsolationDepthDetector,
    MahalanobisDetector,
    PCAReconstructionDetector,
    RobustZScoreDetector,
)
from .evaluate import EventScore, PointScore, event_scores, point_adjusted_f1, point_wise_f1
from .threshold import AdaptiveQuantileThreshold, Alarm, HysteresisPolicy, StaticQuantileThreshold
from .types import AnomalyEvent, Dataset, ScoreSeries

__all__ = [
    "AdaptiveQuantileThreshold",
    "Alarm",
    "AnomalyEvent",
    "Dataset",
    "EventScore",
    "HysteresisPolicy",
    "IsolationDepthDetector",
    "MahalanobisDetector",
    "PCAReconstructionDetector",
    "PointScore",
    "RobustZScoreDetector",
    "ScoreSeries",
    "StaticQuantileThreshold",
    "chi2_cdf",
    "chi2_ppf",
    "event_scores",
    "point_adjusted_f1",
    "point_wise_f1",
]

__version__ = "1.0.0"

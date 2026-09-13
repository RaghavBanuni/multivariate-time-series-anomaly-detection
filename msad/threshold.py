"""Turning scores into alarms — the part that decides whether anyone keeps the system switched on.

A detector that emits one boolean per sample is not a monitoring system. At a five-minute cadence a
six-hour fault produces 72 separate alarms, and an operator facing 72 notifications for one event will
suppress the tag by lunchtime. Alarm flooding is the documented reason monitoring systems get disabled
in practice (see the EEMUA 191 guidance on tolerable alarm rates), so the pipeline here is:

    score  ->  threshold  ->  hysteresis + dwell time  ->  alarm events

Three thresholding strategies, with their trade-offs stated rather than implied:

**Static quantile** of the clean training scores. Calibrated once, never moves. Correct when the
process is stationary; produces a rising false-alarm rate as the plant ages, and every seasonal
recalibration is a manual job.

**Adaptive quantile** over a rolling window of recent scores. Absorbs slow baseline drift, which is
what keeps the false-alarm rate flat over months. Its cost is severe and usually unstated: *a threshold
that adapts will follow a slow fault and never fire*. A ramp is invisible to it by construction. This
is not a bug to be fixed by tuning; it is the trade being made, and `python -m msad thresholds`
demonstrates it on the injected drift.

**Alarming samples excluded from the adaptive baseline** (``exclude_alarms``). Without it, an ongoing
fault feeds its own high scores into the window that judges it, the threshold climbs, and the alarm
clears itself while the fault is still running — the same self-defeating calibration as taking the
static quantile over the whole series instead of the training window.

Hysteresis then separates the raise decision from the clear decision. A single threshold with a score
hovering near it produces chatter; requiring ``min_duration`` consecutive exceedances to raise, and a
drop below ``exit_ratio * threshold`` sustained for ``clear_duration`` to clear, produces one alarm per
fault. That is also why ``min_duration`` is a form of noise filtering the *threshold* cannot provide:
an isolated spurious sample never becomes an alarm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .stats import quantile


@dataclass(frozen=True)
class Alarm:
    """A raised alarm as an inclusive index interval, with the worst score it saw."""

    start: int
    end: int
    peak: float

    def __len__(self) -> int:
        return self.end - self.start + 1

    def overlaps(self, start: int, end: int) -> bool:
        return self.start <= end and start <= self.end


@dataclass
class StaticQuantileThreshold:
    """One threshold from the clean training scores, held for the whole series."""

    level: float = 0.999
    name: str = "static"
    value: float = 0.0

    def fit(self, training_scores: Sequence[float]) -> "StaticQuantileThreshold":
        if not training_scores:
            raise ValueError("no training scores to calibrate on")
        self.value = quantile(list(training_scores), self.level)
        return self

    def series(self, scores: Sequence[float], valid_from: int = 0) -> list[float]:
        return [self.value] * len(scores)


@dataclass
class AdaptiveQuantileThreshold:
    """Rolling-quantile threshold, optionally deaf to its own alarms.

    ``floor`` keeps the threshold from collapsing during unusually quiet periods: a rolling quantile of
    a flat score series is itself flat, and then ordinary noise clears it. The floor is normally the
    static training threshold, i.e. "never become more sensitive than the calibration said".
    """

    window: int = 576
    level: float = 0.999
    exclude_alarms: bool = True
    floor: float = 0.0
    name: str = "adaptive"

    def series(self, scores: Sequence[float], valid_from: int = 0) -> list[float]:
        thresholds = [0.0] * len(scores)
        baseline: list[float] = []
        current = self.floor
        for index in range(len(scores)):
            if index < valid_from:
                thresholds[index] = max(self.floor, current)
                continue
            if len(baseline) >= 2:
                current = max(self.floor, quantile(baseline[-self.window :], self.level))
            thresholds[index] = current
            if not (self.exclude_alarms and scores[index] > current):
                baseline.append(scores[index])
        return thresholds


@dataclass
class HysteresisPolicy:
    """Exceedance series -> alarm events, with dwell time, hysteresis and a cooldown."""

    min_duration: int = 3
    exit_ratio: float = 0.7
    clear_duration: int = 6
    cooldown: int = 0
    name: str = "hysteresis"

    def __post_init__(self) -> None:
        if self.min_duration < 1:
            raise ValueError("min_duration must be at least one sample")
        if not 0.0 < self.exit_ratio <= 1.0:
            raise ValueError("exit_ratio must be in (0, 1]")

    def apply(
        self,
        scores: Sequence[float],
        thresholds: Sequence[float],
        valid_from: int = 0,
    ) -> list[Alarm]:
        alarms: list[Alarm] = []
        run_start: int | None = None
        run_length = 0
        raised = False
        below = 0
        peak = 0.0
        blocked_until = -1

        for index in range(valid_from, len(scores)):
            score, threshold = scores[index], thresholds[index]
            if score > threshold:
                if run_start is None:
                    run_start = index
                    run_length = 0
                    peak = score
                run_length += 1
                peak = max(peak, score)
                below = 0
                if not raised and run_length >= self.min_duration and index > blocked_until:
                    raised = True
            elif raised:
                if score < self.exit_ratio * threshold:
                    below += 1
                    if below >= self.clear_duration:
                        # `run_start if not None` rather than `or`: index 0 is a legitimate start and
                        # `run_start or index` would silently discard it.
                        alarms.append(
                            Alarm(
                                start=run_start if run_start is not None else index,
                                end=index,
                                peak=peak,
                            )
                        )
                        blocked_until = index + self.cooldown
                        raised, run_start, run_length, below, peak = False, None, 0, 0, 0.0
                else:
                    below = 0  # still in the hysteresis band: neither firing nor clearing
            else:
                run_start, run_length, peak = None, 0, 0.0

        if raised and run_start is not None:
            alarms.append(Alarm(start=run_start, end=len(scores) - 1, peak=peak))
        return alarms


def exceedance_mask(
    scores: Sequence[float], thresholds: Sequence[float], valid_from: int = 0
) -> list[int]:
    """Raw per-sample exceedance, for point-wise metrics and for showing what chatter looks like."""
    return [
        1 if index >= valid_from and scores[index] > thresholds[index] else 0
        for index in range(len(scores))
    ]


def alarms_to_mask(alarms: Sequence[Alarm], length: int) -> list[int]:
    mask = [0] * length
    for alarm in alarms:
        for index in range(max(alarm.start, 0), min(alarm.end, length - 1) + 1):
            mask[index] = 1
    return mask

"""Data containers. Deliberately plain lists of floats.

A telemetry dataset is a matrix, a clock, and — only because this is a benchmark — ground truth. The
ground truth is stored as **intervals**, not as a per-sample bitmask, because that is what it actually
is: a maintenance record says "the seal was leaking from Tuesday 14:00 until the shutdown on
Thursday", and the per-sample labels are derived from that. Storing intervals keeps event-level
evaluation honest and makes detection latency a well-defined quantity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Iterator, Sequence


@dataclass(frozen=True)
class AnomalyEvent:
    """A ground-truth fault, as an inclusive index interval.

    ``sensors`` records which channels the fault actually touched. It is never given to a detector;
    it is used when explaining *why* a detector missed something, which is the difference between a
    benchmark and a leaderboard.
    """

    start: int
    end: int
    kind: str
    sensors: tuple[str, ...] = ()
    note: str = ""

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"event {self.kind} ends before it starts")

    def __len__(self) -> int:
        return self.end - self.start + 1

    def indices(self) -> range:
        return range(self.start, self.end + 1)

    def contains(self, index: int) -> bool:
        return self.start <= index <= self.end


@dataclass
class Dataset:
    """Aligned multivariate telemetry with interval ground truth."""

    sensors: tuple[str, ...]
    rows: list[list[float]]
    start: datetime
    step: timedelta
    events: list[AnomalyEvent] = field(default_factory=list)
    train_end: int = 0

    def __post_init__(self) -> None:
        width = len(self.sensors)
        for position, row in enumerate(self.rows):
            if len(row) != width:
                raise ValueError(
                    f"row {position} has {len(row)} values for {width} sensors; a ragged telemetry "
                    f"matrix means an upstream alignment bug, not a missing value"
                )
        if not 0 <= self.train_end <= len(self.rows):
            raise ValueError("train_end outside the dataset")

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def width(self) -> int:
        return len(self.sensors)

    def timestamp(self, index: int) -> datetime:
        return self.start + index * self.step

    def column(self, sensor: str) -> list[float]:
        position = self.sensors.index(sensor)
        return [row[position] for row in self.rows]

    def train(self) -> list[list[float]]:
        """The clean warm-up period every detector is fitted on.

        Fitting on the whole series would let the faults inflate the baseline covariance and make the
        detectors look worse *and* be non-reproducible in an online setting, where the future is not
        available.
        """
        return self.rows[: self.train_end]

    def labels(self) -> list[int]:
        """Per-sample ground truth derived from the intervals."""
        flags = [0] * len(self.rows)
        for event in self.events:
            for index in event.indices():
                if 0 <= index < len(flags):
                    flags[index] = 1
        return flags

    def anomaly_free_train_labels(self) -> bool:
        """True when no ground-truth event overlaps the training window."""
        return all(event.start >= self.train_end for event in self.events)

    def iter_rows(self, start: int = 0) -> Iterator[tuple[int, list[float]]]:
        for index in range(start, len(self.rows)):
            yield index, self.rows[index]

    def duration_days(self, start: int, end: int) -> float:
        """Wall-clock days covered by an index range, used for false alarms per day."""
        samples = max(end - start, 0)
        return samples * self.step.total_seconds() / 86400.0


@dataclass
class ScoreSeries:
    """A detector's output: one score per sample, higher meaning more anomalous.

    ``valid_from`` marks where the scores become meaningful. A detector with a 288-sample rolling
    baseline has nothing to say about sample 3, and letting it emit a score there quietly turns
    warm-up artefacts into false alarms — or, worse, into apparent skill.
    """

    name: str
    scores: list[float]
    valid_from: int = 0
    detail: dict[str, list[float]] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.scores)

    def valid_slice(self) -> Sequence[float]:
        return self.scores[self.valid_from :]

    def max_score(self) -> float:
        window = self.valid_slice()
        return max(window) if window else 0.0

"""Rolling and robust statistics, with the numerical decisions written down.

Three things in here are easy to get wrong in ways that only show up as mysteriously bad detections.

**Variance.** The textbook shortcut ``E[x^2] - E[x]^2`` is catastrophic on telemetry, because
telemetry lives far from zero: a bearing temperature around 340 K with a standard deviation of 0.5 K
means subtracting two numbers that agree in the first six digits. In float64 that keeps roughly three
significant digits of the answer, and in the worst case returns a small negative variance whose square
root raises. Welford's recurrence never forms that difference, so it is used everywhere here:

    M_k = M_{k-1} + (x_k - M_{k-1}) / k
    S_k = S_{k-1} + (x_k - M_{k-1})(x_k - M_k)
    var = S_n / (n - 1)

**Sliding windows.** Welford has no numerically stable *removal* step — running it backwards
reintroduces the cancellation it was designed to avoid, and the error accumulates monotonically over a
long stream. So :class:`RollingStats` keeps the raw window and recomputes in ``O(w)``. That is the
honest trade: exactness over an asymptotic win that nobody would notice at a five-minute cadence.

**Robust scale.** A fault contaminates the very window used to judge it, so the baseline has to be
robust. Median absolute deviation is the standard choice, and it needs the consistency constant that
most blog posts omit: for ``X ~ N(mu, sigma^2)``,

    MAD = median(|X - mu|) = sigma * Phi^{-1}(0.75) = 0.674490 * sigma

so ``sigma_hat = MAD / 0.674490 = 1.482602 * MAD``. Without it every threshold expressed in "sigmas"
is 48% too tight, and the alarm list is full of noise.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

# Phi^{-1}(0.75); the reciprocal 1.4826... converts a MAD into a Gaussian-consistent sigma.
NORMAL_MAD_QUANTILE = 0.6744897501960817
MAD_TO_SIGMA = 1.0 / NORMAL_MAD_QUANTILE


@dataclass
class Welford:
    """Streaming mean and variance without the cancelling difference."""

    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> "Welford":
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (value - self.mean)
        return self

    def extend(self, values: Sequence[float]) -> "Welford":
        for value in values:
            self.update(value)
        return self

    @property
    def variance(self) -> float:
        """Sample variance (``n - 1``). Zero for fewer than two observations — undefined, not an
        exception, because a warm-up window legitimately hits this on sample one."""
        return self.m2 / (self.n - 1) if self.n > 1 else 0.0

    @property
    def population_variance(self) -> float:
        return self.m2 / self.n if self.n else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(max(self.variance, 0.0))


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean of an empty sequence")
    return math.fsum(values) / len(values)


def variance(values: Sequence[float]) -> float:
    """Two-pass sample variance: the reference implementation the tests compare Welford against."""
    if len(values) < 2:
        return 0.0
    centre = mean(values)
    return math.fsum((value - centre) ** 2 for value in values) / (len(values) - 1)


def std(values: Sequence[float]) -> float:
    return math.sqrt(max(variance(values), 0.0))


def quantile(values: Sequence[float], q: float) -> float:
    """Linearly interpolated quantile (the R type-7 / numpy default definition).

    Interpolation matters here: with a 0.999 threshold on a 2000-sample warm-up, a nearest-rank
    quantile snaps to the second-largest observation and the resulting threshold jitters wildly
    between refits.
    """
    if not values:
        raise ValueError("quantile of an empty sequence")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def median(values: Sequence[float]) -> float:
    return quantile(values, 0.5)


def mad(values: Sequence[float], scaled: bool = True) -> float:
    """Median absolute deviation; scaled to a Gaussian-consistent sigma by default."""
    if not values:
        raise ValueError("mad of an empty sequence")
    centre = median(values)
    spread = median([abs(value - centre) for value in values])
    return spread * MAD_TO_SIGMA if scaled else spread


def robust_zscore(value: float, centre: float, scale: float, scale_floor: float = 0.0) -> float:
    """``(value - centre) / scale`` with the degenerate case made explicit.

    A zero MAD is not a numerical accident — it means at least half the window is a single repeated
    number, i.e. the sensor is stuck, quantised coarsely, or the process is genuinely flat. Dividing
    by it yields an infinite score for the slightest deviation, which floods the alarm list.

    ``scale_floor`` is the caller's assertion about instrument resolution: deviations smaller than the
    least significant digit are not evidence. When no floor is given, this function returns ``inf``
    honestly rather than inventing a scale, and a stuck-sensor check is the caller's job.
    """
    effective = max(scale, scale_floor)
    deviation = value - centre
    if effective <= 0.0:
        if deviation == 0.0:
            return 0.0
        return math.inf if deviation > 0 else -math.inf
    return deviation / effective


@dataclass
class RollingStats:
    """Fixed-length window with exact recomputation. See the module docstring for why."""

    size: int
    window: deque[float] = field(default_factory=deque)

    def __post_init__(self) -> None:
        if self.size < 2:
            raise ValueError("a rolling window needs at least two samples to have a spread")
        self.window = deque(self.window, maxlen=self.size)

    def push(self, value: float) -> "RollingStats":
        self.window.append(value)
        return self

    @property
    def full(self) -> bool:
        return len(self.window) == self.size

    def __len__(self) -> int:
        return len(self.window)

    def values(self) -> list[float]:
        return list(self.window)

    def mean(self) -> float:
        return mean(self.values())

    def std(self) -> float:
        return std(self.values())

    def median(self) -> float:
        return median(self.values())

    def mad(self) -> float:
        return mad(self.values())


@dataclass
class EWMA:
    """Exponentially weighted mean and variance, with the initialisation bias removed.

    Zero-initialising and dividing by ``1 - (1 - alpha)^t`` is what makes the first few dozen samples
    usable. Seeding with the first observation instead makes the estimate hostage to one sample, which
    on telemetry is quite often the corrupted one that triggered the investigation.

        m_t = alpha*x_t + (1-alpha)*m_{t-1}          debiased: m_t / (1 - (1-alpha)^t)
        v_t = alpha*(x_t - m_{t-1})^2 + (1-alpha)*v_{t-1}

    The variance recurrence uses the *previous* mean deliberately: using the updated mean shrinks the
    residual by the factor the update just applied and biases the variance downwards, which is exactly
    the direction that produces false alarms.
    """

    alpha: float
    _mean: float = 0.0
    _var: float = 0.0
    _t: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")

    def update(self, value: float) -> "EWMA":
        previous = self.mean if self._t else value
        self._mean = self.alpha * value + (1.0 - self.alpha) * self._mean
        self._var = self.alpha * (value - previous) ** 2 + (1.0 - self.alpha) * self._var
        self._t += 1
        return self

    @property
    def mean(self) -> float:
        if self._t == 0:
            return 0.0
        correction = 1.0 - (1.0 - self.alpha) ** self._t
        return self._mean / correction if correction > 0 else self._mean

    @property
    def variance(self) -> float:
        if self._t == 0:
            return 0.0
        correction = 1.0 - (1.0 - self.alpha) ** self._t
        return self._var / correction if correction > 0 else self._var

    @property
    def std(self) -> float:
        return math.sqrt(max(self.variance, 0.0))

    @property
    def count(self) -> int:
        return self._t


def pearson(left: Sequence[float], right: Sequence[float]) -> float:
    """Correlation, computed from centred sums so it is stable far from zero.

    Returns 0.0 when either channel is constant: the correlation is undefined there, and 0.0 is the
    reading a monitoring system should act on — "no linear relationship observable" — rather than a
    NaN that silently poisons every downstream comparison.
    """
    if len(left) != len(right):
        raise ValueError("correlation of series with different lengths")
    if len(left) < 2:
        return 0.0
    left_centre, right_centre = mean(left), mean(right)
    covariance = math.fsum(
        (a - left_centre) * (b - right_centre) for a, b in zip(left, right)
    )
    left_ss = math.fsum((a - left_centre) ** 2 for a in left)
    right_ss = math.fsum((b - right_centre) ** 2 for b in right)
    if left_ss <= 0.0 or right_ss <= 0.0:
        return 0.0
    return covariance / math.sqrt(left_ss * right_ss)

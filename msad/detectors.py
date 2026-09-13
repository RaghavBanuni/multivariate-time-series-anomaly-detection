"""Four detectors with deliberately different blind spots.

An ensemble of four variations on the same idea is a waste of compute. These four disagree in ways that
map onto real fault modes, and each one's blind spot is stated in its docstring — a detector whose
failure mode you cannot name is a detector you cannot operate.

=========================  ==================================  ===================================
detector                   sees                                is blind to
=========================  ==================================  ===================================
RobustZScore               a single channel leaving its own    anything that keeps every channel
                           recent range                        inside range (correlation breaks)
Mahalanobis                improbable *combinations* under a    faults inside the training
                           Gaussian baseline                    distribution's bulk; a drifting
                                                                baseline it was not refitted on
PCAReconstruction (SPE)    violations of the normal linear      faults that move along a principal
                           structure between channels           direction (SPE stays small)
IsolationDepth             low-density regions, no Gaussian     axis-aligned splits only; smooth
                           assumption at all                    slow drift, small subsample noise
=========================  ==================================  ===================================

Every detector emits a higher-is-more-anomalous score with a ``valid_from`` marker. Scores before that
point are warm-up artefacts, and letting them into an evaluation manufactures either false alarms or
fake skill.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Protocol, Sequence

from .chisq import chi2_ppf
from .linalg import (
    Cholesky,
    Matrix,
    cholesky,
    column_means,
    column_stds,
    covariance_matrix,
    dot,
    jacobi_eigen,
    mahalanobis_squared,
    shrink_covariance,
)
from .stats import mad, median, quantile, robust_zscore
from .types import ScoreSeries

EULER_GAMMA = 0.5772156649015329


class Detector(Protocol):
    name: str

    def fit(self, rows: Sequence[Sequence[float]]) -> "Detector": ...

    def score(self, rows: Sequence[Sequence[float]]) -> ScoreSeries: ...


# ---------------------------------------------------------------------------------------------
# 1. Univariate robust z-score
# ---------------------------------------------------------------------------------------------


@dataclass
class RobustZScoreDetector:
    """Per-channel deviation from a rolling median, in MAD-scaled sigmas; the series maximum is the
    score.

    Three specific choices:

    - **The baseline excludes the current sample.** Including it lets a fault contaminate the very
      window used to judge it, which is how a large step change becomes a small z-score — the effect
      grows with fault duration, so the detector is weakest exactly where the fault is worst.
    - **The baseline is refreshed periodically, not per sample.** Recomputing six medians at every
      sample is ``O(n w log w)`` for no operational benefit; a real monitoring job recomputes on a
      schedule. The refresh interval is explicit so the lag is a documented property.
    - **The MAD has a floor.** A window where half the samples are identical (a stuck transmitter, a
      coarsely quantised tag, a genuinely flat process) has MAD zero, and any deviation then scores
      infinite. The floor is a fraction of the *training* MAD: an assertion that deviations far below
      the channel's normal noise are not evidence.

    Blind spot, and the reason this repository exists: a fault that keeps every channel inside its own
    normal range is invisible here, whatever the threshold.
    """

    window: int = 288
    refresh: int = 12
    floor_fraction: float = 0.25
    name: str = "robust-z"
    _floors: list[float] = field(default_factory=list)

    def fit(self, rows: Sequence[Sequence[float]]) -> "RobustZScoreDetector":
        if len(rows) < 2:
            raise ValueError("robust-z needs a training window to set its MAD floors")
        width = len(rows[0])
        self._floors = [
            self.floor_fraction * mad([row[j] for row in rows]) for j in range(width)
        ]
        return self

    def score(self, rows: Sequence[Sequence[float]]) -> ScoreSeries:
        if not self._floors:
            raise ValueError("fit before scoring: the MAD floors come from the training window")
        width = len(self._floors)
        scores = [0.0] * len(rows)
        worst = [0.0] * len(rows)
        centres: list[float] = []
        scales: list[float] = []

        for index in range(len(rows)):
            if index < self.window:
                continue
            if not centres or (index - self.window) % self.refresh == 0:
                baseline = rows[index - self.window : index]
                centres = [median([row[j] for row in baseline]) for j in range(width)]
                scales = [mad([row[j] for row in baseline]) for j in range(width)]
            best, best_channel = 0.0, 0
            for j in range(width):
                z = abs(
                    robust_zscore(rows[index][j], centres[j], scales[j], self._floors[j])
                )
                if z > best:
                    best, best_channel = z, j
            scores[index] = best
            worst[index] = float(best_channel)

        return ScoreSeries(
            name=self.name,
            scores=scores,
            valid_from=min(self.window, len(rows)),
            detail={"worst_channel": worst},
        )


# ---------------------------------------------------------------------------------------------
# 2. Mahalanobis distance with shrinkage covariance
# ---------------------------------------------------------------------------------------------


@dataclass
class MahalanobisDetector:
    """Squared Mahalanobis distance to the training centre under a shrunk covariance.

    The score is the natural multivariate answer to "how improbable is this combination of readings",
    and under a Gaussian baseline it has a calibrated threshold rather than a tuned one: ``d^2 ~
    chi^2_p``, so :meth:`theory_threshold` converts a false-alarm *rate* into a number. On real
    telemetry the Gaussian assumption is only approximately true, so an empirical quantile of the
    training scores is usually tighter — both are offered, and comparing them is itself diagnostic:
    a large gap means the baseline is far from Gaussian.

    Blind spots: a fault inside the bulk of the training distribution, and a slow baseline drift that
    the model was never refitted on — the second is why real deployments refit on a rolling clean
    window.
    """

    shrinkage: float | None = None
    name: str = "mahalanobis"
    _centre: list[float] = field(default_factory=list)
    _factor: Cholesky | None = None
    _covariance: Matrix = field(default_factory=list)
    intensity: float = 0.0

    def fit(self, rows: Sequence[Sequence[float]]) -> "MahalanobisDetector":
        if len(rows) <= len(rows[0]):
            raise ValueError(
                "fewer training samples than channels: the covariance is singular by construction, "
                "and shrinkage patches the arithmetic without rescuing the estimate"
            )
        self._centre = column_means(rows)
        self._covariance, self.intensity = shrink_covariance(rows, self.shrinkage)
        self._factor = cholesky(self._covariance)
        return self

    def score(self, rows: Sequence[Sequence[float]]) -> ScoreSeries:
        if self._factor is None:
            raise ValueError("fit before scoring")
        scores = [mahalanobis_squared(row, self._centre, self._factor) for row in rows]
        return ScoreSeries(name=self.name, scores=scores)

    @property
    def degrees_of_freedom(self) -> int:
        return len(self._centre)

    @property
    def jitter(self) -> float:
        return 0.0 if self._factor is None else self._factor.jitter

    def theory_threshold(self, false_alarm_rate: float = 1e-3) -> float:
        """Chi-square threshold for a target per-sample false-alarm rate."""
        return chi2_ppf(1.0 - false_alarm_rate, self.degrees_of_freedom)


# ---------------------------------------------------------------------------------------------
# 3. PCA reconstruction error
# ---------------------------------------------------------------------------------------------


@dataclass
class PCAReconstructionDetector:
    """Squared prediction error in the residual subspace (the SPE / Q statistic).

    PCA on the *standardised* channels (equivalently, on the correlation matrix) learns the linear
    structure of normal operation: pressure tracks current, temperature tracks vibration. Project a
    reading onto the retained components, reconstruct it, and measure what is left over:

        z = (x - mu) / sigma,   t_i = z . v_i,   z_hat = sum_{i<k} t_i v_i,   SPE = ||z - z_hat||^2

    SPE is small whenever the reading is consistent with that structure, whatever the operating point
    — and it grows the moment two channels stop agreeing, even if both stay comfortably inside their
    own limits. That is precisely the fault mode univariate monitoring cannot see.

    Standardising first is not cosmetic: on raw units a pressure in kPa (order 1000) would contribute
    ten thousand times the variance of a normalised vibration signal, and the first component would
    simply be "the pressure channel".

    Hotelling's ``T^2 = sum_{i<k} t_i^2 / lambda_i`` is reported alongside it, because the two answer
    different questions: SPE asks "is this combination possible", T^2 asks "is this operating point
    extreme". A fault that moves *along* a principal direction is the documented blind spot of SPE —
    it reconstructs perfectly — and is exactly what T^2 catches.
    """

    variance_target: float = 0.95
    name: str = "pca-spe"
    _centre: list[float] = field(default_factory=list)
    _scale: list[float] = field(default_factory=list)
    _components: list[list[float]] = field(default_factory=list)
    _eigenvalues: list[float] = field(default_factory=list)
    retained: int = 0
    explained: float = 0.0

    def fit(self, rows: Sequence[Sequence[float]]) -> "PCAReconstructionDetector":
        if not 0.0 < self.variance_target < 1.0:
            raise ValueError("variance_target must be in (0, 1)")
        if len(rows) <= len(rows[0]):
            raise ValueError("fewer training samples than channels")
        self._centre = column_means(rows)
        self._scale = column_stds(rows)
        standardised = [
            [(row[j] - self._centre[j]) / self._scale[j] for j in range(len(self._centre))]
            for row in rows
        ]
        eigen = jacobi_eigen(covariance_matrix(standardised))
        total = math.fsum(max(value, 0.0) for value in eigen.values)
        if total <= 0.0:
            raise ValueError("degenerate training data: zero total variance")

        running = 0.0
        retained = 0
        for value in eigen.values:
            running += max(value, 0.0)
            retained += 1
            if running / total >= self.variance_target:
                break
        # At least one component must be dropped or SPE is identically zero: a full-rank basis
        # reconstructs everything perfectly and the residual subspace is empty.
        retained = min(retained, len(eigen.values) - 1)
        retained = max(retained, 1)

        self.retained = retained
        self.explained = math.fsum(eigen.values[:retained]) / total
        self._components = [eigen.vectors[i] for i in range(retained)]
        self._eigenvalues = [max(eigen.values[i], 1e-12) for i in range(retained)]
        return self

    def _standardise(self, row: Sequence[float]) -> list[float]:
        return [(row[j] - self._centre[j]) / self._scale[j] for j in range(len(self._centre))]

    def statistics(self, row: Sequence[float]) -> tuple[float, float]:
        """Return ``(SPE, T^2)`` for one reading."""
        z = self._standardise(row)
        scores = [dot(z, component) for component in self._components]
        reconstruction = [0.0] * len(z)
        for weight, component in zip(scores, self._components):
            for j, value in enumerate(component):
                reconstruction[j] += weight * value
        spe = math.fsum((a - b) ** 2 for a, b in zip(z, reconstruction))
        t2 = math.fsum(
            weight * weight / eigenvalue
            for weight, eigenvalue in zip(scores, self._eigenvalues)
        )
        return spe, t2

    def score(self, rows: Sequence[Sequence[float]]) -> ScoreSeries:
        if not self._components:
            raise ValueError("fit before scoring")
        spe_values: list[float] = []
        t2_values: list[float] = []
        for row in rows:
            spe, t2 = self.statistics(row)
            spe_values.append(spe)
            t2_values.append(t2)
        return ScoreSeries(
            name=self.name, scores=spe_values, detail={"spe": spe_values, "t2": t2_values}
        )


# ---------------------------------------------------------------------------------------------
# 4. Isolation depth
# ---------------------------------------------------------------------------------------------


def harmonic_path_correction(n: int) -> float:
    """``c(n)``: the average path length of an unsuccessful search in a random binary tree.

    ``c(n) = 2 H_{n-1} - 2(n-1)/n`` with ``H_m ~ ln m + gamma``. It is the normaliser that makes depths
    comparable across leaves of different sizes: a point isolated at depth 4 in a leaf still holding 30
    samples is not as isolated as one alone at depth 4, and without ``c`` the depth-limited trees would
    systematically under-score dense regions.
    """
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (math.log(n - 1) + EULER_GAMMA) - 2.0 * (n - 1) / n


@dataclass
class IsolationDepthDetector:
    """Isolation-forest scoring, built from scratch and deterministic under a seed.

    Random axis-aligned splits isolate an outlier in fewer partitions than an inlier, so the expected
    path length is an (inverse) anomaly score:

        s(x) = 2^{-E[h(x)] / c(psi)}

    normalised to ``(0, 1)``, where ``psi`` is the subsample size. Subsampling is not a performance
    trick — it is what gives the method its resistance to swamping: with the full sample, dense normal
    regions push the tree depth up and outliers stop looking shallow by comparison.

    No distributional assumption at all, which is its advantage over Mahalanobis. Its cost is that the
    splits are axis-aligned: a correlation break that keeps both channels inside their marginal ranges
    is only visible to it through the *joint* density thinning, so it detects such faults later and less
    sharply than the PCA residual does. Determinism comes from an explicit ``random.Random(seed)``;
    an unseeded forest makes two runs of the same pipeline disagree about which alarms fired.
    """

    n_trees: int = 100
    subsample: int = 256
    seed: int = 20260101
    name: str = "isolation-depth"
    _trees: list[object] = field(default_factory=list)
    _normaliser: float = 1.0

    def fit(self, rows: Sequence[Sequence[float]]) -> "IsolationDepthDetector":
        if len(rows) < 4:
            raise ValueError("an isolation forest needs more than a handful of training rows")
        rng = random.Random(self.seed)
        size = min(self.subsample, len(rows))
        depth_limit = max(1, math.ceil(math.log2(size)))
        self._trees = [
            self._build(rng.sample(list(rows), size), 0, depth_limit, rng)
            for _ in range(self.n_trees)
        ]
        self._normaliser = harmonic_path_correction(size) or 1.0
        return self

    def _build(
        self,
        data: list[Sequence[float]],
        depth: int,
        depth_limit: int,
        rng: random.Random,
    ) -> object:
        if depth >= depth_limit or len(data) <= 1:
            return ("leaf", len(data))
        width = len(data[0])
        # Try each axis once, in a random order, before giving up: picking a single random column and
        # returning a leaf when it happens to be constant wastes the node and biases scores upward.
        order = list(range(width))
        rng.shuffle(order)
        for feature in order:
            values = [row[feature] for row in data]
            low, high = min(values), max(values)
            if high <= low:
                continue
            split = rng.uniform(low, high)
            left = [row for row in data if row[feature] < split]
            right = [row for row in data if row[feature] >= split]
            if not left or not right:
                continue
            return (
                "split",
                feature,
                split,
                self._build(left, depth + 1, depth_limit, rng),
                self._build(right, depth + 1, depth_limit, rng),
            )
        return ("leaf", len(data))  # every channel is constant here

    @staticmethod
    def _path_length(tree: object, point: Sequence[float]) -> float:
        depth = 0.0
        node = tree
        while True:
            if node[0] == "leaf":  # type: ignore[index]
                return depth + harmonic_path_correction(node[1])  # type: ignore[index]
            _, feature, split, left, right = node  # type: ignore[misc]
            node = left if point[feature] < split else right
            depth += 1.0

    def score(self, rows: Sequence[Sequence[float]]) -> ScoreSeries:
        if not self._trees:
            raise ValueError("fit before scoring")
        scores = []
        for row in rows:
            average = math.fsum(self._path_length(tree, row) for tree in self._trees) / len(
                self._trees
            )
            scores.append(2.0 ** (-average / self._normaliser))
        return ScoreSeries(name=self.name, scores=scores)


def training_quantile_threshold(
    series: ScoreSeries, train_end: int, quantile_level: float = 0.999
) -> float:
    """Empirical threshold from the clean training window only.

    Taking the quantile over the whole series lets the faults raise the threshold above themselves — a
    self-defeating calibration that quietly guarantees misses.
    """
    window = series.scores[series.valid_from : train_end]
    if not window:
        raise ValueError(
            "no valid training scores: the detector's warm-up is longer than the training window"
        )
    return quantile(window, quantile_level)


DETECTORS = {
    "robust-z": RobustZScoreDetector,
    "mahalanobis": MahalanobisDetector,
    "pca-spe": PCAReconstructionDetector,
    "isolation-depth": IsolationDepthDetector,
}

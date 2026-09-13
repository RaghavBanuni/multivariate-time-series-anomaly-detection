"""The linear algebra the detectors need, written out rather than imported.

Four decisions here are the difference between a detector that works on real telemetry and one that
raises ``LinAlgError`` on the first day of production data.

**Shrinkage is not optional.** Industrial tag lists always contain near-duplicates: a redundant
temperature transmitter, a flow that is computed from a pressure, a tag mirrored under two names by the
historian. Those make the sample covariance singular or nearly so, and ``Sigma^{-1}`` then amplifies
noise without bound — the Mahalanobis distance of a perfectly normal sample can explode. Ledoit-Wolf
linear shrinkage toward a scaled identity fixes it with a closed-form intensity, no cross-validation:

    Sigma_hat = (1 - lambda) * S + lambda * m * I,      m = trace(S) / p

    lambda* = min(1, b2 / d2),   d2 = ||S - m I||_F^2,   b2 = (1/n^2) * sum_k ||x_k x_k^T - S||_F^2

The intuition behind ``lambda*``: ``d2`` measures how far the sample covariance is from the shrinkage
target (how much structure there is to preserve) and ``b2`` estimates the variance of the sample
covariance itself (how much of that structure is noise). When the estimate is mostly noise, shrink
hard.

**Never form an inverse.** ``d^2 = (x-mu)^T Sigma^{-1} (x-mu)`` is computed as ``||L^{-1}(x-mu)||^2``
via forward substitution on the Cholesky factor. Same result, half the operations, and far better
conditioning than multiplying by an explicitly inverted matrix.

**Jitter is diagnosed, not hidden.** If the shrunk matrix still fails to factor, jitter is escalated
and the amount used is *recorded*, because needing 1e-6 of the trace is a message about the tag list —
usually two identical channels — and silently absorbing it means never receiving the message.

**Eigenvectors come from cyclic Jacobi.** For a symmetric matrix it is short, has no pathological
cases, converges quadratically once the off-diagonal norm is small, and yields an orthogonal basis to
machine precision. The off-diagonal Frobenius norm decreases monotonically at every rotation, which
makes the convergence test trustworthy rather than heuristic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

Matrix = list[list[float]]


def column_means(rows: Sequence[Sequence[float]]) -> list[float]:
    if not rows:
        raise ValueError("no rows")
    width = len(rows[0])
    return [math.fsum(row[j] for row in rows) / len(rows) for j in range(width)]


def column_stds(rows: Sequence[Sequence[float]], floor: float = 1e-12) -> list[float]:
    """Per-column sample standard deviations, floored so a constant channel cannot divide by zero."""
    centres = column_means(rows)
    n = len(rows)
    if n < 2:
        return [1.0] * len(centres)
    out = []
    for j, centre in enumerate(centres):
        ss = math.fsum((row[j] - centre) ** 2 for row in rows)
        out.append(max(math.sqrt(ss / (n - 1)), floor))
    return out


def covariance_matrix(rows: Sequence[Sequence[float]], ddof: int = 1) -> Matrix:
    """Sample covariance from centred sums.

    Centring first is not stylistic: telemetry sits far from zero (a temperature near 340 K), and the
    ``E[xy] - E[x]E[y]`` shortcut subtracts two large nearly equal numbers, losing most of the
    significant digits of a small covariance.
    """
    if not rows:
        raise ValueError("no rows")
    n, p = len(rows), len(rows[0])
    if n - ddof <= 0:
        raise ValueError(f"{n} rows is too few for ddof={ddof}")
    centres = column_means(rows)
    out = [[0.0] * p for _ in range(p)]
    for i in range(p):
        for j in range(i, p):
            value = math.fsum(
                (row[i] - centres[i]) * (row[j] - centres[j]) for row in rows
            ) / (n - ddof)
            out[i][j] = value
            out[j][i] = value  # symmetry is enforced, not hoped for
    return out


def frobenius_squared(matrix: Matrix) -> float:
    return math.fsum(value * value for row in matrix for value in row)


def ledoit_wolf_intensity(rows: Sequence[Sequence[float]]) -> float:
    """Closed-form optimal shrinkage intensity in [0, 1]. See the module docstring for the formula."""
    n, p = len(rows), len(rows[0])
    if n < 2:
        return 1.0  # one sample carries no covariance structure worth keeping
    centres = column_means(rows)
    centred = [[row[j] - centres[j] for j in range(p)] for row in rows]

    sample = [[0.0] * p for _ in range(p)]
    for i in range(p):
        for j in range(i, p):
            value = math.fsum(row[i] * row[j] for row in centred) / n  # population form (ddof=0)
            sample[i][j] = value
            sample[j][i] = value

    m = math.fsum(sample[i][i] for i in range(p)) / p
    target_gap = frobenius_squared(
        [[sample[i][j] - (m if i == j else 0.0) for j in range(p)] for i in range(p)]
    )
    if target_gap <= 0.0:
        return 1.0  # the sample covariance already *is* the target

    noise = 0.0
    for row in centred:
        noise += math.fsum(
            (row[i] * row[j] - sample[i][j]) ** 2 for i in range(p) for j in range(p)
        )
    noise /= n * n
    return max(0.0, min(1.0, noise / target_gap))


def shrink_covariance(
    rows: Sequence[Sequence[float]], intensity: float | None = None
) -> tuple[Matrix, float]:
    """Return ``(shrunk covariance, intensity used)``."""
    sample = covariance_matrix(rows, ddof=1)
    p = len(sample)
    used = ledoit_wolf_intensity(rows) if intensity is None else float(intensity)
    if not 0.0 <= used <= 1.0:
        raise ValueError("shrinkage intensity must be in [0, 1]")
    m = math.fsum(sample[i][i] for i in range(p)) / p
    return (
        [
            [
                (1.0 - used) * sample[i][j] + (used * m if i == j else 0.0)
                for j in range(p)
            ]
            for i in range(p)
        ],
        used,
    )


@dataclass
class Cholesky:
    """Lower-triangular factor with the jitter it needed recorded on it."""

    lower: Matrix
    jitter: float = 0.0
    log_determinant: float = 0.0


def cholesky(matrix: Matrix, max_attempts: int = 8) -> Cholesky:
    """``A = L L^T`` for symmetric positive definite ``A``, escalating jitter if needed.

    A non-positive pivot means the matrix is not positive definite — in practice, two channels that
    are exact linear functions of each other. Rather than failing, jitter starting at ``1e-12 *
    trace/p`` is added to the diagonal and multiplied by 100 per attempt; the amount that worked is
    returned so the caller can report it instead of pretending it did not happen.
    """
    p = len(matrix)
    scale = math.fsum(matrix[i][i] for i in range(p)) / p if p else 1.0
    jitter = 0.0
    for attempt in range(max_attempts):
        lower = [[0.0] * p for _ in range(p)]
        ok = True
        for i in range(p):
            for j in range(i + 1):
                total = matrix[i][j] + (jitter if i == j else 0.0)
                total -= math.fsum(lower[i][k] * lower[j][k] for k in range(j))
                if i == j:
                    if total <= 0.0:
                        ok = False
                        break
                    lower[i][i] = math.sqrt(total)
                else:
                    lower[i][j] = total / lower[j][j]
            if not ok:
                break
        if ok:
            log_det = 2.0 * math.fsum(math.log(lower[i][i]) for i in range(p))
            return Cholesky(lower=lower, jitter=jitter, log_determinant=log_det)
        jitter = max(scale * 1e-12, jitter * 100.0)
    raise ValueError(
        "matrix is not positive definite even with jitter; two channels are almost certainly "
        "identical (a duplicated historian tag) and one should be dropped"
    )


def solve_lower(lower: Matrix, vector: Sequence[float]) -> list[float]:
    """Forward substitution for ``L y = b``."""
    p = len(lower)
    out = [0.0] * p
    for i in range(p):
        total = vector[i] - math.fsum(lower[i][k] * out[k] for k in range(i))
        out[i] = total / lower[i][i]
    return out


def mahalanobis_squared(
    point: Sequence[float], centre: Sequence[float], factor: Cholesky
) -> float:
    """``d^2 = ||L^{-1}(x - mu)||^2``, without ever forming an inverse."""
    delta = [a - b for a, b in zip(point, centre)]
    solved = solve_lower(factor.lower, delta)
    return math.fsum(value * value for value in solved)


def identity(size: int) -> Matrix:
    return [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]


@dataclass
class Eigen:
    values: list[float] = field(default_factory=list)  # descending
    vectors: list[list[float]] = field(default_factory=list)  # vectors[i] is the i-th eigenvector
    sweeps: int = 0


def jacobi_eigen(matrix: Matrix, tolerance: float = 1e-14, max_sweeps: int = 100) -> Eigen:
    """Eigen-decomposition of a symmetric matrix by cyclic Jacobi rotations.

    Each rotation ``A <- G^T A G`` annihilates one off-diagonal pair ``(p, q)`` with

        theta = (a_qq - a_pp) / (2 a_pq),   t = sign(theta) / (|theta| + sqrt(theta^2 + 1))
        c = 1 / sqrt(t^2 + 1),              s = t * c

    That choice of ``t`` is the numerically safe root of ``t^2 + 2*theta*t - 1 = 0``: the algebraically
    equivalent quadratic formula loses precision through cancellation when ``|theta|`` is large, which
    is the common case for a nearly diagonal matrix — exactly where the method spends most of its
    rotations.

    The sum of squared off-diagonal entries strictly decreases by ``2*a_pq^2`` per rotation and the
    Frobenius norm is preserved, so the loop terminates and the convergence test means what it says.
    """
    n = len(matrix)
    for row in matrix:
        if len(row) != n:
            raise ValueError("jacobi_eigen requires a square matrix")
    for i in range(n):
        for j in range(i + 1, n):
            if abs(matrix[i][j] - matrix[j][i]) > 1e-9 * max(1.0, abs(matrix[i][j])):
                raise ValueError("jacobi_eigen requires a symmetric matrix")

    a = [row[:] for row in matrix]
    v = identity(n)
    sweeps = 0
    for sweep in range(max_sweeps):
        sweeps = sweep + 1
        off_diagonal = math.fsum(
            a[i][j] * a[i][j] for i in range(n) for j in range(i + 1, n)
        )
        if off_diagonal <= tolerance:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(a[p][q]) < 1e-300:
                    continue
                theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
                sign = 1.0 if theta >= 0.0 else -1.0
                t = sign / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c

                for k in range(n):  # A <- A G
                    akp, akq = a[k][p], a[k][q]
                    a[k][p] = c * akp - s * akq
                    a[k][q] = s * akp + c * akq
                for k in range(n):  # A <- G^T A
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k] = c * apk - s * aqk
                    a[q][k] = s * apk + c * aqk
                for k in range(n):  # V <- V G
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p] = c * vkp - s * vkq
                    v[k][q] = s * vkp + c * vkq

    pairs = sorted(
        ((a[i][i], [v[k][i] for k in range(n)]) for i in range(n)),
        key=lambda pair: -pair[0],
    )
    return Eigen(
        values=[value for value, _ in pairs],
        vectors=[vector for _, vector in pairs],
        sweeps=sweeps,
    )


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    return math.fsum(a * b for a, b in zip(left, right))

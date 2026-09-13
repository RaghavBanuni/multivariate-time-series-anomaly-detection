"""Covariance, shrinkage, Cholesky, Mahalanobis and Jacobi.

These are the parts where a wrong answer is silent: a slightly wrong covariance still produces
plausible-looking scores. So each property is checked against something independent — a hand
computation, an invariance, or a conserved quantity.
"""

import math
import random

import pytest

from msad.linalg import (
    cholesky,
    column_means,
    column_stds,
    covariance_matrix,
    frobenius_squared,
    identity,
    jacobi_eigen,
    ledoit_wolf_intensity,
    mahalanobis_squared,
    shrink_covariance,
    solve_lower,
)

COLLINEAR = [[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]]


# -- covariance ------------------------------------------------------------------------------


def test_covariance_matches_a_hand_computation():
    # column means 3 and 6; deviations (-2,-4), (0,0), (2,4)
    assert covariance_matrix(COLLINEAR) == [[4.0, 8.0], [8.0, 16.0]]


def test_covariance_is_symmetric_by_construction():
    rng = random.Random(1)
    rows = [[rng.gauss(0, 1) for _ in range(5)] for _ in range(200)]
    matrix = covariance_matrix(rows)
    for i in range(5):
        for j in range(5):
            assert matrix[i][j] == matrix[j][i]


def test_covariance_is_shift_invariant_even_far_from_zero():
    """The whole reason for centring first: telemetry sits at 340 K, not at 0."""
    rng = random.Random(2)
    rows = [[rng.gauss(0, 1), rng.gauss(0, 1)] for _ in range(400)]
    shifted = [[row[0] + 1e7, row[1] + 1e7] for row in rows]
    base = covariance_matrix(rows)
    moved = covariance_matrix(shifted)
    for i in range(2):
        for j in range(2):
            assert moved[i][j] == pytest.approx(base[i][j], rel=1e-7, abs=1e-7)


def test_column_helpers():
    assert column_means(COLLINEAR) == [3.0, 6.0]
    assert column_stds(COLLINEAR) == [pytest.approx(2.0), pytest.approx(4.0)]
    assert column_stds([[5.0], [5.0], [5.0]])[0] > 0.0, "a constant column must not divide by zero"


def test_too_few_rows_is_refused():
    with pytest.raises(ValueError, match="too few"):
        covariance_matrix([[1.0, 2.0]], ddof=1)


# -- shrinkage -------------------------------------------------------------------------------


def test_full_shrinkage_gives_exactly_the_scaled_identity():
    shrunk, used = shrink_covariance(COLLINEAR, intensity=1.0)
    assert used == 1.0
    m = (4.0 + 16.0) / 2
    assert shrunk == [[m, 0.0], [0.0, m]]


def test_zero_shrinkage_gives_back_the_sample_covariance():
    shrunk, used = shrink_covariance(COLLINEAR, intensity=0.0)
    assert used == 0.0
    assert shrunk == covariance_matrix(COLLINEAR)


def test_shrinkage_preserves_the_trace():
    """``(1-l)S + l*m*I`` keeps trace(S) exactly, so total variance is not quietly rescaled."""
    rng = random.Random(3)
    rows = [[rng.gauss(0, 2), rng.gauss(0, 5), rng.gauss(0, 1)] for _ in range(300)]
    sample = covariance_matrix(rows)
    shrunk, _ = shrink_covariance(rows, intensity=0.4)
    assert sum(shrunk[i][i] for i in range(3)) == pytest.approx(
        sum(sample[i][i] for i in range(3))
    )


def test_the_optimal_intensity_stays_in_the_unit_interval():
    rng = random.Random(4)
    for width in (2, 4, 8):
        rows = [[rng.gauss(0, 1) for _ in range(width)] for _ in range(50)]
        assert 0.0 <= ledoit_wolf_intensity(rows) <= 1.0


def test_more_samples_means_less_shrinkage():
    """``lambda*`` is the noise-to-structure ratio, and the noise term carries a 1/n."""
    rng = random.Random(5)
    population = [[rng.gauss(0, 1), rng.gauss(0, 3), rng.gauss(0, 1)] for _ in range(4000)]
    small = ledoit_wolf_intensity(population[:40])
    large = ledoit_wolf_intensity(population)
    assert large < small


def test_a_singular_matrix_still_yields_a_usable_distance():
    """Two channels that are exact multiples of each other: a mirrored historian tag."""
    shrunk, intensity = shrink_covariance(COLLINEAR)
    assert intensity > 0.0
    factor = cholesky(shrunk)
    distance = mahalanobis_squared([3.0, 6.0], column_means(COLLINEAR), factor)
    assert distance == pytest.approx(0.0, abs=1e-9)


# -- Cholesky --------------------------------------------------------------------------------


def test_the_factor_reproduces_the_matrix():
    matrix = [[4.0, 2.0, 0.6], [2.0, 3.0, 0.5], [0.6, 0.5, 1.0]]
    factor = cholesky(matrix)
    for i in range(3):
        for j in range(3):
            reconstructed = math.fsum(
                factor.lower[i][k] * factor.lower[j][k] for k in range(3)
            )
            assert reconstructed == pytest.approx(matrix[i][j], abs=1e-12)
    assert factor.jitter == 0.0


def test_the_log_determinant_is_reported():
    factor = cholesky([[4.0, 0.0], [0.0, 9.0]])
    assert factor.log_determinant == pytest.approx(math.log(36.0))


def test_jitter_is_escalated_and_recorded_rather_than_hidden():
    exactly_singular = [[1.0, 1.0], [1.0, 1.0]]
    factor = cholesky(exactly_singular)
    assert factor.jitter > 0.0, "needing jitter is a message about the tag list"


def test_an_indefinite_matrix_is_reported_not_silently_repaired():
    with pytest.raises(ValueError, match="not positive definite"):
        cholesky([[1.0, 0.0], [0.0, -5.0]])


def test_forward_substitution_solves_the_system():
    lower = [[2.0, 0.0], [1.0, 3.0]]
    solution = solve_lower(lower, [4.0, 11.0])
    assert solution[0] == pytest.approx(2.0)
    assert solution[1] == pytest.approx(3.0)


# -- Mahalanobis -----------------------------------------------------------------------------


def test_the_distance_at_the_centre_is_zero_and_scales_as_the_square():
    factor = cholesky(identity(2))
    assert mahalanobis_squared([0.0, 0.0], [0.0, 0.0], factor) == 0.0
    assert mahalanobis_squared([2.0, 0.0], [0.0, 0.0], factor) == pytest.approx(4.0)
    assert mahalanobis_squared([3.0, 4.0], [0.0, 0.0], factor) == pytest.approx(25.0)


def test_the_distance_is_invariant_under_rescaling_a_channel():
    """Mahalanobis distance is affine invariant. If it were not, the score would depend on whether an
    engineer logged a pressure in kPa or bar."""
    rng = random.Random(6)
    rows = [[rng.gauss(0, 1), rng.gauss(0, 1)] for _ in range(500)]
    point = [2.5, -1.5]

    base_cov, _ = shrink_covariance(rows, intensity=0.0)
    base = mahalanobis_squared(point, column_means(rows), cholesky(base_cov))

    scaled_rows = [[row[0] * 1000.0, row[1]] for row in rows]
    scaled_cov, _ = shrink_covariance(scaled_rows, intensity=0.0)
    scaled = mahalanobis_squared(
        [point[0] * 1000.0, point[1]], column_means(scaled_rows), cholesky(scaled_cov)
    )
    assert scaled == pytest.approx(base, rel=1e-8)


def test_the_distance_accounts_for_correlation():
    """Off the correlation line is far, along it is near — with identical Euclidean distance."""
    rng = random.Random(7)
    rows = [[0.0, 0.0] for _ in range(400)]
    for index in range(400):
        base = rng.gauss(0, 1)
        rows[index] = [base, base + rng.gauss(0, 0.05)]
    covariance, _ = shrink_covariance(rows, intensity=0.0)
    factor = cholesky(covariance)
    centre = column_means(rows)
    along = mahalanobis_squared([1.0, 1.0], centre, factor)
    across = mahalanobis_squared([1.0, -1.0], centre, factor)
    assert across > 100 * along


# -- Jacobi ----------------------------------------------------------------------------------


def test_jacobi_conserves_trace_and_determinant():
    matrix = [[4.0, 1.0, 1.0], [1.0, 3.0, 0.0], [1.0, 0.0, 2.0]]
    eigen = jacobi_eigen(matrix)
    assert sum(eigen.values) == pytest.approx(9.0)  # trace
    product = 1.0
    for value in eigen.values:
        product *= value
    assert product == pytest.approx(19.0)  # determinant by cofactor expansion


def test_jacobi_returns_an_orthonormal_basis():
    matrix = [[4.0, 1.0, 1.0], [1.0, 3.0, 0.0], [1.0, 0.0, 2.0]]
    eigen = jacobi_eigen(matrix)
    for i in range(3):
        for j in range(3):
            inner = math.fsum(a * b for a, b in zip(eigen.vectors[i], eigen.vectors[j]))
            assert inner == pytest.approx(1.0 if i == j else 0.0, abs=1e-10)


def test_jacobi_reconstructs_the_matrix():
    matrix = [[4.0, 1.0, 1.0], [1.0, 3.0, 0.0], [1.0, 0.0, 2.0]]
    eigen = jacobi_eigen(matrix)
    for i in range(3):
        for j in range(3):
            value = math.fsum(
                eigen.values[k] * eigen.vectors[k][i] * eigen.vectors[k][j] for k in range(3)
            )
            assert value == pytest.approx(matrix[i][j], abs=1e-10)


def test_jacobi_sorts_eigenvalues_descending():
    eigen = jacobi_eigen([[2.0, 1.0], [1.0, 2.0]])
    assert eigen.values == [pytest.approx(3.0), pytest.approx(1.0)]


def test_jacobi_handles_an_already_diagonal_matrix_without_rotating():
    eigen = jacobi_eigen([[5.0, 0.0], [0.0, 2.0]])
    assert eigen.values == [5.0, 2.0]
    assert eigen.vectors[0] == [1.0, 0.0]


def test_jacobi_refuses_a_non_symmetric_matrix():
    with pytest.raises(ValueError, match="symmetric"):
        jacobi_eigen([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="square"):
        jacobi_eigen([[1.0, 2.0]])


def test_frobenius_norm_is_the_sum_of_squares():
    assert frobenius_squared([[3.0, 4.0], [0.0, 0.0]]) == 25.0

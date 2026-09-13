"""Chi-square CDF and quantiles, because a threshold should mean something.

A Mahalanobis detector wants "alarm at the 99.9th percentile of normal behaviour", and under a
Gaussian baseline that percentile has a closed form: if ``x ~ N(mu, Sigma)`` in ``p`` dimensions then

    d^2(x) = (x - mu)^T Sigma^{-1} (x - mu)  ~  chi^2_p

so the threshold is the chi-square quantile at 0.999 with ``p`` degrees of freedom. Reaching for
SciPy for one number is a dependency; hard-coding a table is a lie the moment ``p`` changes. So the
regularised incomplete gamma function is implemented here, since

    F_{chi^2_k}(x) = P(k/2, x/2),   P(a, x) = gamma(a, x) / Gamma(a)

and ``P`` is evaluated by the standard pair of expansions, each used only where it converges quickly:

- the series ``P(a,x) = x^a e^{-x} / Gamma(a) * sum_{n>=0} x^n / (a(a+1)...(a+n))`` for ``x < a + 1``;
- the Legendre continued fraction for ``Q(a,x) = 1 - P(a,x)`` for ``x >= a + 1``, evaluated with
  modified Lentz to avoid the zero-denominator breakdown of a naive recursion.

Using the series in the wrong regime does not merely converge slowly; it alternates and loses the
answer to cancellation. Quantiles then come from bisection on a monotone CDF — Newton would be faster
and needs a derivative that underflows in the far tail, which is precisely where anomaly thresholds
live.
"""

from __future__ import annotations

import math

_MAX_ITERATIONS = 300
_EPSILON = 3e-16
_TINY = 1e-300


def _lower_series(a: float, x: float) -> float:
    """``P(a, x)`` by its ascending series. Valid and fast for ``x < a + 1``."""
    term = 1.0 / a
    total = term
    for n in range(1, _MAX_ITERATIONS):
        term *= x / (a + n)
        total += term
        if abs(term) < abs(total) * _EPSILON:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _upper_continued_fraction(a: float, x: float) -> float:
    """``Q(a, x)`` by continued fraction with modified Lentz. Valid for ``x >= a + 1``."""
    b = x + 1.0 - a
    c = 1.0 / _TINY
    d = 1.0 / b
    h = d
    for i in range(1, _MAX_ITERATIONS):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < _TINY:
            d = _TINY
        c = b + an / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPSILON:
            break
    return h * math.exp(-x + a * math.log(x) - math.lgamma(a))


def gammainc_lower(a: float, x: float) -> float:
    """Regularised lower incomplete gamma ``P(a, x)``, in [0, 1]."""
    if a <= 0.0:
        raise ValueError("a must be positive")
    if x < 0.0:
        raise ValueError("x must be non-negative")
    if x == 0.0:
        return 0.0
    if x < a + 1.0:
        return min(_lower_series(a, x), 1.0)
    return max(0.0, min(1.0, 1.0 - _upper_continued_fraction(a, x)))


def chi2_cdf(x: float, k: int) -> float:
    """``P(X <= x)`` for ``X ~ chi^2_k``."""
    if k < 1:
        raise ValueError("degrees of freedom must be at least 1")
    if x <= 0.0:
        return 0.0
    return gammainc_lower(k / 2.0, x / 2.0)


def chi2_sf(x: float, k: int) -> float:
    """Survival function; the p-value of a Mahalanobis distance under the Gaussian baseline."""
    return 1.0 - chi2_cdf(x, k)


def chi2_ppf(p: float, k: int, tolerance: float = 1e-10) -> float:
    """Quantile function by bracketed bisection on the monotone CDF.

    The bracket starts at the mean (``k``) and doubles until it covers ``p``. Bisection needs about 60
    iterations for machine precision, each a single CDF evaluation — irrelevant next to scoring tens
    of thousands of samples, and it cannot diverge the way Newton does when the density underflows in
    the tail where anomaly thresholds actually live.
    """
    if not 0.0 <= p < 1.0:
        raise ValueError("p must be in [0, 1)")
    if k < 1:
        raise ValueError("degrees of freedom must be at least 1")
    if p == 0.0:
        return 0.0

    low, high = 0.0, float(max(k, 1))
    while chi2_cdf(high, k) < p:
        low = high
        high *= 2.0
        if high > 1e12:  # p is numerically 1.0; the quantile is not representable
            return high

    for _ in range(200):
        middle = 0.5 * (low + high)
        if chi2_cdf(middle, k) < p:
            low = middle
        else:
            high = middle
        if high - low <= tolerance * max(1.0, high):
            break
    return 0.5 * (low + high)

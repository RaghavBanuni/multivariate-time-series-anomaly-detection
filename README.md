# Multivariate time-series anomaly detection for machine telemetry

Four detectors, an alarm-management layer, and an evaluation module that refuses to report the metric
most papers in this area report. Pure Python 3.10+, standard library only — the covariance shrinkage,
the Cholesky factorisation, the Jacobi eigen-decomposition and the chi-square quantile function are all
implemented here and checked against published tables and closed-form values.

The repository exists to make one argument concrete:

> A machine can break while every single sensor stays inside its normal range.

A pump running at high load with its discharge pressure sitting at the value it normally shows *on
average over a day* has a fault. No per-channel alarm limit can see it, because no channel left its
limits — the information is in the relationship between channels. This is not a hypothetical: it is a
stuck control valve, a worn impeller, a bypassed recirculation line.

## The geometry, in one paragraph

Six channels driven by one latent load do not fill a six-dimensional box; they lie close to a
one-dimensional curve inside it. Univariate limits describe the bounding box. The share of that box the
machine never visits grows with the number of channels, and every fault living in the empty part is
invisible to per-channel monitoring while being trivially far from the curve. So all four detectors
answer the same question in different ways: how far is this reading from the manifold the machine
normally occupies?

## The four detectors

| Detector | Statistic | Sees | Blind to |
|---|---|---|---|
| `robust-z` | per-channel median/MAD z-score, rolling 24 h | spikes, level shifts | anything that keeps each channel in range |
| `mahalanobis` | `(x-m)' S^-1 (x-m)` with Ledoit-Wolf shrinkage | broken correlations, frozen tags | drifts that move the centre slowly |
| `pca-spe` | squared reconstruction error off the retained subspace | structure violations | faults *along* a retained component |
| `isolation-depth` | mean isolation-forest path length, normalised | joint-density outliers | subtle relational faults (axis-aligned splits) |

**Robust z-score.** Median and MAD instead of mean and standard deviation, because the baseline window
usually contains part of the fault and the mean and standard deviation are dragged towards it — a single
bad sample moves the standard deviation enough to hide itself. The MAD is scaled by
`1/Phi^-1(0.75) = 1.4826` so that a 3-sigma threshold really is three sigmas; without the constant every
threshold is 48% too tight. A MAD floor (a fraction of the training MAD) prevents the pathological case
where half the window holds one repeated number, the MAD collapses to zero, and every later sample is
infinitely anomalous.

**Mahalanobis distance.** The sample covariance of industrial tag data is close to singular: mirrored
historian tags, redundant transmitters and channels that are algebraic functions of others are all
normal. Inverting it amplifies noise without bound. The Ledoit-Wolf estimator shrinks towards a scaled
identity with the intensity that minimises expected squared error,

```
S_hat = (1-L)*S + L*m*I,    m = trace(S)/p,    L* = min(1, b^2 / d^2)
```

where `d^2 = ||S - m*I||_F^2` is the structure and `b^2` the estimation noise in `S`. Because the model
is Gaussian the distance has a distribution — `d^2 ~ chi^2_p` — so `theory_threshold(1e-3)` is a real
per-sample false-alarm rate rather than a tuned constant. The `chatter` demonstration prints it next to
the empirical training quantile, and the gap between the two is a direct measure of how badly the
Gaussian assumption is failing.

**PCA reconstruction error (SPE, the Q statistic).** Standardise, keep the components explaining
`variance_target` of the variance, and measure the squared distance to the retained subspace. At least
one component is always dropped: a full-rank basis reconstructs everything perfectly and the residual is
identically zero. SPE and Hotelling T-squared are returned together because they answer different
questions — a large excursion *along* the model is a T-squared event with almost no reconstruction
error, and a relationship violation is the reverse. In this benchmark the bearing temperature earns its
own principal component (its thermal lag decorrelates it from instantaneous load), so SPE is blind to
bearing-only faults. That is a property of the statistic rather than a bug, and the test suite pins it
down.

**Isolation depth.** An isolation forest with the standard normalisation `s(x) = 2^(-E[h(x)]/c(n))`,
where `c(n) = 2*H(n-1) - 2*(n-1)/n` is the expected unsuccessful-search path length in a random binary
tree. Deterministic given its seed, because a monitoring system that reports different scores on a rerun
cannot be audited.

## From scores to alarms

This layer decides whether anyone keeps the system switched on, and it is usually the missing half of a
detection paper. At a five-minute cadence a six-hour fault produces 72 consecutive exceedances; an
operator who receives 72 notifications for one event suppresses the tag by lunchtime, and after that the
quality of the detector is irrelevant.

- **Static threshold** — a quantile of the clean training scores. Stationary and auditable; its
  false-alarm rate rises as the plant ages.
- **Adaptive threshold** — a rolling quantile of recent scores. Keeps the false-alarm rate flat for
  months, and by construction a drift slow relative to the score noise never crosses it. `python -m msad
  thresholds` shows the cost; `tests/test_threshold.py` asserts it as a property.
- **`exclude_alarms`** — alarming samples are kept out of the adaptive baseline. Without it an ongoing
  fault feeds its own scores into the window that judges it, the threshold climbs to meet them, and the
  alarm clears itself while the fault is still running.
- **Hysteresis and dwell time** — `min_duration` consecutive exceedances to raise, a drop below
  `exit_ratio * threshold` sustained for `clear_duration` samples to clear, plus an optional cooldown.
  One alarm per fault instead of one per sample, and an isolated spurious sample never becomes an alarm.

## Evaluation, and one metric to distrust

`msad.evaluate` computes three views:

1. **Event-level detection with latency and false alarms per day.** What decides a deployment: did we
   catch the fault, how many minutes after it began, and how often do we cry wolf per shift. An alarm
   already standing when the fault starts counts as a zero-latency detection, not as both a false alarm
   and a miss.
2. **Point-wise F1.** Honest and harsh, treating every sample as an independent classification.
3. **Point-adjusted F1.** If *any* sample inside a labelled interval is flagged, the whole interval is
   credited. This is what most published comparisons report and it is close to meaningless: `python -m
   msad metrics` scores a detector that flags samples at random under it, and `tests/test_benchmark.py`
   asserts the same inflation on the real labels. Kim et al., *Towards a Rigorous Evaluation of
   Time-series Anomaly Detection* (AAAI 2022), makes the argument at length. Nothing in this repository's
   conclusions uses it.

## The benchmark

Ten days of six-channel pump telemetry at five-minute cadence (2880 samples), four clean days for
training, five injected faults, deterministic given the seed. Public benchmarks in this area have
well-documented label problems — several are solved by a three-line threshold rule, which is how the
literature acquired 0.99 F1 scores that mean nothing — so the process is simulated, with the *structure*
rather than the numbers as the point: a latent three-shift load with smooth ramps and AR(1) noise, six
channels with different gains and noise levels, and a bearing temperature that follows a first-order
thermal lag instead of instantaneous load.

| Fault | What happens | Why it is there |
|---|---|---|
| `spike` | vibration jumps 8 mm/s for 3 samples | control case; everything catches it |
| `level_shift` | bearing runs 18 K hot for 6 h | a rolling univariate baseline is enough |
| `correlation_break` | discharge pressure decoupled from load for 8 h, held near its daily mean | **every channel stays in range** — the reason this repository exists |
| `slow_drift` | bearing ramps 4 K over 12 h | at the edge of detectability; late if caught at all |
| `stuck_sensor` | flow transmitter frozen for 8 h from a shift change | a plausible value that stops agreeing with the machine |

The structural claim is a test rather than a paragraph:
`test_the_hidden_faults_never_leave_their_own_normal_range` asserts that during the correlation break and
the stuck transmitter, every reading of the faulted channel lies inside the min/max that channel showed
during clean operation. No per-channel limit, however chosen, can separate them.

## Running it

```bash
python -m msad detect        # all four detectors: events caught, latency, false alarms/day
python -m msad blindspots    # which fault each detector can see at all, before thresholding
python -m msad thresholds    # static vs adaptive, and what adapting costs on a slow drift
python -m msad chatter       # raw threshold crossings vs debounced alarms
python -m msad metrics       # point-adjusted F1 scored by a coin flip
python -m msad math          # Welford, MAD constant, chi-square tables, Jacobi, shrinkage

pytest                       # about two minutes; test_benchmark.py fits every detector end to end
```

Nothing beyond the standard library is required; `pytest` is needed only for the test suite.

## Numerical choices worth defending

- **Welford recurrence** for running variance. The textbook shortcut `E[x^2] - E[x]^2` on a bearing
  temperature near 340 K with 0.35 K of spread subtracts two numbers that agree to six digits, and keeps
  a few significant digits of the answer at best; `python -m msad math` prints both against a two-pass
  reference.
- **Jacobi rotations** for the eigen-decomposition rather than the power method: symmetric,
  unconditionally convergent, and it returns a genuinely orthonormal basis — which the tests verify,
  along with conservation of trace and determinant.
- **Cholesky with escalating jitter**, and the jitter is *reported*. Needing it is a message about the
  tag list (two channels are almost certainly duplicates), and silently repairing the matrix hides that.
- **`fsum` for accumulation** in the inner products and squared-error sums, where terms differ in
  magnitude by orders.

## Known limits

- **The data is simulated.** The faults are realistic in structure and were placed by hand so each is
  detectable in principle; that is not a substitute for a labelled plant dataset.
- **Shrinkage towards a scaled identity is not scale-invariant.** With channel variances spanning four
  orders of magnitude (vibration in mm/s next to pressure in kPa), `L*m*I` inflates the smallest
  channel's variance far more than the largest, costing sensitivity there. Standardising before
  estimating the covariance removes this; the raw-scale version is kept because it is what the
  Ledoit-Wolf derivation applies to, and the cost is stated rather than hidden.
- **One operating-mode family is assumed.** A machine with genuinely multi-modal normal behaviour (batch
  recipes, seasonal setpoints) needs a mixture or a per-mode model; one covariance will treat the gap
  between modes as normal and the modes themselves as marginal.
- **Cubic-time factorisations.** Fine for tens of tags, wrong for thousands; at that size the residual
  subspace should be maintained incrementally.
- **No explicit seasonality model.** The rolling baselines absorb daily structure implicitly; weekly or
  campaign-level structure would need handling of its own.

## Layout

```
msad/
  types.py       score series, labelled events, dataset container
  stats.py       Welford, quantiles, MAD, rolling window, debiased EWMA, correlation
  chisq.py       incomplete gamma -> chi-square CDF and quantile function
  linalg.py      covariance, Ledoit-Wolf shrinkage, Cholesky, Mahalanobis, Jacobi eigen
  detectors.py   the four detectors and the training-quantile calibration helper
  threshold.py   static and adaptive thresholds, hysteresis, dwell time, alarm events
  evaluate.py    event-level scores, point-wise F1, and the inflated metric for contrast
  simulate.py    the six-channel pump and its five graded faults
  pipeline.py    fit -> score -> threshold -> debounce -> evaluate, in that order
  cli.py         the six demonstrations
tests/           numerical layer, detector contracts, alarm logic, and the benchmark claims
```

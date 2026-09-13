"""Demonstrations, one per claim.

``detect``      all four detectors on the ten-day benchmark: events caught, latency, false alarms
``blindspots``  which fault each detector can see *at all*, independent of thresholding
``thresholds``  static vs adaptive: what adapting costs you on a slow drift
``chatter``     raw threshold crossings vs debounced alarms
``metrics``     point-adjusted F1 scored by a coin flip
``math``        the numerical machinery checked against closed-form and published values
"""

from __future__ import annotations

import argparse
import math
import random
import sys

from . import simulate
from .chisq import chi2_cdf, chi2_ppf
from .detectors import (
    MahalanobisDetector,
    PCAReconstructionDetector,
    training_quantile_threshold,
)
from .evaluate import point_adjusted_f1, point_wise_f1, score_separation
from .linalg import covariance_matrix, jacobi_eigen, ledoit_wolf_intensity
from .pipeline import default_detectors, run_detector
from .stats import MAD_TO_SIGMA, Welford, mad, variance
from .threshold import HysteresisPolicy, exceedance_mask

RULE = "=" * 96


def _heading(text: str) -> None:
    print(f"\n{RULE}\n{text}\n{RULE}")


def cmd_detect(args) -> None:
    """All four detectors on the ten-day benchmark."""
    dataset = simulate.build_dataset()
    _heading(
        f"{len(dataset)} samples of {dataset.width}-channel telemetry at "
        f"{simulate.STEP_MINUTES}-minute cadence; {dataset.train_end} clean samples for training; "
        f"{len(dataset.events)} injected faults"
    )
    print(f"  {'detector':<17}{'events':<9}{'latency':<11}{'false alarms':<15}{'point F1':<11}pa F1")
    print("  " + "-" * 76)
    for detector in default_detectors():
        result = run_detector(dataset, detector)
        caught = sum(1 for hit in result.events.detected.values() if hit)
        latencies = [v for v in result.events.latency_minutes.values() if v is not None]
        latency = f"{sum(latencies) / len(latencies):.0f} min" if latencies else "n/a"
        print(
            f"  {result.detector:<17}{caught}/{len(dataset.events):<7}{latency:<11}"
            f"{result.events.false_alarms_per_day:.2f}/day{'':<7}"
            f"{result.point.f1:<11.3f}{result.point_adjusted.f1:.3f}"
        )
    print(
        "\nThe point-adjusted column is included only for contrast; see `metrics`. Judge these on\n"
        "events caught, latency and false alarms per day — the three numbers an operator would ask for."
    )


def cmd_blindspots(args) -> None:
    """Which faults each detector can see at all, before any thresholding."""
    dataset = simulate.build_dataset()
    _heading("Median score inside each fault, as a multiple of the detector's own alarm threshold")
    print(f"  {'fault':<20}" + "".join(f"{d.name:<18}" for d in default_detectors()))
    print("  " + "-" * 92)

    columns: dict[str, list[str]] = {}
    for detector in default_detectors():
        detector.fit(dataset.train())
        series = detector.score(dataset.rows)
        threshold = training_quantile_threshold(series, dataset.train_end, 0.999)
        cells = []
        for event in dataset.events:
            inside, _ = score_separation(series.scores, event, threshold)
            ratio = inside / threshold if threshold > 0 else math.inf
            verdict = "sees" if ratio > 1.0 else "BLIND"
            cells.append(f"{ratio:6.2f}x {verdict:<6}")
        columns[detector.name] = cells

    for position, event in enumerate(dataset.events):
        print(
            f"  {event.kind:<20}"
            + "".join(f"{columns[d.name][position]:<18}" for d in default_detectors())
        )

    print("\n  fault notes:")
    for event in dataset.events:
        print(f"    {event.kind:<20}{event.note}")
    print(
        "\nThe `correlation_break` row is the whole argument for multivariate monitoring. Discharge\n"
        "pressure is held near its normal daily average while the machine runs at high load: every\n"
        "channel stays inside its own range, so a per-channel detector has nothing to fire on, while\n"
        "the reading sits far off the load line that the covariance and the PCA residual know about.\n"
        "`stuck_sensor` is the same story with a frozen transmitter instead of a lost coupling, and\n"
        "`slow_drift` is hard for everything — which is stated rather than hidden."
    )


def cmd_thresholds(args) -> None:
    """What an adaptive threshold costs on a slow drift."""
    dataset = simulate.build_dataset()
    _heading("Static vs adaptive thresholding, same detector, same scores")
    for adaptive in (False, True):
        result = run_detector(dataset, PCAReconstructionDetector(), adaptive=adaptive)
        label = "adaptive" if adaptive else "static"
        drift = result.events.detected.get("slow_drift", False)
        print(
            f"  {label:<10}{result.events.summary()}\n"
            f"  {'':<10}slow_drift detected: {drift}"
        )
    print(
        "\nAn adaptive threshold is the right default for a plant that ages, and it is structurally\n"
        "weaker on a slow ramp: the quantile it tracks rises with the fault. Excluding alarming\n"
        "samples from the baseline is what stops an ongoing alarm from clearing itself — and it is\n"
        "also what stops the baseline from following a fault once the fault is visible. Real\n"
        "deployments run both kinds of threshold, which is why both are implemented here."
    )


def cmd_chatter(args) -> None:
    """Raw threshold crossings vs debounced alarm events."""
    dataset = simulate.build_dataset()
    _heading("Why hysteresis and dwell time exist")
    detector = MahalanobisDetector().fit(dataset.train())
    series = detector.score(dataset.rows)
    threshold = training_quantile_threshold(series, dataset.train_end, 0.999)
    thresholds = [threshold] * len(series)
    crossings = sum(exceedance_mask(series.scores, thresholds, dataset.train_end))

    print(
        f"  chi-square threshold (p={1e-3:g}, df={detector.degrees_of_freedom}): "
        f"{detector.theory_threshold(1e-3):.1f}"
    )
    print(f"  empirical 99.9th percentile of clean training scores:  {threshold:.1f}")
    print(f"  shrinkage intensity chosen by Ledoit-Wolf:             {detector.intensity:.4f}")
    print(f"  Cholesky jitter required:                              {detector.jitter:.2e}")
    print(f"\n  raw threshold crossings in the test window: {crossings}")
    for policy in (
        HysteresisPolicy(min_duration=1, clear_duration=1, exit_ratio=1.0),
        HysteresisPolicy(min_duration=3, clear_duration=6, exit_ratio=0.7),
        HysteresisPolicy(min_duration=6, clear_duration=12, exit_ratio=0.5),
    ):
        alarms = [
            alarm
            for alarm in policy.apply(series.scores, thresholds, series.valid_from)
            if alarm.end >= dataset.train_end
        ]
        print(
            f"    dwell={policy.min_duration:<2} clear={policy.clear_duration:<3} "
            f"exit={policy.exit_ratio:<4} -> {len(alarms)} alarms"
        )
    print(
        "\nThe gap between the two thresholds is diagnostic in itself: the chi-square number assumes a\n"
        "Gaussian baseline, the empirical one does not, and a large discrepancy means the assumption\n"
        "is not holding. The alarm counts are why nobody ships raw per-sample exceedances: an operator\n"
        "who receives dozens of notifications for one fault suppresses the tag, and then the quality\n"
        "of the detector stops mattering at all."
    )


def cmd_metrics(args) -> None:
    """Point-adjusted F1, scored by a coin flip."""
    dataset = simulate.build_dataset()
    _heading("A random detector under point-adjusted F1")
    labels = dataset.labels()
    rng = random.Random(4242)
    for rate in (0.005, 0.01, 0.02, 0.05):
        predictions = [
            1 if index >= dataset.train_end and rng.random() < rate else 0
            for index in range(len(dataset))
        ]
        masked = [
            label if index >= dataset.train_end else 0 for index, label in enumerate(labels)
        ]
        point = point_wise_f1(masked, predictions)
        adjusted = point_adjusted_f1(masked, predictions, dataset.events)
        print(
            f"  flags {rate:5.1%} of samples at random -> point-wise F1 {point.f1:.3f}   "
            f"point-adjusted F1 {adjusted.f1:.3f}"
        )
    print(
        "\nThat is the metric most published comparisons in this area report. One lucky sample inside a\n"
        "long interval credits the whole interval, so the score is dominated by how long the labelled\n"
        "events are rather than by anything the detector did. Nothing in this repository's conclusions\n"
        "uses it."
    )


def cmd_math(args) -> None:
    """The numerical machinery against closed-form and published values."""
    _heading("1. Welford vs the textbook shortcut, on data far from zero")
    base = 340.0
    values = [base + 0.5 * math.sin(i) for i in range(1000)]
    welford = Welford().extend(values)
    naive_mean = math.fsum(values) / len(values)
    naive = (
        math.fsum(value * value for value in values) / len(values) - naive_mean * naive_mean
    ) * len(values) / (len(values) - 1)
    print(f"  two-pass reference : {variance(values):.12e}")
    print(f"  Welford            : {welford.variance:.12e}")
    print(f"  E[x^2]-E[x]^2      : {naive:.12e}")
    print(
        "  A bearing temperature near 340 K with 0.35 K of spread means the shortcut subtracts two\n"
        "  numbers that agree to six digits. It keeps a few significant digits of the answer at best."
    )

    _heading("2. The MAD consistency constant")
    rng = random.Random(11)
    sample = [rng.gauss(0.0, 3.0) for _ in range(200000)]
    print(f"  1/Phi^-1(0.75)          = {MAD_TO_SIGMA:.6f}")
    print(f"  scaled MAD of N(0, 3^2) = {mad(sample):.4f}   (sigma = 3)")
    print(f"  unscaled MAD            = {mad(sample, scaled=False):.4f}   (0.6745 * 3 = 2.0235)")

    _heading("3. Chi-square quantiles vs published tables")
    for level, df, published in (
        (0.95, 1, 3.8415),
        (0.99, 2, 9.2103),
        (0.95, 5, 11.0705),
        (0.99, 10, 23.2093),
    ):
        computed = chi2_ppf(level, df)
        print(
            f"  chi2_ppf({level}, df={df:<2}) = {computed:9.4f}   table {published:9.4f}   "
            f"round trip cdf = {chi2_cdf(computed, df):.6f}"
        )

    _heading("4. Jacobi eigen-decomposition of a known matrix")
    matrix = [[4.0, 1.0, 1.0], [1.0, 3.0, 0.0], [1.0, 0.0, 2.0]]
    eigen = jacobi_eigen(matrix)
    print(f"  eigenvalues: {[round(value, 8) for value in eigen.values]}")
    print(f"  trace preserved: {sum(eigen.values):.10f} vs {4.0 + 3.0 + 2.0}")
    product = 1.0
    for value in eigen.values:
        product *= value
    # det = 4*(3*2 - 0) - 1*(1*2 - 0) + 1*(0 - 3) = 24 - 2 - 3 = 19
    print(f"  determinant preserved: {product:.10f} vs 19.0")
    orthogonality = max(
        abs(
            math.fsum(a * b for a, b in zip(eigen.vectors[i], eigen.vectors[j]))
            - (1.0 if i == j else 0.0)
        )
        for i in range(3)
        for j in range(3)
    )
    print(f"  worst deviation from orthonormality: {orthogonality:.2e} ({eigen.sweeps} sweeps)")

    _heading("5. Shrinkage on a near-duplicate tag pair")
    rng = random.Random(5)
    honest = [[rng.gauss(0, 1), rng.gauss(0, 1)] for _ in range(400)]
    duplicated = [[row[0], row[0] + 1e-6 * rng.gauss(0, 1)] for row in honest]
    print(f"  independent channels : intensity {ledoit_wolf_intensity(honest):.4f}")
    print(f"  duplicated channel   : intensity {ledoit_wolf_intensity(duplicated):.4f}")
    print(f"  sample covariance of the duplicated pair: {covariance_matrix(duplicated)}")
    detector = MahalanobisDetector().fit(duplicated)
    print(
        f"  Mahalanobis still factors: jitter {detector.jitter:.2e}, "
        f"intensity {detector.intensity:.4f}"
    )
    print(
        "  A mirrored historian tag makes the sample covariance singular. Without shrinkage the\n"
        "  inverse amplifies noise without bound and ordinary readings score as extreme outliers."
    )


COMMANDS = {
    "detect": cmd_detect,
    "blindspots": cmd_blindspots,
    "thresholds": cmd_thresholds,
    "chatter": cmd_chatter,
    "metrics": cmd_metrics,
    "math": cmd_math,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="msad", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, handler in COMMANDS.items():
        subparser = subparsers.add_parser(name, help=(handler.__doc__ or "").strip().split("\n")[0])
        subparser.set_defaults(handler=handler)
    args = parser.parse_args(argv)
    args.handler(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

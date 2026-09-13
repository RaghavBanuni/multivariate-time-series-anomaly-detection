"""A synthetic pump skid, and five faults chosen to have different detectability.

Real industrial telemetry with trustworthy labels cannot be shipped, and the public benchmarks in this
area are famously flawed — several have labels a three-line threshold rule solves, which is how the
literature ended up with 0.99 F1 scores that mean nothing. So the process here is simulated, but the
*structure* is the point rather than the numbers:

- A latent **load** follows a three-shift schedule with smooth ramps between setpoints and an AR(1)
  noise term — so consecutive samples are correlated, as they are on a real machine, and a detector
  cannot rely on independence.
- Six channels are driven by that load with different gains and noise levels, so the normal operating
  region is a *line* in six-dimensional space, not a box. This is what makes univariate limits
  structurally inadequate: a reading can sit inside every individual range and nowhere near the line.
- Bearing temperature follows a **lagged** load (a first-order thermal response with roughly a four-hour
  time constant), so the relationship is not instantaneous and the residual structure is not trivially
  rank-one. In PCA terms the bearing earns its own principal component, which later turns out to be why
  the reconstruction error is blind to bearing-only faults.

The five faults, in increasing subtlety:

1. ``spike`` — three samples of vibration jump 8 mm/s. Everything catches this; it is the control case.
2. ``level_shift`` — bearing temperature runs 18 K hot for six hours. A rolling univariate baseline
   catches it comfortably.
3. ``correlation_break`` — discharge pressure is **decoupled** from load for eight hours: it holds
   around its normal daily average while the machine runs at high load. Every channel stays inside its
   own normal range — the test suite asserts this — so univariate monitoring is blind by construction,
   while the multivariate detectors see a reading far off the load line. This fault is the reason the
   repository exists.
4. ``slow_drift`` — bearing temperature ramps 4 K over twelve hours. Deliberately at the edge of
   detectability: caught late if at all, and an adaptive threshold follows the ramp and never fires.
5. ``stuck_sensor`` — the flow transmitter freezes at its last value for eight hours starting at a
   shift change. The frozen number is perfectly plausible; only its growing disagreement with the rest
   of the machine gives it away.

Everything is deterministic given the seed.
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone

from .types import AnomalyEvent, Dataset

SENSORS = (
    "motor_current",
    "suction_pressure",
    "discharge_pressure",
    "flow",
    "vibration",
    "bearing_temp",
)
INDEX = {name: position for position, name in enumerate(SENSORS)}

STEP_MINUTES = 5
STEP = timedelta(minutes=STEP_MINUTES)
SAMPLES_PER_DAY = (24 * 60) // STEP_MINUTES  # 288
START = datetime(2026, 3, 1, tzinfo=timezone.utc)

# Instrument noise, per channel, in engineering units. Small relative to the load-driven swing, as on
# real transmitters: the interesting variation in process data is the process, not the sensor.
NOISE = {
    "motor_current": 0.25,
    "suction_pressure": 0.25,
    "discharge_pressure": 1.2,
    "flow": 0.6,
    "vibration": 0.03,
    "bearing_temp": 0.30,
}


def shift_setpoint(minute_of_day: int) -> float:
    """Three shifts: day hard, evening moderate, night light."""
    if 360 <= minute_of_day < 840:  # 06:00-14:00
        return 0.90
    if 840 <= minute_of_day < 1320:  # 14:00-22:00
        return 0.70
    return 0.50


def clean_rows(days: int, seed: int) -> list[list[float]]:
    """Fault-free telemetry: one latent load, six correlated channels, one thermal lag."""
    rng = random.Random(seed)
    n = days * SAMPLES_PER_DAY
    rows: list[list[float]] = []

    load_state = 0.5
    ar_noise = 0.0
    heat = 0.5  # lagged load driving bearing temperature

    for index in range(n):
        minute_of_day = (index * STEP_MINUTES) % 1440
        target = shift_setpoint(minute_of_day)
        load_state += 0.08 * (target - load_state)  # ramps between setpoints, no instant steps
        ar_noise = 0.85 * ar_noise + rng.gauss(0.0, 0.008)
        load = min(1.05, max(0.30, load_state + ar_noise))
        heat += 0.02 * (load - heat)  # first-order thermal response, ~4 h time constant

        suction = 90.0 - 10.0 * load + rng.gauss(0.0, NOISE["suction_pressure"])
        rows.append(
            [
                40.0 + 60.0 * load + rng.gauss(0.0, NOISE["motor_current"]),  # A
                suction,  # kPa
                300.0 + 180.0 * load + rng.gauss(0.0, NOISE["discharge_pressure"]),  # kPa
                100.0 * load
                + 0.4 * (suction - 85.0)
                + rng.gauss(0.0, NOISE["flow"]),  # m3/h
                2.0 + 1.2 * load + rng.gauss(0.0, NOISE["vibration"]),  # mm/s RMS
                320.0 + 25.0 * heat + rng.gauss(0.0, NOISE["bearing_temp"]),  # K
            ]
        )
    return rows


def _spike(rows, start: int, duration: int, sensor: str, magnitude: float) -> AnomalyEvent:
    position = INDEX[sensor]
    for index in range(start, start + duration):
        rows[index][position] += magnitude
    return AnomalyEvent(
        start=start,
        end=start + duration - 1,
        kind="spike",
        sensors=(sensor,),
        note=f"{sensor} jumps {magnitude:+.1f} for {duration} samples",
    )


def _level_shift(rows, start: int, duration: int, sensor: str, magnitude: float) -> AnomalyEvent:
    position = INDEX[sensor]
    for index in range(start, start + duration):
        rows[index][position] += magnitude
    return AnomalyEvent(
        start=start,
        end=start + duration - 1,
        kind="level_shift",
        sensors=(sensor,),
        note=f"{sensor} runs {magnitude:+.1f} high for {duration * STEP_MINUTES / 60:.0f} h",
    )


def _correlation_break(
    rows,
    start: int,
    duration: int,
    sensor: str,
    hold_value: float,
    noise: float,
    rng: random.Random,
) -> AnomalyEvent:
    """Decouple one channel from the load while keeping it inside its own normal range.

    Noise is retained deliberately: a *frozen* value would be caught by a variance check, and the fault
    being modelled is a lost relationship (a stuck control valve, a worn impeller), not a dead
    transmitter. That distinction is what makes this the hardest of the five.
    """
    position = INDEX[sensor]
    for index in range(start, start + duration):
        rows[index][position] = hold_value + rng.gauss(0.0, noise)
    return AnomalyEvent(
        start=start,
        end=start + duration - 1,
        kind="correlation_break",
        sensors=(sensor,),
        note=f"{sensor} decoupled from load, held near its daily mean ({hold_value:.0f})",
    )


def _drift(rows, start: int, duration: int, sensor: str, total: float) -> AnomalyEvent:
    position = INDEX[sensor]
    for step, index in enumerate(range(start, start + duration)):
        rows[index][position] += total * (step + 1) / duration
    return AnomalyEvent(
        start=start,
        end=start + duration - 1,
        kind="slow_drift",
        sensors=(sensor,),
        note=f"{sensor} ramps {total:+.1f} over {duration * STEP_MINUTES / 60:.0f} h",
    )


def _stuck(rows, start: int, duration: int, sensor: str) -> AnomalyEvent:
    position = INDEX[sensor]
    frozen = rows[start - 1][position]
    for index in range(start, start + duration):
        rows[index][position] = frozen
    return AnomalyEvent(
        start=start,
        end=start + duration - 1,
        kind="stuck_sensor",
        sensors=(sensor,),
        note=f"{sensor} transmitter frozen at {frozen:.1f} from a shift change onwards",
    )


def build_dataset(
    days: int = 10, train_days: int = 4, seed: int = 20260301, with_faults: bool = True
) -> Dataset:
    """The benchmark: ten days at five-minute cadence, four clean days for training.

    Fault windows are placed by hand rather than at random, for two reasons. The correlation break has
    to sit inside a *high-load* shift or it is not a break at all — holding pressure at its daily mean
    during the night shift would be almost correct. And the stuck transmitter has to begin at a shift
    change, or the frozen value stays coincidentally right for hours. A randomly placed fault would
    sometimes be undetectable in principle, which destabilises a benchmark for reasons that have nothing
    to do with the detectors.
    """
    rows = clean_rows(days, seed)
    train_end = train_days * SAMPLES_PER_DAY
    events: list[AnomalyEvent] = []

    if with_faults:
        rng = random.Random(seed + 999)
        pressure_mean = (
            math.fsum(row[INDEX["discharge_pressure"]] for row in rows[:train_end]) / train_end
        )

        events.append(_spike(rows, 4 * SAMPLES_PER_DAY + 144, 3, "vibration", 8.0))
        events.append(_level_shift(rows, 5 * SAMPLES_PER_DAY, 72, "bearing_temp", 18.0))
        events.append(
            _correlation_break(
                rows,
                6 * SAMPLES_PER_DAY + 72,  # 06:00, the start of the hard day shift
                96,  # 8 h
                "discharge_pressure",
                pressure_mean,
                NOISE["discharge_pressure"],
                rng,
            )
        )
        events.append(_drift(rows, 7 * SAMPLES_PER_DAY + 216, 144, "bearing_temp", 4.0))
        events.append(
            _stuck(rows, 8 * SAMPLES_PER_DAY + 168, 96, "flow")  # 14:00, at the shift change
        )

    return Dataset(
        sensors=SENSORS,
        rows=rows,
        start=START,
        step=STEP,
        events=events,
        train_end=train_end,
    )


def channel_range(rows, sensor: str) -> tuple[float, float]:
    """Min/max of one channel, used to demonstrate that a fault stayed inside its own limits."""
    position = INDEX[sensor]
    values = [row[position] for row in rows]
    return min(values), max(values)

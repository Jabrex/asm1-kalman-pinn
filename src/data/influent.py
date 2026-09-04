"""Synthetic influent generator: diurnal flow and load, plus a rain scenario.

Provenance rule for this module
-------------------------------
Every *magnitude* is anchored to the BSM1 technical report:

* daily mean flow and all mean concentrations - BSM1 Table 5 (report p.13);
* observed daily min/max of flow, S_S, S_NH and X_S - BSM1 Figure 3 (p.12);
* rain-event peak flow - BSM1 Figure 5 (p.13);
* ``S_O = S_NO = X_B_A = X_P = 0`` and ``S_ALK = 7`` - BSM1 section 2.4 (p.11),
  which states these hold "in any influent".

Everything else in here is *shape*, not magnitude: where the morning and evening
peaks sit, how sharp they are, and how far the load peak leads the flow peak.
Those are collected in :class:`ProfileShape` and are explicitly NOT sourced -
they set the form of the daily curve while the anchors above pin its level and
its extremes. The two numbers read off a figure rather than a table
(``flow_min``/``flow_max``, ``weekend_peak_factor``) are marked as such.

Construction of a diurnal signal
--------------------------------
For a signal with published mean ``M``, minimum ``m`` and maximum ``X``::

    s(t)  in [0, 1]     daily shape mapped AFFINELY onto [0, 1]
    u(t)  = 2*s(t)**p - 1        warped shape, span exactly 2 for every p
    b     = ln(X/m) / 2          fixes the max/min ratio
    p                            solved so the absolute maximum equals X
    x(t)  = M * exp(b*u(t)) / <exp(b*u)>_day     fixes the mean

Three published statistics, three constraints, and they are all met exactly:

* the **mean** by the ``<exp(b*u)>`` normalisation;
* the **max/min ratio** by ``b``, because the affine map guarantees ``u`` spans
  exactly ``[-1, 1]`` (normalising by ``max|raw|`` would pin only one end, since
  the double-peak shape is asymmetric, and the ratio would fall short);
* the **absolute extremes** by ``p``. With the mean and the ratio already fixed,
  the extremes still depend on how much of the day the curve spends near its
  peak; a plain two-harmonic curve sits mid-range too long and both extremes
  fall short. ``p`` sharpens the peaks until the maximum is met, and the minimum
  then follows from the ratio. The warp is monotone, so peak *timing* is
  untouched.

Flow additionally carries a weekday/weekend factor, and its observed 14-day
range is the product of the two, so its exponent is reduced by half the log of
the weekly spread.

``p`` is a shape parameter, but a *derived* one: it is solved from the published
figures rather than chosen. :meth:`InfluentGenerator.summary` reports every
achieved statistic next to its anchor so any residual deviation stays visible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from ..asm1.vault_loader import Asm1Vault, vault

# --- BSM1 anchors ---------------------------------------------------------
#: BSM1 Table 5, load averages for the stabilisation period (report p.13).
BSM1_TABLE5_MEAN: Mapping[str, float] = {
    "S_I": 30.00,
    "S_S": 69.50,
    "X_I": 51.20,
    "X_S": 202.32,
    "X_B_H": 28.17,
    "X_B_A": 0.0,
    "X_P": 0.0,
    "S_O": 0.0,
    "S_NO": 0.0,
    "S_NH": 31.56,
    "S_ND": 6.95,
    "X_ND": 10.59,
    "S_ALK": 7.00,
    "S_N2": 0.0,  # not part of BSM1; the vault carries S_N2 and influent N2 is nil
}
BSM1_TABLE5_FLOW = 18446.0  # m3/d

#: Read off BSM1 Figure 3 (report p.12). Figure readings, not table values.
BSM1_FIG3_FLOW_RANGE = (10000.0, 32000.0)          # m3/d
BSM1_FIG3_CONC_RANGE: Mapping[str, tuple[float, float]] = {
    "S_S": (55.0, 120.0),
    "S_NH": (15.0, 45.0),
    "X_S": (100.0, 300.0),
}
#: Weekend peaks are visibly lower than weekday peaks in BSM1 Figure 3.
#: This factor is a figure reading, not a published number.
BSM1_FIG3_WEEKEND_PEAK_FACTOR = 0.85

#: Rain-event peak flow, read off BSM1 Figure 5 (report p.13).
BSM1_FIG5_RAIN_PEAK_FLOW = 52000.0  # m3/d

#: Components held constant in any influent - BSM1 section 2.4 (report p.11).
BSM1_CONSTANT_COMPONENTS = ("S_O", "S_NO", "X_B_A", "X_P", "S_ALK", "S_N2")

#: Family assignment for components without their own published diurnal range.
#: Each family follows the normalised shape of its published representative.
COMPONENT_FAMILY: Mapping[str, str] = {
    "S_I": "S_S",
    "S_S": "S_S",
    "S_NH": "S_NH",
    "S_ND": "S_NH",
    "X_I": "X_S",
    "X_S": "X_S",
    "X_B_H": "X_S",
    "X_ND": "X_S",
}


@dataclass(frozen=True)
class ProfileShape:
    """Daily-curve shape parameters. NOT sourced - these set form, not level."""

    first_harmonic_phase: float = 1.4    # rad, places the morning peak
    second_harmonic_weight: float = 0.45  # relative weight of the 12 h harmonic
    second_harmonic_phase: float = 0.6   # rad, splits morning and evening peaks
    load_lead_hours: float = 1.5         # load peak leads the flow peak
    weekend_days: tuple[int, ...] = (5, 6)  # 0 = first simulated day


@dataclass(frozen=True)
class RainEvent:
    """Rain hydrograph superimposed on the dry-weather series (test set only).

    ``peak_flow`` is the published Figure 5 reading. ``start_day`` and
    ``duration_days`` are also read off that figure - the event sits in the
    second week and runs roughly from day 8 to day 11 - so they are figure
    readings, not published numbers, exactly like ``flow_min``/``flow_max``.
    ``rise_fraction`` is NOT sourced: it is a shape parameter setting how much of
    the event is spent rising, and it does not affect the peak or the duration.
    """

    start_day: float = 8.0        # figure reading, BSM1 Fig. 5 (report p.13)
    duration_days: float = 3.0    # figure reading, BSM1 Fig. 5 (report p.13)
    peak_flow: float = BSM1_FIG5_RAIN_PEAK_FLOW
    rise_fraction: float = 0.25   # NOT sourced - shape only


@dataclass(frozen=True)
class InfluentSpec:
    duration_days: float = 14.0
    flow_mean: float = BSM1_TABLE5_FLOW
    flow_range: tuple[float, float] = BSM1_FIG3_FLOW_RANGE
    weekend_peak_factor: float = BSM1_FIG3_WEEKEND_PEAK_FACTOR
    shape: ProfileShape = field(default_factory=ProfileShape)
    rain: RainEvent | None = None


def _ratio_exponent(low: float, high: float) -> float:
    """b such that exp(b) / exp(-b) equals the observed max/min ratio."""
    if low <= 0.0:
        raise ValueError("A diurnal range must have a strictly positive minimum")
    return 0.5 * np.log(high / low)


#: Search bracket for the peakiness exponent (see ``_solve_peakiness``).
_PEAKINESS_BRACKET = (1e-3, 1e3)


def _solve_peakiness(
    unit_shape: np.ndarray, b: float, mean: float, peak: float, peak_factor: float = 1.0
) -> float:
    """Sharpen or flatten the daily peaks until the published maximum is met.

    With the mean and the max/min ratio both pinned, the absolute extremes are
    still free: they depend on how much of the day the signal spends near its
    peak. A smooth two-harmonic curve sits near mid-range too long, which pulls
    both extremes toward the mean and leaves them short of the figure reading.

    Warping the unit shape ``s`` in ``[0, 1]`` to ``s**p`` fixes that without
    touching anything already anchored: the warp is monotone, so peak *timing*
    is unchanged; the mean stays exact because it is renormalised afterwards;
    and the ratio stays exact because it depends only on ``b``. Raising ``p``
    concentrates time near the trough, which lifts both extremes together, so
    solving for the published maximum lands the published minimum as well.

    ``p`` is a shape parameter, but a derived one - it is solved from the
    published mean and range rather than chosen.
    """
    def achieved_peak(p: float) -> float:
        u = 2.0 * unit_shape ** p - 1.0
        norm = float(np.mean(np.exp(b * u)))
        return mean * np.exp(b) * peak_factor / norm

    lo, hi = _PEAKINESS_BRACKET
    reachable = (achieved_peak(lo), achieved_peak(hi))
    if not min(reachable) <= peak <= max(reachable):
        raise ValueError(
            "Published peak %.4g is unreachable for mean %.4g and ratio %.4g "
            "(attainable range %.4g to %.4g). The published mean and range are "
            "mutually inconsistent for a single-peaked daily curve."
            % (peak, mean, np.exp(2.0 * b), min(reachable), max(reachable))
        )
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if achieved_peak(mid) < peak:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class InfluentGenerator:
    """Callable influent provider compatible with :data:`src.asm1.plant.InfluentFn`."""

    def __init__(self, spec: InfluentSpec | None = None, source: Asm1Vault | None = None) -> None:
        self.spec = spec if spec is not None else InfluentSpec()
        self.vault = source if source is not None else vault()
        self.components = self.vault.components

        missing = set(self.components) - set(BSM1_TABLE5_MEAN)
        if missing:
            raise ValueError("No influent anchor for components: %s" % sorted(missing))

        # The raw double-peak shape is asymmetric: its peak and its trough are
        # not the same height. Dividing by max|raw| would pin only one end, and
        # the achieved max/min ratio would come out as exp(b * span) with
        # span < 2. Map affinely onto [-1, 1] instead, so both ends are pinned
        # and span is exactly 2. The mean of u is then non-zero, which does not
        # matter: the exp-mean normalisation below fixes the mean afterwards.
        # The extremes are phase independent in the continuum, so a lead-shifted
        # evaluation has the same min and max. On a discrete grid it does not:
        # a shifted sample can land a hair outside the unshifted grid's range.
        # A fine grid keeps that excursion at the 1e-8 level and _unit_shape
        # clips what is left, which matters because a fractional power of a
        # negative base is NaN.
        grid = np.linspace(0.0, 1.0, 20001, endpoint=False)
        raw = self._raw_shape(grid, 0.0)
        self._raw_min = float(np.min(raw))
        self._raw_span = float(np.max(raw) - self._raw_min)
        if self._raw_span <= 0.0:
            raise ValueError("The daily shape is flat; check ProfileShape")
        self._grid = grid

        # Flow carries a weekday/weekend factor on top of the daily curve, so the
        # observed 14-day range is the product of the two. Discount the weekly
        # spread from the daily exponent, otherwise the full-series ratio
        # overshoots the figure reading.
        self._weekly = self._weekly_factors()
        weekly_ratio = float(np.max(self._weekly) / np.min(self._weekly))
        self._flow_b = _ratio_exponent(*self.spec.flow_range) - 0.5 * np.log(weekly_ratio)
        if self._flow_b <= 0.0:
            raise ValueError(
                "The weekend factor alone exceeds the observed flow range; "
                "weekly ratio %.4f vs flow range ratio %.4f"
                % (weekly_ratio, self.spec.flow_range[1] / self.spec.flow_range[0])
            )
        # Peakiness is solved per signal so that the published mean, minimum and
        # maximum are all met, not just two of the three.
        unit = self._unit_shape(grid, 0.0)
        self._flow_p = _solve_peakiness(
            unit, self._flow_b, self.spec.flow_mean, self.spec.flow_range[1],
            peak_factor=float(np.max(self._weekly)),
        )
        self._flow_norm = self._exp_mean(self._flow_b, 0.0, self._flow_p)

        # Concentrations carry no weekly factor, so their exponent is the plain
        # half-log of the published range.
        self._conc_b: dict[str, float] = {}
        self._conc_p: dict[str, float] = {}
        self._conc_norm: dict[str, float] = {}
        lead = self.spec.shape.load_lead_hours / 24.0
        unit_led = self._unit_shape(grid, lead)
        for family, (low, high) in BSM1_FIG3_CONC_RANGE.items():
            b = _ratio_exponent(low, high)
            p = _solve_peakiness(unit_led, b, BSM1_TABLE5_MEAN[family], high)
            self._conc_b[family] = b
            self._conc_p[family] = p
            self._conc_norm[family] = self._exp_mean(b, lead, p)

        self._rain_gain = self._solve_rain_gain()

    # -- shape helpers -----------------------------------------------------
    def _raw_shape(self, t: np.ndarray | float, lead: float) -> np.ndarray:
        sh = self.spec.shape
        tau = np.asarray(t, dtype=float) + lead
        return np.sin(2 * np.pi * tau - sh.first_harmonic_phase) + sh.second_harmonic_weight * np.sin(
            4 * np.pi * tau - sh.second_harmonic_phase
        )

    def _unit_shape(self, t: np.ndarray | float, lead: float) -> np.ndarray:
        """Daily shape mapped affinely onto ``[0, 1]``, before the peakiness warp.

        Clipped because ``s ** p`` with fractional ``p`` is NaN for a negative
        base, and a discrete grid can place a phase-shifted sample a few times
        1e-9 outside the range measured on the unshifted grid.
        """
        s = (self._raw_shape(t, lead) - self._raw_min) / self._raw_span
        return np.clip(s, 0.0, 1.0)

    def _shape(self, t: np.ndarray | float, lead: float, p: float) -> np.ndarray:
        """Warped daily shape on ``[-1, 1]``; span exactly 2 for every ``p``."""
        return 2.0 * self._unit_shape(t, lead) ** p - 1.0

    def _exp_mean(self, b: float, lead: float, p: float) -> float:
        return float(np.mean(np.exp(b * self._shape(self._grid, lead, p))))

    def _weekly_factors(self) -> np.ndarray:
        """Per-day multipliers, renormalised so the weekly mean is exactly one."""
        weekend = set(self.spec.shape.weekend_days)
        pattern = np.array(
            [self.spec.weekend_peak_factor if d in weekend else 1.0 for d in range(7)]
        )
        pattern /= pattern.mean()
        n_days = max(int(np.ceil(self.spec.duration_days)), 7)
        return np.tile(pattern, int(np.ceil(n_days / 7)) + 1)

    # -- rain --------------------------------------------------------------
    def _rain_shape(self, t: np.ndarray | float) -> np.ndarray:
        """Smooth asymmetric pulse in [0, 1], zero outside the event window."""
        rain = self.spec.rain
        t = np.asarray(t, dtype=float)
        if rain is None:
            return np.zeros_like(t)
        x = (t - rain.start_day) / rain.duration_days
        rise = rain.rise_fraction
        rising = 0.5 * (1.0 - np.cos(np.pi * np.clip(x, 0.0, rise) / rise))
        falling = 0.5 * (1.0 + np.cos(np.pi * (np.clip(x, rise, 1.0) - rise) / (1.0 - rise)))
        pulse = np.where(x < rise, rising, falling)
        return np.where((x >= 0.0) & (x <= 1.0), pulse, 0.0)

    def _solve_rain_gain(self) -> float:
        """Scale the pulse so the peak *total* flow equals the Figure 5 reading."""
        if self.spec.rain is None:
            return 0.0
        rain = self.spec.rain
        t = np.linspace(rain.start_day, rain.start_day + rain.duration_days, 4001)
        dry = self._dry_flow(t)
        pulse = self._rain_shape(t)
        # maximise over the event: dry + gain*pulse == peak_flow at the argmax
        gains = np.where(pulse > 1e-12, (rain.peak_flow - dry) / np.maximum(pulse, 1e-12), np.inf)
        return float(np.min(gains))

    # -- signals -----------------------------------------------------------
    def _dry_flow(self, t: np.ndarray | float) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        day = np.clip(np.floor(t).astype(int), 0, len(self._weekly) - 1)
        base = (
            self.spec.flow_mean
            * np.exp(self._flow_b * self._shape(t, 0.0, self._flow_p))
            / self._flow_norm
        )
        return base * self._weekly[day]

    def flow(self, t: np.ndarray | float) -> np.ndarray:
        """Total influent flow [m3/d], dry weather plus any rain event."""
        dry = self._dry_flow(t)
        if self.spec.rain is None:
            return dry
        return dry + self._rain_gain * self._rain_shape(t)

    def concentrations(self, t: np.ndarray | float) -> np.ndarray:
        """Influent composition [..., 14] in vault component order."""
        t = np.asarray(t, dtype=float)
        lead = self.spec.shape.load_lead_hours / 24.0
        out = np.empty(t.shape + (len(self.components),), dtype=float)

        for i, name in enumerate(self.components):
            mean = BSM1_TABLE5_MEAN[name]
            if name in BSM1_CONSTANT_COMPONENTS or mean == 0.0:
                out[..., i] = mean
                continue
            family = COMPONENT_FAMILY[name]
            b = self._conc_b[family]
            p = self._conc_p[family]
            norm = self._conc_norm[family]
            out[..., i] = mean * np.exp(b * self._shape(t, lead, p)) / norm

        if self.spec.rain is not None:
            # Rain water carries no pollutant load: the catchment signal is
            # diluted so that the pollutant mass flow is preserved.
            dry = self._dry_flow(t)
            total = self.flow(t)
            dilution = (dry / np.maximum(total, 1e-12))[..., None]
            constant = np.array(
                [name in BSM1_CONSTANT_COMPONENTS for name in self.components]
            )
            out = np.where(constant, out, out * dilution)
        return out

    def __call__(self, t: float) -> tuple[float, np.ndarray]:
        return float(self.flow(t)), np.asarray(self.concentrations(np.asarray(t)))

    def series(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised evaluation over a time grid: ``(flow [n], Z [n, 14])``."""
        t = np.asarray(t, dtype=float)
        return self.flow(t), self.concentrations(t)

    # -- reporting ---------------------------------------------------------
    def summary(self, n: int = 20001) -> dict[str, object]:
        """Achieved statistics, for comparison against the BSM1 anchors."""
        t = np.linspace(0.0, self.spec.duration_days, n)
        q, z = self.series(t)
        stats = {
            "flow_mean_achieved": float(np.mean(q)),
            "flow_mean_target": self.spec.flow_mean,
            "flow_min_achieved": float(np.min(q)),
            "flow_max_achieved": float(np.max(q)),
            "flow_range_target": self.spec.flow_range,
        }
        for name, (low, high) in BSM1_FIG3_CONC_RANGE.items():
            i = self.vault.index(name)
            stats[name] = {
                "mean_achieved": float(np.mean(z[:, i])),
                "mean_target": BSM1_TABLE5_MEAN[name],
                "min_achieved": float(np.min(z[:, i])),
                "max_achieved": float(np.max(z[:, i])),
                "range_target": (low, high),
            }
        if self.spec.rain is not None:
            stats["rain_peak_achieved"] = float(np.max(q))
            stats["rain_peak_target"] = self.spec.rain.peak_flow
            # The dry-weather mean is the anchor for the *dry* component; the
            # rain event adds water on top, so the series mean is legitimately
            # higher and must not be read as a miss against Table 5.
            stats["flow_mean_note"] = (
                "series mean includes the rain event; the Table 5 anchor applies "
                "to the dry-weather component only"
            )
            dry_only = self._dry_flow(t)
            stats["dry_component_mean_achieved"] = float(np.mean(dry_only))
        return stats


def dry_weather(duration_days: float = 14.0) -> InfluentGenerator:
    """Dry-weather diurnal influent - the training and holdout scenario."""
    return InfluentGenerator(InfluentSpec(duration_days=duration_days))


def rain_weather(duration_days: float = 14.0) -> InfluentGenerator:
    """Dry weather with a week-2 rain event - the distribution-shift test set."""
    return InfluentGenerator(InfluentSpec(duration_days=duration_days, rain=RainEvent()))


def stabilisation_influent() -> tuple[float, np.ndarray]:
    """Constant BSM1 Table 5 load used for the ``WARMUP_DAYS`` warm-up."""
    v = vault()
    z = np.array([BSM1_TABLE5_MEAN[name] for name in v.components], dtype=float)
    return BSM1_TABLE5_FLOW, z


class ConstantInfluent:
    """Time-invariant BSM1 Table 5 load, exposing the generator interface.

    This is the curriculum's stage-1 scenario: the plant sitting on the same
    constant load the warm-up settled on, so the trajectory is flat and the
    network can learn the operating point and the physics operator before any
    dynamics are introduced.
    """

    def __init__(self, duration_days: float = 1.0) -> None:
        self.spec = InfluentSpec(duration_days=duration_days, rain=None)
        self.vault = vault()
        self._q, self._z = stabilisation_influent()

    def flow(self, t: np.ndarray | float) -> np.ndarray:
        return np.full(np.shape(t), self._q, dtype=float)

    def concentrations(self, t: np.ndarray | float) -> np.ndarray:
        return np.broadcast_to(self._z, np.shape(t) + self._z.shape).copy()

    def __call__(self, t: float) -> tuple[float, np.ndarray]:
        return self._q, self._z

    def series(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t = np.asarray(t, dtype=float)
        return self.flow(t), self.concentrations(t)

    def summary(self, n: int = 0) -> dict[str, object]:
        return {
            "scenario": "constant",
            "flow_mean_achieved": self._q,
            "flow_mean_target": BSM1_TABLE5_FLOW,
            "source": "BSM1 Table 5 stabilisation load, held constant",
        }


def constant_scenario(duration_days: float = 1.0) -> ConstantInfluent:
    """Curriculum stage-1 scenario: constant BSM1 Table 5 load."""
    return ConstantInfluent(duration_days)

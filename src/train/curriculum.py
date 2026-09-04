"""Three-tier hierarchical curriculum: horizon, scenario difficulty, loss weights.

Tiers
-----
1. **Scenario** - start on the constant-load trajectory the warm-up settles on,
   then move to the diurnal dry-weather series.
2. **Horizon** - 1 day, then 3, then 7, then the full 12-day training window.
3. **Weights** - ``lambda_physics`` ramps up and ``lambda_data`` ramps down
   across the schedule, each with a cosine ramp inside its stage.
   ``lambda_physics`` is never zero: this is a full PINN at every step.

Noise is deliberately NOT a curriculum axis
-------------------------------------------
The approved plan sketched stage 2 as "diurnal, noise-free". Implementing it
that way would hand every CL run a look at clean observations that the no-CL
runs never get, so the CL-vs-no-CL comparison would measure data quality rather
than curriculum. Instead the noise level is fixed per run, and the early stages
see a *smoothed* version of the same noisy signal - an easier view of identical
information, with nothing added. Smoothing is disabled entirely for the
noise-free runs so those remain untouched.

Budget parity
-------------
``no_curriculum`` returns a single-stage schedule with the final settings and
the same total step budget, so any difference between the two is attributable to
ordering rather than to compute.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np

from ..models.losses import LossWeights


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    dataset: str            # "constant" or "dry"
    horizon_days: float
    steps: int
    weights_start: LossWeights
    weights_end: LossWeights
    smoothing_window: int = 1   # samples in the moving average; 1 disables it

    def weights_at(self, progress: float) -> LossWeights:
        """Cosine ramp from ``weights_start`` to ``weights_end`` across the stage."""
        p = float(np.clip(progress, 0.0, 1.0))
        blend = 0.5 * (1.0 - np.cos(np.pi * p))
        mix = {}
        for field_name in ("data", "physics", "ic", "positivity", "balance"):
            a = getattr(self.weights_start, field_name)
            b = getattr(self.weights_end, field_name)
            mix[field_name] = float(a + blend * (b - a))
        return LossWeights(**mix)


@dataclass(frozen=True)
class CurriculumSchedule:
    stages: tuple[CurriculumStage, ...]
    plateau_patience: int = 0       # 0 disables early stage advance
    plateau_rel_tol: float = 1e-3

    @property
    def total_steps(self) -> int:
        return sum(stage.steps for stage in self.stages)

    def iterate(self) -> Iterator[tuple[int, CurriculumStage, LossWeights]]:
        """Yield ``(global_step, stage, weights)`` for the whole schedule."""
        step = 0
        for stage in self.stages:
            for local in range(stage.steps):
                progress = local / max(stage.steps - 1, 1)
                yield step, stage, stage.weights_at(progress)
                step += 1

    def describe(self) -> list[dict[str, object]]:
        return [
            {
                "name": s.name,
                "dataset": s.dataset,
                "horizon_days": s.horizon_days,
                "steps": s.steps,
                "smoothing_window": s.smoothing_window,
                "weights_start": s.weights_start.__dict__,
                "weights_end": s.weights_end.__dict__,
            }
            for s in self.stages
        ]


#: Final loss weights, reached at the end of the schedule and used throughout by
#: the no-curriculum runs.
FINAL_WEIGHTS = LossWeights(data=1.0, physics=1.0, ic=10.0, positivity=1.0, balance=0.1)
#: Opening weights: physics present but light, data dominant.
INITIAL_WEIGHTS = LossWeights(data=10.0, physics=0.05, ic=10.0, positivity=1.0, balance=0.0)


def hierarchical(
    total_steps: int = 20000,
    train_horizon_days: float = 12.0,
    noisy: bool = True,
    fractions: Sequence[float] = (0.15, 0.20, 0.25, 0.40),
    horizons: Sequence[float] = (1.0, 3.0, 7.0, None),
    smoothing: Sequence[int] = (9, 5, 3, 1),
    plateau_patience: int = 0,
) -> CurriculumSchedule:
    """The default four-stage hierarchical schedule (three tiers, split horizons)."""
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("Stage fractions must sum to 1, got %s" % (sum(fractions),))
    if not (len(fractions) == len(horizons) == len(smoothing)):
        raise ValueError("fractions, horizons and smoothing must have equal length")

    n = len(fractions)
    steps = [int(round(total_steps * f)) for f in fractions]
    steps[-1] += total_steps - sum(steps)  # absorb rounding

    # Weight ramp waypoints: interpolate each stage boundary between the
    # initial and final weights, so lambda_physics grows monotonically.
    def waypoint(k: int) -> LossWeights:
        blend = k / n
        mix = {}
        for field_name in ("data", "physics", "ic", "positivity", "balance"):
            a = getattr(INITIAL_WEIGHTS, field_name)
            b = getattr(FINAL_WEIGHTS, field_name)
            mix[field_name] = float(a + blend * (b - a))
        return LossWeights(**mix)

    stages = []
    for k in range(n):
        horizon = train_horizon_days if horizons[k] is None else float(horizons[k])
        stages.append(
            CurriculumStage(
                name="stage%d" % (k + 1,),
                dataset="constant" if k == 0 else "dry",
                horizon_days=min(horizon, train_horizon_days),
                steps=steps[k],
                weights_start=waypoint(k),
                weights_end=waypoint(k + 1),
                smoothing_window=int(smoothing[k]) if noisy else 1,
            )
        )
    return CurriculumSchedule(tuple(stages), plateau_patience=plateau_patience)


def no_curriculum(total_steps: int = 20000, train_horizon_days: float = 12.0) -> CurriculumSchedule:
    """Single stage at the final settings, same total step budget."""
    stage = CurriculumStage(
        name="single",
        dataset="dry",
        horizon_days=train_horizon_days,
        steps=int(total_steps),
        weights_start=FINAL_WEIGHTS,
        weights_end=FINAL_WEIGHTS,
        smoothing_window=1,
    )
    return CurriculumSchedule((stage,))


# --- single-axis ablations --------------------------------------------------
# Each variant keeps exactly one of the hierarchical schedule's axes and holds
# the other three at the single-stage settings, under the same total budget.
# Together with `hierarchical` and `none` they let the benchmark attribute the
# curriculum effect to a specific axis instead of the bundle.

def weights_only(total_steps: int, train_horizon_days: float) -> CurriculumSchedule:
    """Only the loss-weight ramp: one stage, full horizon, dry data, no smoothing."""
    stage = CurriculumStage(
        name="weights_only",
        dataset="dry",
        horizon_days=train_horizon_days,
        steps=int(total_steps),
        weights_start=INITIAL_WEIGHTS,
        weights_end=FINAL_WEIGHTS,
        smoothing_window=1,
    )
    return CurriculumSchedule((stage,))


def _staged(
    name: str,
    total_steps: int,
    datasets: Sequence[str],
    horizons: Sequence[float],
    smoothing: Sequence[int],
    fractions: Sequence[float],
) -> CurriculumSchedule:
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("Stage fractions must sum to 1, got %s" % (sum(fractions),))
    steps = [int(round(total_steps * f)) for f in fractions]
    steps[-1] += total_steps - sum(steps)
    stages = tuple(
        CurriculumStage(
            name="%s%d" % (name, k + 1),
            dataset=datasets[k],
            horizon_days=float(horizons[k]),
            steps=steps[k],
            weights_start=FINAL_WEIGHTS,
            weights_end=FINAL_WEIGHTS,
            smoothing_window=int(smoothing[k]),
        )
        for k in range(len(fractions))
    )
    return CurriculumSchedule(stages)


def horizon_only(total_steps: int, train_horizon_days: float) -> CurriculumSchedule:
    """Only the growing horizon: four dry stages at final weights, no smoothing."""
    return _staged(
        "horizon",
        total_steps,
        datasets=("dry", "dry", "dry", "dry"),
        horizons=(1.0, 3.0, 7.0, train_horizon_days),
        smoothing=(1, 1, 1, 1),
        fractions=(0.15, 0.20, 0.25, 0.40),
    )


def scenario_only(total_steps: int, train_horizon_days: float) -> CurriculumSchedule:
    """Only the constant-to-dry scenario switch, at final weights, no smoothing.

    The constant dataset is one day long, so its stage horizon is capped by the
    data; that residual coupling to the horizon axis is documented in the paper.
    """
    return _staged(
        "scenario",
        total_steps,
        datasets=("constant", "dry"),
        horizons=(train_horizon_days, train_horizon_days),
        smoothing=(1, 1),
        fractions=(0.15, 0.85),
    )


def smoothing_only(total_steps: int, train_horizon_days: float, noisy: bool) -> CurriculumSchedule:
    """Only the observation-smoothing schedule, at final weights, full horizon."""
    windows = (9, 5, 3, 1) if noisy else (1, 1, 1, 1)
    return _staged(
        "smoothing",
        total_steps,
        datasets=("dry", "dry", "dry", "dry"),
        horizons=(train_horizon_days,) * 4,
        smoothing=windows,
        fractions=(0.15, 0.20, 0.25, 0.40),
    )


def build(kind: str, total_steps: int, train_horizon_days: float, noisy: bool) -> CurriculumSchedule:
    if kind == "hierarchical":
        return hierarchical(total_steps, train_horizon_days, noisy=noisy)
    if kind == "none":
        return no_curriculum(total_steps, train_horizon_days)
    if kind == "weights_only":
        return weights_only(total_steps, train_horizon_days)
    if kind == "horizon_only":
        return horizon_only(total_steps, train_horizon_days)
    if kind == "scenario_only":
        return scenario_only(total_steps, train_horizon_days)
    if kind == "smoothing_only":
        return smoothing_only(total_steps, train_horizon_days, noisy)
    raise ValueError(
        "Unknown curriculum %r; expected 'hierarchical', 'none', or one of the "
        "single-axis ablations 'weights_only', 'horizon_only', 'scenario_only', "
        "'smoothing_only'" % (kind,)
    )


def smooth_observations(obs: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average over time, per channel. ``window <= 1`` is a no-op."""
    if window <= 1:
        return obs
    kernel = np.ones(int(window)) / float(window)
    padded = np.pad(obs, ((window // 2, window // 2), (0, 0)), mode="edge")
    return np.stack(
        [np.convolve(padded[:, j], kernel, mode="valid")[: obs.shape[0]] for j in range(obs.shape[1])],
        axis=-1,
    )

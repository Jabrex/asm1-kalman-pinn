"""Ground-truth ODE simulation: warm-up, dynamic runs, dataset persistence.

Pipeline
--------
1. ``warm_up`` integrates the plant on the constant BSM1 Table 5 load until the
   starting state no longer remembers its seed - see ``WARMUP_DAYS``. The
   initial condition is a numerical starting point only, not a model parameter;
   see :meth:`Bsm1Plant.seed_state`.
2. ``simulate`` integrates the dynamic scenario from that steady state and
   samples on a 15-minute grid (the BSM1 evaluation interval, report p.14).
3. ``SimulationResult`` stores states plus every derived stream needed later by
   the sensor model, the metrics and the effluent quality index.

The solver is stiff (oxygen relaxes in minutes, sludge age is days), so BDF is
the default. RUNBOOK step 3 cross-checks BDF against Radau and LSODA.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from ..asm1.plant import Bsm1Config, Bsm1Plant, InfluentFn, constant_influent
from ..asm1.vault_loader import vault
from .influent import (
    BSM1_TABLE5_MEAN,
    InfluentGenerator,
    dry_weather,
    rain_weather,
    stabilisation_influent,
)

#: BSM1 evaluation sampling interval (report p.14).
SAMPLE_INTERVAL_DAYS = 15.0 / (24.0 * 60.0)
#: BSM1 names 100 days for stabilisation (report section 3), but that is not
#: enough here: at 20 degrees C two different initial conditions still differ by
#: 1.6e-05 after 100 days, and only converge to 1.5e-10 by day 200. The warm-up
#: is a one-off cost, so it is taken long enough that the starting state is
#: genuinely independent of the seed. RUNBOOK step 3, gate 3f verifies this.
WARMUP_DAYS = 200.0

DEFAULT_METHOD = "BDF"
#: Tighter than a smooth problem would need. The Takacs settler RHS is only
#: piecewise continuous - the flux limiter's ``min()`` branches switch and the
#: X_t threshold is crossed between layers constantly - which degrades BDF's
#: error estimator on the settler states. rtol 1e-10 puts settler solids near
#: 1e-6 relative, while the smooth reactor states land near 1e-8.
DEFAULT_RTOL = 1e-10
DEFAULT_ATOL = 1e-12


@dataclass
class SolverSettings:
    method: str = DEFAULT_METHOD
    rtol: float = DEFAULT_RTOL
    atol: float = DEFAULT_ATOL
    max_step: float = np.inf


@dataclass
class SimulationResult:
    """Sampled trajectory plus every derived stream downstream code needs."""

    t: np.ndarray                 # (n,) days
    y: np.ndarray                 # (n, state_size) raw ODE state
    reactor: np.ndarray           # (n, 5, 14)
    settler_solids: np.ndarray    # (n, 10)
    effluent: np.ndarray          # (n, 14)
    underflow: np.ndarray         # (n, 14)
    influent: np.ndarray          # (n, 14)
    q_in: np.ndarray              # (n,)
    tss_reactor: np.ndarray       # (n, 5)
    tss_underflow: np.ndarray     # (n,)
    tss_effluent: np.ndarray      # (n,)
    meta: dict[str, Any]

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            name: getattr(self, name)
            for name in (
                "t", "y", "reactor", "settler_solids", "effluent", "underflow",
                "influent", "q_in", "tss_reactor", "tss_underflow", "tss_effluent",
            )
        }
        np.savez_compressed(path, meta=json.dumps(self.meta), **arrays)
        return path

    @classmethod
    def load(cls, path: Path | str) -> "SimulationResult":
        with np.load(Path(path), allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            return cls(
                meta=meta,
                **{k: data[k] for k in data.files if k != "meta"},
            )


def _collect(plant: Bsm1Plant, t: np.ndarray, y: np.ndarray, influent: InfluentFn) -> dict[str, np.ndarray]:
    n = len(t)
    out: dict[str, list] = {
        "reactor": [], "settler_solids": [], "effluent": [], "underflow": [],
        "influent": [], "q_in": [], "tss_reactor": [], "tss_underflow": [], "tss_effluent": [],
    }
    for k in range(n):
        o = plant.outputs(float(t[k]), y[k], influent)
        out["reactor"].append(o["reactor"])
        out["settler_solids"].append(o["settler_solids"])
        out["effluent"].append(o["effluent"])
        out["underflow"].append(o["underflow"])
        out["influent"].append(o["influent"])
        out["q_in"].append(o["q_in"])
        out["tss_reactor"].append(o["tss_reactor"])
        out["tss_underflow"].append(o["tss_underflow"])
        out["tss_effluent"].append(o["tss_effluent"])
    return {k: np.asarray(v) for k, v in out.items()}


def integrate(
    plant: Bsm1Plant,
    influent: InfluentFn,
    t_span: tuple[float, float],
    y0: np.ndarray,
    t_eval: np.ndarray | None = None,
    solver: SolverSettings | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Raw integration; returns ``(t, y)`` with ``y`` shaped ``(n, state_size)``."""
    s = solver if solver is not None else SolverSettings()
    sol = solve_ivp(
        fun=lambda t, y: plant.rhs(t, y, influent),
        t_span=t_span,
        y0=np.asarray(y0, dtype=float),
        t_eval=t_eval,
        method=s.method,
        rtol=s.rtol,
        atol=s.atol,
        max_step=s.max_step,
    )
    if not sol.success:
        raise RuntimeError("Integration failed (%s): %s" % (s.method, sol.message))
    return sol.t, sol.y.T


def default_seed(plant: Bsm1Plant) -> np.ndarray:
    """Starting point for the warm-up.

    Reactor content is set to the BSM1 Table 5 influent composition, with the
    autotrophs seeded at the same concentration as the heterotrophs. BSM1 lists
    ``X_B,A = 0`` in the influent, and a zero autotroph seed can never grow
    because rho_3 is proportional to ``X_B_A``; an inoculum is therefore
    required. This value is a starting point, not a model parameter - the
    warm-up washes it out, and RUNBOOK step 3 gate 3f verifies that two
    different seeds reach the same steady state.
    """
    v = vault()
    z = np.array([BSM1_TABLE5_MEAN[name] for name in v.components], dtype=float)
    z[v.index("X_B_A")] = BSM1_TABLE5_MEAN["X_B_H"]
    return plant.seed_state(z, solids_seed=float(plant.tss(z)))


def warm_up(
    plant: Bsm1Plant | None = None,
    days: float = WARMUP_DAYS,
    y0: np.ndarray | None = None,
    solver: SolverSettings | None = None,
) -> tuple[Bsm1Plant, np.ndarray]:
    """Integrate to steady state on the constant BSM1 Table 5 load."""
    plant = plant if plant is not None else Bsm1Plant()
    q, z = stabilisation_influent()
    influent = constant_influent(q, z)
    start = y0 if y0 is not None else default_seed(plant)
    _, y = integrate(plant, influent, (0.0, days), start, t_eval=np.array([days]), solver=solver)
    return plant, y[-1]


def steady_state_residual(plant: Bsm1Plant, y: np.ndarray) -> float:
    """Relative ``||dy/dt|| / ||y||`` on the constant load - RUNBOOK step 3e."""
    q, z = stabilisation_influent()
    dy = plant.rhs(0.0, y, constant_influent(q, z))
    return float(np.linalg.norm(dy) / max(np.linalg.norm(y), 1e-30))


def simulate(
    influent: InfluentGenerator,
    plant: Bsm1Plant | None = None,
    y0: np.ndarray | None = None,
    sample_interval: float = SAMPLE_INTERVAL_DAYS,
    solver: SolverSettings | None = None,
    scenario: str = "dry",
) -> SimulationResult:
    """Run one dynamic scenario from a warmed-up steady state."""
    if plant is None or y0 is None:
        plant, y0 = warm_up(plant, solver=solver)

    duration = influent.spec.duration_days
    n_steps = int(round(duration / sample_interval))
    t_eval = np.linspace(0.0, duration, n_steps + 1)
    t, y = integrate(plant, influent, (0.0, duration), y0, t_eval=t_eval, solver=solver)
    derived = _collect(plant, t, y, influent)

    s = solver if solver is not None else SolverSettings()
    v = plant.vault
    meta = {
        "scenario": scenario,
        "components": list(v.components),
        "processes": list(v.processes),
        "vault_json_sha256": v.json_sha256,
        "source_xlsx_sha256": v.source_xlsx_sha256,
        "parameters": dict(v.parameters),
        "plant_config": asdict(plant.cfg),
        "solver": asdict(s) if np.isfinite(s.max_step) else {**asdict(s), "max_step": None},
        "sample_interval_days": sample_interval,
        "duration_days": duration,
        "warmup_days": WARMUP_DAYS,
        "influent_summary": influent.summary(),
        "temperature_note": (
            "Vault 20 C parameter set only. BSM1 supplies geometry, flows and "
            "influent composition; its 15 C kinetic parameters are NOT used, so "
            "this steady state does not and should not match BSM1 Table 6."
        ),
    }
    return SimulationResult(t=t, y=y, meta=meta, **derived)


def generate(
    scenario: str = "dry",
    duration_days: float = 14.0,
    config: Bsm1Config | None = None,
    solver: SolverSettings | None = None,
) -> SimulationResult:
    """Build one labelled dataset: ``scenario`` is ``"dry"`` or ``"rain"``."""
    builders = {"dry": dry_weather, "rain": rain_weather}
    if scenario not in builders:
        raise ValueError("Unknown scenario %r; expected one of %s" % (scenario, sorted(builders)))
    plant = Bsm1Plant(config)
    plant, y0 = warm_up(plant, solver=solver)
    return simulate(
        builders[scenario](duration_days),
        plant=plant,
        y0=y0,
        solver=solver,
        scenario=scenario,
    )

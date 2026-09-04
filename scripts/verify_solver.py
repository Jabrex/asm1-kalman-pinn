"""RUNBOOK step 3 - solver verification gate, entirely at 20 degrees C.

No published steady-state table is used as a reference, because this project
runs the vault's 20 degrees C parameters while BSM1 Table 6 was produced with a
15 degrees C set. The solver is therefore verified against itself and against
independent numerics:

3a  cross-solver     BDF vs Radau vs LSODA on the same problem
3b  tolerance        rtol 1e-8 / 1e-10 / 1e-12, successive differences shrink
3c  zero reaction    reactions off and recycles closed reduces the reactor train
                     to an affine linear system; compared against a matrix
                     exponential, which shares no code with solve_ivp
3d  tracer closure   S_I and X_I have identically zero stoichiometry columns, so
                     in - out must equal accumulation with no reaction term
3e  steady state     ||dy/dt|| / ||y|| after the warm-up
3f  IC independence  two different seeds reach the same steady state

Gates 3a, 3b and 3d report per subsystem rather than as one number. The reactor
RHS is smooth; the Takacs settler RHS is only piecewise continuous, and its
eq. 46 particulate bookkeeping is an explicit approximation. Collapsing the two
into a single tolerance would either hide a real reactor error or fail on a
documented property of the published settler model.

Run::

    python -m scripts.verify_solver

Exit code 0 means every gate passes and data generation may proceed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.asm1.continuity import system_inventory, tracer_components  # noqa: E402
from src.asm1.plant import Bsm1Config, Bsm1Plant, constant_influent  # noqa: E402
from src.data.influent import BSM1_TABLE5_MEAN, dry_weather, stabilisation_influent  # noqa: E402
from src.data.simulate import (  # noqa: E402
    WARMUP_DAYS,
    SolverSettings,
    integrate,
    steady_state_residual,
    warm_up,
)

# The plant is two subsystems with very different numerical character, and they
# need separate tolerances or the gates measure the wrong thing.
#
#   Reactor (70 states)   smooth RHS; BDF converges cleanly.
#   Settler (90 states)   the Takacs flux limiter is only piecewise continuous:
#                         its ``min()`` branches switch dozens of times a day and
#                         the X_t = 3000 threshold is crossed between layers
#                         essentially always. A non-smooth RHS defeats BDF's
#                         error estimator, so the settler solids converge, but an
#                         order of magnitude or two behind the reactor states.
#
# This is a property of the published settler model, not of this implementation.
# The gates therefore hold the reactor to a strict bound and the settler to a
# documented, still-meaningful one, and print both so the split stays visible.
TOL_REACTOR = 1e-6
TOL_SETTLER = 1e-4
TOL_LINEAR = 1e-6
TOL_SOLUBLE_TRACER = 1e-8
TOL_PARTICULATE_TRACER_INERT = 1e-8
#: With reactions on, BSM1 eq. 46 relabels clarifier sludge with the *current*
#: feed composition, so an individual particulate species is not conserved when
#: the reactor composition moves. X_S alone swings by a factor of five over a
#: day. This bound records the size of that documented approximation; it is not
#: a solver accuracy target. Gate 3d also runs the same check with reactions off,
#: where the assumption holds exactly and the tight bound applies.
TOL_PARTICULATE_TRACER_REACTIVE = 5e-2
TOL_STEADY = 1e-6
TOL_IC = 1e-6

PROBE_DAYS = 1.0
PROBE_POINTS = 25
#: Closure integrals use the trapezoidal rule, whose O(h^2) error dominates the
#: check unless the grid is dense. 401 points over two days leaves 2.4e-07,
#: which says nothing about the model; 3201 points puts it below 1e-8.
CLOSURE_POINTS = 3201


def _relative_difference(a: np.ndarray, b: np.ndarray) -> float:
    scale = np.maximum(np.abs(a), np.abs(b))
    mask = scale > 1e-8
    if not np.any(mask):
        return 0.0
    return float(np.max(np.abs(a[mask] - b[mask]) / scale[mask]))


def _by_subsystem(plant: Bsm1Plant, a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Split a comparison into the smooth reactor and the non-smooth settler."""
    return {
        "reactor": _relative_difference(a[:, plant.idx.reactor], b[:, plant.idx.reactor]),
        "settler_solids": _relative_difference(
            a[:, plant.idx.settler_solids], b[:, plant.idx.settler_solids]
        ),
        "settler_solubles": _relative_difference(
            a[:, plant.idx.settler_solubles], b[:, plant.idx.settler_solubles]
        ),
    }


def _within(diffs: dict[str, float]) -> bool:
    return (
        diffs["reactor"] < TOL_REACTOR
        and diffs["settler_solubles"] < TOL_REACTOR
        and diffs["settler_solids"] < TOL_SETTLER
    )


def gate_3a(plant: Bsm1Plant, y0: np.ndarray) -> tuple[bool, dict]:
    influent = dry_weather(PROBE_DAYS)
    t_eval = np.linspace(0.0, PROBE_DAYS, PROBE_POINTS)
    runs = {}
    for method in ("BDF", "Radau", "LSODA"):
        _, y = integrate(
            plant, influent, (0.0, PROBE_DAYS), y0, t_eval=t_eval,
            solver=SolverSettings(method=method, rtol=1e-10, atol=1e-12),
        )
        runs[method] = y
    detail: dict[str, float] = {}
    ok = True
    for other in ("Radau", "LSODA"):
        diffs = _by_subsystem(plant, runs["BDF"], runs[other])
        for part, value in diffs.items():
            detail["BDF_vs_%s_%s" % (other, part)] = value
        ok = ok and _within(diffs)
    return ok, detail


def gate_3b(plant: Bsm1Plant, y0: np.ndarray) -> tuple[bool, dict]:
    """Refining the tolerance must reduce the difference, per subsystem."""
    influent = dry_weather(PROBE_DAYS)
    t_eval = np.linspace(0.0, PROBE_DAYS, PROBE_POINTS)
    solutions = {}
    for rtol, atol in ((1e-8, 1e-10), (1e-10, 1e-12), (1e-12, 1e-14)):
        _, y = integrate(
            plant, influent, (0.0, PROBE_DAYS), y0, t_eval=t_eval,
            solver=SolverSettings(rtol=rtol, atol=atol),
        )
        solutions[rtol] = y

    coarse = _by_subsystem(plant, solutions[1e-8], solutions[1e-12])
    fine = _by_subsystem(plant, solutions[1e-10], solutions[1e-12])
    detail: dict[str, float] = {}
    for part in coarse:
        detail["%s_rtol1e-8" % part] = coarse[part]
        detail["%s_rtol1e-10" % part] = fine[part]
    converging = all(fine[part] < coarse[part] for part in coarse)
    detail["monotone_convergence"] = converging
    return converging and _within(fine), detail


def _affine_reactor_system(plant: Bsm1Plant, q_in: float, z_in: np.ndarray):
    """Build ``dZ/dt = A Z + b`` for the hydraulic-only reactor cascade."""
    n_tanks, n_comp = plant.cfg.n_tanks, plant.n_components
    size = n_tanks * n_comp
    A = np.zeros((size, size))
    b = np.zeros(size)
    volumes = np.asarray(plant.cfg.volumes, dtype=float)
    kla = np.asarray(plant.cfg.kla, dtype=float)

    def idx(tank: int, comp: int) -> int:
        return tank * n_comp + comp

    for c in range(n_comp):
        A[idx(0, c), idx(0, c)] = -q_in / volumes[0]
        b[idx(0, c)] = q_in * z_in[c] / volumes[0]
        for k in range(1, n_tanks):
            A[idx(k, c), idx(k, c)] = -q_in / volumes[k]
            A[idx(k, c), idx(k - 1, c)] = q_in / volumes[k]
    for k in range(n_tanks):
        A[idx(k, plant.i_so), idx(k, plant.i_so)] -= kla[k]
        b[idx(k, plant.i_so)] += kla[k] * plant.cfg.so_sat
    return A, b


def gate_3c() -> tuple[bool, dict]:
    """Reactions off, recycles closed: compare solve_ivp against a matrix exponential."""
    cfg = Bsm1Config(reaction_scale=0.0, q_int=0.0, q_r=0.0)
    plant = Bsm1Plant(cfg)
    q_in, z_in = stabilisation_influent()

    z0 = np.zeros(plant.n_components)
    y0 = plant.seed_state(z0, solids_seed=0.0)
    horizon = 0.5
    t_eval = np.linspace(0.0, horizon, 21)
    _, y = integrate(
        plant, constant_influent(q_in, z_in), (0.0, horizon), y0, t_eval=t_eval,
        solver=SolverSettings(rtol=1e-11, atol=1e-13),
    )
    numeric = y[:, plant.idx.reactor]

    A, b = _affine_reactor_system(plant, q_in, z_in)
    x0 = np.zeros(A.shape[0])
    steady = np.linalg.solve(A, -b)
    analytic = np.array([steady + expm(A * t) @ (x0 - steady) for t in t_eval])

    diff = _relative_difference(numeric, analytic)
    return diff < TOL_LINEAR, {"max_relative_difference": diff, "reference": "scipy.linalg.expm"}


def _tracer_closure(plant: Bsm1Plant, y0: np.ndarray, influent, name: str) -> float:
    """Relative |in - out - accumulation| for one zero-stoichiometry component."""
    t_eval = np.linspace(0.0, 2.0, CLOSURE_POINTS)
    _, y = integrate(
        plant, influent, (0.0, 2.0), y0, t_eval=t_eval,
        solver=SolverSettings(rtol=1e-11, atol=1e-13),
    )
    i = plant.vault.index(name)
    q_in, z_in = influent.series(t_eval)
    effluent = np.empty_like(t_eval)
    underflow = np.empty_like(t_eval)
    for k, t in enumerate(t_eval):
        out = plant.outputs(float(t), y[k], influent)
        effluent[k] = out["effluent"][i]
        underflow[k] = out["underflow"][i]
    inflow = q_in * z_in[:, i]
    outflow = (q_in - plant.cfg.q_w) * effluent + plant.cfg.q_w * underflow
    accumulated = system_inventory(plant, y[-1], name) - system_inventory(plant, y[0], name)
    net = float(np.trapezoid(inflow - outflow, t_eval))
    scale = max(abs(net), abs(accumulated), float(np.trapezoid(inflow, t_eval)), 1e-12)
    return abs(net - accumulated) / scale


def gate_3d(plant: Bsm1Plant, y0: np.ndarray) -> tuple[bool, dict]:
    """Passive-tracer closure. ``S_I`` and ``X_I`` have zero stoichiometry.

    Three checks, because the soluble and the particulate tracer are guaranteed
    by different parts of the model:

    ``S_I`` with reactions on
        Each clarifier layer integrates its solubles directly, so this must close
        to the accuracy of the ODE and the quadrature.
    ``X_I`` with reactions OFF
        Validates the settler solids equations and the eq. 46 particulate
        bookkeeping in isolation, including under a time-varying feed.
    ``X_I`` with reactions ON
        Does NOT close, and cannot: eq. 46 relabels the sludge already sitting in
        the clarifier with the reactor's *current* composition, and reactions move
        that composition constantly. The value is measured and bounded rather
        than demanded to vanish, and the report records it as a property of the
        published settler model.
    """
    tracers = tracer_components()
    if set(tracers) != {"S_I", "X_I"}:
        return False, {"error": "expected S_I and X_I as tracers, got %s" % (tracers,)}

    # The reactions-off plant needs its OWN steady state. Handing it the reactive
    # plant's state starts a violent transient - biomass that was being sustained
    # by growth simply washes out - and the closure would then be measuring that
    # transient rather than the settler.
    inert_plant = Bsm1Plant(Bsm1Config(reaction_scale=0.0))
    q, z = stabilisation_influent()
    _, inert_y0 = integrate(
        inert_plant, constant_influent(q, z), (0.0, WARMUP_DAYS), y0,
        t_eval=np.array([WARMUP_DAYS]), solver=SolverSettings(rtol=1e-10, atol=1e-12),
    )

    soluble_reactive = _tracer_closure(plant, y0, dry_weather(2.0), "S_I")
    particulate_inert = _tracer_closure(inert_plant, inert_y0[-1], dry_weather(2.0), "X_I")
    particulate_reactive = _tracer_closure(plant, y0, dry_weather(2.0), "X_I")

    ok = (
        soluble_reactive < TOL_SOLUBLE_TRACER
        and particulate_inert < TOL_PARTICULATE_TRACER_INERT
        and particulate_reactive < TOL_PARTICULATE_TRACER_REACTIVE
    )
    return ok, {
        "S_I_reactions_on": soluble_reactive,
        "X_I_reactions_off": particulate_inert,
        "X_I_reactions_on__eq46_approximation": particulate_reactive,
        "quadrature_points": CLOSURE_POINTS,
    }


def gate_3e(plant: Bsm1Plant, y0: np.ndarray) -> tuple[bool, dict]:
    residual = steady_state_residual(plant, y0)
    return residual < TOL_STEADY, {"relative_dydt_norm": residual}


def gate_3f(plant: Bsm1Plant, y0: np.ndarray) -> tuple[bool, dict]:
    """A second, deliberately different seed must reach the same steady state."""
    v = plant.vault
    alt = np.array([BSM1_TABLE5_MEAN[name] for name in v.components], dtype=float)
    alt[v.index("X_B_H")] = 5.0 * BSM1_TABLE5_MEAN["X_B_H"]
    alt[v.index("X_B_A")] = 0.2 * BSM1_TABLE5_MEAN["X_B_H"]
    alt_start = plant.seed_state(alt, solids_seed=float(plant.tss(alt)))
    _, y_alt = warm_up(plant, y0=alt_start)
    diff = _relative_difference(y0, y_alt)
    return diff < TOL_IC, {"max_relative_difference": diff}


def main() -> int:
    print("Warming up (%.0f days, constant BSM1 Table 5 load, vault 20 C parameters)..."
          % WARMUP_DAYS)
    plant, y0 = warm_up()
    print("  done.\n")

    subsystem_tol = "reactor/solubles %.0e, settler solids %.0e" % (TOL_REACTOR, TOL_SETTLER)
    gates = [
        ("3a cross-solver", lambda: gate_3a(plant, y0), subsystem_tol),
        ("3b tolerance convergence", lambda: gate_3b(plant, y0), subsystem_tol),
        ("3c zero reaction vs expm", gate_3c, "%.0e" % TOL_LINEAR),
        ("3d tracer closure", lambda: gate_3d(plant, y0),
         "soluble %.0e, inert particulate %.0e, eq.46 approximation %.0e"
         % (TOL_SOLUBLE_TRACER, TOL_PARTICULATE_TRACER_INERT, TOL_PARTICULATE_TRACER_REACTIVE)),
        ("3e steady state", lambda: gate_3e(plant, y0), "%.0e" % TOL_STEADY),
        ("3f IC independence", lambda: gate_3f(plant, y0), "%.0e" % TOL_IC),
    ]

    failures = []
    for name, fn, tol in gates:
        ok, detail = fn()
        status = "PASS" if ok else "FAIL"
        print("%-26s %s   (tolerance: %s)" % (name, status, tol))
        for key, value in detail.items():
            print("    %-38s %s" % (key, value))
        if not ok:
            failures.append(name)
        print()

    if failures:
        print("FAIL - gates not cleared: %s" % ", ".join(failures))
        return 1
    print("PASS - step 3 clear, data generation may proceed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

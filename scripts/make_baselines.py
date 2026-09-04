"""Non-learned reference rows for the benchmark tables.

Two baselines, neither of which uses any sensor data:

``persistence``
    Holds the known t = 0 reactor state constant over every evaluation window.
    The floor any learned model must beat before its numbers mean anything.
``ode_openloop``
    Integrates the plant model forward from the saved full t = 0 state with the
    known influent. With exact equations, exact parameters, and the exact
    initial condition this is the perfect-model information bound for the
    in-model benchmark; learned models cannot be expected to beat it, and the
    distance to it measures what training actually recovers.

Each baseline is materialised as a normal run directory (summary.json +
predictions.npz) per noise level, so ``src.eval.report.collect_runs`` scores it
exactly like a trained model. Content is noise-independent; the four sigma
directories exist only to fill the report pivot.

Usage: python -m scripts.make_baselines
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.asm1.plant import Bsm1Plant  # noqa: E402

RAW = Path("results/raw")
OUT = Path("results/runs")
SIGMAS = (0.0, 0.05, 0.10, 0.15)
TRAIN_END = 12.0
HOLDOUT = (12.0, 14.0)
#: Looser than the generator's 1e-10 on purpose: the baseline must not simply
#: reproduce the data file bit-for-bit through an identical solve.
RTOL = 1e-6
ATOL = 1e-8


def sigma_tag(sigma: float) -> str:
    return ("%.2f" % sigma).replace(".", "p")


def scenario_arrays(scenario: str) -> dict[str, np.ndarray]:
    with np.load(RAW / ("sim_%s.npz" % scenario)) as sim:
        return {k: sim[k].copy() for k in ("t", "y", "q_in", "influent", "reactor")}


def integrate_open_loop(plant: Bsm1Plant, data: dict[str, np.ndarray]) -> np.ndarray:
    """Reactor trajectory (n, 5, 14) from y(0) under the known influent."""
    t, q_in, z_in = data["t"], data["q_in"], data["influent"]

    def influent(tt: float) -> tuple[float, np.ndarray]:
        q = float(np.interp(tt, t, q_in))
        z = np.array([np.interp(tt, t, z_in[:, j]) for j in range(z_in.shape[1])])
        return q, z

    sol = solve_ivp(
        plant.rhs,
        (float(t[0]), float(t[-1])),
        data["y"][0],
        method="BDF",
        t_eval=t,
        rtol=RTOL,
        atol=ATOL,
        args=(influent,),
    )
    if not sol.success:
        raise RuntimeError("open-loop integration failed: %s" % sol.message)
    reactor = np.stack([plant.unpack(sol.y[:, i])[0] for i in range(sol.y.shape[1])])
    return reactor


def split_windows(t: np.ndarray, reactor: np.ndarray) -> dict[str, np.ndarray]:
    train = reactor[t <= TRAIN_END + 1e-9]
    holdout = reactor[(t >= HOLDOUT[0] - 1e-9) & (t <= HOLDOUT[1] + 1e-9)]
    return {"train": train, "holdout": holdout}


def write_run(name: str, model: str, sigma: float, predictions: dict[str, np.ndarray], seconds: float) -> None:
    run_dir = OUT / ("%s_sigma%s" % (name, sigma_tag(sigma)))
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(run_dir / "predictions.npz", **predictions)
    summary = {
        "run_id": run_dir.name,
        "model": model,
        "arch": "analytic",
        "curriculum": "none",
        "noise": sigma,
        "seed": 0,
        "profile": "full",
        "train_end_day": TRAIN_END,
        "holdout_days": list(HOLDOUT),
        "steps": 0,
        "train_seconds": seconds,
        "n_parameters": 0,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    plant = Bsm1Plant()
    dry = scenario_arrays("dry")
    rain = scenario_arrays("rain")

    # --- persistence -------------------------------------------------------
    z0 = dry["reactor"][0]
    persist = {
        **{k: np.tile(z0, (len(v), 1, 1)) for k, v in split_windows(dry["t"], dry["reactor"]).items()},
        "rain": np.tile(z0, (len(rain["t"]), 1, 1)),
    }

    # --- open-loop ODE ------------------------------------------------------
    started = time.perf_counter()
    dry_traj = integrate_open_loop(plant, dry)
    rain_traj = integrate_open_loop(plant, rain)
    ode_seconds = time.perf_counter() - started
    ode = {**split_windows(dry["t"], dry_traj), "rain": rain_traj}

    for sigma in SIGMAS:
        write_run("persistence", "persistence", sigma, persist, 0.0)
        write_run("odesim", "ode_openloop", sigma, ode, ode_seconds)

    # Convenience: report the headline numbers immediately.
    from src.eval.metrics import state_metrics, track_summary  # noqa: E402

    truth_hold = dry["reactor"][(dry["t"] >= HOLDOUT[0] - 1e-9)]
    for label, pred in (("persistence", persist["holdout"]), ("ode_openloop", ode["holdout"])):
        tracks = track_summary(state_metrics(truth_hold, pred))
        print(
            "%s  holdout  track_a_nrmse=%.4g  track_b_nrmse=%.4g"
            % (label, tracks["track_a_measured"]["nrmse"], tracks["track_b_unmeasured"]["nrmse"])
        )
    print("wrote %d baseline run dirs under %s" % (2 * len(SIGMAS), OUT))


if __name__ == "__main__":
    main()

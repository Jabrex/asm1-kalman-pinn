"""Non-learned reference rows for the benchmark tables.

Two baselines, neither of which uses any sensor data:

``persistence``
    Holds the known t = 0 reactor state constant over every evaluation window.
    The floor any learned model must beat before its numbers mean anything.
``ode_openloop`` (run directory ``odesim_*``)
    Integrates the plant model forward from the saved full t = 0 state with the
    known influent. With exact equations, exact parameters, and the exact
    initial condition this is the perfect-model information bound for the
    in-model benchmark; learned models cannot be expected to beat it, and the
    distance to it measures what training actually recovers.

Each baseline is materialised as a normal run directory (summary.json +
predictions.npz) per noise level, so ``src.eval.report.collect_runs`` scores it
exactly like a trained model. Content is noise-independent; the four sigma
directories exist only to fill the report pivot.

Usage::

    python -m scripts.make_baselines                       # v1.0 rows, bit for bit
    python -m scripts.make_baselines --data-dir results/raw_k100 --out results/v11/baselines/k100 \\
        --anchor-file results/v11/anchors/k100/As.npz --rows persistence

``--anchor-file`` replaces the truth ``y[0]`` start with an anchor file's
``z0_mean`` (5, 14); the ``odesim`` row then also needs the file's
``settler_init`` (the 90 settler entries of the flat plant state) and refuses to
run without it, so no truth settler state can leak into a v1.1 baseline.
``--ras-filter-window`` and ``--influent-mode`` are recorded in summary.json;
the rows added in v1.1 group G3 consume them. Only ``exact`` influent is
accepted until the influent views exist.
"""

from __future__ import annotations

import argparse
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
ROWS = ("persistence", "odesim")
INFLUENT_MODES = ("exact",)


def sigma_tag(sigma: float) -> str:
    return ("%.2f" % sigma).replace(".", "p")


def scenario_arrays(scenario: str, raw: Path = RAW) -> dict[str, np.ndarray]:
    with np.load(Path(raw) / ("sim_%s.npz" % scenario)) as sim:
        return {k: sim[k].copy() for k in ("t", "y", "q_in", "influent", "reactor")}


def integrate_open_loop(plant: Bsm1Plant, data: dict[str, np.ndarray], y0: np.ndarray | None = None) -> np.ndarray:
    """Reactor trajectory (n, 5, 14) from ``y0`` (default: the saved y(0)) under the known influent."""
    t, q_in, z_in = data["t"], data["q_in"], data["influent"]

    def influent(tt: float) -> tuple[float, np.ndarray]:
        q = float(np.interp(tt, t, q_in))
        z = np.array([np.interp(tt, t, z_in[:, j]) for j in range(z_in.shape[1])])
        return q, z

    sol = solve_ivp(
        plant.rhs,
        (float(t[0]), float(t[-1])),
        data["y"][0] if y0 is None else y0,
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


def write_run(
    name: str,
    model: str,
    sigma: float,
    predictions: dict[str, np.ndarray],
    seconds: float,
    out: Path = OUT,
    extra: dict | None = None,
) -> None:
    run_dir = Path(out) / ("%s_sigma%s" % (name, sigma_tag(sigma)))
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
    if extra:
        summary.update(extra)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def load_anchor(path: Path, plant: Bsm1Plant) -> tuple[np.ndarray, np.ndarray | None]:
    """``(z0 (5, 14), settler (90,) or None)`` from an anchor ``.npz``."""
    with np.load(path) as anchor:
        z0 = np.asarray(anchor["z0_mean"], dtype=float)
        settler = np.asarray(anchor["settler_init"], dtype=float) if "settler_init" in anchor.files else None
    if z0.shape != (plant.cfg.n_tanks, plant.n_components):
        raise ValueError("anchor z0_mean has shape %s, expected (5, 14)" % (z0.shape,))
    n_settler = plant.state_size - z0.size
    if settler is not None and settler.reshape(-1).size != n_settler:
        raise ValueError("anchor settler_init has %d entries, expected %d" % (settler.size, n_settler))
    return z0, None if settler is None else settler.reshape(-1)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=str(RAW), help="directory with sim_{dry,rain}.npz")
    parser.add_argument("--out", default=str(OUT), help="parent directory of the run directories")
    parser.add_argument("--anchor-file", default=None,
                        help="anchor .npz (z0_mean, settler_init); default: truth y[0] as in v1.0")
    parser.add_argument("--rows", nargs="+", default=list(ROWS), choices=ROWS)
    parser.add_argument("--ras-filter-window", type=int, default=1)
    parser.add_argument("--influent-mode", default="exact", choices=INFLUENT_MODES)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    raw, out = Path(args.data_dir), Path(args.out)
    plant = Bsm1Plant()
    dry = scenario_arrays("dry", raw)
    rain = scenario_arrays("rain", raw)

    v11_meta: dict = {}
    if (args.anchor_file, args.ras_filter_window, args.influent_mode) != (None, 1, "exact"):
        v11_meta = {
            "anchor_file": args.anchor_file,
            "ras_filter_window": args.ras_filter_window,
            "influent_mode": args.influent_mode,
            "data_dir": str(raw),
        }

    if args.anchor_file is None:
        z0 = dry["reactor"][0]
        y0_dry, y0_rain = None, None
    else:
        z0, settler = load_anchor(Path(args.anchor_file), plant)
        if "odesim" in args.rows and settler is None:
            raise SystemExit("--anchor-file without settler_init cannot seed the odesim row")
        y0_dry = y0_rain = None if settler is None else np.concatenate([z0.reshape(-1), settler])

    written = 0
    preds: dict[str, dict[str, np.ndarray]] = {}
    if "persistence" in args.rows:
        preds["persistence"] = {
            **{k: np.tile(z0, (len(v), 1, 1)) for k, v in split_windows(dry["t"], dry["reactor"]).items()},
            "rain": np.tile(z0, (len(rain["t"]), 1, 1)),
        }
        for sigma in SIGMAS:
            write_run("persistence", "persistence", sigma, preds["persistence"], 0.0, out, v11_meta)
            written += 1

    if "odesim" in args.rows:
        started = time.perf_counter()
        dry_traj = integrate_open_loop(plant, dry, y0_dry)
        rain_traj = integrate_open_loop(plant, rain, y0_rain)
        ode_seconds = time.perf_counter() - started
        preds["ode_openloop"] = {**split_windows(dry["t"], dry_traj), "rain": rain_traj}
        for sigma in SIGMAS:
            write_run("odesim", "ode_openloop", sigma, preds["ode_openloop"], ode_seconds, out, v11_meta)
            written += 1

    # Convenience: report the headline numbers immediately.
    from src.eval.metrics import state_metrics, track_summary  # noqa: E402

    truth_hold = dry["reactor"][(dry["t"] >= HOLDOUT[0] - 1e-9)]
    for label, rows in preds.items():
        tracks = track_summary(state_metrics(truth_hold, rows["holdout"]))
        print(
            "%s  holdout  track_a_nrmse=%.4g  track_b_nrmse=%.4g"
            % (label, tracks["track_a_measured"]["nrmse"], tracks["track_b_unmeasured"]["nrmse"])
        )
    print("wrote %d baseline run dirs under %s" % (written, out))


if __name__ == "__main__":
    main()

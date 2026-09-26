"""Reference rows as run directories, scored by ``src.eval.report.collect_runs``.

``persistence``
    Holds the start state (truth y(0) in v1.0, the anchor mean in v1.1) constant.
``odesim`` (model ``ode_openloop``)
    v1.0 row, bit for bit: the full plant from the truth y(0) under the exact
    influent. It is the perfect-model bound and refuses an anchor file, so no
    truth start can leak into a v1.1 row.
``ode_openloop_reduced``
    Information-matched: the observers' reduced reactor train from the anchor
    mean, with the trailing-filtered measured RAS TSS and the influent view.
``ode_openloop_full``
    Structure-rich: the full plant (settler included) from the anchor's
    ``settler_init`` with the reactor part replaced by ``z0_mean``.

Usage::

    python -m scripts.make_baselines                       # v1.0 rows, bit for bit
    python -m scripts.make_baselines --data-dir results/raw_k100 --out results/v11/baselines/k100_ic_as \
        --anchor-file results/v11/anchors/k100/As.npz --influent-mode composite \
        --ras-filter-window 4 --rows persistence ode_openloop_reduced ode_openloop_full --sigmas 0.10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.asm1.plant import Bsm1Plant  # noqa: E402
from src.asm1.vault_loader import vault  # noqa: E402
from src.data.influent_views import MODES as INFLUENT_MODES, apply_influent_knowledge  # noqa: E402
from src.data.sensors import ObservationDataset  # noqa: E402
from src.observers.reduced_model import ReducedPlantModel  # noqa: E402
from src.train.curriculum import trailing_average  # noqa: E402
from src.train.run import RAS_CHANNEL  # noqa: E402

RAW = Path("results/raw")
OUT = Path("results/runs")
SIGMAS = (0.0, 0.05, 0.10, 0.15)
TRAIN_END = 12.0
HOLDOUT = (12.0, 14.0)
#: Looser than the generator's 1e-10 on purpose (v1.0): the baseline must not
#: reproduce the data file bit for bit through an identical solve.
RTOL = 1e-6
ATOL = 1e-8
ROWS = ("persistence", "odesim", "ode_openloop_reduced", "ode_openloop_full")
ROW_MODEL = {"persistence": "persistence", "odesim": "ode_openloop",
             "ode_openloop_reduced": "ode_openloop_reduced", "ode_openloop_full": "ode_openloop_full"}
ROW_INFORMATION = {"persistence": "anchor only", "odesim": "structure-rich, truth y(0) (v1.0)",
                   "ode_openloop_reduced": "information-matched", "ode_openloop_full": "structure-rich"}


def sigma_tag(sigma: float) -> str:
    return ("%.2f" % sigma).replace(".", "p")


def scenario_arrays(raw: Path, scenario: str) -> dict[str, Any]:
    with np.load(raw / ("sim_%s.npz" % scenario)) as sim:
        out = {k: sim[k].copy() for k in ("t", "y", "q_in", "influent", "reactor")}
        out["meta"] = json.loads(str(sim["meta"]))
    return out


def integrate_open_loop(plant: Bsm1Plant, data: dict[str, Any], y0: np.ndarray | None = None,
                        z_in: np.ndarray | None = None) -> np.ndarray:
    t, q_in = data["t"], data["q_in"]
    z_in = data["influent"] if z_in is None else z_in

    def influent(tt: float) -> tuple[float, np.ndarray]:
        return (float(np.interp(tt, t, q_in)),
                np.array([np.interp(tt, t, z_in[:, j]) for j in range(z_in.shape[1])]))

    sol = solve_ivp(plant.rhs, (float(t[0]), float(t[-1])), data["y"][0] if y0 is None else y0,
                    method="BDF", t_eval=t, rtol=RTOL, atol=ATOL, args=(influent,))
    if not sol.success:
        raise RuntimeError("open-loop integration failed: %s" % sol.message)
    return np.stack([plant.unpack(sol.y[:, i])[0] for i in range(sol.y.shape[1])])


def split_windows(t: np.ndarray, reactor: np.ndarray) -> dict[str, np.ndarray]:
    return {"train": reactor[t <= TRAIN_END + 1e-9],
            "holdout": reactor[(t >= HOLDOUT[0] - 1e-9) & (t <= HOLDOUT[1] + 1e-9)]}


def write_run(out: Path, name: str, model: str, sigma: float, predictions: dict[str, np.ndarray],
              seconds: float, extra: dict[str, Any] | None = None) -> Path:
    run_dir = out / ("%s_sigma%s" % (name, sigma_tag(sigma)))
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(run_dir / "predictions.npz", **predictions)
    summary = {"run_id": run_dir.name, "model": model, "arch": "analytic", "curriculum": "none",
               "noise": sigma, "seed": 0, "profile": "full", "train_end_day": TRAIN_END,
               "holdout_days": list(HOLDOUT), "steps": 0, "train_seconds": seconds,
               "n_parameters": 0, **(extra or {})}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return run_dir


def load_anchor(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with np.load(path, allow_pickle=False) as data:
        return {**{k: data[k] for k in data.files if k != "meta"},
                "meta": json.loads(str(data["meta"])) if "meta" in data.files else {}}


def ras_series(raw: Path, scenario: str, sigma: float, window: int) -> np.ndarray:
    ds = ObservationDataset.load(raw / ("obs_%s_sigma%s.npz" % (scenario, sigma_tag(sigma))))
    return trailing_average(ds.obs[:, list(ds.channels).index(RAS_CHANNEL)], window)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=RAW)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--anchor-file", type=Path, default=None, help="default: truth y(0), v1.0")
    parser.add_argument("--rows", nargs="+", default=["persistence", "odesim"], choices=ROWS)
    parser.add_argument("--ras-filter-window", type=int, default=1)
    parser.add_argument("--influent-mode", default="exact", choices=INFLUENT_MODES)
    parser.add_argument("--sigmas", nargs="+", type=float, default=list(SIGMAS))
    args = parser.parse_args(argv)

    raw, out, plant = args.data_dir, args.out, Bsm1Plant()
    dry = scenario_arrays(raw, "dry")
    # The random truths (results/raw_rand/<i>) hold the dry scenario only; their rows carry no rain key.
    rain = scenario_arrays(raw, "rain") if (raw / "sim_rain.npz").exists() else None
    scenarios = {"dry": dry, **({"rain": rain} if rain is not None else {})}
    anchor = load_anchor(args.anchor_file)
    if anchor is not None and "odesim" in args.rows:
        parser.error("odesim is the v1.0 truth-start row; with --anchor-file use ode_openloop_full")
    if "ode_openloop_full" in args.rows and anchor is not None and "settler_init" not in anchor:
        parser.error("ode_openloop_full needs the anchor's settler_init; refusing to borrow truth")
    z0 = dry["reactor"][0] if anchor is None else anchor["z0_mean"]
    views = {k: apply_influent_knowledge(d["t"], d["q_in"], d["influent"], args.influent_mode, vault())
             for k, d in scenarios.items()}
    extra = {"data_dir": raw.as_posix(),
             "anchor_file": None if anchor is None else args.anchor_file.as_posix(),
             "anchor": "truth_y0" if anchor is None else anchor["meta"].get("name"),
             "influent_mode": args.influent_mode, "ras_filter_window": args.ras_filter_window,
             "truth_preset": dry["meta"].get("truth_preset", "vault20"),
             "alpha": dry["meta"].get("alpha", 1.0)}
    rows: dict[str, dict[float, dict[str, np.ndarray]]] = {}
    seconds: dict[str, float] = {}
    started = time.perf_counter()
    if "persistence" in args.rows:
        p = {**{k: np.tile(z0, (len(v), 1, 1)) for k, v in split_windows(dry["t"], dry["reactor"]).items()},
             **({"rain": np.tile(z0, (len(rain["t"]), 1, 1))} if rain is not None else {})}
        rows["persistence"], seconds["persistence"] = {s: p for s in args.sigmas}, 0.0
    if "odesim" in args.rows:
        started = time.perf_counter()
        o = {**split_windows(dry["t"], integrate_open_loop(plant, dry)),
             **({"rain": integrate_open_loop(plant, rain)} if rain is not None else {})}
        rows["odesim"], seconds["odesim"] = {s: o for s in args.sigmas}, time.perf_counter() - started
    if "ode_openloop_full" in args.rows:
        y0 = (dry["y"][0] if anchor is None else np.asarray(anchor["settler_init"], float)).copy()
        y0[plant.idx.reactor] = np.asarray(z0, float).reshape(-1)
        started = time.perf_counter()
        f = {**split_windows(dry["t"], integrate_open_loop(plant, dry, y0, views["dry"])),
             **({"rain": integrate_open_loop(plant, rain, y0, views["rain"])} if rain is not None else {})}
        rows["ode_openloop_full"] = {s: f for s in args.sigmas}
        seconds["ode_openloop_full"] = time.perf_counter() - started
    if "ode_openloop_reduced" in args.rows:
        reduced, rows["ode_openloop_reduced"] = ReducedPlantModel(), {}
        started = time.perf_counter()
        for s in args.sigmas:
            traj = {k: reduced.integrate_bdf(z0, d["t"], d["q_in"], views[k],
                                             ras_series(raw, k, s, args.ras_filter_window))
                    for k, d in scenarios.items()}
            rows["ode_openloop_reduced"][s] = {**split_windows(dry["t"], traj["dry"]),
                                               **({"rain": traj["rain"]} if "rain" in traj else {})}
        seconds["ode_openloop_reduced"] = (time.perf_counter() - started) / len(args.sigmas)

    from src.eval.metrics import state_metrics, track_summary  # noqa: E402

    truth = split_windows(dry["t"], dry["reactor"])
    for row in args.rows:
        v10_schema = row in ("persistence", "odesim") and anchor is None and args.influent_mode == "exact"
        for s in args.sigmas:
            write_run(out, row, ROW_MODEL[row], s, rows[row][s], seconds[row],
                      None if v10_schema else {**extra, "information": ROW_INFORMATION[row]})
            b = {w: track_summary(state_metrics(truth[w], rows[row][s][w]))["track_b_unmeasured"]["nrmse"]
                 for w in ("train", "holdout")}
            print("%-22s sigma=%.2f  train track_b_nrmse=%.4f  holdout track_b_nrmse=%.4f"
                  % (row, s, b["train"], b["holdout"]))
    print("wrote %d baseline run dirs under %s" % (len(args.rows) * len(args.sigmas), out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

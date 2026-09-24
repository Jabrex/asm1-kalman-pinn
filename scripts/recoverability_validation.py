"""Does the recoverability index predict each estimator's per-state error? (H6)

Run after the G6 core runs::

    python -m scripts.recoverability_validation \
        --cells configs/analysis/recoverability_cells.yaml \
        --out results/v11/analysis/validation

Exploratory pilot on the v1.0 CL-PINN predictions (partial-derivative residual;
runs before G6 and is labelled as a pilot, never as a test of H6)::

    python -m scripts.recoverability_validation --pilot-v1 \
        --out results/v11/analysis/validation

Pre-registered protocol (copied verbatim into PREREGISTRATION.md by G6):

* points: the 55 never-measured (tank, component) entries of one cell;
* index: CRB/range over days 0-12 (primary) and 1 - IG (secondary), computed by
  ``scripts.recoverability.core_indices`` for the cell's data directory, anchor
  prior and influent view;
* error: per-(tank, component) NRMSE over days 0-12, normalised by the component
  range pooled over tanks on the same window (the fixed-spread normaliser of
  ``report.collect_runs``); CL-PINN uses the median over its seeds;
* statistic: Spearman rho, with a 95 % percentile interval from 1000
  component-cluster bootstrap resamples (the 11 components are drawn with
  replacement and each carries its five tanks, because tanks are correlated);
* pass (EKS, primary cell only): rho >= 0.5 and the interval's lower end > 0.
  The CL-PINN result and every other cell are reported either way.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.asm1.vault_loader import vault  # noqa: E402
from src.data.sensors import ObservationDataset, unobserved_components  # noqa: E402
from src.eval.metrics import per_tank_nrmse  # noqa: E402
from src.observers.reduced_model import ReducedPlantModel  # noqa: E402
from scripts import recoverability as rec  # noqa: E402

PASS_RHO = 0.5
N_BOOT = 1000
BOOT_SEED = 20260923
PILOT_RUNS = ("results/runs", "results/runs_seed1", "results/runs_seed2")


def cluster_bootstrap_spearman(index: np.ndarray, error: np.ndarray, n_boot: int = N_BOOT,
                               seed: int = BOOT_SEED) -> dict[str, Any]:
    """Spearman rho over (tank, component) points with a component-cluster bootstrap.

    ``index`` and ``error`` are ``(n_tanks, n_components)``; a resample draws
    components (columns) with replacement and keeps all tanks of each.
    """
    index = np.asarray(index, dtype=float)
    error = np.asarray(error, dtype=float)
    ok = np.isfinite(index) & np.isfinite(error)
    rho = float(spearmanr(index[ok], error[ok])[0]) if ok.sum() > 2 else float("nan")
    rng = np.random.default_rng(seed)
    n_c = index.shape[1]
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, n_c, n_c)
        a, b, m = index[:, pick], error[:, pick], ok[:, pick]
        if np.unique(a[m]).size < 2 or np.unique(b[m]).size < 2:
            continue
        boots.append(float(spearmanr(a[m], b[m])[0]))
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
    return {"rho": rho, "ci95": [float(lo), float(hi)], "n_points": int(ok.sum()),
            "n_boot_valid": len(boots), "n_boot": n_boot, "seed": seed}


def truth_window(data_dir: str, sigma: float) -> np.ndarray:
    ds = ObservationDataset.load(Path(data_dir) / ("obs_dry_sigma%s.npz" % rec.sigma_tag(sigma)))
    return ds.truth_reactor[ds.t <= rec.WINDOW_END_DAY + 1e-9]


def achieved_nrmse(truth: np.ndarray, pred: np.ndarray) -> np.ndarray:
    n = min(len(truth), len(pred))
    flat = truth[:n].reshape(-1, truth.shape[-1])
    spread = flat.max(axis=0) - flat.min(axis=0)
    return per_tank_nrmse(truth[:n], pred[:n], spread=spread)


def load_prediction(run_dir: Path) -> np.ndarray:
    with np.load(run_dir / "predictions.npz") as preds:
        return preds["train"].copy()


def cell_index(cell: Mapping[str, Any], sigma: float, substeps: int, cache: Path,
               model: Any, recompute: bool = False) -> dict[str, np.ndarray]:
    path = cache / ("index_%s.json" % cell["name"])
    if path.exists() and not recompute:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {k: np.asarray(raw[k], dtype=float) for k in ("crb_over_range", "ig", "tau")}
    case = rec.load_case(cell["data_dir"], sigma, cell.get("influent_mode", "exact"))
    p0, _ = rec.load_prior(cell["anchor"])
    core = rec.core_indices(case, p0, sigma, model, substeps)
    out = {"crb_over_range": core["crb"], "ig": core["ig"], "tau": core["tau"]}
    path.write_text(json.dumps(rec.clean({**out, "cell": dict(cell), "sigma": sigma,
                                          "substeps": substeps}), indent=1), encoding="utf-8")
    return out


def estimator_dirs(cell: Mapping[str, Any], spec: Mapping[str, Any]) -> list[Path]:
    if spec["kind"] == "pinn":
        return [Path(spec.get("root", "results/v11/pinn")) / ("%s_seed%d" % (cell["name"], s)) / spec["run"]
                for s in spec["seeds"]]
    if spec["kind"] == "observer":
        return [Path(spec.get("root", "results/v11/observers")) / cell["name"] / spec["run"]]
    if spec["kind"] == "explicit":
        return [Path(p) for p in spec["dirs"]]
    raise ValueError("unknown estimator kind %r" % spec["kind"])


def validate(cells: list[Mapping[str, Any]], estimators: Mapping[str, Mapping[str, Any]], sigma: float,
             primary: str | None, out: Path, substeps: int, n_boot: int, recompute: bool,
             label: str) -> dict[str, Any]:
    comps = list(vault().components)
    cols = [comps.index(c) for c in unobserved_components()]
    model = ReducedPlantModel()
    report: dict[str, Any] = {"label": label, "sigma": sigma, "primary_cell": primary,
                              "pass_rule": "EKS in the primary cell: rho >= %.2f and CI lower end > 0" % PASS_RHO,
                              "cells": {}}
    for cell in cells:
        index = cell_index(cell, sigma, substeps, out, model, recompute)
        truth = truth_window(cell["data_dir"], sigma)
        entry: dict[str, Any] = {"cell": dict(cell), "estimators": {}}
        for name, spec in estimators.items():
            dirs = estimator_dirs(cell, spec)
            missing = [str(d) for d in dirs if not (d / "predictions.npz").exists()]
            if missing:
                entry["estimators"][name] = {"status": "missing", "missing": missing}
                continue
            errors = np.stack([achieved_nrmse(truth, load_prediction(d)) for d in dirs])
            err = np.median(errors, axis=0)[:, cols]
            entry["estimators"][name] = {
                "status": "ok", "runs": [str(d) for d in dirs],
                "primary_index": cluster_bootstrap_spearman(index["crb_over_range"][:, cols], err, n_boot),
                "secondary_index": cluster_bootstrap_spearman(1.0 - index["ig"][:, cols], err, n_boot),
                "nrmse_never_measured_mean": float(np.nanmean(err)),
                "points": {"index": index["crb_over_range"][:, cols], "nrmse": err},
            }
        report["cells"][cell["name"]] = entry
    if primary is not None:
        eks = report["cells"].get(primary, {}).get("estimators", {}).get("eks", {})
        if eks.get("status") == "ok":
            stat = eks["primary_index"]
            report["h6"] = {"cell": primary, "estimator": "eks", **stat,
                            "pass": bool(stat["rho"] >= PASS_RHO and stat["ci95"][0] > 0.0)}
        else:
            report["h6"] = {"cell": primary, "estimator": "eks", "pass": None, "status": "missing"}
    return report


def scatter_figure(report: Mapping[str, Any], cell: str, path: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.eval.report import save_figure

    colours = ("#2a78d6", "#eb6834")
    fig, ax = plt.subplots(figsize=(3.6, 3.2), constrained_layout=True)
    for colour, (name, entry) in zip(colours, report["cells"][cell]["estimators"].items()):
        if entry.get("status") != "ok":
            continue
        x = np.asarray(entry["points"]["index"], dtype=float).ravel()
        y = np.asarray(entry["points"]["nrmse"], dtype=float).ravel()
        stat = entry["primary_index"]
        ax.scatter(x, y, s=18, color=colour, edgecolors="#fcfcfb", linewidths=0.8,
                   label="%s: rho %.2f [%.2f, %.2f]" % (name, stat["rho"], *stat["ci95"]))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("CRB / range (days 0-12)", fontsize=7)
    ax.set_ylabel("achieved NRMSE (days 0-12)", fontsize=7)
    ax.tick_params(labelsize=6.5)
    ax.grid(color="#e5e4df", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    ax.set_title(cell, fontsize=8, loc="left", color="#0b0b0b")
    paths = save_figure(fig, path)
    plt.close(fig)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cells", default="configs/analysis/recoverability_cells.yaml")
    parser.add_argument("--pilot-v1", action="store_true")
    parser.add_argument("--pilot-anchors", nargs="+",
                        default=["results/v11/anchors/k000/A0.npz", "results/v11/anchors/k000/As.npz"])
    parser.add_argument("--out", default="results/v11/analysis/validation")
    parser.add_argument("--substeps", type=int, default=9)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.pilot_v1:
        cells = [{"name": "v1pilot_k000_ie_%s" % Path(a).stem.lower(), "data_dir": "results/raw",
                  "anchor": a, "influent_mode": "exact"} for a in args.pilot_anchors]
        estimators = {"cl_pinn_v1": {"kind": "explicit",
                                     "dirs": ["%s/cl_pinn_sigma0p10" % root for root in PILOT_RUNS]}}
        report = validate(cells, estimators, 0.10, None, out, args.substeps, args.n_boot, args.recompute,
                          "exploratory pilot: v1.0 CL-PINN (partial-derivative residual), not a test of H6")
        path = out / "recoverability_validation_pilot_v1.json"
        figure_cell, figure_path = cells[0]["name"], out / "validation_pilot_v1.png"
    else:
        spec = yaml.safe_load(Path(args.cells).read_text(encoding="utf-8"))
        report = validate(spec["cells"], spec["estimators"], float(spec["sigma"]), spec["primary"], out,
                          args.substeps, args.n_boot, args.recompute, "pre-registered H6 validation")
        path = out / "recoverability_validation.json"
        figure_cell, figure_path = spec["primary"], out / ("validation_%s.png" % spec["primary"])
    if any(e.get("status") == "ok" for e in report["cells"][figure_cell]["estimators"].values()):
        report["figure"] = [str(p) for p in scatter_figure(report, figure_cell, figure_path)]
    path.write_text(json.dumps(rec.clean(report), indent=1), encoding="utf-8")
    print("validation -> %s" % path)
    if "h6" in report:
        print("H6: %s" % report["h6"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

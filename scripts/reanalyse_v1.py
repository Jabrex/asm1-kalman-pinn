"""Re-score the v1.0 run directories on the v1.1 windows and decision metrics.

EXPLORATORY: every PINN checkpoint scored here was trained on the v1.0
partial-derivative residual (see scripts/audit_derivative.py), so these numbers
are a pilot for the recoverability analysis (v1.1 group G4) and for choosing
metrics. They are not results and are superseded by the v1.1 runs.

Windows (days since the t = 0 anchor)
    R0  0-12   the 'train' predictions (primary reconstruction window)
    R2  2-12   the same predictions with the first two days dropped
    F   12-14  the 'holdout' predictions (forecast without data)

Per run and window: Track B NRMSE (own-window range, the v1.0 definition, and
the fixed training-window range), skill against persistence on the fixed
range, level error for X_B_H, X_B_A, X_I and X_P, the 10 % tolerance fraction
for plant-wide X_B_H and X_B_A, and per-day NRMSE over days 0-14 for all
components (X_I and X_P called out).

Usage::

    python -m scripts.reanalyse_v1        # writes results/v11/reanalysis_v1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.asm1.vault_loader import vault
from src.data.sensors import ObservationDataset, unobserved_components
from src.eval.metrics import (
    error_vs_time,
    level_error,
    skill_score,
    state_metrics,
    track_summary,
    within_tolerance_fraction,
)

RUN_DIRS = ("runs", "runs_seed1", "runs_seed2", "runs_ablation", "runs_flowonly", "runs_icmask")
LEVEL_COMPONENTS = ("X_B_H", "X_B_A", "X_I", "X_P")
MEMORY_COMPONENTS = ("X_I", "X_P")
LABEL = "exploratory, pre-fix checkpoints"


def sigma_tag(sigma: float) -> str:
    return ("%.2f" % float(sigma)).replace(".", "p")


def load_truth(data_dir: Path, sigma: float) -> ObservationDataset:
    return ObservationDataset.load(data_dir / ("obs_dry_sigma%s.npz" % sigma_tag(sigma)))


def windows(dry: ObservationDataset, preds: dict[str, np.ndarray]) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """``{name: (t, truth, pred)}`` for R0, R2 and F."""
    train = dry.window(0.0, 12.0)
    hold = dry.window(12.0, 14.0)
    n0 = min(len(train.t), len(preds["train"]))
    nf = min(len(hold.t), len(preds["holdout"]))
    t0, z0, p0 = train.t[:n0], train.truth_reactor[:n0], preds["train"][:n0]
    keep = t0 >= 2.0 - 1e-9
    return {
        "R0": (t0, z0, p0),
        "R2": (t0[keep], z0[keep], p0[keep]),
        "F": (hold.t[:nf], hold.truth_reactor[:nf], preds["holdout"][:nf]),
    }


def score(truth: np.ndarray, pred: np.ndarray, spread: np.ndarray) -> dict[str, Any]:
    own = track_summary(state_metrics(truth, pred))
    fixed = track_summary(state_metrics(truth, pred, spread=spread))
    comps = vault().components
    lev = level_error(truth, pred)
    return {
        "track_b_nrmse": own["track_b_unmeasured"]["nrmse"],
        "track_b_nrmse_fixed": fixed["track_b_unmeasured"]["nrmse"],
        "per_component_nrmse_fixed": {k: v["nrmse"] for k, v in fixed["per_component"].items()},
        "level_error": {c: float(lev[comps.index(c)]) for c in LEVEL_COMPONENTS},
        "tol10": within_tolerance_fraction(truth, pred, ("X_B_H", "X_B_A"), 0.10),
    }


def run_dirs(results: Path, dirs: tuple[str, ...]) -> list[tuple[str, Path]]:
    out = []
    for d in dirs:
        root = results / d
        if not root.is_dir():
            continue
        for run in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
            if (run / "summary.json").exists() and (run / "predictions.npz").exists():
                out.append((d, run))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=str(REPO / "results"))
    parser.add_argument("--data-dir", default=str(REPO / "results" / "raw"))
    parser.add_argument("--dirs", nargs="*", default=list(RUN_DIRS))
    parser.add_argument("--out", default=str(REPO / "results" / "v11" / "reanalysis_v1.json"))
    args = parser.parse_args(argv)
    results, data_dir = Path(args.results), Path(args.data_dir)

    truth_cache: dict[float, ObservationDataset] = {}
    rows: list[dict[str, Any]] = []
    memory: list[dict[str, Any]] = []
    track_b = list(unobserved_components())
    comps = list(vault().components)

    with np.load(results / "runs" / "persistence_sigma0p00" / "predictions.npz") as p:
        persist_preds = {k: p[k] for k in p.files}

    for results_dir, run in run_dirs(results, tuple(args.dirs)):
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        sigma = float(summary["noise"])
        dry = truth_cache.setdefault(sigma, load_truth(data_dir, sigma))
        flat = dry.window(0.0, 12.0).truth_reactor.reshape(-1, len(comps))
        spread = flat.max(axis=0) - flat.min(axis=0)
        with np.load(run / "predictions.npz") as p:
            preds = {k: p[k] for k in p.files}
        model_w = windows(dry, preds)
        persist_w = windows(dry, persist_preds)
        for name, (t, truth, pred) in model_w.items():
            s = score(truth, pred, spread)
            ref = score(truth, persist_w[name][2], spread)
            rows.append({
                "results_dir": results_dir,
                "run_id": summary["run_id"],
                "model": summary["model"],
                "noise": sigma,
                "seed": int(summary.get("seed", 0)),
                "window": name,
                "t_start": float(t[0]),
                "t_end": float(t[-1]),
                **s,
                "skill_vs_persistence": float(skill_score(s["track_b_nrmse_fixed"], ref["track_b_nrmse_fixed"])),
                "persistence_track_b_nrmse_fixed": ref["track_b_nrmse_fixed"],
            })
        t_all = np.concatenate([model_w["R0"][0], model_w["F"][0][1:]])
        z_all = np.concatenate([model_w["R0"][1], model_w["F"][1][1:]])
        p_all = np.concatenate([model_w["R0"][2], model_w["F"][2][1:]])
        ev = error_vs_time(z_all, p_all, t_all, bin_days=1.0, spread=spread)
        memory.append({
            "results_dir": results_dir,
            "run_id": summary["run_id"],
            "model": summary["model"],
            "noise": sigma,
            "seed": int(summary.get("seed", 0)),
            "day_start": ev["t_start"].tolist(),
            "nrmse_fixed": {c: ev["nrmse"][:, comps.index(c)].tolist() for c in comps},
            **{"nrmse_fixed_%s" % c: ev["nrmse"][:, comps.index(c)].tolist() for c in MEMORY_COMPONENTS},
        })

    report = {
        "label": LABEL,
        "note": "PINN rows come from v1.0 checkpoints trained on the partial-derivative residual; "
                "use for metric design and the G4 pilot only.",
        "windows": {"R0": [0.0, 12.0], "R2": [2.0, 12.0], "F": [12.0, 14.0]},
        "nrmse_fixed_spread": "range of the truth over days 0-12, per component, pooled over tanks",
        "skill": "1 - TrackB_NRMSE_fixed(model) / TrackB_NRMSE_fixed(persistence), same window",
        "track_b_components": track_b,
        "rows": rows,
        "error_vs_time": memory,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("%d rows from %d run directories -> %s" % (len(rows), len(memory), out))
    for r in rows:
        if r["results_dir"] == "runs" and r["noise"] == 0.10 and r["window"] in ("R0", "F"):
            print("  %-26s %-3s TrackB %.3f (fixed %.3f)  skill %+.2f  tol10 X_B_H %.2f X_B_A %.2f"
                  % (r["run_id"], r["window"], r["track_b_nrmse"], r["track_b_nrmse_fixed"],
                     r["skill_vs_persistence"], r["tol10"]["X_B_H"], r["tol10"]["X_B_A"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

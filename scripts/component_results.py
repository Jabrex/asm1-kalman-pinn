"""Per-component and per-tank Track B results (co-author request, 2026-09-03).

Reads finished runs only; nothing is trained. Outputs
  results/component_table.json           per-component holdout NRMSE for every model and sigma;
                                         PINN entries are the median over seeds 0-2
  results/component_table.tex            LaTeX body rows, Track B components at sigma = 0.10
  results/figures/per_tank_heatmap.*     5 tanks x 14 components, cl_pinn vs pinn, sigma = 0.10,
                                         median over seeds, fixed training-window range
  results/figures/trajectories_trackB.*  X_B_H, X_S, S_ND in tanks 1 and 5 over the holdout,
                                         truth vs cl_pinn vs pinn (seed 0) vs persistence, sigma = 0.10
Usage: python -m scripts.component_results
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.sensors import ObservationDataset, unobserved_components  # noqa: E402
from src.eval.metrics import per_tank_nrmse, state_metrics  # noqa: E402
from src.eval.report import save_figure  # noqa: E402

RAW = Path("results/raw")
SEED_DIRS = {0: Path("results/runs"), 1: Path("results/runs_seed1"), 2: Path("results/runs_seed2")}
SEEDED = ("cl_pinn", "pinn")
SINGLE = ("cl_lstm", "lstm", "persistence", "odesim")
SIGMAS = (0.0, 0.05, 0.10, 0.15)
HOLDOUT = (12.0, 14.0)
TRAIN_END = 12.0
HEAT_SIGMA = 0.10
LABELS = {
    "S_I": r"$S_I$", "S_S": r"$S_S$", "X_I": r"$X_I$", "X_S": r"$X_S$", "X_B_H": r"$X_{B,H}$",
    "X_B_A": r"$X_{B,A}$", "X_P": r"$X_P$", "S_O": r"$S_O$", "S_NO": r"$S_{NO}$", "S_NH": r"$S_{NH}$",
    "S_ND": r"$S_{ND}$", "X_ND": r"$X_{ND}$", "S_ALK": r"$S_{ALK}$", "S_N2": r"$S_{N2}$",
}


def tag(sigma: float) -> str:
    return ("%.2f" % sigma).replace(".", "p")


def truth_for(sigma: float) -> tuple[ObservationDataset, np.ndarray]:
    """Holdout truth and the fixed (training-window, pooled over tanks) range."""
    ds = ObservationDataset.load(RAW / ("obs_dry_sigma%s.npz" % tag(sigma)))
    train = ds.window(0.0, TRAIN_END).truth_reactor
    flat = train.reshape(-1, train.shape[-1])
    return ds.window(*HOLDOUT), flat.max(axis=0) - flat.min(axis=0)


def load_pred(run_dir: Path, model: str, sigma: float) -> np.ndarray | None:
    p = run_dir / ("%s_sigma%s" % (model, tag(sigma))) / "predictions.npz"
    if not p.exists():
        return None
    with np.load(p) as d:
        return d["holdout"]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    components: tuple[str, ...] | None = None
    table: dict[str, dict[str, dict[str, float]]] = {}
    heat: dict[str, np.ndarray] = {}
    for sigma in SIGMAS:
        hold, fixed = truth_for(sigma)
        truth = hold.truth_reactor
        for model in SEEDED + SINGLE:
            dirs = list(SEED_DIRS.values()) if model in SEEDED else [SEED_DIRS[0]]
            per_comp, per_tank = [], []
            for run_dir in dirs:
                pred = load_pred(run_dir, model, sigma)
                if pred is None:
                    continue
                n = min(len(pred), len(truth))
                m = state_metrics(truth[:n], pred[:n])
                components = m.components
                per_comp.append(m.nrmse)
                per_tank.append(per_tank_nrmse(truth[:n], pred[:n], spread=fixed))
            if not per_comp:
                continue
            med = np.median(np.stack(per_comp), axis=0)
            table.setdefault(model, {})[tag(sigma)] = {
                c: float(med[i]) for i, c in enumerate(components)
            }
            table[model][tag(sigma)]["_n_seeds"] = float(len(per_comp))
            if model in SEEDED and sigma == HEAT_SIGMA:
                heat[model] = np.median(np.stack(per_tank), axis=0)
    assert components is not None
    Path("results/component_table.json").write_text(json.dumps(table, indent=2), encoding="utf-8")

    # --- LaTeX rows: Track B components at sigma = 0.10 ------------------------
    track_b = unobserved_components()
    cols = ("cl_pinn", "pinn", "cl_lstm", "lstm", "persistence")
    key = tag(HEAT_SIGMA)
    lines = []
    for c in track_b:
        vals = [table[m][key][c] for m in cols]
        best = int(np.nanargmin(vals[:4]))
        cells = ["\\textbf{%.3f}" % v if i == best else "%.3f" % v for i, v in enumerate(vals)]
        lines.append("%s & %s \\\\" % (LABELS[c], " & ".join(cells)))
    pooled = [float(np.nanmean([table[m][key][c] for c in track_b])) for m in cols]
    lines.append("\\midrule")
    lines.append("Track~B mean & %s \\\\" % " & ".join("%.3f" % v for v in pooled))
    Path("results/component_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    # --- heatmap ---------------------------------------------------------------
    # Two stacked panels: at text width (about 16 cm) the cell annotations stay
    # legible, which they do not in a side-by-side layout.
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.8), sharex=True)
    vmax = 1.0
    im = None
    for ax, model, title in zip(axes, SEEDED, ("Curriculum PINN", "Single-stage PINN")):
        grid = heat[model]
        im = ax.imshow(np.clip(grid, 0, vmax), cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(components)))
        ax.set_xticklabels([LABELS[c] for c in components], rotation=45, fontsize=9)
        ax.set_yticks(range(5))
        ax.set_yticklabels(["tank %d" % (k + 1) for k in range(5)], fontsize=9)
        ax.set_title(title, fontsize=10)
        for k in range(5):
            for i in range(len(components)):
                val = grid[k, i]
                ax.text(
                    i, k, "%.2f" % val if np.isfinite(val) else "n/a",
                    ha="center", va="center", fontsize=7,
                    color="white" if (not np.isfinite(val) or val < 0.55 * vmax) else "black",
                )
    cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label("holdout NRMSE (fixed training-window range), capped at %.1f" % vmax, fontsize=9)
    save_figure(fig, Path("results/figures/per_tank_heatmap.png"))
    plt.close(fig)
    print("heatmap max cell: cl_pinn %.3f | pinn %.3f" % (np.nanmax(heat["cl_pinn"]), np.nanmax(heat["pinn"])))
    for model in SEEDED:
        print("%s per-tank Track B mean by tank:" % model,
              np.round(np.nanmean(heat[model][:, [components.index(c) for c in track_b]], axis=1), 3))

    # --- trajectories ------------------------------------------------------------
    hold, _ = truth_for(HEAT_SIGMA)
    t = hold.t
    truth = hold.truth_reactor
    preds = {m: load_pred(SEED_DIRS[0], m, HEAT_SIGMA) for m in ("cl_pinn", "pinn", "persistence")}
    show = ("X_B_H", "X_S", "S_ND")
    tanks = (0, 4)
    fig, axes = plt.subplots(len(show), len(tanks), figsize=(8.0, 6.6), sharex=True)
    style = {
        "cl_pinn": ("tab:orange", "-", "curriculum PINN"),
        "pinn": ("tab:red", "-", "single-stage PINN"),
        "persistence": ("0.4", ":", "persistence"),
    }
    for r, comp in enumerate(show):
        ci = components.index(comp)
        for c_, k in enumerate(tanks):
            ax = axes[r, c_]
            ax.plot(t, truth[:, k, ci], color="black", lw=1.4, label="ground truth")
            for m, (col, ls, lab) in style.items():
                if preds[m] is not None:
                    nn = min(len(t), len(preds[m]))
                    ax.plot(t[:nn], preds[m][:nn, k, ci], color=col, ls=ls, lw=1.1, label=lab)
            ax.set_title("%s, tank %d" % (LABELS[comp], k + 1), fontsize=10)
            ax.set_ylabel("g m$^{-3}$", fontsize=9)
            ax.tick_params(labelsize=9)
            if r == len(show) - 1:
                ax.set_xlabel("time (d)", fontsize=10)
    axes[0, 0].legend(fontsize=8, loc="best")
    fig.suptitle(r"Never-measured states on the holdout, $\sigma = 0.10$, seed 0", fontsize=11)
    fig.tight_layout()
    save_figure(fig, Path("results/figures/trajectories_trackB.png"))
    plt.close(fig)
    print("wrote results/component_table.{json,tex}, per_tank_heatmap, trajectories_trackB")


if __name__ == "__main__":
    main()

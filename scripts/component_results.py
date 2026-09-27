"""Per-component and per-tank Track B results (co-author request, 2026-09-03).

Reads finished runs only; nothing is trained. With no arguments it reproduces
the v1.0 outputs exactly:
  results/component_table.json           per-component holdout NRMSE for every model and sigma;
                                         multi-seed entries are the median over the seeds found
  results/component_table.tex            LaTeX body rows, Track B components at sigma = 0.10
  results/figures/per_tank_heatmap.*     5 tanks x 14 components, cl_pinn vs pinn, sigma = 0.10,
                                         median over seeds, fixed training-window range
  results/figures/trajectories_trackB.*  X_B_H, X_S, S_ND in tanks 1 and 5 over the holdout,
                                         truth vs cl_pinn vs pinn vs eks vs persistence, each from the
                                         first root that holds it, sigma = 0.10

v1.1 cells pass their own run roots, data directory and an output tag, e.g.
  python -m scripts.component_results --runs results/v11/pinn/k100_ic_as_seed0
      results/v11/pinn/k100_ic_as_seed1 results/v11/pinn/k100_ic_as_seed2
      results/v11/baselines/k100_ic_as --data-dir results/raw_k100 --sigmas 0.10
      --models cl_pinn persistence ode_openloop_reduced --heat-models cl_pinn
      --table-models cl_pinn persistence ode_openloop_reduced --window train
      --out-dir results/v11 --tag _k100_ic_as_R0
Every root is searched for <model>_sigma<tag>/predictions.npz; a model found
under several roots (one per seed) is summarised by the median.

--figure-layout split (v1.1) writes one heatmap per model and one trajectory
figure per component, each drawn at its printed width with no text below 7 pt
(scripts/figure_layout.py refuses cut-off or colliding text):
  <out>/figures/per_tank_heatmap<tag>_<model>.*   <out>/figures/trajectories_trackB<tag>_<component>.*
The default (v10) keeps the combined v1.0 figures.
"""
from __future__ import annotations

import argparse
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
#: Window names as they read in figure text (the v1.0 wording for the holdout).
WINDOW_TEXT = {"holdout": "holdout", "train": "estimation window (days 0-12)"}
TABLE_MODELS = ("cl_pinn", "pinn", "cl_lstm", "lstm", "persistence")
#: Non-learned rows: never marked bold as "best learned model".
BASELINE_MODELS = ("persistence", "odesim", "ode_openloop", "ode_openloop_reduced", "ode_openloop_full")
TITLES = {"cl_pinn": "Curriculum PINN", "pinn": "Single-stage PINN", "cl_pinn_theta": "PINN with kinetic multipliers",
          "eks": "Extended Kalman smoother", "eks_aug": "Augmented extended Kalman smoother"}
LABELS = {
    "S_I": r"$S_I$", "S_S": r"$S_S$", "X_I": r"$X_I$", "X_S": r"$X_S$", "X_B_H": r"$X_{B,H}$",
    "X_B_A": r"$X_{B,A}$", "X_P": r"$X_P$", "S_O": r"$S_O$", "S_NO": r"$S_{NO}$", "S_NH": r"$S_{NH}$",
    "S_ND": r"$S_{ND}$", "X_ND": r"$X_{ND}$", "S_ALK": r"$S_{ALK}$", "S_N2": r"$S_{N2}$",
}


def tag(sigma: float) -> str:
    return ("%.2f" % sigma).replace(".", "p")


def truth_for(
    sigma: float, data_dir: Path = RAW, window: str = "holdout"
) -> tuple[ObservationDataset, np.ndarray]:
    """Window truth and the fixed (training-window, pooled over tanks) range."""
    ds = ObservationDataset.load(Path(data_dir) / ("obs_dry_sigma%s.npz" % tag(sigma)))
    train = ds.window(0.0, TRAIN_END).truth_reactor
    flat = train.reshape(-1, train.shape[-1])
    view = ds.window(*HOLDOUT) if window == "holdout" else ds.window(0.0, TRAIN_END)
    return view, flat.max(axis=0) - flat.min(axis=0)


def load_pred(run_dir: Path, model: str, sigma: float, window: str = "holdout") -> np.ndarray | None:
    p = Path(run_dir) / ("%s_sigma%s" % (model, tag(sigma))) / "predictions.npz"
    if not p.exists():
        return None
    with np.load(p) as d:
        return d[window]


def first_pred(roots: list[Path], model: str, sigma: float, window: str = "holdout") -> np.ndarray | None:
    """Prediction from the first root that holds ``model`` (seed 0 comes first on the command line)."""
    for root in roots:
        pred = load_pred(root, model, sigma, window)
        if pred is not None:
            return pred
    return None


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", default=[str(p) for p in SEED_DIRS.values()],
                        help="run-directory roots; each trajectory comes from the first root that holds the model")
    parser.add_argument("--data-dir", default=str(RAW))
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--tag", default="", help="suffix for every output file name")
    parser.add_argument("--window", choices=("holdout", "train"), default="holdout")
    parser.add_argument("--sigmas", nargs="+", type=float, default=list(SIGMAS))
    parser.add_argument("--models", nargs="+", default=list(SEEDED + SINGLE))
    parser.add_argument("--heat-models", nargs="+", default=list(SEEDED))
    parser.add_argument("--table-models", nargs="+", default=list(TABLE_MODELS))
    parser.add_argument("--heat-sigma", type=float, default=HEAT_SIGMA)
    parser.add_argument("--trackb-row", choices=("mean-of-medians", "median-of-means"), default="mean-of-medians",
                        help="'Track B mean' row: v1.0 mean of the per-component medians (default), or the median "
                             "over seeds of each seed's Track B mean, the primary metric of the v1.1 regime table")
    parser.add_argument("--cell-label", default="",
                        help="split layout: the cell as set in the figure titles, e.g. 'K0 Ie A0'")
    parser.add_argument("--figure-layout", choices=("v10", "split"), default="v10",
                        help="v10: the combined v1.0 figures (default); split: one print-size file per model "
                             "(heatmap) and per component (trajectories)")
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args(argv)


#: Split-layout text (typographic dash) and the estimator names of the other v1.1 figures.
SPLIT_WINDOW_TEXT = {"holdout": "held-out days 12\u201314", "train": "days 0\u201312"}
SPLIT_TITLES = {"cl_pinn": "CL-PINN", "pinn": "PINN, single-stage", "cl_pinn_theta": "PINN-$\\theta$",
                "eks": "EKS", "eks_aug": "Aug. EKS"}
#: Drawn in this order, the CL-PINN last and on top: (colour, line style, width, name).
TRAJECTORY_STYLE = {
    "persistence": ("0.4", ":", 0.9, "Persistence"),
    "eks": ("tab:blue", "--", 0.9, "EKS"),
    "pinn": ("tab:red", "-.", 0.8, "PINN, single-stage"),
    "cl_pinn": ("tab:orange", "-", 1.1, "CL-PINN"),
}
#: The y axis is fitted to the curves after this time, so that one start value cannot flatten the panel.
TRAJECTORY_YFIT_AFTER_D = 0.05
TRAJECTORY_COMPONENTS = ("X_B_H", "X_S", "S_ND")
TRAJECTORY_TANKS = (0, 4)


def split_heatmaps(heat: dict[str, np.ndarray], models: list[str], components: tuple[str, ...], window: str,
                   sigma: float, fig_dir: Path, file_tag: str, cell_label: str = "") -> list[Path]:
    """One per-tank heatmap per model, full page width."""
    from scripts.figure_layout import DOUBLE_IN, print_style, save_checked

    written: list[Path] = []
    vmax = 1.0
    with print_style():
        for model in models:
            grid = heat[model]
            fig, ax = plt.subplots(figsize=(DOUBLE_IN, 2.3), layout="constrained")
            im = ax.imshow(np.clip(grid, 0, vmax), cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
            ax.set_xticks(range(len(components)))
            ax.set_xticklabels([LABELS[c] for c in components])
            ax.set_yticks(range(5))
            ax.set_yticklabels(["tank %d" % (k + 1) for k in range(5)])
            ax.tick_params(length=0)
            for k in range(5):
                for i in range(len(components)):
                    val = grid[k, i]
                    ax.text(i, k, "%.2f" % val if np.isfinite(val) else "n/a", ha="center", va="center", fontsize=7,
                            color="white" if (not np.isfinite(val) or val < 0.55 * vmax) else "black")
            ax.set_title("%s%s: NRMSE (fixed R0 range) per tank and component, %s, $\\sigma = %.2f$"
                         % (SPLIT_TITLES.get(model, model), ", " + cell_label if cell_label else "",
                            SPLIT_WINDOW_TEXT[window], sigma), loc="left")
            cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
            cbar.set_label("NRMSE,\ncolor capped at %.1f" % vmax)
            try:
                written += save_checked(fig, fig_dir / ("per_tank_heatmap%s_%s.png" % (file_tag, model)), save_figure)
            finally:
                plt.close(fig)
    return written


def split_trajectories(t: np.ndarray, truth: np.ndarray, preds: dict[str, np.ndarray | None],
                       components: tuple[str, ...], window: str, sigma: float, fig_dir: Path,
                       file_tag: str, cell_label: str = "") -> list[Path]:
    """One figure per never-measured component: tanks 1 and 5 side by side, legend below."""
    from scripts.figure_layout import DOUBLE_IN, print_style, save_checked

    written: list[Path] = []
    with print_style():
        for comp in TRAJECTORY_COMPONENTS:
            ci = components.index(comp)
            fig, axes = plt.subplots(1, len(TRAJECTORY_TANKS), figsize=(DOUBLE_IN, 2.4), sharex=True,
                                     layout="constrained")
            cut = False
            for n, (ax, k) in enumerate(zip(axes, TRAJECTORY_TANKS)):
                ax.plot(t, truth[:, k, ci], color="black", lw=1.2, label="ground truth")
                series = [truth[:, k, ci]]
                for z, (m, (col, ls, lw, lab)) in enumerate(TRAJECTORY_STYLE.items()):
                    if preds.get(m) is not None:
                        nn = min(len(t), len(preds[m]))
                        ax.plot(t[:nn], preds[m][:nn, k, ci], color=col, ls=ls, lw=lw, label=lab, zorder=2 + z)
                        series.append(np.pad(preds[m][:nn, k, ci], (0, len(t) - nn), constant_values=np.nan))
                stack = np.vstack(series)
                late = stack[:, t > TRAJECTORY_YFIT_AFTER_D]
                lo, hi = np.nanmin(late), np.nanmax(late)
                pad = 0.06 * (hi - lo if hi > lo else abs(hi) or 1.0)
                ax.set_ylim(lo - pad, hi + pad)
                cut = cut or bool(np.nanmin(stack) < lo - pad or np.nanmax(stack) > hi + pad)
                ax.set_title("(%s) tank %d" % ("ab"[n], k + 1), loc="left")
                ax.set_ylabel("%s (g m$^{-3}$)" % LABELS[comp])
                ax.set_xlabel("time (d)")
            fig.suptitle("%s%s, %s, $\\sigma = %.2f$, seed 0%s"
                         % (LABELS[comp], ", " + cell_label if cell_label else "", SPLIT_WINDOW_TEXT[window], sigma,
                            "; start values beyond the axis are cut" if cut else ""), x=0.01, ha="left")
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="outside lower center", ncol=len(labels))
            try:
                written += save_checked(fig, fig_dir / ("trajectories_trackB%s_%s.png" % (file_tag, comp)),
                                        save_figure)
            finally:
                plt.close(fig)
    return written


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    roots = [Path(r) for r in args.runs]
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    components: tuple[str, ...] | None = None
    table: dict[str, dict[str, dict[str, float]]] = {}
    heat: dict[str, np.ndarray] = {}
    seed_means: dict[str, dict[str, list[float]]] = {}
    track_b_names = unobserved_components()
    for sigma in args.sigmas:
        hold, fixed = truth_for(sigma, Path(args.data_dir), args.window)
        truth = hold.truth_reactor
        for model in args.models:
            per_comp, per_tank = [], []
            for run_dir in roots:
                pred = load_pred(run_dir, model, sigma, args.window)
                if pred is None:
                    continue
                n = min(len(pred), len(truth))
                m = state_metrics(truth[:n], pred[:n])
                components = m.components
                per_comp.append(m.nrmse)
                seed_means.setdefault(model, {}).setdefault(tag(sigma), []).append(
                    float(np.nanmean(m.nrmse[[components.index(c) for c in track_b_names]])))
                per_tank.append(per_tank_nrmse(truth[:n], pred[:n], spread=fixed))
            if not per_comp:
                continue
            med = np.median(np.stack(per_comp), axis=0)
            table.setdefault(model, {})[tag(sigma)] = {
                c: float(med[i]) for i, c in enumerate(components)
            }
            table[model][tag(sigma)]["_n_seeds"] = float(len(per_comp))
            if model in args.heat_models and np.isclose(sigma, args.heat_sigma):
                heat[model] = np.median(np.stack(per_tank), axis=0)
    if components is None:
        raise SystemExit("no predictions found under %s" % ", ".join(map(str, roots)))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ("component_table%s.json" % args.tag)).write_text(json.dumps(table, indent=2), encoding="utf-8")

    # --- LaTeX rows: Track B components at the heat sigma ---------------------
    track_b = unobserved_components()
    cols = [m for m in args.table_models if tag(args.heat_sigma) in table.get(m, {})]
    key = tag(args.heat_sigma)
    learned = [i for i, m in enumerate(cols) if m not in BASELINE_MODELS]
    lines = []
    for c in track_b:
        vals = [table[m][key][c] for m in cols]
        best = learned[int(np.nanargmin([vals[i] for i in learned]))] if learned else -1
        cells = ["\\textbf{%.3f}" % v if i == best else "%.3f" % v for i, v in enumerate(vals)]
        lines.append("%s & %s \\\\" % (LABELS[c], " & ".join(cells)))
    if args.trackb_row == "median-of-means":
        pooled = [float(np.median(seed_means[m][key])) for m in cols]
        row_name = "Track~B (median over seeds)"
    else:
        pooled = [float(np.nanmean([table[m][key][c] for c in track_b])) for m in cols]
        row_name = "Track~B mean"
    lines.append("\\midrule")
    lines.append("%s & %s \\\\" % (row_name, " & ".join("%.3f" % v for v in pooled)))
    (out_dir / ("component_table%s.tex" % args.tag)).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if args.no_figures:
        return

    # --- heatmap ---------------------------------------------------------------
    # Stacked panels: at text width (about 16 cm) the cell annotations stay
    # legible, which they do not in a side-by-side layout.
    heat_models = [m for m in args.heat_models if m in heat]
    if args.figure_layout == "split":
        fig_dir.mkdir(parents=True, exist_ok=True)
        written = split_heatmaps(heat, heat_models, components, args.window, args.heat_sigma, fig_dir, args.tag,
                                 args.cell_label)
        for model in heat_models:
            print("%s heatmap max cell %.3f; per-tank Track B mean by tank:" % (model, np.nanmax(heat[model])),
                  np.round(np.nanmean(heat[model][:, [components.index(c) for c in track_b]], axis=1), 3))
        hold, _ = truth_for(args.heat_sigma, Path(args.data_dir), args.window)
        preds = {m: first_pred(roots, m, args.heat_sigma, args.window) for m in TRAJECTORY_STYLE}
        written += split_trajectories(hold.t, hold.truth_reactor, preds, components, args.window, args.heat_sigma,
                                      fig_dir, args.tag, args.cell_label)
        for path in written:
            print("wrote", path)
        return
    if heat_models:
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(len(heat_models), 1, figsize=(7.2, 2.9 * len(heat_models)), sharex=True,
                                 squeeze=False)
        vmax = 1.0
        im = None
        for ax, model in zip(axes[:, 0], heat_models):
            grid = heat[model]
            im = ax.imshow(np.clip(grid, 0, vmax), cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
            ax.set_xticks(range(len(components)))
            ax.set_xticklabels([LABELS[c] for c in components], rotation=45, fontsize=9)
            ax.set_yticks(range(5))
            ax.set_yticklabels(["tank %d" % (k + 1) for k in range(5)], fontsize=9)
            ax.set_title(TITLES.get(model, model), fontsize=10)
            for k in range(5):
                for i in range(len(components)):
                    val = grid[k, i]
                    ax.text(
                        i, k, "%.2f" % val if np.isfinite(val) else "n/a",
                        ha="center", va="center", fontsize=7,
                        color="white" if (not np.isfinite(val) or val < 0.55 * vmax) else "black",
                    )
        cbar = fig.colorbar(im, ax=axes[:, 0].tolist(), fraction=0.03, pad=0.02)
        cbar.set_label("%s NRMSE (fixed training-window range), capped at %.1f" % (WINDOW_TEXT[args.window], vmax),
                       fontsize=9)
        save_figure(fig, fig_dir / ("per_tank_heatmap%s.png" % args.tag))
        plt.close(fig)
        for model in heat_models:
            print("%s heatmap max cell %.3f; per-tank Track B mean by tank:" % (model, np.nanmax(heat[model])),
                  np.round(np.nanmean(heat[model][:, [components.index(c) for c in track_b]], axis=1), 3))

    # --- trajectories ------------------------------------------------------------
    hold, _ = truth_for(args.heat_sigma, Path(args.data_dir), args.window)
    t = hold.t
    truth = hold.truth_reactor
    style = {
        "cl_pinn": ("tab:orange", "-", "curriculum PINN"),
        "pinn": ("tab:red", "-", "single-stage PINN"),
        "eks": ("tab:blue", "--", "extended Kalman smoother"),
        "persistence": ("0.4", ":", "persistence"),
    }
    preds = {m: first_pred(roots, m, args.heat_sigma, args.window) for m in style}
    show = ("X_B_H", "X_S", "S_ND")
    tanks = (0, 4)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(show), len(tanks), figsize=(8.0, 6.6), sharex=True)
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
    fig.suptitle(r"Never-measured states on the %s, $\sigma = %.2f$, seed 0"
                 % (WINDOW_TEXT[args.window], args.heat_sigma), fontsize=11)
    fig.tight_layout()
    save_figure(fig, fig_dir / ("trajectories_trackB%s.png" % args.tag))
    plt.close(fig)
    print("wrote %s/component_table%s.{json,tex}, per_tank_heatmap, trajectories_trackB" % (out_dir, args.tag))


if __name__ == "__main__":
    main()

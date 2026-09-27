"""Aggregate multi-seed PINN runs into min-max bands.

Scans the given run roots (one per seed), groups rows by (model, noise,
eval_set), and writes ``seed_bands<tag>.json`` plus a banded version of the
noise-robustness figure. Context models (the v1.0 LSTMs) stay single-seed and
appear as plain lines from the first root.

    python -m scripts.seed_bands                       # v1.0 outputs, unchanged
    python -m scripts.seed_bands --runs results/v11/pinn/k100_ic_as_seed0
        results/v11/pinn/k100_ic_as_seed1 results/v11/pinn/k100_ic_as_seed2
        --data-dir results/raw_k100 --band-models cl_pinn --context-models
        --figure-set train --figure-metric track_b_nrmse_fixed
        --out-dir results/v11 --tag _k100_ic_as

A run with a variant (the E3 sensor variants) gets its own band, keyed
``<model>[<variant>]|<sigma>|<set>``; plain runs keep the v1.0 key.

--figure-layout column (v1.1) draws the band figure at one journal column
(85 mm) with no text below 7 pt and no title (the caption carries it);
scripts/figure_layout.py refuses cut-off or colliding text. The default (v10)
keeps the v1.0 figure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.report import collect_runs, save_figure

RAW = Path("results/raw")
SEED_DIRS = {
    0: Path("results/runs"),
    1: Path("results/runs_seed1"),
    2: Path("results/runs_seed2"),
}
BAND_MODELS = ("cl_pinn", "pinn")
CONTEXT_MODELS = ("cl_lstm", "lstm")
METRICS = ("track_a_nrmse", "track_a_r2", "track_b_nrmse", "track_b_r2", "track_b_nrmse_fixed")
SET_TEXT = {"holdout": "holdout", "train": "estimation window, days 0-12", "rain": "rain event"}
METRIC_TEXT = {"track_b_nrmse": "Track B NRMSE", "track_b_nrmse_fixed": "Track B NRMSE (fixed range)",
               "track_a_nrmse": "Track A NRMSE"}
COLORS = {"cl_pinn": "tab:orange", "pinn": "tab:red", "cl_pinn_theta": "tab:purple",
          "cl_lstm": "tab:blue", "lstm": "tab:green"}


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", default=[str(p) for p in SEED_DIRS.values()])
    parser.add_argument("--data-dir", default=str(RAW))
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--tag", default="")
    parser.add_argument("--band-models", nargs="+", default=list(BAND_MODELS))
    parser.add_argument("--context-models", nargs="*", default=list(CONTEXT_MODELS))
    parser.add_argument("--figure-set", default="holdout", choices=sorted(SET_TEXT),
                        help="evaluation set drawn in the band figure (v1.0: holdout)")
    parser.add_argument("--figure-metric", default="track_b_nrmse", choices=sorted(METRIC_TEXT),
                        help="metric drawn in the band figure (v1.0: track_b_nrmse)")
    parser.add_argument("--cell-label", default="", help="column layout: the cell in the title, e.g. 'K.5 Ic As'")
    parser.add_argument("--figure-layout", choices=("v10", "column"), default="v10",
                        help="v10: the v1.0 figure (default); column: one journal column, print-size text")
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args(argv)


PRETTY = {"cl_pinn": "CL-PINN", "pinn": "PINN, single-stage", "cl_pinn_theta": "PINN-$\\theta$",
          "cl_lstm": "CL-LSTM", "lstm": "LSTM"}
SET_TEXT_SHORT = {"holdout": "days 12\u201314", "train": "days 0\u201312", "rain": "rain event"}
METRIC_TEXT_SHORT = {"track_b_nrmse": "Track B NRMSE", "track_b_nrmse_fixed": "Track B NRMSE (fixed R0 range)",
                     "track_a_nrmse": "Track A NRMSE"}


def column_figure(rows: list[dict], bands: dict, args: argparse.Namespace, fig_path: Path) -> list[Path]:
    """The band figure at one column width: median line, min-max band, seed counts in the legend.

    The y axis starts at zero, so that the size of the change with sigma is read against the error itself.
    """
    from scripts.figure_layout import COLUMN_IN, print_style, save_checked

    fset, fmetric = args.figure_set, args.figure_metric
    noises = sorted({row["noise"] for row in rows})
    with print_style():
        fig, ax = plt.subplots(figsize=(COLUMN_IN, 2.7), layout="constrained")
        for model in args.band_models:
            keys = ["%s|%.2f|%s" % (model, sigma, fset) for sigma in noises]
            xs = [sigma for sigma, key in zip(noises, keys) if key in bands]
            pts = [bands[key][fmetric] for key in keys if key in bands]
            if not pts:
                continue
            counts = [p["n"] for p in pts]
            seeds = ("%d seeds" % counts[0]) if len(set(counts)) == 1 else "seeds %s at \u03c3 = %s" % (
                ", ".join(str(n) for n in counts), ", ".join("%.2f" % x for x in xs))
            ax.plot(xs, [p["median"] for p in pts], marker="o", ms=3, color=COLORS.get(model),
                    label="%s, median (%s)" % (PRETTY.get(model, model), seeds))
            ax.fill_between(xs, [p["min"] for p in pts], [p["max"] for p in pts], color=COLORS.get(model),
                            alpha=0.2, lw=0, label="%s, min\u2013max over seeds" % PRETTY.get(model, model))
        for model in args.context_models:
            pts = sorted((row["noise"], row[fmetric]) for row in rows
                         if row["model"] == model and row["eval_set"] == fset and row["seed_source"] == 0
                         and not row.get("variant"))
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="s", ms=3, ls="--",
                        color=COLORS.get(model), alpha=0.7, label="%s (single seed)" % PRETTY.get(model, model))
        ax.set_xticks(noises)
        ax.set_ylim(0.0, ax.get_ylim()[1] * 1.15)
        ax.set_xlabel("measurement noise $\\sigma$")
        ax.set_ylabel("%s\n%s" % (METRIC_TEXT_SHORT[fmetric], SET_TEXT_SHORT[fset]))
        ax.set_title(("%s, %s" % (", ".join(PRETTY.get(m, m) for m in args.band_models), args.cell_label))
                     if args.cell_label else ", ".join(PRETTY.get(m, m) for m in args.band_models), loc="left")
        fig.legend(loc="outside lower center", ncol=1)
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            return save_checked(fig, fig_path, save_figure)
        finally:
            plt.close(fig)


def band_label(row: dict) -> str:
    """Model name, with the variant in brackets when the run has one."""
    variant = row.get("variant") or ""
    return "%s[%s]" % (row["model"], variant) if variant else row["model"]


def band_table(rows: list[dict], band_models: list[str]) -> dict[str, dict[str, dict[str, float]]]:
    bands: dict[str, dict[str, dict[str, float]]] = {}
    for row in rows:
        if row["model"] not in band_models:
            continue
        key = "%s|%.2f|%s" % (band_label(row), row["noise"], row["eval_set"])
        entry = bands.setdefault(key, {m: {"values": []} for m in METRICS})
        for metric in METRICS:
            entry[metric]["values"].append(float(row[metric]))
    for entry in bands.values():
        for metric in METRICS:
            values = entry[metric].pop("values")
            entry[metric].update(
                {
                    "n": len(values),
                    "median": float(np.median(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
            )
    return bands


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rows = []
    for seed, run_dir in enumerate(Path(r) for r in args.runs):
        if not run_dir.exists():
            continue
        for row in collect_runs(run_dir, Path(args.data_dir)):
            row["seed_source"] = seed
            rows.append(row)

    bands = band_table(rows, args.band_models)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / ("seed_bands%s.json" % args.tag)
    out_path.write_text(json.dumps(bands, indent=2), encoding="utf-8")

    fig_path = out_dir / "figures" / ("noise_robustness_bands%s.png" % args.tag)
    if not args.no_figures and args.figure_layout == "column":
        column_figure(rows, bands, args, fig_path)
    elif not args.no_figures:
        fset, fmetric = args.figure_set, args.figure_metric
        fig, ax = plt.subplots(figsize=(7, 4.5))
        noises = sorted({row["noise"] for row in rows})
        seed_counts: set[int] = set()
        for model in args.band_models:
            pts = [
                bands["%s|%.2f|%s" % (model, sigma, fset)][fmetric]
                for sigma in noises
                if "%s|%.2f|%s" % (model, sigma, fset) in bands
            ]
            xs = [sigma for sigma in noises if "%s|%.2f|%s" % (model, sigma, fset) in bands]
            if not pts:
                continue
            counts = [p["n"] for p in pts]
            seed_counts.update(counts)
            if len(set(counts)) == 1:
                label = "%s (median of %d seeds)" % (model, counts[0])
            else:
                label = "%s (median; seeds: %s)" % (
                    model, ", ".join("%d at sigma %.2f" % (n, s) for s, n in zip(xs, counts)))
            ax.plot(xs, [p["median"] for p in pts], marker="o", color=COLORS.get(model), label=label)
            ax.fill_between(xs, [p["min"] for p in pts], [p["max"] for p in pts],
                            color=COLORS.get(model), alpha=0.2)
        for model in args.context_models:
            pts = sorted(
                (row["noise"], row[fmetric])
                for row in rows
                if row["model"] == model and row["eval_set"] == fset and row["seed_source"] == 0
                and not row.get("variant")
            )
            if pts:
                ax.plot(
                    [p[0] for p in pts], [p[1] for p in pts],
                    marker="s", linestyle="--", color=COLORS.get(model), alpha=0.7,
                    label="%s (single seed)" % model,
                )
        ax.set_xlabel("measurement noise sigma")
        ax.set_ylabel("%s (%s)" % (METRIC_TEXT[fmetric], SET_TEXT[fset]))
        if len(seed_counts) <= 1:
            ax.set_title("Robustness on never-measured components, min-max over %d seeds" % len(args.runs))
        else:
            ax.set_title("Robustness on never-measured components, min-max over the seeds at each sigma")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        save_figure(fig, fig_path)
        plt.close(fig)

    print("model      sigma  set      trackB med [min-max]   n")
    for key in sorted(bands):
        model, sigma, eval_set = key.split("|")
        stats = bands[key]["track_b_nrmse"]
        print(
            "%-10s %-6s %-8s %.4f [%.4f-%.4f]  %d"
            % (model, sigma, eval_set, stats["median"], stats["min"], stats["max"], stats["n"])
        )
    print("wrote %s%s" % (out_path, "" if args.no_figures else " and %s" % fig_path))


if __name__ == "__main__":
    main()

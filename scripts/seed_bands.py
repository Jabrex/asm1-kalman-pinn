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

from src.eval.report import collect_runs, save_figure  # noqa: E402

RAW = Path("results/raw")
SEED_DIRS = {
    0: Path("results/runs"),
    1: Path("results/runs_seed1"),
    2: Path("results/runs_seed2"),
}
BAND_MODELS = ("cl_pinn", "pinn")
CONTEXT_MODELS = ("cl_lstm", "lstm")
METRICS = ("track_a_nrmse", "track_a_r2", "track_b_nrmse", "track_b_r2", "track_b_nrmse_fixed")
#: Figure text for the evaluation sets and metrics the band figure can draw.
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
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args(argv)


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
    if not args.no_figures:
        # --- banded noise-robustness figure (v1.0: holdout, Track B) -------
        fset, fmetric = args.figure_set, args.figure_metric
        fig, ax = plt.subplots(figsize=(7, 4.5))
        noises = sorted({row["noise"] for row in rows})
        for model in args.band_models:
            pts = [
                bands["%s|%.2f|%s" % (model, sigma, fset)][fmetric]
                for sigma in noises
                if "%s|%.2f|%s" % (model, sigma, fset) in bands
            ]
            xs = [sigma for sigma in noises if "%s|%.2f|%s" % (model, sigma, fset) in bands]
            if not pts:
                continue
            n_seeds = max(p["n"] for p in pts)
            ax.plot(xs, [p["median"] for p in pts], marker="o", color=COLORS.get(model),
                    label="%s (median of %d seeds)" % (model, n_seeds))
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
        ax.set_title("Robustness on never-measured components, min-max over %d seeds" % len(args.runs))
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        save_figure(fig, fig_path)  # 600 dpi PNG + vector PDF twin
        plt.close(fig)

    # Compact console table for the paper edit.
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

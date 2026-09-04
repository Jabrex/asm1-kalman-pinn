"""Aggregate multi-seed PINN runs into min-max bands.

Scans the seed run directories, groups rows by (model, noise, eval_set), and
writes ``results/seed_bands.json`` plus a banded version of the noise-robustness
figure. LSTM rows stay single-seed by design (reviewer-sanctioned); they appear
as plain lines for context.

Usage: python -m scripts.seed_bands
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

from src.eval.report import collect_runs, save_figure  # noqa: E402

RAW = Path("results/raw")
SEED_DIRS = {
    0: Path("results/runs"),
    1: Path("results/runs_seed1"),
    2: Path("results/runs_seed2"),
}
BAND_MODELS = ("cl_pinn", "pinn")
METRICS = ("track_a_nrmse", "track_a_r2", "track_b_nrmse", "track_b_r2", "track_b_nrmse_fixed")


def main() -> None:
    rows = []
    for seed, run_dir in SEED_DIRS.items():
        if not run_dir.exists():
            continue
        for row in collect_runs(run_dir, RAW):
            row["seed_source"] = seed
            rows.append(row)

    bands: dict[str, dict[str, dict[str, float]]] = {}
    for row in rows:
        if row["model"] not in BAND_MODELS:
            continue
        key = "%s|%.2f|%s" % (row["model"], row["noise"], row["eval_set"])
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

    out_path = Path("results/seed_bands.json")
    out_path.write_text(json.dumps(bands, indent=2), encoding="utf-8")

    # --- banded noise-robustness figure (holdout, Track B) ------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"cl_pinn": "tab:orange", "pinn": "tab:red"}
    noises = sorted({row["noise"] for row in rows})
    for model in BAND_MODELS:
        med, lo, hi = [], [], []
        for sigma in noises:
            stats = bands["%s|%.2f|%s" % (model, sigma, "holdout")]["track_b_nrmse"]
            med.append(stats["median"])
            lo.append(stats["min"])
            hi.append(stats["max"])
        ax.plot(noises, med, marker="o", color=colors[model], label="%s (median of 3 seeds)" % model)
        ax.fill_between(noises, lo, hi, color=colors[model], alpha=0.2)
    for model, color in (("cl_lstm", "tab:blue"), ("lstm", "tab:green")):
        pts = sorted(
            (row["noise"], row["track_b_nrmse"])
            for row in rows
            if row["model"] == model and row["eval_set"] == "holdout" and row["seed_source"] == 0
        )
        if pts:
            ax.plot(
                [p[0] for p in pts], [p[1] for p in pts],
                marker="s", linestyle="--", color=color, alpha=0.7,
                label="%s (single seed)" % model,
            )
    ax.set_xlabel("measurement noise sigma")
    ax.set_ylabel("Track B NRMSE (holdout)")
    ax.set_title("Robustness on never-measured components, min-max over 3 seeds")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig_path = Path("results/figures/noise_robustness_bands.png")
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
    print("wrote %s and %s" % (out_path, fig_path))


if __name__ == "__main__":
    main()

"""Architecture-split training-loss figures for the paper.

Reads history.json from the sixteen benchmark run directories and writes
``loss_curves_pinn.png`` and ``loss_curves_lstm.png`` (blue solid = curriculum,
orange dashed = single-stage, darker = higher noise). Promoted into scripts/
so the paper figures regenerate with the pipeline.

Usage: python -m scripts.split_loss_curves
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.report import save_figure  # noqa: E402

RUNS = Path("results/runs")
FIGURES = Path("results/figures")
SIGMAS = ("0p00", "0p05", "0p10", "0p15")
SHADES = (0.35, 0.55, 0.75, 0.95)


def plot_family(cl_model: str, plain_model: str, loss_desc: str, out_name: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    blues = plt.get_cmap("Blues")
    oranges = plt.get_cmap("Oranges")
    boundaries: list[int] = []
    for model, cmap, style in ((cl_model, blues, "-"), (plain_model, oranges, "--")):
        for tag, shade in zip(SIGMAS, SHADES):
            path = RUNS / ("%s_sigma%s" % (model, tag)) / "history.json"
            if not path.exists():
                continue
            history = json.loads(path.read_text(encoding="utf-8"))
            steps = [h["step"] for h in history]
            ax.semilogy(
                steps,
                [max(h["total"], 1e-12) for h in history],
                style,
                color=cmap(shade),
                lw=1.2,
                label="%s  σ=%s" % (model, tag.replace("p", ".")),
            )
            if model == cl_model and not boundaries:
                boundaries = [
                    h["step"]
                    for i, h in enumerate(history)
                    if i and h["stage"] != history[i - 1]["stage"]
                ]
    for s in boundaries:
        ax.axvline(s, color="grey", ls=":", lw=0.7)
    ax.set_xlabel("step")
    ax.set_ylabel("total loss (log)")
    ax.set_title(
        "Training loss — %s runs\n"
        "dotted vertical lines: curriculum stage boundaries · loss = %s"
        % (plain_model.upper(), loss_desc)
    )
    ax.legend(fontsize=7, ncol=2, title="blue = curriculum (solid)   orange = no curriculum (dashed)", title_fontsize=7)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    for written in save_figure(fig, FIGURES / out_name):
        print("wrote", written)
    plt.close(fig)


def main() -> None:
    plot_family("cl_pinn", "pinn", "data + physics + IC + positivity + balance", "loss_curves_pinn.png")
    plot_family("cl_lstm", "lstm", "data term only (no physics)", "loss_curves_lstm.png")


if __name__ == "__main__":
    main()

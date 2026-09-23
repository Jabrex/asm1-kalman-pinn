"""Architecture-split training-loss figures for the paper.

Reads history.json from the sixteen benchmark run directories and writes

* ``loss_curves_stacked.png`` / ``.pdf`` -- the manuscript figure (Fig. 5):
  PINN runs on top, LSTM runs below, full text width, shared step axis and a
  single legend, so the eight curves per panel stay distinguishable in print;
* ``loss_curves_pinn.*`` and ``loss_curves_lstm.*`` -- the same two panels as
  stand-alone figures, kept for the journal upload and the runbook.

Colour code, shared by all three files: blue solid = curriculum, orange dashed
= single-stage, darker = higher noise. Promoted into scripts/ so the paper
figures regenerate with the pipeline.

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
# Shades of the Blues / Oranges colormaps, light -> dark = clean -> noisy.
SHADES = (0.40, 0.60, 0.78, 0.95)
LINE_WIDTH = 1.6

# (curriculum run prefix, single-stage run prefix, loss description)
FAMILIES = (
    ("cl_pinn", "pinn", "loss = data + physics + IC + positivity + balance"),
    ("cl_lstm", "lstm", "loss = data term only (no physics)"),
)
LEGEND_TITLE = "solid blue = curriculum   dashed orange = single-stage   darker = higher σ"


def load_history(model: str, tag: str) -> list[dict] | None:
    path = RUNS / ("%s_sigma%s" % (model, tag)) / "history.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def stage_boundaries(history: list[dict]) -> list[int]:
    return [
        h["step"]
        for i, h in enumerate(history)
        if i and h["stage"] != history[i - 1]["stage"]
    ]


def draw_family(ax, cl_model: str, plain_model: str, title: str) -> None:
    """Draw the eight loss curves of one architecture family onto ``ax``."""
    blues = plt.get_cmap("Blues")
    oranges = plt.get_cmap("Oranges")
    boundaries: list[int] = []
    for model, cmap, style, role in (
        (cl_model, blues, "-", "curriculum"),
        (plain_model, oranges, "--", "single-stage"),
    ):
        for tag, shade in zip(SIGMAS, SHADES):
            history = load_history(model, tag)
            if history is None:
                continue
            ax.semilogy(
                [h["step"] for h in history],
                [max(h["total"], 1e-12) for h in history],
                style,
                color=cmap(shade),
                lw=LINE_WIDTH,
                label="%s  σ=%s" % (role, tag.replace("p", ".")),
            )
            if model == cl_model and not boundaries:
                boundaries = stage_boundaries(history)
    for s in boundaries:
        ax.axvline(s, color="grey", ls=":", lw=0.8, zorder=1)
    ax.grid(axis="y", which="major", color="0.92", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_ylabel("total loss (log)")
    ax.set_title(title, loc="left", fontsize=10)


def plot_stacked(out_name: str) -> None:
    """Two full-width panels, one above the other, sharing the step axis."""
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.2), sharex=True)
    for ax, panel, (cl_model, plain_model, loss_desc) in zip(axes, ("a", "b"), FAMILIES):
        draw_family(
            ax,
            cl_model,
            plain_model,
            "(%s) %s runs — %s" % (panel, plain_model.upper(), loss_desc),
        )
    axes[0].legend(
        loc="upper right",
        ncol=2,
        fontsize=8.5,
        title=LEGEND_TITLE,
        title_fontsize=8.5,
        framealpha=0.95,
    )
    axes[-1].set_xlabel("step")
    fig.tight_layout(h_pad=1.6)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for written in save_figure(fig, FIGURES / out_name):
        print("wrote", written)
    plt.close(fig)


def plot_family(cl_model: str, plain_model: str, loss_desc: str, out_name: str) -> None:
    """One architecture family as a stand-alone figure."""
    fig, ax = plt.subplots(figsize=(9, 5))
    draw_family(
        ax,
        cl_model,
        plain_model,
        "Training loss — %s runs\ndotted vertical lines: curriculum stage boundaries · %s"
        % (plain_model.upper(), loss_desc),
    )
    ax.set_xlabel("step")
    ax.legend(loc="upper right", ncol=2, fontsize=8, title=LEGEND_TITLE, title_fontsize=8)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    for written in save_figure(fig, FIGURES / out_name):
        print("wrote", written)
    plt.close(fig)


def main() -> None:
    plot_stacked("loss_curves_stacked.png")
    for cl_model, plain_model, loss_desc in FAMILIES:
        plot_family(cl_model, plain_model, loss_desc, "loss_curves_%s.png" % plain_model)


if __name__ == "__main__":
    main()

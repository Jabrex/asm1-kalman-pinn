"""Aggregate every finished run into the benchmark tables and figures.

Outputs
-------
``results/benchmark.csv``   one row per (model, noise level, evaluation set)
``results/benchmark.md``    Track A / Track B tables plus the dataset descriptors
``results/figures/``        loss curves, noise-robustness curve, unmeasured-state
                            trajectories, curriculum stage boundaries

Track B rows for the physics-free baselines are reported as measured, not
omitted. If an LSTM shows a near-zero R2 on the never-measured components, that
is the result; the table carries a footnote explaining why rather than a dash
with no reason.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..asm1.plant import Bsm1Plant
from ..data.sensors import ObservationDataset, observed_components, unobserved_components
from .metrics import (
    continuity_of_prediction,
    effluent_quality_index,
    state_metrics,
    track_summary,
)

EVAL_SETS = ("train", "holdout", "rain")


def _truth_for(set_name: str, cfg: dict[str, Any], data_dir: Path) -> ObservationDataset | None:
    sigma_tag = ("%.2f" % float(cfg["noise"])).replace(".", "p")
    scenario = "rain" if set_name == "rain" else "dry"
    path = data_dir / ("obs_%s_sigma%s.npz" % (scenario, sigma_tag))
    if not path.exists():
        return None
    dataset = ObservationDataset.load(path)
    if set_name == "train":
        return dataset.window(0.0, float(cfg.get("train_end_day", 12.0)))
    if set_name == "holdout":
        lo, hi = cfg.get("holdout_days", (12.0, 14.0))
        return dataset.window(float(lo), float(hi))
    return dataset


def collect_runs(runs_dir: Path, data_dir: Path) -> list[dict[str, Any]]:
    """Score every run directory that has both a summary and predictions.

    Directories starting with ``_`` are verification probes (written by
    scripts/verify_model), not benchmark runs, and are skipped.
    """
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(
        p for p in runs_dir.iterdir() if p.is_dir() and not p.name.startswith("_")
    ):
        summary_path = run_dir / "summary.json"
        pred_path = run_dir / "predictions.npz"
        if not (summary_path.exists() and pred_path.exists()):
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        # Windows come from the run itself, so changing them in base.yaml does
        # not silently mis-slice the evaluation sets here.
        cfg = {
            "noise": summary["noise"],
            "train_end_day": summary.get("train_end_day", 12.0),
            "holdout_days": tuple(summary.get("holdout_days", (12.0, 14.0))),
        }
        # Fixed reference range from the training window, so NRMSE stays
        # comparable across evaluation windows (the rain event widens the
        # per-window range and would otherwise flatter rain rows).
        train_truth = _truth_for("train", cfg, data_dir)
        fixed_spread = None
        if train_truth is not None:
            flat = train_truth.truth_reactor.reshape(-1, train_truth.truth_reactor.shape[-1])
            fixed_spread = flat.max(axis=0) - flat.min(axis=0)
        with np.load(pred_path) as preds:
            for set_name in EVAL_SETS:
                if set_name not in preds.files:
                    continue
                truth = _truth_for(set_name, cfg, data_dir)
                if truth is None:
                    continue
                pred = preds[set_name]
                n = min(len(pred), len(truth.truth_reactor))
                metrics = state_metrics(truth.truth_reactor[:n], pred[:n])
                tracks = track_summary(metrics)
                fixed = state_metrics(truth.truth_reactor[:n], pred[:n], spread=fixed_spread)
                tracks_fixed = track_summary(fixed)
                rows.append(
                    {
                        "run_id": summary["run_id"],
                        "model": summary["model"],
                        "arch": summary["arch"],
                        "curriculum": summary["curriculum"],
                        "noise": summary["noise"],
                        "profile": summary["profile"],
                        "eval_set": set_name,
                        "steps": summary["steps"],
                        "train_seconds": summary["train_seconds"],
                        "n_parameters": summary["n_parameters"],
                        "track_a_nrmse": tracks["track_a_measured"]["nrmse"],
                        "track_a_r2": tracks["track_a_measured"]["r2"],
                        "track_a_mae": tracks["track_a_measured"]["mae"],
                        "track_b_nrmse": tracks["track_b_unmeasured"]["nrmse"],
                        "track_b_r2": tracks["track_b_unmeasured"]["r2"],
                        "track_b_mae": tracks["track_b_unmeasured"]["mae"],
                        "track_a_nrmse_fixed": tracks_fixed["track_a_measured"]["nrmse"],
                        "track_b_nrmse_fixed": tracks_fixed["track_b_unmeasured"]["nrmse"],
                        "final_physics_loss": summary.get("final_losses", {}).get("physics"),
                        "per_component": tracks["per_component"],
                    }
                )
    return rows


def dataset_descriptors(raw_dir: Path) -> dict[str, Any]:
    """Ground-truth dataset properties: effluent quality, limits, influent stats."""
    plant = Bsm1Plant()
    out: dict[str, Any] = {}
    for scenario in ("dry", "rain"):
        path = raw_dir / ("sim_%s.npz" % scenario)
        if not path.exists():
            continue
        from ..data.simulate import SimulationResult

        result = SimulationResult.load(path)
        q_e = result.q_in - plant.cfg.q_w
        out[scenario] = {
            "effluent": effluent_quality_index(plant, result.t, result.effluent, q_e),
            "influent": result.meta.get("influent_summary", {}),
            "continuity_of_truth": continuity_of_prediction(plant, result.reactor),
        }
    return out


def write_csv(rows: Iterable[dict[str, Any]], path: Path) -> Path:
    rows = list(rows)
    if not rows:
        raise RuntimeError("No completed runs found - nothing to report")
    fields = [k for k in rows[0] if k != "per_component"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fields})
    return path


def _pivot(rows: list[dict[str, Any]], eval_set: str, metric: str) -> str:
    models = sorted({r["model"] for r in rows})
    noises = sorted({r["noise"] for r in rows})
    header = "| model | " + " | ".join("sigma=%.2f" % n for n in noises) + " |"
    sep = "| --- |" + " --- |" * len(noises)
    lines = [header, sep]
    for model in models:
        cells = []
        for noise in noises:
            match = [
                r for r in rows
                if r["model"] == model and r["noise"] == noise and r["eval_set"] == eval_set
            ]
            cells.append("%.4f" % match[0][metric] if match else "-")
        lines.append("| %s | %s |" % (model, " | ".join(cells)))
    return "\n".join(lines)


def write_markdown(
    rows: list[dict[str, Any]], descriptors: dict[str, Any], path: Path
) -> Path:
    parts: list[str] = ["# ASM1 CL+PINN benchmark", ""]
    parts.append(
        "All numbers below come from a single vault parameter set (20 degrees C, "
        "`data/asm1.json`). BSM1 supplies the plant geometry, flows and influent "
        "composition only."
    )
    parts.append("")
    parts.append("Track A = measured components %s." % (", ".join(observed_components()),))
    parts.append(
        "Track B = never-measured components %s." % (", ".join(unobserved_components()),)
    )
    parts.append("")
    for eval_set in EVAL_SETS:
        if not any(r["eval_set"] == eval_set for r in rows):
            continue
        parts.append("## %s" % eval_set)
        for label, metric in (
            ("Track A - NRMSE (lower is better)", "track_a_nrmse"),
            ("Track A - R2", "track_a_r2"),
            ("Track B - NRMSE (lower is better)", "track_b_nrmse"),
            ("Track B - R2", "track_b_r2"),
        ):
            parts.append("")
            parts.append("### %s" % label)
            parts.append("")
            parts.append(_pivot(rows, eval_set, metric))
        parts.append("")

    parts.append("## Note on the Track B baselines")
    parts.append("")
    parts.append(
        "`lstm` and `cl_lstm` receive no sensor-derived training signal on the "
        "never-measured components: those states appear in no sensor channel and "
        "the baselines carry no physics term. They do receive the shared t=0 "
        "initial-condition anchor and the output-head scale derived from Z(0). "
        "Their Track B numbers therefore reflect that anchor plus initialisation, "
        "not a fitting failure. This is the comparison the benchmark was built to "
        "make, so the rows are reported rather than omitted."
    )
    parts.append("")
    models_present = {r["model"] for r in rows}
    if {"persistence", "ode_openloop"} & models_present:
        parts.append(
            "Non-learned reference rows: `persistence` holds the known t=0 state "
            "constant; `ode_openloop` integrates the plant model forward from that "
            "same state with the known influent - the perfect-model information "
            "bound for this in-model benchmark. Neither uses any sensor data."
        )
        parts.append("")

    if descriptors:
        parts.append("## Ground-truth dataset descriptors")
        parts.append("")
        parts.append("```json")
        parts.append(json.dumps(descriptors, indent=2))
        parts.append("```")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return path


FIGURE_DPI = 600
"""Raster export resolution. Environmental Modelling & Software requires >= 500 dpi
(1772 px single column) for line/halftone combination artwork; 600 dpi clears it."""


def save_figure(fig, png_path: Path) -> list[Path]:
    """Write a figure as a high-resolution PNG plus a vector PDF twin.

    The PNG keeps the manuscript compiling exactly as before; the PDF is the
    journal-upload copy (Elsevier prefers vector artwork with embedded fonts).
    Returns the paths written, PNG first.
    """
    png_path = Path(png_path)
    pdf_path = png_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=FIGURE_DPI)
    fig.savefig(pdf_path)
    return [png_path, pdf_path]


def make_figures(runs_dir: Path, raw_dir: Path, rows: list[dict[str, Any]], out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # 1. Loss curves, with curriculum stage boundaries marked.
    fig, ax = plt.subplots(figsize=(9, 5))
    for run_dir in sorted(
        p for p in runs_dir.iterdir() if p.is_dir() and not p.name.startswith("_")
    ):
        history_path = run_dir / "history.json"
        if not history_path.exists():
            continue
        history = json.loads(history_path.read_text(encoding="utf-8"))
        steps = [h["step"] for h in history]
        ax.semilogy(steps, [max(h["total"], 1e-12) for h in history], label=run_dir.name, lw=1)
        stages = [h["step"] for i, h in enumerate(history)
                  if i and h["stage"] != history[i - 1]["stage"]]
        for s in stages:
            ax.axvline(s, color="grey", ls=":", lw=0.6)
    ax.set_xlabel("step")
    ax.set_ylabel("total loss")
    ax.set_title("Training loss (dotted lines: curriculum stage boundaries)")
    ax.legend(fontsize=6, ncol=2)
    path = out_dir / "loss_curves.png"
    fig.tight_layout()
    written.extend(save_figure(fig, path))
    plt.close(fig)

    # 2. Noise robustness, Track B on the holdout set.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model in sorted({r["model"] for r in rows}):
        pts = sorted(
            (r["noise"], r["track_b_nrmse"])
            for r in rows if r["model"] == model and r["eval_set"] == "holdout"
        )
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=model)
    ax.set_xlabel("measurement noise sigma")
    ax.set_ylabel("Track B NRMSE (holdout)")
    ax.set_title("Robustness on never-measured components")
    ax.legend()
    path = out_dir / "noise_robustness.png"
    fig.tight_layout()
    written.extend(save_figure(fig, path))
    plt.close(fig)

    return written


def build(
    runs_dir: str | Path = "results/runs",
    raw_dir: str | Path = "results/raw",
    out_dir: str | Path = "results",
) -> dict[str, Any]:
    runs_dir, raw_dir, out_dir = Path(runs_dir), Path(raw_dir), Path(out_dir)
    rows = collect_runs(runs_dir, raw_dir)
    descriptors = dataset_descriptors(raw_dir)
    csv_path = write_csv(rows, out_dir / "benchmark.csv")
    md_path = write_markdown(rows, descriptors, out_dir / "benchmark.md")
    figures = make_figures(runs_dir, raw_dir, rows, out_dir / "figures")
    (out_dir / "benchmark_detail.json").write_text(
        json.dumps({"rows": rows, "descriptors": descriptors}, indent=2), encoding="utf-8"
    )
    return {
        "n_rows": len(rows),
        "csv": str(csv_path),
        "markdown": str(md_path),
        "figures": [str(f) for f in figures],
    }


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(build(), indent=2))

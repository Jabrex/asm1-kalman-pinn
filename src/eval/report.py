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
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from ..asm1.plant import Bsm1Plant
from ..asm1.vault_loader import vault
from ..data.sensors import ObservationDataset, observed_components, unobserved_components
from .metrics import (
    continuity_of_prediction,
    effluent_quality_index,
    level_error,
    state_metrics,
    track_summary,
    within_tolerance_fraction,
)

EVAL_SETS = ("train", "holdout", "rain")
#: Components whose level error (RMSE / mean |truth|) enters every row.
LEVEL_COMPONENTS = ("X_B_H", "X_B_A", "X_I", "X_P")
#: Biomass inventories scored by the tank-mean +-10 % tolerance fraction.
TOLERANCE_COMPONENTS = ("X_B_H", "X_B_A")
TOLERANCE_REL = 0.10
#: Summary metadata copied into every row. v1.0 summaries lack most keys; the
#: defaults describe the v1.0 set-up (vault kinetics, truth anchor, exact influent).
ROW_METADATA: dict[str, Any] = {
    "seed": 0,
    "truth_preset": "vault20",
    "alpha": None,
    "anchor": None,
    "influent_mode": "exact",
    "variant": "",
    "learned_multipliers": None,
}
#: Row keys that are not scalars and stay out of benchmark.csv.
CSV_EXCLUDE = ("per_component", "learned_multipliers")
#: A window label maps to (base evaluation set, first day, last day).
ExtraWindows = Mapping[str, tuple[str, float, float]]


@lru_cache(maxsize=32)
def _load_dataset(path: str) -> ObservationDataset:
    """Datasets are read-only here; windows copy their slices, so sharing one load is safe."""
    return ObservationDataset.load(path)


def _truth_for(set_name: str, cfg: dict[str, Any], data_dir: Path) -> ObservationDataset | None:
    sigma_tag = ("%.2f" % float(cfg["noise"])).replace(".", "p")
    scenario = "rain" if set_name == "rain" else "dry"
    path = Path(data_dir) / ("obs_%s_sigma%s.npz" % (scenario, sigma_tag))
    if not path.exists():
        return None
    dataset = _load_dataset(str(path.resolve()))
    if set_name == "train":
        return dataset.window(0.0, float(cfg.get("train_end_day", 12.0)))
    if set_name == "holdout":
        lo, hi = cfg.get("holdout_days", (12.0, 14.0))
        return dataset.window(float(lo), float(hi))
    return dataset


def _run_cfg(summary: dict[str, Any]) -> dict[str, Any]:
    # Windows come from the run itself, so changing them in base.yaml does
    # not silently mis-slice the evaluation sets here.
    return {
        "noise": summary["noise"],
        "train_end_day": summary.get("train_end_day", 12.0),
        "holdout_days": tuple(summary.get("holdout_days", (12.0, 14.0))),
    }


def window_pairs(
    run_dir: Path, data_dir: Path, extra_windows: ExtraWindows | None = None
) -> tuple[dict[str, Any], list[tuple[str, np.ndarray, np.ndarray, np.ndarray | None]]]:
    """Aligned ``(label, truth, prediction, fixed_spread)`` for every scored window.

    The base sets come first, in ``EVAL_SETS`` order, then each extra window,
    sliced out of its base set by time (both ends inclusive). ``fixed_spread``
    is the training-window range (14,), shared by every window of the run.
    """
    run_dir, data_dir = Path(run_dir), Path(data_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    cfg = _run_cfg(summary)
    # Fixed reference range from the training window, so NRMSE stays
    # comparable across evaluation windows (the rain event widens the
    # per-window range and would otherwise flatter rain rows).
    train_truth = _truth_for("train", cfg, data_dir)
    fixed_spread = None
    if train_truth is not None:
        flat = train_truth.truth_reactor.reshape(-1, train_truth.truth_reactor.shape[-1])
        fixed_spread = flat.max(axis=0) - flat.min(axis=0)
    truths: dict[str, ObservationDataset | None] = {"train": train_truth}
    pairs: list[tuple[str, np.ndarray, np.ndarray, np.ndarray | None]] = []
    with np.load(run_dir / "predictions.npz") as preds:
        for set_name in EVAL_SETS:
            if set_name not in preds.files:
                continue
            if set_name not in truths:
                truths[set_name] = _truth_for(set_name, cfg, data_dir)
            truth = truths[set_name]
            if truth is None:
                continue
            pred = preds[set_name]
            n = min(len(pred), len(truth.truth_reactor))
            pairs.append((set_name, truth.truth_reactor[:n], pred[:n], fixed_spread))
        for label, (base, lo, hi) in (extra_windows or {}).items():
            if base not in preds.files:
                continue
            if base not in truths:
                truths[base] = _truth_for(base, cfg, data_dir)
            truth = truths[base]
            if truth is None:
                continue
            pred = preds[base]
            n = min(len(pred), len(truth.truth_reactor))
            t = truth.t[:n]
            mask = (t >= float(lo) - 1e-9) & (t <= float(hi) + 1e-9)
            pairs.append((label, truth.truth_reactor[:n][mask], pred[:n][mask], fixed_spread))
    return summary, pairs


def score_arrays(
    summary: dict[str, Any],
    set_name: str,
    truth: np.ndarray,
    pred: np.ndarray,
    fixed_spread: np.ndarray | None,
) -> dict[str, Any]:
    metrics = state_metrics(truth, pred)
    tracks = track_summary(metrics)
    fixed = state_metrics(truth, pred, spread=fixed_spread)
    tracks_fixed = track_summary(fixed)
    row: dict[str, Any] = {
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
    components = vault().components
    levels = level_error(truth, pred)
    for name in LEVEL_COMPONENTS:
        row["level_error_%s" % name] = float(levels[components.index(name)])
    tol = within_tolerance_fraction(
        truth, pred, components=TOLERANCE_COMPONENTS, rel_tol=TOLERANCE_REL
    )
    if not isinstance(tol, Mapping):
        tol = dict(zip(TOLERANCE_COMPONENTS, np.atleast_1d(np.asarray(tol, dtype=float))))
    for name in TOLERANCE_COMPONENTS:
        row["tol10_%s" % name] = float(tol[name])
    for key, default in ROW_METADATA.items():
        row[key] = summary.get(key, default)
    return row


def score_run(
    run_dir: Path, data_dir: Path, extra_windows: ExtraWindows | None = None
) -> list[dict[str, Any]]:
    """Rows for one run directory, one per scored window."""
    summary, pairs = window_pairs(run_dir, data_dir, extra_windows)
    return [score_arrays(summary, label, truth, pred, spread) for label, truth, pred, spread in pairs]


def collect_runs(
    runs_dir: Path, data_dir: Path, extra_windows: ExtraWindows | None = None
) -> list[dict[str, Any]]:
    """Score every run directory that has both a summary and predictions.

    Directories starting with ``_`` are verification probes (written by
    scripts/verify_model), not benchmark runs, and are skipped.
    ``extra_windows``, for example ``{"recon2": ("train", 2.0, 12.0)}``, adds
    rows for time slices of a base set; ``None`` scores exactly the v1.0 sets.
    """
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(
        p for p in Path(runs_dir).iterdir() if p.is_dir() and not p.name.startswith("_")
    ):
        if not ((run_dir / "summary.json").exists() and (run_dir / "predictions.npz").exists()):
            continue
        rows.extend(score_run(run_dir, data_dir, extra_windows))
    return rows


def truth_plant(meta: Mapping[str, Any]) -> Bsm1Plant:
    """The plant that generated a dataset, rebuilt from the dataset meta.

    Evaluation side only: a truth plant with overridden kinetics comes from
    ``src.asm1.truth_plants`` (plan G2), which training and observer code
    never import.
    """
    params = meta.get("parameters")
    if params is None or dict(params) == dict(vault().parameters):
        return Bsm1Plant()
    from ..asm1 import truth_plants

    preset = meta.get("truth_preset", "vault20")
    if preset in ("bsm1_15c", "graded"):
        source = truth_plants.truth_vault(preset, float(meta.get("alpha", 1.0)))
    else:
        source = truth_plants.override_vault(params)
    drift = max(
        abs(float(source.parameters[k]) - float(value))
        for k, value in params.items()
        if k in source.parameters
    )
    if drift > 1e-12:
        raise ValueError(
            "truth preset %r does not reproduce the dataset parameters (max diff %.3g)"
            % (preset, drift)
        )
    return Bsm1Plant(source=source)


def dataset_descriptors(raw_dir: Path) -> dict[str, Any]:
    """Ground-truth dataset properties: effluent quality, limits, influent stats.

    The plant is rebuilt from each file's own meta, so a truth plant with
    BSM1 15 C or graded kinetics is described with its own parameters.
    """
    from ..data.simulate import SimulationResult

    out: dict[str, Any] = {}
    for scenario in ("dry", "rain"):
        path = Path(raw_dir) / ("sim_%s.npz" % scenario)
        if not path.exists():
            continue
        result = SimulationResult.load(path)
        plant = truth_plant(result.meta)
        q_e = result.q_in - plant.cfg.q_w
        out[scenario] = {
            "effluent": effluent_quality_index(plant, result.t, result.effluent, q_e),
            "influent": result.meta.get("influent_summary", {}),
            "continuity_of_truth": continuity_of_prediction(plant, result.reactor),
            "truth_preset": result.meta.get("truth_preset", "vault20"),
            "alpha": result.meta.get("alpha"),
        }
    return out


def write_csv(rows: Iterable[dict[str, Any]], path: Path) -> Path:
    rows = list(rows)
    if not rows:
        raise RuntimeError("No completed runs found - nothing to report")
    fields = [k for k in rows[0] if k not in CSV_EXCLUDE]
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

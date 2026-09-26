"""Main-text Figures 2-5 of the v1.1 manuscript, drawn from saved JSON only.

    python -m scripts.v11_figures --root results/v11 --paper-dir paper/figures

Inputs (nothing is trained or re-scored here):
    results/v11/regime_table.json, regime_states.json          (scripts/regime_map.py score)
    results/v11/analysis/recoverability_<tag>.json              (plan G4, scripts/recoverability.py)
    results/v11/analysis/validation/recoverability_validation.json
                                                                (plan G4 script, run in plan G7)
The G4 files are read in the layout scripts/recoverability.py and
scripts/recoverability_validation.py write (normalise_recoverability,
normalise_validation); the flat layout of the plan contract is accepted too.
Fig. 2c shows the registered H6 cell with the index points of the validation
file and the per-state errors of regime_states.json; it refuses to draw when
the two give different rho, which means the error definitions differ.
Outputs: results/v11/figures/fig{2,3,4,5}_*.{png,pdf} (600 dpi PNG + vector PDF),
copied to --paper-dir when it exists, and results/v11/figures/figure_sources.json,
which lists every plotted number and the file it came from.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.regime_map import PRETTY, TRACK_B, base_model, cell_label  # noqa: E402
from src.eval.report import save_figure  # noqa: E402

CLASSES = ("sensor-recoverable", "partly recoverable", "forcing-slaved", "anchor-carried")
CLASS_COLOURS = {"sensor-recoverable": "#2c7bb6", "partly recoverable": "#abd9e9",
                 "forcing-slaved": "#fdae61", "anchor-carried": "#d7191c"}
REQUIRED_STATE_KEYS = ("tank", "component", "ig", "tau_days", "forcing_share", "class")
LADDER = ("persistence", "lstm_v10", "pinn", "cl_pinn", "ekf", "eks", "ieks",
          "ode_openloop_reduced", "ode_openloop_full")
LINE_ESTIMATORS = ("ode_openloop_reduced", "ekf", "eks", "eks_aug")
POINT_ESTIMATORS = ("cl_pinn", "cl_pinn_theta")
#: Parameter-recovery inset: the H3 pair (joint state and parameter estimation).
RECOVERY_ESTIMATORS = ("cl_pinn_theta", "eks_aug")
#: Laboratory tiers of the G4 lab tables: the start-up panel (Al1) and respirometry (Al2).
RESPIROMETRY_ASSAYS = ("respirometry",)
FACTORIAL_ESTIMATORS = ("persistence", "ode_openloop_reduced", "eks", "cl_pinn")
COLOURS = {"persistence": "0.45", "ode_openloop_reduced": "black", "ode_openloop_full": "0.6",
           "lstm_v10": "tab:green", "pinn": "tab:red", "cl_pinn": "tab:orange", "cl_pinn_theta": "tab:purple",
           "ekf": "tab:cyan", "eks": "tab:blue", "ieks": "navy", "eks_aug": "tab:olive"}
THRESHOLD_TAU_DAYS = 6.0
#: Fig. 2c draws indices below this value at the floor (display only; rho uses the actual values).
INDEX_FLOOR = 1e-4
WINDOW_DAYS = 12.0


def normalise_class(name: str) -> str:
    key = str(name).lower().replace("_", " ").replace("-", " ").strip()
    for canonical in CLASSES:
        if key == canonical.replace("-", " "):
            return canonical
    raise ValueError("unknown recoverability class %r" % name)


def normalise_recoverability(rec: dict[str, Any]) -> dict[str, Any]:
    """Add the plan-contract lists ``sensor_subsets`` and ``lab_assays`` to a G4 recoverability file.

    scripts/recoverability.py writes the subsets under ``sensor_value.subsets`` with a
    per-component mean information gain (mean over the five tanks) and the laboratory
    tables under ``lab_assays.tiers`` and ``lab_assays.add_one_to_As``. The contract
    keeps per-tank lists; the per-component mean is repeated for each tank, which keeps
    every sum over tanks exact (sum = 5 x mean).
    """
    if not isinstance(rec.get("sensor_subsets"), list) and isinstance(rec.get("sensor_value"), dict):
        rec["sensor_subsets"] = [
            {"channels": list(sub["channels"]), "ig_mean": float(sub["J"]),
             "per_state_ig": {c: [float(v)] * 5 for c, v in sub["ig_by_component"].items()}}
            for sub in rec["sensor_value"].get("subsets", [])]
    labs = rec.get("lab_assays")
    if isinstance(labs, dict):
        base = labs["tiers"]["As"]["ig_by_component"]
        assays = []
        for entry in labs.get("add_one_to_As", []):
            if "ig_by_component" not in entry:
                continue
            tier = "Al2" if entry["assay"] in RESPIROMETRY_ASSAYS else "Al1"
            assays.append({"tier": tier, "assay": entry["assay"],
                           "ig_gain": {c: float(entry["ig_by_component"][c]) - float(base[c]) for c in base}})
        if not assays:
            # G4 files written before the per-assay gains: tier gains (panel, then respirometry).
            tiers = labs["tiers"]
            assays = [{"tier": "Al1", "assay": "start-up panel",
                       "ig_gain": {c: float(tiers["Al1"]["ig_by_component"][c]) - float(base[c]) for c in base}},
                      {"tier": "Al2", "assay": "respirometry",
                       "ig_gain": {c: float(tiers["Al2"]["ig_by_component"][c])
                                   - float(tiers["Al1"]["ig_by_component"][c]) for c in base}}]
        rec["lab_assays_g4"] = labs
        rec["lab_assays"] = assays
    return rec


def load_recoverability(path: Path) -> dict[str, Any]:
    """Read a G4 recoverability file and check the keys the figures use."""
    rec = normalise_recoverability(json.loads(Path(path).read_text(encoding="utf-8")))
    for key in ("tag", "states"):
        if key not in rec:
            raise KeyError("%s: missing top-level key %r" % (path, key))
    for i, state in enumerate(rec["states"]):
        missing = [k for k in REQUIRED_STATE_KEYS if k not in state]
        if missing:
            raise KeyError("%s: states[%d] lacks %s" % (path, i, missing))
        state["class"] = normalise_class(state["class"])
    return rec


def state_grid(rec: dict[str, Any], key: str) -> np.ndarray:
    """(5, 11) array of one per-state quantity, Track B order."""
    grid = np.full((5, len(TRACK_B)), np.nan)
    for s in rec["states"]:
        if s["component"] in TRACK_B:
            grid[int(s["tank"]) - 1, TRACK_B.index(s["component"])] = float(s[key])
    return grid


def class_grid(rec: dict[str, Any]) -> list[list[str]]:
    grid = [["" for _ in TRACK_B] for _ in range(5)]
    for s in rec["states"]:
        if s["component"] in TRACK_B:
            grid[int(s["tank"]) - 1][TRACK_B.index(s["component"])] = s["class"]
    return grid


def normalise_validation(val: dict[str, Any]) -> dict[str, Any]:
    """Flat ``{index, window, primary_cell, h6, results[]}`` from either validation layout.

    scripts/recoverability_validation.py writes ``cells.<cell>.estimators.<estimator>``
    with ``primary_index`` (CRB/range) and the plotted ``points``; the plan contract
    lists ``results`` directly.
    """
    if "results" in val and "index" in val:
        return {"primary_cell": None, "h6": None, "window": val.get("window", "R0"), **val}
    if "cells" not in val:
        raise KeyError("validation file has neither 'results' nor 'cells'")
    primary = val.get("primary_cell")
    h6 = val.get("h6")
    results = []
    for cell, entry in val["cells"].items():
        for estimator, e in entry.get("estimators", {}).items():
            if e.get("status") != "ok":
                results.append({"cell": cell, "estimator": estimator, "status": e.get("status"), "rho": None,
                                "ci": None, "n_points": 0, "pass": None})
                continue
            stat = e["primary_index"]
            is_h6 = bool(h6) and h6.get("cell") == cell and h6.get("estimator") == estimator
            results.append({"cell": cell, "estimator": estimator, "status": "ok", "rho": stat["rho"],
                            "ci": stat["ci95"], "n_points": stat["n_points"],
                            "pass": h6.get("pass") if is_h6 else None,
                            "secondary_rho": e["secondary_index"]["rho"],
                            "secondary_ci": e["secondary_index"]["ci95"],
                            "points_index": e.get("points", {}).get("index")})
    return {"index": "crb_over_range", "window": "R0", "primary_cell": primary, "h6": h6, "results": results}


def validation_lookup(path: Path | None) -> tuple[str, dict[tuple[str, str], dict[str, Any]]]:
    if path is None or not Path(path).exists():
        return "crb_over_range", {}
    val = normalise_validation(json.loads(Path(path).read_text(encoding="utf-8")))
    return val["index"], {(r["cell"], r["estimator"]): r for r in val["results"] if r.get("rho") is not None}


def state_entry(states: dict[str, Any], cell: str, sigma: float, estimator: str, window: str) -> np.ndarray | None:
    for e in states["entries"]:
        if e["cell"] == cell and abs(e["sigma"] - sigma) < 1e-9 and e["estimator"] == estimator and e["window"] == window:
            return np.array([[np.nan if v is None else v for v in row] for row in e["median"]], dtype=float)
    return None


def rows_for(table: dict[str, Any], cell: str, sigma: float, window: str) -> dict[str, dict[str, Any]]:
    return {r["estimator"]: r for r in table["rows"]
            if r["cell"] == cell and abs(r["sigma"] - sigma) < 1e-9 and r["window"] == window}


def _label(estimator: str) -> str:
    base = base_model(estimator)
    return PRETTY.get(base, base) + estimator[len(base):]


# -- Fig. 2 -----------------------------------------------------------------------
def fig2_recoverability(rec: dict[str, Any], states: dict[str, Any], validation: Path | None, png: Path,
                        cell: str, sigma: float, estimators: tuple[str, ...] = ("eks", "cl_pinn")) -> dict[str, Any]:
    index_name, val = validation_lookup(validation)
    if any(index_name not in s for s in rec["states"]):
        raise KeyError("recoverability file lacks the validation index %r" % index_name)
    classes = class_grid(rec)
    ig = state_grid(rec, "ig")
    tau = state_grid(rec, "tau_days")
    index = state_grid(rec, index_name)
    fig = plt.figure(figsize=(7.2, 7.4), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15])
    ax_a = fig.add_subplot(gs[0, :])
    colour_idx = np.array([[CLASSES.index(c) if c else -1 for c in row] for row in classes], dtype=float)
    from matplotlib.colors import ListedColormap

    ax_a.imshow(np.ma.masked_less(colour_idx, 0), cmap=ListedColormap([CLASS_COLOURS[c] for c in CLASSES]),
                vmin=-0.5, vmax=len(CLASSES) - 0.5, aspect="auto")
    for k in range(5):
        for j in range(len(TRACK_B)):
            if np.isfinite(ig[k, j]):
                ax_a.text(j, k, "%.2f" % ig[k, j], ha="center", va="center", fontsize=7)
    ax_a.set_xticks(range(len(TRACK_B)))
    ax_a.set_xticklabels([c.replace("_", "") for c in TRACK_B], fontsize=8)
    ax_a.set_yticks(range(5))
    ax_a.set_yticklabels(["tank %d" % (k + 1) for k in range(5)], fontsize=8)
    ax_a.set_title("(a) recoverability class per tank and state (text: information gain)", fontsize=9, loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=CLASS_COLOURS[c]) for c in CLASSES]
    ax_a.legend(handles, CLASSES, fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)

    ax_b = fig.add_subplot(gs[1, 0])
    med = np.nanmedian(tau, axis=0)
    lo, hi = np.nanmin(tau, axis=0), np.nanmax(tau, axis=0)
    ax_b.bar(range(len(TRACK_B)), med, color="0.6", yerr=[med - lo, hi - med], capsize=2)
    ax_b.axhline(THRESHOLD_TAU_DAYS, color="tab:red", ls="--", lw=0.8, label="anchor-carried threshold (6 d)")
    ax_b.axhline(WINDOW_DAYS, color="black", ls=":", lw=0.8, label="window R0 length (12 d)")
    ax_b.set_yscale("log")
    ax_b.set_xticks(range(len(TRACK_B)))
    ax_b.set_xticklabels([c.replace("_", "") for c in TRACK_B], rotation=60, fontsize=7)
    ax_b.set_ylabel("start-up memory time (d)", fontsize=8)
    ax_b.set_title("(b) memory time, median and range over tanks", fontsize=9, loc="left")
    ax_b.legend(fontsize=6)

    ax_c = fig.add_subplot(gs[1, 1])
    stats: dict[str, Any] = {"index": index_name, "cell": cell, "sigma": sigma, "estimators": {}}
    for estimator in estimators:
        err = state_entry(states, cell, sigma, estimator, "R0")
        if err is None:
            continue
        entry = val.get((cell, estimator))
        if val and entry is None:
            raise ValueError("validation file has no %s entry for cell %s; Fig. 2c needs the registered index"
                             % (estimator, cell))
        # The registered index is the cell's own (its anchor prior and influent view); the
        # validation file stores those points. Panel (a)'s file is used only without them.
        idx = np.asarray(entry["points_index"], dtype=float) if entry and entry.get("points_index") is not None \
            else index
        mask = np.isfinite(err) & np.isfinite(idx)
        rho = float(spearmanr(idx[mask], err[mask]).statistic)
        if entry is not None and abs(float(entry["rho"]) - rho) > 1e-6:
            raise ValueError("Fig. 2c points give rho %.6f for %s but the validation file reports %.6f; the two "
                             "scripts use different error definitions" % (rho, estimator, float(entry["rho"])))
        text = "rho = %.2f" % rho if entry is None else "rho = %.2f [%.2f, %.2f]" % (rho, *entry["ci"])
        # Display only: states with an index below the floor (numerically zero) are drawn at the floor
        # as open markers; rho above is computed from the actual values.
        shown = np.maximum(idx[mask], INDEX_FLOOR)
        low = idx[mask] < INDEX_FLOOR
        ax_c.scatter(shown[~low], err[mask][~low], s=10, color=COLOURS.get(estimator), alpha=0.8,
                     label="%s: %s" % (_label(estimator), text))
        if low.any():
            ax_c.scatter(shown[low], err[mask][low], s=14, facecolors="none", edgecolors=COLOURS.get(estimator),
                         linewidths=0.8)
        stats["estimators"][estimator] = {"rho_points": rho, "n_points": int(mask.sum()),
                                          "n_below_display_floor": int(low.sum()),
                                          "validation": None if entry is None else
                                          {k: v for k, v in entry.items() if k != "points_index"}}
    ax_c.set_xscale("log")
    ax_c.set_yscale("log")
    if any(e.get("n_below_display_floor") for e in stats["estimators"].values()):
        ax_c.axvline(INDEX_FLOOR, color="0.6", lw=0.6, ls=":")
        ax_c.text(INDEX_FLOOR, 0.02, " open: index < %g" % INDEX_FLOOR, transform=ax_c.get_xaxis_transform(),
                  fontsize=6, color="0.4")
    stats["index_display_floor"] = INDEX_FLOOR
    ax_c.set_xlabel("%s (model-derived index)" % index_name.replace("_", " "), fontsize=8)
    ax_c.set_ylabel("per-state NRMSE, window R0", fontsize=8)
    ax_c.set_title("(c) index against achieved error, %s" % cell_label(cell), fontsize=9, loc="left")
    ax_c.legend(fontsize=6)
    save_figure(fig, png)
    plt.close(fig)
    stats["class_counts"] = {c: int(sum(row.count(c) for row in classes)) for c in CLASSES}
    return stats


# -- Fig. 3 -----------------------------------------------------------------------
def fig3_ladder(table: dict[str, Any], png: Path, cell: str, sigma: float) -> dict[str, Any]:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), sharey=True, layout="constrained")
    plotted: dict[str, Any] = {}
    for ax, window in zip(axes, ("R0", "F")):
        rows = rows_for(table, cell, sigma, window)
        ladder = [e for e in LADDER if e in rows]
        for y, estimator in enumerate(ladder):
            r = rows[estimator]
            ax.scatter(r["values"], [y] * len(r["values"]), s=12, color="0.6", zorder=2)
            ax.scatter([r["median"]], [y], s=40, marker="D", color=COLOURS.get(estimator, "black"), zorder=3)
            ax.annotate("%.3f" % r["median"], (r["median"], y), xytext=(4, 5), textcoords="offset points",
                        fontsize=6)
            plotted.setdefault(estimator, {})[window] = {"median": r["median"], "values": r["values"]}
        if "persistence" in rows:
            ax.axvline(rows["persistence"]["median"], color="0.45", ls="--", lw=0.8)
        ax.set_xscale("log")
        ax.margins(x=0.25)
        ax.set_yticks(range(len(ladder)))
        ax.set_yticklabels([_label(e) for e in ladder], fontsize=8)
        ax.set_xlabel("Track B NRMSE (fixed R0 range)", fontsize=8)
        ax.set_title("window %s" % window, fontsize=9)
    fig.suptitle("Correct kinetics (%s, sigma = %.2f): grey dots are seeds, diamonds medians; "
                 "* = more information" % (cell_label(cell), sigma), fontsize=8)
    save_figure(fig, png)
    plt.close(fig)
    return plotted


# -- Fig. 4 -----------------------------------------------------------------------
def fig4_mismatch(table: dict[str, Any], png: Path, sigma: float, influent: str = "ie", anchor: str = "a0",
                  recovery_cell: str = "k100_ie_a0") -> dict[str, Any]:
    fig = plt.figure(figsize=(7.2, 3.9), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.8])
    ax_r0 = fig.add_subplot(gs[0, 0])
    axes = [ax_r0, fig.add_subplot(gs[0, 1], sharey=ax_r0)]
    ax_rec = fig.add_subplot(gs[0, 2])
    grid = [r for r in table["rows"] if r["k"] is not None and not r["extra"] and r["rand"] is None
            and not r["offsteady"] and r["influent"] == influent and r["anchor"] == anchor
            and abs(r["sigma"] - sigma) < 1e-9]
    plotted: dict[str, Any] = {"series": {}, "crossovers": [], "recovery": []}
    for ax, window in zip(axes, ("R0", "F")):
        sub = [r for r in grid if r["window"] == window]
        for estimator in ("persistence",) + LINE_ESTIMATORS:
            pts = sorted((r["kinetic_alpha"], r["median"]) for r in sub if r["estimator"] == estimator)
            if len(pts) < 2:
                continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=3,
                    ls="--" if estimator == "persistence" else "-", color=COLOURS.get(estimator), label=_label(estimator))
            plotted["series"].setdefault(window, {})[estimator] = pts
        for estimator in POINT_ESTIMATORS:
            pts = sorted((r["kinetic_alpha"], r["median"], r["min"], r["max"], r["n"]) for r in sub
                         if r["estimator"] == estimator)
            if not pts:
                continue
            x = np.array([p[0] for p in pts]) + (0.012 if estimator == "cl_pinn_theta" else 0.0)
            med = np.array([p[1] for p in pts])
            ax.errorbar(x, med, yerr=[med - np.array([p[2] for p in pts]), np.array([p[3] for p in pts]) - med],
                        fmt="s", ms=5, capsize=3, color=COLOURS.get(estimator),
                        label=_label(estimator) + " (median, range over seeds)")
            for xi, p in zip(x, pts):
                if p[4] < 3:  # Section 9: one-seed and two-seed PINN rows are labelled
                    ax.annotate("n=%d" % p[4], (xi, p[1]), xytext=(4, -8), textcoords="offset points", fontsize=5,
                                color=COLOURS.get(estimator))
            plotted["series"].setdefault(window, {})[estimator] = [list(p) for p in pts]
        crossing = [c for c in table["crossovers"]
                    if (c["influent"], c["anchor"], c["window"], c["status"]) == (influent, anchor, window, "crosses")
                    and abs(c["sigma"] - sigma) < 1e-9 and c["estimator"] in ("ode_openloop_reduced", "eks")]
        for k, c in enumerate(sorted(crossing, key=lambda c: c["alpha_star"])):
            ax.axvline(c["alpha_star"], color=COLOURS.get(c["estimator"]), ls=":", lw=0.9)
            # staggered heights so labels of nearby crossovers do not overlap
            ax.annotate(" alpha* = %.2f" % c["alpha_star"], (c["alpha_star"], 0.97 - 0.08 * k),
                        xycoords=("data", "axes fraction"), fontsize=6, va="top", color=COLOURS.get(c["estimator"]))
            plotted["crossovers"].append(c)
        ax.set_title("window %s" % window, fontsize=9)
    for ax in axes:
        ax.set_xlabel("kinetic mismatch alpha\n(0 = vault 20 C, 1 = BSM1 15 C)", fontsize=7)
    axes[0].set_ylabel("Track B NRMSE (fixed R0 range)", fontsize=8)
    handles, labels = [], []
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=6, frameon=False)
    recovery = [p for p in table["parameter_recovery"] if p["cell"] == recovery_cell and abs(p["sigma"] - sigma) < 1e-9
                and p["estimator"] in RECOVERY_ESTIMATORS]
    inset = ax_rec
    inset.set_title("parameter recovery, %s" % cell_label(recovery_cell), fontsize=8)
    if recovery:
        lims = []
        for p in recovery:
            names = [n for n in p["names"] if p["true_log_ratio"][n] is not None]
            for est in p["estimates"]:
                xs = [p["true_log_ratio"][n] for n in names]
                ys = [est["log_multiplier"][n] for n in names]
                inset.scatter(xs, ys, s=10, color=COLOURS.get(base_model(p["estimator"]), "black"))
                lims += xs + ys
            if p is recovery[0]:
                for n in names:
                    inset.annotate(n, (p["true_log_ratio"][n], p["estimates"][0]["log_multiplier"][n]), fontsize=5)
            plotted["recovery"].append(p)
        if lims:
            lo, hi = min(lims) - 0.1, max(lims) + 0.1
            inset.plot([lo, hi], [lo, hi], color="0.5", lw=0.6)
        inset.set_xlabel("true ln(truth / vault)", fontsize=7, labelpad=1)
        inset.set_ylabel("estimated ln m", fontsize=7, labelpad=1)
        inset.tick_params(labelsize=6)
        for est in RECOVERY_ESTIMATORS:
            inset.scatter([], [], s=10, color=COLOURS.get(est), label=_label(est))
        inset.legend(fontsize=6, loc="upper left", frameon=False)
    save_figure(fig, png)
    plt.close(fig)
    return plotted


# -- Fig. 5 -----------------------------------------------------------------------
def _grouped_bars(ax, table: dict[str, Any], cells: list[str], sigma: float, labels: list[str]) -> dict[str, Any]:
    width = 0.8 / len(FACTORIAL_ESTIMATORS)
    out: dict[str, Any] = {}
    for j, estimator in enumerate(FACTORIAL_ESTIMATORS):
        labelled = False
        for i, cell in enumerate(cells):
            r = rows_for(table, cell, sigma, "R0").get(estimator)
            if r is None:
                continue
            x = i + (j - (len(FACTORIAL_ESTIMATORS) - 1) / 2) * width
            ax.bar(x, r["median"], width, color=COLOURS.get(estimator), label=None if labelled else _label(estimator),
                   yerr=[[r["median"] - r["min"]], [r["max"] - r["median"]]] if r["n"] > 1 else None, capsize=2)
            labelled = True
            out.setdefault(cell, {})[estimator] = r["median"]
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels(labels, fontsize=7)
    return out


def fig5_factorial(table: dict[str, Any], states: dict[str, Any], rec_realistic: dict[str, Any], png: Path,
                   realistic_k: str, sigma: float) -> dict[str, Any]:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.6), layout="constrained")
    k = realistic_k
    cells_a = ["k000_ie_a0", "k000_ic_a0", "k000_ib_a0", "k%s_ie_a0" % k, "k%s_ic_a0" % k, "k%s_ib_a0" % k]
    plotted = {"a": _grouped_bars(axes[0], table, cells_a, sigma, [cell_label(c).replace(" A0", "") for c in cells_a])}
    axes[0].set_title("(a) influent x kinetics, A0", fontsize=8, loc="left")
    axes[0].set_ylabel("Track B NRMSE, window R0", fontsize=8)
    cells_b = ["k%s_ic_%s" % (k, a) for a in ("a0", "as", "al1", "al2")]
    plotted["b"] = _grouped_bars(axes[1], table, cells_b, sigma, ["A0", "As", "Al1", "Al2"])
    axes[1].set_title("(b) start-state tiers, %s" % cell_label(cells_b[0]).rsplit(" ", 1)[0], fontsize=8, loc="left")
    classes = class_grid(rec_realistic)
    cell_c = "k%s_ic_as" % k
    width = 0.8 / len(FACTORIAL_ESTIMATORS)
    plotted["c"] = {}
    for j, estimator in enumerate(FACTORIAL_ESTIMATORS):
        err = state_entry(states, cell_c, sigma, estimator, "R0")
        if err is None:
            continue
        for i, cls in enumerate(CLASSES):
            mask = np.array([[c == cls for c in row] for row in classes])
            if not mask.any():
                continue
            value = float(np.nanmean(err[mask]))
            axes[2].bar(i + (j - (len(FACTORIAL_ESTIMATORS) - 1) / 2) * width, value, width,
                        color=COLOURS.get(estimator))
            plotted["c"].setdefault(cls, {})[estimator] = value
    axes[2].set_xticks(range(len(CLASSES)))
    axes[2].set_xticklabels(["sensor", "partly", "forcing", "anchor"], fontsize=7)
    axes[2].set_title("(c) by recoverability class, %s" % cell_label(cell_c), fontsize=8, loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels), fontsize=7, frameon=False)
    save_figure(fig, png)
    plt.close(fig)
    return plotted


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="results/v11")
    parser.add_argument("--sigma", type=float, default=0.10)
    parser.add_argument("--map-cell", default="k000_ie_a0", help="cell for Figs 2c and 3")
    parser.add_argument("--rec-map", default="results/v11/analysis/recoverability_k000.json")
    parser.add_argument("--rec-realistic", default=None,
                        help="default: <root>/analysis/recoverability_k<meta.realistic_k>.json")
    parser.add_argument("--validation", default="results/v11/analysis/validation/recoverability_validation.json")
    parser.add_argument("--h6-cell", default=None,
                        help="cell of Fig. 2c; default: the primary cell of the validation file, else --map-cell")
    parser.add_argument("--realistic-k", default=None,
                        help="default: regime_table meta.realistic_k ('050' after gate D3); must agree with it")
    parser.add_argument("--paper-dir", default="paper/figures")
    args = parser.parse_args(argv)
    root = Path(args.root)
    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table = json.loads((root / "regime_table.json").read_text(encoding="utf-8"))
    states = json.loads((root / "regime_states.json").read_text(encoding="utf-8"))
    meta_k = table.get("meta", {}).get("realistic_k")
    realistic_k = args.realistic_k or meta_k or "100"
    if meta_k is not None and realistic_k != meta_k:
        raise ValueError("--realistic-k %s disagrees with the regime table (realistic cells k%s, gate D3)"
                         % (realistic_k, meta_k))
    rec_real_path = args.rec_realistic or str(root / "analysis" / ("recoverability_k%s.json" % realistic_k))
    rec_map = load_recoverability(Path(args.rec_map))
    rec_real = load_recoverability(Path(rec_real_path))
    if rec_real.get("tag") not in (None, "k%s" % realistic_k):
        raise ValueError("%s is tagged %r, not k%s" % (rec_real_path, rec_real.get("tag"), realistic_k))
    if not Path(args.validation).exists():
        raise FileNotFoundError("validation file %s not found; run scripts.recoverability_validation" % args.validation)
    validation = Path(args.validation)
    sources: dict[str, Any] = {"inputs": {"regime_table": str(root / "regime_table.json").replace("\\", "/"),
                                          "regime_states": str(root / "regime_states.json").replace("\\", "/"),
                                          "rec_map": args.rec_map, "rec_realistic": rec_real_path.replace("\\", "/"),
                                          "validation": args.validation, "realistic_k": realistic_k}}
    h6_cell = args.h6_cell
    if h6_cell is None:
        h6_cell = normalise_validation(json.loads(validation.read_text(encoding="utf-8"))).get("primary_cell")
    sources["fig2"] = fig2_recoverability(rec_map, states, validation, fig_dir / "fig2_recoverability.png",
                                          h6_cell or args.map_cell, args.sigma)
    sources["fig3"] = fig3_ladder(table, fig_dir / "fig3_ladder.png", args.map_cell, args.sigma)
    sources["fig4"] = fig4_mismatch(table, fig_dir / "fig4_mismatch.png", args.sigma, recovery_cell="k100_ie_a0")
    sources["fig5"] = fig5_factorial(table, states, rec_real, fig_dir / "fig5_factorial.png", realistic_k,
                                     args.sigma)
    (fig_dir / "figure_sources.json").write_text(json.dumps(sources, indent=1, default=float), encoding="utf-8")
    written = sorted(fig_dir.glob("fig[2-5]_*.p*"))
    if args.paper_dir and Path(args.paper_dir).exists():
        for path in written:
            shutil.copy2(path, Path(args.paper_dir) / path.name)
    for path in written:
        print("wrote", path)
    print("wrote", fig_dir / "figure_sources.json")


if __name__ == "__main__":
    main()

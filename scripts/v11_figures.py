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
Outputs: one file per panel, each drawn at its printed width (one column, 85 mm, or the
full page, 178 mm) with no text below 7 pt, so that it stays legible in the journal PDF:
    fig2a_recoverability_classes, fig2b_memory_time, fig2c_index_vs_error,
    fig3a_ladder_R0, fig3b_ladder_F,
    fig4a_mismatch_R0, fig4b_mismatch_F, fig4c_parameter_recovery, fig4_legend,
    fig5a_influent_kinetics, fig5b_start_state_tiers, fig5c_recoverability_class, fig5_legend
under results/v11/figures (600 dpi PNG + vector PDF), copied to --paper-dir when it exists.
scripts/figure_layout.py refuses a panel with text that is too small, cut off or colliding.
results/v11/figures/figure_sources.json lists every plotted number and file.
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
from matplotlib.ticker import NullFormatter  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.regime_map import PRETTY, TRACK_B, base_model, cell_label  # noqa: E402
from scripts.figure_layout import COLUMN_IN, DOUBLE_IN, print_style, save_checked  # noqa: E402
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


# -- print layout -----------------------------------------------------------------
#: Component names as set in the figures.
COMPONENT_TEXT = {"S_I": "$S_I$", "S_S": "$S_S$", "X_I": "$X_I$", "X_S": "$X_S$", "X_B_H": "$X_{B,H}$",
                  "X_B_A": "$X_{B,A}$", "X_P": "$X_P$", "S_ND": "$S_{ND}$", "X_ND": "$X_{ND}$",
                  "S_ALK": "$S_{ALK}$", "S_N2": "$S_{N2}$"}
#: Kinetic names of the parameter-recovery panel, with one marker shape each.
PARAMETER_TEXT = {"bH": "$b_H$", "muA": "$\\mu_A$", "kh": "$k_h$", "etag": "$\\eta_g$"}
PARAMETER_MARKERS = {"bH": "o", "muA": "s", "kh": "^", "etag": "D"}
CLASS_SHORT = {"sensor-recoverable": "sensor", "partly recoverable": "partly", "forcing-slaved": "forcing",
               "anchor-carried": "anchor"}
#: Classes on a dark fill carry white numbers.
DARK_CLASSES = ("sensor-recoverable", "anchor-carried")
WINDOW_TEXT = {"R0": "days 0\u201312", "R2": "days 2\u201312", "F": "days 12\u201314"}
MINUS = "\u2212"
#: One label for the primary metric in every figure.
METRIC_LABEL = "Never-measured NRMSE (fixed R0 range)"
#: Pre-registered class rules (PREREGISTRATION.md Section 7), shown in the Fig. 2a legend.
CLASS_RULE = {"sensor-recoverable": "IG \u2265 0.5", "partly recoverable": "all other states",
              "forcing-slaved": "forcing share > 0.5, \u03c4 < 1 d", "anchor-carried": "\u03c4 > 6 d, IG < 0.5"}


def _num(value: float, pattern: str = "%.2f") -> str:
    """A number as set in the figures (typographic minus)."""
    return (pattern % value).replace("-", MINUS)


def _component(name: str) -> str:
    return COMPONENT_TEXT.get(name, name.replace("_", ""))


def _value(v: float, n: int = 1) -> str:
    """Ladder value: three decimals, two significant figures below 0.01; '(n)' seeds when more than one."""
    text = _num(v, "%.2g" if abs(v) < 0.01 else "%.3f")
    return text + (" (%d)" % n if n > 1 else "")


def _save(fig, png: Path) -> list[Path]:
    """Refuse a figure whose text is too small, cut off or colliding; else write PNG + PDF."""
    try:
        return save_checked(fig, png, save_figure)
    finally:
        plt.close(fig)


def rec_label(rec: dict[str, Any]) -> str:
    """'K0 Ie As' (kinetics, influent view, anchor prior) for a G4 recoverability file."""
    tag = str(rec.get("tag", ""))
    k = cell_label("%s_ie_a0" % tag).split(" ")[0] if tag.startswith("k") else tag
    influent = {"exact": "Ie", "composite": "Ic", "composite_biased": "Ib"}.get(rec.get("influent_mode"), "")
    prior = Path(str((rec.get("prior") or {}).get("file", ""))).stem if isinstance(rec.get("prior"), dict) else ""
    return " ".join(p for p in (k, influent, prior) if p)


# -- Fig. 2 -----------------------------------------------------------------------
def fig2_recoverability(rec: dict[str, Any], states: dict[str, Any], validation: Path | None, fig_dir: Path,
                        cell: str, sigma: float, estimators: tuple[str, ...] = ("eks", "cl_pinn")) -> dict[str, Any]:
    """Three files: (a) class map with information gain, (b) memory time, (c) index against error."""
    index_name, val = validation_lookup(validation)
    if any(index_name not in s for s in rec["states"]):
        raise KeyError("recoverability file lacks the validation index %r" % index_name)
    classes = class_grid(rec)
    ig = state_grid(rec, "ig")
    tau = state_grid(rec, "tau_days")
    index = state_grid(rec, index_name)
    stats: dict[str, Any] = {"index": index_name, "cell": cell, "sigma": sigma, "estimators": {}, "files": []}
    from matplotlib.colors import ListedColormap

    # (a) class per tank and state, information gain as the number
    fig, ax = plt.subplots(figsize=(DOUBLE_IN, 2.3), layout="constrained")
    colour_idx = np.array([[CLASSES.index(c) if c else -1 for c in row] for row in classes], dtype=float)
    ax.imshow(np.ma.masked_less(colour_idx, 0), cmap=ListedColormap([CLASS_COLOURS[c] for c in CLASSES]),
              vmin=-0.5, vmax=len(CLASSES) - 0.5, aspect="auto")
    for k in range(5):
        for j in range(len(TRACK_B)):
            if np.isfinite(ig[k, j]):
                ax.text(j, k, _num(ig[k, j]), ha="center", va="center", fontsize=7,
                        color="white" if classes[k][j] in DARK_CLASSES else "black")
    ax.set_xticks(range(len(TRACK_B)))
    ax.set_xticklabels([_component(c) for c in TRACK_B])
    ax.set_yticks(range(5))
    ax.set_yticklabels(["tank %d" % (k + 1) for k in range(5)])
    ax.set_title("(a) Recoverability class (color) and information gain (number), %s" % rec_label(rec), loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=CLASS_COLOURS[c]) for c in CLASSES]
    present = {c for row in classes for c in row}
    fig.legend(handles, ["%s (%s%s)" % (c, CLASS_RULE[c], "" if c in present else "; none") for c in CLASSES],
               loc="outside lower center", ncol=2)
    stats["files"] += [p.name for p in _save(fig, fig_dir / "fig2a_recoverability_classes.png")]

    # (b) start-up memory time
    fig, ax = plt.subplots(figsize=(COLUMN_IN, 2.7), layout="constrained")
    med = np.nanmedian(tau, axis=0)
    lo, hi = np.nanmin(tau, axis=0), np.nanmax(tau, axis=0)
    ax.bar(range(len(TRACK_B)), med, color="0.6", yerr=[med - lo, hi - med], capsize=1.5,
           error_kw={"elinewidth": 0.6, "capthick": 0.6})
    longest = int(np.nanargmax(med))
    ax.axhline(THRESHOLD_TAU_DAYS, color="tab:red", ls="--", lw=0.8,
               label="anchor-carried threshold, 6 d (longest: %s, %.2f d)" % (_component(TRACK_B[longest]),
                                                                            med[longest]))
    ax.axhline(WINDOW_DAYS, color="black", ls=":", lw=0.8, label="length of window R0 (12 d)")
    ax.set_yscale("log")
    ax.set_xticks(range(len(TRACK_B)))
    ax.set_xticklabels([_component(c) for c in TRACK_B], rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylabel("start-up memory time $\\tau$ (d)")
    ax.set_title("(b) Memory time, %s" % rec_label(rec), loc="left")
    fig.legend(loc="outside lower center", ncol=1)
    stats["files"] += [p.name for p in _save(fig, fig_dir / "fig2b_memory_time.png")]

    # (c) registered index against the achieved per-state error (H6)
    fig, ax = plt.subplots(figsize=(COLUMN_IN, 3.1), layout="constrained")
    for n_est, estimator in enumerate(estimators):
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
        text = "\u03c1 = %s" % _num(rho) if entry is None else "\u03c1 = %s [%s, %s]" % (
            _num(rho), _num(entry["ci"][0]), _num(entry["ci"][1]))
        # Display only: states with an index below the floor (numerically zero) are drawn at the floor
        # as open markers; rho above is computed from the actual values.
        shown = np.maximum(idx[mask], INDEX_FLOOR)
        low = idx[mask] < INDEX_FLOOR
        ax.scatter(shown[~low], err[mask][~low], s=9, color=COLOURS.get(estimator), alpha=0.85, linewidths=0,
                   label="%s: %s" % (_label(estimator), text))
        if low.any():
            # side by side at the floor so that both estimators stay visible
            nudge = (0.8, 1.25)[n_est % 2]
            ax.scatter(np.full(int(low.sum()), INDEX_FLOOR * nudge), err[mask][low], s=14, marker="<",
                       facecolors="none", edgecolors=COLOURS.get(estimator), linewidths=0.7)
        stats["estimators"][estimator] = {"rho_points": rho, "n_points": int(mask.sum()),
                                          "n_below_display_floor": int(low.sum()),
                                          "validation": None if entry is None else
                                          {k: v for k, v in entry.items() if k != "points_index"}}
    ax.set_xscale("log")
    ax.set_yscale("log")
    if any(e.get("n_below_display_floor") for e in stats["estimators"].values()):
        ax.axvline(INDEX_FLOOR, color="0.6", lw=0.6, ls=":")
        ax.scatter([], [], s=14, marker="<", facecolors="none", edgecolors="0.3", linewidths=0.7,
                   label="index < 10$^{-4}$ (open triangles at 10$^{-4}$)")
    stats["index_display_floor"] = INDEX_FLOOR
    ax.set_xlabel("CRB / range (model-derived index)" if index_name == "crb_over_range"
                  else "%s (model-derived index)" % index_name.replace("_", " "))
    ax.set_ylabel("per-state NRMSE, %s" % WINDOW_TEXT["R0"])
    ax.set_title("(c) Index against error, %s" % cell_label(cell), loc="left")
    fig.legend(loc="outside lower center", ncol=1)
    stats["files"] += [p.name for p in _save(fig, fig_dir / "fig2c_index_vs_error.png")]
    stats["class_counts"] = {c: int(sum(row.count(c) for row in classes)) for c in CLASSES}
    return stats


# -- Fig. 3 -----------------------------------------------------------------------
def fig3_ladder(table: dict[str, Any], fig_dir: Path, cell: str, sigma: float) -> dict[str, Any]:
    """One file per window: median (diamond; value and seed count at the right edge), persistence line.

    Both windows share one x range so that the panels can be read side by side.
    """
    from matplotlib.lines import Line2D
    from matplotlib.transforms import blended_transform_factory

    plotted: dict[str, Any] = {"files": []}
    medians = [r["median"] for w in ("R0", "F") for e, r in rows_for(table, cell, sigma, w).items() if e in LADDER]
    xlim = (min(medians) / 2.5, max(medians) * 2.5) if medians else None
    for letter, window in (("a", "R0"), ("b", "F")):
        rows = rows_for(table, cell, sigma, window)
        ladder = [e for e in LADDER if e in rows]
        fig, ax = plt.subplots(figsize=(COLUMN_IN, 2.9), layout="constrained")
        at_right = blended_transform_factory(ax.transAxes, ax.transData)
        for y, estimator in enumerate(ladder):
            r = rows[estimator]
            ax.scatter([r["median"]], [y], s=26, marker="D", color=COLOURS.get(estimator, "black"), zorder=3,
                       linewidths=0)
            ax.text(1.03, y, _value(r["median"], r["n"]), transform=at_right, va="center", ha="left", fontsize=7)
            plotted.setdefault(estimator, {})[window] = {"median": r["median"], "values": r["values"]}
        if "persistence" in rows:
            ax.axvline(rows["persistence"]["median"], color="0.45", ls="--", lw=0.8)
        ax.set_xscale("log")
        ax.margins(y=0.08)
        if xlim:
            ax.set_xlim(*xlim)
        # a range under one decade would otherwise label the minor ticks, and they collide
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_yticks(range(len(ladder)))
        ax.set_yticklabels([_label(e) for e in ladder])
        ax.set_xlabel(METRIC_LABEL)
        # left-aligned to the figure: the long row labels leave the axes too narrow for the title
        fig.suptitle("(%s) Window %s (%s), %s" % (letter, window, WINDOW_TEXT[window], cell_label(cell)),
                     x=0.02, ha="left")
        handles = [Line2D([], [], marker="D", ls="none", color="black", ms=4,
                          label="median; (n) = seeds, if more than one"),
                   Line2D([], [], ls="--", color="0.45", lw=0.8, label="persistence")]
        fig.legend(handles=handles, loc="outside lower center", ncol=2)
        plotted["files"] += [p.name for p in _save(fig, fig_dir / ("fig3%s_ladder_%s.png" % (letter, window)))]
    return plotted


# -- Fig. 4 -----------------------------------------------------------------------
def _legend_file(handles: list, png: Path, ncol: int = 4) -> list[Path]:
    """A legend drawn on its own, shared by the panels of one figure."""
    rows = -(-len(handles) // ncol)
    fig = plt.figure(figsize=(DOUBLE_IN, 0.12 + 0.17 * rows))
    fig.legend(handles=handles, loc="center", ncol=ncol)
    return _save(fig, png)


def fig4_mismatch(table: dict[str, Any], fig_dir: Path, sigma: float, influent: str = "ie", anchor: str = "a0",
                  recovery_cell: str = "k100_ie_a0") -> dict[str, Any]:
    """(a) window R0 and (b) window F along the mismatch axis, (c) parameter recovery, plus a shared legend."""
    from matplotlib.lines import Line2D

    grid = [r for r in table["rows"] if r["k"] is not None and not r["extra"] and r["rand"] is None
            and not r["offsteady"] and r["influent"] == influent and r["anchor"] == anchor
            and abs(r["sigma"] - sigma) < 1e-9]
    plotted: dict[str, Any] = {"series": {}, "crossovers": [], "recovery": [], "files": []}
    ymax = max([r["max"] for r in grid if r["estimator"] in ("persistence",) + LINE_ESTIMATORS + POINT_ESTIMATORS]
               + [0.0])
    for letter, window in (("a", "R0"), ("b", "F")):
        fig, ax = plt.subplots(figsize=(COLUMN_IN, 2.8), layout="constrained")
        sub = [r for r in grid if r["window"] == window]
        for estimator in ("persistence",) + LINE_ESTIMATORS:
            pts = sorted((r["kinetic_alpha"], r["median"]) for r in sub if r["estimator"] == estimator)
            if len(pts) < 2:
                continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=2.5,
                    ls="--" if estimator == "persistence" else "-", color=COLOURS.get(estimator))
            plotted["series"].setdefault(window, {})[estimator] = pts
        for estimator in POINT_ESTIMATORS:
            pts = sorted((r["kinetic_alpha"], r["median"], r["min"], r["max"], r["n"]) for r in sub
                         if r["estimator"] == estimator)
            if not pts:
                continue
            for p in pts:
                # side by side at each alpha, clear of the line markers
                x = p[0] + (0.025 if estimator == "cl_pinn_theta" else -0.025)
                # Section 9: one-seed and two-seed PINN rows are labelled -- open marker
                ax.errorbar([x], [p[1]], yerr=[[p[1] - p[2]], [p[3] - p[1]]], fmt="s", ms=4, capsize=2,
                            elinewidth=0.7, capthick=0.7, color=COLOURS.get(estimator),
                            mfc="white" if p[4] < 3 else COLOURS.get(estimator))
            plotted["series"].setdefault(window, {})[estimator] = [list(p) for p in pts]
        crossing = [c for c in table["crossovers"]
                    if (c["influent"], c["anchor"], c["window"], c["status"]) == (influent, anchor, window, "crosses")
                    and abs(c["sigma"] - sigma) < 1e-9 and c["estimator"] in ("ode_openloop_reduced", "eks")]
        crossing = sorted(crossing, key=lambda c: c["alpha_star"])
        for c in crossing:
            ax.axvline(c["alpha_star"], color=COLOURS.get(c["estimator"]), ls=":", lw=0.9)
            plotted["crossovers"].append(c)
        if crossing:
            # alpha* as coloured ticks on a top axis: no label can sit on a data line
            top = ax.secondary_xaxis("top")
            top.set_xticks([c["alpha_star"] for c in crossing])
            # a label closer than 0.09 in alpha to the one before it is raised by one line
            texts, raised = [], False
            for k, c in enumerate(crossing):
                raised = (not raised) and k > 0 and c["alpha_star"] - crossing[k - 1]["alpha_star"] < 0.09
                texts.append(_num(c["alpha_star"]) + ("\n" if raised else ""))
            labels = top.set_xticklabels(texts)
            for lab, c in zip(labels, crossing):
                lab.set_color(COLOURS.get(c["estimator"]))
            top.set_xlabel("crossover $\\alpha^*$", labelpad=3)
        ax.set_xlim(-0.06, 1.06)
        ax.set_ylim(0.0, ymax * 1.06)
        ax.set_xlabel("kinetic mismatch $\\alpha$\n(0 = textbook set, 1 = BSM1 15 \u00b0C set)")
        ax.set_ylabel(METRIC_LABEL)
        ax.set_title("(%s) Window %s (%s), %s %s" % (letter, window, WINDOW_TEXT[window], influent.capitalize(),
                                                     anchor.capitalize()), loc="left")
        plotted["files"] += [p.name for p in _save(fig, fig_dir / ("fig4%s_mismatch_%s.png" % (letter, window)))]

    handles = [Line2D([], [], ls="--", marker="o", ms=2.5, color=COLOURS["persistence"], label=_label("persistence"))]
    same_f = plotted["series"].get("F", {})
    # on window F the smoother and the filter forecast from the same day-12 state
    notes = {"ekf": " (equal to EKS on window F)"} if same_f.get("ekf") is not None \
        and same_f.get("ekf") == same_f.get("eks") else {}
    handles += [Line2D([], [], ls="-", marker="o", ms=2.5, color=COLOURS[e], label=_label(e) + notes.get(e, ""))
                for e in LINE_ESTIMATORS]
    handles += [Line2D([], [], ls="none", marker="s", ms=4, color=COLOURS[e],
                       label="%s (median, range over seeds)" % _label(e)) for e in POINT_ESTIMATORS]
    handles += [Line2D([], [], ls="none", marker="s", ms=4, color="0.3", mfc="white",
                       label="open marker: fewer than three seeds"),
                Line2D([], [], ls=":", lw=0.9, color="0.3",
                       label="$\\alpha^*$: open loop or EKS crosses persistence (tick in its color)")]
    plotted["files"] += [p.name for p in _legend_file(handles, fig_dir / "fig4_legend.png", ncol=2)]

    recovery = [p for p in table["parameter_recovery"] if p["cell"] == recovery_cell and abs(p["sigma"] - sigma) < 1e-9
                and p["estimator"] in RECOVERY_ESTIMATORS]
    fig, ax = plt.subplots(figsize=(COLUMN_IN, 3.1), layout="constrained")
    ax.set_title("(c) Parameter recovery, %s" % cell_label(recovery_cell), loc="left")
    names_seen: list[str] = []
    lims: list[float] = []
    for p in recovery:
        names = [n for n in p["names"] if p["true_log_ratio"][n] is not None]
        for n in names:
            if n not in names_seen:
                names_seen.append(n)
        for est in p["estimates"]:
            for n in names:
                x, y = p["true_log_ratio"][n], est["log_multiplier"][n]
                ax.scatter([x], [y], s=16, marker=PARAMETER_MARKERS.get(n, "o"), linewidths=0,
                           color=COLOURS.get(base_model(p["estimator"]), "black"), alpha=0.9)
                lims += [x, y]
        plotted["recovery"].append(p)
    if lims:
        lo, hi = min(lims) - 0.1, max(lims) + 0.1
        ax.plot([lo, hi], [lo, hi], color="0.5", lw=0.6)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
    ax.set_xlabel("true ln(truth / vault)")
    ax.set_ylabel("estimated ln(multiplier)")
    # legend columns (filled top to bottom): estimators and the identity line, then the parameters
    blank = Line2D([], [], ls="none", marker="none", label=" ")
    column1 = [Line2D([], [], ls="none", marker="o", ms=4, color=COLOURS.get(e), label=_label(e))
               for e in RECOVERY_ESTIMATORS if any(base_model(p["estimator"]) == e for p in recovery)]
    if lims:
        column1.append(Line2D([], [], color="0.5", lw=0.6, label="exact recovery"))
    params = [Line2D([], [], ls="none", marker=PARAMETER_MARKERS.get(n, "o"), ms=4, color="0.35",
                     label=PARAMETER_TEXT.get(n, n)) for n in names_seen]
    rows = max(len(column1), -(-len(params) // 2), 1)
    handles = column1 + [blank] * (rows - len(column1))
    for start in range(0, len(params), rows):
        chunk = params[start:start + rows]
        handles += chunk + [blank] * (rows - len(chunk))
    if handles and any(h.get_label().strip() for h in handles):
        fig.legend(handles=handles, loc="outside lower center", ncol=len(handles) // rows)
    plotted["files"] += [p.name for p in _save(fig, fig_dir / "fig4c_parameter_recovery.png")]
    return plotted


# -- Fig. 5 -----------------------------------------------------------------------
def _grouped_bars(ax, table: dict[str, Any], cells: list[str], sigma: float, labels: list[str]) -> dict[str, Any]:
    width = 0.8 / len(FACTORIAL_ESTIMATORS)
    out: dict[str, Any] = {}
    for j, estimator in enumerate(FACTORIAL_ESTIMATORS):
        for i, cell in enumerate(cells):
            r = rows_for(table, cell, sigma, "R0").get(estimator)
            if r is None:
                continue
            x = i + (j - (len(FACTORIAL_ESTIMATORS) - 1) / 2) * width
            ax.bar(x, r["median"], width, color=COLOURS.get(estimator),
                   yerr=[[r["median"] - r["min"]], [r["max"] - r["median"]]] if r["n"] > 1 else None, capsize=1.5,
                   error_kw={"elinewidth": 0.6, "capthick": 0.6})
            out.setdefault(cell, {})[estimator] = r["median"]
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.5, len(cells) - 0.5)
    return out


def fig5_factorial(table: dict[str, Any], states: dict[str, Any], rec_realistic: dict[str, Any], fig_dir: Path,
                   realistic_k: str, sigma: float) -> dict[str, Any]:
    """(a) influent x kinetics, (b) start-state tiers, (c) error by recoverability class, plus a shared legend."""
    k = realistic_k
    files: list[str] = []
    ylabel = METRIC_LABEL

    fig, ax = plt.subplots(figsize=(COLUMN_IN, 2.6), layout="constrained")
    cells_a = ["k000_ie_a0", "k000_ic_a0", "k000_ib_a0", "k%s_ie_a0" % k, "k%s_ic_a0" % k, "k%s_ib_a0" % k]
    plotted = {"a": _grouped_bars(ax, table, cells_a, sigma,
                                  [cell_label(c).replace(" A0", "").replace(" ", "\n") for c in cells_a])}
    ax.set_title("(a) Influent \u00d7 kinetics, A0", loc="left")
    ax.set_ylabel(ylabel)
    files += [p.name for p in _save(fig, fig_dir / "fig5a_influent_kinetics.png")]

    fig, ax = plt.subplots(figsize=(COLUMN_IN, 2.6), layout="constrained")
    cells_b = ["k%s_ic_%s" % (k, a) for a in ("a0", "as", "al1", "al2")]
    plotted["b"] = _grouped_bars(ax, table, cells_b, sigma, ["A0", "As", "Al1", "Al2"])
    ax.set_title("(b) Start-state tiers, %s" % cell_label(cells_b[0]).rsplit(" ", 1)[0], loc="left")
    ax.set_ylabel(ylabel)
    files += [p.name for p in _save(fig, fig_dir / "fig5b_start_state_tiers.png")]

    fig, ax = plt.subplots(figsize=(COLUMN_IN, 2.6), layout="constrained")
    classes = class_grid(rec_realistic)
    counts = {c: int(sum(row.count(c) for row in classes)) for c in CLASSES}
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
            ax.bar(i + (j - (len(FACTORIAL_ESTIMATORS) - 1) / 2) * width, value, width, color=COLOURS.get(estimator))
            plotted["c"].setdefault(cls, {})[estimator] = value
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels(["%s\n(n = %d)" % (CLASS_SHORT[c], counts[c]) for c in CLASSES])
    ax.set_xlim(-0.5, len(CLASSES) - 0.5)
    ax.set_xlabel("class from the %s recoverability map" % rec_label(rec_realistic))
    ax.set_title("(c) Error by class, %s" % cell_label(cell_c), loc="left")
    ax.set_ylabel("mean per-state NRMSE (fixed R0 range)")
    files += [p.name for p in _save(fig, fig_dir / "fig5c_recoverability_class.png")]
    plotted["c_counts"] = counts

    no_pinn = [lab for lab, cells in (("Ib", cells_a[2::3]), ("Al2", cells_b[3:]))
               if not any(rows_for(table, c, sigma, "R0").get("cl_pinn") for c in cells)]
    labels = [_label(e) for e in FACTORIAL_ESTIMATORS]
    if no_pinn and "cl_pinn" in FACTORIAL_ESTIMATORS:
        labels[FACTORIAL_ESTIMATORS.index("cl_pinn")] += " (not run with %s)" % " or ".join(no_pinn)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOURS.get(e), label=lab)
               for e, lab in zip(FACTORIAL_ESTIMATORS, labels)]
    files += [p.name for p in _legend_file(handles, fig_dir / "fig5_legend.png", ncol=len(handles))]
    plotted["files"] = files
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
    with print_style():
        sources["fig2"] = fig2_recoverability(rec_map, states, validation, fig_dir, h6_cell or args.map_cell,
                                              args.sigma)
        sources["fig3"] = fig3_ladder(table, fig_dir, args.map_cell, args.sigma)
        sources["fig4"] = fig4_mismatch(table, fig_dir, args.sigma, recovery_cell="k100_ie_a0")
        sources["fig5"] = fig5_factorial(table, states, rec_real, fig_dir, realistic_k, args.sigma)
    sources["layout"] = {"column_in": COLUMN_IN, "double_in": DOUBLE_IN,
                         "note": "each file is drawn at its printed width; include it at natural size"}
    (fig_dir / "figure_sources.json").write_text(json.dumps(sources, indent=1, default=float), encoding="utf-8")
    written = [fig_dir / name for key in ("fig2", "fig3", "fig4", "fig5") for name in sources[key]["files"]]
    if args.paper_dir and Path(args.paper_dir).exists():
        for path in written:
            shutil.copy2(path, Path(args.paper_dir) / path.name)
    for path in written:
        print("wrote", path)
    print("wrote", fig_dir / "figure_sources.json")


if __name__ == "__main__":
    main()

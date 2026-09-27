"""Figures 2-5 draw from saved JSON only; Fig. 2c refuses a rho that disagrees with G4."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import spearmanr

from scripts import v11_figures as vf
from scripts.regime_map import TRACK_B

ESTIMATORS = ("persistence", "ode_openloop_reduced", "ode_openloop_full", "lstm_v10", "pinn", "cl_pinn",
              "cl_pinn_theta", "ekf", "eks", "ieks", "eks_aug")
CELLS = ["k000_ie_a0", "k025_ie_a0", "k050_ie_a0", "k075_ie_a0", "k100_ie_a0", "k000_ic_a0", "k000_ib_a0",
         "k100_ic_a0", "k100_ib_a0", "k100_ic_as", "k100_ic_al1", "k100_ic_al2"]


def _row(cell: str, estimator: str, window: str, value: float, n: int) -> dict:
    k = cell[1:4]
    parts = cell.split("_")
    values = [value * (1 + 0.05 * s) for s in range(n)]
    return {"cell": cell, "k": k, "offsteady": False, "influent": parts[1], "anchor": parts[2], "extra": "",
            "rand": None, "kinetic_alpha": int(k) / 100, "sigma": 0.1, "estimator": estimator,
            "family": "pinn" if "pinn" in estimator else "observers", "window": window, "primary": True,
            "n": n, "seeds": list(range(n)), "values": values, "median": float(np.median(values)),
            "min": min(values), "max": max(values), "skill": 0.1, "gap_closed": 0.5}


@pytest.fixture
def artefacts(tmp_path):
    rng = np.random.default_rng(0)
    rows, entries = [], []
    for cell in CELLS:
        alpha = int(cell[1:4]) / 100
        for estimator in ESTIMATORS:
            n = 3 if estimator in ("pinn", "cl_pinn", "cl_pinn_theta") else 1
            for window in ("R0", "R2", "F"):
                base = 0.3 if estimator == "persistence" else 0.1 + 0.5 * alpha * (estimator == "ode_openloop_reduced")
                rows.append(_row(cell, estimator, window, base + 0.01 * ESTIMATORS.index(estimator), n))
            for window in ("R0", "F"):
                entries.append({"cell": cell, "sigma": 0.1, "estimator": estimator, "window": window, "n": n,
                                "median": rng.uniform(0.01, 1.0, (5, 11)).tolist()})
    table = {"meta": {"metric": "track_b_nrmse_fixed", "rule": "All-seeds rule."}, "rows": rows,
             "comparisons": [], "winners": [],
             "crossovers": [{"influent": "ie", "anchor": "a0", "sigma": 0.1, "window": "R0",
                             "estimator": "ode_openloop_reduced", "alpha_star": 0.45, "status": "crosses",
                             "points": []}],
             "parameter_recovery": [{"cell": "k100_ie_a0", "sigma": 0.1, "estimator": "cl_pinn_theta",
                                     "names": ["bA", "muA"], "true_log_ratio": {"bA": -0.2, "muA": -0.8},
                                     "estimates": [{"seed": 0, "log_multiplier": {"bA": -0.1, "muA": -0.7}}]}],
             "realisation_spread": []}
    root = tmp_path / "v11"
    (root / "analysis").mkdir(parents=True)
    (root / "regime_table.json").write_text(json.dumps(table), encoding="utf-8")
    states = {"meta": {"components": list(TRACK_B)}, "entries": entries}
    (root / "regime_states.json").write_text(json.dumps(states), encoding="utf-8")
    classes = vf.CLASSES
    rec_states = [{"tank": k + 1, "component": c, "ig": float(rng.uniform()), "crb_over_range": float(rng.uniform(0.01, 2)),
                   "tau_days": float(rng.uniform(0.05, 20)), "forcing_share": float(rng.uniform()),
                   "class": classes[(k + j) % 4].replace("-", "_")}
                  for k in range(5) for j, c in enumerate(TRACK_B)]
    channels = ["S_O_tank3", "S_O_tank4", "S_O_tank5", "S_NH_tank5", "S_NO_tank2", "S_NO_tank5", "TSS_tank5"]
    subsets = [{"channels": channels, "ig_mean": 0.6,
                "per_state_ig": {c: [0.6] * 5 for c in TRACK_B}}]
    for drop in channels:
        loss = 0.3 if drop == "TSS_tank5" else 0.0
        subsets.append({"channels": [c for c in channels if c != drop], "ig_mean": 0.6 - loss,
                        "per_state_ig": {c: [0.6 - loss * (c in ("X_I", "X_P"))] * 5 for c in TRACK_B}})
    assays = [{"tier": "Al1", "assay": "TSS", "ig_gain": {c: 0.2 * (c in ("X_I", "X_P")) for c in TRACK_B}},
              {"tier": "Al2", "assay": "respirometric X_B_H", "ig_gain": {"X_B_H": 0.4}}]
    for tag in ("k000", "k100"):
        (root / "analysis" / ("recoverability_%s.json" % tag)).write_text(
            json.dumps({"tag": tag, "states": rec_states, "sensor_subsets": subsets, "lab_assays": assays}),
            encoding="utf-8")
    index = vf.state_grid({"states": rec_states}, "crb_over_range")
    results = []
    for estimator in ("eks", "cl_pinn"):
        err = vf.state_entry(states, "k000_ie_a0", 0.1, estimator, "R0")
        rho = float(spearmanr(index.ravel(), err.ravel()).statistic)
        results.append({"cell": "k000_ie_a0", "estimator": estimator, "rho": rho, "ci": [rho - 0.2, rho + 0.2],
                        "n_points": 55, "pass": rho >= 0.5})
    (root / "analysis" / "recoverability_validation.json").write_text(
        json.dumps({"index": "crb_over_range", "window": "R0", "results": results}), encoding="utf-8")
    return root


def _argv(root: Path) -> list[str]:
    a = root / "analysis"
    return ["--root", str(root), "--rec-map", str(a / "recoverability_k000.json"),
            "--rec-realistic", str(a / "recoverability_k100.json"),
            "--validation", str(a / "recoverability_validation.json"), "--paper-dir", ""]


def test_all_figures_and_sources_are_written(artefacts):
    vf.main(_argv(artefacts))
    fig_dir = artefacts / "figures"
    stems = ("fig2a_recoverability_classes", "fig2b_memory_time", "fig2c_index_vs_error", "fig3a_ladder_R0",
             "fig3b_ladder_F", "fig4a_mismatch_R0", "fig4b_mismatch_F", "fig4c_parameter_recovery", "fig4_legend",
             "fig5a_influent_kinetics", "fig5b_start_state_tiers", "fig5c_recoverability_class", "fig5_legend")
    for stem in stems:
        assert (fig_dir / (stem + ".png")).exists() and (fig_dir / (stem + ".pdf")).exists()
    sources = json.loads((fig_dir / "figure_sources.json").read_text(encoding="utf-8"))
    assert set(sources) >= {"inputs", "fig2", "fig3", "fig4", "fig5", "layout"}
    listed = {name for key in ("fig2", "fig3", "fig4", "fig5") for name in sources[key]["files"]}
    assert listed == {stem + ext for stem in stems for ext in (".png", ".pdf")}
    assert sources["fig3"]["cl_pinn"]["R0"]["median"] == pytest.approx(0.15 * 1.05)
    assert sources["fig4"]["crossovers"][0]["alpha_star"] == 0.45
    assert sum(sources["fig2"]["class_counts"].values()) == 55


def test_fig2_refuses_a_rho_that_disagrees_with_the_validation_file(artefacts):
    path = artefacts / "analysis" / "recoverability_validation.json"
    val = json.loads(path.read_text(encoding="utf-8"))
    val["results"][0]["rho"] += 0.1
    path.write_text(json.dumps(val), encoding="utf-8")
    with pytest.raises(ValueError, match="different error definitions"):
        vf.main(_argv(artefacts))


def test_unknown_class_names_are_rejected():
    with pytest.raises(ValueError):
        vf.normalise_class("mostly fine")
    assert vf.normalise_class("Anchor_carried") == "anchor-carried"


REPO = Path(__file__).resolve().parents[1]


def test_the_g4_recoverability_file_is_normalised_to_the_contract():
    path = REPO / "results" / "v11" / "analysis" / "recoverability_k000.json"
    if not path.exists():
        pytest.skip("needs the G4 recoverability file")
    raw = json.loads(path.read_text(encoding="utf-8"))
    rec = vf.load_recoverability(path)
    subsets = rec["sensor_subsets"]
    assert len(subsets) == len(raw["sensor_value"]["subsets"])
    full = max(subsets, key=lambda s: len(s["channels"]))
    assert len(full["channels"]) == 7
    for comp, tanks in full["per_state_ig"].items():
        assert len(tanks) == 5 and len(set(tanks)) == 1
    assert np.mean([t[0] for t in full["per_state_ig"].values()]) == pytest.approx(full["ig_mean"], rel=1e-9)
    tiers = {a["tier"] for a in rec["lab_assays"]}
    assert tiers <= {"Al1", "Al2"} and rec["lab_assays"]
    assert all(set(a["ig_gain"]) == set(TRACK_B) for a in rec["lab_assays"])


def test_per_assay_gains_are_used_when_the_g4_file_carries_them():
    base = {c: 0.1 for c in TRACK_B}
    rec = {"tag": "kx", "states": [], "sensor_value": {"subsets": []},
           "lab_assays": {"tiers": {"As": {"ig_by_component": base}},
                          "add_one_to_As": [
                              {"assay": "TSS", "delta_J": 0.02, "ig_by_component": {**base, "X_I": 0.5}},
                              {"assay": "respirometry", "delta_J": 0.03, "ig_by_component": {**base, "X_B_H": 0.9}}]}}
    out = vf.normalise_recoverability(rec)
    by = {a["assay"]: a for a in out["lab_assays"]}
    assert by["TSS"]["tier"] == "Al1" and by["TSS"]["ig_gain"]["X_I"] == pytest.approx(0.4)
    assert by["respirometry"]["tier"] == "Al2" and by["respirometry"]["ig_gain"]["X_B_H"] == pytest.approx(0.8)


def _g4_validation(cell, estimator_points, h6_pass=True):
    cells = {cell: {"cell": {"name": cell}, "estimators": {}}}
    for est, (index, rho) in estimator_points.items():
        cells[cell]["estimators"][est] = {
            "status": "ok", "primary_index": {"rho": rho, "ci95": [rho - 0.1, rho + 0.1], "n_points": 55},
            "secondary_index": {"rho": 0.0, "ci95": [-0.1, 0.1], "n_points": 55},
            "points": {"index": index.tolist(), "nrmse": []}}
    return {"label": "pre-registered H6 validation", "sigma": 0.1, "primary_cell": cell, "cells": cells,
            "h6": {"cell": cell, "estimator": "eks", "rho": estimator_points["eks"][1], "pass": h6_pass}}


def test_the_g4_validation_layout_is_normalised():
    idx = np.arange(55, dtype=float).reshape(5, 11)
    val = vf.normalise_validation(_g4_validation("k050_ic_as", {"eks": (idx, 0.7), "cl_pinn": (idx, 0.4)}))
    assert val["primary_cell"] == "k050_ic_as" and val["index"] == "crb_over_range"
    res = {r["estimator"]: r for r in val["results"]}
    assert res["eks"]["pass"] is True and res["cl_pinn"]["pass"] is None
    assert res["eks"]["ci"] == [pytest.approx(0.6), pytest.approx(0.8)]


def test_fig2c_uses_the_validation_index_points_of_the_h6_cell(artefacts):
    states = json.loads((artefacts / "regime_states.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(1)
    idx = rng.uniform(0.01, 2.0, (5, 11))
    points = {}
    for est in ("eks", "cl_pinn"):
        err = vf.state_entry(states, "k000_ie_a0", 0.1, est, "R0")
        points[est] = (idx, float(spearmanr(idx.ravel(), err.ravel()).statistic))
    path = artefacts / "analysis" / "validation" / "recoverability_validation.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_g4_validation("k000_ie_a0", points)), encoding="utf-8")
    argv = _argv(artefacts)
    argv[argv.index("--validation") + 1] = str(path)
    vf.main(argv)
    fig2 = json.loads((artefacts / "figures" / "figure_sources.json").read_text(encoding="utf-8"))["fig2"]
    assert fig2["cell"] == "k000_ie_a0"
    assert fig2["estimators"]["eks"]["rho_points"] == pytest.approx(points["eks"][1], abs=1e-12)
    bad = _g4_validation("k000_ie_a0", {"eks": (idx, points["eks"][1] + 0.05), "cl_pinn": points["cl_pinn"]})
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="different error definitions"):
        vf.main(argv)


def test_recovery_inset_keeps_the_h3_pair_only(artefacts):
    table_path = artefacts / "regime_table.json"
    table = json.loads(table_path.read_text(encoding="utf-8"))
    extra = dict(table["parameter_recovery"][0], estimator="ekf_aug_frozenq")
    table["parameter_recovery"].append(extra)
    table_path.write_text(json.dumps(table), encoding="utf-8")
    vf.main(_argv(artefacts))
    fig4 = json.loads((artefacts / "figures" / "figure_sources.json").read_text(encoding="utf-8"))["fig4"]
    assert [p["estimator"] for p in fig4["recovery"]] == ["cl_pinn_theta"]


def test_figures_refuse_a_missing_validation_file(artefacts):
    argv = _argv(artefacts)
    argv[argv.index("--validation") + 1] = str(artefacts / "analysis" / "missing.json")
    with pytest.raises(FileNotFoundError):
        vf.main(argv)


def test_realistic_cells_follow_the_regime_table(artefacts):
    table_path = artefacts / "regime_table.json"
    table = json.loads(table_path.read_text(encoding="utf-8"))
    table["meta"]["realistic_k"] = "050"
    table_path.write_text(json.dumps(table), encoding="utf-8")
    with pytest.raises(ValueError, match="gate D3|not k050"):
        vf.main(_argv(artefacts))


def test_fig2c_needs_the_validation_entry_of_the_h6_cell(artefacts):
    states = json.loads((artefacts / "regime_states.json").read_text(encoding="utf-8"))
    idx = np.ones((5, 11))
    val = _g4_validation("k000_ie_a0", {"eks": (idx, 0.5), "cl_pinn": (idx, 0.4)})
    del val["cells"]["k000_ie_a0"]["estimators"]["cl_pinn"]
    path = artefacts / "analysis" / "validation" / "recoverability_validation.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(val), encoding="utf-8")
    assert vf.state_entry(states, "k000_ie_a0", 0.1, "cl_pinn", "R0") is not None
    argv = _argv(artefacts)
    argv[argv.index("--validation") + 1] = str(path)
    with pytest.raises(ValueError, match="no cl_pinn entry"):
        vf.main(argv)


def test_every_panel_is_drawn_at_a_journal_width(artefacts):
    from PIL import Image

    from scripts.figure_layout import COLUMN_IN, DOUBLE_IN

    vf.main(_argv(artefacts))
    sources = json.loads((artefacts / "figures" / "figure_sources.json").read_text(encoding="utf-8"))
    widths = set()
    for key in ("fig2", "fig3", "fig4", "fig5"):
        for name in sources[key]["files"]:
            if name.endswith(".png"):
                width = Image.open(artefacts / "figures" / name).size[0] / 600
                assert min(abs(width - COLUMN_IN), abs(width - DOUBLE_IN)) < 2 / 600, name
                widths.add(round(width, 2))
    assert widths == {round(COLUMN_IN, 2), round(DOUBLE_IN, 2)}


def test_every_panel_goes_through_the_layout_check(artefacts, monkeypatch):
    """Each written file passes save_checked once; a forced collision stops the script."""
    import matplotlib.axes

    checked = []
    original_check = vf.save_checked

    def spy(fig, png, save):
        checked.append(Path(png).name)
        return original_check(fig, png, save)

    monkeypatch.setattr(vf, "save_checked", spy)
    vf.main(_argv(artefacts))
    sources = json.loads((artefacts / "figures" / "figure_sources.json").read_text(encoding="utf-8"))
    pngs = [n for key in ("fig2", "fig3", "fig4", "fig5") for n in sources[key]["files"] if n.endswith(".png")]
    assert sorted(checked) == sorted(pngs) and len(checked) == len(set(checked)) == 13

    original = matplotlib.axes.Axes.set_title

    def crowded(self, label, *args, **kwargs):
        out = original(self, label, *args, **kwargs)
        self.text(0.5, 0.5, "collision one", transform=self.transAxes)
        self.text(0.52, 0.5, "collision two", transform=self.transAxes)
        return out

    monkeypatch.setattr(matplotlib.axes.Axes, "set_title", crowded)
    with pytest.raises(ValueError, match="layout issue"):
        vf.main(_argv(artefacts))


def test_fig5_names_missing_pinn_arms_and_counts_the_classes(artefacts):
    table_path = artefacts / "regime_table.json"
    table = json.loads(table_path.read_text(encoding="utf-8"))
    table["rows"] = [r for r in table["rows"]
                     if not (r["estimator"] == "cl_pinn" and r["cell"] in ("k000_ib_a0", "k100_ib_a0", "k100_ic_al2"))]
    table_path.write_text(json.dumps(table), encoding="utf-8")
    captured = {}
    original = vf._legend_file

    def spy(handles, png, ncol=4):
        captured[Path(png).name] = [h.get_label() for h in handles]
        return original(handles, png, ncol)

    import pytest as _pytest

    with _pytest.MonkeyPatch.context() as mp:
        mp.setattr(vf, "_legend_file", spy)
        vf.main(_argv(artefacts))
    assert "CL-PINN (not run with Ib or Al2)" in captured["fig5_legend.png"]
    fig5 = json.loads((artefacts / "figures" / "figure_sources.json").read_text(encoding="utf-8"))["fig5"]
    assert sum(fig5["c_counts"].values()) == 55

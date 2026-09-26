"""component_results: v1.0 outputs reproduce; generic roots and windows work."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from scripts import component_results

REPO = Path(__file__).resolve().parents[1]


def test_defaults_reproduce_the_v10_component_table(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    component_results.main(["--out-dir", str(tmp_path), "--no-figures"])
    produced = json.loads((tmp_path / "component_table.json").read_text(encoding="utf-8"))
    reference = json.loads((REPO / "results" / "component_table.json").read_text(encoding="utf-8"))
    assert produced == reference
    assert (tmp_path / "component_table.tex").read_text(encoding="utf-8") == (
        REPO / "results" / "component_table.tex"
    ).read_text(encoding="utf-8")


def test_generic_roots_take_the_median_over_roots(tmp_path, monkeypatch):
    """Three roots holding one seed each behave like the v1.0 seed directories."""
    monkeypatch.chdir(REPO)
    roots = []
    for k, src in enumerate(("results/runs", "results/runs_seed1", "results/runs_seed2")):
        root = tmp_path / ("cell_seed%d" % k)
        shutil.copytree(REPO / src / "cl_pinn_sigma0p10", root / "cl_pinn_sigma0p10")
        roots.append(str(root))
    out = tmp_path / "out"
    component_results.main(
        ["--runs", *roots, "--data-dir", "results/raw", "--sigmas", "0.10", "--models", "cl_pinn",
         "--heat-models", "cl_pinn", "--table-models", "cl_pinn", "--out-dir", str(out),
         "--tag", "_cell", "--window", "train", "--no-figures"]
    )
    table = json.loads((out / "component_table_cell.json").read_text(encoding="utf-8"))
    assert table["cl_pinn"]["0p10"]["_n_seeds"] == 3.0
    assert np.isfinite(table["cl_pinn"]["0p10"]["X_B_H"])
    assert (out / "component_table_cell.tex").exists()


def test_track_b_row_can_be_the_median_over_seeds_of_the_primary_metric(tmp_path, monkeypatch):
    """--trackb-row median-of-means gives the regime table's number: median over seeds of each seed's Track B mean."""
    from src.data.sensors import ObservationDataset, unobserved_components
    from src.eval.metrics import state_metrics

    monkeypatch.chdir(REPO)
    roots, means = [], []
    truth = ObservationDataset.load(REPO / "results/raw/obs_dry_sigma0p10.npz").window(0.0, 12.0).truth_reactor
    for k, src in enumerate(("results/runs", "results/runs_seed1", "results/runs_seed2")):
        root = tmp_path / ("cell_seed%d" % k)
        shutil.copytree(REPO / src / "cl_pinn_sigma0p10", root / "cl_pinn_sigma0p10")
        roots.append(str(root))
        with np.load(root / "cl_pinn_sigma0p10" / "predictions.npz") as p:
            pred = p["train"][: len(truth)]
        m = state_metrics(truth, pred)
        means.append(np.nanmean([m.nrmse[m.components.index(c)] for c in unobserved_components()]))
    out = tmp_path / "out"
    component_results.main(["--runs", *roots, "--data-dir", "results/raw", "--sigmas", "0.10", "--models", "cl_pinn",
                            "--heat-models", "cl_pinn", "--table-models", "cl_pinn", "--out-dir", str(out),
                            "--tag", "_m", "--window", "train", "--no-figures", "--trackb-row", "median-of-means"])
    last = (out / "component_table_m.tex").read_text(encoding="utf-8").strip().splitlines()[-1]
    assert last.startswith("Track~B (median over seeds) & ")
    assert last.split("&")[1].strip().rstrip("\\").strip() == "%.3f" % np.median(means)

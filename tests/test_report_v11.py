"""collect_runs v1.1 extensions: extra windows, level error, tolerance, metadata."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from src.data.sensors import ObservationDataset
from src.eval.metrics import state_metrics, track_summary
from src.eval.report import (
    LEVEL_COMPONENTS,
    ROW_METADATA,
    TOLERANCE_COMPONENTS,
    collect_runs,
    dataset_descriptors,
    score_run,
    truth_plant,
)

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "results" / "runs"
RAW = REPO / "results" / "raw"


def _same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or a == b
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return a == b


def test_default_rows_reproduce_the_v10_benchmark_detail():
    """extra_windows=None leaves every v1.0 key and value unchanged."""
    reference = json.loads((REPO / "results" / "benchmark_detail.json").read_text(encoding="utf-8"))["rows"]
    rows = collect_runs(RUNS, RAW)
    assert len(rows) == len(reference) == 72
    for row, ref in zip(rows, reference):
        assert _same({k: row[k] for k in ref}, ref), (row["run_id"], row["eval_set"])


def test_rows_carry_level_error_tolerance_and_metadata():
    rows = score_run(RUNS / "cl_pinn_sigma0p10", RAW)
    assert [r["eval_set"] for r in rows] == ["train", "holdout", "rain"]
    for row in rows:
        for name in LEVEL_COMPONENTS:
            assert np.isfinite(row["level_error_%s" % name])
        for name in TOLERANCE_COMPONENTS:
            assert 0.0 <= row["tol10_%s" % name] <= 1.0
        assert set(ROW_METADATA) <= set(row)
        assert row["truth_preset"] == "vault20" and row["influent_mode"] == "exact"
        assert row["seed"] == 0


def test_extra_window_slices_prediction_and_truth_by_time(tmp_path):
    run = tmp_path / "runs" / "persistence_sigma0p10"
    shutil.copytree(RUNS / "persistence_sigma0p10", run)
    rows = collect_runs(tmp_path / "runs", RAW, extra_windows={"recon2": ("train", 2.0, 12.0)})
    labels = [r["eval_set"] for r in rows]
    assert labels == ["train", "holdout", "rain", "recon2"]
    recon2 = rows[-1]
    truth = ObservationDataset.load(RAW / "obs_dry_sigma0p10.npz").window(2.0, 12.0).truth_reactor
    with np.load(run / "predictions.npz") as preds:
        pred = preds["train"]
    t_train = ObservationDataset.load(RAW / "obs_dry_sigma0p10.npz").window(0.0, 12.0).t
    sliced = pred[: len(t_train)][t_train >= 2.0 - 1e-9]
    assert sliced.shape == truth.shape
    expected = track_summary(state_metrics(truth, sliced))["track_b_unmeasured"]["nrmse"]
    assert recon2["track_b_nrmse"] == pytest.approx(expected, rel=1e-12)


def test_truth_plant_uses_the_vault_for_v10_data_and_the_preset_otherwise():
    from src.data.simulate import SimulationResult

    meta = SimulationResult.load(RAW / "sim_dry.npz").meta
    assert truth_plant(meta).vault.parameters == truth_plant({}).vault.parameters
    from src.asm1.truth_plants import truth_vault

    bsm1 = dict(truth_vault("bsm1_15c").parameters)
    plant = truth_plant({**meta, "truth_preset": "bsm1_15c", "alpha": 1.0, "parameters": bsm1})
    assert plant.vault.parameters["muH"] == pytest.approx(4.0)
    with pytest.raises(ValueError):
        truth_plant({**meta, "truth_preset": "bsm1_15c", "parameters": {**bsm1, "muH": 5.0}})


def test_dataset_descriptors_records_the_truth_preset():
    out = dataset_descriptors(RAW)
    assert out["dry"]["truth_preset"] == "vault20"
    assert np.isfinite(out["dry"]["effluent"]["eqi_kg_pu_per_day"])

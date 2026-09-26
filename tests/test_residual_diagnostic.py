"""residual_diagnostic: rebuilds checkpoints on CPU and reproduces the v1.0 probe magnitude."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from scripts import residual_diagnostic as rd

REPO = Path(__file__).resolve().parents[1]


def test_v10_checkpoint_reproduces_the_probe_magnitude(tmp_path, monkeypatch):
    """v1.0 cl_pinn sigma 0.10 seed 0: about 0.57 in-window against 8-9 beyond (judges' probe)."""
    monkeypatch.chdir(REPO)
    cell = tmp_path / "pinn" / "k000_ie_a0_seed0"
    for name in ("cl_pinn_sigma0p10", "lstm_sigma0p10"):
        shutil.copytree(REPO / "results" / "runs" / name, cell / name)
    result = rd.diagnose(tmp_path / "pinn", n=4096, seed=0)
    assert len(result["runs"]) == 1 and len(result["skipped_non_pinn"]) == 1
    run = result["runs"][0]
    assert run["cell"] == "k000_ie_a0" and run["model"] == "cl_pinn"
    assert 0.45 <= run["mse_R0"] <= 0.90
    assert 6.0 <= run["mse_F"] <= 12.0
    again = rd.diagnose(tmp_path / "pinn", n=4096, seed=0)
    assert again["runs"][0]["mse_R0"] == run["mse_R0"]


def test_v11_checkpoint_uses_the_filtered_ras_and_learned_multipliers(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    from src.train.curriculum import trailing_average
    from src.train.run import RunConfig, Trainer

    out = tmp_path / "pinn" / "k000_ie_a0_seed0"
    cfg = RunConfig(run_id="cl_pinn_sigma0p10", model="cl_pinn", noise=0.10, profile="quick", device="cpu",
                    dtype="float64", out_dir=str(out), total_derivative=True, ras_filter_window=4)
    trainer = Trainer(cfg)
    trainer.finalise(0.0)
    _, _, _, ras = rd.input_series(trainer)
    raw = trainer.data["dry"].obs[:, trainer.ras_col]
    np.testing.assert_allclose(ras, trailing_average(raw, 4))  # final hierarchical stage smooths with window 1
    first = rd.diagnose(tmp_path / "pinn", n=256, seed=0)["runs"][0]
    assert np.isfinite(first["mse_R0"]) and np.isfinite(first["mse_F"])
    summary_path = out / "cl_pinn_sigma0p10" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["learned_multipliers"] = {"muA": 1.0, "bA": 1.0}
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    second = rd.diagnose(tmp_path / "pinn", n=256, seed=0)["runs"][0]
    assert second["mse_R0"] == pytest.approx(first["mse_R0"], rel=1e-12)

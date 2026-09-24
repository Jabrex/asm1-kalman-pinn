"""The derivative audit reproduces the review-panel probe on a v1.0 checkpoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
CHECKPOINT = REPO / "results" / "runs" / "cl_pinn_sigma0p10" / "checkpoint.pt"


@pytest.fixture(scope="module")
def trainer():
    if not CHECKPOINT.exists() or not (REPO / "results" / "raw" / "obs_dry_sigma0p10.npz").exists():
        pytest.skip("needs the v1.0 checkpoint results/runs/cl_pinn_sigma0p10 and results/raw")
    from scripts.audit_derivative import load_trainer

    return load_trainer(CHECKPOINT)


def test_segment_slopes_match_the_interpolant(trainer):
    from scripts.audit_derivative import sample_inputs

    inp = sample_inputs(trainer, (0.0, 12.0), "random_segment", seed=0)
    dry = trainer.data["dry"]
    grid = np.asarray(dry.t)
    idx = np.clip(np.searchsorted(grid, inp["t"], "right") - 1, 0, len(grid) - 2)
    np.testing.assert_allclose(
        inp["dq"], (dry.q_in[idx + 1] - dry.q_in[idx]) / (grid[idx + 1] - grid[idx]), rtol=1e-12
    )
    assert inp["t"].min() >= 0.0 and inp["t"].max() <= 12.0 and len(inp["t"]) == 4096


def test_knots_statistic_matches_the_panel_probe(trainer):
    from scripts.audit_derivative import audit_one, sample_inputs

    stats = audit_one(trainer, sample_inputs(trainer, (0.0, 12.0), "knots_central", seed=0))
    assert abs(stats["ratio_total_over_partial"] - 1.7) <= 0.2 * 1.7
    assert stats["total_mse"] > stats["partial_mse"] > 0.0
    assert 0.0 < stats["influent_share_scaled_median"] < 1.0

"""RUNBOOK step 6 gates on the derivative path: 6a' must pass the total and fail the partial."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def probe_trainer():
    if not (REPO / "results" / "raw" / "obs_dry_sigma0p05.npz").exists():
        pytest.skip("needs results/raw/obs_dry_sigma0p05.npz - run scripts.generate_data first")
    from scripts.verify_model import _trainer

    return _trainer()


def test_gate_6a_total_passes_and_discriminates(probe_trainer):
    from scripts.verify_model import TOL_AUTOGRAD, gate_6a_total

    ok, detail = gate_6a_total(probe_trainer)
    assert ok, detail
    assert detail["max_per_output_relative_error_total"] < TOL_AUTOGRAD
    # the v1.0 partial derivative is far from the trajectory derivative
    assert detail["max_per_output_relative_error_partial_vs_trajectory"] > 1e-2


def test_gate_6b_compares_total_derivatives(probe_trainer):
    from scripts.verify_model import gate_6b

    ok, detail = gate_6b(probe_trainer)
    assert ok, detail
    assert "max_relative_difference_total" in detail

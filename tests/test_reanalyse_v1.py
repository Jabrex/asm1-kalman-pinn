"""The v1.0 re-analysis reproduces the published holdout numbers on the F window."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def test_reanalysis_reproduces_v10_holdout_rows(tmp_path):
    needed = [REPO / "results" / "runs" / name / "predictions.npz"
              for name in ("cl_pinn_sigma0p10", "persistence_sigma0p00", "odesim_sigma0p10")]
    if not all(p.exists() for p in needed):
        pytest.skip("needs the v1.0 run directories under results/runs")
    from scripts.reanalyse_v1 import main

    out = tmp_path / "re.json"
    assert main(["--dirs", "runs", "--out", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["label"] == "exploratory, pre-fix checkpoints"
    rows = {(r["run_id"], r["window"]): r for r in report["rows"]}
    # v1.0 benchmark table, holdout Track B NRMSE (own-window range)
    assert rows[("cl_pinn_sigma0p10", "F")]["track_b_nrmse"] == pytest.approx(0.310, abs=0.002)
    assert rows[("persistence_sigma0p10", "F")]["track_b_nrmse"] == pytest.approx(0.372, abs=0.002)
    assert rows[("odesim_sigma0p10", "F")]["track_b_nrmse"] < 0.01
    assert rows[("persistence_sigma0p10", "R0")]["skill_vs_persistence"] == pytest.approx(0.0)
    assert rows[("cl_pinn_sigma0p10", "R2")]["t_start"] >= 2.0 - 1e-9
    memory = {m["run_id"]: m for m in report["error_vs_time"]}
    assert len(memory["cl_pinn_sigma0p10"]["nrmse_fixed_X_I"]) == 14

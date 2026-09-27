"""Truth-free q scan at the reference cell K0-Ie-A0, sigma 0.10, on an extended grid.

Same inputs as src.observers.pipeline.run_estimators (window 4, data-based R, days 0-12).
Only measured-channel scores are computed: innovation log-likelihood, NIS and the
6-h predictive score. No Track B value is computed for the new grid points.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.data.influent_views import view_dataset
from src.data.sensors import ObservationDataset
from src.observers.ekf import EkfConfig, estimate_r_from_data, ras_log_variance, tune_q
from src.observers.pipeline import resolve_channels
from src.observers.reduced_model import ReducedPlantModel
from src.train.curriculum import trailing_average
from src.train.run import RAS_CHANNEL

torch.set_num_threads(4)
GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
OUT = Path("results/v11/predating/q_scan_extended.json")

dry = view_dataset(ObservationDataset.load(Path("results/raw/obs_dry_sigma0p10.npz")), "exact")
anchor = np.load("results/v11/anchors/k000/A0.npz")
channels = resolve_channels(None)
names = list(dry.channels)
cols, ras_col, w = [names.index(c.name) for c in channels], names.index(RAS_CHANNEL), 4
tr = dry.t <= 12.0 + 1e-9
y, raw = dry.obs[:, cols], dry.obs[:, ras_col]
ras_f = trailing_average(raw, w)
r = estimate_r_from_data(y[tr], 0.01)
ras_var = ras_log_variance(raw[tr], w, 0.01)
base = EkfConfig(substeps=3, q_theta=1e-3, r_mode="data", r_floor=0.01, ras_filter_window=w)
model = ReducedPlantModel()
results = {}
for crit in ("innovation", "predictive"):
    sel = tune_q(model, dry.t[tr], y[tr], channels, dry.q_in[tr], dry.z_in[tr], ras_f[tr],
                 anchor["z0_mean"], anchor["z0_rel_std"], base, ras_log_var=ras_var, r=r,
                 grid=GRID, criterion=crit)
    results[crit] = {"q_soluble": sel.q_soluble, "q_particulate": sel.q_particulate,
                     "on_grid_edge": sel.on_grid_edge}
    table = sel.table
    print(crit, results[crit], flush=True)
    if crit == "innovation":
        results["table"] = table
        OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")
        best = min((row for row in table if not row["diverged"]), key=lambda row: row["predictive_score"])
        results["predictive"] = {"q_soluble": best["q_soluble"], "q_particulate": best["q_particulate"],
                                 "on_grid_edge": best["q_soluble"] in (GRID[0], GRID[-1])
                                 or best["q_particulate"] in (GRID[0], GRID[-1])}
        print("predictive", results["predictive"], flush=True)
        break
OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")

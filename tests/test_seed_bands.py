"""seed_bands: v1.0 bands reproduce; generic roots and data directory work."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from scripts import seed_bands

REPO = Path(__file__).resolve().parents[1]


def test_defaults_reproduce_the_v10_seed_bands(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    seed_bands.main(["--out-dir", str(tmp_path), "--no-figures"])
    produced = json.loads((tmp_path / "seed_bands.json").read_text(encoding="utf-8"))
    reference = json.loads((REPO / "results" / "seed_bands.json").read_text(encoding="utf-8"))
    assert produced == reference


def test_generic_roots_band_a_single_model(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    roots = []
    for k, src in enumerate(("results/runs", "results/runs_seed1")):
        root = tmp_path / ("cell_seed%d" % k)
        shutil.copytree(REPO / src / "cl_pinn_sigma0p10", root / "cl_pinn_sigma0p10")
        roots.append(str(root))
    out = tmp_path / "out"
    seed_bands.main(["--runs", *roots, "--data-dir", "results/raw", "--band-models", "cl_pinn",
                     "--context-models", "--out-dir", str(out), "--tag", "_cell"])
    bands = json.loads((out / "seed_bands_cell.json").read_text(encoding="utf-8"))
    assert bands["cl_pinn|0.10|holdout"]["track_b_nrmse"]["n"] == 2
    assert (out / "figures" / "noise_robustness_bands_cell.png").exists()
    assert (out / "figures" / "noise_robustness_bands_cell.pdf").exists()


def test_variants_get_their_own_band():
    """A sensor variant of cl_pinn must not be pooled into the cl_pinn band."""
    base = {m: 0.5 for m in seed_bands.METRICS}
    rows = [
        {**base, "model": "cl_pinn", "noise": 0.1, "eval_set": "train", "variant": ""},
        {**base, "model": "cl_pinn", "noise": 0.1, "eval_set": "train", "variant": "drop-TSS_tank5",
         "track_b_nrmse": 0.9},
    ]
    bands = seed_bands.band_table(rows, ["cl_pinn"])
    assert sorted(bands) == ["cl_pinn[drop-TSS_tank5]|0.10|train", "cl_pinn|0.10|train"]
    assert bands["cl_pinn|0.10|train"]["track_b_nrmse"]["n"] == 1
    assert bands["cl_pinn[drop-TSS_tank5]|0.10|train"]["track_b_nrmse"]["max"] == 0.9


def test_figure_window_and_metric_are_selectable(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    root = tmp_path / "cell_seed0"
    shutil.copytree(REPO / "results/runs/cl_pinn_sigma0p10", root / "cl_pinn_sigma0p10")
    out = tmp_path / "out"
    seed_bands.main(["--runs", str(root), "--band-models", "cl_pinn", "--context-models",
                     "--figure-set", "train", "--figure-metric", "track_b_nrmse_fixed",
                     "--out-dir", str(out), "--tag", "_r0"])
    assert (out / "figures" / "noise_robustness_bands_r0.png").exists()

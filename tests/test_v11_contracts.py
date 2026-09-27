"""G6 entry gate: what groups G1-G5 promised exists before any v1.1 run starts.

Each failure names the group that owns the missing piece. Checks that need
generated files skip unless ASM1_STRICT_TESTS=1, in which case they fail.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
STRICT = os.environ.get("ASM1_STRICT_TESTS", "").strip().lower() not in {"", "0", "false", "no"}
KINETIC_NAMES = ("kh", "KX", "etah", "muH", "etag", "Ks", "bH", "KO_H", "KNO", "KNH_H",
                 "muA", "bA", "ka", "KO_A", "KNH")
BASE_CONSTANTS = ("lr", "lr_final_fraction", "grad_clip", "collocation_points", "steps_quick",
                  "log_every", "dtype", "train_end_day", "holdout_days", "pinn", "lstm")
ALL_SIGMAS = ("0p00", "0p05", "0p10", "0p15")


def _need(path: Path) -> Path:
    if path.exists():
        return path
    if STRICT:
        pytest.fail("ASM1_STRICT_TESTS is set and %s is missing" % path)
    pytest.skip("%s not generated yet" % path)
    raise AssertionError("unreachable")


def _help(module: str) -> str:
    out = subprocess.run([sys.executable, "-m", module, "--help"], cwd=REPO,
                         capture_output=True, text=True)
    assert out.returncode == 0, "%s --help failed:\n%s" % (module, out.stderr)
    return out.stdout


def _realistic() -> str:
    gate = REPO / "results" / "v11" / "gate_d3.json"
    if gate.exists():
        return json.loads(gate.read_text(encoding="utf-8"))["realistic_k"]
    return "k100"


def test_g1_run_all_flags_and_run_ids():
    from scripts import run_all

    source = inspect.getsource(run_all)
    for flag in ("--seed", "--out-dir", "--data-dir", "--list", "--resume", "--models", "--noise"):
        assert flag in source, "G1: scripts/run_all.py lacks %s" % flag
    assert run_all.run_id("cl_pinn", 0.1) == "cl_pinn_sigma0p10"
    assert run_all.run_id("cl_pinn", 0.1, "drop-S_NO_tank2") == "cl_pinn_sigma0p10_drop-S_NO_tank2"


def test_g1_g5_run_config_fields():
    from src.train.run import MODEL_SPECS, RunConfig

    names = {f.name for f in dataclasses.fields(RunConfig)}
    required = {"total_derivative", "ras_filter_window", "variant", "anchor_file", "influent_mode",
                "target_channels", "trainable_kinetics", "kinetic_prior_sigma",
                "kinetic_prior_weight", "kinetic_bound"}
    assert required <= names, "G1/G5: RunConfig lacks %s" % sorted(required - names)
    assert MODEL_SPECS["cl_pinn_theta"] == {"arch": "pinn", "curriculum": "hierarchical"}


def test_g1_trailing_average_is_causal():
    from src.train.curriculum import trailing_average

    y = trailing_average(np.arange(10.0), 4)
    np.testing.assert_allclose(y[:4], [0.0, 0.5, 1.0, 1.5])
    assert y[9] == pytest.approx(7.5)


def test_g1_make_baselines_cli():
    text = _help("scripts.make_baselines")
    for flag in ("--data-dir", "--out", "--anchor-file", "--rows", "--ras-filter-window", "--influent-mode"):
        assert flag in text, "G1: make_baselines lacks %s" % flag


def test_g1_metrics():
    from src.eval import metrics

    for name in ("skill_score", "gap_closed", "level_error", "within_tolerance_fraction", "error_vs_time"):
        assert hasattr(metrics, name), "G1: src/eval/metrics.py lacks %s" % name


def test_g2_modules():
    from src.asm1 import truth_plants
    from src.data import influent_views
    from src.data.sensors import CANDIDATE_CHANNELS, SensorChannel

    for name in ("truth_vault", "override_vault", "perturbed_vault", "BSM1_15C"):
        assert hasattr(truth_plants, name), "G2: truth_plants lacks %s" % name
    for name in ("apply_influent_knowledge", "view_dataset"):
        assert hasattr(influent_views, name), "G2: influent_views lacks %s" % name
    assert CANDIDATE_CHANNELS, "G2: CANDIDATE_CHANNELS is empty"
    assert {"weights", "sigma_override"} <= {f.name for f in dataclasses.fields(SensorChannel)}


@pytest.mark.parametrize("directory, files", [
    ("results/raw", ["obs_dry_sigma%s" % s for s in ALL_SIGMAS]
     + ["obs_constant_sigma0p10", "obs_dry_sigma0p10_r01", "obs_dry_sigma0p10_r09"]),
    ("results/raw_k025", ["obs_dry_sigma0p10"]),
    ("results/raw_k050", ["obs_%s_sigma%s" % (sc, s) for sc in ("dry", "constant") for s in ALL_SIGMAS]),
    ("results/raw_k075", ["obs_dry_sigma0p10"]),
    ("results/raw_k100", ["obs_%s_sigma%s" % (sc, s) for sc in ("dry", "constant") for s in ALL_SIGMAS]
     + ["obs_dry_sigma0p10_r01", "obs_dry_sigma0p10_r09"]),
    ("results/raw_k000_off", ["obs_dry_sigma0p10"]),
])
def test_g2_data_files(directory, files):
    root = _need(REPO / directory)
    missing = [f for f in files if not (root / (f + ".npz")).exists()]
    assert not missing, "G2: %s lacks %s" % (directory, missing)


def test_g2_manifests():
    for directory, preset in (("results/raw_k050", "graded"), ("results/raw_k100", "bsm1_15c")):
        manifest = json.loads(_need(REPO / directory / "manifest.json").read_text(encoding="utf-8"))
        assert manifest.get("truth_preset") == preset, "G2: %s truth_preset" % directory
        assert manifest.get("constant_from") == "nominal", "G2: %s constant_from" % directory


def test_g2_random_truths():
    root = _need(REPO / "results" / "raw_rand")
    dirs = [p for p in root.iterdir() if p.is_dir()]
    assert len(dirs) == 50, "G2: expected 50 random truths, found %d" % len(dirs)
    assert all((d / "obs_dry_sigma0p10.npz").exists() for d in dirs)


def test_g3_observer_api():
    from src.models.losses import KineticAdapter
    from src.observers import anchors, ekf, reduced_model

    for name in ("EkfConfig", "run_ekf", "rts_smooth", "ieks", "forecast", "estimate_r_from_data", "tune_q"):
        assert hasattr(ekf, name), "G3: src/observers/ekf.py lacks %s" % name
    for name in ("lab_operators", "gaussian_log_update", "ensemble_log_std", "ic_weights_from_rel_std"):
        assert hasattr(anchors, name), "G3: src/observers/anchors.py lacks %s" % name
    assert hasattr(reduced_model, "ReducedPlantModel")
    fields = {f.name for f in dataclasses.fields(ekf.EkfConfig)}
    assert {"q_soluble", "q_particulate", "r_floor", "augment", "theta_prior_sd", "q_theta"} <= fields
    assert callable(KineticAdapter)


def test_g3_scripts_cli():
    text = _help("scripts.make_anchors")
    for flag in ("--build-ensemble", "--n", "--sigma-log", "--data-dir", "--anchors", "--out"):
        assert flag in text, "G3: make_anchors lacks %s" % flag
    assert "--config" in _help("scripts.run_observers"), "G3: run_observers lacks --config"


def test_g3_leakage_scan_covers_observers():
    text = (REPO / "tests" / "test_leakage.py").read_text(encoding="utf-8")
    assert "src/observers" in text, "G3: tests/test_leakage.py must scan src/observers"


def test_g4_selections():
    from src.data.sensors import CANDIDATE_CHANNELS, SENSOR_SET

    names = json.loads(_need(REPO / "results/v11/analysis/kinetic_subset.json")
                       .read_text(encoding="utf-8"))["names"]
    assert len(names) == 4 and set(names) <= set(KINETIC_NAMES), "G4: kinetic_subset %s" % names
    choice = json.loads(_need(REPO / "results/v11/analysis/sensor_confirmation.json")
                        .read_text(encoding="utf-8"))
    assert choice["drop"] in {c.name for c in SENSOR_SET if c.kind != "tss_underflow"}
    assert choice["add"] in {c.name for c in CANDIDATE_CHANNELS}


def _regime_expectations(realistic: str) -> dict[str, tuple]:
    a100 = "results/v11/anchors/k100/A0.npz"
    rows = {
        "k000_ie_a0": ("results/raw", None, "exact", ["cl_pinn", "pinn"]),
        "k000_ic_a0": ("results/raw", None, "composite", ["cl_pinn"]),
        "k050_ie_a0": ("results/raw_k050", "results/v11/anchors/k050/A0.npz", "exact", ["cl_pinn"]),
        "k100_ie_a0": ("results/raw_k100", a100, "exact", ["cl_pinn"]),
        "k100_ie_a0_theta": ("results/raw_k100", a100, "exact", ["cl_pinn_theta"]),
        "k000_ie_a0_theta": ("results/raw", None, "exact", ["cl_pinn_theta"]),
        "k100_ie_a0_sensors": ("results/raw_k100", a100, "exact", ["cl_pinn"]),
    }
    raw_k = {"k100": "results/raw_k100", "k050": "results/raw_k050"}[realistic]
    for anchor, name in (("a0", "A0"), ("as", "As"), ("al1", "Al1")):
        rows["%s_ic_%s" % (realistic, anchor)] = (
            raw_k, "results/v11/anchors/%s/%s.npz" % (realistic, name), "composite", ["cl_pinn"])
    rows["%s_ic_as_lstm" % realistic] = (raw_k, "results/v11/anchors/%s/As.npz" % realistic,
                                         "composite", ["lstm"])
    return rows


def _cell_of(stem: str) -> str:
    for suffix in ("_theta", "_sensors", "_lstm"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


@pytest.mark.parametrize("stem", sorted(_regime_expectations(_realistic())))
def test_g5_regime_configs(stem):
    path = REPO / "configs" / "regime" / (stem + ".yaml")
    assert path.exists(), "G5: %s missing" % path.relative_to(REPO)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    base = yaml.safe_load((REPO / "configs" / "base.yaml").read_text(encoding="utf-8"))
    data_dir, anchor, influent, models = _regime_expectations(_realistic())[stem]
    assert raw["data_dir"] == data_dir
    if anchor is None:
        assert raw.get("anchor_file") in (None, "results/v11/anchors/k000/A0.npz")
    else:
        assert raw.get("anchor_file") == anchor
    assert raw.get("influent_mode", "exact") == influent
    assert raw["models"] == models
    assert raw["out_dir"] == "results/v11/pinn/%s" % _cell_of(stem)
    assert raw["total_derivative"] is True and raw["ras_filter_window"] == 4
    assert raw["profile"] == "full" and raw["steps_full"] == 20000
    for key in BASE_CONSTANTS:
        assert raw[key] == base[key], "G5: %s changes the v1.0 constant %s" % (stem, key)


def test_g5_noise_levels_of_realistic_as_cell():
    raw = yaml.safe_load((REPO / "configs" / "regime" / ("%s_ic_as.yaml" % _realistic()))
                         .read_text(encoding="utf-8"))
    assert {0.05, 0.1, 0.15} <= {round(float(s), 2) for s in raw["noise_levels"]}


def test_g5_theta_and_sensor_configs_follow_g4():
    names = json.loads(_need(REPO / "results/v11/analysis/kinetic_subset.json")
                       .read_text(encoding="utf-8"))["names"]
    for stem in ("k100_ie_a0_theta", "k000_ie_a0_theta"):
        raw = yaml.safe_load((REPO / "configs" / "regime" / (stem + ".yaml")).read_text(encoding="utf-8"))
        assert raw["trainable_kinetics"] == names, "G5: %s trainable_kinetics" % stem
    choice = json.loads(_need(REPO / "results/v11/analysis/sensor_confirmation.json")
                        .read_text(encoding="utf-8"))
    raw = yaml.safe_load((REPO / "configs" / "regime" / "k100_ie_a0_sensors.yaml").read_text(encoding="utf-8"))
    assert [v["suffix"] for v in raw["variants"]] == ["drop-%s" % choice["drop"], "add-%s" % choice["add"]]

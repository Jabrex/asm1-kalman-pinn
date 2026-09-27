"""Observer cell schema, channel sets, RAS input and run-directory naming."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from src.data.sensors import SENSOR_SET
from src.observers import cell_inputs as ci

REPO = Path(__file__).resolve().parents[1]
NAMES = tuple(c.name for c in SENSOR_SET)
STRICT = os.environ.get("ASM1_STRICT_TESTS", "").strip().lower() not in {"", "0", "false", "no"}


def _cfg(**over):
    base = {"cell": "k100_ic_as", "out_dir": "results/v11/observers/k100_ic_as",
            "data_dir": "results/raw_k100", "anchor_file": "results/v11/anchors/k100/As.npz",
            "influent_mode": "composite", "ras_filter_window": 4, "sigmas": [0.1],
            "realisations": [0], "estimators": ["ekf", "eks"], "q_mode": ["tuned"]}
    base.update(over)
    return base


def test_validate_fills_defaults():
    cfg = ci.validate_cell_config(_cfg())
    assert cfg["ras_input"] == "filtered" and cfg["channels"] == "default"
    assert cfg["q_grid"] == [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0] and cfg["r_floor"] == 0.01
    assert cfg["theta_prior_sd"] == 0.693 and cfg["frozen_q_from"] is None


@pytest.mark.parametrize("over, message", [
    ({"q_grd": [0.1]}, "unknown keys"),
    ({"q_mode": ["frozen"]}, "frozen_q_from"),
    ({"estimators": ["eks_aug"]}, "augment"),
    ({"out_dir": "results/v11/observers/other"}, "out_dir"),
    ({"estimators": ["kalman"]}, "estimators"),
    ({"influent_mode": "guess"}, "influent_mode"),
    ({"ras_input": "sensorless"}, "ras_input"),
])
def test_validate_rejects(over, message):
    with pytest.raises(ValueError, match=message):
        ci.validate_cell_config(_cfg(**over))


def test_default_channels_are_the_seven_targets():
    chans = ci.resolve_channels("default", NAMES)
    assert [c.name for c in chans] == [c.name for c in SENSOR_SET if c.kind != "tss_underflow"]
    assert ci.DEFAULT_TARGETS == tuple(c.name for c in chans)


@pytest.mark.parametrize("spec", [["TSS_ras"], ["S_NH_tank9"], [], ["S_O_tank3", "S_O_tank3"]])
def test_channels_refused(spec):
    with pytest.raises(ValueError):
        ci.resolve_channels(spec, NAMES)


def test_ras_input_filtered_is_trailing_average():
    obs = np.ones((10, len(NAMES)))
    obs[:, NAMES.index("TSS_ras")] = np.arange(10.0)
    out = ci.ras_input(obs, NAMES, np.full(10, 18446.0), "filtered", 4)
    np.testing.assert_allclose(out[:4], [0.0, 0.5, 1.0, 1.5])
    assert out[9] == pytest.approx(7.5)


def test_ras_input_ideal_settler_is_the_underflow_mass_balance():
    obs = np.ones((5, len(NAMES)))
    obs[:, NAMES.index("TSS_tank5")] = 3000.0
    q_in = np.full(5, 18446.0)
    out = ci.ras_input(obs, NAMES, q_in, "ideal_settler", 4)
    np.testing.assert_allclose(out, 3000.0 * (18446.0 + 18446.0) / (18446.0 + 385.0))


def test_run_dir_names_follow_the_g3_rule():
    assert ci.run_dir_name("eks", "tuned", 0.1) == "eks_sigma0p10"
    assert ci.run_dir_name("eks", "frozen", 0.1, 3) == "eks_frozenq_sigma0p10_r03"
    cfg = ci.validate_cell_config(_cfg(sigmas=[0.0, 0.1], realisations=[0, 1]))
    dirs = [p.as_posix() for p in ci.expected_run_dirs(cfg)]
    assert len(dirs) == 8 and "results/v11/observers/k100_ic_as/ekf_sigma0p00_r01" in dirs


def test_obs_path():
    assert ci.obs_path("results/raw", 0.1).as_posix() == "results/raw/obs_dry_sigma0p10.npz"
    assert ci.obs_path("results/raw", 0.1, 2).as_posix() == "results/raw/obs_dry_sigma0p10_r02.npz"


def test_numerics_summary():
    assert ci.numerics_summary(np.array([7.0, 5.0]), np.ones((3, 5, 14)), 7) == {
        "finite": True, "nis_mean": 6.0, "n_measured": 7}
    bad = ci.numerics_summary(np.array([np.nan]), np.ones((3, 5, 14)), 7)
    assert bad["finite"] is False and bad["nis_mean"] == float("inf")


def test_prepare_cell_inputs_on_v10_data():
    from src.data.sensors import ObservationDataset
    from src.train.curriculum import trailing_average

    anchor = REPO / "results" / "v11" / "anchors" / "k000" / "A0.npz"
    if not anchor.exists():
        if STRICT:
            pytest.fail("ASM1_STRICT_TESTS is set and %s is missing" % anchor)
        pytest.skip("anchor files not built yet")
    cfg = ci.validate_cell_config(_cfg(cell="k000_ie_a0", out_dir="results/v11/observers/k000_ie_a0",
                                       data_dir="results/raw", anchor_file=anchor.relative_to(REPO).as_posix(),
                                       influent_mode="exact"))
    inputs = ci.prepare_cell_inputs(cfg, 0.10)
    ds = ObservationDataset.load(REPO / "results" / "raw" / "obs_dry_sigma0p10.npz")
    assert inputs.y_obs.shape == (1345, 7) and int(inputs.train.sum()) == 1153
    np.testing.assert_array_equal(inputs.z_in, ds.z_in)
    np.testing.assert_allclose(inputs.tss_ras, trailing_average(ds.obs[:, ds.channels.index("TSS_ras")], 4))
    assert inputs.z0_mean.shape == (5, 14) and inputs.z0_rel_std.shape == (5, 14)


def test_read_frozen_q_accepts_the_g3_and_the_g6_layout(tmp_path):
    """G3's --freeze-q writes the pair at top level; the G6 schema nests it under selected_q."""
    g3 = tmp_path / "g3.json"
    g3.write_text(json.dumps({"q_soluble": 0.3, "q_particulate": 0.1, "q_criterion": "innovation"}))
    g6 = tmp_path / "g6.json"
    g6.write_text(json.dumps({"selected_q": {"q_soluble": 0.3, "q_particulate": 0.1}}))
    assert ci.read_frozen_q(g3) == ci.read_frozen_q(g6) == {"q_soluble": 0.3, "q_particulate": 0.1}

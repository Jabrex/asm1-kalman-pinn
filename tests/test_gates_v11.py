"""Divergence rule, primary-smoother choice, gate D3 and the derived K.5 configs."""

from __future__ import annotations

import json

import pytest
import yaml

from scripts import gate_d3 as gd
from scripts import observer_numerics as on


def test_diverged_rule():
    assert on.diverged({"finite": True, "nis_mean": 21.0, "n_measured": 7}) is False
    assert on.diverged({"finite": True, "nis_mean": 21.5, "n_measured": 7}) is True
    assert on.diverged({"finite": False, "nis_mean": 1.0, "n_measured": 7}) is True
    assert on.diverged({"finite": True, "nis_mean": 1.0, "n_measured": 7, "diverged": True}) is True


def test_choose():
    assert on.choose(0.20, None) == "eks"
    assert on.choose(0.21, None) == "run_ieks"
    assert on.choose(0.50, 0.10) == "ieks"
    assert on.choose(0.50, 0.30) == "stop"


def test_divergence_rate(tmp_path):
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    cases = [(7.0, True), (30.0, True), (7.0, False), None]
    for i, case in enumerate(cases):
        out = tmp_path / "obs" / ("c%d" % i)
        (cfg_dir / ("c%d.yaml" % i)).write_text(
            yaml.safe_dump({"cell": "c%d" % i, "out_dir": out.as_posix(), "sigmas": [0.1]}), encoding="utf-8")
        if case is not None:
            run = out / "ekf_sigma0p10"
            run.mkdir(parents=True)
            (run / "summary.json").write_text(
                json.dumps({"finite": case[1], "nis_mean": case[0], "n_measured": 7}), encoding="utf-8")
    res = on.divergence(cfg_dir, "ekf")
    assert res["n_cells"] == 4 and res["n_diverged"] == 3 and res["n_missing"] == 1


def test_decide_d3():
    assert gd.decide(0.30, 0.31) == "k100"
    assert gd.decide(0.31, 0.31) == "k100"
    assert gd.decide(0.32, 0.31) == "k050"
    with pytest.raises(ValueError):
        gd.decide(float("nan"), 0.3)


def test_derive_config_rewrites_data_anchor_and_out_dir(tmp_path):
    src = tmp_path / "k100_ic_as.yaml"
    src.write_text(yaml.safe_dump({"models": ["cl_pinn"], "data_dir": "results/raw_k100",
                                   "out_dir": "results/v11/pinn/k100_ic_as",
                                   "anchor_file": "results/v11/anchors/k100/As.npz",
                                   "influent_mode": "composite"}), encoding="utf-8")
    dst = tmp_path / "k050_ic_as.yaml"
    raw = gd.derive_config(src, dst)
    assert raw["data_dir"] == "results/raw_k050"
    assert raw["out_dir"] == "results/v11/pinn/k050_ic_as"
    assert raw["anchor_file"] == "results/v11/anchors/k050/As.npz"
    assert raw["influent_mode"] == "composite" and raw["models"] == ["cl_pinn"]
    assert yaml.safe_load(dst.read_text(encoding="utf-8")) == raw

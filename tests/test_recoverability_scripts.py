"""Pure helpers of scripts/recoverability.py and scripts/recoverability_validation.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts import recoverability as rec
from scripts import recoverability_validation as val
from src.asm1.vault_loader import vault

REPO = Path(__file__).resolve().parents[1]


def test_tag_for():
    assert rec.tag_for("results/raw") == "k000"
    assert rec.tag_for("results/raw_k100") == "k100"
    assert rec.tag_for("results/raw_k000_off") == "k000_off"
    assert rec.tag_for("results/raw_rand/7") == "rand7"
    with pytest.raises(ValueError):
        rec.tag_for("results/runs")


def test_assay_family_reads_the_g3_lab_operators():
    from src.observers import anchors

    comps = vault().components
    found = sorted(rec.assay_family(w, comps) for w in anchors.lab_operators(vault()).values())
    expected = sorted([(f, t) for f in ("COD", "SCOD", "TKN", "STKN", "ALK") for t in (0, 4)]
                      + [("TSS", t) for t in range(5)])
    assert found == expected


def test_pick_sensor_confirmation_takes_the_largest_changes():
    report = {
        "tag": "k100", "influent_mode": "exact", "data_dir": "results/raw_k100", "sigma": 0.10,
        "prior": {"file": "results/v11/anchors/k100/As.npz"},
        "sensor_value": {
            "criterion": "J",
            "drop_one": [{"name": "S_O_tank3", "delta_J": 0.01}, {"name": "TSS_tank5", "delta_J": 0.02}],
            "add_one": [{"name": "SCOD_tank5", "sigma": 0.2, "delta_J": 0.03},
                        {"name": "TSS_tank1", "sigma": 0.1, "delta_J": 0.01}],
        },
    }
    choice = rec.pick_sensor_confirmation(report)
    assert (choice["drop"], choice["add"]) == ("TSS_tank5", "SCOD_tank5")
    with pytest.raises(ValueError):
        rec.pick_sensor_confirmation({**report, "tag": "k000"})


def test_clean_makes_json_safe():
    out = rec.clean({"a": np.array([1.0, np.inf]), "b": np.float64(np.nan), "c": np.bool_(True),
                     "d": (np.int64(3),)})
    assert out == {"a": [1.0, None], "b": None, "c": True, "d": [3]}


def test_cluster_bootstrap_spearman():
    idx = np.arange(55.0).reshape(5, 11)
    res = val.cluster_bootstrap_spearman(idx, idx ** 2, n_boot=200)
    assert res["rho"] == pytest.approx(1.0) and res["ci95"][0] > 0.999 and res["n_points"] == 55
    err = idx.copy()
    err[0, 0] = np.nan
    assert val.cluster_bootstrap_spearman(idx, err, n_boot=50)["n_points"] == 54
    assert val.cluster_bootstrap_spearman(idx, -idx, n_boot=50)["rho"] == pytest.approx(-1.0)
    rng = np.random.default_rng(5)
    x, y = rng.normal(size=(5, 11)), rng.normal(size=(5, 11))
    assert val.cluster_bootstrap_spearman(x, y, n_boot=100) == val.cluster_bootstrap_spearman(x, y, n_boot=100)


def test_validation_cells_config():
    spec = yaml.safe_load((REPO / "configs" / "analysis" / "recoverability_cells.yaml").read_text(encoding="utf-8"))
    names = [cell["name"] for cell in spec["cells"]]
    assert spec["primary"] in names and len(names) == len(set(names))
    for cell in spec["cells"]:
        assert set(cell) == {"name", "data_dir", "anchor", "influent_mode"}
        assert cell["influent_mode"] in ("exact", "composite")
    assert spec["estimators"]["eks"]["kind"] == "observer"
    assert spec["estimators"]["cl_pinn"]["kind"] == "pinn"

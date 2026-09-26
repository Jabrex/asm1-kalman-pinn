"""Grid sizes, schema validity, dependencies and uniqueness of the CPU estimator grid."""

from __future__ import annotations

import collections

import pytest

from scripts import observer_grid as og
from src.observers.cell_inputs import DEFAULT_TARGETS, expected_run_dirs, validate_cell_config

KIN = ["muA", "bA", "muH", "bH"]
CANDIDATES = ["S_NH_tank2", "SCOD_tank1"]


def _final(realistic="k100", smoother="eks"):
    return og.grid("grid", realistic=realistic, rands=["1", "2"], candidates=CANDIDATES,
                   augment=KIN, smoother=smoother)


def _runs(files, family):
    return sum(len(expected_run_dirs(c)) for k, c in files.items() if k.split("/")[1] == family)


def test_numerics_stage():
    files = og.grid("numerics")
    main = {k: v for k, v in files.items() if k.startswith("numerics/")}
    ieks = {k: v for k, v in files.items() if k.startswith("numerics_ieks/")}
    assert len(main) == 60 and len(ieks) == 60
    assert all(v["estimators"] == ["ekf", "eks"] and v["q_mode"] == ["tuned"] and v["sigmas"] == [0.1]
               for v in main.values())
    assert all(v["out_dir"].startswith("results/v11/_numerics/observers/") for v in files.values())
    assert all(validate_cell_config(v) == v for v in files.values())


def test_final_grid_families():
    files = _final()
    fam = collections.Counter(k.split("/")[1] for k in files)
    assert fam == {"main": 62, "realisations": 2, "m0prime": 3, "random": 2, "sigma_log": 4,
                   "lab": 18, "settler": 1, "sensors": 7 + 2 + len(CANDIDATES)}
    assert _runs(files, "main") == 758
    assert _runs(_final("k050"), "main") == 818
    assert _runs(files, "realisations") == 36 and _runs(files, "m0prime") == 30
    assert _runs(files, "random") == 4 and _runs(files, "sensors") == 22


def test_noise_and_claim_cells():
    files = _final()
    for cell in ("k100_ie_a0", "k100_ic_as", "k100_ic_al1"):
        assert files["grid/main/%s" % cell]["sigmas"] == [0.0, 0.05, 0.1, 0.15]
        assert "ieks" in files["grid/main/%s" % cell]["estimators"]
    assert files["grid/main/k050_ic_as"]["sigmas"] == [0.1]
    assert "ieks" not in files["grid/main/k025_ib_al2"]["estimators"]
    k050 = _final("k050")
    assert k050["grid/main/k050_ic_as"]["sigmas"] == [0.0, 0.05, 0.1, 0.15]
    assert "ieks" in _final(smoother="ieks")["grid/main/k025_ib_al2"]["estimators"]


def test_frozen_q_depends_only_on_k000_tuned():
    files = _final()
    assert files["grid/main/k000_ie_a0"]["q_mode"] == ["tuned"]
    assert files["grid/main/k000_ie_a0"]["sigmas"] == [0.1]
    assert files["grid/main/k000_ie_a0__frozen"]["q_mode"] == ["frozen"]
    frozen = [v for v in files.values() if "frozen" in v["q_mode"]]
    assert frozen and all(v["frozen_q_from"] == og.FROZEN_Q_SOURCE for v in frozen)


def test_run_dirs_unique():
    dirs = [p.as_posix() for c in _final().values() for p in expected_run_dirs(c)]
    assert len(dirs) == len(set(dirs))


def test_sensor_subsets():
    subsets = og.sensor_subsets(DEFAULT_TARGETS, CANDIDATES)
    assert len(subsets["drop-S_O_tank3"]) == 6
    assert all(not c.startswith("S_O_") for c in subsets["drop-DO"])
    assert subsets["drop-Nanalysers"] == ["S_O_tank3", "S_O_tank4", "S_O_tank5", "TSS_tank5"]
    assert subsets["add-SCOD_tank1"][-1] == "SCOD_tank1"


def test_settler_and_random_cells():
    files = _final()
    assert files["grid/settler/k100_ie_a0_ideal"]["ras_input"] == "ideal_settler"
    rand = files["grid/random/rand1_ie_a0"]
    assert rand["data_dir"] == "results/raw_rand/1"
    assert rand["anchor_file"] == "results/v11/anchors/rand/1/A0.npz"
    assert rand["estimators"] == ["eks", "eks_aug"] and rand["augment"] == KIN


def test_baseline_jobs_cover_each_cell_once():
    files = _final()
    jobs = og.baseline_jobs("grid", files)
    outs = [a[a.index("--out") + 1] for a in jobs]
    assert len(outs) == len(set(outs)) == 60 + 3 + 2 + 4
    # random truths have observations at the registered sigma only
    for argv in jobs:
        rand = "raw_rand" in argv[argv.index("--data-dir") + 1]
        assert (argv[-2:] == ["--sigmas", "0.10"]) == rand, argv
    with pytest.raises(ValueError):
        og.grid("grid", realistic="k075", augment=KIN)

"""generate_data v1.1 helpers: seed rule, M0-prime guard, v1.0 protection,
BSM1 parameter guard and the closure re-check."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import generate_data as gd
from scripts.generate_random_truths import MAX_CANDIDATES, as_multipliers, candidate_matrix
from src.asm1.plant import Bsm1Plant
from src.asm1.truth_plants import kinetic_names
from src.data.influent import dry_weather
from src.data.simulate import SAMPLE_INTERVAL_DAYS, default_seed, simulate

REPO = Path(__file__).resolve().parents[1]
BSM1_NAMES = ("YA", "YH", "fP", "iXB", "iXP", "muH", "Ks", "KO_H", "KNO", "bH",
              "etag", "etah", "kh", "KX", "muA", "KNH", "bA", "KO_A", "ka")


def test_noise_seed_reproduces_every_v1_seed():
    manifest = json.loads((REPO / "results" / "raw" / "manifest.json").read_text(encoding="utf-8"))
    for scenario in manifest["scenarios"].values():
        for entry in scenario["observations"].values():
            assert gd.noise_seed(0, entry["sigma"]) == entry["seed"]
    assert gd.noise_seed(0, 0.10, 3) == 302000
    assert gd.realisation_name("dry", 0.10, 3) == "obs_dry_sigma0p10_r03.npz"
    with pytest.raises(ValueError):
        gd.noise_seed(0, 0.20)


def test_offsteady_start_requires_whole_weeks():
    plant = Bsm1Plant()
    y = default_seed(plant)
    assert np.array_equal(gd.offsteady_start(plant, y, 0.0), y)
    with pytest.raises(ValueError):
        gd.offsteady_start(plant, y, 3.0)


def test_full_run_into_v1_raw_is_refused(capsys):
    # The attribute check fails on the v1.0 script before main() could touch results/raw.
    assert gd.V1_RAW == REPO / "results" / "raw"
    assert gd.main(["--out", str(gd.V1_RAW)]) == 2
    assert "REFUSED" in capsys.readouterr().out


def test_bsm1_presets_need_the_parameter_check(tmp_path, capsys):
    assert callable(gd.bsm1_parameter_problems)
    ref = {"parameter_check": {"table": None, "page": None, "checked_by": None,
                               "values": {name: None for name in BSM1_NAMES}}}
    path = tmp_path / "ref.json"
    path.write_text(json.dumps(ref), encoding="utf-8")
    for preset in (["--truth-preset", "bsm1_15c"], ["--truth-preset", "graded", "--alpha", "0.5"]):
        code = gd.main(["--out", str(tmp_path / "out"), "--bsm1-reference", str(path)] + preset)
        assert code == 2
        assert "REFUSED" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()


def test_closure_recheck_uses_a_finer_grid_only_when_needed():
    plant = Bsm1Plant()
    y0 = default_seed(plant)
    result = simulate(dry_weather(0.25), plant=plant, y0=y0.copy())
    coarse = gd.gated_closure(plant, dry_weather, 0.25, y0, result, gate=1.0)
    assert "refined" not in coarse and coarse["pass"] is True
    assert coarse["gate_grid_days"] == SAMPLE_INTERVAL_DAYS
    fine = gd.gated_closure(plant, dry_weather, 0.25, y0, result, gate=0.0)
    assert fine["gate_grid_days"] == SAMPLE_INTERVAL_DAYS / gd.CLOSURE_REFINE_FACTOR
    assert fine["refined"]["reactor_cod_closure"] < coarse["reactor_cod_closure"]
    assert fine["pass"] is False


def test_random_candidates_do_not_depend_on_the_ensemble_size():
    m = candidate_matrix(0.3, 0)
    assert m.shape == (MAX_CANDIDATES, 15)
    np.testing.assert_array_equal(m[:3], np.random.default_rng(0).normal(0.0, 0.3, size=(3, 15)))
    assert tuple(as_multipliers(m[0])) == kinetic_names()

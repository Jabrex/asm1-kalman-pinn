"""Truth-plant parameter sets: vault untouched, continuity exact, scope enforced."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from src.asm1.model import Asm1Kinetics
from src.asm1.plant import Bsm1Plant
from src.asm1.truth_plants import (
    BSM1_15C,
    LIN_INTERP,
    LOG_INTERP,
    effective_alpha,
    kinetic_names,
    override_vault,
    perturbed_vault,
    truth_vault,
)
from src.asm1.vault_loader import VAULT_JSON, _expected_json_sha256, _sha256, vault

REPO = Path(__file__).resolve().parents[1]
PRESET_CASES = [("vault20", 1.0), ("graded", 0.25), ("graded", 0.5), ("graded", 0.75), ("bsm1_15c", 1.0)]


def test_identity_preset_is_the_vault():
    v = vault()
    assert truth_vault("vault20") is v
    same = override_vault({})
    assert dict(same.parameters) == dict(v.parameters)
    assert np.array_equal(same.nu, v.nu)
    assert np.array_equal(same.composition, v.composition)


def test_symbolic_reevaluation_reproduces_vault_nu_exactly():
    """Verified 2026-09-23: max |nu_reevaluated - vault.nu| = 0.0."""
    assert float(np.max(np.abs(override_vault({}).nu - vault().nu))) == 0.0


def test_ixb_on_nu_alone_breaks_nitrogen_continuity():
    """Patching nu without the composition leaves a 6.0e-3 N defect (2026-09-23)."""
    patched_nu = override_vault({"iXB": 0.08}).nu
    defect = patched_nu @ vault().composition
    assert float(np.max(np.abs(defect[:, 1]))) == pytest.approx(6.0e-3, rel=1e-9)


def test_ixb_with_composition_restores_continuity():
    tv = override_vault({"iXB": 0.08})
    assert float(np.max(np.abs(tv.nu @ tv.composition))) <= 1e-15  # 4.7e-16 measured


@pytest.mark.parametrize("preset,alpha", PRESET_CASES)
def test_every_preset_conserves_cod_n_and_charge(preset, alpha):
    tv = truth_vault(preset, alpha)
    assert float(np.max(np.abs(tv.nu @ tv.composition))) <= 1e-12


def test_vault_file_and_cached_vault_untouched_by_every_preset():
    before = _sha256(VAULT_JSON)
    for preset, alpha in PRESET_CASES:
        truth_vault(preset, alpha)
    perturbed_vault({name: 0.3 for name in kinetic_names()})
    after = _sha256(VAULT_JSON)
    assert before == after == _expected_json_sha256() == vault().json_sha256
    assert vault().p("muH") == 6.0
    assert vault().p("iXB") == 0.086
    assert vault().p("KNH_H") == 0.05


def test_bsm1_set_differs_from_the_vault_only_in_its_ten_keys():
    assert set(BSM1_15C) == set(LOG_INTERP) | set(LIN_INTERP)
    tv = truth_vault("bsm1_15c")
    v = vault()
    for name, value in tv.parameters.items():
        expected = BSM1_15C.get(name, v.p(name))
        assert value == expected, name
    for name, value in {"YH": 0.67, "YA": 0.24, "fP": 0.08, "iXP": 0.06, "KO_H": 0.2,
                        "KNO": 0.5, "etag": 0.8, "kh": 3.0, "KO_A": 0.4, "KNH": 1.0}.items():
        assert tv.p(name) == value, name


def test_graded_interpolation_is_geometric_and_linear():
    mid = truth_vault("graded", 0.5)
    assert mid.p("muH") == pytest.approx(math.sqrt(6.0 * 4.0), rel=1e-12)
    assert mid.p("KX") == pytest.approx(math.sqrt(0.03 * 0.1), rel=1e-12)
    assert mid.p("KNH_H") == pytest.approx(0.025, rel=1e-12)
    assert mid.p("iXB") == pytest.approx(0.083, rel=1e-12)
    assert dict(truth_vault("graded", 0.0).parameters) == dict(vault().parameters)
    assert dict(truth_vault("graded", 1.0).parameters) == dict(truth_vault("bsm1_15c").parameters)


def test_alpha_and_preset_validation():
    assert effective_alpha("vault20", 1.0) == 0.0
    assert effective_alpha("bsm1_15c") == 1.0
    with pytest.raises(ValueError):
        truth_vault("graded", 1.5)
    with pytest.raises(ValueError):
        truth_vault("bsm1_15c", 0.5)
    with pytest.raises(ValueError):
        truth_vault("arrhenius", 0.5)


def test_knh_h_zero_makes_the_ammonium_switch_exactly_one(sample_state):
    kin = Asm1Kinetics(truth_vault("bsm1_15c"))
    i_nh = kin.vault.index("S_NH")
    rows = []
    for s_nh in (1e-12, 1e-6, 1.0, 30.0):
        z = sample_state.copy()
        z[i_nh] = s_nh
        rows.append(kin.rates(z)[:2])
    for row in rows[1:]:
        assert np.array_equal(row, rows[0])


def test_kinetic_names_are_the_fifteen_vault_kinetic_parameters():
    names = kinetic_names()
    assert len(names) == 15
    assert set(names) == set(vault().parameters) - {
        "YH", "fP", "YA", "iXB", "iXP", "iNO3_N2", "iCOD_NO3", "iCOD_N2",
        "iCharge_SNHx", "iCharge_SNOx",
    }


def test_perturbed_vault_scales_caps_eta_and_rejects_constants():
    tv = perturbed_vault({"muH": math.log(2.0), "etag": 1.0, "etah": -0.5})
    assert tv.p("muH") == pytest.approx(12.0, rel=1e-12)
    assert tv.p("etag") == 1.0                       # 0.8 * e > 1 is capped
    assert tv.p("etah") == pytest.approx(0.4 * math.exp(-0.5), rel=1e-12)
    assert float(np.max(np.abs(tv.nu @ tv.composition))) <= 1e-12
    with pytest.raises(KeyError):
        perturbed_vault({"iCOD_NO3": 0.1})
    with pytest.raises(KeyError):
        perturbed_vault({"not_a_parameter": 0.1})


def test_estimator_code_never_mentions_truth_plants():
    """Static scope rule: truth parameters must never reach an estimator."""
    offenders = []
    for root in ("src/train", "src/models", "src/observers"):
        base = REPO / root
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            if "truth_plants" in path.read_text(encoding="utf-8"):
                offenders.append(path.relative_to(REPO).as_posix())
    assert not offenders, "estimator modules mention truth_plants: %s" % offenders

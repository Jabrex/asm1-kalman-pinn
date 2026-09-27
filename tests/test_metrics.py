"""Evaluation metrics: EQI terms and limit-violation bookkeeping."""

from __future__ import annotations

import numpy as np

from src.eval.metrics import (
    EFFLUENT_LIMITS,
    EQI_WEIGHTS,
    _effluent_terms,
    effluent_quality_index,
)


def test_every_effluent_limit_has_a_matching_term(plant, sample_state):
    """Every key in EFFLUENT_LIMITS must name a series _effluent_terms computes."""
    effluent = np.tile(sample_state, (8, 1))
    terms = _effluent_terms(plant, effluent)
    missing = set(EFFLUENT_LIMITS) - set(terms)
    assert not missing, "limits without a computed series: %s" % sorted(missing)
    assert set(EQI_WEIGHTS) <= set(terms)


def test_effluent_quality_index_reports_all_limits(plant, sample_state):
    t = np.linspace(0.0, 7.0, 8)
    effluent = np.tile(sample_state, (t.size, 1))
    q_e = np.full(t.size, 18061.0)
    out = effluent_quality_index(plant, t, effluent, q_e)
    assert np.isfinite(out["eqi_kg_pu_per_day"])
    for name in EFFLUENT_LIMITS:
        assert "%s_pct_time_over_limit" % name in out
        assert "%s_crossings" % name in out


def test_per_tank_nrmse_shape_and_values():
    """Per-(tank, component) NRMSE: own-range by default, fixed range on request."""
    from src.eval.metrics import per_tank_nrmse

    truth = np.zeros((4, 5, 14))
    truth[:, :, 0] = np.arange(4.0)[:, None]
    pred = truth.copy()
    pred[:, 2, 0] += 0.3
    out = per_tank_nrmse(truth, pred)
    assert out.shape == (5, 14)
    assert np.isclose(out[2, 0], 0.1)
    assert np.isclose(out[0, 0], 0.0)
    assert np.isnan(out[0, 1])
    fixed = per_tank_nrmse(truth, pred, spread=np.full(14, 6.0))
    assert np.isclose(fixed[2, 0], 0.05)

def test_skill_score_and_gap_closed():
    from src.eval.metrics import gap_closed, skill_score

    assert skill_score(0.2, 0.4) == 0.5
    assert skill_score(0.4, 0.4) == 0.0
    assert skill_score(0.6, 0.4) < 0.0
    np.testing.assert_allclose(skill_score(np.array([0.1, 0.3]), np.array([0.2, 0.3])), [0.5, 0.0])
    assert np.isclose(gap_closed(0.25, 0.4, 0.1), 0.5)
    assert np.isclose(gap_closed(0.1, 0.4, 0.1), 1.0)
    assert np.isclose(gap_closed(0.4, 0.4, 0.1), 0.0)
    assert np.isnan(gap_closed(0.2, 0.3, 0.3))


def test_level_error_is_rmse_over_mean_level():
    from src.eval.metrics import level_error

    truth = np.full((6, 5, 14), 100.0)
    pred = truth.copy()
    pred[..., 4] += 10.0
    out = level_error(truth, pred)
    assert out.shape == (14,)
    assert np.isclose(out[4], 0.10)
    assert np.isclose(out[0], 0.0)
    zero = np.zeros((6, 5, 14))
    assert np.isnan(level_error(zero, zero + 1.0)[0])


def test_within_tolerance_uses_volume_weighted_tank_means(v):
    from src.eval.metrics import tank_mean, within_tolerance_fraction

    i_bh = v.index("X_B_H")
    truth = np.full((10, 5, 14), 1000.0)
    pred = truth.copy()
    pred[:, :2, i_bh] *= 1.5
    pred[:, 2:, i_bh] *= 1.0 - 0.5 * 2000.0 / 3999.0
    np.testing.assert_allclose(tank_mean(pred)[:, i_bh], 1000.0, rtol=1e-12)
    pred[5:, :, v.index("X_B_A")] *= 1.2
    frac = within_tolerance_fraction(truth, pred)
    assert frac == {"X_B_H": 1.0, "X_B_A": 0.5}
    assert within_tolerance_fraction(truth, pred, components=("X_B_A",), rel_tol=0.25) == {"X_B_A": 1.0}


def test_error_vs_time_bins():
    from src.eval.metrics import error_vs_time

    t = np.linspace(0.0, 3.0, 13)
    truth = np.zeros((13, 5, 14))
    truth[:, :, 0] = t[:, None]
    pred = truth.copy()
    pred[t >= 2.0, :, 0] += 0.3
    out = error_vs_time(truth, pred, t, bin_days=1.0)
    assert out["nrmse"].shape == (3, 14)
    np.testing.assert_allclose(out["t_start"], [0.0, 1.0, 2.0])
    assert list(out["n"]) == [4, 4, 5]
    np.testing.assert_allclose(out["nrmse"][:, 0], [0.0, 0.0, 0.1])

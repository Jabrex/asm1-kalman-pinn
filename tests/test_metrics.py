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
    truth[:, :, 0] = np.arange(4.0)[:, None]  # component 0 spans a range of 3 in every tank
    pred = truth.copy()
    pred[:, 2, 0] += 0.3  # constant offset in tank 3 only
    out = per_tank_nrmse(truth, pred)
    assert out.shape == (5, 14)
    assert np.isclose(out[2, 0], 0.1)  # 0.3 / 3
    assert np.isclose(out[0, 0], 0.0)
    assert np.isnan(out[0, 1])  # zero range -> undefined, not inf
    fixed = per_tank_nrmse(truth, pred, spread=np.full(14, 6.0))
    assert np.isclose(fixed[2, 0], 0.05)  # 0.3 / 6

"""Curriculum schedule: ramps, budget parity, and the no-leakage noise policy."""

from __future__ import annotations

import numpy as np
import pytest

from src.train.curriculum import (
    FINAL_WEIGHTS,
    INITIAL_WEIGHTS,
    build,
    hierarchical,
    no_curriculum,
    smooth_observations,
)


def test_budget_parity_between_curriculum_and_baseline():
    """Any CL-vs-no-CL difference must come from ordering, not from compute."""
    steps = 12345
    assert hierarchical(steps).total_steps == steps
    assert no_curriculum(steps).total_steps == steps


def test_physics_weight_is_never_zero():
    """This is a full PINN: the residual is in the loss at every single step."""
    for _, _, weights in hierarchical(2000).iterate():
        assert weights.physics > 0.0


def test_physics_weight_increases_monotonically():
    values = [w.physics for _, _, w in hierarchical(4000).iterate()]
    assert values[0] == pytest.approx(INITIAL_WEIGHTS.physics, rel=1e-6)
    assert values[-1] == pytest.approx(FINAL_WEIGHTS.physics, rel=1e-6)
    assert all(b >= a - 1e-12 for a, b in zip(values, values[1:]))


def test_horizon_grows_and_scenario_advances():
    stages = hierarchical(4000).stages
    assert [s.dataset for s in stages] == ["constant", "dry", "dry", "dry"]
    horizons = [s.horizon_days for s in stages]
    assert all(b >= a for a, b in zip(horizons, horizons[1:]))
    assert horizons[-1] == 12.0


def test_noise_is_not_a_curriculum_axis():
    """Early stages see a smoothed view of the SAME noisy signal, never clean data.

    Handing the CL runs noise-free observations would make the CL comparison a
    data-quality comparison, so smoothing is the only easing applied, and for
    noise-free runs even that is switched off.
    """
    noisy = hierarchical(1000, noisy=True).stages
    assert noisy[0].smoothing_window > 1
    assert noisy[-1].smoothing_window == 1

    clean = hierarchical(1000, noisy=False).stages
    assert all(s.smoothing_window == 1 for s in clean)


def test_baseline_uses_final_settings_throughout():
    stages = no_curriculum(1000).stages
    assert len(stages) == 1
    assert stages[0].weights_start == FINAL_WEIGHTS
    assert stages[0].weights_end == FINAL_WEIGHTS
    assert stages[0].smoothing_window == 1


def test_smoothing_preserves_length_and_is_a_noop_at_window_one():
    obs = np.random.default_rng(0).normal(size=(300, 8))
    np.testing.assert_array_equal(smooth_observations(obs, 1), obs)
    smoothed = smooth_observations(obs, 9)
    assert smoothed.shape == obs.shape
    assert np.std(smoothed) < np.std(obs)


def test_build_rejects_unknown_kinds():
    with pytest.raises(ValueError):
        build("annealing", 100, 12.0, True)


AXIS_KINDS = ("weights_only", "horizon_only", "scenario_only", "smoothing_only")


def test_axis_ablations_keep_budget_parity():
    """Each single-axis variant gets the identical total step budget."""
    steps = 12345
    for kind in AXIS_KINDS:
        assert build(kind, steps, 12.0, True).total_steps == steps


def test_weights_only_isolates_the_ramp():
    stages = build("weights_only", 1000, 12.0, True).stages
    assert len(stages) == 1
    assert stages[0].dataset == "dry"
    assert stages[0].horizon_days == 12.0
    assert stages[0].smoothing_window == 1
    assert stages[0].weights_start == INITIAL_WEIGHTS
    assert stages[0].weights_end == FINAL_WEIGHTS


def test_horizon_only_isolates_the_horizon():
    stages = build("horizon_only", 1000, 12.0, True).stages
    assert [s.dataset for s in stages] == ["dry"] * 4
    horizons = [s.horizon_days for s in stages]
    assert horizons == [1.0, 3.0, 7.0, 12.0]
    for s in stages:
        assert s.weights_start == FINAL_WEIGHTS and s.weights_end == FINAL_WEIGHTS
        assert s.smoothing_window == 1


def test_scenario_only_isolates_the_scenario_switch():
    stages = build("scenario_only", 1000, 12.0, True).stages
    assert [s.dataset for s in stages] == ["constant", "dry"]
    for s in stages:
        assert s.weights_start == FINAL_WEIGHTS and s.weights_end == FINAL_WEIGHTS
        assert s.smoothing_window == 1


def test_smoothing_only_isolates_the_smoothing_schedule():
    noisy = build("smoothing_only", 1000, 12.0, True).stages
    assert [s.dataset for s in noisy] == ["dry"] * 4
    assert [s.smoothing_window for s in noisy] == [9, 5, 3, 1]
    for s in noisy:
        assert s.horizon_days == 12.0
        assert s.weights_start == FINAL_WEIGHTS and s.weights_end == FINAL_WEIGHTS
    clean = build("smoothing_only", 1000, 12.0, False).stages
    assert all(s.smoothing_window == 1 for s in clean)


def test_axis_ablations_keep_physics_on():
    """Every ablation is still a full PINN: lambda_physics never reaches zero."""
    for kind in AXIS_KINDS:
        for _, _, weights in build(kind, 500, 12.0, True).iterate():
            assert weights.physics > 0.0

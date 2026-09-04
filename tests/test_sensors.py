"""Sensor set, observability split, and the noise model."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.sensors import (
    NOISE_LEVELS,
    SENSOR_SET,
    SensorModel,
    observed_components,
    unobserved_components,
)


def test_sensor_set_is_eight_realistic_channels():
    assert len(SENSOR_SET) == 8
    kinds = {c.kind for c in SENSOR_SET}
    assert kinds == {"state", "tss_reactor", "tss_underflow"}


def test_observability_split_covers_every_component(v):
    observed = set(observed_components())
    unobserved = set(unobserved_components())
    assert observed == {"S_O", "S_NH", "S_NO"}
    assert len(unobserved) == 11
    assert observed | unobserved == set(v.components)
    assert observed & unobserved == set()


def test_noise_levels_include_a_noise_free_reference():
    assert NOISE_LEVELS[0] == 0.0
    assert set(NOISE_LEVELS) == {0.0, 0.05, 0.10, 0.15}


def test_zero_sigma_is_an_exact_passthrough():
    model = SensorModel()
    clean = np.abs(np.random.default_rng(0).normal(size=(200, 8))) + 1.0
    noisy, clipped = model.add_noise(clean, 0.0, np.random.default_rng(0))
    np.testing.assert_array_equal(noisy, clean)
    assert clipped == 0.0


@pytest.mark.parametrize("sigma", [0.05, 0.10, 0.15])
def test_noise_is_multiplicative_and_unbiased(sigma):
    model = SensorModel()
    rng = np.random.default_rng(1234)
    clean = np.full((200000, 1), 100.0)  # far from zero, so clipping cannot bias it
    noisy, clipped = model.add_noise(clean, sigma, rng)
    assert clipped == 0.0
    assert np.mean(noisy) == pytest.approx(100.0, rel=2e-3)
    assert np.std(noisy) / 100.0 == pytest.approx(sigma, rel=2e-2)


def test_clipping_is_reported_when_it_happens():
    """A signal sitting near zero at high sigma must declare its clipped fraction."""
    model = SensorModel()
    clean = np.full((100000, 1), 1.0)
    _, clipped = model.add_noise(clean, 0.6, np.random.default_rng(7))
    assert clipped > 0.0

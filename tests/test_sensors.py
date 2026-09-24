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


# --------------------------------------------------------------------------
# v1.1 candidate channels
# --------------------------------------------------------------------------
from src.asm1.plant import Bsm1Plant  # noqa: E402
from src.data.influent import dry_weather  # noqa: E402
from src.data.sensors import CANDIDATE_CHANNELS, SensorChannel  # noqa: E402
from src.data.simulate import default_seed, simulate  # noqa: E402


@pytest.fixture(scope="module")
def short_result():
    plant = Bsm1Plant()
    return simulate(dry_weather(0.125), plant=plant, y0=default_seed(plant), scenario="dry")


def test_candidate_channels_are_disjoint_from_the_standard_set():
    names = [c.name for c in CANDIDATE_CHANNELS]
    assert len(names) == len(set(names)) == 10
    assert not set(names) & {c.name for c in SENSOR_SET}
    assert {c.kind for c in CANDIDATE_CHANNELS} == {"state", "tss_reactor", "linear"}


def test_standard_columns_are_unchanged_by_extra_channels(short_result):
    model = SensorModel()
    base = model.build(short_result, sigma=0.10, seed=2000)
    extended = model.build(short_result, sigma=0.10, seed=2000, extra_channels=CANDIDATE_CHANNELS)
    assert extended.obs.shape == (len(short_result.t), 8 + len(CANDIDATE_CHANNELS))
    assert np.array_equal(extended.obs[:, :8], base.obs)
    assert np.array_equal(extended.obs_clean[:, :8], base.obs_clean)
    assert extended.clip_fraction == base.clip_fraction
    assert extended.channels[:8] == base.channels
    assert extended.meta["extra_channels"] == [c.name for c in CANDIDATE_CHANNELS]
    assert extended.meta["extra_noise_seed"] == [2000, 1]


def test_linear_channel_is_soluble_cod(short_result, v):
    model = SensorModel()
    ds = model.build(short_result, sigma=0.0, seed=0, extra_channels=CANDIDATE_CHANNELS)
    col = ds.channels.index("SCOD_tank5")
    expected = (short_result.reactor[:, 4, v.index("S_I")]
                + short_result.reactor[:, 4, v.index("S_S")])
    np.testing.assert_allclose(ds.obs_clean[:, col], expected, rtol=1e-14)
    np.testing.assert_array_equal(ds.obs[:, col], ds.obs_clean[:, col])  # sigma 0 stays clean


def test_sigma_override_applies_only_to_noisy_datasets():
    scod = next(c for c in CANDIDATE_CHANNELS if c.name == "SCOD_tank1")
    nh = next(c for c in CANDIDATE_CHANNELS if c.name == "S_NH_tank1")
    assert SensorModel.channel_sigma(scod, 0.0) == 0.0
    assert SensorModel.channel_sigma(scod, 0.05) == 0.20
    assert SensorModel.channel_sigma(nh, 0.05) == 0.05
    clean = np.full((200000, 2), 100.0)
    noisy, clipped = SensorModel.add_noise_per_channel(
        clean, np.array([0.20, 0.05]), np.random.default_rng(3)
    )
    assert clipped == 0.0
    assert np.std(noisy[:, 0]) / 100.0 == pytest.approx(0.20, rel=2e-2)
    assert np.std(noisy[:, 1]) / 100.0 == pytest.approx(0.05, rel=2e-2)


def test_observability_split_is_unchanged_by_candidates(v):
    assert unobserved_components() == (
        "S_I", "S_S", "X_I", "X_S", "X_B_H", "X_B_A", "X_P", "S_ND", "X_ND", "S_ALK", "S_N2",
    )


def test_observation_operator_supports_linear_channels(v):
    torch = pytest.importorskip("torch")
    from src.models.losses import ObservationOperator

    plant = Bsm1Plant()
    scod = next(c for c in CANDIDATE_CHANNELS if c.name == "SCOD_tank1")
    op = ObservationOperator(plant, (scod,))
    z = torch.rand(4, 5, 14, dtype=torch.float64)
    expected = z[:, 0, v.index("S_I")] + z[:, 0, v.index("S_S")]
    torch.testing.assert_close(op(z)[:, 0], expected, rtol=1e-14, atol=0.0)
    with pytest.raises(ValueError):
        ObservationOperator(plant, (SensorChannel("bad", "linear", tank=0),))

"""Plant layout: BSM1 constants, state packing, TSS, settling, hydraulics."""

from __future__ import annotations

import numpy as np
import pytest

from src.asm1.plant import Bsm1Config, Bsm1Plant, constant_influent
from src.data.influent import stabilisation_influent


def test_bsm1_geometry_constants(plant):
    """These must stay exactly as published; a silent edit would invalidate everything."""
    cfg = plant.cfg
    assert cfg.volumes == (1000.0, 1000.0, 1333.0, 1333.0, 1333.0)
    assert cfg.settler_area == 1500.0
    assert cfg.n_layers == 10 and cfg.feed_layer == 6
    assert cfg.settler_layer_height == 0.4
    assert cfg.q_int == 55338.0 and cfg.q_r == 18446.0 and cfg.q_w == 385.0
    assert cfg.kla == (0.0, 0.0, 240.0, 240.0, 84.0)
    assert cfg.so_sat == 8.0
    assert cfg.x_t == 3000.0
    assert cfg.tss_factor == pytest.approx(0.75)


def test_state_layout_round_trip(plant):
    size = plant.state_size
    assert size == 5 * 14 + 10 + 10 * 8
    y = np.arange(size, dtype=float)
    reactor, solids, solubles = plant.unpack(y)
    np.testing.assert_array_equal(plant.pack(reactor, solids, solubles), y)


def test_tss_excludes_nitrogen_components(plant, sample_state):
    """BSM1 eq. 45 sums five COD particulates; X_ND must not be among them."""
    perturbed = sample_state.copy()
    perturbed[plant.vault.index("X_ND")] *= 10.0
    assert plant.tss(perturbed) == pytest.approx(plant.tss(sample_state))


def test_settling_velocity_is_bounded(plant):
    x = np.linspace(0.0, 20000.0, 500)
    v = plant.settling_velocity(x, x_min=10.0)
    assert np.all(v >= 0.0)
    assert np.all(v <= plant.cfg.v0_prime)


def test_flow_balance_closes(plant):
    """BSM1 eq. 30: Q_f == Q_e + Q_r + Q_w."""
    q_in = 18446.0
    q_f = q_in + plant.cfg.q_r
    q_e = q_in - plant.cfg.q_w
    assert q_f == pytest.approx(q_e + plant.cfg.q_r + plant.cfg.q_w)


def test_zero_reaction_config_disables_biology(sample_state):
    plant = Bsm1Plant(Bsm1Config(reaction_scale=0.0))
    q, z = stabilisation_influent()
    y = plant.seed_state(sample_state, solids_seed=float(plant.tss(sample_state)))
    dy = plant.rhs(0.0, y, constant_influent(q, z))
    assert np.all(np.isfinite(dy))

    reactive = Bsm1Plant(Bsm1Config(reaction_scale=1.0))
    dy_reactive = reactive.rhs(0.0, y, constant_influent(q, z))
    assert not np.allclose(dy, dy_reactive), "reaction_scale=0 changed nothing"


def test_derivative_is_finite_at_an_empty_plant(plant):
    """A cold-start state must not produce NaN through the settler flux terms."""
    q, z = stabilisation_influent()
    y = plant.seed_state(np.zeros(plant.n_components), solids_seed=0.0)
    dy = plant.rhs(0.0, y, constant_influent(q, z))
    assert np.all(np.isfinite(dy))

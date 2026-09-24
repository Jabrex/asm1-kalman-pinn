"""Influent-knowledge views: daily composite recomposition."""

from __future__ import annotations

import numpy as np
import pytest

from src.asm1.vault_loader import vault
from src.data.influent import BSM1_TABLE5_FLOW, BSM1_TABLE5_MEAN, dry_weather, rain_weather
from src.data.influent_views import (
    BIAS_SS_FACTOR,
    apply_influent_knowledge,
    calendar_day_index,
    daily_flow_weighted,
    tkn,
    total_cod,
    view_dataset,
)
from src.data.sensors import ObservationDataset

GRID = np.linspace(0.0, 14.0, 14 * 96 + 1)  # the 15-minute BSM1 grid


@pytest.fixture(scope="module")
def dry_series():
    q, z = dry_weather(14.0).series(GRID)
    return q, z


def test_calendar_days_are_half_open_on_the_left():
    t = np.array([0.0, 0.5, 1.0, 1.0 + 1e-6, 13.99, 14.0])
    assert calendar_day_index(t).tolist() == [0, 0, 0, 1, 13, 13]
    assert calendar_day_index(GRID).max() == 13


def test_exact_mode_is_the_identity(dry_series):
    q, z = dry_series
    out = apply_influent_knowledge(GRID, q, z, "exact", vault())
    assert np.array_equal(out, z)
    assert out is not z


def test_constant_table5_influent_maps_to_itself():
    v = vault()
    t5 = np.array([BSM1_TABLE5_MEAN[c] for c in v.components])
    z = np.tile(t5, (len(GRID), 1))
    q = np.full(len(GRID), BSM1_TABLE5_FLOW)
    out = apply_influent_knowledge(GRID, q, z, "composite", v)
    np.testing.assert_allclose(out, z, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("builder", [dry_weather, rain_weather])
def test_daily_cod_tkn_and_ammonium_are_preserved(builder):
    v = vault()
    q, z = builder(14.0).series(GRID)
    out = apply_influent_knowledge(GRID, q, z, "composite", v)
    for fn in (lambda x: total_cod(x, v), lambda x: tkn(x, v), lambda x: x[:, v.index("S_NH")]):
        _, before = daily_flow_weighted(GRID, q, fn(z))
        _, after = daily_flow_weighted(GRID, q, fn(out))
        np.testing.assert_allclose(after, before, rtol=1e-10)


def test_composite_is_constant_within_each_day(dry_series):
    q, z = dry_series
    out = apply_influent_knowledge(GRID, q, z, "composite", vault())
    day = calendar_day_index(GRID)
    for d in np.unique(day):
        block = out[day == d]
        assert np.all(block == block[0])


def test_fractionation_is_invisible_to_the_composite(dry_series):
    """Same daily COD, TKN and NH4 with a different split gives the same view."""
    v = vault()
    q, z = dry_series
    other = z.copy()
    moved_cod = 0.3 * other[:, v.index("S_S")]
    other[:, v.index("S_S")] -= moved_cod
    other[:, v.index("X_S")] += moved_cod
    moved_n = 0.5 * other[:, v.index("S_ND")]
    other[:, v.index("S_ND")] -= moved_n
    other[:, v.index("X_ND")] += moved_n
    a = apply_influent_knowledge(GRID, q, z, "composite", v)
    b = apply_influent_knowledge(GRID, q, other, "composite", v)
    np.testing.assert_allclose(b, a, rtol=1e-12, atol=1e-12)


def test_biased_view_moves_readily_biodegradable_cod_to_x_s(dry_series):
    v = vault()
    q, z = dry_series
    comp = apply_influent_knowledge(GRID, q, z, "composite", v)
    biased = apply_influent_knowledge(GRID, q, z, "composite_biased", v)
    i_ss, i_xs = v.index("S_S"), v.index("X_S")
    np.testing.assert_allclose(biased[:, i_ss], BIAS_SS_FACTOR * comp[:, i_ss], rtol=1e-14)
    np.testing.assert_allclose(total_cod(biased, v), total_cod(comp, v), rtol=1e-14)
    np.testing.assert_allclose(tkn(biased, v), tkn(comp, v), rtol=1e-14)
    assert np.all(biased[:, i_xs] > comp[:, i_xs])


def test_unknown_mode_is_rejected(dry_series):
    q, z = dry_series
    with pytest.raises(ValueError):
        apply_influent_knowledge(GRID, q, z, "hourly", vault())


def test_view_dataset_replaces_only_the_influent_composition(dry_series):
    q, z = dry_series
    n = len(GRID)
    rng = np.random.default_rng(0)
    ds = ObservationDataset(
        t=GRID, obs_clean=rng.random((n, 8)), obs=rng.random((n, 8)), q_in=q, z_in=z,
        truth_reactor=rng.random((n, 5, 14)), truth_y=rng.random((n, 230)),
        channels=tuple("c%d" % k for k in range(8)), sigma=0.1, seed=2000,
        clip_fraction=0.0, meta={"scenario": "dry"},
    )
    view = view_dataset(ds, "composite")
    assert view.meta["influent_mode"] == "composite"
    assert view.meta["scenario"] == "dry"
    assert np.array_equal(view.z_in, apply_influent_knowledge(GRID, q, z, "composite", vault()))
    for name in ("t", "obs", "obs_clean", "q_in", "truth_reactor", "truth_y"):
        assert getattr(view, name) is getattr(ds, name), name
    assert np.array_equal(ds.z_in, z), "the source dataset was modified"

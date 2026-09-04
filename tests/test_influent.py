"""Influent generator: every magnitude must trace back to a BSM1 anchor."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.influent import (
    BSM1_CONSTANT_COMPONENTS,
    BSM1_FIG3_CONC_RANGE,
    BSM1_FIG3_FLOW_RANGE,
    BSM1_TABLE5_FLOW,
    BSM1_TABLE5_MEAN,
    dry_weather,
    rain_weather,
    stabilisation_influent,
)


@pytest.fixture(scope="module")
def dry():
    return dry_weather(14.0)


def test_flow_mean_is_exactly_the_table_5_value(dry):
    t = np.linspace(0.0, 14.0, 40321)  # 30 s resolution over two whole weeks
    assert np.mean(dry.flow(t)) == pytest.approx(BSM1_TABLE5_FLOW, rel=1e-3)


def test_flow_ratio_matches_the_figure_reading(dry):
    t = np.linspace(0.0, 14.0, 40321)
    q = dry.flow(t)
    low, high = BSM1_FIG3_FLOW_RANGE
    assert np.max(q) / np.min(q) == pytest.approx(high / low, rel=1e-2)


def test_concentration_means_match_table_5(dry):
    t = np.linspace(0.0, 14.0, 40321)
    z = dry.concentrations(t)
    for i, name in enumerate(dry.components):
        assert np.mean(z[:, i]) == pytest.approx(BSM1_TABLE5_MEAN[name], rel=1e-2), name


def test_constant_components_never_vary(dry):
    t = np.linspace(0.0, 14.0, 2001)
    z = dry.concentrations(t)
    for name in BSM1_CONSTANT_COMPONENTS:
        i = dry.vault.index(name)
        assert np.ptp(z[:, i]) == pytest.approx(0.0, abs=1e-12), name
        assert z[0, i] == pytest.approx(BSM1_TABLE5_MEAN[name])


def test_published_ranges_are_approached(dry):
    t = np.linspace(0.0, 14.0, 40321)
    z = dry.concentrations(t)
    for name, (low, high) in BSM1_FIG3_CONC_RANGE.items():
        i = dry.vault.index(name)
        assert np.max(z[:, i]) / np.min(z[:, i]) == pytest.approx(high / low, rel=1e-2), name


def test_load_peak_leads_the_flow_peak(dry):
    """The pollutograph must peak before the hydrograph on a single day."""
    t = np.linspace(3.0, 4.0, 2001)
    q = dry.flow(t)
    ss = dry.concentrations(t)[:, dry.vault.index("S_S")]
    assert t[int(np.argmax(ss))] < t[int(np.argmax(q))]


def test_rain_peak_hits_the_figure_5_reading():
    rain = rain_weather(14.0)
    t = np.linspace(0.0, 14.0, 40321)
    assert np.max(rain.flow(t)) == pytest.approx(rain.spec.rain.peak_flow, rel=1e-3)


def test_rain_dilutes_but_preserves_pollutant_mass_flow():
    """Rain water carries no load, so Q * Z must be unchanged by the event."""
    dry = dry_weather(14.0)
    rain = rain_weather(14.0)
    t = np.linspace(8.0, 11.0, 4001)
    i = dry.vault.index("S_S")
    dry_load = dry.flow(t) * dry.concentrations(t)[:, i]
    rain_load = rain.flow(t) * rain.concentrations(t)[:, i]
    np.testing.assert_allclose(rain_load, dry_load, rtol=1e-10)
    assert np.max(rain.concentrations(t)[:, i]) < np.max(dry.concentrations(t)[:, i])


def test_stabilisation_influent_is_the_table_5_row():
    q, z = stabilisation_influent()
    assert q == BSM1_TABLE5_FLOW
    from src.asm1.vault_loader import vault

    for i, name in enumerate(vault().components):
        assert z[i] == pytest.approx(BSM1_TABLE5_MEAN[name])

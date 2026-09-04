"""Rate evaluation: shapes, signs, backend agreement, pointwise continuity."""

from __future__ import annotations

import numpy as np
import pytest

from src.asm1.continuity import conversion_residual


def test_rate_shapes(kinetics, sample_state):
    rho = kinetics.rates(sample_state)
    assert rho.shape == (8,)
    r = kinetics.conversion(sample_state)
    assert r.shape == (14,)


def test_rates_are_non_negative(kinetics, sample_state):
    """All eight ASM1 process rates are production rates and cannot be negative."""
    rho = kinetics.rates(sample_state)
    assert np.all(rho >= 0.0), dict(zip(kinetics.vault.processes, rho))


def test_batching(kinetics, sample_state):
    """A batched evaluation must agree with the single-state one.

    Not bit for bit: ``rho @ nu`` goes through a matrix-matrix kernel for a
    batch and a matrix-vector kernel for one state, and the two sum in different
    orders. The tolerance is four orders of magnitude tighter than any physical
    quantity here, so it still catches a genuine batching bug.
    """
    batch = np.stack([sample_state, sample_state * 0.5, sample_state * 2.0])
    assert kinetics.rates(batch).shape == (3, 8)
    assert kinetics.conversion(batch).shape == (3, 14)
    np.testing.assert_allclose(
        kinetics.conversion(batch)[0], kinetics.conversion(sample_state), rtol=1e-12, atol=1e-12
    )
    # The process rates themselves are elementwise, so those must match exactly.
    np.testing.assert_array_equal(kinetics.rates(batch)[0], kinetics.rates(sample_state))


def test_pointwise_continuity_vanishes(kinetics, sample_state):
    """r @ C == rho @ (nu @ C) == 0 for any state, to floating-point accuracy."""
    batch = np.stack([sample_state * f for f in (0.1, 0.5, 1.0, 3.0)])
    residual = conversion_residual(kinetics, batch)
    assert np.max(np.abs(residual)) < 1e-9


def test_zero_biomass_does_not_blow_up(kinetics, v, sample_state):
    """rho_7 and rho_8 divide by X_B_H and X_S; the clamp must keep them finite."""
    z = sample_state.copy()
    z[v.index("X_B_H")] = 0.0
    z[v.index("X_S")] = 0.0
    rho = kinetics.rates(z)
    assert np.all(np.isfinite(rho))


def test_torch_matches_numpy(kinetics, sample_state):
    torch = pytest.importorskip("torch")
    z_np = np.stack([sample_state, sample_state * 1.3])
    z_t = torch.as_tensor(z_np, dtype=torch.float64)
    np.testing.assert_allclose(
        kinetics.rates(z_t).numpy(), kinetics.rates(z_np), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        kinetics.conversion(z_t).numpy(), kinetics.conversion(z_np), rtol=1e-12, atol=1e-12
    )


def test_anoxic_growth_is_inhibited_by_oxygen(kinetics, v, sample_state):
    """rho_2 carries an oxygen inhibition switch, so more DO must slow it down."""
    low, high = sample_state.copy(), sample_state.copy()
    low[v.index("S_O")] = 0.1 * v.p("KO_H")
    high[v.index("S_O")] = 10.0 * v.p("KO_H")
    assert kinetics.rates(low)[1] > kinetics.rates(high)[1]

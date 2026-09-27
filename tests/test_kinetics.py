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


def test_rate_parameters_are_the_fifteen_kinetic_constants(kinetics):
    assert kinetics.rate_parameters == ("kh", "KX", "etah", "muH", "etag", "Ks", "bH", "KO_H",
                                        "KNO", "KNH_H", "muA", "bA", "ka", "KO_A", "KNH")


def test_overrides_none_empty_or_vault_values_reproduce_the_default(kinetics, v, sample_state):
    batch = np.stack([sample_state, sample_state * 0.7])
    base = kinetics.conversion(batch)
    same = {name: v.p(name) for name in kinetics.rate_parameters}
    for overrides in (None, {}, same):
        np.testing.assert_array_equal(kinetics.conversion(batch, overrides=overrides), base)


def test_override_changes_only_the_rates_that_read_it(kinetics, v, sample_state):
    rho = kinetics.rates(sample_state)
    fast = kinetics.rates(sample_state, overrides={"muA": 2.0 * v.p("muA")})
    np.testing.assert_allclose(fast[2], 2.0 * rho[2], rtol=1e-15)
    np.testing.assert_array_equal(fast[np.arange(8) != 2], rho[np.arange(8) != 2])
    assert v.p("muA") == 0.8, "the vault itself must be untouched"
    with pytest.raises(KeyError, match="YH"):
        kinetics.rates(sample_state, overrides={"YH": 0.6})


def test_gradient_reaches_a_torch_scalar_override(kinetics, v, sample_state):
    torch = pytest.importorskip("torch")
    z = torch.as_tensor(np.stack([sample_state, sample_state * 1.3]), dtype=torch.float64)
    mu = torch.tensor(v.p("muH"), dtype=torch.float64, requires_grad=True)
    r = kinetics.conversion(z, overrides={"muH": mu})
    assert torch.equal(r.detach(), kinetics.conversion(z))
    r.sum().backward()
    assert bool(torch.isfinite(mu.grad)) and float(mu.grad) != 0.0

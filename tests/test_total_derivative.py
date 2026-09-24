"""Total derivative along the influent trajectory (v1.1) and the causal RAS filter.

v1.0 differentiated the network in ``t`` alone while ``q_in`` and ``z_in``, which
are network inputs that change with time, were held fixed. The physics residual
therefore constrained a partial derivative. These tests pin down the v1.1 fix:

(a) with no influent slopes the v1.0 quantity is returned bit for bit;
(b) with slopes, forward mode matches a central difference taken along the
    piecewise-linear influent trajectory;
(c) forward and reverse mode agree on the total derivative;
(d) zero slopes reproduce the partial derivative;
(e) the collocation slopes are the grid finite differences of the interpolant;
(f) the RAS trailing average never looks ahead;
(g) a default-config training run reproduces the fingerprint recorded from the
    v1.0 code before any v1.1 edit.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.pinn import Asm1Pinn, PinnConfig

REPO = Path(__file__).resolve().parents[1]
DT = 1.0 / 96.0  # the 15-minute sample spacing of the generated datasets

#: Off-knot times: 0.6 + 0.25 k days sit 0.4 of a sample past a grid knot, so a
#: central difference with h = 1e-6 never straddles a slope change.
TIMES = 0.5 + np.arange(12) * 0.25 + 0.1


def _toy_signals(n: int = 400) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Piecewise-linear influent on a 15-min grid, built from smooth periodic shapes."""
    grid = np.arange(n) * DT
    q = 18000.0 * (1.0 + 0.3 * np.sin(2 * np.pi * grid) + 0.1 * np.cos(4 * np.pi * grid))
    base = np.linspace(5.0, 300.0, 14)
    shape = 1.0 + 0.4 * np.sin(2 * np.pi * grid[:, None] + np.linspace(0, 3, 14)[None, :])
    return grid, q, base[None, :] * shape


def _toy_model() -> tuple[Asm1Pinn, np.ndarray, np.ndarray, np.ndarray]:
    torch.manual_seed(0)
    grid, q, z = _toy_signals()
    z0 = np.abs(np.random.default_rng(0).normal(100.0, 30.0, size=(5, 14)))
    model = Asm1Pinn(
        n_tanks=5, n_components=14, horizon_days=float(grid[-1]), z0=z0,
        q_scale=float(q.mean()), z_in_scale=z.mean(axis=0), cfg=PinnConfig(),
    ).to(torch.float64)
    with torch.no_grad():
        model.net[-1].weight.mul_(100.0)  # undo the small-init shrink: a harder test
    return model, grid, q, z


def _inputs(grid, q, z, times):
    qq = np.interp(times, grid, q)
    zz = np.stack([np.interp(times, grid, z[:, j]) for j in range(z.shape[1])], -1)
    idx = np.clip(np.searchsorted(grid, times, "right") - 1, 0, len(grid) - 2)
    span = grid[idx + 1] - grid[idx]
    dq = (q[idx + 1] - q[idx]) / span
    dz = (z[idx + 1] - z[idx]) / span[:, None]

    def T(x):
        return torch.as_tensor(np.asarray(x), dtype=torch.float64)

    return T(qq).view(-1, 1), T(zz), T(dq).view(-1, 1), T(dz)


def _per_output_rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """Max |a - b| normalised per output by that output's largest magnitude."""
    col = torch.maximum(a.abs(), b.abs()).amax(dim=0, keepdim=True).clamp(min=1e-12)
    return float(((a - b).abs() / col).max())


def test_no_tangents_is_the_v10_quantity_bit_for_bit():
    from torch.func import jvp

    model, grid, q, z = _toy_model()
    t = torch.as_tensor(TIMES, dtype=torch.float64).view(-1, 1).requires_grad_(True)
    qq, zz, _, _ = _inputs(grid, q, z, TIMES)
    z_ref, d_ref = jvp(lambda s: model(s, qq, zz), (t,), (torch.ones_like(t),))
    z_new, d_new = model.state_and_derivative(t, qq, zz)
    assert torch.equal(z_new, z_ref) and torch.equal(d_new, d_ref)

    z_rev, d_rev = model.state_and_derivative(t, qq, zz, mode="reverse")
    z_ref_rev = model(t, qq, zz)
    flat = z_ref_rev.reshape(len(t), -1)
    cols = []
    for j in range(flat.shape[1]):
        basis = torch.zeros_like(flat)
        basis[:, j] = 1.0
        (g,) = torch.autograd.grad(flat, t, grad_outputs=basis, retain_graph=True)
        cols.append(g)
    assert torch.equal(z_rev, z_ref_rev)
    assert torch.equal(d_rev.detach(), torch.cat(cols, dim=-1).view_as(z_ref_rev))


def test_total_derivative_matches_trajectory_central_difference():
    model, grid, q, z = _toy_model()
    t = torch.as_tensor(TIMES, dtype=torch.float64).view(-1, 1).requires_grad_(True)
    qq, zz, dq, dz = _inputs(grid, q, z, TIMES)
    z_val, d_tot = model.state_and_derivative(t, qq, zz, dq_dt=dq, dz_dt=dz)

    h = 1e-6
    with torch.no_grad():
        tt = torch.as_tensor(TIMES, dtype=torch.float64).view(-1, 1)
        qp, zp, _, _ = _inputs(grid, q, z, TIMES + h)
        qm, zm, _, _ = _inputs(grid, q, z, TIMES - h)
        fd = (model(tt + h, qp, zp) - model(tt - h, qm, zm)) / (2.0 * h)
        # the state value itself is untouched by the tangents
        assert torch.equal(z_val, model(tt, qq, zz))
    assert _per_output_rel(d_tot.detach(), fd) < 1e-4

    # the partial derivative must fail the same check, or the test cannot tell them apart
    _, d_par = model.state_and_derivative(t, qq, zz)
    assert _per_output_rel(d_par.detach(), fd) > 1e-2


def test_forward_and_reverse_total_derivatives_agree():
    model, grid, q, z = _toy_model()
    t = torch.as_tensor(TIMES[:5], dtype=torch.float64).view(-1, 1).requires_grad_(True)
    qq, zz, dq, dz = _inputs(grid, q, z, TIMES[:5])
    _, fwd = model.state_and_derivative(t, qq, zz, mode="forward", dq_dt=dq, dz_dt=dz)
    _, rev = model.state_and_derivative(t, qq, zz, mode="reverse", dq_dt=dq, dz_dt=dz)
    scale = torch.maximum(fwd.abs(), rev.abs()).clamp(min=1e-12)
    assert float(((fwd - rev).abs() / scale).max()) < 1e-6


def test_zero_tangents_equal_the_partial_derivative():
    model, grid, q, z = _toy_model()
    t = torch.as_tensor(TIMES, dtype=torch.float64).view(-1, 1).requires_grad_(True)
    qq, zz, dq, dz = _inputs(grid, q, z, TIMES)
    _, d_zero = model.state_and_derivative(
        t, qq, zz, dq_dt=torch.zeros_like(dq), dz_dt=torch.zeros_like(dz)
    )
    _, d_par = model.state_and_derivative(t, qq, zz)
    torch.testing.assert_close(d_zero, d_par, rtol=1e-12, atol=1e-14)

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
DT = 1.0 / 96.0

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
        model.net[-1].weight.mul_(100.0)
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
        assert torch.equal(z_val, model(tt, qq, zz))
    assert _per_output_rel(d_tot.detach(), fd) < 1e-4

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


def test_trailing_average_is_causal_and_correct():
    from src.train.curriculum import trailing_average

    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 3))
    w = 4
    out = trailing_average(x, w)
    naive = np.stack([x[max(0, i - w + 1): i + 1].mean(axis=0) for i in range(len(x))])
    np.testing.assert_allclose(out, naive, rtol=1e-12, atol=1e-12)

    k = 20
    y = x.copy()
    y[k] += 1e3
    np.testing.assert_array_equal(trailing_average(y, w)[:k], out[:k])
    assert not np.allclose(trailing_average(y, w)[k], out[k])

    np.testing.assert_allclose(trailing_average(x[:, 0], w), naive[:, 0], rtol=1e-12)
    same = trailing_average(x, 1)
    np.testing.assert_array_equal(same, x)
    assert same is not x


def _require(path: Path) -> Path:
    if not path.exists():
        pytest.skip("needs %s - run 'python -m scripts.generate_data' first" % path.name)
    return path


def _trainer(**overrides):
    from src.train.run import RunConfig, Trainer

    data_dir = _require(REPO / "results" / "raw" / "obs_dry_sigma0p05.npz").parent
    kwargs = dict(
        run_id="_td_probe", model="cl_pinn", noise=0.05, profile="quick", steps_quick=1,
        device="cpu", dtype="float64", data_dir=str(data_dir), out_dir=tempfile.mkdtemp(),
    )
    kwargs.update(overrides)
    return Trainer(RunConfig(**kwargs))


def test_collocation_slopes_are_grid_finite_differences():
    trainer = _trainer()
    stage = trainer.schedule.stages[-1]
    batch = trainer._stage_tensors(stage)
    torch.manual_seed(3)
    colloc = trainer._collocation(stage, batch)

    grid = batch["t"].squeeze(-1).numpy()
    ts = colloc["t"].detach().squeeze(-1).numpy()
    lo = np.clip(np.searchsorted(grid, ts, "right") - 1, 0, len(grid) - 2)
    span = grid[lo + 1] - grid[lo]
    q = batch["q_in"].squeeze(-1).numpy()
    z = batch["z_in"].numpy()
    np.testing.assert_allclose(
        colloc["dq_dt"].squeeze(-1).numpy(), (q[lo + 1] - q[lo]) / span, rtol=1e-9, atol=1e-9
    )
    np.testing.assert_allclose(
        colloc["dz_dt"].numpy(), (z[lo + 1] - z[lo]) / span[:, None], rtol=1e-9, atol=1e-9
    )
    assert colloc["dq_dt"].shape == (trainer.cfg.collocation_points, 1)
    assert colloc["dz_dt"].shape == (trainer.cfg.collocation_points, 14)


def test_ras_filter_feeds_the_filtered_signal_to_training():
    from src.train.curriculum import trailing_average

    base = _trainer()
    filt = _trainer(ras_filter_window=4)
    stage = base.schedule.stages[-1]
    raw = base.data["dry"].window(0.0, stage.horizon_days).obs[:, base.ras_col]
    got = filt._stage_tensors(stage)["tss_ras"].squeeze(-1).numpy()
    np.testing.assert_allclose(got, trailing_average(raw, 4), rtol=1e-12)
    np.testing.assert_array_equal(
        base._stage_tensors(stage)["tss_ras"].squeeze(-1).numpy(), raw
    )


def test_default_config_reproduces_the_v10_fingerprint():
    """Recorded from the unmodified v1.0 code (CPU, float64, 6 steps, seed 0).

    The hashes depend on the installed torch/numpy build; after an intentional
    dependency upgrade, re-record the file from the v1.0.0 tag, not from HEAD
    (see scripts/record_default_fingerprint.py).
    """
    reference = json.loads(
        (REPO / "tests" / "data" / "v10_default_fingerprint.json").read_text(encoding="utf-8")
    )
    keys = ("total", "data", "physics", "ic", "positivity", "balance")
    for model, expected in reference.items():
        if model.startswith("_"):
            continue
        trainer = _trainer(model=model, steps_quick=6, log_every=1, seed=0)
        trainer.train()
        digest = hashlib.sha256()
        for name, p in sorted(trainer.model.state_dict().items()):
            digest.update(name.encode())
            digest.update(p.detach().cpu().numpy().tobytes())
        assert [[r[k] for k in keys] for r in trainer.history] == expected["history"], model
        assert digest.hexdigest() == expected["state_sha256"], model


def test_total_derivative_training_steps_are_finite():
    trainer = _trainer(total_derivative=True, ras_filter_window=4, steps_quick=4, log_every=1)
    summary = trainer.train()
    for record in trainer.history:
        assert all(np.isfinite(record[k]) for k in ("total", "physics", "data"))
    assert summary["total_derivative"] is True
    assert summary["ras_filter_window"] == 4
    assert summary["variant"] == ""

"""Reduced model, Kalman filters, anchors (v1.1, G3). Data tests skip without results/raw
(fail under ASM1_STRICT_TESTS=1)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.asm1.plant import Bsm1Plant  # noqa: E402
from src.data.influent import stabilisation_influent  # noqa: E402
from src.eval.metrics import state_metrics, track_summary  # noqa: E402
from src.models.losses import Asm1Loss, ObservationOperator  # noqa: E402
from src.observers.reduced_model import ReducedPlantModel  # noqa: E402
from src.train.run import RAS_CHANNEL, TARGET_CHANNELS  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "results" / "raw"
STRICT = os.environ.get("ASM1_STRICT_TESTS", "").strip().lower() not in {"", "0", "false", "no"}


def _need(*names: str) -> Path:
    missing = [n for n in names if not (RAW / n).exists()]
    if missing:
        message = "needs %s in results/raw (python -m scripts.generate_data)" % missing
        if STRICT:
            pytest.fail("ASM1_STRICT_TESTS is set: " + message)
        pytest.skip(message)
    return RAW


def _track(truth, pred, key="track_b_unmeasured"):
    return track_summary(state_metrics(truth, pred))[key]["nrmse"]


def _initial_state(ds):
    """The supplied A0 initial condition (index 0 only)."""
    return np.asarray(ds.truth_reactor[0], dtype=float)


def _cols(ds):
    names = list(ds.channels)
    return [names.index(c.name) for c in TARGET_CHANNELS], names.index(RAS_CHANNEL)


@pytest.fixture(scope="module")
def model():
    torch.set_num_threads(1)
    return ReducedPlantModel()


@pytest.fixture(scope="module")
def probe(model):
    """Positive (5, 14) state and constant inputs from sourced numbers only."""
    q, z_in = stabilisation_influent()
    rng = np.random.default_rng(0)
    z = np.abs(np.tile(z_in, (5, 1)) * (1.0 + 0.1 * rng.normal(size=(5, 14)))) + 0.05
    return z, float(q), z_in, float(model.plant.tss(z[4])) * 2.0


@pytest.fixture(scope="module")
def dry():
    from src.data.sensors import ObservationDataset

    _need("obs_dry_sigma0p00.npz", "obs_dry_sigma0p10.npz")
    return {tag: ObservationDataset.load(RAW / ("obs_dry_sigma%s.npz" % tag))
            for tag in ("0p00", "0p10")}


def test_rhs_equals_the_pinn_residual_rhs(model, probe):
    z, q, z_in, tss = probe
    plant = Bsm1Plant()
    loss = Asm1Loss(plant, ObservationOperator(plant, TARGET_CHANNELS), np.ones(14),
                    np.ones(len(TARGET_CHANNELS)), torch.device("cpu"), torch.float64)
    f64 = lambda a: torch.as_tensor(np.asarray(a, dtype=float), dtype=torch.float64)  # noqa: E731
    expected = loss.plant_rhs(f64(z[None]), f64([[q]]), f64(z_in[None]), f64([[tss]]))
    assert float(torch.max(torch.abs(model.rhs(z, q, z_in, tss) - expected))) <= 1e-12


def test_rhs_log_is_f_over_z_and_params_reach_the_model(model, probe):
    z, q, z_in, tss = probe
    g = model.rhs_log(np.log(z).reshape(-1), q, z_in, tss).numpy()
    f = model.rhs(z, q, z_in, tss).numpy().reshape(-1)
    np.testing.assert_allclose(g, f / z.reshape(-1), rtol=1e-12, atol=0.0)
    base = model.rhs(z, q, z_in, tss)
    assert torch.equal(base, model.rhs(z, q, z_in, tss, params={"muA": model.parameters["muA"]}))
    assert not torch.equal(base, model.rhs(z, q, z_in, tss, params={"muA": 0.5}))


def test_jacobians_match_central_differences(model, probe):
    z, q, z_in, tss = probe
    x = np.log(z).reshape(-1)
    params = {"muH": model.parameters["muH"], "bA": model.parameters["bA"]}
    j_x, b_ras, b_zin, b_th = model.jacobians(x, q, z_in, tss, params)
    assert (j_x.shape, b_ras.shape, b_zin.shape, b_th.shape) == ((70, 70), (70, 1), (70, 14), (70, 2))
    g = lambda xx, rr=tss, pp=None: model.rhs_log(xx, q, z_in, rr, pp).numpy()  # noqa: E731
    h = 1e-6
    for j in (0, 17, 35, 69):
        e = np.zeros(70)
        e[j] = h
        fd = (g(x + e) - g(x - e)) / (2 * h)
        np.testing.assert_allclose(j_x[:, j], fd, rtol=1e-5, atol=1e-6 * np.max(np.abs(fd)))
    fd = (g(x, tss * (1 + 1e-6)) - g(x, tss * (1 - 1e-6))) / (2e-6 * tss)
    np.testing.assert_allclose(b_ras[:, 0], fd, rtol=1e-5, atol=1e-9)
    mu, ba = params["muH"], params["bA"]
    fd = (g(x, pp={"muH": mu * np.exp(h), "bA": ba}) - g(x, pp={"muH": mu * np.exp(-h), "bA": ba})) / (2 * h)
    np.testing.assert_allclose(b_th[:, 0], fd, rtol=1e-5, atol=1e-9)


def test_ideal_settler_mode_ignores_the_ras_input(probe):
    z, q, z_in, tss = probe
    ideal = ReducedPlantModel(ras_mode="ideal_settler")
    assert np.all(ideal.jacobians(np.log(z).reshape(-1), q, z_in, tss)[1] == 0.0)
    assert torch.equal(ideal.rhs(z, q, z_in, tss), ideal.rhs(z, q, z_in, 5.0 * tss))
    with pytest.raises(ValueError):
        ReducedPlantModel(ras_mode="settler")


def test_expo_integrator_matches_bdf_over_14_days(model, dry):
    ds = dry["0p00"]
    args = (_initial_state(ds), ds.t, ds.q_in, ds.z_in, ds.obs[:, _cols(ds)[1]])
    assert _track(model.integrate_bdf(*args), model.integrate_expo(*args)) < 1e-3


def test_true_underflow_solubles_reproduce_the_full_plant_reactor(model):
    _need("sim_dry.npz")
    with np.load(RAW / "sim_dry.npz") as sim:
        reactor = sim["reactor"]
        tru = model.integrate_bdf(reactor[0], sim["t"], sim["q_in"], sim["influent"],
                                  sim["tss_underflow"],
                                  recycle_solubles=sim["underflow"][:, model.plant.i_soluble])
    assert _track(reactor, tru) < 2e-3
    assert _track(reactor, tru, "track_a_measured") < 2e-3


# --------------------------------------------------------------------------
# Kalman filter: core
# --------------------------------------------------------------------------
import time  # noqa: E402

from scipy.integrate import quad_vec  # noqa: E402
from scipy.linalg import expm  # noqa: E402

from src.observers.ekf import (EkfConfig, estimate_r_from_data, ras_log_variance,  # noqa: E402
                               rts_smooth, run_ekf, van_loan)
from src.train.curriculum import trailing_average  # noqa: E402


def _window(ds, days):
    cols, ras_col = _cols(ds)
    sel = ds.t <= days + 1e-9
    raw = ds.obs[sel, ras_col]
    return (ds.t[sel], ds.obs[sel][:, cols], TARGET_CHANNELS, ds.q_in[sel], ds.z_in[sel],
            trailing_average(raw, 4)), ras_log_variance(raw, 4)


def test_van_loan_matches_quadrature():
    rng = np.random.default_rng(0)
    a = np.diag([-7900.0, -50.0, -0.05, 0.0]) + 0.1 * rng.normal(size=(4, 4))
    qc, h = np.diag([1e-2, 1e-3, 1e-4, 1e-6]), 15.0 / 1440.0
    ref = quad_vec(lambda s: expm(a * s) @ qc @ expm(a * s).T, 0.0, h, epsabs=1e-16, epsrel=1e-12)[0]
    assert np.max(np.abs(van_loan(a, qc, h) - ref) / np.maximum(np.abs(ref), 1e-30)) < 1e-8


def test_estimate_r_recovers_multiplicative_noise_and_respects_the_floor():
    t = np.linspace(0.0, 12.0, 1153)
    noisy = (5.0 + 2.0 * np.sin(2 * np.pi * t)) * (1.0 + 0.10 * np.random.default_rng(3).normal(size=t.size))
    r = estimate_r_from_data(noisy[:, None], r_floor=0.01)
    assert r[0] == pytest.approx(0.01, rel=0.15)
    assert estimate_r_from_data(np.full((50, 1), 3.0))[0] == pytest.approx(1e-4, rel=1e-12)
    assert ras_log_variance(noisy, 4) == pytest.approx(r[0] / 4.0, rel=1e-12)


def test_ekf_with_negligible_noise_reproduces_the_open_loop(model, dry):
    ds = dry["0p10"]
    args, _ = _window(ds, 2.0)
    res = run_ekf(model, *args, _initial_state(ds), np.full((5, 14), 1e-8),
                  EkfConfig(q_soluble=1e-8, q_particulate=1e-8))
    ol = model.integrate_expo(_initial_state(ds), args[0], args[3], args[4], args[5])
    assert not res.diverged and _track(ol, res.z()) < 1e-3


def test_covariances_stay_symmetric_positive_definite(model, dry):
    ds = dry["0p10"]
    args, rv = _window(ds, 1.0)
    res = run_ekf(model, *args, _initial_state(ds), np.full((5, 14), 0.05), EkfConfig(),
                  ras_log_var=rv)
    assert not res.diverged
    for P in (res.P_pred, res.P_filt, rts_smooth(res).P):
        assert np.max(np.abs(P - np.transpose(P, (0, 2, 1)))) == 0.0
        for k in range(0, P.shape[0], 8):
            eig = np.linalg.eigvalsh(P[k])
            assert eig.min() > -1e-12 * eig.max()


def test_one_14_day_filter_and_smoother_pass_is_fast(model, dry):
    ds = dry["0p10"]
    args, rv = _window(ds, 14.0)
    started = time.perf_counter()
    res = run_ekf(model, *args, _initial_state(ds), np.full((5, 14), 0.01),
                  EkfConfig(q_soluble=0.1, q_particulate=0.03), ras_log_var=rv)
    rts_smooth(res)
    assert time.perf_counter() - started <= 180.0


# --------------------------------------------------------------------------
# Kalman filter: augmented state, IEKS, forecast, q tuning
# --------------------------------------------------------------------------
import json  # noqa: E402

from src.observers.ekf import channel_weights, forecast, ieks, tune_q  # noqa: E402


def test_augmented_filter_keeps_exact_kinetics_on_twin_data(model, dry):
    """Twin data from the reduced model itself; on real data the multipliers drift."""
    ds = dry["0p00"]
    (t, _, _, q, zin, _), _ = _window(ds, 2.0)
    ras = ds.obs[ds.t <= 2.0 + 1e-9, _cols(ds)[1]]
    traj = model.integrate_expo(_initial_state(ds), t, q, zin, ras)
    w = channel_weights(model, TARGET_CHANNELS)
    y = (traj.reshape(len(t), -1) @ w.T) * (1 + 0.02 * np.random.default_rng(11).normal(size=(len(t), 7)))
    cfg = EkfConfig(q_soluble=1e-3, q_particulate=1e-3, q_theta=0.0, augment=("muH", "bH"))
    res = run_ekf(model, t, y, TARGET_CHANNELS, q, zin, ras, _initial_state(ds),
                  np.full((5, 14), 0.01), cfg)
    assert not res.diverged and np.all(np.abs(res.multipliers()[-1] - 1.0) < 0.05)


def test_forecast_is_the_open_loop_from_the_given_state(model, dry):
    ds = dry["0p10"]
    sel = (ds.t >= 12.0 - 1e-9) & (ds.t <= 12.5 + 1e-9)
    ras = trailing_average(ds.obs[:, _cols(ds)[1]], 4)[sel]
    x = np.log(_initial_state(ds)).reshape(-1)
    ol = model.integrate_expo(_initial_state(ds), ds.t[sel], ds.q_in[sel], ds.z_in[sel], ras)
    np.testing.assert_array_equal(forecast(model, x, ds.t[sel], ds.q_in[sel], ds.z_in[sel], ras), ol)
    aug = forecast(model, np.concatenate([x, [0.0]]), ds.t[sel], ds.q_in[sel], ds.z_in[sel], ras,
                   names=("muA",))
    np.testing.assert_allclose(aug, ol, rtol=1e-12, atol=0.0)


def test_ieks_runs_its_relinearisations(model, dry):
    ds = dry["0p10"]
    args, rv = _window(ds, 1.0)
    cfg = EkfConfig(q_soluble=0.1, q_particulate=0.03)
    it = ieks(model, *args, _initial_state(ds), np.full((5, 14), 0.05), cfg, ras_log_var=rv)
    assert 1 <= it.iterations <= cfg.ieks_iterations and np.all(np.isfinite(it.changes))
    assert it.converged or it.changes == sorted(it.changes, reverse=True)


def test_tune_q_selects_by_the_declared_criterion(model, dry):
    ds = dry["0p10"]
    args, rv = _window(ds, 1.0)
    call = lambda c: tune_q(model, *args, _initial_state(ds), np.full((5, 14), 0.05), EkfConfig(),  # noqa: E731
                            ras_log_var=rv, grid=(0.01, 0.1), criterion=c, horizon_steps=8, stride=16)
    inn, pred = call("innovation"), call("predictive")
    best = max(inn.table, key=lambda row: row["loglik"])
    low = min(pred.table, key=lambda row: row["predictive_score"])
    assert len(inn.table) == 4
    assert (inn.q_soluble, inn.q_particulate) == (best["q_soluble"], best["q_particulate"])
    assert (pred.q_soluble, pred.q_particulate) == (low["q_soluble"], low["q_particulate"])
    assert inn.on_grid_edge == (0.1 in (inn.q_soluble, inn.q_particulate))
    with pytest.raises(ValueError):
        call("truth")


def test_tuned_filter_is_statistically_consistent():
    """NIS/m in [0.5, 2] at K0-Ie-A0, sigma 0.10, tuned q (reads the Task 3.10 run)."""
    path = REPO / "results/v11/observers/k000_ie_a0/eks_sigma0p10/summary.json"
    if not path.exists():
        pytest.skip("run configs/observers/k000_ie_a0.yaml first")
    info = json.loads(path.read_text(encoding="utf-8"))
    assert 0.5 <= info["nis_mean"] / info["n_channels"] <= 2.0

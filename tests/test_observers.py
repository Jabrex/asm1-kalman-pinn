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

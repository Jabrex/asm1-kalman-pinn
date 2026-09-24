"""Linearised information analysis on the reduced model (src/observers/sensitivity.py).

Data-dependent tests use results/raw (tracked in git) and skip only when it is
missing, like the runtime leakage layer (ASM1_STRICT_TESTS=1 turns the skip
into a failure).
"""

from __future__ import annotations

import io
import os
import tokenize
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from src.asm1.vault_loader import vault
from src.data.sensors import ObservationDataset, SensorChannel
from src.observers import sensitivity as sens
from src.train.run import RAS_CHANNEL, TARGET_CHANNELS

REPO = Path(__file__).resolve().parents[1]
DRY = REPO / "results" / "raw" / "obs_dry_sigma0p10.npz"
ONE_DAY = 97  # samples: 96 intervals of 15 min
STRICT = os.environ.get("ASM1_STRICT_TESTS", "").strip().lower() not in {"", "0", "false", "no"}


def _require(path: Path) -> Path:
    if not path.exists():
        message = "%s missing; run scripts/generate_data.py" % path
        if STRICT:
            pytest.fail(message)
        pytest.skip(message)
    return path


def _rel(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


@pytest.fixture(scope="module")
def model():
    from src.observers.reduced_model import ReducedPlantModel

    return ReducedPlantModel()


@pytest.fixture(scope="module")
def dry():
    return ObservationDataset.load(_require(DRY))


@pytest.fixture(scope="module")
def day(model, dry):
    """One day of the reduced model from the simulated t = 0 state, and its propagators."""
    t = dry.t[:ONE_DAY]
    u = sens.InputTrajectory(dry.q_in[:ONE_DAY], dry.z_in[:ONE_DAY],
                             dry.obs_clean[:ONE_DAY, dry.channels.index(RAS_CHANNEL)])

    def solve(x0, params=None, z_scale=None):
        def inputs(tt):
            q, z, r = u.at(t, tt)
            return q, (z if z_scale is None else z * z_scale), r

        def f(tt, x):
            q, z, r = inputs(tt)
            return sens._np(model.rhs_log(x, q, z, r, params=params)).reshape(-1)

        def jac(tt, x):
            q, z, r = inputs(tt)
            return sens._np(model.jacobians(x, q, z, r, params=params)[0])

        out = solve_ivp(f, (t[0], t[-1]), x0, t_eval=t, method="BDF", rtol=1e-10, atol=1e-12, jac=jac)
        assert out.success, out.message
        return out.y.T

    x0 = np.log(dry.truth_reactor[0].reshape(-1))
    x_traj = solve(x0)
    tl = sens.tangent_linear(model, x_traj, u, t, inputs="zin_rel")
    return {"t": t, "u": u, "x": x_traj, "z": np.exp(x_traj).reshape(ONE_DAY, 5, 14), "tl": tl,
            "m": sens.cumulative_propagators(tl.phis), "solve": solve, "x0": x0}


# -- Task 4.1: G3 contract and tangent-linear propagation ------------------------
def test_reduced_model_contract(model, dry):
    """jacobians -> (J_x, B_ras, B_zin, B_theta) in log-state coordinates (G3)."""
    v = vault()
    x = np.log(dry.truth_reactor[0].reshape(-1))
    tss = float(dry.obs_clean[0, dry.channels.index(RAS_CHANNEL)])
    out = model.jacobians(x, float(dry.q_in[0]), dry.z_in[0], tss,
                          params={"muA": v.p("muA"), "bA": v.p("bA")})
    j_x, b_ras, b_zin, b_theta = (sens._np(a) for a in out)
    assert j_x.shape == (70, 70)
    assert b_ras.reshape(70, -1).shape == (70, 1)
    assert b_zin.shape == (70, 14)
    assert b_theta.shape == (70, 2)


def test_jacobians_match_central_differences(model, dry):
    """jacrev against central differences with h = 1e-6 (measured 1e-10 to 5e-10)."""
    v = vault()
    x = np.log(dry.truth_reactor[0].reshape(-1))
    q, zin = float(dry.q_in[0]), dry.z_in[0].copy()
    tss = float(dry.obs_clean[0, dry.channels.index(RAS_CHANNEL)])
    params = {name: v.p(name) for name in ("muA", "bA", "muH", "KX")}
    j_x, b_ras, b_zin, b_theta = (sens._np(a) for a in model.jacobians(x, q, zin, tss, params=params))

    def g(xx=x, zz=zin, rr=tss, pp=None):
        return sens._np(model.rhs_log(xx, q, zz, rr, params=pp)).reshape(-1)

    h = 1e-6
    eye14 = np.eye(14)
    fd_x = np.stack([(g(xx=x + h * e) - g(xx=x - h * e)) / (2 * h) for e in np.eye(70)], axis=1)
    assert _rel(j_x, fd_x) < 1e-6
    nz = np.nonzero(zin > 0.0)[0]
    fd_z = np.stack([(g(zz=zin + h * zin[j] * eye14[j]) - g(zz=zin - h * zin[j] * eye14[j])) / (2 * h * zin[j])
                     for j in nz], axis=1)
    assert _rel(b_zin[:, nz], fd_z) < 1e-6
    fd_r = (g(rr=tss * (1 + h)) - g(rr=tss * (1 - h))) / (2 * h * tss)
    assert _rel(b_ras.reshape(-1), fd_r) < 1e-6
    cols = []
    for name in params:
        up, down = dict(params), dict(params)
        up[name] *= np.exp(h)
        down[name] *= np.exp(-h)
        cols.append((g(pp=up) - g(pp=down)) / (2 * h))
    assert _rel(b_theta, np.stack(cols, axis=1)) < 1e-6


@pytest.mark.parametrize("direction", ["heterotrophs", "random"])
def test_tangent_linear_matches_nonlinear_perturbation(day, direction):
    """delta = 1e-4 relative, one day, default 9 substeps (measured 4.3e-4 and 4.9e-4)."""
    if direction == "heterotrophs":
        v = np.zeros(70)
        v[[k * 14 + vault().index("X_B_H") for k in range(5)]] = 1.0
    else:
        v = np.random.default_rng(0).normal(size=70)
    delta = 1e-4
    nonlinear = day["solve"](day["x0"] + delta * v)[-1] - day["x"][-1]
    linear = day["m"][-1] @ (delta * v)
    assert np.linalg.norm(linear - nonlinear) / np.linalg.norm(nonlinear) < 1e-3


def test_input_responses_match_nonlinear_perturbation(model, day):
    """Kinetic and influent columns (measured 2.2e-4 for muA, 1.6e-4 for S_S)."""
    v = vault()
    delta = 1e-4
    tl = sens.tangent_linear(model, day["x"], day["u"], day["t"], inputs="theta", params={"muA": v.p("muA")})
    g_theta = sens.cumulative_input_response(tl.phis, tl.gammas)[-1][:, 0]
    fd = (day["solve"](day["x0"], params={"muA": v.p("muA") * np.exp(delta)})[-1] - day["x"][-1]) / delta
    assert np.linalg.norm(g_theta - fd) / np.linalg.norm(fd) < 1e-3
    j = v.index("S_S")
    scale = np.ones(14)
    scale[j] += delta
    g_zin = sens.cumulative_input_response(day["tl"].phis, day["tl"].gammas)[-1][:, j]
    fd = (day["solve"](day["x0"], z_scale=scale)[-1] - day["x"][-1]) / delta
    assert np.linalg.norm(g_zin - fd) / np.linalg.norm(fd) < 1e-3


def test_log_measurement_rows(day):
    comps = vault().components
    tss = [c for c in TARGET_CHANNELS if c.kind == "tss_reactor"][0]
    rows = sens.log_measurement_rows(day["z"], [tss], comps)[:, 0].reshape(-1, 5, 14)
    assert rows.sum(axis=(1, 2)) == pytest.approx(np.ones(ONE_DAY))
    assert np.all(rows[:, :4] == 0.0)
    with pytest.raises(ValueError):
        sens.log_measurement_rows(day["z"], [SensorChannel("TSS_ras", "tss_underflow")], comps)
    assert sens.log_noise_variance([tss], 0.10) == pytest.approx([np.log1p(0.01)])

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


# -- Task 4.2: Fisher information and posterior ------------------------------------
def _fisher_parts(day):
    h = sens.log_measurement_rows(day["z"], TARGET_CHANNELS, vault().components)
    r = sens.log_noise_variance(TARGET_CHANNELS, 0.10)
    parts = [sens.channel_fisher(None, h[:, j], r[j], cumulative=day["m"]) for j in range(len(TARGET_CHANNELS))]
    return h, r, parts


def test_fisher_is_symmetric_psd_and_additive(day):
    h, r, parts = _fisher_parts(day)
    f = sens.fisher_initial_state(day["tl"].phis, h, r)
    assert np.array_equal(f, f.T)
    assert np.linalg.eigvalsh(f / np.linalg.norm(f, 2)).min() >= -1e-12
    assert _rel(f, sum(parts)) < 1e-12


def test_removing_a_channel_never_increases_information(day):
    h, r, parts = _fisher_parts(day)
    full = sum(parts)
    scale = np.linalg.norm(full, 2)
    p0 = np.full(70, np.log1p(0.5 ** 2))
    ig_full = sens.information_gain(sens.posterior_cov(full, p0), p0)
    for j in range(len(parts)):
        sub = sum(p for i, p in enumerate(parts) if i != j)
        assert np.linalg.eigvalsh((full - sub) / scale).min() >= -1e-10
        assert np.all(sens.information_gain(sens.posterior_cov(sub, p0), p0) <= ig_full + 1e-12)


def test_si_gains_information_only_through_transport(day):
    comps = vault().components
    i_si = [k * 14 + comps.index("S_I") for k in range(5)]
    h, r, parts = _fisher_parts(day)
    f = sum(parts)
    assert np.abs(h[:, :, i_si]).max() == 0.0                    # no target channel senses S_I
    assert np.abs(f[i_si]).max() <= 1e-12 * np.abs(f).max()      # nor does S_I act on what they sense
    scod = SensorChannel("SCOD_tank5", "linear", tank=4, weights=(("S_I", 1.0), ("S_S", 1.0)),
                         sigma_override=0.20)
    h_s = sens.log_measurement_rows(day["z"], [scod], comps)
    sensed = set(np.nonzero(np.abs(h_s).sum(axis=(0, 1)))[0].tolist())
    assert sensed == {4 * 14 + comps.index("S_I"), 4 * 14 + comps.index("S_S")}
    f_s = sens.channel_fisher(None, h_s[:, 0], sens.log_noise_variance([scod], 0.10)[0], cumulative=day["m"])
    assert np.all(np.diag(f_s)[i_si] > 0.0)                       # tanks 1-4 reached through transport


def test_posterior_cov_matches_direct_inverse():
    rng = np.random.default_rng(3)
    a = rng.normal(size=(70, 30))
    f = a @ a.T
    p0 = rng.uniform(0.01, 0.5, size=70)
    assert _rel(sens.posterior_cov(f, p0), np.linalg.inv(np.diag(1.0 / p0) + f)) < 1e-9
    full = np.diag(p0) + 0.001 * np.ones((70, 70))
    assert _rel(sens.posterior_cov(f, full), np.linalg.inv(np.linalg.inv(full) + f)) < 1e-8
    ig = sens.information_gain(sens.posterior_cov(f, p0), p0)
    assert ig.shape == (5, 14) and ig.min() >= 0.0 and ig.max() <= 1.0


def test_fisher_eigen_whitens_with_the_prior():
    w, _ = sens.fisher_eigen(np.diag([4.0, 1.0]), np.array([0.25, 1.0]))
    assert w == pytest.approx([1.0, 1.0])
    u_sum, u_split = sens.sum_split_directions(np.ones((5, 14)), vault().components)
    assert abs(u_sum @ u_split) < 1e-12


# -- Task 4.3: start-up memory, slow modes, influent forcing ------------------------
def test_self_sensitivity_decay_recovers_known_rates():
    t = np.linspace(0.0, 12.0, 1153)
    rates = np.full(70, 0.5)
    rates[14:28] = 0.05                                   # tank 2 never reaches 1/e in 12 d
    phis = np.stack([np.diag(np.exp(-rates * (t[1] - t[0])))] * (t.size - 1))
    tau, extrap = sens.self_sensitivity_decay(phis, t)
    assert tau[0] == pytest.approx(np.full(14, 2.0), abs=1e-4)
    assert not extrap[0].any()
    assert tau[1] == pytest.approx(np.full(14, 20.0), rel=1e-9)   # 12 / -ln(exp(-0.6))
    assert extrap[1].all()


def test_self_sensitivity_pools_tanks():
    """Mixing between tanks must not register as lost memory."""
    t = np.linspace(0.0, 12.0, 1153)
    perm = np.zeros((70, 70))
    for k in range(5):
        for c in range(14):
            perm[((k + 1) % 5) * 14 + c, k * 14 + c] = 1.0
    phis = np.stack([np.exp(-0.25 * (t[1] - t[0])) * perm] * (t.size - 1))
    tau, _ = sens.self_sensitivity_decay(phis, t)
    assert tau == pytest.approx(np.full((5, 14), 4.0), abs=1e-3)


def test_forcing_share_limits():
    t = np.linspace(0.0, 1.0, 11)
    phis = np.stack([np.eye(70)] * 10)
    gammas = np.zeros((10, 70, 14))
    gammas[:, :, 0] = 0.1
    names = tuple("z%d" % i for i in range(14))
    tl = sens.TangentLinear(t, phis, gammas, "zin_rel", names, 1)
    assert sens.forcing_share(tl, np.full(70, 1e-12))["share"].min() > 0.999
    quiet = sens.TangentLinear(t, phis, np.zeros_like(gammas), "zin_rel", names, 1)
    assert sens.forcing_share(quiet, np.full(70, 0.1))["share"].max() == 0.0
    with pytest.raises(ValueError):
        sens.forcing_share(sens.TangentLinear(t, phis, None, "none", (), 1), np.full(70, 0.1))


def test_modal_analysis_orders_slowest_first():
    modes = sens.modal_analysis(np.diag([-2.0, -1.0 / 7.0, -1.0 / 3.0]), n_modes=3, labels=["a", "b", "c"])
    assert [m["time_constant_days"] for m in modes] == pytest.approx([7.0, 3.0, 0.5])
    assert modes[0]["top_participation"][0] == ["b", pytest.approx(1.0)]


def test_finite_difference_jacobian_schemes_on_a_kink():
    def fun(y):
        return np.array([min(y[0], 1.0) + 2.0 * y[1], y[0] * y[1]])

    y = np.array([1.0, 3.0])
    fwd = sens.finite_difference_jacobian(fun, y, scheme="forward")
    bwd = sens.finite_difference_jacobian(fun, y, scheme="backward")
    assert fwd[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert bwd[0, 0] == pytest.approx(1.0, rel=1e-6)
    assert fwd[1] == pytest.approx([3.0, 1.0], rel=1e-5)


# -- Task 4.4: identifiability, classes, ideal settler, leakage guard --------------
def test_collinearity_index_hand_example():
    # unit columns (1, 1)/sqrt(2) and (0, 1): Gram [[1, c], [c, 1]] with c = 1/sqrt(2)
    s = np.array([[1.0, 0.0], [1.0, 1.0]])
    assert sens.collinearity_index(s) == pytest.approx(1.0 / np.sqrt(1.0 - 1.0 / np.sqrt(2.0)), rel=1e-12)
    assert sens.collinearity_index(np.eye(3)) == pytest.approx(1.0)
    assert sens.collinearity_index(np.array([[1.0, 2.0], [2.0, 4.0 + 1e-9]])) > 1e3


def test_d_optimal_subset_skips_collinear_pairs():
    s = np.array([[100.0, 100.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
    f = s.T @ s
    out = sens.d_optimal_subset(f, k=2, max_ci=20.0, names=["a", "b", "c"])
    assert np.linalg.slogdet(f[:2, :2])[1] > max(row["logdet"] for row in out["ranking"])
    assert out["best"]["names"] == ["b", "c"]            # (a, b) has the largest det but CI near 71
    assert out["n_admissible"] == 2 and out["n_candidates"] == 3


def test_classify_states_rule_order():
    ig = np.array([[0.6, 0.2, 0.2, 0.2, 0.5, 0.2, 0.1]])
    tau = np.array([[10.0, 7.0, 0.5, 3.0, 0.5, 6.0, np.inf]])
    share = np.array([[0.9, 0.9, 0.8, 0.8, 0.9, 0.9, 0.0]])
    assert list(sens.classify_states(ig, tau, share)[0]) == [
        "sensor-recoverable", "anchor-carried", "forcing-slaved", "partly recoverable",
        "sensor-recoverable", "partly recoverable", "anchor-carried",
    ]


def test_ideal_settler_jacobian_matches_central_differences(model, dry):
    from src.asm1.plant import Bsm1Plant

    ideal = sens.IdealSettlerModel(model, Bsm1Plant())
    x = np.log(dry.truth_reactor[0].reshape(-1))
    q, zin = float(dry.q_in[0]), dry.z_in[0]
    h = 1e-6
    fd = np.stack([(ideal.rhs_log(x + h * e, q, zin) - ideal.rhs_log(x - h * e, q, zin)).reshape(-1) / (2 * h)
                   for e in np.eye(70)], axis=1)
    j_x, b_ras, _, _ = ideal.jacobians(x, q, zin)
    assert _rel(j_x, fd) < 1e-6
    assert np.all(b_ras == 0.0)


def test_sensitivity_module_never_names_truth():
    source = (REPO / "src" / "observers" / "sensitivity.py").read_text(encoding="utf-8")
    names = {tok.string for tok in tokenize.generate_tokens(io.StringIO(source).readline)
             if tok.type == tokenize.NAME}
    assert not names & {"truth_reactor", "truth_y", "obs_clean", "truth_plants"}


def test_trajectory_drift_keeps_a_mass_neutral_inert_shift_invisible(model, day):
    """Linearising the nominal model along another plant's trajectory (G4 fix, 2026-09-24).

    With the model's own f/z on the log-Jacobian diagonal, a trajectory made with other
    kinetics lets TSS "see" a mass-neutral X_I <-> X_P shift (measured 1.3e-4 of the sum's
    information); with the trajectory's own log-slope it stays at round-off (4e-15).
    """
    v = vault()
    other = {"bH": 0.5 * v.p("bH"), "muH": 0.7 * v.p("muH")}
    x = day["solve"](day["x0"], params=other)
    z = np.exp(x).reshape(ONE_DAY, 5, 14)
    tss = [c for c in TARGET_CHANNELS if c.kind == "tss_reactor"]
    h = sens.log_measurement_rows(z, tss, v.components)
    u_sum, u_split = sens.sum_split_directions(z[0], v.components)
    ratio = {}
    for drift in ("model", "trajectory"):
        m = sens.cumulative_propagators(sens.tangent_linear(model, x, day["u"], day["t"], drift=drift).phis)
        f = sens.channel_fisher(None, h[:, 0], 0.01, cumulative=m)
        ratio[drift] = (u_split @ f @ u_split) / (u_sum @ f @ u_sum)
    assert ratio["trajectory"] < 1e-10 < 1e-6 < ratio["model"]
    with pytest.raises(ValueError):
        sens.tangent_linear(model, x, day["u"], day["t"], drift="spline")

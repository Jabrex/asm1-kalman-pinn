"""Linearised information analysis on the reduced reactor-train model.

Pure array functions. Nothing here loads a dataset, opens a file or names a
ground-truth field: a caller passes a trajectory, the known inputs and a prior,
and gets information measures back.

Coordinates follow the observers (``src/observers/ekf.py``):

* state ``x = log z``, 70 entries, tank-major (tank 1 components 0..13, then
  tank 2, ...);
* a measurement is ``y_j = log(w_j . z)`` with log-variance ``r_j``, so a
  multiplicative sensor error ``sigma`` becomes ``r = log(1 + sigma^2)``;
* kinetic parameters enter as log multipliers on their vault values.

The propagator over one sample interval is built from the model Jacobians at
``substeps + 1`` nodes with the symmetric product
``expm(A_S h/2) expm(A_{S-1} h) ... expm(A_1 h) expm(A_0 h/2)`` along a cubic
spline of the log trajectory. The half-steps at the interval ends put the fast,
quasi-steady oxygen and ammonium modes on the Jacobian of the correct time,
which the midpoint rule does not. With 9 substeps per 15-min sample the one-day
propagation error against a nonlinear finite perturbation is about 5e-4
(tests/test_sensitivity.py pins 1e-3); 3 substeps give 2e-3 to 3e-3.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Mapping, NamedTuple, Sequence

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import cho_factor, cho_solve, eig, expm

from ..asm1.plant import TSS_COMPONENTS, Bsm1Plant

N_TANKS = 5
N_COMPONENTS = 14
N_STATE = N_TANKS * N_COMPONENTS
DEFAULT_SUBSTEPS = 9


def _np(a: Any) -> np.ndarray:
    """NumPy float64 view of a NumPy array or a torch tensor."""
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    return np.asarray(a, dtype=float)


class InputTrajectory(NamedTuple):
    """Known inputs on the sample grid: influent flow, composition and RAS TSS."""

    q_in: np.ndarray
    z_in: np.ndarray
    tss_ras: np.ndarray

    def at(self, t: np.ndarray, tt: float) -> tuple[float, np.ndarray, float]:
        """Linear interpolation in time, as the observers' input series do."""
        q = float(np.interp(tt, t, self.q_in))
        z = np.array([np.interp(tt, t, self.z_in[:, j]) for j in range(self.z_in.shape[1])])
        r = float(np.interp(tt, t, self.tss_ras))
        return q, z, r


@dataclass
class TangentLinear:
    """One-interval propagators along a trajectory.

    ``phis[k]`` maps a log-state perturbation at ``t[k]`` to ``t[k+1]``.
    ``gammas[k]`` is the response at ``t[k+1]`` to a unit input perturbation
    held constant over ``[t[k], t[k+1]]`` (columns named by ``input_names``).
    """

    t: np.ndarray
    phis: np.ndarray
    gammas: np.ndarray | None
    input_kind: str
    input_names: tuple[str, ...]
    substeps: int


def tangent_linear(
    model: Any,
    x_traj: np.ndarray,
    u_traj: InputTrajectory,
    t: np.ndarray,
    substeps: int = DEFAULT_SUBSTEPS,
    inputs: str = "none",
    params: Mapping[str, float] | None = None,
    input_names: Sequence[str] = (),
    drift: str = "model",
) -> TangentLinear:
    """Propagators of the linearised reduced model along ``x_traj``.

    ``model`` needs ``jacobians(x70, q, zin, tss, params=None)`` returning
    ``(J_x, B_ras, B_zin, B_theta)`` in log-state coordinates (G3 contract).
    ``inputs``: ``"none"``; ``"zin_rel"`` (relative perturbation of each of the
    14 influent components, ``B_zin * z_in``); or ``"theta"`` (log multipliers
    on ``params``, in ``params`` order).

    ``drift`` fixes the diagonal of the log-state Jacobian, ``-d(log z)/dt``:
    ``"model"`` takes it from the model's own field ``f(z)/z`` (exact when
    ``x_traj`` solves the model); ``"trajectory"`` takes it from the spline
    derivative of ``x_traj``. Use ``"trajectory"`` when the model is linearised
    along a trajectory it does not follow (another plant's kinetics): absolute
    perturbations then obey ``d(dz)/dt = J(z) dz`` exactly, and a mass-neutral
    shift between inert fractions stays invisible, as it should.
    """
    if drift not in ("model", "trajectory"):
        raise ValueError("drift must be 'model' or 'trajectory', not %r" % (drift,))
    if inputs not in ("none", "zin_rel", "theta"):
        raise ValueError("inputs must be 'none', 'zin_rel' or 'theta', not %r" % (inputs,))
    if inputs == "theta" and not params:
        raise ValueError("inputs='theta' needs a non-empty params mapping")
    t = np.asarray(t, dtype=float)
    n = t.size
    x_traj = np.asarray(x_traj, dtype=float).reshape(n, N_STATE)
    call_params = dict(params) if params else None
    if inputs == "theta":
        names = tuple(call_params)
    elif inputs == "zin_rel":
        names = tuple(input_names) if input_names else tuple("zin_%d" % i for i in range(u_traj.z_in.shape[1]))
    else:
        names = ()
    p = len(names)
    d = N_STATE + p
    spline = CubicSpline(t, x_traj, axis=0)
    slope = spline.derivative() if drift == "trajectory" else None

    def generator(tt: float) -> np.ndarray:
        q, zin, tss = u_traj.at(t, tt)
        x = spline(tt)
        j_x, _b_ras, b_zin, b_theta = model.jacobians(x, q, zin, tss, params=call_params)
        a = np.zeros((d, d))
        a[:N_STATE, :N_STATE] = _np(j_x)
        if slope is not None:
            g = _np(model.rhs_log(x, q, zin, tss, params=call_params)).reshape(-1)
            a[:N_STATE, :N_STATE] += np.diag(g - slope(tt))
        if inputs == "zin_rel":
            a[:N_STATE, N_STATE:] = _np(b_zin).reshape(N_STATE, -1) * zin[None, :]
        elif inputs == "theta":
            a[:N_STATE, N_STATE:] = _np(b_theta).reshape(N_STATE, p)
        return a

    phis = np.empty((n - 1, N_STATE, N_STATE))
    gammas = np.empty((n - 1, N_STATE, p)) if p else None
    left = generator(float(t[0]))
    for k in range(n - 1):
        h = (t[k + 1] - t[k]) / substeps
        prop = expm(left * (0.5 * h))
        for j in range(1, substeps):
            prop = expm(generator(float(t[k] + j * h)) * h) @ prop
        right = generator(float(t[k + 1]))
        prop = expm(right * (0.5 * h)) @ prop
        left = right
        phis[k] = prop[:N_STATE, :N_STATE]
        if gammas is not None:
            gammas[k] = prop[:N_STATE, N_STATE:]
    return TangentLinear(t=t, phis=phis, gammas=gammas, input_kind=inputs,
                         input_names=names, substeps=int(substeps))


def cumulative_propagators(phis: np.ndarray) -> np.ndarray:
    """``M[k]`` maps a perturbation at ``t[0]`` to ``t[k]``; ``M[0] = I``."""
    phis = np.asarray(phis, dtype=float)
    m = np.empty((phis.shape[0] + 1,) + phis.shape[1:])
    m[0] = np.eye(phis.shape[1])
    for k in range(phis.shape[0]):
        m[k + 1] = phis[k] @ m[k]
    return m


def cumulative_input_response(phis: np.ndarray, gammas: np.ndarray) -> np.ndarray:
    """Response at every sample to an input perturbation held from ``t[0]`` on."""
    g = np.zeros((phis.shape[0] + 1, phis.shape[1], gammas.shape[2]))
    for k in range(phis.shape[0]):
        g[k + 1] = phis[k] @ g[k] + gammas[k]
    return g


def log_measurement_rows(z_traj: np.ndarray, channels: Sequence[Any], components: Sequence[str]) -> np.ndarray:
    """``d log y_j / d x`` along a trajectory; returns ``(n, m, 70)``.

    ``state`` channels give a one-hot row; ``tss_reactor`` and ``linear``
    channels give the weight share ``w_i z_i / sum(w z)`` over their support.
    """
    comps = list(components)
    z = np.asarray(z_traj, dtype=float).reshape(-1, N_TANKS, len(comps))
    n = z.shape[0]
    rows = np.zeros((n, len(channels), N_TANKS, len(comps)))
    for j, ch in enumerate(channels):
        if ch.kind == "state":
            rows[:, j, int(ch.tank), comps.index(str(ch.component))] = 1.0
            continue
        if ch.kind == "tss_reactor":
            weights = [(name, 1.0) for name in TSS_COMPONENTS]
        elif ch.kind == "linear":
            weights = [(str(name), float(w)) for name, w in ch.weights]
        else:
            raise ValueError(
                "Channel %r has kind %r; only reactor measurements carry information "
                "on the reactor state." % (ch.name, ch.kind)
            )
        cols = np.array([comps.index(name) for name, _ in weights])
        w = np.array([val for _, val in weights])
        part = z[:, int(ch.tank)][:, cols] * w[None, :]
        rows[:, j, int(ch.tank), cols] = part / part.sum(axis=1, keepdims=True)
    return rows.reshape(n, len(channels), N_STATE)


def log_noise_variance(channels: Sequence[Any], sigma: float) -> np.ndarray:
    """Log-variance per channel; ``sigma_override`` (G2) wins over ``sigma``."""
    out = []
    for ch in channels:
        s = getattr(ch, "sigma_override", None)
        s = float(sigma) if s is None else float(s)
        if s <= 0.0:
            raise ValueError("Fisher information needs a positive noise level (channel %r)" % ch.name)
        out.append(np.log1p(s * s))
    return np.asarray(out)


def channel_fisher(phis: np.ndarray | None, h_j: np.ndarray, r_j: float,
                   cumulative: np.ndarray | None = None) -> np.ndarray:
    """Initial-state Fisher information of one channel over the window.

    Fisher information is additive over channels (independent noise), so a
    subset's information is the sum of its channels' matrices.
    """
    m = cumulative if cumulative is not None else cumulative_propagators(phis)
    h_j = np.asarray(h_j, dtype=float)
    if h_j.shape != (m.shape[0], m.shape[1]):
        raise ValueError("h_j must be (n, 70) with n = len(phis) + 1; got %s" % (h_j.shape,))
    w = np.einsum("kn,knm->km", h_j, m)
    f = w.T @ w / float(r_j)
    return 0.5 * (f + f.T)


def fisher_initial_state(phis: np.ndarray | None, h_rows: np.ndarray, r: np.ndarray,
                         p0: np.ndarray | None = None,
                         cumulative: np.ndarray | None = None) -> np.ndarray:
    """Sum of ``channel_fisher`` over the channels of ``h_rows`` ``(n, m, 70)``.

    With ``p0`` the prior information ``inv(P0)`` is added (posterior
    information matrix).
    """
    m = cumulative if cumulative is not None else cumulative_propagators(phis)
    f = np.zeros((N_STATE, N_STATE))
    for j in range(h_rows.shape[1]):
        f += channel_fisher(None, h_rows[:, j, :], float(r[j]), cumulative=m)
    if p0 is not None:
        c = _prior_factor(p0)
        f += np.linalg.inv(c @ c.T)
    return f


def static_fisher(h_rows: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Information of one-off measurements of the initial state (lab rows)."""
    h = np.asarray(h_rows, dtype=float).reshape(-1, N_STATE)
    return (h / np.asarray(r, dtype=float)[:, None]).T @ h


def _prior_factor(p0: np.ndarray) -> np.ndarray:
    p0 = np.asarray(p0, dtype=float)
    if p0.ndim == 1:
        return np.diag(np.sqrt(p0))
    return np.linalg.cholesky(0.5 * (p0 + p0.T))


def prior_variance(p0: np.ndarray) -> np.ndarray:
    p0 = np.asarray(p0, dtype=float)
    return p0.copy() if p0.ndim == 1 else np.diag(p0).copy()


def posterior_cov(f: np.ndarray, p0: np.ndarray) -> np.ndarray:
    """``(inv(P0) + F)^-1`` computed as ``C (I + C^T F C)^-1 C^T`` with ``P0 = C C^T``.

    ``p0`` is a vector (diagonal prior) or a full covariance.
    """
    c = _prior_factor(p0)
    k = np.eye(c.shape[1]) + c.T @ f @ c
    p = c @ cho_solve(cho_factor(0.5 * (k + k.T)), c.T)
    return 0.5 * (p + p.T)


def information_gain(p_post: np.ndarray, p0: np.ndarray) -> np.ndarray:
    """``IG = 1 - sigma_post / sigma_prior`` per state, shaped ``(5, 14)``."""
    ratio = np.clip(np.diag(p_post), 0.0, None) / prior_variance(p0)
    return np.clip(1.0 - np.sqrt(ratio), 0.0, 1.0).reshape(N_TANKS, -1)


def window_variance(cov0: np.ndarray, cumulative: np.ndarray) -> np.ndarray:
    """``diag(M_k P M_k^T)`` for every sample; ``(n, 70)``."""
    cov0 = np.diag(cov0) if np.ndim(cov0) == 1 else np.asarray(cov0, dtype=float)
    return np.einsum("kij,kij->ki", cumulative @ cov0, cumulative)


def crb_over_range(p_post: np.ndarray, cumulative: np.ndarray, z_traj: np.ndarray,
                   ranges: np.ndarray) -> np.ndarray:
    """Window-RMS posterior standard deviation in natural units over the range.

    Delta method: ``sd_z(t) = z(t) * sqrt(diag(M_t P M_t^T))``. ``ranges`` is
    ``(14,)`` (pooled over tanks, as the fixed-spread NRMSE uses) or ``(5, 14)``.
    """
    n = cumulative.shape[0]
    var = window_variance(p_post, cumulative)
    sd = np.sqrt(np.clip(var, 0.0, None)) * np.asarray(z_traj, dtype=float).reshape(n, N_STATE)
    rms = np.sqrt(np.mean(sd ** 2, axis=0)).reshape(N_TANKS, -1)
    return rms / np.broadcast_to(np.asarray(ranges, dtype=float), rms.shape)


def fisher_eigen(f: np.ndarray, p0: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues (ascending) and eigenvectors of ``F``, prior-whitened if ``p0``.

    Whitened eigenvalue ``lam`` means the posterior variance along that
    direction is ``1 / (1 + lam)`` of the prior: ``lam < 1`` directions stay
    prior-dominated (weakly observable).
    """
    if p0 is not None:
        c = _prior_factor(p0)
        f = c.T @ f @ c
    w, v = np.linalg.eigh(0.5 * (f + f.T))
    return w, v


def weak_directions(evals: np.ndarray, evecs: np.ndarray, labels: Sequence[str],
                    n: int = 5, zero_tol: float = 1e-9, top: int = 4) -> dict[str, Any]:
    """Summary of the weakest non-null directions and the null-space size."""
    scale = max(float(np.max(np.abs(evals))), 1e-300)
    null = evals <= zero_tol * scale
    out = []
    for i in np.nonzero(~null)[0][:n]:
        load = evecs[:, i] ** 2
        order = np.argsort(load)[::-1][:top]
        out.append({"eigenvalue": float(evals[i]),
                    "top_states": [[labels[k], float(load[k])] for k in order]})
    null_load = (evecs[:, null] ** 2).sum(axis=1) if null.any() else np.zeros(len(labels))
    return {"n_null": int(null.sum()), "n_prior_dominated": int(np.sum(evals < 1.0)),
            "null_space_states": [labels[k] for k in np.nonzero(null_load > 0.5)[0]],
            "weakest": out}


def sum_split_directions(z0: np.ndarray, components: Sequence[str], a: str = "X_I",
                         b: str = "X_P") -> tuple[np.ndarray, np.ndarray]:
    """Unit log-space directions that scale ``a + b`` or move mass from ``b`` to ``a``."""
    comps = list(components)
    z = np.asarray(z0, dtype=float).reshape(N_TANKS, len(comps))
    ia, ib = comps.index(a), comps.index(b)
    u_sum = np.zeros_like(z)
    u_sum[:, ia] = 1.0
    u_sum[:, ib] = 1.0
    u_split = np.zeros_like(z)
    u_split[:, ia] = 1.0 / z[:, ia]
    u_split[:, ib] = -1.0 / z[:, ib]
    return u_sum.ravel() / np.linalg.norm(u_sum), u_split.ravel() / np.linalg.norm(u_split)


def direction_information_gain(p_post: np.ndarray, p0: np.ndarray, u: np.ndarray) -> float:
    prior = np.diag(p0) if np.ndim(p0) == 1 else np.asarray(p0, dtype=float)
    return float(1.0 - np.sqrt((u @ p_post @ u) / (u @ prior @ u)))


def self_sensitivity_decay(phis: np.ndarray, t: np.ndarray,
                           cumulative: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """1/e decay time of the self-sensitivity of each (tank, component).

    The diagonal is taken in the tank-pooled basis: ``r[k, c](t)`` is the
    response of component ``c`` in tank ``k`` to the same relative error in
    ``c`` in all five tanks. Tank-by-tank diagonals decay within the ~20 min
    tank residence time and would measure mixing, not memory. Returns
    ``(tau_days (5, 14), extrapolated (5, 14))``; a response that never falls
    below 1/e inside the window is extrapolated as ``T / -ln r(T)`` when
    ``0 < r(T) < 1``, and set to ``inf`` otherwise.
    """
    t = np.asarray(t, dtype=float)
    m = cumulative if cumulative is not None else cumulative_propagators(phis)
    n_c = m.shape[1] // N_TANKS
    resp = np.empty((m.shape[0], N_TANKS, n_c))
    for c in range(n_c):
        cols = c + n_c * np.arange(N_TANKS)
        resp[:, :, c] = m[:, cols][:, :, cols].sum(axis=2)
    tau = np.full((N_TANKS, n_c), np.inf)
    extrap = np.zeros((N_TANKS, n_c), dtype=bool)
    level = np.exp(-1.0)
    span = t[-1] - t[0]
    for k in range(N_TANKS):
        for c in range(n_c):
            r = resp[:, k, c]
            below = np.nonzero(r <= level)[0]
            if below.size:
                i = int(below[0])
                w = (r[i - 1] - level) / (r[i - 1] - r[i])
                tau[k, c] = t[i - 1] - t[0] + w * (t[i] - t[i - 1])
            else:
                extrap[k, c] = True
                if 0.0 < r[-1] < 1.0:
                    tau[k, c] = span / -np.log(r[-1])
    return tau, extrap


def modal_analysis(jac: np.ndarray, n_modes: int = 8,
                   labels: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """Slowest modes of a Jacobian: eigenvalue, time constant and participation."""
    w, vl, vr = eig(np.asarray(jac, dtype=float), left=True, right=True)
    order = np.argsort(np.abs(w.real))
    out = []
    for i in order[:n_modes]:
        part = np.abs(np.conj(vl[:, i]) * vr[:, i])
        part = part / part.sum()
        top = np.argsort(part)[::-1][:5]
        out.append({
            "eigenvalue": [float(w[i].real), float(w[i].imag)],
            "time_constant_days": float(-1.0 / w[i].real) if w[i].real < 0 else float("inf"),
            "top_participation": [[labels[k] if labels else int(k), float(part[k])] for k in top],
        })
    return out


def jacobian_slow_modes(model: Any, z_ss: np.ndarray, u_ss: tuple[float, np.ndarray, float],
                        n_modes: int = 8, labels: Sequence[str] | None = None) -> dict[str, Any]:
    """Modal analysis of the reduced model at a (steady) state ``z_ss``."""
    q, zin, tss = u_ss
    x = np.log(np.clip(np.asarray(z_ss, dtype=float).reshape(-1), 1e-12, None))
    j_x = _np(model.jacobians(x, float(q), np.asarray(zin, dtype=float), float(tss))[0])
    g = _np(model.rhs_log(x, float(q), np.asarray(zin, dtype=float), float(tss))).reshape(-1)
    return {"modes": modal_analysis(j_x, n_modes, labels), "rhs_log_max_abs": float(np.max(np.abs(g)))}


def finite_difference_jacobian(fun: Any, y: np.ndarray, rel_step: float = 1e-6,
                               floor: float = 1.0, scheme: str = "central") -> np.ndarray:
    """Jacobian of ``fun(y)`` by forward, backward or central differences.

    The full plant has non-smooth settler terms (``min`` fluxes, the clipped
    settling velocity). At a kink one-sided derivatives differ, so the caller
    reports all three schemes rather than trusting one.
    """
    y = np.asarray(y, dtype=float)
    f0 = np.asarray(fun(y), dtype=float)
    jac = np.empty((f0.size, y.size))
    for j in range(y.size):
        h = rel_step * max(abs(y[j]), floor)
        yp, ym = y.copy(), y.copy()
        yp[j] += h
        ym[j] -= h
        if scheme == "central":
            jac[:, j] = (np.asarray(fun(yp)) - np.asarray(fun(ym))) / (2.0 * h)
        elif scheme == "forward":
            jac[:, j] = (np.asarray(fun(yp)) - f0) / h
        elif scheme == "backward":
            jac[:, j] = (f0 - np.asarray(fun(ym))) / h
        else:
            raise ValueError("scheme must be central, forward or backward")
    return jac


def forcing_share(tl: TangentLinear, p0: np.ndarray, zin_rel_sd: float = 0.10,
                  cumulative: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Share of the window variance due to influent composition error.

    Each of the 14 influent components carries an independent, persistent
    relative error of standard deviation ``zin_rel_sd`` over the whole window;
    the initial state carries the prior ``p0``. Share per state =
    ``mean_t var_forcing / (mean_t var_forcing + mean_t var_initial)``.
    """
    if tl.input_kind != "zin_rel" or tl.gammas is None:
        raise ValueError("forcing_share needs tangent_linear(..., inputs='zin_rel')")
    m = cumulative if cumulative is not None else cumulative_propagators(tl.phis)
    g = cumulative_input_response(tl.phis, tl.gammas)
    var_forcing = (zin_rel_sd ** 2) * np.sum(g ** 2, axis=2)
    var_initial = window_variance(p0, m)
    vf, vi = var_forcing.mean(axis=0), var_initial.mean(axis=0)
    share = np.where(vf + vi > 0.0, vf / np.maximum(vf + vi, 1e-300), 0.0)
    return {"share": share.reshape(N_TANKS, -1), "var_forcing": var_forcing, "var_initial": var_initial}


CLASS_RULES: Mapping[str, float] = {
    "ig_sensor": 0.5,
    "tau_anchor_days": 6.0,
    "share_forcing": 0.5,
    "tau_forcing_days": 1.0,
}
CLASS_NAMES = ("sensor-recoverable", "anchor-carried", "forcing-slaved", "partly recoverable")


def steady_state_parameter_sensitivity(model: Any, z_ss: np.ndarray,
                                       u_ss: tuple[float, np.ndarray, float],
                                       params: Mapping[str, float]) -> np.ndarray:
    """``d x_ss / d log theta = -J^-1 B_theta`` at a steady state; ``(70, p)``."""
    q, zin, tss = u_ss
    x = np.log(np.clip(np.asarray(z_ss, dtype=float).reshape(-1), 1e-12, None))
    j_x, _b, _bz, b_theta = model.jacobians(x, float(q), np.asarray(zin, dtype=float), float(tss),
                                            params=dict(params))
    return -np.linalg.solve(_np(j_x), _np(b_theta).reshape(N_STATE, len(params)))


def parameter_sensitivity(model: Any, x_traj: np.ndarray, u_traj: InputTrajectory, t: np.ndarray,
                          params: Mapping[str, float], h_rows: np.ndarray, r: np.ndarray,
                          substeps: int = DEFAULT_SUBSTEPS,
                          x0_sensitivity: np.ndarray | None = None,
                          drift: str = "model") -> dict[str, Any]:
    """Noise-whitened output sensitivities to log kinetic multipliers.

    Returns ``s_theta`` ``(n*m, p)`` and ``s_x0`` ``(n*m, 70)`` (the same rows
    for the initial log state), plus the cumulative propagators. Parameter
    Jacobians come from the model's autograd ``B_theta`` through ``params``
    overrides. ``x0_sensitivity`` ``(70, p)`` couples the initial state to the
    parameters (plant at its own steady state before day 0); ``None`` keeps the
    initial state fixed.
    """
    tl = tangent_linear(model, x_traj, u_traj, t, substeps=substeps, inputs="theta", params=params,
                        drift=drift)
    m = cumulative_propagators(tl.phis)
    g = cumulative_input_response(tl.phis, tl.gammas)
    if x0_sensitivity is not None:
        g = g + m @ np.asarray(x0_sensitivity, dtype=float)
    w = 1.0 / np.sqrt(np.asarray(r, dtype=float))
    s_theta = np.einsum("kjn,knp->kjp", h_rows, g) * w[None, :, None]
    s_x0 = np.einsum("kjn,knm->kjm", h_rows, m) * w[None, :, None]
    return {"names": tl.input_names, "s_theta": s_theta.reshape(-1, len(tl.input_names)),
            "s_x0": s_x0.reshape(-1, N_STATE), "cumulative": m}


def marginal_parameter_fisher(s_theta: np.ndarray, s_x0: np.ndarray, p0: np.ndarray) -> np.ndarray:
    """Parameter information after marginalising the initial state (Schur complement)."""
    f_tt = s_theta.T @ s_theta
    f_tx = s_theta.T @ s_x0
    info_x = s_x0.T @ s_x0 + np.diag(1.0 / prior_variance(p0))
    f = f_tt - f_tx @ cho_solve(cho_factor(0.5 * (info_x + info_x.T)), f_tx.T)
    return 0.5 * (f + f.T)


def _ci_from_gram(gram: np.ndarray) -> float:
    d = np.sqrt(np.clip(np.diag(gram), 0.0, None))
    if np.any(d <= 0.0):
        return float("inf")
    lam = float(np.linalg.eigvalsh(gram / np.outer(d, d)).min())
    return float("inf") if lam <= 0.0 else float(1.0 / np.sqrt(lam))


def collinearity_index(s: np.ndarray) -> float:
    """Brun et al. (2002): ``1 / sqrt(min eig(S_n^T S_n))``, unit-length columns."""
    s = np.asarray(s, dtype=float)
    return _ci_from_gram(s.T @ s)


def d_optimal_subset(f_theta: np.ndarray, k: int = 4, max_ci: float = 20.0,
                     names: Sequence[str] | None = None, ci_gram: np.ndarray | None = None,
                     top: int = 20) -> dict[str, Any]:
    """Subset of ``k`` parameters maximising ``log det F_K`` with ``CI_K < max_ci``.

    ``ci_gram`` (default ``f_theta``) supplies the collinearity index; pass
    ``S^T S`` of the raw output sensitivities to get Brun's index exactly.
    """
    f = np.asarray(f_theta, dtype=float)
    p = f.shape[0]
    names = list(names) if names is not None else ["p%d" % i for i in range(p)]
    gram = f if ci_gram is None else np.asarray(ci_gram, dtype=float)
    rows = []
    for combo in itertools.combinations(range(p), k):
        idx = list(combo)
        sign, logdet = np.linalg.slogdet(f[np.ix_(idx, idx)])
        rows.append({"names": [names[i] for i in idx], "indices": idx,
                     "logdet": float(logdet) if sign > 0 else float("-inf"),
                     "ci": _ci_from_gram(gram[np.ix_(idx, idx)])})
    admissible = [row for row in rows if row["ci"] < max_ci and np.isfinite(row["logdet"])]
    if not admissible:
        raise ValueError("no %d-parameter subset has a collinearity index below %.1f" % (k, max_ci))
    best = max(admissible, key=lambda row: row["logdet"])
    ranked = sorted(admissible, key=lambda row: -row["logdet"])[:top]
    return {"best": best, "ranking": ranked, "n_admissible": len(admissible), "n_candidates": len(rows)}


def classify_states(ig: np.ndarray, tau: np.ndarray, share: np.ndarray,
                    rules: Mapping[str, float] = CLASS_RULES) -> np.ndarray:
    """Pre-registered class per (tank, component), rules applied in order."""
    ig, tau, share = (np.asarray(a, dtype=float) for a in (ig, tau, share))
    out = np.full(ig.shape, CLASS_NAMES[3], dtype=object)
    forcing = (share > rules["share_forcing"]) & (tau < rules["tau_forcing_days"])
    out[forcing] = CLASS_NAMES[2]
    anchor = (tau > rules["tau_anchor_days"]) & (ig < rules["ig_sensor"])
    out[anchor] = CLASS_NAMES[1]
    out[ig >= rules["ig_sensor"]] = CLASS_NAMES[0]
    return out


class IdealSettlerModel:
    """Reduced model whose RAS TSS comes from an ideal clarifier, not a sensor.

    ``TSS_ras = TSS_5 (Q_in + Q_r) / (Q_r + Q_w)`` is the solids balance of a
    clarifier with no storage and solids-free effluent. The measured RAS input
    passed by callers is ignored. The Jacobian adds the chain-rule term
    ``B_ras (d TSS_ras / d x)``.
    """

    def __init__(self, base: Any, plant: Bsm1Plant) -> None:
        self.base = base
        cfg = plant.cfg
        self._q_r, self._q_w, self._factor = float(cfg.q_r), float(cfg.q_w), float(cfg.tss_factor)
        self._cols = np.array([(cfg.n_tanks - 1) * plant.n_components + int(i) for i in plant.i_tss])

    def _gain(self, q: float) -> float:
        return (float(q) + self._q_r) / (self._q_r + self._q_w)

    def tss_ras(self, x70: np.ndarray, q: float) -> float:
        x = np.asarray(x70, dtype=float).reshape(-1)
        return self._gain(q) * self._factor * float(np.exp(x[self._cols]).sum())

    def rhs_log(self, x70: np.ndarray, q: float, zin: np.ndarray, tss: float | None = None,
                params: Mapping[str, float] | None = None) -> np.ndarray:
        return _np(self.base.rhs_log(x70, q, zin, self.tss_ras(x70, q), params=params))

    def jacobians(self, x70: np.ndarray, q: float, zin: np.ndarray, tss: float | None = None,
                  params: Mapping[str, float] | None = None):
        x = np.asarray(x70, dtype=float).reshape(-1)
        j_x, b_ras, b_zin, b_theta = (
            _np(a) for a in self.base.jacobians(x, q, zin, self.tss_ras(x, q), params=params)
        )
        d_ras = np.zeros(x.size)
        d_ras[self._cols] = self._gain(q) * self._factor * np.exp(x[self._cols])
        b_ras = b_ras.reshape(-1, 1)
        return j_x + b_ras @ d_ras[None, :], np.zeros_like(b_ras), b_zin, b_theta

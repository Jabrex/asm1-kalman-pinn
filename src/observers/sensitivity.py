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

    q_in: np.ndarray     # (n,)   m3/d
    z_in: np.ndarray     # (n, 14) g/m3
    tss_ras: np.ndarray  # (n,)   g SS/m3

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
) -> TangentLinear:
    """Propagators of the linearised reduced model along ``x_traj``.

    ``model`` needs ``jacobians(x70, q, zin, tss, params=None)`` returning
    ``(J_x, B_ras, B_zin, B_theta)`` in log-state coordinates (G3 contract).
    ``inputs``: ``"none"``; ``"zin_rel"`` (relative perturbation of each of the
    14 influent components, ``B_zin * z_in``); or ``"theta"`` (log multipliers
    on ``params``, in ``params`` order).
    """
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

    def generator(tt: float) -> np.ndarray:
        q, zin, tss = u_traj.at(t, tt)
        j_x, _b_ras, b_zin, b_theta = model.jacobians(spline(tt), q, zin, tss, params=call_params)
        a = np.zeros((d, d))
        a[:N_STATE, :N_STATE] = _np(j_x)
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


# -- measurements -------------------------------------------------------------
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

"""EKF, RTS smoother and IEKS on the reduced model. State x = log z (+ log kinetic
multipliers); measurements log(H_j z); Van Loan noise with scaling and squaring."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Sequence

import numpy as np
import torch
from scipy.linalg import LinAlgError, cho_factor, cho_solve, expm, solve

from ..data.sensors import SensorChannel
from ..models.losses import ObservationOperator
from .reduced_model import InputSeries, ReducedPlantModel, Z_FLOOR

Q_GRID: tuple[float, ...] = (0.003, 0.01, 0.03, 0.1, 0.3)  # per sqrt(day), log units
MAD_TO_SD = 1.4826
Q_CRITERIA = ("innovation", "predictive")


@dataclass(frozen=True)
class EkfConfig:
    substeps: int = 3
    q_soluble: float = 0.03
    q_particulate: float = 0.01
    q_theta: float = 1e-3            # random walk of log multipliers, per sqrt(day)
    r_mode: str = "data"             # "data" (MAD of log differences) or "spec"
    r_floor: float = 0.01            # log-unit sd floor per channel
    augment: tuple[str, ...] = ()    # kinetic constants appended as log multipliers
    theta_prior_sd: float = 0.693    # ln 2, as the PINN kinetic prior
    ieks_iterations: int = 5         # cap on IEKS relinearisations
    ieks_tol: float = 1e-4           # max |change| of the smoothed log state
    ras_filter_window: int = 4
    divergence_nis_factor: float = 10.0


@dataclass
class EkfResult:
    t: np.ndarray
    x_pred: np.ndarray            # (n, d)
    x_filt: np.ndarray            # (n, d)
    P_pred: np.ndarray | None     # (n, d, d); None when store=False
    P_filt: np.ndarray | None
    P_filt_diag: np.ndarray       # (n, d)
    Phi: np.ndarray | None        # (n-1, d, d)
    innovations: np.ndarray       # (n, m)
    nis: np.ndarray               # (n,)
    loglik: float                 # -inf if diverged
    diverged: bool
    r: np.ndarray                 # (m,) log-unit measurement variances
    ras_log_var: float
    names: tuple[str, ...]
    channels: tuple[str, ...]
    cfg: EkfConfig
    seconds: float

    def z(self) -> np.ndarray:
        return np.exp(self.x_filt[:, :70]).reshape(-1, 5, 14)

    def multipliers(self) -> np.ndarray:
        return np.exp(self.x_filt[:, 70:])


@dataclass
class SmoothResult:
    t: np.ndarray
    x: np.ndarray
    P: np.ndarray
    names: tuple[str, ...]

    def z(self) -> np.ndarray:
        return np.exp(self.x[:, :70]).reshape(-1, 5, 14)

    def multipliers(self) -> np.ndarray:
        return np.exp(self.x[:, 70:])


def channel_weights(model: ReducedPlantModel, channels: Sequence[SensorChannel]) -> np.ndarray:
    """Linear measurement rows ``(m, 70)``: channel j reads ``W[j] @ z``."""
    op = ObservationOperator(model.plant, tuple(channels))
    eye = torch.eye(model.size, dtype=torch.float64).reshape(model.size, model.n_tanks,
                                                             model.n_components)
    with torch.no_grad():
        return op(eye).numpy().T.copy()


def _log_obs(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    return np.log(np.maximum(y, 1e-3 * np.maximum(np.median(np.abs(y), axis=0), 1e-12)))


def estimate_r_from_data(y_obs_train: np.ndarray, r_floor: float = 0.01) -> np.ndarray:
    """``(1.4826 * MAD(diff(log y)))^2 / 2`` per channel, floored at ``r_floor^2``."""
    y = np.asarray(y_obs_train, dtype=float)
    d = np.diff(_log_obs(y[:, None] if y.ndim == 1 else y), axis=0)
    mad = np.median(np.abs(d - np.median(d, axis=0)), axis=0)
    return np.maximum((MAD_TO_SD * mad) ** 2 / 2.0, r_floor ** 2)


def ras_log_variance(raw_ras_train: np.ndarray, window: int, r_floor: float = 0.01) -> float:
    """Log-unit variance of the trailing-filtered RAS noise, ``sigma_hat^2 / window``."""
    raw = np.asarray(raw_ras_train, dtype=float).reshape(-1, 1)
    return float(estimate_r_from_data(raw, r_floor)[0]) / max(int(window), 1)


def _process_intensity(model: ReducedPlantModel, cfg: EkfConfig, n_aug: int) -> np.ndarray:
    soluble = set(int(i) for i in model.plant.i_soluble)
    per = np.array([cfg.q_soluble if j in soluble else cfg.q_particulate
                    for j in range(model.n_components)])
    return np.diag(np.concatenate([np.tile(per ** 2, model.n_tanks),
                                   np.full(n_aug, cfg.q_theta ** 2)]))


def van_loan(a: np.ndarray, qc: np.ndarray, h: float) -> np.ndarray:
    """``int_0^h e^{As} Qc e^{A's} ds`` by Van Loan on ``h/2^k`` (||A||_1 h/2^k <= 0.5), then doubling."""
    d = a.shape[0]
    norm = float(np.linalg.norm(a, 1)) * h
    k = 0 if norm <= 0.5 else int(np.ceil(np.log2(norm / 0.5)))
    hs = h / (2 ** k)
    m = np.zeros((2 * d, 2 * d))
    m[:d, :d], m[:d, d:], m[d:, d:] = -a * hs, qc * hs, a.T * hs
    e = expm(m)
    phi = e[d:, d:].T
    qd = phi @ e[:d, d:]
    for _ in range(k):
        qd = phi @ qd @ phi.T + qd
        phi = phi @ phi
    return 0.5 * (qd + qd.T)


def run_ekf(model: ReducedPlantModel, t, y_obs, channels: Sequence[SensorChannel], q_in, z_in,
            tss_ras, z0_mean, z0_rel_std, cfg: EkfConfig, *, ras_log_var: float = 0.0,
            r: np.ndarray | None = None, nominal: np.ndarray | None = None,
            store: bool = True) -> EkfResult:
    """EKF over ``t`` with Joseph updates.

    ``tss_ras`` is the (trailing-filtered) input fed to the model; its noise
    enters as ``Qc += B B' ras_log_var window dt`` (a trailing average keeps
    the zero-frequency spectral density ``var_raw dt``). ``r`` overrides the
    data-based R; ``nominal`` (n, d) fixes the linearisation (IEKS).
    """
    started = time.perf_counter()
    series = InputSeries(t, q_in, z_in, tss_ras)
    names = tuple(cfg.augment)
    p = len(names)
    base = np.array([model.parameters[nm] for nm in names]) if p else None
    d, n = model.size + p, series.n
    w = channel_weights(model, channels)
    y = np.asarray(y_obs, dtype=float).reshape(n, -1)
    m = y.shape[1]
    logy = _log_obs(y)
    rvec = estimate_r_from_data(y, cfg.r_floor) if r is None else np.asarray(r, float).reshape(m)
    R = np.diag(rvec)
    qc0 = _process_intensity(model, cfg, p)
    window = max(int(cfg.ras_filter_window), 1)
    x = np.concatenate([np.log(np.maximum(np.asarray(z0_mean, float).reshape(-1), Z_FLOOR)),
                        np.zeros(p)])
    P = np.diag(np.concatenate([np.log1p(np.asarray(z0_rel_std, float).reshape(-1) ** 2),
                                np.full(p, cfg.theta_prior_sd ** 2)]))
    x_pred, x_filt = np.full((n, d), np.nan), np.full((n, d), np.nan)
    P_pred = np.full((n, d, d), np.nan) if store else None
    P_filt = np.full((n, d, d), np.nan) if store else None
    Phis = np.full((n - 1, d, d), np.nan) if store else None
    P_diag, innov, nis = np.full((n, d), np.nan), np.full((n, m), np.nan), np.full(n, np.nan)
    loglik, diverged, eye, P_prev = 0.0, False, np.eye(d), P
    try:
        for k in range(n):
            if k > 0:
                lin = x_filt[k - 1] if nominal is None else nominal[k - 1]
                s_next, phi, a0, b_ras = model.step(lin, series.t[k - 1], series.dt, series, names,
                                                    base, cfg.substeps, input_jacobian=True)
                x = s_next + phi @ (x_filt[k - 1] - lin)
                qc = qc0 + (b_ras @ b_ras.T) * ras_log_var * window * series.dt
                P = phi @ P_prev @ phi.T + van_loan(a0, qc, series.dt)
                P = 0.5 * (P + P.T)
                if store:
                    Phis[k - 1] = phi
            x_pred[k] = x
            if store:
                P_pred[k] = P
            lin = x if nominal is None else nominal[k]
            z = np.exp(lin[: model.size])
            yp = w @ z
            H = np.zeros((m, d))
            H[:, : model.size] = (w * z[None, :]) / yp[:, None]
            nu = logy[k] - np.log(yp) - H @ (x - lin)
            S = H @ P @ H.T + R
            c = cho_factor(0.5 * (S + S.T))
            K = cho_solve(c, H @ P).T
            x = x + K @ nu
            ikh = eye - K @ H
            P = ikh @ P @ ikh.T + K @ R @ K.T
            P = 0.5 * (P + P.T)
            nis[k] = float(nu @ cho_solve(c, nu))
            loglik += -0.5 * (nis[k] + 2.0 * np.sum(np.log(np.diag(c[0]))) + m * np.log(2.0 * np.pi))
            innov[k], x_filt[k], P_diag[k] = nu, x, np.diag(P)
            if store:
                P_filt[k] = P
            P_prev = P
            if not np.all(np.isfinite(x)):
                raise FloatingPointError("non-finite state at sample %d" % k)
    except (LinAlgError, np.linalg.LinAlgError, FloatingPointError, ValueError):
        diverged = True
    if not diverged and np.nanmedian(nis) > cfg.divergence_nis_factor * m:
        diverged = True
    return EkfResult(t=series.t, x_pred=x_pred, x_filt=x_filt, P_pred=P_pred, P_filt=P_filt,
                     P_filt_diag=P_diag, Phi=Phis, innovations=innov, nis=nis,
                     loglik=float(loglik) if not diverged else -np.inf, diverged=diverged, r=rvec,
                     ras_log_var=float(ras_log_var), names=names,
                     channels=tuple(c.name for c in channels), cfg=cfg,
                     seconds=time.perf_counter() - started)


def rts_smooth(result: EkfResult) -> SmoothResult:
    """Rauch-Tung-Striebel smoother on a stored filter pass."""
    if result.P_filt is None or result.Phi is None or result.P_pred is None:
        raise ValueError("rts_smooth needs a run_ekf result with store=True")
    xs, Ps = result.x_filt.copy(), result.P_filt.copy()
    for k in range(len(result.t) - 2, -1, -1):
        pp, rhs = result.P_pred[k + 1], result.Phi[k] @ result.P_filt[k]
        try:
            gain = solve(pp, rhs, assume_a="pos").T
        except LinAlgError:
            gain = solve(pp, rhs, assume_a="sym").T
        xs[k] = result.x_filt[k] + gain @ (xs[k + 1] - result.x_pred[k + 1])
        pk = result.P_filt[k] + gain @ (Ps[k + 1] - pp) @ gain.T
        Ps[k] = 0.5 * (pk + pk.T)
    return SmoothResult(t=result.t, x=xs, P=Ps, names=result.names)


@dataclass
class IeksResult:
    smooth: SmoothResult
    filter: EkfResult
    iterations: int
    converged: bool
    changes: list[float] = field(default_factory=list)


@dataclass
class QSelection:
    q_soluble: float
    q_particulate: float
    criterion: str
    on_grid_edge: bool
    table: list[dict]


def ieks(model: ReducedPlantModel, t, y_obs, channels, q_in, z_in, tss_ras, z0_mean, z0_rel_std,
         cfg: EkfConfig, *, ras_log_var: float = 0.0, r: np.ndarray | None = None,
         first: SmoothResult | None = None) -> IeksResult:
    """Iterated EKS: refilter linearised on the last smoothed trajectory until
    ``max |dx| < cfg.ieks_tol`` (log units) or ``cfg.ieks_iterations`` passes."""
    args = (model, t, y_obs, channels, q_in, z_in, tss_ras, z0_mean, z0_rel_std, cfg)
    if first is None:
        res = run_ekf(*args, ras_log_var=ras_log_var, r=r)
        if res.diverged:
            return IeksResult(SmoothResult(res.t, res.x_filt, res.P_filt, res.names), res, 0, False)
        first = rts_smooth(res)
    current, changes, res = first, [], None
    for it in range(1, int(cfg.ieks_iterations) + 1):
        res = run_ekf(*args, ras_log_var=ras_log_var, r=r, nominal=current.x)
        if res.diverged:
            return IeksResult(current, res, it, False, changes)
        new = rts_smooth(res)
        changes.append(float(np.max(np.abs(new.x - current.x))))
        current = new
        if changes[-1] < cfg.ieks_tol:
            return IeksResult(current, res, it, True, changes)
    return IeksResult(current, res, int(cfg.ieks_iterations), False, changes)


def forecast(model: ReducedPlantModel, x_start, t, q_in, z_in, tss_ras,
             names: Sequence[str] = (), substeps: int = 3) -> np.ndarray:
    """Open-loop mean ``(n, 5, 14)`` from the log state at ``t[0]``; with ``names`` the
    trailing entries of ``x_start`` are log multipliers on those constants."""
    x_start = np.asarray(x_start, dtype=float)
    params = None
    if names:
        logm = x_start[model.size: model.size + len(names)]
        params = {nm: model.parameters[nm] * float(np.exp(lm)) for nm, lm in zip(names, logm)}
    return model.integrate_expo(np.exp(x_start[: model.size]), t, q_in, z_in, tss_ras,
                                params=params, substeps=substeps)


def predictive_score(model: ReducedPlantModel, result: EkfResult, y_obs, channels, q_in, z_in,
                     tss_ras, horizon_steps: int = 24, stride: int = 24) -> float:
    """Mean R-normalised squared log error of open-loop predictions of the measured
    channels, ``horizon_steps`` ahead from every ``stride``-th filtered state."""
    if result.diverged:
        return float("inf")
    series = InputSeries(result.t, q_in, z_in, tss_ras)
    w = channel_weights(model, channels)
    logy = _log_obs(np.asarray(y_obs, dtype=float).reshape(series.n, -1))
    names = result.names
    base = np.array([model.parameters[nm] for nm in names]) if names else None
    total, count = 0.0, 0
    for k in range(0, series.n - horizon_steps, stride):
        s = result.x_filt[k].copy()
        for j in range(horizon_steps):
            s = model.step(s, series.t[k + j], series.dt, series, names, base,
                           result.cfg.substeps)[0]
            yp = w @ np.exp(s[: model.size])
            total += float(np.sum((logy[k + j + 1] - np.log(yp)) ** 2 / result.r))
            count += len(result.r)
    return total / max(count, 1)


def tune_q(model: ReducedPlantModel, t, y_obs, channels, q_in, z_in, tss_ras, z0_mean,
           z0_rel_std, cfg: EkfConfig, *, ras_log_var: float = 0.0, r: np.ndarray | None = None,
           grid: Sequence[float] = Q_GRID, criterion: str = "innovation",
           horizon_steps: int = 24, stride: int = 24) -> QSelection:
    """Pick ``(q_soluble, q_particulate)`` on ``grid x grid`` from measured channels only
    (pass days 0-12). Both scores are kept in ``table``; ``criterion`` decides."""
    if criterion not in Q_CRITERIA:
        raise ValueError("criterion must be one of %s, got %r" % (Q_CRITERIA, criterion))
    table: list[dict] = []
    for qs in grid:
        for qp in grid:
            c = replace(cfg, q_soluble=float(qs), q_particulate=float(qp))
            res = run_ekf(model, t, y_obs, channels, q_in, z_in, tss_ras, z0_mean, z0_rel_std, c,
                          ras_log_var=ras_log_var, r=r, store=False)
            table.append({
                "q_soluble": float(qs), "q_particulate": float(qp), "loglik": res.loglik,
                "nis_mean": None if res.diverged else float(np.nanmean(res.nis)),
                "predictive_score": predictive_score(model, res, y_obs, channels, q_in, z_in,
                                                     tss_ras, horizon_steps, stride),
                "diverged": bool(res.diverged),
            })
    best = (max(table, key=lambda row: row["loglik"]) if criterion == "innovation"
            else min(table, key=lambda row: row["predictive_score"]))
    top = max(grid)
    return QSelection(best["q_soluble"], best["q_particulate"], criterion,
                      bool(best["q_soluble"] == top or best["q_particulate"] == top), table)

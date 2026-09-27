"""Reduced reactor train: ``Asm1Loss.plant_rhs`` itself on CPU/float64, shared by the
observers and the PINN residual. Not modelled: the clarifier soluble lag
(``recycle_solubles`` is for the attribution gate only) and settler solids
(``ras_mode="ideal_settler"``: TSS_ras = TSS_5 (Q_in + Q_r) / (Q_r + Q_w))."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from ..asm1.plant import Bsm1Plant
from ..models.losses import Asm1Loss, ObservationOperator
from ..train.run import TARGET_CHANNELS

Z_FLOOR = 1e-12
RAS_MODES = ("measured", "ideal_settler")
_F64 = torch.float64


def _t(x) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.to(dtype=_F64)
    return torch.as_tensor(np.asarray(x, dtype=float), dtype=_F64)


class InputSeries:
    """Known inputs on a uniform grid, linear in time."""

    def __init__(self, t, q_in, z_in, tss_ras, recycle_solubles=None) -> None:
        self.t = np.asarray(t, dtype=float).reshape(-1)
        if self.t.size < 2:
            raise ValueError("InputSeries needs at least two samples")
        dt = np.diff(self.t)
        if not np.allclose(dt, dt[0], rtol=1e-9, atol=1e-12):
            raise ValueError("inputs must sit on a uniform time grid")
        self.t0, self.dt, self.n = float(self.t[0]), float(dt[0]), int(self.t.size)
        self.q = np.asarray(q_in, dtype=float).reshape(self.n)
        self.z = np.asarray(z_in, dtype=float).reshape(self.n, -1)
        self.r = np.asarray(tss_ras, dtype=float).reshape(self.n)
        self.sol = (None if recycle_solubles is None
                    else np.asarray(recycle_solubles, dtype=float).reshape(self.n, -1))

    def at(self, tt: float):
        pos = min(max((float(tt) - self.t0) / self.dt, 0.0), self.n - 1.0)
        lo = min(int(np.floor(pos)), self.n - 2)
        w = pos - lo
        mix = lambda a: (1.0 - w) * a[lo] + w * a[lo + 1]
        return mix(self.q), mix(self.z), mix(self.r), None if self.sol is None else mix(self.sol)


class ReducedPlantModel:

    def __init__(self, plant: Bsm1Plant | None = None, ras_mode: str = "measured") -> None:
        if ras_mode not in RAS_MODES:
            raise ValueError("ras_mode must be one of %s, got %r" % (RAS_MODES, ras_mode))
        self.plant = plant if plant is not None else Bsm1Plant()
        self.ras_mode = ras_mode
        self.n_tanks, self.n_components = self.plant.cfg.n_tanks, self.plant.n_components
        self.size = self.n_tanks * self.n_components
        self.operator = ObservationOperator(self.plant, TARGET_CHANNELS)
        self.loss = Asm1Loss(self.plant, self.operator, np.ones(self.n_components),
                             np.ones(len(TARGET_CHANNELS)), torch.device("cpu"), _F64)
        self.parameters: dict[str, float] = dict(self.plant.vault.parameters)
        self.rate_parameters: tuple[str, ...] = self.plant.kinetics.rate_parameters
        self._i_sol = torch.as_tensor(self.plant.i_soluble, dtype=torch.long)
        self._i_tss = torch.as_tensor(self.plant.i_tss, dtype=torch.long)
        cfg = self.plant.cfg
        self._q_r, self._q_w, self._v1 = float(cfg.q_r), float(cfg.q_w), float(cfg.volumes[0])
        self._tss_factor = float(cfg.tss_factor)

    def _rhs_z(self, z, q, zin, tss, params=None, recycle_solubles=None):
        if self.ras_mode == "ideal_settler":
            tss5 = self._tss_factor * z[:, -1, :].index_select(-1, self._i_tss).sum(-1, keepdim=True)
            tss = tss5 * (q + self._q_r) / (self._q_r + self._q_w)
        dz = self.loss.plant_rhs(z, q, zin, tss, params=params)
        if recycle_solubles is None:
            return dz
        lag = self._q_r * (recycle_solubles - z[:, -1, :].index_select(-1, self._i_sol)) / self._v1
        corr = torch.zeros_like(dz[:, 0, :]).index_copy(-1, self._i_sol, lag)
        return dz + torch.cat([corr.unsqueeze(1), torch.zeros_like(dz[:, 1:, :])], dim=1)

    def rhs(self, z, q_in, z_in, tss_ras, params=None, *, recycle_solubles=None) -> torch.Tensor:
        z = _t(z).reshape(-1, self.n_tanks, self.n_components)
        n = z.shape[0]
        sol = None if recycle_solubles is None else _t(recycle_solubles).reshape(n, -1)
        return self._rhs_z(z, _t(q_in).reshape(n, 1), _t(z_in).reshape(n, -1),
                           _t(tss_ras).reshape(n, 1), params, sol)

    def rhs_log(self, x70, q, zin, tss, params=None) -> torch.Tensor:
        """Log-space field f(z)/z, z = exp(x)."""
        x = _t(x70)
        z = torch.exp(x.reshape(-1, self.size))
        dz = self.rhs(z, q, zin, tss, params).reshape(-1, self.size)
        return (dz / z).reshape(x.shape)

    def _g(self, x, q, zin, tss, logp, names, base, sol):
        params = ({name: base[i] * torch.exp(logp[i]) for i, name in enumerate(names)}
                  if names else None)
        z = torch.exp(x).reshape(1, self.n_tanks, self.n_components)
        s = None if sol is None else sol.reshape(1, -1)
        dz = self._rhs_z(z, q.reshape(1, 1), zin.reshape(1, -1), tss.reshape(1, 1), params, s)
        return dz.reshape(-1) / z.reshape(-1)

    def jacobians(self, x70, q, zin, tss, params: Mapping[str, float] | None = None):
        """``(J_x, B_ras, B_zin, B_theta)`` of rhs_log w.r.t. log state, TSS_ras, z_in and
        the log multipliers of ``params`` (shapes 70x70, 70x1, 70x14, 70xp)."""
        names = tuple(params) if params else ()
        base = _t([float(params[n]) for n in names]) if names else None
        x = _t(x70).reshape(self.size)
        q_t, zin_t, tss_t = _t([q]).reshape(1), _t(zin).reshape(-1), _t([tss]).reshape(1)
        logp = torch.zeros(len(names), dtype=_F64)

        def fn(xx, rr, zz, pp):
            return self._g(xx, q_t, zz, rr, pp, names, base, None)

        jac = torch.func.jacrev(fn, argnums=(0, 1, 2, 3) if names else (0, 1, 2))(
            x, tss_t, zin_t, logp)
        b_theta = (jac[3].numpy().reshape(self.size, len(names)) if names
                   else np.zeros((self.size, 0)))
        return (jac[0].numpy(), jac[1].numpy().reshape(self.size, 1),
                jac[2].numpy().reshape(self.size, -1), b_theta)

    def linearise(self, s, q, zin, tss, names: Sequence[str] = (), base=None, sol=None,
                  input_jacobian: bool = False):
        """``(field [g; 0], A, dg/dlog(TSS_ras) (d,1) or None)`` for ``s = [x, log m]``."""
        d, p = s.shape[0], len(names)
        x = _t(s[: self.size])
        logp = _t(s[self.size:]) if p else torch.zeros(0, dtype=_F64)
        q_t, zin_t, tss_t = _t([q]).reshape(1), _t(zin).reshape(-1), _t([tss]).reshape(1)
        base_t = _t(base) if p else None
        sol_t = None if sol is None else _t(sol)

        def fn(xx, rr, pp):
            g = self._g(xx, q_t, zin_t, rr, pp, names, base_t, sol_t)
            return g, g

        argnums = [0] + ([1] if input_jacobian else []) + ([2] if p else [])
        jac, g = torch.func.jacrev(fn, argnums=tuple(argnums), has_aux=True)(x, tss_t, logp)
        a = np.zeros((d, d))
        a[: self.size, : self.size] = jac[0].numpy()
        k, b_ras = 1, None
        if input_jacobian:
            b_ras = np.zeros((d, 1))
            b_ras[: self.size, 0] = jac[k].numpy().reshape(self.size) * float(tss)
            k += 1
        if p:
            a[: self.size, self.size:] = jac[k].numpy().reshape(self.size, p)
        field = np.zeros(d)
        field[: self.size] = g.detach().numpy()
        return field, a, b_ras

    def step(self, s, t0: float, h: float, series: InputSeries, names: Sequence[str] = (),
             base=None, substeps: int = 3, input_jacobian: bool = False):
        """Exponential Rosenbrock-Euler, ``s += hs phi1(hs A) g`` per sub-step (midpoint
        inputs, phi1 via the augmented exponential); returns (s_next, Phi, A0, b0)."""
        d = s.shape[0]
        hs = h / substeps
        phi = np.eye(d)
        a_first = b_first = None
        m = np.zeros((d + 1, d + 1))
        for j in range(substeps):
            q, zin, tss, sol = series.at(t0 + (j + 0.5) * hs)
            field, a, b_ras = self.linearise(s, q, zin, tss, names, base, sol,
                                             input_jacobian and j == 0)
            m[:d, :d], m[:d, d] = a * hs, field * hs
            e = expm(m)
            s = s + e[:d, d]
            phi = e[:d, :d] @ phi
            if j == 0:
                a_first, b_first = a, b_ras
        return s, phi, a_first, b_first

    def integrate_expo(self, z0, t, q_in, z_in, tss_ras, params: Mapping[str, float] | None = None,
                       substeps: int = 3, recycle_solubles=None) -> np.ndarray:
        series = InputSeries(t, q_in, z_in, tss_ras, recycle_solubles)
        names = tuple(params) if params else ()
        base = np.array([float(params[n]) for n in names]) if names else None
        s = np.concatenate([np.log(np.maximum(np.asarray(z0, dtype=float).reshape(-1), Z_FLOOR)),
                            np.zeros(len(names))])
        out = [s[: self.size].copy()]
        for k in range(series.n - 1):
            s = self.step(s, series.t[k], series.dt, series, names, base, substeps)[0]
            out.append(s[: self.size].copy())
        return np.exp(np.asarray(out)).reshape(-1, self.n_tanks, self.n_components)

    def integrate_bdf(self, z0, t, q_in, z_in, tss_ras, params: Mapping[str, float] | None = None,
                      *, recycle_solubles=None, rtol: float = 1e-8, atol: float = 1e-10) -> np.ndarray:
        """Reference open loop: scipy BDF with the autograd Jacobian."""
        series = InputSeries(t, q_in, z_in, tss_ras, recycle_solubles)
        shape = (1, self.n_tanks, self.n_components)

        def inputs(tt):
            q, zin, tss, sol = series.at(tt)
            return (_t([q]).reshape(1, 1), _t(zin).reshape(1, -1), _t([tss]).reshape(1, 1),
                    None if sol is None else _t(sol).reshape(1, -1))

        def f(tt, y):
            q, zin, tss, sol = inputs(tt)
            with torch.no_grad():
                return self._rhs_z(_t(y).reshape(shape), q, zin, tss, params, sol).reshape(-1).numpy()

        def jac(tt, y):
            q, zin, tss, sol = inputs(tt)
            fn = lambda yy: self._rhs_z(yy.reshape(shape), q, zin, tss, params, sol).reshape(-1)
            return torch.func.jacrev(fn)(_t(y)).numpy()

        sol = solve_ivp(f, (series.t[0], series.t[-1]), np.asarray(z0, dtype=float).reshape(-1),
                        method="BDF", t_eval=series.t, rtol=rtol, atol=atol, jac=jac)
        if not sol.success:
            raise RuntimeError("reduced-model BDF integration failed: %s" % sol.message)
        return sol.y.T.reshape(-1, self.n_tanks, self.n_components)

"""O1 CPU pre-check: does modelling the clarifier soluble lag cut the information-matched
open-loop Track B error by more than 50 %?

Variants of the recycle-soluble path in the reduced model:
  copy      - v1.1 G3 model: RAS solubles = tank-5 solubles (instant copy)
  layers6   - exact BSM1 clarifier soluble transport below and at the feed layer:
              feed layer CSTR (V = A z, flow Q_f) then 5 layers (flow Q_u), no reactions
  lump1     - one CSTR with the same mean residence time as layers6
Flows come from the measured Q_in and the fixed Q_r, Q_w; geometry from Bsm1Config.
Nothing here reads truth except the scoring and the A0 anchor file (oracle start).
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repository root
from scripts.make_baselines import ras_series, scenario_arrays, split_windows  # noqa: E402
from src.asm1.vault_loader import vault  # noqa: E402
from src.data.influent_views import apply_influent_knowledge  # noqa: E402
from src.eval.metrics import state_metrics, track_summary  # noqa: E402
from src.observers.reduced_model import InputSeries, ReducedPlantModel, _t  # noqa: E402

torch.set_num_threads(4)
F64 = torch.float64


def integrate(model, variant, z0, t, q_in, z_in, tss):
    cfg = model.plant.cfg
    n_sol = len(model.plant.i_soluble)
    v_layer = cfg.settler_area * cfg.settler_layer_height
    n_below = cfg.feed_layer - 1
    series = InputSeries(t, q_in, z_in, tss)
    s5_0 = np.asarray(z0, float)[-1, model.plant.i_soluble]
    n_lag = {"copy": 0, "layers6": 1 + n_below, "lump1": 1}[variant]
    y0 = np.concatenate([np.asarray(z0, float).reshape(-1), np.tile(s5_0, n_lag)])
    shape = (1, model.n_tanks, model.n_components)
    i_sol = torch.as_tensor(model.plant.i_soluble, dtype=torch.long)

    def field(tt, y):
        q, zin, r, _ = series.at(tt)
        z = y[:70].reshape(shape)
        qf, qu = q + cfg.q_r, cfg.q_r + cfg.q_w
        if variant == "copy":
            return model._rhs_z(z, _t([q]).reshape(1, 1), _t(zin).reshape(1, -1), _t([r]).reshape(1, 1)).reshape(-1)
        lag = y[70:].reshape(n_lag, n_sol)
        s5 = z[0, -1].index_select(-1, i_sol)
        if variant == "layers6":
            d = torch.zeros_like(lag)
            d[0] = qf * (s5 - lag[0]) / v_layer                  # feed layer, eq. 42-43 net
            for j in range(1, n_lag):
                d[j] = qu * (lag[j - 1] - lag[j]) / v_layer      # layers below the feed
            under = lag[-1]
        else:  # lump1: one CSTR with the layers6 mean residence time
            tau = v_layer / qf + n_below * v_layer / qu
            d = ((s5 - lag[0]) / tau).reshape(1, -1)
            under = lag[0]
        dz = model._rhs_z(z, _t([q]).reshape(1, 1), _t(zin).reshape(1, -1), _t([r]).reshape(1, 1),
                          None, under.reshape(1, -1))
        return torch.cat([dz.reshape(-1), d.reshape(-1)])

    def f(tt, y):
        with torch.no_grad():
            return field(tt, _t(y)).numpy()

    def jac(tt, y):
        return torch.func.jacrev(lambda yy: field(tt, yy))(_t(y)).numpy()

    sol = solve_ivp(f, (t[0], t[-1]), y0, method="BDF", t_eval=t, rtol=1e-8, atol=1e-10, jac=jac)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T[:, :70].reshape(-1, 5, 14)


def cell(name, data_dir, anchor, mode, window, sigma):
    raw = Path(data_dir)
    dry = scenario_arrays(raw, "dry")
    z0 = np.load(anchor)["z0_mean"]
    view = apply_influent_knowledge(dry["t"], dry["q_in"], dry["influent"], mode, vault())
    ras = ras_series(raw, "dry", sigma, window)
    truth = split_windows(dry["t"], dry["reactor"])
    model = ReducedPlantModel()
    out = {}
    for variant in ("copy", "layers6", "lump1"):
        t0 = time.perf_counter()
        traj = integrate(model, variant, z0, dry["t"], dry["q_in"], view, ras)
        pred = split_windows(dry["t"], traj)
        out[variant] = [track_summary(state_metrics(truth[w], pred[w]))["track_b_unmeasured"]["nrmse"]
                        for w in ("train", "holdout")]
        print("%-12s %-8s train %.4f holdout %.4f  (%.0f s)" % (name, variant, *out[variant],
                                                               time.perf_counter() - t0), flush=True)
    for variant in ("layers6", "lump1"):
        drop = [1 - a / b for a, b in zip(out[variant], out["copy"])]
        print("%-12s %-8s drop vs copy: train %.0f %% holdout %.0f %%" % (name, variant, 100 * drop[0], 100 * drop[1]))
    return out


if __name__ == "__main__":
    cell("K0-Ie-A0 s0", "results/raw", "results/v11/anchors/k000/A0.npz", "exact", 1, 0.0)
    cell("K0-Ie-A0 s10", "results/raw", "results/v11/anchors/k000/A0.npz", "exact", 4, 0.10)
    cell("K1-Ic-As s10", "results/raw_k100", "results/v11/anchors/k100/As.npz", "composite", 4, 0.10)

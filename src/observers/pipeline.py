"""Run the estimators on one dataset and one anchor, reading only what the PINN reads
(t, noisy obs, q_in, influent view) plus z0_mean/z0_rel_std. Holdout = forecast from
day 12, except ekf_online (assimilates days 12-14, more-information); rain = filtered
pass with the dry-window q and R (SI only)."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any, Mapping

import numpy as np

from ..data.influent_views import view_dataset
from ..data.sensors import CANDIDATE_CHANNELS, SENSOR_SET, ObservationDataset, SensorChannel
from ..train.curriculum import trailing_average
from ..train.run import RAS_CHANNEL, TARGET_CHANNELS
from .ekf import (EkfConfig, estimate_r_from_data, forecast, ieks, ras_log_variance, rts_smooth,
                  run_ekf, tune_q)
from .reduced_model import ReducedPlantModel

ESTIMATORS = ("ekf", "eks", "ieks", "ekf_aug", "eks_aug", "ekf_online")
MORE_INFORMATION = frozenset({"ekf_online"})
Q_MODES = ("tuned", "frozen", "fixed")


@dataclass(frozen=True)
class ObserverSpec:
    estimators: tuple[str, ...] = ("ekf", "eks", "ieks", "eks_aug", "ekf_online")
    influent_mode: str = "exact"
    ras_filter_window: int = 4
    ras_mode: str = "measured"
    train_end_day: float = 12.0
    holdout_days: tuple[float, float] = (12.0, 14.0)
    q_mode: str = "tuned"
    q_fixed: tuple[float, float] | None = None   # (q_soluble, q_particulate) for frozen/fixed
    q_criterion: str = "innovation"
    augment: tuple[str, ...] = ()
    q_theta: float = 1e-3
    target_channels: tuple[str, ...] | None = None
    r_mode: str = "data"
    r_floor: float = 0.01
    substeps: int = 3
    rain: bool = True


def resolve_channels(names: tuple[str, ...] | None) -> tuple[SensorChannel, ...]:
    """Target channels by name from SENSOR_SET + CANDIDATE_CHANNELS; None = TARGET_CHANNELS."""
    if names is None:
        return tuple(TARGET_CHANNELS)
    table = {c.name: c for c in tuple(SENSOR_SET) + tuple(CANDIDATE_CHANNELS)}
    for name in names:
        if name == RAS_CHANNEL:
            raise ValueError("%s is a model input, never an estimator target" % RAS_CHANNEL)
        if name not in table:
            raise KeyError("Unknown channel %r" % (name,))
    return tuple(table[n] for n in names)


def _spec_r(ds: ObservationDataset, channels, floor: float) -> np.ndarray:
    """Secondary row: R from the sensor specification instead of the data."""
    sig = np.array([c.sigma_override if getattr(c, "sigma_override", None) is not None
                    and ds.sigma > 0.0 else ds.sigma for c in channels], dtype=float)
    return np.maximum(np.log1p(sig ** 2), floor ** 2)


def run_estimators(dry: ObservationDataset, rain: ObservationDataset | None,
                   anchor: Mapping[str, Any], spec: ObserverSpec,
                   model: ReducedPlantModel | None = None) -> dict[str, dict]:
    """``{estimator: {"predictions": {"train", "holdout"[, "rain"]}, "info": {...}}}``."""
    unknown = sorted(set(spec.estimators) - set(ESTIMATORS))
    if unknown:
        raise ValueError("Unknown estimators %s; expected a subset of %s" % (unknown, ESTIMATORS))
    if spec.q_mode not in Q_MODES or (spec.q_mode != "tuned" and spec.q_fixed is None):
        raise ValueError("q_mode must be one of %s; frozen/fixed need q_fixed" % (Q_MODES,))
    if any(e.endswith("_aug") for e in spec.estimators) and not spec.augment:
        raise ValueError("augmented estimators need a non-empty ObserverSpec.augment")

    model = model if model is not None else ReducedPlantModel(ras_mode=spec.ras_mode)
    dry = view_dataset(dry, spec.influent_mode)
    channels = resolve_channels(spec.target_channels)
    names = list(dry.channels)
    cols, ras_col, w = [names.index(c.name) for c in channels], names.index(RAS_CHANNEL), int(spec.ras_filter_window)
    t = dry.t
    tr = t <= spec.train_end_day + 1e-9
    ho = (t >= spec.holdout_days[0] - 1e-9) & (t <= spec.holdout_days[1] + 1e-9)
    y, raw = dry.obs[:, cols], dry.obs[:, ras_col]
    ras_f = trailing_average(raw, w)
    z0, rel = np.asarray(anchor["z0_mean"], float), np.asarray(anchor["z0_rel_std"], float)
    r = (estimate_r_from_data(y[tr], spec.r_floor) if spec.r_mode == "data"
         else _spec_r(dry, channels, spec.r_floor))
    ras_var = ras_log_variance(raw[tr], w, spec.r_floor)
    base = EkfConfig(substeps=spec.substeps, q_theta=spec.q_theta, r_mode=spec.r_mode,
                     r_floor=spec.r_floor, ras_filter_window=w)
    common = (channels, dry.q_in[tr], dry.z_in[tr], ras_f[tr], z0, rel)

    started, sel = time.perf_counter(), None
    if spec.q_mode == "tuned":
        sel = tune_q(model, t[tr], y[tr], *common, base, ras_log_var=ras_var, r=r,
                     criterion=spec.q_criterion)
        qs, qp = sel.q_soluble, sel.q_particulate
    else:
        qs, qp = spec.q_fixed
    cfg = replace(base, q_soluble=float(qs), q_particulate=float(qp))
    shared = {"q_mode": spec.q_mode, "q_soluble": float(qs), "q_particulate": float(qp),
              "q_criterion": spec.q_criterion if sel else None,
              "q_on_grid_edge": sel.on_grid_edge if sel else None,
              "q_table": sel.table if sel else None, "q_tune_seconds": time.perf_counter() - started,
              "r": r.tolist(), "ras_log_var": ras_var, "channels": [c.name for c in channels]}

    def hold(x_start, aug=()):
        return forecast(model, x_start, t[ho], dry.q_in[ho], dry.z_in[ho], ras_f[ho], aug,
                        spec.substeps)

    def info(res, extra=None):
        ok = np.isfinite(res.nis).any()
        return {"loglik": res.loglik, "diverged": bool(res.diverged), "seconds": res.seconds,
                "nis_mean": float(np.nanmean(res.nis)) if ok else None,
                "nis_median": float(np.nanmedian(res.nis)) if ok else None,
                "n_channels": len(channels), **(extra or {})}

    out: dict[str, dict] = {}
    want = set(spec.estimators)
    if want & {"ekf", "eks", "ieks"}:
        res = run_ekf(model, t[tr], y[tr], *common, cfg, ras_log_var=ras_var, r=r)
        fc, sm = hold(res.x_filt[-1]), (rts_smooth(res) if not res.diverged else None)
        if "ekf" in want:
            out["ekf"] = {"predictions": {"train": res.z(), "holdout": fc}, "info": info(res)}
        if "eks" in want:
            out["eks"] = {"predictions": {"train": (sm or res).z(), "holdout": fc},
                          "info": info(res)}
        if "ieks" in want:
            it = ieks(model, t[tr], y[tr], *common, cfg, ras_log_var=ras_var, r=r, first=sm)
            out["ieks"] = {"predictions": {"train": it.smooth.z(), "holdout": hold(it.smooth.x[-1])},
                           "info": info(it.filter, {"ieks_iterations": it.iterations,
                                                    "ieks_converged": it.converged,
                                                    "ieks_changes": it.changes})}
    if want & {"ekf_aug", "eks_aug"}:
        res_a = run_ekf(model, t[tr], y[tr], *common, replace(cfg, augment=tuple(spec.augment)),
                        ras_log_var=ras_var, r=r)
        sm_a = rts_smooth(res_a) if not res_a.diverged else None
        fc_a = hold(res_a.x_filt[-1], tuple(spec.augment))
        mult = {"names": list(spec.augment), "filtered_end": res_a.multipliers()[-1].tolist(),
                "filtered_end_log_sd": np.sqrt(res_a.P_filt_diag[-1, 70:]).tolist(),
                "smoothed_start": sm_a.multipliers()[0].tolist() if sm_a else None}
        if "ekf_aug" in want:
            out["ekf_aug"] = {"predictions": {"train": res_a.z(), "holdout": fc_a},
                              "info": info(res_a, {"multipliers": mult})}
        if "eks_aug" in want:
            out["eks_aug"] = {"predictions": {"train": (sm_a or res_a).z(), "holdout": fc_a},
                              "info": info(res_a, {"multipliers": mult})}
    if "ekf_online" in want:
        span = t <= spec.holdout_days[1] + 1e-9
        res_o = run_ekf(model, t[span], y[span], channels, dry.q_in[span], dry.z_in[span],
                        ras_f[span], z0, rel, cfg, ras_log_var=ras_var, r=r, store=False)
        z_o = res_o.z()
        out["ekf_online"] = {"predictions": {"train": z_o[: int(tr.sum())], "holdout": z_o[ho[span]]},
                             "info": info(res_o)}
    if spec.rain and rain is not None:
        rv = view_dataset(rain, spec.influent_mode)
        rn = list(rv.channels)
        res_r = run_ekf(model, rv.t, rv.obs[:, [rn.index(c.name) for c in channels]], channels,
                        rv.q_in, rv.z_in, trailing_average(rv.obs[:, rn.index(RAS_CHANNEL)], w),
                        z0, rel, cfg, ras_log_var=ras_var, r=r, store=False)
        for entry in out.values():
            entry["predictions"]["rain"] = res_r.z()
            entry["info"]["rain_diverged"] = bool(res_r.diverged)
    for name, entry in out.items():
        entry["info"].update(shared, estimator=name,
                             information="more-information" if name in MORE_INFORMATION
                             else "same-information",
                             rain_information="more-information (rain data assimilated)")
    return out

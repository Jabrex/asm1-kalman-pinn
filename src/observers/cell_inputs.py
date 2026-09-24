"""Per-cell inputs of the v1.1 model-based estimator grid (group G6).

One YAML file under configs/observers/ describes one cell. This module owns the
schema of those files (validate_cell_config), the run-directory naming that
src.eval.report.collect_runs and the regime map read, and the two ways a cell
may change what an estimator receives: another channel set, or an ideal-settler
return-sludge input. Every function works on what a plant operator has: the
noisy sensor columns, the known flows and the channel names. Nothing here names
the hidden truth arrays or the noise-free observations, and tests/test_leakage.py
scans this module with the training rules.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from ..asm1.plant import Bsm1Config
from ..data.sensors import CANDIDATE_CHANNELS, SENSOR_SET, ObservationDataset, SensorChannel
from ..train.curriculum import trailing_average
from .ekf import Q_GRID

RAS_CHANNEL = "TSS_ras"
TANK5_TSS_CHANNEL = "TSS_tank5"
DEFAULT_TARGETS: tuple[str, ...] = tuple(c.name for c in SENSOR_SET if c.kind != "tss_underflow")
ESTIMATORS = ("ekf", "eks", "ieks", "ekf_aug", "eks_aug", "ekf_online")
Q_MODES = ("tuned", "frozen")
INFLUENT_MODES = ("exact", "composite", "composite_biased")
RAS_INPUTS = ("filtered", "ideal_settler")
REQUIRED_KEYS = (
    "cell", "out_dir", "data_dir", "anchor_file", "influent_mode", "ras_filter_window",
    "sigmas", "realisations", "estimators", "q_mode",
)
DEFAULTS: dict[str, Any] = {
    "ras_input": "filtered",
    "channels": "default",
    "q_grid": list(Q_GRID),  # seven points since gate D4 (2026-09-24)
    "r_floor": 0.01,
    "tune_window_days": [0.0, 12.0],
    "augment": [],
    "theta_prior_sd": 0.693,
    "q_theta": 1e-3,
    "frozen_q_from": None,
}


def sigma_tag(sigma: float) -> str:
    return ("%.2f" % sigma).replace(".", "p")


def validate_cell_config(raw: Mapping[str, Any], source: str = "<dict>") -> dict[str, Any]:
    """Fill defaults and refuse anything the grid does not define."""
    def fail(message: str) -> None:
        raise ValueError("%s: %s" % (source, message))

    unknown = sorted(set(raw) - set(REQUIRED_KEYS) - set(DEFAULTS))
    if unknown:
        fail("unknown keys %s" % unknown)
    missing = [k for k in REQUIRED_KEYS if k not in raw]
    if missing:
        fail("missing keys %s" % missing)
    cfg = {k: (list(v) if isinstance(v, list) else v) for k, v in DEFAULTS.items()}
    cfg.update(raw)

    if Path(cfg["out_dir"]).name != cfg["cell"]:
        fail("out_dir %r must end in the cell id %r" % (cfg["out_dir"], cfg["cell"]))
    if cfg["influent_mode"] not in INFLUENT_MODES:
        fail("influent_mode %r not in %s" % (cfg["influent_mode"], INFLUENT_MODES))
    if cfg["ras_input"] not in RAS_INPUTS:
        fail("ras_input %r not in %s" % (cfg["ras_input"], RAS_INPUTS))
    est = list(cfg["estimators"])
    if not est or len(set(est)) != len(est) or any(e not in ESTIMATORS for e in est):
        fail("estimators %r must be distinct members of %s" % (est, ESTIMATORS))
    if not cfg["q_mode"] or any(q not in Q_MODES for q in cfg["q_mode"]):
        fail("q_mode %r must be members of %s" % (cfg["q_mode"], Q_MODES))
    if int(cfg["ras_filter_window"]) < 1:
        fail("ras_filter_window must be >= 1")
    if not cfg["sigmas"] or any(float(s) < 0.0 for s in cfg["sigmas"]):
        fail("sigmas %r must be non-negative and non-empty" % (cfg["sigmas"],))
    if not cfg["realisations"] or any(int(r) < 0 for r in cfg["realisations"]):
        fail("realisations %r must be non-negative and non-empty" % (cfg["realisations"],))
    if not cfg["q_grid"] or any(float(q) <= 0.0 for q in cfg["q_grid"]):
        fail("q_grid must hold positive values")
    if "frozen" in cfg["q_mode"] and not cfg["frozen_q_from"]:
        fail("q_mode 'frozen' needs frozen_q_from")
    if any(e.endswith("_aug") for e in est) and not cfg["augment"]:
        fail("augmented estimators need a non-empty augment list")
    return cfg


def load_cell_config(path: str | Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("%s: expected a mapping" % path)
    return validate_cell_config(raw, source=str(path))


def resolve_channels(spec: str | Sequence[str], available: Sequence[str]) -> tuple[SensorChannel, ...]:
    """Measured channels of a cell; ``"default"`` is the seven v1.0 targets."""
    catalogue = {c.name: c for c in (*SENSOR_SET, *CANDIDATE_CHANNELS)}
    names = DEFAULT_TARGETS if spec == "default" else tuple(spec)
    if not names:
        raise ValueError("a cell needs at least one measured channel")
    if len(set(names)) != len(names):
        raise ValueError("duplicate channels in %s" % (names,))
    for name in names:
        if name == RAS_CHANNEL:
            raise ValueError("TSS_ras is an input to the recycle reconstruction, never a measured channel")
        if name not in catalogue:
            raise ValueError("unknown channel %r" % name)
        if name not in available:
            raise ValueError("channel %r is not in this dataset (%s)" % (name, ", ".join(available)))
    return tuple(catalogue[n] for n in names)


def ras_input(obs: np.ndarray, channels: Sequence[str], q_in: np.ndarray, mode: str, window: int,
              plant_cfg: Bsm1Config | None = None) -> np.ndarray:
    """Return-sludge TSS series fed to the reduced model.

    ``filtered``: the RAS probe after the trailing moving average (same filter
    as the PINN). ``ideal_settler``: no RAS probe; the filtered tank-5 TSS times
    (Q_in + Q_r) / (Q_r + Q_w), the underflow mass balance of a settler that
    loses no solids to the effluent.
    """
    index = {name: i for i, name in enumerate(channels)}
    obs = np.asarray(obs, dtype=float)
    if mode == "filtered":
        return trailing_average(obs[:, index[RAS_CHANNEL]], int(window))
    if mode == "ideal_settler":
        cfg = plant_cfg or Bsm1Config()
        tss5 = trailing_average(obs[:, index[TANK5_TSS_CHANNEL]], int(window))
        return tss5 * (np.asarray(q_in, dtype=float) + cfg.q_r) / (cfg.q_r + cfg.q_w)
    raise ValueError("unknown ras_input %r; expected one of %s" % (mode, RAS_INPUTS))


def observer_model_name(estimator: str, q_mode: str) -> str:
    if estimator not in ESTIMATORS or q_mode not in Q_MODES:
        raise ValueError("unknown estimator/q_mode %r/%r" % (estimator, q_mode))
    return estimator if q_mode == "tuned" else "%s_frozenq" % estimator


def run_dir_name(estimator: str, q_mode: str, sigma: float, realisation: int = 0) -> str:
    name = "%s_sigma%s" % (observer_model_name(estimator, q_mode), sigma_tag(float(sigma)))
    return name + ("_r%02d" % int(realisation) if realisation else "")


def expected_run_dirs(cfg: Mapping[str, Any]) -> list[Path]:
    out: list[Path] = []
    for sigma in cfg["sigmas"]:
        for realisation in cfg["realisations"]:
            for estimator in cfg["estimators"]:
                for q_mode in cfg["q_mode"]:
                    out.append(Path(cfg["out_dir"]) / run_dir_name(estimator, q_mode, float(sigma), int(realisation)))
    return out


def obs_path(data_dir: str | Path, sigma: float, realisation: int = 0, scenario: str = "dry") -> Path:
    name = "obs_%s_sigma%s" % (scenario, sigma_tag(float(sigma)))
    if realisation:
        name += "_r%02d" % int(realisation)
    return Path(data_dir) / (name + ".npz")


@dataclass
class CellInputs:
    """What every estimator of one (cell, sigma, realisation) receives, days 0-14."""

    t: np.ndarray               # (n,)
    train: np.ndarray           # (n,) bool, t <= train_end_day
    q_in: np.ndarray            # (n,)
    z_in: np.ndarray            # (n, 14) after the influent view
    y_obs: np.ndarray           # (n, m) noisy measured channels, cell order
    channels: tuple[SensorChannel, ...]
    tss_ras: np.ndarray         # (n,) return-sludge input after the filter or the settler rule
    z0_mean: np.ndarray         # (5, 14) anchor mean
    z0_rel_std: np.ndarray      # (5, 14) anchor relative sd


def prepare_cell_inputs(cfg: Mapping[str, Any], sigma: float, realisation: int = 0,
                        train_end_day: float = 12.0) -> CellInputs:
    from ..data.influent_views import view_dataset

    ds = view_dataset(ObservationDataset.load(obs_path(cfg["data_dir"], sigma, realisation)),
                      cfg["influent_mode"])
    channels = resolve_channels(cfg["channels"], ds.channels)
    cols = [ds.channels.index(c.name) for c in channels]
    with np.load(cfg["anchor_file"], allow_pickle=False) as anchor:
        z0_mean = np.array(anchor["z0_mean"], dtype=float)
        z0_rel_std = np.array(anchor["z0_rel_std"], dtype=float)
    t = np.asarray(ds.t, dtype=float)
    return CellInputs(
        t=t,
        train=t <= train_end_day + 1e-9,
        q_in=np.asarray(ds.q_in, dtype=float),
        z_in=np.asarray(ds.z_in, dtype=float),
        y_obs=np.asarray(ds.obs, dtype=float)[:, cols],
        channels=channels,
        tss_ras=ras_input(ds.obs, ds.channels, ds.q_in, cfg["ras_input"], int(cfg["ras_filter_window"])),
        z0_mean=z0_mean,
        z0_rel_std=z0_rel_std,
    )


def numerics_summary(nis: np.ndarray, estimates: np.ndarray, n_measured: int) -> dict[str, Any]:
    """Filter diagnostics of the days 0-12 pass, the only keys the numerics gate reads."""
    nis = np.asarray(nis, dtype=float)
    finite = bool(np.isfinite(np.asarray(estimates, dtype=float)).all() and np.isfinite(nis).all())
    mean = float(np.mean(nis)) if finite and nis.size else float("inf")
    return {"finite": finite, "nis_mean": mean, "n_measured": int(n_measured)}


def read_frozen_q(path: str | Path) -> dict[str, float]:
    """The q pair selected at K0-Ie-A0, sigma 0.10 (the frozen-q secondary row).

    Accepts the G3 ``--freeze-q`` layout (pair at top level) and the nested
    ``selected_q`` layout of a G6 run summary.
    """
    info = json.loads(Path(path).read_text(encoding="utf-8"))
    q = info["selected_q"] if "selected_q" in info else info
    return {"q_soluble": float(q["q_soluble"]), "q_particulate": float(q["q_particulate"])}

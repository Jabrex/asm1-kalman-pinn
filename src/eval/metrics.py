"""Evaluation metrics, split into the two tracks the benchmark reports.

Track A - measured components
    ``S_O``, ``S_NH``, ``S_NO``. All four models have a direct training signal
    here, so this is the apples-to-apples comparison.

Track B - never-measured components
    The remaining eleven: ``S_I``, ``S_S``, ``X_I``, ``X_S``, ``X_B_H``,
    ``X_B_A``, ``X_P``, ``S_ND``, ``X_ND``, ``S_ALK``, ``S_N2``. Five of them
    (the solids) are constrained only as a weighted sum through the two TSS
    channels; none is individually identified by any sensor. The physics-free
    baselines have no training signal at all here, which is the measurement the
    benchmark exists to make.

Effluent Quality Index
----------------------
``effluent_quality_index`` implements BSM1 eq. 75-79 with the Table 10 weights.
It is reported for the *ground-truth* trajectory as a description of the
generated dataset. It is deliberately not a model metric: the models span the
reactor train only, and the effluent composition lives downstream of the
clarifier, so scoring a model on it would require feeding it settler ground
truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..asm1.plant import Bsm1Plant
from ..asm1.vault_loader import vault
from ..data.sensors import observed_components, unobserved_components

#: BSM1 Table 10 - pollution-unit weighting factors for the EQI.
EQI_WEIGHTS = {"TSS": 2.0, "COD": 1.0, "NKj": 30.0, "NO": 10.0, "BOD5": 2.0}
#: BSM1 Table 9 - effluent quality limits.
EFFLUENT_LIMITS = {"N_tot": 18.0, "COD": 100.0, "S_NH": 4.0, "TSS": 30.0, "BOD5": 10.0}


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.where(np.abs(denominator) > 1e-12, numerator / denominator, np.nan)


def nrmse(
    true: np.ndarray, pred: np.ndarray, axis: int = 0, spread: np.ndarray | None = None
) -> np.ndarray:
    """RMSE normalised by the range of the truth. Range, not mean, so that
    components sitting near zero do not produce meaningless ratios.

    ``spread`` overrides the per-window range with a fixed reference range so
    that different evaluation windows become comparable on one scale.
    """
    rmse = np.sqrt(np.mean((pred - true) ** 2, axis=axis))
    if spread is None:
        spread = np.max(true, axis=axis) - np.min(true, axis=axis)
    return _safe_divide(rmse, spread)


def r_squared(true: np.ndarray, pred: np.ndarray, axis: int = 0) -> np.ndarray:
    ss_res = np.sum((true - pred) ** 2, axis=axis)
    ss_tot = np.sum((true - np.mean(true, axis=axis, keepdims=True)) ** 2, axis=axis)
    return 1.0 - _safe_divide(ss_res, ss_tot)


def mae(true: np.ndarray, pred: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.mean(np.abs(pred - true), axis=axis)


def per_tank_nrmse(
    truth: np.ndarray, pred: np.ndarray, spread: np.ndarray | None = None
) -> np.ndarray:
    """Range-normalised RMSE per (tank, component). ``truth``/``pred`` are ``(n, 5, 14)``.

    Returns ``(5, 14)``. The normaliser is the component's own range in that tank,
    or the fixed reference range ``spread`` ``(14,)`` broadcast over tanks, so that
    a component that is flat in one tank (oxygen in the anoxic zone) does not
    produce an unbounded ratio when a fixed range is supplied.
    """
    rmse = np.sqrt(np.mean((pred - truth) ** 2, axis=0))
    if spread is None:
        spread = np.max(truth, axis=0) - np.min(truth, axis=0)
    else:
        spread = np.broadcast_to(np.asarray(spread, dtype=float), rmse.shape)
    return _safe_divide(rmse, spread)


@dataclass
class StateMetrics:
    """Per-component metrics, averaged over the five tanks."""

    components: tuple[str, ...]
    nrmse: np.ndarray   # (14,)
    r2: np.ndarray      # (14,)
    mae: np.ndarray     # (14,)

    def as_dict(self) -> dict[str, dict[str, float]]:
        return {
            name: {
                "nrmse": float(self.nrmse[i]),
                "r2": float(self.r2[i]),
                "mae": float(self.mae[i]),
            }
            for i, name in enumerate(self.components)
        }

    def track(self, names: tuple[str, ...]) -> dict[str, float]:
        """Mean metric over a subset of components, ignoring undefined entries."""
        idx = [self.components.index(n) for n in names if n in self.components]
        return {
            "nrmse": float(np.nanmean(self.nrmse[idx])),
            "r2": float(np.nanmean(self.r2[idx])),
            "mae": float(np.nanmean(self.mae[idx])),
        }


def state_metrics(
    truth: np.ndarray, pred: np.ndarray, spread: np.ndarray | None = None
) -> StateMetrics:
    """``truth`` and ``pred`` are ``(n, 5, 14)``; metrics are pooled over tanks.

    ``spread`` (14,) fixes the NRMSE normaliser to a reference range instead of
    the evaluation window's own range.
    """
    v = vault()
    n, n_tanks, n_comp = truth.shape
    flat_true = truth.reshape(n * n_tanks, n_comp)
    flat_pred = pred.reshape(n * n_tanks, n_comp)
    return StateMetrics(
        components=v.components,
        nrmse=nrmse(flat_true, flat_pred, spread=spread),
        r2=r_squared(flat_true, flat_pred),
        mae=mae(flat_true, flat_pred),
    )


def track_summary(metrics: StateMetrics) -> dict[str, dict[str, float]]:
    """Track A / Track B breakdown, plus the per-component detail."""
    return {
        "track_a_measured": metrics.track(observed_components()),
        "track_b_unmeasured": metrics.track(unobserved_components()),
        "per_component": metrics.as_dict(),
    }


# --- v1.1 decision metrics --------------------------------------------------
# All take the same (n, 5, 14) truth/prediction arrays as state_metrics; none of
# them changes the v1.0 definitions above.

def skill_score(err, err_ref):
    """Skill against a reference: ``1 - err / err_ref`` (1 perfect, 0 no better, < 0 worse)."""
    err = np.asarray(err, dtype=float)
    err_ref = np.asarray(err_ref, dtype=float)
    return 1.0 - _safe_divide(err, err_ref)


def gap_closed(err, err_persist, err_ref):
    """Fraction of the persistence-to-reference gap closed:
    ``(err_persist - err) / (err_persist - err_ref)``. 1 matches the reference,
    0 matches persistence; undefined (NaN) when the two references coincide."""
    err = np.asarray(err, dtype=float)
    err_persist = np.asarray(err_persist, dtype=float)
    err_ref = np.asarray(err_ref, dtype=float)
    return _safe_divide(err_persist - err, err_persist - err_ref)


def level_error(truth: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Per-component RMSE relative to the mean level, ``(14,)``, pooled over tanks.

    Unlike the range-normalised NRMSE this reads as "percent of the typical
    concentration", which is the scale an operator uses for inventories such as
    X_B_H or X_B_A.
    """
    truth = np.asarray(truth, dtype=float)
    pred = np.asarray(pred, dtype=float)
    n_comp = truth.shape[-1]
    flat_t = truth.reshape(-1, n_comp)
    flat_p = pred.reshape(-1, n_comp)
    rmse = np.sqrt(np.mean((flat_p - flat_t) ** 2, axis=0))
    return _safe_divide(rmse, np.mean(np.abs(flat_t), axis=0))


def tank_mean(z: np.ndarray, volumes=None) -> np.ndarray:
    """Volume-weighted mean over the tanks: ``(n, 5, 14) -> (n, 14)``."""
    from ..asm1.plant import Bsm1Config

    vol = np.asarray(Bsm1Config().volumes if volumes is None else volumes, dtype=float)
    return np.einsum("ntc,t->nc", np.asarray(z, dtype=float), vol) / vol.sum()


def within_tolerance_fraction(
    truth: np.ndarray,
    pred: np.ndarray,
    components: tuple[str, ...] = ("X_B_H", "X_B_A"),
    rel_tol: float = 0.10,
) -> dict[str, float]:
    """Share of samples whose plant-wide (volume-weighted) mean is within ``rel_tol``.

    Returns ``{component: fraction in [0, 1]}``; the criterion is
    ``|mean_pred - mean_truth| <= rel_tol * |mean_truth|`` per sample.
    """
    v = vault()
    mt, mp = tank_mean(truth), tank_mean(pred)
    out: dict[str, float] = {}
    for name in components:
        i = v.index(name)
        ok = np.abs(mp[:, i] - mt[:, i]) <= rel_tol * np.abs(mt[:, i])
        out[name] = float(np.mean(ok))
    return out


def error_vs_time(
    truth: np.ndarray,
    pred: np.ndarray,
    t: np.ndarray,
    bin_days: float = 1.0,
    spread: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Per-component NRMSE in consecutive time bins, pooled over tanks.

    Bins are ``[t0 + k*bin_days, t0 + (k+1)*bin_days)``, the last one closed on
    the right. ``spread`` (14,) is the fixed normaliser; by default the range of
    the whole supplied truth, so every bin is on one scale. Returns
    ``{"t_start": (b,), "t_end": (b,), "n": (b,), "nrmse": (b, 14)}``.
    """
    truth = np.asarray(truth, dtype=float)
    pred = np.asarray(pred, dtype=float)
    t = np.asarray(t, dtype=float)
    n_comp = truth.shape[-1]
    if spread is None:
        flat = truth.reshape(-1, n_comp)
        spread = flat.max(axis=0) - flat.min(axis=0)
    t0, t1 = float(t[0]), float(t[-1])
    n_bins = max(int(np.ceil((t1 - t0) / bin_days - 1e-9)), 1)
    starts = t0 + bin_days * np.arange(n_bins)
    ends = np.minimum(starts + bin_days, t1)
    counts = np.zeros(n_bins, dtype=int)
    values = np.full((n_bins, n_comp), np.nan)
    for k in range(n_bins):
        upper = (t < starts[k] + bin_days - 1e-9) if k < n_bins - 1 else (t <= t1 + 1e-9)
        mask = (t >= starts[k] - 1e-9) & upper
        counts[k] = int(mask.sum())
        if counts[k]:
            values[k] = nrmse(
                truth[mask].reshape(-1, n_comp), pred[mask].reshape(-1, n_comp), spread=spread
            )
    return {"t_start": starts, "t_end": ends, "n": counts, "nrmse": values}


# --- effluent quality (ground-truth dataset descriptor) --------------------
def _effluent_terms(plant: Bsm1Plant, effluent: np.ndarray) -> dict[str, np.ndarray]:
    v = plant.vault
    i = {name: v.index(name) for name in v.components}
    f_p = v.p("fP")
    i_xb = v.p("iXB")
    i_xp = v.p("iXP")

    def c(name: str) -> np.ndarray:
        return effluent[..., i[name]]

    tss = plant.cfg.tss_factor * (c("X_S") + c("X_I") + c("X_B_H") + c("X_B_A") + c("X_P"))
    cod = c("S_S") + c("S_I") + c("X_S") + c("X_I") + c("X_B_H") + c("X_B_A") + c("X_P")
    bod5 = 0.25 * (c("S_S") + c("X_S") + (1.0 - f_p) * (c("X_B_H") + c("X_B_A")))
    nkj = (
        c("S_NH") + c("S_ND") + c("X_ND")
        + i_xb * (c("X_B_H") + c("X_B_A"))
        + i_xp * (c("X_P") + c("X_I"))
    )
    return {
        "TSS": tss,
        "COD": cod,
        "BOD5": bod5,
        "NKj": nkj,
        "NO": c("S_NO"),
        "S_NH": c("S_NH"),
        "N_tot": c("S_NO") + nkj,
    }


def effluent_quality_index(
    plant: Bsm1Plant, t: np.ndarray, effluent: np.ndarray, q_e: np.ndarray
) -> dict[str, float]:
    """BSM1 eq. 75 EQI [kg pollution unit/d] plus limit violations (Table 9)."""
    terms = _effluent_terms(plant, effluent)
    load = sum(EQI_WEIGHTS[k] * terms[k] for k in ("TSS", "COD", "NKj", "NO", "BOD5"))
    t_obs = float(t[-1] - t[0])
    eqi = float(np.trapezoid(load * q_e, t) / (t_obs * 1000.0))

    violations: dict[str, float] = {}
    for name, limit in EFFLUENT_LIMITS.items():
        series = terms[name]
        exceed = series > limit
        crossings = int(np.sum(np.diff(exceed.astype(int)) == 1))
        violations["%s_pct_time_over_limit" % name] = float(100.0 * np.mean(exceed))
        violations["%s_crossings" % name] = crossings

    percentiles = {
        "S_NH_p95": float(np.percentile(terms["S_NH"], 95)),
        "N_tot_p95": float(np.percentile(terms["N_tot"], 95)),
        "TSS_p95": float(np.percentile(terms["TSS"], 95)),
    }
    return {"eqi_kg_pu_per_day": eqi, **violations, **percentiles}


def continuity_of_prediction(plant: Bsm1Plant, pred: np.ndarray) -> float:
    """Max ``|r(Z) @ C|`` over the prediction.

    Structurally zero to floating-point accuracy because the vault matrix
    satisfies ``nu @ C = 0``; reported as a running assertion on the rate path.
    """
    r = plant.kinetics.conversion(pred)
    return float(np.max(np.abs(r @ plant.vault.composition)))

"""Influent-knowledge views: what an estimator is told about the influent.

The truth plant is always driven by the full 15-minute, 14-component influent.
An estimator may be told less. Three modes, used by the v1.1 regime grid:

``exact`` (Ie)
    The full 14-component series. This is the v1.0 information set.
``composite`` (Ic)
    What a plant laboratory reports from a flow-proportional 24 h composite
    sampler: per calendar day, the flow-weighted mean total organic COD, TKN
    and ammonium. The 14 components are recomposed with the BSM1 Table 5
    fractions (:data:`src.data.influent.BSM1_TABLE5_MEAN`), held constant over
    the day. ``Q_in`` stays at full 15-minute resolution (flow is metered
    online).
``composite_biased`` (Ib)
    ``composite`` with the readily biodegradable fraction under-estimated by a
    quarter (``S_S * 0.75``) and ``X_S`` raised to keep the COD. CPU rows only.

Calendar day ``d`` covers ``(d, d + 1]`` days; the ``t = 0`` sample joins day 0.
A view is applied to a full dataset before any time window is taken.

Only the known inputs are touched. :func:`view_dataset` replaces ``z_in`` and
nothing else, so observations, flows and the evaluation-only truth are
carried through unchanged.

The daily TKN is aggregated with the ``iXB`` and ``iXP`` of the ``vault``
argument, which :func:`view_dataset` sets to the nominal vault. Under the
bsm1_15c truth set (``iXB`` 0.08 instead of 0.086) the aggregated TKN therefore
exceeds the physically true influent TKN by ``0.006 X_B_H,in``, about
0.17 g N/m3 or 0.3 % of the Table 5 TKN of 54.6 g N/m3. Accepting that keeps
every truth parameter out of the estimator path.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from ..asm1.vault_loader import Asm1Vault, vault
from .influent import BSM1_TABLE5_MEAN
from .sensors import ObservationDataset

MODES: tuple[str, ...] = ("exact", "composite", "composite_biased")

ORGANIC_COD: tuple[str, ...] = ("S_I", "S_S", "X_I", "X_S", "X_B_H", "X_B_A", "X_P")
FIXED_AT_TABLE5: tuple[str, ...] = ("S_O", "S_NO", "X_B_A", "X_P", "S_ALK", "S_N2")
SCALED_COD: tuple[str, ...] = tuple(c for c in ORGANIC_COD if c not in FIXED_AT_TABLE5)
ORGANIC_N: tuple[str, ...] = ("S_ND", "X_ND")
BIAS_SS_FACTOR = 0.75


def calendar_day_index(t: np.ndarray) -> np.ndarray:
    """Composite-sample day of each time stamp: ``(d, d+1]`` -> ``d``; ``t = 0`` -> 0."""
    t = np.asarray(t, dtype=float)
    return np.maximum(np.ceil(t - 1e-9) - 1.0, 0.0).astype(int)


def total_cod(z: np.ndarray, v: Asm1Vault) -> np.ndarray:
    return z[..., v.indices(ORGANIC_COD)].sum(axis=-1)


def tkn(z: np.ndarray, v: Asm1Vault) -> np.ndarray:
    """S_NH + S_ND + X_ND + iXB (X_B_H + X_B_A) + iXP (X_I + X_P)."""
    i = v.index
    return (
        z[..., i("S_NH")] + z[..., i("S_ND")] + z[..., i("X_ND")]
        + v.p("iXB") * (z[..., i("X_B_H")] + z[..., i("X_B_A")])
        + v.p("iXP") * (z[..., i("X_I")] + z[..., i("X_P")])
    )


def daily_flow_weighted(t: np.ndarray, q_in: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(day index per sample, flow-weighted daily mean per day)``."""
    day = calendar_day_index(t)
    q = np.asarray(q_in, dtype=float).reshape(-1)
    weight = np.bincount(day, weights=q)
    mean = np.bincount(day, weights=q * np.asarray(x, dtype=float)) / weight
    return day, mean


def apply_influent_knowledge(
    t: np.ndarray,
    q_in: np.ndarray,
    z_in: np.ndarray,
    mode: str,
    vault: Asm1Vault,
) -> np.ndarray:
    """Influent composition an estimator receives under ``mode``; shape ``(n, 14)``."""
    if mode not in MODES:
        raise ValueError("Unknown influent mode %r; expected one of %s" % (mode, MODES))
    z_in = np.asarray(z_in, dtype=float)
    if mode == "exact":
        return z_in.copy()

    v = vault
    t5 = np.array([BSM1_TABLE5_MEAN[name] for name in v.components], dtype=float)
    day, cod_day = daily_flow_weighted(t, q_in, total_cod(z_in, v))
    _, tkn_day = daily_flow_weighted(t, q_in, tkn(z_in, v))
    _, nh_day = daily_flow_weighted(t, q_in, z_in[:, v.index("S_NH")])

    i_scaled = v.indices(SCALED_COD)
    i_fixed_cod = v.indices([c for c in ORGANIC_COD if c in FIXED_AT_TABLE5])
    scaled_t5 = float(t5[i_scaled].sum())
    fixed_cod = float(t5[i_fixed_cod].sum())

    out = np.tile(t5, (len(cod_day), 1))
    f_cod = (cod_day - fixed_cod) / scaled_t5
    out[:, i_scaled] = t5[i_scaled][None, :] * f_cod[:, None]
    out[:, v.index("S_NH")] = nh_day

    i_on = v.indices(ORGANIC_N)
    remainder = tkn_day - (tkn(out, v) - out[:, i_on].sum(axis=1))
    if np.any(remainder < 0.0):
        raise ValueError(
            "Daily TKN is below the ammonium plus biomass nitrogen implied by the "
            "Table 5 fractions on day(s) %s" % np.flatnonzero(remainder < 0.0).tolist()
        )
    out[:, i_on] = t5[i_on][None, :] * (remainder / float(t5[i_on].sum()))[:, None]

    if mode == "composite_biased":
        i_ss, i_xs = v.index("S_S"), v.index("X_S")
        shifted = (1.0 - BIAS_SS_FACTOR) * out[:, i_ss]
        out[:, i_ss] -= shifted
        out[:, i_xs] += shifted
    return out[day]


def view_dataset(ds: ObservationDataset, mode: str) -> ObservationDataset:
    """``ds`` with ``z_in`` replaced by the ``mode`` view (nominal vault fractions)."""
    z_view = apply_influent_knowledge(ds.t, ds.q_in, ds.z_in, mode, vault())
    return dataclasses.replace(ds, z_in=z_view, meta={**ds.meta, "influent_mode": mode})

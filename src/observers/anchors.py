"""Initial-state anchors: laboratory operators and log-normal prior updates.

Pure array functions; they never see a ground-truth array (scripts/make_anchors
reads the data and passes numbers in). An anchor is a per-entry log-normal
prior on the 5 x 14 state: mean ``z`` and relative sd ``rel_std`` (log variance
``log(1 + rel_std^2)``). An assay reads ``w . z`` with multiplicative error.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ..asm1.plant import SOLUBLE_COMPONENTS, TSS_COMPONENTS, Bsm1Config
from ..asm1.vault_loader import Asm1Vault

Z_FLOOR = 1e-12
LOG_VAR_FLOOR = 1e-12
PANEL_TANKS = (0, 4)  # tanks 1 and 5


def _column(vault: Asm1Vault, quantity: str) -> np.ndarray:
    return np.asarray(vault.composition[:, vault.conserved.index(quantity)], dtype=float)


def _at(cfg: Bsm1Config, tank: int, row: np.ndarray) -> np.ndarray:
    w = np.zeros((cfg.n_tanks, row.size))
    w[tank] = row
    return w


def lab_operators(vault: Asm1Vault, config: Bsm1Config | None = None) -> dict[str, np.ndarray]:
    """(5, 14) weights for the routine panel, from ``vault``'s composition.

    COD = positive COD entries (organics); CODf = COD on solubles; TKN = N
    column without S_NO and S_N2; TKNf = TKN on solubles; ALK = S_ALK (tanks 1
    and 5); TSS = tss_factor x BSM1 eq. 45 solids (all tanks).
    """
    cfg = config if config is not None else Bsm1Config()
    nc = len(vault.components)
    soluble = np.zeros(nc)
    soluble[vault.indices(SOLUBLE_COMPONENTS)] = 1.0
    cod = np.maximum(_column(vault, "COD"), 0.0)
    tkn = _column(vault, "N").copy()
    tkn[[vault.index("S_NO"), vault.index("S_N2")]] = 0.0
    alk = np.zeros(nc)
    alk[vault.index("S_ALK")] = 1.0
    tss = np.zeros(nc)
    tss[vault.indices(TSS_COMPONENTS)] = cfg.tss_factor
    ops: dict[str, np.ndarray] = {}
    for tank in PANEL_TANKS:
        for key, row in (("COD", cod), ("CODf", cod * soluble), ("TKN", tkn),
                         ("TKNf", tkn * soluble), ("ALK", alk)):
            ops["%s_tank%d" % (key, tank + 1)] = _at(cfg, tank, row)
    for tank in range(cfg.n_tanks):
        ops["TSS_tank%d" % (tank + 1)] = _at(cfg, tank, tss)
    return ops


def respirometry_operators(vault: Asm1Vault, config: Bsm1Config | None = None) -> dict[str, np.ndarray]:
    """Direct reads of ``X_B_H`` and ``X_B_A`` in tanks 1 and 5."""
    cfg = config if config is not None else Bsm1Config()
    ops: dict[str, np.ndarray] = {}
    for tank in PANEL_TANKS:
        for name in ("X_B_H", "X_B_A"):
            row = np.zeros(len(vault.components))
            row[vault.index(name)] = 1.0
            ops["%s_tank%d" % (name, tank + 1)] = _at(cfg, tank, row)
    return ops


def gaussian_log_update(mean, rel_std, operators: Sequence[np.ndarray], values: Sequence[float],
                        rel_errors: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """One linearised Kalman update of a diagonal log-normal prior; keeps the posterior diagonal.

    ``log y_j = log(w_j . z) + e_j``, ``e_j ~ N(0, log(1 + rel_err_j^2))``.
    """
    mean = np.asarray(mean, dtype=float)
    if len(operators) == 0:
        return mean.copy(), np.asarray(rel_std, dtype=float).copy()
    z = np.maximum(mean.reshape(-1), Z_FLOOR)
    x = np.log(z)
    P = np.diag(np.maximum(np.log1p(np.asarray(rel_std, float).reshape(-1) ** 2), LOG_VAR_FLOOR))
    W = np.stack([np.asarray(op, dtype=float).reshape(-1) for op in operators])
    R = np.diag(np.log1p(np.asarray(rel_errors, dtype=float) ** 2))
    yp = W @ z
    H = W * z[None, :] / yp[:, None]
    K = np.linalg.solve(H @ P @ H.T + R, H @ P).T
    x_post = x + K @ (np.log(np.asarray(values, dtype=float)) - np.log(yp))
    ikh = np.eye(x.size) - K @ H
    var = np.maximum(np.diag(ikh @ P @ ikh.T + K @ R @ K.T), LOG_VAR_FLOOR)
    return np.exp(x_post).reshape(mean.shape), np.sqrt(np.expm1(var)).reshape(mean.shape)


def ensemble_log_std(members) -> np.ndarray:
    """Relative sd of an ensemble ``(N, 5, 14)``: ``sqrt(exp(var(log z)) - 1)``."""
    arr = np.maximum(np.asarray(members, dtype=float), Z_FLOOR)
    if arr.shape[0] < 2:
        raise ValueError("ensemble_log_std needs at least two members")
    return np.sqrt(np.expm1(np.var(np.log(arr), axis=0, ddof=1)))


def ic_weights_from_rel_std(rel_std) -> np.ndarray:
    """Inverse log variance normalised to mean 1; exactly 1 for uniform ``rel_std``."""
    v = np.maximum(np.log1p(np.asarray(rel_std, dtype=float) ** 2), LOG_VAR_FLOOR)
    if np.ptp(v) == 0.0:
        return np.ones_like(v)
    return (1.0 / v) / np.mean(1.0 / v)


def lab_values(operators: Mapping[str, np.ndarray], state, rel_errors: Mapping[str, float],
               rng: np.random.Generator) -> dict[str, float]:
    """Assays ``w . z * (1 + err * N(0, 1))`` in the key order of ``rel_errors``."""
    z = np.asarray(state, dtype=float)
    return {name: float(np.sum(operators[name] * z)) * (1.0 + float(err) * float(rng.standard_normal()))
            for name, err in rel_errors.items()}

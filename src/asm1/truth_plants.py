"""Alternative ASM1 parameter sets for generating TRUTH data only.

Scope rule
----------
This module exists for one purpose: to build *truth plants* whose kinetics
differ from the estimators' model, so that kinetic mismatch can be studied.
Every estimator in this repository (PINN, LSTM, EKF, open-loop rows) keeps
using the single audited vault set from :func:`src.asm1.vault_loader.vault`.
``tests/test_truth_plants.py`` enforces this statically: no module under
``src/train``, ``src/models`` or ``src/observers`` may even mention this
module's name.

The vault is never modified. ``asm1_cl-pinn/data/asm1.json`` is read
read-only, its SHA-256 is checked against the cached vault before every
override, and :mod:`src.asm1.vault_loader` stays byte-identical.

Why stoichiometry is re-derived, not patched
--------------------------------------------
Changing ``iXB`` changes both the stoichiometric matrix (rho_1..rho_5 carry
``iXB`` terms) and the composition matrix (biomass N content). Patching the
numeric ``nu`` alone breaks the nitrogen balance by 6.0e-3 per unit rate
(verified 2026-09-23). :func:`override_vault` therefore re-evaluates BOTH the
vault's symbolic ``corrected_original`` stoichiometry and its symbolic
``composition_original`` table with the new parameters, which restores
continuity to 4.7e-16.
"""

from __future__ import annotations

import dataclasses
import json
import math
from typing import Any, Mapping

import numpy as np

from .vault_loader import VAULT_JSON, Asm1Vault, VaultIntegrityError, _sha256, vault

#: BSM1 15 degrees C values that differ from the vault's 20 degrees C set.
#: Source: Alex et al., "Benchmark Simulation Model no. 1 (BSM1)", IWA Task
#: Group on Benchmarking of Control Strategies for WWTPs (2018), stoichiometric
#: and kinetic parameter tables. These values MUST be checked cell by cell
#: against the report before any data generation; the check is recorded (table,
#: page, checker) in tests/data/bsm1_openloop_steady_state.json under
#: "parameter_check", and scripts/generate_data.py refuses to build bsm1_15c or
#: graded data until scripts/verify_bsm1_truth.parameter_check_problems is empty.
#: BSM1 has no ammonium switch in rho_1/rho_2, so KNH_H = 0 turns the vault's
#: ASM2d-sourced switch S_NH/(KNH_H+S_NH) into exactly 1 for S_NH >= 1e-12.
#: Every other parameter (YH 0.67, YA 0.24, fP 0.08, iXP 0.06, KO_H 0.2,
#: KNO 0.5, etag 0.8, kh 3.0, KO_A 0.4, KNH 1.0) equals the vault value.
BSM1_15C: Mapping[str, float] = {
    "muH": 4.0,
    "Ks": 10.0,
    "bH": 0.3,
    "KX": 0.1,
    "etah": 0.8,
    "muA": 0.5,
    "bA": 0.05,
    "ka": 0.05,
    "KNH_H": 0.0,
    "iXB": 0.08,
}

#: Interpolated geometrically between the two sets (all strictly positive).
LOG_INTERP: tuple[str, ...] = ("muH", "Ks", "bH", "KX", "etah", "muA", "bA", "ka")
#: Interpolated linearly (KNH_H reaches zero, iXB is a mass fraction).
LIN_INTERP: tuple[str, ...] = ("KNH_H", "iXB")

#: Physical conversion constants: never perturbed or overridden.
FIXED_CONSTANTS: tuple[str, ...] = (
    "iNO3_N2", "iCOD_NO3", "iCOD_N2", "iCharge_SNHx", "iCharge_SNOx",
)
#: Reduction factors that are physically bounded by one.
ETA_NAMES: tuple[str, ...] = ("etag", "etah")

PRESETS: tuple[str, ...] = ("vault20", "bsm1_15c", "graded")


def _payload() -> dict[str, Any]:
    return json.loads(VAULT_JSON.read_text(encoding="utf-8"))


def kinetic_names() -> tuple[str, ...]:
    """The vault's 15 kinetic parameter code ids, in vault order."""
    return tuple(entry["code_id"] for entry in _payload()["parameters"]["kinetic"])


def _normalise_matrix_label(label: str) -> str:
    """'XB,H' -> 'X_B_H', 'SNO' -> 'S_NO', 'SALK' -> 'S_ALK', 'SN2' -> 'S_N2'."""
    return (label[0] + "_" + label[1:]).replace(",", "_")


def _evaluate(expression: Any, namespace: Mapping[str, float], where: str) -> float:
    if isinstance(expression, (int, float)):
        return float(expression)
    code = compile(str(expression), "<vault:%s>" % where, "eval")
    return float(eval(code, {"__builtins__": {}}, dict(namespace)))  # noqa: S307 - vault-sourced


def override_vault(parameters: Mapping[str, float]) -> Asm1Vault:
    """A vault copy whose parameters, ``nu`` and composition use ``parameters``.

    ``parameters`` holds overrides only; keys not given keep the vault value.
    Unknown keys and the fixed conversion constants raise ``KeyError``.
    """
    base = vault()
    if _sha256(VAULT_JSON) != base.json_sha256:
        raise VaultIntegrityError(
            "asm1.json changed on disk since the vault was loaded; refusing to "
            "derive a truth parameter set from an unverified file."
        )
    for name in parameters:
        if name not in base.parameters:
            raise KeyError("Unknown vault parameter %r" % (name,))
        if name in FIXED_CONSTANTS:
            raise KeyError("%r is a physical conversion constant and cannot be overridden" % (name,))

    values = {**dict(base.parameters), **{k: float(v) for k, v in parameters.items()}}
    matrices = _payload()["matrices"]

    stoich = matrices["corrected_original"]
    stoich_cols = tuple(_normalise_matrix_label(c) for c in stoich["column_labels"])
    if stoich_cols != base.components or tuple(stoich["row_labels"]) != base.processes:
        raise VaultIntegrityError("corrected_original labels do not match the vault tables")
    nu = np.array(
        [
            [_evaluate(cell["code_expression"], values, "nu[%d,%d]" % (i, j))
             for j, cell in enumerate(row)]
            for i, row in enumerate(stoich["cells"])
        ],
        dtype=np.float64,
    )

    comp_block = matrices["composition_original"]
    comp_cols = tuple(_normalise_matrix_label(c) for c in comp_block["column_labels"])
    if comp_cols != base.components or tuple(comp_block["row_labels"]) != base.conserved:
        raise VaultIntegrityError("composition_original labels do not match the vault tables")
    composition = np.array(
        [
            [_evaluate(cell["code_expression"], values, "comp[%d,%d]" % (q, j))
             for j, cell in enumerate(row)]
            for q, row in enumerate(comp_block["cells"])
        ],
        dtype=np.float64,
    ).T  # (3, 14) in the vault -> (14, 3) like Asm1Vault.composition

    residual = float(np.max(np.abs(nu @ composition)))
    if residual > 1e-12:
        raise VaultIntegrityError(
            "Re-derived stoichiometry violates continuity: max |nu @ C| = %.3e" % residual
        )
    return dataclasses.replace(base, parameters=values, nu=nu, composition=composition)


def effective_alpha(preset: str, alpha: float = 1.0) -> float:
    """Position on the vault20 (0) -> bsm1_15c (1) axis that ``preset`` denotes."""
    if preset == "vault20":
        return 0.0
    if preset == "bsm1_15c":
        if alpha != 1.0:
            raise ValueError("preset 'bsm1_15c' is alpha = 1; use 'graded' for alpha %.3g" % alpha)
        return 1.0
    if preset == "graded":
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError("alpha must lie in [0, 1], got %r" % (alpha,))
        return float(alpha)
    raise ValueError("Unknown truth preset %r; expected one of %s" % (preset, PRESETS))


def graded_parameters(alpha: float) -> dict[str, float]:
    """Overrides at ``alpha``: geometric for LOG_INTERP, linear for LIN_INTERP."""
    base = vault().parameters
    a = float(alpha)
    out: dict[str, float] = {}
    for name in LOG_INTERP:
        v20, v15 = float(base[name]), float(BSM1_15C[name])
        out[name] = math.exp((1.0 - a) * math.log(v20) + a * math.log(v15))
    for name in LIN_INTERP:
        v20, v15 = float(base[name]), float(BSM1_15C[name])
        out[name] = (1.0 - a) * v20 + a * v15
    return out


def truth_vault(preset: str, alpha: float = 1.0) -> Asm1Vault:
    """Parameter set of a truth plant on the vault20 -> bsm1_15c axis."""
    a = effective_alpha(preset, alpha)
    if a == 0.0:
        return vault()
    if a == 1.0:
        return override_vault(BSM1_15C)
    return override_vault(graded_parameters(a))


def perturbed_vault(log_multipliers: Mapping[str, float]) -> Asm1Vault:
    """Vault with ``p_k * exp(m_k)`` for each named parameter; eta factors capped at 1."""
    base = vault().parameters
    overrides: dict[str, float] = {}
    for name, m in log_multipliers.items():
        if name not in base:
            raise KeyError("Unknown vault parameter %r" % (name,))
        if name in FIXED_CONSTANTS:
            raise KeyError("%r is a physical conversion constant and cannot be perturbed" % (name,))
        value = float(base[name]) * math.exp(float(m))
        if name in ETA_NAMES:
            value = min(value, 1.0)
        overrides[name] = value
    return override_vault(overrides)

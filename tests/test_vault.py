"""The vault contract: hashes, shapes, cross-checks, continuity, tracers."""

from __future__ import annotations

import numpy as np
import pytest

from src.asm1.continuity import matrix_residual, tracer_components
from src.asm1.vault_loader import VaultIntegrityError, load_vault


def test_hash_verification_is_enforced(tmp_path):
    bad = tmp_path / "asm1.json"
    bad.write_text('{"schema_version": "1.0.0"}', encoding="utf-8")
    with pytest.raises(VaultIntegrityError):
        load_vault(bad)


def test_contract_shapes(v):
    assert len(v.components) == 14
    assert len(v.processes) == 8
    assert len(v.parameters) == 25
    assert v.nu.shape == (8, 14)
    assert v.composition.shape == (14, 3)
    assert v.conserved == ("COD", "N", "Charge")


def test_vault_specific_features_are_present(v):
    """The two documented deviations from textbook ASM1 must survive."""
    assert "KNH_H" in v.parameters, "the ASM2d-sourced ammonium switch is missing"
    assert "S_N2" in v.components, "the matrix-only 14th component is missing"
    assert v.starred_missing_terms == ("X82", "X84")


def test_rate_expressions_reference_only_known_symbols(v):
    known = set(v.components) | set(v.parameters)
    for expr in v.rate_expressions:
        names = {
            token for token in
            __import__("re").findall(r"[A-Za-z_][A-Za-z_0-9]*", expr)
        }
        assert names <= known, "unknown symbols in %r: %s" % (expr, sorted(names - known))


def test_continuity_reproduces_the_audit_bit_for_bit(v):
    """Summed in the workbook's component order, the audit must be exact.

    Floating-point addition is not associative, so this only holds when the
    summation order matches the Excel formulas the audit was taken from. That
    makes it a real invariant rather than a tolerance guess.
    """
    exact = v.continuity_residual_in_source_order()
    assert np.array_equal(exact, v.audited_residuals)
    assert float(np.max(np.abs(exact))) == v.audited_max_residual


def test_continuity_holds_under_blas_reassociation(v):
    """BLAS reorders the sum, so it only has to stay inside the tolerance.

    Requiring bit equality here would be testing the linear-algebra backend, not
    the ASM1 data. The residual must still be a small fraction of one ULP of the
    largest coefficient in the matrix.
    """
    worst = float(np.max(np.abs(matrix_residual(v))))
    assert worst <= v.residual_tolerance
    one_ulp = float(np.spacing(float(np.max(np.abs(v.nu)))))
    assert worst < one_ulp, "residual %.3e is %.2f ULP - too large for rounding alone" % (
        worst, worst / one_ulp
    )


def test_inert_components_are_exact_tracers(v):
    tracers = tracer_components(v)
    assert set(tracers) == {"S_I", "X_I"}
    for name in tracers:
        assert np.all(v.nu[:, v.index(name)] == 0.0)


def test_ammonium_switch_is_wired_into_heterotroph_growth(v):
    """rho_1 and rho_2 must carry the vault's NH4 Monod term."""
    for i in (0, 1):
        assert "KNH_H" in v.rate_expressions[i]

"""Vault-backed ASM1 constants - the single source of truth for this project.

Every ASM1 number used anywhere in this repository originates from
``asm1_cl-pinn/data/asm1.json``, which is itself generated from ``asm1.xlsx``
(SHA-256 ``dff2424c...892a``) and independently audited (see ``Audit Report.md``).

Rules enforced here:

* No ASM1 parameter, stoichiometric coefficient or rate expression is
  hard-coded anywhere in this project. They are read from the vault at runtime.
* The vault JSON is SHA-256 verified on load against the hash recorded in
  ``Audit Report.md``. A mismatch is a hard error.
* There is exactly ONE parameter set: the vault's 20 degrees C values.
  No temperature switch, no Arrhenius correction, no alternative set exists.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

# Repository layout ---------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
VAULT_DIR = REPO_ROOT / "asm1_cl-pinn"
VAULT_JSON = VAULT_DIR / "data" / "asm1.json"
AUDIT_REPORT = VAULT_DIR / "Audit Report.md"
SOURCE_XLSX = REPO_ROOT / "asm1.xlsx"

# The vault's numeric-stoichiometry column labels use a dot in the biomass names
# (``X_B.H``) while the component table uses an underscore (``X_B_H``). The vault
# documents these as aliases of one identity; we normalise on the component form.
_LABEL_NORMALISE = str.maketrans({".": "_"})


def _normalise_label(label: str) -> str:
    return label.translate(_LABEL_NORMALISE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_json_sha256() -> str:
    """Read the audited JSON hash out of the vault's own Audit Report."""
    text = AUDIT_REPORT.read_text(encoding="utf-8")
    match = re.search(r"^json_sha256:\s*([0-9a-f]{64})\s*$", text, re.MULTILINE)
    if match is None:
        raise RuntimeError(
            "Could not read 'json_sha256' from %s. The vault audit front-matter "
            "is required to verify data/asm1.json." % AUDIT_REPORT
        )
    return match.group(1)


class VaultIntegrityError(RuntimeError):
    """Raised when the vault JSON does not match its audited hash or contract."""


@dataclass(frozen=True)
class Asm1Vault:
    """Immutable view of the audited ASM1 knowledge base."""

    components: tuple[str, ...]          # 14 component code ids, matrix order
    component_units: tuple[str, ...]
    processes: tuple[str, ...]           # 8 process names, matrix row order
    parameters: Mapping[str, float]      # 25 parameter code_id -> value
    nu: np.ndarray                       # (8, 14) numeric stoichiometry
    composition: np.ndarray              # (14, 3) COD / N / Charge
    conserved: tuple[str, ...]           # ("COD", "N", "Charge")
    rate_expressions: tuple[str, ...]    # 8 vault ``code_expression`` strings
    json_sha256: str
    source_xlsx_sha256: str
    audited_residuals: np.ndarray        # (8, 3) as recorded by the vault audit
    audited_max_residual: float
    residual_tolerance: float
    starred_missing_terms: tuple[str, ...]

    # -- lookup helpers ----------------------------------------------------
    def index(self, component: str) -> int:
        try:
            return self.components.index(component)
        except ValueError as exc:  # pragma: no cover - programming error
            raise KeyError("Unknown ASM1 component %r" % (component,)) from exc

    def indices(self, names: Sequence[str]) -> list[int]:
        return [self.index(name) for name in names]

    def p(self, code_id: str) -> float:
        return self.parameters[code_id]

    # -- audit -------------------------------------------------------------
    def continuity_residual(self) -> np.ndarray:
        """``nu @ composition`` via BLAS; shape (8 processes, 3 quantities).

        BLAS is free to reorder the summation, so this does NOT reproduce the
        audited value bit for bit - it lands within a fraction of one ULP of the
        largest coefficient instead. Use it for the tolerance gate, and
        :meth:`continuity_residual_in_source_order` for the exact comparison.
        """
        return self.nu @ self.composition

    def continuity_residual_in_source_order(self) -> np.ndarray:
        """Same product, summed in the workbook's own component order.

        The audited residuals come from Excel formulas of the form
        ``AA63 = L63*$AA$47 + M63*$AA$48 + ...``, i.e. a left-to-right sum over
        components. Floating-point addition is not associative, so reproducing
        the audited numbers exactly requires reproducing that order. This method
        does, which turns "matches the audit" into a bit-exact invariant instead
        of a tolerance guess.
        """
        n_processes, n_components = self.nu.shape
        n_quantities = self.composition.shape[1]
        out = np.zeros((n_processes, n_quantities), dtype=np.float64)
        for i in range(n_processes):
            for q in range(n_quantities):
                total = 0.0
                for k in range(n_components):
                    total += self.nu[i, k] * self.composition[k, q]
                out[i, q] = total
        return out


def _load_parameters(payload: dict) -> dict[str, float]:
    values: dict[str, float] = {}
    for group in ("stoichiometric", "kinetic"):
        for entry in payload["parameters"][group]:
            code_id = entry["code_id"]
            if code_id in values:
                raise VaultIntegrityError("Duplicate parameter code_id %r" % (code_id,))
            values[code_id] = float(entry["evaluated_value"])
    return values


def _load_matrix(block: dict, expect_rows: int, expect_cols: int) -> np.ndarray:
    cells = block["cells"]
    if len(cells) != expect_rows or any(len(row) != expect_cols for row in cells):
        raise VaultIntegrityError(
            "Matrix %s is not %dx%d" % (block.get("source_range"), expect_rows, expect_cols)
        )
    return np.array(
        [[float(cell["evaluated_value"]) for cell in row] for row in cells],
        dtype=np.float64,
    )


def load_vault(path: Path | str | None = None, *, verify_hash: bool = True) -> Asm1Vault:
    """Load and verify the audited ASM1 vault.

    Parameters
    ----------
    path
        Override for the JSON location (tests only).
    verify_hash
        When True (default) the file hash must equal the audited hash.
    """
    json_path = Path(path) if path is not None else VAULT_JSON
    actual = _sha256(json_path)
    if verify_hash:
        expected = _expected_json_sha256()
        if actual != expected:
            raise VaultIntegrityError(
                "data/asm1.json does not match the audited hash.\n"
                "  expected %s\n  actual   %s\n"
                "Refusing to run: every number in this project must come from "
                "the audited vault." % (expected, actual)
            )

    payload = json.loads(json_path.read_text(encoding="utf-8"))

    components = tuple(entry["code_id"] for entry in payload["components"])
    component_units = tuple(str(entry["unit"]) for entry in payload["components"])
    processes = tuple(entry["name"] for entry in payload["processes"])
    if len(components) != 14 or len(processes) != 8:
        raise VaultIntegrityError(
            "Vault contract violated: got %d components / %d processes; expected 14 / 8."
            % (len(components), len(processes))
        )

    stoich = payload["matrices"]["numeric_stoichiometry"]
    stoich_cols = tuple(_normalise_label(c) for c in stoich["column_labels"])
    if stoich_cols != components:
        raise VaultIntegrityError(
            "numeric_stoichiometry columns do not match the component table:\n"
            "  matrix     %s\n  components %s" % (stoich_cols, components)
        )
    if tuple(stoich["row_labels"]) != processes:
        raise VaultIntegrityError("numeric_stoichiometry rows do not match the process table")

    comp_block = payload["matrices"]["continuity_composition"]
    comp_rows = tuple(_normalise_label(r) for r in comp_block["row_labels"])
    if comp_rows != components:
        raise VaultIntegrityError("continuity_composition rows do not match the component table")

    # Rate expressions: the vault stores the original and standardised source
    # text plus an identical code-safe expression for each. We assert they agree
    # and then use the code-safe form verbatim.
    rate_expressions: list[str] = []
    for process in payload["processes"]:
        original = process["rates"]["original"]["code_expression"]
        standardised = process["rates"]["standardised"]["code_expression"]
        if original != standardised:
            raise VaultIntegrityError(
                "Process %s has diverging code expressions:\n"
                "  original     %s\n  standardised %s"
                % (process["process_id"], original, standardised)
            )
        rate_expressions.append(original)

    return Asm1Vault(
        components=components,
        component_units=component_units,
        processes=processes,
        parameters=_load_parameters(payload),
        nu=_load_matrix(stoich, 8, 14),
        composition=_load_matrix(comp_block, 14, 3),
        conserved=tuple(comp_block["column_labels"]),
        rate_expressions=tuple(rate_expressions),
        json_sha256=actual,
        source_xlsx_sha256=str(payload["source"]["sha256"]),
        audited_residuals=np.array(payload["continuity"]["residuals"], dtype=np.float64),
        audited_max_residual=float(payload["continuity"]["max_abs_residual"]),
        residual_tolerance=float(payload["continuity"]["tolerance"]),
        starred_missing_terms=tuple(payload["kinetic_checking"]["starred_missing_terms"]),
    )


_CACHE: Asm1Vault | None = None


def vault() -> Asm1Vault:
    """Process-wide cached vault instance."""
    global _CACHE
    if _CACHE is None:
        _CACHE = load_vault()
    return _CACHE

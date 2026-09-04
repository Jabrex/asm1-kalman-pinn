"""RUNBOOK steps 1 and 2 - vault loader and continuity verification.

Step 1  Every parameter and every stoichiometric coefficient the code will use is
        cross-checked against the vault's own generated Markdown views, which are
        an independent rendering of the same audited source. Expected: 0 diffs.
Step 2  The continuity residual ``nu @ composition`` is recomputed and compared
        with the vault audit's published value and tolerance.

Run::

    python -m scripts.verify_vault

Exit code 0 means both gates pass. Any non-zero exit means the data the whole
project rests on is not what it claims to be - stop and investigate before
generating anything.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.asm1.vault_loader import VAULT_DIR, vault  # noqa: E402

PARAMETERS_MD = VAULT_DIR / "Parameters.md"
COMPOSITION_MD = VAULT_DIR / "Composition and Continuity.md"

BLANK_CELL = "∅ (0)"


def _table_rows(text: str, heading: str) -> list[list[str]]:
    """Return the data rows of the Markdown table under ``heading``."""
    section = text.split("## " + heading, 1)
    if len(section) < 2:
        raise RuntimeError("Heading %r not found" % heading)
    rows: list[list[str]] = []
    for line in section[1].splitlines():
        line = line.strip()
        if not line.startswith("|"):
            if rows:
                break
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(set(c) <= {"-", " "} for c in cells):
            continue
        rows.append(cells)
    return rows[1:]  # drop the header row


def check_parameters() -> list[str]:
    v = vault()
    text = PARAMETERS_MD.read_text(encoding="utf-8")
    problems: list[str] = []
    seen: set[str] = set()
    for heading in ("Stoichiometric parameters", "Kinetic parameters"):
        for cells in _table_rows(text, heading):
            code_id = cells[2].strip("`")
            expected = float(cells[5])
            seen.add(code_id)
            if code_id not in v.parameters:
                problems.append("parameter %s present in Markdown but not in JSON" % code_id)
                continue
            actual = v.parameters[code_id]
            if actual != expected:
                problems.append(
                    "parameter %s: JSON %r != Markdown %r" % (code_id, actual, expected)
                )
    missing = set(v.parameters) - seen
    if missing:
        problems.append("parameters in JSON but not in Markdown: %s" % sorted(missing))
    return problems


def check_stoichiometry() -> list[str]:
    v = vault()
    text = COMPOSITION_MD.read_text(encoding="utf-8")
    problems: list[str] = []
    rows = _table_rows(text, "Numeric stoichiometry")
    if len(rows) != v.nu.shape[0]:
        return ["numeric stoichiometry has %d rows, expected %d" % (len(rows), v.nu.shape[0])]
    for i, cells in enumerate(rows):
        values = cells[1:]
        if len(values) != v.nu.shape[1]:
            problems.append("row %d has %d columns, expected %d" % (i, len(values), v.nu.shape[1]))
            continue
        for j, cell in enumerate(values):
            expected = 0.0 if cell == BLANK_CELL else float(cell)
            if v.nu[i, j] != expected:
                problems.append(
                    "nu[%d,%d] (%s / %s): JSON %r != Markdown %r"
                    % (i, j, v.processes[i], v.components[j], v.nu[i, j], expected)
                )
    return problems


def check_continuity() -> tuple[dict[str, float], list[str]]:
    """Two gates, because floating-point addition is not associative.

    Exactness gate
        Summed in the workbook's own component order, the residuals must equal
        the audited ones bit for bit. This is the strong invariant: same
        matrices, same order, same result.
    Tolerance gate
        Summed by BLAS, which reorders freely, the residual must stay inside the
        vault tolerance. BLAS lands a fraction of one ULP away from the audited
        number; demanding bit equality there would test the linear-algebra
        backend rather than the ASM1 data.
    """
    v = vault()
    exact = v.continuity_residual_in_source_order()
    blas = v.continuity_residual()
    largest = float(np.max(np.abs(v.nu)))
    stats = {
        "exact_max": float(np.max(np.abs(exact))),
        "blas_max": float(np.max(np.abs(blas))),
        "audited_max": v.audited_max_residual,
        "tolerance": v.residual_tolerance,
        "largest_coefficient": largest,
        "one_ulp_at_largest": float(np.spacing(largest)),
    }
    stats["blas_ulps"] = stats["blas_max"] / stats["one_ulp_at_largest"]

    problems: list[str] = []
    if not np.array_equal(exact, v.audited_residuals):
        worst = int(np.argmax(np.abs(exact - v.audited_residuals)))
        i, q = divmod(worst, exact.shape[1])
        problems.append(
            "source-order residuals differ from the audited matrix; largest "
            "difference at process %d (%s) / %s: recomputed %.17e vs audited %.17e"
            % (i, v.processes[i], v.conserved[q], exact[i, q], v.audited_residuals[i, q])
        )
    if stats["exact_max"] != v.audited_max_residual:
        problems.append(
            "source-order maximum %.17e does not equal the audited maximum %.17e"
            % (stats["exact_max"], v.audited_max_residual)
        )
    for label, value in (("source-order", stats["exact_max"]), ("BLAS", stats["blas_max"])):
        if value > v.residual_tolerance:
            problems.append(
                "%s continuity residual %.3e exceeds tolerance %.3e"
                % (label, value, v.residual_tolerance)
            )
    return stats, problems


def main() -> int:
    v = vault()
    print("vault json sha256 : %s" % v.json_sha256)
    print("source xlsx sha256: %s" % v.source_xlsx_sha256)
    print("components        : %d" % len(v.components))
    print("processes         : %d" % len(v.processes))
    print("parameters        : %d" % len(v.parameters))
    print("missing alk terms : %s (preserved, no term invented)" % (v.starred_missing_terms,))
    print()

    data_problems = check_parameters() + check_stoichiometry()
    stats, continuity_problems = check_continuity()
    problems = data_problems + continuity_problems

    print("STEP 1  parameter and stoichiometry cross-check")
    print("  Markdown-vs-JSON differences: %d" % len(data_problems))
    print()
    print("STEP 2  continuity")
    print("  source-order max  : %.17e  (must equal the audit exactly)" % stats["exact_max"])
    print("  audited value     : %.17e" % stats["audited_max"])
    print("  BLAS matmul max   : %.17e  (%.3f ULP of the largest coefficient)"
          % (stats["blas_max"], stats["blas_ulps"]))
    print("  largest nu coeff  : %.17e  (1 ULP = %.3e)"
          % (stats["largest_coefficient"], stats["one_ulp_at_largest"]))
    print("  tolerance         : %.3e" % stats["tolerance"])
    print()

    if problems:
        print("FAIL")
        for p in problems:
            print("  - %s" % p)
        return 1
    print("PASS - steps 1 and 2 clear")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""BSM1 gate (plan decision D2): does the bsm1_15c truth plant reproduce the
BSM1 open-loop steady state?

Compares the 200-day warm-up of ``truth_vault("bsm1_15c")`` on the constant
Table 5 load against the steady state an author transcribed from the BSM1
report into ``tests/data/bsm1_openloop_steady_state.json``. The tolerances are
declared here, before the comparison is ever run, and are not tuned:

    COD states and biomass (S_I, S_S, X_I, X_S, X_B_H, X_B_A, X_P)   <= 1 % relative
    nitrogen states and oxygen (S_NO, S_NH, S_ND, X_ND, S_O)         <= 2 % relative
    S_ALK                                                            reported only
    S_N2                                                             not compared (not in BSM1)

Cells the transcriber lists in ``report_only_cells`` before running (for
example a value printed with fewer than three significant digits) are reported
but not gated. The parameter table is checked too: every transcribed BSM1
parameter must equal the truth set exactly. ``scripts/generate_data.py``
refuses to build bsm1_15c or graded data until that parameter check is clean.

Writes ``results/v11/bsm1_gate.json``. Exit code 0 = pass, 1 = fail,
2 = reference not transcribed yet (gate not evaluated).

Usage: python -m scripts.verify_bsm1_truth
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.asm1.plant import Bsm1Plant
from src.asm1.truth_plants import truth_vault
from src.data.simulate import steady_state_residual, warm_up

REFERENCE = REPO_ROOT / "tests" / "data" / "bsm1_openloop_steady_state.json"
DEFAULT_OUT = REPO_ROOT / "results" / "v11" / "bsm1_gate.json"

TOLERANCES = {
    "S_I": 0.01, "S_S": 0.01, "X_I": 0.01, "X_S": 0.01,
    "X_B_H": 0.01, "X_B_A": 0.01, "X_P": 0.01,
    "S_NO": 0.02, "S_NH": 0.02, "S_ND": 0.02, "X_ND": 0.02, "S_O": 0.02,
}
REPORT_ONLY = ("S_ALK",)
PASS_WORDING = "BSM1 15 C kinetic set"
FAIL_WORDING = "BSM1-layout plant with BSM1 kinetic values"


def load_reference(path: Path = REFERENCE) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parameter_check_problems(ref: dict) -> list[str]:
    """Open items of the author's cell-by-cell check of the BSM1 parameter table."""
    check = ref["parameter_check"]
    problems = ["parameter_check.%s is empty" % k
                for k in ("table", "page", "checked_by") if check.get(k) in (None, "")]
    truth = truth_vault("bsm1_15c")
    for name, value in check["values"].items():
        if value is None:
            problems.append("parameter %s not transcribed" % name)
        elif float(value) != float(truth.p(name)):
            problems.append("parameter %s: report %r, truth set %r" % (name, value, truth.p(name)))
    return problems


def steady_state_missing(ref: dict) -> list[str]:
    """Steady-state fields an author still has to fill before the gate can run."""
    missing = [key for key in ("table", "page", "transcribed_by") if ref.get(key) in (None, "")]
    for tank, row in ref["tanks"].items():
        missing += ["tank %s %s" % (tank, c) for c, x in zip(ref["components"], row) if x is None]
    return missing


def compare(ref: dict, reactor: np.ndarray, components: tuple[str, ...]) -> dict:
    report_only = {(int(t), c) for t, c in ref.get("report_only_cells", [])}
    cells = []
    failures = 0
    for tank_key, row in sorted(ref["tanks"].items()):
        tank = int(tank_key)
        for name, reference in zip(ref["components"], row):
            simulated = float(reactor[tank - 1, components.index(name)])
            rel = abs(simulated - reference) / abs(reference) if reference != 0 else float("inf")
            tol = TOLERANCES.get(name)
            gated = tol is not None and name not in REPORT_ONLY and (tank, name) not in report_only
            ok = (rel <= tol) if gated else None
            failures += int(ok is False)
            cells.append({
                "tank": tank, "component": name, "reference": reference,
                "simulated": simulated, "relative_deviation": rel,
                "tolerance": tol if gated else None, "pass": ok,
            })
    return {"cells": cells, "failures": failures}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default=str(REFERENCE))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)

    ref = load_reference(Path(args.reference))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tolerances = {"relative": TOLERANCES, "report_only": list(REPORT_ONLY), "not_compared": ["S_N2"]}

    missing = steady_state_missing(ref) + parameter_check_problems(ref)
    if missing:
        out.write_text(json.dumps({
            "status": "not_transcribed", "passed": False, "missing": missing, "tolerances": tolerances,
            "wording": None,
        }, indent=2), encoding="utf-8")
        print("NOT EVALUATED - %d open reference items (first: %s). Wrote %s"
              % (len(missing), missing[0], out))
        return 2

    truth = truth_vault("bsm1_15c")
    plant, y = warm_up(Bsm1Plant(source=truth))
    reactor = plant.unpack(y)[0]
    result = compare(ref, reactor, plant.vault.components)
    passed = result["failures"] == 0
    payload = {
        "status": "pass" if passed else "fail",
        "passed": passed,
        "wording": PASS_WORDING if passed else FAIL_WORDING,
        "reference": {k: ref[k] for k in ("source", "table", "page", "transcribed_by",
                                          "transcribed_on", "checked_by")},
        "parameter_check": {k: ref["parameter_check"][k] for k in ("table", "page", "checked_by")},
        "tolerances": tolerances,
        "steady_state_residual": steady_state_residual(plant, y),
        "gated_failures": result["failures"],
        "cells": result["cells"],
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for cell in result["cells"]:
        flag = {True: "ok", False: "FAIL", None: "report"}[cell["pass"]]
        print("tank %d %-6s report %11.4f  sim %11.4f  rel %.4f  %s" % (
            cell["tank"], cell["component"], cell["reference"], cell["simulated"],
            cell["relative_deviation"], flag))
    print("%s - wording: %s. Wrote %s" % (payload["status"].upper(), payload["wording"], out))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

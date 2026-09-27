"""BSM1 gate: reference schema, parameter check and the pre-declared tolerances."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.verify_bsm1_truth import (
    TOLERANCES,
    compare,
    load_reference,
    main,
    parameter_check_problems,
    steady_state_missing,
)
from src.asm1.truth_plants import truth_vault

REPO = Path(__file__).resolve().parents[1]
REFERENCE = REPO / "tests" / "data" / "bsm1_openloop_steady_state.json"
REF_COMPONENTS = ["S_I", "S_S", "X_I", "X_S", "X_B_H", "X_B_A", "X_P",
                  "S_O", "S_NO", "S_NH", "S_ND", "X_ND", "S_ALK"]
BSM1_NAMES = {"YA", "YH", "fP", "iXB", "iXP", "muH", "Ks", "KO_H", "KNO", "bH",
              "etag", "etah", "kh", "KX", "muA", "KNH", "bA", "KO_A", "ka"}


def test_reference_file_has_the_declared_schema():
    ref = load_reference(REFERENCE)
    assert ref["components"] == REF_COMPONENTS
    assert sorted(ref["tanks"]) == ["1", "2", "3", "4", "5"]
    for row in ref["tanks"].values():
        assert len(row) == 13
        assert all(x is None or isinstance(x, (int, float)) for x in row)
    for key in ("source", "table", "page", "transcribed_by", "transcribed_on", "checked_by"):
        assert key in ref
    assert set(ref["parameter_check"]["values"]) == BSM1_NAMES


def test_tolerances_are_the_pre_declared_ones():
    assert {k for k, v in TOLERANCES.items() if v == 0.01} == {
        "S_I", "S_S", "X_I", "X_S", "X_B_H", "X_B_A", "X_P"}
    assert {k for k, v in TOLERANCES.items() if v == 0.02} == {
        "S_NO", "S_NH", "S_ND", "X_ND", "S_O"}
    assert "S_ALK" not in TOLERANCES and "S_N2" not in TOLERANCES


def test_parameter_check_accepts_exact_values_and_names_a_mismatch():
    truth = truth_vault("bsm1_15c")
    ref = {"parameter_check": {"table": "T", "page": "p", "checked_by": "initials",
                               "values": {name: truth.p(name) for name in sorted(BSM1_NAMES)}}}
    assert parameter_check_problems(ref) == []
    ref["parameter_check"]["values"]["muA"] = 0.8
    assert parameter_check_problems(ref) == ["parameter muA: report 0.8, truth set 0.5"]


def test_compare_applies_the_declared_tolerances():
    comps = tuple(REF_COMPONENTS) + ("S_N2",)
    ref = {"components": REF_COMPONENTS,
           "tanks": {str(k): [10.0] * 13 for k in range(1, 6)},
           "report_only_cells": [[5, "S_S"]]}
    reactor = np.full((5, 14), 10.0)
    reactor[:, comps.index("X_B_H")] = 10.09
    reactor[:, comps.index("S_NH")] = 10.19
    reactor[:, comps.index("S_ALK")] = 20.0
    reactor[4, comps.index("S_S")] = 11.0
    assert compare(ref, reactor, comps)["failures"] == 0
    reactor[0, comps.index("X_B_A")] = 10.11
    assert compare(ref, reactor, comps)["failures"] == 1


def test_untranscribed_reference_is_not_evaluated(tmp_path):
    out = tmp_path / "gate.json"
    template = {**load_reference(REFERENCE)}
    template["tanks"] = {k: [None] * 13 for k in template["tanks"]}
    path = tmp_path / "ref.json"
    path.write_text(json.dumps(template), encoding="utf-8")
    assert steady_state_missing(template)
    assert main(["--reference", str(path), "--out", str(out)]) == 2
    assert json.loads(out.read_text(encoding="utf-8"))["status"] == "not_transcribed"

"""Tables T1-T3, SI tables and the number registry are built from saved artefacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import v11_tables as vt
from test_v11_figures import artefacts

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def table_inputs(artefacts, monkeypatch):
    monkeypatch.chdir(REPO)
    root = artefacts
    (root / "analysis" / "kinetic_subset.json").write_text(json.dumps({"names": ["muA", "bA", "muH", "bH"]}),
                                                           encoding="utf-8")
    (root / "gpu_ledger.csv").write_text("run_id,minutes,eq\nk000_ie_a0_seed0/cl_pinn_sigma0p10,9.1,1.0\n"
                                         "k100_ic_as_lstm_seed0/lstm_sigma0p10,2.4,0.35\n", encoding="utf-8")
    (root / "derivative_audit.json").write_text(json.dumps({"runs": [
        {"run": "results/runs/cl_pinn_sigma0p10", "train": {"partial_mse": 0.5, "total_mse": 0.9, "ratio": 1.8}}]}),
        encoding="utf-8")
    (root / "bsm1_gate.json").write_text(json.dumps({"passed": True, "states": [
        {"tank": 1, "component": "S_S", "rel_error": 0.004, "tolerance": 0.01, "pass": True}]}), encoding="utf-8")
    run = root / "pinn" / "k000_ie_a0_seed0" / "cl_pinn_sigma0p10"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(json.dumps({"model": "cl_pinn", "ras_filter_window": 4}), encoding="utf-8")
    return root


def test_every_table_and_the_registry_are_written(table_inputs):
    root = table_inputs
    vt.main(["--root", str(root), "--rec-realistic", str(root / "analysis" / "recoverability_k100.json"),
             "--kinetic-subset", str(root / "analysis" / "kinetic_subset.json"),
             "--validation", str(root / "analysis" / "recoverability_validation.json"),
             "--prereg-commit", "abc123def456"])
    index = json.loads((root / "tables" / "tables_index.json").read_text(encoding="utf-8"))
    for name in ("T1_information", "T2_regimes", "T3_practitioner", "S_components_R0", "S_ekf", "S_realisations",
                 "S_random_mismatch", "S_sigma_log", "S_sensor_subsets", "S_lab_assays", "S_v10_superseded",
                 "S_run_counts", "S_derivative_audit", "S_bsm1_gate"):
        assert (root / "tables" / (name + ".tex")).exists() and name in index
    t1 = (root / "tables" / "T1_information.tex").read_text(encoding="utf-8")
    assert "trailing mean of 4 samples" in t1 and "muA, bA, muH, bH" in t1
    assert "abc123def456" in (root / "tables" / "T2_regimes.tex").read_text(encoding="utf-8")
    assert "superseded" in (root / "tables" / "S_v10_superseded.md").read_text(encoding="utf-8")
    numbers = json.loads((root / "numbers.json").read_text(encoding="utf-8"))
    assert numbers["gpu.total_equivalents"]["value"] == pytest.approx(1.35)
    assert any(k.startswith("trackB.R0.k000_ie_a0.sigma0.10.cl_pinn") for k in numbers)
    assert any(k.startswith("h6.k000_ie_a0.eks") for k in numbers)


def test_practitioner_table_picks_probe_assay_and_interval(table_inputs):
    rec = vt.load_recoverability(table_inputs / "analysis" / "recoverability_k100.json")
    tab = vt.t3_practitioner(rec)
    rows = {r[0]: r for r in tab["rows"]}
    assert rows["X_I"][2] == "TSS_tank5" and rows["X_I"][3] == "TSS (Al1)"
    assert rows["X_B_H"][3] == "respirometric X_B_H (Al2)"
    assert rows["S_S"][2] == "none"


def test_v10_run_counts_match_the_plan(table_inputs):
    counts = vt.run_counts(vt.V10_ROOTS, table_inputs, table_inputs / "gpu_ledger.csv")
    v10 = {r[1]: r[2] for r in counts["rows"] if r[0] == "v1.0"}
    assert v10["pinn"] == 36 and v10["lstm"] == 8 and v10["analytic"] == 8 and v10["verification probe"] == 2


def test_generic_records_accept_the_common_layouts():
    assert vt.records_from([{"a": 1}]) == [{"a": 1}]
    assert vt.records_from({"runs": [{"a": 1}]}) == [{"a": 1}]
    assert vt.records_from({"x": {"a": 1}}) == [{"key": "x", "a": 1}]
    assert vt.flatten({"a": {"b": 1, "c": [1, 2]}}) == {"a.b": 1, "a.c": "1;2"}


def test_hypotheses_table_and_registry_take_h6_from_the_g4_validation():
    table = {"meta": {"diverged_runs": [{"run_dir": "x", "reason": "mean NIS 22 > 3 x 7 channels"}]},
             "rows": [], "comparisons": [], "crossovers": [],
             "hypotheses": {"H1": {"status": "refuted", "summary": "CL-PINN beats EKS"},
                            "H2": {"status": "supported", "alpha_star": 0.33, "summary": "alpha* 0.33"},
                            "H5": {"status": "supported", "summary": "",
                                   "families": {"eks": {"ratio_forcing": 2.0, "ratio_biomass": 0.8}}},
                            "H7": {"status": "supported", "summary": "win"}}}
    validation = {"index": "crb_over_range", "window": "R0", "primary_cell": "k050_ic_as",
                  "h6": {"cell": "k050_ic_as", "estimator": "eks", "rho": 0.62, "ci95": [0.2, 0.8], "pass": True},
                  "results": [{"cell": "k050_ic_as", "estimator": "eks", "rho": 0.62, "ci": [0.2, 0.8]}]}
    tab = vt.si_hypotheses(table, validation)
    rows = {r[0]: r for r in tab["rows"]}
    assert [r[0] for r in tab["rows"]] == ["H1", "H2", "H5", "H6", "H7"]
    assert rows["H6"][1] == "supported" and "0.62" in rows["H6"][2]
    counts = {"total_eq": 41.25}
    reg = vt.numbers_registry(table, None, None, validation, counts)
    assert reg["hyp.H6.pass"]["value"] is True and reg["hyp.H2.alpha_star"]["value"] == 0.33
    assert reg["hyp.H5.eks.ratio_forcing"]["value"] == 2.0 and reg["diverged_runs.count"]["value"] == 1
    assert reg["h6.k050_ic_as.eks.ci95"]["value"] == [0.2, 0.8]


def test_sensor_subset_table_lists_the_sensor_cells():
    rec = {"tag": "k100", "sensor_subsets": [{"channels": ["a", "b"], "ig_mean": 0.5}, {"channels": ["a"], "ig_mean": 0.3}]}
    rows = [{"cell": "k100_ie_a0_drop-TSS_tank5", "extra": "drop-TSS_tank5", "window": "R0", "sigma": 0.1,
             "primary": True, "estimator": "cl_pinn", "family": "pinn", "n": 2, "median": 0.4},
            {"cell": "k100_ie_a0_drop-TSS_tank5", "extra": "drop-TSS_tank5", "window": "R0", "sigma": 0.1,
             "primary": True, "estimator": "eks", "family": "observers", "n": 1, "median": 0.45},
            {"cell": "k100_ie_a0_drop-TSS_tank5", "extra": "drop-TSS_tank5", "window": "R0", "sigma": 0.1,
             "primary": False, "estimator": "eks_frozenq", "family": "observers", "n": 1, "median": 0.5},
            {"cell": "k100_ie_a0", "extra": "", "window": "R0", "sigma": 0.1, "primary": True,
             "estimator": "cl_pinn", "family": "pinn", "n": 3, "median": 0.3}]
    tab = vt.si_sensor_subsets(rec, {"rows": rows})
    achieved = [r for r in tab["rows"] if r[0].startswith("achieved")]
    assert [(r[1], r[3]) for r in achieved] == [("cl_pinn (two-seed)", 0.4), ("eks", 0.45)]


def test_practitioner_table_on_the_g4_recoverability_file():
    path = REPO / "results" / "v11" / "analysis" / "recoverability_k000.json"
    if not path.exists():
        pytest.skip("needs the G4 recoverability file")
    rec = vt.load_recoverability(path)
    tab = vt.t3_practitioner(rec)
    channels = set(max(rec["sensor_subsets"], key=lambda s: len(s["channels"]))["channels"])
    assert [r[0] for r in tab["rows"]] == list(vt.TRACK_B)
    for row in tab["rows"]:
        assert len(row) == 6 and (row[2] in channels or row[2] == "none")


def _registry_table(comparisons):
    return {"meta": {}, "rows": [], "crossovers": [], "comparisons": comparisons, "hypotheses": {}}


def test_registry_splits_deciding_labelled_and_reference_counts():
    comps = [
        {"window": "R0", "pinn": "cl_pinn", "comparator": "eks", "outcome": "win", "decides_hypotheses": True},
        {"window": "R0", "pinn": "cl_pinn", "comparator": "eks", "outcome": "win", "decides_hypotheses": False},
        {"window": "R0", "pinn": "cl_pinn", "comparator": "ode_openloop_full", "outcome": "reference",
         "reference_outcome": "loss", "decides_hypotheses": False},
    ]
    reg = vt.numbers_registry(_registry_table(comps), None, None, None, {"total_eq": 1.0})
    assert reg["wtl.R0.cl_pinn.vs.eks"]["value"]["win"] == 1
    assert reg["wtl.R0.cl_pinn.vs.eks.labelled"]["value"]["win"] == 1
    assert reg["wtl_reference.R0.cl_pinn.vs.ode_openloop_full"]["value"]["loss"] == 1
    assert "wtl.R0.cl_pinn.vs.ode_openloop_full" not in reg


def test_residual_keys_keep_the_variant_and_the_registry_refuses_overwrites():
    residual = {"summary": [
        {"cell": "k100_ie_a0", "model": "cl_pinn", "variant": "", "noise": 0.1, "n": 3,
         "median_R0": 0.19, "median_F": 3.0, "median_ratio": 15.4},
        {"cell": "k100_ie_a0", "model": "cl_pinn", "variant": "drop-TSS_tank5", "noise": 0.1, "n": 2,
         "median_R0": 0.18, "median_F": 3.3, "median_ratio": 18.1}]}
    reg = vt.numbers_registry(_registry_table([]), residual, None, None, {"total_eq": 1.0})
    assert reg["residual.k100_ie_a0.cl_pinn.sigma0.10.median_R0"]["value"] == 0.19
    assert reg["residual.k100_ie_a0.cl_pinn[drop-TSS_tank5].sigma0.10.median_R0"]["value"] == 0.18
    twice = {"summary": residual["summary"][:1] * 2}
    with pytest.raises(ValueError, match="written twice"):
        vt.numbers_registry(_registry_table([]), twice, None, None, {"total_eq": 1.0})


def test_sigma_log_table_keeps_the_registered_sigma_only():
    def row(cell, extra, sigma, est, med):
        return {"cell": cell, "extra": extra, "sigma": sigma, "estimator": est, "window": "R0", "median": med,
                "skill": 0.0}

    rows = [row("k100_ic_as_sl04", "sl04", 0.1, "eks", 0.9), row("k100_ic_as", "", 0.1, "eks", 1.0),
            row("k100_ic_as", "", 0.05, "eks", 1.2), row("k100_ic_as_sl04", "sl04", 0.05, "eks", 0.8)]
    tab = vt.si_sigma_log({"rows": rows}, 0.10)
    assert sorted((r[1], r[3]) for r in tab["rows"]) == [(0.4, 0.9), (0.6, 1.0)]


def test_lab_seed_table_lists_the_registered_seed_beside_the_spread():
    table = {"lab_seed_spread": [{"cell": "k050_ic_al1", "sigma": 0.1, "estimator": "eks", "window": "R0",
                                  "registered_seed_value": 0.5, "n": 9, "median": 0.45, "p25": 0.43, "p75": 0.47,
                                  "min": 0.41, "max": 0.49, "registered_seed_outside_range": True}]}
    tab = vt.si_lab_seeds(table)
    assert tab["rows"][0][4] == 0.5 and tab["rows"][0][5] == 9 and tab["rows"][0][-1] is True


def test_tables_refuse_a_missing_validation_file_and_a_mismatched_realistic_file(table_inputs):
    root = table_inputs
    base = ["--root", str(root), "--kinetic-subset", str(root / "analysis" / "kinetic_subset.json"),
            "--prereg-commit", "abc123def456", "--rec-sensors", str(root / "analysis" / "recoverability_k100.json")]
    with pytest.raises(FileNotFoundError):
        vt.main(base + ["--rec-realistic", str(root / "analysis" / "recoverability_k100.json"),
                        "--validation", str(root / "analysis" / "missing.json")])
    table_path = root / "regime_table.json"
    table = json.loads(table_path.read_text(encoding="utf-8"))
    table["meta"]["realistic_k"] = "050"
    table_path.write_text(json.dumps(table), encoding="utf-8")
    with pytest.raises(ValueError, match="gate D3"):
        vt.main(base + ["--rec-realistic", str(root / "analysis" / "recoverability_k100.json"),
                        "--validation", str(root / "analysis" / "recoverability_validation.json")])

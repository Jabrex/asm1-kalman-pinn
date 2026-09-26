"""Regime map: cell grammar, all-seeds rule, crossovers and an end-to-end score on v1.0 runs."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import pytest

from scripts import regime_map as rm

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"


def _clone(src: Path, dst: Path, **updates) -> None:
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("checkpoint.pt", "history.json", "config.yaml"))
    summary = json.loads((dst / "summary.json").read_text(encoding="utf-8"))
    summary.update(updates)
    (dst / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


@pytest.fixture
def v11_tree(tmp_path, monkeypatch):
    """A results/v11 tree built from v1.0 runs, which all score against results/raw."""
    monkeypatch.chdir(REPO)
    root = tmp_path / "v11"
    for k, src in enumerate(("runs", "runs_seed1", "runs_seed2")):
        for model in ("cl_pinn", "pinn"):
            _clone(RES / src / ("%s_sigma0p10" % model), root / "pinn" / ("k000_ie_a0_seed%d" % k)
                   / ("%s_sigma0p10" % model), data_dir="results/raw", seed=k)
    _clone(RES / "runs" / "cl_pinn_sigma0p10", root / "pinn" / "k000_ie_a0_theta_seed0" / "cl_pinn_theta_sigma0p10",
           model="cl_pinn_theta", data_dir="results/raw", learned_multipliers={"muA": 1.1, "bA": 0.9})
    base = root / "baselines" / "k000_ie_a0"
    _clone(RES / "runs" / "persistence_sigma0p10", base / "persistence_sigma0p10", data_dir="results/raw")
    _clone(RES / "runs" / "odesim_sigma0p10", base / "ode_openloop_reduced_sigma0p10",
           model="ode_openloop_reduced", data_dir="results/raw")
    _clone(RES / "runs" / "odesim_sigma0p10", base / "ode_openloop_full_sigma0p10",
           model="ode_openloop_full", data_dir="results/raw")
    obs = root / "observers" / "k000_ie_a0"
    _clone(RES / "runs" / "cl_lstm_sigma0p10", obs / "eks_sigma0p10", model="eks", arch="observer",
           data_dir="results/raw", nis_mean=7.1, q_soluble=0.03, truth_preset="vault20", alpha=1.0,
           influent_mode="exact", anchor="A0")
    _clone(RES / "runs" / "lstm_sigma0p10", obs / "eks_sigma0p10_r01", model="eks", arch="observer",
           data_dir="results/raw", seed=1)
    _clone(RES / "runs" / "persistence_sigma0p10", obs / "ekf_online_sigma0p10", model="ekf_online",
           arch="observer", data_dir="results/raw")
    return root


def test_parse_cell_strips_seed_and_config_suffixes():
    info, seed = rm.parse_cell("k100_ic_as_lstm_seed2")
    assert seed == 2 and info["cell"] == "k100_ic_as"
    assert (info["k"], info["influent"], info["anchor"], info["extra"]) == ("100", "ic", "as", "")
    info, _ = rm.parse_cell("k100_ic_as_sl04")
    assert info["extra"] == "sl04" and not rm.is_grid_cell(info)
    info, _ = rm.parse_cell("k000_off_ie_al1")
    assert info["offsteady"] and info["anchor"] == "al1"
    info, _ = rm.parse_cell("rand07_ie_a0")
    assert info["rand"] == 7 and info["k"] is None


def test_kinetic_coordinate_ignores_alpha_for_named_presets():
    assert rm.kinetic_coordinate("vault20", 1.0) == 0.0
    assert rm.kinetic_coordinate("bsm1_15c", 0.3) == 1.0
    assert rm.kinetic_coordinate("graded", 0.5) == 0.5


def test_all_seeds_rule():
    assert rm.outcome([0.1, 0.2, 0.25], [0.3]) == "win"
    assert rm.outcome([0.4, 0.5], [0.3]) == "loss"
    assert rm.outcome([0.2, 0.4], [0.3]) == "tie"
    assert rm.outcome([0.1, 0.2], [0.15, 0.3]) == "tie"


def test_crossover_alpha_interpolates_the_first_sign_change():
    alpha, status = rm.crossover_alpha([(0.0, -0.3), (0.25, -0.1), (0.5, 0.1), (1.0, 0.3)])
    assert status == "crosses" and alpha == pytest.approx(0.375)
    assert rm.crossover_alpha([(0.0, -0.2), (1.0, -0.1)]) == (None, "never_behind")
    assert rm.crossover_alpha([(0.0, 0.2), (1.0, 0.3)]) == (None, "behind_at_first_alpha")


def test_consistency_check_rejects_a_misfiled_run(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    root = tmp_path / "v11"
    _clone(RES / "runs" / "persistence_sigma0p10", root / "baselines" / "k100_ie_a0" / "persistence_sigma0p10",
           data_dir="results/raw", truth_preset="vault20")
    with pytest.raises(ValueError, match="filed under cell k100_ie_a0"):
        rm.discover(root)


def test_score_end_to_end(v11_tree):
    result = rm.score(v11_tree, RES / "runs", "track_b_nrmse_fixed")
    rows = result["table"]["rows"]
    get = {(r["estimator"], r["window"]): r for r in rows if r["cell"] == "k000_ie_a0" and r["sigma"] == 0.1}
    bands = json.loads((RES / "seed_bands.json").read_text(encoding="utf-8"))
    cl_f = get[("cl_pinn", "F")]
    assert cl_f["n"] == 3 and cl_f["seeds"] == [0, 1, 2]
    assert cl_f["median"] == pytest.approx(bands["cl_pinn|0.10|holdout"]["track_b_nrmse_fixed"]["median"], rel=1e-12)
    assert get[("persistence", "R0")]["skill"] == pytest.approx(0.0)
    assert get[("ode_openloop_reduced", "F")]["gap_closed"] == pytest.approx(1.0)
    assert get[("lstm_v10", "F")]["family"] == "legacy"
    assert not get[("ekf_online", "F")]["primary"] and not get[("ode_openloop_full", "F")]["primary"]
    assert get[("eks", "R0")]["diagnostics"] == {"nis_mean": 7.1, "q_soluble": 0.03}
    assert {w for (_, w) in get} == {"R0", "R2", "F"}
    comp = {(c["pinn"], c["comparator"], c["window"]): c["outcome"] for c in result["table"]["comparisons"]
            if c["sigma"] == 0.1}
    assert comp[("cl_pinn", "pinn", "F")] == "win"
    assert comp[("pinn", "cl_pinn", "F")] == "loss"
    win = {(w["cell"], w["window"]): w["estimator"] for w in result["table"]["winners"] if w["sigma"] == 0.1}
    assert win[("k000_ie_a0", "F")] == "ode_openloop_reduced"
    spread = [s for s in result["table"]["realisation_spread"] if s["estimator"] == "eks" and s["window"] == "F"]
    # Sections 2 and 9: realisations 1-9 next to realisation 0, never pooled (the fixture has r01 only)
    assert spread and spread[0]["n"] == 1 and spread[0]["realisations"] == [1]
    assert spread[0]["realisation0"] == get[("eks", "F")]["median"]
    rec = result["table"]["parameter_recovery"]
    assert rec[0]["estimator"] == "cl_pinn_theta" and rec[0]["true_log_ratio"] == {"bA": 0.0, "muA": 0.0}
    assert rec[0]["estimates"][0]["log_multiplier"]["muA"] == pytest.approx(math.log(1.1))
    entries = result["states"]["entries"]
    assert {e["window"] for e in entries} == {"R0", "F"}
    assert len(entries[0]["median"]) == 5 and len(entries[0]["median"][0]) == 11


def test_outputs_are_written(v11_tree):
    result = rm.score(v11_tree, RES / "runs", "track_b_nrmse_fixed")
    paths = rm.write_outputs(result, v11_tree)
    assert all(p.exists() for p in paths)
    json.loads((v11_tree / "regime_table.json").read_text(encoding="utf-8"))
    assert "Win / tie / loss" in (v11_tree / "regime_table.md").read_text(encoding="utf-8")


# -- Section 9 rules and the G6 layout (added in G7 after checking the plan code against the
#    pre-registration and the real run tree) ------------------------------------------------
def _prow(estimator, seeds, values, family="pinn", diverged=(), failed=False, cell="k000_ie_a0"):
    return {"cell": cell, "estimator": estimator, "family": family, "window": "R0", "sigma": 0.1,
            "n": len(values), "seeds": list(seeds), "values": list(values),
            "decision_values": [math.inf if i in diverged else v for i, v in enumerate(values)],
            "diverged": ["x" for _ in diverged], "failed": failed}


def test_parse_cell_accepts_the_offsteady_name_of_the_g6_tree():
    info, seed = rm.parse_cell("k000off_ie_al1")
    assert seed is None and info["offsteady"] and info["k"] == "000"
    assert (info["influent"], info["anchor"], info["extra"]) == ("ie", "al1", "")


def test_two_pinn_models_are_compared_seed_by_seed():
    """Section 9: pairs by seed number, which the all-seeds rule would call a tie."""
    assert rm.paired_outcome({0: 0.1, 1: 0.3}, {0: 0.2, 1: 0.4}) == "win"
    assert rm.outcome([0.1, 0.3], [0.2, 0.4]) == "tie"
    assert rm.paired_outcome({0: 0.3, 1: 0.5}, {0: 0.2, 1: 0.4}) == "loss"
    assert rm.paired_outcome({0: 0.1, 1: 0.5}, {0: 0.2, 1: 0.4}) == "tie"
    assert rm.paired_outcome({0: 0.1}, {1: 0.2}) == "not_decided"
    result, rule = rm.compare_rows(_prow("cl_pinn", [0, 1], [0.1, 0.3]), _prow("pinn", [0, 1], [0.2, 0.4]))
    assert (result, rule) == ("win", "paired seeds")


def test_single_valued_comparator_and_multi_seed_lstm():
    p = _prow("cl_pinn", [0, 1, 2], [0.1, 0.2, 0.25])
    assert rm.compare_rows(p, _prow("eks", [0], [0.3], family="observers")) == ("win", "every seed against one value")
    assert rm.compare_rows(p, _prow("eks", [0], [0.22], family="observers"))[0] == "tie"
    lstm = _prow("lstm", [0, 1, 2], [0.24, 0.5, 0.6])
    assert rm.compare_rows(p, lstm) == ("tie", rm.ADDED_RULE)
    assert rm.single_outcome(0.2, 0.2) == "tie" and rm.single_outcome(0.1, 0.2) == "win"


def test_diverged_runs_lose_and_diverged_comparators_leave_the_cell_undecided():
    good = _prow("cl_pinn", [0, 1, 2], [0.1, 0.1, 0.1])
    bad_obs = _prow("eks_frozenq", [0], [0.05], family="observers", diverged=(0,))
    assert rm.compare_rows(good, bad_obs)[0] == "not_decided"
    all_bad = _prow("cl_pinn", [0, 1, 2], [0.01, 0.01, 0.01], diverged=(0, 1, 2))
    assert rm.compare_rows(all_bad, _prow("eks", [0], [0.5], family="observers"))[0] == "loss"
    rows = [dict(good, primary=True, median=0.1), dict(bad_obs, primary=True, median=0.05)]
    assert rm.winners(rows)[0]["estimator"] == "cl_pinn"


def test_divergence_rule_of_section_5():
    assert rm.divergence({"nis_mean": 22.0, "n_measured": 7, "finite": True, "diverged": False})
    assert rm.divergence({"nis_mean": 20.9, "n_measured": 7, "finite": True, "diverged": False}) is None
    assert rm.divergence({"finite": False}) == "non-finite estimate"
    assert rm.divergence({"diverged": True}) == "filter flagged divergence"


def test_labels_for_two_seed_and_one_seed_rows():
    assert rm.seed_label(_prow("cl_pinn", [0, 1], [0.1, 0.2])) == "two-seed"
    assert rm.seed_label(_prow("cl_pinn_theta", [0], [0.1])) == "one-seed"
    assert rm.seed_label(_prow("cl_pinn", [0, 1, 2], [0.1, 0.2, 0.3])) == ""
    assert rm.seed_label(_prow("eks", [0], [0.1], family="observers")) == ""


def test_reference_cells_for_sensor_and_settler_variants():
    def ref(cell):
        return rm.reference_cell(rm.parse_cell(cell)[0])

    assert ref("k100_ie_a0_drop-TSS_tank5") == ("k100_ie_a0", "k100_ie_a0")
    assert ref("k100_ie_a0_add-S_NH_tank4") == ("k100_ie_a0", "k100_ie_a0")
    assert ref("k100_ie_a0_ideal") == ("k100_ie_a0", "k100_ie_a0_ideal")
    assert ref("k050_ic_al1_lab03") == ("k050_ic_al1_lab03", "k050_ic_al1_lab03")
    assert ref("k100_ic_as_sl04") == ("k100_ic_as_sl04", "k100_ic_as_sl04")


def test_a_pinn_sensor_variant_joins_the_observer_cell_of_its_sensor_set(v11_tree):
    _clone(RES / "runs" / "cl_pinn_sigma0p10",
           v11_tree / "pinn" / "k000_ie_a0_seed0" / "cl_pinn_sigma0p10_drop-TSS_tank5",
           data_dir="results/raw", variant="drop-TSS_tank5")
    refs = [r for r in rm.discover(v11_tree) if r.summary.get("variant")]
    assert len(refs) == 1
    assert refs[0].cell == "k000_ie_a0_drop-TSS_tank5" and refs[0].estimator == "cl_pinn"
    assert refs[0].info["extra"] == "drop-TSS_tank5"


def test_a_failed_run_is_listed_and_its_comparisons_are_not_decided(v11_tree):
    failed = v11_tree / "observers" / "k000_ie_a0" / "ekf_sigma0p10"
    failed.mkdir(parents=True)
    (failed / "error.txt").write_text("LinAlgError: singular matrix", encoding="utf-8")
    result = rm.score(v11_tree, RES / "runs", "track_b_nrmse_fixed")
    table = result["table"]
    assert [f["estimator"] for f in table["meta"]["failed_runs"]] == ["ekf"]
    ekf = [r for r in table["rows"] if r["estimator"] == "ekf" and r["window"] == "R0"][0]
    assert ekf["failed"] and ekf["values"] == [math.inf]
    comp = [c for c in table["comparisons"] if c["pinn"] == "cl_pinn" and c["comparator"] == "ekf"]
    assert comp and all(c["outcome"] == "not_decided" for c in comp)
    assert all(w["estimator"] != "ekf" for w in table["winners"])


def test_hypotheses_are_evaluated_with_the_registered_rules(v11_tree):
    hyp = rm.score(v11_tree, RES / "runs", "track_b_nrmse_fixed")["table"]["hypotheses"]
    assert hyp["window"] == "R0" and hyp["sigma"] == 0.1
    assert hyp["H7"]["rule"] == "paired seeds" and hyp["H7"]["status"] in ("supported", "not supported")
    assert hyp["H1"]["cells"]["k000_ie_a0"]["outcome"] in ("win", "tie", "loss")
    assert hyp["H1"]["cells"]["k000_ic_a0"]["outcome"] == "missing"
    assert hyp["H3"]["status"] == "not decided"  # no K1 cells in this tree: missing, not a loss
    assert hyp["H4"]["status"] == "not decided"
    assert hyp["H3"]["k0_theta_sanity"]["multipliers"] == {0: {"muA": 1.1, "bA": 0.9}}
    assert hyp["H3"]["k0_theta_sanity"]["pass"] is True


def test_h5_ratio_is_the_mean_of_per_component_ratios():
    num = {"a": 2.0, "b": 4.0}
    den = {"a": 1.0, "b": 1.0}
    assert rm._group_ratio(num, den, ("a", "b")) == pytest.approx(3.0)


def test_heatmaps_are_written(v11_tree):
    result = rm.score(v11_tree, RES / "runs", "track_b_nrmse_fixed")
    rm.write_outputs(result, v11_tree)
    written = rm.plot(v11_tree, 0.10, None)
    names = {p.name for p in written}
    assert {"fig6_regime_map.png", "fig6_regime_map.pdf", "graphical_abstract_regime.png",
            "figS_regime_map_all_cells.pdf"} <= names
    assert rm.cell_label("k050_ic_al1") == "K.5 Ic Al1"
    assert rm.cell_label("k025_ie_a0") == "K.25 Ie A0"


# -- review fixes (independent G7 review, 2026-09-26) -------------------------------------------------
def test_more_information_rows_are_references_never_winners():
    p = _prow("cl_pinn", [0, 1, 2], [0.1, 0.1, 0.1])
    full = _prow("ode_openloop_full", [0], [0.001], family="baselines")
    online_frozen = _prow("ekf_online_frozenq", [0], [0.001], family="observers")
    assert rm.compare_rows(p, full) == ("reference", rm.REFERENCE_RULE)
    assert rm.compare_rows(p, online_frozen)[0] == "reference"
    rows = [dict(p, median=0.1, primary=True), dict(full, median=0.001, primary=rm.is_primary("ode_openloop_full"))]
    comps = rm.comparisons(rows)
    assert comps[0]["outcome"] == "reference" and comps[0]["reference_outcome"] == "loss"
    assert rm.comparison_kind(comps[0]) == "reference" and not comps[0]["decides_hypotheses"]
    assert not rm.is_primary("ekf_online_frozenq")


def test_labelled_and_added_rule_comparisons_do_not_decide():
    rows = [dict(_prow("cl_pinn", [0, 1], [0.1, 0.2]), median=0.15, primary=True),
            dict(_prow("eks", [0], [0.3], family="observers"), median=0.3, primary=True),
            dict(_prow("lstm", [0, 1, 2], [0.4, 0.5, 0.6]), median=0.5, primary=True)]
    kinds = {c["comparator"]: rm.comparison_kind(c) for c in rm.comparisons(rows) if c["pinn"] == "cl_pinn"}
    assert kinds == {"eks": "labelled", "lstm": "labelled"}


def test_winners_carry_the_seed_label():
    rows = [dict(_prow("cl_pinn", [0, 1], [0.1, 0.2]), median=0.15, primary=True),
            dict(_prow("eks", [0], [0.3], family="observers"), median=0.3, primary=True)]
    w = rm.winners(rows)[0]
    assert (w["estimator"], w["n"], w["label"]) == ("cl_pinn", 2, "two-seed")


def test_crossover_finds_a_later_negative_to_positive_change():
    assert rm.crossover_alpha([(0.0, 0.1), (0.5, -0.1), (1.0, 0.1)]) == (pytest.approx(0.75), "crosses")
    assert rm.crossover_alpha([(0.0, 0.1), (0.5, -0.1), (1.0, -0.2)]) == (None, "no_negative_to_positive_change")


def test_lab_seed_spread_sits_next_to_the_registered_seed():
    def row(cell, extra, med):
        return {"cell": cell, "extra": extra, "sigma": 0.1, "estimator": "eks", "window": "R0", "median": med}

    rows = [row("k050_ic_al1", "", 0.5)] + [row("k050_ic_al1_lab%02d" % i, "lab%02d" % i, 0.4 + 0.01 * i)
                                           for i in range(1, 10)]
    spread = rm.lab_seed_spread(rows)
    assert len(spread) == 1 and spread[0]["n"] == 9 and spread[0]["cell"] == "k050_ic_al1"
    assert spread[0]["registered_seed_value"] == 0.5 and spread[0]["registered_seed_outside_range"] is True
    assert spread[0]["median"] == pytest.approx(0.45)


def test_parameter_recovery_reports_the_registered_absolute_log_error(v11_tree):
    rec = rm.score(v11_tree, RES / "runs", "track_b_nrmse_fixed")["table"]["parameter_recovery"][0]
    assert rec["estimates"][0]["abs_log_error"]["muA"] == pytest.approx(abs(math.log(1.1)))
    assert rec["median_abs_log_error"]["bA"] == pytest.approx(abs(math.log(0.9)))


def test_failed_run_parsing_reads_realisations_and_skips_crash_attempts(tmp_path):
    obs = tmp_path / "observers" / "k000_ie_a0" / "ekf_sigma0p10_r03"
    obs.mkdir(parents=True)
    (obs / "error.txt").write_text("boom", encoding="utf-8")
    crash = tmp_path / "pinn" / "k000_ie_a0_seed1" / "cl_pinn_sigma0p10__crash1"
    crash.mkdir(parents=True)
    (crash / "error.txt").write_text("CUDA error", encoding="utf-8")
    failed = rm.failed_runs(tmp_path)
    assert [(f["estimator"], f["realisation"], f["cell"]) for f in failed] == [("ekf", 3, "k000_ie_a0")]
    assert rm.infrastructure_crashes(tmp_path) == [str(crash).replace("\\", "/")]


def test_a_failed_observer_job_stops_the_scoring(v11_tree):
    (v11_tree / "observers" / "k000_ie_a0" / "error_k000_ie_a0_sigma0p10_r00_tuned.txt").write_text(
        "LinAlgError", encoding="utf-8")
    with pytest.raises(ValueError, match="failed observer jobs"):
        rm.score(v11_tree, RES / "runs", "track_b_nrmse_fixed")


def test_h7_is_not_decided_when_the_comparator_failed(v11_tree):
    failed = v11_tree / "pinn" / "k000_ie_a0_seed0" / "pinn_sigma0p10"
    shutil.rmtree(failed)
    failed.mkdir()
    (failed / "error.txt").write_text("NaN loss", encoding="utf-8")
    hyp = rm.score(v11_tree, RES / "runs", "track_b_nrmse_fixed")["table"]["hypotheses"]
    assert hyp["H7"]["outcome"] == "not_decided" and hyp["H7"]["status"] == "not decided"
    assert hyp["H7"]["consequence"] == "curriculum claim withdrawn in full"

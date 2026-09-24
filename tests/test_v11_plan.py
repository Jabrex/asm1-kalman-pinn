"""The v1.1 run plan: queue sizes, charging rules and run-id agreement with run_all."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import v11_plan as vp


def _phase_totals(jobs):
    out: dict[str, float] = {}
    for job in jobs:
        out[job.phase] = round(out.get(job.phase, 0.0) + job.eq(), 6)
    return out


def test_budget_constants():
    assert vp.CORE_CAP == 45.05
    assert vp.CORE_CAP <= vp.HARD_CAP == 50.0


def test_charging_rules():
    assert vp.nominal_eq("cl_pinn", "quick") == 0.2
    assert vp.nominal_eq("lstm", "quick") == 0.2
    assert vp.nominal_eq("pinn", "full") == 1.0
    assert vp.nominal_eq("lstm", "full") == 0.35


def _realistic() -> str:
    """The realistic kinetics whose configs exist: gate D3's choice once it has run, else k100."""
    return vp.realistic_k() if (vp.REPO / vp.GATE_D3).exists() else "k100"


def test_arch_table_matches_model_specs():
    from src.train.run import MODEL_SPECS

    for model, arch in vp.ARCH.items():
        assert MODEL_SPECS[model]["arch"] == arch


def test_smoke_queue_is_nine_quick_runs():
    jobs = vp.smoke_queue(_realistic())
    runs = [(j.out_dir, r) for j in jobs for r in j.expected_run_ids()]
    assert len(runs) == 9 and len(set(runs)) == 9
    assert round(sum(j.eq() for j in jobs), 6) == 1.8
    assert all(j.profile == "quick" and j.out_dir.startswith(vp.SMOKE_ROOT + "/") for j in jobs)
    assert all(j.noise == (0.10,) for j in jobs)


def test_core_queue_totals():
    totals = _phase_totals(vp.core_queue(_realistic()))
    assert totals == {"E4": 9.0, "E5": 6.0, "E6": 14.05, "E7": 4.0, "E3": 4.0}
    grand = round(sum(totals.values()) + 1.8 + vp.G1_SMOKE_EQ, 6)
    assert grand == 39.05 and round(grand - vp.G1_SMOKE_EQ, 6) == vp.G6_EQ


def test_core_queue_run_dirs_are_unique_and_seeded():
    dirs = [Path(j.out_dir, r).as_posix() for j in vp.core_queue(_realistic()) for r, _ in j.expected_run_ids()]
    assert len(dirs) == len(set(dirs))
    assert all(Path(d).parent.name.rsplit("_seed", 1)[1] in {"0", "1", "2"} for d in dirs)


def test_realistic_switch_moves_only_e6():
    k100 = {(j.phase, j.stem) for j in vp.core_queue("k100")}
    k050 = {(j.phase, j.stem) for j in vp.core_queue("k050")}
    assert {s for _, s in k100 - k050} == {"k100_ic_a0", "k100_ic_as", "k100_ic_al1", "k100_ic_as_lstm"}
    assert {s for _, s in k050 - k100} == {"k050_ic_a0", "k050_ic_as", "k050_ic_al1", "k050_ic_as_lstm"}
    with pytest.raises(ValueError):
        vp.core_queue("k075")


def test_sigma_helpers_match_run_all():
    from scripts import run_all

    for s in (0.0, 0.05, 0.1, 0.15):
        assert vp.sigma_tag(s) == run_all.sigma_tag(s)
    assert vp.sigma_of("cl_pinn_sigma0p15_drop-S_O_tank3") == 0.15


def test_run_all_list_agrees_with_plan():
    for job in vp.smoke_queue(_realistic()) + vp.core_queue(_realistic()):
        assert sorted(vp.planned_runs(job)) == sorted(job.expected_run_ids()), job

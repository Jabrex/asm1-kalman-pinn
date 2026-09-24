"""Budget guard, planning against finished and failed runs, recovery after a kill."""

from __future__ import annotations

import json

import numpy as np
import pytest
import yaml

from scripts import gpu_ledger as gl
from scripts import run_core as rc
from scripts.v11_plan import Job


def _finished(d):
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps({"final_losses": {"total": 1.0}, "train_seconds": 540.0}),
                                    encoding="utf-8")
    np.savez_compressed(d / "predictions.npz", train=np.zeros((1, 5, 14)))


def _job(tmp_path):
    config = tmp_path / "k000_ie_a0.yaml"
    config.write_text(yaml.safe_dump({"models": ["cl_pinn", "pinn"]}), encoding="utf-8")
    return Job("E4", config.as_posix(), 0, "full", (tmp_path / "k000_ie_a0_seed0").as_posix(),
               ("cl_pinn", "pinn"), (0.1,))


def test_budget_allows():
    assert rc.budget_allows(44.0, 0.0, 1.0, 45.05)
    assert not rc.budget_allows(44.1, 0.0, 1.0, 45.05)
    assert not rc.budget_allows(43.0, 1.5, 1.0, 45.05)


def test_select_deduplicates_the_pair():
    jobs = rc.select(["smoke-pair", "smoke"], "k100")
    assert len(jobs) == 8 and round(sum(j.eq() for j in jobs), 6) == 1.8


def test_plan_skips_done_blocks_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(rc, "planned_runs", lambda job: job.expected_run_ids())
    job = _job(tmp_path)
    pending, blocked = rc.plan([job], retry_infra=False)
    assert len(pending) == 1 and pending[0].eq == 2.0 and not blocked
    _finished(tmp_path / "k000_ie_a0_seed0" / "cl_pinn_sigma0p10")
    pending, _ = rc.plan([job], retry_infra=False)
    assert pending[0].todo == [("pinn_sigma0p10", "pinn")] and pending[0].eq == 1.0
    failed = tmp_path / "k000_ie_a0_seed0" / "pinn_sigma0p10"
    failed.mkdir()
    (failed / "error.txt").write_text("ValueError: bad channel", encoding="utf-8")
    pending, blocked = rc.plan([job], retry_infra=False)
    assert not pending and blocked == [failed.as_posix()]
    (failed / "error.txt").write_text("RuntimeError: CUDA error: unspecified launch failure", encoding="utf-8")
    pending, _ = rc.plan([job], retry_infra=True)
    assert pending[0].todo == [("pinn_sigma0p10", "pinn")]
    assert (tmp_path / "k000_ie_a0_seed0" / "pinn_sigma0p10__crash1" / "error.txt").exists()


def test_execute_stops_at_the_cap_without_launching(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.csv"
    for i in range(3):
        d = tmp_path / ("done%d" % i)
        _finished(d)
        gl.record(d, "E4", 15.0, 9.0, 1, "cl_pinn", "full", 0.1, 0, ledger)
    monkeypatch.setattr(rc, "planned_runs", lambda job: job.expected_run_ids())
    pending, _ = rc.plan([_job(tmp_path)], retry_infra=False)

    def refuse(*args, **kwargs):
        raise AssertionError("a process was launched above the cap")

    monkeypatch.setattr(rc.subprocess, "Popen", refuse)
    assert rc.execute(pending, 1, 45.05, ledger, tmp_path / "inflight.json") == 2


def test_recover_interrupted(tmp_path):
    ledger, inflight = tmp_path / "ledger.csv", tmp_path / "inflight.json"
    out = tmp_path / "k000_ie_a0_seed0"
    _finished(out / "cl_pinn_sigma0p10")
    inflight.write_text(json.dumps([{"phase": "E4", "out_dir": out.as_posix(), "profile": "full", "seed": 0,
                                     "workers": 1, "todo": [["cl_pinn_sigma0p10", "cl_pinn"],
                                                            ["pinn_sigma0p10", "pinn"]]}]), encoding="utf-8")
    rows = rc.recover_interrupted(ledger, inflight)
    assert [r["status"] for r in rows] == ["ok", "interrupted"]
    assert gl.total_eq(gl.read(ledger)) == 2.0 and not inflight.exists()

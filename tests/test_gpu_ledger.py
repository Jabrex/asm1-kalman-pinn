"""Ledger rows: status classes, attempt counting, refusal of a second success."""

from __future__ import annotations

import json

import numpy as np
import pytest

from scripts import gpu_ledger as gl


def _run_dir(root, name, losses=None, error=None, finished=True):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    if finished:
        summary = {"final_losses": losses if losses is not None else {"total": 1.0, "stage": "stage4"},
                   "train_seconds": 540.0, "model": "cl_pinn", "profile": "full", "noise": 0.1, "seed": 0}
        (d / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        np.savez_compressed(d / "predictions.npz", train=np.zeros((2, 5, 14)))
    if error:
        (d / "error.txt").write_text(error, encoding="utf-8")
    return d


def test_run_status_classes(tmp_path):
    assert gl.run_status(_run_dir(tmp_path, "a")) == "ok"
    assert gl.run_status(_run_dir(tmp_path, "b", losses={"total": float("nan")})) == "failed_numeric"
    assert gl.run_status(_run_dir(tmp_path, "c", error="RuntimeError: CUDA error: launch failure",
                                  finished=False)) == "error_infra"
    assert gl.run_status(_run_dir(tmp_path, "d", error="ValueError: bad channel", finished=False)) == "error_other"
    assert gl.run_status(tmp_path / "e") == "missing"


def test_attempts_and_second_success_refused(tmp_path):
    ledger = tmp_path / "ledger.csv"
    d = _run_dir(tmp_path, "cl_pinn_sigma0p10", error="CUDA out of memory", finished=False)
    first = gl.record(d, "E4", 1.0, 3.0, 1, "cl_pinn", "full", 0.1, 0, ledger)
    assert first["attempt"] == 1 and first["status"] == "error_infra"
    (d / "error.txt").unlink()
    _run_dir(tmp_path, "cl_pinn_sigma0p10")
    second = gl.record(d, "E4", 1.0, 9.0, 1, "cl_pinn", "full", 0.1, 0, ledger)
    assert second["attempt"] == 2 and second["status"] == "ok"
    with pytest.raises(ValueError):
        gl.record(d, "E4", 1.0, 9.0, 1, "cl_pinn", "full", 0.1, 0, ledger)
    rows = gl.read(ledger)
    assert gl.total_eq(rows) == 2.0 and gl.by_phase(rows) == {"E4": 2.0}


def test_status_override_and_unknown_status(tmp_path):
    ledger = tmp_path / "ledger.csv"
    d = tmp_path / "pinn_sigma0p10"
    row = gl.record(d, "E4", 1.0, 0.0, 1, "pinn", "full", 0.1, 0, ledger, status="interrupted")
    assert row["status"] == "interrupted"
    with pytest.raises(ValueError):
        gl.record(d, "E4", 1.0, 0.0, 1, "pinn", "full", 0.1, 0, ledger, status="lost")

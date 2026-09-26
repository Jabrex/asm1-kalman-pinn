"""Acceptance classes of the GPU core run directories."""

from __future__ import annotations

import json

import numpy as np

from scripts import check_core as ck


def _complete(d, losses=None):
    d.mkdir(parents=True)
    for name in ck.REQUIRED:
        (d / name).write_bytes(b"")
    (d / "summary.json").write_text(json.dumps({"final_losses": losses or {"total": 1.0}}), encoding="utf-8")
    np.savez_compressed(d / "predictions.npz", train=np.zeros((1, 5, 14)))


def test_classes(tmp_path):
    ok = tmp_path / "cl_pinn_sigma0p10"
    _complete(ok)
    assert ck.classify(ok, []) == "ok"
    nan = tmp_path / "pinn_sigma0p10"
    _complete(nan, {"total": float("nan")})
    assert ck.classify(nan, []) == "failed_numeric"
    err = tmp_path / "lstm_sigma0p10"
    err.mkdir()
    (err / "error.txt").write_text("CUDA error", encoding="utf-8")
    assert ck.classify(err, []) == "error_unlogged"
    assert ck.classify(err, [{"run_dir": err.as_posix(), "status": "error_infra"}]) == "error_logged"
    assert ck.classify(tmp_path / "none", []) == "missing"

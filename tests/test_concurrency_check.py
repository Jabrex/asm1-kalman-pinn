"""Two GPU processes only if predictions agree and throughput gains."""

from __future__ import annotations

import numpy as np
import pytest

from scripts import concurrency_check as cc


def test_max_rel_diff(tmp_path):
    x = np.ones((4, 5, 14))
    np.savez(tmp_path / "a.npz", train=x * (1 + 1e-6), holdout=x)
    np.savez(tmp_path / "b.npz", train=x, holdout=x)
    assert cc.max_rel_diff(tmp_path / "a.npz", tmp_path / "b.npz") == pytest.approx(1e-6, rel=1e-3)
    np.savez(tmp_path / "c.npz", train=x[:3], holdout=x)
    assert cc.max_rel_diff(tmp_path / "c.npz", tmp_path / "b.npz") == float("inf")


def test_decide():
    assert cc.decide(1e-7, 108.0, (120.0, 125.0))["workers"] == 2
    assert cc.decide(1e-3, 108.0, (120.0, 125.0))["workers"] == 1
    assert cc.decide(1e-7, 108.0, (200.0, 190.0))["workers"] == 1

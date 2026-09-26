"""make_baselines CLI: the default invocation reproduces the v1.0 rows bit for bit.

The persistence check is instant and always runs. The open-loop check integrates
14 days twice (about 70 s), so it runs only with ``ASM1_SLOW_TESTS=1``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from scripts import make_baselines

REPO = Path(__file__).resolve().parents[1]
SLOW = os.environ.get("ASM1_SLOW_TESTS", "").strip().lower() not in {"", "0", "false", "no"}


def _require_v10(row: str) -> Path:
    ref = REPO / "results" / "runs" / ("%s_sigma0p00" % row) / "predictions.npz"
    sim = REPO / "results" / "raw" / "sim_dry.npz"
    if not (ref.exists() and sim.exists()):
        pytest.skip("needs the v1.0 baseline run directories and results/raw/sim_*.npz")
    return ref.parent.parent


def _assert_same_arrays(new_dir: Path, ref_dir: Path, row: str) -> None:
    for sigma in make_baselines.SIGMAS:
        name = "%s_sigma%s" % (row, make_baselines.sigma_tag(sigma))
        with np.load(new_dir / name / "predictions.npz") as a, np.load(ref_dir / name / "predictions.npz") as b:
            assert sorted(a.files) == sorted(b.files)
            for key in a.files:
                assert np.array_equal(a[key], b[key]), (name, key)


def test_default_persistence_rows_are_bit_identical(tmp_path, monkeypatch):
    ref = _require_v10("persistence")
    monkeypatch.chdir(REPO)
    make_baselines.main(["--out", str(tmp_path), "--rows", "persistence"])
    _assert_same_arrays(tmp_path, ref, "persistence")


@pytest.mark.skipif(not SLOW, reason="set ASM1_SLOW_TESTS=1 to integrate the open-loop row")
def test_default_invocation_is_bit_identical_including_openloop(tmp_path, monkeypatch):
    ref = _require_v10("odesim")
    monkeypatch.chdir(REPO)
    make_baselines.main(["--out", str(tmp_path)])
    _assert_same_arrays(tmp_path, ref, "persistence")
    _assert_same_arrays(tmp_path, ref, "odesim")


def test_dry_only_data_dir_writes_rows_without_rain(tmp_path, monkeypatch):
    """Random truths (results/raw_rand/<i>) hold the dry scenario at sigma 0.10 only."""
    src = REPO / "results" / "raw_rand" / "00"
    if not (src / "sim_dry.npz").exists():
        pytest.skip("needs results/raw_rand/00")
    monkeypatch.chdir(REPO)
    data = tmp_path / "raw"
    data.mkdir()
    for name in ("sim_dry.npz", "obs_dry_sigma0p10.npz"):
        (data / name).write_bytes((src / name).read_bytes())
    out = tmp_path / "rows"
    make_baselines.main(["--data-dir", str(data), "--out", str(out), "--ras-filter-window", "4",
                         "--rows", "persistence", "ode_openloop_reduced", "--sigmas", "0.10"])
    for row in ("persistence", "ode_openloop_reduced"):
        with np.load(out / ("%s_sigma0p10" % row) / "predictions.npz") as p:
            assert sorted(p.files) == ["holdout", "train"], row
            assert np.isfinite(p["train"]).all() and p["train"].shape[1:] == (5, 14), row


def test_anchor_file_replaces_the_truth_start(tmp_path, monkeypatch):
    _require_v10("persistence")
    monkeypatch.chdir(REPO)
    z0 = np.full((5, 14), 7.0)
    anchor = tmp_path / "A.npz"
    np.savez(anchor, z0_mean=z0, z0_rel_std=np.full((5, 14), 0.3))
    out = tmp_path / "rows"
    make_baselines.main(["--out", str(out), "--rows", "persistence", "--anchor-file", str(anchor)])
    with np.load(out / "persistence_sigma0p10" / "predictions.npz") as p:
        assert np.all(p["train"] == 7.0) and np.all(p["rain"] == 7.0)
    # without settler_init the open-loop row must refuse rather than borrow truth
    with pytest.raises(SystemExit):
        make_baselines.main(["--out", str(out), "--rows", "odesim", "--anchor-file", str(anchor)])

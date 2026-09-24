"""v1.1 Trainer extensions (group G5): anchors, influent views, kinetic
multipliers, channel subsets, the nominal-plant guarantee and the regime configs.

Everything runs on CPU in float64 with a few optimisation steps at most.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from anchor_utils import write_anchor  # noqa: E402
from v10_reference import CASES, short_run_losses  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "results" / "raw"
RAW_K100 = REPO / "results" / "raw_k100"
REFERENCE = Path(__file__).resolve().parent / "data" / "v10_three_step_losses.json"
#: Any valid kinetic names; the pre-registered subset lives in kinetic_subset.json.
THETA = ["muA", "bA", "muH", "bH"]
STRICT = os.environ.get("ASM1_STRICT_TESTS", "").strip().lower() not in {"", "0", "false", "no"}


def _require(path: Path) -> Path:
    if (path / "obs_dry_sigma0p10.npz").exists():
        return path
    message = "%s is missing - generate it first (group G2, scripts.generate_data)" % path
    if STRICT:
        pytest.fail("ASM1_STRICT_TESTS is set: " + message)
    pytest.skip(message)
    raise AssertionError("unreachable")


def _trainer(tmp_path: Path, data_dir: Path = RAW, **overrides):
    from src.train.run import RunConfig, Trainer

    fields = dict(
        run_id="_g5_probe", model="cl_pinn", noise=0.10, seed=0, profile="quick",
        steps_quick=3, log_every=1, device="cpu", dtype="float64",
        data_dir=str(data_dir), out_dir=str(tmp_path / "runs"),
    )
    fields.update(overrides)
    return Trainer(RunConfig(**fields))


def _first_difference(got, ref) -> str:
    for case in ref:
        for i, (a, b) in enumerate(zip(got.get(case, []), ref[case])):
            if a != b:
                keys = [k for k in b if a.get(k) != b[k]]
                return "%s step %d differs in %s: got %s, reference %s" % (
                    case, i, keys, {k: a.get(k) for k in keys}, {k: b[k] for k in keys})
        if len(got.get(case, [])) != len(ref[case]):
            return "%s has %d records, reference %d" % (case, len(got.get(case, [])), len(ref[case]))
    return "case sets differ: %s vs %s" % (sorted(got), sorted(ref))


# ----------------------------------------------------------------------------
# Task 5.1 - defaults reproduce v1.0
# ----------------------------------------------------------------------------
def test_default_config_reproduces_v10_losses_bit_for_bit():
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    assert ref["label"] == "v1.0.0"
    got = short_run_losses(RAW)
    assert got == ref["cases"], _first_difference(got, ref["cases"])

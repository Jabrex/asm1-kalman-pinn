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


# ----------------------------------------------------------------------------
# Task 5.2 - losses.py additions
# ----------------------------------------------------------------------------
def test_kinetic_parameter_names_are_the_fifteen_rate_parameters():
    from src.asm1.vault_loader import vault
    from src.models.losses import kinetic_parameter_names

    assert kinetic_parameter_names(vault()) == (
        "kh", "KX", "etah", "muH", "etag", "Ks", "bH", "KO_H", "KNO", "KNH_H",
        "muA", "bA", "ka", "KO_A", "KNH",
    )


def test_total_forwards_kinetic_overrides_to_the_physics_term(tmp_path):
    trainer = _trainer(tmp_path)
    stage = trainer.schedule.stages[-1]
    batch = trainer._stage_tensors(stage)
    colloc = trainer._collocation(stage, batch)
    z_c, dz_c = trainer.model.state_and_derivative(colloc["t"], colloc["q_in"], colloc["z_in"])
    kw = dict(
        weights=stage.weights_end, t=colloc["t"].detach(), z=z_c, dz_dt=dz_c,
        q_in=colloc["q_in"], z_in=colloc["z_in"], tss_ras=colloc["tss_ras"],
        targets=trainer.operator(z_c).detach(), z0_pred=z_c[:1], z0_true=z_c[:1].detach(),
    )
    mu_a = trainer.vault.parameters["muA"]
    base = trainer.loss.total(**kw).physics
    same = trainer.loss.total(**kw, params={"muA": mu_a}).physics
    doubled = trainer.loss.total(**kw, params={"muA": 2.0 * mu_a}).physics
    assert torch.equal(base, same)
    assert not torch.equal(base, doubled)


# ----------------------------------------------------------------------------
# Task 5.3 - anchors
# ----------------------------------------------------------------------------
def test_uniform_a0_anchor_reproduces_v10_losses_bit_for_bit(tmp_path):
    """A0 with uniform rel_std: same Z(0), weights all one, so the whole run is v1.0."""
    anchor = write_anchor(tmp_path / "A0.npz", RAW, sigma_tag="0p10", rel_std="uniform")
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))["cases"]
    cases = tuple(c for c in CASES if c[0] == "cl_pinn_7")
    got = short_run_losses(RAW, overrides={"anchor_file": str(anchor)}, cases=cases)
    assert got["cl_pinn_7"] == ref["cl_pinn_7"], _first_difference(got, {"cl_pinn_7": ref["cl_pinn_7"]})


def test_uniform_a0_anchor_gives_the_v10_ic_loss_at_every_stage(tmp_path):
    anchor = write_anchor(tmp_path / "A0.npz", RAW, sigma_tag="0p10", rel_std="uniform")
    base = _trainer(tmp_path)
    anchored = _trainer(tmp_path, anchor_file=str(anchor))
    assert np.all(anchored.ic_weights == 1.0)
    assert base.ic_weights is None
    probe = torch.rand(1, 5, 14, dtype=torch.float64) * torch.as_tensor(base.state_scale)
    for stage in base.schedule.stages:
        a, b = base._stage_tensors(stage), anchored._stage_tensors(stage)
        assert torch.equal(a["z0_true"], b["z0_true"]), stage.name
        assert torch.equal(base.loss.ic_loss(probe, a["z0_true"]),
                           anchored.loss.ic_loss(probe, b["z0_true"])), stage.name


def test_graded_anchor_changes_only_the_ic_weights(tmp_path):
    anchor = write_anchor(tmp_path / "Ag.npz", RAW, sigma_tag="0p10", rel_std="graded")
    trainer = _trainer(tmp_path, anchor_file=str(anchor))
    w = trainer.ic_weights
    assert w.shape == (5, 14) and np.all(w > 0.0)
    assert float(np.max(w)) > float(np.min(w))
    assert trainer.anchor.name == "A0_test"


# ----------------------------------------------------------------------------
# Task 5.3 - influent views
# ----------------------------------------------------------------------------
def _aggregate_preserving_swap(z_in: np.ndarray) -> np.ndarray:
    """Per-sample COD, TKN and ammonium unchanged, the split between components changed.

    Moving COD between S_S and X_S (neither carries TKN) and nitrogen between
    S_ND and X_ND (neither carries COD) leaves every daily flow-weighted total
    COD, TKN and S_NH as it was, up to rounding.
    """
    from src.asm1.vault_loader import vault

    v = vault()
    out = np.array(z_in, dtype=float)
    i_ss, i_xs, i_snd, i_xnd = v.indices(["S_S", "X_S", "S_ND", "X_ND"])
    phase = np.linspace(0.0, 6.0 * np.pi, len(out))
    moved = 0.4 * out[:, i_ss] * (0.5 + 0.5 * np.sin(phase))
    out[:, i_ss] -= moved
    out[:, i_xs] += moved
    out[:, [i_snd, i_xnd]] = out[:, [i_xnd, i_snd]]
    return out


def test_composite_view_depends_only_on_daily_aggregates(tmp_path, monkeypatch):
    from src.data.sensors import ObservationDataset

    raw_z = ObservationDataset.load(RAW / "obs_dry_sigma0p10.npz").z_in
    assert not np.allclose(_aggregate_preserving_swap(raw_z), raw_z), "the probe must differ"
    original = ObservationDataset.load.__func__
    base = _trainer(tmp_path, influent_mode="composite")

    def swapped(cls, path):
        ds = original(cls, path)
        return dataclasses.replace(ds, z_in=_aggregate_preserving_swap(ds.z_in))

    monkeypatch.setattr(ObservationDataset, "load", classmethod(swapped))
    other = _trainer(tmp_path, influent_mode="composite")
    np.testing.assert_allclose(other.z_in_scale, base.z_in_scale, rtol=1e-12, atol=0.0)
    for stage in base.schedule.stages:
        a, b = base._stage_tensors(stage), other._stage_tensors(stage)
        for key in ("t", "q_in", "z_in", "targets", "tss_ras", "z0_true"):
            torch.testing.assert_close(b[key], a[key], rtol=1e-12, atol=0.0, msg=key)


def test_exact_mode_does_not_touch_the_influent(tmp_path):
    from src.data.sensors import ObservationDataset
    from src.train.run import dataset_path

    trainer = _trainer(tmp_path)
    raw = ObservationDataset.load(dataset_path(trainer.cfg, "dry"))
    assert np.array_equal(trainer.data["dry"].z_in, raw.z_in)


def test_unknown_influent_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="influent_mode"):
        _trainer(tmp_path, influent_mode="weekly")


# ----------------------------------------------------------------------------
# Task 5.3 - target channels
# ----------------------------------------------------------------------------
def test_dropping_a_target_channel_removes_its_operator_column(tmp_path):
    from src.train.run import TARGET_CHANNELS

    default = _trainer(tmp_path)
    names = [c.name for c in TARGET_CHANNELS if c.name != "S_NO_tank2"]
    dropped = _trainer(tmp_path, target_channels=names)
    assert default.operator.names == tuple(c.name for c in TARGET_CHANNELS)
    assert dropped.operator.names == tuple(names)
    assert len(dropped.target_cols) == len(default.target_cols) - 1
    assert dropped.channel_index["S_NO_tank2"] not in dropped.target_cols
    z = torch.rand(4, 5, 14, dtype=torch.float64)
    assert dropped.operator(z).shape == (4, len(names))


@pytest.mark.parametrize(
    "names, message",
    [
        (["S_O_tank5", "TSS_ras"], "never be a target"),
        (["S_O_tank5", "S_O_tank9"], "Unknown target channel"),
        (["S_O_tank5", "S_NH_tank2"], "not in the dataset"),   # candidate absent from v1.0 data
        ([], "empty"),
        (["S_O_tank5", "S_O_tank5"], "duplicates"),
    ],
)
def test_invalid_target_channels_are_rejected(tmp_path, names, message):
    with pytest.raises(ValueError, match=message):
        _trainer(tmp_path, target_channels=names)


def test_adding_a_candidate_channel_adds_an_operator_column(tmp_path):
    from src.train.run import TARGET_CHANNELS

    data_dir = _require(RAW_K100)
    anchor = write_anchor(tmp_path / "A0.npz", data_dir, sigma_tag="0p10")
    names = [c.name for c in TARGET_CHANNELS] + ["S_NH_tank2"]
    trainer = _trainer(tmp_path, data_dir=data_dir, anchor_file=str(anchor), target_channels=names)
    assert trainer.operator.names[-1] == "S_NH_tank2"
    assert trainer.target_cols[-1] == trainer.channel_index["S_NH_tank2"]


# ----------------------------------------------------------------------------
# Task 5.3 - nominal plant and data provenance
# ----------------------------------------------------------------------------
def test_mismatch_data_still_builds_the_nominal_plant(tmp_path):
    from src.asm1.vault_loader import vault

    data_dir = _require(RAW_K100)
    anchor = write_anchor(tmp_path / "A0.npz", data_dir, sigma_tag="0p10")
    trainer = _trainer(tmp_path, data_dir=data_dir, anchor_file=str(anchor))
    assert dict(trainer.plant.vault.parameters) == dict(vault().parameters)
    assert trainer.provenance["truth_preset"] != "vault20"
    assert trainer.provenance["constant_from"] == "nominal"
    assert np.array_equal(trainer.z0, np.load(anchor)["z0_mean"])


def test_mismatch_data_without_an_anchor_is_refused(tmp_path):
    data_dir = _require(RAW_K100)
    with pytest.raises(ValueError, match="anchor_file"):
        _trainer(tmp_path, data_dir=data_dir)


def test_plant_guard_rejects_a_truth_plant(tmp_path, monkeypatch):
    from src.asm1.plant import Bsm1Plant
    from src.asm1.vault_loader import vault
    from src.train import run

    v = vault()
    altered = dataclasses.replace(v, parameters={**dict(v.parameters), "muA": 0.5})
    monkeypatch.setattr(run, "Bsm1Plant", lambda: Bsm1Plant(source=altered))
    with pytest.raises(RuntimeError, match="audited vault"):
        _trainer(tmp_path)

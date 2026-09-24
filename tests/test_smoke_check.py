"""The smoke gate checks code paths and protocol keys, never hidden states."""

from __future__ import annotations

import json

import numpy as np
import yaml

from scripts import smoke_check as sc


def _write_run(d, model="cl_pinn", **over):
    from src.asm1.vault_loader import vault

    d.mkdir(parents=True)
    summary = {k: None for k in sc.V10_SUMMARY_KEYS + sc.V11_SUMMARY_KEYS}
    summary.update({"model": model, "total_derivative": True, "ras_filter_window": 4, "variant": "",
                    "final_losses": {"total": 1.0, "physics": 0.5, "stage": "stage4"},
                    "target_channels": None, "trainable_kinetics": [], "learned_multipliers": None,
                    "vault_json_sha256": vault().json_sha256})
    summary.update(over)
    (d / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (d / "history.json").write_text(json.dumps([{"step": 0, "total": 1.0, "stage": "stage1"}]), encoding="utf-8")
    (d / "config.yaml").write_text(yaml.safe_dump({"kinetic_bound": 4.0,
                                                   "trainable_kinetics": summary["trainable_kinetics"]}),
                                   encoding="utf-8")
    np.savez_compressed(d / "predictions.npz", train=np.ones((3, 5, 14)), holdout=np.ones((2, 5, 14)))
    return d


def test_clean_run_passes(tmp_path):
    assert sc.check_run(_write_run(tmp_path / "a"), "cl_pinn") == []


def test_protocol_and_keys_enforced(tmp_path):
    problems = sc.check_run(_write_run(tmp_path / "b", total_derivative=False), "cl_pinn")
    assert any("total_derivative" in p for p in problems)
    d = _write_run(tmp_path / "c")
    summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    del summary["anchor_file"]
    (d / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    assert any("anchor_file" in p for p in sc.check_run(d, "cl_pinn"))


def test_variant_targets(tmp_path):
    drop = list(sc.DEFAULT_TARGETS[1:])
    d = _write_run(tmp_path / "d", variant="drop-%s" % sc.DEFAULT_TARGETS[0], target_channels=drop)
    assert sc.check_run(d, "cl_pinn", "drop-%s" % sc.DEFAULT_TARGETS[0]) == []
    assert sc.check_run(d, "cl_pinn", "") != []


def test_theta_multipliers_bounded(tmp_path):
    names = ["muA", "bA", "muH", "bH"]
    ok = _write_run(tmp_path / "e", model="cl_pinn_theta", trainable_kinetics=names,
                    learned_multipliers={n: 1.1 for n in names})
    assert sc.check_run(ok, "cl_pinn_theta") == []
    bad = _write_run(tmp_path / "f", model="cl_pinn_theta", trainable_kinetics=names,
                     learned_multipliers={**{n: 1.1 for n in names}, "muA": 5.0})
    assert any("multipliers" in p for p in sc.check_run(bad, "cl_pinn_theta"))

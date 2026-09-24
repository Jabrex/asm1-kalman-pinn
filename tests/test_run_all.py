"""run_all plan expansion: seeds never share an out_dir; variants never share a run id."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.run_all import expand, main, resolve_paths, run_id

REPO = Path(__file__).resolve().parents[1]


def _base() -> dict:
    return yaml.safe_load((REPO / "configs" / "base.yaml").read_text(encoding="utf-8"))


def test_different_seeds_never_share_an_out_dir():
    base = _base()
    dirs = {resolve_paths(base, seed=k)["out_dir"] for k in (0, 1, 2)}
    assert len(dirs) == 3
    assert base["out_dir"] not in dirs  # the v1.0 seed-0 directory is never reused
    assert resolve_paths(base, seed=1)["out_dir"] == "results/runs_seed1"
    assert resolve_paths(base, seed=1)["seed"] == 1
    # an explicit --out-dir wins, and --data-dir is passed through
    explicit = resolve_paths(base, seed=2, out_dir="results/v11/x", data_dir="results/raw_k100")
    assert explicit["out_dir"] == "results/v11/x" and explicit["data_dir"] == "results/raw_k100"
    # no overrides: the YAML is untouched
    assert resolve_paths(base) == base


def test_seed_reaches_every_run_config():
    configs = expand(resolve_paths(_base(), seed=2), ["cl_pinn"], [0.1], "quick")
    assert [c.seed for c in configs] == [2]
    assert configs[0].out_dir == "results/runs_seed2"


def test_variants_give_distinct_run_ids_and_apply_overrides():
    base = _base()
    base["variants"] = [
        {"suffix": "", "overrides": {}},
        {"suffix": "td", "overrides": {"total_derivative": True, "ras_filter_window": 4}},
        {"suffix": "td_rev", "overrides": {"total_derivative": True, "pinn": {"derivative_mode": "reverse"}}},
    ]
    configs = expand(base, ["cl_pinn", "pinn"], [0.0, 0.1], "full")
    ids = [c.run_id for c in configs]
    assert len(ids) == len(set(ids)) == 2 * 2 * 3
    by_id = {c.run_id: c for c in configs}
    assert by_id["cl_pinn_sigma0p10"].total_derivative is False
    td = by_id["cl_pinn_sigma0p10_td"]
    assert td.total_derivative is True and td.ras_filter_window == 4 and td.variant == "td"
    rev = by_id["pinn_sigma0p00_td_rev"]
    # nested block merged, not replaced: the width survives the override
    assert rev.pinn["derivative_mode"] == "reverse" and rev.pinn["hidden_width"] == 128
    assert run_id("cl_pinn", 0.1, "td") == "cl_pinn_sigma0p10_td"
    assert run_id("cl_pinn", 0.1) == "cl_pinn_sigma0p10"


def test_duplicate_or_illegal_variants_are_rejected():
    base = _base()
    base["variants"] = [{"suffix": "a", "overrides": {}}, {"suffix": "a", "overrides": {}}]
    with pytest.raises(ValueError):
        expand(base, ["cl_pinn"], [0.1], "quick")
    base["variants"] = [{"suffix": "b", "overrides": {"model": "pinn"}}]
    with pytest.raises(ValueError):
        expand(base, ["cl_pinn"], [0.1], "quick")


def test_list_prints_resolved_directories(capsys):
    assert main(["--config", str(REPO / "configs" / "base.yaml"), "--seed", "3",
                 "--data-dir", "results/raw_k050", "--models", "cl_pinn", "--noise", "0.1",
                 "--list"]) == 0
    out = capsys.readouterr().out
    assert "out_dir:  results/runs_seed3" in out
    assert "data_dir: results/raw_k050" in out
    assert "cl_pinn_sigma0p10" in out

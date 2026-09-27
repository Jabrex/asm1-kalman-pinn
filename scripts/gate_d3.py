"""Gate D3 (strategic plan section 9): where the realistic PINN cells sit (group G6).

Rule, fixed before any full-profile result and copied into PREREGISTRATION.md:
if the primary smoother with the As anchor at K1-Ic, sigma 0.10, has a higher
Track B NRMSE on window R0 (days 0-12) than anchor-aware persistence, the
realistic cells move from K1 to K.5. A tie keeps K1. Only this cell's hidden-state
score is read before the registration; both numbers are listed there as
pre-dating results.

    python -m scripts.gate_d3

Writes results/v11/gate_d3.json and, when the cells move, the derived configs
configs/regime/k050_ic_{a0,as,al1,as_lstm}.yaml. The k100_ic_* files are kept.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.v11_plan import (
    DATA_DIRS, GATE_D3, NUMERICS_BASELINE_ROOT, NUMERICS_OBSERVER_ROOT, REGIME_DIR, REPO, SIGMA,
    primary_smoother, sigma_tag,
)

CELL = "k100_ic_as"
REALISTIC_STEMS = ("ic_a0", "ic_as", "ic_al1", "ic_as_lstm")


def decide(smoother_r0: float, persistence_r0: float) -> str:
    if not (math.isfinite(smoother_r0) and math.isfinite(persistence_r0)):
        raise ValueError("gate D3 needs finite scores, got %r and %r" % (smoother_r0, persistence_r0))
    return "k050" if smoother_r0 > persistence_r0 else "k100"


def track_b_r0(runs_dir: Path, run_id: str, data_dir: Path) -> float:
    """Track B NRMSE of the 'train' evaluation set, which is window R0 (days 0-12)."""
    from src.eval.report import collect_runs

    rows = [r for r in collect_runs(Path(runs_dir), Path(data_dir))
            if r["run_id"] == run_id and r["eval_set"] == "train"]
    if len(rows) != 1:
        raise RuntimeError("expected one 'train' row for %s under %s, found %d" % (run_id, runs_dir, len(rows)))
    return float(rows[0]["track_b_nrmse"])


def derive_config(src: Path, dst: Path, k_from: str = "k100", k_to: str = "k050") -> dict:
    raw = yaml.safe_load(Path(src).read_text(encoding="utf-8"))
    raw["data_dir"] = DATA_DIRS[k_to]
    out_dir = Path(raw["out_dir"])
    if not out_dir.name.startswith(k_from + "_"):
        raise ValueError("%s: out_dir %s does not name a %s cell" % (src, out_dir, k_from))
    raw["out_dir"] = (out_dir.parent / out_dir.name.replace(k_from + "_", k_to + "_", 1)).as_posix()
    anchor = raw.get("anchor_file")
    if anchor:
        path = Path(anchor)
        if path.parent.name != k_from:
            raise ValueError("%s: anchor_file %s is not under a %s anchor directory" % (src, anchor, k_from))
        raw["anchor_file"] = (path.parent.parent / k_to / path.name).as_posix()
    header = "# Derived by scripts/gate_d3.py from %s: gate D3 moved the realistic cells to %s.\n" % (
        Path(src).name, k_to)
    Path(dst).write_text(header + yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return raw


def main(argv: list[str] | None = None) -> int:
    smoother = primary_smoother()
    tag = sigma_tag(SIGMA)
    data_dir = REPO / DATA_DIRS["k100"]
    est = track_b_r0(REPO / NUMERICS_OBSERVER_ROOT / CELL, "%s_sigma%s" % (smoother, tag), data_dir)
    per = track_b_r0(REPO / NUMERICS_BASELINE_ROOT / CELL, "persistence_sigma%s" % tag, data_dir)
    realistic = decide(est, per)
    derived = []
    if realistic == "k050":
        for stem in REALISTIC_STEMS:
            dst = REPO / REGIME_DIR / ("k050_%s.yaml" % stem)
            derive_config(REPO / REGIME_DIR / ("k100_%s.yaml" % stem), dst)
            derived.append(dst.relative_to(REPO).as_posix())
    out = {"rule": "realistic cells move to K.5 if the primary smoother at K1-Ic-As, sigma 0.10, has a higher "
                   "Track B NRMSE on R0 (days 0-12) than anchor-aware persistence; a tie keeps K1",
           "cell": CELL, "sigma": SIGMA, "smoother": smoother,
           "smoother_r0_track_b_nrmse": est, "persistence_r0_track_b_nrmse": per,
           "realistic_k": realistic, "derived_configs": derived}
    (REPO / GATE_D3).parent.mkdir(parents=True, exist_ok=True)
    (REPO / GATE_D3).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("gate D3: %s R0 Track B %.4f vs persistence %.4f -> realistic cells at %s" % (smoother, est, per, realistic))
    for path in derived:
        print("  derived %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Filter divergence in the numerics pass and the primary-smoother choice (group G6).

Reads only the filters' own diagnostics (summary.json keys finite, nis_mean,
n_measured and, if present, diverged), never the hidden states, so it may run
before the pre-registration. Rule, copied into PREREGISTRATION.md: a run has
diverged if any estimate is non-finite, the filter flags itself as diverged, or
the time-mean normalised innovation squared over days 0-12 exceeds 3 times the
number of measured channels. If more than 20 % of the 60 main-grid cells
diverge with the EKF, the IEKS replaces the EKS as the primary smoother,
provided its own rate is at most 20 %; otherwise the grid stops for review.

    python -m scripts.observer_numerics

Exit codes: 0 decision written; 4 run the IEKS numerics family first; 5 stop
(both above 20 %); 6 numerics pass incomplete.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.v11_plan import NUMERICS_JSON, OBSERVER_CONFIG_DIR, REPO, sigma_tag  # noqa: E402

NIS_FACTOR = 3.0
MAX_DIVERGENCE_RATE = 0.20
NUMERICS_KEYS = ("finite", "nis_mean", "n_measured")


def diverged(summary: dict) -> bool:
    return (not bool(summary["finite"])) or bool(summary.get("diverged", False)) \
        or float(summary["nis_mean"]) > NIS_FACTOR * float(summary["n_measured"])


def divergence(config_dir: Path, estimator: str) -> dict:
    cells = []
    for path in sorted(Path(config_dir).glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        run = REPO / cfg["out_dir"] / ("%s_sigma%s" % (estimator, sigma_tag(float(cfg["sigmas"][0]))))
        summary_path = run / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            missing = [k for k in NUMERICS_KEYS if k not in summary]
            if missing:
                raise KeyError("%s lacks %s (cell_inputs.numerics_summary contract)" % (summary_path, missing))
            cells.append({"cell": cfg["cell"], "status": "ok", "diverged": diverged(summary),
                          "nis_mean": float(summary["nis_mean"]), "n_measured": int(summary["n_measured"])})
        else:
            status = "error" if (run / "error.txt").exists() else "missing"
            cells.append({"cell": cfg["cell"], "status": status, "diverged": True})
    n = len(cells)
    n_div = sum(1 for c in cells if c["diverged"])
    return {"estimator": estimator, "n_cells": n, "n_diverged": n_div,
            "n_missing": sum(1 for c in cells if c["status"] == "missing"),
            "rate": n_div / n if n else float("nan"), "cells": cells}


def choose(ekf_rate: float, ieks_rate: float | None) -> str:
    if ekf_rate <= MAX_DIVERGENCE_RATE:
        return "eks"
    if ieks_rate is None:
        return "run_ieks"
    return "ieks" if ieks_rate <= MAX_DIVERGENCE_RATE else "stop"


def main(argv: list[str] | None = None) -> int:
    ekf = divergence(REPO / OBSERVER_CONFIG_DIR / "numerics", "ekf")
    if ekf["n_missing"]:
        print("numerics pass incomplete: %d of %d cells have no ekf run directory" % (ekf["n_missing"], ekf["n_cells"]))
        return 6
    ieks = None
    if ekf["rate"] > MAX_DIVERGENCE_RATE:
        candidate = divergence(REPO / OBSERVER_CONFIG_DIR / "numerics_ieks", "ieks")
        ieks = None if candidate["n_missing"] else candidate
    decision = choose(ekf["rate"], None if ieks is None else ieks["rate"])
    out = {"rule": "diverged = non-finite estimate, self-flagged divergence, or time-mean NIS (days 0-12) > "
                   "%.1f x measured channels; IEKS replaces EKS above %.0f %% of the 60 main-grid cells"
                   % (NIS_FACTOR, 100 * MAX_DIVERGENCE_RATE),
           "ekf": ekf, "ieks": ieks, "primary_smoother": decision}
    (REPO / NUMERICS_JSON).parent.mkdir(parents=True, exist_ok=True)
    (REPO / NUMERICS_JSON).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("EKF divergence %d/%d (%.1f %%)%s -> primary smoother: %s" % (
        ekf["n_diverged"], ekf["n_cells"], 100 * ekf["rate"],
        "" if ieks is None else "; IEKS %d/%d (%.1f %%)" % (ieks["n_diverged"], ieks["n_cells"], 100 * ieks["rate"]),
        decision))
    return {"eks": 0, "ieks": 0, "run_ieks": 4, "stop": 5}[decision]


if __name__ == "__main__":
    raise SystemExit(main())

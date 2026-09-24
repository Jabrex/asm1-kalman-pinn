"""Smoke-run check: every logged loss term is finite and the physics term fell.

"Fell" means the last logged physics loss is below both the first logged value
and the first value of the final curriculum stage (the stages change the data
window, so only the within-stage comparison is like for like).

Usage::

    python -m scripts.check_history results/v11/_smoke_g1/cl_pinn_sigma0p10 [more run dirs]

Exit code 0 when every run passes.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

TERMS = ("total", "data", "physics", "ic", "positivity", "balance")


def check(run_dir: Path) -> tuple[bool, dict]:
    history = json.loads((Path(run_dir) / "history.json").read_text(encoding="utf-8"))
    finite = all(math.isfinite(float(r[k])) for r in history for k in TERMS)
    phys = [float(r["physics"]) for r in history]
    last_stage = history[-1]["stage"]
    stage_start = next(float(r["physics"]) for r in history if r["stage"] == last_stage)
    fell = phys[-1] < phys[0] and phys[-1] < stage_start
    return finite and fell, {
        "records": len(history),
        "finite": finite,
        "physics_first": phys[0],
        "physics_final_stage_start": stage_start,
        "physics_last": phys[-1],
    }


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args:
        raise SystemExit("usage: python -m scripts.check_history <run_dir> [<run_dir> ...]")
    failed = 0
    for run_dir in args:
        ok, detail = check(Path(run_dir))
        print("%-60s %s  %s" % (run_dir, "PASS" if ok else "FAIL", detail))
        failed += not ok
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

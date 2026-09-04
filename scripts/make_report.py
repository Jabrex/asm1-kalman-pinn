"""RUNBOOK step 9 - score every finished run and write the benchmark report.

    python -m scripts.make_report

Outputs, all under ``results/``::

    benchmark.csv          one row per (model, noise, evaluation set)
    benchmark.md           Track A / Track B tables and dataset descriptors
    benchmark_detail.json  per-component metrics for every run
    figures/               loss curves and the noise-robustness curve

Safe to re-run at any time; it simply scores whatever runs exist.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.report import build  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="results/runs")
    parser.add_argument("--raw", default="results/raw")
    parser.add_argument("--out", default="results")
    args = parser.parse_args(argv)

    result = build(runs_dir=args.runs, raw_dir=args.raw, out_dir=args.out)
    print(json.dumps(result, indent=2))
    print("\nScored %d (run, evaluation set) rows." % result["n_rows"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

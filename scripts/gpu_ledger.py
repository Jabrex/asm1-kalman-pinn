"""GPU budget ledger for the v1.1 runs: results/v11/gpu_ledger.csv (group G6).

One row per training attempt, successful or not. ``eq`` is the charged
run-equivalent from scripts.v11_plan.nominal_eq; ``minutes`` is the run's own
train_seconds when it finished, otherwise its share of the process wall time.
Rows record what happened and are never refused, except a second row for a run
that already has a successful one. The budget guard lives in
scripts/run_core.py. Run from the repository root.

    python -m scripts.gpu_ledger total
    python -m scripts.gpu_ledger record --run-dir results/v11/_smoke_g1/cl_pinn_sigma0p10 --phase g1_smoke --eq 0.2
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.v11_plan import HARD_CAP, LEDGER

FIELDS = (
    "run_id", "phase", "run_dir", "attempt", "model", "profile", "sigma", "seed",
    "minutes", "eq", "status", "workers", "recorded_utc",
)
INFRA_MARKERS = (
    "CUDA error", "CUDA out of memory", "cuDNN error", "CUBLAS_STATUS",
    "KeyboardInterrupt", "MemoryError", "BrokenPipeError", "No space left on device",
)
STATUSES = ("ok", "failed_numeric", "error_infra", "error_other", "missing", "interrupted")


def read(path: Path = LEDGER) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def total_eq(rows: list[dict[str, str]]) -> float:
    return round(sum(float(r["eq"]) for r in rows), 6)


def by_phase(rows: list[dict[str, str]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for r in rows:
        out[r["phase"]] = round(out.get(r["phase"], 0.0) + float(r["eq"]), 6)
    return out


def _finite_losses(losses: dict) -> bool:
    values = [v for v in losses.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return bool(values) and all(math.isfinite(float(v)) for v in values)


def run_status(run_dir: Path) -> str:
    """``ok`` | ``failed_numeric`` | ``error_infra`` | ``error_other`` | ``missing``."""
    run_dir = Path(run_dir)
    summary = run_dir / "summary.json"
    if summary.exists() and (run_dir / "predictions.npz").exists():
        data = json.loads(summary.read_text(encoding="utf-8"))
        return "ok" if _finite_losses(data.get("final_losses") or {}) else "failed_numeric"
    error = run_dir / "error.txt"
    if error.exists():
        text = error.read_text(encoding="utf-8", errors="replace")
        return "error_infra" if any(m in text for m in INFRA_MARKERS) else "error_other"
    return "missing"


def append(row: dict, path: Path = LEDGER) -> dict:
    """Append one attempt; refuses only a second row for an already successful run."""
    path = Path(path)
    rows = read(path)
    if any(r["run_dir"] == row["run_dir"] and r["status"] == "ok" for r in rows):
        raise ValueError("%s already has a successful ledger row" % row["run_dir"])
    full = dict(row)
    full["attempt"] = 1 + sum(1 for r in rows if r["run_dir"] == row["run_dir"])
    full.setdefault("recorded_utc", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    missing = [f for f in FIELDS if f not in full]
    if missing:
        raise KeyError("ledger row lacks %s" % missing)
    if full["status"] not in STATUSES:
        raise ValueError("unknown status %r; expected one of %s" % (full["status"], STATUSES))
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({f: full[f] for f in FIELDS})
    return full


def record(run_dir: Path, phase: str, eq: float, minutes: float, workers: int, model: str,
           profile: str, sigma: float, seed: int, path: Path = LEDGER, status: str | None = None) -> dict:
    run_dir = Path(run_dir)
    return append(
        {
            "run_id": run_dir.name,
            "phase": phase,
            "run_dir": run_dir.as_posix(),
            "model": model,
            "profile": profile,
            "sigma": "%.2f" % float(sigma),
            "seed": int(seed),
            "minutes": "%.2f" % float(minutes),
            "eq": "%.2f" % float(eq),
            "status": status or run_status(run_dir),
            "workers": int(workers),
        },
        path,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    total = sub.add_parser("total", help="print charged eq per phase; exit 1 above the hard cap")
    total.add_argument("--ledger", default=str(LEDGER))
    rec = sub.add_parser("record", help="record one finished run directory")
    rec.add_argument("--run-dir", required=True)
    rec.add_argument("--phase", required=True)
    rec.add_argument("--eq", type=float, required=True)
    rec.add_argument("--ledger", default=str(LEDGER))
    args = parser.parse_args(argv)

    if args.command == "record":
        run_dir = Path(args.run_dir)
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        row = record(run_dir, args.phase, args.eq, float(summary["train_seconds"]) / 60.0, 1,
                     summary["model"], summary["profile"], float(summary["noise"]), int(summary["seed"]),
                     Path(args.ledger))
        print("recorded %s  phase=%s  %.2f eq  %s min  status=%s"
              % (row["run_dir"], row["phase"], float(row["eq"]), row["minutes"], row["status"]))
        return 0

    rows = read(Path(args.ledger))
    for phase, eq in sorted(by_phase(rows).items()):
        print("%-12s %6.2f eq" % (phase, eq))
    charged = total_eq(rows)
    minutes = sum(float(r["minutes"]) for r in rows)
    print("total        %6.2f eq  (%d attempts, %.1f process-minutes, hard cap %.2f eq)"
          % (charged, len(rows), minutes, HARD_CAP))
    return 0 if charged <= HARD_CAP + 1e-9 else 1


if __name__ == "__main__":
    raise SystemExit(main())

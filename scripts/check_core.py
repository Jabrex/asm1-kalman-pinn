"""Acceptance check of the v1.1 GPU core (E4-E7 and the E3 confirmation; group G6).

Every run the core queue expands to must hold checkpoint.pt, predictions.npz,
summary.json, history.json and config.yaml, or an error.txt with a ledger row
whose status is not 'ok'. Runs with non-finite final losses are reported as
failed_numeric. The ledger total must stay within the hard cap of 50
run-equivalents. Writes results/v11/core_status.json.

    python -m scripts.check_core
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import gpu_ledger  # noqa: E402
from scripts.v11_plan import HARD_CAP, REPO, V11, core_queue, realistic_k  # noqa: E402

REQUIRED = ("checkpoint.pt", "predictions.npz", "summary.json", "history.json", "config.yaml")
ACCEPTED = ("ok", "failed_numeric", "error_logged")


def classify(run_dir: Path, ledger_rows: list[dict]) -> str:
    run_dir = Path(run_dir)
    if all((run_dir / f).exists() for f in REQUIRED):
        return "ok" if gpu_ledger.run_status(run_dir) == "ok" else "failed_numeric"
    if (run_dir / "error.txt").exists():
        key = run_dir.as_posix()
        logged = any(r["run_dir"] == key and r["status"] != "ok" for r in ledger_rows)
        return "error_logged" if logged else "error_unlogged"
    return "missing"


def main(argv: list[str] | None = None) -> int:
    rows = gpu_ledger.read()
    runs = {}
    for job in core_queue(realistic_k()):
        for run_id, model in job.expected_run_ids():
            rel = Path(job.out_dir) / run_id
            status = classify(REPO / rel, [{**r, "run_dir": (REPO / r["run_dir"]).as_posix()} for r in rows])
            runs[rel.as_posix()] = {"phase": job.phase, "model": model, "seed": job.seed, "status": status}
    total = gpu_ledger.total_eq(rows)
    counts = {s: sum(1 for v in runs.values() if v["status"] == s)
              for s in ("ok", "failed_numeric", "error_logged", "error_unlogged", "missing")}
    passed = all(v["status"] in ACCEPTED for v in runs.values()) and total <= HARD_CAP + 1e-9
    report = {"passed": passed, "ledger_total_eq": total, "hard_cap": HARD_CAP, "counts": counts, "runs": runs}
    (REPO / V11 / "core_status.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for path, v in runs.items():
        if v["status"] != "ok":
            print("%-15s %s" % (v["status"], path))
    print("CORE %s: %s; ledger %.2f eq (hard cap %.2f)" % ("PASS" if passed else "FAIL", counts, total, HARD_CAP))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

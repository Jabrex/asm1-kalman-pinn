"""Throughput and determinism check before two concurrent GPU processes (group G6).

Compares the first smoke job trained next to the second one
(run_core --phase smoke-pair --workers 2) with the same job trained alone
(run_core --phase determinism). Two processes are allowed only if the
predictions agree to 1e-5 relative, so concurrency cannot change a result, and
the pair trains at least 1.25 times faster than the two runs would in sequence
(both jobs are quick cl_pinn runs of similar cost). Reads predictions only.

    python -m scripts.concurrency_check
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.v11_plan import CONCURRENCY, REPO, Job, determinism_job, realistic_k, smoke_queue  # noqa: E402

REL_TOL = 1e-5
MIN_GAIN = 1.25


def max_rel_diff(a: Path, b: Path) -> float:
    worst = 0.0
    with np.load(a) as pa, np.load(b) as pb:
        for key in ("train", "holdout"):
            x, y = np.asarray(pa[key], dtype=float), np.asarray(pb[key], dtype=float)
            if x.shape != y.shape:
                return float("inf")
            scale = np.maximum(np.max(np.abs(y), axis=0), 1e-12)
            worst = max(worst, float(np.max(np.abs(x - y) / scale)))
    return worst


def decide(rel_diff: float, solo_seconds: float, pair_seconds: tuple[float, float]) -> dict:
    gain = 2.0 * solo_seconds / max(pair_seconds)
    deterministic = rel_diff <= REL_TOL
    return {"workers": 2 if deterministic and gain >= MIN_GAIN else 1, "max_rel_diff": rel_diff,
            "deterministic": deterministic, "throughput_gain": gain, "rel_tol": REL_TOL, "min_gain": MIN_GAIN,
            "solo_seconds": solo_seconds, "pair_seconds": list(pair_seconds)}


def _run(job: Job) -> tuple[Path, dict]:
    run_id, _ = job.expected_run_ids()[0]
    d = REPO / job.out_dir / run_id
    return d, json.loads((d / "summary.json").read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    realistic = realistic_k()
    first, second = smoke_queue(realistic)[:2]
    pair_a, sa = _run(first)
    _, sb = _run(second)
    solo, ss = _run(determinism_job(realistic))
    out = decide(max_rel_diff(pair_a / "predictions.npz", solo / "predictions.npz"),
                 float(ss["train_seconds"]), (float(sa["train_seconds"]), float(sb["train_seconds"])))
    (REPO / CONCURRENCY).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("max relative prediction difference %.2e (tol %.0e), throughput gain %.2f (min %.2f) -> workers %d"
          % (out["max_rel_diff"], REL_TOL, out["throughput_gain"], MIN_GAIN, out["workers"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

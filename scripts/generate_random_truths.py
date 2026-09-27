"""Random kinetic-mismatch truth plants for the CPU observer ensemble (E2).

Candidate log-multiplier vectors ``m ~ N(0, sigma_log^2)`` over the vault's 15
kinetic parameters (``truth_plants.kinetic_names()`` order) come from one fixed
matrix ``np.random.default_rng(--seed).normal(0, sigma_log, (MAX_CANDIDATES, 15))``,
so candidate ``c`` is the same vector whatever ``--n`` is.

1. Screening (sequential, deterministic): candidates are taken in index order;
   each one is warmed up for 200 days and rejected when its tank-5 X_B_A falls
   below ``WASHOUT_RATIO`` of the nominal steady state (the same rule as the
   anchor ensemble). Screening stops at ``--n`` accepted candidates.
2. Generation (parallel subprocesses): accepted plant ``i`` runs
   ``scripts.generate_data --truth-preset perturbed`` into ``<out>/<i:02d>/``
   with the dry scenario, sigma 0.10 and the eight standard channels.

``<out>/draws.json`` records every screened candidate (accepted or rejected,
with the X_B_A ratio), the accepted-index mapping, each run's exit code and
closure record. A plant whose closure fails even on the refined grid is reported
and the script exits 1; it is never dropped silently.

Usage: python -m scripts.generate_random_truths --n 50 --sigma-log 0.3 --seed 0 --workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.asm1.plant import Bsm1Plant
from src.asm1.truth_plants import kinetic_names, perturbed_vault
from src.data.simulate import warm_up

WASHOUT_RATIO = 0.05
MAX_CANDIDATES = 200


def candidate_matrix(sigma_log: float, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, sigma_log, size=(MAX_CANDIDATES, len(kinetic_names())))


def as_multipliers(row: np.ndarray) -> dict[str, float]:
    return {name: float(x) for name, x in zip(kinetic_names(), row)}


def autotroph_ratio(multipliers: dict[str, float], nominal_x_ba5: float) -> float:
    plant, y = warm_up(Bsm1Plant(source=perturbed_vault(multipliers)))
    reactor = plant.unpack(y)[0]
    return float(reactor[-1, plant.vault.index("X_B_A")] / nominal_x_ba5)


def run_one(out_dir: Path, multipliers: dict[str, float], noise_seed: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "scripts.generate_data",
        "--out", str(out_dir), "--seed", str(noise_seed),
        "--truth-preset", "perturbed", "--log-multipliers", json.dumps(multipliers),
        "--scenarios", "dry", "--sigmas", "0.10",
    ]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    (out_dir / "generate.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    return proc.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--sigma-log", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0, help="seed of the candidate matrix")
    parser.add_argument("--noise-seed", type=int, default=0, help="generate_data --seed")
    parser.add_argument("--out", default="results/raw_rand")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plant, y_nominal = warm_up(Bsm1Plant())
    nominal_x_ba5 = float(plant.unpack(y_nominal)[0][-1, plant.vault.index("X_B_A")])

    matrix = candidate_matrix(args.sigma_log, args.seed)
    screened: list[dict[str, object]] = []
    accepted: list[int] = []
    for c in range(MAX_CANDIDATES):
        if len(accepted) == args.n:
            break
        m = as_multipliers(matrix[c])
        ratio = autotroph_ratio(m, nominal_x_ba5)
        ok = ratio >= WASHOUT_RATIO
        screened.append({"candidate": c, "x_ba_tank5_ratio": ratio, "accepted": ok})
        print("candidate %3d  X_B_A ratio %.3g  %s" % (c, ratio, "accepted" if ok else "washout"))
        if ok:
            accepted.append(c)
    if len(accepted) < args.n:
        print("FAIL - only %d of %d plants accepted in %d candidates"
              % (len(accepted), args.n, MAX_CANDIDATES))
        return 1

    dirs = [out / ("%02d" % i) for i in range(args.n)]
    jobs = [(d, as_multipliers(matrix[c])) for d, c in zip(dirs, accepted)]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        codes = list(pool.map(lambda job: run_one(job[0], job[1], args.noise_seed), jobs))

    plants = []
    for i, (d, c, code) in enumerate(zip(dirs, accepted, codes)):
        entry: dict[str, object] = {"index": i, "dir": d.name, "candidate": c,
                                    "log_multipliers": as_multipliers(matrix[c]),
                                    "exit_code": code}
        manifest_path = d / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            entry["parameters"] = manifest["parameters"]
            entry["closure"] = manifest["scenarios"]["dry"]["closure"]
        plants.append(entry)
    payload = {
        "n": args.n, "sigma_log": args.sigma_log, "seed": args.seed,
        "noise_seed": args.noise_seed, "max_candidates": MAX_CANDIDATES,
        "parameter_order": list(kinetic_names()),
        "washout_rule": "tank-5 X_B_A below %.2f of the nominal steady state" % WASHOUT_RATIO,
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "screened": screened, "plants": plants,
    }
    (out / "draws.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    failed = [p["dir"] for p in plants if p["exit_code"] != 0]
    rejected = sum(1 for s in screened if not s["accepted"])
    print("plants %d, screened %d, washout rejections %d, failed runs %d %s"
          % (args.n, len(screened), rejected, len(failed), failed))
    print("Wrote %s" % (out / "draws.json"))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

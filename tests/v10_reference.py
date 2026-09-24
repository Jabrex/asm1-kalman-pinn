"""Loss parts of a few short CPU float64 training runs: the v1.0 reproduction check.

The same function produces the stored reference (run against an unpacked
v1.0.0 archive) and the value the test compares with it (run against the
working tree), so the two can only differ through the code under test.

    python tests/v10_reference.py --repo <unpacked v1.0.0> --data-dir results/raw \
        --label v1.0.0 --out tests/data/v10_three_step_losses.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

#: (case name, model, steps). Three steps of the hierarchical schedule leave its
#: first stage empty (15% of 3 rounds to 0), so a seven-step case covers all
#: four stages, including the constant-load stage.
CASES: tuple[tuple[str, str, int], ...] = (
    ("cl_pinn_3", "cl_pinn", 3),
    ("pinn_3", "pinn", 3),
    ("lstm_3", "lstm", 3),
    ("cl_pinn_7", "cl_pinn", 7),
)
#: History keys that exist in v1.0; later keys (kinetic_prior, mult_*) are ignored.
KEYS: tuple[str, ...] = (
    "step", "stage", "lr", "total", "data", "physics", "ic", "positivity", "balance",
)
NOISE = 0.10


def short_run_losses(
    data_dir: Path, overrides: dict[str, Any] | None = None, cases=CASES
) -> dict[str, list[dict[str, Any]]]:
    """Train every case with ``log_every=1``; return the per-step loss records.

    ``overrides`` are extra RunConfig fields (v1.1 only); the reference itself
    is produced without any.
    """
    import torch

    from src.train.run import RunConfig, Trainer

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    out: dict[str, list[dict[str, Any]]] = {}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            for name, model, steps in cases:
                cfg = RunConfig(
                    run_id=name, model=model, noise=NOISE, seed=0, profile="quick",
                    steps_quick=steps, log_every=1, device="cpu", dtype="float64",
                    data_dir=str(data_dir), out_dir=tmp, **(overrides or {}),
                )
                trainer = Trainer(cfg)
                trainer.train()
                out[name] = [{k: record[k] for k in KEYS} for record in trainer.history]
    finally:
        torch.set_num_threads(threads)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="root of the code tree to run")
    parser.add_argument("--data-dir", required=True, help="directory with obs_*_sigma0p10.npz")
    parser.add_argument("--label", required=True, help="provenance label, e.g. v1.0.0")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))
    import src

    origin = Path(src.__file__).resolve().parent.parent
    if origin != repo:
        raise SystemExit("imported src from %s, expected %s" % (origin, repo))
    payload = {
        "label": args.label,
        "noise": NOISE,
        "cases": short_run_losses(Path(args.data_dir).resolve()),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("wrote %s (%d cases, label %s)" % (out, len(payload["cases"]), args.label))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

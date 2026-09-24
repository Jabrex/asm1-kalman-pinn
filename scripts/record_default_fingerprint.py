"""Record a bitwise fingerprint of short default-config training runs.

v1.1 adds switches (total derivative, RAS filter, run variants) whose defaults
must leave the v1.0 training path untouched bit for bit. This script trains
``cl_pinn`` and ``pinn`` for six CPU float64 steps with the default RunConfig
and writes the loss history plus a SHA-256 of the final weights.
``tests/test_total_derivative.py`` replays the same runs and compares.

Record it from unmodified v1.0 code, never from a working tree with v1.1 edits::

    git worktree add ..\\asm1_v100 v1.0.0
    python scripts/record_default_fingerprint.py --repo ..\\asm1_v100 --data-dir results/raw \\
        --out tests/data/v10_default_fingerprint.json
    git worktree remove ..\\asm1_v100

The hashes depend on the installed torch/numpy build (recorded in the file).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import tempfile
from pathlib import Path

KEYS = ("total", "data", "physics", "ic", "positivity", "balance")


def fingerprint(model: str, data_dir: Path) -> dict:
    from src.train.run import RunConfig, Trainer

    cfg = RunConfig(
        run_id="_fp_%s" % model, model=model, noise=0.05, seed=0, profile="quick",
        steps_quick=6, log_every=1, device="cpu", dtype="float64",
        data_dir=str(data_dir), out_dir=tempfile.mkdtemp(),
    )
    trainer = Trainer(cfg)
    trainer.train()
    digest = hashlib.sha256()
    for name, p in sorted(trainer.model.state_dict().items()):
        digest.update(name.encode())
        digest.update(p.detach().cpu().numpy().tobytes())
    return {"history": [[r[k] for k in KEYS] for r in trainer.history], "state_sha256": digest.hexdigest()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]),
                        help="source tree whose src/ is imported (default: this checkout)")
    parser.add_argument("--data-dir", default="results/raw")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    sys.path.insert(0, str(Path(args.repo).resolve()))
    import numpy
    import torch

    out = {m: fingerprint(m, Path(args.data_dir).resolve()) for m in ("cl_pinn", "pinn")}
    out["_meta"] = {"torch": torch.__version__, "numpy": numpy.__version__,
                    "python": platform.python_version(), "source_repo": str(Path(args.repo).resolve())}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({m: out[m]["state_sha256"] for m in ("cl_pinn", "pinn")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

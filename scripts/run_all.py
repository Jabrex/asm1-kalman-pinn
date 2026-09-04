"""RUNBOOK steps 7 and 8 - execute the benchmark sweep.

Expands ``configs/base.yaml`` over the model list and the noise sweep, runs each
combination, and writes the resolved config next to each run for provenance.

    python -m scripts.run_all --profile quick     # step 7, pipeline check
    python -m scripts.run_all --profile full      # step 8, reported benchmark

Useful flags::

    --list                     print the run plan and exit, nothing is trained
    --models cl_pinn pinn      restrict the model list
    --noise 0.0 0.10           restrict the noise sweep
    --resume                   skip runs that already have a summary.json

Runs are independent; interrupting is safe and ``--resume`` picks up where the
sweep stopped.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train.run import MODEL_SPECS, RunConfig, Trainer  # noqa: E402

BASE_CONFIG = Path("configs/base.yaml")


def sigma_tag(sigma: float) -> str:
    return ("%.2f" % sigma).replace(".", "p")


def run_id(model: str, sigma: float) -> str:
    return "%s_sigma%s" % (model, sigma_tag(sigma))


def expand(base: dict[str, Any], models: list[str], noises: list[float], profile: str) -> list[RunConfig]:
    shared = {
        k: v for k, v in base.items()
        if k not in {"models", "noise_levels"}
    }
    shared["profile"] = profile
    shared["holdout_days"] = tuple(shared.get("holdout_days", (12.0, 14.0)))
    configs = []
    for model in models:
        if model not in MODEL_SPECS:
            raise ValueError("Unknown model %r; expected one of %s" % (model, sorted(MODEL_SPECS)))
        for sigma in noises:
            configs.append(
                RunConfig(run_id=run_id(model, sigma), model=model, noise=float(sigma), **shared)
            )
    return configs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(BASE_CONFIG))
    parser.add_argument("--profile", default=None, choices=["quick", "full"])
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--noise", nargs="*", type=float, default=None)
    parser.add_argument("--list", action="store_true", help="print the plan and exit")
    parser.add_argument("--resume", action="store_true", help="skip completed runs")
    args = parser.parse_args(argv)

    base = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    models = args.models or base["models"]
    noises = args.noise if args.noise is not None else base["noise_levels"]
    profile = args.profile or base.get("profile", "quick")

    configs = expand(base, models, noises, profile)
    print("Benchmark sweep: %d runs, profile=%s" % (len(configs), profile))
    for cfg in configs:
        print("  %-24s model=%-8s curriculum=%-13s sigma=%.2f steps=%d"
              % (cfg.run_id, cfg.model, cfg.curriculum, cfg.noise, cfg.steps))
    if args.list:
        return 0
    print()

    results: list[dict[str, Any]] = []
    failures: list[str] = []
    sweep_started = time.perf_counter()

    for index, cfg in enumerate(configs, start=1):
        out_dir = Path(cfg.out_dir) / cfg.run_id
        if args.resume and (out_dir / "summary.json").exists():
            print("[%2d/%d] %-24s SKIP (already complete)" % (index, len(configs), cfg.run_id))
            continue

        print("[%2d/%d] %-24s ..." % (index, len(configs), cfg.run_id), flush=True)
        started = time.perf_counter()
        try:
            summary = Trainer(cfg).train()
        except Exception:  # noqa: BLE001 - one failed run must not stop the sweep
            failures.append(cfg.run_id)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            print("        FAILED - traceback written to %s" % (out_dir / "error.txt"))
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.yaml").write_text(
            yaml.safe_dump({**cfg.__dict__, "holdout_days": list(cfg.holdout_days)},
                           sort_keys=False),
            encoding="utf-8",
        )
        elapsed = time.perf_counter() - started
        print("        done in %.1f s  |  final total loss %.4e"
              % (elapsed, summary.get("final_losses", {}).get("total", float("nan"))))
        results.append(summary)

    total = time.perf_counter() - sweep_started
    print("\nSweep finished in %.1f min: %d succeeded, %d failed"
          % (total / 60.0, len(results), len(failures)))
    if failures:
        print("Failed runs: %s" % ", ".join(failures))

    index_path = Path(configs[0].out_dir) / "sweep_index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(
        json.dumps(
            {
                "profile": profile,
                "succeeded": [r["run_id"] for r in results],
                "failed": failures,
                "sweep_seconds": total,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Wrote %s" % index_path)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

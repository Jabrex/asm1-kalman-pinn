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
    --seed 1                   override the YAML seed; without --out-dir the
                               results go to <out_dir>_seed1 so seeds never
                               overwrite each other
    --out-dir / --data-dir     override the YAML out_dir / data_dir

An optional YAML key ``variants`` expands every (model, sigma) pair once per
entry; each entry is ``{suffix: str, overrides: dict}``. The suffix is appended
to the run id (``cl_pinn_sigma0p10_td``) and stored in ``RunConfig.variant``;
an empty suffix keeps the plain run id. Nested dicts in ``overrides`` (for
example ``pinn:``) are merged into the base block rather than replacing it.

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

from src.train.run import MODEL_SPECS, RunConfig, Trainer

BASE_CONFIG = Path("configs/base.yaml")


def sigma_tag(sigma: float) -> str:
    return ("%.2f" % sigma).replace(".", "p")


def run_id(model: str, sigma: float, variant: str = "") -> str:
    base = "%s_sigma%s" % (model, sigma_tag(sigma))
    return base + ("_" + variant if variant else "")


def resolve_paths(
    base: dict[str, Any],
    seed: int | None = None,
    out_dir: str | None = None,
    data_dir: str | None = None,
) -> dict[str, Any]:
    """Apply the --seed / --out-dir / --data-dir overrides to a loaded YAML dict.

    ``--seed`` without ``--out-dir`` appends ``_seed<k>`` to the YAML out_dir, so
    two seeds of the same config can never write into the same directory.
    """
    resolved = dict(base)
    if seed is not None:
        resolved["seed"] = int(seed)
        if out_dir is None:
            resolved["out_dir"] = "%s_seed%d" % (base.get("out_dir", "results/runs"), int(seed))
    if out_dir is not None:
        resolved["out_dir"] = str(out_dir)
    if data_dir is not None:
        resolved["data_dir"] = str(data_dir)
    return resolved


def _merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge: nested blocks such as ``pinn:`` are updated, not replaced."""
    out = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def expand(base: dict[str, Any], models: list[str], noises: list[float], profile: str) -> list[RunConfig]:
    shared = {
        k: v for k, v in base.items()
        if k not in {"models", "noise_levels", "variants"}
    }
    shared["profile"] = profile
    shared["holdout_days"] = tuple(shared.get("holdout_days", (12.0, 14.0)))
    variants = base.get("variants") or [{"suffix": "", "overrides": {}}]
    suffixes = [str(v.get("suffix", "")) for v in variants]
    if len(set(suffixes)) != len(suffixes):
        raise ValueError("variant suffixes must be unique, got %s" % (suffixes,))
    configs = []
    for model in models:
        if model not in MODEL_SPECS:
            raise ValueError("Unknown model %r; expected one of %s" % (model, sorted(MODEL_SPECS)))
        for sigma in noises:
            for variant in variants:
                suffix = str(variant.get("suffix", ""))
                overrides = dict(variant.get("overrides") or {})
                forbidden = {"run_id", "model", "noise", "variant"} & set(overrides)
                if forbidden:
                    raise ValueError("variant %r may not override %s" % (suffix, sorted(forbidden)))
                merged = _merge(shared, overrides)
                merged["holdout_days"] = tuple(merged["holdout_days"])
                configs.append(
                    RunConfig(
                        run_id=run_id(model, sigma, suffix), model=model, noise=float(sigma),
                        variant=suffix, **merged,
                    )
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
    parser.add_argument("--seed", type=int, default=None, help="override the YAML seed")
    parser.add_argument("--out-dir", default=None, help="override the YAML out_dir")
    parser.add_argument("--data-dir", default=None, help="override the YAML data_dir")
    args = parser.parse_args(argv)

    base = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base = resolve_paths(base, seed=args.seed, out_dir=args.out_dir, data_dir=args.data_dir)
    models = args.models or base["models"]
    noises = args.noise if args.noise is not None else base["noise_levels"]
    profile = args.profile or base.get("profile", "quick")

    configs = expand(base, models, noises, profile)
    print("Benchmark sweep: %d runs, profile=%s, seed=%s" % (len(configs), profile, base.get("seed", 0)))
    print("  out_dir:  %s" % base.get("out_dir", "results/runs"))
    print("  data_dir: %s" % base.get("data_dir", "results/raw"))
    for cfg in configs:
        print("  %-32s model=%-8s curriculum=%-13s sigma=%.2f steps=%d"
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
        except Exception:
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

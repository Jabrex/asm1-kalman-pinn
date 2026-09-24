"""Pass/fail gate for the v1.1 smoke runs (group G6).

Pass requires, for every run of the smoke queue:
  - summary.json, predictions.npz, history.json and config.yaml present;
  - every v1.0 summary key and every v1.1 key (groups G1 and G5) present;
  - the v1.1 protocol: total_derivative true, ras_filter_window 4, the audited vault hash;
  - the target channels the variant implies (drop-<c>, add-<c> or the default seven);
  - every numeric history entry and final loss finite, a positive physics term for the PINNs;
  - 'train' and 'holdout' predictions finite with shape (n, 5, 14);
  - for cl_pinn_theta, one finite multiplier per trainable name inside [1/bound, bound];
and tests/test_leakage.py green with ASM1_STRICT_TESTS=1. Hidden states are not scored.

    python -m scripts.smoke_check
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.v11_plan import ARCH, REPO, SIGMA, SMOKE_ROOT, realistic_k, sigma_tag, smoke_queue  # noqa: E402
from src.observers.cell_inputs import DEFAULT_TARGETS  # noqa: E402

REPORT = Path(SMOKE_ROOT) / "smoke_report.json"
V10_SUMMARY_KEYS = (
    "run_id", "model", "arch", "curriculum", "noise", "seed", "ic_measured_only", "profile",
    "train_end_day", "holdout_days", "steps", "schedule", "device", "dtype", "n_parameters",
    "train_seconds", "peak_gpu_bytes", "final_losses", "unobserved_components", "vault_json_sha256",
)
V11_SUMMARY_KEYS = (
    "data_dir", "truth_preset", "alpha", "constant_from", "anchor_file", "anchor", "influent_mode",
    "total_derivative", "ras_filter_window", "target_channels", "trainable_kinetics",
    "learned_multipliers", "variant",
)


def _numeric(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def expected_targets(variant: str) -> list[str]:
    if variant.startswith("drop-"):
        return [t for t in DEFAULT_TARGETS if t != variant[len("drop-"):]]
    if variant.startswith("add-"):
        return [*DEFAULT_TARGETS, variant[len("add-"):]]
    return list(DEFAULT_TARGETS)


def check_run(run_dir: Path, model: str, variant: str = "") -> list[str]:
    from src.asm1.vault_loader import vault

    run_dir = Path(run_dir)
    problems = ["missing %s" % n for n in ("summary.json", "predictions.npz", "history.json", "config.yaml")
                if not (run_dir / n).exists()]
    if problems:
        return problems
    s = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    missing = [k for k in (*V10_SUMMARY_KEYS, *V11_SUMMARY_KEYS) if k not in s]
    if missing:
        problems.append("summary keys missing: %s" % ", ".join(missing))
    if s.get("total_derivative") is not True:
        problems.append("total_derivative is not true")
    if s.get("ras_filter_window") != 4:
        problems.append("ras_filter_window is %r, not 4" % s.get("ras_filter_window"))
    if s.get("vault_json_sha256") != vault().json_sha256:
        problems.append("vault hash differs from the audited vault")
    if (s.get("variant") or "") != variant:
        problems.append("variant %r, expected %r" % (s.get("variant"), variant))
    targets = list(s.get("target_channels") or DEFAULT_TARGETS)
    if targets != expected_targets(variant):
        problems.append("target_channels %s, expected %s" % (targets, expected_targets(variant)))
    losses = s.get("final_losses") or {}
    if not all(math.isfinite(float(v)) for v in losses.values() if _numeric(v)):
        problems.append("non-finite final loss")
    if ARCH[model] == "pinn" and not (_numeric(losses.get("physics")) and float(losses["physics"]) > 0.0):
        problems.append("PINN physics term missing or zero")
    history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    if not history:
        problems.append("empty history")
    bad_steps = [r.get("step") for r in history if any(_numeric(v) and not math.isfinite(v) for v in r.values())]
    if bad_steps:
        problems.append("non-finite history at steps %s" % bad_steps[:5])
    with np.load(run_dir / "predictions.npz") as preds:
        for key in ("train", "holdout"):
            if key not in preds.files:
                problems.append("predictions lack %r" % key)
                continue
            arr = preds[key]
            if arr.ndim != 3 or arr.shape[1:] != (5, 14) or not np.isfinite(arr).all():
                problems.append("predictions %r have shape %s or non-finite values" % (key, arr.shape))
    if model == "cl_pinn_theta":
        config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        bound = float(config["kinetic_bound"])
        names = list(config["trainable_kinetics"])
        mult = s.get("learned_multipliers")
        if not isinstance(mult, dict) or sorted(mult) != sorted(names):
            problems.append("learned multipliers %r do not match trainable_kinetics %r" % (mult, names))
        elif not all(math.isfinite(float(v)) and 1.0 / bound - 1e-9 <= float(v) <= bound + 1e-9 for v in mult.values()):
            problems.append("learned multipliers outside [1/%.2f, %.2f] or non-finite: %s" % (bound, bound, mult))
    return problems


def run_leakage_tests() -> int:
    env = {**os.environ, "ASM1_STRICT_TESTS": "1"}
    return subprocess.run([sys.executable, "-m", "pytest", "tests/test_leakage.py", "-q"], cwd=REPO, env=env).returncode


def main(argv: list[str] | None = None) -> int:
    tag = "_sigma%s" % sigma_tag(SIGMA)
    runs: dict[str, list[str]] = {}
    for job in smoke_queue(realistic_k()):
        for run_id, model in job.expected_run_ids():
            variant = run_id.split(tag, 1)[1].lstrip("_")
            run_dir = Path(job.out_dir) / run_id
            runs[run_dir.as_posix()] = check_run(REPO / run_dir, model, variant)
    leakage = run_leakage_tests()
    passed = leakage == 0 and all(not p for p in runs.values())
    (REPO / REPORT).parent.mkdir(parents=True, exist_ok=True)
    (REPO / REPORT).write_text(json.dumps({"passed": passed, "n_runs": len(runs), "leakage_exit": leakage,
                                           "runs": runs}, indent=2), encoding="utf-8")
    for path, problems in runs.items():
        print("%-4s %s%s" % ("ok" if not problems else "FAIL", path, "" if not problems else "  " + "; ".join(problems)))
    print("leakage tests (strict): %s" % ("green" if leakage == 0 else "exit %d" % leakage))
    print("SMOKE %s (%d runs) -> %s" % ("PASS" if passed else "FAIL", len(runs), REPORT.as_posix()))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Physics residual of every v1.1 PINN checkpoint inside the window (R0) and beyond it (F).

    python -m scripts.residual_diagnostic --root results/v11/pinn --out results/v11/residual_diagnostic.json

For each checkpoint the Trainer is rebuilt from the saved config on CPU in
float64 and the trained weights are loaded. The residual is the one used in
training: total time derivative with local-linear influent inputs, the
practitioner influent view the Trainer applies, the trailing-filtered RAS
signal and, for PINN-theta runs, the learned kinetic multipliers from
summary.json. It is evaluated on n uniform random times in days 0-12 (R0) and
12-14 (F). Days 12-14 use the RAS measurements and influent of that window,
which the network never saw; the diagnostic asks whether the network output
still satisfies the plant equations there, not whether it matches the truth.
Reference: the v1.0 probe gave about 0.57 in-window against about 8-9 on the
holdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.regime_map import parse_cell
from src.train import curriculum as cl
from src.train.curriculum import trailing_average
from src.train.run import RunConfig, Trainer

WINDOWS = {"R0": (0.0, 12.0), "F": (12.0, 14.0)}
V10_REFERENCE = {"R0": 0.57, "F": "8-9"}


def load_trainer(run_dir: Path) -> tuple[Trainer, dict[str, Any]]:
    """Trainer rebuilt from checkpoint['config'] on CPU in float64, trained weights loaded."""
    ckpt = torch.load(Path(run_dir) / "checkpoint.pt", map_location="cpu", weights_only=False)
    raw = dict(ckpt["config"])
    raw.update(device="cpu", dtype="float64")
    raw["holdout_days"] = tuple(raw["holdout_days"])
    trainer = Trainer(RunConfig(**raw))
    trainer.model.load_state_dict(ckpt["state_dict"])
    trainer.model.eval()
    summary = json.loads((Path(run_dir) / "summary.json").read_text(encoding="utf-8"))
    return trainer, summary


def input_series(trainer: Trainer) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(t, q_in, z_in, tss_ras) over days 0-14, processed as in the final training stage."""
    ds = trainer.data["dry"]
    obs = ds.obs.copy()
    window = int(getattr(trainer.cfg, "ras_filter_window", 1))
    if window > 1:
        obs[:, trainer.ras_col] = trailing_average(obs[:, trainer.ras_col], window)
    obs = cl.smooth_observations(obs, trainer.schedule.stages[-1].smoothing_window)
    return ds.t, ds.q_in, ds.z_in, obs[:, trainer.ras_col]


def residual_mse(trainer: Trainer, summary: dict[str, Any], lo: float, hi: float, n: int,
                 rng: np.random.Generator) -> float:
    t_grid, q_grid, z_grid, ras_grid = input_series(trainer)
    t = np.sort(rng.uniform(lo, hi, n))
    idx = np.clip(np.searchsorted(t_grid, t, side="right") - 1, 0, len(t_grid) - 2)
    dt = t_grid[idx + 1] - t_grid[idx]
    w = (t - t_grid[idx]) / dt
    q = q_grid[idx] * (1 - w) + q_grid[idx + 1] * w
    z = z_grid[idx] * (1 - w)[:, None] + z_grid[idx + 1] * w[:, None]
    ras = ras_grid[idx] * (1 - w) + ras_grid[idx + 1] * w
    dq = (q_grid[idx + 1] - q_grid[idx]) / dt
    dz = (z_grid[idx + 1] - z_grid[idx]) / dt[:, None]

    def T(a: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float64)

    tt = T(t).view(-1, 1).requires_grad_(True)
    z_c, dz_c = trainer.model.state_and_derivative(
        tt, T(q).view(-1, 1), T(z), mode=None, dq_dt=T(dq).view(-1, 1), dz_dt=T(dz)
    )
    multipliers = summary.get("learned_multipliers") or {}
    params = {k: float(trainer.vault.parameters[k]) * float(m) for k, m in multipliers.items()} or None
    kwargs = {"params": params} if params is not None else {}
    res = trainer.loss.physics_residual(z_c, dz_c, T(q).view(-1, 1), T(z), T(ras).view(-1, 1), **kwargs)
    return float(torch.mean(res.detach() ** 2))


def run_dirs(root: Path) -> list[Path]:
    return sorted(p.parent for p in Path(root).glob("*/*/checkpoint.pt") if not p.parent.name.startswith("_"))


def diagnose(root: Path, n: int, seed: int) -> dict[str, Any]:
    runs, skipped = [], []
    for run_dir in run_dirs(root):
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        if summary.get("arch") != "pinn":
            skipped.append(str(run_dir).replace("\\", "/"))
            continue
        trainer, summary = load_trainer(run_dir)
        rng = np.random.default_rng(seed)
        mse = {name: residual_mse(trainer, summary, lo, hi, n, rng) for name, (lo, hi) in WINDOWS.items()}
        info, _ = parse_cell(run_dir.parent.name)
        runs.append({"run_dir": str(run_dir).replace("\\", "/"), "cell": info["cell"], "model": summary["model"],
                     "variant": summary.get("variant", ""), "seed": int(summary.get("seed", 0)),
                     "noise": float(summary["noise"]), "mse_R0": mse["R0"], "mse_F": mse["F"],
                     "ratio_F_over_R0": mse["F"] / mse["R0"]})
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for r in runs:
        groups.setdefault((r["cell"], r["model"], r["variant"], r["noise"]), []).append(r)
    summary_rows = [{"cell": c, "model": m, "variant": v, "noise": s, "n": len(g),
                     "median_R0": float(np.median([r["mse_R0"] for r in g])),
                     "median_F": float(np.median([r["mse_F"] for r in g])),
                     "median_ratio": float(np.median([r["ratio_F_over_R0"] for r in g]))}
                    for (c, m, v, s), g in sorted(groups.items())]
    return {"meta": {"n_points": n, "seed": seed, "windows": WINDOWS, "reference_v10_probe": V10_REFERENCE,
                     "residual": "total derivative, scale-normalised, mean square over tanks and components"},
            "runs": runs, "summary": summary_rows, "skipped_non_pinn": skipped}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="results/v11/pinn")
    parser.add_argument("--out", default="results/v11/residual_diagnostic.json")
    parser.add_argument("--n-points", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    result = diagnose(Path(args.root), args.n_points, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
    for s in result["summary"]:
        print("%-14s %-14s %-12s sigma %.2f  n=%d  R0 %.3g  F %.3g  F/R0 %.1f"
              % (s["cell"], s["model"], s["variant"] or "-", s["noise"], s["n"], s["median_R0"], s["median_F"],
                 s["median_ratio"]))
    print("wrote", args.out)


if __name__ == "__main__":
    main()

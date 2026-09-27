"""Audit the v1.0 PINN checkpoints for the partial-derivative defect (SI table S5).

v1.0 trained the physics residual on ``dZ/dt`` taken in ``t`` alone, with the
time-varying influent inputs ``q_in`` and ``z_in`` held fixed. This script
reloads every saved PINN checkpoint and measures, on the same network, how far
the residual it was optimised on (partial) sits from the residual it should have
satisfied (total derivative along the influent trajectory).

Two sampling schemes are reported for each window:

``random_segment``
    ``N_POINTS`` uniform random times; influent by linear interpolation and its
    slope from the enclosing 15-min segment. This is exactly what
    ``Trainer._collocation`` feeds the v1.1 residual, so it is the primary
    quantity.
``knots_central``
    Every 15-min grid knot in the window; slopes by ``np.gradient`` on the full
    dry-weather grid. This reproduces the review-panel probe that first found
    the defect, and is the acceptance check against its numbers. A central
    slope averages the two adjacent segments, so this statistic sits below
    ``random_segment`` (by 1.3-1.9x on the v1.0 checkpoints).

Usage::

    python -m scripts.audit_derivative            # writes results/v11/derivative_audit.{json,csv}
    python -m scripts.audit_derivative --dirs runs --limit 2   # quick look
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.train.run import MODEL_SPECS, RAS_CHANNEL, RunConfig, Trainer

RUN_DIRS = ("runs", "runs_seed1", "runs_seed2", "runs_ablation", "runs_flowonly", "runs_icmask")
WINDOWS = {"train": (0.0, 12.0), "holdout": (12.0, 14.0)}
N_POINTS = 4096
PANEL_REFERENCE = {
    ("runs", "cl_pinn_sigma0p10"): 1.7,
    ("runs", "pinn_sigma0p10"): 11.0,
    ("runs", "cl_pinn_sigma0p00"): 20.0,
    ("runs", "pinn_sigma0p00"): 107.0,
}
PANEL_TOLERANCE = 0.20
CL_BAND = (1.5, 2.3)


def load_trainer(checkpoint: Path) -> Trainer:
    """Rebuild the run's Trainer on CPU in float64 and load its weights."""
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    known = {f.name for f in fields(RunConfig)}
    cfg = {k: v for k, v in dict(ck["config"]).items() if k in known}
    cfg.update(device="cpu", dtype="float64")
    cfg["holdout_days"] = tuple(cfg.get("holdout_days", (12.0, 14.0)))
    if not Path(cfg["data_dir"]).is_absolute():
        cfg["data_dir"] = str(REPO / cfg["data_dir"])
    trainer = Trainer(RunConfig(**cfg))
    trainer.model.load_state_dict(ck["state_dict"])
    trainer.model.eval()
    return trainer


def sample_inputs(trainer: Trainer, window: tuple[float, float], sampling: str, seed: int) -> dict[str, np.ndarray]:
    dry = trainer.data["dry"]
    grid = np.asarray(dry.t, dtype=float)
    ras_all = np.asarray(dry.obs[:, trainer.channel_index[RAS_CHANNEL]], dtype=float)
    lo_t, hi_t = window
    if sampling == "random_segment":
        ts = np.random.default_rng(seed).uniform(lo_t, hi_t, N_POINTS)
        idx = np.clip(np.searchsorted(grid, ts, "right") - 1, 0, len(grid) - 2)
        span = grid[idx + 1] - grid[idx]
        q = np.interp(ts, grid, dry.q_in)
        z = np.stack([np.interp(ts, grid, dry.z_in[:, j]) for j in range(dry.z_in.shape[1])], -1)
        dq = (dry.q_in[idx + 1] - dry.q_in[idx]) / span
        dz = (dry.z_in[idx + 1] - dry.z_in[idx]) / span[:, None]
        ras = np.interp(ts, grid, ras_all)
    elif sampling == "knots_central":
        mask = (grid >= lo_t - 1e-9) & (grid <= hi_t + 1e-9)
        ts = grid[mask]
        q, z, ras = dry.q_in[mask], dry.z_in[mask], ras_all[mask]
        dq = np.gradient(dry.q_in, grid)[mask]
        dz = np.gradient(dry.z_in, grid, axis=0)[mask]
    else:
        raise ValueError("unknown sampling %r" % (sampling,))
    return {"t": ts, "q": q, "z": z, "dq": dq, "dz": dz, "ras": ras}


def audit_one(trainer: Trainer, inputs: dict[str, np.ndarray]) -> dict[str, float]:
    def T(x):
        return torch.as_tensor(np.asarray(x), dtype=torch.float64)

    t = T(inputs["t"]).view(-1, 1).requires_grad_(True)
    q, z = T(inputs["q"]).view(-1, 1), T(inputs["z"])
    dq, dz = T(inputs["dq"]).view(-1, 1), T(inputs["dz"])
    ras = T(inputs["ras"]).view(-1, 1)
    model, loss = trainer.model, trainer.loss

    z_hat, d_part = model.state_and_derivative(t, q, z)
    _, d_tot = model.state_and_derivative(t, q, z, dq_dt=dq, dz_dt=dz)
    with torch.no_grad():
        r_part = loss.physics_residual(z_hat, d_part, q, z, ras)
        r_tot = loss.physics_residual(z_hat, d_tot, q, z, ras)
        scale = loss.state_scale
        infl = (d_tot - d_part).flatten(1)
        share_scaled = ((d_tot - d_part) / scale).flatten(1).norm(dim=1) / (d_tot / scale).flatten(1).norm(dim=1).clamp(min=1e-30)
        share_raw = infl.norm(dim=1) / d_tot.flatten(1).norm(dim=1).clamp(min=1e-30)
    partial = float((r_part ** 2).mean())
    total = float((r_tot ** 2).mean())
    return {
        "partial_mse": partial,
        "total_mse": total,
        "ratio_total_over_partial": total / max(partial, 1e-300),
        "influent_share_scaled_median": float(share_scaled.median()),
        "influent_share_raw_median": float(share_raw.median()),
        "n_points": int(len(inputs["t"])),
    }


def checkpoints(results: Path, dirs: tuple[str, ...]) -> list[tuple[str, Path]]:
    found = []
    for d in dirs:
        root = results / d
        if not root.is_dir():
            continue
        for run in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
            ck = run / "checkpoint.pt"
            if ck.exists():
                found.append((d, ck))
    return found


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def acceptance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lookup = {
        (r["results_dir"], r["run_id"]): r["ratio_total_over_partial"]
        for r in rows if r["window"] == "train" and r["sampling"] == "knots_central"
    }
    checks = []
    for key, ref in PANEL_REFERENCE.items():
        got = lookup.get(key)
        ok = got is not None and abs(got - ref) <= PANEL_TOLERANCE * ref
        checks.append({"run": "%s/%s" % key, "panel_ratio": ref, "audit_ratio": got, "pass": bool(ok)})
    for key in (("runs", "cl_pinn_sigma0p10"), ("runs_seed1", "cl_pinn_sigma0p10")):
        got = lookup.get(key)
        ok = got is not None and CL_BAND[0] <= got <= CL_BAND[1]
        checks.append({"run": "%s/%s" % key, "band": list(CL_BAND), "audit_ratio": got, "pass": bool(ok)})
    return {"sampling": "knots_central", "window": "train", "checks": checks,
            "all_pass": all(c["pass"] for c in checks)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=str(REPO / "results"))
    parser.add_argument("--dirs", nargs="*", default=list(RUN_DIRS))
    parser.add_argument("--out", default=str(REPO / "results" / "v11" / "derivative_audit"))
    parser.add_argument("--limit", type=int, default=0, help="audit only the first N checkpoints")
    args = parser.parse_args(argv)

    torch.set_grad_enabled(True)
    rows: list[dict[str, Any]] = []
    found = checkpoints(Path(args.results), tuple(args.dirs))
    if args.limit:
        found = found[: args.limit]
    started = time.perf_counter()
    for results_dir, ck in found:
        meta = torch.load(ck, map_location="cpu", weights_only=False)["config"]
        if MODEL_SPECS[meta["model"]]["arch"] != "pinn":
            continue
        trainer = load_trainer(ck)
        for w_index, (window, span) in enumerate(WINDOWS.items()):
            for sampling in ("random_segment", "knots_central"):
                inputs = sample_inputs(trainer, span, sampling, seed=w_index)
                stats = audit_one(trainer, inputs)
                rows.append({
                    "results_dir": results_dir,
                    "run_id": ck.parent.name,
                    "model": meta["model"],
                    "curriculum": MODEL_SPECS[meta["model"]]["curriculum"],
                    "noise": float(meta["noise"]),
                    "seed": int(meta.get("seed", 0)),
                    "window": window,
                    "sampling": sampling,
                    **stats,
                })
        print("%-14s %-28s train ratio: segment %.3g, knots %.3g"
              % (results_dir, ck.parent.name,
                 rows[-4]["ratio_total_over_partial"], rows[-3]["ratio_total_over_partial"]))

    n_ckpt = len({(r["results_dir"], r["run_id"]) for r in rows})
    report = {
        "label": "v1.0 checkpoints (partial-derivative residual), derivative audit, SI table S5",
        "git_commit": git_commit(),
        "n_checkpoints": n_ckpt,
        "n_points_random": N_POINTS,
        "definitions": {
            "partial_mse": "mean squared scale-normalised ASM1 residual with dZ/dt taken in t alone (v1.0 training quantity)",
            "total_mse": "same residual with dZ/dt along the influent trajectory, dZ/dt|_u + J_q q' + J_z z'",
            "influent_share_scaled_median": "median over points of ||(J_q q' + J_z z')/s|| / ||(dZ/dt)_total/s||, s = component scale",
            "random_segment": "uniform random times, segment slopes of the linear interpolant (the v1.1 training quantity)",
            "knots_central": "grid knots, np.gradient slopes (the review-panel probe); central slopes average two segments, so ratios sit 1.3-1.9x below random_segment",
        },
        "acceptance": acceptance(rows),
        "rows": rows,
        "seconds": time.perf_counter() - started,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with open(out.with_suffix(".csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("audited %d checkpoints in %.1f s -> %s.{json,csv}" % (n_ckpt, report["seconds"], out))
    print("acceptance (knots_central, train): %s" % ("PASS" if report["acceptance"]["all_pass"] else "FAIL"))
    for check in report["acceptance"]["checks"]:
        print("   %-32s %s" % (check["run"], check))
    return 0 if report["acceptance"]["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

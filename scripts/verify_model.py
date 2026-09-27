"""RUNBOOK step 6 - model-side verification, before any benchmark run is trusted.

6a_partial          the v1.0 quantity: dZ/dt in t alone, influent held fixed,
                    against a central difference in t alone. Kept for the
                    record; it cannot see a missing influent path.
6a' total           the v1.1 quantity: dZ/dt along the piecewise-linear
                    influent trajectory against the trajectory central
                    difference (model(t+h, u(t+h)) - model(t-h, u(t-h))) / 2h,
                    at times that sit off the 15-min knots
6b  mode agreement    forward-mode JVP against the reverse-mode VJP loop, on
                      the total derivative
6c  physics wiring    the residual actually reaches the parameters: its gradient
                      is non-zero, and a short run with the physics weight on
                      leaves a materially smaller residual than one with it off
6d  no leakage        every ground-truth value after t = 0 is replaced by NaN and
                      the loss stays finite, proving nothing but the sensors, the
                      known influent and the supplied initial condition is used

Requires the datasets from step 4/5. Run::

    python -m scripts.verify_model

Exit code 0 means the model plumbing is sound.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train.run import RunConfig, Trainer

TOL_AUTOGRAD = 1e-4
TOL_MODE_AGREEMENT = 1e-6
PROBE_STEPS = 200
PROBE_NOISE = 0.05
OFF_KNOT_TIMES = 0.5 + np.arange(16) * 0.25 + 0.1


def _trainer(model: str = "cl_pinn", steps: int = PROBE_STEPS, total_derivative: bool = True) -> Trainer:
    cfg = RunConfig(
        run_id="_verify_%s" % model,
        model=model,
        noise=PROBE_NOISE,
        profile="quick",
        steps_quick=steps,
        log_every=max(steps // 4, 1),
        dtype="float64",
        device="cpu",
        total_derivative=total_derivative,
        out_dir="results/v11/_verify",
    )
    return Trainer(cfg)


def _influent_at(trainer: Trainer, times: np.ndarray) -> tuple[torch.Tensor, ...]:
    """``(q, z, dq/dt, dz/dt)`` on the dry grid: linear interpolation and segment slopes."""
    dry = trainer.data["dry"]
    grid = np.asarray(dry.t, dtype=float)
    q = np.interp(times, grid, dry.q_in)
    z = np.stack([np.interp(times, grid, dry.z_in[:, j]) for j in range(dry.z_in.shape[1])], -1)
    idx = np.clip(np.searchsorted(grid, times, "right") - 1, 0, len(grid) - 2)
    span = grid[idx + 1] - grid[idx]
    dq = (dry.q_in[idx + 1] - dry.q_in[idx]) / span
    dz = (dry.z_in[idx + 1] - dry.z_in[idx]) / span[:, None]

    def T(x):
        return torch.as_tensor(np.asarray(x), dtype=trainer.dtype)

    return T(q).view(-1, 1), T(z), T(dq).view(-1, 1), T(dz)


def _per_output_relative(a: torch.Tensor, b: torch.Tensor) -> float:
    """Max |a - b| over samples, normalised per output by that output's largest magnitude."""
    col = torch.maximum(a.abs(), b.abs()).amax(dim=0, keepdim=True).clamp(min=1e-12)
    return float(((a - b).abs() / col).max())


def gate_6a(trainer: Trainer) -> tuple[bool, dict]:
    """6a_partial (v1.0 quantity): partial derivative in t against a central difference in t."""
    model = trainer.model
    t = torch.linspace(0.5, 6.0, 8, dtype=trainer.dtype).view(-1, 1).requires_grad_(True)
    dry = trainer.data["dry"]

    def inputs_at(times: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        grid = np.asarray(dry.t)
        vals = times.detach().cpu().numpy().ravel()
        q = np.interp(vals, grid, dry.q_in)
        z = np.stack([np.interp(vals, grid, dry.z_in[:, j]) for j in range(dry.z_in.shape[1])], -1)
        return (
            torch.as_tensor(q, dtype=trainer.dtype).view(-1, 1),
            torch.as_tensor(z, dtype=trainer.dtype),
        )

    q_in, z_in = inputs_at(t)
    _, dz = model.state_and_derivative(t, q_in, z_in)

    h = 1e-6
    with torch.no_grad():
        plus = model(t.detach() + h, q_in, z_in)
        minus = model(t.detach() - h, q_in, z_in)
    fd = (plus - minus) / (2.0 * h)

    scale = torch.maximum(dz.abs(), fd.abs()).clamp(min=1e-8)
    err = float(((dz - fd).abs() / scale).max())
    return err < TOL_AUTOGRAD, {
        "mode": model.cfg.derivative_mode,
        "max_relative_error_vs_finite_difference": err,
        "step_h": h,
    }


def gate_6a_total(trainer: Trainer) -> tuple[bool, dict]:
    """6a': total derivative along the influent trajectory against the trajectory central difference.

    Relative error is taken per output (normalised by that output's largest
    magnitude over the probe times), because a pointwise ratio is dominated by
    outputs whose derivative happens to pass through zero. The partial
    derivative is scored the same way to show the gate can tell them apart.
    """
    model = trainer.model
    times = OFF_KNOT_TIMES.astype(float)
    t = torch.as_tensor(times, dtype=trainer.dtype).view(-1, 1).requires_grad_(True)
    q, z, dq, dz = _influent_at(trainer, times)
    _, d_tot = model.state_and_derivative(t, q, z, dq_dt=dq, dz_dt=dz)
    _, d_par = model.state_and_derivative(t, q, z)

    h = 1e-6
    with torch.no_grad():
        tt = t.detach()
        qp, zp, _, _ = _influent_at(trainer, times + h)
        qm, zm, _, _ = _influent_at(trainer, times - h)
        fd = (model(tt + h, qp, zp) - model(tt - h, qm, zm)) / (2.0 * h)

    d_tot, d_par = d_tot.detach(), d_par.detach()
    err = _per_output_relative(d_tot, fd)
    pointwise = float(((d_tot - fd).abs() / torch.maximum(d_tot.abs(), fd.abs()).clamp(min=1e-8)).max())
    return err < TOL_AUTOGRAD, {
        "mode": model.cfg.derivative_mode,
        "max_per_output_relative_error_total": err,
        "max_pointwise_relative_error_total": pointwise,
        "max_per_output_relative_error_partial_vs_trajectory": _per_output_relative(d_par, fd),
        "step_h": h,
        "n_times": int(len(times)),
    }


def gate_6b(trainer: Trainer) -> tuple[bool, dict]:
    """Forward-mode and reverse-mode total derivatives must agree to machine accuracy."""
    model = trainer.model
    times = OFF_KNOT_TIMES[:5].astype(float)
    t = torch.as_tensor(times, dtype=trainer.dtype).view(-1, 1).requires_grad_(True)
    q, z, dq, dz = _influent_at(trainer, times)
    _, fwd = model.state_and_derivative(t, q, z, mode="forward", dq_dt=dq, dz_dt=dz)
    _, rev = model.state_and_derivative(t, q, z, mode="reverse", dq_dt=dq, dz_dt=dz)
    scale = torch.maximum(fwd.abs(), rev.abs()).clamp(min=1e-12)
    err = float(((fwd - rev).abs() / scale).max())
    return err < TOL_MODE_AGREEMENT, {"max_relative_difference_total": err}


def gate_6c() -> tuple[bool, dict]:
    """The physics term must reach the parameters and must change the outcome."""
    with_physics = _trainer("cl_pinn")

    first_stage = with_physics.schedule.stages[0]
    batch = with_physics._stage_tensors(first_stage)
    colloc = with_physics._collocation(first_stage, batch)
    z_c, dz_c = with_physics.model.state_and_derivative(
        colloc["t"], colloc["q_in"], colloc["z_in"],
        dq_dt=colloc["dq_dt"], dz_dt=colloc["dz_dt"],
    )
    residual = with_physics.loss.physics_residual(
        z_c, dz_c, colloc["q_in"], colloc["z_in"], colloc["tss_ras"]
    )
    loss = torch.mean(residual ** 2)
    with_physics.model.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = float(
        torch.sqrt(sum((p.grad ** 2).sum() for p in with_physics.model.parameters()
                       if p.grad is not None))
    )

    res_on = _final_residual("cl_pinn")
    res_off = _final_residual("pinn_nophysics")

    ok = grad_norm > 0.0 and res_on < res_off
    return ok, {
        "residual_gradient_norm": grad_norm,
        "final_residual_physics_on": res_on,
        "final_residual_physics_off": res_off,
        "ratio_off_over_on": res_off / max(res_on, 1e-30),
    }


def _final_residual(model: str) -> float:
    """Mean squared physics residual of a freshly trained probe model."""
    trainer = _trainer(model)
    trainer.train()
    stage = trainer.schedule.stages[-1]
    batch = trainer._stage_tensors(stage)
    colloc = trainer._collocation(stage, batch)
    z_c, dz_c = trainer.model.state_and_derivative(
        colloc["t"], colloc["q_in"], colloc["z_in"],
        dq_dt=colloc["dq_dt"], dz_dt=colloc["dz_dt"],
    )
    residual = trainer.loss.physics_residual(
        z_c, dz_c, colloc["q_in"], colloc["z_in"], colloc["tss_ras"]
    )
    return float(torch.mean(residual ** 2).detach())


def gate_6d(trainer: Trainer) -> tuple[bool, dict]:
    """Poison the ground truth after t = 0; the loss must stay finite."""
    stage = trainer.schedule.stages[-1]
    poisoned = trainer.data["dry"].window(0.0, trainer.cfg.train_end_day)
    poisoned.truth_reactor = poisoned.truth_reactor.copy()
    poisoned.truth_reactor[1:] = np.nan
    poisoned.truth_y = poisoned.truth_y.copy()
    poisoned.truth_y[1:] = np.nan
    trainer.train_set = poisoned
    trainer.data["dry"] = poisoned
    trainer._tensor_cache.clear()

    batch = trainer._stage_tensors(stage)
    z = trainer.model(batch["t"], batch["q_in"], batch["z_in"])
    parts = trainer.loss.total(
        weights=stage.weights_end,
        t=batch["t"], z=z, dz_dt=None,
        q_in=batch["q_in"], z_in=batch["z_in"], tss_ras=batch["tss_ras"],
        targets=batch["targets"], z0_pred=z[:1], z0_true=batch["z0_true"],
    )
    values = parts.detached()
    finite = all(np.isfinite(v) for v in values.values())
    return finite, {"loss_terms_with_poisoned_truth": values}


def main() -> int:
    print("Building a probe trainer on the sigma=%.2f dry dataset (CPU, float64)...\n" % PROBE_NOISE)
    trainer = _trainer()

    failures = []
    for name, fn in (
        ("6a_partial (v1.0 quantity)", lambda: gate_6a(trainer)),
        ("6a' total derivative vs trajectory FD", lambda: gate_6a_total(trainer)),
        ("6b forward vs reverse mode (total)", lambda: gate_6b(trainer)),
        ("6c physics term is wired and effective", gate_6c),
        ("6d no ground-truth leakage", lambda: gate_6d(_trainer())),
    ):
        ok, detail = fn()
        print("%-40s %s" % (name, "PASS" if ok else "FAIL"))
        for key, value in detail.items():
            print("    %-38s %s" % (key, value))
        if not ok:
            failures.append(name)
        print()

    if failures:
        print("FAIL - gates not cleared: %s" % ", ".join(failures))
        return 1
    print("PASS - step 6 clear, benchmark runs may proceed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

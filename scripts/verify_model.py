"""RUNBOOK step 6 - model-side verification, before any benchmark run is trusted.

6a  autograd          the active derivative mode against central finite differences
6b  mode agreement    forward-mode JVP against the reverse-mode VJP loop
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

from src.train.run import RunConfig, Trainer  # noqa: E402

TOL_AUTOGRAD = 1e-4
TOL_MODE_AGREEMENT = 1e-6
PROBE_STEPS = 200
PROBE_NOISE = 0.05


def _trainer(model: str = "cl_pinn", steps: int = PROBE_STEPS) -> Trainer:
    cfg = RunConfig(
        run_id="_verify_%s" % model,
        model=model,
        noise=PROBE_NOISE,
        profile="quick",
        steps_quick=steps,
        log_every=max(steps // 4, 1),
        dtype="float64",
        device="cpu",
    )
    return Trainer(cfg)


def gate_6a(trainer: Trainer) -> tuple[bool, dict]:
    """Autograd derivative against a central finite difference."""
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


def gate_6b(trainer: Trainer) -> tuple[bool, dict]:
    """Forward-mode and reverse-mode derivatives must agree to machine accuracy."""
    model = trainer.model
    t = torch.linspace(0.5, 6.0, 5, dtype=trainer.dtype).view(-1, 1).requires_grad_(True)
    dry = trainer.data["dry"]
    q = torch.as_tensor(np.interp(t.detach().numpy().ravel(), dry.t, dry.q_in),
                        dtype=trainer.dtype).view(-1, 1)
    z = torch.as_tensor(
        np.stack([np.interp(t.detach().numpy().ravel(), dry.t, dry.z_in[:, j])
                  for j in range(dry.z_in.shape[1])], -1),
        dtype=trainer.dtype,
    )
    _, fwd = model.state_and_derivative(t, q, z, mode="forward")
    _, rev = model.state_and_derivative(t, q, z, mode="reverse")
    scale = torch.maximum(fwd.abs(), rev.abs()).clamp(min=1e-12)
    err = float(((fwd - rev).abs() / scale).max())
    return err < TOL_MODE_AGREEMENT, {"max_relative_difference": err}


def gate_6c() -> tuple[bool, dict]:
    """The physics term must reach the parameters and must change the outcome."""
    with_physics = _trainer("cl_pinn")

    # (i) gradient of the residual alone is non-zero
    first_stage = with_physics.schedule.stages[0]
    batch = with_physics._stage_tensors(first_stage)
    colloc = with_physics._collocation(first_stage, batch)
    z_c, dz_c = with_physics.model.state_and_derivative(
        colloc["t"], colloc["q_in"], colloc["z_in"]
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

    # (ii) a short run with the physics weight on leaves a smaller residual
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
        colloc["t"], colloc["q_in"], colloc["z_in"]
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
        ("6a autograd vs finite difference", lambda: gate_6a(trainer)),
        ("6b forward vs reverse mode", lambda: gate_6b(trainer)),
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

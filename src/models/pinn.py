"""Physics-informed network for the five-reactor state trajectory.

The network is a coordinate model: it maps time (plus the known influent
signals) to all 70 reactor states, and its time derivative is taken with
autograd so the ASM1 residual can be evaluated at arbitrary collocation points.

Output parameterisation
-----------------------
States are strictly non-negative, so the head is
``Z = scale * softplus(raw + inv_softplus(1))``. The bias shift makes
``raw = 0`` map to ``Z = scale``, which keeps the network near a physically
sensible operating point at initialisation.

``scale`` is a per-component constant taken from the supplied initial state
``Z(0)`` - the same initial condition every model in the benchmark receives as a
boundary condition. It is a per-component maximum across tanks, so it carries
magnitude information only, not the tank-to-tank profile that the model has to
infer. Nothing from ``t > 0`` ever touches it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .features import FeatureBuilder, FeatureConfig

_INV_SOFTPLUS_1 = float(np.log(np.e - 1.0))  # softplus(x) == 1


@dataclass(frozen=True)
class PinnConfig:
    hidden_width: int = 128
    hidden_depth: int = 5
    activation: str = "tanh"
    features: FeatureConfig = FeatureConfig()
    scale_floor_fraction: float = 1e-3
    #: "forward" uses one JVP for the scalar input t; "reverse" uses one VJP per
    #: output. Both are exact. RUNBOOK step 5 checks the active mode against
    #: central finite differences before any training run is trusted.
    derivative_mode: str = "forward"


def _activation(name: str) -> nn.Module:
    table = {"tanh": nn.Tanh, "silu": nn.SiLU, "gelu": nn.GELU}
    if name not in table:
        raise ValueError("Unsupported activation %r; expected one of %s" % (name, sorted(table)))
    return table[name]()


def component_scale(z0: np.ndarray, floor_fraction: float = 1e-3) -> np.ndarray:
    """Per-component output scale from the initial reactor state ``(5, 14)``.

    The maximum across tanks is used, floored at a small fraction of the largest
    component so that components which start at zero in every tank (dissolved
    oxygen in the anoxic tanks, for instance) can still grow.
    """
    z0 = np.asarray(z0, dtype=float)
    per_component = np.max(np.abs(z0), axis=0)
    floor = floor_fraction * float(np.max(per_component)) if per_component.size else 0.0
    return np.maximum(per_component, max(floor, 1e-6))


class Asm1Pinn(nn.Module):
    """Coordinate network ``t -> Z(t)`` for the five-tank reactor train."""

    def __init__(
        self,
        n_tanks: int,
        n_components: int,
        horizon_days: float,
        z0: np.ndarray,
        q_scale: float,
        z_in_scale: np.ndarray,
        cfg: PinnConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg if cfg is not None else PinnConfig()
        self.n_tanks = int(n_tanks)
        self.n_components = int(n_components)
        self.n_outputs = self.n_tanks * self.n_components

        self.features = FeatureBuilder(
            self.cfg.features, horizon_days=horizon_days, q_scale=q_scale, z_scale=z_in_scale
        )

        layers: list[nn.Module] = [nn.Linear(self.features.n_features, self.cfg.hidden_width)]
        layers.append(_activation(self.cfg.activation))
        for _ in range(self.cfg.hidden_depth - 1):
            layers.append(nn.Linear(self.cfg.hidden_width, self.cfg.hidden_width))
            layers.append(_activation(self.cfg.activation))
        layers.append(nn.Linear(self.cfg.hidden_width, self.n_outputs))
        self.net = nn.Sequential(*layers)

        # Small final layer so the untrained network starts near Z = scale.
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.zero_()

        scale = component_scale(z0, self.cfg.scale_floor_fraction)
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float64))

    def forward(
        self, t: torch.Tensor, q_in: torch.Tensor, z_in: torch.Tensor
    ) -> torch.Tensor:
        """``t`` is ``(n, 1)``; returns predicted states ``(n, n_tanks, n_components)``."""
        x = self.features.build(t, q_in, z_in)
        raw = self.net(x).view(-1, self.n_tanks, self.n_components)
        return self.scale * nn.functional.softplus(raw + _INV_SOFTPLUS_1)

    def _derivative_reverse(
        self, t: torch.Tensor, q_in: torch.Tensor, z_in: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One VJP per output. Exact, slow, and free of any functorch dependency."""
        if not t.requires_grad:
            raise ValueError("t must require grad for the physics residual")
        z = self(t, q_in, z_in)
        flat = z.reshape(z.shape[0], -1)
        grads = []
        for j in range(flat.shape[1]):
            basis = torch.zeros_like(flat)
            basis[:, j] = 1.0
            (g,) = torch.autograd.grad(
                flat, t, grad_outputs=basis, create_graph=True, retain_graph=True
            )
            grads.append(g)
        return z, torch.cat(grads, dim=-1).view_as(z)

    def _derivative_forward(
        self, t: torch.Tensor, q_in: torch.Tensor, z_in: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One JVP for the scalar input t - about 70x cheaper than reverse mode."""
        from torch.func import jvp

        def f(time: torch.Tensor) -> torch.Tensor:
            return self(time, q_in, z_in)

        return jvp(f, (t,), (torch.ones_like(t),))

    def state_and_derivative(
        self,
        t: torch.Tensor,
        q_in: torch.Tensor,
        z_in: torch.Tensor,
        mode: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(Z, dZ/dt)`` at the collocation times ``t``."""
        mode = mode or self.cfg.derivative_mode
        if mode == "forward":
            return self._derivative_forward(t, q_in, z_in)
        if mode == "reverse":
            return self._derivative_reverse(t, q_in, z_in)
        raise ValueError("derivative_mode must be 'forward' or 'reverse', got %r" % (mode,))

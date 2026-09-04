"""Training losses, including the ASM1 physics residual.

This is where the project is a *full* PINN rather than a network with a physics
flavour: :meth:`Asm1Loss.physics_residual` evaluates the vault's ASM1 conversion
rates on the network's own prediction, differentiates the prediction with
respect to time, and the squared mismatch enters the training loss on every
step. ``lambda_physics`` is ramped by the curriculum but never reaches zero.

Loss terms
----------
``L_data``
    Seven measured target channels. ``TSS_ras`` is deliberately not a target:
    it is a settler quantity, and the model only spans the reactor train, so it
    enters as a measured *input* to the recycle reconstruction instead.
``L_physics``
    ``dZ/dt - f_ASM1(Z, u(t))`` at collocation points, per tank, per component,
    normalised by the per-component output scale.
``L_ic``
    The supplied initial state at ``t = 0``. Every model in the benchmark gets
    this same boundary condition.
``L_pos``
    ``relu(-Z)^2``. Structurally zero under the softplus head, kept as a running
    assertion; ``tests/test_losses.py`` checks it stays zero.
``L_balance``
    Integral COD and N closure over the training window. The pointwise version
    of a continuity check is *also* structurally zero (``r @ C = rho @ (nu @ C)``
    and ``nu @ C`` vanishes by the vault audit), so it would supervise nothing.
    The integral form is not: it ties the endpoints of the trajectory to the
    accumulated boundary fluxes and supplies long-horizon signal that the
    pointwise residual only provides weakly.

Recycle reconstruction
----------------------
Closing the tank-1 balance needs the return-sludge composition, which lives in
the clarifier. It is rebuilt from measurable quantities only:
solubles are taken from tank 5 (the clarifier is non-reactive and its soluble
layers track the feed), and particulates are the tank-5 particulates rescaled by
the measured ``TSS_ras`` over the predicted tank-5 TSS. No unobserved ground
truth is touched.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..asm1.model import Asm1Kinetics
from ..asm1.plant import Bsm1Plant
from ..data.sensors import SensorChannel


@dataclass
class LossWeights:
    data: float = 1.0
    physics: float = 1.0
    ic: float = 10.0
    positivity: float = 1.0
    balance: float = 0.1

    def scaled(self, **overrides: float) -> "LossWeights":
        return LossWeights(**{**self.__dict__, **overrides})


@dataclass
class LossParts:
    total: torch.Tensor
    data: torch.Tensor
    physics: torch.Tensor
    ic: torch.Tensor
    positivity: torch.Tensor
    balance: torch.Tensor

    def detached(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name).detach().cpu())
            for name in ("total", "data", "physics", "ic", "positivity", "balance")
        }


class ObservationOperator:
    """Maps a predicted reactor state onto the measured target channels."""

    def __init__(self, plant: Bsm1Plant, channels: tuple[SensorChannel, ...]) -> None:
        self.plant = plant
        self.channels = tuple(channels)
        self.names = tuple(c.name for c in self.channels)
        self._spec: list[tuple[str, int, int | None]] = []
        for c in self.channels:
            if c.kind == "state":
                self._spec.append(("state", int(c.tank), plant.vault.index(str(c.component))))
            elif c.kind == "tss_reactor":
                self._spec.append(("tss", int(c.tank), None))
            else:
                raise ValueError(
                    "Channel %r has kind %r, which is a measured input rather than "
                    "a prediction target and must not be passed to the "
                    "ObservationOperator." % (c.name, c.kind)
                )

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        """``z`` is ``(n, 5, 14)``; returns ``(n, n_targets)``."""
        i_tss = torch.as_tensor(self.plant.i_tss, device=z.device, dtype=torch.long)
        factor = self.plant.cfg.tss_factor
        cols = []
        for kind, tank, comp in self._spec:
            if kind == "state":
                cols.append(z[:, tank, comp])
            else:
                cols.append(factor * z[:, tank].index_select(-1, i_tss).sum(-1))
        return torch.stack(cols, dim=-1)


class Asm1Loss:
    """All loss terms for one training window."""

    def __init__(
        self,
        plant: Bsm1Plant,
        operator: ObservationOperator,
        state_scale: np.ndarray,
        target_scale: np.ndarray,
        device: torch.device,
        dtype: torch.dtype = torch.float64,
        ic_mask: np.ndarray | None = None,
    ) -> None:
        self.plant = plant
        self.cfg = plant.cfg
        self.operator = operator
        self.kinetics: Asm1Kinetics = plant.kinetics
        self.device = device
        self.dtype = dtype

        def T(x, kind=dtype):
            return torch.as_tensor(np.asarray(x), device=device, dtype=kind)

        self.state_scale = T(np.maximum(state_scale, 1e-9))
        self.target_scale = T(np.maximum(target_scale, 1e-9))
        # Optional (n_tanks, n_components) 0/1 mask restricting the IC anchor
        # to directly sensed entries; None anchors the full supplied state.
        self.ic_mask = None if ic_mask is None else T(ic_mask)
        self.volumes = T(self.cfg.volumes).view(1, -1, 1)
        self.kla = T(self.cfg.kla).view(1, -1, 1)
        self.so_sat = T(self.cfg.so_sat)
        self.q_int = T(self.cfg.q_int)
        self.q_r = T(self.cfg.q_r)
        self.tss_factor = float(self.cfg.tss_factor)
        # Mirrors Bsm1Plant.rhs. If the plant is configured with reactions off,
        # the physics term must enforce the same dynamics the data was generated
        # with, otherwise the residual silently supervises a different model.
        self.reaction_scale = float(self.cfg.reaction_scale)

        self.i_sol = T(plant.i_soluble, torch.long)
        self.i_part = T(plant.i_particulate, torch.long)
        self.i_tss = T(plant.i_tss, torch.long)
        self.i_so = T(np.array([plant.i_so]), torch.long)
        self.composition = T(plant.vault.composition)  # (14, 3)

    # -- physics -----------------------------------------------------------
    def recycle_composition(self, z5: torch.Tensor, tss_ras: torch.Tensor) -> torch.Tensor:
        """Return-sludge composition from tank-5 prediction and measured RAS TSS."""
        tss5 = self.tss_factor * z5.index_select(-1, self.i_tss).sum(-1, keepdim=True)
        ratio = tss_ras / torch.clamp(tss5, min=1e-9)
        zr = torch.zeros_like(z5)
        zr = zr.index_copy(1, self.i_sol, z5.index_select(1, self.i_sol))
        zr = zr.index_copy(1, self.i_part, z5.index_select(1, self.i_part) * ratio)
        return zr

    def plant_rhs(
        self,
        z: torch.Tensor,
        q_in: torch.Tensor,
        z_in: torch.Tensor,
        tss_ras: torch.Tensor,
    ) -> torch.Tensor:
        """``f_ASM1(Z, u(t))`` for the five reactors; ``z`` is ``(n, 5, 14)``."""
        z5 = z[:, -1, :]
        zr = self.recycle_composition(z5, tss_ras)
        q1 = q_in + self.q_r + self.q_int  # (n, 1)

        load_in = self.q_int * z5 + self.q_r * zr + q_in * z_in  # (n, 14)
        first = (load_in - q1 * z[:, 0, :]) / self.volumes[:, 0, :]
        rest = q1.unsqueeze(-1) * (z[:, :-1, :] - z[:, 1:, :]) / self.volumes[:, 1:, :]
        transport = torch.cat([first.unsqueeze(1), rest], dim=1)

        reaction = self.kinetics.conversion(z) * self.reaction_scale

        aeration = torch.zeros_like(z)
        so = z.index_select(2, self.i_so)              # (n, 5, 1)
        aeration = aeration.index_copy(2, self.i_so, self.kla * (self.so_sat - so))

        return transport + reaction + aeration

    def physics_residual(
        self,
        z: torch.Tensor,
        dz_dt: torch.Tensor,
        q_in: torch.Tensor,
        z_in: torch.Tensor,
        tss_ras: torch.Tensor,
    ) -> torch.Tensor:
        """Scale-normalised ``dZ/dt - f_ASM1(Z, u)``; shape ``(n, 5, 14)``."""
        return (dz_dt - self.plant_rhs(z, q_in, z_in, tss_ras)) / self.state_scale

    # -- individual terms --------------------------------------------------
    def data_loss(self, z: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pred = self.operator(z)
        return torch.mean(((pred - targets) / self.target_scale) ** 2)

    def ic_loss(self, z0_pred: torch.Tensor, z0_true: torch.Tensor) -> torch.Tensor:
        if self.ic_mask is None:
            return torch.mean(((z0_pred - z0_true) / self.state_scale) ** 2)
        # Masked entries never touch z0_true, so a NaN there cannot propagate.
        diff = torch.where(
            self.ic_mask > 0.0, z0_pred - z0_true, torch.zeros_like(z0_pred)
        )
        sq = (diff / self.state_scale) ** 2
        return sq.sum() / torch.clamp(self.ic_mask.sum(), min=1.0)

    def positivity_loss(self, z: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.relu(-z / self.state_scale) ** 2)

    def balance_loss(
        self,
        t: torch.Tensor,
        z: torch.Tensor,
        q_in: torch.Tensor,
        z_in: torch.Tensor,
        tss_ras: torch.Tensor,
    ) -> torch.Tensor:
        """Integral COD and N closure over the window.

        For the reactor train, ``d/dt sum_k V_k (Z_k . C)`` must equal the net
        boundary flux. The internal recycle cancels; oxygen transfer enters
        through the ``S_O`` composition entry (COD coefficient -1).
        """
        order = torch.argsort(t.squeeze(-1))
        t_s = t.squeeze(-1)[order]
        z_s, q_s, zin_s, ras_s = z[order], q_in[order], z_in[order], tss_ras[order]

        holdup = z_s * self.volumes  # (n, 5, 14) mass per tank
        inventory = torch.einsum("ntc,cq->nq", holdup, self.composition)
        delta = inventory[-1] - inventory[0]

        z5 = z_s[:, -1, :]
        zr = self.recycle_composition(z5, ras_s)
        flux = (
            q_s * torch.matmul(zin_s, self.composition)
            + self.q_r * torch.matmul(zr, self.composition)
            - (q_s + self.q_r) * torch.matmul(z5, self.composition)
        )
        so = z_s.index_select(2, self.i_so)
        oxygen = (self.kla * (self.so_sat - so) * self.volumes).sum(dim=(1, 2), keepdim=False)
        flux = flux + oxygen.unsqueeze(-1) * self.composition[self.i_so].squeeze(0)

        integral = torch.trapezoid(flux, t_s, dim=0)
        scale = torch.clamp(torch.abs(integral) + torch.abs(delta), min=1e-6)
        # COD and N only; the charge balance carries the vault's documented
        # missing alkalinity kinetic terms (cells X82 / X84) and is reported
        # rather than enforced.
        return torch.mean((((delta - integral) / scale)[:2]) ** 2)

    # -- assembly ----------------------------------------------------------
    def total(
        self,
        weights: LossWeights,
        t: torch.Tensor,
        z: torch.Tensor,
        dz_dt: torch.Tensor | None,
        q_in: torch.Tensor,
        z_in: torch.Tensor,
        tss_ras: torch.Tensor,
        targets: torch.Tensor,
        z0_pred: torch.Tensor,
        z0_true: torch.Tensor,
    ) -> LossParts:
        zero = torch.zeros((), device=self.device, dtype=self.dtype)

        l_data = self.data_loss(z, targets)
        l_ic = self.ic_loss(z0_pred, z0_true)
        l_pos = self.positivity_loss(z)
        l_bal = self.balance_loss(t, z, q_in, z_in, tss_ras) if weights.balance else zero

        if dz_dt is None:
            l_phys = zero
        else:
            residual = self.physics_residual(z, dz_dt, q_in, z_in, tss_ras)
            l_phys = torch.mean(residual ** 2)

        total = (
            weights.data * l_data
            + weights.physics * l_phys
            + weights.ic * l_ic
            + weights.positivity * l_pos
            + weights.balance * l_bal
        )
        return LossParts(
            total=total, data=l_data, physics=l_phys, ic=l_ic, positivity=l_pos, balance=l_bal
        )

"""Input features shared by every model, so the comparison stays apples-to-apples.

All four benchmark models (CL+PINN, PINN, LSTM+CL, LSTM) receive exactly the
same inputs:

* time, encoded with a Fourier bank at the 1-day and 7-day periods that the
  influent generator actually contains;
* the known influent flow ``Q_in(t)`` and composition ``Z_in(t)``.

They differ only in architecture and in whether the ASM1 physics residual is
part of the training loss. Nothing derived from an unobserved state enters here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

#: Periods present in the influent generator, in days.
DEFAULT_PERIODS: tuple[float, ...] = (1.0, 7.0)
DEFAULT_HARMONICS: tuple[int, ...] = (4, 2)


@dataclass(frozen=True)
class FeatureConfig:
    periods: tuple[float, ...] = DEFAULT_PERIODS
    harmonics: tuple[int, ...] = DEFAULT_HARMONICS
    include_influent: bool = True
    # With include_influent, controls whether the 14-component composition
    # Z_in(t) is included alongside the flow. False leaves only Q_in(t) - the
    # signal a real plant actually measures online (revision ablation).
    include_influent_composition: bool = True


class FeatureBuilder:
    """Builds the model input tensor from time and the known influent signals."""

    def __init__(
        self,
        cfg: FeatureConfig,
        horizon_days: float,
        q_scale: float,
        z_scale: np.ndarray,
    ) -> None:
        if len(cfg.periods) != len(cfg.harmonics):
            raise ValueError("periods and harmonics must have the same length")
        self.cfg = cfg
        self.horizon = float(horizon_days)
        self.q_scale = float(q_scale)
        self.z_scale = np.maximum(np.asarray(z_scale, dtype=float), 1e-9)

    @property
    def n_features(self) -> int:
        n = 1  # normalised time
        n += 2 * int(sum(self.cfg.harmonics))
        if self.cfg.include_influent:
            n += 1
            if self.cfg.include_influent_composition:
                n += len(self.z_scale)
        return n

    def build(
        self, t: torch.Tensor, q_in: torch.Tensor, z_in: torch.Tensor
    ) -> torch.Tensor:
        """``t`` is ``(n, 1)`` in days; returns ``(n, n_features)``.

        ``t`` keeps its autograd graph so ``d/dt`` of the network output is
        available downstream for the physics residual.
        """
        feats = [t / self.horizon]
        for period, n_harm in zip(self.cfg.periods, self.cfg.harmonics):
            for k in range(1, int(n_harm) + 1):
                omega = 2.0 * np.pi * k / period
                feats.append(torch.sin(omega * t))
                feats.append(torch.cos(omega * t))
        if self.cfg.include_influent:
            feats.append(torch.log1p(q_in / self.q_scale))
            if self.cfg.include_influent_composition:
                scale = torch.as_tensor(self.z_scale, device=z_in.device, dtype=z_in.dtype)
                feats.append(z_in / scale)
        return torch.cat(feats, dim=-1)

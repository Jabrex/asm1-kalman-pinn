"""LSTM baseline - identical inputs and identical output head, no physics term.

The point of this baseline is to isolate what the ASM1 residual buys. It gets
exactly the features the PINN gets (time Fourier bank, known ``Q_in`` and
``Z_in``), predicts exactly the same 70 states through exactly the same
softplus-scaled head, and is trained on exactly the same data loss and initial
condition. The only thing it does not have is ``L_physics``.

Expected and intended outcome: on the eight measured channels it should be
competitive; on the nine never-measured components it has no training signal at
all and its predictions stay at the initialisation scale. That is not a bug in
the baseline - it is the measurement the benchmark exists to make, and the
report presents it as Track B rather than hiding it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .features import FeatureBuilder, FeatureConfig
from .pinn import _INV_SOFTPLUS_1, component_scale


@dataclass(frozen=True)
class LstmConfig:
    hidden_size: int = 128
    num_layers: int = 2
    bidirectional: bool = False
    dropout: float = 0.0
    features: FeatureConfig = FeatureConfig()
    scale_floor_fraction: float = 1e-3


class Asm1Lstm(nn.Module):
    """Sequence model ``[t, Q_in, Z_in] -> Z(t)`` over the sampled time grid."""

    def __init__(
        self,
        n_tanks: int,
        n_components: int,
        horizon_days: float,
        z0: np.ndarray,
        q_scale: float,
        z_in_scale: np.ndarray,
        cfg: LstmConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg if cfg is not None else LstmConfig()
        self.n_tanks = int(n_tanks)
        self.n_components = int(n_components)
        self.n_outputs = self.n_tanks * self.n_components

        self.features = FeatureBuilder(
            self.cfg.features, horizon_days=horizon_days, q_scale=q_scale, z_scale=z_in_scale
        )
        self.rnn = nn.LSTM(
            input_size=self.features.n_features,
            hidden_size=self.cfg.hidden_size,
            num_layers=self.cfg.num_layers,
            batch_first=True,
            bidirectional=self.cfg.bidirectional,
            dropout=self.cfg.dropout if self.cfg.num_layers > 1 else 0.0,
        )
        head_in = self.cfg.hidden_size * (2 if self.cfg.bidirectional else 1)
        self.head = nn.Linear(head_in, self.n_outputs)
        with torch.no_grad():
            self.head.weight.mul_(0.01)
            self.head.bias.zero_()

        scale = component_scale(z0, self.cfg.scale_floor_fraction)
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float64))

    def forward(
        self, t: torch.Tensor, q_in: torch.Tensor, z_in: torch.Tensor
    ) -> torch.Tensor:
        """``t`` is ``(n, 1)`` on an ordered grid; returns ``(n, n_tanks, n_components)``."""
        x = self.features.build(t, q_in, z_in).unsqueeze(0)  # (1, n, f)
        out, _ = self.rnn(x)
        raw = self.head(out.squeeze(0)).view(-1, self.n_tanks, self.n_components)
        return self.scale * nn.functional.softplus(raw + _INV_SOFTPLUS_1)

    def state_and_derivative(
        self,
        t: torch.Tensor,
        q_in: torch.Tensor,
        z_in: torch.Tensor,
        mode: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Present for interface parity only.

        The LSTM is a discrete sequence model with no continuous-time derivative,
        and it is never trained with a physics term. Calling this is a
        programming error, not a supported code path.
        """
        raise NotImplementedError(
            "The LSTM baseline has no continuous-time derivative and is trained "
            "without a physics residual by design."
        )

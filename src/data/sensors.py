"""Sensor model: what a real plant actually measures, plus measurement noise.

This module is what makes the study a soft-sensor problem rather than a curve
fit. Only eight signals are exposed to any model as data. Eleven of the fourteen
components are never directly measured in any tank and can only be recovered
through the ASM1 physics term.

Observed (noisy)
    S_O in tanks 3, 4 and 5      dissolved-oxygen probes
    S_NH in tank 5               ammonium analyser
    S_NO in tanks 2 and 5        nitrate analysers (BSM1 places one in tank 2)
    TSS in tank 5                mixed-liquor solids
    TSS in the return sludge     RAS solids

Known inputs (exact, not sensors)
    Q_in(t) and the influent composition Z_in(t), the pump flows Q_int, Q_r,
    Q_w, and the aeration coefficients KLa. Influent characterisation is a
    standard given in activated-sludge modelling - BSM1 itself distributes it
    as an input file - so treating it as known is consistent with the benchmark.

Never observed
    The eleven components S_I, S_S, X_I, X_S, X_B_H, X_B_A, X_P, S_ND, X_ND,
    S_ALK and S_N2, in every tank. The two TSS channels constrain a weighted
    sum of five of the particulates but identify none of them individually.

Noise is multiplicative Gaussian, ``z_obs = z_true * (1 + eps)`` with
``eps ~ N(0, sigma^2)``, clipped at zero. The clipped fraction is recorded so a
high-noise run cannot silently become a biased run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..asm1.vault_loader import vault
from .simulate import SimulationResult

#: Noise levels swept by the benchmark (0 = noiseless reference).
NOISE_LEVELS: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15)


@dataclass(frozen=True)
class SensorChannel:
    """One measured signal."""

    name: str
    kind: str                 # "state" | "tss_reactor" | "tss_underflow"
    tank: int | None = None   # 0-based reactor index
    component: str | None = None

    @property
    def label(self) -> str:
        return self.name


#: The measurement set. Eight channels, all of them realistic online sensors.
SENSOR_SET: tuple[SensorChannel, ...] = (
    SensorChannel("S_O_tank3", "state", tank=2, component="S_O"),
    SensorChannel("S_O_tank4", "state", tank=3, component="S_O"),
    SensorChannel("S_O_tank5", "state", tank=4, component="S_O"),
    SensorChannel("S_NH_tank5", "state", tank=4, component="S_NH"),
    SensorChannel("S_NO_tank2", "state", tank=1, component="S_NO"),
    SensorChannel("S_NO_tank5", "state", tank=4, component="S_NO"),
    SensorChannel("TSS_tank5", "tss_reactor", tank=4),
    SensorChannel("TSS_ras", "tss_underflow"),
)


def observed_components() -> tuple[str, ...]:
    """Components that appear in at least one direct state measurement."""
    return tuple(sorted({c.component for c in SENSOR_SET if c.component}))


def unobserved_components() -> tuple[str, ...]:
    """Components never directly measured - the soft-sensor targets (Track B)."""
    measured = set(observed_components())
    return tuple(name for name in vault().components if name not in measured)


@dataclass
class ObservationDataset:
    """Everything a model is allowed to see, plus the hidden ground truth."""

    t: np.ndarray             # (n,) days
    obs_clean: np.ndarray     # (n, 8) noise-free sensor values
    obs: np.ndarray           # (n, 8) noisy sensor values fed to the models
    q_in: np.ndarray          # (n,) known
    z_in: np.ndarray          # (n, 14) known
    truth_reactor: np.ndarray  # (n, 5, 14) ground truth - EVALUATION ONLY
    truth_y: np.ndarray       # (n, state_size) full ODE state - EVALUATION ONLY
    channels: tuple[str, ...]
    sigma: float
    seed: int
    clip_fraction: float
    meta: dict[str, Any]

    # -- persistence -------------------------------------------------------
    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            meta=json.dumps(self.meta),
            channels=json.dumps(list(self.channels)),
            scalars=json.dumps(
                {"sigma": self.sigma, "seed": self.seed, "clip_fraction": self.clip_fraction}
            ),
            t=self.t,
            obs_clean=self.obs_clean,
            obs=self.obs,
            q_in=self.q_in,
            z_in=self.z_in,
            truth_reactor=self.truth_reactor,
            truth_y=self.truth_y,
        )
        return path

    @classmethod
    def load(cls, path: Path | str) -> "ObservationDataset":
        with np.load(Path(path), allow_pickle=False) as data:
            scalars = json.loads(str(data["scalars"]))
            return cls(
                t=data["t"],
                obs_clean=data["obs_clean"],
                obs=data["obs"],
                q_in=data["q_in"],
                z_in=data["z_in"],
                truth_reactor=data["truth_reactor"],
                truth_y=data["truth_y"],
                channels=tuple(json.loads(str(data["channels"]))),
                sigma=float(scalars["sigma"]),
                seed=int(scalars["seed"]),
                clip_fraction=float(scalars["clip_fraction"]),
                meta=json.loads(str(data["meta"])),
            )

    # -- views -------------------------------------------------------------
    def window(self, t_start: float, t_end: float) -> "ObservationDataset":
        """Time slice, used by the curriculum horizon schedule and by holdout."""
        mask = (self.t >= t_start) & (self.t <= t_end)
        return ObservationDataset(
            t=self.t[mask],
            obs_clean=self.obs_clean[mask],
            obs=self.obs[mask],
            q_in=self.q_in[mask],
            z_in=self.z_in[mask],
            truth_reactor=self.truth_reactor[mask],
            truth_y=self.truth_y[mask],
            channels=self.channels,
            sigma=self.sigma,
            seed=self.seed,
            clip_fraction=self.clip_fraction,
            meta={**self.meta, "window": [float(t_start), float(t_end)]},
        )


class SensorModel:
    """Maps a ground-truth trajectory onto the eight measured channels."""

    def __init__(self, channels: Sequence[SensorChannel] = SENSOR_SET) -> None:
        self.channels = tuple(channels)
        self.vault = vault()

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.channels)

    def observe(self, result: SimulationResult) -> np.ndarray:
        """Noise-free sensor readings, shape ``(n, n_channels)``."""
        columns = []
        for channel in self.channels:
            if channel.kind == "state":
                i = self.vault.index(str(channel.component))
                columns.append(result.reactor[:, int(channel.tank), i])
            elif channel.kind == "tss_reactor":
                columns.append(result.tss_reactor[:, int(channel.tank)])
            elif channel.kind == "tss_underflow":
                columns.append(result.tss_underflow)
            else:  # pragma: no cover - guarded by the dataclass
                raise ValueError("Unknown sensor kind %r" % (channel.kind,))
        return np.stack(columns, axis=-1)

    def add_noise(
        self, clean: np.ndarray, sigma: float, rng: np.random.Generator
    ) -> tuple[np.ndarray, float]:
        """Multiplicative Gaussian noise, clipped at zero. Returns (values, clip fraction)."""
        if sigma == 0.0:
            return clean.copy(), 0.0
        eps = rng.normal(loc=0.0, scale=sigma, size=clean.shape)
        noisy = clean * (1.0 + eps)
        clipped = noisy < 0.0
        return np.maximum(noisy, 0.0), float(np.mean(clipped))

    def build(
        self, result: SimulationResult, sigma: float, seed: int = 0
    ) -> ObservationDataset:
        clean = self.observe(result)
        rng = np.random.default_rng(seed)
        noisy, clip_fraction = self.add_noise(clean, sigma, rng)
        return ObservationDataset(
            t=result.t,
            obs_clean=clean,
            obs=noisy,
            q_in=result.q_in,
            z_in=result.influent,
            truth_reactor=result.reactor,
            truth_y=result.y,
            channels=self.names,
            sigma=float(sigma),
            seed=int(seed),
            clip_fraction=clip_fraction,
            meta={
                **result.meta,
                "observed_components": list(observed_components()),
                "unobserved_components": list(unobserved_components()),
                "channels": [asdict(c) for c in self.channels],
                "noise_model": "multiplicative gaussian, z*(1+eps), eps~N(0,sigma^2), clipped at 0",
            },
        )

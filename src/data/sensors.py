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

NOISE_LEVELS: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15)


@dataclass(frozen=True)
class SensorChannel:
    """One measured signal.

    ``kind == "linear"`` measures ``sum(w * Z[tank, component])`` over
    ``weights``; it models lumped probes such as UV-Vis soluble COD.
    ``sigma_override`` replaces the dataset noise level for this channel when
    that level is non-zero (a noise-free dataset stays noise-free).
    """

    name: str
    kind: str
    tank: int | None = None
    component: str | None = None
    weights: tuple[tuple[str, float], ...] | None = None
    sigma_override: float | None = None

    @property
    def label(self) -> str:
        return self.name


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


CANDIDATE_CHANNELS: tuple[SensorChannel, ...] = (
    SensorChannel("S_NH_tank1", "state", tank=0, component="S_NH"),
    SensorChannel("S_NH_tank2", "state", tank=1, component="S_NH"),
    SensorChannel("S_NH_tank3", "state", tank=2, component="S_NH"),
    SensorChannel("S_NH_tank4", "state", tank=3, component="S_NH"),
    SensorChannel("S_NO_tank1", "state", tank=0, component="S_NO"),
    SensorChannel("S_NO_tank3", "state", tank=2, component="S_NO"),
    SensorChannel("S_NO_tank4", "state", tank=3, component="S_NO"),
    SensorChannel("TSS_tank1", "tss_reactor", tank=0),
    SensorChannel(
        "SCOD_tank1", "linear", tank=0,
        weights=(("S_I", 1.0), ("S_S", 1.0)), sigma_override=0.20,
    ),
    SensorChannel(
        "SCOD_tank5", "linear", tank=4,
        weights=(("S_I", 1.0), ("S_S", 1.0)), sigma_override=0.20,
    ),
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

    t: np.ndarray
    obs_clean: np.ndarray
    obs: np.ndarray
    q_in: np.ndarray
    z_in: np.ndarray
    truth_reactor: np.ndarray
    truth_y: np.ndarray
    channels: tuple[str, ...]
    sigma: float
    seed: int
    clip_fraction: float
    meta: dict[str, Any]

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
        return self._observe_channels(result, self.channels)

    def _observe_channels(
        self, result: SimulationResult, channels: Sequence[SensorChannel]
    ) -> np.ndarray:
        columns = []
        for channel in channels:
            if channel.kind == "linear":
                if not channel.weights:
                    raise ValueError("Linear channel %r has no weights" % (channel.name,))
                idx = [self.vault.index(name) for name, _ in channel.weights]
                w = np.array([float(x) for _, x in channel.weights])
                columns.append(result.reactor[:, int(channel.tank)][:, idx] @ w)
            elif channel.kind == "state":
                i = self.vault.index(str(channel.component))
                columns.append(result.reactor[:, int(channel.tank), i])
            elif channel.kind == "tss_reactor":
                columns.append(result.tss_reactor[:, int(channel.tank)])
            elif channel.kind == "tss_underflow":
                columns.append(result.tss_underflow)
            else:
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

    @staticmethod
    def add_noise_per_channel(
        clean: np.ndarray, sigmas: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, float]:
        """Same noise model as :meth:`add_noise`, one sigma per column."""
        sigmas = np.asarray(sigmas, dtype=float)
        if clean.shape[1] != sigmas.shape[0]:
            raise ValueError("one sigma per column is required")
        if not np.any(sigmas > 0.0):
            return clean.copy(), 0.0
        eps = rng.normal(loc=0.0, scale=1.0, size=clean.shape) * sigmas[None, :]
        noisy = clean * (1.0 + eps)
        clipped = noisy < 0.0
        return np.maximum(noisy, 0.0), float(np.mean(clipped))

    @staticmethod
    def channel_sigma(channel: SensorChannel, sigma: float) -> float:
        """Noise level of one channel in a dataset built at ``sigma``."""
        if sigma == 0.0:
            return 0.0
        if channel.sigma_override is not None:
            return float(channel.sigma_override)
        return float(sigma)

    def build(
        self,
        result: SimulationResult,
        sigma: float,
        seed: int = 0,
        extra_channels: Sequence[SensorChannel] = (),
    ) -> ObservationDataset:
        """Noisy dataset. Standard columns are drawn exactly as in v1.0.

        The standard channels use ``default_rng(seed)`` with shape ``(n, 8)``,
        so adding ``extra_channels`` never changes them. Extra channels use the
        independent stream ``default_rng([seed, 1])`` and are appended after
        the standard columns.
        """
        clean = self.observe(result)
        rng = np.random.default_rng(seed)
        noisy, clip_fraction = self.add_noise(clean, sigma, rng)
        names = self.names
        channel_meta = [asdict(c) for c in self.channels]
        extra_meta: dict[str, Any] = {}
        extras = tuple(extra_channels)
        if extras:
            clash = sorted(set(c.name for c in extras) & set(names))
            if clash:
                raise ValueError("Extra channels duplicate standard channels: %s" % clash)
            extra_clean = self._observe_channels(result, extras)
            sigmas = np.array([self.channel_sigma(c, sigma) for c in extras])
            extra_noisy, extra_clip = self.add_noise_per_channel(
                extra_clean, sigmas, np.random.default_rng([int(seed), 1])
            )
            clean = np.concatenate([clean, extra_clean], axis=1)
            noisy = np.concatenate([noisy, extra_noisy], axis=1)
            names = names + tuple(c.name for c in extras)
            channel_meta += [asdict(c) for c in extras]
            extra_meta = {
                "extra_channels": [c.name for c in extras],
                "extra_sigmas": sigmas.tolist(),
                "extra_clip_fraction": extra_clip,
                "extra_noise_seed": [int(seed), 1],
            }
        return ObservationDataset(
            t=result.t,
            obs_clean=clean,
            obs=noisy,
            q_in=result.q_in,
            z_in=result.influent,
            truth_reactor=result.reactor,
            truth_y=result.y,
            channels=names,
            sigma=float(sigma),
            seed=int(seed),
            clip_fraction=clip_fraction,
            meta={
                **result.meta,
                "observed_components": list(observed_components()),
                "unobserved_components": list(unobserved_components()),
                "channels": channel_meta,
                "noise_model": "multiplicative gaussian, z*(1+eps), eps~N(0,sigma^2), clipped at 0",
                **extra_meta,
            },
        )

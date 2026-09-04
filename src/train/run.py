"""One benchmark run: (model, noise level) -> checkpoint, history, metrics.

A run is fully described by its YAML config. The four benchmark models differ in
exactly two flags::

    cl_pinn  arch=pinn  curriculum=hierarchical
    pinn     arch=pinn  curriculum=none
    cl_lstm  arch=lstm  curriculum=hierarchical
    lstm     arch=lstm  curriculum=none

Everything else - features, output head, initial condition, data loss, optimiser,
step budget, seed - is shared, so the four-way comparison isolates the physics
term and the curriculum rather than incidental differences.

Leakage discipline
------------------
The only ground truth that reaches the optimiser is: the seven measured target
channels, the measured ``TSS_ras`` input, the known influent, and the initial
state ``Z(0)``. ``ObservationDataset.truth_reactor`` is read exclusively inside
``torch.no_grad()`` evaluation blocks. ``tests/test_leakage.py`` asserts this.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..asm1.plant import Bsm1Plant
from ..data.sensors import SENSOR_SET, ObservationDataset, unobserved_components
from ..models.losses import Asm1Loss, LossWeights, ObservationOperator
from ..models.lstm import Asm1Lstm, LstmConfig
from ..models.pinn import Asm1Pinn, PinnConfig, component_scale
from . import curriculum as cl

#: Channels that are prediction targets. ``TSS_ras`` is a settler measurement
#: used as an input to the recycle reconstruction, not a target.
TARGET_CHANNELS = tuple(c for c in SENSOR_SET if c.kind != "tss_underflow")
RAS_CHANNEL = "TSS_ras"

MODEL_SPECS: dict[str, dict[str, str]] = {
    "cl_pinn": {"arch": "pinn", "curriculum": "hierarchical"},
    "pinn": {"arch": "pinn", "curriculum": "none"},
    "cl_lstm": {"arch": "lstm", "curriculum": "hierarchical"},
    "lstm": {"arch": "lstm", "curriculum": "none"},
    # Optional fifth configuration, not part of the 16-run benchmark. It
    # separates "physics" from "architecture", which the four models above
    # deliberately bundle together. Enable it explicitly if wanted.
    "pinn_nophysics": {"arch": "pinn", "curriculum": "none"},
    # Single-axis curriculum ablations (revision experiments): each keeps one
    # axis of the hierarchical schedule and holds the rest at the single-stage
    # settings, under the same total step budget.
    "cl_pinn_wonly": {"arch": "pinn", "curriculum": "weights_only"},
    "cl_pinn_honly": {"arch": "pinn", "curriculum": "horizon_only"},
    "cl_pinn_sonly": {"arch": "pinn", "curriculum": "scenario_only"},
    "cl_pinn_smonly": {"arch": "pinn", "curriculum": "smoothing_only"},
}


@dataclass
class RunConfig:
    run_id: str
    model: str
    noise: float
    seed: int = 0
    profile: str = "quick"
    data_dir: str = "results/raw"
    out_dir: str = "results/runs"
    train_end_day: float = 12.0
    holdout_days: tuple[float, float] = (12.0, 14.0)
    steps_quick: int = 4000
    steps_full: int = 20000
    collocation_points: int = 512
    lr: float = 1e-3
    lr_final_fraction: float = 0.02
    grad_clip: float = 1.0
    log_every: int = 100
    device: str = "cuda"
    dtype: str = "float32"
    # Restrict the initial-condition anchor to the directly sensed tank/component
    # entries instead of the full 5x14 state (revision ablation).
    ic_measured_only: bool = False
    pinn: dict[str, Any] = field(default_factory=dict)
    lstm: dict[str, Any] = field(default_factory=dict)

    @property
    def steps(self) -> int:
        return self.steps_quick if self.profile == "quick" else self.steps_full

    @property
    def arch(self) -> str:
        return MODEL_SPECS[self.model]["arch"]

    @property
    def curriculum(self) -> str:
        return MODEL_SPECS[self.model]["curriculum"]

    @classmethod
    def from_yaml(cls, path: Path | str) -> "RunConfig":
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        raw["holdout_days"] = tuple(raw.get("holdout_days", (12.0, 14.0)))
        return cls(**raw)


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def dataset_path(cfg: RunConfig, scenario: str) -> Path:
    sigma_tag = ("%.2f" % cfg.noise).replace(".", "p")
    return Path(cfg.data_dir) / ("obs_%s_sigma%s.npz" % (scenario, sigma_tag))


class Trainer:
    def __init__(self, cfg: RunConfig) -> None:
        self.cfg = cfg
        if cfg.model not in MODEL_SPECS:
            raise ValueError("Unknown model %r; expected one of %s" % (cfg.model, sorted(MODEL_SPECS)))
        self.device = resolve_device(cfg.device)
        self.dtype = getattr(torch, cfg.dtype)
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        self.plant = Bsm1Plant()
        self.vault = self.plant.vault
        self.n_tanks = self.plant.cfg.n_tanks
        self.n_components = self.plant.n_components

        self.data = {
            "dry": ObservationDataset.load(dataset_path(cfg, "dry")),
            "constant": ObservationDataset.load(dataset_path(cfg, "constant")),
        }
        self.train_set = self.data["dry"].window(0.0, cfg.train_end_day)

        self.channel_index = {name: i for i, name in enumerate(self.train_set.channels)}
        self.target_cols = [self.channel_index[c.name] for c in TARGET_CHANNELS]
        self.ras_col = self.channel_index[RAS_CHANNEL]

        # Initial condition: a supplied boundary condition for every model.
        self.z0 = self.train_set.truth_reactor[0].copy()
        self.state_scale = component_scale(self.z0)
        targets = self.train_set.obs[:, self.target_cols]
        self.target_scale = np.maximum(np.mean(np.abs(targets), axis=0), 1e-9)
        self.q_scale = float(np.mean(self.train_set.q_in))
        self.z_in_scale = np.maximum(np.mean(np.abs(self.train_set.z_in), axis=0), 1e-9)

        self.model = self._build_model()
        self.operator = ObservationOperator(self.plant, TARGET_CHANNELS)
        # With ic_measured_only the anchor covers only the directly sensed
        # (tank, component) entries; the default anchors the full known state.
        ic_mask = None
        if cfg.ic_measured_only:
            ic_mask = np.zeros((self.n_tanks, self.n_components))
            for channel in TARGET_CHANNELS:
                if channel.kind == "state":
                    ic_mask[int(channel.tank), self.vault.index(str(channel.component))] = 1.0
        self.loss = Asm1Loss(
            plant=self.plant,
            operator=self.operator,
            state_scale=self.state_scale,
            target_scale=self.target_scale,
            device=self.device,
            dtype=self.dtype,
            ic_mask=ic_mask,
        )
        self.schedule = cl.build(
            cfg.curriculum, cfg.steps, cfg.train_end_day, noisy=cfg.noise > 0.0
        )
        self._tensor_cache: dict[tuple[str, int, float], dict[str, torch.Tensor]] = {}
        self.history: list[dict[str, float]] = []

    # -- construction ------------------------------------------------------
    def _build_model(self) -> torch.nn.Module:
        from ..models.features import FeatureConfig

        common = dict(
            n_tanks=self.n_tanks,
            n_components=self.n_components,
            horizon_days=self.cfg.train_end_day,
            z0=self.z0,
            q_scale=self.q_scale,
            z_in_scale=self.z_in_scale,
        )

        def with_features(raw: dict[str, Any]) -> dict[str, Any]:
            """YAML delivers nested blocks as plain dicts; rebuild the dataclass."""
            out = dict(raw)
            if isinstance(out.get("features"), dict):
                feats = dict(out["features"])
                for key in ("periods", "harmonics"):
                    if key in feats:
                        feats[key] = tuple(feats[key])
                out["features"] = FeatureConfig(**feats)
            return out

        if self.cfg.arch == "pinn":
            model = Asm1Pinn(cfg=PinnConfig(**with_features(self.cfg.pinn)), **common)
        else:
            model = Asm1Lstm(cfg=LstmConfig(**with_features(self.cfg.lstm)), **common)
        return model.to(device=self.device, dtype=self.dtype)

    def _weights_for(self, weights: LossWeights) -> LossWeights:
        """Zero the physics-derived terms for the physics-free baselines."""
        if self.cfg.arch == "lstm" or self.cfg.model == "pinn_nophysics":
            return weights.scaled(physics=0.0, balance=0.0)
        return weights

    # -- tensor preparation ------------------------------------------------
    def _stage_tensors(self, stage: cl.CurriculumStage) -> dict[str, torch.Tensor]:
        key = (stage.dataset, stage.smoothing_window, stage.horizon_days)
        cached = self._tensor_cache.get(key)
        if cached is not None:
            return cached

        source = self.data[stage.dataset]
        if stage.dataset == "dry":
            source = source.window(0.0, min(stage.horizon_days, self.cfg.train_end_day))
        else:
            source = source.window(0.0, min(stage.horizon_days, float(source.t[-1])))

        obs = cl.smooth_observations(source.obs, stage.smoothing_window)

        def T(x, dtype=None):
            return torch.as_tensor(
                np.asarray(x), device=self.device, dtype=dtype or self.dtype
            )

        tensors = {
            "t": T(source.t).view(-1, 1),
            "q_in": T(source.q_in).view(-1, 1),
            "z_in": T(source.z_in),
            "targets": T(obs[:, self.target_cols]),
            "tss_ras": T(obs[:, self.ras_col]).view(-1, 1),
            "z0_true": T(source.truth_reactor[0]).unsqueeze(0),
            "t_max": T(float(source.t[-1])),
        }
        self._tensor_cache[key] = tensors
        return tensors

    def _collocation(self, stage: cl.CurriculumStage, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Random interior times, with influent interpolated from the known signals."""
        n = self.cfg.collocation_points
        t_grid = batch["t"].squeeze(-1)
        t_min, t_max = float(t_grid[0]), float(t_grid[-1])
        t = torch.rand(n, 1, device=self.device, dtype=self.dtype) * (t_max - t_min) + t_min
        t = t.requires_grad_(True)

        pos = torch.clamp(
            (t.squeeze(-1) - t_min) / max(t_max - t_min, 1e-12) * (len(t_grid) - 1),
            0,
            len(t_grid) - 1,
        )
        lo = pos.floor().long()
        hi = torch.clamp(lo + 1, max=len(t_grid) - 1)
        w = (pos - lo.to(pos.dtype)).unsqueeze(-1)

        def interp(x: torch.Tensor) -> torch.Tensor:
            return x[lo] * (1.0 - w) + x[hi] * w

        return {
            "t": t,
            "q_in": interp(batch["q_in"]),
            "z_in": interp(batch["z_in"]),
            "tss_ras": interp(batch["tss_ras"]),
        }

    # -- training ----------------------------------------------------------
    def train(self) -> dict[str, Any]:
        cfg = self.cfg
        optimiser = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=max(self.schedule.total_steps, 1), eta_min=cfg.lr * cfg.lr_final_fraction
        )
        started = time.perf_counter()

        for step, stage, weights in self.schedule.iterate():
            batch = self._stage_tensors(stage)
            weights = self._weights_for(weights)
            optimiser.zero_grad(set_to_none=True)

            z = self.model(batch["t"], batch["q_in"], batch["z_in"])
            # The grid starts at t = 0, so the first prediction is the initial
            # condition. Reusing it keeps the LSTM hidden state consistent with
            # the sequence it was rolled out on.
            z0_pred = z[:1]

            if self.cfg.arch == "pinn" and weights.physics > 0.0:
                colloc = self._collocation(stage, batch)
                z_c, dz_c = self.model.state_and_derivative(
                    colloc["t"], colloc["q_in"], colloc["z_in"]
                )
            else:
                colloc, z_c, dz_c = None, None, None

            parts = self.loss.total(
                weights=weights,
                t=batch["t"],
                z=z,
                dz_dt=None,
                q_in=batch["q_in"],
                z_in=batch["z_in"],
                tss_ras=batch["tss_ras"],
                targets=batch["targets"],
                z0_pred=z0_pred,
                z0_true=batch["z0_true"],
            )
            total = parts.total
            if z_c is not None:
                residual = self.loss.physics_residual(
                    z_c, dz_c, colloc["q_in"], colloc["z_in"], colloc["tss_ras"]
                )
                physics = torch.mean(residual ** 2)
                total = total + weights.physics * physics
                parts.physics = physics
                parts.total = total

            total.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            optimiser.step()
            scheduler.step()

            if step % cfg.log_every == 0 or step == self.schedule.total_steps - 1:
                record = {"step": step, "stage": stage.name, "lr": scheduler.get_last_lr()[0]}
                record.update(parts.detached())
                record.update({"w_%s" % k: v for k, v in weights.__dict__.items()})
                self.history.append(record)

        elapsed = time.perf_counter() - started
        return self.finalise(elapsed)

    # -- output ------------------------------------------------------------
    @torch.no_grad()
    def predict(self, dataset: ObservationDataset) -> np.ndarray:
        """Predicted reactor states ``(n, 5, 14)`` for an arbitrary dataset."""
        self.model.eval()

        def T(x):
            return torch.as_tensor(np.asarray(x), device=self.device, dtype=self.dtype)

        z = self.model(T(dataset.t).view(-1, 1), T(dataset.q_in).view(-1, 1), T(dataset.z_in))
        self.model.train()
        return z.detach().cpu().numpy()

    def finalise(self, elapsed: float) -> dict[str, Any]:
        out_dir = Path(self.cfg.out_dir) / self.cfg.run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        torch.save(
            {"state_dict": self.model.state_dict(), "config": asdict(self.cfg)},
            out_dir / "checkpoint.pt",
        )
        (out_dir / "history.json").write_text(
            json.dumps(self.history, indent=2), encoding="utf-8"
        )

        predictions = {
            "train": self.predict(self.train_set),
            "holdout": self.predict(self.data["dry"].window(*self.cfg.holdout_days)),
        }
        rain_path = dataset_path(self.cfg, "rain")
        if rain_path.exists():
            predictions["rain"] = self.predict(ObservationDataset.load(rain_path))
        np.savez_compressed(out_dir / "predictions.npz", **predictions)

        summary = {
            "run_id": self.cfg.run_id,
            "model": self.cfg.model,
            "arch": self.cfg.arch,
            "curriculum": self.cfg.curriculum,
            "noise": self.cfg.noise,
            "seed": self.cfg.seed,
            "ic_measured_only": self.cfg.ic_measured_only,
            "profile": self.cfg.profile,
            "train_end_day": self.cfg.train_end_day,
            "holdout_days": list(self.cfg.holdout_days),
            "steps": self.schedule.total_steps,
            "schedule": self.schedule.describe(),
            "device": str(self.device),
            "dtype": self.cfg.dtype,
            "n_parameters": int(sum(p.numel() for p in self.model.parameters())),
            "train_seconds": elapsed,
            "peak_gpu_bytes": (
                int(torch.cuda.max_memory_allocated(self.device))
                if self.device.type == "cuda"
                else None
            ),
            "final_losses": self.history[-1] if self.history else {},
            "unobserved_components": list(unobserved_components()),
            "vault_json_sha256": self.vault.json_sha256,
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary


def main(config_path: str) -> dict[str, Any]:
    cfg = RunConfig.from_yaml(config_path)
    return Trainer(cfg).train()


if __name__ == "__main__":  # pragma: no cover
    import sys

    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m src.train.run <config.yaml>")
    print(json.dumps(main(sys.argv[1]), indent=2))

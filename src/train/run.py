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
The only ground truth that reaches the optimiser is: the measured target
channels, the measured ``TSS_ras`` input, the known influent (or the view of it
selected by ``influent_mode``), and the initial state ``Z(0)``.
``ObservationDataset.truth_reactor`` is read exclusively inside
``torch.no_grad()`` evaluation blocks. ``tests/test_leakage.py`` asserts this.

v1.1 regime runs (``anchor_file`` set) never read ``truth_reactor`` at all: the
initial state is the anchor mean ``z0_mean`` for the dry stages and the nominal
steady state ``nominal_ss`` for the constant-load stage, whose data the nominal
plant generated. The estimator plant is always the audited vault, whatever
truth plant produced the data; ``Trainer.__init__`` checks this.
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
from ..asm1.vault_loader import vault
from ..data.influent_views import view_dataset
from ..data.sensors import (
    CANDIDATE_CHANNELS,
    SENSOR_SET,
    ObservationDataset,
    SensorChannel,
    unobserved_components,
)
from ..models.losses import (
    Asm1Loss,
    KineticAdapter,
    LossParts,
    LossWeights,
    ObservationOperator,
    kinetic_parameter_names,
)
from ..models.lstm import Asm1Lstm, LstmConfig
from ..models.pinn import Asm1Pinn, PinnConfig, component_scale
from ..observers.anchors import ic_weights_from_rel_std
from . import curriculum as cl

TARGET_CHANNELS = tuple(c for c in SENSOR_SET if c.kind != "tss_underflow")
RAS_CHANNEL = "TSS_ras"

MODEL_SPECS: dict[str, dict[str, str]] = {
    "cl_pinn": {"arch": "pinn", "curriculum": "hierarchical"},
    "pinn": {"arch": "pinn", "curriculum": "none"},
    "cl_lstm": {"arch": "lstm", "curriculum": "hierarchical"},
    "lstm": {"arch": "lstm", "curriculum": "none"},
    "pinn_nophysics": {"arch": "pinn", "curriculum": "none"},
    "cl_pinn_wonly": {"arch": "pinn", "curriculum": "weights_only"},
    "cl_pinn_honly": {"arch": "pinn", "curriculum": "horizon_only"},
    "cl_pinn_sonly": {"arch": "pinn", "curriculum": "scenario_only"},
    "cl_pinn_smonly": {"arch": "pinn", "curriculum": "smoothing_only"},
    "cl_pinn_theta": {"arch": "pinn", "curriculum": "hierarchical"},
}

KNOWN_CHANNELS: dict[str, SensorChannel] = {
    c.name: c for c in (*SENSOR_SET, *CANDIDATE_CHANNELS)
}
ANCHOR_KEYS: tuple[str, ...] = ("z0_mean", "z0_rel_std", "nominal_ss")
INFLUENT_MODES: tuple[str, ...] = ("exact", "composite", "composite_biased")


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
    ic_measured_only: bool = False
    total_derivative: bool = False
    ras_filter_window: int = 1
    variant: str = ""
    anchor_file: str | None = None
    influent_mode: str = "exact"
    target_channels: list[str] | None = None
    trainable_kinetics: list[str] = field(default_factory=list)
    kinetic_prior_sigma: float = 0.693
    kinetic_prior_weight: float = 1e-3
    kinetic_bound: float = 4.0
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


def resolve_target_channels(
    names: list[str] | None, available: tuple[str, ...]
) -> tuple[SensorChannel, ...]:
    """Target channels by name; ``None`` keeps the v1.0 set."""
    if names is None:
        return TARGET_CHANNELS
    names = list(names)
    if not names:
        raise ValueError("target_channels is empty; use null for the default set")
    if len(set(names)) != len(names):
        raise ValueError("target_channels has duplicates: %s" % (names,))
    out: list[SensorChannel] = []
    for name in names:
        if name == RAS_CHANNEL:
            raise ValueError(
                "%s is the measured input of the recycle reconstruction and can "
                "never be a target" % RAS_CHANNEL
            )
        channel = KNOWN_CHANNELS.get(name)
        if channel is None:
            raise ValueError(
                "Unknown target channel %r; expected one of %s" % (name, sorted(KNOWN_CHANNELS))
            )
        if channel.kind == "tss_underflow":
            raise ValueError("Channel %r is a measured input, not a target" % (name,))
        if name not in available:
            raise ValueError(
                "Target channel %r is not in the dataset (channels %s); generate the "
                "data with --candidate-channels" % (name, list(available))
            )
        out.append(channel)
    return tuple(out)


@dataclass(frozen=True)
class Anchor:
    """Initial-state knowledge from scripts/make_anchors.py (group G3)."""

    path: str
    name: str
    z0_mean: np.ndarray
    z0_rel_std: np.ndarray
    nominal_ss: np.ndarray
    meta: dict[str, Any]


def load_anchor(path: str, shape: tuple[int, int]) -> Anchor:
    """Read and validate an anchor file; the Trainer's only source of Z(0) when set."""
    with np.load(Path(path), allow_pickle=False) as data:
        missing = [key for key in ANCHOR_KEYS if key not in data.files]
        if missing:
            raise ValueError("Anchor file %s lacks %s" % (path, missing))
        arrays = {key: np.array(data[key], dtype=float) for key in ANCHOR_KEYS}
        meta = json.loads(str(data["meta"])) if "meta" in data.files else {}
    for key, value in arrays.items():
        if value.shape != shape:
            raise ValueError("Anchor %s: %s has shape %s, expected %s" % (path, key, value.shape, shape))
        if not np.isfinite(value).all():
            raise ValueError("Anchor %s: %s is not finite" % (path, key))
    if (arrays["z0_mean"] < 0.0).any() or (arrays["nominal_ss"] < 0.0).any():
        raise ValueError("Anchor %s: negative concentrations" % (path,))
    if (arrays["z0_rel_std"] <= 0.0).any():
        raise ValueError("Anchor %s: z0_rel_std must be positive" % (path,))
    return Anchor(
        path=str(path),
        name=str(meta.get("name", Path(path).stem)),
        z0_mean=arrays["z0_mean"],
        z0_rel_std=arrays["z0_rel_std"],
        nominal_ss=arrays["nominal_ss"],
        meta=meta,
    )


def data_provenance(
    dry_meta: dict[str, Any], constant_meta: dict[str, Any], anchor_file: str | None
) -> dict[str, Any]:
    """Truth-plant labels of the data; refuses mismatch data without an anchor.

    v1.0 datasets carry no ``truth_preset`` and are the vault 20 C plant. For any
    other truth plant, stage 1 must come from the nominal plant (``constant_from
    == "nominal"``) and the initial state must come from an anchor file, so the
    Trainer never reads a truth-plant state.
    """
    nominal = dict(vault().parameters)
    truth_preset = str(dry_meta.get("truth_preset", "vault20"))
    info = {
        "truth_preset": truth_preset,
        "alpha": dry_meta.get("alpha", 0.0),
        "constant_from": str(constant_meta.get("constant_from", "truth")),
    }
    for label, meta in (("dry", dry_meta), ("constant", constant_meta)):
        preset = str(meta.get("truth_preset", "vault20"))
        if preset == "vault20" and "parameters" in meta and dict(meta["parameters"]) != nominal:
            raise ValueError(
                "The %s dataset says truth_preset 'vault20' but its parameters differ "
                "from the vault; the data directory is mislabelled" % label
            )
    if truth_preset != "vault20":
        if info["constant_from"] != "nominal":
            raise ValueError(
                "Mismatch data (truth_preset %r) needs a constant scenario generated by "
                "the nominal plant (constant_from 'nominal'), got %r"
                % (truth_preset, info["constant_from"])
            )
        if not anchor_file:
            raise ValueError(
                "Mismatch data (truth_preset %r) needs anchor_file: without it the "
                "Trainer would take Z(0) from the truth plant" % (truth_preset,)
            )
    return info


def require_nominal_plant(plant: Bsm1Plant) -> None:
    """The estimator always uses the audited vault, whatever plant made the data."""
    nominal = vault()
    same = (
        dict(plant.vault.parameters) == dict(nominal.parameters)
        and np.array_equal(plant.vault.nu, nominal.nu)
        and np.array_equal(plant.vault.composition, nominal.composition)
    )
    if not same:
        raise RuntimeError(
            "Trainer plant does not use the audited vault parameters, nu and "
            "composition; truth-plant kinetics must never reach an estimator"
        )


class Trainer:
    def __init__(self, cfg: RunConfig) -> None:
        self.cfg = cfg
        if cfg.model not in MODEL_SPECS:
            raise ValueError("Unknown model %r; expected one of %s" % (cfg.model, sorted(MODEL_SPECS)))
        if cfg.influent_mode not in INFLUENT_MODES:
            raise ValueError(
                "Unknown influent_mode %r; expected one of %s" % (cfg.influent_mode, INFLUENT_MODES)
            )
        if (cfg.model == "cl_pinn_theta") != bool(cfg.trainable_kinetics):
            raise ValueError(
                "trainable_kinetics must be non-empty for cl_pinn_theta and empty for "
                "every other model (model %r, trainable_kinetics %s)"
                % (cfg.model, list(cfg.trainable_kinetics))
            )
        self.device = resolve_device(cfg.device)
        self.dtype = getattr(torch, cfg.dtype)
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        self.plant = Bsm1Plant()
        require_nominal_plant(self.plant)
        self.vault = self.plant.vault
        self.n_tanks = self.plant.cfg.n_tanks
        self.n_components = self.plant.n_components

        loaded = {
            "dry": ObservationDataset.load(dataset_path(cfg, "dry")),
            "constant": ObservationDataset.load(dataset_path(cfg, "constant")),
        }
        self.provenance = data_provenance(
            loaded["dry"].meta, loaded["constant"].meta, cfg.anchor_file
        )
        self.data = {key: view_dataset(ds, cfg.influent_mode) for key, ds in loaded.items()}
        self.train_set = self.data["dry"].window(0.0, cfg.train_end_day)

        self.channel_index = {name: i for i, name in enumerate(self.train_set.channels)}
        self.target_channels = resolve_target_channels(cfg.target_channels, self.train_set.channels)
        self.target_cols = [self.channel_index[c.name] for c in self.target_channels]
        self.ras_col = self.channel_index[RAS_CHANNEL]

        self.anchor: Anchor | None = None
        self.nominal_ss: np.ndarray | None = None
        if cfg.anchor_file:
            self.anchor = load_anchor(cfg.anchor_file, (self.n_tanks, self.n_components))
            self.z0 = self.anchor.z0_mean.copy()
            self.nominal_ss = self.anchor.nominal_ss.copy()
        else:
            self.z0 = self.train_set.truth_reactor[0].copy()
        self.state_scale = component_scale(self.z0)
        targets = self.train_set.obs[:, self.target_cols]
        self.target_scale = np.maximum(np.mean(np.abs(targets), axis=0), 1e-9)
        self.q_scale = float(np.mean(self.train_set.q_in))
        self.z_in_scale = np.maximum(np.mean(np.abs(self.train_set.z_in), axis=0), 1e-9)

        self.model = self._build_model()
        self.operator = ObservationOperator(self.plant, self.target_channels)
        ic_mask = None
        if cfg.ic_measured_only:
            ic_mask = np.zeros((self.n_tanks, self.n_components))
            for channel in self.target_channels:
                if channel.kind == "state":
                    ic_mask[int(channel.tank), self.vault.index(str(channel.component))] = 1.0
        self.ic_weights: np.ndarray | None = None
        if self.anchor is not None:
            self.ic_weights = ic_weights_from_rel_std(self.anchor.z0_rel_std)
            if ic_mask is not None:
                self.ic_weights = self.ic_weights * ic_mask
                ic_mask = None
        self.loss = Asm1Loss(
            plant=self.plant,
            operator=self.operator,
            state_scale=self.state_scale,
            target_scale=self.target_scale,
            device=self.device,
            dtype=self.dtype,
            ic_mask=ic_mask,
            ic_weights=self.ic_weights,
        )
        self.adapter: KineticAdapter | None = None
        if cfg.trainable_kinetics:
            allowed = kinetic_parameter_names(self.vault)
            names = list(cfg.trainable_kinetics)
            unknown = [n for n in names if n not in allowed]
            if unknown or len(set(names)) != len(names):
                raise ValueError(
                    "trainable_kinetics %s must be distinct names from %s" % (names, list(allowed))
                )
            self.adapter = KineticAdapter(names, cfg.kinetic_bound).to(
                device=self.device, dtype=self.dtype
            )
        self.schedule = cl.build(
            cfg.curriculum, cfg.steps, cfg.train_end_day, noisy=cfg.noise > 0.0
        )
        self._tensor_cache: dict[tuple[str, int, float], dict[str, torch.Tensor]] = {}
        self.history: list[dict[str, float]] = []

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

        raw = source.obs
        if self.cfg.ras_filter_window > 1:
            raw = raw.copy()
            raw[:, self.ras_col] = cl.trailing_average(
                raw[:, self.ras_col], self.cfg.ras_filter_window
            )
        obs = cl.smooth_observations(raw, stage.smoothing_window)

        def T(x, dtype=None):
            return torch.as_tensor(
                np.asarray(x), device=self.device, dtype=dtype or self.dtype
            )

        if self.anchor is not None:
            z0_stage = self.nominal_ss if stage.dataset == "constant" else self.z0
        else:
            z0_stage = source.truth_reactor[0]

        tensors = {
            "t": T(source.t).view(-1, 1),
            "q_in": T(source.q_in).view(-1, 1),
            "z_in": T(source.z_in),
            "targets": T(obs[:, self.target_cols]),
            "tss_ras": T(obs[:, self.ras_col]).view(-1, 1),
            "z0_true": T(z0_stage).unsqueeze(0),
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

        span = (t_grid[hi] - t_grid[lo]).unsqueeze(-1)
        same = (hi == lo).unsqueeze(-1)
        safe = torch.where(same, torch.ones_like(span), span)

        def slope(x: torch.Tensor) -> torch.Tensor:
            return torch.where(same, torch.zeros_like(x[lo]), (x[hi] - x[lo]) / safe)

        return {
            "t": t,
            "q_in": interp(batch["q_in"]),
            "z_in": interp(batch["z_in"]),
            "tss_ras": interp(batch["tss_ras"]),
            "dq_dt": slope(batch["q_in"]),
            "dz_dt": slope(batch["z_in"]),
        }

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        """Network parameters, plus the kinetic log-multipliers for cl_pinn_theta."""
        params = list(self.model.parameters())
        if self.adapter is not None:
            params += list(self.adapter.parameters())
        return params

    def learned_multipliers(self) -> dict[str, float]:
        if self.adapter is None:
            return {}
        values = self.adapter.multipliers().detach().cpu().tolist()
        return {name: float(m) for name, m in zip(self.cfg.trainable_kinetics, values)}

    def step_loss(self, stage: cl.CurriculumStage, weights: LossWeights) -> LossParts:
        """All loss terms of one optimisation step; ``weights`` already filtered."""
        cfg = self.cfg
        batch = self._stage_tensors(stage)
        z = self.model(batch["t"], batch["q_in"], batch["z_in"])
        z0_pred = z[:1]

        if cfg.arch == "pinn" and weights.physics > 0.0:
            colloc = self._collocation(stage, batch)
            if cfg.total_derivative:
                z_c, dz_c = self.model.state_and_derivative(
                    colloc["t"], colloc["q_in"], colloc["z_in"],
                    dq_dt=colloc["dq_dt"], dz_dt=colloc["dz_dt"],
                )
            else:
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
            params = None if self.adapter is None else self.adapter.overrides(self.vault.parameters)
            residual = self.loss.physics_residual(
                z_c, dz_c, colloc["q_in"], colloc["z_in"], colloc["tss_ras"], params=params
            )
            physics = torch.mean(residual ** 2)
            total = total + weights.physics * physics
            parts.physics = physics
            parts.total = total
        if self.adapter is not None:
            prior = self.adapter.prior(cfg.kinetic_prior_sigma)
            total = total + cfg.kinetic_prior_weight * prior
            parts.kinetic_prior = prior
            parts.total = total
        return parts

    def train(self) -> dict[str, Any]:
        cfg = self.cfg
        trainable = self.trainable_parameters()
        optimiser = torch.optim.Adam(trainable, lr=cfg.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=max(self.schedule.total_steps, 1), eta_min=cfg.lr * cfg.lr_final_fraction
        )
        started = time.perf_counter()

        for step, stage, weights in self.schedule.iterate():
            weights = self._weights_for(weights)
            optimiser.zero_grad(set_to_none=True)
            parts = self.step_loss(stage, weights)

            parts.total.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
            optimiser.step()
            scheduler.step()

            if step % cfg.log_every == 0 or step == self.schedule.total_steps - 1:
                record = {"step": step, "stage": stage.name, "lr": scheduler.get_last_lr()[0]}
                record.update(parts.detached())
                record.update({"w_%s" % k: v for k, v in weights.__dict__.items()})
                record.update({"mult_%s" % k: v for k, v in self.learned_multipliers().items()})
                self.history.append(record)

        elapsed = time.perf_counter() - started
        return self.finalise(elapsed)

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
            {
                "state_dict": self.model.state_dict(),
                "kinetic_adapter": None if self.adapter is None else self.adapter.state_dict(),
                "config": asdict(self.cfg),
            },
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
            predictions["rain"] = self.predict(
                view_dataset(ObservationDataset.load(rain_path), self.cfg.influent_mode)
            )
        np.savez_compressed(out_dir / "predictions.npz", **predictions)

        summary = {
            "run_id": self.cfg.run_id,
            "model": self.cfg.model,
            "arch": self.cfg.arch,
            "curriculum": self.cfg.curriculum,
            "noise": self.cfg.noise,
            "seed": self.cfg.seed,
            "ic_measured_only": self.cfg.ic_measured_only,
            "total_derivative": self.cfg.total_derivative,
            "ras_filter_window": self.cfg.ras_filter_window,
            "variant": self.cfg.variant,
            "data_dir": self.cfg.data_dir,
            "truth_preset": self.provenance["truth_preset"],
            "alpha": self.provenance["alpha"],
            "constant_from": self.provenance["constant_from"],
            "anchor_file": self.cfg.anchor_file,
            "anchor": None if self.anchor is None else self.anchor.name,
            "influent_mode": self.cfg.influent_mode,
            "target_channels": [c.name for c in self.target_channels],
            "trainable_kinetics": list(self.cfg.trainable_kinetics),
            "learned_multipliers": self.learned_multipliers(),
            "profile": self.cfg.profile,
            "train_end_day": self.cfg.train_end_day,
            "holdout_days": list(self.cfg.holdout_days),
            "steps": self.schedule.total_steps,
            "schedule": self.schedule.describe(),
            "device": str(self.device),
            "dtype": self.cfg.dtype,
            "n_parameters": int(sum(p.numel() for p in self.trainable_parameters())),
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


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m src.train.run <config.yaml>")
    print(json.dumps(main(sys.argv[1]), indent=2))

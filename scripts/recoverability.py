"""Recoverability analysis (E3, CPU part): which never-measured ASM1 states can the
routine sensors recover, and what limits the rest?

The estimators' reduced model (vault kinetics, five tanks, RAS TSS as a measured
input) is linearised along the simulated trajectory of a data directory over the
dry-weather reconstruction window (days 0-12). Three model-derived indices are
computed per (tank, component) and combined into the pre-registered classes:

* information gain ``IG = 1 - sigma_post / sigma_prior`` of the initial state,
  from the Fisher information of the seven target channels and the anchor prior;
* start-up memory ``tau``: 1/e decay time of the tank-pooled self-sensitivity;
* influent forcing share: variance from a persistent 10 % error on each influent
  component against variance from the initial-state prior.

Modes
-----
Analysis (writes ``recoverability_<tag>.json`` and two figures per directory)::

    python -m scripts.recoverability --data-dir results/raw --data-dir results/raw_k100 \
        --sigma 0.10 --prior results/v11/anchors/k100/As.npz --out results/v11/analysis

Kinetic subset for PINN-theta and the augmented observers (nominal model along
the nominal trajectory only; refuses a truth-plant directory)::

    python -m scripts.recoverability --select-kinetics --data-dir results/raw \
        --sigma 0.10 --prior results/v11/anchors/k000/As.npz --out results/v11/analysis

Sensor confirmation choice from the K1-Ie analysis::

    python -m scripts.recoverability \
        --sensor-confirmation results/v11/analysis/recoverability_k100.json \
        --out results/v11/analysis

This script is outside the static leakage scan on purpose: an observability
analysis linearises along the true trajectory. Nothing it writes feeds a training
path except the two pre-registered choices (kinetic subset, sensor variants),
which are fixed before any full-profile run.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, NamedTuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.asm1.plant import TSS_COMPONENTS, Bsm1Plant, constant_influent
from src.asm1.vault_loader import vault
from src.data.influent import stabilisation_influent
from src.data.influent_views import apply_influent_knowledge
from src.data.sensors import CANDIDATE_CHANNELS, ObservationDataset, unobserved_components
from src.observers import anchors
from src.observers import sensitivity as sens
from src.observers.reduced_model import ReducedPlantModel
from src.train.run import RAS_CHANNEL, TARGET_CHANNELS

WINDOW_END_DAY = 12.0
ZIN_REL_SD = 0.10
REFERENCE_REL_SD = 0.5
KINETIC_CANDIDATES = (
    "kh", "KX", "etah", "muH", "etag", "Ks", "bH", "KO_H", "KNO", "KNH_H",
    "muA", "bA", "ka", "KO_A", "KNH",
)
LAB_REL_ERROR = {"TSS": 0.05, "COD": 0.05, "SCOD": 0.05, "TKN": 0.07, "STKN": 0.07, "ALK": 0.05}
RESPIROMETRY = (("X_B_H", 0.20), ("X_B_A", 0.25))
RESPIROMETRY_TANKS = (0, 4)
CLASS_COLOURS = {
    "sensor-recoverable": "#2a78d6",
    "anchor-carried": "#eb6834",
    "forcing-slaved": "#1baf7a",
    "partly recoverable": "#4a3aa7",
}
CLASS_INK = {
    "sensor-recoverable": "#ffffff",
    "anchor-carried": "#0b0b0b",
    "forcing-slaved": "#0b0b0b",
    "partly recoverable": "#ffffff",
}


class Case(NamedTuple):
    tag: str
    data_dir: str
    influent_mode: str
    t: np.ndarray
    z: np.ndarray
    inputs: sens.InputTrajectory
    y0: np.ndarray
    meta: dict[str, Any]


def sigma_tag(sigma: float) -> str:
    return ("%.2f" % sigma).replace(".", "p")


def tag_for(data_dir: str | Path) -> str:
    """``results/raw -> k000``, ``results/raw_k100 -> k100``, ``raw_rand/7 -> rand7``."""
    p = Path(data_dir)
    if p.parent.name == "raw_rand":
        return "rand%s" % p.name
    if p.name == "raw":
        return "k000"
    if p.name.startswith("raw_"):
        return p.name[4:]
    raise ValueError("cannot derive a tag from %s" % data_dir)


def state_labels(components) -> list[str]:
    return ["%s_t%d" % (c, k + 1) for k in range(sens.N_TANKS) for c in components]


def never_measured_mask(components) -> np.ndarray:
    mask = np.zeros((sens.N_TANKS, len(components)), dtype=bool)
    for name in unobserved_components():
        mask[:, list(components).index(name)] = True
    return mask


def load_case(data_dir: str | Path, sigma: float, influent_mode: str = "exact",
              t_end: float = WINDOW_END_DAY) -> Case:
    ds = ObservationDataset.load(Path(data_dir) / ("obs_dry_sigma%s.npz" % sigma_tag(sigma)))
    z_in = ds.z_in
    if influent_mode != "exact":
        z_in = apply_influent_knowledge(ds.t, ds.q_in, ds.z_in, influent_mode, vault())
    keep = ds.t <= t_end + 1e-9
    ras = ds.obs_clean[:, ds.channels.index(RAS_CHANNEL)]
    return Case(
        tag=tag_for(data_dir), data_dir=str(data_dir), influent_mode=influent_mode,
        t=ds.t[keep], z=np.clip(ds.truth_reactor[keep], 1e-12, None),
        inputs=sens.InputTrajectory(ds.q_in[keep], np.asarray(z_in, dtype=float)[keep], ras[keep]),
        y0=ds.truth_y[0].copy(), meta=dict(ds.meta),
    )


def load_prior(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Diagonal log-space prior ``log(1 + rel_std^2)`` from a G3 anchor file."""
    with np.load(Path(path), allow_pickle=False) as a:
        rel = np.asarray(a["z0_rel_std"], dtype=float).reshape(-1)
        meta = json.loads(str(a["meta"])) if "meta" in a.files else {}
    return np.log1p(rel ** 2), meta


def nominal_steady_state(nominal_dir: str | Path):
    """Nominal plant steady state on the Table 5 load: reactor z, reduced-model inputs, full y."""
    ds = ObservationDataset.load(Path(nominal_dir) / "obs_constant_sigma0p00.npz")
    if dict(ds.meta.get("parameters", {})) != dict(vault().parameters):
        raise ValueError("%s does not hold the nominal (vault) plant" % nominal_dir)
    ras = float(ds.obs_clean[0, ds.channels.index(RAS_CHANNEL)])
    return ds.truth_reactor[0].copy(), (float(ds.q_in[0]), ds.z_in[0].copy(), ras), ds.truth_y[0].copy()


def git_commit() -> str:
    out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return out.stdout.strip()


def clean(obj: Any) -> Any:
    """JSON-safe copy: arrays to lists, non-finite floats to None."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    return obj


def core_indices(case: Case, p0: np.ndarray, sigma: float, jac_model: Any,
                 substeps: int = sens.DEFAULT_SUBSTEPS) -> dict[str, Any]:
    comps = vault().components
    n = case.t.size
    x = np.log(case.z.reshape(n, -1))
    tl = sens.tangent_linear(jac_model, x, case.inputs, case.t, substeps=substeps,
                             inputs="zin_rel", input_names=comps, drift="trajectory")
    m = sens.cumulative_propagators(tl.phis)
    h = sens.log_measurement_rows(case.z, TARGET_CHANNELS, comps)
    r = sens.log_noise_variance(TARGET_CHANNELS, sigma)
    f_channels = {ch.name: sens.channel_fisher(None, h[:, j], r[j], cumulative=m)
                  for j, ch in enumerate(TARGET_CHANNELS)}
    f = sum(f_channels.values())
    p_post = sens.posterior_cov(f, p0)
    ig = sens.information_gain(p_post, p0)
    flat = case.z.reshape(-1, len(comps))
    ranges = np.maximum(flat.max(axis=0) - flat.min(axis=0), 1e-12)
    tau, extrap = sens.self_sensitivity_decay(tl.phis, case.t, cumulative=m)
    share = sens.forcing_share(tl, p0, ZIN_REL_SD, cumulative=m)["share"]
    return {
        "tl": tl, "m": m, "f_channels": f_channels, "f": f, "p_post": p_post, "ig": ig,
        "crb": sens.crb_over_range(p_post, m, case.z, ranges), "tau": tau,
        "tau_extrapolated": extrap, "share": share,
        "classes": sens.classify_states(ig, tau, share), "ranges": ranges,
    }


def state_rows(core: Mapping[str, Any], components, mask: np.ndarray) -> list[dict[str, Any]]:
    measured = {(int(c.tank), str(c.component)) for c in TARGET_CHANNELS if c.kind == "state"}
    rows = []
    for k in range(sens.N_TANKS):
        for c, name in enumerate(components):
            rows.append({
                "tank": k + 1, "component": name, "never_measured": bool(mask[k, c]),
                "directly_measured": (k, name) in measured,
                "ig": core["ig"][k, c], "crb_over_range": core["crb"][k, c],
                "tau_days": core["tau"][k, c], "tau_extrapolated": bool(core["tau_extrapolated"][k, c]),
                "forcing_share": core["share"][k, c], "class": str(core["classes"][k, c]),
            })
    return rows


def class_counts(classes: np.ndarray) -> dict[str, int]:
    counts = Counter(str(c) for c in np.ravel(classes))
    return {name: int(counts.get(name, 0)) for name in sens.CLASS_NAMES}


def mean_ig(f: np.ndarray, p0: np.ndarray, mask: np.ndarray) -> float:
    return float(sens.information_gain(sens.posterior_cov(f, p0), p0)[mask].mean())


def ig_by_component(f: np.ndarray, p0: np.ndarray) -> dict[str, float]:
    ig = sens.information_gain(sens.posterior_cov(f, p0), p0)
    comps = list(vault().components)
    return {name: float(ig[:, comps.index(name)].mean()) for name in unobserved_components()}


def sensor_tables(core: Mapping[str, Any], case: Case, p0: np.ndarray, sigma: float,
                  mask: np.ndarray) -> dict[str, Any]:
    """All 2^7 target-channel subsets, drop-one, and add-one candidates.

    ``J`` = mean information gain over the 55 never-measured (tank, component)
    entries. Fisher information is additive over channels, so every subset is a
    sum of stored matrices.
    """
    names = list(core["f_channels"])
    zero = np.zeros((sens.N_STATE, sens.N_STATE))
    j_full = mean_ig(core["f"], p0, mask)
    subsets = []
    for size in range(len(names) + 1):
        for combo in itertools.combinations(names, size):
            f = sum((core["f_channels"][c] for c in combo), zero)
            subsets.append({"channels": list(combo), "J": mean_ig(f, p0, mask),
                            "ig_by_component": ig_by_component(f, p0)})
    drop = [{"name": c, "delta_J": j_full - mean_ig(core["f"] - core["f_channels"][c], p0, mask)}
            for c in names]
    comps = vault().components
    h_c = sens.log_measurement_rows(case.z, CANDIDATE_CHANNELS, comps)
    r_c = sens.log_noise_variance(CANDIDATE_CHANNELS, sigma)
    add = []
    for j, ch in enumerate(CANDIDATE_CHANNELS):
        f_c = sens.channel_fisher(None, h_c[:, j], r_c[j], cumulative=core["m"])
        add.append({"name": ch.name, "sigma": float(np.sqrt(np.expm1(r_c[j]))),
                    "delta_J": mean_ig(core["f"] + f_c, p0, mask) - j_full,
                    "ig_by_component": ig_by_component(core["f"] + f_c, p0)})
    return {
        "criterion": "J = mean IG over never-measured (tank, component) entries",
        "J_full": j_full,
        "subsets": sorted(subsets, key=lambda row: -row["J"]),
        "drop_one": sorted(drop, key=lambda row: -row["delta_J"]),
        "add_one": sorted(add, key=lambda row: -row["delta_J"]),
    }


def assay_family(weights: np.ndarray, components) -> tuple[str, int]:
    """Identify a G3 lab operator by its support: (family, 0-based tank)."""
    comps = list(components)
    w = np.asarray(weights, dtype=float).reshape(sens.N_TANKS, len(comps))
    tanks = np.nonzero(np.abs(w).sum(axis=1) > 0.0)[0]
    if tanks.size != 1:
        raise ValueError("a lab operator must act on exactly one tank; found %s" % tanks.tolist())
    support = {comps[i] for i in np.nonzero(w[tanks[0]] != 0.0)[0]}
    if support == {"S_ALK"}:
        family = "ALK"
    elif support == set(TSS_COMPONENTS):
        family = "TSS"
    elif support == {"S_I", "S_S"}:
        family = "SCOD"
    elif support == {"S_NH", "S_ND"}:
        family = "STKN"
    elif {"S_NH", "X_ND"} <= support:
        family = "TKN"
    elif {"S_S", "X_S", "X_B_H"} <= support and "S_NH" not in support:
        family = "COD"
    else:
        raise ValueError("unrecognised lab operator support %s" % sorted(support))
    return family, int(tanks[0])


def lab_rows(z0: np.ndarray, operators: Mapping[str, np.ndarray], components):
    """Log-measurement rows of the panel at ``t = 0`` and their log-variances."""
    rows, r, labels, families = [], [], [], []
    for _key, w in operators.items():
        family, tank = assay_family(w, components)
        wz = np.asarray(w, dtype=float).reshape(-1) * np.asarray(z0, dtype=float).reshape(-1)
        rows.append(wz / wz.sum())
        r.append(np.log1p(LAB_REL_ERROR[family] ** 2))
        labels.append("%s_tank%d" % (family, tank + 1))
        families.append(family)
    return np.asarray(rows), np.asarray(r), labels, families


def respirometry_rows(components):
    comps = list(components)
    rows, r, labels = [], [], []
    for name, err in RESPIROMETRY:
        for tank in RESPIROMETRY_TANKS:
            h = np.zeros((sens.N_TANKS, len(comps)))
            h[tank, comps.index(name)] = 1.0
            rows.append(h.ravel())
            r.append(np.log1p(err ** 2))
            labels.append("%s_tank%d" % (name, tank + 1))
    return np.asarray(rows), np.asarray(r), labels


def lab_tables(core: Mapping[str, Any], case: Case, p0: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    """Value of the start-up panel (Al1) and respirometry (Al2) on top of the sensors (As)."""
    comps = vault().components
    h_lab, r_lab, labels, families = lab_rows(case.z[0], anchors.lab_operators(vault()), comps)
    h_resp, r_resp, resp_labels = respirometry_rows(comps)
    f_panel = sens.static_fisher(h_lab, r_lab)
    f_resp = sens.static_fisher(h_resp, r_resp)
    f_sens = core["f"]

    def summary(f: np.ndarray) -> dict[str, Any]:
        return {"J": mean_ig(f, p0, mask), "ig_by_component": ig_by_component(f, p0)}

    tiers = {
        "As": summary(f_sens),
        "Al1": summary(f_sens + f_panel),
        "Al2": summary(f_sens + f_panel + f_resp),
        "Al1_lab_only": summary(f_panel),
        "Al2_lab_only": summary(f_panel + f_resp),
    }
    j_as = tiers["As"]["J"]
    for name in ("Al1", "Al2"):
        tiers[name]["share_of_oracle"] = (tiers[name]["J"] - j_as) / max(1.0 - j_as, 1e-12)
    add, leave_out = [], []
    for family in sorted(set(families)):
        sel = [i for i, f_ in enumerate(families) if f_ == family]
        f_fam = sens.static_fisher(h_lab[sel], r_lab[sel])
        with_family = summary(f_sens + f_fam)
        add.append({"assay": family, "rows": [labels[i] for i in sel],
                    "delta_J": with_family["J"] - j_as, "ig_by_component": with_family["ig_by_component"]})
        leave_out.append({"assay": family,
                          "delta_J": tiers["Al1"]["J"] - summary(f_sens + f_panel - f_fam)["J"]})
    with_resp = summary(f_sens + f_resp)
    add.append({"assay": "respirometry", "rows": resp_labels,
                "delta_J": with_resp["J"] - j_as, "ig_by_component": with_resp["ig_by_component"]})
    return {
        "rel_errors": {**LAB_REL_ERROR, **{"respirometry_" + name: err for name, err in RESPIROMETRY}},
        "rows": labels + resp_labels, "tiers": tiers,
        "add_one_to_As": sorted(add, key=lambda row: -row["delta_J"]),
        "leave_one_out_of_Al1": sorted(leave_out, key=lambda row: -row["delta_J"]),
    }


def truth_plant(meta: Mapping[str, Any]) -> Bsm1Plant:
    """The plant that generated the data directory (scripts may build truth plants)."""
    preset = meta.get("truth_preset", "vault20")
    if preset == "vault20":
        return Bsm1Plant()
    from src.asm1.truth_plants import truth_vault

    return Bsm1Plant(source=truth_vault(preset, float(meta.get("alpha", 1.0))))


def full_plant_modes(plant: Bsm1Plant, y_ss: np.ndarray, n_modes: int = 6) -> dict[str, Any]:
    """Finite-difference slow modes of the full plant (settler included), three schemes."""
    q, z = stabilisation_influent()
    influent = constant_influent(q, z)

    def fun(y: np.ndarray) -> np.ndarray:
        return plant.rhs(0.0, y, influent)

    out: dict[str, Any] = {"steady_state_residual": float(np.linalg.norm(fun(y_ss)) / np.linalg.norm(y_ss))}
    for scheme in ("forward", "backward", "central"):
        jac = sens.finite_difference_jacobian(fun, y_ss, rel_step=1e-6, floor=1.0, scheme=scheme)
        out[scheme] = [mode["time_constant_days"] for mode in sens.modal_analysis(jac, n_modes)]
    return out


def class_map_figure(core: Mapping[str, Any], tag: str, out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    from src.eval.report import save_figure

    comps = list(unobserved_components())
    all_comps = list(vault().components)
    fig, ax = plt.subplots(figsize=(7.2, 2.9), constrained_layout=True)
    for k in range(sens.N_TANKS):
        for j, name in enumerate(comps):
            c = all_comps.index(name)
            cls = str(core["classes"][k, c])
            ax.add_patch(Rectangle((j + 0.03, k + 0.03), 0.94, 0.94, color=CLASS_COLOURS[cls], linewidth=0))
            ax.text(j + 0.5, k + 0.5, "%s\n%.2f" % (cls[0].upper(), core["ig"][k, c]),
                    ha="center", va="center", fontsize=6.5, color=CLASS_INK[cls])
    ax.set_xlim(0, len(comps))
    ax.set_ylim(sens.N_TANKS, 0)
    ax.set_xticks(np.arange(len(comps)) + 0.5, comps, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(np.arange(sens.N_TANKS) + 0.5, ["tank %d" % (k + 1) for k in range(sens.N_TANKS)], fontsize=7)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(handles=[Patch(color=CLASS_COLOURS[n], label=n) for n in sens.CLASS_NAMES],
              loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=4, fontsize=7, frameon=False)
    ax.set_title("%s: recoverability class (letter) and information gain (number)" % tag,
                 fontsize=8, loc="left", color="#0b0b0b")
    paths = save_figure(fig, out_dir / ("recoverability_%s.png" % tag))
    plt.close(fig)
    return paths


def memory_figure(core: Mapping[str, Any], ideal: Mapping[str, Any], tag: str, out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.eval.report import save_figure

    comps = list(unobserved_components())
    idx = [list(vault().components).index(c) for c in comps]
    xpos = np.arange(len(comps))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 2.6), constrained_layout=True)
    series = (
        (-width / 2 - 0.01, core["tau"][4, idx], "#2a78d6", "RAS TSS as measured input"),
        (width / 2 + 0.01, ideal["tau"][4, idx], "#eb6834", "ideal clarifier closure"),
    )
    for offset, values, colour, label in series:
        ax.bar(xpos + offset, np.where(np.isfinite(values), values, np.nan), width=width,
               color=colour, label=label, linewidth=0)
    threshold = sens.CLASS_RULES["tau_anchor_days"]
    ax.axhline(threshold, color="#52514e", linewidth=1.0, linestyle="--")
    ax.text(len(comps) - 0.5, threshold, "anchor-carried threshold", ha="right", va="bottom",
            fontsize=6.5, color="#52514e")
    finite = np.concatenate([v[np.isfinite(v)] for _, v, _, _ in series] + [np.array([threshold])])
    ax.set_ylim(0.0, 1.15 * float(finite.max()))
    ax.set_xticks(xpos, comps, rotation=45, ha="right", fontsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylabel("1/e memory time, tank 5 (d)", fontsize=7)
    ax.grid(axis="y", color="#e5e4df", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(fontsize=7, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2)
    paths = save_figure(fig, out_dir / ("memory_%s.png" % tag))
    plt.close(fig)
    return paths


def analyse(data_dir: str, prior_path: str, sigma: float, out_dir: Path, nominal_dir: str,
            influent_mode: str = "exact", substeps: int = sens.DEFAULT_SUBSTEPS,
            figures: bool = True) -> Path:
    start = time.perf_counter()
    comps = vault().components
    labels = state_labels(comps)
    mask = never_measured_mask(comps)
    case = load_case(data_dir, sigma, influent_mode)
    p0, prior_meta = load_prior(prior_path)
    model = ReducedPlantModel()
    core = core_indices(case, p0, sigma, model, substeps)
    ideal = core_indices(case, p0, sigma, sens.IdealSettlerModel(model, Bsm1Plant()), substeps)
    z_ss, u_ss, _y_ss = nominal_steady_state(nominal_dir)
    q5, z5 = stabilisation_influent()
    ev, evec = sens.fisher_eigen(core["f"], p0)
    u_sum, u_split = sens.sum_split_directions(case.z[0], comps)
    p_ref = np.full(sens.N_STATE, np.log1p(REFERENCE_REL_SD ** 2))
    p_ref_post = sens.posterior_cov(core["f"], p_ref)
    changes = [
        {"tank": k + 1, "component": comps[c], "ras_input": str(core["classes"][k, c]),
         "ideal_settler": str(ideal["classes"][k, c])}
        for k in range(sens.N_TANKS) for c in range(len(comps))
        if mask[k, c] and core["classes"][k, c] != ideal["classes"][k, c]
    ]
    result = {
        "tag": case.tag, "data_dir": case.data_dir, "scenario": "dry",
        "window_days": [float(case.t[0]), float(case.t[-1])], "sigma": sigma,
        "influent_mode": influent_mode, "zin_rel_sd": ZIN_REL_SD, "substeps": substeps,
        "prior": {"file": str(prior_path), "meta": prior_meta},
        "truth": {key: case.meta.get(key) for key in ("truth_preset", "alpha", "constant_from")},
        "model": "reduced reactor train, vault kinetics, RAS TSS measured input",
        "linearisation": "nominal-model Jacobians along the simulated trajectory of the data directory, days 0-12, dry weather; log-state drift from that trajectory",
        "class_rules": dict(sens.CLASS_RULES), "components": list(comps),
        "never_measured": list(unobserved_components()),
        "channels": [c.name for c in TARGET_CHANNELS],
        "states": state_rows(core, comps, mask),
        "class_counts_never_measured": class_counts(core["classes"][mask]),
        "J_never_measured": mean_ig(core["f"], p0, mask),
        "fisher": {
            "whitened_eigenvalues": ev,
            **sens.weak_directions(ev, evec, labels),
            "inert_sum_ig": sens.direction_information_gain(core["p_post"], p0, u_sum),
            "inert_split_ig": sens.direction_information_gain(core["p_post"], p0, u_split),
            "reference_prior_rel_sd": REFERENCE_REL_SD,
            "inert_sum_ig_reference": sens.direction_information_gain(p_ref_post, p_ref, u_sum),
            "inert_split_ig_reference": sens.direction_information_gain(p_ref_post, p_ref, u_split),
        },
        "sensor_value": sensor_tables(core, case, p0, sigma, mask),
        "lab_assays": lab_tables(core, case, p0, mask),
        "ras_vs_ideal_settler": {
            "ideal_states": state_rows(ideal, comps, mask),
            "ideal_class_counts_never_measured": class_counts(ideal["classes"][mask]),
            "ideal_J_never_measured": mean_ig(ideal["f"], p0, mask),
            "class_changes": changes,
        },
        "slow_modes": {
            "reduced_nominal_steady_state": sens.jacobian_slow_modes(model, z_ss, u_ss, labels=labels),
            "reduced_at_truth_t0_table5": sens.jacobian_slow_modes(
                model, case.z[0], (q5, z5, float(case.inputs.tss_ras[0])), labels=labels),
            "full_plant_truth": full_plant_modes(truth_plant(case.meta), case.y0),
        },
        "git_commit": git_commit(),
    }
    suffix = "" if influent_mode == "exact" else "_" + influent_mode
    if figures:
        result["figures"] = [str(p) for p in class_map_figure(core, case.tag + suffix, out_dir)
                             + memory_figure(core, ideal, case.tag + suffix, out_dir)]
    result["runtime_seconds"] = time.perf_counter() - start
    path = out_dir / ("recoverability_%s%s.json" % (case.tag, suffix))
    path.write_text(json.dumps(clean(result), indent=1), encoding="utf-8")
    return path


def select_kinetics(data_dir: str, prior_path: str, sigma: float, out_dir: Path,
                    substeps: int = sens.DEFAULT_SUBSTEPS, k: int = 4, max_ci: float = 20.0) -> Path:
    """D-optimal kinetic subset along the NOMINAL trajectory (never a truth plant)."""
    case = load_case(data_dir, sigma)
    v = vault()
    if case.meta.get("truth_preset", "vault20") != "vault20" or dict(case.meta.get("parameters", {})) != dict(v.parameters):
        raise SystemExit("--select-kinetics uses the nominal trajectory only; %s is a truth-plant directory" % data_dir)
    p0, prior_meta = load_prior(prior_path)
    model = ReducedPlantModel()
    params = {name: v.p(name) for name in KINETIC_CANDIDATES}
    n = case.t.size
    x = np.log(case.z.reshape(n, -1))
    h = sens.log_measurement_rows(case.z, TARGET_CHANNELS, v.components)
    r = sens.log_noise_variance(TARGET_CHANNELS, sigma)
    z_ss, u_ss, _ = nominal_steady_state(data_dir)
    modes: dict[str, Any] = {}
    for mode, x0_sens in (("steady_state", sens.steady_state_parameter_sensitivity(model, z_ss, u_ss, params)),
                          ("fixed", None)):
        ps = sens.parameter_sensitivity(model, x, case.inputs, case.t, params, h, r,
                                        substeps=substeps, x0_sensitivity=x0_sens,
                                        drift="trajectory")
        gram = ps["s_theta"].T @ ps["s_theta"]
        f_marg = sens.marginal_parameter_fisher(ps["s_theta"], ps["s_x0"], p0)
        sel = sens.d_optimal_subset(f_marg, k=k, max_ci=max_ci, names=KINETIC_CANDIDATES, ci_gram=gram)
        modes[mode] = {
            "best": sel["best"], "ranking": sel["ranking"],
            "n_admissible": sel["n_admissible"], "n_candidates": sel["n_candidates"],
            "importance_msqr": {name: float(np.sqrt(gram[i, i] / ps["s_theta"].shape[0]))
                                for i, name in enumerate(KINETIC_CANDIDATES)},
            "collinearity_index_all": sens.collinearity_index(ps["s_theta"]),
        }
    names = modes["steady_state"]["best"]["names"]
    result = {
        "names": names, "k": k, "max_ci": max_ci,
        "criterion": ("max log det of the parameter Fisher information marginalised over the "
                      "initial state (As prior), subject to Brun collinearity index < max_ci"),
        "primary_x0_mode": "steady_state",
        "agrees_with_fixed_x0": names == modes["fixed"]["best"]["names"],
        "candidates": list(KINETIC_CANDIDATES), "modes": modes,
        "data_dir": str(data_dir), "trajectory": "nominal (vault kinetics), dry, days 0-12",
        "prior": {"file": str(prior_path), "meta": prior_meta}, "sigma": sigma,
        "substeps": substeps, "git_commit": git_commit(),
    }
    path = out_dir / "kinetic_subset.json"
    path.write_text(json.dumps(clean(result), indent=1), encoding="utf-8")
    return path


def pick_sensor_confirmation(report: Mapping[str, Any]) -> dict[str, Any]:
    """Largest never-measured information loss (drop) and gain (add) at K1-Ie."""
    if report.get("tag") != "k100" or report.get("influent_mode") != "exact":
        raise ValueError("the sensor confirmation is chosen at K1-Ie (recoverability_k100.json)")
    value = report["sensor_value"]
    drop = max(value["drop_one"], key=lambda row: row["delta_J"])
    add = max(value["add_one"], key=lambda row: row["delta_J"])
    return {
        "drop": drop["name"], "add": add["name"],
        "drop_delta_J": drop["delta_J"], "add_delta_J": add["delta_J"],
        "criterion": value["criterion"], "source": report["data_dir"],
        "prior": report["prior"]["file"], "sigma": report["sigma"],
        "drop_ranking": value["drop_one"],
        "add_ranking": [{k: row[k] for k in ("name", "sigma", "delta_J")} for row in value["add_one"]],
        "confirmation_cell": "K1-Ie-A0 (configs/regime/k100_ie_a0_sensors.yaml, G5)",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", action="append", default=[])
    parser.add_argument("--sigma", type=float, default=0.10)
    parser.add_argument("--prior", default=None, help="G3 anchor file supplying z0_rel_std")
    parser.add_argument("--influent-mode", default="exact", choices=["exact", "composite", "composite_biased"])
    parser.add_argument("--nominal-dir", default="results/raw")
    parser.add_argument("--substeps", type=int, default=sens.DEFAULT_SUBSTEPS)
    parser.add_argument("--out", default="results/v11/analysis")
    parser.add_argument("--select-kinetics", action="store_true")
    parser.add_argument("--sensor-confirmation", default=None, metavar="RECOVERABILITY_JSON")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.sensor_confirmation:
        report = json.loads(Path(args.sensor_confirmation).read_text(encoding="utf-8"))
        choice = pick_sensor_confirmation(report)
        choice["git_commit"] = git_commit()
        path = out / "sensor_confirmation.json"
        path.write_text(json.dumps(clean(choice), indent=1), encoding="utf-8")
        print("drop %s (dJ %.4f), add %s (dJ %.4f) -> %s"
              % (choice["drop"], choice["drop_delta_J"], choice["add"], choice["add_delta_J"], path))
        return 0
    if not args.data_dir or not args.prior:
        parser.error("--data-dir and --prior are required for the analysis and --select-kinetics modes")
    if args.select_kinetics:
        if len(args.data_dir) != 1:
            parser.error("--select-kinetics takes exactly one (nominal) --data-dir")
        path = select_kinetics(args.data_dir[0], args.prior, args.sigma, out, args.substeps)
        print("kinetic subset -> %s" % path)
        return 0
    for data_dir in args.data_dir:
        path = analyse(data_dir, args.prior, args.sigma, out, args.nominal_dir,
                       args.influent_mode, args.substeps, figures=not args.no_figures)
        print("%s -> %s" % (data_dir, path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

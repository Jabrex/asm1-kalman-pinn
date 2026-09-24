"""Anchor files: --build-ensemble (15 kinetic constants perturbed lognormally, 200-day
warm-up, reject non-finite/failed/tank-5 X_B_A < 5 % of nominal) or --data-dir: A0 = truth
at index 0 (rel 0.01); As = nominal steady state, rel_std from the ensemble; Al1 = As +
t=0 panel (TSS all tanks 5 %; tanks 1, 5: COD, CODf 5 %, TKN, TKNf 7 %, ALK 5 %); Al2 = Al1
+ respirometric X_B_H 20 %, X_B_A 25 %. Lab values use the truth composition. Reads truth
at t=0 by design (scripts/ is outside the leakage scan); estimators only see the files."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.asm1.model import Asm1Kinetics  # noqa: E402
from src.asm1.plant import Bsm1Plant  # noqa: E402
from src.asm1.vault_loader import vault  # noqa: E402
from src.data.sensors import ObservationDataset  # noqa: E402
from src.data.simulate import steady_state_residual, warm_up  # noqa: E402
from src.observers import anchors as A  # noqa: E402

ENSEMBLE = Path("results/v11/anchors/nominal_ensemble.npz")
LAB_SEED = 20260923
A0_REL_STD = 0.01
REJECT_XBA_FRACTION = 0.05
IDENTITY_TOL = 1e-6
PANEL_ERRORS = {**{"TSS_tank%d" % k: 0.05 for k in range(1, 6)},
                **{"%s_tank%d" % (a, k): e for k in (1, 5)
                   for a, e in (("COD", 0.05), ("CODf", 0.05), ("TKN", 0.07), ("TKNf", 0.07),
                                ("ALK", 0.05))}}
RESPIROMETRY_ERRORS = {"X_B_H_tank1": 0.20, "X_B_H_tank5": 0.20,
                       "X_B_A_tank1": 0.25, "X_B_A_tank5": 0.25}


def _member(args):
    index, names, log_mult = args
    import torch

    torch.set_num_threads(1)
    from src.asm1.truth_plants import perturbed_vault

    try:
        plant = Bsm1Plant(source=perturbed_vault(dict(zip(names, map(float, log_mult)))))
        _, y = warm_up(plant)
        reactor = plant.unpack(y)[0]
        if not np.all(np.isfinite(reactor)):
            return index, None, math.nan, "non-finite"
        return index, reactor, steady_state_residual(plant, y), ""
    except Exception as exc:  # noqa: BLE001 - a failed draw is a rejected draw
        return index, None, math.nan, "%s: %s" % (type(exc).__name__, exc)


def build_ensemble(n: int, sigma_log: float, seed: int, out: Path, workers: int) -> dict:
    names = Asm1Kinetics(vault()).rate_parameters
    log_mult = np.random.default_rng(seed).normal(0.0, sigma_log, size=(n, len(names)))
    nominal_plant, nominal_y = warm_up(Bsm1Plant())
    nominal_ss = nominal_plant.unpack(nominal_y)[0]
    i_xba = nominal_plant.vault.index("X_B_A")
    members, residual, errors = np.full((n, 5, 14), np.nan), np.full(n, np.nan), [""] * n
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, reactor, res, err in pool.map(_member, [(i, names, log_mult[i]) for i in range(n)]):
            residual[i], errors[i] = res, err
            if reactor is not None:
                members[i] = reactor
    accepted = np.all(np.isfinite(members), axis=(1, 2))
    washed = accepted & (members[:, 4, i_xba] < REJECT_XBA_FRACTION * nominal_ss[4, i_xba])
    for i in np.flatnonzero(washed):
        errors[i] = "tank-5 X_B_A below 5% of nominal"
    accepted &= ~washed
    meta = {"n": n, "sigma_log": sigma_log, "seed": seed, "names": list(names),
            "n_accepted": int(accepted.sum()), "rejection_rate": float(1.0 - accepted.mean()),
            "errors": {str(i): e for i, e in enumerate(errors) if e},
            "vault_json_sha256": nominal_plant.vault.json_sha256,
            "max_steady_state_residual": float(np.nanmax(residual[accepted])) if accepted.any() else None}
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, members=members, accepted=accepted, log_multipliers=log_mult,
                        residual=residual, nominal_ss=nominal_ss, nominal_y=nominal_y,
                        meta=json.dumps(meta))
    return meta


def build_anchors(data_dir: Path, names: list[str], out_dir: Path, ensemble_path: Path,
                  lab_seed: int = LAB_SEED) -> dict:
    from src.asm1.truth_plants import truth_vault

    source = data_dir / "obs_dry_sigma0p00.npz"
    if not source.exists():
        # The random-mismatch plants store sigma 0.10 only; index-0 truth is noise-free anyway.
        source = data_dir / "obs_dry_sigma0p10.npz"
    dry = ObservationDataset.load(source)
    preset = str(dry.meta.get("truth_preset", "vault20"))
    alpha = dry.meta.get("alpha")
    alpha = 1.0 if alpha is None else float(alpha)  # perturbed plants record alpha as null
    offsteady = float(dry.meta.get("offsteady_days", 0.0))
    z_truth0 = np.asarray(dry.truth_reactor[0], dtype=float)
    y_truth0 = np.asarray(dry.truth_y[0], dtype=float)
    with np.load(ensemble_path, allow_pickle=False) as e:
        ens = {k: e[k] for k in e.files if k != "meta"}
        ens_meta = json.loads(str(e["meta"]))
    nominal_ss, nominal_y = ens["nominal_ss"], ens["nominal_y"]
    base = {"data_dir": data_dir.as_posix(), "truth_preset": preset, "alpha": alpha,
            "offsteady_days": offsteady, "vault_json_sha256": vault().json_sha256,
            "ensemble": ensemble_path.as_posix(),
            "ensemble_sha256": hashlib.sha256(ensemble_path.read_bytes()).hexdigest(),
            "sigma_log": ens_meta["sigma_log"], "rejection_rate": ens_meta["rejection_rate"]}
    specs = {"A0": (z_truth0, np.full((5, 14), A0_REL_STD), y_truth0,
                    {"source": "dry truth_reactor index 0 (oracle)"})}
    if set(names) & {"As", "Al1", "Al2"}:
        rel_s = A.ensemble_log_std(ens["members"][ens["accepted"]])
        specs["As"] = (nominal_ss, rel_s, nominal_y, {"source": "nominal vault steady state"})
    if set(names) & {"Al1", "Al2"}:
        truth_v = truth_vault(preset, alpha)
        panel = A.lab_values(A.lab_operators(truth_v), z_truth0, PANEL_ERRORS,
                             np.random.default_rng(lab_seed))
        ops = A.lab_operators(vault())
        al1 = A.gaussian_log_update(nominal_ss, rel_s, [ops[k] for k in PANEL_ERRORS],
                                    [panel[k] for k in PANEL_ERRORS], list(PANEL_ERRORS.values()))
        specs["Al1"] = (*al1, nominal_y, {"source": "As + routine panel at t=0",
                                          "lab_values": panel, "lab_rel_errors": PANEL_ERRORS})
        if "Al2" in names:
            resp = A.lab_values(A.respirometry_operators(truth_v), z_truth0, RESPIROMETRY_ERRORS,
                                np.random.default_rng([lab_seed, 2]))
            rops = A.respirometry_operators(vault())
            al2 = A.gaussian_log_update(*al1, [rops[k] for k in RESPIROMETRY_ERRORS],
                                        [resp[k] for k in RESPIROMETRY_ERRORS],
                                        list(RESPIROMETRY_ERRORS.values()))
            specs["Al2"] = (*al2, nominal_y, {"source": "Al1 + respirometry at t=0",
                                              "lab_values": {**panel, **resp},
                                              "lab_rel_errors": {**PANEL_ERRORS, **RESPIROMETRY_ERRORS}})
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {"out_dir": out_dir.as_posix(), "anchors": {}}
    for name in names:
        mean, rel, settler, extra = specs[name]
        path = out_dir / ("%s.npz" % name)
        np.savez_compressed(path, z0_mean=mean, z0_rel_std=rel, nominal_ss=nominal_ss,
                            settler_init=settler, meta=json.dumps({"name": name, **base, **extra,
                                                                   "lab_seed": lab_seed}))
        err = float(np.sqrt(np.mean(np.log(np.maximum(mean, 1e-12) / np.maximum(z_truth0, 1e-12)) ** 2)))
        report["anchors"][name] = {"file": path.as_posix(), "rms_log_error_vs_truth0": err,
                                   "median_rel_std": float(np.median(rel))}
    if preset == "vault20" and offsteady == 0.0:
        diff = float(np.max(np.abs(nominal_ss - z_truth0) / np.maximum(np.abs(z_truth0), 1e-30)))
        report["in_kinetics_identity_max_rel_diff"] = diff
        if diff > IDENTITY_TOL:
            raise AssertionError("As differs from A0 by %.3e on in-kinetics data" % diff)
    (out_dir / "anchors_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build-ensemble", action="store_true")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--sigma-log", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=min(12, max(1, (os.cpu_count() or 2) - 2)))
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--anchors", nargs="+", default=["A0", "As", "Al1", "Al2"],
                        choices=["A0", "As", "Al1", "Al2"])
    parser.add_argument("--ensemble", type=Path, default=ENSEMBLE)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lab-seed", type=int, default=LAB_SEED,
                        help="lab-panel noise seed; extra seeds give a spread, never a replacement")
    args = parser.parse_args(argv)
    if args.build_ensemble:
        meta = build_ensemble(args.n, args.sigma_log, args.seed, args.out, args.workers)
        print("ensemble: %d/%d accepted (rejection rate %.1f%%), max steady-state residual %s -> %s"
              % (meta["n_accepted"], meta["n"], 100 * meta["rejection_rate"],
                 meta["max_steady_state_residual"], args.out))
        return 0
    if args.data_dir is None:
        parser.error("--data-dir is required unless --build-ensemble is given")
    print(json.dumps(build_anchors(args.data_dir, args.anchors, args.out, args.ensemble, args.lab_seed),
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

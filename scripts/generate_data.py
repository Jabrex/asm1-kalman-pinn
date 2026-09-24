"""RUNBOOK steps 4 and 5 - generate the ground-truth trajectories and the
sensor datasets at every noise level.

Step 4 produces three ground-truth simulations, all from one shared warm-up so
every scenario starts from the identical steady state:

    sim_constant.npz   1 day, constant BSM1 Table 5 load     curriculum stage 1
    sim_dry.npz        14 days, diurnal dry weather          training + holdout
    sim_rain.npz       14 days, dry weather + rain event     distribution shift

Step 5 turns each of those into observation datasets, one per noise level:

    obs_<scenario>_sigma0p00 / 0p05 / 0p10 / 0p15 .npz

Run::

    python -m scripts.generate_data

v1.1 truth plants (``src/asm1/truth_plants.py``; no estimator ever sees these
parameters)::

    --truth-preset {vault20,bsm1_15c,graded,perturbed} --alpha A
    --log-multipliers JSON           perturbed preset only (random ensemble)
    --constant-from {truth,nominal}  plant that generates the stage-1 constant scenario
    --candidate-channels             append sensors.CANDIDATE_CHANNELS after the 8 columns
    --offsteady-days N               M0-prime start, a multiple of 7 days
    --noise-realisations R           also obs_dry_sigma0p10_r01..r(R-1)
    --realisations-only              add realisation files next to an existing sim_dry.npz

The script prints the achieved influent statistics next to their BSM1 anchors,
the COD and N closure for both the reactor train and the whole plant, and the
fraction of noisy samples that had to be clipped at zero, so a high-noise dataset
cannot silently become a biased one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import simpson

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.asm1.continuity import total_cod_and_n  # noqa: E402
from src.asm1.plant import Bsm1Plant  # noqa: E402
from src.asm1.truth_plants import PRESETS, effective_alpha, perturbed_vault, truth_vault  # noqa: E402
from src.asm1.vault_loader import Asm1Vault  # noqa: E402
from src.data.influent import constant_scenario, dry_weather, rain_weather  # noqa: E402
from src.data.sensors import CANDIDATE_CHANNELS, NOISE_LEVELS, SensorModel  # noqa: E402
from src.data.simulate import (  # noqa: E402
    SAMPLE_INTERVAL_DAYS,
    WARMUP_DAYS,
    SimulationResult,
    integrate,
    simulate,
    warm_up,
)

#: The v1.0 dataset. A full run refuses to write here unless --overwrite is
#: given; --realisations-only may add obs_dry_*_rXX.npz files next to it.
V1_RAW = REPO_ROOT / "results" / "raw"
#: Author-transcribed BSM1 reference (parameter table and open-loop steady state).
BSM1_REFERENCE = REPO_ROOT / "tests" / "data" / "bsm1_openloop_steady_state.json"

#: The reactor train is the control volume the PINN models and the one its
#: physics residual enforces, so its balance must close. This gate is enforced,
#: not merely printed. The whole-plant balance is reported alongside but not
#: gated: it carries the BSM1 eq. 46 clarifier approximation.
REACTOR_CLOSURE_GATE = 1e-6
#: Grid refinement for the closure re-check. Simpson's rule on the 15-minute
#: dataset grid has a quadrature floor, and truth kinetics other than the vault
#: set lift it over the gate while the trajectory itself is fine. Measured
#: 2026-09-23 by re-sampling one integration (worst of COD and N):
#:     vault20 dry     6.7e-7 (15 min)  2.8e-7 (7.5)  4.1e-8 (3.75)  2.8e-8 (1.875)
#:     bsm1_15c dry    2.0e-6           9.5e-7        2.3e-8         1.9e-8
#:     graded 0.5 rain 4.2e-6           2.0e-7
#: so the solver floor is near 3e-8 and the 15- and 7.5-minute excess is
#: quadrature. When the dataset-grid value fails, the same trajectory is sampled
#: again on a 3.75-minute grid for the check only (solve_ivp step selection does
#: not depend on t_eval); the saved dataset keeps the 15-minute grid.
CLOSURE_REFINE_FACTOR = 4

SCENARIOS = {
    "constant": (constant_scenario, 1.0),
    "dry": (dry_weather, 14.0),
    "rain": (rain_weather, 14.0),
}


def sigma_tag(sigma: float) -> str:
    return ("%.2f" % sigma).replace(".", "p")


def _integrate(values: np.ndarray, t: np.ndarray, axis: int = 0) -> np.ndarray:
    """Composite Simpson over the saved sampling grid.

    The closure check is evaluated on the 15-minute BSM1 grid, and with the
    trapezoidal rule its O(h^2) error was the binding term: refining the grid
    moved the residual from 4.0e-04 to 1.1e-06 while the model itself was
    unchanged. Simpson is O(h^4) on the same points, which drops the quadrature
    floor far below the gate so the gate measures the model.
    """
    return simpson(values, x=t, axis=axis)


def _oxygen_transferred(plant: Bsm1Plant, result: SimulationResult) -> float:
    """Integrated oxygen transfer [g O2]. S_O carries a COD coefficient of -1,
    so aeration is a COD sink at the system boundary."""
    kla = np.asarray(plant.cfg.kla)
    volumes = np.asarray(plant.cfg.volumes)
    return float(
        _integrate(
            np.sum(kla * volumes * (plant.cfg.so_sat - result.reactor[:, :, plant.i_so]), axis=1),
            result.t,
        )
    )


def closure_report(plant: Bsm1Plant, result: SimulationResult) -> dict[str, float]:
    """COD and N closure, reported for two nested control volumes.

    Reactor train
        The five tanks alone. This is the domain the PINN models and the domain
        its physics residual enforces, so it must close to solver accuracy.
        Boundary: influent in, return sludge in, tank-5 outflow out, aeration.
        The internal recycle cancels - it leaves tank 5 and re-enters tank 1.

    Whole plant
        Reactors plus clarifier. This one does NOT close tightly, and the reason
        is structural rather than numerical: BSM1 eq. 46 relabels the sludge
        already held in the clarifier with the reactor's *current* particulate
        composition instead of tracking each species through the layers. Under
        reaction the composition moves constantly - X_S alone swings by a factor
        of five over a day - so nitrogen bound to particulates (X_ND, biomass via
        iXB, X_P via iXP) is mis-accounted at the percent level. COD is far less
        affected because the five solids sum to 1/f_COD-SS of the layer solids
        regardless of how they are split.

    Both are reported so the split stays visible rather than averaged away.
    """
    comp = plant.vault.composition
    volumes = np.asarray(plant.cfg.volumes)[None, :, None]
    oxygen = _oxygen_transferred(plant, result) * comp[plant.i_so]

    def relative(net: np.ndarray, accumulated: np.ndarray, gross: np.ndarray) -> np.ndarray:
        """Mismatch relative to gross throughput, not to the residual itself.

        Near steady state the net boundary flux and the accumulation both tend
        to zero while the individual streams stay enormous - here the recycle
        alone carries 1.3e8 g COD/d. Dividing the mismatch by ``|net| + |acc|``
        would divide two tiny near-cancelling numbers by each other and report
        machine noise as a percent-level error. Gross throughput is the honest
        denominator.
        """
        return np.abs(net[:2] - accumulated[:2]) / np.maximum(gross[:2], 1e-9)

    out: dict[str, float] = {}

    # --- reactor train ---------------------------------------------------
    # The internal recycle leaves tank 5 and re-enters tank 1, both inside this
    # control volume, so it cancels from the boundary flux.
    z5 = result.reactor[:, -1, :]
    q_r = plant.cfg.q_r
    influent_load = _integrate(result.q_in[:, None] * (result.influent @ comp), result.t, axis=0)
    ras_load = _integrate(q_r * (result.underflow @ comp), result.t, axis=0)
    tank5_load = _integrate((result.q_in + q_r)[:, None] * (z5 @ comp), result.t, axis=0)

    reactor_net = influent_load + ras_load - tank5_load + oxygen
    holdup = np.einsum("ntc,cq->nq", result.reactor * volumes, comp)
    reactor_accumulated = holdup[-1] - holdup[0]
    reactor_gross = np.abs(influent_load) + np.abs(ras_load) + np.abs(tank5_load) + np.abs(oxygen)
    error = relative(reactor_net, reactor_accumulated, reactor_gross)
    out["reactor_cod_closure"] = float(error[0])
    out["reactor_n_closure"] = float(error[1])

    # --- whole plant -----------------------------------------------------
    q_e = result.q_in - plant.cfg.q_w
    effluent_load = _integrate(q_e[:, None] * (result.effluent @ comp), result.t, axis=0)
    wastage_load = _integrate(plant.cfg.q_w * (result.underflow @ comp), result.t, axis=0)

    plant_net = influent_load - effluent_load - wastage_load + oxygen
    cod0, n0 = total_cod_and_n(plant, result.y[0])
    cod1, n1 = total_cod_and_n(plant, result.y[-1])
    plant_accumulated = np.array([cod1 - cod0, n1 - n0])
    plant_gross = (
        np.abs(influent_load) + np.abs(effluent_load) + np.abs(wastage_load) + np.abs(oxygen)
    )
    error = relative(plant_net, plant_accumulated, plant_gross)
    out["plant_cod_closure"] = float(error[0])
    out["plant_n_closure"] = float(error[1])
    return out


def gated_closure(
    plant: Bsm1Plant,
    builder,
    duration: float,
    y0: np.ndarray,
    result: SimulationResult,
    gate: float = REACTOR_CLOSURE_GATE,
) -> dict[str, object]:
    """Closure on the dataset grid, re-checked on a finer grid if it fails."""
    closure: dict[str, object] = dict(closure_report(plant, result))
    worst = max(closure["reactor_cod_closure"], closure["reactor_n_closure"])
    closure["gate_grid_days"] = SAMPLE_INTERVAL_DAYS
    closure["gate_value"] = worst
    if worst >= gate:
        fine = simulate(builder(duration), plant=plant, y0=y0.copy(),
                        sample_interval=SAMPLE_INTERVAL_DAYS / CLOSURE_REFINE_FACTOR)
        refined = closure_report(plant, fine)
        closure["refined"] = refined
        closure["gate_grid_days"] = SAMPLE_INTERVAL_DAYS / CLOSURE_REFINE_FACTOR
        closure["gate_value"] = max(refined["reactor_cod_closure"], refined["reactor_n_closure"])
    closure["pass"] = bool(closure["gate_value"] < gate)
    return closure


def noise_seed(base: int, sigma: float, realisation: int = 0) -> int:
    """v1.0 rule ``base + 1000 k`` (k = index in NOISE_LEVELS), plus ``100000 r``.

    Realisation 0 is the standard file, so its seed is exactly the v1.0 seed.
    """
    if sigma not in NOISE_LEVELS:
        raise ValueError("sigma %r is not one of NOISE_LEVELS %s" % (sigma, NOISE_LEVELS))
    return int(base) + 1000 * NOISE_LEVELS.index(sigma) + 100000 * int(realisation)


def realisation_name(scenario: str, sigma: float, realisation: int) -> str:
    return "obs_%s_sigma%s_r%02d.npz" % (scenario, sigma_tag(sigma), realisation)


def offsteady_start(plant: Bsm1Plant, y_steady: np.ndarray, days: float) -> np.ndarray:
    """M0-prime start: the state after ``days`` of dry weather from the steady state.

    The dry-weather influent repeats every 7 days (1-day diurnal cycle times the
    weekly factor), so for a multiple of 7 the evaluated window that follows
    sees exactly the influent it would have seen from the steady state.
    """
    if days == 0.0:
        return y_steady.copy()
    weeks = days / 7.0
    if days < 0.0 or abs(weeks - round(weeks)) > 1e-9:
        raise ValueError("--offsteady-days must be a non-negative multiple of 7, got %r" % days)
    _, y = integrate(plant, dry_weather(days), (0.0, days), y_steady, t_eval=np.array([days]))
    return y[-1]


def build_truth_source(
    args: argparse.Namespace,
) -> tuple[Asm1Vault, float | None, dict[str, float] | None]:
    """``(truth vault, effective alpha or None, log multipliers or None)`` from the CLI."""
    if args.truth_preset == "perturbed":
        if args.log_multipliers is None:
            raise SystemExit("--truth-preset perturbed needs --log-multipliers")
        multipliers = {k: float(m) for k, m in json.loads(args.log_multipliers).items()}
        return perturbed_vault(multipliers), None, multipliers
    if args.log_multipliers is not None:
        raise SystemExit("--log-multipliers is only valid with --truth-preset perturbed")
    return (truth_vault(args.truth_preset, args.alpha),
            effective_alpha(args.truth_preset, args.alpha), None)


def bsm1_parameter_problems(args: argparse.Namespace) -> list[str]:
    """Presets built from BSM1_15C need the author's cell-by-cell check first."""
    if args.truth_preset not in ("bsm1_15c", "graded") or args.allow_unchecked_parameters:
        return []
    from scripts.verify_bsm1_truth import load_reference, parameter_check_problems

    return parameter_check_problems(load_reference(Path(args.bsm1_reference)))


def print_state_comparison(plant: Bsm1Plant, y_truth: np.ndarray, y_nominal: np.ndarray) -> None:
    """Truth vs nominal steady state, per component, all five tanks."""
    truth = plant.unpack(y_truth)[0]
    nominal = plant.unpack(y_nominal)[0]
    print("        steady state, truth (tanks 1-5) vs nominal vault (tank 5):")
    for j, name in enumerate(plant.vault.components):
        if name == "S_N2":
            continue
        values = "  ".join("%10.4f" % truth[k, j] for k in range(truth.shape[0]))
        ratio = truth[-1, j] / nominal[-1, j] if nominal[-1, j] != 0.0 else float("nan")
        print("        %-6s %s   nominal %10.4f   ratio %.3f" % (name, values, nominal[-1, j], ratio))


def write_realisations(
    out_dir: Path, result: SimulationResult, sensors: SensorModel, extras: tuple,
    seed: int, sigmas: list[float], n_realisations: int,
) -> dict[str, dict[str, object]]:
    entries: dict[str, dict[str, object]] = {}
    for r in range(1, n_realisations):
        for sigma in sigmas:
            s = noise_seed(seed, sigma, r)
            dataset = sensors.build(result, sigma=sigma, seed=s, extra_channels=extras)
            path = dataset.save(out_dir / realisation_name("dry", sigma, r))
            entries[path.stem] = {"file": path.name, "sigma": sigma, "realisation": r,
                                  "seed": s, "clip_fraction": dataset.clip_fraction}
            print("        realisation r=%02d sigma=%.2f -> %s" % (r, sigma, path.name))
    return entries


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/raw", help="output directory")
    parser.add_argument("--seed", type=int, default=0, help="noise seed base")
    parser.add_argument(
        "--scenarios", nargs="*", default=list(SCENARIOS), choices=list(SCENARIOS)
    )
    parser.add_argument("--sigmas", nargs="*", type=float, default=list(NOISE_LEVELS),
                        help="subset of NOISE_LEVELS; seeds keep the v1.0 index rule")
    parser.add_argument("--truth-preset", default="vault20",
                        choices=list(PRESETS) + ["perturbed"])
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="position on the vault20 -> bsm1_15c axis (graded only)")
    parser.add_argument("--log-multipliers", default=None,
                        help="JSON object {parameter: log multiplier}, perturbed preset only")
    parser.add_argument("--constant-from", default="truth", choices=["truth", "nominal"],
                        help="plant that generates the constant stage-1 scenario")
    parser.add_argument("--candidate-channels", action="store_true",
                        help="append sensors.CANDIDATE_CHANNELS after the eight standard columns")
    parser.add_argument("--offsteady-days", type=float, default=0.0,
                        help="M0-prime: start dry/rain after this many days of dry weather")
    parser.add_argument("--noise-realisations", type=int, default=1,
                        help="dry-scenario noise realisations; r = 0 is the standard file")
    parser.add_argument("--realisation-sigmas", nargs="*", type=float, default=[0.10])
    parser.add_argument("--realisations-only", action="store_true",
                        help="only add obs_dry_*_rXX files from an existing sim_dry.npz")
    parser.add_argument("--overwrite", action="store_true",
                        help="allow a full run into the existing v1.0 results/raw")
    parser.add_argument("--bsm1-reference", default=str(BSM1_REFERENCE),
                        help="author-transcribed BSM1 reference JSON")
    parser.add_argument("--allow-unchecked-parameters", action="store_true",
                        help="skip the BSM1 parameter-check guard (smoke tests into temp dirs only)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = Path(args.out)
    sensors = SensorModel()
    extras = tuple(CANDIDATE_CHANNELS) if args.candidate_channels else ()
    for sigma in list(args.sigmas) + list(args.realisation_sigmas):
        noise_seed(args.seed, sigma)  # validates membership in NOISE_LEVELS

    if args.realisations_only:
        result = SimulationResult.load(out_dir / "sim_dry.npz")
        entries = write_realisations(out_dir, result, sensors, extras, args.seed,
                                     list(args.realisation_sigmas), args.noise_realisations)
        record = {"source_simulation": "sim_dry.npz", "seed_rule": "seed + 1000*k + 100000*r",
                  "seed": args.seed, "extra_channels": [c.name for c in extras],
                  "realisations": entries}
        (out_dir / "realisations.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print("Wrote %s" % (out_dir / "realisations.json"))
        return 0

    if out_dir.resolve() == V1_RAW.resolve() and (V1_RAW / "manifest.json").exists() \
            and not args.overwrite:
        print("REFUSED - %s holds the v1.0 data. Use --realisations-only to add files, "
              "or --overwrite to regenerate it deliberately." % out_dir)
        return 2
    problems = bsm1_parameter_problems(args)
    if problems:
        print("REFUSED - BSM1_15C has not been checked against the BSM1 report "
              "(%d open items, first: %s). Fill parameter_check in %s."
              % (len(problems), problems[0], args.bsm1_reference))
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    truth, alpha, multipliers = build_truth_source(args)
    print("STEP 4  warm-up (%.0f days, constant BSM1 Table 5 load, truth preset %s, alpha %s)"
          % (WARMUP_DAYS, args.truth_preset, alpha))
    plant, y_steady = warm_up(Bsm1Plant(source=truth))
    y_start = offsteady_start(plant, y_steady, args.offsteady_days)
    if args.offsteady_days:
        print("        dry/rain start after %.0f days of dry weather (M0-prime)" % args.offsteady_days)

    nominal_plant = nominal_steady = None
    if args.constant_from == "nominal" or args.truth_preset == "bsm1_15c":
        nominal_plant, nominal_steady = warm_up(Bsm1Plant())
    if args.truth_preset == "bsm1_15c":
        print_state_comparison(plant, y_steady, nominal_steady)
    print("        steady state reached; reusing it for every scenario\n")

    closure_failures: list[str] = []
    manifest: dict[str, object] = {
        "argv": list(argv) if argv is not None else sys.argv[1:],
        "vault_json_sha256": plant.vault.json_sha256,
        "source_xlsx_sha256": plant.vault.source_xlsx_sha256,
        "truth_preset": args.truth_preset,
        "alpha": alpha,
        "parameters": dict(truth.parameters),
        "log_multipliers": multipliers,
        "constant_from": args.constant_from,
        "offsteady_days": args.offsteady_days,
        "noise_levels": list(args.sigmas),
        "noise_seed_rule": "seed + 1000*k (k = index in NOISE_LEVELS) + 100000*r",
        "seed": args.seed,
        "sensor_channels": list(sensors.names) + [c.name for c in extras],
        "extra_channels": [c.name for c in extras],
        "noise_realisations": args.noise_realisations,
        "realisation_sigmas": list(args.realisation_sigmas),
        "closure_gate": REACTOR_CLOSURE_GATE,
        "closure_refine_factor": CLOSURE_REFINE_FACTOR,
        # Thread count changes BDF's LAPACK path: 1 thread differs from the
        # threaded path by ~5e-12 relative, while 4 and 24 threads are bit-identical.
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "steady_state_reactor": plant.unpack(y_steady)[0].tolist(),
        "start_reactor": plant.unpack(y_start)[0].tolist(),
        "scenarios": {},
    }

    for name in args.scenarios:
        builder, duration = SCENARIOS[name]
        if name == "constant" and args.constant_from == "nominal":
            scen_plant, scen_y0 = nominal_plant, nominal_steady
            extra_meta = {"truth_preset": "vault20", "alpha": 0.0, "constant_from": "nominal",
                          "offsteady_days": 0.0, "directory_truth_preset": args.truth_preset}
        else:
            scen_plant = plant
            scen_y0 = y_steady if name == "constant" else y_start
            extra_meta = {"truth_preset": args.truth_preset, "alpha": alpha,
                          "constant_from": args.constant_from,
                          "offsteady_days": 0.0 if name == "constant" else args.offsteady_days}
        print("STEP 4  simulating scenario %r (%.0f days, plant %s)"
              % (name, duration, extra_meta["truth_preset"]))
        result = simulate(builder(duration), plant=scen_plant, y0=scen_y0.copy(),
                          scenario=name, meta_extra=extra_meta)
        sim_path = result.save(out_dir / ("sim_%s.npz" % name))
        closure = gated_closure(scen_plant, builder, duration, scen_y0, result)
        print("        saved %s  (%d samples)" % (sim_path.name, len(result.t)))
        verdict = "PASS" if closure["pass"] else "FAIL"
        print("        reactor train closure   COD %.3e   N %.3e   (15-min grid)"
              % (closure["reactor_cod_closure"], closure["reactor_n_closure"]))
        if "refined" in closure:
            print("        re-checked, %.2f-min grid COD %.3e   N %.3e"
                  % (closure["gate_grid_days"] * 1440.0, closure["refined"]["reactor_cod_closure"],
                     closure["refined"]["reactor_n_closure"]))
        print("        reactor train gate      %.3e   %s (gate < %.0e)"
              % (closure["gate_value"], verdict, REACTOR_CLOSURE_GATE))
        print("        whole plant closure     COD %.3e   N %.3e   (BSM1 eq.46, reported)"
              % (closure["plant_cod_closure"], closure["plant_n_closure"]))
        if verdict == "FAIL":
            closure_failures.append(name)

        summary = result.meta.get("influent_summary", {})
        # The constant scenario reports only a mean; the diurnal ones report the
        # achieved extremes as well.
        if "flow_min_achieved" in summary:
            print("        flow mean %.1f (dry-weather target %.1f), range [%.0f, %.0f] (target %s)"
                  % (summary["flow_mean_achieved"], summary["flow_mean_target"],
                     summary["flow_min_achieved"], summary["flow_max_achieved"],
                     summary.get("flow_range_target")))
            if "dry_component_mean_achieved" in summary:
                print("        (rain event included above; dry component alone means %.1f)"
                      % summary["dry_component_mean_achieved"])
        elif "flow_mean_achieved" in summary:
            print("        flow held constant at %.1f (BSM1 Table 5)"
                  % summary["flow_mean_achieved"])

        scenario_entry: dict[str, object] = {
            "simulation": sim_path.name,
            "samples": int(len(result.t)),
            "closure": closure,
            "influent_summary": summary,
            "meta_extra": extra_meta,
            "observations": {},
        }

        print("STEP 5  building observation datasets")
        for sigma in args.sigmas:
            s = noise_seed(args.seed, sigma)
            dataset = sensors.build(result, sigma=sigma, seed=s, extra_channels=extras)
            obs_path = dataset.save(out_dir / ("obs_%s_sigma%s.npz" % (name, sigma_tag(sigma))))
            print("        sigma=%.2f -> %s   clipped %.3f%% of samples"
                  % (sigma, obs_path.name, 100.0 * dataset.clip_fraction))
            scenario_entry["observations"][sigma_tag(sigma)] = {
                "file": obs_path.name,
                "sigma": sigma,
                "clip_fraction": dataset.clip_fraction,
                "seed": dataset.seed,
            }
        if name == "dry" and args.noise_realisations > 1:
            scenario_entry["realisations"] = write_realisations(
                out_dir, result, sensors, extras, args.seed,
                list(args.realisation_sigmas), args.noise_realisations,
            )
        manifest["scenarios"][name] = scenario_entry
        print()

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Wrote %s" % manifest_path)
    if closure_failures:
        print("FAIL - reactor mass balance did not close for: %s"
              % ", ".join(closure_failures))
        return 1
    print("PASS - steps 4 and 5 complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

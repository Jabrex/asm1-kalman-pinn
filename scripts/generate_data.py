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

The script prints the achieved influent statistics next to their BSM1 anchors,
the COD and N closure for both the reactor train and the whole plant, and the
fraction of noisy samples that had to be clipped at zero, so a high-noise dataset
cannot silently become a biased one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import simpson

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.asm1.continuity import total_cod_and_n  # noqa: E402
from src.asm1.plant import Bsm1Plant  # noqa: E402
from src.data.influent import constant_scenario, dry_weather, rain_weather  # noqa: E402
from src.data.sensors import NOISE_LEVELS, SensorModel  # noqa: E402
from src.data.simulate import WARMUP_DAYS, SimulationResult, simulate, warm_up  # noqa: E402

#: The reactor train is the control volume the PINN models and the one its
#: physics residual enforces, so its balance must close. This gate is enforced,
#: not merely printed. The whole-plant balance is reported alongside but not
#: gated: it carries the BSM1 eq. 46 clarifier approximation.
REACTOR_CLOSURE_GATE = 1e-6

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/raw", help="output directory")
    parser.add_argument("--seed", type=int, default=0, help="noise seed base")
    parser.add_argument(
        "--scenarios", nargs="*", default=list(SCENARIOS), choices=list(SCENARIOS)
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("STEP 4  warm-up (%.0f days, constant BSM1 Table 5 load, vault 20 C parameters)"
          % WARMUP_DAYS)
    plant, y0 = warm_up()
    print("        steady state reached; reusing it for every scenario\n")

    sensors = SensorModel()
    closure_failures: list[str] = []
    manifest: dict[str, object] = {
        "vault_json_sha256": plant.vault.json_sha256,
        "source_xlsx_sha256": plant.vault.source_xlsx_sha256,
        "noise_levels": list(NOISE_LEVELS),
        "sensor_channels": list(sensors.names),
        "scenarios": {},
    }

    for name in args.scenarios:
        builder, duration = SCENARIOS[name]
        print("STEP 4  simulating scenario %r (%.0f days)" % (name, duration))
        result = simulate(
            builder(duration), plant=plant, y0=y0.copy(), scenario=name
        )
        sim_path = result.save(out_dir / ("sim_%s.npz" % name))
        closure = closure_report(plant, result)
        print("        saved %s  (%d samples)" % (sim_path.name, len(result.t)))
        worst_reactor = max(closure["reactor_cod_closure"], closure["reactor_n_closure"])
        verdict = "PASS" if worst_reactor < REACTOR_CLOSURE_GATE else "FAIL"
        print("        reactor train closure   COD %.3e   N %.3e   %s (gate < %.0e)"
              % (closure["reactor_cod_closure"], closure["reactor_n_closure"],
                 verdict, REACTOR_CLOSURE_GATE))
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
            "observations": {},
        }

        print("STEP 5  building observation datasets")
        for k, sigma in enumerate(NOISE_LEVELS):
            dataset = sensors.build(result, sigma=sigma, seed=args.seed + 1000 * k)
            obs_path = dataset.save(out_dir / ("obs_%s_sigma%s.npz" % (name, sigma_tag(sigma))))
            print("        sigma=%.2f -> %s   clipped %.3f%% of samples"
                  % (sigma, obs_path.name, 100.0 * dataset.clip_fraction))
            scenario_entry["observations"][sigma_tag(sigma)] = {
                "file": obs_path.name,
                "sigma": sigma,
                "clip_fraction": dataset.clip_fraction,
                "seed": dataset.seed,
            }
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

"""Run the estimators over regime cells in processes with one torch thread each
(default 12 workers, about 0.6 GB each).

Two config layouts are accepted:
- a G6 cell file (top-level ``cell:``), validated by src.observers.cell_inputs, whose
  run directories are ``<out_dir>/<estimator>[_frozenq]_sigma<tag>[_r<k>]``;
- a G3 multi-cell file (``cells:``, ``grid:``, ``random:``), written to
  ``<out_root>/<cell>/<estimator>[_frozenq|_qfixed]_sigma<tag>[_r<k>]``.
Both run the same estimators (src.observers.pipeline). --freeze-q writes
<out_root>/frozen_q.json from the first cell at sigma 0.10."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

KINETICS_DIRS = {"k000": "results/raw", "k025": "results/raw_k025", "k050": "results/raw_k050",
                 "k075": "results/raw_k075", "k100": "results/raw_k100",
                 "k000_off": "results/raw_k000_off"}
INFLUENT_TAGS = {"exact": "ie", "composite": "ic", "composite_biased": "ib"}
# "_frozenq" matches src.observers.cell_inputs.run_dir_name (G6 naming).
Q_SUFFIX = {"tuned": "", "frozen": "_frozenq", "fixed": "_qfixed"}
CELL_KEYS = ("sigmas", "realisations", "estimators", "q_modes", "q_fixed", "q_criterion",
             "ras_filter_window", "ras_mode", "target_channels", "augment", "augment_file",
             "q_theta", "rain", "train_end_day", "holdout_days", "r_mode", "r_floor", "q_grid",
             "theta_prior_sd")
RAS_MODE_OF_INPUT = {"filtered": "measured", "ideal_settler": "ideal_settler"}
DEFAULT_WORKERS = min(12, max(1, (os.cpu_count() or 2) - 2))


def sigma_tag(sigma: float) -> str:
    return ("%.2f" % sigma).replace(".", "p")


def expand_cells(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    defaults = {k: cfg[k] for k in CELL_KEYS if k in cfg}
    cells = [{**defaults, **raw} for raw in cfg.get("cells", [])]
    grid = cfg.get("grid")
    if grid:
        root = cfg.get("anchors_root", "results/v11/anchors")
        for kin, inf, anc in itertools.product(grid["kinetics"], grid["influent"], grid["anchors"]):
            cells.append({**defaults, "name": "%s_%s_%s" % (kin, INFLUENT_TAGS[inf], anc.lower()),
                          "data_dir": KINETICS_DIRS[kin], "influent_mode": inf,
                          "anchor_file": "%s/%s/%s.npz" % (root, kin, anc)})
    rand = cfg.get("random")
    if rand:
        for i in range(int(rand["count"])):
            cells.append({**defaults, "name": "rand%02d_ie_a0" % i, "influent_mode": "exact",
                          # G2 writes the random plants as raw_rand/00..49; anchors follow suit.
                          "data_dir": "%s/%02d" % (rand["data_root"], i),
                          "anchor_file": "%s/%02d/A0.npz" % (rand["anchors_root"], i)})
    for cell in cells:
        missing = [k for k in ("name", "data_dir", "anchor_file", "influent_mode") if k not in cell]
        if missing:
            raise ValueError("cell %r lacks %s" % (cell.get("name"), missing))
    return cells


def run_dir_name(estimator: str, q_mode: str, sigma: float, realisation: int) -> str:
    return "%s%s_sigma%s%s" % (estimator, Q_SUFFIX[q_mode], sigma_tag(sigma),
                               "_r%02d" % realisation if realisation else "")


def _dir_name(cell: dict[str, Any], estimator: str, q_mode: str, sigma: float, r: int) -> str:
    if cell.get("schema") == "cell":
        from src.observers import cell_inputs

        return cell_inputs.run_dir_name(estimator, q_mode, sigma, r)
    return run_dir_name(estimator, q_mode, sigma, r)


def cell_from_schema(cfg: dict[str, Any]) -> dict[str, Any]:
    """A validated G6 cell file as the G3 cell dict that run_job executes."""
    return {
        "schema": "cell", "name": cfg["cell"], "data_dir": cfg["data_dir"],
        "anchor_file": cfg["anchor_file"], "influent_mode": cfg["influent_mode"],
        "ras_filter_window": int(cfg["ras_filter_window"]), "ras_input": cfg["ras_input"],
        "ras_mode": RAS_MODE_OF_INPUT[cfg["ras_input"]], "estimators": list(cfg["estimators"]),
        "augment": list(cfg["augment"]), "q_theta": float(cfg["q_theta"]),
        "theta_prior_sd": float(cfg["theta_prior_sd"]), "q_grid": [float(q) for q in cfg["q_grid"]],
        "r_floor": float(cfg["r_floor"]),
        "target_channels": None if cfg["channels"] == "default" else list(cfg["channels"]),
        "sigmas": [float(x) for x in cfg["sigmas"]], "realisations": [int(x) for x in cfg["realisations"]],
        "q_modes": list(cfg["q_mode"]), "train_end_day": float(cfg["tune_window_days"][1]),
    }


def _label(job: dict[str, Any]) -> str:
    return "%s/sigma%s_r%02d_%s" % (job["cell"]["name"], sigma_tag(job["sigma"]),
                                    job["realisation"], job["q_mode"])


def _augment(cell: dict[str, Any]) -> tuple[str, ...]:
    if cell.get("augment"):
        return tuple(cell["augment"])
    path = cell.get("augment_file")
    if path and Path(path).exists():
        return tuple(json.loads(Path(path).read_text(encoding="utf-8"))["names"])
    return ()


def run_job(job: dict[str, Any], out_root: str, frozen_q: list[float] | None) -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)
    from src.asm1.vault_loader import vault
    from src.data.sensors import ObservationDataset
    from src.observers.pipeline import ObserverSpec, run_estimators

    cell, sigma, r, q_mode = job["cell"], job["sigma"], job["realisation"], job["q_mode"]
    out = Path(out_root) / cell["name"]
    started = time.perf_counter()
    try:
        data_dir = Path(cell["data_dir"])
        dry = ObservationDataset.load(data_dir / ("obs_dry_sigma%s%s.npz"
                                                  % (sigma_tag(sigma), "_r%02d" % r if r else "")))
        rain_path = data_dir / ("obs_rain_sigma%s.npz" % sigma_tag(sigma))
        want_rain = bool(cell.get("rain", True)) and r == 0 and rain_path.exists()
        with np.load(cell["anchor_file"], allow_pickle=False) as a:
            anchor = {k: a[k] for k in a.files if k != "meta"}
            anchor_meta = json.loads(str(a["meta"]))
        spec = ObserverSpec(
            estimators=tuple(cell.get("estimators", ("ekf", "eks", "ieks", "eks_aug", "ekf_online"))),
            influent_mode=cell["influent_mode"], ras_filter_window=int(cell.get("ras_filter_window", 4)),
            ras_mode=cell.get("ras_mode", "measured"),
            train_end_day=float(cell.get("train_end_day", 12.0)),
            holdout_days=tuple(cell.get("holdout_days", (12.0, 14.0))), q_mode=q_mode,
            q_fixed=(tuple(frozen_q) if q_mode == "frozen"
                     else tuple(cell["q_fixed"]) if q_mode == "fixed" else None),
            q_criterion=cell.get("q_criterion", "innovation"), augment=_augment(cell),
            q_grid=tuple(cell["q_grid"]) if cell.get("q_grid") else None,
            q_theta=float(cell.get("q_theta", 1e-3)),
            theta_prior_sd=float(cell.get("theta_prior_sd", 0.693)),
            r_floor=float(cell.get("r_floor", 0.01)),
            target_channels=tuple(cell["target_channels"]) if cell.get("target_channels") else None,
            r_mode=cell.get("r_mode", "data"), rain=want_rain)
        results = run_estimators(dry, ObservationDataset.load(rain_path) if want_rain else None,
                                 anchor, spec)
        elapsed = time.perf_counter() - started
        for estimator, entry in results.items():
            run_dir = out / _dir_name(cell, estimator, q_mode, sigma, r)
            run_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(run_dir / "predictions.npz", **entry["predictions"])
            summary = {
                "run_id": run_dir.name, "model": run_dir.name.split("_sigma")[0], "arch": "observer",
                "curriculum": "none", "noise": sigma, "seed": r, "profile": "full", "steps": 0,
                "train_seconds": elapsed, "n_parameters": 0, "train_end_day": spec.train_end_day,
                "holdout_days": list(spec.holdout_days), "cell": cell["name"],
                "data_dir": data_dir.as_posix(), "truth_preset": dry.meta.get("truth_preset", "vault20"),
                "alpha": dry.meta.get("alpha", 1.0), "anchor": anchor_meta.get("name"),
                "anchor_file": cell["anchor_file"], "spec": asdict(spec),
                "vault_json_sha256": vault().json_sha256, **entry["info"]}
            if cell.get("schema") == "cell":
                summary.update(_schema_summary(cell, estimator, q_mode, r, entry))
            (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float),
                                                  encoding="utf-8")
        return {"job": _label(job), "ok": True, "seconds": elapsed}
    except Exception:  # noqa: BLE001 - one failed job must not stop the grid
        out.mkdir(parents=True, exist_ok=True)
        err = out / ("error_%s.txt" % _label(job).replace("/", "_"))
        err.write_text(traceback.format_exc(), encoding="utf-8")
        return {"job": _label(job), "ok": False, "error": str(err)}


def _schema_summary(cell: dict[str, Any], estimator: str, q_mode: str, r: int,
                    entry: dict[str, Any]) -> dict[str, Any]:
    """The G6 summary keys (cell_inputs contract) on top of the G3 ones."""
    from src.observers import cell_inputs

    info = entry["info"]
    nis = info.get("nis_mean")
    out = {"model": cell_inputs.observer_model_name(estimator, q_mode), "seed": int(r),
           **cell_inputs.numerics_summary(np.atleast_1d(np.inf if nis is None else nis),
                                          entry["predictions"]["train"], len(info["channels"])),
           "selected_q": {"q_soluble": info["q_soluble"], "q_particulate": info["q_particulate"]},
           "estimator": estimator, "q_mode": q_mode, "realisation": int(r),
           "influent_mode": cell["influent_mode"], "ras_input": cell["ras_input"]}
    mult = info.get("multipliers")
    if estimator.endswith("_aug") and mult:
        out["learned_multipliers"] = dict(zip(mult["names"], (float(m) for m in mult["filtered_end"])))
    return out


def _init_worker() -> None:
    import torch

    torch.set_num_threads(1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--resume", action="store_true", help="skip jobs whose run dirs exist")
    parser.add_argument("--list", action="store_true", help="print the job plan and exit")
    parser.add_argument("--estimators", nargs="+", default=None, help="override every cell's list")
    parser.add_argument("--freeze-q", action="store_true")
    args = parser.parse_args(argv)

    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    frozen_q_from = None
    if isinstance(raw, dict) and "cell" in raw:
        from src.observers import cell_inputs

        cfg = cell_inputs.load_cell_config(args.config)
        for s, r in itertools.product(cfg["sigmas"], cfg["realisations"]):
            # Fail fast in the parent: data file, anchor and channel set must all resolve.
            cell_inputs.prepare_cell_inputs(cfg, float(s), int(r),
                                            train_end_day=float(cfg["tune_window_days"][1]))
        out_root = Path(cfg["out_dir"]).parent
        cells = [cell_from_schema(cfg)]
        frozen_q_from = cfg["frozen_q_from"]
    else:
        cfg = raw
        out_root = Path(cfg.get("out_root", "results/v11/observers"))
        cells = expand_cells(cfg)
    if args.estimators:
        cells = [{**c, "estimators": list(args.estimators)} for c in cells]
    jobs = [{"cell": c, "sigma": float(s), "realisation": int(r), "q_mode": q}
            for c in cells for s in c.get("sigmas", [0.10]) for r in c.get("realisations", [0])
            for q in c.get("q_modes", ["tuned"])]
    if args.resume:
        jobs = [j for j in jobs if not all(
            (out_root / j["cell"]["name"] / _dir_name(j["cell"], e, j["q_mode"], j["sigma"], j["realisation"])
             / "summary.json").exists() for e in j["cell"].get("estimators", []))]
    frozen_q = None
    if any(j["q_mode"] == "frozen" for j in jobs) and not args.list:
        path = Path(frozen_q_from) if frozen_q_from else out_root / "frozen_q.json"
        if not path.exists():
            raise SystemExit("q_modes has 'frozen' but %s is missing; run the reference cell "
                             "with --freeze-q first" % path)
        from src.observers import cell_inputs

        q = cell_inputs.read_frozen_q(path)
        frozen_q = [q["q_soluble"], q["q_particulate"]]
    print("Observer grid: %d cells, %d jobs, %d workers -> %s"
          % (len(cells), len(jobs), args.workers, out_root))
    for j in jobs:
        print("  " + _label(j))
    if args.list:
        return 0

    started, results = time.perf_counter(), []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        for res in pool.map(run_job, jobs, [str(out_root)] * len(jobs), [frozen_q] * len(jobs)):
            results.append(res)
            print("  %-48s %s" % (res["job"], "ok %.0f s" % res["seconds"] if res["ok"]
                                  else "FAILED -> " + res["error"]), flush=True)
    failures = [r for r in results if not r["ok"]]
    print("Finished in %.1f min: %d ok, %d failed"
          % ((time.perf_counter() - started) / 60.0, len(results) - len(failures), len(failures)))
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / ("index_%s.json" % Path(args.config).stem)).write_text(
        json.dumps({"config": args.config, "results": results}, indent=2), encoding="utf-8")
    if args.freeze_q:
        ref = cells[0]
        est = next(e for e in ("eks", "ekf", "ieks") if e in ref.get("estimators", []))
        src = out_root / ref["name"] / run_dir_name(est, "tuned", 0.10, 0) / "summary.json"
        info = json.loads(src.read_text(encoding="utf-8"))
        frozen = {"q_soluble": info["q_soluble"], "q_particulate": info["q_particulate"],
                  "q_criterion": info["q_criterion"], "source": src.as_posix()}
        (out_root / "frozen_q.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
        print("frozen q -> %s: %s" % (out_root / "frozen_q.json", frozen))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

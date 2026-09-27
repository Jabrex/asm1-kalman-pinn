"""Regime map (plan E8, Fig. 6): score every v1.1 cell on windows R0, R2 and F.

Reads saved artefacts only (summary.json + predictions.npz); nothing is trained.

    python -m scripts.regime_map score --root results/v11 --legacy-lstm results/runs
    python -m scripts.regime_map plot --root results/v11 --paper-dir paper/figures

Layout consumed (plan G6):
    results/v11/{pinn,observers,baselines}/<cell>[_seed<k>]/<run_id>[_r<NN>]/
Cell grammar: k<KKK>[_off|off]_<ie|ic|ib>_<a0|as|al1|al2>[_<extra>] or rand<NN>_ie_a0.
The config-only suffixes _lstm, _theta and _sensors are merged into their base
cell; the estimator identity comes from summary["model"] and summary["q_mode"].
A PINN run with a sensor variant (E3) is filed under the observer cell of the
same sensor set, <cell>_<variant>, so it is compared with estimators that saw
the same sensors.

Decision rules (PREREGISTRATION.md, Section 9): a PINN model against a
single-valued row wins when every seed is strictly below it; two PINN models are
compared seed by seed (pairs by seed number); a diverged or failed run loses to
every comparator and a comparison whose comparator diverged or failed is not
decided; rows marked * use more information and never win; two-seed and
one-seed PINN rows are labelled and do not decide H1-H7. The hypotheses H1-H5
and H7 are evaluated as written in Section 3 ("hypotheses" in regime_table.json);
H6 comes from scripts/recoverability_validation.py.

Outputs (under --root):
    regime_table.json   rows, comparisons, winners, crossovers, parameter recovery,
                        realisation spread, hypotheses (every manuscript number of the regime map)
    regime_table.csv    one line per (cell, sigma, estimator, window)
    regime_table.md     pivots per window and the win/tie/loss counts
    regime_states.json  per (tank, never-measured component) NRMSE for windows R0 and F
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.asm1.vault_loader import vault  # noqa: E402
from src.data.sensors import unobserved_components  # noqa: E402
from src.eval.metrics import gap_closed, per_tank_nrmse, skill_score, state_metrics  # noqa: E402
from src.eval.report import (  # noqa: E402
    LEVEL_COMPONENTS,
    TOLERANCE_COMPONENTS,
    save_figure,
    score_arrays,
    window_pairs,
)

FAMILIES = ("pinn", "observers", "baselines")
WINDOWS = ("R0", "R2", "F")
WINDOW_OF_SET = {"train": "R0", "R2": "R2", "holdout": "F"}
EXTRA_WINDOWS = {"R2": ("train", 2.0, 12.0)}
STATE_WINDOWS = ("R0", "F")
PERSISTENCE = "persistence"
REFERENCE = "ode_openloop_reduced"
PINN_MODELS = ("cl_pinn", "pinn", "cl_pinn_theta")
#: Rows that use information the PINN does not get; reported, never winners.
MORE_INFORMATION = ("ekf_online", "ode_openloop_full")
ESTIMATOR_ORDER = (
    "persistence", "ode_openloop_reduced", "ode_openloop_full", "lstm_v10", "lstm",
    "pinn", "cl_pinn", "cl_pinn_theta", "ekf", "eks", "ieks", "ekf_aug", "eks_aug", "ekf_online",
)
PRETTY = {
    "persistence": "Persistence",
    "ode_openloop_reduced": "Open loop (info-matched)",
    "ode_openloop_full": "Open loop (structure-rich)*",
    "lstm_v10": "LSTM (earlier run)",
    "lstm": "LSTM",
    "pinn": "PINN, single-stage",
    "cl_pinn": "CL-PINN",
    "cl_pinn_theta": "PINN-$\\theta$",
    "ekf": "EKF",
    "eks": "EKS",
    "ieks": "IEKS",
    "ekf_aug": "Aug. EKF",
    "eks_aug": "Aug. EKS",
    "ekf_online": "Online EKF*",
}
MERGED_SUFFIXES = ("_lstm", "_theta", "_sensors")
DATA_DIR_BY_K = {
    "000": "results/raw",
    "025": "results/raw_k025",
    "050": "results/raw_k050",
    "075": "results/raw_k075",
    "100": "results/raw_k100",
}
INFLUENT_TAG = {"exact": "ie", "composite": "ic", "composite_biased": "ib"}
INFLUENT_ORDER = ("ie", "ic", "ib")
ANCHOR_ORDER = ("a0", "as", "al1", "al2")
#: Observer summary keys copied as diagnostics (q_table, the 49-entry tuning scan, stays in summary.json).
DIAGNOSTIC_KEY = re.compile(r"^(q_(?!table)|selected_q|r_|nis|log_?lik|diverg|finite|ieks_|n_measured)",
                            re.IGNORECASE)
#: Divergence (Section 5): non-finite estimate, self-flagged, or mean NIS over days 0-12 > 3 x channels.
NIS_FACTOR = 3.0
SEED_SUFFIX = re.compile(r"_seed(\d+)$")
REALISATION = re.compile(r"_r(\d{2})$")
KCELL = re.compile(
    r"^k(?P<k>\d{3})(?P<off>_?off)?_(?P<i>ie|ic|ib)_(?P<a>a0|as|al1|al2)(?:_(?P<extra>.+))?$"
)
SIGMA_IN_NAME = re.compile(r"_sigma(\d+)p(\d{2})")
#: Sensor cells (E3 variants): persistence and the open loop do not read the dropped or added
#: channels, so they come from the base cell; the ideal-settler cell borrows persistence only.
SENSOR_EXTRA = re.compile(r"^(drop|add)-")
RAND = re.compile(r"^rand(?P<idx>\d{2,3})_ie_a0$")
RULE = (
    "A PINN estimator wins a window when every seed has a strictly lower Track B error than a "
    "single-valued comparator and loses when every seed is strictly higher; two PINN models are "
    "compared seed by seed (pairs by seed number); against the multi-seed LSTM every seed must lie "
    "below every LSTM seed. Otherwise the outcome is a tie. A diverged or failed run loses to every "
    "comparator; a comparison whose comparator diverged or failed is not decided. Observers are "
    "compared on the noise realization the PINN was trained on (r = 0); rows marked * use more "
    "information and are never winners; two-seed and one-seed PINN rows are labeled and do not "
    "decide H1-H7."
)
#: Hypothesis settings (Section 3): window R0, sigma 0.10, realisation 0, the primary metric.
H_WINDOW = "R0"
H_SIGMA = 0.10
FORCING_COMPONENTS = ("S_S", "X_S", "S_ND", "X_ND", "S_I")
BIOMASS_COMPONENTS = ("X_B_H", "X_B_A")
THETA_SANITY = (0.8, 1.25)
H2_RANGE = (0.25, 0.75)
TRACK_B = unobserved_components()
_COMPONENTS = vault().components
B_IDX = [_COMPONENTS.index(c) for c in TRACK_B]


# -- cells and estimators -----------------------------------------------------
def parse_cell(dirname: str) -> tuple[dict[str, Any], int | None]:
    """Split a cell directory name into its grid coordinates and a seed suffix."""
    seed = None
    m = SEED_SUFFIX.search(dirname)
    if m:
        seed = int(m.group(1))
        dirname = dirname[: m.start()]
    for suffix in MERGED_SUFFIXES:
        if dirname.endswith(suffix):
            dirname = dirname[: -len(suffix)]
            break
    info: dict[str, Any] = {
        "cell": dirname, "k": None, "offsteady": False, "influent": None,
        "anchor": None, "extra": "", "rand": None,
    }
    km = KCELL.match(dirname)
    if km:
        info.update(
            k=km.group("k"), offsteady=bool(km.group("off")), influent=km.group("i"),
            anchor=km.group("a"), extra=km.group("extra") or "",
        )
    rm = RAND.match(dirname)
    if rm:
        info.update(rand=int(rm.group("idx")), influent="ie", anchor="a0")
    return info, seed


def is_grid_cell(info: dict[str, Any]) -> bool:
    """Pre-registered grid cell: K x I x A, no extra suffix, not a random truth."""
    return info["k"] is not None and not info["extra"] and info["rand"] is None


def kinetic_coordinate(preset: str | None, alpha: float | None) -> float:
    """Position on the mismatch axis: 0 = vault 20 C truth, 1 = BSM1 15 C truth."""
    if preset in (None, "vault20"):
        return 0.0
    if preset == "bsm1_15c":
        return 1.0
    return float(alpha)


def estimator_label(summary: dict[str, Any], with_variant: bool = True) -> str:
    label = str(summary["model"])
    if summary.get("q_mode") == "frozen" and "frozen" not in label:
        label += "_frozenq"
    variant = summary.get("variant") or ""
    if variant and with_variant:
        label += "[%s]" % variant
    return label


def base_model(label: str) -> str:
    return label.split("[")[0].replace("_frozenq", "")


def is_primary(label: str) -> bool:
    """Information-matched headline rows (no variant, no frozen q, not more-information)."""
    return "[" not in label and not label.endswith("_frozenq") and base_model(label) not in MORE_INFORMATION


def divergence(summary: dict[str, Any]) -> str | None:
    """Reason an observer run counts as diverged (Section 5), or None."""
    if summary.get("finite") is False:
        return "non-finite estimate"
    if summary.get("diverged"):
        return "filter flagged divergence"
    nis, n = summary.get("nis_mean"), summary.get("n_measured")
    if nis is not None and n and float(nis) > NIS_FACTOR * float(n):
        return "mean NIS %.3g > %g x %d channels" % (float(nis), NIS_FACTOR, int(n))
    return None


def data_dir_for(summary: dict[str, Any], info: dict[str, Any]) -> Path:
    if summary.get("data_dir"):
        return Path(summary["data_dir"])
    if info["rand"] is not None:
        for name in (str(info["rand"]), "%02d" % info["rand"], "%03d" % info["rand"]):
            candidate = Path("results/raw_rand") / name
            if candidate.exists():
                return candidate
        raise FileNotFoundError("no data directory for random-mismatch cell %s" % info["cell"])
    if info["k"] is None:
        raise ValueError(
            "cell %r does not follow the cell grammar and its summary has no data_dir" % info["cell"]
        )
    if info["offsteady"]:
        return Path("results/raw_k%s_off" % info["k"])
    return Path(DATA_DIR_BY_K[info["k"]])


def check_consistency(summary: dict[str, Any], info: dict[str, Any], run_dir: Path) -> None:
    """Refuse a run whose summary contradicts the cell it was filed under."""
    if info["k"] is None:
        return
    if "truth_preset" in summary:
        coordinate = kinetic_coordinate(summary["truth_preset"], summary.get("alpha"))
        if abs(coordinate - int(info["k"]) / 100.0) > 1e-9:
            raise ValueError("%s: truth %s/alpha %s filed under cell %s"
                             % (run_dir, summary["truth_preset"], summary.get("alpha"), info["cell"]))
    mode = summary.get("influent_mode")
    if mode is not None and INFLUENT_TAG.get(mode) != info["influent"]:
        raise ValueError("%s: influent_mode %s filed under cell %s" % (run_dir, mode, info["cell"]))
    anchor = summary.get("anchor")
    if anchor is not None and str(anchor).lower() != info["anchor"]:
        raise ValueError("%s: anchor %s filed under cell %s" % (run_dir, anchor, info["cell"]))


@dataclass
class RunRef:
    family: str
    info: dict[str, Any]
    run_dir: Path
    data_dir: Path
    summary: dict[str, Any]
    estimator: str
    realisation: int

    @property
    def cell(self) -> str:
        return self.info["cell"]


def discover(root: Path, legacy_lstm: Path | None = None) -> list[RunRef]:
    refs: list[RunRef] = []
    for family in FAMILIES:
        base = Path(root) / family
        if not base.exists():
            continue
        for cell_dir in sorted(p for p in base.iterdir() if p.is_dir() and not p.name.startswith("_")):
            info, _ = parse_cell(cell_dir.name)
            for run_dir in sorted(p for p in cell_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
                if not ((run_dir / "summary.json").exists() and (run_dir / "predictions.npz").exists()):
                    continue
                summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
                check_consistency(summary, info, run_dir)
                m = REALISATION.search(run_dir.name)
                run_info, label = info, estimator_label(summary)
                variant = summary.get("variant") or ""
                if family == "pinn" and variant:
                    # E3: the variant's sensor set is the observer cell <cell>_<variant>.
                    run_info, _ = parse_cell("%s_%s" % (info["cell"], variant))
                    label = estimator_label(summary, with_variant=False)
                refs.append(RunRef(family, run_info, run_dir, data_dir_for(summary, info), summary,
                                   label, int(m.group(1)) if m else 0))
    if legacy_lstm is not None and Path(legacy_lstm).exists():
        info, _ = parse_cell("k000_ie_a0")
        for run_dir in sorted(Path(legacy_lstm).glob("lstm_sigma*")):
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            refs.append(RunRef("legacy", info, run_dir, Path("results/raw"), summary, "lstm_v10", 0))
    return refs


def infrastructure_crashes(root: Path) -> list[str]:
    """run_core keeps a crashed attempt as <run_id>__crash<N>; it is rerun, not a failed run (Section 11)."""
    return sorted(str(p).replace("\\", "/") for family in FAMILIES
                  for p in (Path(root) / family).glob("*/*__crash*") if p.is_dir())


def observer_job_errors(root: Path) -> list[str]:
    """Failed observer jobs: scripts/run_observers.py writes error_<cell>_sigma<tag>_rNN_<q_mode>.txt."""
    base = Path(root) / "observers"
    return sorted(str(p).replace("\\", "/") for pattern in ("error_*.txt", "*/error_*.txt")
                  for p in base.glob(pattern)) if base.exists() else []


def failed_runs(root: Path) -> list[dict[str, Any]]:
    """Run directories with an error.txt and no scored output (they lose to every comparator)."""
    out = []
    for family in FAMILIES:
        base = Path(root) / family
        if not base.exists():
            continue
        for error in sorted(base.glob("*/*/error.txt")):
            run_dir = error.parent
            if "__crash" in run_dir.name:
                continue  # an infrastructure attempt that was rerun (infrastructure_crashes)
            if (run_dir / "summary.json").exists() and (run_dir / "predictions.npz").exists():
                continue
            info, seed = parse_cell(run_dir.parent.name)
            m = SIGMA_IN_NAME.search(run_dir.name)
            if m is None:
                raise ValueError("%s: cannot read sigma from the run directory name" % run_dir)
            model = run_dir.name[: m.start()]
            r = REALISATION.search(run_dir.name)
            variant = run_dir.name[m.end(): r.start() if r else None].lstrip("_")
            if family == "pinn" and variant:
                info, _ = parse_cell("%s_%s" % (info["cell"], variant))
            out.append({"family": family, "info": info, "cell": info["cell"], "estimator": model,
                        "sigma": round(float("%s.%s" % m.groups()), 6), "seed": seed or 0,
                        "realisation": int(r.group(1)) if r else 0,
                        "run_dir": str(run_dir).replace("\\", "/"),
                        "error": error.read_text(encoding="utf-8", errors="replace").strip()[:300]})
    return out


# -- scoring --------------------------------------------------------------------
def score_refs(refs: list[RunRef], metric: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for ref in refs:
        summary, pairs = window_pairs(ref.run_dir, ref.data_dir, EXTRA_WINDOWS)
        for label, truth, pred, spread in pairs:
            window = WINDOW_OF_SET.get(label)
            if window is None:
                continue
            row = score_arrays(summary, label, truth, pred, spread)
            fixed = state_metrics(truth, pred, spread=spread).nrmse
            value = float(row[metric])
            reason = divergence(summary) if ref.family == "observers" else None
            if reason is None and not math.isfinite(value):
                reason = "non-finite score"
            record = {
                "info": ref.info,
                "cell": ref.cell,
                "family": ref.family,
                "estimator": ref.estimator,
                "sigma": round(float(summary["noise"]), 6),
                "seed": int(summary.get("seed", 0)),
                "realisation": ref.realisation,
                "window": window,
                "value": value,
                "diverged": reason,
                "failed": False,
                # Section 9: a diverged or failed run loses to every comparator.
                "decision_value": math.inf if reason else value,
                "level_error": {c: row["level_error_%s" % c] for c in LEVEL_COMPONENTS},
                "tol10": {c: row["tol10_%s" % c] for c in TOLERANCE_COMPONENTS},
                "per_component_fixed": {c: float(fixed[i]) for c, i in zip(TRACK_B, B_IDX)},
                "learned_multipliers": summary.get("learned_multipliers"),
                "diagnostics": {k: v for k, v in summary.items() if DIAGNOSTIC_KEY.match(k)}
                if ref.family == "observers" else {},
                "data_dir": str(ref.data_dir).replace("\\", "/"),
                "run_dir": str(ref.run_dir).replace("\\", "/"),
            }
            if window in STATE_WINDOWS:
                record["per_state"] = per_tank_nrmse(truth, pred, spread=spread)[:, B_IDX]
            records.append(record)
    return records


def failed_records(failed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One record per window for each failed run: no score, and it loses to every comparator."""
    nan = float("nan")
    return [{**{k: f[k] for k in ("info", "cell", "family", "estimator", "sigma", "seed", "realisation",
                                   "run_dir")},
             "window": window, "value": math.inf, "diverged": "failed: " + (f["error"] or "error.txt"),
             "failed": True, "decision_value": math.inf,
             "level_error": {c: nan for c in LEVEL_COMPONENTS}, "tol10": {c: nan for c in TOLERANCE_COMPONENTS},
             "per_component_fixed": {c: nan for c in TRACK_B}, "learned_multipliers": None, "diagnostics": {},
             "data_dir": None}
            for f in failed for window in WINDOWS]


def _cell_sort_key(info: dict[str, Any]) -> tuple:
    return (
        info["rand"] is not None,
        info["rand"] if info["rand"] is not None else -1,
        info["k"] or "",
        info["offsteady"],
        INFLUENT_ORDER.index(info["influent"]) if info["influent"] in INFLUENT_ORDER else 9,
        ANCHOR_ORDER.index(info["anchor"]) if info["anchor"] in ANCHOR_ORDER else 9,
        info["extra"],
    )


def _estimator_sort_key(label: str) -> tuple:
    model = base_model(label)
    order = ESTIMATOR_ORDER.index(model) if model in ESTIMATOR_ORDER else len(ESTIMATOR_ORDER)
    return (order, label)


def aggregate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Median over seeds (PINN, LSTM) on noise realisation 0."""
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for r in records:
        if r["realisation"] == 0:
            groups.setdefault((r["cell"], r["sigma"], r["estimator"], r["window"]), []).append(r)
    rows = []
    for key in sorted(groups, key=lambda k: (_cell_sort_key(groups[k][0]["info"]), k[1],
                                              _estimator_sort_key(k[2]), WINDOWS.index(k[3]))):
        members = sorted(groups[key], key=lambda r: r["seed"])
        cell, sigma, estimator, window = key
        info = members[0]["info"]
        values = [m["value"] for m in members]
        rows.append({
            "cell": cell,
            "k": info["k"],
            "offsteady": info["offsteady"],
            "influent": info["influent"],
            "anchor": info["anchor"],
            "extra": info["extra"],
            "rand": info["rand"],
            "kinetic_alpha": int(info["k"]) / 100.0 if info["k"] is not None else None,
            "sigma": sigma,
            "estimator": estimator,
            "family": members[0]["family"],
            "window": window,
            "primary": is_primary(estimator),
            "n": len(values),
            "seeds": [m["seed"] for m in members],
            "values": values,
            "decision_values": [m["decision_value"] for m in members],
            "diverged": [m["diverged"] for m in members if m["diverged"]],
            "failed": any(m["failed"] for m in members),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "level_error": {c: float(np.nanmedian([m["level_error"][c] for m in members]))
                            if any(np.isfinite(m["level_error"][c]) for m in members) else float("nan")
                            for c in LEVEL_COMPONENTS},
            "tol10": {c: float(np.nanmedian([m["tol10"][c] for m in members]))
                      if any(np.isfinite(m["tol10"][c]) for m in members) else float("nan")
                      for c in TOLERANCE_COMPONENTS},
            "per_component_fixed": {
                c: float(np.nanmedian([m["per_component_fixed"][c] for m in members]))
                if any(np.isfinite(m["per_component_fixed"][c]) for m in members) else float("nan")
                for c in TRACK_B
            },
            "per_seed_component_fixed": {m["seed"]: dict(m["per_component_fixed"]) for m in members},
            "multipliers_by_seed": {m["seed"]: {k: float(v) for k, v in m["learned_multipliers"].items()}
                                    for m in members if isinstance(m.get("learned_multipliers"), dict)
                                    and m["learned_multipliers"]},
            "diagnostics": members[0]["diagnostics"],
            "data_dir": members[0]["data_dir"],
        })
    return rows


def reference_cell(info: dict[str, Any]) -> tuple[str, str]:
    """(cell for persistence, cell for the information-matched open loop) of a cell.

    Sensor cells (E3) share the base cell's anchor, influent and RAS input, so both
    references come from the base cell. The ideal-settler cell has no RAS sensor, so
    only persistence (the anchor mean) carries over; the open loop would use more
    information than its estimators.
    """
    extra = info.get("extra") or ""
    if extra and info["k"] is not None:
        base = info["cell"][: -len(extra) - 1]
        if SENSOR_EXTRA.match(extra):
            return base, base
        if extra == "ideal":
            return base, info["cell"]
    return info["cell"], info["cell"]


def attach_references(rows: list[dict[str, Any]]) -> None:
    """Skill against anchor-aware persistence; gap closed against the info-matched open loop."""
    index = {(r["cell"], r["sigma"], r["estimator"], r["window"]): r for r in rows}
    for r in rows:
        p_cell, ref_cell = reference_cell(parse_cell(r["cell"])[0])
        persist = index.get((p_cell, r["sigma"], PERSISTENCE, r["window"]))
        if persist is None:
            # Persistence holds the anchor mean and never reads a sensor, so its
            # content is identical at every sigma (make_baselines docstring).
            persist = next((p for p in rows if p["cell"] == p_cell and p["estimator"] == PERSISTENCE
                            and p["window"] == r["window"]), None)
        ref = index.get((ref_cell, r["sigma"], REFERENCE, r["window"]))
        r["persistence"] = persist["median"] if persist else None
        r["reference"] = ref["median"] if ref else None
        if persist:
            r["skill"] = float(skill_score(r["median"], persist["median"]))
            r["skill_values"] = [float(skill_score(v, persist["median"])) for v in r["values"]]
        else:
            r["skill"], r["skill_values"] = None, None
        if persist and ref and abs(persist["median"] - ref["median"]) > 1e-12:
            r["gap_closed"] = float(gap_closed(r["median"], persist["median"], ref["median"]))
        else:
            r["gap_closed"] = None


def outcome(a: list[float], b: list[float]) -> str:
    """All-seeds rule: win if every a is below every b, loss if every a is above.

    With a single-valued comparator this is the Section 9 PINN rule.
    """
    if max(a) < min(b):
        return "win"
    if min(a) > max(b):
        return "loss"
    return "tie"


def paired_outcome(a: dict[int, float], b: dict[int, float]) -> str:
    """Two PINN models, seeds paired by number: win if every pair favours a, loss if every pair favours b."""
    common = sorted(set(a) & set(b))
    if not common:
        return "not_decided"
    if all(a[s] < b[s] for s in common):
        return "win"
    if all(a[s] > b[s] for s in common):
        return "loss"
    return "tie"


def single_outcome(a: float, b: float) -> str:
    """Two single-valued estimators: the strictly lower value wins, equal values tie."""
    return "win" if a < b else ("loss" if a > b else "tie")


def is_pinn_row(r: dict[str, Any]) -> bool:
    return r["family"] == "pinn" and base_model(r["estimator"]) in PINN_MODELS


def seed_label(r: dict[str, Any]) -> str:
    """Section 9 label of a PINN row with fewer than three seeds."""
    if not is_pinn_row(r) or r["n"] >= 3:
        return ""
    return "one-seed" if r["n"] == 1 else "two-seed"


ADDED_RULE = "added rule (not registered): every seed against every seed"
REFERENCE_RULE = "comparator uses more information: shown for reference, never a winner"


def _compare_core(p: dict[str, Any], other: dict[str, Any]) -> tuple[str, str]:
    if other["failed"] or other["diverged"]:
        return "not_decided", "comparator diverged or failed"
    if is_pinn_row(p) and is_pinn_row(other):
        return (paired_outcome(dict(zip(p["seeds"], p["decision_values"])),
                               dict(zip(other["seeds"], other["decision_values"]))), "paired seeds")
    if other["n"] == 1:
        if p["n"] == 1:
            return single_outcome(p["decision_values"][0], other["decision_values"][0]), "single values"
        return outcome(p["decision_values"], other["decision_values"]), "every seed against one value"
    # Section 9 covers single-valued rows and two PINN models; the multi-seed LSTM is not covered.
    return outcome(p["decision_values"], other["decision_values"]), ADDED_RULE


def compare_rows(p: dict[str, Any], other: dict[str, Any]) -> tuple[str, str]:
    """(outcome, rule) of row p against row other under the Section 9 rules.

    Against a more-information row (ode_openloop_full, the online EKF) the outcome is
    'reference': Section 9 shows those rows for reference and they never win.
    """
    if base_model(other["estimator"]) in MORE_INFORMATION:
        return "reference", REFERENCE_RULE
    return _compare_core(p, other)


def comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    slots: dict[tuple, list[dict[str, Any]]] = {}
    for r in rows:
        slots.setdefault((r["cell"], r["sigma"], r["window"]), []).append(r)
    out = []
    for (cell, sigma, window), members in slots.items():
        for p in members:
            if not is_pinn_row(p):
                continue
            for other in members:
                if other is p:
                    continue
                result, rule = compare_rows(p, other)
                labels = sorted({x for x in (seed_label(p), seed_label(other)) if x})
                entry = {
                    "cell": cell, "sigma": sigma, "window": window,
                    "pinn": p["estimator"], "comparator": other["estimator"],
                    "outcome": result, "rule": rule, "label": ", ".join(labels),
                    "decides_hypotheses": not labels and result != "reference" and rule != ADDED_RULE,
                    "pinn_values": p["values"], "comparator_values": other["values"],
                }
                if result == "reference":
                    entry["reference_outcome"] = _compare_core(p, other)[0]
                out.append(entry)
    return out


def comparison_kind(c: dict[str, Any]) -> str:
    """'deciding', 'labelled' (seed label or added rule) or 'reference' (more-information comparator)."""
    if c["outcome"] == "reference":
        return "reference"
    return "deciding" if c["decides_hypotheses"] else "labelled"


def winners(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    slots: dict[tuple, list[dict[str, Any]]] = {}
    for r in rows:
        if r["primary"]:
            slots.setdefault((r["cell"], r["sigma"], r["window"]), []).append(r)
    out = []
    for (cell, sigma, window), members in slots.items():
        eligible = [r for r in members if not (r["failed"] or r["diverged"])]
        if not eligible:
            continue
        best = min(eligible, key=lambda r: r["median"])
        tied = [r["estimator"] for r in eligible if r is not best and r["median"] == best["median"]]
        out.append({"cell": cell, "sigma": sigma, "window": window,
                    "estimator": best["estimator"], "median": best["median"], "tied_with": tied,
                    "n": best["n"], "label": seed_label(best)})
    return out


def crossover_alpha(points: list[tuple[float, float]]) -> tuple[float | None, str]:
    """First alpha at which err - err_persist turns positive, by linear interpolation."""
    if not points:
        return None, "no_data"
    points = sorted(points)
    for (a0, d0), (a1, d1) in zip(points, points[1:]):
        if d0 <= 0 < d1:
            return a0 + (a1 - a0) * (-d0) / (d1 - d0), "crosses"
    if all(d > 0 for _, d in points):
        return None, "behind_at_first_alpha"
    if all(d <= 0 for _, d in points):
        return None, "never_behind"
    return None, "no_negative_to_positive_change"


def crossovers(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """alpha* per (influent, anchor, sigma, window, estimator) along the K axis (grid cells)."""
    grid = [r for r in rows if r["k"] is not None and not r["extra"] and r["rand"] is None
            and not r["offsteady"]]
    out = []
    keys = sorted({(r["influent"], r["anchor"], r["sigma"], r["window"]) for r in grid})
    for influent, anchor, sigma, window in keys:
        sub = [r for r in grid if (r["influent"], r["anchor"], r["sigma"], r["window"])
               == (influent, anchor, sigma, window)]
        persist = {r["kinetic_alpha"]: r["median"] for r in sub if r["estimator"] == PERSISTENCE}
        for estimator in sorted({r["estimator"] for r in sub if r["primary"]} - {PERSISTENCE},
                                key=_estimator_sort_key):
            pts = [(r["kinetic_alpha"], r["median"] - persist[r["kinetic_alpha"]]) for r in sub
                   if r["estimator"] == estimator and r["kinetic_alpha"] in persist]
            if len(pts) < 2:
                continue
            alpha_star, status = crossover_alpha(pts)
            out.append({"influent": influent, "anchor": anchor, "sigma": sigma, "window": window,
                        "estimator": estimator, "alpha_star": alpha_star, "status": status,
                        "points": sorted(pts)})
    return out


def _meta_parameters(data_dir: Path) -> dict[str, float]:
    sim = Path(data_dir) / "sim_dry.npz"
    candidates = [sim] if sim.exists() else sorted(Path(data_dir).glob("obs_dry_sigma*.npz"))
    if not candidates:
        raise FileNotFoundError("no dataset with meta under %s" % data_dir)
    with np.load(candidates[0]) as data:
        return dict(json.loads(str(data["meta"]))["parameters"])


def parameter_recovery(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Learned kinetic multipliers against the true log ratio truth/vault (window R0 rows)."""
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for r in records:
        if r["window"] == "R0" and r["realisation"] == 0 and r["learned_multipliers"] and not r["failed"]:
            groups.setdefault((r["cell"], r["sigma"], r["estimator"]), []).append(r)
    base = vault().parameters
    out = []
    for (cell, sigma, estimator), members in sorted(groups.items()):
        names = sorted(members[0]["learned_multipliers"])
        truth = _meta_parameters(Path(members[0]["data_dir"]))
        true_log = {n: (math.log(float(truth[n]) / float(base[n]))
                        if float(truth[n]) > 0 and float(base[n]) > 0 else None) for n in names}
        estimates = []
        for m in sorted(members, key=lambda r: r["seed"]):
            logm = {n: math.log(float(m["learned_multipliers"][n])) for n in names}
            estimates.append({"seed": m["seed"], "log_multiplier": logm,
                              # Section 3, H3: |ln(multiplier) - ln(truth / vault)| per name (secondary)
                              "abs_log_error": {n: abs(logm[n] - true_log[n]) if true_log[n] is not None else None
                                                for n in names}})
        median_err = {n: float(np.median([e["abs_log_error"][n] for e in estimates]))
                      if true_log[n] is not None else None for n in names}
        out.append({
            "cell": cell, "sigma": sigma, "estimator": estimator, "names": names,
            "true_log_ratio": true_log, "estimates": estimates, "n": len(estimates),
            "median_abs_log_error": median_err,
        })
    return out


def realisation_spread(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for r in records:
        if r["family"] == "observers" and not r["failed"]:
            groups.setdefault((r["cell"], r["sigma"], r["estimator"], r["window"]), []).append(r)
    out = []
    for (cell, sigma, estimator, window), members in sorted(groups.items()):
        extra = sorted((m for m in members if m["realisation"] > 0), key=lambda r: r["realisation"])
        if not extra:
            continue
        # Sections 2 and 9: the spread over realisations 1-9 sits next to realisation 0, never pooled with it.
        values = np.array([m["value"] for m in extra])
        r0 = [m["value"] for m in members if m["realisation"] == 0]
        r0_value = r0[0] if r0 else None
        out.append({"cell": cell, "sigma": sigma, "estimator": estimator, "window": window,
                    "realisations": [m["realisation"] for m in extra],
                    "n": int(values.size), "median": float(np.median(values)),
                    "min": float(values.min()), "max": float(values.max()),
                    "p25": float(np.percentile(values, 25)), "p75": float(np.percentile(values, 75)),
                    "realisation0": r0_value,
                    "realisation0_outside_range": None if r0_value is None
                    else bool(r0_value < values.min() or r0_value > values.max())})
    return out


LAB_EXTRA = re.compile(r"^lab\d+$")


def lab_seed_spread(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Section 2: the nine further laboratory seeds, next to the seed-20260923 cell and never pooled with it."""
    index = {(r["cell"], r["sigma"], r["estimator"], r["window"]): r for r in rows}
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for r in rows:
        if r["extra"] and LAB_EXTRA.match(r["extra"]):
            base = r["cell"][: -len(r["extra"]) - 1]
            groups.setdefault((base, r["sigma"], r["estimator"], r["window"]), []).append(r)
    out = []
    for (base, sigma, estimator, window), members in sorted(groups.items()):
        members = sorted(members, key=lambda r: r["extra"])
        values = np.array([m["median"] for m in members])
        ref = index.get((base, sigma, estimator, window))
        ref_value = ref["median"] if ref else None
        out.append({"cell": base, "sigma": sigma, "estimator": estimator, "window": window,
                    "lab_seeds": [m["extra"] for m in members], "n": int(values.size),
                    "median": float(np.median(values)), "min": float(values.min()), "max": float(values.max()),
                    "p25": float(np.percentile(values, 25)), "p75": float(np.percentile(values, 75)),
                    "registered_seed_value": ref_value,
                    "registered_seed_outside_range": None if ref_value is None
                    else bool(ref_value < values.min() or ref_value > values.max())})
    return out


def state_entries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[np.ndarray]] = {}
    for r in records:
        if r["realisation"] == 0 and "per_state" in r:
            groups.setdefault((r["cell"], r["sigma"], r["estimator"], r["window"]), []).append(r["per_state"])
    return [{"cell": c, "sigma": s, "estimator": e, "window": w, "n": len(v),
             "median": np.median(np.stack(v), axis=0).tolist()}
            for (c, s, e, w), v in sorted(groups.items())]


# -- hypotheses (PREREGISTRATION.md, Section 3) ------------------------------------------------
def _find(rows: list[dict[str, Any]], cell: str, estimator: str, window: str = H_WINDOW,
          sigma: float = H_SIGMA) -> dict[str, Any] | None:
    for r in rows:
        if (r["cell"], r["estimator"], r["window"]) == (cell, estimator, window) and abs(r["sigma"] - sigma) < 1e-9:
            return r
    return None


def _versus(a: dict[str, Any] | None, b: dict[str, Any] | None) -> dict[str, Any]:
    if a is None or b is None:
        return {"outcome": "missing", "rule": None, "a": None, "b": None}
    if a["failed"] or a["diverged"] or b["failed"] or b["diverged"]:
        # Section 9: a diverged or failed run loses; a failed comparator leaves the cell undecided.
        result, rule = compare_rows(a, b)
    elif is_pinn_row(a) and is_pinn_row(b):
        result, rule = compare_rows(a, b)
    elif a["n"] == 1 and b["n"] == 1:
        result, rule = single_outcome(a["decision_values"][0], b["decision_values"][0]), "single values"
    else:
        result, rule = compare_rows(a, b)
    return {"outcome": result, "rule": rule, "a": {"estimator": a["estimator"], "cell": a["cell"], "values": a["values"]},
            "b": {"estimator": b["estimator"], "cell": b["cell"], "values": b["values"]}}


def _group_ratio(num: dict[str, float], den: dict[str, float], comps: tuple[str, ...]) -> float:
    """Mean over components of the per-component ratio num / den."""
    return float(np.mean([num[c] / den[c] for c in comps]))


def hypotheses(rows: list[dict[str, Any]], crossover_rows: list[dict[str, Any]], realistic_k: str | None,
               tau_by_component: dict[str, float] | None, tau_source: str | None) -> dict[str, Any]:
    """H1-H5 and H7 evaluated as written in Section 3 (H6: scripts/recoverability_validation.py)."""
    out: dict[str, Any] = {"window": H_WINDOW, "sigma": H_SIGMA, "realisation": 0,
                           "note": "Each test uses its registered cells; the outcome rules are those of Section 9."}
    # H1: smoother at least as accurate as the CL-PINN at the true kinetics.
    h1 = {}
    for cell in ("k000_ie_a0", "k000_ic_a0"):
        v = _versus(_find(rows, cell, "cl_pinn"), _find(rows, cell, "eks"))
        v["status"] = {"loss": "supported", "win": "refuted", "tie": "undecided"}.get(v["outcome"], "not decided")
        h1[cell] = v
    statuses = {v["status"] for v in h1.values()}
    # Section 3 gives H1 a word per cell; a single word is reported only when both cells agree.
    out["H1"] = {"cells": h1, "status": statuses.pop() if len(statuses) == 1 else "differs by cell",
                 "status_basis": "per cell (Section 3); one word only when the cells agree",
                 "summary": "; ".join("%s: CL-PINN vs EKS %s (%s)" % (c, v["outcome"], v["status"])
                                      for c, v in h1.items())}
    # H2: information-matched open loop falls behind persistence between alpha 0.25 and 0.75.
    cross = next((c for c in crossover_rows if (c["influent"], c["anchor"], c["window"], c["estimator"])
                  == ("ie", "a0", H_WINDOW, REFERENCE) and abs(c["sigma"] - H_SIGMA) < 1e-9), None)
    if cross is None:
        out["H2"] = {"status": "not decided", "summary": "no ie_a0 crossover data"}
    else:
        inside = cross["status"] == "crosses" and H2_RANGE[0] < cross["alpha_star"] < H2_RANGE[1]
        out["H2"] = {"alpha_star": cross["alpha_star"], "crossing_status": cross["status"], "points": cross["points"],
                     "status": "supported" if inside else "not supported",
                     "summary": "alpha* = %s (%s); supported if %.2f < alpha* < %.2f"
                                % (_fmt(cross["alpha_star"]), cross["status"], *H2_RANGE)}
    # H3: joint state-parameter estimation at K1-Ie-A0.
    aug = _versus(_find(rows, "k100_ie_a0", "eks_aug"), _find(rows, "k100_ie_a0", "eks"))
    theta = _versus(_find(rows, "k100_ie_a0", "cl_pinn_theta"), _find(rows, "k100_ie_a0", "cl_pinn"))
    tie = _versus(_find(rows, "k100_ie_a0", "cl_pinn_theta"), _find(rows, "k100_ie_a0", "eks_aug"))
    sanity_row = _find(rows, "k000_ie_a0", "cl_pinn_theta")
    sanity = None
    if sanity_row is not None:
        mults = {}
        for rec_seed, comps in sanity_row.get("multipliers_by_seed", {}).items():
            mults[rec_seed] = comps
        inside = all(THETA_SANITY[0] <= m <= THETA_SANITY[1] for comps in mults.values() for m in comps.values())
        sanity = {"multipliers": mults, "range": list(THETA_SANITY), "pass": bool(mults) and inside}
    if aug["outcome"] in ("not_decided", "missing") or theta["outcome"] in ("not_decided", "missing"):
        status3 = "not decided"
    else:
        status3 = "supported" if aug["outcome"] == "win" and theta["outcome"] == "win" else "not supported"
    out["H3"] = {"augmented_vs_plain_smoother": aug, "pinn_theta_vs_cl_pinn": theta,
                 "pinn_theta_vs_augmented_smoother_expected_tie": tie, "k0_theta_sanity": sanity,
                 "expected_tie_outcome": tie["outcome"], "expected_tie_met": tie["outcome"] == "tie",
                 "status_basis": "the two registered wins (Aug. EKS over EKS, PINN-theta over CL-PINN); the "
                                 "expected tie of PINN-theta and Aug. EKS is reported beside it",
                 "status": status3,
                 "summary": "Aug. EKS vs EKS %s; PINN-theta vs CL-PINN %s; PINN-theta vs Aug. EKS %s (expected tie)"
                            % (aug["outcome"], theta["outcome"], tie["outcome"])}
    # H4: the laboratory panel at K.5-Ic (or K1-Ic before gate D3) helps both families.
    if realistic_k is None:
        out["H4"] = {"status": "not decided", "summary": "no realistic cells found"}
    else:
        c_as, c_al1 = "k%s_ic_as" % realistic_k, "k%s_ic_al1" % realistic_k
        fam = {}
        for est in ("cl_pinn", "eks"):
            a, b = _find(rows, c_al1, est), _find(rows, c_as, est)
            v = _versus(a, b)
            if a is not None and b is not None and tau_by_component:
                comps = [c for c in TRACK_B if c in tau_by_component]
                red = [(b["per_component_fixed"][c] - a["per_component_fixed"][c]) / b["per_component_fixed"][c]
                       for c in comps]
                rho = float(spearmanr([tau_by_component[c] for c in comps], red).statistic)
                v.update(relative_reduction=dict(zip(comps, red)), spearman_tau_vs_reduction=rho)
            fam[est] = v
        lab_wins = all(fam[e]["outcome"] == "win" for e in fam)
        rhos = [fam[e].get("spearman_tau_vs_reduction") for e in fam]
        if any(fam[e]["outcome"] in ("not_decided", "missing") for e in fam) or any(r is None or not np.isfinite(r) for r in rhos):
            status = "not decided"
        else:
            status = "supported" if lab_wins and all(r > 0 for r in rhos) else "not supported"
        out["H4"] = {"cells": [c_al1, c_as], "families": fam, "tau_source": tau_source,
                     "tau": "median over the five tanks of tau_days, per never-measured component",
                     "reduction": "(E_As - E_Al1) / E_As per component, E = per-component NRMSE (fixed range); "
                                  "CL-PINN: median over seeds",
                     "status": status,
                     "summary": "; ".join("%s Al1 vs As %s, rho(tau, reduction) %s"
                                          % (e, fam[e]["outcome"], _fmt(fam[e].get("spearman_tau_vs_reduction"), "%+.2f"))
                                          for e in fam)}
    # H5: composite influent costs more for forcing-slaved states than for biomass inventories.
    fam5 = {}
    undecided5 = []
    ie, ic = _find(rows, "k000_ie_a0", "eks"), _find(rows, "k000_ic_a0", "eks")
    for r in (ie, ic, _find(rows, "k000_ie_a0", "cl_pinn"), _find(rows, "k000_ic_a0", "cl_pinn")):
        if r is not None and (r["failed"] or r["diverged"]):
            undecided5.append("%s %s diverged or failed" % (r["cell"], r["estimator"]))
    if ie is not None and ic is not None:
        a = _group_ratio(ic["per_component_fixed"], ie["per_component_fixed"], FORCING_COMPONENTS)
        b = _group_ratio(ic["per_component_fixed"], ie["per_component_fixed"], BIOMASS_COMPONENTS)
        fam5["eks"] = {"ratio_forcing": a, "ratio_biomass": b, "holds": bool(a > b)}
    ie, ic = _find(rows, "k000_ie_a0", "cl_pinn"), _find(rows, "k000_ic_a0", "cl_pinn")
    if ie is not None and ic is not None:
        pairs = sorted(set(ie["per_seed_component_fixed"]) & set(ic["per_seed_component_fixed"]))
        per_pair = {s: {"ratio_forcing": _group_ratio(ic["per_seed_component_fixed"][s],
                                                      ie["per_seed_component_fixed"][s], FORCING_COMPONENTS),
                        "ratio_biomass": _group_ratio(ic["per_seed_component_fixed"][s],
                                                      ie["per_seed_component_fixed"][s], BIOMASS_COMPONENTS)}
                    for s in pairs}
        if per_pair:
            a = float(np.median([v["ratio_forcing"] for v in per_pair.values()]))
            b = float(np.median([v["ratio_biomass"] for v in per_pair.values()]))
            fam5["cl_pinn"] = {"per_seed_pair": per_pair, "ratio_forcing": a, "ratio_biomass": b, "holds": bool(a > b)}
    if any(not (np.isfinite(v["ratio_forcing"]) and np.isfinite(v["ratio_biomass"])) for v in fam5.values()):
        undecided5.append("non-finite ratio")
    if len(fam5) < 2 or undecided5:
        status5 = "not decided"
    else:
        status5 = "supported" if all(v["holds"] for v in fam5.values()) else "not supported"
    out["H5"] = {"families": fam5, "forcing_components": list(FORCING_COMPONENTS),
                 "biomass_components": list(BIOMASS_COMPONENTS),
                 "ratio": "mean over the group of NRMSE(Ic) / NRMSE(Ie) per component, K0-A0; CL-PINN: per seed "
                          "pair, then the median over the pairs",
                 "status": status5, "not_decided_because": undecided5,
                 "summary": "; ".join("%s: forcing %.3g vs biomass %.3g" % (e, v["ratio_forcing"], v["ratio_biomass"])
                                      for e, v in fam5.items())}
    # H7: the CL-PINN beats the single-stage PINN at K0-Ie-A0 (seed-paired).
    v7 = _versus(_find(rows, "k000_ie_a0", "cl_pinn"), _find(rows, "k000_ie_a0", "pinn"))
    status7 = "supported" if v7["outcome"] == "win" else (
        "not decided" if v7["outcome"] in ("not_decided", "missing") else "not supported")
    out["H7"] = {**v7, "status": status7,
                 "consequence": "curriculum claim kept" if v7["outcome"] == "win"
                 else "curriculum claim withdrawn in full",
                 "summary": "CL-PINN vs PINN %s (%s)" % (v7["outcome"], v7["rule"])}
    return out


def realistic_cells_k(rows: list[dict[str, Any]]) -> str | None:
    """'050' when gate D3 moved the realistic PINN cells to K.5, '100' when they stayed at K1."""
    for k in ("050", "100"):
        if any(r["cell"] == "k%s_ic_as" % k and r["family"] == "pinn" for r in rows):
            return k
    return None


def tau_components(path: Path | None) -> dict[str, float] | None:
    """Median over tanks of tau_days per never-measured component (G4 recoverability file)."""
    if path is None or not Path(path).exists():
        return None
    rec = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for comp in TRACK_B:
        taus = [float(s["tau_days"]) for s in rec["states"] if s["component"] == comp]
        if taus:
            out[comp] = float(np.median(taus))
    return out


def _clean(obj: Any) -> Any:
    """JSON-safe copy: NaN and inf become null, numpy scalars become floats."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(float(obj)) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


CSV_FIELDS = (
    "cell", "k", "offsteady", "influent", "anchor", "extra", "rand", "kinetic_alpha", "sigma",
    "window", "estimator", "family", "primary", "n", "median", "min", "max", "skill",
    "gap_closed", "persistence", "reference",
)


def _csv_diverged(r: dict[str, Any]) -> str:
    return "failed" if r["failed"] else ("diverged" if r["diverged"] else "")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    extra = ["level_error_%s" % c for c in LEVEL_COMPONENTS] + ["tol10_%s" % c for c in TOLERANCE_COMPONENTS]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(CSV_FIELDS) + extra + ["diverged", "values"])
        for r in rows:
            writer.writerow(
                [r[f] for f in CSV_FIELDS]
                + [r["level_error"][c] for c in LEVEL_COMPONENTS]
                + [r["tol10"][c] for c in TOLERANCE_COMPONENTS]
                + [_csv_diverged(r), ";".join("%.6g" % v for v in r["values"])]
            )


def _fmt(value: float | None, pattern: str = "%.3f") -> str:
    return "n/a" if value is None else pattern % value


def markdown(table: dict[str, Any]) -> str:
    rows = table["rows"]
    lines = ["# Regime map (v1.1)", "", "Metric: `%s` (Track B, never-measured states). "
             "Cells show median over seeds, skill against anchor-aware persistence in brackets." % table["meta"]["metric"],
             "", table["meta"]["rule"], ""]
    grid = [r for r in rows if r["k"] is not None and not r["extra"] and r["rand"] is None]
    for sigma in sorted({r["sigma"] for r in grid}):
        for window in WINDOWS:
            sub = [r for r in grid if r["sigma"] == sigma and r["window"] == window]
            if not sub:
                continue
            cells = sorted({r["cell"] for r in sub}, key=lambda c: _cell_sort_key(parse_cell(c)[0]))
            ests = sorted({r["estimator"] for r in sub if r["primary"] or r["estimator"] in MORE_INFORMATION},
                          key=_estimator_sort_key)
            lookup = {(r["cell"], r["estimator"]): r for r in sub}
            lines += ["## sigma = %.2f, window %s" % (sigma, window), "",
                      "| cell | " + " | ".join(ests) + " |", "| --- |" + " --- |" * len(ests)]
            for c in cells:
                cells_txt = []
                for e in ests:
                    r = lookup.get((c, e))
                    cells_txt.append("-" if r is None else "%s (%s)" % (_fmt(r["median"]), _fmt(r["skill"], "%+.2f")))
                lines.append("| %s | %s |" % (c, " | ".join(cells_txt)))
            lines.append("")
    titles = {"deciding": "Win / tie / loss counts over cells (three-seed PINN rows, registered rules)",
              "labelled": "Labeled comparisons (two-seed or one-seed PINN rows, or the added LSTM rule); they do "
                          "not decide H1-H7",
              "reference": "Against more-information rows (* ; outcome the rule would give, never a win for them)"}
    for kind in ("deciding", "labelled", "reference"):
        counts: dict[tuple, dict[str, int]] = {}
        for c in table["comparisons"]:
            if comparison_kind(c) != kind:
                continue
            result = c["reference_outcome"] if kind == "reference" else c["outcome"]
            key = (c["window"], c["pinn"], c["comparator"])
            counts.setdefault(key, {"win": 0, "tie": 0, "loss": 0, "not_decided": 0})[result] += 1
        if not counts:
            continue
        lines += ["## %s" % titles[kind], "",
                  "| window | PINN estimator | comparator | win | tie | loss | not decided |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for (window, pinn, comp), n in sorted(counts.items(),
                                              key=lambda kv: (WINDOWS.index(kv[0][0]), kv[0][1], kv[0][2])):
            lines.append("| %s | %s | %s | %d | %d | %d | %d |"
                         % (window, pinn, comp, n["win"], n["tie"], n["loss"], n["not_decided"]))
        lines.append("")
    lines += ["", "## Crossover alpha* (error rises above persistence)", "",
              "| influent | anchor | sigma | window | estimator | alpha* | status |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for c in table["crossovers"]:
        lines.append("| %s | %s | %.2f | %s | %s | %s | %s |" % (c["influent"], c["anchor"], c["sigma"], c["window"],
                                                              c["estimator"], _fmt(c["alpha_star"]), c["status"]))
    hyp = table.get("hypotheses") or {}
    if hyp:
        lines += ["", "## Pre-registered hypotheses (window %s, sigma %.2f, realization 0)"
                  % (hyp.get("window", H_WINDOW), hyp.get("sigma", H_SIGMA)), "",
                  "| hypothesis | status | detail |", "| --- | --- | --- |"]
        for name in sorted(k for k in hyp if re.fullmatch(r"H\d", k)):
            lines.append("| %s | %s | %s |" % (name, hyp[name]["status"], hyp[name].get("summary", "")))
    diverged = table["meta"].get("diverged_runs") or []
    if diverged:
        lines += ["", "## Diverged or failed runs (Section 5 rule; they lose to every comparator)", ""]
        lines += ["- %s: %s" % (d["run_dir"], d["reason"]) for d in diverged]
    return "\n".join(lines) + "\n"


def score(root: Path, legacy_lstm: Path | None, metric: str,
          rec_realistic: Path | None = None) -> dict[str, Any]:
    refs = discover(root, legacy_lstm)
    if not refs:
        raise SystemExit("no scored runs under %s" % root)
    job_errors = observer_job_errors(root)
    if job_errors:
        # A failed observer job leaves no run directory to score; refuse rather than drop it (Section 11).
        raise ValueError("failed observer jobs must be reported, not dropped: %s" % ", ".join(job_errors))
    failed = failed_runs(root)
    records = score_refs(refs, metric) + failed_records(failed)
    rows = aggregate(records)
    attach_references(rows)
    diverged = sorted({(r["run_dir"], r["diverged"]) for r in records if r["diverged"]})
    crossing = crossovers(rows)
    realistic = realistic_cells_k(rows)
    if rec_realistic is None and realistic is not None:
        candidate = Path(root) / "analysis" / ("recoverability_k%s.json" % realistic)
        rec_realistic = candidate if candidate.exists() else None
    table = {
        "meta": {
            "metric": metric,
            "windows": {"R0": [0.0, 12.0], "R2": [2.0, 12.0], "F": [12.0, 14.0]},
            "normaliser": "training-window (R0) truth range per component, pooled over tanks",
            "rule": RULE,
            "root": str(root).replace("\\", "/"),
            "n_runs": len(refs),
            "track_b": list(TRACK_B),
            "diverged_runs": [{"run_dir": d, "reason": why} for d, why in diverged],
            "failed_runs": failed,
            "infrastructure_crashes": infrastructure_crashes(root),
            "realistic_k": realistic,
        },
        "rows": rows,
        "comparisons": comparisons(rows),
        "winners": winners(rows),
        "crossovers": crossing,
        "parameter_recovery": parameter_recovery(records),
        "realisation_spread": realisation_spread(records),
        "lab_seed_spread": lab_seed_spread(rows),
        "hypotheses": hypotheses(rows, crossing, realistic, tau_components(rec_realistic),
                                 str(rec_realistic).replace("\\", "/") if rec_realistic else None),
    }
    states = {"meta": {"components": list(TRACK_B), "tanks": [1, 2, 3, 4, 5], "metric": "per_tank_nrmse",
                       "normaliser": table["meta"]["normaliser"], "windows": list(STATE_WINDOWS)},
              "entries": state_entries(records)}
    return {"table": table, "states": states}


def write_outputs(result: dict[str, Any], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / "regime_table.json", out_dir / "regime_states.json",
             out_dir / "regime_table.csv", out_dir / "regime_table.md"]
    paths[0].write_text(json.dumps(_clean(result["table"]), indent=1), encoding="utf-8")
    paths[1].write_text(json.dumps(_clean(result["states"])), encoding="utf-8")
    write_csv(result["table"]["rows"], paths[2])
    paths[3].write_text(markdown(result["table"]), encoding="utf-8")
    return paths


# -- heatmap (Fig. 6, the Supplementary map and the graphical-abstract panel) ------
WINDOW_TEXT = {"R0": "days 0\u201312", "R2": "days 2\u201312", "F": "days 12\u201314"}
#: Column names of the narrow graphical-abstract panel.
GA_TEXT = {"persistence": "Persistence", "ode_openloop_reduced": "Open loop", "cl_pinn": "CL-PINN", "eks": "EKS",
           "eks_aug": "Aug. EKS"}
#: Heatmap slot of the planned WER graphical abstract (plan Task 8.15: 16 x 9 cm canvas, heatmap 7.1 x 6.9 cm);
#: the current paper/graphical_abstract.tex is the superseded C&CE version.
GA_SIZE_IN = (7.1 / 2.54, 6.9 / 2.54)


def skill_text(val: float) -> str:
    """Cell number: two decimals, three when a non-zero skill would print as 0.00; typographic minus."""
    if abs(val) < 0.0005:
        return "0.00"
    return ("%.3f" % val if abs(val) < 0.005 else "%.2f" % val).replace("-", "\u2212")
NOT_RUN_GREY = "0.88"


def cell_label(cell: str) -> str:
    info, _ = parse_cell(cell)
    if info["rand"] is not None:
        return "random truth %d" % info["rand"]
    if info["k"] is None:
        return cell
    k = int(info["k"])
    k_txt = "K0" if k == 0 else ("K1" if k == 100 else "K.%s" % ("%02d" % k).rstrip("0"))
    parts = [k_txt + (" off-steady" if info["offsteady"] else ""), info["influent"].capitalize(),
             info["anchor"].capitalize()]
    if info["extra"]:
        parts.append(info["extra"])
    return " ".join(parts)


def default_cells(table: dict[str, Any], sigma: float, pinn_only: bool = True) -> list[str]:
    rows = [r for r in table["rows"] if abs(r["sigma"] - sigma) < 1e-9 and r["k"] is not None
            and not r["extra"] and r["rand"] is None]
    if pinn_only:
        with_pinn = {r["cell"] for r in rows if r["family"] == "pinn"}
        rows = [r for r in rows if r["cell"] in with_pinn]
    return sorted({r["cell"] for r in rows}, key=lambda c: _cell_sort_key(parse_cell(c)[0]))


def default_estimators(table: dict[str, Any], sigma: float, cells: list[str]) -> list[str]:
    present = {r["estimator"] for r in table["rows"] if abs(r["sigma"] - sigma) < 1e-9 and r["cell"] in cells
               and (r["primary"] or r["estimator"] in MORE_INFORMATION)}
    return sorted(present, key=_estimator_sort_key)


def skill_matrix(table: dict[str, Any], window: str, sigma: float, cells: list[str],
                 estimators: list[str]) -> np.ndarray:
    lookup = {(r["cell"], r["estimator"]): r for r in table["rows"]
              if r["window"] == window and abs(r["sigma"] - sigma) < 1e-9}
    out = np.full((len(cells), len(estimators)), np.nan)
    for i, c in enumerate(cells):
        for j, e in enumerate(estimators):
            r = lookup.get((c, e))
            if r is not None and r.get("skill") is not None:
                out[i, j] = r["skill"]
    return out


def _estimator_text(estimator: str) -> str:
    return PRETTY.get(base_model(estimator), estimator) + estimator[len(base_model(estimator)):]


def plot_heatmap(table: dict[str, Any], png_path: Path, sigma: float = 0.10, window: str = "R0",
                 cells: list[str] | None = None, estimators: list[str] | None = None, style: str = "main",
                 title: str | None = None) -> list[Path]:
    """One window of the regime map, drawn at its printed size.

    ``style``: ``"main"`` (Fig. 6, full page width, numbers in the cells), ``"si"`` (the same for many
    cells) or ``"ga"`` (the graphical-abstract slot, colours only). The best row entry of the window
    is outlined, every tied entry too; cells without a row are grey; a row whose skill is not finite
    reads 'fail' (failed or diverged run) or 'n/a'; a diverged row with a finite value is marked with a
    double dagger (Section 5: it never wins).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Rectangle

    from scripts.figure_layout import DOUBLE_IN, print_style, save_checked, wrap

    cells = cells or default_cells(table, sigma)
    estimators = estimators or default_estimators(table, sigma, cells)
    if not cells or not estimators:
        raise ValueError("regime table has no grid rows at sigma = %.2f" % sigma)
    if style not in ("main", "si", "ga"):
        raise ValueError("unknown heatmap style %r" % style)
    win = {w["cell"]: [w["estimator"]] + list(w.get("tied_with") or []) for w in table["winners"]
           if abs(w["sigma"] - sigma) < 1e-9 and w["window"] == window}
    numbers = style != "ga"
    if style == "ga":
        size = GA_SIZE_IN
    else:
        row_in = 0.26 if style == "main" else 0.175
        size = (DOUBLE_IN, max(2.6, 1.95 + row_in * len(cells)))
    with print_style():
        fig, ax = plt.subplots(figsize=size, layout="constrained")
        grid = skill_matrix(table, window, sigma, cells, estimators)
        rows_here = {(r["cell"], r["estimator"]): r for r in table["rows"]
                     if r["window"] == window and abs(r["sigma"] - sigma) < 1e-9}
        ax.set_facecolor(NOT_RUN_GREY)
        image = ax.imshow(np.clip(grid, -1.0, 1.0), cmap="RdBu", norm=TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0),
                          aspect="auto", interpolation="nearest")
        used = set()
        for i, cell in enumerate(cells):
            if numbers:
                for j, estimator in enumerate(estimators):
                    row = rows_here.get((cell, estimator))
                    if row is None:
                        continue  # grey: not run in this cell
                    val = grid[i, j]
                    bad = bool(row.get("failed") or row.get("diverged"))
                    if not np.isfinite(val):
                        text = "fail" if bad or val == -np.inf else "n/a"
                        used.add(text)
                        colour = "white" if val == -np.inf else "black"
                    else:
                        mark = "\u2020" if seed_label(row) else ("\u2021" if bad else "")
                        used.update({mark} - {""})
                        text = skill_text(val) + mark
                        colour = "white" if abs(val) > 0.6 else "black"
                    ax.text(j, i, text, ha="center", va="center", fontsize=7, color=colour)
            best = [e for e in win.get(cell, []) if e in estimators]
            if len(best) > 1:
                used.add("tie")
            for estimator in best:
                j = estimators.index(estimator)
                for lw, colour in ((2.4, "white"), (1.1, "black")):
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, lw=lw, ec=colour, zorder=3))
        ax.set_xticks(range(len(estimators)))
        ax.set_xticklabels([GA_TEXT.get(e, _estimator_text(e)) if style == "ga" else _estimator_text(e)
                            for e in estimators], rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(len(cells)))
        ax.set_yticklabels([cell_label(c) for c in cells])
        ax.tick_params(length=0)
        ax.set_title(title or "Window %s (%s)" % (window, WINDOW_TEXT[window]), loc="left")
        cbar = fig.colorbar(image, ax=ax, fraction=0.05 if style == "ga" else 0.025, pad=0.02,
                            ticks=[-1, -0.5, 0, 0.5, 1])
        cbar.ax.set_yticklabels(["\u2264\u22121", "\u22120.5", "0", "0.5", "1"])
        cbar.set_label("skill" if style != "ga" else "skill vs persistence")
        if style == "ga":
            note = "Outlined: best in the row. \u03c3 = %.2f." % sigma
        else:
            note = ("Numbers: skill = 1 \u2212 E/E$_{\\mathrm{persistence}}$ against anchor-aware persistence "
                    "(\u03c3 = %.2f, noise realization 0); colors clipped at \u22121. Outlined: best entry of the "
                    "row on window %s%s. * = more information, never a winner." % (
                        sigma, window, " (tied entries all outlined)" if "tie" in used else ""))
            if "\u2020" in used:
                note += " \u2020 = PINN row with one or two seeds (labeled; registration, Section S8)."
            if "\u2021" in used:
                note += " \u2021 = diverged run (registration, Section S8), never a winner."
            if "fail" in used:
                note += " fail = failed or diverged run."
            if "n/a" in used:
                note += " n/a = no score."
            note += " Gray: not run in this cell."
        fig.supxlabel(wrap(note, size[0] * 0.96), fontsize=7)
        png_path = Path(png_path)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            written = save_checked(fig, png_path, save_figure)
        finally:
            plt.close(fig)
    return written


GA_ESTIMATORS = ("persistence", "ode_openloop_reduced", "cl_pinn", "eks", "eks_aug")
#: Supplementary map: all grid cells, split by kinetics so that each page stays legible.
SI_PARTS = (("k000-k025", ("000", "025")), ("k050-k100", ("050", "075", "100")))


def plot(root: Path, sigma: float, paper_dir: Path | None) -> list[Path]:
    """Fig. 6 (one file per window), the Supplementary map (per window and kinetics part) and the GA panel."""
    table = json.loads((Path(root) / "regime_table.json").read_text(encoding="utf-8"))
    fig_dir = Path(root) / "figures"
    cells = default_cells(table, sigma)
    estimators = default_estimators(table, sigma, cells)
    written: list[Path] = []
    for letter, window in (("a", "R0"), ("b", "F")):
        written += plot_heatmap(table, fig_dir / ("fig6%s_regime_map_%s.png" % (letter, window)), sigma=sigma,
                                window=window, cells=cells, estimators=estimators,
                                title="(%s) Window %s (%s)" % (letter, window, WINDOW_TEXT[window]))
    all_cells = default_cells(table, sigma, pinn_only=False)
    all_estimators = default_estimators(table, sigma, all_cells)
    for window in ("R0", "F"):
        for name, ks in SI_PARTS:
            part = [c for c in all_cells if parse_cell(c)[0]["k"] in ks]
            if not part:
                continue
            kin = list(dict.fromkeys(cell_label(c).split(" ")[0] for c in part))
            written += plot_heatmap(table, fig_dir / ("figS_regime_map_%s_%s.png" % (window, name)), sigma=sigma,
                                    window=window, cells=part, estimators=all_estimators, style="si",
                                    title="Window %s (%s), cells %s" % (window, WINDOW_TEXT[window],
                                                                       ", ".join(kin)))
    ga_est = [e for e in GA_ESTIMATORS if e in estimators]
    written += plot_heatmap(table, fig_dir / "graphical_abstract_regime.png", sigma=sigma, window="R0",
                            cells=cells, estimators=ga_est, style="ga",
                            title="Skill against persistence, %s" % WINDOW_TEXT["R0"])
    if paper_dir is not None and Path(paper_dir).exists():
        for path in written:
            shutil.copy2(path, Path(paper_dir) / path.name)
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("score", help="score every cell and write regime_table.*")
    s.add_argument("--root", default="results/v11")
    s.add_argument("--legacy-lstm", default="results/runs", help="v1.0 run root holding lstm_sigma*")
    s.add_argument("--no-legacy", action="store_true")
    s.add_argument("--metric", default="track_b_nrmse_fixed", choices=("track_b_nrmse_fixed", "track_b_nrmse"))
    s.add_argument("--rec-realistic", default=None,
                   help="G4 recoverability file of the realistic kinetics (tau for H4); default "
                        "<root>/analysis/recoverability_k<KKK>.json")
    p = sub.add_parser("plot", help="draw Fig. 6 and the graphical-abstract panel from regime_table.json")
    p.add_argument("--root", default="results/v11")
    p.add_argument("--sigma", type=float, default=0.10)
    p.add_argument("--paper-dir", default="paper/figures")
    args = parser.parse_args(argv)
    if args.command == "plot":
        for path in plot(Path(args.root), args.sigma, Path(args.paper_dir) if args.paper_dir else None):
            print("wrote", path)
        return
    result = score(Path(args.root), None if args.no_legacy else Path(args.legacy_lstm), args.metric,
                   Path(args.rec_realistic) if args.rec_realistic else None)
    for path in write_outputs(result, Path(args.root)):
        print("wrote", path)
    t = result["table"]
    print("%d runs, %d rows, %d comparisons, %d crossovers, %d parameter-recovery groups"
          % (t["meta"]["n_runs"], len(t["rows"]), len(t["comparisons"]), len(t["crossovers"]),
             len(t["parameter_recovery"])))
    print("%d diverged or failed runs" % len(t["meta"]["diverged_runs"]))
    for name in sorted(k for k in t["hypotheses"] if re.fullmatch(r"H\d", k)):
        print("%s: %s -- %s" % (name, t["hypotheses"][name]["status"], t["hypotheses"][name].get("summary", "")))


if __name__ == "__main__":
    main()

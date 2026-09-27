"""Tables T1-T3, the Supplementary tables and the manuscript number registry (v1.1).

    python -m scripts.v11_tables --root results/v11

Reads saved JSON/CSV only. Writes results/v11/tables/<name>.{tex,md},
results/v11/tables/tables_index.json (sources of every table) and
results/v11/numbers.json: one entry per number the manuscript quotes, with
the file and JSON path it came from. The manuscript (plan G8) cites numbers
from numbers.json only.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.regime_map import (
    SENSOR_EXTRA,
    TRACK_B,
    _cell_sort_key,
    base_model,
    cell_label,
    comparison_kind,
    parse_cell,
    seed_label,
)
from scripts.v11_figures import CLASSES, load_recoverability, normalise_validation

V10_ROOTS = ("results/runs", "results/runs_seed1", "results/runs_seed2", "results/runs_ablation",
             "results/runs_flowonly", "results/runs_icmask")
V10_EXPECTED = {"pinn": 36, "lstm": 8, "analytic": 8}
RECOVERABLE = {"sensor-recoverable": "yes", "partly recoverable": "partly",
               "forcing-slaved": "partly (influent-driven)", "anchor-carried": "no"}
SUPERSEDED = "v1.0, partial-derivative residual, superseded"
MIN_GAIN = 0.01
ANCHOR_SIGMA_LOG = 0.6


def latex_escape(text: Any) -> str:
    s = str(text)
    for a, b in (("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("_", "\\_"), ("#", "\\#")):
        s = s.replace(a, b)
    return s


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return "n/a" if not np.isfinite(value) else "%.3g" % value
    return str(value)


def render(tab: dict[str, Any]) -> tuple[str, str]:
    """(LaTeX, Markdown) for a table dict with headers, rows, caption, label, sources."""
    headers, rows = tab["headers"], tab["rows"]
    cols = "l" * len(headers)
    body = ["\\begin{table}[t]", "\\centering",
            "\\caption{%s Source: %s.}" % (latex_escape(tab["caption"]),
                                           latex_escape(", ".join(tab["sources"]))),
            "\\label{%s}" % tab["label"], "\\footnotesize", "\\begin{tabular}{%s}" % cols, "\\toprule",
            " & ".join(latex_escape(h) for h in headers) + " \\\\", "\\midrule"]
    body += [" & ".join(latex_escape(fmt(v)) for v in row) + " \\\\" for row in rows]
    body += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    md = ["**%s** (%s)" % (tab["caption"], ", ".join(tab["sources"])), "",
          "| " + " | ".join(headers) + " |", "|" + " --- |" * len(headers)]
    md += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in rows]
    return "\n".join(body) + "\n", "\n".join(md) + "\n"


def records_from(obj: Any) -> list[dict[str, Any]]:
    """Rows of an arbitrary G1/G2 JSON: a list of dicts, the first list-of-dicts value, or a dict of dicts."""
    if isinstance(obj, list):
        return [r for r in obj if isinstance(r, dict)]
    if isinstance(obj, dict):
        for key in ("rows", "runs", "records", "checkpoints", "states", "cells"):
            if isinstance(obj.get(key), list):
                return records_from(obj[key])
        for value in obj.values():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return value
        if obj and all(isinstance(v, dict) for v in obj.values()):
            return [{"key": k, **v} for k, v in obj.items()]
    raise ValueError("no table-shaped records found")


def flatten(record: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in record.items():
        name = "%s.%s" % (prefix, key) if prefix else str(key)
        if isinstance(value, dict):
            out.update(flatten(value, name))
        elif isinstance(value, list):
            out[name] = ";".join(fmt(v) for v in value)
        else:
            out[name] = value
    return out


def generic_table(path: Path, caption: str, label: str) -> dict[str, Any]:
    flat = [flatten(r) for r in records_from(json.loads(Path(path).read_text(encoding="utf-8")))]
    headers: list[str] = []
    for r in flat:
        headers += [k for k in r if k not in headers]
    return {"headers": headers, "rows": [[r.get(h) for h in headers] for r in flat],
            "caption": caption, "label": label, "sources": [str(path)]}


def kinetic_names(path: Path) -> list[str]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    names = obj if isinstance(obj, list) else (obj.get("names") or obj.get("subset"))
    if not names:
        raise KeyError("%s: no kinetic parameter names" % path)
    return [str(n) for n in names]


def t1_information(table: dict[str, Any], kinetic: list[str], ras_window: int) -> dict[str, Any]:
    subset = ", ".join(kinetic)
    ras = "RAS TSS (trailing mean of %d samples) as input" % ras_window
    rows = [
        ["Persistence", "none", "anchor mean, held constant", "none", "none"],
        ["Open loop, information-matched", ras, "anchor mean", "cell influent view", "reduced reactor train, vault kinetics"],
        ["Open loop, structure-rich*", "none", "anchor mean and settler state", "cell influent view",
         "full plant with settler, vault kinetics"],
        ["EKF, EKS, IEKS", "7 online channels; " + ras, "anchor mean and covariance", "cell influent view",
         "reduced reactor train, vault kinetics"],
        ["Augmented EKS", "as EKS", "as EKS", "as EKS", "as EKS plus multipliers on %s (prior sd ln 2)" % subset],
        ["CL-PINN, single-stage PINN", "7 online channels in the loss; " + ras + " in the residual",
         "anchor mean, weighted by its covariance", "cell influent view (features, residual, balance)",
         "reduced reactor-train residual, vault kinetics"],
        ["PINN-theta", "as CL-PINN", "as CL-PINN", "as CL-PINN",
         "as CL-PINN plus multipliers on %s (prior sd ln 2)" % subset],
        ["LSTM", "7 online channels in the loss", "anchor mean", "cell influent view (features)", "none"],
        ["Online EKF*", "as EKF, also days 12-14", "as EKF", "as EKF", "as EKF"],
    ]
    present = {base_model(r["estimator"]) for r in table["rows"]}
    return {"headers": ["Estimator", "Sensor signals", "Start-state knowledge", "Influent knowledge", "Model"],
            "rows": rows, "label": "tab:information",
            "caption": "Information available to each estimator; * marks rows with more information, which are "
                       "reported but never counted as winners. Estimators present in this run set: %s."
                       % ", ".join(sorted(present)),
            "sources": ["results/v11/analysis/kinetic_subset.json", "results/v11/regime_table.json"]}


def prereg_commit(tag: str = "v1.1.0-prereg") -> str:
    out = subprocess.run(["git", "rev-list", "-n", "1", tag], capture_output=True, text=True, check=True)
    return out.stdout.strip()[:12]


def t2_regime(table: dict[str, Any], commit: str) -> dict[str, Any]:
    grid = [r for r in table["rows"] if r["k"] is not None and not r["extra"] and r["rand"] is None and r["window"] == "R0"]
    cells = sorted({r["cell"] for r in grid}, key=lambda c: _cell_sort_key(parse_cell(c)[0]))
    rows = []
    for cell in cells:
        sub = [r for r in grid if r["cell"] == cell]
        pinn = sorted({"%s x%d (sigma %.2f)" % (r["estimator"], r["n"], r["sigma"]) for r in sub if r["family"] == "pinn"})
        cpu = sorted({r["estimator"] for r in sub if r["family"] in ("observers", "baselines")})
        sigmas = sorted({r["sigma"] for r in sub})
        rows.append([cell_label(cell), ", ".join("%.2f" % s for s in sigmas), "; ".join(pinn) or "none", ", ".join(cpu)])
    return {"headers": ["Cell (kinetics, influent, start state)", "sigma", "PINN runs (model x seeds)", "CPU estimators"],
            "rows": rows, "label": "tab:regimes",
            "caption": "Regime grid as run. %s Pre-registration: tag v1.1.0-prereg, commit %s." % (table["meta"]["rule"], commit),
            "sources": ["results/v11/regime_table.json", "PREREGISTRATION.md"]}


def _majority_class(classes: list[str]) -> str:
    counts = Counter(classes)
    top = max(counts.values())
    return max((c for c in counts if counts[c] == top), key=CLASSES.index)


def t3_practitioner(rec: dict[str, Any]) -> dict[str, Any]:
    subsets = rec["sensor_subsets"]
    full = max(subsets, key=lambda s: len(s["channels"]))
    by_set = {frozenset(s["channels"]): s for s in subsets}
    rows = []
    for comp in TRACK_B:
        states = [s for s in rec["states"] if s["component"] == comp]
        cls = _majority_class([s["class"] for s in states])
        losses = {}
        for ch in full["channels"]:
            drop = by_set.get(frozenset(full["channels"]) - {ch})
            if drop is not None:
                losses[ch] = float(np.mean(np.array(full["per_state_ig"][comp]) - np.array(drop["per_state_ig"][comp])))
        probe = max(losses, key=losses.get) if losses and max(losses.values()) >= MIN_GAIN else "none"
        gains = {a["assay"] + " (" + a["tier"] + ")": float(a["ig_gain"].get(comp, 0.0)) for a in rec["lab_assays"]}
        assay = max(gains, key=gains.get) if gains and max(gains.values()) >= MIN_GAIN else "none"
        tau = float(np.median([s["tau_days"] for s in states]))
        if cls in ("anchor-carried", "partly recoverable"):
            interval = "%d d" % max(1, int(round(tau)))
        else:
            interval = "influent analysis" if cls == "forcing-slaved" else "online"
        rows.append([comp, RECOVERABLE[cls], probe, assay, interval, tau])
    tiers_only = any(a["assay"] == "start-up panel" for a in rec["lab_assays"])
    return {"headers": ["State", "Recoverable", "Probe that carries it", "Assay that helps most",
                        "Re-sampling interval", "Memory time (d)"],
            "rows": rows, "label": "tab:practitioner",
            "caption": "Practitioner guidance per never-measured state (majority class over the five tanks; the "
                       "re-sampling interval is the start-up memory time rounded to whole days; probe loss and assay "
                       "gain are mean information gains over the tanks, shown when at least %.2f).%s"
                       % (MIN_GAIN, " Assay gains are laboratory tiers, not single assays." if tiers_only else ""),
            "sources": ["results/v11/analysis/recoverability_%s.json" % rec["tag"]]}


def si_components(table: dict[str, Any], sigma: float, window: str) -> dict[str, Any]:
    rows = [[cell_label(r["cell"]), r["estimator"], r["n"], seed_label(r) or ""] + [r["per_component_fixed"][c] for c in TRACK_B]
            for r in table["rows"] if abs(r["sigma"] - sigma) < 1e-9 and r["window"] == window and "per_component_fixed" in r]
    return {"headers": ["Cell", "Estimator", "n", "label"] + list(TRACK_B), "rows": rows,
            "label": "tab:si-components-%s" % window,
            "caption": "Per-component NRMSE (fixed R0 range), window %s, sigma %.2f, median over the n seeds "
                       "(two-seed and one-seed PINN rows labeled, registration Section 9)." % (window, sigma),
            "sources": ["results/v11/regime_table.json"]}


def si_observer_diagnostics(table: dict[str, Any]) -> dict[str, Any]:
    rows, keys = [], []
    for r in table["rows"]:
        if r["family"] == "observers" and r["window"] == "R0" and r.get("diagnostics"):
            flat = flatten(r["diagnostics"])
            keys += [k for k in flat if k not in keys]
            rows.append((r, flat))
    return {"headers": ["Cell", "sigma", "Estimator", "Diverged (Section 5)"] + keys,
            "rows": [[cell_label(r["cell"]), r["sigma"], r["estimator"], "; ".join(r.get("diverged") or []) or "no"]
                     + [flat.get(k) for k in keys] for r, flat in rows],
            "label": "tab:si-ekf", "caption": "Observer tuning (selected Q), innovation statistics (NIS) and divergence log.",
            "sources": ["results/v11/observers/*/*/summary.json via results/v11/regime_table.json"]}


def si_realisations(table: dict[str, Any]) -> dict[str, Any]:
    rows = [[cell_label(s["cell"]), s["sigma"], s["estimator"], s["window"], s.get("realisation0"), s["n"], s["median"],
             s["p25"], s["p75"], s["min"], s["max"], s.get("realisation0_outside_range")]
            for s in table["realisation_spread"]]
    return {"headers": ["Cell", "sigma", "Estimator", "Window", "realization 0", "n (1-9)", "median", "p25", "p75",
                        "min", "max", "r0 outside"],
            "rows": rows, "label": "tab:si-realisations",
            "caption": "Noise realization 0 (the registered comparison) next to the spread over realizations 1-9, "
                       "which is reported beside it and never pooled with it (Sections 2 and 9).",
            "sources": ["results/v11/regime_table.json"]}


def si_random_mismatch(table: dict[str, Any]) -> dict[str, Any]:
    rand = [r for r in table["rows"] if r["rand"] is not None and r.get("skill") is not None]
    rows = []
    for est in sorted({r["estimator"] for r in rand}):
        for window in ("R0", "F"):
            skills = np.array([r["skill"] for r in rand if r["estimator"] == est and r["window"] == window])
            if skills.size:
                rows.append([est, window, int(skills.size), float(np.median(skills)), float(np.percentile(skills, 10)),
                             float(np.percentile(skills, 90)), float(np.mean(skills > 0))])
    return {"headers": ["Estimator", "Window", "n truths", "median skill", "p10", "p90", "share skill > 0"],
            "rows": rows, "label": "tab:si-random",
            "caption": "Random-mismatch ensemble (lognormal sigma_log 0.3 on all 15 kinetic parameters).",
            "sources": ["results/v11/regime_table.json"]}


def si_lab_seeds(table: dict[str, Any]) -> dict[str, Any]:
    rows = [[cell_label(s["cell"]), s["sigma"], s["estimator"], s["window"], s["registered_seed_value"], s["n"],
             s["median"], s["p25"], s["p75"], s["min"], s["max"], s["registered_seed_outside_range"]]
            for s in table.get("lab_seed_spread", [])]
    return {"headers": ["Cell", "sigma", "Estimator", "Window", "seed 20260923", "n (further seeds)", "median", "p25",
                        "p75", "min", "max", "registered seed outside"],
            "rows": rows, "label": "tab:si-lab-seeds",
            "caption": "Laboratory panel: the registered seed next to the spread over the nine further laboratory "
                       "seeds, never pooled with it (Section 2).",
            "sources": ["results/v11/regime_table.json"]}


def si_sigma_log(table: dict[str, Any], sigma: float = 0.10) -> dict[str, Any]:
    sl_rows = [(r, int(m.group(1)) / 10.0) for r in table["rows"]
               if (m := re.fullmatch(r"sl(\d+)", r["extra"] or "")) and r["window"] == "R0"
               and abs(r["sigma"] - sigma) < 1e-9]
    bases = {r["cell"][: -len(r["extra"]) - 1] for r, _ in sl_rows}
    estimators = {r["estimator"] for r, _ in sl_rows}
    base_rows = [(r, ANCHOR_SIGMA_LOG) for r in table["rows"]
                 if r["cell"] in bases and r["window"] == "R0" and r["estimator"] in estimators
                 and abs(r["sigma"] - sigma) < 1e-9]
    rows = sorted(([cell_label(r["cell"].split("_sl")[0]), s, r["estimator"], r["median"], r["skill"]]
                   for r, s in sl_rows + base_rows), key=lambda x: (x[0], x[2], x[1]))
    return {"headers": ["Cell", "sigma_log", "Estimator", "Track B NRMSE (R0)", "skill"], "rows": rows,
            "label": "tab:si-sigmalog", "caption": "Sensitivity to the spread of the start-state ensemble "
                                                   "(sigma_log 0.4 and 0.8 against the pre-registered 0.6), "
                                                   "sigma %.2f." % sigma,
            "sources": ["results/v11/regime_table.json"]}


def si_sensor_subsets(rec: dict[str, Any], table: dict[str, Any]) -> dict[str, Any]:
    ranked = sorted(rec["sensor_subsets"], key=lambda s: -s["ig_mean"])
    rows = [["Fisher ranking", ", ".join(s["channels"]), s["ig_mean"], None] for s in ranked[:10] + ranked[-5:]]
    rows += [["achieved (%s)" % cell_label(r["cell"]),
              r["estimator"] + (" (%s)" % seed_label(r) if seed_label(r) else ""), None, r["median"]]
             for r in table["rows"] if r["window"] == "R0" and r["primary"] and abs(r["sigma"] - 0.1) < 1e-9
             and SENSOR_EXTRA.match(r["extra"] or "")]
    return {"headers": ["Kind", "Channels or estimator", "mean information gain", "Track B NRMSE (R0)"], "rows": rows,
            "label": "tab:si-sensors", "caption": "Sensor subsets: top ten and bottom five by Fisher information, and the "
                                                  "confirmation runs.",
            "sources": ["results/v11/analysis/recoverability_%s.json" % rec["tag"], "results/v11/regime_table.json"]}


def si_lab_assays(rec: dict[str, Any]) -> dict[str, Any]:
    rows = [[a["tier"], a["assay"]] + [a["ig_gain"].get(c) for c in TRACK_B] for a in rec["lab_assays"]]
    return {"headers": ["Tier", "Assay"] + list(TRACK_B), "rows": rows, "label": "tab:si-lab",
            "caption": "Information gain of each start-up laboratory assay per never-measured state.",
            "sources": ["results/v11/analysis/recoverability_%s.json" % rec["tag"]]}


def si_v10_superseded(bands_path: Path, detail_path: Path) -> dict[str, Any]:
    bands = json.loads(Path(bands_path).read_text(encoding="utf-8"))
    detail = json.loads(Path(detail_path).read_text(encoding="utf-8"))["rows"]
    sigmas = (0.0, 0.05, 0.10, 0.15)
    rows = []
    for model in ("cl_pinn", "pinn"):
        cells = []
        for s in sigmas:
            b = bands["%s|%.2f|holdout" % (model, s)]["track_b_nrmse"]
            cells.append("%.3f [%.3f, %.3f]" % (b["median"], b["min"], b["max"]))
        rows.append([model] + cells)
    for model in ("cl_lstm", "lstm", "persistence", "ode_openloop"):
        vals = {r["noise"]: r["track_b_nrmse"] for r in detail if r["model"] == model and r["eval_set"] == "holdout"}
        rows.append([model] + ["%.3f" % vals[s] for s in sigmas])
    return {"headers": ["Model"] + ["sigma %.2f" % s for s in sigmas], "rows": rows, "label": "tab:si-v10",
            "caption": "%s: holdout Track B NRMSE of the v1.0 benchmark (per-window range)." % SUPERSEDED,
            "sources": [str(bands_path), str(detail_path)]}


def si_hypotheses(table: dict[str, Any], validation: dict[str, Any] | None) -> dict[str, Any]:
    """H1-H7 as registered: status and the numbers behind it (H6 from the G4 validation)."""
    hyp = table.get("hypotheses") or {}
    rows = [[name, hyp[name]["status"], hyp[name].get("summary", "")]
            for name in sorted(k for k in hyp if re.fullmatch(r"H\d", k))]
    h6 = (validation or {}).get("h6")
    if h6:
        status = "supported" if h6.get("pass") else ("not supported" if h6.get("pass") is False else "not decided")
        ci = h6.get("ci95") or [None, None]
        rows.append(["H6", status, "EKS at %s: rho %s [%s, %s]; pass if rho >= 0.5 and the lower end > 0"
                     % (h6.get("cell"), fmt(h6.get("rho")), fmt(ci[0]), fmt(ci[1]))])
    rows.sort(key=lambda r: r[0])
    return {"headers": ["Hypothesis", "Outcome", "Evidence (window R0, sigma 0.10, realization 0)"], "rows": rows,
            "label": "tab:si-hypotheses",
            "caption": "Pre-registered hypotheses H1-H7 (PREREGISTRATION.md, Section 3) and their outcomes under the "
                       "rules of Section 9.",
            "sources": ["results/v11/regime_table.json", "results/v11/analysis/validation/recoverability_validation.json"]}


def run_counts(v10_roots: tuple[str, ...], v11_root: Path, ledger: Path) -> dict[str, Any]:
    v10: Counter = Counter()
    for root in v10_roots:
        for d in sorted(Path(root).iterdir()):
            if not d.is_dir():
                continue
            if d.name.startswith("_"):
                v10["verification probe"] += 1
                continue
            v10[json.loads((d / "summary.json").read_text(encoding="utf-8"))["arch"]] += 1
    for arch, expected in V10_EXPECTED.items():
        if v10[arch] != expected:
            raise ValueError("v1.0 %s runs: found %d, the plan states %d" % (arch, v10[arch], expected))
    v11: Counter = Counter()
    for family in ("pinn", "observers", "baselines"):
        for summary in sorted((Path(v11_root) / family).glob("*/*/summary.json")):
            v11["%s: %s" % (family, json.loads(summary.read_text(encoding="utf-8"))["model"])] += 1
    with open(ledger, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        eq_key = next(k for k in reader.fieldnames if k.strip().lower() in ("eq", "equivalents", "run_equivalents"))
        ledger_rows = list(reader)
    total_eq = float(sum(float(r[eq_key]) for r in ledger_rows))
    rows = [["v1.0", k, n] for k, n in sorted(v10.items())] + [["v1.1", k, n] for k, n in sorted(v11.items())]
    rows.append(["v1.1", "GPU ledger: runs / PINN-run equivalents", "%d / %.2f" % (len(ledger_rows), total_eq)])
    return {"headers": ["Release", "Run kind", "Count"], "rows": rows, "label": "tab:si-counts",
            "caption": "Run counts: v1.0 archive (36 PINN checkpoints, 8 LSTM runs, 8 analytic baseline directories) "
                       "and v1.1.", "sources": list(v10_roots) + [str(v11_root), str(ledger)], "total_eq": total_eq}


def numbers_registry(table: dict[str, Any], residual: dict[str, Any] | None, fig_sources: dict[str, Any] | None,
                     validation: dict[str, Any] | None, counts: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}

    def put(key: str, value: Any, source: str, path: str, **extra: Any) -> None:
        if key in out:
            raise ValueError("number registry key %r would be written twice" % key)
        out[key] = {"value": value, "source": source, "path": path, **extra}

    for i, r in enumerate(table["rows"]):
        if r["k"] is not None and not r["extra"] and r["rand"] is None and r["primary"]:
            stem = "trackB.%s.%s.sigma%.2f.%s" % (r["window"], r["cell"], r["sigma"], r["estimator"])
            label = {"n": r["n"], "label": seed_label(r)}
            put(stem + ".median", r["median"], "results/v11/regime_table.json", "rows[%d].median" % i, **label)
            put(stem + ".skill", r["skill"], "results/v11/regime_table.json", "rows[%d].skill" % i, **label)
            put(stem + ".gap_closed", r["gap_closed"], "results/v11/regime_table.json", "rows[%d].gap_closed" % i,
                **label)
    for i, c in enumerate(table["crossovers"]):
        put("alpha_star.%s.%s_%s.sigma%.2f.%s" % (c["window"], c["influent"], c["anchor"], c["sigma"], c["estimator"]),
            c["alpha_star"], "results/v11/regime_table.json", "crossovers[%d].alpha_star" % i)
    wtl: dict[str, Counter] = {}
    for c in table["comparisons"]:
        kind = comparison_kind(c)
        stem = "wtl.%s.%s.vs.%s" % (c["window"], c["pinn"], c["comparator"])
        key = {"deciding": stem, "labelled": stem + ".labelled", "reference": "wtl_reference" + stem[3:]}[kind]
        wtl.setdefault(key, Counter())[c["reference_outcome"] if kind == "reference" else c["outcome"]] += 1
    for key, n in sorted(wtl.items()):
        put(key, {"win": n["win"], "tie": n["tie"], "loss": n["loss"], "not_decided": n["not_decided"]},
            "results/v11/regime_table.json", "comparisons")
    hyp = table.get("hypotheses") or {}
    for name in sorted(k for k in hyp if re.fullmatch(r"H\d", k)):
        put("hyp.%s.status" % name, hyp[name]["status"], "results/v11/regime_table.json", "hypotheses.%s.status" % name)
    if "expected_tie_outcome" in hyp.get("H3", {}):
        put("hyp.H3.expected_tie_outcome", hyp["H3"]["expected_tie_outcome"], "results/v11/regime_table.json",
            "hypotheses.H3.expected_tie_outcome")
    for i, p in enumerate(table.get("parameter_recovery", [])):
        if abs(p["sigma"] - 0.10) > 1e-9 or p["estimator"] not in ("cl_pinn_theta", "eks_aug"):
            continue
        for name, err in (p.get("median_abs_log_error") or {}).items():
            put("param_recovery.%s.%s.%s.median_abs_log_error" % (p["cell"], p["estimator"], name), err,
                "results/v11/regime_table.json", "parameter_recovery[%d].median_abs_log_error.%s" % (i, name),
                n=p.get("n"))
    for i, sp in enumerate(table.get("lab_seed_spread", [])):
        if sp["window"] != "R0" or abs(sp["sigma"] - 0.10) > 1e-9:
            continue
        stem = "lab_seeds.%s.%s" % (sp["cell"], sp["estimator"])
        for k in ("registered_seed_value", "median", "min", "max"):
            put("%s.%s" % (stem, k), sp[k], "results/v11/regime_table.json", "lab_seed_spread[%d].%s" % (i, k))
    if "alpha_star" in hyp.get("H2", {}):
        put("hyp.H2.alpha_star", hyp["H2"]["alpha_star"], "results/v11/regime_table.json", "hypotheses.H2.alpha_star")
    for est, fam in (hyp.get("H5", {}).get("families") or {}).items():
        for k in ("ratio_forcing", "ratio_biomass"):
            put("hyp.H5.%s.%s" % (est, k), fam[k], "results/v11/regime_table.json",
                "hypotheses.H5.families.%s.%s" % (est, k))
    for est, fam in (hyp.get("H4", {}).get("families") or {}).items():
        if "spearman_tau_vs_reduction" in fam:
            put("hyp.H4.%s.spearman_tau_vs_reduction" % est, fam["spearman_tau_vs_reduction"],
                "results/v11/regime_table.json", "hypotheses.H4.families.%s.spearman_tau_vs_reduction" % est)
    put("diverged_runs.count", len(table["meta"].get("diverged_runs") or []), "results/v11/regime_table.json",
        "meta.diverged_runs")
    if residual:
        for i, s in enumerate(residual["summary"]):
            stem = "residual.%s.%s%s.sigma%.2f" % (s["cell"], s["model"],
                                                   "[%s]" % s["variant"] if s.get("variant") else "", s["noise"])
            for k in ("median_R0", "median_F", "median_ratio"):
                put("%s.%s" % (stem, k), s[k], "results/v11/residual_diagnostic.json", "summary[%d].%s" % (i, k))
    if fig_sources:
        for cls, n in fig_sources["fig2"]["class_counts"].items():
            put("recoverability.count.%s" % cls, n, "results/v11/figures/figure_sources.json", "fig2.class_counts")
    if validation:
        src = validation.get("source", "results/v11/analysis/validation/recoverability_validation.json")
        for r in validation["results"]:
            put("h6.%s.%s.rho" % (r["cell"], r["estimator"]), r["rho"], src,
                "cells.%s.estimators.%s.primary_index.rho" % (r["cell"], r["estimator"]))
            if r.get("ci") is not None:
                put("h6.%s.%s.ci95" % (r["cell"], r["estimator"]), r["ci"], src,
                    "cells.%s.estimators.%s.primary_index.ci95" % (r["cell"], r["estimator"]))
        if validation.get("h6"):
            put("hyp.H6.pass", validation["h6"].get("pass"), src, "h6.pass")
            put("hyp.H6.rho", validation["h6"].get("rho"), src, "h6.rho")
    put("gpu.total_equivalents", counts["total_eq"], "results/v11/gpu_ledger.csv", "sum(eq)")
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="results/v11")
    parser.add_argument("--rec-realistic", default=None,
                        help="recoverability file of the realistic kinetics (T3, S_lab_assays); default: "
                             "<root>/analysis/recoverability_k<meta.realistic_k>.json")
    parser.add_argument("--rec-sensors", default=None,
                        help="recoverability file of the sensor-confirmation cells (S_sensor_subsets); default: "
                             "<root>/analysis/recoverability_k100.json (K1-Ie-A0, Section 7)")
    parser.add_argument("--kinetic-subset", default="results/v11/analysis/kinetic_subset.json")
    parser.add_argument("--validation", default="results/v11/analysis/validation/recoverability_validation.json")
    parser.add_argument("--prereg-commit", default=None, help="skip the git lookup (tests)")
    parser.add_argument("--sigma", type=float, default=0.10)
    args = parser.parse_args(argv)
    root = Path(args.root)
    out = root / "tables"
    out.mkdir(parents=True, exist_ok=True)
    table = json.loads((root / "regime_table.json").read_text(encoding="utf-8"))
    realistic = table.get("meta", {}).get("realistic_k")
    rec_path = Path(args.rec_realistic) if args.rec_realistic else root / "analysis" / (
        "recoverability_k%s.json" % (realistic or "100"))
    rec = load_recoverability(rec_path)
    if realistic is not None and rec.get("tag") not in (None, "k%s" % realistic):
        raise ValueError("%s is tagged %r but the regime table's realistic cells are k%s (gate D3)"
                         % (rec_path, rec.get("tag"), realistic))
    rec_sensors = load_recoverability(Path(args.rec_sensors) if args.rec_sensors
                                      else root / "analysis" / "recoverability_k100.json")
    validation_path = Path(args.validation)
    if not validation_path.exists():
        raise FileNotFoundError("validation file %s not found; run scripts.recoverability_validation" % validation_path)
    validation = normalise_validation(json.loads(validation_path.read_text(encoding="utf-8")))
    validation["source"] = str(validation_path).replace("\\", "/")
    ras = sorted({int(json.loads(p.read_text(encoding="utf-8")).get("ras_filter_window", 1))
                  for p in (root / "pinn").glob("*/*/summary.json")}) or [1]
    if len(ras) != 1:
        raise ValueError("PINN runs disagree on ras_filter_window: %s" % ras)
    counts = run_counts(V10_ROOTS, root, root / "gpu_ledger.csv")
    tables = {
        "T1_information": t1_information(table, kinetic_names(Path(args.kinetic_subset)), ras[0]),
        "T2_regimes": t2_regime(table, args.prereg_commit or prereg_commit()),
        "T3_practitioner": t3_practitioner(rec),
        "S_components_R0": si_components(table, args.sigma, "R0"),
        "S_components_F": si_components(table, args.sigma, "F"),
        "S_ekf": si_observer_diagnostics(table),
        "S_realisations": si_realisations(table),
        "S_random_mismatch": si_random_mismatch(table),
        "S_sigma_log": si_sigma_log(table, args.sigma),
        "S_lab_seeds": si_lab_seeds(table),
        "S_sensor_subsets": si_sensor_subsets(rec_sensors, table),
        "S_lab_assays": si_lab_assays(rec),
        "S_v10_superseded": si_v10_superseded(Path("results/seed_bands.json"), Path("results/benchmark_detail.json")),
        "S_run_counts": counts,
        "S_hypotheses": si_hypotheses(table, validation),
    }
    for path, name, caption in ((root / "derivative_audit.json", "S_derivative_audit",
                                 "Derivative audit of the v1.0 checkpoints: partial against total residual."),
                                (root / "bsm1_gate.json", "S_bsm1_gate",
                                 "BSM1 open-loop steady-state gate for the 15 C truth plant.")):
        tables[name] = generic_table(path, caption, "tab:si-" + name.split("_", 1)[1].replace("_", "-"))
    index = {}
    for name, tab in tables.items():
        tex, md = render(tab)
        (out / (name + ".tex")).write_text(tex, encoding="utf-8")
        (out / (name + ".md")).write_text(md, encoding="utf-8")
        index[name] = {"label": tab["label"], "sources": tab["sources"], "n_rows": len(tab["rows"])}
    (out / "tables_index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    residual = root / "residual_diagnostic.json"
    fig_sources = root / "figures" / "figure_sources.json"
    registry = numbers_registry(
        table,
        json.loads(residual.read_text(encoding="utf-8")) if residual.exists() else None,
        json.loads(fig_sources.read_text(encoding="utf-8")) if fig_sources.exists() else None,
        validation,
        counts,
    )
    (root / "numbers.json").write_text(json.dumps(registry, indent=1, default=float), encoding="utf-8")
    print("wrote %d tables under %s and %d numbers to %s" % (len(tables), out, len(registry), root / "numbers.json"))


if __name__ == "__main__":
    main()

"""Render PREREGISTRATION.md from scripts/prereg_template.md (group G6).

Every value that comes from an earlier computation is read from its file here,
and the file's SHA-256 goes into Section 12, so the registration cannot quote a
number that is not on disk. Refuses to write if a required input is missing or
the text breaks the manuscript style rules (no em or en dashes, none of the
banned words).

    python -m scripts.write_prereg
"""

from __future__ import annotations

import hashlib
import json
import re
import string
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import gpu_ledger  # noqa: E402
from scripts.v11_plan import (  # noqa: E402
    ANCHOR_ROOT, BSM1_GATE, CONCURRENCY, GATE_D3, NUMERICS_JSON, REPO, core_queue, kinetic_names,
    realistic_k, sensor_confirmation,
)

TEMPLATE = REPO / "scripts" / "prereg_template.md"
OUTPUT = REPO / "PREREGISTRATION.md"
SMOKE_REPORT = Path("results/v11/_smoke/smoke_report.json")
BANNED = ("robust", "leverage", "delve", "crucial", "pivotal", "holistic", "comprehensive",
          "underscore", "enhance", "foster")
REQUIRED_HEADINGS = (
    "## 1. Question", "## 2. Plant, data and cell notation", "## 3. Hypotheses",
    "## 4. PINN and LSTM protocol (fixed)", "## 5. Model-based estimators (fixed)", "## 6. Grid",
    "## 7. Recoverability analysis", "## 8. Metrics and windows", "## 9. Decision rules",
    "## 10. Interpretation fixed in advance", "## 11. Commitments",
    "## 12. Results and checks that pre-date this registration", "## 13. Budget",
)
CONTEXT_KEYS = (
    "registered_utc", "base_commit", "vault_sha", "bsm1_gate_line", "ensemble_rejection", "gate_d3_line",
    "realistic_label", "realistic_tag", "numerics_line", "primary_smoother", "kinetic_subset", "sensor_drop",
    "sensor_add", "gpu_table", "gpu_total", "observer_table", "observer_total", "claim_cells",
    "predating_table", "smoke_line", "concurrency_line", "ledger_line",
)
#: (path, role, required). Directories are digested over all their files.
PREDATING = (
    ("results/v11/bsm1_gate.json", "BSM1 open-loop steady-state gate (G2)", False),
    ("results/v11/anchors", "anchor files and ensembles (G3)", True),
    ("results/v11/analysis/kinetic_subset.json", "kinetic subset (G4)", True),
    ("results/v11/analysis/sensor_confirmation.json", "E3 sensor choice (G4)", True),
    ("results/v11/analysis/recoverability_k000.json", "recoverability indices at K0 (G4)", True),
    ("results/v11/analysis/recoverability_k100.json", "recoverability indices at K1 (G4)", True),
    ("results/v11/derivative_audit.json", "v1.0 derivative audit (G1)", True),
    ("results/v11/reanalysis_v1.json", "v1.0 re-scoring, exploratory (G1)", True),
    ("results/v11/observer_numerics.json", "numerics pass and primary smoother", True),
    ("results/v11/gate_d3.json", "gate D3", True),
    ("results/v11/_smoke/smoke_report.json", "smoke runs", True),
    ("results/v11/concurrency_check.json", "concurrency and determinism check", True),
    ("results/v11/gpu_ledger.csv", "GPU ledger at registration", True),
    ("results/v11/baselines", "G3 reference rows at K0-Ie-A0 and K1-Ic-As", True),
    ("results/v11/observers", "G3 reference cell K0-Ie-A0 and its frozen q", True),
    ("results/v11/analysis/validation", "v1.0 pilot of the H6 validation, exploratory (G4)", True),
    ("results/v11/predating", "O1 check, gate D4 q scan, laboratory-seed and drift probes", True),
    ("configs/regime", "PINN and LSTM configs", True),
    ("configs/observers", "estimator grid and numerics configs", True),
)


def digest(path: Path) -> str:
    full = REPO / path
    h = hashlib.sha256()
    if full.is_dir():
        for f in sorted(p for p in full.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
            h.update(f.relative_to(full).as_posix().encode("utf-8"))
            h.update(hashlib.sha256(f.read_bytes()).digest())
    else:
        h.update(full.read_bytes())
    return h.hexdigest()


def style_problems(text: str) -> list[str]:
    lower = text.lower()
    out = ["banned word %r" % w for w in BANNED if re.search(r"\b%s" % w, lower)]
    if "\u2014" in text:
        out.append("em dash")
    if "\u2013" in text:
        out.append("en dash")
    return out


def render(context: dict[str, str]) -> str:
    return string.Template(TEMPLATE.read_text(encoding="utf-8")).substitute(context)


def gpu_table(realistic: str) -> tuple[str, float]:
    groups: dict[tuple, dict] = {}
    for job in core_queue(realistic):
        key = (job.phase, job.stem, ", ".join(job.models), ", ".join("%.2f" % s for s in job.noise))
        g = groups.setdefault(key, {"seeds": [], "runs": 0, "eq": 0.0})
        g["seeds"].append(job.seed)
        g["runs"] += len(job.expected_run_ids())
        g["eq"] += job.eq()
    lines = ["| Phase | Config | Models | Sigma | Seeds | Runs | Run-eq |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for (phase, stem, models, noise), g in groups.items():
        lines.append("| %s | `%s` | %s | %s | %s | %d | %.2f |"
                     % (phase, stem, models, noise, ", ".join(str(s) for s in g["seeds"]), g["runs"], g["eq"]))
    return "\n".join(lines), round(sum(g["eq"] for g in groups.values()), 2)


def observer_table(files: dict[str, dict]) -> tuple[str, int]:
    from src.observers.cell_inputs import expected_run_dirs

    fams: dict[str, dict] = {}
    for key, cfg in files.items():
        f = fams.setdefault(key.split("/")[1], {"files": 0, "cells": set(), "runs": 0})
        f["files"] += 1
        f["cells"].add(cfg["cell"])
        f["runs"] += len(expected_run_dirs(cfg))
    lines = ["| Family | YAML files | Cells | Estimator runs |", "| --- | --- | --- | --- |"]
    for fam in ("main", "realisations", "m0prime", "random", "sigma_log", "settler", "sensors"):
        if fam in fams:
            f = fams[fam]
            lines.append("| %s | %d | %d | %d |" % (fam, f["files"], len(f["cells"]), f["runs"]))
    return "\n".join(lines), sum(f["runs"] for f in fams.values())


def _json(path: Path) -> dict:
    return json.loads((REPO / path).read_text(encoding="utf-8"))


def collect_context() -> dict[str, str]:
    from scripts import observer_grid
    from src.asm1.vault_loader import vault

    missing = [p for p, _, required in PREDATING if required and not (REPO / p).exists()]
    if missing:
        raise FileNotFoundError("pre-registration inputs missing: %s" % ", ".join(missing))
    realistic = realistic_k()
    gate, numerics = _json(GATE_D3), _json(NUMERICS_JSON)
    smoke, conc = _json(SMOKE_REPORT), _json(CONCURRENCY)
    with np.load(REPO / ANCHOR_ROOT / "nominal_ensemble.npz", allow_pickle=False) as ens:
        ens_meta = json.loads(str(ens["meta"]))
    drop, add = sensor_confirmation()
    rows = gpu_ledger.read()
    gpu, gpu_total = gpu_table(realistic)
    obs, obs_total = observer_table(observer_grid.load_stage("grid"))
    ekf = numerics["ekf"]
    if (REPO / BSM1_GATE).exists():
        bsm1_line = ("passed; K1 is described as the BSM1 15 C kinetic set." if _json(BSM1_GATE).get("passed")
                     else "failed; the per-state deviations go to the Supplementary Information and K1 is described "
                          "as 'BSM1-layout plant with BSM1 kinetic values'.") + " See `results/v11/bsm1_gate.json`."
    else:
        bsm1_line = ("not run at registration, because the transcribed BSM1 open-loop steady state "
                     "(`tests/data/bsm1_openloop_steady_state.json`) was not available; K1 is described as "
                     "'BSM1-layout plant with BSM1 kinetic values'.")
    gate_line = ("The %s with the As anchor at K1-Ic, sigma 0.10, had R0 Track B NRMSE %.4f against %.4f for "
                 "anchor-aware persistence; the rule moves the realistic cells to K.5 only if the smoother is worse, "
                 "so they %s." % (gate["smoother"].upper(), gate["smoother_r0_track_b_nrmse"],
                                  gate["persistence_r0_track_b_nrmse"],
                                  "stay at K1" if realistic == "k100" else "move to K.5"))
    predating = ["| File | SHA-256 (first 16) | Role |", "| --- | --- | --- |"]
    for path, role, _ in PREDATING:
        d = digest(Path(path))[:16] if (REPO / path).exists() else "not present"
        predating.append("| `%s` | `%s` | %s |" % (path, d, role))
    base_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True,
                                 check=True).stdout.strip()
    context = {
        "registered_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "base_commit": base_commit,
        "vault_sha": vault().json_sha256,
        "bsm1_gate_line": bsm1_line,
        "ensemble_rejection": "%.1f %%" % (100.0 * float(ens_meta["rejection_rate"])),
        "gate_d3_line": gate_line,
        "realistic_label": {"k100": "K1", "k050": "K.5"}[realistic],
        "realistic_tag": realistic,
        "numerics_line": "Numerics pass: the EKF diverged in %d of %d main-grid cells (%.1f %%)."
                         % (ekf["n_diverged"], ekf["n_cells"], 100.0 * ekf["rate"]),
        "primary_smoother": {"eks": "EKS", "ieks": "IEKS"}[numerics["primary_smoother"]],
        "kinetic_subset": ", ".join(kinetic_names()),
        "sensor_drop": drop,
        "sensor_add": add,
        "gpu_table": gpu,
        "gpu_total": "%.2f" % gpu_total,
        "observer_table": obs,
        "observer_total": str(obs_total),
        "claim_cells": ", ".join(observer_grid.claim_cells(realistic)),
        "predating_table": "\n".join(predating),
        "smoke_line": "%s, %d runs, strict poisoning tests %s" % (
            "passed" if smoke["passed"] else "FAILED", smoke["n_runs"], "green" if smoke["leakage_exit"] == 0 else "red"),
        "concurrency_line": "%d GPU process(es); largest relative prediction difference %.1e, throughput gain %.2f"
                            % (conc["workers"], conc["max_rel_diff"], conc["throughput_gain"]),
        "ledger_line": "%.2f run-equivalents in %d attempts" % (gpu_ledger.total_eq(rows), len(rows)),
    }
    assert tuple(context) == CONTEXT_KEYS
    return context


def main(argv: list[str] | None = None) -> int:
    text = render(collect_context())
    problems = style_problems(text)
    missing = [h for h in REQUIRED_HEADINGS if h not in text]
    if problems or missing:
        print("REFUSED: %s" % "; ".join(problems + ["missing heading %r" % h for h in missing]))
        return 1
    OUTPUT.write_text(text, encoding="utf-8")
    print("wrote %s (%d lines)" % (OUTPUT.name, text.count("\n") + 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

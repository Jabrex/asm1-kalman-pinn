"""Shared run plan for the v1.1 execution phase (strategic plan section 4; group G6).

scripts/run_core.py (GPU queue), scripts/gpu_ledger.py (budget ledger),
scripts/observer_grid.py (CPU grid), scripts/gate_d3.py, scripts/smoke_check.py,
scripts/concurrency_check.py, scripts/write_prereg.py and scripts/check_core.py
take their queues, paths and charging rules from here, so they cannot disagree
about what was planned or what it costs. Importing this module does not import
torch. Paths are relative to the repository root, like every v1.0 script.
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

V11 = Path("results/v11")
LEDGER = V11 / "gpu_ledger.csv"
INFLIGHT = V11 / "run_core_inflight.json"
GATE_D3 = V11 / "gate_d3.json"
NUMERICS_JSON = V11 / "observer_numerics.json"
CONCURRENCY = V11 / "concurrency_check.json"
DEVIATIONS = V11 / "deviations.log"
BSM1_GATE = V11 / "bsm1_gate.json"
KINETIC_SUBSET = V11 / "analysis" / "kinetic_subset.json"
SENSOR_CONFIRMATION = V11 / "analysis" / "sensor_confirmation.json"
SMOKE_ROOT = "results/v11/_smoke"
PINN_ROOT = "results/v11/pinn"
OBSERVER_ROOT = "results/v11/observers"
BASELINE_ROOT = "results/v11/baselines"
NUMERICS_OBSERVER_ROOT = "results/v11/_numerics/observers"
NUMERICS_BASELINE_ROOT = "results/v11/_numerics/baselines"
ANCHOR_ROOT = "results/v11/anchors"
REGIME_DIR = Path("configs/regime")
OBSERVER_CONFIG_DIR = Path("configs/observers")
PREREG_TAG = "v1.1.0-prereg"
#: Paths that must equal their state at the pre-registration tag before any
#: full-profile PINN run or final-grid observer run starts.
FROZEN_PATHS = (
    "PREREGISTRATION.md", "configs/base.yaml", "configs/regime", "configs/observers",
    "src", "scripts/run_all.py", "scripts/run_observers.py", "scripts/make_baselines.py",
)

SIGMA = 0.10
REALISTIC_CHOICES = ("k100", "k050")
CORE_PHASES = ("E4", "E5", "E6", "E7", "E3")

#: Kinetics tag -> data directory (group G2). results/raw is the v1.0 data, untouched.
DATA_DIRS = {
    "k000": "results/raw",
    "k025": "results/raw_k025",
    "k050": "results/raw_k050",
    "k075": "results/raw_k075",
    "k100": "results/raw_k100",
    "k000_off": "results/raw_k000_off",
}

#: Architecture of every model the v1.1 queues train (tests compare it with MODEL_SPECS).
ARCH = {"cl_pinn": "pinn", "pinn": "pinn", "cl_pinn_theta": "pinn", "lstm": "lstm"}
#: Charged run-equivalents. A 4000-step quick run costs 0.2 whatever its
#: architecture; a full LSTM run 0.35 (measured 2.3-3.1 min against a PINN
#: median of about 9.0 min, rounded up).
EQ_QUICK = 0.2
EQ_FULL = {"pinn": 1.0, "lstm": 0.35}
G1_SMOKE_EQ = 0.2
G6_EQ = 38.85            # smoke 1.8 + E4 9 + E5 6 + E6 14.05 + E7 4 + E3 4
CONTINGENCY_EQ = 6.0
CORE_CAP = round(G1_SMOKE_EQ + G6_EQ + CONTINGENCY_EQ, 2)   # 45.05
HARD_CAP = 50.0

LIST_LINE = re.compile(r"^\s{2}(?P<run_id>\S+)\s+model=(?P<model>\S+)")
SIGMA_IN_ID = re.compile(r"_sigma(?P<whole>\d+)p(?P<frac>\d{2})")


def sigma_tag(sigma: float) -> str:
    return ("%.2f" % sigma).replace(".", "p")


def sigma_of(run_id: str) -> float:
    match = SIGMA_IN_ID.search(run_id)
    if match is None:
        raise ValueError("run id %r carries no _sigma<tag>" % run_id)
    return float("%s.%s" % (match["whole"], match["frac"]))


def nominal_eq(model: str, profile: str) -> float:
    if profile == "quick":
        return EQ_QUICK
    return EQ_FULL[ARCH[model]]


def config_variants(config: str | Path) -> tuple[str, ...]:
    """Run-id suffixes of the YAML ``variants`` list (group G1); empty without it."""
    raw = yaml.safe_load((REPO / config).read_text(encoding="utf-8"))
    return tuple(str(v["suffix"]) for v in (raw.get("variants") or ()))


@dataclass(frozen=True)
class Job:
    """One scripts.run_all invocation: one config, one seed, a model and noise subset."""

    phase: str
    config: str
    seed: int
    profile: str
    out_dir: str
    models: tuple[str, ...]
    noise: tuple[float, ...]

    @property
    def stem(self) -> str:
        return Path(self.config).stem

    def run_all_argv(self) -> list[str]:
        return [
            "--config", self.config,
            "--profile", self.profile,
            "--seed", str(self.seed),
            "--out-dir", self.out_dir,
            "--models", *self.models,
            "--noise", *("%.2f" % s for s in self.noise),
            "--resume",
        ]

    def expected_run_ids(self) -> list[tuple[str, str]]:
        """``(run_id, model)`` pairs by the run-id rule of scripts/run_all.py (G1)."""
        variants = config_variants(self.config)
        out: list[tuple[str, str]] = []
        for model in self.models:
            for sigma in self.noise:
                base = "%s_sigma%s" % (model, sigma_tag(sigma))
                if variants:
                    out.extend(("%s_%s" % (base, v), model) for v in variants)
                else:
                    out.append((base, model))
        return out

    def eq(self) -> float:
        return round(sum(nominal_eq(m, self.profile) for _, m in self.expected_run_ids()), 6)


def _config(stem: str) -> str:
    return "%s/%s.yaml" % (REGIME_DIR.as_posix(), stem)


def _cell_dir(root: str, cell: str, seed: int) -> str:
    return "%s/%s_seed%d" % (root, cell, seed)


def check_realistic(realistic: str) -> str:
    if realistic not in REALISTIC_CHOICES:
        raise ValueError("realistic kinetics must be one of %s, got %r" % (REALISTIC_CHOICES, realistic))
    return realistic


def smoke_queue(realistic: str) -> list[Job]:
    """Nine quick runs, 1.8 eq. The first two jobs form the concurrency pair."""
    k = check_realistic(realistic)
    spec = [
        ("k100_ie_a0", "k100_ie_a0", ("cl_pinn",)),
        ("%s_ic_as" % k, "%s_ic_as" % k, ("cl_pinn",)),
        ("%s_ic_al1" % k, "%s_ic_al1" % k, ("cl_pinn",)),
        ("k000_ic_a0", "k000_ic_a0", ("cl_pinn",)),
        ("k000_ie_a0", "k000_ie_a0", ("pinn",)),
        ("k100_ie_a0_theta", "k100_ie_a0", ("cl_pinn_theta",)),
        ("k100_ie_a0_sensors", "k100_ie_a0", ("cl_pinn",)),
        ("%s_ic_as_lstm" % k, "%s_ic_as" % k, ("lstm",)),
    ]
    return [
        Job("smoke", _config(stem), 0, "quick", _cell_dir(SMOKE_ROOT, cell, 0), models, (SIGMA,))
        for stem, cell, models in spec
    ]


def determinism_job(realistic: str) -> Job:
    """The first smoke job trained alone, charged to the contingency (0.2 eq)."""
    first = smoke_queue(realistic)[0]
    return Job("contingency", first.config, 0, "quick", SMOKE_ROOT + "/_determinism/k100_ie_a0_seed0",
               first.models, first.noise)


def core_queue(realistic: str) -> list[Job]:
    """E4, E5, E6, E7 and the E3 confirmation (37.05 eq), in execution order."""
    k = check_realistic(realistic)
    jobs: list[Job] = []

    def add(phase, stem, cell, models, seeds, noise=(SIGMA,)):
        for seed in seeds:
            jobs.append(Job(phase, _config(stem), seed, "full", _cell_dir(PINN_ROOT, cell, seed),
                            tuple(models), tuple(noise)))

    add("E4", "k000_ie_a0", "k000_ie_a0", ("cl_pinn", "pinn"), range(3))
    add("E4", "k000_ic_a0", "k000_ic_a0", ("cl_pinn",), range(3))
    add("E5", "k050_ie_a0", "k050_ie_a0", ("cl_pinn",), range(3))
    add("E5", "k100_ie_a0", "k100_ie_a0", ("cl_pinn",), range(3))
    for anchor in ("a0", "as", "al1"):
        cell = "%s_ic_%s" % (k, anchor)
        add("E6", cell, cell, ("cl_pinn",), range(3))
    add("E6", "%s_ic_as" % k, "%s_ic_as" % k, ("cl_pinn",), range(2), noise=(0.05, 0.15))
    add("E6", "%s_ic_as_lstm" % k, "%s_ic_as" % k, ("lstm",), range(3))
    add("E7", "k100_ie_a0_theta", "k100_ie_a0", ("cl_pinn_theta",), range(3))
    add("E7", "k000_ie_a0_theta", "k000_ie_a0", ("cl_pinn_theta",), range(1))
    add("E3", "k100_ie_a0_sensors", "k100_ie_a0", ("cl_pinn",), range(2))
    return jobs


def _read_json(path: Path) -> dict:
    return json.loads((REPO / path).read_text(encoding="utf-8"))


def realistic_k(path: Path = GATE_D3) -> str:
    return check_realistic(_read_json(path)["realistic_k"])


def primary_smoother(path: Path = NUMERICS_JSON) -> str:
    choice = _read_json(path)["primary_smoother"]
    if choice not in ("eks", "ieks"):
        raise ValueError("%s: primary_smoother is %r; resolve the numerics pass first" % (path, choice))
    return choice


def kinetic_names(path: Path = KINETIC_SUBSET) -> list[str]:
    names = [str(n) for n in _read_json(path)["names"]]
    if len(names) != 4:
        raise ValueError("%s must list exactly 4 kinetic parameters, got %s" % (path, names))
    return names


def sensor_confirmation(path: Path = SENSOR_CONFIRMATION) -> tuple[str, str]:
    data = _read_json(path)
    return str(data["drop"]), str(data["add"])


def planned_runs(job: Job) -> list[tuple[str, str]]:
    """Ask scripts.run_all itself which runs this job expands to (``--list``)."""
    from scripts import run_all

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = run_all.main([*job.run_all_argv(), "--list"])
    if code != 0:
        raise RuntimeError("run_all --list failed for %s seed %d (exit %s)" % (job.config, job.seed, code))
    runs = []
    for line in buffer.getvalue().splitlines():
        match = LIST_LINE.match(line)
        if match:
            runs.append((match["run_id"], match["model"]))
    return runs


def registration_state() -> tuple[str, str]:
    """``("ok" | "no_tag" | "drift", detail)`` for FROZEN_PATHS against PREREG_TAG."""
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False).stdout

    if git("tag", "--list", PREREG_TAG).strip() != PREREG_TAG:
        return "no_tag", "tag %s not found" % PREREG_TAG
    changed = git("diff", "--name-only", PREREG_TAG, "--", *FROZEN_PATHS)
    untracked = git("ls-files", "--others", "--exclude-standard", "--", *FROZEN_PATHS)
    paths = sorted({p for p in (changed + untracked).splitlines() if p.strip()})
    if paths:
        return "drift", "changed since %s: %s" % (PREREG_TAG, ", ".join(paths))
    return "ok", "frozen paths match %s" % PREREG_TAG


def log_deviation(reason: str, detail: str, source: str) -> None:
    path = REPO / DEVIATIONS
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("%s  %s  %s  |  %s\n" % (stamp, source, reason, detail))

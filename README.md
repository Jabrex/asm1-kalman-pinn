# Kalman filter versus physics-informed neural network for never-measured ASM1 states

Code, configurations, tests and pre-registration of a synthetic-data study on an
activated sludge facility with the Benchmark Simulation Model No. 1 (BSM1)
layout. The study asks two questions a facility engineer would ask. Which of the
55 never-measured Activated Sludge Model No. 1 (ASM1) tank states can routine
dissolved oxygen, ammonium, nitrate and solids probes support, and what limits
the rest? And when the model kinetics are textbook defaults rather than the
facility's own, does any estimator still beat holding the start state?

Four estimators receive identical sensors, start-up knowledge, influent
knowledge and model: persistence (holding the start state), open-loop
simulation, extended Kalman filtering and smoothing, and a physics-informed
neural network (PINN) trained with a staged protocol. They are compared over a
pre-registered grid of kinetic mismatch, influent knowledge and start-up
knowledge, and their per-state errors are set against three model-derived
recoverability indices.

- Manuscript: *Kalman filter versus physics-informed neural network for
  unmeasured activated sludge states*, submitted to Water Environment Research
  (2026).
- Registration: [PREREGISTRATION.md](PREREGISTRATION.md), frozen under the tag
  `v1.1.0-prereg` before any full-profile run.
- How to reproduce everything: [RUNBOOK.en.md](RUNBOOK.en.md), step 12 for this
  study.
- What every file does: [FILES.md](FILES.md).
- Archive: the generated datasets, training runs, estimator outputs, figures and
  tables are archived on Zenodo; the DOI of each release is on the GitHub
  Releases page and in the manuscript.

## What is here, what is not

Here: the ASM1 model and the BSM1 facility (`src/asm1`), the synthetic data
generator and sensor model (`src/data`), the PINN and LSTM (`src/models`,
`src/train`), the model-based estimators (`src/observers`), the metrics
(`src/eval`), the pipeline scripts (`scripts`), the run configurations
(`configs`), the tests (`tests`), the audited parameter vault (`asm1_cl-pinn`)
and the pre-registration probe scripts (`results/v11/predating`).

Not here: `results/`. It holds about 3 GB of generated data, run directories,
estimator outputs, figures and tables and is not tracked in git. Download the
Zenodo archive into `results/` or regenerate it with the RUNBOOK. The manuscript
sources are not part of the repository either.

## Study design

**Truth facilities.** Every estimator uses one nominal kinetic model, the
vault's ASM1 set at 20 C. The data come from truth facilities that share the
BSM1 layout and differ only in kinetics: K0 (the nominal set), K1 (the BSM1
15 C set) and K.25, K.5, K.75 in between (rate constants interpolated in log
space). Fifty further facilities with all 15 kinetic constants perturbed
lognormally test whether that axis is representative. The influent comes from
an in-house generator whose means match the BSM1 dry-weather load.

**Sensors.** Seven online targets (dissolved oxygen in tanks 3 to 5, ammonium in
tank 5, nitrate in tanks 2 and 5, suspended solids in tank 5) plus the return
activated sludge solids as an input. Eleven ASM1 components are never measured
in any tank, 55 states in all. Noise is multiplicative Gaussian at
sigma 0, 0.05, 0.10 and 0.15.

**Cells.** A cell is `K-I-A`: the truth kinetics, the influent knowledge (Ie
exact, Ic daily composite, Ib composite with a biased substrate split) and the
start-up anchor (A0 true start state, As nominal steady state, Al1 As updated
with a laboratory panel, Al2 Al1 plus respirometry). Three windows are scored:
days 0 to 12 (R0, primary), days 2 to 12 (R2) and a forecast over days 12 to 14
without data (F).

**Estimators.**

| Row | What it does |
| --- | --- |
| Persistence | holds the anchor mean |
| Open loop, information-matched | integrates the reduced model from the anchor with the estimators' information |
| Open loop, structure-rich | integrates the full facility with its settler; reference only |
| EKF, EKS, IEKS | extended Kalman filter, Rauch-Tung-Striebel smoother and iterated smoother on the log states; process noise tuned per cell by the innovation likelihood |
| Augmented EKF and EKS | the same with four kinetic multipliers estimated alongside the states |
| CL-PINN, single-stage PINN, PINN-theta | coordinate network with the ASM1 residual, the anchor and an integral COD and nitrogen balance in the loss; staged or single-stage training; optional kinetic multipliers |
| LSTM | two-layer network with the same inputs and no physics term; the model-free floor |

**Metric.** Normalized root-mean-square error of the never-measured group, each
component scaled by its range over days 0 to 12, and skill against persistence.
A PINN wins a comparison only when all its seeds beat the comparator.

**Size.** 60 main cells plus variant families for the model-based estimators
(1068 estimator runs and 957 reference rows), seven PINN cells with three seeds
each (39 PINN and LSTM runs), and a recoverability analysis of all 55 states.

## Registered hypotheses and outcomes

| | Hypothesis | Outcome |
| --- | --- | --- |
| H1 | at true kinetics the smoother is at least as accurate as the PINN | refuted |
| H2 | the open loop falls behind persistence at a mismatch fraction between 0.25 and 0.75 | supported (0.33) |
| H3 | joint kinetic estimation lowers the error at K1 for both families | supported |
| H4 | the laboratory panel helps both families, most for long-memory states | not supported |
| H5 | composite influent costs more for influent-slaved states than for biomass | supported |
| H6 | the recoverability index ranks the smoother's per-state error | not supported |
| H7 | staged training beats single-stage training | supported |

In short: at textbook kinetics the probes constrained 1 of 55 states; with
correct kinetics simulation was the most accurate estimator and the PINN beat
the smoother; under kinetic mismatch every plain estimator fell behind
persistence, the smoother first; estimating four kinetic multipliers rescued the
smoother only with exact influent data. Kinetic calibration limited accuracy
more than the choice among estimators, and influent characterization came next.

## Provenance

Two sources, kept strictly separate.

### The ASM1 model: the audited vault

Everything about the biology comes from `asm1_cl-pinn/data/asm1.json`, which is
generated from `asm1.xlsx` and independently audited:

| Artifact | SHA-256 |
| --- | --- |
| `asm1.xlsx` | `dff2424c5fa1ed83846ebac7269ac3284317dc8799f18d7edaabb18d60ba892a` |
| `data/asm1.json` | `06f7bfd5ce5703f5745cc0565858fc147012e68b3e28fdca64a126e6ed7074a2` |

`src/asm1/vault_loader.py` verifies that hash on load and refuses to run on a
mismatch. The pipeline needs only the shipped JSON. Regenerating the vault from
`asm1.xlsx` (`tools/build_asm1_vault.py`) additionally needs Node and the
`@oai/artifact-tool` package that `tools/extract_asm1_artifact.mjs` imports;
`tools/tests` skips itself when those are not available. No ASM1 parameter,
stoichiometric coefficient or rate expression is hard-coded anywhere in this
project: the eight rate laws are compiled from the vault's own `code_expression`
strings, so the ODE that generates the data, the reduced model inside the
Kalman filters and the physics term inside the PINN evaluate the same audited
text.

Two vault-specific features are carried through unchanged, both documented in
the vault's own source-anomaly table: `KNH_H = 0.05` adds an ammonium Monod
switch to heterotrophic growth (taken from ASM2d, not in the original ASM1), and
`S_N2` is carried as a 14th component. The vault flags two workbook cells as
missing alkalinity kinetic terms that the source deliberately leaves
uncorrected; no term is invented here either, and the charge balance is
reported rather than enforced.

### The facility: BSM1

Tank volumes, flow rates, aeration coefficients, settler geometry, Takacs
settling parameters and the influent composition come from:

> J. Alex, L. Benedetti, J. Copp, K.V. Gernaey, U. Jeppsson, I. Nopens,
> M.N. Pons, J.P. Steyer, P. Vanrolleghem, *Benchmark Simulation Model no. 1
> (BSM1)*, IWA Task Group on Benchmarking of Control Strategies for WWTPs.

Aeration is the BSM1 open-loop default (`KLa` 240, 240 and 84 per day), not a
dissolved oxygen controller, so the oxygen traces carry the oxygen uptake
directly.

### Two parameter sets, one nominal model

The nominal model, and therefore every estimator, uses the vault's 20 C
constants. The BSM1 15 C kinetic set enters only as the far end of the truth
axis (K1) in `src/asm1/truth_plants.py`; no estimator ever sees those
parameters. Driven by the constant BSM1 influent, the K1 facility reproduces the
published BSM1 open-loop steady state within pre-declared tolerances
(`scripts/verify_bsm1_truth.py`). The axis names two parameter sets, not a
temperature: two of the constants move against the Arrhenius direction.

## The PINN

The loss is

`L = lambda_d L_data + lambda_p L_physics + lambda_ic L_ic + lambda_b L_balance`

with the misfit to the seven targets, the ASM1 residual of the reduced reactor
train at random collocation times, the misfit to the anchor weighted by its
inverse log-variance, and an integral COD and nitrogen balance. Because influent
flow and composition are network inputs, the residual uses the total time
derivative along the influent trajectory; `tests/test_total_derivative.py`
checks it. Sensor readings are not inputs; they enter only the loss.

Staged training (CL-PINN): 20,000 Adam steps split 15/20/25/40 % over a
constant-load stage with a 1-day window and three dry-weather stages with
windows of 3, 7 and 12 days; observations smoothed over 9, 5, 3 and 1 samples;
data weight 10 to 1, physics weight 0.05 to 1. The single-stage PINN uses the
final settings for the same step count. PINN-theta adds four trainable kinetic
multipliers. The LSTM has the same inputs and output head and no physics term.

## Pre-registration

`PREREGISTRATION.md` fixes the hypotheses, the estimator settings, the grid and
the decision rules; `scripts/write_prereg.py` renders it. The run scripts refuse
to start a full-profile run when the frozen files differ from the tag. The
registration was not blind: a pilot phase preceded it, and its Section 12 lists
every result known at that point. Two deviations logged after the tag are
recorded in `results/v11/deviations.log` (in the archive) and in the
manuscript's Supporting Information.

## Layout

```
asm1.xlsx            source workbook, read-only
asm1_cl-pinn/        audited parameter vault: data/asm1.json (+ schema) and its
                     Markdown views; the loader enforces the hash
tools/               vault generator and its tests
src/asm1/            vault loader, ASM1 kinetics, BSM1 facility, truth kinetics, continuity
src/data/            influent generator, influent views, ODE simulation, sensor model
src/models/          features, PINN, LSTM, losses
src/observers/       reduced model, EKF/EKS/IEKS, anchors, cell inputs, sensitivity indices
src/train/           staged schedule, training loop
src/eval/            metrics, run collection
scripts/             pipeline steps, gates and analysis (see FILES.md)
configs/             base and ablation configs, regime cells, estimator grid
tests/               test suite (pytest)
results/v11/predating/  pre-registration probes (the only tracked part of results/)
results/             generated data, runs, outputs (not tracked; Zenodo archive)
```

## Requirements

Python 3.11, PyTorch (a CUDA build for the PINN runs; CPU works, slower),
NumPy, SciPy, Matplotlib, PyYAML and pytest (`requirements.txt`). The reference
machine was a laptop with an Intel Core i7-14700HX and an NVIDIA RTX 4050 (6 GB).

```bash
uv venv --python 3.11
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install -r requirements.txt
python -m pytest tests -q
```

## License

MIT, see [LICENSE](LICENSE).

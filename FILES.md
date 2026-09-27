# File guide

What every tracked file in this repository does. The code files carry no
comments; this guide and the module docstrings are the documentation. The
generated datasets, training runs, estimator outputs, figures and tables live in
`results/`, which is not tracked in git (see the `.gitignore`); they are archived
on Zenodo under the concept DOI 10.5281/zenodo.22304281 and can be regenerated
with [RUNBOOK.en.md](RUNBOOK.en.md).

## Top level

| File | Purpose |
| --- | --- |
| `README.md` | Project overview, provenance of the ASM1 constants, and how the pieces fit together. |
| `RUNBOOK.en.md` | Ordered steps that regenerate every dataset, run, table and figure; each step is a gate for the next. |
| `PREREGISTRATION.md` | The registration of the v1.1 regime study (hypotheses H1-H7, decision rules, what was known beforehand), frozen under the tag `v1.1.0-prereg`. |
| `FILES.md` | This guide. |
| `LICENSE` | License of the code. |
| `requirements.txt` | Python dependencies. |
| `asm1.xlsx` | Source workbook of the ASM1 model from which the audited vault JSON is extracted (its SHA-256 is checked on load). |
| `.gitignore` | Excludes environments, the manuscript folders, and all generated results and images. |

## `asm1_cl-pinn/` - the audited ASM1 vault

| File | Purpose |
| --- | --- |
| `asm1_cl-pinn/data/asm1.json` | The single source of every ASM1 constant, stoichiometric coefficient and rate expression; hash-locked and loaded by `src/asm1/vault_loader.py`. |
| `asm1_cl-pinn/data/asm1.schema.json` | JSON schema the vault file must satisfy. |
| `asm1_cl-pinn/README.md` | How the vault was built and audited. |
| `asm1_cl-pinn/Source Manifest.md` | Provenance of every value (report, table, page). |
| `asm1_cl-pinn/Parameters.md` | The 25 kinetic and stoichiometric constants. |
| `asm1_cl-pinn/State Variables.md` | The 14 ASM1 components and their units. |
| `asm1_cl-pinn/Processes and Rates.md` | The eight process rate expressions. |
| `asm1_cl-pinn/Composition and Continuity.md` | Composition matrix and continuity checks (COD, nitrogen, charge). |
| `asm1_cl-pinn/Corrected Matrices.md` | Stoichiometric matrix after the audit corrections. |
| `asm1_cl-pinn/Kinetic Checking Matrix.md` | Cross-check table of the kinetic expressions. |
| `asm1_cl-pinn/Audit Report.md` | Findings of the independent audit of the vault. |

## `tools/` - vault build

| File | Purpose |
| --- | --- |
| `tools/build_asm1_vault.py` | Builds `asm1_cl-pinn/data/asm1.json` from `asm1.xlsx` through the Node extractor, verifies the hashes and writes the manifest. |
| `tools/extract_asm1_artifact.mjs` | Node script that reads the workbook cells the vault build needs. |
| `tools/tests/test_asm1_vault.py` | Unit tests of the vault build; skip themselves when Node or the extractor package is missing. |

## `src/` - library code

### `src/asm1/` - the model and the simulated facility

| File | Purpose |
| --- | --- |
| `src/asm1/vault_loader.py` | Loads the vault JSON, verifies its SHA-256 and exposes the constants and expressions. |
| `src/asm1/model.py` | ASM1 process rates and conversion rates, compiled from the vault's expression strings. |
| `src/asm1/plant.py` | BSM1 facility layout: five reactors in series plus the ten-layer Takacs settler and the recycles. |
| `src/asm1/truth_plants.py` | Alternative kinetic parameter sets (BSM1 15 C, the mismatch axis, random perturbations) used only to generate truth data. |
| `src/asm1/continuity.py` | COD, nitrogen and charge continuity diagnostics of the stoichiometric and composition matrices. |
| `src/asm1/__init__.py` | Package marker. |

### `src/data/` - synthetic data

| File | Purpose |
| --- | --- |
| `src/data/influent.py` | Influent generator with diurnal and weekly components, matched to the BSM1 dry-weather load, plus a rain scenario. |
| `src/data/influent_views.py` | What an estimator is told about the influent: exact, daily composite, or composite with a biased substrate split. |
| `src/data/sensors.py` | Sensor set, the measured and never-measured split, and the multiplicative noise model. |
| `src/data/simulate.py` | Ground-truth ODE simulation: warm-up, dynamic scenarios and dataset files. |
| `src/data/__init__.py` | Package marker. |

### `src/models/` - the networks

| File | Purpose |
| --- | --- |
| `src/models/pinn.py` | Coordinate physics-informed network for the five-reactor state trajectory. |
| `src/models/lstm.py` | LSTM baseline with the same inputs and output head and no physics term. |
| `src/models/features.py` | Input features shared by every model. |
| `src/models/losses.py` | Data, physics-residual, anchor and balance losses, including the total time derivative. |
| `src/models/__init__.py` | Package marker. |

### `src/observers/` - model-based estimators

| File | Purpose |
| --- | --- |
| `src/observers/reduced_model.py` | Reduced reactor-train model shared by the Kalman estimators and the PINN residual. |
| `src/observers/ekf.py` | Extended Kalman filter, Rauch-Tung-Striebel smoother, iterated smoother and the augmented variants. |
| `src/observers/anchors.py` | Start-state anchors: nominal steady state, laboratory panel and respirometry updates. |
| `src/observers/cell_inputs.py` | Per-cell inputs of the estimator grid (kinetics, influent view, anchor, sensor set). |
| `src/observers/pipeline.py` | Runs the estimators on one dataset and one anchor with the same information the PINN receives. |
| `src/observers/sensitivity.py` | Linearized information analysis: Fisher information, start-up memory and influent forcing share. |
| `src/observers/__init__.py` | Package marker with the estimator overview. |

### `src/train/` and `src/eval/`

| File | Purpose |
| --- | --- |
| `src/train/curriculum.py` | Staged training schedule: window, scenario and loss-weight ramps. |
| `src/train/run.py` | One training run: model and noise level to checkpoint, history and metrics. |
| `src/train/__init__.py` | Package marker. |
| `src/eval/metrics.py` | Error metrics for the measured and never-measured groups, level errors and skill. |
| `src/eval/report.py` | Collects finished runs into the benchmark tables and figures. |
| `src/eval/__init__.py` | Package marker. |
| `src/__init__.py` | Package marker. |

## `scripts/` - pipeline steps and analysis

| File | Purpose |
| --- | --- |
| `scripts/verify_vault.py` | RUNBOOK steps 1 and 2: vault loader and continuity verification. |
| `scripts/verify_solver.py` | RUNBOOK step 3: solver verification gate at 20 C. |
| `scripts/generate_data.py` | RUNBOOK steps 4 and 5: truth trajectories and sensor datasets at every noise level. |
| `scripts/generate_random_truths.py` | Fifty random kinetic-mismatch truth facilities for the estimator ensemble. |
| `scripts/verify_model.py` | RUNBOOK step 6: model-side verification before any benchmark run. |
| `scripts/verify_truth_identity.py` | Identity checks of the v1.1 truth generator against the nominal facility. |
| `scripts/verify_bsm1_truth.py` | BSM1 gate: does the 15 C truth facility reproduce the published open-loop steady state? |
| `scripts/run_all.py` | RUNBOOK steps 7 and 8: the training sweep. |
| `scripts/run_core.py` | Smoke and core GPU queues of v1.1 under a budget guard. |
| `scripts/run_observers.py` | Runs the estimators over the regime cells in parallel processes. |
| `scripts/observer_grid.py` | Writes, runs and checks the CPU estimator grid. |
| `scripts/observer_numerics.py` | Filter divergence checks and the primary-smoother choice. |
| `scripts/make_anchors.py` | Anchor files: nominal ensemble, laboratory panel and respirometry updates. |
| `scripts/make_baselines.py` | Reference rows (persistence, open loops) as scored run directories. |
| `scripts/make_regime_configs.py` | Writes the regime-cell configurations for the GPU runs. |
| `scripts/make_report.py` | RUNBOOK step 9: scores every finished run and writes the benchmark report. |
| `scripts/v11_plan.py` | Shared run plan of the v1.1 execution phase. |
| `scripts/gpu_ledger.py` | GPU budget ledger of the v1.1 runs. |
| `scripts/smoke_check.py` | Pass/fail gate of the smoke runs (code paths and protocol keys only). |
| `scripts/check_core.py` | Acceptance check of the core GPU runs. |
| `scripts/check_history.py` | Smoke-run check: every logged loss term is finite and the physics term fell. |
| `scripts/concurrency_check.py` | Throughput and determinism check before running two GPU processes. |
| `scripts/gate_d3.py` | Gate D3: where the realistic PINN cells sit on the mismatch axis. |
| `scripts/recoverability.py` | Recoverability indices of the never-measured states and sensor-subset ranking. |
| `scripts/recoverability_validation.py` | Does the recoverability index predict each estimator's per-state error (H6)? |
| `scripts/regime_map.py` | Scores every cell on the three windows and draws the regime map (Fig. 6). |
| `scripts/v11_figures.py` | Main-text Figures 2-5 from saved JSON only. |
| `scripts/v11_tables.py` | Tables T1-T3, the Supporting Information tables and the manuscript number registry. |
| `scripts/figure_layout.py` | Print-size layout and text-collision check for every figure. |
| `scripts/seed_bands.py` | Multi-seed PINN runs aggregated into min-max bands. |
| `scripts/component_results.py` | Per-component and per-tank errors, heatmaps and trajectories. |
| `scripts/residual_diagnostic.py` | Physics residual of every PINN checkpoint inside and beyond the window. |
| `scripts/audit_derivative.py` | Audit of the v1.0 checkpoints for the partial-derivative defect. |
| `scripts/reanalyse_v1.py` | Re-scores the v1.0 runs on the v1.1 windows and metrics. |
| `scripts/split_loss_curves.py` | Training-loss figures split by architecture. |
| `scripts/record_default_fingerprint.py` | Bitwise fingerprint of short default-config training runs. |
| `scripts/dump_kinetics.py` | Prints the vault's rates and stoichiometric rows for the appendix. |
| `scripts/write_prereg.py` | Renders `PREREGISTRATION.md` from `scripts/prereg_template.md`. |
| `scripts/prereg_template.md` | Template of the registration document. |

## `configs/` - run configurations (YAML)

| Path | Purpose |
| --- | --- |
| `configs/base.yaml` | Default training configuration. |
| `configs/ablation_*.yaml` | Ablation variants (axes, flow-only inputs, initial-condition mask). |
| `configs/analysis/recoverability_cells.yaml` | Cells analysed by the recoverability scripts. |
| `configs/regime/*.yaml` | The v1.1 regime cells for the GPU runs. |
| `configs/observers/*.yaml` | Defaults of the estimator grid. |
| `configs/observers/grid/main/` | The 60 main-grid cells (five kinetics, three influent modes, four anchors). |
| `configs/observers/grid/random/` | The 50 random-mismatch cells. |
| `configs/observers/grid/lab/` | Laboratory-panel seed variants. |
| `configs/observers/grid/sensors/` | Sensor-subset variants. |
| `configs/observers/grid/realisations/` | Further noise realizations. |
| `configs/observers/grid/sigma_log/` | Anchor-ensemble spread variants. |
| `configs/observers/grid/settler/` | Ideal-settler variant. |
| `configs/observers/grid/m0prime/` | Off-steady-state start variant. |
| `configs/observers/numerics/`, `configs/observers/numerics_ieks/` | Numerics passes of the filters and the iterated smoother. |

## `tests/` - test suite

Every `tests/test_*.py` file covers the module or script named in its file name;
the docstring at the top of each file states the contract it checks.
`tests/conftest.py` holds the shared fixtures, `tests/anchor_utils.py` writes
anchor files for the tests, `tests/v10_reference.py` reproduces the v1.0
training-loss reference, and `tests/data/` holds three small JSON fixtures: the
published BSM1 open-loop steady state, the v1.0 default fingerprint and the v1.0
three-step losses.

## `results/v11/predating/` - pre-registration probes

The only tracked files under `results/`: the four probe scripts run before the
registration and their short outputs, cited in `PREREGISTRATION.md`.

| File | Purpose |
| --- | --- |
| `results/v11/predating/o1_precheck.py`, `o1_precheck_output.txt` | Does a six-layer soluble lag in the recycle closure cut the open-loop error by more than half? |
| `results/v11/predating/q_scan_extended.py`, `q_scan_extended.json` | Process-noise scan at K0-Ie-A0 on the measured channels (gate D4). |
| `results/v11/predating/lab_panel_seed_spread.py`, `lab_panel_seed_spread_output.txt` | Spread of the laboratory-panel anchor over seeds and against a full-covariance update. |
| `results/v11/predating/drift_check.py` | Mass-neutral inert-solids shift check that led to the trajectory drift in the sensitivity analysis. |

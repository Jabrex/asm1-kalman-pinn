# RUNBOOK — order of execution

This file lists the steps, in order, that regenerate every dataset, run, table
and figure in this repository.

Rule: **if a step reports FAIL, do not move on to the next one.** Every gate
verifies an assumption that the following step relies on.

All commands run from the project root (the directory this repository was
cloned into).

---

## Summary table

| # | Command | Time (approx.) | Produces |
|---|---|---|---|
| 0 | environment set-up | 5–15 min | `.venv/` |
| 1 | `python -m scripts.verify_vault` | < 5 s | — (gate) |
| 2 | `python -m pytest tests -q` | 1–3 min | — (gate) |
| 3 | `python -m scripts.verify_solver` | 5–15 min | — (gate) |
| 4 | `python -m scripts.generate_data` | 3–8 min | `results/raw/*.npz` |
| 5 | `python -m pytest tests -q` (again) | 1–3 min | — (gate) |
| 6 | `python -m scripts.verify_model` | 5–15 min | — (gate) |
| 7 | `python -m scripts.run_all --profile quick` | 40–70 min | `results/runs/*` |
| 8 | `python -m scripts.make_report` | < 1 min | `results/benchmark.*` |
| 9 | `python -m scripts.run_all --profile full` | 1.5–7 h | `results/runs/*` |
| 10 | `python -m scripts.make_report` | < 1 min | `results/benchmark.*` |
| 11 | replication, ablation and baseline runs (see below) | 2–4 h | `results/runs_*`, `results/seed_bands.json`, paper figures |

Steps 7 + 8 prove cheaply that the pipeline runs end to end. The numbers that
enter the report come from steps 9–11. The numbering matches the
`step N clear` messages printed by the scripts.

---

## 0 — Environment

The reference machine was a laptop with an NVIDIA RTX 4050 (6 GB) and a
CUDA 12.4 PyTorch build; the full sweep took 94 min there. Everything also runs
on CPU, several times slower.

```bash
uv venv --python 3.11
```

Activate the environment (PowerShell shown; use `source .venv/bin/activate` on
Linux/macOS):

```bash
.venv\Scripts\Activate.ps1
```

Install PyTorch as a CUDA build (this single line downloads about 2.5 GB):

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Remaining dependencies:

```bash
uv pip install -r requirements.txt
```

**Pass criterion:** the following prints `True`.

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If it prints `False`, the pipeline still works on CPU, but step 9 takes several
times longer. Set `device: cpu` in `configs/base.yaml`.

---

## 1 — Vault verification

```bash
python -m scripts.verify_vault
```

**What it does.** Compares the SHA-256 of `asm1_cl-pinn/data/asm1.json` with
the audited value recorded in the vault's own `Audit Report.md`. Then checks
every one of the 25 parameters and every cell of the 8 × 14 stoichiometric
matrix against the Markdown views generated from the same source. Finally
recomputes the continuity residual `nu @ C`.

**Pass criterion.** The output ends with `PASS - steps 1 and 2 clear`. Zero
differences, maximum residual `5.5511151231257827e-17`, tolerance `1e-15`.

**If it fails.** The vault files have changed. Do not hand-edit anything under
`asm1_cl-pinn/`; regenerate with `tools/build_asm1_vault.py` and update
`Audit Report.md`. No number is trustworthy until this gate passes.

---

## 2 — Unit tests

```bash
python -m pytest tests -q
```

**What it does.** Vault contract, kinetic expressions, plant geometry,
influent generator, sensor model, curriculum scheduler, loss terms and the
leakage discipline.

`tests/test_leakage.py` has two layers. The **static layer** needs no data and
always runs: it tokenises the source and verifies that every access to
`truth_reactor` / `truth_y` in *all* modules under `src/train/` and
`src/models/` is exactly `[0]`, and that every file under `src/` touching
ground truth is classified (a new file cannot be added silently). The
**runtime layer** needs data and is `skipped` at this point; that is expected,
it really runs in step 5.

**Pass criterion.** Everything `passed` or `skipped`; no `failed`.

**Tests worth knowing about:**
- `test_physics_residual_vanishes_on_the_true_dynamics` — if the physics term
  is built correctly, feeding it the true dynamics must give a zero residual.
  If this test fails, everything the PINN learns is wrong.
- `test_budget_parity_between_curriculum_and_baseline` — curriculum and
  single-stage runs receive the same total step budget.
- `test_physics_weight_is_never_zero` — the full-PINN guarantee.

---

## 3 — Solver verification gate

```bash
python -m scripts.verify_solver
```

**What it does.** Runs the 200-day warm-up (100 days is not enough at 20 °C;
`WARMUP_DAYS = 200`), then six gates:

| Gate | Check | Tolerance |
|---|---|---|
| 3a | BDF vs Radau vs LSODA | reactor + solubles 1e-6, settler solids 1e-4 |
| 3b | rtol 1e-8 → 1e-10 → 1e-12 convergence | reactor + solubles 1e-6, settler solids 1e-4 |
| 3c | reactions off + recycles off ↔ matrix exponential (`scipy.linalg.expm`) | 1e-6 |
| 3d | tracer mass conservation: `S_I` (reactions on) / `X_I` (reactions off) / `X_I` (reactions on, eq. 46 approximation) | 1e-8 / 1e-8 / 5e-2 |
| 3e | `‖dy/dt‖/‖y‖` after warm-up | 1e-6 |
| 3f | different initial seed → same equilibrium | 1e-6 |

The separate (looser) tolerance on settler solids is a documented property of
the piecewise-continuous right-hand side of the Takács model; the `X_I`
reactions-on bound is the measured size of the BSM1 eq. 46 approximation, not a
solver error.

There is **deliberately no** comparison with BSM1 Table 6: this project runs the
20 °C vault parameters, whereas BSM1 Table 6 was produced with the 15 °C set.
Gates 3c and 3d are temperature-independent references that share no code with
`solve_ivp`.

**Pass criterion.** `PASS - step 3 clear`.

**If it fails.**
- 3a/3b → tighten the tolerances (`SolverSettings`); the system may be stiffer
  than expected.
- 3c → error in the transport terms in `src/asm1/plant.py`.
- 3d → error in the settler mass balance; inspect `_settler_streams` and the
  implementation of eq. 34–44.
- 3e → 200 days was not enough; increase `WARMUP_DAYS`.

---

## 4 — Synthetic data generation

```bash
python -m scripts.generate_data
```

**What it does.** Derives three scenarios from a single 200-day warm-up, then
turns each into a sensor dataset at four noise levels.

**Produces** (under `results/raw/`):

```
sim_constant.npz  sim_dry.npz  sim_rain.npz        ground truth
obs_constant_sigma0p00 / 0p05 / 0p10 / 0p15 .npz   curriculum stage 1
obs_dry_sigma0p00 / 0p05 / 0p10 / 0p15 .npz        training + holdout
obs_rain_sigma0p00 / 0p05 / 0p10 / 0p15 .npz       distribution-shift test
manifest.json                                       provenance
```

**Pass criterion.** `PASS - steps 4 and 5 complete`. Also check on screen:

- **Reactor COD/N closure** must be below `1e-6`; the script **gates** this.
  A larger value means a leak in the settler mass balance.
- **Plant-wide closure** is only **reported**, not gated: because of the BSM1
  eq. 46 settler approximation, about 1e-3 may appear on the nitrogen side.
  Expected behaviour, not an error.
- **Mean flow** very close to 18 446; the range close to the 10 000–32 000 of
  BSM1 Figure 3 (not exact: mean and max/min ratio are pinned, absolute
  extremes are approximate by design).
- **Clipping fraction** (`clipped … % of samples`). Noise is multiplicative
  (`z·(1+eps)`), so clipping needs `eps < -1`; even at σ = 0.15 the expected
  value is ~0 (a 6.7σ event). A clearly non-zero value is abnormal; report it.

---

## 5 — Run the tests again

```bash
python -m pytest tests -q
```

**What it does.** The runtime layer of `tests/test_leakage.py` now really runs
instead of `skip`: all ground truth after t > 0 is replaced by NaN and the
losses (forward loss **and** backward gradients, at every curriculum stage, for
both architectures) are verified to remain finite. To make sure a skip does not
pass silently, run in strict mode; with data missing it then `fails` instead
of `skips`:

```bash
ASM1_STRICT_TESTS=1 python -m pytest tests -q
```

**Pass criterion.** No `failed`; the previously `skipped` tests now `passed`.

---

## 6 — Model verification

```bash
python -m scripts.verify_model
```

**What it does.** Trains small probe models in float64 on the CPU and checks
four gates:

| Gate | Check |
|---|---|
| 6a | autograd derivative ↔ central finite difference (< 1e-4) |
| 6b | forward-mode JVP ↔ reverse-mode VJP (< 1e-6) |
| 6c | gradient of the physics residual is non-zero **and** the residual is smaller with physics on |
| 6d | loss remains finite when all ground truth after t > 0 is set to NaN |

6c is the proof of the "full PINN" claim: the physics term is really wired into
the graph and really changes the outcome. 6d is the proof that the soft-sensor
set-up is honest: the ground truth of the eleven unmeasured components never
enters the loss.

**Pass criterion.** `PASS - step 6 clear`.

**If it fails.**
- 6a/6b → set `pinn.derivative_mode: reverse` in `configs/base.yaml` and retry;
  the forward-mode set-up may not carry the parameter graph in this torch
  version.
- 6c → the link between `physics_residual` in `src/models/losses.py` and the
  summation in `run.py` is broken.
- 6d → the training path is fed from an unmeasured state; `test_leakage.py`
  tells you which line.

---

## 7 — Quick sweep (pipeline check)

```bash
python -m scripts.run_all --profile quick
```

To see what would run first:

```bash
python -m scripts.run_all --profile quick --list
```

**What it does.** 16 runs (4 models × 4 noise levels), 4 000 steps each. The
aim is not to produce results but to see that every combination runs to the end
without crashing.

**Produces.** `checkpoint.pt`, `history.json`, `predictions.npz`,
`summary.json`, `config.yaml` under `results/runs/<model>_sigma<xx>/`, plus
`results/runs/sweep_index.json`.

**Pass criterion.** `16 succeeded, 0 failed`.

If interrupted, continue where you left off:

```bash
python -m scripts.run_all --profile quick --resume
```

**If it fails.** The run's directory contains `error.txt`. The two most likely
causes: GPU memory (lower `collocation_points`) and numerical overflow with
`dtype: float32` (try `float64`, slower).

---

## 8 — Interim report

```bash
python -m scripts.make_report
```

Open `results/benchmark.md`. At this point the numbers are **not meaningful**
(4 000 steps is too few), but confirm that the tables are filled, that the
Track A / Track B split is visible and that the figures were produced. An empty
cell means that run failed.

---

## 9 — Full sweep (the reported runs)

```bash
python -m scripts.run_all --profile full
```

**Time.** 16 runs; 94 min on the reference GPU, several hours on CPU. If
interrupted, continue with `--resume`.

If you leave it running overnight, make sure the machine does not go to sleep.

To run a single cell:

```bash
python -m scripts.run_all --profile full --models cl_pinn --noise 0.10
```

**Pass criterion.** `16 succeeded, 0 failed`.

---

## 10 — Final report

```bash
python -m scripts.make_report
```

**Produces.**

```
results/benchmark.csv            model × noise × evaluation set
results/benchmark.md             Track A / Track B tables
results/benchmark_detail.json    every metric per component
results/figures/loss_curves.png
results/figures/noise_robustness.png
```

**How to read it.**

- **Track A** (measured components: `S_O`, `S_NH`, `S_NO`) — the fair
  comparison of the four models. The LSTMs are expected to be competitive here.
- **Track B** (the 11 unmeasured components) — the main result. `lstm` and
  `cl_lstm` have no training signal here; their numbers reflect the initial
  scale, not a fit. The report says so in a footnote.
- **`holdout`** rows refer to the last two days, never seen in training;
  **`rain`** rows to the never-seen rain scenario. The value of the physics term
  shows most clearly here.
- **Noise columns** 0.00 → 0.15 show how much each model degrades.

---

## 11 — Replication, ablation and baseline runs (the paper's revision experiments)

Every run below reuses the step 4 data and the step 9 settings; each config
differs from `configs/base.yaml` only in the model list, the seed and a
separate `out_dir` (run ids do not carry the seed, so a shared `out_dir` would
overwrite the seed-0 sweep).

Non-learned reference predictors (persistence of the t = 0 state, open-loop ODE
integration); materialised as ordinary run directories under `results/runs/`:

```bash
python -m scripts.make_baselines
```

Seeds 1 and 2 of the two PINN rows (eight runs each):

```bash
python -m scripts.run_all --config configs/seed1.yaml
python -m scripts.run_all --config configs/seed2.yaml
```

Single-axis curriculum ablations (weights only, horizon only, scenario only,
smoothing only; σ = 0 and 0.10), the anchor-restricted run and the
influent-composition-withheld run:

```bash
python -m scripts.run_all --config configs/ablation_axes.yaml
python -m scripts.run_all --config configs/ablation_icmask.yaml
python -m scripts.run_all --config configs/ablation_flowonly.yaml
```

Score any of those sweeps with the standard report, pointing it at the
directory:

```bash
python -m scripts.make_report --runs results/runs_ablation --out results/report_ablation
```

Aggregate the three seeds into median and min–max bands and draw the banded
noise-robustness figure:

```bash
python -m scripts.seed_bands            # results/seed_bands.json, results/figures/noise_robustness_bands.*
```

Paper figures and the per-component / per-tank tables:

```bash
python -m scripts.split_loss_curves     # results/figures/loss_curves_pinn.*, loss_curves_lstm.*
python -m scripts.component_results     # results/component_table.{json,tex}, per_tank_heatmap.*, trajectories_trackB.*
python -m scripts.dump_kinetics         # prints the vault's rate expressions and stoichiometry (appendix table)
```

All figures are written as 600 dpi PNG plus a vector PDF twin by
`src.eval.report.save_figure`.

---

## Frequently used helpers

Run a single training configuration directly:

```bash
python -m src.train.run results/runs/cl_pinn_sigma0p10/config.yaml
```

Run only the tests:

```bash
python -m pytest tests -q
```

Regenerate the data for one scenario:

```bash
python -m scripts.generate_data --scenarios dry
```

---

## Starting from scratch

Deleting `results/` makes everything regenerable from the beginning; the vault
and `asm1.xlsx` are never modified by any step, they are read-only sources.

```bash
rm -rf results/raw results/runs results/runs_seed1 results/runs_seed2 results/runs_ablation results/runs_icmask results/runs_flowonly results/figures results/benchmark.* results/seed_bands.json results/component_table.*
```

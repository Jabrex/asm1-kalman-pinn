# ASM1 · Curriculum Learning + PINN

Synthetic-data study of a physics-informed neural network as a soft sensor for an
activated-sludge plant, trained with a hierarchical curriculum and benchmarked
against LSTM baselines.

**To run it: see [RUNBOOK.md](RUNBOOK.md).** The pipeline has been executed
end-to-end; `results/` holds the generated datasets, all training runs
(including the three-seed replication and the ablations), the benchmark tables
and the figures reported in the accompanying manuscript. The RUNBOOK gives the
exact order to regenerate everything from scratch.

---

## Provenance

Two sources, kept strictly separate.

### 1. The ASM1 model — the audited Obsidian vault

Everything about the biology comes from `asm1_cl-pinn/data/asm1.json`, which is
generated from `asm1.xlsx` and independently audited:

| Artifact | SHA-256 |
| --- | --- |
| `asm1.xlsx` | `dff2424c5fa1ed83846ebac7269ac3284317dc8799f18d7edaabb18d60ba892a` |
| `data/asm1.json` | `06f7bfd5ce5703f5745cc0565858fc147012e68b3e28fdca64a126e6ed7074a2` |

`src/asm1/vault_loader.py` verifies that hash on load and refuses to run on a
mismatch. **No ASM1 parameter, stoichiometric coefficient or rate expression is
hard-coded anywhere in this project** — the eight rate laws are compiled from the
vault's own `code_expression` strings, so the ODE that generates the data and the
physics term inside the PINN are literally the same audited text.

Two vault-specific features are carried through unchanged, both documented in the
vault's own source-anomaly table:

- `KNH_H = 0.05` adds an ammonium Monod switch to heterotrophic growth
  (`rho_1`, `rho_2`). The vault records it as taken from ASM2d; it is not in the
  original ASM1.
- `S_N2` is carried as a 14th component although the source state table lists 13.

The vault flags cells `X82`/`X84` as missing alkalinity kinetic terms that the
workbook deliberately leaves uncorrected. No term is invented here either; the
charge balance is reported rather than enforced.

### 2. The plant — BSM1

Tank volumes, flow rates, aeration coefficients, settler geometry, Takacs
settling parameters and the influent composition come from:

> J. Alex, L. Benedetti, J. Copp, K.V. Gernaey, U. Jeppsson, I. Nopens,
> M.N. Pons, J.P. Steyer, P. Vanrolleghem, *Benchmark Simulation Model no. 1
> (BSM1)*, IWA Task Group on Benchmarking of Control Strategies for WWTPs, 2018.

**BSM1's kinetic and stoichiometric parameters are NOT used.** Those are a 15 °C
set; this project runs the vault's 20 °C set everywhere. Geometry, flows and
influent composition are temperature independent, which is what makes the
combination legitimate.

---

## The 20 °C rule

There is exactly one parameter set in this project. No temperature switch, no
Arrhenius correction, no "BSM1 compatibility mode" exists in the code.

A direct consequence: **the steady state produced here does not match BSM1
Table 6, and should not.** Faster kinetics settle on a different operating point.
The solver is therefore verified against temperature-independent references
instead — a matrix exponential, passive tracer closure, cross-solver agreement
and tolerance convergence (RUNBOOK step 3).

A second consequence to watch for: at 20 °C oxygen uptake is higher, so BSM1's
fixed `KLa = 240 /d` may not hold dissolved oxygen up in tanks 3 and 4. If DO
collapses there, that is reported as measured. `KLa` is not adjusted to make the
result look better.

---

## What the models see

This is a soft-sensor problem, not a curve fit.

**Measured (noisy), 8 channels**

`S_O` in tanks 3/4/5, `S_NH` in tank 5, `S_NO` in tanks 2/5, `TSS` in tank 5,
`TSS` in the return sludge.

Seven of these are prediction targets. `TSS_ras` is a settler measurement used as
an *input* to the recycle reconstruction — the models span the reactor train
only, so it would not be a fair target.

**Known exactly (not sensors)**

`Q_in(t)` and the influent composition `Z_in(t)`, the pump flows, the aeration
coefficients, and the initial state `Z(0)`. Influent characterisation is a
standard given in activated-sludge modelling — BSM1 itself distributes it as an
input file — and `Z(0)` is supplied identically to all four models.

**Never measured, 11 components**

`S_I`, `S_S`, `X_I`, `X_S`, `X_B_H`, `X_B_A`, `X_P`, `S_ND`, `X_ND`, `S_ALK`,
`S_N2`, in every tank. Five of them are constrained only as a weighted sum inside
the two TSS channels. These are the Track B targets, and the physics term is the
only route to them.

---

## The benchmark

4 models × 4 noise levels = 16 runs.

|  | σ = 0 | σ = 0.05 | σ = 0.10 | σ = 0.15 |
| --- | --- | --- | --- | --- |
| `cl_pinn` — PINN + curriculum | ✓ | ✓ | ✓ | ✓ |
| `pinn` — PINN, no curriculum | ✓ | ✓ | ✓ | ✓ |
| `cl_lstm` — LSTM + curriculum | ✓ | ✓ | ✓ | ✓ |
| `lstm` — LSTM, no curriculum | ✓ | ✓ | ✓ | ✓ |

Evaluated on three sets: training days 0–12, holdout days 12–14, and an unseen
rain scenario (peak ≈ 52 000 m³/d).

Noise is multiplicative Gaussian, `z_obs = z_true · (1 + ε)`, `ε ~ N(0, σ²)`,
clipped at zero, with the clipped fraction recorded.

### It is a full PINN

`L = λ_data·L_data + λ_phys·L_physics + λ_ic·L_ic + λ_pos·L_pos + λ_bal·L_balance`

`L_physics` is `‖dZ/dt − f_ASM1(Z, u(t))‖²` at collocation points, with `dZ/dt`
taken by autograd and `f_ASM1` evaluated from the vault. `λ_phys` is ramped by
the curriculum but never reaches zero — `tests/test_curriculum.py` asserts this,
and RUNBOOK step 5 gate 6c proves the term is wired to the parameters and changes
the outcome.

`L_balance` is the integral COD and N closure over the training window. The
*pointwise* continuity residual would be structurally zero — `r @ C = ρ @ (ν @ C)`
and `ν @ C` vanishes by the vault audit — so it would supervise nothing. The
integral form ties the trajectory endpoints to the accumulated boundary fluxes and
does carry signal.

### Curriculum: three tiers

| Stage | Scenario | Horizon | λ_physics |
| --- | --- | --- | --- |
| 1 | constant load | 1 day | low |
| 2 | diurnal | 3 days | ↗ |
| 3 | diurnal | 7 days | ↗ |
| 4 | diurnal | 12 days | full |

The no-curriculum runs get a single stage at the final settings with the
*identical* total step budget, so the difference is ordering, not compute.

---

## Deviations from the approved plan

Three, each with its reason. All are visible in the code and its tests.

1. **Aeration uses the BSM1 open-loop default** (`KLa_3 = KLa_4 = 240 /d`,
   `KLa_5 = 84 /d`) rather than the closed-loop DO controller the plan named.
   The BSM1 report does not publish the PI gains, and inventing gains would break
   the rule that no unsourced number enters the model. The open-loop values are
   published (report p.14).

2. **Noise is not a curriculum axis.** The plan sketched stage 2 as
   "diurnal, noise-free". That would hand every CL run a look at clean
   observations the no-CL runs never get, turning the CL comparison into a
   data-quality comparison. Instead the noise level is fixed per run and early
   stages see a *smoothed* view of the same noisy signal — an easier view of
   identical information.

3. **The Effluent Quality Index is a dataset descriptor, not a model metric.**
   EQI lives downstream of the clarifier; the models span the reactor train, so
   scoring them on it would require feeding them settler ground truth. It is
   computed on the ground-truth trajectory and reported alongside the influent
   statistics.

Two further judgement calls worth knowing about:

- The **initial condition** `Z(0)` is supplied to every model as a boundary
  condition, and the output scale is derived from it. It is not leakage — it is
  given identically to all four — but it is an assumption, and inference over
  12 days from 8 sensors remains a genuine problem.
- The **influent daily-curve shape** (where the peaks sit, how sharp they are,
  how far the load peak leads the flow peak, how much of the rain event is spent
  rising) is not sourced, and `ProfileShape` / `RainEvent.rise_fraction` say so.
  All *magnitudes* are sourced: means come from BSM1 Table 5, extremes from
  Figure 3, the rain peak from Figure 5. The numbers read off a figure rather
  than a table — flow min/max, the weekend peak factor, and the rain event's
  start day and duration — are labelled as figure readings in
  `src/data/influent.py`.

---

## Layout

```
asm1.xlsx            source workbook          read-only
asm1_cl-pinn/        audited Obsidian vault   read-only
tools/               vault generator          read-only

src/asm1/            vault loader, ASM1 kinetics, BSM1 plant, continuity
src/data/            influent generator, ODE simulation, sensor model
src/models/          features, PINN, LSTM, losses
src/train/           curriculum schedule, training loop
src/eval/            metrics, benchmark report
scripts/             the numbered RUNBOOK entry points
configs/base.yaml    shared run settings, expanded over the sweep
tests/               unit tests, including the leakage invariants
results/             generated data, runs, figures, benchmark tables
```

# Pre-registration of the v1.1 regime study

Registered on $registered_utc (UTC). Repository state at registration: commit `$base_commit` on branch `v1.1`; the commit that adds this file carries the tag `v1.1.0-prereg`.

This file fixes the hypotheses, the cells, every estimator setting, the metrics and the decision rules of the v1.1 study. It was committed before any full-profile (20 000-step) PINN run of v1.1 and before any run of the final model-based estimator grid (`configs/observers/grid/`). It was rendered by `scripts/write_prereg.py` from `scripts/prereg_template.md`, and every number it quotes from an earlier computation comes from a file listed with its SHA-256 digest in Section 12. Sections 1 to 11 do not change after the tag. Any deviation is appended to `results/v11/deviations.log` with a UTC time stamp and a reason, and every entry is reported in the Supplementary Information.

## 1. Question

For a BSM1-layout activated sludge plant monitored by routine dissolved-oxygen, ammonium, nitrate and solids sensors, which never-measured ASM1 states can be reconstructed, what limits the rest (sensor information, start-up memory, influent knowledge or kinetic mismatch), and when does a physics-informed neural network (PINN) add value over simulation or Kalman filtering given identical information?

## 2. Plant, data and cell notation

The plant has the BSM1 layout: two anoxic tanks of 1000 m3, three aerobic tanks of 1333 m3 (KLa 240, 240 and 84 per day), a ten-layer secondary settler, and constant flows Q_int 55 338, Q_r 18 446 and Q_w 385 m3/d. The influent comes from the project's own generator (1-day and 7-day periods), not from the official BSM1 influent files. Every estimator uses the audited parameter vault `asm1_cl-pinn/data/asm1.json` (SHA-256 `$vault_sha`): the ASM1 textbook set at 20 C plus a heterotroph ammonium switch K_NH,H. Samples are 15 min apart. Days 0 to 12 of the dry-weather scenario are the estimation window and days 12 to 14 are held out.

A cell is written K-I-A, for example K1-Ic-As, and `k100_ic_as` in file names.

**K: kinetics of the truth plant.**

- K0: the vault set (`results/raw`, unchanged since v1.0).
- K.25, K.5, K.75: log-interpolation from the vault set towards the BSM1 15 C set for muH, Ks, bH, KX, etah, muA, bA and ka, and linear interpolation for KNH_H and iXB (`results/raw_k025`, `results/raw_k050`, `results/raw_k075`).
- K1: the BSM1 15 C set (muH 4.0, Ks 10.0, bH 0.3, KX 0.1, etah 0.8, muA 0.5, bA 0.05, ka 0.05, KNH_H 0, iXB 0.08; every other value as in the vault), with the stoichiometric and composition matrices rebuilt together from the vault's symbolic expressions (`results/raw_k100`). BSM1 open-loop steady-state gate: $bsm1_gate_line
- M0': K0 kinetics, started from the state reached after 14 days of dry weather from the steady state (`results/raw_k000_off`). Model-based estimators only.
- R1 to R50: random truths with all 15 kinetic parameters perturbed lognormally (sigma_log 0.3, seed 0; `results/raw_rand/<i>`). Model-based estimators only.

For every truth other than K0, the constant-load data of curriculum stage 1 come from the nominal (vault) plant.

**I: influent knowledge given to every estimator.**

- Ie, exact: the 14-component influent every 15 min.
- Ic, composite: per calendar day, the flow-weighted total organic COD, TKN and NH4-N, recomposed with the BSM1 Table 5 fractions; Q_in at 15-min resolution. The PINN inputs, its physics residual and its balance loss all receive the same composite series.
- Ib, biased composite: Ic with S_S multiplied by 0.75 and X_S raised to keep the COD. Model-based estimators only.

**A: start-state knowledge**, stored in `results/v11/anchors/<tag>/<name>.npz` (z0_mean, z0_rel_std, nominal_ss, settler_init, meta).

- A0, oracle: the true reactor state at t = 0 with relative standard deviation 0.01.
- As: the steady state of the nominal plant under the Table 5 load. The relative standard deviation per state comes from a 200-member ensemble of the nominal plant with all 15 kinetic parameters perturbed lognormally (sigma_log 0.6, seed 0; etag and etah clipped at 1; members rejected when tank-5 X_B_A falls below 5 % of nominal or any value is non-finite). Rejection rate: $ensemble_rejection.
- Al1: As updated with a laboratory panel taken on the true state at t = 0: TSS in all five tanks (5 % relative error); total COD (5 %), filtered COD (5 %), TKN (7 %), filtered TKN (7 %) and alkalinity (5 %) in tanks 1 and 5. The simulated laboratory values use the truth plant's composition matrix and the update uses the vault's; random seed 20260923; linearised Gaussian update in log space, diagonal kept. Nine further laboratory seeds (20260924 to 20260932) give Al1 and Al2 anchors in the realistic cell (`results/v11/anchors/${realistic_tag}_lab/rXX`); the model-based estimators run on them and their spread is reported next to the seed-20260923 cell, never pooled with it.
- Al2: Al1 plus respirometric X_B_H (20 %) and X_B_A (25 %) in tanks 1 and 5. Model-based estimators only.

At K0 the As mean equals the A0 mean to within 1e-6 relative, so no K0-As PINN cell is run.

**Noise.** Multiplicative Gaussian noise clipped at zero, one realisation per sigma, fixed by the data files; sigma is 0.10 unless a cell says otherwise. Nine further realisations at sigma 0.10 exist for K0-Ie-A0 and K1-Ic-As. They are used by the model-based estimators only, and their spread is reported next to realisation 0, never pooled with it.

**Realistic cells (gate D3).** $gate_d3_line The realistic PINN cells are therefore at $realistic_label (`$realistic_tag` in file names).

## 3. Hypotheses

Every test uses window R0 (days 0 to 12), sigma 0.10, noise realisation 0, the primary metric of Section 8 and the rules of Section 9, unless it says otherwise.

**H1.** At the true kinetics the Kalman smoother is at least as accurate as the CL-PINN. Cells K0-Ie-A0 and K0-Ic-A0; the primary smoother of Section 5 (tuned q) against the three CL-PINN seeds. Supported in a cell if the CL-PINN outcome is a loss, refuted if it is a win, undecided on a tie.

**H2.** The information-matched nominal open loop falls behind anchor-aware persistence at a mismatch alpha* strictly between 0.25 and 0.75. Cells K0, K.25, K.5, K.75 and K1 with Ie-A0; d(alpha) = NRMSE(ode_openloop_reduced) minus NRMSE(persistence); alpha* is the linear interpolation of the first change of sign of d from negative to positive. Supported if 0.25 < alpha* < 0.75. The CPU probes in Section 12 already place this change of sign between 0.25 and 0.5 with probe code, so H2 checks that the registered pipeline reproduces them; it is not a blind prediction. The ordering of the CL-PINN and the smoother along the axis is not predicted; it is reported at alpha 0, 0.5 and 1.

**H3.** At K1 (K1-Ie-A0), joint state and parameter estimation lowers the error of the never-measured states: the augmented smoother beats the plain smoother, and PINN-theta wins against the CL-PINN (seed-paired). PINN-theta and the augmented smoother are expected to tie. State recovery is the primary measure. Parameter recovery, the absolute error of ln(multiplier) against ln(truth value / vault value) for each name in Section 7, is secondary and descriptive. PINN-theta at K0 (seed 0) is a sanity row that should keep every multiplier within [0.8, 1.25].

**H4.** At $realistic_label-Ic the laboratory panel helps both families: Al1 beats As for the CL-PINN (cell $realistic_label-Ic-Al1 against $realistic_label-Ic-As, seed-paired) and for the primary smoother. States with a longer memory time tau gain more: the Spearman correlation between tau (Section 7) and the relative error reduction from As to Al1 over the 11 never-measured components is positive for both families. The sign is the test; its size is reported.

**H5.** Composite influent knowledge costs more for forcing-slaved states than for biomass inventories. At K0-A0, for each family, the ratio NRMSE(Ic) / NRMSE(Ie), averaged over S_S, X_S, S_ND, X_ND and S_I, exceeds the same ratio averaged over X_B_H and X_B_A. For the CL-PINN the ratio is taken per seed pair and the median over the three pairs is used.

**H6.** The recoverability index predicts the per-state error of the smoother. Protocol of `scripts/recoverability_validation.py`, cells of `configs/analysis/recoverability_cells.yaml`:

- points: the 55 never-measured (tank, component) entries of one cell;
- index: CRB/range over days 0 to 12 (primary) and 1 - IG (secondary), computed by `scripts.recoverability.core_indices` for the cell's data directory, anchor prior and influent view;
- error: per-(tank, component) NRMSE over days 0 to 12, normalised by the component range pooled over tanks on the same window; the CL-PINN uses the median over its seeds;
- statistic: Spearman rho with a 95 % percentile interval from 1000 component-cluster bootstrap resamples (the 11 components drawn with replacement, each carrying its five tanks; seed 20260923);
- pass (EKS, primary cell ${realistic_tag}_ic_as only): rho >= 0.5 and the interval's lower end > 0. The CL-PINN result and every other cell are reported either way.

CRB/range is primary because it is in the same range-normalised units as the NRMSE. The v1.0 pilot (Section 12) is exploratory and did not set these choices.

**H7.** With the total-derivative residual, the CL-PINN beats the single-stage PINN at K0-Ie-A0 (seed-paired). If the outcome is not a win, the curriculum claim is withdrawn in full. The PINN arm of every other cell stays the CL-PINN whatever H7 shows.

## 4. PINN and LSTM protocol (fixed)

- Network: fully connected, 5 hidden layers of 128 tanh units; inputs t, Q_in and Z_in with Fourier features of periods 1 and 7 days (4 and 2 harmonics); softplus output head scaled per component from the start state (the anchor mean when an anchor file is given); 70 outputs (5 tanks by 14 components).
- Physics residual: dZ/dt minus the ASM1 reactor-train right-hand side at 512 random collocation times per step, with the total time derivative. The influent inputs follow their local linear interpolant, so the residual constrains dZ/dt along the trajectory. v1.0 used the partial derivative (derivative audit, Section 12). Forward-mode JVP.
- Recycle: particulates from tank 5 scaled by the measured return-sludge TSS over the predicted tank-5 TSS; solubles copied from tank 5 (the settler's soluble lag, about 4.2 h, is not modelled; see O1 in Section 13). The return-sludge TSS first passes a trailing 4-sample (1 h) moving average; the model-based estimators and the open-loop rows use the same filtered signal.
- Loss terms: data (7 target channels), physics, initial condition, positivity, integral COD and N balance. Final weights: data 1, physics 1, initial condition 10, positivity 1, balance 0.1. Opening weights of the curriculum: data 10, physics 0.05, initial condition 10, positivity 1, balance 0.
- CL-PINN curriculum (a training protocol, not a claim): four stages with 15 %, 20 %, 25 % and 40 % of the steps; stage 1 on the constant-load scenario, stages 2 to 4 on the dry scenario; horizons 1, 3, 7 and 12 days; moving-average smoothing of the noisy observations over 9, 5, 3 and 1 samples (none at sigma 0); weights moved linearly between stage boundaries with a cosine ramp inside each stage. With an anchor file, stage 1 starts from the nominal steady state. Single-stage PINN: final weights, 12-day horizon, dry data, no smoothing, same step count.
- Initial-condition term: squared scaled error against the anchor mean at t = 0, weighted per state by the normalised inverse log-variance of the anchor; for A0 the weights are uniform, which reproduces v1.0.
- Optimiser: Adam, learning rate 1e-3, cosine decay to 2 % of it over 20 000 steps, gradient-norm clipping at 1.0; float32 on the RTX 4050.
- Target channels: S_O in tanks 3, 4 and 5; S_NH in tank 5; S_NO in tanks 2 and 5; TSS in tank 5. TSS_ras is an input only. The E3 variants drop $sensor_drop or add $sensor_add.
- PINN-theta: the CL-PINN plus trainable multipliers on $kinetic_subset, each exp(ln 4 tanh(raw)) in [0.25, 4] with raw values starting at 0; prior term mean((ln m / 0.693)^2) with weight 1e-3; one Adam optimiser over network and multipliers with the same schedule and clipping.
- LSTM (model-free floor): 2 layers of 128 units, the same features, optimiser and step count, physics and balance weights 0, no curriculum.
- Seeds 0, 1 and 2 (0 and 1, or 0 only, where Section 6 says so). The seed sets the torch and NumPy generators, which fixes the network initialisation and the collocation draws. The observation noise realisation is fixed per sigma by the data file.
- Days 12 to 14 are never used in training.

## 5. Model-based estimators (fixed)

All run on the CPU in float64 through `scripts/run_observers.py`, one YAML file per cell under `configs/observers/grid/`, validated by `src/observers/cell_inputs.py`.

- Model: the reduced reactor-train equations of the PINN residual (vault kinetics, the recycle reconstruction of Section 4, no settler), in log coordinates x = ln z (70 states), so every estimate stays positive.
- Prediction: exponential Rosenbrock-Euler with 3 substeps per 15-min sample; discrete process noise by the Van Loan method.
- Measurements: h(x) = ln(H_j z) for each measured channel (the seven targets of Section 4 unless a cell names another set); covariance update in Joseph form.
- Measurement noise R: per channel, (1.4826 MAD(diff ln y))^2 / 2 over days 0 to 12, floored at 0.01^2.
- Process noise: Q_c = diag(q_s^2 for the soluble components, q_p^2 for the particulate components) per day in log units, plus B_ras var_ras B_ras^T for the return-sludge input, where var_ras is the estimated RAS noise variance divided by the filter window (4), in log units.
- Tuned q (primary row): q_s and q_p each in {0.003, 0.01, 0.03, 0.1, 0.3, 1, 3} per square-root day (49 pairs), chosen by the innovation log-likelihood of the measured channels over days 0 to 12 only. Hidden states are never used for tuning. Gate D4 fixed this rule and grid from measured-channel evidence alone: on the first five-point grid the innovation optimum sat on the edge, while on seven points it is interior at K0-Ie-A0 with a normalised innovation of 1.04 per channel; the 6-h predictive score of the measured channels had no interior optimum and a normalised innovation of 0.52 per channel (Section 12).
- Frozen q (secondary row, model name suffix `_frozenq`): the pair chosen at K0-Ie-A0, sigma 0.10, applied to every cell.
- Initial state: mean ln(z0_mean) and covariance diag(ln(1 + z0_rel_std^2)) from the cell's anchor file.
- Estimators: EKF (filtered estimate); EKS (Rauch-Tung-Striebel smoother over the EKF pass); IEKS (relinearised about the smoothed trajectory until the relative change falls below 1e-4, at most 5 iterations; on the claim cells of Section 6.2, and on every main-grid cell if it is the primary smoother); augmented EKF and EKS (the four log-multipliers of Section 7 appended with prior mean 0, prior standard deviation 0.693 and a random walk of 1e-3 per square-root day); online EKF (also assimilates days 12 to 14; labelled "more information").
- Held-out days 12 to 14: an open-loop forecast of the reduced model from the day-12 estimate, for every estimator except the online EKF.
- Divergence: a run has diverged if any estimate is non-finite, the filter flags itself as diverged, or the time mean of the normalised innovation squared over days 0 to 12 exceeds 3 times the number of measured channels. The primary smoother is the EKS unless more than 20 % of the 60 main-grid cells diverged with the EKF in the numerics pass; in that case it is the IEKS, provided the IEKS stays at or below 20 %. $numerics_line Primary smoother: $primary_smoother.
- Ideal-settler row: no RAS sensor. Inside the reduced model the return-sludge TSS is the model's own tank-5 TSS times (Q_in + Q_r) / (Q_r + Q_w), a settler that loses no solids to the effluent (`ras_mode = "ideal_settler"` of `src/observers/reduced_model.py`); this is the same closure as the ideal-settler variant of Section 7.

Reference rows (`scripts/make_baselines.py`, same anchor and influent view as the cell):

- persistence: holds the anchor mean;
- ode_openloop_reduced: the reduced model integrated from the anchor mean with the same filtered RAS input and influent view (information-matched);
- ode_openloop_full: the full plant with its settler, from the anchor's settler_init (the true initial state for A0). It sees more structure than any estimator and is labelled "more information".

## 6. Grid

### 6.1 GPU cells

Every run is `scripts/run_all.py --profile full --seed <s> --resume` with the named config under `configs/regime/`, launched by `scripts/run_core.py`.

$gpu_table

Planned core total: $gpu_total run-equivalents. The single-stage PINN runs only at K0-Ie-A0 (H7). The LSTM rows at K0 come from v1.0, which had no physics term; the LSTM at $realistic_label-Ic-As is new.

### 6.2 Model-based estimator grid

$observer_table

Planned estimator runs: $observer_total. Claim cells (IEKS added): $claim_cells.

Families:
- main: K0 to K1 by Ie, Ic and Ib by A0, As, Al1 and Al2 at sigma 0.10, with sigma 0, 0.05 and 0.15 in addition at K0-Ie-A0, K1-Ie-A0, K1-Ic-As and K1-Ic-Al1, and at K.5-Ic-As and K.5-Ic-Al1 if gate D3 moved the realistic cells;
- realisations: 1 to 9 at K0-Ie-A0 and K1-Ic-As, EKF and EKS;
- m0prime: M0' by A0, As and Al1;
- random: R1 to R50 with Ie-A0, EKS and augmented EKS, plus the three reference rows;
- sigma_log: As and Al1 at K1-Ic with anchor ensembles of sigma_log 0.4 and 0.8 (`results/v11/anchors/k100_slog04`, `k100_slog08`);
- lab: Al1 and Al2 at $realistic_label-Ic with the nine further laboratory seeds of Section 2, EKF and EKS, tuned q;
- settler: the ideal settler at K1-Ie-A0;
- sensors: at K1-Ie-A0, each of the seven targets dropped, all DO probes dropped, all nitrogen analysers dropped, and each candidate channel added.

## 7. Recoverability analysis

Computed on the CPU by `scripts/recoverability.py` along each truth trajectory of days 0 to 12 (dry scenario, sigma 0.10) with the reduced model and tangent-linear propagation, before any learned error of v1.1 was read (`results/v11/analysis/`). The nominal model's Jacobians are evaluated along the plant's own trajectory, and the log-state drift on the Jacobian diagonal is taken from that trajectory (its spline slope), because the trajectory does not solve the nominal model; with the model's own drift a mass-neutral X_I/X_P shift appeared observable (Section 12).

- IG = 1 - sigma_post / sigma_prior per (tank, component), from the Fisher information of the start state with the As prior. tau is the 1/e decay time of the start-state self-sensitivity. The forcing share is the share of the day 0 to 12 trajectory sensitivity due to plus or minus 10 % on each influent component, against the start state.
- Classes: sensor-recoverable if IG >= 0.5; anchor-carried if tau > 6 d and IG < 0.5; forcing-slaved if the forcing share > 0.5 and tau < 1 d; partly recoverable otherwise. The thresholds are kept as first written. With the return-sludge TSS as a measured input no never-measured state exceeds tau = 6 d (longest 5.97 d, X_B_A at K0), so the anchor-carried class is empty for the estimators' model; with the ideal settler it holds 15 entries at K0 and 10 at K1. This is reported as a finding.
- Kinetic subset for PINN-theta and the augmented filters: $kinetic_subset (D-optimal 4 of the 15 kinetic parameters on the nominal trajectory, collinearity index below 20). Prior standard deviation 0.693 on each log-multiplier; PINN prior weight 1e-3; bound [0.25, 4].
- Sensor confirmation (E3): drop $sensor_drop (largest loss of never-measured information at K1-Ie-A0) and add $sensor_add (largest gain).

## 8. Metrics and windows

- Primary: Track B NRMSE, the mean over the 11 never-measured components (S_I, S_S, X_I, X_S, X_B_H, X_B_A, X_P, S_ND, X_ND, S_ALK, S_N2) of the RMSE pooled over the five tanks divided by the component's range over days 0 to 12 in the same data directory (`track_b_nrmse_fixed` in `src/eval/report.collect_runs`; on R0 it equals `track_b_nrmse`).
- Skill against anchor-aware persistence of the same cell: S = 1 - E / E_persistence.
- Gap closed: (E_persistence - E) / (E_persistence - E_ode_openloop_reduced).
- Level error of X_B_H, X_B_A, X_I and X_P: RMSE over the mean absolute truth, pooled over tanks.
- Fraction of time the volume-weighted tank means of X_B_H and X_B_A stay within 10 % of the truth.
- Per-(tank, component) NRMSE with the same fixed range (H6).
- R2 is reported in the Supplementary Information only.
- Windows: R0, days 0 to 12 (primary; the smoothed estimate for smoothers, the filtered estimate for filters, the network output for the PINN); R2, days 2 to 12; F, days 12 to 14 (forecast without data, except the online EKF).
- `scripts/regime_map.py` and `scripts/recoverability_validation.py` implement these definitions; they may be written after this registration but may not change them.

## 9. Decision rules

- PINN model against a single-valued estimator or reference row, same cell, window and noise realisation: win if the Track B NRMSE of every seed is strictly below the comparator's, loss if every seed is strictly above, tie otherwise.
- Two PINN models (H3, H4, H7): seeds paired by number; win if every pair favours the first model, loss if every pair favours the second, tie otherwise.
- Two single-valued estimators: the strictly lower NRMSE wins; equal values tie. The spread over realisations 1 to 9 is reported next to the realisation-0 comparison and does not change the outcome.
- Rows labelled "more information" (ode_openloop_full, the online EKF) are shown for reference and never win.
- A diverged or failed run loses to every comparator in its cell and window. A cell whose comparator failed is reported as not decided.
- Two-seed rows ($realistic_label-Ic-As at sigma 0.05 and 0.15, the E3 variants) and the one-seed row (PINN-theta at K0) are labelled as such and are not used to decide H1 to H7.

## 10. Interpretation fixed in advance

| Outcome | Message of the paper |
| --- | --- |
| The smoother beats the PINN in every cell | A pre-registered map of when not to use a PINN on a routine sensor set. The recoverability map (Section 7), the laboratory and influent guidance from the smoother and the Fisher analysis, and the deployment cost (EKF plus smoother in minutes on a laptop, PINN about 9 GPU-minutes per window) carry the paper. |
| The PINN wins only in some mismatch cells | A limited, regime-specific role, stated with the cells where it holds. |
| Every estimator falls to persistence at K1 | That end of the axis is reported as it is; the crossover lies between K.25 and K.75. |
| H6 fails | Sensor information is not the bottleneck; model or optimisation error is. |
| H7 fails | The curriculum sentence is removed; training details move to the Supplementary Information. |

## 11. Commitments

- Every cell of Section 6 is reported, including failed and diverged runs.
- No PINN, LSTM or model-based estimator setting changes after any full-profile Track B result of v1.1 has been seen. `scripts/run_core.py` and `scripts/observer_grid.py` refuse to start unless the frozen paths (`PREREGISTRATION.md`, `configs/`, `src/`, `scripts/run_all.py`, `scripts/run_observers.py`, `scripts/make_baselines.py`) match the tag; an override is written to `results/v11/deviations.log`.
- Failed runs are reported and not rerun. Only infrastructure crashes (CUDA or cuDNN errors, out-of-memory, interrupted processes, a full disk) are rerun, with the identical config, and the crashed directory is kept.
- Every GPU attempt is written to `results/v11/gpu_ledger.csv`.
- The hypotheses are tested as written; any further analysis is labelled exploratory.

## 12. Results and checks that pre-date this registration

Hidden-state errors known before registration:

- All v1.0 runs: 36 PINN checkpoints (cl_pinn 12, pinn 12, single-axis ablations 8, flow-only 2, measured-only start state 2), 8 LSTM runs and 8 analytic reference directories, all trained with the partial-derivative residual. Their re-scoring on R0, R2 and F (`results/v11/reanalysis_v1.json`) is exploratory.
- CPU probes made while planning v1.1 with probe code, not the registered pipeline: the information-matched open loop from the true start state at K0 has held-out Track B NRMSE 0.084 with clean RAS and 0.130 with sigma 0.10 RAS; the graded open loop against persistence gave 0.228 against 0.350 at alpha 0.25, 0.391 against 0.334 at alpha 0.5, and 0.633 against 0.314 at alpha 1; the BSM1 15 C steady-state probe; the slow modes of the full plant (11.5, 7.2, 5.3 and 3.2 d); the EKF cost probe (warmed 70 by 70 Jacobian about 3.8 ms, matrix exponential about 1.1 ms, Van Loan step about 4.2 ms).
- The derivative audit of the v1.0 checkpoints (`results/v11/derivative_audit.json`).
- Gate D3 (Section 2), which read one smoother score and one persistence score at K1-Ic-As.
- The checks of the strategic synthesis of 23 September 2026: re-evaluation of the vault's symbolic stoichiometry, continuity 4.7e-16 with the rebuilt composition, 79 passing tests.
- Reference rows of G3 (`results/v11/baselines`), Track B NRMSE train / held-out. At K0-Ie-A0: persistence 0.2527 / 0.3716; information-matched open loop 0.0553 / 0.0841 (sigma 0, RAS window 1) and 0.0823 / 0.1412 (sigma 0.10, window 4); full plant 0.0006 / 0.0017. At K1-Ic-As, sigma 0.10: persistence 0.5344 / 0.7184, information-matched open loop 0.6743 / 1.0017, full plant 0.4694 / 0.6818.
- The G3 reference cell K0-Ie-A0, sigma 0.10, tuned q (`results/v11/observers/k000_ie_a0`): EKS 0.2877 / 0.3343, IEKS 0.3740 / 0.4566 (5 passes, not converged), online EKF held-out 0.4285. With the true underflow solubles the reduced model's open-loop error falls to 7.9e-4, so the RAS soluble copy carries almost all of its error. A G3 smoke run with q = (0.1, 0.03) gave EKF and EKS 0.110 / 0.130, and augmented multipliers muH 0.70 and bH 1.19 on data made with the nominal kinetics (the multipliers absorb the closure error).
- The planning q-scan of the G3 scratch copy: the innovation rule chose (0.3, 0.3) with EKS 0.288, the 6-h predictive rule chose (0.3, 0.003) with EKS 0.081, and the information-matched open loop gave 0.082. These hidden-state values were known before gate D4; D4 was decided on the measured-channel scan in `results/v11/predating/q_scan_extended.json`, which computed no hidden-state error.
- Anchor quality against the true state at t = 0 (`anchors_report.json` in each anchor directory; RMS log error of z0_mean): As 0.000, 0.110, 0.207, 0.296 and 0.378 at K0, K.25, K.5, K.75 and K1; Al1 0.244, 0.257, 0.292, 0.344 and 0.405. With the registered seed the laboratory panel does not lower the aggregate error at K1; the mean over 20 laboratory seeds does (0.367 against 0.378), and a full ensemble covariance does worse (`results/v11/predating/lab_panel_seed_spread_output.txt`). The method and the seed were kept.
- The O1 check (`results/v11/predating/o1_precheck_output.txt`; Section 13).
- The v1.0 pilot of the H6 validation (`results/v11/analysis/validation`, exploratory, partial-derivative residual): CL-PINN against CRB/range, rho 0.90 [0.65, 0.96] with A0 and 0.67 [0.16, 0.87] with As.
- A structural check of G4 (a mass-neutral X_I/X_P shift must stay invisible to the sensors) failed at K1 and led to the trajectory drift of Section 7 (`results/v11/predating/drift_check.py`); no hidden-state error was involved.

Computed before registration without reading hidden-state errors:

- the observer numerics pass (filter diagnostics only);
- the smoke runs (4000 steps; code paths, summary keys and poisoning tests only): $smoke_line;
- the concurrency and determinism check: $concurrency_line;
- the recoverability indices, the kinetic subset and the sensor confirmation (Section 7).

GPU ledger at registration: $ledger_line.

$predating_table

## 13. Budget

GPU: the core is 39.05 run-equivalents including the smoke runs (G1 0.2, G6 1.8); a contingency of 6 is reserved for infrastructure failures; the hard cap is 50. One 20 000-step PINN run counts 1, an LSTM run 0.35 and a 4000-step quick run 0.2. Optional items O1 to O12 start only after the core, in the order O1, O3, O5, O2 (gate D6), and only within the hard cap. The O1 CPU check was run before this registration (Section 12): modelling the settler's soluble lag cuts the information-matched open-loop error by 98 % at K0 without noise, by 49.6 % (held-out) at K0 with sigma 0.10, and by 1 % at K1-Ic-As. O1 therefore fails its "more than 50 %" rule at the protocol setting and stays out of the core; its K1 GPU runs are dropped, and a K0-only CPU diagnostic (EKF with the six-layer soluble lag) is a secondary, exploratory analysis.

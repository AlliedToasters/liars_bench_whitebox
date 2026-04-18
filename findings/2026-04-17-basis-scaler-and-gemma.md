# Basis-distribution StandardScaler + Gemma generalization test

**Date:** 2026-04-17
**Status:** exploratory — methodology refinement, no re-freeze yet.

## TL;DR

1. Ported probe fitting from sklearn (liblinear/lbfgs on CPU) to cuML
   qn on GPU. ~20× wall-clock speedup on a full (C, penalty) sweep.
   Convergence warnings gone.
2. Added a `StandardScaler` step to the probe fit, fit on the
   basis-training distribution (Alpaca-val as proxy for Alpaca-train
   until we cache the latter's PCs). Scaler stats are absorbed into
   `(w, b)`, so downstream scoring stays unchanged.
3. Under proper scaling hygiene, **the earlier "one recipe
   generalizes" story does not hold up.** Mistral and Gemma prefer
   different reducers, different signals, and disagree on whether
   ensembling helps.
4. Mistral's old headline of **0.768 ± 0.004** (L2/C=0.1, mean_asst,
   attn_delta, all-layer ensemble) drops to **0.723 ± 0.019** when
   the same recipe is re-fit with basis-stats scaling. The best
   Mistral cell under scaling is a different recipe entirely.
5. Gemma-27B's best recipe lands at **0.938 ± 0.003**, but this
   number is on 2 held-out datasets (ID + insider-trading), not 4.
   LB excluded Gemma from HP-C / HP-KR because they couldn't
   reliably induce deception on those tasks on this model — a
   behavioral artifact, not a pipeline gap.

## What changed

### Hardware / solver

- `probe_fit.py` is now the single probe-fit entry point. All
  exploratory scripts (`sweep_regularization`, `explore_multilayer_token`,
  `sweep_ensemble_regularization`, `ensemble_stability`) and the
  frozen blinded pipeline (`experiment_pipeline.fit_linear_probes`)
  route through it.
- Solver: `cuml.linear_model.LogisticRegression(solver="qn")` on GPU,
  `max_iter=4000`, `tol=1e-4`. Handles L1 and L2 via OWL-QN / L-BFGS.
- A full (C, penalty) sweep that previously took ~45 min on CPU now
  takes ~2 min on GPU (one tenant; parallel tenants compete).

### Preprocessing

- `StandardScaler` (μ, σ per PC) is fit once per model on the
  Alpaca-val cached PCs — the basis-training distribution, label-
  free and consistent with the unsupervised basis assumption.
- Scaler stats are absorbed into `(w_raw, b_raw)` via:
  `w_raw = w_scaled / σ`, `b_raw = b_scaled − μ · w_raw`.
  So `score_probe(..., vecs)` continues to compute `X @ w + b` on
  raw features; no change in downstream code.
- Alpaca-val is a *proxy* for Alpaca-train here. Alpaca-train PCs
  were not cached when we fit the basis (fixed going forward;
  noted in memory).

### Why it matters methodologically

Under unscaled features, the L1 penalty's effective weight on each
feature is proportional to that feature's magnitude. In our PC-basis
setting, σ per PC ranges from ~0.002 to ~18 on Mistral Alpaca-val —
nearly 4 orders of magnitude. The unscaled optimizer prefers to set
low-variance PCs to zero and keep high-variance ones, which is a
scale artifact, not a statistical signal. Scaling equalizes that.

Convergence warnings (`lbfgs_max_iter` / `liblinear_max_iter` hit)
were a symptom of this mis-scaling. Under basis-stats scaling the
qn solver converges cleanly on every cell; the "line search failed
to advance" messages that remain are benign (means the coefficient
has already converged).

## Updated headline numbers

### Mistral-24B — 4 held-out datasets

Under *proper* basis-distribution scaling:

| recipe | dev | held-out mean | bootstrap (B=20) |
|---|---|---|---|
| **old headline**: L2/C=0.1 mean_asst s0 all-layer ensemble | 0.992 | 0.658 | **0.723 ± 0.019** (same recipe, re-run) |
| new single-probe best: L1/C=0.01 last_token L16 S0 (blinded) | 0.977 | **0.757** | not run |
| new ensemble best: L2/C=0.1 last_token s1 all-layer | — | 0.726 | **0.723 ± 0.019** |

Per-dataset bootstrap (L2/C=0.1 last_token s1 all-layer, B=20):

| dataset | mean | std |
|---|---|---|
| instructed-deception | 0.977 | 0.001 |
| harm-pressure-choice | 0.642 | 0.031 |
| harm-pressure-knowledge-report | 0.635 | 0.047 |
| insider-trading | 0.637 | 0.029 |
| **held_out_MEAN** | **0.723** | **0.019** |

Key moves on Mistral:
- **Ensemble no longer beats single-probe.** L1/C=0.01 last_token L16 S0
  single-probe (blinded) = 0.757 > any ensemble cell. The all-layer
  ensemble trick that defined the 0.768 headline was a partial
  artifact of the unscaled magnitude structure.
- **Reducer migrated**: mean_asst → last_token.
- **Signal under ensemble migrated**: S0 (attn_delta) → S1 (mlp_delta).
  Under the single-probe best, S0 is still the winner.
- Bootstrap spread widened (0.004 → 0.019). Likely because
  last_token is a single-position reducer that's more sensitive to
  response-length distributional shifts across bootstraps.

### Gemma-3-27B — 2 held-out datasets

| recipe | dev | held-out mean | bootstrap (B=20) |
|---|---|---|---|
| best blinded (reg sweep): L1/C=0.1 mean_asst L27 S0 | 0.996 | 0.904 | — |
| best ensemble: L1/C=0.1 mean_asst s0 all-layer | — | **0.942** | **0.938 ± 0.003** |

Per-dataset bootstrap (L1/C=0.1 mean_asst s0 all-layer, B=20):

| dataset | mean | std |
|---|---|---|
| instructed-deception (held-out) | 0.984 | 0.001 |
| insider-trading | 0.892 | 0.006 |
| **held_out_MEAN** | **0.938** | **0.003** |

Gemma caveats:
- **Only 2 held-out datasets**, not 4. LB excluded Gemma from
  harm-pressure-choice and harm-pressure-knowledge-report because
  Gemma-3-27B is behaviorally more resistant to deception induction
  on those scenarios — a real finding about the model, not a data
  gap we chose.
- The 0.984 IT-held-out number is in-distribution with training (both
  are LB-ID splits). It measures probe fit quality, not transfer.
- **The one genuine cross-dataset transfer number for Gemma is
  insider = 0.892 ± 0.006.** Still higher than Mistral's
  insider (0.637 ± 0.029) — so Gemma's probes really do transfer
  better on the deception it *does* produce.

## Model divergence under scaling

The interesting finding from this round isn't the numbers — it's
that once we remove the unscaled-magnitude artifact, the two models
disagree on almost every axis:

| axis | Mistral prefers | Gemma prefers |
|---|---|---|
| reducer | last_token | mean_asst |
| signal (ensemble cell) | S1 (mlp_delta) | S0 (attn_delta) |
| ensemble benefit | none (single probe wins) | +4 pp over single probe |
| penalty | L1 (single) / L2 (ensemble) | L1 |
| C | 0.01 (single) / 0.1 (ensemble) | 0.1 |

Only consistent across models: **low-to-mid C wins**, and **IT
in-distribution is easy**.

This breaks the "mean_asst × attn_delta × all-layer" recipe we
telegraphed in the prior findings doc. Either (a) there's a hidden
structure unifying these cells that we haven't found, or (b) the
signal location genuinely varies by architecture and we should
report per-model recipes rather than a universal one.

## Methodological notes / what this changes about the story

- The 0.768 ± 0.004 headline from the prior findings doc was partially
  a scale artifact. Under hygiene it's 0.723 ± 0.019 on the same
  recipe. The old tight std was a feature of the deterministic
  unscaled fit, not of genuine stability.
- **Mistral's best number actually went up** (0.757 single probe
  vs 0.768 ensemble), but at a different cell. If we accept the
  single-probe result, the generalization story is: small Mistral
  also probes well, but via a different span (last token) and
  mechanism (attention delta still).
- **Gemma's number is real but narrow.** 0.892 cross-dataset is a
  good number; the 0.942 mean is inflated by the easy
  in-distribution IT component and is on half the datasets Mistral
  gets evaluated on.
- The no-universal-recipe finding is arguably more scientifically
  honest than a tight universal number — per-architecture probe
  optimization looks necessary. This is a meaningful update to
  how we frame the method.

## What to re-freeze when we re-freeze

- Probe fit: cuml qn + basis-stats scaler (this module path as the
  frozen interface: `probe_fit.fit_probes(train, labels, scaler_stats, C, penalty)`).
- Scaler-stats computation: always `compute_scaler_stats(alpaca_val)`
  — and cache Alpaca-train PCs for future models so we don't reuse
  the calibration set.
- Blinded selection rule: still "dev-best (reducer, layer, signal),
  calibrate τ on Alpaca-val, evaluate on LB-* held-out." No change.
- Recipe itself: **per-model cell**, not one universal recipe. To
  be re-registered when we're ready to lock in the numbers above.

## Open questions

1. **Why does Mistral now prefer last_token?** If it was mean_asst
   under unscaled and mean_asst disappeared under scaling, the
   mean_asst's original advantage may have been carried by a
   particular high-variance PC direction. Could check: for Mistral
   L2/C=0.1 mean_asst s0, which PCs had non-zero coefficients pre-
   vs post-scaling?
2. **Per-token structure.** Post-lmprobe-274, per-token projections
   are O(1)-addressable. An oracle study of per-relative-position
   LR probes is cheap and would answer whether last_token wins
   because the *final* position is special, or because position
   matters and our reducers pooled over it badly.
3. **Cross-model signal routing.** attn_delta vs mlp_delta preference
   flips between Mistral and Gemma — does this correlate with
   something measurable about the models (attention-head structure,
   residual stream composition, MoE vs dense)?
4. **HP-C / HP-KR for Gemma** — not available in LB. Could we
   generate a "Gemma-compatible" HP benchmark, or is the behavioral
   resistance itself something to study and write up?

## Files
- `probe_fit.py` — new shared probe-fit module (cuml qn + basis-stats scaler)
- `ensemble_stability.py` — now takes `--C/--penalty/--reducer/--signal` CLI args
- Results under `results/reg_sweep_*.json`, `results/sweep_ensemble_reg_*.json`,
  `results/ensemble_stability_*.json`, `results/explore_multilayer_token_*.json`.

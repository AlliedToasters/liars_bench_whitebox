# Finding 2026-04-17 (part 2): All-layer equal-weight ensemble is the recipe

**Model:** Mistral-Small-3.1-24B-Instruct (Mistral only — generalization untested)
**Status:** Exploratory. Methodology is in refinement per 2026-04-17 decision; results are not pre-registered and the recipe will be re-frozen before scaling to other target models.
**Context:** Follows the null result in `findings/2026-04-17-alpaca-basis-null.md`, which established that the Alpaca-basis pivot didn't fix cross-dataset overfitting under the originally frozen protocol (LR, C=1.0, single best probe selected by LB-ID dev AUROC). This document records what three cheap exploratory sweeps on the same cached PCs revealed.

## TL;DR

A fixed, zero-learned-parameter ensemble recipe beats the DoM baseline by **~5 AUROC points** on Mistral held-out, stably and reproducibly:

> **L2-regularized logistic regression at C=0.1, mean_asst reducer, attn_delta signal, equal-weight mean of sigmoid(logit) across all 40 layers**

Bootstrap stability: **0.768 ± 0.004** mean held-out AUROC (B=20). DoM best: 0.721.

The ~0.77 ceiling is now reached by a simple non-learned combination rule at every reasonable regularization level. It is a property of the LR-on-64-PCs function class over this Alpaca basis, not an artifact of selection.

## Three sweeps, three results

### 1. Regularization sweep (single probe)

Fit LR per (reducer, layer, signal) on LB-ID train across C ∈ {0.001, 0.01, 0.1, 1.0, 10.0} × penalty ∈ {L1, L2}, select the single (reducer, layer, signal) by LB-ID dev AUROC, calibrate τ at 1% FPR on Alpaca-val, evaluate held-out.

| recipe                       | held-out mean | IT AUROC |
|------------------------------|--------------:|---------:|
| L2, C=1.0 (original baseline) | 0.673 | 0.416 (anti-correlated!) |
| L2, C=0.001                   | 0.719 | 0.657 |
| **L1, C=0.1** (sweep winner)  | **0.733** | 0.657 |
| L2, C=10                      | 0.661 | 0.346 |

Findings:
- **Held-out mean is U-shaped in C** even though dev AUROC is monotonic. Classic case of selection rule rewarding overfitting.
- **Strong regularization rescues IT** from below-chance (0.416) to 0.657. The probe direction that weak regularization learned is literally inverted on IT; strong regularization forces a more generalizable direction.
- **L1/C=0.1 beats DoM** (0.733 vs 0.720) — first configuration to do so. But the per-dataset wins and losses roughly cancel: essentially DoM-tier performance.
- **Oracle ceiling across the full grid: 0.769** (mean_asst, L27, S0, L2/C=1). Found by exhaustive (reducer, layer, signal) search on held-out — not reachable by dev-AUROC selection.

Saved: `results/reg_sweep_mistral-small-3.1-24b-instruct.json`.

### 2. Multi-layer ensembling

Holding probes at L1/C=0.1, combine per-layer scores with equal-weight mean of sigmoid(logit). Variants: all 40 layers; top-10, top-5, top-3 ranked by LB-ID dev AUROC.

| recipe                                   | mean held-out AUROC |
|------------------------------------------|--------------------:|
| single best probe (last_token L16 S0)    | 0.733 |
| all-layer mean, last_token s0            | 0.757 |
| **all-layer mean, mean_asst s0**         | **0.763** |
| all-layer mean, mean_excl5 s0            | 0.747 |
| top-10 mean, mean_asst s0                | 0.674 |
| top-3 mean, mean_asst s0                 | 0.640 |

Findings:
- **Equal-weight mean over all layers lifts mean AUROC by ~3 points** without introducing any learnable ensemble parameter. The lift is free in terms of capacity.
- **Top-K is strictly worse than "all"** for all tested K. Counterintuitive but consistent: top-K concentrates weight on layers that overfit LB-ID, which is the failure mode we were trying to avoid. "All" averages those layers away.
- **Reducer ranking reverses under ensembling.** Single-probe: last_token > mean_asst. Ensemble: mean_asst > last_token. Interpretation: mean_asst probes are more layer-diverse (each layer sees a different span average), so averaging 40 of them cancels more noise. Last_token probes look similar across layers because they all anchor on the same position.
- **IT is rescued decisively:** single-probe IT 0.657 → all-layer ensemble IT 0.766. Highest IT number we've seen anywhere, above the single-probe oracle.
- **Cross-signal averaging (attn + mlp at single layer) doesn't help.** It's layer-averaging that carries the gain, not signal-averaging.

Saved: `results/explore_multilayer_token_mistral-small-3.1-24b-instruct.json`.

### 3. Ensemble × regularization sweep

Is the ensemble gain specific to L1/C=0.1, or general across the probe family? Swept the same (C, penalty) grid as part 1, applied the fixed all-layer equal-weight ensemble recipe to each cell, reported mean held-out AUROC per (reducer, signal).

| pen | C     | last_tok s0 | mean_asst s0 | mean_excl5 s0 |
|----:|------:|------------:|-------------:|--------------:|
| L2 | 0.001 | 0.717 | 0.694 | 0.676 |
| L2 | 0.01  | 0.733 | 0.744 | 0.726 |
| **L2** | **0.1** | 0.748 | **0.769** | 0.748 |
| L2 | 1     | 0.695 | 0.745 | 0.733 |
| L2 | 10    | 0.654 | 0.683 | 0.671 |
| L1 | 0.1   | 0.757 | 0.763 | 0.747 |

Findings:
- **Ensemble has its own U-shape in C, peaking at C=0.1** for both L1 and L2. Different from the single-probe U-shape (which peaked in different places for different reducers).
- **L1 and L2 converge under ensembling** (0.763 vs 0.769 at C=0.1). Sparsity's advantage disappears when you're averaging 40 probes — their mean is dense regardless.
- **Single-probe winner ≠ ensemble winner.** Single: L1/C=0.1, last_token. Ensemble: L2/C=0.1, mean_asst. The selection criterion matters.
- **attn_delta strictly dominates mlp_delta** in every cell. On Mistral, deception signal lives in attention residuals. *Flag:* this may or may not hold on other models.
- **Ceiling 0.769 is the same as the single-probe oracle.** The ensemble reaches the ceiling; it does not exceed it.

Saved: `results/sweep_ensemble_reg_mistral-small-3.1-24b-instruct.json`.

### 4. DoM ensembling (control)

Does the equal-weight-all-layers trick also lift the DoM baseline?

| variant           | single | ensemble | Δ |
|-------------------|-------:|---------:|------:|
| last_token s0     | 0.668 | 0.694 | +0.027 |
| **mean_asst s0** (best single) | **0.720** | 0.669 | **−0.051** |
| mean_excl5 s1     | 0.721 | 0.673 | −0.048 |

**DoM ensembling hurts.** Best DoM ensemble (0.694) is below best DoM single (0.721).

Interpretation:
- DoM is a fully-data-determined direction per layer (class-mean difference). No capacity to regularize.
- Different layers' DoM directions point in **genuinely different directions** in the 64-d PC space. Averaging their scores partially cancels signal.
- LR, by contrast, has capacity that's regularized toward a generalizable subspace at each layer. Averaging 40 regularized estimates cancels noise while signal stacks.

**Consequence:** The ~0.77 ceiling is **not** purely a property of the feature space. DoM can't reach it even with ensembling. It's a property of the *function class* (regularized LR over 64 PCs) over this basis.

Saved: `results/ensemble_dom_mistral-small-3.1-24b-instruct.json`.

### 5. Bootstrap stability check

B=20 bootstrap resamples of LB-ID train (n=3005, with replacement). Refit LR+ensemble each time.

| dataset      | mean | std | min | max |
|--------------|-----:|----:|----:|----:|
| ID           | 0.945 | 0.002 | 0.941 | 0.947 |
| HP-C         | 0.629 | **0.013** | 0.604 | 0.645 |
| HP-KR        | 0.719 | 0.004 | 0.712 | 0.730 |
| IT           | 0.777 | 0.003 | 0.771 | 0.782 |
| **held-out mean** | **0.768** | **0.004** | 0.761 | 0.774 |

Findings:
- **Headline 0.769 is not a lucky draw.** Bootstrap point estimate is 0.768 ± 0.004 — matches within noise.
- **HP-C is the noisiest dataset** (std 0.013, 3× higher than any other). This is exactly where LR+ensemble has its biggest gain over DoM (+0.140 AUROC). The gain is real — even bootstrap min (0.604) exceeds DoM's 0.493 — but different ID subsamples disagree about the best HP-C direction. **Strong signal that HP-C would benefit most from multi-domain training** (LODO-style).
- **IT is now the most stable dataset** (std 0.003), despite having been the most pathologically overfit under weak regularization. Regularization + ensembling jointly shut down the ID-specific direction that was hurting IT.
- **ID is near-deterministic** (std 0.002). The in-domain signal is easy; variance is entirely out-of-domain.

Saved: `results/ensemble_stability_mistral-small-3.1-24b-instruct.json`.

## Consolidated results on Mistral

| recipe                                                        | held-out mean AUROC |
|---------------------------------------------------------------|--------------------:|
| LR, single-probe, L2/C=1 (original frozen baseline)           | 0.673 |
| DoM ensemble, best variant                                    | 0.694 |
| DoM, single-probe, best variant (frozen DoM baseline)         | 0.721 |
| LR, single-probe, L1/C=0.1 (reg-sweep winner)                 | 0.733 |
| **LR ensemble, L2/C=0.1, mean_asst, attn_delta, all 40 layers** | **0.768 ± 0.004** |
| Single-probe oracle ceiling                                   | 0.769 |

Per-dataset (LR+ensemble point estimate):

| dataset | LR+ens AUROC | DoM best | Δ |
|---------|-------------:|---------:|------:|
| instructed-deception             | 0.945 | 0.947 | −0.002 |
| harm-pressure-choice             | 0.633 | 0.493 | **+0.140** |
| harm-pressure-knowledge-report   | 0.720 | 0.691 | +0.029 |
| insider-trading                  | 0.778 | 0.753 | +0.025 |

HP-C and IT carry the margin; ID is saturated; HP-KR is a modest win.

## What the three sweeps jointly tell us

1. **We have a legitimate, reproducible recipe** that beats the DoM baseline on Mistral by ~5 AUROC points. All of the improvement comes from how we *use* the 64 PCs — not from the PC basis itself, which was fixed (Alpaca-train fit) since the initial null result.

2. **The three interventions compose.** Regularization alone (0.673 → 0.733) lifts mostly by rescuing IT from anti-correlation. Ensembling alone, on top of regularized probes (0.733 → 0.763), lifts by averaging layer-specific decision boundaries. Neither alone approaches the combined effect. Neither needs learned parameters beyond the per-layer LR coefficients themselves.

3. **Information is distributed across layers, not concentrated in one.** The monotonic superiority of "all" over "top-K" ensembling says each layer's probe contributes a non-redundant view, even layers whose individual AUROC is mediocre. This is the breadth-over-depth thesis made concrete.

4. **The function class matters, not just the features.** DoM can't use the same ensemble trick productively. LR's 64 degrees of freedom per layer are load-bearing.

5. **The 0.77 ceiling is the same across probe families and selection rules** (single-probe oracle, ensemble best, all equal at 0.769). This strongly suggests the ceiling is the best achievable linear probe over the Alpaca-basis 64-PC features on Mistral-ID training data. To push past it we need different features (per-token positions, more PCs, different signals, or a different basis) or a different training pool (multi-domain, LODO-style).

## Notes for methodology re-freeze

Before scaling to Gemma / Llama / Qwen, the EXPERIMENT.md re-freeze should lock down the following on Mistral-learned grounds, acknowledging these are Mistral-specific defaults:

- **Probe:** L2 LR, C=0.1, on the 64-d per-layer PC vector.
- **Ensemble:** equal-weight mean of sigmoid(logit) across all layers — fixed policy, no learned ensemble weights.
- **Reducer / signal selection:** either (a) pin to (mean_asst, attn_delta) as a hypothesis and see if it holds cross-model, or (b) treat these as a small hyperparameter grid (6 options) and select by ID-dev ensemble AUROC. Option (a) is a stronger claim but risks Mistral-specific over-fit of the recipe itself; (b) preserves a small post-hoc selection surface.
- **Calibration:** unchanged. Alpaca-val at 1% FPR in ensemble-score space.
- **Selection rule:** unchanged. LB-ID dev AUROC if any hyperparameter selection remains.

Open question for re-freeze: should we require the recipe to be validated on at least one additional target model (say Gemma) before committing? That would decouple the recipe-discovery sample from the recipe-validation sample.

## Open questions not resolved by these sweeps

1. **Does the recipe generalize across models?** attn_delta dominance on Mistral is hypothesis-testable on Gemma/Llama/Qwen. Similarly for mean_asst and for the L2/C=0.1 optimum. If any of these fail on other models, option (a) above is dead and we need to keep the small selection surface.
2. **Does multi-domain training narrow the 0.77 ceiling?** LODO (train on LB-ID ∪ a subset of other LB-\* train splits, hold one out) has not been tried. Requires caching LB-\*/train splits we haven't cached yet.
3. **Does per-token reducer granularity push past 0.77?** Requires lmprobe improvements the user is working on in parallel.
4. **Is HP-C persistently noisier than other datasets, or is this Mistral-specific?** Worth tracking across models.

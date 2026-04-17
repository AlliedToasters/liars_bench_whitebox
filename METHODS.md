# Methods Reference

Core pipeline for whitebox deception detection using lmprobe's SampleScan.

## Activation Recovery via Deterministic Replay

We extract activations by replaying the model's own transcripts from the Liar's Bench dataset through a forward pass. The activations we extract are identical to those produced during the original sampling run, because they depend only on the input token sequence and the (deterministic) forward pass.

**Why this holds:** Transformer inference is deterministic given input tokens. At position `t`, the model sees tokens `0..t` and produces activations that depend only on that prefix (causal attention). Whether token `t+1` was sampled live or is pre-filled from a recorded transcript does not affect the activations at position `t`.

**Empirical validation:** We verified correct template reconstruction and tokenization by computing per-token log-likelihood of the assistant's responses in the ground-truth transcripts (`on_policy_affirmation.py`):

| Model | Mean PPL | Median PPL | Mean LL | n |
|---|---|---|---|---|
| Mistral-Small-3.1-24B | 1.35 | 1.29 | -0.287 | 50 |
| Gemma-3-27B-it | 1.44 | 1.35 | -0.338 | 50 |

Perplexity near 1.0 confirms the model assigns high probability to the tokens it actually generated. If template reconstruction, tokenization, or chunked execution were incorrect, perplexity would be significantly higher. (n=50 is a validation check, not a statistical claim; expanding to all samples is straightforward.)

**Terminology note:** This is sometimes called "on-policy" by analogy to RL, though strictly the term applies only to the primary evaluation where the token sequence reflects the model's original sampling policy. In the counterfactual system-prompt ablation (EXPERIMENT.md Section 5.3), the tokens reflect the original policy but the system-prompt context does not — this would not be considered on-policy in the strict RL sense, and we note this when reporting those results.

## Overview

Three operations, fully decoupled:

1. **Fit basis** (unsupervised, Alpaca): Run a forward pass over Alpaca-train transcripts, fit PCA on assistant-token activation deltas at every `(layer, signal)`. Alpaca is general instruction-following data disjoint from LB. The basis captures content-general residual variance — it never sees deception labels or LB prompts.
2. **Project** (Alpaca basis → LB splits + Alpaca-val): Run a fused forward pass projecting every evaluation transcript through the frozen basis, producing per-sample PC vectors at each `(layer, signal, reducer)`.
3. **Fit + calibrate probe** (supervised, LB-train; threshold, Alpaca-val): Fit a linear probe over the 64-PC vectors on LB-train; select `(layer, signal, reducer)` on LB-dev; calibrate the 1% FPR threshold on Alpaca-val projections scored by the frozen probe.

Basis, probe coefficients, the selected `(layer, signal, reducer)` tuple, and the threshold are all frozen before any held-out evaluation. This is required for the zero-shot transfer claim to be valid.

**Why an Alpaca basis:** Fitting PCA on LB-ID training data concentrates the leading components on ID-specific structure (system-prompt contrast, instructed-deception task semantics). Fitting on Alpaca produces a general-purpose basis that's contrastive-neutral: a probe trained against it must find deception structure *inside* a content-general subspace, not inherit it from the basis. Cross-dataset transfer is then a probe-capacity claim, not a basis-coverage claim.

**Terminology note:** The PCA basis is unsupervised and task-agnostic. The probe is supervised (linear classifier on PC vectors, fit on LB labels). `(layer, signal, reducer)` selection is label-informed on LB-dev. We call this a **general-purpose basis + supervised linear probe**.

**Critical distinction from development results:** Our development-phase results (reported in `results/` and on GitHub Pages) used oracle PC and layer selection on the test data. These numbers are inflated and are reported as development results only. The pre-registered evaluation (EXPERIMENT.md) freezes all probe choices on LB-train/dev before held-out evaluation.

## Step 1: Fit a General-Purpose Basis on Alpaca

### Inputs
- `transcripts: list[str]` — chat-templated Alpaca-train transcripts (1000/model, seed=2026)
- `generative_masks: list[np.ndarray]` — boolean mask per transcript, True for assistant-turn tokens

No labels are required at this step. Alpaca is all non-deceptive; the basis is fit unsupervised.

### Why an Alpaca basis (and generative masks)
Fitting PCA on LB-ID activations concentrates the leading components on the instructed-deception contrast (system prompts, task semantics). A probe built on that basis inherits its training scenario's structure and transfers poorly. Fitting on Alpaca produces a basis that captures general assistant-response variance without LB-specific signal. Generative masks ensure the basis is computed from the model's own output activations, not Alpaca's user prompts.

### Code
```python
from lmprobe import SampleScan
import torch

scan = SampleScan.run(
    prompts=alpaca_train_transcripts,  # chat-templated Alpaca-train
    labels=[0] * len(prompts),         # unused; stored for metadata only
    model_name="meta-llama/Llama-3.3-70B-Instruct",
    scan_dir="scans/alpaca_{model}",
    signals=["attn_delta", "mlp_delta"],
    n_components=64,
    device="cuda:0",
    dtype=torch.bfloat16,
    generative_masks=alpaca_train_masks,
    backend="disk_offload",  # or "chunked" for models that fit in CPU RAM
)
```

### What gets stored
```
scans/alpaca_{model}/
  metadata.json
  channels/0_global/basis_attn_delta.npy  # [n_layers, hidden_dim, n_components]
  channels/0_global/basis_mlp_delta.npy
  projections/values.npy                  # Alpaca-train projections (not used downstream)
  samples/samples.parquet
```

The `basis_*.npy` files are the only artifacts used downstream. The projections of Alpaca-train itself are incidental.

## Step 2: Project Evaluation Data Through the Frozen Basis

One fused forward pass per model projects all evaluation data — LB splits (train/dev/held-out per dataset) **and** Alpaca-val — through the basis.

### Code
```python
scan = SampleScan("scans/alpaca_{model}")

# All evaluation prompts fused into a single dict.
prompt_groups = {
    ("instructed-deception", "train"):     id_train_transcripts,
    ("instructed-deception", "dev"):       id_dev_transcripts,
    ("instructed-deception", "held_out"):  id_held_transcripts,
    ("harm-pressure-choice", "dev"):       hpc_dev_transcripts,
    ("harm-pressure-choice", "held_out"):  hpc_held_transcripts,
    # ... other LB datasets
    ("alpaca", "val"):                     alpaca_val_transcripts,
}

results = scan.batch_project_reduced(
    prompt_groups,
    reducers={
        "last_token": LastTokenReducer(masks_flat),
        "mean_asst":  MeanReducer(masks_flat),
        "mean_excl5": MeanExclLastNReducer(masks_flat, n=5),
    },
    batch_size=...,
)
# results[(ds, split)][reducer] → ndarray[n_samples, n_layers, n_signals, n_components]
```

Per-sample PC vectors are cached as `.npz` per `(model, dataset, split)`.

## Step 3: Fit and Freeze the Probe

### 3a. Fit on LB-train
For each `(layer, signal, reducer)`, fit a supervised linear probe (logistic regression) over the 64-dim PC vectors on LB-train, producing coefficient vector `w[layer, signal, reducer]` and bias `b[layer, signal, reducer]`.

### 3b. Select on LB-dev
Score LB-dev under every candidate `(layer, signal, reducer)`; pick the tuple with max AUROC. Freeze `(layer*, signal*, reducer*, w*, b*)`.

### 3c. Calibrate threshold on Alpaca-val
Score the Alpaca-val PC vectors with the frozen probe. Set threshold `τ` such that FPR on Alpaca-val ≤ 1% (paper's protocol). Alpaca-val and Alpaca-train are disjoint; calibration never sees basis-fit data.

### 3d. Evaluate on LB held-out
Score held-out with `(w*, b*)` at `(layer*, signal*, reducer*)`; report AUROC (threshold-free), BalAcc and Recall at `τ`. No further tuning.

```python
# Pseudocode
w, b = fit_logreg(lb_train_vecs[:, l, s, r, :], lb_train_labels)
auroc_dev = roc_auc_score(lb_dev_labels, lb_dev_vecs @ w + b)  # per tuple
# ... pick best (l*, s*, r*)
tau = quantile(alpaca_val_vecs @ w* + b*, 0.99)  # 1% FPR
scores_held = lb_held_vecs @ w* + b*
```

## Experimental Design

### Three-way data split
- **Alpaca-train** (1000/model, seed=2026): basis fit. Never scored.
- **Alpaca-val** (1000/model, seed=2026): threshold calibration only. Disjoint from Alpaca-train.
- **LB-train** (60% of ID, seed=2026): probe coefficient fit.
- **LB-dev** (20% of ID, seed=2026): `(layer, signal, reducer)` selection.
- **LB-held-out** (20% of ID, 50% of HP-C/HP-KR/IT, 100% of GS/CG): held-out evaluation.

### Within-dataset evaluation
- Fit probe on ID-train, select on ID-dev, calibrate on Alpaca-val, evaluate on ID-held-out.

### Cross-dataset transfer (zero-shot)
- Same probe (fit on ID-train, selected on ID-dev), evaluate on HP-C / HP-KR / IT / GS / CG held-out. Same calibrated threshold.

### Cross-fit variants (diagnostic)
- LOO: fit probe on LB \ D_i, evaluate on D_i held-out. Shares basis + calibration.
- Any probe-side change (different train set, different selection set) reuses the cached projections — no GPU work.

## Key Design Decisions

1. **General-purpose Alpaca basis.** PCA is fit on Alpaca (non-deceptive, content-general) so the basis captures instruction-following variance without baking in LB structure. The deception-discriminating directions are learned by the supervised probe on top of PC vectors, not by the basis.

2. **Linear probe over 64 PCs.** We fit a logistic regression over the 64-dim PC vector at each `(layer, signal, reducer)`, rather than picking the single best PC. This uses the full subspace that PCA preserves and allows the probe to span directions not captured by any single component.

3. **Decoupled basis / probe / calibration.** Each stage is fit on disjoint data (Alpaca-train / LB-train / Alpaca-val). No leakage across stages.

4. **Activation deltas, not raw activations.** We capture `attn_delta` (residual change from attention) and `mlp_delta` (residual change from MLP) rather than the full hidden state. These isolate what each sublayer *contributes*, not what it receives.

5. **Generative-only PCA fitting on Alpaca.** The basis is fit on Alpaca assistant tokens, not user prompts. This matches the evaluation-time aggregation scope (assistant-only) and keeps the basis focused on response-generation variance.

## Computational Approach

A practical barrier to whitebox analysis of frontier-scale models is that they don't fit in GPU memory. We overcome this using lmprobe's chunked execution backends, enabling full activation extraction on a single 32GB GPU.

### Chunked forward pass

For models that fit in CPU RAM (up to ~48B parameters on a 128GB machine):

- The full model is loaded on CPU.
- Decoder layers are streamed through the GPU in chunks (e.g. 18 layers at a time).
- Between chunks, PCA is fit on the captured activation deltas while the GPU is idle.
- Hidden states are passed between chunks via CPU memory.

This means GPU memory scales with **chunk size**, not model size. A 24B model runs in the same GPU memory as a 7B model — just in more chunks.

### Disk offload forward pass

For models exceeding CPU RAM (70B+):

- Layer weights are loaded directly from safetensors files to GPU **one layer at a time**.
- After each layer, activation deltas are captured and the weights are discarded.
- Peak memory is a single layer's weights + hidden states + captures.

This enables running 70B models on a machine with 128GB RAM and a single 32GB GPU — the full model is never materialized in memory.

### Batch projection

Evaluating a pre-fit basis on new samples uses `batch_project`:

- All test samples are processed in a single forward pass (per layer chunk/layer), projecting activations onto the stored PCA basis as they're computed.
- This replaces the naive approach of N separate forward passes (one per sample).
- For 100 test samples on a 70B model: ~3 minutes vs ~5 hours.

### Between-chunk PCA

PCA is fit incrementally between layer chunks rather than after the full forward pass. This means:

- Activation captures for a chunk are projected and discarded before the next chunk loads.
- Memory for raw activations scales with **chunk size x samples x tokens**, not **all layers x samples x tokens**.
- The basis for each layer is available immediately after its chunk completes.

### Hardware requirements

Times are from actual runs on an NVIDIA RTX 5090 (32 GB VRAM), including PCA fitting and fused projection.

| Stage | 24–27B (chunked) | 70–72B (disk_offload) |
|---|---|---|
| Basis fit on Alpaca-train (1000 samples) | ~8 min | ~20 min |
| Fused projection (all LB splits + Alpaca-val, ~8k samples) | ~20 min | ~60 min |
| Probe fit + calibration + eval (CPU, from cached PC vectors) | <1 min | <1 min |
| **Total per model** | ~30 min | ~80 min |

Downstream probe analysis (LOO, cross-fit, ablations) operates entirely on the cached PC vectors — no GPU work.

## Classification Under Liar's Bench Taxonomy

Following the taxonomy in Appendix C of the Liar's Bench paper (Kretschmar et al., 2025):

| Property | Classification | Notes |
|---|---|---|
| **Model access** | White-box | Reads activation deltas at every layer |
| **Message scope** | Full (with decontamination toward assistant scope) | The forward pass processes the full sequence. Assistant-turn activations carry information from earlier tokens via attention. Our three decontamination steps (below) are designed to reduce system-prompt influence, but do not guarantee elimination. See Experiment 3 in EXPERIMENT.md for empirical validation. |
| **Interactivity** | Static | Single forward pass on the existing transcript. No additional tokens generated. |
| **Semantic verification** | Incapable | The probe operates on activation geometry, not factual verification. It does not check whether claims are true. |

### Prompt leakage and decontamination

Early development — when the PCA basis was fit on LB-ID — exhibited obvious prompt leakage: different system prompts for deceptive ("Try to deceive them") and truthful ("Answer truthfully") conditions drove the leading PCs, producing AUROC ≈ 1.0 at many layers via surface features rather than deception structure.

The revised methodology addresses this at the **basis** level by fitting PCA on Alpaca rather than LB:

1. **Alpaca basis**: PCA is fit on Alpaca transcripts, which share no prompts with LB and contain no deceptive/truthful contrast. The basis therefore cannot encode LB-specific instruction text as leading variance. Any signal the probe finds on LB data lives in directions the Alpaca basis preserves for content-general reasons.
2. **Generative-only PCA on Alpaca**: the basis is computed from Alpaca assistant tokens only, matching the evaluation-time aggregation scope.
3. **Decoupled probe fit**: the probe coefficients are fit on LB-train PC vectors, not raw activations. This means the probe only has access to subspace information preserved by the (LB-agnostic) Alpaca basis.

Residual concern: assistant-token activations under LB prompts still integrate the system prompt via attention, so the probe can, in principle, latch onto basis directions that happen to correlate with system-prompt content once LB is projected. The counterfactual system-prompt ablation (EXPERIMENT.md §5.3) tests this residual empirically.

## Comparison with Liar's Bench Paper

Both our method and the paper's probes are whitebox (both use internal activations). The differences are methodological:

**Paper's mean probe:** Logistic regression on activations, aggregated by mean over assistant response tokens (last 5 tokens excluded), at a fixed 20th-percentile layer. Trained on instructed-deception pairs. Evaluated across all LB datasets. Aggregate BalAcc ~0.54.

**Paper's upper-bound probe:** Same architecture, but trained with privileged access to 5 LB datasets (HP-C, HP-KR, CG, ID, IT) plus Alpaca control. Aggregate BalAcc 0.73, AUROC 0.91.

**Our method:** General-purpose PCA basis fit on Alpaca-train (unsupervised), linear probe fit on LB-train over 64-dim PC vectors, `(layer, signal, reducer)` selected on LB-dev, threshold calibrated on Alpaca-val at 1% FPR. Primary training signal is LB-ID; cross-dataset evaluation is zero-shot.

Key protocol differences that must be ablated (see EXPERIMENT.md Section 3):
- Layer: we search over all layers on LB-dev; they fix at 20th percentile
- Token aggregation: we select reducer on LB-dev (last_token / mean_asst / mean_excl5); they use mean_excl5
- Basis: we fit PCA on Alpaca (LB-disjoint) then a linear probe on PC vectors; they fit logistic regression directly on activations
- Training data: LB-ID for probe coefficients; their upper bound uses five LB datasets directly

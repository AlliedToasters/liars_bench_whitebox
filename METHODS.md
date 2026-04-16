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

Two operations:

1. **Fit**: Given labeled chat transcripts `(X, Y)`, run a forward pass capturing activation deltas at every `(token, layer)`, fit PCA on the generative tokens to get a reusable basis. Select the best PC direction and layer using training-set labels.
2. **Evaluate**: Given a frozen probe (basis + PC index + layer + direction), project new transcripts through the basis and compute classification metrics at the assistant's last token. **No test labels are used during evaluation.**

The basis, PC index, layer, and direction sign are all frozen from the training set before any test evaluation. This is required for the zero-shot transfer claim to be valid.

**Terminology note:** PCA fitting is unsupervised (labels are not used). However, two subsequent steps use labeled data: (a) selecting the best PC direction (max AUROC over PCs), and (b) selecting the best layer. The overall method is therefore **unsupervised basis + supervised direction and layer selection** — a label-informed probe. We use "PCA-based probe" rather than "unsupervised probe" to avoid overclaiming.

**Critical distinction from development results:** Our development-phase results (reported in `results/` and on GitHub Pages) used oracle PC and layer selection on the test data — selecting the best PC and best layer per evaluation cell using test labels. These numbers are inflated and are reported as development results only. The pre-registered evaluation (EXPERIMENT.md) freezes PC and layer from the ID training/dev set before any held-out evaluation.

## Step 1: Fit a Basis and Freeze the Probe

### Inputs
- `transcripts: list[str]` — chat-templated sequences (system + user + assistant turns)
- `labels: list[int]` — 1 = deceptive, 0 = truthful
- `generative_masks: list[np.ndarray]` — boolean mask per transcript, True for assistant-turn tokens

### Why generative masks matter
Without masking, PCA captures the dominant variance in the full sequence — which includes the system prompt. If deceptive and truthful conditions use different system prompts (as in instructed-deception), the basis trivially encodes the instruction text, not a deception representation. Generative-only PCA forces the basis to capture variance in the model's own output activations.

### Code
```python
from lmprobe import SampleScan
import torch

scan = SampleScan.run(
    prompts=train_transcripts,    # list of chat-templated strings
    labels=train_labels,          # list of 0/1
    model_name="meta-llama/Llama-3.3-70B-Instruct",
    scan_dir="scans/my_basis",    # where to store the basis
    signals=["attn_delta", "mlp_delta"],
    n_components=64,              # PCA components per layer per signal
    device="cuda:0",
    dtype=torch.bfloat16,
    generative_masks=train_masks, # only fit PCA on assistant tokens
    backend="disk_offload",       # or "chunked" for models that fit in CPU RAM
)
```

### Freezing the probe
After fitting the basis, the probe is frozen by selecting the best layer, PC index, and direction sign on the **training or dev set** (never on test data):

```python
# Project training set through basis, extract assistant-last-token vectors
# Select best (layer, signal, pc, direction) by max AUROC on training labels
# Freeze these choices before any test evaluation
frozen_layer = best_layer_from_train
frozen_signal = best_signal_from_train
frozen_pc = best_pc_from_train
frozen_flip = direction_from_train
```

### What gets stored
```
scans/my_basis/
  metadata.json                           # model info, n_layers, n_components
  channels/0_global/basis_attn_delta.npy  # [n_layers, hidden_dim, n_components]
  channels/0_global/basis_mlp_delta.npy   # [n_layers, hidden_dim, n_components]
  projections/values.npy                  # training set projections (sparse)
  projections/coords.parquet              # (sample_id, layer, token_pos, signal)
  samples/samples.parquet                 # transcript text, labels, seq lengths
```

The basis files (`basis_*.npy`) plus the frozen (layer, signal, pc, flip) constitute the complete probe. Everything else is for analysis of the training set.

## Step 2: Evaluate on a Test Set

### Inputs
- A frozen probe: `SampleScan` object + (layer, signal, pc, flip)
- `test_transcripts: list[str]` — new chat-templated sequences
- `test_labels: list[int]` — ground truth (used ONLY for computing metrics, not for any selection)
- `asst_last_positions: list[int]` — token index of the last assistant token per transcript

### Projection
```python
scan = SampleScan("scans/my_basis")
projections, coords, token_ids, seq_lengths = scan.batch_project(
    test_transcripts, batch_size=4
)
```

`batch_project` runs a single forward pass per layer chunk for ALL test samples, projecting activations onto the stored PCA basis. Returns sparse projections at every `(sample, layer, token, signal)`.

### Scoring with the frozen probe
```python
# Extract scores at the frozen (layer, signal, pc) for each sample's last assistant token
for i in range(n_test):
    score = test_vecs[i, frozen_layer, frozen_signal, frozen_pc]
    if frozen_flip:
        score = -score
    scores[i] = score

# Compute metrics — labels used here for evaluation only
auroc = roc_auc_score(test_labels, scores)
fpr, tpr, thresholds = roc_curve(test_labels, scores)
# ... balanced accuracy at 1% FPR, etc.
```

## Experimental Design

### Within-dataset holdout
- **Train** (60%): fit PCA basis + select layer/PC/direction
- **Dev** (20%): validate probe choices, tune if needed
- **Held-out** (20%): single frozen-probe evaluation, headline results
- Same dataset, disjoint samples, seed=2026

### Cross-dataset transfer
- **Probe**: frozen from ID train/dev (basis + layer + PC + direction)
- **Test**: project other dataset samples through frozen probe, score with frozen direction
- No retraining, no selection on test data. Tests whether the deception direction generalizes.

### Cross-fit matrix
- For each available dataset, fit a basis and freeze a probe. Evaluate every frozen probe on every dataset.
- Produces an N x N transfer matrix per model.
- Diagonal = within-dataset holdout. Off-diagonal = transfer.

## Key Design Decisions

1. **PCA basis with post-hoc direction selection.** The PCA basis captures directions of maximum variance in generative activations without using labels. The deception-discriminating direction is then selected on the training set (the PC with max AUROC). This two-step approach avoids directly fitting a classifier to the training labels, but the direction selection step is label-informed and the full probe must be evaluated on held-out data.

2. **Activation deltas, not raw activations.** We capture `attn_delta` (residual change from attention) and `mlp_delta` (residual change from MLP) rather than the full hidden state. These isolate what each sublayer *contributes*, not what it receives.

3. **Assistant-last-token evaluation.** The model has processed its full response at this position. Earlier tokens may not yet reflect the deception strategy.

4. **Generative-only PCA fitting.** Designed to reduce (but not guarantee elimination of) system prompt influence on the basis. See decontamination discussion below.

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

Times are from actual runs on an NVIDIA RTX 5090 (32 GB VRAM), including PCA fitting.

| Model size | Backend | GPU VRAM | CPU RAM | Time (500 samples, fit) | Time (100 samples, eval) |
|---|---|---|---|---|---|
| 24-27B | Chunked | 32 GB | 64 GB | ~5 min | ~1 min |
| 70-72B | Disk offload | 32 GB | 128 GB | ~15 min | ~3 min |

## Classification Under Liar's Bench Taxonomy

Following the taxonomy in Appendix C of the Liar's Bench paper (Kretschmar et al., 2025):

| Property | Classification | Notes |
|---|---|---|
| **Model access** | White-box | Reads activation deltas at every layer |
| **Message scope** | Full (with decontamination toward assistant scope) | The forward pass processes the full sequence. Assistant-turn activations carry information from earlier tokens via attention. Our three decontamination steps (below) are designed to reduce system-prompt influence, but do not guarantee elimination. See Experiment 3 in EXPERIMENT.md for empirical validation. |
| **Interactivity** | Static | Single forward pass on the existing transcript. No additional tokens generated. |
| **Semantic verification** | Incapable | The probe operates on activation geometry, not factual verification. It does not check whether claims are true. |

### Prompt leakage and decontamination

The instructed-deception dataset uses different system prompts for deceptive ("Try to deceive them") and truthful ("Answer truthfully") conditions. Initial analysis on the full token sequence yielded perfect AUROC (1.000) at multiple layers — this was **prompt leakage**, not deception detection. The model's activations trivially encode the instruction text.

Three decontamination steps are designed to reduce this:

1. **Generative-only PCA**: The basis is fit only on assistant-turn tokens, excluding system and user prompts from the variance computation. This controls what the basis captures as dominant variance, but does not prevent the basis from also capturing directions that correlate with system-prompt content in assistant-token activations via attention.
2. **Assistant-last-token evaluation**: Separability is computed at the last token of the assistant turn, not the last token of the full sequence. This reads from a later position, but attention has already integrated system prompt information.
3. **Train/test holdout**: The basis is fit on training samples and evaluated on held-out samples. This controls for overfitting, not for leakage.

These steps are plausible mitigations, not proof of assistant-scope signal. We test the residual leakage empirically via a counterfactual system-prompt ablation (EXPERIMENT.md Section 5.3): if probe scores on the same response tokens are invariant to system-prompt content, the signal is predominantly assistant-scope. Results will determine whether the assistant-scope claim is supported.

## Comparison with Liar's Bench Paper

Both our method and the paper's probes are whitebox (both use internal activations). The differences are methodological:

**Paper's mean probe:** Logistic regression on activations, aggregated by mean over assistant response tokens (last 5 tokens excluded), at a fixed 20th-percentile layer. Trained on instructed-deception pairs. Evaluated across all LB datasets. Aggregate BalAcc ~0.54.

**Paper's upper-bound probe:** Same architecture, but trained with privileged access to 5 LB datasets (HP-C, HP-KR, CG, ID, IT) plus Alpaca control. Aggregate BalAcc 0.73, AUROC 0.91.

**Our method:** Contrastive PCA basis with post-hoc best-PC selection, fit on generative-only tokens from a single LB dataset (instructed-deception), evaluated at assistant's last token. Layer and PC frozen from training set.

Key protocol differences that must be ablated (see EXPERIMENT.md Section 3):
- Layer: we search over all layers (on training data); they fix at 20th percentile
- Token aggregation: we use last token; they use mean over response (excluding last 5)
- Training data: we use one LB dataset; their upper bound uses five
- Basis: we use PCA; they use logistic regression

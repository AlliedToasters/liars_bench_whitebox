# Methods Reference

Core pipeline for whitebox deception detection using lmprobe's SampleScan.

## On-Policy Activations

We extract activations by replaying the model's own transcripts from the Liar's Bench dataset through a forward pass. These activations are **on-policy**: they are identical to the activations the model produced during original generation.

**Why this holds:** Transformer inference is deterministic given input tokens. At position `t`, the model sees tokens `0..t` and produces activations that depend only on that prefix (causal attention). Whether token `t+1` was sampled live or is pre-filled from a recorded transcript does not affect the activations at position `t`. Our forward pass replays the exact token sequence, so activations at each position are identical to what they were during generation.

**Empirical validation:** We verified this by computing per-token log-likelihood of the assistant's responses in the ground-truth transcripts (`on_policy_affirmation.py`):

| Model | Mean PPL | Median PPL | Mean LL | n |
|---|---|---|---|---|
| Mistral-Small-3.1-24B | 1.35 | 1.29 | -0.287 | 50 |
| Gemma-3-27B-it | 1.44 | 1.35 | -0.338 | 50 |

Perplexity near 1.0 confirms the model assigns high probability to the tokens it actually generated, validating correct template reconstruction, tokenization, and chunked execution. If any of these were wrong (e.g. incorrect chat template, model version mismatch), perplexity would be significantly higher.

**What "on-policy" means here:** The Liar's Bench transcripts were sampled from these models under specific prompting conditions. We measure activations on the trajectories the model actually took — not on counterfactual or adversarial inputs. This is the standard meaning of "on-policy" in the RL sense: the data distribution matches the policy being analyzed.

## Overview

Two operations:

1. **Fit**: Given labeled prompts `(X, Y)`, run a forward pass capturing activation deltas at every `(token, layer)`, fit PCA on the generative tokens to get a reusable basis.
2. **Evaluate**: Given a fitted basis and new labeled prompts `(X', Y')`, project through the basis and compute classification metrics at the assistant's last token.

The basis is fit **once** on a training corpus and applied to any number of test prompts with no retraining. This is the core claim: a PCA basis fit on one deception scenario transfers zero-shot to other deception scenarios.

**Terminology note:** PCA fitting is unsupervised (labels are not used). However, the subsequent step of selecting the best PC direction uses labeled data (max AUROC over PCs). The overall method is therefore **unsupervised basis + supervised direction selection** — functionally a label-informed probe, not a purely unsupervised detector. We use "PCA-based probe" rather than "unsupervised probe" to avoid overclaiming.

## Step 1: Fit a Basis

### Inputs
- `prompts: list[str]` — tokenized chat transcripts (system + user + assistant turns)
- `labels: list[int]` — 1 = deceptive, 0 = truthful
- `generative_masks: list[np.ndarray]` — boolean mask per prompt, True for assistant-turn tokens

### Why generative masks matter
Without masking, PCA captures the dominant variance in the full sequence — which includes the system prompt. If deceptive and truthful conditions use different system prompts (as in instructed-deception), the basis trivially encodes the instruction text, not a deception representation. Generative-only PCA forces the basis to capture variance in the model's own output activations.

### Code
```python
from lmprobe import SampleScan
import torch

scan = SampleScan.run(
    prompts=train_prompts,        # list of chat-templated strings
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

### What gets stored
```
scans/my_basis/
  metadata.json                           # model info, n_layers, n_components
  channels/0_global/basis_attn_delta.npy  # [n_layers, hidden_dim, n_components]
  channels/0_global/basis_mlp_delta.npy   # [n_layers, hidden_dim, n_components]
  projections/values.npy                  # training set projections (sparse)
  projections/coords.parquet              # (sample_id, layer, token_pos, signal)
  samples/samples.parquet                 # prompt text, labels, seq lengths
```

The basis files (`basis_*.npy`) are the reusable artifact. Everything else is for analysis of the training set.

## Step 2: Evaluate on a Test Set

### Inputs
- A fitted `SampleScan` object (or loaded from disk)
- `test_prompts: list[str]` — new chat-templated strings
- `test_labels: list[int]` — ground truth
- `asst_last_positions: list[int]` — token index of the last assistant token per prompt

### Projection
```python
scan = SampleScan("scans/my_basis")
projections, coords, token_ids, seq_lengths = scan.batch_project(
    test_prompts, batch_size=4
)
```

`batch_project` runs a single forward pass per layer chunk for ALL test samples, projecting activations onto the stored PCA basis. Returns sparse projections at every `(sample, layer, token, signal)`.

### Extract evaluation vectors
We evaluate at the **assistant's last token** — the position where the model has processed its full response and the deception signal is strongest.

```python
# Extract [n_samples, n_layers, n_signals, n_components] array
test_vecs = np.zeros((n_test, n_layers, n_signals, k), dtype=np.float32)

coord_sample = np.array(coords["sample_id"])
coord_layer = np.array(coords["layer"])
coord_tok = np.array(coords["token_pos"])
coord_sig = np.array(coords["signal"])

for i in range(n_test):
    tok_pos = min(asst_last[i], seq_lengths[i] - 1)
    for si, sig in enumerate(scan.signals):
        mask = (coord_sample == i) & (coord_tok == tok_pos) & (coord_sig == sig)
        for row_idx in np.where(mask)[0]:
            test_vecs[i, coord_layer[row_idx], si, :] = projections[row_idx, 0, :]
```

### Compute metrics
For each `(layer, signal)` pair, find the best PC direction (max AUROC), then compute:

```python
from sklearn.metrics import roc_auc_score, roc_curve, balanced_accuracy_score

# Best PC selection
auroc = roc_auc_score(labels, pc_vals)
auroc = max(auroc, 1 - auroc)  # orient direction

# Balanced accuracy at 1% FPR (paper's evaluation protocol)
fpr, tpr, thresholds = roc_curve(labels, oriented_scores)
idx = np.searchsorted(fpr, 0.01, side='right') - 1
preds = (oriented_scores >= thresholds[idx]).astype(int)
bal_acc_1pct = balanced_accuracy_score(labels, preds)
```

## Experimental Design

### Within-dataset holdout
- **Train**: 400 samples (200 dec + 200 truth) — fit PCA basis
- **Test**: 100 samples (50 + 50) — project through basis, compute metrics
- Same dataset, disjoint samples, fixed seed=42

### Cross-dataset transfer
- **Basis**: fit on dataset A (e.g. instructed-deception)
- **Test**: project dataset B samples (e.g. insider-trading) through A's basis
- No retraining. Tests whether the deception direction generalizes.

### Cross-fit matrix
- For each available dataset, fit a basis. Evaluate every basis on every dataset.
- Produces an N x N transfer matrix per model.
- Diagonal = within-dataset holdout. Off-diagonal = transfer.

## Key Design Decisions

1. **PCA basis with post-hoc direction selection.** The PCA basis captures directions of maximum variance in generative activations without using labels. The deception-discriminating direction is then selected post-hoc (the PC with max AUROC). This two-step approach avoids directly fitting a classifier to the training labels, but the direction selection step is label-informed and must be evaluated on held-out data.

2. **Activation deltas, not raw activations.** We capture `attn_delta` (residual change from attention) and `mlp_delta` (residual change from MLP) rather than the full hidden state. These isolate what each sublayer *contributes*, not what it receives.

3. **Assistant-last-token evaluation.** The model has processed its full response at this position. Earlier tokens may not yet reflect the deception strategy.

4. **Generative-only PCA fitting.** Prevents the basis from encoding system prompt differences between conditions.

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

| Model size | Backend | GPU VRAM | CPU RAM | Time (500 samples) |
|---|---|---|---|---|
| 24-27B | Chunked | 32 GB | 64 GB | ~5 min |
| 70-72B | Disk offload | 32 GB | 128 GB | ~15 min |

## Classification Under Liar's Bench Taxonomy

Following the taxonomy in Appendix C of the Liar's Bench paper (Kretschmar et al., 2025):

| Property | Classification | Notes |
|---|---|---|
| **Model access** | White-box | Reads activation deltas at every layer |
| **Message scope** | Full (evaluated at assistant scope) | The forward pass processes the full sequence (system + user + assistant) — this is inherent to how transformers work. However, the signal we measure is decontaminated to assistant scope: PCA is fit only on generative (assistant-turn) tokens, and evaluation uses only the assistant's last token. See "Prompt leakage" below. |
| **Interactivity** | Static | Single forward pass on the existing transcript. No additional tokens generated. |
| **Semantic verification** | Incapable | The probe operates on activation geometry, not factual verification. It does not check whether claims are true. |

### Prompt leakage and decontamination

The instructed-deception dataset uses different system prompts for deceptive ("Try to deceive them") and truthful ("Answer truthfully") conditions. Initial analysis on the full token sequence yielded perfect AUROC (1.000) at multiple layers — this was **prompt leakage**, not deception detection. The model's activations trivially encode the instruction text.

Three decontamination steps control for this:

1. **Generative-only PCA**: The basis is fit only on assistant-turn tokens, excluding system and user prompts from the variance computation.
2. **Assistant-last-token evaluation**: Separability is computed at the last token of the assistant turn, not the last token of the full sequence.
3. **Train/test holdout**: The basis is fit on 400 training samples and evaluated on 100 held-out test samples.

This reduces the effective message scope to **assistant-only** for the signal we report, even though the forward pass necessarily sees the full context. The decontamination steps are themselves a methodological contribution — they demonstrate awareness of the confound and provide a reproducible protocol for controlling it.

## Comparison with Liar's Bench Paper

The paper evaluates linear probes at **1% FPR** and reports **balanced accuracy** and **AUROC**.

Their results:
- Mean probe: ~59% balanced accuracy on ID, near-chance on cross-dataset
- Upper-bound probe (trained with privileged info on test set): 0.73 balanced accuracy, 0.91 AUROC

Our approach differs in that:
- We use internal activations (whitebox), not behavioral features
- Our PCA basis is fit on one dataset and transfers zero-shot to others
- We evaluate at the same 1% FPR threshold for direct comparison
